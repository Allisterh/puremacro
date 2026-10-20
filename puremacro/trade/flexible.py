"""Flexible CGE extensions of the trade model, on the consistent accounting.

The core model (:mod:`puremacro.trade._accounting`) has Cobb-Douglas value
added, Leontief gross output and fixed-coefficient final-demand sourcing. This
module generalises those blocks while keeping the same accounting (producer
prices, output-revenue taxes, tax-inclusive final expenditure, schedule duties
rebated lump sum, fixed foreign saving and a first-producer-price numeraire):

- nested CES technology in calibrated share form: a value-added nest
  (``rho_va``), a gross-output nest between value added and the intermediate
  composite (``sigma_y``) and CES sourcing of intermediates across origins
  (``sigma_inter``);
- an optional smooth capacity barrier that raises the factor requirement per
  unit of value added in one or more country-sector cells;
- Stone-Geary (LES) household demand with income-dependent subsistence;
- fixed-coefficient or Armington CES final-demand sourcing;
- Atkeson-Burstein variable markups, whose profits accrue to the households of
  the producing country.

:func:`solve_flexible_trade_equilibrium` solves the model on a condensed system
in ``[log r; log w; T]``: producer prices follow from the zero-profit
conditions and gross outputs from goods-market clearing, so only the factor
markets and the government budgets (plus the numeraire) remain. At the default
configuration every block coincides with the core model, and
:func:`_flexible_blocks` equals :func:`puremacro.trade._accounting.evaluate`.

Pure NumPy, SciPy and pandas (Pyodide-compatible).
"""
from __future__ import annotations

import copy
import sys
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import MISSING, dataclass, field, fields, replace
from typing import Any

import numpy as np
import pandas as pd
import scipy.linalg as la

from puremacro.trade._results import TradeCalibrationResult
from puremacro.trade.equilibrium import (
    _get_ces_weights,
    check_accounting,
    check_policy_options,
    unpack_equilibrium_vector,
)


def _get_replace_caller() -> Any | None:
    """Inspect the call stack for the instance being copied by ``dataclasses.replace``.

    Python 3.13+ routes ``replace(obj, **changes)`` through ``obj.__replace__`` ->
    ``dataclasses._replace(self, ...)``; Python 3.11 and 3.12 call the class
    directly from ``dataclasses.replace(obj, ...)``. Both shapes are recognised.
    """
    for i in range(1, 6):
        try:
            f = sys._getframe(i)
        except (AttributeError, ValueError):
            break
        if f.f_globals.get("__name__") != "dataclasses":
            continue
        if f.f_code.co_name == "_replace" and "self" in f.f_locals:
            return f.f_locals["self"]
        if f.f_code.co_name == "replace" and "obj" in f.f_locals:
            return f.f_locals["obj"]
    return None


def _get_replace_explicit_keys() -> set[str] | None:
    """Explicit keys passed to ``dataclasses.replace`` (Python 3.13+), else None."""
    for i in range(1, 6):
        try:
            f = sys._getframe(i)
            if f.f_code.co_name == "_replace" and f.f_globals.get("__name__") == "dataclasses":
                # Only 3.13+ keeps the caller's explicit keys apart; 3.11/3.12 fill
                # ``changes`` with every field, so callers fall back to value comparison.
                f_parent = sys._getframe(i + 1)
                if f_parent.f_code.co_name == "replace" and "changes" in f_parent.f_locals:
                    return set(f_parent.f_locals["changes"].keys())
        except (AttributeError, ValueError):
            break
    return None


def _val_equal(a: Any, b: Any) -> bool:
    """Equality of configuration values (None, arrays, mappings or scalars)."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return bool(np.array_equal(a, b))
    return bool(a == b)


def _has_nonzero_setting(value: Any) -> bool:
    """Whether an optional scalar/array/mapping contains a non-zero setting.

    Values that cannot be read as numbers count as set (``True``), so a
    malformed value is never mistaken for an inactive default.
    """
    if value is None:
        return False
    try:
        if isinstance(value, Mapping):
            vals = [np.asarray(v, dtype=float) for v in value.values()]
            return any(bool(np.any(np.abs(v) > 1e-12)) for v in vals)
        return bool(np.any(np.abs(np.asarray(value, dtype=float)) > 1e-12))
    except (TypeError, ValueError):
        return True


def _differs_from(value: Any, baseline: Any) -> bool:
    """Whether a scalar/array/None setting differs from its (scalar or None) default."""
    if baseline is None:
        return value is not None
    if value is None:
        return True
    try:
        return bool(np.any(np.abs(np.asarray(value, dtype=float) - float(baseline)) > 1e-12))
    except (TypeError, ValueError):
        return True


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


# =============================================================================
# Declarative configuration
# =============================================================================

@dataclass(frozen=True)
class FlexibleTechnologyConfig:
    """Nested CES production technology and the optional capacity barrier.

    Parameters
    ----------
    rho_va : float, default 1.0
        Elasticity of substitution between capital and labour in the
        value-added nest (1.0 is Cobb-Douglas, the core model). Aliased with
        ``sigma_va``. Must be strictly positive.
    sigma_y : float, default 0.0
        Elasticity of substitution between value added and the intermediate
        composite in gross output (0.0 is Leontief, the core model). Must be
        non-negative.
    sigma_inter : float, default 0.0
        Elasticity of substitution of intermediate inputs across origins (0.0
        is fixed coefficients, the core model). Must be non-negative.
    capacity_margins : float, Mapping or np.ndarray, optional
        Spare-capacity margins of a smooth capacity barrier. The penalty
        ``pen(y) = penalty_scale * (y / ((1 + margin) y_0))**penalty_exponent``
        multiplies the factor requirement per unit of value added,
        ``c_va (1 + pen(y))``, in the zero-profit price and in the labour and
        capital demands, so revenue net of production taxes still equals cost.
        The penalty is positive at the benchmark output, so an active margin
        moves the equilibrium away from the calibrated point even without a
        shock. Accepted forms:

        - a number: the margin of the one cell ``(capacity_country,
          capacity_sector)``; both codes are required;
        - a mapping keyed by sector code: margins of those sectors in
          ``capacity_country`` (required);
        - a mapping keyed by ``(country, sector)`` pairs (codes or positions);
        - an array broadcastable to ``(1, ns, nc)``: a margin for every cell.

        A missing ``capacity_country`` or ``capacity_sector`` raises
        ``TypeError`` naming it; nothing defaults to a particular country or
        sector.
    capacity_country, capacity_sector : str, optional
        Country and sector codes the capacity margins refer to (see above).
    penalty_scale : float, default 0.05
        Scale of the capacity barrier.
    penalty_exponent : float, default 2.0
        Exponent of the capacity barrier.
    sigma_va : float, optional
        Alias for ``rho_va``.
    """

    rho_va: float = 1.0
    sigma_y: float = 0.0
    sigma_inter: float = 0.0
    capacity_margins: Any = None
    capacity_country: str | None = None
    capacity_sector: str | None = None
    penalty_scale: float = 0.05
    penalty_exponent: float = 2.0
    sigma_va: float | None = None

    def __post_init__(self) -> None:
        if not _is_number(self.rho_va):
            raise TypeError(f"rho_va must be a numeric float, got {type(self.rho_va).__name__}.")
        if self.sigma_va is not None and not _is_number(self.sigma_va):
            raise TypeError(f"sigma_va must be a numeric float, got {type(self.sigma_va).__name__}.")

        old_inst = _get_replace_caller()
        if old_inst is not None and isinstance(old_inst, FlexibleTechnologyConfig):
            exp_keys = _get_replace_explicit_keys()
            if exp_keys is not None:
                rho_changed = "rho_va" in exp_keys
                sigma_changed = "sigma_va" in exp_keys
            else:
                rho_changed = self.rho_va != old_inst.rho_va
                sigma_changed = self.sigma_va is not None and self.sigma_va != old_inst.sigma_va
            if sigma_changed and not rho_changed:
                target_rho = self.sigma_va
            elif rho_changed and sigma_changed and self.rho_va != self.sigma_va:
                raise ValueError(
                    f"Conflicting values provided for rho_va ({self.rho_va}) and sigma_va ({self.sigma_va})."
                )
            else:
                target_rho = self.rho_va
        else:
            target_rho = self.rho_va
            if self.sigma_va is not None:
                if self.rho_va != 1.0 and self.rho_va != self.sigma_va:
                    raise ValueError(
                        f"Conflicting values provided for rho_va ({self.rho_va}) and sigma_va ({self.sigma_va})."
                    )
                target_rho = self.sigma_va

        for name in ("sigma_y", "sigma_inter", "penalty_scale", "penalty_exponent"):
            if not _is_number(getattr(self, name)):
                raise TypeError(f"{name} must be a numeric float, got {type(getattr(self, name)).__name__}.")

        float_rho = float(target_rho if target_rho is not None else 1.0)
        object.__setattr__(self, "rho_va", float_rho)
        object.__setattr__(self, "sigma_va", float_rho)
        object.__setattr__(self, "sigma_y", float(self.sigma_y))
        object.__setattr__(self, "sigma_inter", float(self.sigma_inter))
        object.__setattr__(self, "penalty_scale", float(self.penalty_scale))
        object.__setattr__(self, "penalty_exponent", float(self.penalty_exponent))

        if not np.isfinite(float_rho) or float_rho <= 0.0:
            raise ValueError(
                f"rho_va (value-added substitution elasticity) must be strictly positive (got {float_rho}). "
                "A non-positive elasticity of substitution violates strict quasi-concavity of the production "
                "function and leaves factor cost minimisation undefined."
            )
        if not np.isfinite(self.sigma_y) or self.sigma_y < 0.0:
            raise ValueError(
                f"sigma_y (gross output substitution elasticity) must be non-negative (got {self.sigma_y}). "
                "A negative elasticity of substitution violates cost minimisation."
            )
        if not np.isfinite(self.sigma_inter) or self.sigma_inter < 0.0:
            raise ValueError(
                f"sigma_inter (intermediate sourcing elasticity) must be non-negative (got {self.sigma_inter})."
            )
        if self.penalty_scale < 0.0 or not np.isfinite(self.penalty_scale):
            raise ValueError(f"penalty_scale must be finite and non-negative (got {self.penalty_scale}).")
        if self.penalty_exponent <= 0.0 or not np.isfinite(self.penalty_exponent):
            raise ValueError(f"penalty_exponent must be finite and positive (got {self.penalty_exponent}).")
        _check_capacity_arguments(self.capacity_margins, self.capacity_country, self.capacity_sector)


def _check_capacity_arguments(margins: Any, country: str | None, sector: str | None) -> None:
    """Raise ``TypeError`` naming the code a capacity specification lacks."""
    if margins is None:
        return
    if _is_number(margins):
        for name, value in (("capacity_country", country), ("capacity_sector", sector)):
            if value is None:
                raise TypeError(
                    "a scalar capacity_margins applies to one country and sector: "
                    f"pass {name}=<code>"
                )
    elif isinstance(margins, Mapping):
        sector_keys = [k for k in margins if not isinstance(k, tuple)]
        if sector_keys and country is None:
            raise TypeError(
                "capacity_margins keyed by sector code apply to one country: pass capacity_country=<code>"
            )
    elif not isinstance(margins, np.ndarray):
        raise TypeError(
            "capacity_margins must be a number, a mapping or a NumPy array, "
            f"got {type(margins).__name__}"
        )


@dataclass(frozen=True)
class FlexiblePreferenceConfig:
    """Non-homothetic household preferences and final-demand sourcing.

    Parameters
    ----------
    subsistence_shares : float, Mapping[str, float] or np.ndarray, optional
        Stone-Geary subsistence shares ``mu_s`` in ``[0, 1)``: the committed
        part of the benchmark household purchases of each sector. Default 0
        (no subsistence: the fixed-proportion household basket of the core
        model). Aliased with ``mu_s`` and ``subsistence_ratio``. Supernumerary
        spending buys the calibrated basket in fixed proportions, so a uniform
        share leaves equilibrium demand unchanged; Engel effects need
        sector-specific shares (a mapping or an array).
    sigma_trade : float or np.ndarray, optional
        Final-demand sourcing across origin countries. ``None`` (default) keeps
        fixed coefficients at the benchmark shares (the core model); a value
        strictly greater than 1.0 switches on Armington CES sourcing at that
        elasticity. Final-demand deliveries are valued at producer prices in
        the trade and duty accounts under either rule, and the purchaser value
        of each country's deliveries equals its spending exactly.
    mu_s, subsistence_ratio : optional
        Aliases for ``subsistence_shares``; differing values raise ``ValueError``.
    """

    subsistence_shares: Any = None
    sigma_trade: float | np.ndarray | None = None
    mu_s: Any = None
    subsistence_ratio: Any = None

    def __post_init__(self) -> None:
        for name in ("sigma_trade", "subsistence_shares", "mu_s", "subsistence_ratio"):
            if isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} cannot be a boolean.")

        old_inst = _get_replace_caller()
        if old_inst is not None and isinstance(old_inst, FlexiblePreferenceConfig):
            sub_aliases = ("subsistence_shares", "mu_s", "subsistence_ratio")
            exp_keys = _get_replace_explicit_keys()
            if exp_keys is not None:
                changed_keys = [k for k in sub_aliases if k in exp_keys]
            else:
                changed_keys = [k for k in sub_aliases if not _val_equal(getattr(self, k), getattr(old_inst, k))]
            if changed_keys:
                vals_list = [getattr(self, k) for k in changed_keys]
                if any(not _val_equal(u, vals_list[0]) for u in vals_list[1:]):
                    raise ValueError("Conflicting values provided for subsistence parameters.")
                raw_mu = vals_list[0]
            else:
                raw_mu = self.subsistence_shares if self.subsistence_shares is not None else 0.0
        else:
            candidates = [
                ("subsistence_shares", self.subsistence_shares),
                ("mu_s", self.mu_s),
                ("subsistence_ratio", self.subsistence_ratio),
            ]
            non_none = [(name, val) for name, val in candidates if val is not None]
            if len(non_none) > 1:
                first_name, first_val = non_none[0]
                for other_name, other_val in non_none[1:]:
                    if not _val_equal(first_val, other_val):
                        raise ValueError(
                            f"Conflicting values provided for subsistence parameters: "
                            f"'{first_name}' and '{other_name}'."
                        )
                raw_mu = first_val
            elif len(non_none) == 1:
                raw_mu = non_none[0][1]
            else:
                raw_mu = 0.0

        if self.sigma_trade is None:
            pass
        elif _is_number(self.sigma_trade):
            sig = float(self.sigma_trade)
            if not np.isfinite(sig) or sig <= 1.0:
                raise ValueError(
                    f"sigma_trade (Armington substitution elasticity) must be strictly greater than 1.0 (got {sig}). "
                    "Varieties from different origins must be gross substitutes; pass None for "
                    "fixed-coefficient sourcing."
                )
            object.__setattr__(self, "sigma_trade", sig)
        elif isinstance(self.sigma_trade, np.ndarray):
            if not np.all(np.isfinite(self.sigma_trade)) or np.any(self.sigma_trade <= 1.0):
                raise ValueError(
                    "All elements of sigma_trade must be finite and strictly greater than 1.0 "
                    f"(got min={np.min(self.sigma_trade)})."
                )
        else:
            raise TypeError(f"sigma_trade must be None, numeric or ndarray, got {type(self.sigma_trade).__name__}.")

        if _is_number(raw_mu):
            mu_val = float(raw_mu)
            if mu_val < 0.0:
                raise ValueError(
                    f"mu_s / subsistence_shares must be non-negative (got {mu_val}). "
                    "A negative subsistence share would make the good an economic bad."
                )
            if mu_val >= 1.0:
                raise ValueError(
                    f"mu_s / subsistence_shares must be strictly less than 1.0 (got {mu_val}). "
                    "A share of one or more commits the whole benchmark budget and leaves no "
                    "supernumerary income."
                )
            resolved: Any = mu_val
        elif isinstance(raw_mu, Mapping):
            resolved_dict: dict[str, float] = {}
            for k, v in raw_mu.items():
                if not _is_number(v):
                    raise TypeError(f"Subsistence share for sector '{k}' must be numeric, got {type(v).__name__}.")
                v_f = float(v)
                if v_f < 0.0:
                    raise ValueError(f"mu_s / subsistence_shares for sector '{k}' must be non-negative (got {v_f}).")
                if v_f >= 1.0:
                    raise ValueError(
                        f"mu_s / subsistence_shares for sector '{k}' must be strictly less than 1.0 (got {v_f})."
                    )
                resolved_dict[str(k)] = v_f
            resolved = resolved_dict
        elif isinstance(raw_mu, np.ndarray):
            if np.any(raw_mu < 0.0):
                raise ValueError(f"All subsistence shares must be non-negative (got min={np.min(raw_mu)}).")
            if np.any(raw_mu >= 1.0):
                raise ValueError(f"All subsistence shares must be strictly less than 1.0 (got max={np.max(raw_mu)}).")
            resolved = raw_mu
        else:
            raise TypeError(
                f"subsistence_shares / mu_s must be numeric, dict, or ndarray, got {type(raw_mu).__name__}."
            )
        object.__setattr__(self, "subsistence_shares", resolved)
        object.__setattr__(self, "mu_s", resolved)
        object.__setattr__(self, "subsistence_ratio", resolved)


@dataclass(frozen=True)
class FlexibleMarketStructureConfig:
    """Market structure: Atkeson-Burstein variable markups.

    Parameters
    ----------
    variable_markups : bool, default False
        Endogenous Cournot markups. When False, pricing is competitive. When
        True the producer price satisfies ``p (1 - tax) = mu_rel(p) c_y`` with
        ``mu_rel`` the destination-averaged markup relative to its benchmark
        (benchmark markups are embedded in the calibrated costs), and the
        profits ``(p (1 - tax) - c_y) y`` accrue to the households of the
        producing country.
    sigma_j : float, default 6.0
        Within-industry substitution elasticity across origins (> 1).
    theta_j : float, default 2.0
        Across-industry substitution elasticity (> 1 and <= ``sigma_j``).
    variety_condensation : bool, default False
        Whether Dixit-Stiglitz varieties are condensed out through free entry.
        Aliased with ``condense_varieties`` and ``variety_expansion``.
    condense_varieties, variety_expansion : bool, default False
        Aliases for ``variety_condensation``.
    markup_min : float, default 1.0
        Lower markup bound (>= 1).
    markup_max : float, default 5.0
        Upper markup bound (> ``markup_min``).
    markup_bounds : tuple[float, float], optional
        ``(markup_min, markup_max)`` in one argument.
    cournot_weights : Any, optional
        Conduct parameters multiplying the market shares.
    clamping_threshold : float, default 1e-4
        Floor of the markup denominator.
    fl, fk : float, default 0.0
        Fixed labour and capital requirements per firm (non-negative).
    """

    variable_markups: bool = False
    sigma_j: float = 6.0
    theta_j: float = 2.0
    variety_condensation: bool = False
    condense_varieties: bool = False
    variety_expansion: bool = False
    markup_min: float = 1.0
    markup_max: float = 5.0
    markup_bounds: tuple[float, float] | None = None
    cournot_weights: Any = None
    clamping_threshold: float = 1e-4
    fl: float = 0.0
    fk: float = 0.0

    def __post_init__(self) -> None:
        old_inst = _get_replace_caller()
        cond_aliases = ("variety_condensation", "condense_varieties", "variety_expansion")
        if old_inst is not None and isinstance(old_inst, FlexibleMarketStructureConfig):
            exp_keys = _get_replace_explicit_keys()
            if exp_keys is not None:
                exp_cond = [k for k in cond_aliases if k in exp_keys]
            else:
                exp_cond = [k for k in cond_aliases if getattr(self, k) != getattr(old_inst, k)]
            if exp_cond:
                vals = [getattr(self, k) for k in exp_cond]
                if any(c != vals[0] for c in vals[1:]):
                    raise ValueError("Conflicting values provided for variety condensation flags.")
                cond = bool(vals[0])
            else:
                cond = bool(self.condense_varieties or self.variety_condensation or self.variety_expansion)
        else:
            cond = bool(self.condense_varieties or self.variety_condensation or self.variety_expansion)
        for name in cond_aliases:
            object.__setattr__(self, name, cond)

        def _bounds_from_tuple() -> tuple[float, float]:
            mb = self.markup_bounds
            if (not isinstance(mb, (tuple, list)) or len(mb) != 2
                    or not all(_is_number(x) for x in mb)):
                raise TypeError("markup_bounds must be a tuple of two numbers (markup_min, markup_max).")
            return float(mb[0]), float(mb[1])

        def _check_min_max() -> None:
            if not _is_number(self.markup_min):
                raise TypeError("markup_min must be numeric.")
            if not _is_number(self.markup_max):
                raise TypeError("markup_max must be numeric.")

        if old_inst is not None and isinstance(old_inst, FlexibleMarketStructureConfig):
            exp_keys = _get_replace_explicit_keys()
            if exp_keys is not None:
                b_changed = "markup_bounds" in exp_keys
                min_changed = "markup_min" in exp_keys
                max_changed = "markup_max" in exp_keys
            else:
                b_changed = self.markup_bounds != old_inst.markup_bounds
                min_changed = self.markup_min != old_inst.markup_min
                max_changed = self.markup_max != old_inst.markup_max
            if b_changed:
                b_min, b_max = _bounds_from_tuple()
                if min_changed or max_changed:
                    _check_min_max()
                    if min_changed and float(self.markup_min) != b_min:
                        raise ValueError(
                            f"Conflicting values provided for markup_bounds {self.markup_bounds} "
                            f"and markup_min ({self.markup_min})."
                        )
                    if max_changed and float(self.markup_max) != b_max:
                        raise ValueError(
                            f"Conflicting values provided for markup_bounds {self.markup_bounds} "
                            f"and markup_max ({self.markup_max})."
                        )
            else:
                _check_min_max()
                b_min, b_max = float(self.markup_min), float(self.markup_max)
        elif self.markup_bounds is not None:
            b_min, b_max = _bounds_from_tuple()
            if (self.markup_min != 1.0 and float(self.markup_min) != b_min) or (
                    self.markup_max != 5.0 and float(self.markup_max) != b_max):
                raise ValueError(
                    f"Conflicting values provided for markup_bounds {self.markup_bounds} "
                    f"and (markup_min={self.markup_min}, markup_max={self.markup_max})."
                )
        else:
            _check_min_max()
            b_min, b_max = float(self.markup_min), float(self.markup_max)
        if b_min < 1.0:
            raise ValueError(f"markup_min (markup_bounds lower bound) must be >= 1.0, got {b_min}.")
        if b_min >= b_max:
            raise ValueError(f"markup bounds must satisfy markup_min < markup_max, got ({b_min}, {b_max}).")
        object.__setattr__(self, "markup_min", b_min)
        object.__setattr__(self, "markup_max", b_max)
        object.__setattr__(self, "markup_bounds", (b_min, b_max))

        if not _is_number(self.clamping_threshold):
            raise TypeError("clamping_threshold must be numeric.")
        if self.clamping_threshold <= 0.0:
            raise ValueError(f"clamping_threshold must be strictly positive, got {self.clamping_threshold}.")
        if not _is_number(self.fl):
            raise TypeError("fl must be numeric.")
        if not _is_number(self.fk):
            raise TypeError("fk must be numeric.")
        if self.fl < 0.0 or self.fk < 0.0:
            raise ValueError(f"Fixed cost parameters fl and fk must be non-negative (got fl={self.fl}, fk={self.fk}).")

        if _is_number(self.sigma_j):
            if self.sigma_j <= 1.0:
                raise ValueError(
                    f"sigma_j must be strictly greater than 1.0, got {self.sigma_j}. "
                    "Finite Cournot markups require sigma_j > 1."
                )
        elif not (isinstance(self.sigma_j, (dict, np.ndarray)) or hasattr(self.sigma_j, "__len__")):
            raise TypeError(f"sigma_j must be numeric, dict, or ndarray, got {type(self.sigma_j).__name__}.")
        if _is_number(self.theta_j):
            if self.theta_j <= 1.0:
                raise ValueError(
                    f"theta_j must be strictly greater than 1.0, got {self.theta_j}. "
                    "The monopoly markup theta_j / (theta_j - 1) requires theta_j > 1."
                )
        elif not (isinstance(self.theta_j, (dict, np.ndarray)) or hasattr(self.theta_j, "__len__")):
            raise TypeError(f"theta_j must be numeric, dict, or ndarray, got {type(self.theta_j).__name__}.")
        if _is_number(self.sigma_j) and _is_number(self.theta_j) and self.theta_j > self.sigma_j:
            raise ValueError(
                f"theta_j ({self.theta_j}) cannot exceed sigma_j ({self.sigma_j}): the across-industry "
                "elasticity is bounded by the within-industry one."
            )


@dataclass(frozen=True)
class FlexibleTradeModelConfig:
    """Composite configuration of the flexible trade model.

    Parameters
    ----------
    technology : FlexibleTechnologyConfig
        Nested CES technology and capacity barrier.
    preference : FlexiblePreferenceConfig
        Stone-Geary household demand and final-demand sourcing.
    market_structure : FlexibleMarketStructureConfig
        Atkeson-Burstein markups.
    max_inner_iter : int, default 10
        Cap (at most 10) on the inner price fixed-point iterations in each
        finite-difference Jacobian evaluation of the condensed solver. The
        evaluations that accept a Newton step and that produce the reported
        residual iterate the inner loops to 1e-13 regardless of this setting.
    """

    technology: FlexibleTechnologyConfig = field(default_factory=FlexibleTechnologyConfig)
    preference: FlexiblePreferenceConfig = field(default_factory=FlexiblePreferenceConfig)
    market_structure: FlexibleMarketStructureConfig = field(default_factory=FlexibleMarketStructureConfig)
    max_inner_iter: int = 10

    def __post_init__(self) -> None:
        if not isinstance(self.max_inner_iter, int) or isinstance(self.max_inner_iter, bool):
            raise TypeError(f"max_inner_iter must be an integer, got {type(self.max_inner_iter).__name__}.")
        if self.max_inner_iter < 1:
            raise ValueError(f"max_inner_iter must be >= 1, got {self.max_inner_iter}.")
        if self.max_inner_iter > 10:
            raise ValueError(
                f"max_inner_iter ({self.max_inner_iter}) exceeds the latency cap of 10. It bounds the inner price "
                "loop of the finite-difference Jacobian evaluations; the step-acceptance and reported-residual "
                "evaluations iterate that loop to 1e-13 independently of this setting."
            )


def _config_field_default(cls: type, name: str) -> Any:
    """The declared default of dataclass field ``name`` on ``cls``."""
    for f in fields(cls):
        if f.name == name:
            if f.default is not MISSING:
                return f.default
            if f.default_factory is not MISSING:  # type: ignore[misc]
                return f.default_factory()  # type: ignore[misc]
            break
    raise KeyError(f"{cls.__name__} has no defaulted field {name!r}.")


def _active_flexible_settings(config: FlexibleTradeModelConfig) -> tuple[str, ...]:
    """Names of the flexible blocks whose settings differ from the dataclass defaults.

    A configuration is *active* (its equations differ from the core model)
    when this tuple is non-empty. Baselines are the dataclass field defaults,
    with an absolute tolerance of 1e-12 on numbers:

    - ``rho_va`` (1.0), ``sigma_y`` (0.0), ``sigma_inter`` (0.0);
    - ``capacity_margins`` (any value other than ``None``);
    - ``subsistence_shares`` (any non-zero share under any alias);
    - ``sigma_trade`` (any value other than ``None``);
    - ``variable_markups`` (``True``).

    Settings that only parameterise an inactive block (``penalty_scale``
    without ``capacity_margins``, ``sigma_j`` without ``variable_markups``)
    are not reported.
    """
    tech, pref, market = config.technology, config.preference, config.market_structure
    active: list[str] = []
    for name in ("rho_va", "sigma_y", "sigma_inter"):
        if _differs_from(getattr(tech, name), _config_field_default(FlexibleTechnologyConfig, name)):
            active.append(name)
    if tech.capacity_margins is not None:
        active.append("capacity_margins")
    if (_has_nonzero_setting(pref.subsistence_shares) or _has_nonzero_setting(pref.mu_s)
            or _has_nonzero_setting(pref.subsistence_ratio)):
        active.append("subsistence_shares")
    if _differs_from(pref.sigma_trade, _config_field_default(FlexiblePreferenceConfig, "sigma_trade")):
        active.append("sigma_trade")
    if bool(market.variable_markups):
        active.append("variable_markups")
    return tuple(active)


# =============================================================================
# Technology: nested CES unit costs, factor and intermediate demands
# =============================================================================

def _compute_benchmark_cva0(calib: TradeCalibrationResult) -> tuple[np.ndarray, np.ndarray]:
    """Benchmark value-added unit cost ``c_va0`` and its validity mask.

    ``c_va0 = 1 / (beta * alpha**alpha * (1 - alpha)**(1 - alpha))``, the
    Cobb-Douglas unit cost at unit factor prices, which equals the benchmark
    value-added share of output. Single-factor sectors (``alpha`` of 0 or 1)
    use the limit ``x**x -> 1``.
    """
    alpha = calib.alpha
    beta = calib.beta
    mask_valid = (beta > 0.0) & (alpha >= 0.0) & (alpha <= 1.0)
    alpha_safe = np.where(mask_valid, alpha, 0.5)
    beta_safe = np.where(mask_valid, beta, 1.0)
    term_a = np.where(alpha_safe > 0.0, alpha_safe ** alpha_safe, 1.0)
    term_1ma = np.where(alpha_safe < 1.0, (1.0 - alpha_safe) ** (1.0 - alpha_safe), 1.0)
    c_va_0 = np.where(mask_valid, 1.0 / (beta_safe * term_a * term_1ma), 0.0)
    return c_va_0, mask_valid


def _code_index(codes: Sequence[str], key: Any, kind: str, n: int) -> int:
    """Position of a country/sector given by code or integer position."""
    if _is_number(key) and float(key) == int(key):
        idx = int(key)
        if not 0 <= idx < n:
            raise IndexError(f"{kind} position {idx} is out of range for {n} {kind}s")
        return idx
    codes = list(codes)
    if key in codes:
        return codes.index(key)
    raise KeyError(f"unknown {kind} code {key!r}; known codes: {codes}")


def _capacity_margin_cells(
    calib: TradeCalibrationResult, tech_cfg: FlexibleTechnologyConfig
) -> np.ndarray:
    """Capacity margins as a ``(1, ns, nc)`` array (NaN where no barrier applies)."""
    ns, nc = calib.n_sectors, calib.n_countries
    margins = tech_cfg.capacity_margins
    out = np.full((1, ns, nc), np.nan)
    if margins is None:
        return out
    _check_capacity_arguments(margins, tech_cfg.capacity_country, tech_cfg.capacity_sector)
    if isinstance(margins, np.ndarray):
        return np.broadcast_to(np.asarray(margins, dtype=float), (1, ns, nc)).copy()
    if _is_number(margins):
        c_i = _code_index(calib.country_codes, tech_cfg.capacity_country, "country", nc)
        s_i = _code_index(calib.sector_codes, tech_cfg.capacity_sector, "sector", ns)
        out[0, s_i, c_i] = float(margins)
        return out
    for key, margin in margins.items():
        if isinstance(key, tuple):
            if len(key) != 2:
                raise ValueError(f"capacity_margins key {key!r} must be a (country, sector) pair")
            c_i = _code_index(calib.country_codes, key[0], "country", nc)
            s_i = _code_index(calib.sector_codes, key[1], "sector", ns)
        else:
            c_i = _code_index(calib.country_codes, tech_cfg.capacity_country, "country", nc)
            s_i = _code_index(calib.sector_codes, key, "sector", ns)
        out[0, s_i, c_i] = float(margin)
    return out


def compute_capacity_penalty(
    ytot: np.ndarray,
    calib: TradeCalibrationResult,
    tech_cfg: FlexibleTechnologyConfig,
) -> np.ndarray:
    """Smooth capacity barrier ``pen(y)`` of shape ``(1, ns, nc)``.

    ``pen(y) = penalty_scale * (y / ((1 + margin) y_0))**penalty_exponent``
    in the cells that ``tech_cfg.capacity_margins`` names (see
    :class:`FlexibleTechnologyConfig` for the accepted forms and the required
    ``capacity_country`` / ``capacity_sector``), zero elsewhere.
    """
    margins = _capacity_margin_cells(calib, tech_cfg)
    active = np.isfinite(margins)
    pen = np.zeros_like(margins)
    if not np.any(active):
        return pen
    y_bar = (1.0 + np.where(active, margins, 0.0)) * np.asarray(calib.ytot, dtype=float)
    ratio = np.maximum(np.asarray(ytot, dtype=float) / np.maximum(y_bar, 1e-12), 0.0)
    return np.where(active, tech_cfg.penalty_scale * ratio ** tech_cfg.penalty_exponent, 0.0)


def compute_inner_ces_cost(
    r: np.ndarray,
    w: np.ndarray,
    calib: TradeCalibrationResult,
    tech_cfg: FlexibleTechnologyConfig,
    r0: float = 1.0,
    w0: float = 1.0,
) -> np.ndarray:
    """Value-added unit cost ``c_va`` in calibrated share form, shape ``(1, ns, nc)``.

    ``c_va = c_va0 * (alpha (r/r0)**(1-rho) + (1-alpha) (w/w0)**(1-rho))**(1/(1-rho))``,
    with the exact Cobb-Douglas limit ``c_va0 (r/r0)**alpha (w/w0)**(1-alpha)``
    at ``rho_va = 1`` (the homogeneous cost of the core model).

    Parameters
    ----------
    r, w : np.ndarray
        Rental rates and wages, broadcastable to ``(1, ns, nc)``.
    calib : TradeCalibrationResult
        Calibrated model.
    tech_cfg : FlexibleTechnologyConfig
        Technology configuration.
    r0, w0 : float, default 1.0
        Benchmark factor prices.
    """
    if np.any(r < 0.0):
        raise ValueError("Capital rental rates r must be non-negative.")
    if np.any(w < 0.0):
        raise ValueError("Wage rates w must be non-negative.")
    c_va_0, mask_valid = _compute_benchmark_cva0(calib)
    alpha_safe = np.where(mask_valid, calib.alpha, 0.5)
    r_use = np.maximum(r, 1e-12)
    w_use = np.maximum(w, 1e-12)
    rho_va = tech_cfg.rho_va
    if abs(rho_va - 1.0) < 1e-6:
        r_rel = np.clip(r_use / r0, 1e-12, 1e12)
        w_rel = np.clip(w_use / w0, 1e-12, 1e12)
        term_r = np.where(alpha_safe > 0.0, r_rel ** alpha_safe, 1.0)
        term_w = np.where(alpha_safe < 1.0, w_rel ** (1.0 - alpha_safe), 1.0)
        return np.where(mask_valid, c_va_0 * term_r * term_w, 0.0)
    # Reference-price factorisation keeps every power term at most 1.
    e = 1.0 - rho_va
    r_rel = np.clip(r_use / r0, 1e-30, 1e30)
    w_rel = np.clip(w_use / w0, 1e-30, 1e30)
    ref = np.where(e > 0, np.maximum(r_rel, w_rel), np.minimum(r_rel, w_rel))
    ref_safe = np.maximum(ref, 1e-12)
    bracket = alpha_safe * (r_rel / ref_safe) ** e + (1.0 - alpha_safe) * (w_rel / ref_safe) ** e
    ratio = ref * np.maximum(bracket, 1e-300) ** (1.0 / e)
    return np.where(mask_valid, c_va_0 * ratio, 0.0)


def compute_inner_ces_derivatives(
    r: np.ndarray,
    w: np.ndarray,
    c_va: np.ndarray,
    calib: TradeCalibrationResult,
    tech_cfg: FlexibleTechnologyConfig,
    r0: float = 1.0,
    w0: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalised gradients ``(1/c_va0) dc_va/dw`` and ``(1/c_va0) dc_va/dr``.

    Dividing by ``c_va0`` makes the benchmark factor demands reproduce the
    calibrated endowments exactly.

    Returns
    -------
    norm_dc_dw, norm_dc_dr : np.ndarray, shape (1, ns, nc)
    """
    c_va_0, mask_valid = _compute_benchmark_cva0(calib)
    alpha_safe = np.where(mask_valid, calib.alpha, 0.5)
    r_rel = np.clip(np.maximum(r, 1e-12) / r0, 1e-30, 1e30)
    w_rel = np.clip(np.maximum(w, 1e-12) / w0, 1e-30, 1e30)
    c_va_0_safe = np.where(mask_valid, np.maximum(c_va_0, 1e-12), 1.0)
    c_va_ratio = np.clip(np.maximum(c_va, 1e-12) / c_va_0_safe, 1e-30, 1e30)
    ratio_w = c_va_ratio / w_rel
    ratio_r = c_va_ratio / r_rel
    rho_va = tech_cfg.rho_va
    if abs(rho_va - 1.0) < 1e-6:
        pow_w, pow_r = ratio_w, ratio_r
    else:
        pow_w = np.exp(np.clip(rho_va * np.log(np.maximum(ratio_w, 1e-300)), -300.0, 300.0))
        pow_r = np.exp(np.clip(rho_va * np.log(np.maximum(ratio_r, 1e-300)), -300.0, 300.0))
    norm_dc_dw = np.where(mask_valid & (alpha_safe < 1.0), ((1.0 - alpha_safe) / w0) * pow_w, 0.0)
    norm_dc_dr = np.where(mask_valid & (alpha_safe > 0.0), (alpha_safe / r0) * pow_r, 0.0)
    return norm_dc_dw, norm_dc_dr


def compute_intermediate_composite_price(
    p: np.ndarray,
    tau_a: np.ndarray,
    calib: TradeCalibrationResult,
    sigma: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Purchaser price index ``P_M`` of each sector's intermediate composite.

    The weights are the benchmark origin shares ``a / sum(a)``; ``P_M = 1`` at
    unit prices without tariffs. ``sigma`` below 1e-6 is fixed coefficients,
    1 is Cobb-Douglas, otherwise CES.

    Returns
    -------
    P_M : np.ndarray, shape (1, ns, nc)
        Composite price index.
    inter_cost : np.ndarray, shape (1, ns, nc)
        Intermediate cost per unit of output, ``A * P_M``.
    p_tau_safe : np.ndarray, shape (ns*nc, ns, nc)
        Purchaser prices ``p * tau_a`` (floored at 1e-12).
    """
    A_mat, omega = _get_ces_weights(calib)
    p_vec = p.flatten(order="F") if p.ndim == 3 else np.asarray(p, dtype=float).ravel()
    p_tau = p_vec[:, np.newaxis, np.newaxis] * tau_a
    p_tau_safe = np.maximum(p_tau, 1e-12)
    if sigma < 1e-6:
        P_M_2d = np.sum(np.where(omega > 0, omega * p_tau, 0.0), axis=0)
    elif abs(sigma - 1.0) < 1e-6:
        ln_p_tau = np.where(omega > 0, np.log(p_tau_safe), 0.0)
        P_M_2d = np.exp(np.sum(np.where(omega > 0, omega * ln_p_tau, 0.0), axis=0))
    else:
        e = 1.0 - sigma
        has_active = np.any(omega > 0, axis=0, keepdims=True)
        if e < 0:
            p_ref = np.where(has_active, np.min(np.where(omega > 0, p_tau_safe, np.inf), axis=0, keepdims=True), 1.0)
        else:
            p_ref = np.where(has_active, np.max(np.where(omega > 0, p_tau_safe, -np.inf), axis=0, keepdims=True), 1.0)
        p_ref_safe = np.maximum(p_ref, 1e-12)
        u = np.where(omega > 0, p_tau_safe / p_ref_safe, 1.0)
        inner = np.sum(np.where(omega > 0, omega * (u ** e), 0.0), axis=0)
        P_M_2d = p_ref.squeeze(0) * (np.maximum(inner, 1e-300) ** (1.0 / e))
        P_M_2d = np.where(has_active.squeeze(0), P_M_2d, 1.0)
    P_M = P_M_2d[np.newaxis, :, :]
    inter_cost = (A_mat * P_M_2d)[np.newaxis, :, :]
    return P_M, inter_cost, p_tau_safe


def _outer_benchmark(calib: TradeCalibrationResult) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Benchmark gross-output cost ``c_y0`` and the value-added / intermediate shares."""
    A_mat, _ = _get_ces_weights(calib)
    theta_m0 = A_mat[np.newaxis, :, :]
    c_va_0, mask_valid = _compute_benchmark_cva0(calib)
    theta_va0 = np.where(mask_valid, c_va_0, 1.0 - calib.tax - theta_m0)
    c_y0 = theta_va0 + theta_m0
    c_y0_safe = np.where(c_y0 > 0, c_y0, 1.0)
    s_va0 = np.where(c_y0 > 0, theta_va0 / c_y0_safe, 0.5)
    s_m0 = np.where(c_y0 > 0, theta_m0 / c_y0_safe, 0.5)
    return c_y0, s_va0, s_m0


def compute_outer_ces_cost(
    c_va: np.ndarray,
    P_M: np.ndarray,
    calib: TradeCalibrationResult,
    tech_cfg: FlexibleTechnologyConfig,
    normalized: bool = False,
) -> np.ndarray:
    """Gross-output unit cost ``c_y`` in calibrated share form.

    Parameters
    ----------
    c_va : np.ndarray, shape (1, ns, nc)
        Value-added unit cost (including any capacity penalty).
    P_M : np.ndarray, shape (1, ns, nc)
        Intermediate composite price index (1 at the benchmark).
    calib : TradeCalibrationResult
        Calibrated model.
    tech_cfg : FlexibleTechnologyConfig
        Technology configuration (``sigma_y``).
    normalized : bool, default False
        True returns the index (1 at the benchmark); False the level
        ``c_y0 * index``, whose benchmark value is the benchmark cost share of
        output (``1 - tax``).
    """
    c_y0, s_va0, s_m0 = _outer_benchmark(calib)
    c_va_0, mask_valid = _compute_benchmark_cva0(calib)
    theta_va0 = np.where(mask_valid, c_va_0, c_y0 * s_va0)
    rel_c_va = np.maximum(c_va / np.where(theta_va0 > 0, theta_va0, 1.0), 1e-12)
    rel_P_M = np.maximum(P_M, 1e-12)
    sigma_y = tech_cfg.sigma_y
    if sigma_y < 1e-6:
        c_y_norm = s_va0 * rel_c_va + s_m0 * rel_P_M
    elif abs(sigma_y - 1.0) < 1e-6:
        c_y_norm = (rel_c_va ** s_va0) * (rel_P_M ** s_m0)
    else:
        e = 1.0 - sigma_y
        P_ref = np.minimum(rel_c_va, rel_P_M) if e < 0 else np.maximum(rel_c_va, rel_P_M)
        P_ref_safe = np.maximum(P_ref, 1e-12)
        bracket = s_va0 * (rel_c_va / P_ref_safe) ** e + s_m0 * (rel_P_M / P_ref_safe) ** e
        c_y_norm = P_ref * (np.maximum(bracket, 1e-300) ** (1.0 / e))
    if normalized:
        return c_y_norm
    return c_y0 * c_y_norm


def compute_nested_ces_costs(
    r: np.ndarray,
    w: np.ndarray,
    P_M: np.ndarray,
    calib: TradeCalibrationResult,
    tech_cfg: FlexibleTechnologyConfig,
    *,
    ytot: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Value-added cost ``c_va`` and normalised gross-output cost ``c_y``.

    With a capacity barrier (``tech_cfg.capacity_margins``) and ``ytot``
    given, the outer nest is evaluated at the penalised value-added cost
    ``c_va (1 + pen(ytot))``; the returned ``c_va`` stays the factor-price cost
    of one unit of value added. :func:`compute_nested_factor_demands` scales
    labour and capital by the same ``1 + pen``, so cost is exhausted. Pass the
    same ``ytot`` to both functions.

    Returns
    -------
    c_va : np.ndarray, shape (1, ns, nc)
        Value-added unit cost at factor prices (without the penalty).
    c_y : np.ndarray, shape (1, ns, nc)
        Gross-output unit cost index (1 at the benchmark without a penalty).
    """
    c_va = compute_inner_ces_cost(r, w, calib, tech_cfg)
    c_va_eff = c_va
    if tech_cfg.capacity_margins is not None and ytot is not None:
        c_va_eff = c_va * (1.0 + compute_capacity_penalty(np.asarray(ytot, dtype=float), calib, tech_cfg))
    c_y = compute_outer_ces_cost(c_va_eff, P_M, calib, tech_cfg, normalized=True)
    return c_va, c_y


def compute_nested_factor_demands(
    ytot: np.ndarray,
    r: np.ndarray,
    w: np.ndarray,
    P_M: np.ndarray,
    c_va: np.ndarray,
    c_y: np.ndarray,
    p: np.ndarray,
    tau: np.ndarray,
    calib: TradeCalibrationResult,
    tech_cfg: FlexibleTechnologyConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Labour, capital and intermediate demands of the nested CES technology.

    With ``c_y`` the normalised gross-output cost of
    :func:`compute_nested_ces_costs` and ``pen = 0`` unless a capacity barrier
    is set::

        xl    = c_va0 Y (c_y / (c_va (1 + pen) / c_va0))**sigma_y [dc_va/dw / c_va0] (1 + pen)
        xk    = c_va0 Y (c_y / (c_va (1 + pen) / c_va0))**sigma_y [dc_va/dr / c_va0] (1 + pen)
        x_mat = a Y (c_y / P_M)**sigma_y (P_M / (p tau))**sigma_inter

    The benchmark demands reproduce the calibrated factors and intermediate
    coefficients, and ``w xl + r xk + sum_i p_i tau_i x_mat_i = c_y0 c_y Y``.

    Returns
    -------
    xl, xk : np.ndarray, shape (1, ns, nc)
        Labour and capital demands.
    x_mat : np.ndarray, shape (ns*nc, ns, nc)
        Intermediate deliveries by origin cell.
    """
    for name, arr in (("ytot", ytot), ("r", r), ("w", w), ("P_M", P_M), ("p", p)):
        if np.any(np.asarray(arr) < 0.0):
            raise ValueError(f"{name} must be non-negative.")
    c_va_0, mask_valid = _compute_benchmark_cva0(calib)
    mask_active = mask_valid & (ytot > 0.0)
    norm_dc_dw, norm_dc_dr = compute_inner_ces_derivatives(r, w, c_va, calib, tech_cfg)

    pen_factor: float | np.ndarray = 1.0
    if tech_cfg.capacity_margins is not None:
        pen_factor = 1.0 + compute_capacity_penalty(ytot, calib, tech_cfg)

    sigma_y = tech_cfg.sigma_y
    if sigma_y < 1e-6:
        term_va_outer: float | np.ndarray = 1.0
        term_mat_outer: float | np.ndarray = 1.0
    else:
        cva0_safe = np.where(c_va_0 > 0, c_va_0, 1.0)
        cva_rel = np.clip(np.maximum(c_va * pen_factor, 1e-12) / cva0_safe, 1e-30, 1e30)
        cy_rel = np.clip(np.maximum(c_y, 1e-12), 1e-30, 1e30)
        pm_rel = np.clip(np.maximum(P_M, 1e-12), 1e-30, 1e30)
        term_va_outer = np.exp(np.clip(sigma_y * np.log(cy_rel / cva_rel), -300.0, 300.0))
        term_mat_outer = np.exp(np.clip(sigma_y * np.log(cy_rel / pm_rel), -300.0, 300.0))

    xl = np.where(mask_active, c_va_0 * ytot * term_va_outer * norm_dc_dw * pen_factor, 0.0)
    xk = np.where(mask_active, c_va_0 * ytot * term_va_outer * norm_dc_dr * pen_factor, 0.0)

    p_vec = p.flatten(order="F") if p.ndim == 3 else np.asarray(p, dtype=float).ravel()
    p_tau_safe = np.maximum(p_vec[:, np.newaxis, np.newaxis] * tau, 1e-12)
    term_sourcing: float | np.ndarray = 1.0
    if tech_cfg.sigma_inter > 1e-6:
        sourcing_ratio = np.clip(np.maximum(P_M, 1e-12) / p_tau_safe, 1e-12, 1e12)
        term_sourcing = np.exp(np.clip(tech_cfg.sigma_inter * np.log(sourcing_ratio), -80.0, 80.0))
    x_mat = np.where(ytot > 0, calib.a * ytot * term_mat_outer * term_sourcing, 0.0)
    return xl, xk, x_mat


def compute_zero_profit_price(
    c_y: np.ndarray,
    calib: TradeCalibrationResult,
    sub_prod: np.ndarray | None = None,
    ytot: np.ndarray | None = None,
    normalized: bool = True,
) -> np.ndarray:
    """Zero-profit producer prices ``pp = c_y_level / (1 - tax) - sub_prod``.

    Parameters
    ----------
    c_y : np.ndarray, shape (1, ns, nc)
        Gross-output unit cost: the index of :func:`compute_nested_ces_costs`
        (``normalized=True``, the default) or the level of
        :func:`compute_outer_ces_cost` (``normalized=False``).
    calib : TradeCalibrationResult
        Calibrated model.
    sub_prod : np.ndarray, optional
        Production subsidy per unit of output.
    ytot : np.ndarray, optional
        Output; cells with zero output get price 1.
    normalized : bool, default True
        Whether ``c_y`` is the index or the level.
    """
    c_y_level = _outer_benchmark(calib)[0] * c_y if normalized else c_y
    pp = c_y_level / np.maximum(1.0 - calib.tax, 1e-12)
    if sub_prod is not None:
        pp = pp - sub_prod
    y_mask = ytot if ytot is not None else calib.ytot
    return np.where(y_mask > 0, pp, 1.0)


def compute_zero_profit_prices(
    c_y: np.ndarray,
    tax: np.ndarray,
    sub_prod: np.ndarray | None = None,
    ytot: np.ndarray | None = None,
) -> np.ndarray:
    """Zero-profit prices ``c_y / (1 - tax) - sub_prod`` for a unit-cost level ``c_y``."""
    pp = c_y / np.maximum(1.0 - tax, 1e-12)
    if sub_prod is not None:
        pp = pp - sub_prod
    if ytot is not None:
        pp = np.where(ytot > 0, pp, 1.0)
    return pp


# =============================================================================
# Preferences: Stone-Geary household demand and final-demand sourcing
# =============================================================================

def smooth_subsistence_scaling(u: np.ndarray | float) -> np.ndarray | float:
    """Subsistence scaling ``g(u) = tanh(3u) / tanh(3)``.

    ``g(1) = 1``, ``g(0) = 0``, ``g`` is strictly increasing and concave on
    ``u > 0`` and tends to ``1 / tanh(3)`` (about 1.005) as ``u`` grows, so
    subsistence quantities shrink smoothly when income collapses.
    """
    denom = np.tanh(3.0)
    if isinstance(u, (int, float, np.floating, np.integer)):
        return float(np.tanh(3.0 * u) / denom)
    return np.tanh(3.0 * np.asarray(u, dtype=float)) / denom


def _extract_benchmark_household_data(
    calib: TradeCalibrationResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Benchmark household budget, sector shares and sector quantities.

    The household is final-use category 0. Its budget is its benchmark
    spending on goods net of final-use taxes (the ``basic`` spending of
    :func:`puremacro.trade._accounting._parameters`); at unit prices without
    tariffs the sector quantities equal the sector spending.

    Returns
    -------
    budget_0 : np.ndarray, shape (1, 1, nc)
        Benchmark household spending on goods (net of final-use taxes).
    theta_sec_0 : np.ndarray, shape (1, ns, nc)
        Benchmark sector shares of that spending (summing to one).
    c_s0 : np.ndarray, shape (1, ns, nc)
        Benchmark sector quantities (composite units).
    """
    from puremacro.trade._accounting import _parameters

    nc, ns, nfd = calib.n_countries, calib.n_sectors, calib.n_final_demand
    _, basic0, _ = _parameters(calib)
    budget_0 = basic0[:, 0:1, :].copy()
    theta_sec_0 = np.sum(np.asarray(calib.afd, dtype=float)[:, 0, :].reshape(nc, ns, nc), axis=0)[np.newaxis]
    sums = np.sum(theta_sec_0, axis=1, keepdims=True)
    theta_sec_0 = np.where(sums > 0, theta_sec_0 / np.where(sums > 0, sums, 1.0), 1.0 / ns)
    c_s0 = theta_sec_0 * budget_0
    return budget_0, theta_sec_0, c_s0


def _resolve_subsistence_shares(
    calib: TradeCalibrationResult,
    pref_cfg: FlexiblePreferenceConfig | None = None,
) -> np.ndarray:
    """Subsistence shares as a ``(1, ns, nc)`` array.

    A number applies to every sector; a mapping is keyed by sector code
    (case-insensitive) or position (``"S0"``, ``"sector_0"``, ``"0"``), and
    sectors it omits get 0; an unknown key raises ``KeyError``. An array of
    shape ``(ns,)``, ``(ns, nc)`` or ``(1, ns, nc)`` is used as given.
    """
    ns, nc = calib.n_sectors, calib.n_countries
    mu_arr = np.zeros((1, ns, nc), dtype=float)
    if pref_cfg is None:
        return mu_arr
    mu_input = pref_cfg.subsistence_shares if pref_cfg.subsistence_shares is not None else pref_cfg.mu_s
    if mu_input is None:
        return mu_arr
    if _is_number(mu_input):
        mu_arr[:] = float(mu_input)
    elif isinstance(mu_input, Mapping):
        codes = [str(c) for c in calib.sector_codes]
        for k, v in mu_input.items():
            k_str = str(k).strip()
            s_idx = None
            for idx, code in enumerate(codes):
                if k_str.lower() == code.lower():
                    s_idx = idx
                    break
            if s_idx is None:
                for idx in range(ns):
                    if k_str.lower() in (f"s{idx}", f"s{idx:02d}", f"sector_{idx}") or k_str == str(idx):
                        s_idx = idx
                        break
            if s_idx is None:
                raise KeyError(f"unknown sector {k!r} in subsistence_shares; known codes: {codes}")
            mu_arr[0, s_idx, :] = float(v)
    elif isinstance(mu_input, np.ndarray):
        if mu_input.ndim == 1:
            mu_arr[0, :, :] = mu_input[:, np.newaxis]
        elif mu_input.ndim == 2:
            mu_arr[0, :, :] = mu_input
        else:
            mu_arr = np.broadcast_to(mu_input, (1, ns, nc)).copy()
    return mu_arr


def compute_les_marginal_budget_shares(
    calib: TradeCalibrationResult,
    pref_cfg: FlexiblePreferenceConfig | None = None,
) -> np.ndarray:
    """Marginal budget shares ``theta_LES`` of the household, shape ``(1, ns, nc)``.

    ``theta_LES_s = (1 - mu_s) E_s0 / (E_0 - sum_k mu_k E_k0)`` with ``E_s0``
    the benchmark sector spending; they sum to one in every country, and equal
    the benchmark shares when there is no subsistence.
    """
    budget_0, theta_sec_0, E_s0 = _extract_benchmark_household_data(calib)
    mu_arr = _resolve_subsistence_shares(calib, pref_cfg)
    if np.all(mu_arr == 0.0):
        return theta_sec_0
    denom = budget_0 - np.sum(mu_arr * E_s0, axis=1, keepdims=True)
    theta_les = np.divide((1.0 - mu_arr) * E_s0, np.where(denom > 1e-12, denom, 1.0),
                          out=theta_sec_0.copy(), where=(denom > 1e-12))
    sums = np.sum(theta_les, axis=1, keepdims=True)
    return np.where(sums > 0, theta_les / np.where(sums > 0, sums, 1.0), theta_sec_0)


def _les_demand(
    budget: np.ndarray,
    P_C: np.ndarray,
    calib: TradeCalibrationResult,
    pref_cfg: FlexiblePreferenceConfig | None,
    *,
    warn: bool,
) -> np.ndarray:
    """Household sector quantities ``c_bar + theta_LES (budget - P c_bar) / P_LES``."""
    budget_0, theta_sec_0, c_s0 = _extract_benchmark_household_data(calib)
    P_safe = np.maximum(P_C, 1e-12)
    mu_arr = _resolve_subsistence_shares(calib, pref_cfg)
    if np.all(mu_arr == 0.0):
        return theta_sec_0 * budget / np.sum(theta_sec_0 * P_safe, axis=1, keepdims=True)
    u = np.maximum(budget / np.maximum(budget_0, 1e-300), 0.0)
    c_bar = mu_arr * c_s0 * smooth_subsistence_scaling(u)
    theta_les = compute_les_marginal_budget_shares(calib, pref_cfg)
    sub_exp = np.sum(P_safe * c_bar, axis=1, keepdims=True)
    raw_super = budget - sub_exp
    scale = np.maximum(np.abs(budget), np.abs(sub_exp))
    infeasible = raw_super < -1e-12 * scale
    if warn and np.any(infeasible):
        overspend = np.where(infeasible, -raw_super / np.maximum(np.abs(budget), 1e-300), 0.0)
        warnings.warn(
            "Stone-Geary subsistence spending exceeds the household budget in "
            f"{int(np.count_nonzero(infeasible))} of {infeasible.size} countries "
            f"(largest overspend {100.0 * float(np.max(overspend)):.4g}% of the budget); "
            "supernumerary spending is clamped at zero, so the returned bundle costs more "
            "than the budget. Lower the subsistence shares or raise income.",
            RuntimeWarning,
            stacklevel=3,
        )
    p_les = np.sum(theta_les * P_safe, axis=1, keepdims=True)
    return c_bar + theta_les * np.maximum(raw_super, 0.0) / np.maximum(p_les, 1e-300)


def compute_stone_geary_final_demand(
    budget: np.ndarray,
    P_C: np.ndarray,
    calib: TradeCalibrationResult,
    pref_cfg: FlexiblePreferenceConfig | None = None,
) -> np.ndarray:
    """Household sector quantities under Stone-Geary (LES) preferences.

    ``c_s = c_bar_s + theta_LES_s (B - sum_k P_k c_bar_k) / P_LES`` with
    ``c_bar_s = mu_s c_s0 g(B / B_0)``, ``P_LES = sum_k theta_LES_k P_k`` and
    ``g`` from :func:`smooth_subsistence_scaling`: subsistence quantities
    scale smoothly with the budget and supernumerary spending buys the
    benchmark basket in fixed real proportions, so at ``mu = 0`` this is the
    core model's fixed-proportion basket.

    Feasibility: supernumerary spending is clamped at zero. While it is
    non-negative the bundle exhausts the budget, ``sum_s P_s c_s = B``; when
    subsistence spending exceeds the budget the bundle is ``c_bar`` itself,
    which costs more than the budget, and a ``RuntimeWarning`` reports the
    overspend. Because ``g(u) > u`` on ``(0, 1)``, at benchmark prices the
    clamp can bind only if ``sum_s mu_s theta_s0 > tanh(3)/3`` (about 0.332).

    Parameters
    ----------
    budget : np.ndarray, shape (1, 1, nc)
        Household spending on goods net of final-use taxes, ``B``.
    P_C : np.ndarray, shape (1, ns, nc) or (ns, nc)
        Sector composite purchaser prices (1 at the benchmark).
    calib : TradeCalibrationResult
        Calibrated model; ``B_0`` and ``c_s0`` are its benchmark household
        budget and sector quantities (:func:`_extract_benchmark_household_data`).
    pref_cfg : FlexiblePreferenceConfig, optional
        Subsistence shares.

    Returns
    -------
    np.ndarray, shape (1, ns, nc)
        Household sector quantities.
    """
    nc, ns = calib.n_countries, calib.n_sectors
    budget_3d = np.asarray(budget, dtype=float).reshape((1, 1, nc))
    P_C_3d = np.asarray(P_C, dtype=float)
    P_C_3d = P_C_3d[np.newaxis] if P_C_3d.ndim == 2 else P_C_3d.reshape((1, ns, nc))
    return _les_demand(budget_3d, P_C_3d, calib, pref_cfg, warn=True)


def _get_armington_weights(
    calib: TradeCalibrationResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Benchmark final-demand sourcing weights (cached on the calibration).

    Returns
    -------
    theta_sec : np.ndarray, shape (ns, 1, nfd, nc)
        Sector shares of each category's basket.
    b_4d : np.ndarray, shape (ns, nc, nfd, nc)
        Origin shares within each sector composite (summing to one over
        origins; a negative benchmark delivery keeps its sign).
    active_mask : np.ndarray, shape (ns, nfd, nc)
        Sector composites with a non-zero benchmark share.
    """
    cached = getattr(calib, "_armington_weights_signed_cache", None)
    if cached is not None:
        return cached
    ns, nc, nfd = calib.n_sectors, calib.n_countries, calib.n_final_demand
    afd_4d = np.asarray(calib.afd, dtype=float).reshape(nc, ns, nfd, nc).transpose(1, 0, 2, 3)
    theta_sec = np.sum(afd_4d, axis=1, keepdims=True)
    nonzero = theta_sec != 0.0
    b_4d = np.divide(afd_4d, np.where(nonzero, theta_sec, 1.0), out=np.zeros_like(afd_4d), where=nonzero)
    cached_val = (theta_sec, b_4d, nonzero[:, 0, :, :])
    try:
        object.__setattr__(calib, "_armington_weights_signed_cache", cached_val)
    except Exception:
        pass
    return cached_val


def _armington_split(
    b_4d: np.ndarray,
    p_tau_safe: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split signed origin shares into a fixed-coefficient part and a CES part.

    Returns ``(neg_cost, b_plus, pos)``: ``sum_{b<0} b p tau`` and
    ``sum_{b>0} b`` (shape ``(ns, 1, nfd, nc)``) and the mask of positive cells.
    """
    pos = b_4d > 0.0
    neg = b_4d < 0.0
    neg_cost = np.sum(np.where(neg, b_4d * p_tau_safe, 0.0), axis=1, keepdims=True)
    b_plus = np.sum(np.where(pos, b_4d, 0.0), axis=1, keepdims=True)
    return neg_cost, b_plus, pos


def _fd_price_tensors(
    p: np.ndarray, tau_fd: np.ndarray, calib: TradeCalibrationResult
) -> np.ndarray:
    """Purchaser prices ``p_i tau_i`` of final deliveries, shape ``(ns, nc, nfd, nc)``."""
    ns, nc, nfd = calib.n_sectors, calib.n_countries, calib.n_final_demand
    if tau_fd.ndim == 3 and tau_fd.shape == (ns * nc, nfd, nc):
        taufd_4d = tau_fd.reshape(nc, ns, nfd, nc).transpose(1, 0, 2, 3)
    elif tau_fd.ndim == 4:
        taufd_4d = np.asarray(tau_fd, dtype=float)
    else:
        raise ValueError(f"Unexpected tau_fd shape: {tau_fd.shape}")
    if p.ndim == 3:
        p_4d = p.squeeze(0)[:, :, np.newaxis, np.newaxis]
    elif p.ndim == 1:
        p_4d = p.reshape(nc, ns).T[:, :, np.newaxis, np.newaxis]
    else:
        p_4d = np.asarray(p, dtype=float).reshape(ns, nc, 1, 1)
    return np.maximum(p_4d * taufd_4d, 1e-12)


def compute_armington_purchaser_prices(
    p: np.ndarray,
    tau_fd: np.ndarray,
    calib: TradeCalibrationResult,
    sigma_trade: float | np.ndarray | None = None,
) -> np.ndarray:
    """Sector composite purchaser prices of final demand, shape ``(ns, nfd, nc)``.

    With all benchmark origin shares positive,
    ``P = [sum_i b_i (p_i tau_i)**(1 - sigma)]**(1 / (1 - sigma))``. Origins
    with a negative benchmark share (an inventory drawdown, for example) are
    fixed-coefficient deliveries and the positive origins form a CES aggregate
    with weight ``B_+ = sum_{b>0} b``: ``P = sum_{b<0} b p tau + B_+ P_+``. At
    ``p = tau = 1``, ``P = 1`` for any signed data. ``sigma_trade`` of
    ``None`` (default) or below 1e-6 is fixed coefficients,
    ``P = sum_i b_i p_i tau_i``.

    Parameters
    ----------
    p : np.ndarray, shape (1, ns, nc) or (ns*nc,)
        Producer prices.
    tau_fd : np.ndarray, shape (ns*nc, nfd, nc) or (ns, nc, nfd, nc)
        Final-demand tariff multipliers.
    calib : TradeCalibrationResult
        Calibrated model.
    sigma_trade : float, np.ndarray or None
        Armington elasticity across origins (a scalar or one per sector).
    """
    ns = calib.n_sectors
    _, b_4d, active_mask = _get_armington_weights(calib)
    sigma_trade = 0.0 if sigma_trade is None else sigma_trade
    p_tau_safe = _fd_price_tensors(p, tau_fd, calib)

    neg_cost_4d, b_plus_4d, pos = _armington_split(b_4d, p_tau_safe)
    has_neg_4d = np.any(b_4d < 0.0, axis=1, keepdims=True)
    b_plus_safe = np.where(b_plus_4d > 0.0, b_plus_4d, 1.0)
    w_pos = np.where(pos, np.where(has_neg_4d, b_4d / b_plus_safe, b_4d), 0.0)
    has_neg = has_neg_4d.squeeze(1)
    neg_cost = neg_cost_4d.squeeze(1)
    b_plus = b_plus_4d.squeeze(1)

    def _combine(P_plus: np.ndarray) -> np.ndarray:
        return np.where(has_neg, neg_cost + b_plus * P_plus, P_plus)

    P_leo = np.sum(np.where(b_4d != 0.0, b_4d * p_tau_safe, 0.0), axis=1)
    has_active = np.any(pos, axis=1, keepdims=True)
    if _is_number(sigma_trade):
        sig = float(sigma_trade)
        if sig < 1e-6:
            P = P_leo
        elif abs(sig - 1.0) < 1e-6:
            ln_pt = np.where(pos, np.log(p_tau_safe), 0.0)
            P = _combine(np.exp(np.sum(np.where(pos, w_pos * ln_pt, 0.0), axis=1)))
        else:
            e = 1.0 - sig
            if e < 0:
                p_ref = np.where(has_active, np.min(np.where(pos, p_tau_safe, np.inf), axis=1, keepdims=True), 1.0)
            else:
                p_ref = np.where(has_active, np.max(np.where(pos, p_tau_safe, -np.inf), axis=1, keepdims=True), 1.0)
            u_pt = np.where(pos, p_tau_safe / np.maximum(p_ref, 1e-12), 1.0)
            inner = np.sum(np.where(pos, w_pos * (u_pt ** e), 0.0), axis=1)
            P = _combine(p_ref.squeeze(1) * (np.maximum(inner, 1e-300) ** (1.0 / e)))
    else:
        sig_arr = np.asarray(sigma_trade, dtype=float).reshape(ns, 1, 1)
        e_arr = 1.0 - sig_arr
        p_min = np.where(has_active, np.min(np.where(pos, p_tau_safe, np.inf), axis=1, keepdims=True), 1.0)
        p_max = np.where(has_active, np.max(np.where(pos, p_tau_safe, -np.inf), axis=1, keepdims=True), 1.0)
        e_4d = e_arr[:, :, np.newaxis]
        p_ref = np.where(e_4d < 0, p_min, p_max)
        u_pt = np.where(pos, p_tau_safe / np.maximum(p_ref, 1e-12), 1.0)
        inner = np.sum(np.where(pos, w_pos * (u_pt ** e_4d), 0.0), axis=1)
        P_ces = _combine(p_ref.squeeze(1) * (np.maximum(inner, 1e-300) ** (1.0 / e_arr)))
        ln_pt = np.where(pos, np.log(p_tau_safe), 0.0)
        P_cd = _combine(np.exp(np.sum(np.where(pos, w_pos * ln_pt, 0.0), axis=1)))
        P = np.where(sig_arr < 1e-6, P_leo, np.where(np.abs(sig_arr - 1.0) < 1e-6, P_cd, P_ces))
    return np.where(active_mask, P, 1.0)


def compute_armington_final_demands(
    c_sec: np.ndarray,
    P_C: np.ndarray,
    p: np.ndarray,
    tau_fd: np.ndarray,
    calib: TradeCalibrationResult,
    sigma_trade: float | np.ndarray | None = None,
    as_3d: bool = True,
) -> np.ndarray:
    """Final deliveries by origin, ``x_i = b_i c (P_+ / (p_i tau_i))**sigma``.

    Negative-share origins deliver the fixed quantity ``b_i c``; positive
    origins are sourced at the CES sub-aggregate price
    ``P_+ = (P - sum_{b<0} b p tau) / B_+``, so ``sum_i p_i tau_i x_i = P c``
    exactly. ``sigma_trade=None`` (default) is fixed coefficients, ``x = b c``.

    Parameters
    ----------
    c_sec : np.ndarray, shape (ns, nfd, nc)
        Sector composite quantities by category and destination.
    P_C : np.ndarray, shape (ns, nfd, nc)
        Sector composite prices (:func:`compute_armington_purchaser_prices`).
    p, tau_fd, calib, sigma_trade
        As in :func:`compute_armington_purchaser_prices`.
    as_3d : bool, default True
        Return shape ``(ns*nc, nfd, nc)`` (True) or ``(ns, nc, nfd, nc)``.
    """
    ns, nc, nfd = calib.n_sectors, calib.n_countries, calib.n_final_demand
    _, b_4d, _ = _get_armington_weights(calib)
    sigma_trade = 0.0 if sigma_trade is None else sigma_trade
    p_tau_safe = _fd_price_tensors(p, tau_fd, calib)
    c_sec_use = np.repeat(c_sec.transpose(1, 0, 2), nfd, axis=1) if c_sec.shape == (1, ns, nc) else c_sec
    P_C_use = np.repeat(P_C.transpose(1, 0, 2), nfd, axis=1) if P_C.shape == (1, ns, nc) else P_C

    neg_cost_4d, b_plus_4d, pos = _armington_split(b_4d, p_tau_safe)
    has_neg_4d = np.any(b_4d < 0.0, axis=1, keepdims=True)
    b_plus_safe = np.where(b_plus_4d > 0.0, b_plus_4d, 1.0)
    P_ref_4d = P_C_use[:, np.newaxis, :, :]
    P_plus_4d = np.where(has_neg_4d, (P_ref_4d - neg_cost_4d) / b_plus_safe, P_ref_4d)
    rel_p = np.clip(P_plus_4d / p_tau_safe, 1e-30, 1e30)
    if _is_number(sigma_trade):
        term_sourcing = np.exp(np.clip(float(sigma_trade) * np.log(rel_p), -80.0, 80.0))
    else:
        sig_arr = np.asarray(sigma_trade, dtype=float).reshape(ns, 1, 1, 1)
        term_sourcing = np.exp(np.clip(sig_arr * np.log(rel_p), -80.0, 80.0))
    term_sourcing = np.where(pos, term_sourcing, 1.0)
    xfd_4d = b_4d * c_sec_use[:, np.newaxis, :, :] * term_sourcing
    if as_3d:
        return xfd_4d.transpose(1, 0, 2, 3).reshape(ns * nc, nfd, nc)
    return xfd_4d


def compute_multi_category_final_demands(
    budget: np.ndarray,
    p: np.ndarray,
    tau_fd: np.ndarray,
    calib: TradeCalibrationResult,
    pref_cfg: FlexiblePreferenceConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Final deliveries of every final-use category (any number of categories).

    ``budget[0, f, c]`` is category ``f``'s spending on goods in country ``c``
    net of final-use taxes, ``X = expenditure (1 - fd_tax)`` in the consistent
    accounting (foreign saving already deducted from the investment
    category). Category 0 is the household: Stone-Geary demand when
    ``pref_cfg`` sets subsistence shares (:func:`compute_stone_geary_final_demand`),
    otherwise the fixed-proportion basket. Every other category buys its
    benchmark sector basket in fixed proportions, ``c_s = theta_s X / Q`` with
    ``Q = sum_s theta_s P_s``. Origins follow ``pref_cfg.sigma_trade``
    (:func:`compute_armington_final_demands`), so the purchaser value of each
    category's deliveries equals ``X`` (exactly for fixed proportions and
    Armington sourcing, and for the LES while supernumerary spending is
    non-negative).

    Returns
    -------
    xc : np.ndarray, shape (ns*nc, nfd, nc)
        Deliveries by origin cell, category and destination.
    P_C : np.ndarray, shape (ns, nfd, nc)
        Sector composite purchaser prices.
    Q : np.ndarray, shape (1, nfd, nc)
        Category price index at benchmark sector weights (1 for categories with
        no benchmark basket).
    c_sec : np.ndarray, shape (ns, nfd, nc)
        Sector composite quantities.
    """
    if pref_cfg is None:
        pref_cfg = FlexiblePreferenceConfig()
    ns, nc, nfd = calib.n_sectors, calib.n_countries, calib.n_final_demand
    theta_sec, _, _ = _get_armington_weights(calib)
    budget = np.asarray(budget, dtype=float).reshape((1, nfd, nc))
    P_C = compute_armington_purchaser_prices(p, tau_fd, calib, sigma_trade=pref_cfg.sigma_trade)
    shares = theta_sec[:, 0, :, :]
    Q = np.sum(shares * P_C, axis=0, keepdims=True)
    active = np.sum(shares, axis=0, keepdims=True) > 0
    Q = np.where(active, Q, 1.0)
    c_sec = shares * budget / Q
    if np.any(_resolve_subsistence_shares(calib, pref_cfg) > 0.0):
        c_hh = _les_demand(budget[:, 0:1, :], P_C[:, 0, :][np.newaxis], calib, pref_cfg, warn=False)
        c_sec[:, 0, :] = c_hh[0]
    xc = compute_armington_final_demands(c_sec, P_C, p, tau_fd, calib, sigma_trade=pref_cfg.sigma_trade)
    return xc, P_C, Q, c_sec


# =============================================================================
# Market structure: Atkeson-Burstein markups and Dixit-Stiglitz varieties
# =============================================================================

def compute_benchmark_market_shares(calib: TradeCalibrationResult) -> np.ndarray:
    """Benchmark destination market shares ``s0[j, i, n]``, shape ``(ns, nc, nc)``.

    Share of origin ``i`` in the total absorption (intermediate plus final) of
    sector ``j`` in destination ``n``; shares sum to one over origins in every
    active market. Cached on the calibration.
    """
    cached = getattr(calib, "_benchmark_market_shares_cache", None)
    if cached is not None:
        return cached
    ns, nc, nfd = calib.n_sectors, calib.n_countries, calib.n_final_demand
    M = ns * nc
    data = getattr(calib, "data_calibra", None)
    if data is not None and data.shape[0] >= M and data.shape[1] >= M + nfd * nc:
        x_4d = data[:M, :M].reshape(nc, ns, nc, ns).transpose(1, 0, 3, 2)
        xfd_4d = data[:M, M:M + nfd * nc].reshape(nc, ns, nc, nfd).transpose(1, 0, 3, 2)
        flow_tot = np.sum(x_4d, axis=2) + np.sum(xfd_4d, axis=2)
    else:
        ytot_arr = np.asarray(calib.ytot, dtype=float).reshape(1, ns, nc)
        flow_interm = np.sum(np.asarray(calib.a, dtype=float) * ytot_arr, axis=1).reshape(nc, ns, nc).transpose(1, 0, 2)
        flow_fd = np.sum(np.asarray(calib.afd, dtype=float), axis=1).reshape(nc, ns, nc).transpose(1, 0, 2)
        flow_tot = flow_interm + flow_fd
    abs_tot = np.sum(flow_tot, axis=1, keepdims=True)
    s_0 = np.divide(flow_tot, abs_tot, out=np.zeros_like(flow_tot), where=(abs_tot > 0))
    try:
        object.__setattr__(calib, "_benchmark_market_shares_cache", s_0)
    except Exception:
        pass
    return s_0


def _markup_formula(
    s: np.ndarray, sigma_j: Any, theta_j: Any, clamping_threshold: float,
    markup_min: float, markup_max: float, cournot_weights: Any,
) -> np.ndarray:
    s_eff = s * cournot_weights if cournot_weights is not None else s
    denom = sigma_j - 1.0 + (1.0 - sigma_j / theta_j) * s_eff
    return np.clip(sigma_j / np.maximum(denom, clamping_threshold), markup_min, markup_max)


def compute_benchmark_markups(
    calib: TradeCalibrationResult,
    market_cfg: FlexibleMarketStructureConfig | None = None,
) -> np.ndarray:
    """Benchmark Atkeson-Burstein markups ``mu0[j, i, n]``, shape ``(ns, nc, nc)``.

    ``mu0 = clip(sigma_j / max(sigma_j - 1 + (1 - sigma_j/theta_j) s0, threshold), markup_min, markup_max)``.
    """
    if market_cfg is None:
        market_cfg = FlexibleMarketStructureConfig()
    return _markup_formula(
        compute_benchmark_market_shares(calib), market_cfg.sigma_j, market_cfg.theta_j,
        market_cfg.clamping_threshold, market_cfg.markup_min, market_cfg.markup_max,
        market_cfg.cournot_weights,
    )


def compute_atkeson_burstein_markups(
    s_ni: np.ndarray,
    c_i: np.ndarray | None = None,
    market_cfg: FlexibleMarketStructureConfig | None = None,
    sigma_j: float = 6.0,
    theta_j: float = 2.0,
    mu_0: np.ndarray | float | None = None,
    s_0: np.ndarray | float | None = None,
    s0: np.ndarray | float | None = None,
    cournot_weights: Any = None,
    clamping_threshold: float = 1e-4,
    markup_min: float = 1.0,
    markup_max: float = 5.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Atkeson-Burstein Cournot markups and, given marginal costs, seller prices.

    ``mu = clip(sigma_j / max(sigma_j - 1 + (1 - sigma_j/theta_j) s_eff, threshold), markup_min, markup_max)``
    with ``s_eff = s_ni * cournot_weights``. With ``c_i`` the price is relative
    to the benchmark markup, ``p = (mu / mu_0) c_i``, when ``mu_0`` (or the
    benchmark share ``s_0``/``s0``) is given, and ``p = mu c_i`` otherwise.
    ``market_cfg``, when given, supplies the elasticities, bounds, threshold
    and (unless passed) the Cournot weights.

    Returns
    -------
    mu_ni : np.ndarray
        Markups.
    p_ni : np.ndarray or None
        Seller prices (None without ``c_i``).
    """
    if s_0 is None and s0 is not None:
        s_0 = s0
    if market_cfg is not None:
        sigma_j, theta_j = market_cfg.sigma_j, market_cfg.theta_j
        markup_min, markup_max = market_cfg.markup_min, market_cfg.markup_max
        clamping_threshold = market_cfg.clamping_threshold
        if cournot_weights is None:
            cournot_weights = market_cfg.cournot_weights
    mu_ni = _markup_formula(s_ni, sigma_j, theta_j, clamping_threshold, markup_min, markup_max, cournot_weights)
    if c_i is None:
        return mu_ni, None
    if mu_0 is None and s_0 is not None:
        mu_0 = _markup_formula(np.asarray(s_0, dtype=float), sigma_j, theta_j, clamping_threshold,
                               markup_min, markup_max, cournot_weights)
    if mu_0 is not None:
        return mu_ni, (mu_ni / np.maximum(np.asarray(mu_0, dtype=float), 1e-12)) * c_i
    return mu_ni, mu_ni * c_i


def compute_dixit_stiglitz_varieties(
    pi_op: np.ndarray | float,
    w: np.ndarray | float = 1.0,
    r: np.ndarray | float = 1.0,
    fl: float = 0.0,
    fk: float = 0.0,
    market_cfg: FlexibleMarketStructureConfig | None = None,
) -> np.ndarray | float:
    """Number of varieties from free entry, ``N = max(pi_op / (w fl + r fk), 0)``.

    Without fixed costs (``fl`` and ``fk`` both zero) the variety count stays
    at 1. ``market_cfg``, when given, supplies ``fl`` and ``fk``.
    """
    if market_cfg is not None:
        fl, fk = market_cfg.fl, market_cfg.fk
    if fl <= 0.0 and fk <= 0.0:
        if isinstance(pi_op, (int, float, np.floating, np.integer)):
            return 1.0
        return np.ones_like(pi_op, dtype=float)
    fixed_cost = w * fl + r * fk
    if isinstance(pi_op, (int, float, np.floating, np.integer)) and isinstance(
            fixed_cost, (int, float, np.floating, np.integer)):
        if fixed_cost <= 0.0:
            return 1.0
        return max(float(pi_op) / float(fixed_cost), 0.0)
    fixed_cost_arr = np.asarray(fixed_cost, dtype=float)
    safe_fixed = np.where(fixed_cost_arr > 1e-12, fixed_cost_arr, 1e-12)
    N_arr = np.maximum(np.asarray(pi_op, dtype=float) / safe_fixed, 0.0)
    return np.where(fixed_cost_arr > 1e-12, N_arr, 1.0)


compute_condensed_varieties = compute_dixit_stiglitz_varieties


def compute_variety_price_scaling(
    p: np.ndarray | float | None = None,
    N: np.ndarray | float | None = None,
    N0: np.ndarray | float = 1.0,
    sigma_j: float = 6.0,
    market_cfg: FlexibleMarketStructureConfig | None = None,
) -> np.ndarray | float:
    """Price index with Dixit-Stiglitz variety effects, ``p (N / N0)**(1 / (1 - sigma_j))``.

    With one positional argument it is read as ``N`` (``p = 1``). With a
    ``market_cfg`` whose variety condensation is off, ``p`` is returned as is.
    """
    if market_cfg is not None:
        sigma_j = market_cfg.sigma_j
        if not market_cfg.variety_condensation:
            return 1.0 if p is None else p
    if N is None:
        if p is None:
            return 1.0
        N, p = p, 1.0
    if p is None:
        p = 1.0
    if sigma_j <= 1.0:
        raise ValueError(f"sigma_j must be strictly greater than 1.0, got {sigma_j}.")
    exponent = 1.0 / (1.0 - sigma_j)
    if isinstance(N, (int, float, np.floating, np.integer)) and isinstance(N0, (int, float, np.floating, np.integer)):
        scale = 1.0 if float(N) == float(N0) else max(float(N) / max(float(N0), 1e-12), 1e-12) ** exponent
        if isinstance(p, (int, float, np.floating, np.integer)):
            return float(p) if scale == 1.0 else float(p) * scale
        return np.asarray(p, dtype=float) * scale
    N_arr = np.asarray(N, dtype=float)
    N0_arr = np.asarray(N0, dtype=float)
    scale_arr = np.maximum(N_arr / np.maximum(N0_arr, 1e-12), 1e-12) ** exponent
    scale_arr = np.where(N_arr == N0_arr, 1.0, scale_arr)
    if isinstance(p, (int, float, np.floating, np.integer)) and float(p) == 1.0:
        return scale_arr
    return np.asarray(p, dtype=float) * scale_arr


def _market_shares_and_markups(
    p_vec: np.ndarray,
    calib: TradeCalibrationResult,
    market_cfg: FlexibleMarketStructureConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Market shares and markups at producer prices ``p_vec``, each ``(ns, nc, nc)``.

    ``s_ni = s0_ni p_i**(1 - sigma_j) / sum_k s0_nk p_k**(1 - sigma_j)``: the
    within-industry elasticity ``sigma_j`` of the markup model also governs
    how shares respond to prices.
    """
    ns, nc = calib.n_sectors, calib.n_countries
    s_0 = compute_benchmark_market_shares(calib)
    p_2d = np.asarray(p_vec, dtype=float).reshape((ns, nc), order="F")
    sigma = market_cfg.sigma_j
    sig_val = float(sigma) if _is_number(sigma) else float(np.mean(np.asarray(sigma, dtype=float)))
    shares_unnorm = s_0 * np.maximum(p_2d[:, :, None], 1e-12) ** (1.0 - sig_val)
    sum_shares = np.sum(shares_unnorm, axis=1, keepdims=True)
    s_ni = np.divide(shares_unnorm, sum_shares, out=s_0.copy(), where=(sum_shares > 1e-12))
    mu_0 = compute_benchmark_markups(calib, market_cfg)
    mu_ni, _ = compute_atkeson_burstein_markups(s_ni=s_ni, market_cfg=market_cfg, mu_0=mu_0)
    return s_ni, mu_ni, mu_0


def _relative_markups(
    p_vec: np.ndarray,
    calib: TradeCalibrationResult,
    market_cfg: FlexibleMarketStructureConfig,
) -> np.ndarray:
    """Relative markup ``mu_rel_i = mean_n(mu_ni / mu0_ni)`` per producing cell, shape ``(M,)``.

    The pricing wedge ``p_i (1 - tax_i) = mu_rel_i(p) c_y,i`` of the variable
    markup model; ``mu_rel = 1`` at the benchmark.
    """
    _, mu_ni, mu_0 = _market_shares_and_markups(p_vec, calib, market_cfg)
    return np.mean(mu_ni / np.maximum(mu_0, 1e-12), axis=2).ravel(order="F")


# =============================================================================
# Convergence diagnostics
# =============================================================================

_REMEDIATION = {
    "Goods Market Clearing": (
        "Suggestion: review intermediate sourcing tariffs (tau), final-demand expenditure shares, "
        "or reduce the shock."
    ),
    "Zero-Profit Condition": (
        "Suggestion: review the value-added substitution elasticity (rho_va), factor prices, "
        "or intermediate cost shares."
    ),
    "Labor Market Clearing": (
        "Suggestion: examine the labour endowment calibration (l_endow) or the labour share (1 - alpha)."
    ),
    "Capital Market Clearing": (
        "Suggestion: examine the capital endowment calibration (k_endow) or the capital share (alpha)."
    ),
    "Trade Balance": (
        "Suggestion: check the foreign-saving closure (foreign_saving_units) and the baseline "
        "foreign balances."
    ),
    "Fiscal Budget Consistency": (
        "Suggestion: verify tariff revenue and the lump-sum government budget closure."
    ),
}


def compute_convergence_diagnostics(
    residual: np.ndarray | Sequence[float] | None,
    calib: Any = None,
    layout: str = "codebase",
    n_top: int = 3,
) -> dict[str, Any]:
    """The worst equations of a residual vector, named by market, country and sector.

    Parameters
    ----------
    residual : array-like or None
        Residual vector of length ``2*ns*nc + 4*nc - 1``.
    calib : TradeCalibrationResult
        Calibration supplying ``n_countries``, ``n_sectors`` and the codes.
        Required (``ValueError`` without it) whenever ``residual`` is not empty.
    layout : {"codebase", "spec"}, default "codebase"
        Equation ordering. ``"codebase"`` is the order of
        :func:`puremacro.trade.compute_equilibrium_residuals`: goods (the first
        replaced by the numeraire), zero profit, labour, capital, foreign
        balances, government budgets. ``"spec"`` swaps the first two blocks and
        the labour and capital blocks.
    n_top : int, default 3
        Number of equations to report.

    Returns
    -------
    dict
        ``top_equations`` (list of dicts with ``rank``, ``index``, ``block``,
        ``equation``, ``country``, ``sector``, ``residual``, ``abs_residual``),
        ``max_residual`` / ``worst_residual``, ``l1_norm`` and ``remediation``.
    """
    if layout not in ("codebase", "spec"):
        raise ValueError(f"layout must be 'codebase' or 'spec', got {layout!r}")
    if residual is None:
        return {"top_equations": [], "max_residual": 0.0, "worst_residual": 0.0, "l1_norm": 0.0,
                "remediation": "No residual vector provided."}
    res_arr = np.asarray(residual, dtype=float).ravel()
    if len(res_arr) == 0:
        return {"top_equations": [], "max_residual": 0.0, "worst_residual": 0.0, "l1_norm": 0.0,
                "remediation": "Empty residual vector provided."}
    if calib is None or getattr(calib, "n_countries", None) is None or getattr(calib, "n_sectors", None) is None:
        raise ValueError(
            "compute_convergence_diagnostics needs the calibration (calib=...) to map equations "
            "to countries and sectors"
        )
    nc, ns = int(calib.n_countries), int(calib.n_sectors)
    M = ns * nc
    if len(res_arr) != 2 * M + 4 * nc - 1:
        raise ValueError(
            f"residual has length {len(res_arr)}, expected {2 * M + 4 * nc - 1} for "
            f"{nc} countries and {ns} sectors"
        )
    raw_c = getattr(calib, "country_codes", None)
    c_codes = [str(x) for x in raw_c] if raw_c and len(raw_c) == nc else [f"C{i:02d}" for i in range(nc)]
    raw_s = getattr(calib, "sector_codes", None)
    s_codes = [str(x) for x in raw_s] if raw_s and len(raw_s) == ns else [f"S{j:02d}" for j in range(ns)]

    if layout == "codebase":
        blocks = [("Goods Market Clearing", M, True), ("Zero-Profit Condition", M, True),
                  ("Labor Market Clearing", nc, False), ("Capital Market Clearing", nc, False),
                  ("Trade Balance", nc - 1, False), ("Fiscal Budget Consistency", nc, False)]
    else:
        blocks = [("Zero-Profit Condition", M, True), ("Goods Market Clearing", M, True),
                  ("Capital Market Clearing", nc, False), ("Labor Market Clearing", nc, False),
                  ("Trade Balance", nc - 1, False), ("Fiscal Budget Consistency", nc, False)]

    def _locate(idx: int) -> tuple[str, str, str | None, str]:
        start = 0
        for name, size, by_cell in blocks:
            if idx < start + size:
                k = idx - start
                if by_cell:
                    country, sector = c_codes[k // ns], s_codes[k % ns]
                    if name == "Goods Market Clearing" and k == 0:
                        return name, country, sector, f"Numeraire (first producer price): Sector {sector}, Country {country}"
                    return name, country, sector, f"{name}: Sector {sector}, Country {country}"
                return name, c_codes[k], None, f"{name}: Country {c_codes[k]}"
            start += size
        raise IndexError(idx)  # pragma: no cover - lengths are checked above

    abs_res = np.abs(res_arr)
    rank_keys = np.where(np.isnan(abs_res), np.inf, abs_res)
    order = np.argsort(-rank_keys, kind="stable")
    top_list: list[dict[str, Any]] = []
    for rank, idx in enumerate(order[:min(max(int(n_top), 0), len(res_arr))], start=1):
        idx = int(idx)
        block, country, sector, eq_str = _locate(idx)
        raw_val = float(res_arr[idx])
        top_list.append({
            "rank": int(rank), "index": idx, "block": block, "equation": eq_str,
            "country": country, "sector": sector, "residual": raw_val,
            "abs_residual": float(abs_res[idx]) if not np.isnan(raw_val) else float("nan"),
        })
    top_eq = top_list[0] if top_list else None
    if top_eq is None:
        remediation = "No residual equations evaluated."
    elif not np.isfinite(top_eq["abs_residual"]):
        remediation = (
            f"Severe numerical divergence detected in {top_eq['block']}. "
            "Check for division by zero, non-positive factor prices, or degenerate substitution bounds."
        )
    else:
        remediation = f"Largest residual is in {top_eq['equation']}. " + _REMEDIATION[top_eq["block"]]
    finite_vals = abs_res[np.isfinite(abs_res)]
    max_res = float(np.max(finite_vals)) if len(finite_vals) else 0.0
    return {
        "top_equations": top_list,
        "max_residual": max_res,
        "worst_residual": max_res,
        "l1_norm": float(np.sum(finite_vals)) if len(finite_vals) else 0.0,
        "remediation": remediation,
    }


# =============================================================================
# Equilibrium blocks on the consistent accounting
# =============================================================================

def _check_schedules(ta: np.ndarray, tf: np.ndarray, calib: TradeCalibrationResult) -> None:
    ns, nc, nfd = calib.n_sectors, calib.n_countries, calib.n_final_demand
    n = ns * nc
    for schedule, shape in ((ta, (n, ns, nc)), (tf, (n, nfd, nc))):
        if schedule.shape != shape or not np.isfinite(schedule).all() or np.any(schedule <= 0):
            raise ValueError("Tariff multipliers must have the expected shape and be finite and positive")
        for c in range(nc):
            if not np.allclose(schedule[c * ns:(c + 1) * ns, :, c], 1., rtol=0, atol=1e-14):
                raise ValueError("Import tariffs must be one on domestic transactions")


def _unit_costs(
    r: np.ndarray, w: np.ndarray, P_M: np.ndarray, y: np.ndarray,
    calib: TradeCalibrationResult, tech: FlexibleTechnologyConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(c_va, c_y index, c_y level)`` with the capacity penalty evaluated at ``y``."""
    c_va, c_y = compute_nested_ces_costs(r, w, P_M, calib, tech, ytot=y)
    return c_va, c_y, _outer_benchmark(calib)[0] * c_y


def _flexible_blocks(
    x_full: np.ndarray,
    calib: TradeCalibrationResult,
    config: FlexibleTradeModelConfig,
    ta: np.ndarray,
    tf: np.ndarray,
    foreign_saving_units: str = "numeraire",
) -> dict[str, Any]:
    """Blocks of the flexible model at the full state ``x_full``.

    ``ta`` and ``tf`` are the intermediate and final-demand tariff
    multipliers, shapes ``(ns*nc, ns, nc)`` and ``(ns*nc, nfd, nc)``. The
    returned dictionary has the keys of
    :func:`puremacro.trade._accounting.evaluate` (so
    :func:`puremacro.trade._accounting.postprocess` applies) plus
    ``markup_profits``, ``household_budget``, ``sector_prices`` and
    ``sector_quantities``. At the default configuration the blocks equal those
    of ``evaluate``.
    """
    from puremacro.trade._accounting import FOREIGN_SAVING_UNITS, _parameters

    if foreign_saving_units not in FOREIGN_SAVING_UNITS:
        raise ValueError("foreign_saving_units must be 'numeraire' or 'world_income'")
    nc, ns, nfd = calib.n_countries, calib.n_sectors, calib.n_final_demand
    n = nc * ns
    fd_tax, basic0, inv = _parameters(calib)
    ta = np.asarray(ta, dtype=float)
    tf = np.asarray(tf, dtype=float)
    _check_schedules(ta, tf, calib)
    tech, pref, market = config.technology, config.preference, config.market_structure

    v = unpack_equilibrium_vector(x_full, ns=ns, nc=nc, nfd=nfd)
    p, y, r, w, T, B = v.p, v.y, v.r, v.w, v.T, v.invforT
    pv = p.ravel(order="F")
    P_M, _, _ = compute_intermediate_composite_price(p, ta, calib, sigma=tech.sigma_inter)
    c_va, c_y, c_y_level = _unit_costs(r, w, P_M, y, calib, tech)
    one_minus_tax = 1.0 - calib.tax
    if market.variable_markups:
        mu_rel = _relative_markups(pv, calib, market).reshape((1, ns, nc), order="F")
        pp = mu_rel * c_y_level / one_minus_tax
        profits = np.sum((p * one_minus_tax - c_y_level) * y, axis=1).reshape(1, 1, nc)
    else:
        pp = c_y_level / one_minus_tax
        profits = np.zeros((1, 1, nc))
    labor, capital, Z = compute_nested_factor_demands(y, r, w, P_M, c_va, c_y, p, ta, calib, tech)

    income = w * calib.l_endow.reshape(1, 1, nc) + r * calib.k_endow.reshape(1, 1, nc) + T + profits
    expenditure = calib.theta * income
    expenditure[:, inv, :] -= B
    budget = expenditure * (1.0 - fd_tax)
    F3, P_C, Q, c_sec = compute_multi_category_final_demands(budget, p, tf, calib, pref)
    Q = np.where(basic0 > 0, Q, 1.)
    P = Q / (1 - fd_tax)
    quantity = expenditure / P
    final_tax = fd_tax * expenditure
    production_tax = calib.tax * p * y
    duties_i = (ta - 1) * pv[:, None, None] * Z
    duties_f = (tf - 1) * pv[:, None, None] * F3
    tariffs = duties_i.sum(axis=(0, 1)) + duties_f.sum(axis=(0, 1))
    government = production_tax.sum(axis=1).ravel() + final_tax.sum(axis=1).ravel() + tariffs
    zm = Z.reshape(n, n, order="F")
    fm = F3.reshape(n, nfd * nc, order="F")
    producer_values = np.hstack([zm, fm]) * pv[:, None]
    trade = producer_values[:, :n].reshape(nc, ns, nc, ns).sum(axis=(1, 3))
    trade += producer_values[:, n:].reshape(nc, ns, nc, nfd).sum(axis=(1, 3))
    np.fill_diagonal(trade, 0.)
    net_exports = trade.sum(1) - trade.sum(0)
    goods = y.ravel(order="F") - zm.sum(1) - fm.sum(1)
    prices = (pp - p).ravel(order="F")
    labor_gap = calib.l_endow.ravel() - labor.sum(1).ravel()
    capital_gap = calib.k_endow.ravel() - capital.sum(1).ravel()
    budget_gap = T.ravel() - government
    if foreign_saving_units == "numeraire":
        balances = v.XN - np.asarray(calib.invforT).ravel()[:nc - 1]
    else:
        base_world = float(np.sum(calib.l_endow) + np.sum(calib.k_endow))
        world = float(np.sum(w.ravel() * calib.l_endow.ravel() + r.ravel() * calib.k_endow.ravel()))
        balances = v.XN - np.asarray(calib.invforT).ravel()[:nc - 1] * (world / base_world)
    solver_goods = goods.copy()
    solver_goods[0] = pv[0] - 1.
    residuals = np.concatenate([solver_goods, prices, labor_gap, capital_gap, balances, budget_gap])
    physical = np.concatenate([goods, prices, labor_gap, capital_gap, net_exports - B.ravel(), budget_gap])
    return dict(p=p, pp=pp, y=y, r=r, w=w, T=T, B=B, XN=v.XN, Z=zm, F=fm,
                Q=Q, P=P, c=quantity, income=income, expenditure=expenditure,
                final_tax=final_tax, production_tax=production_tax, labor=labor, capital=capital,
                duties_i=duties_i, duties_f=duties_f, tariffs=tariffs, trade=trade,
                producer_values=producer_values, ta=ta, tf=tf, basic0=basic0, fd_tax=fd_tax,
                residuals=residuals, physical_residuals=physical, goods_gap=goods,
                net_exports_gap=net_exports - B.ravel(), government=government,
                foreign_saving_units=foreign_saving_units,
                markup_profits=profits.ravel(), household_budget=budget[0, 0, :].copy(),
                sector_prices=P_C, sector_quantities=c_sec)


def _foreign_saving(calib: TradeCalibrationResult, r: np.ndarray, w: np.ndarray, units: str) -> np.ndarray:
    """Foreign saving ``B_c`` (length nc) at factor returns ``r``, ``w``."""
    b = np.asarray(calib.invforT, dtype=float).ravel()
    if units == "numeraire":
        return b.copy()
    base_world = float(np.sum(calib.l_endow) + np.sum(calib.k_endow))
    world = float(np.sum(np.ravel(w) * calib.l_endow.ravel() + np.ravel(r) * calib.k_endow.ravel()))
    return b * world / base_world


# Inner loop controls. Jacobian evaluations honour max_inner_iter (<= 10);
# step acceptance and the reported residual iterate to tight tolerances.
_INNER_TOL = 1e-8
_CERT_INNER_TOL = 1e-13
_CERT_INNER_MAX_ITER = 5000
_QUANTITY_FP_TOL = 1e-10
_CERT_QUANTITY_FP_TOL = 1e-13
_QUANTITY_FP_MAX_ITER = 50
_CERT_QUANTITY_FP_MAX_ITER = 500


class _CondensedModel:
    """The flexible model condensed to the macro unknowns ``[log r; log w; T]``.

    Given factor returns and transfers, producer prices solve the zero-profit
    conditions (a linear Leontief system at fixed coefficients, otherwise a
    damped fixed point) and gross outputs solve goods-market clearing (linear
    given prices; with a capacity barrier or markups, prices and income depend
    on output and a fixed point on ``y`` closes the loop). The macro residuals
    are ``[p_0 - 1, labour gaps of countries 2..nc, capital gaps, budget gaps]``:
    by Walras' law the first country's labour market clears once every goods
    market, zero-profit condition, budget and the world foreign-saving identity
    hold, so a root of this system solves the full model (audited on the full
    blocks).
    """

    def __init__(self, calib: TradeCalibrationResult, config: FlexibleTradeModelConfig,
                 ta: np.ndarray, tf: np.ndarray, foreign_saving_units: str) -> None:
        self.calib, self.config, self.ta, self.tf = calib, config, ta, tf
        self.units = foreign_saving_units
        ns, nc = calib.n_sectors, calib.n_countries
        self.ns, self.nc, self.M = ns, nc, ns * nc
        tech, market = config.technology, config.market_structure
        self.tax_flat = calib.tax.flatten(order="F")
        self.one_minus_tax = 1.0 - self.tax_flat
        a_eff = (calib.a * ta).reshape((self.M, self.M), order="F")
        self.lu_P = la.lu_factor(np.eye(self.M) - a_eff.T / self.one_minus_tax[:, None])
        self.a_2d = calib.a.reshape((self.M, self.M), order="F")
        self.lu_Y = la.lu_factor(np.eye(self.M) - self.a_2d)
        self.use_markups = bool(market.variable_markups)
        self.use_capacity = tech.capacity_margins is not None
        self.linear_prices = tech.sigma_y < 1e-6 and tech.sigma_inter < 1e-6 and not self.use_markups
        self.leontief_intermediates = tech.sigma_y < 1e-6 and tech.sigma_inter < 1e-6
        self.needs_y_loop = self.use_markups or self.use_capacity
        self.k_inner = min(max(1, int(config.max_inner_iter)), 10)
        self.y_bench = np.asarray(calib.ytot, dtype=float).flatten(order="F")
        from puremacro.trade._accounting import _parameters
        self.fd_tax, self.basic0, self.inv = _parameters(calib)

    def split(self, xm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nc = self.nc
        r = np.exp(xm[:nc]).reshape((1, 1, nc))
        w = np.exp(xm[nc:2 * nc]).reshape((1, 1, nc))
        return r, w, xm[2 * nc:3 * nc].reshape((1, 1, nc))

    def _prices(self, r, w, y3, p_init, k_max, tol):
        """Inner zero-profit prices; returns ``(p, iterations, last update)``."""
        calib, tech = self.calib, self.config.technology
        c_va = compute_inner_ces_cost(r, w, calib, tech)
        if self.use_capacity:
            c_va = c_va * (1.0 + compute_capacity_penalty(y3, calib, tech))
        v_P = c_va.flatten(order="F") / self.one_minus_tax
        p_lin = la.lu_solve(self.lu_P, v_P)
        if self.linear_prices:
            return p_lin, 0, 0.0
        p_curr = p_init.copy() if p_init is not None else p_lin
        diff, iters = float("inf"), 0
        for _ in range(k_max):
            iters += 1
            P_M, _, _ = compute_intermediate_composite_price(p_curr, self.ta, calib, sigma=tech.sigma_inter)
            c_y_level = compute_outer_ces_cost(c_va, P_M, calib, tech, normalized=False).flatten(order="F")
            if self.use_markups:
                c_y_level = _relative_markups(p_curr, calib, self.config.market_structure) * c_y_level
            p_target = c_y_level / self.one_minus_tax
            diff = float(np.max(np.abs(p_target - p_curr)))
            p_curr = 0.2 * p_curr + 0.8 * p_target
            if diff <= tol:
                break
        return p_curr, iters, diff

    def state(self, xm: np.ndarray, p_warm=None, y_warm=None, certify: bool = False):
        """Prices and outputs at the macro point; returns ``(p, y, extras)``."""
        calib, config = self.calib, self.config
        tech = config.technology
        ns, nc, M = self.ns, self.nc, self.M
        r, w, T = self.split(xm)
        B = _foreign_saving(calib, r, w, self.units)
        k_max = _CERT_INNER_MAX_ITER if certify else self.k_inner
        tol_inner = _CERT_INNER_TOL if certify else _INNER_TOL
        if self.needs_y_loop:
            n_loop = _CERT_QUANTITY_FP_MAX_ITER if certify else _QUANTITY_FP_MAX_ITER
            fp_tol = _CERT_QUANTITY_FP_TOL if certify else _QUANTITY_FP_TOL
        else:
            n_loop, fp_tol = 1, 0.0
        y_guess = np.asarray(y_warm, dtype=float).copy() if (y_warm is not None and self.needs_y_loop) \
            else self.y_bench.copy()
        p_guess = p_warm
        extras = {"inner_converged": True, "quantity_fixed_point_converged": not self.needs_y_loop,
                  "inner_iterations": 0}
        base_income = w * calib.l_endow.reshape(1, 1, nc) + r * calib.k_endow.reshape(1, 1, nc) + T
        y_new = y_guess
        p_vec = None
        for _ in range(n_loop):
            y3 = np.maximum(y_guess, 1e-300).reshape((1, ns, nc), order="F")
            p_vec, it_in, inner_diff = self._prices(r, w, y3, p_guess, k_max, tol_inner)
            extras["inner_iterations"] += it_in
            if certify and inner_diff > tol_inner:
                extras["inner_converged"] = False
            p3 = p_vec.reshape((1, ns, nc), order="F")
            P_M, _, _ = compute_intermediate_composite_price(p3, self.ta, calib, sigma=tech.sigma_inter)
            c_va, c_y, c_y_level = _unit_costs(r, w, P_M, y3, calib, tech)
            income = base_income
            if self.use_markups:
                margin = (p3 * (1.0 - calib.tax) - c_y_level) * y3
                income = base_income + np.sum(margin, axis=1).reshape(1, 1, nc)
            expenditure = calib.theta * income
            expenditure[:, self.inv, :] -= B
            xc, _, _, _ = compute_multi_category_final_demands(
                expenditure * (1.0 - self.fd_tax), p3, self.tf, calib, config.preference)
            d = np.sum(xc.reshape((M, -1), order="F"), axis=1)
            if self.leontief_intermediates:
                y_new = la.lu_solve(self.lu_Y, d)
            else:
                _, _, x_unit = compute_nested_factor_demands(
                    np.ones((1, ns, nc)), r, w, P_M, c_va, c_y, p3, self.ta, calib, tech)
                y_new = np.linalg.solve(np.eye(M) - x_unit.reshape((M, M), order="F"), d)
            if not self.needs_y_loop:
                break
            change = float(np.max(np.abs(y_new - y_guess))) / max(1.0, float(np.max(np.abs(y_new))))
            y_guess, p_guess = y_new, p_vec
            if change <= fp_tol:
                extras["quantity_fixed_point_converged"] = True
                break
        return p_vec, y_new, extras

    def full_state(self, xm: np.ndarray, p_vec: np.ndarray, y_vec: np.ndarray) -> np.ndarray:
        nc = self.nc
        r, w, T = self.split(xm)
        B = _foreign_saving(self.calib, r, w, self.units)
        return np.concatenate([np.log(np.maximum(p_vec, 1e-300)), np.log(np.maximum(y_vec, 1e-300)),
                               xm[:nc], xm[nc:2 * nc], xm[2 * nc:3 * nc], B[:nc - 1]])

    def evaluate(self, xm, p_warm=None, y_warm=None, certify=False):
        """Macro residuals, the full blocks and the reconstructed state."""
        p_vec, y_vec, extras = self.state(xm, p_warm, y_warm, certify)
        x_full = self.full_state(xm, p_vec, y_vec)
        blocks = _flexible_blocks(x_full, self.calib, self.config, self.ta, self.tf, self.units)
        nc, M = self.nc, self.M
        res = blocks["residuals"]
        labor = res[2 * M:2 * M + nc]
        capital = res[2 * M + nc:2 * M + 2 * nc]
        budget = res[2 * M + 3 * nc - 1:]
        f_m = np.concatenate([[p_vec[0] - 1.0], labor[1:], capital, budget])
        # Negative output is infeasible: keep it visible in the audit.
        if np.any(y_vec < 0):
            extras["negative_output"] = True
        return f_m, blocks, x_full, p_vec, y_vec, extras


def _macro_from_x0(model: _CondensedModel, x0: np.ndarray | None) -> np.ndarray:
    from puremacro.trade.solver import build_initial_guess

    calib = model.calib
    nc, M = model.nc, model.M
    x = build_initial_guess(calib) if x0 is None else np.asarray(x0, dtype=float).ravel()
    if len(x) == 3 * nc:
        return x.copy()
    if len(x) != 2 * M + 4 * nc - 1:
        raise ValueError(
            f"Invalid initial guess length {len(x)}: expected {2 * M + 4 * nc - 1} (full state) "
            f"or {3 * nc} (macro vector [log r; log w; T])."
        )
    return x[2 * M:2 * M + 3 * nc].copy()


def _audit(blocks: dict[str, Any], extras: dict[str, Any]) -> float:
    """Largest absolute residual over the solved and physical equations."""
    vals = np.concatenate([blocks["residuals"], blocks["physical_residuals"]])
    if not np.isfinite(vals).all():
        return float("inf")
    return float(np.max(np.abs(vals)))


def _condensed_solve(
    calib: TradeCalibrationResult,
    config: FlexibleTradeModelConfig,
    ta: np.ndarray,
    tf: np.ndarray,
    *,
    x0: np.ndarray | None = None,
    tol: float = 2.5e-3,
    max_iter: int = 50,
    eps_fd: float = 1e-4,
    foreign_saving_units: str = "numeraire",
) -> tuple[np.ndarray, bool, int, dict[str, Any], dict[str, Any]]:
    """Condensed Newton, with a tariff homotopy when a direct solve stalls.

    ``max_iter`` bounds the Newton iterations of the whole call, homotopy
    included, and the returned iteration count is the total spent. Newton
    from ``x0`` stops early when its line search finds no decrease (a stall).
    If it does not converge, the iterations it left unused go to a tariff
    homotopy: the schedule is reached in 4 (then 16) equal steps of the
    multipliers from the free-trade schedule, each step warm-started from the
    previous solution and given what remains of the budget. The first path
    that converges at every step is returned (``meta["tariff_homotopy_steps"]``);
    otherwise the direct attempt is reported, with
    ``meta["tariff_homotopy_steps"] = 0`` when a homotopy was tried.
    """
    direct = _condensed_newton(
        calib, config, ta, tf, x0=x0, max_iter=max_iter,
        tol=tol, eps_fd=eps_fd, foreign_saving_units=foreign_saving_units,
    )
    if direct[1] or max_iter == 0 or (np.all(ta == 1.0) and np.all(tf == 1.0)):
        return direct
    spent = direct[2]
    tried = False
    for n_steps in (4, 16):
        if max_iter - spent < n_steps:
            break
        tried = True
        x_path, result = x0, None
        for k in range(1, n_steps + 1):
            lam = k / n_steps
            result = _condensed_newton(
                calib, config, 1.0 + lam * (ta - 1.0), 1.0 + lam * (tf - 1.0),
                x0=x_path, max_iter=max_iter - spent,
                tol=tol, eps_fd=eps_fd, foreign_saving_units=foreign_saving_units,
            )
            spent += result[2]
            if not result[1] or (k < n_steps and spent >= max_iter):
                result = None
                break
            x_path = result[0]
        if result is not None:
            x_sol, conv, _, blocks, meta = result
            meta["tariff_homotopy_steps"] = n_steps
            return x_sol, conv, spent, blocks, meta
    x_sol, conv, _, blocks, meta = direct
    if tried:
        meta["tariff_homotopy_steps"] = 0
    return x_sol, conv, spent, blocks, meta


def _condensed_newton(
    calib: TradeCalibrationResult,
    config: FlexibleTradeModelConfig,
    ta: np.ndarray,
    tf: np.ndarray,
    *,
    x0: np.ndarray | None = None,
    tol: float = 2.5e-3,
    max_iter: int = 50,
    eps_fd: float = 1e-4,
    foreign_saving_units: str = "numeraire",
) -> tuple[np.ndarray, bool, int, dict[str, Any], dict[str, Any]]:
    """Damped Newton on the condensed macro system.

    Returns ``(x_sol, converged, iterations, blocks, meta)``. The reported
    state is certified: the inner price and quantity loops are iterated to
    1e-13 from a cold start, and convergence requires every solved and
    physical residual of the full blocks to be at most ``tol``. ``max_iter=0``
    performs that evaluation at the starting point only. Newton stops early,
    unconverged, when the backtracking line search finds no decrease of the
    scaled residual norm (``meta["newton_stalled"]``); ``iterations`` is then
    the number of iterations taken, not ``max_iter``.
    """
    t_start = time.perf_counter()
    model = _CondensedModel(calib, config, ta, tf, foreign_saving_units)
    nc = model.nc
    xm = _macro_from_x0(model, x0)

    def certified(xm_c):
        return model.evaluate(xm_c, certify=True)

    def finish(xm_f, ok, iters, ev, stalled=False):
        f_m, blocks, x_full, _, y_vec, extras = ev
        audit = _audit(blocks, extras)
        internal = bool(extras["inner_converged"] and extras["quantity_fixed_point_converged"])
        feasible = bool(np.all(y_vec > 0) and np.all(blocks["expenditure"] >= 0))
        conv = bool(ok and internal and feasible and audit <= tol)
        meta = {
            "inner_price_converged": bool(extras["inner_converged"]),
            "quantity_fixed_point_converged": bool(extras["quantity_fixed_point_converged"]),
            "residual_evaluation": "certified: cold-start inner loops to 1e-13",
            "audit_max_residual": audit,
            "solve_duration_seconds": time.perf_counter() - t_start,
            "newton_stalled": bool(stalled),
        }
        if not conv:
            diag = compute_convergence_diagnostics(blocks["residuals"], calib)
            notes = []
            if stalled:
                notes.append("Newton stalled: the line search found no decrease of the residual.")
            elif iters >= max_iter:
                notes.append("Increase solver max_iter or reduce the shock.")
            if not internal:
                notes.append("The certified inner price / quantity fixed point did not reach its tolerance "
                             "at the returned state.")
            if not feasible:
                notes.append("The returned state has negative output or final expenditure.")
            if notes:
                diag["remediation"] = " ".join([diag["remediation"], *notes])
            meta["convergence_diagnostic"] = diag
        return x_full, conv, iters, blocks, meta

    ev = certified(xm)
    if max_iter == 0:
        return finish(xm, False, 0, ev)
    if _audit(ev[1], ev[5]) <= tol:
        return finish(xm, True, 0, ev)

    value_scale = float(np.median(calib.l_endow.ravel() + calib.k_endow.ravel()))
    h = np.array([eps_fd if j < 2 * nc else eps_fd * max(abs(xm[j]), value_scale) for j in range(3 * nc)])
    f_m, p_sol, y_sol = ev[0], ev[3], ev[4]
    for it in range(max_iter):
        f_base = model.evaluate(xm, p_warm=p_sol, y_warm=y_sol)[0] if not model.linear_prices \
            or model.needs_y_loop else f_m
        J = np.empty((3 * nc, 3 * nc))
        for j in range(3 * nc):
            xp = xm.copy()
            xp[j] += h[j]
            J[:, j] = (model.evaluate(xp, p_warm=p_sol, y_warm=y_sol)[0] - f_base) / h[j]
        c_norm = np.linalg.norm(J, axis=0)
        c_norm[c_norm == 0] = 1.0
        D_R = 1.0 / c_norm
        J_s = J * D_R[None, :]
        r_norm = np.linalg.norm(J_s, axis=1)
        r_norm[r_norm == 0] = 1.0
        D_L = 1.0 / r_norm
        try:
            step = D_R * la.solve(D_L[:, None] * J_s, -D_L * f_m)
        except (la.LinAlgError, ValueError):
            step = D_R * la.lstsq(D_L[:, None] * J_s, -D_L * f_m)[0]
        if not np.all(np.isfinite(step)):
            return finish(xm, False, it + 1, certified(xm), stalled=True)
        big = float(np.max(np.abs(step[:2 * nc])))
        if big > 0.3:
            step *= 0.3 / big
        norm_0 = float(np.linalg.norm(D_L * f_m))
        alpha_step = 1.0
        ev_t = None
        decreased = False
        for _ in range(15):
            xm_t = xm + alpha_step * step
            xm_t[:2 * nc] = np.clip(xm_t[:2 * nc], -5.0, 5.0)
            ev_t = model.evaluate(xm_t, p_warm=p_sol, y_warm=y_sol, certify=True)
            if np.all(np.isfinite(ev_t[0])) and float(np.linalg.norm(D_L * ev_t[0])) < norm_0:
                decreased = True
                break
            alpha_step *= 0.5
        if not decreased:
            return finish(xm, False, it + 1, certified(xm), stalled=True)
        xm = xm_t
        assert ev_t is not None
        f_m, p_sol, y_sol = ev_t[0], ev_t[3], ev_t[4]
        if np.all(np.isfinite(f_m)) and float(np.max(np.abs(f_m))) <= tol:
            ev_c = certified(xm)
            if _audit(ev_c[1], ev_c[5]) <= tol:
                return finish(xm, True, it + 1, ev_c)
    return finish(xm, False, max_iter, certified(xm))


# =============================================================================
# Result container
# =============================================================================

def _blocks_of_result(result: Any, calib: TradeCalibrationResult,
                      config: FlexibleTradeModelConfig | None) -> dict[str, Any]:
    """Model blocks at a result's own state, tariffs and closure."""
    from puremacro.trade.solver import _resolve_tariffs

    meta = result.metadata if isinstance(getattr(result, "metadata", None), dict) else {}
    ta = meta.get("intermediate_tariff_multipliers")
    tf = meta.get("final_tariff_multipliers")
    if ta is None or tf is None:
        tariffs = meta.get("tariff_inputs") or {}
        ta, tf, _, _ = _resolve_tariffs(calib, tariffs.get("tau"), tariffs.get("tau_fd"),
                                        tariffs.get("tauf"), tariffs.get("tauf_fd"))
    return _flexible_blocks(np.asarray(result.x_sol, dtype=float).ravel(), calib,
                            config if config is not None else FlexibleTradeModelConfig(),
                            np.asarray(ta, dtype=float), np.asarray(tf, dtype=float),
                            meta.get("foreign_saving_units", "numeraire"))


@dataclass
class FlexibleTradeEquilibriumResult:
    """Solution of the flexible trade model.

    Attributes
    ----------
    x_sol : np.ndarray
        State vector ``[log p; log y; log r; log w; T; XN]`` of length
        ``2*ns*nc + 4*nc - 1``.
    converged : bool
        Whether every solved and physical residual is within the tolerance.
    iterations : int
        Outer iterations.
    residual_norm : float
        Largest absolute residual of the solved equations at ``x_sol``.
    residuals : np.ndarray, optional
        Residual vector in the order of
        :func:`puremacro.trade.compute_equilibrium_residuals`.
    metadata : dict
        Routing, configuration, tariffs, closure and the flows of the solved
        model (see :func:`solve_flexible_trade_equilibrium`).
    markups : pd.DataFrame
        Markups by origin, destination and sector.
    calib : TradeCalibrationResult, optional
        Calibration of the solve; the price, quantity and flow properties
        need it (``ValueError`` without it).
    config : FlexibleTradeModelConfig, optional
        Configuration of the solve (the default configuration when None).

    Notes
    -----
    ``cpi``, ``terms_of_trade``, ``exports``, ``imports``, ``gdp`` and
    ``gdp_fc`` are computed from the flows of the solved model, at its own
    tariffs and foreign-saving closure, with the consistent-accounting
    postprocessing (:func:`puremacro.trade._accounting.postprocess`). For an
    active configuration the CPI is the fixed-basket index of the category
    price indices at benchmark sector weights.
    """

    x_sol: np.ndarray
    converged: bool = True
    iterations: int = 0
    residual_norm: float = 0.0
    residuals: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    markups: pd.DataFrame = field(default_factory=pd.DataFrame)
    calib: Any = None
    config: FlexibleTradeModelConfig | None = None
    _factors_frame: pd.DataFrame | None = field(default=None, repr=False)
    _blocks_cache: dict[str, Any] | None = field(default=None, repr=False)
    _flows_cache: dict[str, Any] | None = field(default=None, repr=False)

    def _require_calib(self) -> TradeCalibrationResult:
        if self.calib is None:
            raise ValueError(
                "this FlexibleTradeEquilibriumResult has no calibration (calib=None), so its state "
                "vector cannot be unpacked; build it with calib=<TradeCalibrationResult>"
            )
        return self.calib

    def _get_unpacked(self) -> Any:
        calib = self._require_calib()
        return unpack_equilibrium_vector(self.x_sol, ns=calib.n_sectors, nc=calib.n_countries,
                                         nfd=calib.n_final_demand)

    def _blocks(self) -> dict[str, Any]:
        if self._blocks_cache is None:
            self._blocks_cache = _blocks_of_result(self, self._require_calib(), self.config)
        return self._blocks_cache

    def _flows(self) -> dict[str, Any]:
        if self._flows_cache is None:
            from puremacro.trade._accounting import postprocess
            self._flows_cache = postprocess(self._blocks(), self._require_calib())
        return self._flows_cache

    @property
    def p_sol(self) -> np.ndarray:
        """Producer prices, shape (1, ns, nc)."""
        return self._get_unpacked().p

    @property
    def y_sol(self) -> np.ndarray:
        """Gross outputs, shape (1, ns, nc)."""
        return self._get_unpacked().y

    @property
    def r_sol(self) -> np.ndarray:
        """Rental rates, shape (1, 1, nc)."""
        return self._get_unpacked().r

    @property
    def w_sol(self) -> np.ndarray:
        """Wages, shape (1, 1, nc)."""
        return self._get_unpacked().w

    @property
    def T_sol(self) -> np.ndarray:
        """Government receipts rebated lump sum, shape (1, 1, nc)."""
        return self._get_unpacked().T

    @property
    def XN_sol(self) -> np.ndarray:
        """Foreign balances of the first nc - 1 countries."""
        return self._get_unpacked().XN

    @property
    def cpi(self) -> np.ndarray:
        """Consumer price index (fixed benchmark basket), divided by the relative wage."""
        return self._flows()["cpi"]

    @property
    def terms_of_trade(self) -> np.ndarray:
        """Export unit value over import unit value, by country."""
        return self._flows()["terms_of_trade"]

    @property
    def exports(self) -> np.ndarray:
        """Gross exports at producer prices, by country."""
        return self._flows()["exports"]

    @property
    def imports(self) -> np.ndarray:
        """Gross imports at producer prices, by country."""
        return self._flows()["imports"]

    @property
    def gdp(self) -> np.ndarray:
        """GDP: factor income plus government receipts."""
        return self._flows()["gdp"]

    @property
    def gdp_fc(self) -> np.ndarray:
        """GDP at factor cost, ``w L + r K``."""
        return self._flows()["gdp_fc"]

    def summary_markups(self, by_sector: bool = True) -> pd.DataFrame:
        """Mean, min, max and standard deviation of the markups (by sector or overall)."""
        if isinstance(self.markups, pd.DataFrame) and not self.markups.empty and "markup" in self.markups.columns:
            if by_sector and "sector" in self.markups.columns:
                grouped = (self.markups.groupby("sector", as_index=False, sort=False)["markup"]
                           .agg(mean="mean", min="min", max="max", std="std"))
                grouped["std"] = grouped["std"].fillna(0.0)
                return grouped
            m = self.markups["markup"]
            return pd.DataFrame({
                "metric": ["mean", "min", "max", "std"],
                "value": [float(m.mean()), float(m.min()), float(m.max()),
                          float(m.std()) if len(m) > 1 else 0.0],
            })
        if by_sector and self.calib is not None:
            s_codes = list(self.calib.sector_codes) or [f"S{i:02d}" for i in range(self.calib.n_sectors)]
            n = len(s_codes)
            return pd.DataFrame({"sector": s_codes, "mean": [1.0] * n, "min": [1.0] * n,
                                 "max": [1.0] * n, "std": [0.0] * n})
        return pd.DataFrame({"metric": ["mean", "min", "max", "std"], "value": [1.0, 1.0, 1.0, 0.0]})

    def factor_allocation_frame(self) -> pd.DataFrame:
        """Labour and capital demand by country and sector at the solution.

        Columns ``country``, ``sector``, ``labor``, ``capital`` and the aliases
        ``xl``, ``xk``. The demands are those of the solved model (its
        technology, tariffs and capacity barrier), so at a converged solution
        the national totals equal the endowments to solver tolerance.
        """
        if self._factors_frame is not None:
            return self._factors_frame
        if self.calib is None:
            return pd.DataFrame(columns=["country", "sector", "labor", "capital", "xl", "xk"])
        calib = self.calib
        b = self._blocks()
        nc, ns = calib.n_countries, calib.n_sectors
        c_codes = list(calib.country_codes) or [f"C{i:02d}" for i in range(nc)]
        s_codes = list(calib.sector_codes) or [f"S{i:02d}" for i in range(ns)]
        records = []
        for c_idx, c_code in enumerate(c_codes):
            for s_idx, s_code in enumerate(s_codes):
                xl = float(b["labor"][0, s_idx, c_idx])
                xk = float(b["capital"][0, s_idx, c_idx])
                records.append({"country": c_code, "sector": s_code, "labor": xl, "capital": xk,
                                "xl": xl, "xk": xk})
        self._factors_frame = pd.DataFrame(records)
        return self._factors_frame

    def welfare_decomposition(self, base_result: Any = None) -> pd.DataFrame:
        """Household welfare change against a baseline, split into terms of trade and efficiency.

        ``EV`` is the money-metric change of the household (final-use category
        0) at the baseline's sector prices, computed from both results' own
        flows (their household budgets and sector composite prices, at their
        own tariffs). Without subsistence the household buys a fixed basket,
        and ``EV = B_1 P(p_0) / P(p_1) - B_0`` with ``P`` the basket cost is the
        exact equivalent variation. With Stone-Geary subsistence the solved
        subsistence bundle is valued at baseline prices and supernumerary
        spending is deflated by the marginal basket, ``EV = p_0 c_bar_1 +
        S_1 P_LES(p_0) / P_LES(p_1) - B_0``; because subsistence quantities
        move with income, this is a money-metric approximation.
        ``terms_of_trade`` is ``(tot_1 - tot_0)`` times average gross trade
        ``(X_1 + M_1) / 2`` (each result's own terms of trade, exports and
        imports) and ``efficiency`` is the remainder ``EV - terms_of_trade``.

        Parameters
        ----------
        base_result : FlexibleTradeEquilibriumResult or TradeEquilibriumResult, optional
            Baseline. None, ``self`` or an identical state give zero rows. A
            core result is evaluated as the default configuration at its own
            tariffs and closure.

        Returns
        -------
        pd.DataFrame
            Columns ``country``, ``EV``, ``terms_of_trade``, ``efficiency``.
        """
        if base_result is not None:
            x_base = getattr(base_result, "x_sol", None)
            if x_base is not None and len(np.ravel(x_base)) != len(self.x_sol):
                raise ValueError(
                    f"Dimension mismatch in welfare decomposition: {len(np.ravel(x_base))} vs {len(self.x_sol)}."
                )
        calib = self.calib if self.calib is not None else getattr(base_result, "calib", None)
        if calib is None:
            raise ValueError("welfare_decomposition needs the calibration (calib) of the result")
        nc = calib.n_countries
        c_codes = list(calib.country_codes) or [f"C{i:02d}" for i in range(nc)]
        x_base = getattr(base_result, "x_sol", None) if base_result is not None else None
        if base_result is None or base_result is self or (
                x_base is not None and np.allclose(self.x_sol, np.ravel(x_base), rtol=0, atol=1e-12)):
            zeros = [0.0] * nc
            return pd.DataFrame({"country": c_codes, "EV": zeros, "terms_of_trade": zeros, "efficiency": zeros})

        from puremacro.trade._accounting import postprocess

        b1 = self._blocks()
        f1 = self._flows()
        if isinstance(base_result, FlexibleTradeEquilibriumResult):
            b0, f0 = base_result._blocks(), base_result._flows()
        else:
            b0 = _blocks_of_result(base_result, calib, None)
            f0 = postprocess(b0, calib)
        pref = self.config.preference if self.config is not None else FlexiblePreferenceConfig()
        X0, X1 = b0["household_budget"], b1["household_budget"]
        P0, P1 = b0["sector_prices"][:, 0, :], b1["sector_prices"][:, 0, :]
        mu = _resolve_subsistence_shares(calib, pref)[0]
        budget_0, theta_sec_0, c_s0 = _extract_benchmark_household_data(calib)
        if np.all(mu == 0.0):
            theta = theta_sec_0[0]
            EV = X1 * np.sum(theta * P0, axis=0) / np.sum(theta * P1, axis=0) - X0
        else:
            theta_les = compute_les_marginal_budget_shares(calib, pref)[0]
            c_bar1 = mu * c_s0[0] * np.asarray(smooth_subsistence_scaling(np.maximum(X1 / budget_0[0, 0], 0.0)))[None, :]
            super1 = np.maximum(X1 - np.sum(P1 * c_bar1, axis=0), 0.0)
            EV = (np.sum(P0 * c_bar1, axis=0)
                  + super1 * np.sum(theta_les * P0, axis=0) / np.sum(theta_les * P1, axis=0) - X0)
        tot_effect = (np.ravel(f1["terms_of_trade"]) - np.ravel(f0["terms_of_trade"])) * 0.5 * (
            np.ravel(f1["exports"]) + np.ravel(f1["imports"]))
        efficiency = EV - tot_effect
        return pd.DataFrame({
            "country": c_codes,
            "EV": [float(v) for v in EV],
            "terms_of_trade": [float(v) for v in tot_effect],
            "efficiency": [float(v) for v in efficiency],
        })

    def welfare_summary(self, base_result: Any = None) -> pd.DataFrame:
        """Alias for :meth:`welfare_decomposition`."""
        return self.welfare_decomposition(base_result)

    @property
    def convergence_diagnostic(self) -> dict[str, Any]:
        """Convergence diagnostic of an unconverged solve (empty when converged)."""
        diag = self.metadata.get("convergence_diagnostic", {})
        return diag if isinstance(diag, dict) else {}

    @property
    def top_offending_equations(self) -> list[dict[str, Any]]:
        """The worst residual equations of an unconverged solve."""
        return self.convergence_diagnostic.get("top_equations", [])


FlexibleEquilibriumResult = FlexibleTradeEquilibriumResult


# =============================================================================
# Configuration resolution and the solver entry point
# =============================================================================

TECH_FIELDS = {
    "rho_va", "sigma_va", "sigma_y", "sigma_inter", "capacity_margins", "capacity_country",
    "capacity_sector", "penalty_scale", "penalty_exponent",
}
PREF_FIELDS = {"subsistence_shares", "mu_s", "subsistence_ratio", "sigma_trade"}
MARKET_FIELDS = {
    "variable_markups", "sigma_j", "theta_j", "variety_condensation", "condense_varieties",
    "variety_expansion", "markup_min", "markup_max", "markup_bounds", "cournot_weights",
    "clamping_threshold", "fl", "fk",
}
MODEL_FIELDS = {"max_inner_iter", "technology", "preference", "market_structure"}
SOLVER_KWARGS = {
    "tau", "tau_fd", "tauf", "tauf_fd", "tau_a", "taufd_a", "tol", "max_iter", "method", "x0",
    "eps_fd", "base_result", "accounting", "fiscal_closure", "recycle_country", "foreign_saving_units",
}


def _resolve_flexible_model_config(
    config: FlexibleTradeModelConfig | None = None,
    rho_va: float | None = None,
    sigma_y: float | None = None,
    **kwargs: Any,
) -> tuple[FlexibleTradeModelConfig, dict[str, Any]]:
    """Split keyword arguments into a model configuration and solver options.

    Raises
    ------
    TypeError
        For a keyword outside the configuration fields and solver options.
    ValueError
        For conflicting alias values or out-of-range settings.
    """
    spurious = set(kwargs) - (TECH_FIELDS | PREF_FIELDS | MARKET_FIELDS | MODEL_FIELDS | SOLVER_KWARGS)
    if spurious:
        raise TypeError(f"Unknown keyword argument(s): {sorted(spurious)}")
    tech_kwargs = {k: v for k, v in kwargs.items() if k in TECH_FIELDS}
    pref_kwargs = {k: v for k, v in kwargs.items() if k in PREF_FIELDS}
    mkt_kwargs = {k: v for k, v in kwargs.items() if k in MARKET_FIELDS}
    model_kwargs = {k: v for k, v in kwargs.items() if k in MODEL_FIELDS}
    solver_kwargs = {k: v for k, v in kwargs.items() if k in SOLVER_KWARGS}
    if rho_va is not None:
        tech_kwargs["rho_va"] = rho_va
    if sigma_y is not None:
        tech_kwargs["sigma_y"] = sigma_y

    if "rho_va" in tech_kwargs and "sigma_va" in tech_kwargs:
        if tech_kwargs["rho_va"] != tech_kwargs["sigma_va"]:
            raise ValueError(
                f"Conflicting values provided for rho_va ({tech_kwargs['rho_va']}) "
                f"and sigma_va ({tech_kwargs['sigma_va']})."
            )
    elif "rho_va" in tech_kwargs:
        tech_kwargs["sigma_va"] = tech_kwargs["rho_va"]
    elif "sigma_va" in tech_kwargs:
        tech_kwargs["rho_va"] = tech_kwargs["sigma_va"]

    sub_keys = [k for k in ("subsistence_shares", "mu_s", "subsistence_ratio") if k in pref_kwargs]
    if sub_keys:
        first_val = pref_kwargs[sub_keys[0]]
        for k in sub_keys[1:]:
            if not _val_equal(first_val, pref_kwargs[k]):
                raise ValueError("Conflicting values provided for subsistence parameters.")
        for k in ("subsistence_shares", "mu_s", "subsistence_ratio"):
            pref_kwargs[k] = first_val

    if "markup_bounds" in mkt_kwargs:
        mb = mkt_kwargs["markup_bounds"]
        if isinstance(mb, (tuple, list)) and len(mb) == 2:
            mkt_kwargs["markup_min"], mkt_kwargs["markup_max"] = float(mb[0]), float(mb[1])
            mkt_kwargs["markup_bounds"] = (float(mb[0]), float(mb[1]))
    elif "markup_min" in mkt_kwargs or "markup_max" in mkt_kwargs:
        cur_min = config.market_structure.markup_min if config is not None else 1.0
        cur_max = config.market_structure.markup_max if config is not None else 5.0
        b_min = float(mkt_kwargs.get("markup_min", cur_min))
        b_max = float(mkt_kwargs.get("markup_max", cur_max))
        mkt_kwargs.update(markup_min=b_min, markup_max=b_max, markup_bounds=(b_min, b_max))
    cond_keys = ("condense_varieties", "variety_condensation", "variety_expansion")
    if any(ck in mkt_kwargs for ck in cond_keys):
        flag = any(bool(mkt_kwargs.get(ck)) for ck in cond_keys)
        for ck in cond_keys:
            mkt_kwargs[ck] = flag

    base = config if config is not None else FlexibleTradeModelConfig()
    tech = model_kwargs.get("technology", base.technology)
    pref = model_kwargs.get("preference", base.preference)
    mkt = model_kwargs.get("market_structure", base.market_structure)
    if tech_kwargs:
        tech = replace(tech, **tech_kwargs)
    if pref_kwargs:
        pref = replace(pref, **pref_kwargs)
    if mkt_kwargs:
        mkt = replace(mkt, **mkt_kwargs)
    args: dict[str, Any] = {"technology": tech, "preference": pref, "market_structure": mkt}
    if "max_inner_iter" in model_kwargs:
        args["max_inner_iter"] = model_kwargs["max_inner_iter"]
    return replace(base, **args), solver_kwargs


def _build_markups_frame(
    x_sol: np.ndarray,
    calib: TradeCalibrationResult,
    config: FlexibleTradeModelConfig,
) -> pd.DataFrame:
    """Markups by origin, destination and sector at the solution (1 when markups are off)."""
    nc, ns = calib.n_countries, calib.n_sectors
    c_arr = np.array(list(calib.country_codes) or [f"C{i:02d}" for i in range(nc)])
    s_arr = np.array(list(calib.sector_codes) or [f"S{i:02d}" for i in range(ns)])
    origins = np.repeat(c_arr, nc * ns)
    dests = np.tile(np.repeat(c_arr, ns), nc)
    sectors = np.tile(s_arr, nc * nc)
    if not config.market_structure.variable_markups:
        markup = np.ones(nc * nc * ns, dtype=float)
    else:
        uv = unpack_equilibrium_vector(x_sol, ns=ns, nc=nc, nfd=calib.n_final_demand)
        _, mu_ni, _ = _market_shares_and_markups(uv.p[0].flatten(order="F"), calib, config.market_structure)
        markup = np.transpose(mu_ni, (1, 2, 0)).ravel()
    return pd.DataFrame({"origin": origins, "destination": dests, "sector": sectors, "markup": markup})


# Largest calibration (country-sector cells) routed automatically to the
# condensed solver for an active configuration.
_AUTO_QUASI_CONDENSED_MAX_CELLS = 100

_LUMP_SUM_CLOSURES = (None, "lump_sum", "baseline", "")


def solve_flexible_trade_equilibrium(
    calib: TradeCalibrationResult,
    config: FlexibleTradeModelConfig | None = None,
    rho_va: float | None = None,
    sigma_y: float | None = None,
    **kwargs: Any,
) -> FlexibleTradeEquilibriumResult:
    """Solve the trade equilibrium with flexible technology, preferences and markups.

    Settings are given as a :class:`FlexibleTradeModelConfig` and/or keyword
    overrides (``rho_va=0.7``, ``variable_markups=True``,
    ``subsistence_shares={"S01": 0.3}``, ...). The model is the consistent
    accounting of :func:`puremacro.trade.solve_trade_equilibrium` with the
    configured blocks in place of Cobb-Douglas/Leontief technology and
    fixed-proportion demand.

    Routing
    -------
    A configuration is *active* when a flexible block differs from its default
    (``rho_va != 1``, ``sigma_y > 0``, ``sigma_inter > 0``, capacity margins,
    a non-zero subsistence share, ``sigma_trade`` set or
    ``variable_markups=True``; see :func:`_active_flexible_settings`).

    1. Default configuration: the core solver
       (:func:`puremacro.trade.solve_trade_equilibrium`, ``method`` as given,
       Newton by default); it solves the same model.
    2. ``method="quasi_condensed"``, or an active configuration without
       ``method`` on a calibration of at most 100 country-sector cells: the
       condensed solver (damped Newton on ``[log r; log w; T]``, a dense
       ``3 nc x 3 nc`` finite-difference Jacobian per iteration, each column
       an inner price and quantity solve). When Newton stalls (its line
       search finds no decrease) or runs out of iterations under a non-zero
       tariff schedule, the solver retries by a tariff homotopy: the schedule
       is reached in 4, then 16, equal steps from free trade, each warm-started
       from the last. ``max_iter`` bounds the Newton iterations of the whole
       solve, homotopy included, so ``iterations`` never exceeds it; the
       homotopy only gets the iterations the direct attempt left unused.
       ``metadata["tariff_homotopy_steps"]`` records the path that converged
       (0 when a homotopy was tried and failed; absent when none ran) and
       ``metadata["newton_stalled"]`` whether the reported attempt stalled.
    3. An active configuration with a core ``method``, or above 100 cells
       without ``method``: ``ValueError`` (the core solvers cannot apply the
       settings; pass ``method="quasi_condensed"``).

    ``max_iter=0`` evaluates the residual at the starting point without
    solving. The reported residual of the condensed solver is certified at
    ``x_sol`` (inner loops to 1e-13 from a cold start), so calling again with
    ``x0=x_sol`` and ``max_iter=0`` reproduces it.

    Parameters
    ----------
    calib : TradeCalibrationResult
        Calibration from :func:`puremacro.trade.calibrate_trade_model`.
    config : FlexibleTradeModelConfig, optional
        Base configuration; keyword overrides are applied on top of it.
    rho_va, sigma_y : float, optional
        Technology overrides.
    **kwargs
        Configuration fields (technology: ``sigma_va``, ``sigma_inter``,
        ``capacity_margins``, ``capacity_country``, ``capacity_sector``,
        ``penalty_scale``, ``penalty_exponent``; preferences:
        ``subsistence_shares``/``mu_s``/``subsistence_ratio``,
        ``sigma_trade``; market structure: ``variable_markups``, ``sigma_j``,
        ``theta_j``, ``markup_bounds``, ...; ``max_inner_iter``) and solver
        options: ``tau``, ``tau_fd``, ``tauf``, ``tauf_fd`` (conventions of
        :func:`puremacro.trade.solve_trade_equilibrium`; ``tau_a``/``taufd_a``
        are aliases), ``method``, ``tol`` (default 2.5e-3), ``max_iter``
        (default 50), ``x0``, ``eps_fd``, ``base_result``,
        ``foreign_saving_units`` (``"numeraire"`` or ``"world_income"``),
        ``accounting`` (only ``"consistent"``), ``fiscal_closure`` and
        ``recycle_country``. Any other keyword raises ``TypeError``.

    Returns
    -------
    FlexibleTradeEquilibriumResult
        ``metadata`` carries ``config``, ``routing`` (``"default"``,
        ``"explicit"`` or ``"auto"``), ``requested_method``, ``method``,
        ``effective_method``, ``active_flexible_settings``,
        ``flexible_settings_applied``, ``tariff_inputs`` and the resolved
        ``intermediate_tariff_multipliers``/``final_tariff_multipliers``,
        ``foreign_saving_units``, ``accounting``, ``final_demand_sourcing``,
        ``trade_balance_valuation``, the flows of the solved model
        (``bilateral_trade``, ``household_income``, ``final_demand_values``
        (purchaser-valued deliveries, ``ns*nc x nfd x nc``), ``markup_profits``,
        ``negative_final_demand_composites``), the consistent-accounting
        ledger (``account_residuals``, ``physical_residuals``), and for the
        condensed solver ``inner_price_converged`` and
        ``quantity_fixed_point_converged``.

    Raises
    ------
    TypeError
        An unknown keyword; a capacity barrier without ``capacity_country`` /
        ``capacity_sector``; a non-lump-sum ``fiscal_closure`` without
        ``recycle_country``.
    NotImplementedError
        A non-lump-sum fiscal closure (only lump-sum rebates are implemented).
    ValueError
        Active settings on a path that cannot apply them, ``accounting`` other
        than ``"consistent"``, or an invalid setting.

    Examples
    --------
    >>> from puremacro.trade import calibrate_trade_model, solve_flexible_trade_equilibrium
    >>> from puremacro.trade.data import synthetic_icio_table
    >>> calib = calibrate_trade_model(synthetic_icio_table(4, 3, seed=2), ns=3, nc=4)
    >>> res = solve_flexible_trade_equilibrium(calib, rho_va=0.7, tau=0.1)
    >>> bool(res.converged)
    True
    """
    from puremacro.trade._accounting import FOREIGN_SAVING_UNITS, postprocess
    from puremacro.trade.solver import _resolve_tariffs, build_initial_guess, solve_trade_equilibrium

    if not isinstance(calib, TradeCalibrationResult):
        raise TypeError(f"calib must be a TradeCalibrationResult, got {type(calib).__name__}")
    config, sk = _resolve_flexible_model_config(config=config, rho_va=rho_va, sigma_y=sigma_y, **kwargs)
    check_accounting(sk.pop("accounting", "consistent"))
    fiscal_closure = sk.pop("fiscal_closure", "lump_sum")
    recycle_country = sk.pop("recycle_country", None)
    check_policy_options(fiscal_closure if fiscal_closure is not None else "lump_sum", recycle_country)
    if fiscal_closure not in _LUMP_SUM_CLOSURES:
        raise NotImplementedError(
            f"fiscal_closure={fiscal_closure!r}: the flexible model implements lump-sum rebates of "
            "all government receipts only"
        )
    units = sk.pop("foreign_saving_units", "numeraire")
    if units not in FOREIGN_SAVING_UNITS:
        raise ValueError("foreign_saving_units must be 'numeraire' or 'world_income'")
    for alias, name in (("tau_a", "tau"), ("taufd_a", "tau_fd")):
        if alias in sk:
            value = sk.pop(alias)
            if sk.get(name) is not None and value is not None:
                raise ValueError(f"pass either {name} or its alias {alias}, not both")
            if value is not None:
                sk[name] = value
    tariff_inputs = {k: copy.deepcopy(sk.get(k)) for k in ("tau", "tau_fd", "tauf", "tauf_fd")}
    ta, tf, _, _ = _resolve_tariffs(calib, sk.get("tau"), sk.get("tau_fd"), sk.get("tauf"), sk.get("tauf_fd"))
    ta = np.asarray(ta, dtype=float)
    tf = np.asarray(tf, dtype=float)
    _check_schedules(ta, tf, calib)
    requested = sk.get("method")
    tol = float(sk.get("tol", 2.5e-3))
    max_iter = int(sk.get("max_iter", 50))
    x0 = sk.get("x0")

    active = _active_flexible_settings(config)
    n_cells = int(calib.n_countries * calib.n_sectors)
    if requested == "quasi_condensed":
        route, routing = "quasi_condensed", "explicit"
    elif not active:
        route, routing = "core", ("default" if requested is None else "explicit")
    elif requested is None and (n_cells <= _AUTO_QUASI_CONDENSED_MAX_CELLS or max_iter == 0):
        route, routing = "quasi_condensed", "auto"
    else:
        path = (f"method={requested!r}, a solver of the core Cobb-Douglas/Leontief model"
                if requested is not None else
                f"automatic routing, which uses the condensed solver only up to "
                f"{_AUTO_QUASI_CONDENSED_MAX_CELLS} country-sector cells (this calibration has {n_cells})")
        raise ValueError(
            f"Flexible settings {list(active)} cannot be applied on {path}. Pass "
            f"method='quasi_condensed' to solve the configured model (dense Newton on "
            f"{3 * calib.n_countries} macro unknowns)."
        )

    meta: dict[str, Any] = {}
    if route == "quasi_condensed":
        x_sol, conv, iters, blocks, meta = _condensed_solve(
            calib, config, ta, tf, x0=x0, tol=tol, max_iter=max_iter,
            eps_fd=float(sk.get("eps_fd", 1e-4)), foreign_saving_units=units,
        )
        method, effective = "quasi_condensed", ("quasi_condensed" if max_iter > 0 else "residual_evaluation")
    elif max_iter == 0:
        x_sol = build_initial_guess(calib) if x0 is None else np.asarray(x0, dtype=float).ravel()
        blocks = _flexible_blocks(x_sol, calib, config, ta, tf, units)
        conv, iters = False, 0
        diag = compute_convergence_diagnostics(blocks["residuals"], calib)
        diag["remediation"] = diag["remediation"] + " Increase solver max_iter or reduce the shock."
        meta["convergence_diagnostic"] = diag
        method, effective = (requested if requested is not None else "newton"), "residual_evaluation"
    else:
        core_options = {"eps_fd": sk["eps_fd"]} if "eps_fd" in sk else {}
        core = solve_trade_equilibrium(
            calib, tau=ta, tau_fd=tf, x0=x0, method=requested if requested is not None else "newton",
            tol=tol, max_iter=max_iter, base_result=sk.get("base_result"), foreign_saving_units=units,
            **core_options,
        )
        x_sol = np.asarray(core.x_sol, dtype=float).ravel()
        conv, iters = bool(core.converged), int(core.iterations)
        blocks = _flexible_blocks(x_sol, calib, config, ta, tf, units)
        for key in ("solver_termination", "solver_residual_history", "solve_duration_seconds", "fd_step"):
            if key in core.metadata:
                meta[key] = core.metadata[key]
        if not conv:
            meta["convergence_diagnostic"] = compute_convergence_diagnostics(blocks["residuals"], calib)
        method = requested if requested is not None else "newton"
        effective = core.metadata.get("effective_method", method)

    flows = postprocess(blocks, calib)
    nc, nfd = calib.n_countries, calib.n_final_demand
    c_codes = list(calib.country_codes) or [f"C{i:02d}" for i in range(nc)]
    negative = [
        {"category": int(f), "country": c_codes[c], "expenditure": float(blocks["expenditure"][0, f, c])}
        for f in range(nfd) for c in range(nc)
        if float(blocks["expenditure"][0, f, c]) < 0.0 and calib.theta is not None and float(calib.theta[0, f, c]) > 0.0
    ]
    acc = flows["accounting_metadata"]
    meta.update({
        "config": config,
        "routing": routing,
        "requested_method": requested,
        "method": method,
        "effective_method": effective,
        "flexible_settings_applied": True,
        "active_flexible_settings": list(active),
        "tol": tol,
        "max_iter": max_iter,
        "tariff_inputs": tariff_inputs,
        "intermediate_tariff_multipliers": ta,
        "final_tariff_multipliers": tf,
        "foreign_saving_units": units,
        "accounting": "consistent",
        "fiscal_closure": "lump_sum",
        "final_demand_sourcing": "fixed_coefficients" if config.preference.sigma_trade is None else "armington",
        "trade_balance_valuation": "producer prices",
        "bilateral_trade": blocks["trade"],
        "household_income": blocks["income"].ravel().copy(),
        "household_budget": blocks["household_budget"],
        "final_demand_values": acc["purchaser_final_values"].reshape(
            (calib.n_sectors * nc, nfd, nc), order="F"),
        "negative_final_demand_composites": negative,
        "account_residuals": acc["account_residuals"],
        "physical_residuals": acc["physical_residuals"],
    })
    if config.market_structure.variable_markups:
        meta["markup_profits"] = blocks["markup_profits"]
    residuals = blocks["residuals"]
    return FlexibleTradeEquilibriumResult(
        x_sol=np.asarray(x_sol, dtype=float).ravel(),
        converged=bool(conv),
        iterations=int(iters),
        residual_norm=float(np.max(np.abs(residuals))) if np.isfinite(residuals).all() else float("inf"),
        residuals=residuals,
        metadata=meta,
        markups=_build_markups_frame(x_sol, calib, config),
        calib=calib,
        config=config,
        _blocks_cache=blocks,
        _flows_cache=flows,
    )


__all__ = [
    "FlexibleTechnologyConfig",
    "FlexiblePreferenceConfig",
    "FlexibleMarketStructureConfig",
    "FlexibleTradeModelConfig",
    "FlexibleTradeEquilibriumResult",
    "FlexibleEquilibriumResult",
    "compute_capacity_penalty",
    "compute_inner_ces_cost",
    "compute_outer_ces_cost",
    "compute_intermediate_composite_price",
    "compute_nested_ces_costs",
    "compute_nested_factor_demands",
    "compute_zero_profit_price",
    "compute_stone_geary_final_demand",
    "compute_les_marginal_budget_shares",
    "smooth_subsistence_scaling",
    "compute_armington_purchaser_prices",
    "compute_armington_final_demands",
    "compute_multi_category_final_demands",
    "compute_atkeson_burstein_markups",
    "compute_benchmark_market_shares",
    "compute_benchmark_markups",
    "compute_dixit_stiglitz_varieties",
    "compute_condensed_varieties",
    "compute_variety_price_scaling",
    "compute_convergence_diagnostics",
    "solve_flexible_trade_equilibrium",
]
