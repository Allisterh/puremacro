"""Equilibrium state vector and residual system of the static trade model.

The unknowns are ``x = [log p; log y; log r; log w; T; XN]`` (``2*ns*nc + 4*nc - 1``
entries): producer prices and gross output by sector and country, the rental
rate and the wage by country, fiscal receipts by country, and the foreign
balances of the first ``nc - 1`` countries (the last one closes the world
identity). The residual blocks are goods-market clearing, zero profit, labour
and capital market clearing, the foreign-balance closure and the government
budget; :mod:`puremacro.trade._accounting` evaluates them with consistent
producer/purchaser valuation, which is the only accounting the library offers.

Pure NumPy, vectorised.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from puremacro.trade._results import TradeCalibrationResult

ACCOUNTING_REMOVED_MESSAGE = (
    "legacy accounting was removed; consistent producer/purchaser accounting "
    "(accounting='consistent', the default) is the only accounting"
)


def check_accounting(accounting: str) -> None:
    """Raise ``ValueError`` unless ``accounting == "consistent"``."""
    if accounting != "consistent":
        raise ValueError(f"accounting={accounting!r}: {ACCOUNTING_REMOVED_MESSAGE}")


class EquilibriumVariables(NamedTuple):
    """Unpacked economic variables from the general equilibrium state vector."""

    p: np.ndarray        # Gross output prices, shape (1, ns, nc)
    y: np.ndarray        # Gross output quantities, shape (1, ns, nc)
    r: np.ndarray        # Capital rental rates, shape (1, 1, nc)
    w: np.ndarray        # Wage rates, shape (1, 1, nc)
    T: np.ndarray        # Fiscal revenues / transfers (in levels), shape (1, 1, nc)
    XN: np.ndarray       # Net foreign transfers (in levels), shape (nc - 1,)
    invforT: np.ndarray  # Complete foreign current accounts with global closure, shape (1, nc)


def unpack_equilibrium_vector(
    x: np.ndarray,
    ns: int,
    nc: int,
    nfd: int = 3,
    return_levels: bool = True,
) -> EquilibriumVariables:
    """Unpack state vector x into component economic tensors.

    Parameters
    ----------
    x : np.ndarray
        1D array of length 2*ns*nc + 3*nc + (nc - 1), representing
        [log(p); log(y); log(r); log(w); T; XN].
    ns : int
        Number of sectors.
    nc : int
        Number of countries.
    nfd : int, default 3
        Number of final demand categories (only reported in the error message;
        the state vector does not depend on it).
    return_levels : bool, default True
        If True, exponentiates log-prices, log-outputs, and log-factor-returns.

    Returns
    -------
    EquilibriumVariables
        NamedTuple with unpacked fields p, y, r, w, T, XN, and invforT.
    """
    x_flat = np.asarray(x, dtype=float).ravel()
    expected_len = 2 * ns * nc + 3 * nc + (nc - 1)
    if len(x_flat) != expected_len:
        raise ValueError(
            f"State vector length {len(x_flat)} does not match expected {expected_len} "
            f"for ns={ns}, nc={nc}, nfd={nfd}."
        )

    i_p_end = ns * nc
    i_y_end = 2 * ns * nc
    i_r_end = 2 * ns * nc + nc
    i_w_end = 2 * ns * nc + 2 * nc
    i_T_end = 2 * ns * nc + 3 * nc
    i_XN_end = 2 * ns * nc + 4 * nc - 1

    p_raw = x_flat[:i_p_end]
    y_raw = x_flat[i_p_end:i_y_end]
    r_raw = x_flat[i_y_end:i_r_end]
    w_raw = x_flat[i_r_end:i_w_end]
    T_raw = x_flat[i_w_end:i_T_end]
    XN_raw = x_flat[i_T_end:i_XN_end]

    if return_levels:
        p = np.exp(p_raw).reshape((1, ns, nc), order="F")
        y = np.exp(y_raw).reshape((1, ns, nc), order="F")
        r = np.exp(r_raw).reshape((1, 1, nc), order="F")
        w = np.exp(w_raw).reshape((1, 1, nc), order="F")
    else:
        p = p_raw.reshape((1, ns, nc), order="F")
        y = y_raw.reshape((1, ns, nc), order="F")
        r = r_raw.reshape((1, 1, nc), order="F")
        w = w_raw.reshape((1, 1, nc), order="F")

    T = T_raw.reshape((1, 1, nc), order="F")
    XN = XN_raw.copy()

    # Global trade balance closure: sum(invforT) == 0 identity
    # Country nc (index nc - 1) is determined residually
    invforT = np.zeros((1, nc), dtype=float)
    invforT[0, :nc - 1] = XN
    invforT[0, nc - 1] = -np.sum(XN)

    return EquilibriumVariables(p=p, y=y, r=r, w=w, T=T, XN=XN, invforT=invforT)


def pack_equilibrium_vector(
    p: np.ndarray,
    y: np.ndarray,
    r: np.ndarray,
    w: np.ndarray,
    T: np.ndarray,
    XN: np.ndarray,
    log_transformed: bool = True,
) -> np.ndarray:
    """Pack economic tensors into the 1D general equilibrium state vector.

    The order is ``x = [log(p); log(y); log(r); log(w); T; XN]``, with sector
    blocks stacked within each country (column-major over ``(ns, nc)``).

    Parameters
    ----------
    p : np.ndarray
        Gross output prices, shape (1, ns, nc) or (ns*nc,).
    y : np.ndarray
        Gross output quantities, shape (1, ns, nc) or (ns*nc,).
    r : np.ndarray
        Capital rental rates, shape (1, 1, nc) or (nc,).
    w : np.ndarray
        Wage rates, shape (1, 1, nc) or (nc,).
    T : np.ndarray
        Government fiscal revenues/transfers (in levels), shape (1, 1, nc) or (nc,).
    XN : np.ndarray
        Net foreign transfers (in levels), shape (nc - 1,).
    log_transformed : bool, default True
        If True, applies np.log to p, y, r, and w. If False, assumes they are already logs.

    Returns
    -------
    np.ndarray
        1D state vector of shape (2*ns*nc + 3*nc + nc - 1,).
    """
    p_flat = np.asarray(p, dtype=float).flatten(order="F")
    y_flat = np.asarray(y, dtype=float).flatten(order="F")
    r_flat = np.asarray(r, dtype=float).flatten(order="F")
    w_flat = np.asarray(w, dtype=float).flatten(order="F")
    T_flat = np.asarray(T, dtype=float).flatten(order="F")
    XN_flat = np.asarray(XN, dtype=float).ravel()

    if log_transformed:
        p_part = np.log(p_flat)
        y_part = np.log(y_flat)
        r_part = np.log(r_flat)
        w_part = np.log(w_flat)
    else:
        p_part = p_flat
        y_part = y_flat
        r_part = r_flat
        w_part = w_flat

    return np.concatenate([p_part, y_part, r_part, w_part, T_flat, XN_flat])


def _get_ces_weights(calib: TradeCalibrationResult) -> tuple[np.ndarray, np.ndarray]:
    """Precompute and cache CES aggregation weights on the calibration instance."""
    cached = getattr(calib, "_ces_weights_cache", None)
    if cached is not None:
        return cached

    A_mat = np.sum(calib.a, axis=0)  # shape (ns, nc)
    A_mat_safe = np.where(A_mat > 0, A_mat, 1.0)
    omega = calib.a / A_mat_safe[np.newaxis, :, :]  # shape (ns*nc, ns, nc)
    cached_val = (A_mat, omega)
    try:
        object.__setattr__(calib, "_ces_weights_cache", cached_val)
    except Exception:
        pass
    return cached_val


def check_policy_options(
    fiscal_closure: str = "lump_sum",
    recycle_country: str | None = None,
    capacity_margins: Any = None,
    capacity_country: str | None = None,
    capacity_sector: str | None = None,
) -> None:
    """Validate the fiscal-recycling and capacity options.

    A non-lump-sum ``fiscal_closure`` needs ``recycle_country`` and capacity
    margins need ``capacity_country`` and ``capacity_sector``; none of them has
    a default, so a missing one raises ``TypeError`` naming it. The options are
    checked here, before the model is evaluated; consistent accounting then
    reports that it implements lump-sum rebates and no capacity barrier
    (``NotImplementedError``).
    """
    if fiscal_closure not in ("lump_sum", "baseline", "") and recycle_country is None:
        raise TypeError(
            f"fiscal_closure={fiscal_closure!r} recycles one country's tariff revenue: "
            "pass recycle_country=<country code>"
        )
    if capacity_margins is not None:
        for name, value in (("capacity_country", capacity_country), ("capacity_sector", capacity_sector)):
            if value is None:
                raise TypeError(f"capacity_margins applies to one country and sector: pass {name}=<code>")


def _recycling_params(recycle_country: str | None) -> dict[str, Any] | None:
    return None if recycle_country is None else {"recycle_country": recycle_country}


def compute_equilibrium_residuals(
    x: np.ndarray,
    calib: TradeCalibrationResult,
    tau: np.ndarray | None = None,
    tau_fd: np.ndarray | None = None,
    tauf: np.ndarray | None = None,
    tauf_fd: np.ndarray | None = None,
    *,
    tau_a: np.ndarray | None = None,
    taufd_a: np.ndarray | None = None,
    sigma: float = 0.0,
    accounting: str = "consistent",
    fiscal_closure: str = "lump_sum",
    recycle_country: str | None = None,
    capacity_margins: dict[str, float] | np.ndarray | None = None,
    capacity_country: str | None = None,
    capacity_sector: str | None = None,
    foreign_saving_units: str = "numeraire",
    **kwargs: Any,
) -> np.ndarray:
    """Residual vector ``F(x)`` of the ``2*ns*nc + 4*nc - 1`` equilibrium equations.

    Producer-price trade and duty bases, homogeneous Cobb-Douglas factor costs,
    tax-inclusive actual final spending and lump-sum rebates. Block 4 fixes the
    baseline foreign balances, and the first goods equation is replaced by the
    numeraire condition (first producer price equal to one); postprocessing
    checks the omitted goods equation and the realised trade balances.

    Parameters
    ----------
    x : np.ndarray
        State vector of length 2*ns*nc + 3*nc + (nc - 1).
    calib : TradeCalibrationResult
        Calibrated model parameters from :func:`calibrate_trade_model`.
    tau, tau_fd : np.ndarray, optional
        Intermediate and final-demand tariffs, with the conventions of
        :func:`puremacro.trade.solver.solve_trade_equilibrium`.
    tauf, tauf_fd : np.ndarray, optional
        National tariff rates by importing country, shape (nc,).
    tau_a, taufd_a : np.ndarray, optional (keyword-only)
        Aliases of ``tau`` and ``tau_fd`` that take precedence when given.
    sigma : float, default 0.0
        Intermediate sourcing elasticity of substitution (0.0 = Leontief).
    accounting : {"consistent"}, default "consistent"
        Any other value raises ``ValueError`` (legacy accounting was removed).
    fiscal_closure, recycle_country, capacity_margins, capacity_country, capacity_sector
        Checked by :func:`check_policy_options`; only lump-sum rebates without a
        capacity barrier are implemented.
    foreign_saving_units : {"numeraire", "world_income"}, default "numeraire"
        Unit in which baseline foreign saving is held fixed (see
        :mod:`puremacro.trade._accounting`).

    Returns
    -------
    np.ndarray
        1D residual vector of length 2*ns*nc + 3*nc + (nc - 1).

    Raises
    ------
    TypeError
        If unknown keyword arguments are passed (a misspelled option such as
        ``fiscal_closur=`` would otherwise silently change the model solved), or
        a policy option misses its country or sector.
    """
    return _evaluate_equilibrium(
        x, calib, tau, tau_fd, tauf, tauf_fd,
        tau_a=tau_a, taufd_a=taufd_a, sigma=sigma, accounting=accounting,
        fiscal_closure=fiscal_closure, recycle_country=recycle_country,
        capacity_margins=capacity_margins, capacity_country=capacity_country,
        capacity_sector=capacity_sector, foreign_saving_units=foreign_saving_units, **kwargs,
    )["residuals"]


def _evaluate_equilibrium(
    x: np.ndarray,
    calib: TradeCalibrationResult,
    tau: np.ndarray | None = None,
    tau_fd: np.ndarray | None = None,
    tauf: np.ndarray | None = None,
    tauf_fd: np.ndarray | None = None,
    *,
    tau_a: np.ndarray | None = None,
    taufd_a: np.ndarray | None = None,
    sigma: float = 0.0,
    accounting: str = "consistent",
    fiscal_closure: str = "lump_sum",
    recycle_country: str | None = None,
    capacity_margins: dict[str, float] | np.ndarray | None = None,
    capacity_country: str | None = None,
    capacity_sector: str | None = None,
    foreign_saving_units: str = "numeraire",
    **kwargs: Any,
) -> dict[str, Any]:
    """Shared economic blocks for residuals and postprocessing."""
    if kwargs:
        raise TypeError(
            f"compute_equilibrium_residuals() got unexpected keyword argument(s): {sorted(kwargs)}"
        )
    check_accounting(accounting)
    check_policy_options(fiscal_closure, recycle_country, capacity_margins, capacity_country, capacity_sector)
    from ._accounting import evaluate
    return evaluate(x, calib, tau_a if tau_a is not None else tau,
                    taufd_a if taufd_a is not None else tau_fd, tauf, tauf_fd,
                    sigma=sigma, fiscal_closure=fiscal_closure,
                    recycling_params=_recycling_params(recycle_country),
                    capacity_margins=capacity_margins,
                    foreign_saving_units=foreign_saving_units)


def evaluate_equilibrium_residuals(
    x: np.ndarray,
    calib: TradeCalibrationResult,
    tau: np.ndarray | None = None,
    tau_fd: np.ndarray | None = None,
    tau_a: np.ndarray | None = None,
    taufd_a: np.ndarray | None = None,
    tauf: np.ndarray | None = None,
    tauf_fd: np.ndarray | None = None,
    **options: Any,
) -> np.ndarray:
    """:func:`compute_equilibrium_residuals` with ``tau_a``/``taufd_a`` as positional aliases."""
    actual_tau = tau_a if tau_a is not None else tau
    actual_tau_fd = taufd_a if taufd_a is not None else tau_fd
    return compute_equilibrium_residuals(
        x=x, calib=calib, tau=actual_tau, tau_fd=actual_tau_fd, tauf=tauf, tauf_fd=tauf_fd, **options,
    )


__all__ = [
    "EquilibriumVariables",
    "unpack_equilibrium_vector",
    "pack_equilibrium_vector",
    "compute_equilibrium_residuals",
    "evaluate_equilibrium_residuals",
]
