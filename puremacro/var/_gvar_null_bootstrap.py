"""Restricted independent-country bootstrap reference for GVAR diagnostics.

This does not impose the general weak-exogeneity null. Independence between
country innovation processes and absence of cross-country lag dynamics are
stronger assumptions. Re-estimating every VARX* in every draw accounts for the
generated residuals under this specified reference DGP.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .estimate import companion


@dataclass(frozen=True, eq=False)
class GVARIndependentNullResult:
    """Bootstrap reference under independent country processes, not general WE.

    ``table`` retains the observed residual-augmented F statistics but uses
    ``p_independent_null`` for this stronger null. ``p_holm_independent_null``
    covers all rows in the table. Monte Carlo errors concern a fixed fitted
    null, not uncertainty about the null DGP or validity of its assumptions.
    """

    table: pd.DataFrame
    statistic_draws: np.ndarray
    metadata: dict[str, Any]

    def to_frame(self) -> pd.DataFrame:
        return self.table.copy()

    def summary(self) -> str:
        return (
            "GVAR independent-country null bootstrap reference\n"
            "This is NOT the general weak-exogeneity or VECX* error-correction test.\n"
            f"{self.metadata['n_boot']} draws; {self.metadata['integration']} country processes; "
            f"innovation block length {self.metadata['block_length']}; "
            "all country VARX* and auxiliary regressions refitted in every draw."
        )


def _integer(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"independent_null_bootstrap: {name} must be an integer >= {minimum}")
    return int(value)


def _country_null(series: np.ndarray, lags: int, intercept: bool) -> dict[str, Any]:
    """Unrestricted own-country stationary VAR, with no stability repair."""
    from .gvar import _multivariate_ols

    T, k = series.shape
    if T - lags <= lags * k + int(intercept) + k:
        raise ValueError("independent_null_bootstrap: insufficient observations for country null VAR")
    parts = [series[lags - lag:T - lag] for lag in range(1, lags + 1)]
    if intercept:
        parts.insert(0, np.ones((T - lags, 1)))
    X = np.column_stack(parts)
    fit = _multivariate_ols(X, series[lags:], name="independent-country null VAR")
    B = fit["coef"]
    A = [B[int(intercept) + lag * k:int(intercept) + (lag + 1) * k].T for lag in range(lags)]
    root = float(np.max(np.abs(np.linalg.eigvals(companion(A)))))
    if not np.isfinite(root) or root >= 1:
        raise ValueError("independent_null_bootstrap: fitted country null VAR is not stationary; no roots clipped")
    resid = fit["resid"].copy()
    resid -= resid.mean(axis=0)
    return {"B": B, "residuals": resid, "initial": series[:lags].copy(), "root": root,
            "lags": lags, "intercept": intercept}


def _draw_country(null: dict[str, Any], n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    p = null["lags"]
    residuals = null["residuals"]
    count = n - p
    starts = rng.integers(len(residuals), size=(count + block_length - 1) // block_length)
    idx = ((starts[:, None] + np.arange(block_length)) % len(residuals)).ravel()[:count]
    path = np.empty((n, residuals.shape[1]))
    path[:p] = null["initial"]
    for t in range(p, n):
        parts = [path[t - lag] for lag in range(1, p + 1)]
        if null["intercept"]:
            parts.insert(0, np.ones(1))
        path[t] = np.concatenate(parts) @ null["B"] + residuals[idx[t - p]]
    return path


def independent_null_bootstrap(
    result: Any, *, null: str, integration: str, n_boot: int,
    block_length: int, null_lags: int = 1, seed: int | None = None,
    alpha: float = 0.05,
) -> GVARIndependentNullResult:
    """Simulate an explicit independent-country reference, then refit both stages.

    ``null`` must be ``'independent_countries'``. ``integration`` must be
    ``'stationary'`` (VAR in levels) or ``'i1'`` (VAR in differences,
    cumulatively integrated with observed initial level). No unit-root
    pretest chooses between them; cointegrated or mixed-order country blocks
    are outside the I(1) mode. Fixed original lag orders and auxiliary lags
    are reused; data-driven lag selection is not repeated.
    """
    from .gvar import _fit_blocks, _holm, _weak_exogeneity_test

    if null != "independent_countries":
        raise ValueError("independent_null_bootstrap supports only null='independent_countries', not general weak exogeneity")
    if integration not in ("stationary", "i1"):
        raise ValueError("integration must be explicitly 'stationary' or 'i1'")
    n_boot = _integer(n_boot, "n_boot", 19)
    block_length = _integer(block_length, "block_length", 1)
    null_lags = _integer(null_lags, "null_lags", 1)
    if seed is not None:
        seed = _integer(seed, "seed", 0)
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and strictly between zero and one")
    spec = result.spec
    diagnostic = result.weak_exogeneity
    if spec is None or diagnostic is None:
        raise ValueError("independent_null_bootstrap requires a fitted GVAR with weak_exogeneity=True")
    if spec.exog is not None or spec.trend not in ("n", "c"):
        raise ValueError("independent-country reference supports no exog and trend='n' or 'c' only")
    if diagnostic.skipped or diagnostic.table.empty:
        raise ValueError("independent_null_bootstrap requires all auxiliary regressions to be estimable")
    X = result.panel.to_numpy(dtype=float, copy=True)
    if X.shape != (spec.T, spec.k) or not np.isfinite(X).all():
        raise ValueError("independent_null_bootstrap: invalid retained global panel")
    fitted = {}
    for country in spec.countries:
        own = X[:, spec.own_idx[country]]
        stationary = np.diff(own, axis=0) if integration == "i1" else own
        fitted[country] = _country_null(stationary, null_lags, spec.trend == "c")
        if len(fitted[country]["residuals"]) < 2 * block_length:
            raise ValueError("independent_null_bootstrap needs at least two innovation block lengths")
    # Recompute the observed F rather than trusting a caller-modified table.
    observed_blocks, _ = _fit_blocks(X, spec)
    observed = _weak_exogeneity_test(observed_blocks, spec, X, diagnostic.lags,
                                     diagnostic.transform, alpha, "independent_null_bootstrap", strict_rank=True)
    table = observed.table[["country", "star_variable", "F", "df_num", "df_den", "n_obs", "lags"]].copy()
    draws = np.empty((n_boot, len(table)))
    rng = np.random.default_rng(seed)
    for b in range(n_boot):
        simulated = np.empty_like(X)
        for country in spec.countries:
            own_idx = spec.own_idx[country]
            path = _draw_country(fitted[country], spec.T - int(integration == "i1"), block_length, rng)
            if integration == "i1":
                simulated[0, own_idx] = X[0, own_idx]
                simulated[1:, own_idx] = X[0, own_idx] + np.cumsum(path, axis=0)
            else:
                simulated[:, own_idx] = path
        try:
            blocks, _ = _fit_blocks(simulated, spec)
            reference = _weak_exogeneity_test(blocks, spec, simulated, diagnostic.lags,
                                               diagnostic.transform, alpha, "independent_null_bootstrap", strict_rank=True)
            stat = reference.table["F"].to_numpy()
            if stat.shape != (len(table),) or not np.isfinite(stat).all():
                raise ValueError("nonfinite or missing auxiliary statistic")
            draws[b] = stat
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise ValueError(f"independent_null_bootstrap draw {b} failed; no draws discarded: {exc}") from exc
    observed_F = table["F"].to_numpy()
    if not np.isfinite(observed_F).all():
        raise ValueError("independent_null_bootstrap: nonfinite observed auxiliary statistic")
    p = (1 + np.sum(draws >= observed_F, axis=0)) / (n_boot + 1)
    table["p_independent_null"] = p
    table["p_holm_independent_null"] = _holm(p)
    table["reject_independent_null"] = p < alpha
    table["critical_independent_null"] = np.quantile(draws, 1 - alpha, axis=0, method="higher")
    table["mc_se"] = np.sqrt(p * (1 - p) / (n_boot + 1))
    return GVARIndependentNullResult(table, draws, {
        "null": null, "method": "country_VAR_circular_residual_blocks_refit_both_stages",
        "integration": integration, "n_boot": n_boot, "block_length": block_length,
        "null_lags": null_lags, "seed": seed, "alpha": float(alpha),
        "original_lag_orders": dict(spec.lag_orders), "auxiliary_lags": dict(diagnostic.lags),
        "auxiliary_transform": diagnostic.transform, "country_null_max_roots": {
            c: fitted[c]["root"] for c in spec.countries},
        "country_draws_independent": True, "first_stage_refitted": True,
        "lag_selection_repeated": False, "draws_discarded": 0,
        "p_value_resolution": 1 / (n_boot + 1),
        "scope": "independent country processes; not general weak exogeneity or a VECX* test",
    })
