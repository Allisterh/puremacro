"""Thin wrapper: estimate Smets-Wouters (2007) via the generic
puremacro.dsge.estimate_dsge engine.

The model-specific bits — bundled-data loading, OBSERVED_VARS
validation, the fixed (non-estimated) calibrated parameters, and the
initial-params construction from SW07_POSTERIOR_MODE + SW07_SHOCK_STDS —
live here. Everything else (mode refinement, Hessian, proposal-cov
construction, MH chains) is in puremacro.dsge.estimate.
"""
from __future__ import annotations

import importlib.resources
from typing import Optional

import pandas as pd

from puremacro.dsge._results import DSGEPosteriorResult
from puremacro.dsge.estimate import estimate_dsge
from puremacro.dsge.smets_wouters import SW07_POSTERIOR_MODE, SW07_SHOCK_STDS
from puremacro.dsge.sw07_observation import OBSERVED_VARS, make_state_space
from puremacro.dsge.sw07_priors import PRIORS


# Calibrated (NOT estimated) SW07 parameters; merged into every observation_eq
# call. These appear in SW07_POSTERIOR_MODE but NOT in PRIORS.
_FIXED_PARAMS = {
    "ctou":     0.025,
    "clandaw":  1.5,
    "cg":       0.18,
    "curvp":    10.0,
    "curvw":    10.0,
}


def _load_bundled_data() -> pd.DataFrame:
    """The bundled SW07 observables, 1966Q1-2004Q4 (156 quarters).

    Built by ``tools/build_sw07_data.py`` from FRED following the SW07 data
    appendix (ECB WP 722, printed p. 47): per-capita real GDP, consumption
    and investment growth and real-wage growth in percent (100 x log
    differences), ``log_hours`` = 100 x log of per-capita hours (NFB average
    hours x civilian employment / population 16+, demeaned), ``infl`` = 100 x
    log difference of the GDP deflator, ``ffr`` = federal funds rate / 4. The
    index is the first day of each quarter.
    """
    pkg = importlib.resources.files("puremacro.dsge")
    with (pkg / "_sw07_data.csv").open("r", encoding="utf-8") as fh:
        df = pd.read_csv(fh, comment="#", index_col="date")
    # "1966Q1" labels -> first day of the quarter (no dateutil fallback warning).
    idx = pd.PeriodIndex(df.index, freq="Q").to_timestamp()
    df.index = pd.DatetimeIndex(idx.to_numpy(), name="date")
    return df


def _validate_data(df: pd.DataFrame) -> None:
    missing = set(OBSERVED_VARS) - set(df.columns)
    if missing:
        raise ValueError(f"data missing columns: {sorted(missing)}")
    if len(df) < 50:
        raise ValueError(f"data has only {len(df)} obs; need >= 50")
    if df[list(OBSERVED_VARS)].isna().any().any():
        raise ValueError("data contains NaN values")


def estimate_sw07(
    data: Optional[pd.DataFrame] = None,
    *,
    n_draws: int = 10_000,
    n_chains: int = 2,
    burn_in: int = 2_000,
    seed: int = 0,
) -> DSGEPosteriorResult:
    """Bayesian estimation of Smets-Wouters (2007) via Random-Walk MH.

    Thin wrapper over :func:`puremacro.dsge.estimate.estimate_dsge`.

    Parameters
    ----------
    data : DataFrame with columns OBSERVED_VARS; if None, loads the
        bundled 1966Q1-2004Q4 US dataset (156 quarterly obs x 7 cols, SW07
        data-appendix definitions; see ``_load_bundled_data``). User data
        must use the same units: growth rates and hours in 100 x log,
        inflation and the interest rate in quarterly percent.
    n_draws, n_chains, burn_in, seed : MCMC controls.
    """
    df = _load_bundled_data() if data is None else data.copy()
    _validate_data(df)
    initial_params = {**SW07_POSTERIOR_MODE, **SW07_SHOCK_STDS}
    return estimate_dsge(
        df,
        observation_eq=make_state_space,
        priors=PRIORS,
        observed_vars=list(OBSERVED_VARS),
        initial_params=initial_params,
        fixed_params=_FIXED_PARAMS,
        model_name="SW07",
        n_draws=n_draws, n_chains=n_chains, burn_in=burn_in, seed=seed,
    )


__all__ = ["estimate_sw07"]
