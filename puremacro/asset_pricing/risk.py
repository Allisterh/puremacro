"""Risk statistics: Sharpe ratio with HAC standard errors, drawdowns, VaR/ES.

References
----------
Lo, A.W. (2002). The statistics of Sharpe ratios. Financial Analysts
    Journal 58(4), 36-52.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ._results import RiskSummaryResult
from ._validate import _validate_returns
from .returns import _check_ppy, excess_returns

__all__ = ["risk_summary"]


def _drawdown(x: np.ndarray) -> tuple[float, int]:
    """Max drawdown (positive fraction) and longest run below the running peak."""
    # Log wealth avoids overflow on long samples; a return <= -1 wipes the
    # position out (log wealth -inf, drawdown 1) from then on.
    with np.errstate(divide="ignore"):
        logw = np.concatenate([[0.0], np.cumsum(np.log1p(np.maximum(x, -1.0)))])
    peak = np.maximum.accumulate(logw)
    with np.errstate(invalid="ignore"):
        dd = np.where(np.isneginf(logw), 1.0, -np.expm1(logw - peak))
    under = logw < peak
    longest = run = 0
    for u in under:
        run = run + 1 if u else 0
        longest = max(longest, run)
    return float(dd.max()), int(longest)


def _sharpe_se(x: np.ndarray, lags: int) -> float:
    """Per-period se of ``mu / sigma`` by the GMM delta method (Lo 2002)."""
    from ..inference.gmm import _long_run_cov

    T = x.size
    mu = x.mean()
    d = x - mu
    var = np.mean(d ** 2)
    sd = np.sqrt(var)
    h = np.column_stack([d, d ** 2 - var])
    S = _long_run_cov(h, lags)
    grad = np.array([1.0 / sd, -mu / (2.0 * sd ** 3)])
    return float(np.sqrt(max(grad @ S @ grad, 0.0) / T))


def risk_summary(
    returns: Any,
    *,
    rf: Any = None,
    periods_per_year: float,
    hac_lags: int | None = None,
) -> RiskSummaryResult:
    """Per-column risk statistics with a HAC standard error for the Sharpe ratio.

    Parameters
    ----------
    returns : Series, DataFrame or array
        Per-period net returns in decimals.
    rf : scalar or Series, optional
        Per-period risk-free rate; subtracted (pairing by date) before every
        statistic, so ``mean``, ``sharpe`` and ``sortino`` are for excess
        returns.
    periods_per_year : float
        12 for monthly data, 4 quarterly, 252 daily. Required.
    hac_lags : int, optional
        Bartlett bandwidth for the Sharpe standard error; ``None`` uses
        ``floor(4 (T/100)^(2/9))`` and ``0`` the iid case.

    Notes
    -----
    ``sharpe = mean / std(ddof=0) * sqrt(ppy)``. Its standard error is the
    GMM delta-method se of Lo (2002) from the moments
    ``(x - mu, (x - mu)^2 - sigma^2)`` with a Bartlett long-run covariance,
    scaled by ``sqrt(ppy)``. With ``hac_lags=0`` it equals
    ``sqrt((1 - SR skew + SR^2 (kurt - 1) / 4) / T)``. Annualizing by
    ``sqrt(ppy)`` ignores Lo's serial-correlation factor eta(q): with
    autocorrelated returns the annualized Sharpe ratio itself is biased even
    though its per-period se is HAC-robust. ``sortino`` uses the downside
    deviation ``sqrt(mean(min(x, 0)^2))`` (target 0). ``max_drawdown`` is a
    positive fraction of the running peak of ``cumprod(1 + x)`` and
    ``max_drawdown_duration`` the longest run of periods below that peak.
    """
    from ..inference.gmm import _resolve_lags

    ppy = _check_ppy(periods_per_year)
    if rf is not None:
        x_in = excess_returns(_validate_returns(returns, "returns"), rf)
    else:
        x_in = _validate_returns(returns, "returns")
    X = x_in.to_numpy(dtype=float)
    T, N = X.shape
    if T < 3:
        raise ValueError(f"risk_summary: need at least 3 periods; got {T}")
    lags = _resolve_lags(hac_lags, T)
    mean = X.mean(axis=0)
    sd0 = X.std(axis=0, ddof=0)
    if np.any(sd0 == 0):
        bad = [str(c) for c, s in zip(x_in.columns, sd0) if s == 0]
        raise ValueError(f"risk_summary: constant return series {bad}")
    d = X - mean
    skew = np.mean(d ** 3, axis=0) / sd0 ** 3
    kurt = np.mean(d ** 4, axis=0) / sd0 ** 4 - 3.0
    downside = np.sqrt(np.mean(np.minimum(X, 0.0) ** 2, axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        sortino = np.where(downside > 0, mean / downside, np.inf) * np.sqrt(ppy)
    se = np.array([_sharpe_se(X[:, j], lags) for j in range(N)]) * np.sqrt(ppy)
    dds = [_drawdown(X[:, j]) for j in range(N)]
    return RiskSummaryResult(
        names=tuple(str(c) for c in x_in.columns),
        n_obs=int(T),
        periods_per_year=ppy,
        hac_lags=int(lags),
        mean=mean * ppy,
        volatility=X.std(axis=0, ddof=1) * np.sqrt(ppy),
        sharpe=mean / sd0 * np.sqrt(ppy),
        sharpe_se=se,
        sortino=sortino,
        skewness=skew,
        excess_kurtosis=kurt,
        max_drawdown=np.array([v[0] for v in dds]),
        max_drawdown_duration=np.array([v[1] for v in dds], dtype=int),
    )
