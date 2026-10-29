"""Risk statistics: Sharpe ratio with HAC standard errors, drawdowns, VaR/ES.

References
----------
Lo, A.W. (2002). The statistics of Sharpe ratios. Financial Analysts
    Journal 58(4), 36-52.
Kupiec, P.H. (1995). Techniques for verifying the accuracy of risk
    measurement models. Journal of Derivatives 3(2), 73-84.
Christoffersen, P.F. (1998). Evaluating interval forecasts. International
    Economic Review 39(4), 841-862.
Barone-Adesi, G., Giannopoulos, K. and Vosper, L. (1999). VaR without
    correlations for portfolios of derivative securities. Journal of Futures
    Markets 19(5), 583-602 (filtered historical simulation).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ._results import RiskSummaryResult, VaRBacktestResult
from ._validate import _align, _validate_finite, _validate_returns
from .returns import _check_ppy, excess_returns

__all__ = ["risk_summary", "value_at_risk", "expected_shortfall", "var_backtest"]


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
    check_scale: bool = True,
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
    check_scale : bool
        Refuse percent-scaled input (median ``|r| > 0.2`` or more than 5% of
        values above 1); ``False`` skips the check.

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
        x_in = excess_returns(_validate_returns(returns, "returns", check_scale), rf,
                              check_scale=check_scale)
    else:
        x_in = _validate_returns(returns, "returns", check_scale)
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


# ---------------------------------------------------------------------------
# Value at risk and expected shortfall
# ---------------------------------------------------------------------------
_VAR_METHODS = ("historical", "normal", "garch")


def _check_alpha(alpha: Any) -> float:
    a = float(alpha)
    if not 0.0 < a < 1.0:
        raise ValueError(f"alpha must lie in (0, 1); got {alpha!r}")
    return a


def _var_es_one(x: np.ndarray, alpha: float, method: str, horizon: int) -> tuple[float, float]:
    from scipy import stats

    if method == "historical":
        if horizon > 1:
            c = np.concatenate([[0.0], np.cumsum(x)])
            x = c[horizon:] - c[:-horizon]
        q = float(np.quantile(x, alpha, method="linear"))
        return -q, -float(x[x <= q].mean())
    if method == "normal":
        mu, sd = float(x.mean()), float(x.std(ddof=1))
        z = float(stats.norm.ppf(alpha))
        var = -(horizon * mu + np.sqrt(horizon) * sd * z)
        es = -(horizon * mu - np.sqrt(horizon) * sd * float(stats.norm.pdf(z)) / alpha)
        return float(var), float(es)
    # filtered historical simulation on a constant-mean GARCH(1,1)
    from ..garch import garch11_fit

    fit = garch11_fit(x, mean="constant")
    if not fit.converged:
        raise ValueError(
            "method='garch': the GARCH(1,1) fit did not converge, so its filtered "
            "volatility cannot be trusted; use method='historical' or 'normal', or "
            "check the series (length, scale, outliers)")
    mean = float(x.mean())
    eps = x - mean
    sig = np.asarray(fit.sigma, dtype=float)
    zres = eps / sig
    s_next = float(np.sqrt(fit.omega + fit.alpha * eps[-1] ** 2 + fit.beta * sig[-1] ** 2))
    qz = float(np.quantile(zres, alpha, method="linear"))
    return -(mean + s_next * qz), -(mean + s_next * float(zres[zres <= qz].mean()))


def _var_es(returns: Any, alpha: float, method: str, horizon: int, which: int,
            check_scale: bool = True):
    a = _check_alpha(alpha)
    if method not in _VAR_METHODS:
        raise ValueError(f"method must be one of {_VAR_METHODS}; got {method!r}")
    if isinstance(horizon, (bool, np.bool_)) or not isinstance(horizon, (int, np.integer)) \
            or horizon < 1:
        raise ValueError(f"horizon must be a positive integer; got {horizon!r}")
    if method == "garch" and horizon != 1:
        raise ValueError(
            "method='garch' supports horizon=1 only: a multi-step filtered "
            "historical simulation needs simulated paths (out of scope)")
    df = _validate_returns(returns, "returns", check_scale)
    if len(df) - horizon + 1 < 2:
        raise ValueError(f"need more than {horizon} observations; got {len(df)}")
    vals = np.array([_var_es_one(df[c].to_numpy(), a, method, int(horizon))[which]
                     for c in df.columns])
    if isinstance(returns, pd.DataFrame) or np.ndim(returns) == 2:
        return pd.Series(vals, index=df.columns, name="VaR" if which == 0 else "ES")
    return float(vals[0])


def value_at_risk(returns: Any, alpha: float = 0.05, method: str = "historical",
                  horizon: int = 1, *, check_scale: bool = True):
    """Next-period value at risk, reported as a positive loss ``-q_alpha``.

    ``method="historical"``: the empirical ``alpha`` quantile
    (``np.quantile(method="linear")``) of overlapping ``horizon``-period sums
    of returns. ``"normal"``: ``-(h mu + sqrt(h) sigma Phi^-1(alpha))`` with
    the sample mean and ``std(ddof=1)``. ``"garch"`` (``horizon=1`` only):
    filtered historical simulation, ``-(mean + sigma_{T+1} q_alpha(z))`` with
    ``z`` the standardized residuals of a constant-mean
    :func:`puremacro.garch.garch11_fit` and
    ``sigma_{T+1}^2 = omega + alpha eps_T^2 + beta sigma_T^2``. A GARCH fit
    that did not converge raises ``ValueError`` rather than returning a VaR
    built on its volatility.

    A Series (or 1-D array) gives a float, a DataFrame one value per column.
    ``check_scale=False`` skips the percent-scale check.
    """
    return _var_es(returns, alpha, method, horizon, 0, check_scale)


def expected_shortfall(returns: Any, alpha: float = 0.05, method: str = "historical",
                       horizon: int = 1, *, check_scale: bool = True):
    """Next-period expected shortfall ``-E[r | r <= q_alpha]`` (positive loss).

    Same methods and conventions as :func:`value_at_risk`; ``"normal"`` uses
    ``-(h mu - sqrt(h) sigma phi(z) / alpha)``, ``z = Phi^-1(alpha)``, and
    ``"garch"`` the mean of the standardized-residual tail.
    """
    return _var_es(returns, alpha, method, horizon, 1, check_scale)


def _xlogy(x: float, p: float) -> float:
    """``x log p`` with ``0 log 0 = 0``."""
    return 0.0 if x == 0 else float(x * np.log(p))


def var_backtest(returns: Any, var: Any, alpha: float = 0.05, *,
                 check_scale: bool = True) -> VaRBacktestResult:
    """Kupiec unconditional-coverage and Christoffersen independence tests.

    A violation is ``r_t < -var_t`` (``var`` positive, as returned by
    :func:`value_at_risk`; a scalar is broadcast, a series is paired with
    ``returns`` by date). ``lr_uc`` (Kupiec 1995) is ``chi2(1)``; ``lr_ind``
    (Christoffersen 1998, first-order Markov alternative) is ``chi2(1)``;
    ``lr_cc = lr_uc + lr_ind`` is ``chi2(2)``. ``0 log 0 = 0`` throughout, so
    samples with no violations (or no violation-to-violation transitions)
    give finite statistics.
    """
    from scipy import stats

    a = _check_alpha(alpha)
    r = _validate_returns(returns, "returns", check_scale)
    if r.shape[1] != 1:
        raise ValueError(f"var_backtest: returns must be a single series; got {r.shape[1]} columns")
    if np.ndim(var) == 0 and not isinstance(var, (pd.Series, pd.DataFrame)):
        v = pd.DataFrame({"var": np.full(len(r), float(var))}, index=r.index)
        if not np.isfinite(float(var)):
            raise ValueError("var: NaN or inf")
    else:
        v = _validate_finite(var, "var")
        if v.shape[1] != 1:
            raise ValueError("var must be a scalar or a single series")
        r, v = _align(r, v, names=("returns", "var"))
    x = (r.iloc[:, 0].to_numpy() < -v.iloc[:, 0].to_numpy()).astype(int)
    T = x.size
    if T < 2:
        raise ValueError("var_backtest: need at least 2 observations")
    n1 = int(x.sum())
    n0 = T - n1
    pi_hat = n1 / T
    lr_uc = -2.0 * (_xlogy(n0, 1 - a) + _xlogy(n1, a)
                    - _xlogy(n0, 1 - pi_hat) - _xlogy(n1, pi_hat))
    prev, cur = x[:-1], x[1:]
    n00 = int(np.sum((prev == 0) & (cur == 0)))
    n01 = int(np.sum((prev == 0) & (cur == 1)))
    n10 = int(np.sum((prev == 1) & (cur == 0)))
    n11 = int(np.sum((prev == 1) & (cur == 1)))
    p01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    p11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    p = (n01 + n11) / (T - 1)
    l0 = _xlogy(n00 + n10, 1 - p) + _xlogy(n01 + n11, p)
    l1 = (_xlogy(n00, 1 - p01) + _xlogy(n01, p01)
          + _xlogy(n10, 1 - p11) + _xlogy(n11, p11))
    lr_ind = max(-2.0 * (l0 - l1), 0.0)
    lr_uc = max(lr_uc, 0.0)
    lr_cc = lr_uc + lr_ind
    return VaRBacktestResult(
        n_obs=int(T),
        alpha=a,
        n_violations=n1,
        violation_rate=float(pi_hat),
        lr_uc=float(lr_uc),
        p_uc=float(stats.chi2.sf(lr_uc, 1)),
        lr_ind=float(lr_ind),
        p_ind=float(stats.chi2.sf(lr_ind, 1)),
        lr_cc=float(lr_cc),
        p_cc=float(stats.chi2.sf(lr_cc, 2)),
    )
