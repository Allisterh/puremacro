"""Time-series (GRS, HAC Wald) and cross-sectional (Fama-MacBeth) factor tests.

References
----------
Gibbons, M.R., Ross, S.A. and Shanken, J. (1989). A test of the efficiency
    of a given portfolio. Econometrica 57(5), 1121-1152.
Campbell, J.Y., Lo, A.W. and MacKinlay, A.C. (1997). The Econometrics of
    Financial Markets. Princeton University Press, ch. 5.
Fama, E.F. and MacBeth, J.D. (1973). Risk, return, and equilibrium:
    empirical tests. Journal of Political Economy 81(3), 607-636.
Shanken, J. (1992). On the estimation of beta-pricing models. Review of
    Financial Studies 5(1), 1-33.
Cochrane, J.H. (2005). Asset Pricing, revised edition. Princeton University
    Press, ch. 12.
"""
from __future__ import annotations

from typing import Any

import numpy as np

import pandas as pd

from ._results import FamaMacBethResult, TimeSeriesTestResult
from ._validate import _align, _validate_finite, _validate_returns

__all__ = ["time_series_test", "fama_macbeth"]


def _first_pass(R: np.ndarray, F: np.ndarray):
    """Per-asset OLS of R on [1, F]: coefficients (N, K+1), residuals (T, N)."""
    T = R.shape[0]
    X = np.column_stack([np.ones(T), F])
    coef, *_ = np.linalg.lstsq(X, R, rcond=None)
    return X, coef.T, R - X @ coef


def time_series_test(
    excess_returns: Any,
    factors: Any,
    *,
    hac_lags: int | None = None,
    traded: bool = True,
) -> TimeSeriesTestResult:
    """Time-series regressions of excess returns on factors; test alpha = 0.

    Each asset's excess return is regressed by OLS on ``[1, f_t]``. Two
    tests of ``alpha = 0`` are reported:

    * GRS (Gibbons, Ross and Shanken 1989) in the Campbell-Lo-MacKinlay
      (1997) form ``(T-N-K)/N * alpha' Sigma^-1 alpha / (1 + mu' Omega^-1 mu)``
      with MLE ``Sigma = e'e/T`` and ``Omega`` (factor covariance), which is
      exactly ``F(N, T-N-K)`` under iid normal errors. It needs traded
      excess-return factors; ``traded=False`` reports ``NaN``.
    * A HAC Wald statistic ``alpha' V_aa^-1 alpha``, ``chi2(N)``, with ``V``
      the Bartlett sandwich of the stacked OLS moments ``e_t (x) [1, f_t]``
      (no degrees-of-freedom correction; ``hac_lags=None`` uses
      ``floor(4 (T/100)^(2/9))``, ``0`` the heteroskedasticity-robust case).

    ``se`` are the square roots of the diagonal of ``V`` (equal to statsmodels'
    OLS HAC standard errors). Rows of DataFrame inputs are paired by date.
    ``T <= N + K`` or a singular residual covariance raise ``ValueError``.
    """
    from scipy import stats

    from ..inference.gmm import _long_run_cov, _resolve_lags

    Rdf = _validate_returns(excess_returns, "excess_returns")
    Fdf = _validate_returns(factors, "factors")
    Rdf, Fdf = _align(Rdf, Fdf, names=("excess_returns", "factors"))
    R = Rdf.to_numpy(dtype=float)
    F = Fdf.to_numpy(dtype=float)
    T, N = R.shape
    K = F.shape[1]
    if T <= N + K:
        raise ValueError(
            f"time_series_test: need T > N + K for the GRS test; got T = {T}, "
            f"N = {N}, K = {K}")
    lags = _resolve_lags(hac_lags, T)
    X, coef, e = _first_pass(R, F)
    if np.linalg.matrix_rank(X) < K + 1:
        raise ValueError("time_series_test: the factors are collinear (with the constant)")
    Sigma = e.T @ e / T
    if np.linalg.matrix_rank(Sigma) < N:
        raise ValueError(
            "time_series_test: the residual covariance is singular (an asset is "
            "a combination of the factors and other assets)")
    alphas = coef[:, 0]

    # HAC sandwich of the stacked moments e_t (x) x_t (asset-major)
    p = K + 1
    g = (e[:, :, None] * X[:, None, :]).reshape(T, N * p)
    S = _long_run_cov(g, lags)
    Qinv = np.linalg.inv(X.T @ X / T)
    Binv = np.kron(np.eye(N), Qinv)
    V = Binv @ S @ Binv / T
    V = 0.5 * (V + V.T)
    se = np.sqrt(np.clip(np.diag(V), 0.0, None)).reshape(N, p)
    ia = np.arange(N) * p
    Vaa = V[np.ix_(ia, ia)]
    wald = float(alphas @ np.linalg.solve(Vaa, alphas))
    wald_p = float(stats.chi2.sf(wald, N))

    df2 = T - N - K
    if traded:
        mu = F.mean(axis=0)
        fc = F - mu
        Omega = fc.T @ fc / T
        quad = float(alphas @ np.linalg.solve(Sigma, alphas))
        denom = 1.0 + float(mu @ np.linalg.solve(Omega, mu))
        grs = df2 / N * quad / denom
        grs_p = float(stats.f.sf(grs, N, df2))
    else:
        grs, grs_p = float("nan"), float("nan")

    yc = R - R.mean(axis=0)
    r2 = 1.0 - np.sum(e ** 2, axis=0) / np.sum(yc ** 2, axis=0)
    return TimeSeriesTestResult(
        coefs=coef,
        se=se,
        cov=V,
        names=tuple(str(c) for c in Rdf.columns),
        factor_names=tuple(str(c) for c in Fdf.columns),
        n_obs=int(T),
        converged=True,
        grs_stat=float(grs),
        grs_pvalue=grs_p,
        grs_df=(int(N), int(df2)),
        wald_stat=wald,
        wald_pvalue=wald_p,
        wald_df=int(N),
        r2=r2,
        hac_lags=int(lags),
        traded=bool(traded),
    )


def fama_macbeth(
    excess_returns: Any,
    factors: Any = None,
    *,
    betas: Any = None,
    shanken: bool = True,
    hac_lags: int | None = None,
    intercept: bool = True,
) -> FamaMacBethResult:
    """Fama-MacBeth (1973) two-pass cross-sectional regressions.

    Pass exactly one of ``factors`` (betas estimated by full-sample
    time-series OLS of each asset on ``[1, f_t]``) or ``betas`` ``(N, K)``.
    Each period's excess returns are regressed on ``Z = [1, beta]``
    (``intercept=True``) or ``beta``, giving ``lambda_t``; the estimate is
    their mean.

    Standard errors: ``se_fm = std(lambda_t, ddof=1) / sqrt(T)``;
    ``se_nw`` the Bartlett HAC long-run standard deviation of the demeaned
    ``lambda_t`` over ``sqrt(T)`` (``hac_lags=None`` uses
    ``floor(4 (T/100)^(2/9))``); ``se_shanken`` the Shanken (1992)
    errors-in-variables correction in the form of Cochrane (2005, eq. 12.19),
    ``(1 + c) A Sigma A' / T + Sigma_f / T`` with ``A = (Z'Z)^-1 Z'``,
    ``Sigma = e'e/T`` the first-pass residual covariance, ``Sigma_f`` the MLE
    factor covariance (zero on the intercept row/column) and
    ``c = lambda_f' Sigma_f^-1 lambda_f``. With ``betas`` given there are no
    first-pass residuals: ``se_shanken`` is ``NaN`` and ``se = se_fm``.
    ``se`` (and ``cov``) is Shanken when ``shanken=True`` and available,
    Fama-MacBeth otherwise.
    """
    from ..inference.gmm import _long_run_cov, _resolve_lags

    if (factors is None) == (betas is None):
        raise ValueError("fama_macbeth: pass exactly one of factors or betas")
    Rdf = _validate_returns(excess_returns, "excess_returns")
    Sigma = Sigma_f = None
    if factors is not None:
        Fdf = _validate_returns(factors, "factors")
        if not isinstance(factors, (pd.Series, pd.DataFrame)):
            Fdf.columns = [f"f{k}" for k in range(Fdf.shape[1])]
        Rdf, Fdf = _align(Rdf, Fdf, names=("excess_returns", "factors"))
        R = Rdf.to_numpy(dtype=float)
        F = Fdf.to_numpy(dtype=float)
        T, N = R.shape
        K = F.shape[1]
        if T <= K + 1:
            raise ValueError(f"fama_macbeth: need T > K + 1 periods; got T = {T}")
        _, coef, e = _first_pass(R, F)
        B = coef[:, 1:]
        Sigma = e.T @ e / T
        fc = F - F.mean(axis=0)
        Sigma_f = fc.T @ fc / T
        fnames = tuple(str(c) for c in Fdf.columns)
    else:
        Bdf = _validate_finite(betas, "betas", prefix="f")
        if not isinstance(betas, pd.DataFrame):
            Bdf.columns = [f"f{k}" for k in range(Bdf.shape[1])]
        R = Rdf.to_numpy(dtype=float)
        T, N = R.shape
        B = Bdf.to_numpy(dtype=float)
        if B.shape[0] != N:
            raise ValueError(f"fama_macbeth: betas must have N = {N} rows; got {B.shape[0]}")
        K = B.shape[1]
        fnames = tuple(str(c) for c in Bdf.columns)
    Z = np.column_stack([np.ones(N), B]) if intercept else B
    p = Z.shape[1]
    if N <= p:
        raise ValueError(f"fama_macbeth: need N > {p} assets for the cross-section; got {N}")
    if np.linalg.matrix_rank(Z) < p:
        raise ValueError("fama_macbeth: the cross-sectional regressors [1, beta] are collinear")
    A = np.linalg.solve(Z.T @ Z, Z.T)                      # (p, N)
    lam_t = R @ A.T                                         # (T, p)
    lam = lam_t.mean(axis=0)
    d = lam_t - lam
    cov_fm = d.T @ d / (T - 1) / T
    se_fm = np.sqrt(np.diag(cov_fm))
    lags = _resolve_lags(hac_lags, T)
    cov_nw = _long_run_cov(d, lags) / T
    se_nw = np.sqrt(np.clip(np.diag(cov_nw), 0.0, None))
    if Sigma is not None:
        lam_f = lam[1:] if intercept else lam
        c = float(lam_f @ np.linalg.solve(Sigma_f, lam_f))
        pad = np.zeros((p, p))
        pad[p - K:, p - K:] = Sigma_f
        cov_sh = (1.0 + c) * (A @ Sigma @ A.T) / T + pad / T
        cov_sh = 0.5 * (cov_sh + cov_sh.T)
        se_sh = np.sqrt(np.diag(cov_sh))
    else:
        cov_sh = np.full((p, p), np.nan)
        se_sh = np.full(p, np.nan)
    use_sh = bool(shanken) and Sigma is not None
    mean_ret = R.mean(axis=0)
    pe = mean_ret - Z @ lam
    cs_r2 = float(1.0 - pe.var() / mean_ret.var())
    names = (("const",) if intercept else ()) + fnames
    return FamaMacBethResult(
        coefs=lam,
        se=se_sh if use_sh else se_fm,
        cov=cov_sh if use_sh else cov_fm,
        names=names,
        n_obs=int(T),
        converged=True,
        se_fm=se_fm,
        se_shanken=se_sh,
        se_nw=se_nw,
        lambda_series=lam_t,
        cs_r2=cs_r2,
        pricing_errors=pe,
        mean_returns=mean_ret,
        betas=B,
        asset_names=tuple(str(c) for c in Rdf.columns),
        hac_lags=int(lags),
        intercept=bool(intercept),
        se_type="shanken" if use_sh else "fama-macbeth",
    )
