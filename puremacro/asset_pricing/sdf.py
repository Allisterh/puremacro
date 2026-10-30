"""Linear stochastic discount factor GMM and the Hansen-Jagannathan distance.

References
----------
Cochrane, J.H. (2005). Asset Pricing, revised edition. Princeton University
    Press, ch. 13.
Hansen, L.P. and Jagannathan, R. (1997). Assessing specification errors in
    stochastic discount factor models. Journal of Finance 52(2), 557-590.
Jagannathan, R. and Wang, Z. (1996). The conditional CAPM and the
    cross-section of expected returns. Journal of Finance 51(1), 3-53.
Kan, R. and Robotti, C. (2008). Specification tests of asset pricing models
    using excess returns. Journal of Empirical Finance 15(5), 816-838.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ._results import HJDistanceResult, SDFResult
from ._validate import _align, _validate_finite, _validate_returns

__all__ = ["sdf_gmm", "hj_distance"]

_STAGES = {"first": "identity", "two_step": "two_step", "iterated": "iterated"}


def _inputs(excess_returns: Any, factors: Any, check_scale: bool = True):
    Rdf = _validate_returns(excess_returns, "excess_returns", check_scale)
    Fdf = _validate_finite(factors, "factors", prefix="f")  # factors need not be returns
    if not hasattr(factors, "columns") and not hasattr(factors, "name"):
        Fdf.columns = [f"f{k}" for k in range(Fdf.shape[1])]
    Rdf, Fdf = _align(Rdf, Fdf, names=("excess_returns", "factors"))
    return Rdf, Fdf, Rdf.to_numpy(dtype=float), Fdf.to_numpy(dtype=float)


def _sym_sqrt(A: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(0.5 * (A + A.T))
    return (V * np.sqrt(np.clip(w, 0.0, None))) @ V.T


def _require_full_rank(A: np.ndarray, label: str) -> np.ndarray:
    """Check column rank without making identification depend on units."""
    scale = np.max(np.abs(A), axis=0)
    if np.any(scale == 0):
        raise ValueError(f"{label}: rank deficient; parameters not identified")
    normalized = A / scale
    if np.linalg.matrix_rank(normalized) < A.shape[1]:
        raise ValueError(f"{label}: rank deficient; parameters not identified")
    return scale


def sdf_gmm(
    excess_returns: Any,
    factors: Any,
    *,
    stage: str = "two_step",
    demeaned: bool = True,
    hac_lags: int | None = None,
    check_scale: bool = True,
) -> SDFResult:
    """GMM estimation of a linear SDF on excess returns, ``E[R_t m_t] = 0``.

    ``demeaned=True`` (default): ``m_t = 1 - b'(f_t - mu)`` with ``mu``
    estimated jointly, moments ``[R_t m_t; f_t - mu]`` (``N + K`` moments,
    ``2K`` parameters). ``demeaned=False``: ``m_t = 1 - b'f_t`` (``N``
    moments). ``stage``: ``"first"`` (identity weighting), ``"two_step"`` or
    ``"iterated"``, all through :func:`puremacro.inference.gmm` with a
    Bartlett long-run covariance (``hac_lags``). Start values are the
    closed-form first-stage ``b`` with ``mu = f_bar``.

    ``lambdas = Sigma_f b`` (demeaned; ``Sigma_f`` the MLE factor covariance,
    treated as known in ``lambda_se``). For ``demeaned=False`` the implied
    premia are ``Sigma_f b / (1 - b' f_bar)``. ``pricing_errors`` are the
    pricing-moment means; the J test has ``N - K`` degrees of freedom.
    """
    from ..inference import gmm

    if stage not in _STAGES:
        raise ValueError(f"sdf_gmm: stage must be one of {tuple(_STAGES)}; got {stage!r}")
    Rdf, Fdf, R, F = _inputs(excess_returns, factors, check_scale)
    T, N = R.shape
    K = F.shape[1]
    if N <= K:
        raise ValueError(f"sdf_gmm: need N > K test assets; got N = {N}, K = {K}")
    fbar = F.mean(axis=0)
    Rbar = R.mean(axis=0)
    names = tuple(str(c) for c in Fdf.columns)
    if demeaned:
        D = R.T @ (F - fbar) / T
        scale = _require_full_rank(D, "sdf_gmm factor loadings")
        b0 = np.linalg.lstsq(D / scale, Rbar, rcond=None)[0] / scale

        def moments(th, data):
            Rv, Fv = data
            b, mu = th[:K], th[K:]
            m = 1.0 - (Fv - mu) @ b
            return np.column_stack([Rv * m[:, None], Fv - mu])

        def jac(th, data):
            Rv, Fv = data
            b, mu = th[:K], th[K:]
            top = np.hstack([-Rv.T @ (Fv - mu) / T, np.outer(Rv.mean(axis=0), b)])
            bot = np.hstack([np.zeros((K, K)), -np.eye(K)])
            return np.vstack([top, bot])

        theta0 = np.concatenate([b0, fbar])
        pnames = names + tuple(f"mu_{n}" for n in names)
    else:
        D = R.T @ F / T
        scale = _require_full_rank(D, "sdf_gmm factor loadings")
        b0 = np.linalg.lstsq(D / scale, Rbar, rcond=None)[0] / scale

        def moments(th, data):
            Rv, Fv = data
            return Rv * (1.0 - Fv @ th)[:, None]

        def jac(th, data):
            Rv, Fv = data
            return -Rv.T @ Fv / T

        theta0 = b0
        pnames = names
    # Pricing moments are small relative to the factor-mean moments. Tighten
    # stopping so the jointly estimated means do not distort the second-stage
    # weight; the direct BFGS reference checks parameter agreement to 1e-6.
    fit = gmm(moments, theta0, (R, F), weighting=_STAGES[stage], hac_lags=hac_lags,
              jacobian=jac, names=pnames, tol=1e-12)
    b = fit.coefs[:K]
    cov_b = fit.cov[:K, :K]
    fc = F - fbar
    Sigma_f = fc.T @ fc / T
    if demeaned:
        lambdas = Sigma_f @ b
        dlam = Sigma_f
        mu = fit.coefs[K:].copy()
    else:
        s = 1.0 - float(b @ fbar)
        lambdas = Sigma_f @ b / s
        dlam = Sigma_f / s + np.outer(Sigma_f @ b, fbar) / s ** 2
        mu = None
    lam_cov = dlam @ cov_b @ dlam.T
    return SDFResult(
        coefs=b.copy(),
        se=np.sqrt(np.clip(np.diag(cov_b), 0.0, None)),
        cov=cov_b.copy(),
        names=names,
        n_obs=int(T),
        converged=fit.converged,
        mu=mu,
        lambdas=lambdas,
        lambda_se=np.sqrt(np.clip(np.diag(lam_cov), 0.0, None)),
        j_stat=fit.j_stat,
        j_df=fit.j_df,
        j_pvalue=fit.j_pvalue,
        pricing_errors=fit.moment_means[:N].copy(),
        asset_names=tuple(str(c) for c in Rdf.columns),
        stage=stage,
        demeaned=bool(demeaned),
        gmm=fit,
    )


def hj_distance(
    excess_returns: Any,
    factors: Any,
    *,
    hac_lags: int | None = None,
    n_sim: int = 100_000,
    seed: int = 0,
    check_scale: bool = True,
) -> HJDistanceResult:
    """Hansen-Jagannathan (1997) distance of the linear SDF ``m = 1 - b'f``.

    With excess returns ``R``: ``W = (R'R/T)^-1``, ``D = R'f/T``,
    ``b = (D'WD)^-1 D'W R_bar``, pricing errors ``g = R_bar - D b`` and
    ``distance = sqrt(g'Wg)``. Under the null ``T distance^2`` is
    asymptotically ``sum_j lambda_j chi2_1`` (Jagannathan and Wang 1996), the
    ``lambda_j`` being the ``N - K`` nonzero eigenvalues of
    ``S^1/2 W^1/2 (I - P) W^1/2 S^1/2`` with
    ``P = W^1/2 D (D'WD)^-1 D' W^1/2`` and ``S`` the Bartlett long-run
    covariance of ``R_t m_t`` at ``b``; the p-value is simulated from
    ``n_sim`` draws with ``numpy.random.default_rng(seed)``. ``cov`` / ``se``
    are the fixed-``W`` GMM sandwich (:func:`puremacro.inference.gmm` with
    ``weighting=W``).

    With excess returns the SDF mean is not identified, so the distance
    depends on the normalization ``m = 1 - b'f`` (Kan and Robotti 2008).
    """
    from ..inference import gmm
    from ..inference.gmm import _long_run_cov, _resolve_lags

    if not isinstance(n_sim, (int, np.integer)) or n_sim < 100:
        raise ValueError(f"hj_distance: n_sim must be an integer >= 100; got {n_sim!r}")
    Rdf, Fdf, R, F = _inputs(excess_returns, factors, check_scale)
    T, N = R.shape
    K = F.shape[1]
    if N <= K:
        raise ValueError(f"hj_distance: need N > K test assets; got N = {N}, K = {K}")
    lags = _resolve_lags(hac_lags, T)
    _require_full_rank(R, "hj_distance test assets")
    W = np.linalg.inv(R.T @ R / T)
    W = 0.5 * (W + W.T)
    D = R.T @ F / T
    _require_full_rank(D, "hj_distance factor loadings")
    Rbar = R.mean(axis=0)
    DWD = D.T @ W @ D
    b = np.linalg.solve(DWD, D.T @ W @ Rbar)
    g = Rbar - D @ b
    distance = float(np.sqrt(max(g @ W @ g, 0.0)))

    def moments(th, data):
        Rv, Fv = data
        return Rv * (1.0 - Fv @ th)[:, None]

    def jac(th, data):
        Rv, Fv = data
        return -Rv.T @ Fv / T

    names = tuple(str(c) for c in Fdf.columns)
    fit = gmm(moments, b, (R, F), weighting=W, hac_lags=lags, jacobian=jac, names=names)
    h = R * (1.0 - F @ b)[:, None]
    S = _long_run_cov(h, lags)
    Wh = _sym_sqrt(W)
    Sh = _sym_sqrt(S)
    P = Wh @ D @ np.linalg.solve(DWD, D.T @ Wh)
    M = Sh @ Wh @ (np.eye(N) - P) @ Wh @ Sh
    eig = np.sort(np.linalg.eigvalsh(0.5 * (M + M.T)))[::-1][: N - K]
    rng = np.random.default_rng(seed)
    draws = rng.chisquare(1.0, size=(int(n_sim), N - K)) @ eig
    stat = T * distance ** 2
    pvalue = float(np.mean(draws >= stat))
    return HJDistanceResult(
        distance=distance,
        coefs=b,
        se=fit.se.copy(),
        cov=fit.cov.copy(),
        names=names,
        n_obs=int(T),
        converged=fit.converged,
        pvalue=pvalue,
        weights=eig,
        n_sim=int(n_sim),
        seed=int(seed),
        pricing_errors=g,
        asset_names=tuple(str(c) for c in Rdf.columns),
        hac_lags=int(lags),
    )
