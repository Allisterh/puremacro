"""Generic generalized method of moments (Hansen 1982).

:func:`gmm` estimates ``theta`` from user-supplied per-period moment
contributions ``g_t(theta)`` with ``E[g_t(theta_0)] = 0``. It is the one
optimizer-bearing core of the package's moment-based estimators
(:mod:`puremacro.asset_pricing` builds its SDF, Hansen-Jagannathan and
consumption-CAPM estimators on it).

Conventions
-----------
* Long-run covariance (uncentered, Newey-West Bartlett):
  ``S(theta) = G0 + sum_{l=1..L} (1 - l/(L+1)) (Gl + Gl')`` with
  ``Gl = g[l:]' g[:-l] / T``. ``hac_lags=None`` uses the package's default
  bandwidth ``floor(4 (T/100)^(2/9))``; ``hac_lags=0`` is the iid
  (heteroskedasticity-robust) case.
* Estimation minimizes ``gbar' W gbar`` with :func:`scipy.optimize.least_squares`
  on ``chol(W)' gbar``.
* Covariance: ``B G' W S(theta_hat) W G B / T`` with ``B = (G' W G)^-1``,
  ``W`` the last weight used and ``G`` the Jacobian of the moment means at
  ``theta_hat``. This is the efficient form ``(G' S^-1 G)^-1 / T`` when
  ``W = S^-1``, and the correct sandwich for any other ``W``.
* ``J = T gbar' W gbar`` is reported for the efficient weightings
  (``"two_step"``, ``"iterated"``), where it is asymptotically
  ``chi2(q - p)``; it is ``NaN`` for ``"identity"`` or a user matrix, and
  exactly zero (p-value ``NaN``) when the model is just identified.

References
----------
Hansen, L.P. (1982). Large sample properties of generalized method of
    moments estimators. Econometrica 50(4), 1029-1054.
Newey, W.K. and West, K.D. (1987). A simple, positive semi-definite,
    heteroskedasticity and autocorrelation consistent covariance matrix.
    Econometrica 55(3), 703-708.
Hall, A.R. (2005). Generalized Method of Moments. Oxford University Press.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
from scipy import optimize, stats

from ._results import GMMFitResult

__all__ = ["gmm"]

_STRING_WEIGHTINGS = ("two_step", "iterated", "identity")


def _check_hac_lags(value: Any, name: str = "hac_lags") -> int | None:
    """``None`` or a non-negative whole number; floats and bools refused."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be None or a non-negative integer; got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative; got {value}")
    return int(value)


def _resolve_lags(hac_lags: int | None, n_obs: int) -> int:
    from ..regress.ols import _default_maxlags

    lags = _check_hac_lags(hac_lags)
    return _default_maxlags(n_obs) if lags is None else lags


def _long_run_cov(g: np.ndarray, lags: int) -> np.ndarray:
    """Uncentered Bartlett long-run covariance of the rows of ``g`` (T, q)."""
    g = np.asarray(g, dtype=float)
    if g.ndim == 1:
        g = g[:, None]
    T = g.shape[0]
    S = g.T @ g / T
    for lag in range(1, min(lags, T - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        gam = g[lag:].T @ g[:-lag] / T
        S = S + w * (gam + gam.T)
    return S


def _numeric_jacobian(mean_fn: Callable[[np.ndarray], np.ndarray],
                      theta: np.ndarray) -> np.ndarray:
    """Central differences, step ``eps**(1/3) * max(|theta_j|, 1)``."""
    p = theta.size
    base = np.finfo(float).eps ** (1.0 / 3.0)
    cols = []
    for j in range(p):
        h = base * max(abs(theta[j]), 1.0)
        tp = theta.copy()
        tm = theta.copy()
        tp[j] += h
        tm[j] -= h
        cols.append((mean_fn(tp) - mean_fn(tm)) / (tp[j] - tm[j]))
    return np.column_stack(cols)


def _sym_inv(A: np.ndarray, what: str) -> np.ndarray:
    try:
        out = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        raise ValueError(f"gmm: {what} is singular") from None
    return 0.5 * (out + out.T)


def gmm(
    moments: Callable[[np.ndarray, Any], np.ndarray],
    theta0: Sequence[float] | np.ndarray,
    data: Any,
    *,
    weighting: str | np.ndarray = "two_step",
    w0: np.ndarray | None = None,
    hac_lags: int | None = None,
    kernel: str = "bartlett",
    jacobian: Callable[[np.ndarray, Any], np.ndarray] | None = None,
    names: Sequence[str] | None = None,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> GMMFitResult:
    """Generalized method of moments estimation.

    Parameters
    ----------
    moments : callable
        ``moments(theta, data) -> ndarray (T, q)``: per-period moment
        contributions whose expectation is zero at the true ``theta``.
    theta0 : array_like, shape (p,)
        Starting values.
    data : any
        Passed through to ``moments`` and ``jacobian``.
    weighting : {"two_step", "iterated", "identity"} or ndarray (q, q)
        ``"two_step"``: a first step with ``w0`` then ``W = S(theta_1)^-1``.
        ``"iterated"``: re-estimate with ``W = S(theta_k)^-1`` until
        ``max|theta_{k+1} - theta_k| < tol`` or ``max_iter`` weight updates.
        ``"identity"`` or a positive-definite matrix: one step.
    w0 : ndarray (q, q), optional
        First-step weight for ``"two_step"`` / ``"iterated"``; identity by
        default.
    hac_lags : int, optional
        Bartlett bandwidth of the long-run covariance; ``None`` uses
        ``floor(4 (T/100)^(2/9))``, ``0`` the iid case.
    kernel : {"bartlett"}
        Only the Bartlett kernel is implemented.
    jacobian : callable, optional
        ``jacobian(theta, data) -> ndarray (q, p)``: derivative of the moment
        *means*. Central differences when omitted.
    names : sequence of str, optional
        Parameter names; default ``("theta0", ..., "theta{p-1}")``.
    max_iter : int
        Maximum number of weight updates for ``"iterated"``.
    tol : float
        Optimizer tolerance (``xtol = ftol = gtol = tol``) and the iterated
        convergence criterion.

    Returns
    -------
    GMMFitResult
    """
    theta0 = np.atleast_1d(np.asarray(theta0, dtype=float)).ravel()
    p = theta0.size
    if p == 0 or not np.all(np.isfinite(theta0)):
        raise ValueError("gmm: theta0 must be a non-empty finite vector")
    if kernel != "bartlett":
        raise ValueError(f"gmm: only kernel='bartlett' is implemented; got {kernel!r}")
    lags_arg = _check_hac_lags(hac_lags)
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError(f"gmm: max_iter must be a positive integer; got {max_iter!r}")

    def _g(theta: np.ndarray) -> np.ndarray:
        return np.asarray(moments(theta, data), dtype=float)

    g0 = _g(theta0)
    if g0.ndim != 2:
        raise ValueError(
            f"gmm: moments(theta, data) must return a 2-D array (T, q); got "
            f"shape {g0.shape}")
    T, q = g0.shape
    if q < p:
        raise ValueError(f"gmm: fewer moments ({q}) than parameters ({p})")
    if not np.all(np.isfinite(g0)):
        raise ValueError("gmm: moments are non-finite at theta0")
    lags = _resolve_lags(lags_arg, T)

    if names is None:
        names = tuple(f"theta{j}" for j in range(p))
    else:
        names = tuple(str(n) for n in names)
        if len(names) != p:
            raise ValueError(f"gmm: {len(names)} names for {p} parameters")

    if isinstance(weighting, str):
        if weighting not in _STRING_WEIGHTINGS:
            raise ValueError(
                f"gmm: weighting must be one of {_STRING_WEIGHTINGS} or a (q, q) "
                f"array; got {weighting!r}")
        mode = weighting
        W_user = None
    else:
        W_user = np.asarray(weighting, dtype=float)
        if W_user.shape != (q, q) or not np.all(np.isfinite(W_user)):
            raise ValueError(
                f"gmm: weighting array must be finite with shape ({q}, {q}); got "
                f"{W_user.shape}")
        mode = "array"
    if w0 is None:
        W_first = np.eye(q)
    else:
        W_first = np.asarray(w0, dtype=float)
        if W_first.shape != (q, q) or not np.all(np.isfinite(W_first)):
            raise ValueError(f"gmm: w0 must be finite with shape ({q}, {q})")

    def gbar(theta: np.ndarray) -> np.ndarray:
        return _g(theta).mean(axis=0)

    if jacobian is None:
        def G_of(theta: np.ndarray) -> np.ndarray:
            return _numeric_jacobian(gbar, theta)
    else:
        def G_of(theta: np.ndarray) -> np.ndarray:
            G = np.asarray(jacobian(theta, data), dtype=float)
            if G.shape != (q, p):
                raise ValueError(
                    f"gmm: jacobian must return shape ({q}, {p}); got {G.shape}")
            return G

    def _minimize(W: np.ndarray, start: np.ndarray) -> tuple[np.ndarray, bool]:
        Wsym = 0.5 * (W + W.T)
        try:
            L = np.linalg.cholesky(Wsym)
        except np.linalg.LinAlgError:
            raise ValueError("gmm: weighting matrix is not positive definite") from None
        Lt = L.T
        sol = optimize.least_squares(
            lambda th: Lt @ gbar(th), start,
            jac=lambda th: Lt @ G_of(th),
            method="trf", xtol=tol, ftol=tol, gtol=tol, max_nfev=2000 * max(p, 1),
        )
        return np.asarray(sol.x, dtype=float), bool(sol.success)

    def _efficient_weight(theta: np.ndarray) -> np.ndarray:
        return _sym_inv(_long_run_cov(_g(theta), lags),
                        "the long-run moment covariance S(theta)")

    iterations = 0
    if mode in ("identity", "array"):
        W = np.eye(q) if mode == "identity" else W_user
        theta, converged = _minimize(W, theta0)
    else:
        theta, converged = _minimize(W_first, theta0)
        if mode == "two_step":
            W = _efficient_weight(theta)
            theta, ok = _minimize(W, theta)
            converged = converged and ok
            iterations = 1
        else:
            converged_iter = False
            for _ in range(int(max_iter)):
                W = _efficient_weight(theta)
                new, ok = _minimize(W, theta)
                iterations += 1
                step = float(np.max(np.abs(new - theta)))
                theta = new
                if step < tol:
                    converged_iter = True
                    break
            converged = converged and ok and converged_iter

    g_hat = _g(theta)
    if not np.all(np.isfinite(g_hat)):
        raise ValueError("gmm: moments are non-finite at the estimate")
    m_hat = g_hat.mean(axis=0)
    S = _long_run_cov(g_hat, lags)
    G = G_of(theta)
    Wsym = 0.5 * (W + W.T)
    B = _sym_inv(G.T @ Wsym @ G, "G' W G (parameters not identified)")
    meat = G.T @ Wsym @ S @ Wsym @ G
    cov = B @ meat @ B / T
    cov = 0.5 * (cov + cov.T)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))

    j_df = q - p
    if j_df == 0:
        j_stat, j_pvalue = 0.0, float("nan")
    elif mode in ("two_step", "iterated"):
        j_stat = float(T * m_hat @ Wsym @ m_hat)
        j_pvalue = float(stats.chi2.sf(j_stat, j_df))
    else:
        j_stat, j_pvalue = float("nan"), float("nan")

    return GMMFitResult(
        coefs=theta,
        se=se,
        cov=cov,
        names=names,
        n_obs=int(T),
        converged=bool(converged),
        j_stat=j_stat,
        j_df=int(j_df),
        j_pvalue=j_pvalue,
        weighting=mode,
        iterations=int(iterations),
        moment_means=m_hat,
        weight_matrix=Wsym,
        hac_lags=int(lags),
    )
