"""Hansen-Singleton (1982) power-utility consumption CAPM by GMM.

References
----------
Hansen, L.P. and Singleton, K.J. (1982). Generalized instrumental variables
    estimation of nonlinear rational expectations models. Econometrica 50(5),
    1269-1286.
Stock, J.H. and Wright, J.H. (2000). GMM with weak identification.
    Econometrica 68(5), 1055-1096.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from ._results import CCAPMResult
from ._validate import _align, _validate_finite

__all__ = ["ccapm"]


class _Data(NamedTuple):
    R: np.ndarray        # (T-1, N) gross returns dated t+1
    g: np.ndarray        # (T-1,) gross consumption growth dated t+1
    Z: np.ndarray        # (T-1, L) instruments dated t
    asset_names: tuple
    index: Any


def _check_gross(df: pd.DataFrame, name: str) -> None:
    low = [str(c) for c in df.columns if float(df[c].mean()) < 0.5]
    if low:
        raise ValueError(
            f"{name}: column(s) {low} have mean < 0.5; ccapm needs gross values, "
            "pass gross returns, e.g. 1 + r")
    if np.any(df.to_numpy() <= 0):
        raise ValueError(f"{name}: gross values must be positive")


def _prepare(gross_returns: Any, consumption_growth: Any, instruments: Any = None,
             rf: Any = None) -> _Data:
    """Validate, align and lag: row t pairs instruments at t with R, g at t+1."""
    R = _validate_finite(gross_returns, "gross_returns")
    g = _validate_finite(consumption_growth, "consumption_growth", prefix="g")
    if g.shape[1] != 1:
        raise ValueError("consumption_growth must be a single series")
    _check_gross(R, "gross_returns")
    _check_gross(g, "consumption_growth")
    frames = [R, g]
    names = ["gross_returns", "consumption_growth"]
    if rf is not None:
        rf_df = _validate_finite(rf, "rf", prefix="rf")
        if rf_df.shape[1] != 1:
            raise ValueError("rf must be a single series")
        _check_gross(rf_df, "rf")
        rf_df.columns = ["rf"]
        frames.append(rf_df)
        names.append("rf")
    if instruments is not None:
        Zdf = _validate_finite(instruments, "instruments", prefix="z")
        frames.append(Zdf)
        names.append("instruments")
    frames = list(_align(*frames, names=tuple(names)))
    R, g = frames[0], frames[1]
    if rf is not None:
        R = pd.concat([R, frames[2]], axis=1)
    Rv = R.to_numpy(dtype=float)
    gv = g.iloc[:, 0].to_numpy(dtype=float)
    T = Rv.shape[0]
    if instruments is None:
        Zv = np.column_stack([np.ones(T), gv, frames[0].to_numpy(dtype=float)])
    else:
        Zv = frames[-1].to_numpy(dtype=float)
    if T < 3:
        raise ValueError("ccapm: need at least 3 periods")
    return _Data(R=Rv[1:], g=gv[1:], Z=Zv[:-1],
                 asset_names=tuple(str(c) for c in R.columns), index=R.index[1:])


def _moments(theta: np.ndarray, data: _Data) -> np.ndarray:
    """``(beta g_{t+1}^-gamma R_{i,t+1} - 1) * z_t``, asset-major, (T-1, N L)."""
    beta, gamma = theta
    u = beta * data.g[:, None] ** (-gamma) * data.R - 1.0          # (T, N)
    T = u.shape[0]
    return (u[:, :, None] * data.Z[:, None, :]).reshape(T, -1)


def _jacobian(theta: np.ndarray, data: _Data) -> np.ndarray:
    beta, gamma = theta
    m = data.g[:, None] ** (-gamma) * data.R                       # (T, N)
    T = m.shape[0]
    d_beta = (m[:, :, None] * data.Z[:, None, :]).reshape(T, -1).mean(axis=0)
    dm = -beta * np.log(data.g)[:, None] * m
    d_gamma = (dm[:, :, None] * data.Z[:, None, :]).reshape(T, -1).mean(axis=0)
    return np.column_stack([d_beta, d_gamma])


def ccapm(
    gross_returns: Any,
    consumption_growth: Any,
    *,
    instruments: Any = None,
    rf: Any = None,
    theta0: tuple[float, float] = (0.99, 2.0),
    weighting: str | np.ndarray = "two_step",
    hac_lags: int | None = None,
) -> CCAPMResult:
    """Power-utility Euler equations ``E[(beta g^-gamma R_i - 1) z_t] = 0``.

    Parameters
    ----------
    gross_returns : DataFrame, Series or array (T, N)
        Gross returns ``1 + r`` realized at ``t`` (from ``t-1`` to ``t``).
    consumption_growth : Series or array (T,)
        Gross consumption growth ``C_t / C_{t-1}``, same dating (see
        :func:`puremacro.asset_pricing.fred_consumption_growth`).
    instruments : DataFrame or array (T, L), optional
        Instruments dated ``t`` (known when the ``t+1`` return is priced);
        include a column of ones for the unconditional moments. Default
        ``z_t = [1, g_t, R_{1,t}, ..., R_{N,t}]``. Either way one period is
        lost to the lag.
    rf : Series or array (T,), optional
        Gross risk-free return, appended as an extra asset (not an
        instrument).
    theta0, weighting, hac_lags
        Start values for ``(beta, gamma)``, and :func:`puremacro.inference.gmm`
        options.

    Notes
    -----
    ``gamma`` is weakly identified in post-war consumption data (Stock and
    Wright 2000): report the J test and the confidence intervals, not the
    point estimate alone. Net returns (mean < 0.5) raise ``ValueError``.
    """
    from ..inference import gmm

    data = _prepare(gross_returns, consumption_growth, instruments, rf)
    th0 = np.asarray(theta0, dtype=float)
    if th0.shape != (2,):
        raise ValueError("theta0 must be (beta, gamma)")
    fit = gmm(_moments, th0, data, weighting=weighting, hac_lags=hac_lags,
              jacobian=_jacobian, names=("beta", "gamma"))
    beta, gamma = fit.coefs
    pe = (beta * data.g[:, None] ** (-gamma) * data.R - 1.0).mean(axis=0)
    return CCAPMResult(
        coefs=fit.coefs.copy(),
        se=fit.se.copy(),
        cov=fit.cov.copy(),
        names=("beta", "gamma"),
        n_obs=fit.n_obs,
        converged=fit.converged,
        j_stat=fit.j_stat,
        j_df=fit.j_df,
        j_pvalue=fit.j_pvalue,
        pricing_errors=pe,
        asset_names=data.asset_names,
        gmm=fit,
    )
