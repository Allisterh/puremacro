r"""Sequence-Space Whittle Likelihood & Minimum-Distance IRF Estimation for HANK/DSGE.

Provides two fast, zero-dependency estimation routes for sequence-space models
(`BlockDAG`, `HANKModel`, or moving-average representations $\{M_s\}_{s=0}^{T-1}$):

1. Frequency-Domain Whittle Gaussian Log-Likelihood (`hank_loglik_whittle`,
   `estimate_hank_whittle`):
   Transforms the sequence-space Wold moving-average representation
   $\{M_s\}_{s=0}^{T_{\text{ma}}-1}$ into the transfer function
   $$\mathcal{M}(\omega_k) = \sum_{s=0}^{T_{\text{ma}}-1} M_s e^{-i \omega_k s}$$
   via `numpy.fft.fft`, forms the spectral covariance matrix
   $$S(\omega_k) = \mathcal{M}(\omega_k) \Sigma_\varepsilon \mathcal{M}(\omega_k)^* + \Sigma_\eta,$$
   and evaluates the Whittle log-likelihood across Fourier frequencies
   $\omega_k = 2\pi k / T_{\text{obs}}$:
   $$\ln L_{\text{Whittle}}(Y \mid \theta) = -\frac{1}{2} \sum_{k=0}^{T_{\text{obs}}-1}
     \Bigl[ n_y \ln(2\pi) + \ln \det S(\omega_k) + d(\omega_k)^* S(\omega_k)^{-1} d(\omega_k) \Bigr].$$

2. Sequence-Space IRF Minimum-Distance Estimation (`estimate_hank_irf_matching`):
   Matches sequence-space impulse responses $G^{Y, Z}(\theta)$ against empirical
   LP/SVAR target trajectories with diagonal or full HAC weighting matrices.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

if TYPE_CHECKING:
    import matplotlib.figure

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.optimize

from puremacro.dsge.ssj_dag import BlockDAG
from puremacro.reports import _df_to_latex, _df_to_markdown, _df_to_typst

__all__ = [
    "HANKEstimationResult",
    "hank_loglik_whittle",
    "estimate_hank_whittle",
    "estimate_hank_irf_matching",
]


@dataclass(frozen=True)
class HANKEstimationResult:
    """Result container for sequence-space HANK/DSGE Whittle or IRF-matching estimation."""

    params: dict[str, float]
    se: dict[str, float]
    vcov: np.ndarray
    objective_value: float
    log_likelihood: float | None
    method: str
    converged: bool
    iterations: int
    fitted_irfs: pd.DataFrame | None = None
    target_irfs: pd.DataFrame | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for name, val in self.params.items():
            se_v = float(self.se.get(name, np.nan))
            t_stat = val / se_v if np.isfinite(se_v) and se_v > 0.0 else np.nan
            rows.append(
                {
                    "parameter": name,
                    "estimate": float(val),
                    "std_err": se_v,
                    "t_stat": t_stat,
                    "ci_lower_95": float(val - 1.959963984540054 * se_v) if np.isfinite(se_v) else np.nan,
                    "ci_upper_95": float(val + 1.959963984540054 * se_v) if np.isfinite(se_v) else np.nan,
                }
            )
        return pd.DataFrame(rows).set_index("parameter")

    def summary(self) -> pd.DataFrame:
        return self.to_dataframe()

    def to_markdown(self, **kwargs: Any) -> str:
        return _df_to_markdown(self.to_dataframe(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        return _df_to_latex(self.to_dataframe(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        return _df_to_typst(self.to_dataframe(), **kwargs)

    def plot(self, figsize: tuple[float, float] = (9.5, 4.2)) -> matplotlib.figure.Figure:
        import matplotlib.pyplot as plt

        if self.fitted_irfs is not None and self.target_irfs is not None:
            cols = list(self.fitted_irfs.columns[:4])
            fig, axes = plt.subplots(1, len(cols), figsize=figsize, constrained_layout=True)
            if len(cols) == 1:
                axes = [axes]
            for ax, col in zip(axes, cols):
                ax.plot(self.fitted_irfs.index, self.fitted_irfs[col], lw=2.0, label="Model SSJ")
                if col in self.target_irfs.columns:
                    ax.plot(self.target_irfs.index, self.target_irfs[col], ls="--", lw=1.8, label="Target")
                ax.axhline(0.0, color="gray", lw=0.8, ls=":")
                ax.set_title(col)
                ax.legend(frameon=False)
            return fig

        df = self.to_dataframe()
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        y_pos = np.arange(len(df))
        ax.errorbar(
            df["estimate"].to_numpy(),
            y_pos,
            xerr=1.96 * np.nan_to_num(df["std_err"].to_numpy(), nan=0.0),
            fmt="o",
            capsize=4,
        )
        ax.set_yticks(y_pos, list(df.index))
        ax.set_title(f"HANK Estimation ({self.method})")
        ax.set_xlabel("Parameter Estimate")
        return fig


def hank_loglik_whittle(
    ma_or_dag: np.ndarray | BlockDAG,
    data: np.ndarray | pd.DataFrame,
    shock_cov: np.ndarray,
    *,
    steady_state: Mapping[str, float] | None = None,
    params: Mapping[str, float] | None = None,
    observables: Sequence[str] | None = None,
    measurement_cov: np.ndarray | None = None,
    T_ma: int = 256,
    demean: bool = True,
    exclude_zero_freq: bool = False,
) -> float:
    """Evaluate exact frequency-domain Whittle Gaussian log-likelihood from SSJ MA matrices.

    Parameters
    ----------
    ma_or_dag : np.ndarray of shape (T_ma, n_obs, n_shocks) or BlockDAG
        Sequence-space moving-average array `M` or a `BlockDAG` from which `M`
        is extracted.
    data : array_like of shape (T_obs, n_obs)
        Observed time series.
    shock_cov : np.ndarray of shape (n_shocks, n_shocks)
        Structural innovation covariance matrix $\\Sigma_\\varepsilon$.
    steady_state, params : Mapping[str, float], optional
        Required when `ma_or_dag` is a `BlockDAG`.
    observables : Sequence[str], optional
        Observable variable names when `ma_or_dag` is a `BlockDAG`.
    measurement_cov : np.ndarray of shape (n_obs, n_obs), optional
        Measurement error covariance $\\Sigma_\\eta$.
    T_ma : int, default 256
        Truncation horizon when extracting MA matrices from a `BlockDAG`.
    demean : bool, default True
        Whether to demean `data` along the time dimension before FFT.
    exclude_zero_freq : bool, default False
        Whether to omit frequency $\\omega_0 = 0$ from the sum.

    Returns
    -------
    loglik : float
        Whittle Gaussian log-likelihood.
    """
    if isinstance(ma_or_dag, BlockDAG):
        if steady_state is None or params is None:
            raise ValueError("`steady_state` and `params` are required when `ma_or_dag` is a BlockDAG")
        M = ma_or_dag.moving_average_matrices(
            steady_state,
            params,
            observables=observables,
            T=int(T_ma),
        )
    else:
        M = np.asarray(ma_or_dag, dtype=np.float64)
        if M.ndim != 3:
            raise ValueError(f"`ma_or_dag` array must have shape (T_ma, n_obs, n_shocks), got {M.shape}")

    Y = np.asarray(data, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    T_obs, n_obs = Y.shape
    T_m, n_y_m, n_shocks = M.shape
    if n_obs != n_y_m:
        raise ValueError(f"Data has {n_obs} observables, but MA representation has {n_y_m}")

    Sigma_e = np.asarray(shock_cov, dtype=np.float64)
    if Sigma_e.ndim == 0:
        Sigma_e = np.array([[float(Sigma_e)]], dtype=np.float64)
    elif Sigma_e.ndim == 1:
        Sigma_e = np.diag(Sigma_e)
    if Sigma_e.shape != (n_shocks, n_shocks):
        raise ValueError(f"`shock_cov` shape {Sigma_e.shape} does not match n_shocks={n_shocks}")

    Sigma_eta = (
        np.zeros((n_obs, n_obs), dtype=np.float64)
        if measurement_cov is None
        else np.asarray(measurement_cov, dtype=np.float64)
    )

    Y_use = Y - np.mean(Y, axis=0, keepdims=True) if demean else Y
    # Unitary DFT of observations: shape (T_obs, n_obs)
    D_freq = np.fft.fft(Y_use, axis=0) / np.sqrt(float(T_obs))

    # Evaluate transfer function M(e^{-i omega_k}) at the exact T_obs Fourier frequencies
    if T_m <= T_obs:
        M_pad = np.zeros((T_obs, n_obs, n_shocks), dtype=np.float64)
        M_pad[:T_m] = M
        H_freq = np.fft.fft(M_pad, axis=0)
    else:
        # Fold lags modulo T_obs so all T_m coefficients contribute at omega_k = 2*pi*k/T_obs
        M_fold = np.zeros((T_obs, n_obs, n_shocks), dtype=np.float64)
        for s in range(T_m):
            M_fold[s % T_obs] += M[s]
        H_freq = np.fft.fft(M_fold, axis=0)

    start_k = 1 if exclude_zero_freq else 0
    total_ll = 0.0
    const_term = n_obs * np.log(2.0 * np.pi)
    eye_obs = np.eye(n_obs, dtype=np.complex128)

    for k in range(start_k, T_obs):
        Hk = H_freq[k]  # (n_obs, n_shocks)
        Sk = Hk @ Sigma_e @ Hk.conj().T + Sigma_eta
        Sk = 0.5 * (Sk + Sk.conj().T)
        dk = D_freq[k]
        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                sign, logdet = np.linalg.slogdet(Sk)
            if np.real(sign) <= 0.0 or not np.isfinite(logdet):
                Sk = Sk + 1e-12 * eye_obs
                with np.errstate(divide="ignore", invalid="ignore"):
                    sign, logdet = np.linalg.slogdet(Sk)
                if np.real(sign) <= 0.0 or not np.isfinite(logdet):
                    return -np.inf
            sol = np.linalg.solve(Sk, dk)
            quad = float(np.real(np.vdot(dk, sol)))
            total_ll += -0.5 * (const_term + float(np.real(logdet)) + quad)
        except np.linalg.LinAlgError:
            return -np.inf

    return float(total_ll)


def _numerical_hessian(fn: Callable[[np.ndarray], float], x: np.ndarray, step: float = 1e-4) -> np.ndarray:
    n = len(x)
    H = np.zeros((n, n), dtype=np.float64)
    f0 = fn(x)
    for i in range(n):
        ei = np.zeros(n, dtype=np.float64)
        ei[i] = step * max(1.0, abs(float(x[i])))
        fp = fn(x + ei)
        fm = fn(x - ei)
        H[i, i] = (fp - 2.0 * f0 + fm) / (ei[i] ** 2)
        for j in range(i + 1, n):
            ej = np.zeros(n, dtype=np.float64)
            ej[j] = step * max(1.0, abs(float(x[j])))
            fpp = fn(x + ei + ej)
            fpm = fn(x + ei - ej)
            fmp = fn(x - ei + ej)
            fmm = fn(x - ei - ej)
            hij = (fpp - fpm - fmp + fmm) / (4.0 * ei[i] * ej[j])
            H[i, j] = hij
            H[j, i] = hij
    return H


def estimate_hank_whittle(
    ma_builder: Callable[[Mapping[str, float]], tuple[np.ndarray, np.ndarray]],
    data: np.ndarray | pd.DataFrame,
    init_params: Mapping[str, float],
    *,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    measurement_cov: np.ndarray | None = None,
    demean: bool = True,
) -> HANKEstimationResult:
    """Maximum-Whittle-likelihood estimator for sequence-space HANK / DSGE models.

    Parameters
    ----------
    ma_builder : Callable[[Mapping[str, float]], tuple[np.ndarray, np.ndarray]]
        Callable mapping a candidate parameter dict `theta` to `(M, shock_cov)`
        where `M` has shape `(T_ma, n_obs, n_shocks)` and `shock_cov` has shape
        `(n_shocks, n_shocks)`.
    data : array_like of shape (T_obs, n_obs)
        Observed macroeconomic time series.
    init_params : Mapping[str, float]
        Initial parameter guess.
    bounds : Mapping[str, tuple[float, float]], optional
        Parameter box bounds for `L-BFGS-B`.
    measurement_cov : np.ndarray, optional
        Measurement error covariance matrix.
    demean : bool, default True
        Whether to demean `data`.
    """
    names = list(init_params.keys())
    x0 = np.array([float(init_params[k]) for k in names], dtype=np.float64)
    scipy_bounds = (
        [bounds.get(k, (None, None)) for k in names] if bounds is not None else None
    )

    def _neg_ll(x: np.ndarray) -> float:
        p_dict = {k: float(v) for k, v in zip(names, x)}
        try:
            M, shock_cov = ma_builder(p_dict)
            ll = hank_loglik_whittle(
                M,
                data,
                shock_cov,
                measurement_cov=measurement_cov,
                demean=demean,
            )
            return -ll if np.isfinite(ll) else 1e12
        except Exception:
            return 1e12

    opt = scipy.optimize.minimize(_neg_ll, x0, method="L-BFGS-B", bounds=scipy_bounds)
    x_hat = np.asarray(opt.x, dtype=np.float64)
    ll_hat = -float(opt.fun)

    H = _numerical_hessian(_neg_ll, x_hat)
    try:
        vcov = np.linalg.pinv(0.5 * (H + H.T))
        se_vec = np.sqrt(np.maximum(np.diag(vcov), 0.0))
    except Exception:
        vcov = np.full((len(names), len(names)), np.nan)
        se_vec = np.full(len(names), np.nan)

    return HANKEstimationResult(
        params={k: float(v) for k, v in zip(names, x_hat)},
        se={k: float(s) for k, s in zip(names, se_vec)},
        vcov=vcov,
        objective_value=float(opt.fun),
        log_likelihood=ll_hat,
        method="whittle_mle",
        converged=bool(opt.success),
        iterations=int(getattr(opt, "nit", 0)),
    )


def estimate_hank_irf_matching(
    irf_builder: Callable[[Mapping[str, float], int], pd.DataFrame | Mapping[str, np.ndarray]],
    target_irfs: pd.DataFrame | Mapping[str, np.ndarray],
    init_params: Mapping[str, float],
    *,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    weighting_matrix: np.ndarray | None = None,
) -> HANKEstimationResult:
    """Minimum-distance Sequence-Space IRF matching estimator.

    Parameters
    ----------
    irf_builder : Callable[[Mapping[str, float], int], pd.DataFrame | Mapping[str, np.ndarray]]
        Function `(params_dict, H) -> irf_dict_or_df` returning model impulse
        responses over horizon `H`.
    target_irfs : pd.DataFrame or Mapping[str, np.ndarray]
        Empirical target impulse responses (e.g. from LP or SVAR).
    init_params : Mapping[str, float]
        Initial parameter values.
    bounds : Mapping[str, tuple[float, float]], optional
        Box bounds for `L-BFGS-B`.
    weighting_matrix : np.ndarray, optional
        Positive-definite weighting matrix $W$ of shape `(H * n_vars, H * n_vars)`.
    """
    target_df = pd.DataFrame(target_irfs).copy()
    var_cols = list(target_df.columns)
    H = len(target_df)
    target_vec = target_df.to_numpy(dtype=np.float64).ravel()
    n_m = len(target_vec)
    W = np.eye(n_m, dtype=np.float64) if weighting_matrix is None else np.asarray(weighting_matrix, dtype=np.float64)

    names = list(init_params.keys())
    x0 = np.array([float(init_params[k]) for k in names], dtype=np.float64)
    scipy_bounds = [bounds.get(k, (None, None)) for k in names] if bounds is not None else None

    def _eval_vec(x: np.ndarray) -> np.ndarray:
        p_dict = {k: float(v) for k, v in zip(names, x)}
        raw = irf_builder(p_dict, H)
        df = pd.DataFrame(raw)[var_cols].iloc[:H]
        return df.to_numpy(dtype=np.float64).ravel()

    def _obj(x: np.ndarray) -> float:
        try:
            diff = _eval_vec(x) - target_vec
            return float(diff @ W @ diff)
        except Exception:
            return 1e12

    opt = scipy.optimize.minimize(_obj, x0, method="L-BFGS-B", bounds=scipy_bounds)
    x_hat = np.asarray(opt.x, dtype=np.float64)

    # Compute sandwich/Gauss-Newton covariance (G' W G)^{-1}
    p = len(names)
    G = np.zeros((n_m, p), dtype=np.float64)
    m0 = _eval_vec(x_hat)
    step = 1e-5
    for j in range(p):
        ej = np.zeros(p, dtype=np.float64)
        ej[j] = step * max(1.0, abs(float(x_hat[j])))
        G[:, j] = (_eval_vec(x_hat + ej) - _eval_vec(x_hat - ej)) / (2.0 * ej[j])

    try:
        vcov = np.linalg.pinv(G.T @ W @ G)
        se_vec = np.sqrt(np.maximum(np.diag(vcov), 0.0))
    except Exception:
        vcov = np.full((p, p), np.nan)
        se_vec = np.full(p, np.nan)

    fitted_df = pd.DataFrame(m0.reshape(H, len(var_cols)), index=target_df.index, columns=var_cols)
    return HANKEstimationResult(
        params={k: float(v) for k, v in zip(names, x_hat)},
        se={k: float(s) for k, s in zip(names, se_vec)},
        vcov=vcov,
        objective_value=float(opt.fun),
        log_likelihood=None,
        method="ssj_irf_matching",
        converged=bool(opt.success),
        iterations=int(getattr(opt, "nit", 0)),
        fitted_irfs=fitted_df,
        target_irfs=target_df,
    )
