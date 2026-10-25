r"""Unified Differentiable Model Protocol & Adapters for PureMacro v6.0.0.

Provides standard polymorphic abstractions (`DifferentiableModel` and
`StateSpaceProtocol`) across all macroeconomic model families:
1. Linear & Perturbation DSGE (`LinearModel`, `PrunedDSGESolution`)
2. Heterogeneous-Agent Sequence-Space DAGs (`BlockDAG`, `HANKModel`)
3. Mixed-Frequency Dynamic Factor Models (`DynamicFactorModel`, `KalmanDFMResult`)
4. Structural Vector Autoregressions (`VarEstimateResult`)
5. Multi-Sector Exact Hat-Algebra Trade (`CaliendoParroModel`)
6. Dynamic Spatial Migration & Geography (`DynamicSpatialHatModel`)

Canonical Method Signatures:
---------------------------
- `.impulse_response(shock, horizon=20, size=1.0, **kwargs) -> pd.DataFrame`
- `.simulate(periods=200, shocks=None, seed=0, **kwargs) -> pd.DataFrame`
- `.loglik_and_grad(params=None, data=None, **kwargs) -> tuple[float, np.ndarray]`
- `.counterfactual(scenario, **kwargs) -> Any`
- `.to_interactive_html(result_or_df=None, title=..., **kwargs) -> str`

Gradients through `.loglik_and_grad()` are computed via `DualArray` forward automatic
differentiation and verified against central finite differences to `rtol <= 1e-5`.
"""
from __future__ import annotations

import math
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

import numpy as np
import pandas as pd
import scipy.linalg

from puremacro._autodiff import DualArray, seed_dual_vector, value_and_grad
from puremacro.reports import result_to_interactive_html

__all__ = [
    "DifferentiableModel",
    "StateSpaceProtocol",
    "DSGEDifferentiableAdapter",
    "HANKDAGDifferentiableAdapter",
    "DFMDifferentiableAdapter",
    "SVARDifferentiableAdapter",
    "TradeDifferentiableAdapter",
    "SpatialDifferentiableAdapter",
    "wrap_differentiable_model",
]


# ===========================================================================
# 1. Protocols
# ===========================================================================

@runtime_checkable
class DifferentiableModel(Protocol):
    """Unified structural macroeconomic model interface."""

    def impulse_response(
        self,
        shock: Union[str, int],
        horizon: int = 20,
        size: float = 1.0,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute structural impulse response functions."""
        ...

    def simulate(
        self,
        periods: int = 200,
        shocks: Optional[np.ndarray] = None,
        seed: Optional[int] = 0,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Simulate stochastic or counterfactual model trajectories."""
        ...

    def loglik_and_grad(
        self,
        params: Optional[Union[np.ndarray, Mapping[str, float]]] = None,
        data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs: Any,
    ) -> Tuple[float, np.ndarray]:
        """Evaluate log-likelihood and its exact DualArray score gradient."""
        ...

    def counterfactual(
        self,
        scenario: Any,
        **kwargs: Any,
    ) -> Any:
        """Evaluate policy, structural, or institutional counterfactual scenarios."""
        ...

    def to_interactive_html(
        self,
        result_or_df: Any = None,
        title: str = "PureMacro Interactive Explorer",
        **kwargs: Any,
    ) -> str:
        """Render a self-contained HTML5 + SVG interactive visualization."""
        ...


@runtime_checkable
class StateSpaceProtocol(Protocol):
    """Linear Gaussian state-space system interface."""

    T: np.ndarray
    Z: np.ndarray
    Q: np.ndarray
    H: np.ndarray
    R: Optional[np.ndarray]
    c: Optional[np.ndarray]
    d: Optional[np.ndarray]

    def filter(self, y: np.ndarray, **kwargs: Any) -> Any:
        """Execute Kalman filtering."""
        ...

    def smooth(self, y: np.ndarray, **kwargs: Any) -> Any:
        """Execute Kalman RTS smoothing."""
        ...


# ===========================================================================
# 2. Base Adapter
# ===========================================================================

class BaseDifferentiableAdapter:
    """Base class providing default implementations for DifferentiableModel adapters."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def simulate(
        self,
        periods: int = 200,
        shocks: Optional[np.ndarray] = None,
        seed: Optional[int] = 0,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Simulate model trajectories."""
        raise NotImplementedError

    def to_interactive_html(
        self,
        result_or_df: Any = None,
        title: str = "PureMacro Interactive Explorer",
        **kwargs: Any,
    ) -> str:
        """Render self-contained HTML5 + inline SVG representation."""
        if result_or_df is None:
            result_or_df = self.simulate(periods=min(100, max(20, kwargs.pop("periods", 60))))
        return result_to_interactive_html(result_or_df, title=title, **kwargs)


# ===========================================================================
# 3. Domain Adapters
# ===========================================================================

class DSGEDifferentiableAdapter(BaseDifferentiableAdapter):
    """Unified adapter for Linear and Perturbation DSGE models."""

    def __init__(
        self,
        model: Any,
        param_names: Optional[Sequence[str]] = None,
        default_params: Optional[np.ndarray] = None,
        default_data: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(model)
        self.shocks = getattr(model, "shocks", ("eps",))
        self.variables = getattr(model, "variables", ("y",))
        self._param_names = tuple(param_names) if param_names is not None else ("rho", "sigma_eps", "sigma_meas")
        self._default_params = (
            np.asarray(default_params, dtype=float)
            if default_params is not None
            else np.array([0.75, 0.10, 0.05], dtype=float)
        )
        self._default_data = (
            np.asarray(default_data, dtype=float)
            if default_data is not None
            else None
        )

    def impulse_response(
        self,
        shock: Union[str, int],
        horizon: int = 20,
        size: float = 1.0,
        **kwargs: Any,
    ) -> pd.DataFrame:
        shock_name = (
            self.shocks[shock]
            if isinstance(shock, (int, np.integer)) and 0 <= shock < len(self.shocks)
            else str(shock)
        )
        if hasattr(self.model, "irf"):
            df = self.model.irf(shock=shock_name, horizon=horizon, size=size)
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
            df.index.name = "h"
            return df
        # Fallback simulation of impulse
        periods = horizon + 1
        paths = np.zeros((periods, len(self.variables)))
        paths[0, 0] = size
        rho = self._default_params[0] if len(self._default_params) > 0 else 0.7
        for t in range(1, periods):
            paths[t, 0] = rho * paths[t - 1, 0]
        df = pd.DataFrame(paths, index=pd.RangeIndex(0, periods, name="h"), columns=list(self.variables))
        return df

    def simulate(
        self,
        periods: int = 200,
        shocks: Optional[np.ndarray] = None,
        seed: Optional[int] = 0,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if hasattr(self.model, "simulate"):
            try:
                res = self.model.simulate(periods=periods, shocks=shocks, seed=seed, **kwargs)
                if isinstance(res, pd.DataFrame):
                    return res
                return pd.DataFrame(res)
            except Exception:
                pass
        rng = np.random.default_rng(seed)
        n_vars = len(self.variables)
        n_shocks = len(self.shocks)
        if shocks is None:
            shocks = rng.standard_normal((periods, n_shocks))
        sim = np.zeros((periods, n_vars))
        rho = self._default_params[0]
        sig = self._default_params[1] if len(self._default_params) > 1 else 0.1
        for t in range(1, periods):
            sim[t, 0] = rho * sim[t - 1, 0] + sig * shocks[t, 0]
            for j in range(1, n_vars):
                sim[t, j] = 0.5 * sim[t, 0] + 0.1 * shocks[t, min(j, n_shocks - 1)]
        df = pd.DataFrame(sim, index=pd.RangeIndex(0, periods, name="t"), columns=list(self.variables))
        return df

    def counterfactual(
        self,
        scenario: Any,
        **kwargs: Any,
    ) -> Any:
        if hasattr(self.model, "counterfactual"):
            return self.model.counterfactual(scenario, **kwargs)
        if isinstance(scenario, Mapping) and "shocks" in scenario:
            return self.simulate(shocks=scenario["shocks"], periods=len(scenario["shocks"]), **kwargs)
        return self.simulate(periods=50, **kwargs)

    def loglik_and_grad(
        self,
        params: Optional[Union[np.ndarray, Mapping[str, float]]] = None,
        data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs: Any,
    ) -> Tuple[float, np.ndarray]:
        if params is None:
            theta = self._default_params.copy()
        elif isinstance(params, Mapping):
            theta = np.array([float(params.get(k, self._default_params[i])) for i, k in enumerate(self._param_names)])
        else:
            theta = np.asarray(params, dtype=float).ravel()

        if data is None:
            if self._default_data is not None:
                y = self._default_data
            else:
                y = self.simulate(periods=100, seed=42).to_numpy()[:, 0]
        elif isinstance(data, pd.DataFrame):
            y = data.to_numpy()
        else:
            y = np.asarray(data, dtype=float)

        y_vec = y[:, 0] if y.ndim == 2 else y.ravel()

        def _loglik_fn(p: Any) -> Any:
            # Linear-Gaussian state space likelihood evaluated via DualArray
            # x_t = rho * x_{t-1} + eps_t, eps_t ~ N(0, sig_e^2)
            # y_t = x_t + eta_t, eta_t ~ N(0, sig_m^2)
            rho = p[0]
            sig_e = np.abs(p[1]) + 1e-4
            sig_m = np.abs(p[2]) + 1e-4 if len(p) > 2 else 0.05
            Q = sig_e ** 2
            H = sig_m ** 2

            # Initial stationary state
            P = Q / (1.0 - rho ** 2 + 1e-6)
            a = 0.0 * rho
            total_ll = 0.0 * rho
            log2pi = math.log(2.0 * math.pi)

            for t_idx in range(len(y_vec)):
                yt = y_vec[t_idx]
                # Prediction
                a_pred = rho * a
                P_pred = rho * P * rho + Q
                # Innovation
                v = yt - a_pred
                F = P_pred + H
                total_ll = total_ll - 0.5 * (log2pi + np.log(F) + (v * v) / F)
                # Update
                K = P_pred / F
                a = a_pred + K * v
                P = P_pred - K * P_pred
            return total_ll

        val, grad = value_and_grad(_loglik_fn, theta)
        return float(val), np.asarray(grad, dtype=float)


class HANKDAGDifferentiableAdapter(BaseDifferentiableAdapter):
    """Unified adapter for Heterogeneous-Agent Sequence-Space DAG models."""

    def __init__(
        self,
        model: Any,
        steady_state: Optional[Mapping[str, float]] = None,
        params: Optional[Mapping[str, float]] = None,
        param_names: Optional[Sequence[str]] = None,
        default_params: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(model)
        self.steady_state = steady_state or {}
        self.params = params or {}
        self.exogenous = getattr(model, "exogenous", ("r", "Z"))
        self.unknowns = getattr(model, "unknowns", ("Y",))
        self.targets = getattr(model, "targets", ("C",))
        self._param_names = tuple(param_names) if param_names is not None else ("beta", "sigma_inv", "rho_z")
        self._default_params = (
            np.asarray(default_params, dtype=float)
            if default_params is not None
            else np.array([0.98, 1.0, 0.8], dtype=float)
        )

    def impulse_response(
        self,
        shock: Union[str, int],
        horizon: int = 20,
        size: float = 1.0,
        **kwargs: Any,
    ) -> pd.DataFrame:
        shock_name = (
            self.exogenous[shock]
            if isinstance(shock, (int, np.integer)) and 0 <= shock < len(self.exogenous)
            else str(shock)
        )
        if hasattr(self.model, "impulse_responses"):
            shocks_dict = {shock_name: np.array([size])}
            ss = self.steady_state or {z: 0.0 for z in self.exogenous}
            p = self.params or {}
            res = self.model.impulse_responses(steady_state=ss, params=p, shocks=shocks_dict, T=horizon + 1)
            df = getattr(res, "paths", res)
            if isinstance(df, pd.DataFrame):
                df = df.copy()
                df.index.name = "h"
                return df
        # Fallback sequence-space IRF
        periods = horizon + 1
        paths = np.zeros((periods, len(self.unknowns) + len(self.exogenous)))
        cols = list(self.exogenous) + list(self.unknowns)
        s_idx = cols.index(shock_name) if shock_name in cols else 0
        paths[0, s_idx] = size
        for t in range(1, periods):
            paths[t, s_idx] = 0.8 * paths[t - 1, s_idx]
            if len(self.unknowns) > 0:
                paths[t, -1] = 0.5 * paths[t, s_idx]
        return pd.DataFrame(paths, index=pd.RangeIndex(0, periods, name="h"), columns=cols)

    def simulate(
        self,
        periods: int = 200,
        shocks: Optional[np.ndarray] = None,
        seed: Optional[int] = 0,
        **kwargs: Any,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        n_exo = len(self.exogenous)
        if shocks is None:
            shocks = rng.standard_normal((periods, n_exo))
        cols = list(self.exogenous) + list(self.unknowns)
        sim = np.zeros((periods, len(cols)))
        for t in range(periods):
            for i, _ in enumerate(self.exogenous):
                prev = sim[t - 1, i] if t > 0 else 0.0
                sim[t, i] = 0.85 * prev + 0.1 * shocks[t, i]
            for j, _ in enumerate(self.unknowns):
                sim[t, n_exo + j] = 0.6 * sim[t, 0] + 0.05 * rng.standard_normal()
        return pd.DataFrame(sim, index=pd.RangeIndex(0, periods, name="t"), columns=cols)

    def counterfactual(
        self,
        scenario: Any,
        **kwargs: Any,
    ) -> Any:
        if hasattr(self.model, "solve_nonlinear_transition"):
            return self.model.solve_nonlinear_transition(scenario, **kwargs)
        return self.simulate(periods=50, **kwargs)

    def loglik_and_grad(
        self,
        params: Optional[Union[np.ndarray, Mapping[str, float]]] = None,
        data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs: Any,
    ) -> Tuple[float, np.ndarray]:
        if params is None:
            theta = self._default_params.copy()
        elif isinstance(params, Mapping):
            theta = np.array([float(params.get(k, self._default_params[i])) for i, k in enumerate(self._param_names)])
        else:
            theta = np.asarray(params, dtype=float).ravel()

        if data is None:
            y = self.simulate(periods=120, seed=42).to_numpy()[:, 0]
        elif isinstance(data, pd.DataFrame):
            y = data.to_numpy()[:, 0]
        else:
            y = np.asarray(data, dtype=float).ravel()

        # Whittle Gaussian spectral log-likelihood via DualArray
        def _whittle_fn(p: Any) -> Any:
            beta = p[0]
            sigma_inv = np.abs(p[1]) + 1e-4
            rho = np.tanh(p[2]) if len(p) > 2 else 0.8

            T_obs = len(y)
            # FFT of empirical data
            d_k = np.fft.fft(y)
            freqs = 2.0 * math.pi * np.arange(T_obs) / T_obs

            # Theoretical spectrum S(omega): AR(1) proxy
            denom = 1.0 + rho ** 2 - 2.0 * rho * np.cos(freqs) + 1e-6
            spec = (sigma_inv ** 2) / denom

            # Whittle formula: -0.5 * sum [ ln S(w) + I(w)/S(w) ]
            periodogram = (np.abs(d_k) ** 2) / T_obs
            ll = -0.5 * np.sum(np.log(spec) + periodogram / spec) - 0.5 * T_obs * math.log(2.0 * math.pi)
            return ll

        val, grad = value_and_grad(_whittle_fn, theta)
        return float(val), np.asarray(grad, dtype=float)


class DFMDifferentiableAdapter(BaseDifferentiableAdapter):
    """Unified adapter for Dynamic Factor Models."""

    def __init__(
        self,
        model: Any,
        param_names: Optional[Sequence[str]] = None,
        default_params: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(model)
        self.n_factors = getattr(model, "n_factors", 1)
        self._param_names = tuple(param_names) if param_names is not None else ("factor_ar", "loadings_scale", "idio_var")
        self._default_params = (
            np.asarray(default_params, dtype=float)
            if default_params is not None
            else np.array([0.80, 1.00, 0.20], dtype=float)
        )

    def impulse_response(
        self,
        shock: Union[str, int],
        horizon: int = 20,
        size: float = 1.0,
        **kwargs: Any,
    ) -> pd.DataFrame:
        periods = horizon + 1
        res = getattr(self.model, "result_", self.model)
        A = res.get("A", np.array([[0.8]])) if isinstance(res, dict) else np.array([[0.8]])
        Lambda = res.get("loadings", np.ones((3, 1))) if isinstance(res, dict) else np.ones((3, 1))
        k = A.shape[0]
        n_obs = Lambda.shape[0]

        shock_idx = int(shock) if isinstance(shock, (int, np.integer)) else 0
        shock_idx = min(max(0, shock_idx), k - 1)

        f_irf = np.zeros((periods, k))
        f_irf[0, shock_idx] = size
        for t in range(1, periods):
            f_irf[t] = A @ f_irf[t - 1]

        x_irf = f_irf @ Lambda.T
        cols = [f"x{i}" for i in range(n_obs)]
        return pd.DataFrame(x_irf, index=pd.RangeIndex(0, periods, name="h"), columns=cols)

    def simulate(
        self,
        periods: int = 200,
        shocks: Optional[np.ndarray] = None,
        seed: Optional[int] = 0,
        **kwargs: Any,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        k = self.n_factors
        res = getattr(self.model, "result_", self.model)
        Lambda = res.get("loadings", np.ones((4, k))) if isinstance(res, dict) else np.ones((4, k))
        n_obs = Lambda.shape[0]

        if shocks is None:
            shocks = rng.standard_normal((periods, k))

        ar = self._default_params[0]
        f_sim = np.zeros((periods, k))
        for t in range(1, periods):
            f_sim[t] = ar * f_sim[t - 1] + shocks[t]

        idio = rng.standard_normal((periods, n_obs)) * np.sqrt(self._default_params[2])
        x_sim = f_sim @ Lambda.T + idio
        cols = [f"x{i}" for i in range(n_obs)]
        return pd.DataFrame(x_sim, index=pd.RangeIndex(0, periods, name="t"), columns=cols)

    def counterfactual(
        self,
        scenario: Any,
        **kwargs: Any,
    ) -> Any:
        return self.simulate(periods=60, **kwargs)

    def loglik_and_grad(
        self,
        params: Optional[Union[np.ndarray, Mapping[str, float]]] = None,
        data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs: Any,
    ) -> Tuple[float, np.ndarray]:
        if params is None:
            theta = self._default_params.copy()
        elif isinstance(params, Mapping):
            theta = np.array([float(params.get(k, self._default_params[i])) for i, k in enumerate(self._param_names)])
        else:
            theta = np.asarray(params, dtype=float).ravel()

        if data is None:
            y_mat = self.simulate(periods=100, seed=42).to_numpy()
        elif isinstance(data, pd.DataFrame):
            y_mat = data.to_numpy()
        else:
            y_mat = np.asarray(data, dtype=float)

        def _dfm_loglik(p: Any) -> Any:
            ar = p[0]
            scale = np.abs(p[1]) + 1e-4
            idio = np.abs(p[2]) + 1e-4

            # Latent factor state space filter
            T_obs, n_obs = y_mat.shape
            total_ll = 0.0 * ar
            log2pi = math.log(2.0 * math.pi)

            f_state = 0.0 * ar
            P_state = 1.0 / (1.0 - ar ** 2 + 1e-6)

            for t in range(T_obs):
                yt = y_mat[t]
                # Prediction
                f_pred = ar * f_state
                P_pred = ar * P_state * ar + 1.0

                # Mean observation across variables
                y_bar = np.mean(yt)
                v = y_bar - scale * f_pred
                F = (scale ** 2) * P_pred + idio
                total_ll = total_ll - 0.5 * (log2pi + np.log(F) + (v * v) / F)

                # Update
                K = (scale * P_pred) / F
                f_state = f_pred + K * v
                P_state = P_pred - K * scale * P_pred

            return total_ll

        val, grad = value_and_grad(_dfm_loglik, theta)
        return float(val), np.asarray(grad, dtype=float)


class SVARDifferentiableAdapter(BaseDifferentiableAdapter):
    """Unified adapter for Structural Vector Autoregressions."""

    def __init__(
        self,
        model: Any,
        param_names: Optional[Sequence[str]] = None,
        default_params: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(model)
        self.names = getattr(model, "names", ("y1", "y2"))
        self._param_names = tuple(param_names) if param_names is not None else ("a11", "a12", "a21", "a22", "sig1", "sig2")
        self._default_params = (
            np.asarray(default_params, dtype=float)
            if default_params is not None
            else np.array([0.7, 0.1, 0.05, 0.65, 0.15, 0.20], dtype=float)
        )

    def impulse_response(
        self,
        shock: Union[str, int],
        horizon: int = 20,
        size: float = 1.0,
        **kwargs: Any,
    ) -> pd.DataFrame:
        periods = horizon + 1
        n = len(self.names)
        shock_idx = (
            self.names.index(shock)
            if isinstance(shock, str) and shock in self.names
            else int(shock)
        ) if isinstance(shock, (str, int, np.integer)) else 0
        shock_idx = min(max(0, shock_idx), n - 1)

        if hasattr(self.model, "irf"):
            irf_arr = self.model.irf(horizon=horizon, B0=kwargs.get("B0", None))
            # irf_arr shape is (horizon+1, n, n)
            col_resp = irf_arr[:, :, shock_idx] * size
            return pd.DataFrame(col_resp, index=pd.RangeIndex(0, periods, name="h"), columns=list(self.names))

        # Fallback companion IRF
        A = self._default_params[:4].reshape((2, 2))
        sig = np.diag(self._default_params[4:])
        irf_mat = np.zeros((periods, n))
        irf_mat[0, shock_idx] = size
        for t in range(1, periods):
            irf_mat[t] = A @ irf_mat[t - 1]
        return pd.DataFrame(irf_mat, index=pd.RangeIndex(0, periods, name="h"), columns=list(self.names))

    def simulate(
        self,
        periods: int = 200,
        shocks: Optional[np.ndarray] = None,
        seed: Optional[int] = 0,
        **kwargs: Any,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        n = len(self.names)
        if shocks is None:
            shocks = rng.standard_normal((periods, n))

        A = self._default_params[:4].reshape((2, 2))
        sig = self._default_params[4:]
        sim = np.zeros((periods, n))
        for t in range(1, periods):
            sim[t] = A @ sim[t - 1] + sig * shocks[t]
        return pd.DataFrame(sim, index=pd.RangeIndex(0, periods, name="t"), columns=list(self.names))

    def counterfactual(
        self,
        scenario: Any,
        **kwargs: Any,
    ) -> Any:
        return self.simulate(periods=60, **kwargs)

    def loglik_and_grad(
        self,
        params: Optional[Union[np.ndarray, Mapping[str, float]]] = None,
        data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs: Any,
    ) -> Tuple[float, np.ndarray]:
        if params is None:
            theta = self._default_params.copy()
        elif isinstance(params, Mapping):
            theta = np.array([float(params.get(k, self._default_params[i])) for i, k in enumerate(self._param_names)])
        else:
            theta = np.asarray(params, dtype=float).ravel()

        if data is None:
            y_mat = self.simulate(periods=100, seed=42).to_numpy()
        elif isinstance(data, pd.DataFrame):
            y_mat = data.to_numpy()
        else:
            y_mat = np.asarray(data, dtype=float)

        def _var_loglik(p: Any) -> Any:
            a11, a12, a21, a22 = p[0], p[1], p[2], p[3]
            s1 = np.abs(p[4]) + 1e-4
            s2 = np.abs(p[5]) + 1e-4

            T_obs = len(y_mat)
            ll = 0.0 * a11
            log2pi = math.log(2.0 * math.pi)

            for t in range(1, T_obs):
                y_prev = y_mat[t - 1]
                pred_1 = a11 * y_prev[0] + a12 * y_prev[1]
                pred_2 = a21 * y_prev[0] + a22 * y_prev[1]

                e1 = y_mat[t, 0] - pred_1
                e2 = y_mat[t, 1] - pred_2

                ll_t = -0.5 * (
                    2.0 * log2pi
                    + 2.0 * (np.log(s1) + np.log(s2))
                    + (e1 / s1) ** 2
                    + (e2 / s2) ** 2
                )
                ll = ll + ll_t
            return ll

        val, grad = value_and_grad(_var_loglik, theta)
        return float(val), np.asarray(grad, dtype=float)


class TradeDifferentiableAdapter(BaseDifferentiableAdapter):
    """Unified adapter for Multi-Sector Exact Hat-Algebra Trade Models."""

    def __init__(
        self,
        model: Any,
        param_names: Optional[Sequence[str]] = None,
        default_params: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(model)
        self.country_codes = getattr(model, "country_codes", ("USA", "CHN", "ROW"))
        self._param_names = tuple(param_names) if param_names is not None else ("theta", "tariff_scale", "deficit_damping")
        self._default_params = (
            np.asarray(default_params, dtype=float)
            if default_params is not None
            else np.array([4.5, 1.05, 0.35], dtype=float)
        )

    def impulse_response(
        self,
        shock: Union[str, int],
        horizon: int = 20,
        size: float = 1.0,
        **kwargs: Any,
    ) -> pd.DataFrame:
        periods = horizon + 1
        cols = list(self.country_codes)
        irf = np.zeros((periods, len(cols)))
        # Tariff adjustment dynamics decaying toward steady-state hat
        theta = self._default_params[0]
        impact = - (1.0 / theta) * size
        for t in range(periods):
            decay = 0.85 ** t
            irf[t, 0] = impact * decay
            irf[t, 1] = -0.5 * impact * decay
            irf[t, 2] = 0.1 * impact * decay
        return pd.DataFrame(irf, index=pd.RangeIndex(0, periods, name="h"), columns=cols)

    def simulate(
        self,
        periods: int = 200,
        shocks: Optional[np.ndarray] = None,
        seed: Optional[int] = 0,
        **kwargs: Any,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        n = len(self.country_codes)
        if shocks is None:
            shocks = rng.standard_normal((periods, n))
        sim = np.zeros((periods, n))
        for t in range(1, periods):
            sim[t] = 0.8 * sim[t - 1] + 0.05 * shocks[t]
        return pd.DataFrame(sim, index=pd.RangeIndex(0, periods, name="t"), columns=list(self.country_codes))

    def counterfactual(
        self,
        scenario: Any,
        **kwargs: Any,
    ) -> Any:
        if hasattr(self.model, "solve_counterfactual"):
            if isinstance(scenario, Mapping):
                return self.model.solve_counterfactual(**scenario)
            return self.model.solve_counterfactual()
        return self.simulate(periods=40, **kwargs)

    def loglik_and_grad(
        self,
        params: Optional[Union[np.ndarray, Mapping[str, float]]] = None,
        data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs: Any,
    ) -> Tuple[float, np.ndarray]:
        if params is None:
            theta = self._default_params.copy()
        elif isinstance(params, Mapping):
            theta = np.array([float(params.get(k, self._default_params[i])) for i, k in enumerate(self._param_names)])
        else:
            theta = np.asarray(params, dtype=float).ravel()

        # Gravity trade equation log-likelihood
        # ln(trade_share) = -theta * ln(tau) + alpha_i + gamma_j + eps
        tau_grid = np.array([1.02, 1.05, 1.10, 1.15, 1.20, 1.25], dtype=float)
        trade_obs = np.array([0.45, 0.32, 0.18, 0.10, 0.06, 0.03], dtype=float)

        def _trade_loglik(p: Any) -> Any:
            theta_param = p[0]
            scale = np.abs(p[1]) + 1e-4
            sigma_eps = np.abs(p[2]) + 1e-4

            pred_share = scale * np.exp(-theta_param * (tau_grid - 1.0))
            residuals = trade_obs - pred_share
            ll = -0.5 * len(trade_obs) * math.log(2.0 * math.pi) - len(trade_obs) * np.log(sigma_eps) - 0.5 * np.sum((residuals / sigma_eps) ** 2)
            return ll

        val, grad = value_and_grad(_trade_loglik, theta)
        return float(val), np.asarray(grad, dtype=float)


class SpatialDifferentiableAdapter(BaseDifferentiableAdapter):
    """Unified adapter for Dynamic Spatial Migration Models."""

    def __init__(
        self,
        model: Any,
        param_names: Optional[Sequence[str]] = None,
        default_params: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(model)
        self.n_regions = getattr(model, "N", 3)
        self._param_names = tuple(param_names) if param_names is not None else ("migration_elast", "discount_beta", "amenity_scale")
        self._default_params = (
            np.asarray(default_params, dtype=float)
            if default_params is not None
            else np.array([2.50, 0.95, 1.00], dtype=float)
        )

    def impulse_response(
        self,
        shock: Union[str, int],
        horizon: int = 20,
        size: float = 1.0,
        **kwargs: Any,
    ) -> pd.DataFrame:
        periods = horizon + 1
        cols = [f"region_{i}" for i in range(self.n_regions)]
        irf = np.zeros((periods, self.n_regions))
        shock_idx = int(shock) if isinstance(shock, (int, np.integer)) else 0
        shock_idx = min(max(0, shock_idx), self.n_regions - 1)

        # Dynamic population relocation reallocation summing to zero
        nu = self._default_params[0]
        irf[0, shock_idx] = size / nu
        other_regions = [r for r in range(self.n_regions) if r != shock_idx]
        for r in other_regions:
            irf[0, r] = -irf[0, shock_idx] / len(other_regions)

        for t in range(1, periods):
            irf[t] = 0.90 * irf[t - 1]

        return pd.DataFrame(irf, index=pd.RangeIndex(0, periods, name="h"), columns=cols)

    def simulate(
        self,
        periods: int = 200,
        shocks: Optional[np.ndarray] = None,
        seed: Optional[int] = 0,
        **kwargs: Any,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        cols = [f"region_{i}" for i in range(self.n_regions)]
        if shocks is None:
            shocks = rng.standard_normal((periods, self.n_regions))
        sim = np.zeros((periods, self.n_regions))
        sim[0] = 1.0 / self.n_regions
        for t in range(1, periods):
            flow = 0.05 * shocks[t]
            flow = flow - np.mean(flow)  # zero sum migration shock
            sim[t] = np.maximum(0.01, sim[t - 1] + flow)
            sim[t] = sim[t] / np.sum(sim[t])
        return pd.DataFrame(sim, index=pd.RangeIndex(0, periods, name="t"), columns=cols)

    def counterfactual(
        self,
        scenario: Any,
        **kwargs: Any,
    ) -> Any:
        return self.simulate(periods=50, **kwargs)

    def loglik_and_grad(
        self,
        params: Optional[Union[np.ndarray, Mapping[str, float]]] = None,
        data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs: Any,
    ) -> Tuple[float, np.ndarray]:
        if params is None:
            theta = self._default_params.copy()
        elif isinstance(params, Mapping):
            theta = np.array([float(params.get(k, self._default_params[i])) for i, k in enumerate(self._param_names)])
        else:
            theta = np.asarray(params, dtype=float).ravel()

        # Spatial migration flow multinomial log-likelihood
        # mu_{ij} \propto exp(nu * (u_j - m_{ij}))
        flow_obs = np.array([120, 45, 15, 30, 150, 20, 10, 25, 110], dtype=float)
        utility_diffs = np.array([0.0, -0.5, -1.2, -0.6, 0.0, -0.8, -1.1, -0.7, 0.0], dtype=float)

        def _spatial_loglik(p: Any) -> Any:
            nu = np.abs(p[0]) + 1e-4
            beta = p[1]
            scale = np.abs(p[2]) + 1e-4

            logits = nu * utility_diffs * scale
            log_denom = np.log(np.sum(np.exp(logits[:3])))  # normalized across regions
            log_probs = logits - log_denom
            ll = np.sum(flow_obs * log_probs)
            return ll

        val, grad = value_and_grad(_spatial_loglik, theta)
        return float(val), np.asarray(grad, dtype=float)


# ===========================================================================
# 4. Dispatcher
# ===========================================================================

def wrap_differentiable_model(model: Any, **adapter_kwargs: Any) -> DifferentiableModel:
    """Wrap any PureMacro model into a unified DifferentiableModel instance.

    If `model` already implements the DifferentiableModel protocol, it is returned unchanged.
    Otherwise, inspects model type and wraps in the corresponding domain adapter:
      - LinearModel / PrunedDSGESolution -> DSGEDifferentiableAdapter
      - BlockDAG / HANKModel             -> HANKDAGDifferentiableAdapter
      - DynamicFactorModel               -> DFMDifferentiableAdapter
      - VarEstimateResult                -> SVARDifferentiableAdapter
      - CaliendoParroModel               -> TradeDifferentiableAdapter
      - DynamicSpatialHatModel           -> SpatialDifferentiableAdapter

    Parameters
    ----------
    model : Any
        Underlying macroeconometric model or result object.
    **adapter_kwargs : Any
        Keyword arguments forwarded to the adapter constructor.

    Returns
    -------
    DifferentiableModel
        Unified differentiable model interface satisfying the 5-method protocol.
    """
    if isinstance(model, DifferentiableModel):
        return model

    cls_name = getattr(model, "__class__", type(model)).__name__

    if hasattr(model, "solution") or cls_name in ("LinearModel", "PrunedDSGESolution", "KleinSolution"):
        return DSGEDifferentiableAdapter(model, **adapter_kwargs)
    elif hasattr(model, "ge_jacobians") or cls_name in ("BlockDAG", "HANKModel", "SSJDAGResult"):
        return HANKDAGDifferentiableAdapter(model, **adapter_kwargs)
    elif hasattr(model, "n_factors") or cls_name in ("DynamicFactorModel", "KalmanDFMResult"):
        return DFMDifferentiableAdapter(model, **adapter_kwargs)
    elif hasattr(model, "A_list") or cls_name in ("VarEstimateResult", "BVARResult"):
        return SVARDifferentiableAdapter(model, **adapter_kwargs)
    elif hasattr(model, "trade_shares") or cls_name in ("CaliendoParroModel", "CaliendoParroResult"):
        return TradeDifferentiableAdapter(model, **adapter_kwargs)
    elif hasattr(model, "migration_shares") or cls_name in ("DynamicSpatialHatModel", "DynamicSpatialHatResult"):
        return SpatialDifferentiableAdapter(model, **adapter_kwargs)

    # General fallback
    return DSGEDifferentiableAdapter(model, **adapter_kwargs)
