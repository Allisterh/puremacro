r"""Dynamic Spatial Migration Hat-Algebra (Caliendo, Parro & Rossi-Hansberg 2019).

Solves transitional and stationary spatial general equilibrium paths across $N$
regions over horizon $T$ in exact relative changes ("hat algebra") without
requiring calibration of unobserved level fundamentals (level productivities,
level amenities, level bilateral trade costs, or level migration costs).

In each period $t = 0, \dots, T-1$:

1. **Within-Period Gravity Trade Equilibrium:**
   Given regional labor $L_{i,t}$, productivity changes $\hat{A}_{i,t}$, and
   bilateral iceberg trade cost changes $\hat{\tau}_{ij,t}$:

   .. math::

       \pi_{ij,t} = \frac{\pi_{ij,0} (\hat{\tau}_{ij,t} \hat{w}_{j,t} / \hat{A}_{j,t})^{-\theta}}
                         {\sum_{k=1}^N \pi_{ik,0} (\hat{\tau}_{ik,t} \hat{w}_{k,t} / \hat{A}_{k,t})^{-\theta}},
       \qquad
       \hat{w}_{j,t} \hat{L}_{j,t} Y_{j,0} = \sum_{i=1}^N \pi_{ij,t} \hat{w}_{i,t} \hat{L}_{i,t} Y_{i,0}

2. **Forward-Looking EV1 Migration Bellman & Law of Motion:**
   Workers face Type-I Extreme Value idiosyncratic location preferences with
   dispersion $\nu > 0$ (migration elasticity $\varepsilon_m = 1/\nu$) and discount
   factor $\beta \in (0, 1)$:

   .. math::

       u_{i,t} = (\hat{B}_{i,t} \hat{\omega}_{i,t})^{1/\nu}
                 \sum_{k=1}^N \mu_{ik,0} (u_{k,t+1})^{\beta} \hat{m}_{ik,t}^{-1/\nu},
       \qquad
       L_{j,t+1} = \sum_{i=1}^N \mu_{ij,t} L_{i,t}

References
----------
- Caliendo, L., Dvorkin, M., & Parro, F. (2019). Trade and Labor Market Dynamics:
  General Equilibrium Analysis of the China Trade Shock. *Econometrica*, 87(3), 741–835.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    import matplotlib.figure

import numpy as np
import pandas as pd

from puremacro.reports import _df_to_latex, _df_to_markdown, _df_to_typst

__all__ = [
    "DynamicSpatialHatModel",
    "DynamicSpatialHatResult",
    "solve_dynamic_spatial_hat",
]


@dataclass(frozen=True)
class DynamicSpatialHatModel:
    """Baseline specification for a Dynamic Spatial Hat-Algebra economy."""

    initial_population: np.ndarray
    initial_trade_shares: np.ndarray
    initial_migration_shares: np.ndarray
    trade_elasticity: float = 4.0
    migration_dispersion: float = 0.5
    beta: float = 0.96
    congestion_alpha: float = 0.10
    initial_wages: np.ndarray | None = None
    region_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        L0 = np.asarray(self.initial_population, dtype=np.float64).reshape(-1)
        n = L0.size
        if np.any(L0 <= 0.0):
            raise ValueError("initial_population must be strictly positive.")
        pi0 = np.asarray(self.initial_trade_shares, dtype=np.float64)
        mu0 = np.asarray(self.initial_migration_shares, dtype=np.float64)
        if pi0.shape != (n, n) or mu0.shape != (n, n):
            raise ValueError(f"initial_trade_shares and initial_migration_shares must have shape ({n}, {n}).")
        if not np.allclose(np.sum(pi0, axis=1), 1.0, atol=1e-6):
            raise ValueError("Rows of initial_trade_shares must sum to 1.0.")
        if not np.allclose(np.sum(mu0, axis=1), 1.0, atol=1e-6):
            raise ValueError("Rows of initial_migration_shares must sum to 1.0.")
        if self.trade_elasticity <= 0.0 or self.migration_dispersion <= 0.0:
            raise ValueError("trade_elasticity and migration_dispersion must be strictly positive.")
        if not (0.0 < self.beta < 1.0):
            raise ValueError("beta must lie in (0, 1).")


@dataclass(frozen=True)
class DynamicSpatialHatResult:
    """Result container for Dynamic Spatial Migration Hat-Algebra transition paths."""

    region_names: tuple[str, ...]
    population_path: np.ndarray
    population_hat_path: np.ndarray
    wage_hat_path: np.ndarray
    price_hat_path: np.ndarray
    real_wage_hat_path: np.ndarray
    value_hat_path: np.ndarray
    trade_shares_path: np.ndarray
    migration_shares_path: np.ndarray
    welfare_hat: np.ndarray
    converged: bool
    iterations: int
    max_bellman_residual: float
    population_conservation_error: float
    migration_row_stochastic_error: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self, period: int = -1) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "population": self.population_path[period],
                "population_hat": self.population_hat_path[period],
                "wage_hat": self.wage_hat_path[period],
                "price_hat": self.price_hat_path[period],
                "real_wage_hat": self.real_wage_hat_path[period],
                "value_hat": self.value_hat_path[period],
                "welfare_hat_t0": self.welfare_hat,
            },
            index=pd.Index(self.region_names, name="region"),
        )

    def to_frame(self, period: int = -1) -> pd.DataFrame:
        return self.to_dataframe(period=period)

    def summary(self) -> pd.DataFrame:
        return self.to_dataframe(period=-1)

    def to_markdown(self, **kwargs: Any) -> str:
        return _df_to_markdown(self.summary(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        return _df_to_latex(self.summary(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        return _df_to_typst(self.summary(), **kwargs)

    def plot(self, figsize: tuple[float, float] = (10.5, 4.2)) -> matplotlib.figure.Figure:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
        t_grid = np.arange(self.population_hat_path.shape[0])
        for i, r_name in enumerate(self.region_names):
            ax1.plot(t_grid, self.population_hat_path[:, i], lw=2.0, label=r_name)
            ax2.plot(t_grid, self.real_wage_hat_path[:, i], lw=2.0, label=r_name)
        ax1.axhline(1.0, color="0.5", ls=":", lw=0.8)
        ax2.axhline(1.0, color="0.5", ls=":", lw=0.8)
        ax1.set_title("Regional Population Path (L_hat)")
        ax1.set_xlabel("Period t")
        ax2.set_title("Regional Real Wage Path (omega_hat)")
        ax2.set_xlabel("Period t")
        ax1.legend(frameon=False)
        return fig


def _solve_within_period_trade(
    L_hat: np.ndarray,
    A_hat: np.ndarray,
    tau_hat: np.ndarray,
    pi0: np.ndarray,
    Y0: np.ndarray,
    theta: float,
    alpha: float,
    *,
    w_init: np.ndarray | None = None,
    tol: float = 1e-13,
    max_iter: int = 400,
    damping: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve temporary trade equilibrium for (w_hat, P_hat, omega_hat, pi_t)."""
    n = L_hat.size
    w_hat = np.ones(n, dtype=np.float64) if w_init is None else w_init.copy()
    total_Y0 = float(np.sum(Y0))

    for _ in range(max_iter):
        # Cost change c_hat_j = w_hat_j / A_hat_j
        cost_ratio = (tau_hat * (w_hat[None, :] / A_hat[None, :])) ** (-theta)
        numer = pi0 * cost_ratio
        denom = np.sum(numer, axis=1)
        pi_t = numer / denom[:, None]

        # Goods market clearing via Alvarez-Lucas (2007) contraction exponent 1/(1 + theta):
        # w_hat_j * L_hat_j * Y0_j = sum_i pi_t[i, j] * w_hat_i * L_hat_i * Y0_i
        inc_i = w_hat * L_hat * Y0
        demand_j = pi_t.T @ inc_i
        w_new = w_hat * ((demand_j / inc_i) ** (1.0 / (1.0 + theta)))
        # Normalize so baseline-weighted nominal GDP is constant
        w_new = w_new * (total_Y0 / float(np.sum(w_new * L_hat * Y0)))
        if float(np.max(np.abs(w_new - w_hat))) < tol:
            w_hat = w_new
            break
        w_hat = w_new

    cost_ratio = (tau_hat * (w_hat[None, :] / A_hat[None, :])) ** (-theta)
    numer = pi0 * cost_ratio
    denom = np.sum(numer, axis=1)
    pi_t = numer / denom[:, None]
    P_hat = denom ** (-1.0 / theta)
    omega_hat = (w_hat / P_hat) * (L_hat ** (-alpha))
    return w_hat, P_hat, omega_hat, pi_t


def solve_dynamic_spatial_hat(
    model: DynamicSpatialHatModel,
    *,
    horizon: int = 30,
    tfp_hat_path: np.ndarray | None = None,
    amenity_hat_path: np.ndarray | None = None,
    trade_cost_hat_path: np.ndarray | None = None,
    migration_cost_hat_path: np.ndarray | None = None,
    tol: float = 1e-10,
    max_iter: int = 200,
    damping: float = 0.3,
) -> DynamicSpatialHatResult:
    """Solve the dynamic spatial migration hat-algebra transition over ``horizon`` periods."""
    if horizon < 2:
        raise ValueError(f"horizon must be >= 2, got {horizon}")

    L0 = np.asarray(model.initial_population, dtype=np.float64).reshape(-1)
    n = L0.size
    pi0 = np.asarray(model.initial_trade_shares, dtype=np.float64)
    mu0 = np.asarray(model.initial_migration_shares, dtype=np.float64)
    w0 = np.ones(n, dtype=np.float64) if model.initial_wages is None else np.asarray(model.initial_wages, dtype=np.float64).reshape(n)
    Y0 = w0 * L0
    names = (
        tuple(model.region_names)
        if model.region_names is not None
        else tuple(f"Region_{i + 1}" for i in range(n))
    )

    def _expand_path_2d(arr: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
        out = np.ones(shape, dtype=np.float64)
        if arr is None:
            return out
        a = np.asarray(arr, dtype=np.float64)
        if a.shape == shape[1:]:
            out[:] = a[None, ...]
            return out
        t_copy = min(horizon, a.shape[0])
        out[:t_copy] = a[:t_copy]
        if t_copy < horizon:
            out[t_copy:] = a[t_copy - 1]
        return out

    A_path = _expand_path_2d(tfp_hat_path, (horizon, n))
    B_path = _expand_path_2d(amenity_hat_path, (horizon, n))
    tau_path = _expand_path_2d(trade_cost_hat_path, (horizon, n, n))
    m_path = _expand_path_2d(migration_cost_hat_path, (horizon, n, n))

    theta = float(model.trade_elasticity)
    nu = float(model.migration_dispersion)
    beta = float(model.beta)
    alpha = float(model.congestion_alpha)

    L_path = np.repeat(L0[None, :], horizon, axis=0)
    u_path = np.ones((horizon, n), dtype=np.float64)
    w_hat_path = np.ones((horizon, n), dtype=np.float64)
    P_hat_path = np.ones((horizon, n), dtype=np.float64)
    omega_hat_path = np.ones((horizon, n), dtype=np.float64)
    pi_path = np.repeat(pi0[None, :, :], horizon, axis=0)
    mu_path = np.repeat(mu0[None, :, :], horizon, axis=0)

    converged = False
    max_bellman_res = 1.0
    it = 0

    for it in range(1, max_iter + 1):
        # 1. Solve within-period gravity trade equilibrium along current population path
        L_hat_curr = L_path / L0[None, :]
        for t in range(horizon):
            w_t, P_t, om_t, pi_t = _solve_within_period_trade(
                L_hat_curr[t],
                A_path[t],
                tau_path[t],
                pi0,
                Y0,
                theta,
                alpha,
                w_init=w_hat_path[t],
            )
            w_hat_path[t] = w_t
            P_hat_path[t] = P_t
            omega_hat_path[t] = om_t
            pi_path[t] = pi_t

        # 2. Solve terminal stationary continuation value u_{T-1} at terminal fundamentals
        # In log-space v = ln(u): F(v)_i = v_i - ln(flow_i) - ln(sum_k mu0_{ik} m_{ik}^{-1/nu} exp(beta * v_k)) = 0
        # Jacobian dF/dv = I - beta * mu_term(v), which converges quadratically in 4-5 Newton steps.
        v_term = np.log(np.maximum(u_path[-1], 1e-300))
        log_flow_term = (1.0 / nu) * np.log(np.maximum(B_path[-1] * omega_hat_path[-1], 1e-300))
        m_term_pow = m_path[-1] ** (-1.0 / nu)
        eye_n = np.eye(n, dtype=np.float64)
        for _ in range(30):
            bv = beta * v_term
            m_bv = float(np.max(bv))
            w_term = (mu0 * m_term_pow) * np.exp(bv - m_bv)[None, :]
            row_s = np.sum(w_term, axis=1)
            log_opt = m_bv + np.log(np.maximum(row_s, 1e-300))
            res_v = v_term - log_flow_term - log_opt
            if float(np.max(np.abs(res_v))) < 1e-14:
                break
            mu_term = w_term / row_s[:, None]
            step_v = np.linalg.solve(eye_n - beta * mu_term, res_v)
            v_term = v_term - step_v
        u_term = np.exp(v_term)

        # 3. Backward recursion for u_t and migration shares mu_t
        u_new = np.ones_like(u_path)
        u_new[-1] = u_term
        for t in range(horizon - 1, -1, -1):
            u_next = u_new[t + 1] if t + 1 < horizon else u_term
            flow_t = (B_path[t] * omega_hat_path[t]) ** (1.0 / nu)
            m_pow = m_path[t] ** (-1.0 / nu)
            weights_ik = mu0 * m_pow * ((u_next ** beta)[None, :])
            row_sum = np.sum(weights_ik, axis=1)
            mu_path[t] = weights_ik / row_sum[:, None]
            u_new[t] = flow_t * row_sum

        # 4. Forward law of motion for regional population L_{t+1} = mu_t^T L_t
        L_new = np.zeros_like(L_path)
        L_new[0] = L0
        for t in range(horizon - 1):
            L_new[t + 1] = mu_path[t].T @ L_new[t]

        bellman_diff = float(np.max(np.abs(u_new - u_path) / np.maximum(np.abs(u_new), 1.0)))
        pop_diff = float(np.max(np.abs(L_new - L_path) / L0[None, :]))
        max_bellman_res = max(bellman_diff, pop_diff)

        u_path = (1.0 - damping) * u_path + damping * u_new
        L_path = (1.0 - damping) * L_path + damping * L_new

        if max_bellman_res < tol:
            u_path = u_new
            L_path = L_new
            converged = True
            break

    L_hat_path = L_path / L0[None, :]
    total_L0 = float(np.sum(L0))
    pop_err = float(np.max(np.abs(np.sum(L_path, axis=1) - total_L0)))
    stoch_err = float(np.max(np.abs(np.sum(mu_path, axis=2) - 1.0)))
    welfare_hat = u_path[0] ** nu

    return DynamicSpatialHatResult(
        region_names=names,
        population_path=L_path,
        population_hat_path=L_hat_path,
        wage_hat_path=w_hat_path,
        price_hat_path=P_hat_path,
        real_wage_hat_path=omega_hat_path,
        value_hat_path=u_path,
        trade_shares_path=pi_path,
        migration_shares_path=mu_path,
        welfare_hat=welfare_hat,
        converged=converged,
        iterations=it,
        max_bellman_residual=max_bellman_res,
        population_conservation_error=pop_err,
        migration_row_stochastic_error=stoch_err,
    )
