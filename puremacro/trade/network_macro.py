r"""Baqaee–Farhi (2019, 2020) Second-Order Nonlinear Network Macro Aggregation.

Implements exact first-order (Hulten 1978) and second-order (Baqaee & Farhi 2019,
*Econometrica*; Baqaee & Farhi 2020, *QJE*) macroeconomic aggregation of
sector-level productivity shocks and markup wedges across an input-output
production network:

.. math::

    \Delta \ln Y \approx \underbrace{\sum_{i=1}^N \lambda_i \Delta \ln A_i}_{\text{Hulten 1st-Order}}
    + \underbrace{\frac{1}{2} (\theta_0 - 1) \operatorname{Var}_b(\Psi \Delta \ln A)
      + \frac{1}{2} \sum_{i=1}^N \lambda_i (\theta_i - 1) \operatorname{Var}_{\tilde{\Omega}_i}(\Psi \Delta \ln A)}_{\text{Baqaee–Farhi 2nd-Order Network Amplification}}
    + \Delta \mathcal{AE}(\mu)

Along with the analytical second-order Taylor decomposition and exact $N \times N$
macroeconomic Hessian $\nabla^2_{\ln A} \ln Y$, this module also solves the exact
nonlinear CES unit-cost fixed point $p(\Delta \ln A)$ so users can compare
first-order Hulten, second-order Baqaee–Farhi, and exact nonlinear CES welfare
side-by-side.
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
    "NetworkMacroDecompositionResult",
    "baqaee_farhi_decomposition",
    "solve_exact_ces_network_prices",
]


@dataclass(frozen=True)
class NetworkMacroDecompositionResult:
    """Result container for Baqaee–Farhi (2019, 2020) 2nd-order network aggregation."""

    sector_names: tuple[str, ...]
    domar_weights: np.ndarray
    cost_domar_weights: np.ndarray
    leontief_inverse: np.ndarray
    first_order_dlog_y: float
    second_order_dlog_y: float
    allocative_efficiency_dlog_y: float
    total_second_order_dlog_y: float
    exact_ces_dlog_y: float
    hessian_matrix: np.ndarray
    sector_first_order_contributions: np.ndarray
    node_second_order_contributions: np.ndarray
    final_demand_second_order: float
    equilibrium_log_prices: np.ndarray
    tfp_shocks: np.ndarray
    elasticities: np.ndarray
    final_elasticity: float
    markups: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "domar_weight": self.domar_weights,
                "cost_domar_weight": self.cost_domar_weights,
                "tfp_shock": self.tfp_shocks,
                "elasticity": self.elasticities,
                "markup": self.markups,
                "first_order_contrib": self.sector_first_order_contributions,
                "node_second_order_contrib": self.node_second_order_contributions,
                "exact_log_price": self.equilibrium_log_prices,
            },
            index=pd.Index(self.sector_names, name="sector"),
        )

    def to_frame(self) -> pd.DataFrame:
        return self.to_dataframe()

    def summary(self) -> pd.DataFrame:
        rows = [
            {"Component": "Hulten 1st-Order Aggregate TFP (d1 ln Y)", "Value": self.first_order_dlog_y},
            {"Component": "Baqaee-Farhi 2nd-Order Network Curvature (d2 ln Y)", "Value": self.second_order_dlog_y},
            {"Component": "  - Producer Nodes Curvature Sum", "Value": float(np.sum(self.node_second_order_contributions))},
            {"Component": "  - Final Demand Node Curvature", "Value": self.final_demand_second_order},
            {"Component": "Allocative Efficiency / Markup Wedge Shift", "Value": self.allocative_efficiency_dlog_y},
            {"Component": "Total 2nd-Order Approximation (d1 + d2 + dAE)", "Value": self.total_second_order_dlog_y},
            {"Component": "Exact Nonlinear CES Fixed-Point (d ln Y_exact)", "Value": self.exact_ces_dlog_y},
            {"Component": "3rd+ Order Taylor Remainder |exact - 2nd|", "Value": abs(self.exact_ces_dlog_y - (self.first_order_dlog_y + self.second_order_dlog_y))},
            {"Component": "Aggregate Domar Sum (sum lambda_i)", "Value": float(np.sum(self.domar_weights))},
        ]
        return pd.DataFrame(rows).set_index("Component")

    def to_markdown(self, **kwargs: Any) -> str:
        return _df_to_markdown(self.summary(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        return _df_to_latex(self.summary(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        return _df_to_typst(self.summary(), **kwargs)

    def plot(self, figsize: tuple[float, float] = (10.0, 4.2)) -> matplotlib.figure.Figure:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
        labels = ["1st-Order\n(Hulten)", "2nd-Order\nCurvature", "Total 2nd\nApprox.", "Exact CES\nFixed-Point"]
        vals = [
            self.first_order_dlog_y,
            self.second_order_dlog_y,
            self.total_second_order_dlog_y,
            self.exact_ces_dlog_y,
        ]
        ax1.bar(labels, vals, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#111111"], alpha=0.85)
        ax1.axhline(0.0, color="0.4", lw=0.8)
        ax1.set_title("Aggregate Output Decomposition (Δ ln Y)")
        ax1.set_ylabel("Log change")

        x = np.arange(len(self.sector_names))
        w = 0.38
        ax2.bar(x - w / 2, self.sector_first_order_contributions, width=w, label="1st-Order (λ_i Δln A_i)")
        ax2.bar(x + w / 2, self.node_second_order_contributions, width=w, label="2nd-Order Node Curvature")
        ax2.set_xticks(x)
        ax2.set_xticklabels(self.sector_names)
        ax2.axhline(0.0, color="0.4", lw=0.8)
        ax2.set_title("Sector-Level Contributions")
        ax2.legend(frameon=False)
        return fig


def solve_exact_ces_network_prices(
    io_matrix: np.ndarray,
    expenditure_shares: np.ndarray,
    elasticities: np.ndarray | float,
    tfp_shocks: np.ndarray,
    *,
    final_elasticity: float = 1.0,
    tol: float = 1e-13,
    max_iter: int = 500,
) -> tuple[np.ndarray, float]:
    """Solve exact nonlinear CES unit-cost fixed point and exact log real GDP change."""
    Omega = np.asarray(io_matrix, dtype=np.float64)
    n = Omega.shape[0]
    b = np.asarray(expenditure_shares, dtype=np.float64).reshape(n)
    b = b / np.sum(b)
    z = np.asarray(tfp_shocks, dtype=np.float64).reshape(n)
    if np.ndim(elasticities) == 0:
        theta = np.full(n, float(elasticities), dtype=np.float64)
    else:
        theta = np.asarray(elasticities, dtype=np.float64).reshape(n)

    alpha = 1.0 - np.sum(Omega, axis=1)
    if np.any(alpha <= -1e-12):
        raise ValueError("Each row of io_matrix must sum to <= 1 (positive primary factor share required).")
    alpha = np.maximum(alpha, 0.0)

    # Initial guess from first-order Leontief inverse: ln p = -(I - Omega)^{-1} z
    eye_n = np.eye(n, dtype=np.float64)
    log_p = -np.linalg.solve(eye_n - Omega, z)

    for _ in range(max_iter):
        log_p_next = np.zeros(n, dtype=np.float64)
        J_share = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            th_i = float(theta[i])
            if abs(th_i - 1.0) < 1e-10:
                log_p_next[i] = -z[i] + float(Omega[i, :] @ log_p)
                J_share[i, :] = Omega[i, :]
            else:
                exponent = (1.0 - th_i) * log_p
                m_shift = max(0.0, float(np.max(exponent)))
                w_interm = Omega[i, :] * np.exp(exponent - m_shift)
                w_labor = float(alpha[i] * np.exp(-m_shift))
                total_ces = float(np.sum(w_interm)) + w_labor
                log_p_next[i] = -z[i] + (m_shift + np.log(max(total_ces, 1e-300))) / (1.0 - th_i)
                J_share[i, :] = w_interm / max(total_ces, 1e-300)
        res = log_p - log_p_next
        if float(np.max(np.abs(res))) < 1e-15:
            break
        step = np.linalg.solve(eye_n - J_share, res)
        log_p = log_p - step
        if float(np.max(np.abs(step))) < tol:
            break

    if abs(final_elasticity - 1.0) < 1e-8:
        log_P_y = float(b @ log_p)
    else:
        exp_f = (1.0 - final_elasticity) * log_p
        m_f = float(np.max(exp_f))
        log_P_y = (m_f + float(np.log(np.sum(b * np.exp(exp_f - m_f))))) / (1.0 - final_elasticity)

    exact_dlog_y = -log_P_y
    return log_p, exact_dlog_y


def baqaee_farhi_decomposition(
    io_matrix: np.ndarray,
    expenditure_shares: np.ndarray,
    elasticities: np.ndarray | float,
    tfp_shocks: np.ndarray,
    *,
    final_elasticity: float = 1.0,
    markups: np.ndarray | None = None,
    sector_names: Sequence[str] | None = None,
) -> NetworkMacroDecompositionResult:
    r"""Compute Baqaee–Farhi (2019, 2020) 2nd-order macroeconomic network decomposition.

    Parameters
    ----------
    io_matrix : np.ndarray
        $N \times N$ input-output revenue/cost matrix $\Omega$ where $\Omega_{ij}$ is
        sector $i$'s expenditure share on intermediate input $j$, with row sums $< 1$.
    expenditure_shares : np.ndarray
        Length-$N$ final consumption expenditure share vector $b$ ($\sum_i b_i = 1$).
    elasticities : np.ndarray or float
        Sectoral elasticities of substitution $\theta_i > 0$ across inputs.
        $\theta_i < 1$ represents gross complements (strong downstream amplification
        of negative supply shocks); $\theta_i = 1$ is Cobb–Douglas (Hulten exact);
        $\theta_i > 1$ is gross substitutes.
    tfp_shocks : np.ndarray
        Length-$N$ vector of log Hicks-neutral productivity shocks $\Delta \ln A_i$.
    final_elasticity : float, default=1.0
        Final demand CES elasticity of substitution $\theta_0 > 0$.
    markups : np.ndarray or None, default=None
        Optional length-$N$ gross markup vector $\mu_i \ge 1$ for Baqaee–Farhi (2020)
        distorted network allocative efficiency calculation.
    sector_names : Sequence[str] or None, default=None
        Optional human-readable sector labels.
    """
    Omega = np.asarray(io_matrix, dtype=np.float64)
    if Omega.ndim != 2 or Omega.shape[0] != Omega.shape[1]:
        raise ValueError(f"io_matrix must be square 2D array, got shape {Omega.shape}")
    n = Omega.shape[0]

    b = np.asarray(expenditure_shares, dtype=np.float64).reshape(n)
    if np.any(b < 0.0) or float(np.sum(b)) <= 0.0:
        raise ValueError("expenditure_shares must be non-negative with positive sum.")
    b = b / float(np.sum(b))

    if np.ndim(elasticities) == 0:
        theta = np.full(n, float(elasticities), dtype=np.float64)
    else:
        theta = np.asarray(elasticities, dtype=np.float64).reshape(n)
    if np.any(theta <= 0.0) or final_elasticity <= 0.0:
        raise ValueError("All elasticities of substitution must be strictly positive.")

    z = np.asarray(tfp_shocks, dtype=np.float64).reshape(n)
    mu = np.ones(n, dtype=np.float64) if markups is None else np.asarray(markups, dtype=np.float64).reshape(n)
    if np.any(mu <= 0.0):
        raise ValueError("markups must be strictly positive.")

    row_sums = np.sum(Omega, axis=1)
    if np.any(row_sums >= 1.0 - 1e-12):
        raise ValueError("Each row of io_matrix must sum to strictly less than 1 (Hawkins-Simon + positive labor share).")

    names = (
        tuple(str(s) for s in sector_names)
        if sector_names is not None
        else tuple(f"Sector_{i + 1}" for i in range(n))
    )

    # Cost-based I-O matrix and Leontief inverse Psi = (I - Omega)^{-1}
    eye_n = np.eye(n, dtype=np.float64)
    Psi = np.linalg.solve(eye_n - Omega, eye_n)

    # Revenue-based I-O matrix when markups mu_i != 1: Omega_rev[i, j] = Omega[i, j] / mu_i
    Omega_rev = Omega / mu[:, None]
    Psi_rev = np.linalg.solve(eye_n - Omega_rev, eye_n)
    domar = b @ Psi_rev
    cost_domar = b @ Psi

    # 1. First-order Hulten effect (using cost Domar weights in competitive equilibrium)
    sector_d1 = cost_domar * z
    d1_log_y = float(np.sum(sector_d1))

    # 2. Exact analytical Hessian H_{kl} = d^2 (ln Y) / (d z_k d z_l) at z = 0
    # Let v_j = (Psi z)_j = -d ln p_j^{(1)}.
    # For each producer i, Cov_{\tilde{\Omega}_i}(Psi_{\cdot, k}, Psi_{\cdot, l})
    # where labor has weight alpha_i = 1 - sum_j Omega_{ij} and value 0:
    # Cov_{\tilde{\Omega}_i}(u, w) = sum_j Omega_{ij} u_j w_j - (sum_j Omega_{ij} u_j)(sum_j Omega_{ij} w_j)
    hessian = np.zeros((n, n), dtype=np.float64)

    # Final demand node (node 0)
    b_psi = b @ Psi  # shape (n,)
    cov_0 = (Psi.T * b[None, :]) @ Psi - np.outer(b_psi, b_psi)
    hessian += (final_elasticity - 1.0) * cov_0

    node_d2 = np.zeros(n, dtype=np.float64)
    v = Psi @ z  # shape (n,)
    var_0 = float(np.sum(b * (v ** 2)) - (float(b @ v)) ** 2)
    final_d2 = 0.5 * (final_elasticity - 1.0) * var_0

    for i in range(n):
        om_i = Omega[i, :]
        om_psi = om_i @ Psi  # shape (n,)
        cov_i = (Psi.T * om_i[None, :]) @ Psi - np.outer(om_psi, om_psi)
        hessian += cost_domar[i] * (theta[i] - 1.0) * cov_i

        var_i = float(np.sum(om_i * (v ** 2)) - (float(om_i @ v)) ** 2)
        node_d2[i] = 0.5 * cost_domar[i] * (theta[i] - 1.0) * var_i

    d2_log_y = float(0.5 * (z @ hessian @ z))

    # 3. Baqaee–Farhi (2020, QJE) allocative efficiency shift from markup wedges:
    # When markups mu_i != 1, revenue Domar weights differ from cost Domar weights
    # and the network reallocation of labor share Lambda_L alters allocative efficiency.
    allocative_dlog_y = 0.0
    if markups is not None and not np.allclose(mu, 1.0):
        # Wedge gap between cost-based and revenue-based Domar sensitivity + curvature reallocation
        wedge_gap = float((domar - cost_domar) @ z)
        # Reallocation of expenditure shares under price changes d ln p = -v
        d_log_b = (1.0 - final_elasticity) * (-v + float(b @ v))
        d_domar_approx = (b * d_log_b) @ Psi_rev
        reallocation_effect = -float(np.sum((1.0 - 1.0 / mu) * d_domar_approx))
        allocative_dlog_y = wedge_gap + 0.5 * reallocation_effect

    total_d2 = d1_log_y + d2_log_y + allocative_dlog_y

    # 4. Exact nonlinear CES fixed point for ground-truth comparison
    log_p_exact, exact_ces_dlog_y = solve_exact_ces_network_prices(
        Omega,
        b,
        theta,
        z,
        final_elasticity=final_elasticity,
    )

    return NetworkMacroDecompositionResult(
        sector_names=names,
        domar_weights=domar,
        cost_domar_weights=cost_domar,
        leontief_inverse=Psi,
        first_order_dlog_y=d1_log_y,
        second_order_dlog_y=d2_log_y,
        allocative_efficiency_dlog_y=allocative_dlog_y,
        total_second_order_dlog_y=total_d2,
        exact_ces_dlog_y=exact_ces_dlog_y,
        hessian_matrix=hessian,
        sector_first_order_contributions=sector_d1,
        node_second_order_contributions=node_d2,
        final_demand_second_order=final_d2,
        equilibrium_log_prices=log_p_exact,
        tfp_shocks=z,
        elasticities=theta,
        final_elasticity=float(final_elasticity),
        markups=mu,
    )
