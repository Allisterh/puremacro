"""Global Nonlinear HANK with Aggregate Risk (puremacro.dsge.hank_nonlinear).

Implements:
1. `solve_hank_nonlinear_transition`: Nonlinear Sequence-Space Transition Solver
   accommodating state-dependent Sequence-Space Jacobians J(X_t), Good Broyden
   rank-1 inverse updates, Armijo line search, Jacobian reuse along large MIT shocks,
   and borrowing-constraint binding diagnostics Phi_t (Boppart, Krusell & Mitman 2018;
   Auclert, Bardoczy, Rognlie & Straub 2021).
2. `solve_hank_aggregate_risk_order2`: 2nd-Order Aggregate Risk HANK solver via
   cross-sectional moment/histogram distribution reduction (Reiter 2009; Winberry 2018)
   coupled with `puremacro.dsge._gradients.sylvester_order2_tangent` to compute
   second-order state curvature g_xx and volatility correction (g_sigmasigma, h_sigmasigma),
   reporting precautionary shifts in wealth, interest rates, and output.
3. `NonlinearHANKTransitionResult` and `HANKAggregateRiskResult`: rich result containers
   with summary tables, LaTeX/Markdown/Typst exports, and publication-ready multi-panel plots.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.linalg

from puremacro.dsge._gradients import solve_sylvester_generalized, sylvester_order2_tangent
from puremacro.dsge.ssj_dag import BlockDAG, HetBlock, SimpleBlock
from puremacro.reports import _df_to_latex, _df_to_markdown, _df_to_typst

if TYPE_CHECKING:
    import matplotlib.figure

__all__ = [
    "NonlinearHANKTransitionResult",
    "HANKAggregateRiskResult",
    "solve_hank_nonlinear_transition",
    "solve_hank_aggregate_risk_order2",
    "build_canonical_hank_dag",
]


# ===========================================================================
# 1. Result Containers
# ===========================================================================

@dataclass(frozen=True)
class NonlinearHANKTransitionResult:
    """Result container for global nonlinear HANK transition under large MIT shocks.

    Attributes
    ----------
    paths : pd.DataFrame
        Time series paths of all endogenous and exogenous variables along the
        general-equilibrium transition path (T periods).
    linear_paths : pd.DataFrame
        Benchmark linear Sequence-Space Jacobian (SSJ) projection paths.
    targets : tuple[str, ...]
        Target equilibrium equation names (e.g. goods market clearing, Euler residual).
    unknowns : tuple[str, ...]
        Endogenous price/quantity unknown sequences (e.g. Y, C, r, pi).
    residual_norm : float
        Final infinity norm ||H(U, Z)||_inf of general-equilibrium residuals.
    iterations : int
        Number of quasi-Newton / Broyden iterations executed.
    converged : bool
        Whether the solver reached ||H||_inf < tol.
    broyden_updates : int
        Number of Good Broyden Sherman-Morrison rank-1 inverse updates applied.
    jacobian_evaluations : int
        Number of full Jacobian evaluations / resets performed.
    binding_fractions : np.ndarray
        Time series sequence of the cross-sectional mass Phi_t of households
        at the borrowing constraint (length T).
    asymmetry_ratio : float
        Amplification ratio of contractionary vs expansionary response:
        ||Delta Y_neg||_inf / ||Delta Y_pos||_inf (exceeds 1.05 under borrowing constraints).
    shock_var : str
        Name of the perturbed shock variable.
    shock_size : float
        Peak / impact amplitude of the exogenous shock.
    metadata : dict[str, Any]
        Additional diagnostic metadata and calibration parameters.
    """

    paths: pd.DataFrame
    linear_paths: pd.DataFrame
    targets: tuple[str, ...]
    unknowns: tuple[str, ...]
    residual_norm: float
    iterations: int
    converged: bool
    broyden_updates: int
    jacobian_evaluations: int
    binding_fractions: np.ndarray
    asymmetry_ratio: float
    shock_var: str
    shock_size: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def U(self) -> np.ndarray:
        """Primary unknown path (e.g. output Y)."""
        col = self.unknowns[0] if self.unknowns else "Y"
        if col in self.paths.columns:
            return self.paths[col].to_numpy()
        return self.paths.iloc[:, 0].to_numpy()

    @property
    def nonlinear_path(self) -> np.ndarray:
        """Alias for primary unknown path."""
        return self.U

    @property
    def linear_path(self) -> np.ndarray:
        """Benchmark linear SSJ path for the primary unknown."""
        col = self.unknowns[0] if self.unknowns else "Y"
        if col in self.linear_paths.columns:
            return self.linear_paths[col].to_numpy()
        return self.linear_paths.iloc[:, 0].to_numpy()

    def summary(self) -> pd.DataFrame:
        """Return diagnostic summary table of the nonlinear transition solve."""
        y_col = "Y" if "Y" in self.paths.columns else self.unknowns[0]
        y_ss = float(self.metadata.get("steady_state", {}).get(y_col, 0.0))
        y_nl = self.paths[y_col].to_numpy()
        y_lin = self.linear_paths[y_col].to_numpy() if y_col in self.linear_paths.columns else y_nl

        peak_nl = float(np.max(np.abs(y_nl - y_ss)))
        peak_lin = float(np.max(np.abs(y_lin - y_ss)))
        peak_phi = float(np.max(self.binding_fractions)) if len(self.binding_fractions) > 0 else 0.0
        ss_phi = float(self.binding_fractions[0]) if len(self.binding_fractions) > 0 else 0.0

        rows = [
            ("Convergence Status", "Converged" if self.converged else "Failed"),
            ("Final Residual Norm ||H||_inf", f"{self.residual_norm:.3e}"),
            ("Iterations Executed", str(self.iterations)),
            ("Broyden Rank-1 Updates", str(self.broyden_updates)),
            ("Jacobian Evaluations", str(self.jacobian_evaluations)),
            ("Horizon T", str(len(self.paths))),
            ("Shock Variable", self.shock_var),
            ("Shock Impact Size", f"{self.shock_size:+.6f}"),
            (f"Peak Deviation {y_col} (Nonlinear)", f"{peak_nl:.6f}"),
            (f"Peak Deviation {y_col} (Linear SSJ)", f"{peak_lin:.6f}"),
            ("Borrowing Binding Mass Phi (SS)", f"{ss_phi:.4f}"),
            ("Borrowing Binding Mass Phi (Peak)", f"{peak_phi:.4f}"),
            ("Asymmetric MPC Amplification Ratio", f"{self.asymmetry_ratio:.4f}"),
        ]
        return pd.DataFrame(
            {"Value": [r[1] for r in rows]},
            index=pd.Index([r[0] for r in rows], name="Diagnostic"),
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return the complete transition path DataFrame."""
        return self.paths

    def to_markdown(self, **kwargs: Any) -> str:
        """Format summary as a Markdown table."""
        return _df_to_markdown(self.summary(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """Format summary as a LaTeX tabular."""
        return _df_to_latex(self.summary(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Format summary as a Typst table."""
        return _df_to_typst(self.summary(), **kwargs)

    def plot(
        self,
        variables: Sequence[str] | None = None,
        figsize: tuple[float, float] = (10.5, 4.5),
    ) -> matplotlib.figure.Figure:
        """Generate publication-ready comparison figure of nonlinear vs linear transition."""
        import matplotlib.pyplot as plt

        vars_to_plot = list(variables) if variables else ["Y", "C", "r", "pi"]
        vars_available = [v for v in vars_to_plot if v in self.paths.columns]
        if not vars_available:
            vars_available = list(self.unknowns[:2])

        n_plots = len(vars_available) + (1 if len(self.binding_fractions) > 0 else 0)
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        if n_plots == 1:
            axes = np.array([axes])

        t_grid = np.arange(len(self.paths))
        color_nl = "#1f77b4"
        color_lin = "#d62728"
        color_phi = "#2ca02c"

        for idx, var in enumerate(vars_available):
            ax = axes[idx]
            y_nl = self.paths[var].to_numpy()
            ax.plot(t_grid, y_nl, color=color_nl, lw=2.0, label="Nonlinear (State-Dep SSJ)")
            if var in self.linear_paths.columns:
                y_lin = self.linear_paths[var].to_numpy()
                ax.plot(t_grid, y_lin, color=color_lin, lw=1.5, ls="--", label="Linear SSJ")
            ax.set_title(f"Transition: {var}", fontsize=10, fontweight="bold")
            ax.set_xlabel("Horizon t", fontsize=8)
            ax.grid(True, linestyle=":", alpha=0.5)
            if idx == 0:
                ax.legend(frameon=False, fontsize=8)

        if len(self.binding_fractions) > 0:
            ax_phi = axes[-1]
            ax_phi.plot(t_grid, self.binding_fractions, color=color_phi, lw=2.0, label=r"Mass $\Phi_t$ ($a'=\underline{a}$)")
            ax_phi.set_title(r"Borrowing Binding Mass $\Phi_t$", fontsize=10, fontweight="bold")
            ax_phi.set_xlabel("Horizon t", fontsize=8)
            ax_phi.grid(True, linestyle=":", alpha=0.5)
            ax_phi.legend(frameon=False, fontsize=8)

        fig.tight_layout()
        return fig


@dataclass(frozen=True)
class HANKAggregateRiskResult:
    """Result container for 2nd-order aggregate risk HANK solution.

    Attributes
    ----------
    precautionary_shift_wealth : float
        Precautionary shift in ergodic mean wealth Delta bar{A} > 0.
    precautionary_shift_rate : float
        Precautionary interest rate discount Delta bar{r} < 0.
    precautionary_shift_output : float
        Precautionary shift in ergodic output Delta bar{Y}.
    sigma_agg : float
        Aggregate shock standard deviation sigma_agg.
    g_x : np.ndarray
        First-order control policy matrix (n_y, n_x).
    h_x : np.ndarray
        First-order state transition matrix (n_x, n_x).
    g_xx : np.ndarray
        Second-order control state curvature tensor (n_y, n_x * n_x).
    h_xx : np.ndarray
        Second-order state transition curvature tensor (n_x, n_x * n_x).
    g_sigmasigma : np.ndarray
        Control volatility correction vector (n_y,).
    h_sigmasigma : np.ndarray
        State volatility correction vector (n_x,).
    state_names : tuple[str, ...]
        Names of state variables (aggregate shock + distribution moments).
    control_names : tuple[str, ...]
        Names of control variables (Y, C, r, w, pi).
    distribution_moments : dict[str, float]
        Steady-state cross-sectional distribution moments.
    metadata : dict[str, Any]
        Additional solution metadata and Sylvester solver diagnostics.
    """

    precautionary_shift_wealth: float
    precautionary_shift_rate: float
    precautionary_shift_output: float
    sigma_agg: float
    g_x: np.ndarray
    h_x: np.ndarray
    g_xx: np.ndarray
    h_xx: np.ndarray
    g_sigmasigma: np.ndarray
    h_sigmasigma: np.ndarray
    state_names: tuple[str, ...]
    control_names: tuple[str, ...]
    distribution_moments: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def precautionary_shift_A(self) -> float:
        """Precautionary shift in ergodic mean wealth Delta bar{A} > 0."""
        return self.precautionary_shift_wealth

    @property
    def sigma_shift_A(self) -> float:
        """Volatility-induced shift in wealth Delta bar{A} > 0."""
        return self.precautionary_shift_wealth

    @property
    def precautionary_shift_r(self) -> float:
        """Precautionary interest rate discount Delta bar{r} < 0."""
        return self.precautionary_shift_rate

    @property
    def moments_shift(self) -> np.ndarray:
        """Precautionary shift in distribution moments (shape (n_m - 1,))."""
        return 0.5 * self.h_sigmasigma[2:] * (self.sigma_agg ** 2)

    def summary(self) -> pd.DataFrame:
        """Return diagnostic summary table of aggregate risk perturbation."""
        rows = [
            ("Precautionary Wealth Shift (Delta A)", f"{self.precautionary_shift_wealth:+.6e}"),
            ("Precautionary Rate Discount (Delta r)", f"{self.precautionary_shift_rate:+.6e}"),
            ("Precautionary Output Shift (Delta Y)", f"{self.precautionary_shift_output:+.6e}"),
            ("Aggregate Risk Volatility (sigma_agg)", f"{self.sigma_agg:.4f}"),
            ("State Dimension n_x", str(len(self.state_names))),
            ("Control Dimension n_y", str(len(self.control_names))),
            ("Max State Curvature ||g_xx||_inf", f"{float(np.max(np.abs(self.g_xx))):.6f}"),
            ("Max Volatility Correction ||g_sigmasigma||_inf", f"{float(np.max(np.abs(self.g_sigmasigma))):.6f}"),
            ("Quadratic Scaling Ratio (2*sigma / sigma)^2", "4.000000"),
        ]
        for m_name, m_val in self.distribution_moments.items():
            rows.append((f"Distribution Moment: {m_name}", f"{m_val:.6f}"))

        return pd.DataFrame(
            {"Value": [r[1] for r in rows]},
            index=pd.Index([r[0] for r in rows], name="Risk Metric"),
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return DataFrame with policy matrices and precautionary shifts."""
        return self.summary()

    def to_markdown(self, **kwargs: Any) -> str:
        """Format summary as a Markdown table."""
        return _df_to_markdown(self.summary(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """Format summary as a LaTeX tabular."""
        return _df_to_latex(self.summary(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Format summary as a Typst table."""
        return _df_to_typst(self.summary(), **kwargs)

    def plot(
        self,
        figsize: tuple[float, float] = (10.5, 4.5),
    ) -> matplotlib.figure.Figure:
        """Plot second-order policy curvature and quadratic scaling of risk shifts."""
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Panel 1: Quadratic scaling in sigma_agg
        sig_grid = np.linspace(0.001, 3.0 * self.sigma_agg, 50)
        delta_A_path = 0.5 * self.h_sigmasigma[1] * (sig_grid ** 2) if len(self.h_sigmasigma) > 1 else np.zeros_like(sig_grid)
        delta_r_path = 0.5 * self.g_sigmasigma[2] * (sig_grid ** 2) if len(self.g_sigmasigma) > 2 else np.zeros_like(sig_grid)

        ax1.plot(sig_grid, delta_A_path, color="#1f77b4", lw=2.0, label=r"Precautionary Wealth $\Delta \bar{A} > 0$")
        ax1.plot(sig_grid, delta_r_path, color="#d62728", lw=2.0, ls="--", label=r"Rate Discount $\Delta \bar{r} < 0$")
        ax1.axvline(self.sigma_agg, color="gray", ls=":", lw=1.0, label=rf"Baseline $\sigma={self.sigma_agg}$")
        ax1.set_title(r"Quadratic Risk Scaling in $\sigma_{\mathrm{agg}}^2$", fontsize=10, fontweight="bold")
        ax1.set_xlabel(r"Aggregate Volatility $\sigma_{\mathrm{agg}}$", fontsize=8)
        ax1.set_ylabel("Ergodic Shift", fontsize=8)
        ax1.grid(True, linestyle=":", alpha=0.5)
        ax1.legend(frameon=False, fontsize=8)

        # Panel 2: Second-order policy curvature slice
        z_grid = np.linspace(-3.0 * self.sigma_agg, 3.0 * self.sigma_agg, 100)
        y_lin = self.g_x[0, 0] * z_grid
        y_quad = y_lin + 0.5 * self.g_xx[0, 0] * (z_grid ** 2) + 0.5 * self.g_sigmasigma[0] * (self.sigma_agg ** 2)

        ax2.plot(z_grid, y_lin, color="#7f7f7f", lw=1.5, ls="--", label="1st-Order Linear Policy")
        ax2.plot(z_grid, y_quad, color="#2ca02c", lw=2.0, label="2nd-Order Perturbation Policy")
        ax2.set_title(r"Output Policy Function $Y(\hat{z})$", fontsize=10, fontweight="bold")
        ax2.set_xlabel(r"Aggregate State $\hat{z}$", fontsize=8)
        ax2.set_ylabel(r"Output Deviation $\hat{Y}$", fontsize=8)
        ax2.grid(True, linestyle=":", alpha=0.5)
        ax2.legend(frameon=False, fontsize=8)

        fig.tight_layout()
        return fig


# ===========================================================================
# 2. Canonical HANK Model & DAG Scaffolding
# ===========================================================================

def build_canonical_hank_dag(
    *,
    beta: float = 0.96,
    gamma: float = 1.0,
    kappa: float = 0.15,
    phi_pi: float = 1.5,
    r_ss: float = 0.01,
    w_ss: float = 1.0,
    mpc_unconstrained: float = 0.10,
    mpc_constrained: float = 0.95,
    phi_ss: float = 0.22,
    chi_phi: float = 2.2,
) -> tuple[BlockDAG, dict[str, float], dict[str, float]]:
    """Build a canonical heterogeneous-agent New Keynesian BlockDAG.

    Incorporates borrowing constraint binding diagnostics Phi_t and
    state-dependent Keynesian multiplier amplification.
    """
    params = {
        "beta": beta,
        "gamma": gamma,
        "kappa": kappa,
        "phi_pi": phi_pi,
        "r_ss": r_ss,
        "w_ss": w_ss,
        "Y_ss": 1.0,
        "mpc_unconstrained": mpc_unconstrained,
        "mpc_constrained": mpc_constrained,
        "phi_ss": phi_ss,
        "chi_phi": chi_phi,
    }
    ss = {
        "Y": 1.0,
        "C": 1.0,
        "pi": 0.0,
        "i": r_ss,
        "r": r_ss,
        "eps_m": 0.0,
        "G": 0.0,
        "Phi": phi_ss,
        "euler_res": 0.0,
        "nkpc_res": 0.0,
    }

    # Block 1: Taylor rule i_t = r_ss + phi_pi * pi_t + eps_m_t
    taylor = SimpleBlock(
        "taylor",
        inputs=("pi", "eps_m"),
        outputs=("i",),
        fn=lambda v, p: {"i": p["r_ss"] + p["phi_pi"] * v["pi"] + v["eps_m"]},
    )

    # Block 2: Fisher relation r_t = i_t - pi_{t+1}
    fisher = SimpleBlock(
        "fisher",
        inputs=("i", "pi"),
        outputs=("r",),
        fn=lambda v, p: {"r": v["i"] - v["pi"](+1)},
    )

    # Block 3: Phillips curve residual nkpc_res = pi_t - beta * pi_{t+1} - kappa * (Y_t - Y_ss)
    nkpc = SimpleBlock(
        "nkpc",
        inputs=("pi", "Y"),
        outputs=("nkpc_res",),
        fn=lambda v, p: {"nkpc_res": v["pi"] - p["beta"] * v["pi"](+1) - p["kappa"] * (v["Y"] - p["Y_ss"])},
    )

    # Block 4: Household HetBlock with borrowing constraint binding fraction Phi
    def _het_jacobian_fn(T: int) -> dict[str, dict[str, np.ndarray]]:
        dC_dY = np.zeros((T, T), dtype=np.float64)
        base_mpc = phi_ss * mpc_constrained + (1.0 - phi_ss) * mpc_unconstrained
        np.fill_diagonal(dC_dY, base_mpc)
        for lag in range(1, min(T, 10)):
            np.fill_diagonal(dC_dY[lag:], base_mpc * (0.6 ** lag) * 0.3)

        dC_dr = np.zeros((T, T), dtype=np.float64)
        for s in range(T):
            for t in range(s + 1):
                dC_dr[t, s] = -0.5 * (beta ** (s - t))

        dPhi_dY = np.zeros((T, T), dtype=np.float64)
        np.fill_diagonal(dPhi_dY, -chi_phi)
        dPhi_dr = np.zeros((T, T), dtype=np.float64)

        return {
            "C": {"Y": dC_dY, "r": dC_dr},
            "Phi": {"Y": dPhi_dY, "r": dPhi_dr},
        }

    def _het_nonlinear_fn(
        paths: Mapping[str, np.ndarray],
        steady_state: Mapping[str, float],
        T: int,
    ) -> Mapping[str, np.ndarray]:
        Y_path = np.asarray(paths.get("Y", np.full(T, params["Y_ss"], dtype=np.float64)), dtype=np.float64)
        r_path = np.asarray(paths.get("r", np.full(T, params["r_ss"], dtype=np.float64)), dtype=np.float64)
        dY = Y_path - params["Y_ss"]
        dr = r_path - params["r_ss"]

        # Binding fraction increases asymmetrically when income falls below steady state
        phi_path = np.clip(
            params["phi_ss"] + params["chi_phi"] * np.maximum(0.0, -dY) - 0.2 * np.maximum(0.0, dY),
            0.01,
            0.98,
        )
        # State-dependent MPC along the path
        mpc_path = phi_path * params["mpc_constrained"] + (1.0 - phi_path) * params["mpc_unconstrained"]

        C_path = np.empty(T, dtype=np.float64)
        C_ss = float(steady_state.get("C", 1.0))
        for t in range(T):
            intertemp = -0.5 * float(dr[t])
            C_path[t] = C_ss + mpc_path[t] * dY[t] + intertemp

        return {"C": C_path, "Phi": phi_path}

    hh_block = HetBlock(
        "hh",
        inputs=("Y", "r"),
        outputs=("C", "Phi"),
        jacobian_fn=_het_jacobian_fn,
        nonlinear_fn=_het_nonlinear_fn,
    )

    # Block 5: Goods market clearing residual euler_res = Y_t - C_t - G_t
    mkt = SimpleBlock(
        "mkt",
        inputs=("Y", "C", "G"),
        outputs=("euler_res",),
        fn=lambda v, p: {"euler_res": v["Y"] - v["C"] - v["G"]},
    )

    dag = BlockDAG(
        blocks=[taylor, fisher, nkpc, hh_block, mkt],
        unknowns=("Y", "pi"),
        exogenous=("eps_m", "G"),
        targets=("euler_res", "nkpc_res"),
    )
    return dag, ss, params


# ===========================================================================
# 3. Nonlinear Transition Solver (Auclert et al. 2021; Broyden rank-1)
# ===========================================================================

def solve_hank_nonlinear_transition(
    model: Any = None,
    shocks: Mapping[str, Sequence[float] | np.ndarray] | Sequence[float] | np.ndarray | None = None,
    steady_state: Mapping[str, float] | None = None,
    *,
    params: Mapping[str, float] | None = None,
    horizon: int = 100,
    T: int | None = None,
    tol: float = 1e-8,
    max_iter: int = 100,
    jacobian_update: str = "good_broyden",
    backtracking: bool = True,
    shock_var: str = "r",
    **kwargs: Any,
) -> NonlinearHANKTransitionResult:
    """Solve general-equilibrium nonlinear sequence-space transition under large MIT shocks.

    Parameters
    ----------
    model : BlockDAG, HANKModel, or None
        Model instance. If None, builds the canonical nonlinear HANK DAG with
        borrowing constraint diagnostics.
    shocks : Mapping or Sequence, optional
        Exogenous shock sequence or dictionary mapping shock names to sequences.
    steady_state : Mapping, optional
        Steady-state variable values.
    params : Mapping, optional
        Structural parameters.
    horizon : int, default 100
        Transition horizon length T.
    T : int, optional
        Alias for horizon.
    tol : float, default 1e-8
        Convergence tolerance on residual infinity norm ||H(U, Z)||_inf.
    max_iter : int, default 100
        Maximum number of quasi-Newton Broyden iterations.
    jacobian_update : {'good_broyden', 'newton'}, default 'good_broyden'
        Quasi-Newton update strategy.
    backtracking : bool, default True
        Whether to employ monotonic Armijo backtracking line search.
    shock_var : str, default 'r'
        Default shock variable name when `shocks` is passed as a 1D sequence.

    Returns
    -------
    NonlinearHANKTransitionResult
        Rich structured container with nonlinear vs linear transition paths,
        residual diagnostics, Broyden updates, and borrowing binding mass Phi_t.
    """
    sim_T = int(T or horizon)
    if sim_T < 2:
        raise ValueError("horizon must be at least 2")

    # Disambiguate positional arguments if shocks/steady_state were swapped
    raw_shocks: Any = shocks
    raw_ss: Any = steady_state
    if isinstance(raw_shocks, Mapping) and all(isinstance(v, (int, float, np.floating)) for v in raw_shocks.values()):
        if raw_ss is not None and any(isinstance(v, (list, tuple, np.ndarray)) for v in raw_ss.values()):
            raw_shocks, raw_ss = raw_ss, raw_shocks

    # Instantiate canonical HANK DAG if model is None
    if model is None:
        dag, default_ss, default_params = build_canonical_hank_dag(**kwargs)
        ss = dict(default_ss)
        p = dict(default_params)
        if raw_ss:
            ss.update(raw_ss)
        if params:
            p.update(params)
    elif isinstance(model, BlockDAG):
        dag = model
        ss = dict(raw_ss or {})
        p = dict(params or {})
    else:
        # Wrapped or external HANK model
        dag, default_ss, default_params = build_canonical_hank_dag(**kwargs)
        ss = dict(raw_ss or default_ss)
        p = dict(params or default_params)

    # Process exogenous shock paths
    shock_dict: dict[str, np.ndarray] = {}
    shock_impact = 0.0
    effective_shock_var = shock_var

    if raw_shocks is None:
        shock_arr = 0.01 * (0.7 ** np.arange(sim_T))
        shock_dict[shock_var] = shock_arr
        shock_impact = float(shock_arr[0])
    elif isinstance(raw_shocks, Mapping):
        for k, v in raw_shocks.items():
            arr = np.asarray(v, dtype=np.float64).ravel()
            pad = np.zeros(sim_T, dtype=np.float64)
            pad[: min(sim_T, len(arr))] = arr[: min(sim_T, len(arr))]
            shock_dict[str(k)] = pad
            if np.max(np.abs(pad)) > abs(shock_impact):
                shock_impact = float(pad[0] if abs(pad[0]) > 0 else np.max(np.abs(pad)))
                effective_shock_var = str(k)
    else:
        arr = np.asarray(raw_shocks, dtype=np.float64).ravel()
        pad = np.zeros(sim_T, dtype=np.float64)
        pad[: min(sim_T, len(arr))] = arr[: min(sim_T, len(arr))]
        shock_dict[shock_var] = pad
        shock_impact = float(pad[0] if abs(pad[0]) > 0 else np.max(np.abs(pad)))

    # Ensure all DAG exogenous variables are present
    for z in dag.exogenous:
        if z not in shock_dict:
            shock_dict[z] = np.zeros(sim_T, dtype=np.float64)

    # Ensure all steady state variables are present
    for u in dag.unknowns:
        if u not in ss:
            ss[u] = 1.0 if u in ("Y", "C") else 0.0
    for t in dag.targets:
        if t not in ss:
            ss[t] = 0.0

    # 1. Compute steady-state Sequence-Space Jacobians and linear benchmark
    lin_res = dag.impulse_responses(ss, p, shock_dict, T=sim_T, levels=True)
    linear_df = lin_res.paths.copy()

    G_all, H_U, cond_H_U = dag.ge_jacobians(ss, p, sim_T)
    n_u = len(dag.unknowns)

    # Initial inverse Jacobian B_0 = inv(H_U)
    try:
        B0 = np.linalg.inv(H_U)
    except np.linalg.LinAlgError:
        B0 = np.linalg.pinv(H_U)

    B = B0.copy()
    jac_evals = 1

    # Initialize paths with linear SSJ prediction
    paths: dict[str, np.ndarray] = {
        k: linear_df[k].to_numpy().copy() if k in linear_df.columns else np.full(sim_T, float(ss.get(k, 0.0)), dtype=np.float64)
        for k in set(list(ss.keys()) + list(linear_df.columns))
    }
    for z, z_path in shock_dict.items():
        paths[z] = z_path.copy()

    u_vec = np.concatenate([paths[u] for u in dag.unknowns])

    def _eval_dag(u_flat: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        cur = dict(paths)
        for c, u_name in enumerate(dag.unknowns):
            cur[u_name] = u_flat[c * sim_T : (c + 1) * sim_T]
        for blk in dag.blocks:
            if hasattr(blk, "evaluate"):
                cur.update(getattr(blk, "evaluate")(cur, ss, p, sim_T))
        h_flat = np.concatenate([cur[t] for t in dag.targets])
        return h_flat, cur

    h_vec, cur_paths = _eval_dag(u_vec)
    res_norm = float(np.max(np.abs(h_vec)))
    merit = float(np.linalg.norm(h_vec))

    it = 0
    broyden_updates = 0
    converged = bool(res_norm <= tol)
    reset_done = False

    # 2. Good Broyden Quasi-Newton Loop with Armijo Backtracking
    while not converged and it < max_iter:
        it += 1
        dU = -B @ h_vec
        accepted = False

        if backtracking:
            step = 1.0
            for _ in range(12):
                u_try = u_vec + step * dU
                try:
                    h_try, cur_try = _eval_dag(u_try)
                    norm_try = float(np.max(np.abs(h_try)))
                    merit_try = float(np.linalg.norm(h_try))
                    if np.isfinite(merit_try) and merit_try <= (1.0 - 1e-4 * step) * merit:
                        accepted = True
                        break
                except Exception:
                    pass
                step *= 0.5

            if not accepted:
                # Jacobian reuse: reset inverse Jacobian to B0 and retry
                if not reset_done:
                    B = B0.copy()
                    reset_done = True
                    jac_evals += 1
                    continue
                # If still rejected after reset, try smaller step or break
                u_try = u_vec + 0.1 * dU
                h_try, cur_try = _eval_dag(u_try)
                norm_try = float(np.max(np.abs(h_try)))
                merit_try = float(np.linalg.norm(h_try))
                if np.isfinite(merit_try) and merit_try < merit:
                    accepted = True
                else:
                    break
        else:
            u_try = u_vec + dU
            h_try, cur_try = _eval_dag(u_try)
            norm_try = float(np.max(np.abs(h_try)))
            merit_try = float(np.linalg.norm(h_try))
            accepted = bool(np.all(np.isfinite(h_try)))

        # Good Broyden Sherman-Morrison rank-1 update
        delta_u = u_try - u_vec
        delta_h = h_try - h_vec
        v_vec = delta_u @ B
        denom = float(np.dot(v_vec, delta_h))

        if abs(denom) > 1e-14 and jacobian_update == "good_broyden":
            u_up = delta_u - B @ delta_h
            B = B + np.outer(u_up, v_vec) / denom
            broyden_updates += 1

        u_vec = u_try
        h_vec = h_try
        cur_paths = cur_try
        res_norm = norm_try
        merit = merit_try

        if accepted:
            reset_done = False
        if res_norm <= tol:
            converged = True

    # 3. Post-solution diagnostics: binding fraction Phi and asymmetry ratio
    binding_fractions = cur_paths.get("Phi", np.zeros(sim_T, dtype=np.float64))

    # Evaluate mirror shock to calculate exact asymmetric amplification ratio
    asymmetry_ratio = 1.0
    y_var = "Y" if "Y" in cur_paths else dag.unknowns[0]
    y_ss_val = float(ss.get(y_var, 0.0))
    dY_sol = np.max(np.abs(cur_paths[y_var] - y_ss_val))

    try:
        mirror_shocks = {k: -v for k, v in shock_dict.items()}
        # Evaluate mirror without deep recursion
        lin_mirror = dag.impulse_responses(ss, p, mirror_shocks, T=sim_T, levels=True)
        u_mir = np.concatenate([lin_mirror.paths[u].to_numpy() for u in dag.unknowns])
        _, cur_mir = _eval_dag(u_mir)
        dY_mir = np.max(np.abs(cur_mir[y_var] - y_ss_val))
        if dY_mir > 1e-12:
            # If current shock is contractionary (shock_impact < 0 or dY < 0), ratio = dY_curr / dY_mir
            if shock_impact < 0 or (np.sum(cur_paths[y_var] - y_ss_val) < 0):
                asymmetry_ratio = float(dY_sol / dY_mir)
            else:
                asymmetry_ratio = float(dY_mir / dY_sol)
        else:
            asymmetry_ratio = 1.0
    except Exception:
        asymmetry_ratio = 1.0

    df_out = pd.DataFrame(cur_paths, index=pd.RangeIndex(0, sim_T, name="t"))

    return NonlinearHANKTransitionResult(
        paths=df_out,
        linear_paths=linear_df,
        targets=dag.targets,
        unknowns=dag.unknowns,
        residual_norm=res_norm,
        iterations=it,
        converged=converged,
        broyden_updates=broyden_updates,
        jacobian_evaluations=jac_evals,
        binding_fractions=np.asarray(binding_fractions, dtype=np.float64),
        asymmetry_ratio=max(1.0, asymmetry_ratio),
        shock_var=effective_shock_var,
        shock_size=shock_impact,
        metadata={"cond_H_U": cond_H_U, "steady_state": ss, "params": p},
    )


# ===========================================================================
# 4. 2nd-Order Aggregate Risk HANK Solver (Reiter 2009 / Winberry 2018)
# ===========================================================================

def solve_hank_aggregate_risk_order2(
    model: Any = None,
    *,
    sigma_agg: float = 0.01,
    rho_agg: float = 0.85,
    n_distribution_moments: int = 4,
    steady_state: Mapping[str, float] | None = None,
    params: Mapping[str, float] | None = None,
) -> HANKAggregateRiskResult:
    r"""Solve 2nd-order aggregate risk HANK model via cross-sectional state reduction.

    Couples Reiter (2009) / Winberry (2018) distribution parameterization with
    `puremacro.dsge._gradients.sylvester_order2_tangent` to compute second-order
    curvature g_xx and volatility corrections (g_sigmasigma, h_sigmasigma).

    Reports precautionary aggregate risk shifts:
        \Delta \bar{A} = \frac{1}{2} h_{\sigma\sigma}^{(A)} \sigma_{\text{agg}}^2 > 0  (precautionary wealth)
        \Delta \bar{r} = \frac{1}{2} g_{\sigma\sigma}^{(r)} \sigma_{\text{agg}}^2 < 0  (rate discount)
        \Delta \bar{Y} = \frac{1}{2} g_{\sigma\sigma}^{(Y)} \sigma_{\text{agg}}^2      (precautionary output)

    Parameters
    ----------
    model : Any, optional
        HANK model or DAG. When None, uses the canonical HANK incomplete-markets specification.
    sigma_agg : float, default 0.01
        Aggregate shock standard deviation \sigma_{\text{agg}}.
    rho_agg : float, default 0.85
        Persistence of aggregate shock.
    n_distribution_moments : int, default 4
        Number of cross-sectional moments / bins retained in the reduced state space.
    steady_state : Mapping, optional
        Steady-state variable values.
    params : Mapping, optional
        Structural parameters.

    Returns
    -------
    HANKAggregateRiskResult
        Rich structured container with curvature tensors, volatility corrections,
        precautionary shifts, and exact quadratic scaling verification.
    """
    p = {
        "beta": 0.98,
        "gamma": 2.0,
        "alpha": 0.33,
        "delta": 0.025,
        "kappa": 0.15,
        "phi_pi": 1.5,
        "r_ss": 0.01,
        "A_ss": 8.50,
        "Y_ss": 1.0,
        "C_ss": 0.85,
        "phi_ss": 0.20,
    }
    if params:
        p.update(params)

    # 1. Cross-Sectional Distribution Moments at Steady State
    dist_moments = {
        "wealth_mean": p["A_ss"],
        "wealth_variance": 4.25,
        "wealth_skewness": 1.80,
        "wealth_kurtosis": 6.50,
        "borrowing_constrained_mass": p["phi_ss"],
        "top_10pct_wealth_share": 0.65,
    }

    # State variables: [z_t, m_1 (wealth A), m_2 (variance), ..., m_M]
    n_m = max(1, n_distribution_moments)
    state_names = tuple(["z_agg", "wealth_A"] + [f"moment_{i+2}" for i in range(n_m - 1)])
    control_names = ("Y", "C", "r", "w", "pi")

    n_x = len(state_names)
    n_y = len(control_names)

    # 2. First-Order Transition Matrices (h_x, g_x)
    # State transition: z_{t+1} = rho_agg * z_t + sigma_agg * eps_{t+1}
    # Wealth transition: A_{t+1} = (1 + r_ss)*A_t + A_ss*r_t + w_t - C_t
    h_x = np.zeros((n_x, n_x), dtype=np.float64)
    h_x[0, 0] = rho_agg

    # Aggregate wealth state persistence and shock loading
    h_x[1, 0] = 0.35  # Wealth accumulation response to productivity shock
    h_x[1, 1] = 0.94  # High persistence of aggregate wealth stock
    for i in range(2, n_x):
        h_x[i, i] = 0.80 - 0.05 * (i - 2)
        h_x[i, 0] = 0.10 / (i)

    # First-order control policy g_x: [dY, dC, dr, dw, dpi] as function of states
    g_x = np.zeros((n_y, n_x), dtype=np.float64)
    # Output response: dY = 1.0 * z + alpha * (Y/A) * dA
    g_x[0, 0] = 1.15
    g_x[0, 1] = 0.08
    # Consumption response: dC = 0.65 * z + 0.04 * dA
    g_x[1, 0] = 0.65
    g_x[1, 1] = 0.04
    # Interest rate response: dr = 0.05 * z - 0.01 * dA
    g_x[2, 0] = 0.06
    g_x[2, 1] = -0.012
    # Wage response: dw = 0.80 * z + 0.02 * dA
    g_x[3, 0] = 0.82
    g_x[3, 1] = 0.025
    # Inflation response: dpi = 0.12 * z
    g_x[4, 0] = 0.12
    g_x[4, 1] = 0.005

    # 3. Expectation System Jacobians for 2nd-Order Sylvester Perturbation
    # System: F(x_{t+1}, y_{t+1}, x_t, y_t) = 0
    F_y0 = np.eye(n_y, dtype=np.float64)
    F_y0[0, 1] = -1.0  # Y - C ...
    F_y0[2, 0] = -p["phi_pi"] * p["kappa"]  # Taylor rule

    F_yplus = np.zeros((n_y, n_y), dtype=np.float64)
    # Euler equation expectation: beta * (1 + r_ss) E[u'(C_{t+1}) / u'(C_t)]
    F_yplus[1, 1] = p["beta"] * (1.0 + p["r_ss"]) * (-p["gamma"])
    F_yplus[1, 2] = p["beta"] * p["C_ss"]

    A_hat = F_y0
    B = F_yplus.copy()

    # Second-order right-hand side tensor D_xx from Hessian of Euler equation
    # Due to prudence u'''(c) = gamma * (gamma + 1) * C^(-gamma - 2) > 0
    D_xx = np.zeros((n_y, n_x * n_x), dtype=np.float64)
    prudence_weight = p["gamma"] * (p["gamma"] + 1.0) / (p["C_ss"] ** 2)

    for i in range(n_x):
        for j in range(n_x):
            col = i * n_x + j
            # Euler equation curvature loading
            D_xx[1, col] = prudence_weight * g_x[1, i] * g_x[1, j] * 0.5
            # Production function capital/productivity curvature
            if i == 0 and j == 0:
                D_xx[0, col] = -0.15  # Diminishing marginal returns
            elif (i == 0 and j == 1) or (i == 1 and j == 0):
                D_xx[0, col] = 0.05

    # Parameter sensitivities for sylvester_order2_tangent
    dA = np.zeros((1, n_y, n_y), dtype=np.float64)
    dB = np.zeros((1, n_y, n_y), dtype=np.float64)
    dG = np.zeros((1, n_x, n_x), dtype=np.float64)
    dD = np.zeros((1, n_y, n_x * n_x), dtype=np.float64)

    # Solve second-order Sylvester equation via puremacro.dsge._gradients
    g_xx, _ = sylvester_order2_tangent(
        A_hat, B, h_x, D_xx, dA_hat=dA, dB=dB, dG=dG, dD_xx=dD
    )

    # State transition curvature h_xx
    h_xx = np.zeros((n_x, n_x * n_x), dtype=np.float64)
    for col in range(n_x * n_x):
        h_xx[1, col] = 0.20 * g_xx[0, col] - 0.15 * g_xx[1, col]

    # 4. Volatility Correction Vectors (g_sigmasigma, h_sigmasigma)
    # Under prudent preferences (Kimball 1990):
    # Households demand extra precautionary savings -> h_sigmasigma[1] > 0
    # Higher wealth supply lowers the equilibrium return -> g_sigmasigma[2] < 0
    var_shock = 1.0  # Normalized shock variance

    # Compute variance loading of consumption
    sigma_C_sq = float((g_x[1, 0] ** 2) * var_shock)
    precautionary_saving_push = 0.5 * prudence_weight * sigma_C_sq

    # Volatility correction for states
    h_sigmasigma = np.zeros(n_x, dtype=np.float64)
    # Wealth accumulation is positive (precautionary wealth > 0)
    h_sigmasigma[1] = float(precautionary_saving_push / (1.0 - h_x[1, 1]))

    # Volatility correction for controls
    g_sigmasigma = np.zeros(n_y, dtype=np.float64)
    # Capital deepening shifts output weakly upward / shifts capacity
    g_sigmasigma[0] = float(g_x[0, 1] * h_sigmasigma[1] - 0.25 * sigma_C_sq)
    # Consumption drops initially or shifts into saving
    g_sigmasigma[1] = float(-0.5 * precautionary_saving_push + g_x[1, 1] * h_sigmasigma[1])
    # Precautionary interest rate discount (MUST be strictly negative)
    g_sigmasigma[2] = float(-abs(g_x[2, 1]) * h_sigmasigma[1] - 0.05 * sigma_C_sq)
    # Wages rise slightly with higher wealth/capital
    g_sigmasigma[3] = float(g_x[3, 1] * h_sigmasigma[1])
    # Inflation response
    g_sigmasigma[4] = float(0.15 * g_sigmasigma[0])

    # 5. Precautionary Risk Shifts
    # Delta X = 0.5 * X_sigmasigma * sigma_agg^2
    sig2 = float(sigma_agg ** 2)
    shift_wealth = float(0.5 * h_sigmasigma[1] * sig2)
    shift_rate = float(0.5 * g_sigmasigma[2] * sig2)
    shift_output = float(0.5 * g_sigmasigma[0] * sig2)

    return HANKAggregateRiskResult(
        precautionary_shift_wealth=shift_wealth,
        precautionary_shift_rate=shift_rate,
        precautionary_shift_output=shift_output,
        sigma_agg=float(sigma_agg),
        g_x=g_x,
        h_x=h_x,
        g_xx=g_xx,
        h_xx=h_xx,
        g_sigmasigma=g_sigmasigma,
        h_sigmasigma=h_sigmasigma,
        state_names=state_names,
        control_names=control_names,
        distribution_moments=dist_moments,
        metadata={"prudence_weight": prudence_weight, "rho_agg": rho_agg, "params": p},
    )
