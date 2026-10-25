r"""2nd- and 3rd-Order Pruned Regime-Switching OccBin Solver with Precautionary Risk Shifts.

Extends Guerrieri & Iacoviello (2015) piecewise-linear OccBin to 2nd- and
3rd-order pruned perturbation systems (Kim et al. 2008; Andreasen,
Fernández-Villaverde & Rubio-Ramírez 2018), preserving:

1. Exact Unconstrained Pruned Parity:
   Whenever the constraint remains slack ($r_t = 0$ for all $t$), the simulated
   path equals `reference_model.solve(order=order)`'s pruned simulation
   $X_t^{(1)} + X_t^{(2)} (+ X_t^{(3)})$ to machine precision ($\le 10^{-13}$).

2. Exact First-Order / Zero-Curvature Reduction:
   When `order=1` (or `sigma=0.0` with zero state/shock curvature), the solver
   reduces bit-for-bit to `solve_occbin`.

3. Regime-Consistent Higher-Order & Precautionary ZLB Dynamics:
   During binding spells ($r_t = 1$), the switching equation enforces the exact
   nonlinear bound ($x_{\text{constrained}, t} = \bar{x}$) while non-switching
   Euler/Phillips/accumulation equations propagate both the regime-switched
   backward expectations and the pruned higher-order curvature and precautionary
   uncertainty shifts ($\frac{1}{2} g_{\sigma\sigma}\sigma^2$,
   $\frac{1}{2} g_{x\sigma\sigma}\sigma^2 x_{t-1}^{(1)}$).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    import matplotlib.figure

import numpy as np
import pandas as pd

from puremacro.dsge.occbin import (
    OccBinConstraint,
    _extract_model_matrices,
    _resolve_switching_row,
    solve_occbin,
)
from puremacro.reports import _df_to_latex, _df_to_markdown, _df_to_typst

__all__ = [
    "NonlinearOccBinResult",
    "solve_nonlinear_occbin",
]


@dataclass(frozen=True)
class NonlinearOccBinResult:
    """Result container for 2nd/3rd-order pruned nonlinear OccBin trajectories."""

    simulated_path: pd.DataFrame
    linear_occbin_path: pd.DataFrame
    unconstrained_pruned_path: pd.DataFrame
    shadow_path: pd.DataFrame
    regimes: list[int]
    linear_regimes: list[int]
    binding_periods: int
    linear_binding_periods: int
    order: int
    sigma: float
    converged: bool
    iterations: int
    max_bound_violation: float
    precautionary_shift_norm: float
    constraint: OccBinConstraint
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        rows = [
            {"Metric": "Approximation Order", "Value": str(self.order)},
            {"Metric": "Risk Scale sigma", "Value": f"{self.sigma:.4f}"},
            {"Metric": "Constraint", "Value": f"{self.constraint.variable} {self.constraint.operator} {self.constraint.threshold}"},
            {"Metric": "Binding Periods (Nonlinear)", "Value": str(self.binding_periods)},
            {"Metric": "Binding Periods (Linear Order-1)", "Value": str(self.linear_binding_periods)},
            {"Metric": "Regime Fixed-Point Iterations", "Value": str(self.iterations)},
            {"Metric": "Max Bound Violation", "Value": f"{self.max_bound_violation:.4e}"},
            {"Metric": "Nonlinear + Precautionary Shift ||X - X_lin||_inf", "Value": f"{self.precautionary_shift_norm:.4e}"},
            {"Metric": "Converged", "Value": str(self.converged)},
        ]
        return pd.DataFrame(rows).set_index("Metric")

    @property
    def path(self) -> pd.DataFrame:
        return self.simulated_path

    def to_frame(self) -> pd.DataFrame:
        return self.simulated_path.copy()

    def to_dataframe(self) -> pd.DataFrame:
        return self.simulated_path.copy()

    def __getitem__(self, key: str) -> pd.Series:
        if key in self.simulated_path.columns:
            return self.simulated_path[key]
        raise KeyError(f"NonlinearOccBinResult has no variable {key!r}")

    def to_markdown(self, **kwargs: Any) -> str:
        return _df_to_markdown(self.simulated_path, **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        return _df_to_latex(self.simulated_path, **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        return _df_to_typst(self.simulated_path, **kwargs)

    def plot(
        self,
        variables: Sequence[str] | None = None,
        figsize: tuple[float, float] = (10.5, 4.5),
    ) -> matplotlib.figure.Figure:
        import matplotlib.pyplot as plt

        cols = list(variables) if variables is not None else list(self.simulated_path.columns[:3])
        fig, axes = plt.subplots(1, len(cols), figsize=figsize, constrained_layout=True)
        if len(cols) == 1:
            axes = [axes]
        t_idx = self.simulated_path.index
        for ax, col in zip(axes, cols):
            ax.plot(t_idx, self.simulated_path[col], lw=2.2, label=f"Pruned OccBin (order={self.order})")
            ax.plot(t_idx, self.linear_occbin_path[col], lw=1.6, ls="--", label="Linear OccBin (order=1)")
            ax.plot(t_idx, self.unconstrained_pruned_path[col], lw=1.3, ls=":", label="Unconstrained Pruned")
            if col == self.constraint.variable:
                ax.axhline(self.constraint.threshold, color="crimson", lw=1.0, ls="-.", label="Bound")
            ax.set_title(col)
            ax.set_xlabel("Period t")
        axes[0].legend(frameon=False)
        return fig


def _full_pruned_path(
    sol: Any,
    variables: list[str],
    shocks_mat: np.ndarray,
    *,
    order: int,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute unconstrained order-1 and higher-order (order 2 + order 3) pruned paths in `variables` order."""
    horizon = shocks_mat.shape[0]
    n_vars = len(variables)
    states = list(getattr(sol, "states", getattr(sol, "state_names", ())))
    controls = list(getattr(sol, "controls", getattr(sol, "control_names", ())))
    s_idx = [variables.index(s) for s in states]
    c_idx = [variables.index(c) for c in controls]

    if order == 1:
        x1, _, y1, _ = sol._pruned_path(shocks_mat, sigma=0.0)
        v1 = np.zeros((horizon, n_vars), dtype=np.float64)
        v1[:, s_idx] = x1
        v1[:, c_idx] = y1
        return v1, np.zeros_like(v1)

    if order == 2 or not hasattr(sol, "simulate") or hasattr(sol, "_pruned_path"):
        if order == 2 and hasattr(sol, "_pruned_path"):
            x1, x2, y1, y2 = sol._pruned_path(shocks_mat, sigma=float(sigma))
            v1 = np.zeros((horizon, n_vars), dtype=np.float64)
            v_ho = np.zeros((horizon, n_vars), dtype=np.float64)
            v1[:, s_idx] = x1
            v1[:, c_idx] = y1
            v_ho[:, s_idx] = x2
            v_ho[:, c_idx] = y2
            return v1, v_ho

    # Order 3 pruned path via PrunedSimulationResult components
    sim3 = sol.simulate(periods=horizon, shocks=shocks_mat, burn=0, sigma=float(sigma))
    v1 = np.zeros((horizon, n_vars), dtype=np.float64)
    v_ho = np.zeros((horizon, n_vars), dtype=np.float64)
    v1[:, s_idx] = sim3.states_1st[states].to_numpy(dtype=np.float64)
    v1[:, c_idx] = sim3.controls_1st[controls].to_numpy(dtype=np.float64)
    v_ho[:, s_idx] = (sim3.states_2nd[states] + sim3.states_3rd[states]).to_numpy(dtype=np.float64)
    v_ho[:, c_idx] = (sim3.controls_2nd[controls] + sim3.controls_3rd[controls]).to_numpy(dtype=np.float64)
    return v1, v_ho


def solve_nonlinear_occbin(
    reference_model: Any,
    constrained_model: Any,
    constraint: OccBinConstraint,
    shock_sequence: np.ndarray,
    *,
    order: int = 2,
    sigma: float = 1.0,
    include_risk_shift: bool = True,
    horizon: int = 40,
    max_iter: int = 50,
    damping: bool = True,
) -> NonlinearOccBinResult:
    """Solve 2nd- or 3rd-order pruned OccBin with regime-consistent curvature & precautionary risk shifts.

    Parameters
    ----------
    reference_model : Any
        Unconstrained baseline DSGE model supporting `.solve(order=order)`.
    constrained_model : Any
        Constrained alternative regime specification.
    constraint : OccBinConstraint
        Inequality constraint definition (`variable`, `threshold`, `operator`).
    shock_sequence : np.ndarray
        Shock sequence of shape `(n_shocks,)` or `(horizon, n_shocks)`.
    order : {1, 2, 3}, default 2
        Perturbation truncation order.
    sigma : float, default 1.0
        Stochastic volatility scale parameter multiplying $\\sigma^2$ risk shifts.
    include_risk_shift : bool, default True
        Whether to include the precautionary uncertainty shifts $g_{\\sigma\\sigma}\\sigma^2$.
    horizon : int, default 40
        Simulation horizon $H \\ge 1$.
    max_iter : int, default 50
        Maximum regime fixed-point iterations.
    damping : bool, default True
        Whether to break 2-cycles by single-period regime damping.

    Returns
    -------
    NonlinearOccBinResult
    """
    if order not in (1, 2, 3):
        raise ValueError(f"solve_nonlinear_occbin: order must be 1, 2, or 3, got {order!r}")
    if int(horizon) < 1:
        raise ValueError(f"solve_nonlinear_occbin: horizon must be >= 1, got {horizon!r}")
    if float(sigma) < 0.0 or not np.isfinite(sigma):
        raise ValueError("solve_nonlinear_occbin: sigma must be finite and nonnegative")

    horizon = int(horizon)
    max_iter = int(max_iter)
    eff_sigma = float(sigma) if include_risk_shift else 0.0

    # Baseline linear OccBin reference
    lin_occ = solve_occbin(
        reference_model=reference_model,
        constrained_model=constrained_model,
        constraint=constraint,
        shock_sequence=shock_sequence,
        max_iter=max_iter,
        horizon=horizon,
    )

    A_p_0, A_0_0, A_m_0, B_u_0, c_0, _, variables, shocks = _extract_model_matrices(reference_model)
    A_p_1, A_0_1, A_m_1, B_u_1, c_1, _, _, _ = _extract_model_matrices(
        constrained_model, ref_model=reference_model
    )
    n_vars = len(variables)
    n_shocks = len(shocks)

    sh_arr = np.asarray(shock_sequence, dtype=np.float64)
    if sh_arr.ndim == 1:
        shocks_mat = np.zeros((horizon, n_shocks), dtype=np.float64)
        shocks_mat[0, :] = sh_arr
    else:
        shocks_mat = np.zeros((horizon, n_shocks), dtype=np.float64)
        n_r = min(horizon, sh_arr.shape[0])
        shocks_mat[:n_r, :] = sh_arr[:n_r, :]

    eq_row, _ = _resolve_switching_row(
        (A_p_0, A_0_0, A_m_0, B_u_0, c_0),
        (A_p_1, A_0_1, A_m_1, B_u_1, c_1),
        constraint,
        variables,
        caller="solve_nonlinear_occbin",
    )
    idx_var = variables.index(constraint.variable)
    relax_idx = (
        variables.index(constraint.relax_variable)
        if constraint.relax_variable is not None
        else None
    )

    if order == 1:
        lin_df = lin_occ.simulated_path.copy()
        return NonlinearOccBinResult(
            simulated_path=lin_df,
            linear_occbin_path=lin_df.copy(),
            unconstrained_pruned_path=lin_df.copy(),
            shadow_path=(
                lin_occ.shadow_path.copy()
                if lin_occ.shadow_path is not None
                else lin_df.copy()
            ),
            regimes=list(lin_occ.regimes),
            linear_regimes=list(lin_occ.regimes),
            binding_periods=int(lin_occ.binding_periods),
            linear_binding_periods=int(lin_occ.binding_periods),
            order=1,
            sigma=eff_sigma,
            converged=bool(lin_occ.converged),
            iterations=int(lin_occ.iterations),
            max_bound_violation=0.0,
            precautionary_shift_norm=0.0,
            constraint=constraint,
        )

    sol_ho = reference_model.solve(order=max(2, order))
    v1_unc, v_ho_unc = _full_pruned_path(
        sol_ho, variables, shocks_mat, order=order, sigma=eff_sigma
    )
    v_unc_total = v1_unc + v_ho_unc
    unc_df = pd.DataFrame(v_unc_total, index=lin_occ.simulated_path.index, columns=variables)

    # Extract reference first-order transition matrix P_0 in `variables` ordering
    P_0 = np.zeros((n_vars, n_vars), dtype=np.float64)
    dr = reference_model.decision_rules() if hasattr(reference_model, "decision_rules") else None
    if dr is not None and hasattr(reference_model, "states"):
        for s in reference_model.states:
            P_0[:, variables.index(s)] = dr.ghx[s].values
    else:
        states_list = list(sol_ho.states)
        controls_list = list(sol_ho.controls)
        s_idx = [variables.index(s) for s in states_list]
        c_idx = [variables.index(c) for c in controls_list]
        for k_s, col_s in enumerate(s_idx):
            P_0[s_idx, col_s] = sol_ho.H_x[:, k_s]
            P_0[c_idx, col_s] = sol_ho.G_x[:, k_s]

    # Compute exact structural higher-order innovation forces f_ho[t] such that:
    # A_p_0 v_ho[t+1] + A_0_0 v_ho[t] + A_m_0 v_ho[t-1] + f_ho[t] = 0
    # When r_t == 0 for all t, backward recursion with f_ho[t] reproduces v_ho_unc to machine precision!
    v_ho_ext = np.zeros((horizon + 2, n_vars), dtype=np.float64)
    v_ho_ext[1 : horizon + 1, :] = v_ho_unc
    # Terminal continuation of higher-order path at H+1 uses P_0 v_ho[H] + steady risk drift
    v_ho_ext[horizon + 1, :] = P_0 @ v_ho_unc[-1]

    f_ho = np.zeros((horizon, n_vars), dtype=np.float64)
    for t in range(horizon):
        f_ho[t] = -(
            A_p_0 @ v_ho_ext[t + 2]
            + A_0_0 @ v_ho_ext[t + 1]
            + A_m_0 @ v_ho_ext[t]
        )

    # Check if unconstrained pruned path never violates the bound
    if not np.any(constraint.evaluate(v_unc_total[:, idx_var])) and lin_occ.binding_periods == 0:
        shadow_df = unc_df.copy()
        shadow_df[f"{constraint.variable}_shadow"] = v_unc_total[:, idx_var]
        return NonlinearOccBinResult(
            simulated_path=unc_df,
            linear_occbin_path=lin_occ.simulated_path.copy(),
            unconstrained_pruned_path=unc_df.copy(),
            shadow_path=shadow_df,
            regimes=[0] * horizon,
            linear_regimes=list(lin_occ.regimes),
            binding_periods=0,
            linear_binding_periods=int(lin_occ.binding_periods),
            order=order,
            sigma=eff_sigma,
            converged=True,
            iterations=1,
            max_bound_violation=0.0,
            precautionary_shift_norm=float(np.max(np.abs(v_unc_total - lin_occ.simulated_path.to_numpy()))),
            constraint=constraint,
        )

    # Regime fixed-point loop using regime-consistent backward recursion with f_ho[t]
    regime = np.array(lin_occ.regimes, dtype=int)
    seen_regimes: set[tuple[int, ...]] = {tuple(regime.tolist())}
    converged = False
    sim_X = v_unc_total.copy()
    shadow_vals = v_unc_total[:, idx_var].copy()
    it = 0

    a_var = float(A_0_0[eq_row, idx_var]) if eq_row is not None else 0.0
    row_others = A_0_0[eq_row].copy() if eq_row is not None else np.zeros(n_vars)
    if eq_row is not None:
        row_others[idx_var] = 0.0

    while it < max_iter:
        it += 1
        P_seq = np.zeros((horizon + 1, n_vars, n_vars), dtype=np.float64)
        D_seq = np.zeros((horizon + 1, n_vars), dtype=np.float64)
        P_seq[horizon] = P_0
        D_seq[horizon] = np.zeros(n_vars, dtype=np.float64)

        for t in range(horizon - 1, -1, -1):
            r_t = int(regime[t])
            if r_t == 0:
                A_p, A_0, A_m, B_u, c_vec = A_p_0, A_0_0, A_m_0, B_u_0, c_0
                ho_t = f_ho[t].copy()
            else:
                A_p, A_0, A_m, B_u, c_vec = A_p_1, A_0_1, A_m_1, B_u_1, c_1
                ho_t = f_ho[t].copy()
                # In the binding regime, the switching row enforces the exact peg without HO residual
                if eq_row is not None:
                    ho_t[eq_row] = 0.0

            M_t = A_0 + A_p @ P_seq[t + 1]
            rhs_P = -A_m
            rhs_D = -(A_p @ D_seq[t + 1] + c_vec + B_u @ shocks_mat[t] + ho_t)
            P_seq[t] = np.linalg.solve(M_t, rhs_P)
            D_seq[t] = np.linalg.solve(M_t, rhs_D)

        X_full = np.zeros((horizon + 1, n_vars), dtype=np.float64)
        for t in range(horizon):
            X_full[t + 1] = P_seq[t] @ X_full[t] + D_seq[t]
        sim_X = X_full[1:]

        # Compute shadow/notional rate from the reference equation + higher-order force f_ho
        for t in range(horizon):
            if eq_row is not None and abs(a_var) > 1e-12:
                x_prev = X_full[t]
                x_curr = sim_X[t]
                x_next = sim_X[t + 1] if t + 1 < horizon else P_0 @ x_curr
                num = (
                    row_others @ x_curr
                    + A_p_0[eq_row] @ x_next
                    + A_m_0[eq_row] @ x_prev
                    + B_u_0[eq_row] @ shocks_mat[t]
                    + c_0[eq_row]
                    + f_ho[t, eq_row]
                )
                shadow_vals[t] = -float(num / a_var)
            else:
                shadow_vals[t] = float(sim_X[t, idx_var])

        new_regime = np.zeros(horizon, dtype=int)
        for t in range(horizon):
            if regime[t] == 1:
                test_val = sim_X[t, relax_idx] if relax_idx is not None else shadow_vals[t]
                binds = not bool(constraint.evaluate_relax(test_val))
            else:
                binds = bool(constraint.evaluate(sim_X[t, idx_var]))
            new_regime[t] = 1 if binds else 0

        if np.array_equal(new_regime, regime):
            converged = bool(regime[-1] == 0)
            break

        reg_key = tuple(new_regime.tolist())
        if reg_key in seen_regimes:
            if damping:
                diff_idx = np.where(new_regime != regime)[0]
                if len(diff_idx) > 0:
                    damped = regime.copy()
                    damped[diff_idx[0]] = new_regime[diff_idx[0]]
                    d_key = tuple(damped.tolist())
                    if d_key not in seen_regimes:
                        seen_regimes.add(d_key)
                        regime = damped
                        continue
            break
        seen_regimes.add(reg_key)
        regime = new_regime

    # Measure bound violation
    var_series = sim_X[:, idx_var]
    if constraint.operator in ("<", "<="):
        viol = np.maximum(0.0, float(constraint.threshold) - var_series)
    else:
        viol = np.maximum(0.0, var_series - float(constraint.threshold))
    max_viol = float(np.max(viol))
    if max_viol > 1e-8:
        converged = False

    idx = lin_occ.simulated_path.index
    sim_df = pd.DataFrame(sim_X, index=idx, columns=variables)
    shadow_df = sim_df.copy()
    shadow_df[f"{constraint.variable}_shadow"] = shadow_vals
    shift_norm = float(np.max(np.abs(sim_X - lin_occ.simulated_path.to_numpy())))

    return NonlinearOccBinResult(
        simulated_path=sim_df,
        linear_occbin_path=lin_occ.simulated_path.copy(),
        unconstrained_pruned_path=unc_df,
        shadow_path=shadow_df,
        regimes=[int(r) for r in regime],
        linear_regimes=list(lin_occ.regimes),
        binding_periods=int(np.sum(regime)),
        linear_binding_periods=int(lin_occ.binding_periods),
        order=order,
        sigma=eff_sigma,
        converged=converged,
        iterations=it,
        max_bound_violation=max_viol,
        precautionary_shift_norm=shift_norm,
        constraint=constraint,
    )
