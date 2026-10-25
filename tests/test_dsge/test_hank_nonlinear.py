"""Comprehensive unit test suite for puremacro.dsge.hank_nonlinear (Track II.3).

Verifies:
1. Small shock linear SSJ parity (< 1e-4 difference for 1e-4 shock).
2. Large MIT shock convergence (path residuals < 1e-8).
3. Asymmetric MPC amplification for large negative shocks (ratio > 1.05) and binding mass Phi_t.
4. Good Broyden rank-1 inverse updates, Armijo line search, and Jacobian reuse.
5. 2nd-order aggregate risk HANK solver via Reiter/Winberry reduction:
   precautionary wealth accumulation Delta bar{A} > 0, precautionary rate discount Delta bar{r} < 0.
6. Exact quadratic scaling in sigma_agg^2 (< 1e-6 tolerance).
7. sylvester_order2_tangent integration for second-order state curvature g_xx.
8. Custom BlockDAG solving support.
9. Rich result containers (summary, DataFrame, Markdown, LaTeX, Typst, plot).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend

import numpy as np
import pandas as pd
import pytest

from puremacro.dsge import (
    BlockDAG,
    HANKAggregateRiskResult,
    NonlinearHANKTransitionResult,
    SimpleBlock,
    build_canonical_hank_dag,
    solve_hank_aggregate_risk_order2,
    solve_hank_nonlinear_transition,
)


def _build_test_linear_nk_dag():
    """Build a standard 3-equation New Keynesian DAG for testing custom DAG solves."""
    taylor = SimpleBlock(
        "taylor",
        inputs=("pi", "eps_m"),
        outputs=("i",),
        fn=lambda v, p: {"i": p["r_ss"] + p["phi_pi"] * v["pi"] + v["eps_m"]},
    )
    fisher = SimpleBlock(
        "fisher",
        inputs=("i", "pi"),
        outputs=("r",),
        fn=lambda v, p: {"r": v["i"] - v["pi"](+1)},
    )
    nkpc = SimpleBlock(
        "nkpc",
        inputs=("pi", "C"),
        outputs=("nkpc_res",),
        fn=lambda v, p: {"nkpc_res": v["pi"] - p["beta"] * v["pi"](+1) - p["kappa"] * (v["C"] - p["C_ss"])},
    )
    euler = SimpleBlock(
        "euler",
        inputs=("C", "r"),
        outputs=("euler_res",),
        fn=lambda v, p: {"euler_res": (v["C"] - v["C"](+1)) + (1.0 / p["sigma"]) * (v["r"] - p["r_ss"])},
    )
    dag = BlockDAG(
        blocks=[nkpc, euler, fisher, taylor],
        unknowns=("C", "pi"),
        exogenous=("eps_m",),
        targets=("euler_res", "nkpc_res"),
    )
    ss = {"C": 1.0, "pi": 0.0, "i": 0.01, "r": 0.01, "eps_m": 0.0, "euler_res": 0.0, "nkpc_res": 0.0}
    params = {"beta": 0.99, "sigma": 1.0, "kappa": 0.15, "phi_pi": 1.5, "r_ss": 0.01, "C_ss": 1.0}
    return dag, ss, params


# ===========================================================================
# 1. Nonlinear Transition Tests
# ===========================================================================

def test_nonlinear_transition_small_shock_parity():
    """For an infinitesimal shock (1e-4), nonlinear transition matches linear SSJ to < 1e-4."""
    T = 40
    shock_mag = 1e-4
    shock_seq = shock_mag * (0.7 ** np.arange(T))

    result = solve_hank_nonlinear_transition(
        shocks={"eps_m": shock_seq},
        horizon=T,
        tol=1e-8,
        max_iter=50,
    )
    assert result.converged
    assert result.residual_norm < 1e-8

    # Parity check: difference between nonlinear and linear SSJ paths
    y_nl = result.paths["Y"].to_numpy()
    y_lin = result.linear_paths["Y"].to_numpy()
    diff_inf = float(np.max(np.abs(y_nl - y_lin)))

    # For 1e-4 shock, nonlinear second-order curvature is O(1e-8), well below 1e-4
    assert diff_inf < 1e-4, f"Parity difference {diff_inf:.3e} exceeds 1e-4"
    assert result.shock_size == pytest.approx(shock_mag, rel=1e-3)


def test_nonlinear_transition_large_shock_convergence():
    """For a large MIT shock (500 bps = 0.05), solver reduces residuals to < 1e-8."""
    T = 60
    shock_mag = 0.05
    shock_seq = shock_mag * (0.65 ** np.arange(T))

    result = solve_hank_nonlinear_transition(
        shocks={"eps_m": shock_seq},
        horizon=T,
        tol=1e-8,
        max_iter=100,
        backtracking=True,
    )
    assert result.converged
    assert result.residual_norm < 1e-8, f"Residual {result.residual_norm:.3e} >= 1e-8"
    assert result.iterations > 0
    assert result.iterations < 100


def test_asymmetric_borrowing_constraint_amplification():
    """Large negative shock induces higher binding fraction Phi_t and amplified output drop (> 1.05)."""
    T = 50
    shock_amp = 0.04
    shock_neg = -shock_amp * (0.7 ** np.arange(T))
    shock_pos = +shock_amp * (0.7 ** np.arange(T))

    # Solve contractionary (G < 0) vs expansionary (G > 0) shocks
    res_neg = solve_hank_nonlinear_transition(
        shocks={"G": shock_neg},
        horizon=T,
        tol=1e-8,
        max_iter=60,
    )
    res_pos = solve_hank_nonlinear_transition(
        shocks={"G": shock_pos},
        horizon=T,
        tol=1e-8,
        max_iter=60,
    )

    assert res_neg.converged
    assert res_pos.converged

    # Maximum deviation of output from steady state
    y_ss = float(res_neg.metadata["steady_state"].get("Y", 1.0))
    drop_neg = float(np.max(np.abs(res_neg.paths["Y"].to_numpy() - y_ss)))
    rise_pos = float(np.max(np.abs(res_pos.paths["Y"].to_numpy() - y_ss)))

    amplification = drop_neg / rise_pos
    assert amplification > 1.05, f"Amplification ratio {amplification:.4f} must exceed 1.05"
    assert res_neg.asymmetry_ratio > 1.05

    # Check borrowing binding fraction Phi_t dynamics
    phi_neg = res_neg.binding_fractions
    phi_pos = res_pos.binding_fractions
    # Under contraction, binding mass surges above steady-state level
    phi_ss = float(res_neg.metadata["steady_state"].get("Phi", 0.22))
    assert np.max(phi_neg) > phi_ss
    assert np.max(phi_neg) > np.max(phi_pos)

    # Also verify monetary policy tightening amplification (rate hike vs cut)
    hike_res = solve_hank_nonlinear_transition(shocks={"eps_m": shock_amp * (0.7 ** np.arange(T))}, horizon=T)
    cut_res = solve_hank_nonlinear_transition(shocks={"eps_m": -shock_amp * (0.7 ** np.arange(T))}, horizon=T)
    drop_hike = float(np.max(np.abs(hike_res.paths["Y"].to_numpy() - y_ss)))
    rise_cut = float(np.max(np.abs(cut_res.paths["Y"].to_numpy() - y_ss)))
    assert drop_hike / rise_cut > 1.05


def test_broyden_jacobian_reuse_and_armijo():
    """Test Good Broyden Sherman-Morrison rank-1 updates and Armijo line search."""
    T = 40
    shock_seq = 0.03 * (0.6 ** np.arange(T))

    result = solve_hank_nonlinear_transition(
        shocks={"eps_m": shock_seq},
        horizon=T,
        tol=1e-8,
        jacobian_update="good_broyden",
        backtracking=True,
    )
    assert result.converged
    assert result.broyden_updates >= 1
    assert result.jacobian_evaluations >= 1
    assert len(result.binding_fractions) == T


def test_custom_block_dag_transition():
    """Solve nonlinear transition on a user-supplied BlockDAG."""
    dag, ss, params = _build_test_linear_nk_dag()
    T = 30
    eps_path = 0.01 * (0.6 ** np.arange(T))

    res = solve_hank_nonlinear_transition(
        model=dag,
        shocks={"eps_m": eps_path},
        steady_state=ss,
        params=params,
        horizon=T,
        tol=1e-8,
    )
    assert res.converged
    assert res.residual_norm < 1e-8
    assert "C" in res.paths.columns
    assert "pi" in res.paths.columns
    assert len(res.paths) == T


# ===========================================================================
# 2. 2nd-Order Aggregate Risk Tests
# ===========================================================================

def test_hank_aggregate_risk_precautionary_signs():
    """2nd-order perturbation produces precautionary wealth accumulation (Delta A > 0)
    and precautionary rate discount (Delta r < 0)."""
    sigma_agg = 0.01
    res = solve_hank_aggregate_risk_order2(
        sigma_agg=sigma_agg,
        rho_agg=0.85,
        n_distribution_moments=4,
    )

    # 1. Precautionary wealth accumulation: prudent households save more against aggregate risk
    assert res.precautionary_shift_wealth > 0.0, (
        f"Delta bar{{A}} must be positive, got {res.precautionary_shift_wealth:.3e}"
    )

    # 2. Precautionary interest rate discount: higher wealth supply lowers equilibrium rate
    assert res.precautionary_shift_rate < 0.0, (
        f"Delta bar{{r}} must be negative, got {res.precautionary_shift_rate:.3e}"
    )

    # 3. Precautionary output shift is finite and well-defined
    assert np.isfinite(res.precautionary_shift_output)
    assert res.sigma_agg == sigma_agg


def test_aggregate_risk_quadratic_scaling():
    """Precautionary shifts scale quadratically in sigma_agg^2 to < 1e-6 error."""
    sigma_1 = 0.01
    sigma_2 = 2.0 * sigma_1

    res_1 = solve_hank_aggregate_risk_order2(sigma_agg=sigma_1)
    res_2 = solve_hank_aggregate_risk_order2(sigma_agg=sigma_2)

    # In 2nd-order perturbation, Delta X = 0.5 * X_sigmasigma * sigma^2
    # Doubling sigma quadruples Delta X: ratio must be exactly 4.0
    ratio_wealth = res_2.precautionary_shift_wealth / res_1.precautionary_shift_wealth
    ratio_rate = res_2.precautionary_shift_rate / res_1.precautionary_shift_rate
    ratio_output = res_2.precautionary_shift_output / res_1.precautionary_shift_output

    assert abs(ratio_wealth - 4.0) < 1e-6, f"Wealth shift ratio {ratio_wealth:.7f} != 4.0"
    assert abs(ratio_rate - 4.0) < 1e-6, f"Rate discount ratio {ratio_rate:.7f} != 4.0"
    assert abs(ratio_output - 4.0) < 1e-6, f"Output shift ratio {ratio_output:.7f} != 4.0"


def test_sylvester_order2_tangent_integration():
    """Verify that g_xx state curvature tensor is genuinely computed via sylvester_order2_tangent."""
    res = solve_hank_aggregate_risk_order2(sigma_agg=0.01, n_distribution_moments=4)
    n_x = len(res.state_names)
    n_y = len(res.control_names)

    assert res.g_xx.shape == (n_y, n_x * n_x)
    assert res.h_xx.shape == (n_x, n_x * n_x)
    # Tensor must not be identically zero
    assert np.linalg.norm(res.g_xx) > 1e-3
    assert np.all(np.isfinite(res.g_xx))
    assert np.all(np.isfinite(res.h_sigmasigma))
    assert np.all(np.isfinite(res.g_sigmasigma))


# ===========================================================================
# 3. Result Container and Export Methods Tests
# ===========================================================================

def test_result_containers_and_exports():
    """Verify summary tables, DataFrame conversion, text export formats, and plotting."""
    T = 30
    shock_seq = 0.02 * (0.7 ** np.arange(T))
    nl_res = solve_hank_nonlinear_transition(shocks=shock_seq, horizon=T)

    # 1. Summary DataFrame
    df_sum = nl_res.summary()
    assert isinstance(df_sum, pd.DataFrame)
    assert "Value" in df_sum.columns
    assert "Convergence Status" in df_sum.index

    # 2. to_dataframe
    df_paths = nl_res.to_dataframe()
    assert isinstance(df_paths, pd.DataFrame)
    assert len(df_paths) == T
    assert "Y" in df_paths.columns

    # 3. Markup exports
    md_str = nl_res.to_markdown()
    assert isinstance(md_str, str)
    assert "Convergence Status" in md_str

    latex_str = nl_res.to_latex()
    assert isinstance(latex_str, str)
    assert "\\begin{tabular}" in latex_str

    typst_str = nl_res.to_typst()
    assert isinstance(typst_str, str)
    assert "#table" in typst_str

    # 4. Plotting (no plt.show())
    fig = nl_res.plot(variables=["Y", "C"])
    assert fig is not None
    assert len(fig.axes) >= 2

    # 5. Aggregate risk result exports
    risk_res = solve_hank_aggregate_risk_order2(sigma_agg=0.01)
    df_risk_sum = risk_res.summary()
    assert isinstance(df_risk_sum, pd.DataFrame)
    assert "Precautionary Wealth Shift (Delta A)" in df_risk_sum.index

    md_risk = risk_res.to_markdown()
    assert "Precautionary Wealth Shift" in md_risk

    fig_risk = risk_res.plot()
    assert fig_risk is not None
    assert len(fig_risk.axes) == 2


def test_hank_aggregate_risk_higher_moments_dimension_stability():
    """Verify solve_hank_aggregate_risk_order2 dimension stability across moments [1, 2, 4, 5, 6, 8, 10]."""
    for n_m in [1, 2, 4, 5, 6, 8, 10]:
        res = solve_hank_aggregate_risk_order2(
            sigma_agg=0.01,
            n_distribution_moments=n_m,
        )
        assert isinstance(res, HANKAggregateRiskResult)
        assert len(res.state_names) == n_m + 1
        assert res.g_xx.shape == (5, (n_m + 1) ** 2)
        assert np.all(np.isfinite(res.g_xx))
        assert np.all(np.isfinite(res.h_xx))
        assert res.precautionary_shift_A > 0.0
        assert res.precautionary_shift_r < 0.0
        assert res.moments_shift.shape == (n_m - 1,)

        # All presentation methods execute cleanly
        df_sum = res.summary()
        assert isinstance(df_sum, pd.DataFrame)
        assert "Precautionary Wealth Shift (Delta A)" in df_sum.index

        df_out = res.to_dataframe()
        assert isinstance(df_out, pd.DataFrame)
        assert len(df_out) == len(df_sum)

        latex_str = res.to_latex()
        assert isinstance(latex_str, str)
        assert "\\begin{tabular}" in latex_str

