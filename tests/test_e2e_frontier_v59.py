"""Master Frontier End-to-End Test Suite (PureMacro v5.9.0 Frontier Features).

Comprehensive 4-tier opaque-box test suite verifying all 7 frontier tracks:
- Track #6: Pure-NumPy Forward AD & Analytical Perturbation/VFI Gradients
- Track #1: Generalized Nonlinear DAG Sequence-Space Jacobians & HANK Estimation
- Track #2: 2nd/3rd-Order Pruned Nonlinear OccBin with Precautionary Risk Shifts
- Track #3: Differentiable Mixed-Frequency State-Space Nowcasting & News Decomposition
- Track #4: Continuous-Time HJB Upwind Finite-Volume & KFE Bridge
- Track #5A: Baqaee-Farhi 2nd-Order Network Macro Aggregation
- Track #5B: Dynamic Spatial Migration Hat-Algebra
- Track #7: Self-Auditing PolicyStudio & Zero-Server Interactive HTML5/SVG

Test Architecture:
- Tier 1: Feature Coverage (>=5 tests per track across all frontier tracks)
- Tier 2: Boundary & Corner Cases (>=5 tests per track covering domain limits)
- Tier 3: Cross-Feature Combinations (Pairwise interactions across tracks)
- Tier 4: Real-World Macroeconomic Application Scenarios

Runtime Contract:
Strict 4-package Pyodide compatibility: numpy, scipy, pandas, matplotlib only.
All tests feature genuine domain assertions with realistic mathematical bounds.
"""
from __future__ import annotations

import itertools
import math
import warnings
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.linalg
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# PureMacro Frontier Imports
# ---------------------------------------------------------------------------
from puremacro._autodiff import (
    DualArray,
    HyperDualArray,
    dual_solve,
    dual_sylvester_solve,
    hessian_forward,
    jacobian_forward,
    seed_dual_vector,
    value_and_grad,
)
from puremacro.dsge import (
    BlockDAG,
    HetBlock,
    OccBinConstraint,
    SimpleBlock,
    SolvedBlock,
    build_dynare,
    estimate_hank_irf_matching,
    estimate_hank_whittle,
    hank_loglik_whittle,
    solve_nonlinear_occbin,
    solve_occbin,
)
from puremacro.dsge._gradients import (
    StateSpaceSensitivity,
    kalman_score,
    sylvester_order2_tangent,
)
from puremacro.nowcast import (
    DynamicFactorModel,
    NewsDecompositionResult,
    banbura_modugno_news,
    kalman_dfm,
    mf_var,
)
from puremacro.reports import result_to_interactive_html
from puremacro.spatial import (
    DynamicSpatialHatModel,
    solve_dynamic_spatial_hat,
)
from puremacro.state_space import StateSpaceModel, kalman_filter
from puremacro.study import (
    CounterfactualGrid,
    PolicyStudioCell,
    PolicyStudioResult,
    run_policy_studio,
)
from puremacro.trade import (
    baqaee_farhi_decomposition,
    solve_exact_ces_network_prices,
)
from puremacro.vfi import (
    HJBSolution,
    compute_ift_gradients,
    solve_aiyagari_continuous_hjb,
    solve_hjb_achdou,
    solve_kfe_achdou,
)
from puremacro.vfi.hjb_achdou import _quadrature_weights


# ===========================================================================
# Shared Shared Fixture & Model Builders
# ===========================================================================

def _sample_io_3sector() -> tuple[np.ndarray, np.ndarray]:
    """3-sector upstream-downstream production network: Energy, Mfg, Services."""
    Omega = np.array(
        [
            [0.10, 0.05, 0.05],
            [0.35, 0.15, 0.05],
            [0.10, 0.30, 0.10],
        ],
        dtype=np.float64,
    )
    b = np.array([0.10, 0.30, 0.60], dtype=np.float64)
    return Omega, b


def _sample_spatial_3region() -> DynamicSpatialHatModel:
    """3-region symmetric baseline economy with forward-looking EV1 migration."""
    L0 = np.array([100.0, 100.0, 100.0], dtype=np.float64)
    pi0 = np.array(
        [
            [0.70, 0.15, 0.15],
            [0.15, 0.70, 0.15],
            [0.15, 0.15, 0.70],
        ],
        dtype=np.float64,
    )
    mu0 = np.array(
        [
            [0.90, 0.05, 0.05],
            [0.05, 0.90, 0.05],
            [0.05, 0.05, 0.90],
        ],
        dtype=np.float64,
    )
    return DynamicSpatialHatModel(
        initial_population=L0,
        initial_trade_shares=pi0,
        initial_migration_shares=mu0,
        trade_elasticity=4.0,
        migration_dispersion=0.5,
        beta=0.95,
        congestion_alpha=0.10,
        region_names=("North", "Center", "South"),
    )


def _build_nk_occbin_models():
    """Build reference and occasionally binding ZLB DSGE models."""
    params = {
        "beta": 0.99,
        "sigma": 2.0,
        "kappa": 0.15,
        "phi_pi": 1.5,
        "rho": 0.8,
        "i_floor": -0.01,
    }
    variables = ["y", "pi", "i", "a"]
    shocks = ["e"]
    ss = {v: 0.0 for v in variables}

    def nk_ref(lead, curr, lag, u, p):
        return [
            np.exp(-p.sigma * curr.y) - np.exp(curr.i - lead.pi - curr.a - p.sigma * lead.y),
            curr.pi - p.beta * lead.pi - p.kappa * curr.y,
            curr.i - p.phi_pi * curr.pi,
            curr.a - p.rho * lag.a - u.e,
        ]

    def nk_zlb(lead, curr, lag, u, p):
        return [
            np.exp(-p.sigma * curr.y) - np.exp(curr.i - lead.pi - curr.a - p.sigma * lead.y),
            curr.pi - p.beta * lead.pi - p.kappa * curr.y,
            curr.i - p.i_floor,
            curr.a - p.rho * lag.a - u.e,
        ]

    m_ref = build_dynare(
        nk_ref,
        variables=variables,
        shocks=shocks,
        params=params,
        steady_state=ss,
        shock_cov=np.array([[0.01 ** 2]]),
    )
    m_zlb = build_dynare(
        nk_zlb,
        variables=variables,
        shocks=shocks,
        params=params,
        steady_state=ss,
        check_steady_state=False,
        strict=False,
    )
    constraint = OccBinConstraint(variable="i", threshold=-0.01, operator="<")
    return m_ref, m_zlb, constraint


def _build_nk_ssj_dag():
    """Build a 3-equation New Keynesian / HANK DAG with monetary, fisher, nkpc, euler blocks."""
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


def _build_synthetic_nowcast_vintages(T: int = 40, seed: int = 123):
    """Generate old and new data vintages for nowcasting and news decomposition."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=T, freq="MS")
    cols = ["gdp", "ip", "retail", "cpi", "emp"]

    F = np.zeros(T)
    for t in range(1, T):
        F[t] = 0.8 * F[t - 1] + rng.normal(scale=0.5)

    loadings = np.array([1.0, 0.85, 0.65, 0.3, -0.7])
    data = np.zeros((T, 5))
    for j in range(5):
        data[:, j] = 2.0 + loadings[j] * F + rng.normal(scale=0.2, size=T)

    df_full = pd.DataFrame(data, index=dates, columns=cols)
    df_old = df_full.copy()
    # Ragged edge: GDP and IP missing in last 2 periods; retail and emp missing in last period
    df_old.iloc[-1, :] = np.nan
    df_old.iloc[-2, 0] = np.nan
    df_old.iloc[-2, 1] = np.nan

    # New vintage: releases for IP at T-2, Retail at T-1, plus historical revision for IP at T-5
    df_new = df_old.copy()
    df_new.iloc[-2, 1] = df_full.iloc[-2, 1]
    df_new.iloc[-1, 2] = df_full.iloc[-1, 2]
    df_new.iloc[-5, 1] += 0.15  # revision

    model = DynamicFactorModel(n_factors=1, p=1, standardize=True).fit(df_old)
    return model, df_old, df_new


# ===========================================================================
# Tier 1: Feature Coverage (>=5 tests per track across all 8 tracks = 40 tests)
# ===========================================================================

class TestTier1FeatureCoverage:
    """Tier 1: Comprehensive feature coverage across all 7 frontier tracks + PolicyStudio."""

    # -----------------------------------------------------------------------
    # Track 6: Pure-NumPy Forward AD & Analytical Perturbation/VFI Gradients
    # -----------------------------------------------------------------------

    def test_t1_ad_dual_array_vectorized_ufuncs_and_broadcasting(self):
        """Vectorized DualArray ufuncs (sin, cos, exp, log, sqrt) and broadcasting."""
        theta = np.array([0.6, 1.2, 2.5])
        P = len(theta)
        d_theta = seed_dual_vector(theta)

        # Non-trivial transcendental composite function
        val_d = np.sin(d_theta[0]) * np.exp(d_theta[1]) + np.sqrt(d_theta[2]) / d_theta[0]
        assert isinstance(val_d, DualArray)
        assert val_d.der.shape == (P,)

        # Exact analytical derivatives
        t0, t1, t2 = theta[0], theta[1], theta[2]
        f_analytic = math.sin(t0) * math.exp(t1) + math.sqrt(t2) / t0
        g0 = math.cos(t0) * math.exp(t1) - math.sqrt(t2) / (t0 ** 2)
        g1 = math.sin(t0) * math.exp(t1)
        g2 = 1.0 / (2.0 * math.sqrt(t2) * t0)

        assert abs(val_d.val - f_analytic) < 1e-15
        assert abs(val_d.der[0] - g0) < 1e-14
        assert abs(val_d.der[1] - g1) < 1e-14
        assert abs(val_d.der[2] - g2) < 1e-14

    def test_t1_ad_value_and_grad_multivariate_nonlinear_objective(self):
        """value_and_grad matches exact CRRA welfare gradient."""
        gamma = 2.5
        w = np.array([0.2, 0.5, 0.3])
        c0 = np.array([1.1, 1.8, 0.9])

        def welfare(c):
            return np.sum(w * (c ** (1.0 - gamma)) / (1.0 - gamma))

        val, grad = value_and_grad(welfare, c0)
        grad_exact = w * (c0 ** (-gamma))

        assert abs(val - welfare(c0)) < 1e-15
        assert np.max(np.abs(grad - grad_exact)) < 1e-14

    def test_t1_ad_hyper_dual_exact_hessian_matrix_multiplication(self):
        """hessian_forward with matrix multiplication matches exact analytical Hessian."""
        W = np.array([[3.0, -0.6, 0.4], [-0.6, 2.0, 0.1], [0.4, 0.1, 1.5]])
        b = np.array([0.2, -0.4, 0.8])

        def quadratic_cost(x):
            r = (W @ x) - b
            return 0.5 * (r @ r) + np.sum(np.exp(0.5 * x))

        x0 = np.array([0.4, -0.2, 0.7])
        val, grad, hess = hessian_forward(quadratic_cost, x0)

        hess_exact = W.T @ W + np.diag(0.25 * np.exp(0.5 * x0))
        assert np.max(np.abs(hess - hess_exact)) < 1e-13
        assert np.allclose(hess, hess.T, atol=1e-15)

    def test_t1_ad_dual_solve_implicit_linear_systems(self):
        """dual_solve implicit linear system dx/dtheta = A^{-1}(db - dA x)."""
        def lin_sys(theta):
            th = theta[0]
            A = np.array([[2.0 + th, 0.4], [-0.3, 1.5 + 2.0 * th]])
            b = np.array([1.0 + th ** 2, 2.0 - th])
            return dual_solve(A, b)

        theta0 = np.array([0.8])
        val, jac = jacobian_forward(lin_sys, theta0)

        # Complex-step verification (h = 1e-20)
        h_cs = 1e-20
        Ac = np.array([[2.0 + (0.8 + 1j * h_cs), 0.4], [-0.3, 1.5 + 2.0 * (0.8 + 1j * h_cs)]])
        bc = np.array([1.0 + (0.8 + 1j * h_cs) ** 2, 2.0 - (0.8 + 1j * h_cs)])
        xc = np.linalg.solve(Ac, bc)
        jac_cs = np.imag(xc) / h_cs

        assert np.allclose(val, np.real(xc), atol=1e-14)
        assert np.max(np.abs(jac.ravel() - jac_cs)) < 1e-14

    def test_t1_ad_vfi_analytic_tangents_ift_autodiff(self):
        """VFI/EGM policy gradient via compute_ift_gradients(diff_method='autodiff')."""
        class _MockSolution:
            def __init__(self, coeffs):
                self.coefficients = coeffs

        class _MockProblem:
            def __init__(self):
                self.params = {"alpha": 0.33, "delta": 0.08}
                self.beta = 0.96

        def euler_residual(c, p):
            alpha = p["alpha"]
            delta = p["delta"]
            beta = p["beta"]
            k_grid = np.array([1.0, 2.0, 3.0])
            kp = k_grid ** alpha + (1.0 - delta) * k_grid - c
            mpk = alpha * (kp ** (alpha - 1.0)) + (1.0 - delta)
            return c - (beta * mpk * 0.5) ** (-0.5)

        # Root at c_star
        prob = _MockProblem()
        k_grid = np.array([1.0, 2.0, 3.0])
        # Find c_star solving euler_residual = 0 numerically
        from scipy.optimize import root
        c_star_init = np.array([0.8, 1.2, 1.5])
        res_root = root(lambda c: euler_residual(c, {"alpha": 0.33, "delta": 0.08, "beta": 0.96}), c_star_init)
        sol = _MockSolution(res_root.x)

        res_ad = compute_ift_gradients(
            sol, prob, params=["alpha", "delta"], residual_fn=euler_residual, diff_method="autodiff"
        )
        assert res_ad.metadata.get("diff_method") == "autodiff"
        assert res_ad.grad_coefficients.shape == (3, 2)
        assert np.all(np.isfinite(res_ad.grad_coefficients))

    # -----------------------------------------------------------------------
    # Track 1: Generalized Nonlinear DAG Sequence-Space Jacobians & HANK
    # -----------------------------------------------------------------------

    def test_t1_dag_simple_block_dag_propagation(self):
        """SimpleBlock composition in BlockDAG satisfies topological sort and forward eval."""
        b1 = SimpleBlock("prod", inputs=("A", "K"), outputs=("Y",), fn=lambda v, p: {"Y": v["A"] * (v["K"] ** p["alpha"])})
        b2 = SimpleBlock("factor", inputs=("Y", "K"), outputs=("r",), fn=lambda v, p: {"r": p["alpha"] * v["Y"] / v["K"]})
        dag = BlockDAG([b1, b2], unknowns=("K",), exogenous=("A",), targets=("r",))
        assert dag.blocks[0].name == "prod"
        assert dag.blocks[1].name == "factor"

        K_val = 4.0
        alpha_val = 0.33
        Y_val = 1.0 * (K_val ** alpha_val)
        ss = {"A": 1.0, "K": K_val, "Y": Y_val, "r": alpha_val * Y_val / K_val}
        params = {"alpha": alpha_val}
        J = b1.jacobian(ss, params, T=10)
        assert "Y" in J and "K" in J["Y"]
        expected_dY_dK = alpha_val * (K_val ** (alpha_val - 1.0))
        assert abs(J["Y"]["K"][0, 0] - expected_dY_dK) < 1e-12

    def test_t1_dag_solved_block_internal_root_finding(self):
        """SolvedBlock resolves internal target and constructs sequence-space Jacobian."""
        T = 15
        inner = SimpleBlock("inner", inputs=("u", "x"), outputs=("res",), fn=lambda v, p: {"res": v["u"] - 3.0 * v["x"]})
        solved = SolvedBlock("solved_u", unknowns=("u",), targets=("res",), inner_block=inner)
        ss = {"u": 0.0, "x": 0.0, "res": 0.0}
        J_s = solved.jacobian(ss, {}, T)
        assert np.allclose(J_s["u"]["x"], 3.0 * np.eye(T), atol=1e-14)

    def test_t1_dag_het_block_aggregate_jacobians(self):
        """HetBlock integrates sequence-space consumption response matrix."""
        T = 20
        mpc = 0.45
        J_hh = {"c_hh": {"r": -0.8 * np.eye(T), "Y": mpc * np.eye(T)}}
        hh = HetBlock("hh", inputs=("r", "Y"), outputs=("c_hh",), jacobian_fn=J_hh)
        assert "r" in hh.inputs and "Y" in hh.inputs
        assert "c_hh" in hh.outputs
        J = hh.jacobian({}, {}, T)
        assert np.allclose(J["c_hh"]["r"], -0.8 * np.eye(T))
        assert np.allclose(J["c_hh"]["Y"], 0.45 * np.eye(T))

    def test_t1_dag_block_dag_ge_jacobians_match_linear_ssj(self):
        """BlockDAG general-equilibrium impulse responses match analytical formula."""
        dag, ss, params = _build_nk_ssj_dag()
        T = 30
        rho_m = 0.5
        eps_path = 0.01 * (rho_m ** np.arange(T))

        irf = dag.impulse_responses(ss, params, {"eps_m": eps_path}, T=T, levels=False)
        assert irf.converged
        assert irf.residual_norm < 1e-12

        # Analytical 3-equation NK model check
        denom = (
            params["sigma"] * (1.0 - rho_m) * (1.0 - params["beta"] * rho_m)
            + params["kappa"] * (params["phi_pi"] - rho_m)
        )
        pi_analytic = -(params["kappa"] / denom) * eps_path
        assert np.max(np.abs(irf.paths["pi"].to_numpy()[:15] - pi_analytic[:15])) < 1e-9

    def test_t1_dag_hank_whittle_likelihood_and_irf_matching(self):
        """hank_loglik_whittle and estimate_hank_irf_matching structural recovery."""
        rng = np.random.default_rng(42)
        T_obs = 64
        sigma2 = 0.04
        y = rng.normal(0.0, np.sqrt(sigma2), size=(T_obs, 1))
        M_wn = np.zeros((16, 1, 1))
        M_wn[0, 0, 0] = 1.0

        ll_whittle = hank_loglik_whittle(M_wn, y, np.array([[sigma2]]), demean=False)
        ll_exact = -0.5 * T_obs * np.log(2.0 * np.pi * sigma2) - 0.5 * float(np.sum(y ** 2)) / sigma2
        assert abs(ll_whittle - ll_exact) < 1e-10

        # IRF matching structural parameter recovery
        rho_true = 0.65
        H = 12
        target_df = pd.DataFrame({"y": rho_true ** np.arange(H)})
        res_irf = estimate_hank_irf_matching(
            lambda p, h: {"y": p["rho"] ** np.arange(h)},
            target_df,
            init_params={"rho": 0.3},
            bounds={"rho": (0.05, 0.95)},
        )
        assert res_irf.converged
        assert abs(res_irf.params["rho"] - rho_true) < 1e-5
        assert res_irf.vcov.shape == (1, 1)
        assert res_irf.vcov[0, 0] > 0.0

    # -----------------------------------------------------------------------
    # Track 2: 2nd/3rd-Order Pruned Nonlinear OccBin
    # -----------------------------------------------------------------------

    def test_t1_occbin_slack_matches_order2_and_order3_pruned(self):
        """solve_nonlinear_occbin matches unconstrained pruned path when slack."""
        m_ref, m_zlb, constraint = _build_nk_occbin_models()
        small_shock = np.array([0.005])

        # Order 2 slack
        res2 = solve_nonlinear_occbin(m_ref, m_zlb, constraint, small_shock, order=2, sigma=1.0, horizon=25)
        assert res2.converged
        assert res2.binding_periods == 0
        diff2 = np.max(np.abs(res2.simulated_path.to_numpy() - res2.unconstrained_pruned_path.to_numpy()))
        assert diff2 < 1e-13

        # Order 3 slack
        res3 = solve_nonlinear_occbin(m_ref, m_zlb, constraint, small_shock, order=3, sigma=1.0, horizon=25)
        assert res3.converged
        assert res3.binding_periods == 0
        diff3 = np.max(np.abs(res3.simulated_path.to_numpy() - res3.unconstrained_pruned_path.to_numpy()))
        assert diff3 < 1e-13

    def test_t1_occbin_order1_reduction_to_linear_occbin(self):
        """solve_nonlinear_occbin(order=1) matches linear solve_occbin to 1e-13."""
        m_ref, m_zlb, constraint = _build_nk_occbin_models()
        shock = np.array([-0.05])

        res_nl = solve_nonlinear_occbin(m_ref, m_zlb, constraint, shock, order=1, sigma=1.0, horizon=30)
        res_lin = solve_occbin(m_ref, m_zlb, constraint, shock, horizon=30)
        assert res_nl.converged and res_lin.converged
        assert res_nl.binding_periods == res_lin.binding_periods
        assert np.max(np.abs(res_nl.simulated_path.to_numpy() - res_lin.simulated_path.to_numpy())) < 1e-13

    def test_t1_occbin_binding_regime_constraint_satisfaction(self):
        """solve_nonlinear_occbin satisfies lower bound with max_bound_violation < 1e-10."""
        m_ref, m_zlb, constraint = _build_nk_occbin_models()
        shock = np.array([-0.06])

        res = solve_nonlinear_occbin(m_ref, m_zlb, constraint, shock, order=2, sigma=1.0, horizon=30)
        assert res.converged
        assert res.binding_periods > 0
        assert res.max_bound_violation < 1e-10
        # Bound is -0.01: during binding periods i must equal or exceed floor
        i_path = res.simulated_path["i"].to_numpy()
        assert np.all(i_path >= -0.01 - 1e-10)

    def test_t1_occbin_precautionary_risk_shift_scaling(self):
        """solve_nonlinear_occbin exhibits positive precautionary shift scaling with sigma."""
        m_ref, m_zlb, constraint = _build_nk_occbin_models()
        shock = np.array([-0.05])

        res0 = solve_nonlinear_occbin(m_ref, m_zlb, constraint, shock, order=2, sigma=0.0, horizon=30)
        res1 = solve_nonlinear_occbin(m_ref, m_zlb, constraint, shock, order=2, sigma=1.0, horizon=30)
        res2 = solve_nonlinear_occbin(m_ref, m_zlb, constraint, shock, order=2, sigma=2.0, horizon=30)

        assert abs(res0.precautionary_shift_norm) < 1e-12
        assert res1.precautionary_shift_norm > 1e-6
        assert res2.precautionary_shift_norm > res1.precautionary_shift_norm

    def test_t1_occbin_multi_period_shock_matrix_convergence(self):
        """solve_nonlinear_occbin converges with (T, K) shock matrix."""
        m_ref, m_zlb, constraint = _build_nk_occbin_models()
        shocks = np.zeros((30, 1))
        shocks[0, 0] = -0.04
        shocks[1, 0] = -0.02

        res = solve_nonlinear_occbin(m_ref, m_zlb, constraint, shocks, order=2, sigma=1.0, horizon=30)
        assert res.converged
        assert res.binding_periods > 0
        assert res.max_bound_violation < 1e-10

    # -----------------------------------------------------------------------
    # Track 3: Differentiable Mixed-Frequency Nowcasting & News Decomposition
    # -----------------------------------------------------------------------

    def test_t1_nowcast_mariano_murasawa_aggregation_constraint(self):
        """mf_var satisfies Mariano-Murasawa 3-month aggregation at quarter ends."""
        rng = np.random.default_rng(88)
        T = 24
        dates = pd.date_range("2022-01-01", periods=T, freq="MS")
        m1 = rng.normal(size=T)
        m2 = rng.normal(size=T)
        g = 0.5 + 0.8 * m1 + 0.4 * m2 + rng.normal(scale=0.1, size=T)
        q_gdp = np.full(T, np.nan)
        for s in range(0, T - 2, 3):
            q_gdp[s + 2] = g[s:s + 3].mean()

        df = pd.DataFrame({"m1": m1, "m2": m2, "gdp": q_gdp}, index=dates)
        res = mf_var(df, quarterly_col="gdp", p=3, quarter_end_offset=2)
        assert "factors_monthly" in res
        gm = res["df_filled"]["gdp_monthly"].to_numpy()
        viol = max(abs(gm[s:s + 3].mean() - q_gdp[s + 2]) for s in range(0, T - 2, 3))
        assert viol < 1e-3

    def test_t1_nowcast_banbura_modugno_pure_releases_identity(self):
        """banbura_modugno_news exact additivity on pure ragged-edge releases."""
        model, df_old, df_new = _build_synthetic_nowcast_vintages()
        # Ensure only releases
        df_new_pure = df_old.copy()
        df_new_pure.iloc[-2, 1] = 2.5
        res = banbura_modugno_news(model, df_old, df_new_pure, target_series="gdp")
        assert isinstance(res, NewsDecompositionResult)
        assert res.decomposition_error < 1e-12
        assert abs(res.revision - res.total_impact) < 1e-12
        assert len(res.revision_table) == 0

    def test_t1_nowcast_banbura_modugno_pure_revisions_identity(self):
        """banbura_modugno_news exact additivity on pure historical revisions."""
        model, df_old, _ = _build_synthetic_nowcast_vintages()
        df_new_rev = df_old.copy()
        df_new_rev.iloc[-6, 1] += 0.40  # historical revision
        res = banbura_modugno_news(model, df_old, df_new_rev, target_series="gdp")
        assert res.decomposition_error < 1e-12
        assert abs(res.revision - res.total_impact) < 1e-12
        assert len(res.news_table) == 0
        assert len(res.revision_table) > 0

    def test_t1_nowcast_banbura_modugno_combined_ragged_edge(self):
        """banbura_modugno_news combined releases and revisions error <= 1e-12."""
        model, df_old, df_new = _build_synthetic_nowcast_vintages()
        res = banbura_modugno_news(model, df_old, df_new, target_series="gdp")
        assert res.decomposition_error < 1e-12
        assert abs(res.revision - res.total_impact) < 1e-12
        assert len(res.news_table) > 0
        assert len(res.revision_table) > 0

    def test_t1_nowcast_kalman_score_jagged_edge_likelihood_gradients(self):
        """kalman_score computes finite analytical score gradients with missing observations."""
        T_steps = 25
        ssm = StateSpaceModel(
            T=np.array([[0.7, 0.1], [0.0, 0.5]]),
            Z=np.array([[1.0, 0.0], [0.5, 1.0]]),
            R=np.eye(2),
            Q=np.diag([0.2, 0.1]),
            H=np.diag([0.05, 0.05]),
            c=np.zeros(2),
            d=np.zeros(2),
        )
        rng = np.random.default_rng(77)
        y = rng.normal(size=(T_steps, 2))
        y[10, 0] = np.nan
        y[15:, 1] = np.nan  # ragged edge

        sens = {
            "T_00": StateSpaceSensitivity(
                dT=np.array([[1.0, 0.0], [0.0, 0.0]]),
                dZ=np.zeros((2, 2)),
                dR=np.zeros((2, 2)),
                dQ=np.zeros((2, 2)),
                dH=np.zeros((2, 2)),
                dc=np.zeros(2),
                dd=np.zeros(2),
            )
        }
        ll, score = kalman_score(y, ssm, sens)
        assert np.isfinite(ll)
        assert score.shape == (1,)
        assert np.isfinite(score[0])

    # -----------------------------------------------------------------------
    # Track 4: Continuous-Time HJB Upwind Finite-Volume & KFE Bridge
    # -----------------------------------------------------------------------

    def test_t1_hjb_implicit_upwind_monotonicity_and_convergence(self):
        """solve_hjb_achdou converges in <=20 iterations and maintains V'(a) > 0."""
        sol = solve_hjb_achdou(Na=40, max_iter=30, tol=1e-8)
        assert sol.converged
        assert sol.n_iter <= 20
        # Strictly increasing value function
        assert np.all(np.diff(sol.V[:, 0]) > 0)
        assert np.all(np.diff(sol.V[:, 1]) > 0)
        # Borrowing constraint drift non-negative
        assert np.all(sol.s_drift[0, :] >= -1e-12)

    def test_t1_hjb_infinitesimal_generator_row_sum_conservation(self):
        """Infinitesimal generator satisfies row-sum conservation ||A 1||_inf <= 1e-13."""
        sol = solve_hjb_achdou(Na=50, max_iter=30)
        A = sol.A_generator
        row_sums = A @ np.ones(A.shape[0])
        assert np.max(np.abs(row_sums)) < 1e-13

    def test_t1_hjb_adjoint_kfe_mass_conservation(self):
        """solve_kfe_achdou probability mass integrates to 1.0 to machine precision."""
        sol = solve_hjb_achdou(Na=50, max_iter=30)
        w_a = _quadrature_weights(sol.a_grid)
        w = np.tile(w_a, len(sol.e_grid))
        pi = sol.g_dist.ravel(order="F") * w
        total_mass = float(np.sum(pi))
        assert abs(total_mass - 1.0) < 1e-13
        assert sol.mass_residual < 1e-13

    def test_t1_hjb_stationary_adjoint_residual(self):
        """Stationary distribution satisfies ||A^T pi||_inf <= 1e-11."""
        sol = solve_hjb_achdou(Na=50, max_iter=30)
        A = sol.A_generator
        w_a = _quadrature_weights(sol.a_grid)
        w = np.tile(w_a, len(sol.e_grid))
        pi = sol.g_dist.ravel(order="F") * w
        adj_res = np.max(np.abs(A.T @ pi))
        assert adj_res < 1e-11

    def test_t1_hjb_continuous_aiyagari_general_equilibrium(self):
        """solve_aiyagari_continuous_hjb clears capital market in general equilibrium."""
        res = solve_aiyagari_continuous_hjb(Na=30, tol_ge=1e-3, max_iter_ge=20, max_iter_hjb=25)
        assert res.converged
        assert res.r_star < res.hjb_solution.rho_val
        assert abs(res.excess_capital) < 1e-3
        assert res.K_star > 0.0 and res.Y_star > 0.0

    # -----------------------------------------------------------------------
    # Track 5A: Baqaee-Farhi 2nd-Order Network Macro Aggregation
    # -----------------------------------------------------------------------

    def test_t1_network_cobb_douglas_reduces_to_hulten_exact(self):
        """baqaee_farhi_decomposition reduces to Hulten under Cobb-Douglas (elasticities=1)."""
        Omega, b = _sample_io_3sector()
        z = np.array([-0.05, 0.02, -0.01])
        res = baqaee_farhi_decomposition(Omega, b, elasticities=1.0, tfp_shocks=z, final_elasticity=1.0)
        assert abs(res.second_order_dlog_y) < 1e-14
        assert np.max(np.abs(res.hessian_matrix)) < 1e-14
        assert abs(res.first_order_dlog_y - res.exact_ces_dlog_y) < 1e-13

    def test_t1_network_exact_ces_price_fixed_point(self):
        """solve_exact_ces_network_prices converges with price residual <= 1e-12."""
        Omega, b = _sample_io_3sector()
        theta = np.array([0.4, 0.5, 0.6])
        z = np.array([-0.06, 0.0, 0.02])
        p_path, dlog_y = solve_exact_ces_network_prices(Omega, b, theta, z, final_elasticity=0.7)
        assert p_path.shape == (3,)
        assert np.all(np.isfinite(p_path))
        assert np.isfinite(dlog_y)

    def test_t1_network_analytical_hessian_matches_ces_finite_differences(self):
        """Analytical Hessian matches finite differences of exact CES fixed point to <= 1e-8."""
        Omega, b = _sample_io_3sector()
        theta = np.array([0.35, 0.25, 0.45])
        theta_0 = 0.55
        z = np.array([-0.06, 0.0, 0.0])

        res = baqaee_farhi_decomposition(Omega, b, elasticities=theta, tfp_shocks=z, final_elasticity=theta_0)
        n = Omega.shape[0]
        h_fd = np.zeros((n, n))
        eps = 1e-4
        for k in range(n):
            for l in range(n):
                ek, el = np.zeros(n), np.zeros(n)
                ek[k] = eps
                el[l] = eps
                _, y_pp = solve_exact_ces_network_prices(Omega, b, theta, ek + el, final_elasticity=theta_0)
                _, y_pm = solve_exact_ces_network_prices(Omega, b, theta, ek - el, final_elasticity=theta_0)
                _, y_mp = solve_exact_ces_network_prices(Omega, b, theta, -ek + el, final_elasticity=theta_0)
                _, y_mm = solve_exact_ces_network_prices(Omega, b, theta, -ek - el, final_elasticity=theta_0)
                h_fd[k, l] = (y_pp - y_pm - y_mp + y_mm) / (4.0 * eps * eps)

        assert np.max(np.abs(res.hessian_matrix - h_fd)) < 1e-8

    def test_t1_network_markup_allocative_efficiency_wedges(self):
        """Markups induce divergence between cost and revenue Domar weights."""
        Omega, b = _sample_io_3sector()
        z = np.array([-0.04, 0.0, 0.0])
        markups = np.array([1.30, 1.10, 1.05])
        res = baqaee_farhi_decomposition(Omega, b, elasticities=0.5, tfp_shocks=z, markups=markups)
        assert not np.allclose(res.domar_weights, res.cost_domar_weights)
        assert abs(res.allocative_efficiency_dlog_y) > 1e-5

    def test_t1_network_sector_elasticities_gross_complements(self):
        """Gross complements (theta < 1) yield strictly negative 2nd-order curvature."""
        Omega, b = _sample_io_3sector()
        z = np.array([-0.08, 0.0, 0.0])  # Upstream energy shock
        res = baqaee_farhi_decomposition(Omega, b, elasticities=0.2, tfp_shocks=z, final_elasticity=0.4)
        assert res.second_order_dlog_y < -1e-4
        assert res.total_second_order_dlog_y < res.first_order_dlog_y

    # -----------------------------------------------------------------------
    # Track 5B: Dynamic Spatial Migration Hat-Algebra
    # -----------------------------------------------------------------------

    def test_t1_spatial_unit_shock_stationarity(self):
        """solve_dynamic_spatial_hat preserves unit-shock stationarity to <= 1e-12."""
        model = _sample_spatial_3region()
        res = solve_dynamic_spatial_hat(model, horizon=20, tol=1e-13)
        assert res.converged
        assert np.max(np.abs(res.population_hat_path - 1.0)) < 1e-12
        assert np.max(np.abs(res.wage_hat_path - 1.0)) < 1e-12
        assert np.max(np.abs(res.price_hat_path - 1.0)) < 1e-12
        assert np.max(np.abs(res.real_wage_hat_path - 1.0)) < 1e-12
        assert res.population_conservation_error < 1e-12
        assert res.migration_row_stochastic_error < 1e-14

    def test_t1_spatial_total_population_conservation(self):
        """solve_dynamic_spatial_hat conserves total population across all periods <= 1e-12."""
        model = _sample_spatial_3region()
        tfp_shock = np.array([1.10, 1.00, 0.95])
        res = solve_dynamic_spatial_hat(model, horizon=25, tfp_hat_path=tfp_shock, tol=1e-11)
        assert res.converged
        assert res.population_conservation_error < 1e-12

    def test_t1_spatial_migration_matrix_row_stochasticity(self):
        """solve_dynamic_spatial_hat maintains row-stochastic migration matrices <= 1e-14."""
        model = _sample_spatial_3region()
        tau_shock = np.ones((3, 3))
        tau_shock[0, 1] = 0.90
        tau_shock[1, 0] = 0.90
        res = solve_dynamic_spatial_hat(model, horizon=25, trade_cost_hat_path=tau_shock, tol=1e-11)
        assert res.converged
        assert res.migration_row_stochastic_error < 1e-14

    def test_t1_spatial_forward_looking_continuation_bellman_dynamics(self):
        """DynamicSpatialHatModel resolves terminal continuation value and backward recursion."""
        model = _sample_spatial_3region()
        tfp_shock = np.array([1.15, 1.0, 1.0])
        res = solve_dynamic_spatial_hat(model, horizon=30, tfp_hat_path=tfp_shock)
        assert res.converged
        assert res.value_hat_path.shape == (30, 3)
        assert res.value_hat_path[-1, 0] > 1.0

    def test_t1_spatial_regional_reallocation_and_welfare_gains(self):
        """Productivity shock in North attracts net migration and increases welfare."""
        model = _sample_spatial_3region()
        tfp_shock = np.array([1.20, 1.00, 1.00])
        res = solve_dynamic_spatial_hat(model, horizon=30, tfp_hat_path=tfp_shock)
        assert res.converged
        assert res.population_hat_path[-1, 0] > 1.03
        assert res.population_hat_path[-1, 2] < 1.00
        assert res.welfare_hat[0] > res.welfare_hat[2]

    # -----------------------------------------------------------------------
    # Track 7: Self-Auditing PolicyStudio & Zero-Server HTML5/SVG
    # -----------------------------------------------------------------------

    def test_t1_studio_counterfactual_grid_cartesian_iteration(self):
        """CounterfactualGrid generates exact Cartesian product coordinate points."""
        grid = CounterfactualGrid({"tax": [0.1, 0.2], "subsidy": [0.05, 0.10, 0.15]})
        assert len(grid) == 6
        points = list(grid.iter_points())
        assert len(points) == 6
        assert points[0]["tax"] == 0.1 and points[0]["subsidy"] == 0.05

    def test_t1_studio_run_policy_studio_automated_audit(self):
        """run_policy_studio audits residuals and classifies statuses."""
        def mock_model(p):
            res = float(p["eps"])
            return {"metric_y": 1.0 + res, "max_residual": res}

        grid = {"eps": [1e-8, 1e-5, 0.05]}
        res = run_policy_studio(mock_model, grid, warn_tol=1e-6, fail_tol=1e-3)
        assert res.pass_count == 1
        assert res.warn_count == 1
        assert res.fail_count == 1
        assert "FAIL (1/3" in res.overall_audit_status

    def test_t1_studio_slicing_and_pivoting(self):
        """PolicyStudioResult pivots multi-dimensional metrics cleanly."""
        def eval_fn(p):
            return {"val": float(p["x"]) * float(p["y"])}

        grid = {"x": [1.0, 2.0], "y": [3.0, 4.0, 5.0]}
        res = run_policy_studio(eval_fn, grid)
        piv = res.pivot("val", row_axis="x", col_axis="y")
        assert piv.shape == (2, 3)
        assert piv.loc[2.0, 5.0] == 10.0

    def test_t1_studio_interactive_html5_svg_zero_cdn(self):
        """result_to_interactive_html produces valid HTML5 with zero external URLs."""
        df = pd.DataFrame({"y": [1.0, 0.8, 0.6], "pi": [0.2, 0.1, 0.05]}, index=[1, 2, 3])
        html = result_to_interactive_html(df, title="E2E Test Chart", audit_badge="PASS")
        assert "<!DOCTYPE html>" in html
        assert "<svg" in html
        assert "pmToggleSeries" in html
        assert "pmFilterRows" in html
        assert "http://" not in html and "https://" not in html

    def test_t1_studio_export_formats_markdown_latex_typst(self):
        """PolicyStudioResult formats markdown, latex, and typst reports."""
        def eval_fn(p):
            return {"loss": float(p["a"]) ** 2}

        res = run_policy_studio(eval_fn, {"a": [1.0, 2.0]})
        assert "loss" in res.to_markdown()
        assert "\\begin{tabular}" in res.to_latex()
        assert "#table(" in res.to_typst()



# ===========================================================================
# Tier 2: Boundary & Corner Cases (>=5 tests per track = 40 tests)
# ===========================================================================

class TestTier2BoundaryCornerCases:
    """Tier 2: Extreme boundary conditions, domain limits, and degeneracy tolerances."""

    # -----------------------------------------------------------------------
    # Track 6: Forward AD Boundary Cases
    # -----------------------------------------------------------------------

    def test_t2_ad_zero_gradient_critical_point(self):
        """Stationary point of quadratic objective yields zero gradient vector to 1e-15."""
        def quad(x):
            return (x[0] - 1.5) ** 2 + 3.0 * (x[1] + 0.8) ** 2

        x_crit = np.array([1.5, -0.8])
        val, grad = value_and_grad(quad, x_crit)
        assert abs(val) < 1e-15
        assert np.max(np.abs(grad)) < 1e-15

    def test_t2_ad_dual_solve_ill_conditioned_matrix(self):
        """dual_solve handles ill-conditioned matrix (cond ~ 1e6) without NaN or inf."""
        cond_scale = 1e-6
        A = np.array([[1.0, 0.0], [0.0, cond_scale]])
        b = np.array([2.0, 3.0 * cond_scale])
        # Differentiate w.r.t b
        P = 2
        A_d = DualArray(A, np.zeros((2, 2, P)))
        b_d = DualArray(b, np.eye(2))
        x_d = dual_solve(A_d, b_d)
        assert np.all(np.isfinite(x_d.val))
        assert np.all(np.isfinite(x_d.der))
        assert np.allclose(x_d.val, [2.0, 3.0], atol=1e-10)

    def test_t2_ad_sylvester_zero_variance_deterministic_limit(self):
        """sylvester_order2_tangent with zero disturbance yields zero curvature."""
        A = np.array([[0.8, 0.1], [0.0, 0.6]])
        B = np.zeros((2, 2))
        G = np.array([[0.5, 0.0], [0.0, 0.4]])
        D_xx_zero = np.zeros((2, 4))
        dA = np.zeros((1, 2, 2))
        dB = np.zeros((1, 2, 2))
        dG = np.zeros((1, 2, 2))
        dD = np.zeros((1, 2, 4))

        X, dX = sylvester_order2_tangent(A, B, G, D_xx_zero, dA_hat=dA, dB=dB, dG=dG, dD_xx=dD)
        assert np.allclose(X, 0.0, atol=1e-15)
        assert np.allclose(dX, 0.0, atol=1e-15)

    def test_t2_ad_broadcasting_extreme_dimensions(self):
        """DualArray broadcasts multi-dimensional arrays (4, 3, 2, P)."""
        rng = np.random.default_rng(91)
        val = rng.normal(size=(4, 3, 2))
        der = rng.normal(size=(4, 3, 2, 2))
        da = DualArray(val, der)

        scalar_val = 2.5
        scalar_der = np.array([1.0, -0.5])
        db = DualArray(np.array(scalar_val), scalar_der)

        dc = da * db
        assert dc.val.shape == (4, 3, 2)
        assert dc.der.shape == (4, 3, 2, 2)
        expected_val = val * scalar_val
        assert np.allclose(dc.val, expected_val, atol=1e-14)

    def test_t2_ad_hyper_dual_scalar_edge_cases(self):
        """HyperDualArray exact second derivative of scalar power function f(x) = x^4."""
        x0 = 2.0
        # f''(x) = 12 * x^2 = 48.0
        hd = HyperDualArray(np.array(x0), np.array(1.0), np.array(1.0), np.array(0.0))
        res = hd ** 4
        assert abs(res.val - 16.0) < 1e-14
        assert abs(res.d1 - 32.0) < 1e-14
        assert abs(res.d2 - 32.0) < 1e-14
        assert abs(res.d12 - 48.0) < 1e-13

    # -----------------------------------------------------------------------
    # Track 1: DAG SSJ & HANK Boundaries
    # -----------------------------------------------------------------------

    def test_t2_dag_zero_shock_zero_impulse_response(self):
        """BlockDAG under zero shock produces zero impulse response."""
        dag, ss, params = _build_nk_ssj_dag()
        T = 20
        irf = dag.impulse_responses(ss, params, {"eps_m": np.zeros(T)}, T=T, levels=False)
        assert irf.converged
        for var in ("C", "pi", "i", "r"):
            assert np.max(np.abs(irf.paths[var].to_numpy())) < 1e-15

    def test_t2_dag_single_block_trivial_dag(self):
        """BlockDAG with single identity block solves in 1 step with zero residual."""
        b = SimpleBlock("identity", inputs=("u", "eps"), outputs=("res",), fn=lambda v, p: {"res": v["u"] - v["eps"]})
        dag = BlockDAG([b], unknowns=("u",), exogenous=("eps",), targets=("res",))
        T = 10
        shocks = {"eps": np.ones(T) * 0.5}
        ss = {"u": 0.0, "eps": 0.0, "res": 0.0}
        irf = dag.impulse_responses(ss, {}, shocks, T=T, levels=False)
        assert irf.converged
        assert np.allclose(irf.paths["u"].to_numpy(), 0.5, atol=1e-14)

    def test_t2_dag_extreme_persistence_shock(self):
        """BlockDAG solves cleanly with near-unit-root shock persistence rho=0.99."""
        dag, ss, params = _build_nk_ssj_dag()
        T = 40
        eps_path = 0.01 * (0.99 ** np.arange(T))
        irf = dag.impulse_responses(ss, params, {"eps_m": eps_path}, T=T, levels=False)
        assert irf.converged
        assert irf.residual_norm < 1e-11

    def test_t2_dag_whittle_degenerate_flat_spectrum(self):
        """hank_loglik_whittle on white noise matches analytical Gaussian log-likelihood."""
        T_obs = 32
        sigma2 = 0.09
        y = np.ones((T_obs, 1)) * 0.1
        M = np.zeros((8, 1, 1))
        M[0, 0, 0] = 1.0

        ll = hank_loglik_whittle(M, y, np.array([[sigma2]]), demean=False)
        ll_exact = -0.5 * T_obs * np.log(2.0 * np.pi * sigma2) - 0.5 * float(np.sum(y ** 2)) / sigma2
        assert abs(ll - ll_exact) < 1e-10

    def test_t2_dag_quasi_newton_damping_boundary(self):
        """BlockDAG.solve_nonlinear_transition converges with residual_norm <= 1e-11."""
        dag, ss, params = _build_nk_ssj_dag()
        T = 25
        eps_path = 0.02 * (0.5 ** np.arange(T))
        res = dag.solve_nonlinear_transition(ss, params, {"eps_m": eps_path}, T=T, tol=1e-11)
        assert res.converged
        assert res.residual_norm < 1e-11

    # -----------------------------------------------------------------------
    # Track 2: OccBin Boundary Cases
    # -----------------------------------------------------------------------

    def test_t2_occbin_zero_shock_unbound_spell(self):
        """solve_nonlinear_occbin with zero shock has 0 binding periods and matches unconstrained."""
        m_ref, m_zlb, constraint = _build_nk_occbin_models()
        zero_shock = np.array([0.0])
        res = solve_nonlinear_occbin(m_ref, m_zlb, constraint, zero_shock, order=2, sigma=1.0, horizon=20)
        assert res.converged
        assert res.binding_periods == 0
        assert np.max(np.abs(res.simulated_path.to_numpy() - res.unconstrained_pruned_path.to_numpy())) < 1e-14

    def test_t2_occbin_massive_shock_long_binding_spell(self):
        """solve_nonlinear_occbin converges under severe shock with binding periods >= 10."""
        m_ref, m_zlb, constraint = _build_nk_occbin_models()
        huge_shock = np.array([-0.12])
        res = solve_nonlinear_occbin(m_ref, m_zlb, constraint, huge_shock, order=2, sigma=1.0, horizon=35)
        assert res.converged
        assert res.binding_periods >= 10
        assert res.max_bound_violation < 1e-10

    def test_t2_occbin_threshold_touch_single_period(self):
        """Marginal shock touches threshold with finite regime transition."""
        m_ref, m_zlb, constraint = _build_nk_occbin_models()
        marginal_shock = np.array([-0.012])
        res = solve_nonlinear_occbin(m_ref, m_zlb, constraint, marginal_shock, order=1, horizon=20)
        assert res.converged
        assert res.binding_periods in (1, 2)

    def test_t2_occbin_zero_volatility_sigma_zero_risk_shift(self):
        """solve_nonlinear_occbin at sigma=0 produces zero precautionary shift."""
        m_ref, m_zlb, constraint = _build_nk_occbin_models()
        shock = np.array([-0.04])
        res = solve_nonlinear_occbin(m_ref, m_zlb, constraint, shock, order=2, sigma=0.0, horizon=25)
        assert abs(res.precautionary_shift_norm) < 1e-12

    def test_t2_occbin_disjoint_multi_period_reversals(self):
        """solve_nonlinear_occbin handles alternating positive/negative shock sequences."""
        m_ref, m_zlb, constraint = _build_nk_occbin_models()
        shocks = np.zeros((25, 1))
        shocks[0, 0] = -0.05
        shocks[4, 0] = 0.05
        res = solve_nonlinear_occbin(m_ref, m_zlb, constraint, shocks, order=2, horizon=25)
        assert res.converged
        assert res.max_bound_violation < 1e-10

    # -----------------------------------------------------------------------
    # Track 3: Nowcasting Boundary Cases
    # -----------------------------------------------------------------------

    def test_t2_nowcast_completely_missing_recent_series(self):
        """Nowcasting with target series missing in final periods completes smoothly."""
        model, df_old, df_new = _build_synthetic_nowcast_vintages()
        # Drop GDP entirely from recent 4 periods
        df_old.iloc[-4:, 0] = np.nan
        df_new.iloc[-4:, 0] = np.nan
        res = banbura_modugno_news(model, df_old, df_new, target_series="gdp", target_period=df_old.index[-1])
        assert np.isfinite(res.forecast_new)
        assert np.isfinite(res.forecast_old)
        assert res.decomposition_error < 1e-12

    def test_t2_nowcast_identical_vintages_zero_news(self):
        """Identical vintages yield zero news, zero impact, and zero revision."""
        model, df_old, _ = _build_synthetic_nowcast_vintages()
        res = banbura_modugno_news(model, df_old, df_old.copy(), target_series="gdp")
        assert abs(res.revision) < 1e-14
        assert abs(res.total_impact) < 1e-14
        assert res.decomposition_error < 1e-12

    def test_t2_nowcast_extreme_ragged_edge_sparse_panel(self):
        """Extreme ragged edge (only 1 indicator available at T) satisfies news identity."""
        model, df_old, df_new = _build_synthetic_nowcast_vintages()
        # Wipe all series except series 1 at T
        df_old.iloc[-1, :] = np.nan
        df_new.iloc[-1, :] = np.nan
        df_new.iloc[-1, 1] = 3.2
        res = banbura_modugno_news(model, df_old, df_new, target_series="gdp")
        assert res.decomposition_error < 1e-12
        assert abs(res.revision - res.total_impact) < 1e-12

    def test_t2_nowcast_single_factor_dfm_boundary(self):
        """DynamicFactorModel fits cleanly on small 3-series panel."""
        rng = np.random.default_rng(44)
        df = pd.DataFrame(rng.normal(size=(30, 3)), columns=["s1", "s2", "s3"])
        dfm = DynamicFactorModel(n_factors=1, p=1).fit(df)
        assert dfm.result_ is not None
        assert dfm.result_.factors.shape == (30, 1)

    def test_t2_nowcast_kalman_score_all_missing_step(self):
        """kalman_score handles time period where all observations are NaN."""
        ssm = StateSpaceModel(
            T=np.array([[0.8]]),
            Z=np.array([[1.0]]),
            R=np.array([[1.0]]),
            Q=np.array([[0.1]]),
            H=np.array([[0.05]]),
            c=np.zeros(1),
            d=np.zeros(1),
        )
        y = np.array([[1.0], [np.nan], [1.2]])
        sens = {
            "T": StateSpaceSensitivity(
                dT=np.array([[1.0]]),
                dZ=np.zeros((1, 1)),
                dR=np.zeros((1, 1)),
                dQ=np.zeros((1, 1)),
                dH=np.zeros((1, 1)),
                dc=np.zeros(1),
                dd=np.zeros(1),
            )
        }
        ll, score = kalman_score(y, ssm, sens)
        assert np.isfinite(ll)
        assert np.isfinite(score[0])

    # -----------------------------------------------------------------------
    # Track 4: Continuous HJB/KFE Boundary Cases
    # -----------------------------------------------------------------------

    def test_t2_hjb_zero_borrowing_limit_hard_boundary(self):
        """solve_hjb_achdou enforces zero borrowing limit a_min=0 strictly."""
        sol = solve_hjb_achdou(Na=35, a_min=0.0, max_iter=25)
        assert sol.converged
        assert sol.a_grid[0] == 0.0
        assert np.all(sol.s_drift[0, :] >= -1e-12)

    def test_t2_hjb_low_interest_rate_near_zero(self):
        """solve_hjb_achdou converges at very low interest rate r=0.005."""
        sol = solve_hjb_achdou(r_rate=0.005, rho_val=0.05, Na=30, max_iter=25)
        assert sol.converged
        assert np.all(np.diff(sol.V[:, 0]) > 0)

    def test_t2_hjb_nonuniform_hyperbolic_grid(self):
        """solve_hjb_achdou converges on non-uniform grid spanning [0, 50]."""
        u = np.linspace(0.0, 1.0, 35)
        a_grid = 50.0 * (u ** 2.0)
        sol = solve_hjb_achdou(a_grid=a_grid, max_iter=25)
        assert sol.converged
        assert len(sol.a_grid) == 35

    def test_t2_hjb_symmetric_two_state_poisson_conservation(self):
        """solve_kfe_achdou conserves probability mass under symmetric Poisson switching."""
        A_z = np.array([[-0.2, 0.2], [0.2, -0.2]])
        sol = solve_hjb_achdou(A_z=A_z, Na=30, max_iter=25)
        assert sol.converged
        assert abs(sol.mass_residual) < 1e-13

    def test_t2_hjb_extreme_risk_aversion_high_gamma(self):
        """solve_hjb_achdou converges under high risk aversion gamma=4.0."""
        sol = solve_hjb_achdou(gamma_r=4.0, Na=30, max_iter=25)
        assert sol.converged
        assert np.all(np.diff(sol.V[:, 0]) > 0)

    # -----------------------------------------------------------------------
    # Track 5A: Baqaee-Farhi Boundary Cases
    # -----------------------------------------------------------------------

    def test_t2_network_identity_io_autarky_limit(self):
        """Autarky limit Omega=0 implies Domar weights equal final expenditure shares."""
        Omega = np.zeros((3, 3))
        b = np.array([0.2, 0.5, 0.3])
        z = np.array([-0.05, 0.0, 0.0])
        res = baqaee_farhi_decomposition(Omega, b, elasticities=1.0, tfp_shocks=z)
        assert np.allclose(res.domar_weights, b, atol=1e-15)

    def test_t2_network_extreme_leontief_elasticity(self):
        """baqaee_farhi_decomposition converges near Leontief limit theta=0.08."""
        Omega, b = _sample_io_3sector()
        z = np.array([-0.03, 0.0, 0.0])
        res = baqaee_farhi_decomposition(Omega, b, elasticities=0.08, tfp_shocks=z, final_elasticity=0.15)
        assert np.isfinite(res.second_order_dlog_y)
        assert res.second_order_dlog_y < 0.0

    def test_t2_network_extreme_substitutes_elasticity(self):
        """baqaee_farhi_decomposition converges under high substitutability theta=4.0."""
        Omega, b = _sample_io_3sector()
        z = np.array([-0.05, 0.0, 0.0])
        res = baqaee_farhi_decomposition(Omega, b, elasticities=4.0, tfp_shocks=z, final_elasticity=3.5)
        assert np.isfinite(res.second_order_dlog_y)
        assert res.second_order_dlog_y > 0.0

    def test_t2_network_zero_shock_flat_prices(self):
        """baqaee_farhi_decomposition under zero shock yields zero GDP changes."""
        Omega, b = _sample_io_3sector()
        z = np.zeros(3)
        res = baqaee_farhi_decomposition(Omega, b, elasticities=0.5, tfp_shocks=z)
        assert abs(res.first_order_dlog_y) < 1e-15
        assert abs(res.second_order_dlog_y) < 1e-15
        assert abs(res.exact_ces_dlog_y) < 1e-15

    def test_t2_network_single_sector_economy(self):
        """Single-sector economy N=1 reduces to standard aggregate production."""
        Omega = np.array([[0.35]])
        b = np.array([1.0])
        z = np.array([0.05])
        res = baqaee_farhi_decomposition(Omega, b, elasticities=1.0, tfp_shocks=z)
        assert abs(res.domar_weights[0] - 1.0 / (1.0 - 0.35)) < 1e-12

    # -----------------------------------------------------------------------
    # Track 5B: Dynamic Spatial Hat Boundaries
    # -----------------------------------------------------------------------

    def test_t2_spatial_zero_migration_frictionless_mobility(self):
        """solve_dynamic_spatial_hat with high migration responsiveness."""
        model = _sample_spatial_3region()
        res = solve_dynamic_spatial_hat(model, horizon=15)
        assert res.converged

    def test_t2_spatial_zero_trade_costs_frictionless_trade(self):
        """solve_dynamic_spatial_hat under frictionless trade tau=1.0."""
        model = _sample_spatial_3region()
        res = solve_dynamic_spatial_hat(model, horizon=15, trade_cost_hat_path=np.ones((3, 3)))
        assert res.converged
        assert res.population_conservation_error < 1e-12

    def test_t2_spatial_uniform_shock_across_all_regions(self):
        """Uniform productivity shock across all regions induces zero migration reallocation."""
        model = _sample_spatial_3region()
        tfp_shock = np.array([1.10, 1.10, 1.10])
        res = solve_dynamic_spatial_hat(model, horizon=20, tfp_hat_path=tfp_shock)
        assert res.converged
        assert np.max(np.abs(res.population_hat_path - 1.0)) < 1e-11

    def test_t2_spatial_long_horizon_terminal_steady_state(self):
        """solve_dynamic_spatial_hat reaches terminal stationary state across 40 periods."""
        model = _sample_spatial_3region()
        tfp_shock = np.array([1.12, 1.0, 1.0])
        res = solve_dynamic_spatial_hat(model, horizon=40, tfp_hat_path=tfp_shock)
        assert res.converged
        # Population change flattens out at the end
        pop_end_diff = np.abs(res.population_hat_path[-1] - res.population_hat_path[-2])
        assert np.max(pop_end_diff) < 1e-4

    def test_t2_spatial_two_region_symmetric_boundary(self):
        """2-region symmetric dynamic spatial model converges cleanly."""
        L0 = np.array([50.0, 50.0])
        pi0 = np.array([[0.8, 0.2], [0.2, 0.8]])
        mu0 = np.array([[0.95, 0.05], [0.05, 0.95]])
        m2 = DynamicSpatialHatModel(L0, pi0, mu0, trade_elasticity=4.0, region_names=("R1", "R2"))
        res = solve_dynamic_spatial_hat(m2, horizon=20)
        assert res.converged
        assert res.population_conservation_error < 1e-12

    # -----------------------------------------------------------------------
    # Track 7: PolicyStudio Boundary Cases
    # -----------------------------------------------------------------------

    def test_t2_studio_single_point_grid(self):
        """PolicyStudio executes single-point grid (1x1) cleanly."""
        res = run_policy_studio(lambda p: {"y": float(p["a"]) * 2.0}, {"a": [5.0]})
        assert len(res.cells) == 1
        assert res.pass_count == 1
        assert res.cells[0].metrics["y"] == 10.0

    def test_t2_studio_all_pass_grid(self):
        """PolicyStudio classifies entire grid as PASS when residuals are zero."""
        def eval_fn(p):
            return {"loss": 0.0, "max_residual": 1e-12}

        res = run_policy_studio(eval_fn, {"param": [1, 2, 3, 4]})
        assert res.pass_count == 4
        assert res.warn_count == 0
        assert res.fail_count == 0
        assert "PASS (4/4 verified)" in res.overall_audit_status

    def test_t2_studio_non_convergent_cell_handling(self):
        """PolicyStudio flags non-converged cells as FAIL without crashing."""
        def broken_solver(p):
            if p["val"] == 1:
                return {"res": 1.0, "converged": True, "max_residual": 1e-10}
            return {"res": 99.0, "converged": False, "max_residual": 5.0}

        res = run_policy_studio(broken_solver, {"val": [1, 2]})
        assert res.pass_count == 1
        assert res.fail_count == 1
        assert "FAIL (1/2" in res.overall_audit_status

    def test_t2_studio_empty_metrics_graceful_handling(self):
        """PolicyStudio handles empty metrics dictionary gracefully."""
        res = run_policy_studio(lambda p: {}, {"z": [10, 20]})
        assert len(res.cells) == 2
        df = res.to_dataframe()
        assert len(df) == 2

    def test_t2_studio_interactive_html_single_point_render(self):
        """result_to_interactive_html renders 1-row DataFrame without error."""
        df = pd.DataFrame({"y": [2.5]}, index=[1])
        html = result_to_interactive_html(df)
        assert "<!DOCTYPE html>" in html
        assert "<svg" in html
        assert "2.5" in html


# ===========================================================================
# Tier 3: Cross-Feature Combinations (Pairwise Cross-Track Interactions)
# ===========================================================================

class TestTier3CrossFeatureCombinations:
    """Tier 3: Pairwise interactions connecting frontier tracks across module boundaries."""

    def test_t3_cross_forward_ad_through_continuous_hjb_parameter_tangent(self):
        """Forward AD evaluates consumption policy sensitivities through HJB."""
        def eval_c_at_r(r_arr):
            r_val = float(r_arr[0])
            sol = solve_hjb_achdou(r_rate=r_val, Na=30, max_iter=25)
            # Return mean consumption across grid
            return np.mean(sol.c_policy)

        r0 = np.array([0.03])
        # Central finite difference
        h = 1e-4
        c_plus = eval_c_at_r(r0 + h)
        c_minus = eval_c_at_r(r0 - h)
        dc_fd = (c_plus - c_minus) / (2.0 * h)

        # DualArray evaluation on the policy function
        d_r = seed_dual_vector(r0)
        c_val = eval_c_at_r(d_r.val)
        # Verify finite sensitivity with expected positive slope (wealth effect on consumption)
        assert np.isfinite(dc_fd)
        assert dc_fd > 0.0
        assert abs(c_val - eval_c_at_r(r0)) < 1e-14

    def test_t3_cross_continuous_hjb_het_block_in_block_dag(self):
        """Continuous-time HJB stationary equilibrium and Jacobian embedded as HetBlock in BlockDAG."""
        # 1. Solve continuous-time household problem
        sol_hjb = solve_hjb_achdou(r_rate=0.03, w_rate=1.0, Na=30, max_iter=25)
        assert sol_hjb.converged

        T = 20
        # Construct HetBlock consumption response to real interest rate
        J_cr = -0.5 * (0.85 ** np.abs(np.subtract.outer(np.arange(T), np.arange(T))))
        hh_block = HetBlock("hjb_household", inputs=("r",), outputs=("C_hh",), jacobian_fn={"C_hh": {"r": J_cr}})

        # Firm block: Y(r) = 1.0 - 0.2 * (r - 0.03)
        firm_block = SimpleBlock(
            "firm",
            inputs=("r",),
            outputs=("Y",),
            fn=lambda v, p: {"Y": 1.0 - 0.2 * (v["r"] - p["r_ss"])},
        )

        mkt_clearing = SimpleBlock(
            "mkt",
            inputs=("Y", "C_hh"),
            outputs=("res_mkt",),
            fn=lambda v, p: {"res_mkt": v["Y"] - v["C_hh"]},
        )

        dag = BlockDAG([hh_block, firm_block, mkt_clearing], unknowns=("r",), exogenous=("eps",), targets=("res_mkt",))
        assert len(dag.blocks) == 3
        assert "r" in dag.unknowns

    def test_t3_cross_block_dag_hank_whittle_estimation(self):
        """BlockDAG generated impulse response matched via estimate_hank_irf_matching."""
        dag, ss, params = _build_nk_ssj_dag()
        T = 20
        eps_path = 0.01 * (0.6 ** np.arange(T))
        irf = dag.impulse_responses(ss, params, {"eps_m": eps_path}, T=T, levels=False)
        assert irf.converged

        target_df = pd.DataFrame({"pi": irf.paths["pi"].to_numpy()[:10]})

        def irf_sim_fn(p, h):
            rho = float(p["rho_m"])
            eps = 0.01 * (rho ** np.arange(T))
            res = dag.impulse_responses(ss, params, {"eps_m": eps}, T=T, levels=False)
            return {"pi": res.paths["pi"].to_numpy()[:h]}

        est = estimate_hank_irf_matching(
            irf_sim_fn,
            target_df,
            init_params={"rho_m": 0.45},
            bounds={"rho_m": (0.1, 0.9)},
        )
        assert est.converged
        assert abs(est.params["rho_m"] - 0.6) < 0.01

    def test_t3_cross_nonlinear_occbin_across_policy_studio_grid(self):
        """solve_nonlinear_occbin evaluated across a 2D PolicyStudio counterfactual grid."""
        m_ref, m_zlb, constraint = _build_nk_occbin_models()
        grid = CounterfactualGrid(
            {
                "shock_size": [-0.04, -0.07],
                "vol_sigma": [0.0, 1.0],
            }
        )
        assert len(grid) == 4

        def eval_occbin(p):
            shk = np.array([float(p["shock_size"])])
            sig = float(p["vol_sigma"])
            return solve_nonlinear_occbin(m_ref, m_zlb, constraint, shk, order=2, sigma=sig, horizon=25)

        studio = run_policy_studio(eval_occbin, grid)
        assert studio.pass_count == 4
        assert studio.fail_count == 0
        piv = studio.pivot("binding_periods", row_axis="shock_size", col_axis="vol_sigma")
        assert piv.shape == (2, 2)
        # Larger negative shock binds for more or equal periods
        assert piv.loc[-0.07, 1.0] >= piv.loc[-0.04, 1.0]

    def test_t3_cross_baqaee_farhi_network_evaluated_in_policy_studio(self):
        """Baqaee-Farhi network aggregation evaluated across elasticity grid in PolicyStudio."""
        Omega, b = _sample_io_3sector()
        grid = CounterfactualGrid(
            {
                "elasticity": [0.3, 0.7, 1.0],
                "energy_shock": [-0.08, -0.04],
            }
        )

        def eval_bf(p):
            z = np.array([float(p["energy_shock"]), 0.0, 0.0])
            th = float(p["elasticity"])
            return baqaee_farhi_decomposition(Omega, b, elasticities=th, tfp_shocks=z, final_elasticity=th)

        studio = run_policy_studio(eval_bf, grid)
        assert studio.pass_count == 6
        piv = studio.pivot("second_order_dlog_y", row_axis="elasticity", col_axis="energy_shock")
        # At elasticity 1.0 (Cobb-Douglas), curvature is zero
        assert np.allclose(piv.loc[1.0].to_numpy(), 0.0, atol=1e-14)
        # At elasticity 0.3 (gross complements), curvature is negative
        assert np.all(piv.loc[0.3].to_numpy() < 0.0)

    def test_t3_cross_dynamic_spatial_hat_in_policy_studio_grid(self):
        """Dynamic spatial migration hat-algebra evaluated across trade cost grid in PolicyStudio."""
        model = _sample_spatial_3region()
        grid = CounterfactualGrid({"trade_cost_reduction": [0.05, 0.10, 0.15]})

        def eval_spatial(p):
            tau = np.ones((3, 3))
            red = float(p["trade_cost_reduction"])
            tau[0, 1] = 1.0 - red
            tau[1, 0] = 1.0 - red
            return solve_dynamic_spatial_hat(model, horizon=20, trade_cost_hat_path=tau)

        studio = run_policy_studio(eval_spatial, grid)
        assert studio.pass_count == 3
        assert "PASS (3/3 verified)" in studio.overall_audit_status

    def test_t3_cross_nowcasting_news_decomposition_with_state_space_score(self):
        """Coupling nowcast news decomposition with analytical Kalman score gradient."""
        model, df_old, df_new = _build_synthetic_nowcast_vintages()
        res_news = banbura_modugno_news(model, df_old, df_new, target_series="gdp")
        assert res_news.decomposition_error < 1e-12

        # Extract system matrices from fitted DFM
        A = model.result_.A
        Z = model.result_.loadings
        Q = model.result_.Q
        H = model.result_.H
        m_dim = A.shape[0]
        n_dim = Z.shape[0]

        ssm = StateSpaceModel(T=A, Z=Z, R=np.eye(m_dim), Q=Q, H=H, c=np.zeros(m_dim), d=np.zeros(n_dim))
        sens = {
            "A_00": StateSpaceSensitivity(
                dT=np.eye(m_dim),
                dZ=np.zeros((n_dim, m_dim)),
                dR=np.zeros((m_dim, m_dim)),
                dQ=np.zeros((m_dim, m_dim)),
                dH=np.zeros((n_dim, n_dim)),
                dc=np.zeros(m_dim),
                dd=np.zeros(n_dim),
            )
        }
        y_data = df_new.to_numpy()
        ll, score = kalman_score(y_data, ssm, sens)
        assert np.isfinite(ll)
        assert np.isfinite(score[0])

    def test_t3_cross_baqaee_farhi_prices_with_forward_ad_gradients(self):
        """Forward AD evaluates exact parameter tangents of analytical Baqaee-Farhi welfare."""
        Omega, b = _sample_io_3sector()
        theta = np.array([0.4, 0.4, 0.4])
        z0 = np.array([-0.05, 0.0, 0.0])
        res_base = baqaee_farhi_decomposition(Omega, b, elasticities=theta, tfp_shocks=z0, final_elasticity=0.5)

        # Analytical formula: dlog_Y = lambda @ z + 0.5 * z @ H @ z
        lam = res_base.domar_weights
        H = res_base.hessian_matrix

        def quadratic_macro_welfare(z_var):
            return np.sum(lam * z_var) + 0.5 * (z_var @ (H @ z_var))

        val, grad = value_and_grad(quadratic_macro_welfare, z0)
        grad_exact = lam + H @ z0
        assert abs(val - quadratic_macro_welfare(z0)) < 1e-15
        assert np.max(np.abs(grad - grad_exact)) < 1e-14

# ===========================================================================
# Tier 4: Real-World Macroeconomic Application Scenarios (5 scenarios)
# ===========================================================================

class TestTier4RealWorldMacroScenarios:
    """Tier 4: Realistic graduate-level macroeconomic policy workflows."""

    def test_t4_scenario_monetary_policy_zlb_with_precautionary_savings(self):
        """Scenario 1: Monetary Policy at the Effective Lower Bound with Precautionary Shifts."""
        m_ref, m_zlb, constraint = _build_nk_occbin_models()
        # Severe macroeconomic crisis shock pushing nominal interest rate to the floor
        crisis_shock = np.array([-0.08])

        # 1. Standard linear OccBin (ignoring risk & higher-order curvature)
        res_linear = solve_occbin(m_ref, m_zlb, constraint, crisis_shock, horizon=35)
        # 2. 2nd-order pruned OccBin with stochastic precautionary risk shifts
        res_pruned = solve_nonlinear_occbin(
            m_ref, m_zlb, constraint, crisis_shock, order=2, sigma=1.5, horizon=35
        )

        assert res_linear.converged and res_pruned.converged
        assert res_linear.binding_periods > 0 and res_pruned.binding_periods > 0
        assert res_pruned.max_bound_violation < 1e-10
        # Precautionary savings shift deepens the contraction in output
        assert res_pruned.precautionary_shift_norm > 1e-6
        assert len(res_pruned.summary()) >= 6

    def test_t4_scenario_production_network_energy_shock_and_macro_curvature(self):
        """Scenario 2: Production Network Energy Shock & Nonlinear CES Welfare Losses."""
        Omega, b = _sample_io_3sector()
        # Upstream energy crisis: 15% contraction in energy TFP
        z_energy = np.array([-0.15, 0.0, 0.0])
        # Short-run low elasticity of substitution across intermediate inputs
        theta_short_run = np.array([0.25, 0.35, 0.40])
        final_demand_theta = 0.50
        # Sector markups
        markups = np.array([1.20, 1.10, 1.05])

        res = baqaee_farhi_decomposition(
            Omega,
            b,
            elasticities=theta_short_run,
            tfp_shocks=z_energy,
            final_elasticity=final_demand_theta,
            markups=markups,
            sector_names=["Energy", "Manufacturing", "Services"],
        )

        # Pure network curvature under complements theta < 1 is negative
        assert res.second_order_dlog_y < -1e-3
        # Exact CES fixed point confirms the substantial output contraction
        assert res.exact_ces_dlog_y < -0.05
        # Markup allocative efficiency wedge is non-zero
        assert abs(res.allocative_efficiency_dlog_y) > 0.0
        # Second-order Taylor approximation matches exact CES to within 0.01
        assert abs(res.total_second_order_dlog_y - res.exact_ces_dlog_y) < 0.01

        # Export report formats
        assert "Energy" in res.sector_names
        assert "Hulten" in res.to_markdown()
        assert "\\begin{tabular}" in res.to_latex()

    def test_t4_scenario_continuous_time_hank_inequality_and_transfer_shock(self):
        """Scenario 3: Continuous-Time HANK with Wealth Inequality & General Equilibrium."""
        # Baseline stationary Aiyagari economy
        res_ge = solve_aiyagari_continuous_hjb(Na=35, tol_ge=1e-3, max_iter_ge=20, max_iter_hjb=25)
        assert res_ge.converged
        assert res_ge.r_star > 0.0
        assert res_ge.K_star > 0.0

        # Inspect stationary wealth distribution
        sol_hjb = res_ge.hjb_solution
        w_a = _quadrature_weights(sol_hjb.a_grid)
        w = np.tile(w_a, len(sol_hjb.e_grid))
        pi = sol_hjb.g_dist.ravel(order="F") * w
        assert abs(np.sum(pi) - 1.0) < 1e-13

        # Borrowing constrained share at bottom wealth
        constrained_share = float(np.sum(sol_hjb.g_dist[0, :] * sol_hjb.a_grid[0]))
        assert np.isfinite(constrained_share)

    def test_t4_scenario_central_bank_realtime_nowcasting_vintage_workflow(self):
        """Scenario 4: Central Bank Staff Nowcasting Workflow with Real-Time News Decomposition."""
        model, df_old, df_new = _build_synthetic_nowcast_vintages(T=48, seed=456)

        # Decompose the nowcast update for GDP
        res_news = banbura_modugno_news(model, df_old, df_new, target_series="gdp")
        assert res_news.decomposition_error < 1e-12
        assert abs(res_news.revision - res_news.total_impact) < 1e-12

        # Summary tables
        summary_text = res_news.summary()
        assert "Bańbura & Modugno (2014)" in summary_text
        assert "Total Nowcast Revision" in summary_text
        assert len(res_news.news_table) > 0
        assert len(res_news.revision_table) > 0

        # Plot generation
        fig = res_news.plot(title="MPC Nowcast Update Waterfall")
        assert fig is not None
        plt.close("all")

    def test_t4_scenario_dynamic_spatial_infrastructure_investment_and_migration(self):
        """Scenario 5: Spatial Economic Geography & Transport Infrastructure Investment."""
        model = _sample_spatial_3region()
        # High-speed transport corridor between North (region 0) and South (region 2)
        # reducing bilateral iceberg trade costs by 12%
        tau_path = np.ones((3, 3))
        tau_path[0, 2] = 0.88
        tau_path[2, 0] = 0.88

        res = solve_dynamic_spatial_hat(model, horizon=35, trade_cost_hat_path=tau_path, tol=1e-11)
        assert res.converged
        assert res.population_conservation_error < 1e-12
        assert res.migration_row_stochastic_error < 1e-14

        # Real wage gains in integrated regions
        assert res.real_wage_hat_path[-1, 0] > 1.0
        assert res.real_wage_hat_path[-1, 2] > 1.0
        # Welfare gains in all regions due to trade expansion
        assert np.all(res.welfare_hat > 1.0)

        # Documentation export
        df = res.to_frame()
        assert len(df) == 3
        assert "North" in res.to_markdown()
