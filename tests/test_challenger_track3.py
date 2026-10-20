"""Empirical Adversarial Stress Suite for Track 3 (Milestone M5-1).

Targets:
1. Two-Asset HANK vs MIT shocks (tolerance < 1e-4).
2. Analytical Kalman Score gradients vs 5-point CFD (tolerance < 1e-5).
3. Pruned OccBin reduction and multi-period binding regimes (tolerance < 1e-12).
"""
import math
import numpy as np
import pandas as pd
import pytest
import scipy.linalg

from puremacro import dp
from puremacro.dp._egm_two_asset import TwoAssetEGMPlan, TwoAssetHouseholdSolution
from puremacro.dsge import load_mod, build_dynare
from puremacro.dsge._gradients import (
    ScoreDiagnosticsResult,
    build_state_space_sensitivities,
    compute_decision_rule_derivatives,
    kalman_score,
    log_posterior_and_gradient,
    solve_sylvester_generalized,
)
from puremacro.dsge.observation import make_state_space_from_varobs
from puremacro.state_space import kalman_filter
from puremacro.models.hank_sequence_space import (
    _two_asset_grid,
    _solve_two_asset_household_block,
    _two_asset_household_jacobians,
    _two_asset_transition,
    _two_asset_jacobians,
    dp_sequence_jacobian,
    dp_nonlinear_transition,
)


# ==============================================================================
# TARGET 1: TWO-ASSET HANK VS MIT SHOCKS
# ==============================================================================

@pytest.fixture(scope="module")
def two_asset_dp_model():
    """Calibrated two-asset household dynamic programming model."""
    m = dp.Model("two_asset_household")
    n_a = 20
    n_b = 20
    a_grid = _two_asset_grid(0.0, 40.0, n_a, curvature=2.0)
    b_grid = _two_asset_grid(0.0, 10.0, n_b, curvature=1.5)

    m.state("a", a_grid)
    m.state("b", b_grid)

    pi_s = np.array([[0.95, 0.05], [0.10, 0.90]])
    s_grid = np.array([0.5, 1.5])
    m.exogenous("z", dp.Markov(s_grid, pi_s))

    m.parameters(
        beta=0.98,
        gamma=1.0,
        r_b=0.005,
        r_a=0.0125,
        w=1.0,
        chi_0=1.0,
        chi_1=1.0,
        a_bar=0.25,
    )
    m.local("d = a(+1) - (1 + r_a) * a")
    m.local("chi = chi_0 / (1 + chi_1) * abs(d) * (abs(d) / (a + a_bar))^chi_1")
    m.local("c = (1 + r_b) * b + (1 + r_a) * a + w * z - a(+1) - b(+1) - chi")
    m.reward("crra(c, gamma)")
    m.subject_to("c > 0")
    m.aggregate(C="c", A="a", B="b", D="d", CHI="chi")
    return m


def test_t1_01_two_asset_egm_plan_stationary_solution(two_asset_dp_model):
    """Empirical check: TwoAssetEGMPlan solves stationary policy and distribution."""
    comp = two_asset_dp_model.check()
    plan = TwoAssetEGMPlan(comp)
    assert plan.n_a == 20
    assert plan.n_b == 20
    assert plan.illiquid_name == "a"
    assert plan.liquid_name == "b"

    params = comp.param_values()
    sol = plan.solve_household(params, tol=1e-8, max_iter=500)
    assert isinstance(sol, TwoAssetHouseholdSolution)
    assert sol.converged
    assert sol.c.shape == (20, 20, 2)
    assert np.all(sol.c > 0.0)

    dist = plan.distribution(sol)
    assert dist.shape == (20, 20, 2)
    assert np.all(dist >= 0.0)
    np.testing.assert_allclose(np.sum(dist), 1.0, atol=1e-10)


def test_t1_02_two_asset_internal_ssj_vs_mit_shock():
    """Empirical check: Low-level two-asset Fake-News SSJ vs non-linear MIT shocks (< 1e-4)."""
    hh = _solve_two_asset_household_block(
        beta=0.98,
        gamma=1.0,
        r_b_ss=0.005,
        r_a_ss=0.0125,
        w_ss=1.0,
        chi_0=1.0,
        chi_1=1.0,
        a_bar=0.25,
        n_a=20,
        n_b=20,
        a_max=40.0,
        b_max=10.0,
    )
    T = 24
    raw_J = _two_asset_household_jacobians(hh, T=T, inputs=["r_b", "r_a", "z"], outputs=["C", "A", "B", "D", "CHI"])

    # Test perturbation on r_b
    eps = 1e-5
    rho = 0.7
    shock_seq = eps * (rho ** np.arange(T))
    r_b_base = np.full(T, hh.r_b_ss)
    r_a_base = np.full(T, hh.r_a_ss)
    z_base = np.full(T, hh.w_ss)

    nl_up = _two_asset_transition(hh, r_b_base + shock_seq, r_a_base, z_base)
    nl_dn = _two_asset_transition(hh, r_b_base - shock_seq, r_a_base, z_base)

    for o in ["C", "A", "B"]:
        actual = 0.5 * (nl_up[o] - nl_dn[o])
        pred = raw_J[(o, "r_b")] @ shock_seq
        rel_err = float(np.linalg.norm(pred - actual) / np.linalg.norm(actual))
        assert rel_err < 1e-4, f"Output {o} relative error {rel_err:.2e} >= 1e-4"


def test_t1_03_two_asset_dp_sequence_jacobian_interface(two_asset_dp_model):
    """Adversarial probe: Test high-level dp_sequence_jacobian and HANKModel with 2-asset DP."""
    sol = two_asset_dp_model.solve(method="egm", tol=1e-8)
    T = 24
    try:
        J = dp_sequence_jacobian(sol, inputs=["r_b", "r_a", "w"], outputs=["C", "A", "B", "D", "CHI"], T=T)
    except ValueError as e:
        pytest.fail(f"dp_sequence_jacobian failed for 2-asset model: {e}")


# ==============================================================================
# TARGET 2: ANALYTICAL KALMAN SCORE GRADIENTS VS 5-POINT CENTRAL FD
# ==============================================================================

def eval_5point_cfd(mod_code, varobs, base_params, param_names, y_data, h=1e-6):
    """Compute 5-point central finite differences of the log-likelihood."""
    T_obs, n_obs = y_data.shape
    grad_cfd = np.zeros(len(param_names))

    def eval_ll_at_params(p_dict):
        sh_cov = np.array([[p_dict.get("stderr_e_a", 1.0) ** 2]])
        # Build state space for evaluated params
        m_eval = load_mod(
            mod_code,
            params={k: v for k, v in p_dict.items() if k != "stderr_e_a"},
        )
        ssm = make_state_space_from_varobs(m_eval, varobs, shock_cov=sh_cov)
        RQR = ssm.R @ ssm.Q @ ssm.R.T
        P0 = scipy.linalg.solve_discrete_lyapunov(ssm.T, RQR)
        res = kalman_filter(y_data, ssm, P0=P0)
        return float(res["loglik"])

    for i, p_name in enumerate(param_names):
        v0 = base_params[p_name]
        step = h * max(abs(v0), 1.0)

        p_p2 = dict(base_params); p_p2[p_name] = v0 + 2 * step
        p_p1 = dict(base_params); p_p1[p_name] = v0 + step
        p_m1 = dict(base_params); p_m1[p_name] = v0 - step
        p_m2 = dict(base_params); p_m2[p_name] = v0 - 2 * step

        f_p2 = eval_ll_at_params(p_p2)
        f_p1 = eval_ll_at_params(p_p1)
        f_m1 = eval_ll_at_params(p_m1)
        f_m2 = eval_ll_at_params(p_m2)

        grad_cfd[i] = (-f_p2 + 8 * f_p1 - 8 * f_m1 + f_m2) / (12 * step)

    return grad_cfd


def test_t2_01_kalman_score_3eq_model_multiple_seeds_and_lengths():
    """Adversarial probe: test 3-eq model over varying data lengths and noise seeds."""
    mod_code = """
    var y c a;
    varexo e_a;
    parameters beta sigma rho_a;
    beta = 0.99;
    sigma = 1.0;
    rho_a = 0.8;

    model;
      c = c(+1) - (1/sigma)*y;
      y = c + a;
      a = rho_a * a(-1) + e_a;
    end;

    steady_state_model;
      y = 0;
      c = 0;
      a = 0;
    end;
    """
    m = load_mod(mod_code)
    varobs = ["y"]
    param_names = ["rho_a", "sigma", "stderr_e_a"]
    base_params = {"beta": 0.99, "sigma": 1.0, "rho_a": 0.8, "stderr_e_a": 1.0}

    for seed, T_obs in [(101, 30), (202, 75), (303, 120)]:
        rng = np.random.default_rng(seed)
        y_sim = rng.normal(0.0, 0.4, size=(T_obs, 1))

        sens = build_state_space_sensitivities(m, varobs, param_names)
        ll_an, score_an = kalman_score(y_sim, sens.state_space, sens)
        score_cfd = eval_5point_cfd(mod_code, varobs, base_params, param_names, y_sim, h=1e-6)

        abs_err = np.abs(score_an - score_cfd)
        rel_err = abs_err / np.maximum(1.0, np.abs(score_cfd))

        max_rel = np.max(rel_err)
        assert max_rel < 1e-5, (
            f"Seed {seed}, T={T_obs}: max rel error {max_rel:.2e} >= 1e-5.\n"
            f"score_an={score_an}\nscore_cfd={score_cfd}\nrel_err={rel_err}"
        )


def test_t2_02_kalman_score_neoclassical_model_regular():
    """Verify analytical score matches 5-point CFD (< 1e-5) when model is non-singular."""
    mod_code = """
    var c k y;
    varexo e_a;
    parameters beta alpha delta;
    beta = 0.96;
    alpha = 0.33;
    delta = 0.10;

    model;
      1/c = beta * (1/c(+1)) * (alpha * y(+1)/k + 1 - delta);
      y = k(-1)^alpha;
      k = y - c + (1 - delta)*k(-1) + e_a;
    end;

    steady_state_model;
      k = ((1/beta - (1 - delta))/alpha)^(1/(alpha - 1));
      y = k^alpha;
      c = y - delta*k;
    end;
    """
    m = load_mod(mod_code)
    varobs = ["y"]
    param_names = ["alpha", "delta"]
    base_params = {"beta": 0.96, "alpha": 0.33, "delta": 0.10, "stderr_e_a": 1.0}

    rng = np.random.default_rng(456)
    T_obs = 50
    y_sim = rng.normal(1.5, 0.1, size=(T_obs, 1))

    sens = build_state_space_sensitivities(m, varobs, param_names)
    ll_an, score_an = kalman_score(y_sim, sens.state_space, sens)

    score_cfd = eval_5point_cfd(mod_code, varobs, base_params, param_names, y_sim, h=1e-6)
    abs_err = np.abs(score_an - score_cfd)
    rel_err = abs_err / np.maximum(1.0, np.abs(score_cfd))

    max_rel = np.max(rel_err)
    assert max_rel < 1e-5, (
        f"Neoclassical model: max rel error {max_rel:.2e} >= 1e-5.\n"
        f"score_an={score_an}\nscore_cfd={score_cfd}\nrel_err={rel_err}"
    )


# ==============================================================================
# TARGET 3: PRUNED OCCBIN ADVERSARIAL INTEGRITY
# ==============================================================================

def test_t3_01_solve_pruned_occbin_exported_in_dsge():
    """Adversarial check: solve_pruned_occbin must be importable from puremacro.dsge and puremacro.dsge.occbin."""
    import puremacro.dsge as dsge
    import puremacro.dsge.occbin as occbin

    assert hasattr(occbin, "solve_pruned_occbin"), (
        "CRITICAL BUG: solve_pruned_occbin is missing from puremacro.dsge.occbin!"
    )
    assert hasattr(dsge, "solve_pruned_occbin"), (
        "CRITICAL BUG: solve_pruned_occbin is missing from puremacro.dsge exports!"
    )


def test_t3_02_solve_pruned_occbin_linear_reduction_and_binding():
    """Adversarial check: reduction to linear OccBin when sigma=0 and multi-period binding regimes."""
    from puremacro.dsge.occbin import OccBinConstraint, solve_occbin
    try:
        from puremacro.dsge.occbin import solve_pruned_occbin
    except ImportError:
        pytest.fail("Cannot test solve_pruned_occbin: symbol does not exist")

    params = {"beta": 0.99, "sigma": 1.0, "kappa": 0.1, "phi_pi": 1.5, "phi_y": 0.125, "rho_g": 0.8, "r_ss": 0.01}
    variables = ["y", "pi", "r", "g"]
    shocks = ["eps_r", "eps_g"]
    steady_state = {v: 0.0 for v in variables}

    def nk_ref(lead, curr, lag, shocks_v, p):
        return [
            curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.g,
            curr.pi - p.beta * lead.pi - p.kappa * curr.y,
            curr.r - p.phi_pi * curr.pi - p.phi_y * curr.y - shocks_v.eps_r,
            curr.g - p.rho_g * lag.g - shocks_v.eps_g,
        ]

    def nk_cons(lead, curr, lag, shocks_v, p):
        return [
            curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.g,
            curr.pi - p.beta * lead.pi - p.kappa * curr.y,
            curr.r - (-p.r_ss),
            curr.g - p.rho_g * lag.g - shocks_v.eps_g,
        ]

    ref_m = build_dynare(nk_ref, variables=variables, shocks=shocks, params=params, steady_state=steady_state)
    cons_m = build_dynare(nk_cons, variables=variables, shocks=shocks, params=params, steady_state=steady_state, check_steady_state=False, strict=False)
    constraint = OccBinConstraint(variable="r", threshold=-params["r_ss"], operator="<")

    horizon = 30
    shocks_mat = np.zeros((horizon, 2))
    shocks_mat[0, 1] = -0.05  # Severe demand shock causing multi-period binding

    res_lin = solve_occbin(ref_m, cons_m, constraint, shocks_mat, horizon=horizon)
    res_pruned = solve_pruned_occbin(ref_m, cons_m, constraint, shocks_mat, order=2, sigma=0.0, horizon=horizon)

    diff = np.max(np.abs(res_pruned.simulated_path.to_numpy() - res_lin.simulated_path.to_numpy()))
    assert diff < 1e-12, f"sigma=0 linear reduction error {diff:.2e} >= 1e-12"
    assert res_pruned.binding_periods > 1, f"Expected multiple binding periods, got {res_pruned.binding_periods}"
