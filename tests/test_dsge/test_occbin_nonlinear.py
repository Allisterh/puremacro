"""Tests for puremacro.dsge.occbin_nonlinear (2nd/3rd-order pruned OccBin with risk shifts)."""
from __future__ import annotations

import numpy as np

from puremacro.dsge import OccBinConstraint, build_dynare, solve_nonlinear_occbin, solve_occbin


def _build_nonlinear_nk_zlb():
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
    return m_ref, m_zlb, OccBinConstraint(variable="i", threshold=-0.01, operator="<")


def test_nonlinear_occbin_slack_matches_pruned_and_order1_matches_linear():
    m_ref, m_zlb, constraint = _build_nonlinear_nk_zlb()

    # 1. Small positive shock: ZLB never binds -> must match unconstrained order-2 and order-3 pruned path to 1e-13
    small_shock = np.array([0.005])
    res_slack = solve_nonlinear_occbin(
        m_ref, m_zlb, constraint, small_shock, order=2, sigma=1.0, horizon=30
    )
    assert res_slack.converged
    assert res_slack.binding_periods == 0
    assert np.max(np.abs(res_slack.simulated_path.to_numpy() - res_slack.unconstrained_pruned_path.to_numpy())) < 1e-13

    res_slack_o3 = solve_nonlinear_occbin(
        m_ref, m_zlb, constraint, small_shock, order=3, sigma=1.0, horizon=30
    )
    assert res_slack_o3.converged
    assert res_slack_o3.binding_periods == 0
    assert np.max(np.abs(res_slack_o3.simulated_path.to_numpy() - res_slack_o3.unconstrained_pruned_path.to_numpy())) <= 1e-13

    # 2. Order=1 matches linear solve_occbin bit-for-bit on a binding ZLB shock
    big_shock = np.array([-0.06])
    res_o1 = solve_nonlinear_occbin(
        m_ref, m_zlb, constraint, big_shock, order=1, sigma=1.0, horizon=35
    )
    lin_occ = solve_occbin(m_ref, m_zlb, constraint, big_shock, horizon=35)
    assert res_o1.converged and lin_occ.converged
    assert res_o1.binding_periods > 0
    assert res_o1.regimes == lin_occ.regimes
    assert np.max(np.abs(res_o1.simulated_path.to_numpy() - lin_occ.simulated_path.to_numpy())) < 1e-13

    # 3. Order=2 and Order=3 binding ZLB solve: bound holds and curvature/risk shift is non-zero
    res_o2 = solve_nonlinear_occbin(
        m_ref, m_zlb, constraint, big_shock, order=2, sigma=1.0, horizon=35
    )
    res_o3 = solve_nonlinear_occbin(
        m_ref, m_zlb, constraint, big_shock, order=3, sigma=1.0, horizon=35
    )
    assert res_o2.converged and res_o3.converged
    assert res_o2.binding_periods > 0 and res_o3.binding_periods > 0
    assert res_o2.max_bound_violation < 1e-12
    assert res_o3.max_bound_violation < 1e-12
    assert res_o2.precautionary_shift_norm > 1e-6
    assert "Approximation Order" in res_o2.to_markdown() or len(res_o2.summary()) >= 6


def test_nonlinear_occbin_sigma_scaling_and_reporting():
    m_ref, m_zlb, constraint = _build_nonlinear_nk_zlb()
    big_shock = np.array([-0.06])

    res_sig0 = solve_nonlinear_occbin(
        m_ref, m_zlb, constraint, big_shock, order=2, sigma=0.0, horizon=35
    )
    res_sig2 = solve_nonlinear_occbin(
        m_ref, m_zlb, constraint, big_shock, order=2, sigma=2.0, horizon=35
    )
    assert res_sig0.converged and res_sig2.converged
    # Stochastic risk scaling (sigma=2 vs sigma=0) shifts the trajectory via 0.5*G_sigmasigma*sigma^2
    diff_sigma = np.max(np.abs(res_sig2.simulated_path.to_numpy() - res_sig0.simulated_path.to_numpy()))
    assert diff_sigma > 1e-6

    df = res_sig2.to_frame()
    assert list(df.columns) == ["y", "pi", "i", "a"]
    assert np.allclose(res_sig2["i"].to_numpy(), df["i"].to_numpy())
    assert "\\begin{tabular}" in res_sig2.to_latex()
    assert "#table(" in res_sig2.to_typst()


def test_nonlinear_occbin_multi_period_shocks():
    m_ref, m_zlb, constraint = _build_nonlinear_nk_zlb()
    horizon = 35
    shocks_mat = np.zeros((horizon, 1), dtype=np.float64)
    shocks_mat[0, 0] = -0.04
    shocks_mat[1, 0] = -0.03
    shocks_mat[2, 0] = -0.02

    res_multi = solve_nonlinear_occbin(
        m_ref, m_zlb, constraint, shocks_mat, order=2, sigma=1.0, horizon=horizon
    )
    assert res_multi.converged
    assert res_multi.binding_periods > 0
    assert res_multi.max_bound_violation <= 1e-10


