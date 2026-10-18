"""Tests for the dp.Model <-> Sequence-Space Jacobian Bridge, Schur Sylvester
Solvers, Tempered Particle Filter, and Fast NUTS Score Gradients.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dataclasses import dataclass
import scipy.linalg

from puremacro import dp
from puremacro.dp.model import Model as DPModel
from puremacro.dsge._sylvester import (
    solve_generalized_sylvester_kronecker,
    solve_generalized_sylvester_schur,
    solve_order1_sylvester,
    solve_order3_sylvester_kronecker,
    solve_order3_sylvester_schur,
)
from puremacro.dsge.hank import (
    dp_nonlinear_transition,
    dp_sequence_jacobian,
    load_hank_mod,
    solve_hank_bridge,
)
from puremacro.dsge.nuts import (
    _log_prob_and_numerical_grad,
    _numerical_gradient,
    nuts_sample,
)
from puremacro.dsge.particle_filter import (
    particle_filter,
    tempered_particle_filter,
)
from puremacro.dsge.pruning import canonical_growth_2nd_order
from puremacro.state_space import StateSpaceModel, kalman_filter


# ---------------------------------------------------------------------------
# 1. puremacro.dp.Model <-> Sequence-Space Jacobian Bridge (ABRS 2021)
# ---------------------------------------------------------------------------

def _make_household_dp_model(with_equilibrium_price: bool = False) -> DPModel:
    m = DPModel("aiyagari_hh")
    a_grid = np.geomspace(1e-4, 30.0 + 1e-4, 60) - 1e-4
    m.state("a", a_grid)
    m.exogenous("z", dp.AR1(rho=0.85, sigma=0.15, n=3))
    if with_equilibrium_price:
        m.parameters(beta=0.96, gamma=2.0, alpha=0.36, delta=0.08)
        m.prices(r=(0.01, 0.041))
        m.local("w = (1 - alpha) * (alpha / (r + delta))^(alpha / (1 - alpha))")
        m.local("c = (1 + r) * a + w * exp(z) - a(+1)")
        m.reward("crra(c, gamma)")
        m.subject_to("c > 0")
        m.aggregate(C="c", A="a", L="exp(z)")
        m.clear("A - L * (alpha / (r + delta))^(1 / (1 - alpha))")
    else:
        m.parameters(beta=0.96, gamma=2.0, r=0.02, w=1.0)
        m.local("c = (1 + r) * a + w * exp(z) - a(+1)")
        m.reward("crra(c, gamma)")
        m.subject_to("c > 0")
        m.aggregate(C="c", A="a")
    return m


def test_dp_sequence_jacobian_budget_identities():
    """Verify exact intertemporal budget identities on dp_sequence_jacobian."""
    m = _make_household_dp_model(with_equilibrium_price=False)
    sol = m.solve(method="egm", tol=1e-12)
    T = 35
    J = dp_sequence_jacobian(sol, inputs=["r", "w"], outputs=["C", "A"], T=T)

    assert set(J.keys()) == {"C", "A"}
    assert set(J["C"].keys()) == {"r", "w"}
    assert J["C"]["r"].shape == (T, T)
    assert J["A"]["w"].shape == (T, T)

    r_ss = 0.02
    D_ss = sol.distribution
    Z_ss = float(np.sum(D_ss * sol.policy("exp(z)")))
    A_ss = float(np.sum(D_ss * sol.policy("a(+1)")))

    # 1. Wage shock budget identity: dC_t + dA_t - (1 + r_ss) dA_{t-1} == Z_ss * I_{t,s}
    J_C_w = J["C"]["w"]
    J_A_w = J["A"]["w"]
    J_A_w_lag = np.vstack([np.zeros((1, T)), J_A_w[:-1, :]])
    lhs_w = J_C_w + J_A_w - (1.0 + r_ss) * J_A_w_lag
    rhs_w = Z_ss * np.eye(T)
    np.testing.assert_allclose(lhs_w, rhs_w, atol=1e-11)

    # 2. Interest-rate shock budget identity: dC_t + dA_t - (1 + r_ss) dA_{t-1} == A_ss * I_{t,s}
    J_C_r = J["C"]["r"]
    J_A_r = J["A"]["r"]
    J_A_r_lag = np.vstack([np.zeros((1, T)), J_A_r[:-1, :]])
    lhs_r = J_C_r + J_A_r - (1.0 + r_ss) * J_A_r_lag
    rhs_r = A_ss * np.eye(T)
    np.testing.assert_allclose(lhs_r, rhs_r, atol=1e-11)


def test_dp_sequence_jacobian_matches_nonlinear_transition():
    """First-order Jacobian projection J @ dX matches nonlinear transition path."""
    m = _make_household_dp_model(with_equilibrium_price=True)
    T = 25
    J = dp_sequence_jacobian(m, inputs=["r", "w"], outputs=["C", "A"], T=T)

    eps = 1e-5
    t_grid = np.arange(T)
    dr = eps * (0.75 ** t_grid)
    dw = -0.5 * eps * (0.65 ** t_grid)

    nl_pos = dp_nonlinear_transition(m, {"r": dr, "w": dw}, outputs=["C", "A"])
    nl_neg = dp_nonlinear_transition(m, {"r": -dr, "w": -dw}, outputs=["C", "A"])

    dC_nl = 0.5 * (nl_pos["C"] - nl_neg["C"])
    dA_nl = 0.5 * (nl_pos["A"] - nl_neg["A"])

    dC_lin = J["C"]["r"] @ dr + J["C"]["w"] @ dw
    dA_lin = J["A"]["r"] @ dr + J["A"]["w"] @ dw

    np.testing.assert_allclose(dC_lin, dC_nl, rtol=1e-4, atol=1e-9)
    np.testing.assert_allclose(dA_lin, dA_nl, rtol=1e-4, atol=1e-9)


def test_hank_bridge_with_dp_household_model():
    """HANKModel and solve_hank_bridge accept household_model=dp.Model."""
    m_hh = _make_household_dp_model(with_equilibrium_price=False)
    mod_text = """
    var Y C pi i r;
    varexo eps_m;
    parameters beta gamma phi_pi kappa r_ss;
    beta = 0.96;
    gamma = 2.0;
    phi_pi = 1.5;
    kappa = 0.1;
    r_ss = 0.02;
    model;
        Y = C;
        pi = beta * pi(+1) + kappa * Y;
        i = phi_pi * pi + eps_m;
        r = i - pi(+1);
    end;
    hetagent_block;
        model = hank_dp;
        assets = 1;
    end;
    """
    hank_m = load_hank_mod(mod_text, household_model=m_hh)
    jacs = hank_m.compute_jacobians(T=20)
    assert "J_C_r" in jacs and "J_C_Y" in jacs
    assert jacs["J_C_r"].shape == (20, 20)

    res = solve_hank_bridge(mod_text, shock="eps_m", magnitude=-0.0025, horizon=20, household_model=m_hh)
    assert res.converged
    np.testing.assert_allclose(res.transition_paths["Y"], res.transition_paths["C"], atol=1e-12)
    assert res.transition_paths["Y"].iloc[0] > 0.0  # Monetary easing expands output


# ---------------------------------------------------------------------------
# 2. Real Schur / Bartels-Stewart Generalized Sylvester Solvers
# ---------------------------------------------------------------------------

def test_schur_sylvester_order2_and_order3_with_complex_schur_blocks():
    """Verify Real Schur Bartels-Stewart solver with 2x2 complex eigenvalue blocks."""
    rng = np.random.default_rng(2026)
    N, n_x = 5, 4
    A_hat = 3.5 * np.eye(N) + 0.15 * rng.standard_normal((N, N))
    A_plus = 0.25 * rng.standard_normal((N, N))

    # Construct h_x with oscillatory 2x2 blocks (complex-conjugate eigenvalues) + real blocks
    theta = 0.6
    rot = 0.8 * np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    h_x_block = np.block([
        [rot, 0.15 * rng.standard_normal((2, 2))],
        [np.zeros((2, 2)), np.array([[0.7, 0.2], [-0.15, 0.65]])],
    ])
    Q_orth, _ = np.linalg.qr(rng.standard_normal((n_x, n_x)))
    h_x = Q_orth @ h_x_block @ Q_orth.T

    # Order 1: A_hat X + A_+ X h_x = -K_x
    K_x = rng.standard_normal((N, n_x))
    X1 = solve_order1_sylvester(A_hat, A_plus, h_x, K_x)
    res1 = A_hat @ X1 + A_plus @ X1 @ h_x + K_x
    assert np.max(np.abs(res1)) < 1e-13

    # Order 2: A_hat X + A_+ X (h_x kron h_x) = -K_xx
    K_xx = rng.standard_normal((N, n_x**2))
    X2 = solve_generalized_sylvester_schur(A_hat, A_plus, h_x, K_xx)
    X2_alias = solve_generalized_sylvester_kronecker(A_hat, A_plus, h_x, K_xx)
    np.testing.assert_allclose(X2, X2_alias, atol=1e-15)

    kron2 = np.kron(h_x, h_x)
    res2 = A_hat @ X2 + A_plus @ X2 @ kron2 + K_xx
    assert np.max(np.abs(res2)) < 1e-12

    # Order 3: A_hat X + A_+ X (h_x kron h_x kron h_x) = -K_xxx
    K_xxx = rng.standard_normal((N, n_x**3))
    X3 = solve_order3_sylvester_schur(A_hat, A_plus, h_x, K_xxx)
    X3_alias = solve_order3_sylvester_kronecker(A_hat, A_plus, h_x, K_xxx)
    np.testing.assert_allclose(X3, X3_alias, atol=1e-15)

    kron3 = np.kron(kron2, h_x)
    res3 = A_hat @ X3 + A_plus @ X3 @ kron3 + K_xxx
    assert np.max(np.abs(res3)) < 1e-12


# ---------------------------------------------------------------------------
# 3. Tempered Particle Filter (Herbst & Schorfheide 2019)
# ---------------------------------------------------------------------------

@dataclass
class _MockLinearDSGE:
    G: np.ndarray
    N: np.ndarray
    F: np.ndarray
    L: np.ndarray
    states: list[str]
    controls: list[str]
    shocks: list[str]
    steady_state: dict[str, float]
    _shock_cov: np.ndarray


def test_tempered_particle_filter_kalman_parity_and_stages():
    """Verify TPF matches exact Kalman log-likelihood on a linear Gaussian state-space model."""
    rng = np.random.default_rng(123)
    T = 30
    G = np.array([[0.75, 0.0], [0.0, 0.65]])
    N_mat = np.eye(2)
    Q = np.diag([0.04**2, 0.03**2])
    model = _MockLinearDSGE(
        G=G,
        N=N_mat,
        F=np.zeros((0, 2)),
        L=np.zeros((0, 2)),
        states=["x1", "x2"],
        controls=[],
        shocks=["e1", "e2"],
        steady_state={"x1": 0.0, "x2": 0.0},
        _shock_cov=Q,
    )

    meas_sd = 0.015
    H = np.eye(2) * (meas_sd**2)
    x = np.zeros(2)
    rows = []
    for _ in range(T):
        e = rng.multivariate_normal(np.zeros(2), Q)
        x = G @ x + N_mat @ e
        y_obs = x + rng.multivariate_normal(np.zeros(2), H)
        rows.append(y_obs)
    df = pd.DataFrame(rows, columns=["x1", "x2"])

    P_stat = scipy.linalg.solve_discrete_lyapunov(G, N_mat @ Q @ N_mat.T)
    P0 = G @ P_stat @ G.T + N_mat @ Q @ N_mat.T
    ssm = StateSpaceModel(T=G, R=N_mat, Q=Q, Z=np.eye(2), H=H)
    ll_kalman = float(kalman_filter(df.to_numpy(), ssm, a0=np.zeros(2), P0=P0)["loglik"])

    res_tpf = tempered_particle_filter(
        model,
        df,
        observed_vars=["x1", "x2"],
        n_particles=1500,
        target_ineff=2.0,
        max_stages=8,
        n_mh_steps=2,
        measurement_error={"x1": meas_sd, "x2": meas_sd},
        seed=77,
    )

    assert np.isfinite(res_tpf.log_likelihood)
    assert abs(res_tpf.log_likelihood - ll_kalman) / abs(ll_kalman) < 0.03
    assert res_tpf.n_tempering_stages is not None
    assert len(res_tpf.n_tempering_stages) == T
    assert int(res_tpf.n_tempering_stages.max()) >= 2  # Tight measurement error triggers >1 stage
    assert np.all(res_tpf.effective_sample_size > 0.0)
    assert res_tpf.filtered_states.shape == (T, 2)

    # Also test dispatch via particle_filter(..., method="tempered")
    res_dispatch = particle_filter(
        model,
        df,
        observed_vars=["x1", "x2"],
        n_particles=500,
        method="tempered",
        measurement_error={"x1": meas_sd, "x2": meas_sd},
        seed=77,
    )
    assert res_dispatch.method == "tempered"
    assert res_dispatch.n_tempering_stages is not None


def test_tempered_particle_filter_nonlinear_pruned_model():
    """Verify TPF runs cleanly on a 2nd-order PrunedDSGESolution."""
    pruned = canonical_growth_2nd_order()
    sim = pruned.simulate(periods=20, seed=99)
    obs_vars = list(pruned.state_names[:1]) + list(pruned.control_names[:1])
    df = pd.concat([sim.states, sim.controls], axis=1)[obs_vars].copy()
    for v in obs_vars:
        if pruned.steady_state is not None and v in pruned.steady_state:
            df[v] = df[v] + float(pruned.steady_state[v])

    res = tempered_particle_filter(
        pruned,
        df,
        observed_vars=obs_vars,
        n_particles=600,
        target_ineff=2.0,
        measurement_error={v: 0.02 for v in obs_vars},
        seed=11,
    )
    assert np.isfinite(res.log_likelihood)
    assert res.filtered_states.shape[0] == 20


# ---------------------------------------------------------------------------
# 4. Fast Numerical & Custom Score Gradients for DSGE NUTS
# ---------------------------------------------------------------------------

def test_numerical_gradient_forward_central_complex_and_boundary_fallback():
    """Verify _numerical_gradient accuracy, evaluation count, and boundary fallback."""
    mu = np.array([1.5, -0.8, 0.25])
    prec = np.array([[2.0, 0.3, 0.0], [0.3, 1.5, -0.2], [0.0, -0.2, 3.0]])
    calls = {"n": 0}

    def log_prob(x: np.ndarray) -> float:
        calls["n"] += 1
        d = x - mu
        return -0.5 * float(d @ prec @ d)

    def exact_grad(x: np.ndarray) -> np.ndarray:
        return -(prec @ (x - mu))

    x0 = np.array([1.2, -0.5, 0.4])
    g_true = exact_grad(x0)

    # 1. Forward difference reusing cached fx uses exactly d = 3 calls
    fx0 = log_prob(x0)
    calls["n"] = 0
    g_fwd = _numerical_gradient(log_prob, x0, eps=1e-6, fx=fx0, method="forward")
    assert calls["n"] == 3
    np.testing.assert_allclose(g_fwd, g_true, rtol=1e-4, atol=1e-4)

    # 2. Central difference
    g_cen = _numerical_gradient(log_prob, x0, eps=1e-5, fx=fx0, method="central")
    np.testing.assert_allclose(g_cen, g_true, rtol=1e-8, atol=1e-8)

    # 3. Complex-step differentiation
    g_cplx = _numerical_gradient(
        lambda z: -0.5 * np.sum((z - mu) * (prec @ (z - mu))),
        x0,
        eps=1e-8,
        method="complex",
    )
    np.testing.assert_allclose(g_cplx, g_true, rtol=1e-12, atol=1e-12)

    # 4. Vectorized batch mode
    def log_prob_batch(X: np.ndarray) -> np.ndarray:
        D = X - mu[None, :]
        return -0.5 * np.sum((D @ prec) * D, axis=1)

    g_vec = _numerical_gradient(log_prob_batch, x0, eps=1e-5, method="central", vectorized=True)
    np.testing.assert_allclose(g_vec, g_true, rtol=1e-8, atol=1e-8)

    # 5. Boundary fallback when x is within 1e-6 of a hard lower bound
    bounds = [(1.2 - 1e-7, 5.0), (-5.0, 5.0), (-5.0, 0.4 + 1e-7)]

    def bounded_log_prob(x: np.ndarray) -> float:
        for i, (lo, hi) in enumerate(bounds):
            if x[i] < lo or x[i] > hi:
                return -np.inf
        return log_prob(x)

    lp_b, g_b = _log_prob_and_numerical_grad(bounded_log_prob, x0, eps=1e-4, bounds=bounds)
    assert np.isfinite(lp_b)
    assert np.all(np.isfinite(g_b))
    np.testing.assert_allclose(g_b, g_true, rtol=1e-3, atol=1e-3)


def test_nuts_sample_with_scalar_log_prob_and_custom_grad():
    """Verify nuts_sample supports scalar log_post + grad_log_post_fn or auto numerical grad."""
    mean_true = np.array([0.5, -0.5])
    inv_cov = np.diag([4.0, 2.0])

    def log_post_scalar(th: np.ndarray) -> float:
        d = th - mean_true
        return -0.5 * float(d @ inv_cov @ d)

    def grad_fn(th: np.ndarray) -> np.ndarray:
        return -(inv_cov @ (th - mean_true))

    res_custom = nuts_sample(
        log_post_scalar,
        init_params=np.zeros(2),
        n_draws=120,
        n_chains=2,
        warmup=80,
        grad_log_post_fn=grad_fn,
        seed=42,
    )
    post_mean = res_custom.draws.reshape(-1, 2).mean(axis=0)
    np.testing.assert_allclose(post_mean, mean_true, atol=0.15)

    res_auto = nuts_sample(
        log_post_scalar,
        init_params=np.zeros(2),
        n_draws=80,
        n_chains=1,
        warmup=60,
        grad_method="forward",
        seed=42,
    )
    assert np.all(np.isfinite(res_auto.draws))
