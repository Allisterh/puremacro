"""Tests for puremacro.dsge.ssj_dag and puremacro.dsge.hank_estimate."""
from __future__ import annotations

import numpy as np
import pandas as pd

from puremacro.dsge import (
    BlockDAG,
    HetBlock,
    SimpleBlock,
    SolvedBlock,
    estimate_hank_irf_matching,
    estimate_hank_whittle,
    hank_loglik_whittle,
)


def _build_nk_ssj_dag():
    """Build a 3-equation New Keynesian / HANK-limit DAG with Euler, NKPC, and Taylor rule."""
    # Block 1: Monetary policy rule i_t = r_ss + phi_pi * pi_t + eps_m_t
    taylor = SimpleBlock(
        "taylor",
        inputs=("pi", "eps_m"),
        outputs=("i",),
        fn=lambda v, p: {"i": p["r_ss"] + p["phi_pi"] * v["pi"] + v["eps_m"]},
    )

    # Block 2: Real rate r_t = i_t - pi_{t+1}
    fisher = SimpleBlock(
        "fisher",
        inputs=("i", "pi"),
        outputs=("r",),
        fn=lambda v, p: {"r": v["i"] - v["pi"](+1)},
    )

    # Block 3: Phillips curve residual: pi_t - beta * pi_{t+1} - kappa * (C_t - C_ss)
    nkpc = SimpleBlock(
        "nkpc",
        inputs=("pi", "C"),
        outputs=("nkpc_res",),
        fn=lambda v, p: {"nkpc_res": v["pi"] - p["beta"] * v["pi"](+1) - p["kappa"] * (v["C"] - p["C_ss"])},
    )

    # Block 4: Euler / IS equation residual: C_t - C_{t+1} + (1/sigma)*(r_t - r_ss)
    euler = SimpleBlock(
        "euler",
        inputs=("C", "r"),
        outputs=("euler_res",),
        fn=lambda v, p: {
            "euler_res": (v["C"] - v["C"](+1)) + (1.0 / p["sigma"]) * (v["r"] - p["r_ss"])
        },
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


def test_ssj_dag_analytical_irf_and_nonlinear_transition():
    dag, ss, params = _build_nk_ssj_dag()
    T = 40
    rho_m = 0.6
    eps_path = 0.01 * (rho_m ** np.arange(T))

    irf_res = dag.impulse_responses(ss, params, {"eps_m": eps_path}, T=T, levels=False)
    assert irf_res.converged
    assert irf_res.residual_norm < 1e-12

    # Analytical solution of the 3-equation NK model with AR(1) monetary shock rho_m:
    # [sigma*(1 - rho_m)*(1 - beta*rho_m) + kappa*(phi_pi - rho_m)] * pi_t = -kappa * eps_m_t
    denom = (
        params["sigma"] * (1.0 - rho_m) * (1.0 - params["beta"] * rho_m)
        + params["kappa"] * (params["phi_pi"] - rho_m)
    )
    pi_analytic = -(params["kappa"] / denom) * eps_path
    c_analytic = ((1.0 - params["beta"] * rho_m) / params["kappa"]) * pi_analytic

    # Interior horizon agreement (before finite-T truncation tail)
    assert np.max(np.abs(irf_res.paths["pi"].to_numpy()[:20] - pi_analytic[:20])) < 1e-9
    assert np.max(np.abs(irf_res.paths["C"].to_numpy()[:20] - c_analytic[:20])) < 1e-9

    # Nonlinear transition solver on the DAG must converge in 1 step for linear blocks
    nl_res = dag.solve_nonlinear_transition(ss, params, {"eps_m": eps_path}, T=T, tol=1e-12)
    assert nl_res.converged
    assert nl_res.residual_norm < 1e-12
    assert np.max(np.abs(nl_res.paths["C"].to_numpy() - (1.0 + irf_res.paths["C"].to_numpy()))) < 1e-12
    assert len(nl_res.summary()) >= 5
    assert "C" in nl_res.to_markdown()


def test_het_block_and_solved_block_integration():
    T = 20
    # Simple SolvedBlock resolving y = 2 * x
    inner = SimpleBlock("inner", inputs=("u", "x"), outputs=("res",), fn=lambda v, p: {"res": v["u"] - 2.0 * v["x"]})
    solved = SolvedBlock("solved_u", unknowns=("u",), targets=("res",), inner_block=inner)
    ss = {"u": 0.0, "x": 0.0, "res": 0.0, "c_hh": 0.0, "mkt": 0.0}
    J_s = solved.jacobian(ss, {}, T)
    assert np.allclose(J_s["u"]["x"], 2.0 * np.eye(T), atol=1e-14)

    # HetBlock with known MPC matrix
    mpc = 0.35
    J_hh = {"c_hh": {"u": mpc * np.eye(T)}}
    hh = HetBlock("hh", inputs=("u",), outputs=("c_hh",), jacobian_fn=J_hh)
    mkt = SimpleBlock("mkt", inputs=("c_hh", "x"), outputs=("mkt",), fn=lambda v, p: {"mkt": v["c_hh"] - v["x"]})
    dag = BlockDAG([solved, hh, mkt], unknowns=("x",), exogenous=("eps",), targets=("mkt",))
    assert len(dag.blocks) == 3


def test_whittle_loglik_white_noise_exact_parseval_and_estimation():
    rng = np.random.default_rng(42)
    T_obs = 128
    sigma2 = 0.04
    y = rng.normal(0.0, np.sqrt(sigma2), size=(T_obs, 1))
    M_wn = np.zeros((16, 1, 1))
    M_wn[0, 0, 0] = 1.0

    ll_whittle = hank_loglik_whittle(M_wn, y, np.array([[sigma2]]), demean=False)
    ll_exact = -0.5 * T_obs * np.log(2.0 * np.pi * sigma2) - 0.5 * float(np.sum(y ** 2)) / sigma2
    assert abs(ll_whittle - ll_exact) < 1e-10

    # Estimate AR(1) persistence rho via Whittle MLE and IRF matching
    rho_true = 0.72
    T_sim = 240
    eps = rng.normal(0.0, 0.02, size=T_sim)
    y_ar = np.zeros(T_sim)
    for t in range(1, T_sim):
        y_ar[t] = rho_true * y_ar[t - 1] + eps[t]

    def ma_builder(p):
        rho = p["rho"]
        M = (rho ** np.arange(64)).reshape(64, 1, 1)
        return M, np.array([[0.02 ** 2]])

    res_w = estimate_hank_whittle(
        ma_builder, y_ar, init_params={"rho": 0.4}, bounds={"rho": (0.05, 0.95)}
    )
    assert res_w.converged
    assert abs(res_w.params["rho"] - rho_true) < 0.12
    # Positive definiteness of Whittle covariance matrix
    eig_w = np.linalg.eigvalsh(res_w.vcov)
    assert np.all(eig_w > 0.0)
    assert np.all(np.diag(res_w.vcov) > 0.0)

    # IRF matching recovery
    H = 15
    target_df = pd.DataFrame({"y": rho_true ** np.arange(H)})
    res_irf = estimate_hank_irf_matching(
        lambda p, h: {"y": p["rho"] ** np.arange(h)},
        target_df,
        init_params={"rho": 0.3},
        bounds={"rho": (0.01, 0.98)},
    )
    assert res_irf.converged
    assert abs(res_irf.params["rho"] - rho_true) < 1e-5
    # Positive definiteness of IRF matching covariance matrix
    eig_irf = np.linalg.eigvalsh(res_irf.vcov)
    assert np.all(eig_irf > 0.0)
    assert np.all(np.diag(res_irf.vcov) > 0.0)


def test_ssj_dag_genuinely_nonlinear_transition():
    beta = 0.99
    r_ss = 1.0 / beta - 1.0
    params = {"beta": beta, "sigma": 1.0, "kappa": 0.15, "phi_pi": 1.5, "r_ss": r_ss, "C_ss": 1.0}
    ss = {"C": 1.0, "pi": 0.0, "i": r_ss, "r": r_ss, "eps_m": 0.0, "euler_res": 0.0, "nkpc_res": 0.0}

    # Nonlinear blocks:
    # 1. Nonlinear Taylor rule: (1 + r_ss) * (1 + pi)^phi_pi * exp(eps_m) - 1
    taylor = SimpleBlock(
        "taylor",
        inputs=("pi", "eps_m"),
        outputs=("i",),
        fn=lambda v, p: {"i": (1.0 + p["r_ss"]) * ((1.0 + v["pi"]) ** p["phi_pi"]) * np.exp(v["eps_m"]) - 1.0},
    )
    # 2. Nonlinear Fisher equation: (1 + i) / (1 + pi(+1)) - 1
    fisher = SimpleBlock(
        "fisher",
        inputs=("i", "pi"),
        outputs=("r",),
        fn=lambda v, p: {"r": (1.0 + v["i"]) / (1.0 + v["pi"](+1)) - 1.0},
    )
    # 3. Rotemberg pricing nonlinear Phillips curve:
    # pi * (1 + pi) - beta * pi(+1) * (1 + pi(+1)) - kappa * (C - C_ss)
    nkpc = SimpleBlock(
        "nkpc",
        inputs=("pi", "C"),
        outputs=("nkpc_res",),
        fn=lambda v, p: {
            "nkpc_res": v["pi"] * (1.0 + v["pi"])
            - p["beta"] * v["pi"](+1) * (1.0 + v["pi"](+1))
            - p["kappa"] * (v["C"] - p["C_ss"])
        },
    )
    # 4. Nonlinear CRRA Euler equation:
    # C^(-sigma) - beta * (1 + r) * C(+1)^(-sigma)
    euler = SimpleBlock(
        "euler",
        inputs=("C", "r"),
        outputs=("euler_res",),
        fn=lambda v, p: {
            "euler_res": (v["C"] ** (-p["sigma"]))
            - p["beta"] * (1.0 + v["r"]) * (v["C"](+1) ** (-p["sigma"]))
        },
    )

    dag = BlockDAG(
        [nkpc, euler, fisher, taylor],
        unknowns=("C", "pi"),
        exogenous=("eps_m",),
        targets=("euler_res", "nkpc_res"),
    )

    T = 40
    eps_path = 0.02 * (0.6 ** np.arange(T))
    solve_fn = getattr(dag, "nonlinear_transition", dag.solve_nonlinear_transition)
    nl_res = solve_fn(ss, params, {"eps_m": eps_path}, T=T, tol=1e-11)
    assert nl_res.converged
    assert nl_res.iterations > 1  # Genuine nonlinearity requires multi-step Newton iteration
    assert nl_res.residual_norm < 1e-11

