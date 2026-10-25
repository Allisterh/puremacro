"""Adversarial stress-test suite for PureMacro v5.9.0 Tracks 1, 2, 3, 6.

Written by Challenger 1 (challenger_v59_1).
Tests empirical corner cases, stress loads, and mathematical exactness:
1. Forward AD (Track 6): HyperDualArray exactness on 5x5 and 10x10 quadratic forms,
   transcendental compositions, and nested operations vs 4th-order FD <= 1e-10.
2. DAG SSJ & Whittle (Track 1): Multi-equation nonlinear macro DAG with large shocks
   (up to 20%), residual norm <= 1e-11; Whittle estimation on ill-conditioned spectral
   densities with positive-definite covariance recovery.
3. Nonlinear OccBin (Track 2): Multi-period alternating positive/negative shocks,
   regime switching convergence, and bound compliance <= 1e-10 under Order 2 and 3.
4. Nowcasting News (Track 3): Random jagged-edge panels with release updates and
   historical revisions, Banbura-Modugno exact additivity |Delta y_hat - sum news| <= 1e-12.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro._autodiff import (
    HyperDualArray,
    hessian_forward,
    jacobian_forward,
)
from puremacro.dsge import (
    BlockDAG,
    OccBinConstraint,
    SimpleBlock,
    build_dynare,
    estimate_hank_whittle,
    solve_nonlinear_occbin,
)
from puremacro.nowcast.dfm import DynamicFactorModel
from puremacro.nowcast.news import banbura_modugno_news


# ===========================================================================
# Helper: 4th-order Central Finite Differences Oracle for Hessian
# ===========================================================================

def _fourth_order_fd_hessian(fn, x: np.ndarray, h: float = 1e-3) -> np.ndarray:
    """Compute 4th-order central finite difference Hessian using exact forward gradients.

    O(h^4) truncation error:
    H_{:, j} = (-g(x + 2h e_j) + 8 g(x + h e_j) - 8 g(x - h e_j) + g(x - 2h e_j)) / (12 h)
    """
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    P = x_arr.size
    hess_fd = np.zeros((P, P), dtype=np.float64)
    for j in range(P):
        e_j = np.zeros(P, dtype=np.float64)
        e_j[j] = 1.0
        _, g_p2 = jacobian_forward(fn, x_arr + 2.0 * h * e_j)
        _, g_p1 = jacobian_forward(fn, x_arr + 1.0 * h * e_j)
        _, g_m1 = jacobian_forward(fn, x_arr - 1.0 * h * e_j)
        _, g_m2 = jacobian_forward(fn, x_arr - 2.0 * h * e_j)
        hess_fd[:, j] = (-g_p2 + 8.0 * g_p1 - 8.0 * g_m1 + g_m2) / (12.0 * h)
    return 0.5 * (hess_fd + hess_fd.T)


# ===========================================================================
# 1. Forward AD Exactness (Track 6)
# ===========================================================================

class TestTrack6ForwardADExactness:
    """Adversarial challenge for HyperDualArray exact 2nd-order differentiation."""

    def test_hyperdual_matrix_quadratic_form_5x5(self):
        """Stress-test 5x5 matrix quadratic form with transcendental perturbation."""
        rng = np.random.default_rng(1005)
        P = 5
        W = rng.normal(size=(P, P))
        W = W.T @ W + 0.5 * np.eye(P)
        b = rng.normal(size=P)

        def obj_5x5(x):
            r = W @ x - b
            return 0.5 * (r @ r) + np.sum(np.exp(0.2 * x) + np.sin(x))

        x0 = np.array([0.35, -0.42, 0.68, -0.15, 0.55])
        val, grad, hess_ad = hessian_forward(obj_5x5, x0)

        # Exact analytical Hessian: W^T W + diag(0.04 exp(0.2 x) - sin(x))
        hess_exact = W.T @ W + np.diag(0.04 * np.exp(0.2 * x0) - np.sin(x0))
        hess_fd = _fourth_order_fd_hessian(obj_5x5, x0, h=1e-3)

        diff_exact = np.max(np.abs(hess_ad - hess_exact))
        diff_fd = np.max(np.abs(hess_ad - hess_fd))

        assert diff_exact <= 1e-13, f"AD vs Exact analytical diff: {diff_exact}"
        assert diff_fd <= 1e-10, f"AD vs 4th-order FD diff: {diff_fd}"
        assert np.allclose(hess_ad, hess_ad.T, atol=1e-15)

    def test_hyperdual_matrix_quadratic_form_10x10_ill_conditioned(self):
        """Stress-test 10x10 matrix quadratic form with condition number > 200."""
        rng = np.random.default_rng(1010)
        P = 10
        # Ill-conditioned matrix with singular values ranging over 2 orders of magnitude
        U, _ = np.linalg.qr(rng.normal(size=(P, P)))
        s = np.geomspace(10.0, 0.05, P)
        W = U @ np.diag(s) @ U.T
        b = rng.normal(size=P)

        def obj_10x10(x):
            r = W @ x - b
            return 0.5 * (r @ r) + np.sum(np.log(1.0 + x ** 2) + np.cos(x))

        x0 = np.linspace(0.1, 0.9, P)
        val, grad, hess_ad = hessian_forward(obj_10x10, x0)

        # Analytical second derivative of log(1 + x^2) + cos(x):
        # d/dx [ 2x / (1+x^2) - sin(x) ] = (2*(1+x^2) - 4x^2)/(1+x^2)^2 - cos(x) = (2 - 2x^2)/(1+x^2)^2 - cos(x)
        d2_trans = (2.0 - 2.0 * (x0 ** 2)) / ((1.0 + x0 ** 2) ** 2) - np.cos(x0)
        hess_exact = W.T @ W + np.diag(d2_trans)
        hess_fd = _fourth_order_fd_hessian(obj_10x10, x0, h=1e-3)

        diff_exact = np.max(np.abs(hess_ad - hess_exact))
        diff_fd = np.max(np.abs(hess_ad - hess_fd))

        assert diff_exact <= 1e-13, f"AD vs Exact analytical diff: {diff_exact}"
        assert diff_fd <= 1e-10, f"AD vs 4th-order FD diff: {diff_fd}"
        assert np.allclose(hess_ad, hess_ad.T, atol=1e-15)

    def test_hyperdual_transcendental_compositions(self):
        """Stress-test complex transcendental compositions involving sqrt, exp, log, sin, cos."""
        def transcendental_pot(x):
            term1 = np.sum(np.log(np.exp(0.5 * x) + np.exp(-0.5 * x)))
            term2 = np.sum(np.sin(x[:-1] * x[1:]))
            term3 = np.sum(np.sqrt(1.5 + x ** 2))
            return term1 + term2 + term3

        x0 = np.array([0.25, -0.6, 0.45, -0.3, 0.8])
        val, grad, hess_ad = hessian_forward(transcendental_pot, x0)
        hess_fd = _fourth_order_fd_hessian(transcendental_pot, x0, h=1e-3)

        diff_fd = np.max(np.abs(hess_ad - hess_fd))
        assert diff_fd <= 1e-10, f"AD vs 4th-order FD diff: {diff_fd}"
        assert np.allclose(hess_ad, hess_ad.T, atol=1e-15)

    def test_hyperdual_nested_operations(self):
        """Stress-test nested matrix-vector operations with nonlinear denominator coupling."""
        rng = np.random.default_rng(2024)
        P = 6
        A = rng.normal(size=(P, P))
        A = A.T @ A + np.eye(P)
        C = rng.normal(size=(P, P))
        C = 0.5 * (C + C.T)
        b = rng.normal(size=P)

        def nested_fn(x):
            quad_a = x @ (A @ x) + 2.0
            quad_c = x @ (C @ x)
            denom = 1.0 + np.exp(b @ x)
            return np.sqrt(quad_a) / denom + np.cos(quad_c)

        x0 = np.array([0.15, -0.22, 0.31, -0.18, 0.09, -0.25])
        val, grad, hess_ad = hessian_forward(nested_fn, x0)
        hess_fd = _fourth_order_fd_hessian(nested_fn, x0, h=1e-3)

        diff_fd = np.max(np.abs(hess_ad - hess_fd))
        assert diff_fd <= 1e-10, f"AD vs 4th-order FD diff: {diff_fd}"
        assert np.allclose(hess_ad, hess_ad.T, atol=1e-15)


# ===========================================================================
# 2. DAG SSJ & Whittle Estimation (Track 1)
# ===========================================================================

def _build_adversarial_macro_dag():
    """Build a 4-equation fully nonlinear general-equilibrium macro DAG."""
    beta = 0.99
    sigma = 1.5
    kappa = 0.15
    phi_pi = 1.5
    gamma = 1.2
    r_ss = 1.0 / beta - 1.0
    C_ss = 1.0

    taylor = SimpleBlock(
        "taylor",
        inputs=("pi", "eps_m"),
        outputs=("i",),
        fn=lambda v, p: {
            "i": (1.0 + p["r_ss"]) * ((1.0 + v["pi"]) ** p["phi_pi"]) * np.exp(v["eps_m"]) - 1.0
        },
    )

    fisher = SimpleBlock(
        "fisher",
        inputs=("i", "pi"),
        outputs=("r",),
        fn=lambda v, p: {"r": (1.0 + v["i"]) / (1.0 + v["pi"](+1)) - 1.0},
    )

    nkpc = SimpleBlock(
        "nkpc",
        inputs=("pi", "C"),
        outputs=("nkpc_res",),
        fn=lambda v, p: {
            "nkpc_res": (
                v["pi"] * (1.0 + v["pi"])
                - p["beta"] * v["pi"](+1) * (1.0 + v["pi"](+1))
                - p["kappa"] * ((v["C"] / p["C_ss"]) ** p["gamma"] - 1.0)
            )
        },
    )

    euler = SimpleBlock(
        "euler",
        inputs=("C", "r"),
        outputs=("euler_res",),
        fn=lambda v, p: {
            "euler_res": (
                (v["C"] ** (-p["sigma"]))
                - p["beta"] * (1.0 + v["r"]) * (v["C"](+1) ** (-p["sigma"]))
            )
        },
    )

    dag = BlockDAG(
        blocks=[nkpc, euler, fisher, taylor],
        unknowns=("C", "pi"),
        exogenous=("eps_m",),
        targets=("euler_res", "nkpc_res"),
    )

    ss = {"C": C_ss, "pi": 0.0, "i": r_ss, "r": r_ss, "eps_m": 0.0, "euler_res": 0.0, "nkpc_res": 0.0}
    params = {
        "beta": beta,
        "sigma": sigma,
        "kappa": kappa,
        "phi_pi": phi_pi,
        "gamma": gamma,
        "r_ss": r_ss,
        "C_ss": C_ss,
    }
    return dag, ss, params


class TestTrack1DAGSSJAndWhittle:
    """Adversarial challenge for sequence-space nonlinear DAG transitions and Whittle estimation."""

    @pytest.mark.parametrize("shock_size", [0.05, 0.10, 0.15, 0.20])
    def test_nonlinear_macro_dag_large_shocks_convergence(self, shock_size: float):
        """Verify nonlinear DAG transitions converge with residual norm <= 1e-11 up to 20% shocks."""
        dag, ss, params = _build_adversarial_macro_dag()
        T = 40
        eps_path = float(shock_size) * (0.6 ** np.arange(T))

        res = dag.solve_nonlinear_transition(
            ss, params, {"eps_m": eps_path}, T=T, tol=1e-12, max_iter=70
        )
        assert res.converged, f"Failed to converge at shock {shock_size*100:.0f}%"
        assert res.residual_norm <= 1e-11, f"Residual norm {res.residual_norm} exceeds 1e-11"
        assert len(res.paths) == T
        assert res.mode == "nonlinear_dag_newton"

    def test_whittle_estimator_ill_conditioned_spectral_density_univariate(self):
        """Stress-test Whittle MLE on near-unit-root persistent process with tiny observation noise."""
        rng = np.random.default_rng(888)
        T_sim = 220
        rho_true = 0.93
        eps = rng.normal(0.0, 0.005, size=T_sim)
        y = np.zeros(T_sim)
        for t in range(1, T_sim):
            y[t] = rho_true * y[t - 1] + eps[t]
        # Ill-conditioned dynamic range
        y = y + rng.normal(0.0, 1e-5, size=T_sim)

        def ma_builder(p):
            rho = p["rho"]
            M = (rho ** np.arange(128)).reshape(128, 1, 1)
            return M, np.array([[0.005 ** 2]])

        res = estimate_hank_whittle(
            ma_builder, y, init_params={"rho": 0.75}, bounds={"rho": (0.1, 0.99)}
        )
        assert res.converged
        assert abs(res.params["rho"] - rho_true) < 0.15
        assert res.vcov.shape == (1, 1)
        assert np.all(np.diag(res.vcov) > 0.0)
        eigs = np.linalg.eigvalsh(res.vcov)
        assert np.all(eigs > 0.0), f"Whittle vcov is not positive definite: eigs={eigs}"

    def test_whittle_estimator_ill_conditioned_spectral_density_bivariate(self):
        """Stress-test bivariate Whittle estimator with highly collinear shock covariance (corr=0.985)."""
        rng = np.random.default_rng(999)
        T_sim = 240
        # Highly correlated shocks
        corr = 0.985
        cov_true = np.array([[1.0, corr], [corr, 1.0]]) * 0.01 ** 2
        L = np.linalg.cholesky(cov_true)
        shocks = rng.normal(size=(T_sim, 2)) @ L.T

        rho_true = 0.82
        theta_true = 0.38
        Y = np.zeros((T_sim, 2))
        for t in range(1, T_sim):
            Y[t, 0] = rho_true * Y[t - 1, 0] + shocks[t, 0]
            Y[t, 1] = theta_true * Y[t - 1, 1] + shocks[t, 1]

        def bivariate_ma_builder(p):
            r = p["rho"]
            th = p["theta"]
            T_ma = 64
            M = np.zeros((T_ma, 2, 2))
            for s in range(T_ma):
                M[s, 0, 0] = r ** s
                M[s, 1, 1] = th ** s
            cov = np.array([[1.0, corr], [corr, 1.0]]) * 0.01 ** 2
            return M, cov

        res = estimate_hank_whittle(
            bivariate_ma_builder,
            Y,
            init_params={"rho": 0.65, "theta": 0.5},
            bounds={"rho": (0.1, 0.98), "theta": (0.05, 0.95)},
        )
        assert res.converged
        assert abs(res.params["rho"] - rho_true) < 0.18
        assert abs(res.params["theta"] - theta_true) < 0.18
        assert res.vcov.shape == (2, 2)
        assert np.all(np.diag(res.vcov) > 0.0)
        eigs = np.linalg.eigvalsh(res.vcov)
        assert np.all(eigs > 0.0), f"Bivariate Whittle vcov not positive definite: eigs={eigs}"


# ===========================================================================
# 3. Nonlinear OccBin (Track 2)
# ===========================================================================

def _build_occbin_nk_model():
    """Build Dynare-like nonlinear NK model with ZLB constraint on nominal interest rate."""
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


class TestTrack2NonlinearOccBin:
    """Adversarial stress-testing of 2nd/3rd-order pruned Nonlinear OccBin solver."""

    @pytest.mark.parametrize("order_val", [2, 3])
    def test_nonlinear_occbin_multi_period_alternating_shocks(self, order_val: int):
        """Stress-test with multi-period alternating positive/negative shocks."""
        m_ref, m_zlb, constraint = _build_occbin_nk_model()
        horizon = 40
        shocks_mat = np.zeros((horizon, 1), dtype=np.float64)
        # Sequence of alternating shocks: negative shock pulls into ZLB, positive pulls out
        shocks_mat[0, 0] = -0.05
        shocks_mat[1, 0] = +0.03
        shocks_mat[2, 0] = -0.04
        shocks_mat[3, 0] = +0.02
        shocks_mat[4, 0] = -0.03

        res = solve_nonlinear_occbin(
            m_ref,
            m_zlb,
            constraint,
            shocks_mat,
            order=order_val,
            sigma=1.0,
            horizon=horizon,
        )
        assert res.converged, f"Order {order_val} failed to converge on alternating shocks"
        assert res.binding_periods > 0, "Expected binding periods under negative shocks"
        # Bound compliance <= 1e-10
        assert res.max_bound_violation <= 1e-10, (
            f"Bound violation {res.max_bound_violation} exceeds 1e-10"
        )
        # Check actual values during binding periods
        i_path = res["i"].to_numpy()
        for t, r in enumerate(res.regimes):
            if r == 1:
                assert i_path[t] >= constraint.threshold - 1e-10
                assert abs(i_path[t] - constraint.threshold) <= 1e-10

    @pytest.mark.parametrize("order_val", [2, 3])
    def test_nonlinear_occbin_multi_spell_regime_switching(self, order_val: int):
        """Stress-test multi-spell regime switching: in-and-out of ZLB separated by gaps."""
        m_ref, m_zlb, constraint = _build_occbin_nk_model()
        horizon = 40
        shocks_mat = np.zeros((horizon, 1), dtype=np.float64)
        # Spell 1
        shocks_mat[0, 0] = -0.04
        shocks_mat[1, 0] = -0.03
        # Intervening expansionary push
        shocks_mat[6, 0] = +0.05
        shocks_mat[7, 0] = +0.04
        # Spell 2
        shocks_mat[14, 0] = -0.04
        shocks_mat[15, 0] = -0.03

        res = solve_nonlinear_occbin(
            m_ref,
            m_zlb,
            constraint,
            shocks_mat,
            order=order_val,
            sigma=1.0,
            horizon=horizon,
        )
        assert res.converged
        assert res.binding_periods > 0
        assert res.max_bound_violation <= 1e-10


# ===========================================================================
# 4. Nowcasting News Additivity (Track 3)
# ===========================================================================

class TestTrack3NowcastingNewsAdditivity:
    """Adversarial challenge for Banbura-Modugno exact analytical news additivity."""

    @pytest.mark.parametrize("seed", [101, 202, 303, 404, 505])
    def test_banbura_modugno_news_random_jagged_edge_panels(self, seed: int):
        """Stress-test random jagged-edge patterns, releases, and revisions asserting identity <= 1e-12."""
        rng = np.random.default_rng(seed)
        T = 48
        n = 8
        dates = pd.date_range("2020-01-01", periods=T, freq="MS")
        cols = [f"series_{i}" for i in range(n)]

        # 2-factor latent process
        F = np.zeros((T, 2))
        for t in range(1, T):
            F[t, 0] = 0.8 * F[t - 1, 0] + rng.normal(scale=0.5)
            F[t, 1] = 0.5 * F[t - 1, 1] + rng.normal(scale=0.3)

        Lambda = rng.uniform(0.5, 1.5, size=(n, 2))
        X = F @ Lambda.T + rng.normal(scale=0.2, size=(T, n))
        df_true = pd.DataFrame(X, index=dates, columns=cols)

        # Generate arbitrary jagged edges in the last 4 periods
        df_old = df_true.copy()
        mask_last = rng.uniform(size=(4, n)) < 0.45
        df_old.iloc[-4:, :] = np.where(mask_last, np.nan, df_old.iloc[-4:, :])
        # Random intermediate missingness
        hist_mask = rng.uniform(size=(T - 4, n)) < 0.04
        df_old.iloc[:-4, :] = np.where(hist_mask, np.nan, df_old.iloc[:-4, :])

        model = DynamicFactorModel(n_factors=2, p=1, standardize=True).fit(df_old)

        # Update vintage: simultaneous random new releases and historical revisions
        df_new = df_old.copy()
        missing_coords = np.argwhere(df_old.isna().to_numpy())
        if len(missing_coords) > 0:
            n_releases = min(len(missing_coords), int(rng.integers(1, 5)))
            chosen_rel = rng.choice(len(missing_coords), size=n_releases, replace=False)
            for idx in chosen_rel:
                r, c = missing_coords[idx]
                df_new.iloc[r, c] = df_true.iloc[r, c]

        observed_coords = np.argwhere(~df_old.isna().to_numpy())
        if len(observed_coords) > 0:
            n_revisions = int(rng.integers(1, 4))
            chosen_rev = rng.choice(len(observed_coords), size=n_revisions, replace=False)
            for idx in chosen_rev:
                r, c = observed_coords[idx]
                df_new.iloc[r, c] += float(rng.normal(scale=0.25))

        target_series = cols[0]
        res = banbura_modugno_news(model, df_old, df_new, target_series=target_series)

        # Banbura-Modugno exact additivity identity |Delta y_hat - sum news| <= 1e-12
        assert res.decomposition_error <= 1e-12, (
            f"Seed {seed}: decomposition error {res.decomposition_error:.2e} exceeds 1e-12"
        )
        assert abs(res.revision - res.total_impact) <= 1e-12, (
            f"Seed {seed}: revision - total_impact = {abs(res.revision - res.total_impact):.2e}"
        )

    def test_banbura_modugno_news_extreme_cascade_releases(self):
        """Stress-test cascade of data releases across multiple vintage updates."""
        rng = np.random.default_rng(777)
        T = 36
        n = 6
        dates = pd.date_range("2021-01-01", periods=T, freq="MS")
        cols = ["gdp", "ip", "retail", "cpi", "emp", "rate"]

        F = np.zeros(T)
        for t in range(1, T):
            F[t] = 0.7 * F[t - 1] + rng.normal(scale=0.4)

        loadings = np.array([1.2, 0.9, 0.8, 0.3, -0.7, 0.4])
        X = F[:, None] * loadings[None, :] + rng.normal(scale=0.15, size=(T, n))
        df_base = pd.DataFrame(X, index=dates, columns=cols)

        # Baseline vintage: heavy jagged edges
        df_v0 = df_base.copy()
        df_v0.iloc[-1, 1:] = np.nan
        df_v0.iloc[-2, 2:] = np.nan
        df_v0.iloc[-3, 4:] = np.nan

        model = DynamicFactorModel(n_factors=1, p=1, standardize=True).fit(df_v0)

        # Step 1: Release IP and Retail at T-2
        df_v1 = df_v0.copy()
        df_v1.iloc[-2, 2] = df_base.iloc[-2, 2]
        df_v1.iloc[-2, 3] = df_base.iloc[-2, 3]
        res1 = banbura_modugno_news(model, df_v0, df_v1, target_series="gdp")
        assert res1.decomposition_error <= 1e-12

        # Step 2: Release IP at T-1 and revise CPI at T-5
        df_v2 = df_v1.copy()
        df_v2.iloc[-1, 1] = df_base.iloc[-1, 1]
        df_v2.iloc[-5, 3] += 0.45
        res2 = banbura_modugno_news(model, df_v1, df_v2, target_series="gdp")
        assert res2.decomposition_error <= 1e-12

        # Step 3: Full release of remaining ragged edges
        df_v3 = df_base.copy()
        df_v3.iloc[-5, 3] = df_v2.iloc[-5, 3]  # keep revised value
        res3 = banbura_modugno_news(model, df_v2, df_v3, target_series="gdp")
        assert res3.decomposition_error <= 1e-12
