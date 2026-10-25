"""Adversarial stress-test suite for PureMacro v5.9.0 Frontier Tracks 4, 5A, 5B, 7.

Rigorously verifies:
1. Track 4: Continuous-time HJB upwind finite-volume & adjoint KFE conservation laws
   across extreme densities (Na in [30, 200]), borrowing limits, and non-uniform grids;
   plus BlockDAG general-equilibrium transition paths with residual < 1e-11.
2. Track 5A: Baqaee-Farhi 2nd-order network macro aggregation exactness across random
   sector sizes (N=3, 5, 8), Cobb-Douglas reduction <= 1e-14, and analytical Hessian
   vs exact CES finite difference <= 1e-8.
3. Track 5B: Dynamic spatial migration hat-algebra unit-shock stationarity <= 1e-12,
   total population conservation <= 1e-12, and row-stochastic migration <= 1e-14 across
   multi-region economies (N=3, 5) with asymmetric trade cost shocks.
4. Track 7: Self-auditing PolicyStudio zero-server interactive HTML5/SVG export with
   zero external CDN dependencies (http:// or https://) and valid inline SVG polylines.
"""
from __future__ import annotations

import re
import numpy as np
import scipy.linalg
import pytest

from puremacro.vfi.hjb_achdou import solve_hjb_achdou, _quadrature_weights
from puremacro.dsge import BlockDAG, HetBlock, SimpleBlock
from puremacro.trade import (
    baqaee_farhi_decomposition,
    solve_exact_ces_network_prices,
)
from puremacro.spatial import (
    DynamicSpatialHatModel,
    solve_dynamic_spatial_hat,
)
from puremacro.study import (
    CounterfactualGrid,
    PolicyStudioResult,
    run_policy_studio,
)


# ===========================================================================
# 1. Continuous-Time HJB/KFE Adversarial Verification (Track 4)
# ===========================================================================

class TestAdversarialContinuousTimeHJB:
    """Stress tests for continuous-time implicit upwind HJB and adjoint KFE."""

    @pytest.mark.parametrize("Na", [30, 45, 75, 120, 200])
    def test_extreme_grid_densities_conservation(self, Na: int):
        """Generator row-sum <= 1e-13, mass conservation <= 1e-13, adjoint residual <= 1e-11."""
        sol = solve_hjb_achdou(Na=Na, a_min=0.0, a_max=35.0, r_rate=0.03, w_rate=1.0, tol=1e-8)
        assert sol.converged
        assert sol.g_dist is not None

        A = sol.A_generator
        # 1. Generator row-sum conservation: ||A 1||_inf <= 1e-13
        row_sums = A @ np.ones(A.shape[0])
        max_row_sum = float(np.max(np.abs(row_sums)))
        assert max_row_sum <= 1e-13, f"Na={Na}: max_row_sum={max_row_sum} > 1e-13"

        # 2. Distribution mass conservation: |sum g_i Delta a_i - 1| <= 1e-13
        w_a = _quadrature_weights(sol.a_grid)
        total_mass = float(np.sum(sol.g_dist * w_a[:, None]))
        mass_error = abs(total_mass - 1.0)
        assert mass_error <= 1e-13, f"Na={Na}: mass_error={mass_error} > 1e-13"
        assert sol.mass_residual <= 1e-13

        # 3. Stationary adjoint residual: ||A^T pi||_inf <= 1e-11
        pi = (sol.g_dist * w_a[:, None]).ravel(order="F")
        adj_residual = float(np.max(np.abs(A.T @ pi)))
        assert adj_residual <= 1e-11, f"Na={Na}: adj_residual={adj_residual} > 1e-11"

    @pytest.mark.parametrize("a_min", [-5.0, -2.0, 0.0, 1.0, 5.0])
    def test_borrowing_limit_shifts(self, a_min: float):
        """Verify strict conservation across positive, zero, and negative borrowing limits."""
        Na = 60
        sol = solve_hjb_achdou(Na=Na, a_min=a_min, a_max=40.0, r_rate=0.03, w_rate=1.0, tol=1e-8)
        assert sol.converged

        A = sol.A_generator
        max_row_sum = float(np.max(np.abs(A @ np.ones(A.shape[0]))))
        assert max_row_sum <= 1e-13

        w_a = _quadrature_weights(sol.a_grid)
        total_mass = float(np.sum(sol.g_dist * w_a[:, None]))
        assert abs(total_mass - 1.0) <= 1e-13

        pi = (sol.g_dist * w_a[:, None]).ravel(order="F")
        adj_res = float(np.max(np.abs(A.T @ pi)))
        assert adj_res <= 1e-11

    @pytest.mark.parametrize("power", [1.5, 2.0, 3.0])
    def test_nonuniform_polynomial_and_clustered_grids(self, power: float):
        """Verify conservation on highly non-uniform grids clustered near borrowing constraint."""
        Na = 70
        s = np.linspace(0.0, 1.0, Na) ** power
        a_grid = -1.0 + 35.0 * s
        sol = solve_hjb_achdou(a_grid=a_grid, r_rate=0.03, w_rate=1.0, tol=1e-8)
        assert sol.converged

        A = sol.A_generator
        assert float(np.max(np.abs(A @ np.ones(A.shape[0])))) <= 1e-13

        w_a = _quadrature_weights(sol.a_grid)
        assert abs(float(np.sum(sol.g_dist * w_a[:, None])) - 1.0) <= 1e-13

        pi = (sol.g_dist * w_a[:, None]).ravel(order="F")
        assert float(np.max(np.abs(A.T @ pi))) <= 1e-11

    def test_perturbed_random_spacings_grid(self):
        """Verify conservation on irregular asset grid with random non-uniform step sizes."""
        rng = np.random.default_rng(2026)
        Na = 80
        # Positive random increments
        da = rng.uniform(0.1, 1.5, size=Na - 1)
        a_grid = np.insert(np.cumsum(da), 0, 0.0)

        sol = solve_hjb_achdou(a_grid=a_grid, r_rate=0.03, w_rate=1.0, tol=1e-8)
        assert sol.converged
        A = sol.A_generator
        assert float(np.max(np.abs(A @ np.ones(A.shape[0])))) <= 1e-13
        w_a = _quadrature_weights(sol.a_grid)
        assert abs(float(np.sum(sol.g_dist * w_a[:, None])) - 1.0) <= 1e-13
        pi = (sol.g_dist * w_a[:, None]).ravel(order="F")
        assert float(np.max(np.abs(A.T @ pi))) <= 1e-11

    @pytest.mark.parametrize("shock_size", [-0.03, 0.01, 0.04])
    def test_hjb_hetblock_dag_ge_transitions(self, shock_size: float):
        """Verify BlockDAG general equilibrium transition residual norm < 1e-11."""
        r_ss = 0.03
        w_ss = 1.0
        sol_ss = solve_hjb_achdou(Na=40, r_rate=r_ss, w_rate=w_ss, tol=1e-8)
        w_a = _quadrature_weights(sol_ss.a_grid)
        A_ss = float(np.sum(sol_ss.a_grid[:, None] * sol_ss.g_dist * w_a[:, None]))

        eps = 1e-4
        sol_r = solve_hjb_achdou(Na=40, r_rate=r_ss + eps, w_rate=w_ss, tol=1e-8)
        A_r = float(np.sum(sol_r.a_grid[:, None] * sol_r.g_dist * w_a[:, None]))
        dK_dr = (A_r - A_ss) / eps

        sol_w = solve_hjb_achdou(Na=40, r_rate=r_ss, w_rate=w_ss + eps, tol=1e-8)
        A_w = float(np.sum(sol_w.a_grid[:, None] * sol_w.g_dist * w_a[:, None]))
        dK_dw = (A_w - A_ss) / eps

        T = 20
        decay = 0.85 ** np.arange(T)
        J_A_r = scipy.linalg.toeplitz(dK_dr * decay, np.zeros(T))
        J_A_w = scipy.linalg.toeplitz(dK_dw * decay, np.zeros(T))

        hh = HetBlock(
            "hjb_household",
            inputs=("r", "w"),
            outputs=("A",),
            jacobian_fn={"A": {"r": J_A_r, "w": J_A_w}},
        )

        alpha = 0.33
        delta = 0.05
        K_ss = A_ss
        Z_ss = 1.0
        Y_ss = Z_ss * (K_ss ** alpha)

        firm = SimpleBlock(
            "firm",
            inputs=("K", "Z_shock"),
            outputs=("r", "w", "Y"),
            fn=lambda v, p: {
                "r": p["alpha"] * (v["Z_shock"] + p["Z_ss"]) * (v["K"] ** (p["alpha"] - 1.0)) - p["delta"],
                "w": (1.0 - p["alpha"]) * (v["Z_shock"] + p["Z_ss"]) * (v["K"] ** p["alpha"]),
                "Y": (v["Z_shock"] + p["Z_ss"]) * (v["K"] ** p["alpha"]),
            },
        )

        mkt = SimpleBlock(
            "mkt_clearing",
            inputs=("A", "K"),
            outputs=("asset_mkt_res",),
            fn=lambda v, p: {"asset_mkt_res": v["A"] - v["K"]},
        )

        dag = BlockDAG(
            blocks=[firm, hh, mkt],
            unknowns=("K",),
            exogenous=("Z_shock",),
            targets=("asset_mkt_res",),
        )

        ss = {
            "K": K_ss,
            "Z_shock": 0.0,
            "r": r_ss,
            "w": w_ss,
            "Y": Y_ss,
            "A": A_ss,
            "asset_mkt_res": 0.0,
        }
        params = {"alpha": alpha, "delta": delta, "Z_ss": Z_ss}

        shock_path = shock_size * (0.75 ** np.arange(T))
        nl_res = dag.solve_nonlinear_transition(
            ss, params, {"Z_shock": shock_path}, T=T, tol=1e-11
        )
        assert nl_res.converged
        assert nl_res.residual_norm < 1e-11, f"shock={shock_size}: residual_norm={nl_res.residual_norm} >= 1e-11"


# ===========================================================================
# 2. Baqaee-Farhi 2nd-Order Network Macro Adversarial Verification (Track 5A)
# ===========================================================================

class TestAdversarialBaqaeeFarhiNetwork:
    """Stress tests for Baqaee-Farhi (2019, 2020) input-output aggregation."""

    @pytest.mark.parametrize("N", [3, 5, 8])
    def test_cobb_douglas_reduction_across_sector_sizes(self, N: int):
        """Cobb-Douglas reduction error <= 1e-14 across random sector sizes N in {3, 5, 8}."""
        rng = np.random.default_rng(N * 31 + 7)
        for trial in range(3):
            Omega = rng.uniform(0.01, 0.20, size=(N, N))
            row_target = rng.uniform(0.3, 0.7, size=N)
            Omega = Omega * (row_target[:, None] / Omega.sum(axis=1)[:, None])

            b = rng.uniform(0.1, 1.0, size=N)
            b = b / float(np.sum(b))

            z = rng.normal(0, 0.06, size=N)

            res = baqaee_farhi_decomposition(
                Omega,
                b,
                elasticities=1.0,
                tfp_shocks=z,
                final_elasticity=1.0,
            )

            # Assert Cobb-Douglas reduction error <= 1e-14
            assert abs(res.second_order_dlog_y) <= 1e-14
            assert np.max(np.abs(res.hessian_matrix)) <= 1e-14
            # Exact nonlinear fixed point matches 1st-order Hulten to machine precision
            assert abs(res.exact_ces_dlog_y - res.first_order_dlog_y) <= 1e-13

    @pytest.mark.parametrize("N", [3, 5, 8])
    def test_analytical_hessian_vs_finite_difference_match(self, N: int):
        """Analytical Hessian matches finite differences of exact CES fixed point <= 1e-8."""
        rng = np.random.default_rng(N * 47 + 13)
        Omega = rng.uniform(0.02, 0.15, size=(N, N))
        row_target = rng.uniform(0.3, 0.6, size=N)
        Omega = Omega * (row_target[:, None] / Omega.sum(axis=1)[:, None])

        b = rng.uniform(0.2, 0.8, size=N)
        b = b / float(np.sum(b))

        z = rng.normal(0, 0.04, size=N)
        theta = rng.uniform(0.4, 1.6, size=N)
        theta_0 = float(rng.uniform(0.5, 1.5))

        res = baqaee_farhi_decomposition(
            Omega,
            b,
            elasticities=theta,
            tfp_shocks=z,
            final_elasticity=theta_0,
        )

        # Finite difference evaluation with Richardson extrapolation
        h = 2.0e-3
        def fd_mat(eps):
            mat = np.zeros((N, N), dtype=np.float64)
            for k in range(N):
                for l in range(N):
                    ek = np.zeros(N)
                    el = np.zeros(N)
                    ek[k] = eps
                    el[l] = eps
                    _, y_pp = solve_exact_ces_network_prices(Omega, b, theta, ek + el, final_elasticity=theta_0)
                    _, y_pm = solve_exact_ces_network_prices(Omega, b, theta, ek - el, final_elasticity=theta_0)
                    _, y_mp = solve_exact_ces_network_prices(Omega, b, theta, -ek + el, final_elasticity=theta_0)
                    _, y_mm = solve_exact_ces_network_prices(Omega, b, theta, -ek - el, final_elasticity=theta_0)
                    mat[k, l] = (y_pp - y_pm - y_mp + y_mm) / (4.0 * eps * eps)
            return mat

        h1 = fd_mat(h)
        h2 = fd_mat(2.0 * h)
        h_rich = (4.0 * h1 - h2) / 3.0

        max_diff = float(np.max(np.abs(res.hessian_matrix - h_rich)))
        assert max_diff <= 1e-8, f"N={N}: max_diff={max_diff} > 1e-8"

    def test_asymmetric_markups_allocative_efficiency(self):
        """Verify markup distortions generate positive allocative efficiency gaps."""
        N = 4
        rng = np.random.default_rng(404)
        Omega = rng.uniform(0.05, 0.18, size=(N, N))
        Omega = Omega * (0.50 / Omega.sum(axis=1)[:, None])
        b = np.array([0.25, 0.25, 0.25, 0.25])
        z = np.array([-0.05, 0.02, -0.01, 0.03])
        markups = np.array([1.35, 1.20, 1.10, 1.05])

        res = baqaee_farhi_decomposition(
            Omega,
            b,
            elasticities=0.45,
            tfp_shocks=z,
            final_elasticity=0.65,
            markups=markups,
        )
        assert abs(res.allocative_efficiency_dlog_y) > 1e-5
        assert np.all(res.domar_weights < res.cost_domar_weights)


# ===========================================================================
# 3. Dynamic Spatial Migration Hat-Algebra Adversarial Verification (Track 5B)
# ===========================================================================

class TestAdversarialDynamicSpatialHat:
    """Stress tests for Caliendo, Parro & Rossi-Hansberg (2019) dynamic hat-algebra."""

    def _make_stationary_model(self, N: int, seed: int) -> DynamicSpatialHatModel:
        rng = np.random.default_rng(seed)
        mu0 = rng.uniform(0.01, 0.05, size=(N, N))
        np.fill_diagonal(mu0, rng.uniform(0.8, 0.95, size=N))
        mu0 = mu0 / mu0.sum(axis=1)[:, None]

        # Stationary population L0 is the left eigenvector of mu0
        eigvals, eigvecs = scipy.linalg.eig(mu0.T)
        idx = np.argmin(np.abs(eigvals - 1.0))
        L0 = np.abs(np.real(eigvecs[:, idx]))
        L0 = L0 / float(np.sum(L0)) * 100.0

        pi0 = rng.uniform(0.05, 0.2, size=(N, N))
        np.fill_diagonal(pi0, rng.uniform(0.5, 0.8, size=N))
        pi0 = pi0 / pi0.sum(axis=1)[:, None]

        # Stationary nominal GDP Y0 is the left eigenvector of pi0
        eigvals_pi, eigvecs_pi = scipy.linalg.eig(pi0.T)
        idx_pi = np.argmin(np.abs(eigvals_pi - 1.0))
        Y0 = np.abs(np.real(eigvecs_pi[:, idx_pi]))
        Y0 = Y0 / float(np.sum(Y0)) * 100.0
        w0 = Y0 / L0

        return DynamicSpatialHatModel(
            initial_population=L0,
            initial_trade_shares=pi0,
            initial_migration_shares=mu0,
            trade_elasticity=4.0,
            migration_dispersion=0.5,
            beta=0.96,
            congestion_alpha=0.10,
            initial_wages=w0,
        )

    @pytest.mark.parametrize("N", [3, 5])
    def test_unit_shock_stationarity_exactness(self, N: int):
        """Unit-shock stationarity: |x_hat - 1| <= 1e-12 for all macroeconomic variables."""
        model = self._make_stationary_model(N, seed=N * 100 + 42)
        res = solve_dynamic_spatial_hat(model, horizon=25, tol=1e-13)

        assert res.converged
        assert float(np.max(np.abs(res.population_hat_path - 1.0))) <= 1e-12
        assert float(np.max(np.abs(res.wage_hat_path - 1.0))) <= 1e-12
        assert float(np.max(np.abs(res.price_hat_path - 1.0))) <= 1e-12
        assert float(np.max(np.abs(res.real_wage_hat_path - 1.0))) <= 1e-12
        assert float(np.max(np.abs(res.value_hat_path - 1.0))) <= 1e-12
        assert float(np.max(np.abs(res.welfare_hat - 1.0))) <= 1e-12
        assert res.population_conservation_error <= 1e-12
        assert res.migration_row_stochastic_error <= 1e-14

    @pytest.mark.parametrize("N", [3, 5])
    def test_asymmetric_trade_cost_shocks_conservation(self, N: int):
        """Asymmetric trade shocks preserve population <= 1e-12 and stochasticity <= 1e-14."""
        model = self._make_stationary_model(N, seed=N * 200 + 55)
        horizon = 25
        tau_hat = np.ones((horizon, N, N), dtype=np.float64)

        # Strongly asymmetric shock: corridor trade cost cuts and retaliatory tariffs
        tau_hat[:, 0, 1] = 0.82
        tau_hat[:, 1, 0] = 0.94
        if N >= 3:
            tau_hat[:, 2, 0] = 1.15
            tau_hat[:, 0, 2] = 1.08
        if N >= 5:
            tau_hat[:, 3, 4] = 0.78
            tau_hat[:, 4, 3] = 1.20

        res = solve_dynamic_spatial_hat(model, horizon=horizon, trade_cost_hat_path=tau_hat, tol=1e-11)
        assert res.converged

        # Assert total population conservation error <= 1e-12
        assert res.population_conservation_error <= 1e-12, (
            f"N={N}: population_conservation_error={res.population_conservation_error} > 1e-12"
        )
        # Assert row-stochastic migration matrices error <= 1e-14
        assert res.migration_row_stochastic_error <= 1e-14, (
            f"N={N}: migration_row_stochastic_error={res.migration_row_stochastic_error} > 1e-14"
        )

    def test_joint_asymmetric_shocks_welfare_monotonicity(self):
        """Joint asymmetric productivity and trade shocks maintain conservation laws."""
        N = 3
        model = self._make_stationary_model(N, seed=777)
        horizon = 30
        A_hat = np.ones((horizon, N))
        A_hat[:, 0] = 1.12
        A_hat[:, 2] = 0.95

        tau_hat = np.ones((horizon, N, N))
        tau_hat[:, 0, 1] = 0.88

        res = solve_dynamic_spatial_hat(
            model,
            horizon=horizon,
            tfp_hat_path=A_hat,
            trade_cost_hat_path=tau_hat,
            tol=1e-11,
        )
        assert res.converged
        assert res.population_conservation_error <= 1e-12
        assert res.migration_row_stochastic_error <= 1e-14
        # Booming region attracts workers and registers welfare gain
        assert res.population_hat_path[-1, 0] > 1.01
        assert res.welfare_hat[0] > 1.0


# ===========================================================================
# 4. PolicyStudio Zero-Server HTML5/SVG Adversarial Verification (Track 7)
# ===========================================================================

class TestAdversarialPolicyStudioHTML5:
    """Stress tests for PolicyStudio zero-server HTML5/SVG interactive exporter."""

    def test_zero_server_html5_svg_strict_dependencies(self):
        """Assert zero occurrences of http:// or https:// and valid inline SVG tags."""
        grid = CounterfactualGrid({"tax": [0.10, 0.20, 0.30], "subsidy": [0.0, 0.05]})
        res = run_policy_studio(
            lambda p: {
                "gdp": 100.0 * (1.0 - p["tax"]) * (1.0 + p["subsidy"]),
                "welfare": 50.0 - 20.0 * (p["tax"] ** 2),
                "residual": 1e-12,
            },
            grid,
        )

        html = res.to_interactive_html(
            title="Frontier Counterfactual Studio",
            subtitle="Testing zero CDN dependencies and inline SVG polylines",
        )

        # 1. Zero occurrences of http:// or https://
        assert "http://" not in html.lower(), "Found http:// URL in PolicyStudio HTML5 export!"
        assert "https://" not in html.lower(), "Found https:// URL in PolicyStudio HTML5 export!"

        # 2. Inline <svg> tags present and valid
        assert "<svg" in html and "</svg>" in html, "Missing inline <svg> tag in HTML5 export!"
        assert "<polyline" in html, "Missing inline SVG polyline elements!"
        assert 'class="pm-chart"' in html
        assert 'class="pm-series"' in html
        assert 'class="pm-toggle"' in html
        assert 'id="pm-horizon-slider"' in html

    def test_html_security_escaping_and_status_badges(self):
        """Assert proper HTML entity escaping and correct PASS/WARN/FAIL status badges."""
        grid = CounterfactualGrid({"policy_name": ["Baseline", "Reformed & Tested"]})

        # Test warning status
        res_warn = run_policy_studio(
            lambda p: {"metric": 10.0, "residual": 5e-5},
            grid,
            warn_tol=1e-6,
            fail_tol=1e-3,
        )
        html_warn = res_warn.to_interactive_html(
            title="<script>alert('xss')</script> & Policy Studio",
            subtitle="<b>Unescaped</b> test",
        )

        # Script tags must be escaped
        assert "<script>alert('xss')</script>" not in html_warn
        assert "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;" in html_warn
        assert "&amp;" in html_warn
        assert "WARN" in html_warn
        assert "http://" not in html_warn.lower()
        assert "https://" not in html_warn.lower()

        # Test failure status
        res_fail = run_policy_studio(
            lambda p: {"metric": 10.0, "residual": 0.05},
            grid,
            warn_tol=1e-6,
            fail_tol=1e-3,
        )
        html_fail = res_fail.to_interactive_html()
        assert "FAIL" in html_fail
        assert "http://" not in html_fail.lower()
        assert "https://" not in html_fail.lower()
        assert "<svg" in html_fail and "</svg>" in html_fail
