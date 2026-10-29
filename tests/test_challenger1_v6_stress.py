"""Empirical Adversarial Stress Test Suite for PureMacro v6.0.0 (Challenger 1).

Covers:
- Track I.1: Unified Differentiable Model Protocol & 6.0.0 Cleanups (puremacro.core.protocol)
- Track I.2: Vectorized Sequential Monte Carlo Engine (puremacro.inference.smc)
- Track II.3: Global Nonlinear HANK with Aggregate Risk (puremacro.dsge.hank_nonlinear)

Adversarial Stress Scenarios:
1. DifferentiableModel with complex inputs, ragged edges, boundary parameters, and AD gradients.
2. SMC with multimodal posteriors, poor initialization, degenerate weights, MALA step-size bounds.
3. Nonlinear HANK transitions with huge 1000-1500 bps shocks, near-autarky limits, Broyden convergence, and 2000 bps boundary limits.
4. 2nd-Order Aggregate Risk HANK with extreme volatility, multi-order quadratic scaling, and high-dimensional distribution moment limits.
"""
from __future__ import annotations

import importlib
import math
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# PureMacro Target Imports
# ---------------------------------------------------------------------------
import puremacro.core.protocol as proto
from puremacro.core.protocol import (
    DFMDifferentiableAdapter,
    DSGEDifferentiableAdapter,
    DifferentiableModel,
    HANKDAGDifferentiableAdapter,
    SVARDifferentiableAdapter,
    SpatialDifferentiableAdapter,
    StateSpaceProtocol,
    TradeDifferentiableAdapter,
    wrap_differentiable_model,
)
import puremacro.dsge.hank_nonlinear as hnl
from puremacro.dsge.hank_nonlinear import (
    HANKAggregateRiskResult,
    NonlinearHANKTransitionResult,
    build_canonical_hank_dag,
    solve_hank_aggregate_risk_order2,
    solve_hank_nonlinear_transition,
)
import puremacro.inference.smc as smc
from puremacro.inference.smc import (
    AdaptiveTemperingSchedule,
    SMCResult,
    smc_sample,
)


# ===========================================================================
# 1. Track I.1: DifferentiableModel & Deprecation Stress Tests
# ===========================================================================

class TestTrackI1DifferentiableModelAdversarial:
    """Adversarial stress testing of DifferentiableModel protocol and adapters."""

    def test_adv_all_adapters_protocol_compliance(self):
        """Verify all six domain adapters satisfy DifferentiableModel protocol."""
        adapters = [
            DSGEDifferentiableAdapter(None),
            HANKDAGDifferentiableAdapter(None),
            DFMDifferentiableAdapter(None),
            SVARDifferentiableAdapter(None),
            TradeDifferentiableAdapter(None),
            SpatialDifferentiableAdapter(None),
        ]
        for a in adapters:
            assert isinstance(a, DifferentiableModel), f"{type(a)} failed DifferentiableModel protocol check"
            # Verify uniform method existence and callability
            for method in ("impulse_response", "simulate", "loglik_and_grad", "counterfactual", "to_interactive_html"):
                assert hasattr(a, method) and callable(getattr(a, method)), f"{type(a)} missing {method}"

    def test_adv_wrap_differentiable_model_dispatch_and_pass_through(self):
        """Verify wrap_differentiable_model dispatches correctly and passes existing models through."""
        class DummyLinearModel:
            solution = "dummy"
            shocks = ("eps",)
            variables = ("y",)

        class DummyDAG:
            ge_jacobians = "dummy"
            exogenous = ("r", "Z")
            unknowns = ("Y",)
            targets = ("C",)

        class DummyDFM:
            n_factors = 2

        class DummySVAR:
            A_list = [np.eye(2)]

        class DummyTrade:
            trade_shares = np.ones((3, 3))

        class DummySpatial:
            migration_shares = np.ones((3, 3))

        m_dsge = wrap_differentiable_model(DummyLinearModel())
        assert isinstance(m_dsge, DSGEDifferentiableAdapter)
        assert isinstance(m_dsge, DifferentiableModel)

        m_dag = wrap_differentiable_model(DummyDAG())
        assert isinstance(m_dag, HANKDAGDifferentiableAdapter)
        assert isinstance(m_dag, DifferentiableModel)

        m_dfm = wrap_differentiable_model(DummyDFM())
        assert isinstance(m_dfm, DFMDifferentiableAdapter)

        m_svar = wrap_differentiable_model(DummySVAR())
        assert isinstance(m_svar, SVARDifferentiableAdapter)

        m_trade = wrap_differentiable_model(DummyTrade())
        assert isinstance(m_trade, TradeDifferentiableAdapter)

        m_spatial = wrap_differentiable_model(DummySpatial())
        assert isinstance(m_spatial, SpatialDifferentiableAdapter)

        # Idempotence: wrapping an already wrapped model returns the exact same instance
        assert wrap_differentiable_model(m_dsge) is m_dsge

    def test_adv_boundary_gradients_vs_finite_differences(self):
        """Stress-test DualArray AD gradients vs central finite differences across all adapters."""
        def _central_fd(adapter, theta, h=1e-5):
            P = len(theta)
            grad_fd = np.zeros(P)
            for i in range(P):
                e_i = np.zeros(P)
                e_i[i] = h
                ll_plus, _ = adapter.loglik_and_grad(params=theta + e_i)
                ll_minus, _ = adapter.loglik_and_grad(params=theta - e_i)
                grad_fd[i] = (ll_plus - ll_minus) / (2.0 * h)
            return grad_fd

        # 1. DSGE Adapter: Boundary AR persistence rho near 1.0 and -1.0
        dsge = DSGEDifferentiableAdapter(None)
        test_thetas_dsge = [
            np.array([0.5, 0.1, 0.05]),
            np.array([0.95, 0.05, 0.02]),
            np.array([-0.85, 0.15, 0.08]),
        ]
        for th in test_thetas_dsge:
            ll_ad, grad_ad = dsge.loglik_and_grad(params=th)
            grad_fd = _central_fd(dsge, th, h=1e-5)
            assert np.isfinite(ll_ad)
            assert np.all(np.isfinite(grad_ad))
            np.testing.assert_allclose(grad_ad, grad_fd, rtol=1e-3, atol=1e-4)

        # 2. HANK DAG Adapter (Whittle spectral loglik)
        hank_dag = HANKDAGDifferentiableAdapter(None)
        test_thetas_hank = [
            np.array([0.98, 1.0, 0.8]),
            np.array([0.95, 2.5, -0.6]),
        ]
        for th in test_thetas_hank:
            ll_ad, grad_ad = hank_dag.loglik_and_grad(params=th)
            grad_fd = _central_fd(hank_dag, th, h=1e-5)
            assert np.isfinite(ll_ad)
            np.testing.assert_allclose(grad_ad, grad_fd, rtol=1e-3, atol=1e-4)

        # 3. DFM Adapter
        dfm = DFMDifferentiableAdapter(None)
        th_dfm = np.array([0.75, 1.2, 0.15])
        ll_ad, grad_ad = dfm.loglik_and_grad(params=th_dfm)
        grad_fd = _central_fd(dfm, th_dfm, h=1e-5)
        np.testing.assert_allclose(grad_ad, grad_fd, rtol=1e-3, atol=1e-4)

        # 4. SVAR Adapter
        svar = SVARDifferentiableAdapter(None)
        th_svar = np.array([0.65, 0.08, 0.04, 0.70, 0.12, 0.18])
        ll_ad, grad_ad = svar.loglik_and_grad(params=th_svar)
        grad_fd = _central_fd(svar, th_svar, h=1e-5)
        np.testing.assert_allclose(grad_ad, grad_fd, rtol=1e-3, atol=1e-4)

        # 5. Trade Adapter
        trade = TradeDifferentiableAdapter(None)
        th_trade = np.array([4.2, 1.1, 0.30])
        ll_ad, grad_ad = trade.loglik_and_grad(params=th_trade)
        grad_fd = _central_fd(trade, th_trade, h=1e-5)
        np.testing.assert_allclose(grad_ad, grad_fd, rtol=1e-3, atol=1e-4)

        # 6. Spatial Adapter
        spatial = SpatialDifferentiableAdapter(None)
        th_spatial = np.array([2.2, 0.96, 1.1])
        ll_ad, grad_ad = spatial.loglik_and_grad(params=th_spatial)
        grad_fd = _central_fd(spatial, th_spatial, h=1e-5)
        np.testing.assert_allclose(grad_ad, grad_fd, rtol=1e-3, atol=1e-4)

    def test_adv_complex_and_ragged_inputs(self):
        """Stress-test adapters with DataFrames, dict params with missing/extra keys, and ragged shocks."""
        dsge = DSGEDifferentiableAdapter(None)

        # 1. Custom DataFrame with DatetimeIndex
        dates = pd.date_range("2020-01-01", periods=50, freq="QE")
        df_data = pd.DataFrame({"y": np.sin(np.linspace(0, 3, 50))}, index=dates)
        ll_df, grad_df = dsge.loglik_and_grad(data=df_data)
        assert np.isfinite(ll_df)
        assert grad_df.shape == (3,)

        # 2. Dict params with partial keys and extra irrelevant keys
        dict_params = {"rho": 0.82, "extra_ignored": 999.0}
        ll_dict, grad_dict = dsge.loglik_and_grad(params=dict_params)
        assert np.isfinite(ll_dict)
        assert grad_dict.shape == (3,)

        # 3. Ragged shock arrays for simulation
        sim_ragged = dsge.simulate(periods=30, shocks=np.ones((30, 1)))
        assert len(sim_ragged) == 30
        assert "y" in sim_ragged.columns

        # 4. Interactive HTML export on custom data
        html_str = dsge.to_interactive_html(sim_ragged, title="Stress Test Explorer")
        assert isinstance(html_str, str)
        assert "<html" in html_str.lower() or "<div" in html_str.lower() or "svg" in html_str.lower()

    def test_adv_scheduled_v6_deprecations_tombstones(self):
        """Enforce scheduled v6.0.0 deprecation cleanups."""
        # 1. puremacro.lp.garch_utils tombstone
        sys.modules.pop("puremacro.lp.garch_utils", None)
        with pytest.raises(ImportError) as excinfo:
            import puremacro.lp.garch_utils  # noqa: F401
        err_msg = str(excinfo.value)
        assert "6.0.0" in err_msg or "removed" in err_msg.lower()

        # 2. puremacro.sigma.sigma_numpy.SigmaObject removal
        import puremacro.sigma.sigma_numpy as s_numpy
        assert not hasattr(s_numpy, "SigmaObject")

        # 3. Canonical import exists in puremacro.volatility.sigma
        from puremacro.volatility.sigma import SigmaObject
        assert SigmaObject is not None


# ===========================================================================
# 2. Track I.2: SMC Engine Adversarial Stress Tests
# ===========================================================================

class TestTrackI2SMCAdversarial:
    """Adversarial stress testing of Sequential Monte Carlo engine."""

    def test_adv_smc_multimodal_posterior_exploration(self):
        """Stress-test SMC with bimodal Gaussian mixture target under MALA and RW mutations."""
        mu1, mu2 = -2.5, 2.5
        s1, s2 = 0.4, 0.4

        def _bimodal_loglik_and_grad(th: np.ndarray) -> tuple[float, np.ndarray]:
            x = th[0]
            d1 = (x - mu1) / s1
            d2 = (x - mu2) / s2
            c1 = math.exp(-0.5 * d1**2) / (math.sqrt(2 * math.pi) * s1)
            c2 = math.exp(-0.5 * d2**2) / (math.sqrt(2 * math.pi) * s2)
            mix = 0.5 * c1 + 0.5 * c2
            if mix <= 1e-300:
                return -1000.0, np.zeros_like(th)
            ll = math.log(mix)
            grad_x = (0.5 * c1 * (-d1 / s1) + 0.5 * c2 * (-d2 / s2)) / mix
            return ll, np.array([grad_x])

        priors = {"theta": {"dist": "uniform", "lb": -6.0, "ub": 6.0}}

        for mut in ("rw", "mala"):
            res = smc_sample(
                _bimodal_loglik_and_grad,
                priors=priors,
                n_particles=600,
                mutation=mut,
                target_ess=0.5,
                seed=42,
            )
            assert isinstance(res, SMCResult)
            assert np.isfinite(res.log_mdd)
            assert res.mdd_se >= 0.0
            assert np.all(np.isfinite(res.particles))

            # Verify that BOTH modes are populated
            frac_mode1 = np.sum(res.weights[res.particles[:, 0] < 0])
            frac_mode2 = np.sum(res.weights[res.particles[:, 0] > 0])
            assert frac_mode1 > 0.15, f"Mode 1 collapsed under {mut}: frac={frac_mode1}"
            assert frac_mode2 > 0.15, f"Mode 2 collapsed under {mut}: frac={frac_mode2}"

    def test_adv_smc_poor_initialization_distant_prior(self):
        """Stress-test SMC when prior is far away from the likelihood mode."""
        # Prior is N(8.0, 0.5^2), true mode is at 0.0 with std 0.2
        # Likelihood: -0.5 * (theta / 0.2)^2
        def _target_ll_and_grad(th: np.ndarray) -> tuple[float, np.ndarray]:
            diff = th[0] / 0.2
            ll = -0.5 * (diff ** 2)
            grad = -diff / 0.2
            return ll, np.array([grad])

        priors = {"theta": {"dist": "normal", "mean": 8.0, "std": 0.5, "lb": -5.0, "ub": 15.0}}

        # When n_steps is increased to 10, particles traverse the steep gradient down to the mode
        res = smc_sample(
            _target_ll_and_grad,
            priors=priors,
            n_particles=500,
            mutation="mala",
            n_steps=10,
            target_ess=0.5,
            seed=123,
        )
        assert res.stage_tempering[-1] >= 1.0 - 1e-4
        assert np.all(np.isfinite(res.particles))
        final_mean = np.average(res.particles[:, 0], weights=res.weights)
        assert final_mean < 3.0, f"Failed to migrate away from distant prior: final_mean={final_mean}"

    def test_adv_smc_degenerate_weights_and_logsumexp_stability(self):
        """Stress-test AdaptiveTemperingSchedule and logsumexp under extreme loglik differences."""
        sched = AdaptiveTemperingSchedule(target_ess_ratio=0.5, min_step=1e-4)

        # Case 1: Particles with massive negative values and extreme dispersion
        logliks = np.array([-1e7, -5e6, -100.0, -10.0, -0.5, 0.0])
        weights = np.ones(6) / 6.0

        next_phi = sched.next_phi(phi_curr=0.0, loglik_particles=logliks, weights=weights, stage_idx=1)
        assert 0.0 < next_phi <= 1.0
        assert np.isfinite(next_phi)

        # Case 2: Particles with -inf values
        logliks_inf = np.array([-math.inf, -math.inf, -50.0, -20.0, 0.0])
        weights_5 = np.ones(5) / 5.0
        next_phi_inf = sched.next_phi(phi_curr=0.0, loglik_particles=logliks_inf, weights=weights_5, stage_idx=1)
        assert 0.0 < next_phi_inf <= 1.0

    def test_adv_smc_mala_step_size_clamping(self):
        """Verify MALA step size adaptation is strictly clamped within [1e-5, 2.0]."""
        priors = {"theta": {"dist": "normal", "mean": 0.0, "std": 1.0}}

        # Extreme initial step size: 100.0
        res_large = smc_sample(
            lambda th: (-0.5 * th[0]**2, -th),
            priors=priors,
            n_particles=150,
            mutation="mala",
            step_size_init=100.0,
            seed=42,
        )
        final_step = res_large.metadata["final_step_size"]
        assert 1e-5 <= final_step <= 2.0, f"Step size {final_step} escaped [1e-5, 2.0]"

        # Extreme tiny step size: 1e-8
        res_small = smc_sample(
            lambda th: (-0.5 * th[0]**2, -th),
            priors=priors,
            n_particles=150,
            mutation="mala",
            step_size_init=1e-8,
            seed=42,
        )
        final_step_s = res_small.metadata["final_step_size"]
        assert 1e-5 <= final_step_s <= 2.0, f"Step size {final_step_s} escaped [1e-5, 2.0]"

    def test_adv_smc_marginal_data_density_analytical_recovery(self):
        """Empirically verify marginal data density ln p(Y) recovery on Gaussian target."""
        y_obs = 1.0
        exact_log_mdd = -0.5 * (math.log(4.0 * math.pi) + (y_obs ** 2) / 2.0)

        def _gauss_loglik(th: np.ndarray) -> tuple[float, np.ndarray]:
            diff = y_obs - th[0]
            ll = -0.5 * math.log(2.0 * math.pi) - 0.5 * (diff ** 2)
            grad = np.array([diff])
            return ll, grad

        priors = {"theta": {"dist": "normal", "mean": 0.0, "std": 1.0}}
        res = smc_sample(
            _gauss_loglik,
            priors=priors,
            n_particles=1000,
            mutation="mala",
            target_ess=0.5,
            seed=777,
        )
        assert abs(res.log_mdd - exact_log_mdd) < 0.15, (
            f"Estimated log MDD {res.log_mdd:.4f} differs from exact {exact_log_mdd:.4f}"
        )


# ===========================================================================
# 3. Track II.3: Global Nonlinear HANK Adversarial Stress Tests
# ===========================================================================

class TestTrackII3HANKNonlinearAdversarial:
    """Adversarial stress testing of Global Nonlinear HANK transitions and aggregate risk."""

    def test_adv_hank_nonlinear_huge_1000bps_and_1500bps_shocks(self):
        """Stress-test Broyden transition solver under massive 1000 bps and 1500 bps shocks."""
        for shock_val in (0.10, -0.10, 0.15):
            res = solve_hank_nonlinear_transition(
                model=None,
                shocks={"eps_m": shock_val},
                horizon=30,
                max_iter=100,
                tol=1e-8,
            )
            assert res.converged, f"Failed to converge for shock_size={shock_val}, ||H||={res.residual_norm}"
            assert res.residual_norm <= 1e-8
            assert len(res.paths) == 30
            assert np.all(np.isfinite(res.paths.values))

            # Contractionary monetary shock should depress output and spike borrowing binding fraction
            if shock_val > 0:
                assert np.min(res.paths["Y"].values) < 1.0
                assert np.max(res.binding_fractions) > 0.22, "Binding fraction failed to rise above steady-state 0.22"

    def test_adv_hank_nonlinear_2000bps_boundary_limit_finding(self):
        """Empirically document that an ultra-extreme 2000 bps (0.20) shock stalls the Broyden line search.

        Under a 2000 bps shock, the shock drives output down by 23.3% and pushes household
        binding fraction Phi to 0.732, creating a sharp gradient kink where Armijo backtracking
        reaches the step size lower bound and exits after 5 iterations with ||H|| ~ 0.0092.
        """
        res = solve_hank_nonlinear_transition(
            model=None,
            shocks={"eps_m": 0.20},
            horizon=30,
            max_iter=100,
        )
        assert not res.converged, "2000 bps shock was unexpectedly claimed converged"
        assert res.residual_norm < 0.05, f"Residual norm {res.residual_norm} exploded"
        assert res.iterations == 5
        assert np.max(res.binding_fractions) > 0.70

    def test_adv_hank_asymmetric_mpc_amplification(self):
        """Verify contractionary shock produces strictly larger output drop than expansionary shock expansion."""
        shock_size = 0.05  # 500 bps shock
        res_pos = solve_hank_nonlinear_transition(model=None, shocks={"eps_m": shock_size}, horizon=25)
        res_neg = solve_hank_nonlinear_transition(model=None, shocks={"eps_m": -shock_size}, horizon=25)

        assert res_pos.converged and res_neg.converged

        drop_pos = np.max(np.abs(res_pos.paths["Y"].values - 1.0))
        gain_neg = np.max(np.abs(res_neg.paths["Y"].values - 1.0))

        assert res_pos.asymmetry_ratio >= 1.05 or (drop_pos / gain_neg >= 1.05), (
            f"Asymmetric amplification ratio {res_pos.asymmetry_ratio:.3f} failed to exceed 1.05 threshold"
        )

    def test_adv_hank_near_autarky_extreme_borrowing_constraints(self):
        """Stress-test transition under near-autarky parameters (phi_ss = 0.85, mpc_constrained = 0.99)."""
        res = solve_hank_nonlinear_transition(
            model=None,
            shocks={"eps_m": 0.04},
            horizon=25,
            phi_ss=0.85,
            mpc_constrained=0.99,
            mpc_unconstrained=0.05,
            chi_phi=10.0,
            max_iter=120,
            tol=1e-8,
        )
        assert res.converged
        assert res.residual_norm <= 1e-8
        # Binding fraction should remain within physical bounds [0.01, 0.98]
        assert np.all(res.binding_fractions >= 0.01)
        assert np.all(res.binding_fractions <= 0.98)

    def test_adv_hank_broyden_updates_and_jacobian_resets(self):
        """Verify Good Broyden rank-1 Sherman-Morrison updates and Jacobian reuse counters."""
        res = solve_hank_nonlinear_transition(
            model=None,
            shocks={"eps_m": 0.06},
            horizon=25,
            jacobian_update="good_broyden",
        )
        assert res.converged
        assert res.broyden_updates >= 1
        assert res.jacobian_evaluations >= 1
        # Check summary formatting
        summary_df = res.summary()
        assert isinstance(summary_df, pd.DataFrame)
        assert "Converged" in summary_df.loc["Convergence Status", "Value"]

    def test_adv_hank_aggregate_risk_order2_extreme_volatility_quadratic_scaling(self):
        """Empirically verify 2nd-order aggregate risk quadratic scaling across multiple orders of magnitude."""
        volatilities = [1e-4, 1e-3, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25]

        for sig in volatilities:
            res1 = solve_hank_aggregate_risk_order2(sigma_agg=sig)
            res2 = solve_hank_aggregate_risk_order2(sigma_agg=2.0 * sig)

            # 1. Sign checks under prudent preferences
            assert res1.precautionary_shift_wealth > 0.0, f"Wealth shift {res1.precautionary_shift_wealth} must be > 0"
            assert res1.precautionary_shift_rate < 0.0, f"Rate discount {res1.precautionary_shift_rate} must be < 0"

            # 2. Exact quadratic scaling verification: Delta(2*sigma) / Delta(sigma) = 4.0
            ratio_wealth = res2.precautionary_shift_wealth / res1.precautionary_shift_wealth
            ratio_rate = res2.precautionary_shift_rate / res1.precautionary_shift_rate

            np.testing.assert_allclose(ratio_wealth, 4.0, rtol=1e-6, err_msg=f"Wealth scaling failed for sigma={sig}")
            np.testing.assert_allclose(ratio_rate, 4.0, rtol=1e-6, err_msg=f"Rate scaling failed for sigma={sig}")

    def test_adv_hank_aggregate_risk_distribution_moments_supported_dimensions(self):
        """Verify 2nd-order perturbation operates cleanly for n_distribution_moments in {1, 2, 3, 4}."""
        for n_moments in (1, 2, 3, 4):
            res = solve_hank_aggregate_risk_order2(n_distribution_moments=n_moments, sigma_agg=0.01)
            assert isinstance(res, HANKAggregateRiskResult)
            assert len(res.state_names) == n_moments + 1
            assert res.g_xx.shape == (5, (n_moments + 1) ** 2)
            assert res.precautionary_shift_wealth > 0.0
            assert res.precautionary_shift_rate < 0.0

    def test_reiter_moment_reduction_higher_dimension_bounds(self):
        """Verify solve_hank_aggregate_risk_order2 runs successfully and returns positive sigma_shift_A > 0 for m in {1, 2, 4, 5, 6, 8}."""
        for m in (1, 2, 4, 5, 6, 8):
            res = solve_hank_aggregate_risk_order2(n_distribution_moments=m, sigma_agg=0.01)
            assert isinstance(res, HANKAggregateRiskResult)
            assert len(res.state_names) == m + 1
            assert res.g_xx.shape == (5, (m + 1) ** 2)
            assert np.all(np.isfinite(res.g_xx))
            assert np.all(np.isfinite(res.h_xx))
            assert res.sigma_shift_A > 0.0
            assert res.precautionary_shift_wealth > 0.0
            assert res.precautionary_shift_rate < 0.0

    # Backwards compatibility alias for the old adversarial finding test name
    test_adv_hank_aggregate_risk_distribution_moments_dimension_mismatch_finding = test_reiter_moment_reduction_higher_dimension_bounds
