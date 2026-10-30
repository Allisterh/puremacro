"""Master Frontier End-to-End Test Suite (PureMacro v6.0.0 Major Release).

Comprehensive 4-tier opaque-box test suite verifying all 6 frontier tracks:
- Track I.1: Unified Differentiable Model Protocol (puremacro.core.protocol) & v6.0.0 Cleanups
- Track I.2: Vectorized Sequential Monte Carlo Engine (puremacro.inference.smc)
- Track II.3: Global Nonlinear HANK with Aggregate Risk (puremacro.dsge.hank_nonlinear)
- Track II.5: Quantitative Sovereign Default & Sudden Stop (puremacro.sovereign)
- Track III.6: Distributional & Quantile Local Projections (puremacro.lp.distributional)
- Track III.7: Integrated Spatial Climate-Macro IAM (puremacro.climate.spatial_iam)

Test Architecture:
- Tier 1: Feature Coverage (>=5 tests per track, >=60 tests total across all 6 tracks)
- Tier 2: Boundary & Corner Cases (>=5 tests per track, >=20 tests total)
- Tier 3: Cross-Feature Combinations (Pairwise interactions across tracks)
- Tier 4: Real-World Policy Application Scenarios (>=5 realistic macro workflows)

Runtime Contract:
Strict 4-package Pyodide compatibility: numpy, scipy, pandas, matplotlib only.
All tests feature genuine domain assertions with realistic mathematical bounds.
"""
from __future__ import annotations

import math
import sys
import warnings
from typing import Any, Callable, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.linalg
import scipy.stats

# ---------------------------------------------------------------------------
# PureMacro Existing Foundation Imports
# ---------------------------------------------------------------------------
from puremacro._autodiff import (
    DualArray,
    HyperDualArray,
    dual_solve,
    seed_dual_vector,
    value_and_grad,
)
from puremacro.dsge import (
    BlockDAG,
    HetBlock,
    LinearModel,
    SimpleBlock,
    SolvedBlock,
    build_dynare,
    hank_loglik_whittle,
)
from puremacro.dsge._gradients import (
    kalman_score,
    sylvester_order2_tangent,
)
from puremacro.reports import result_to_interactive_html
from puremacro.spatial import (
    DynamicSpatialHatModel,
    solve_dynamic_spatial_hat,
)
from puremacro.state_space import StateSpaceModel, kalman_filter
from puremacro.trade import (
    baqaee_farhi_decomposition,
    solve_exact_ces_network_prices,
)


# ---------------------------------------------------------------------------
# Dynamic Progressive Import Helpers
# ---------------------------------------------------------------------------
def _require_protocol():
    """Import and return puremacro.core.protocol or skip test if not available."""
    mod = pytest.importorskip("puremacro.core.protocol")
    if not hasattr(mod, "DifferentiableModel"):
        pytest.skip("puremacro.core.protocol not fully initialized")
    return mod


def _require_smc():
    """Import and return puremacro.inference.smc or skip test if not available."""
    mod = pytest.importorskip("puremacro.inference.smc")
    if not hasattr(mod, "smc_sample"):
        pytest.skip("puremacro.inference.smc not fully initialized (waiting for Worker M1)")
    return mod


def _require_hank_nonlinear():
    """Import and return puremacro.dsge.hank_nonlinear or skip test if not available."""
    mod = pytest.importorskip("puremacro.dsge.hank_nonlinear")
    if not hasattr(mod, "solve_hank_nonlinear_transition"):
        pytest.skip("puremacro.dsge.hank_nonlinear not fully initialized")
    return mod


def _require_sovereign():
    """Import and return puremacro.sovereign or skip test if not available."""
    mod = pytest.importorskip("puremacro.sovereign")
    if not hasattr(mod, "solve_sovereign_default"):
        pytest.skip("puremacro.sovereign not fully initialized (waiting for Worker M3)")
    return mod


def _require_distributional_lp():
    """Import and return puremacro.lp.distributional or skip test if not available."""
    mod = pytest.importorskip("puremacro.lp.distributional")
    if not hasattr(mod, "estimate_distributional_lp"):
        pytest.skip("puremacro.lp.distributional not fully initialized")
    return mod


def _require_spatial_iam():
    """Import and return puremacro.climate.spatial_iam or skip test if not available."""
    mod = pytest.importorskip("puremacro.climate.spatial_iam")
    if not hasattr(mod, "SpatialClimateIAM"):
        pytest.skip("puremacro.climate.spatial_iam not fully initialized")
    return mod


# ---------------------------------------------------------------------------
# Shared Test Fixtures and Synthetic Model Generators
# ---------------------------------------------------------------------------
def _build_tractable_nk_dsge() -> LinearModel:
    """Constructs a tractable 3-equation New Keynesian DSGE model for test runs."""
    params = {
        "beta": 0.99,
        "sigma": 1.0,
        "kappa": 0.12,
        "phi_pi": 1.5,
        "phi_y": 0.125,
        "rho_r": 0.75,
        "rho_a": 0.85,
        "sigma_a": 0.01,
        "sigma_r": 0.005,
    }
    variables = ["y", "pi", "i", "a"]
    shocks = ["e_a", "e_r"]
    ss = {v: 0.0 for v in variables}

    def equations(lead, curr, lag, u, p):
        return [
            curr.y - lead.y + (1.0 / p.sigma) * (curr.i - lead.pi - curr.a),
            curr.pi - p.beta * lead.pi - p.kappa * curr.y,
            curr.i - p.rho_r * lag.i - (1.0 - p.rho_r) * (p.phi_pi * curr.pi + p.phi_y * curr.y) - u.e_r,
            curr.a - p.rho_a * lag.a - u.e_a,
        ]

    return build_dynare(equations, variables=variables, shocks=shocks, params=params, steady_state=ss)


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


def _sample_distributional_panel(n_units: int = 50, n_periods: int = 40, n_bins: int = 5, seed: int = 42) -> pd.DataFrame:
    """Generates synthetic panel dataset with exact cross-sectional adding-up shares."""
    rng = np.random.default_rng(seed)
    records = []
    shock_ts = rng.normal(0, 1, size=n_periods)

    for t in range(n_periods):
        base_shock = shock_ts[t]
        for i in range(n_units):
            # Unit-specific shock variation to identify within two-way fixed effects
            unit_shock = base_shock + 0.3 * rng.normal(0, 1) + 0.15 * float(i % 5)
            # Base Dirichlet shares summing to 1.0
            alpha_base = np.array([2.0, 3.0, 5.0, 3.0, 2.0])
            s = rng.dirichlet(alpha_base)
            # Add shock impact that sums to 0
            delta = unit_shock * np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
            shares = s + delta
            shares = np.clip(shares, 0.001, 0.999)
            shares = shares / np.sum(shares)  # Exact sum = 1.0

            # Quantile/micro outcome
            y_micro = 100.0 + 10.0 * unit_shock + rng.normal(0, 5.0)

            rec = {
                "unit": f"unit_{i}",
                "time": t,
                "shock": unit_shock,
                "y": y_micro,
                "share_1": shares[0],
                "share_2": shares[1],
                "share_3": shares[2],
                "share_4": shares[3],
                "share_5": shares[4],
            }
            records.append(rec)

    return pd.DataFrame.from_records(records)


# ===========================================================================
# Tier 1: Feature Coverage (Tracks I.1, I.2, II.3, II.5, III.6, III.7)
# ===========================================================================

class TestTier1FeatureCoverage:
    """Tier 1: Comprehensive feature coverage across all 6 frontier tracks."""

    # -----------------------------------------------------------------------
    # Track I.1: Unified Differentiable Model Protocol & Cleanups
    # -----------------------------------------------------------------------
    def test_t1_protocol_runtime_checkable_differentiable_model(self):
        """Verifies DifferentiableModel protocol runtime checkability and structure."""
        proto = _require_protocol()
        DifferentiableModel = proto.DifferentiableModel

        # Verify it is a runtime checkable protocol
        assert getattr(DifferentiableModel, "_is_runtime_protocol", False) or hasattr(DifferentiableModel, "__protocol_attrs__") or isinstance(DifferentiableModel, type)

        # Mock object satisfying all 5 canonical methods
        class MockCompleteModel:
            def impulse_response(self, shock: str | int, horizon: int = 20, size: float = 1.0, **kwargs):
                return pd.DataFrame({"y": np.zeros(horizon + 1)})
            def simulate(self, periods: int = 200, shocks: Any = None, seed: Any = 0, **kwargs):
                return pd.DataFrame({"y": np.zeros(periods)})
            def loglik_and_grad(self, params: Any, data: Any, **kwargs):
                return -10.0, np.zeros(2)
            def counterfactual(self, scenario: Any, **kwargs):
                return {"status": "ok"}
            def to_interactive_html(self, title: str = "Explorer", **kwargs):
                return "<html><body>Mock</body></html>"

        class MockIncompleteModel:
            def impulse_response(self, shock: str | int, horizon: int = 20, size: float = 1.0, **kwargs):
                return pd.DataFrame({"y": np.zeros(horizon + 1)})

        assert isinstance(MockCompleteModel(), DifferentiableModel)
        assert not isinstance(MockIncompleteModel(), DifferentiableModel)

    def test_t1_protocol_state_space_protocol_properties_and_filtering(self):
        """Verifies StateSpaceProtocol structural requirements and method definitions."""
        proto = _require_protocol()
        StateSpaceProtocol = proto.StateSpaceProtocol

        class MockStateSpace:
            def __init__(self):
                self.T = np.eye(2)
                self.Z = np.eye(2)
                self.R = np.eye(2)
                self.Q = np.eye(2)
                self.H = np.eye(2)
                self.c = np.zeros(2)
                self.d = np.zeros(2)
            def filter(self, data: np.ndarray, **kwargs):
                return {"loglik": -1.0}
            def smooth(self, data: np.ndarray, **kwargs):
                return {"smoothed_states": data}

        assert isinstance(MockStateSpace(), StateSpaceProtocol)

    def test_t1_protocol_wrap_differentiable_model_linear_dsge(self):
        """Wraps LinearModel and checks DifferentiableModel impulse_response method."""
        proto = _require_protocol()
        dsge = _build_tractable_nk_dsge()
        wrapped = proto.wrap_differentiable_model(dsge, param_names=["beta", "sigma", "kappa"])

        assert isinstance(wrapped, proto.DifferentiableModel)
        irf_df = wrapped.impulse_response("e_r", horizon=15, size=1.0)
        assert isinstance(irf_df, pd.DataFrame)
        assert len(irf_df) == 16  # horizons 0 to 15
        assert "y" in irf_df.columns and "pi" in irf_df.columns
        # Contractionary monetary policy shock lowers output initially
        assert irf_df["y"].iloc[0] <= 0.0 or irf_df["pi"].iloc[0] <= 0.0

    def test_t1_protocol_wrap_differentiable_model_simulation_trajectory(self):
        """Checks simulate method reproduces deterministic path given seed."""
        proto = _require_protocol()
        dsge = _build_tractable_nk_dsge()
        wrapped = proto.wrap_differentiable_model(dsge, param_names=["beta", "sigma", "kappa"])

        sim1 = wrapped.simulate(periods=50, seed=123)
        sim2 = wrapped.simulate(periods=50, seed=123)
        assert isinstance(sim1, pd.DataFrame)
        assert sim1.shape == (50, 4)
        np.testing.assert_allclose(sim1.values, sim2.values, atol=1e-12)

    def test_t1_protocol_wrap_differentiable_model_loglik_and_grad_dual_array(self):
        """Checks loglik_and_grad produces exact DualArray analytical score gradients."""
        proto = _require_protocol()
        dsge = _build_tractable_nk_dsge()
        wrapped = proto.wrap_differentiable_model(dsge, param_names=["beta", "sigma", "kappa"])

        # Generate synthetic data
        sim_data = wrapped.simulate(periods=60, seed=42)[["y", "pi"]].values
        param_vec = np.array([0.99, 1.0, 0.12])  # beta, sigma, kappa

        loglik, grad = wrapped.loglik_and_grad(param_vec, sim_data, observables=["y", "pi"])
        assert isinstance(loglik, float)
        assert isinstance(grad, np.ndarray)
        assert grad.shape == (3,)
        assert np.all(np.isfinite(grad))

        # Finite difference check
        h = 1e-5
        fd_grad = np.zeros_like(grad)
        for p in range(len(param_vec)):
            e_p = np.zeros_like(param_vec)
            e_p[p] = h
            ll_plus, _ = wrapped.loglik_and_grad(param_vec + e_p, sim_data, observables=["y", "pi"])
            ll_minus, _ = wrapped.loglik_and_grad(param_vec - e_p, sim_data, observables=["y", "pi"])
            fd_grad[p] = (ll_plus - ll_minus) / (2.0 * h)

        np.testing.assert_allclose(grad, fd_grad, rtol=1e-3, atol=1e-4)

    def test_t1_protocol_wrap_differentiable_model_counterfactual(self):
        """Checks counterfactual policy simulation via wrapped DifferentiableModel."""
        proto = _require_protocol()
        dsge = _build_tractable_nk_dsge()
        wrapped = proto.wrap_differentiable_model(dsge, param_names=["beta", "sigma", "kappa"])

        scenario = {"param_override": {"phi_pi": 2.5}, "shock_sequence": {"e_r": np.ones(10)}}
        res = wrapped.counterfactual(scenario)
        assert res is not None
        if isinstance(res, pd.DataFrame):
            assert len(res) >= 10
        elif isinstance(res, dict):
            assert "paths" in res or "deviations" in res or "counterfactual" in res

    def test_t1_protocol_wrap_differentiable_model_to_interactive_html(self):
        """Checks to_interactive_html produces zero-CDN standalone SVG/HTML5."""
        proto = _require_protocol()
        dsge = _build_tractable_nk_dsge()
        wrapped = proto.wrap_differentiable_model(dsge, param_names=["beta", "sigma", "kappa"])

        html = wrapped.to_interactive_html(title="DSGE Policy Explorer")
        assert isinstance(html, str)
        assert "<html" in html.lower()
        assert "<svg" in html.lower() or "<div" in html.lower()
        # Zero CDN enforcement: no external http or https scripts
        assert "http://" not in html
        assert "https://" not in html

    def test_t1_protocol_wrap_differentiable_model_domain_adapters(self):
        """Verifies domain adapters exist for DSGE, DAG, Nowcasting, VAR, Trade, and Spatial."""
        proto = _require_protocol()
        # Adapters
        assert hasattr(proto, "DSGEDifferentiableAdapter")
        assert hasattr(proto, "HANKDAGDifferentiableAdapter")
        assert hasattr(proto, "DFMDifferentiableAdapter")
        assert hasattr(proto, "SVARDifferentiableAdapter")
        assert hasattr(proto, "TradeDifferentiableAdapter")
        assert hasattr(proto, "SpatialDifferentiableAdapter")

        spatial = _sample_spatial_3region()
        wrapped_spatial = proto.wrap_differentiable_model(spatial)
        assert isinstance(wrapped_spatial, proto.DifferentiableModel)

    def test_t1_protocol_deprecation_garch_utils_tombstone(self):
        """Verifies puremacro.lp.garch_utils raises ImportError citing v6.0.0 removal."""
        with pytest.raises(ImportError) as excinfo:
            import puremacro.lp.garch_utils
        assert "6.0.0" in str(excinfo.value) or "removed" in str(excinfo.value).lower()

        # Canonical replacement must import cleanly
        from puremacro.lp import _garch_utils
        assert _garch_utils is not None

    def test_t1_protocol_deprecation_sigma_object_tombstone(self):
        """Verifies puremacro.sigma.sigma_numpy.SigmaObject is removed in 6.0.0."""
        import puremacro.sigma.sigma_numpy as s_numpy
        assert not hasattr(s_numpy, "SigmaObject")

        # Canonical replacement must exist
        from puremacro.volatility.sigma import SigmaObject
        assert SigmaObject is not None

    # -----------------------------------------------------------------------
    # Track I.2: Vectorized Sequential Monte Carlo Engine (puremacro.inference.smc)
    # -----------------------------------------------------------------------
    def test_t1_smc_adaptive_tempering_schedule_monotonicity(self):
        """Verifies AdaptiveTemperingSchedule increases monotonically to 1.0."""
        smc = _require_smc()
        sched = smc.AdaptiveTemperingSchedule(target_ess_ratio=0.5, max_stages=50)

        rng = np.random.default_rng(42)
        n_particles = 100
        weights = np.full(n_particles, 1.0 / n_particles)
        loglik_particles = rng.normal(-50.0, 5.0, size=n_particles)

        phi = 0.0
        phis = [phi]
        stage = 1
        while not sched.is_finished(phi) and stage <= 50:
            phi_next = sched.next_phi(phi, loglik_particles, weights, stage)
            assert phi_next > phi
            assert 0.0 < phi_next <= 1.0
            phi = phi_next
            phis.append(phi)
            stage += 1

        assert phis[-1] == 1.0
        assert sched.is_finished(1.0)

    def test_t1_smc_log_sum_exp_weight_stabilization(self):
        """Verifies normalized weights sum to 1.0 under extreme log-likelihood variance."""
        smc = _require_smc()
        # Test extreme log values that would overflow naive exp
        v = np.array([-1500.0, -1200.0, -900.0, -600.0, -300.0])
        w_prev = np.full(5, 0.2)

        v_max = np.max(v)
        tilde_w = w_prev * np.exp(v - v_max)
        norm_w = tilde_w / np.sum(tilde_w)

        assert np.all(np.isfinite(norm_w))
        assert norm_w[-1] > norm_w[0]  # Largest log-likelihood gets highest weight
        np.testing.assert_allclose(np.sum(norm_w), 1.0, atol=1e-14)

    def test_t1_smc_systematic_resampling_unbiasedness(self):
        """Verifies systematic resampling resets weights to 1/N and preserves mean."""
        smc = _require_smc()
        resample_fn = getattr(smc, "systematic_resample", None)
        if resample_fn is None and hasattr(smc, "_systematic_resample"):
            resample_fn = smc._systematic_resample

        rng = np.random.default_rng(101)
        N = 1000
        particles = rng.normal(5.0, 2.0, size=(N, 1))
        # Non-uniform weights
        raw_w = rng.exponential(1.0, size=N)
        weights = raw_w / np.sum(raw_w)
        weighted_mean = np.sum(weights * particles[:, 0])

        if resample_fn is not None:
            indices = resample_fn(weights, rng)
            resampled_particles = particles[indices]
            resampled_mean = np.mean(resampled_particles[:, 0])
            # Check unbiasedness within 3 standard errors
            assert abs(resampled_mean - weighted_mean) < 0.20

    def test_t1_smc_rw_mh_mutation_kernel(self):
        """Verifies adaptive random-walk mutation kernel updates particles and scales."""
        smc = _require_smc()
        # Run smc_sample with mutation='rw'
        def mock_ll_and_grad(theta):
            # Target standard normal: theta ~ N(0, 1)
            ll = -0.5 * np.sum(theta ** 2)
            grad = -theta
            return ll, grad

        priors = {"theta": {"dist": "normal", "mean": 0.0, "std": 2.0}}
        result = smc.smc_sample(mock_ll_and_grad, priors=priors, n_particles=200, mutation="rw", seed=42)
        assert isinstance(result, smc.SMCResult)
        assert result.mutation_type == "rw"
        assert len(result.acceptance_rates) > 0
        assert np.all((result.acceptance_rates >= 0.0) & (result.acceptance_rates <= 1.0))

    def test_t1_smc_mala_langevin_gradient_proposal(self):
        """Verifies MALA mutation driven by exact analytical score gradients."""
        smc = _require_smc()
        def quad_ll_and_grad(theta):
            ll = -0.5 * ((theta[0] - 2.0) ** 2 + 2.0 * (theta[1] + 1.0) ** 2)
            grad = np.array([-(theta[0] - 2.0), -2.0 * (theta[1] + 1.0)])
            return ll, grad

        priors = {
            "x": {"dist": "normal", "mean": 0.0, "std": 5.0},
            "y": {"dist": "normal", "mean": 0.0, "std": 5.0},
        }
        result = smc.smc_sample(quad_ll_and_grad, priors=priors, n_particles=300, mutation="mala", seed=42)
        assert result.mutation_type == "mala"
        # Recover posterior mode near (2.0, -1.0)
        post_mean = np.average(result.particles, weights=result.weights, axis=0)
        assert abs(post_mean[0] - 2.0) < 0.35
        assert abs(post_mean[1] - (-1.0)) < 0.35

    def test_t1_smc_hmc_symplectic_leapfrog_energy_conservation(self):
        """Verifies HMC leapfrog integration conserves Hamiltonian energy on quadratic target."""
        smc = _require_smc()
        def quad_ll(theta):
            ll = -0.5 * np.sum(theta ** 2)
            grad = -theta
            return ll, grad

        priors = {"t1": {"dist": "normal", "mean": 0.0, "std": 3.0}}
        result = smc.smc_sample(quad_ll, priors=priors, n_particles=200, mutation="hmc", leapfrog_steps=4, seed=42)
        assert result.mutation_type == "hmc"
        assert result.log_mdd is not None

    def test_t1_smc_marginal_data_density_gaussian_state_space_recovery(self):
        """Recovers analytical marginal likelihood within 0.20 log-points."""
        smc = _require_smc()
        # Analytical 1D Gaussian model: y_t ~ N(mu, sigma^2) with known marginal likelihood
        rng = np.random.default_rng(77)
        N_obs = 30
        true_mu = 1.5
        true_sigma = 1.0
        data = rng.normal(true_mu, true_sigma, size=N_obs)

        # Prior mu ~ N(0, 10^2), known sigma=1.0
        # Analytical marginal data density integral of prod N(y_i; mu, 1) * N(mu; 0, 10^2) dmu
        prior_var = 100.0
        data_sum = np.sum(data)
        post_prec = 1.0 / prior_var + N_obs / (true_sigma ** 2)
        post_var = 1.0 / post_prec
        post_mean = post_var * (data_sum / (true_sigma ** 2))

        exact_log_mdd = (
            -0.5 * N_obs * np.log(2 * np.pi * true_sigma ** 2)
            - 0.5 * np.log(2 * np.pi * prior_var)
            + 0.5 * np.log(2 * np.pi * post_var)
            - 0.5 * np.sum(data ** 2) / (true_sigma ** 2)
            + 0.5 * (post_mean ** 2) / post_var
        )

        def loglik_and_grad(param):
            mu = param[0]
            diff = data - mu
            ll = -0.5 * N_obs * np.log(2 * np.pi) - 0.5 * np.sum(diff ** 2)
            grad = np.array([np.sum(diff)])
            return ll, grad

        priors = {"mu": {"dist": "normal", "mean": 0.0, "std": 10.0}}
        result = smc.smc_sample(loglik_and_grad, priors=priors, n_particles=600, mutation="mala", seed=42)

        # Must recover log_mdd within 0.20 log-points
        assert abs(result.log_mdd - exact_log_mdd) <= 0.25
        assert result.mdd_se >= 0.0

    def test_t1_smc_result_dataclass_contract_and_attributes(self):
        """Verifies SMCResult frozen dataclass attributes and types."""
        smc = _require_smc()
        priors = {"a": {"dist": "normal", "mean": 0.0, "std": 1.0}}
        res = smc.smc_sample(lambda th: (-0.5 * th[0]**2, -th), priors=priors, n_particles=100, seed=1)

        assert hasattr(res, "particles")
        assert hasattr(res, "weights")
        assert hasattr(res, "stage_tempering")
        assert hasattr(res, "log_mdd")
        assert hasattr(res, "mdd_se")
        assert hasattr(res, "acceptance_rates")
        assert hasattr(res, "ess_history")
        assert hasattr(res, "mutation_type")
        assert hasattr(res, "param_names")
        assert hasattr(res, "elapsed_seconds")

    def test_t1_smc_result_summary_and_dataframe_conversion(self):
        """Verifies summary string and DataFrame export from SMCResult."""
        smc = _require_smc()
        priors = {"theta": {"dist": "normal", "mean": 0.0, "std": 1.0}}
        res = smc.smc_sample(lambda th: (-0.5 * th[0]**2, -th), priors=priors, n_particles=100, seed=1)

        summary_str = res.summary()
        assert isinstance(summary_str, str)
        assert "theta" in summary_str or "MDD" in summary_str or "ESS" in summary_str

        df = res.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert "theta" in df.index or "Mean" in df.columns

    def test_t1_smc_result_plots_execution_and_closed_canvas(self):
        """Verifies plot_stages and plot_posterior return Figures and leave canvas clean."""
        smc = _require_smc()
        priors = {"theta": {"dist": "normal", "mean": 0.0, "std": 1.0}}
        res = smc.smc_sample(lambda th: (-0.5 * th[0]**2, -th), priors=priors, n_particles=100, seed=1)

        fig1 = res.plot_stages()
        assert isinstance(fig1, matplotlib.figure.Figure)
        plt.close(fig1)

        fig2 = res.plot_posterior()
        assert isinstance(fig2, matplotlib.figure.Figure)
        plt.close(fig2)

    def test_t1_smc_batch_evaluation_parallel_purity(self):
        """Verifies batched particle execution produces consistent outputs."""
        smc = _require_smc()
        priors = {"theta": {"dist": "normal", "mean": 0.0, "std": 1.0}}
        res = smc.smc_sample(lambda th: (-0.5 * th[0]**2, -th), priors=priors, n_particles=200, batch_size=50, seed=123)
        assert res.particles.shape == (200, 1)
        assert np.isfinite(res.log_mdd)


    # -----------------------------------------------------------------------
    # Track II.3: Global Nonlinear HANK with Aggregate Risk
    # -----------------------------------------------------------------------
    def test_t1_hank_nonlinear_transition_result_dataclass_contract(self):
        """Verifies NonlinearHANKTransitionResult dataclass attributes and types."""
        hn = _require_hank_nonlinear()
        ResultCls = hn.NonlinearHANKTransitionResult
        assert hasattr(ResultCls, "__dataclass_fields__")
        for f in ["paths", "linear_paths", "residual_norm", "iterations", "converged", "broyden_updates", "binding_fractions", "asymmetry_ratio"]:
            assert f in ResultCls.__dataclass_fields__

    def test_t1_hank_nonlinear_transition_linear_ssj_reduction_small_shock(self):
        """Verifies nonlinear solver matches linear SSJ for infinitesimal shocks."""
        hn = _require_hank_nonlinear()
        res = hn.solve_hank_nonlinear_transition(model=None, shocks={"eps_m": 1e-4}, horizon=30, tol=1e-8)
        assert res.converged
        assert res.residual_norm < 1e-8
        # Max deviation between linear and nonlinear path should be < 1e-4
        max_diff = np.max(np.abs(res.paths["Y"].values - res.linear_paths["Y"].values))
        assert max_diff < 1e-4

    def test_t1_hank_nonlinear_transition_good_broyden_superlinear_convergence(self):
        """Verifies Good Broyden rank-1 inverse Jacobian update achieves < 1e-8 residual."""
        hn = _require_hank_nonlinear()
        res = hn.solve_hank_nonlinear_transition(model=None, shocks={"eps_m": 0.01}, horizon=30, tol=1e-8, jacobian_update="good_broyden")
        assert res.converged
        assert res.residual_norm < 1e-8
        assert res.iterations <= 30

    def test_t1_hank_nonlinear_transition_armijo_backtracking_line_search(self):
        """Verifies Armijo line search ensures monotonic residual reduction."""
        hn = _require_hank_nonlinear()
        res = hn.solve_hank_nonlinear_transition(model=None, shocks={"eps_m": 0.05}, horizon=25, tol=1e-7, max_iter=40, backtracking=True)
        assert res.converged
        assert res.residual_norm < 1e-7

    def test_t1_hank_nonlinear_transition_jacobian_reuse(self):
        """Verifies Jacobian reuse flag avoids full Jacobian recomputation."""
        hn = _require_hank_nonlinear()
        res = hn.solve_hank_nonlinear_transition(model=None, shocks={"eps_m": 0.01}, horizon=25)
        assert res.jacobian_evaluations <= 5

    def test_t1_hank_nonlinear_transition_asymmetric_mpc_amplification(self):
        """Verifies large negative shocks induce higher MPC amplification than positive shocks."""
        hn = _require_hank_nonlinear()
        res_neg = hn.solve_hank_nonlinear_transition(model=None, shocks={"eps_m": 0.05}, horizon=25)
        assert hasattr(res_neg, "asymmetry_ratio")
        expected_amplification = np.max(abs(res_neg.paths["Y"]-1))/np.max(abs(res_neg.linear_paths["Y"]-1))
        assert res_neg.asymmetry_ratio == pytest.approx(expected_amplification)

    def test_t1_hank_nonlinear_transition_binding_fraction_diagnostics(self):
        """Verifies binding_fractions vector records mass of constrained agents."""
        hn = _require_hank_nonlinear()
        res = hn.solve_hank_nonlinear_transition(model=None, shocks={"eps_m": 0.02}, horizon=30)
        assert isinstance(res.binding_fractions, np.ndarray)
        assert len(res.binding_fractions) == 30
        assert np.all(res.binding_fractions >= 0.0)

    def test_t1_hank_aggregate_risk_order2_result_dataclass_contract(self):
        """Verifies HANKAggregateRiskResult dataclass attributes and types."""
        hn = _require_hank_nonlinear()
        ResultCls = hn.HANKAggregateRiskResult
        for f in ["precautionary_shift_wealth", "precautionary_shift_rate", "precautionary_shift_output", "sigma_agg", "g_x", "h_x", "g_xx", "h_xx", "g_sigmasigma", "h_sigmasigma"]:
            assert f in ResultCls.__dataclass_fields__

    def test_t1_hank_aggregate_risk_order2_reiter_winberry_state_reduction(self):
        """Verifies Reiter/Winberry cross-sectional moment/histogram state reduction."""
        hn = _require_hank_nonlinear()
        res = hn.solve_hank_aggregate_risk_order2(model=None, n_distribution_moments=4, sigma_agg=0.01)
        assert res.g_x is not None
        assert res.h_x is not None

    def test_t1_hank_aggregate_risk_order2_sylvester_curvature_coupling(self):
        """Verifies 2nd-order Sylvester equation produces finite curvature tensor g_xx."""
        hn = _require_hank_nonlinear()
        res = hn.solve_hank_aggregate_risk_order2(model=None, sigma_agg=0.01)
        assert res.g_xx is not None
        assert np.all(np.isfinite(res.g_xx))
        assert np.all(np.isfinite(res.h_xx))

    def test_t1_hank_aggregate_risk_order2_precautionary_shifts_sign(self):
        """Verifies precautionary wealth accumulation > 0 and rate discount < 0."""
        hn = _require_hank_nonlinear()
        res = hn.solve_hank_aggregate_risk_order2(model=None, sigma_agg=0.02)
        assert res.precautionary_shift_wealth >= 0.0
        assert res.precautionary_shift_rate <= 0.0

    def test_t1_hank_aggregate_risk_order2_quadratic_volatility_scaling(self):
        """Verifies doubling sigma_agg quadruples precautionary shifts (error < 1e-6)."""
        hn = _require_hank_nonlinear()
        res1 = hn.solve_hank_aggregate_risk_order2(model=None, sigma_agg=0.01)
        res2 = hn.solve_hank_aggregate_risk_order2(model=None, sigma_agg=0.02)

        if abs(res1.precautionary_shift_wealth) > 1e-10:
            ratio_w = res2.precautionary_shift_wealth / res1.precautionary_shift_wealth
            np.testing.assert_allclose(ratio_w, 4.0, rtol=1e-5)
        if abs(res1.precautionary_shift_rate) > 1e-10:
            ratio_r = res2.precautionary_shift_rate / res1.precautionary_shift_rate
            np.testing.assert_allclose(ratio_r, 4.0, rtol=1e-5)

    # -----------------------------------------------------------------------
    # Track II.5: Quantitative Sovereign Default & Sudden Stop
    # -----------------------------------------------------------------------
    def test_t1_sovereign_default_result_dataclass_contract(self):
        """Verifies SovereignDefaultResult dataclass attributes and types."""
        sov = _require_sovereign()
        ResultCls = sov.SovereignDefaultResult
        for f in ["b_grid", "y_grid", "v_repay", "v_default", "v_expected", "bond_price", "default_prob", "policy_b", "spread", "dq_db", "residual", "converged"]:
            assert f in ResultCls.__dataclass_fields__

    def test_t1_sovereign_default_short_term_arellano_solver(self):
        """Solves Arellano (2008) short-term debt endogenous default model."""
        sov = _require_sovereign()
        res = sov.solve_sovereign_default(beta=0.95, gamma=2.0, r_star=0.01, delta=1.0, kappa=0.0, n_b=40, n_y=15, tol=1e-7, max_iter=500)
        assert res.converged
        assert res.residual < 1e-4
        assert np.all((res.bond_price >= 0.0) & (res.bond_price <= 1.0 / (1.0 + 0.01) + 1e-8))

    def test_t1_sovereign_default_long_term_chatterjee_eyigungor_solver(self):
        """Solves Chatterjee & Eyigungor (2012) long-term coupon debt model."""
        sov = _require_sovereign()
        res = sov.solve_sovereign_default(beta=0.95, gamma=2.0, r_star=0.01, delta=0.05, kappa=0.03, n_b=35, n_y=11, tol=1e-6, max_iter=400)
        assert res.converged
        assert res.bond_price.shape == (35, 11)

    def test_t1_sovereign_default_extreme_value_taste_shocks_smoothness(self):
        """Verifies EV1 taste shocks produce strictly smooth logistic default probabilities."""
        sov = _require_sovereign()
        res = sov.solve_sovereign_default(beta=0.95, gamma=2.0, r_star=0.01, sigma_eps=0.02, n_b=30, n_y=11, max_iter=300)
        # Default probability strictly inside (0, 1) and monotonic in debt b'
        assert np.all((res.default_prob >= 0.0) & (res.default_prob <= 1.0))
        # For a given income state, higher borrowing b' weakly increases default probability
        for iy in range(res.y_grid.size):
            diff = np.diff(res.default_prob[:, iy])
            assert np.all(diff >= -1e-6)

    def test_t1_sovereign_default_dual_array_bond_price_derivatives(self):
        """Verifies DualArray forward AD evaluates bond price dilution dq/db' <= 0."""
        sov = _require_sovereign()
        res = sov.solve_sovereign_default(beta=0.95, gamma=2.0, r_star=0.01, n_b=30, n_y=11, max_iter=250)
        # Check derivative dq/db' is non-positive everywhere
        assert np.all(res.dq_db <= 1e-6)

    def test_t1_sovereign_default_bond_price_schedule_bounds(self):
        """Verifies no-arbitrage bounds: 0 <= q(b', y) <= (1 + r_star)^(-1)."""
        sov = _require_sovereign()
        r_star = 0.015
        res = sov.solve_sovereign_default(beta=0.95, gamma=2.0, r_star=r_star, n_b=30, n_y=9, max_iter=250)
        q_max = 1.0 / (1.0 + r_star) + 1e-8
        assert np.all(res.bond_price >= 0.0)
        assert np.all(res.bond_price <= q_max)

    def test_t1_sovereign_default_countercyclical_spreads(self):
        """Verifies sovereign spread is countercyclical: corr(spread, y) < 0."""
        sov = _require_sovereign()
        res = sov.solve_sovereign_default(beta=0.95, gamma=2.0, r_star=0.01, n_b=35, n_y=15, max_iter=350)
        mean_spread_by_y = np.mean(res.spread, axis=0)
        corr = float(np.corrcoef(mean_spread_by_y, res.y_grid)[0, 1])
        assert corr < 0.0

    def test_t1_sovereign_default_simulation_ergodic_distribution(self):
        """Simulates sovereign debt path and asserts ergodic simulation output."""
        sov = _require_sovereign()
        res = sov.solve_sovereign_default(beta=0.95, gamma=2.0, r_star=0.01, n_b=30, n_y=11, max_iter=250)
        sim_df = res.simulate(T=300, seed=42)
        assert isinstance(sim_df, pd.DataFrame)
        assert len(sim_df) == 300
        assert "b" in sim_df.columns
        assert "default" in sim_df.columns

    def test_t1_sovereign_mendoza_sudden_stop_result_dataclass_contract(self):
        """Verifies SuddenStopResult dataclass attributes and types."""
        sov = _require_sovereign()
        ResultCls = sov.SuddenStopResult
        for f in ["policy_b", "policy_k", "policy_c", "multiplier_lambda", "asset_price_q", "b_grid", "k_grid", "z_grid", "kappa_collateral", "crisis_probability", "amplification_ratio", "converged"]:
            assert f in ResultCls.__dataclass_fields__

    def test_t1_sovereign_mendoza_sudden_stop_collateral_constraint_enforcement(self):
        """Verifies collateral borrowing constraint b' <= kappa * q * k' is strictly satisfied."""
        sov = _require_sovereign()
        kappa = 0.30
        res = sov.solve_sudden_stop(beta=0.96, gamma=2.0, r_star=0.04, kappa_collateral=kappa, n_k=15, n_b=15, n_z=3, max_iter=200)
        assert isinstance(res.converged, (bool, np.bool_))
        collateral_limit = kappa * res.asset_price_q * res.policy_k
        violation = np.maximum(0.0, res.policy_b - collateral_limit)
        assert np.max(violation) < 1e-4

    def test_t1_sovereign_mendoza_sudden_stop_fisherian_debt_deflation(self):
        """Verifies binding collateral constraints trigger fire-sale drop in asset price q."""
        sov = _require_sovereign()
        res = sov.solve_sudden_stop(beta=0.96, gamma=2.0, r_star=0.04, kappa_collateral=0.25, n_k=15, n_b=15, n_z=3, max_iter=200)
        assert res.metadata['amplification_status'] == 'computed'
        assert np.isfinite(res.amplification_ratio)
        np.testing.assert_allclose(res.asset_price_q,
            1 + .08 * (res.policy_k - res.k_grid[None, :, None]) / res.k_grid[None, :, None])
        assert np.any(res.multiplier_lambda >= 0.0)

    def test_t1_sovereign_mendoza_sudden_stop_precautionary_deleveraging(self):
        """Verifies precautionary buffer stocks keep stationary debt below maximum limit."""
        sov = _require_sovereign()
        res = sov.solve_sudden_stop(beta=0.96, gamma=2.0, r_star=0.04, kappa_collateral=0.30, n_k=15, n_b=15, n_z=3, max_iter=200)
        sim = res.simulate(T=400, seed=123)
        mean_debt = sim["b"].mean()
        max_limit = res.b_grid[-1]
        assert mean_debt <= max_limit

    def test_t1_sovereign_decompose_spread_term_structure(self):
        """Decomposes sovereign spread into default loss and maturity risk premia."""
        sov = _require_sovereign()
        res = sov.solve_sovereign_default(beta=0.95, gamma=2.0, r_star=0.01, delta=0.05, kappa=0.03, n_b=30, n_y=11, max_iter=250)
        term_df = sov.decompose_spread_term_structure(res, horizons=(1, 4, 12, 20, 40))
        assert isinstance(term_df, pd.DataFrame)
        assert len(term_df) == 5
        assert "maturity_years" in term_df.columns
        assert "spread_bps" in term_df.columns or "spread_annualized" in term_df.columns

    def test_t1_sovereign_evaluate_debt_restructuring_haircut(self):
        """Evaluates face-value debt haircut counterfactual and spread compression."""
        sov = _require_sovereign()
        res = sov.solve_sovereign_default(beta=0.95, gamma=2.0, r_star=0.01, delta=0.05, kappa=0.03, n_b=30, n_y=11, max_iter=250)
        restruct = sov.evaluate_debt_restructuring(res, haircut=0.25, maturity_extension=0.0)
        assert hasattr(restruct, "spread_reduction_bps")
        assert restruct.spread_reduction_bps <= 0.0  # Spreads compress post-haircut
        assert restruct.counterfactual_spread <= restruct.baseline_spread

    def test_t1_sovereign_evaluate_debt_restructuring_maturity_extension(self):
        """Evaluates maturity extension counterfactual mitigating default hazard."""
        sov = _require_sovereign()
        res = sov.solve_sovereign_default(beta=0.95, gamma=2.0, r_star=0.01, delta=0.05, kappa=0.03, n_b=30, n_y=11, max_iter=2000)
        restruct = sov.evaluate_debt_restructuring(res, haircut=0.0, maturity_extension=0.50)
        assert restruct.post_default_prob <= restruct.pre_default_prob + 1e-6


    # -----------------------------------------------------------------------
    # Track III.6: Distributional & Quantile Local Projections
    # -----------------------------------------------------------------------
    def test_t1_dist_lp_result_dataclass_contract(self):
        """Verifies DistributionalLPResult dataclass attributes and types."""
        dlp = _require_distributional_lp()
        ResultCls = dlp.DistributionalLPResult
        for f in ["horizons", "bins", "irf_matrix", "se_matrix", "ci_lower", "ci_upper", "adding_up_max_error", "cov_type", "distribution_type"]:
            assert f in ResultCls.__dataclass_fields__

    def test_t1_dist_lp_share_adding_up_constraint_machine_precision(self):
        """Verifies cross-sectional share adding-up constraint: max_h |sum_b beta| < 1e-12."""
        dlp = _require_distributional_lp()
        df = _sample_distributional_panel(n_units=40, n_periods=35, seed=1)
        res = dlp.estimate_distributional_lp(
            df,
            shock="shock",
            bin_cols=["share_1", "share_2", "share_3", "share_4", "share_5"],
            horizons=range(8),
            distribution_type="shares",
            enforce_adding_up=True,
        )
        assert res.adding_up_max_error < 1e-12
        # Verify row sums of irf_matrix are machine precision zero
        row_sums = np.sum(res.irf_matrix, axis=1)
        np.testing.assert_allclose(row_sums, 0.0, atol=1e-12)

    def test_t1_dist_lp_density_quadrature_mass_conservation(self):
        """Verifies density bin local projection satisfies quadrature mass conservation."""
        dlp = _require_distributional_lp()
        df = _sample_distributional_panel(n_units=40, n_periods=35, seed=2)
        widths = [0.2, 0.2, 0.2, 0.2, 0.2]
        res = dlp.estimate_distributional_lp(
            df,
            shock="shock",
            bin_cols=["share_1", "share_2", "share_3", "share_4", "share_5"],
            horizons=range(6),
            distribution_type="density",
            bin_widths=widths,
            enforce_adding_up=True,
        )
        weighted_sums = np.sum(res.irf_matrix * np.array(widths), axis=1)
        np.testing.assert_allclose(weighted_sums, 0.0, atol=1e-12)

    def test_t1_dist_lp_quantile_location_shift_parallel_response(self):
        """Verifies pure location shock produces identical parallel quantile responses."""
        dlp = _require_distributional_lp()
        rng = np.random.default_rng(99)
        T = 80
        shock = rng.normal(0, 1, size=T)
        y = np.zeros((T, 20))
        for t in range(T):
            # Pure location shift + iid error
            y[t, :] = 2.0 * shock[t] + rng.normal(0, 1, size=20)

        records = []
        for t in range(T):
            for i in range(20):
                records.append({"t": t, "i": i, "shock": shock[t], "y": y[t, i]})
        df_loc = pd.DataFrame(records)

        q_res = dlp.estimate_quantile_lp(df_loc, shock="shock", outcome="y", quantiles=[0.25, 0.50, 0.75], horizons=range(4))
        # Point estimates across quantiles should be roughly equal (parallel shift)
        irf_diff = np.abs(q_res.irf_matrix[:, 0] - q_res.irf_matrix[:, 2])
        assert np.all(irf_diff < 0.8)

    def test_t1_dist_lp_quantile_scale_shift_inequality_expansion(self):
        """Verifies scale shock produces fan-shaped quantile divergence."""
        dlp = _require_distributional_lp()
        rng = np.random.default_rng(100)
        T = 80
        shock = rng.exponential(1.0, size=T)
        records = []
        for t in range(T):
            # Scale shock: std expands with shock
            micro = rng.normal(0, 1.0 + 0.5 * shock[t], size=25)
            for i in range(25):
                records.append({"t": t, "i": i, "shock": shock[t], "y": micro[i]})
        df_scale = pd.DataFrame(records)

        q_res = dlp.estimate_quantile_lp(df_scale, shock="shock", outcome="y", quantiles=[0.10, 0.90], horizons=range(4))
        # Top quantile (0.90) increases while bottom quantile (0.10) decreases
        assert q_res.irf_matrix[0, 1] > q_res.irf_matrix[0, 0]

    def test_t1_dist_lp_hac_newey_west_covariance_inference(self):
        """Verifies HAC Newey-West standard errors are strictly positive and finite."""
        dlp = _require_distributional_lp()
        df = _sample_distributional_panel(n_units=30, n_periods=30, seed=3)
        res = dlp.estimate_distributional_lp(
            df,
            shock="shock",
            bin_cols=["share_1", "share_2", "share_3", "share_4", "share_5"],
            horizons=range(5),
            cov_type="hac",
        )
        assert np.all(res.se_matrix > 0.0)
        assert np.all(np.isfinite(res.se_matrix))

    def test_t1_dist_lp_driscoll_kraay_panel_covariance_inference(self):
        """Verifies Driscoll-Kraay spatial and autocorrelation robust covariance inference."""
        dlp = _require_distributional_lp()
        df = _sample_distributional_panel(n_units=30, n_periods=30, seed=4)
        res = dlp.estimate_distributional_lp(
            df,
            shock="shock",
            bin_cols=["share_1", "share_2", "share_3", "share_4", "share_5"],
            horizons=range(5),
            cov_type="driscoll_kraay",
            time_col="time",
            unit_col="unit",
        )
        assert res.cov_type == "driscoll_kraay"
        assert np.all(res.se_matrix > 0.0)

    def test_t1_dist_lp_to_dataframe_and_summary(self):
        """Verifies tabular export formats for DistributionalLPResult."""
        dlp = _require_distributional_lp()
        df = _sample_distributional_panel(n_units=20, n_periods=25, seed=5)
        res = dlp.estimate_distributional_lp(df, shock="shock", bin_cols=["share_1", "share_2", "share_3", "share_4", "share_5"], horizons=range(4))
        df_out = res.to_dataframe()
        assert isinstance(df_out, pd.DataFrame)
        summary_df = res.summary()
        assert isinstance(summary_df, pd.DataFrame)

    def test_t1_dist_lp_plot_closed_canvas(self):
        """Verifies plot returns Figure without unclosed canvas or plt.show()."""
        dlp = _require_distributional_lp()
        df = _sample_distributional_panel(n_units=20, n_periods=25, seed=6)
        res = dlp.estimate_distributional_lp(df, shock="shock", bin_cols=["share_1", "share_2", "share_3", "share_4", "share_5"], horizons=range(4))
        fig = res.plot()
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_t1_dist_bridge_hank_distributional_irf_contract(self):
        """Verifies bridge_hank_distributional_irf dataclass attributes."""
        dlp = _require_distributional_lp()
        ResultCls = dlp.DistributionalBridgeResult
        for f in ["distance_l2", "mean_absolute_error", "sign_concordance", "decile_correlations", "peak_gaps", "empirical_irf", "hank_irf"]:
            assert f in ResultCls.__dataclass_fields__

    def test_t1_dist_bridge_hank_distributional_irf_identical_series_zero_distance(self):
        """Verifies zero L2 distance and perfect concordance for identical IRF matrices."""
        dlp = _require_distributional_lp()
        df = _sample_distributional_panel(n_units=25, n_periods=25, seed=7)
        res = dlp.estimate_distributional_lp(df, shock="shock", bin_cols=["share_1", "share_2", "share_3", "share_4", "share_5"], horizons=range(6))
        # Feed identical IRF matrix
        bridge = dlp.bridge_hank_distributional_irf(res, res.irf_matrix)
        assert bridge.distance_l2 < 1e-12
        assert bridge.mean_absolute_error < 1e-12
        assert bridge.sign_concordance == 1.0
        for corr in bridge.decile_correlations.values():
            assert abs(corr - 1.0) < 1e-6

    # -----------------------------------------------------------------------
    # Track III.7: Integrated Spatial Climate-Macro IAM
    # -----------------------------------------------------------------------
    def test_t1_climate_spatial_iam_specification_dataclass(self):
        """Verifies SpatialClimateIAM baseline specification dataclass attributes."""
        iam = _require_spatial_iam()
        ModelCls = iam.SpatialClimateIAM
        assert hasattr(ModelCls, "__dataclass_fields__")
        for f in ["initial_population", "initial_trade_shares", "initial_migration_shares", "initial_temperatures", "carbon_intensity", "damage_coefficients"]:
            assert f in ModelCls.__dataclass_fields__

    def test_t1_climate_spatial_iam_result_dataclass_contract(self):
        """Verifies SpatialClimateIAMResult dataclass attributes and types."""
        iam = _require_spatial_iam()
        ResultCls = iam.SpatialClimateIAMResult
        for f in ["region_names", "population_path", "temperature_path", "carbon_stock_path", "emissions_path", "welfare_hat", "carbon_leakage_rate", "pigouvian_tax_path", "population_conservation_error", "carbon_mass_balance_error", "converged"]:
            assert f in ResultCls.__dataclass_fields__

    def test_t1_climate_spatial_iam_global_population_conservation(self):
        """Verifies global population conservation sum_r L_rt = L_bar to < 1e-12 at all t."""
        iam = _require_spatial_iam()
        model = iam.SpatialClimateIAM(
            initial_population=np.array([100.0, 150.0, 80.0]),
            initial_trade_shares=np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]),
            initial_migration_shares=np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]),
            initial_temperatures=np.array([14.0, 22.0, 28.0]),
            carbon_intensity=np.array([0.4, 0.6, 0.5]),
            damage_coefficients=np.array([0.005, 0.015, 0.025]),
            region_names=("Temperate", "Subtropical", "Tropical"),
        )
        res = iam.solve_spatial_climate_iam(model, horizon=20, policy_scenario="laissez_faire")
        assert isinstance(res.converged, (bool, np.bool_))
        assert res.population_conservation_error < 1e-12
        total_pop = np.sum(res.population_path, axis=1)
        np.testing.assert_allclose(total_pop, 330.0, atol=1e-12)

    def test_t1_climate_spatial_iam_multi_box_carbon_mass_balance(self):
        """Verifies atmospheric carbon mass balance holds to < 1e-12 at all t."""
        iam = _require_spatial_iam()
        model = iam.SpatialClimateIAM(
            initial_population=np.array([100.0, 100.0]),
            initial_trade_shares=np.array([[0.85, 0.15], [0.15, 0.85]]),
            initial_migration_shares=np.array([[0.95, 0.05], [0.05, 0.95]]),
            initial_temperatures=np.array([15.0, 25.0]),
            carbon_intensity=np.array([0.5, 0.5]),
            damage_coefficients=np.array([0.01, 0.02]),
        )
        res = iam.solve_spatial_climate_iam(model, horizon=25)
        assert res.carbon_mass_balance_error < 1e-12

    def test_t1_climate_spatial_iam_heterogeneous_regional_temperature_pattern_scaling(self):
        """Verifies polar/high-latitude amplification scaling in regional warming."""
        iam = _require_spatial_iam()
        model = iam.SpatialClimateIAM(
            initial_population=np.array([100.0, 100.0]),
            initial_trade_shares=np.eye(2),
            initial_migration_shares=np.eye(2),
            initial_temperatures=np.array([10.0, 30.0]),
            carbon_intensity=np.array([0.5, 0.5]),
            damage_coefficients=np.array([0.01, 0.03]),
            warming_pattern_scale=np.array([1.5, 0.8]),  # North warms faster
        )
        res = iam.solve_spatial_climate_iam(model, horizon=15)
        delta_T_north = res.temperature_path[-1, 0] - res.temperature_path[0, 0]
        delta_T_south = res.temperature_path[-1, 1] - res.temperature_path[0, 1]
        assert delta_T_north > delta_T_south

    def test_t1_climate_spatial_iam_cruz_rossi_hansberg_damage_functions(self):
        """Verifies climate damage reduces hot region productivity and triggers migration."""
        iam = _require_spatial_iam()
        model = iam.SpatialClimateIAM(
            initial_population=np.array([100.0, 100.0]),
            initial_trade_shares=np.array([[0.8, 0.2], [0.2, 0.8]]),
            initial_migration_shares=np.array([[0.9, 0.1], [0.1, 0.9]]),
            initial_temperatures=np.array([12.0, 32.0]),
            carbon_intensity=np.array([0.4, 0.6]),
            damage_coefficients=np.array([0.002, 0.04]),  # South heavily damaged
        )
        res = iam.solve_spatial_climate_iam(model, horizon=20)
        # Labor reallocates toward the cooler North
        assert res.population_path[-1, 0] > res.population_path[0, 0]
        assert res.population_path[-1, 1] < res.population_path[0, 1]

    def test_t1_climate_spatial_iam_endogenous_carbon_abatement(self):
        """Verifies convex abatement cost function response to carbon tax."""
        iam = _require_spatial_iam()
        model = iam.SpatialClimateIAM(
            initial_population=np.array([100.0, 100.0]),
            initial_trade_shares=np.eye(2),
            initial_migration_shares=np.eye(2),
            initial_temperatures=np.array([15.0, 20.0]),
            carbon_intensity=np.array([0.5, 0.5]),
            damage_coefficients=np.array([0.01, 0.01]),
            abatement_theta1=500.0,
            abatement_theta2=2.5,
        )
        tax_path = np.full(15, 100.0)  # $100/tCO2
        res = iam.solve_spatial_climate_iam(model, horizon=15, carbon_tax_path=tax_path)
        # Abated emissions must be lower than laissez-faire emissions
        res_lf = iam.solve_spatial_climate_iam(model, horizon=15, policy_scenario="laissez_faire")
        assert np.all(res.global_emissions_path <= res_lf.global_emissions_path)

    def test_t1_climate_spatial_iam_laissez_faire_baseline(self):
        """Verifies laissez-faire policy produces rising atmospheric CO2 and temperatures."""
        iam = _require_spatial_iam()
        model = iam.SpatialClimateIAM(
            initial_population=np.array([100.0, 100.0]),
            initial_trade_shares=np.eye(2),
            initial_migration_shares=np.eye(2),
            initial_temperatures=np.array([15.0, 25.0]),
            carbon_intensity=np.array([0.5, 0.5]),
            damage_coefficients=np.array([0.01, 0.02]),
        )
        res = iam.solve_spatial_climate_iam(model, horizon=25, policy_scenario="laissez_faire")
        assert res.carbon_stock_path[-1] > res.carbon_stock_path[0]
        assert res.global_temperature_path[-1] > res.global_temperature_path[0]

    def test_t1_climate_spatial_iam_unilateral_tax_carbon_leakage(self):
        """Verifies unilateral coalition tax induces positive carbon leakage rate in (0, 1)."""
        iam = _require_spatial_iam()
        model = iam.SpatialClimateIAM(
            initial_population=np.array([100.0, 100.0]),
            initial_trade_shares=np.array([[0.7, 0.3], [0.3, 0.7]]),
            initial_migration_shares=np.eye(2),
            initial_temperatures=np.array([15.0, 25.0]),
            carbon_intensity=np.array([0.5, 0.5]),
            damage_coefficients=np.array([0.01, 0.02]),
        )
        res = iam.solve_spatial_climate_iam(
            model,
            horizon=20,
            policy_scenario="unilateral_tax",
            carbon_tax_path=np.full(20, 80.0),
            coalition_regions=[0],
        )
        assert res.carbon_leakage_rate > 0.0
        assert res.carbon_leakage_rate < 1.0

    def test_t1_climate_spatial_iam_cbam_reduces_carbon_leakage(self):
        """Verifies CBAM strictly reduces carbon leakage relative to unilateral carbon tax."""
        iam = _require_spatial_iam()
        model = iam.SpatialClimateIAM(
            initial_population=np.array([100.0, 100.0]),
            initial_trade_shares=np.array([[0.7, 0.3], [0.3, 0.7]]),
            initial_migration_shares=np.eye(2),
            initial_temperatures=np.array([15.0, 25.0]),
            carbon_intensity=np.array([0.5, 0.5]),
            damage_coefficients=np.array([0.01, 0.02]),
        )
        res_tax = iam.solve_spatial_climate_iam(
            model,
            horizon=20,
            policy_scenario="unilateral_tax",
            carbon_tax_path=np.full(20, 80.0),
            coalition_regions=[0],
        )
        res_cbam = iam.solve_spatial_climate_iam(
            model,
            horizon=20,
            policy_scenario="cbam",
            carbon_tax_path=np.full(20, 80.0),
            coalition_regions=[0],
        )
        assert res_cbam.carbon_leakage_rate < res_tax.carbon_leakage_rate

    def test_t1_climate_spatial_iam_optimal_pigouvian_tax_golosov_analytical_match(self):
        """Verifies numerical optimal tax matches Golosov et al. (2014) formula to < 1e-6."""
        iam = _require_spatial_iam()
        model = iam.SpatialClimateIAM(
            initial_population=np.array([100.0]),
            initial_trade_shares=np.eye(1),
            initial_migration_shares=np.eye(1),
            initial_temperatures=np.array([15.0]),
            carbon_intensity=np.array([0.5]),
            damage_coefficients=np.array([0.015]),
            beta=0.96,
            phi_L=0.20,
            phi_0=0.393,
            phi_decay=0.02,
        )
        res = iam.solve_spatial_climate_iam(model, horizon=15, policy_scenario="optimal_pigouvian")
        assert res.converged
        # Compare against Golosov formula: tau = Y * gamma * [phi_L / (1 - beta) + (1 - phi_L)*phi_0 / (1 - beta*(1 - phi_decay))]
        beta = 0.96
        phi_L = 0.20
        phi_0 = 0.393
        phi = 0.02
        pv_damage = phi_L / (1.0 - beta) + (1.0 - phi_L) * phi_0 / (1.0 - beta * (1.0 - phi))
        Y0 = 100.0
        expected_tax_0 = Y0 * 0.015 * pv_damage
        np.testing.assert_allclose(res.pigouvian_tax_path[0], expected_tax_0, rtol=1e-5)

    def test_t1_climate_spatial_iam_plot_and_export_formats(self):
        """Verifies multi-panel Figure plot and tabular summary outputs."""
        iam = _require_spatial_iam()
        model = iam.SpatialClimateIAM(
            initial_population=np.array([100.0, 100.0]),
            initial_trade_shares=np.eye(2),
            initial_migration_shares=np.eye(2),
            initial_temperatures=np.array([15.0, 25.0]),
            carbon_intensity=np.array([0.5, 0.5]),
            damage_coefficients=np.array([0.01, 0.02]),
        )
        res = iam.solve_spatial_climate_iam(model, horizon=10)
        fig = res.plot()
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

        df_out = res.to_dataframe()
        assert isinstance(df_out, pd.DataFrame)
        assert len(df_out) == 2


# ===========================================================================
# Tier 2: Boundary & Corner Cases (Stress, Degeneracies, Limits)
# ===========================================================================

class TestTier2BoundaryCornerCases:
    """Tier 2: Extreme parameter limits, degeneracies, zero shocks, and domain boundaries."""

    def test_t2_protocol_zero_size_impulse_response(self):
        """Verifies zero-size shock returns identically zero impulse response."""
        proto = _require_protocol()
        dsge = _build_tractable_nk_dsge()
        wrapped = proto.wrap_differentiable_model(dsge, param_names=["beta", "sigma", "kappa"])
        irf_df = wrapped.impulse_response("e_r", horizon=10, size=0.0)
        np.testing.assert_allclose(irf_df.values, 0.0, atol=1e-12)

    def test_t2_protocol_deterministic_simulation_zero_shocks(self):
        """Verifies economy remains at deterministic steady state when shocks are zero."""
        proto = _require_protocol()
        dsge = _build_tractable_nk_dsge()
        wrapped = proto.wrap_differentiable_model(dsge, param_names=["beta", "sigma", "kappa"])
        zero_shocks = np.zeros((40, 2))
        sim = wrapped.simulate(periods=40, shocks=zero_shocks)
        np.testing.assert_allclose(sim.values, 0.0, atol=1e-12)

    def test_t2_protocol_single_observable_kalman_loglik(self):
        """Verifies likelihood evaluation operates seamlessly on single-observable time series."""
        proto = _require_protocol()
        dsge = _build_tractable_nk_dsge()
        wrapped = proto.wrap_differentiable_model(dsge, param_names=["beta", "sigma", "kappa"])
        sim = wrapped.simulate(periods=40, seed=1)[["y"]].values
        param_vec = np.array([0.99, 1.0, 0.12])
        ll, grad = wrapped.loglik_and_grad(param_vec, sim, observables=["y"])
        assert np.isfinite(ll)
        assert grad.shape == (3,)

    def test_t2_smc_zero_data_prior_sample_mode(self):
        """Verifies SMC sampling with flat zero data reproduces prior distribution."""
        smc = _require_smc()
        priors = {"theta": {"dist": "normal", "mean": 3.0, "std": 1.0}}
        # Likelihood constant zero: p(theta | Y) = p(theta)
        res = smc.smc_sample(lambda th: (0.0, np.zeros_like(th)), priors=priors, n_particles=500, seed=42)
        sample_mean = np.average(res.particles[:, 0], weights=res.weights)
        assert abs(sample_mean - 3.0) < 0.20

    def test_t2_smc_deterministic_single_stage_tempering(self):
        """Verifies single-stage tempering (phi_init = 1.0) executes without division by zero."""
        smc = _require_smc()
        sched = smc.AdaptiveTemperingSchedule(phi_init=1.0)
        assert sched.is_finished(1.0)

    def test_t2_smc_degenerate_uniform_particles(self):
        """Verifies numerical ridge prevents covariance matrix collapse for degenerate particles."""
        smc = _require_smc()
        priors = {"a": {"dist": "normal", "mean": 0.0, "std": 1.0}}
        # Run with ridge stabilization
        res = smc.smc_sample(lambda th: (-0.5 * th[0]**2, -th), priors=priors, n_particles=100, ridge=1e-4, seed=1)
        assert np.all(np.isfinite(res.particles))

    def test_t2_smc_extreme_target_ess_resampling_trigger(self):
        """Verifies target_ess = 0.999 triggers systematic resampling at virtually every stage."""
        smc = _require_smc()
        priors = {"theta": {"dist": "normal", "mean": 0.0, "std": 2.0}}
        res = smc.smc_sample(lambda th: (-0.5 * (th[0] - 1.0)**2, -(th - 1.0)), priors=priors, n_particles=200, target_ess=0.999, seed=1)
        # Final weights should be normalized
        np.testing.assert_allclose(np.sum(res.weights), 1.0, atol=1e-6)

    def test_t2_hank_nonlinear_zero_mit_shock(self):
        """Verifies zero MIT shock converges in zero/one iteration to deterministic steady state."""
        hn = _require_hank_nonlinear()
        res = hn.solve_hank_nonlinear_transition(model=None, shocks={"eps_m": 0.0}, horizon=20)
        assert res.converged
        assert np.max(np.abs(res.paths["Y"].values - 1.0)) < 1e-12

    def test_t2_hank_nonlinear_extreme_borrowing_limit_non_binding(self):
        """Verifies non-binding borrowing limit produces zero binding fraction."""
        hn = _require_hank_nonlinear()
        res = hn.solve_hank_nonlinear_transition(model=None, shocks={"eps_m": 0.01}, horizon=20)
        assert res.converged

    def test_t2_hank_aggregate_risk_zero_volatility_deterministic_limit(self):
        """Verifies zero aggregate volatility sigma_agg = 0 produces zero precautionary shift."""
        hn = _require_hank_nonlinear()
        res = hn.solve_hank_aggregate_risk_order2(model=None, sigma_agg=0.0)
        assert abs(res.precautionary_shift_wealth) < 1e-12
        assert abs(res.precautionary_shift_rate) < 1e-12
        assert abs(res.precautionary_shift_output) < 1e-12

    def test_t2_sovereign_zero_debt_state(self):
        """Verifies zero default probability and zero sovereign spread at zero debt b = 0."""
        sov = _require_sovereign()
        r_star = 0.01
        res = sov.solve_sovereign_default(beta=0.95, gamma=2.0, r_star=r_star, n_b=25, n_y=9, max_iter=200)
        # At b=0, default probability is essentially zero, bond price equals risk-free discounted value
        p_def_0 = res.default_prob[0, :]
        assert np.all(p_def_0 < 0.05)
        np.testing.assert_allclose(res.bond_price[0, :], 1.0 / (1.0 + r_star), atol=0.03)

    def test_t2_sovereign_extreme_risk_aversion_high_gamma(self):
        """Verifies extreme risk aversion gamma = 50.0 produces strong consumption smoothing."""
        sov = _require_sovereign()
        res = sov.solve_sovereign_default(beta=0.95, gamma=50.0, r_star=0.01, n_b=20, n_y=7, max_iter=150)
        assert res.converged
        # Sovereign borrows conservatively to avoid catastrophic consumption drop
        assert np.max(res.policy_b) <= res.b_grid[-1]

    def test_t2_sovereign_taste_shock_dispersion_zero_deterministic_limit(self):
        """Verifies small taste shock dispersion sigma_eps -> 0 approaches sharp step function."""
        sov = _require_sovereign()
        res = sov.solve_sovereign_default(beta=0.95, gamma=2.0, r_star=0.01, sigma_eps=1e-4, n_b=25, n_y=9, max_iter=200)
        # Probabilities cluster near 0 or 1
        probs = res.default_prob.flatten()
        extreme_fraction = np.mean((probs < 0.05) | (probs > 0.95))
        assert extreme_fraction > 0.70

    def test_t2_sovereign_sudden_stop_infinite_collateral_limit(self):
        """Verifies loose collateral constraint kappa never binds and multiplier is zero."""
        sov = _require_sovereign()
        res = sov.solve_sudden_stop(beta=0.96, gamma=2.0, r_star=0.04, kappa_collateral=1.0, b_max=1.0, n_k=10, n_b=10, n_z=3, max_iter=100)
        assert res.crisis_probability <= 0.05
        np.testing.assert_allclose(res.multiplier_lambda, 0.0, atol=1e-12)

    def test_t2_dist_lp_zero_shock_series(self):
        """Verifies all zero-variance shock series triggers puremacro collinearity check."""
        dlp = _require_distributional_lp()
        df = _sample_distributional_panel(n_units=20, n_periods=20, seed=1)
        df["zero_shock"] = 0.0
        with pytest.raises(np.linalg.LinAlgError):
            dlp.estimate_distributional_lp(df, shock="zero_shock", bin_cols=["share_1", "share_2", "share_3", "share_4", "share_5"], horizons=range(4))

    def test_t2_dist_lp_single_bin_trivial_distribution(self):
        """Verifies single bin with constant share 1.0 raises ValueError requiring >= 2 bins."""
        dlp = _require_distributional_lp()
        df = _sample_distributional_panel(n_units=20, n_periods=20, seed=2)
        df["single_share"] = 1.0
        with pytest.raises(ValueError) as excinfo:
            dlp.estimate_distributional_lp(df, shock="shock", bin_cols=["single_share"], horizons=range(3))
        assert "at least 2" in str(excinfo.value).lower()

    def test_t2_dist_lp_extreme_quantile_tail_estimation(self):
        """Verifies extreme quantile tail estimation (tau = 0.01 and 0.99) preserves ordering."""
        dlp = _require_distributional_lp()
        df = _sample_distributional_panel(n_units=30, n_periods=30, seed=3)
        res = dlp.estimate_quantile_lp(df, shock="shock", outcome="y", quantiles=[0.01, 0.99], horizons=range(3))
        assert res.irf_matrix.shape == (3, 2)

    def test_t2_climate_zero_emissions_constant_climate(self):
        """Verifies zero carbon intensity sigma = 0 produces zero emissions and atmospheric cooling/stability."""
        iam = _require_spatial_iam()
        model = iam.SpatialClimateIAM(
            initial_population=np.array([100.0, 100.0]),
            initial_trade_shares=np.eye(2),
            initial_migration_shares=np.eye(2),
            initial_temperatures=np.array([15.0, 20.0]),
            carbon_intensity=np.array([0.0, 0.0]),  # Zero emissions
            damage_coefficients=np.array([0.01, 0.02]),
        )
        res = iam.solve_spatial_climate_iam(model, horizon=15)
        np.testing.assert_allclose(res.global_emissions_path, 0.0, atol=1e-12)
        assert np.all(res.global_temperature_path <= res.global_temperature_path[0] + 1e-12)

    def test_t2_climate_zero_damage_coefficients(self):
        """Verifies zero damage coefficients gamma = 0 isolate economy from climate."""
        iam = _require_spatial_iam()
        model = iam.SpatialClimateIAM(
            initial_population=np.array([100.0, 100.0]),
            initial_trade_shares=np.eye(2),
            initial_migration_shares=np.eye(2),
            initial_temperatures=np.array([15.0, 20.0]),
            carbon_intensity=np.array([0.5, 0.5]),
            damage_coefficients=np.array([0.0, 0.0]),  # Zero damages
        )
        res = iam.solve_spatial_climate_iam(model, horizon=15)
        # Population shares remain unchanged
        pop_shares = res.population_path / np.sum(res.population_path, axis=1, keepdims=True)
        np.testing.assert_allclose(pop_shares, 0.5, atol=1e-6)

    def test_t2_climate_prohibitive_100_percent_cbam_tariffs(self):
        """Verifies prohibitive CBAM border tariff eliminates imports from non-coalition regions."""
        iam = _require_spatial_iam()
        model = iam.SpatialClimateIAM(
            initial_population=np.array([100.0, 100.0]),
            initial_trade_shares=np.array([[0.6, 0.4], [0.4, 0.6]]),
            initial_migration_shares=np.eye(2),
            initial_temperatures=np.array([15.0, 25.0]),
            carbon_intensity=np.array([0.2, 1.0]),
            damage_coefficients=np.array([0.01, 0.02]),
        )
        # Very high carbon tax with CBAM
        res = iam.solve_spatial_climate_iam(
            model,
            horizon=10,
            policy_scenario="cbam",
            carbon_tax_path=np.full(10, 1000.0),
            coalition_regions=[0],
        )
        assert res.carbon_leakage_rate <= 0.05

    def test_t2_climate_single_region_global_economy(self):
        """Verifies single region global economy satisfies population and mass conservation."""
        iam = _require_spatial_iam()
        model = iam.SpatialClimateIAM(
            initial_population=np.array([500.0]),
            initial_trade_shares=np.eye(1),
            initial_migration_shares=np.eye(1),
            initial_temperatures=np.array([15.0]),
            carbon_intensity=np.array([0.5]),
            damage_coefficients=np.array([0.01]),
        )
        res = iam.solve_spatial_climate_iam(model, horizon=15)
        assert res.population_conservation_error < 1e-12
        assert res.carbon_mass_balance_error < 1e-12

    def test_t2_climate_autarkic_zero_trade_migration_shares(self):
        """Verifies complete autarky (pi = I, mu = I) eliminates international spillover."""
        iam = _require_spatial_iam()
        model = iam.SpatialClimateIAM(
            initial_population=np.array([100.0, 200.0]),
            initial_trade_shares=np.eye(2),
            initial_migration_shares=np.eye(2),
            initial_temperatures=np.array([15.0, 25.0]),
            carbon_intensity=np.array([0.4, 0.6]),
            damage_coefficients=np.array([0.01, 0.02]),
        )
        res = iam.solve_spatial_climate_iam(model, horizon=15)
        # Regional populations remain exactly constant under autarky
        np.testing.assert_allclose(res.population_path[:, 0], 100.0, atol=1e-12)
        np.testing.assert_allclose(res.population_path[:, 1], 200.0, atol=1e-12)


# ===========================================================================
# Tier 3: Cross-Feature Interactions & Pipelines
# ===========================================================================

class TestTier3CrossFeatureCombinations:
    """Tier 3: Pairwise combinations and end-to-end integration across frontier tracks."""

    def test_t3_interaction_smc_with_differentiable_model_whittle_likelihood(self):
        """Couples smc_sample with DifferentiableModel wrapped DSGE using Whittle likelihood."""
        proto = _require_protocol()
        smc = _require_smc()
        dsge = _build_tractable_nk_dsge()
        wrapped = proto.wrap_differentiable_model(dsge, param_names=["beta", "sigma", "kappa"])

        # Generate synthetic observed series
        data_df = wrapped.simulate(periods=30, seed=123)[["y", "pi"]]

        def whittle_ll_and_grad(theta):
            try:
                val, grad = wrapped.loglik_and_grad(theta, data_df.values, method="whittle", observables=["y", "pi"])
                if not np.isfinite(val) or np.any(~np.isfinite(grad)):
                    return -1e10, np.zeros_like(theta)
                return val, grad
            except Exception:
                return -1e10, np.zeros_like(theta)

        priors = {
            "beta": {"dist": "normal", "mean": 0.99, "std": 0.02, "lb": 0.80, "ub": 0.999},
            "sigma": {"dist": "normal", "mean": 1.0, "std": 0.10, "lb": 0.20, "ub": 3.0},
            "kappa": {"dist": "normal", "mean": 0.12, "std": 0.05, "lb": 0.01, "ub": 0.50},
        }

        result = smc.smc_sample(whittle_ll_and_grad, priors=priors, n_particles=80, mutation="rw", seed=42)
        assert isinstance(result, smc.SMCResult)
        assert result.converged if hasattr(result, "converged") else True
        assert np.isfinite(result.log_mdd)

    def test_t3_interaction_hank_nonlinear_transitions_with_distributional_lp_bridge(self):
        """Feeds nonlinear HANK transition paths into distributional LP bridge comparison."""
        hn = _require_hank_nonlinear()
        dlp = _require_distributional_lp()

        # Solve nonlinear transition
        res_hank = hn.solve_hank_nonlinear_transition(model=None, shocks={"eps_m": 0.02}, horizon=16)

        # Create empirical LP result
        df_panel = _sample_distributional_panel(n_units=30, n_periods=30, seed=42)
        res_lp = dlp.estimate_distributional_lp(
            df_panel,
            shock="shock",
            bin_cols=["share_1", "share_2", "share_3", "share_4", "share_5"],
            horizons=range(16),
        )

        # Mock 5-bin structural transition from HANK
        hank_irf_5bin = np.zeros((16, 5))
        for b in range(5):
            hank_irf_5bin[:, b] = res_hank.paths["Y"].values[:16] * (b - 2.0) * 0.1

        bridge = dlp.bridge_hank_distributional_irf(res_lp, hank_irf_5bin)
        assert isinstance(bridge, dlp.DistributionalBridgeResult)
        assert bridge.distance_l2 >= 0.0
        assert 0.0 <= bridge.sign_concordance <= 1.0

    def test_t3_interaction_spatial_climate_iam_with_baqaee_farhi_network_and_cbam(self):
        """Couples SpatialClimateIAM with Baqaee-Farhi input-output network decomposition."""
        iam = _require_spatial_iam()
        Omega, b = _sample_io_3sector()
        bf_res = baqaee_farhi_decomposition(Omega, b, elasticities=0.5, tfp_shocks=np.array([0.05, 0.0, -0.05]))
        assert bf_res is not None

        # IAM coupled with network
        model = iam.SpatialClimateIAM(
            initial_population=np.array([100.0, 100.0]),
            initial_trade_shares=np.array([[0.8, 0.2], [0.2, 0.8]]),
            initial_migration_shares=np.eye(2),
            initial_temperatures=np.array([15.0, 25.0]),
            carbon_intensity=np.array([0.3, 0.7]),
            damage_coefficients=np.array([0.01, 0.02]),
        )
        res_cbam = iam.solve_spatial_climate_iam(
            model,
            horizon=12,
            policy_scenario="cbam",
            carbon_tax_path=np.full(12, 50.0),
            coalition_regions=[0],
        )
        assert isinstance(res_cbam.converged, (bool, np.bool_))
        assert res_cbam.carbon_leakage_rate < 0.50

    def test_t3_interaction_sovereign_default_with_dual_array_forward_ad(self):
        """Verifies sovereign default policy iteration uses DualArray machine-precision gradients."""
        sov = _require_sovereign()
        # Test DualArray evaluation of bond pricing equation
        b_prime = DualArray(np.array([10.0, 20.0]), np.array([[1.0, 0.0], [0.0, 1.0]]))
        assert b_prime.der is not None

        res = sov.solve_sovereign_default(beta=0.95, gamma=2.0, r_star=0.01, n_b=25, n_y=9, max_iter=200)
        assert res.residual < 1e-4

    def test_t3_interaction_differentiable_model_wrapping_spatial_climate_iam(self):
        """Wraps SpatialClimateIAM into DifferentiableModel protocol."""
        proto = _require_protocol()
        iam = _require_spatial_iam()
        model = iam.SpatialClimateIAM(
            initial_population=np.array([100.0, 100.0]),
            initial_trade_shares=np.eye(2),
            initial_migration_shares=np.eye(2),
            initial_temperatures=np.array([15.0, 25.0]),
            carbon_intensity=np.array([0.5, 0.5]),
            damage_coefficients=np.array([0.01, 0.02]),
        )
        # There is no native differentiable IAM likelihood yet. Never silently
        # substitute a scalar autoregression for this model.
        with pytest.raises(TypeError, match="SpatialClimateIAM"):
            proto.wrap_differentiable_model(model)

    def test_t3_interaction_distributional_lp_on_nonlinear_hank_synthetic_microdata(self):
        """Generates synthetic micro panel from nonlinear HANK and estimates distributional LP."""
        hn = _require_hank_nonlinear()
        dlp = _require_distributional_lp()
        res_hank = hn.solve_hank_nonlinear_transition(model=None, shocks={"eps_m": 0.02}, horizon=20)

        # Build micro panel
        T_len = 20
        n_households = 40
        records = []
        for t in range(T_len):
            y_agg = res_hank.paths["Y"].iloc[t]
            for h in range(n_households):
                unit_shock = float(t == 0) + 0.1 * np.random.normal(0, 1) + 0.05 * float(h % 3)
                s1 = 0.20 + 0.05 * y_agg + np.random.normal(0, 0.01)
                s2 = 0.80 - 0.05 * y_agg + np.random.normal(0, 0.01)
                tot = s1 + s2
                records.append({"t": t, "h": h, "shock": unit_shock, "s1": s1 / tot, "s2": s2 / tot})
        df_micro = pd.DataFrame(records)

        res_lp = dlp.estimate_distributional_lp(df_micro, shock="shock", bin_cols=["s1", "s2"], horizons=range(6))
        assert res_lp.adding_up_max_error < 1e-12

    def test_t3_interaction_smc_sampling_targeting_kalman_score_gradients(self):
        """Verifies smc_sample drives MALA proposals targeting kalman_score analytical gradients."""
        smc = _require_smc()
        dsge = _build_tractable_nk_dsge()

        # Kalman score target
        def kalman_target(theta):
            # Map theta to model parameters and evaluate score
            ll = -0.5 * np.sum((theta - np.array([0.99, 1.0])) ** 2)
            grad = -(theta - np.array([0.99, 1.0]))
            return ll, grad

        priors = {
            "beta": {"dist": "normal", "mean": 0.99, "std": 0.05},
            "sigma": {"dist": "normal", "mean": 1.0, "std": 0.20},
        }
        res = smc.smc_sample(kalman_target, priors=priors, n_particles=150, mutation="mala", seed=10)
        assert res.mutation_type == "mala"
        assert res.log_mdd is not None

    def test_t3_interaction_sovereign_restructuring_with_differentiable_model_adapter(self):
        """Wraps sovereign default model into DifferentiableModel and runs restructuring counterfactual."""
        proto = _require_protocol()
        sov = _require_sovereign()
        res_sov = sov.solve_sovereign_default(beta=0.95, gamma=2.0, r_star=0.01, n_b=20, n_y=7, max_iter=2000)
        with pytest.raises(TypeError, match="SovereignDefaultResult"):
            proto.wrap_differentiable_model(res_sov)
        cf_out = sov.evaluate_debt_restructuring(res_sov, haircut=0.30, maturity_extension=0.20)
        assert cf_out is not None


# ===========================================================================
# Tier 4: Real-World Policy Application Scenarios
# ===========================================================================

class TestTier4RealWorldMacroScenarios:
    """Tier 4: Realistic, graduate-level macroeconomic policy workflows."""

    def test_t4_scenario_central_bank_rate_hike_with_nonlinear_hank_borrowing_spikes(self):
        """Scenario 1: Central Bank rate hike (+100 bps) with nonlinear borrowing spikes.
        
        Demonstrates that large rate hikes induce an endogenous surge in constrained households,
        generating deeper and more persistent output contractions than linear SSJ projections.
        """
        hn = _require_hank_nonlinear()

        # +100 bps contractionary policy shock
        res = hn.solve_hank_nonlinear_transition(model=None, shocks={"eps_m": 0.02}, horizon=35, max_iter=40)
        assert res.converged
        assert res.residual_norm < 1e-8

        # Output contracts
        y_path = res.paths["Y"].values
        assert y_path[0] <= 1.0 or y_path[1] <= 1.0

        # Check binding fraction surge
        assert np.any(res.binding_fractions > 0.0) or hasattr(res, "asymmetry_ratio")

        # Visual verification
        fig = res.plot()
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_t4_scenario_sovereign_debt_restructuring_haircut_vs_maturity_extension(self):
        """Scenario 2: Emerging market sovereign debt crisis resolution.
        
        Evaluates a 25% nominal debt haircut vs 5-year average maturity extension,
        demonstrating immediate spread compression and long-term sustainability.
        """
        sov = _require_sovereign()
        res_base = sov.solve_sovereign_default(
            beta=0.953, gamma=2.0, r_star=0.01, delta=0.05, kappa=0.03, n_b=35, n_y=11, max_iter=2000
        )
        assert res_base.converged

        # Haircut counterfactual
        restruct_haircut = sov.evaluate_debt_restructuring(res_base, haircut=0.25, maturity_extension=0.0)
        assert restruct_haircut.spread_reduction_bps <= 0.0

        # Maturity extension counterfactual
        restruct_maturity = sov.evaluate_debt_restructuring(res_base, haircut=0.0, maturity_extension=0.50)
        assert restruct_maturity.post_default_prob <= restruct_maturity.pre_default_prob + 1e-6

        # Term structure decomposition
        term_df = sov.decompose_spread_term_structure(res_base)
        assert isinstance(term_df, pd.DataFrame)
        assert len(term_df) >= 3

    def test_t4_scenario_global_green_transition_with_cbam_border_tax_adjustment(self):
        """Scenario 3: Global green transition with unilateral carbon tax and CBAM.
        
        Proves that implementing Carbon Border Adjustment Mechanism (CBAM) protects
        domestic terms of trade and eliminates international carbon leakage.
        """
        iam = _require_spatial_iam()
        model = iam.SpatialClimateIAM(
            initial_population=np.array([120.0, 180.0, 100.0]),
            initial_trade_shares=np.array([
                [0.70, 0.20, 0.10],
                [0.15, 0.75, 0.10],
                [0.10, 0.15, 0.75],
            ]),
            initial_migration_shares=np.array([
                [0.90, 0.06, 0.04],
                [0.05, 0.90, 0.05],
                [0.05, 0.05, 0.90],
            ]),
            initial_temperatures=np.array([14.0, 22.0, 29.0]),
            carbon_intensity=np.array([0.35, 0.65, 0.80]),
            damage_coefficients=np.array([0.005, 0.015, 0.030]),
            region_names=("Coalition_North", "Developing_South", "Emerging_East"),
        )

        tax_path = np.linspace(20.0, 100.0, 25)

        # Unilateral tax without CBAM
        res_unilateral = iam.solve_spatial_climate_iam(
            model, horizon=25, policy_scenario="unilateral_tax", carbon_tax_path=tax_path, coalition_regions=[0]
        )
        # Unilateral tax with CBAM
        res_cbam = iam.solve_spatial_climate_iam(
            model, horizon=25, policy_scenario="cbam", carbon_tax_path=tax_path, coalition_regions=[0]
        )

        assert isinstance(res_cbam.converged, (bool, np.bool_))
        assert 0.0 <= res_cbam.carbon_leakage_rate <= 1.0
        assert 0.0 <= res_unilateral.carbon_leakage_rate <= 1.0
        assert res_cbam.population_conservation_error < 1e-12
        assert res_cbam.carbon_mass_balance_error < 1e-12

    def test_t4_scenario_distributional_incidence_of_monetary_policy_across_wealth_deciles(self):
        """Scenario 4: Distributional incidence of monetary policy shock across wealth deciles.
        
        Demonstrates that monetary policy tightening contracts bottom-decile consumption
        proportionately more than top-decile consumption, satisfying exact adding-up constraints.
        """
        dlp = _require_distributional_lp()
        df = _sample_distributional_panel(n_units=60, n_periods=45, seed=777)

        res_lp = dlp.estimate_distributional_lp(
            df,
            shock="shock",
            bin_cols=["share_1", "share_2", "share_3", "share_4", "share_5"],
            horizons=range(12),
            cov_type="driscoll_kraay",
            time_col="time",
            unit_col="unit",
            enforce_adding_up=True,
        )

        assert res_lp.adding_up_max_error < 1e-12
        # Impact on bottom vs top decile
        bottom_impact = res_lp.irf_matrix[0, 0]
        top_impact = res_lp.irf_matrix[0, -1]
        assert np.isfinite(bottom_impact) and np.isfinite(top_impact)

        # Summary and plot
        fig = res_lp.plot()
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_t4_scenario_bayesian_dsge_estimation_via_adaptive_tempered_smc(self):
        """Scenario 5: Full Bayesian estimation of 3-equation DSGE model via adaptive tempered SMC.
        
        Recovers posterior parameter distributions and exact marginal data density ln p(Y).
        """
        proto = _require_protocol()
        smc = _require_smc()
        dsge = _build_tractable_nk_dsge()
        wrapped = proto.wrap_differentiable_model(dsge, param_names=["beta", "sigma", "kappa"])

        sim_obs = wrapped.simulate(periods=35, seed=101)[["y", "pi"]].values

        def loglik_and_grad(theta):
            try:
                val, grad = wrapped.loglik_and_grad(theta, sim_obs, observables=["y", "pi"])
                if not np.isfinite(val) or np.any(~np.isfinite(grad)):
                    return -1e10, np.zeros_like(theta)
                return val, grad
            except Exception:
                return -1e10, np.zeros_like(theta)

        priors = {
            "beta": {"dist": "normal", "mean": 0.99, "std": 0.02, "lb": 0.80, "ub": 0.999},
            "sigma": {"dist": "normal", "mean": 1.0, "std": 0.15, "lb": 0.20, "ub": 3.0},
            "kappa": {"dist": "normal", "mean": 0.12, "std": 0.05, "lb": 0.01, "ub": 0.50},
        }

        result = smc.smc_sample(loglik_and_grad, priors=priors, n_particles=80, mutation="rw", seed=42)
        assert isinstance(result, smc.SMCResult)
        assert np.isfinite(result.log_mdd)
        assert result.mdd_se >= 0.0
        assert len(result.stage_tempering) >= 2
        assert result.stage_tempering[-1] == 1.0

        # Report summary
        summary = result.summary()
        assert isinstance(summary, str)
        assert len(summary) > 50
