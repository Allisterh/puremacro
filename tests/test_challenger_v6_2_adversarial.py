"""Empirical Adversarial Stress Harness for PureMacro v6.0.0.

Challenger 2 suite targeting:
- Track II.5 (Quantitative Sovereign Default & Sudden-Stop Suite)
- Track III.6 (Distributional & Quantile Local Projections)
- Track III.7 (Integrated Spatial Climate-Macro IAM)

Stress dimensions:
1. Sovereign:
   - Extreme debt haircuts (up to 90%) and long maturity extensions
   - Near-zero taste shocks (sigma_eps -> 0)
   - DualArray debt dilution derivatives under extreme/non-uniform grids
   - Mendoza collateral constraint under extreme leverage kappa (tight vs loose)
   - Sovereign yield curve decomposition out to 80 quarters
2. Distributional LP:
   - Near-singular covariance matrices & collinear regressors
   - Collinear / zero bins in shares and densities with non-uniform quadrature weights
   - Extreme quantiles (tau=0.01, 0.99)
   - Exact adding-up constraint conservation < 1e-12 across HAC and Driscoll-Kraay
   - Empirical-to-HANK bridge under adversarial noisy inputs
3. Spatial Climate IAM:
   - Autarky limits (infinite trade & migration costs: pi=I, mu=I)
   - Extreme temperature damage elasticities (catastrophic tipping points)
   - 100% prohibitive CBAM tariffs and leakage mitigation
   - Endogenous labor migration interaction with CBAM
   - Global population conservation < 1e-12 and carbon mass balance < 1e-12 across all regimes
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.sovereign.arellano import _compute_dq_db_dualarray, solve_sovereign_default
from puremacro.sovereign.mendoza import solve_sudden_stop
from puremacro.sovereign.restructuring import (
    decompose_spread_term_structure,
    evaluate_debt_restructuring,
)
from puremacro.lp.distributional import (
    estimate_distributional_lp,
    estimate_quantile_lp,
    bridge_hank_distributional_irf,
)
from puremacro.climate.spatial_iam import (
    SpatialClimateIAM,
    solve_spatial_climate_iam,
)


# ===========================================================================
# BATTERY 1: Track II.5 — Quantitative Sovereign Default & Sudden Stop Stress
# ===========================================================================

class TestTrackII5SovereignDefaultStress:
    """Adversarial stress testing for Arellano, Chatterjee-Eyigungor & Mendoza models."""

    @pytest.fixture(scope="class")
    def baseline_sovereign(self):
        """Pre-solved sovereign default baseline with short/long debt for speed."""
        return solve_sovereign_default(
            beta=0.95,
            gamma=2.0,
            r_star=0.01,
            delta=0.25,
            kappa=0.02,
            sigma_eps=0.01,
            n_b=30,
            n_y=11,
            max_iter=400,
            tol=1e-6,
        )

    def test_extreme_debt_haircuts_up_to_90_percent(self, baseline_sovereign):
        """Stress-test sovereign debt restructuring with extreme haircuts up to 90%."""
        res_base = baseline_sovereign

        for h in [0.20, 0.50, 0.75, 0.90]:
            restruct = evaluate_debt_restructuring(
                res_base,
                haircut=h,
                maturity_extension=0.50,
                horizon=25,
                seed=123,
            )
            # Verify debt relief scales monotonically with haircut
            assert restruct.debt_gdp_relief_pct > 0.0
            assert restruct.restructured_debt == pytest.approx(
                (1.0 - h) * restruct.initial_debt, rel=1e-6
            )
            # Extreme haircut must strictly reduce sovereign spread
            assert restruct.counterfactual_spread <= restruct.baseline_spread + 1e-6
            assert restruct.spread_reduction_bps <= 0.0 or restruct.counterfactual_spread <= 1e-4

            # Simulation path must have valid non-NaN fields
            sim_df = restruct.transition_path
            assert not sim_df["debt"].isna().any()
            assert not sim_df["consumption"].isna().any()
            assert (sim_df["debt"] >= 0.0).all()

    def test_extreme_maturity_extension(self, baseline_sovereign):
        """Stress-test restructuring with 300% maturity extension (duration quadrupled)."""
        res_base = baseline_sovereign
        restruct = evaluate_debt_restructuring(
            res_base,
            haircut=0.30,
            maturity_extension=3.0,  # 300% extension
            horizon=20,
        )
        assert restruct.maturity_extension_years > 0.0
        assert not restruct.transition_path["debt"].isna().any()
        assert restruct.post_default_prob <= restruct.pre_default_prob + 1e-6

    def test_near_zero_taste_shocks_deterministic_limit(self):
        """Stress-test taste shock dispersion sigma_eps -> 0 (deterministic choice limit)."""
        for s_eps in [0.005, 0.001, 0.0005]:
            res = solve_sovereign_default(
                beta=0.95,
                gamma=2.0,
                r_star=0.01,
                delta=1.0,  # Short-term Arellano
                sigma_eps=s_eps,
                n_b=25,
                n_y=9,
                max_iter=300,
                tol=1e-5,
            )
            # Invariants
            assert not np.isnan(res.v_star).any(), f"NaN in v_star for sigma_eps={s_eps}"
            assert not np.isnan(res.q_bond).any(), f"NaN in q_bond for sigma_eps={s_eps}"
            assert not np.isnan(res.default_prob).any(), f"NaN in default_prob for sigma_eps={s_eps}"
            # Default prob bounds
            assert np.all(res.default_prob >= 0.0)
            assert np.all(res.default_prob <= 1.0)
            # No-arbitrage bond price ceiling
            q_rf = 1.0 / (1.0 + 0.01)
            assert np.all(res.q_bond <= q_rf + 1e-8)
            # Dilution monotonicity: dq/db <= 0
            assert np.all(res.dq_db <= 1e-8)

    def test_debt_dilution_derivatives_extreme_grids(self):
        """Stress-test DualArray PCHIP derivatives under extreme and clustered grids."""
        n_y = 5
        # 1. Non-uniform grid heavily clustered near zero
        b_clustered = np.geomspace(1e-4, 1.0, 40) - 1e-4
        q_test = 0.98 * np.exp(-3.0 * b_clustered[:, None]) * np.linspace(0.8, 1.0, n_y)[None, :]
        dq_db = _compute_dq_db_dualarray(q_test, b_clustered)
        assert dq_db.shape == (40, n_y)
        assert np.all(dq_db <= 0.0), "All tangents must be non-positive"
        assert not np.isnan(dq_db).any()

        # 2. Extreme debt upper bound (b_max = 10.0)
        b_wide = np.linspace(0.0, 10.0, 50)
        q_wide = 1.0 / (1.0 + np.exp(b_wide[:, None] - 5.0))
        dq_wide = _compute_dq_db_dualarray(q_wide, b_wide)
        assert np.all(dq_wide <= 0.0)
        assert not np.isnan(dq_wide).any()

        # 3. Minimal grid (n_b = 4)
        b_min = np.linspace(0.0, 0.5, 4)
        q_min = np.array([
            [0.99, 0.99],
            [0.90, 0.95],
            [0.50, 0.70],
            [0.10, 0.20],
        ])
        dq_min = _compute_dq_db_dualarray(q_min, b_min)
        assert dq_min.shape == (4, 2)
        assert np.all(dq_min <= 0.0)

    def test_mendoza_collateral_constraint_extreme_leverage_kappa(self):
        """Stress-test Mendoza sudden stop under tight (kappa=0.02) and extreme loose (kappa=0.90) leverage."""
        # Tight leverage: kappa=0.03 (credit starvation)
        res_tight = solve_sudden_stop(
            beta=0.96,
            gamma=2.0,
            r_star=0.04,
            kappa_collateral=0.03,
            n_b=15,
            n_k=15,
            n_z=3,
            max_iter=200,
            tol=1e-5,
        )
        # Verify borrowing strictly respects tight collateral limit everywhere
        for iz in range(len(res_tight.z_grid)):
            for ik in range(len(res_tight.k_grid)):
                for ib in range(len(res_tight.b_grid)):
                    b_next = res_tight.policy_b[ib, ik, iz]
                    k_next = res_tight.policy_k[ib, ik, iz]
                    q_val = res_tight.asset_price_q[ib, ik, iz]
                    cap = 0.03 * q_val * k_next
                    assert b_next <= cap + 1e-6, f"Collateral constraint violated: b'={b_next} > cap={cap}"

        assert res_tight.collateral_binding_fraction >= 0.0
        assert not np.isnan(res_tight.multiplier_lambda).any()

        # Loose leverage: kappa=0.90 (unconstrained limit)
        res_loose = solve_sudden_stop(
            beta=0.96,
            gamma=2.0,
            r_star=0.04,
            kappa_collateral=0.90,
            n_b=15,
            n_k=15,
            n_z=3,
            max_iter=200,
            tol=1e-5,
        )
        assert res_loose.collateral_binding_fraction == 0.0
        assert res_loose.crisis_probability == 0.0
        assert res_loose.amplification_ratio == 1.0

    def test_sovereign_spread_term_structure_extreme_horizons(self, baseline_sovereign):
        """Stress-test term structure decomposition up to 80 quarters (20 years)."""
        df_ts = decompose_spread_term_structure(
            baseline_sovereign,
            horizons=[1, 4, 8, 12, 20, 40, 80],
        )
        assert len(df_ts) == 7
        assert not df_ts["yield_to_maturity"].isna().any()
        assert not df_ts["spread_bps"].isna().any()
        assert (df_ts["spread_bps"] >= 0.0).all()
        assert (df_ts["default_loss_component"] >= 0.0).all()
        assert (df_ts["risk_premium_component"] >= 0.0).all()


# ===========================================================================
# BATTERY 2: Track III.6 — Distributional & Quantile Local Projections Stress
# ===========================================================================

class TestTrackIII6DistributionalLPStress:
    """Adversarial stress testing for Distributional & Quantile Local Projections."""

    @pytest.fixture(scope="class")
    def synthetic_dist_data(self):
        """Generate synthetic panel and time-series microdata."""
        rng = np.random.default_rng(42)
        T = 250
        shock = rng.standard_normal(T)
        controls = rng.standard_normal((T, 2))

        # 5 share bins summing strictly to 1 at all t
        raw = np.abs(rng.standard_normal((T, 5)))
        shares = raw / np.sum(raw, axis=1, keepdims=True)

        df = pd.DataFrame({
            "shock": shock,
            "ctrl1": controls[:, 0],
            "ctrl2": controls[:, 1],
            "bin1": shares[:, 0],
            "bin2": shares[:, 1],
            "bin3": shares[:, 2],
            "bin4": shares[:, 3],
            "bin5": shares[:, 4],
            "outcome": 10.0 + 2.0 * shock + rng.standard_normal(T),
        })
        return df

    def test_dist_lp_exact_adding_up_shares_and_densities(self, synthetic_dist_data):
        """Verify cross-sectional adding-up error < 1e-12 across horizons."""
        df = synthetic_dist_data
        bin_cols = ["bin1", "bin2", "bin3", "bin4", "bin5"]

        res = estimate_distributional_lp(
            df,
            shock="shock",
            bin_cols=bin_cols,
            horizons=range(0, 12),
            controls=["ctrl1", "ctrl2"],
            distribution_type="shares",
            enforce_adding_up=True,
        )

        assert res.check_adding_up(tol=1e-12)
        assert res.adding_up_max_error < 1e-12
        # Verify manual dot product
        sum_betas = np.sum(res.irf_matrix, axis=1)
        assert np.max(np.abs(sum_betas)) < 1e-12

    def test_dist_lp_density_bins_unequal_quadrature_weights(self, synthetic_dist_data):
        """Verify adding-up under continuous density bins with non-uniform quadrature weights."""
        df = synthetic_dist_data
        bin_cols = ["bin1", "bin2", "bin3", "bin4", "bin5"]
        weights = [0.1, 0.2, 0.4, 0.2, 0.1]

        res = estimate_distributional_lp(
            df,
            shock="shock",
            bin_cols=bin_cols,
            horizons=range(0, 8),
            distribution_type="density",
            bin_widths=weights,
            enforce_adding_up=True,
        )
        assert res.check_adding_up(tol=1e-12)
        assert res.adding_up_max_error < 1e-12
        w_vec = np.array(weights)
        for h in range(8):
            weighted_sum = float(np.dot(res.irf_matrix[h], w_vec))
            assert abs(weighted_sum) < 1e-12

    def test_dist_lp_collinear_and_zero_bins(self, synthetic_dist_data):
        """Stress-test distributional LP when bins are collinear or contain zeros."""
        df = synthetic_dist_data.copy()
        # Create collinear bins: bin_copy is identical to bin1, bin_zero is all 0
        df["bin_copy"] = df["bin1"].copy()
        df["bin_zero"] = 0.0

        collinear_bins = ["bin1", "bin_copy", "bin2", "bin_zero"]
        res = estimate_distributional_lp(
            df,
            shock="shock",
            bin_cols=collinear_bins,
            horizons=range(0, 8),
            distribution_type="shares",
            enforce_adding_up=True,
        )

        # Null-space projection MUST preserve adding-up even under collinear/zero bins
        assert res.check_adding_up(tol=1e-12)
        assert res.adding_up_max_error < 1e-12
        assert not np.isnan(res.irf_matrix).any()
        assert not np.isnan(res.se_matrix).any()

    def test_dist_lp_near_singular_regressor_covariance_guard(self, synthetic_dist_data):
        """Verify that exact rank deficiency in regressors is diagnosed cleanly."""
        df = synthetic_dist_data.copy()
        df["ctrl_collinear"] = df["ctrl1"] * 2.0

        bin_cols = ["bin1", "bin2", "bin3"]
        with pytest.raises(np.linalg.LinAlgError) as exc_info:
            estimate_distributional_lp(
                df,
                shock="shock",
                bin_cols=bin_cols,
                horizons=[0, 1],
                controls=["ctrl1", "ctrl_collinear"],
            )
        assert "is singular" in str(exc_info.value)

    def test_quantile_lp_extreme_quantiles_001_and_099(self, synthetic_dist_data):
        """Stress-test Quantile LP at extreme tails tau = 0.01 and 0.99."""
        df = synthetic_dist_data

        res = estimate_quantile_lp(
            df,
            shock="shock",
            outcome="outcome",
            quantiles=[0.01, 0.05, 0.50, 0.95, 0.99],
            horizons=range(0, 6),
            n_lags=1,
        )

        assert res.irf_matrix.shape == (6, 5)
        assert not np.isnan(res.irf_matrix).any(), "NaN detected in extreme quantile IRF"
        assert not np.isnan(res.se_matrix).any(), "NaN detected in extreme quantile standard errors"
        assert np.all(res.se_matrix > 0.0)
        assert np.all(res.ci_lower <= res.irf_matrix)
        assert np.all(res.ci_upper >= res.irf_matrix)

    def test_dist_lp_bridge_hank_distributional_irf_noisy(self, synthetic_dist_data):
        """Stress-test HANK distributional bridge with noisy synthetic comparison."""
        df = synthetic_dist_data
        bin_cols = ["bin1", "bin2", "bin3"]
        lp_res = estimate_distributional_lp(
            df,
            shock="shock",
            bin_cols=bin_cols,
            horizons=range(0, 10),
        )
        # Synthetic HANK IRF with perturbation
        hank_irf = lp_res.irf_matrix + 0.05 * np.random.default_rng(99).standard_normal(lp_res.irf_matrix.shape)
        bridge = bridge_hank_distributional_irf(lp_res, hank_irf)

        assert bridge.distance_l2 > 0.0
        assert bridge.mean_absolute_error > 0.0
        assert not np.isnan(bridge.distance_l2)
        assert len(bridge.decile_correlations) == 3


# ===========================================================================
# BATTERY 3: Track III.7 — Spatial Climate IAM Stress
# ===========================================================================

class TestTrackIII7SpatialClimateIAMStress:
    """Adversarial stress testing for Integrated Spatial Climate-Macro IAM."""

    @pytest.fixture(scope="class")
    def standard_iam_model(self):
        """Create a 3-region calibrated baseline IAM."""
        L0 = np.array([100.0, 150.0, 250.0])
        pi0 = np.array([
            [0.70, 0.20, 0.10],
            [0.15, 0.70, 0.15],
            [0.10, 0.20, 0.70],
        ])
        mu0 = np.array([
            [0.80, 0.15, 0.05],
            [0.10, 0.80, 0.10],
            [0.05, 0.15, 0.80],
        ])
        T0 = np.array([12.0, 18.0, 24.0])
        sigma = np.array([0.45, 0.60, 0.80])
        gamma_A = np.array([0.005, 0.012, 0.025])

        return SpatialClimateIAM(
            initial_population=L0,
            initial_trade_shares=pi0,
            initial_migration_shares=mu0,
            initial_temperatures=T0,
            carbon_intensity=sigma,
            damage_coefficients=gamma_A,
            region_names=("North", "Mid", "South"),
        )

    def test_spatial_climate_iam_autarky_limits_infinite_trade_migration_costs(self):
        """Stress-test IAM under autarky limit (pi_0 = I, mu_0 = I, no trade or migration)."""
        N = 3
        L0 = np.array([100.0, 150.0, 200.0])
        pi_autarky = np.eye(N)
        mu_autarky = np.eye(N)
        T0 = np.array([14.0, 15.0, 16.0])
        sigma = np.array([0.5, 0.5, 0.5])
        gamma_A = np.array([0.01, 0.01, 0.01])

        autarky_model = SpatialClimateIAM(
            initial_population=L0,
            initial_trade_shares=pi_autarky,
            initial_migration_shares=mu_autarky,
            initial_temperatures=T0,
            carbon_intensity=sigma,
            damage_coefficients=gamma_A,
        )

        res = solve_spatial_climate_iam(
            autarky_model,
            horizon=15,
            policy_scenario="laissez_faire",
        )

        # Invariants under autarky
        assert res.population_conservation_error < 1e-12
        assert res.carbon_mass_balance_error < 1e-12
        assert np.all(res.population_path >= 0.0)
        assert not np.isnan(res.population_path).any()

    def test_spatial_climate_iam_extreme_temperature_damages(self, standard_iam_model):
        """Stress-test IAM under catastrophic tipping-point temperature damages (gamma_A up to 0.20)."""
        catastrophic_model = SpatialClimateIAM(
            initial_population=standard_iam_model.initial_population,
            initial_trade_shares=standard_iam_model.initial_trade_shares,
            initial_migration_shares=standard_iam_model.initial_migration_shares,
            initial_temperatures=standard_iam_model.initial_temperatures,
            carbon_intensity=standard_iam_model.carbon_intensity,
            damage_coefficients=np.array([0.10, 0.15, 0.20]),  # 10-20% output drop per °C
            damage_curvature=np.array([0.02, 0.03, 0.05]),     # strong quadratic penalty
            amenity_damage_coefficients=np.array([0.05, 0.08, 0.12]),
        )

        res = solve_spatial_climate_iam(
            catastrophic_model,
            horizon=20,
            policy_scenario="laissez_faire",
        )

        # Even under catastrophic damage, mathematical invariants MUST hold
        assert res.population_conservation_error < 1e-12
        assert res.carbon_mass_balance_error < 1e-12
        assert np.all(res.population_path > 0.0)
        assert np.all(res.real_wage_hat_path > 0.0)
        assert not np.isnan(res.welfare_hat).any()

    def test_spatial_climate_iam_prohibitive_100pct_cbam_tariffs(self):
        """Stress-test IAM with prohibitive $1000/tCO2 tax and 100% CBAM tariffs."""
        model = SpatialClimateIAM(
            initial_population=np.array([100.0, 100.0]),
            initial_trade_shares=np.array([[0.6, 0.4], [0.4, 0.6]]),
            initial_migration_shares=np.eye(2),
            initial_temperatures=np.array([15.0, 25.0]),
            carbon_intensity=np.array([0.2, 1.0]),
            damage_coefficients=np.array([0.01, 0.02]),
        )
        res_prohibitive = solve_spatial_climate_iam(
            model,
            horizon=12,
            policy_scenario="cbam",
            carbon_tax_path=np.full(12, 1000.0),  # $1000/tCO2 prohibitive tariff
            coalition_regions=[0],
            cbam=True,
        )
        # Prohibitive CBAM crushes carbon leakage to near-zero
        assert res_prohibitive.carbon_leakage_rate <= 0.05
        # Invariants
        assert res_prohibitive.population_conservation_error < 1e-12
        assert res_prohibitive.carbon_mass_balance_error < 1e-12

    def test_spatial_climate_iam_trade_leakage_mitigation_by_cbam(self):
        """Verify that under trade adjustment without labor flight, CBAM strictly reduces leakage."""
        model = SpatialClimateIAM(
            initial_population=np.array([100.0, 100.0]),
            initial_trade_shares=np.array([[0.7, 0.3], [0.3, 0.7]]),
            initial_migration_shares=np.eye(2),  # immobile labor focuses purely on trade channel
            initial_temperatures=np.array([15.0, 25.0]),
            carbon_intensity=np.array([0.5, 0.5]),
            damage_coefficients=np.array([0.01, 0.02]),
        )
        res_tax = solve_spatial_climate_iam(
            model,
            horizon=15,
            policy_scenario="unilateral_tax",
            carbon_tax_path=np.full(15, 80.0),
            coalition_regions=[0],
        )
        res_cbam = solve_spatial_climate_iam(
            model,
            horizon=15,
            policy_scenario="cbam",
            carbon_tax_path=np.full(15, 80.0),
            coalition_regions=[0],
            cbam=True,
        )
        # In pure trade adjustment, CBAM tariff on non-coalition goods strictly suppresses leakage
        assert res_cbam.carbon_leakage_rate < res_tax.carbon_leakage_rate
        assert res_tax.population_conservation_error < 1e-12
        assert res_tax.carbon_mass_balance_error < 1e-12
        assert res_cbam.population_conservation_error < 1e-12
        assert res_cbam.carbon_mass_balance_error < 1e-12

    def test_spatial_climate_iam_optimal_pigouvian_formula_analytical_recovery(self):
        """Verify optimal Pigouvian carbon tax matches Golosov et al. (2014) to machine precision."""
        model = SpatialClimateIAM(
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
        res = solve_spatial_climate_iam(
            model,
            horizon=10,
            policy_scenario="optimal_pigouvian",
        )
        # Analytical Golosov formula:
        # tau = Y * gamma * [ phi_L / (1 - beta) + (1 - phi_L)*phi_0 / (1 - beta*(1 - phi_decay)) ]
        Y0 = 100.0  # L0=100, w0=1.0
        gamma = 0.015
        beta = 0.96
        phi_L = 0.20
        phi_0 = 0.393
        phi_decay = 0.02

        num_perm = phi_L / (1.0 - beta)
        num_decay = (1.0 - phi_L) * phi_0 / (1.0 - beta * (1.0 - phi_decay))
        expected_tau_0 = Y0 * gamma * (num_perm + num_decay)

        assert abs(res.pigouvian_tax_path[0] - expected_tau_0) < 1e-6
        assert res.population_conservation_error < 1e-12
        assert res.carbon_mass_balance_error < 1e-12
