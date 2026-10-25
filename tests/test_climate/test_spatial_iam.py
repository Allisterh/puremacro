"""Unit tests for puremacro.climate.spatial_iam (Track III.7)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.climate.spatial_iam import (
    SpatialClimateIAM,
    SpatialClimateIAMResult,
    solve_spatial_climate_iam,
)


def _build_3region_iam_model() -> SpatialClimateIAM:
    """Construct a calibrated 3-region spatial climate-macro economy:

    Region 1: Temperate / North (cool initial temperature 10°C, low damage)
    Region 2: Tropical / South (hot initial temperature 28°C, high damage, carbon-intensive)
    Region 3: Emerging / East (moderate temperature 18°C, intermediate damage)
    """
    initial_population = np.array([100.0, 150.0, 120.0])
    # Symmetric baseline gravity trade shares (row-stochastic)
    initial_trade_shares = np.array([
        [0.70, 0.15, 0.15],
        [0.10, 0.75, 0.15],
        [0.12, 0.18, 0.70],
    ])
    # Baseline migration shares (row-stochastic)
    initial_migration_shares = np.array([
        [0.92, 0.04, 0.04],
        [0.05, 0.90, 0.05],
        [0.04, 0.04, 0.92],
    ])
    initial_temperatures = np.array([10.0, 28.0, 18.0])
    carbon_intensity = np.array([0.25, 0.65, 0.45])  # GtCO2 per unit of output
    damage_coefficients = np.array([0.002, 0.025, 0.012])  # gamma_A
    amenity_damage = np.array([0.001, 0.015, 0.006])  # gamma_B
    warming_pattern_scale = np.array([1.25, 0.95, 1.10])  # polar amplification in Region 1

    return SpatialClimateIAM(
        initial_population=initial_population,
        initial_trade_shares=initial_trade_shares,
        initial_migration_shares=initial_migration_shares,
        initial_temperatures=initial_temperatures,
        carbon_intensity=carbon_intensity,
        damage_coefficients=damage_coefficients,
        amenity_damage_coefficients=amenity_damage,
        warming_pattern_scale=warming_pattern_scale,
        trade_elasticity=4.0,
        migration_dispersion=0.5,
        beta=0.96,
        congestion_alpha=0.10,
        climate_sensitivity=3.0,
        preindustrial_carbon=588.0,
        initial_carbon_stock=850.0,
        phi_L=0.20,
        phi_0=0.393,
        phi_decay=0.02,
        abatement_theta1=550.0,
        abatement_theta2=2.6,
        region_names=("North", "South", "East"),
    )


def test_spatial_climate_population_conservation():
    """Verify global population conservation sum_r L_{r,t} = L_bar to < 10^-12 at all t."""
    model = _build_3region_iam_model()
    res = solve_spatial_climate_iam(
        model,
        horizon=25,
        policy_scenario="laissez_faire",
    )

    assert isinstance(res, SpatialClimateIAMResult)
    L_bar = float(np.sum(model.initial_population))
    pop_sums = np.sum(res.population_path, axis=1)

    max_err = float(np.max(np.abs(pop_sums - L_bar)))
    assert max_err < 1e-12, f"Population conservation error exceeded 1e-12: {max_err:.2e}"
    assert res.population_conservation_error < 1e-12


def test_spatial_climate_carbon_mass_balance():
    """Verify carbon cycle mass balance: cumulative injected = stored in reservoirs + sink to < 10^-12."""
    model = _build_3region_iam_model()
    res = solve_spatial_climate_iam(
        model,
        horizon=20,
        policy_scenario="laissez_faire",
    )

    assert res.carbon_mass_balance_error < 1e-12, (
        f"Carbon mass balance error exceeded 1e-12: {res.carbon_mass_balance_error:.2e}"
    )


def test_spatial_climate_cbam_leakage_reduction():
    """Verify that CBAM border tariffs strictly reduce carbon leakage relative to unilateral pricing."""
    model = _build_3region_iam_model()
    coalition = [0]  # Region North unilaterally abates

    tax_trajectory = 80.0 * (1.02 ** np.arange(20))

    # 1. Unilateral carbon tax without CBAM
    res_unilateral = solve_spatial_climate_iam(
        model,
        horizon=20,
        policy_scenario="unilateral_tax",
        carbon_tax_path=tax_trajectory,
        coalition_regions=coalition,
        cbam=False,
    )

    # 2. Unilateral carbon tax WITH CBAM
    res_cbam = solve_spatial_climate_iam(
        model,
        horizon=20,
        policy_scenario="cbam",
        carbon_tax_path=tax_trajectory,
        coalition_regions=coalition,
        cbam=True,
    )

    leakage_unilateral = res_unilateral.carbon_leakage_rate
    leakage_cbam = res_cbam.carbon_leakage_rate

    # Mandatory guarantee: CBAM strictly reduces carbon leakage
    assert leakage_cbam < leakage_unilateral, (
        f"CBAM leakage ({leakage_cbam:.4f}) must be strictly less than unilateral leakage ({leakage_unilateral:.4f})"
    )


def test_spatial_climate_golosov_pigouvian_limit():
    """Verify that Pigouvian optimal tax matches the analytical Golosov et al. (2014) formula to < 10^-6."""
    model = _build_3region_iam_model()
    res = solve_spatial_climate_iam(
        model,
        horizon=15,
        policy_scenario="optimal_pigouvian",
    )

    assert res.policy_scenario == "optimal_pigouvian"
    assert res.pigouvian_tax_path.shape == (15,)
    assert np.all(res.pigouvian_tax_path > 0.0)

    # Compute analytical Golosov formula in log-utility limit
    # tau_t = Y_t * gamma * [phi_L / (1 - beta) + (1 - phi_L)*phi_0 / (1 - beta*(1 - phi_decay))]
    num_perm = model.phi_L / (1.0 - model.beta)
    num_decay = (1.0 - model.phi_L) * model.phi_0 / (1.0 - model.beta * (1.0 - model.phi_decay))
    brackets = num_perm + num_decay

    Y0 = np.sum(model.initial_population)  # with initial wages = 1
    mean_gamma = res.metadata["mean_gamma"]

    for t in range(15):
        # Effective world output at t
        w_hat = res.population_hat_path[t]  # population-scaled
        y_world_t = float(np.sum(res.population_path[t]))  # in wage-normalized units
        tau_analytical = y_world_t * mean_gamma * brackets
        tau_numerical = float(res.pigouvian_tax_path[t])

        rel_diff = abs(tau_numerical - tau_analytical) / tau_analytical
        assert rel_diff < 1e-6, (
            f"Pigouvian tax {tau_numerical:.4f} differed from Golosov formula {tau_analytical:.4f} "
            f"by {rel_diff:.2e} at t={t}"
        )


def test_spatial_climate_migration_direction():
    """Verify climate-induced labor migration flow: outflow from hot damaged South to cooler North."""
    model = _build_3region_iam_model()
    res = solve_spatial_climate_iam(
        model,
        horizon=20,
        policy_scenario="laissez_faire",
    )

    # Region North (index 0) has low temperature & damages
    # Region South (index 1) has high temperature (28°C) & high damages
    north_pop_change = res.population_hat_path[-1, 0] - 1.0
    south_pop_change = res.population_hat_path[-1, 1] - 1.0

    # South should experience net population outflow, North should receive net inflow
    assert north_pop_change > south_pop_change, (
        f"Expected North population relative growth ({north_pop_change:.3f}) > South ({south_pop_change:.3f})"
    )


def test_spatial_climate_result_reporting_and_network_macro():
    """Verify result container summary, dataframe, markdown, latex, and network macro decomposition."""
    model = _build_3region_iam_model()
    res = solve_spatial_climate_iam(model, horizon=10)

    df = res.to_dataframe()
    assert df.shape == (3, 6)
    assert set(df.columns) >= {"population", "population_hat", "real_wage_hat", "temperature", "emissions", "welfare_hat"}

    summary = res.summary()
    assert len(summary) == 3

    md = res.to_markdown()
    assert "North" in md

    lat = res.to_latex()
    assert "North" in lat

    fig = res.plot()
    assert fig is not None

    # Verify Baqaee-Farhi network macro decomposition is present in metadata
    assert "network_macro" in res.metadata
    decomp = res.metadata["network_macro"]
    assert hasattr(decomp, "first_order_dlog_y")
    assert hasattr(decomp, "second_order_dlog_y")
    assert decomp.first_order_dlog_y <= 0.0  # climate damage causes negative aggregate TFP shift
