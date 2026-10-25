"""Tests for puremacro.spatial.dynamic_hat (Caliendo, Parro & Rossi-Hansberg 2019 dynamic hat-algebra)."""
from __future__ import annotations

import numpy as np

from puremacro.spatial import (
    DynamicSpatialHatModel,
    solve_dynamic_spatial_hat,
)


def _baseline_3region_economy() -> DynamicSpatialHatModel:
    # Symmetric stationary 3-region baseline so L0 = [100, 100, 100] is exact stationary state
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


def test_dynamic_spatial_hat_unit_shock_stationarity():
    model = _baseline_3region_economy()
    res = solve_dynamic_spatial_hat(model, horizon=25, tol=1e-13)

    assert res.converged
    assert np.max(np.abs(res.population_hat_path - 1.0)) < 1e-13
    assert np.max(np.abs(res.wage_hat_path - 1.0)) < 1e-13
    assert np.max(np.abs(res.price_hat_path - 1.0)) < 1e-13
    assert np.max(np.abs(res.real_wage_hat_path - 1.0)) < 1e-13
    assert np.max(np.abs(res.value_hat_path - 1.0)) < 1e-13
    assert res.population_conservation_error < 1e-13
    assert res.migration_row_stochastic_error < 1e-15


def test_dynamic_spatial_hat_regional_productivity_and_trade_shock():
    model = _baseline_3region_economy()
    # Permanent 15% TFP boom in North + bilateral trade integration between North and Center
    tfp_shock = np.array([1.15, 1.00, 1.00], dtype=np.float64)
    tau_shock = np.ones((3, 3), dtype=np.float64)
    tau_shock[0, 1] = 0.92
    tau_shock[1, 0] = 0.92

    res = solve_dynamic_spatial_hat(
        model,
        horizon=35,
        tfp_hat_path=tfp_shock,
        trade_cost_hat_path=tau_shock,
        tol=1e-11,
    )
    assert res.converged
    assert res.population_conservation_error < 1e-12
    assert res.migration_row_stochastic_error < 1e-14
    # North attracts migration over the transition
    assert res.population_hat_path[-1, 0] > 1.02
    assert res.population_hat_path[-1, 2] < 1.00
    # North welfare gain exceeds South welfare gain
    assert res.welfare_hat[0] > res.welfare_hat[2] > 1.0

    df = res.to_frame()
    assert list(df.index) == ["North", "Center", "South"]
    assert "North" in res.to_markdown()
    assert "\\begin{tabular}" in res.to_latex()
    assert "#table(" in res.to_typst()
