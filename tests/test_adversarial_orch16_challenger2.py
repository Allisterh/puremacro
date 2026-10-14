"""Adversarial stress-test suite for the Allen-Arkolakis (2014) spatial GE engine.

Author: Challenger 2 (orchestrator_16 adversarial verification)
Scope, Allen-Arkolakis (2014):
   - Exact uniqueness boundary alpha + beta = theta / (1 + theta)
   - Asymmetric trade cost matrices (tau_{ij} != tau_{ji})
   - Localized climate and environmental shocks (mild, severe, and 99% catastrophe)
   - Verification of labor mass conservation (|sum L_i - L_bar| < 1e-12)
   - Verification of spatial price and utility equalization (std(u_i) / u_bar < 1e-7)
"""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.spatial.allen_arkolakis import AllenArkolakisModel


# =============================================================================
# ALLEN-ARKOLAKIS (2014) ADVERSARIAL TEST SUITE
# =============================================================================

class TestAllenArkolakisAdversarial:
    """Adversarial challenge tests for Allen-Arkolakis (2014) spatial GE engine."""

    def test_exact_uniqueness_boundary(self):
        """Stress-test exact uniqueness boundary alpha + beta = theta / (1 + theta)."""
        # Parameter set: theta = 1.0, beta = -0.20, alpha = 0.70
        # Condition: alpha + beta = 0.50 == theta / (1 + theta) = 0.50, and alpha <= 1 / theta = 1.0
        trade_costs = np.array([
            [1.0, 1.3],
            [1.3, 1.0],
        ])
        fund_A = np.array([1.0, 1.2])
        fund_a = np.array([1.0, 0.9])
        total_pop = 100.0

        model = AllenArkolakisModel(
            trade_costs=trade_costs,
            fundamental_productivity=fund_A,
            fundamental_amenity=fund_a,
            theta=1.0,
            alpha=0.70,
            beta=-0.20,
            total_population=total_pop,
        )
        assert model.is_unique

        res = model.solve_equilibrium()
        assert res.converged
        # Verify exact labor mass conservation
        assert res.labor_conservation_residual < 1e-12
        np.testing.assert_allclose(np.sum(res.population), total_pop, atol=1e-12)
        # Verify spatial utility equalization
        assert res.spatial_utility_variance < 1e-7

    def test_uniqueness_boundary_positive_beta(self):
        """Stress-test boundary with beta > 0 (theta = 2.0, alpha = 0.5, beta = 1/6)."""
        trade_costs = np.array([
            [1.0, 1.3],
            [1.3, 1.0],
        ])
        fund_A = np.array([1.0, 1.2])
        fund_a = np.array([1.0, 0.9])
        total_pop = 100.0

        with pytest.warns(UserWarning, match="beta is non-negative"):
            model = AllenArkolakisModel(
                trade_costs=trade_costs,
                fundamental_productivity=fund_A,
                fundamental_amenity=fund_a,
                theta=2.0,
                alpha=0.50,
                beta=2.0 / 3.0 - 0.50,  # 1/6
                total_population=total_pop,
            )
        assert model.is_unique

        res = model.solve_equilibrium()
        assert res.converged
        assert res.labor_conservation_residual < 1e-12
        assert res.spatial_utility_variance < 1e-7

    def test_asymmetric_trade_cost_matrix(self):
        """Stress-test strongly asymmetric trade cost matrix tau_{ij} != tau_{ji}."""
        trade_costs = np.array([
            [1.0, 1.2, 1.8, 2.5],
            [1.5, 1.0, 1.3, 2.0],
            [2.2, 1.6, 1.0, 1.4],
            [3.0, 2.4, 1.7, 1.0],
        ])
        fund_A = np.array([1.2, 1.0, 0.9, 1.1])
        fund_a = np.array([1.0, 1.3, 0.8, 1.0])
        total_pop = 250.0

        model = AllenArkolakisModel(
            trade_costs=trade_costs,
            fundamental_productivity=fund_A,
            fundamental_amenity=fund_a,
            theta=4.0,
            alpha=0.08,
            beta=-0.35,
            total_population=total_pop,
            region_names=["North", "East", "South", "West"],
        )

        res = model.solve_equilibrium(tol=1e-10, max_iter=3000)
        assert res.converged
        assert res.labor_conservation_residual < 1e-12
        np.testing.assert_allclose(np.sum(res.population), total_pop, atol=1e-12)
        assert res.spatial_utility_variance < 1e-7

        # Check goods market clearing: income == sales
        income = res.wages * res.population
        sales = np.sum(res.trade_shares * income[:, None], axis=0)
        max_mc_res = float(np.max(np.abs(income - sales)))
        assert max_mc_res < 1e-6, f"Goods market clearing error {max_mc_res} >= 1e-6"

    @pytest.mark.parametrize("shock_level", [0.70, 0.20, 0.01])
    def test_localized_climate_shocks(self, shock_level: float):
        """Stress-test mild (30%), severe (80%), and catastrophic (99%) localized productivity shocks."""
        coords = np.array([
            [40.7128, -74.0060],  # NYC
            [34.0522, -118.2437], # LA
            [41.8781, -87.6298],  # Chicago
            [29.7604, -95.3698],  # Houston
        ])
        names = ["NYC", "LA", "CHI", "HOU"]
        total_pop = 1000.0

        model = AllenArkolakisModel.from_coordinates(
            coords,
            region_names=names,
            theta=4.0,
            alpha=0.08,
            beta=-0.35,
            total_population=total_pop,
        )

        res = model.simulate_climate_shock(productivity_shocks={"HOU": shock_level})
        assert res.converged
        # Strict labor mass conservation
        assert res.labor_conservation_residual < 1e-12
        np.testing.assert_allclose(np.sum(res.population), total_pop, atol=1e-12)
        # Spatial utility equalization
        assert res.spatial_utility_variance < 1e-7
        # Populations strictly positive
        assert np.all(res.population > 0.0)
        # Shocked location loses population
        assert res.L_hat is not None and res.L_hat[3] < 1.0
        # Aggregate welfare falls
        assert res.welfare_pct is not None and res.welfare_pct < 0.0
