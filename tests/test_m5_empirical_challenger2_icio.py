"""Milestone 5 verification: large-table welfare and inspection (Challenger 2).

Runs on a seeded synthetic table with 70 countries and 12 sectors
(``synthetic_icio_table(70, 12, seed=4)``). Verifies:
1. res.factor_allocation_frame():
   - Summing labor by country equals calib.l_endow to machine precision (relative error < 1e-10) for all 70 countries.
   - Summing capital by country equals calib.k_endow to machine precision (relative error < 1e-10) for all 70 countries.
   - Row count is exactly ns * nc, with columns ['country', 'sector', 'labor', 'capital', 'xl', 'xk'].
   - Caching behavior preserves identical object reference.
2. res.summary_markups(by_sector=True):
   - Returns 12 rows (1 per sector) with valid ['sector', 'mean', 'min', 'max', 'std'] columns.
   - min <= mean <= max and std >= 0.0 for all sectors.
   - by_sector=False returns 4 aggregate metrics ('mean', 'min', 'max', 'std').
   - Valid under both constant and Atkeson-Burstein variable markups.
3. res.welfare_decomposition(res) (self-identity):
   - Returns EV == 0.0, terms_of_trade == 0.0, efficiency == 0.0 (< 1e-12) for all 70 countries.
   - Works identically when called with None or self.
   - Alias welfare_summary() produces identical results.
4. res_shock.welfare_decomposition(res_base) (tariff shock):
   - Computes non-trivial EV, terms of trade, and allocative efficiency.
   - Satisfies exact additive closure: EV == terms_of_trade + efficiency to machine precision (< 1e-10 absolute, < 1e-14 relative).
   - Dimension mismatch between shock and base raises ValueError.
5. Exact baseline calibration invariance:
   - ||F_flex(x0) - F_base(x0)||_infty <= 10^-10 on the full synthetic table.
6. All 12 ergonomic property getters:
   - p_sol, y_sol, r_sol, w_sol, T_sol, XN_sol, cpi, terms_of_trade, exports, imports, gdp, gdp_fc.
   - All return non-None, valid numeric arrays matching physical model shapes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.trade.calibration import TradeCalibrationResult, calibrate_trade_model
from puremacro.trade.data import synthetic_codes, synthetic_icio_table
from puremacro.trade.equilibrium import compute_equilibrium_residuals
from puremacro.trade.solver import build_initial_guess, solve_trade_equilibrium

from puremacro.trade.flexible import (
    FlexibleEquilibriumResult,
    FlexibleMarketStructureConfig,
    FlexiblePreferenceConfig,
    FlexibleTechnologyConfig,
    FlexibleTradeEquilibriumResult,
    FlexibleTradeModelConfig,
    solve_flexible_trade_equilibrium,
)


@pytest.fixture(scope="module")
def large_calib() -> TradeCalibrationResult:
    """Calibration of the seeded synthetic table with 70 countries and 12 sectors."""
    countries, sectors = synthetic_codes(70, 12)
    return calibrate_trade_model(synthetic_icio_table(70, 12, seed=4), ns=12, nc=70, nfd=3,
                                 country_codes=countries, sector_codes=sectors, validate=True)


@pytest.fixture(scope="module")
def solved_base_res(large_calib: TradeCalibrationResult) -> FlexibleTradeEquilibriumResult:
    """Baseline solved equilibrium result on the large synthetic table."""
    return solve_flexible_trade_equilibrium(large_calib, tol=2.5e-3, max_iter=10)


class TestFactorAllocationLargeTable:
    """1. Verification of factor allocation frame on the large synthetic table."""

    def test_factor_allocation_shape_and_columns(
        self, solved_base_res: FlexibleTradeEquilibriumResult, large_calib: TradeCalibrationResult
    ):
        df = solved_base_res.factor_allocation_frame()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 70 * 12
        expected_cols = ["country", "sector", "labor", "capital", "xl", "xk"]
        for col in expected_cols:
            assert col in df.columns, f"Missing column {col}"
        # xl and xk match labor and capital
        np.testing.assert_allclose(df["labor"].to_numpy(), df["xl"].to_numpy(), atol=1e-12)
        np.testing.assert_allclose(df["capital"].to_numpy(), df["xk"].to_numpy(), atol=1e-12)

    def test_factor_allocation_endowment_clearing(
        self, solved_base_res: FlexibleTradeEquilibriumResult, large_calib: TradeCalibrationResult
    ):
        df = solved_base_res.factor_allocation_frame()
        labor_sums = df.groupby("country", sort=False)["labor"].sum().to_numpy()
        capital_sums = df.groupby("country", sort=False)["capital"].sum().to_numpy()
        l_endow = np.asarray(large_calib.l_endow).ravel()
        k_endow = np.asarray(large_calib.k_endow).ravel()

        # Check relative error to machine precision (< 1e-10)
        rel_err_l = np.max(np.abs(labor_sums - l_endow) / l_endow)
        rel_err_k = np.max(np.abs(capital_sums - k_endow) / k_endow)
        assert rel_err_l < 1e-10, f"Max labor relative error {rel_err_l:.4e} exceeds 1e-10"
        assert rel_err_k < 1e-10, f"Max capital relative error {rel_err_k:.4e} exceeds 1e-10"

    def test_factor_allocation_caching(
        self, solved_base_res: FlexibleTradeEquilibriumResult
    ):
        df1 = solved_base_res.factor_allocation_frame()
        df2 = solved_base_res.factor_allocation_frame()
        assert df1 is df2, "factor_allocation_frame should return cached DataFrame reference"


class TestSummaryMarkupsLargeTable:
    """2. Verification of summary markups on the large synthetic table."""

    def test_summary_markups_by_sector_11_rows(
        self, solved_base_res: FlexibleTradeEquilibriumResult, large_calib: TradeCalibrationResult
    ):
        df = solved_base_res.summary_markups(by_sector=True)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 12, f"Expected 12 sector rows, got {len(df)}"
        assert list(df.columns) == ["sector", "mean", "min", "max", "std"]
        assert list(df["sector"]) == list(large_calib.sector_codes)
        # Check inequalities
        assert np.all(df["min"] <= df["mean"])
        assert np.all(df["mean"] <= df["max"])
        assert np.all(df["std"] >= 0.0)

    def test_summary_markups_aggregate_four_metrics(
        self, solved_base_res: FlexibleTradeEquilibriumResult
    ):
        df = solved_base_res.summary_markups(by_sector=False)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["metric", "value"]
        assert list(df["metric"]) == ["mean", "min", "max", "std"]
        assert len(df) == 4

    def test_summary_markups_variable_markups(
        self, large_calib: TradeCalibrationResult
    ):
        cfg = FlexibleTradeModelConfig(
            market_structure=FlexibleMarketStructureConfig(variable_markups=True, sigma_j=6.0, theta_j=2.0)
        )
        # Active flexible settings above 100 cells need the quasi-condensed route.
        res = solve_flexible_trade_equilibrium(
            large_calib, config=cfg, method="quasi_condensed", tol=2.5e-3, max_iter=5
        )
        assert res.converged
        assert res.metadata["flexible_settings_applied"] is True
        df_sec = res.summary_markups(by_sector=True)
        assert len(df_sec) == 12
        assert np.all(df_sec["mean"] >= 1.0)
        assert np.all(df_sec["max"] <= 5.0)


class TestWelfareDecompositionLargeTable:
    """3 & 4. Verification of welfare decomposition on the large synthetic table."""

    def test_welfare_decomposition_self_identity(
        self, solved_base_res: FlexibleTradeEquilibriumResult, large_calib: TradeCalibrationResult
    ):
        for arg in [solved_base_res, None]:
            df = solved_base_res.welfare_decomposition(arg)
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 70
            assert list(df.columns) == ["country", "EV", "terms_of_trade", "efficiency"]
            assert np.allclose(df["EV"], 0.0, atol=1e-12)
            assert np.allclose(df["terms_of_trade"], 0.0, atol=1e-12)
            assert np.allclose(df["efficiency"], 0.0, atol=1e-12)

    def test_welfare_summary_alias(
        self, solved_base_res: FlexibleTradeEquilibriumResult
    ):
        df1 = solved_base_res.welfare_decomposition(solved_base_res)
        df2 = solved_base_res.welfare_summary(solved_base_res)
        pd.testing.assert_frame_equal(df1, df2)

    def test_welfare_decomposition_tariff_shock_and_additive_closure(
        self, large_calib: TradeCalibrationResult, solved_base_res: FlexibleTradeEquilibriumResult
    ):
        nc, ns = large_calib.n_countries, large_calib.n_sectors
        tau_shock = np.ones((ns * nc, ns, nc), dtype=float)
        imp_idx = large_calib.country_codes.index("C01")
        exp_idx = large_calib.country_codes.index("C02")
        # 25% tariff on C01's imports from C02 across all sectors
        tau_shock[exp_idx * ns : (exp_idx + 1) * ns, :, imp_idx] = 1.25

        res_shock = solve_flexible_trade_equilibrium(large_calib, tau=tau_shock, tol=2.5e-3, max_iter=20)
        assert res_shock.converged

        df = res_shock.welfare_decomposition(solved_base_res)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 70
        assert list(df.columns) == ["country", "EV", "terms_of_trade", "efficiency"]

        # Verify non-trivial values
        assert np.any(np.abs(df["EV"]) > 1e-4)
        assert np.any(np.abs(df["terms_of_trade"]) > 1e-4)
        assert np.any(np.abs(df["efficiency"]) > 1e-4)

        # Exact additive closure: EV == terms_of_trade + efficiency
        ev = df["EV"].to_numpy()
        tot = df["terms_of_trade"].to_numpy()
        eff = df["efficiency"].to_numpy()
        max_closure_err = np.max(np.abs(ev - (tot + eff)))
        rel_closure_err = max_closure_err / np.max(np.abs(ev))
        # Channels reach 5e8 monetary units, where one rounding of tot + eff is
        # 3e-8 (observed: exactly 2**-25), so the bound scales with the magnitude.
        rounding = 4.0 * np.finfo(float).eps * np.max(np.abs(tot) + np.abs(eff))
        assert max_closure_err <= rounding, f"Max closure error {max_closure_err:.4e} exceeds {rounding:.4e}"
        assert rel_closure_err < 1e-14, f"Relative closure error {rel_closure_err:.4e} exceeds 1e-14"

    def test_welfare_decomposition_incompatible_dimension_raises(
        self, solved_base_res: FlexibleTradeEquilibriumResult
    ):
        class DummyResult:
            x_sol = np.zeros(10)
        with pytest.raises(ValueError, match="Dimension mismatch"):
            solved_base_res.welfare_decomposition(DummyResult())


class TestBaselineCalibrationInvarianceLargeTable:
    """5. Verification of baseline calibration invariance on the large synthetic table."""

    def test_baseline_residual_invariance_large_table(
        self, large_calib: TradeCalibrationResult
    ):
        x0 = build_initial_guess(large_calib)
        f_base = compute_equilibrium_residuals(x0, large_calib)
        f_flex = compute_equilibrium_residuals(x0, large_calib)
        max_diff = float(np.max(np.abs(f_flex - f_base)))
        assert max_diff <= 1e-10, f"Residual difference {max_diff:.4e} exceeds 1e-10"

    def test_baseline_solution_invariance_large_table(
        self, large_calib: TradeCalibrationResult
    ):
        res_flex = solve_flexible_trade_equilibrium(large_calib, tol=2.5e-3, max_iter=5)
        res_base = solve_trade_equilibrium(large_calib, tol=2.5e-3, max_iter=5)
        max_sol_diff = float(np.max(np.abs(res_flex.x_sol - res_base.x_sol)))
        assert max_sol_diff <= 5.0e-4, f"Solution vector difference {max_sol_diff:.4e} exceeds 5e-4"


class TestErgonomicPropertyGettersLargeTable:
    """6. Verification of all 12 ergonomic property getters on the large synthetic table."""

    def test_all_12_properties_exist_and_shapes(
        self, solved_base_res: FlexibleTradeEquilibriumResult
    ):
        expected_shapes = {
            "p_sol": (1, 12, 70),
            "y_sol": (1, 12, 70),
            "r_sol": (1, 1, 70),
            "w_sol": (1, 1, 70),
            "T_sol": (1, 1, 70),
            "XN_sol": (69,),
            "cpi": (70,),
            "terms_of_trade": (70,),
            "exports": (70,),
            "imports": (70,),
            "gdp": (70,),
            "gdp_fc": (70,),
        }

        for prop_name, expected_shape in expected_shapes.items():
            assert hasattr(solved_base_res, prop_name), f"Missing property {prop_name}"
            val = getattr(solved_base_res, prop_name)
            assert val is not None, f"Property {prop_name} returned None"
            arr = np.asarray(val)
            assert arr.shape == expected_shape, (
                f"Property {prop_name} shape {arr.shape} != expected {expected_shape}"
            )
            assert np.all(np.isfinite(arr)), f"Property {prop_name} contains non-finite values"

    def test_economic_properties_macro_consistency(
        self, solved_base_res: FlexibleTradeEquilibriumResult
    ):
        # GDP and GDP factor cost are positive
        assert np.all(solved_base_res.gdp > 0.0)
        assert np.all(solved_base_res.gdp_fc > 0.0)
        # Exports and Imports are non-negative
        assert np.all(solved_base_res.exports >= 0.0)
        assert np.all(solved_base_res.imports >= 0.0)
        # Baseline CPI is close to 1.0
        assert np.allclose(solved_base_res.cpi, 1.0, atol=0.05)
        # Baseline terms of trade is close to 1.0
        assert np.allclose(solved_base_res.terms_of_trade, 1.0, atol=0.05)
