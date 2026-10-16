"""Unit and integration tests of the trade general equilibrium.

Covers:
- Tier 1: a synthetic 2-country 2-sector model: the calibrated state is an
  equilibrium, every solver converges on it, Newton and SciPy agree under a
  10% tariff, packing round-trips, and the result renders.
- Tier 2: a synthetic 70-country 12-sector table: the calibrated state solves
  the full system at the benchmark prices (no table dimension is assumed).

Runs offline on synthetic tables only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.trade import (
    TradeCalibrationResult,
    build_initial_guess,
    calibrate_trade_model,
    compute_equilibrium_residuals,
    compute_postprocessing_flows,
    evaluate_equilibrium_residuals,
    pack_equilibrium_vector,
    solve_trade_equilibrium,
    unpack_equilibrium_vector,
)
from puremacro.trade.data import synthetic_icio_table


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synthetic_2c_2s_calib() -> TradeCalibrationResult:
    """Construct a balanced, internally consistent 2-country 2-sector synthetic model."""
    nc, ns, nfd = 2, 2, 3
    data = np.zeros((ns * nc + 3, ns * nc + nfd * nc), dtype=float)

    # 1. Intermediate transactions block (4x4)
    data[:4, :4] = np.array([
        [10.0, 15.0, 5.0, 5.0],
        [15.0, 20.0, 10.0, 10.0],
        [5.0, 5.0, 12.0, 18.0],
        [10.0, 10.0, 18.0, 22.0],
    ])
    inter_col_sums = data[:4, :4].sum(axis=0)
    y = np.array([100.0, 150.0, 120.0, 180.0])
    va = y - inter_col_sums
    taxes = 0.05 * y
    va_fac = va - taxes
    labor = (2.0 / 3.0) * va_fac
    capital = (1.0 / 3.0) * va_fac

    data[4, :4] = taxes
    data[5, :4] = labor
    data[6, :4] = capital

    # 2. Final demand blocks (4x6)
    inter_row_sums = data[:4, :4].sum(axis=1)
    fd_row_sums = y - inter_row_sums

    for i in range(4):
        tot_fd = fd_row_sums[i]
        if i < 2:
            data[i, 4] = tot_fd * 0.50
            data[i, 5] = tot_fd * 0.25
            data[i, 6] = tot_fd * 0.05
            data[i, 7] = tot_fd * 0.10
            data[i, 8] = tot_fd * 0.08
            data[i, 9] = tot_fd * 0.02
        else:
            data[i, 4] = tot_fd * 0.10
            data[i, 5] = tot_fd * 0.08
            data[i, 6] = tot_fd * 0.02
            data[i, 7] = tot_fd * 0.50
            data[i, 8] = tot_fd * 0.25
            data[i, 9] = tot_fd * 0.05

    fd_col_sums = data[:4, 4:].sum(axis=0)
    data[4, 4:] = 0.02 * fd_col_sums

    return calibrate_trade_model(data, ns=ns, nc=nc, nfd=nfd, validate=True)


# ---------------------------------------------------------------------------
# Tier 1: Fast Synthetic 2x2 Unit Tests (< 0.1s)
# ---------------------------------------------------------------------------
class TestSyntheticEquilibrium:
    """Unit tests on synthetic 2-country 2-sector fixture."""

    def test_synthetic_initial_guess_residual(self, synthetic_2c_2s_calib: TradeCalibrationResult) -> None:
        """Verify that calibrated baseline parameters yield near-zero initial residual."""
        calib = synthetic_2c_2s_calib
        x0 = build_initial_guess(calib)
        # Length: 2*2*2 + 3*2 + (2-1) = 8 + 6 + 1 = 15
        assert len(x0) == 15

        res0 = compute_equilibrium_residuals(x0, calib)
        assert len(res0) == 15
        assert np.max(np.abs(res0)) < 1e-10

    def test_synthetic_newton_solve_baseline(self, synthetic_2c_2s_calib: TradeCalibrationResult) -> None:
        """Verify that pure NumPy Newton converges immediately on baseline."""
        calib = synthetic_2c_2s_calib
        res = solve_trade_equilibrium(calib, method="newton", tol=2.5e-3)
        assert res.converged is True
        assert res.iterations <= 1
        assert res.max_residual < 1e-10
        assert np.allclose(res.p_sol, 1.0, atol=1e-10)
        assert np.allclose(res.w_sol, 1.0, atol=1e-10)
        assert np.allclose(res.r_sol, 1.0, atol=1e-10)
        assert res.intermediate_flows is not None
        assert res.final_demand_flows is not None

    def test_synthetic_broyden_solve_baseline(self, synthetic_2c_2s_calib: TradeCalibrationResult) -> None:
        """Verify that pure NumPy Broyden converges on baseline."""
        calib = synthetic_2c_2s_calib
        res = solve_trade_equilibrium(calib, method="broyden", tol=2.5e-3)
        assert res.converged is True
        assert res.max_residual < 1e-10

    def test_synthetic_scipy_hybr_baseline(self, synthetic_2c_2s_calib: TradeCalibrationResult) -> None:
        """Verify that SciPy hybr wrapper converges on baseline."""
        calib = synthetic_2c_2s_calib
        res = solve_trade_equilibrium(calib, method="hybr", tol=1e-6)
        assert res.converged is True
        assert res.max_residual < 1e-8

    def test_synthetic_scipy_lm_baseline(self, synthetic_2c_2s_calib: TradeCalibrationResult) -> None:
        """Verify that SciPy Levenberg-Marquardt wrapper converges on baseline."""
        calib = synthetic_2c_2s_calib
        res = solve_trade_equilibrium(calib, method="lm", tol=1e-6)
        assert res.converged is True
        assert res.max_residual < 1e-8

    def test_synthetic_newton_and_hybr_parity_under_shock(
        self, synthetic_2c_2s_calib: TradeCalibrationResult
    ) -> None:
        """Verify that Newton and SciPy hybr converge to identical solutions under 10% tariff."""
        calib = synthetic_2c_2s_calib
        nc, ns = calib.n_countries, calib.n_sectors

        # 10% tariff on country 0 imports from country 1
        tau = np.ones((ns * nc, ns, nc), dtype=float)
        tau_fd = np.ones((ns * nc, calib.n_final_demand, nc), dtype=float)
        tau[2:, :, 0] = 1.10
        tau_fd[2:, :, 0] = 1.10

        res_newt = solve_trade_equilibrium(calib, tau=tau, tau_fd=tau_fd, method="newton", tol=2.5e-3)
        assert res_newt.converged is True
        assert res_newt.max_residual < 2.5e-3
        assert res_newt.iterations < 10

        res_hybr = solve_trade_equilibrium(calib, tau=tau, tau_fd=tau_fd, method="hybr", tol=1e-6)
        assert res_hybr.converged is True

        # Relative discrepancy between Newton and SciPy solutions < 1e-4
        max_rel_diff = np.max(np.abs(res_newt.x_sol - res_hybr.x_sol) / (np.abs(res_hybr.x_sol) + 1.0))
        assert max_rel_diff < 1e-4

    @pytest.mark.parametrize("method", ["hybr", "lm"])
    @pytest.mark.parametrize("tol", [1e-4, 1e-5, 1e-7])
    def test_scipy_fallbacks_meet_the_absolute_residual_tolerance(
        self, synthetic_2c_2s_calib: TradeCalibrationResult, method: str, tol: float
    ) -> None:
        """tol is an absolute residual bound, not MINPACK's relative step xtol.

        Up to 4.4.0 hybr stopped at max|F| = 1.06e-4, 1.38e-5 and 1.82e-7 for
        these tolerances and reported converged=False. The oracle is the
        residual recomputed with compute_equilibrium_residuals.
        """
        from puremacro.trade import compute_equilibrium_residuals

        calib = synthetic_2c_2s_calib
        nc, ns = calib.n_countries, calib.n_sectors
        tau = np.ones((ns * nc, ns, nc), dtype=float)
        tau_fd = np.ones((ns * nc, calib.n_final_demand, nc), dtype=float)
        tau[2:, :, 0] = 1.10
        tau_fd[2:, :, 0] = 1.10
        res = solve_trade_equilibrium(calib, tau=tau, tau_fd=tau_fd, method=method, tol=tol)
        assert res.converged is True
        recomputed = compute_equilibrium_residuals(res.x_sol, calib, tau=tau, tau_fd=tau_fd)
        assert np.max(np.abs(recomputed)) <= tol

    def test_packing_unpacking_roundtrip(self, synthetic_2c_2s_calib: TradeCalibrationResult) -> None:
        """Verify pack_equilibrium_vector and unpack_equilibrium_vector roundtrip."""
        calib = synthetic_2c_2s_calib
        nc, ns, nfd = calib.n_countries, calib.n_sectors, calib.n_final_demand
        x0 = build_initial_guess(calib)

        unpacked = unpack_equilibrium_vector(x0, ns=ns, nc=nc, nfd=nfd, return_levels=True)
        assert unpacked.p.shape == (1, ns, nc)
        assert unpacked.y.shape == (1, ns, nc)
        assert unpacked.r.shape == (1, 1, nc)
        assert unpacked.w.shape == (1, 1, nc)
        assert unpacked.T.shape == (1, 1, nc)
        assert unpacked.XN.shape == (nc - 1,)
        assert unpacked.invforT.shape == (1, nc)

        # Repack
        repacked = pack_equilibrium_vector(
            p=unpacked.p,
            y=unpacked.y,
            r=unpacked.r,
            w=unpacked.w,
            T=unpacked.T,
            XN=unpacked.XN,
            log_transformed=True,
        )
        assert np.allclose(x0, repacked, atol=1e-12)

    def test_postprocessing_synthetic(self, synthetic_2c_2s_calib: TradeCalibrationResult) -> None:
        """Verify post-processing reconstruction on synthetic model."""
        calib = synthetic_2c_2s_calib
        x0 = build_initial_guess(calib)
        flows = compute_postprocessing_flows(x0, calib)

        assert flows["intermediate_flows"].shape == (2, 2, 2, 2)
        assert flows["intermediate_matrix"].shape == (4, 4)
        assert flows["final_demand_flows"].shape == (2, 2, 3, 2)
        assert flows["final_demand_matrix"].shape == (4, 6)
        assert flows["gdp"].shape == (2,)
        assert flows["gdp_fc"].shape == (2,)
        assert flows["exports"].shape == (2,)
        assert flows["imports"].shape == (2,)
        assert flows["tariffs"].shape == (2,)
        assert flows["cpi"].shape == (2,)
        assert flows["terms_of_trade"].shape == (2,)

    def test_trade_equilibrium_result_presentation(
        self, synthetic_2c_2s_calib: TradeCalibrationResult
    ) -> None:
        """Verify presentation methods (.summary, .to_frame, .to_markdown, .to_latex)."""
        calib = synthetic_2c_2s_calib
        res = solve_trade_equilibrium(calib, method="newton")

        # Default summary table
        df_diag = res.summary(detailed=False)
        assert isinstance(df_diag, pd.DataFrame)
        assert "CONVERGED" in res.to_markdown(detailed=False)
        assert "\\begin{tabular}" in res.to_latex(detailed=False)

        # Detailed summary table
        df_detail = res.summary(detailed=True)
        assert isinstance(df_detail, pd.DataFrame)
        assert "GDP_Market_Prices" in df_detail.columns
        assert "Exports" in df_detail.columns
        assert "Imports" in df_detail.columns

        # Frame and table emitters
        assert isinstance(res.bilateral_trade_frame(), pd.DataFrame)
        assert isinstance(res.sectoral_tariffs_frame(), pd.DataFrame)


# ---------------------------------------------------------------------------
# Tier 2: a large synthetic table
# ---------------------------------------------------------------------------
class TestLargeSyntheticTable:
    """The full 70-country, 12-sector system on a synthetic table."""

    @pytest.fixture(scope="class")
    def calib_70x12(self) -> TradeCalibrationResult:
        return calibrate_trade_model(synthetic_icio_table(70, 12, seed=4))

    def test_benchmark_state_solves_the_full_system(self, calib_70x12: TradeCalibrationResult) -> None:
        calib = calib_70x12
        x0 = build_initial_guess(calib)
        assert len(x0) == 2 * 70 * 12 + 4 * 70 - 1
        residuals = compute_equilibrium_residuals(x0, calib)
        assert np.max(np.abs(residuals)) < 1e-8
        res = solve_trade_equilibrium(calib, method="newton", tol=1e-6)
        assert res.converged is True
        assert res.iterations == 0
        np.testing.assert_allclose(res.p_sol, 1.0, atol=1e-12)
        np.testing.assert_allclose(res.w_sol, 1.0, atol=1e-12)
        np.testing.assert_allclose(res.r_sol, 1.0, atol=1e-12)

    def test_alias_entry_point_matches(self, calib_70x12: TradeCalibrationResult) -> None:
        x0 = build_initial_guess(calib_70x12)
        np.testing.assert_array_equal(
            evaluate_equilibrium_residuals(x0, calib_70x12),
            compute_equilibrium_residuals(x0, calib_70x12),
        )
