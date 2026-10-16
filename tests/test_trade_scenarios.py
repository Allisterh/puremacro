"""Unit tests of tariff scenarios, the batch runner and Geary-Khamis PPP.

Covers:
- Synthetic model unit tests: solve_multilateral_ppp (iterative and linear),
  restore_capital_formation, build_tariff_matrices, run_scenario and
  run_scenario_batch on a 2-country 2-sector model.
- Negative tests: removed named scenarios, unknown country/sector codes,
  negative or non-numeric tariffs, invalid PPP options.

Runs offline on synthetic tables only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.trade import (
    ScenarioBatchResult,
    TariffScenario,
    TradeCalibrationResult,
    TradeEquilibriumResult,
    build_tariff_matrices,
    calibrate_trade_model,
    restore_capital_formation,
    run_scenario,
    run_scenario_batch,
    solve_multilateral_ppp,
)


@pytest.fixture(scope="module")
def synthetic_2c_2s_calib() -> TradeCalibrationResult:
    """Construct a mathematically balanced 2-country 2-sector synthetic model."""
    nc, ns, nfd = 2, 2, 3
    data = np.zeros((ns * nc + 3, ns * nc + nfd * nc), dtype=float)

    # Intermediate transactions (4x4)
    data[:4, :4] = np.array([
        [10.0, 15.0, 5.0, 5.0],
        [15.0, 20.0, 10.0, 10.0],
        [5.0, 5.0, 12.0, 18.0],
        [10.0, 10.0, 18.0, 22.0],
    ])
    y = np.array([100.0, 150.0, 120.0, 180.0])
    inter_col_sums = data[:4, :4].sum(axis=0)
    va = y - inter_col_sums
    taxes = 0.05 * y
    va_fac = va - taxes
    data[4, :4] = taxes
    data[5, :4] = (2.0 / 3.0) * va_fac
    data[6, :4] = (1.0 / 3.0) * va_fac

    # Final demand (4x6)
    fd_row_sums = y - data[:4, :4].sum(axis=1)
    for i in range(4):
        tot_fd = fd_row_sums[i]
        shares = [0.50, 0.25, 0.05, 0.10, 0.08, 0.02] if i < 2 else [0.10, 0.08, 0.02, 0.50, 0.25, 0.05]
        for c_idx, sh in enumerate(shares):
            data[i, 4 + c_idx] = tot_fd * sh

    data[4, 4:] = 0.02 * data[:4, 4:].sum(axis=0)
    return calibrate_trade_model(
        data,
        ns=ns,
        nc=nc,
        nfd=nfd,
        country_codes=("C1", "C2"),
        sector_codes=("S1", "S2"),
        validate=True,
    )


# ---------------------------------------------------------------------------
# Synthetic model unit tests
# ---------------------------------------------------------------------------

class TestSyntheticModelUnit:
    """Fast synthetic unit tests ensuring fundamental algorithmic correctness."""

    def test_solve_multilateral_ppp_synthetic(self) -> None:
        """Test Geary-Khamis iterative and linear solvers on synthetic prices and quantities."""
        p = np.array([[1.0, 1.2], [1.1, 0.9], [1.05, 1.15]])
        q = np.array([[100.0, 120.0], [50.0, 60.0], [10.0, 15.0]])

        # 1. Iterative solve
        pi_iter, ppp_iter, conv_iter, it_count = solve_multilateral_ppp(
            p=p, q=q, method="iterative", tol=1e-12
        )
        assert conv_iter is True
        assert it_count > 0
        assert pi_iter.shape == (3,)
        assert ppp_iter.shape == (2,)

        # 2. Linear solve
        pi_lin, ppp_lin, conv_lin, it_lin = solve_multilateral_ppp(
            p=p, q=q, method="linear"
        )
        assert conv_lin is True
        assert it_lin == 1

        # 3. Assert exact mathematical equivalence between iterative and linear
        np.testing.assert_allclose(pi_iter, pi_lin, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(ppp_iter, ppp_lin, rtol=1e-10, atol=1e-10)

    def test_solve_multilateral_ppp_normalization(self) -> None:
        """Test normalization options pi1 and numeraire."""
        p = np.array([[1.0, 1.2], [1.1, 0.9], [1.05, 1.15]])
        q = np.array([[100.0, 120.0], [50.0, 60.0], [10.0, 15.0]])

        # normalize='pi1'
        pi_n1, ppp_n1, _, _ = solve_multilateral_ppp(p=p, q=q, normalize="pi1")
        assert pytest.approx(pi_n1[0], rel=1e-10) == 1.0

        # normalize='numeraire' needs the country index; no country is implied
        for idx in (0, 1):
            pi_n2, ppp_n2, _, _ = solve_multilateral_ppp(p=p, q=q, normalize="numeraire", numeraire_idx=idx)
            assert pytest.approx(ppp_n2[idx], rel=1e-10) == 1.0
        with pytest.raises(ValueError, match="numeraire_idx"):
            solve_multilateral_ppp(p=p, q=q, normalize="numeraire")

    def test_restore_capital_formation(self) -> None:
        """Foreign investment is added to the investment category; balances add to zero."""
        c = np.ones((3, 4)) * 10.0
        xn = np.array([2.0, -1.0, 0.5])  # 3 countries given, 4th is residual

        # The fourth balance is -sum(XN[:3]) = -1.5
        c_adj = restore_capital_formation(c, xn)
        assert c_adj.shape == (3, 4)
        np.testing.assert_allclose(c_adj[1, :], [12.0, 9.0, 10.5, 8.5])
        np.testing.assert_array_equal(c_adj[[0, 2], :], 10.0)  # other categories untouched

        # A full vector is used as given
        full = restore_capital_formation(c, np.array([2.0, -1.0, 0.5, -1.5]))
        np.testing.assert_array_equal(full, c_adj)

        # A single final-use category takes the balances itself
        single = restore_capital_formation(np.ones((1, 1, 4)), xn)
        np.testing.assert_allclose(single[0, 0], [3.0, 0.0, 1.5, -0.5])

    def test_build_tariff_matrices_synthetic(
        self, synthetic_2c_2s_calib: TradeCalibrationResult
    ) -> None:
        """build_tariff_matrices creates the 4D tensors and keeps domestic sales at 1.0."""
        calib = synthetic_2c_2s_calib
        scen = TariffScenario("C1", {("C2", "*"): 0.25}, name="test_shock")
        tau, tau_fd = build_tariff_matrices(calib, scen)

        assert tau.shape == (2, 2, 2, 2)
        assert tau_fd.shape == (2, 2, 3, 2)

        # Domestic diagonal: origin == destination must strictly equal 1.0
        for k in range(2):
            assert np.all(tau[:, k, :, k] == 1.0)
            assert np.all(tau_fd[:, k, :, k] == 1.0)
        np.testing.assert_allclose(tau[:, 1, :, 0], 1.25)
        np.testing.assert_allclose(tau[:, 0, :, 1], 1.0)

    def test_run_scenario_synthetic(
        self, synthetic_2c_2s_calib: TradeCalibrationResult
    ) -> None:
        """run_scenario solves the general equilibrium of a zero-tariff scenario."""
        calib = synthetic_2c_2s_calib
        res = run_scenario(calib, TariffScenario("C1", {}, name="no_tariff"), max_iter=20)
        assert isinstance(res, TradeEquilibriumResult)
        assert res.converged is True
        assert res.metadata["scenario_name"] == "no_tariff"

    def test_run_scenario_batch_synthetic(
        self, synthetic_2c_2s_calib: TradeCalibrationResult
    ) -> None:
        """run_scenario_batch executes and builds comparative tables on synthetic model."""
        calib = synthetic_2c_2s_calib
        scens = [TariffScenario("C1", {("C2", "*"): 0.10}, name="ten_percent")]
        batch = run_scenario_batch(calib, scens, warm_start=True, compute_gk=True)
        assert isinstance(batch, ScenarioBatchResult)
        assert "baseline" in batch.scenarios
        assert "ten_percent" in batch.scenarios
        assert "ten_percent" in batch.geary_khamis
        assert isinstance(batch.real_gdp_table, pd.DataFrame)
        assert batch.real_gdp_table.shape == (2, 2)


# ---------------------------------------------------------------------------
# Negative Tests: Error Handling and Validation
# ---------------------------------------------------------------------------

class TestNegativeScenariosAndPPP:
    """Validate strict defensive error handling on invalid specifications."""

    def test_named_scenarios_removed(
        self, synthetic_2c_2s_calib: TradeCalibrationResult
    ) -> None:
        """A scenario name raises TypeError pointing at TariffScenario."""
        with pytest.raises(TypeError, match="named scenarios were removed"):
            build_tariff_matrices(synthetic_2c_2s_calib, "unknown_shock_scenario")
        with pytest.raises(TypeError, match="TariffScenario"):
            run_scenario(synthetic_2c_2s_calib, "unknown_shock_scenario")

    def test_invalid_country_code(
        self, synthetic_2c_2s_calib: TradeCalibrationResult
    ) -> None:
        """Verify specifying unknown country code in tariffs raises ValueError."""
        scen = TariffScenario("C1", {("NONEXISTENT_COUNTRY", "*"): 0.25}, name="bad_country")
        with pytest.raises(ValueError, match="Unknown country code"):
            build_tariff_matrices(synthetic_2c_2s_calib, scen)

    def test_invalid_sector_code(
        self, synthetic_2c_2s_calib: TradeCalibrationResult
    ) -> None:
        """Verify specifying unknown sector code raises ValueError."""
        scen = TariffScenario("C2", {("C1", "NONEXISTENT_SECTOR"): 0.25}, name="bad_sector")
        with pytest.raises(ValueError, match="Unknown sector code"):
            build_tariff_matrices(synthetic_2c_2s_calib, scen)

    def test_negative_tariff_rate(self) -> None:
        """Verify specifying negative tariff rate raises ValueError."""
        with pytest.raises(ValueError, match="Tariff rate"):
            TariffScenario("C2", {("C1", "*"): -0.20}, name="bad_rate")
        with pytest.raises(ValueError, match="Tariff rates"):
            TariffScenario("C2", -0.1 * np.ones((2, 2)), name="bad_rate")

    def test_non_numeric_tariff_rate(self) -> None:
        """Verify non-numeric or NaN tariff rate raises ValueError or TypeError."""
        with pytest.raises((ValueError, TypeError)):
            TariffScenario("C2", {("C1", "*"): float("nan")}, name="nan_rate")

        with pytest.raises((ValueError, TypeError)):
            TariffScenario("C2", {("C1", "*"): "10%"}, name="str_rate")  # type: ignore[dict-item]

    def test_unsupported_ppp_method(self) -> None:
        """Verify requesting unknown PPP method raises ValueError."""
        p = np.ones((3, 2))
        q = np.ones((3, 2))
        with pytest.raises(ValueError, match="Unsupported method"):
            solve_multilateral_ppp(p, q, method="quantum_annealing")

    def test_unsupported_normalize_option(self) -> None:
        """Verify requesting unknown normalization option raises ValueError."""
        p = np.ones((3, 2))
        q = np.ones((3, 2))
        with pytest.raises(ValueError, match="Unsupported normalize"):
            solve_multilateral_ppp(p, q, normalize="invalid_normalization")

    def test_mismatched_ppp_shapes(self) -> None:
        """Verify mismatched price and quantity shapes raise ValueError."""
        p = np.ones((3, 2))
        q = np.ones((3, 5))
        with pytest.raises(ValueError, match="Shape mismatch"):
            solve_multilateral_ppp(p, q)

    def test_invalid_scenario_type(
        self, synthetic_2c_2s_calib: TradeCalibrationResult
    ) -> None:
        """Verify passing invalid scenario object type raises TypeError."""
        with pytest.raises(TypeError, match="scenario must be"):
            build_tariff_matrices(synthetic_2c_2s_calib, 12345)  # type: ignore[arg-type]
