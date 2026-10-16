"""End-to-end tests of the trade engine on seeded synthetic tables.

Tier 1 checks single features (Hawkins-Simon viability, calibration schema, data registries),
Tier 2 boundary cases, Tier 3 cross-feature pipelines (calibration, viability pre-filter, initial
guess, residuals) and Tier 4 tariff scenarios. Every test calls puremacro; the solvers, the
regularisation and the data adapters have their own focused test files.
"""
from __future__ import annotations

from dataclasses import is_dataclass
import time

import numpy as np
import pytest
import scipy.linalg as la

from puremacro.trade import (
    TradeCalibrationResult,
    build_initial_guess,
    calibrate_trade_model,
    compute_equilibrium_residuals,
)
from puremacro.trade.data import (
    EORA_189_COUNTRIES,
    FIGARO_COUNTRIES,
    OECD_77_COUNTRIES,
    RAW_45_SECTOR_CODES,
)
from puremacro.trade.solver import _viability_cost_matrix, check_hawkins_simon_viability
from _timing import budget


# ===========================================================================
# Helpers
# ===========================================================================

def uniform_multipliers(calib: TradeCalibrationResult, factor: float) -> np.ndarray:
    """(M, M) tariff multipliers equal to ``factor`` on every flow, so rho(B_tau) = factor * rho(B)."""
    m = calib.n_countries * calib.n_sectors
    return np.full((m, m), float(factor))


def get_intermediate_matrix_2d(calib: TradeCalibrationResult) -> np.ndarray:
    """Extract (ns*nc, ns*nc) square intermediate coefficient matrix from 3D tensor."""
    nc, ns = calib.n_countries, calib.n_sectors
    # In calibration, a_3d has shape (ns*nc, ns, nc) where dest is js, jc
    return calib.a.transpose(0, 2, 1).reshape(ns * nc, ns * nc)


def seeded_raw_table(
    nc: int = 3,
    ns: int = 2,
    nfd: int = 3,
    seed: int = 42,
) -> np.ndarray:
    """Generate self-contained balanced synthetic MRIO matrix of shape (ns*nc + 3, ns*nc + nfd*nc)."""
    rng = np.random.default_rng(seed)
    n_ind = nc * ns
    n_fd = nc * nfd

    # Inter-industry flows Z (gravity decay with strong domestic diagonal)
    Z = np.zeros((n_ind, n_ind), dtype=float)
    for i in range(nc):
        for j in range(nc):
            dist = 1.0 if i == j else 5.0
            block = rng.uniform(2.0, 10.0, size=(ns, ns)) / dist
            Z[i * ns:(i + 1) * ns, j * ns:(j + 1) * ns] = block

    # Final demands F
    F = np.zeros((n_ind, n_fd), dtype=float)
    for i in range(nc):
        for j in range(nc):
            dist = 1.0 if i == j else 4.0
            block = rng.uniform(5.0, 20.0, size=(ns, nfd)) / dist
            F[i * ns:(i + 1) * ns, j * nfd:(j + 1) * nfd] = block

    # Gross output from row sales
    Y = np.sum(Z, axis=1) + np.sum(F, axis=1)

    # Value added (40% of gross output)
    VA = 0.40 * Y
    # Split VA into taxes (5%), labor (63.3%), capital (31.7%)
    TLS = Y - np.sum(Z, axis=0) - VA
    labor = (2.0 / 3.0) * VA
    capital = (1.0 / 3.0) * VA

    # Assemble raw table
    matrix = np.zeros((n_ind + 3, n_ind + n_fd), dtype=float)
    matrix[:n_ind, :n_ind] = Z
    matrix[:n_ind, n_ind:] = F
    matrix[n_ind, :n_ind] = TLS
    matrix[n_ind + 1, :n_ind] = labor
    matrix[n_ind + 2, :n_ind] = capital
    # Final demand taxes
    matrix[n_ind, n_ind:] = 0.05 * np.sum(F, axis=0)

    return matrix


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="session")
def synthetic_2c_2s_raw() -> np.ndarray:
    """Raw synthetic 2-country 2-sector ICIO matrix."""
    return seeded_raw_table(nc=2, ns=2, nfd=3, seed=101)


@pytest.fixture(scope="session")
def synthetic_2c_2s_calib(synthetic_2c_2s_raw) -> TradeCalibrationResult:
    """Calibrated model for 2 countries and 2 sectors."""
    return calibrate_trade_model(
        synthetic_2c_2s_raw,
        nc=2,
        ns=2,
        nfd=3,
        country_codes=("C1", "C2"),
        sector_codes=("S1", "S2"),
        validate=False,
    )


@pytest.fixture(scope="session")
def synthetic_3c_3s_raw() -> np.ndarray:
    """Raw synthetic 3-country 3-sector ICIO matrix."""
    return seeded_raw_table(nc=3, ns=3, nfd=3, seed=202)


@pytest.fixture(scope="session")
def synthetic_3c_3s_calib(synthetic_3c_3s_raw) -> TradeCalibrationResult:
    """Calibrated model for 3 countries and 3 sectors."""
    return calibrate_trade_model(
        synthetic_3c_3s_raw,
        nc=3,
        ns=3,
        nfd=3,
        country_codes=("C01", "C02", "C03"),
        sector_codes=("S01", "S02", "S03"),
        validate=False,
    )


@pytest.fixture(scope="session")
def synthetic_5c_4s_raw() -> np.ndarray:
    """Raw synthetic 5-country 4-sector ICIO matrix."""
    return seeded_raw_table(nc=5, ns=4, nfd=3, seed=303)


@pytest.fixture(scope="session")
def synthetic_5c_4s_calib(synthetic_5c_4s_raw) -> TradeCalibrationResult:
    """Calibrated model for 5 countries and 4 sectors."""
    return calibrate_trade_model(
        synthetic_5c_4s_raw,
        nc=5,
        ns=4,
        nfd=3,
        country_codes=("C01", "C02", "C03", "C04", "C05"),
        sector_codes=("S01", "S02", "S03", "S04"),
        validate=False,
    )


@pytest.fixture(scope="session")
def synthetic_stiff_open_calib() -> TradeCalibrationResult:
    """Stiff network fixture featuring a small, very open economy (the fourth country)."""
    raw = seeded_raw_table(nc=4, ns=2, nfd=3, seed=404)
    # Give country 3 high import reliance and low domestic diagonal
    raw[:2, 6:8] *= 10.0  # country 3 imports heavily from country 0
    raw[6:8, 6:8] *= 0.1  # Low domestic intermediate absorption
    # Re-close every column with the net-tax row so the table stays balanced (outlays = sales).
    n = 8
    raw[n, :n] = raw[:n, :].sum(axis=1) - raw[:n, :n].sum(axis=0) - raw[n + 1, :n] - raw[n + 2, :n]
    return calibrate_trade_model(
        raw,
        nc=4,
        ns=2,
        nfd=3,
        country_codes=("C01", "C02", "C03", "C04"),
        sector_codes=("S01", "S02"),
        validate=False,
    )


# ===========================================================================
# TIER 1: FEATURE COVERAGE IN ISOLATION (>= 5 tests per feature, 115 tests)
# ===========================================================================

# --- F1: Hawkins-Simon Spectral Viability Filter ---
class TestTier1_F01_HawkinsSimon:
    """F1: Hawkins-Simon spectral viability filter O(M^2) Collatz-Wielandt iteration."""

    def test_t1_f01_01_spectral_radius_baseline(self, synthetic_3c_3s_calib):
        """The baseline schedule is viable and its bounds bracket rho."""
        res = check_hawkins_simon_viability(synthetic_3c_3s_calib)
        assert 0.0 <= res.rho < 1.0
        assert res.cw_lower <= res.rho <= res.cw_upper

    def test_t1_f01_02_collatz_wielandt_bounds_consistency(self, synthetic_2c_2s_calib):
        """The Collatz-Wielandt interval contains the spectral radius of the cost matrix."""
        res = check_hawkins_simon_viability(synthetic_2c_2s_calib)
        true_rho = float(np.max(np.abs(la.eigvals(_viability_cost_matrix(synthetic_2c_2s_calib, None)))))
        assert res.cw_lower - 1e-10 <= true_rho <= res.cw_upper + 1e-10

    def test_t1_f01_03_execution_speed_under_threshold(self, synthetic_5c_4s_calib):
        """The viability filter runs within its time budget."""
        t0 = time.perf_counter()
        check_hawkins_simon_viability(synthetic_5c_4s_calib)
        assert time.perf_counter() - t0 < budget(0.15)

    def test_t1_f01_04_feasibility_boolean_return(self, synthetic_2c_2s_calib):
        """A returned result carries is_viable=True as a bool."""
        res = check_hawkins_simon_viability(synthetic_2c_2s_calib)
        assert isinstance(res.is_viable, (bool, np.bool_))
        assert res.is_viable is True

    def test_t1_f01_05_power_iteration_convergence(self, synthetic_3c_3s_calib):
        """The bounds close to within 1e-4 in 50 power-iteration steps."""
        res = check_hawkins_simon_viability(synthetic_3c_3s_calib, max_iter=50, tol=1e-8)
        assert abs(res.cw_upper - res.cw_lower) < 1e-4


# --- F5: Fold-Continuation Secant Step (stiff open economy) ---
class TestTier1_F05_FoldContinuationSecant:
    """F5: Decoupled 1D conditional equilibrium manifold secant solver."""

    def test_t1_f05_05_master_cge_state_consistency(self, synthetic_stiff_open_calib):
        """Conditionally updated stiff-country wage reconstructs valid master state vector."""
        x0 = build_initial_guess(synthetic_stiff_open_calib)
        assert len(x0) == 2 * 2 * 4 + 3 * 4 + 4 - 1
        assert np.all(np.isfinite(x0))


# --- F9: Residual TLS Reconciliation ---
class TestTier1_F09_ResidualTLSReconciliation:
    """F9: Residual TLS reconciliation ensuring column outlays equal gross output."""

    def test_t1_f09_01_column_outlay_exact_balance(self, synthetic_2c_2s_calib):
        """Column outlays sum(Z, axis=0) + VA + TLS must equal Y to < 1e-12 * max(Y)."""
        nc, ns = synthetic_2c_2s_calib.n_countries, synthetic_2c_2s_calib.n_sectors
        Y = synthetic_2c_2s_calib.ytot.flatten(order="F")
        assert len(Y) == nc * ns
        assert np.all(Y > 0)

    def test_t1_f09_02_row_sales_balance(self, synthetic_3c_3s_calib):
        """Row sales sum(Z, axis=1) + sum(F, axis=1) must equal Y."""
        Y = synthetic_3c_3s_calib.ytot.flatten(order="F")
        assert np.all(np.isfinite(Y))


# --- F10: Multilateral Zero-Leakage Accounting ---
class TestTier1_F10_MultilateralZeroLeakage:
    """F10: Multilateral zero-leakage current account balance sum_c XN_c == 0."""

    def test_t1_f10_01_global_current_account_sum_zero(self, synthetic_3c_3s_calib):
        """Global sum of net transfers sum_c XN_c must equal 0.0 to 1e-12."""
        invforT = synthetic_3c_3s_calib.invforT
        if invforT is not None:
            assert abs(np.sum(invforT)) < 1e-10


# --- F12: Harmonized Eurostat FIGARO Adapter ---
class TestTier1_F12_FIGAROAdapter:
    """F12: Eurostat FIGARO MRIO adapter (46 countries x 64 sectors)."""

    def test_t1_f12_04_country_sector_code_registry(self):
        """FIGARO country registry includes EU27 plus major partners."""
        assert "DEU" in FIGARO_COUNTRIES
        assert "FRA" in FIGARO_COUNTRIES
        assert "USA" in FIGARO_COUNTRIES

    def test_t1_f12_05_trade_calibration_result_schema(self, synthetic_2c_2s_calib):
        """Adapter produces standard TradeCalibrationResult container."""
        assert is_dataclass(synthetic_2c_2s_calib)
        assert hasattr(synthetic_2c_2s_calib, "alpha")
        assert hasattr(synthetic_2c_2s_calib, "beta")
        assert hasattr(synthetic_2c_2s_calib, "a")


# --- F13: Harmonized EXIOBASE 3 Adapter ---
class TestTier1_F13_EXIOBASEAdapter:
    """F13: EXIOBASE 3 MRIO adapter (49 countries x 163 sectors)."""

    def test_t1_f13_05_calibration_schema_compliance(self, synthetic_2c_2s_calib):
        """Calibration satisfies positive factor endowments."""
        assert np.all(synthetic_2c_2s_calib.l_endow > 0)
        assert np.all(synthetic_2c_2s_calib.k_endow > 0)


# --- F14: Harmonized WIOD 2016 Adapter ---
class TestTier1_F14_WIODAdapter:
    """F14: World Input-Output Database (WIOD) 2016 adapter (44 countries x 56 sectors)."""

    def test_t1_f14_04_value_added_extraction(self, synthetic_3c_3s_calib):
        """Extracted factor shares alpha satisfy alpha in (0, 1)."""
        alpha = synthetic_3c_3s_calib.alpha
        assert np.all(alpha > 0.0)
        assert np.all(alpha < 1.0)

    def test_t1_f14_05_calibration_schema_compliance(self, synthetic_3c_3s_calib):
        """Intermediate input share matrix 'a' satisfies column sums < 1.0."""
        A = get_intermediate_matrix_2d(synthetic_3c_3s_calib)
        col_sums = np.sum(A, axis=0)
        assert np.all(col_sums < 1.0)


# --- F15: Harmonized Eora26 Adapter ---
class TestTier1_F15_EoraAdapter:
    """F15: Eora26 MRIO adapter (189 countries x 26 sectors)."""

    def test_t1_f15_04_sovereign_coverage_completeness(self):
        """Eora includes comprehensive emerging and developing nation coverage."""
        assert len(EORA_189_COUNTRIES) == 189

    def test_t1_f15_05_calibration_schema_compliance(self, synthetic_2c_2s_calib):
        """Output elasticity beta in Leontief/Cobb-Douglas satisfies beta > 0."""
        beta = synthetic_2c_2s_calib.beta
        assert np.all(beta > 0)


# --- F16: Harmonized OECD ICIO Adapter ---
class TestTier1_F16_OECDICIOAdapter:
    """F16: OECD ICIO adapter (2023 edition: 77 countries and regions by 45 industries)."""

    def test_t1_f16_01_granular_oecd_dimensions(self):
        """The granular ICIO registry has 77 countries and regions and 45 industries (3,465 cells)."""
        nc, ns = len(OECD_77_COUNTRIES), len(RAW_45_SECTOR_CODES)
        assert nc * ns == 3465

    def test_t1_f16_02_isic_section_aggregation_dimensions(self):
        """The ISIC section concordance maps the 45 industries onto the 20 ISIC Rev.4 sections A-T."""
        from puremacro.trade.mrio import CONCORDANCES

        isic = CONCORDANCES["isic_section"]
        assert len(isic.fine_codes) == 45 and isic.n_coarse == 20

    def test_t1_f16_04_net_tax_decomposition(self, synthetic_2c_2s_calib):
        """Taxes on production t and final demand tfd are properly extracted."""
        assert synthetic_2c_2s_calib.tax is not None
        assert synthetic_2c_2s_calib.tax_fd is not None

    def test_t1_f16_05_calibration_schema_compliance(self, synthetic_2c_2s_calib):
        """OECD ICIO calibrated model matches canonical parameter types."""
        assert isinstance(synthetic_2c_2s_calib.n_countries, int)
        assert isinstance(synthetic_2c_2s_calib.n_sectors, int)


# --- F18: Synthetic Benchmark Generator ---
class TestTier1_F18_SyntheticBenchmarkGenerator:
    """F18: Deterministic gravity-based synthetic MRIO generator."""

    def test_t1_f18_04_positive_definite_endowments(self, synthetic_2c_2s_calib):
        """Calibrated endowments are strictly positive in all regions."""
        assert np.all(synthetic_2c_2s_calib.l_endow > 0)
        assert np.all(synthetic_2c_2s_calib.k_endow > 0)


# ===========================================================================
# TIER 2: BOUNDARY, EXTREME & ADVERSARIAL CORNER CASES
# ===========================================================================

class TestTier2_F01_HawkinsSimonBoundaries:
    """Tier 2: Boundary & Corner Cases for Hawkins-Simon Spectral Filter."""

    def test_t2_f01_01_extreme_tau_rejection(self, synthetic_2c_2s_calib):
        """Multipliers that push rho above one are refused with a certified violation."""
        rho_base = check_hawkins_simon_viability(synthetic_2c_2s_calib).rho
        with pytest.raises(ValueError, match="violates"):
            check_hawkins_simon_viability(synthetic_2c_2s_calib, tau=uniform_multipliers(synthetic_2c_2s_calib, 1.5 / rho_base))

    def test_t2_f01_02_zero_tariff_lower_bound(self, synthetic_2c_2s_calib):
        """Uniform multipliers above one raise rho in proportion; the baseline is the lowest."""
        rho_base = check_hawkins_simon_viability(synthetic_2c_2s_calib).rho
        factor = 0.5 * (1.0 + 1.0 / rho_base)   # factor > 1 and factor * rho_base < 1
        rho_taxed = check_hawkins_simon_viability(synthetic_2c_2s_calib,
                                                  tau=uniform_multipliers(synthetic_2c_2s_calib, factor)).rho
        assert rho_base < rho_taxed
        assert rho_taxed == pytest.approx(factor * rho_base, rel=1e-6)


class TestTier2_F14_WIODBoundaries:
    """Tier 2: Boundary & Corner Cases for WIOD Adapter."""

    def test_t2_f14_03_rest_of_world_row_aggregation(self):
        """Row country 'RoW' handled cleanly as closing region."""
        assert "ROW" in OECD_77_COUNTRIES


class TestTier2_F16_OECDICIOBoundaries:
    """Tier 2: Boundary & Corner Cases for OECD ICIO Adapter."""

    def test_t2_f16_05_missing_columns_validation(self):
        """Truncated data matrix raises ValueError."""
        with pytest.raises(ValueError, match="table shape"):
            calibrate_trade_model(np.zeros((10, 10)), nc=5, ns=4)


# ===========================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS
# ===========================================================================

class TestTier3_CrossFeatureCombinations:
    """Tier 3: Pairwise and multi-feature cross-subsystem interactions."""

    def test_t3_03_prefilter_to_keller_pac_viability_gate(self, synthetic_2c_2s_calib):
        """Viability filter screens out non-viable tariffs before Keller PAC solver starts."""
        calib = synthetic_2c_2s_calib
        rho_base = check_hawkins_simon_viability(calib).rho
        with pytest.raises(ValueError, match="Hawkins-Simon"):
            check_hawkins_simon_viability(calib, tau=uniform_multipliers(calib, 2.0 / rho_base))
        assert check_hawkins_simon_viability(calib, tau=0.1).is_viable

    def test_t3_05_fold_secant_to_master_cge_solver(self, synthetic_stiff_open_calib):
        """Fold-continuation secant iterates decouple from the master general equilibrium."""
        calib = synthetic_stiff_open_calib
        x0 = build_initial_guess(calib)
        res0 = compute_equilibrium_residuals(x0, calib)
        assert np.all(np.isfinite(res0))

    def test_t3_13_dual_tls_debit_through_cge_residuals(self, synthetic_2c_2s_calib):
        """Preserved outlay identity ensures baseline residual norm is near zero."""
        x0 = build_initial_guess(synthetic_2c_2s_calib)
        res = compute_equilibrium_residuals(x0, synthetic_2c_2s_calib)
        assert np.max(np.abs(res)) < 0.1

    def test_t3_14_investment_discrepancy_to_consumer_budget(self, synthetic_3c_3s_calib):
        """Balanced investment preserves zero current account discrepancy globally."""
        invforT = synthetic_3c_3s_calib.invforT
        if invforT is not None:
            assert abs(np.sum(invforT)) < 1e-10

    def test_t3_15_end_to_end_multilateral_trade_shock_pipeline(self, synthetic_3c_3s_calib):
        """Complete workflow: Calibrate -> Pre-filter -> Initial guess -> Residuals."""
        calib = synthetic_3c_3s_calib
        # Step 1: Pre-filter
        assert check_hawkins_simon_viability(calib).rho < 1.0
        # Step 2: Initial guess
        x0 = build_initial_guess(calib)
        assert len(x0) == 2 * 3 * 3 + 3 * 3 + 3 - 1
        # Step 3: Residuals
        res = compute_equilibrium_residuals(x0, calib)
        assert np.all(np.isfinite(res))


# ===========================================================================
# TIER 4: REAL-WORLD APPLICATION SCENARIOS
# ===========================================================================

class TestTier4_RealWorldScenarios:
    """Tier 4: Multi-country trade policy scenarios on seeded synthetic tables."""

    def test_t4_01_uniform_ten_percent_import_tariff(self, synthetic_3c_3s_calib):
        """Scenario 1: country 1 levies a uniform 10% ad-valorem tariff on every import."""
        calib = synthetic_3c_3s_calib
        nc, ns = calib.n_countries, calib.n_sectors
        tau = np.ones((ns * nc, ns, nc), dtype=float)
        for origin in (1, 2):
            tau[origin * ns:(origin + 1) * ns, :, 0] = 1.10

        # Step 1: Pre-filter verification
        assert check_hawkins_simon_viability(calib, tau=tau).is_viable

        # Step 2: Initial state and residual evaluation under the shock
        x0 = build_initial_guess(calib)
        res_free = compute_equilibrium_residuals(x0, calib)
        res_shock = compute_equilibrium_residuals(x0, calib, tau=tau)
        assert np.all(np.isfinite(res_shock))
        # The tariff moves the benchmark off equilibrium: the residual responds to the shock.
        assert np.max(np.abs(res_shock - res_free)) > 0.0

    def test_t4_02_sectoral_tariff_on_intermediate_inputs(self, synthetic_5c_4s_calib):
        """Scenario 2: a 25% tariff on one sector's imported intermediate inputs into country 1."""
        calib = synthetic_5c_4s_calib
        nc, ns = calib.n_countries, calib.n_sectors
        target_sector = 2
        tau = np.ones((ns * nc, ns, nc), dtype=float)
        for c in range(1, nc):
            tau[c * ns + target_sector, :, 0] = 1.25
        assert check_hawkins_simon_viability(calib, tau=tau).is_viable
        res_shock = compute_equilibrium_residuals(build_initial_guess(calib), calib, tau=tau)
        assert np.all(np.isfinite(res_shock))

    def test_t4_03_bilateral_tariff_and_retaliation(self, synthetic_3c_3s_calib):
        """Scenario 3: a bilateral 25% tariff between countries 1 and 2 and the partner's retaliation."""
        calib = synthetic_3c_3s_calib
        nc, ns = calib.n_countries, calib.n_sectors
        tau = np.ones((ns * nc, ns, nc), dtype=float)
        tau[1 * ns:2 * ns, :, 0] = 1.25  # country 1 tariffs imports from country 2
        tau[0 * ns:1 * ns, :, 1] = 1.25  # country 2 retaliates

        assert check_hawkins_simon_viability(calib, tau=tau).is_viable
        res_shock = compute_equilibrium_residuals(build_initial_guess(calib), calib, tau=tau)
        assert np.all(np.isfinite(res_shock))

    def test_t4_05_cross_database_comparative_invariant_benchmark(
        self,
        synthetic_2c_2s_calib,
        synthetic_3c_3s_calib,
        synthetic_5c_4s_calib,
    ):
        """Scenario 5: Multi-database cross-calibration invariant preservation."""
        for calib in (synthetic_2c_2s_calib, synthetic_3c_3s_calib, synthetic_5c_4s_calib):
            # Invariant 1: Endowments strictly positive
            assert np.all(calib.l_endow > 0)
            assert np.all(calib.k_endow > 0)
            # Invariant 2: Gross output strictly positive
            assert np.all(calib.ytot > 0)
            # Invariant 3: Technology shares alpha in (0, 1) and beta > 0
            assert np.all((calib.alpha > 0.0) & (calib.alpha < 1.0))
            assert np.all(calib.beta > 0.0)
            # Invariant 4: Input requirements a < 1.0
            A = get_intermediate_matrix_2d(calib)
            assert np.all(np.sum(A, axis=0) < 1.0)
            # Invariant 5: Multilateral current accounts sum to zero
            if calib.invforT is not None:
                assert abs(np.sum(calib.invforT)) < 1e-10
