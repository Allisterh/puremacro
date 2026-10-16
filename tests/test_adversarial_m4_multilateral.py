"""Adversarial stress tests: multilateral Geary-Khamis axioms and tariff tensors (M4).

1. Additive consistency: world expenditure invariant
     sum_k Nominal GDP_k = sum_m pi_m Q_m = sum_k Real GDP_k
   on random synthetic economies, for every normalization.
2. Solver equivalence: the iterative fixed point and the direct linear solve
   give the same pi and PPP vectors to within floating-point precision (< 1e-12).
3. Domestic tariff invariance: the domestic diagonal of tau and tau_fd is 1.0
   for user-defined scenarios and strategic tariff profiles on a
   70-country, 12-sector container; a domestic tariff is refused.
4. Boundary and inactive categories: zero consumption, zero investment, sparse
   final demand, an all-zero category and negative capital formation.

Every input is synthetic; nothing reads a bundled table.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from puremacro.trade._results import TradeCalibrationResult
from puremacro.trade.data import synthetic_codes
from puremacro.trade.geary_khamis import restore_capital_formation, solve_multilateral_ppp
from puremacro.trade.optimal_tariffs import build_strategic_tariffs
from puremacro.trade.scenarios import TariffScenario, build_tariff_matrices


@pytest.fixture(scope="module")
def synthetic_calib_70c() -> TradeCalibrationResult:
    """Minimal calibration container for 70 countries and 12 sectors with neutral codes."""
    nc, ns, nfd = 70, 12, 3
    countries, sectors = synthetic_codes(nc, ns)
    return TradeCalibrationResult(
        a=np.zeros((ns, nc, ns, nc)),
        afd=np.zeros((ns, nc, nfd, nc)),
        alpha=np.full((1, ns, nc), 1.0 / 3.0),
        beta=np.ones((1, ns, nc)),
        k_endow=np.ones((1, 1, nc)),
        l_endow=np.ones((1, 1, nc)),
        invforT=np.zeros((1, 1, nc)),
        tax=np.zeros((1, ns, nc)),
        ytot=np.ones((1, ns, nc)),
        country_codes=tuple(countries),
        sector_codes=tuple(sectors),
    )


def _assert_domestic_diagonal_is_one(tau, tau_fd, nc, label):
    for k in range(nc):
        assert np.all(tau[:, k, :, k] == 1.0), f"{label}: intermediate diagonal corrupted for country {k}"
        assert np.all(tau_fd[:, k, :, k] == 1.0), f"{label}: final-demand diagonal corrupted for country {k}"


# ---------------------------------------------------------------------------
# 1. Additive consistency
# ---------------------------------------------------------------------------

class TestAdditiveConsistency:
    def test_additive_consistency_random_synthetic_stress(self):
        """Additive consistency across 50 random synthetic economies."""
        rng = np.random.default_rng(42)
        for trial in range(50):
            M = rng.integers(2, 12)
            K = rng.integers(2, 60)
            p = rng.uniform(0.01, 100.0, (M, K))
            q = rng.uniform(0.1, 5000.0, (M, K))
            q_world = np.sum(q, axis=1)
            world_nom = float(np.sum(p * q))
            for method in ("linear", "iterative"):
                pi, _, conv, _ = solve_multilateral_ppp(
                    p, q, method=method, tol=1e-14, max_iter=3000, normalize="none"
                )
                assert conv, f"Method {method} failed to converge on trial {trial}"
                world_real = float(np.sum(q * pi[:, np.newaxis]))
                pi_q = float(np.dot(pi, q_world))
                assert abs(world_nom - pi_q) / world_nom < 1e-12
                assert abs(world_real - pi_q) / world_real < 1e-12
                assert abs(world_nom - world_real) / world_nom < 1e-12

    @pytest.mark.parametrize("normalize, numeraire_idx", [("none", None), ("pi1", None), ("numeraire", 3)])
    def test_algebraic_identity_sum_real_equals_pi_q_under_any_normalization(self, normalize, numeraire_idx):
        """sum_k Real GDP_k == sum_m pi_m Q_m for every normalization."""
        rng = np.random.default_rng(999)
        p = rng.uniform(0.5, 2.0, (4, 10))
        q = rng.uniform(10.0, 100.0, (4, 10))
        pi, ppp, _, _ = solve_multilateral_ppp(p, q, method="linear", normalize=normalize,
                                               numeraire_idx=numeraire_idx)
        sum_real = float(np.sum(q * pi[:, np.newaxis]))
        assert math.isclose(sum_real, float(np.dot(pi, q.sum(axis=1))), rel_tol=1e-14)
        if normalize == "numeraire":
            assert ppp[numeraire_idx] == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 2. Solver equivalence
# ---------------------------------------------------------------------------

class TestSolverEquivalence:
    def test_solver_equivalence_random_synthetic_stress(self):
        """< 1e-12 solver equivalence across 50 random synthetic matrices."""
        rng = np.random.default_rng(2026)
        max_pi, max_ppp = 0.0, 0.0
        for trial in range(50):
            M = rng.integers(2, 15)
            K = rng.integers(2, 80)
            p = rng.uniform(0.1, 10.0, (M, K))
            q = rng.uniform(1.0, 500.0, (M, K))
            pi_lin, ppp_lin, _, _ = solve_multilateral_ppp(p, q, method="linear")
            pi_it, ppp_it, conv, _ = solve_multilateral_ppp(p, q, method="iterative", tol=1e-14, max_iter=4000)
            assert conv, f"Iterative failed to converge on synthetic trial {trial}"
            max_pi = max(max_pi, float(np.max(np.abs(pi_it - pi_lin))))
            max_ppp = max(max_ppp, float(np.max(np.abs(ppp_it - ppp_lin))))
        assert max_pi < 1e-12
        assert max_ppp < 1e-12

    def test_operator_spectral_radius_and_nullspace_structure(self):
        """The Geary-Khamis transition matrix has Perron root 1 and a subdominant second root."""
        rng = np.random.default_rng(777)
        p = rng.uniform(0.5, 2.5, (5, 20))
        q = rng.uniform(10.0, 100.0, (5, 20))
        term1 = (p * q) / np.sum(q, axis=1)[:, np.newaxis]
        term2 = q / np.sum(p * q, axis=0)[np.newaxis, :]
        eigs = sorted(np.linalg.eigvals(term1 @ term2.T), key=abs, reverse=True)
        assert math.isclose(abs(eigs[0]), 1.0, rel_tol=1e-12)
        assert abs(eigs[1]) < 1.0 - 1e-4


# ---------------------------------------------------------------------------
# 3. Domestic tariff invariance
# ---------------------------------------------------------------------------

class TestDomesticTariffInvariance:
    def test_user_scenarios_keep_domestic_diagonal(self, synthetic_calib_70c):
        """Random user-defined scenarios never tariff domestic flows."""
        calib = synthetic_calib_70c
        countries, sectors = calib.country_codes, calib.sector_codes
        rng = np.random.default_rng(11)
        for trial in range(5):
            importer = countries[int(rng.integers(len(countries)))]
            exporters = [c for c in rng.choice(countries, size=8, replace=False) if c != importer]
            tariffs = {(e, "*"): float(rng.uniform(0, 1.5)) for e in exporters[:4]}
            tariffs.update({(e, sectors[int(rng.integers(len(sectors)))]): float(rng.uniform(0, 3))
                            for e in exporters[4:]})
            scen = TariffScenario(importer, tariffs, name=f"random_{trial}")
            tau, tau_fd = build_tariff_matrices(calib, scen)
            _assert_domestic_diagonal_is_one(tau, tau_fd, calib.n_countries, scen.name)
            assert np.any(tau != 1.0)

    def test_domestic_tariff_is_refused(self, synthetic_calib_70c):
        """A scenario that lists a tariff on the importer's own goods raises."""
        calib = synthetic_calib_70c
        own = calib.country_codes[0]
        hostile = TariffScenario(own, {(own, "*"): 0.75, (calib.country_codes[1], "*"): 0.25})
        with pytest.raises(ValueError, match="Domestic flows are never tariffed"):
            build_tariff_matrices(calib, hostile)
        rates = np.zeros((calib.n_countries, calib.n_sectors))
        rates[0, 3] = 0.5
        with pytest.raises(ValueError, match="Domestic flows are never tariffed"):
            build_tariff_matrices(calib, TariffScenario(own, rates))

    def test_strategic_profiles_keep_domestic_diagonal(self, synthetic_calib_70c):
        """Universal and bilateral strategic tariff profiles keep the diagonal at one."""
        calib = synthetic_calib_70c
        c = calib.country_codes
        for mode in ("universal", "final_only", "intermediate_only"):
            profile = {c[0]: 0.45, c[5]: {c[0]: 0.99, c[5]: 0.6, c[9]: 0.3}, c[9]: 1.5}
            tau, tau_fd = build_strategic_tariffs(calib, profile, policy_mode=mode)
            _assert_domestic_diagonal_is_one(tau, tau_fd, calib.n_countries, mode)

    def test_invalid_negative_and_nonfinite_tariffs_rejected(self):
        """Negative, NaN and infinite rates are refused."""
        for bad in (-0.10, float("nan"), float("inf")):
            with pytest.raises(ValueError, match="finite non-negative"):
                TariffScenario("C01", {("C02", "*"): bad})
        with pytest.raises(ValueError, match="finite and non-negative"):
            TariffScenario("C01", np.array([[0.0, -0.25], [0.0, 0.0]]))


# ---------------------------------------------------------------------------
# 4. Boundary and inactive categories
# ---------------------------------------------------------------------------

def _assert_equivalent(p, q):
    pi_lin, ppp_lin, conv_lin, _ = solve_multilateral_ppp(p, q, method="linear")
    pi_it, ppp_it, conv_it, _ = solve_multilateral_ppp(p, q, method="iterative", tol=1e-14)
    assert conv_lin and conv_it
    assert np.max(np.abs(pi_it - pi_lin)) < 1e-12
    assert np.max(np.abs(ppp_it - ppp_lin)) < 1e-12
    return pi_lin, ppp_lin


_P3 = np.array([[1.0, 1.1, 0.95], [1.2, 0.9, 1.05], [0.8, 1.0, 1.15]])


class TestBoundaryAndInactiveHandling:
    def test_zero_investment_single_country(self):
        q = np.array([[100.0, 120.0, 150.0], [0.0, 60.0, 80.0], [20.0, 25.0, 30.0]])
        pi, _ = _assert_equivalent(_P3, q)
        assert math.isclose(float(np.sum(_P3 * q)), float(np.sum(q * pi[:, np.newaxis])), rel_tol=1e-12)

    def test_zero_consumption_single_country(self):
        q = np.array([[100.0, 0.0, 150.0], [50.0, 60.0, 80.0], [20.0, 25.0, 30.0]])
        _assert_equivalent(_P3, q)

    def test_sparse_final_demand_multiple_zeros(self):
        p = np.array([[1.0, 1.1, 0.9, 1.05], [1.2, 0.9, 1.1, 0.95], [0.8, 1.0, 1.2, 1.00]])
        q = np.array([[100.0, 0.0, 150.0, 80.0], [50.0, 60.0, 0.0, 70.0], [0.0, 25.0, 30.0, 0.0]])
        _assert_equivalent(p, q)

    def test_entire_category_zero_world_consumption_raises(self):
        p = np.array([[1.0, 1.1], [1.2, 0.9], [0.8, 1.0]])
        q = np.array([[100.0, 120.0], [0.0, 0.0], [20.0, 30.0]])
        for method in ("linear", "iterative"):
            with pytest.raises(ValueError, match="strictly positive for all categories"):
                solve_multilateral_ppp(p, q, method=method)

    def test_negative_capital_formation_after_restoring_foreign_investment(self):
        """A large net foreign borrower gets negative restored investment; PPPs stay positive."""
        rng = np.random.default_rng(5)
        nc = 6
        p = rng.uniform(0.8, 1.2, (3, nc))
        c = np.vstack([rng.uniform(50, 100, nc), rng.uniform(10, 20, nc), rng.uniform(5, 10, nc)])
        xn = np.array([-40.0, 8.0, 8.0, 8.0, 8.0])      # last country's balance is -sum(xn)
        q = restore_capital_formation(c, xn)
        assert q[1, 0] < 0.0
        _, ppp = _assert_equivalent(p, q)
        assert np.all(ppp > 0.0)

    def test_near_zero_quantities_and_connectedness_limit(self):
        p = np.ones((2, 2))
        _assert_equivalent(p, np.array([[1000.0, 0.1], [0.1, 1000.0]]))
        q_ext = np.array([[1e6, 1e-15], [1e-15, 1e6]])
        pi, ppp, conv, _ = solve_multilateral_ppp(p, q_ext, method="iterative")
        assert conv
        assert np.allclose(pi, [1.0, 1.0], atol=1e-10)
        assert np.allclose(ppp, [1.0, 1.0], atol=1e-10)
