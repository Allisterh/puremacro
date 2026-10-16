"""Adversarial tests of the trade policy analytics and the package namespace.

1. policy_analytics: decompose_welfare_effects with missing flows,
   compute_effective_rate_of_protection with each tariff input form,
   compute_supply_chain_vulnerability and calculate_tariff_revenue_incidence
   across closures; required target countries and the removed named scenarios.
2. __init__: every name of __all__ resolves, the policy analytics names are
   re-exported, no private name leaks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import puremacro.trade as pt
from puremacro.trade._results import TradeCalibrationResult, TradeEquilibriumResult
from puremacro.trade.policy_analytics import (
    EffectiveRateOfProtectionResult,
    SupplyChainVulnerabilityResult,
    TariffRevenueIncidenceResult,
    WelfareDecompositionResult,
    calculate_tariff_revenue_incidence,
    compute_effective_rate_of_protection,
    compute_supply_chain_vulnerability,
    decompose_welfare_effects,
)
from puremacro.trade.scenarios import TariffScenario


# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def mock_calib_2c_2s() -> TradeCalibrationResult:
    """Fast, deterministic 2-country x 2-sector calibrated trade model."""
    nc, ns, nfd = 2, 2, 3
    ytot = np.array([[[1000.0, 800.0], [500.0, 400.0]]], dtype=np.float64)
    a_3d = np.zeros((ns * nc, ns, nc), dtype=np.float64)
    a_3d[0, 0, 0] = 0.15  # domestic AGR into AGR in HOM
    a_3d[1, 1, 0] = 0.10  # domestic MAN into MAN in HOM
    a_3d[2, 0, 0] = 0.20  # foreign AGR into AGR in HOM
    a_3d[3, 1, 0] = 0.20  # foreign MAN into MAN in HOM
    a_3d[2, 0, 1] = 0.12  # domestic AGR into AGR in ROW
    a_3d[3, 1, 1] = 0.08  # domestic MAN into MAN in ROW
    a_3d[0, 0, 1] = 0.15  # foreign AGR into AGR in ROW
    a_3d[1, 1, 1] = 0.15  # foreign MAN into MAN in ROW

    afd_3d = np.full((ns * nc, nfd, nc), 0.05, dtype=np.float64)
    alpha = np.full((1, ns, nc), 0.33, dtype=np.float64)
    beta = np.ones((1, ns, nc), dtype=np.float64)
    k_endow = np.array([[500.0, 400.0]], dtype=np.float64)
    l_endow = np.array([[500.0, 400.0]], dtype=np.float64)
    invforT = np.array([[0.0, 0.0]], dtype=np.float64)
    tax = np.zeros((1, ns, nc), dtype=np.float64)

    return TradeCalibrationResult(
        a=a_3d,
        afd=afd_3d,
        alpha=alpha,
        beta=beta,
        k_endow=k_endow,
        l_endow=l_endow,
        invforT=invforT,
        tax=tax,
        ytot=ytot,
        n_countries=nc,
        n_sectors=ns,
        n_final_demand=nfd,
        country_codes=("HOM", "ROW"),
        sector_codes=("AGR", "MAN"),
    )


@pytest.fixture(scope="module")
def mock_calib_1c_2s() -> TradeCalibrationResult:
    """Minimal single-country 2-sector autarkic trade model."""
    nc, ns, nfd = 1, 2, 3
    ytot = np.array([[[1000.0], [500.0]]], dtype=np.float64)
    a_3d = np.zeros((ns * nc, ns, nc), dtype=np.float64)
    a_3d[0, 0, 0] = 0.20
    a_3d[1, 1, 0] = 0.15
    afd_3d = np.full((ns * nc, nfd, nc), 0.10, dtype=np.float64)
    alpha = np.full((1, ns, nc), 0.33, dtype=np.float64)
    beta = np.ones((1, ns, nc), dtype=np.float64)
    k_endow = np.array([[500.0]], dtype=np.float64)
    l_endow = np.array([[500.0]], dtype=np.float64)
    invforT = np.array([[0.0]], dtype=np.float64)
    tax = np.zeros((1, ns, nc), dtype=np.float64)

    return TradeCalibrationResult(
        a=a_3d,
        afd=afd_3d,
        alpha=alpha,
        beta=beta,
        k_endow=k_endow,
        l_endow=l_endow,
        invforT=invforT,
        tax=tax,
        ytot=ytot,
        n_countries=nc,
        n_sectors=ns,
        n_final_demand=nfd,
        country_codes=("HOM",),
        sector_codes=("AGR", "MAN"),
    )


@pytest.fixture(scope="module")
def mock_equilibrium_pair(mock_calib_2c_2s) -> tuple[TradeEquilibriumResult, TradeEquilibriumResult]:
    """Pair of mock baseline and counterfactual equilibrium results."""
    nc = mock_calib_2c_2s.n_countries
    ns = mock_calib_2c_2s.n_sectors
    nfd = mock_calib_2c_2s.n_final_demand
    ytot = mock_calib_2c_2s.ytot

    # Base equilibrium
    base = TradeEquilibriumResult(
        x_sol=np.zeros(2 * ns * nc + 4 * nc - 1),
        p_sol=np.ones((1, ns, nc)),
        y_sol=ytot.copy(),
        r_sol=np.ones((1, 1, nc)),
        w_sol=np.ones((1, 1, nc)),
        T_sol=np.zeros((1, 1, nc)),
        XN_sol=np.zeros(1),
        tariffs=np.array([0.0, 0.0]),
        cpi=np.array([1.0, 1.0]),
        terms_of_trade=np.array([1.0, 1.0]),
        exports=np.array([200.0, 150.0]),
        imports=np.array([200.0, 150.0]),
        intermediate_flows=np.ones((ns, nc, ns, nc)) * 10.0,
        final_demand_flows=np.ones((ns, nc, nfd, nc)) * 5.0,
        country_codes=("HOM", "ROW"),
        sector_codes=("AGR", "MAN"),
    )

    # Counterfactual equilibrium
    cf = TradeEquilibriumResult(
        x_sol=np.zeros(2 * ns * nc + 4 * nc - 1),
        p_sol=np.ones((1, ns, nc)) * 1.05,
        y_sol=ytot * 0.98,
        r_sol=np.ones((1, 1, nc)) * 0.99,
        w_sol=np.ones((1, 1, nc)) * 0.99,
        T_sol=np.array([[[15.0, 8.0]]]),
        XN_sol=np.zeros(1),
        tariffs=np.array([25.0, 10.0]),
        cpi=np.array([1.03, 1.01]),
        terms_of_trade=np.array([1.02, 0.98]),
        exports=np.array([190.0, 145.0]),
        imports=np.array([175.0, 140.0]),
        intermediate_flows=np.ones((ns, nc, ns, nc)) * 9.5,
        final_demand_flows=np.ones((ns, nc, nfd, nc)) * 4.8,
        country_codes=("HOM", "ROW"),
        sector_codes=("AGR", "MAN"),
    )

    return base, cf


# =============================================================================
# 1. Welfare decomposition, ERP, supply chain and incidence
# =============================================================================

class TestWelfareDecompositionAndERPEdgeCases:
    """Stress-test decompose_welfare_effects, compute_effective_rate_of_protection, etc."""

    def test_welfare_decomposition_budget_balance(self, mock_calib_2c_2s, mock_equilibrium_pair):
        """Verify decompose_welfare_effects satisfies exact Walrasian budget balance residual < 1e-4."""
        base, cf = mock_equilibrium_pair
        res = decompose_welfare_effects(mock_calib_2c_2s, cf, base_result=base, target_country="HOM")
        assert isinstance(res, WelfareDecompositionResult)
        assert res.exact_identity_satisfied is True
        assert res.budget_balance_residual < 1e-4
        assert abs(res.total_welfare_change - (res.tot_effect + res.cascading_distortion + res.allocative_efficiency)) < 1e-4

        df = res.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert "Terms_of_Trade_Effect" in df.index

    def test_decompose_welfare_effects_none_final_demand_flows_adversarial_check(self, mock_calib_2c_2s, mock_equilibrium_pair):
        """Adversarial check: decompose_welfare_effects should not crash when final_demand_flows is None.

        When intermediate_flows is present but final_demand_flows is None on TradeEquilibriumResult,
        the flows code once attempted to subscript NoneType.
        """
        base, cf = mock_equilibrium_pair
        cf_none_fd = TradeEquilibriumResult(
            x_sol=cf.x_sol, p_sol=cf.p_sol, y_sol=cf.y_sol, r_sol=cf.r_sol, w_sol=cf.w_sol,
            T_sol=cf.T_sol, XN_sol=cf.XN_sol, tariffs=cf.tariffs, cpi=cf.cpi,
            terms_of_trade=cf.terms_of_trade, exports=cf.exports, imports=cf.imports,
            intermediate_flows=cf.intermediate_flows,
            final_demand_flows=None,  # Intentionally None
            country_codes=cf.country_codes, sector_codes=cf.sector_codes,
        )
        base_none_fd = TradeEquilibriumResult(
            x_sol=base.x_sol, p_sol=base.p_sol, y_sol=base.y_sol, r_sol=base.r_sol, w_sol=base.w_sol,
            T_sol=base.T_sol, XN_sol=base.XN_sol, tariffs=base.tariffs, cpi=base.cpi,
            terms_of_trade=base.terms_of_trade, exports=base.exports, imports=base.imports,
            intermediate_flows=base.intermediate_flows,
            final_demand_flows=None,  # Intentionally None
            country_codes=base.country_codes, sector_codes=base.sector_codes,
        )

        try:
            res = decompose_welfare_effects(mock_calib_2c_2s, cf_none_fd, base_result=base_none_fd, target_country="HOM")
            assert res is not None
        except TypeError as err:
            pytest.fail(
                f"EXPOSED GAP: decompose_welfare_effects crashes when final_demand_flows is None: {err}"
            )

    def test_erp_different_tariff_inputs(self, mock_calib_2c_2s):
        """compute_effective_rate_of_protection with a 1D vector, a 4D tensor and a TariffScenario."""
        res1 = compute_effective_rate_of_protection(mock_calib_2c_2s, np.array([0.10, 0.25]), "HOM")
        assert isinstance(res1, EffectiveRateOfProtectionResult)
        assert len(res1.erp_balassa) == mock_calib_2c_2s.n_sectors

        scen = TariffScenario(importer="HOM", tariffs={("ROW", "*"): 0.15})
        res2 = compute_effective_rate_of_protection(mock_calib_2c_2s, scen, "HOM")
        assert len(res2.distortion_category) == mock_calib_2c_2s.n_sectors

        tau = np.ones((2, 2, 2, 2))
        tau[:, 1, :, 0] = 1.15  # the same 15% on HOM's imports from ROW
        res3 = compute_effective_rate_of_protection(mock_calib_2c_2s, tau, "HOM")
        np.testing.assert_allclose(res3.erp_corden, res2.erp_corden, rtol=0, atol=1e-14)
        np.testing.assert_allclose(res3.erp_balassa, res2.erp_balassa, rtol=0, atol=1e-14)

    def test_policy_analytics_rejects_name(self, mock_calib_2c_2s):
        """A scenario name is refused: named scenarios were removed."""
        with pytest.raises(TypeError, match="named scenarios were removed"):
            compute_effective_rate_of_protection(mock_calib_2c_2s, "tariff_10", "HOM")

    def test_target_country_is_required(self, mock_calib_2c_2s, mock_equilibrium_pair):
        base, cf = mock_equilibrium_pair
        with pytest.raises(TypeError, match="target_country"):
            decompose_welfare_effects(mock_calib_2c_2s, cf, base_result=base)
        with pytest.raises(TypeError, match="target_country"):
            calculate_tariff_revenue_incidence(mock_calib_2c_2s, cf, base_result=base)
        with pytest.raises(TypeError):
            compute_supply_chain_vulnerability(mock_calib_2c_2s)  # type: ignore[call-arg]

    @pytest.mark.parametrize("code", ["XYZ", "USA", "ROWX"])
    def test_unknown_country_is_refused(self, mock_calib_2c_2s, code):
        """No fallback country: an unknown code raises KeyError naming it."""
        with pytest.raises(KeyError, match=code):
            compute_supply_chain_vulnerability(mock_calib_2c_2s, code)

    def test_country_by_position_and_case(self, mock_calib_2c_2s):
        by_code = compute_supply_chain_vulnerability(mock_calib_2c_2s, "ROW")
        by_pos = compute_supply_chain_vulnerability(mock_calib_2c_2s, 1)
        by_case = compute_supply_chain_vulnerability(mock_calib_2c_2s, "row")
        assert by_code.country_code == by_pos.country_code == by_case.country_code == "ROW"
        np.testing.assert_array_equal(by_code.upstreamness, by_pos.upstreamness)
        with pytest.raises(IndexError):
            compute_supply_chain_vulnerability(mock_calib_2c_2s, 2)

    def test_erp_negative_protection_detection(self, mock_calib_2c_2s):
        """Verify detection of negative effective protection when intermediate tariffs exceed output tariff."""
        res = compute_effective_rate_of_protection(
            mock_calib_2c_2s,
            tariffs=np.array([0.80, 0.80]),
            target_country="HOM",
            output_tariffs={"AGR": 0.05, "MAN": 0.05},
        )
        assert np.any(res.is_negative_erp) or np.any(np.array(res.distortion_category) == "negative_protection")

    def test_supply_chain_vulnerability(self, mock_calib_2c_2s):
        """Verify Antras et al. (2012) upstreamness and supply chain vulnerability indices."""
        res = compute_supply_chain_vulnerability(mock_calib_2c_2s, target_country="HOM")
        assert isinstance(res, SupplyChainVulnerabilityResult)
        assert np.all(res.upstreamness >= 1.0)
        assert len(res.vulnerability_index) == mock_calib_2c_2s.n_sectors
        df = res.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert "Upstreamness" in df.columns

    def test_tariff_revenue_incidence_all_closures(self, mock_calib_2c_2s, mock_equilibrium_pair):
        """Verify tariff revenue incidence under all 5 fiscal recycling closures."""
        base, cf = mock_equilibrium_pair
        closures = ["lump_sum", "labor_tax", "capital_tax", "targeted_subsidy", "deficit_reduction"]

        for cl in closures:
            res = calculate_tariff_revenue_incidence(mock_calib_2c_2s, cf, base_result=base, closure=cl, target_country="HOM")
            assert isinstance(res, TariffRevenueIncidenceResult)
            assert res.closure == cl
            assert res.gross_tariff_revenue >= 0.0
            assert np.isfinite(res.net_efficiency_change)

        with pytest.raises(ValueError, match="Unknown closure"):
            calculate_tariff_revenue_incidence(
                mock_calib_2c_2s, cf, base_result=base, closure="nonexistent_closure", target_country="HOM"
            )


class TestTradeNamespaceAndExportHygiene:
    """Audit symbol re-exports, __all__ completeness, and namespace cleanliness."""

    def test_all_symbols_resolvable(self):
        """Every symbol in pt.__all__ must be resolvable via getattr(pt, symbol)."""
        missing = [sym for sym in pt.__all__ if not hasattr(pt, sym)]
        assert missing == [], f"Symbols listed in puremacro.trade.__all__ missing from package: {missing}"

    def test_policy_analytics_symbols_reexported(self):
        """Every symbol declared in policy_analytics.__all__ must be re-exported at top-level."""
        expected_policy_symbols = [
            "EffectiveRateOfProtectionResult",
            "WelfareDecompositionResult",
            "SupplyChainVulnerabilityResult",
            "TariffRevenueIncidenceResult",
            "compute_effective_rate_of_protection",
            "decompose_welfare_effects",
            "compute_supply_chain_vulnerability",
            "calculate_tariff_revenue_incidence",
        ]
        for sym in expected_policy_symbols:
            assert hasattr(pt, sym), f"puremacro.trade is missing re-export of {sym}"
            assert sym in pt.__all__, f"{sym} not declared in puremacro.trade.__all__"

    def test_result_containers_reexported(self):
        """Key result containers must be re-exported in puremacro.trade."""
        expected_results = [
            "TradeCalibrationResult",
            "TradeEquilibriumResult",
            "GearyKhamisResult",
        ]
        for res_cls in expected_results:
            assert hasattr(pt, res_cls)
            assert res_cls in pt.__all__

    def test_namespace_cleanliness(self):
        """Verify no private helper functions (prefixed with _) leaked into __all__."""
        leaked_private = [sym for sym in pt.__all__ if sym.startswith("_")]
        assert leaked_private == [], f"Private symbols leaked in __all__: {leaked_private}"
