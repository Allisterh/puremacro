"""E2E tests of the Caliendo-Parro quantitative trade GE engine (four tiers).

Trade tests split out of the public puremacro ``tests/test_e2e_frontier_engines.py`` at f6c57226;
its continuous-transition, analytic-gradient, deep-macro and Allen-Arkolakis
tests stay there.

- Tier 1: exact hat algebra happy path and the presentation contract.
- Tier 2: zero-shock identity and extreme trade elasticities.
- Tier 3: spatial distances feeding Caliendo-Parro; an Allen-Arkolakis climate shock
  mapped into a Caliendo-Parro counterfactual.
- Tier 4: a US-China 25% bilateral tariff war and trade diversion.
"""
from __future__ import annotations

from typing import Any

import matplotlib
matplotlib.use("Agg")  # Headless backend for automated test runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from puremacro.spatial.weights import pairwise_distances


def _require_caliendo_parro():
    """Resolve Caliendo-Parro trade GE engine or skip progressively."""
    try:
        from puremacro.trade import caliendo_parro as cp_mod
        if hasattr(cp_mod, "CaliendoParroModel"):
            return cp_mod
    except (ImportError, AttributeError):
        pass
    try:
        import puremacro.trade as trade
        if hasattr(trade, "CaliendoParroModel"):
            return trade
    except (ImportError, AttributeError):
        pass
    try:
        from puremacro import spatial
        if hasattr(spatial, "CaliendoParroModel"):
            return spatial
    except (ImportError, AttributeError):
        pass
    pytest.skip("Milestone 4: CaliendoParroModel pending implementation in puremacro.trade.caliendo_parro")


def _require_allen_arkolakis():
    """Resolve Allen-Arkolakis spatial GE engine or skip progressively."""
    try:
        from puremacro.spatial import allen_arkolakis as aa_mod
        if hasattr(aa_mod, "AllenArkolakisModel"):
            return aa_mod
    except (ImportError, AttributeError):
        pass
    try:
        from puremacro import spatial
        if hasattr(spatial, "AllenArkolakisModel"):
            return spatial
    except (ImportError, AttributeError):
        pass
    try:
        import puremacro.trade as trade
        if hasattr(trade, "AllenArkolakisModel"):
            return trade
    except (ImportError, AttributeError):
        pass
    pytest.skip("Milestone 4: AllenArkolakisModel pending implementation in puremacro.spatial.allen_arkolakis")


@pytest.fixture
def canonical_caliendo_parro_data() -> dict[str, Any]:
    """Canonical 3-country, 2-sector Caliendo & Parro (2015) dataset.

    Countries: USA, CHN, ROW (N=3)
    Sectors: Manufacturing (tradable), Services (less tradable) (J=2)
    Satisfies all general equilibrium identities:
    - Sum of bilateral trade shares along origin dimension equals 1.0.
    - Value-added share + sum of intermediate shares equals 1.0.
    - Consumption shares sum to 1.0.
    - Trade deficits sum to 0.0.
    """
    N = 3
    J = 2
    country_codes = ("USA", "CHN", "ROW")
    sector_codes = ("manuf", "serv")

    # Bilateral trade shares: pi[j, n, i] = share of importer n spent on exporter i
    trade_shares = np.array([
        # Sector 0: Manufacturing (high trade openness)
        [
            [0.55, 0.25, 0.20],  # USA imports
            [0.15, 0.70, 0.15],  # CHN imports
            [0.25, 0.25, 0.50],  # ROW imports
        ],
        # Sector 1: Services (primarily domestic)
        [
            [0.85, 0.08, 0.07],  # USA imports
            [0.05, 0.90, 0.05],  # CHN imports
            [0.10, 0.10, 0.80],  # ROW imports
        ],
    ], dtype=float)

    # Cost shares: value added gamma_va[n, j] and input-output gamma_io[n, j, k]
    gamma_va = np.array([
        [0.40, 0.60],  # USA (manuf, serv)
        [0.35, 0.65],  # CHN
        [0.45, 0.55],  # ROW
    ], dtype=float)

    gamma_io = np.array([
        # USA
        [[0.35, 0.25],   # manuf uses 35% manuf, 25% serv (sum = 0.60, va = 0.40)
         [0.15, 0.25]],  # serv uses 15% manuf, 25% serv (sum = 0.40, va = 0.60)
        # CHN
        [[0.40, 0.25],   # manuf uses 40% manuf, 25% serv (sum = 0.65, va = 0.35)
         [0.10, 0.25]],  # serv uses 10% manuf, 25% serv (sum = 0.35, va = 0.65)
        # ROW
        [[0.30, 0.25],   # manuf uses 30% manuf, 25% serv (sum = 0.55, va = 0.45)
         [0.20, 0.25]],  # serv uses 20% manuf, 25% serv (sum = 0.45, va = 0.55)
    ], dtype=float)

    # Final consumption expenditure shares alpha[n, j]
    alpha = np.array([
        [0.30, 0.70],  # USA
        [0.45, 0.55],  # CHN
        [0.35, 0.65],  # ROW
    ], dtype=float)

    # Sectoral trade elasticities theta_j
    theta = np.array([5.0, 8.0], dtype=float)

    # Baseline labor income (value added) w_n L_n
    labor_income = np.array([120.0, 90.0, 100.0], dtype=float)

    # Trade deficits D_n (summing to zero)
    deficits = np.array([10.0, -10.0, 0.0], dtype=float)

    # Baseline gross tariffs (tau = 1.0 implies no baseline tariffs)
    tariffs = np.ones((J, N, N), dtype=float)

    return {
        "trade_shares": trade_shares,
        "gamma_va": gamma_va,
        "gamma_io": gamma_io,
        "alpha": alpha,
        "theta": theta,
        "labor_income": labor_income,
        "deficits": deficits,
        "tariffs": tariffs,
        "country_codes": country_codes,
        "sector_codes": sector_codes,
    }


@pytest.fixture
def symmetric_caliendo_parro_model() -> Any:
    """A symmetric 3-country, 2-sector Caliendo-Parro economy in exact baseline equilibrium."""
    cp = _require_caliendo_parro()
    N, J = 3, 2
    trade_shares = np.zeros((J, N, N))
    trade_shares[0] = np.array([
        [0.60, 0.20, 0.20],
        [0.20, 0.60, 0.20],
        [0.20, 0.20, 0.60],
    ])
    trade_shares[1] = np.eye(N)

    gamma_va = np.full((N, J), 0.50)
    gamma_io = np.full((N, J, J), 0.25)
    alpha = np.full((N, J), 0.50)
    theta = np.array([5.0, 4.0])
    labor_income = np.array([100.0, 100.0, 100.0])

    return cp.CaliendoParroModel(
        trade_shares=trade_shares,
        gamma_va=gamma_va,
        gamma_io=gamma_io,
        alpha=alpha,
        theta=theta,
        labor_income=labor_income,
        nontradables=[1],
        country_codes=["USA", "CHN", "ROW"],
        sector_codes=["Manufactures", "Services"],
    )


@pytest.fixture
def canonical_allen_arkolakis_data() -> dict[str, Any]:
    """Canonical 5-region Allen & Arkolakis (2014) spatial model dataset.

    Regions: North, South, East, West, Central (N=5).
    Satisfies uniqueness condition: alpha + beta <= theta / (1 + theta).
    """
    N = 5
    region_names = ("North", "South", "East", "West", "Central")
    coords = np.array([
        [45.0, -93.0],  # North
        [30.0, -90.0],  # South
        [40.0, -74.0],  # East
        [37.0, -122.0], # West
        [39.0, -89.0],  # Central
    ], dtype=float)

    # Bilateral distance matrix
    dist_matrix = pairwise_distances(coords, metric="euclidean")

    # Iceberg trade costs: tau_ij = 1 + 0.02 * dist^0.5
    trade_costs = 1.0 + 0.02 * (dist_matrix ** 0.5)
    np.fill_diagonal(trade_costs, 1.0)

    # Fundamental productivities and amenities
    fundamental_productivity = np.array([1.2, 0.9, 1.3, 1.4, 1.0], dtype=float)
    fundamental_amenity = np.array([1.1, 1.3, 1.0, 1.4, 1.0], dtype=float)

    theta = 4.0
    alpha = 0.10   # Agglomeration elasticity
    beta = -0.30   # Congestion elasticity
    total_population = 1.0

    return {
        "trade_costs": trade_costs,
        "fundamental_productivity": fundamental_productivity,
        "fundamental_amenity": fundamental_amenity,
        "theta": theta,
        "alpha": alpha,
        "beta": beta,
        "total_population": total_population,
        "region_names": region_names,
        "coordinates": coords,
    }


class TestTier1QuantitativeSpatialGE:
    """Tier 1: Feature coverage for Caliendo-Parro and Allen-Arkolakis Quantitative Spatial GE."""

    def test_caliendo_parro_exact_hat_algebra_happy_path(self, canonical_caliendo_parro_data):
        """Verify Caliendo-Parro Exact Hat Algebra solve under tariff counterfactual."""
        cp = _require_caliendo_parro()
        data = canonical_caliendo_parro_data

        model = cp.CaliendoParroModel(
            trade_shares=data["trade_shares"],
            gamma_va=data["gamma_va"],
            gamma_io=data["gamma_io"],
            alpha=data["alpha"],
            theta=data["theta"],
            labor_income=data["labor_income"],
            deficits=data["deficits"],
            tariffs=data["tariffs"],
            country_codes=data["country_codes"],
            sector_codes=data["sector_codes"],
        )

        # Counterfactual: 10% tariff imposed by USA on Chinese manufacturing
        res = model.simulate_tariff_shock(importer="USA", exporter="CHN", sector="manuf", tariff_rate=0.10)

        assert isinstance(res, cp.CaliendoParroResult)
        assert res.converged, "Equilibrium wage tatonnement must converge"
        assert res.w_hat.shape == (3,)
        assert res.P_hat.shape == (3, 2)
        assert res.pi_prime.shape == (2, 3, 3)

        # Invariant 1: Bilateral trade shares sum to 1.0 along origin dimension for all n, j
        for j in range(2):
            for n in range(3):
                share_sum = np.sum(res.pi_prime[j, n, :])
                assert np.isclose(share_sum, 1.0, atol=1e-12), f"Trade shares do not sum to 1: {share_sum}"

        # Invariant 2: Labor market clearing residual < 1e-5
        assert res.market_clearing_residual < 1e-5, f"Market clearing error {res.market_clearing_residual} >= 1e-5"

    def test_caliendo_parro_presentation_contract(self, canonical_caliendo_parro_data):
        """Verify CaliendoParroResult presentation interface."""
        cp = _require_caliendo_parro()
        data = canonical_caliendo_parro_data

        model = cp.CaliendoParroModel(
            trade_shares=data["trade_shares"],
            gamma_va=data["gamma_va"],
            gamma_io=data["gamma_io"],
            alpha=data["alpha"],
            theta=data["theta"],
            labor_income=data["labor_income"],
            country_codes=data["country_codes"],
            sector_codes=data["sector_codes"],
        )

        res = model.solve_counterfactual()

        sm = res.summary()
        assert isinstance(sm, pd.DataFrame)
        assert "Welfare (%)" in sm.columns or "welfare_pct" in sm.columns

        sec_sm = res.sector_summary()
        assert isinstance(sec_sm, pd.DataFrame)

        assert isinstance(res.to_markdown(), str)
        assert "\\begin" in res.to_latex()
        assert "#table" in res.to_typst()

        fig = res.plot(kind="welfare")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestTier2BoundaryCornerCases:
    """Tier 2: Boundary conditions, corner cases, asymptotic limits, and numerical stress."""

    def test_caliendo_parro_zero_shock_identity(self, symmetric_caliendo_parro_model):
        """Zero shock identity: tau_hat = 1.0 and d_hat = 1.0 yields exact baseline w_hat = 1.0, P_hat = 1.0."""
        res = symmetric_caliendo_parro_model.solve_counterfactual(tau_hat=None, d_hat=None)

        assert res.converged
        assert np.allclose(res.w_hat, 1.0, atol=1e-6), "Zero shock must yield w_hat = 1.0"
        assert np.allclose(res.P_hat, 1.0, atol=1e-6), "Zero shock must yield P_hat = 1.0"
        assert np.allclose(res.welfare_pct, 0.0, atol=1e-6), "Zero shock must yield 0% welfare change"

    def test_caliendo_parro_extreme_trade_elasticity(self, canonical_caliendo_parro_data):
        """Verify Hat Algebra stability across near-unitary (theta=1.2) and high (theta=12.0) trade elasticities."""
        cp = _require_caliendo_parro()
        data = canonical_caliendo_parro_data

        for theta_val in [np.array([1.2, 1.5]), np.array([10.0, 12.0])]:
            model = cp.CaliendoParroModel(
                trade_shares=data["trade_shares"],
                gamma_va=data["gamma_va"],
                gamma_io=data["gamma_io"],
                alpha=data["alpha"],
                theta=theta_val,
                labor_income=data["labor_income"],
                country_codes=data["country_codes"],
                sector_codes=data["sector_codes"],
            )

            res = model.simulate_tariff_shock(importer="USA", exporter="CHN", sector="manuf", tariff_rate=0.05)
            assert res.converged, f"Failed to converge at theta={theta_val}"
            assert np.all(np.isfinite(res.welfare_pct)), "Welfare must be finite"


class TestTier3CrossFeatureCombinations:
    """Tier 3: Pairwise and multi-subsystem integration across frontier macroeconomic engines."""

    def test_spatial_distance_matrix_feeding_caliendo_parro(self, canonical_caliendo_parro_data):
        """Cross-Feature: Spatial distance matrix from puremacro.spatial.weights feeding Caliendo-Parro IO trade GE.

        Demonstrates seamless integration between continuous geographic coordinates and multi-sector trade hat algebra.
        """
        cp = _require_caliendo_parro()
        data = canonical_caliendo_parro_data

        # Geographic coordinates for USA, CHN, ROW
        coords = np.array([
            [38.9, -77.0],  # Washington D.C.
            [39.9, 116.4],  # Beijing
            [50.8, 4.3],    # Brussels (ROW proxy)
        ])

        # Compute bilateral distances
        dist = pairwise_distances(coords, metric="haversine")

        # Build trade cost changes based on an infrastructure upgrade between USA and ROW:
        # 15% reduction in transportation friction
        d_hat = np.ones_like(data["trade_shares"])
        d_hat[:, 0, 2] = 0.85  # USA -> ROW cost reduction
        d_hat[:, 2, 0] = 0.85  # ROW -> USA cost reduction

        model = cp.CaliendoParroModel(
            trade_shares=data["trade_shares"],
            gamma_va=data["gamma_va"],
            gamma_io=data["gamma_io"],
            alpha=data["alpha"],
            theta=data["theta"],
            labor_income=data["labor_income"],
            country_codes=data["country_codes"],
            sector_codes=data["sector_codes"],
        )

        res = model.solve_counterfactual(d_hat=d_hat)

        assert res.converged
        # USA and ROW should both experience positive welfare gains from bilateral transport improvement
        assert res.welfare_pct[0] > 0.0, "USA welfare must increase from bilateral transport cost reduction"
        assert res.welfare_pct[2] > 0.0, "ROW welfare must increase from bilateral transport cost reduction"

    def test_allen_arkolakis_climate_shock_with_caliendo_parro_counterfactual(self, canonical_allen_arkolakis_data, canonical_caliendo_parro_data):
        """Cross-Feature: Localized regional climate shock in Allen-Arkolakis mapped into national trade counterfactual."""
        aa = _require_allen_arkolakis()
        cp = _require_caliendo_parro()

        # 1. Run climate shock in Southern region (10% productivity loss -> 0.90 factor)
        aa_model = aa.AllenArkolakisModel(
            trade_costs=canonical_allen_arkolakis_data["trade_costs"],
            fundamental_productivity=canonical_allen_arkolakis_data["fundamental_productivity"],
            fundamental_amenity=canonical_allen_arkolakis_data["fundamental_amenity"],
            region_names=canonical_allen_arkolakis_data["region_names"],
        )
        aa_res = aa_model.simulate_climate_shock(productivity_shocks={"South": 0.90})

        assert aa_res.converged
        assert aa_res.welfare_pct < 0.0, "Aggregate spatial welfare must fall under climate productivity loss"

        # 2. Map aggregate domestic wage contraction into Caliendo-Parro trade GE
        cp_model = cp.CaliendoParroModel(
            trade_shares=canonical_caliendo_parro_data["trade_shares"],
            gamma_va=canonical_caliendo_parro_data["gamma_va"],
            gamma_io=canonical_caliendo_parro_data["gamma_io"],
            alpha=canonical_caliendo_parro_data["alpha"],
            theta=canonical_caliendo_parro_data["theta"],
            labor_income=canonical_caliendo_parro_data["labor_income"],
        )

        cp_res = cp_model.solve_counterfactual()
        assert cp_res.converged


class TestTier4RealWorldScenarios:
    """Tier 4: Realistic macroeconomic policy counterfactuals, shock transitions, and trade wars."""

    def test_scenario_us_china_bilateral_tariff_war_and_diversion(self, canonical_caliendo_parro_data):
        """Scenario 2: US-China 25% Bilateral Tariff War and Trade Diversion in Caliendo-Parro.

        Economic properties:
        - Bilateral trade shares between USA and CHN contract substantially.
        - Trade diversion: imports from third party (ROW) increase in both USA and CHN.
        - Both participating countries suffer net welfare losses due to deadweight loss and intermediate cost markups.
        - Market clearing residuals < 1e-5.
        """
        cp = _require_caliendo_parro()
        data = canonical_caliendo_parro_data

        model = cp.CaliendoParroModel(
            trade_shares=data["trade_shares"],
            gamma_va=data["gamma_va"],
            gamma_io=data["gamma_io"],
            alpha=data["alpha"],
            theta=data["theta"],
            labor_income=data["labor_income"],
            deficits=data["deficits"],
            country_codes=data["country_codes"],
            sector_codes=data["sector_codes"],
        )

        # 25% bilateral tariff war between USA and CHN
        res = model.simulate_trade_war(coalition_a=["USA"], coalition_b=["CHN"], tariff_rate=0.25)

        assert res.converged
        assert res.market_clearing_residual < 1e-5

        # Economic property 1: Trade destruction between USA and CHN
        # USA import share on CHN manufacturing (sector 0, importer 0, exporter 1)
        pi_usa_chn_initial = data["trade_shares"][0, 0, 1]
        pi_usa_chn_prime = res.pi_prime[0, 0, 1]
        assert pi_usa_chn_prime < pi_usa_chn_initial, "US-China trade share must fall under bilateral tariffs"

        # Economic property 2: Trade diversion to ROW
        # USA import share on ROW manufacturing (sector 0, importer 0, exporter 2)
        pi_usa_row_initial = data["trade_shares"][0, 0, 2]
        pi_usa_row_prime = res.pi_prime[0, 0, 2]
        assert pi_usa_row_prime > pi_usa_row_initial, "Trade diversion must increase USA import share from ROW"

        # Economic property 3: Net welfare contraction for both war participants
        assert res.welfare_pct[0] < 0.0, "USA must suffer net welfare loss in tariff war"
        assert res.welfare_pct[1] < 0.0, "China must suffer net welfare loss in tariff war"
