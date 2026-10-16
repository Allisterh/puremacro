"""Contract of the static trade model's public entry points.

- Named scenarios are gone: every scenario is an explicit ``TariffScenario``.
- Legacy accounting is gone: ``accounting="consistent"`` is the only accepted value.
- Fiscal recycling and capacity barriers name their country and sector; nothing defaults.
- No hidden table dimensions: a 5-country, 4-sector table runs end to end.
- The Geary-Khamis numeraire is a country code.
"""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.trade import (
    TariffScenario,
    build_tariff_matrices,
    calibrate_trade_model,
    compute_geary_khamis,
    compute_geary_khamis_ppp,
    run_scenario,
    run_scenario_batch,
    solve_equilibrium,
    solve_trade_equilibrium,
)
from puremacro.trade.data import synthetic_icio_table
from puremacro.trade.solver import solve_keller_pac


@pytest.fixture(scope="module")
def calib_5x4():
    return calibrate_trade_model(synthetic_icio_table(5, 4, seed=0))


@pytest.fixture(scope="module")
def ten_percent_on_c02():
    return TariffScenario("C01", {("C02", "*"): 0.10}, name="ten_on_C02")


@pytest.fixture(scope="module")
def solved_5x4(calib_5x4, ten_percent_on_c02):
    return run_scenario(calib_5x4, ten_percent_on_c02)


# ---------------------------------------------------------------------------
# Removed API: clear errors
# ---------------------------------------------------------------------------

def test_named_scenario_removed(calib_5x4):
    with pytest.raises(TypeError, match="TariffScenario"):
        run_scenario(calib_5x4, "t1")
    with pytest.raises(TypeError, match="named scenarios were removed"):
        build_tariff_matrices(calib_5x4, "t1")
    with pytest.raises(TypeError, match="TariffScenario"):
        run_scenario_batch(calib_5x4, ["t1"])
    with pytest.raises(TypeError, match="TariffScenario"):
        solve_keller_pac(calib_5x4, "t1")


def test_legacy_accounting_removed(calib_5x4):
    with pytest.raises(ValueError, match="removed"):
        solve_equilibrium(calib_5x4, accounting="legacy")
    with pytest.raises(ValueError, match="removed"):
        solve_trade_equilibrium(calib_5x4, accounting="anything-else")


def test_unknown_solver_keywords_are_refused(calib_5x4):
    # Removed switches (the revenue mode, the precedence flag) are unknown
    # keywords now, refused instead of silently ignored.
    with pytest.raises(TypeError, match="unexpected keyword"):
        solve_trade_equilibrium(calib_5x4, tariff_revenue_mode="schedule")
    with pytest.raises(TypeError, match="unexpected keyword"):
        solve_keller_pac(calib_5x4, 0.1, recycling_params={})


def test_recycling_needs_country(calib_5x4):
    with pytest.raises(TypeError, match="recycle_country"):
        solve_equilibrium(calib_5x4, fiscal_closure="capital_tax")


def test_capacity_needs_country_and_sector(calib_5x4):
    with pytest.raises(TypeError, match="capacity_country"):
        solve_equilibrium(calib_5x4, capacity_margins={"S01": 0.1}, capacity_sector="S01")
    with pytest.raises(TypeError, match="capacity_sector"):
        solve_equilibrium(calib_5x4, capacity_margins={"S01": 0.1}, capacity_country="C01")


def test_named_policy_options_reach_the_accounting(calib_5x4):
    # With the country named, the request is well formed; consistent accounting
    # implements lump-sum rebates and no capacity barrier, and says so.
    with pytest.raises(NotImplementedError, match="lump-sum"):
        solve_equilibrium(calib_5x4, fiscal_closure="capital_tax", recycle_country="C01")
    with pytest.raises(NotImplementedError, match="capacity"):
        solve_equilibrium(calib_5x4, capacity_margins={"S01": 0.1},
                          capacity_country="C01", capacity_sector="S01")


def test_accelerated_backends_are_refused(calib_5x4):
    with pytest.raises(NotImplementedError, match="NumPy"):
        solve_trade_equilibrium(calib_5x4, backend="mlx")


# ---------------------------------------------------------------------------
# Any table shape
# ---------------------------------------------------------------------------

def test_runs_on_5x4(calib_5x4, solved_5x4):
    assert (calib_5x4.n_countries, calib_5x4.n_sectors) == (5, 4)
    assert solved_5x4.converged is True
    assert solved_5x4.metadata["scenario_name"] == "ten_on_C02"
    assert solved_5x4.metadata["scenario_importer"] == "C01"
    assert solved_5x4.metadata["accounting"] == "consistent"
    # The tariff is collected by C01 only.
    assert solved_5x4.tariffs[0] > 0
    np.testing.assert_array_equal(solved_5x4.tariffs[1:], 0.0)


def test_tariff_matrices_follow_the_scenario(calib_5x4, ten_percent_on_c02):
    tau, tau_fd = build_tariff_matrices(calib_5x4, ten_percent_on_c02)
    assert tau.shape == (4, 5, 4, 5)
    assert tau_fd.shape == (4, 5, calib_5x4.n_final_demand, 5)
    np.testing.assert_allclose(tau[:, 1, :, 0], 1.10)
    np.testing.assert_allclose(tau_fd[:, 1, :, 0], 1.10)
    expected = np.ones_like(tau)
    expected[:, 1, :, 0] = 1.10
    np.testing.assert_array_equal(tau, expected)


def test_sector_entry_overrides_wildcard(calib_5x4):
    scen = TariffScenario("C03", {("C01", "*"): 0.2, ("C01", "S02"): 0.05, ("C04", "S04"): 0.3})
    tau, _ = build_tariff_matrices(calib_5x4, scen)
    np.testing.assert_allclose(tau[[0, 2, 3], 0, :, 2], 1.2)
    np.testing.assert_allclose(tau[1, 0, :, 2], 1.05)
    np.testing.assert_allclose(tau[3, 3, :, 2], 1.3)
    np.testing.assert_allclose(tau[:3, 3, :, 2], 1.0)


def test_array_tariffs_match_mapping(calib_5x4):
    rates = np.zeros((5, 4))
    rates[1, :] = 0.10
    from_array = build_tariff_matrices(calib_5x4, TariffScenario("C01", rates))
    from_mapping = build_tariff_matrices(calib_5x4, TariffScenario("C01", {("C02", "*"): 0.10}))
    for a, b in zip(from_array, from_mapping):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("scenario, message", [
    (TariffScenario("XX", {("C02", "*"): 0.1}), "Unknown country code 'XX'"),
    (TariffScenario("C01", {("XX", "*"): 0.1}), "Unknown country code 'XX'"),
    (TariffScenario("C01", {("C02", "XX"): 0.1}), "Unknown sector code 'XX'"),
    (TariffScenario("C01", {("C01", "S01"): 0.1}), "Domestic flows"),
    (TariffScenario("C01", np.zeros((4, 4))), "shape"),
])
def test_bad_scenarios_name_the_problem(calib_5x4, scenario, message):
    with pytest.raises(ValueError, match=message):
        build_tariff_matrices(calib_5x4, scenario)


@pytest.mark.parametrize("tariffs, error", [
    ({("C02", "*"): -0.1}, ValueError),
    ({("C02", "*"): float("nan")}, ValueError),
    ({("C02", "*"): "10%"}, TypeError),
    ({"C02": 0.1}, TypeError),
    ([0.1], TypeError),
])
def test_bad_rates_are_rejected(tariffs, error):
    with pytest.raises(error):
        TariffScenario("C01", tariffs)


def test_batch_runs_on_5x4(calib_5x4, ten_percent_on_c02):
    batch = run_scenario_batch(calib_5x4, [ten_percent_on_c02], numeraire="C05")
    assert list(batch.scenarios) == ["baseline", "ten_on_C02"]
    assert all(res.converged for res in batch.scenarios.values())
    np.testing.assert_array_equal(batch.geary_khamis["baseline"].real_gdp_growth, 0.0)
    assert batch.real_gdp_table.shape == (5, 2)
    with pytest.raises(ValueError, match="distinct"):
        run_scenario_batch(calib_5x4, [ten_percent_on_c02, ten_percent_on_c02])


def test_real_gdp_growth_does_not_depend_on_the_numeraire(calib_5x4, ten_percent_on_c02):
    # Review fix: growth used to compare pi-valued counterfactual quantities with
    # an unweighted baseline sum, so it flipped sign with the numeraire.
    growth = {
        num: run_scenario_batch(calib_5x4, [ten_percent_on_c02], numeraire=num)
        .geary_khamis["ten_on_C02"].real_gdp_growth
        for num in (None, "C01", "C05")
    }
    assert np.all(np.isfinite(growth[None]))
    np.testing.assert_allclose(growth["C01"], growth[None], rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(growth["C05"], growth[None], rtol=1e-6, atol=1e-12)


def test_real_gdp_growth_is_a_quantity_index():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.5, 2.0, size=(3, 4))
    q_base = rng.uniform(1.0, 5.0, size=(3, 4))
    for num in (None, "A", "D"):
        same = compute_geary_khamis_ppp(pfd_sol=p, c_sol=q_base, c_base=q_base,
                                        country_codes=("A", "B", "C", "D"), numeraire=num)
        np.testing.assert_allclose(same.real_gdp_growth, 0.0, atol=1e-12)
        grown = compute_geary_khamis_ppp(pfd_sol=p, c_sol=1.1 * q_base, c_base=q_base,
                                         country_codes=("A", "B", "C", "D"), numeraire=num)
        np.testing.assert_allclose(grown.real_gdp_growth, 10.0, rtol=1e-10)


def test_keller_pac_accepts_a_scenario(calib_5x4, ten_percent_on_c02, solved_5x4):
    res = solve_keller_pac(calib_5x4, ten_percent_on_c02)
    assert res.converged is True
    np.testing.assert_allclose(res.w_sol, solved_5x4.w_sol, rtol=1e-6)


# ---------------------------------------------------------------------------
# Geary-Khamis numeraire by code
# ---------------------------------------------------------------------------

def test_numeraire_by_code(calib_5x4, solved_5x4):
    base = solve_equilibrium(calib_5x4)
    gk = compute_geary_khamis(solved_5x4, base, calib_5x4, numeraire="C03")
    assert abs(gk.ppp[2] - 1.0) <= 1e-12
    assert gk.metadata["numeraire"] == "C03"
    free = compute_geary_khamis(solved_5x4, base, calib_5x4)
    # Normalisation rescales PPPs and keeps their ratios.
    np.testing.assert_allclose(gk.ppp, free.ppp / free.ppp[2], rtol=1e-12)
    with pytest.raises(ValueError, match="Unknown numeraire"):
        compute_geary_khamis(solved_5x4, base, calib_5x4, numeraire="XX")
