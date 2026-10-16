"""The tariff-policy entry points name their players; nothing defaults to a country.

Also covers the removed objective (a welfare metric that lived in a module that
did not return) and the code lookups on a calibration of any size.
"""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.trade import (
    build_strategic_tariffs,
    calibrate_trade_model,
    compute_unilateral_optimal_tariff,
    compute_welfare_payoff_matrix,
    evaluate_national_welfare,
    get_country_code,
    resolve_country_indices,
    solve_multilateral_nash_tariffs,
    solve_trade_equilibrium,
)
from puremacro.trade import game, optimal_tariffs
from puremacro.trade.data import synthetic_codes, synthetic_icio_table


@pytest.fixture(scope="module")
def calib():
    countries, sectors = synthetic_codes(5, 4)
    return calibrate_trade_model(synthetic_icio_table(5, 4, seed=0),
                                 country_codes=countries, sector_codes=sectors)


@pytest.fixture(scope="module")
def base(calib):
    res = solve_trade_equilibrium(calib, tol=1e-8)
    assert res.converged
    return res


def test_players_are_required(calib):
    with pytest.raises(TypeError, match="player_countries"):
        solve_multilateral_nash_tariffs(calib)
    with pytest.raises(TypeError, match="once"):
        solve_multilateral_nash_tariffs(calib, ["C01"], players=["C02"])
    with pytest.raises(TypeError):
        compute_unilateral_optimal_tariff(calib)
    with pytest.raises(TypeError):
        compute_welfare_payoff_matrix(calib)


def test_welfare_needs_a_country(calib, base):
    with pytest.raises(TypeError, match="country_idx"):
        evaluate_national_welfare(calib, base)


def test_removed_metric_is_refused(calib, base):
    with pytest.raises(ValueError, match="Unknown welfare metric 'harberger'"):
        evaluate_national_welfare(calib, base, country_idx="C01", metric="harberger")


def test_codes_resolve_against_the_calibration(calib):
    assert resolve_country_indices(calib, "c03") == [2]
    assert resolve_country_indices(calib, ["C05", 0]) == [0, 4]
    assert get_country_code(calib, 4) == "C05"
    with pytest.raises(IndexError):
        get_country_code(calib, 5)
    with pytest.raises(ValueError, match="XYZ"):
        resolve_country_indices(calib, "XYZ")
    # No EU member in the calibration: the bloc is empty, never a stand-in country.
    assert resolve_country_indices(calib, "EUR") == []


def test_strategic_tariffs_follow_codes(calib):
    tau, tau_fd = build_strategic_tariffs(calib, {"C03": {"C01": 0.2}})
    ns, nc = calib.n_sectors, calib.n_countries
    expected = np.ones((ns, nc, ns, nc))
    expected[:, 0, :, 2] = 1.2
    np.testing.assert_array_equal(tau, expected)
    assert np.all(tau_fd[:, 0, :, 2] == 1.2)
    assert np.count_nonzero(tau_fd != 1.0) == ns * calib.n_final_demand


def test_unilateral_geary_khamis_search_runs_on_a_small_table(calib):
    res = compute_unilateral_optimal_tariff(calib, "C02", tariff_max=0.2, num_grid=3, method="grid")
    assert res.country_code == "C02"
    assert res.tariff_grid.shape == (3,)
    assert np.all(np.isfinite(res.welfare_curve))
    assert 0.0 <= res.optimal_tariff_rate <= 0.2
    assert res.metadata["boundary"] in ("lower", "upper", "interior")


def test_game_module_reexports():
    assert set(game.__all__) <= set(optimal_tariffs.__all__)
    assert set(game.__all__) == {
        "resolve_country_indices", "get_country_code", "evaluate_national_welfare",
        "build_strategic_tariffs", "compute_unilateral_optimal_tariff",
        "solve_multilateral_nash_tariffs", "compute_welfare_payoff_matrix",
    }
