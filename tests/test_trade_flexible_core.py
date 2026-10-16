"""The flexible CGE on the consistent accounting: equivalence, any table size, required codes."""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.trade import (
    calibrate_trade_model,
    compute_equilibrium_residuals,
    solve_flexible_trade_equilibrium,
    solve_trade_equilibrium,
)
from puremacro.trade._accounting import evaluate
from puremacro.trade.data import synthetic_codes, synthetic_icio_table
from puremacro.trade.flexible import FlexibleTradeModelConfig, _flexible_blocks
from puremacro.trade.solver import _resolve_tariffs


def _calib(nc: int, ns: int, seed: int, nfd: int = 3):
    cc, sc = synthetic_codes(nc, ns)
    return calibrate_trade_model(synthetic_icio_table(nc, ns, nfd=nfd, seed=seed), ns=ns, nc=nc, nfd=nfd,
                                 country_codes=cc, sector_codes=sc)


@pytest.fixture(scope="module")
def calib_5x4():
    return _calib(5, 4, seed=0)


@pytest.mark.parametrize("units", ["numeraire", "world_income"])
def test_flexible_blocks_equal_accounting_at_defaults(calib_5x4, units):
    calib = calib_5x4
    ta, tf, _, _ = _resolve_tariffs(calib, 0.1, 0.05)
    x0 = solve_trade_equilibrium(calib, tau=0.1, tau_fd=0.05).x_sol
    x = x0 + np.random.default_rng(1).normal(0.0, 0.02, x0.size)
    a = evaluate(x, calib, ta, tf, None, None, foreign_saving_units=units)
    b = _flexible_blocks(x, calib, FlexibleTradeModelConfig(), ta, tf, units)
    for key, value in a.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_allclose(b[key], value, rtol=1e-12, atol=1e-9 * max(1.0, np.max(np.abs(value))),
                                       err_msg=key)


def test_flexible_on_5x4(calib_5x4):
    calib = calib_5x4
    res = solve_flexible_trade_equilibrium(calib, rho_va=0.7, sigma_y=0.3, variable_markups=True,
                                           sigma_trade=4.0, tau=0.1, tol=1e-8)
    assert res.converged
    assert res.metadata["routing"] == "auto"
    assert res.x_sol.shape == (2 * 20 + 4 * 5 - 1,)
    # Every physical equation, including the goods equation dropped for Walras' law and
    # the realised net exports, holds at the reported state.
    assert np.max(np.abs(res.metadata["physical_residuals"])) <= 1e-8
    frame = res.factor_allocation_frame()
    np.testing.assert_allclose(frame.groupby("country", sort=False)["labor"].sum().to_numpy(),
                               calib.l_endow.ravel(), rtol=1e-9)
    # Re-evaluating the solution reproduces its residual.
    again = solve_flexible_trade_equilibrium(calib, rho_va=0.7, sigma_y=0.3, variable_markups=True,
                                             sigma_trade=4.0, tau=0.1, x0=res.x_sol, max_iter=0)
    np.testing.assert_allclose(again.residual_norm, res.residual_norm, rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize("nc, ns, nfd, seed", [(3, 3, 1, 2), (6, 4, 2, 0), (3, 3, 4, 3)])
def test_flexible_any_table_and_categories(nc, ns, nfd, seed):
    calib = _calib(nc, ns, seed=seed, nfd=nfd)
    res = solve_flexible_trade_equilibrium(calib, subsistence_shares={"S01": 0.3}, rho_va=0.8, tau=0.1,
                                           tol=1e-8)
    assert res.converged
    assert np.max(np.abs(res.metadata["physical_residuals"])) <= 1e-8


def test_condensed_solver_matches_core_at_defaults(calib_5x4):
    core = solve_trade_equilibrium(calib_5x4, tau=0.1, tol=1e-9)
    flex = solve_flexible_trade_equilibrium(calib_5x4, tau=0.1, method="quasi_condensed", tol=1e-9)
    assert core.converged and flex.converged
    np.testing.assert_allclose(flex.x_sol, core.x_sol, rtol=0, atol=1e-7)
    np.testing.assert_allclose(flex.cpi, core.cpi, rtol=1e-8)
    res = compute_equilibrium_residuals(flex.x_sol, calib_5x4, tau=flex.metadata["intermediate_tariff_multipliers"],
                                        tau_fd=flex.metadata["final_tariff_multipliers"])
    assert np.max(np.abs(res)) <= 1e-9


def test_default_configuration_routes_to_core(calib_5x4):
    res = solve_flexible_trade_equilibrium(calib_5x4, tau=0.1)
    core = solve_trade_equilibrium(calib_5x4, tau=0.1)
    assert res.metadata["routing"] == "default"
    np.testing.assert_array_equal(res.x_sol, core.x_sol)
    np.testing.assert_allclose(res.terms_of_trade, core.terms_of_trade, rtol=1e-12)


def test_flexible_capacity_needs_sector(calib_5x4):
    with pytest.raises(TypeError, match="capacity_sector"):
        solve_flexible_trade_equilibrium(calib_5x4, capacity_margins=0.1, capacity_country="C01")


def test_flexible_capacity_needs_country(calib_5x4):
    with pytest.raises(TypeError, match="capacity_country"):
        solve_flexible_trade_equilibrium(calib_5x4, capacity_margins={"S02": 0.1})


def test_flexible_capacity_by_code_solves(calib_5x4):
    res = solve_flexible_trade_equilibrium(calib_5x4, capacity_margins=0.05, capacity_country="C03",
                                           capacity_sector="S02", tau=0.1, tol=1e-8)
    assert res.converged
    pairs = solve_flexible_trade_equilibrium(calib_5x4, capacity_margins={("C03", "S02"): 0.05}, tau=0.1,
                                             tol=1e-8)
    np.testing.assert_allclose(pairs.x_sol, res.x_sol, rtol=0, atol=1e-10)
    with pytest.raises(KeyError, match="C99"):
        solve_flexible_trade_equilibrium(calib_5x4, capacity_margins=0.05, capacity_country="C99",
                                         capacity_sector="S02")


def test_fiscal_recycling_needs_country_then_is_not_implemented(calib_5x4):
    with pytest.raises(TypeError, match="recycle_country"):
        solve_flexible_trade_equilibrium(calib_5x4, rho_va=0.7, fiscal_closure="capital_tax")
    with pytest.raises(NotImplementedError, match="lump-sum"):
        solve_flexible_trade_equilibrium(calib_5x4, rho_va=0.7, fiscal_closure="capital_tax",
                                         recycle_country="C01")


def test_active_settings_on_core_method_raise(calib_5x4):
    with pytest.raises(ValueError, match="quasi_condensed"):
        solve_flexible_trade_equilibrium(calib_5x4, rho_va=0.7, method="newton")


def test_removed_options_are_rejected(calib_5x4):
    with pytest.raises(TypeError, match="allow_legacy_fallback"):
        solve_flexible_trade_equilibrium(calib_5x4, rho_va=0.7, allow_legacy_fallback=True)
    with pytest.raises(ValueError, match="legacy accounting was removed"):
        solve_flexible_trade_equilibrium(calib_5x4, accounting="legacy")


def test_result_without_calibration_cannot_unpack(calib_5x4):
    res = solve_flexible_trade_equilibrium(calib_5x4, rho_va=0.7, tau=0.1)
    from puremacro.trade.flexible import FlexibleTradeEquilibriumResult

    bare = FlexibleTradeEquilibriumResult(x_sol=res.x_sol)
    with pytest.raises(ValueError, match="calib"):
        _ = bare.p_sol


def test_welfare_decomposition_uses_own_flows(calib_5x4):
    base = solve_flexible_trade_equilibrium(calib_5x4, rho_va=0.7, tol=1e-9)
    shock = solve_flexible_trade_equilibrium(calib_5x4, rho_va=0.7, tau=0.2, tol=1e-9)
    table = shock.welfare_decomposition(base)
    np.testing.assert_allclose(table["EV"], table["terms_of_trade"] + table["efficiency"], rtol=1e-12)
    # The terms-of-trade term uses the results' own (tariff-inclusive) terms of trade.
    expected = (shock.terms_of_trade - base.terms_of_trade) * 0.5 * (shock.exports + shock.imports)
    np.testing.assert_allclose(table["terms_of_trade"], expected, rtol=1e-12)
    # Without subsistence EV is the exact fixed-basket equivalent variation.
    hb0, hb1 = base.metadata["household_budget"], shock.metadata["household_budget"]
    assert np.all(np.isfinite(table["EV"])) and table.shape[0] == 5
    assert not np.allclose(hb0, hb1)
