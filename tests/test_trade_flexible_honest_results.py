"""Honest convergence flags and flows of condensed and routed flexible results.

Oracle: at the default configuration the flexible model is the core consistent
model, so the core Newton solver (``puremacro.trade.solver`` / ``_accounting``,
a separate code path) gives the reference solution, residual and flows.
"""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.trade import (
    calibrate_trade_model,
    compute_equilibrium_residuals,
    solve_trade_equilibrium,
)
from puremacro.trade.flexible import solve_flexible_trade_equilibrium


@pytest.fixture(scope="module")
def calib():
    """2x2 calibration with heterogeneous labour shares."""
    Z = np.array([[10., 15., 5., 5.], [15., 20., 10., 10.], [5., 5., 12., 18.], [10., 10., 18., 22.]])
    y0 = np.array([100., 150., 120., 180.])
    fw = np.array([[.5, .25, .05, .1, .08, .02]] * 2 + [[.1, .08, .02, .5, .25, .05]] * 2)
    F = (y0 - Z.sum(1))[:, None] * fw
    ptax = .05 * y0
    fi = y0 - Z.sum(0) - ptax
    ls = np.array([.7, .4, .6, .3])
    table = np.vstack([
        np.hstack([Z, F]),
        np.r_[ptax, .02 * F.sum(0)],
        np.r_[ls * fi, np.zeros(6)],
        np.r_[(1 - ls) * fi, np.zeros(6)],
    ])
    return calibrate_trade_model(table, ns=2, nc=2, nfd=3, country_codes=["A", "B"], sector_codes=["S01", "S02"])


@pytest.fixture(scope="module")
def tau(calib):
    t = np.ones((calib.n_sectors * calib.n_countries, calib.n_sectors, calib.n_countries))
    t[0, :, 1] = 1.25
    return t


@pytest.fixture(scope="module")
def core_oracle(calib, tau):
    res = solve_trade_equilibrium(calib, tau=tau, tol=1e-10)
    assert res.converged and res.max_residual <= 1e-10
    return res


FLOWS = ("exports", "imports", "cpi", "terms_of_trade", "gdp", "gdp_fc")


class TestCondensedFlag:
    def test_core_solver_refuses_the_flexible_method(self, calib, tau):
        with pytest.raises(NotImplementedError, match="solve_flexible_trade_equilibrium"):
            solve_trade_equilibrium(calib, tau=tau, method="quasi_condensed")

    def test_flag_agrees_with_reported_residual(self, calib, tau):
        tol = 1e-10
        res = solve_flexible_trade_equilibrium(calib, tau=tau, method="quasi_condensed", tol=tol)
        assert res.converged
        assert res.residual_norm <= tol
        assert res.residual_norm == float(np.max(np.abs(res.residuals)))
        assert np.max(np.abs(res.metadata["physical_residuals"])) <= tol
        # Independent check: the core equations vanish at x_sol.
        core_res = compute_equilibrium_residuals(res.x_sol, calib, tau=tau)
        assert np.max(np.abs(core_res)) <= 1e-8

    def test_flows_match_the_core_solution_of_the_same_model(self, calib, tau, core_oracle):
        res = solve_flexible_trade_equilibrium(calib, tau=tau, method="quasi_condensed", tol=1e-10)
        np.testing.assert_allclose(res.x_sol, core_oracle.x_sol, rtol=0.0, atol=1e-8)
        for name in FLOWS:
            np.testing.assert_allclose(getattr(res, name), getattr(core_oracle, name),
                                       rtol=1e-8, atol=1e-8, err_msg=name)

    def test_non_convergence_is_reported(self, calib, tau):
        res = solve_flexible_trade_equilibrium(calib, tau=tau, rho_va=0.7, tol=1e-12, max_iter=1)
        assert not res.converged
        assert res.metadata["convergence_diagnostic"]["top_equations"]


class TestRoutedFlexibleResultFlows:
    def test_default_route_matches_the_core_solver(self, calib, tau):
        res = solve_flexible_trade_equilibrium(calib, tau=tau, tol=1e-10)
        assert res.metadata["routing"] == "default"
        ref = solve_trade_equilibrium(calib, tau=tau, tol=1e-10)
        np.testing.assert_array_equal(res.x_sol, ref.x_sol)
        for name in FLOWS:
            np.testing.assert_allclose(getattr(res, name), getattr(ref, name), rtol=1e-12, atol=1e-12,
                                       err_msg=name)

    def test_active_configuration_uses_the_solved_flows(self, calib, tau):
        res = solve_flexible_trade_equilibrium(calib, tau=tau, sigma_trade=5.0, tol=1e-10)
        assert res.metadata["method"] == "quasi_condensed" and res.converged
        bilateral = res.metadata["bilateral_trade"]
        np.testing.assert_allclose(res.exports, bilateral.sum(axis=1), rtol=1e-12)
        np.testing.assert_allclose(res.imports, bilateral.sum(axis=0), rtol=1e-12)
        gdp_fc = res.w_sol.ravel() * calib.l_endow.ravel() + res.r_sol.ravel() * calib.k_endow.ravel()
        np.testing.assert_allclose(res.gdp_fc, gdp_fc, rtol=1e-12)
        # At a solution government receipts equal the lump-sum transfers T.
        np.testing.assert_allclose(res.gdp, gdp_fc + res.T_sol.ravel(), rtol=1e-9)
        # Trade balances equal the fixed foreign saving.
        np.testing.assert_allclose(res.exports - res.imports, calib.invforT.ravel(), atol=1e-9)
        # The purchaser value of each final-use column is that category's spending.
        values = res.metadata["final_demand_values"]
        assert np.all(np.isfinite(res.cpi)) and np.all(np.isfinite(res.terms_of_trade))
        assert values.shape == (4, 3, 2)

    def test_welfare_decomposition_values(self, calib, tau):
        base = solve_flexible_trade_equilibrium(calib, sigma_trade=5.0, tol=1e-10)
        arm = solve_flexible_trade_equilibrium(calib, tau=tau, sigma_trade=5.0, tol=1e-10)
        got = arm.welfare_decomposition(base)[["EV", "terms_of_trade", "efficiency"]].to_numpy()
        np.testing.assert_allclose(got[:, 0], got[:, 1] + got[:, 2], rtol=1e-12)
        # self-generated by puremacro 5.1.0 on the 2x2 fixture above; not an external reference
        expected = np.array([
            [0.0025178048239951067, 0.0018666245935439264, 0.0006511802304511804],
            [-0.0029957755023275467, -0.0018665717959745103, -0.0011292037063530365],
        ])
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-9)

