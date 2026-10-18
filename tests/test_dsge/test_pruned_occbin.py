"""Tests for Pruned Higher-Order OccBin (solve_pruned_occbin).

Verifies:
1. Exact reduction to linear OccBin when sigma = 0 (< 1e-12 error).
2. Exact reduction to 2nd-order pruned perturbation (PrunedDSGESolution)
   when constraints never bind (< 1e-12 error).
3. Deterministic simulation with perfect foresight under 2nd-order pruned dynamics.
4. Binding regime episodes coupling piecewise-linear regimes with 2nd-order pruning.
5. Export from puremacro.dsge and OccBinResult diagnostics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.dsge import (
    OccBinConstraint,
    OccBinResult,
    PrunedDSGESolution,
    build_dynare,
    solve_occbin,
    solve_pruned_occbin,
)


@pytest.fixture(scope="module")
def nk_zlb_model():
    """Calibrated New Keynesian model with Taylor rule and Zero Lower Bound."""
    params = {
        "beta": 0.99,
        "sigma": 1.0,
        "kappa": 0.1,
        "phi_pi": 1.5,
        "phi_y": 0.125,
        "rho_g": 0.8,
        "r_ss": 0.01,
    }

    variables = ["y", "pi", "r", "g"]
    shocks = ["eps_r", "eps_g"]
    steady_state = {v: 0.0 for v in variables}

    def nk_ref(lead, curr, lag, shocks_v, p):
        return [
            curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.g,
            curr.pi - p.beta * lead.pi - p.kappa * curr.y,
            curr.r - p.phi_pi * curr.pi - p.phi_y * curr.y - shocks_v.eps_r,
            curr.g - p.rho_g * lag.g - shocks_v.eps_g,
        ]

    def nk_cons(lead, curr, lag, shocks_v, p):
        return [
            curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.g,
            curr.pi - p.beta * lead.pi - p.kappa * curr.y,
            curr.r - (-p.r_ss),
            curr.g - p.rho_g * lag.g - shocks_v.eps_g,
        ]

    ref_m = build_dynare(nk_ref, variables=variables, shocks=shocks, params=params, steady_state=steady_state)
    cons_m = build_dynare(
        nk_cons,
        variables=variables,
        shocks=shocks,
        params=params,
        steady_state=steady_state,
        check_steady_state=False,
        strict=False,
    )
    constraint = OccBinConstraint(variable="r", threshold=-params["r_ss"], operator="<")

    return {
        "ref_m": ref_m,
        "cons_m": cons_m,
        "constraint": constraint,
        "params": params,
        "variables": variables,
        "shocks": shocks,
    }


@pytest.fixture(scope="module")
def neoclassical_occbin_model():
    """Non-linear neoclassical growth model with genuine 2nd-order curvature and floor."""
    mod_ref = """
    var c k y;
    varexo eps_a;
    parameters beta alpha delta;
    beta = 0.96;
    alpha = 0.33;
    delta = 0.10;

    model;
      1/c = beta * (1/c(+1)) * (alpha * y(+1)/k + 1 - delta);
      y = k(-1)^alpha;
      k = y - c + (1 - delta)*k(-1) + eps_a;
    end;

    steady_state_model;
      k = ((1/beta - (1 - delta))/alpha)^(1/(alpha - 1));
      y = k^alpha;
      c = y - delta*k;
    end;
    """
    m_ref = build_dynare(mod_ref)
    k_ss = float(m_ref.steady_state["k"])

    # Floor at k_dev >= -0.1
    def cons_fn(lead, curr, lag, shocks_v, p):
        return [
            1 / curr.c - p.beta * (1 / lead.c) * (p.alpha * lead.y / curr.k + 1 - p.delta),
            curr.y - lag.k**p.alpha,
            curr.k - (k_ss - 0.1),
        ]

    constraint = OccBinConstraint(variable="k", threshold=-0.1, operator="<")
    return {
        "ref_m": m_ref,
        "cons_fn": cons_fn,
        "constraint": constraint,
        "k_ss": k_ss,
    }


def test_solve_pruned_occbin_exact_reduction_linear(nk_zlb_model):
    """Verify solve_pruned_occbin matches linear solve_occbin when sigma=0 or order=1 to < 1e-12."""
    ref_m = nk_zlb_model["ref_m"]
    cons_m = nk_zlb_model["cons_m"]
    constraint = nk_zlb_model["constraint"]

    horizon = 30
    # Shock triggering ZLB
    shocks = np.zeros((horizon, 2))
    shocks[0, 1] = -0.04

    # 1. Linear OccBin
    res_linear = solve_occbin(ref_m, cons_m, constraint, shocks, horizon=horizon)

    # 2. Pruned OccBin with sigma=0.0
    res_pruned_sig0 = solve_pruned_occbin(
        ref_m, cons_m, constraint, shocks, order=2, sigma=0.0, horizon=horizon
    )

    # 3. Pruned OccBin with order=1
    res_pruned_ord1 = solve_pruned_occbin(
        ref_m, cons_m, constraint, shocks, order=1, sigma=1.0, horizon=horizon
    )

    err_sig0 = np.max(np.abs(res_pruned_sig0.simulated_path.to_numpy() - res_linear.simulated_path.to_numpy()))
    err_ord1 = np.max(np.abs(res_pruned_ord1.simulated_path.to_numpy() - res_linear.simulated_path.to_numpy()))

    assert err_sig0 < 1e-12, f"Expected err_sig0 < 1e-12, got {err_sig0}"
    assert err_ord1 < 1e-12, f"Expected err_ord1 < 1e-12, got {err_ord1}"
    assert res_pruned_sig0.regimes == res_linear.regimes
    assert res_pruned_sig0.binding_periods == res_linear.binding_periods


def test_solve_pruned_occbin_exact_reduction_pruned_slack(neoclassical_occbin_model):
    """Verify solve_pruned_occbin matches PrunedDSGESolution when constraints never bind (< 1e-12)."""
    m_ref = neoclassical_occbin_model["ref_m"]
    cons_fn = neoclassical_occbin_model["cons_fn"]
    constraint = neoclassical_occbin_model["constraint"]

    horizon = 30
    # Positive capital innovation keeps k well above floor
    shocks = np.zeros((horizon, 1))
    shocks[0, 0] = 0.05

    # 1. Direct 2nd-order pruned perturbation
    sol2 = m_ref.solve(order=2)
    assert isinstance(sol2, PrunedDSGESolution)
    x1, x2, y1, y2 = sol2._pruned_path(shocks, sigma=1.0)

    variables = list(sol2.variables)
    states = list(sol2.states)
    controls = list(sol2.controls)
    s_idx = [variables.index(s) for s in states]
    c_idx = [variables.index(c) for c in controls]

    v_ref = np.zeros((horizon, len(variables)))
    v_ref[:, s_idx] = x1 + x2
    v_ref[:, c_idx] = y1 + y2

    # 2. Pruned OccBin
    res_pruned = solve_pruned_occbin(
        m_ref, cons_fn, constraint, shocks, order=2, sigma=1.0, horizon=horizon
    )

    # Verify constraint remained slack
    assert res_pruned.binding_periods == 0
    assert sum(res_pruned.regimes) == 0

    err_pruned = np.max(np.abs(res_pruned.simulated_path.to_numpy() - v_ref))
    assert err_pruned < 1e-12, f"Expected err_pruned < 1e-12, got {err_pruned}"


def test_solve_pruned_occbin_binding_episodes(nk_zlb_model):
    """Verify solve_pruned_occbin successfully captures binding regimes with 2nd-order dynamics."""
    ref_m = nk_zlb_model["ref_m"]
    cons_m = nk_zlb_model["cons_m"]
    constraint = nk_zlb_model["constraint"]

    horizon = 35
    shocks = np.zeros((horizon, 2))
    shocks[0, 1] = -0.05

    res = solve_pruned_occbin(ref_m, cons_m, constraint, shocks, order=2, sigma=1.0, horizon=horizon)

    assert isinstance(res, OccBinResult)
    assert res.binding_periods > 0
    assert 1 in res.regimes
    assert res.converged is True

    # At ZLB periods, r must match threshold within machine precision
    r_path = res.simulated_path["r"].to_numpy()
    for t, reg in enumerate(res.regimes):
        if reg == 1:
            np.testing.assert_allclose(r_path[t], -nk_zlb_model["params"]["r_ss"], atol=1e-12)


def test_solve_pruned_occbin_input_validation(nk_zlb_model):
    """Verify solve_pruned_occbin checks parameter validity."""
    ref_m = nk_zlb_model["ref_m"]
    cons_m = nk_zlb_model["cons_m"]
    constraint = nk_zlb_model["constraint"]

    with pytest.raises(ValueError, match="order"):
        solve_pruned_occbin(ref_m, cons_m, constraint, np.zeros(2), order=3)

    with pytest.raises(ValueError, match="horizon"):
        solve_pruned_occbin(ref_m, cons_m, constraint, np.zeros(2), horizon=0)
