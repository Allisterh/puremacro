"""Tests for Dynare .mod parser extensions: dated shocks in '#' locals, native
steady_state_model compilation with nonlinear/homotopy fallback, native command
block execution, and de-monkeypatched LinearModel methods.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.dsge import (
    LinearModel,
    load_mod,
    parse_mod,
)
from puremacro.dsge._parser import parse_mod_to_dag
from puremacro.dsge._results import DynareCommandResults
from puremacro.dsge.dynare import (
    _expand_multiperiod_leads_lags,
    _shift_mod_expr,
    _shift_mod_expr_leads,
)


# ===========================================================================
# 1. Dated Exogenous Shocks in '#' Model-Local Variables & Equations
# ===========================================================================


def test_dated_exogenous_shocks_in_model_locals_and_equations():
    """Dated exogenous innovations e(-1), e(-2) inside '#' local expressions
    and shifted locals #z(-1) must expand into auxiliary state variables and
    produce identical IRFs and theoretical moments to a manually expanded model.
    """
    mod_with_dated_exo = """
    var y x;
    varexo e;
    parameters rho theta1 theta2 alpha;
    rho = 0.75;
    theta1 = 0.40;
    theta2 = 0.15;
    alpha = 0.50;

    model;
    #ma_shock = e + theta1 * e(-1) + theta2 * e(-2);
    #signal = x + 0.3 * e;
    y = rho * y(-1) + ma_shock;
    x = alpha * signal(-1) + e;
    end;

    shocks;
    var e; stderr 0.02;
    end;
    """

    mod_manual_aux = """
    var y x aux_e1 aux_e2;
    varexo e;
    parameters rho theta1 theta2 alpha;
    rho = 0.75;
    theta1 = 0.40;
    theta2 = 0.15;
    alpha = 0.50;

    model;
    y = rho * y(-1) + e + theta1 * aux_e1(-1) + theta2 * aux_e2(-1);
    x = alpha * (x(-1) + 0.3 * aux_e1(-1)) + e;
    aux_e1 = e;
    aux_e2 = aux_e1(-1);
    end;

    shocks;
    var e; stderr 0.02;
    end;
    """

    m_dated = load_mod(mod_with_dated_exo)
    m_manual = load_mod(mod_manual_aux)

    assert "AUX_EXO_LAG_e_1" in m_dated.variables
    assert "AUX_EXO_LAG_e_2" in m_dated.variables

    irf_dated = m_dated.irf("e", horizon=25)
    irf_manual = m_manual.irf("e", horizon=25)

    np.testing.assert_allclose(irf_dated["y"].to_numpy(), irf_manual["y"].to_numpy(), atol=1e-12)
    np.testing.assert_allclose(irf_dated["x"].to_numpy(), irf_manual["x"].to_numpy(), atol=1e-12)

    mom_dated = m_dated.theoretical_moments()
    mom_manual = m_manual.theoretical_moments()
    np.testing.assert_allclose(
        mom_dated.moments.loc[["y", "x"], "Variance"].to_numpy(),
        mom_manual.moments.loc[["y", "x"], "Variance"].to_numpy(),
        rtol=1e-10,
        atol=1e-12,
    )


def test_dated_exogenous_lead_in_model_locals_and_shift_helpers():
    """Exogenous leads e(+1) in '#' locals and regex helper _shift_mod_expr_leads."""
    shifted = _shift_mod_expr_leads("c + 0.5 * e - 0.2 * e(-1)", -1, ["c"], ["e"])
    assert "c(-1)" in shifted
    assert "e(-1)" in shifted
    assert "e(-2)" in shifted

    eqs, vars_out, ss_out, _ = _expand_multiperiod_leads_lags(
        ["y = 0.8 * y(-1) + e + 0.5 * e(-1) + 0.2 * e(+1)"],
        ["y"],
        steady_state={"y": 0.0},
        shocks=["e"],
    )
    assert "AUX_EXO_LAG_e_1" in vars_out
    assert "AUX_EXO_LEAD_e_1" in vars_out
    assert ss_out["AUX_EXO_LAG_e_1"] == 0.0
    assert ss_out["AUX_EXO_LEAD_e_1"] == 0.0

    mod_lead = """
    var y;
    varexo e;
    parameters rho;
    rho = 0.6;
    model;
    #news = e + 0.25 * e(+1);
    y = rho * y(-1) + news;
    end;
    shocks;
    var e; stderr 0.01;
    end;
    """
    dag = parse_mod_to_dag(mod_lead)
    assert "AUX_EXO_LEAD_e_1" in dag.variables


# ===========================================================================
# 2. De-Monkeypatched LinearModel Methods
# ===========================================================================


def test_linear_model_methods_defined_natively_in_build_module():
    """Perturbation and simulation methods must live directly on LinearModel in
    puremacro.dsge.build rather than being monkeypatched at import time.
    """
    for method_name in (
        "steady",
        "solve",
        "solve_perturbation",
        "solve_second_order",
        "solve_third_order",
        "stoch_simul",
        "ramsey_policy",
        "execute_mod_commands",
        "model_info",
        "write_latex_dynamic_model",
        "perfect_foresight",
        "simulate_surprise_shocks",
    ):
        attr = getattr(LinearModel, method_name)
        assert attr.__module__ == "puremacro.dsge.build", (
            f"LinearModel.{method_name} was defined in {attr.__module__}, expected puremacro.dsge.build"
        )


# ===========================================================================
# 3. Native steady_state_model Compilation & Homotopy / Root-Finding Fallback
# ===========================================================================


_RBC_STEADY_STATE_MOD = """
var y c k i a;
varexo e;
parameters alpha beta delta gamma rho;

alpha = 0.33;
beta = 0.99;
delta = 0.025;
gamma = 2.0;
rho = 0.90;

model;
#rk = alpha * exp(a(+1)) * k^(alpha - 1);
c^(-gamma) = beta * c(+1)^(-gamma) * (rk + 1 - delta);
y = exp(a) * k(-1)^alpha;
k = y - c + (1 - delta) * k(-1);
i = y - c;
a = rho * a(-1) + e;
end;

steady_state_model;
rk = 1 / beta - 1 + delta;
k_y = alpha / rk;
k = (rk / alpha)^(1 / (alpha - 1));
y = k^alpha;
i = delta * k;
c = y - i;
a = 0;
end;

shocks;
var e; stderr 0.01;
end;
"""


def test_steady_state_model_compiled_evaluator_and_parameter_updates():
    """steady_state_model with intermediate local helper variables (rk, k_y)
    even when #rk is also a model-local variable in `model; ... end;` must
    compile into a fast callable _steady_state_model_fn and evaluate to machine precision.
    """
    m = load_mod(_RBC_STEADY_STATE_MOD)

    # Analytical check at baseline parameters
    alpha, beta, delta = 0.33, 0.99, 0.025
    rk_exact = 1.0 / beta - 1.0 + delta
    k_exact = (rk_exact / alpha) ** (1.0 / (alpha - 1.0))
    y_exact = k_exact ** alpha
    i_exact = delta * k_exact
    c_exact = y_exact - i_exact

    assert m.steady_state["k"] == pytest.approx(k_exact, rel=1e-13)
    assert m.steady_state["y"] == pytest.approx(y_exact, rel=1e-13)
    assert m.steady_state["c"] == pytest.approx(c_exact, rel=1e-13)
    assert m.steady_state["i"] == pytest.approx(i_exact, rel=1e-13)
    assert m.steady_state["a"] == pytest.approx(0.0, abs=1e-14)
    # Exact analytical steady state requires no nonlinear solver call
    assert m.steady_state_info is None

    # Direct O(1) callable _steady_state_model_fn
    ss_fn = getattr(m, "_steady_state_model_fn", None)
    assert callable(ss_fn)

    new_params = {"alpha": 0.36, "beta": 0.985, "delta": 0.03}
    rk_new = 1.0 / 0.985 - 1.0 + 0.03
    k_new = (rk_new / 0.36) ** (1.0 / (0.36 - 1.0))
    y_new = k_new ** 0.36
    i_new = 0.03 * k_new
    c_new = y_new - i_new

    ss_series = m.steady(params=new_params)
    assert ss_series["k"] == pytest.approx(k_new, rel=1e-13)
    assert ss_series["y"] == pytest.approx(y_new, rel=1e-13)
    assert ss_series["c"] == pytest.approx(c_new, rel=1e-13)

    # Re-solving model at new parameters updates steady state and decision rules in O(1)
    m_resolved = m.solve(params=new_params)
    assert m_resolved.steady_state["k"] == pytest.approx(k_new, rel=1e-13)
    assert m_resolved.steady_state_info is None


def test_steady_state_model_fallback_to_nonlinear_and_homotopy():
    """When steady_state_model provides only an approximate steady state (so
    dynamic residuals exceed tol), load_mod and model.steady() must automatically
    fall back to the nonlinear root-finding / homotopy solver.
    """
    mod_approx_ss = """
    var c k;
    varexo e;
    parameters A alpha beta delta;
    A = 50.0;
    alpha = 0.33;
    beta = 0.96;
    delta = 0.10;

    model;
    1 / c = beta * (1 / c(+1)) * (alpha * A * k^(alpha - 1) + 1 - delta) + e;
    c + k = A * k(-1)^alpha + (1 - delta) * k(-1);
    end;

    steady_state_model;
    // Intentionally evaluated at A=1 instead of A=50 so residuals > tol
    rk_approx = (1 / beta - 1 + delta) / alpha;
    k = rk_approx^(1 / (alpha - 1));
    c = k^alpha - delta * k;
    end;

    shocks;
    var e; stderr 0.01;
    end;
    """
    # Use homotopy fallback from A=1.0 to A=50.0 starting from the steady_state_model guess
    m = load_mod(
        mod_approx_ss,
        solve_algo="hybr",
        homotopy={"A": (1.0, 50.0)},
        homotopy_steps=5,
    )

    alpha, beta, delta, A = 0.33, 0.96, 0.10, 50.0
    rk_true = (1.0 / beta - 1.0 + delta) / (alpha * A)
    k_true = rk_true ** (1.0 / (alpha - 1.0))
    c_true = A * (k_true ** alpha) - delta * k_true

    assert m.steady_state["k"] == pytest.approx(k_true, rel=1e-6)
    assert m.steady_state["c"] == pytest.approx(c_true, rel=1e-6)
    assert m.steady_state_info is not None
    assert m.steady_state_info["converged"]

    # Also test model.steady() fallback with homotopy
    ss_hom = m.steady(
        solve_algo="hybr",
        homotopy={"A": (1.0, 50.0)},
        homotopy_steps=5,
    )
    assert ss_hom["k"] == pytest.approx(k_true, rel=1e-6)
    assert ss_hom["c"] == pytest.approx(c_true, rel=1e-6)


# ===========================================================================
# 4. Native Dynare Command Blocks (ramsey_policy, osr, calib_smoother,
#    shock_decomposition, histval, perfect_foresight_solver)
# ===========================================================================


def test_native_ramsey_policy_and_planner_objective_blocks():
    """Verify `planner_objective` and `ramsey_policy` blocks work on both full
    models (with instrument specified) and under-determined .mod files (where
    the policy equation is omitted).
    """
    nk_ramsey_full = """
    var pi x i;
    varexo u_c u_p;
    parameters beta kappa sigma phi_pi phi_x;
    beta = 0.99;
    kappa = 0.15;
    sigma = 1.0;
    phi_pi = 1.5;
    phi_x = 0.5;

    model;
    pi = 0.5 * beta * pi(+1) + 0.5 * pi(-1) + kappa * x + u_c;
    x = x(+1) - (1 / sigma) * (i - pi(+1)) + u_p;
    i = phi_pi * pi + phi_x * x;
    end;

    shocks;
    var u_c; stderr 0.01;
    var u_p; stderr 0.01;
    end;

    planner_objective pi^2 + 0.25 * x^2;
    ramsey_policy(planner_discount = 0.99, instruments = (i), irf = 20);
    """

    nk_ramsey_underdetermined = """
    var pi x i;
    varexo u_c u_p;
    parameters beta kappa sigma;
    beta = 0.99;
    kappa = 0.15;
    sigma = 1.0;

    model;
    pi = 0.5 * beta * pi(+1) + 0.5 * pi(-1) + kappa * x + u_c;
    x = x(+1) - (1 / sigma) * (i - pi(+1)) + u_p;
    end;

    shocks;
    var u_c; stderr 0.01;
    var u_p; stderr 0.01;
    end;

    planner_objective(pi^2 + 0.25 * x^2);
    ramsey_policy(planner_discount = beta);
    """

    m_full = load_mod(nk_ramsey_full)
    res_full = m_full.execute_mod_commands()
    assert isinstance(res_full, DynareCommandResults)
    assert res_full.ramsey is not None
    assert "ramsey_policy" in res_full.commands_executed
    assert len(res_full.ramsey.multipliers) == 2

    m_under = load_mod(nk_ramsey_underdetermined)
    res_under = m_under.execute_mod_commands()
    assert res_under.ramsey is not None
    assert len(res_under.ramsey.multipliers) == 2

    # IRFs under commitment from both specifications must match
    irf_full = res_full.ramsey.irf("u_c", horizon=20)
    irf_under = res_under.ramsey.irf("u_c", horizon=20)
    np.testing.assert_allclose(
        irf_full["pi"].to_numpy(), irf_under["pi"].to_numpy(), atol=1e-10
    )
    np.testing.assert_allclose(
        irf_full["x"].to_numpy(), irf_under["x"].to_numpy(), atol=1e-10
    )
    np.testing.assert_allclose(
        irf_full["i"].to_numpy(), irf_under["i"].to_numpy(), atol=1e-10
    )


def test_native_osr_blocks():
    """Verify `osr_params`, `osr_params_bounds`, `optim_weights`, and `osr` blocks."""
    nk_osr_mod = """
    var pi x i;
    varexo u_c u_p;
    parameters beta kappa sigma phi_pi phi_x;
    beta = 0.99;
    kappa = 0.15;
    sigma = 1.0;
    phi_pi = 1.2;
    phi_x = 0.1;

    model;
    pi = 0.5 * beta * pi(+1) + 0.5 * pi(-1) + kappa * x + u_c;
    x = x(+1) - (1 / sigma) * (i - pi(+1)) + u_p;
    i = phi_pi * pi + phi_x * x;
    end;

    shocks;
    var u_c; stderr 0.01;
    var u_p; stderr 0.01;
    end;

    optim_weights;
    pi 1.0;
    x 0.25;
    i 0.10;
    end;

    osr_params phi_pi phi_x;

    osr_params_bounds;
    phi_pi, 1.05, 3.0;
    phi_x, 0.0, 1.5;
    end;

    osr;
    """
    m = load_mod(nk_osr_mod)
    res = m.execute_mod_commands()
    assert res.osr is not None
    assert "osr" in res.commands_executed
    assert res.osr.converged
    assert res.osr.loss_opt <= res.osr.loss_initial + 1e-10
    assert 1.05 <= res.osr.optimal_params["phi_pi"] <= 3.0
    assert 0.0 <= res.osr.optimal_params["phi_x"] <= 1.5


def test_native_calib_smoother_and_shock_decomposition_blocks():
    """Verify `calib_smoother;` and `shock_decomposition;` execute against varobs data."""
    mod_smoother = """
    var y x;
    varexo e_y e_x;
    parameters rho_y rho_x;
    rho_y = 0.8;
    rho_x = 0.5;

    varobs y x;

    model;
    y = rho_y * y(-1) + x + e_y;
    x = rho_x * x(-1) + e_x;
    end;

    shocks;
    var e_y; stderr 0.02;
    var e_x; stderr 0.01;
    end;

    stoch_simul(order = 1, irf = 15);
    calib_smoother;
    shock_decomposition;
    """
    m = load_mod(mod_smoother)
    sim_df = m.simulate(periods=40, seed=42)[["y", "x"]]

    cmd_res = m.execute_mod_commands(data=sim_df)
    assert cmd_res.stoch_simul is not None
    assert cmd_res.smoother is not None
    assert cmd_res.calib_smoother is cmd_res.smoother
    assert cmd_res.shock_decomposition is not None
    assert set(cmd_res.commands_executed) == {
        "stoch_simul",
        "calib_smoother",
        "shock_decomposition",
    }
    assert cmd_res.smoother.states.shape == (40, 2)
    assert "y" in cmd_res.shock_decomposition.components


def test_native_histval_and_perfect_foresight_blocks():
    """Verify `histval;` sets initial conditions for simulation and perfect_foresight_solver."""
    mod_pf = """
    var k a;
    varexo e;
    parameters rho delta;
    rho = 0.7;
    delta = 0.1;

    model;
    k = (1 - delta) * k(-1) + a;
    a = rho * a(-1) + e;
    end;

    steady_state_model;
    k = 0;
    a = 0;
    end;

    histval;
    k(0) = 2.0;
    a(0) = 0.5;
    end;

    shocks;
    var e; stderr 0.01;
    end;

    perfect_foresight_setup(periods = 25);
    perfect_foresight_solver;
    """
    m = load_mod(mod_pf)
    assert m.histval == {"k": 2.0, "a": 0.5}

    # Zero-noise simulation with burn=0 starts from histval
    sim_zero = m.simulate(periods=5, sigma=0.0, burn=0)
    # Period 0 in simulate: k_0 = (1 - delta) * k_{-1} + a_0 where a_0 = rho * a_{-1}
    a_0 = 0.7 * 0.5
    k_0 = 0.9 * 2.0 + a_0
    assert sim_zero.loc[0, "a"] == pytest.approx(a_0, abs=1e-12)
    assert sim_zero.loc[0, "k"] == pytest.approx(k_0, abs=1e-12)

    cmd_res = m.execute_mod_commands()
    assert cmd_res.histval == {"k": 2.0, "a": 0.5}
    assert cmd_res.perfect_foresight is not None
    assert cmd_res.perfect_foresight.converged
    assert "histval" in cmd_res.commands_executed
    assert "perfect_foresight_solver" in cmd_res.commands_executed

    # Presentation methods on DynareCommandResults
    assert "Dynare .mod Command Execution Summary" in cmd_res.summary()
    assert isinstance(cmd_res.to_frame(), pd.DataFrame)
    assert "PerfectForesightResult" in cmd_res.to_markdown()
    assert r"\begin{tabular}" in cmd_res.to_latex()
    assert "#table(" in cmd_res.to_typst()
