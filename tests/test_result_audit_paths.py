"""Independent path identities and perturbations for transition result audits."""
from copy import deepcopy
from dataclasses import replace
import socket

import numpy as np
import pytest

from puremacro.dsge import load_mod
from puremacro.dsge.perfect_foresight import solve_perfect_foresight
from puremacro.models import hank_sequence_space as hank
from puremacro.validation import PathAuditContext, audit_result


def rows(report):
    return {row.check: row for row in report.records}


@pytest.fixture(scope="module")
def editable_hank_paths():
    from puremacro.dsge.hank import load_hank_mod

    source = """
    var Y C r pi i; varexo eps;
    parameters beta gamma r_ss phi_pi kappa;
    beta=.985; gamma=1; r_ss=.01; phi_pi=1.5; kappa=.1;
    hetagent_block; model=one_asset_hank; n_a=25; a_max=30; end;
    model(linear);
    Y=C; pi=beta*pi(+1)+kappa*Y; i=phi_pi*pi+eps; r=i-pi(+1);
    end;
    """
    m = load_hank_mod(source)
    kwargs = dict(horizon=6, magnitude=-.001, shock="eps", closure="model",
                  household_inputs={"r": "r", "Y": "Y"}, household_outputs={"C": "C"})
    return m.simulate(**kwargs), m.simulate(**kwargs, nonlinear=True, closure_tol=1e-10)


@pytest.mark.parametrize("which", [0, 1])
def test_editable_hank_recomputes_aggregate_paths_and_names_missing_micro_evidence(editable_hank_paths, which):
    result = editable_hank_paths[which]
    checked = rows(audit_result(result))
    assert checked["aggregate_equations"].status == "passed"
    assert checked["household_consistency"].status == "passed"
    assert checked["household_budget"].status == "not_checked"
    assert checked["household_euler_kkt"].status == "not_checked"
    assert checked["distribution_transport"].status == "not_checked"
    bad = replace(result, transition_paths=result.transition_paths.copy())
    bad.transition_paths.loc[1, "pi"] += .02
    check = rows(audit_result(bad))["aggregate_equations"]
    assert check.status == "failed"
    # Taylor-rule error is 1.5*.02, larger than NKPC/lead errors.
    assert check.value == pytest.approx(.03, abs=1e-11)


def test_editable_nonlinear_hank_changed_inputs_invalidate_retained_response(editable_hank_paths):
    result = editable_hank_paths[1]
    bad = replace(result, transition_paths=result.transition_paths.copy())
    bad.transition_paths.loc[0, "r"] += .01
    checked = rows(audit_result(bad))
    assert checked["household_input_context:r"].status == "failed"
    assert checked["household_consistency"].status == "not_checked"
    # Changing output alone still allows the retained response comparison.
    bad = replace(result, transition_paths=result.transition_paths.copy())
    bad.transition_paths.loc[0, "C"] += .03
    checked = rows(audit_result(bad))
    assert checked["household_consistency"].status == "failed"
    assert checked["household_consistency"].value == pytest.approx(.03, abs=1e-11)
    # Retention is an exact conditioning identity, not a residual tolerance:
    # a small price perturbation can have a large response in a sensitive model.
    tiny = replace(result, transition_paths=result.transition_paths.copy())
    tiny.transition_paths.loc[0, "r"] += 1e-12
    checked = rows(audit_result(tiny))
    assert checked["household_input_context:r"].status == "passed"
    assert checked["household_input_identity:r"].status == "failed"
    assert checked["household_consistency"].status == "not_checked"


def test_editable_hank_shock_mass_and_context_perturbations(editable_hank_paths):
    result = editable_hank_paths[0]
    bad = replace(result, transition_paths=result.transition_paths.copy(),
                  asset_distribution=result.asset_distribution * 1.1)
    bad.transition_paths.loc[0, "eps"] += .02
    checked = rows(audit_result(bad))
    assert checked["shock_context:eps"].status == "failed"
    assert checked["asset_distribution:mass"].status == "failed"
    assert rows(audit_result(replace(result, _audit_context=None)))["hank_model_context"].status == "not_checked"
    assert rows(audit_result(replace(result, horizon=7)))["hank_model_equations"].status == "failed"


def test_editable_hank_audit_never_reenters_household_solver(editable_hank_paths, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("hidden household solve")
    monkeypatch.setattr(hank, "_household_transition", forbidden)
    monkeypatch.setattr(hank, "_egm_step", forbidden)
    for result in editable_hank_paths:
        assert rows(audit_result(result))["household_consistency"].status == "passed"


@pytest.mark.mechanism_control
def test_editable_hank_solver_blocker_control(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("hidden household solve")
    monkeypatch.setattr(hank, "_household_transition", forbidden)
    monkeypatch.setattr(hank, "_egm_step", forbidden)
    with pytest.raises(AssertionError, match="hidden household"):
        hank._household_transition(None, None, None)
    with pytest.raises(AssertionError, match="hidden household"):
        hank._egm_step(None, None, None)


@pytest.fixture(scope="module")
def ar_path():
    def equations(lead, now, lag, shock):
        return now - .5 * lag - shock
    return solve_perfect_foresight(equations, y_init=np.array([0.]), y_ss=np.array([0.]),
                                  shocks=np.array([1., 0., 0., 0.]), periods=4)


def test_pf_recomputes_actual_path_against_analytic_ar1(ar_path):
    np.testing.assert_allclose(ar_path.path.iloc[:, 0], [.5**t for t in range(4)], atol=1e-13)
    report = audit_result(ar_path)
    assert rows(report)["path_equations"].status == "passed"
    assert rows(report)["terminal_distance"].status == "not_checked"
    assert rows(report)["terminal_distance"].value == .125
    bad_path = ar_path.path.copy()
    bad_path.iloc[1, 0] += .1
    bad = replace(ar_path, path=bad_path, residual_norm=0., converged=True)
    check = rows(audit_result(bad))["path_equations"]
    assert check.status == "failed"
    assert check.value == pytest.approx(.1)
    assert check.location == (1, 0)
    assert check.details["worst_coordinates"]["period"] == 2


def test_pf_context_missing_and_explicit_boundaries(ar_path):
    legacy = replace(ar_path, _audit_context=None)
    assert rows(audit_result(legacy))["path_equations"].status == "not_checked"
    context = PathAuditContext(lambda p, c, l, e: c - .5*l - e,
                               [0.], [0.], [1., 0., 0., 0.])
    assert rows(audit_result(legacy, model=context))["path_equations"].status == "passed"
    bad = replace(context, y_init=[1.])
    assert rows(audit_result(legacy, model=bad))["path_equations"].value == pytest.approx(.5)
    assert rows(audit_result(legacy, model=replace(context, mode="surprise")))["path_equations"].status == "unsupported"
    assert rows(audit_result(legacy, model=object()))["path_context"].status == "unsupported"


@pytest.mark.parametrize("context_change", [
    {"exogenous_path": [1., 0.]}, {"y_end": [np.nan]}, {"y_init": [0., 1.]},
    {"variable_names": ("wrong",)}, {"equations": lambda *a: [0., 0.]},
])
def test_pf_invalid_context_never_passes(ar_path, context_change):
    context = PathAuditContext(lambda p, c, l, e: c - .5*l - e,
                               [0.], [0.], [1., 0., 0., 0.])
    assert rows(audit_result(ar_path, model=replace(context, **context_change)))["path_equations"].status == "failed"


def test_pf_fixed_steady_state_and_zero_shock_dimension():
    source = """var y x; varexo e; parameters rho; rho=.5;
    model; y=.5*y(-1)+e; x=STEADY_STATE(y); end;
    initval; y=0; x=0; end; steady;
    endval; e=.5; y=1; x=1; end; steady;
    """
    result = load_mod(source).solve_perfect_foresight(periods=5)
    np.testing.assert_allclose(result.path["y"], 1.-.5**np.arange(1,6), atol=1e-12)
    np.testing.assert_allclose(result.path["x"], 1., atol=1e-12)
    assert rows(audit_result(result))["path_equations"].status == "passed"
    zero = solve_perfect_foresight(lambda p,c,l,e:c-.5*l, y_init=np.array([1.]),
                                  y_ss=np.array([0.]), shocks=np.zeros((3,0)), periods=3)
    assert rows(audit_result(zero))["path_equations"].status == "passed"


def test_explicit_callable_gets_fixed_terminal_reference(ar_path):
    fixed = []
    def equations(p, c, l, e, *, steady_state):
        fixed.append(steady_state.copy())
        return c - .5*l - e + steady_state
    context = PathAuditContext(equations, [0.], [0.], [1.,0.,0.,0.])
    assert rows(audit_result(ar_path, model=context))["path_equations"].status == "passed"
    assert len(fixed) == 4
    np.testing.assert_array_equal(fixed, np.zeros((4,1)))


@pytest.fixture(scope="module")
def nonlinear_hank():
    ss = hank.solve_hank_sequence_space(T=8, n_a=25, a_max=50)
    result = hank.solve_nonlinear_transition(ss, shock_seq=[.001], horizon=8, tol=1e-10,
                                            phi_pi=1.8, kappa=.12, retain_audit_paths=True)
    assert result.converged
    return result


def test_hank_equations_and_effective_parameters(nonlinear_hank):
    report = audit_result(nonlinear_hank)
    checks = rows(report)
    for name in ("goods_market", "nkpc", "policy_rule", "return_timing", "household_budget",
                 "consumption_aggregation", "household_goods_market", "distribution_transport",
                 "path_mass_error", "asset_lottery_accounting"):
        assert checks[name].status == "passed", name
    assert nonlinear_hank._audit_context["phi_pi"] == 1.8
    assert nonlinear_hank.steady_state_model.phi_pi == 1.5
    assert checks["household_euler_kkt"].value > nonlinear_hank.tol
    assert checks["upper_boundary_mass"].status == "failed"  # deliberately small asset domain
    context = nonlinear_hank._audit_context
    c = context["paths"]["consumption"]
    D = context["paths"]["distributions"]
    # Independent aggregation is sensitive to policies and the entering-date mass.
    manual = np.einsum("tas,tas->t", c, D[:-1])
    np.testing.assert_allclose(manual, context["C_ss"] + nonlinear_hank.irf_consumption_nonlinear, atol=1e-14)
    assert report.to_json() == audit_result(nonlinear_hank).to_json()


@pytest.mark.parametrize("field,check", [
    ("irf_consumption_nonlinear", "goods_market"),
    ("irf_rate_nonlinear", "policy_rule"),
    ("irf_inflation_nonlinear", "nkpc"),
    ("nonlinear_path", "output_consistency:nonlinear_path"),
])
def test_hank_aggregate_path_tampering(nonlinear_hank, field, check):
    values = getattr(nonlinear_hank, field).copy()
    values[2] += .02
    bad = replace(nonlinear_hank, **{field: values}, residuals=np.zeros(8), converged=True)
    assert rows(audit_result(bad))[check].status == "failed"


@pytest.mark.parametrize("path,check", [
    ("consumption", "household_budget"), ("assets", "household_budget"),
    ("distributions", "distribution_transport"), ("realized_returns", "return_timing"),
    ("gross_wages", "wage_rule"),
])
def test_hank_retained_household_tampering(nonlinear_hank, path, check):
    context = deepcopy(nonlinear_hank._audit_context)
    value = context["paths"][path]
    value.flat[value.size//2] += .01
    assert rows(audit_result(replace(nonlinear_hank, _audit_context=context)))[check].status == "failed"


def test_hank_zero_mass_perturbation_and_negative_mass(nonlinear_hank):
    context = deepcopy(nonlinear_hank._audit_context)
    D = context["paths"]["distributions"]
    D[3,0,0] += .01
    D[3,1,0] -= .01
    checks = rows(audit_result(replace(nonlinear_hank, _audit_context=context)))
    assert checks["path_mass_error"].status == "passed"
    assert checks["distribution_transport"].status == "failed"
    assert checks["path_negative_mass"].status == "failed"


def test_hank_retention_is_opt_in_and_preserves_paths(nonlinear_hank):
    retained = nonlinear_hank
    compact = hank.solve_nonlinear_transition(retained.steady_state_model, shock_seq=[.001], horizon=8,
                                             tol=1e-10, phi_pi=1.8, kappa=.12)
    assert compact._audit_context["paths"] is None
    for name in ("U", "residuals", "irf_consumption_nonlinear", "irf_rate_nonlinear"):
        np.testing.assert_array_equal(getattr(compact, name), getattr(retained, name))
    checks = rows(audit_result(compact))
    assert checks["goods_market"].status == "passed"
    assert checks["household_budget"].status == "not_checked"
    assert checks["distribution_transport"].status == "not_checked"
    legacy = rows(audit_result(replace(compact, _audit_context=None)))
    assert legacy["ge_parameters"].status == "not_checked"


@pytest.mark.parametrize("shock_var,shock", [("G",.001),("r",0.)])
def test_hank_fiscal_and_zero_shock_paths(nonlinear_hank, shock_var, shock):
    result = hank.solve_nonlinear_transition(nonlinear_hank.steady_state_model, shock_seq=[shock],
                                            shock_var=shock_var, horizon=4, tol=1e-10,
                                            retain_audit_paths=True)
    checks = rows(audit_result(result))
    for check in ("goods_market", "household_goods_market", "household_budget", "distribution_transport"):
        assert checks[check].status == "passed"
    if not shock:
        np.testing.assert_allclose(result.U, 0., atol=1e-10)


def test_hank_invalid_retained_context_never_passes(nonlinear_hank):
    context = deepcopy(nonlinear_hank._audit_context)
    context["household"]["asset_grid"] = context["household"]["asset_grid"][::-1]
    assert rows(audit_result(nonlinear_hank, model=context))["hank_path_context"].status == "failed"
    context = deepcopy(nonlinear_hank._audit_context)
    context["paths"]["terminal_consumption"][0,0] = -1.
    checks = rows(audit_result(nonlinear_hank, model=context))
    assert checks["positive_terminal_consumption"].status == "failed"
    assert checks["household_euler_kkt"].status == "not_checked"


def test_hank_euler_expectation_against_independent_interpolator(nonlinear_hank):
    from scipy.interpolate import interp1d
    context = nonlinear_hank._audit_context
    p, h = context["paths"], context["household"]
    t, i, j = 3, 9, 1
    expectation = sum(h["transition"][j,k] * float(interp1d(
        h["asset_grid"], (1+p["realized_returns"][t+1]) * p["consumption"][t+1,:,k] ** (-h["gamma"])
    )(p["assets"][t,i,j])) for k in range(len(h["productivity"])))
    gradient = context["beta"] * expectation / p["consumption"][t,i,j] ** (-h["gamma"]) - 1
    assert h["asset_grid"][0] < p["assets"][t,i,j] < h["asset_grid"][-1]
    assert audit_result(nonlinear_hank).maps["household_euler_kkt"][t,i,j] == pytest.approx(gradient, abs=2e-15)


def test_path_audits_do_not_resolve_or_use_network(ar_path, nonlinear_hank, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected solve/network")
    monkeypatch.setattr(hank, "_egm_step", forbidden)
    monkeypatch.setattr(hank, "_household_transition", forbidden)
    monkeypatch.setattr(hank, "solve_nonlinear_transition", forbidden)
    monkeypatch.setattr(np.linalg, "solve", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    before = nonlinear_hank._audit_context["paths"]["distributions"].copy()
    assert rows(audit_result(ar_path))["path_equations"].status == "passed"
    assert rows(audit_result(nonlinear_hank))["household_budget"].status == "passed"
    np.testing.assert_array_equal(before, nonlinear_hank._audit_context["paths"]["distributions"])


@pytest.mark.mechanism_control
def test_solve_network_blocker_positive_control(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected solve/network")
    monkeypatch.setattr(hank, "_egm_step", forbidden)
    monkeypatch.setattr(np.linalg, "solve", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    for function, args in ((hank._egm_step, ()), (np.linalg.solve, (np.eye(1), np.ones(1))),
                            (socket.create_connection, (("example.invalid",443),))):
        with pytest.raises(AssertionError, match="unexpected solve/network"):
            function(*args)
