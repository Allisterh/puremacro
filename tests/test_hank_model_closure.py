"""Editable aggregate equations must determine the actual HANK paths."""
from __future__ import annotations

import numpy as np
import pytest

from puremacro import dp
from puremacro.dsge.hank import load_hank_mod, solve_hank_bridge
from puremacro.models.hank_sequence_space import dp_sequence_jacobian


SOURCE = """
var Y C r pi i;
varexo eps_m eps_g;
parameters beta gamma r_ss phi_pi kappa;
beta=.985; gamma=1; r_ss=.01; phi_pi=1.5; kappa=.1;
hetagent_block; model=one_asset_hank; n_a=30; a_max=30; end;
model(linear);
Y=C+eps_g;
pi=beta*pi(+1)+kappa*Y;
i=phi_pi*pi+eps_m;
r=i-pi(+1);
end;
"""
MAPS = {"closure": "model", "household_inputs": {"r": "r", "Y": "Y"},
        "household_outputs": {"C": "C"}}


@pytest.fixture(scope="module")
def model():
    return load_hank_mod(SOURCE)


def test_editable_equations_reproduce_fixed_closure_and_report_scope(model):
    fixed = model.simulate(horizon=12)
    editable = model.simulate(horizon=12, **MAPS)
    for name in ("Y", "C", "r", "pi", "i"):
        np.testing.assert_allclose(editable.transition_paths[name], fixed.transition_paths[name], atol=2e-12)
    assert editable.closure == "model" and fixed.closure == "builtin"
    assert editable.diagnostics["rank"] == 60
    assert "not Blanchard-Kahn" in editable.diagnostics["rank_scope"]
    assert editable.diagnostics["boundary"] == "zero deviations outside horizon"
    assert editable.diagnostics["aggregate_residual_max_abs"] < 1e-14
    assert editable.diagnostics["household_residual_max_abs"] < 1e-14


def test_edited_rule_and_explicit_shocks_change_actual_paths(model):
    T = 9
    monetary = -.002 * .6 ** np.arange(T)
    fiscal = .001 * .4 ** np.arange(T)
    paths = {"eps_m": monetary, "eps_g": fiscal}
    baseline = model.simulate(horizon=T, shock_paths=paths, **MAPS)
    changed = solve_hank_bridge(SOURCE.replace("i=phi_pi*pi+eps_m", "i=.4*i(-1)+phi_pi*pi+eps_m"),
                               horizon=T, shock_paths=paths, **MAPS)
    out = changed.transition_paths
    assert np.max(np.abs(out.Y - baseline.transition_paths.Y)) > 1e-5
    np.testing.assert_allclose(out.Y - out.C, fiscal, atol=1e-14)
    np.testing.assert_allclose(out.i - .4 * np.r_[0, out.i[:-1]] - 1.5*out.pi, monetary, atol=1e-14)
    np.testing.assert_allclose(out.pi - .985*np.r_[out.pi[1:], 0] - .1*out.Y, 0, atol=1e-14)
    monetary[:] = 100
    np.testing.assert_allclose(out.eps_m, -.002*.6**np.arange(T))


def test_explicit_household_shock_mapping_and_zero_boundaries():
    source = SOURCE.replace("var Y C r pi i;", "var C q;").replace(
        "Y=C+eps_g;\npi=beta*pi(+1)+kappa*Y;\ni=phi_pi*pi+eps_m;\nr=i-pi(+1);",
        "q = C(+1) + eps_g(-1);")
    m = load_hank_mod(source)
    T = 5
    result = m.simulate(horizon=T, closure="model", household_inputs={"Y": "eps_m"},
                        household_outputs={"C": "C"}, shock_paths={"eps_m": np.ones(T)*.01, "eps_g": np.arange(T)*.02})
    out = result.transition_paths
    np.testing.assert_allclose(out.C, m.compute_jacobians(T)["J_C_Y"] @ np.full(T, .01))
    np.testing.assert_allclose(out.q, np.r_[out.C[1:], 0] + np.r_[0, out.eps_g[:-1]])


@pytest.mark.parametrize("kwargs,match", [
    ({"household_inputs": None}, "explicit non-empty"),
    ({"household_inputs": {"r": "r", "Y": "r"}}, "duplicate"),
    ({"household_outputs": {"C": "eps_m"}}, "undeclared"),
    ({"household_inputs": {"r": "typo"}}, "undeclared"),
    ({"household_inputs": {"bad": "r"}}, "unsupported built-in"),
    ({"shock_paths": {"typo": np.zeros(5)}}, "undeclared"),
    ({"shock_paths": {"eps_m": [np.nan]*5}}, "finite"),
    ({"shock_paths": {"eps_m": np.zeros(6)}}, "shape"),
    ({"shock": "typo"}, "not declared"),
    ({"closure_tol": 0}, "positive"),
    ({"horizon": 2.5}, "positive integer"),
])
def test_invalid_explicit_configuration(model, kwargs, match):
    with pytest.raises(ValueError, match=match):
        model.simulate(**{"horizon": 5, **MAPS, **kwargs})


@pytest.mark.parametrize("replacement,match", [
    (("Y=C+eps_g", "Y=C+eps_g+1"), "not homogeneous"),
    (("Y=C+eps_g", "Y=C^2+eps_g"), "affine"),
    (("Y=C+eps_g", "Y=exp(C)+eps_g"), "affine"),
    (("Y=C+eps_g", "Y=Y"), "rank deficient"),
    (("Y=C+eps_g;", ""), "variable count"),
])
def test_incomplete_nonlinear_or_singular_model_fails(replacement, match):
    with pytest.raises(ValueError, match=match):
        load_hank_mod(SOURCE.replace(*replacement)).simulate(horizon=5, **MAPS)


def test_default_rejects_ignored_mapping_options(model):
    with pytest.raises(ValueError, match="require closure"):
        model.simulate(shock_paths={"eps_m": np.zeros(5)}, horizon=5)
    with pytest.raises(ValueError, match="closure must"):
        model.simulate(closure="typo")


def test_parser_inferred_affine_model_has_same_explicit_closure(model):
    inferred = load_hank_mod(SOURCE.replace("model(linear)", "model"))
    np.testing.assert_allclose(inferred.simulate(horizon=5, **MAPS).transition_paths,
                               model.simulate(horizon=5, **MAPS).transition_paths)


def test_dp_household_preserves_current_vs_saving_aggregate_timing():
    household = dp.Model("mapped_household")
    household.state("a", np.geomspace(1e-4, 20.0001, 35)-1e-4)
    household.exogenous("z", dp.Markov([.5, 1.5], [[.9, .1], [.1, .9]]))
    household.parameters(beta=.96, gamma=2., r=.02, w=1.)
    household.local("c=(1+r)*a+w*z-a(+1)")
    household.reward("crra(c,gamma)")
    household.subject_to("c>0")
    household.aggregate(C="c", A="a", S="a(+1)")
    solution = household.solve(method="egm", tol=1e-11)
    source = """var C A S w r; varexo income;
    model(linear); w=income; r=0; end;"""
    T = 7
    wage = .001 * .6 ** np.arange(T)
    result = solve_hank_bridge(source, household_model=solution, horizon=T,
        closure="model", household_inputs={"w": "w", "r": "r"},
        household_outputs={"C": "C", "A": "A", "S": "S"}, shock_paths={"income": wage})
    expected = dp_sequence_jacobian(solution, inputs=["w", "r"], outputs=["C", "A", "S"], T=T)
    for name in ("C", "A", "S"):
        np.testing.assert_allclose(result.transition_paths[name], expected[name]["w"] @ wage, atol=1e-13)
    assert result.transition_paths.A.iloc[0] == 0
    assert abs(result.transition_paths.S.iloc[0]) > 1e-5
    np.testing.assert_allclose(result.transition_paths.A.iloc[1:], result.transition_paths.S.iloc[:-1], atol=1e-12)


def test_residual_context_freezes_parameters_and_household_derivatives(model):
    result = model.simulate(horizon=5, **MAPS)
    original = model.params["kappa"]
    try:
        model.params["kappa"] = 200
        result.jacobians["J_C_r"][:] = 200
        aggregate, household = result._audit_context["residuals"](result.transition_paths)
        assert max(np.max(np.abs(aggregate)), np.max(np.abs(household))) < 1e-14
        wrong = result.transition_paths.copy()
        wrong.loc[0, "C"] += .1
        aggregate, household = result._audit_context["residuals"](wrong)
        assert np.max(np.abs(aggregate)) >= .09 and np.max(np.abs(household)) >= .09
    finally:
        model.params["kappa"] = original


@pytest.mark.pyodide_smoke
def test_nonlinear_editable_one_asset_matches_existing_transition(model):
    editable = model.simulate(horizon=8, nonlinear=True, closure_tol=1e-10, **MAPS)
    builtin = model.simulate(horizon=8, nonlinear=True)
    assert editable.converged
    np.testing.assert_allclose(editable.transition_paths.Y, builtin.transition_paths.Y, atol=2e-8)
    assert editable.diagnostics["household_residual_max_abs"] < 1e-10
    assert editable.diagnostics["household_budget_max_abs"] < 1e-12
    assert editable.diagnostics["distribution_mass_max_abs"] < 1e-12
    assert editable._audit_context["mode"] == "hank_model_nonlinear"
    assert "residuals" not in editable._audit_context  # Audits cannot trigger a hidden household solve.
    altered = editable.transition_paths.copy()
    altered.loc[0, "Y"] += .1
    assert np.max(np.abs(editable._audit_context["aggregate_residuals_fn"](altered))) > .09


def test_nonlinear_failure_returns_last_path_with_false_convergence(model):
    linear = model.simulate(horizon=8, magnitude=-.01, **MAPS)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        failed = model.simulate(horizon=8, magnitude=-.01, nonlinear=True,
                                closure_tol=1e-14, closure_max_iter=1, **MAPS)
    assert not failed.converged
    assert failed.diagnostics["iterations"] == 1
    assert failed.diagnostics["household_residual_max_abs"] > 1e-14
    assert np.max(np.abs(failed.transition_paths.Y - linear.transition_paths.Y)) > 1e-7


TWO_SOURCE = """
var Y C CHI rb ra pi i; varexo eps_m;
parameters beta phi_pi kappa alpha_ab;
beta=.98; phi_pi=1.5; kappa=.1; alpha_ab=.5;
hetagent_block; model=two_asset_hank; assets=2; n_a=16; n_b=13; a_max=40; b_max=10; end;
model(linear);
Y=C+CHI;
pi=beta*pi(+1)+kappa*Y;
i=phi_pi*pi+eps_m;
rb=i-pi(+1);
ra=alpha_ab*rb;
end;
"""


def test_nonlinear_two_asset_closed_household_small_shock_limit():
    model = load_hank_mod(TWO_SOURCE)
    kwargs = dict(horizon=6, magnitude=-1e-4, closure="model",
        household_inputs={"Y": "Y", "r_b": "rb", "r_a": "ra"}, household_outputs={"C": "C", "CHI": "CHI"})
    linear = model.simulate(**kwargs)
    nonlinear = model.simulate(**kwargs, nonlinear=True, closure_tol=1e-10)
    assert nonlinear.converged
    np.testing.assert_allclose(nonlinear.transition_paths.Y, linear.transition_paths.Y, atol=1e-7)
    np.testing.assert_allclose(nonlinear.transition_paths.Y, nonlinear.transition_paths.C + nonlinear.transition_paths.CHI, atol=1e-12)
    assert nonlinear.diagnostics["household_budget_max_abs"] < 1e-12
    assert nonlinear.diagnostics["distribution_mass_max_abs"] < 1e-12


@pytest.fixture(scope="module")
def two_asset_solution():
    m = dp.Model("nonlinear_two_asset")
    m.state("a", np.linspace(0, np.sqrt(40), 16)**2)
    m.state("b", np.linspace(0, np.sqrt(10), 13)**2)
    m.exogenous("z", dp.Markov([.5, 1.5], [[.95, .05], [.1, .9]]))
    m.parameters(beta=.98, gamma=1., r_b=.005, r_a=.0125, w=1., chi_0=1., chi_1=1., a_bar=.25)
    m.local("d=a(+1)-(1+r_a)*a")
    m.local("chi=chi_0/(1+chi_1)*abs(d)*(abs(d)/(a+a_bar))^chi_1")
    m.local("c=(1+r_b)*b+(1+r_a)*a+w*z-a(+1)-b(+1)-chi")
    m.reward("crra(c,gamma)")
    m.subject_to("c>0")
    m.aggregate(C="c", A="a", B="b")
    return m.solve(method="egm", tol=1e-10)


def test_nonlinear_dp_two_asset_is_exact_transition_at_endogenous_prices(two_asset_solution):
    from puremacro.models.hank_sequence_space import dp_nonlinear_transition
    source = """var C A B w rb ra; varexo income;
    model(linear); w=income+.05*C; rb=0; ra=0; end;"""
    T = 6
    result = solve_hank_bridge(source, household_model=two_asset_solution, horizon=T, closure="model",
        household_inputs={"w": "w", "r_b": "rb", "r_a": "ra"},
        household_outputs={"C": "C", "A": "A", "B": "B"},
        shock_paths={"income": .002*.6**np.arange(T)}, nonlinear=True, closure_tol=1e-11)
    assert result.converged
    out = result.transition_paths
    independent = dp_nonlinear_transition(two_asset_solution, {"w": out.w, "r_b": out.rb, "r_a": out.ra}, outputs=["C", "A", "B"])
    for name in independent:
        np.testing.assert_allclose(out[name], independent[name], atol=1e-11)
    np.testing.assert_allclose(out.w-.05*out.C, out.income, atol=1e-14)
    assert result.diagnostics["household_budget_max_abs"] < 1e-12
    assert result.diagnostics["distribution_mass_max_abs"] < 1e-12


def test_nonlinear_dp_one_asset_preserves_current_and_saving_paths():
    from puremacro.models.hank_sequence_space import dp_nonlinear_transition
    m = dp.Model("nonlinear_one_asset")
    m.state("a", np.geomspace(1e-4, 20.0001, 35)-1e-4)
    m.exogenous("z", dp.Markov([.5, 1.5], [[.9, .1], [.1, .9]]))
    m.parameters(beta=.96, gamma=2., r=.02, w=1.)
    m.local("c=(1+r)*a+w*z-a(+1)")
    m.reward("crra(c,gamma)")
    m.subject_to("c>0")
    m.aggregate(C="c", A="a", S="a(+1)")
    sol = m.solve(method="egm", tol=1e-11)
    source = "var C A S w r; varexo income; model(linear); w=income+.05*C; r=0; end;"
    result = solve_hank_bridge(source, household_model=sol, horizon=6, closure="model",
        household_inputs={"w": "w", "r": "r"}, household_outputs={"C": "C", "A": "A", "S": "S"},
        shock_paths={"income": .002*.6**np.arange(6)}, nonlinear=True, closure_tol=1e-11)
    out = result.transition_paths
    expected = dp_nonlinear_transition(sol, {"w": out.w, "r": out.r}, outputs=["C", "A", "S"])
    assert result.converged
    for name in expected:
        np.testing.assert_allclose(out[name], expected[name], atol=1e-11)
    np.testing.assert_allclose(out.A.iloc[1:], out.S.iloc[:-1], atol=1e-9)
    assert result.diagnostics["household_budget_max_abs"] < 1e-12
    assert result.diagnostics["distribution_mass_max_abs"] < 1e-12


def test_nonlinear_infeasible_household_path_rejected(two_asset_solution):
    source = """var C; varexo income; model(linear); end;"""
    with pytest.raises(ValueError, match="nonpositive"):
        solve_hank_bridge(source, household_model=two_asset_solution, horizon=4, closure="model",
            household_inputs={"w": "income"}, household_outputs={"C": "C"},
            shock_paths={"income": np.full(4, -2.)}, nonlinear=True)


def test_duplicate_household_price_aliases_rejected(two_asset_solution):
    source = """var C r rb; varexo income; model(linear); r=income; rb=0; end;"""
    with pytest.raises(ValueError, match="duplicate aliases"):
        solve_hank_bridge(source, household_model=two_asset_solution, horizon=4, closure="model",
            household_inputs={"r": "r", "rb": "rb"}, household_outputs={"C": "C"}, shock_paths={})
