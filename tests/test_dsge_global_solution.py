"""Independent equilibrium-policy oracles and accuracy/failure contracts."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from puremacro.dsge.global_solution import (
    EquilibriumProblem, audit_policy, compare_policies, gaussian_quadrature,
    refine_time_iteration, solve_adaptive_time_iteration, solve_time_iteration,
)
from puremacro.dsge.local_accuracy import evaluate_perturbation
from puremacro.dsge.model import read_mod


def brock_mirman(domain=((0.1, 0.5), (-0.05, 0.05))):
    alpha, beta = 0.33, 0.95

    def resource(state):
        return np.exp(state[1]) * state[0] ** alpha

    def residual(state, control, next_state, next_control, innovation):
        consumption = resource(state) - control[0]
        next_consumption = resource(next_state) - next_control[0]
        marginal_return = alpha * np.exp(next_state[1]) * next_state[0] ** (alpha - 1)
        return [1 / consumption - beta * marginal_return / next_consumption]

    problem = EquilibriumProblem(
        ("capital", "log_productivity"), ("savings",), domain,
        lambda s, x, e: np.array([x[0], e[0]]), residual,
        [[-0.05], [0.05]], [0.5, 0.5],
        control_bounds=lambda s: [(1e-6, resource(s) - 1e-6)],
        admissible=lambda s, x: 0 < x[0] < resource(s),
    )
    exact = lambda s: np.array([alpha * beta * resource(s)])
    initial = lambda s: np.array([0.26 * resource(s)])
    return problem, exact, initial


@pytest.mark.parametrize("basis,orders", [("chebyshev", (7, 2)), ("smolyak", 3)])
@pytest.mark.pyodide_smoke
def test_stochastic_brock_mirman_off_grid_oracle(basis, orders):
    problem, exact, initial = brock_mirman()
    result = solve_time_iteration(problem, initial, basis=basis, orders=orders, tol=1e-8)
    assert result.converged
    points = np.column_stack((np.linspace(0.11, 0.49, 27), np.linspace(-0.048, 0.048, 27)))
    reference = np.array([exact(s) for s in points])
    np.testing.assert_allclose(result(points), reference, rtol=0, atol=2e-5)
    audit = audit_policy(problem, result, points, tol=2e-3)
    assert audit.passed
    assert audit.diagnostics.next_state_outside_mass.max() == 0
    assert list(result.to_frame()) == ["capital", "log_productivity", "savings"]
    # The independent oracle satisfies the stochastic Euler equation itself.
    assert audit_policy(problem, exact, points, tol=1e-12).passed


def test_grid_convergence_does_not_claim_off_grid_accuracy():
    problem, exact, initial = brock_mirman(domain=((0.03, 1.0), (-0.05, 0.05)))
    result = solve_time_iteration(problem, initial, orders=(7, 2), tol=1e-7)
    assert result.converged
    points = np.column_stack((np.linspace(0.04, 0.97, 45), np.linspace(-0.047, 0.047, 45)))
    audit = audit_policy(problem, result, points, tol=1e-4)
    assert not audit.passed
    assert audit.max_kkt_residual > 0.01
    comparison = compare_policies(problem, {"projection": result, "analytic": exact}, points, tol=1e-4)
    assert not comparison.loc["projection", "passed"]
    assert comparison.loc["analytic", "passed"]


def test_expected_residual_uses_weights_and_correlated_gaussian_rule():
    covariance = np.array([[0.04, -0.015], [-0.015, 0.09]])
    nodes, weights = gaussian_quadrature(covariance, order=3)
    np.testing.assert_allclose(weights @ nodes, 0, atol=1e-15)
    np.testing.assert_allclose(nodes.T @ (nodes * weights[:, None]), covariance, atol=1e-14)
    np.testing.assert_allclose(weights @ nodes[:, 0] ** 4, 3 * covariance[0, 0] ** 2, atol=1e-14)
    singular, singular_weights = gaussian_quadrature([[1, 1], [1, 1]], order=3)
    assert len(singular) == 3
    np.testing.assert_allclose(singular[:, 0], singular[:, 1], atol=1e-14)
    np.testing.assert_allclose(singular.T @ (singular * singular_weights[:, None]), [[1, 1], [1, 1]], atol=1e-14)
    problem = EquilibriumProblem(("s",), ("x",), ((0, 1),),
                                 lambda s, x, e: s,
                                 lambda s, x, sp, xp, e: x - e,
                                 [[0], [2]], [0.75, 0.25])
    assert audit_policy(problem, lambda s: np.array([0.5]), [[0.4]], tol=1e-12).passed
    assert not audit_policy(problem, lambda s: np.array([1.0]), [[0.4]], tol=1e-12).passed


def test_complementarity_kink_and_infeasible_policy_are_distinct():
    problem = EquilibriumProblem(("s",), ("x",), ((-1, 1),),
                                 lambda s, x, e: 0.5 * s,
                                 lambda s, x, sp, xp, e: x - s,
                                 control_bounds=[(0, 2)], complementarity={0: 0})
    result = solve_time_iteration(problem, lambda s: np.array([0.5]), basis="linear", orders=2, tol=1e-7)
    assert result.converged
    points = np.linspace(-0.98, 0.98, 33)[:, None]
    np.testing.assert_allclose(result(points)[:, 0], np.maximum(points[:, 0], 0), atol=2e-6)
    audit = audit_policy(problem, result, points, tol=2e-6)
    assert audit.passed
    assert audit.max_residual > 0.9  # Binding residual need not be zero.
    assert audit.max_kkt_residual < 2e-6
    bad = audit_policy(problem, lambda s: np.array([-0.1]), [[-0.5]], tol=1e-6)
    assert not bad.passed
    assert not bad.diagnostics.feasible.all()


def test_domain_exit_requires_explicit_conditional_acceptance():
    problem = EquilibriumProblem(("s",), ("x",), ((0, 1),),
                                 lambda s, x, e: s + 2,
                                 lambda s, x, sp, xp, e: x - 1)
    with pytest.warns(RuntimeWarning, match="converged=False"):
        rejected = solve_time_iteration(problem, [1], max_iter=2, orders=2)
    assert not rejected.converged
    assert rejected.audit.diagnostics.next_state_outside_mass.min() == 1
    result = solve_time_iteration(problem, [1], orders=2, require_domain=False, extrapolation="allow")
    assert result.converged and not result.require_domain
    assert not result.audit.require_domain
    assert result.audit.diagnostics.next_state_outside_mass.max() == 1
    assert not audit_policy(problem, result, [[0.4]]).passed
    assert result([2.0])[0] == pytest.approx(1)
    with pytest.raises(ValueError, match="outside"):
        rejected([2.0])


@pytest.mark.parametrize("basis,orders", [("chebyshev", (2, 2)), ("smolyak", 2)])
@pytest.mark.pyodide_smoke
def test_explicit_polynomial_extrapolation_preserves_affine_policy(basis, orders):
    exact = lambda s: np.array([2 + 3 * s[0] - 2 * s[1]])
    problem = EquilibriumProblem(
        ("s", "t"), ("x",), ((0, 1), (0, 1)),
        lambda s, x, e: s + 1,
        lambda s, x, sp, xp, e: x - 0.5 * xp - (exact(s) - 0.5 * exact(sp)),
    )
    result = solve_time_iteration(problem, exact, basis=basis, orders=orders,
                                  require_domain=False, extrapolation="allow", tol=1e-9)
    assert result.converged
    points = np.array([[-0.2, 0.4], [1.8, 2.1], [0.3, 0.7]])
    np.testing.assert_allclose(result(points), [exact(s) for s in points], atol=1e-10)
    audit = audit_policy(problem, result, [[0.3, 0.7]], require_domain=False, tol=1e-9)
    assert audit.passed
    assert audit.diagnostics.next_state_outside_mass.iloc[0] == 1


def test_policy_result_reports_and_explicit_headless_slice(monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    problem = EquilibriumProblem(
        ("state_a", "state_b"), ("control_#",), ((0, 1), (0, 1)),
        lambda s, x, e: s, lambda s, x, sp, xp, e: x - s[0] - 2 * s[1],
    )
    points = np.array([[0.1, 0.2], [0.5, 0.2], [0.9, 0.2]])
    refined = refine_time_iteration(problem, [0], [{"orders": (1, 1)}, {"orders": (2, 2)}],
                                     points, max_solves=2)
    for report in (refined, refined.result, refined.audit):
        pd.testing.assert_frame_equal(report.to_dataframe(), report.to_frame())
        assert isinstance(report.to_markdown(index=False), str)
        assert "\\begin{tabular}" in report.to_latex()
        assert "#table(" in report.to_typst()
        frame = report.to_frame()
        frame.iloc[0, 0] = -99
        assert report.to_frame().iloc[0, 0] != -99
    assert "control\\_\\#" in refined.result.to_latex()
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))
    figure = refined.result.plot(points, state="state_a")
    axis = figure.axes[0]
    np.testing.assert_allclose(axis.lines[0].get_xdata(), points[:, 0])
    np.testing.assert_allclose(axis.lines[0].get_ydata(), points[:, 0] + 2 * points[:, 1])
    assert refined.plot(points, state=0, ax=axis) is axis
    assert not shown
    refined.result.plot(points, ax=axis, show=True)
    assert shown == [True]
    with pytest.raises(ValueError, match="states must have shape"):
        refined.result.plot([[0.1], [0.2]])
    plt.close(figure)


def test_refinement_budget_cannot_be_reported_as_accuracy():
    problem, _, initial = brock_mirman()
    points = np.column_stack((np.linspace(0.12, 0.47, 19), np.linspace(-0.04, 0.04, 19)))
    levels = [{"orders": (5, 2)}, {"orders": (9, 3)}, {"orders": (13, 3)}]
    partial = refine_time_iteration(problem, initial, levels, points, max_solves=1, tol=1e-4)
    assert not partial.accepted and len(partial.history) == 1
    refined = refine_time_iteration(problem, initial, levels, points, max_solves=3, tol=1e-4)
    assert refined.accepted
    assert refined.audit.passed
    assert refined.history.iloc[-1].policy_change <= 1e-4


def test_adaptive_anisotropic_orders_are_checked_off_grid():
    problem, _, initial = brock_mirman()
    points = np.column_stack((np.linspace(0.12, 0.47, 19), np.linspace(-0.04, 0.04, 19)))
    result = solve_adaptive_time_iteration(problem, initial, points, orders=(3, 2),
                                           tol=1e-4, max_solves=5)
    assert result.accepted and result.audit.passed
    assert len(result.history) >= 2
    assert result.history.iloc[-1].orders[0] > 3
    assert len(set(result.history.iloc[-1].orders)) > 1
    limited = solve_adaptive_time_iteration(problem, initial, points, orders=(3, 2),
                                            tol=1e-4, max_solves=1)
    assert not limited.accepted


def test_conditional_markov_probabilities_not_unconditional_weights():
    problem = EquilibriumProblem(
        ("state",), ("decision",), ((0, 1),),
        lambda s, x, e: e,
        lambda s, x, sp, xp, e: x - sp,
        shock_rule=lambda state: (np.array([[0], [1]]), np.array([0.8-0.6*state[0], 0.2+0.6*state[0]])),
    )
    result = solve_time_iteration(problem, [0.5], orders=2, tol=1e-9)
    assert result.converged
    points = np.linspace(0.03, 0.97, 17)[:, None]
    np.testing.assert_allclose(result(points)[:, 0], 0.2+0.6*points[:, 0], atol=1e-9)
    assert audit_policy(problem, result, points, tol=1e-9).passed
    invalid = replace(problem, shock_rule=lambda state: (np.array([[0], [1]]), np.array([state[0], state[0]])))
    audit = audit_policy(invalid, result, [[0.3]])
    assert not audit.passed
    assert "sum to one" in audit.diagnostics.failure.iloc[0]


def test_compiled_model_bridge_copies_calibration_and_accepts_named_reference():
    model = read_mod("var x; parameters target; target=2; model; x=target; end;")
    problem = EquilibriumProblem.from_model(
        model, state_names=("state",), control_names=("x",), domain=((0, 1),),
        transition=lambda s, x, e: s, current=lambda s, x: x,
        lagged=lambda s: np.zeros(1), equations=(0,), steady_state={"x": 2},
    )
    model.parameter_values["target"] = -5
    assert audit_policy(problem, lambda s: np.array([2.0]), [[0.5]], tol=1e-12).passed
    for selector in (0.9, False):
        with pytest.raises(ValueError, match="names or integer indices"):
            EquilibriumProblem.from_model(
                model, state_names=("state",), control_names=("x",), domain=((0, 1),),
                transition=lambda s, x, e: s, current=lambda s, x: x,
                lagged=lambda s: np.zeros(1), equations=(selector,),
            )


@pytest.mark.parametrize("updates", [
    {"shock_weights": [0.2, 0.2]}, {"shock_weights": [-1, 2]},
    {"domain": ((0, 0), (-0.1, 0.1))}, {"residual_scale": [0]},
    {"complementarity": {0: 3}},
])
def test_invalid_problem_contracts_are_rejected(updates):
    problem, _, _ = brock_mirman()
    with pytest.raises(ValueError):
        replace(problem, **updates)


def test_solver_budgets_nonfinite_policies_and_failure_reporting():
    problem, _, initial = brock_mirman()
    with pytest.raises(ValueError, match="max_nodes"):
        solve_time_iteration(problem, initial, orders=(1000, 1000), max_nodes=100)
    with pytest.raises(ValueError, match="max_nodes"):
        solve_time_iteration(problem, initial, basis="smolyak", orders=8, max_nodes=10)
    with pytest.raises(ValueError, match="initial_policy"):
        solve_time_iteration(problem, [np.nan], orders=(3, 2))
    with pytest.raises(ValueError, match="order"):
        solve_time_iteration(problem, initial, orders=True)
    with pytest.warns(RuntimeWarning):
        incomplete = solve_time_iteration(problem, initial, orders=(4, 2), max_iter=1)
    assert not incomplete.converged
    with pytest.raises(RuntimeError, match="converged=False"):
        solve_time_iteration(problem, initial, orders=(4, 2), max_iter=1, strict=True)
    with pytest.raises(ValueError, match="max_nodes"):
        gaussian_quadrature(np.eye(3), order=5, max_nodes=10)
    with pytest.raises(ValueError, match="semidefinite"):
        gaussian_quadrature([[1, 2], [2, 1]])
    with pytest.raises(ValueError, match="semidefinite"):
        gaussian_quadrature([[-1e-20]])


def test_mod_bridge_preserves_conditional_euler_and_neoclassical_labor():
    model = read_mod(Path(__file__).resolve().parents[1] / "curso/notebooks/modelos/neoclasico_prevision_perfecta.mod")
    p = model.parameter_values
    alpha, delta, rho = p["alpha"], p["delta"], p["rho_a"]
    names = list(model.variables)

    def current(state, control):
        capital, productivity = state
        consumption, labor = control
        output = np.exp(productivity) * capital ** alpha * labor ** (1 - alpha)
        row = dict(y=output, c=consumption, k=output + (1-delta)*capital-consumption, l=labor, a=productivity)
        return np.array([row[name] for name in names])

    def lagged(state):
        return np.array([state[0] if name == "k" else 0 for name in names])

    shocks, weights = gaussian_quadrature([[0.02**2]], order=3)
    problem = EquilibriumProblem.from_model(
        model, state_names=("k_lag", "a"), control_names=("c", "l"),
        domain=((1.7, 5), (-0.12, 0.12)),
        transition=lambda s, x, e: np.array([current(s, x)[names.index("k")], rho*s[1]+e[0]]),
        current=current, lagged=lagged, equations=("Euler", "intratemporal"),
        shock_nodes=shocks, shock_weights=weights,
        control_bounds=lambda s: [(1e-6, (1-delta)*s[0]+np.exp(s[1])*s[0]**alpha*0.1**(1-alpha)-1e-5), (0.1, 0.9)],
        admissible=lambda s, x: current(s, x)[names.index("k")] > 0,
    )
    local = model.solve(order=2)
    rules = local.decision_rules()
    def initial(state):
        previous = np.array([state[0] if name == "k" else 0 for name in rules.state_variables])
        value = evaluate_perturbation(local, previous, [state[1]])
        return value[[list(rules.variable_names).index(name) for name in ("c", "l")]]
    result = solve_time_iteration(problem, initial, orders=(7, 3), tol=1e-7,
                                  require_domain=False, extrapolation="allow")
    assert result.converged
    assert result.audit.diagnostics.next_state_outside_mass.max() > 0
    points = np.column_stack((np.linspace(2.6, 3.8, 31), np.linspace(-0.035, 0.035, 31)))
    audit = audit_policy(problem, result, points, tol=1e-5)
    assert audit.passed and audit.max_residual < 1e-5
    assert audit.diagnostics.next_state_outside_mass.max() == 0
    assert audit_policy(problem, initial, points, tol=1e-5).max_residual > 1e-4
