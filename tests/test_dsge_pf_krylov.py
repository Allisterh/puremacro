"""Front-door Krylov parity, boundary semantics and numerical failure controls."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from puremacro.dsge import extended_path, load_mod, solve_perfect_foresight
from puremacro.dsge._parser import parse_mod_to_dag
from puremacro.dsge.stacked_newton import StackedProblem, _resolve_preconditioner


FIXTURES = Path(__file__).parent / "fixtures/dynare_issue_177"


@pytest.mark.parametrize("name", ["endval_literal", "endval_solved", "endval_no_steady", "endval_with_steady"])
@pytest.mark.parametrize("method", ["central", "complex"])
def test_krylov_matches_native_terminal_and_steady_state_fixtures(name, method):
    model = load_mod((FIXTURES / f"{name}.mod").read_text(encoding="utf-8"))
    result = solve_perfect_foresight(model, periods=10, method=method, linear_solver="krylov")
    reference = pd.read_csv(FIXTURES / f"{name}_dynare_out.csv").iloc[1:11]
    assert result.converged
    assert result.linear_solver == "krylov"
    np.testing.assert_allclose(result.path, reference, rtol=0, atol=1e-9)
    assert result.solver_diagnostics["converged_by"] == "residual_and_step"
    assert result.solver_diagnostics["linear_history"]


@pytest.mark.parametrize("solver", ["direct", "krylov"])
def test_explicit_terminal_reference_padding_and_context_copy(solver):
    model = parse_mod_to_dag("""
        var y x; varexo e;
        model; y = 0.5 + 0.5*y(+1) + e; x = STEADY_STATE(y); end;
        initval; y=1; x=1; end; steady;
    """)
    end = np.array([3.0, 3.0])
    shocks = np.array([[99.0], [0.1], [0.2], [-0.1], [77.0]])
    result = solve_perfect_foresight(model, periods=3, y_ss=[1, 1], y_end=end,
                                     shocks=shocks, linear_solver=solver)
    y = np.empty(3)
    lead = 3.0
    for t in range(2, -1, -1):
        y[t] = 0.5 + 0.5 * lead + shocks[t + 1, 0]
        lead = y[t]
    assert result.converged
    np.testing.assert_allclose(result.path.y, y, atol=1e-10)
    np.testing.assert_allclose(result.path.x, 3.0, atol=1e-10)
    end[:] = -100
    shocks[:] = -100
    context = result._audit_context
    assert context["mode"] == "perfect_foresight"
    np.testing.assert_array_equal(context["y_end"], [3, 3])
    np.testing.assert_allclose(context["equations"]([3, 3], [2, 3], [1, 1], [0]), [0, 0])
    np.testing.assert_allclose(context["exogenous_path"].ravel(), [0.1, 0.2, -0.1])


def nonlinear(lead, current, lag, shock):
    return [current[0] ** 2 - 0.2 * lead[0] - 0.3 * lag[0] - 0.5 - np.asarray(shock).item()]


def test_nonlinear_path_matches_direct_and_preserves_dampening():
    kwargs = dict(y_init=[0.9], y_ss=[1.0], periods=12,
                  shocks=np.linspace(0.1, 0, 12), dampening=0.5, max_iter=90)
    direct = solve_perfect_foresight(nonlinear, **kwargs)
    krylov = solve_perfect_foresight(nonlinear, linear_solver="krylov", **kwargs)
    assert direct.converged and krylov.converged
    np.testing.assert_allclose(krylov.path, direct.path, atol=2e-8, rtol=0)
    steps = [r["step"] for r in krylov.solver_diagnostics["linear_history"]]
    assert any(0 < step <= 0.5 for step in steps)
    assert max(steps) <= 0.5


def test_krylov_failure_returns_last_path_and_never_falls_back():
    with pytest.warns(RuntimeWarning, match="krylov.*iteration limit"):
        result = solve_perfect_foresight(nonlinear, y_init=[1], y_ss=[1],
                                         shocks=np.full(4, 4.0), periods=4,
                                         max_iter=1, linear_solver="krylov")
    assert not result.converged
    assert result.solver_diagnostics["converged_by"] == "iteration_limit"
    assert result.residual_norm > 1e-8
    assert not np.allclose(result.path, 1.0)


@pytest.mark.parametrize("options,exception", [
    ({"linear_solver": "bad"}, ValueError),
    ({"krylov_options": {}}, ValueError),
    ({"linear_solver": "krylov", "mcp": True}, NotImplementedError),
    ({"linear_solver": "krylov", "mcp_bounds": {0: (0, None)}}, NotImplementedError),
    ({"linear_solver": "krylov", "krylov_options": {"step_tol": None}}, ValueError),
    ({"linear_solver": "krylov", "krylov_options": {"krylov_method": "bad"}}, ValueError),
    ({"linear_solver": "krylov", "krylov_options": {"preconditioner": "bad"}}, ValueError),
    ({"linear_solver": "krylov", "krylov_options": {"preconditioner": 0}}, TypeError),
    ({"linear_solver": "krylov", "krylov_options": {"inner_m": None}}, ValueError),
])
def test_invalid_options_rejected_before_model_evaluation(options, exception):
    calls = []
    def equations(lead, current, lag, shock):
        calls.append(True)
        return [current[0]]
    with pytest.raises(exception):
        solve_perfect_foresight(equations, y_init=[0], y_ss=[0], periods=3, **options)
    assert calls == []


@pytest.mark.parametrize("route", ["extended", "surprise"])
def test_rolling_krylov_forwarding_preserves_shocks_and_fixed_reference(route):
    def equations(lead, current, lag, shock, *, steady_state=None):
        return [current[0] - 0.4 * lag[0] - 0.6 * steady_state[0] - np.asarray(shock).item()]
    shocks = np.array([0.2, -0.1, 0.3])
    if route == "extended":
        result = extended_path(equations, periods=3, horizon=4, y_init=[2], y_ss=[2],
                               shocks=shocks[:, None], linear_solver="krylov", strict=True)
    else:
        result = solve_perfect_foresight(equations, periods=3, y_init=[2], y_ss=[2],
                                        shocks=shocks, surprise=True, linear_solver="krylov")
        assert result.linear_solver == "krylov"
        assert result._audit_context is None
    expected = []
    y = 2.0
    for e in shocks:
        y = 0.4 * y + 1.2 + e
        expected.append(y)
    assert result.converged
    np.testing.assert_allclose(result.path.iloc[:, 0], expected, atol=1e-9)


def test_current_preconditioner_reuses_only_exact_unchanged_blocks():
    coefficient = np.array([2.0])
    def eq(lead, current, lag, shock):
        return coefficient * current - shock
    def blocks(lead, current, lag, shock):
        return np.zeros((1, 1)), coefficient.reshape(1, 1), np.zeros((1, 1))
    problem = StackedProblem(eq, [0], [0], np.ones(3), blocks_fn=blocks)
    _, factory = _resolve_preconditioner(problem, "current_block_tridiagonal", None)
    first = factory(np.zeros(3))
    assert factory(np.ones(3)) is first
    np.testing.assert_allclose(first @ np.ones(3), 0.5)
    coefficient[:] = np.nextafter(2.0, 3.0)
    second = factory(np.ones(3))
    assert second is not first  # even sub-tolerance changes must invalidate
    coefficient[:] = 4.0
    third = factory(np.ones(3))
    assert third is not second
    np.testing.assert_allclose(third @ np.ones(3), 0.25)


def test_callback_typeerror_propagates_with_krylov():
    def equations(lead, current, lag, shock, *, steady_state=None):
        raise TypeError("callback body failure")
    with pytest.raises(TypeError, match="callback body failure"):
        solve_perfect_foresight(equations, y_init=[1], y_ss=[1], periods=2,
                                linear_solver="krylov", method="central")


def test_surprise_failure_is_not_reported_as_converged():
    with pytest.warns(RuntimeWarning, match="krylov"):
        result = solve_perfect_foresight(nonlinear, y_init=[1], y_ss=[1], periods=2,
                                        shocks=[4.0, 4.0], surprise=True,
                                        max_iter=1, linear_solver="krylov")
    assert not result.converged
    assert result.solver_diagnostics["mode"] == "surprise"
