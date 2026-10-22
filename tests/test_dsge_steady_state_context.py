"""Independent regressions for frozen STEADY_STATE residual contexts."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from puremacro.dsge import extended_path, load_mod, solve_perfect_foresight
from puremacro.dsge._ast import Call, Var
from puremacro.dsge._parser import parse_mod_to_dag


_FIXTURES = Path(__file__).parent / "fixtures" / "dynare_issue_177"


@pytest.mark.parametrize("solver", ["pf", "mcp", "surprise", "extended"])
def test_trajectory_preserves_unrelated_optional_callback_argument(solver):
    def residual(lead, curr, lag, shocks, target=2.0):
        return [curr[0] - target]

    if solver == "extended":
        result = extended_path(residual, periods=3, horizon=5, y_init=[0.0],
                               y_ss=[0.0], shocks=np.zeros((3, 1)))
    else:
        result = solve_perfect_foresight(residual, periods=3, y_init=[0.0], y_ss=[0.0],
                                        mcp=solver == "mcp", surprise=solver == "surprise")
    assert result.converged
    np.testing.assert_allclose(result.path.to_numpy(), 2.0, atol=1e-10)


@pytest.mark.parametrize("solver", ["pf", "extended"])
def test_callback_body_typeerror_is_not_retried_without_context(solver):
    def residual(lead, curr, lag, shocks, steady_state=None):
        if steady_state is not None:
            raise TypeError("intentional residual failure")
        return [curr[0]]

    with pytest.raises(TypeError, match="intentional residual failure"):
        if solver == "extended":
            extended_path(residual, periods=3, horizon=5, y_ss=[0.0],
                          shocks=np.zeros((3, 1)))
        else:
            solve_perfect_foresight(residual, periods=3, y_init=[0.0], y_ss=[0.0])


@pytest.mark.parametrize("solver", ["pf", "extended"])
def test_keyword_only_steady_state_context(solver):
    def residual(lead, curr, lag, shocks, *, steady_state=None):
        target = 0.0 if steady_state is None else steady_state[0]
        return [curr[0] - target]

    if solver == "extended":
        result = extended_path(residual, periods=3, horizon=5, y_ss=[2.0],
                               shocks=np.zeros((3, 1)))
    else:
        result = solve_perfect_foresight(residual, periods=3, y_init=[0.0], y_ss=[2.0])
    assert result.converged
    np.testing.assert_allclose(result.path.to_numpy(), 2.0, atol=1e-10)


def test_positional_only_context_preserves_intervening_callback_defaults():
    def residual(lead, curr, lag, shocks, scale=3.0, steady_state=None, /):
        target = 0.0 if steady_state is None else steady_state[0]
        return [curr[0] - scale * target]

    result = solve_perfect_foresight(residual, periods=3, y_init=[0.0], y_ss=[2.0])
    assert result.converged
    np.testing.assert_allclose(result.path.to_numpy(), 6.0, atol=1e-10)


@pytest.mark.parametrize("with_steady", [False, True])
def test_endval_literal_target_is_distinct_from_solved_steady_state(with_steady):
    name = "endval_solved" if with_steady else "endval_literal"
    text = (_FIXTURES / f"{name}.mod").read_text()
    model = load_mod(text)
    result = model.solve_perfect_foresight(periods=10)
    target = 2.0 if with_steady else 3.0
    assert result.converged
    np.testing.assert_allclose(result.path["x"], target, atol=1e-10)
    np.testing.assert_allclose(result.path["y"], 2.0 - 0.5 ** np.arange(1, 11), atol=1e-10)
    reference = pd.read_csv(_FIXTURES / f"{name}_dynare_out.csv").iloc[1:11]
    np.testing.assert_allclose(result.path.to_numpy(), reference.to_numpy(), rtol=0, atol=1e-10)


def test_parsed_dag_perfect_foresight_supports_steady_state_context():
    dag = parse_mod_to_dag((_FIXTURES / "minimal_177.mod").read_text())
    shocks = np.zeros((4, 1))
    shocks[0] = 0.5
    result = solve_perfect_foresight(dag, periods=4, shocks=shocks)
    assert result.converged
    np.testing.assert_allclose(result.path["x"], 1.0, atol=1e-10)


def test_endval_steady_option_requests_terminal_solve():
    text = (_FIXTURES / "endval_literal.mod").read_text()
    text = text.replace("perfect_foresight_setup(periods=10)",
                        "perfect_foresight_setup(periods=10, endval_steady)")
    result = load_mod(text).solve_perfect_foresight(periods=10)
    assert result.converged
    np.testing.assert_allclose(result.path["x"], 2.0, atol=1e-10)


@pytest.mark.parametrize("method", ["central", "complex"])
def test_y_end_sets_lead_boundary_and_overrides_y_ss_reference(method):
    model = parse_mod_to_dag("""
        var y x; varexo e;
        model;
            y = 0.5 + 0.5*y(+1) + e;
            x = STEADY_STATE(y);
        end;
        initval; y=1; x=1; end;
        steady;
    """)
    result = solve_perfect_foresight(
        model, periods=5, y_ss=np.array([1.0, 1.0]), y_end=np.array([3.0, 3.0]), method=method,
    )
    assert result.converged
    np.testing.assert_allclose(result.path["x"], 3.0, atol=1e-10)
    np.testing.assert_allclose(result.path["y"], 1.0 + 2.0 * 0.5 ** np.arange(5, 0, -1), atol=1e-10)


def test_nested_steady_state_ast_eval_uses_current_coordinate_not_mapping_order():
    expr = Call("STEADY_STATE", (Call("STEADY_STATE", (Var("y", 2),)),))
    values = {("y", 0): 2.0, ("y", 1): 8.0, ("y", -1): 9.0}
    assert expr.eval(values, {}) == 2.0


@pytest.mark.parametrize("implementation", ["dynare", "occbin"])
def test_numeric_model_jacobians_preserve_optional_parameters(implementation):
    from puremacro.dsge.build import _Vec
    from puremacro.dsge.dynare import _lead_lag_jacobians
    from puremacro.dsge.occbin import _differentiate_dynare_eqs

    def equations(lead, curr, lag, shocks, params, scale=_Vec(["y"], [2.0])):
        return [scale.y * curr.y - 0.5 * lag.y - shocks.e - 1.5]

    if implementation == "dynare":
        matrices = _lead_lag_jacobians(equations, ["y"], ["e"], _Vec([], []),
                                     np.array([1.0]), "complex")
    else:
        matrices = _differentiate_dynare_eqs(equations, ["y"], ["e"], {}, np.array([1.0]))
    for actual, expected in zip(matrices[:4], [0.0, 2.0, -0.5, -1.0]):
        np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_nested_steady_state_renderers_match_ast_with_explicit_reference():
    from types import SimpleNamespace
    from puremacro.dsge._ast import BinOp, Param
    from puremacro.dsge._parser import _render_equations_hoisted

    expr = Call("STEADY_STATE", (BinOp("+", Var("y", 2),
                Call("STEADY_STATE", (BinOp("*", Param("rho"), Var("y", -2)),))),))
    context = {"y": 9.0, "_steady_state": {"y": 2.0}}
    expected = expr.eval(context, {"rho": 0.5})
    assert expected == 3.0
    namespace = {"steady_state": SimpleNamespace(y=2.0), "params": SimpleNamespace(rho=0.5)}
    assert eval(expr.to_python(), namespace) == expected
    prologue, expressions = _render_equations_hoisted([expr], set())
    exec("\n".join(prologue), namespace)
    assert eval(expressions[0], namespace) == expected


@pytest.mark.parametrize("loader", [parse_mod_to_dag, load_mod])
@pytest.mark.parametrize("expression,local", [
    ("STEADY_STATE(e)", ""),
    ("STEADY_STATE(STEADY_STATE(e))", ""),
    ("STEADY_STATE(local_shock)", "#local_shock = e;"),
])
def test_steady_state_exogenous_is_rejected_like_native_dynare(loader, expression, local):
    text = (_FIXTURES / "exo_steady.mod").read_text()
    text = text.replace("model;", "model;\n" + local)
    text = text.replace("STEADY_STATE(e)", expression)
    with pytest.raises(ValueError, match="STEADY_STATE.*exogenous"):
        loader(text)


def test_steady_state_exogenous_manual_ast_is_not_silently_zero():
    from puremacro.dsge._parser import _render_equations_hoisted

    expression = Call("STEADY_STATE", (Var("e"),))
    with pytest.raises(ValueError, match="STEADY_STATE.*exogenous"):
        expression.to_python(shock_names={"e"})
    with pytest.raises(ValueError, match="STEADY_STATE.*exogenous"):
        _render_equations_hoisted([expression], {"e"})
