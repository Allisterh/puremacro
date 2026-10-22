"""Terminal-condition provenance across public .mod execution entry points."""
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from puremacro.dsge import ModCompatibilityError, build_dynare, load_mod, parse_mod, solve_perfect_foresight
from puremacro.dsge._parser import parse_mod_to_dag


_FIXTURES = Path(__file__).parent / "fixtures" / "dynare_issue_177"


def _build_from_parsed(source):
    parsed = parse_mod(source)
    model = build_dynare(parsed["equations"], variables=parsed["variables"],
                         shocks=parsed["shocks"], params=parsed["params"], guess=parsed["guess"])
    # The builder carries the parsed DAG through the compiled callable. Unlike
    # load_mod, it does not copy boundary declarations onto private attributes.
    assert not hasattr(model, "_endval")
    assert model.endval == parsed["endval"]
    return model


def _source(terminal, command):
    name = "endval_solved" if terminal == "steady" else "endval_literal"
    source = (_FIXTURES / f"{name}.mod").read_text()
    if terminal == "endval_steady":
        source = source.replace("perfect_foresight_setup(periods=10)",
                                "perfect_foresight_setup(periods=10, endval_steady)")
    if command == "simul":
        source = source[:source.index("perfect_foresight_setup")]
        source += "simul(periods=10" + (", endval_steady" if terminal == "endval_steady" else "") + ");\n"
    return source


@pytest.mark.parametrize("terminal", ["literal", "steady", "endval_steady"])
@pytest.mark.parametrize("command", ["setup_solver", "simul"])
@pytest.mark.parametrize("loader", [load_mod, partial(load_mod, command_mode="legacy"), _build_from_parsed],
                         ids=["default", "legacy", "builder"])
def test_legacy_dispatch_preserves_parsed_terminal_provenance(terminal, command, loader):
    source = _source(terminal, command)
    model = loader(source)
    direct = model.solve_perfect_foresight(periods=10)
    dispatched = model.execute_mod_commands()
    result = dispatched.perfect_foresight
    assert direct.converged and result is not None and result.converged

    # Native Dynare outputs distinguish the literal guess 3 from the solved
    # terminal equilibrium 2. Direct/dispatcher parity alone cannot detect a
    # shared wrong boundary or a missing permanent exogenous path.
    name = "endval_literal" if terminal == "literal" else "endval_solved"
    reference = pd.read_csv(_FIXTURES / f"{name}_dynare_out.csv").iloc[1:11]
    np.testing.assert_allclose(result.path.to_numpy(), reference.to_numpy(), rtol=0, atol=1e-10)
    np.testing.assert_allclose(result.path.to_numpy(), direct.path.to_numpy(), rtol=0, atol=1e-10)
    np.testing.assert_allclose(result.path["y"], 2.0 - 0.5 ** np.arange(1, 11), rtol=0, atol=1e-10)
    assert dispatched.endval == {"e": 0.5, "y": 3.0, "x": 3.0}
    assert model.endval == dispatched.endval
    assert dispatched.commands_executed == ("endval", "perfect_foresight_solver")


@pytest.mark.parametrize("loader", [load_mod, _build_from_parsed], ids=["load_mod", "builder"])
def test_parsed_exogenous_only_endval_remains_literal_without_steady(loader):
    source = _source("literal", "setup_solver").replace("y = 3;\nx = 3;", "")
    model = loader(source)
    result = model.execute_mod_commands().perfect_foresight
    assert result is not None and result.converged
    # Parsed endval alone does not request the implicit static solve that an
    # explicit Python exogenous-only endval mapping requests.
    np.testing.assert_allclose(result.path["x"], 1.0, rtol=0, atol=1e-10)
    np.testing.assert_allclose(result.path["y"], 2.0 - 0.5 ** np.arange(1, 11), rtol=0, atol=1e-10)
    explicit = model.solve_perfect_foresight(periods=10, endval={"e": 0.5})
    assert explicit.converged
    np.testing.assert_allclose(explicit.path["x"], 2.0, rtol=0, atol=1e-10)


@pytest.mark.parametrize("terminal", ["steady", "endval_steady"])
@pytest.mark.parametrize("loader", [load_mod, _build_from_parsed], ids=["load_mod", "builder"])
@pytest.mark.parametrize("method", ["auto", "central", "complex"])
@pytest.mark.parametrize("override,target,shock", [
    ({"endval": {"e": 1.0, "y": 4.0, "x": 4.0}}, 4.0, 1.0),
    ({"endval": {"e": 1.0}}, 3.0, 1.0),
    ({"endval": {}}, 1.0, 0.0),
    ({"y_end": np.array([5.0, 5.0])}, 5.0, 0.5),
])
def test_explicit_terminal_override_takes_precedence_over_parsed_steady(terminal, loader, method, override, target, shock):
    model = loader(_source(terminal, "setup_solver"))
    result = solve_perfect_foresight(model, periods=10, method=method, **override)
    assert result.converged
    np.testing.assert_allclose(result.path["x"], target, rtol=0, atol=1e-9)
    np.testing.assert_allclose(result.path["y"], 1.0 + 2.0 * shock * (1.0 - 0.5 ** np.arange(1, 11)),
                               rtol=0, atol=1e-9)
    # An override must not rewrite the model's parsed declaration for a later
    # command invocation.
    assert model.endval == {"e": 0.5, "y": 3.0, "x": 3.0}
    dispatched = model.execute_mod_commands().perfect_foresight
    assert dispatched is not None and dispatched.converged
    np.testing.assert_allclose(dispatched.path["x"], 2.0, rtol=0, atol=1e-10)


@pytest.mark.parametrize("terminal", ["literal", "steady", "endval_steady"])
def test_ordered_mode_still_reports_perfect_foresight_as_unsupported(terminal):
    source = _source(terminal, "setup_solver")
    with pytest.raises(ModCompatibilityError) as loading:
        load_mod(source, command_mode="ordered")
    assert any(issue.code == "ORDERED_COMMAND_UNSUPPORTED" for issue in loading.value.report.issues)
    model = load_mod(source)
    with pytest.raises(ModCompatibilityError) as execution:
        model.execute_mod_commands(ordered=True)
    assert any(issue.code == "ORDERED_COMMAND_UNSUPPORTED" for issue in execution.value.report.issues)


@pytest.mark.parametrize("loader", [load_mod, _build_from_parsed, parse_mod_to_dag],
                         ids=["load_mod", "builder", "dag"])
@pytest.mark.parametrize("terminal", ["literal", "steady"])
@pytest.mark.parametrize("override,initial", [
    ({}, 4.0),
    ({"histval": {}}, 1.0),
    ({"histval": {"y": 2.0}}, 2.0),
    ({"y_init": np.array([6.0, 1.0])}, 6.0),
    ({"histval": {"y": 2.0}, "y_init": np.array([6.0, 1.0])}, 6.0),
])
def test_historical_boundary_matches_across_construction_routes(loader, terminal, override, initial):
    source = _source(terminal, "setup_solver").replace(
        "endval;", "histval; y(0)=4; x(0)=1; end;\nendval;")
    model = loader(source)
    if loader is _build_from_parsed:
        assert not hasattr(model, "_histval")
        assert model.histval == {"y": 4.0, "x": 1.0}
    result = solve_perfect_foresight(model, periods=10, **override)
    assert result.converged
    # Independent recurrence for permanent e=.5: y_t = 1 + .5*y_(t-1).
    expected = 2.0 + (initial - 2.0) * 0.5 ** np.arange(1, 11)
    np.testing.assert_allclose(result.path["y"], expected, rtol=0, atol=1e-10)
    np.testing.assert_allclose(result.path["x"], 3.0 if terminal == "literal" else 2.0,
                               rtol=0, atol=1e-10)
    if loader is not parse_mod_to_dag:
        # Explicit call arguments must not rewrite the source declaration.
        dispatched = model.execute_mod_commands()
        assert dispatched.histval == {"y": 4.0, "x": 1.0}
        assert dispatched.perfect_foresight is not None and dispatched.perfect_foresight.converged
        np.testing.assert_allclose(dispatched.perfect_foresight.path["y"],
                                   2.0 + 2.0 * 0.5 ** np.arange(1, 11), rtol=0, atol=1e-10)
