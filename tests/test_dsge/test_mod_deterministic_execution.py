"""Source-ordered deterministic scenarios with independent numerical oracles."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from puremacro.dsge import read_mod, load_mod, inspect_mod, ModCompatibilityError, ModExecutionError


BASE = "var x; varexo e; parameters rho; rho=.5; model; x=rho*x(-1)+e; end;"


def test_dated_shock_expressions_use_the_overridden_calibration_snapshot():
    source = BASE + """
        parameters impulse; impulse=2;
        initval; x=0; e=0; end;
        shocks; var e; periods 1; values impulse; end;
        simul(periods=3);
    """
    result = read_mod(source, params={"impulse": 3}).execute_mod_commands()
    np.testing.assert_allclose(result.perfect_foresight.path.x, [3, 1.5, .75], atol=1e-10)
    assert result.invocations[0].effective_options["input_state"]["dated_shocks"]["e"]["values"] == [3]


@pytest.mark.parametrize("terminal_solve", ["steady; perfect_foresight_setup(periods=3);", "perfect_foresight_setup(periods=3,endval_steady);"])
@pytest.mark.pyodide_smoke
def test_analytical_steady_block_seeds_positive_domain_without_initval(terminal_solve):
    source = """
        var x; varexo e; parameters center; center=2;
        model; log(x)=.2*log(x(-1))+.8*log(center)+e; end;
        steady_state_model; x=center; end;
        steady;
        endval; e=.8*log(4); end;
    """ + terminal_solve + "perfect_foresight_solver(tolf=1e-10);"
    results = read_mod(source, params={"center": 3}).execute_mod_commands()
    initial_steady = results.invocations[0].result
    setup = next(entry.result for entry in results.invocations if entry.command == "perfect_foresight_setup")
    assert initial_steady.x == pytest.approx(3)
    np.testing.assert_allclose(setup.y_init, [3], atol=1e-10)
    np.testing.assert_allclose(setup.y_end, [12], atol=1e-9)
    np.testing.assert_allclose(setup.exogenous_path, .8*np.log(4), atol=1e-12)
    reference = 3 * 4 ** (1 - .2 ** np.arange(1, 4))
    np.testing.assert_allclose(results.perfect_foresight.path.x, reference, atol=1e-8)


@pytest.mark.pyodide_smoke
def test_setup_is_a_snapshot_and_each_result_survives_later_shocks():
    model = read_mod(BASE + """
        initval; x=0; e=0; end; steady;
        shocks; var e; periods 1; values 1; end;
        perfect_foresight_setup(periods=3);
        shocks; var e; periods 1; values 9; end;
        perfect_foresight_solver(tolf=1e-11);
        perfect_foresight_setup(periods=3);
        perfect_foresight_solver;
    """)
    results = model.execute_mod_commands()
    setups = [i.result for i in results.invocations if i.command == "perfect_foresight_setup"]
    assert "terminal date 4" in setups[0].summary()
    paths = [i for i in results.invocations if i.command == "perfect_foresight_solver"]
    np.testing.assert_allclose(setups[0].to_frame().e, [1, 0, 0])
    pd.testing.assert_frame_equal(setups[0].to_dataframe(), setups[0].to_frame())
    assert "e" in setups[0].to_markdown()
    assert "tabular" in setups[0].to_latex()
    assert "#table" in setups[0].to_typst()
    np.testing.assert_allclose(setups[1].to_frame().e, [9, 0, 0])
    np.testing.assert_allclose(paths[0].result.path.x, [1, .5, .25], atol=1e-10)
    np.testing.assert_allclose(paths[1].result.path.x, [9, 4.5, 2.25], atol=1e-10)
    assert paths[0].input_fingerprint != paths[1].input_fingerprint
    assert paths[1].effective_options["tol"] == 1e-11
    assert paths[0].effective_options["input_state"]["prepared"]["shocks"] == [[1], [0], [0]]
    results.perfect_foresight.path.iloc[0, 0] = -100
    assert paths[1].result.path.x.iloc[0] == pytest.approx(9)


@pytest.mark.parametrize("terminal_command,expected_terminal", [("", 6), ("steady;", 2)])
def test_literal_and_solved_terminal_conditions_for_a_forward_only_model(terminal_command, expected_terminal):
    model = read_mod(f"""
        var x y; varexo e;
        model; x=.5*x(+1)+e; y=STEADY_STATE(x); end;
        initval; x=0; y=0; e=0; end; steady;
        endval; x=6; y=6; e=1; end; {terminal_command}
        perfect_foresight_setup(periods=4); perfect_foresight_solver;
    """)
    result = model.execute_mod_commands().perfect_foresight
    expected = 2 + (expected_terminal - 2) * .5 ** np.arange(4, 0, -1)
    np.testing.assert_allclose(result.path.x, expected, atol=1e-9)
    np.testing.assert_allclose(result.path.y, expected_terminal, atol=1e-9)


def test_endval_steady_option_and_partial_endval_blocks():
    model = read_mod("""
        var x y; varexo e; model; x=.5*x(-1)+e; y=STEADY_STATE(x); end;
        initval; x=0; y=0; e=0; end; steady;
        endval; e=1; end; endval; x=20; end;
        perfect_foresight_setup(periods=3,endval_steady);
        perfect_foresight_solver;
    """)
    result = model.execute_mod_commands()
    assert result.endval == pytest.approx({"x": 2, "y": 2, "e": 1})
    np.testing.assert_allclose(result.perfect_foresight.path.x, [1, 1.5, 1.75], atol=1e-9)
    np.testing.assert_allclose(result.perfect_foresight.path.y, 2, atol=1e-9)


def test_histval_and_new_initval_experiments_keep_source_order():
    result = read_mod(BASE + """
        initval; x=0; e=0; end; histval; x=2; end;
        simul(periods=3);
        initval; x=4; e=0; end;
        shocks(overwrite); var e; periods 2; values 1; end;
        simul(periods=3,linear_solver='krylov');
    """).execute_mod_commands()
    first, second = result.invocations
    np.testing.assert_allclose(first.result.path.x, [1, .5, .25], atol=1e-10)
    np.testing.assert_allclose(second.result.path.x, [2, 2, 1], atol=1e-10)
    assert second.result.linear_solver == "krylov"


@pytest.mark.parametrize("loader", [read_mod, lambda s: load_mod(s, command_mode="ordered")])
def test_legacy_loaded_models_can_use_the_same_ordered_runner(loader):
    model = loader(BASE + "initval; x=0; e=0; end; steady; shocks; var e; periods 1; values 1; end; simul(periods=3);")
    result = model.execute_mod_commands(ordered=True)
    np.testing.assert_allclose(result.perfect_foresight.path.x, [1, .5, .25], atol=1e-10)


@pytest.mark.parametrize("name", ["endval_literal", "endval_solved", "endval_no_steady", "endval_with_steady"])
@pytest.mark.parametrize("backend", ["direct", "krylov"])
def test_ordered_paths_match_frozen_dynare_terminal_fixtures(name, backend):
    fixtures = Path(__file__).resolve().parents[1] / "fixtures/dynare_issue_177"
    source = (fixtures / f"{name}.mod").read_text().replace(", tolx=1e-12", "")
    source = source.replace("tolf=1e-12", f"tolf=1e-12,linear_solver='{backend}'")
    result = read_mod(source).execute_mod_commands().perfect_foresight
    reference = pd.read_csv(fixtures / f"{name}_dynare_out.csv").iloc[1:11]
    np.testing.assert_allclose(result.path, reference, rtol=0, atol=1e-9)


@pytest.mark.parametrize("tail", [
    "perfect_foresight_solver;",
    "perfect_foresight_setup(periods=3); perfect_foresight_solver(stack_solve_algo=2);",
    "perfect_foresight_setup(periods=3); perfect_foresight_solver(tol=1e-8,tolf=1e-8);",
    "perfect_foresight_setup(periods=3); perfect_foresight_solver(tolx=1e-8);",
    "perfect_foresight_setup(periods=3); perfect_foresight_solver(periods=4);",
    "perfect_foresight_setup(periods=3); perfect_foresight_solver(linear_solver='krylov',mcp);",
    "histval; x=1; end; endval; x=2; end; simul(periods=3);",
    "simul(periods=3); rho=.7; simul;",
    "histval; x(-1)=2; end; simul(periods=3);",
    "initval; rho=.9; x=0; end; simul(periods=3);",
])
def test_unsupported_programs_are_rejected_before_execution(tail):
    source = BASE + "initval; x=0; e=0; end;" + tail
    report = inspect_mod(source, ordered=True)
    assert report.exit_code != 0
    model = read_mod(source)
    with pytest.raises(ModCompatibilityError):
        model.execute_mod_commands()


def test_failed_steady_state_retains_the_failed_invocation():
    model = read_mod("var x; model; x^2+1=0; end; initval; x=0; end; steady; simul(periods=3);")
    with pytest.raises(ModExecutionError, match="verified equilibrium") as caught:
        model.execute_mod_commands()
    result = caught.value.results
    assert len(result.invocations) == 1 and result.invocations[0].status == "failed"
    assert result.commands_executed == ()
    assert "SteadyStateError" in result.invocations[0].error


def test_mcp_equation_tags_are_forwarded_with_explicit_native_option():
    result = read_mod("""
        var x r; varexo e; parameters floor; floor=0;
        model; [mcp='r > floor'] r=-1+e; x=r; end;
        initval; x=0; r=0; e=0; end;
        simul(periods=3,mcp);
    """).execute_mod_commands().perfect_foresight
    assert result.converged
    np.testing.assert_allclose(result.path, 0, atol=1e-9)
