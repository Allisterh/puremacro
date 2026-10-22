"""Read-only inspection and fixed-calibration command-program regressions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import socket

import numpy as np
import pandas as pd
import pytest

from puremacro.dsge import (
    inspect_mod, load_mod, LinearModel, ModCompatibilityError, ModExecutionError,
)
from puremacro.dsge._ast import BinOp, Call, Const, Param, UnaryOp, Var
from puremacro.dsge._parser import ParsedModelDAG
from puremacro.dsge.cli import run_cli

BASE = """var y x;
varexo e u;
parameters rho;
rho=.8;
model(linear);
 #r = rho;
 [name='y; equation'] y=r*y(-1)+e;
 x=.4*x(-1)+u;
end;
initval; y=0; x=0; end;
shocks; var e; stderr 1; var u; stderr 1; end;
"""


def codes(report):
    return {i.code for i in report.issues}


def forbidden(*args, **kwargs):
    raise AssertionError("inspection entered a numerical/network boundary")


@pytest.mark.mechanism_control
def test_sentinel_positive_control(monkeypatch):
    monkeypatch.setattr(Const, "eval", forbidden)
    with pytest.raises(AssertionError, match="numerical/network"):
        Const(1).eval({}, {})
    monkeypatch.setattr(socket, "socket", forbidden)
    with pytest.raises(AssertionError, match="numerical/network"):
        socket.socket()


def test_inspection_never_evaluates_compiles_solves_or_changes_inputs(monkeypatch, tmp_path):
    import puremacro.dsge.dynare as dynare
    for cls in (BinOp, Call, Const, Param, UnaryOp, Var):
        monkeypatch.setattr(cls, "eval", forbidden)
        monkeypatch.setattr(cls, "diff", forbidden)
    monkeypatch.setattr(ParsedModelDAG, "compile_equations", forbidden)
    monkeypatch.setattr(dynare, "load_mod", forbidden)
    monkeypatch.setattr(dynare, "parse_mod", forbidden)
    for method in ("stoch_simul", "smoother", "estimate", "shock_decomposition"):
        monkeypatch.setattr(LinearModel, method, forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    source = tmp_path / "model.mod"
    source.write_text(BASE + "steady_state_model; y=log(-1); x=1/0; end;\nstoch_simul(order=1) y;")
    before = source.read_bytes()
    rng_before = np.random.get_state()
    report = inspect_mod(source, ordered=True)
    assert report.exit_code == 0
    assert report.declarations == dict(variables=2, shocks=2, parameters=1, equations=2, model_locals=1, observables=0)
    assert source.read_bytes() == before
    assert report.source_hashes[str(source)] == hashlib.sha256(before).hexdigest()
    after = np.random.get_state()
    assert rng_before[0] == after[0] and np.array_equal(rng_before[1], after[1]) and rng_before[2:] == after[2:]
    assert [s.category for s in report.statements].count("equation") == 2
    assert any("y; equation" in s.excerpt for s in report.statements if s.category == "equation")


def test_inventory_names_skipped_script_and_options():
    script = """for i=1:2
 if i>1
  disp('inside; quote');
 end
end;
made_up(irf=3);
stoch_simul(order=1, irrf=8) y;
"""
    report = inspect_mod(BASE + script)
    assert report.coverage == "complete"
    scripts = [s for s in report.statements if s.category == "external/script statement"]
    assert [s.name for s in scripts] == ["for", "made_up"]
    assert "inside; quote" in scripts[0].excerpt
    assert {"UNSUPPORTED_STATEMENT", "UNSUPPORTED_OPTION", "VARIABLE_SELECTION_IGNORED"} <= codes(report)
    assert report.exit_code == 1


def test_unknown_block_and_expression_do_not_disappear():
    report = inspect_mod(BASE + "unknown_block; a=unknown_fn(rho); end; stoch_simul;")
    assert {"UNSUPPORTED_STATEMENT", "UNSUPPORTED_FUNCTION"} <= codes(report)


def test_expectation_is_named_without_compilation():
    report = inspect_mod(BASE.replace("r*y(-1)", "EXPECTATION(-1)(y)"))
    assert "EXPECTATION_UNSUPPORTED" in codes(report)


def test_legacy_collapsing_and_load_only_reported():
    source = BASE + "stoch_simul(order=1,irf=3) y; stoch_simul(irf=7) x;"
    report = inspect_mod(source)
    assert "REPEATED_COMMAND_COLLAPSED" in codes(report)
    assert len([s for s in report.statements if s.name == "stoch_simul"]) == 2
    load_report = inspect_mod(source, entry_point="load_mod")
    assert "COMMAND_NOT_EXECUTED" in codes(load_report)
    assert "UNSUPPORTED_OPTION" in codes(load_report)


def test_macro_include_loop_provenance_and_unavailable_columns(tmp_path):
    inc = tmp_path / "part.mod"
    inc.write_text("@#for v in [\"y\",\"x\"]\nvar @{v};\n@#endfor\n")
    source = tmp_path / "main.mod"
    source.write_text('@#include "part.mod"\nvarexo e;\nmodel; y=.5*y(-1)+e; x=.3*x(-1); end;\nstoch_simul(order=1);\n')
    before = {p: p.read_bytes() for p in (source, inc)}
    report = inspect_mod(source, ordered=True)
    assert report.exit_code == 0
    declarations = [s for s in report.statements if s.name == "var"]
    assert len(declarations) == 2
    assert {s.location.line for s in declarations} == {2}
    assert {s.location.source for s in declarations} == {str(inc)}
    assert all(s.location.column is None for s in declarations)
    assert declarations[0].location.expansion_instance != declarations[1].location.expansion_instance
    assert declarations[0].location.include_chain == (str(source), str(inc))
    assert set(report.source_hashes) == {str(source), str(inc)}
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize("root_newline", ["\n", "\r\n", "\r"], ids=["lf", "crlf", "cr"])
@pytest.mark.parametrize("include_newline", ["\n", "\r\n", "\r"], ids=["lf", "crlf", "cr"])
def test_file_hashes_preserve_bytes_across_newlines(tmp_path, root_newline, include_newline):
    inc = tmp_path / "part.mod"
    source = tmp_path / "main.mod"
    inc_text = '// résumé\n@#for v in ["y","x"]\nvar @{v};\n@#endfor\n'
    root_text = ('@#include "part.mod"\nvarexo e;\n'
                 'model; y=.5*y(-1)+e; x=.3*x(-1); end;\nstoch_simul(order=1);\n')
    originals = {
        inc: inc_text.replace("\n", include_newline).encode("utf-8"),
        source: root_text.replace("\n", root_newline).encode("utf-8"),
    }
    for path, raw in originals.items():
        path.write_bytes(raw)
    report = inspect_mod(source, ordered=True)
    assert report.exit_code == 0
    assert report.source_hashes == {
        str(path): hashlib.sha256(raw).hexdigest() for path, raw in originals.items()
    }
    assert {path: path.read_bytes() for path in originals} == originals
    declarations = [s for s in report.statements if s.name == "var"]
    assert len(declarations) == 2
    assert {s.location.line for s in declarations} == {3}
    assert {s.location.source for s in declarations} == {str(inc)}
    assert all(s.location.include_chain == (str(source), str(inc)) for s in declarations)
    equations = [s for s in report.statements if s.category == "equation"]
    assert len(equations) == 2
    assert {s.location.line for s in equations} == {3}
    assert {s.location.source for s in equations} == {str(source)}
    command = next(s for s in report.statements if s.name == "stoch_simul")
    assert (command.location.line, command.location.column) == (4, 1)


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"], ids=["lf", "crlf", "cr"])
def test_inline_macro_only_source_preserves_hash_and_line_numbers(newline):
    source = '@#define n = 1\n@#define m = 2\n'.replace("\n", newline)
    report = inspect_mod(source)
    assert report.coverage == "complete"
    assert report.source_hashes == {"<text>": hashlib.sha256(source.encode("utf-8")).hexdigest()}
    assert codes(report) == {"MISSING_MODEL"}
    assert [s.location.line for s in report.statements] == [1, 2]


@pytest.mark.parametrize("tail", ['@#include "missing.mod"', '@#for i in 1:1000000000\nvar y@{i};\n@#endfor', '@#define huge = [1,2]^10000', '@#define huge = "x" * 1000000000'])
def test_failed_or_excessive_macro_expansion_is_incomplete(tail, tmp_path):
    report = inspect_mod(BASE + tail + "\n", base_dir=tmp_path, macro_limit=100)
    assert report.coverage == "incomplete" and report.exit_code == 2
    assert "INSPECTION_INCOMPLETE" in codes(report)


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"], ids=["lf", "crlf", "cr"])
def test_nested_failed_include_preserves_read_hashes(tmp_path, newline):
    originals = {
        tmp_path / "a.mod": ('@#include "b.mod"' + newline).encode("utf-8"),
        tmp_path / "b.mod": ('@#include "missing.mod"' + newline).encode("utf-8"),
    }
    for path, raw in originals.items():
        path.write_bytes(raw)
    report = inspect_mod(tmp_path / "a.mod")
    assert report.exit_code == 2
    assert report.source_hashes == {
        str(path): hashlib.sha256(raw).hexdigest() for path, raw in originals.items()
    }
    assert {path: path.read_bytes() for path in originals} == originals


def test_macro_unary_and_inactive_branch_inventory():
    report = inspect_mod('@#define sign = -1\n@#if 0\n@#include "absent.mod"\n@#endif\n' + BASE)
    assert report.exit_code == 0
    assert {s.name for s in report.statements if s.category == "macro directive"} == {"define", "if", "include", "endif"}


@pytest.mark.parametrize("source", ["var y; model; y=1;", BASE + "shocks; var e; stderr 1;", BASE + "@#if 1\n"])
def test_truncated_input_cannot_be_clear(source):
    assert inspect_mod(source).exit_code == 2


def test_hank_extension_requires_correct_route_and_context():
    source = BASE + 'hetagent_block; model="one_asset_hank"; end;'
    assert "HANK_EXTENSION" in codes(inspect_mod(source))
    assert "HANK_EXTENSION" not in codes(inspect_mod(source, entry_point="load_hank_mod"))
    assert "HANK_EXTENSION" not in codes(inspect_mod(source, entry_point="load_hank_mod", household_model=object()))


def test_data_requirements_and_schema_are_separate():
    source = BASE + "varobs y x; calib_smoother(prefilter=1);"
    assert "DATA_REQUIRED" in codes(inspect_mod(source, ordered=True))
    report = inspect_mod(source, ordered=True, data=pd.DataFrame({"y": [0]}))
    assert "DATA_SCHEMA" in codes(report)
    assert "DATA_REQUIRED" not in codes(report)
    valid = inspect_mod(source, ordered=True, data=pd.DataFrame({"y": [0], "x": [0]}))
    assert valid.exit_code == 0
    assert "likelihood not evaluated" in valid.data_check


def test_cli_deterministic_json_text_and_explicit_output(tmp_path, capsys):
    source = tmp_path / "rbc.mod"
    source.write_text(BASE + "stoch_simul(order=1);")
    assert run_cli(["inspect", str(source), "--ordered", "--format", "json"]) == 0
    output = capsys.readouterr().out
    assert json.loads(output) == inspect_mod(source, ordered=True).to_dict()
    assert run_cli(["inspect", str(source), "--ordered", "--format", "json"]) == 0
    assert capsys.readouterr().out == output
    target = tmp_path / "report.txt"
    assert run_cli(["inspect", str(source), "--ordered", "--output", str(target)]) == 0
    assert target.read_text().strip() == inspect_mod(source, ordered=True).summary()
    assert capsys.readouterr().out == ""
    assert run_cli(["inspect", str(tmp_path / "missing.mod")]) == 2


def test_ordered_simulations_match_direct_calls_and_select_presentation():
    source = BASE + "stoch_simul(order=1,irf=3,ar=1) y; stoch_simul(order=1,irf=7) x;"
    model = load_mod(source, command_mode="ordered")
    result = model.execute_mod_commands(ordered=True)
    assert result.commands_executed == ("stoch_simul", "stoch_simul")
    assert len(result.invocations) == 2
    for invocation, horizon, variable in zip(result.invocations, (3, 7), ("y", "x")):
        direct = model.stoch_simul(order=1, irf=horizon, ar=1)
        assert invocation.status == "success"
        assert invocation.requested_variables == (variable,)
        assert invocation.result.variable_names == direct.variable_names
        assert invocation.selected_result.variable_names == (variable,)
        pd.testing.assert_frame_equal(invocation.result.to_frame(), direct.to_frame())
        assert invocation.result.dr is invocation.selected_result.dr
        assert set(invocation.selected_result.irfs) == {f"{variable}_e", f"{variable}_u"}
        assert invocation.effective_options["ar"] == 1
        assert len(invocation.input_fingerprint) == 64
    assert result.stoch_simul is result.invocations[-1].result
    assert result.to_frame()["invocation"].tolist() == [0, 1]
    assert len(result.invocations[0].result.irfs["y_e"]) == 4


def test_option_lifetimes_omitted_order_resets_selection_resets_and_context_isolation():
    source = BASE + "stoch_simul(order=1,irf=3,ar=1) y; stoch_simul(irf=7); stoch_simul(order=1,ar=5) x;"
    report = inspect_mod(source, ordered=True)
    commands = [s for s in report.statements if s.name == "stoch_simul"]
    assert [s.effective_options["order"] for s in commands] == [1, 2, 1]
    assert [s.effective_options["ar"] for s in commands] == [1, 1, 5]
    assert [s.effective_options["irf"] for s in commands] == [3, 7, 7]
    model = load_mod(source, command_mode="ordered")
    result = model.execute_mod_commands(ordered=True)
    assert result.invocations[1].selected_result.variable_names == ("y", "x")
    assert result.invocations[1].result.order == 2
    fresh = load_mod(BASE + "stoch_simul(order=1,irf=0);", command_mode="ordered").execute_mod_commands(ordered=True)
    assert fresh.invocations[0].effective_options["ar"] == 5


def test_legacy_mode_preserves_aggregation():
    model = load_mod(BASE + "stoch_simul(order=1,irf=3,ar=1) y; stoch_simul(order=1,irf=7) x;")
    legacy = model.execute_mod_commands()
    assert legacy.commands_executed == ("stoch_simul",) and legacy.invocations == ()
    assert legacy.stoch_simul.variable_names == ("y", "x")
    assert len(legacy.stoch_simul.irfs["y_e"]) == 8


@pytest.mark.parametrize("tail", [
    "stoch_simul(order=1); rho=.5; stoch_simul(order=1);",
    "stoch_simul(order=1); shocks; var e; stderr 2; end;",
    "stoch_simul(order=1); perfect_foresight_solver;",
    "osr; stoch_simul(order=1);",
    "stoch_simul(order=1,unknown=2);", "stoch_simul(order=1,irf=-1);",
    "stoch_simul(order=1,nograph=0);",
])
def test_unsupported_sequences_fail_before_parse_mod(monkeypatch, tail):
    import puremacro.dsge.dynare as dynare
    monkeypatch.setattr(dynare, "parse_mod", forbidden)
    with pytest.raises(ModCompatibilityError) as caught:
        dynare.load_mod(BASE + tail, command_mode="ordered")
    assert caught.value.report.exit_code != 0
    assert any(i.location is not None for i in caught.value.report.issues)


def test_smoother_and_decomposition_options_data_and_selections():
    data = pd.DataFrame({"y": np.linspace(-.1, .2, 12), "x": np.linspace(.2, -.1, 12)})
    source = BASE + "varobs y x; calib_smoother(prefilter=1,first_obs=2,nobs=5) y; calib_smoother(prefilter=0); shock_decomposition x;"
    model = load_mod(source, command_mode="ordered")
    result = model.execute_mod_commands(ordered=True, data=data)
    assert result.commands_executed == ("calib_smoother", "calib_smoother", "shock_decomposition")
    frame = data.iloc[1:6]
    direct = model.smoother(frame, prefilter=True)
    pd.testing.assert_frame_equal(result.invocations[0].result.smoothed_obs, direct.smoothed_obs)
    assert list(result.invocations[0].selected_result.smoothed_obs) == ["y"]
    pd.testing.assert_frame_equal(result.smoother.smoothed_obs, model.smoother(frame).smoothed_obs)
    pd.testing.assert_frame_equal(result.shock_decomposition.components["x"], model.shock_decomposition(frame).components["x"])
    assert result.invocations[2].effective_options["first_obs"] == 2
    assert result.invocations[2].selected_result.variable_names == ["x"]


def test_inherited_prefilter_decomposition_rejected_before_loading():
    with pytest.raises(ModCompatibilityError, match="inherits prefilter"):
        load_mod(BASE + "varobs y x; calib_smoother(prefilter=1); shock_decomposition;", command_mode="ordered")


def test_missing_data_prevents_all_commands(monkeypatch):
    model = load_mod(BASE + "varobs y x; stoch_simul(order=1); calib_smoother;", command_mode="ordered")
    monkeypatch.setattr(LinearModel, "stoch_simul", forbidden)
    with pytest.raises(ModCompatibilityError, match="DATA_REQUIRED"):
        model.execute_mod_commands(ordered=True)
    with pytest.raises(ValueError, match="Missing observable"):
        model.execute_mod_commands(ordered=True, data=pd.DataFrame({"y": [1]}))


def test_failure_retains_completed_entries(monkeypatch):
    model = load_mod(BASE + "stoch_simul(order=1,irf=3); stoch_simul(order=1,irf=7);", command_mode="ordered")
    original = LinearModel.stoch_simul
    def fail_second(self, *args, **kwargs):
        if kwargs.get("irf") == 7:
            raise ValueError("deliberate numerical boundary failure")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(LinearModel, "stoch_simul", fail_second)
    with pytest.raises(ModExecutionError) as caught:
        model.execute_mod_commands(ordered=True)
    partial = caught.value.results
    assert partial.commands_executed == ("stoch_simul",)
    assert [i.status for i in partial.invocations] == ["success", "failed"]
    assert partial.stoch_simul is partial.invocations[0].result
    assert partial.invocations[1].result is None
    assert "deliberate" in partial.invocations[1].error

@pytest.mark.parametrize("tail", ["/* unclosed", "made_up('unclosed);", "stoch_simul(irf=1+2);", "shocks; wobble e; end;"])
def test_malformed_lexing_or_ignored_tokens_are_incomplete(tail):
    assert inspect_mod(BASE + tail).exit_code == 2


def test_nonfinite_options_serialize_without_false_success():
    report = inspect_mod(BASE + "stoch_simul(irf=1e999);", ordered=True)
    assert "UNSUPPORTED_OPTION_VALUE" in codes(report)
    decoded = json.loads(report.to_json())
    assert decoded["exit_code"] == 1
    assert "nonfinite_number" in report.to_json()


def test_exogenous_steady_state_and_repeated_calibration_are_reported():
    assert "STEADY_STATE_EXOGENOUS" in codes(inspect_mod(BASE.replace("r*y(-1)", "STEADY_STATE(e)")))
    assert "REPEATED_CALIBRATION_ASSIGNMENT" in codes(inspect_mod(BASE + "rho=.4; stoch_simul;", ordered=True))


def test_non_numeric_data_rejected_before_first_simulation(monkeypatch):
    model = load_mod(BASE + "varobs y x; stoch_simul(order=1); calib_smoother;", command_mode="ordered")
    monkeypatch.setattr(LinearModel, "stoch_simul", forbidden)
    with pytest.raises(ValueError, match="numeric"):
        model.execute_mod_commands(ordered=True, data=pd.DataFrame({"y": ["bad"], "x": [1]}))
    with pytest.raises(ValueError, match="infinity"):
        model.execute_mod_commands(ordered=True, data=pd.DataFrame({"y": [np.inf], "x": [1]}))


def test_cached_include_snapshot_is_used_for_execution(tmp_path):
    inc = tmp_path / "program.mod"
    inc.write_text(BASE + "stoch_simul(order=1,irf=3) y;")
    root = tmp_path / "main.mod"
    root.write_text('@#include "program.mod"\n')
    model = load_mod(root, command_mode="ordered")
    inc.write_text("not the loaded model;")
    result = model.execute_mod_commands(ordered=True)
    assert len(result.stoch_simul.irfs["y_e"]) == 4
    assert result.invocations[0].source.source == str(inc)


def test_frozen_dynare_driver_option_lifetimes():
    folder = Path(__file__).resolve().parents[1] / "fixtures" / "dynare_commands"
    corpus = json.loads((folder / "lifetimes.json").read_text())
    report = inspect_mod(folder / "lifetimes.mod", ordered=True)
    commands = [s for s in report.statements if s.name in {"stoch_simul", "calib_smoother", "shock_decomposition"}]
    assert len(commands) == len(corpus["expected"])
    for command, expected in zip(commands, corpus["expected"]):
        assert command.name == expected["command"]
        for key, value in expected.items():
            if key == "variables":
                assert list(command.variables) == value
            elif key != "command":
                assert command.effective_options[key] == value
    assert "INHERITED_OPTION_UNSUPPORTED" in codes(report)

@pytest.mark.parametrize("tail", [
    "stoch_simul(order=);", "stoch_simul(order=,irf=3);",
    "stoch_simul(order=1,order=2);",
    "initval; y=not_a_function(1); end;", "steady_state_model; y=not_a_function(1); end;",
    "non_parameter=1;", "varobs bogus; stoch_simul(order=1); calib_smoother;",
    "varobs y; varobs y;", "stoch_simul(order=1) y y;",
    "shocks(bogus); var e; stderr 1; end;", "shocks; var bogus; stderr 2; end;",
    "initval; bogus=1; end;", "shocks; var e; periods 1; values 5; end; stoch_simul(order=1);",
])
def test_adversarial_semantic_inventory_never_reports_clear(tail):
    report = inspect_mod(BASE + tail, ordered=True)
    assert report.exit_code != 0
    assert report.issues


@pytest.mark.parametrize("header", ["var(log) y x;", "var(bogus=1) y x;"])
def test_unconsumed_var_declaration_options_are_reported(header):
    report = inspect_mod(BASE.replace("var y x;", header), ordered=True)
    assert "UNSUPPORTED_DECLARATION_OPTION" in codes(report)


def test_indirect_steady_state_shock_and_math_arity():
    assert "STEADY_STATE_EXOGENOUS" in codes(inspect_mod(BASE.replace("#r = rho;", "#r = e;").replace("r*y(-1)", "STEADY_STATE(r)")))
    assert "UNSUPPORTED_FUNCTION_ARITY" in codes(inspect_mod(BASE.replace("r*y(-1)", "exp(y,y)")))


def test_reversing_independent_commands_reverses_entries_and_preserves_options():
    first = "stoch_simul(order=1,irf=3,ar=1) y;"
    second = "stoch_simul(order=1,irf=7,ar=2) x;"
    forward = load_mod(BASE+first+second, command_mode="ordered").execute_mod_commands(ordered=True)
    reverse = load_mod(BASE+second+first, command_mode="ordered").execute_mod_commands(ordered=True)
    for lhs, rhs in zip(forward.invocations, reversed(reverse.invocations)):
        assert lhs.requested_variables == rhs.requested_variables
        assert lhs.explicit_options == rhs.explicit_options
        pd.testing.assert_frame_equal(lhs.result.to_frame(), rhs.result.to_frame())


def test_selected_report_does_not_overwrite_earlier_results():
    result = load_mod(BASE + "stoch_simul(order=1,irf=3) y; stoch_simul(order=1,irf=3) x;",
                      command_mode="ordered").execute_mod_commands(ordered=True)
    original = result.invocations[0].result.irfs["y_e"].copy()
    result.invocations[1].result.irfs["y_e"].iloc[:] = 99
    pd.testing.assert_series_equal(original, result.invocations[0].result.irfs["y_e"])


def test_inspection_does_not_import_plotting_in_fresh_process():
    import subprocess
    import sys
    script = "from puremacro.dsge import inspect_mod; import sys; inspect_mod('var y; varexo e; model; y=.5*y(-1)+e; end;'); assert not any(k.startswith('matplotlib') for k in sys.modules)"
    subprocess.run([sys.executable, "-c", script], check=True)

@pytest.mark.parametrize("definition", ["bad=bad", "bad=other", "bad=exp(1,2)", "bad=gamma(2)"])
def test_unused_model_locals_are_also_inspected(definition):
    report = inspect_mod(BASE.replace("#r = rho;", f"#r=rho; #{definition};"), ordered=True)
    assert report.exit_code != 0
    assert any(i.location and i.location.line == 6 for i in report.issues)


def test_state_dependent_calibration_is_outside_fixed_subset():
    assert "STATE_DEPENDENT_CALIBRATION" in codes(inspect_mod(BASE.replace("rho=.8;", "rho=y;"), ordered=True))


def test_unconsumed_hank_fields_remain_visible():
    report = inspect_mod(BASE + "hetagent_block; grid=hyperbolic; end;", entry_point="load_hank_mod")
    assert "HANK_OPTION_NOT_CONSUMED" in codes(report)
