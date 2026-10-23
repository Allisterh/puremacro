"""DP file declarations: independent Bellman checks, compiler parity and rejection."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from puremacro import dp


HOUSEHOLD = """
parameters beta gamma r rho sigma n;
beta = .95; gamma = 2; r = .02; rho = .5; sigma = .1; n = 30;
state a = 10*linspace(0, 1, n)^2;
exogenous z = rouwenhorst(rho=rho, sigma=sigma, n=3);
model(name='saving household');
  c = (1+r)*a + exp(z) - a(+1);
  reward = crra(c, gamma);
end;
constraints; c > 0; end;
aggregates; A = a; C = c; end;
solve(method=egm, tol=1e-9);
"""


def _household():
    m = dp.Model("saving household").parameters(beta=.95, gamma=2, r=.02,
                                               rho=.5, sigma=.1, n=30)
    m.state("a", 10*np.linspace(0, 1, 30)**2)
    m.exogenous("z", dp.AR1(rho=.5, sigma=.1, n=3, method="rouwenhorst"))
    m.local("c = (1+r)*a + exp(z) - a(+1)")
    m.reward("crra(c, gamma)").subject_to("c > 0")
    m.aggregate(A="a", C="c")
    return m


@pytest.mark.parametrize("method", ["vfi", "egm"])
@pytest.mark.pyodide_smoke
def test_solved_file_matches_python_model(method):
    sol = dp.solve_mod(HOUSEHOLD, method=method)
    ref = _household().solve(method=method, tol=1e-9)
    np.testing.assert_array_equal(sol.policy("a(+1)"), ref.policy("a(+1)"))
    np.testing.assert_array_equal(sol.V, ref.V)
    np.testing.assert_array_equal(sol.distribution, ref.distribution)
    assert sol.aggregates == ref.aggregates


def test_vfi_file_satisfies_independent_bellman_maximum():
    # Two states and a contemporaneous discrete option; independently enumerate
    # the choices and evaluate the Bellman maximum, without the dp compiler.
    text = """
    parameters beta; beta=.8;
    state a = [0, 1, 2]; state b = [0, 1]; discrete h = [0, 1];
    exogenous z = markov([.5, 1.5], [.8,.2; .3,.7]);
    model;
      c = 1 + z + .9*a + .8*b + .4*h - a(+1) - b(+1);
      reward = log(c) - .2*h;
    end;
    constraints; c > 0; end;
    solve(method=vfi,tol=1e-11);
    """
    sol = dp.solve_mod(text)
    # dp flattens state grids with last-declared state varying fastest.
    states = [(a, b) for a in [0, 1, 2] for b in [0, 1]]
    P = np.array([[.8, .2], [.3, .7]])
    for i, (a, b) in enumerate(states):
        for j, z in enumerate([.5, 1.5]):
            values = []
            for k, (ap, bp) in enumerate(states):
                for h in [0, 1]:
                    c = 1 + z + .9*a + .8*b + .4*h - ap - bp
                    if c > 0:
                        values.append(np.log(c) - .2*h + .8*(P[j] @ sol.V[k]))
            assert sol.V[i, j] == pytest.approx(max(values), abs=2e-10)


def test_load_never_solves_and_validates_last_command(monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("solver was called")
    monkeypatch.setattr(dp.Model, "solve", forbidden)
    model = dp.load_mod(HOUSEHOLD)
    assert model.name == "saving household"
    with pytest.raises(dp.ModelSpecError, match="unsupported|unknown|expected"):
        dp.load_mod(HOUSEHOLD.replace("solve(method=egm, tol=1e-9)", "system('touch /tmp/forbidden')"))


def test_parameter_overrides_recompute_dependent_calibration_and_grid():
    text = HOUSEHOLD.replace("n = 30", "n = 10*gamma")
    model = dp.load_mod(text, params={"gamma": 3})
    assert model._params["gamma"] == 3
    assert model._params["n"] == 30
    assert len(model._states[0][1]) == 30
    with pytest.raises(dp.ModelSpecError, match="override names"):
        dp.load_mod(text, params={"typo": 3})


@pytest.mark.parametrize("ending", ["\n", "\r\n", "\r"])
def test_file_bytes_preserve_provenance(tmp_path, ending):
    path = tmp_path / "household.mod"
    raw = HOUSEHOLD.replace("\n", ending).encode()
    path.write_bytes(raw)
    m = dp.load_mod(path)
    assert m._mod_source["sha256"] == hashlib.sha256(raw).hexdigest()
    assert dp.load_mod(str(path)).name == m.name


def test_comments_and_strings_do_not_split_statements():
    text = HOUSEHOLD.replace("model(name='saving household')", "model(name='saving; // household')")
    text = "% heading\n" + text.replace("state a", "/* ignored ; end;\n */ state a")
    assert dp.load_mod(text).name == "saving; // household"


def test_escaped_quotes_preserve_literal_title():
    title = r"model(name='household\'s ^ [a;b]')"
    assert dp.load_mod(HOUSEHOLD.replace("model(name='saving household')", title)).name == "household's ^ [a;b]"


@pytest.mark.parametrize("expr, message", [
    ("linspace([0,1], [1,2], 1000000)", "endpoints must be scalars"),
    ("full(1000000, [1,2])", "scalar fill value"),
    ("max([full(2000,0)], full(2000,1))", None),
])
def test_numeric_allocation_boundaries(expr, message):
    from puremacro.dp.modfile import _Numbers
    if message is None:
        assert _Numbers({}).evaluate(expr).shape == (1, 2000)
    else:
        with pytest.raises(dp.ModelSpecError, match=message):
            _Numbers({}).evaluate(expr)


def test_broadcast_function_rejects_large_output_before_call():
    from puremacro.dp.modfile import _Numbers
    numbers = _Numbers({"column": np.ones((2000, 1)), "row": np.ones(2000)})
    with pytest.raises(dp.ModelSpecError, match="too large"):
        numbers.evaluate("max(column,row)")


def test_nested_array_children_cannot_accumulate_unbounded_temporaries(monkeypatch):
    import puremacro.dp.modfile as reader
    monkeypatch.setattr(reader, "_MAX_ARRAY", 8)
    calls = []
    real_full = reader._full

    def counted_full(n, value):
        calls.append(n)
        return real_full(n, value)

    monkeypatch.setitem(reader._NUMERIC_FUNCTIONS, "full", counted_full)
    with pytest.raises(dp.ModelSpecError, match="too large"):
        reader._Numbers({}).evaluate("[" + ",".join(["full(4,0)"]*20) + "]")
    assert calls == [4, 4, 4]  # Stops at the first oversized child, before collecting all twenty.
    np.testing.assert_array_equal(reader._Numbers({}).evaluate("[full(4,0),full(4,1)]"), [[0]*4, [1]*4])


@pytest.mark.parametrize("declaration", ["n = {expr}", "solve(method=egm,tol={expr})"])
def test_deep_numeric_and_declaration_expressions_have_normalized_errors(declaration):
    deep = declaration.format(expr="-"*1100+"1")
    text = HOUSEHOLD.replace("n = 30", deep) if declaration.startswith("n") else HOUSEHOLD.replace("solve(method=egm, tol=1e-9)", deep)
    with pytest.raises(dp.ModelSpecError, match="deeply nested"):
        dp.load_mod(text)


def test_cli_deep_expression_is_a_clean_error(tmp_path, capsys):
    from puremacro.dp.cli import main
    path = tmp_path / "deep.mod"
    path.write_text(HOUSEHOLD.replace("n = 30", "n = " + "-"*1100 + "1"))
    assert main([str(path), "--check"]) == 1
    error = capsys.readouterr().err
    assert "deeply nested" in error and "Traceback" not in error


@pytest.mark.parametrize("option", ["backend=cupy", "howard=17", "xtol=1e-7", "max_evals=20"])
def test_explicit_inapplicable_egm_file_options_rejected(option):
    with pytest.raises(dp.ModelSpecError, match="do not apply"):
        dp.load_mod(HOUSEHOLD.replace("tol=1e-9", option))


@pytest.mark.parametrize("option", ["backend=numpy", "howard=20"])
def test_explicit_inapplicable_hjb_file_options_rejected(option):
    source = f"""parameters rho; rho=.05;
    state a=linspace(0,5,20); control c; drift a=1+.02*a-c; discount_rate rho;
    model; reward=log(c); end; solve(method=hjb,{option});"""
    with pytest.raises(dp.ModelSpecError, match="do not apply"):
        dp.load_mod(source)


def test_finite_horizon_and_equilibrium_cannot_ignore_explicit_options():
    finite = HOUSEHOLD.replace("model(name=", "horizon=3; model(name=")
    with pytest.raises(dp.ModelSpecError, match="do not apply"):
        dp.load_mod(finite)
    with pytest.raises(dp.ModelSpecError, match="distribution=False"):
        dp.load_mod(finite.replace("tol=1e-9", "distribution=false"))
    with pytest.raises(dp.ModelSpecError, match="positive"):
        dp.solve_mod(HOUSEHOLD, tol=True)


def test_final_method_override_controls_option_applicability():
    text = HOUSEHOLD.replace("tol=1e-9", "backend=numpy")
    with pytest.raises(dp.ModelSpecError, match="do not apply"):
        dp.load_mod(text)
    solved = dp.solve_mod(text, method="vfi")
    assert solved.method == "vfi"
    with pytest.raises(dp.ModelSpecError, match="do not apply"):
        dp.solve_mod(HOUSEHOLD, backend="cupy")


def test_error_names_file_and_source_line(tmp_path):
    path = tmp_path / "bad.mod"
    path.write_text("parameters beta;\nbeta = .9;\nstate a = linspace(0,1,2.5);\n")
    with pytest.raises(dp.ModelSpecError, match=r"bad\.mod:3:.*integer"):
        dp.load_mod(path)
    path.write_text("parameters beta; beta=.9;\nstate a=[0,1];\nmodel;\nreward=log(missing);\nend;")
    with pytest.raises(dp.ModelSpecError, match=r"bad\.mod:4:.*unknown name"):
        dp.load_mod(path)


def test_finite_horizon_survival_and_discrete_choice():
    text = """
    parameters beta T; beta=.9; T=3;
    state a = linspace(0,3,15); discrete h = [0,1];
    horizon=T; survival=[1,.95,.9];
    model; c = 1 + a + h - a(+1); reward=log(c)-.3*h; end;
    constraints; c > 0; end;
    solve(method=vfi);
    """
    sol = dp.solve_mod(text)
    assert sol.V.shape == (3, 15, 1)
    # At the final age saving has no benefit and labor choice is enumerated.
    a = sol.grids["a"]
    expected = np.maximum(np.log(1+a), np.log(2+a)-.3)
    np.testing.assert_allclose(sol.V[-1, :, 0], expected, atol=1e-12)


def test_hjb_matches_explicit_model():
    text = """
    parameters rho r gamma; rho=.05; r=.02; gamma=2;
    state a = linspace(0,5,40);
    exogenous z = jump([.2,1], [-.1,.1; .1,-.1]);
    control c; drift a = r*a + z - c; discount_rate rho;
    model; reward=crra(c,gamma); end;
    solve(method=hjb, tol=1e-7);
    """
    m = dp.Model().parameters(rho=.05, r=.02, gamma=2)
    m.state("a", np.linspace(0,5,40)).exogenous("z", dp.Jump([.2,1], [[-.1,.1],[.1,-.1]]))
    m.control("c").drift("a", "r*a + z - c").discount_rate("rho").reward("crra(c,gamma)")
    sol, ref = dp.solve_mod(text), m.solve("hjb", tol=1e-7)
    np.testing.assert_array_equal(sol.V, ref.V)
    np.testing.assert_array_equal(sol.distribution, ref.distribution)


@pytest.mark.parametrize("change, message", [
    (lambda s: s + "beta=.9;", "final statement"),
    (lambda s: s.replace("n = 30", "n = unknown"), "unknown calibration"),
    (lambda s: s.replace("n = 30", "n = __import__('os').getpid()"), "calibration accepts"),
    (lambda s: s.replace("n = 30", "n = linspace(0,1,1000000000)"), "grid length"),
    (lambda s: s.replace("n = 30", "n = 10^1000000000"), "overflow"),
    (lambda s: s.replace("n = 30", "n = [1,2][0]"), "calibration accepts"),
    (lambda s: s.replace("beta = .95", "beta = 1/0"), "divide"),
    (lambda s: s.replace("gamma = 2", "gamma = 2; gamma=3"), "more than once"),
    (lambda s: s.replace("state a", "state z"), "duplicate"),
    (lambda s: s.replace("c > 0", "unknown > 0"), "unknown name"),
    (lambda s: s.replace("tol=1e-9", "surprise=true"), "named options"),
    (lambda s: s.replace("tol=1e-9", "max_iter=1.5"), "integer"),
    (lambda s: s.replace("tol=1e-9", "distribution=1"), "true or false"),
    (lambda s: s.replace("tol=1e-9", "tol=-1"), "positive"),
    (lambda s: s.replace("end;", "", 1), "nested blocks"),
    (lambda s: s.rstrip()[:-1], "end with"),
])
def test_invalid_and_executable_inputs_fail_explicitly(change, message):
    with pytest.raises(dp.ModelSpecError, match=message):
        dp.load_mod(change(HOUSEHOLD))


def test_explicit_registered_function_uses_existing_dp_contract():
    text = HOUSEHOLD.replace("exp(z)", "income(z)").replace("method=egm", "method=vfi")
    m = dp.load_mod(text, functions={"income": np.exp})
    assert m._functions["income"] is np.exp
    with pytest.raises(dp.ModelSpecError, match="not supported|unknown"):
        dp.load_mod(text)


def test_market_clearing_declarations_reach_existing_compiler():
    text = HOUSEHOLD.replace("parameters beta gamma r rho sigma n", "parameters beta gamma rho sigma n")
    text = text.replace("r = .02;", "price r = [-.02, .04];")
    text = text.replace("solve(", "clear = A - 2; solve(")
    m = dp.load_mod(text)
    assert m._prices == {"r": (-.02, .04)}
    assert m._clear == "A - 2"


def test_missing_path_is_not_interpreted_as_model_text(tmp_path):
    with pytest.raises(FileNotFoundError):
        dp.load_mod(str(tmp_path / "missing.mod"))


def test_whitespace_in_declarations_and_call_headers():
    text = HOUSEHOLD.replace("parameters beta", "parameters\n beta")
    text = text.replace("state a", "state\t a").replace("solve(", "solve (")
    assert len(dp.load_mod(text)._states[0][1]) == 30


def test_cli_check_and_solve_and_error(tmp_path, capsys):
    from puremacro.dp.cli import main
    path = tmp_path / "household.mod"
    path.write_text(HOUSEHOLD)
    assert main([str(path), "--check"]) == 0
    assert "No numerical solve" in capsys.readouterr().out
    assert main([str(path), "--method", "vfi", "--param", "gamma=1"]) == 0
    assert main([str(path), "--param", "missing=1"]) == 1
    assert "override names" in capsys.readouterr().err


def test_shipped_growth_example_against_exact_policy():
    from importlib.resources import files
    source = files("puremacro.dp").joinpath("_examples/growth.mod").read_text()
    sol = dp.solve_mod(source)
    k = sol.grids["k"]
    exact = .36*.95*k**.36
    # Global grid comparison: discrete optimization is accurate to adjacent-grid
    # resolution, independently of the Python compiler parity tests above.
    assert np.max(np.abs(sol.policy("k(+1)")[:, 0] - exact)) < 2*np.max(np.diff(k))


@pytest.mark.parametrize("name", ["household", "lifecycle"])
def test_shipped_household_examples(name):
    from importlib.resources import files
    source = files("puremacro.dp").joinpath(f"_examples/{name}.mod").read_text()
    sol = dp.solve_mod(source)
    assert np.all(sol.policy("c") > 0)
    np.testing.assert_allclose(sol.distribution.sum(axis=(-2, -1)), 1)
