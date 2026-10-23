"""Unsolved equilibrium specifications and explicit solver selection."""
from pathlib import Path

import numpy as np
import pytest

from puremacro.dsge import ModelError, NonlinearModel, read_mod, load_mod, solve_perfect_foresight


@pytest.mark.pyodide_smoke
def test_read_mod_does_not_require_a_local_policy_or_a_steady_state():
    source = "var x; model; x=2*x(+1); end; initval; x=17; end;"
    with pytest.raises(ModelError, match="predetermined"):
        load_mod(source)
    model = read_mod(source)
    assert model.steady_state is None
    assert model.guess == {"x": 17.0}
    result = solve_perfect_foresight(model, y_init=[0], y_end=[1], periods=4)
    assert result.converged
    np.testing.assert_allclose(result.path.x, [16, 8, 4, 2], atol=1e-10)


def test_model_metadata_parameter_override_and_fixed_steady_reference():
    model = read_mod("""
        var x y; varexo e; parameters rho; rho=.5;
        model;
        #intercept=1-rho;
        [name='state'] x=intercept+rho*x(-1)+e;
        [name='reference'] y=STEADY_STATE(x);
        end;
        steady_state_model; x=1; y=1; end;
        initval; x=rho; y=1; end;
    """, params={"rho": .8})
    assert model.variables == ("x", "y") and model.shocks == ("e",)
    assert model.equation_names == ("state", "reference")
    assert model.timing["x"] == (-1, 0)
    assert len(model.equation_sources) == 2
    assert len(model.source_hashes["<text>"]) == 64
    assert model.guess == {"x": .8, "y": 1}
    np.testing.assert_allclose(model.residuals([1, 1], [1, 3], [1, 1], [0], steady_state=[3, 3]), 0, atol=1e-14)
    point = np.array([1 + 1e-20j, 1])
    assert model.residuals(point, point, [1, 1], [0]).imag[0] != 0
    assert model.steady() == pytest.approx({"x": 1, "y": 1})
    local = model.solve(order=1)
    assert local.solution.G[0, 0] == pytest.approx(.8)


def test_callback_model_agrees_with_mod_for_parameters_and_complex_inputs():
    def equations(lead, current, lag, shock, *, params, steady_state):
        return [current[0] - params["rho"] * lag[0] - shock[0],
                current[1] - steady_state[0]]

    callback = NonlinearModel.from_equations(equations, variables=["x", "y"], shocks=["e"],
        params={"rho": .4}, equation_names=["law", "reference"], guess=[0, 0])
    source = read_mod("var x y; varexo e; parameters rho; rho=.4; model; x=rho*x(-1)+e; y=STEADY_STATE(x); end;")
    arguments = dict(y_init=[1, 0], y_end=[3, 3], shocks=np.array([[.2], [0], [0]]), periods=3)
    left = callback.solve_perfect_foresight(**arguments)
    right = source.solve_perfect_foresight(**arguments)
    assert left.converged and right.converged
    np.testing.assert_allclose(left.path, right.path, atol=1e-10)


def test_callback_body_typeerror_is_not_retried():
    calls = []
    def broken(lead, current, lag, shock):
        calls.append(True)
        raise TypeError("economic callback failed")
    model = NonlinearModel.from_equations(broken, variables=["x"])
    with pytest.raises(TypeError, match="economic callback failed"):
        model.residuals([0], [0], [0], [])
    assert calls == [True]


def test_covariance_parameter_override_agrees_between_solution_and_commands():
    model = read_mod("""
        var x; varexo e; parameters rho sigma; rho=.5; sigma=.1;
        model; x=rho*x(-1)+e; end; initval; x=0; e=0; end;
        shocks; var e; stderr sigma; end; stoch_simul(order=1,irf=2);
    """, params={"sigma": .3})
    np.testing.assert_allclose(model.shock_cov, [[.09]])
    direct = model.solve().stoch_simul(order=1, irf=2)
    ordered = model.execute_mod_commands().stoch_simul
    np.testing.assert_allclose(direct.irfs["x_e"], ordered.irfs["x_e"])


def test_read_mod_retains_expanded_source_when_include_changes(tmp_path):
    included = tmp_path / "equations.mod"
    included.write_text("model; x=.5*x(-1); end;")
    main = tmp_path / "main.mod"
    main.write_text('var x;\n@#include "equations.mod"\ninitval; x=2; end; simul(periods=3);')
    model = read_mod(main)
    hashes = dict(model.source_hashes)
    included.write_text("model; x=.9*x(-1); end;")
    result = model.execute_mod_commands().perfect_foresight
    np.testing.assert_allclose(result.path.x, [1, .5, .25], atol=1e-10)
    assert model.source_hashes == hashes and str(included) in hashes


@pytest.mark.parametrize("method", ["read", "callback"])
def test_vector_and_specification_contracts(method):
    model = (read_mod("var x; model; x=.5*x(-1); end;") if method == "read" else
             NonlinearModel.from_equations(lambda p, c, l, e: c-.5*l, variables=["x"]))
    with pytest.raises(ValueError, match="shocks must have shape"):
        model.residuals([0], [0], [0], [1])
    with pytest.raises(ValueError, match="current must have shape"):
        model.residuals([0], [0, 0], [0], [])
    with pytest.raises(ValueError, match="missing"):
        model.residuals({"x": 0}, {}, {"x": 0}, [])


def test_invalid_model_and_unknown_parameters_are_diagnostic():
    with pytest.raises(ModelError, match="one equation per variable"):
        read_mod("var x y; model; x=y; end;")
    with pytest.raises(ModelError, match="Unknown parameters"):
        read_mod("var x; model; x=1; end;", params={"unused": 1})
    with pytest.raises(ModelError, match="distinct"):
        NonlinearModel.from_equations(lambda *a: [0], variables=["x"], shocks=["x"])


def test_course_specification_shares_original_nonlinear_equations():
    path = Path(__file__).resolve().parents[2] / "curso/notebooks/modelos/neoclasico_prevision_perfecta.mod"
    model = read_mod(path)
    steady = model.steady()
    state = np.array([steady[v] for v in model.variables])
    np.testing.assert_allclose(model.residuals(state, state, state, [0]), 0, atol=1e-10)
    nonlinear = state.copy()
    nonlinear[0] *= 1.4
    nonlinear[1] *= 1.25
    nonlinear[2] *= .8
    assert np.max(np.abs(model.residuals(state, nonlinear, state, [0]))) > .1
