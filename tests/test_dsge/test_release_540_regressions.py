"""Independent numerical regressions found during the 5.4.0 release review."""
import inspect

import numpy as np
import pandas as pd
import pytest

from puremacro.dsge import (
    LinearModel, OccBinConstraint, PrunedDSGESolution, build_dynare,
    load_mod, solve_occbin, solve_pruned_occbin,
)
from puremacro.dsge._gradients import (
    ScoreDiagnosticsResult, StateSpaceSensitivity, build_state_space_sensitivities,
    compute_decision_rule_derivatives, kalman_score,
    log_posterior_and_gradient,
)
from puremacro.dsge.particle_filter import tempered_particle_filter
from puremacro.dsge.nuts import _numerical_gradient
from puremacro.state_space import StateSpaceModel


def test_native_stoch_simul_signature_is_inspectable():
    # 5.3.0 wrapped an alias with itself, making inspect.signature loop.
    signature = inspect.signature(LinearModel.stoch_simul)
    assert "simul_replic" in signature.parameters
    assert "hp_filter" in signature.parameters


@pytest.mark.parametrize("method", ["forward", "central"])
@pytest.mark.parametrize("vectorized", [False, True])
def test_numerical_gradient_never_probes_outside_narrow_bounds(method, vectorized):
    def objective(x):
        assert np.all(x >= 0.) and np.all(x <= 1e-8)
        return np.sum(3. * x, axis=-1)

    gradient = _numerical_gradient(objective, np.array([1e-8]), eps=1e-5,
                                   method=method, vectorized=vectorized,
                                   bounds=[(0., 1e-8)])
    np.testing.assert_allclose(gradient, [3.], rtol=1e-14)


def test_decision_rule_derivative_restores_zero_calibrated_lag():
    model = load_mod("""
        var x z; varexo e; parameters rho; rho=0;
        model; x=rho*x(-1)+z; z=.5*z(-1)+e; end;
        steady_state_model; x=0; z=0; end;
    """)
    assert model.states == ("z",)
    dF, dL = compute_decision_rule_derivatives(model, "rho")
    np.testing.assert_allclose(dF, [[1, 0], [0, 0]], atol=1e-10)
    np.testing.assert_allclose(dL, 0, atol=1e-10)


def test_decision_rule_derivative_respects_nonanalytic_model_method():
    def equations(lead, curr, lag, shocks, params):
        return [curr.x - params.rho * abs(lag.x) - (1 - params.rho) - shocks.e]

    model = build_dynare(equations, variables=["x"], shocks=["e"],
                         params={"rho": .5}, steady_state={"x": 1.}, method="central")
    dF, dL = compute_decision_rule_derivatives(model, "rho")
    np.testing.assert_allclose(dF, [[1]], atol=1e-5)
    np.testing.assert_allclose(dL, [[0]], atol=1e-5)


def _state_space(rho):
    return StateSpaceModel(T=[[rho]], Z=[[1]], Q=[[.2]], H=[[.1]], c=[.1])


def _rho_sensitivity():
    return StateSpaceSensitivity(dT=np.ones((1, 1)), dZ=np.zeros((1, 1)),
                                 dR=np.zeros((1, 1)), dQ=np.zeros((1, 1)),
                                 dH=np.zeros((1, 1)), dc=np.zeros(1), dd=np.zeros(1))


@pytest.mark.parametrize("initial", [{}, {"a0": np.array([.8]), "P0": np.array([[2.]])}])
def test_kalman_score_initial_distribution_matches_finite_differences(initial):
    data = np.array([[.4], [.7], [-.2], [.1]])
    ll, score = kalman_score(data, _state_space(.6), {"rho": _rho_sensitivity()}, **initial)
    h = 1e-6
    numerical = (kalman_score(data, _state_space(.6 + h), {}, **initial)[0]
                 - kalman_score(data, _state_space(.6 - h), {}, **initial)[0]) / (2 * h)
    assert np.isfinite(ll)
    assert score[0] == pytest.approx(numerical, rel=1e-7, abs=1e-8)


def test_score_diagnostics_verifies_the_stationary_likelihood():
    data = np.array([[.4], [.7], [-.2], [.1]])
    ll, score = kalman_score(data, _state_space(.6), {"rho": _rho_sensitivity()})
    diagnostics = ScoreDiagnosticsResult(ll, score, ("rho",), 0.)
    result = diagnostics.verify_numerical_gradient(data, lambda p: _state_space(p["rho"]), {"rho": .6})
    assert result.loc[0, "status"] == "PASS"


def test_standard_deviation_sensitivity_is_zero_at_zero():
    model = load_mod("var x; varexo e; model; x=.5*x(-1)+e; end;")
    sensitivities = build_state_space_sensitivities(
        model, ["x"], ["stderr_x", "stderr_e"], shock_cov=np.zeros((1, 1)),
        measurement_errors={"x": 0.},
    )
    np.testing.assert_array_equal(sensitivities["stderr_x"].dH, 0.)
    np.testing.assert_array_equal(sensitivities["stderr_e"].dQ, 0.)


def test_analytic_posterior_rejects_unsupported_noise_parameter_draws():
    model = load_mod("var x; varexo e; model; x=.5*x(-1)+e; end;")
    # Previously the likelihood was constant across stderr_e draws while
    # the returned score was nonzero: these were different targets.
    with pytest.raises(ValueError, match="structural parameters only"):
        log_posterior_and_gradient(np.array([.5]), ["stderr_e"], model,
                                   np.array([[.2], [.1], [-.2]]), ["x"], {})


def test_nuts_noise_parameters_use_observation_equation_target(monkeypatch):
    import puremacro.dsge.nuts as nuts
    from puremacro.dsge.estimate import estimate_dsge

    model = load_mod("var x; varexo e; model; x=.5*x(-1)+e; end;")
    monkeypatch.setattr(nuts, "nuts_sample", lambda target, **kwargs: target)
    data = pd.DataFrame({"x": np.random.default_rng(3).normal(0., .4, 20)})
    target = estimate_dsge(
        data, observation_eq=lambda p: StateSpaceModel(
            T=[[.5]], Z=[[1.]], Q=[[p["stderr_e"]**2]], H=[[.1]]),
        priors={"stderr_e": {"dist": "normal", "mean": 1., "std": .5, "lb": .1, "ub": 3.}},
        observed_vars=["x"], initial_params={"stderr_e": 1.},
        model_template=model, method="nuts", n_draws=4, n_chains=1, burn_in=2,
    )
    lp, score = target(np.array([.8]))
    h = 1e-5
    numerical = (target(np.array([.8 + h]))[0] - target(np.array([.8 - h]))[0]) / (2*h)
    assert abs(lp - target(np.array([1.2]))[0]) > 1.
    assert score[0] == pytest.approx(numerical, rel=1e-6)


@pytest.mark.parametrize("initial, expected", [
    ("steady_state", [.1, .15, .175]), ("stationary", [.2, .2, .2]),
])
def test_tempered_filter_preserves_pure_second_order_risk_drift(initial, expected):
    model = PrunedDSGESolution(
        G=np.array([[.5]]), N=np.zeros((1, 1)), F=np.zeros((0, 1)), L=np.zeros((0, 1)),
        H_xx=np.zeros((1, 1)), H_sigmasigma=np.array([.2]),
        G_xx=np.zeros((0, 1)), G_sigmasigma=np.zeros(0),
        state_names=("x",), control_names=(), shock_names=("e",),
    )
    result = tempered_particle_filter(model, np.array([[.1], [.15], [.175]]), ["x"],
                                      n_particles=32, init_states=initial,
                                      measurement_error={"x": .1}, seed=12)
    np.testing.assert_allclose(result.filtered_states["x"], expected, atol=1e-14)


def test_pruned_linear_limit_keeps_anticipated_future_shocks():
    def reference(lead, curr, lag, shocks, params):
        return [curr.x - .5 * lag.x - shocks.e, curr.y - .8 * lead.y - curr.x]

    def binding(lead, curr, lag, shocks, params):
        return [curr.x - .5 * lag.x - shocks.e, curr.y + 100]

    model = build_dynare(reference, variables=["x", "y"], shocks=["e"],
                         params={}, steady_state={"x": 0., "y": 0.})
    constraint = OccBinConstraint(variable="y", threshold=-100, operator="<")
    shocks = np.zeros((8, 1))
    shocks[3, 0] = .1
    reference_result = solve_occbin(model, binding, constraint, shocks, horizon=8)
    result = solve_pruned_occbin(model, binding, constraint, shocks, horizon=8)
    assert reference_result.simulated_path.iloc[0]["y"] == pytest.approx(.0853333333333333)
    np.testing.assert_allclose(result.simulated_path, reference_result.simulated_path, atol=1e-14)


@pytest.mark.parametrize("shock_period, message", [(0, "nonlinear binding"), (2, "anticipated future")])
def test_pruned_occbin_rejects_unsupported_nonlinear_paths(shock_period, message):
    def reference(lead, curr, lag, shocks, params):
        return [curr.x - .5 * lag.x - .2 * curr.y - shocks.e, curr.y - curr.x - curr.x**2]

    def binding(lead, curr, lag, shocks, params):
        return [curr.x - .5 * lag.x - .2 * curr.y - shocks.e, curr.y + .1]

    model = build_dynare(reference, variables=["x", "y"], shocks=["e"],
                         params={}, steady_state={"x": 0., "y": 0.})
    shocks = np.zeros((8, 1))
    shocks[shock_period, 0] = -.2
    constraint = OccBinConstraint(variable="y", threshold=-.1, operator="<")
    # The previous binding path reported convergence while violating the
    # unchanged first equation by 0.015625 in period 1.
    with pytest.raises(NotImplementedError, match=message):
        solve_pruned_occbin(model, binding, constraint, shocks, horizon=8)


def test_pruned_occbin_does_not_hide_second_order_solver_failure(monkeypatch):
    model = load_mod("var x; varexo e; model; x=.5*x(-1)+e; end;")

    def fail(*args, **kwargs):
        raise RuntimeError("second-order failure")

    monkeypatch.setattr(LinearModel, "solve", fail)
    with pytest.raises(RuntimeError, match="second-order failure"):
        solve_pruned_occbin(model, model, OccBinConstraint("x", -1, "<"), np.array([.1]))
