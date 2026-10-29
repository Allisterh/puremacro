"""Independent native-model oracles for the differentiable adapter boundary."""
from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import solve_discrete_lyapunov

from puremacro.core.protocol import DSGEDifferentiableAdapter, HANKDAGDifferentiableAdapter
from puremacro.dsge import build_dynare, BlockDAG, SimpleBlock
from puremacro.dsge.observation import make_state_space_from_varobs
from puremacro.dsge.hank_estimate import hank_loglik_whittle
from puremacro.dsge.hank_nonlinear import solve_hank_aggregate_risk_order2
from puremacro.dsge.hank_nonlinear import solve_hank_nonlinear_transition
from puremacro.core.protocol import (
    DFMDifferentiableAdapter, SVARDifferentiableAdapter,
    TradeDifferentiableAdapter, SpatialDifferentiableAdapter,
)


@pytest.mark.parametrize("adapter_type", [DFMDifferentiableAdapter, SVARDifferentiableAdapter,
                                         TradeDifferentiableAdapter, SpatialDifferentiableAdapter])
@pytest.mark.parametrize("method,kwargs", [("simulate", {}), ("impulse_response", {"shock": 0}),
                                          ("counterfactual", {"scenario": {}})])
def test_unsupported_native_operations_cannot_return_demonstration(adapter_type, method, kwargs):
    with pytest.raises(NotImplementedError, match="[Nn]ative"):
        getattr(adapter_type(object()), method)(**kwargs)


def test_nonlinear_transition_rejects_unknown_model_instead_of_replacing_it():
    with pytest.raises(TypeError, match="BlockDAG"):
        solve_hank_nonlinear_transition(model=object(), horizon=5)


def test_dsge_counterfactual_resolves_the_parameter_override():
    adapter = DSGEDifferentiableAdapter(ar1(), param_names=["rho"])
    result = adapter.counterfactual({"param_override": {"rho": 0.2},
                                     "shock_sequence": {"e": np.r_[1.0, np.zeros(5)]}})
    np.testing.assert_allclose(result["y"], 0.2**np.arange(6), atol=1e-10)


def test_reduced_transition_reports_unclipped_amplification():
    result = solve_hank_nonlinear_transition(shocks={"eps_m": -0.02}, horizon=25)
    expected = np.max(abs(result.paths["Y"]-1))/np.max(abs(result.linear_paths["Y"]-1))
    assert result.asymmetry_ratio == pytest.approx(expected)


def ar1(scale=1.0):
    def equations(lead, curr, lag, shocks, p):
        return [curr.y - p.rho * lag.y - p.scale * shocks.e]
    return build_dynare(equations, variables=["y"], shocks=["e"],
                        params={"rho": 0.6, "scale": scale}, steady_state={"y": 0.0})


def test_dsge_likelihood_uses_native_model_and_structural_score():
    y = np.random.default_rng(44).normal(size=(60, 1))
    values = []
    for scale in (1.0, 2.0):
        model = ar1(scale)
        adapter = DSGEDifferentiableAdapter(model, param_names=["rho"])
        ll, score = adapter.loglik_and_grad({"rho": 0.6}, y, observables=["y"])
        ssm = make_state_space_from_varobs(model, ["y"])
        # Match the documented stationary initialization, not the filter's
        # default diffuse prior. The native filter is independent of the score.
        p0 = solve_discrete_lyapunov(ssm.T, ssm.R @ ssm.Q @ ssm.R.T)
        oracle = ssm.filter(y, a0=np.zeros(ssm.T.shape[0]), P0=p0)["loglik"]
        assert ll == pytest.approx(oracle, abs=1e-8)
        h = 1e-5
        fd = (adapter.loglik_and_grad({"rho": 0.6+h}, y, observables=["y"])[0]
              - adapter.loglik_and_grad({"rho": 0.6-h}, y, observables=["y"])[0])/(2*h)
        assert score[0] == pytest.approx(fd, rel=1e-5, abs=1e-6)
        values.append(ll)
    assert abs(values[1] - values[0]) > 1.0


def test_dsge_invalid_simulation_is_not_replaced_with_synthetic_data():
    with pytest.raises(ValueError, match="shocks"):
        DSGEDifferentiableAdapter(ar1()).simulate(periods=10, shocks=np.zeros((10, 2)))


def test_dsge_requires_observed_data_and_checks_parameter_names():
    adapter = DSGEDifferentiableAdapter(ar1(), param_names=["rho"])
    with pytest.raises(ValueError, match="data"):
        adapter.loglik_and_grad()
    with pytest.raises(ValueError, match="parameter"):
        adapter.loglik_and_grad({"misspelled_rho": 0.6}, np.zeros((5, 1)), observables=["y"])


def simple_dag():
    block = SimpleBlock("market", inputs=("Y", "Z"), outputs=("clearing",),
                        fn=lambda v, p: {"clearing": v["Y"] - p["beta"] * v["Y"](-1) - v["Z"]})
    return BlockDAG(blocks=[block], unknowns=("Y",), exogenous=("Z",), targets=("clearing",))


def test_dag_impulse_response_requires_explicit_steady_state():
    with pytest.raises(ValueError, match="steady_state"):
        HANKDAGDifferentiableAdapter(simple_dag(), params={"beta": 0.6}).impulse_response("Z")


def test_hank_adapter_uses_dag_spectrum_and_beta_changes_likelihood():
    dag = simple_dag()
    ss = {"Y": 0.0, "Z": 0.0, "clearing": 0.0}
    y = np.random.default_rng(41).normal(size=(40, 1))
    adapter = HANKDAGDifferentiableAdapter(dag, steady_state=ss, params={"beta": 0.6}, param_names=["beta"])
    opts = dict(observables=["Y"], shock_cov=np.array([[1.0]]), T_ma=40)
    ll, grad = adapter.loglik_and_grad({"beta": 0.6}, y, **opts)
    oracle = hank_loglik_whittle(dag, y, opts["shock_cov"], steady_state=ss,
                                params={"beta": 0.6}, observables=["Y"], T_ma=40)
    assert ll == pytest.approx(oracle, abs=1e-9)
    assert abs(grad[0]) > 1e-3
    assert abs(adapter.loglik_and_grad({"beta": 0.7}, y, **opts)[0] - ll) > 1e-3


def test_hank_simulation_uses_supplied_shocks_and_native_irfs():
    dag = simple_dag()
    adapter = HANKDAGDifferentiableAdapter(dag, steady_state={"Y": 0., "Z": 0., "clearing": 0.}, params={"beta": 0.6})
    shocks = np.zeros((20, 1)); shocks[0] = 1.0
    actual = adapter.simulate(periods=20, shocks=shocks)
    np.testing.assert_allclose(actual["Y"], 0.6 ** np.arange(20), atol=1e-6)


def test_aggregate_risk_rejects_unsupported_model_instead_of_ignoring_it():
    with pytest.raises(NotImplementedError, match="model"):
        solve_hank_aggregate_risk_order2(model=simple_dag())


def test_aggregate_risk_rejects_unused_steady_state():
    with pytest.raises(NotImplementedError, match="steady_state"):
        solve_hank_aggregate_risk_order2(steady_state={"Y": 2.0})


@pytest.mark.parametrize("stem", ["local_llm_uncertainty", "local_llm_uncertainty_es"])
def test_local_llm_notebook_uses_shipped_read_only_fallback(stem, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root / "notebooks")
    monkeypatch.syspath_prepend(str(root / "notebooks"))
    monkeypatch.setenv("PUREMACRO_LLM_MODE", "cache")
    namespace = runpy.run_path(str(root / "notebooks" / f"{stem}.py"))
    assert 0.0 <= float(namespace["cached_score"]) <= 1.0
