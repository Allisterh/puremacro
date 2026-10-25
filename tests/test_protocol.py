"""Unit tests for puremacro.core.protocol (DifferentiableModel & StateSpaceProtocol).

Verifies:
1. Protocol runtime checks: isinstance(adapter, DifferentiableModel) is True for all adapters.
2. StateSpaceProtocol runtime check: isinstance(StateSpaceModel, StateSpaceProtocol) is True.
3. wrap_differentiable_model correctly dispatches and wraps diverse models.
4. All 5 canonical methods (.impulse_response, .simulate, .counterfactual, .to_interactive_html,
   .loglik_and_grad) adhere to return contracts.
5. DualArray gradients through .loglik_and_grad() match central finite differences
   to rtol <= 1e-5 across all adapters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.core.protocol import (
    DFMDifferentiableAdapter,
    DSGEDifferentiableAdapter,
    DifferentiableModel,
    HANKDAGDifferentiableAdapter,
    SpatialDifferentiableAdapter,
    StateSpaceProtocol,
    SVARDifferentiableAdapter,
    TradeDifferentiableAdapter,
    wrap_differentiable_model,
)
from puremacro.state_space import StateSpaceModel


def _central_finite_difference(adapter: DifferentiableModel, theta: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Compute central finite differences of the adapter's log-likelihood."""
    P = len(theta)
    g_fd = np.zeros(P, dtype=float)
    for p in range(P):
        e_p = np.zeros(P, dtype=float)
        e_p[p] = h
        ll_plus, _ = adapter.loglik_and_grad(params=theta + e_p)
        ll_minus, _ = adapter.loglik_and_grad(params=theta - e_p)
        g_fd[p] = (ll_plus - ll_minus) / (2.0 * h)
    return g_fd


# ===========================================================================
# 1. Protocol Runtime Compliance
# ===========================================================================

def test_state_space_protocol_compliance():
    """StateSpaceModel implements StateSpaceProtocol."""
    ssm = StateSpaceModel(
        T=np.array([[0.8]]),
        Z=np.array([[1.0]]),
        Q=np.array([[0.05]]),
        H=np.array([[0.01]]),
    )
    assert isinstance(ssm, StateSpaceProtocol)
    y = np.array([[1.0], [1.2], [0.9]])
    filt = ssm.filter(y)
    assert "loglik" in filt
    sm = ssm.smooth(y)
    assert "a_smooth" in sm


def test_adapters_implement_differentiable_model():
    """All 6 domain adapters satisfy isinstance(adapter, DifferentiableModel)."""
    adapters = [
        DSGEDifferentiableAdapter(None),
        HANKDAGDifferentiableAdapter(None),
        DFMDifferentiableAdapter(None),
        SVARDifferentiableAdapter(None),
        TradeDifferentiableAdapter(None),
        SpatialDifferentiableAdapter(None),
    ]
    for adapter in adapters:
        assert isinstance(adapter, DifferentiableModel), f"{type(adapter).__name__} fails protocol"


def test_wrap_differentiable_model():
    """wrap_differentiable_model correctly dispatches based on model attributes."""
    class DummyDSGE:
        solution = object()
        shocks = ("eps",)
        variables = ("y",)

    class DummyDAG:
        ge_jacobians = object()
        exogenous = ("r",)
        unknowns = ("Y",)

    class DummyDFM:
        n_factors = 2

    class DummySVAR:
        A_list = [np.eye(2)]
        names = ("y1", "y2")

    class DummyTrade:
        trade_shares = np.ones((1, 2, 2))

    class DummySpatial:
        migration_shares = np.ones((2, 2))

    w_dsge = wrap_differentiable_model(DummyDSGE())
    assert isinstance(w_dsge, DSGEDifferentiableAdapter)
    assert isinstance(w_dsge, DifferentiableModel)

    w_dag = wrap_differentiable_model(DummyDAG())
    assert isinstance(w_dag, HANKDAGDifferentiableAdapter)

    w_dfm = wrap_differentiable_model(DummyDFM())
    assert isinstance(w_dfm, DFMDifferentiableAdapter)

    w_svar = wrap_differentiable_model(DummySVAR())
    assert isinstance(w_svar, SVARDifferentiableAdapter)

    w_trade = wrap_differentiable_model(DummyTrade())
    assert isinstance(w_trade, TradeDifferentiableAdapter)

    w_spatial = wrap_differentiable_model(DummySpatial())
    assert isinstance(w_spatial, SpatialDifferentiableAdapter)

    # Idempotent wrapping
    assert wrap_differentiable_model(w_dsge) is w_dsge


# ===========================================================================
# 2. Canonical Methods Contract
# ===========================================================================

@pytest.mark.parametrize("adapter_cls", [
    DSGEDifferentiableAdapter,
    HANKDAGDifferentiableAdapter,
    DFMDifferentiableAdapter,
    SVARDifferentiableAdapter,
    TradeDifferentiableAdapter,
    SpatialDifferentiableAdapter,
])
def test_adapter_methods_contracts(adapter_cls):
    """Verify that all 5 methods return properly shaped outputs."""
    adapter = adapter_cls(None)

    # 1. impulse_response
    irf = adapter.impulse_response(shock=0, horizon=15, size=1.0)
    assert isinstance(irf, pd.DataFrame)
    assert len(irf) == 16
    assert irf.index.name == "h"

    # 2. simulate
    sim = adapter.simulate(periods=50, seed=123)
    assert isinstance(sim, pd.DataFrame)
    assert len(sim) == 50

    # 3. counterfactual
    cf = adapter.counterfactual({"shocks": np.zeros((30, 1))})
    assert cf is not None

    # 4. to_interactive_html
    html = adapter.to_interactive_html(title="Test Explorer")
    assert isinstance(html, str)
    assert "<html" in html or "<!DOCTYPE html>" in html
    assert "http://" not in html and "https://" not in html

    # 5. loglik_and_grad
    ll, grad = adapter.loglik_and_grad()
    assert isinstance(ll, float)
    assert isinstance(grad, np.ndarray)
    assert grad.ndim == 1
    assert len(grad) > 0


# ===========================================================================
# 3. DualArray Gradient vs Finite Difference Parity
# ===========================================================================

@pytest.mark.parametrize("adapter_cls, test_theta", [
    (DSGEDifferentiableAdapter, np.array([0.72, 0.15, 0.08])),
    (HANKDAGDifferentiableAdapter, np.array([0.96, 1.20, 0.75])),
    (DFMDifferentiableAdapter, np.array([0.65, 0.90, 0.25])),
    (SVARDifferentiableAdapter, np.array([0.60, 0.12, 0.08, 0.55, 0.20, 0.18])),
    (TradeDifferentiableAdapter, np.array([3.80, 1.10, 0.30])),
    (SpatialDifferentiableAdapter, np.array([2.20, 0.92, 0.95])),
])
def test_dual_array_gradient_parity(adapter_cls, test_theta):
    """DualArray gradients through .loglik_and_grad() match central finite differences to rtol <= 1e-5."""
    adapter = adapter_cls(None)
    ll_ad, g_ad = adapter.loglik_and_grad(params=test_theta)
    g_fd = _central_finite_difference(adapter, test_theta, h=1e-6)

    # Parity check: max relative difference <= 1e-5
    rel_err = np.abs(g_ad - g_fd) / np.maximum(1.0, np.abs(g_fd))
    max_err = float(np.max(rel_err))
    assert max_err <= 1e-5, f"{adapter_cls.__name__} gradient error {max_err:.2e} > 1e-5: AD={g_ad}, FD={g_fd}"
