"""Fiscal block: government spending in utility on the short-term Arellano solver."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from puremacro.sovereign import (
    FiscalSovereignResult,
    solve_fiscal_sovereign_default,
    solve_sovereign_default,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sovereign_fiscal" / "illustrative_101x11.json"

PUBLIC = dict(beta=0.95282, r_star=0.017, theta=0.282, n_b=101, n_y=9, sigma_eps=0.0,
              y_penalty_type="arellano", enforce_monotonicity=False, tol=1e-10, max_iter=10000)


@pytest.mark.parametrize("search", ["dense", "monotone"])
def test_reduces_to_sovereign_solver_exactly(search: str) -> None:
    fiscal = solve_fiscal_sovereign_default(tau=1.0, psi=1.0, gamma_g=2.0, private_consumption=False,
                                            b_min=-0.15, b_max=0.33, policy_search=search, **PUBLIC)
    plain = solve_sovereign_default(gamma=2.0, b_min=-0.15, b_max=0.33, policy_search=search, **PUBLIC)
    assert isinstance(fiscal, FiscalSovereignResult)
    for name in ("v_repay", "v_default", "v_expected", "bond_price", "default_prob", "policy_b",
                 "dq_db", "spread"):
        assert np.array_equal(getattr(fiscal, name), getattr(plain, name)), name
    assert fiscal.iterations == plain.iterations
    assert fiscal.converged == plain.converged
    assert fiscal.spread_mean == plain.spread_mean


def test_crra_homogeneity_in_resources() -> None:
    common = dict(PUBLIC, tol=1e-12, max_iter=20000)
    full = solve_fiscal_sovereign_default(tau=1.0, psi=1.0, private_consumption=False,
                                          b_min=-0.15, b_max=0.33, **common)
    scaled = solve_fiscal_sovereign_default(tau=0.2, psi=1.0, private_consumption=False,
                                            b_min=-0.03, b_max=0.066, **common)
    beta = common["beta"]
    assert np.max(np.abs(scaled.b_grid - 0.2 * full.b_grid)) <= 1e-16
    assert np.array_equal(scaled.policy_index, full.policy_index)
    assert np.array_equal(scaled.default_prob, full.default_prob)
    assert np.max(np.abs(scaled.bond_price - full.bond_price)) <= 1e-10
    # u(0.2 c) = 0.2**-1 u(c) + (0.2**-1 - 1) / (1 - gamma) for gamma = 2.
    expected = 0.2 ** -1 * full.v_repay + (0.2 ** -1 - 1.0) / (-1.0 * (1.0 - beta))
    assert np.max(np.abs(scaled.v_repay - expected) / np.abs(expected)) <= 1e-10


@pytest.mark.parametrize("private", [True, False])
@pytest.mark.parametrize("sigma_eps", [0.0, 0.01])
@pytest.mark.parametrize("gamma_g", [1.0, 2.0, 5.0])
def test_monotone_equals_dense(gamma_g: float, sigma_eps: float, private: bool) -> None:
    kwargs = dict(tau=0.2, psi=1.0, gamma_g=gamma_g, sigma_eps=sigma_eps, private_consumption=private,
                  n_b=41, n_y=7)
    dense = solve_fiscal_sovereign_default(policy_search="dense", **kwargs)
    fast = solve_fiscal_sovereign_default(policy_search="monotone", **kwargs)
    assert np.array_equal(fast.policy_index, dense.policy_index)
    assert fast.iterations == dense.iterations
    for name in ("v_repay", "v_default", "v_expected", "bond_price", "default_prob"):
        assert np.max(np.abs(getattr(fast, name) - getattr(dense, name))) <= 1e-12, name
    assert fast.metadata["monotone_certification_gap"] <= 1e-11
    assert dense.metadata["monotone_certification_gap"] is None


def test_fiscal_accounting() -> None:
    res = solve_fiscal_sovereign_default(tau=0.2, psi=1.0, n_b=41, n_y=7)
    b, y, tau = res.b_grid, res.y_grid, res.tau
    q_chosen = np.take_along_axis(res.bond_price, res.policy_index, axis=0)
    assert np.array_equal(res.policy_b, b[res.policy_index])
    assert np.array_equal(res.g_policy, tau * y[None, :] - b[:, None] + q_chosen * res.policy_b)
    assert np.max(np.abs(res.primary_balance - (tau * y[None, :] - res.g_policy))) <= 1e-15
    assert np.array_equal(res.debt_to_gdp, res.policy_b / y[None, :])
    assert np.array_equal(res.g_default, tau * res.metadata["y_default_vec"])
    assert np.array_equal(res.private_consumption_repay, (1.0 - tau) * y)
    assert np.array_equal(res.private_consumption_default, (1.0 - tau) * res.metadata["y_default_vec"])
    assert res.metadata["resources"] == "tau * y"
    assert res.metadata["rule"] is None
    assert res.b_grid[res.metadata["zero_index"]] == 0.0
    assert "gamma" not in res.metadata
    for key in ("gamma_c", "gamma_g", "tau", "psi", "private_consumption"):
        assert key in res.metadata
    rows = dict(zip(res.summary()["metric"], res.summary()["value"]))
    assert rows["tau"] == 0.2 and rows["psi"] == 1.0 and rows["private_consumption"] is True


def test_illustrative_small_grid_matches_frozen() -> None:
    frozen = json.loads(FIXTURE.read_text())
    res = solve_fiscal_sovereign_default(tau=0.2, psi=1.0, n_b=101, n_y=11)
    assert res.iterations == frozen["iterations"]
    assert np.array_equal(res.policy_index, np.array(frozen["policy_index"]))
    for name in ("bond_price", "v_repay", "v_default"):
        assert np.max(np.abs(getattr(res, name) - np.array(frozen[name]))) <= 1e-10, name


def test_tau_one_requires_private_consumption_off() -> None:
    with pytest.raises(ValueError, match="private_consumption"):
        solve_fiscal_sovereign_default(tau=1.0, psi=1.0, private_consumption=True, n_b=21, n_y=5)


@pytest.mark.parametrize("tau", [0.0, -0.1, 1.2, float("nan")])
def test_rejects_tax_rate_outside_unit_interval(tau: float) -> None:
    with pytest.raises(ValueError, match="tau"):
        solve_fiscal_sovereign_default(tau=tau, psi=1.0, n_b=21, n_y=5)


@pytest.mark.parametrize("psi", [0.0, -1.0, float("inf")])
def test_rejects_nonpositive_or_infinite_spending_weight(psi: float) -> None:
    with pytest.raises(ValueError, match="psi"):
        solve_fiscal_sovereign_default(tau=0.2, psi=psi, n_b=21, n_y=5)


@pytest.mark.parametrize("name", ["gamma_c", "gamma_g"])
def test_rejects_negative_curvature(name: str) -> None:
    with pytest.raises(ValueError, match=name):
        solve_fiscal_sovereign_default(tau=0.2, psi=1.0, n_b=21, n_y=5, **{name: -1.0})


def test_rejects_unknown_policy_search() -> None:
    with pytest.raises(ValueError, match="policy_search"):
        solve_fiscal_sovereign_default(tau=0.2, psi=1.0, n_b=21, n_y=5, policy_search="fast")


def test_monotone_barrier_points_to_dense() -> None:
    with pytest.raises(ValueError, match='policy_search="dense"'):
        solve_fiscal_sovereign_default(tau=0.2, psi=1.0, b_max=0.5, n_b=21, n_y=5)


@pytest.mark.parametrize("psi", [1e-12, 1.0])
def test_dense_barrier_holds_down_to_the_documented_psi(psi: float) -> None:
    res = solve_fiscal_sovereign_default(tau=0.2, psi=psi, b_max=0.5, n_b=61, n_y=7, policy_search="dense")
    repay = res.default_prob < 0.5
    assert res.converged and repay.any()
    assert np.min(res.g_policy[repay]) > 1e-6
