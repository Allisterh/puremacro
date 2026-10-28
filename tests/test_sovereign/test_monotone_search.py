"""Certified monotone policy search and vectorized price-schedule tangents."""
from __future__ import annotations

import numpy as np
import pytest

from puremacro._autodiff import DualArray
from puremacro.sovereign.arellano import _compute_dq_db_dualarray


def _reference_dq_db(q: np.ndarray, b_grid: np.ndarray) -> np.ndarray:
    """The per-node DualArray loop used before vectorization, kept as the oracle."""
    n_b, n_y = q.shape
    dq_db = np.zeros_like(q)
    h = np.diff(b_grid)
    for j in range(n_y):
        q_col = q[:, j]
        delta = np.diff(q_col) / h
        d = np.zeros(n_b, dtype=np.float64)
        for k in range(1, n_b - 1):
            if delta[k - 1] * delta[k] > 0.0:
                d[k] = 2.0 / (1.0 / delta[k - 1] + 1.0 / delta[k])
        d[0] = delta[0] if len(delta) > 0 and delta[0] <= 0.0 else 0.0
        d[-1] = delta[-1] if len(delta) > 0 and delta[-1] <= 0.0 else 0.0
        for k in list(range(n_b - 1)) + [n_b - 1]:
            left = min(k, n_b - 2)
            bk, hk = b_grid[left], h[left]
            t = (DualArray(b_grid[k], np.array([1.0])) - bk) / hk
            q_eval = ((2.0 * t ** 3 - 3.0 * t ** 2 + 1.0) * q_col[left]
                      + (t ** 3 - 2.0 * t ** 2 + t) * (hk * d[left])
                      + (-2.0 * t ** 3 + 3.0 * t ** 2) * q_col[left + 1]
                      + (t ** 3 - t ** 2) * (hk * d[left + 1]))
            dq_db[k, j] = q_eval.der[0]
    return np.minimum(dq_db, 0.0)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("shape,kind", [((2, 3), "decreasing"), ((41, 7), "decreasing"),
                                        ((41, 7), "rough"), ((200, 5), "flat"),
                                        ((60, 4), "uneven")])
def test_dq_db_vectorized_matches_loop(seed, shape, kind):
    rng = np.random.default_rng(seed)
    n_b, n_y = shape
    b_grid = np.linspace(-0.3, 0.4, n_b)
    if kind == "uneven":
        b_grid = np.sort(rng.uniform(-0.3, 0.4, n_b))
    q = np.sort(rng.uniform(0.0, 0.99, shape), axis=0)[::-1]
    if kind == "rough":
        q = rng.uniform(0.0, 0.99, shape)
    if kind == "flat":
        q[n_b // 3: n_b // 2] = q[n_b // 3]
        q[-n_b // 4:] = 0.0
    assert np.allclose(_compute_dq_db_dualarray(q, b_grid), _reference_dq_db(q, b_grid),
                       atol=1e-12, rtol=0)


def _crra(gamma):
    if gamma == 1.0:
        return np.log
    return lambda c: (c ** (1.0 - gamma) - 1.0) / (1.0 - gamma)


def _inputs(seed, n_b, n_y, saving=True):
    rng = np.random.default_rng(seed)
    b_grid = np.linspace(-0.3 if saving else 0.0, 0.4, n_b)
    y_grid = np.sort(rng.uniform(0.8, 1.2, n_y))
    q = np.sort(rng.uniform(0.0, 1 / 1.01, (n_b, n_y)), axis=0)[::-1].copy()
    ev = rng.normal(-30.0, 2.0, (n_b, n_y))
    return b_grid, y_grid, q, ev


def _dense(b_grid, y_grid, q, ev, beta, u):
    c = (y_grid[None, None, :] - 1.0 * b_grid[:, None, None]) + q[None, :, :] * b_grid[None, :, None]
    w = u(c) + beta * ev[None, :, :]
    return w.max(axis=1), w.argmax(axis=1), c.min()


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("n_b,n_y", [(2, 2), (3, 5), (17, 2), (64, 5)])
@pytest.mark.parametrize("gamma", [2.0, 5.0])
@pytest.mark.parametrize("saving", [True, False])
def test_maximize_matches_brute_force(seed, n_b, n_y, gamma, saving):
    from puremacro.sovereign._monotone import maximize

    args = (*_inputs(seed, n_b, n_y, saving), 0.95, _crra(gamma))
    value, policy = maximize(*args)
    dense_value, dense_policy, _ = _dense(*args)
    assert np.array_equal(value, dense_value)
    assert np.array_equal(policy, dense_policy)


def test_ties_choose_lowest_debt_index():
    from puremacro.sovereign._monotone import maximize

    b_grid, y_grid, q, ev = _inputs(3, 41, 5)
    q[25:, :] = 0.0          # fully defaulting region: no proceeds ...
    ev[25:, :] = ev[25, :]   # ... and identical continuation (V = V_D)
    ev[:25, :] -= 50.0       # make the tied region optimal for every row
    args = (b_grid, y_grid, q, ev, 0.95, _crra(2.0))
    value, policy = maximize(*args)
    dense_value, dense_policy, _ = _dense(*args)
    assert np.array_equal(policy, dense_policy)
    assert np.all(policy == 25)
    assert np.array_equal(value, dense_value)


def test_minimal_grid():
    from puremacro.sovereign._monotone import maximize

    args = (*_inputs(0, 2, 2), 0.95, _crra(2.0))
    value, policy = maximize(*args)
    dense_value, dense_policy, _ = _dense(*args)
    assert np.array_equal(value, dense_value) and np.array_equal(policy, dense_policy)


@pytest.mark.parametrize("block_rows", [1, 7, 128])
def test_exhaustive_matches_dense(block_rows):
    from puremacro.sovereign._monotone import exhaustive_maximize

    args = (*_inputs(5, 50, 4), 0.95, _crra(2.0))
    value, policy = exhaustive_maximize(*args, block_rows=block_rows)
    dense_value, dense_policy, _ = _dense(*args)
    assert np.array_equal(value, dense_value) and np.array_equal(policy, dense_policy)


@pytest.mark.parametrize("seed", range(5))
def test_min_candidate_consumption_is_exact(seed):
    from puremacro.sovereign._monotone import min_candidate_consumption

    b_grid, y_grid, q, ev = _inputs(seed, 30, 6)
    assert min_candidate_consumption(b_grid, y_grid, q) == _dense(b_grid, y_grid, q, ev, 0.95, np.log)[2]
