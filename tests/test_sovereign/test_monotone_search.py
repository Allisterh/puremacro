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
