"""simulate_panel takes integer grid indices; a float policy is refused, not truncated."""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.vfi.simulate import simulate_panel


def test_float_policy_raises():
    with pytest.raises(TypeError, match=r"puremacro\.dp.*simulate"):
        simulate_panel(np.array([[0.7], [1.6]]), np.eye(1), n_agents=2, n_periods=2)


def test_integral_valued_float_policy_also_raises():
    # A float array is refused even when its values happen to be whole numbers:
    # the dtype, not the values, says whether these are grid indices.
    with pytest.raises(TypeError, match="integer"):
        simulate_panel(np.array([[0.0], [1.0]]), np.eye(1), n_agents=2, n_periods=2)


def test_integer_policy_unchanged():
    # Paths recorded from the implementation before the dtype check was added.
    pol = np.array([[1, 0], [2, 1], [2, 2]], dtype=np.int32)
    P = np.array([[0.9, 0.1], [0.3, 0.7]])

    a, z = simulate_panel(pol, P, n_agents=4, n_periods=6, seed=3)
    assert a.tolist() == [[0, 1, 2, 2, 2, 2], [0, 1, 2, 2, 2, 2],
                          [0, 0, 0, 0, 0, 0], [0, 1, 2, 2, 2, 2]]
    assert z.tolist() == [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1],
                          [1, 1, 1, 1, 1, 0], [0, 0, 0, 1, 0, 0]]

    a, z = simulate_panel(pol, P, n_agents=4, n_periods=6, seed=3,
                          z0_index=1, burn_in=2)
    assert a.tolist() == [[1, 2, 2, 2, 2, 2], [1, 2, 2, 2, 2, 2],
                          [0, 0, 0, 0, 0, 1], [0, 1, 2, 2, 2, 2]]
    assert z.tolist() == [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1],
                          [1, 1, 1, 1, 0, 0], [0, 0, 1, 0, 0, 0]]
    assert a.dtype == np.intp and z.dtype == np.intp
