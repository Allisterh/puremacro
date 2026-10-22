import numpy as np
import pytest

from puremacro._wstats import w_gini, w_top_share


def test_w_gini_uniform_weights():
    # Simple case: x = [1, 2, 3], W = [1, 1, 1]
    # Gini = 2/9 ≈ 0.22222222
    x = np.array([1.0, 2.0, 3.0])
    W = np.array([[1.0], [1.0], [1.0]])
    g = w_gini(x, W)
    assert g.shape == (1,)
    assert g[0] == pytest.approx(2 / 9)

    # All same values -> Gini = 0
    x2 = np.array([5.0, 5.0, 5.0])
    g2 = w_gini(x2, W)
    assert g2[0] == pytest.approx(0.0)


def test_w_gini_custom_weights():
    x = np.array([1.0, 2.0, 3.0])
    # Column 1: [1, 1, 1] -> Gini = 2/9
    # Column 2: [2, 1, 0] -> Weights biased towards lower values
    W = np.array([[1.0, 2.0], [1.0, 1.0], [1.0, 0.0]])
    g = w_gini(x, W)
    assert g.shape == (2,)
    assert g[0] == pytest.approx(2 / 9)

    # Independent pairwise mean absolute difference, not Lorenz integration.
    pairs = np.abs(x[:, None] - x[None, :])
    expected = np.array([
        np.sum(w[:, None] * w[None, :] * pairs) / (2 * w.sum() * (w @ x))
        for w in W.T
    ])
    np.testing.assert_allclose(g, expected, rtol=0, atol=1e-15)
    assert g[1] == pytest.approx(1 / 6)



def test_w_gini_negative_values():
    # Negative values can push Gini above 1
    # x = [-1, 2], W = [1, 1]
    x = np.array([-1.0, 2.0])
    W = np.array([[1.0], [1.0]])
    g = w_gini(x, W)
    assert g[0] == pytest.approx(1.5)


def test_w_gini_zero_total_weight():
    # If total weight <= 0, should return NaN
    x = np.array([1.0, 2.0])
    W = np.array([[0.0], [0.0]])
    g = w_gini(x, W)
    assert np.isnan(g[0])

    W2 = np.array([[-1.0], [0.0]])
    g2 = w_gini(x, W2)
    assert np.isnan(g2[0])


def test_w_gini_single_element():
    x = np.array([5.0])
    W = np.array([[1.0]])
    g = w_gini(x, W)
    # N=1, tot=5
    # S=[5], S_prev=[0]
    # sum = 1 * (5) = 5
    # g = 1.0 - 5 / (1 * 5) = 0
    assert g[0] == pytest.approx(0.0)


def test_w_gini_zero_total_value():
    x = np.array([1.0, -1.0])
    W = np.array([[1.0], [1.0]])
    g = w_gini(x, W)
    # x = [-1, 1], W = [1, 1]
    # N = 2, tot = 0
    # Condition: (N > 0) & (tot > 0) -> tot is 0 -> np.nan
    assert np.isnan(g[0])


def test_two_point_top_share_fractional_cutoff():
    x = np.array([1.0, 2.0])
    W = np.array([[0.6], [0.4]])
    assert w_top_share(x, W, 0.5)[0] == pytest.approx(0.9 / 1.4, abs=1e-15)
    assert w_top_share(x, W, 0.2)[0] == pytest.approx(0.4 / 1.4, abs=1e-15)
    assert w_top_share(x, W, 0.8)[0] == pytest.approx(1.2 / 1.4, abs=1e-15)
    assert w_top_share(x, W, 1.0)[0] == pytest.approx(1.0, abs=1e-15)
    assert w_top_share(x, W, 0.0)[0] == pytest.approx(0.0, abs=1e-15)

def test_top_share_zero_or_negative_totals():
    x = np.array([1.0, 2.0])
    W_zero = np.array([[0.0], [0.0]])
    assert np.isnan(w_top_share(x, W_zero, 0.5)[0])
    x_neg = np.array([-2.0, -1.0])
    W = np.array([[0.5], [0.5]])
    assert np.isnan(w_top_share(x_neg, W, 0.5)[0])
    x_zero_sum = np.array([-1.0, 1.0])
    assert np.isnan(w_top_share(x_zero_sum, W, 0.5)[0])
