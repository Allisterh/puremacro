"""Hamilton (2018) regression filter: OLS oracle, identification guard, and
agreement between ``puremacro.cycles`` and ``puremacro.data``.

Hamilton (2018, Review of Economics and Statistics), eq. (20)-(21): regress
``y_{t+h}`` on a constant and ``y_t, y_{t-1}, ..., y_{t-p+1}``; the residual
is the cycle and the fitted value the trend. The regression has ``p + 1``
coefficients and ``T - h - p + 1`` usable rows, so it is exactly fitted (the
cycle is identically zero) unless ``T >= h + 2p + 1``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro import cycles, data


def _ols_oracle(y: np.ndarray, h: int, p: int):
    """Direct OLS of y_{t+h} on (1, y_t, ..., y_{t-p+1}) via numpy lstsq."""
    T = y.size
    rows, rhs = [], []
    for t in range(p - 1, T - h):
        rows.append([1.0] + [y[t - k] for k in range(p)])
        rhs.append(y[t + h])
    X = np.asarray(rows)
    Y = np.asarray(rhs)
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    cyc = np.full(T, np.nan)
    trd = np.full(T, np.nan)
    cyc[h + p - 1:] = Y - X @ beta
    trd[h + p - 1:] = X @ beta
    return cyc, trd


def _random_walk(T: int, seed: int = 0) -> np.ndarray:
    return np.cumsum(np.random.default_rng(seed).standard_normal(T))


@pytest.mark.parametrize("h,p", [(8, 4), (24, 12), (4, 2), (1, 1)])
def test_cycle_matches_direct_ols(h, p):
    y = _random_walk(240, seed=h * 100 + p)
    c, t = cycles.hamilton_filter(y, h=h, p=p)
    c_ref, t_ref = _ols_oracle(y, h, p)
    n_nan = h + p - 1
    assert np.all(np.isnan(c[:n_nan])) and np.all(np.isnan(t[:n_nan]))
    assert np.all(np.isfinite(c[n_nan:]))
    np.testing.assert_allclose(c[n_nan:], c_ref[n_nan:], atol=1e-10)
    np.testing.assert_allclose(t[n_nan:], t_ref[n_nan:], atol=1e-10)
    np.testing.assert_allclose(c[n_nan:] + t[n_nan:], y[n_nan:], atol=1e-10)


@pytest.mark.parametrize("h,p", [(8, 4), (24, 12)])
def test_data_wrapper_equals_cycles(h, p):
    idx = pd.date_range("1960-01-01", periods=240, freq="QS")
    y = pd.Series(_random_walk(240, seed=7), index=idx)
    c_d, t_d = data.hamilton_filter(y, h=h, p=p)
    c_c, t_c = cycles.hamilton_filter(y, h=h, p=p)
    assert isinstance(c_d, pd.Series) and isinstance(t_d, pd.Series)
    assert c_d.name == "cycle" and t_d.name == "trend"
    assert c_d.index.equals(idx) and t_d.index.equals(idx)
    np.testing.assert_array_equal(c_d.isna().values, np.isnan(np.asarray(c_c)))
    np.testing.assert_allclose(c_d.values, np.asarray(c_c), atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(t_d.values, np.asarray(t_c), atol=1e-12, equal_nan=True)
    c_ref, _ = _ols_oracle(y.values, h, p)
    np.testing.assert_allclose(c_d.values, c_ref, atol=1e-10, equal_nan=True)


def test_data_wrapper_ndarray_gets_rangeindex():
    y = _random_walk(100, seed=3)
    c, t = data.hamilton_filter(y, h=8, p=4)
    assert isinstance(c, pd.Series) and isinstance(t, pd.Series)
    assert c.index.equals(pd.RangeIndex(100))


@pytest.mark.parametrize(
    "h,p,bad,good",
    [(8, 4, [13, 14, 15, 16], 17), (24, 12, [47, 48], 49)],
)
@pytest.mark.parametrize("fn", [cycles.hamilton_filter, data.hamilton_filter])
def test_under_identified_sample_raises(fn, h, p, bad, good):
    for T in bad:
        y = _random_walk(T, seed=T)
        with pytest.raises(ValueError, match=rf"h={h}.*p={p}.*{h + 2 * p + 1}"):
            fn(y, h=h, p=p)
    y = _random_walk(good, seed=good)
    c, _ = fn(y, h=h, p=p)
    c = np.asarray(c, dtype=float)
    c_ref, _ = _ols_oracle(y, h, p)
    np.testing.assert_allclose(c, c_ref, atol=1e-10, equal_nan=True)
    assert np.nanmax(np.abs(c)) > 1e-6


def test_guard_applies_after_alias_resolution():
    y = _random_walk(48, seed=1)
    with pytest.raises(ValueError, match="49"):
        cycles.hamilton_filter(y, horizon=24, lags=12)
    cycles.hamilton_filter(_random_walk(49, seed=1), horizon=24, lags=12)


def test_linear_trend_has_zero_cycle():
    y = 2.0 + 0.3 * np.arange(120, dtype=float)
    c, t = cycles.hamilton_filter(y, h=8, p=4)
    np.testing.assert_allclose(c[11:], 0.0, atol=1e-10)


@pytest.mark.parametrize("fn", [cycles.hamilton_filter, data.hamilton_filter])
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_input_raises_valueerror(fn, bad):
    y = _random_walk(100, seed=5)
    y[50] = bad
    with pytest.raises(ValueError, match="finite"):
        fn(y, h=8, p=4)


@pytest.mark.parametrize("fn", [cycles.hamilton_filter, data.hamilton_filter])
@pytest.mark.parametrize("h,p", [(0, 4), (8, 0), (-1, 4), (2.5, 4)])
def test_invalid_h_p_raise(fn, h, p):
    with pytest.raises(ValueError):
        fn(_random_walk(100), h=h, p=p)


def test_data_wrapper_delegates_to_cycles(monkeypatch):
    """``data.hamilton_filter`` calls ``cycles.hamilton_filter`` once, with the
    float array and the same ``h``/``p``, and returns its cycle and trend."""
    real = cycles.hamilton_filter
    calls = []

    def spy(y, **kwargs):
        calls.append((np.array(y, copy=True), kwargs))
        return real(y, **kwargs)

    monkeypatch.setattr(cycles, "hamilton_filter", spy)
    y = pd.Series(_random_walk(120, seed=11), index=pd.RangeIndex(120))
    c, t = data.hamilton_filter(y, h=8, p=4)

    assert len(calls) == 1
    y_seen, kwargs = calls[0]
    assert kwargs == {"h": 8, "p": 4}
    assert y_seen.dtype == float
    np.testing.assert_array_equal(y_seen, y.values)
    res = real(y.values, h=8, p=4)
    np.testing.assert_array_equal(c.values, np.asarray(res.cycle, dtype=float))
    np.testing.assert_array_equal(t.values, np.asarray(res.trend, dtype=float))
