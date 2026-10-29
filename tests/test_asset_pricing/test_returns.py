"""Tests for puremacro.asset_pricing.returns and the shared input validation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.asset_pricing import (
    annualize_return,
    annualize_volatility,
    compound,
    excess_returns,
    to_returns,
)
from puremacro.asset_pricing._validate import _align, _validate_returns


def test_to_returns_simple_and_log():
    p = pd.Series([100.0, 110.0, 99.0], name="px")
    r = to_returns(p)
    np.testing.assert_allclose(r.to_numpy(), [0.1, -0.1], atol=1e-15)
    assert r.name == "px"
    lr = to_returns(p, kind="log")
    np.testing.assert_allclose(lr.to_numpy(), np.log([1.1, 0.9]), atol=1e-15)
    df = to_returns(pd.DataFrame({"a": [100.0, 110.0, 99.0], "b": [1.0, 2.0, 1.0]}))
    assert list(df.columns) == ["a", "b"]
    np.testing.assert_allclose(df["b"].to_numpy(), [1.0, -0.5])
    with pytest.raises(ValueError, match="kind"):
        to_returns(p, kind="pct")


def test_to_returns_rejects_nonpositive_prices():
    with pytest.raises(ValueError, match="positive"):
        to_returns(pd.Series([1.0, 0.0, 2.0]))
    with pytest.raises(ValueError, match="positive"):
        to_returns(pd.DataFrame({"a": [1.0, 2.0], "b": [1.0, -1.0]}))


def _monthly(n, start="2000-01-01"):
    return pd.date_range(start, periods=n, freq="MS")


def test_excess_returns_aligns_by_index_and_checks_frequency():
    idx = _monthly(24)
    rng = np.random.default_rng(0)
    r = pd.DataFrame(rng.normal(0.01, 0.04, (24, 2)), index=idx, columns=["a", "b"])
    rf = pd.Series(np.linspace(0.001, 0.003, 24), index=idx, name="RF")
    expected = r.sub(rf, axis=0)
    # rf with one extra date, in shuffled order: pairs by date, not position
    rf_extra = pd.concat([rf, pd.Series([0.5], index=[idx[-1] + pd.offsets.MonthBegin(1)])])
    rf_shuffled = rf_extra.sample(frac=1.0, random_state=1)
    got = excess_returns(r, rf_shuffled)
    pd.testing.assert_frame_equal(got, expected, check_freq=False)
    got_s = excess_returns(r["a"], rf_shuffled)
    assert isinstance(got_s, pd.Series)
    np.testing.assert_allclose(got_s.to_numpy(), expected["a"].to_numpy())
    # monthly returns vs a quarterly risk-free rate: refused
    rf_q = pd.Series(0.003, index=pd.date_range("2000-01-01", periods=8, freq="QS"))
    with pytest.raises(ValueError, match="frequenc"):
        excess_returns(r, rf_q)
    # positional when no DatetimeIndex: lengths must match
    with pytest.raises(ValueError, match="length"):
        excess_returns(np.zeros(5), np.zeros(4))
    np.testing.assert_allclose(excess_returns(np.full(3, 0.02), 0.01), np.full(3, 0.01))


def test_percent_scaled_input_rejected():
    with pytest.raises(ValueError, match="percent"):
        _validate_returns(pd.Series([1.2, -0.4, 2.5], name="Mkt-RF"), "returns")
    r = pd.Series([0.012, -0.004, 0.025], index=_monthly(3))
    rf = pd.Series([5.0, 5.1, 5.2], index=_monthly(3))  # TB3MS-style, annual percent
    with pytest.raises(ValueError, match="percent"):
        excess_returns(r, rf)


def test_nan_rejected_names_column():
    df = pd.DataFrame({"good": [0.01, 0.02, 0.0], "bad": [0.01, np.nan, 0.0],
                       "worse": [np.inf, 0.0, 0.0]})
    with pytest.raises(ValueError, match="bad") as exc:
        _validate_returns(df, "returns")
    assert "worse" in str(exc.value) and "good" not in str(exc.value)


def test_validate_shapes_and_names():
    v = _validate_returns(np.zeros((4, 3)), "returns")
    assert list(v.columns) == ["r0", "r1", "r2"]
    s = _validate_returns(pd.Series([0.0, 0.01], name="spy"), "returns")
    assert list(s.columns) == ["spy"]
    a, b = _align(pd.DataFrame({"x": [1.0, 2.0]}), pd.DataFrame({"y": [3.0, 4.0]}))
    assert len(a) == len(b) == 2


def test_compound_monthly_to_quarterly():
    idx = _monthly(6)
    r = pd.Series(0.01, index=idx, name="a")
    q = compound(r, to="Q")
    np.testing.assert_allclose(q.to_numpy(), [1.01 ** 3 - 1] * 2, atol=1e-15)
    assert list(q.index) == [pd.Timestamp("2000-01-01"), pd.Timestamp("2000-04-01")]
    ql = compound(r, to="Q", kind="log")
    np.testing.assert_allclose(ql.to_numpy(), [0.03, 0.03], atol=1e-15)
    a = compound(pd.DataFrame({"a": r, "b": 2 * r}), to="A")
    np.testing.assert_allclose(a.loc[pd.Timestamp("2000-01-01"), "b"], 1.02 ** 6 - 1)
    with pytest.raises(ValueError, match="DatetimeIndex"):
        compound(pd.Series([0.01, 0.02]), to="Q")
    with pytest.raises(ValueError, match="to"):
        compound(r, to="W")


def test_annualize_helpers():
    r = pd.Series(np.full(24, 0.01))
    np.testing.assert_allclose(annualize_return(r, 12), 1.01 ** 12 - 1, atol=1e-14)
    pm = pd.Series(np.tile([0.01, -0.01], 50))
    np.testing.assert_allclose(annualize_volatility(pm, 12), pm.std(ddof=1) * np.sqrt(12))
    df = pd.DataFrame({"a": r, "b": r})
    out = annualize_return(df, 12)
    assert isinstance(out, pd.Series) and list(out.index) == ["a", "b"]
    with pytest.raises(ValueError, match="periods_per_year"):
        annualize_return(r, 0)
    with pytest.raises(TypeError):
        annualize_volatility(r)  # periods_per_year is required


def test_scale_guard_accepts_genuine_large_decimal_returns():
    # Real decimal data can exceed 100% in one period (25 Size-B/M SMALL LoBM,
    # 1932-07: 1.478): a single large value must not be mistaken for percent.
    from puremacro.asset_pricing import risk_summary, time_series_test

    rng = np.random.default_rng(20)
    idx = _monthly(600)
    r = pd.Series(rng.normal(0.01, 0.06, 600), index=idx, name="small_lobm")
    r.iloc[78] = 1.478
    assert _validate_returns(r, "returns").shape == (600, 1)
    risk_summary(r, periods_per_year=12)
    f = pd.DataFrame({"mkt": rng.normal(0.005, 0.04, 600)}, index=idx)
    time_series_test(r.to_frame(), f)
    # a percent-scaled series still raises, with a message that does not tell
    # decimal users to divide by 100
    with pytest.raises(ValueError, match="percent") as exc:
        _validate_returns(r * 100, "returns")
    assert "check_scale=False" in str(exc.value)
    assert "divide by 100" not in str(exc.value)
    # a column whose share of |r| > 1 exceeds 5% raises even with a small median
    crazy = pd.Series(np.where(np.arange(100) % 10 == 0, 1.5, 0.01))
    with pytest.raises(ValueError, match="percent"):
        _validate_returns(crazy, "returns")


def test_check_scale_false_bypasses_the_guard():
    from puremacro.asset_pricing import (
        annualize_volatility, risk_summary, time_series_test, value_at_risk,
    )

    rng = np.random.default_rng(21)
    idx = _monthly(300)
    pct = pd.DataFrame(rng.normal(1.0, 5.0, (300, 2)), index=idx, columns=["a", "b"])
    with pytest.raises(ValueError, match="percent"):
        risk_summary(pct, periods_per_year=12)
    assert risk_summary(pct, periods_per_year=12, check_scale=False).sharpe.shape == (2,)
    assert np.isfinite(annualize_volatility(pct["a"], 12, check_scale=False))
    assert np.isfinite(value_at_risk(pct["a"], check_scale=False))
    excess_returns(pct, pd.Series(0.5, index=idx), check_scale=False)
    f = pd.DataFrame({"m": rng.normal(1.0, 4.0, 300)}, index=idx)
    time_series_test(pct, f, check_scale=False)
