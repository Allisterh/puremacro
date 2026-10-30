"""Tests for puremacro.asset_pricing.risk_summary."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from puremacro.asset_pricing import RiskSummaryResult, risk_summary


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def test_sharpe_se_iid_closed_form():
    rng = np.random.default_rng(0)
    x = 0.005 + 0.02 * rng.standard_t(6, size=5_000) + 0.002 * rng.standard_normal(5_000) ** 2
    res = risk_summary(x, periods_per_year=1, hac_lags=0)
    T = x.size
    mu, sd = x.mean(), x.std(ddof=0)
    sr = mu / sd
    skew = np.mean((x - mu) ** 3) / sd ** 3
    kurt = np.mean((x - mu) ** 4) / sd ** 4
    closed = np.sqrt((1 - sr * skew + sr ** 2 * (kurt - 1) / 4) / T)
    np.testing.assert_allclose(res.sharpe_se[0], closed, rtol=0, atol=1e-10)
    np.testing.assert_allclose(res.sharpe[0], sr, atol=1e-12)
    assert res.hac_lags == 0


def test_sharpe_se_normal_large_t():
    rng = np.random.default_rng(1)
    T = 200_000
    x = 0.5 * 0.1 + 0.1 * rng.standard_normal(T)
    res = risk_summary(x, periods_per_year=1)
    sr = res.sharpe[0]
    np.testing.assert_allclose(res.sharpe_se[0] * np.sqrt(T), np.sqrt(1 + sr ** 2 / 2), rtol=0.02)


def test_drawdown_known_path():
    # wealth 1.1, 0.55, 0.66, 0.99, 1.089: peak 1.1, trough 0.55 -> drawdown 0.5;
    # four periods below the 1.1 peak (1.089 < 1.1 has not recovered).
    res = risk_summary(pd.Series([0.1, -0.5, 0.2, 0.5, 0.1], name="a"), periods_per_year=1)
    np.testing.assert_allclose(res.max_drawdown[0], 0.5, atol=1e-12)
    assert res.max_drawdown_duration[0] == 4
    # recovery above the peak ends the run: three periods under water
    rec = risk_summary(pd.Series([0.1, -0.5, 0.2, 0.5, 0.2, -0.01], name="a"), periods_per_year=1)
    assert rec.max_drawdown_duration[0] == 3
    up = risk_summary(pd.Series([0.01, 0.02, 0.03]), periods_per_year=1)
    assert up.max_drawdown[0] == 0.0 and up.max_drawdown_duration[0] == 0


def test_sortino_and_moments_match_pandas():
    rng = np.random.default_rng(2)
    x = rng.normal(0.008, 0.04, 600) + 0.01 * rng.standard_exponential(600)
    res = risk_summary(x, periods_per_year=12)
    np.testing.assert_allclose(res.skewness[0], stats.skew(x, bias=True), atol=1e-12)
    np.testing.assert_allclose(res.excess_kurtosis[0], stats.kurtosis(x, fisher=True, bias=True),
                               atol=1e-12)
    dd = np.sqrt(np.mean(np.minimum(x, 0.0) ** 2))
    np.testing.assert_allclose(res.sortino[0], x.mean() / dd * np.sqrt(12), atol=1e-12)
    np.testing.assert_allclose(res.mean[0], x.mean() * 12, atol=1e-14)
    np.testing.assert_allclose(res.volatility[0], x.std(ddof=1) * np.sqrt(12), atol=1e-14)
    np.testing.assert_allclose(res.sharpe[0], x.mean() / x.std(ddof=0) * np.sqrt(12), atol=1e-12)


def test_rf_subtracted_before_statistics():
    idx = pd.date_range("2001-01-01", periods=120, freq="MS")
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.01, 0.05, 120), index=idx, name="eq")
    rf = pd.Series(0.002, index=idx)
    a = risk_summary(r, rf=rf, periods_per_year=12)
    b = risk_summary(r - rf, periods_per_year=12)
    for f in ("mean", "sharpe", "sharpe_se", "sortino", "max_drawdown"):
        np.testing.assert_allclose(getattr(a, f), getattr(b, f), atol=1e-14)


def test_dataframe_columns_preserved():
    rng = np.random.default_rng(4)
    df = pd.DataFrame(rng.normal(0.01, 0.05, (100, 3)), columns=["us", "uk", "jp"])
    res = risk_summary(df, periods_per_year=12)
    assert res.names == ("us", "uk", "jp")
    assert res.sharpe.shape == (3,)
    for j, c in enumerate(df.columns):
        one = risk_summary(df[c], periods_per_year=12)
        np.testing.assert_allclose(res.sharpe_se[j], one.sharpe_se[0])
    with pytest.raises(ValueError, match="percent"):
        risk_summary(df * 100, periods_per_year=12)
    with pytest.raises(TypeError):
        risk_summary(df)  # periods_per_year is required


def test_report_quintet_and_plot():
    rng = np.random.default_rng(5)
    res = risk_summary(pd.DataFrame(rng.normal(0.01, 0.05, (60, 2)), columns=["a", "b"]),
                       periods_per_year=12)
    assert isinstance(res, RiskSummaryResult)
    df = res.to_dataframe()
    assert list(df["name"]) == ["a", "b"] and "sharpe_se" in df.columns
    assert res.to_frame().equals(df)
    assert "|" in res.to_markdown()
    assert "\\begin{tabular}" in res.to_latex()
    assert "#table(" in res.to_typst()
    assert "Sharpe" in res.summary()
    assert isinstance(res.plot(), matplotlib.figure.Figure)
    _, ax = plt.subplots()
    assert res.plot(ax=ax) is ax
