"""Tests for value_at_risk, expected_shortfall and var_backtest."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from puremacro.asset_pricing import (
    VaRBacktestResult,
    expected_shortfall,
    value_at_risk,
    var_backtest,
)


def test_historical_var_es_known_sample():
    r = pd.Series(np.round(np.arange(-0.10, 0.0999, 0.01), 10), name="a")
    assert r.size == 20
    var = value_at_risk(r, alpha=0.1)
    q = np.quantile(r.to_numpy(), 0.1)
    assert isinstance(var, float)
    np.testing.assert_allclose(var, -q, atol=1e-15)
    es = expected_shortfall(r, alpha=0.1)
    np.testing.assert_allclose(es, -r[r <= q].mean(), atol=1e-15)
    # horizon: overlapping two-period sums
    s2 = r.to_numpy()[1:] + r.to_numpy()[:-1]
    np.testing.assert_allclose(value_at_risk(r, alpha=0.1, horizon=2),
                               -np.quantile(s2, 0.1), atol=1e-15)
    df = pd.DataFrame({"a": r, "b": 2 * r})
    out = value_at_risk(df, alpha=0.1)
    assert isinstance(out, pd.Series) and list(out.index) == ["a", "b"]
    np.testing.assert_allclose(out["b"], 2 * var, atol=1e-15)


def test_normal_var_closed_form():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.02, 1000))
    mu, sd = r.mean(), r.std(ddof=1)
    for h in (1, 5):
        z = stats.norm.ppf(0.01)
        np.testing.assert_allclose(value_at_risk(r, 0.01, method="normal", horizon=h),
                                   -(h * mu + np.sqrt(h) * sd * z), atol=1e-14)
        np.testing.assert_allclose(expected_shortfall(r, 0.01, method="normal", horizon=h),
                                   -(h * mu - np.sqrt(h) * sd * stats.norm.pdf(z) / 0.01),
                                   atol=1e-14)


def _simulate_garch(T, seed, omega=1e-6, a=0.08, b=0.9):
    rng = np.random.default_rng(seed)
    e = np.empty(T)
    s2 = omega / (1 - a - b)
    for t in range(T):
        e[t] = np.sqrt(s2) * rng.standard_normal()
        s2 = omega + a * e[t] ** 2 + b * s2
    return 0.0005 + e


def test_garch_var_uses_filtered_residuals():
    from puremacro.garch import garch11_fit

    r = _simulate_garch(3_000, seed=1)
    fit = garch11_fit(r, mean="constant")
    mean = r.mean()
    eps = r - mean
    sig = fit.sigma.to_numpy()
    z = eps / sig
    s_next = np.sqrt(fit.omega + fit.alpha * eps[-1] ** 2 + fit.beta * sig[-1] ** 2)
    qz = np.quantile(z, 0.05)
    np.testing.assert_allclose(value_at_risk(r, 0.05, method="garch"),
                               -(mean + s_next * qz), rtol=0, atol=1e-12)
    np.testing.assert_allclose(expected_shortfall(r, 0.05, method="garch"),
                               -(mean + s_next * z[z <= qz].mean()), rtol=0, atol=1e-12)


def test_garch_horizon_rejected():
    r = _simulate_garch(500, seed=2)
    with pytest.raises(ValueError, match="horizon"):
        value_at_risk(r, method="garch", horizon=5)
    with pytest.raises(ValueError, match="method"):
        value_at_risk(r, method="cornish-fisher")
    with pytest.raises(ValueError, match="alpha"):
        expected_shortfall(r, alpha=1.5)


def test_backtest_size_correct_var():
    rng = np.random.default_rng(10)
    var = -0.01 * stats.norm.ppf(0.05)
    rej = 0
    for _ in range(1_000):
        r = 0.01 * rng.standard_normal(1_000)
        rej += var_backtest(r, var, alpha=0.05).p_uc < 0.05
    assert 0.03 <= rej / 1_000 <= 0.075


def test_backtest_power_wrong_var():
    rng = np.random.default_rng(11)
    var = -0.01 * stats.norm.ppf(0.10)  # the 10% quantile labelled as 5% VaR
    rej = 0
    for _ in range(200):
        r = 0.01 * rng.standard_normal(1_000)
        rej += var_backtest(r, var, alpha=0.05).p_uc < 0.05
    assert rej / 200 > 0.9


def test_independence_detects_clustering():
    rng = np.random.default_rng(12)
    T, reps = 2_000, 500
    var = 0.05
    hits = 0
    for _ in range(reps):
        u = rng.random(T)
        v = np.zeros(T, dtype=bool)
        for t in range(1, T):
            v[t] = u[t] < (0.4 if v[t - 1] else 0.02)
        r = np.where(v, -0.06, 0.0)
        res = var_backtest(r, var, alpha=0.05)
        hits += res.p_ind < 0.05
    assert hits / reps > 0.8


def test_backtest_zero_violations_is_finite():
    r = np.full(250, 0.001)
    res = var_backtest(r, 0.02, alpha=0.01)
    assert isinstance(res, VaRBacktestResult)
    assert res.n_violations == 0 and res.violation_rate == 0.0
    for f in ("lr_uc", "p_uc", "lr_ind", "p_ind", "lr_cc", "p_cc"):
        assert np.isfinite(getattr(res, f)), f
    np.testing.assert_allclose(res.lr_uc, -2 * 250 * np.log(0.99))
    assert res.lr_ind == 0.0
    allv = var_backtest(np.full(50, -0.05), 0.02, alpha=0.01)
    assert allv.n_violations == 50 and np.isfinite(allv.lr_uc) and np.isfinite(allv.lr_ind)


def test_backtest_matches_manual_counts_and_aligns():
    idx = pd.date_range("2020-01-01", periods=8, freq="D")
    r = pd.Series([-0.03, -0.03, 0.01, -0.03, 0.0, 0.0, -0.03, 0.01], index=idx)
    var = pd.Series(0.02, index=idx).sample(frac=1.0, random_state=0)
    res = var_backtest(r, var, alpha=0.25)
    # violations 1 1 0 1 0 0 1 0 : n1 = 4; transitions 00:1 01:2 10:3 11:1
    assert res.n_violations == 4 and res.n_obs == 8
    x = np.array([1, 1, 0, 1, 0, 0, 1, 0])
    n00, n01, n10, n11 = 1, 2, 3, 1
    p = 0.5
    lr_uc = -2 * (4 * np.log(0.75) + 4 * np.log(0.25) - (8 * np.log(0.5)))
    np.testing.assert_allclose(res.lr_uc, lr_uc, atol=1e-12)
    p01, p11, pi = n01 / (n00 + n01), n11 / (n10 + n11), (n01 + n11) / 7
    l0 = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
    l1 = n00 * np.log(1 - p01) + n01 * np.log(p01) + n10 * np.log(1 - p11) + n11 * np.log(p11)
    np.testing.assert_allclose(res.lr_ind, -2 * (l0 - l1), atol=1e-12)
    np.testing.assert_allclose(res.lr_cc, res.lr_uc + res.lr_ind, atol=1e-14)
    np.testing.assert_allclose(res.p_cc, stats.chi2.sf(res.lr_cc, 2))
    assert x.sum() == 4 and p == 0.5


def test_backtest_report_quintet():
    res = var_backtest(0.01 * np.random.default_rng(0).standard_normal(300), 0.0165)
    df = res.to_dataframe()
    assert {"statistic", "value"} <= set(df.columns)
    assert res.to_frame().equals(df)
    assert "|" in res.to_markdown()
    assert "\\begin{tabular}" in res.to_latex()
    assert "#table(" in res.to_typst()
    assert "Kupiec" in res.summary() and "Christoffersen" in res.summary()


def test_garch_var_refuses_unconverged_fit(monkeypatch):
    import dataclasses

    import puremacro.garch as garch_pkg
    from puremacro.garch import garch11_fit as real_fit

    def unconverged(x, mean="zero"):
        return dataclasses.replace(real_fit(x, mean=mean), converged=False)

    monkeypatch.setattr(garch_pkg, "garch11_fit", unconverged)
    r = _simulate_garch(1_000, seed=3)
    with pytest.raises(ValueError, match="did not converge"):
        value_at_risk(r, method="garch")
    with pytest.raises(ValueError, match="did not converge"):
        expected_shortfall(r, method="garch")
