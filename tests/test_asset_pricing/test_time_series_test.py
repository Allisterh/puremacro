"""Tests for puremacro.asset_pricing.time_series_test (alphas, GRS, HAC Wald)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from puremacro.asset_pricing import TimeSeriesTestResult, time_series_test


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _dgp(seed=0, T=600, N=10, K=3, alpha=0.0, hetero=True, sd=0.02):
    rng = np.random.default_rng(seed)
    f = 0.005 + 0.04 * rng.standard_normal((T, K))
    B = rng.uniform(0.5, 1.5, (K, N))
    scale = (1.0 + np.abs(f[:, :1]) / 0.04) if hetero else 1.0
    e = sd * rng.standard_normal((T, N)) * scale
    r = alpha + f @ B + e
    idx = pd.date_range("1970-01-01", periods=T, freq="MS")
    R = pd.DataFrame(r, index=idx, columns=[f"p{i}" for i in range(N)])
    F = pd.DataFrame(f, index=idx, columns=[f"f{k}" for k in range(K)])
    return R, F


def test_alphas_betas_se_match_statsmodels_hac():
    sm = pytest.importorskip("statsmodels.api")
    R, F = _dgp()
    res = time_series_test(R, F, hac_lags=4)
    X = np.column_stack([np.ones(len(F)), F.to_numpy()])
    for i, c in enumerate(R.columns):
        ref = sm.OLS(R[c].to_numpy(), X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
        np.testing.assert_allclose(res.coefs[i], ref.params, atol=1e-8)
        np.testing.assert_allclose(res.se[i], ref.bse, atol=1e-8)
        np.testing.assert_allclose(res.r2[i], ref.rsquared, atol=1e-8)
    np.testing.assert_allclose(res.alphas, res.coefs[:, 0])
    np.testing.assert_allclose(res.betas, res.coefs[:, 1:])
    assert res.cov.shape == (40, 40)


def test_wald_matches_linearmodels_traded_factor_model():
    pytest.importorskip("linearmodels", minversion="6.0")
    from linearmodels.asset_pricing import TradedFactorModel

    R, F = _dgp()
    mod = TradedFactorModel(R, F)
    r0 = time_series_test(R, F, hac_lags=0)
    ref0 = mod.fit(cov_type="robust", debiased=False)
    np.testing.assert_allclose(r0.wald_stat, ref0.j_statistic.stat, atol=1e-8)
    np.testing.assert_allclose(r0.alphas, ref0.alphas.to_numpy(), atol=1e-10)
    np.testing.assert_allclose(r0.betas, ref0.betas.to_numpy(), atol=1e-10)
    r4 = time_series_test(R, F, hac_lags=4)
    ref4 = mod.fit(cov_type="kernel", kernel="bartlett", bandwidth=4, debiased=False)
    np.testing.assert_allclose(r4.wald_stat, ref4.j_statistic.stat, atol=1e-8)
    assert r4.wald_df == 10
    np.testing.assert_allclose(r4.wald_pvalue, stats.chi2.sf(r4.wald_stat, 10))


def test_grs_iid_wald_identity():
    R, F = _dgp(alpha=0.001)
    res = time_series_test(R, F, hac_lags=0)
    T, N, K = 600, 10, 3
    X = np.column_stack([np.ones(T), F.to_numpy()])
    coef = np.linalg.lstsq(X, R.to_numpy(), rcond=None)[0]
    e = R.to_numpy() - X @ coef
    a = coef[0]
    Sig = e.T @ e / T
    mu = F.to_numpy().mean(0)
    fc = F.to_numpy() - mu
    Om = fc.T @ fc / T
    w = T * a @ np.linalg.solve(Sig, a) / (1 + mu @ np.linalg.solve(Om, mu))
    np.testing.assert_allclose(res.grs_stat * N * T / (T - N - K), w, rtol=0, atol=1e-10)
    assert res.grs_df == (N, T - N - K)
    np.testing.assert_allclose(res.grs_pvalue, stats.f.sf(res.grs_stat, N, T - N - K))


def test_grs_exact_size():
    rej = 0
    for s in range(2_000):
        R, F = _dgp(seed=1000 + s, T=60, N=10, K=3, hetero=False)
        rej += time_series_test(R, F, hac_lags=0).grs_pvalue < 0.05
    assert 0.035 <= rej / 2_000 <= 0.065


def test_wald_size_large_t():
    rej = 0
    for s in range(2_000):
        R, F = _dgp(seed=5000 + s, T=600, N=5, K=3, hetero=False)
        rej += time_series_test(R, F, hac_lags=0).wald_pvalue < 0.05
    assert 0.035 <= rej / 2_000 <= 0.075, rej / 2_000


def test_grs_power():
    rej = 0
    for s in range(200):
        R, F = _dgp(seed=9000 + s, T=120, N=10, K=3, alpha=0.005, hetero=False, sd=0.02)
        rej += time_series_test(R, F, hac_lags=0).grs_pvalue < 0.05
    assert rej / 200 > 0.9


def test_grs_requires_t_greater_than_n_plus_k():
    R, F = _dgp(T=13, N=10, K=3)
    with pytest.raises(ValueError, match="T > N \\+ K"):
        time_series_test(R, F)
    R, F = _dgp(T=100, N=4, K=2)
    R["dup"] = R["p0"] + 0.5 * F["f0"]  # residual identical to p0's residual
    R["dup2"] = R["p0"]
    with pytest.raises(ValueError, match="singular"):
        time_series_test(R, F)


def test_nontraded_factor_reports_nan_grs():
    R, F = _dgp()
    res = time_series_test(R, F, traded=False)
    assert np.isnan(res.grs_stat) and np.isnan(res.grs_pvalue)
    assert np.isfinite(res.wald_stat)
    assert "traded" in res.summary()


def test_time_series_test_aligns_dataframes_by_index():
    R, F = _dgp()
    base = time_series_test(R, F, hac_lags=2)
    F_extra = pd.concat([F, pd.DataFrame(0.01, index=[F.index[-1] + pd.offsets.MonthBegin(1)],
                                         columns=F.columns)])
    shuffled = F_extra.sample(frac=1.0, random_state=3)
    got = time_series_test(R, shuffled, hac_lags=2)
    np.testing.assert_allclose(got.coefs, base.coefs, atol=1e-14)
    np.testing.assert_allclose(got.wald_stat, base.wald_stat, atol=1e-10)
    np.testing.assert_allclose(got.grs_stat, base.grs_stat, atol=1e-10)
    with pytest.raises(ValueError, match="percent"):
        time_series_test(R, F * 100)


def test_report_quintet_and_plot():
    R, F = _dgp(T=120, N=4, K=2)
    res = time_series_test(R, F)
    assert isinstance(res, TimeSeriesTestResult)
    assert res.names == tuple(R.columns) and res.factor_names == ("f0", "f1")
    df = res.to_dataframe()
    assert list(df["asset"]) == list(R.columns) and "alpha" in df.columns
    assert res.to_frame().equals(df)
    assert "|" in res.to_markdown()
    assert "\\begin{tabular}" in res.to_latex()
    assert "#table(" in res.to_typst()
    assert "GRS" in res.summary()
    assert isinstance(res.plot(), matplotlib.figure.Figure)
    _, ax = plt.subplots()
    assert res.plot(ax=ax) is ax
