"""Tests for puremacro.asset_pricing.fama_macbeth."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from puremacro.asset_pricing import FamaMacBethResult, fama_macbeth


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _dgp(seed=1, T=600, N=25, K=3):
    rng = np.random.default_rng(seed)
    f = 0.005 + 0.04 * rng.standard_normal((T, K))
    B = rng.uniform(0.3, 1.7, (K, N))
    e = 0.02 * rng.standard_normal((T, N))
    r = f @ B + e
    idx = pd.date_range("1960-01-01", periods=T, freq="MS")
    return (pd.DataFrame(r, index=idx, columns=[f"p{i}" for i in range(N)]),
            pd.DataFrame(f, index=idx, columns=["mkt", "smb", "hml"][:K]))


def test_lambdas_match_linearmodels():
    pytest.importorskip("linearmodels", minversion="6.0")
    from linearmodels.asset_pricing import LinearFactorModel

    R, F = _dgp()
    nc = fama_macbeth(R, F, intercept=False)
    ref = LinearFactorModel(R, F, risk_free=False).fit()
    np.testing.assert_allclose(nc.coefs, ref.risk_premia.to_numpy(), atol=1e-10)
    wc = fama_macbeth(R, F, intercept=True)
    ref = LinearFactorModel(R, F, risk_free=True).fit()
    np.testing.assert_allclose(wc.coefs, ref.risk_premia.to_numpy(), atol=1e-10)
    assert wc.names == ("const", "mkt", "smb", "hml")
    np.testing.assert_allclose(wc.lambdas, wc.coefs)


def test_shanken_close_to_linearmodels_gmm_se():
    # linearmodels reports a GMM robust se for the risk premia, asymptotically
    # equal to the Shanken (1992) se under iid errors but not the same
    # finite-sample definition (measured gap ~0.5% here), hence rtol 0.02.
    pytest.importorskip("linearmodels", minversion="6.0")
    from linearmodels.asset_pricing import LinearFactorModel

    R, F = _dgp(seed=2, T=6_000)
    res = fama_macbeth(R, F, intercept=False)
    ref = LinearFactorModel(R, F, risk_free=False).fit(cov_type="robust", debiased=False)
    np.testing.assert_allclose(res.se_shanken, ref.risk_premia_se.to_numpy(), rtol=0.02)
    np.testing.assert_allclose(res.se, res.se_shanken)


def test_shanken_exceeds_fm_for_nontraded_factor():
    rng = np.random.default_rng(3)
    T, N = 600, 25
    f = 0.01 * rng.standard_normal((T, 1))          # non-traded, sd 0.01
    beta = rng.uniform(0.5, 1.5, N)
    lam = 0.05                                       # lambda' Sigma_f^-1 lambda ~ 25
    r = lam * beta + (f - f.mean()) * beta + 0.02 * rng.standard_normal((T, N))
    res = fama_macbeth(r, f, intercept=False)
    assert res.se_shanken[0] > 1.5 * res.se_fm[0]


def test_given_betas_path():
    R, F = _dgp()
    est = fama_macbeth(R, F)
    given = fama_macbeth(R, betas=est.betas)
    np.testing.assert_allclose(given.coefs, est.coefs, atol=1e-12)
    np.testing.assert_allclose(given.se_fm, est.se_fm, atol=1e-12)
    assert np.all(np.isnan(given.se_shanken))
    np.testing.assert_allclose(given.se, given.se_fm)
    assert given.names[0] == "const"


def test_factors_and_betas_exclusive():
    R, F = _dgp()
    with pytest.raises(ValueError, match="exactly one"):
        fama_macbeth(R)
    with pytest.raises(ValueError, match="exactly one"):
        fama_macbeth(R, F, betas=np.ones((25, 3)))
    with pytest.raises(ValueError, match="betas"):
        fama_macbeth(R, betas=np.ones((24, 3)))


def test_nw_se_equals_fm_at_zero_lags():
    R, F = _dgp()
    res = fama_macbeth(R, F, hac_lags=0)
    T = res.n_obs
    np.testing.assert_allclose(res.se_nw, res.se_fm * np.sqrt((T - 1) / T), rtol=0, atol=1e-12)
    np.testing.assert_allclose(res.lambda_series.mean(axis=0), res.coefs, atol=1e-14)
    fm = fama_macbeth(R, F, shanken=False)
    np.testing.assert_allclose(fm.se, fm.se_fm)


def test_cs_r2_and_pricing_errors():
    R, F = _dgp()
    res = fama_macbeth(R, F)
    Z = np.column_stack([np.ones(25), res.betas])
    np.testing.assert_allclose(res.pricing_errors, res.mean_returns - Z @ res.coefs, atol=1e-14)
    np.testing.assert_allclose(res.cs_r2, 1 - res.pricing_errors.var() / res.mean_returns.var())


def test_report_quintet_and_plot():
    R, F = _dgp(T=120, N=6, K=2)
    res = fama_macbeth(R, F)
    assert isinstance(res, FamaMacBethResult)
    df = res.to_dataframe()
    assert list(df["name"]) == ["const", "mkt", "smb"]
    assert {"se_fm", "se_shanken", "se_nw"} <= set(df.columns)
    assert res.to_frame().equals(df)
    assert "|" in res.to_markdown()
    assert "\\begin{tabular}" in res.to_latex()
    assert "#table(" in res.to_typst()
    assert "Shanken" in res.summary()
    assert isinstance(res.plot(), matplotlib.figure.Figure)
    _, ax = plt.subplots()
    assert res.plot(ax=ax) is ax
