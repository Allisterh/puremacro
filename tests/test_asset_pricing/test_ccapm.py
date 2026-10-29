"""Tests for the Hansen-Singleton CCAPM and the FRED macro-data helpers."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from puremacro.asset_pricing import CCAPMResult, ccapm


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


BETA, GAMMA, MU, SD = 0.98, 4.0, 0.005, 0.02


def _Eg(a):
    """E[g^a] for log g ~ N(MU, SD^2)."""
    return np.exp(a * MU + 0.5 * a ** 2 * SD ** 2)


def _lucas(T=50_000, seed=2):
    rng = np.random.default_rng(seed)
    g = np.exp(MU + SD * rng.standard_normal(T))
    q1 = BETA * _Eg(1 - GAMMA)
    q2 = BETA * _Eg(3 - GAMMA)
    k1, k2 = q1 / (1 - q1), q2 / (1 - q2)
    R1 = (k1 + 1) / k1 * g
    R2 = (k2 + 1) / k2 * g ** 3
    rf = np.full(T, 1.0 / (BETA * _Eg(-GAMMA)))
    idx = pd.date_range("1800-01-01", periods=T, freq="D")
    R = pd.DataFrame({"equity": R1, "levered": R2}, index=idx)
    return R, pd.Series(g, index=idx, name="g"), pd.Series(rf, index=idx, name="rf")


def test_ccapm_recovers_beta_gamma():
    R, g, rf = _lucas()
    res = ccapm(R, g, rf=rf, instruments=np.ones((len(g), 1)))
    assert res.converged
    assert res.names == ("beta", "gamma")
    assert np.all(np.abs(res.coefs - np.array([BETA, GAMMA])) < 4 * res.se), (res.coefs, res.se)


def test_ccapm_j_has_df_one():
    R, g, rf = _lucas()
    res = ccapm(R, g, rf=rf, instruments=np.ones((len(g), 1)))
    assert res.j_df == 1
    assert res.j_pvalue > 0.001
    assert res.pricing_errors.shape == (3,)
    assert res.n_obs == len(g) - 1


def test_ccapm_moment_means_zero_at_truth():
    from puremacro.asset_pricing.ccapm import _moments, _prepare

    R, g, rf = _lucas()
    data = _prepare(R, g, np.ones((len(g), 1)), rf)
    u = _moments(np.array([BETA, GAMMA]), data)
    se = u.std(axis=0, ddof=1) / np.sqrt(u.shape[0])
    assert np.all(np.abs(u.mean(axis=0)) < 4 * se)
    # timing: the moment for row t uses g and R dated t+1
    gs, Rs = g.to_numpy(), R.to_numpy()
    np.testing.assert_allclose(u[0, 0], BETA * gs[1] ** -GAMMA * Rs[1, 0] - 1.0)


def test_ccapm_default_instruments_shape():
    R, g, _ = _lucas(T=2_000)
    # In the Lucas tree both returns are exact functions of g, so the default
    # instruments [1, g_t, R_1t, R_2t] would be collinear; add idiosyncratic
    # noise (mean one) to make them a valid instrument set.
    rng = np.random.default_rng(9)
    R = R * np.exp(0.05 * rng.standard_normal(R.shape) - 0.5 * 0.05 ** 2)
    res = ccapm(R, g)
    assert res.gmm.moment_means.shape == (8,)   # N (Z = [1, g_t, R_1t, R_2t]) = 2 * 4
    assert res.n_obs == 2_000 - 1
    assert res.j_df == 6


def test_ccapm_rejects_net_returns():
    R, g, rf = _lucas(T=500)
    with pytest.raises(ValueError, match="gross"):
        ccapm(R - 1.0, g)
    with pytest.raises(ValueError, match="gross"):
        ccapm(R, g - 1.0)
    with pytest.raises(ValueError, match="gross"):
        ccapm(R, g, rf=rf - 1.0)
    bad = R.copy()
    bad.iloc[3, 1] = np.nan
    with pytest.raises(ValueError, match="levered"):
        ccapm(bad, g)


def _synthetic_fred(missing=()):
    idx = pd.date_range("2000-01-01", periods=12, freq="MS")
    n = np.arange(12, dtype=float)
    df = pd.DataFrame({
        "PCEND": 100.0 + n,
        "PCES": 300.0 + 2 * n,
        "DNDGRG3M086SBEA": 100.0 * (1.0 + 0.01 * n),
        "DSERRG3M086SBEA": np.full(12, 100.0),
        "POPTHM": np.full(12, 1000.0) + n,
        "TB3MS": np.linspace(1.2, 2.4, 12),
    }, index=pd.DatetimeIndex(idx, name="date"))
    for m in missing:
        df = df.drop(columns=m)
    df.attrs["missing"] = [{"id": m, "reason": "HTTP 404"} for m in missing]
    return df


def test_fred_consumption_growth_raises_on_missing_series(monkeypatch):
    import puremacro.fetch.rates as rates
    from puremacro.asset_pricing import fred_consumption_growth

    def fake(ids, **kw):
        # fetch_fred_many never raises: the failed id is only listed in attrs
        d = _synthetic_fred(missing=("DSERRG3M086SBEA",))
        out = d[[i for i in ids if i in d.columns]].copy()
        out.attrs["missing"] = d.attrs["missing"]
        return out

    monkeypatch.setattr(rates, "fetch_fred_many", fake)
    with pytest.raises(RuntimeError, match="DSERRG3M086SBEA"):
        fred_consumption_growth()


def test_fred_helpers_transform(monkeypatch):
    import puremacro.fetch.rates as rates
    from puremacro.asset_pricing import fred_consumption_growth, fred_risk_free

    calls = []

    def fake(ids, **kw):
        calls.append(list(ids))
        d = _synthetic_fred()[list(ids)]
        d.attrs["missing"] = []
        return d

    monkeypatch.setattr(rates, "fetch_fred_many", fake)
    src = _synthetic_fred()
    real = (src["PCEND"] / (src["DNDGRG3M086SBEA"] / 100) + src["PCES"] / (src["DSERRG3M086SBEA"] / 100))
    pc = real / src["POPTHM"]
    gm = fred_consumption_growth(frequency="M")
    np.testing.assert_allclose(gm.to_numpy(), (pc / pc.shift(1)).iloc[1:].to_numpy(), rtol=1e-14)
    assert gm.index[0] == pd.Timestamp("2000-02-01")
    gq = fred_consumption_growth(frequency="Q")
    q = pc.groupby(pc.index.to_period("Q")).mean().to_numpy()
    np.testing.assert_allclose(gq.to_numpy(), q[1:] / q[:-1], rtol=1e-14)
    assert list(gq.index) == [pd.Timestamp("2000-04-01"), pd.Timestamp("2000-07-01"),
                              pd.Timestamp("2000-10-01")]
    rfm = fred_risk_free()
    np.testing.assert_allclose(rfm.to_numpy(), src["TB3MS"].to_numpy() / 100 / 12)
    rfq = fred_risk_free(frequency="Q")
    tq = src["TB3MS"].groupby(src.index.to_period("Q")).mean().to_numpy()
    np.testing.assert_allclose(rfq.to_numpy(), tq / 100 / 4)
    assert calls[0] == ["PCEND", "PCES", "DNDGRG3M086SBEA", "DSERRG3M086SBEA", "POPTHM"]
    with pytest.raises(ValueError, match="frequency"):
        fred_risk_free(frequency="A")


def test_report_quintet_and_plot():
    R, g, rf = _lucas(T=3_000)
    res = ccapm(R, g, rf=rf, instruments=np.ones((len(g), 1)))
    assert isinstance(res, CCAPMResult)
    df = res.to_dataframe()
    assert list(df["name"]) == ["beta", "gamma"] and {"ci_low", "ci_high"} <= set(df.columns)
    assert res.to_frame().equals(df)
    assert "|" in res.to_markdown()
    assert "\\begin{tabular}" in res.to_latex()
    assert "#table(" in res.to_typst()
    s = res.summary()
    assert "J" in s and "95%" in s
    assert isinstance(res.plot(), matplotlib.figure.Figure)
    _, ax = plt.subplots()
    assert res.plot(ax=ax) is ax
