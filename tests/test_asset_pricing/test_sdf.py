"""Tests for sdf_gmm (linear SDF GMM) and hj_distance (Hansen-Jagannathan)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy import optimize

from puremacro.asset_pricing import HJDistanceResult, SDFResult, hj_distance, sdf_gmm


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _null(seed=0, T=600, N=6, K=2, sd=0.02):
    """R = f B + e with traded excess-return factors: m = 1 - b'f prices R."""
    rng = np.random.default_rng(seed)
    mu = np.array([0.006, 0.003, 0.004][:K])
    f = mu + 0.04 * rng.standard_normal((T, K))
    B = rng.uniform(0.5, 1.5, (K, N))
    r = f @ B + sd * rng.standard_normal((T, N))
    idx = pd.date_range("1950-01-01", periods=T, freq="MS")
    return (pd.DataFrame(r, index=idx, columns=[f"p{i}" for i in range(N)]),
            pd.DataFrame(f, index=idx, columns=[f"f{k}" for k in range(K)]), mu)


def _bartlett(g, L):
    T = g.shape[0]
    S = g.T @ g / T
    for l in range(1, L + 1):
        G = g[l:].T @ g[:-l] / T
        S += (1 - l / (L + 1)) * (G + G.T)
    return S


def test_first_stage_uncentered_closed_form():
    R, F, _ = _null()
    res = sdf_gmm(R, F, stage="first", demeaned=False)
    Rv, Fv = R.to_numpy(), F.to_numpy()
    D = Rv.T @ Fv / len(Rv)
    b = np.linalg.solve(D.T @ D, D.T @ Rv.mean(0))
    np.testing.assert_allclose(res.coefs, b, rtol=0, atol=1e-10)
    assert res.mu is None
    assert np.isnan(res.j_stat)


def test_two_step_matches_direct_minimization():
    R, F, _ = _null(seed=4)
    Rv, Fv = R.to_numpy(), F.to_numpy()
    T, N = Rv.shape
    K = Fv.shape[1]
    L = 2

    def g_t(th):
        b, mu = th[:K], th[K:]
        m = 1.0 - (Fv - mu) @ b
        return np.column_stack([Rv * m[:, None], Fv - mu])

    def G_of(th):  # derivative of the moment means, written out independently
        b, mu = th[:K], th[K:]
        G = np.zeros((N + K, 2 * K))
        G[:N, :K] = -Rv.T @ (Fv - mu) / T
        G[:N, K:] = np.outer(Rv.mean(0), b)
        G[N:, K:] = -np.eye(K)
        return G

    def obj(th, W):
        gb = g_t(th).mean(0)
        return T * gb @ W @ gb

    def grad(th, W):
        return 2.0 * T * G_of(th).T @ W @ g_t(th).mean(0)

    th0 = np.concatenate([np.zeros(K), Fv.mean(0)])
    opts = {"gtol": 1e-10, "maxiter": 10_000}
    th1 = optimize.minimize(obj, th0, args=(np.eye(N + K),), jac=grad, method="BFGS",
                            options=opts).x
    W = np.linalg.inv(_bartlett(g_t(th1), L))
    th2 = optimize.minimize(obj, th1, args=(W,), jac=grad, method="BFGS", options=opts).x
    res = sdf_gmm(R, F, stage="two_step", hac_lags=L)
    np.testing.assert_allclose(res.coefs, th2[:K], rtol=0, atol=1e-6)
    np.testing.assert_allclose(res.mu, th2[K:], rtol=0, atol=1e-6)
    assert res.j_df == N - K


def test_sdf_recovers_b_and_j_size():
    R, F, mu = _null(seed=7, T=20_000)
    Sf = 0.04 ** 2 * np.eye(2)
    res = sdf_gmm(R, F)                         # demeaned: b = Sigma_f^-1 mu
    assert np.all(np.abs(res.coefs - np.linalg.solve(Sf, mu)) < 4 * res.se)
    unc = sdf_gmm(R, F, demeaned=False)         # uncentered: b = E[ff']^-1 mu
    Eff = Sf + np.outer(mu, mu)
    assert np.all(np.abs(unc.coefs - np.linalg.solve(Eff, mu)) < 4 * unc.se)
    rej = 0
    for s in range(500):
        R, F, _ = _null(seed=100 + s, T=600)
        rej += sdf_gmm(R, F, hac_lags=0).j_pvalue < 0.05
    assert 0.02 <= rej / 500 <= 0.09, rej / 500


def test_implied_lambdas_close_to_linearmodels_gmm():
    # LinearFactorModelGMM estimates the beta representation by GMM, a
    # different estimator of the same risk premia: rtol 0.05, not 1e-8.
    pytest.importorskip("linearmodels", minversion="6.0")
    from linearmodels.asset_pricing import LinearFactorModelGMM

    R, F, _ = _null(seed=8, T=20_000)
    res = sdf_gmm(R, F)
    ref = LinearFactorModelGMM(R, F).fit(disp=0)
    np.testing.assert_allclose(res.lambdas, ref.risk_premia.to_numpy(), rtol=0.05)
    Fv = F.to_numpy()
    Sf = np.cov(Fv.T, ddof=0)
    np.testing.assert_allclose(res.lambdas, Sf @ res.coefs, atol=1e-14)
    assert res.lambda_se.shape == (2,)


def test_hj_closed_form_and_exact_identification():
    R, F, _ = _null(seed=9)
    res = hj_distance(R, F)
    Rv, Fv = R.to_numpy(), F.to_numpy()
    T = len(Rv)
    W = np.linalg.inv(Rv.T @ Rv / T)
    D = Rv.T @ Fv / T
    b = np.linalg.solve(D.T @ W @ D, D.T @ W @ Rv.mean(0))
    g = Rv.mean(0) - D @ b
    np.testing.assert_allclose(res.distance, np.sqrt(g @ W @ g), rtol=0, atol=1e-12)
    np.testing.assert_allclose(res.coefs, b, atol=1e-10)
    np.testing.assert_allclose(res.pricing_errors, g, atol=1e-12)
    assert res.weights.shape == (4,) and np.all(res.weights > 0)
    one = hj_distance(R.iloc[:, :3], F)
    assert one.weights.shape == (1,)
    with pytest.raises(ValueError, match="N > K"):
        hj_distance(R.iloc[:, :2], F)


def test_hj_pvalue_reproducible_with_seed():
    R, F, _ = _null(seed=10)
    a = hj_distance(R, F, n_sim=20_000, seed=5)
    b = hj_distance(R, F, n_sim=20_000, seed=5)
    assert a.pvalue == b.pvalue and a.seed == 5 and a.n_sim == 20_000
    c = hj_distance(R, F, n_sim=20_000, seed=6)
    assert abs(c.pvalue - a.pvalue) < 0.02


def test_hj_size():
    rej = 0
    for s in range(300):
        R, F, _ = _null(seed=2000 + s, T=600)
        rej += hj_distance(R, F, hac_lags=0, n_sim=20_000, seed=s).pvalue < 0.05
    assert 0.02 <= rej / 300 <= 0.10, rej / 300


def test_report_quintet_and_plot():
    R, F, _ = _null(T=200, N=5)
    for res, cls in ((sdf_gmm(R, F), SDFResult), (hj_distance(R, F, n_sim=2_000), HJDistanceResult)):
        assert isinstance(res, cls)
        df = res.to_dataframe()
        assert list(df["name"]) == ["f0", "f1"]
        assert res.to_frame().equals(df)
        assert "|" in res.to_markdown()
        assert "\\begin{tabular}" in res.to_latex()
        assert "#table(" in res.to_typst()
        assert isinstance(res.summary(), str)
        assert isinstance(res.plot(), matplotlib.figure.Figure)
        _, ax = plt.subplots()
        assert res.plot(ax=ax) is ax
    assert "J" in sdf_gmm(R, F).summary()
    assert "HJ distance" in hj_distance(R, F, n_sim=2_000).summary()
    with pytest.raises(ValueError, match="stage"):
        sdf_gmm(R, F, stage="cue")


@pytest.mark.parametrize('demeaned', [True, False])
def test_sdf_rejects_unidentified_factor_loadings(demeaned):
    R, F, _ = _null()
    F['f1'] = .1 * F['f0']
    with pytest.raises(ValueError, match='not identified|rank deficient'):
        sdf_gmm(R, F, demeaned=demeaned)


def test_hj_rejects_redundant_test_assets():
    R, F, _ = _null()
    R['p5'] = .1 * R['p0']
    with pytest.raises(ValueError, match='test assets.*rank|rank.*test assets'):
        hj_distance(R, F, n_sim=100)


def test_hj_rejects_unidentified_factor_loadings():
    R, F, _ = _null()
    F['f1'] = .1 * F['f0']
    with pytest.raises(ValueError, match='not identified|rank deficient'):
        hj_distance(R, F, n_sim=100)
