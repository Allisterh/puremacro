"""Tests for the public generic GMM engine :func:`puremacro.inference.gmm`.

Oracles: ``linearmodels.iv.IV2SLS`` / ``IVGMM`` (test-only) and the
package's own :func:`puremacro.inference.hansen_j`. Tolerances are 1e-8
where the definitions coincide.
"""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.inference.gmm import gmm


# ---------------------------------------------------------------------------
# Shared fixture: linear IV with a heteroskedastic error
# ---------------------------------------------------------------------------

def _linear_iv(seed: int = 3, T: int = 500, ar: float = 0.0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((T, 3))
    v = rng.standard_normal(T)
    x = z @ np.array([0.8, 0.5, 0.3]) + v
    e = rng.standard_normal(T) * (1.0 + np.abs(z[:, 0]))
    if ar:
        for t in range(1, T):
            e[t] = ar * e[t - 1] + e[t]
    u = 0.5 * v + e
    y = 1.0 + 2.0 * x + u
    Z = np.column_stack([np.ones(T), z])
    X = np.column_stack([np.ones(T), x])
    return y, X, Z, x, z


def _iv_moments(theta, data):
    y, X, Z = data
    return Z * (y - X @ theta)[:, None]


def _iv_jacobian(theta, data):
    y, X, Z = data
    return -(Z.T @ X) / len(y)


def _lm_frames(y, x, z):
    import pandas as pd

    T = len(y)
    dep = pd.Series(y, name="y")
    exog = pd.DataFrame({"const": np.ones(T)})
    endog = pd.DataFrame({"x": x})
    instr = pd.DataFrame(z, columns=["z1", "z2", "z3"])
    return dep, exog, endog, instr


# ---------------------------------------------------------------------------
# Linear IV parity
# ---------------------------------------------------------------------------

def test_one_step_with_zz_weight_equals_2sls():
    lm_iv = pytest.importorskip("linearmodels.iv", minversion=None)
    pytest.importorskip("linearmodels", minversion="6.0")
    y, X, Z, x, z = _linear_iv()
    T = len(y)
    W = np.linalg.inv(Z.T @ Z / T)
    res = gmm(_iv_moments, np.zeros(2), (y, X, Z), weighting=W, hac_lags=0)
    ref = lm_iv.IV2SLS(*_lm_frames(y, x, z)).fit(cov_type="robust", debiased=False)
    np.testing.assert_allclose(res.coefs, ref.params.to_numpy(), atol=1e-8)
    np.testing.assert_allclose(res.se, ref.std_errors.to_numpy(), atol=1e-8)
    assert res.weighting == "array"
    assert np.isnan(res.j_stat) and np.isnan(res.j_pvalue)


def test_two_step_equals_ivgmm_and_hansen_j():
    lm_iv = pytest.importorskip("linearmodels.iv", minversion=None)
    pytest.importorskip("linearmodels", minversion="6.0")
    from puremacro.inference import hansen_j

    y, X, Z, x, z = _linear_iv()
    T = len(y)
    w0 = np.linalg.inv(Z.T @ Z / T)
    res = gmm(_iv_moments, np.zeros(2), (y, X, Z), weighting="two_step", w0=w0,
              hac_lags=0)
    ref = lm_iv.IVGMM(*_lm_frames(y, x, z), weight_type="robust").fit(
        cov_type="robust", debiased=False)
    np.testing.assert_allclose(res.coefs, ref.params.to_numpy(), atol=1e-8)
    np.testing.assert_allclose(res.se, ref.std_errors.to_numpy(), atol=1e-8)
    np.testing.assert_allclose(res.j_stat, ref.j_stat.stat, atol=1e-8)
    np.testing.assert_allclose(res.j_stat, hansen_j(y, x, z)["stat"], atol=1e-8)
    assert res.j_df == 2
    assert res.weighting == "two_step"
    assert 0.0 <= res.j_pvalue <= 1.0


def test_two_step_hac_equals_ivgmm_kernel():
    lm_iv = pytest.importorskip("linearmodels.iv", minversion=None)
    pytest.importorskip("linearmodels", minversion="6.0")
    y, X, Z, x, z = _linear_iv(ar=0.5)
    T = len(y)
    w0 = np.linalg.inv(Z.T @ Z / T)
    res = gmm(_iv_moments, np.zeros(2), (y, X, Z), weighting="two_step", w0=w0,
              hac_lags=3)
    # linearmodels' Bartlett weights are 1 - l/(bandwidth+1), l = 0..bandwidth,
    # so bandwidth=3 is the same kernel as hac_lags=3 here (no mapping).
    ref = lm_iv.IVGMM(*_lm_frames(y, x, z), weight_type="kernel",
                      kernel="bartlett", bandwidth=3).fit(
        cov_type="kernel", kernel="bartlett", bandwidth=3, debiased=False)
    np.testing.assert_allclose(res.coefs, ref.params.to_numpy(), atol=1e-8)
    np.testing.assert_allclose(res.se, ref.std_errors.to_numpy(), atol=1e-8)
    np.testing.assert_allclose(res.j_stat, ref.j_stat.stat, atol=1e-8)
    assert res.hac_lags == 3


def test_analytic_and_numeric_jacobian_agree():
    y, X, Z, _, _ = _linear_iv()
    a = gmm(_iv_moments, np.zeros(2), (y, X, Z), jacobian=_iv_jacobian, hac_lags=2)
    n = gmm(_iv_moments, np.zeros(2), (y, X, Z), hac_lags=2)
    np.testing.assert_allclose(a.coefs, n.coefs, atol=1e-10)
    np.testing.assert_allclose(a.se, n.se, atol=1e-6)


# ---------------------------------------------------------------------------
# Nonlinear
# ---------------------------------------------------------------------------

def _poisson_data(T: int = 20_000, seed: int = 5):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(T) * 0.5
    y = rng.poisson(np.exp(0.5 + 1.0 * x)).astype(float)
    X = np.column_stack([np.ones(T), x])
    Z = np.column_stack([np.ones(T), x, x ** 2])
    return y, X, Z


def _exp_moments(theta, data):
    y, X, Z = data
    return Z * (y - np.exp(X @ theta))[:, None]


def test_nonlinear_exponential_moment_recovers_truth():
    y, X, Z = _poisson_data()
    res = gmm(_exp_moments, np.zeros(2), (y, X, Z), names=("b0", "b1"))
    assert res.converged
    assert res.names == ("b0", "b1")
    assert np.all(np.abs(res.coefs - np.array([0.5, 1.0])) < 4 * res.se)
    assert res.j_df == 1
    assert res.j_pvalue > 0.001


def test_iterated_converges_and_reports_iterations():
    y, X, Z = _poisson_data(T=5_000)
    res = gmm(_exp_moments, np.zeros(2), (y, X, Z), weighting="iterated")
    assert res.converged
    assert res.weighting == "iterated"
    assert 2 <= res.iterations <= 50
    capped = gmm(_exp_moments, np.zeros(2), (y, X, Z), weighting="iterated",
                 max_iter=1, tol=1e-14)
    assert not capped.converged
    assert capped.iterations == 1


def test_identity_weighting_has_nan_j():
    y, X, Z, _, _ = _linear_iv()
    res = gmm(_iv_moments, np.zeros(2), (y, X, Z), weighting="identity")
    assert res.weighting == "identity"
    assert np.isnan(res.j_stat) and np.isnan(res.j_pvalue)
    np.testing.assert_allclose(res.weight_matrix, np.eye(4))
    assert "J not reported" in res.summary()


def test_just_identified_j_is_zero():
    y, X, Z, _, _ = _linear_iv()
    res = gmm(_iv_moments, np.zeros(2), (y, X, Z[:, :2]))
    assert res.j_df == 0
    assert res.j_stat == 0.0
    assert np.isnan(res.j_pvalue)
    # just identified: IV estimator (Z'X)^-1 Z'y
    beta = np.linalg.solve(Z[:, :2].T @ X, Z[:, :2].T @ y)
    np.testing.assert_allclose(res.coefs, beta, atol=1e-8)


def test_wald_matches_manual():
    y, X, Z, _, _ = _linear_iv()
    res = gmm(_iv_moments, np.zeros(2), (y, X, Z))
    w = res.wald(np.eye(2)[[1]], [2.0])
    np.testing.assert_allclose(w["stat"], ((res.coefs[1] - 2.0) / res.se[1]) ** 2,
                               atol=1e-10)
    assert w["df"] == 1
    assert 0.0 <= w["pvalue"] <= 1.0
    w0 = res.wald(np.eye(2))
    assert w0["df"] == 2


def test_rejects_bad_inputs():
    y, X, Z, _, _ = _linear_iv()
    data = (y, X, Z)
    with pytest.raises(ValueError, match="2-D"):
        gmm(lambda th, d: np.ones(10), np.zeros(2), data)
    with pytest.raises(ValueError, match="fewer moments"):
        gmm(_iv_moments, np.zeros(2), (y, X, Z[:, :1]))
    with pytest.raises(ValueError, match="non-finite"):
        gmm(lambda th, d: np.full((10, 3), np.nan), np.zeros(2), data)
    with pytest.raises(ValueError, match="weighting"):
        gmm(_iv_moments, np.zeros(2), data, weighting="cue")
    with pytest.raises(ValueError, match="hac_lags"):
        gmm(_iv_moments, np.zeros(2), data, hac_lags=-1)
    with pytest.raises(ValueError, match="hac_lags"):
        gmm(_iv_moments, np.zeros(2), data, hac_lags=2.5)
    with pytest.raises(ValueError, match="kernel"):
        gmm(_iv_moments, np.zeros(2), data, kernel="parzen")
    with pytest.raises(ValueError, match="weighting"):
        gmm(_iv_moments, np.zeros(2), data, weighting=np.eye(3))


def test_default_bandwidth_is_package_rule():
    from puremacro.regress.ols import _default_maxlags

    y, X, Z, _, _ = _linear_iv()
    res = gmm(_iv_moments, np.zeros(2), (y, X, Z))
    assert res.hac_lags == _default_maxlags(len(y))
    assert res.n_obs == len(y)
    assert res.moment_means.shape == (4,)
    assert res.weight_matrix.shape == (4, 4)


def test_report_quintet_and_plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y, X, Z, _, _ = _linear_iv()
    res = gmm(_iv_moments, np.zeros(2), (y, X, Z), names=("const", "x"))
    df = res.to_dataframe()
    assert list(df.columns) == ["name", "coef", "se", "z", "pvalue"]
    assert list(df["name"]) == ["const", "x"]
    assert res.to_frame().equals(df)
    assert "const" in res.to_markdown()
    assert "\\begin{tabular}" in res.to_latex()
    assert "#table(" in res.to_typst()
    assert "Hansen J" in res.summary()
    fig = res.plot()
    assert isinstance(fig, matplotlib.figure.Figure)
    fig2, ax = plt.subplots()
    assert res.plot(ax=ax) is ax
    plt.close("all")


def test_package_attribute_is_the_function():
    import puremacro.inference as inf
    from puremacro.inference.gmm import gmm as f

    assert inf.gmm is f
    assert "gmm" in inf.__all__ and "GMMFitResult" in inf.__all__
