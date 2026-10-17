"""Tests for the Growth-at-Risk module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, skew

from puremacro.gar import (
    qar,
    fit_skewt_to_quantiles, SkewTFit,
    skewt_pdf, skewt_cdf, skewt_ppf,
    fci, fci_rolling,
)


# ---------------------------------------------------------------------------
# QAR
# ---------------------------------------------------------------------------


def test_qar_documented_columns(rng):
    y = rng.standard_normal(300)
    out = qar(y, quantiles=(0.10, 0.50, 0.90), horizons=(1,),
              p=2, n_boot=20)
    assert {"h", "tau", "q_pred", "lo", "hi"}.issubset(out.columns)
    assert len(out) == 3   # 3 quantiles, 1 horizon


def test_qar_quantiles_increasing_in_tau(rng):
    """At a typical in-sample row the predicted quantile should be
    increasing in τ; the QR estimator returns ill-ordered quantiles
    only on small samples."""
    rng2 = np.random.default_rng(42)
    y = rng2.standard_normal(500)
    out = qar(y, quantiles=(0.10, 0.50, 0.90), horizons=(1,),
              p=2, n_boot=10)
    qs = out.set_index("tau")["q_pred"].sort_index()
    diffs = np.diff(qs.values)
    assert np.all(diffs > -0.5)   # monotone modulo small noise


# ---------------------------------------------------------------------------
# Skew-t
# ---------------------------------------------------------------------------


@pytest.mark.pyodide_smoke
def test_skewt_pdf_is_symmetric_t_when_alpha_zero():
    """At α = 0 the skew-t collapses to a Student-t."""
    from scipy.stats import t as student_t
    xs = np.linspace(-3, 3, 7)
    f_skewt = skewt_pdf(xs, mu=0.0, sigma=1.0, alpha=0.0, nu=5.0)
    f_t = student_t.pdf(xs, df=5.0)
    np.testing.assert_allclose(f_skewt, f_t, rtol=1e-10)


def test_skewt_cdf_monotone_and_bracketed():
    xs = np.linspace(-5, 5, 11)
    F = skewt_cdf(xs, mu=0.0, sigma=1.0, alpha=2.0, nu=6.0)
    assert (np.diff(F) >= -1e-10).all()
    assert F[0] < 0.05 and F[-1] > 0.95


def test_skewt_ppf_inverts_cdf():
    """ppf(cdf(x)) ≈ x at sane points."""
    for x in (-2.0, -0.5, 0.5, 2.0):
        F = float(skewt_cdf(np.array([x]), mu=0.0, sigma=1.0, alpha=1.5, nu=8.0)[0])
        x_back = skewt_ppf(F, mu=0.0, sigma=1.0, alpha=1.5, nu=8.0)
        assert abs(x_back - x) < 1e-2


def test_fit_skewt_recovers_symmetric_normal():
    """Feed quantiles of a standard Normal; the skew-t fit should
    converge to roughly (μ ≈ 0, σ ≈ 1, α ≈ 0, ν large)."""
    taus = (0.05, 0.25, 0.50, 0.75, 0.95)
    quantile_dict = {tau: float(norm.ppf(tau)) for tau in taus}
    fit = fit_skewt_to_quantiles(quantile_dict)
    assert isinstance(fit, SkewTFit)
    assert abs(fit.mu) < 0.2
    assert abs(fit.sigma - 1.0) < 0.2
    assert abs(fit.alpha) < 1.5         # near-zero skew
    # ν may go large; just check the skew-t respects it.
    assert fit.nu > 5.0


def test_fit_skewt_captures_left_skew():
    """Construct quantiles with an obvious left skew (lower tail
    further from the median than the upper). The fit should produce
    α < 0 (left-skewed Azzalini convention)."""
    quantile_dict = {
        0.05: -4.0,
        0.25: -1.2,
        0.50: 0.0,
        0.75: 0.7,
        0.95: 1.5,
    }
    fit = fit_skewt_to_quantiles(quantile_dict)
    assert fit.alpha < 0.0


def test_skewt_fit_downside_quantile_consistent():
    """``downside_quantile(0.05)`` should approximately match the
    ``Q_0.05`` we fed in."""
    quantile_dict = {
        0.05: -3.5, 0.25: -1.0, 0.50: 0.5, 0.75: 1.7, 0.95: 3.0,
    }
    fit = fit_skewt_to_quantiles(quantile_dict)
    assert abs(fit.downside_quantile(0.05) - quantile_dict[0.05]) < 0.5


def test_skewt_expected_shortfall_below_5pct():
    """ES(0.05) ≤ Q_0.05 by construction (it's the conditional mean
    over the lower 5% tail)."""
    quantile_dict = {
        0.05: -3.5, 0.25: -1.0, 0.50: 0.5, 0.75: 1.7, 0.95: 3.0,
    }
    fit = fit_skewt_to_quantiles(quantile_dict)
    es = fit.expected_shortfall(0.05)
    q5 = fit.downside_quantile(0.05)
    assert es < q5


def test_skewt_fit_is_frozen():
    """SkewTFit is a frozen dataclass: post-fit mutation must raise."""
    quantile_dict = {0.05: -3.0, 0.5: 0.0, 0.95: 3.0}
    fit = fit_skewt_to_quantiles(quantile_dict)
    assert isinstance(fit, SkewTFit)
    with pytest.raises(Exception):
        fit.mu = 99.0


# ---------------------------------------------------------------------------
# FCI
# ---------------------------------------------------------------------------


def test_fci_extracts_common_factor(rng):
    """Synthetic panel with one common factor + idiosyncratic noise.
    The fitted FCI should correlate strongly with the latent factor."""
    T, n = 240, 6
    f = rng.standard_normal(T)
    loadings = rng.standard_normal(n)
    X = f[:, None] * loadings + 0.3 * rng.standard_normal((T, n))
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(n)])
    out = fci(df)
    rho = float(np.corrcoef(out["index"].values, f)[0, 1])
    assert abs(rho) > 0.85
    assert 0.0 < out["var_explained"] < 1.0


def test_fci_handles_nan_drop(rng):
    T, n = 100, 4
    X = rng.standard_normal((T, n))
    X[:5, 0] = np.nan      # ragged front edge
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(n)])
    out = fci(df, handle_nan="drop")
    assert out["index"].shape == (T,)


def test_fci_rolling_returns_series(rng):
    T, n = 200, 4
    f = np.cumsum(rng.standard_normal(T) * 0.05)
    X = f[:, None] * rng.standard_normal(n) + 0.3 * rng.standard_normal((T, n))
    df = pd.DataFrame(X, index=pd.date_range("2010-01-01", periods=T, freq="MS"),
                       columns=[f"x{i}" for i in range(n)])
    series = fci_rolling(df, window=60)
    assert isinstance(series, pd.Series)
    assert series.notna().sum() > T - 80   # window=60 gives T-59 valid points


def test_fci_rejects_short_window():
    df = pd.DataFrame({"x": np.arange(50)})
    with pytest.raises(ValueError):
        fci_rolling(df, window=10)


def test_fci_rejects_unmatched_tightening_columns():
    """Names that match no column left the sign normalisation unanchored.

    The named-column rule was skipped (nothing matched) and so was the
    argmax default (it sat in the ``else`` arm), so the index carried the
    raw LAPACK SVD sign: on a 4-column panel 12/24 to 18/24 column
    orderings returned the inverted series.
    """
    rng = np.random.default_rng(0)
    f = rng.standard_normal(200)
    df = pd.DataFrame({
        "credit_spread": 0.95 * f + 0.4 * rng.standard_normal(200),
        "equity_vol":    0.75 * f + 0.4 * rng.standard_normal(200),
        "equity_price": -0.65 * f + 0.4 * rng.standard_normal(200),
    })
    with pytest.raises(ValueError, match="tightening_columns"):
        fci(df, tightening_columns=["CREDIT_SPREAD", "vix"])
    # a partial match still anchors, and is column-order invariant
    a = fci(df, tightening_columns=["credit_spread", "vix"])["index"].values
    b = fci(df[["equity_price", "equity_vol", "credit_spread"]],
            tightening_columns=["credit_spread", "vix"])["index"].values
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_fci_drop_scores_ragged_rows_without_attenuation():
    """Noiseless exact single-factor panel: every column is a rescaling of
    the same factor, so blanking cells destroys no information about
    ``f_t`` and the index must be unchanged. The old projection step
    zero-filled the missing cells (both arms of that if/else were the same
    line), attenuating each ragged row by ``||v_obs||**2 == (n-1)/n``."""
    for n, k in ((4, 20), (6, 30)):
        T = 200
        rng = np.random.default_rng(0)
        f = rng.standard_normal(T)
        cols = [f"x{j}" for j in range(n)]
        complete = pd.DataFrame(np.outer(f, np.linspace(0.5, 1.5, n)),
                                columns=cols)
        ragged = complete.copy()
        ragged.iloc[-k:, 0] = np.nan          # ragged right edge
        i_c = fci(complete)["index"].values
        i_r = fci(ragged)["index"].values
        assert np.abs(i_c - i_r).max() < 1e-10, (n, np.abs(i_c - i_r).max())


def test_fci_drop_loadings_are_correlation_pc1_of_fitted_rows():
    """Unbalanced panel whose missingness tracks a common volatility
    factor. ``drop`` fits the complete-cases rows, so the loadings must be
    the first eigenvector of *their* correlation matrix. Standardising
    with the full-sample ``nanstd`` instead scaled the fitted columns by
    sds computed on a different row set (0.38, 0.38, 0.38, 1.0), landing
    ~53 degrees away and inverting the loading order."""
    for T, seed in ((600, 0), (2000, 3)):
        rng = np.random.default_rng(seed)
        vol = np.where(rng.random(T) < 0.4, 4.0, 1.0)
        f = rng.standard_normal(T)
        e = rng.standard_normal((T, 4))
        X = vol[:, None] * (np.outer(f, [1.0, 0.8, 0.6, 0.5])
                            + e * [0.5, 0.7, 0.9, 1.1]) * [1.0, 30.0, 0.01, 5.0]
        df = pd.DataFrame(X, columns=list("abcd"))
        df.loc[vol > 1, "d"] = np.nan     # this series only exists when calm
        out = fci(df, handle_nan="drop")

        w, V = np.linalg.eigh(np.corrcoef(df.dropna().values, rowvar=False))
        L = out["loadings"].values
        cos = abs(L @ V[:, -1]) / np.linalg.norm(L) / np.linalg.norm(V[:, -1])
        assert cos > 1 - 1e-10, (T, seed, cos)
        assert abs(out["var_explained"] - w[-1] / w.sum()) < 1e-10


def test_fci_rolling_holds_one_orientation():
    """Constant loadings, one loose-when-high column. Each window used to
    re-draw the argmax sign anchor, so a single returned series mixed two
    opposite orientations and correlated with the full-sample FCI at
    rho = -0.06 to +0.60 instead of 0.99."""
    for T, seed in ((300, 1), (400, 0), (400, 4)):
        rng = np.random.default_rng(seed)
        f = rng.standard_normal(T)
        df = pd.DataFrame({
            "spread":  1.00 * f + 0.6 * rng.standard_normal(T),
            "vol":     0.55 * f + 0.6 * rng.standard_normal(T),
            "equity": -1.02 * f + 0.6 * rng.standard_normal(T),
        })
        roll = fci_rolling(df, window=60)
        full = fci(df)["index"]
        m = roll.notna().values
        rho = float(np.corrcoef(roll.values[m], full.values[m])[0, 1])
        assert rho > 0.9, (T, seed, rho)
