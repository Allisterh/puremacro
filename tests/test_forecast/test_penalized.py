"""Tests for High-Dimensional Penalized Macro Forecasting."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.forecast.penalized import (
    PenalizedForecastResult,
    forecast_penalized,
)


@pytest.fixture
def synthetic_sparse_macro():
    rng = np.random.default_rng(123)
    T = 120
    P = 20
    X = rng.normal(size=(T, P))
    
    # Predictive DGP for horizon h=1: y_{t+1} depends on X_t[:, 0] and X_t[:, 2]
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 1.5 + 2.0 * X[t-1, 0] - 1.5 * X[t-1, 2] + rng.normal(scale=0.3)
    y[0] = 1.5 + rng.normal(scale=0.3)
    
    df_X = pd.DataFrame(X, columns=[f"Pred_{i+1}" for i in range(P)])
    s_y = pd.Series(y, name="Inflation")
    return df_X, s_y


def test_forecast_penalized_basic(synthetic_sparse_macro):
    df_X, s_y = synthetic_sparse_macro
    res = forecast_penalized(df_X, s_y, horizon=1, alpha=0.5, adaptive=True)

    assert isinstance(res, PenalizedForecastResult)
    assert isinstance(res.forecast, float)
    assert not np.isnan(res.forecast)
    assert len(res.coefficients) == 20
    assert 0.0 <= res.in_sample_r2 <= 1.0
    assert res.horizon == 1
    
    # Check that true sparse features are recovered
    assert "Pred_1" in res.selected_features
    assert "Pred_3" in res.selected_features
    # Check sparsity (less than half features selected)
    assert len(res.selected_features) <= 10

    s = res.summary()
    assert "Penalized Macro Forecasting" in s
    assert "Selected Predictors" in s
    assert "Optimal Penalty" in s


# ---------------------------------------------------------------------------
# Regression tests (v2.3.x audit fixes)
# ---------------------------------------------------------------------------


def _ar_panel(T=160, P=30, seed=123):
    rng = np.random.default_rng(seed)
    X = np.zeros((T, P))
    for j in range(P):
        rho = rng.uniform(0.3, 0.8)
        for t in range(1, T):
            X[t, j] = rho * X[t - 1, j] + rng.normal(scale=0.8)
    y = np.zeros(T)
    active, w = [1, 5, 12, 22], [1.8, -1.4, 1.2, -0.9]
    active, w = zip(*[(i, wi) for i, wi in zip(active, w) if i < P])
    for t in range(1, T):
        y[t] = 2.0 + sum(wi * X[t - 1, i] for wi, i in zip(w, active)) + rng.normal(scale=0.5)
    y[0] = 2.0
    return (pd.DataFrame(X, columns=[f"M{j + 1:02d}" for j in range(P)]), pd.Series(y))


def test_alpha_zero_is_a_working_ridge():
    """Regression: ``alpha=0.0`` ('Ridge' per the docstring) built a lambda
    grid 1e4 too high (the ``1e-4`` floor in lambda_max) and, with df
    counted as non-zeros, returned the sample mean (in-sample R² 0.0002)
    or pinned to lambda_min. It now solves the ridge path in closed form
    on an eigenvalue-anchored grid with the hat-matrix trace as df."""
    X, y = _ar_panel()
    Xn = X.iloc[:-1].to_numpy(); yn = y.iloc[1:].to_numpy()
    Xc = np.column_stack([np.ones(len(Xn)), Xn])
    b_ols = np.linalg.lstsq(Xc, yn, rcond=None)[0]
    r2_ols = 1 - ((yn - Xc @ b_ols) ** 2).sum() / ((yn - yn.mean()) ** 2).sum()
    Xs = (Xn - Xn.mean(0)) / Xn.std(0)
    e_max_plain = np.linalg.svd(Xs, compute_uv=False).max() ** 2 / len(Xs)
    for adaptive in (False, True):
        res = forecast_penalized(X, y, horizon=1, alpha=0.0, adaptive=adaptive)
        assert res.in_sample_r2 > r2_ols - 0.02, (adaptive, res.in_sample_r2, r2_ols)
        assert len(res.selected_features) == X.shape[1]          # ridge keeps everything
        lam_hi, lam_lo = res.bic_path.index[0], res.bic_path.index[-1]
        assert np.isclose(lam_hi / lam_lo, 1e3)                    # lambda_min_ratio honoured
        if not adaptive:                                           # grid anchored on the spectrum
            assert np.isclose(lam_hi, 10.0 * e_max_plain) and lam_lo < 0.05
        assert np.isfinite(res.forecast)
    # the adaptive ridge lands at an interior BIC optimum on this panel
    res_ad = forecast_penalized(X, y, horizon=1, alpha=0.0, adaptive=True)
    assert res_ad.optimal_lambda not in (res_ad.bic_path.index[0], res_ad.bic_path.index[-1])


def test_ridge_closed_form_matches_coordinate_descent():
    from puremacro.forecast.penalized import _fit_coordinate_descent, _ridge_path
    X, y = _ar_panel(T=100, P=8)
    Xn = X.to_numpy(); Xs = (Xn - Xn.mean(0)) / Xn.std(0); yv = y.to_numpy()
    w = np.linspace(0.5, 2.0, 8)
    for lam in (0.05, 0.5, 3.0):
        b0, b_cd = _fit_coordinate_descent(Xs, yv, lam, 0.0, w, max_iter=20000, tol=1e-12)
        (b0_r, b_r, df), = _ridge_path(Xs, yv, np.array([lam]), w)
        np.testing.assert_allclose(b_r, b_cd, atol=1e-7)
        assert np.isclose(b0_r, b0)
        assert 0.0 < df < 8.0


def test_alpha_outside_unit_interval_raises():
    """``alpha`` used to be unvalidated (``alpha=2.0`` ran)."""
    X, y = _ar_panel(T=60, P=5)
    for bad in (-0.1, 1.5, 2.0):
        with pytest.raises(ValueError, match="alpha must be in"):
            forecast_penalized(X, y, horizon=1, alpha=bad)


def test_length_mismatch_raises_clear_error():
    X, y = _ar_panel(T=60, P=5)
    with pytest.raises(ValueError, match="alignment is positional"):
        forecast_penalized(X, y.iloc[:-5], horizon=1)


def test_penalized_result_presentation_contract():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt

    X, y = _ar_panel()
    res = forecast_penalized(X, y, horizon=1, alpha=1.0, adaptive=True)
    md = res.to_markdown()
    lines = [l for l in md.splitlines() if "|" in l]
    assert len(lines) == 2 + X.shape[1] and "---" in lines[1] and "selected" in lines[0]
    assert "tabular" in res.to_latex() and "#table(" in res.to_typst()
    frame = res.to_frame()
    assert frame["selected"].sum() == len(res.selected_features)
    assert isinstance(res.plot(), Figure)
    plt.close("all")
    assert "% of candidates" in res.summary() and "sparsity" not in res.summary()
    # nothing selected -> the BIC path is plotted instead of empty bars
    rng = np.random.default_rng(0)
    noise = forecast_penalized(rng.normal(size=(80, 5)), rng.normal(size=80), alpha=1.0, adaptive=False)
    assert isinstance(noise.plot(), Figure)
    plt.close("all")


@pytest.fixture
def scale_dgp():
    """y_{t+1} = 1.5 + 1.2*x0 - 0.8*x4 + N(0, 0.5^2), X standardised to unit
    column variance, so the population coefficient on P1 is exactly 1.2."""
    rng = np.random.default_rng(0)
    T, P = 600, 12
    F = rng.normal(size=(T, 3))
    L = rng.normal(size=(3, P))
    X = F @ L + rng.normal(size=(T, P))
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = np.empty(T)
    y[0] = 1.5 + rng.normal(scale=0.5)
    for t in range(1, T):
        y[t] = 1.5 + 1.2 * X[t - 1, 0] - 0.8 * X[t - 1, 4] + rng.normal(scale=0.5)
    df_X = pd.DataFrame(X, columns=[f"P{j+1}" for j in range(P)])
    return df_X, pd.Series(y)


def test_forecast_penalized_raises_on_non_finite_input(synthetic_sparse_macro):
    """A NaN cell used to sail through: coefficients, intercept and forecast all
    came back NaN while `in_sample_r2` was reported as exactly 1.0, because
    `max(0.0, min(1.0, nan))` is a comparison fold that returns 1.0."""
    df_X, s_y = synthetic_sparse_macro

    bad_X = df_X.copy()
    bad_X.iloc[60, 3] = np.nan
    bad_X.iloc[61, 7] = np.inf
    with pytest.raises(ValueError, match=r"2 non-finite value"):
        forecast_penalized(bad_X, s_y, horizon=1)

    bad_y = s_y.copy()
    bad_y.iloc[60] = np.nan
    with pytest.raises(ValueError, match=r"non-finite value"):
        forecast_penalized(df_X, bad_y, horizon=1)

    # The clean panel still fits, and its R2 is nowhere near the manufactured 1.0.
    res = forecast_penalized(df_X, s_y, horizon=1)
    assert 0.0 < res.in_sample_r2 < 1.0


def test_forecast_penalized_is_equivariant_in_target_units(scale_dgp):
    """coef(c*y) = c*coef(y) holds for the linear projection by construction, so
    rescaling the target -- percent to basis points -- must not change anything.
    It did: with the target fitted in its own units the single penalty `lam`
    multiplied an L1 term homogeneous of degree 1 and an L2 term homogeneous of
    degree 2, and coef("P1")/c fell from 1.195 at c=1 to 0.544 at c=1000
    against a population truth of 1.2."""
    df_X, s_y = scale_dgp
    base = forecast_penalized(df_X, s_y, horizon=1)

    for c in (1000.0, 7.0, 0.01):
        scaled = forecast_penalized(df_X, c * s_y, horizon=1)
        np.testing.assert_allclose(
            scaled.coefficients.to_numpy() / c, base.coefficients.to_numpy(), atol=1e-8
        )
        assert scaled.forecast / c == pytest.approx(base.forecast, abs=1e-8)
        assert scaled.intercept / c == pytest.approx(base.intercept, abs=1e-8)
        assert scaled.in_sample_r2 == pytest.approx(base.in_sample_r2, abs=1e-10)
        assert scaled.selected_features == base.selected_features

    # Equivariant about the right value: the truth on P1 is 1.2.
    assert base.coefficients["P1"] == pytest.approx(1.2, abs=0.1)
    assert forecast_penalized(df_X, 1000.0 * s_y, horizon=1).coefficients["P1"] / 1000.0 == pytest.approx(1.2, abs=0.1)


def test_forecast_penalized_ridge_alpha_zero_is_consistent():
    """alpha=0.0 is documented as Ridge. No finite penalty zeroes a ridge
    solution, so the L1 grid anchor `|x'y| / (T * alpha * w)` does not exist
    there; the old `np.maximum(alpha * w, 1e-4)` clamp put the whole grid 1e4
    above the natural scale and the estimator converged to beta/(1+lam) ~ 0.097
    instead of to the true 3.0, at every sample size."""
    beta_true = np.zeros(15)
    beta_true[0], beta_true[4], beta_true[9] = 3.0, -2.0, 1.0

    prev_err = np.inf
    for T in (400, 4000):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(T, 15))
        y = np.empty(T)
        y[0] = 0.7 + rng.normal(scale=0.5)
        for t in range(1, T):
            y[t] = 0.7 + X[t - 1] @ beta_true + rng.normal(scale=0.5)
        df_X = pd.DataFrame(X, columns=[f"x{j+1}" for j in range(15)])

        res = forecast_penalized(df_X, pd.Series(y), horizon=1, alpha=0.0, adaptive=False)
        err = float(np.max(np.abs(res.coefficients.to_numpy() - beta_true)))
        # Ridge over a grid that reaches the weakly-penalised region is
        # consistent, so the error has to shrink with T towards zero.
        assert err < 0.25, f"T={T}: max coefficient error {err:.4f}"
        assert err < prev_err
        prev_err = err
        assert res.coefficients.iloc[0] == pytest.approx(3.0, rel=0.05)
        assert res.in_sample_r2 > 0.9

