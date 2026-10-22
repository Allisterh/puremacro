"""Independent scalar recurrences guard lag ordering in batched simulations."""
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from puremacro.inference.bootstrap import residual_bootstrap
from puremacro.var.bvar_sv import BVAR_SVResult


def _recur(initial, coefficients, intercept, innovations):
    draws, steps, n = innovations.shape
    p = len(coefficients)
    paths = np.empty((draws, p + steps, n))
    paths[:, :p] = initial
    for b in range(draws):
        for t in range(p, p + steps):
            paths[b, t] = intercept + innovations[b, t - p]
            for lag, coef in enumerate(coefficients, 1):
                paths[b, t] += coef @ paths[b, t - lag]
    return paths


@pytest.mark.parametrize('p', [1, 3])
@pytest.mark.parametrize('layout', ['fortran', 'strided'])
def test_residual_paths_match_scalar_recurrence(p, layout):
    rng = np.random.default_rng(219)
    raw = rng.normal(size=(60, 6))
    data = np.asfortranarray(raw[:, :3]) if layout == 'fortran' else raw[:, ::2]
    coefficients = [rng.normal(scale=.06, size=(3, 3)) for _ in range(p)]
    intercept = np.array([.1, -.2, .3])
    residuals = rng.normal(size=(len(data) - p, 3))
    expected_rng = np.random.default_rng(87)
    innovations = residuals[expected_rng.integers(0, len(residuals), (7, len(residuals)))]
    expected = _recur(data[:p], coefficients, intercept, innovations)
    actual_rng = np.random.default_rng(87)
    result = residual_bootstrap(residuals, data, coefficients, intercept,
                                n_draws=7, irf_fn=lambda y, p, h: y.copy(), rng=actual_rng)
    np.testing.assert_allclose(result['draws'], expected, rtol=1e-14, atol=1e-14)
    assert actual_rng.bit_generator.state == expected_rng.bit_generator.state


@pytest.mark.parametrize('p', [1, 3])
@pytest.mark.parametrize('kind', ['wild', 'recursive', 'pilot'])
def test_bootstrap_refits_scalar_paths(monkeypatch, p, kind):
    data = np.random.default_rng(349).normal(size=(60, 3))
    captured = []
    module = importlib.import_module('puremacro.inference.wild_bootstrap' if kind == 'wild'
                                     else 'puremacro.var.bootstrap')
    fit_name = '_ols_var' if kind == 'wild' else 'estimate_var'
    fit = getattr(module, fit_name)
    coeffs, intercept, _, residuals, _ = fit(data, p)

    def capture(y, lag):
        captured.append(y.copy())
        return fit(y, lag)

    monkeypatch.setattr(module, fit_name, capture)
    expected_rng = np.random.default_rng(49)
    if kind == 'wild':
        innovations = residuals[None] * expected_rng.choice([-1., 1.], (7, len(residuals)))[:, :, None]
        module.wild_bootstrap_var(data, p=p, horizon=4, n_boot=7, seed=49,
                                  impact_fn=lambda a, s, r: np.linalg.cholesky(s))
    else:
        innovations = residuals[expected_rng.integers(0, len(residuals), (7, len(residuals)))]
        actual_rng = np.random.default_rng(49)
        if kind == 'pilot':
            module._kilian_bias_correct(data, p, n_pilot=7, rng=actual_rng)
        else:
            module.bootstrap_bands(data, p, lambda a, s: np.linalg.cholesky(s), 4,
                                   n_boot=7, rng=actual_rng)
        assert actual_rng.bit_generator.state == expected_rng.bit_generator.state
    expected = _recur(data[:p], coeffs, intercept, innovations)
    # The first fit is the original sample; every subsequent fit sees a draw.
    np.testing.assert_allclose(captured[1:], expected, rtol=1e-13, atol=1e-13)


def make_forecast_result(p=3, draws=8, n=3):
    rng = np.random.default_rng(841)
    return BVAR_SVResult(
        beta_draws=rng.normal(scale=.04, size=(draws, 1 + p * n, n)),
        h_draws=rng.normal(scale=.1, size=(draws, 15, n)),
        a_draws=np.repeat((np.eye(n) + .03 * np.tril(np.ones((n, n)), -1))[None], draws, axis=0),
        r_hat={}, data=pd.DataFrame(rng.normal(size=(15 + p, n))), lags=p,
        mu_draws=np.full((draws, n), -.2), phi_draws=np.full((draws, n), .4),
        sigma_h_draws=np.full((draws, n), .1), log_scores=np.zeros(15),
        n_draws=draws, n_burn=0,
    )


@pytest.mark.parametrize('p', [1, 3])
def test_forecast_scalar_lags_and_no_mutation(p):
    result = make_forecast_result(p)
    before = {k: v.copy() for k, v in vars(result).items() if isinstance(v, (np.ndarray, pd.DataFrame))}
    rng = np.random.default_rng(83)
    draws, n, horizon = result.n_total_draws, result.n, 9
    history = np.repeat(result.data.to_numpy()[None], draws, axis=0)
    h_prev = result.h_draws[:, -1].copy()
    expected, expected_h = [], []
    for _ in range(horizon):
        h = result.mu_draws + result.phi_draws * (h_prev - result.mu_draws) + result.sigma_h_draws * rng.standard_normal((draws, n))
        nu = np.exp(h / 2) * rng.standard_normal((draws, n))
        y = np.empty((draws, n))
        for b in range(draws):
            y[b] = result.beta_draws[b, 0] + np.linalg.solve(result.a_draws[b], nu[b])
            for lag in range(1, p + 1):
                y[b] += history[b, -lag] @ result.beta_draws[b, 1 + (lag - 1) * n:1 + lag * n]
        expected.append(y)
        expected_h.append(h)
        history = np.concatenate([history, y[:, None]], axis=1)
        h_prev = h
    actual = result.forecast(horizon=horizon, seed=83)
    np.testing.assert_allclose(actual.paths, np.stack(expected, axis=1), rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(actual.h_paths, np.stack(expected_h, axis=1))
    for key, value in before.items():
        if isinstance(value, pd.DataFrame):
            pd.testing.assert_frame_equal(getattr(result, key), value)
        else:
            np.testing.assert_array_equal(getattr(result, key), value)
    np.testing.assert_array_equal(result.forecast(horizon=horizon, seed=83).paths, actual.paths)
