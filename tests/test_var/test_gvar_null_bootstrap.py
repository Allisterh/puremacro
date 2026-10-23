"""Restricted null construction, generated-regressor refits, and boundaries."""
import numpy as np
import pandas as pd
import pytest

from puremacro.var import GVARIndependentNullResult, gvar
from puremacro.var._gvar_null_bootstrap import _country_null


def fitted(rho=1., T=90, seed=51, **kwargs):
    rng = np.random.default_rng(seed)
    X = np.zeros((T, 6))
    for t in range(1, T):
        X[t] = rho * X[t - 1] + rng.standard_normal(6)
    countries = ["A", "B", "C"]
    frames = {c: pd.DataFrame(X[:, j * 2:j * 2 + 2], columns=["y", "r"])
              for j, c in enumerate(countries)}
    W = pd.DataFrame([[0, .6, .4], [.5, 0, .5], [.4, .6, 0]], index=countries, columns=countries)
    return gvar(frames, W, p=1, q=1, stability_warn=False, **kwargs)


def run(result, **kwargs):
    args = dict(null="independent_countries", integration="i1", n_boot=39,
                block_length=1, seed=731)
    args.update(kwargs)
    return result.independent_null_bootstrap(**args)


def test_reproducibility_schema_and_original_diagnostic_unchanged():
    result = fitted()
    before = result.weak_exogeneity.table.copy(deep=True)
    panel_before = result.panel.copy(deep=True)
    b = run(result)
    assert isinstance(b, GVARIndependentNullResult)
    np.testing.assert_array_equal(b.statistic_draws, run(result).statistic_draws)
    pd.testing.assert_frame_equal(result.weak_exogeneity.table, before)
    pd.testing.assert_frame_equal(result.panel, panel_before)
    assert b.statistic_draws.shape == (39, 6)
    p = (1 + np.sum(b.statistic_draws >= b.table.F.to_numpy(), axis=0)) / 40
    np.testing.assert_array_equal(b.table.p_independent_null, p)
    np.testing.assert_allclose(b.table.mc_se, np.sqrt(p * (1 - p) / 40))
    assert np.all(b.table.p_holm_independent_null >= p)
    assert not np.array_equal(p, before.p_value)
    assert "NOT the general weak-exogeneity" in b.summary()
    assert "reject_independent_null" in b.to_frame()
    assert b.metadata["first_stage_refitted"] is True
    assert b.metadata["draws_discarded"] == 0


def test_first_i1_draw_matches_independent_null_simulator_and_public_refit():
    result = fitted(T=70)
    b = run(result, n_boot=19, block_length=3)
    rng = np.random.default_rng(731)
    raw = result.panel.to_numpy()
    paths = {}
    for j, country in enumerate(result.countries):
        levels = raw[:, j * 2:j * 2 + 2]
        dy = np.diff(levels, axis=0)
        Z = np.column_stack([np.ones(len(dy) - 1), dy[:-1]])
        B = np.linalg.lstsq(Z, dy[1:], rcond=None)[0]
        residuals = dy[1:] - Z @ B
        residuals -= residuals.mean(axis=0)
        starts = rng.integers(len(residuals), size=(len(dy) - 1 + 2) // 3)
        indices = ((starts[:, None] + np.arange(3)) % len(residuals)).ravel()[:len(dy) - 1]
        simulated_dy = np.empty_like(dy)
        simulated_dy[0] = dy[0]
        for t in range(1, len(dy)):
            simulated_dy[t] = np.r_[1., simulated_dy[t - 1]] @ B + residuals[indices[t - 1]]
        simulated = np.vstack([levels[0], levels[0] + np.cumsum(simulated_dy, axis=0)])
        paths[country] = pd.DataFrame(simulated, columns=["y", "r"])
    reference = gvar(paths, result.weights, p=1, q=1, stability_warn=False)
    np.testing.assert_allclose(b.statistic_draws[0], reference.weak_exogeneity.table.F, rtol=1e-10, atol=1e-11)
    # A frozen generated-regressor construction would retain the observed
    # first-stage residuals and disagree with this independent refit.
    assert np.max(np.abs(reference.resid.to_numpy() - result.resid.to_numpy())) > 1


def test_stationary_reference_is_explicitly_different_from_i1_reference():
    result = fitted(rho=.5)
    stationary = run(result, integration="stationary", n_boot=19)
    integrated = run(result, integration="i1", n_boot=19)
    assert stationary.metadata["integration"] == "stationary"
    assert np.max(np.abs(stationary.statistic_draws - integrated.statistic_draws)) > .5
    assert all(root < 1 for root in stationary.metadata["country_null_max_roots"].values())


@pytest.mark.parametrize("kwargs,match", [
    ({"null": "weak_exogeneity"}, "only null"),
    ({"integration": "auto"}, "explicitly"),
    ({"n_boot": 18}, "n_boot"),
    ({"n_boot": True}, "n_boot"),
    ({"block_length": 0}, "block_length"),
    ({"block_length": 45}, "two innovation"),
    ({"null_lags": 0}, "null_lags"),
    ({"seed": -1}, "seed"),
    ({"alpha": np.inf}, "alpha"),
])
def test_unsupported_nulls_and_invalid_settings_fail(kwargs, match):
    with pytest.raises(ValueError, match=match):
        run(fitted(), **kwargs)


def test_missing_diagnostic_exog_and_trends_are_not_silently_ignored():
    with pytest.raises(ValueError, match="weak_exogeneity=True"):
        run(fitted(weak_exogeneity=False))
    with pytest.raises(ValueError, match="no exog and trend"):
        run(fitted(trend="ct"))
    exog = pd.DataFrame({"z": np.random.default_rng(4).normal(size=90)})
    with pytest.raises(ValueError, match="no exog and trend"):
        run(fitted(exog=exog))


def test_nonstationary_null_fit_is_rejected_without_root_clipping():
    rng = np.random.default_rng(9)
    path = np.zeros((80, 1))
    for t in range(1, len(path)):
        path[t] = 1.1 * path[t - 1] + rng.normal()
    with pytest.raises(ValueError, match="not stationary; no roots clipped"):
        _country_null(path, 1, True)


def test_statistic_is_recomputed_from_data_not_mutable_diagnostic_table():
    result = fitted()
    expected = run(result, n_boot=19)
    result.weak_exogeneity.table.loc[:, "F"] = 0
    result.weak_exogeneity.table.loc[:, "p_value"] = 1
    actual = run(result, n_boot=19)
    np.testing.assert_array_equal(actual.table.F, expected.table.F)


def test_nonestimable_generated_residual_restrictions_rejected_only_by_new_reference():
    # With two levels lags, the fitted VARX*(1,1) residual at t-1 is
    # algebraically in the span of those regressors. The legacy diagnostic
    # still exists, but it cannot support a bootstrap inference reference.
    result = fitted(rho=.5, we_lags=2, we_transform="levels")
    assert len(result.weak_exogeneity.table) == 6
    with pytest.raises(ValueError, match="unrestricted auxiliary design is rank deficient"):
        run(result, integration="stationary", n_boot=19)
    # Full-rank first-difference design remains supported with the same lags.
    valid = run(fitted(rho=.5, we_lags=2), integration="stationary", n_boot=19)
    assert np.isfinite(valid.statistic_draws).all()
