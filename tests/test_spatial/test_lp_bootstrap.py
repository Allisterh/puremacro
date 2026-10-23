"""Independent OLS references, block alignment, and honest bootstrap scope."""
import numpy as np
import pandas as pd
import pytest

from puremacro.spatial import SpatialLPBootstrap, SpatialLPBootstrapResult, knn_weights, spatial_lp
from puremacro.spatial._lp_bootstrap import _balanced_fit, _circular_indices


def panel(seed=42, T=38):
    rng = np.random.default_rng(seed)
    ids = [f"r{i}" for i in range(8)]
    coords = pd.DataFrame({"lat": rng.uniform(30, 45, 8), "lon": rng.uniform(-100, -70, 8)}, index=ids)
    W = knn_weights(coords, 2)
    x = rng.standard_normal((8, T))
    dy = .6 * x + .3 * W.to_dense() @ x + .3 * rng.standard_normal(x.shape)
    frame = pd.DataFrame({"x": x.ravel(), "y": np.cumsum(dy, axis=1).ravel()},
                         index=pd.MultiIndex.from_product([ids, range(T)], names=["code", "date"]))
    return frame, W


def fit(config=None, **kwargs):
    data, W = panel()
    return spatial_lp(data, "y", "x", W, horizons=range(3), n_lags=1,
                      lagged_outcome="difference", bootstrap=config, **kwargs)


def test_balanced_engine_matches_independent_dummy_ols():
    rng = np.random.default_rng(24)
    values = rng.normal(size=(11, 5, 4))
    values[:, :, 0] += np.arange(5)[None, :] + np.arange(11)[:, None]
    result = _balanced_fit(values, np.arange(5))
    # Explicit dummy-variable regression, independent of any within helper.
    y = values[:, :, 0].ravel()
    X = values[:, :, 1:].reshape(55, 3)
    D = np.column_stack([np.ones(55), np.tile(np.eye(5)[:, 1:], (11, 1)),
                         np.repeat(np.eye(11)[:, 1:], 5, axis=0)])
    Z = np.column_stack([X, D])
    beta = np.linalg.lstsq(Z, y, rcond=None)[0]
    np.testing.assert_allclose(result["beta"], beta[:3], atol=1e-13)
    np.testing.assert_allclose(result["residuals"], y - Z @ beta, atol=2e-13)


def test_circular_blocks_keep_runs_and_wrap():
    sample = _circular_indices(np.random.default_rng(1), 13, 4)
    assert len(sample) == 13
    for block in sample[:12].reshape(3, 4):
        np.testing.assert_array_equal(np.diff(block) % 13, 1)
    assert len(np.unique(sample)) < 13  # fixture actually repeats observations


def test_common_support_bands_draws_and_seed_leave_original_fit_unchanged():
    baseline = fit()
    config = SpatialLPBootstrap(39, 4, True, 91)
    result = fit(config)
    repeat = fit(config).bootstrap_result
    b = result.bootstrap_result
    assert isinstance(b, SpatialLPBootstrapResult)
    np.testing.assert_array_equal(result.coef, baseline.coef)
    np.testing.assert_array_equal(result.vcov, baseline.vcov)
    np.testing.assert_array_equal(b.coefficient_draws, repeat.coefficient_draws)
    assert b.metadata["n_periods"] == 34  # Δy lag consumes 2, lead consumes 2
    assert b.metadata["point_sample"].startswith("intersection")
    assert b.coefficient_draws.shape == (39, 3, 2)
    assert b.statistic_draws.shape == (39, 3, 3)
    # The earliest horizon's common-support estimate differs from the legacy
    # varying-horizon one: this fixture would catch hiding the sample change.
    assert abs(b.table.loc[(0, "direct"), "point"] - result.coef[0, 0]) > 1e-5
    assert np.all(b.table.simultaneous_lo <= b.table.lo)
    assert np.all(b.table.simultaneous_hi >= b.table.hi)
    assert b.joint_p_value >= 1 / 40
    observed = b.joint_statistic
    draw_max = np.abs(b.statistic_draws[:, :, 1]).max(axis=1)
    assert b.joint_p_value == (1 + np.count_nonzero(draw_max >= observed)) / 40
    assert b.metadata["joint_labels"] == ("indirect",)
    assert "common-support" in b.summary()
    assert "p_value" in b.to_frame()


def test_difference_control_matches_manual_regression():
    data, W = panel()
    result = spatial_lp(data, "y", "x", W, horizons=[0], n_lags=1,
                        lagged_outcome="difference", cumulative=False)
    x = data.x.to_numpy().reshape(8, -1).T
    y = data.y.to_numpy().reshape(8, -1).T
    wx = x @ W.to_dense().T
    # h=0, Δy[t-1], x[t-1], Wx[t-1]; direct and indirect first.
    values = np.stack([y[2:] - y[1:-1], x[2:], wx[2:], x[1:-1],
                       y[1:-1] - y[:-2], wx[1:-1]], axis=-1)
    explicit = _balanced_fit(values, np.arange(8))
    np.testing.assert_allclose(result.coef[0], explicit["beta"][:2], atol=2e-13)
    assert result.spec["lagged_outcome"] == "difference"


@pytest.mark.parametrize("config,match", [
    (SpatialLPBootstrap(19, 3), "stationary"),
    (SpatialLPBootstrap(18, 3, True), "n_boot"),
    (SpatialLPBootstrap(True, 3, True), "n_boot"),
    (SpatialLPBootstrap(19, 0, True), "block_length"),
    (SpatialLPBootstrap(19, 2.2, True), "block_length"),
    (SpatialLPBootstrap(19, 3, True, -1), "seed"),
    (SpatialLPBootstrap(19, 20, True), "two block lengths"),
])
def test_explicit_assumptions_and_work_budget(config, match):
    with pytest.raises(ValueError, match=match):
        fit(config)


def test_unbalanced_panel_rejected_instead_of_changing_resampling_population():
    data, W = panel()
    data = data.drop(("r0", 10))
    with pytest.raises(ValueError, match="balanced complete"):
        spatial_lp(data, "y", "x", W, horizons=range(2), n_lags=0,
                   bootstrap=SpatialLPBootstrap(19, 3, True, 3))


def test_no_spillover_family_and_conley_studentization():
    b = fit(SpatialLPBootstrap(19, 3, True, 5), spillover_orders=()).bootstrap_result
    assert b.joint_p_value is None
    assert np.all(b.table.xs("indirect", level="label").se == 0)
    data, W = panel()
    coords = pd.DataFrame({"lat": np.arange(8) + 30, "lon": np.arange(8) - 100}, index=W.ids)
    r = spatial_lp(data, "y", "x", W, horizons=[0], n_lags=0, cov_type="conley",
                   coords=coords, cutoff_km=1000,
                   bootstrap=SpatialLPBootstrap(19, 3, True, 5))
    assert r.bootstrap_result.metadata["cov_type"] == "conley"


def test_fixed_entity_count_excludes_cluster_bootstrap_but_preserves_legacy_fit():
    legacy = fit(cov_type="cluster")
    assert np.isfinite(legacy.coef).all()
    for alias in ("cluster", "clustered"):
        with pytest.raises(ValueError, match="entity-cluster covariance"):
            fit(SpatialLPBootstrap(19, 3, True, 5), cov_type=alias)


def test_bootstrap_reestimates_means_after_sampling():
    rng = np.random.default_rng(91)
    values = rng.normal(size=(10, 4, 3))
    sample = np.array([1, 2, 3, 1, 2, 3, 7, 8, 9, 7])
    out = _balanced_fit(values[sample], np.arange(4))
    within = out["X_within"].reshape(10, 4, 2)
    np.testing.assert_allclose(within.mean(axis=0), 0, atol=1e-14)
    np.testing.assert_allclose(within.mean(axis=1), 0, atol=1e-14)
    original = _balanced_fit(values, np.arange(4))["X_within"].reshape(10, 4, 2)[sample]
    assert np.max(np.abs(original.mean(axis=0))) > .1


def test_real_singular_draw_aborts_instead_of_selectively_retrying():
    # Three retained dates: the fixed RNG's second draw repeats one date,
    # annihilating within variation. No mocked solver or covariance is used.
    data, W = panel(T=4)
    with pytest.raises(ValueError, match="draw 1 failed; no draws discarded"):
        spatial_lp(data, "y", "x", W, horizons=[0], n_lags=0,
                   bootstrap=SpatialLPBootstrap(19, 1, True, 0))
