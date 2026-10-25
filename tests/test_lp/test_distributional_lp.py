"""Unit tests for puremacro.lp.distributional (Track III.6)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from puremacro.lp.distributional import (
    DistributionalBridgeResult,
    DistributionalLPResult,
    bridge_hank_distributional_irf,
    estimate_distributional_lp,
    estimate_quantile_lp,
)


def _generate_synthetic_share_data(n_obs: int = 250, n_bins: int = 10, seed: int = 42) -> tuple[pd.DataFrame, list[str]]:
    """Generate synthetic time series of decile shares summing to 1.0 and a shock."""
    rng = np.random.default_rng(seed)
    shock = rng.normal(0.0, 1.0, size=n_obs)
    macro_ctrl = rng.normal(0.0, 1.0, size=n_obs)

    # Base shares (e.g. log-normal wealth distribution)
    grid = np.linspace(-1.5, 1.5, n_bins)
    base_shares = np.exp(grid) / np.sum(np.exp(grid))

    shares = np.zeros((n_obs, n_bins), dtype=np.float64)
    # Impact vector summing to 0 (top deciles gain, bottom deciles lose)
    impact = np.linspace(-0.02, 0.02, n_bins)
    impact -= np.mean(impact)

    s_t = base_shares.copy()
    for t in range(n_obs):
        # AR(1) dynamics with shock
        s_t = 0.85 * s_t + 0.15 * base_shares + impact * shock[t] + 0.005 * rng.normal(size=n_bins)
        # Normalize to strictly sum to 1.0
        s_t = np.maximum(s_t, 1e-4)
        s_t = s_t / np.sum(s_t)
        shares[t] = s_t

    bin_cols = [f"D{i + 1}" for i in range(n_bins)]
    data_dict = {col: shares[:, i] for i, col in enumerate(bin_cols)}
    data_dict["shock"] = shock
    data_dict["gdp_growth"] = macro_ctrl

    df = pd.DataFrame(data_dict)
    return df, bin_cols


def test_distributional_lp_exact_adding_up():
    """Verify that distributional LP satisfies exact adding-up max_h |sum_b beta_{b,h}| < 1e-12."""
    df, bin_cols = _generate_synthetic_share_data(n_obs=200, n_bins=10, seed=123)

    res = estimate_distributional_lp(
        df,
        shock="shock",
        bin_cols=bin_cols,
        horizons=range(0, 21),
        n_lags=2,
        controls=["gdp_growth"],
        cov_type="hac",
        enforce_adding_up=True,
    )

    assert isinstance(res, DistributionalLPResult)
    assert res.irf_matrix.shape == (21, 10)
    assert res.se_matrix.shape == (21, 10)
    assert res.ci_lower.shape == (21, 10)
    assert res.ci_upper.shape == (21, 10)

    # Mandatory mathematical constraint: max_h |sum_b beta_{b, h}| < 10^-12
    sum_across_bins = np.sum(res.irf_matrix, axis=1)
    max_err = float(np.max(np.abs(sum_across_bins)))
    assert max_err < 1e-12, f"Adding-up error exceeded 1e-12: {max_err:.2e}"
    assert res.check_adding_up(tol=1e-12)
    assert res.adding_up_max_error < 1e-12

    # Verification of container export formats
    df_long = res.to_dataframe()
    assert len(df_long) == 21 * 10
    assert set(df_long.columns) >= {"horizon", "bin", "beta", "se", "ci_lower", "ci_upper"}

    df_h0 = res.to_dataframe(horizon=0)
    assert len(df_h0) == 10
    assert np.allclose(df_h0["beta"].values, res.irf_matrix[0], atol=1e-14)

    summary_df = res.summary()
    assert len(summary_df) == 10
    assert "h=0_beta" in summary_df.columns

    md_str = res.to_markdown()
    assert "h=0_beta" in md_str

    lat_str = res.to_latex()
    assert "h=0" in lat_str

    typ_str = res.to_typst()
    assert len(typ_str) > 0

    fig = res.plot_deciles()
    assert fig is not None


def test_distributional_lp_density_quadrature():
    """Verify that continuous density bins satisfy exact quadrature adding-up sum_b beta_b w_b = 0."""
    rng = np.random.default_rng(77)
    n_obs = 180
    n_bins = 8
    # Non-uniform quadrature weights
    w_vec = np.array([0.5, 1.0, 1.5, 2.0, 2.0, 1.5, 1.0, 0.5])
    w_vec = w_vec / np.sum(w_vec)  # sum to 1

    shock = rng.normal(0, 1, size=n_obs)
    densities = np.zeros((n_obs, n_bins))
    for t in range(n_obs):
        # Generate random density values
        d = np.abs(rng.normal(1.0, 0.2, size=n_bins))
        densities[t] = d / np.dot(d, w_vec)

    bin_cols = [f"bin_{i}" for i in range(n_bins)]
    df = pd.DataFrame(densities, columns=bin_cols)
    df["shock"] = shock

    res = estimate_distributional_lp(
        df,
        shock="shock",
        bin_cols=bin_cols,
        horizons=range(0, 10),
        bin_widths=w_vec,
        distribution_type="density",
        enforce_adding_up=True,
    )

    weighted_sums = res.irf_matrix @ w_vec
    max_err = float(np.max(np.abs(weighted_sums)))
    assert max_err < 1e-12, f"Weighted density adding-up error exceeded 1e-12: {max_err:.2e}"
    assert res.check_adding_up(tol=1e-12)


def test_quantile_lp_monte_carlo_location_scale():
    """Verify that estimate_quantile_lp recovers known DGP location-scale shifts within 95% CI."""
    rng = np.random.default_rng(2026)
    n_obs = 400
    shock = rng.normal(0.0, 1.0, size=n_obs)

    # Location-scale DGP: y_t = mu_t + sigma_t * eps_t
    # mu_t = 0.6 * shock_t
    # sigma_t = 1.0 + 0.35 * shock_t (bounded away from 0)
    # At impact h=0: true effect on tau quantile is 0.6 + 0.35 * Phi^-1(tau)
    eps = rng.normal(0.0, 1.0, size=n_obs)
    sigma = 1.0 + 0.35 * np.tanh(shock)  # smooth positive
    y = 0.6 * shock + sigma * eps

    df = pd.DataFrame({"y": y, "shock": shock})
    test_quantiles = (0.10, 0.25, 0.50, 0.75, 0.90)

    res = estimate_quantile_lp(
        df,
        shock="shock",
        outcome="y",
        quantiles=test_quantiles,
        horizons=[0, 1, 2],
        n_lags=1,
        ci=0.95,
    )

    assert isinstance(res, DistributionalLPResult)
    assert res.distribution_type == "quantiles"
    assert res.irf_matrix.shape == (3, len(test_quantiles))

    # Test recovery of location-scale shift at h=0
    for j, tau in enumerate(test_quantiles):
        # Linear approximation around shock=0: d y / d shock = 0.6 + 0.35 * z_tau
        z_tau = float(norm.ppf(tau))
        true_beta = 0.6 + 0.35 * z_tau
        est_beta = float(res.irf_matrix[0, j])
        lo = float(res.ci_lower[0, j])
        hi = float(res.ci_upper[0, j])

        # Assert estimate is within 95% confidence bands
        assert lo <= est_beta <= hi, f"Point estimate {est_beta} outside CI [{lo}, {hi}] for tau={tau}"
        # Assert true beta is covered or close to confidence bounds
        dist_to_ci = max(0.0, lo - true_beta, true_beta - hi)
        assert dist_to_ci < 0.25, f"True beta {true_beta:.3f} too far from CI [{lo:.3f}, {hi:.3f}] for tau={tau}"


def test_distributional_lp_panel_driscoll_kraay():
    """Verify distributional LP on panel data with Driscoll-Kraay covariance."""
    rng = np.random.default_rng(999)
    n_units = 5
    n_periods = 80
    n_bins = 4

    records = []
    for u in range(n_units):
        shocks = rng.normal(0, 1, size=n_periods)
        for t in range(n_periods):
            shares = np.array([0.1, 0.2, 0.3, 0.4]) + 0.01 * np.array([-1, -0.5, 0.5, 1]) * shocks[t]
            shares = shares / np.sum(shares)
            row = {"unit": f"country_{u}", "time": t, "shock": shocks[t]}
            for b in range(n_bins):
                row[f"share_{b}"] = shares[b]
            records.append(row)

    df_panel = pd.DataFrame(records)

    res = estimate_distributional_lp(
        df_panel,
        shock="shock",
        bin_cols=[f"share_{b}" for b in range(n_bins)],
        horizons=range(0, 5),
        cov_type="driscoll_kraay",
        unit_col="unit",
        time_col="time",
        enforce_adding_up=True,
    )

    assert res.cov_type == "driscoll_kraay"
    assert res.check_adding_up(tol=1e-12)
    assert np.all(res.se_matrix > 0)


def test_bridge_hank_distributional_irf():
    """Verify bridge_hank_distributional_irf alignment metrics."""
    df, bin_cols = _generate_synthetic_share_data(n_obs=150, n_bins=5, seed=54321)

    lp_res = estimate_distributional_lp(
        df,
        shock="shock",
        bin_cols=bin_cols,
        horizons=range(0, 8),
        n_lags=1,
    )

    # Simulated structural HANK model IRF (slightly perturbed empirical)
    H, B = lp_res.irf_matrix.shape
    hank_irf = lp_res.irf_matrix + 0.002 * np.sin(np.arange(H)[:, None])

    bridge = bridge_hank_distributional_irf(lp_res, hank_irf)

    assert isinstance(bridge, DistributionalBridgeResult)
    assert bridge.distance_l2 >= 0.0
    assert bridge.distance_mahalanobis >= 0.0
    assert bridge.mean_absolute_error >= 0.0
    assert 0.0 <= bridge.sign_concordance <= 1.0

    for b in bin_cols:
        assert b in bridge.decile_correlations
        assert -1.0 <= bridge.decile_correlations[b] <= 1.0
        assert b in bridge.peak_gaps

    summary_df = bridge.summary()
    assert "L2 Distance (RMSD)" in summary_df.index
    assert "Average Decile Correlation" in summary_df.index

    fig = bridge.plot()
    assert fig is not None


def test_bridge_hank_with_dataframe_and_mapping():
    """Verify bridge accepts a DataFrame with custom column mapping."""
    df, bin_cols = _generate_synthetic_share_data(n_obs=150, n_bins=3, seed=888)

    lp_res = estimate_distributional_lp(
        df,
        shock="shock",
        bin_cols=bin_cols,
        horizons=range(0, 6),
    )

    # DataFrame with different column names e.g. "C_decile_1", etc.
    hank_df = pd.DataFrame({
        "C_decile_1": lp_res.irf_matrix[:, 0] * 0.9,
        "C_decile_2": lp_res.irf_matrix[:, 1] * 0.95,
        "C_decile_3": lp_res.irf_matrix[:, 2] * 1.05,
    })

    mapping = {"D1": "C_decile_1", "D2": "C_decile_2", "D3": "C_decile_3"}
    bridge = bridge_hank_distributional_irf(lp_res, hank_df, bin_mapping=mapping)

    assert bridge.distance_l2 > 0.0
    assert len(bridge.decile_correlations) == 3
    assert bridge.to_dataframe().shape == (3, 2)
