"""Tests for puremacro.inference.proxy (effective sample size, references, matched filter)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import integrate

from puremacro.inference import proxy as P


def _ar1(phi: float, n: int, rng) -> np.ndarray:
    x = np.zeros(n)
    e = rng.standard_normal(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def _irregular_ages(rng, lo=100.0, hi=1850.0, n=90):
    return np.sort(rng.uniform(lo, hi, n))


def _gp_from(t, unc, v1, tau1, v2, tau2, firn, trend_df=2):
    S = (v1 * P.ou_firn_covariance(t, tau1, firn) + v2 * P.ou_firn_covariance(t, tau2, firn)
         + np.diag(unc ** 2))
    return P.GPFit(v1, tau1, v2, tau2, firn, trend_df, 0.0, len(t), S, t)


def _check_contract(res):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    assert isinstance(res.summary(), str) and res.summary()
    fr = res.to_frame()
    assert isinstance(fr, pd.DataFrame) and len(fr) >= 1
    assert isinstance(res.to_markdown(), str)
    assert "tabular" in res.to_latex()
    assert isinstance(res.to_typst(), str)
    fig = res.plot()
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


# ----------------------------------------------------------------------------- effective sample size
def test_ess_ar1_matches_theory():
    rng = np.random.default_rng(0)
    phi, n = 0.9, 40_000
    r = P.effective_sample_size(_ar1(phi, n, rng), "ar1")
    assert r.rho1 == pytest.approx(phi, abs=0.01)
    assert r.n_eff_ar1 / n == pytest.approx((1 - phi) / (1 + phi), rel=0.1)
    assert r.tau_int == pytest.approx((1 + phi) / (1 - phi), rel=0.2)
    # first differences of an AR(1) with phi near one are close to white noise
    assert abs(r.rho1_diff) < 0.1


def test_ess_white_noise_is_close_to_n_and_capped():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(5000)
    r = P.effective_sample_size(x)
    assert r.n_eff_int == pytest.approx(5000, rel=0.15)
    assert r.n_eff_ar1 <= r.n
    # over-differenced: first differences of white noise have rho1 = -0.5; n_eff is capped at n - 1
    assert r.n_eff_ar1_diff <= r.n - 1


def test_ess_smooth_series_has_few_observations():
    rng = np.random.default_rng(2)
    from scipy.ndimage import gaussian_filter1d
    x = gaussian_filter1d(rng.standard_normal(2000), 40)
    r = P.effective_sample_size(x)
    assert r.n_eff_int < 60


def test_ess_rejects_short_input_and_drops_nan():
    with pytest.raises(ValueError):
        P.effective_sample_size([1.0, 2.0, np.nan])
    r = P.effective_sample_size(np.r_[np.cumsum(np.random.default_rng(14).standard_normal(50)), np.nan])
    assert r.n == 50


def test_ess_contract():
    _check_contract(P.effective_sample_size(_ar1(0.5, 300, np.random.default_rng(3)), "x"))


# ----------------------------------------------------------------------------- smoother df
def test_smoother_df_bounds_and_monotone():
    x = np.sort(np.random.default_rng(4).uniform(0, 100, 40))
    d_small = P.smoother_df(x, lam=1e-2)
    d_large = P.smoother_df(x, lam=1e3)
    assert 2.0 - 1e-6 <= d_large < d_small <= 40 + 1e-6


# ----------------------------------------------------------------------------- references
def test_rotation_reference_detects_planted_shift_and_counts():
    rng = np.random.default_rng(5)
    n = 200
    y = rng.standard_normal(n)
    y[100:110] += 4.0
    stat = lambda v: v[100:110].mean() - v.mean()  # noqa: E731
    r = P.rotation_reference(stat, y)
    assert r.n_reference == n - 1
    assert r.p_two_sided < 0.02
    assert r.p_one_sided <= r.p_two_sided


def test_rotation_reference_is_roughly_uniform_under_null():
    rng = np.random.default_rng(6)
    ps = []
    for _ in range(150):
        y = rng.standard_normal(80)
        ps.append(P.rotation_reference(lambda v: v[30:35].mean(), y).p_two_sided)
    ps = np.asarray(ps)
    assert 0.38 < ps.mean() < 0.62
    assert np.mean(ps < 0.05) < 0.12


def test_feasible_onset_placebo_exclusion():
    onsets = np.arange(0, 100)
    r = P.feasible_onset_placebo(lambda o: float(o), onsets, actual_onset=50,
                                 catalog_onsets=[10], exclusion=5)
    # excluded: |o-50| <= 5 (11 onsets) and |o-10| <= 5 (11 onsets)
    assert r.n_reference == 100 - 22
    assert r.observed == 50.0
    _check_contract(r)


def test_empty_reference_raises():
    with pytest.raises(ValueError):
        P.feasible_onset_placebo(lambda o: 1.0, [1, 2], actual_onset=1, exclusion=10)


# ----------------------------------------------------------------------------- covariance
def test_ou_firn_covariance_limits_and_symmetry():
    t = np.linspace(0, 200, 41)
    C0 = P.ou_firn_covariance(t, 20.0, 0.0)
    D = t[:, None] - t[None, :]
    np.testing.assert_allclose(C0, np.exp(-np.abs(D) / 20.0))
    C = P.ou_firn_covariance(t, 20.0, 5.0)
    np.testing.assert_allclose(C, C.T, atol=1e-12)
    assert np.linalg.eigvalsh(C).min() > -1e-10
    # two-record form: C(t, t2; s1, s2) = C(t2, t; s2, s1)^T
    t2 = np.array([3.0, 77.0, 150.5])
    A = P.ou_firn_covariance(t, 20.0, 5.0, t2, 2.0)
    B = P.ou_firn_covariance(t2, 20.0, 2.0, t, 5.0)
    np.testing.assert_allclose(A, B.T, rtol=1e-10)


@pytest.mark.parametrize("D", [0.0, 3.0, 25.0])
def test_ou_firn_covariance_matches_quadrature(D):
    tau, s1, s2 = 8.0, 4.0, 3.0
    s = np.hypot(s1, s2)
    f = lambda d: np.exp(-abs(D + d) / tau) * np.exp(-0.5 * (d / s) ** 2) / (s * np.sqrt(2 * np.pi))  # noqa: E731
    val, _ = integrate.quad(f, -12 * s, 12 * s, points=[-D], limit=200)
    got = P.ou_firn_covariance(np.array([D]), tau, s1, np.array([0.0]), s2)[0, 0]
    assert got == pytest.approx(val, rel=1e-7)


# ----------------------------------------------------------------------------- template
def test_window_template_shape():
    ages = np.arange(1300.0, 1450.0)
    h = P.window_template(ages, 1347, 1353, lifetime=10.0, firn_sigma=0.0)
    assert np.all(h[ages < 1347] == 0)
    assert h.max() == pytest.approx(1.0)
    assert ages[np.argmax(h)] == 1353
    # exponential decay after the window
    assert h[ages == 1363][0] == pytest.approx(np.exp(-1.0), rel=1e-6)
    hs = P.window_template(ages, 1347, 1353, lifetime=10.0, firn_sigma=5.0)
    assert hs.max() < 1.0 and hs[ages == 1340][0] > 0


def test_window_template_outside_grid_raises():
    with pytest.raises(ValueError):
        P.window_template(np.arange(10.0), 5000, 5010, 10.0, 0.0)


# ----------------------------------------------------------------------------- matched filter
@pytest.fixture(scope="module")
def design():
    rng = np.random.default_rng(7)
    t = _irregular_ages(rng, n=90)
    t = np.r_[t, [1348.0, 1350.5, 1352.0]]
    t.sort()
    unc = np.full(len(t), 0.6)
    gp = _gp_from(t, unc, v1=1.0, tau1=5.0, v2=4.0, tau2=300.0, firn=5.0, trend_df=2)
    return t, unc, gp


def test_matched_filter_recovers_planted_amplitude(design):
    t, unc, gp = design
    L = np.linalg.cholesky(gp.Sigma)
    h = P.window_template(t, 1347, 1353, 10.0, gp.firn_sigma)
    rng = np.random.default_rng(8)
    y = 280.0 + 0.002 * (t - t.mean()) + L @ rng.standard_normal(len(t))
    base = P.matched_filter(t, y, unc, gp, 1347, 1353, 10.0, n_boot=200)
    A_true = 6 * base.se
    r = P.matched_filter(t, y + A_true * h, unc, gp, 1347, 1353, 10.0, n_boot=400, label="planted")
    assert abs(r.amplitude - (base.amplitude + A_true)) < 1e-8 * max(1.0, abs(A_true))
    assert abs(r.amplitude - A_true) < 4 * r.se
    assert r.p_normal < 0.01 and r.p_boot_two < 0.02
    assert r.n_samples_in_window == 3
    assert r.mde_80 == pytest.approx((1.959964 + 0.841621) * r.se, rel=1e-5)
    assert len(r.rotation_t) == r.n_rotation == len(r.rotation_onsets)
    _check_contract(r)


def test_matched_filter_t_is_standard_normal_under_null(design):
    t, unc, gp = design
    L = np.linalg.cholesky(gp.Sigma)
    rng = np.random.default_rng(9)
    ts = [P.matched_filter(t, L @ rng.standard_normal(len(t)), unc, gp, 1347, 1353, 10.0,
                           n_boot=10, onset_step=50).t for _ in range(300)]
    assert np.std(ts) == pytest.approx(1.0, abs=0.12)
    assert abs(np.mean(ts)) < 0.2


def test_matched_filter_student_t_option(design):
    t, unc, gp = design
    y = np.linalg.cholesky(gp.Sigma) @ np.random.default_rng(10).standard_normal(len(t))
    rn = P.matched_filter(t, y, unc, gp, 1347, 1353, 10.0, n_boot=50)
    rt = P.matched_filter(t, y, unc, gp, 1347, 1353, 10.0, n_boot=50, t_df=5)
    assert np.isnan(rn.p_t) and np.isnan(rn.t_df)
    assert rt.p_t >= rt.p_normal
    assert rt.mde_80 > rn.mde_80
    assert rt.amplitude == rn.amplitude


def test_matched_filter_checks_sigma_shape(design):
    t, unc, gp = design
    with pytest.raises(ValueError):
        P.matched_filter(t[:-1], np.zeros(len(t) - 1), unc[:-1], gp, 1347, 1353, 10.0)


def test_ou_firn_gp_fit_runs_and_masks(design):
    t, unc, gp = design
    L = np.linalg.cholesky(gp.Sigma)
    y = 5.0 + L @ np.random.default_rng(11).standard_normal(len(t))
    fit = P.OUFirnGP(firn_sigma=5.0, trend_df=2).fit(t, y, unc, n_starts=1, mask_window=(1340, 1400))
    assert fit.Sigma.shape == (len(t), len(t))
    assert fit.n == int(np.sum(~((t >= 1340) & (t <= 1400))))
    assert np.isfinite(fit.loglik)
    total = fit.var_fast + fit.var_slow
    assert 0.2 * 5.0 < total < 5.0 * 5.0
    _check_contract(fit)
    fixed = P.OUFirnGP(firn_sigma=5.0, trend_df=2).fit(t, y, unc, fixed_tau=(5.0, 300.0))
    assert fixed.tau_fast == pytest.approx(5.0) and fixed.tau_slow == pytest.approx(300.0)


# ----------------------------------------------------------------------------- stacking
def test_stacked_matched_filter_pools_records():
    rng = np.random.default_rng(12)
    ta = np.sort(rng.uniform(1000, 1850, 60))
    tb = np.sort(rng.uniform(1000, 1850, 70))
    recs0 = [P.ProxyRecord("A", ta, np.zeros(60), np.full(60, 1.0), 5.0, 2),
             P.ProxyRecord("B", tb, np.zeros(70), np.full(70, 0.5), 3.0, 2)]
    S = P._stacked_cov(recs0, None, 1.0, 5.0, 4.0, 300.0)
    y = np.linalg.cholesky(S) @ rng.standard_normal(130)
    recs = [P.ProxyRecord("A", ta, 280 + y[:60], np.full(60, 1.0), 5.0, 2),
            P.ProxyRecord("B", tb, 283 + y[60:], np.full(70, 0.5), 3.0, 2)]
    r = P.stacked_matched_filter(recs, 1347, 1353, 15.0, n_boot=200, n_starts=1, label="stack")
    assert r.n_total == 130 and r.records == ("A", "B")
    assert r.se <= r.per_record["se"].min() + 1e-9
    assert np.isfinite(r.p_rotation_two)
    _check_contract(r)


def test_proxy_record_validates_lengths():
    with pytest.raises(ValueError):
        P.ProxyRecord("bad", np.arange(3.0), np.arange(4.0), np.ones(3), 5.0)


# ----------------------------------------------------------------------------- calibration
def test_calibration_fixed_covariance_size_and_power(design):
    t, unc, gp = design
    c = P.calibrate_matched_filter(t, unc, gp, 1347, 1353, 10.0, amplitudes=(0.0, 1.0), R=200,
                                   onset_step=50, seed=13)
    tb = c.table.set_index("amplitude_mde")
    assert tb.loc[0.0, "reject_normal"] < 0.1
    assert 0.65 < tb.loc[1.0, "reject_normal"] < 0.93
    assert tb.loc[0.0, "sd_over_se"] == pytest.approx(1.0, abs=0.15)
    assert not c.refit
    _check_contract(c)
