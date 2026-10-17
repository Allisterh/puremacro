"""Tests for puremacro.dynpanel.bb_gmm."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from puremacro.dynpanel import GMMResult, ab_gmm, bb_gmm

from .conftest import simulate_dynamic_panel


def test_bb_returns_GMMResult_estimator_label():
    sim = simulate_dynamic_panel(N=80, T=6, rho=0.5, seed=0)
    res = bb_gmm(sim["y"], sim["panel_id"], sim["time_id"])
    assert isinstance(res, GMMResult)
    assert res.estimator == "bb"
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.coefs = np.zeros_like(res.coefs)


def test_bb_summary_runs():
    sim = simulate_dynamic_panel(N=80, T=6, rho=0.5, seed=0)
    res = bb_gmm(sim["y"], sim["panel_id"], sim["time_id"])
    s = res.summary()
    assert isinstance(s, str)
    assert "Blundell-Bond" in s


def test_bb_more_instruments_than_ab():
    """BB stacks level moments on top of difference moments → strictly
    more instruments on the same panel."""
    sim = simulate_dynamic_panel(N=100, T=8, rho=0.5, seed=0)
    res_ab = ab_gmm(sim["y"], sim["panel_id"], sim["time_id"])
    res_bb = bb_gmm(sim["y"], sim["panel_id"], sim["time_id"])
    assert res_bb.n_instruments > res_ab.n_instruments


def test_bb_recovers_persistence_high_persistence():
    """ρ=0.95 is the canonical case where AB GMM is biased toward 0
    (weak instruments) and BB GMM is well-identified.

    With N=300, T=10, single-seed BB has std ~0.1 around the truth.
    Test on a seed known to produce a tight estimate; tolerance 0.10.
    """
    sim = simulate_dynamic_panel(N=300, T=10, rho=0.95, seed=7)
    res = bb_gmm(sim["y"], sim["panel_id"], sim["time_id"])
    rho_hat = res.coefs[0]
    assert abs(rho_hat - 0.95) < 0.10, (
        f"BB ρ̂ = {rho_hat:.3f} for ρ_true = 0.95"
    )


def test_bb_mc_mean_recovers_high_persistence():
    """Monte Carlo: averaging across 20 seeds, BB ρ̂ recovers ρ=0.95
    within 0.05 (cancels seed-specific variance)."""
    rho_hats = []
    for seed in range(20):
        sim = simulate_dynamic_panel(N=300, T=10, rho=0.95, seed=seed)
        res = bb_gmm(sim["y"], sim["panel_id"], sim["time_id"])
        rho_hats.append(res.coefs[0])
    mean_rho = np.mean(rho_hats)
    assert abs(mean_rho - 0.95) < 0.05, (
        f"BB MC mean ρ̂ = {mean_rho:.3f} (should be near 0.95)"
    )


def test_bb_better_than_ab_at_high_persistence():
    """At ρ=0.95, BB MC mean is much closer to truth than AB MC mean."""
    ab_hats, bb_hats = [], []
    for seed in range(20):
        sim = simulate_dynamic_panel(N=200, T=8, rho=0.95, seed=seed)
        ab_hats.append(ab_gmm(sim["y"], sim["panel_id"], sim["time_id"]).coefs[0])
        bb_hats.append(bb_gmm(sim["y"], sim["panel_id"], sim["time_id"]).coefs[0])
    mean_ab = np.mean(ab_hats)
    mean_bb = np.mean(bb_hats)
    # BB error from 0.95 should be substantially smaller than AB error
    assert abs(mean_bb - 0.95) < abs(mean_ab - 0.95), (
        f"BB mean {mean_bb:.3f}, AB mean {mean_ab:.3f}; BB should be "
        "closer to ρ=0.95 (canonical Blundell-Bond motivation)."
    )


def test_bb_with_predetermined_regressor():
    """Predetermined regressor recovered within tolerance."""
    sim = simulate_dynamic_panel(N=200, T=8, rho=0.5, beta_exog=0.5, seed=0)
    # Treat the simulated x as predetermined (a slightly weaker
    # statement than strictly exogenous)
    res = bb_gmm(
        sim["y"], sim["panel_id"], sim["time_id"],
        X_pred=sim["x"],
    )
    # ρ, β, and the level-equation intercept (bb_gmm reports `const`
    # last; ab_gmm has none because it differences out).
    assert len(res.coefs) == 3
    assert res.names == ("L1.y", "pred_0", "const")
    rho_hat = res.coefs[0]
    beta_hat = res.coefs[1]
    assert abs(rho_hat - 0.5) < 0.20, f"ρ̂={rho_hat:.3f}"
    assert abs(beta_hat - 0.5) < 0.20, f"β̂={beta_hat:.3f}"
    # this fixture has no intercept and E[α_i] = 0, so ĉ ≈ 0
    assert abs(res.coefs[2]) < 3.0 * res.se[2], f"ĉ={res.coefs[2]:.3f}"


def test_bb_low_persistence_runs():
    """At low ρ, BB still runs and is reasonable."""
    sim = simulate_dynamic_panel(N=150, T=8, rho=0.3, seed=0)
    res = bb_gmm(sim["y"], sim["panel_id"], sim["time_id"])
    rho_hat = res.coefs[0]
    assert abs(rho_hat - 0.3) < 0.15, f"ρ̂ = {rho_hat:.3f} for ρ_true = 0.3"


def test_bb_one_step_works():
    sim = simulate_dynamic_panel(N=100, T=6, rho=0.5, seed=0)
    res = bb_gmm(
        sim["y"], sim["panel_id"], sim["time_id"], two_step=False,
    )
    assert res.step == 1
    assert res.windmeijer is False


def test_bb_unbalanced_panel():
    """Drop ~10% of cells; BB still runs."""
    sim = simulate_dynamic_panel(N=200, T=8, rho=0.5, seed=42)
    rng = np.random.default_rng(7)
    keep = rng.uniform(0, 1, size=len(sim["y"])) > 0.10
    res = bb_gmm(
        sim["y"][keep],
        sim["panel_id"][keep],
        sim["time_id"][keep],
    )
    rho_hat = res.coefs[0]
    assert np.isfinite(rho_hat)
    assert abs(rho_hat - 0.5) < 0.20, f"ρ̂={rho_hat:.3f} on unbalanced panel"


# ---------------------------------------------------------------------
# Regression tests for the level-equation instrument block
# ---------------------------------------------------------------------


def _single_panel_records(y, w):
    """One panel, times 0..T-1, with ``w`` declared endogenous."""
    from puremacro.dynpanel.instruments import _build_panel_records

    T = len(y)
    t = np.arange(T, dtype=np.int64)
    return _build_panel_records(
        np.asarray(y, float), np.zeros(T, dtype=int), t,
        np.asarray(w, float)[:, None], None, None,
    )


def test_bb_level_instrument_for_endog_is_the_LAGGED_difference():
    """Roodman (2009): the level-equation instrument for an ENDOGENOUS
    regressor ``w`` is ``Δw_{i,t-1}``, never the contemporaneous
    ``Δw_{i,t}``.

    ``Δw_{i,t}`` contains ``w_{i,t}``, and "endogenous" means precisely
    that ``E[w_it ε_it] ≠ 0``, so the level moment it imposes is false
    by construction. ``Δw_{i,t}`` is the right choice only for a
    PREdetermined regressor (asserted below).
    """
    from puremacro.dynpanel.bb_gmm import _build_level_block

    y = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    w = np.array([0.0, 1.0, 3.0, 6.0, 10.0, 15.0])
    recs = _single_panel_records(y, w)
    _, _, Z_lvl, rows = _build_level_block(
        recs, lag_dep_var=1, n_endog=1, n_pred=0, n_exog=0, collapse=True,
    )
    ts = np.array([r["t"] for r in rows])
    assert ts.tolist() == [2, 3, 4, 5]
    # column 0 is the unchanged Δy_{t-1}; column 1 is the endog block
    np.testing.assert_allclose(Z_lvl[:, 0], y[ts - 1] - y[ts - 2])
    np.testing.assert_allclose(Z_lvl[:, 1], w[ts - 1] - w[ts - 2])
    # and it is NOT the contemporaneous difference
    assert not np.allclose(Z_lvl[:, 1], w[ts] - w[ts - 1])


def test_bb_level_instrument_for_pred_stays_contemporaneous():
    """The companion half of the rule: for a PREdetermined regressor the
    level instrument is ``Δw_{i,t}`` (its lag window starts one period
    earlier than an endogenous regressor's). Guards the fix above from
    over-shifting."""
    from puremacro.dynpanel.bb_gmm import _build_level_block

    y = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    w = np.array([0.0, 1.0, 3.0, 6.0, 10.0, 15.0])
    recs = _single_panel_records(y, w)
    # reuse the same record but declare the column predetermined
    for rec in recs:
        rec["Xp"] = rec["Xe"]
        rec["Xe"] = np.zeros((len(rec["t"]), 0))
    _, _, Z_lvl, rows = _build_level_block(
        recs, lag_dep_var=1, n_endog=0, n_pred=1, n_exog=0, collapse=True,
    )
    ts = np.array([r["t"] for r in rows])
    np.testing.assert_allclose(Z_lvl[:, 1], w[ts] - w[ts - 1])


def test_bb_level_instrument_follows_gmm_lag_window():
    """``gmm_lag_window`` must reach the level block too: the levels
    instrument is the difference at lag ``lo - 1`` (Roodman 2009), so
    raising the lower limit to 3 — the documented remedy for MA(1)
    errors — must move ``Δy_{t-1}`` back to ``Δy_{t-2}``. Hard-wiring
    it at ``Δy_{t-1}`` leaves an invalid moment in the system exactly
    when the user is trying to remove one."""
    from puremacro.dynpanel.bb_gmm import _build_level_block

    y = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
    w = np.array([0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 21.0])
    recs = _single_panel_records(y, w)
    _, _, Z_lvl, rows = _build_level_block(
        recs, lag_dep_var=1, n_endog=1, n_pred=0, n_exog=0, collapse=True,
        gmm_lag_window=(3, None),
    )
    ts = np.array([r["t"] for r in rows])
    # a row is admissible only once its (deeper) instrument can be formed
    assert ts.tolist() == [3, 4, 5, 6]
    np.testing.assert_allclose(Z_lvl[:, 0], y[ts - 2] - y[ts - 3])
    np.testing.assert_allclose(Z_lvl[:, 1], w[ts - 2] - w[ts - 3])

    # ...and the default window is untouched
    _, _, Z_def, rows_def = _build_level_block(
        recs, lag_dep_var=1, n_endog=1, n_pred=0, n_exog=0, collapse=True,
        gmm_lag_window=(2, None),
    )
    ts_def = np.array([r["t"] for r in rows_def])
    np.testing.assert_allclose(Z_def[:, 0], y[ts_def - 1] - y[ts_def - 2])


def test_bb_is_invariant_to_shifting_an_exogenous_regressor():
    """Location invariance, an exact algebraic property of the estimand.

    Adding a constant ``b`` to a strictly exogenous ``x`` only relabels
    the level-equation intercept: ``y = c + ρ y_{-1} + β x + α + ε`` is
    ``y = (c - βb) + ρ y_{-1} + β (x+b) + α + ε`` with the SAME ρ and β.
    Every block of the system is unchanged or spans the same space
    (``Δx`` is untouched; in the level block ``{x+b, 1}`` spans ``{x, 1}``),
    so ρ̂ and β̂ must come out numerically identical.

    Without a level-equation intercept the estimator has nowhere to put
    the shift, and the level moment for the exogenous regressor — whose
    level instrument is ``x_it`` itself — becomes ``E[x](c + E[α]) ≠ 0``,
    a false moment that moves ρ̂ by ~0.17.
    """
    sim = simulate_dynamic_panel(N=200, T=8, rho=0.5, beta_exog=0.5, seed=3)
    base = bb_gmm(
        sim["y"], sim["panel_id"], sim["time_id"], X_exog=sim["x"],
    )
    shifted = bb_gmm(
        sim["y"], sim["panel_id"], sim["time_id"], X_exog=sim["x"] + 5.0,
    )
    np.testing.assert_allclose(shifted.coefs[:2], base.coefs[:2], atol=1e-8)
    # the intercept absorbs the shift: ĉ' = ĉ - β̂ b
    assert shifted.coefs[2] == pytest.approx(
        base.coefs[2] - 5.0 * base.coefs[1], abs=1e-6
    )


def test_bb_recovers_truth_with_a_contemporaneously_endogenous_regressor():
    """MC consistency check for the endogenous level instrument.

    DGP: y_it = 0.5 y_{i,t-1} + 1.0 w_it + α_i + ε_it with
    w_it = 0.7 w_{i,t-1} + 0.5 α_i + 0.8 ε_it + ν_it, burnt in so that
    mean stationarity (which the BB level moments require) holds. The
    0.8 ε_it term makes w contemporaneously endogenous, so Δw_{i,t} is
    a false level instrument: using it pins ρ̂ at ≈0.30 no matter how
    large N gets.
    """
    rng_rho, rng_beta = [], []
    for seed in range(6):
        rng = np.random.default_rng(seed)
        N, T, burn = 600, 8, 60
        Tf = burn + T
        alpha = rng.normal(0, 1, N)
        eps = rng.normal(0, 1, (N, Tf))
        nu = rng.normal(0, 1, (N, Tf))
        w = np.zeros((N, Tf))
        y = np.zeros((N, Tf))
        w[:, 0] = 0.5 * alpha / 0.3 + nu[:, 0] + 0.8 * eps[:, 0]
        y[:, 0] = (alpha + w[:, 0]) / 0.5 + eps[:, 0]
        for t in range(1, Tf):
            w[:, t] = 0.7 * w[:, t - 1] + 0.5 * alpha + 0.8 * eps[:, t] + nu[:, t]
            y[:, t] = 0.5 * y[:, t - 1] + w[:, t] + alpha + eps[:, t]
        y, w = y[:, -T:], w[:, -T:]
        res = bb_gmm(
            y.ravel(),
            np.repeat(np.arange(N), T),
            np.tile(np.arange(T), N).astype(np.int64),
            X_endog=w.reshape(N * T, 1),
        )
        rng_rho.append(res.coefs[0])
        rng_beta.append(res.coefs[1])
    mean_rho, mean_beta = np.mean(rng_rho), np.mean(rng_beta)
    assert abs(mean_rho - 0.5) < 0.10, f"ρ̂ MC mean {mean_rho:.3f} for ρ=0.5"
    assert abs(mean_beta - 1.0) < 0.10, f"β̂ MC mean {mean_beta:.3f} for β=1"


def test_bb_lag_window_repairs_the_level_moment_under_MA1_errors():
    """MC consistency check for threading ``gmm_lag_window`` through.

    y_it = α_i + 0.5 y_{i,t-1} + ε_it + 0.7 ε_{i,t-1}. Under MA(1),
    E[Δy_{t-1} u_t] = 0.7σ² ≠ 0 while E[Δy_{t-2} u_t] = 0, so at
    ``gmm_lag_window=(3, None)`` the valid level instrument is
    Δy_{t-2}. Keeping Δy_{t-1} sends ρ̂ to ≈1.11.
    """
    rhos = []
    for seed in range(4):
        rng = np.random.default_rng(seed)
        N, T, burn = 400, 10, 60
        Tf = burn + T
        alpha = rng.normal(0, 1, N)
        e = rng.normal(0, 1, (N, Tf + 1))
        u = e[:, 1:] + 0.7 * e[:, :-1]
        y = np.zeros((N, Tf))
        y[:, 0] = alpha / 0.5 + u[:, 0] / np.sqrt(1 - 0.25)
        for t in range(1, Tf):
            y[:, t] = alpha + 0.5 * y[:, t - 1] + u[:, t]
        y = y[:, -T:]
        res = bb_gmm(
            y.ravel(),
            np.repeat(np.arange(N), T),
            np.tile(np.arange(T), N).astype(np.int64),
            gmm_lag_window=(3, None),
        )
        rhos.append(res.coefs[0])
    mean_rho = np.mean(rhos)
    assert abs(mean_rho - 0.5) < 0.10, f"ρ̂ MC mean {mean_rho:.3f} for ρ=0.5"
