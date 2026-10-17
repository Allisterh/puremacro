"""Tests for the modern staggered-DiD estimators.

Each test feeds a synthetic panel with a known true ATT (and known
heterogeneity across cohorts where relevant) and asks whether the
estimator recovers it within bootstrap tolerance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.did import (
    callaway_santanna, sun_abraham,
    borusyak_jaravel_spiess, synthetic_did,
    CallawaySantannaResult, SunAbrahamResult,
    BorusyakJaravelSpiessResult, SyntheticDiDResult,
)


def _staggered_panel(
    rng,
    *,
    n_units: int = 30,
    T: int = 20,
    cohorts: tuple = (8, 12, None),       # None = never-treated
    true_att_by_cohort: dict | None = None,
    sigma: float = 0.5,
):
    """Two-way FE panel with cohort-specific treatment effects.

    Y_{i,t} = α_i + λ_t + (treat dummy * τ_g) + ε_{i,t}.
    """
    if true_att_by_cohort is None:
        true_att_by_cohort = {c: 1.0 for c in cohorts if c is not None}

    units = [f"u{i:02d}" for i in range(n_units)]
    times = list(range(T))
    cohort_assign = rng.choice(cohorts, size=n_units, replace=True)

    rows = []
    alpha = rng.standard_normal(n_units)
    lam = rng.standard_normal(T) * 0.3
    for i, u in enumerate(units):
        g = cohort_assign[i]
        for t in times:
            treat = 0.0 if g is None else (1.0 if t >= g else 0.0)
            tau = 0.0 if g is None else true_att_by_cohort[g]
            y = alpha[i] + lam[t] + tau * treat + sigma * rng.standard_normal()
            rows.append({
                "unit": u, "time": t, "y": y,
                "treat_time": np.nan if g is None else float(g),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Callaway-Sant'Anna
# ---------------------------------------------------------------------------


def test_callaway_santanna_recovers_homogeneous_att(rng):
    df = _staggered_panel(rng, n_units=80, T=16,
                            cohorts=(6, 10, None),
                            true_att_by_cohort={6: 1.5, 10: 1.5})
    res = callaway_santanna(df, n_boot=30)
    assert isinstance(res, CallawaySantannaResult)
    assert abs(res.att_overall - 1.5) < 0.5


def test_callaway_santanna_recovers_heterogeneous_atts(rng):
    df = _staggered_panel(rng, n_units=100, T=18,
                            cohorts=(6, 12, None),
                            true_att_by_cohort={6: 0.5, 12: 2.0})
    res = callaway_santanna(df, n_boot=30)
    att_gt = res.att_gt
    # Cohort-6 ATTs: should average ≈ 0.5
    g6 = att_gt[(att_gt["g"] == 6) & (att_gt["event_time"] >= 0)]["att"].mean()
    g12 = att_gt[(att_gt["g"] == 12) & (att_gt["event_time"] >= 0)]["att"].mean()
    assert abs(g6 - 0.5) < 0.5
    assert abs(g12 - 2.0) < 0.5


def test_callaway_santanna_event_study_columns(rng):
    df = _staggered_panel(rng, n_units=40, T=12, cohorts=(5, 8, None))
    res = callaway_santanna(df, n_boot=20)
    es = res.att_event_study
    expected_cols = {"event_time", "att", "se", "lo", "hi", "n_cohorts"}
    assert expected_cols.issubset(es.columns)


def test_callaway_santanna_result_is_frozen(rng):
    df = _staggered_panel(rng, n_units=40, T=12, cohorts=(5, 8, None))
    res = callaway_santanna(df, n_boot=10)
    assert isinstance(res, CallawaySantannaResult)
    with pytest.raises(Exception):
        res.att_overall = 99.0


# ---------------------------------------------------------------------------
# Sun-Abraham
# ---------------------------------------------------------------------------


def test_sun_abraham_recovers_share_weighted_att(rng):
    df = _staggered_panel(rng, n_units=120, T=18,
                            cohorts=(6, 12, None),
                            true_att_by_cohort={6: 0.5, 12: 2.0})
    res = sun_abraham(df, n_boot=20)
    assert isinstance(res, SunAbrahamResult)
    # The aggregated ATT should land between the two cohort effects.
    assert 0.3 < res.att_overall < 2.3


def test_sun_abraham_event_study_n_cohorts(rng):
    df = _staggered_panel(rng, n_units=40, T=12, cohorts=(5, 8, None))
    res = sun_abraham(df, n_boot=15)
    assert (res.att_event_study["n_cohorts"] >= 1).all()


def test_sun_abraham_result_is_frozen(rng):
    df = _staggered_panel(rng, n_units=30, T=10, cohorts=(4, 7, None))
    res = sun_abraham(df, n_boot=10)
    assert isinstance(res, SunAbrahamResult)
    with pytest.raises(Exception):
        res.att_overall = 0.0


# ---------------------------------------------------------------------------
# Borusyak-Jaravel-Spiess
# ---------------------------------------------------------------------------


def test_bjs_recovers_homogeneous_att(rng):
    df = _staggered_panel(rng, n_units=80, T=15,
                            cohorts=(5, 10, None),
                            true_att_by_cohort={5: 1.0, 10: 1.0})
    res = borusyak_jaravel_spiess(df, n_boot=20)
    assert isinstance(res, BorusyakJaravelSpiessResult)
    assert abs(res.att_overall - 1.0) < 0.5


def test_bjs_event_study_centred_on_zero_pre_treatment(rng):
    """Pre-treatment event times should have a zero ATT under correct
    parallel trends, but this test sees only the *post* event times
    because BJS evaluates τ̂ on treated cells."""
    df = _staggered_panel(rng, n_units=60, T=14, cohorts=(5, 9, None))
    res = borusyak_jaravel_spiess(df, n_boot=15)
    # All event times in the BJS output are ≥ 0 (treated cells only).
    assert (res.att_event_study["event_time"] >= 0).all()


def test_bjs_result_is_frozen(rng):
    df = _staggered_panel(rng, n_units=30, T=10, cohorts=(4, 7, None))
    res = borusyak_jaravel_spiess(df, n_boot=10)
    assert isinstance(res, BorusyakJaravelSpiessResult)
    with pytest.raises(Exception):
        res.att_overall = 0.0


def _exact_additive_panel(specs, T=10, seed=0, tau=2.0, trend=0.0):
    """Zero-noise Y = α_i + λ_t + τ·D_it, so the two-way FE model the BJS
    imputation fits is *exactly* correct and every τ̂_{i,t} must equal τ."""
    rng = np.random.default_rng(seed)
    alpha = rng.standard_normal(len(specs))
    lam = rng.standard_normal(T) * 0.7 + trend * np.arange(T)
    rows = []
    for i, g in enumerate(specs):
        for t in range(T):
            d = 0.0 if g is None else float(t >= g)
            rows.append({"unit": f"u{i:03d}", "time": float(t),
                         "y": alpha[i] + lam[t] + tau * d,
                         "treat_time": np.nan if g is None else float(g)})
    return pd.DataFrame(rows)


def test_bjs_drops_cells_whose_fixed_effect_is_not_identified():
    """A unit treated in every period has no α_i, and a period in which
    everyone is treated has no λ_t.  Imputing those as zero is not a small
    bias — it is an arbitrary number folded into the reported ATTs."""
    # (a) one always-treated unit alongside two ordinary cohorts.
    df = _exact_additive_panel([None] * 60 + [5] * 60 + [8] * 50 + [0] * 1)
    with pytest.warns(UserWarning, match="no untreated observation"):
        res = borusyak_jaravel_spiess(df, n_boot=0, unidentified="drop")
    assert np.abs(res.tau_it["tau"] - 2.0).max() < 1e-8
    assert abs(res.att_overall - 2.0) < 1e-8
    # e = 5..9 exist only for the always-treated unit, which is dropped.
    assert list(res.att_event_study["event_time"]) == [0, 1, 2, 3, 4]

    # (b) no never-treated cohort at all: t = 8, 9 have no untreated cell.
    df = _exact_additive_panel([5] * 30 + [8] * 30, T=10)
    with pytest.warns(UserWarning, match="no untreated observation"):
        res = borusyak_jaravel_spiess(df, n_boot=0, unidentified="drop")
    assert np.abs(res.tau_it["tau"] - 2.0).max() < 1e-8
    assert abs(res.att_overall - 2.0) < 1e-8


def test_bjs_two_way_imputation_is_exact_on_a_rotating_panel():
    """The untreated design here is connected but poorly so — each control
    unit is seen for three consecutive periods only — which is where a
    fixed-sweep backfitting solver stops far short of its own tolerance."""
    rows = []
    rng = np.random.default_rng(3)
    k = 0
    T, W, gap = 60, 3, 1000.0
    for start in range(T - W + 1):
        for _ in range(2):
            a = rng.standard_normal()
            for t in range(start, start + W):
                rows.append({"unit": f"r{k:04d}", "time": float(t),
                             "y": a + gap * t, "treat_time": np.nan})
            k += 1
    for j in range(3):
        a = rng.standard_normal()
        for t in range(T):
            rows.append({"unit": f"c{j}", "time": float(t),
                         "y": a + gap * t + 2.0 * float(t >= T - 5),
                         "treat_time": float(T - 5)})
    res = borusyak_jaravel_spiess(pd.DataFrame(rows), n_boot=0)
    assert np.abs(res.tau_it["tau"] - 2.0).max() < 1e-6
    assert abs(res.att_overall - 2.0) < 1e-6


def test_bjs_is_invariant_to_rescaling_the_time_axis(rng):
    """``time`` enters only through grouping and the ``t ≥ g`` comparison,
    so rescaling time (and cohorts) by 0.25 — a quarterly grid written as
    fractional years — must leave every number in the output unchanged and
    merely relabel ``event_time``."""
    df = _staggered_panel(rng, n_units=40, T=12, cohorts=(4, 7, None))
    scaled = df.copy()
    scaled["time"] = scaled["time"] * 0.25
    scaled["treat_time"] = scaled["treat_time"] * 0.25

    a = borusyak_jaravel_spiess(df, n_boot=60, seed=1).att_event_study
    b = borusyak_jaravel_spiess(scaled, n_boot=60, seed=1).att_event_study

    assert len(b) == len(a)
    assert b["event_time"].is_unique
    np.testing.assert_allclose(b["event_time"].astype(float),
                               a["event_time"].astype(float) * 0.25)
    for col in ("att", "se", "lo", "hi", "n_obs"):
        np.testing.assert_allclose(b[col], a[col], rtol=1e-10, atol=1e-12)
    # Each row's band must bracket its own point estimate.
    assert (b["lo"] <= b["att"]).all() and (b["att"] <= b["hi"]).all()


# ---------------------------------------------------------------------------
# Synthetic-DiD
# ---------------------------------------------------------------------------


def _single_cohort_panel(rng, *, n_units=20, T=20, treat_time=10,
                            n_treated=3, true_att=1.5, sigma=0.4):
    units = [f"u{i:02d}" for i in range(n_units)]
    treated = set(units[:n_treated])
    rows = []
    alpha = rng.standard_normal(n_units)
    lam = rng.standard_normal(T) * 0.3
    for i, u in enumerate(units):
        is_treated = u in treated
        for t in range(T):
            treat = 1.0 if (is_treated and t >= treat_time) else 0.0
            y = alpha[i] + lam[t] + true_att * treat + sigma * rng.standard_normal()
            rows.append({
                "unit": u, "time": t, "y": y,
                "treat_time": float(treat_time) if is_treated else np.nan,
            })
    return pd.DataFrame(rows)


def test_synthetic_did_recovers_treatment_effect(rng):
    df = _single_cohort_panel(rng, n_units=40, T=24, treat_time=12,
                                n_treated=4, true_att=2.0)
    res = synthetic_did(df, n_boot=40)
    assert isinstance(res, SyntheticDiDResult)
    # Spot-check fields exist
    assert hasattr(res, "tau") and hasattr(res, "omega")
    assert hasattr(res, "lambda_w") and hasattr(res, "treatment_time")
    assert abs(res.tau - 2.0) < 1.0


def test_synthetic_did_weights_sum_to_one(rng):
    df = _single_cohort_panel(rng, n_units=20, T=18, treat_time=10)
    res = synthetic_did(df, n_boot=10)
    np.testing.assert_allclose(float(res.omega.sum()), 1.0, atol=1e-6)
    np.testing.assert_allclose(float(res.lambda_w.sum()), 1.0, atol=1e-6)


def test_synthetic_did_rejects_staggered_panel(rng):
    """SDID requires a single common treatment time."""
    df = _staggered_panel(rng, n_units=20, T=12, cohorts=(5, 8, None))
    with pytest.raises(ValueError, match="single common treatment time"):
        synthetic_did(df, n_boot=10)


def test_synthetic_did_result_is_frozen(rng):
    df = _single_cohort_panel(rng, n_units=15, T=14, treat_time=8)
    res = synthetic_did(df, n_boot=5)
    assert isinstance(res, SyntheticDiDResult)
    with pytest.raises(Exception):
        res.tau = 0.0


# ---------------------------------------------------------------------------
# summary() smoke tests
# ---------------------------------------------------------------------------


def test_callaway_santanna_summary_smoke(rng):
    df = _staggered_panel(rng, n_units=30, T=10, cohorts=(4, 7, None))
    res = callaway_santanna(df, n_boot=5)
    s = res.summary()
    assert "Callaway-Sant" in s


def test_synthetic_did_summary_smoke(rng):
    df = _single_cohort_panel(rng, n_units=15, T=12, treat_time=6)
    res = synthetic_did(df, n_boot=5)
    s = res.summary()
    assert "Synthetic-DiD" in s


def test_synthetic_did_lambda_w_field_name():
    """Field is named ``lambda_w`` (not ``lambda``) since ``lambda`` is
    a Python reserved keyword."""
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(SyntheticDiDResult)}
    assert "lambda_w" in field_names
    assert "lambda" not in field_names


# ---------------------------------------------------------------------------
# Synthetic-DiD — invariances the estimand has by construction
# ---------------------------------------------------------------------------


def test_synthetic_did_is_scale_equivariant(rng):
    """τ̂(c·y) must equal c·τ̂(y) exactly.

    The QP is min ‖Aω − b‖² + ζ‖ω‖² on the simplex; y → c·y sends
    (A, b) → (cA, cb) and ζ → c²ζ (it is built from the variance of donor
    first differences), so the objective is multiplied by c² on an unchanged
    feasible set and the minimiser is IDENTICAL. τ̂ is then linear in y.
    Regression test for an absolute SLSQP ``ftol`` on a y²-scaled objective,
    which used to declare failure and substitute uniform weights.
    """
    df = _single_cohort_panel(rng, n_units=40, T=24, treat_time=12, n_treated=4)
    base = synthetic_did(df, n_boot=0).tau
    for c in (100.0, 10_000.0, 1e6):
        scaled = df.copy()
        scaled["y"] = scaled["y"] * c
        assert abs(synthetic_did(scaled, n_boot=0).tau / c - base) < 1e-8 * max(
            abs(base), 1.0)


def test_synthetic_did_invariant_to_treated_level_shift(rng):
    """Adding a constant to the treated units in EVERY period is a pure unit
    fixed effect, so the ATT is unchanged by construction.

    AAHI's free intercepts ω₀ / λ₀ (eq. 2.6-2.8) deliver this exactly: ω₀
    absorbs the shift, λ uses donors only, and τ̂ moves by c − c·Σₜλ̂ₜ = 0.
    Without them ω has to chase the treated *level* and τ̂ drifts.
    """
    df = _single_cohort_panel(rng, n_units=25, T=20, treat_time=10, n_treated=3)
    base = synthetic_did(df, n_boot=0).tau
    treated = df["treat_time"].notna()
    for c in (5.0, 50.0):
        shifted = df.copy()
        shifted.loc[treated, "y"] = shifted.loc[treated, "y"] + c
        assert abs(synthetic_did(shifted, n_boot=0).tau - base) < 1e-8


def test_synthetic_did_time_weights_balance_a_common_factor():
    """With one common factor whose post-period level lies strictly inside its
    pre-period range, the exactly-balancing λ is feasible on the simplex and
    SDID is exact for ANY ω — that is what the time weights are for.

    y_it = 5 + f_t·μ_i + τ·D_it with no noise, f rising 0→1 over the pre
    period and flat at 0.80 after. τ̂ = τ + (μ_tr − ω·μ)(0.80 − Σₜ λ̂ₜ f_t),
    so τ̂ = 1 iff λ̂ balances the factor. Over-penalising λ flattens it toward
    uniform, where Σₜ λ̂ₜ f_t = mean(f_pre) = 0.50 and τ̂ is biased upward.
    """
    T_pre, T_post, J, n_tr, tau = 8, 40, 15, 40, 1.0
    f = np.concatenate([np.linspace(0.0, 1.0, T_pre), np.full(T_post, 0.80)])
    mu = np.concatenate([np.full(n_tr, 3.0), np.linspace(0.5, 2.0, J)])
    rows = []
    for i in range(n_tr + J):
        is_treated = i < n_tr
        for t in range(T_pre + T_post):
            treat = 1.0 if (is_treated and t >= T_pre) else 0.0
            rows.append({
                "unit": f"u{i:03d}", "time": t,
                "y": 5.0 + f[t] * mu[i] + tau * treat,
                "treat_time": float(T_pre) if is_treated else np.nan,
            })
    res = synthetic_did(pd.DataFrame(rows), n_boot=0)
    assert abs(float((res.lambda_w.values * f[:T_pre]).sum()) - 0.80) < 1e-6
    assert abs(res.tau - tau) < 1e-6


def test_synthetic_did_se_reflects_treated_block_variability():
    """τ̂ − τ depends on the idiosyncratic errors of the treated block as well
    as the donors', so the bootstrap must resample treated units too.

    Here every donor is an exact copy of the same series, so a donor-only
    resample reproduces the identical panel every draw and reports se = 0
    with a degenerate interval — while the four treated units plainly differ
    from one another and τ̂ is anything but certain.
    """
    gen = np.random.default_rng(3)
    T, treat_time = 12, 6
    g = gen.standard_normal(T) * 0.5
    rows = []
    for j in range(4):                      # four *identical* donors
        for t in range(T):
            rows.append({"unit": f"d{j}", "time": t, "y": g[t],
                         "treat_time": np.nan})
    for i in range(4):                      # four *different* treated units
        a = float(gen.standard_normal())
        e = gen.standard_normal(T) * 0.5
        for t in range(T):
            rows.append({"unit": f"t{i}", "time": t,
                         "y": a + g[t] + e[t] + (2.0 if t >= treat_time else 0.0),
                         "treat_time": float(treat_time)})
    res = synthetic_did(pd.DataFrame(rows), n_boot=200, seed=1)
    assert res.se > 1e-3
    assert res.hi - res.lo > 1e-3


def test_synthetic_did_rejects_unbalanced_panel(rng):
    """An unbalanced panel pivots to NaN cells. Refuse rather than average the
    treated block over a changing unit set (silently biased τ̂) or hand a NaN
    τ̂ back with a finite-looking SE and CI."""
    df = _single_cohort_panel(rng, n_units=20, T=16, treat_time=8, n_treated=3)
    gappy = df[~((df["unit"] == "u00") & (df["time"] < 3))]
    with pytest.raises(ValueError, match="balanced"):
        synthetic_did(gappy, n_boot=0)
    gappy_donor = df[~((df["unit"] == "u19") & (df["time"] < 3))]
    with pytest.raises(ValueError, match="balanced"):
        synthetic_did(gappy_donor, n_boot=10)
