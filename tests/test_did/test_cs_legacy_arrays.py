"""Regression tests for the legacy four-array form of ``callaway_santanna``.

The array entry point is a separate implementation from the DataFrame one, so
it is checked here against invariances the estimand holds by construction:

* on a noiseless two-way-FE panel with a constant effect, every ATT(g, t) is
  exactly ``tau`` after treatment and exactly ``0`` before it — whatever rows
  the panel happens to be missing, since a unit without both endpoints of the
  long difference simply does not enter the cell;
* the two entry points estimate the same thing, so their ATT(g, t) point
  estimates must agree cell by cell for the same ``control`` group.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.did import callaway_santanna

# Non-linear common time path: a flat one would hide a control group whose
# long difference is mismeasured, since then every long difference is zero.
_D = {1: 0.0, 2: 1.0, 3: 1.7, 4: 2.1, 5: 2.3, 6: 2.4}
_TAU = 2.5
_COHORTS = (np.inf, 3.0, 5.0)


def _exact_panel(n_units=30, *, drop=()):
    """Noiseless y = a_i + d_t + tau*1{t >= g}; ``drop`` = (unit, time) holes."""
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 3.0, n_units)
    y, g, t, u = [], [], [], []
    for i in range(n_units):
        gi = _COHORTS[i % 3]
        for tt in _D:
            if (float(i), float(tt)) in drop:
                continue
            y.append(a[i] + _D[tt] + _TAU * (tt >= gi))
            g.append(gi)
            t.append(float(tt))
            u.append(float(i))
    return (np.array(y), np.array(g), np.array(t), np.array(u))


def _mcar_holes(n_units, p, seed):
    rng = np.random.default_rng(seed)
    return {(float(i), float(tt)) for i in range(n_units) for tt in _D
            if rng.random() < p}


def test_legacy_arrays_exact_on_balanced_panel():
    """Baseline: the invariance below is not vacuous."""
    out = callaway_santanna(*_exact_panel())
    assert out["overall_att"] == pytest.approx(_TAU, abs=1e-12)


@pytest.mark.parametrize("n_units, p, seed", [(30, 0.10, 1), (60, 0.25, 2)])
def test_legacy_arrays_unaffected_by_missing_rows(n_units, p, seed):
    """Holes must drop units from cells, not enter them with dy = 0.

    Units preallocated at dy = 0 used to be counted as controls, which biased
    the overall ATT by +0.25 at 10 % missing and +0.48 at 25 % — a bias that
    did not shrink with N — and fabricated nonzero pre-treatment cells.
    """
    holes = _mcar_holes(n_units, p, seed)
    assert holes, "the hole pattern must not be empty"
    out = callaway_santanna(*_exact_panel(n_units, drop=holes))

    assert out["overall_att"] == pytest.approx(_TAU, abs=1e-12)

    groups, periods, A = out["groups"], out["periods"], out["att_gt"]
    for gi, g in enumerate(groups):
        for ti, t in enumerate(periods):
            if np.isnan(A[gi, ti]):
                continue
            want = _TAU if t >= g else 0.0
            assert A[gi, ti] == pytest.approx(want, abs=1e-12), (g, t)


def test_legacy_arrays_single_hole_closed_form():
    """One missing control row, one cell, an answer worked out by hand.

    Dropping never-treated unit 0's t = 4 row leaves ATT(3, 4) = tau. The
    phantom-zero control made it (d4 - d2 + tau) - (9/10)*(d4 - d2) = 2.61.
    """
    out = callaway_santanna(*_exact_panel(30, drop={(0.0, 4.0)}))
    gi = int(np.flatnonzero(out["groups"] == 3.0)[0])
    ti = int(np.flatnonzero(out["periods"] == 4.0)[0])
    assert out["att_gt"][gi, ti] == pytest.approx(_TAU, abs=1e-12)


# ---------------------------------------------------------------------------
# control= on the legacy path
# ---------------------------------------------------------------------------


def _control_panel(sigma=0.0, seed=3):
    """Cohorts 3 and 5 with exact parallel trends; the never-treated group
    carries a jump at t >= 4, so the two control choices disagree."""
    rng = np.random.default_rng(seed)
    y, g, t, u = [], [], [], []
    for i in range(30):
        gi = _COHORTS[i % 3]
        a = rng.normal(0.0, 2.0)
        for tt in _D:
            jump = 12.0 if (np.isinf(gi) and tt >= 4) else 0.0
            y.append(a + _D[tt] + _TAU * (tt >= gi) + jump
                     + sigma * rng.standard_normal())
            g.append(gi)
            t.append(float(tt))
            u.append(float(i))
    return (np.array(y), np.array(g), np.array(t), np.array(u))


def _df_of(y, g, t, u):
    return pd.DataFrame({"unit": u, "time": t, "y": y,
                         "treat_time": np.where(np.isinf(g), np.nan, g)})


@pytest.mark.parametrize("control", ["never_treated", "not_yet_treated"])
def test_legacy_arrays_match_dataframe_path(control):
    """Same panel, same control group, same ATT(g, t) — the point estimates
    do not depend on the entry point (the bootstrap SEs do; hence n_boot=0)."""
    y, g, t, u = _control_panel(sigma=0.4)
    arr = callaway_santanna(y, g, t, u, control=control)
    res = callaway_santanna(_df_of(y, g, t, u), control=control, n_boot=0)

    cells = {(row.g, row.t): row.att for row in res.att_gt.itertuples()}
    seen = 0
    for gi, gg in enumerate(arr["groups"]):
        for ti, tt in enumerate(arr["periods"]):
            if (gg, tt) not in cells or np.isnan(arr["att_gt"][gi, ti]):
                continue
            assert arr["att_gt"][gi, ti] == pytest.approx(cells[(gg, tt)],
                                                          abs=1e-10), (gg, tt)
            seen += 1
    assert seen >= 8


def test_legacy_arrays_control_choice_changes_the_answer():
    """The keyword used to be accepted and thrown away, so both calls
    returned the never-treated estimand."""
    y, g, t, u = _control_panel()
    never = callaway_santanna(y, g, t, u, control="never_treated")
    notyet = callaway_santanna(y, g, t, u, control="not_yet_treated")
    assert abs(never["overall_att"] - notyet["overall_att"]) > 0.5


def test_legacy_arrays_reject_unknown_control():
    y, g, t, u = _control_panel()
    with pytest.raises(ValueError, match="control must be one of"):
        callaway_santanna(y, g, t, u, control="nearest_neighbour")


# ---------------------------------------------------------------------------
# nothing identified
# ---------------------------------------------------------------------------


def _no_never_treated_panel(n_units=20, T=5, seed=4):
    rng = np.random.default_rng(seed)
    y, g, t, u = [], [], [], []
    lam = rng.normal(0.0, 1.0, T + 1)
    for i in range(n_units):
        gi = 3.0 if i % 2 == 0 else 4.0
        a = rng.normal()
        for tt in range(1, T + 1):
            y.append(a + lam[tt] + 1.5 * (tt >= gi) + 0.2 * rng.standard_normal())
            g.append(gi)
            t.append(float(tt))
            u.append(float(i))
    return (np.array(y), np.array(g), np.array(t), np.array(u))


def test_legacy_arrays_overall_att_is_nan_when_no_cell_is_identified():
    """No never-treated units, so no post cell is estimable: the mean of an
    empty set is NaN, not a confident 0.0. The DataFrame path raises here."""
    y, g, t, u = _no_never_treated_panel()
    out = callaway_santanna(y, g, t, u)
    # Only the two base-period normalisations survive; nothing is estimated.
    assert np.count_nonzero(np.isfinite(out["att_gt"])) == 2
    assert np.isnan(out["overall_att"])

    with pytest.raises(ValueError, match="never-treated"):
        callaway_santanna(_df_of(y, g, t, u), n_boot=0)
