"""Coverage tests for puremacro.causal.synthetic_control.

The subject of most of this file is the **indexing convention**:
``treat_time`` is the 0-indexed row of the first treated period, the
pre-window is ``[0, treat_time)`` and the post-window
``[treat_time, T)``. Those two intervals must partition the sample.

Until 1.3.2 the port carried MATLAB's ``T_pre = treat_time - 1`` into
0-indexed Python, which orphaned row ``treat_time - 1``: it fitted no
weight and scored no effect. ``TestPreWindowPartition`` pins that down
with a DGP where the orphaned row is the *only* informative one, so the
old behaviour is not a small numeric drift but a different answer.

Tests are organized by concern:
- Pre-window partition / the orphaned row (regression)
- Degenerate short pre-windows (treat_time = 1)
- Simplex constraints on the weights
- Recovery of a known constant post-treatment effect
- Shapes, keys, and pass-through
"""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.causal.synthetic_control import synthetic_control


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _orphan_panel(treat_time: int, n_post: int = 3) -> np.ndarray:
    """Panel whose only informative pre-row is ``treat_time - 1``.

    Two donors. On rows ``0 .. treat_time - 2`` both donors read 1.0, so
    every convex weight fits those rows exactly and they identify
    nothing. Row ``treat_time - 1`` — the one the pre-1.3.2 code dropped
    — reads ``(0.0, 10.0)`` against a treated value of 3.0, which pins
    ``w = (0.7, 0.3)`` uniquely on the simplex.

    Column 0 is the treated unit; columns 1 and 2 are the donors.
    """
    T = treat_time + n_post
    d1 = np.ones(T)
    d2 = np.ones(T)
    d1[treat_time - 1] = 0.0
    d2[treat_time - 1] = 10.0

    y = np.ones(T)
    y[treat_time - 1] = 0.7 * 0.0 + 0.3 * 10.0  # == 3.0

    return np.column_stack([y, d1, d2])


def _exact_panel(
    T: int = 30,
    treat_time: int = 20,
    n_donors: int = 4,
    tau: float = 2.5,
    seed: int = 11,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Treated unit is an exact convex combination of donors, plus ``tau``
    from ``treat_time`` on. Returns ``(Y, true_weights, tau)``."""
    rng = np.random.default_rng(seed)
    donors = rng.normal(size=(T, n_donors))
    w_true = np.array([0.5, 0.2, 0.2, 0.1])[:n_donors]
    w_true = w_true / w_true.sum()

    y = donors @ w_true
    y[treat_time:] += tau

    return np.column_stack([y, donors]), w_true, tau


# ===========================================================================
# The pre-window / post-window partition
# ===========================================================================

class TestPreWindowPartition:
    """``[0, treat_time)`` and ``[treat_time, T)`` cover every row once."""

    @pytest.mark.parametrize("treat_time", [2, 3, 6, 12])
    def test_orphaned_row_is_fitted(self, treat_time):
        """Row ``treat_time - 1`` must reach the objective.

        Fails against the pre-1.3.2 ``T_pre = treat_time - 1``, which
        never saw the discriminating row and so returned the SLSQP
        starting point (0.5, 0.5) untouched.
        """
        Y = _orphan_panel(treat_time)
        w = synthetic_control(Y, treated_idx=0, treat_time=treat_time)["weights"]

        np.testing.assert_allclose(w, [0.7, 0.3], atol=1e-6)
        # And explicitly: not the uninformative starting point.
        assert abs(w[0] - 0.5) > 0.1, f"weights look like w0, got {w}"

    def test_pre_window_length_is_treat_time(self):
        """Behavioural probe: perturbing row ``treat_time - 1`` moves the
        weights, and perturbing row ``treat_time`` does not.

        This brackets the boundary from both sides without reaching into
        the function's internals.
        """
        treat_time = 8
        Y_base = _orphan_panel(treat_time)
        w_base = synthetic_control(Y_base, treated_idx=0, treat_time=treat_time)["weights"]

        Y_last_pre = Y_base.copy()
        Y_last_pre[treat_time - 1, 0] += 4.0  # last PRE row -> must matter
        w_last_pre = synthetic_control(Y_last_pre, treated_idx=0, treat_time=treat_time)["weights"]
        assert not np.allclose(w_base, w_last_pre, atol=1e-4), (
            "row treat_time - 1 is in the pre-window and must move the fit"
        )

        Y_first_post = Y_base.copy()
        Y_first_post[treat_time, 0] += 4.0  # first POST row -> must not
        w_first_post = synthetic_control(Y_first_post, treated_idx=0, treat_time=treat_time)["weights"]
        np.testing.assert_allclose(w_base, w_first_post, atol=1e-6)

    def test_partition_is_complete_no_gap(self):
        """Pre-length + post-length == T, measured behaviourally.

        The post-window length is read off ``att_post``: shifting one
        post row by ``delta`` moves the mean by ``delta / n_post``. The
        pre-window length is read off the count of rows that move the
        weights. The two must sum to ``T``.
        """
        T, treat_time = 14, 9
        rng = np.random.default_rng(3)
        Y = rng.normal(size=(T, 4))

        base = synthetic_control(Y, treated_idx=0, treat_time=treat_time)
        delta = 6.0

        Y_bump = Y.copy()
        Y_bump[T - 1, 0] += delta
        bumped = synthetic_control(Y_bump, treated_idx=0, treat_time=treat_time)
        n_post = delta / (bumped["att_post"] - base["att_post"])
        assert n_post == pytest.approx(T - treat_time, abs=1e-6)

        n_pre = 0
        for row in range(T):
            Y_r = Y.copy()
            Y_r[row, 0] += delta
            w_r = synthetic_control(Y_r, treated_idx=0, treat_time=treat_time)["weights"]
            if not np.allclose(w_r, base["weights"], atol=1e-5):
                n_pre += 1

        assert n_pre + n_post == pytest.approx(T), (
            f"pre={n_pre} + post={n_post} != T={T}; a row is in neither window"
        )

    def test_att_path_covers_every_row(self):
        T, treat_time = 12, 5
        rng = np.random.default_rng(4)
        Y = rng.normal(size=(T, 3))
        res = synthetic_control(Y, treated_idx=0, treat_time=treat_time)

        assert res["att_path"].shape == (T,)
        np.testing.assert_allclose(
            res["att_post"], np.mean(res["att_path"][treat_time:]), rtol=1e-12
        )


# ===========================================================================
# Degenerate short pre-windows
# ===========================================================================

class TestShortPreWindow:
    def test_treat_time_one_fits_one_row(self):
        """``treat_time = 1`` leaves exactly one pre-treatment row.

        The pre-1.3.2 code fitted on zero rows: the objective was
        identically zero for every ``w``, SLSQP returned its starting
        point ``w0`` after no useful work, and the caller got a
        meaningless answer with no warning. One row does not identify
        three weights either, but it does constrain them — the fit must
        reproduce that row exactly.
        """
        Y = np.array(
            [
                [3.0, 1.0, 5.0, 9.0],  # the single pre row
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        res = synthetic_control(Y, treated_idx=0, treat_time=1)
        w = res["weights"]

        n_donors = 3
        w0 = np.ones(n_donors) / n_donors
        assert not np.allclose(w, w0, atol=1e-8), (
            f"optimizer returned its starting point untouched: {w}"
        )
        # y_pre = 3.0 is inside the donor hull [1, 9], so it is matched exactly.
        assert res["y_synthetic"][0] == pytest.approx(3.0, abs=1e-6)

    def test_treat_time_two_uses_both_rows(self):
        Y = _orphan_panel(2)
        w = synthetic_control(Y, treated_idx=0, treat_time=2)["weights"]
        np.testing.assert_allclose(w, [0.7, 0.3], atol=1e-6)


# ===========================================================================
# Simplex constraints
# ===========================================================================

class TestWeightConstraints:
    @pytest.mark.parametrize("treat_time", [1, 2, 5, 20])
    def test_weights_on_simplex(self, treat_time):
        rng = np.random.default_rng(treat_time)
        Y = rng.normal(size=(25, 6))
        w = synthetic_control(Y, treated_idx=0, treat_time=treat_time)["weights"]

        assert w.shape == (5,)
        assert np.all(w >= -1e-9), f"negative weight: {w}"
        assert np.sum(w) == pytest.approx(1.0, abs=1e-6)

    def test_weights_stay_on_simplex_with_extrapolating_target(self):
        """Treated unit far outside the donor hull: the fit cannot match
        it, and must not escape the simplex trying."""
        rng = np.random.default_rng(99)
        donors = rng.normal(size=(20, 4))
        Y = np.column_stack([donors[:, 0] * 50.0, donors])
        w = synthetic_control(Y, treated_idx=0, treat_time=15)["weights"]

        assert np.all(w >= -1e-9)
        assert np.sum(w) == pytest.approx(1.0, abs=1e-6)


# ===========================================================================
# Effect recovery
# ===========================================================================

class TestEffectRecovery:
    def test_recovers_known_constant_effect(self):
        Y, w_true, tau = _exact_panel()
        res = synthetic_control(Y, treated_idx=0, treat_time=20)

        # SLSQP's default ftol leaves ~1e-4 on the weights; the point of
        # this test is recovery, not the optimizer's last digit.
        np.testing.assert_allclose(res["weights"], w_true, atol=1e-3)
        assert res["att_post"] == pytest.approx(tau, abs=1e-3)
        # Pre-treatment fit is exact by construction, including the last
        # pre row, which the pre-1.3.2 code excluded from the objective.
        np.testing.assert_allclose(res["att_path"][:20], 0.0, atol=1e-3)

    def test_zero_effect_reads_as_zero(self):
        Y, _, _ = _exact_panel(tau=0.0)
        res = synthetic_control(Y, treated_idx=0, treat_time=20)
        assert res["att_post"] == pytest.approx(0.0, abs=1e-3)


# ===========================================================================
# Contract: keys, shapes, pass-through
# ===========================================================================

class TestReturnContract:
    def test_six_keys(self):
        rng = np.random.default_rng(0)
        res = synthetic_control(rng.normal(size=(10, 4)), treated_idx=0, treat_time=6)
        assert set(res) == {
            "weights",
            "y_treated",
            "y_synthetic",
            "att_path",
            "att_post",
            "treat_time",
        }

    def test_shapes_and_types(self):
        rng = np.random.default_rng(1)
        T, N = 10, 4
        res = synthetic_control(rng.normal(size=(T, N)), treated_idx=0, treat_time=6)

        assert res["weights"].shape == (N - 1,)
        assert res["y_treated"].shape == (T,)
        assert res["y_synthetic"].shape == (T,)
        assert res["att_path"].shape == (T,)
        assert isinstance(res["att_post"], float)
        assert res["treat_time"] == 6

    def test_treated_column_excluded_from_donors(self):
        """Treating a middle column must not let the unit explain itself."""
        rng = np.random.default_rng(5)
        Y = rng.normal(size=(15, 5))
        res = synthetic_control(Y, treated_idx=2, treat_time=10)

        assert res["weights"].shape == (4,)
        np.testing.assert_allclose(res["y_treated"], Y[:, 2])

    def test_accepts_list_input(self):
        Y = [[1.0, 1.0, 2.0], [2.0, 1.0, 3.0], [9.0, 1.0, 2.0]]
        res = synthetic_control(Y, treated_idx=0, treat_time=2)
        assert res["y_treated"].dtype == np.float64
