"""Singular GMM weight matrices in ab_gmm / bb_gmm.

When the step-1 matrix ``Z'HZ`` (or ``Z'Z``) or the step-2 matrix
``sum_i Z_i'u_i u_i'Z_i`` is singular, xtabond2 does not stop: it warns and
uses a generalized inverse (Roodman 2009, Stata Journal 9(1), p. 98 and
footnote 6). ``ab_gmm`` and ``bb_gmm`` used to raise ``LinAlgError`` instead,
for example on an unbalanced panel with staggered start dates and
uncollapsed instruments (the GMM columns of the last period are linearly
dependent) and on system GMM with year dummies (singular step-2 matrix).

Oracle for the generalized-inverse path: the GMM estimator, its covariance
and the Hansen statistic are invariant to adding an instrument column that
is an exact linear combination of the others when the weight is the
Moore-Penrose inverse (if ``Z_aug = Z T'`` with ``T`` of full column rank,
``T'(T A T')^+ T = A^{-1}``). So the fit with a redundant column must equal
the fit with that column removed by hand, where every weight matrix is
nonsingular and the ordinary inverse applies.
"""
from __future__ import annotations

import importlib
import warnings

import numpy as np
import pytest

from puremacro._linalg import safe_cholesky
from puremacro.dynpanel import ab_gmm, bb_gmm

from .conftest import simulate_dynamic_panel

ab_mod = importlib.import_module("puremacro.dynpanel.ab_gmm")
bb_mod = importlib.import_module("puremacro.dynpanel.bb_gmm")


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _dummy_panel():
    """Balanced panel with an exogenous regressor and one year dummy."""
    d = simulate_dynamic_panel(N=200, T=7, rho=0.5, beta_exog=0.5, seed=1)
    dummy = (d["time_id"] == 4).astype(float)
    X = np.column_stack([d["x"][:, 0], dummy])
    return d["y"], d["panel_id"], d["time_id"], X


def _staggered_panel():
    """Unbalanced panel with staggered start dates (T_i 6-9, offsets 0-2).

    With uncollapsed instruments the latest period has more GMM-lag
    columns than observations, so the instrument matrix loses rank.
    """
    rng = np.random.default_rng(0)
    ids, ts, ys, xs = [], [], [], []
    for i in range(80):
        T = rng.integers(6, 10)
        t0 = rng.integers(0, 3)
        y = 0.0
        for t in range(T):
            x = rng.normal()
            y = 0.5 * y + 0.3 * x + rng.normal() + (0.5 if i % 2 else 0.0)
            ids.append(i)
            ts.append(t0 + t)
            ys.append(y)
            xs.append(x)
    return np.array(ys), np.array(ids), np.array(ts), np.array(xs)[:, None]


def _independent_columns(Z: np.ndarray) -> list:
    """Greedy left-to-right choice of linearly independent columns."""
    keep: list = []
    for j in range(Z.shape[1]):
        trial = keep + [j]
        if np.linalg.matrix_rank(Z[:, trial]) == len(trial):
            keep = trial
    return keep


def _patch_ab(monkeypatch, edit):
    real = ab_mod.build_instruments

    def wrapper(*args, **kwargs):
        bundle = dict(real(*args, **kwargs))
        bundle["Z"], bundle["instr_labels"] = edit(
            bundle["Z"], list(bundle["instr_labels"])
        )
        return bundle

    monkeypatch.setattr(ab_mod, "build_instruments", wrapper)


def _patch_bb(monkeypatch, edit):
    real = bb_mod._build_system

    def wrapper(*args, **kwargs):
        system = dict(real(*args, **kwargs))
        system["Z"], system["instr_labels"] = edit(
            system["Z"], list(system["instr_labels"])
        )
        return system

    monkeypatch.setattr(bb_mod, "_build_system", wrapper)


def _duplicate(label):
    def edit(Z, labels):
        j = labels.index(label)
        return np.column_stack([Z, Z[:, j]]), labels + [label + "_copy"]

    return edit


def _drop_dependent(Z, labels):
    keep = _independent_columns(Z)
    return Z[:, keep], [labels[j] for j in keep]


def _fit(f, y, p, t, X, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return f(y, p, t, X_exog=X, **kw)


def _assert_same_fit(a, b, rtol=1e-7):
    np.testing.assert_allclose(a.coefs, b.coefs, rtol=rtol, atol=1e-10)
    np.testing.assert_allclose(a.se, b.se, rtol=rtol, atol=1e-10)
    np.testing.assert_allclose(a.cov, b.cov, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(a.hansen_j, b.hansen_j, rtol=1e-6, atol=1e-9)
    assert a.hansen_j_df == b.hansen_j_df
    assert a.n_instruments == b.n_instruments
    if np.isfinite(b.hansen_j_p):
        np.testing.assert_allclose(a.hansen_j_p, b.hansen_j_p, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(a.ar1_p, b.ar1_p, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(a.ar2_p, b.ar2_p, rtol=1e-6, atol=1e-9)


SPECS = [
    dict(two_step=True, windmeijer=True),
    dict(two_step=True, windmeijer=False),
    dict(two_step=False),
]


# ---------------------------------------------------------------------
# weight-matrix helper
# ---------------------------------------------------------------------


def test_full_rank_weight_is_the_legacy_cholesky_inverse_bit_for_bit():
    rng = np.random.default_rng(5)
    G = rng.normal(size=(60, 9))
    A = G.T @ G
    W, rank = ab_mod._inv_weight(A, name="test")
    # the pre-fix computation, inlined
    L = safe_cholesky(0.5 * (A + A.T), name="test")
    L_inv = np.linalg.solve(L, np.eye(9))
    assert rank == 9
    assert np.array_equal(W, L_inv.T @ L_inv)


def test_singular_weight_is_moore_penrose_inverse():
    rng = np.random.default_rng(6)
    G = rng.normal(size=(40, 6))
    G = np.column_stack([G, G[:, 1] - 2.0 * G[:, 4]])  # exact dependence
    A = G.T @ G
    W, rank = ab_mod._inv_weight(A, name="test")
    assert rank == 6
    np.testing.assert_allclose(W, np.linalg.pinv(A, hermitian=True), rtol=1e-8, atol=1e-12)
    # Penrose conditions
    np.testing.assert_allclose(A @ W @ A, A, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(W @ A @ W, W, rtol=1e-8, atol=1e-12)


# ---------------------------------------------------------------------
# full-rank fits: unchanged, no notes
# ---------------------------------------------------------------------


@pytest.mark.parametrize("f", [ab_gmm, bb_gmm])
def test_full_rank_fit_has_no_rank_note_and_full_instrument_count(f):
    y, p, t, X = _dummy_panel()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        r = f(y, p, t, X_exog=X, collapse=True)
    assert not any("rank" in n for n in r.notes)
    k = r.coefs.size
    assert r.hansen_j_df == r.n_instruments - k


# ---------------------------------------------------------------------
# oracle: a duplicated dummy instrument changes nothing
# ---------------------------------------------------------------------


@pytest.mark.parametrize("collapse", [True, False])
@pytest.mark.parametrize("spec", SPECS, ids=["2s-wc", "2s", "1s"])
@pytest.mark.parametrize(
    "f, patch", [(ab_gmm, _patch_ab), (bb_gmm, _patch_bb)], ids=["ab", "bb"]
)
def test_duplicated_dummy_instrument_equals_fit_without_it(
    monkeypatch, f, patch, spec, collapse
):
    y, p, t, X = _dummy_panel()
    base = _fit(f, y, p, t, X, collapse=collapse, **spec)
    assert not base.notes

    patch(monkeypatch, _duplicate("Xx1_diff"))
    with pytest.warns(RuntimeWarning, match="generalized inverse"):
        aug = f(y, p, t, X_exog=X, collapse=collapse, **spec)

    _assert_same_fit(aug, base)
    # xtabond2 counts instruments and Hansen df after collinear columns
    # are removed (e(j) vs e(j0), see the module docstring of ab_gmm).
    assert aug.n_instruments == base.n_instruments
    assert any("rank" in n and "generalized inverse" in n for n in aug.notes)


# ---------------------------------------------------------------------
# staggered unbalanced panel, uncollapsed: used to raise at step 1
# ---------------------------------------------------------------------


@pytest.mark.parametrize("lag", [1, 2])
@pytest.mark.parametrize("spec", SPECS, ids=["2s-wc", "2s", "1s"])
@pytest.mark.parametrize(
    "f, patch", [(ab_gmm, _patch_ab), (bb_gmm, _patch_bb)], ids=["ab", "bb"]
)
def test_staggered_panel_equals_fit_with_dependent_columns_dropped_by_hand(
    monkeypatch, f, patch, spec, lag
):
    y, p, t, X = _staggered_panel()
    with pytest.warns(RuntimeWarning, match="generalized inverse"):
        r = f(y, p, t, X_exog=X, lag_dep_var=lag, collapse=False, **spec)
    assert np.all(np.isfinite(r.coefs)) and np.all(np.isfinite(r.se))
    assert r.converged
    assert any("step-1" in n and "rank" in n for n in r.notes)

    patch(monkeypatch, _drop_dependent)
    ref = _fit(f, y, p, t, X, lag_dep_var=lag, collapse=False, **spec)
    assert not any("rank" in n for n in ref.notes)
    _assert_same_fit(r, ref, rtol=1e-6)


# ---------------------------------------------------------------------
# singular step-2 matrix with a full-rank instrument matrix
# ---------------------------------------------------------------------


def _year_dummy_panel():
    """System GMM with year dummies for the third and later periods.

    Each dummy is its own instrument in both equations, so at the one-step
    estimate the residuals of its period satisfy two exact linear
    restrictions and ``sum_i Z_i'u_i u_i'Z_i`` loses rank although ``Z``
    itself has full column rank (the same mechanism as system GMM with year
    dummies on Stata's abdata).
    """
    d = simulate_dynamic_panel(N=120, T=6, rho=0.5, beta_exog=0.5, seed=3)
    years = np.unique(d["time_id"])
    D = np.column_stack([(d["time_id"] == s).astype(float) for s in years[2:]])
    X = np.column_stack([d["x"][:, 0], D])
    return d["y"], d["panel_id"], d["time_id"], X


@pytest.mark.parametrize("collapse", [True, False])
def test_singular_step2_matrix_uses_pinv_and_full_instrument_count(collapse):
    y, p, t, X = _year_dummy_panel()
    system = bb_mod._build_system(
        y, p, t, lag_dep_var=1, X_endog=None, X_pred=None, X_exog=X,
        gmm_lag_window=(2, None), collapse=collapse,
    )
    Z = system["Z"]
    m = Z.shape[1]
    assert np.linalg.matrix_rank(Z) == m  # instruments are not collinear

    with pytest.warns(RuntimeWarning, match="generalized inverse"):
        r = bb_gmm(y, p, t, X_exog=X, collapse=collapse)
    k = r.coefs.size
    assert np.all(np.isfinite(r.coefs)) and np.all(np.isfinite(r.se))
    assert any("step-2" in n and "rank" in n for n in r.notes)
    # xtabond2 bases the Hansen df on the step-1 instrument count and does
    # not reduce it for a singular step-2 matrix.
    assert r.n_instruments == m
    assert r.hansen_j_df == m - k
