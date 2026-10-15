"""Exact-distribution statistics on DPSolution (gini, lorenz, top_share, quantile).

They share one weighted-statistics helper with the survey estimators, so a
MicroFrame built from the grid and the distribution gives the same numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro import _wstats
from puremacro.fetch.micro import MicroFrame, SurveyDesign

LC_T, LC_SURVIVAL = 10, 0.9           # the lc_sol fixture in conftest.py


def _grid_frame(sol, expr="a"):
    df = pd.DataFrame({"x": sol.policy(expr).ravel(), "w": sol.distribution.ravel()})
    return MicroFrame(df, SurveyDesign.weights_only("w"), unit="node", source="model", vintage="grid")


def test_two_point_gini_closed_form():
    # mass p at 0 and 1 - p at 1: the Gini is p
    for p in (0.1, 0.3, 0.5, 0.8):
        x = np.array([0.0, 1.0])
        W = np.array([[p], [1 - p]])
        assert _wstats.w_gini(x, W)[0] == pytest.approx(p, abs=1e-15)


def test_gini_matches_microframe_on_the_grid(aiyagari_sol):
    sol = aiyagari_sol
    mf = _grid_frame(sol)
    g = sol.gini("a")
    assert isinstance(g, float) and 0.2 < g < 0.9
    assert g == pytest.approx(mf.gini("x").iloc[0, 0], abs=1e-12)
    assert sol.top_share("a", 0.1) == pytest.approx(mf.top_share("x", 0.1).iloc[0, 0], abs=1e-12)
    assert sol.quantile("a", 0.5) == pytest.approx(mf.quantile("x", 0.5).iloc[0, 0], abs=1e-12)
    assert sol.gini("c") == pytest.approx(_grid_frame(sol, "c").gini("x").iloc[0, 0], abs=1e-12)


def test_negative_wealth_gini_matches_microframe(borrowing_sol):
    sol = borrowing_sol
    a = sol.policy("a")
    assert (sol.distribution[a[:, 0] < 0].sum()) > 0.01     # real mass in debt
    mf = _grid_frame(sol)
    assert sol.gini("a") == pytest.approx(mf.gini("x").iloc[0, 0], abs=1e-12)
    assert sol.top_share("a", 0.01) == pytest.approx(mf.top_share("x", 0.01).iloc[0, 0], abs=1e-12)


def test_lorenz_curve_is_consistent_with_top_share_and_gini(aiyagari_sol):
    sol = aiyagari_sol
    lz = sol.lorenz("a", n=1001)
    assert list(lz.columns) == ["pop_share", "value_share"] and len(lz) == 1001
    assert lz.iloc[0].tolist() == [0.0, 0.0]
    assert lz.iloc[-1].tolist() == pytest.approx([1.0, 1.0], abs=1e-12)
    assert np.all(np.diff(lz["value_share"]) >= -1e-15)
    assert np.all(lz["value_share"] <= lz["pop_share"] + 1e-12)
    # 1 - L(0.9) is the top-10% share, exactly, since the curve is piecewise linear
    assert 1 - lz.loc[900, "value_share"] == pytest.approx(sol.top_share("a", 0.1), abs=1e-12)
    # the Gini is twice the area between the diagonal and the curve
    x, y = lz["pop_share"].to_numpy(), lz["value_share"].to_numpy()
    area = float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2))  # trapezoid rule, any NumPy
    assert 1 - 2 * area == pytest.approx(sol.gini("a"), abs=2e-3)


def test_quantile_vector_and_bounds(aiyagari_sol):
    sol = aiyagari_sol
    qs = sol.quantile("a", [0.1, 0.5, 0.9])
    assert isinstance(qs, np.ndarray) and qs.shape == (3,)
    assert np.all(np.diff(qs) >= 0)
    assert qs[1] == sol.quantile("a", 0.5)
    assert sol.quantile("a", [0.0, 1.0]).tolist() == [sol.grids["a"][0], sol.grids["a"][-1]]
    with pytest.raises(ValueError, match="q"):
        sol.quantile("a", 1.5)
    with pytest.raises(ValueError, match="q"):
        sol.quantile("a", np.nan)
    with pytest.raises(ValueError, match="q"):
        sol.quantile("a", [0.5, np.nan])
    with pytest.raises(ValueError, match="p"):
        sol.top_share("a", 0.0)
    with pytest.raises(ValueError, match="age"):
        sol.gini("a", age=0)


def test_lifecycle_population_shares_follow_survival(lc_sol):
    s = lc_sol.population_shares()
    assert s.shape == (LC_T,)
    assert s[1] / s[0] == pytest.approx(LC_SURVIVAL)
    assert s[-1] / s[0] == pytest.approx(LC_SURVIVAL ** (LC_T - 1))
    assert s.sum() == pytest.approx(1.0)
    # sol.distribution itself is untouched: every cohort keeps mass 1
    np.testing.assert_allclose(lc_sol.distribution.sum(axis=(1, 2)), 1.0, atol=1e-12)


def test_lifecycle_pooled_gini_uses_population_weights(lc_sol):
    sol = lc_sol
    s = sol.population_shares()
    x = sol.policy("a").ravel()
    w = (s[:, None, None] * sol.distribution).ravel()
    hand = _wstats.w_gini(x, w[:, None])[0]
    assert sol.gini("a") == pytest.approx(hand, abs=1e-12)
    # equal cohort weights would give a different answer
    flat = _wstats.w_gini(x, sol.distribution.ravel()[:, None])[0]
    assert abs(sol.gini("a") - flat) > 1e-4
    # one cohort, and every cohort at once
    j = 5
    one = _wstats.w_gini(sol.policy("a")[j].ravel(), sol.distribution[j].ravel()[:, None])[0]
    assert sol.gini("a", age=j) == pytest.approx(one, abs=1e-12)
    by_age = sol.gini("a", age="all")
    assert isinstance(by_age, pd.Series) and by_age.index.name == "age"
    assert list(by_age.index) == list(range(LC_T))
    assert by_age[j] == pytest.approx(one, abs=1e-12)
    assert np.isnan(by_age[0])            # newborns all hold a[0] = 0: total wealth is zero
    assert isinstance(sol.top_share("a", 0.1, age="all"), pd.Series)
    assert sol.lorenz("a", n=11, age="all").shape == (11 * LC_T, 3)
    q = sol.quantile("a", [0.5, 0.9], age="all")
    assert q.shape == (LC_T, 2) and q.loc[j, 0.9] == sol.quantile("a", 0.9, age=j)


def test_quantile_ignores_nodes_without_mass(lc_sol):
    # at age 3 only the two lowest asset nodes hold mass; the most negative value
    # of -a sits on an empty node and is not the 0-quantile
    a = lc_sol.grids["a"]
    assert np.count_nonzero(lc_sol.distribution[3].sum(axis=1)) == 2
    assert lc_sol.quantile("-a", 0.0, age=3) == -a[1]


def test_population_shares_are_life_cycle_only(aiyagari_sol):
    with pytest.raises(ValueError, match="life-cycle"):
        aiyagari_sol.population_shares()


def test_hjb_gini_works_and_ks_raises(hjb_sol, ks_sol):
    mf = _grid_frame(hjb_sol)
    assert hjb_sol.gini("a") == pytest.approx(mf.gini("x").iloc[0, 0], abs=1e-12)
    assert 0 < hjb_sol.gini("a") < 1
    assert hjb_sol.lorenz("a").shape == (101, 2)
    for call in (lambda: ks_sol.gini("a"), lambda: ks_sol.lorenz("a"),
                 lambda: ks_sol.top_share("a"), lambda: ks_sol.quantile("a")):
        with pytest.raises(ValueError, match="no stationary distribution: Krusell-Smith solutions have none"):
            call()


def test_survey_estimators_unchanged():
    # values computed with the estimators before the helpers moved to puremacro._wstats
    df = pd.DataFrame({
        "a": [3.005, 4.195, 1.903, -0.562, 1.181, -0.967, 3.241, 8.361, 1.031, 0.518, 4.959, 4.428],
        "w": [1.137, 1.613, 1.761, 1.884, 2.989, 2.482, 2.055, 2.972, 1.038, 0.901, 2.031, 0.61],
        "g": ["x", "y"] * 6})
    mf = MicroFrame(df, SurveyDesign.weights_only("w"), unit="agent", source="model", vintage="sim")
    assert mf.gini("a").iloc[0, 0] == pytest.approx(0.582281886028963, abs=1e-15)
    assert mf.top_share("a", 0.1).iloc[0, 0] == pytest.approx(0.30212483055350203, abs=1e-15)
    assert mf.quantile("a", 0.5).iloc[0, 0] == 1.903
    np.testing.assert_allclose(mf.gini("a", by="g")["estimate"], [0.301168095423457, 0.7038675644593598],
                               rtol=0, atol=1e-15)
    np.testing.assert_allclose(mf.top_share("a", 0.25, by="g")["estimate"],
                               [0.44166916458728056, 0.6981218875823747], rtol=0, atol=1e-15)
