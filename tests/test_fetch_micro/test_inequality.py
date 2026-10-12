"""Gini, top-share and ratio estimators against brute-force formulas."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.fetch.micro import MicroFrame, SurveyDesign, replicate_variance


def _gini_pairs(x, w):
    """Mean absolute difference over all pairs: the Gini's definition."""
    return (np.abs(x[:, None] - x[None, :]) * np.outer(w, w)).sum() / (2 * w.sum() * (w @ x))


def _top_share_expanded(x, w, p):
    """Top share on a sample where unit weights are integers: expand each
    unit into w_i identical people and take the richest p of them."""
    people = np.sort(np.repeat(x, w.astype(int)))[::-1]
    k = p * people.size
    assert k == int(k)
    return people[: int(k)].sum() / people.sum()


def _frame(n=300, R=40, seed=0, implicates=1):
    rng = np.random.default_rng(seed)
    w = rng.uniform(1, 9, n)
    reps = w[:, None] * rng.uniform(0.5, 1.5, (n, R))
    design = SurveyDesign(weight="w", replicate_weights=tuple(f"r{k}" for k in range(R)),
                          method="bootstrap", scale=1 / (R - 1), mse=False,
                          implicate="imp" if implicates > 1 else None)
    frames = []
    for i in range(1, implicates + 1):
        f = pd.DataFrame(reps, columns=list(design.replicate_weights))
        f["w"] = w
        f["nw"] = rng.lognormal(11, 1.6, n) - 20_000.0     # some negative net worth
        f["inc"] = rng.lognormal(10.8, 0.7, n)
        if implicates > 1:
            f["imp"] = i
        frames.append(f)
    return MicroFrame(pd.concat(frames, ignore_index=True), design, "household", "t", "x")


def test_gini_equals_the_mean_absolute_difference():
    mf = _frame()
    d = mf.data
    res = mf.gini("nw")
    assert res.loc[0, "estimate"] == pytest.approx(_gini_pairs(d["nw"].to_numpy(), d["w"].to_numpy()),
                                                   rel=1e-12)


def test_gini_is_invariant_to_tie_order_and_scale():
    x = np.array([3.0, 1.0, 3.0, 0.0, 7.0])
    w = np.array([1.0, 2.0, 1.0, 1.0, 0.5])
    df = pd.DataFrame({"x": x, "w": w})
    df2 = df.iloc[[2, 4, 0, 3, 1]].assign(x=lambda f: 5 * f["x"])
    g1 = MicroFrame(df, SurveyDesign.weights_only("w"), "u", "t", "x").gini("x")
    g2 = MicroFrame(df2, SurveyDesign.weights_only("w"), "u", "t", "x").gini("x")
    assert g1.loc[0, "estimate"] == pytest.approx(g2.loc[0, "estimate"], rel=1e-12)
    assert g1.loc[0, "estimate"] == pytest.approx(_gini_pairs(x, w), rel=1e-12)


def test_equal_values_have_zero_gini_and_proportional_top_shares():
    df = pd.DataFrame({"x": np.full(10, 4.0), "w": np.arange(1.0, 11.0)})
    mf = MicroFrame(df, SurveyDesign.weights_only("w"), "u", "t", "x")
    assert mf.gini("x").loc[0, "estimate"] == pytest.approx(0.0, abs=1e-14)
    assert mf.top_share("x", 0.1).loc[0, "estimate"] == pytest.approx(0.1)


@pytest.mark.parametrize("p", [0.5, 0.1, 0.01])
def test_top_share_splits_the_unit_at_the_cut_off(p):
    rng = np.random.default_rng(1)
    x = rng.lognormal(0, 1.5, 37)
    w = rng.integers(1, 9, 37).astype(float)
    w[0] += 400 - w.sum()                       # 400 people: p*400 is whole
    assert w.min() >= 1
    df = pd.DataFrame({"x": x, "w": w})
    res = MicroFrame(df, SurveyDesign.weights_only("w"), "u", "t", "x").top_share("x", p)
    assert res.loc[0, "estimate"] == pytest.approx(_top_share_expanded(x, w, p), rel=1e-12)


def test_ratio_of_means_and_of_medians():
    mf = _frame()
    d = mf.data
    w = d["w"].to_numpy()
    r = mf.ratio("nw", "inc")
    assert r.loc[0, "estimate"] == pytest.approx((w @ d["nw"]) / (w @ d["inc"]))
    med = mf.ratio("nw", "inc", q=0.5).loc[0, "estimate"]
    assert med == pytest.approx(mf.quantile("nw", 0.5).loc[0, "estimate"]
                                / mf.quantile("inc", 0.5).loc[0, "estimate"])


def test_ratio_drops_rows_missing_either_column():
    mf = _frame()
    d = mf.data.copy()
    d.loc[:9, "inc"] = np.nan
    mf2 = MicroFrame(d, mf.design, "household", "t", "x")
    w = d["w"].to_numpy()[10:]
    expect = (w @ d["nw"].to_numpy()[10:]) / (w @ d["inc"].to_numpy()[10:])
    res = mf2.ratio("nw", "inc")
    assert res.loc[0, "estimate"] == pytest.approx(expect)
    assert res.loc[0, "n"] == len(d) - 10


def test_replicate_se_is_the_spread_of_the_replicate_statistics():
    mf = _frame()
    d = mf.data
    x, R = d["nw"].to_numpy(), 40
    theta = _gini_pairs(x, d["w"].to_numpy())
    reps = np.array([_gini_pairs(x, d[f"r{k}"].to_numpy()) for k in range(R)])
    se = np.sqrt(replicate_variance(theta, reps, scale=1 / (R - 1), mse=False))
    res = mf.gini("nw")
    assert res.loc[0, "se"] == pytest.approx(se, rel=1e-10)
    top = mf.top_share("nw", 0.1)
    assert 0 < top.loc[0, "se"] < top.loc[0, "estimate"]


def test_implicates_and_groups_flow_through():
    mf = _frame(implicates=5)
    d = mf.data
    ests = [_gini_pairs(g["nw"].to_numpy(), g["w"].to_numpy()) for _, g in d.groupby("imp")]
    res = mf.gini("nw")
    assert res.loc[0, "estimate"] == pytest.approx(np.mean(ests), rel=1e-12)
    assert res.loc[0, "implicates"] == 5
    d = d.assign(grp=np.where(d["inc"] > d["inc"].median(), "hi", "lo"))
    by = MicroFrame(d, mf.design, "household", "t", "x").top_share("nw", 0.1, by="grp")
    assert list(by["grp"]) == ["hi", "lo"] and by["se"].notna().all()


def test_argument_checks_and_taylor_designs():
    mf = _frame()
    with pytest.raises(ValueError):
        mf.top_share("nw", 1.0)
    with pytest.raises(ValueError):
        mf.ratio("nw", "inc", q=1.5)
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "w": 1.0, "h": [1, 1, 2, 2], "u": [1, 2, 3, 4]})
    tay = MicroFrame(df, SurveyDesign(weight="w", method="taylor", strata="h", psu="u"),
                     "u", "t", "x")
    with pytest.raises(NotImplementedError):
        tay.gini("x")
    assert tay.mean("x").loc[0, "estimate"] == pytest.approx(2.5)


def test_nonpositive_total_gives_nan_not_a_number():
    df = pd.DataFrame({"x": [-5.0, 1.0, 2.0], "w": 1.0})
    mf = MicroFrame(df, SurveyDesign.weights_only("w"), "u", "t", "x")
    assert np.isnan(mf.gini("x").loc[0, "estimate"])
    assert np.isnan(mf.top_share("x", 0.5).loc[0, "estimate"])
