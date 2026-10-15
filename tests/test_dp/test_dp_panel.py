"""dp.Panel: the container for simulated individual histories.

Built here from hand-made arrays, so the tests do not depend on a solver.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import puremacro.dp
from puremacro import _wstats
from puremacro.dp import ModelSpecError, Panel
from puremacro.fetch.micro import MicroFrame, SurveyDesign

N, T = 4, 3


def _arrays():
    a = np.arange(N * T, dtype=float).reshape(N, T) - 2.0     # includes negatives
    z = np.tile(np.array([[0.5], [1.0], [1.5], [2.0]]), (1, T))
    return {"a": a, "z": z}


def _infinite():
    return Panel(_arrays(), age=None, alive=None, source="test model")


def _life_cycle():
    alive = np.ones((N, T), dtype=bool)
    alive[0, 1:] = False                    # agent 0 dies after age 20
    alive[2, 2] = False                     # agent 2 dies after age 21
    arrays = _arrays()
    for v in arrays.values():
        v[~alive] = np.nan
    return Panel(arrays, age=np.array([20, 21, 22]), alive=alive, source="test model")


def test_panel_exported_from_dp():
    assert "Panel" in puremacro.dp.__all__
    assert puremacro.dp.Panel is Panel


def test_properties_and_repr():
    p = _infinite()
    assert p.names == ["a", "z"]
    assert (p.n_agents, p.periods) == (N, T)
    assert p.age is None and p.alive is None
    np.testing.assert_array_equal(p["a"], _arrays()["a"])
    assert repr(p) == "Panel(n_agents=4, periods=3, columns=['a', 'z'])"
    with pytest.raises(KeyError):
        p["c"]


def test_to_frame_shape_and_columns():
    p = _infinite()
    df = p.to_frame()
    assert list(df.columns) == ["id", "t", "a", "z"]
    assert len(df) == N * T
    # agent-major order: agent i's history is contiguous
    np.testing.assert_array_equal(df["id"].to_numpy(), np.repeat(np.arange(N), T))
    np.testing.assert_array_equal(df["t"].to_numpy(), np.tile(np.arange(T), N))
    np.testing.assert_array_equal(df["a"].to_numpy(), p["a"].ravel())
    assert list(p.to_frame(columns=["z"]).columns) == ["id", "t", "z"]

    lc = _life_cycle()
    df = lc.to_frame()
    assert list(df.columns) == ["id", "t", "age", "alive", "a", "z"]
    assert len(df) == N * T                 # dead rows are kept
    np.testing.assert_array_equal(df["age"].to_numpy(), np.tile([20, 21, 22], N))
    assert df["alive"].dtype == bool and int((~df["alive"]).sum()) == 3


def test_to_microframe_drops_dead_rows():
    lc = _life_cycle()
    mf = lc.to_microframe()
    assert isinstance(mf, MicroFrame)
    assert len(mf) == N * T - 3
    assert mf.design == SurveyDesign.weights_only("_w")
    assert (mf.data["_w"] == 1.0).all()
    assert mf.unit == "agent" and mf.source == "test model" and mf.vintage == "simulated"
    assert not mf.data["a"].isna().any()
    assert {"id", "t", "age", "a", "z"} <= set(mf.data.columns)

    one = lc.to_microframe(t=2)
    assert len(one) == 2                    # agents 1 and 3 are alive at t = 2
    assert sorted(one.data["id"]) == [1, 3]
    assert list(lc.to_microframe(columns=["a"]).data.columns).count("z") == 0

    full = _infinite().to_microframe()
    assert len(full) == N * T


def test_gini_by_t_matches_microframe():
    p = _infinite()
    g = p.gini("a", by="t")
    assert isinstance(g, pd.Series) and g.index.name == "t"
    mf = p.to_microframe()
    expected = mf.gini("a", by="t").set_index("t")["estimate"]
    pd.testing.assert_series_equal(g, expected, check_names=False)
    # and directly from the shared helper, one cross-section at a time
    for t in range(T):
        x = p["a"][:, t]
        assert g[t] == pytest.approx(_wstats.w_gini(x, np.ones((N, 1)))[0], abs=1e-15)

    pooled = p.gini("a")
    assert isinstance(pooled, float)
    assert pooled == pytest.approx(mf.gini("a").iloc[0]["estimate"], abs=1e-15)


def test_statistics_take_expressions_and_by_age():
    lc = _life_cycle()
    mf = lc.to_microframe()
    d = mf.data
    alive_a = d["a"].to_numpy()
    assert lc.mean("a") == pytest.approx(alive_a.mean())
    assert lc.mean("a + 2*z") == pytest.approx((d["a"] + 2 * d["z"]).mean())
    assert lc.mean("z^2") == pytest.approx((d["z"] ** 2).mean())
    m = lc.mean("a", by="age")
    assert m.index.name == "age" and list(m.index) == [20, 21, 22]
    assert m[22] == pytest.approx(d.loc[d["age"] == 22, "a"].mean())
    q = lc.quantile("a", 0.5, by="age")
    assert q[20] == pytest.approx(mf.quantile("a", 0.5, by="age").iloc[0]["estimate"])
    ts = lc.top_share("a", 0.25)
    assert ts == pytest.approx(mf.top_share("a", 0.25).iloc[0]["estimate"])
    with pytest.raises(ValueError, match="by"):
        _infinite().mean("a", by="age")
    with pytest.raises(ValueError, match="by"):
        lc.mean("a", by="id")
    with pytest.raises(ModelSpecError, match="c"):
        lc.mean("c + a")


def test_cross_section_and_path():
    p = _infinite()
    cs = p.cross_section(1)
    assert list(cs.columns) == ["id", "t", "a", "z"]
    np.testing.assert_array_equal(cs["a"].to_numpy(), p["a"][:, 1])
    assert (cs["t"] == 1).all()
    path = p.path(2)
    assert list(path.columns) == ["id", "t", "a", "z"]
    np.testing.assert_array_equal(path["a"].to_numpy(), p["a"][2])
    np.testing.assert_array_equal(path["t"].to_numpy(), np.arange(T))
    with pytest.raises(IndexError):
        p.cross_section(T)
    with pytest.raises(IndexError):
        p.path(N)

    lc = _life_cycle()
    assert list(lc.path(0)["alive"]) == [True, False, False]
    assert list(lc.cross_section(2)["age"]) == [22] * N


def test_summary():
    p = _infinite()
    s = p.summary()
    assert list(s.index) == ["a", "z"] and list(s.columns) == ["mean", "std", "gini"]
    last = p["a"][:, -1]
    assert s.loc["a", "mean"] == pytest.approx(last.mean())
    assert s.loc["a", "std"] == pytest.approx(last.std())
    assert s.loc["a", "gini"] == pytest.approx(p.gini("a", by="t").iloc[-1])

    lc = _life_cycle()
    s = lc.summary()
    assert s.index.names == ["variable", "age"] and len(s) == 2 * T
    alive_last = lc["a"][lc.alive[:, 2], 2]
    assert s.loc[("a", 22), "mean"] == pytest.approx(alive_last.mean())


def test_constructor_validates_shapes():
    a = np.zeros((N, T))
    with pytest.raises(ValueError, match="shape"):
        Panel({"a": a, "b": np.zeros((N, T + 1))}, age=None, alive=None, source="x")
    with pytest.raises(ValueError, match="age"):
        Panel({"a": a}, age=np.arange(T + 1), alive=None, source="x")
    with pytest.raises(ValueError, match="alive"):
        Panel({"a": a}, age=None, alive=np.ones((N, T)), source="x")
    with pytest.raises(ValueError, match="reserved"):
        Panel({"t": a}, age=None, alive=None, source="x")
    with pytest.raises(ValueError, match="at least one"):
        Panel({}, age=None, alive=None, source="x")
