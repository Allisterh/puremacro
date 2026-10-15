"""DPSolution.simulate for life-cycle (finite-horizon) solutions."""
from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.stats import binomtest

from puremacro import dp
from puremacro.dp import Panel

T = 60


def _life_cycle(survival: float, T: int = T, n_a: int = 120):
    """The life-cycle model of docs/dp.md with a constant survival probability."""
    lc = dp.Model("life cycle")
    lc.parameters(beta=0.96, gamma=2.0, r=0.03, T_ret=45)
    lc.exogenous("z", dp.AR1(rho=0.95, sigma=0.1, n=5))
    lc.state("a", np.linspace(0.0, 30.0, n_a))
    lc.local("y = (age < T_ret)*exp(z + 0.04*age - 0.0008*age^2) + (age >= T_ret)*0.6")
    lc.local("c = (1 + r)*a + y - a(+1)")
    lc.reward("crra(c, gamma)")
    lc.subject_to("c > 0")
    lc.horizon(T, survival=np.full(T, survival))
    return lc


@pytest.fixture(scope="module")
def lc_vfi():
    return _life_cycle(1.0).solve("vfi")


@pytest.fixture(scope="module")
def lc_egm():
    return _life_cycle(1.0).solve("egm")


@pytest.mark.parametrize("which", ["lc_vfi", "lc_egm"])
def test_lifecycle_mean_by_age_matches_cohort_profile(which, request):
    sol = request.getfixturevalue(which)
    n = 20_000
    panel = sol.simulate(n, seed=1)
    assert panel.periods == T and np.array_equal(panel.age, np.arange(T))
    sim = panel.mean("a", by="age")
    exact = sol.mean("a")
    assert list(sim.index) == list(range(T))
    se = panel["a"].std(axis=0) / np.sqrt(n)
    assert np.all(np.abs(sim.to_numpy() - exact) <= 3 * se + 1e-9)
    # newborns: the lowest asset node, shocks from the stationary distribution
    assert np.all(panel["a"][:, 0] == sol.grids["a"][0])


def test_survival_frequencies():
    sol = _life_cycle(0.9, T=30, n_a=60).solve("egm")
    n = 5000
    panel = sol.simulate(n, seed=3)
    alive = panel.alive
    assert alive.dtype == bool and alive.shape == (n, 30)
    assert alive[:, 0].all()
    # once dead, always dead
    assert not np.any(~alive[:, :-1] & alive[:, 1:])
    for j in range(30):
        k = int(alive[:, j].sum())
        assert binomtest(k, n, 0.9 ** j).pvalue > 0.001, j
    # dead agents keep stepping, so the arrays stay finite
    assert np.isfinite(panel["a"]).all() and np.isfinite(panel["c"]).all()


def test_everyone_dead_gives_nan_not_error():
    sol = _life_cycle(0.5, T=40, n_a=60).solve("egm")
    panel = sol.simulate(500, seed=0)
    assert not panel.alive[:, -1].any()
    m = panel.mean("a", by="age")
    assert list(m.index) == list(range(40))
    assert np.isfinite(m.iloc[0]) and np.isnan(m.iloc[-1])
    g = panel.gini("c", by="age")
    assert len(g) == 40 and np.isnan(g.iloc[-1])
    q = panel.quantile("a", 0.5, by="age")
    assert len(q) == 40 and np.isnan(q.iloc[-1])
    s = panel.summary()
    assert np.isnan(s.loc[("a", 39), "mean"])
    assert np.isfinite(panel.mean("a"))


def test_periods_argument_warns_in_lifecycle(lc_egm):
    with pytest.warns(UserWarning, match="periods"):
        p = lc_egm.simulate(10, 200, seed=0)
    assert p.periods == T
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert lc_egm.simulate(10, seed=0).periods == T
        assert lc_egm.simulate(10, T, seed=0).periods == T


def test_lifecycle_vfi_and_egm_both_simulate(lc_vfi, lc_egm):
    for sol in (lc_vfi, lc_egm):
        p1 = sol.simulate(300, seed=5, record=["a", "z", "c", "y", "a(+1)"])
        p2 = sol.simulate(300, seed=5, record=["a", "z", "c", "y", "a(+1)"])
        assert isinstance(p1, Panel) and p1.periods == T
        for name in p1.names:
            assert p1[name].tobytes() == p2[name].tobytes()
        assert np.array_equal(p1["a"][:, 1:], p1["a(+1)"][:, :-1])
        assert (p1["c"] > 0).all()
        # the age-dependent local sees each period's age
        ret = p1["y"][:, 45:]
        assert np.allclose(ret, 0.6)
        work = p1["y"][:, :45]
        expected = np.exp(p1["z"][:, :45] + 0.04 * np.arange(45) - 0.0008 * np.arange(45) ** 2)
        np.testing.assert_allclose(work, expected, rtol=1e-12)
        # assets peak at retirement, as the exact profile does
        assert int(p1.mean("a", by="age").to_numpy().argmax()) in (44, 45, 46)
    vfi_grid = lc_vfi.grids["a"]
    assert np.isin(lc_vfi.simulate(50, seed=0)["a"], vfi_grid).all()


def test_age_varying_parameter_reaches_record():
    T_ = 8
    m = dp.Model("age varying")
    m.parameters(beta=0.96, gamma=2.0, r=np.linspace(0.01, 0.04, T_))
    m.exogenous("z", dp.AR1(rho=0.9, sigma=0.1, n=3))
    m.state("a", np.linspace(0.0, 10.0, 40))
    m.local("c = (1 + r)*a + exp(z) - a(+1)")
    m.reward("crra(c, gamma)")
    m.subject_to("c > 0")
    m.horizon(T_)
    sol = m.solve("egm")
    p = sol.simulate(20, seed=0, record=["a", "r", "c"])
    np.testing.assert_allclose(p["r"], np.broadcast_to(np.linspace(0.01, 0.04, T_), (20, T_)))
    assert p.alive is None


def test_declared_newborns_and_init_rules(lc_egm):
    a = lc_egm.grids["a"]
    nb = np.zeros((a.size, 5))
    nb[10, 2] = 3.0                          # normalised: every newborn at node 10, z index 2
    m = _life_cycle(1.0, T=5, n_a=a.size)
    m.horizon(5, newborns=nb)
    sol = m.solve("egm")
    p = sol.simulate(50, seed=0)
    assert np.all(p["a"][:, 0] == a[10])
    assert np.all(p["z"][:, 0] == sol.grids["z"][2])
    # a dict fixes the age-0 states
    p2 = lc_egm.simulate(20, seed=0, init={"a": 2.0})
    assert np.all(p2["a"][:, 0] == 2.0)
    with pytest.raises(ValueError, match="newborn"):
        lc_egm.simulate(20, init="stationary")
    with pytest.raises(ValueError, match="burn_in"):
        lc_egm.simulate(20, burn_in=3)


def test_newborn_rejected_for_infinite_horizon(aiyagari_sol):
    with pytest.raises(ValueError, match="stationary"):
        aiyagari_sol.simulate(10, 5, init="newborn")
