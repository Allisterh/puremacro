"""DPSolution.simulate for infinite-horizon VFI and EGM solutions."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from puremacro import dp
from puremacro.dp import ModelSpecError, Panel

N_AGENTS, PERIODS, BURN_IN = 20_000, 400, 200


def _aiyagari():
    """The Aiyagari economy of docs/dp.md."""
    m = dp.Model("aiyagari")
    m.parameters(beta=0.96, gamma=1.0, alpha=0.36, delta=0.08)
    m.prices(r=(0.005, 0.0396))
    m.exogenous("z", dp.AR1(rho=0.9, sigma=0.2, n=5))
    m.state("a", np.linspace(1e-4, 80.0, 150))
    m.local("w = (1 - alpha)*(alpha/(r + delta))^(alpha/(1 - alpha))")
    m.local("c = w*exp(z) + (1 + r)*a - a(+1)")
    m.reward("crra(c, gamma)")
    m.subject_to("c > 0")
    m.aggregate(K="a", L="exp(z)")
    m.clear("K - L*(alpha/(r + delta))^(1/(1 - alpha))")
    return m


@pytest.fixture(scope="module")
def aiyagari_vfi():
    return _aiyagari().solve("vfi")


@pytest.fixture(scope="module")
def aiyagari_egm():
    return _aiyagari().solve("egm")


def _log_cake(grid):
    """Log utility, no income, gross return R: c = (1-beta)*R*a, a(+1) = beta*R*a."""
    m = dp.Model("log cake")
    m.parameters(beta=0.96, R=1.03)
    m.state("a", grid)
    m.local("c = R*a - a(+1)")
    m.reward("log(c)")
    m.subject_to("c > 0")
    return m


def _check_stationary(sol):
    panel = sol.simulate(N_AGENTS, PERIODS, seed=1, burn_in=BURN_IN)
    assert isinstance(panel, Panel)
    assert panel.n_agents == N_AGENTS and panel.periods == PERIODS
    assert panel.names == ["a", "z", "c"]
    last = PERIODS - 1
    a = panel["a"][:, last]
    se = a.std() / np.sqrt(N_AGENTS)
    assert abs(a.mean() - sol.mean("a")) < 3 * se
    # Gini of the last cross-section, through the survey estimator
    g_last = panel.to_microframe("a", t=last).gini("a")["estimate"].iloc[0]
    assert abs(g_last - sol.gini("a")) < 0.01
    assert abs(panel.gini("a", by=None) - sol.gini("a")) < 0.01
    # Kolmogorov-Smirnov distance to the exact CDF. The exact distribution is a
    # lattice on the grid nodes, the mass at node k standing for the values within
    # half a cell of it, so the empirical CDF is read at the cell midpoints (for
    # VFI, whose states are nodes, this is the plain CDF at the nodes).
    grid = sol.grids["a"]
    exact = np.cumsum(sol.distribution.sum(axis=1))[:-1]
    mid = 0.5 * (grid[:-1] + grid[1:])
    emp = np.searchsorted(np.sort(a), mid, side="right") / N_AGENTS
    assert np.max(np.abs(emp - exact)) < 1.36 / np.sqrt(N_AGENTS) + 0.01


def test_vfi_panel_reproduces_stationary_distribution(aiyagari_vfi):
    _check_stationary(aiyagari_vfi)


def test_egm_panel_reproduces_stationary_distribution(aiyagari_egm):
    _check_stationary(aiyagari_egm)


def test_log_utility_closed_form_consumption():
    beta, R = 0.96, 1.03
    sol = _log_cake(np.linspace(1e-6, 100.0, 2000)).solve("egm", tol=1e-12)
    panel = sol.simulate(5, 50, init={"a": 50.0}, record=["a", "c", "a(+1)"])
    a, c = panel["a"], panel["c"]
    np.testing.assert_allclose(c, (1 - beta) * R * a, rtol=0, atol=1e-6)
    np.testing.assert_allclose(panel["a(+1)"], beta * R * a, rtol=0, atol=1e-6)
    np.testing.assert_allclose(a[:, -1], 50.0 * (beta * R) ** 49, rtol=1e-8)
    np.testing.assert_allclose(a[:, 1:], panel["a(+1)"][:, :-1], rtol=0, atol=0)


@pytest.mark.parametrize("which", ["aiyagari_vfi", "aiyagari_egm"])
def test_same_seed_bit_identical_and_seed_matters(which, request):
    sol = request.getfixturevalue(which)
    p1 = sol.simulate(500, 30, seed=7, burn_in=5)
    p2 = sol.simulate(500, 30, seed=7, burn_in=5)
    p3 = sol.simulate(500, 30, seed=8, burn_in=5)
    for name in p1.names:
        assert np.array_equal(p1[name], p2[name])
        assert p1[name].tobytes() == p2[name].tobytes()
    assert not np.array_equal(p1["a"], p3["a"])
    assert not np.array_equal(p1["z"], p3["z"])


def test_multi_asset_vfi_simulates():
    m = dp.Model("two asset")
    m.parameters(beta=0.96, gamma=2.0, rl=0.01, ri=0.05, kappa=0.05)
    m.exogenous("z", dp.AR1(rho=0.9, sigma=0.25, n=5))
    m.state("m", np.linspace(0.0, 15.0, 20))
    m.state("k", np.linspace(0.0, 15.0, 20))
    m.local("c = (1 + rl)*m + (1 + ri)*k + exp(z) - m(+1) - k(+1) - kappa*abs(k(+1) - k)")
    m.reward("crra(c, gamma)")
    m.subject_to("c > 0")
    sol = m.solve(tol=1e-9, howard=30)
    panel = sol.simulate(4000, 60, seed=3, record=["m", "k", "z", "c", "m(+1)", "k(+1)"])
    assert panel.names == ["m", "k", "z", "c", "m(+1)", "k(+1)"]
    gm, gk = sol.grids["m"], sol.grids["k"]
    assert np.isin(panel["m"], gm).all() and np.isin(panel["k"], gk).all()
    assert np.array_equal(panel["m"][:, 1:], panel["m(+1)"][:, :-1])
    assert np.array_equal(panel["k"][:, 1:], panel["k(+1)"][:, :-1])
    # each agent's step follows the index policy at its cell (C order, m slowest)
    mi = np.searchsorted(gm, panel["m"][:, 0])
    ki = np.searchsorted(gk, panel["k"][:, 0])
    zi = np.searchsorted(sol.grids["z"], panel["z"][:, 0])
    pm, pk = sol.raw.policy_components()
    flat = mi * gk.size + ki
    assert np.array_equal(panel["m(+1)"][:, 0], gm[pm[flat, zi]])
    assert np.array_equal(panel["k(+1)"][:, 0], gk[pk[flat, zi]])
    assert (panel["c"] > 0).all()
    # stationary from the start: mean holdings match the exact distribution
    for name in ("m", "c"):
        x = panel[name][:, -1]
        assert abs(x.mean() - sol.mean(name)) < 3 * x.std() / np.sqrt(panel.n_agents)


@pytest.mark.parametrize("method", ["vfi", "egm"])
def test_no_shock_model_simulates(method):
    sol = _log_cake(np.linspace(1e-6, 100.0, 300)).solve(method, tol=1e-10)
    assert sol._c.shock_names == []
    panel = sol.simulate(50, 20, seed=0)
    assert panel.names == ["a", "c"]
    frame = panel.to_frame()
    assert list(frame.columns) == ["id", "t", "a", "c"]
    assert len(frame) == 50 * 20
    assert np.isfinite(panel["a"]).all() and (panel["c"] > 0).all()
    # deterministic dynamics: every period after the first shrinks wealth
    assert (np.diff(panel["a"][panel["a"][:, 0] > 1.0], axis=1) <= 0).all()


def test_init_dict_off_grid(aiyagari_vfi, aiyagari_egm):
    grid = aiyagari_vfi.grids["a"]
    off = 0.5 * (grid[10] + grid[11]) + 0.1 * (grid[11] - grid[10])   # nearer grid[11]
    with pytest.warns(UserWarning, match="snapped to the nearest node") as rec:
        pv = aiyagari_vfi.simulate(10, 5, seed=0, init={"a": off})
    assert len([w for w in rec if issubclass(w.category, UserWarning)]) == 1
    assert np.all(pv["a"][:, 0] == grid[11])
    # on-grid values do not warn
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        aiyagari_vfi.simulate(10, 5, seed=0, init={"a": grid[3]})
        pe = aiyagari_egm.simulate(10, 5, seed=0, init={"a": off})
    assert np.all(pe["a"][:, 0] == off)
    # per-agent arrays and fixed shocks
    vals = np.linspace(1.0, 5.0, 10)
    zg = aiyagari_egm.grids["z"]
    pe2 = aiyagari_egm.simulate(10, 3, seed=0, init={"a": vals, "z": zg[2]})
    assert np.array_equal(pe2["a"][:, 0], vals) and np.all(pe2["z"][:, 0] == zg[2])
    with pytest.raises(ValueError, match="every state"):
        aiyagari_egm.simulate(10, 3, init={"z": zg[2]})
    with pytest.raises(ValueError, match="shock grid"):
        aiyagari_egm.simulate(10, 3, init={"a": 1.0, "z": 0.123})


@pytest.mark.parametrize("record, match", [
    (["a", "consumption"], "consumption"),
    (["z(+1)"], r"z\(\+1\)"),
])
def test_record_unknown_name_raises_before_simulating(aiyagari_egm, monkeypatch, record, match):
    from puremacro.dp import _simulate

    def boom(*args, **kwargs):
        raise AssertionError("simulation started before record was validated")

    monkeypatch.setattr(_simulate, "_initial_state", boom)
    with pytest.raises(ModelSpecError, match=match) as exc:
        aiyagari_egm.simulate(100, 10, record=record)
    assert "record" in str(exc.value)


def test_hjb_and_ks_simulate(hjb_sol, ks_sol):
    p_hjb = hjb_sol.simulate(n_agents=2000, periods=50, seed=11, burn_in=20)
    assert isinstance(p_hjb, Panel)
    assert p_hjb.names == ["a", "e", "c"]
    assert p_hjb["a"].shape == (2000, 50)
    assert (p_hjb["c"] > 0).all()
    assert abs(p_hjb["a"][:, -1].mean() - hjb_sol.mean("a")) < 0.25

    p_ks = ks_sol.simulate(n_agents=500, periods=30, seed=12, burn_in=5)
    assert isinstance(p_ks, Panel)
    assert p_ks.names == ["a", "e", "Z", "K", "c"]
    assert p_ks["a"].shape == (500, 30)
    assert (p_ks["c"] > 0).all()
    np.testing.assert_allclose(p_ks["K"][0, :], p_ks["a"].mean(axis=0), rtol=1e-12)


def test_size_guard(aiyagari_egm):
    with pytest.raises(ValueError, match=r"bytes") as exc:
        aiyagari_egm.simulate(1_000_000, 1_000)
    # three recorded arrays (a, z, c) of 8 bytes each
    assert f"{8 * 3 * 1_000_000 * 1_000:,}" in str(exc.value)
    aiyagari_egm.simulate(10, 5)                 # small requests are fine


@pytest.mark.parametrize("method", ["vfi", "egm"])
def test_stationary_init_needs_a_distribution(method):
    sol = _log_cake(np.linspace(1e-6, 100.0, 300)).solve(method, tol=1e-10, distribution=False)
    assert sol.distribution is None
    with pytest.raises(ValueError, match=r"init=\{"):
        sol.simulate(10, 5)
    panel = sol.simulate(10, 5, init={"a": sol.grids["a"][100]})
    assert panel["a"].shape == (10, 5)


def test_init_fixes_a_shock_on_an_unsorted_markov_grid():
    """A Markov grid may list its values in any order; init must still find them."""
    m = dp.Model("unsorted markov")
    m.parameters(beta=0.95, R=1.02)
    m.exogenous("y", dp.Markov([1.5, 0.5, 1.0], [[0.8, 0.1, 0.1],
                                                  [0.1, 0.8, 0.1],
                                                  [0.1, 0.1, 0.8]]))
    m.state("a", np.linspace(0.0, 10.0, 60))
    m.local("c = y + R*a - a(+1)")
    m.reward("log(c)")
    m.subject_to("c > 0")
    for method in ("vfi", "egm"):
        sol = m.solve(method)
        yg = sol.grids["y"]
        assert not np.all(np.diff(yg) > 0)
        for val in (0.5, 1.0, 1.5):
            p = sol.simulate(20, 3, seed=0, init={"a": sol.grids["a"][5], "y": val})
            assert np.all(p["y"][:, 0] == val)
        with pytest.raises(ValueError, match="shock grid"):
            sol.simulate(20, 3, seed=0, init={"a": sol.grids["a"][5], "y": 0.75})


def test_to_microframe_and_elapsed(aiyagari_egm, lc_sol, hjb_sol):
    assert aiyagari_egm.elapsed > 0.0
    assert lc_sol.elapsed > 0.0
    assert hjb_sol.elapsed > 0.0
    for sol in (aiyagari_egm, lc_sol, hjb_sol):
        mf = sol.to_microframe()
        assert np.isclose(float(mf.gini("a")["estimate"].iloc[0]), sol.gini("a"), rtol=1e-12)
        assert np.isclose(float(mf.top_share("a", 0.10)["estimate"].iloc[0]),
                          sol.top_share("a", 0.10), rtol=1e-12)
        assert np.isclose(float(mf.quantile("a", 0.50)["estimate"].iloc[0]),
                          sol.quantile("a", 0.50), rtol=1e-12)
    mf_age = lc_sol.to_microframe(age=3)
    assert np.isclose(float(mf_age.gini("a")["estimate"].iloc[0]), lc_sol.gini("a", age=3), rtol=1e-12)


def test_custom_functions_in_dp_model():
    m = dp.Model("progressive tax")
    m.parameters(beta=0.96, gamma=2.0, r=0.03, w=1.0)
    m.functions(
        tax=lambda y, xp=np: 0.15 * y + 0.10 * xp.maximum(y - 1.0, 0.0),
        bonus=lambda z: 0.05 * np.exp(z),
    )
    m.exogenous("z", dp.AR1(rho=0.9, sigma=0.2, n=5))
    m.state("a", np.linspace(0.0, 40.0, 80))
    m.local("gross_y = w*exp(z) + bonus(z)")
    m.local("c = (1 + r)*a + gross_y - tax(gross_y) - a(+1)")
    m.reward("crra(c, gamma)")
    m.subject_to("c > 0")
    m.aggregate(T="tax(gross_y)")
    sol_vfi = m.solve("vfi")
    sol_egm = m.solve("egm")
    assert sol_vfi.aggregates["T"] > 0.0
    assert abs(sol_vfi.aggregates["T"] - sol_egm.aggregates["T"]) < 0.02
    panel = sol_egm.simulate(200, 10, seed=3, record=["a", "c", "tax(gross_y)"])
    assert (panel["tax(gross_y)"] > 0).all()

    # Custom function in reward cannot be symbolically differentiated in EGM/HJB
    m_bad = dp.Model("custom u")
    m_bad.parameters(beta=0.96, r=0.03)
    m_bad.functions(u_fn=lambda c: np.log(c))
    m_bad.state("a", np.linspace(0.1, 10.0, 30))
    m_bad.local("c = (1 + r)*a - a(+1)")
    m_bad.reward("u_fn(c)")
    m_bad.subject_to("c > 0")
    assert m_bad.solve("vfi").V.shape == (30, 1)
    with pytest.raises(ModelSpecError, match="cannot differentiate"):
        m_bad.solve("egm")
