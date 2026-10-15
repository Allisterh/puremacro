"""DPSolution.simulate with a discrete choice, with and without taste shocks."""
from __future__ import annotations

import numpy as np
import pytest

from puremacro import dp


def _work_or_not(taste: bool = True, n_a: int = 200):
    """The work-or-not model of docs/dp.md; ``taste=False`` drops the taste shocks."""
    w = dp.Model("work or not")
    w.parameters(beta=0.95, gamma=2.0, r=0.03, chi=0.5, b=0.3, sig=0.05)
    w.exogenous("z", dp.AR1(rho=0.9, sigma=0.2, n=5))
    w.state("a", np.linspace(0.0, 60.0, n_a))
    w.discrete("h", [0.0, 1.0])
    w.local("c = (1 + r)*a + h*exp(z) + (1 - h)*b - a(+1)")
    w.reward("crra(c, gamma) - chi*h")
    w.subject_to("c > 0")
    if taste:
        w.taste_shocks("sig")
    w.aggregate(H="h", A="a")
    return w


@pytest.fixture(scope="module")
def work_taste():
    return _work_or_not().solve("egm")


def test_taste_shock_choice_frequencies_match_logit(work_taste):
    sol = work_taste
    assert sol.mean("h") == pytest.approx(0.63298, abs=5e-5)          # docs/dp.md
    n = 20_000
    panel = sol.simulate(n, 200, seed=1)
    assert panel.names == ["a", "z", "c", "h"]
    h = panel["h"]
    assert set(np.unique(h)) <= {0.0, 1.0}                              # options, not averages
    share = h[:, -1].mean()
    se = np.sqrt(share * (1 - share) / n)
    assert abs(share - sol.mean("h")) < 3 * se


def test_taste_shock_choices_follow_logit_probabilities_by_cell(work_taste):
    # agents placed on grid nodes choose with the solver's probabilities there
    sol = work_taste
    a = sol.grids["a"]
    n, node, zi = 40_000, 20, 2
    panel = sol.simulate(n, 1, seed=4, init={"a": a[node], "z": sol.grids["z"][zi]})
    p = sol.raw.choice_prob[1, node, zi]
    share = panel["h"][:, 0].mean()
    assert abs(share - p) < 3 * np.sqrt(p * (1 - p) / n)
    # the next asset is the chosen option's policy, not the probability-weighted mean
    rec = sol.simulate(200, 1, seed=4, init={"a": a[node], "z": sol.grids["z"][zi]},
                       record=["h", "a(+1)"])
    for k in (0, 1):
        sel = rec["h"][:, 0] == k
        assert sel.any()
        np.testing.assert_allclose(rec["a(+1)"][sel, 0], sol.raw.aprime_by_choice[k, node, zi],
                                   rtol=0, atol=1e-12)


@pytest.mark.parametrize("method", ["vfi", "egm"])
def test_deterministic_choice_follows_argmax(method):
    sol = _work_or_not(taste=False, n_a=120).solve(method)
    panel = sol.simulate(2000, 50, seed=2, record=["a", "z", "h"])
    a, h = sol.grids["a"], panel["h"]
    zi = np.searchsorted(np.sort(sol.grids["z"]), panel["z"])
    zi = np.argsort(sol.grids["z"])[zi]
    # the agent's cell: its node under VFI, the bracketing node nearer its asset under EGM
    j = np.clip(np.searchsorted(a, panel["a"]), 1, a.size - 1)
    ai = np.where(np.abs(panel["a"] - a[j - 1]) <= np.abs(a[j] - panel["a"]), j - 1, j)
    if method == "vfi":
        assert np.array_equal(a[ai], panel["a"])
    np.testing.assert_array_equal(h, sol.policy("h")[ai, zi])
    assert 0.0 < h.mean() < 1.0                                          # both options occur


def test_deterministic_choice_next_asset_is_the_policy():
    sol = _work_or_not(taste=False, n_a=120).solve("vfi")
    panel = sol.simulate(500, 20, seed=5, record=["a", "a(+1)"])
    np.testing.assert_array_equal(panel["a(+1)"][:, :-1], panel["a"][:, 1:])


def _retire(T: int, sigma: float, n_a: int = 101):
    m = dp.Model("work or retire")
    m.parameters(beta=0.96, r=0.03, chi=0.4, sig=sigma)
    m.exogenous("z", dp.AR1(rho=0.9, sigma=0.1, n=3))
    m.state("a", np.linspace(0.0, 20.0, n_a))
    m.discrete("h", [0.0, 1.0])
    m.local("c = (1 + r)*a + 0.3 + 0.9*h*exp(z) + 2*(age < 3) - a(+1)")
    m.reward("log(c) - chi*h")
    m.subject_to("c > 0")
    m.taste_shocks("sig")
    m.horizon(T, survival=np.full(T, 0.98))
    return m


def test_lifecycle_taste_shocks_simulate():
    T = 10
    sol = _retire(T, 0.1).solve("egm")
    assert sol.raw.choice_prob.shape == (T, 2, 101, 3)
    n = 20_000
    panel = sol.simulate(n, seed=6)
    assert panel.periods == T and panel.names == ["a", "z", "c", "h"]
    h = panel["h"]
    assert set(np.unique(h)) <= {0.0, 1.0}
    # participation by age matches the cohort's exact mean within Monte Carlo error
    # (every agent counts: the exact cohort distribution ignores survival)
    share = h.mean(axis=0)
    exact = sol.mean("h")
    se = np.sqrt(np.maximum(exact * (1 - exact), 1e-12) / n)
    assert np.all(np.abs(share - exact) < 4 * se + 1e-3)
    np.testing.assert_allclose(panel["a"].mean(axis=0), sol.mean("a"), atol=0.02)
    # determinism: the choice uniforms come from the one generator
    again = sol.simulate(n, seed=6)
    assert np.array_equal(again["h"], h) and np.array_equal(again["a"], panel["a"])
    assert np.array_equal(again.alive, panel.alive)
    assert not np.array_equal(sol.simulate(n, seed=7)["h"], h)


def test_lifecycle_deterministic_choice_simulates_with_vfi():
    m = _retire(6, 0.0, n_a=61)
    m._taste = None
    sol = m.solve("vfi")
    panel = sol.simulate(1000, seed=8, record=["a", "z", "h"])
    a = sol.grids["a"]
    ai = np.searchsorted(a, panel["a"])
    zi = np.argsort(sol.grids["z"])[np.searchsorted(np.sort(sol.grids["z"]), panel["z"])]
    pol = sol.policy("h")
    for j in range(6):
        np.testing.assert_array_equal(panel["h"][:, j], pol[j][ai[:, j], zi[:, j]])
