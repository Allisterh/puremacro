"""Two-asset panels: interpolation, timing, budgets, initial mass and RNG."""
import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

from puremacro import dp


def household(reverse=False, shocks=True):
    m = dp.Model("two_asset_panel")
    grids = {"a": np.linspace(0, np.sqrt(30), 16)**2,
             "b": np.linspace(0, np.sqrt(8), 13)**2}
    for name in (("b", "a") if reverse else ("a", "b")):
        m.state(name, grids[name])
    if shocks:
        m.exogenous("z", dp.Markov([.5, 1.5], [[.95, .05], [.1, .9]]))
    m.parameters(beta=.98, gamma=1., r_b=.005, r_a=.0125, w=1., chi_0=1., chi_1=1., a_bar=.25)
    m.local("d=a(+1)-(1+r_a)*a")
    m.local("chi=chi_0/(1+chi_1)*abs(d)*(abs(d)/(a+a_bar))^chi_1")
    income = "w*z" if shocks else "w"
    m.local(f"c=(1+r_b)*b+(1+r_a)*a+{income}-a(+1)-b(+1)-chi")
    m.reward("crra(c,gamma)")
    m.subject_to("c>0")
    m.aggregate(C="c", A="a", B="b")
    return m.solve(method="egm", tol=1e-9)


@pytest.fixture(scope="module")
def solution():
    return household()


RECORD = ["a", "b", "z", "a(+1)", "b(+1)", "c", "chi", "d"]


def test_joint_bilinear_policies_timing_and_each_agent_budget(solution):
    init = {"a": np.array([0., .24, 2.1, 30.]), "b": np.array([.2, 0., .73, 8.]), "z": [ .5, 1.5, .5, 1.5]}
    panel = solution.simulate(n_agents=4, periods=5, init=init, record=RECORD, seed=28)
    a_grid, b_grid = solution._c.state_grids
    for t in range(5):
        points = np.column_stack([panel["a"][:, t], panel["b"][:, t]])
        for name, policy in (("a(+1)", solution.raw.aprime), ("b(+1)", solution.raw.bprime)):
            for z, value in enumerate([.5, 1.5]):
                mask = panel["z"][:, t] == value
                expected = RegularGridInterpolator((a_grid, b_grid), policy[:, :, z])(points[mask])
                np.testing.assert_allclose(panel[name][mask, t], expected, atol=1e-13)
    np.testing.assert_allclose(panel["a"][:, 1:], panel["a(+1)"][:, :-1])
    np.testing.assert_allclose(panel["b"][:, 1:], panel["b(+1)"][:, :-1])
    resources = 1.0125*panel["a"] + 1.005*panel["b"] + panel["z"]
    np.testing.assert_allclose(panel["c"] + panel["a(+1)"] + panel["b(+1)"] + panel["chi"], resources, atol=2e-14)
    assert np.all(panel["c"] > 0)
    np.testing.assert_allclose(panel["d"], panel["a(+1)"] - 1.0125*panel["a"])
    assert panel.to_frame().shape[0] == 20
    assert panel.to_microframe(t=0) is not None


def test_stationary_initial_joint_distribution_and_markov_frequencies(solution):
    panel = solution.simulate(n_agents=60_000, periods=2, seed=71, record=["a", "b", "z"])
    # Independent joint-cell counts, including asset/shock correlation.
    ia = np.searchsorted(solution._c.state_grids[0], panel["a"][:, 0])
    ib = np.searchsorted(solution._c.state_grids[1], panel["b"][:, 0])
    iz = (panel["z"][:, 0] == 1.5).astype(int)
    counts = np.zeros_like(solution.distribution)
    np.add.at(counts, (ia, ib, iz), 1)
    expected = solution.distribution
    sigma = np.sqrt(expected*(1-expected)/panel.n_agents)
    assert np.all(np.abs(counts/panel.n_agents - expected) <= 6*sigma + 6/panel.n_agents)
    for z, value in enumerate([.5, 1.5]):
        mask = panel["z"][:, 0] == value
        frequency = np.mean(panel["z"][mask, 1] == 1.5)
        assert abs(frequency - solution.P[z, 1]) < .01


def test_seed_and_burn_in_reproduce_exact_tail(solution):
    full = solution.simulate(n_agents=20, periods=9, seed=313, record=RECORD)
    same = solution.simulate(n_agents=20, periods=9, seed=313, record=RECORD)
    tail = solution.simulate(n_agents=20, periods=5, burn_in=4, seed=313, record=RECORD)
    changed = solution.simulate(n_agents=20, periods=9, seed=314, record=RECORD)
    for name in RECORD:
        np.testing.assert_array_equal(full[name], same[name])
        np.testing.assert_array_equal(full[name][:, 4:], tail[name])
    assert not np.array_equal(full["z"], changed["z"])


def test_reversed_declaration_order_matches_canonical_policy(solution):
    reverse = household(reverse=True)
    init = {"a": .3, "b": .7, "z": 1.5}
    left = solution.simulate(n_agents=8, periods=4, init=init, seed=82, record=RECORD)
    right = reverse.simulate(n_agents=8, periods=4, init=init, seed=82, record=RECORD)
    for name in RECORD:
        np.testing.assert_array_equal(left[name], right[name])
    # Stationary mass remains canonical (a,b,z), even if declarations are (b,a).
    left = solution.simulate(n_agents=8, periods=4, seed=2, record=RECORD)
    right = reverse.simulate(n_agents=8, periods=4, seed=2, record=RECORD)
    for name in RECORD:
        np.testing.assert_array_equal(left[name], right[name])


def test_parameter_mapping_order_cannot_change_recorded_budget(solution):
    import copy
    reordered = copy.copy(solution)
    reordered.params = dict(reversed(list(solution.params.items())))
    left = solution.simulate(5, 3, seed=73, record=RECORD)
    right = reordered.simulate(5, 3, seed=73, record=RECORD)
    for name in RECORD:
        np.testing.assert_array_equal(left[name], right[name])


def test_no_productivity_chain_is_supported():
    sol = household(shocks=False)
    panel = sol.simulate(n_agents=3, periods=2, seed=4, record=["a", "b", "c"])
    assert panel.names == ["a", "b", "c"]
    assert np.all(panel["c"] > 0)


@pytest.mark.parametrize("init", ["newborn", {"a": 1}, {"a": 1, "b": 1, "unknown": 0},
    {"a": np.nan, "b": 1}, {"a": 1, "b": [1, 2]}, {"a": 1, "b": 1, "z": .9}])
def test_invalid_initial_states_fail(solution, init):
    with pytest.raises(ValueError):
        solution.simulate(n_agents=3, periods=2, init=init)


def test_off_grid_states_are_continuous_and_outside_bounds_warn(solution):
    with pytest.warns(UserWarning, match="clipped"):
        panel = solution.simulate(n_agents=2, periods=1, init={"a": [-1, .123], "b": [.456, 100]}, record=["a", "b"])
    np.testing.assert_array_equal(panel["a"][:, 0], [0, .123])
    np.testing.assert_array_equal(panel["b"][:, 0], [.456, 8.000000000000002])
