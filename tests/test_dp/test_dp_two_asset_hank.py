"""Tests for Two-Asset HANK Dynamic Programming and Fake-News Sequence-Space Jacobians.

Verifies:
1. Two-Asset EGM solver compilation via TwoAssetEGMPlan and Model.solve(method="egm").
2. Joint stationary distribution and policy evaluation on (a, b, s) grids.
3. dp_sequence_jacobian Fake-News algorithm (Auclert et al. 2021 §4) for two-asset models.
4. Parity of linear SSJ transition vs exact non-linear MIT-shock transition within < 1e-4 relative error.
5. Integration with puremacro.dsge.hank.HANKModel for 2-asset general equilibrium simulation.
"""
from __future__ import annotations

import numpy as np
import pytest

from puremacro import dp
from puremacro.dp._egm_two_asset import TwoAssetEGMPlan, TwoAssetHouseholdSolution
from puremacro.dsge.hank import HANKModel, load_hank_mod
from puremacro.models.hank_sequence_space import (
    _two_asset_grid,
    dp_nonlinear_transition,
    dp_sequence_jacobian,
)


@pytest.fixture(scope="module")
def two_asset_dp_model():
    """Create a calibrated two-asset household dynamic programming model."""
    m = dp.Model("two_asset_household")
    n_a = 20
    n_b = 20
    a_grid = _two_asset_grid(0.0, 40.0, n_a, curvature=2.0)
    b_grid = _two_asset_grid(0.0, 10.0, n_b, curvature=1.5)

    m.state("a", a_grid)
    m.state("b", b_grid)

    # 2-state idiosyncratic labor productivity chain
    pi_s = np.array([[0.95, 0.05], [0.10, 0.90]])
    s_grid = np.array([0.5, 1.5])
    m.exogenous("z", dp.Markov(s_grid, pi_s))

    m.parameters(
        beta=0.98,
        gamma=1.0,
        r_b=0.005,
        r_a=0.0125,
        w=1.0,
        chi_0=1.0,
        chi_1=1.0,
        a_bar=0.25,
    )
    m.local("d = a(+1) - (1 + r_a) * a")
    m.local("chi = chi_0 / (1 + chi_1) * abs(d) * (abs(d) / (a + a_bar))^chi_1")
    m.local("c = (1 + r_b) * b + (1 + r_a) * a + w * z - a(+1) - b(+1) - chi")
    m.reward("crra(c, gamma)")
    m.subject_to("c > 0")
    m.aggregate(C="c", A="a", B="b", D="d", CHI="chi")
    return m


def test_two_asset_egm_plan_solve(two_asset_dp_model):
    """Verify TwoAssetEGMPlan compiles and solves the two-asset model to high precision."""
    comp = two_asset_dp_model.check()
    plan = TwoAssetEGMPlan(comp)
    assert plan.n_a == 20
    assert plan.n_b == 20
    assert plan.illiquid_name == "a"
    assert plan.liquid_name == "b"

    params = comp.param_values()
    sol = plan.solve_household(params, tol=1e-8, max_iter=500)
    assert isinstance(sol, TwoAssetHouseholdSolution)
    assert sol.converged
    assert sol.c.shape == (20, 20, 2)
    assert np.all(sol.c > 0.0)
    assert np.all(sol.aprime >= plan.a_grid[0])
    assert np.all(sol.bprime >= plan.b_grid[0])

    dist = plan.distribution(sol)
    assert dist.shape == (20, 20, 2)
    assert np.all(dist >= 0.0)
    np.testing.assert_allclose(np.sum(dist), 1.0, atol=1e-10)


def test_model_solve_egm_two_asset(two_asset_dp_model):
    """Verify Model.solve(method='egm') dispatches seamlessly for two-asset models."""
    sol = two_asset_dp_model.solve(method="egm", tol=1e-8)
    assert sol.distribution is not None
    assert sol.distribution.shape == (20, 20, 2)
    np.testing.assert_allclose(np.sum(sol.distribution), 1.0, atol=1e-10)

    # Test aggregate means
    mean_c = sol.mean("c")
    mean_a = sol.mean("a")
    mean_b = sol.mean("b")
    assert mean_c > 0.0
    assert mean_a > 0.0
    assert mean_b >= 0.0


def test_two_asset_dp_sequence_jacobian_vs_nonlinear_mit_shock(two_asset_dp_model):
    """Verify dp_sequence_jacobian Fake-News algorithm matches non-linear MIT shocks (< 1e-4)."""
    sol = two_asset_dp_model.solve(method="egm", tol=1e-8)
    T = 24
    J = dp_sequence_jacobian(sol, inputs=["r_b", "r_a", "w"], outputs=["C", "A", "B", "D", "CHI"], T=T)

    assert "C" in J
    assert "r_b" in J["C"]
    assert "r_a" in J["C"]
    assert "w" in J["C"]
    assert J["C"]["r_b"].shape == (T, T)
    assert J["C"]["r_a"].shape == (T, T)
    assert J["C"]["w"].shape == (T, T)

    # 1. Non-linear MIT shock test: interest rate perturbation sequence
    eps_0 = 2e-5
    rho = 0.7
    shock_seq = eps_0 * (rho ** np.arange(T))
    nl_up = dp_nonlinear_transition(sol, input_paths={"r_b": shock_seq}, outputs=["C", "A", "B"])
    nl_dn = dp_nonlinear_transition(sol, input_paths={"r_b": -shock_seq}, outputs=["C", "A", "B"])

    # Linear prediction from sequence Jacobian
    linear_pred_C = J["C"]["r_b"] @ shock_seq
    linear_pred_A = J["A"]["r_b"] @ shock_seq
    linear_pred_B = J["B"]["r_b"] @ shock_seq

    # Central difference non-linear response
    actual_C = 0.5 * (nl_up["C"] - nl_dn["C"])
    actual_A = 0.5 * (nl_up["A"] - nl_dn["A"])
    actual_B = 0.5 * (nl_up["B"] - nl_dn["B"])

    # Verify matching within < 1e-4 relative error
    rel_err_C = float(np.linalg.norm(linear_pred_C - actual_C) / np.linalg.norm(actual_C))
    rel_err_A = float(np.linalg.norm(linear_pred_A - actual_A) / np.linalg.norm(actual_A))
    rel_err_B = float(np.linalg.norm(linear_pred_B - actual_B) / np.linalg.norm(actual_B))

    assert rel_err_C < 1e-4, f"C Jacobian relative error {rel_err_C:.2e} exceeds 1e-4"
    assert rel_err_A < 1e-4, f"A Jacobian relative error {rel_err_A:.2e} exceeds 1e-4"
    assert rel_err_B < 1e-4, f"B Jacobian relative error {rel_err_B:.2e} exceeds 1e-4"


    # 2. Wage shock test
    shock_seq_w = 2e-5 * (0.6 ** np.arange(T))
    nl_up_w = dp_nonlinear_transition(sol, input_paths={"w": shock_seq_w}, outputs=["C"])
    nl_dn_w = dp_nonlinear_transition(sol, input_paths={"w": -shock_seq_w}, outputs=["C"])
    actual_C_w = 0.5 * (nl_up_w["C"] - nl_dn_w["C"])
    linear_pred_C_w = J["C"]["w"] @ shock_seq_w
    rel_err_C_w = float(np.linalg.norm(linear_pred_C_w - actual_C_w) / np.linalg.norm(actual_C_w))
    assert rel_err_C_w < 1e-4, f"C wage Jacobian relative error {rel_err_C_w:.2e} exceeds 1e-4"


def test_hank_model_with_two_asset_dp_block(two_asset_dp_model):
    """Verify HANKModel accepts a 2-asset DP household model and simulates GE transitions."""
    mod_path = "puremacro/dsge/_references/hank_two_asset.mod"
    # Initialize HANKModel with 2-asset DP block
    hm = load_hank_mod(mod_path, household_model=two_asset_dp_model)
    assert hm.is_two_asset is True
    assert hm.asset_grid.shape == (20,)
    assert hm.liquid_asset_grid.shape == (20,)
    assert hm.joint_distribution is not None
    assert hm.joint_distribution.shape == (20, 20)
    assert np.all(hm.mpc_distribution > 0.0)

    # Compute Jacobians
    J_dict = hm.compute_jacobians(T=24)
    assert "J_C_rb" in J_dict or "J_C_r" in J_dict

    # Simulate monetary policy rate cut
    res = hm.simulate(shock="eps_m", magnitude=-0.0025, rho=0.5, horizon=24)
    assert res.converged
    assert "Y" in res.transition_paths.columns
    assert "C" in res.transition_paths.columns
    assert "r_b" in res.transition_paths.columns
    # Output and consumption expand on impact after an expansionary rate cut
    assert res.transition_paths["Y"].iloc[0] > 0.0
    assert res.transition_paths["C"].iloc[0] > 0.0
    with pytest.raises(NotImplementedError, match="nonlinear general-equilibrium"):
        hm.simulate(horizon=4, nonlinear=True)


@pytest.mark.parametrize("change", ["reward", "budget", "constraint", "narrow_constraint", "finite", "multiple_shocks"])
def test_two_asset_egm_rejects_unsupported_declarations(two_asset_dp_model, change):
    """The specialized backend must not silently solve a different model."""
    import copy
    from puremacro.dp._expr import ModelSpecError

    m = copy.deepcopy(two_asset_dp_model)
    if change == "reward":
        m.reward("-c*c")
    elif change == "budget":
        m._locals[-1] = ("c", m._locals[-1][1] + " + 0.2")
    elif change == "constraint":
        m.subject_to("a(+1) <= a")
    elif change == "narrow_constraint":
        m.subject_to("abs(b(+1) - 5.673058346259636) > 0.06519579979010634")
    elif change == "finite":
        m.horizon(3)
    else:
        m.exogenous("other", dp.Markov(np.array([0., 1.]), np.eye(2)))
    with pytest.raises(ModelSpecError, match="two-asset EGM"):
        m.solve(method="egm")


def test_two_asset_current_states_and_discount_parameter(two_asset_dp_model):
    import copy

    m = copy.deepcopy(two_asset_dp_model)
    m.parameters(patience=0.96)
    m.discount("patience")
    sol = m.solve(method="egm", tol=1e-8)
    assert sol.raw.hh_block.beta == 0.96
    np.testing.assert_array_equal(sol.policy("a"), np.broadcast_to(m._states[0][1][:, None, None], sol.V.shape))
    np.testing.assert_array_equal(sol.policy("b"), np.broadcast_to(m._states[1][1][None, :, None], sol.V.shape))
    np.testing.assert_array_equal(sol.policy("a(+1)"), sol.raw.aprime)
    np.testing.assert_allclose(sol.policy("c"), sol.raw.c, atol=1e-12)
    np.testing.assert_allclose(sol.policy("d"), sol.raw.d, atol=1e-12)
    with pytest.raises(NotImplementedError, match="two-asset EGM panel simulation"):
        sol.simulate(10, 2)


def test_two_asset_custom_outputs_and_registered_functions(two_asset_dp_model):
    """A named aggregate must follow its expression, including direct price effects."""
    import copy

    m = copy.deepcopy(two_asset_dp_model)
    m.functions(twice=lambda x, xp=np: 2 * x)
    m.aggregate(C="twice(c)", PRICE="w*c", WEALTH="a+b", AA="2*a")
    sol = m.solve(method="egm", tol=1e-10)
    outputs = ["C", "c", "PRICE", "WEALTH", "A", "AA"]
    J = dp_sequence_jacobian(sol, inputs=["w"], outputs=outputs, T=8, dx=1e-5)
    np.testing.assert_allclose(J["C"]["w"], 2 * J["c"]["w"], atol=1e-12)
    np.testing.assert_allclose(J["AA"]["w"], 2 * J["A"]["w"], atol=1e-12)
    np.testing.assert_allclose(J["A"]["w"][0], 0, atol=1e-12)
    np.testing.assert_allclose(J["PRICE"]["w"] - J["c"]["w"],
                               sol.mean("c") * np.eye(8), atol=1e-8)
    shock = 1e-5 * 0.6 ** np.arange(8)
    up = dp_nonlinear_transition(sol, {"w": shock}, outputs=outputs)
    dn = dp_nonlinear_transition(sol, {"w": -shock}, outputs=outputs)
    for name in outputs:
        np.testing.assert_allclose(J[name]["w"] @ shock, (up[name] - dn[name]) / 2,
                                   rtol=1e-4, atol=1e-9)
    with pytest.raises(ValueError, match="Unknown two-asset input"):
        dp_sequence_jacobian(sol, inputs=["bogus"], T=2)


def test_two_asset_reports_nonconvergence(two_asset_dp_model):
    with pytest.warns(RuntimeWarning, match="did not converge"):
        sol = two_asset_dp_model.solve(method="egm", tol=1e-10, max_iter=1)
    assert not sol.raw.converged


def test_two_asset_periodic_reducible_stationary_failure():
    from puremacro.models.hank_sequence_space import _stationary_distribution_2d

    # A transient state feeding a two-cycle and a separate absorbing class.
    # Normalization and finiteness alone do not establish stationarity.
    transition = np.array([[0., 1., 1., 0.], [1., 0., 0., 0.],
                           [0., 0., 0., 0.], [0., 0., 0., 1.]])
    with pytest.warns(RuntimeWarning, match="stationary distribution.*did not reach"):
        dist = _stationary_distribution_2d(transition)
    np.testing.assert_array_equal(dist, [0.25, 0.5, 0., 0.25])
    assert np.max(np.abs(transition @ dist - dist)) == 0.25


@pytest.mark.parametrize("backend", ["declarative", "builtin"])
@pytest.mark.filterwarnings("ignore:two-asset steady state:RuntimeWarning")
def test_two_asset_stationary_failure_prevents_convergence(two_asset_dp_model, monkeypatch, backend):
    """Propagate a real stationary-iteration failure through both household APIs."""
    import copy
    import puremacro.dp._egm_two_asset as declarative
    import puremacro.models.hank_sequence_space as builtin

    module = declarative if backend == "declarative" else builtin
    original_transition = module._build_transition_matrix_2d

    def periodic_reducible_transition(*args):
        transition = np.eye(original_transition(*args).shape[0])
        transition[:4, :4] = [[0., 1., 1., 0.], [1., 0., 0., 0.],
                              [0., 0., 0., 0.], [0., 0., 0., 1.]]
        return transition

    if backend == "declarative":
        model = copy.deepcopy(two_asset_dp_model)
        model._states = [(name, np.linspace(0., 2., 3)) for name, _ in model._states]

        def solve():
            return model.solve(method="egm", tol=1e-8)

        assert solve().raw.converged
    else:
        def solve():
            return builtin._solve_two_asset_household_block(n_a=3, n_b=3, a_max=2., b_max=2., tol=1e-8)

        assert solve().converged

    monkeypatch.setattr(module, "_build_transition_matrix_2d", periodic_reducible_transition)
    with pytest.warns(RuntimeWarning, match="stationary distribution.*did not reach"):
        result = solve()
    hh = result.raw.hh_block if backend == "declarative" else result
    assert hh.residual < hh.tol  # Policy iteration itself converged.
    assert not hh.converged
    assert np.max(np.abs(hh.Lambda @ hh.D_ss.ravel() - hh.D_ss.ravel())) > 1e-2
    if backend == "declarative":
        assert not result.raw.converged
        with pytest.raises(RuntimeError, match="require a converged household"):
            dp_sequence_jacobian(result, inputs=["w"], outputs=["C"], T=2)


def test_two_asset_lottery_preserves_mean_on_narrow_grids():
    from puremacro.models.hank_sequence_space import _lottery_2d

    grid = np.array([0.0, 1e-8, 2e-8])
    values = np.array([0.0, 0.5e-8, 1.5e-8, 2e-8])
    ia, ja, wa, va, ib, jb, wb, vb = _lottery_2d(values, values, grid, grid)
    np.testing.assert_allclose(wa * grid[ia] + va * grid[ja], values, atol=1e-24)
    np.testing.assert_allclose(wb * grid[ib] + vb * grid[jb], values, atol=1e-24)


@pytest.mark.parametrize("deterministic", [False, True])
def test_two_asset_reversed_unequal_grids_and_no_shock(two_asset_dp_model, deterministic):
    import copy

    m = copy.deepcopy(two_asset_dp_model)
    m._states = [("b", m._states[1][1][::2]), ("a", m._states[0][1])]
    if deterministic:
        m._shocks = []
        m._locals[-1] = ("c", m._locals[-1][1].replace("w * z", "w"))
    sol = m.solve(method="egm", tol=1e-8)
    assert sol.distribution.shape == (20, 10, 1 if deterministic else 2)
    np.testing.assert_array_equal(sol.policy("a"), np.broadcast_to(m._states[1][1][:, None, None], sol.V.shape))
    np.testing.assert_array_equal(sol.policy("b"), np.broadcast_to(m._states[0][1][None, :, None], sol.V.shape))
    np.testing.assert_allclose(sol.policy("c"), sol.raw.c, atol=1e-12)
    frame = sol.to_microframe().data
    assert set(frame["a"]).issubset(set(m._states[1][1]))
    assert set(frame["b"]).issubset(set(m._states[0][1]))
    np.testing.assert_allclose(np.sum(frame["weight"]), 1.0)
