"""Offline checks against author-code exports and economic moment conventions."""
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest
from scipy.special import ndtr

from tools.reference_validation.arellano_2008 import (
    FIXTURE, GROSS_R, Solution, annual_spread, author_grid, compare_original,
    eligible_windows, income_process, load_original, monte_carlo, simulate, solve,
    summarize_path, verified_inputs,
)


def test_archived_sources_and_external_export_integrity():
    verified_inputs()
    manifest = json.loads((FIXTURE / "reference_manifest.json").read_text())
    assert hashlib.sha256((FIXTURE / "reference.npz").read_bytes()).hexdigest() == manifest["reference_sha256"]
    assert manifest["outer_iterations"] <= 500
    assert manifest["price_error"] < 1e-8


def test_study_solver_matches_actual_author_fortran():
    result = solve()
    comparison = compare_original(result, load_original())
    assert comparison["passed"], comparison
    assert result.diagnostics["bellman_residual"] < 2e-11
    assert result.diagnostics["price_residual"] < 1e-8


def test_public_calibration_study_matches_the_public_solver():
    from puremacro.sovereign import solve_sovereign_default

    reference = solve(n_y=9, timing="canonical", calibration="public")
    result = solve_sovereign_default(beta=.95282, r_star=.017, theta=.282,
                                     n_b=401, n_y=9, b_min=-.15, b_max=.33,
                                     sigma_eps=0, y_penalty_type="arellano",
                                     howard_steps=0, enforce_monotonicity=False,
                                     tol=1e-10, max_iter=10000)
    assert result.converged
    np.testing.assert_allclose(reference.assets, -10*result.b_grid[::-1], atol=1e-14)
    np.testing.assert_allclose(reference.income, 10*result.y_grid, atol=1e-12)
    np.testing.assert_allclose(reference.price, result.q_bond[::-1], atol=1e-8, rtol=0)
    np.testing.assert_allclose(reference.value, .1*(result.v_star[::-1]-1/(1-.95282)), atol=1e-8, rtol=0)
    np.testing.assert_array_equal(reference.default, result.default_prob[::-1] > .5)
    np.testing.assert_allclose(reference.assets[reference.policy], -10*result.policy_b_prime[::-1], atol=1e-12)


@pytest.mark.parametrize("duplicates", [False, True])
def test_monotone_search_matches_exhaustive_cash_choice(duplicates):
    from tools.reference_validation._arellano_monge import maximize

    rng = np.random.default_rng(2026)
    assets = np.linspace(-.3, .2, 63)
    income = np.linspace(1., 2., 7)
    cash = rng.uniform(-.2, .3, (63, 7))
    continuation = rng.normal(scale=.2, size=(63, 7))
    if duplicates:
        cash[:] = 0.
        continuation[:] = 0.
    value, policy = maximize(assets, income, cash, continuation, np.argsort(-cash, axis=0, kind="stable"))
    objective = -1/(income[None, None, :]+assets[:, None, None]+cash[None, :, :])+continuation[None, :, :]
    np.testing.assert_allclose(value, objective.max(axis=1), atol=1e-15, rtol=0)
    np.testing.assert_array_equal(policy, 62-np.argmax(objective[:, ::-1], axis=1))


def test_accelerated_equilibrium_matches_author_fortran_and_exhaustive_check():
    reference = solve(policy_search="monotone")
    assert compare_original(reference, load_original())["passed"]
    assert reference.diagnostics["exhaustive_bellman_residual"] < 2e-11


def test_grid_refinements_hold_support_and_exact_zero():
    original = load_original()
    np.testing.assert_array_equal(author_grid(), original.assets)
    for refinement in (1, 2):
        fine = author_grid(refinement)
        np.testing.assert_array_equal(fine[::2**refinement], original.assets)
        assert np.sum(fine == 0) == 1
        assert np.all(np.diff(fine) > 0)


def test_archived_income_process_matches_rounded_tauchen_inputs():
    z, archived = verified_inputs()
    expected_z = np.linspace(-3*.025/np.sqrt(1-.945**2), 3*.025/np.sqrt(1-.945**2), 21)
    edges = np.r_[-np.inf, (expected_z[1:]+expected_z[:-1])/2, np.inf]
    expected_p = np.diff(ndtr((edges[None, :]-.945*expected_z[:, None])/.025), axis=1)
    np.testing.assert_allclose(z, expected_z, atol=5e-9, rtol=0)
    np.testing.assert_allclose(archived, expected_p, atol=5e-9, rtol=0)
    y, p = income_process(41)
    np.testing.assert_allclose(y[::2], 10*np.exp(z), atol=6e-8, rtol=0)
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=5e-16, rtol=0)


def test_compound_annual_spread_is_not_simple_annualization():
    q = np.array([1/1.017, 1/1.05])
    actual = annual_spread(q, 1.017)
    np.testing.assert_allclose(actual, [0, 1.05**4-1.017**4], atol=1e-15)
    assert actual[1] != pytest.approx(4*(1.05-1.017))
    with pytest.raises(ValueError, match="positive"):
        annual_spread(np.array([0.0]), GROSS_R)


def test_predefault_windows_require_contiguous_market_access():
    price = np.ones(300)
    entry = np.zeros(300, dtype=bool)
    entry[[60, 100, 180, 260]] = True
    price[[60, 61, 100, 101, 180, 181, 200, 260]] = 0
    # 60 has too little history, 100 includes exclusion, 260 contains 200.
    np.testing.assert_array_equal(eligible_windows(dict(price=price, entry=entry)), [180])


def test_exclusion_timing_and_default_entries_are_distinct():
    toy = Solution(np.array([-1., 0.]), np.array([1.]), np.ones((1, 1)), np.array([.9]),
                   np.zeros((2, 1)), np.ones((2, 1)), np.ones((2, 1), dtype=int),
                   np.ones((2, 1), dtype=bool), "archive", theta=1.)
    archived = simulate(toy, seed=1, periods=6, burn=0)
    canonical = simulate(replace(toy, timing="canonical"), seed=1, periods=6, burn=0)
    np.testing.assert_array_equal(archived["entry"], [1, 0, 1, 0, 1, 0])
    np.testing.assert_array_equal(canonical["entry"], np.ones(6, dtype=bool))
    assert np.all(archived["excluded"]) and archived["entry"].sum() == 3
    np.testing.assert_array_equal(archived["consumption"], archived["output"])


def test_moments_match_author_matlab_block_run_in_octave():
    reference = json.loads((FIXTURE / "moments_reference.json").read_text())
    actual = summarize_path(simulate(load_original(), seed=reference["seed"],
                                    periods=reference["periods"], burn=reference["burn"]))
    for name, expected in reference["moments"].items():
        assert actual["moments"][name] == pytest.approx(expected, abs=1e-10, rel=0)
    d = actual["diagnostics"]
    hazard = d["default_entries"]/(d["default_entries"]+d["repayment_periods"])
    assert d["annual_access_hazard_pct"] == pytest.approx(100*(1-(1-hazard)**4))
    assert d["annual_access_hazard_pct"] < actual["moments"]["annual_default_pct"]
    assert d["boundary_choice_share_pct"] == 0


def test_uncertainty_requires_independent_replications_and_enough_episodes():
    with pytest.raises(ValueError, match="distinct seeds"):
        monte_carlo(load_original(), [2, 2], 100, 10)
    path = simulate(load_original(), seed=2, periods=100, burn=10)
    with pytest.raises(ValueError, match="eligible windows"):
        summarize_path(path)


def test_grid_gate_distinguishes_drift_from_simulation_uncertainty():
    from tools.reference_validation.check_arellano_grids import TOLERANCES, paired_comparison

    def runs(debts):
        return dict(runs=[dict(seed=i, moments={name: debt if name == "debt_pct" else 0.
                                              for name in TOLERANCES}) for i, debt in enumerate(debts)])

    baseline = runs([0., 0., 0., 0.])
    assert paired_comparison(baseline, runs([.05]*4))["passed"]
    drift = paired_comparison(baseline, runs([.15]*4))
    assert drift["moments"]["debt_pct"]["status"] == "grid_shift_exceeds_tolerance"
    imprecise = paired_comparison(baseline, runs([-.2, -.2, .2, .2]))
    assert not imprecise["passed"]
    assert imprecise["moments"]["debt_pct"]["status"] == "sampling_precision_insufficient"
    # Pair by seed, not by list position: otherwise an identical experiment
    # reordered on disk spuriously acquires sampling uncertainty.
    reference = runs([-.2, -.2, .2, .2])
    reordered = dict(runs=list(reversed(reference["runs"])))
    assert paired_comparison(reference, reordered)["passed"]
    with pytest.raises(ValueError, match="same distinct seeds"):
        paired_comparison(baseline, dict(runs=reordered["runs"][:-1]))
