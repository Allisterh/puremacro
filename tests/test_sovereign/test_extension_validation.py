"""Independent extension equations and regressions for fabricated diagnostics."""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.sovereign import (solve_sudden_stop, solve_sovereign_default, SuddenStopResult,
                                 evaluate_debt_restructuring, decompose_spread_term_structure)
from tools.reference_validation.sovereign_extensions import collateral, long_term


@pytest.mark.parametrize('share', [.3, .9])
def test_collateral_matches_exhaustive_independent_reference(share):
    res = solve_sudden_stop(n_b=7, n_k=6, n_z=3, b_max=1.2,
                            kappa_collateral=share, tol=1e-10, max_iter=2000)
    ref = collateral(res.b_grid, res.k_grid, res.z_grid, res.p_trans, collateral_share=share)
    assert res.converged
    for name in ('policy_b', 'policy_k', 'policy_c'):
        np.testing.assert_allclose(getattr(res, name), ref[name], atol=1e-10, rtol=0)
    np.testing.assert_allclose(res.metadata['value_function'], ref['value'], atol=1e-8, rtol=0)
    assert res.metadata['bellman_residual'] < 1e-8
    if share == .3:
        assert res.metadata['grid_binding_mask'].any()


def test_no_productivity_change_has_no_invented_amplification():
    res = solve_sudden_stop(n_b=5, n_k=5, n_z=1, b_max=.4)
    assert np.isnan(res.amplification_ratio)
    assert res.metadata['amplification_status'] == 'undefined_zero_benchmark_response'


def test_large_collateral_share_does_not_force_zero_binding():
    res = solve_sudden_stop(kappa_collateral=.9, n_b=9, b_max=2.2,
                            n_k=5, k_min=2., k_max=2.2, n_z=3, tol=1e-9)
    assert res.metadata['grid_binding_mask'].any()
    assert res.collateral_binding_fraction > 0


def test_infeasible_collateral_grid_is_rejected():
    with pytest.raises(ValueError, match='feasible'):
        solve_sudden_stop(n_b=4, b_max=20., n_k=4, k_min=.8, k_max=1., n_z=3)


def test_grid_binding_fraction_is_not_copied_into_crisis_frequency():
    res = SuddenStopResult(k_grid=np.array([1.]), b_grid=np.array([0.]), z_grid=np.array([1.]),
                           collateral_binding_fraction=.5, crisis_probability=0.)
    assert res.crisis_probability == 0


@pytest.mark.parametrize('delta,coupon', [(.2, .03), (.5, .01), (1., 0.)])
def test_long_term_operator_matches_independent_reference(delta, coupon):
    res = solve_sovereign_default(n_b=17, b_min=-.1, b_max=.25, n_y=5,
                                  delta=delta, kappa=coupon, sigma_eps=.01,
                                  enforce_monotonicity=False, howard_steps=0,
                                  tol=1e-10, max_iter=10000)
    ref = long_term(res.b_grid, res.y_grid, res.p_trans, res.metadata['y_default_vec'],
                     delta=delta, coupon=coupon)
    assert res.converged
    for field, key in [('v_repay', 'v_repay'), ('v_default', 'v_default'), ('q_bond', 'q'),
                        ('default_prob', 'default'), ('policy_b', 'modal_policy')]:
        np.testing.assert_allclose(getattr(res, field), ref[key], atol=1e-7, rtol=0)


def test_haircut_reuses_the_original_equilibrium(monkeypatch):
    res = solve_sovereign_default(n_b=17, n_y=5, policy_search='monotone')
    import puremacro.sovereign.arellano as module
    def unexpected_solve(**kwargs):
        pytest.fail('a haircut changes the inherited state, not equilibrium parameters')
    monkeypatch.setattr(module, 'solve_sovereign_default', unexpected_solve)
    changed = evaluate_debt_restructuring(res, haircut=.2, maturity_extension=0,
                                         b_distress=.2, horizon=3)
    assert changed.metadata['res_restructured'] is res


def test_restructuring_path_starts_at_the_selected_income_state():
    res = solve_sovereign_default(n_b=17, n_y=5, sigma_eps=0, y_penalty_type='arellano')
    initial = 0
    seed = next(seed for seed in range(100) if np.random.default_rng(seed).choice(5, p=res.p_trans[initial]) != initial)
    changed = evaluate_debt_restructuring(res, haircut=0, maturity_extension=0,
                                         b_distress=0, y_distress=res.y_grid[initial], horizon=3, seed=seed)
    assert changed.transition_path.iloc[0]['output'] == res.y_grid[initial]


@pytest.mark.parametrize('extension', [0., .5])
def test_restructuring_matches_independent_long_term_state_values(extension):
    res = solve_sovereign_default(n_b=17, n_y=5, b_min=-.1, b_max=.25,
                                  delta=.2, kappa=.03, enforce_monotonicity=False,
                                  tol=1e-10, max_iter=10000)
    changed = evaluate_debt_restructuring(res, haircut=.2, maturity_extension=extension,
                                         b_distress=.2, y_distress=1., horizon=3)
    ref = long_term(res.b_grid, res.y_grid, res.p_trans, res.metadata['y_default_vec'],
                     delta=.2 / (1 + extension))
    ib = np.argmin(abs(res.b_grid - changed.restructured_debt))
    iy = np.argmin(abs(res.y_grid - 1.))
    assert changed.post_default_prob == pytest.approx(ref['default'][ib, iy], abs=1e-7)
    new_delta = .2 / (1 + extension)
    spread = 4 * max(0, (new_delta + (1 - new_delta) * .03) / ref['q'][ib, iy] - new_delta - .01)
    assert changed.counterfactual_spread == pytest.approx(spread, abs=1e-7)


def test_crisis_and_capacity_are_reproducible_measurements():
    res = solve_sudden_stop(n_b=9, n_k=7, n_z=3, tol=1e-9)
    path = res.simulate(T=2500, seed=42)
    sample = path.iloc[500:]
    expected_crisis = (path['collateral_binding'] & (path['consumption'].diff() < 0)
                       & (path['current_account'].diff() > 0))
    np.testing.assert_array_equal(path['crisis'], expected_crisis)
    assert res.crisis_probability == sample['crisis'].mean()
    capacity = res.kappa_collateral * sample['tobin_q'] * sample['k_next']
    assert res.metadata['mean_collateral_capacity'] == pytest.approx(capacity.mean())
    assert res.metadata['precautionary_buffer_pct'] == pytest.approx(100 * (capacity.mean() - sample['b_next'].mean()) / capacity.mean())


def test_long_term_simulation_uses_independently_checked_debt_probabilities():
    res = solve_sovereign_default(n_b=17, n_y=5, b_min=-.1, b_max=.25,
                                  delta=.2, kappa=.03, enforce_monotonicity=False,
                                  tol=1e-10, max_iter=10000)
    ref = long_term(res.b_grid, res.y_grid, res.p_trans, res.metadata['y_default_vec'])
    np.testing.assert_allclose(res.metadata['debt_choice_probabilities'], ref['probabilities'], atol=1e-7, rtol=0)
    rng = np.random.default_rng(42)
    ib, iy, excluded = np.argmin(abs(res.b_grid)), 2, False
    expected_debt = []
    for _ in range(200):
        iy = rng.choice(5, p=res.p_trans[iy])
        if excluded:
            if rng.random() < .282:
                excluded, ib = False, np.argmin(abs(res.b_grid))
            else:
                expected_debt.append(0.)
                continue
        if rng.random() < ref['default'][ib, iy]:
            excluded = True
            expected_debt.append(0.)
        else:
            ib = rng.choice(len(res.b_grid), p=ref['probabilities'][ib, :, iy])
            expected_debt.append(res.b_grid[ib])
    np.testing.assert_array_equal(res.simulate(T=200, seed=42)['b_next'], expected_debt)
    sample = res.simulate(T=1500, seed=42).query('repaid')
    assert len(sample) == res.metadata['spread_observations']
    assert sample['spread'].mean() == pytest.approx(res.spread_mean, abs=1e-14)
    from puremacro.sovereign.measurement import _simulate_arrays
    fast = _simulate_arrays(res, 1500, 42)
    full = res.simulate(T=1500, seed=42)
    for key in ('y', 'default', 'repaid', 'b', 'b_next', 'spread'):
        np.testing.assert_array_equal(fast[key], full[key])


@pytest.mark.parametrize('hazard', [0., .05])
def test_survival_term_structure_has_no_structural_risk_premium(hazard):
    from dataclasses import replace
    res = solve_sovereign_default(n_b=7, n_y=3, delta=.2, kappa=.03,
                                  b_min=-.05, b_max=.1, enforce_monotonicity=False)
    res = replace(res, default_prob=np.full_like(res.default_prob, hazard))
    frame = decompose_spread_term_structure(res, horizons=[1, 4, 20])
    h = frame['horizon_quarters'].to_numpy()
    survival = (1-hazard)**h
    price = survival / 1.01**h
    np.testing.assert_allclose(frame['bond_price'], price, atol=1e-13, rtol=0)
    expected_spread = (price**(-4/h)-1.01**4)*10000
    expected_loss = -4/h*np.log(survival)*10000
    np.testing.assert_allclose(frame['spread_bps'], expected_spread, atol=1e-10, rtol=0)
    np.testing.assert_allclose(frame['risk_premium_component'], expected_spread-expected_loss, atol=1e-10, rtol=0)
    assert 'compounding' in frame.attrs['risk_premium_component']


def test_long_term_survival_uses_the_full_choice_distribution():
    res = solve_sovereign_default(n_b=11, n_y=3, delta=.2, kappa=.03,
                                  b_min=-.05, b_max=.2, enforce_monotonicity=False)
    weights = res.metadata['debt_choice_probabilities']
    survival = np.ones_like(res.q_bond)
    ib, iy = 5, 1
    expected = []
    for horizon in range(1, 5):
        following = np.empty_like(survival)
        for i in range(len(res.b_grid)):
            for j in range(len(res.y_grid)):
                following[i, j] = sum(res.p_trans[j, k] * (1-res.default_prob[i, k])
                                      * sum(weights[i, l, k]*survival[l, k] for l in range(len(res.b_grid)))
                                      for k in range(len(res.y_grid)))
        survival = following
        expected.append(survival[ib, iy]/1.01**horizon)
    frame = decompose_spread_term_structure(res, horizons=[1, 2, 3, 4],
                                            b_eval=res.b_grid[ib], y_eval=res.y_grid[iy])
    np.testing.assert_allclose(frame['bond_price'], expected, atol=1e-13, rtol=0)


def test_frozen_extension_checks_match_the_source():
    import hashlib
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    record = json.loads((root/'tests/fixtures/sovereign_extensions/independent_validation.json').read_text(encoding='utf-8'))
    assert record['passed']
    assert record['tolerances'] == dict(value_price=1e-7, accounting=1e-8)
    assert record['collateral'][0]['grid_binding_fraction'] > 0
    for name, digest in record['source_sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name
    for group in ('long_term', 'collateral', 'restructuring'):
        assert all(case['passed'] for case in record[group])


def test_zero_survival_is_not_replaced_by_a_positive_price_floor():
    from dataclasses import replace
    res = solve_sovereign_default(n_b=7, n_y=3)
    res = replace(res, default_prob=np.ones_like(res.default_prob))
    frame = decompose_spread_term_structure(res, horizons=[1, 4])
    np.testing.assert_array_equal(frame['bond_price'], 0.)
    assert np.isinf(frame['yield_to_maturity']).all()
    assert np.isinf(frame['default_loss_component']).all()
    assert frame['risk_premium_component'].isna().all()


@pytest.mark.parametrize('state', [{'b_distress': np.nan}, {'y_distress': np.inf}])
def test_restructuring_rejects_nonfinite_distress_state(state):
    res = solve_sovereign_default(n_b=7, n_y=3)
    with pytest.raises(ValueError, match='finite'):
        evaluate_debt_restructuring(res, **state)
