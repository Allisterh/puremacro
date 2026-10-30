"""Independent author-code equilibrium and economic-accounting checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from puremacro.models.one_asset_hank import calibrate_one_asset_hank, solve_one_asset_hank

FIXTURES = Path(__file__).resolve().parents[1] / 'fixtures' / 'hank_equilibrium'


@pytest.fixture(scope='module')
def steady():
    return calibrate_one_asset_hank(n_a=100, n_e=3)


def test_published_equilibrium_clears_all_markets(steady):
    assert steady.converged
    assert abs(steady.aggregates['A'] - 5.6) < 1e-8
    assert abs(steady.aggregates['NE'] - 1) < 1e-8
    assert abs(steady.aggregates['C'] - 1) < 1e-8
    assert np.min(steady.policy_c) > 0 and np.min(steady.policy_n) > 0
    assert np.min(steady.policy_a) >= 0
    assert np.min(steady.distribution) >= -1e-14
    assert abs(steady.distribution.sum() - 1) < 1e-12
    cal = steady.calibration
    income = cal['w'] * steady.e_grid[None, :] * steady.policy_n
    transfers = (steady.aggregates['Div'] - steady.aggregates['Tax']) * steady.e_grid[None, :]
    resources = (1 + cal['r']) * steady.asset_grid[:, None] + income + transfers
    np.testing.assert_allclose(steady.policy_c + steady.policy_a, resources, rtol=0, atol=1e-10)


def test_matches_independent_author_steady_state(steady):
    stem = 'hank_equilibrium_100_3_40'
    report = json.loads((FIXTURES / f'{stem}.json').read_text(encoding='utf-8'))
    data = FIXTURES / f'{stem}.npz'
    assert hashlib.sha256(data.read_bytes()).hexdigest() == report['fixture_sha256']
    reference = np.load(data)
    for key in ('beta', 'vphi'):
        assert steady.calibration[key] == pytest.approx(report['steady_state'][key], abs=1e-7)
    for name, field in [('a', 'policy_a'), ('c', 'policy_c'), ('n', 'policy_n'), ('D', 'distribution')]:
        np.testing.assert_allclose(getattr(steady, field), reference[name], rtol=0, atol=1e-6 if name == 'a' else 1e-7)


def test_zero_shock_is_stationary(steady):
    result = solve_one_asset_hank(steady, horizon=40, magnitude=0, nonlinear=True)
    assert result.converged
    np.testing.assert_allclose(result.paths[['C', 'Y', 'pi', 'A', 'NE']], 0, rtol=0, atol=1e-8)


@pytest.mark.parametrize('mode,sign', [('linear', 1), ('nonlinear', 1), ('opposite', -1)])
def test_matches_independent_equilibrium_transition(steady, mode, sign):
    reference = np.load(FIXTURES / 'hank_equilibrium_100_3_40.npz')
    result = solve_one_asset_hank(steady, horizon=40, magnitude=sign * -.0025,
                                  nonlinear=mode != 'linear')
    assert result.converged
    for name in ('pi', 'w', 'Y', 'C', 'A', 'NE', 'r', 'Div', 'Tax'):
        np.testing.assert_allclose(result.paths[name], reference[f'{mode}_{name}'], rtol=0,
                                   atol=2.5e-9 if mode == 'linear' else 1e-7)
    assert result.diagnostics['market_residual_max_abs'] < 1e-8


def test_unfinished_nonlinear_solve_does_not_claim_convergence(steady):
    result = solve_one_asset_hank(steady, horizon=40, magnitude=-.02,
                                  nonlinear=True, max_iterations=1)
    assert not result.converged
    assert result.diagnostics['market_residual_max_abs'] > 1e-8


@pytest.mark.parametrize('kwargs', [dict(n_a=2), dict(n_e=1), dict(a_max=3),
                                    dict(sigma_e=0), dict(phi=1), dict(rho_e=1),
                                    dict(B=float('nan'))])
def test_invalid_calibrations_rejected(kwargs):
    with pytest.raises(ValueError):
        calibrate_one_asset_hank(**kwargs)


def test_result_reports_preserve_paths(steady):
    result = solve_one_asset_hank(steady, horizon=10)
    for report in (steady, result):
        assert not report.to_dataframe().empty
        assert report.summary()
        assert report.to_markdown()
        assert report.to_latex()
        assert report.to_typst()
    frame = result.to_dataframe()
    frame.loc[0, 'C'] = 99
    assert result.paths.loc[0, 'C'] != 99


def test_frozen_full_equilibrium_and_refinement_record():
    record = json.loads((FIXTURES / 'convergence.json').read_text(encoding='utf-8'))
    source = Path(__file__).resolve().parents[2] / 'puremacro/models/one_asset_hank.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == record['native_source_sha256']
    assert record['passed'] and record['author_comparison']['passed']
    assert all(row['passed'] for row in record['author_comparison']['checks'])
    assert all(row['passed'] for row in record['refinements'] if row['required'])
    # Failed intermediate resolutions must remain visible after extending the
    # ladder; matching calibrated targets is not the convergence criterion.
    assert any(not row['passed'] for row in record['refinements'])
    assert 'a1000_e31_T600_max600' in record['cases']
