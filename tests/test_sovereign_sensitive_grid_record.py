"""A convergence verdict requires the registered experiment and complete data."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools.reference_validation.check_sovereign_sensitive_grids import (
    BASELINE, DESIGN, GRIDS, SCENARIOS, TOLERANCES, assess, combine,
)


@pytest.fixture
def record():
    path = Path(__file__).parent / 'fixtures/sovereign_brief/sweep.json'
    frozen = json.loads(path.read_text(encoding='utf-8'))
    result = dict(schema=1, baseline=deepcopy(BASELINE), design=deepcopy(DESIGN),
                  tolerances=dict(TOLERANCES), grids=[list(g) for g in GRIDS], scenarios={})
    # Reuse real per-seed measurements at every grid to isolate the assessment
    # contract. These synthetic records are never evidence of grid convergence.
    for name, overrides in SCENARIOS.items():
        cases = {}
        for b, y in GRIDS:
            case = deepcopy(frozen['scenarios'][name]['cases']['6401x641'])
            case.update(n_b=b, n_y=y)
            case['parameters'].update(n_b=b, n_y=y)
            cases[f'{b}x{y}'] = case
        result['scenarios'][name] = dict(overrides=dict(overrides), cases=cases)
    return result


def test_registered_zero_grid_change_passes(record):
    assert assess(record)['passed']


@pytest.mark.parametrize('change', ['empty', 'baseline', 'missing_grid', 'overrides',
                                   'parameters', 'seeds', 'truncated', 'nonfinite'])
def test_incompatible_or_incomplete_study_cannot_pass(record, change):
    scenario = record['scenarios']['theta=0.5']
    case = scenario['cases']['12801x1281']
    if change == 'empty':
        record['scenarios'].clear()
    elif change == 'baseline':
        record['baseline']['beta'] = .9
    elif change == 'missing_grid':
        del scenario['cases']['12801x1281']
    elif change == 'overrides':
        scenario['overrides']['theta'] = .15
    elif change == 'parameters':
        case['parameters']['theta'] = .15
    elif change == 'seeds':
        case['simulation']['seeds'][0] = -1
    else:
        values = case['simulation']['per_seed'][next(iter(TOLERANCES))]
        if change == 'truncated':
            values.pop()
        else:
            values[0] = float('nan')
    with pytest.raises(ValueError):
        assess(record)


def test_failed_numerical_solve_prevents_convergence_verdict(record):
    record['scenarios']['theta=0.5']['cases']['12801x1281']['solver']['converged'] = False
    assert not assess(record)['passed']


def test_empty_record_list_cannot_be_combined():
    with pytest.raises(ValueError, match='completed record'):
        combine([])
