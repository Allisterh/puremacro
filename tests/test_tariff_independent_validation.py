"""Independent equations, equilibrium and expenditure-function tariff checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from puremacro.examples.first_brief_tariffs import load_bundled, solve_brief, national_record
from tools.reference_validation.tariff_reference import (
    LeontiefReference, audit_state, category_changes, resource_changes,
)


@pytest.fixture(scope='module')
def bundled():
    return load_bundled()


@pytest.fixture(scope='module')
def headline(bundled):
    return solve_brief(bundled=bundled)


def test_independent_equilibrium_and_all_final_uses(bundled, headline):
    calib = bundled[1]
    before_ref = LeontiefReference(calib)
    before = before_ref.solve()
    after = LeontiefReference(calib, tariff=.1).solve()
    i = list(calib.country_codes).index('USA')
    change = category_changes(before, after)
    for ref, native in [(before, headline.base), (after, headline.counterfactual)]:
        for key, field in [('p', 'p_sol'), ('y', 'y_sol'), ('w', 'w_sol'), ('c', 'c_sol'), ('P', 'Pfd_final')]:
            np.testing.assert_allclose(np.ravel(getattr(native, field), order='F'), np.ravel(ref[key]), rtol=1e-8, atol=1e-8)
    assert change[i, 0] == pytest.approx(headline.welfare.ev, abs=1e-5, rel=0)
    national = national_record(headline)
    assert change[i, 1] == pytest.approx(national['investment_change_base_prices_musd'], abs=1e-5, rel=0)
    assert change[i, 2] == pytest.approx(national['abroad_change_base_prices_musd'], abs=1e-5, rel=0)
    assert abs(resource_changes(before_ref, before, after).sum()) < 1e-5


@pytest.mark.parametrize('sigma', [0., 2.])
def test_independent_flow_budget_and_ev_evaluator(bundled, headline, sigma):
    sol = headline if sigma == 0 else solve_brief(sigma=sigma, bundled=bundled)
    before = audit_state(sol.calib, sol.base, sigma=sigma)
    after = audit_state(sol.calib, sol.counterfactual, sigma=sigma, tariff=.1)
    assert before['maximum_normalized_error'] < 1e-9
    assert after['maximum_normalized_error'] < 1e-9
    i = list(sol.calib.country_codes).index('USA')
    ev = before['P'][i, 0]*(after['c'][i, 0]-before['c'][i, 0])
    assert ev == pytest.approx(sol.welfare.ev, abs=1e-5, rel=0)


def test_frozen_checks_have_provenance_and_unchanged_thresholds():
    root = Path(__file__).resolve().parents[1]
    record = json.loads((root/'tests/fixtures/first_brief_tariffs/independent_validation.json').read_text(encoding='utf-8'))
    assert record['passed']
    assert record['tolerances'] == dict(ev_musd=1e-5, normalized_equilibrium=1e-9, scaled_state=1e-8, world_resource_musd=1e-5)
    for name, digest in record['source_sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name
    for result in record['closure_sensitivity'].values():
        assert result['passed']
        assert abs(result['world_factor_resource_total_musd']) < 1e-5
        assert result['us_final_use_change_musd'][0] > 0
        assert result['us_final_use_change_musd'][1] < 0
