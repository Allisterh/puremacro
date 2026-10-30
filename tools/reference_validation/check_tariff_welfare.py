"""Reproduce tariff consumption EV and explain the other final-use changes."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy

from puremacro.examples.first_brief_tariffs import load_bundled, solve_brief
from tools.reference_validation.tariff_reference import (
    LeontiefReference, audit_state, category_changes, resource_changes,
)


def check(output: Path):
    bundled = load_bundled()
    icio, calib = bundled
    country = list(calib.country_codes).index('USA')
    record = dict(schema=1, scenario=dict(importer='USA', exporter='CHN', tariff=.1,
                                          sectors='all 20 modeled sectors, including services'),
                  categories=list(icio.fd_codes), unit='million USD at baseline prices',
                  versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
                  tolerances=dict(ev_musd=1e-5, normalized_equilibrium=1e-9,
                                  scaled_state=1e-8, world_resource_musd=1e-5),
                  source_sha256={}, native={}, closure_sensitivity={})
    for name in ('tools/reference_validation/tariff_reference.py',
                 'tools/reference_validation/check_tariff_welfare.py',
                 'puremacro/trade/_accounting.py', 'puremacro/trade/solver.py',
                 'puremacro/examples/first_brief_tariffs.py'):
        record['source_sha256'][name] = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    record['table_sha256'] = hashlib.sha256(np.asarray(icio.matrix, dtype='<f8').tobytes()).hexdigest()
    passed = True
    native_headline = None
    for sigma in (0., 2.):
        sol = solve_brief(sigma=sigma, bundled=bundled)
        before = audit_state(calib, sol.base, sigma=sigma)
        after = audit_state(calib, sol.counterfactual, sigma=sigma, tariff=.1)
        changes = category_changes(before, after)
        ev_error = abs(float(changes[country, 0]) - sol.welfare.ev)
        ok = ev_error <= 1e-5 and max(before['maximum_normalized_error'], after['maximum_normalized_error']) <= 1e-9
        record['native'][str(int(sigma))] = dict(
            consumption_ev_musd=float(changes[country, 0]), ev_error_musd=ev_error,
            baseline_errors=before['errors'], counterfactual_errors=after['errors'],
            us_final_use_change_musd=changes[country].tolist(),
            world_final_use_change_musd=changes.sum(axis=0).tolist(),
            us_quantity_change_pct=(100*(after['c'][country]/before['c'][country]-1)).tolist(),
            us_price_change_pct=(100*(after['P'][country]/before['P'][country]-1)).tolist(),
            us_nominal_income_change_pct=float(100*(after['income'][country]/before['income'][country]-1)),
            passed=bool(ok))
        passed &= ok
        if sigma == 0:
            native_headline = sol
    for label, taxes, foreign in [('headline', True, True), ('no_domestic_taxes', False, True),
                                  ('zero_foreign_balances', True, False), ('neither', False, False)]:
        reference = LeontiefReference(calib, domestic_taxes=taxes, foreign_balances=foreign)
        before = reference.solve()
        after = LeontiefReference(calib, tariff=.1, domestic_taxes=taxes, foreign_balances=foreign).solve()
        changes = category_changes(before, after)
        resources = resource_changes(reference, before, after)
        ok = abs(float(resources.sum())) <= 1e-5
        entry = dict(domestic_taxes=taxes, foreign_balances=foreign,
                     us_final_use_change_musd=changes[country].tolist(),
                     world_final_use_change_musd=changes.sum(axis=0).tolist(),
                     world_factor_resource_change_musd=resources.sum(axis=0).tolist(),
                     world_factor_resource_total_musd=float(resources.sum()),
                     maximum_factor_gap=float(max(np.max(abs(before['factor_gap'])), np.max(abs(after['factor_gap'])))))
        if label == 'headline':
            assert native_headline is not None
            errors = {}
            for prefix, ref, native in [('baseline', before, native_headline.base),
                                        ('counterfactual', after, native_headline.counterfactual)]:
                for key, field in [('p', 'p_sol'), ('y', 'y_sol'), ('w', 'w_sol'), ('c', 'c_sol'),
                                   ('P', 'Pfd_final'), ('transfers', 'T_sol')]:
                    expected = np.ravel(ref[key])
                    actual = np.ravel(getattr(native, field), order='F')
                    errors[f'{prefix}_{key}'] = float(np.max(abs(expected-actual)/np.maximum(1., abs(expected))))
            ev_error = abs(float(changes[country, 0])-native_headline.welfare.ev)
            entry.update(native_scaled_errors=errors, native_ev_error_musd=ev_error)
            ok &= max(errors.values()) <= 1e-8 and ev_error <= 1e-5
        entry['passed'] = bool(ok)
        record['closure_sensitivity'][label] = entry
        passed &= ok
    record['passed'] = bool(passed)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    result = check(parser.parse_args().output)
    print(json.dumps(dict(passed=result['passed'], native=result['native']), indent=2))
    raise SystemExit(0 if result['passed'] else 1)
