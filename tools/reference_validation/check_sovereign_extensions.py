"""Freeze independent small-grid sovereign/collateral equation checks."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy

from puremacro.sovereign import solve_sovereign_default, solve_sudden_stop, evaluate_debt_restructuring
from tools.reference_validation.sovereign_extensions import collateral, long_term


def check(output):
    record = dict(schema=1, scope='small-grid equations; no published-moment replication',
                  versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
                  tolerances=dict(value_price=1e-7, accounting=1e-8),
                  source_sha256={}, long_term=[], collateral=[], restructuring=[])
    for filename in ('puremacro/sovereign/arellano.py', 'puremacro/sovereign/_results.py',
                     'puremacro/sovereign/mendoza.py', 'puremacro/sovereign/restructuring.py',
                     'puremacro/sovereign/measurement.py',
                     'tools/reference_validation/sovereign_extensions.py',
                     'tools/reference_validation/check_sovereign_extensions.py'):
        record['source_sha256'][filename] = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    passed = True
    for delta, coupon in [(.2, .03), (.5, .01), (1., 0.)]:
        params = dict(n_b=17, n_y=5, b_min=-.1, b_max=.25, delta=delta, kappa=coupon,
                      sigma_eps=.01, enforce_monotonicity=False, howard_steps=0, tol=1e-10, max_iter=10000)
        res = solve_sovereign_default(**params)
        ref = long_term(res.b_grid, res.y_grid, res.p_trans, res.metadata['y_default_vec'], delta=delta, coupon=coupon)
        fields = [('v_repay', 'v_repay'), ('v_default', 'v_default'), ('q_bond', 'q'),
                  ('default_prob', 'default'), ('policy_b', 'modal_policy')]
        errors = {field: float(np.max(abs(getattr(res, field)-ref[key]))) for field, key in fields}
        ok = res.converged and max(errors.values()) <= 1e-7
        if delta < .999:
            errors['choice_probabilities'] = float(np.max(abs(res.metadata['debt_choice_probabilities']-ref['probabilities'])))
            ok &= errors['choice_probabilities'] <= 1e-7
        record['long_term'].append(dict(parameters=params, errors=errors,
            reference_residual=ref['residual'], deterministic_optimality_gap=ref['deterministic_optimality_gap'], passed=bool(ok)))
        passed &= ok
        if delta == .2:
            for extension in (0., .5):
                changed = evaluate_debt_restructuring(res, haircut=.2, maturity_extension=extension,
                                                       b_distress=.2, y_distress=1., horizon=3)
                counter = long_term(res.b_grid, res.y_grid, res.p_trans, res.metadata['y_default_vec'], delta=delta/(1+extension))
                ib, iy = np.argmin(abs(res.b_grid-changed.restructured_debt)), np.argmin(abs(res.y_grid-1.))
                err = abs(changed.post_default_prob-counter['default'][ib, iy])
                record['restructuring'].append(dict(haircut=.2, extension=extension, default_probability_error=float(err), passed=bool(err<=1e-7)))
                passed &= err <= 1e-7
    for share in (.3, .9):
        params = dict(n_b=7, n_k=6, n_z=3, b_max=1.2, kappa_collateral=share, tol=1e-10, max_iter=2000)
        res = solve_sudden_stop(**params)
        ref = collateral(res.b_grid, res.k_grid, res.z_grid, res.p_trans, collateral_share=share)
        errors = {field: float(np.max(abs(getattr(res, field)-ref[field]))) for field in ('policy_b', 'policy_k', 'policy_c')}
        errors['value'] = float(np.max(abs(res.metadata['value_function']-ref['value'])))
        ok = res.converged and max(errors.values()) <= 1e-7 and res.metadata['bellman_residual'] <= 1e-8
        record['collateral'].append(dict(parameters=params, errors=errors,
            bellman_residual=res.metadata['bellman_residual'], crisis_probability=res.crisis_probability,
            grid_binding_fraction=res.collateral_binding_fraction,
            amplification_status=res.metadata['amplification_status'], passed=bool(ok)))
        passed &= ok
    record['passed'] = bool(passed)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    result = check(parser.parse_args().output)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
