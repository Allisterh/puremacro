"""Fixed-design refinement of the two grid-sensitive sovereign brief cases."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from puremacro.examples.first_brief_sovereign import (
    BASELINE, DESIGN, TOLERANCES, build_record, paired_change, source_hashes,
)

SCENARIOS = {'theta=0.5': {'theta': .5}, 'rho_y=0.97': {'rho_y': .97}}
GRIDS = [(6401, 641), (12801, 641), (6401, 1281), (12801, 1281)]


def assess(record):
    """Recompute the unchanged registered decision from complete per-seed data."""
    if record.get('schema') != 1 or record.get('baseline') != BASELINE:
        raise ValueError('record does not match the registered schema and baseline')
    if not record.get('scenarios'):
        raise ValueError('at least one registered scenario is required')
    if record['design'] != DESIGN or record['tolerances'] != TOLERANCES:
        raise ValueError('record does not match the registered design and tolerances')
    if record['grids'] != [list(g) for g in GRIDS]:
        raise ValueError('record does not contain the registered crossed grid ladder')
    passed = True
    for name, scenario in record['scenarios'].items():
        if name not in SCENARIOS or scenario['overrides'] != SCENARIOS[name]:
            raise ValueError('record has an unregistered scenario')
        cases = scenario['cases']
        if set(cases) != {f'{b}x{y}' for b, y in GRIDS}:
            raise ValueError(f'{name} is missing one or more required grids')
        for b, y in GRIDS:
            case = cases[f'{b}x{y}']
            if (case['n_b'], case['n_y']) != (b, y) or case['simulation']['seeds'] != DESIGN['seeds']:
                raise ValueError('grid or seed identity differs from its registered case')
            expected = {**BASELINE, **scenario['overrides'], 'n_b': b, 'n_y': y}
            if any(case['parameters'].get(key) != value for key, value in expected.items()):
                raise ValueError('case parameters differ from the registered calibration')
            for moment in TOLERANCES:
                values = np.asarray(case['simulation']['per_seed'][moment], dtype=float)
                if values.shape != (len(DESIGN['seeds']),) or not np.all(np.isfinite(values)):
                    raise ValueError('each moment requires one finite observation per registered seed')
        # build_record's two-grid display is not the validation criterion for
        # this crossed ladder. Require both final refinements explicitly.
        scenario.pop('grid_change', None)
        checks = {}
        for label, coarse, fine in (
            ('assets_at_641_income_states', '6401x641', '12801x641'),
            ('final_asset_refinement', '6401x1281', '12801x1281'),
            ('final_income_refinement', '12801x641', '12801x1281'),
        ):
            checks[label] = dict(coarse=coarse, fine=fine, **paired_change(cases[coarse], cases[fine]))
        scenario['comparisons'] = checks
        scenario['passed'] = all(checks[k]['passed'] for k in ('final_asset_refinement', 'final_income_refinement'))
        for case in cases.values():
            solver = case['solver']
            scenario['passed'] &= (solver['converged'] and solver['monotone_certification_gap'] <= 1e-11
                                    and solver['exhaustive_bellman_residual'] <= 1e-8)
        passed &= scenario['passed']
        print(name, {k: v['passed'] for k, v in checks.items()}, flush=True)
    record['passed'] = bool(passed)
    record['status'] = 'passed' if passed else 'convergence_not_established'
    record['assessment_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return record


def combine(paths):
    """Combine separately run scenarios without rewriting historical provenance."""
    if not paths:
        raise ValueError('at least one completed record is required')
    records = [json.loads(p.read_text(encoding='utf-8')) for p in paths]
    result = copy.deepcopy(records[0])
    result['scenarios'] = {}
    result['load_averages'] = {}
    result['input_record_sha256'] = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    for record in records:
        for key in ('schema', 'source_sha256', 'design', 'baseline', 'tolerances', 'grids', 'versions', 'package_version'):
            if record[key] != result[key]:
                raise ValueError(f'cannot combine incompatible {key}')
        for name, scenario in record['scenarios'].items():
            # A first process may have started another scenario before stopping.
            # Empty entries contain no observations and carry no evidentiary value.
            if not scenario['cases']:
                continue
            if name in result['scenarios']:
                raise ValueError(f'duplicate scenario {name}; provide disjoint complete scenario records')
            result['scenarios'][name] = copy.deepcopy(scenario)
        result['load_averages'].update(record['load_averages'])
    if set(result['scenarios']) != set(SCENARIOS):
        raise ValueError('both registered scenarios are required')
    peaks = [r['peak_rss_gb'] for r in records if r['peak_rss_gb'] is not None]
    result['peak_rss_gb'] = max(peaks) if peaks else None
    return assess(result)


def run(output: Path, scenarios=None):
    output.mkdir(parents=True, exist_ok=True)
    scenarios = SCENARIOS if scenarios is None else scenarios
    protocol = dict(design=DESIGN, scenarios=scenarios, grids=GRIDS,
                     tolerances=TOLERANCES, source_sha256=source_hashes(),
                     rule='abs(paired mean change) + 95% t half-width <= tolerance')
    path = output / 'protocol.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != json.loads(json.dumps(protocol)):
        raise ValueError('existing protocol differs; use a new output directory')
    path.write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')
    record = assess(build_record(GRIDS, design=DESIGN, scenarios=scenarios,
                                 partial=output / 'sensitive_grid.partial.json',
                                 log=lambda message: print(message, flush=True)))
    (output / 'sensitive_grid_convergence.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scenario', choices=list(SCENARIOS))
    parser.add_argument('--combine', nargs='+', type=Path,
                        help='combine disjoint completed scenario records into --output directory')
    args = parser.parse_args()
    selected = None if args.scenario is None else {args.scenario: SCENARIOS[args.scenario]}
    if args.combine:
        result = combine(args.combine)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'sensitive_grid_convergence.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    else:
        result = run(args.output, selected)
    raise SystemExit(0 if result['passed'] else 1)
