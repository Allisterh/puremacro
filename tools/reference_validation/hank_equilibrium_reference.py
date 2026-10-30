"""Author-code benchmark for the ABRS (2021) one-asset HANK equilibrium.

Validation-only: install the pinned sequence-jacobian checkout in a separate
venv. Equations are the authors' examples/hank.py; parameters and equilibrium
targets are from notebooks/hank.ipynb at the same commit. No PureMacro model
or household routine is called to generate these reference arrays.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time

import numpy as np

REFERENCE_COMMIT = 'eb4ac4c3157357a4cacbea4b5a7a281f5b48cab9'
CALIBRATION = dict(r=.005, rstar=.005, eis=.5, frisch=.5, B=5.6, mu=1.2,
                   rho_s=.966, sigma_s=.5, kappa=.1, phi=1.5, Y=1., Z=1., pi=0.,
                   nS=7, amax=150., nA=500)


def generate(n_a=500, n_e=7, horizon=300, magnitude=-.0025, rho=.61):
    from sequence_jacobian import create_model
    from sequence_jacobian.examples import hank as ref
    from scipy.optimize import root

    hh = ref.hh.add_hetinputs([ref.transfers, ref.wages, ref.make_grids])
    hh = hh.add_hetoutputs([ref.labor_supply])
    ss_model = create_model([hh, ref.firm, ref.monetary, ref.fiscal,
                             ref.mkt_clearing, ref.nkpc_ss], name='ABRS_reference_ss')
    model = create_model([hh, ref.firm, ref.monetary, ref.fiscal,
                          ref.mkt_clearing, ref.nkpc], name='ABRS_reference')
    calibration = dict(CALIBRATION, nA=n_a, nS=n_e)
    options = {'hh': {'backward_tol': 1e-12, 'forward_tol': 1e-13}}
    # This pinned Block.solve_steady_state overwrites nested block options.
    # Call its steady-state evaluator directly so the household tolerances
    # actually apply during calibration as well as the final evaluation.
    def residual(x):
        out = ss_model.steady_state(dict(calibration, beta=x[0], vphi=x[1]), options=options)
        return [out['asset_mkt'], out['labor_mkt']]
    fit = root(residual, [.986, .8], tol=1e-10)
    if np.max(np.abs(residual(fit.x))) > 1e-9:
        raise RuntimeError(f'author-code calibration did not converge: {fit.message}')
    ss0 = ss_model.steady_state(dict(calibration, beta=fit.x[0], vphi=fit.x[1]), options=options)
    ss = model.steady_state(ss0, options=options)
    unknowns, targets = ['pi', 'w', 'Y'], ['nkpc_res', 'asset_mkt', 'labor_mkt']
    inputs = {'rstar': magnitude * rho ** np.arange(horizon)}
    options_j = {'hh': {'h': 1e-5, 'twosided': True}}
    linear = model.solve_impulse_linear(ss, unknowns, targets, inputs, options=options_j)
    nonlinear = model.solve_impulse_nonlinear(ss, unknowns, targets, inputs,
                                              options=options_j, tol=1e-10, verbose=False)
    opposite = model.solve_impulse_nonlinear(ss, unknowns, targets,
                                             {'rstar': -inputs['rstar']},
                                             options=options_j, tol=1e-10, verbose=False)
    inner = ss.internals['hh']
    arrays = {k: np.asarray(inner[k]).T for k in ('a', 'c', 'n', 'Va', 'D')}
    arrays.update(a_grid=inner['a_grid'], e_grid=inner['e_grid'], Pi=inner['Pi'])
    names = ('pi', 'w', 'Y', 'C', 'A', 'NE', 'r', 'Div', 'Tax', 'goods_mkt',
             'labor_mkt', 'asset_mkt', 'nkpc_res')
    for mode, out in [('linear', linear), ('nonlinear', nonlinear), ('opposite', opposite)]:
        arrays.update({f'{mode}_{k}': np.asarray(out[k]) for k in names})
    scalars = {k: float(ss[k]) for k in ('beta', 'vphi', 'C', 'A', 'NE', 'r', 'w', 'Y', 'Div', 'Tax')}
    return arrays, dict(calibration=calibration, steady_state=scalars,
                        horizon=horizon, magnitude=magnitude, rho=rho,
                        steady_residuals={k: float(ss[k]) for k in ('asset_mkt', 'labor_mkt', 'goods_mkt')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--n-a', type=int, default=500)
    parser.add_argument('--n-e', type=int, default=7)
    parser.add_argument('--horizon', type=int, default=300)
    args = parser.parse_args()
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.reference_repo,
                                        text=True).strip()
    if revision != REFERENCE_COMMIT:
        raise ValueError(f'expected reference {REFERENCE_COMMIT}; found {revision}')
    import sequence_jacobian
    installed = Path(sequence_jacobian.__file__).parent
    source = args.reference_repo / 'src' / 'sequence_jacobian'
    hashes = {}
    for rel in ('examples/hank.py', 'hetblocks/hh_labor.py', 'blocks/het_block.py',
                'blocks/block.py', 'utilities/discretize.py'):
        data = (source / rel).read_bytes()
        if data != (installed / rel).read_bytes():
            raise ValueError(f'installed reference differs from pinned checkout: {rel}')
        hashes[rel] = hashlib.sha256(data).hexdigest()
    start = time.perf_counter()
    arrays, report = generate(args.n_a, args.n_e, args.horizon)
    args.output.mkdir(parents=True, exist_ok=True)
    stem = f'hank_equilibrium_{args.n_a}_{args.n_e}_{args.horizon}'
    path = args.output / f'{stem}.npz'
    np.savez_compressed(path, **arrays)
    report.update(reference_commit=revision, source_sha256=hashes,
                  reference_url=f'https://github.com/shade-econ/sequence-jacobian/tree/{revision}',
                  notebook='notebooks/hank.ipynb',
                  fixture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  seconds=time.perf_counter()-start,
                  versions={k: importlib.metadata.version(k) for k in ('numpy', 'scipy', 'sequence-jacobian')})
    (args.output / f'{stem}.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
