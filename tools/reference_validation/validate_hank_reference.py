"""Compare the one-asset household block with the authors' SSJ implementation.

Install sequence-jacobian from the pinned commit below in a validation-only
environment. No reference dependency is needed to run the frozen-fixture test.
This compares household policies, distributions and Jacobians on matched grids;
it is not a replication of a published HANK calibration or aggregate-risk model.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from datetime import date

import numpy as np

from puremacro.models.hank_sequence_space import solve_hank_sequence_space

REFERENCE_COMMIT = "eb4ac4c3157357a4cacbea4b5a7a281f5b48cab9"


def income(w, r, B, e_grid):
    # PureMacro's balanced-budget closure holds government debt fixed.
    y = (w - r * B) * e_grid
    return y


def generate(n_a: int, horizon: int) -> tuple[dict, dict]:
    from sequence_jacobian.hetblocks.hh_sim import hh

    direct_url = importlib.metadata.distribution("sequence-jacobian").read_text("direct_url.json")
    revision = json.loads(direct_url or "{}").get("vcs_info", {}).get("commit_id")
    if revision != REFERENCE_COMMIT:
        raise RuntimeError(f"Install the pinned reference commit {REFERENCE_COMMIT}; found {revision}")
    native = solve_hank_sequence_space(T=horizon, n_a=n_a, a_max=100.0)
    household = hh.add_hetinputs([income])
    calibration = dict(a_grid=native.asset_grid, e_grid=np.array([0.5, 1.5]),
                       Pi=np.array([[0.9, 0.1], [0.1, 0.9]]),
                       r=0.01, w=1.0, B=native.government_debt, beta=0.985, eis=1.0)
    ss = household.steady_state(calibration, backward_tol=1e-12, forward_tol=1e-13)
    jac = household.jacobian(ss, inputs=["w", "r"], outputs=["C"], T=horizon,
                             h=1e-4, twosided=False)
    internals = ss.internals[household.name]
    reference = dict(policy_c=internals["c"].T, policy_a=internals["a"].T,
                     distribution=internals["D"].T, jacobian_w=jac["C"]["w"],
                     jacobian_r=jac["C"]["r"], consumption=np.array(ss["C"]))
    actual = dict(policy_c=native.policy_c, policy_a=native.policy_a,
                  distribution=native.distribution,
                  jacobian_w=native.fake_news(T=horizon, shock_input="w").jacobian,
                  jacobian_r=native.fake_news(T=horizon, shock_input="r").jacobian,
                  consumption=np.array(native.steady_state_consumption))
    report = dict(n_a=n_a, horizon=horizon, government_debt=native.government_debt,
                  steady_state_converged=bool(native.ss_converged),
                  upper_grid_mass=float(native.distribution[-1].sum()),
                  reference_assets_minus_debt=float(ss["A"]-native.government_debt),
                  max_abs_errors={k: float(np.max(abs(actual[k]-v))) for k, v in reference.items()})
    return reference, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    for n_a in (50, 100):
        reference, report = generate(n_a, 24)
        np.savez_compressed(args.output / f"hank_ssj_{n_a}.npz", **reference)
        reports.append(report)
    manifest = dict(scope="matched-grid one-asset household block; not full published HANK replication",
                    generated_on=date.today().isoformat(),
                    reference_commit=REFERENCE_COMMIT,
                    reference_url=f"https://github.com/shade-econ/sequence-jacobian/tree/{REFERENCE_COMMIT}",
                    calibration=dict(beta=0.985, gamma=1.0, r_ss=0.01, w_ss=1.0, a_max=100.0),
                    versions={p: importlib.metadata.version(p) for p in ("numpy", "scipy", "sequence-jacobian")},
                    cases=reports,
                    fixtures_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in sorted(args.output.glob("hank_ssj_*.npz"))})
    bounds = dict(policy_c=1e-8, policy_a=1e-8, distribution=1e-8,
                  consumption=1e-8, jacobian_w=1e-7, jacobian_r=5e-4)
    manifest["passed"] = all(r["steady_state_converged"] and r["upper_grid_mass"] < 1e-10
                             and abs(r["reference_assets_minus_debt"]) < 1e-8
                             and all(e <= bounds[k] for k, e in r["max_abs_errors"].items())
                             for r in reports)
    (args.output / "hank_ssj_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    if not manifest["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
