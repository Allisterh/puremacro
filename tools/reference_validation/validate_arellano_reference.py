"""Independent equation-level Arellano benchmark, not a published-moments replication.

Equations: https://eqm.quantecon.org/arellano.html (Sargent and Stachurski).
This implementation uses assets B (negative for debt), a separate Bellman loop,
and no PureMacro discretization, policy, value or pricing helpers. Utilities use
1-1/c; the additive constant changes values but not decisions. The endowment
grid is normalized to stationary mean one, unlike some published calibrations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from datetime import date

import numpy as np
from scipy.special import ndtr


def reference(n_b: int, n_y: int = 9) -> dict:
    beta, reentry, r, rho, sigma = 0.953, 0.282, 0.01, 0.945, 0.025
    assets = np.linspace(-0.4, 0.4, n_b)
    z = np.linspace(-3*sigma/np.sqrt(1-rho*rho), 3*sigma/np.sqrt(1-rho*rho), n_y)
    edges = np.r_[-np.inf, (z[:-1]+z[1:])/2, np.inf]
    P = np.diff(ndtr((edges[None, :] - rho*z[:, None])/sigma), axis=1)
    stationary = np.full(n_y, 1/n_y)
    for _ in range(10000):
        new = stationary @ P
        if np.max(abs(new-stationary)) < 1e-15:
            break
        stationary = new
    y = np.exp(z)/(stationary @ np.exp(z))
    yd = np.minimum(y, 0.969*(stationary @ y))
    V = np.zeros((n_b, n_y))
    vd = np.zeros(n_y)
    q = np.full_like(V, 1/(1+r))
    for iteration in range(10000):
        ev = V @ P.T
        vd_new = 1-1/yd + beta*(reentry*V[n_b//2] + (1-reentry)*vd) @ P.T
        vc = np.empty_like(V)
        policy = np.empty_like(V)
        for j in range(n_y):
            consumption = y[j] + assets[:, None] - q[:, j][None, :]*assets[None, :]
            utility = np.full_like(consumption, -np.inf)
            feasible = consumption > 1e-6
            utility[feasible] = 1-1/consumption[feasible]
            objective = utility + beta*ev[:, j][None, :]
            choice = objective.argmax(axis=1)
            vc[:, j] = objective[np.arange(n_b), choice]
            policy[:, j] = assets[choice]
        V_new = np.maximum(vc, vd_new)
        default = vd_new[None, :] > vc
        q_new = (1-default.astype(float)) @ P.T/(1+r)
        error = max(np.max(abs(V_new-V)), np.max(abs(vd_new-vd)), np.max(abs(q_new-q)))
        V, vd = V_new, vd_new
        q = 0.6*q + 0.4*q_new
        if error < 1e-10:
            break
    else:
        raise RuntimeError(f"Reference failed to converge: {error}")
    return dict(b_grid=-assets[::-1], y_grid=y, p_trans=P, v_repay=vc[::-1],
                v_default=vd, q_bond=q[::-1], default_prob=default[::-1].astype(float),
                policy_b_prime=-policy[::-1])


def main():
    from puremacro.sovereign import solve_sovereign_default

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = []
    for n_b in (41, 81):
        oracle = reference(n_b)
        result = solve_sovereign_default(n_b=n_b, n_y=9, b_min=-0.4, b_max=0.4,
                                         sigma_eps=0, y_penalty_type="arellano", howard_steps=0,
                                         enforce_monotonicity=False, tol=1e-10, max_iter=10000)
        errors = {name: float(np.max(abs(getattr(result, name)-value))) for name, value in oracle.items()}
        np.savez_compressed(args.output / f"arellano_{n_b}.npz", **oracle)
        cases.append(dict(n_b=n_b, n_y=9, converged=bool(result.converged), max_abs_errors=errors))
    manifest = dict(scope="independent short-term deterministic-default equations on matched grids; not paper moments",
                    generated_on=date.today().isoformat(),
                    equations="https://eqm.quantecon.org/arellano.html",
                    original_paper="https://doi.org/10.1257/aer.98.3.690",
                    calibration=dict(beta=0.953, gamma=2, r_star=0.01, rho_y=0.945, sigma_y=0.025,
                                     theta=0.282, b_min=-0.4, b_max=0.4, sigma_eps=0), cases=cases,
                    fixtures_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in sorted(args.output.glob("arellano_*.npz"))})
    manifest["passed"] = all(c["converged"] and max(c["max_abs_errors"].values()) <= 1e-8 for c in cases)
    (args.output / "arellano_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest, indent=2))
    if not manifest["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
