"""Bounded Monte Carlo evidence for the explicitly restricted bootstrap modes.

Run with capped BLAS threads. JSON reports Monte Carlo uncertainty over
independent DGP replications; GVAR rows within a replication are not treated
as independent Bernoulli trials. No thresholds or block sizes are optimized.
This is evidence, not a release gate asserting universal calibration.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import time

import numpy as np
import pandas as pd

from puremacro.spatial import SpatialLPBootstrap, knn_weights, spatial_lp
from puremacro.var import gvar


def _summary(values):
    x = np.asarray(values, dtype=float)
    mean = x.mean(axis=0)
    se = x.std(axis=0, ddof=1) / np.sqrt(len(x))
    n = len(x)
    binary = np.all((x == 0) | (x == 1), axis=0)
    denominator = 1 + 1.96 ** 2 / n
    centre = (mean + 1.96 ** 2 / (2 * n)) / denominator
    radius = 1.96 * np.sqrt(mean * (1 - mean) / n + 1.96 ** 2 / (4 * n ** 2)) / denominator
    lo = np.where(binary, centre - radius, np.maximum(0, mean - 1.96 * se))
    hi = np.where(binary, centre + radius, np.minimum(1, mean + 1.96 * se))
    return {"mean": mean.tolist(), "mc_se": se.tolist(),
            "mc_95_interval": np.stack([lo, hi], axis=-1).tolist(),
            "interval_method": "Wilson for binary replication outcomes; normal cluster-mean approximation otherwise"}


def spatial_case(replications, draws, seed, *, N, T, gamma, block_length):
    rng = np.random.default_rng(941)
    ids = [f"R{i}" for i in range(N)]
    coords = pd.DataFrame({"lat": rng.uniform(30, 45, N), "lon": rng.uniform(-110, -75, N)}, index=ids)
    W = knn_weights(coords, 3)
    matrix = W.to_dense()
    metrics = {k: [] for k in ("asymptotic_reject", "bootstrap_reject", "bootstrap_coverage",
                               "simultaneous_coverage", "joint_reject", "width")}
    failures = []
    for r in range(replications):
        rng = np.random.default_rng(seed + r)
        x = rng.standard_normal((N, T + 100))
        eta = (np.eye(N) + 0.4 * matrix) @ rng.standard_normal(x.shape)
        error = np.zeros_like(eta)
        for t in range(1, error.shape[1]):
            error[:, t] = 0.4 * error[:, t - 1] + eta[:, t]
        dy = 0.5 * x + gamma * matrix @ x + 0.4 * error
        y = np.cumsum(dy, axis=1)[:, 100:]
        x = x[:, 100:]
        frame = pd.DataFrame({"y": y.ravel(), "x": x.ravel()},
                             index=pd.MultiIndex.from_product([ids, range(T)], names=["code", "date"]))
        try:
            result = spatial_lp(frame, "y", "x", W, horizons=range(6), n_lags=2,
                                lagged_outcome="difference", alpha=0.05,
                                bootstrap=SpatialLPBootstrap(draws, block_length, True, seed + 100000 + r))
            b = result.bootstrap_result
            tab = b.table.xs("indirect", level="label")
            metrics["asymptotic_reject"].append(np.abs(result["t_indirect"].to_numpy()) > 1.959963984540054)
            metrics["bootstrap_reject"].append(tab["p_value"].to_numpy() < 0.05)
            metrics["bootstrap_coverage"].append((tab["lo"].to_numpy() <= gamma) & (tab["hi"].to_numpy() >= gamma))
            metrics["joint_reject"].append(b.joint_p_value < 0.05)
            truth = np.array([0.5, gamma, 0.5 + gamma] * 6)
            metrics["simultaneous_coverage"].append(bool(np.all((b.table["simultaneous_lo"] <= truth) & (b.table["simultaneous_hi"] >= truth))))
            metrics["width"].append((tab["hi"] - tab["lo"]).to_numpy())
        except (ValueError, np.linalg.LinAlgError) as exc:
            failures.append({"replication": r, "error": str(exc)})
    out = {k: _summary(v) for k, v in metrics.items() if k != "width" and v}
    if metrics["width"]:
        out["median_width"] = np.median(metrics["width"], axis=0).tolist()
    return {"design": {"N": N, "T": T, "gamma": gamma, "block_length": block_length,
                        "n_lags": 2, "lagged_outcome": "difference", "error_ar": 0.4,
                        "spatial_error": "(I+0.4W) innovations", "horizons": list(range(6))},
            "successful_replications": len(metrics["joint_reject"]), "failures": failures, "results": out}


def gvar_case(replications, draws, seed, *, T, rho, leak, block_length):
    countries = ["A", "B", "C"]
    weights = pd.DataFrame([[0, .6, .4], [.5, 0, .5], [.4, .6, 0]], index=countries, columns=countries)
    original, boot, family = [], [], []
    failures = []
    for r in range(replications):
        rng = np.random.default_rng(seed + r)
        u = rng.standard_normal((T, 6))
        X = np.zeros_like(u)
        for t in range(1, T):
            X[t] = rho * X[t - 1] + u[t]
            if leak:
                X[t, 2:] += leak * u[t - 1, 0]
                if rho != 1:
                    X[t, :2] = .5 * X[t - 1, :2] + u[t, :2]
        frames = {c: pd.DataFrame(X[:, 2 * j:2 * j + 2], columns=["y", "r"]) for j, c in enumerate(countries)}
        try:
            result = gvar(frames, weights, p=1, q=1, stability_warn=False)
            b = result.independent_null_bootstrap(null="independent_countries",
                    integration="i1" if rho == 1 else "stationary", n_boot=draws,
                    block_length=block_length, seed=seed + 100000 + r)
            selection = np.array([True, True, False, False, False, False]) if leak else np.ones(6, dtype=bool)
            original.append((result.weak_exogeneity.table["p_value"].to_numpy()[selection] < .05).mean())
            boot.append((b.table["p_independent_null"].to_numpy()[selection] < .05).mean())
            family.append(bool((b.table["p_holm_independent_null"].to_numpy() < .05).any()))
        except (ValueError, np.linalg.LinAlgError) as exc:
            failures.append({"replication": r, "error": str(exc)})
    return {"design": {"T": T, "rho": rho, "leak": leak, "block_length": block_length,
                        "countries": 3, "variables_per_country": 2, "p": 1, "q": 1},
            "successful_replications": len(boot), "failures": failures,
            "results": {"asymptotic_reject": _summary(original), "independent_null_reject": _summary(boot),
                        "holm_family_reject": _summary(family)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=["spatial", "gvar"], required=True)
    parser.add_argument("--replications", type=int, default=200)
    parser.add_argument("--draws", type=int, default=399)
    parser.add_argument("--seed", type=int, default=812000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.replications < 2 or args.draws < 19:
        parser.error("at least two replications and 19 bootstrap draws required")
    configurations = (
        [dict(N=20, T=50, gamma=0, block_length=4), dict(N=20, T=50, gamma=0, block_length=8),
         dict(N=40, T=50, gamma=0, block_length=4), dict(N=20, T=100, gamma=0, block_length=6),
         dict(N=20, T=50, gamma=.25, block_length=4)] if args.domain == "spatial" else
        [dict(T=200, rho=1., leak=0, block_length=1), dict(T=300, rho=1., leak=0, block_length=1),
         dict(T=200, rho=.5, leak=0, block_length=1), dict(T=200, rho=1., leak=0, block_length=4),
         dict(T=200, rho=1., leak=1., block_length=1),
         dict(T=250, rho=.9, leak=2., block_length=1)])
    report = {"domain": args.domain, "replications": args.replications, "bootstrap_draws": args.draws,
              "seed": args.seed, "python": platform.python_version(), "numpy": np.__version__,
              "inference_status": "design-specific Monte Carlo evidence; no universal calibration claim",
              "uncertainty_unit": "independent DGP replication; correlated GVAR tests averaged within replication",
              "cases": []}
    start = time.perf_counter()
    for config in configurations:
        fn = spatial_case if args.domain == "spatial" else gvar_case
        case = fn(args.replications, args.draws, args.seed, **config)
        report["cases"].append(case)
        report["elapsed_seconds"] = time.perf_counter() - start
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(case), flush=True)


if __name__ == "__main__":
    main()
