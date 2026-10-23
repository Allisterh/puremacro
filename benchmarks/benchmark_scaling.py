"""Reproducible offline scaling measurements; no hardware-independent speed claims.

Usage: python benchmarks/benchmark_scaling.py --profile smoke --repetitions 2 --output report.json
Runtime and traced-allocation measurements reuse benchmark_continuous_vfi's engine.
Every workload also has an explicit numerical check. Exit 1 means a failed check
or workload; exit 2 is invalid usage. This is a source-checkout development tool.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import scipy
from scipy import sparse

from benchmarks.benchmark_continuous_vfi import measure_memory_kib, measure_runtime_ms
from puremacro import __version__


@dataclass
class Workload:
    family: str
    name: str
    dimensions: dict[str, int]
    solve: Callable[[], Any]
    check: Callable[[Any], dict[str, Any]]
    scope: str


def _accuracy(error: float, tolerance: float, kind: str, **extra) -> dict[str, Any]:
    return {"passed": bool(np.isfinite(error) and error <= tolerance),
            "metric": kind, "error": float(error) if np.isfinite(error) else None,
            "tolerance": tolerance, **extra}


def _pf(T: int, backend: str) -> Workload:
    from puremacro.dsge import solve_perfect_foresight

    n = 3
    t = np.arange(1, T + 1)[:, None]
    expected = 1 + 0.08 * np.sin(np.pi * t / (T + 1)) * np.array([[1, 0.8, 0.6]])
    padded = np.vstack([np.ones(n), expected, np.ones(n)])
    def equation(lead, current, lag, shock):
        return current**2 - 0.2 * lead - 0.3 * lag - 0.05 * np.roll(current, 1) - 0.45 - shock
    shocks = np.array([equation(padded[j + 2], padded[j + 1], padded[j], 0) for j in range(T)])
    def solve():
        return solve_perfect_foresight(equation, y_init=np.ones(n), y_ss=np.ones(n),
                                      periods=T, shocks=shocks, linear_solver=backend,
                                      method="complex", tol=1e-10)
    def check(result):
        actual = result.path.to_numpy()
        full = np.vstack([np.ones(n), actual, np.ones(n)])
        residual = max(float(np.max(np.abs(equation(full[j + 2], full[j + 1], full[j], shocks[j]))))
                       for j in range(T))
        error = max(float(np.max(np.abs(actual - expected))), residual)
        report = _accuracy(error, 1e-8, "manufactured_solution_and_equation_max_abs",
                           residual_norm=residual, converged=bool(result.converged))
        report["passed"] &= bool(result.converged)
        return report
    return Workload("pf", f"pf_{backend}_T{T}", {"horizon": T, "variables": n}, solve, check,
                    "Complete nonlinear solve with a manufactured exact path; same equations for both backends.")


def _hank(T: int) -> Workload:
    from puremacro.models.hank_sequence_space import solve_hank_sequence_space

    def solve():
        return solve_hank_sequence_space(T=T, n_a=30)
    def check(r):
        # Independent recomposition of the household response and NK equations.
        y, c, pi, rate = r.irf_output, r.irf_consumption, r.irf_inflation, r.irf_rate
        lead_pi = np.r_[pi[1:], 0.0]
        monetary = 0.0025 * 0.7 ** np.arange(T)
        errors = [np.max(np.abs(c - r.jacobian_c_r @ rate - r.jacobian_c_y @ y)),
                  np.max(np.abs(y - c)), np.max(np.abs(pi - 0.985 * lead_pi - 0.1 * y)),
                  np.max(np.abs(rate - 1.5 * pi + lead_pi - monetary)),
                  abs(float(r.distribution.sum()) - 1)]
        report = _accuracy(float(max(errors)), 1e-8, "linear_GE_equations_and_distribution_mass",
                           household_converged=bool(r.ss_converged))
        report["passed"] &= bool(r.ss_converged)
        return report
    return Workload("hank", f"hank_T{T}", {"horizon": T, "asset_nodes": 30}, solve, check,
                    "Built-in one-asset stationary household, Jacobians and linear GE. No grid-accuracy claim.")


def _var(draws: int, seed: int) -> Workload:
    from puremacro.var.bootstrap import bootstrap_bands
    from puremacro.var.estimate import estimate_var

    rng = np.random.default_rng(seed)
    data = np.zeros((160, 2))
    for t in range(1, len(data)):
        data[t] = np.array([[0.5, 0.1], [-0.05, 0.4]]) @ data[t-1] + rng.normal(size=2)
    def identify(A, covariance):
        return np.linalg.cholesky(covariance)
    def solve():
        return bootstrap_bands(data, 2, identify, horizon=8, n_boot=draws,
                               rng=np.random.default_rng(seed), n_jobs=1)
    # An independently reconstructed first recursive draw anchors RNG/timing.
    A, intercept, _, residuals, _ = estimate_var(data, 2)
    indices = np.random.default_rng(seed).integers(0, len(residuals), size=(draws, len(residuals)))
    sample = data.copy()
    for t in range(2, len(data)):
        sample[t] = intercept + sum(A[j] @ sample[t-j-1] for j in range(2)) + residuals[indices[0, t-2]]
    A_ref, _, covariance, _, _ = estimate_var(sample, 2)
    reference = np.zeros((9, 2, 2))
    reference[0] = np.linalg.cholesky(covariance)
    for h in range(1, 9):
        reference[h] = sum(A_ref[j] @ reference[h-j-1] for j in range(min(h, 2)))
    def check(result):
        finite = all(np.isfinite(result[key]).all() for key in ("draws", "point", "lower", "upper"))
        ordered = bool(np.all(result["lower"] <= result["point"]) and np.all(result["point"] <= result["upper"]))
        report = _accuracy(float(np.max(np.abs(result["draws"][0] - reference))), 1e-10,
                           "first_seeded_draw_vs_scalar_recursion", finite=finite, bands_ordered=ordered)
        report["passed"] &= finite and ordered
        return report
    return Workload("var", f"var_bootstrap_B{draws}", {"draws": draws, "observations": 160, "variables": 2},
                    solve, check, "Residual bootstrap with fixed seed; scalar first-draw oracle, not statistical coverage.")


def _mrio(sectors: int) -> Workload:
    from puremacro.trade.mrio import MRIOTable, aggregate_mrio

    countries, cells = 3, 3 * sectors
    Z = sparse.diags([np.full(cells-1, 0.1), np.full(cells, 0.2), np.full(cells-1, 0.1)],
                     offsets=[-1, 0, 1], format="csr")
    F = np.full((cells, countries, 1), 1 / countries)
    table = MRIOTable.from_arrays(Z, F, np.ones(cells), np.zeros(cells), np.zeros((countries, 1)),
                                  country_codes=[f"C{i}" for i in range(countries)],
                                  sector_codes=[f"S{i}" for i in range(sectors)], fd_codes=["C"],
                                  metadata={"dataset": "synthetic_balanced_scaling_fixture"})
    mapping = {f"S{i}": f"G{i//2}" for i in range(sectors)}
    # Direct six-axis summation, independent of production sparse projection.
    expected = Z.toarray().reshape(countries, sectors//2, 2, countries, sectors//2, 2).sum(axis=(2, 5))
    expected = expected.reshape(cells//2, cells//2)
    def solve():
        return aggregate_mrio(table, mapping)
    def check(result):
        actual = result.Z.toarray() if sparse.issparse(result.Z) else result.Z
        error = max(float(np.max(np.abs(actual - expected))),
                    abs(float(result.F.sum() - F.sum())),
                    abs(float(result.VA.sum() - table.VA.sum())),
                    float(np.max(np.abs(result.output - np.asarray(result.Z.sum(0)).ravel() - result.VA))))
        return _accuracy(error, 1e-10, "aggregation_vs_dense_sum_and_accounting")
    return Workload("mrio", f"mrio_S{sectors}", {"countries": countries, "sectors": sectors, "cells": cells},
                    solve, check, "Synthetic balanced CSR MRIO aggregation only; no native archive or GE solve.")


def environment() -> dict[str, Any]:
    config = io.StringIO()
    with contextlib.redirect_stdout(config):
        np.show_config()
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True))
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "processor": platform.processor(), "cpu_count": os.cpu_count(),
            "puremacro": __version__, "numpy": np.__version__, "scipy": scipy.__version__,
            "revision": revision, "working_tree_dirty": dirty, "numpy_build": config.getvalue(),
            "thread_environment": {name: os.environ.get(name) for name in
                                   ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}}


def run(profile: str = "smoke", repetitions: int = 3, seed: int = 2026,
        families: tuple[str, ...] = ("pf", "hank", "var", "mrio")) -> dict[str, Any]:
    if profile not in ("smoke", "standard") or repetitions < 1:
        raise ValueError("profile must be smoke/standard and repetitions >= 1")
    if not families or set(families) - {"pf", "hank", "var", "mrio"}:
        raise ValueError("families must select pf, hank, var and/or mrio")
    sizes = {"smoke": {"pf": [8, 16], "hank": [6, 12], "var": [8, 16], "mrio": [4, 8]},
             "standard": {"pf": [50, 200, 800], "hank": [40, 100, 200], "var": [100, 500, 1000], "mrio": [32, 128, 512]}}[profile]
    report = {"schema_version": 1, "environment": environment(), "seed": seed, "profile": profile,
              "measurement": {"engine": "benchmark_continuous_vfi", "runtime_repetitions": repetitions,
                              "runtime_warmups": 1, "memory_warmups": 1, "memory_repetitions": 1,
                              "accuracy_evaluations": 1, "runtime_clock": "time.perf_counter",
                              "memory_metric": "tracemalloc peak allocated KiB; not process RSS or total native memory",
                              "setup_timed": False}, "records": [], "passed": True}
    for family in families:
        for size in sizes[family]:
            constructors = ([lambda size=size, backend=backend: _pf(size, backend) for backend in ("direct", "krylov")]
                            if family == "pf" else [lambda family=family, size=size:
                            _hank(size) if family == "hank" else _var(size, seed) if family == "var" else _mrio(size)])
            for construct in constructors:
                try:
                    workload = construct()
                    low, mean, std = measure_runtime_ms(workload.solve, n_runs=repetitions)
                    memory = measure_memory_kib(workload.solve)
                    accuracy = workload.check(workload.solve())
                    row = {"family": family, "name": workload.name, "dimensions": workload.dimensions,
                           "scope": workload.scope, "runtime_ms": {"min": low, "mean": mean, "std": std},
                           "peak_traced_kib": memory, "accuracy": accuracy,
                           "status": "passed" if accuracy["passed"] else "failed"}
                except Exception as exc:
                    row = {"family": family, "size": size, "status": "error",
                           "error": f"{type(exc).__name__}: {exc}"}
                report["records"].append(row)
                report["passed"] &= row["status"] == "passed"
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "standard"], default="smoke")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--families", nargs="+", choices=["pf", "hank", "var", "mrio"], default=["pf", "hank", "var", "mrio"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.repetitions < 1:
        parser.error("--repetitions must be >= 1")
    report = run(args.profile, args.repetitions, args.seed, tuple(args.families))
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
