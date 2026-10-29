"""Reproduce archived policies, published-moment comparisons and grid sensitivity.

Run from the repository root with ``python -m tools.reference_validation.run_arellano_study
--output /tmp/arellano-study``. The frozen original Fortran reference is used by
default; run run_arellano_original.py separately to rebuild it with gfortran.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import scipy

from tools.reference_validation.arellano_2008 import (
    FIXTURE, MOMENTS, Solution, compare_original, load_original, monte_carlo, solve,
)


def native_case() -> Solution:
    """Current public API calibration; a distinct model, not an archive match."""
    from puremacro.sovereign import solve_sovereign_default

    started = time.monotonic()
    r = solve_sovereign_default(beta=.95282, r_star=.017, theta=.282, gamma=2,
                               sigma_eps=0, y_penalty_type="arellano", n_b=321, n_y=21,
                               b_min=-.15, b_max=.33, enforce_monotonicity=False,
                               tol=1e-10, max_iter=10000, howard_steps=0)
    if not r.converged:
        raise RuntimeError(f"Public solver did not converge: {r.residual}")
    # Translate positive debt into negative assets and scale mean income to 10.
    native_indices = np.argmin(abs(r.policy_b_prime[:, :, None]-r.b_grid), axis=2)
    return Solution(-10*r.b_grid[::-1], 10*r.y_grid, r.p_trans,
                    10*np.asarray(r.metadata["y_default_vec"]),
                    .1*(r.v_star[::-1]-1/(1-.95282)), r.q_bond[::-1],
                    len(r.b_grid)-1-native_indices[::-1], r.default_prob[::-1] > .5,
                    "canonical", theta=.282, gross_r=1.017,
                    diagnostics=dict(iterations=r.iterations, reported_residual=r.residual,
                                     elapsed_seconds=time.monotonic()-started,
                                     differences_from_archive=["Canonical exclusion Bellman equation",
                                         "Stationary mean income normalized to 10",
                                         "Default cap is .969 times stationary mean income",
                                         "Uniform 321-point grid on [-3.3, 1.5], exact zero",
                                         "Double-precision calibration and transition matrix"],
                                     simulation="Shared study simulator follows the Bellman timing; built-in 1500-period diagnostics are not used"))


def plot(report: dict, output: Path) -> None:
    import matplotlib.pyplot as plt

    names = ["original_200_21", "refined_399_21", "refined_797_21", "refined_797_41", "refined_797_81"]
    labels = ["200 × 21\nFortran", "399 × 21", "797 × 21", "797 × 41", "797 × 81"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for ax, name, label in zip(axes.flat,
            ["spread_mean_pct", "annual_default_pct", "debt_pct", "output_std_pct"],
            ["Mean annual spread (%)", "Annual default frequency (%)",
             "Mean debt / output (%)", "Standard deviation of log output (%)"]):
        moments = [report["cases"][case]["simulation"]["moments"][name] for case in names]
        means = np.array([m["mean"] for m in moments])
        errors = np.array([m["mc_ci95"][1]-m["mean"] for m in moments])
        ax.errorbar(range(len(names)), means, yerr=errors, fmt="o-", capsize=4, color="#2563a6", label="Study mean ± 95% MC interval")
        ax.axhline(MOMENTS[name], color="#af4f30", linestyle="--", label="Published model moment")
        ax.set_xticks(range(len(names)), labels)
        ax.set_title(label)
        ax.grid(axis="y", alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Arellano (2008): archived calibration and grid sensitivity", fontsize=14)
    fig.savefig(output / "grid_sensitivity.png", dpi=180)
    fig.savefig(output / "grid_sensitivity.pdf")
    plt.close(fig)


def run(args) -> dict:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    original = load_original(args.reference)
    check = solve()
    comparison = compare_original(check, original)
    print("Original-code comparison:", json.dumps(comparison), flush=True)
    if not comparison["passed"]:
        raise RuntimeError("Python study does not match the independent Fortran reference")
    seeds = list(range(1979, 1979+args.replications))
    report = dict(schema_version=1, source_commit=subprocess.check_output(
                      ["git", "rev-parse", "HEAD"], text=True).strip(),
                  python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                  study_code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in Path(__file__).parent.glob("*arellano*.py")},
                  reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                  external_comparison=comparison, matched_solver_diagnostics=check.diagnostics,
                  design=dict(seeds=seeds, periods_including_burn=args.periods, burn=args.burn,
                              windows_per_replication=args.windows, window_quarters=74,
                              sampling="First eligible predefault windows after burn, separately in each independent seeded path",
                              uncertainty="Student-t 95% Monte Carlo CI for the mean across independent seeded replications; replication_sd measures sampling dispersion, not parameter uncertainty",
                              targets="Arellano (2008), Table 4 model column; also in author's July 2007 public draft",
                              paper="https://doi.org/10.1257/aer.98.3.690",
                              archive="https://doi.org/10.3886/E113248V1"), cases={})
    specifications = [
        ("original_200_21", lambda: original),
        ("refined_399_21", lambda: solve(refinement=1)),
        ("refined_797_21", lambda: solve(refinement=2)),
        ("refined_797_41", lambda: solve(refinement=2, n_y=41)),
        ("refined_797_81", lambda: solve(refinement=2, n_y=81)),
        ("canonical_200_21", lambda: solve(timing="canonical")),
        ("public_321_21", native_case),
    ]
    for name, factory in specifications:
        started = time.monotonic()
        solution = factory()
        print(name, "solved", solution.diagnostics, flush=True)
        simulation = monte_carlo(solution, seeds, args.periods, args.burn, args.windows)
        report["cases"][name] = dict(n_assets=len(solution.assets), n_income=len(solution.income),
                                    timing=solution.timing, diagnostics=solution.diagnostics,
                                    simulation=simulation, elapsed_seconds=time.monotonic()-started)
        np.savez_compressed(output / (name+".npz"), assets=solution.assets, income=solution.income,
                            transition=solution.transition, default_income=solution.default_income,
                            value=solution.value, price=solution.price, policy=solution.policy,
                            default=solution.default)
        (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        print(name, "moments", {k: round(v["mean"], 5) for k, v in simulation["moments"].items()}, flush=True)
    write_outputs(report, output)
    return report


def write_outputs(report: dict, output: Path) -> None:
    with (output / "moments.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "moment", "published", "mean", "difference", "mc_se", "mc_ci95_low", "mc_ci95_high", "replication_sd"])
        for case, record in report["cases"].items():
            for name, value in record["simulation"]["moments"].items():
                writer.writerow([case, name, value["published"], value["mean"], value["difference"],
                                 value["mc_se"], *value["mc_ci95"], value["replication_sd"]])
    plot(report, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=FIXTURE / "reference.npz")
    parser.add_argument("--replications", type=int, default=16)
    parser.add_argument("--periods", type=int, default=100000)
    parser.add_argument("--burn", type=int, default=1000)
    parser.add_argument("--windows", type=int, default=100)
    run(parser.parse_args())
