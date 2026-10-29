"""Sovereign-risk policy brief: reentry, income volatility and persistence.

EN: An illustrative policy brief on PureMacro's public calibration of the
short-term Arellano (2008) model. It varies one parameter at a time (the
market-reentry probability, income volatility and income persistence) and
reports sustainable debt, spreads, default exposure and the value of market
access, each with its 95% interval and its numerical grid change. It is not a
claim about any country and fits no parameter to targets.

ES: Un informe de política ilustrativo sobre la calibración pública de
PureMacro del modelo de Arellano (2008) con deuda de corto plazo. Varía un
parámetro cada vez (la probabilidad de reingreso al mercado, la volatilidad y
la persistencia del ingreso) e informa la deuda sostenible, los diferenciales,
la exposición al impago y el valor del acceso al mercado, cada uno con su
intervalo del 95% y su cambio numérico de malla. No es una afirmación sobre
ningún país y no ajusta ningún parámetro a objetivos.

The page is docs/brief_sovereign_risk.md (ES: docs/es/brief_sovereign_risk.md).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t

from puremacro.sovereign import SovereignDefaultResult, solve_sovereign_default
from puremacro.sovereign.measurement import MOMENTS, ergodic_moments, value_of_market_access

# PureMacro's public calibration (tests/test_arellano_2008_study.py::_public_monotone).
BASELINE: dict[str, Any] = dict(
    beta=.95282, r_star=.017, theta=.282, rho_y=.945, sigma_y=.025,
    b_min=-.15, b_max=.33, sigma_eps=0, y_penalty_type="arellano",
    enforce_monotonicity=False, tol=1e-10, max_iter=10000, policy_search="monotone",
)
GRIDS: dict[str, tuple[int, int]] = {"converged": (6401, 641), "error": (3201, 641), "coarse": (801, 161)}
DESIGN: dict[str, Any] = {"periods": 500_000, "seeds": list(range(1979, 1995)), "burn": 1_000}
# The reference study's registered grid tolerances for these moments (pct / pp).
TOLERANCES: dict[str, float] = {m: 0.10 for m in MOMENTS}

_SCENARIO_KEYS = ("theta", "sigma_y", "rho_y")
_SCENARIO_VALUES = {"theta": (.15, .50), "sigma_y": (.020, .030), "rho_y": (.90, .97)}
SCENARIOS: dict[str, dict[str, float]] = {"baseline": {}}
for _key in _SCENARIO_KEYS:
    for _value in _SCENARIO_VALUES[_key]:
        SCENARIOS[f"{_key}={_value:g}"] = {_key: _value}

_SOURCES = ("sovereign/arellano.py", "sovereign/_monotone.py", "sovereign/_results.py",
            "sovereign/measurement.py", "examples/first_brief_sovereign.py")


def parse_scenario(text: str) -> tuple[str, dict[str, float]]:
    """``"baseline"`` or ``key=value`` with key in theta, sigma_y, rho_y -> (canonical name, overrides)."""
    text = text.strip()
    if text == "baseline":
        return "baseline", {}
    key, sep, raw = text.partition("=")
    key = key.strip()
    if not sep or key not in _SCENARIO_KEYS:
        raise ValueError(f"scenario must be 'baseline' or key=value with key one of "
                         f"{', '.join(_SCENARIO_KEYS)}; got {text!r}")
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"scenario value must be a number; got {raw!r}") from None
    if not math.isfinite(value):
        raise ValueError(f"scenario value must be finite; got {raw!r}")
    return f"{key}={value:g}", {key: value}


def solve_scenario(overrides: Mapping[str, float], n_b: int, n_y: int) -> tuple[SovereignDefaultResult, float]:
    """Solve the baseline with ``overrides`` on an ``n_b x n_y`` grid; returns (result, seconds)."""
    start = time.perf_counter()
    result = solve_sovereign_default(**{**BASELINE, **overrides}, n_b=int(n_b), n_y=int(n_y))
    return result, time.perf_counter() - start


def _solver_parameters(result: SovereignDefaultResult) -> dict[str, Any]:
    """Every keyword of ``solve_sovereign_default`` as used for ``result``."""
    md = result.metadata
    params = {
        "beta": md["beta"], "gamma": md["gamma"], "r_star": md["r_star"],
        "delta": result.maturity_param_delta, "kappa": result.coupon_rate_kappa,
        "sigma_eps": result.taste_shock_scale, "rho_y": md["rho_y"], "sigma_y": md["sigma_y"],
        "theta": md["theta"], "d0": md["d0"], "d1": md["d1"],
        "n_b": len(result.b_grid), "b_max": result.b_grid[-1], "b_min": result.b_grid[0],
        "n_y": len(result.y_grid), "m_tauchen": md["m_tauchen"], "tol": md["tol"],
        "max_iter": md["max_iter"], "damping": md["damping"], "recovery_rate": md["recovery_rate"],
        "howard_steps": md["howard_steps"], "enforce_monotonicity": md["enforce_monotonicity"],
        "y_penalty_type": md["y_penalty_type"], "policy_search": md["policy_search"],
    }
    return {k: _plain(v) for k, v in params.items()}


def _plain(value: Any) -> Any:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    return value


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def measure_case(result: SovereignDefaultResult, elapsed: float, *, periods: int,
                 seeds: Sequence[int], burn: int) -> dict:
    """Solver diagnostics, ergodic moments (per seed and summary) and welfare, JSON-ready."""
    seeds = [int(s) for s in seeds]
    table = ergodic_moments(result, periods=periods, seeds=seeds, burn=burn)
    md = result.metadata
    return {
        "n_b": len(result.b_grid),
        "n_y": len(result.y_grid),
        "parameters": _solver_parameters(result),
        "solver": {
            "converged": bool(result.converged),
            "iterations": int(result.iterations),
            "residual": float(result.residual),
            "monotone_certification_gap": _optional_float(md.get("monotone_certification_gap")),
            "exhaustive_bellman_residual": _optional_float(md.get("exhaustive_bellman_residual")),
            "elapsed_s": float(elapsed),
        },
        "simulation": {
            "seeds": seeds,
            "per_seed": {m: [float(table.loc[m, f"seed_{s}"]) for s in seeds] for m in MOMENTS},
            "summary": {m: {"mean": float(table.loc[m, "mean"]), "se": float(table.loc[m, "se"]),
                            "ci95": [float(table.loc[m, "ci95_low"]), float(table.loc[m, "ci95_high"])]}
                        for m in MOMENTS},
        },
        "welfare_pct": float(value_of_market_access(result)),
    }


def paired_change(coarse: dict, fine: dict) -> dict:
    """The reference study's paired grid-change rule, fine minus coarse, per seed."""
    seeds_c, seeds_f = coarse["simulation"]["seeds"], fine["simulation"]["seeds"]
    if list(seeds_c) != list(seeds_f) or len(seeds_c) < 2 or len(set(seeds_c)) != len(seeds_c):
        raise ValueError(f"paired changes need the same distinct seeds (>= 2); got {seeds_c} and {seeds_f}")
    n = len(seeds_c)
    t_crit = float(student_t.ppf(.975, n - 1))
    moments = {}
    for m in MOMENTS:
        d = np.asarray(fine["simulation"]["per_seed"][m], float) - np.asarray(coarse["simulation"]["per_seed"][m], float)
        if not np.all(np.isfinite(d)):
            raise ValueError(f"nonfinite moment: {m}")
        mean = float(d.mean())
        half = float(t_crit * d.std(ddof=1) / np.sqrt(n))
        bound = abs(mean) + half
        tol = TOLERANCES[m]
        status = ("passed" if bound <= tol else "grid_shift_exceeds_tolerance"
                  if abs(mean) > tol else "sampling_precision_insufficient")
        moments[m] = {"change": mean, "ci95": [mean - half, mean + half],
                      "absolute_change_bound": bound, "tolerance": tol, "status": status}
    return {"passed": all(v["status"] == "passed" for v in moments.values()), "moments": moments,
            "welfare_change_pct": float(fine["welfare_pct"]) - float(coarse["welfare_pct"])}


def source_hashes() -> dict[str, str]:
    """sha256 of the solver, measurement and this example, keyed by package-relative path."""
    root = Path(__file__).resolve().parents[1]
    return {f"puremacro/{rel}": hashlib.sha256((root / rel).read_bytes()).hexdigest() for rel in _SOURCES}


def _peak_rss_gb() -> float | None:
    try:
        import resource
    except ImportError:  # Windows
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1.0 if sys.platform == "darwin" else 1024.0  # bytes on macOS, KiB on Linux
    return round(peak * scale / 1e9, 3)


def _versions() -> dict[str, str]:
    import pandas
    import scipy

    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "pandas": pandas.__version__}


def _package_version() -> str:
    import puremacro

    return str(getattr(puremacro, "__version__", "unknown"))


def _write_atomic(path: Path, record: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=1, allow_nan=False) + "\n")
    os.replace(tmp, path)


def _case_key(n_b: int, n_y: int) -> str:
    return f"{n_b}x{n_y}"


def build_record(grids: Sequence[tuple[int, int]], *, design: dict, partial: Path | None = None,
                 log: Callable[[str], Any] = print, scenarios: Mapping[str, Mapping[str, float]] | None = None) -> dict:
    """Solve and measure every scenario on every grid (``grids[0]`` is the fine grid).

    After each case the record so far is written to ``partial`` (atomic replace).
    A run restarted with the same ``partial`` reuses its finished cases only if
    their ``source_sha256`` and ``design`` match the current ones; otherwise the
    file is discarded and every case is solved again.
    """
    grids = [(int(n_b), int(n_y)) for n_b, n_y in grids]
    scenarios = SCENARIOS if scenarios is None else scenarios
    hashes = source_hashes()
    design = json.loads(json.dumps(design))
    finished: dict[str, dict] = {}
    if partial is not None and Path(partial).exists():
        partial = Path(partial)
        try:
            old = json.loads(partial.read_text())
        except (OSError, ValueError):
            old = {}
        if old.get("source_sha256") == hashes and old.get("design") == design:
            finished = {name: s.get("cases", {}) for name, s in old.get("scenarios", {}).items()}
            log(f"Resuming {sum(map(len, finished.values()))} finished cases from {partial.name}")
        else:
            log(f"Discarding {partial.name}: its source hashes or design differ from the current run")
    record: dict[str, Any] = {
        "schema": 1,
        "package_version": _package_version(),
        "versions": _versions(),
        "source_sha256": hashes,
        "design": design,
        "baseline": dict(BASELINE),
        "tolerances": dict(TOLERANCES),
        "grids": [list(g) for g in grids],
        "peak_rss_gb": _peak_rss_gb(),
        "scenarios": {},
    }
    for name, overrides in scenarios.items():
        entry = {"overrides": dict(overrides), "cases": {}}
        record["scenarios"][name] = entry
        for n_b, n_y in grids:
            key = _case_key(n_b, n_y)
            done = finished.get(name, {}).get(key)
            if done is not None:
                entry["cases"][key] = done
                log(f"{name} {key}: reused from partial")
                continue
            result, elapsed = solve_scenario(overrides, n_b, n_y)
            start = time.perf_counter()
            case = measure_case(result, elapsed, periods=design["periods"], seeds=design["seeds"], burn=design["burn"])
            del result
            summary = case["simulation"]["summary"]
            log(f"{name} {key}: solved in {elapsed:.1f} s, measured in {time.perf_counter() - start:.1f} s, "
                f"gap {case['solver']['monotone_certification_gap']}, "
                + ", ".join(f"{m} {summary[m]['mean']:.4f}" for m in MOMENTS)
                + f", welfare {case['welfare_pct']:.4f}%")
            entry["cases"][key] = case
            record["peak_rss_gb"] = _peak_rss_gb()
            if partial is not None:
                _write_atomic(Path(partial), record)
        if len(grids) >= 2:
            entry["grid_change"] = paired_change(entry["cases"][_case_key(*grids[1])],
                                                 entry["cases"][_case_key(*grids[0])])
    record["peak_rss_gb"] = _peak_rss_gb()
    return record
