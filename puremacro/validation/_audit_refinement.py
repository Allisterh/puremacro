"""Explicit, bounded refinement experiments; no implicit solver selection."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ._audit_report import _jsonable


def validate_ladders(ladders):
    names = set()
    for ladder in ladders:
        if not isinstance(ladder.name, str) or not ladder.name.strip() or ladder.name in names:
            raise ValueError("refinement ladder names must be nonempty and unique")
        names.add(ladder.name)
        if (not isinstance(ladder.levels, Sequence) or isinstance(ladder.levels, str)
                or not ladder.levels):
            raise ValueError("a refinement ladder requires a nonempty finite sequence of levels")
        for key in ("max_solves", "consecutive_passes"):
            value = getattr(ladder, key)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"refinement {key} must be a positive integer")
        if ladder.max_cost is not None and (not np.isfinite(ladder.max_cost) or ladder.max_cost < 0):
            raise ValueError("refinement max_cost must be finite and nonnegative")
        if any(not np.isfinite(v) or v < 0 for v in (ladder.atol, ladder.rtol)):
            raise ValueError("refinement atol and rtol must be finite and nonnegative")
        if not callable(ladder.metrics) or (ladder.validate_pair is not None and not callable(ladder.validate_pair)):
            raise ValueError("refinement metrics and optional validate_pair must be callable")
        level_names = set()
        for level in ladder.levels:
            if not isinstance(level.name, str) or not level.name.strip() or level.name in level_names:
                raise ValueError("refinement level names must be nonempty and unique within a ladder")
            level_names.add(level.name)
            if not callable(level.solve) or not np.isfinite(level.cost) or level.cost <= 0:
                raise ValueError("refinement levels require a callable solve and finite positive cost")
            _jsonable(level.settings)


def _metrics(extract, result):
    converged = getattr(result, "converged", getattr(getattr(result, "raw", None), "converged", None))
    if converged is not None and not bool(converged):
        raise ValueError("refinement result declares non-convergence")
    values = dict(extract(result))
    if not values or any(not isinstance(k, str) or not k for k in values):
        raise ValueError("refinement metrics must have nonempty string keys")
    metrics = {}
    for name, value in values.items():
        arr = np.asarray(value)
        if arr.ndim != 0 or not np.isfinite(float(arr)):
            raise ValueError(f"refinement metric {name!r} must be a finite scalar")
        metrics[name] = float(arr)
    return metrics


def audit_ladder(out, baseline_result, ladder):
    """Compare consecutive supplied resolutions, charging every attempted solve."""
    attempts: list[dict] = []
    comparisons = 0
    consecutive = 0
    spent = 0.0
    stop = "levels_exhausted"
    error = None
    baseline = None
    previous_result = baseline_result
    previous_name = "baseline"
    try:
        baseline = previous = _metrics(ladder.metrics, baseline_result)
    except Exception as exc:
        stop, error = "baseline_invalid", f"{type(exc).__name__}: {exc}"
    else:
        for level in ladder.levels:
            if len(attempts) >= ladder.max_solves:
                stop = "solve_budget_exhausted"
                break
            if ladder.max_cost is not None and spent + level.cost > ladder.max_cost:
                stop = "cost_budget_exhausted"
                break
            spent += float(level.cost)
            attempt = {"level": level.name, "settings": _jsonable(level.settings),
                       "cost": float(level.cost), "status": "failed"}
            attempts.append(attempt)
            try:
                refined = level.solve()
            except Exception as exc:
                stop, error = "callback_failed", f"{type(exc).__name__}: {exc}"
                attempt["error"] = error
                break
            try:
                if ladder.validate_pair is not None:
                    ladder.validate_pair(previous_result, refined)
            except Exception as exc:
                stop, error = "pair_validation_failed", f"{type(exc).__name__}: {exc}"
                attempt["error"] = error
                break
            try:
                current = _metrics(ladder.metrics, refined)
                if current.keys() != previous.keys():
                    raise ValueError("refinement metrics changed keys between levels")
            except Exception as exc:
                stop, error = "invalid_result_or_metrics", f"{type(exc).__name__}: {exc}"
                attempt["error"] = error
                break
            stable = True
            for name in sorted(current):
                before, after = previous[name], current[name]
                difference = abs(after - before)
                tolerance = ladder.atol + ladder.rtol * abs(after)
                stable = stable and difference <= tolerance
                out.add("refinement", f"{ladder.name}/{level.name}:{name}", difference, tolerance,
                        ladder.units.get(name, "caller metric units"), ladder.evidence_source,
                        previous_level=previous_name, level=level.name, previous=before, refined=after,
                        atol=ladder.atol, rtol=ladder.rtol,
                        relative_change=difference / abs(after) if abs(after) > ladder.atol else None)
            comparisons += 1
            consecutive = consecutive + 1 if stable else 0
            attempt.update(status="passed" if stable else "failed", metrics=current,
                           comparison_to=previous_name, consecutive_passes=consecutive)
            previous, previous_result, previous_name = current, refined, level.name
            if consecutive >= ladder.consecutive_passes:
                stop = "stable_metrics"
                break
    evidence = {
        "stop_reason": stop, "attempted_solves": len(attempts), "completed_comparisons": comparisons,
        "max_solves": ladder.max_solves, "declared_cost_used": spent, "max_cost": ladder.max_cost,
        "cost_scope": "caller-declared work units; not elapsed time or a hard resource bound",
        "planned_levels": len(ladder.levels), "consecutive_passes": consecutive,
        "consecutive_passes_required": ladder.consecutive_passes,
        "baseline_metrics": baseline, "attempts": attempts,
        "pair_validation": "caller_supplied" if ladder.validate_pair is not None else "not_checked",
        "scope": "stability of selected metrics across adjacent supplied numerical resolutions",
    }
    if error is not None:
        evidence["error"] = error
    status = ("passed" if stop == "stable_metrics" else "not_checked"
              if not attempts and stop in ("solve_budget_exhausted", "cost_budget_exhausted") else "failed")
    out.add("refinement", ladder.name, stop == "stable_metrics", status=status,
            source=ladder.evidence_source, **evidence)
    out.metadata.setdefault("refinement_ladders", {})[ladder.name] = evidence
