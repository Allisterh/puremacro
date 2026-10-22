"""Validated source-ordered adapter over the existing numerical methods."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from ._results import DynareCommandInvocation, DynareCommandResults
from .inspection import ModInspectionReport, _ORDERED_OPTIONS, _data_schema_error, inspect_mod


class ModCompatibilityError(ValueError):
    """An incompatible command program, detected before numerical execution."""

    def __init__(self, report: ModInspectionReport):
        self.report = report
        super().__init__(report.summary())


class ModExecutionError(RuntimeError):
    """A numerical command failed; ``results`` retains completed and failed entries."""

    def __init__(self, message: str, results: DynareCommandResults,
                 report: ModInspectionReport):
        self.results = results
        self.report = report
        super().__init__(message)


def _require_supported(report: ModInspectionReport, *, loading: bool = False) -> None:
    bad = [issue for issue in report.issues
           if not (loading and issue.code == "DATA_REQUIRED")]
    if report.coverage != "complete" or bad:
        raise ModCompatibilityError(report)


def _selected_result(result: Any, names: tuple[str, ...], command: str) -> Any:
    if not names:
        return result
    selection = list(names)
    if command == "stoch_simul":
        moments = result.theoretical_moments
        if moments is not None:
            moments = replace(moments,
                moments=moments.moments.loc[selection].copy(),
                covariance=moments.covariance.loc[selection, selection].copy(),
                correlation=(moments.correlation.loc[selection, selection].copy()
                             if moments.correlation is not None else None),
                autocorr=moments.autocorr.loc[selection].copy(),
                fevd=moments.fevd.loc[selection].copy(),
                autocorr_matrices=[x.loc[selection, selection].copy() for x in moments.autocorr_matrices])
        return replace(result, variable_names=names, theoretical_moments=moments,
            simulated_moments=(result.simulated_moments.loc[selection].copy()
                               if result.simulated_moments is not None else None),
            irfs={f"{v}_{s}": result.irfs[f"{v}_{s}"].copy() for v in names for s in result.shock_names
                  if f"{v}_{s}" in result.irfs},
            _sim_corr=(result._sim_corr.loc[selection, selection].copy() if result._sim_corr is not None else None))
    if command == "shock_decomposition":
        return replace(result, variable_names=selection,
                       components={v: result.components[v].copy() for v in names})
    # Smoother output keeps the full state system and fitted observable system;
    # output selections only narrow the presentation frames that contain them.
    return replace(result,
        states=result.states.loc[:, [v for v in names if v in result.states]].copy(),
        filtered_states=result.filtered_states.loc[:, [v for v in names if v in result.filtered_states]].copy(),
        smoothed_obs=result.smoothed_obs.loc[:, [v for v in names if v in result.smoothed_obs]].copy(),
        varobs=tuple(v for v in names if v in result.smoothed_obs))


def _fingerprint(model: Any, data: Any, sigma: Any, report: ModInspectionReport) -> str:
    payload = dict(params=dict(model._params), variables=list(model.variables),
                   shocks=list(model.shocks), sources=report.source_hashes, sigma=sigma,
                   covariance=np.asarray(model._shock_cov).tolist() if model._shock_cov is not None else None)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=float).encode())
    if data is not None:
        digest.update(pd.util.hash_pandas_object(data, index=True).to_numpy().tobytes())
        digest.update(json.dumps([str(x) for x in data.columns]).encode())
    return digest.hexdigest()


def _execute_ordered(model: Any, data: pd.DataFrame | None, sigma: Any) -> DynareCommandResults:
    source = getattr(model, "_mod_source_text", None)
    if source is None:
        raise ValueError("Ordered execution requires a model returned by load_mod, with source provenance.")
    # Use the immutable expanded snapshot captured at load time: included files
    # may have changed since loading, and must never silently change this study.
    report = getattr(model, "_ordered_inspection", None)
    if report is None:
        if "@#" in source or "@{" in source:
            raise ValueError("Macro programs require load_mod(command_mode='ordered') to retain inspection provenance before solving")
        report = inspect_mod(source, ordered=True, data=data,
                             base_dir=getattr(model, "_mod_base_dir", None))
    elif data is not None:
        report = replace(report, issues=tuple(i for i in report.issues if i.code != "DATA_REQUIRED"))
    _require_supported(report)
    commands = [s for s in report.statements if s.name in _ORDERED_OPTIONS]
    # Validate all data and effective options before the first numerical call.
    planned = []
    for statement in commands:
        opts = dict(statement.effective_options)
        for key in ("order", "irf", "periods", "ar", "drop", "first_obs", "nobs"):
            if key in opts and opts[key] is not None:
                value = opts[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
                    raise ValueError(f"{statement.location.source}:{statement.location.line}: {key} must be an integer")
                if value < (1 if key in {"order", "first_obs", "nobs"} else 0):
                    raise ValueError(f"{key} is outside the supported range")
        if statement.name == "stoch_simul" and opts["order"] not in {1, 2, 3}:
            raise ValueError("Ordered stoch_simul supports order 1, 2, or 3")
        frame = None
        if statement.name in {"calib_smoother", "shock_decomposition"}:
            if not isinstance(data, pd.DataFrame):
                raise ValueError(f"{statement.name} requires data DataFrame")
            required = list(getattr(model, "_varobs", None) or [])
            schema_error = _data_schema_error(data, list(model.variables), required)
            if schema_error:
                raise ValueError(schema_error)
            start = int(opts["first_obs"]) - 1
            stop = None if opts["nobs"] is None else start + int(opts["nobs"])
            frame = data.iloc[start:stop]
            if frame.empty or (stop is not None and stop > len(data)):
                raise ValueError("Requested first_obs/nobs sample lies outside supplied data")
            if statement.name == "calib_smoother" and statement.variables:
                available = set(model.states) | set(required or data.columns)
                absent = set(statement.variables) - available
                if absent:
                    raise ValueError(f"Smoother cannot report non-state, non-observable variables: {sorted(absent)}")
        planned.append((statement, opts, frame))

    # Hash every command input before execution, so even a malformed index or
    # unhashable data value cannot fail after an earlier numerical command.
    fingerprints = [_fingerprint(model, frame, sigma, report) for _, _, frame in planned]
    invocations: list[DynareCommandInvocation] = []
    aliases: dict[str, Any] = {}
    attr = {"stoch_simul": "stoch_simul", "calib_smoother": "smoother",
            "shock_decomposition": "shock_decomposition"}

    def results() -> DynareCommandResults:
        return DynareCommandResults(**aliases,
            commands_executed=tuple(i.command for i in invocations if i.status == "success"),
            invocations=tuple(invocations))

    for (statement, opts, frame), fingerprint in zip(planned, fingerprints):
        common: dict[str, Any] = dict(index=len(invocations), command=statement.name, source=statement.location,
            requested_variables=statement.variables, explicit_options=deepcopy(statement.options),
            effective_options=deepcopy(opts), input_fingerprint=fingerprint)
        try:
            if statement.name == "stoch_simul":
                result = model.stoch_simul(order=int(opts["order"]), irf=int(opts["irf"]),
                    periods=int(opts["periods"]), ar=int(opts["ar"]), burn=int(opts["drop"]),
                    hp_filter=opts["hp_filter"], pruning=opts["pruning"], sigma=sigma)
            elif statement.name == "calib_smoother":
                covariance = model._shock_covariance(sigma) if sigma is not None else None
                result = model.smoother(frame, prefilter=bool(opts["prefilter"]), shock_cov=covariance)
            else:
                result = model.shock_decomposition(frame, sigma=sigma)
            result = deepcopy(result)
            selected = _selected_result(result, statement.variables, statement.name)
            invocations.append(DynareCommandInvocation(**common, status="success", result=result,
                                                       selected_result=selected))
            aliases[attr[statement.name]] = result
        except Exception as exc:
            invocations.append(DynareCommandInvocation(**common, status="failed", error=f"{type(exc).__name__}: {exc}"))
            raise ModExecutionError(
                f"{statement.location.source}:{statement.location.line}: {statement.name} failed: {exc}",
                results(), report) from exc
    return results()
