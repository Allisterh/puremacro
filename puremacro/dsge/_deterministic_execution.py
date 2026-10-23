"""Source-ordered deterministic experiments over the native model and solvers."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from ._parser import Parser, Tokenizer
from ._results import DynareCommandInvocation, DynareCommandResults
from .inspection import _ORDERED_OPTIONS, _PF_COMMANDS, _data_schema_error
from .model import NonlinearModel, _evaluate_assignments, _evaluate_covariance, read_mod


@dataclass(frozen=True)
class PerfectForesightSetup:
    """Resolved deterministic inputs at a particular setup command.

    Arrays are snapshots in declared variable/shock order. The terminal
    boundary is at ``periods + 1``. Later source declarations do not modify
    an existing setup; execute another setup (or ``simul``) to apply them.
    """
    periods: int
    variable_names: tuple[str, ...]
    shock_names: tuple[str, ...]
    y_init: np.ndarray
    y_end: np.ndarray
    exogenous_path: np.ndarray
    initial_path: np.ndarray
    parameters: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        """Describe the prepared horizon and declared array dimensions."""
        return (f"Perfect foresight setup: {self.periods} periods, "
                f"{len(self.variable_names)} endogenous variables, "
                f"{len(self.shock_names)} exogenous variables; terminal date {self.periods + 1}")

    def to_frame(self) -> pd.DataFrame:
        """Return a copy of the resolved shock path with one-based dates."""
        return pd.DataFrame(self.exogenous_path.copy(),
                            index=pd.RangeIndex(1, self.periods + 1, name="t"),
                            columns=self.shock_names)

    def to_dataframe(self) -> pd.DataFrame:
        """Return the resolved shock path as a DataFrame."""
        return self.to_frame()

    def to_markdown(self, **kwargs: Any) -> str:
        """Render the resolved shock path as a Markdown table."""
        from puremacro.reports import _df_to_markdown
        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """Render the resolved shock path as a LaTeX table."""
        from puremacro.reports import _df_to_latex
        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Render the resolved shock path as a Typst table."""
        from puremacro.reports import _df_to_typst
        return _df_to_typst(self.to_frame(), **kwargs)


def _block_parser(text: str, model: NonlinearModel, config: dict[str, Any]) -> Parser:
    parser = Parser(Tokenizer(text, strict=True).tokenize(), raw_text=text)
    parser.variables = list(model.variables)
    parser.shocks = list(model.shocks)
    parser.parameters = list(model.parameter_values)
    parser.param_values = dict(model.parameter_values)
    parser.shocks_config = deepcopy(config)
    return parser


def _execute_deterministic(model: Any, data: pd.DataFrame | None, sigma: Any,
                           report: Any) -> DynareCommandResults:
    from ._mod_execution import ModExecutionError, _selected_result
    from .perfect_foresight import solve_perfect_foresight

    base = model if isinstance(model, NonlinearModel) else read_mod(
        getattr(report, "_expanded_source"), params=model._params)
    base = replace(base, parameter_values=dict(base.parameter_values))
    records = [s for s in report.statements if s.category not in {
        "equation", "model-local", "comment", "macro directive", "empty statement"}]

    # Validate every supplied sample before the first numerical command.
    frames: dict[int, pd.DataFrame] = {}
    for statement in records:
        if statement.name not in {"calib_smoother", "shock_decomposition"}:
            continue
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"{statement.name} requires a data DataFrame")
        required = list(getattr(base._dag, "varobs", None) or [])
        error = _data_schema_error(data, list(base.variables), required)
        if error:
            raise ValueError(error)
        options = statement.effective_options
        first = int(options["first_obs"]) - 1
        stop = None if options["nobs"] is None else first + int(options["nobs"])
        frame = data.iloc[first:stop]
        if frame.empty or (stop is not None and stop > len(data)):
            raise ValueError("Requested first_obs/nobs sample lies outside supplied data")
        frames[statement.index] = frame

    initial = dict.fromkeys(base.variables + base.shocks, 0.0)
    terminal: dict[str, float] | None = None
    history: dict[str, float] = {}
    config: dict[str, Any] = {key: {} for key in ("stderrs", "variances", "covariances", "correlations", "anticipated")}
    shock_cov = np.eye(len(base.shocks))
    prepared: PerfectForesightSetup | None = None
    invocations: list[DynareCommandInvocation] = []
    aliases: dict[str, Any] = {}

    def results() -> DynareCommandResults:
        return DynareCommandResults(**aliases, histval=deepcopy(history) or None,
            endval=deepcopy(terminal), invocations=tuple(invocations),
            commands_executed=tuple(i.command for i in invocations if i.status == "success"))

    def snapshot() -> dict[str, Any]:
        return dict(parameters=dict(base.parameter_values), initial=dict(initial),
                    terminal=deepcopy(terminal), history=dict(history),
                    dated_shocks=deepcopy(config.get("anticipated", {})),
                    covariance=shock_cov.tolist(), source_hashes=report.source_hashes)

    def static_equilibrium(values: dict[str, float], tolerance: float) -> dict[str, float]:
        # An analytical block supplies a domain-valid seed even without
        # initval. It does not replace the active exogenous context: after
        # endval the equations must still be solved and verified at that
        # experiment's permanent exogenous levels.
        guess = dict(values)
        if base.steady_state is not None:
            guess.update(base.steady_state)
        return base.steady(exo_values=values, guess=guess, tol=tolerance)

    def setup(options: dict[str, Any]) -> PerfectForesightSetup:
        nonlocal terminal
        horizon = int(options["periods"])
        if options.get("endval_steady"):
            values = dict(initial if terminal is None else terminal)
            values.update(static_equilibrium(values, float(options.get("tol", 1e-10))))
            terminal = values
        end = initial if terminal is None else terminal
        y_init = np.array([history.get(name, initial[name]) for name in base.variables])
        y_end = np.array([end[name] for name in base.variables])
        shocks = np.tile([end[name] for name in base.shocks], (horizon, 1))
        for name, declaration in config.get("anticipated", {}).items():
            if name not in base.shocks:
                raise ValueError(f"Dated shock {name!r} is not a compiled exogenous variable")
            for period, value in zip(declaration["periods"], declaration["values"]):
                if period > horizon:
                    raise ValueError(f"Shock {name} at period {period} exceeds setup horizon {horizon}")
                shocks[period - 1, base.shocks.index(name)] = value
        return PerfectForesightSetup(horizon, base.variables, base.shocks, y_init.copy(),
            y_end.copy(), np.asarray(shocks, dtype=float), np.tile(y_end, (horizon, 1)),
            dict(base.parameter_values))

    for statement in records:
        name = statement.name
        options = dict(statement.effective_options)
        current_result: Any = None
        if name in {"initval", "histval", "endval"}:
            parser = _block_parser(statement.excerpt, base, config)
            definitions = parser._parse_assignment_block(name)
            if name == "initval":
                initial = dict.fromkeys(base.variables + base.shocks, 0.0)
                initial.update(_evaluate_assignments(definitions, base.parameter_values, initial))
                terminal = None
                history = {}
            elif name == "endval":
                terminal = dict(initial if terminal is None else terminal)
                terminal.update(_evaluate_assignments(definitions, base.parameter_values, terminal))
            else:
                history.update(_evaluate_assignments(definitions, base.parameter_values, initial))
            continue
        if name == "shocks":
            parser = _block_parser(statement.excerpt, base, config)
            parser._parse_shocks_block()
            config = parser.shocks_config
            shock_cov = _evaluate_covariance(config, base.shocks, base.parameter_values)
            continue
        if name not in _ORDERED_OPTIONS and name != "steady":
            continue

        state = snapshot()
        if prepared is not None and name == "perfect_foresight_solver":
            state["prepared"] = dict(periods=prepared.periods, y_init=prepared.y_init.tolist(),
                y_end=prepared.y_end.tolist(), shocks=prepared.exogenous_path.tolist(),
                initial_path=prepared.initial_path.tolist())
        if statement.index in frames:
            state["data_hash"] = hashlib.sha256(pd.util.hash_pandas_object(
                frames[statement.index], index=True).to_numpy().tobytes()).hexdigest()
        state["sigma"] = sigma
        fingerprint = hashlib.sha256(json.dumps(state, sort_keys=True, default=float).encode()).hexdigest()
        common = dict(index=len(invocations), command=name, source=statement.location,
            requested_variables=statement.variables, explicit_options=deepcopy(statement.options),
            effective_options={**deepcopy(options), "input_state": deepcopy(state)},
            input_fingerprint=fingerprint)
        try:
            if name == "steady":
                active = initial if terminal is None else terminal
                solved_values = static_equilibrium(active, float(options.get("tol", 1e-10)))
                active.update(solved_values)
                current_result = pd.Series(solved_values, name="steady_state")
            elif name == "perfect_foresight_setup":
                prepared = setup(options)
                current_result = deepcopy(prepared)
            elif name in {"perfect_foresight_solver", "simul"}:
                if name == "simul":
                    prepared = setup(options)
                if prepared is None:
                    raise ValueError("perfect_foresight_solver requires a preceding perfect_foresight_setup")
                if int(options["periods"]) != prepared.periods:
                    raise ValueError("Solver periods differ from setup; run perfect_foresight_setup again")
                current_result = solve_perfect_foresight(base, periods=prepared.periods,
                    y_init=prepared.y_init, y_end=prepared.y_end,
                    shocks=prepared.exogenous_path, initial_path=prepared.initial_path,
                    tol=float(options["tol"]), max_iter=int(options["maxit"]),
                    linear_solver=options["linear_solver"], mcp=bool(options["mcp"]))
                if not current_result.converged:
                    raise RuntimeError(f"Perfect foresight failed to converge (residual={current_result.residual_norm:g})")
                aliases["perfect_foresight"] = deepcopy(current_result)
                prepared = replace(prepared, initial_path=current_result.path.to_numpy(copy=True))
            else:
                active = initial if terminal is None else terminal
                if any(active[shock] != 0 for shock in base.shocks):
                    raise ValueError("Stochastic commands require zero exogenous reference values; recenter shocks explicitly")
                reference = {v: active[v] for v in base.variables}
                local = replace(base, steady_state=None, guess=reference, shock_cov=shock_cov.copy()).solve(order=1)
                if base._dag is not None and base._dag.varobs:
                    object.__setattr__(local, "_varobs", tuple(base._dag.varobs))
                if name == "stoch_simul":
                    current_result = local.stoch_simul(order=int(options["order"]), irf=int(options["irf"]),
                        periods=int(options["periods"]), ar=int(options["ar"]), burn=int(options["drop"]),
                        hp_filter=options["hp_filter"], pruning=options["pruning"], sigma=sigma)
                    aliases["stoch_simul"] = deepcopy(current_result)
                elif name == "calib_smoother":
                    covariance = local._shock_covariance(sigma) if sigma is not None else None
                    current_result = local.smoother(frames[statement.index],
                        prefilter=bool(options["prefilter"]), shock_cov=covariance)
                    aliases["smoother"] = deepcopy(current_result)
                else:
                    current_result = local.shock_decomposition(frames[statement.index], sigma=sigma)
                    aliases["shock_decomposition"] = deepcopy(current_result)
            current_result = deepcopy(current_result)
            if name in _PF_COMMANDS or name == "steady":
                selected = current_result
                if statement.variables and hasattr(current_result, "path"):
                    selected = current_result.path[list(statement.variables)].copy()
            else:
                selected = _selected_result(current_result, statement.variables, name)
            invocations.append(DynareCommandInvocation(**common, status="success",
                result=current_result, selected_result=selected))
        except Exception as exc:
            invocations.append(DynareCommandInvocation(**common, status="failed", result=current_result,
                error=f"{type(exc).__name__}: {exc}"))
            raise ModExecutionError(f"{statement.location.source}:{statement.location.line}: {name} failed: {exc}",
                                     results(), report) from exc
    return results()
