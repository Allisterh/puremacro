"""Read-only compatibility inspection for Dynare-style sources.

Inspection collects syntax through the native parser, stopping before numerical
calibration and compilation. It does not establish economic or numerical validity.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any
import math

import numpy as np
import pandas as pd

from puremacro import __version__
from ._ast import BinOp, Call, Node, Param, UnaryOp, Var
from ._macro import _MacroTrace, _preprocess_with_provenance, _split_into_items
from ._parser import MATH_FUNCTIONS, Parser, Tokenizer, _LocalRef

# Only arguments actually forwarded by the native runners belong here. The
# ordered adapter consumes these same descriptions (see _mod_execution.py).
_STOCH_DEFAULTS: dict[str, Any] = {
    "order": 2, "irf": 40, "periods": 0, "ar": 5, "drop": 100,
    "hp_filter": None, "pruning": None, "nograph": True,
}
_PF_COMMANDS = {"perfect_foresight_setup", "perfect_foresight_solver", "simul"}
_PF_DEFAULTS: dict[str, Any] = {
    "periods": 100, "tol": 1e-8, "tolf": None, "maxit": 50, "max_iter": None,
    "stack_solve_algo": 0, "linear_solver": "direct", "endval_steady": False, "mcp": False,
}
_ORDERED_OPTIONS: dict[str, dict[str, Any]] = {
    "stoch_simul": _STOCH_DEFAULTS,
    "calib_smoother": {"prefilter": False, "first_obs": 1, "nobs": None},
    "shock_decomposition": {"first_obs": 1, "nobs": None},
    "perfect_foresight_setup": {"periods": 100, "endval_steady": False},
    "perfect_foresight_solver": {key: value for key, value in _PF_DEFAULTS.items() if key != "endval_steady"},
    "simul": _PF_DEFAULTS,
}
_LEGACY_OPTIONS: dict[str, set[str]] = {
    "stoch_simul": {"order", "irf", "periods", "ar", "hp_filter", "pruning", "nograph"},
    "calib_smoother": set(), "shock_decomposition": set(),
    "osr": {"optimizer", "maxiter"},
    "ramsey_policy": {"planner_discount", "instruments"},
    "ramsey_model": {"planner_discount", "instruments"},
    "perfect_foresight_setup": {"periods", "tol", "maxit", "max_iter"},
    "perfect_foresight_solver": {"periods", "tol", "maxit", "max_iter"},
    "simul": {"periods", "tol", "maxit", "max_iter"},
}
_SETUP = {"var", "varexo", "parameters", "predetermined_variables", "varobs", "model",
          "initval", "steady_state_model", "shocks", "shock_groups"}
_PARSED_ONLY = {"estimated_params", "estimated_params_init", "estimated_params_bounds",
                "markov_switching", "trend_var", "log_trend_var", "varexo_det"}
_STATEFUL = {"planner_objective", "optim_weights", "osr_params", "osr_params_bounds",
             "histval", "endval", "hetagent_block"}


@dataclass(frozen=True)
class ModSourceSpan:
    """Original line provenance and expanded coordinates; columns may be unknown."""
    source: str
    line: int
    column: int | None
    end_line: int | None
    end_column: int | None
    expanded_line: int
    expanded_column: int
    expanded_end_line: int
    expanded_end_column: int
    include_chain: tuple[str, ...] = ()
    expansion_instance: int | None = None
    origins: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class ModStatement:
    """One collected statement; model block children have their own entries."""
    index: int
    name: str
    category: str
    status: str
    location: ModSourceSpan
    excerpt: str
    options: dict[str, Any] = field(default_factory=dict)
    variables: tuple[str, ...] = ()
    option_status: dict[str, str] = field(default_factory=dict)
    reason: str = ""
    remedy: str = ""
    preserves_invocation: bool | None = None
    effective_options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModIssue:
    """Stable compatibility finding, independent of numerical correctness."""
    code: str
    status: str
    reason: str
    remedy: str
    location: ModSourceSpan | None = None
    statement_index: int | None = None


@dataclass(frozen=True)
class ModInspectionReport:
    """Deterministic source inventory. Exit codes: 0 clear, 1 issues, 2 incomplete."""
    source_hashes: dict[str, str]
    entry_point: str
    ordered: bool
    coverage: str
    declarations: dict[str, int]
    statements: tuple[ModStatement, ...]
    issues: tuple[ModIssue, ...]
    package_version: str = __version__
    data_check: str = "not supplied"

    @property
    def exit_code(self) -> int:
        return 2 if self.coverage != "complete" else (1 if self.issues else 0)

    def to_dict(self) -> dict[str, Any]:
        def normalize(value: Any) -> Any:
            if isinstance(value, float) and not math.isfinite(value):
                return {"nonfinite_number": str(value)}
            if isinstance(value, dict):
                return {key: normalize(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [normalize(item) for item in value]
            return value
        return normalize({**asdict(self), "exit_code": self.exit_code})

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize deterministically without timestamps or runtime identifiers."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent, allow_nan=False)

    def summary(self) -> str:
        lines = [f".mod inspection: {self.coverage}; {len(self.issues)} finding(s)",
                 f"Entry point: {self.entry_point}; ordered={self.ordered}",
                 "Syntax and execution support only; no numerical validation was run."]
        for statement in self.statements:
            loc = statement.location
            lines.append(f"{statement.index}: {loc.source}:{loc.line}:{loc.column or '?'} "
                         f"{statement.category} {statement.name}: {statement.status}")
            if statement.options or statement.variables:
                lines.append(f"  options={json.dumps(statement.options, sort_keys=True)}; "
                             f"variables={list(statement.variables)}")
        for issue in self.issues:
            issue_loc = issue.location
            where = f"{issue_loc.source}:{issue_loc.line}: " if issue_loc else ""
            lines.append(f"{issue.code}: {where}{issue.reason} Remedy: {issue.remedy}")
        return "\n".join(lines)


def _read_source(path_or_text: str | Path) -> tuple[str, str, Path]:
    value = str(path_or_text)
    if isinstance(path_or_text, Path) or ("\n" not in value and "\r" not in value and ";" not in value):
        path = Path(path_or_text).resolve()
        if path.stat().st_size > 2_000_000:
            raise ValueError("Inspection source size limit exceeded (2 MB)")
        # Keep the exact UTF-8 source for provenance hashes. The inspection
        # preprocessor normalizes newlines separately before parsing.
        return path.read_bytes().decode("utf-8"), str(path), path.parent
    return value, "<text>", Path.cwd()


def _line_offsets(text: str) -> list[int]:
    out = [0]
    for line in text.splitlines(keepends=True):
        out.append(out[-1] + len(line))
    return out


def _span(record: dict[str, Any], expanded: str, trace: _MacroTrace) -> tuple[ModSourceSpan, str]:
    lines = _line_offsets(expanded)
    start = lines[record["line"] - 1] + record["col"] - 1
    end = lines[record["end_line"] - 1] + record["end_col"] - 1
    origins = [dict(x) for x in trace.spans if x["end"] > start and x["start"] < end]
    first = origins[0] if origins else {"source": trace.root, "line": record["line"],
                                      "exact_columns": True, "include_chain": [trace.root]}
    last = origins[-1] if origins else first
    # Whole original lines are mapped; an interpolation invalidates column precision.
    exact = first["exact_columns"] and last["exact_columns"]
    same = first["source"] == last["source"]
    location = ModSourceSpan(first["source"], first["line"], start - first.get("start", 0) + 1 if exact else None,
        last["line"] if same else None, end - last.get("start", 0) + 1 if same and exact else None,
        record["line"], record["col"], record["end_line"], record["end_col"],
        tuple(first["include_chain"]), first.get("expansion_instance"), tuple(origins))
    return location, expanded[start:end]


def _data_schema_error(data: Any, variables: list[str], varobs: list[str]) -> str | None:
    if not isinstance(data, pd.DataFrame):
        return "data must be a pandas DataFrame"
    if data.columns.has_duplicates:
        return "data columns must be unique"
    if data.empty:
        return "data must contain observations"
    expected = varobs or [name for name in variables if name in data.columns]
    missing = set(expected) - set(data.columns)
    if missing:
        return f"Missing observable columns: {sorted(missing)}"
    if not expected:
        return "data has no declared endogenous columns"
    try:
        values = data.loc[:, expected].to_numpy(dtype=float)
    except (TypeError, ValueError):
        return "observable data must contain numeric values (NaN is allowed)"
    if np.isinf(values).any():
        return "observable data cannot contain infinity (NaN is allowed)"
    return None


def inspect_mod(
    path_or_text: str | Path,
    *,
    entry_point: str = "execute_mod_commands",
    ordered: bool = False,
    data: Any = None,
    household_model: Any = None,
    base_dir: str | Path | None = None,
    macro_limit: int = 100_000,
) -> ModInspectionReport:
    """Inspect source without evaluation, compilation, solving, plotting or writes.

    ``entry_point`` is ``load_mod``, ``execute_mod_commands``, or
    ``load_hank_mod``. ``ordered=True`` describes source-ordered execution at a
    fixed calibration, including deterministic boundary and shock blocks.
    Local macro includes are read, with bounded macro expansion.
    Supplied ``data`` is checked for column/schema compatibility only.
    Calibration expressions and numerical option values are deliberately not
    evaluated; symbolic command options are unresolved in ordered mode.
    """
    if entry_point not in {"load_mod", "execute_mod_commands", "load_hank_mod"}:
        raise ValueError("entry_point must be load_mod, execute_mod_commands, or load_hank_mod")
    if macro_limit <= 0:
        raise ValueError("macro_limit must be positive")
    issues: list[ModIssue] = []
    statements: list[ModStatement] = []
    trace = _MacroTrace("<input>", limit=macro_limit)
    coverage = "complete"
    declarations: dict[str, int] = {}
    parser = None
    tokenizer = None
    expanded = ""
    try:
        text, source, source_dir = _read_source(path_or_text)
        trace = _MacroTrace(source, limit=macro_limit)
        trace.source(source, text)
        expanded, trace = _preprocess_with_provenance(text, source=source,
            base_dir=Path(base_dir).resolve() if base_dir is not None else source_dir,
            limit=macro_limit)
        tokenizer = Tokenizer(expanded, strict=True)
        parser = Parser(tokenizer.tokenize(), raw_text=expanded, collect_only=True)
        dag = parser.parse()
        declarations = dict(variables=len(dag.variables), shocks=len(dag.shocks),
            parameters=len(dag.parameters), equations=len(dag.equations),
            model_locals=len(dag.local_variables), observables=len(dag.varobs or []))
        if not dag.variables or not dag.equations:
            issues.append(ModIssue("MISSING_MODEL", "unsupported", "No endogenous variables or model equations were collected.", "Add var declarations and a model block."))
    except Exception as exc:
        coverage = "incomplete"
        trace = getattr(exc, "macro_trace", trace)
        error_location = None
        error_line = getattr(exc, "line", None)
        if error_line:
            error_source = getattr(exc, "macro_source", None)
            error_column = getattr(exc, "col", None) or 1
            if error_source is not None or not expanded:
                error_location = ModSourceSpan(error_source or trace.root, error_line, error_column,
                    None, None, 0, 0, 0, 0, tuple(getattr(exc, "macro_chain", ())))
            elif error_line <= len(_line_offsets(expanded)):
                error_location, _ = _span(dict(line=error_line, col=error_column,
                    end_line=error_line, end_col=error_column+1), expanded, trace)
        issues.append(ModIssue("INSPECTION_INCOMPLETE", "unresolved", str(exc),
            "Correct the input or increase macro_limit for a trusted bounded expansion.", error_location))

    records = sorted((parser.source_statements if parser else []) + (tokenizer.comments if tokenizer else []),
                     key=lambda r: (r["line"], r["col"], -r["end_line"], -r["end_col"]))
    deterministic = ordered and any(r["name"] in _PF_COMMANDS for r in records)
    seen_commands: dict[str, int] = {}
    phase_started = False
    assigned_names: set[str] = set()
    option_state = {**_STOCH_DEFAULTS, **_ORDERED_OPTIONS["calib_smoother"]}
    pf_option_state = dict(_PF_DEFAULTS)
    history_seen = False
    terminal_seen = False
    pf_setup_periods: Any = None
    for record in records:
        name, category = record["name"], record["category"]
        loc, excerpt = _span(record, expanded, trace)
        opts = {k: v for k, v in record.get("options", {}).items() if k != "var_list"}
        variables = tuple(record.get("variables", []))
        status, reason, remedy, code = "supported", "Collected syntax for the selected entry point.", "", ""
        option_status: dict[str, str] = {}
        effective_options: dict[str, Any] = {}
        is_command = name in _LEGACY_OPTIONS
        preserves: bool | None = ordered if is_command else None
        if deterministic and name == "initval":
            history_seen = terminal_seen = False
        elif deterministic and name in {"histval", "endval"}:
            history_seen |= name == "histval"
            terminal_seen |= name == "endval"
            if history_seen and terminal_seen:
                issues.append(ModIssue("BOUNDARY_CONFLICT", "unsupported",
                    "The ordered Dynare subset cannot combine histval and endval within one initval experiment.",
                    "Use initval for initial conditions when specifying endval, or start a new initval experiment.", loc, len(statements)))
        if category == "calibration assignment":
            if name in assigned_names:
                status, code = "unsupported", "REPEATED_CALIBRATION_ASSIGNMENT"
                reason = "Repeated assignments are collapsed by the parser rather than evaluated sequentially."
                remedy = "Express the final fixed calibration once per parameter, or split the studies."
            assigned_names.add(name)
            if parser and name not in parser.parameters:
                status, code = "parsed-only", "UNDECLARED_ASSIGNMENT"
                reason = "Assignment targets an undeclared parameter/script variable."
                remedy = "Declare the calibrated parameter explicitly, or move the script into Python."
        if name == "steady" and deterministic:
            allowed_steady = {"tol", "tolf"}
            option_status = {key: "supported" if key in allowed_steady else "unsupported" for key in opts}
            for key, value in opts.items():
                if key not in allowed_steady or not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                    issues.append(ModIssue("UNSUPPORTED_OPTION", "unsupported",
                        f"steady option {key}={value!r} is outside the supported finite positive tolerance subset.",
                        "Use steady(tolf=...) or steady; and select other methods in Python.", loc, len(statements)))
            if len(opts) > 1:
                issues.append(ModIssue("CONFLICTING_OPTIONS", "unsupported", "Specify only one of tol and tolf.",
                    "Use a single tolerance spelling.", loc, len(statements)))
            effective_options = {"tol": opts.get("tolf", opts.get("tol", 1e-10))}
            reason = "Compute and verify the active initial or terminal steady state at this source position."
        elif name == "steady" and opts:
            status, code = "unsupported", "UNSUPPORTED_OPTION"
            reason, remedy = "Initial steady command options are not consumed by ordered loading.", "Pass supported solve options explicitly to load_mod."
        elif name == "steady" and phase_started:
            status, code = "unsupported", "ORDERED_STATE_UNSUPPORTED"
            reason, remedy = "A steady command after numerical execution is a state transition outside the ordered subset.", "Use a separate fixed-calibration study."
        elif category == "external/script statement":
            if name == "steady" and not opts and excerpt.strip() == "steady;" and not phase_started:
                status, reason = "supported", "Initial steady state is computed by load_mod; this statement adds no separate solve."
            else:
                status, code = "unsupported", "UNSUPPORTED_STATEMENT"
                reason, remedy = "Statement is skipped by the native parser/runner.", "Remove it or execute the study explicitly with the Python API."
        elif name in _PARSED_ONLY:
            status, code = "parsed-only", "PARSED_ONLY_DECLARATION"
            reason, remedy = "Declaration is not executed by this entry point.", "Use the relevant estimation, trend, or regime-switching Python API explicitly."
        elif name == "hetagent_block":
            status, code = "requires caller input", "HANK_EXTENSION"
            reason = "hetagent_block is a puremacro extension requiring the load_hank_mod entry point."
            remedy = "Use load_hank_mod; its built-in household is selected unless household_model is supplied."
            if entry_point == "load_hank_mod" and not ordered:
                status, code = "supported", ""
                reason += (" Caller household supplied; it was not evaluated." if household_model is not None
                           else " The configured built-in household is selected; it was not solved.")
            if parser:
                consumed = {"a_bar", "a_max", "assets", "b_max", "b_min", "chi_0", "chi_1", "model", "n_a", "n_b"}
                for key in sorted(set(parser.hetagent_config or {}) - consumed):
                    issues.append(ModIssue("HANK_OPTION_NOT_CONSUMED", "parsed-only",
                        f"hetagent_block field {key!r} is retained but not consumed by HANKModel.",
                        "Use the supported household configuration or pass a household_model explicitly.", loc, len(statements)))
        elif is_command:
            if ordered and name not in _ORDERED_OPTIONS:
                status, code = "unsupported", "ORDERED_COMMAND_UNSUPPORTED"
                reason, remedy = "The ordered subset does not implement this command's state transitions.", "Run it through its direct Python API in a separate study."
            elif entry_point == "load_mod" and name not in {"ramsey_policy", "ramsey_model", "stoch_simul"}:
                status, code = "parsed-only", "COMMAND_NOT_EXECUTED"
                reason, remedy = "load_mod builds/solves a model but does not execute this command.", "Call execute_mod_commands with the required inputs."
            elif entry_point == "load_mod" and name == "stoch_simul":
                status, code = "parsed-only", "COMMAND_NOT_EXECUTED"
                reason, remedy = "load_mod uses the approximation order but does not produce simulation results.", "Call execute_mod_commands or model.stoch_simul."
            elif entry_point == "load_hank_mod":
                status, code = "parsed-only", "COMMAND_NOT_EXECUTED"
                reason, remedy = "HANK loading does not execute native DSGE command programs.", "Use the returned HANK model's Python methods."
            allowed = (set(_ORDERED_OPTIONS.get(name, {})) if ordered else _LEGACY_OPTIONS.get(name, set()))
            if entry_point == "load_mod" and name == "stoch_simul":
                allowed = {"order"}
            for option, value in opts.items():
                supported = option in allowed
                option_status[option] = "supported" if supported else "unsupported"
                if not supported:
                    issues.append(ModIssue("UNSUPPORTED_OPTION", "unsupported",
                        f"{name} option {option!r} is parsed but is not consumed by this entry point.",
                        "Use a supported option or pass supported arguments directly to the Python method.", loc, len(statements)))
                elif ordered and isinstance(value, str) and not (option == "linear_solver" and value in {"direct", "krylov"}):
                    option_status[option] = "unresolved"
                    issues.append(ModIssue("SYMBOLIC_OPTION", "unresolved",
                        f"Option {option!r} requires expression evaluation, which inspection never performs.",
                        "Use a literal value in the ordered subset.", loc, len(statements)))
            if ordered and name in _ORDERED_OPTIONS:
                state = pf_option_state if name in _PF_COMMANDS else option_state
                state.update(opts)
                if name == "stoch_simul":
                    # Dynare 7.0 generated driver resets this particular option
                    # per statement, unlike ar/irf/filter/sample options.
                    option_state["order"] = opts.get("order", 2)
                elif name == "calib_smoother":
                    option_state["order"] = 1
                if name in _PF_COMMANDS:
                    if "tol" in opts:
                        state["tolf"] = None
                    if "tolf" in opts:
                        state["tol"] = opts["tolf"]
                    if "maxit" in opts:
                        state["max_iter"] = None
                    if "max_iter" in opts:
                        state["maxit"] = opts["max_iter"]
                    for left, right in (("tol", "tolf"), ("maxit", "max_iter")):
                        if left in opts and right in opts:
                            issues.append(ModIssue("CONFLICTING_OPTIONS", "unsupported",
                                f"Specify only one of {left} and {right}.", "Use one spelling per option.", loc, len(statements)))
                effective_options = {key: state[key] for key in _ORDERED_OPTIONS[name]}
                if name in {"perfect_foresight_setup", "simul"}:
                    pf_setup_periods = effective_options["periods"]
                elif name == "perfect_foresight_solver" and pf_setup_periods is None:
                    issues.append(ModIssue("PERFECT_FORESIGHT_SETUP_REQUIRED", "unsupported",
                        "perfect_foresight_solver has no preceding setup command.",
                        "Call perfect_foresight_setup first, or use simul.", loc, len(statements)))
                elif name == "perfect_foresight_solver" and pf_setup_periods != effective_options["periods"]:
                    issues.append(ModIssue("PERFECT_FORESIGHT_SETUP_MISMATCH", "unsupported",
                        "Solver periods differ from the preceding setup.",
                        "Run perfect_foresight_setup with the new horizon first.", loc, len(statements)))
                if name in _PF_COMMANDS and state["linear_solver"] == "krylov":
                    if state["mcp"] or "stack_solve_algo" in opts:
                        issues.append(ModIssue("CONFLICTING_OPTIONS", "unsupported",
                            "Native Krylov cannot be combined with MCP or an explicit Dynare stack_solve_algo.",
                            "Use the direct backend for MCP; choose one backend option.", loc, len(statements)))
                if name == "shock_decomposition" and option_state["prefilter"]:
                    issues.append(ModIssue("INHERITED_OPTION_UNSUPPORTED", "unsupported",
                        "shock_decomposition inherits prefilter from calib_smoother, but its native handler does not implement prefilter.",
                        "Reset prefilter=0 on a preceding calib_smoother, or use a separate study.", loc, len(statements)))
                for key, value in effective_options.items():
                    invalid = False
                    if key in {"order", "irf", "periods", "ar", "drop", "first_obs", "nobs"} and value is not None:
                        invalid = (isinstance(value, bool) or not isinstance(value, (int, float))
                            or not math.isfinite(value) or int(value) != value
                            or value < (1 if key in {"order", "first_obs", "nobs"} else 0))
                        if key == "order":
                            invalid = invalid or value not in {1, 2, 3}
                    elif key in {"prefilter", "pruning"} and value is not None:
                        invalid = value not in (True, False, 0, 1)
                    elif key == "nograph":
                        invalid = value not in (True, 1)
                    elif key == "hp_filter" and value is not None:
                        invalid = not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0
                    elif key in {"tol", "tolf"} and value is not None:
                        invalid = isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
                    elif key in {"maxit", "max_iter"} and value is not None:
                        invalid = isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or int(value) != value or value < 1
                    elif key == "stack_solve_algo":
                        invalid = isinstance(value, bool) or value != 0
                    elif key == "linear_solver":
                        invalid = value not in {"direct", "krylov"}
                    elif key in {"endval_steady", "mcp"}:
                        invalid = value not in (True, False, 0, 1)
                    if key == "periods" and name in _PF_COMMANDS:
                        invalid = invalid or value < 1
                    if invalid:
                        issues.append(ModIssue("UNSUPPORTED_OPTION_VALUE", "unsupported",
                            f"{name} effective option {key}={value!r} is outside the supported subset.",
                            "Use a finite supported literal value (nograph must remain enabled).", loc, len(statements)))
            if variables and parser:
                if len(set(variables)) != len(variables):
                    issues.append(ModIssue("DUPLICATE_VARIABLE_SELECTION", "unsupported",
                        "Requested output variables contain duplicates.", "Select each variable once.", loc, len(statements)))
                unknown = sorted(set(variables) - set(parser.variables))
                if unknown:
                    issues.append(ModIssue("UNKNOWN_VARIABLE_SELECTION", "unsupported",
                        f"Unknown output variables: {unknown}.", "Select declared endogenous variables.", loc, len(statements)))
                elif not ordered:
                    issues.append(ModIssue("VARIABLE_SELECTION_IGNORED", "parsed-only",
                        "The legacy command runner does not preserve this output selection.",
                        "Use ordered=True for supported commands.", loc, len(statements)))
            seen_commands[name] = seen_commands.get(name, 0) + 1
            if seen_commands[name] > 1 and not ordered:
                issues.append(ModIssue("REPEATED_COMMAND_COLLAPSED", "unsupported",
                    "The legacy runner aggregates repeated commands and retains one result.",
                    "Use ordered=True for the supported fixed-calibration subset.", loc, len(statements)))
            if name in {"calib_smoother", "shock_decomposition"} and data is None:
                issues.append(ModIssue("DATA_REQUIRED", "requires caller input",
                    f"{name} requires a caller-supplied DataFrame; datafile options do not load files.",
                    "Pass data= to execute_mod_commands.", loc, len(statements)))
            phase_started = True
        elif ordered and name in _STATEFUL and not (deterministic and name in {"histval", "endval"}):
            status, code = "unsupported", "ORDERED_STATE_UNSUPPORTED"
            reason, remedy = "This state-changing block is outside the ordered fixed-calibration subset.", "Use a separate explicit Python study."
        if phase_started and not is_command and category in {"declaration", "calibration assignment"} and not (deterministic and name in {"initval", "shocks"}):
            status, code = "unsupported", "INTERLEAVED_STATE_CHANGE"
            reason = "Calibration/declarations after a numerical command cannot be replayed against the parser's final calibration."
            remedy = "Move the fixed calibration before all commands, or split the studies."
        if code:
            issues.append(ModIssue(code, status, reason, remedy, loc, len(statements)))
        statements.append(ModStatement(len(statements), name, category, status, loc, excerpt,
            opts, variables, option_status, reason, remedy, preserves, effective_options))

    # Inspection is stricter about incomplete blocks than legacy parsing,
    # whose permissive skipped-token behavior is intentionally unchanged.
    if parser:
        blocks = {"model", "initval", "endval", "histval", "steady_state_model", "shocks",
                  "optim_weights", "osr_params_bounds", "estimated_params", "estimated_params_init",
                  "estimated_params_bounds", "shock_groups", "hetagent_block"}
        for record in records:
            if deterministic and record["name"] in {"initval", "histval", "endval"}:
                boundary_tokens = [t for t in parser.tokens if
                    (record["line"], record["col"]) <= (t.line, t.col) < (record["end_line"], record["end_col"])]
                loc, _ = _span(record, expanded, trace)
                for i, token in enumerate(boundary_tokens[:-1]):
                    if token.value in parser.parameters and boundary_tokens[i + 1].type == "ASSIGN":
                        issues.append(ModIssue("BOUNDARY_PARAMETER_ASSIGNMENT", "unsupported",
                            f"{record['name']} cannot change parameter {token.value!r}.",
                            "Set the fixed calibration before the experiment or use read_mod(params=...).", loc))
                    if token.value in parser.variables + parser.shocks and boundary_tokens[i + 1].type == "LPAREN":
                        tail = boundary_tokens[i + 2:]
                        closing = next((j for j, t in enumerate(tail) if t.type == "RPAREN"), None)
                        if closing is not None and closing + 1 < len(tail) and tail[closing + 1].type == "ASSIGN":
                            index_tokens = tail[:closing]
                            if len(index_tokens) != 1 or index_tokens[0].value != 0:
                                issues.append(ModIssue("HISTORICAL_INDEX_UNSUPPORTED", "unsupported",
                                    "The ordered boundary adapter supports only unindexed values or date zero.",
                                    "Supply an explicitly augmented historical state through the Python API.", loc))
            if record["name"] in {"var", "varexo", "parameters", "shocks", "shock_groups"} and record["category"] not in {"equation", "model-local", "comment"}:
                head_tokens = [t for t in parser.tokens if
                    (record["line"], record["col"]) <= (t.line, t.col) < (record["end_line"], record["end_col"])]
                first_semi = next((i for i, t in enumerate(head_tokens) if t.type == "SEMI"), len(head_tokens))
                declaration_options = [t for t in head_tokens[:first_semi] if t.type == "LPAREN"]
                overwrite = deterministic and record["name"] == "shocks" and [str(t.value) for t in head_tokens[:first_semi]] == ["shocks", "(", "overwrite", ")"]
                if declaration_options and not overwrite:
                    loc, _ = _span(record, expanded, trace)
                    issues.append(ModIssue("UNSUPPORTED_DECLARATION_OPTION", "unsupported",
                        f"Options on the {record['name']} declaration are not implemented by this execution route.",
                        "Use ordinary declarations or the explicit Python API for this feature.", loc))
            if record["name"] in blocks and record["category"] not in {"equation", "model-local"}:
                tokens = [t for t in parser.tokens if
                    (record["line"], record["col"]) <= (t.line, t.col) < (record["end_line"], record["end_col"])]
                if len(tokens) < 2 or tokens[-2].value != "end":
                    coverage = "incomplete"
                    loc, _ = _span(record, expanded, trace)
                    issues.append(ModIssue("UNCLOSED_BLOCK", "unresolved",
                        f"{record['name']} has no closing end;.", "Close the block before inspecting again.", loc))
        for index, tok in enumerate(parser.tokens):
            if str(tok.value).lower() == "steady_state":
                depth = 0
                shock_names = []
                for inner in parser.tokens[index + 1:]:
                    if inner.type == "LPAREN":
                        depth += 1
                    elif inner.type == "RPAREN":
                        depth -= 1
                        if depth == 0:
                            break
                    elif inner.value in parser.shocks:
                        shock_names.append(inner.value)
                if shock_names:
                    loc, _ = _span(dict(line=tok.line, col=tok.col, end_line=tok.line,
                                        end_col=tok.col + len(str(tok.value))), expanded, trace)
                    issues.append(ModIssue("STEADY_STATE_EXOGENOUS", "unsupported",
                        f"STEADY_STATE cannot reference exogenous variables: {sorted(set(shock_names))}.",
                        "Represent the exogenous steady state with a calibrated parameter.", loc))
            if str(tok.value).lower() == "expectation":
                record = dict(line=tok.line, col=tok.col, end_line=tok.line, end_col=tok.col + len(str(tok.value)))
                loc, _ = _span(record, expanded, trace)
                issues.append(ModIssue("EXPECTATION_UNSUPPORTED", "unsupported",
                    "EXPECTATION is parsed but unsupported by compiled residuals.",
                    "Rewrite the information structure explicitly before solving.", loc))
            elif (tok.type == "IDENT" and index + 1 < len(parser.tokens)
                  and parser.tokens[index + 1].type == "LPAREN"
                  and str(tok.value).lower() not in MATH_FUNCTIONS
                  and tok.value not in parser.variables + parser.shocks
                  and tok.value not in parser.local_var_names):
                # Unknown calls in recognized expressions are not external code
                # and must not be mislabeled as executable math.
                in_expression = any((r["category"] in {"equation", "model-local", "calibration assignment"}
                    or r["name"] in {"initval", "steady_state_model", "histval", "endval", "shocks"})
                    and (r["line"], r["col"]) <= (tok.line, tok.col) < (r["end_line"], r["end_col"])
                    for r in records)
                if in_expression:
                    loc, _ = _span(dict(line=tok.line, col=tok.col, end_line=tok.line,
                                        end_col=tok.col + len(str(tok.value))), expanded, trace)
                    issues.append(ModIssue("UNSUPPORTED_FUNCTION", "unsupported",
                        f"Function {tok.value!r} is not implemented by the native expression evaluator.",
                        "Use supported algebra/math functions or a direct Python model.", loc))

    # Traverse the existing syntax DAG, without simplify/eval/diff. This finds
    # indirect exogenous STEADY_STATE references through model locals as well as
    # unsupported function arities hidden in assignment blocks.
    if parser:
        varobs_record = next((r for r in records if r["name"] == "varobs"), None)
        invalid_observables = sorted(set(parser.varobs) - set(parser.variables))
        if invalid_observables or len(set(parser.varobs)) != len(parser.varobs):
            observable_loc = _span(varobs_record, expanded, trace)[0] if varobs_record else None
            issues.append(ModIssue("INVALID_VAROBS", "unsupported",
                f"varobs must contain unique declared endogenous variables; unknown names: {invalid_observables}.",
                "Correct the observable declaration before loading or smoothing.", observable_loc))
        expression_groups: list[tuple[list[Node], dict[str, Any]]] = []
        eq_records = [r for r in records if r["category"] == "equation"]
        expression_groups.extend(([node], r) for node, r in zip(parser.equations, eq_records))
        local_records = [r for r in records if r["category"] == "model-local"]
        expression_groups.extend(([node], r) for node, r in zip(parser.local_var_defs.values(), local_records))
        block_defs = {"initval": parser.initval_defs, "steady_state_model": parser.steady_state_defs,
                      "histval": parser.histval_defs, "endval": parser.endval_defs}
        for record in records:
            if record["name"] in block_defs:
                expression_groups.append((list(block_defs[record["name"]].values()), record))
            elif record["category"] == "calibration assignment" and record["name"] in parser.param_assignments:
                expression_groups.append(([parser.param_assignments[record["name"]]], record))
        for record in records:
            if record["name"] in block_defs and record["category"] not in {"equation", "model-local"}:
                definitions = block_defs[record["name"]]
                used_names = set().union(*(node.parameters() for node in definitions.values())) if definitions else set()
                invalid_targets = set(definitions) - set(parser.variables + parser.shocks + parser.parameters) - used_names
                if invalid_targets:
                    loc, _ = _span(record, expanded, trace)
                    issues.append(ModIssue("UNKNOWN_ASSIGNMENT_TARGET", "unsupported",
                        f"Unknown, unused assignment targets in {record['name']}: {sorted(invalid_targets)}.",
                        "Use declared variables/parameters or referenced local intermediates.", loc))
            if record["name"] == "shocks" and record["category"] not in {"equation", "model-local"}:
                loc, _ = _span(record, expanded, trace)
                shock_symbols: set[str] = set()
                for key in ("variances", "stderrs", "covariances", "correlations", "anticipated"):
                    for names in parser.shocks_config.get(key, {}):
                        shock_symbols.update(names if isinstance(names, tuple) else [names])
                unknown_shocks = shock_symbols - set(parser.shocks + parser.det_shocks)
                if unknown_shocks:
                    issues.append(ModIssue("UNKNOWN_SHOCK", "unsupported", f"Undeclared shocks: {sorted(unknown_shocks)}.",
                        "Declare each shock with varexo/varexo_det.", loc))
                if parser.shocks_config.get("anticipated") and ordered and not deterministic:
                    issues.append(ModIssue("ANTICIPATED_SHOCKS_UNSUPPORTED", "unsupported",
                        "The ordered stochastic subset does not consume dated/anticipated shock paths.",
                        "Use a separate perfect-foresight or news-shock Python study.", loc))
        seen_findings: set[tuple[str, int, int]] = set()
        for roots, record in expression_groups:
            location, _ = _span(record, expanded, trace)
            stack: list[tuple[Node, bool, frozenset[str]]] = [(root, False, frozenset()) for root in roots]
            visited_nodes: set[tuple[int, bool]] = set()
            while stack:
                node, steady_context, visiting = stack.pop()
                visit_key = (id(node), steady_context)
                is_cycle = isinstance(node, _LocalRef) and node.name in visiting
                if visit_key in visited_nodes and not is_cycle:
                    continue
                visited_nodes.add(visit_key)
                finding = ""
                detail = ""
                if isinstance(node, _LocalRef):
                    if node.name in visiting:
                        finding, detail = "MODEL_LOCAL_CYCLE", f"Cyclic model-local definition involving {node.name!r}."
                    elif node.name in parser.local_var_defs:
                        stack.append((parser.local_var_defs[node.name], steady_context, visiting | {node.name}))
                elif isinstance(node, Var):
                    if record["category"] == "calibration assignment":
                        finding, detail = "STATE_DEPENDENT_CALIBRATION", "Calibration references a model variable, requiring execution state that the fixed-calibration runner does not implement."
                    elif steady_context and node.name in parser.shocks:
                        finding, detail = "STEADY_STATE_EXOGENOUS", f"STEADY_STATE references exogenous variable {node.name!r}, possibly through a model-local definition."
                    elif node.name not in parser.variables + parser.shocks + parser.det_shocks:
                        finding, detail = "UNDECLARED_SYMBOL", f"Unknown variable or function {node.name!r} in expression."
                elif isinstance(node, Param):
                    if (node.name not in parser.parameters and node.name not in parser.param_assignments
                            and node.name not in block_defs.get(record["name"], {})):
                        finding, detail = "UNDECLARED_SYMBOL", f"Undeclared parameter {node.name!r} in expression."
                elif isinstance(node, Call):
                    fn = node.func.lower()
                    if fn not in MATH_FUNCTIONS or fn in {"gamma", "lgamma"}:
                        finding, detail = "UNSUPPORTED_FUNCTION", f"Function {node.func!r} is not implemented by the native expression evaluator."
                    elif fn not in {"max", "min", "normcdf", "normpdf", "expectation"} and len(node.args) != 1:
                        finding, detail = "UNSUPPORTED_FUNCTION_ARITY", f"Function {node.func!r} expects one argument."
                    elif fn in {"normcdf", "normpdf"} and len(node.args) not in {1, 3}:
                        finding, detail = "UNSUPPORTED_FUNCTION_ARITY", f"Function {node.func!r} expects one or three arguments."
                    elif fn in {"max", "min"} and not node.args:
                        finding, detail = "UNSUPPORTED_FUNCTION_ARITY", f"Function {node.func!r} requires arguments."
                    elif record["category"] == "equation" and (fn in {"max", "min"} or (fn in {"normcdf", "normpdf"} and len(node.args) != 1)):
                        finding, detail = "UNSUPPORTED_DERIVATIVE", f"The compiled symbolic derivative path does not implement this {node.func!r} call."
                    stack.extend((arg, steady_context or fn == "steady_state", visiting) for arg in node.args)
                elif isinstance(node, BinOp):
                    stack.extend([(node.left, steady_context, visiting), (node.right, steady_context, visiting)])
                elif isinstance(node, UnaryOp):
                    stack.append((node.expr, steady_context, visiting))
                finding_key = (finding, record["line"], record["col"])
                if finding and finding_key not in seen_findings:
                    seen_findings.add(finding_key)
                    issues.append(ModIssue(finding, "unsupported", detail,
                        "Use declared symbols and native supported functions before loading.", location))

    # Keep macro directives, including inactive branches, visible in original source.
    for source, text in trace.sources.items():
        for item in _split_into_items(text):
            if item.is_directive:
                loc = ModSourceSpan(source, item.line_num, item.col_num, item.line_num, None,
                                    0, 0, 0, 0)
                statements.append(ModStatement(len(statements), str(item.name), "macro directive",
                    "supported" if coverage == "complete" else "unresolved", loc, item.content,
                    reason="Macro preprocessing only; inactive directives are inventoried, not executed model commands."))
    data_check = "not supplied"
    if data is not None and parser:
        error = _data_schema_error(data, parser.variables, parser.varobs)
        data_check = error or "schema checked; likelihood not evaluated"
        if error:
            issues.append(ModIssue("DATA_SCHEMA", "requires caller input", error,
                "Supply a numeric DataFrame with the declared observable columns."))
    report = ModInspectionReport(
        {name: hashlib.sha256(text.encode("utf-8")).hexdigest() for name, text in sorted(trace.sources.items())},
        entry_point, ordered, coverage, declarations, tuple(statements), tuple(issues), data_check=data_check)
    # Internal immutable source snapshot: ordered loading consumes exactly the
    # text that was inspected, even if an included file subsequently changes.
    object.__setattr__(report, "_expanded_source", expanded)
    return report
