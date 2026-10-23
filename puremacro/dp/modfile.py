"""Dynare-style declarations compiled to :class:`puremacro.dp.Model`.

This is Puremacro's dynamic-programming file language, not Dynare's DSGE
grammar. Loading validates declarations but never executes a solver, Python,
MATLAB, a macro, an include, or a shell command. Economic expressions use the
existing DP expression compiler; calibration expressions use a small numeric
AST interpreter (no ``eval``).
"""
from __future__ import annotations

import ast
import hashlib
import operator
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ._expr import ModelSpecError
from ._results import DPSolution
from .model import Model
from .processes import AR1, Jump, Markov

_IDENT = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
_ASSIGN = re.compile(r"([A-Za-z][A-Za-z0-9_]*)\s*=(?!=)\s*(.+)\Z", re.S)
_MAX_SOURCE = 4_000_000
_MAX_ARRAY = 1_000_000
_SOLVE_OPTIONS = {
    "method", "backend", "tol", "howard", "max_iter", "distribution", "xtol", "max_evals",
}
_ENUMS = {"vfi", "egm", "hjb", "numpy", "jax", "cupy", "tauchen", "rouwenhorst"}


@dataclass(frozen=True)
class _Statement:
    text: str
    line: int


def _statements(source: str, label: str) -> list[_Statement]:
    """Split only at top-level semicolons, retaining original line numbers."""
    items: list[_Statement] = []
    buf: list[str] = []
    stack: list[str] = []
    line, first, i = 1, 1, 0
    quote: str | None = None
    started = False
    while i < len(source):
        ch = source[i]
        if quote is not None:
            buf.append(ch)
            if ch == "\\" and i + 1 < len(source):
                i += 1
                buf.append(source[i])
            elif ch == quote:
                quote = None
            line += ch == "\n"
            i += 1
            continue
        if ch == "%" or source.startswith("//", i):
            end = source.find("\n", i)
            i = len(source) if end < 0 else end
            buf.append(" ")
            continue
        if source.startswith("/*", i):
            end = source.find("*/", i + 2)
            if end < 0:
                raise ModelSpecError(f"{label}:{line}: unterminated block comment")
            line += source[i:end + 2].count("\n")
            i = end + 2
            buf.append(" ")
            continue
        if not started and not ch.isspace():
            first, started = line, True
        if ch in "\"'":
            quote = ch
        elif ch in "([":
            stack.append(ch)
        elif ch in ")]":
            if not stack or stack.pop() != { ")": "(", "]": "[" }[ch]:
                raise ModelSpecError(f"{label}:{line}: unmatched {ch!r}")
        if ch == ";" and not stack:
            text = "".join(buf).strip()
            if text:
                items.append(_Statement(text, first))
            buf, started = [], False
        else:
            buf.append(ch)
        line += ch == "\n"
        i += 1
    if quote or stack:
        raise ModelSpecError(f"{label}:{first}: unterminated string or delimiter")
    if "".join(buf).strip():
        raise ModelSpecError(f"{label}:{first}: statement must end with ';'")
    return items


def _count(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) or not np.isfinite(value):
        raise ModelSpecError("grid length must be a positive integer")
    if value != int(value) or not 1 <= value <= _MAX_ARRAY:
        raise ModelSpecError(f"grid length must be an integer in [1, {_MAX_ARRAY}]")
    return int(value)


def _linspace(start, stop, n):
    _grid_args(start, stop, n)
    return np.linspace(start, stop, _count(n))


def _geomspace(start, stop, n):
    _grid_args(start, stop, n)
    return np.geomspace(start, stop, _count(n))


def _logspace(start, stop, n):
    _grid_args(start, stop, n)
    return np.logspace(start, stop, _count(n))


def _full(n, value):
    if np.ndim(value):
        raise ModelSpecError("full requires a scalar fill value")
    return np.full(_count(n), value, dtype=float)


def _grid_args(start, stop, n):
    _count(n)
    if np.ndim(start) or np.ndim(stop):
        raise ModelSpecError("grid endpoints must be scalars")


def _bounded_binary(fn, left, right):
    shape = np.broadcast_shapes(np.shape(left), np.shape(right))
    if np.prod(shape, dtype=np.int64) > _MAX_ARRAY:
        raise ModelSpecError("calibration array is too large")
    return fn(left, right)


def _minimum(left, right):
    return _bounded_binary(np.minimum, left, right)


def _maximum(left, right):
    return _bounded_binary(np.maximum, left, right)


_NUMERIC_FUNCTIONS = {
    "linspace": _linspace, "geomspace": _geomspace, "logspace": _logspace,
    "full": _full, "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "abs": np.abs,
    "min": _minimum, "max": _maximum,
}
_OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv, ast.Pow: operator.pow}


def _matrix_syntax(text: str) -> str:
    # Permit MATLAB row separators with comma-separated columns: [1, 0; 0, 1].
    # Nested Python lists also work. Do not rewrite strings.
    out: list[str] = []
    stack: list[int] = []
    rows: set[int] = set()
    quote = None
    escaped = False
    for ch in text:
        if quote:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            continue
        if ch == "^":
            out.append("**")
            continue
        if ch in "\"'":
            quote = ch
        if ch == "[":
            stack.append(len(out))
        elif ch == ";" and stack:
            rows.add(stack[-1])
            out.append("],[")
            continue
        elif ch == "]" and stack:
            start = stack.pop()
            if start in rows:
                out[start] = "[["
                out.append("]]")
                continue
        out.append(ch)
    return "".join(out)


class _Numbers:
    def __init__(self, parameters: Mapping[str, Any]):
        self.parameters = parameters

    def evaluate(self, text: str, *, enums: bool = False):
        node = self._parse(text)
        with np.errstate(all="raise"):
            return self.visit(node, enums=enums)

    @staticmethod
    def _parse(text: str):
        try:
            node = ast.parse(_matrix_syntax(text), mode="eval").body
        except (SyntaxError, ValueError, RecursionError) as exc:
            raise ModelSpecError(f"invalid calibration expression {text!r}") from exc
        pending: list[tuple[ast.AST, int]] = [(node, 0)]
        count = 0
        while pending:
            current, depth = pending.pop()
            count += 1
            if count > 10_000 or depth > 200:
                raise ModelSpecError("calibration expression is too large or deeply nested")
            pending.extend((child, depth + 1) for child in ast.iter_child_nodes(current))
        return node

    def visit(self, node, *, enums=False):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool, str)):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in self.parameters:
                return self.parameters[node.id]
            if node.id in {"true", "false"}:
                return node.id == "true"
            if enums and node.id in _ENUMS:
                return node.id
            raise ModelSpecError(f"unknown calibration name {node.id!r}")
        if isinstance(node, (ast.List, ast.Tuple)):
            # Bound the collected children before materializing another array.
            # Each child is itself bounded, so even the rejected next child
            # cannot accumulate an arbitrarily large list of temporary arrays.
            children, size = [], 0
            for child in node.elts:
                value = self.visit(child, enums=enums)
                size += np.size(value)
                if size > _MAX_ARRAY:
                    raise ModelSpecError("calibration array is too large")
                children.append(value)
            value = np.asarray(children, dtype=float)
            return self.checked(value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = self.visit(node.operand)
            return self.checked(-value if isinstance(node.op, ast.USub) else +value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            left, right = self.visit(node.left), self.visit(node.right)
            if isinstance(left, (str, bool)) or isinstance(right, (str, bool)):
                raise ModelSpecError("arithmetic operands must be numeric")
            # Floating arithmetic avoids unbounded Python integer exponentiation.
            left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
            if np.broadcast_shapes(left.shape, right.shape) and np.prod(
                np.broadcast_shapes(left.shape, right.shape)
            ) > _MAX_ARRAY:
                raise ModelSpecError("calibration array is too large")
            return self.checked(_OPERATORS[type(node.op)](left, right))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in _NUMERIC_FUNCTIONS:
                raise ModelSpecError(f"unsupported calibration function {node.func.id!r}")
            args, kw = self.arguments(node)
            return self.checked(_NUMERIC_FUNCTIONS[node.func.id](*args, **kw))
        raise ModelSpecError("calibration accepts numbers, arrays, parameters and named numeric functions only")

    def arguments(self, node):
        args = [self.visit(x) for x in node.args]
        kw = {}
        for item in node.keywords:
            if item.arg is None or item.arg in kw:
                raise ModelSpecError("expanded or duplicate keyword arguments are not supported")
            kw[item.arg] = self.visit(item.value, enums=True)
        return args, kw

    @staticmethod
    def checked(value):
        arr = np.asarray(value, dtype=float)
        if arr.size > _MAX_ARRAY or not np.all(np.isfinite(arr)):
            raise ModelSpecError("calibration values must be finite and arrays bounded")
        return float(arr) if arr.ndim == 0 else arr

    def call(self, text, name=None):
        node = self._parse(text)
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            raise ModelSpecError("expected a named declaration call")
        if name is not None and node.func.id != name:
            raise ModelSpecError(f"expected {name}(...)")
        return node.func.id, *self.arguments(node)

    def process(self, text):
        name, args, kw = self.call(text)
        if name in {"ar1", "tauchen", "rouwenhorst"}:
            if name != "ar1":
                if "method" in kw:
                    raise ModelSpecError(f"{name} already specifies the discretization method")
                kw["method"] = name
            obj = AR1(*args, **kw)
            if (not np.isfinite(obj.rho) or abs(obj.rho) >= 1 or
                    not np.isfinite(obj.sigma) or obj.sigma <= 0 or
                    not np.isfinite(obj.m) or obj.m <= 0 or _count(obj.n) < 2):
                raise ModelSpecError("AR1 requires |rho| < 1, sigma > 0, m > 0 and integer n >= 2")
            if _count(obj.n)**2 > _MAX_ARRAY:
                raise ModelSpecError("AR1 transition matrix exceeds the file-reader array limit")
            obj.discretize()
            return obj
        if name in {"markov", "jump"}:
            obj = (Markov if name == "markov" else Jump)(*args, **kw)
            grid, matrix = obj.discretize() if name == "markov" else obj.generator()
            if (grid.size == 0 or not np.all(np.isfinite(grid)) or
                    not np.all(np.isfinite(matrix)) or len(np.unique(grid)) != len(grid)):
                raise ModelSpecError("shock values and transition entries must be finite; values must be distinct")
            return obj
        raise ModelSpecError(f"unsupported shock process {name!r}; use ar1, tauchen, rouwenhorst, markov or jump")


def _read_source(source: str | Path) -> tuple[str, str, str]:
    def read(path):
        with path.open("rb") as stream:
            return stream.read(_MAX_SOURCE + 1)

    if isinstance(source, Path):
        raw, label = read(source), str(source)
    elif isinstance(source, str):
        if "\n" not in source and ";" not in source and source.lower().endswith(".mod"):
            path = Path(source)
            raw, label = read(path), str(path)
        else:
            raw, label = source.encode("utf-8"), "<dp.mod>"
    else:
        raise TypeError("source must be .mod source text or a pathlib.Path")
    if len(raw) > _MAX_SOURCE:
        raise ModelSpecError(f"{label}: source exceeds {_MAX_SOURCE} bytes")
    return raw.decode("utf-8-sig"), label, hashlib.sha256(raw).hexdigest()


def _assignment(text: str) -> tuple[str, str]:
    match = _ASSIGN.fullmatch(text)
    if not match:
        raise ModelSpecError(f"expected 'name = expression', got {text!r}")
    return match[1], match[2]


def _options(text: str, numbers: _Numbers) -> dict[str, Any]:
    _, args, kw = numbers.call(text, "solve")
    if args or set(kw) - _SOLVE_OPTIONS:
        raise ModelSpecError(f"solve accepts named options {sorted(_SOLVE_OPTIONS)}")
    return _validate_options(kw)


def _validate_options(options: Mapping[str, Any]) -> dict[str, Any]:
    kw = dict(options)
    for key in {"max_iter", "howard", "max_evals"} & kw.keys():
        v = kw[key]
        if isinstance(v, bool) or np.ndim(v) or int(v) != v or v < (0 if key == "howard" else 1):
            raise ModelSpecError(f"solve option {key} must be an integer in its allowed range")
        kw[key] = int(v)
    for key in {"tol", "xtol"} & kw.keys():
        if isinstance(kw[key], (bool, np.bool_)) or np.ndim(kw[key]) or not np.isfinite(kw[key]) or kw[key] <= 0:
            raise ModelSpecError(f"solve option {key} must be finite and positive")
    if "distribution" in kw and not isinstance(kw["distribution"], bool):
        raise ModelSpecError("solve option distribution must be true or false")
    if "method" in kw and kw["method"] not in {"vfi", "egm", "hjb"}:
        raise ModelSpecError("solve method must be vfi, egm or hjb")
    if "backend" in kw and kw["backend"] not in {"numpy", "jax", "cupy"}:
        raise ModelSpecError("solve backend must be numpy, jax or cupy")
    return kw


def _applicable_options(model: Model, options: Mapping[str, Any]) -> None:
    method = options.get("method", "hjb" if model._drift else "vfi")
    unused: set[str] = set()
    if method in {"egm", "hjb"}:
        unused.update({"backend", "howard"} & options.keys())
    if model._horizon is not None:
        unused.update({"tol", "max_iter", "howard"} & options.keys())
    if not model._prices:
        unused.update({"xtol", "max_evals"} & options.keys())
    if (model._prices or model._horizon is not None) and options.get("distribution") is False:
        unused.add("distribution=False")
    if unused:
        raise ModelSpecError(f"solve options do not apply to this {method} model: {sorted(unused)}")


def _load(source, *, params=None, functions=None, validate_applicability=True) -> tuple[Model, dict[str, Any]]:
    text, label, digest = _read_source(source)
    statements = _statements(text, label)
    m = Model(Path(label).stem if label != "<dp.mod>" else "dp model")
    values: dict[str, Any] = {}
    numbers = _Numbers(values)
    overrides = {} if params is None else dict(params)
    declared: set[str] = set()
    assigned: set[str] = set()
    names: set[str] = set()
    mode: str | None = None
    seen_blocks: set[str] = set()
    singletons: set[str] = set()
    options: dict[str, Any] = {}
    horizon: dict[str, Any] = {}
    solve_seen = False
    validating = False
    current = _Statement("", 1)

    def once(key):
        if key in singletons:
            raise ModelSpecError(f"duplicate {key!r} declaration")
        singletons.add(key)

    def name_once(name):
        if not _IDENT.fullmatch(name) or name in names or name in declared:
            raise ModelSpecError(f"duplicate or invalid declaration name {name!r}")
        names.add(name)

    try:
        if functions:
            m.functions(**functions)
        for current in statements:
            item = current.text
            if solve_seen:
                raise ModelSpecError("solve(...) must be the final statement")
            if item == "end":
                if mode is None:
                    raise ModelSpecError("end without a block")
                mode = None
                continue
            if item in {"model", "constraints", "aggregates"} or re.match(r"model\s*\(", item):
                if mode is not None:
                    raise ModelSpecError("nested blocks are not supported")
                kind = "model" if item.startswith("model") else item
                if kind in seen_blocks:
                    raise ModelSpecError(f"duplicate {kind} block")
                seen_blocks.add(kind)
                if re.match(r"model\s*\(", item):
                    _, args, kw = numbers.call(item, "model")
                    if args or set(kw) != {"name"} or not isinstance(kw["name"], str):
                        raise ModelSpecError("model accepts only name='model title'")
                    m.name = kw["name"]
                mode = kind
                continue
            if mode == "model":
                if item.startswith("#"):
                    item = item[1:].strip()
                key, expr = _assignment(item)
                if key == "reward":
                    once("reward")
                    m.reward(expr)
                else:
                    name_once(key)
                    m.local(item)
                continue
            if mode == "constraints":
                m.subject_to(item)
                continue
            if mode == "aggregates":
                key, expr = _assignment(item)
                name_once(key)
                m.aggregate(**{key: expr})
                continue
            if re.match(r"parameters\s+", item):
                for key in re.split(r"[\s,]+", item[len("parameters"):].strip()):
                    if not _IDENT.fullmatch(key) or key in declared or key in names:
                        raise ModelSpecError(f"duplicate or invalid parameter {key!r}")
                    declared.add(key)
                continue
            parts = re.split(r"\s+", item, maxsplit=1)
            head, tail = parts[0], parts[1] if len(parts) == 2 else ""
            if head in {"state", "exogenous", "discrete", "price"}:
                key, expr = _assignment(tail)
                name_once(key)
                if head == "exogenous":
                    m.exogenous(key, numbers.process(expr))
                else:
                    value = numbers.evaluate(expr)
                    if head == "state":
                        m.state(key, value)
                    elif head == "discrete":
                        arr = np.asarray(value, dtype=float)
                        if arr.ndim != 1 or not np.all(np.isfinite(arr)):
                            raise ModelSpecError("discrete choices must be a finite vector")
                        m.discrete(key, value)
                    else:
                        m.prices(**{key: value})
                continue
            if head in {"control", "discount", "discount_rate"}:
                if not _IDENT.fullmatch(tail):
                    raise ModelSpecError(f"{head} expects one name")
                if head == "control":
                    name_once(tail)
                    m.control(tail)
                else:
                    once(head)
                    getattr(m, head)(tail)
                continue
            if head == "drift":
                key, expr = _assignment(tail)
                once("drift " + key)
                m.drift(key, expr)
                continue
            if re.match(r"solve\s*\(", item):
                options = _options(item, numbers)
                solve_seen = True
                continue
            key, expr = _assignment(item)
            if key in {"horizon", "survival", "terminal", "newborns", "taste_shocks", "clear"}:
                once(key)
                if key == "clear":
                    m.clear(expr)
                elif key == "taste_shocks":
                    m.taste_shocks(numbers.evaluate(expr))
                else:
                    value = numbers.evaluate(expr)
                    horizon[key] = _count(value) if key == "horizon" else value
                continue
            if key not in declared:
                raise ModelSpecError(f"unknown declaration {key!r}; calibrated names need 'parameters'")
            if key in assigned:
                raise ModelSpecError(f"parameter {key!r} is assigned more than once")
            values[key] = numbers.checked(overrides[key] if key in overrides else numbers.evaluate(expr))
            assigned.add(key)
        if mode is not None:
            raise ModelSpecError(f"missing end; for {mode} block")
        if set(overrides) - declared:
            raise ModelSpecError(f"override names not declared as parameters: {sorted(set(overrides) - declared)}")
        if declared - assigned:
            raise ModelSpecError(f"parameters need assignments: {sorted(declared - assigned)}")
        m.parameters(**values)
        if horizon:
            if "horizon" not in horizon:
                raise ModelSpecError("survival/terminal/newborns require horizon")
            m.horizon(horizon.pop("horizon"), **horizon)
        if "model" not in seen_blocks:
            raise ModelSpecError("declare model; ... reward = ...; end;")
        if validate_applicability:
            _applicable_options(m, options)
        validating = True
        if m._drift:
            from ._hjb import HJBPlan
            HJBPlan(m)
        else:
            m.check()
        # Original bytes are fingerprinted even if they use CRLF or a UTF-8 BOM.
        m._mod_source = {"name": label, "sha256": digest, "text": text}
        m._mod_options = dict(options)
        return m, options
    except (ValueError, TypeError, OverflowError, FloatingPointError, KeyError, RecursionError) as exc:
        error_line = current.line
        if validating:
            # Compiler messages name offending symbols. Locate their first
            # occurrence in the original declarations rather than attributing
            # every late validation failure to the final solve statement.
            quoted = re.findall(r"'([A-Za-z][A-Za-z0-9_]*)'", str(exc))
            for name in quoted:
                found = next((s for s in statements if re.search(r"\b" + re.escape(name) + r"\b", s.text)), None)
                if found is not None:
                    error_line = found.line
                    break
        raise ModelSpecError(f"{label}:{error_line}: {exc}") from exc


def load_mod(source: str | Path, *, params: Mapping[str, Any] | None = None,
             functions: Mapping[str, Any] | None = None) -> Model:
    """Read and validate DP declarations without solving.

    ``params`` overrides declared calibrations before dependent expressions are
    evaluated. ``functions`` explicitly registers Python callables through the
    existing Model.functions API; file text cannot define or import callables.
    A file's optional ``solve(...)`` is validated but not executed here. Call
    :func:`solve_mod` to apply it, or use the returned model's ordinary ``solve``.
    """
    return _load(source, params=params, functions=functions)[0]


def solve_mod(source: str | Path, *, params: Mapping[str, Any] | None = None,
              functions: Mapping[str, Any] | None = None, **solve_options) -> DPSolution:
    """Load a DP file and solve once, applying explicit Python options last."""
    model, options = _load(source, params=params, functions=functions, validate_applicability=False)
    unknown = set(solve_options) - _SOLVE_OPTIONS
    if unknown:
        raise ModelSpecError(f"unsupported solve options: {sorted(unknown)}")
    options.update(solve_options)
    options = _validate_options(options)
    _applicable_options(model, options)
    return model.solve(**options)


__all__ = ["load_mod", "solve_mod"]
