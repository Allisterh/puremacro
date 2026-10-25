"""Compiled equilibrium specifications independent of a solution algorithm.

``read_mod`` compiles the model but never solves a steady state or a local
rational-expectations approximation. The same specification supplies the
nonlinear residuals to transition, perturbation and explicit global adapters.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import inspect
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ._equations import _steady_state_adapter
from .build import ModelError, SteadyStateError, _Vec


def _vector(value: Any, names: Sequence[str], label: str) -> np.ndarray:
    if isinstance(value, Mapping):
        missing = set(names) - set(value)
        if missing:
            raise ValueError(f"{label} is missing {sorted(missing)}")
        out = np.asarray([value[name] for name in names])
    else:
        out = np.asarray(value)
    if out.shape != (len(names),):
        raise ValueError(f"{label} must have shape {(len(names),)}, got {out.shape}")
    return out


def _evaluate_assignments(definitions: Mapping[str, Any] | None, params: Mapping[str, float],
                          previous: Mapping[str, float] | None = None,
                          auxiliary: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, float]:
    """Evaluate an assignment block in its explicit sequential value context."""
    values = dict(previous or {})
    out: dict[str, float] = {}
    aux_map = auxiliary or {}
    for name, node in (definitions or {}).items():
        environment: dict[Any, float] = dict(values)
        environment.update({(key, 0): value for key, value in values.items()})
        for k_name, k_val in values.items():
            if k_name in aux_map:
                info = aux_map[k_name]
                kind = info.get("kind")
                step = int(info.get("step", 0))
                root = str(info.get("root", ""))
                if kind == "lag":
                    environment[(root, -step)] = k_val
                elif kind == "exo_lag":
                    environment[(root, -(step - 1))] = k_val
                elif kind in ("lead", "exo_lead"):
                    environment[(root, step)] = k_val
        value = float(node.eval(environment, {**params, **values}))
        if not np.isfinite(value):
            raise ModelError(f"Nonfinite assignment to {name}")
        values[name] = out[name] = value
    return out


def _propagate_auxiliary_values(
    target: dict[str, float],
    evaluated: Mapping[str, float],
    auxiliary: Mapping[str, Mapping[str, Any]] | None,
    *,
    mode: str = "all",
    indexed_zero: set[str] | None = None,
) -> None:
    """Propagate root variable assignments to auxiliary lead/lag variables unless explicitly overridden."""
    if not auxiliary:
        return
    idx0 = indexed_zero or set()
    for aux_name, info in auxiliary.items():
        if aux_name in evaluated:
            target[aux_name] = float(evaluated[aux_name])
            continue
        root = str(info.get("root", ""))
        kind = info.get("kind")
        if root not in evaluated:
            continue
        if mode == "histval":
            if kind == "lag" and root not in idx0:
                target[aux_name] = float(evaluated[root])
            elif kind == "exo_lag" and int(info.get("step", 0)) == 1 and root not in idx0:
                target[aux_name] = float(evaluated[root])
            elif kind == "lead":
                target[aux_name] = float(evaluated[root])
        else:
            if kind in ("lag", "lead") and root not in idx0:
                target[aux_name] = float(evaluated[root])
            elif kind in ("exo_lag", "exo_lead") and root not in idx0:
                target[aux_name] = float(evaluated[root])


def _evaluate_covariance(config: Mapping[str, Any], names: Sequence[str],
                         params: Mapping[str, float], default: np.ndarray | None = None) -> np.ndarray:
    """Evaluate declared covariance expressions at one parameter snapshot."""
    keys = ("stderrs", "variances", "covariances", "correlations")
    if default is not None and not any(config.get(key) for key in keys):
        return np.array(default, dtype=float, copy=True)
    result = np.zeros((len(names), len(names)))
    for key, power in (("stderrs", 2), ("variances", 1)):
        for name, expression in config.get(key, {}).items():
            result[names.index(name), names.index(name)] = float(expression.eval({}, params)) ** power
    for (left, right), expression in config.get("covariances", {}).items():
        i, j = names.index(left), names.index(right)
        result[i, j] = result[j, i] = float(expression.eval({}, params))
    for (left, right), expression in config.get("correlations", {}).items():
        i, j = names.index(left), names.index(right)
        result[i, j] = result[j, i] = float(expression.eval({}, params)) * np.sqrt(result[i, i] * result[j, j])
    if not np.all(np.isfinite(result)) or np.any(np.diag(result) < 0):
        raise ModelError("Shock covariance must be finite with nonnegative variances")
    return result


@dataclass
class NonlinearModel:
    """A calibrated equilibrium specification, without a solved policy rule.

    ``steady_state`` is an optional supplied analytical declaration, not a
    computed or certified equilibrium. ``guess`` contains the initial values.
    Call :meth:`steady` to compute and check an equilibrium explicitly.
    ``residuals`` accepts arrays in ``variables``/``shocks`` order, while
    ``compile_equations`` exposes the existing named-vector solver protocol.
    """

    variables: tuple[str, ...]
    shocks: tuple[str, ...]
    parameter_values: dict[str, float]
    equation_names: tuple[str, ...]
    steady_state: dict[str, float] | None = None
    guess: dict[str, float] | None = None
    timing: dict[str, tuple[int, ...]] = field(default_factory=dict)
    source_hashes: dict[str, str] = field(default_factory=dict)
    equation_sources: tuple[Any, ...] = ()
    histval_values: dict[str, float] | None = None
    endval_values: dict[str, float] | None = None
    shock_cov: np.ndarray | None = None
    _equations: Callable[..., Any] | None = field(default=None, repr=False)
    _dag: Any = field(default=None, repr=False)
    _compiled: Any = field(default=None, repr=False)
    _mod_source_text: str | None = field(default=None, repr=False)
    _mod_base_dir: Path | None = field(default=None, repr=False)
    _ordered_inspection: Any = field(default=None, repr=False)
    _parameter_overrides: dict[str, float] = field(default_factory=dict, repr=False)

    @classmethod
    def from_equations(
        cls, equations: Callable[..., Any], *, variables: Sequence[str],
        shocks: Sequence[str] = (), params: Mapping[str, float] | None = None,
        steady_state: Mapping[str, float] | Sequence[float] | None = None,
        guess: Mapping[str, float] | Sequence[float] | None = None,
        equation_names: Sequence[str] | None = None,
        timing: Mapping[str, Sequence[int]] | None = None,
        shock_cov: np.ndarray | None = None,
    ) -> NonlinearModel:
        """Wrap an array residual callback without evaluating or solving it.

        The callback accepts ``(lead, current, lag, shocks)`` and may declare
        keyword arguments ``params`` (a mapping) and ``steady_state`` (an array).
        Exceptions raised inside the callback propagate without retrying it.
        """
        names, exogenous = tuple(variables), tuple(shocks)
        if not names or len(set(names + exogenous)) != len(names + exogenous):
            raise ModelError("Declare distinct endogenous and exogenous names")
        if not callable(equations):
            raise TypeError("equations must be callable")
        parameters = {str(k): float(v) for k, v in (params or {}).items()}
        if not all(np.isfinite(v) for v in parameters.values()):
            raise ValueError("parameters must be finite")
        eq_names = tuple(equation_names or [f"equation_{i + 1}" for i in range(len(names))])
        if len(eq_names) != len(names) or len(set(eq_names)) != len(eq_names):
            raise ModelError("equation_names must contain one distinct name per variable")
        signature = inspect.signature(equations)

        def named_equations(lead, curr, lag, innovations, parameter_vector, steady_state=None):
            arrays = [np.asarray(list(v)) for v in (lead, curr, lag, innovations)]
            ss = None if steady_state is None else np.asarray(list(steady_state))
            bound = signature.bind_partial(*arrays)
            if "params" in signature.parameters:
                bound.arguments["params"] = {name: parameter_vector[name] for name in parameters}
            if "steady_state" in signature.parameters:
                bound.arguments["steady_state"] = ss
            bound.apply_defaults()
            return equations(*bound.args, **bound.kwargs)

        def values(value):
            return None if value is None else dict(zip(names, map(float, _vector(value, names, "values"))))

        covariance = None if shock_cov is None else np.array(shock_cov, dtype=float, copy=True)
        if covariance is not None and covariance.shape != (len(exogenous), len(exogenous)):
            raise ValueError("shock_cov shape must match the declared shocks")
        return cls(names, exogenous, parameters, eq_names, values(steady_state), values(guess),
                   timing={name: tuple(shifts) for name, shifts in (timing or {}).items()},
                   shock_cov=covariance, _equations=named_equations)

    @property
    def _params(self) -> dict[str, float]:
        return self.parameter_values

    @property
    def _shock_cov(self) -> np.ndarray | None:
        return self.shock_cov

    def compile_equations(self) -> Callable[..., Any]:
        """Return ``f(lead, current, lag, shocks, params, steady_state=None)``."""
        if self._equations is None:
            raise ModelError("The nonlinear specification has no compiled equations")
        return self._equations

    def residuals(self, lead: Any, current: Any, lag: Any, shocks: Any,
                  steady_state: Any = None) -> np.ndarray:
        """Evaluate original nonlinear equations, preserving complex inputs."""
        arrays = [_vector(v, self.variables, label)
                  for v, label in ((lead, "lead"), (current, "current"), (lag, "lag"))]
        innovations = _vector(shocks, self.shocks, "shocks")
        named = [_Vec(self.variables, v) for v in arrays]
        e = _Vec(self.shocks, innovations, what="shock")
        p = _Vec(tuple(self.parameter_values), tuple(self.parameter_values.values()), what="parameter")
        ss = None if steady_state is None else _Vec(
            self.variables, _vector(steady_state, self.variables, "steady_state"))
        out = np.asarray(_steady_state_adapter(self.compile_equations())(*named, e, p, steady_state=ss))
        if out.shape != (len(self.variables),):
            raise ModelError(f"Expected {len(self.variables)} residuals, got shape {out.shape}")
        return out

    def steady(self, *, exo_values: Mapping[str, float] | Sequence[float] | None = None,
               guess: Mapping[str, float] | Sequence[float] | None = None,
               tol: float = 1e-10) -> dict[str, float]:
        """Compute and verify a static equilibrium without a perturbation solve."""
        from .perfect_foresight import find_steady_state
        if not np.isfinite(tol) or tol <= 0:
            raise ValueError("tol must be finite and positive")
        initial = guess if guess is not None else self.steady_state or self.guess
        y = find_steady_state(self, exo_values=exo_values, guess=initial, tol=tol)
        if isinstance(exo_values, Mapping):
            e = np.array([exo_values.get(name, 0.0) for name in self.shocks])
        else:
            e = np.zeros(len(self.shocks)) if exo_values is None else np.asarray(exo_values)
        residual = self.residuals(y, y, y, e)
        if not np.all(np.isfinite(y)) or not np.all(np.isfinite(residual)) or np.max(np.abs(residual)) > tol:
            raise SteadyStateError(f"steady: no verified equilibrium; maximum residual {np.max(np.abs(residual))}")
        return dict(zip(self.variables, map(float, y)))

    def solve(self, *, order: int = 1, **kwargs: Any) -> Any:
        """Solve a local approximation explicitly using the existing engine."""
        from .dynare import build_dynare
        kwargs.setdefault("steady_state", self.steady_state)
        kwargs.setdefault("guess", self.guess)
        kwargs.setdefault("shock_cov", self.shock_cov)
        if self._dag is not None:
            kwargs.setdefault("states", self._dag.predetermined_variables)
        solved = build_dynare(self.compile_equations(), variables=self.variables,
            shocks=self.shocks, params=self.parameter_values, order=order, **kwargs)
        for name in ("_dag", "_compiled", "_mod_source_text", "_mod_base_dir", "_ordered_inspection"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(solved, name, value)
        return solved

    def solve_perfect_foresight(self, **kwargs: Any) -> Any:
        """Solve a deterministic transition from this unsolved specification."""
        from .perfect_foresight import solve_perfect_foresight
        return solve_perfect_foresight(self, **kwargs)

    def execute_mod_commands(self, data: Any = None, *, sigma: Any = None,
                             ordered: bool = True) -> Any:
        """Execute retained source commands; ordered execution is the default."""
        if not ordered:
            return self.solve().execute_mod_commands(data=data, sigma=sigma)
        from ._mod_execution import _execute_ordered
        return _execute_ordered(self, data, sigma)


def read_mod(path_or_text: str | Path, *, params: Mapping[str, float] | None = None,
             base_dir: str | Path | None = None) -> NonlinearModel:
    """Read and compile a `.mod` specification without solving an equilibrium.

    Parameters override the fixed calibration and re-evaluate analytical
    steady-state and initial-value declarations. Top-level assignments are not
    re-run, following ``load_mod``. Source hashes and equation provenance are
    retained. Unsupported command semantics are checked when executing, not
    treated as a request to run the file while reading it.
    """
    from .dynare import parse_mod
    from .inspection import _read_source, inspect_mod
    _, _, source_dir = _read_source(path_or_text)
    source_dir = Path(base_dir).resolve() if base_dir is not None else source_dir
    report = inspect_mod(path_or_text, ordered=True, base_dir=source_dir)
    if report.coverage != "complete":
        raise ModelError(report.summary())
    expanded = getattr(report, "_expanded_source")
    parsed = parse_mod(expanded, base_dir=source_dir)
    dag = parsed.get("_dag")
    if dag is None:
        raise ModelError("read_mod requires the structured model parser")
    if len(dag.equations) != len(dag.variables):
        raise ModelError(f"Expected one equation per variable, got {len(dag.equations)} equations and {len(dag.variables)} variables")
    overrides = {str(k): float(v) for k, v in (params or {}).items()}
    unknown = set(overrides) - set(dag.parameters)
    if unknown:
        raise ModelError(f"Unknown parameters: {sorted(unknown)}")
    merged = {**dag.parameter_values, **overrides}
    if not all(np.isfinite(v) for v in merged.values()):
        raise ModelError("Parameters must have finite calibrated values")
    dag = replace(dag, parameter_values=merged)
    supplied_ss = dag.evaluate_steady_state(merged)
    initial = dag.evaluate_initval(merged)
    names = tuple(dag.equation_tags.get(i, {}).get("name", f"equation_{i + 1}")
                  for i in range(len(dag.equations)))
    timing = {v: tuple(sorted({shift for eq in dag.equations for name, shift in eq.variables() if name == v}))
              for v in dag.variables}
    hist_eval = _evaluate_assignments(dag.histval, merged, auxiliary=dag.auxiliary_variables)
    if hist_eval:
        _propagate_auxiliary_values(hist_eval, dict(hist_eval), dag.auxiliary_variables, mode="histval")
    end_eval = _evaluate_assignments(dag.endval, merged, auxiliary=dag.auxiliary_variables)
    if end_eval:
        _propagate_auxiliary_values(end_eval, dict(end_eval), dag.auxiliary_variables, mode="all")
    result = NonlinearModel(tuple(dag.variables), tuple(dag.shocks), dict(merged), names,
        steady_state=supplied_ss, guess=initial, timing=timing,
        source_hashes=dict(report.source_hashes),
        equation_sources=tuple(s.location for s in report.statements if s.category == "equation"),
        histval_values=hist_eval or None,
        endval_values=end_eval or None,
        shock_cov=_evaluate_covariance(dag.shocks_config, dag.shocks, merged, dag.shock_cov),
        _equations=parsed["equations"], _dag=dag, _compiled=parsed["_compiled"],
        _mod_source_text=expanded, _mod_base_dir=source_dir, _ordered_inspection=report,
        _parameter_overrides=overrides)
    return result
