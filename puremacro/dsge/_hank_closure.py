"""Explicit finite-horizon aggregate closure for the HANK .mod bridge."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import warnings

import numpy as np
import pandas as pd

from ._ast import BinOp, Call, Const, Node, Param, UnaryOp, Var
from ._results import HANKResult


def _affine(node: Node, params: Mapping[str, float], steady: Mapping[Any, Any]) -> tuple[float, dict[tuple[str, int], float]]:
    """Extract affine coefficients structurally, never approximate nonlinear equations."""
    if isinstance(node, Var):
        return 0.0, {(node.name, node.lead): 1.0}
    if isinstance(node, (Const, Param)):
        return float(node.eval({}, params)), {}
    if isinstance(node, Call):
        if node.func.lower() == "steady_state":
            return float(node.eval(dict(steady), params)), {}
        if any(_affine(arg, params, steady)[1] for arg in node.args):
            raise ValueError("model closure requires affine equations; nonlinear function of a variable")
        return float(node.eval(dict(steady), params)), {}
    if isinstance(node, UnaryOp):
        value, terms = _affine(node.expr, params, steady)
        if node.op not in ("+", "-"):
            raise ValueError("model closure requires affine equations")
        factor = -1.0 if node.op == "-" else 1.0
        return factor * value, {key: factor * val for key, val in terms.items()}
    if not isinstance(node, BinOp):
        raise ValueError("unsupported expression in model closure")
    left, lterms = _affine(node.left, params, steady)
    right, rterms = _affine(node.right, params, steady)
    if node.op in ("+", "-"):
        sign = 1.0 if node.op == "+" else -1.0
        out = dict(lterms)
        for key, val in rterms.items():
            out[key] = out.get(key, 0.0) + sign * val
        return left + sign * right, {key: val for key, val in out.items() if val != 0}
    if node.op == "*" and not (lterms and rterms):
        return left * right, {key: right * val for key, val in lterms.items()} | {
            key: left * val for key, val in rterms.items()}
    if node.op == "/" and not rterms:
        return left / right, {key: val / right for key, val in lterms.items()}
    if node.op in ("^", "**") and not rterms:
        if right == 1:
            return left, lterms
        if right == 0:
            return 1.0, {}
    if not lterms and not rterms:
        return float(node.eval(dict(steady), params)), {}
    raise ValueError("model closure requires affine equations; nonlinear product, power or denominator")


def _mapping(value: Any, name: str, targets: set[str]) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be an explicit non-empty mapping")
    out = dict(value)
    if any(not isinstance(key, str) or not isinstance(val, str) for key, val in out.items()):
        raise ValueError(f"{name} names must be strings")
    if len(set(out.values())) != len(out):
        raise ValueError(f"{name} has duplicate mapped targets")
    unknown = set(out.values()) - targets
    if unknown:
        raise ValueError(f"{name} maps to undeclared names: {sorted(unknown)}")
    return out


def _household_jacobians(model: Any, inputs: dict[str, str], outputs: dict[str, str], T: int) -> dict[str, dict[str, np.ndarray]]:
    from puremacro.models.hank_sequence_space import dp_sequence_jacobian

    if model.household_model is not None:
        if model.is_two_asset:
            from puremacro.models.hank_sequence_space import _TA_INPUT_ALIASES
            canonical = [_TA_INPUT_ALIASES.get(name, name) for name in inputs]
            if len(set(canonical)) != len(canonical):
                raise ValueError("household_inputs contains duplicate aliases of the same household price")
        # Raw declared household coordinates: no hidden fiscal or timing closure.
        return dp_sequence_jacobian(model.household_model, inputs=list(inputs), outputs=list(outputs), T=T)
    if model.is_two_asset:
        names = {"r_b": "rb", "r_a": "ra", "Y": "Y"}
        allowed_outputs = {"C", "D", "A", "B", "CHI", "z"}
    else:
        names = {"r": "r", "Y": "Y"}
        allowed_outputs = {"C"}
    if set(inputs) - names.keys() or set(outputs) - allowed_outputs:
        raise ValueError(f"unsupported built-in household mapping; inputs={sorted(names)}, outputs={sorted(allowed_outputs)}")
    flat = model.compute_jacobians(T=T)
    return {out: {inp: flat[f"J_{out}_{names[inp]}"] for inp in inputs} for out in outputs}


def _nonlinear_household(model: Any, inputs: dict[str, str], outputs: dict[str, str], T: int):
    """Prepare once; subsequent calls evaluate household transitions only."""
    from puremacro.models.hank_sequence_space import (
        _prepare_dp_household_block, _dp_transition_from_block,
        _household_transition, _two_asset_closed_transition,
    )
    checks: dict[str, float] = {}
    if model.household_model is not None:
        block = _prepare_dp_household_block(model.household_model, inputs=list(inputs), outputs=list(outputs))

        def dp_response(paths):
            return _dp_transition_from_block(block, paths, check_feasibility=True, diagnostics=checks)
        return dp_response, checks
    hh = model._hh_block
    if model.is_two_asset:
        baseline = {key: float(model.steady_state[key]) for key in outputs if key != "z"}
        baseline["z"] = hh.w_ss

        def two_asset_response(paths):
            Y = hh.Y_ss + np.r_[paths.get("Y", np.zeros(T)), 0.0]
            rb = hh.r_b_ss + np.r_[0.0, paths.get("r_b", np.zeros(T))]
            ra = hh.r_a_ss + np.r_[0.0, paths.get("r_a", np.zeros(T))]
            if np.any(Y <= 0) or np.any(rb <= -1) or np.any(ra <= -1):
                raise ValueError("nonlinear household trial has nonpositive output or gross returns")
            # The existing transition warns for floored consumption or a failed
            # income fixed point. Neither is an admissible GE trial.
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                response = _two_asset_closed_transition(hh, Y, rb, ra, diagnostics=checks)
            return {key: response[key][:T] - baseline[key] for key in outputs}
        return two_asset_response, checks

    def one_asset_response(paths):
        rr = hh.r_ss + np.r_[0.0, paths.get("r", np.zeros(T))]
        wages = hh.w_ss * (1.0 + paths.get("Y", np.zeros(T)) / hh.C_ss)
        if np.any(rr <= -1) or np.any(wages <= 0):
            raise ValueError("nonlinear household trial has nonpositive wages or gross returns")
        retained: dict[str, np.ndarray] = {}
        C, consumption, saving = _household_transition(hh, rr, wages, audit_paths=retained)
        resources = np.array([(1+rr[t])*hh.a_grid[:, None] + hh.net_wage(rr[t], wages[t])*hh.s_grid[None, :] for t in range(T)])
        if not np.all(np.isfinite(consumption)) or np.any(consumption <= 0) or np.max(np.abs(consumption+saving-resources)) > 1e-8:
            raise ValueError("nonlinear household trial violates consumption feasibility or its budget")
        distributions = retained["distributions"]
        mass_error = float(np.max(np.abs(distributions.sum(axis=(1, 2))-1.0)))
        if not np.all(np.isfinite(distributions)) or np.min(distributions) < -1e-12 or mass_error > 1e-10:
            raise ValueError("nonlinear household trial has an invalid distribution")
        checks.update(household_budget_max_abs=float(np.max(np.abs(consumption+saving-resources))),
                      distribution_mass_max_abs=mass_error)
        return {"C": C - hh.C_ss}
    return one_asset_response, checks


def _broyden(residual, initial: np.ndarray, matrix: np.ndarray, tol: float, max_iter: int):
    """Monotone Broyden iterations with the assembled linear closure as seed."""
    try:
        inverse0 = np.linalg.solve(matrix, np.eye(len(initial)))
    except np.linalg.LinAlgError as exc:
        raise ValueError("nonlinear closure initial Jacobian factorization failed") from exc
    inverse = inverse0.copy()
    x = initial.copy()
    try:
        value = residual(x)
    except (ValueError, FloatingPointError, RuntimeWarning):
        x = np.zeros_like(initial)
        value = residual(x)
    history = [float(np.max(np.abs(value)))]
    iterations, reason = 0, "max_iter reached"
    while history[-1] > tol and iterations < max_iter:
        accepted = False
        for reset in (False, True):
            if reset:
                inverse = inverse0.copy()
            direction = -inverse @ value
            step = 1.0
            for _ in range(16):
                trial = x + step * direction
                try:
                    trial_value = residual(trial)
                    merit = float(np.linalg.norm(trial_value))
                except (ValueError, FloatingPointError, RuntimeWarning):
                    merit = np.inf
                if np.isfinite(merit) and merit <= (1.0 - 1e-4*step)*np.linalg.norm(value):
                    accepted = True
                    break
                step *= .5
            if accepted:
                break
        if not accepted:
            reason = "no feasible residual-decreasing step"
            break
        dx, df = trial - x, trial_value - value
        row = dx @ inverse
        denominator = float(row @ df)
        if abs(denominator) > np.finfo(float).eps * np.linalg.norm(row) * np.linalg.norm(df):
            inverse += np.outer(dx - inverse @ df, row) / denominator
        x, value = trial, trial_value
        iterations += 1
        history.append(float(np.max(np.abs(value))))
    converged = bool(np.all(np.isfinite(value)) and history[-1] <= tol)
    return x, converged, iterations, history, "converged" if converged else reason


def solve_model_closure(model: Any, *, horizon: int, shock: str, magnitude: float, rho: float,
                        household_inputs: Any, household_outputs: Any, shock_paths: Any,
                        tol: float, nonlinear: bool = False, max_iter: int = 100) -> HANKResult:
    """Solve homogeneous linear aggregate equations plus mapped household responses."""
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("closure_tol must be finite and positive")
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("closure_max_iter must be a positive integer")
    dag = model.dag
    if not dag.is_linear:
        raise ValueError("closure='model' currently requires linear equations in level deviations")
    variables, shocks = list(dag.variables), list(dag.shocks)
    if len(set(variables)) != len(variables) or len(set(shocks)) != len(shocks) or set(variables) & set(shocks):
        raise ValueError("endogenous and shock declarations must be distinct and unique")
    inputs = _mapping(household_inputs, "household_inputs", set(variables) | set(shocks))
    outputs = _mapping(household_outputs, "household_outputs", set(variables))
    if len(dag.equations) + len(outputs) != len(variables):
        raise ValueError("model equations plus mapped household outputs must equal the endogenous variable count")
    T, n = int(horizon), len(variables)
    paths: dict[str, np.ndarray] = {name: np.zeros(T) for name in shocks}
    if shock_paths is None:
        if shock not in shocks:
            raise ValueError(f"shock {shock!r} is not declared")
        if not np.isfinite(magnitude) or not np.isfinite(rho):
            raise ValueError("magnitude and rho must be finite")
        paths[shock] = magnitude * rho ** np.arange(T)
    else:
        if not isinstance(shock_paths, Mapping):
            raise ValueError("shock_paths must map declared shocks to finite horizon-length arrays")
        if set(shock_paths) - set(shocks):
            raise ValueError("shock_paths contains undeclared shocks")
        for name, path in shock_paths.items():
            values = np.asarray(path, dtype=float)
            if values.shape != (T,) or not np.all(np.isfinite(values)):
                raise ValueError(f"shock path {name!r} must be finite with shape ({T},)")
            paths[name] = values.copy()
    if not all(np.all(np.isfinite(path)) for path in paths.values()):
        raise ValueError("shock paths must be finite")
    steady = {**model.steady_state, **{name: 0.0 for name in shocks}}
    rows = []
    equations, params = tuple(dag.equations), dict(model.params)
    for i, equation in enumerate(equations):
        constant, terms = _affine(equation, params, steady)
        if not np.isfinite(constant) or constant != 0.0:
            raise ValueError(f"model equation {i + 1} is not homogeneous in deviations (constant={constant})")
        if any(name not in set(variables) | set(shocks) or not np.isfinite(val) for (name, _), val in terms.items()):
            raise ValueError(f"model equation {i + 1} contains an unknown name or nonfinite coefficient")
        rows.append(terms)
    jac = _household_jacobians(model, inputs, outputs, T)
    for output in outputs:
        for inp in inputs:
            mat = np.asarray(jac[output][inp], dtype=float)
            if mat.shape != (T, T) or not np.all(np.isfinite(mat)):
                raise ValueError("household Jacobians must be finite horizon-square matrices")
            jac[output][inp] = mat.copy()
    index = {name: i for i, name in enumerate(variables)}
    matrix, rhs = np.zeros((n * T, n * T)), np.zeros(n * T)
    for q, terms in enumerate(rows):
        for (name, shift), coefficient in terms.items():
            for t in range(T):
                date = t + shift
                if not 0 <= date < T:
                    continue  # Explicit zero deviations outside the supplied horizon.
                if name in index:
                    matrix[q * T + t, index[name] * T + date] += coefficient
                else:
                    rhs[q * T + t] -= coefficient * paths[name][date]
    for q, (output, target) in enumerate(outputs.items(), start=len(rows)):
        section = slice(q * T, (q + 1) * T)
        matrix[section, index[target] * T:(index[target] + 1) * T] += np.eye(T)
        for inp, source in inputs.items():
            if source in index:
                matrix[section, index[source] * T:(index[source] + 1) * T] -= jac[output][inp]
            else:
                rhs[section] += jac[output][inp] @ paths[source]
    singular = np.linalg.svd(matrix, compute_uv=False)
    rank_tol = singular[0] * max(matrix.shape) * np.finfo(float).eps
    rank = int(np.count_nonzero(singular > rank_tol))
    if rank != n * T:
        raise ValueError(f"finite-horizon model closure is rank deficient ({rank}/{n * T}); check equations and mappings")
    try:
        solution = np.linalg.solve(matrix, rhs).reshape(n, T)
    except np.linalg.LinAlgError as exc:
        raise ValueError("finite-horizon model closure factorization failed") from exc
    combined = {**dict(zip(variables, solution)), **paths}

    def residuals(candidate: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        aggregate = np.zeros((len(rows), T))
        for q, equation in enumerate(equations):
            coordinates = equation.variables()
            for t in range(T):
                values: dict[Any, Any] = {key: float(candidate[key[0]][t + key[1]]) if 0 <= t + key[1] < T else 0.0 for key in coordinates}
                values["_steady_state"] = steady
                aggregate[q, t] = equation.eval(values, params)
        household = np.array([np.asarray(candidate[target]) - sum(
            (jac[output][inp] @ np.asarray(candidate[source]) for inp, source in inputs.items()), start=np.zeros(T))
            for output, target in outputs.items()])
        return aggregate, household

    nonlinear_info: dict[str, Any] = {}
    converged = True
    if nonlinear:
        household_response, household_checks = _nonlinear_household(model, inputs, outputs, T)

        def nonlinear_residual(vector):
            candidate = {**dict(zip(variables, vector.reshape(n, T))), **paths}
            response = household_response({inp: candidate[source] for inp, source in inputs.items()})
            aggregate, _ = residuals(candidate)
            household = np.array([candidate[target] - response[out] for out, target in outputs.items()])
            value = np.r_[aggregate.ravel(), household.ravel()]
            if not np.all(np.isfinite(value)):
                raise ValueError("nonlinear closure residual is not finite")
            return value

        vector, converged, iterations, history, reason = _broyden(nonlinear_residual, solution.ravel(), matrix, tol, max_iter)
        solution = vector.reshape(n, T)
        combined = {**dict(zip(variables, solution)), **paths}
        response = household_response({inp: combined[source] for inp, source in inputs.items()})
        aggregate, _ = residuals(combined)
        household = np.array([combined[target] - response[out] for out, target in outputs.items()])
        nonlinear_info = {"nonlinear": True, "iterations": iterations, "residual_history": history, "termination": reason, **household_checks}
        if not converged:
            warnings.warn(f"editable nonlinear HANK closure did not converge: {reason}; residual={history[-1]:.3e}. Returned paths do not clear the model.", RuntimeWarning, stacklevel=3)
    else:
        aggregate, household = residuals(combined)
    aggregate_error = float(np.max(np.abs(aggregate), initial=0))
    household_error = float(np.max(np.abs(household), initial=0))
    if not np.all(np.isfinite(solution)) or (converged and max(aggregate_error, household_error) > tol):
        raise RuntimeError("model closure solution failed independent aggregate/household residual checks")
    diagnostics = {"rank": rank, "dimension": n * T, "condition_number": float(singular[0] / singular[-1]),
                   "aggregate_residual_max_abs": aggregate_error, "household_residual_max_abs": household_error,
                   "tolerance": tol, "boundary": "zero deviations outside horizon",
                   "rank_scope": "finite-horizon solvability, not Blanchard-Kahn determinacy",
                   "household_inputs": inputs.copy(), "household_outputs": outputs.copy(), **nonlinear_info}
    context: dict[str, Any] = {"mode": "hank_model_linear", "residuals": residuals,
                              "shock_paths": {name: path.copy() for name, path in paths.items()}}
    if nonlinear:
        # Retain actual final evaluations, never call a hidden household solve
        # from an audit callback. This is not a micro-level Euler certificate.
        context = {"mode": "hank_model_nonlinear", "aggregate_residuals": aggregate.copy(),
                   "aggregate_residuals_fn": lambda candidate: residuals(candidate)[0],
                   "household_residuals": household.copy(),
                   "household_input_paths": {inp: combined[source].copy() for inp, source in inputs.items()},
                   "household_output_paths": {out: response[out].copy() for out in outputs},
                   "shock_paths": {name: path.copy() for name, path in paths.items()}}
    context.update(household_inputs=inputs.copy(), household_outputs=outputs.copy(),
                   variable_names=tuple(variables), horizon=T)
    return HANKResult(
        steady_state=model.steady_state.copy(), transition_paths=pd.DataFrame(combined),
        jacobians={f"J_{out}_{inp}": mat.copy() for out, by_input in jac.items() for inp, mat in by_input.items()},
        asset_distribution=model.asset_distribution.copy(), asset_grid=model.asset_grid.copy(),
        mpc_distribution=model.mpc_distribution, liquid_asset_grid=model.liquid_asset_grid,
        joint_distribution=model.joint_distribution, marginal_distribution_b=model.marginal_distribution_b,
        deposit_distribution=model.deposit_distribution, shock_name=shock if shock_paths is None else ",".join(shock_paths),
        horizon=T, model_name=model.hetagent_config.get("model", "hank_model_closure"), converged=converged,
        closure="model", diagnostics=diagnostics,
        _audit_context=context,
    )
