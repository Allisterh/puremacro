"""Budgeted nonlinear transitions and explicit announcement calendars.

Continuation changes an explicitly stated deterministic experiment, never its
equations silently. Domains use monotone coordinates, not clipped policies.
Horizon acceptance compares the same initial experiment over a common window.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Callable, Mapping, Sequence
from typing import Any
import warnings

import numpy as np
import pandas as pd

from ._equations import _steady_state_adapter
from .build import _Vec
from .perfect_foresight import PerfectForesightResult, MCPResult, solve_perfect_foresight
from .stacked_newton import StackedNewtonError


@dataclass(frozen=True)
class TransitionResult:
    """A transition with its continuation and finite-horizon acceptance record.

    ``converged`` requires the exact target experiment and every requested
    horizon check. A failed/budget-limited run retains its last numerical result
    but never labels that intermediate experiment as the requested solution.
    """

    solution: PerfectForesightResult | MCPResult | None
    converged: bool
    reason: str
    solves: int
    continuation_history: tuple[dict[str, Any], ...]
    horizon_history: tuple[dict[str, Any], ...]
    scaled_residual_norm: float
    preconditioner_builds: int = 0

    @property
    def path(self) -> pd.DataFrame:
        """Copy of the last trajectory; inspect ``converged`` before using it."""
        return self.solution.path.copy() if self.solution is not None else pd.DataFrame()

    def to_frame(self) -> pd.DataFrame:
        """Return a copy of the last numerical trajectory."""
        return self.path

    to_dataframe = to_frame

    def to_markdown(self, *, head: int | None = None, index: bool = True, **kwargs) -> str:
        """Render the last trajectory using the shared Markdown formatter."""
        from puremacro.reports import df_to_markdown
        frame = self.to_frame()
        return df_to_markdown(frame.head(head) if head is not None else frame, index=index, **kwargs)

    def to_latex(self, *, head: int | None = None, index: bool = True, **kwargs) -> str:
        """Render the last trajectory using the shared LaTeX formatter."""
        from puremacro.reports import df_to_latex
        frame = self.to_frame()
        return df_to_latex(frame.head(head) if head is not None else frame, index=index, **kwargs)

    def to_typst(self, *, head: int | None = None, index: bool = True, **kwargs) -> str:
        """Render the last trajectory using the shared Typst formatter."""
        from puremacro.reports import df_to_typst
        frame = self.to_frame()
        return df_to_typst(frame.head(head) if head is not None else frame, index=index, **kwargs)

    def plot(self, variables: Sequence[str] | None = None, style: str = "publication",
             *, ax=None, show: bool = False, **kwargs):
        """Plot the retained trajectory, optionally on caller-supplied axes."""
        if self.solution is None:
            raise ValueError("No numerical solution is available to plot")
        figure = self.solution.plot(variables=variables, style=style, ax=ax, **kwargs)
        if show:
            import matplotlib.pyplot as plt
            plt.show()
        return figure

    def summary(self) -> str:
        return (f"Transition: {self.reason}; converged={self.converged}; solves={self.solves}; "
                f"scaled residual={self.scaled_residual_norm:.3g}; "
                f"preconditioner builds={self.preconditioner_builds}")


@dataclass(frozen=True)
class ShockAnnouncement:
    """Beliefs revealed at a one-based date, including a full calendar shock path.

    ``shocks`` has one row per reported date. Later announcements may revise
    current/future shocks, but cannot rewrite already realized shocks. An
    optional ``y_end`` updates the terminal boundary/reference at that date.
    """

    date: int
    shocks: Any
    y_end: Any = None


def _positive(value, name, *, integer=False):
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    if integer and (not isinstance(value, (int, np.integer))):
        raise ValueError(f"{name} must be an integer")
    return int(value) if integer else float(value)


def _metadata(model, y_init, y_end, variable_names):
    names = tuple(variable_names or getattr(model, "variables", ()) or
                  [f"y_{i}" for i in range(len(y_init))])
    def vector(values, label):
        array = np.asarray([values[name] for name in names] if isinstance(values, Mapping) else values,
                           dtype=float).ravel().copy()
        if array.size != len(names) or not np.isfinite(array).all():
            raise ValueError(f"{label} must have one finite entry per variable")
        return array
    return names, vector(y_init, "y_init"), vector(y_end, "y_end"), vector


def _equations(model, names):
    raw = getattr(model, "_dynare_equations", None)
    if raw is None and hasattr(model, "compile_equations"):
        raw = model.compile_equations()
    if raw is not None:
        evaluate = _steady_state_adapter(raw)
        shocks = tuple(model.shocks)
        params = dict(getattr(model, "_params", None) or getattr(model, "parameter_values", {}))
        pvec = _Vec(list(params), list(params.values()))
        def residual(lead, current, lag, exogenous, *, steady_state=None):
            return evaluate(_Vec(names, lead), _Vec(names, current), _Vec(names, lag),
                            _Vec(shocks, np.atleast_1d(exogenous)), pvec,
                            steady_state=_Vec(names, steady_state))
        return residual
    if hasattr(model, "residuals") and callable(model.residuals):
        def array_residual(lead, current, lag, exogenous, *, steady_state=None):
            return model.residuals(lead, current, lag, np.atleast_1d(exogenous), steady_state=steady_state)
        return array_residual
    if callable(model):
        return _steady_state_adapter(model)
    raise TypeError("Expected a nonlinear model or an equilibrium residual callback")


class _Coordinates:
    def __init__(self, names, bounds, scale):
        n = len(names)
        self.scale = _scale(scale, names, "variable_scale")
        self.lower, self.upper = np.full(n, -np.inf), np.full(n, np.inf)
        for key, pair in dict(bounds or {}).items():
            index = names.index(key) if isinstance(key, str) else int(key)
            if not 0 <= index < n or len(pair) != 2:
                raise ValueError("variable_bounds needs valid variables and (lower, upper) pairs")
            lo, hi = pair
            if lo is not None:
                self.lower[index] = float(lo)
            if hi is not None:
                self.upper[index] = float(hi)
        if np.isnan(self.lower).any() or np.isnan(self.upper).any() or np.any(self.lower >= self.upper):
            raise ValueError("variable_bounds must be ordered open intervals")

    def encode(self, y):
        y = np.asarray(y, dtype=float)
        if not np.isfinite(y).all() or np.any(y <= self.lower) or np.any(y >= self.upper):
            raise ValueError("boundary/initial path lies outside the open variable_bounds domain")
        z = y.copy()
        for j, (lo, hi) in enumerate(zip(self.lower, self.upper)):
            if np.isfinite(lo) and np.isfinite(hi):
                z[..., j] = np.log((y[..., j] - lo) / (hi - y[..., j]))
            elif np.isfinite(lo):
                z[..., j] = np.log(y[..., j] - lo)
            elif np.isfinite(hi):
                z[..., j] = -np.log(hi - y[..., j])
        return z / self.scale

    def decode(self, z):
        raw = np.asarray(z) * self.scale
        y = raw.copy()
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            for j, (lo, hi) in enumerate(zip(self.lower, self.upper)):
                if np.isfinite(lo) and np.isfinite(hi):
                    r = raw[..., j]
                    sigmoid = np.where(r.real >= 0, 1 / (1 + np.exp(-r)), np.exp(r) / (1 + np.exp(r)))
                    y[..., j] = lo + (hi - lo) * sigmoid
                elif np.isfinite(lo):
                    y[..., j] = lo + np.exp(raw[..., j])
                elif np.isfinite(hi):
                    y[..., j] = hi - np.exp(-raw[..., j])
        return y

    def derivative(self, z):
        y = self.decode(z)
        derivative = self.scale.copy()
        for j, (lo, hi) in enumerate(zip(self.lower, self.upper)):
            if np.isfinite(lo) and np.isfinite(hi):
                derivative[j] *= (y[j] - lo) * (hi - y[j]) / (hi - lo)
            elif np.isfinite(lo):
                derivative[j] *= y[j] - lo
            elif np.isfinite(hi):
                derivative[j] *= hi - y[j]
        return derivative


def _scale(scale, names, label):
    if scale is None:
        return np.ones(len(names))
    values = np.asarray([scale[name] for name in names] if isinstance(scale, Mapping) else scale,
                        dtype=float)
    if values.ndim == 0:
        values = np.full(len(names), float(values))
    if values.shape != (len(names),) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError(f"{label} must have positive finite entries")
    return values.copy()


class _SteadyPreconditionerCache:
    """Reuse only an exactly identical steady stacked linear operator."""
    def __init__(self):
        self.key = None
        self.operator = None
        self.builds = 0
        self.reuses = 0

    def get(self, problem):
        from .stacked_newton import _build_block_preconditioner
        blocks = problem.steady_blocks()
        key = (problem.n_periods, *((block.shape, block.dtype.str, block.tobytes()) for block in blocks))
        if key != self.key:
            self.operator, _ = _build_block_preconditioner(*blocks, problem.n_periods, "thomas", None)
            self.key = key
            self.builds += 1
        else:
            self.reuses += 1
        return self.operator


def solve_transition(
    model_or_equations: Any,
    *,
    y_init: Any,
    y_end: Any,
    shocks: Any,
    baseline_shocks: Any = None,
    baseline_end: Any = None,
    baseline_params: Mapping[str, float] | None = None,
    continuation: bool = True,
    initial_step: float = 0.25,
    min_step: float = 1 / 1024,
    max_solves: int = 40,
    horizon_tolerance: float | None = None,
    compare_periods: int = 20,
    max_periods: int | None = None,
    shock_tail: Any = None,
    variable_bounds: Mapping | None = None,
    equation_scale: Any = None,
    variable_scale: Any = None,
    variable_names: Sequence[str] | None = None,
    initial_path: Any = None,
    reuse_preconditioner: bool = True,
    **solver_options: Any,
) -> TransitionResult:
    """Solve the exact target transition with optional continuation/refinement.

    The shock array defines the initial horizon. Continuation starts at the
    stated ``baseline_shocks`` (default zero) and ``baseline_end`` (default
    ``y_init``), verifies that problem, then adaptively approaches the target.
    A failed stage halves its step; only a successful lambda=1 is the target.
    ``max_solves`` bounds all attempts, including baseline and horizon checks.
    For a ``NonlinearModel``, ``baseline_params`` also interpolates the named
    parameters toward the model's target calibration. Boundaries remain explicit;
    this does not solve or guess a changing steady state behind the caller's back.

    Horizon doubling requires an explicit ``shock_tail`` and ``max_periods``;
    no assumption about permanent versus transitory shocks is made silently.
    Acceptance uses the maximum absolute change over ``compare_periods``.

    ``variable_bounds`` are open economic domains, mapped through log/logit
    coordinates. ``equation_scale`` divides each equation before solving;
    an ordinary perfect-foresight solution retains original residuals, with
    the scaled norm reported separately. MCP residuals use transformed/scaled
    complementarity units. ``variable_scale`` scales transformed coordinates.
    MCP bounds are distinct from open domains and are passed to the existing
    direct MCP solver. Solver options otherwise follow solve_perfect_foresight.
    """
    max_solves = _positive(max_solves, "max_solves", integer=True)
    initial_step = _positive(initial_step, "initial_step")
    min_step = _positive(min_step, "min_step")
    if initial_step > 1 or min_step > initial_step:
        raise ValueError("Require 0 < min_step <= initial_step <= 1")
    names, initial, terminal, vector = _metadata(model_or_equations, y_init, y_end, variable_names)
    baseline_terminal = initial.copy() if baseline_end is None else vector(baseline_end, "baseline_end")
    target = np.asarray(shocks, dtype=float).copy()
    if target.ndim not in (1, 2) or len(target) < 1 or not np.isfinite(target).all():
        raise ValueError("shocks must be a finite one- or two-dimensional path")
    T = len(target)
    baseline = np.zeros_like(target) if baseline_shocks is None else np.asarray(baseline_shocks, dtype=float).copy()
    if baseline.shape != target.shape or not np.isfinite(baseline).all():
        raise ValueError("baseline_shocks must have the target shock path shape")
    if any(key in solver_options for key in ("periods", "n_periods", "exogenous_path", "y_ss", "surprise")):
        raise ValueError("The explicit shocks/boundaries define this transition; conflicting solver options")
    if horizon_tolerance is not None:
        horizon_tolerance = _positive(horizon_tolerance, "horizon_tolerance")
        compare_periods = _positive(compare_periods, "compare_periods", integer=True)
        if compare_periods > T:
            raise ValueError("compare_periods cannot exceed the initial horizon")
        if max_periods is None or shock_tail is None:
            raise ValueError("horizon refinement requires explicit max_periods and shock_tail")
        max_periods = _positive(max_periods, "max_periods", integer=True)
        if max_periods <= T:
            raise ValueError("max_periods must exceed the initial horizon")
    coords = _Coordinates(names, variable_bounds, variable_scale)
    coords.encode(initial)
    coords.encode(terminal)
    if continuation:
        coords.encode(baseline_terminal)
    scales = _scale(equation_scale, names, "equation_scale")
    evaluate = _equations(model_or_equations, names)
    options = dict(solver_options)
    tolerance = float(options.get("tol", 1e-8))
    _positive(tolerance, "tol")
    cache = _SteadyPreconditionerCache()
    attempts: list[dict[str, Any]] = []
    horizons: list[dict[str, Any]] = []
    solves = 0
    last = None
    last_scaled = float("inf")
    dag = getattr(model_or_equations, "_dag", None)
    params = dict(getattr(model_or_equations, "_params", None) or
                  getattr(model_or_equations, "parameter_values", {}))
    if baseline_params is not None:
        from .model import NonlinearModel
        if not isinstance(model_or_equations, NonlinearModel):
            raise TypeError("baseline_params requires a NonlinearModel from read_mod/from_equations")
        if set(baseline_params) - params.keys() or not all(np.isfinite(v) for v in baseline_params.values()):
            raise ValueError("baseline_params must contain known finite calibrated parameters")

    def attempt(exogenous, boundary, guess, fraction):
        nonlocal solves, last, last_scaled
        solves += 1
        opts = dict(options)
        stage_params = params
        stage_evaluate = evaluate
        stage_model = model_or_equations
        if baseline_params is not None:
            stage_params = dict(params)
            for name, start in baseline_params.items():
                # Preserve the exact endpoints even under severe cancellation;
                # a recorded fraction 1 must mean the caller's actual target.
                stage_params[name] = (float(params[name]) if fraction == 1 else
                                      float(start) if fraction == 0 else
                                      float((1 - fraction) * start + fraction * params[name]))
            stage_model = replace(model_or_equations, parameter_values=stage_params)
            stage_evaluate = _equations(stage_model, names)
        if opts.get("mcp") and opts.get("mcp_bounds") is None:
            # Resolve source tags before replacing the model by a transformed
            # callback; the callback intentionally has no parsing metadata.
            from .perfect_foresight import _mcp_from_tags
            tag_bounds, tag_equations = _mcp_from_tags(stage_model)
            if not tag_bounds:
                raise ValueError("mcp=True requires explicit mcp_bounds or supported model MCP tags")
            opts["mcp_bounds"] = tag_bounds
            if opts.get("mcp_equations") is None:
                opts["mcp_equations"] = tag_equations
        def raw(yp, yc, yl, eps):
            return np.asarray(stage_evaluate(yp, yc, yl, eps, steady_state=boundary))
        def transformed(zp, zc, zl, eps):
            values = [coords.decode(z) for z in (zp, zc, zl)]
            if any(not np.isfinite(value).all() or np.any(value.real <= coords.lower)
                   or np.any(value.real >= coords.upper) for value in values):
                return np.full(len(names), np.inf)
            return raw(*values, eps) / scales
        original_jacobian = opts.pop("jacobian_fn", None)
        if original_jacobian is None and dag is not None and options.get("method", "auto") in ("auto", "analytic"):
            from ._transition_derivatives import TrajectoryJacobian
            original_jacobian = TrajectoryJacobian(dag, stage_params, boundary).blocks
        jacobian = None
        if original_jacobian is not None:
            def jacobian(zp, zc, zl, eps):
                blocks = original_jacobian(coords.decode(zp), coords.decode(zc), coords.decode(zl), eps)
                return tuple(np.asarray(block) * coords.derivative(z)[None, :] / scales[:, None]
                             for block, z in zip(blocks, (zp, zc, zl)))
        policy = opts.pop("terminal_policy", None)
        policy_jacobian = opts.pop("terminal_jacobian", None)
        if policy_jacobian is not None and policy is None:
            raise ValueError("terminal_jacobian requires terminal_policy")
        if policy is not None:
            opts["terminal_policy"] = lambda z: coords.encode(policy(coords.decode(z)))
            if policy_jacobian is not None:
                def mapped_terminal_jacobian(z):
                    physical = coords.decode(z)
                    output = coords.encode(policy(physical.copy()))
                    return (np.asarray(policy_jacobian(physical)) * coords.derivative(z)[None, :]
                            / coords.derivative(output)[:, None])
                opts["terminal_jacobian"] = mapped_terminal_jacobian
        if opts.get("mcp_bounds"):
            converted = {}
            for key, (lo, hi) in opts["mcp_bounds"].items():
                j = names.index(key) if isinstance(key, str) else int(key)
                def convert(value):
                    if value is None:
                        return None
                    point = initial.copy()
                    point[j] = value
                    return float(coords.encode(point)[j])
                converted[key] = (convert(lo), convert(hi))
            opts["mcp_bounds"] = converted
        z0, zend = coords.encode(initial), coords.encode(boundary)
        zguess = coords.encode(guess) if guess is not None else None
        if zguess is not None and zguess.shape != (len(exogenous), len(names)):
            raise ValueError("initial_path must have shape (periods, variables)")

        def evaluate_path(physical):
            residuals = []
            for t, current in enumerate(physical):
                lead = physical[t + 1] if t + 1 < len(physical) else (policy(current.copy()) if policy else boundary)
                lag = physical[t - 1] if t else initial
                residuals.append(raw(lead, current, lag, exogenous[t]))
            return np.asarray(residuals, dtype=float)

        try:
            weights = np.arange(1, len(exogenous) + 1)[:, None] / (len(exogenous) + 1)
            initial_candidate = coords.decode(zguess if zguess is not None else (1 - weights) * z0 + weights * zend)
            initial_residuals = evaluate_path(initial_candidate)
            if not np.isfinite(initial_residuals).all():
                raise FloatingPointError("Nonfinite residual at this continuation stage's initial path")
            krylov_settings = opts.get("krylov_options") or {}
            if (reuse_preconditioner and opts.get("linear_solver") == "krylov" and
                    "preconditioner" not in krylov_settings and "time_block" not in krylov_settings):
                from .stacked_newton import StackedProblem
                problem = StackedProblem(transformed, z0, zend, exogenous, blocks_fn=jacobian)
                opts["krylov_options"] = dict(opts.get("krylov_options") or {}, preconditioner=cache.get(problem))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", RuntimeWarning)
                numerical = solve_perfect_foresight(
                    transformed, y_init=z0, y_end=zend, shocks=exogenous,
                    periods=len(exogenous), initial_path=zguess, variable_names=names,
                    jacobian_fn=jacobian, **opts,
                )
            physical = coords.decode(numerical.path.to_numpy())
            residuals = evaluate_path(physical)
            last_scaled = (float(numerical.residual_norm) if isinstance(numerical, MCPResult) else
                           float(np.max(np.abs(residuals / scales))))
            success = bool(numerical.converged and np.isfinite(last_scaled) and last_scaled <= tolerance)
            replacements = dict(path=pd.DataFrame(physical, index=numerical.path.index, columns=names),
                                converged=success,
                                terminal_error=float(np.max(np.abs(physical[-1] - boundary))))
            if isinstance(numerical, PerfectForesightResult):
                context = dict(mode="perfect_foresight", equations=raw, y_init=initial.copy(),
                               y_end=boundary.copy(), exogenous_path=np.asarray(exogenous).copy(),
                               variable_names=names, terminal_policy=policy)
                if policy is not None:
                    context.update(terminal_policy_input=physical[-1].copy(),
                                   terminal_policy_output=np.asarray(policy(physical[-1].copy())).copy())
                replacements.update(residual_norm=float(np.max(np.abs(residuals))),
                                    initial_residual_norm=float(np.max(np.abs(initial_residuals))), _audit_context=context,
                                    solver_diagnostics=dict(numerical.solver_diagnostics,
                                        scaled_residual_norm=last_scaled, equation_scale=scales.copy(),
                                        original_equation_residual_norm=float(np.max(np.abs(residuals))),
                                        coordinate_transform=True))
            last = replace(numerical, **replacements)
            attempts.append(dict(solve=solves, fraction=fraction, periods=len(exogenous),
                                 converged=success, scaled_residual_norm=last_scaled,
                                 parameters=dict(stage_params),
                                 iterations=numerical.iterations, warnings=tuple(str(w.message) for w in caught)))
            return last if success else None
        except (StackedNewtonError, np.linalg.LinAlgError, FloatingPointError, OverflowError) as exc:
            attempts.append(dict(solve=solves, fraction=fraction, periods=len(exogenous),
                                 converged=False, parameters=dict(stage_params), error=f"{type(exc).__name__}: {exc}"))
            return None

    def result(success, reason):
        return TransitionResult(last, success, reason, solves, tuple(attempts), tuple(horizons),
                                last_scaled, cache.builds)

    accepted = None
    if continuation:
        accepted = attempt(baseline, baseline_terminal, initial_path, 0.0)
        if accepted is None:
            return result(False, "baseline_failed")
        fraction, step = 0.0, initial_step
        while fraction < 1:
            if solves >= max_solves:
                return result(False, "solve_budget_exhausted")
            candidate = min(1.0, fraction + step)
            stage_shocks = target.copy() if candidate == 1 else (1 - candidate) * baseline + candidate * target
            stage_boundary = terminal.copy() if candidate == 1 else (1 - candidate) * baseline_terminal + candidate * terminal
            trial = attempt(stage_shocks, stage_boundary,
                            accepted.path.to_numpy(), candidate)
            if trial is None:
                step *= 0.5
                if step < min_step:
                    return result(False, "continuation_step_exhausted")
            else:
                accepted, fraction = trial, candidate
                step = min(1.0 - fraction, 1.5 * step)
    else:
        accepted = attempt(target, terminal, initial_path, 1.0)
        if accepted is None:
            return result(False, "target_failed")
    if horizon_tolerance is None:
        return result(True, "target_solved")
    assert max_periods is not None
    while True:
        if solves >= max_solves:
            return result(False, "solve_budget_exhausted")
        new_T = min(2 * T, max_periods)
        if new_T <= T:
            return result(False, "horizon_budget_exhausted")
        tail = np.asarray(shock_tail(T, new_T) if callable(shock_tail) else shock_tail, dtype=float)
        try:
            tail = np.broadcast_to(tail, (new_T - T,) + target.shape[1:]).copy()
        except ValueError as exc:
            raise ValueError("shock_tail must match each exogenous row, or return the full extension") from exc
        target = np.concatenate((target, tail), axis=0)
        guess = np.vstack((accepted.path.to_numpy(), np.tile(terminal, (new_T - T, 1))))
        trial = attempt(target, terminal, guess, 1.0)
        if trial is None:
            return result(False, "horizon_solve_failed")
        difference = float(np.max(np.abs(trial.path.iloc[:compare_periods].to_numpy() -
                                         accepted.path.iloc[:compare_periods].to_numpy())))
        horizons.append(dict(periods=T, refined_periods=new_T, compare_periods=compare_periods,
                             max_absolute_difference=difference, tolerance=horizon_tolerance,
                             accepted=difference <= horizon_tolerance))
        accepted, T = trial, new_T
        if difference <= horizon_tolerance:
            return result(True, "horizon_accepted")
        if T == max_periods:
            return result(False, "horizon_budget_exhausted")


def solve_announced_path(
    model_or_equations: Any,
    announcements: Sequence[ShockAnnouncement],
    *,
    y_init: Any,
    y_end: Any,
    variable_names: Sequence[str] | None = None,
    **solver_options: Any,
) -> PerfectForesightResult:
    """Realize paths between dated revisions to previously announced beliefs.

    At each announcement the remaining finite-horizon problem is solved using
    that date's complete belief path, starting from the state actually reached.
    Only dates before the next announcement are implemented. No future revision
    is revealed early. A failed plan stops execution and retains its diagnostics.
    """
    if not announcements:
        raise ValueError("At least one announcement, at date 1, is required")
    conflicts = {"periods", "n_periods", "exogenous_path", "y_ss", "surprise", "initial_path"}.intersection(solver_options)
    if conflicts:
        raise ValueError("The announcement calendar defines each plan; conflicting solver options: "
                         + ", ".join(sorted(conflicts)))
    names, current, terminal, vector = _metadata(model_or_equations, y_init, y_end, variable_names)
    entries = []
    previous_date, previous_shocks = 0, None
    for announcement in announcements:
        date = _positive(announcement.date, "announcement date", integer=True)
        path = np.array(announcement.shocks, dtype=float, copy=True)
        if path.ndim not in (1, 2) or not len(path) or not np.isfinite(path).all():
            raise ValueError("Each announcement requires a finite full-calendar shock path")
        if date <= previous_date or date > len(path):
            raise ValueError("Announcement dates must increase strictly within the calendar")
        if previous_shocks is None:
            if date != 1:
                raise ValueError("The first announcement must be at date 1")
        elif path.shape != previous_shocks.shape or not np.array_equal(path[:date - 1], previous_shocks[:date - 1]):
            raise ValueError("Announcements must keep the calendar shape and previously realized shocks")
        boundary = None if announcement.y_end is None else vector(announcement.y_end, "announcement y_end")
        entries.append((date, path, boundary))
        previous_date, previous_shocks = date, path
    T = len(entries[0][1])
    realized, records = [], []
    iterations, residual = 0, 0.0
    converged = True
    for i, (date, beliefs, boundary) in enumerate(entries):
        if boundary is not None:
            terminal = boundary
        plan = solve_perfect_foresight(model_or_equations, y_init=current, y_end=terminal,
                                     shocks=beliefs[date - 1:], periods=T - date + 1,
                                     variable_names=names, **solver_options)
        records.append(dict(date=date, converged=plan.converged, residual_norm=plan.residual_norm,
                            terminal=terminal.copy(), shocks=beliefs.copy(),
                            solver_diagnostics=getattr(plan, "solver_diagnostics", {})))
        iterations += plan.iterations
        residual = max(residual, plan.residual_norm)
        if not plan.converged:
            converged = False
            break
        next_date = entries[i + 1][0] if i + 1 < len(entries) else T + 1
        implemented = plan.path.iloc[:next_date - date].to_numpy()
        realized.append(implemented)
        current = implemented[-1].copy()
    array = np.vstack(realized) if realized else np.empty((0, len(names)))
    return PerfectForesightResult(
        path=pd.DataFrame(array, index=pd.RangeIndex(1, len(array) + 1, name="t"), columns=names),
        converged=converged, iterations=iterations, residual_norm=residual,
        terminal_error=float(np.max(np.abs(current - terminal))), variable_names=names,
        linear_solver=solver_options.get("linear_solver", "direct"),
        solver_diagnostics=dict(mode="announcements", announcements=tuple(records),
                                completed_periods=len(array), requested_periods=T),
    )


__all__ = ["TransitionResult", "ShockAnnouncement", "solve_transition", "solve_announced_path"]
