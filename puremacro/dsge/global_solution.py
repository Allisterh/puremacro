"""Equilibrium time iteration with explicit timing, integration and domains.

The callbacks specify a recursive equilibrium, not necessarily a planner's
Bellman problem. The approximation is global only on the declared domain;
convergence is numerical and does not establish existence or uniqueness.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from copy import deepcopy
from itertools import product
from typing import Any, Callable, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import least_squares

from puremacro.vfi.collocation import CollocationBasis
from puremacro.vfi.smolyak import SmolyakBasis

from ._policy_reporting import _PolicyReport

Array = np.ndarray
Policy = Callable[[Array], Array]


def _vector(value: Any, length: int, label: str) -> Array:
    array = np.asarray(value, dtype=float)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite vector of length {length}")
    return array


def _positive(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise ValueError(f"{name} must be finite and positive")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not np.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return numeric


def _integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _points(value: Any, dimension: int) -> tuple[Array, bool]:
    points = np.asarray(value, dtype=float)
    single = points.ndim == 1
    if single:
        points = points.reshape(1, -1)
    if points.ndim != 2 or points.shape[1] != dimension or not np.isfinite(points).all():
        raise ValueError(f"states must have shape ({dimension},) or (n, {dimension}) and be finite")
    return points, single


def gaussian_quadrature(covariance: Any, *, order: int = 3,
                        max_nodes: int = 10000) -> tuple[Array, Array]:
    """Tensor Gauss-Hermite rule for a zero-mean Gaussian innovation vector.

    Singular positive-semidefinite covariance is supported by integrating only
    its positive-eigenvalue subspace. The rule is finite quadrature, not a
    bounded-support representation of the Gaussian distribution.
    """
    order = _integer(order, "order")
    max_nodes = _integer(max_nodes, "max_nodes")
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square")
    scale = max(float(np.max(np.abs(covariance), initial=0)), np.finfo(float).tiny)
    if not np.isfinite(covariance).all() or not np.allclose(covariance, covariance.T, atol=1e-13 * scale, rtol=1e-12):
        raise ValueError("covariance must be finite and symmetric")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tolerance = 1e-12 * scale
    if np.min(eigenvalues, initial=0) < -tolerance:
        raise ValueError("covariance must be positive semidefinite")
    positive = eigenvalues > 0
    rank = int(np.sum(positive))
    if order ** rank > max_nodes:
        raise ValueError("Gaussian quadrature exceeds max_nodes; reduce dimension/order or supply a rule")
    if rank == 0:
        return np.zeros((1, covariance.shape[0])), np.ones(1)
    nodes_1d, weights_1d = np.polynomial.hermite.hermgauss(order)
    indices = np.array(list(product(range(order), repeat=rank)), dtype=int)
    independent = np.sqrt(2.0) * nodes_1d[indices]
    weights = np.prod(weights_1d[indices] / np.sqrt(np.pi), axis=1)
    loading = eigenvectors[:, positive] * np.sqrt(eigenvalues[positive])
    return independent @ loading.T, weights


@dataclass
class EquilibriumProblem:
    """A recursive equilibrium with an explicit conditional expectation.

    ``transition(state, controls, innovation)`` gives next period's state.
    ``residual(state, controls, next_state, next_controls, innovation)`` gives
    one residual per control. The solver takes its probability-weighted mean.
    ``shock_nodes`` are future innovations, not current innovations.
    An optional ``shock_rule(state)`` supplies state-dependent nodes and
    probabilities instead, for example a conditional Markov transition row.

    Bounds can be an (n_controls, 2) array or ``control_bounds(state)``. All
    bounds constrain admissible solutions. ``complementarity`` maps a residual
    index to its control index: at a lower bound its residual must be >= 0,
    at an upper bound <= 0, and in the interior zero. Other residuals must
    vanish even at a bound. ``residual_scale`` is a positive equation scale.
    """

    state_names: Sequence[str]
    control_names: Sequence[str]
    domain: Sequence[tuple[float, float]]
    transition: Callable[[Array, Array, Array], Array]
    residual: Callable[[Array, Array, Array, Array, Array], Array]
    shock_nodes: Any = field(default_factory=lambda: np.zeros((1, 0)))
    shock_weights: Any = field(default_factory=lambda: np.ones(1))
    control_bounds: Any = None
    complementarity: Mapping[int, int] = field(default_factory=dict)
    residual_scale: Any = None
    admissible: Callable[[Array, Array], bool] | None = None
    equation_names: Sequence[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    shock_rule: Callable[[Array], tuple[Array, Array]] | None = None
    transition_jacobian: Callable[[Array, Array, Array], Array] | None = None
    residual_jacobian: Callable[[Array, Array, Array, Array, Array], tuple[Array, Array, Array]] | None = None
    batched: bool = False

    def __post_init__(self) -> None:
        self.state_names = tuple(self.state_names)
        self.control_names = tuple(self.control_names)
        if not self.state_names or not self.control_names:
            raise ValueError("state_names and control_names must be nonempty")
        if len(set(self.state_names)) != len(self.state_names) or len(set(self.control_names)) != len(self.control_names):
            raise ValueError("state and control names must be unique within each group")
        domain = np.asarray(self.domain, dtype=float)
        if domain.shape != (self.n_states, 2) or not np.isfinite(domain).all() or np.any(domain[:, 0] >= domain[:, 1]):
            raise ValueError("domain must contain finite increasing bounds for every state")
        self.domain = tuple(map(tuple, domain.tolist()))
        nodes = np.array(self.shock_nodes, dtype=float, copy=True)
        weights = np.array(self.shock_weights, dtype=float, copy=True)
        if nodes.ndim != 2 or len(nodes) < 1 or not np.isfinite(nodes).all():
            raise ValueError("shock_nodes must be a finite nonempty matrix")
        if weights.shape != (len(nodes),) or not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("shock_weights must be a finite nonnegative vector matching shock_nodes")
        if not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-12):
            raise ValueError("shock_weights must sum to one; no silent normalization is performed")
        positive = weights > 0
        nodes, weights = nodes[positive], weights[positive]
        nodes.setflags(write=False)
        weights.setflags(write=False)
        self.shock_nodes, self.shock_weights = nodes, weights
        scale = np.ones(self.n_controls) if self.residual_scale is None else np.array(self.residual_scale, dtype=float, copy=True)
        if scale.shape != (self.n_controls,) or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("residual_scale must be positive and finite, one entry per equation")
        scale.setflags(write=False)
        self.residual_scale = scale
        self.complementarity = dict(self.complementarity)
        for equation, control in self.complementarity.items():
            if not isinstance(equation, int) or not isinstance(control, int) or not (0 <= equation < self.n_controls and 0 <= control < self.n_controls):
                raise ValueError("complementarity must map valid equation indices to control indices")
        if len(set(self.complementarity.values())) != len(self.complementarity):
            raise ValueError("a control can have only one complementarity equation")
        if self.equation_names is None:
            self.equation_names = tuple(f"equation_{i + 1}" for i in range(self.n_controls))
        else:
            self.equation_names = tuple(self.equation_names)
        if len(self.equation_names) != self.n_controls or len(set(self.equation_names)) != self.n_controls:
            raise ValueError("equation_names must be unique and match the number of controls")
        if not callable(self.transition) or not callable(self.residual):
            raise TypeError("transition and residual must be callable")
        if self.transition_jacobian is not None and not callable(self.transition_jacobian):
            raise TypeError("transition_jacobian must be callable")
        if self.residual_jacobian is not None and not callable(self.residual_jacobian):
            raise TypeError("residual_jacobian must be callable")
        self.bounds(np.mean(domain, axis=1))
        if self.shock_rule is not None:
            if not callable(self.shock_rule):
                raise TypeError("shock_rule must be callable")
            self.integration(np.mean(domain, axis=1))

    @property
    def n_states(self) -> int:
        return len(self.state_names)

    @property
    def n_controls(self) -> int:
        return len(self.control_names)

    def bounds(self, state: Array) -> tuple[Array, Array]:
        if self.control_bounds is None:
            return np.full(self.n_controls, -np.inf), np.full(self.n_controls, np.inf)
        raw = self.control_bounds(state) if callable(self.control_bounds) else self.control_bounds
        bounds = np.asarray(raw, dtype=float)
        if bounds.shape != (self.n_controls, 2) or np.isnan(bounds).any() or np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("control_bounds must contain increasing bounds, one pair per control")
        return bounds[:, 0], bounds[:, 1]

    def contains(self, states: Any) -> Array:
        points, _ = _points(states, self.n_states)
        domain = np.asarray(self.domain)
        slack = 1e-12 * np.maximum(1, domain[:, 1] - domain[:, 0])
        return np.all((points >= domain[:, 0] - slack) & (points <= domain[:, 1] + slack), axis=1)

    def integration(self, state: Array) -> tuple[Array, Array]:
        """Conditional future innovation nodes/weights, including Markov rules.

        A supplied ``shock_rule(state)`` overrides the fixed rule, allowing
        state-dependent transition probabilities. It is validated at every
        evaluation; invalid probabilities are never renormalized.
        """
        if self.shock_rule is None:
            return self.shock_nodes, self.shock_weights
        nodes, weights = (np.asarray(value, dtype=float) for value in self.shock_rule(state))
        if nodes.ndim != 2 or len(nodes) < 1 or not np.isfinite(nodes).all():
            raise ValueError("shock_rule nodes must be a finite nonempty matrix")
        if (weights.shape != (len(nodes),) or not np.isfinite(weights).all()
                or np.any(weights < 0) or not np.isclose(weights.sum(), 1, rtol=0, atol=1e-12)):
            raise ValueError("shock_rule weights must be nonnegative and sum to one")
        positive = weights > 0
        return nodes[positive], weights[positive]

    @classmethod
    def from_model(cls, model: Any, *, state_names: Sequence[str],
                   control_names: Sequence[str], domain: Sequence[tuple[float, float]],
                   transition: Callable[[Array, Array, Array], Array],
                   current: Callable[[Array, Array], Array],
                   lagged: Callable[[Array], Array], equations: Sequence[int | str],
                   exogenous: Callable[[Array], Array] | None = None,
                   steady_state: Any = None, **kwargs: Any) -> "EquilibriumProblem":
        """Bind explicit recursive timing to a compiled nonlinear model.

        ``current`` constructs the full endogenous vector from a state and
        controls; it also constructs the lead vector at the next state.
        ``lagged`` supplies the historical values used in the selected
        equations. Equations eliminated by the transition/current mapping
        must not be selected again. The model is copied so later calibration
        edits do not change this problem's residuals. User callback closures
        must preserve their own calibration. No planner objective is inferred.
        """
        model = deepcopy(model)
        names = tuple(getattr(model, "equation_names", ()))
        if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (str, int, np.integer)) for i in equations):
            raise ValueError("selected equations must be names or integer indices")
        indices = tuple(names.index(i) if isinstance(i, str) else int(i) for i in equations)
        if len(indices) != len(control_names) or len(set(indices)) != len(indices):
            raise ValueError("select exactly one distinct equation per control")
        n_variables = len(model.variables)
        n_equations = len(names) if names else n_variables
        if any(i < 0 or i >= n_equations for i in indices):
            raise ValueError("selected equation index is out of range")
        n_shocks = len(model.shocks)
        if isinstance(steady_state, Mapping):
            if set(model.variables) - set(steady_state):
                raise ValueError("steady_state mapping must contain all endogenous variables")
            steady_state = [steady_state[name] for name in model.variables]
        reference = None if steady_state is None else _vector(steady_state, n_variables, "steady_state").copy()

        def residual(state: Array, control: Array, next_state: Array,
                     next_control: Array, innovation: Array) -> Array:
            yp = _vector(current(next_state, next_control), n_variables, "lead endogenous mapping")
            yc = _vector(current(state, control), n_variables, "current endogenous mapping")
            yl = _vector(lagged(state), n_variables, "lagged endogenous mapping")
            shocks = np.zeros(n_shocks) if exogenous is None else _vector(exogenous(state), n_shocks, "current shock mapping")
            values = np.asarray(model.residuals(yp, yc, yl, shocks, steady_state=reference))
            return values[list(indices)]

        selected_names = tuple(names[i] if names else f"equation_{i + 1}" for i in indices)
        dag = getattr(model, "_dag", None)
        ref_vec: Array | None = reference
        if ref_vec is None and dag is not None:
            ss_map = getattr(model, "steady_state", None) or getattr(dag, "steady_state", None)
            if isinstance(ss_map, Mapping) and not (set(model.variables) - set(ss_map)):
                ref_vec = np.asarray([ss_map[name] for name in model.variables], dtype=float)
            elif "STEADY_STATE" not in "".join(repr(eq) for eq in getattr(dag, "equations", ())):
                ref_vec = np.zeros(n_variables, dtype=float)
        if dag is not None and ref_vec is not None and "residual_jacobian" not in kwargs:
            from ._transition_derivatives import TrajectoryJacobian
            params = dict(getattr(model, "_params", None) or getattr(model, "parameter_values", {}))
            sym_jac = TrajectoryJacobian(dag, params, ref_vec)
            row_idx = np.asarray(indices, dtype=int)
            n_states = len(state_names)
            n_controls = len(control_names)

            def _current_partials(s_val: Array, x_val: Array) -> tuple[Array, Array, Array]:
                base_y = _vector(current(s_val, x_val), n_variables, "current endogenous mapping")
                dy_ds = np.empty((n_variables, n_states), dtype=float)
                dy_dx = np.empty((n_variables, n_controls), dtype=float)
                for j in range(n_states):
                    h = 1e-7 * max(1.0, abs(float(s_val[j])))
                    sp, sm = s_val.copy(), s_val.copy()
                    sp[j] += h
                    sm[j] -= h
                    dy_ds[:, j] = (_vector(current(sp, x_val), n_variables, "current endogenous mapping") -
                                   _vector(current(sm, x_val), n_variables, "current endogenous mapping")) / (2.0 * h)
                for j in range(n_controls):
                    h = 1e-7 * max(1.0, abs(float(x_val[j])))
                    xp, xm = x_val.copy(), x_val.copy()
                    xp[j] += h
                    xm[j] -= h
                    dy_dx[:, j] = (_vector(current(s_val, xp), n_variables, "current endogenous mapping") -
                                   _vector(current(s_val, xm), n_variables, "current endogenous mapping")) / (2.0 * h)
                return base_y, dy_ds, dy_dx

            def residual_jacobian(state: Array, control: Array, next_state: Array,
                                  next_control: Array, innovation: Array) -> tuple[Array, Array, Array]:
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    yp, dyp_dsp, dyp_dxp = _current_partials(next_state, next_control)
                    yc, _, dyc_dx = _current_partials(state, control)
                    yl = _vector(lagged(state), n_variables, "lagged endogenous mapping")
                    shocks = np.zeros(n_shocks) if exogenous is None else _vector(exogenous(state), n_shocks, "current shock mapping")
                    A, B, _ = sym_jac.steady_or_date_blocks(yp, yc, yl, shocks)
                    A_sel = A[row_idx, :]
                    B_sel = B[row_idx, :]
                    return B_sel @ dyc_dx, A_sel @ dyp_dsp, A_sel @ dyp_dxp

            kwargs["residual_jacobian"] = residual_jacobian
        return cls(state_names, control_names, domain, transition, residual,
                   equation_names=selected_names, **kwargs)


@dataclass(frozen=True)
class PolicyAudit(_PolicyReport):
    """Independent pointwise expected-equation, KKT and domain evidence."""

    points: pd.DataFrame
    residuals: pd.DataFrame
    kkt_residuals: pd.DataFrame
    diagnostics: pd.DataFrame
    tolerance: float
    passed: bool
    require_domain: bool = True

    @property
    def max_residual(self) -> float:
        return float(np.max(np.abs(self.residuals.to_numpy())))

    @property
    def max_kkt_residual(self) -> float:
        return float(np.max(np.abs(self.kkt_residuals.to_numpy())))

    def summary(self) -> str:
        return (f"Policy audit: passed={self.passed}; max scaled KKT residual="
                f"{self.max_kkt_residual:.3g}; max next-state outside mass="
                f"{self.diagnostics.next_state_outside_mass.max():.3g}")

    def to_frame(self) -> pd.DataFrame:
        return self.diagnostics.copy()


def _expected(problem: EquilibriumProblem, state: Array, control: Array,
              policy: Policy, counters: dict[str, int] | None = None) -> tuple[Array, float, bool]:
    raw, outside_mass, future_feasible, _, _ = _expected_details(
        problem, state, control, policy, counters=counters
    )
    return raw, outside_mass, future_feasible


def _expected_details(
    problem: EquilibriumProblem,
    state: Array,
    control: Array,
    policy: Policy,
    *,
    counters: dict[str, int] | None = None,
) -> tuple[Array, float, bool, Array, Array]:
    nodes, weights = problem.integration(state)
    q_count = len(weights)
    if problem.batched:
        state_batch = np.broadcast_to(state, (q_count, problem.n_states))
        control_batch = np.broadcast_to(control, (q_count, problem.n_controls))
        if counters is not None:
            counters["transition_evaluations"] = counters.get("transition_evaluations", 0) + 1
        next_states = np.asarray(problem.transition(state_batch, control_batch, nodes), dtype=float)
        if next_states.shape != (q_count, problem.n_states) or not np.all(np.isfinite(next_states)):
            raise ValueError(f"batched transition must return finite array of shape {(q_count, problem.n_states)}")
        inside = problem.contains(next_states)
        outside_mass = float(np.dot(weights, (~inside).astype(float)))
        next_controls = np.asarray(policy(next_states), dtype=float).reshape(q_count, problem.n_controls)
        if not np.all(np.isfinite(next_controls)):
            raise ValueError("future policy returned non-finite controls")
        future_feasible = True
        for q in range(q_count):
            lo, hi = problem.bounds(next_states[q])
            future_feasible &= bool(
                np.all(next_controls[q] >= lo - 1e-10) and np.all(next_controls[q] <= hi + 1e-10)
            )
            if problem.admissible is not None:
                future_feasible &= bool(problem.admissible(next_states[q], next_controls[q]))
        if counters is not None:
            counters["residual_evaluations"] = counters.get("residual_evaluations", 0) + 1
        residuals = np.asarray(
            problem.residual(state_batch, control_batch, next_states, next_controls, nodes),
            dtype=float,
        )
        if residuals.shape != (q_count, problem.n_controls) or not np.all(np.isfinite(residuals)):
            raise ValueError(f"batched residual must return finite array of shape {(q_count, problem.n_controls)}")
        values = np.asarray(weights @ residuals, dtype=float)
        return values, outside_mass, future_feasible, next_states, next_controls

    values = np.zeros(problem.n_controls)
    outside_mass = 0.0
    future_feasible = True
    next_states = np.zeros((q_count, problem.n_states), dtype=float)
    next_controls = np.zeros((q_count, problem.n_controls), dtype=float)
    for q, (innovation, weight) in enumerate(zip(nodes, weights)):
        if counters is not None:
            counters["transition_evaluations"] = counters.get("transition_evaluations", 0) + 1
        next_state = _vector(problem.transition(state, control, innovation), problem.n_states, "transition")
        next_states[q] = next_state
        outside_mass += float(weight) * float(not problem.contains(next_state)[0])
        next_control = _vector(policy(next_state), problem.n_controls, "future policy")
        next_controls[q] = next_control
        lo, hi = problem.bounds(next_state)
        future_feasible &= bool(np.all(next_control >= lo - 1e-10) and np.all(next_control <= hi + 1e-10))
        if problem.admissible is not None:
            future_feasible &= bool(problem.admissible(next_state, next_control))
        if counters is not None:
            counters["residual_evaluations"] = counters.get("residual_evaluations", 0) + 1
        values += weight * _vector(
            problem.residual(state, control, next_state, next_control, innovation),
            problem.n_controls,
            "equilibrium residual",
        )
    return values, outside_mass, future_feasible, next_states, next_controls


def _kkt(problem: EquilibriumProblem, state: Array, control: Array, residual: Array) -> Array:
    out = residual / problem.residual_scale
    lo, hi = problem.bounds(state)
    for equation, index in problem.complementarity.items():
        # Natural-map complementarity residual; lower bound requires f >= 0.
        out[equation] = control[index] - np.clip(control[index] - out[equation], lo[index], hi[index])
    return out


def _kkt_jacobian_from_expected(
    problem: EquilibriumProblem,
    state: Array,
    control: Array,
    raw_residual: Array,
    expected_jacobian: Array,
) -> Array:
    scaled_residual = raw_residual / problem.residual_scale
    jac = expected_jacobian / problem.residual_scale[:, None]
    lo, hi = problem.bounds(state)
    for equation, index in problem.complementarity.items():
        unconstrained = control[index] - scaled_residual[equation]
        if unconstrained <= lo[index] or unconstrained >= hi[index]:
            jac[equation, :] = 0.0
            jac[equation, index] = 1.0
    return jac


def _policy_state_jacobian(policy: Any, next_states: Array, n_states: int, n_controls: int) -> Array:
    if hasattr(policy, "jacobian") and callable(policy.jacobian):
        jac = np.asarray(policy.jacobian(next_states), dtype=float)
        return jac.reshape(len(next_states), n_controls, n_states)
    out = np.zeros((len(next_states), n_controls, n_states), dtype=float)
    for q, sp in enumerate(next_states):
        for dim in range(n_states):
            step = 1e-7 * max(abs(float(sp[dim])), 1.0)
            plus = sp.copy()
            minus = sp.copy()
            plus[dim] += step
            minus[dim] -= step
            out[q, :, dim] = (
                _vector(policy(plus), n_controls, "future policy")
                - _vector(policy(minus), n_controls, "future policy")
            ) / (2.0 * step)
    return out


def _transition_control_jacobian(
    problem: EquilibriumProblem,
    state: Array,
    control: Array,
    nodes: Array,
    next_states: Array,
    *,
    counters: dict[str, int] | None = None,
) -> Array:
    q_count = len(nodes)
    n_s = problem.n_states
    n_x = problem.n_controls
    if problem.transition_jacobian is not None:
        if problem.batched:
            state_batch = np.broadcast_to(state, (q_count, n_s))
            control_batch = np.broadcast_to(control, (q_count, n_x))
            if counters is not None:
                counters["transition_jacobian_evaluations"] = counters.get("transition_jacobian_evaluations", 0) + 1
            jac = np.asarray(problem.transition_jacobian(state_batch, control_batch, nodes), dtype=float)
            if jac.shape != (q_count, n_s, n_x) or not np.all(np.isfinite(jac)):
                raise ValueError(f"batched transition_jacobian must return finite array of shape {(q_count, n_s, n_x)}")
            return jac
        jac = np.zeros((q_count, n_s, n_x), dtype=float)
        for q, innovation in enumerate(nodes):
            if counters is not None:
                counters["transition_jacobian_evaluations"] = counters.get("transition_jacobian_evaluations", 0) + 1
            block = np.asarray(problem.transition_jacobian(state, control, innovation), dtype=float)
            if block.shape != (n_s, n_x) or not np.all(np.isfinite(block)):
                raise ValueError(f"transition_jacobian must return finite array of shape {(n_s, n_x)}")
            jac[q] = block
        return jac

    lo, hi = problem.bounds(state)
    jac = np.zeros((q_count, n_s, n_x), dtype=float)
    for j in range(n_x):
        base_step = 1e-7 * max(abs(float(control[j])), 1.0)
        directions = [1.0, -1.0] if control[j] + base_step <= hi[j] else [-1.0, 1.0]
        computed = False
        for sign in directions:
            step = sign * base_step
            for _ in range(8):
                probe = control.copy()
                probe[j] += step
                if probe[j] < lo[j] - 1e-12 or probe[j] > hi[j] + 1e-12:
                    step *= 0.5
                    continue
                if problem.admissible is not None and not bool(problem.admissible(state, probe)):
                    step *= 0.5
                    continue
                try:
                    if problem.batched:
                        state_batch = np.broadcast_to(state, (q_count, n_s))
                        probe_batch = np.broadcast_to(probe, (q_count, n_x))
                        if counters is not None:
                            counters["transition_evaluations"] = counters.get("transition_evaluations", 0) + 1
                        perturbed = np.asarray(problem.transition(state_batch, probe_batch, nodes), dtype=float)
                        if perturbed.shape != (q_count, n_s) or not np.all(np.isfinite(perturbed)):
                            raise ValueError("nonfinite transition probe")
                    else:
                        perturbed = np.zeros((q_count, n_s), dtype=float)
                        for q, innovation in enumerate(nodes):
                            if counters is not None:
                                counters["transition_evaluations"] = counters.get("transition_evaluations", 0) + 1
                            perturbed[q] = _vector(problem.transition(state, probe, innovation), n_s, "transition")
                    jac[:, :, j] = (perturbed - next_states) / step
                    computed = True
                    break
                except (ValueError, ArithmeticError, FloatingPointError, np.linalg.LinAlgError):
                    step *= 0.5
            if computed:
                break
        if not computed:
            raise ValueError("unable to finite-difference transition inside admissible domain")
    return jac


def _expected_jacobian_analytic_or_hybrid(
    problem: EquilibriumProblem,
    state: Array,
    control: Array,
    policy: Policy,
    next_states: Array,
    next_controls: Array,
    *,
    counters: dict[str, int] | None = None,
) -> Array:
    nodes, weights = problem.integration(state)
    q_count = len(weights)
    n_s = problem.n_states
    n_x = problem.n_controls
    g_x = _transition_control_jacobian(problem, state, control, nodes, next_states, counters=counters)
    pi_sp = _policy_state_jacobian(policy, next_states, n_s, n_x)
    assert problem.residual_jacobian is not None
    if problem.batched:
        state_batch = np.broadcast_to(state, (q_count, n_s))
        control_batch = np.broadcast_to(control, (q_count, n_x))
        if counters is not None:
            counters["residual_jacobian_evaluations"] = counters.get("residual_jacobian_evaluations", 0) + 1
        r_x, r_sp, r_xp = problem.residual_jacobian(
            state_batch, control_batch, next_states, next_controls, nodes
        )
        r_x_arr = np.asarray(r_x, dtype=float)
        r_sp_arr = np.asarray(r_sp, dtype=float)
        r_xp_arr = np.asarray(r_xp, dtype=float)
        if (
            r_x_arr.shape != (q_count, n_x, n_x)
            or r_sp_arr.shape != (q_count, n_x, n_s)
            or r_xp_arr.shape != (q_count, n_x, n_x)
            or not (np.all(np.isfinite(r_x_arr)) and np.all(np.isfinite(r_sp_arr)) and np.all(np.isfinite(r_xp_arr)))
        ):
            raise ValueError("batched residual_jacobian returned invalid or nonfinite blocks")
        total_q = r_x_arr + (r_sp_arr + r_xp_arr @ pi_sp) @ g_x
        return np.asarray(np.tensordot(weights, total_q, axes=(0, 0)), dtype=float)

    expected_jac = np.zeros((n_x, n_x), dtype=float)
    for q, (innovation, weight) in enumerate(zip(nodes, weights)):
        if counters is not None:
            counters["residual_jacobian_evaluations"] = counters.get("residual_jacobian_evaluations", 0) + 1
        r_x, r_sp, r_xp = problem.residual_jacobian(
            state, control, next_states[q], next_controls[q], innovation
        )
        r_x_arr = np.asarray(r_x, dtype=float)
        r_sp_arr = np.asarray(r_sp, dtype=float)
        r_xp_arr = np.asarray(r_xp, dtype=float)
        if (
            r_x_arr.shape != (n_x, n_x)
            or r_sp_arr.shape != (n_x, n_s)
            or r_xp_arr.shape != (n_x, n_x)
            or not (np.all(np.isfinite(r_x_arr)) and np.all(np.isfinite(r_sp_arr)) and np.all(np.isfinite(r_xp_arr)))
        ):
            raise ValueError("residual_jacobian returned invalid or nonfinite blocks")
        expected_jac += float(weight) * (r_x_arr + (r_sp_arr + r_xp_arr @ pi_sp[q]) @ g_x[q])
    return expected_jac


def audit_policy(problem: EquilibriumProblem, policy: Policy, points: Any, *,
                 tol: float = 1e-6, require_domain: bool = True) -> PolicyAudit:
    """Evaluate supplied policy independently of its solver or convergence flag.

    Supply off-grid points and, for quadrature sensitivity, a problem with a
    refined innovation rule. No policy clipping or replacement solve occurs.
    Numerical/domain failures are recorded per point and fail the audit.
    """
    tol = _positive(tol, "tol")
    states, _ = _points(points, problem.n_states)
    if len(states) == 0:
        raise ValueError("audit points must be nonempty")
    raw = np.full((len(states), problem.n_controls), np.inf)
    kkt = raw.copy()
    records = []
    for i, state in enumerate(states):
        failure = ""
        outside_mass = np.inf
        feasible = future_feasible = False
        try:
            control = _vector(policy(state), problem.n_controls, "policy")
            lo, hi = problem.bounds(state)
            feasible = bool(np.all(control >= lo - tol) and np.all(control <= hi + tol))
            if problem.admissible is not None:
                feasible &= bool(problem.admissible(state, control))
            raw[i], outside_mass, future_feasible = _expected(problem, state, control, policy)
            kkt[i] = _kkt(problem, state, control, raw[i])
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
            failure = f"{type(exc).__name__}: {exc}"
        state_inside = bool(problem.contains(state)[0])
        accepted = bool(np.max(np.abs(kkt[i])) <= tol and feasible and future_feasible
                        and (not require_domain or (state_inside and outside_mass == 0)))
        records.append(dict(max_scaled_residual=float(np.max(np.abs(raw[i] / problem.residual_scale))),
                            max_kkt_residual=float(np.max(np.abs(kkt[i]))),
                            state_inside=state_inside, next_state_outside_mass=outside_mass,
                            feasible=feasible, future_feasible=future_feasible,
                            passed=accepted, failure=failure))
    diagnostics = pd.DataFrame(records)
    return PolicyAudit(pd.DataFrame(states, columns=problem.state_names),
                       pd.DataFrame(raw, columns=problem.equation_names),
                       pd.DataFrame(kkt, columns=problem.equation_names),
                       diagnostics, tol, bool(diagnostics.passed.all()), require_domain)


def compare_policies(problem: EquilibriumProblem, policies: Mapping[str, Policy],
                     points: Any, *, tol: float = 1e-6) -> pd.DataFrame:
    """Compare local/global policies using identical points and expectations."""
    records = []
    for name, policy in policies.items():
        audit = audit_policy(problem, policy, points, tol=tol)
        records.append(dict(method=name, passed=audit.passed,
                            max_residual=audit.max_residual,
                            max_kkt_residual=audit.max_kkt_residual,
                            next_state_outside_mass=float(audit.diagnostics.next_state_outside_mass.max()),
                            infeasible_points=int((~audit.diagnostics.feasible).sum())))
    return pd.DataFrame(records).set_index("method") if records else pd.DataFrame()


class _Basis:
    def __init__(self, problem: EquilibriumProblem, orders: Any, basis: str,
                 max_nodes: int) -> None:
        self.domain = np.asarray(problem.domain)
        self.dimension = problem.n_states
        self.kind = basis
        self.orders: tuple[int, ...]
        if basis not in ("chebyshev", "smolyak", "linear"):
            raise ValueError("basis must be 'chebyshev', 'smolyak' or 'linear'")
        if basis == "smolyak":
            level = _integer(orders, "Smolyak level")
            # Count disjoint Clenshaw-Curtis increments before allocating the
            # sparse grid or its dense collocation matrix.
            if level > 8 or self.dimension > 12:
                raise ValueError("Smolyak dimension/level exceeds supported construction budget")
            counts = [1] + [0] * level
            increments = [1, 2] + [2 ** (j - 1) for j in range(2, level + 1)]
            for _ in range(self.dimension):
                counts = [sum(counts[k] * increments[j - k] for k in range(j + 1)) for j in range(level + 1)]
            if sum(counts) > max_nodes:
                raise ValueError("Smolyak state grid exceeds max_nodes")
            self.native: Any = SmolyakBasis(d=self.dimension, mu=level, domain=problem.domain)
            self.nodes = self.native.grid.physical_nodes.copy()
            self.orders = (level,)
        else:
            normalized = (orders,) * self.dimension if isinstance(orders, (int, np.integer)) else tuple(orders)
            if len(normalized) != self.dimension:
                raise ValueError("one polynomial/grid order is required per state")
            self.orders = tuple(_integer(o, "order") for o in normalized)
            count = 1
            for order in self.orders:
                count *= order + 1
            if count > max_nodes:
                raise ValueError("state grid exceeds max_nodes")
            self.native = CollocationBasis(tuple(problem.domain), self.orders)
            self.nodes = self.native.nodes()
        if len(self.nodes) > max_nodes:
            raise ValueError("state grid exceeds max_nodes")
        self.axes = tuple(np.unique(self.nodes[:, i]) for i in range(self.dimension))
        if basis == "chebyshev":
            factors_1d = []
            for d in range(self.dimension):
                axis = self.axes[d]
                canonical_1d = (2.0 * axis - self.domain[d, 0] - self.domain[d, 1]) / (
                    self.domain[d, 1] - self.domain[d, 0]
                )
                vander_1d = np.polynomial.chebyshev.chebvander(canonical_1d, self.orders[d])
                factors_1d.append(lu_factor(vander_1d))
            self.factors_1d: tuple[tuple[Array, Array], ...] | None = tuple(factors_1d)
            self.factor = None
            self.collocation_matrix_allocated = False
            self.storage_bytes = sum(int(lu.nbytes + piv.nbytes) for lu, piv in self.factors_1d)
        elif basis == "smolyak":
            self.factors_1d = None
            self.factor = lu_factor(self.matrix(self.nodes))
            self.collocation_matrix_allocated = True
            self.storage_bytes = int(self.factor[0].nbytes + self.factor[1].nbytes)
        else:
            self.factors_1d = None
            self.factor = None
            self.collocation_matrix_allocated = False
            self.storage_bytes = int(self.nodes.nbytes)

    def matrix(self, points: Array) -> Array:
        # Reuse the public basis for in-domain evaluations. Its kernels clip
        # canonical coordinates; explicitly evaluate polynomial extrapolation
        # outside to avoid hiding domain exits during Newton trial steps.
        canonical = (2 * points - self.domain[:, 0] - self.domain[:, 1]) / (self.domain[:, 1] - self.domain[:, 0])
        if np.all(np.abs(canonical) <= 1 + 1e-14):
            return self.native.evaluate(points)
        if self.kind == "smolyak":
            indices = self.native.polys
            result = np.ones((len(points), len(indices)))
            for i in range(self.dimension):
                univariate = np.polynomial.chebyshev.chebvander(canonical[:, i], int(indices[:, i].max()))
                result *= univariate[:, indices[:, i]]
            return result
        result = np.polynomial.chebyshev.chebvander(canonical[:, 0], self.orders[0])
        for i in range(1, self.dimension):
            next_basis = np.polynomial.chebyshev.chebvander(canonical[:, i], self.orders[i])
            result = (result[:, :, None] * next_basis[:, None, :]).reshape(len(points), -1)
        return result

    def derivative_matrix(self, points: Array, dim: int) -> Array:
        canonical = (2 * points - self.domain[:, 0] - self.domain[:, 1]) / (self.domain[:, 1] - self.domain[:, 0])
        scale = 2.0 / float(self.domain[dim, 1] - self.domain[dim, 0])
        if self.kind == "smolyak":
            indices = self.native.polys
            result = np.ones((len(points), len(indices)))
            for i in range(self.dimension):
                deg = int(indices[:, i].max())
                if i == dim:
                    if deg == 0:
                        univariate = np.zeros((len(points), 1), dtype=float)
                    else:
                        d_coeff = np.polynomial.chebyshev.chebder(np.eye(deg + 1), axis=0)
                        univariate = (
                            np.polynomial.chebyshev.chebvander(canonical[:, i], deg - 1) @ d_coeff
                        ) * scale
                else:
                    univariate = np.polynomial.chebyshev.chebvander(canonical[:, i], deg)
                result *= univariate[:, indices[:, i]]
            return result

        def _factor_1d(i: int) -> Array:
            deg = self.orders[i]
            if i == dim:
                if deg == 0:
                    return np.zeros((len(points), 1), dtype=float)
                d_coeff = np.polynomial.chebyshev.chebder(np.eye(deg + 1), axis=0)
                return (np.polynomial.chebyshev.chebvander(canonical[:, i], deg - 1) @ d_coeff) * scale
            return np.polynomial.chebyshev.chebvander(canonical[:, i], deg)

        result = _factor_1d(0)
        for i in range(1, self.dimension):
            next_basis = _factor_1d(i)
            result = (result[:, :, None] * next_basis[:, None, :]).reshape(len(points), -1)
        return result

    def fit(self, values: Array, *, extrapolation: str) -> "EquilibriumPolicy":
        if self.factors_1d is not None:
            grid_shape = tuple(order + 1 for order in self.orders)
            n_controls = values.shape[1]
            arr = values.reshape(grid_shape + (n_controls,))
            for d, factor_1d in enumerate(self.factors_1d):
                moved = np.moveaxis(arr, d, 0)
                solved = lu_solve(factor_1d, moved.reshape(grid_shape[d], -1)).reshape(moved.shape)
                arr = np.moveaxis(solved, 0, d)
            coefficients: Array | None = arr.reshape(-1, n_controls)
        elif self.factor is not None:
            coefficients = lu_solve(self.factor, values)
        else:
            coefficients = None
        return EquilibriumPolicy(self, values, coefficients, extrapolation)


@dataclass
class EquilibriumPolicy:
    """Interpolated controls; extrapolation is explicit and never clipped."""

    _basis: _Basis
    values: Array
    coefficients: Array | None
    extrapolation: str = "raise"

    def __post_init__(self) -> None:
        self.values = np.array(self.values, copy=True)
        self.values.setflags(write=False)
        if self.coefficients is not None:
            self.coefficients = np.array(self.coefficients, copy=True)
            self.coefficients.setflags(write=False)
        if self.extrapolation not in ("raise", "allow"):
            raise ValueError("extrapolation must be 'raise' or 'allow'")
        self._interpolator = None
        if self._basis.kind == "linear":
            shape = tuple(len(axis) for axis in self._basis.axes) + (self.values.shape[1],)
            self._interpolator = RegularGridInterpolator(self._basis.axes, self.values.reshape(shape), bounds_error=False, fill_value=None)

    @property
    def collocation_matrix_allocated(self) -> bool:
        return self._basis.collocation_matrix_allocated

    @property
    def storage_bytes(self) -> int:
        return self._basis.storage_bytes

    def __call__(self, states: Any) -> Array:
        points, single = _points(states, self._basis.dimension)
        domain = self._basis.domain
        if self.extrapolation == "raise" and np.any((points < domain[:, 0] - 1e-12) | (points > domain[:, 1] + 1e-12)):
            raise ValueError("policy evaluation is outside its declared state domain")
        if self._interpolator is not None:
            output = self._interpolator(points)
        else:
            output = self._basis.matrix(points) @ self.coefficients
        return output[0] if single else output

    def jacobian(self, states: Any) -> Array:
        """Evaluate state Jacobian d pi(s) / d s of shape ([m,] n_controls, n_states)."""
        points, single = _points(states, self._basis.dimension)
        domain = self._basis.domain
        if self.extrapolation == "raise" and np.any((points < domain[:, 0] - 1e-12) | (points > domain[:, 1] + 1e-12)):
            raise ValueError("policy evaluation is outside its declared state domain")
        n_points = len(points)
        n_controls = self.values.shape[1]
        n_states = self._basis.dimension
        jac = np.zeros((n_points, n_controls, n_states), dtype=float)
        if self._interpolator is not None:
            for dim in range(n_states):
                step = max(1e-7 * float(domain[dim, 1] - domain[dim, 0]), 1e-8)
                plus = points.copy()
                minus = points.copy()
                plus[:, dim] += step
                minus[:, dim] -= step
                if self.extrapolation == "raise":
                    plus[:, dim] = np.minimum(plus[:, dim], domain[dim, 1])
                    minus[:, dim] = np.maximum(minus[:, dim], domain[dim, 0])
                denom = np.maximum((plus[:, dim] - minus[:, dim])[:, None], 1e-15)
                jac[:, :, dim] = (self._interpolator(plus) - self._interpolator(minus)) / denom
        else:
            assert self.coefficients is not None
            for dim in range(n_states):
                jac[:, :, dim] = self._basis.derivative_matrix(points, dim) @ self.coefficients
        return jac[0] if single else jac


@dataclass(frozen=True)
class TimeIterationResult(_PolicyReport):
    policy: EquilibriumPolicy
    converged: bool
    iterations: int
    history: pd.DataFrame
    audit: PolicyAudit
    state_names: tuple[str, ...]
    control_names: tuple[str, ...]
    message: str
    require_domain: bool = True
    callback_evaluations: Mapping[str, int] = field(default_factory=dict)

    def __call__(self, states: Any) -> Array:
        return self.policy(states)

    def jacobian(self, states: Any) -> Array:
        return self.policy.jacobian(states)

    def to_frame(self) -> pd.DataFrame:
        states = pd.DataFrame(self.policy._basis.nodes, columns=self.state_names)
        return pd.concat([states, pd.DataFrame(self.policy.values, columns=self.control_names)], axis=1)

    def summary(self) -> str:
        return f"Equilibrium time iteration: converged={self.converged}; iterations={self.iterations}; {self.message}. {self.audit.summary()}"

    def plot(self, points: Any, *, state: str | int = 0,
             controls: Sequence[str] | None = None, ax: Any = None,
             show: bool = False) -> Any:
        """Plot controls at explicit full state points, in supplied point order.

        ``state`` selects the horizontal coordinate. The caller supplies all
        other state coordinates, for example a fixed-productivity capital
        slice; none are inferred. Return the supplied axis, or a new figure.
        """
        states, _ = _points(points, len(self.state_names))
        if isinstance(state, str):
            coordinate = self.state_names.index(state)
        elif isinstance(state, (int, np.integer)) and not isinstance(state, (bool, np.bool_)):
            coordinate = int(state)
        else:
            raise ValueError("state must be a state name or integer index")
        if not 0 <= coordinate < len(self.state_names):
            raise ValueError("state index is out of range")
        selected = self.control_names if controls is None else (controls,) if isinstance(controls, str) else tuple(controls)
        if not selected or any(name not in self.control_names for name in selected):
            raise ValueError("controls must select at least one known control")
        values = self(states)
        import matplotlib.pyplot as plt

        supplied_axis = ax is not None
        if ax is None:
            _, ax = plt.subplots()
        for name in selected:
            ax.plot(states[:, coordinate], values[:, self.control_names.index(name)], label=name)
        ax.set_xlabel(self.state_names[coordinate])
        ax.set_ylabel("Control level")
        ax.legend()
        if show:
            plt.show()
        return ax if supplied_axis else ax.figure


def _solve_node(
    problem: EquilibriumProblem,
    state: Array,
    initial_control: Array,
    previous: Policy,
    *,
    tol: float,
    inner_max_iter: int,
    require_domain: bool,
    counters: dict[str, int],
    initial_trial: tuple[Array, Array, float, bool, Array, Array] | None = None,
) -> tuple[Array, bool, int, str]:
    lo, hi = problem.bounds(state)
    eval_count = 0
    last_rejection = "infeasible trial control"

    def _evaluate_trial(
        ctrl: Array,
        *,
        require_future_feasible: bool = False,
        count_rejection: bool = True,
    ) -> tuple[Array, Array, float, bool, Array, Array] | None:
        nonlocal eval_count, last_rejection
        if not np.all(np.isfinite(ctrl)):
            last_rejection = "non-finite trial control"
            if count_rejection:
                counters["trial_rejections"] = counters.get("trial_rejections", 0) + 1
            return None
        if np.any(ctrl < lo - 1e-12) or np.any(ctrl > hi + 1e-12):
            last_rejection = "trial control violates control_bounds"
            if count_rejection:
                counters["trial_rejections"] = counters.get("trial_rejections", 0) + 1
            return None
        if problem.admissible is not None:
            try:
                is_admissible = bool(problem.admissible(state, ctrl))
            except (ValueError, ArithmeticError, FloatingPointError, np.linalg.LinAlgError) as exc:
                last_rejection = f"admissible check raised {type(exc).__name__}: {exc}"
                if count_rejection:
                    counters["trial_rejections"] = counters.get("trial_rejections", 0) + 1
                return None
            if not is_admissible:
                last_rejection = "trial control violates open admissible domain"
                if count_rejection:
                    counters["trial_rejections"] = counters.get("trial_rejections", 0) + 1
                return None
        try:
            eval_count += 1
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                raw, outside_mass, future_feasible, next_states, next_controls = _expected_details(
                    problem, state, ctrl, previous, counters=counters
                )
            if not np.all(np.isfinite(raw)):
                last_rejection = "non-finite expected residual"
                if count_rejection:
                    counters["trial_rejections"] = counters.get("trial_rejections", 0) + 1
                return None
            if require_future_feasible and not future_feasible:
                last_rejection = "next-state control violates admissible or bound domain"
                if count_rejection:
                    counters["trial_rejections"] = counters.get("trial_rejections", 0) + 1
                return None
            kkt = _kkt(problem, state, ctrl, raw)
            if not np.all(np.isfinite(kkt)):
                last_rejection = "non-finite KKT residual"
                if count_rejection:
                    counters["trial_rejections"] = counters.get("trial_rejections", 0) + 1
                return None
            return raw, kkt, outside_mass, future_feasible, next_states, next_controls
        except (ValueError, ArithmeticError, FloatingPointError, np.linalg.LinAlgError) as exc:
            last_rejection = f"{type(exc).__name__}: {exc}"
            if count_rejection:
                counters["trial_rejections"] = counters.get("trial_rejections", 0) + 1
            return None

    x = np.clip(np.array(initial_control, dtype=float, copy=True), lo, hi)
    trial = (
        initial_trial
        if initial_trial is not None and float(np.max(np.abs(x - initial_control))) <= 1e-13
        else _evaluate_trial(x, require_future_feasible=False, count_rejection=False)
    )
    if trial is None:
        return x, False, eval_count, f"feasible backtracking exhausted: {last_rejection}"
    require_future = bool(trial[3])
    if float(np.max(np.abs(trial[1]))) <= tol:
        return x, True, eval_count, ""

    n_x = problem.n_controls
    for _ in range(inner_max_iter):
        raw, kkt, _, _, next_states, next_controls = trial
        if float(np.max(np.abs(kkt))) <= tol:
            return x, True, eval_count, ""

        try:
            if problem.residual_jacobian is not None:
                expected_jac = _expected_jacobian_analytic_or_hybrid(
                    problem, state, x, previous, next_states, next_controls, counters=counters
                )
                jac = _kkt_jacobian_from_expected(problem, state, x, raw, expected_jac)
            else:
                jac = np.zeros((n_x, n_x), dtype=float)
                for j in range(n_x):
                    base_step = 1e-7 * max(abs(float(x[j])), 1.0)
                    directions = [1.0, -1.0] if x[j] + base_step <= hi[j] else [-1.0, 1.0]
                    col_computed = False
                    for sign in directions:
                        step = sign * base_step
                        for _ in range(10):
                            probe = x.copy()
                            probe[j] += step
                            probe_trial = _evaluate_trial(
                                probe, require_future_feasible=require_future, count_rejection=True
                            )
                            if probe_trial is not None:
                                jac[:, j] = (probe_trial[1] - kkt) / step
                                col_computed = True
                                break
                            step *= 0.5
                        if col_computed:
                            break
                    if not col_computed:
                        return x, False, eval_count, f"feasible backtracking exhausted: {last_rejection}"
        except (ValueError, ArithmeticError, FloatingPointError, np.linalg.LinAlgError) as exc:
            return x, False, eval_count, f"feasible backtracking exhausted: {type(exc).__name__}: {exc}"

        try:
            dx = np.linalg.solve(jac, -kkt)
        except np.linalg.LinAlgError:
            dx = np.linalg.lstsq(jac, -kkt, rcond=None)[0]

        if not np.all(np.isfinite(dx)):
            dx = -kkt.copy()

        # Respect closed box bounds without interior epsilon.
        clamped = ((x + dx < lo) & (dx < 0.0)) | ((x + dx > hi) & (dx > 0.0))
        if (
            np.any(clamped)
            and np.any(~clamped)
            and all(eq == ctrl for eq, ctrl in problem.complementarity.items())
        ):
            free = ~clamped
            dx_clamped = np.clip(x + dx, lo, hi) - x
            rhs_free = -(kkt[free] + jac[np.ix_(free, clamped)] @ dx_clamped[clamped])
            dx[clamped] = dx_clamped[clamped]
            try:
                dx[free] = np.linalg.solve(jac[np.ix_(free, free)], rhs_free)
            except np.linalg.LinAlgError:
                dx[free] = np.linalg.lstsq(jac[np.ix_(free, free)], rhs_free, rcond=None)[0]

        current_merit = 0.5 * float(np.dot(kkt, kkt))
        current_norm = float(np.max(np.abs(kkt)))
        accepted_step = False
        any_feasible = False
        best_feasible: tuple[Array, tuple[Array, Array, float, bool, Array, Array], float] | None = None

        for direction in (dx, -(jac.T @ kkt)):
            dir_norm = float(np.linalg.norm(direction))
            if dir_norm <= 1e-15 or not np.isfinite(dir_norm):
                continue
            alpha = 1.0
            for _ in range(16):
                x_try = np.clip(x + alpha * direction, lo, hi)
                if float(np.max(np.abs(x_try - x))) <= 1e-14:
                    break
                cand = _evaluate_trial(
                    x_try, require_future_feasible=require_future, count_rejection=True
                )
                if cand is None:
                    alpha *= 0.5
                    continue
                any_feasible = True
                cand_kkt = cand[1]
                cand_merit = 0.5 * float(np.dot(cand_kkt, cand_kkt))
                cand_norm = float(np.max(np.abs(cand_kkt)))
                if best_feasible is None or cand_merit < best_feasible[2]:
                    best_feasible = (x_try, cand, cand_merit)
                if (
                    cand_merit <= (1.0 - 1e-4 * alpha) * current_merit
                    or cand_norm < current_norm
                    or cand_norm <= tol
                ):
                    x = x_try
                    trial = cand
                    accepted_step = True
                    break
                alpha *= 0.5
            if accepted_step:
                break

        if not accepted_step:
            if not any_feasible:
                return x, False, eval_count, f"feasible backtracking exhausted: {last_rejection}"
            if best_feasible is not None and best_feasible[2] < current_merit:
                x, trial, _ = best_feasible
            else:
                break

        if float(np.max(np.abs(trial[1]))) <= tol:
            return x, True, eval_count, ""

    converged_node = bool(float(np.max(np.abs(trial[1]))) <= tol)
    return x, converged_node, eval_count, ""


def solve_time_iteration(problem: EquilibriumProblem, initial_policy: Policy | Any,
                         *, orders: Any = 7, basis: str = "chebyshev", tol: float = 1e-7,
                         max_iter: int = 200, inner_max_iter: int = 100,
                         damping: float = 1.0, max_nodes: int = 10000,
                         extrapolation: str = "raise", require_domain: bool = True,
                         strict: bool = False) -> TimeIterationResult:
    """Iterate expected equilibrium conditions with the future policy held fixed.

    Independent node residuals, feasibility and domain closure are checked
    after updating the policy; small policy changes alone never imply success.
    Trial steps use explicit extrapolation, but domain exits prevent accepted
    convergence unless ``require_domain=False`` is explicitly selected. In
    that mode convergence is conditional on the chosen extrapolation; the
    node exit mass remains reported and independent interior audits are needed.
    Returned policies raise on extrapolation by default. For
    constraints with kinks, ``basis='linear'`` avoids polynomial overshoot.
    """
    tol = _positive(tol, "tol")
    max_iter = _integer(max_iter, "max_iter")
    inner_max_iter = _integer(inner_max_iter, "inner_max_iter")
    max_nodes = _integer(max_nodes, "max_nodes")
    if not np.isfinite(damping) or not 0 < damping <= 1:
        raise ValueError("damping must be in (0, 1]")
    if extrapolation not in ("raise", "allow"):
        raise ValueError("extrapolation must be 'raise' or 'allow'")
    approximation = _Basis(problem, orders, basis, max_nodes)
    values = np.array([initial_policy(s) for s in approximation.nodes] if callable(initial_policy) else initial_policy, dtype=float)
    if values.shape == (problem.n_controls,):
        values = np.tile(values, (len(approximation.nodes), 1))
    if values.shape != (len(approximation.nodes), problem.n_controls) or not np.isfinite(values).all():
        raise ValueError("initial_policy must supply one finite control vector per grid point")
    for state, control in zip(approximation.nodes, values):
        lo, hi = problem.bounds(state)
        if np.any(control < lo) or np.any(control > hi):
            raise ValueError("initial policy violates control bounds")
        if problem.admissible is not None and not problem.admissible(state, control):
            raise ValueError("initial policy is economically inadmissible")
    history: list[dict[str, Any]] = []
    converged = False
    message = "iteration limit"
    counters: dict[str, int] = {
        "transition_evaluations": 0,
        "residual_evaluations": 0,
        "transition_jacobian_evaluations": 0,
        "residual_jacobian_evaluations": 0,
        "trial_rejections": 0,
    }
    previous = approximation.fit(values, extrapolation="allow")
    audit: PolicyAudit | None = None
    node_trials: list[tuple[Array, Array, float, bool, Array, Array] | None] = [None] * len(approximation.nodes)
    for iteration in range(1, max_iter + 1):
        candidate = values.copy()
        failed_nodes = 0
        evaluations = 0
        failure = ""
        iter_rejections_before = counters["trial_rejections"]
        for i, state in enumerate(approximation.nodes):
            solved_x, node_ok, node_evals, node_failure = _solve_node(
                problem,
                state,
                values[i],
                previous,
                tol=tol,
                inner_max_iter=inner_max_iter,
                require_domain=require_domain,
                counters=counters,
                initial_trial=node_trials[i],
            )
            candidate[i] = solved_x
            evaluations += node_evals
            if not node_ok:
                failed_nodes += 1
                if node_failure and not failure:
                    failure = node_failure

        # Outer policy update with feasible backtracking when admissible domain is active.
        alpha_outer = float(damping)
        updated = values + alpha_outer * (candidate - values)
        if problem.admissible is not None and not failure:
            outer_ok = False
            for _ in range(12):
                updated = values + alpha_outer * (candidate - values)
                if all(
                    bool(problem.admissible(s, u))
                    for s, u in zip(approximation.nodes, updated)
                ):
                    outer_ok = True
                    break
                counters["trial_rejections"] += 1
                alpha_outer *= 0.5
            if not outer_ok:
                failed_nodes += 1
                failure = "feasible backtracking exhausted: outer policy update violates admissible domain"

        step = float(np.max(np.abs(updated - values)))
        previous = approximation.fit(updated, extrapolation="allow")
        raw_nodes = np.full((len(approximation.nodes), problem.n_controls), np.inf)
        kkt_nodes = raw_nodes.copy()
        records = []
        next_node_trials: list[tuple[Array, Array, float, bool, Array, Array] | None] = [None] * len(approximation.nodes)
        for i, state in enumerate(approximation.nodes):
            audit_fail = ""
            outside_mass = np.inf
            feasible = future_feasible = False
            try:
                control = _vector(previous(state), problem.n_controls, "policy")
                lo, hi = problem.bounds(state)
                feasible = bool(np.all(control >= lo - tol) and np.all(control <= hi + tol))
                if problem.admissible is not None:
                    feasible &= bool(problem.admissible(state, control))
                raw_i, outside_mass, future_feasible, next_states_i, next_controls_i = _expected_details(
                    problem, state, control, previous, counters=counters
                )
                raw_nodes[i] = raw_i
                kkt_nodes[i] = _kkt(problem, state, control, raw_i)
                if feasible and np.all(np.isfinite(kkt_nodes[i])) and float(np.max(np.abs(control - updated[i]))) <= 1e-12:
                    next_node_trials[i] = (
                        raw_i,
                        kkt_nodes[i],
                        outside_mass,
                        future_feasible,
                        next_states_i,
                        next_controls_i,
                    )
            except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
                audit_fail = f"{type(exc).__name__}: {exc}"
            state_inside = bool(problem.contains(state)[0])
            accepted_node = bool(
                np.max(np.abs(kkt_nodes[i])) <= tol
                and feasible
                and future_feasible
                and (not require_domain or (state_inside and outside_mass == 0))
            )
            records.append(
                dict(
                    max_scaled_residual=float(np.max(np.abs(raw_nodes[i] / problem.residual_scale))),
                    max_kkt_residual=float(np.max(np.abs(kkt_nodes[i]))),
                    state_inside=state_inside,
                    next_state_outside_mass=outside_mass,
                    feasible=feasible,
                    future_feasible=future_feasible,
                    passed=accepted_node,
                    failure=audit_fail,
                )
            )
        diagnostics = pd.DataFrame(records)
        audit = PolicyAudit(
            pd.DataFrame(approximation.nodes, columns=problem.state_names),
            pd.DataFrame(raw_nodes, columns=problem.equation_names),
            pd.DataFrame(kkt_nodes, columns=problem.equation_names),
            diagnostics,
            tol,
            bool(diagnostics.passed.all()),
            require_domain,
        )
        node_trials = next_node_trials
        history.append(dict(iteration=iteration, policy_change=step,
                            max_kkt_residual=audit.max_kkt_residual,
                            next_state_outside_mass=float(audit.diagnostics.next_state_outside_mass.max()),
                            inner_failures=failed_nodes, evaluations=evaluations,
                            trial_rejections=counters["trial_rejections"] - iter_rejections_before,
                            failure=failure))
        values = updated
        if audit.passed and step <= tol and failed_nodes == 0:
            converged = True
            message = ("node residuals, feasibility, domain closure and policy change" if require_domain else
                       "node residuals, feasibility and policy change; conditional on extrapolation")
            break
        if failure:
            message = "inner numerical failure: " + failure
            break
    if audit is None:
        audit = audit_policy(problem, previous, approximation.nodes, tol=tol, require_domain=require_domain)
    final_policy = approximation.fit(values, extrapolation=extrapolation)
    result = TimeIterationResult(
        final_policy,
        converged,
        len(history),
        pd.DataFrame(history),
        audit,
        tuple(problem.state_names),
        tuple(problem.control_names),
        message,
        require_domain,
        dict(counters),
    )
    if not converged:
        if strict:
            raise RuntimeError(result.summary())
        warnings.warn(result.summary(), RuntimeWarning, stacklevel=2)
    return result


@dataclass(frozen=True)
class PolicyRefinementResult(_PolicyReport):
    result: TimeIterationResult
    history: pd.DataFrame
    accepted: bool
    audit: PolicyAudit

    def to_frame(self) -> pd.DataFrame:
        """Return an independent copy of the refinement history."""
        return self.history.copy()

    def summary(self) -> str:
        return f"Policy refinement: accepted={self.accepted}; solves={len(self.history)}. {self.audit.summary()}"

    def plot(self, points: Any, **kwargs: Any) -> Any:
        """Plot the final policy at caller-supplied full state points."""
        return self.result.plot(points, **kwargs)


def refine_time_iteration(problem: EquilibriumProblem, initial_policy: Policy | Any,
                          levels: Sequence[Mapping[str, Any]], points: Any, *,
                          tol: float = 1e-5, max_solves: int = 4,
                          **solve_options: Any) -> PolicyRefinementResult:
    """Run a finite user-declared grid/domain/quadrature refinement ladder.

    Each level overrides solver options and may supply ``domain``,
    ``shock_nodes`` and ``shock_weights``. Adjacent policies are compared on
    the same independent points. Passing requires both the policy change and
    the new expected-equation audit; the budget never creates a pass.
    """
    tol = _positive(tol, "tol")
    max_solves = _integer(max_solves, "max_solves")
    if not levels:
        raise ValueError("levels must be nonempty")
    states, _ = _points(points, problem.n_states)
    records = []
    prior: Any = initial_policy
    previous_values = None
    accepted = False
    for number, settings in enumerate(levels[:max_solves], start=1):
        options = dict(settings)
        changes = {name: options.pop(name) for name in ("domain", "shock_nodes", "shock_weights") if name in options}
        current_problem = replace(problem, **changes)
        # Expanded domains can use explicitly declared extrapolation as the
        # initial guess. Acceptance still requires the new domain audit.
        if isinstance(prior, TimeIterationResult):
            initial = replace(prior.policy, extrapolation="allow")
        else:
            initial = prior
        result = solve_time_iteration(current_problem, initial, **(solve_options | options))
        audit = audit_policy(current_problem, result, states, tol=tol)
        current_values = result(states)
        change = np.inf if previous_values is None else float(np.max(np.abs(current_values - previous_values)))
        accepted = bool(result.converged and audit.passed and change <= tol)
        records.append(dict(level=number, nodes=len(result.policy.values), converged=result.converged,
                            max_kkt_residual=audit.max_kkt_residual, policy_change=change,
                            accepted=accepted))
        if accepted:
            break
        previous_values = current_values
        prior = result
    return PolicyRefinementResult(result, pd.DataFrame(records), accepted, audit)


def solve_adaptive_time_iteration(problem: EquilibriumProblem,
                                  initial_policy: Policy | Any, points: Any, *,
                                  orders: Any = 3, tol: float = 1e-5,
                                  max_solves: int = 5, increment: int = 2,
                                  **solve_options: Any) -> PolicyRefinementResult:
    """Refine anisotropic Chebyshev orders using polynomial-tail evidence.

    The dimension with the largest high-degree coefficient tail is refined.
    This is a refinement heuristic, not an error bound: acceptance additionally
    requires an independent expected-equation audit and agreement between
    adjacent policies at the supplied points. ``max_solves`` and the underlying
    ``max_nodes`` cap the work. Domain/quadrature sensitivity can be examined
    separately with :func:`refine_time_iteration`.
    """
    tol = _positive(tol, "tol")
    max_solves = _integer(max_solves, "max_solves")
    increment = _integer(increment, "increment")
    if solve_options.get("basis", "chebyshev") != "chebyshev":
        raise ValueError("automatic anisotropic refinement requires basis='chebyshev'")
    if isinstance(orders, (int, np.integer)):
        chosen = [_integer(orders, "order")] * problem.n_states
    else:
        chosen = [_integer(value, "order") for value in orders]
    if len(chosen) != problem.n_states:
        raise ValueError("one order is required per state")
    states, _ = _points(points, problem.n_states)
    records = []
    previous_values = None
    prior = initial_policy
    accepted = False
    for level in range(1, max_solves + 1):
        result = solve_time_iteration(problem, prior, orders=tuple(chosen), **solve_options)
        audit = audit_policy(problem, result, states, tol=tol)
        values = result(states)
        change = np.inf if previous_values is None else float(np.max(np.abs(values - previous_values)))
        accepted = bool(result.converged and audit.passed and change <= tol)
        assert result.policy.coefficients is not None
        coefficients = result.policy.coefficients.reshape(tuple(order + 1 for order in chosen) + (problem.n_controls,))
        control_scale = np.maximum(np.max(np.abs(result.policy.values), axis=0), 1e-12)
        scores = []
        for dimension, degree in enumerate(chosen):
            tail = np.take(coefficients, range(max(1, degree - 1), degree + 1), axis=dimension)
            scores.append(float(np.max(np.abs(tail) / control_scale)))
        refine_dimension = int(np.argmax(scores))
        records.append(dict(level=level, orders=tuple(chosen), nodes=len(result.policy.values),
                            converged=result.converged, max_kkt_residual=audit.max_kkt_residual,
                            policy_change=change, accepted=accepted,
                            refine_state=problem.state_names[refine_dimension],
                            coefficient_tail=max(scores)))
        if accepted:
            break
        chosen[refine_dimension] += increment
        previous_values = values
        prior = result
    return PolicyRefinementResult(result, pd.DataFrame(records), accepted, audit)
