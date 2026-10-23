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
              policy: Policy) -> tuple[Array, float, bool]:
    values = np.zeros(problem.n_controls)
    outside_mass = 0.0
    future_feasible = True
    nodes, weights = problem.integration(state)
    for innovation, weight in zip(nodes, weights):
        next_state = _vector(problem.transition(state, control, innovation), problem.n_states, "transition")
        outside_mass += float(weight) * float(not problem.contains(next_state)[0])
        next_control = _vector(policy(next_state), problem.n_controls, "future policy")
        lo, hi = problem.bounds(next_state)
        future_feasible &= bool(np.all(next_control >= lo - 1e-10) and np.all(next_control <= hi + 1e-10))
        if problem.admissible is not None:
            future_feasible &= bool(problem.admissible(next_state, next_control))
        values += weight * _vector(problem.residual(state, control, next_state, next_control, innovation), problem.n_controls, "equilibrium residual")
    return values, outside_mass, future_feasible


def _kkt(problem: EquilibriumProblem, state: Array, control: Array, residual: Array) -> Array:
    out = residual / problem.residual_scale
    lo, hi = problem.bounds(state)
    for equation, index in problem.complementarity.items():
        # Natural-map complementarity residual; lower bound requires f >= 0.
        out[equation] = control[index] - np.clip(control[index] - out[equation], lo[index], hi[index])
    return out


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
        self.factor = None if basis == "linear" else lu_factor(self.matrix(self.nodes))

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

    def fit(self, values: Array, *, extrapolation: str) -> "EquilibriumPolicy":
        coefficients = None if self.factor is None else lu_solve(self.factor, values)
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

    def __call__(self, states: Any) -> Array:
        return self.policy(states)

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
    previous = approximation.fit(values, extrapolation="allow")
    audit = audit_policy(problem, previous, approximation.nodes, tol=tol, require_domain=require_domain)
    for iteration in range(1, max_iter + 1):
        candidate = values.copy()
        failed_nodes = 0
        evaluations = 0
        failure = ""
        for i, state in enumerate(approximation.nodes):
            lo, hi = problem.bounds(state)

            def equations(control: Array) -> Array:
                raw, _, _ = _expected(problem, state, control, previous)
                return _kkt(problem, state, control, raw)

            try:
                solved = least_squares(equations, values[i], bounds=(lo, hi),
                                       max_nfev=inner_max_iter, xtol=1e-12, ftol=1e-12, gtol=None)
                # The bound-weighted gradient can be tiny while a natural-map
                # residual remains above tolerance. Accept on the independently
                # evaluated equations, not least_squares' gradient flag.
                candidate[i] = solved.x
                evaluations += solved.nfev
                failed_nodes += int(np.max(np.abs(equations(solved.x))) > tol)
            except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
                failed_nodes += 1
                failure = f"{type(exc).__name__}: {exc}"
        updated = values + damping * (candidate - values)
        step = float(np.max(np.abs(updated - values)))
        previous = approximation.fit(updated, extrapolation="allow")
        audit = audit_policy(problem, previous, approximation.nodes, tol=tol, require_domain=require_domain)
        history.append(dict(iteration=iteration, policy_change=step,
                            max_kkt_residual=audit.max_kkt_residual,
                            next_state_outside_mass=float(audit.diagnostics.next_state_outside_mass.max()),
                            inner_failures=failed_nodes, evaluations=evaluations, failure=failure))
        values = updated
        if audit.passed and step <= tol and failed_nodes == 0:
            converged = True
            message = ("node residuals, feasibility, domain closure and policy change" if require_domain else
                       "node residuals, feasibility and policy change; conditional on extrapolation")
            break
        if failure:
            message = "inner numerical failure: " + failure
            break
    final_policy = approximation.fit(values, extrapolation=extrapolation)
    result = TimeIterationResult(final_policy, converged, len(history), pd.DataFrame(history), audit,
                                 tuple(problem.state_names), tuple(problem.control_names), message, require_domain)
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
