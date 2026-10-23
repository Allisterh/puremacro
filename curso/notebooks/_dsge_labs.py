"""Explicit recursive timing shared by the MAV growth solution laboratories.

Economic equilibrium equations are evaluated from the loaded .mod model.
This adapter only eliminates production/accounting identities and declares
state/control timing, integration and admissibility for the global solver.
"""
from __future__ import annotations

import numpy as np

from puremacro.dsge import EquilibriumProblem, evaluate_perturbation, gaussian_quadrature


def tfp_innovations(productivity, rho, *, initial_log=0.0):
    """Innovations implementing a declared path of productivity LEVELS."""
    values = np.asarray(productivity, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("productivity must be a nonempty positive finite level path")
    log_path = np.log(values)
    return (log_path - float(rho) * np.r_[float(initial_log), log_path[:-1]])[:, None]


def growth_problem(model, steady_state, *, capital_range=(0.7, 1.4),
                   productivity_range=(-0.12, 0.12), quadrature_order=3, sigma=None):
    """Neoclassical/RBC equilibrium with state (inherited k, observed log A).

    The controls are current consumption and hours. The model's tagged
    intratemporal and Euler residuals are retained exactly; static equations
    determine output, investment, current capital and factor prices.
    Gaussian integration is finite quadrature, not bounded-support risk.
    """
    params = dict(model.parameter_values)
    names = tuple(model.variables)
    required = {"y", "c", "k", "l", "a"}
    if not required <= set(names) or tuple(model.shocks) != ("e_a",):
        raise ValueError("This course adapter requires the neoclassical/RBC model and one e_a innovation")
    ss = {name: float(steady_state[name]) for name in names}
    ss_array = np.array([ss[name] for name in names])
    alpha, delta, rho = (params[name] for name in ("alpha", "delta", "rho_a"))
    n = params.get("n", 1.0)
    covariance = model.shock_cov if sigma is None else np.array([[float(sigma) ** 2]])
    if covariance is None:
        raise ValueError("Declare the innovation covariance or an explicit sigma")
    nodes, weights = gaussian_quadrature(covariance, order=quadrature_order)

    def current(state, controls):
        capital, log_tfp = state
        consumption, labor = controls
        output = np.exp(log_tfp) * capital ** alpha * labor ** (1 - alpha)
        investment = output - consumption
        values = {"y": output, "c": consumption, "k": (1 - delta) * capital + investment,
                  "l": labor, "a": log_tfp, "i": investment,
                  "w": (1 - alpha) * output / labor, "r": alpha * output / capital}
        return np.array([values[name] for name in names])

    def transition(state, controls, innovation):
        endogenous = current(state, controls)
        return np.array([endogenous[names.index("k")], rho * state[1] + innovation[0]])

    def lagged(state):
        values = ss_array.copy()
        values[names.index("k")] = state[0]
        values[names.index("a")] = 0.0
        return values

    def admissible(state, controls):
        if state[0] <= 0 or controls[0] <= 0 or not 0 < controls[1] < n:
            return False
        return bool(current(state, controls)[names.index("k")] > 0)

    domain = [(capital_range[0] * ss["k"], capital_range[1] * ss["k"]), productivity_range]
    return EquilibriumProblem.from_model(
        model, state_names=("k_lag", "a"), control_names=("c", "l"), domain=domain,
        transition=transition, current=current, lagged=lagged,
        equations=("intratemporal", "Euler"), exogenous=lambda state: np.array([state[1]]),
        steady_state=ss_array, shock_nodes=nodes, shock_weights=weights,
        control_bounds=np.array([[1e-8, np.inf], [1e-6, n - 1e-6]]),
        residual_scale=np.array([max(1.0, ss["c"]), 1.0 / ss["c"]]),
        admissible=admissible,
        metadata={"source": "course growth .mod", "state_timing": "inherited capital, observed current log productivity"},
    )


def growth_local_policy(solution, *, sigma=1.0):
    """Raw Taylor control rule for the same recursive states as growth_problem.

    With a current AR(1) productivity observation, set lagged log productivity
    to zero and its current innovation to that observation. This is an explicit
    timing map, not a pruned simulation with its additional component history.
    """
    states = tuple(solution.states)
    names = tuple(solution.decision_rules().variable_names)
    shocks = tuple(solution.shocks)
    ss = solution.steady_state
    if set(states) != {"k", "a"} or shocks != ("e_a",):
        raise ValueError("The growth policy adapter requires states k/a and shock e_a")
    control_indices = [names.index("c"), names.index("l")]

    def policy(state):
        state = np.asarray(state, dtype=float)
        single = state.ndim == 1
        rows = np.atleast_2d(state)
        lag = np.tile(np.array([float(ss[name]) for name in states]), (len(rows), 1))
        lag[:, states.index("k")] = rows[:, 0]
        lag[:, states.index("a")] = 0.0
        values = evaluate_perturbation(solution, lag, rows[:, 1:2], sigma=sigma)
        controls = values[:, control_indices]
        return controls[0] if single else controls

    return policy


def growth_audit_points(steady_state, *, count=64, seed=824):
    """Independent interior points shared by every policy in a comparison."""
    rng = np.random.default_rng(seed)
    k = float(steady_state["k"])
    return rng.uniform([0.85 * k, -0.04], [1.2 * k, 0.04], size=(count, 2))
