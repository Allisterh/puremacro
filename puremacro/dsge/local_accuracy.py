"""Evaluate local decision rules and expose their numerical conditioning.

These are raw Taylor policies, suitable for initial guesses and independent
equilibrium-residual comparisons. They are not a replacement for pruned
simulation: pruning retains separate order components of the state history.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ._policy_reporting import _PolicyReport


def evaluate_perturbation(solution: Any, lagged_states: Any, shocks: Any = None,
                          *, sigma: float = 1.0, order: int | None = None,
                          deviations: bool = False) -> np.ndarray:
    """Evaluate an order-1/2/3 Taylor rule in Dynare decision-rule timing.

    ``lagged_states`` contains end-of-previous-period states, in
    ``solution.decision_rules().state_variables`` order. It contains levels
    unless ``deviations=True``. ``shocks`` contains current realized innovations
    in ``shock_names`` order; these are not multiplied by ``sigma``. ``sigma``
    scales anticipated innovation risk in the constant and third-order mixed
    risk terms. Re-solving is necessary to change relative shock covariance.

    Returns endogenous levels in ``variable_names`` order, shape ``(n_vars,)``
    for a single point or ``(n_points,n_vars)`` for a batch. Dense coefficient
    tensors are contracted directly: no Kronecker vectors are allocated.
    """
    rules = solution.decision_rules()
    available = 3 if hasattr(rules, "ghxxx") else 2 if hasattr(rules, "ghxx") else 1
    if order is None:
        order = available
    if isinstance(order, (bool, np.bool_)) or order not in (1, 2, 3) or order > available:
        raise ValueError(f"order must be in 1..{available} for this solution")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative")
    states = np.asarray(lagged_states, dtype=float)
    single = states.ndim == 1
    if single:
        states = states.reshape(1, -1)
    n_states, n_shocks = len(rules.state_variables), len(rules.shock_names)
    variables = list(rules.variable_names)
    n_vars = len(variables)
    if states.ndim != 2 or states.shape[1] != n_states or not np.isfinite(states).all():
        raise ValueError(f"lagged_states must have {n_states} finite columns")
    innovations = np.zeros((len(states), n_shocks)) if shocks is None else np.asarray(shocks, dtype=float)
    if innovations.ndim == 1:
        innovations = np.broadcast_to(innovations, (len(states), n_shocks))
    if innovations.shape != (len(states), n_shocks) or not np.isfinite(innovations).all():
        raise ValueError(f"shocks must have {n_shocks} finite columns and match state rows")
    steady = rules.ys.loc[variables].to_numpy(dtype=float)
    state_steady = rules.ys.loc[list(rules.state_variables)].to_numpy(dtype=float)
    x = states if deviations else states - state_steady
    u = innovations

    def matrix(name: str) -> np.ndarray:
        return getattr(rules, name).loc[variables].to_numpy(dtype=float)

    output = steady + x @ matrix("ghx").T + u @ matrix("ghu").T
    if order >= 2:
        output += 0.5 * np.einsum("ijk,bj,bk->bi", matrix("ghxx").reshape(n_vars, n_states, n_states), x, x, optimize=True)
        output += np.einsum("ijk,bj,bk->bi", matrix("ghxu").reshape(n_vars, n_states, n_shocks), x, u, optimize=True)
        output += 0.5 * np.einsum("ijk,bj,bk->bi", matrix("ghuu").reshape(n_vars, n_shocks, n_shocks), u, u, optimize=True)
        output += 0.5 * sigma ** 2 * matrix("ghs2")
    if order >= 3:
        output += np.einsum("ijkl,bj,bk,bl->bi", matrix("ghxxx").reshape(n_vars, n_states, n_states, n_states), x, x, x, optimize=True) / 6
        output += np.einsum("ijkl,bj,bk,bl->bi", matrix("ghxxu").reshape(n_vars, n_states, n_states, n_shocks), x, x, u, optimize=True) / 2
        output += np.einsum("ijkl,bj,bk,bl->bi", matrix("ghxuu").reshape(n_vars, n_states, n_shocks, n_shocks), x, u, u, optimize=True) / 2
        output += np.einsum("ijkl,bj,bk,bl->bi", matrix("ghuuu").reshape(n_vars, n_shocks, n_shocks, n_shocks), u, u, u, optimize=True) / 6
        output += 0.5 * sigma ** 2 * (x @ matrix("ghxss").T + u @ matrix("ghuss").T)
    return output[0] if single else output


@dataclass(frozen=True)
class LocalDiagnostics(_PolicyReport):
    """Conditioning evidence for an already solved local approximation."""

    roots: pd.DataFrame
    jacobian_rank: int | None
    equation_count: int | None
    jacobian_condition: float | None
    warnings: tuple[str, ...]

    def summary(self) -> str:
        warning = "; ".join(self.warnings) if self.warnings else "no conditioning warning detected"
        return f"Local diagnostics: {warning}; dynamic Jacobian rank={self.jacobian_rank}/{self.equation_count}"

    def to_frame(self) -> pd.DataFrame:
        """Return an independent copy of the transition-root diagnostics."""
        return self.roots.copy()


def diagnose_perturbation(solution: Any, *, unit_root_band: float = 1e-5,
                          condition_warning: float = 1e10) -> LocalDiagnostics:
    """Report near-unit transition roots and dynamic-equation rank/conditioning.

    This report does not redo a Blanchard-Kahn classification or establish
    uniqueness of a nonlinear equilibrium. The transition spectrum describes
    the selected local solution; the original Jacobian is inspected only when
    retained by that solution.
    """
    if not np.isfinite(unit_root_band) or unit_root_band <= 0:
        raise ValueError("unit_root_band must be finite and positive")
    if not np.isfinite(condition_warning) or condition_warning <= 1:
        raise ValueError("condition_warning must exceed one")
    linear = getattr(solution, "first_order", None)
    if linear is None:
        linear = solution
    transition = getattr(solution, "G", None)
    if transition is None:
        solved = getattr(linear, "solution", None)
        transition = getattr(solved, "G", None)
    if transition is None:
        raise ValueError("solution does not retain a state transition matrix")
    transition = np.asarray(transition, dtype=float)
    eigenvalues = np.linalg.eigvals(transition)
    modulus = np.abs(eigenvalues)
    roots = pd.DataFrame(dict(real=eigenvalues.real, imaginary=eigenvalues.imag,
                              modulus=modulus, distance_to_unit_circle=np.abs(modulus - 1),
                              near_unit=np.abs(modulus - 1) <= unit_root_band,
                              stable=modulus < 1))
    messages = []
    if bool(roots.near_unit.any()):
        messages.append("transition roots near the unit circle; persistence and horizon accuracy are sensitive")
    if bool((~roots.stable).any()):
        messages.append("the selected transition has nonstationary roots")
    rank = count = None
    condition = None
    blocks = [getattr(linear, name, None) for name in ("_A_minus", "_A_0", "_A_plus")]
    if all(block is not None for block in blocks):
        jacobian = np.hstack([np.asarray(block, dtype=float) for block in blocks])
        rank = int(np.linalg.matrix_rank(jacobian))
        count = jacobian.shape[0]
        condition = float(np.linalg.cond(jacobian))
        if rank < count:
            messages.append("dynamic equation Jacobian is row-rank deficient")
        if condition > condition_warning:
            messages.append("dynamic equation Jacobian is ill-conditioned in its supplied units")
    else:
        messages.append("original dynamic Jacobian was not retained; rank/conditioning not checked")
    return LocalDiagnostics(roots, rank, count, condition, tuple(messages))
