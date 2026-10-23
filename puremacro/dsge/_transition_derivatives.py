"""Sparse symbolic first derivatives of a fixed-calibration transition model.

Only nonzero DAG derivatives are compiled. STEADY_STATE expressions keep their
fixed trajectory reference, unlike a static steady-state derivative evaluation.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import scipy
import scipy.sparse as sp

from ._ast import Const
from .build import _Vec


class TrajectoryJacobian:
    """Private bridge from parsed equations to sparse date Jacobians/JVPs."""

    def __init__(self, dag: Any, params: dict[str, float], steady_state: np.ndarray):
        self.names = tuple(dag.variables)
        self.shocks = tuple(dag.shocks)
        self.n = len(self.names)
        self.params = _Vec(list(params), list(params.values()), what="parameter")
        self.reference = _Vec(self.names, steady_state.copy(), what="variable")
        names = {name: j for j, name in enumerate(self.names)}
        self.pattern: list[tuple[np.ndarray, np.ndarray]] = []
        expressions: list[str] = []
        self.slices = []
        for shift in (1, 0, -1):
            entries = []
            start = len(expressions)
            for i, equation in enumerate(dag.equations):
                for name, lag in sorted(equation.variables()):
                    if lag != shift or name not in names:
                        continue
                    derivative = equation.diff(name, shift).simplify()
                    if isinstance(derivative, Const) and derivative.value == 0:
                        continue
                    entries.append((i, names[name]))
                    expressions.append(derivative.to_python(shock_names=set(self.shocks)))
            indices = np.asarray(entries, dtype=int).reshape(-1, 2)
            self.pattern.append((indices[:, 0], indices[:, 1]))
            self.slices.append(slice(start, len(expressions)))
        body = "def evaluate(lead, curr, lag, shocks, params, steady_state):\n"
        body += "    return np.asarray([" + ",\n        ".join(expressions) + "], dtype=float)\n"
        scope: dict[str, Any] = {"np": np, "math": math, "scipy": scipy}
        exec(compile(body, "<puremacro trajectory derivatives>", "exec"), scope)
        self.evaluate = scope["evaluate"]

    def values(self, lead, curr, lag, shocks):
        return self.evaluate(
            _Vec(self.names, lead), _Vec(self.names, curr), _Vec(self.names, lag),
            _Vec(self.shocks, np.atleast_1d(shocks)), self.params, self.reference,
        )

    def sparse_blocks(self, lead, curr, lag, shocks):
        values = self.values(lead, curr, lag, shocks)
        return tuple(sp.csr_matrix((values[section], indices), shape=(self.n, self.n))
                     for section, indices in zip(self.slices, self.pattern))

    def blocks(self, lead, curr, lag, shocks):
        return tuple(matrix.toarray() for matrix in self.sparse_blocks(lead, curr, lag, shocks))

    def jvp(self, lead, curr, lag, shocks, v_lead, v_curr, v_lag):
        A, B, C = self.sparse_blocks(lead, curr, lag, shocks)
        return A @ v_lead + B @ v_curr + C @ v_lag
