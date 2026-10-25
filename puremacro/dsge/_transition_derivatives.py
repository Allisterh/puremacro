"""Sparse symbolic first derivatives of a fixed-calibration transition model.

Only nonzero DAG derivatives are compiled. STEADY_STATE expressions keep their
fixed trajectory reference, unlike a static steady-state derivative evaluation.
Coordinate and equation-scale transforms preserve the declared sparse incidence
pattern and cache per-date sparse values once per Newton iterate for matrix-free
Jacobian-vector products without allocating dense n-by-n blocks.
"""
from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import scipy
import scipy.sparse as sp

from ._ast import Const
from .build import _Vec

_COMPILED_DAG_CACHE: dict[tuple[Any, ...], tuple[list[tuple[np.ndarray, np.ndarray]], list[slice], Callable]] = {}


def _compile_dag_derivatives(dag: Any):
    names_tuple = tuple(dag.variables)
    shocks_tuple = tuple(dag.shocks)
    eq_key = tuple(repr(eq) for eq in dag.equations)
    key = (names_tuple, shocks_tuple, eq_key)
    cached = _COMPILED_DAG_CACHE.get(key)
    if cached is not None:
        return cached[0], cached[1], cached[2]

    names = {name: j for j, name in enumerate(names_tuple)}
    pattern: list[tuple[np.ndarray, np.ndarray]] = []
    expressions: list[str] = []
    slices: list[slice] = []
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
                expressions.append(derivative.to_python(shock_names=set(shocks_tuple)))
        indices = np.asarray(entries, dtype=int).reshape(-1, 2)
        ii = indices[:, 0].copy()
        jj = indices[:, 1].copy()
        ii.flags.writeable = False
        jj.flags.writeable = False
        pattern.append((ii, jj))
        slices.append(slice(start, len(expressions)))
    body = "def evaluate(lead, curr, lag, shocks, params, steady_state):\n"
    body += "    return np.asarray([" + ",\n        ".join(expressions) + "], dtype=float)\n"
    scope: dict[str, Any] = {"np": np, "math": math, "scipy": scipy}
    exec(compile(body, "<puremacro trajectory derivatives>", "exec"), scope)
    evaluate_fn = scope["evaluate"]
    if len(_COMPILED_DAG_CACHE) > 64:
        _COMPILED_DAG_CACHE.pop(next(iter(_COMPILED_DAG_CACHE)))
    _COMPILED_DAG_CACHE[key] = (pattern, slices, evaluate_fn)
    return pattern, slices, evaluate_fn


class TrajectoryJacobian:
    """Private bridge from parsed equations to sparse date Jacobians/JVPs."""

    def __init__(
        self,
        dag: Any,
        params: dict[str, float],
        steady_state: np.ndarray,
        *,
        coords: Any = None,
        equation_scale: np.ndarray | None = None,
    ):
        self.names = tuple(dag.variables)
        self.shocks = tuple(dag.shocks)
        self.n = len(self.names)
        param_items = tuple((str(k), float(v)) for k, v in params.items())
        self._params_dict = dict(param_items)
        self.params = _Vec([k for k, _ in param_items], [v for _, v in param_items], what="parameter")
        ref_arr = np.array(steady_state, dtype=float, copy=True).ravel()
        ref_arr.flags.writeable = False
        self._reference_arr = ref_arr
        self.reference = _Vec(self.names, ref_arr, what="variable")
        self._coords = coords
        if equation_scale is None:
            scales = np.ones(self.n, dtype=float)
        else:
            scales = np.array(equation_scale, dtype=float, copy=True).ravel()
        scales.flags.writeable = False
        self._scales = scales
        self._has_transform = coords is not None or equation_scale is not None

        pattern, slices, evaluate_fn = _compile_dag_derivatives(dag)
        self.pattern = list(pattern)
        self.slices = list(slices)
        self.evaluate = evaluate_fn

        coords_key = (
            None
            if coords is None
            else (
                np.asarray(coords.lower, dtype=float).tobytes(),
                np.asarray(coords.upper, dtype=float).tobytes(),
                np.asarray(coords.scale, dtype=float).tobytes(),
                np.asarray(getattr(coords, "physical_scale", np.ones(self.n)), dtype=float).tobytes(),
            )
        )
        self.stage_key = (
            tuple(repr(eq) for eq in dag.equations),
            tuple(param_items),
            ref_arr.tobytes(),
            coords_key,
            scales.tobytes(),
        )

        # Instrumentation counters
        self.value_evaluations = 0
        self.jvp_value_evaluations = 0
        self.preconditioner_value_evaluations = 0
        self.direct_value_evaluations = 0
        self.operator_builds = 0
        self.operator_reuses = 0
        self.jvp_products = 0
        self.dense_blocks_allocated = 0
        self.cache_bytes = 0
        self.cache_invalidations = 0

        self._steady_cache_key: tuple[Any, ...] | None = None
        self._steady_cache_blocks: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._jvp_cache_key: tuple[Any, ...] | None = None
        self._jvp_cache_matvec: Callable[[np.ndarray], np.ndarray] | None = None
        self._symbolic_jacobian = self

    def invalidate_cache(self) -> None:
        """Clear cached steady blocks and per-iterate JVP operators."""
        self.cache_invalidations += 1
        self._steady_cache_key = None
        self._steady_cache_blocks = None
        self._jvp_cache_key = None
        self._jvp_cache_matvec = None

    def _eval_raw(self, lead_phys, curr_phys, lag_phys, shocks, *, role: str = "general") -> np.ndarray:
        self.value_evaluations += 1
        if role == "jvp":
            self.jvp_value_evaluations += 1
        elif role == "preconditioner":
            self.preconditioner_value_evaluations += 1
        elif role == "direct":
            self.direct_value_evaluations += 1
        return self.evaluate(
            _Vec(self.names, lead_phys),
            _Vec(self.names, curr_phys),
            _Vec(self.names, lag_phys),
            _Vec(self.shocks, np.atleast_1d(shocks)),
            self.params,
            self.reference,
        )

    def values(self, lead, curr, lag, shocks, *, role: str = "general") -> np.ndarray:
        """Evaluate flat sparse derivative values in the active coordinates."""
        if not self._has_transform:
            return self._eval_raw(lead, curr, lag, shocks, role=role)
        coords = self._coords
        if coords is not None:
            yp, yc, yl = coords.decode(lead), coords.decode(curr), coords.decode(lag)
            dp, dc, dl = coords.derivative(lead), coords.derivative(curr), coords.derivative(lag)
        else:
            yp, yc, yl = lead, curr, lag
            dp = dc = dl = np.ones(self.n, dtype=float)
        raw = self._eval_raw(yp, yc, yl, shocks, role=role).copy()
        scales = self._scales
        for section, (ii, jj), d_vec in zip(self.slices, self.pattern, (dp, dc, dl)):
            if len(ii):
                raw[section] = raw[section] * d_vec[jj] / scales[ii]
        return raw

    def values_triplet(
        self, lead, curr, lag, shocks, *, role: str = "general"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return sparse value arrays ``(vals_A, vals_B, vals_C)`` matching ``pattern``."""
        vals = self.values(lead, curr, lag, shocks, role=role)
        return (vals[self.slices[0]], vals[self.slices[1]], vals[self.slices[2]])

    def sparse_blocks(self, lead, curr, lag, shocks, *, role: str = "general"):
        values = self.values(lead, curr, lag, shocks, role=role)
        return tuple(
            sp.csr_matrix((values[section], indices), shape=(self.n, self.n))
            for section, indices in zip(self.slices, self.pattern)
        )

    def blocks(self, lead, curr, lag, shocks, *, role: str = "general"):
        self.dense_blocks_allocated += 3
        return tuple(matrix.toarray() for matrix in self.sparse_blocks(lead, curr, lag, shocks, role=role))

    def steady_or_date_blocks(self, lead, curr, lag, shocks):
        """Evaluate dense blocks for preconditioner setup, caching identical steady queries."""
        zp = np.asarray(lead, dtype=float).ravel()
        zc = np.asarray(curr, dtype=float).ravel()
        zl = np.asarray(lag, dtype=float).ravel()
        eps = np.atleast_1d(np.asarray(shocks, dtype=float))
        is_steady = np.array_equal(zp, zc) and np.array_equal(zc, zl)
        if is_steady:
            key = (self.stage_key, zp.tobytes(), eps.tobytes())
            if self._steady_cache_key == key and self._steady_cache_blocks is not None:
                return self._steady_cache_blocks
            vals = self.values(zp, zc, zl, eps, role="preconditioner")
            mats = []
            for section, (ii, jj) in zip(self.slices, self.pattern):
                M = np.zeros((self.n, self.n), dtype=float)
                if len(ii):
                    M[ii, jj] = vals[section]
                M.flags.writeable = False
                mats.append(M)
            out = (mats[0], mats[1], mats[2])
            self._steady_cache_key = key
            self._steady_cache_blocks = out
            return out
        return self.blocks(zp, zc, zl, eps, role="preconditioner")

    def jvp(self, lead, curr, lag, shocks, v_lead, v_curr, v_lag):
        vals_A, vals_B, vals_C = self.values_triplet(lead, curr, lag, shocks, role="jvp")
        out = np.zeros(self.n, dtype=float)
        for vals, (ii, jj), vec in zip(
            (vals_A, vals_B, vals_C),
            self.pattern,
            (
                np.asarray(v_lead, dtype=float).ravel(),
                np.asarray(v_curr, dtype=float).ravel(),
                np.asarray(v_lag, dtype=float).ravel(),
            ),
        ):
            if len(ii):
                np.add.at(out, ii, vals * vec[jj])
        return out

    def make_stacked_jvp_factory(
        self,
        z_init: np.ndarray,
        z_end: np.ndarray,
        exogenous_path: np.ndarray,
        *,
        terminal_policy: Callable | None = None,
        terminal_derivative: Callable | None = None,
    ) -> Callable[[np.ndarray], Callable[[np.ndarray], np.ndarray]]:
        """Build a ``StackedProblem.stacked_jvp_factory`` caching sparse values once per iterate."""
        z0 = np.array(z_init, dtype=float, copy=True).ravel()
        zend = np.array(z_end, dtype=float, copy=True).ravel()
        exo = np.array(exogenous_path, dtype=float, copy=True)
        z0.flags.writeable = False
        zend.flags.writeable = False
        exo.flags.writeable = False
        T = int(exo.shape[0])
        n = self.n
        (ii_A, jj_A), (ii_B, jj_B), (ii_C, jj_C) = self.pattern
        slice_A, slice_B, slice_C = self.slices
        m_A, m_B, m_C = len(ii_A), len(ii_B), len(ii_C)
        index_bytes = int(ii_A.nbytes + jj_A.nbytes + ii_B.nbytes + jj_B.nbytes + ii_C.nbytes + jj_C.nbytes)
        boundary_key = (
            self.stage_key,
            z0.tobytes(),
            zend.tobytes(),
            exo.tobytes(),
            id(terminal_policy) if terminal_policy is not None else None,
        )

        def factory(Y_flat: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
            Y_arr = np.array(Y_flat, dtype=float, copy=True).reshape(T, n)
            Y_arr.flags.writeable = False
            iterate_key = (boundary_key, Y_arr.tobytes())
            if self._jvp_cache_key == iterate_key and self._jvp_cache_matvec is not None:
                self.operator_reuses += 1
                return self._jvp_cache_matvec

            self.operator_builds += 1
            coords = self._coords
            scales = self._scales
            if coords is not None:
                Y_phys = coords.decode(Y_arr)
                y0_phys = coords.decode(z0)
                yT_phys = coords.decode(zend)
                D_Y = coords.derivative(Y_arr)
            else:
                Y_phys = Y_arr
                y0_phys = z0
                yT_phys = zend
                D_Y = None

            vals_B = np.empty((T, m_B), dtype=float)
            vals_A = np.empty((max(0, T - 1), m_A), dtype=float)
            vals_C = np.empty((max(0, T - 1), m_C), dtype=float)
            vals_A_last: np.ndarray | None = None
            D_term_last: np.ndarray | None = None

            for t in range(T):
                yl_phys = y0_phys if t == 0 else Y_phys[t - 1]
                yc_phys = Y_phys[t]
                if t == T - 1:
                    if terminal_policy is not None:
                        zp_last = np.asarray(terminal_policy(Y_arr[t].copy()), dtype=float).ravel()
                        yp_phys = coords.decode(zp_last) if coords is not None else zp_last
                        dp_last = coords.derivative(zp_last) if coords is not None else None
                        if terminal_derivative is not None:
                            D_term_last = np.asarray(terminal_derivative(Y_arr[t].copy()), dtype=float)
                    else:
                        yp_phys = yT_phys
                        dp_last = None
                else:
                    yp_phys = Y_phys[t + 1]
                    dp_last = None

                raw_t = self._eval_raw(yp_phys, yc_phys, yl_phys, exo[t], role="jvp")
                if not self._has_transform:
                    if m_B:
                        vals_B[t] = raw_t[slice_B]
                    if t < T - 1 and m_A:
                        vals_A[t] = raw_t[slice_A]
                    elif t == T - 1 and terminal_policy is not None and m_A:
                        vals_A_last = raw_t[slice_A].copy()
                    if t > 0 and m_C:
                        vals_C[t - 1] = raw_t[slice_C]
                else:
                    dc_t = D_Y[t] if D_Y is not None else None
                    if m_B:
                        b_entries = raw_t[slice_B]
                        if dc_t is not None:
                            b_entries = b_entries * dc_t[jj_B]
                        vals_B[t] = b_entries / scales[ii_B]
                    if t < T - 1 and m_A:
                        dp_t = D_Y[t + 1] if D_Y is not None else None
                        a_entries = raw_t[slice_A]
                        if dp_t is not None:
                            a_entries = a_entries * dp_t[jj_A]
                        vals_A[t] = a_entries / scales[ii_A]
                    elif t == T - 1 and terminal_policy is not None and m_A:
                        a_entries = raw_t[slice_A]
                        if dp_last is not None:
                            a_entries = a_entries * dp_last[jj_A]
                        vals_A_last = (a_entries / scales[ii_A]).copy()
                    if t > 0 and m_C:
                        dl_t = D_Y[t - 1] if D_Y is not None else None
                        c_entries = raw_t[slice_C]
                        if dl_t is not None:
                            c_entries = c_entries * dl_t[jj_C]
                        vals_C[t - 1] = c_entries / scales[ii_C]

            extra_bytes = (0 if vals_A_last is None else vals_A_last.nbytes) + (
                0 if D_term_last is None else D_term_last.nbytes
            )
            self.cache_bytes = int(vals_B.nbytes + vals_A.nbytes + vals_C.nbytes + index_bytes + extra_bytes)

            def matvec(v: np.ndarray) -> np.ndarray:
                self.jvp_products += 1
                V = np.asarray(v, dtype=float).reshape(T, n)
                out_T = np.zeros((n, T), dtype=float)
                if m_B:
                    np.add.at(out_T, ii_B, (vals_B * V[:, jj_B]).T)
                if m_A and T > 1:
                    np.add.at(out_T[:, :-1], ii_A, (vals_A * V[1:, jj_A]).T)
                if m_C and T > 1:
                    np.add.at(out_T[:, 1:], ii_C, (vals_C * V[:-1, jj_C]).T)
                if vals_A_last is not None and D_term_last is not None and m_A:
                    vp_last = D_term_last @ V[-1]
                    np.add.at(out_T[:, -1], ii_A, vals_A_last * vp_last[jj_A])
                return out_T.T.ravel()

            self._jvp_cache_key = iterate_key
            self._jvp_cache_matvec = matvec
            return matvec

        return factory
