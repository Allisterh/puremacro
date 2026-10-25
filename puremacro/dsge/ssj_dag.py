r"""Generalized Directed Acyclic Graph (DAG) Sequence-Space Jacobians for HANK & DSGE.

Implements the Auclert, Bardóczy, Rognlie & Straub (2021, *Econometrica*)
Sequence-Space Jacobian (SSJ) block-DAG compiler and general-equilibrium
solver in pure NumPy/SciPy:

1. `SimpleBlock`:
   Maps input sequences (with arbitrary lags/leads $x_{t+k}$) to output sequences.
   Automatically differentiates the block at the steady state using
   `puremacro._autodiff.DualArray` to assemble exact $(T \times T)$ Toeplitz
   sequence-space Jacobians without manual derivative formulas.

2. `HetBlock`:
   Wraps heterogeneous-agent Fake-News Sequence-Space Jacobians
   ($\mathcal{J}^{Y, X} \in \mathbb{R}^{T \times T}$) from `dp_sequence_jacobian`,
   1-asset/2-asset EGM, or continuous-time Achdou HJB solvers, together with an
   optional nonlinear transition evaluator.

3. `SolvedBlock`:
   Resolves an implicit sub-block $h(u, x) = 0$ via $\mathcal{J}^{u, x} = -H_u^{-1} H_x$.

4. `BlockDAG`:
   Topologically orders blocks, propagates total Jacobians via the sequence-space
   chain rule
   $$\mathcal{J}^{Y, Z} = \sum_{X} \mathcal{J}^{Y, X} \mathcal{J}^{X, Z},$$
   resolves general-equilibrium unknowns via $G_U = -H_U^{-1} H_Z$, extracts
   Wold moving-average matrices $\{M_s\}_{s=0}^{T-1}$ for Whittle likelihood and
   IRF matching, and solves full nonlinear transitions via preconditioned
   quasi-Newton iteration around $H_U^{-1}$.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

if TYPE_CHECKING:
    import matplotlib.figure

import numpy as np
import pandas as pd
import scipy.linalg

from puremacro._autodiff import DualArray, seed_dual_vector
from puremacro.reports import _df_to_latex, _df_to_markdown, _df_to_typst

__all__ = [
    "SimpleBlock",
    "HetBlock",
    "SolvedBlock",
    "BlockDAG",
    "SSJDAGResult",
]


def _shift_series(series: np.ndarray, lead: int, ss_val: float) -> np.ndarray:
    """Shift a 1D path of length `T` by `lead` periods (lead > 0 forward, lead < 0 lag)."""
    T = len(series)
    if lead == 0:
        return series
    out = np.empty(T, dtype=np.float64)
    if lead > 0:
        if lead >= T:
            out.fill(ss_val)
        else:
            out[: T - lead] = series[lead:]
            out[T - lead :] = ss_val
    else:
        lag = -lead
        if lag >= T:
            out.fill(ss_val)
        else:
            out[:lag] = ss_val
            out[lag:] = series[: T - lag]
    return out


def _toeplitz_shift_matrix(T: int, lead: int, weight: float) -> np.ndarray:
    """Return `weight * S_lead` of shape `(T, T)` where `(S_lead @ dx)_t = dx_{t + lead}`."""
    mat = np.zeros((T, T), dtype=np.float64)
    if abs(lead) >= T or weight == 0.0:
        return mat
    if lead == 0:
        np.fill_diagonal(mat, weight)
    elif lead > 0:
        idx = np.arange(T - lead)
        mat[idx, idx + lead] = weight
    else:
        lag = -lead
        idx = np.arange(lag, T)
        mat[idx, idx - lag] = weight
    return mat


class _ShiftedVarProxy:
    """Callable variable handle inside `SimpleBlock` supporting `x(0)`, `x(-1)`, `x(+1)`."""

    def __init__(self, base: Any, shift_resolver: Callable[[int], Any]) -> None:
        self._base = base
        self._shift_resolver = shift_resolver

    def __call__(self, lead: int = 0) -> Any:
        return self._shift_resolver(int(lead))

    def __getattr__(self, item: str) -> Any:
        return getattr(self._base, item)

    def _binary(self, other: Any, op: Callable[[Any, Any], Any]) -> Any:
        other_val = other._base if isinstance(other, _ShiftedVarProxy) else other
        return op(self._base, other_val)

    def __add__(self, other: Any) -> Any:
        return self._binary(other, lambda a, b: a + b)

    def __radd__(self, other: Any) -> Any:
        return self._binary(other, lambda a, b: b + a)

    def __sub__(self, other: Any) -> Any:
        return self._binary(other, lambda a, b: a - b)

    def __rsub__(self, other: Any) -> Any:
        return self._binary(other, lambda a, b: b - a)

    def __mul__(self, other: Any) -> Any:
        return self._binary(other, lambda a, b: a * b)

    def __rmul__(self, other: Any) -> Any:
        return self._binary(other, lambda a, b: b * a)

    def __truediv__(self, other: Any) -> Any:
        return self._binary(other, lambda a, b: a / b)

    def __rtruediv__(self, other: Any) -> Any:
        return self._binary(other, lambda a, b: b / a)

    def __pow__(self, other: Any) -> Any:
        return self._binary(other, lambda a, b: a ** b)

    def __rpow__(self, other: Any) -> Any:
        return self._binary(other, lambda a, b: b ** a)

    def __neg__(self) -> Any:
        return -self._base

    def __pos__(self) -> Any:
        return +self._base

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        unwrapped = [inp._base if isinstance(inp, _ShiftedVarProxy) else inp for inp in inputs]
        return getattr(ufunc, method)(*unwrapped, **kwargs)


@dataclass(frozen=True)
class SimpleBlock:
    """Analytical sequence-space block with automatic `DualArray` Toeplitz differentiation.

    Parameters
    ----------
    name : str
        Block identifier.
    inputs : Sequence[str]
        Input variable names consumed by `fn`.
    outputs : Sequence[str]
        Output variable names returned by `fn`.
    fn : Callable[[Mapping[str, Any], Mapping[str, float]], Mapping[str, Any]]
        Function `(vars_dict, params_dict) -> outputs_dict`. Inside `fn`, any
        input variable `x = vars_dict["x"]` can be used directly for date `t` or
        called as `x(-1)` / `x(+1)` for lags and leads.
    max_lag : int, default 2
        Maximum lag order probed during automatic Toeplitz Jacobian assembly.
    max_lead : int, default 2
        Maximum lead order probed during automatic Toeplitz Jacobian assembly.
    """

    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    fn: Callable[[Mapping[str, Any], Mapping[str, float]], Mapping[str, Any]]
    max_lag: int = 2
    max_lead: int = 2

    def __init__(
        self,
        name: str,
        inputs: Sequence[str],
        outputs: Sequence[str],
        fn: Callable[[Mapping[str, Any], Mapping[str, float]], Mapping[str, Any]],
        *,
        max_lag: int = 2,
        max_lead: int = 2,
    ) -> None:
        object.__setattr__(self, "name", str(name))
        object.__setattr__(self, "inputs", tuple(str(x) for x in inputs))
        object.__setattr__(self, "outputs", tuple(str(y) for y in outputs))
        object.__setattr__(self, "fn", fn)
        object.__setattr__(self, "max_lag", int(max_lag))
        object.__setattr__(self, "max_lead", int(max_lead))

    def jacobian(
        self,
        steady_state: Mapping[str, float],
        params: Mapping[str, float],
        T: int,
    ) -> dict[str, dict[str, np.ndarray]]:
        """Assemble exact `(T, T)` Toeplitz sequence-space Jacobians via `DualArray`."""
        shifts = list(range(-self.max_lag, self.max_lead + 1))
        n_shifts = len(shifts)
        n_in = len(self.inputs)
        P = n_in * n_shifts

        base_vals = np.array(
            [float(steady_state[inp]) for inp in self.inputs for _ in shifts],
            dtype=np.float64,
        )
        dual_vec = seed_dual_vector(base_vals)

        var_proxies: dict[str, Any] = {}
        for i, inp in enumerate(self.inputs):
            offset = i * n_shifts
            zero_idx = offset + shifts.index(0)

            def _make_resolver(off: int, ss_v: float):
                def _resolve(lead: int) -> Any:
                    if lead in shifts:
                        return dual_vec[off + shifts.index(lead)]
                    return ss_v

                return _resolve

            var_proxies[inp] = _ShiftedVarProxy(
                dual_vec[zero_idx],
                _make_resolver(offset, float(steady_state[inp])),
            )

        raw_out = self.fn(var_proxies, params)
        jac_out: dict[str, dict[str, np.ndarray]] = {out: {} for out in self.outputs}

        for out_name in self.outputs:
            val_obj = raw_out[out_name]
            if isinstance(val_obj, _ShiftedVarProxy):
                val_obj = val_obj._base
            if isinstance(val_obj, DualArray):
                ders = np.asarray(val_obj.der, dtype=np.float64).reshape(P)
            else:
                ders = np.zeros(P, dtype=np.float64)

            for i, inp in enumerate(self.inputs):
                mat = np.zeros((T, T), dtype=np.float64)
                offset = i * n_shifts
                for s_idx, lead in enumerate(shifts):
                    w = float(ders[offset + s_idx])
                    if w != 0.0:
                        mat += _toeplitz_shift_matrix(T, lead, w)
                jac_out[out_name][inp] = mat

        return jac_out

    def evaluate(
        self,
        paths: Mapping[str, np.ndarray],
        steady_state: Mapping[str, float],
        params: Mapping[str, float],
        T: int,
    ) -> dict[str, np.ndarray]:
        """Evaluate the nonlinear block along time paths of length `T`."""
        var_proxies: dict[str, Any] = {}
        for inp in self.inputs:
            series = np.asarray(
                paths.get(inp, np.full(T, float(steady_state[inp]), dtype=np.float64)),
                dtype=np.float64,
            )
            ss_v = float(steady_state[inp])

            def _make_path_resolver(ser: np.ndarray, sv: float):
                def _resolve(lead: int) -> np.ndarray:
                    return _shift_series(ser, int(lead), sv)

                return _resolve

            var_proxies[inp] = _ShiftedVarProxy(series, _make_path_resolver(series, ss_v))

        raw_out = self.fn(var_proxies, params)
        out_paths: dict[str, np.ndarray] = {}
        for out_name in self.outputs:
            v = raw_out[out_name]
            if isinstance(v, _ShiftedVarProxy):
                v = v._base
            arr = np.asarray(v, dtype=np.float64)
            if arr.ndim == 0:
                arr = np.full(T, float(arr), dtype=np.float64)
            out_paths[out_name] = arr
        return out_paths


@dataclass(frozen=True)
class HetBlock:
    """Heterogeneous-agent Sequence-Space Jacobian block.

    Parameters
    ----------
    name : str
        Block identifier.
    inputs : Sequence[str]
        Aggregate input names into the household block (e.g. `("r", "w")`).
    outputs : Sequence[str]
        Aggregate output names from the household block (e.g. `("C", "A")`).
    jacobian_fn : Callable[[int], Mapping[str, Mapping[str, np.ndarray]]] | Mapping[str, Mapping[str, np.ndarray]]
        Precomputed nested mapping `{output: {input: J_matrix}}` or callable `fn(T)`
        returning `(T, T)` Fake-News Jacobians.
    nonlinear_fn : Callable[[Mapping[str, np.ndarray], Mapping[str, float], int], Mapping[str, np.ndarray]] | None
        Optional nonlinear transition evaluator. When omitted, evaluates the
        exact first-order sequence-space response around `steady_state`.
    """

    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    jacobian_fn: Any
    nonlinear_fn: Callable[[Mapping[str, np.ndarray], Mapping[str, float], int], Mapping[str, np.ndarray]] | None = None

    def __init__(
        self,
        name: str,
        inputs: Sequence[str],
        outputs: Sequence[str],
        jacobian_fn: Any,
        *,
        nonlinear_fn: Callable[[Mapping[str, np.ndarray], Mapping[str, float], int], Mapping[str, np.ndarray]] | None = None,
    ) -> None:
        object.__setattr__(self, "name", str(name))
        object.__setattr__(self, "inputs", tuple(str(x) for x in inputs))
        object.__setattr__(self, "outputs", tuple(str(y) for y in outputs))
        object.__setattr__(self, "jacobian_fn", jacobian_fn)
        object.__setattr__(self, "nonlinear_fn", nonlinear_fn)

    def jacobian(
        self,
        steady_state: Mapping[str, float],
        params: Mapping[str, float],
        T: int,
    ) -> dict[str, dict[str, np.ndarray]]:
        raw = self.jacobian_fn(T) if callable(self.jacobian_fn) else self.jacobian_fn
        out: dict[str, dict[str, np.ndarray]] = {o: {} for o in self.outputs}
        for o in self.outputs:
            for i in self.inputs:
                mat = np.asarray(raw[o][i], dtype=np.float64)
                if mat.shape != (T, T):
                    raise ValueError(
                        f"HetBlock {self.name!r} Jacobian ({o}, {i}) has shape {mat.shape}, expected ({T}, {T})"
                    )
                out[o][i] = mat
        return out

    def evaluate(
        self,
        paths: Mapping[str, np.ndarray],
        steady_state: Mapping[str, float],
        params: Mapping[str, float],
        T: int,
    ) -> dict[str, np.ndarray]:
        if self.nonlinear_fn is not None:
            return dict(self.nonlinear_fn(paths, steady_state, T))
        jacs = self.jacobian(steady_state, params, T)
        out_paths: dict[str, np.ndarray] = {}
        for o in self.outputs:
            y_path = np.full(T, float(steady_state[o]), dtype=np.float64)
            for i in self.inputs:
                dx = np.asarray(paths[i], dtype=np.float64) - float(steady_state[i])
                y_path = y_path + jacs[o][i] @ dx
            out_paths[o] = y_path
        return out_paths


@dataclass(frozen=True)
class SolvedBlock:
    """Implicit sequence-space block resolving local unknowns $u$ from residuals $h(u, x) = 0$."""

    name: str
    unknowns: tuple[str, ...]
    targets: tuple[str, ...]
    inner_block: SimpleBlock

    def __init__(
        self,
        name: str,
        unknowns: Sequence[str],
        targets: Sequence[str],
        inner_block: SimpleBlock,
    ) -> None:
        u_tup = tuple(str(u) for u in unknowns)
        t_tup = tuple(str(t) for t in targets)
        if len(u_tup) != len(t_tup):
            raise ValueError("SolvedBlock requires equal numbers of unknowns and targets")
        object.__setattr__(self, "name", str(name))
        object.__setattr__(self, "unknowns", u_tup)
        object.__setattr__(self, "targets", t_tup)
        object.__setattr__(self, "inner_block", inner_block)

    @property
    def inputs(self) -> tuple[str, ...]:
        return tuple(x for x in self.inner_block.inputs if x not in self.unknowns)

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.unknowns

    def jacobian(
        self,
        steady_state: Mapping[str, float],
        params: Mapping[str, float],
        T: int,
    ) -> dict[str, dict[str, np.ndarray]]:
        inner_J = self.inner_block.jacobian(steady_state, params, T)
        n_u = len(self.unknowns)
        H_u = np.zeros((n_u * T, n_u * T), dtype=np.float64)
        for r, t_name in enumerate(self.targets):
            for c, u_name in enumerate(self.unknowns):
                H_u[r * T : (r + 1) * T, c * T : (c + 1) * T] = inner_J[t_name][u_name]

        lu, piv = scipy.linalg.lu_factor(H_u)
        out: dict[str, dict[str, np.ndarray]] = {u: {} for u in self.unknowns}
        for inp in self.inputs:
            H_x = np.zeros((n_u * T, T), dtype=np.float64)
            for r, t_name in enumerate(self.targets):
                H_x[r * T : (r + 1) * T, :] = inner_J[t_name][inp]
            G_x = -scipy.linalg.lu_solve((lu, piv), H_x)
            for c, u_name in enumerate(self.unknowns):
                out[u_name][inp] = G_x[c * T : (c + 1) * T, :]
        return out


@dataclass(frozen=True)
class SSJDAGResult:
    """Result container for a solved Sequence-Space Jacobian DAG transition or IRF."""

    paths: pd.DataFrame
    ge_jacobians: dict[str, dict[str, np.ndarray]]
    H_U_condition: float
    residual_norm: float
    iterations: int
    converged: bool
    mode: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        rows = [
            {"Metric": "Solver Mode", "Value": self.mode},
            {"Metric": "Horizon T", "Value": str(len(self.paths))},
            {"Metric": "Variables", "Value": ", ".join(self.paths.columns)},
            {"Metric": "Condition Number cond(H_U)", "Value": f"{self.H_U_condition:.4e}"},
            {"Metric": "Max Target Residual ||H||_inf", "Value": f"{self.residual_norm:.4e}"},
            {"Metric": "Iterations", "Value": str(self.iterations)},
            {"Metric": "Converged", "Value": str(self.converged)},
        ]
        return pd.DataFrame(rows).set_index("Metric")

    def to_dataframe(self) -> pd.DataFrame:
        return self.paths.copy()

    def to_markdown(self, **kwargs: Any) -> str:
        return _df_to_markdown(self.paths, **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        return _df_to_latex(self.paths, **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        return _df_to_typst(self.paths, **kwargs)

    def plot(
        self,
        variables: Sequence[str] | None = None,
        figsize: tuple[float, float] = (10.0, 4.5),
    ) -> matplotlib.figure.Figure:
        import matplotlib.pyplot as plt

        cols = list(variables) if variables is not None else list(self.paths.columns[:4])
        fig, axes = plt.subplots(1, len(cols), figsize=figsize, constrained_layout=True)
        if len(cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, cols):
            ax.plot(self.paths.index, self.paths[col], lw=2.0)
            ax.axhline(0.0, color="gray", ls="--", lw=0.8)
            ax.set_title(col)
            ax.set_xlabel("Period t")
        return fig


class BlockDAG:
    """Directed Acyclic Graph of sequence-space blocks (`SimpleBlock`, `HetBlock`, `SolvedBlock`).

    Parameters
    ----------
    blocks : Sequence[SimpleBlock | HetBlock | SolvedBlock]
        Blocks in the model (automatically sorted topologically).
    unknowns : Sequence[str]
        Aggregate unknown variable names $U$.
    exogenous : Sequence[str]
        Aggregate exogenous shock names $Z$.
    targets : Sequence[str]
        Target residual output names $H(U, Z) = 0$ (must match `len(unknowns)`).
    """

    def __init__(
        self,
        blocks: Sequence[SimpleBlock | HetBlock | SolvedBlock],
        unknowns: Sequence[str],
        exogenous: Sequence[str],
        targets: Sequence[str],
    ) -> None:
        self.unknowns = tuple(str(u) for u in unknowns)
        self.exogenous = tuple(str(z) for z in exogenous)
        self.targets = tuple(str(t) for t in targets)
        if len(self.unknowns) != len(self.targets):
            raise ValueError(
                f"BlockDAG requires equal numbers of unknowns ({len(self.unknowns)}) and targets ({len(self.targets)})"
            )
        self.blocks = self._topological_sort(list(blocks))

    @staticmethod
    def _topological_sort(
        blocks: list[SimpleBlock | HetBlock | SolvedBlock],
    ) -> tuple[SimpleBlock | HetBlock | SolvedBlock, ...]:
        producer: dict[str, int] = {}
        for idx, blk in enumerate(blocks):
            for out in blk.outputs:
                if out in producer:
                    raise ValueError(f"Duplicate block output variable {out!r}")
                producer[out] = idx

        n = len(blocks)
        deps: dict[int, set[int]] = {i: set() for i in range(n)}
        for i, blk in enumerate(blocks):
            for inp in blk.inputs:
                if inp in producer and producer[inp] != i:
                    deps[i].add(producer[inp])

        ordered: list[SimpleBlock | HetBlock | SolvedBlock] = []
        visited: set[int] = set()
        visiting: set[int] = set()

        def _visit(node: int) -> None:
            if node in visited:
                return
            if node in visiting:
                raise ValueError(f"Cycle detected in BlockDAG involving block {blocks[node].name!r}")
            visiting.add(node)
            for d in sorted(deps[node]):
                _visit(d)
            visiting.remove(node)
            visited.add(node)
            ordered.append(blocks[node])

        for i in range(n):
            _visit(i)
        return tuple(ordered)

    def forward_jacobians(
        self,
        steady_state: Mapping[str, float],
        params: Mapping[str, float],
        T: int,
    ) -> dict[str, dict[str, np.ndarray]]:
        """Accumulate total Jacobians $dY / d(U, Z)$ along the topological DAG."""
        seeds = list(self.unknowns) + list(self.exogenous)
        eye = np.eye(T, dtype=np.float64)
        total_J: dict[str, dict[str, np.ndarray]] = {
            s: {s2: (eye.copy() if s == s2 else np.zeros((T, T), dtype=np.float64)) for s2 in seeds}
            for s in seeds
        }

        for blk in self.blocks:
            blk_J = blk.jacobian(steady_state, params, T)
            for out in blk.outputs:
                total_J[out] = {
                    s: np.zeros((T, T), dtype=np.float64) for s in seeds
                }
                for inp in blk.inputs:
                    J_oi = blk_J[out][inp]
                    if inp in total_J:
                        for s in seeds:
                            J_is = total_J[inp][s]
                            if np.any(J_is != 0.0):
                                total_J[out][s] = total_J[out][s] + J_oi @ J_is

        return total_J

    def ge_jacobians(
        self,
        steady_state: Mapping[str, float],
        params: Mapping[str, float],
        T: int,
    ) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray, float]:
        """Compute general-equilibrium Sequence-Space Jacobians $G^{Y, Z}$ for all variables $Y$."""
        total_J = self.forward_jacobians(steady_state, params, T)
        n_u = len(self.unknowns)
        n_z = len(self.exogenous)

        H_U = np.zeros((n_u * T, n_u * T), dtype=np.float64)
        H_Z = np.zeros((n_u * T, n_z * T), dtype=np.float64)

        for r, t_name in enumerate(self.targets):
            for c, u_name in enumerate(self.unknowns):
                H_U[r * T : (r + 1) * T, c * T : (c + 1) * T] = total_J[t_name][u_name]
            for c_z, z_name in enumerate(self.exogenous):
                H_Z[r * T : (r + 1) * T, c_z * T : (c_z + 1) * T] = total_J[t_name][z_name]

        cond_H_U = float(np.linalg.cond(H_U))
        lu, piv = scipy.linalg.lu_factor(H_U)
        G_U_flat = -scipy.linalg.lu_solve((lu, piv), H_Z)

        G_all: dict[str, dict[str, np.ndarray]] = {}
        for c, u_name in enumerate(self.unknowns):
            G_all[u_name] = {}
            for c_z, z_name in enumerate(self.exogenous):
                G_all[u_name][z_name] = G_U_flat[c * T : (c + 1) * T, c_z * T : (c_z + 1) * T]

        for var_name, var_map in total_J.items():
            if var_name in self.unknowns or var_name in self.exogenous:
                continue
            G_all[var_name] = {}
            for z_name in self.exogenous:
                mat = var_map[z_name].copy()
                for u_name in self.unknowns:
                    J_vu = var_map[u_name]
                    if np.any(J_vu != 0.0):
                        mat = mat + J_vu @ G_all[u_name][z_name]
                G_all[var_name][z_name] = mat

        return G_all, H_U, cond_H_U

    def impulse_responses(
        self,
        steady_state: Mapping[str, float],
        params: Mapping[str, float],
        shocks: Mapping[str, np.ndarray],
        *,
        T: int = 100,
        levels: bool = False,
    ) -> SSJDAGResult:
        """Compute linear Sequence-Space impulse responses to arbitrary shock paths `dz`."""
        G_all, H_U, cond_H_U = self.ge_jacobians(steady_state, params, T)

        dz_map: dict[str, np.ndarray] = {}
        for z in self.exogenous:
            if z in shocks:
                arr = np.asarray(shocks[z], dtype=np.float64).ravel()
                pad = np.zeros(T, dtype=np.float64)
                pad[: min(T, len(arr))] = arr[: min(T, len(arr))]
                dz_map[z] = pad
            else:
                dz_map[z] = np.zeros(T, dtype=np.float64)

        data: dict[str, np.ndarray] = {}
        for z in self.exogenous:
            base = float(steady_state.get(z, 0.0)) if levels else 0.0
            data[z] = base + dz_map[z]

        for var_name, z_dict in G_all.items():
            resp = np.zeros(T, dtype=np.float64)
            for z_name, G_vz in z_dict.items():
                if np.any(dz_map[z_name] != 0.0):
                    resp = resp + G_vz @ dz_map[z_name]
            base = float(steady_state.get(var_name, 0.0)) if levels else 0.0
            data[var_name] = base + resp

        res_norm = float(
            max(np.max(np.abs(data[t] - (float(steady_state.get(t, 0.0)) if levels else 0.0))) for t in self.targets)
        )
        df = pd.DataFrame(data, index=pd.RangeIndex(0, T, name="t"))
        return SSJDAGResult(
            paths=df,
            ge_jacobians=G_all,
            H_U_condition=cond_H_U,
            residual_norm=res_norm,
            iterations=1,
            converged=True,
            mode="linear_ssj",
        )

    def solve_nonlinear_transition(
        self,
        steady_state: Mapping[str, float],
        params: Mapping[str, float],
        shocks: Mapping[str, np.ndarray],
        *,
        T: int = 100,
        tol: float = 1e-10,
        max_iter: int = 50,
    ) -> SSJDAGResult:
        """Solve nonlinear transition paths along the DAG using $H_U^{-1}$ quasi-Newton iteration."""
        G_all, H_U, cond_H_U = self.ge_jacobians(steady_state, params, T)
        lu, piv = scipy.linalg.lu_factor(H_U)
        n_u = len(self.unknowns)

        paths: dict[str, np.ndarray] = {
            k: np.full(T, float(v), dtype=np.float64) for k, v in steady_state.items()
        }
        # Seed initial guess from linear SSJ IRF
        lin_res = self.impulse_responses(steady_state, params, shocks, T=T, levels=True)
        for col in lin_res.paths.columns:
            paths[col] = lin_res.paths[col].to_numpy().copy()

        u_vec = np.concatenate([paths[u] for u in self.unknowns])

        def _eval_dag(u_flat: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
            cur = dict(paths)
            for c, u_name in enumerate(self.unknowns):
                cur[u_name] = u_flat[c * T : (c + 1) * T]
            for blk in self.blocks:
                if isinstance(blk, (SimpleBlock, HetBlock)):
                    cur.update(blk.evaluate(cur, steady_state, params, T))
            h_flat = np.concatenate([cur[t] for t in self.targets])
            return h_flat, cur

        h_vec, cur_paths = _eval_dag(u_vec)
        res_norm = float(np.max(np.abs(h_vec)))
        it = 0
        converged = res_norm <= tol

        while not converged and it < max_iter:
            it += 1
            du = -scipy.linalg.lu_solve((lu, piv), h_vec)
            step = 1.0
            accepted = False
            while step >= 1e-6:
                u_trial = u_vec + step * du
                try:
                    h_trial, cur_trial = _eval_dag(u_trial)
                    norm_trial = float(np.max(np.abs(h_trial)))
                    if np.isfinite(norm_trial) and norm_trial < res_norm:
                        u_vec = u_trial
                        h_vec = h_trial
                        cur_paths = cur_trial
                        res_norm = norm_trial
                        accepted = True
                        break
                except Exception:
                    pass
                step *= 0.5
            if not accepted:
                break
            if res_norm <= tol:
                converged = True

        df = pd.DataFrame(cur_paths, index=pd.RangeIndex(0, T, name="t"))
        return SSJDAGResult(
            paths=df,
            ge_jacobians=G_all,
            H_U_condition=cond_H_U,
            residual_norm=res_norm,
            iterations=it,
            converged=converged,
            mode="nonlinear_dag_newton",
        )

    def moving_average_matrices(
        self,
        steady_state: Mapping[str, float],
        params: Mapping[str, float],
        *,
        observables: Sequence[str] | None = None,
        T: int = 128,
    ) -> np.ndarray:
        """Extract sequence-space Wold moving-average array `M` of shape `(T, n_obs, n_shocks)`.

        Uses column `0` (date-0 unanticipated shock impulse response) of each
        general-equilibrium Jacobian $G^{Y, Z}_{:, 0}$.
        """
        G_all, _, _ = self.ge_jacobians(steady_state, params, T)
        obs = tuple(observables) if observables is not None else self.unknowns
        n_obs = len(obs)
        n_shocks = len(self.exogenous)
        M = np.zeros((T, n_obs, n_shocks), dtype=np.float64)
        for i, y_name in enumerate(obs):
            for j, z_name in enumerate(self.exogenous):
                if y_name in G_all and z_name in G_all[y_name]:
                    M[:, i, j] = G_all[y_name][z_name][:, 0]
                elif y_name == z_name:
                    M[0, i, j] = 1.0
        return M
