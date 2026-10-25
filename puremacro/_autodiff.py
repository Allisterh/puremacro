r"""Pure-NumPy vectorized forward-mode automatic differentiation (`DualArray`).

Provides machine-precision first- and second-order analytical derivatives
(`value_and_grad`, `jacobian_forward`, `hessian_forward`, `dual_solve`,
`dual_sylvester_solve`) under the strict four-package Pyodide runtime contract
(`numpy`, `scipy`, `pandas`, `matplotlib`).

Mathematical Foundations
------------------------
1. Vectorized First-Order Tangents (`DualArray`):
   Represents $x(\theta) = \text{val} + \text{der} \cdot d\theta$ where
   $\text{val} \in \mathbb{R}^{S}$ and $\text{der} \in \mathbb{R}^{S \times P}$.
   All $P$ directional derivatives propagate in a single vectorized NumPy pass
   via `__array_ufunc__` and `__array_function__`.

2. Exact Second-Order Hyper-Dual Numbers (`HyperDualArray`):
   Following Fike & Alonso (2011), represents
   $x = f + f_1 \epsilon_1 + f_2 \epsilon_2 + f_{12} \epsilon_1 \epsilon_2$
   with $\epsilon_1^2 = \epsilon_2^2 = 0$, $\epsilon_1 \epsilon_2 \neq 0$,
   yielding exact second derivatives $f_{12}$ free of finite-difference
   truncation or subtractive cancellation error.

3. Implicit Linear & Sylvester Tangent Propagation:
   - Linear system $A(\theta) x(\theta) = b(\theta)$:
     $$dx = A^{-1}\bigl(db - (dA)\,x\bigr)$$
   - Generalized Sylvester equation $A(\theta) X(\theta) + B(\theta) X(\theta) C(\theta) = D(\theta)$:
     $$A (dX) + B (dX) C = dD - (dA)\,X - (dB)\,X\,C - B\,X\,(dC)$$
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import scipy.linalg

__all__ = [
    "DualArray",
    "HyperDualArray",
    "seed_dual_vector",
    "value_and_grad",
    "jacobian_forward",
    "hessian_forward",
    "dual_solve",
    "dual_sylvester_solve",
]


def _broadcast_val_der(
    val: np.ndarray,
    der: np.ndarray,
    target_shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Broadcast `val` to `target_shape` and `der` to `target_shape + (P,)`."""
    P = der.shape[-1]
    val_b = np.broadcast_to(val, target_shape)
    der_b = np.broadcast_to(der, target_shape + (P,))
    return val_b, der_b


def _has_dual(x: Any) -> bool:
    if hasattr(x, "_base"):
        x = x._base
    if isinstance(x, DualArray):
        return True
    if isinstance(x, np.ndarray) and x.dtype == object:
        return any(isinstance(el, DualArray) for el in x.flat)
    if isinstance(x, (list, tuple)):
        return any(_has_dual(el) for el in x)
    return False


def _infer_n_params(x: Any) -> int | None:
    if hasattr(x, "_base"):
        x = x._base
    if isinstance(x, DualArray):
        return x.n_params
    if isinstance(x, np.ndarray) and x.dtype == object:
        for el in x.flat:
            if isinstance(el, DualArray):
                return el.n_params
    if isinstance(x, (list, tuple)):
        for el in x:
            n = _infer_n_params(el)
            if n is not None:
                return n
    return None


def _coerce_dual(
    x: Any,
    n_params: int,
) -> DualArray:
    if hasattr(x, "_base"):
        x = x._base
    if isinstance(x, DualArray):
        if x.n_params != n_params:
            raise ValueError(
                f"Incompatible DualArray tangent dimensions: {x.n_params} vs {n_params}"
            )
        return x
    if isinstance(x, np.ndarray) and x.dtype == object:
        coerced_flat = [_coerce_dual(el, n_params) for el in x.flat]
        elem_shape = coerced_flat[0].shape if coerced_flat else ()
        vals = np.array([c.val for c in coerced_flat], dtype=np.float64).reshape(x.shape + elem_shape)
        ders = np.array([c.der for c in coerced_flat], dtype=np.float64).reshape(x.shape + elem_shape + (n_params,))
        return DualArray(vals, ders)
    if isinstance(x, (list, tuple)) and _has_dual(x):
        return _coerce_dual(np.asarray(x, dtype=object), n_params)
    arr = np.asarray(x, dtype=np.float64)
    return DualArray(arr, np.zeros(arr.shape + (n_params,), dtype=np.float64))


class DualArray:
    """Vectorized forward-mode dual array holding primal `val` and tangent `der`.

    Parameters
    ----------
    val : array_like
        Primal value array of shape `S`.
    der : array_like
        Tangent array of shape `S + (P,)` where `P` is the number of parameters.
    """

    __array_priority__ = 1000.0

    def __init__(self, val: Any, der: Any) -> None:
        v = np.asarray(val, dtype=np.float64)
        d = np.asarray(der, dtype=np.float64)
        if d.ndim != v.ndim + 1:
            raise ValueError(
                f"DualArray `der` must have shape `val.shape + (P,)`, got val.shape={v.shape} and der.shape={d.shape}"
            )
        if d.shape[:-1] != v.shape:
            v, d = _broadcast_val_der(v, d, np.broadcast_shapes(v.shape, d.shape[:-1]))
        self.val: np.ndarray = v
        self.der: np.ndarray = d

    @property
    def shape(self) -> tuple[int, ...]:
        return self.val.shape

    @property
    def ndim(self) -> int:
        return self.val.ndim

    @property
    def size(self) -> int:
        return self.val.size

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.val.dtype

    @property
    def n_params(self) -> int:
        return int(self.der.shape[-1])

    @property
    def T(self) -> DualArray:
        return self.transpose()

    def copy(self) -> DualArray:
        return DualArray(self.val.copy(), self.der.copy())

    def __len__(self) -> int:
        return len(self.val)

    def __iter__(self):
        for i in range(len(self.val)):
            yield self[i]

    def __repr__(self) -> str:
        return f"DualArray(val={self.val!r}, der={self.der!r})"

    def __getitem__(self, idx: Any) -> DualArray:
        if isinstance(idx, tuple):
            return DualArray(self.val[idx], self.der[idx + (slice(None),)])
        return DualArray(self.val[idx], self.der[idx, ...])

    def __setitem__(self, idx: Any, value: Any) -> None:
        v_dual = _coerce_dual(value, self.n_params)
        if isinstance(idx, tuple):
            self.val[idx] = v_dual.val
            self.der[idx + (slice(None),)] = v_dual.der
        else:
            self.val[idx] = v_dual.val
            self.der[idx, ...] = v_dual.der

    def reshape(self, *shape: Any) -> DualArray:
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            new_shape = tuple(shape[0])
        else:
            new_shape = tuple(shape)
        v_new = self.val.reshape(new_shape)
        d_new = self.der.reshape(v_new.shape + (self.n_params,))
        return DualArray(v_new, d_new)

    def ravel(self) -> DualArray:
        return self.reshape(-1)

    def transpose(self, *axes: int) -> DualArray:
        if not axes:
            perm = tuple(reversed(range(self.ndim)))
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            perm = tuple(axes[0])
        else:
            perm = tuple(axes)
        v_t = self.val.transpose(perm)
        d_t = self.der.transpose(perm + (self.ndim,))
        return DualArray(v_t, d_t)

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> DualArray:
        if keepdims:
            v_s = np.asarray(self.val.sum(axis=axis, keepdims=True), dtype=np.float64)
        else:
            v_s = np.asarray(self.val.sum(axis=axis), dtype=np.float64)
        if axis is None:
            if keepdims:
                d_s = np.asarray(self.der.sum(axis=tuple(range(self.ndim)), keepdims=True), dtype=np.float64)
            else:
                d_s = np.asarray(self.der.reshape(-1, self.n_params).sum(axis=0), dtype=np.float64)
        else:
            ax = (axis,) if isinstance(axis, int) else tuple(axis)
            ax_norm = tuple(a if a >= 0 else self.ndim + a for a in ax)
            if keepdims:
                d_s = np.asarray(self.der.sum(axis=ax_norm, keepdims=True), dtype=np.float64)
            else:
                d_s = np.asarray(self.der.sum(axis=ax_norm), dtype=np.float64)
        return DualArray(v_s, d_s)

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> DualArray:
        s = self.sum(axis=axis, keepdims=keepdims)
        count = self.val.size / max(1, s.val.size) if not keepdims else (
            self.val.size if axis is None else np.prod([self.val.shape[a] for a in ((axis,) if isinstance(axis, int) else axis)])
        )
        return s / float(count)

    # Arithmetic and comparison operators
    def __neg__(self) -> DualArray:
        return DualArray(-self.val, -self.der)

    def __pos__(self) -> DualArray:
        return self

    def __abs__(self) -> DualArray:
        sign = np.where(self.val >= 0.0, 1.0, -1.0)
        return DualArray(np.abs(self.val), sign[..., None] * self.der)

    def __add__(self, other: Any) -> DualArray:
        return self.__array_ufunc__(np.add, "__call__", self, other)

    def __radd__(self, other: Any) -> DualArray:
        return self.__array_ufunc__(np.add, "__call__", other, self)

    def __sub__(self, other: Any) -> DualArray:
        return self.__array_ufunc__(np.subtract, "__call__", self, other)

    def __rsub__(self, other: Any) -> DualArray:
        return self.__array_ufunc__(np.subtract, "__call__", other, self)

    def __mul__(self, other: Any) -> DualArray:
        return self.__array_ufunc__(np.multiply, "__call__", self, other)

    def __rmul__(self, other: Any) -> DualArray:
        return self.__array_ufunc__(np.multiply, "__call__", other, self)

    def __truediv__(self, other: Any) -> DualArray:
        return self.__array_ufunc__(np.true_divide, "__call__", self, other)

    def __rtruediv__(self, other: Any) -> DualArray:
        return self.__array_ufunc__(np.true_divide, "__call__", other, self)

    def __pow__(self, other: Any) -> DualArray:
        return self.__array_ufunc__(np.power, "__call__", self, other)

    def __rpow__(self, other: Any) -> DualArray:
        return self.__array_ufunc__(np.power, "__call__", other, self)

    def __matmul__(self, other: Any) -> DualArray:
        return _dual_matmul(self, other)

    def __rmatmul__(self, other: Any) -> DualArray:
        return _dual_matmul(other, self)

    def __lt__(self, other: Any) -> np.ndarray:
        other_val = other.val if isinstance(other, DualArray) else np.asarray(other)
        return np.asarray(self.val < other_val)

    def __le__(self, other: Any) -> np.ndarray:
        other_val = other.val if isinstance(other, DualArray) else np.asarray(other)
        return np.asarray(self.val <= other_val)

    def __gt__(self, other: Any) -> np.ndarray:
        other_val = other.val if isinstance(other, DualArray) else np.asarray(other)
        return np.asarray(self.val > other_val)

    def __ge__(self, other: Any) -> np.ndarray:
        other_val = other.val if isinstance(other, DualArray) else np.asarray(other)
        return np.asarray(self.val >= other_val)

    def __eq__(self, other: object) -> Any:
        other_val = other.val if isinstance(other, DualArray) else np.asarray(other)
        return np.asarray(self.val == other_val)

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        if method != "__call__":
            return NotImplemented
        duals = [x for x in inputs if isinstance(x, DualArray)]
        if not duals:
            return NotImplemented
        n_params = duals[0].n_params

        if len(inputs) == 1:
            x = inputs[0]
            assert isinstance(x, DualArray)
            v, d = x.val, x.der
            if ufunc is np.negative:
                return DualArray(-v, -d)
            if ufunc is np.positive:
                return x
            if ufunc in (np.absolute, np.abs, np.fabs):
                sgn = np.where(v >= 0.0, 1.0, -1.0)
                return DualArray(np.abs(v), sgn[..., None] * d)
            if ufunc is np.square:
                return DualArray(v * v, (2.0 * v)[..., None] * d)
            if ufunc is np.sqrt:
                s = np.sqrt(v)
                return DualArray(s, (0.5 / s)[..., None] * d)
            if ufunc is np.exp:
                ev = np.exp(v)
                return DualArray(ev, ev[..., None] * d)
            if ufunc is np.expm1:
                ev = np.exp(v)
                return DualArray(np.expm1(v), ev[..., None] * d)
            if ufunc is np.log:
                return DualArray(np.log(v), (1.0 / v)[..., None] * d)
            if ufunc is np.log1p:
                return DualArray(np.log1p(v), (1.0 / (1.0 + v))[..., None] * d)
            if ufunc is np.sin:
                return DualArray(np.sin(v), np.cos(v)[..., None] * d)
            if ufunc is np.cos:
                return DualArray(np.cos(v), (-np.sin(v))[..., None] * d)
            if ufunc is np.tan:
                c = np.cos(v)
                return DualArray(np.tan(v), (1.0 / (c * c))[..., None] * d)
            if ufunc is np.tanh:
                th = np.tanh(v)
                return DualArray(th, (1.0 - th * th)[..., None] * d)
            if ufunc is np.sinh:
                return DualArray(np.sinh(v), np.cosh(v)[..., None] * d)
            if ufunc is np.cosh:
                return DualArray(np.cosh(v), np.sinh(v)[..., None] * d)
            if ufunc is np.arctan:
                return DualArray(np.arctan(v), (1.0 / (1.0 + v * v))[..., None] * d)
            if ufunc is np.arcsin:
                return DualArray(np.arcsin(v), (1.0 / np.sqrt(1.0 - v * v))[..., None] * d)
            if ufunc is np.arccos:
                return DualArray(np.arccos(v), (-1.0 / np.sqrt(1.0 - v * v))[..., None] * d)
            if ufunc is np.isfinite:
                return np.isfinite(v) & np.all(np.isfinite(d), axis=-1)
            if ufunc is np.isnan:
                return np.isnan(v) | np.any(np.isnan(d), axis=-1)
            return NotImplemented

        if len(inputs) == 2:
            if ufunc is np.matmul:
                return _dual_matmul(inputs[0], inputs[1])
            a = _coerce_dual(inputs[0], n_params)
            b = _coerce_dual(inputs[1], n_params)
            out_shape = np.broadcast_shapes(a.shape, b.shape)
            av, ad = _broadcast_val_der(a.val, a.der, out_shape)
            bv, bd = _broadcast_val_der(b.val, b.der, out_shape)

            if ufunc is np.add:
                return DualArray(av + bv, ad + bd)
            if ufunc is np.subtract:
                return DualArray(av - bv, ad - bd)
            if ufunc is np.multiply:
                return DualArray(av * bv, bv[..., None] * ad + av[..., None] * bd)
            if ufunc in (np.true_divide, np.divide):
                val_out = av / bv
                der_out = (ad * bv[..., None] - av[..., None] * bd) / (bv * bv)[..., None]
                return DualArray(val_out, der_out)
            if ufunc is np.power:
                val_out = np.power(av, bv)
                # d(a^b) = b * a^(b-1) da + a^b * ln(a) db
                term_a = np.where(av == 0.0, 0.0, bv * np.power(av, bv - 1.0))[..., None] * ad
                if np.any(bd != 0.0):
                    safe_a = np.where(av > 0.0, av, 1.0)
                    term_b = (val_out * np.log(safe_a))[..., None] * bd
                    der_out = term_a + term_b
                else:
                    der_out = term_a
                return DualArray(val_out, der_out)
            if ufunc is np.maximum:
                mask = (av >= bv)[..., None]
                return DualArray(np.maximum(av, bv), np.where(mask, ad, bd))
            if ufunc is np.minimum:
                mask = (av <= bv)[..., None]
                return DualArray(np.minimum(av, bv), np.where(mask, ad, bd))
            if ufunc in (np.less, np.less_equal, np.greater, np.greater_equal, np.equal, np.not_equal):
                return ufunc(av, bv)
            return NotImplemented

        return NotImplemented

    def __array_function__(self, func: Any, types: Any, args: Any, kwargs: Any) -> Any:
        if func is np.sum:
            return self.sum(*args[1:], **kwargs)
        if func is np.mean:
            return self.mean(*args[1:], **kwargs)
        if func is np.reshape:
            return args[0].reshape(*args[1:], **kwargs)
        if func is np.transpose:
            return args[0].transpose(*args[1:], **kwargs)
        if func is np.dot:
            return _dual_matmul(args[0], args[1])
        if func is np.matmul:
            return _dual_matmul(args[0], args[1])
        if func is np.where:
            cond, x, y = args
            duals = [z for z in (x, y) if isinstance(z, DualArray)]
            if not duals:
                return np.where(cond, x, y)
            n_params = duals[0].n_params
            xd = _coerce_dual(x, n_params)
            yd = _coerce_dual(y, n_params)
            c_arr = np.asarray(cond.val if isinstance(cond, DualArray) else cond, dtype=bool)
            out_shape = np.broadcast_shapes(c_arr.shape, xd.shape, yd.shape)
            xv, xder = _broadcast_val_der(xd.val, xd.der, out_shape)
            yv, yder = _broadcast_val_der(yd.val, yd.der, out_shape)
            c_b = np.broadcast_to(c_arr, out_shape)
            return DualArray(np.where(c_b, xv, yv), np.where(c_b[..., None], xder, yder))
        if func is np.clip:
            x, a_min, a_max = args[0], args[1], args[2]
            res = x
            if a_min is not None:
                res = np.maximum(res, a_min)
            if a_max is not None:
                res = np.minimum(res, a_max)
            return res
        if func is np.concatenate:
            seq = args[0]
            axis = kwargs.get("axis", args[1] if len(args) > 1 else 0)
            duals = [z for z in seq if isinstance(z, DualArray)]
            n_params = duals[0].n_params
            coerced = [_coerce_dual(z, n_params) for z in seq]
            v_cat = np.concatenate([z.val for z in coerced], axis=axis)
            ax_d = axis if axis >= 0 else coerced[0].ndim + axis
            d_cat = np.concatenate([z.der for z in coerced], axis=ax_d)
            return DualArray(v_cat, d_cat)
        if func is np.stack:
            seq = args[0]
            axis = kwargs.get("axis", args[1] if len(args) > 1 else 0)
            duals = [z for z in seq if isinstance(z, DualArray)]
            n_params = duals[0].n_params
            coerced = [_coerce_dual(z, n_params) for z in seq]
            v_stk = np.stack([z.val for z in coerced], axis=axis)
            ax_d = axis if axis >= 0 else v_stk.ndim + axis
            d_stk = np.stack([z.der for z in coerced], axis=ax_d)
            return DualArray(v_stk, d_stk)
        return NotImplemented


def _dual_matmul(a: Any, b: Any) -> DualArray:
    duals = [x for x in (a, b) if isinstance(x, DualArray)]
    n_params = duals[0].n_params
    ad = _coerce_dual(a, n_params)
    bd = _coerce_dual(b, n_params)
    av, ader = ad.val, ad.der
    bv, bder = bd.val, bd.der

    if av.ndim == 1 and bv.ndim == 1:
        val_out = np.dot(av, bv)
        der_out = np.einsum("i,ip->p", bv, ader) + np.einsum("i,ip->p", av, bder)
        return DualArray(val_out, der_out)
    if av.ndim == 2 and bv.ndim == 1:
        val_out = av @ bv
        der_out = np.einsum("ijp,j->ip", ader, bv) + np.einsum("ij,jp->ip", av, bder)
        return DualArray(val_out, der_out)
    if av.ndim == 1 and bv.ndim == 2:
        val_out = av @ bv
        der_out = np.einsum("ip,ij->jp", ader, bv) + np.einsum("i,ijp->jp", av, bder)
        return DualArray(val_out, der_out)
    if av.ndim == 2 and bv.ndim == 2:
        val_out = av @ bv
        der_out = np.einsum("ikp,kj->ijp", ader, bv) + np.einsum("ik,kjp->ijp", av, bder)
        return DualArray(val_out, der_out)
    raise ValueError(f"Unsupported DualArray matmul dimensions: {av.shape} @ {bv.shape}")


@dataclass(frozen=True)
class HyperDualArray:
    """Exact second-order hyper-dual number $f + f_1 \\epsilon_1 + f_2 \\epsilon_2 + f_{12} \\epsilon_1 \\epsilon_2$."""

    val: np.ndarray
    d1: np.ndarray
    d2: np.ndarray
    d12: np.ndarray

    __array_priority__ = 1000.0

    @staticmethod
    def coerce(x: Any) -> HyperDualArray:
        if isinstance(x, HyperDualArray):
            return x
        v = np.asarray(x, dtype=np.float64)
        z = np.zeros_like(v)
        return HyperDualArray(v, z, z.copy(), z.copy())

    def __neg__(self) -> HyperDualArray:
        return HyperDualArray(-self.val, -self.d1, -self.d2, -self.d12)

    def __add__(self, other: Any) -> HyperDualArray:
        o = HyperDualArray.coerce(other)
        return HyperDualArray(self.val + o.val, self.d1 + o.d1, self.d2 + o.d2, self.d12 + o.d12)

    def __radd__(self, other: Any) -> HyperDualArray:
        return self.__add__(other)

    def __sub__(self, other: Any) -> HyperDualArray:
        o = HyperDualArray.coerce(other)
        return HyperDualArray(self.val - o.val, self.d1 - o.d1, self.d2 - o.d2, self.d12 - o.d12)

    def __rsub__(self, other: Any) -> HyperDualArray:
        o = HyperDualArray.coerce(other)
        return o.__sub__(self)

    def __mul__(self, other: Any) -> HyperDualArray:
        o = HyperDualArray.coerce(other)
        return HyperDualArray(
            self.val * o.val,
            self.d1 * o.val + self.val * o.d1,
            self.d2 * o.val + self.val * o.d2,
            self.d12 * o.val + self.d1 * o.d2 + self.d2 * o.d1 + self.val * o.d12,
        )

    def __rmul__(self, other: Any) -> HyperDualArray:
        return self.__mul__(other)

    def __matmul__(self, other: Any) -> HyperDualArray:
        return _hyper_dual_matmul(self, other)

    def __rmatmul__(self, other: Any) -> HyperDualArray:
        return _hyper_dual_matmul(other, self)

    def __truediv__(self, other: Any) -> HyperDualArray:
        o = HyperDualArray.coerce(other)
        inv = o._reciprocal()
        return self * inv

    def __rtruediv__(self, other: Any) -> HyperDualArray:
        return HyperDualArray.coerce(other).__truediv__(self)

    def _reciprocal(self) -> HyperDualArray:
        v = self.val
        v2 = v * v
        v3 = v2 * v
        return HyperDualArray(
            1.0 / v,
            -self.d1 / v2,
            -self.d2 / v2,
            -self.d12 / v2 + 2.0 * self.d1 * self.d2 / v3,
        )

    def __pow__(self, power: Any) -> HyperDualArray:
        if isinstance(power, (int, float)):
            p = float(power)
            v = self.val
            f0 = np.power(v, p)
            fp = p * np.power(v, p - 1.0)
            fpp = p * (p - 1.0) * np.power(v, p - 2.0)
            return HyperDualArray(
                f0,
                fp * self.d1,
                fp * self.d2,
                fp * self.d12 + fpp * self.d1 * self.d2,
            )
        o = HyperDualArray.coerce(power)
        return (o * self._log())._exp()

    def _exp(self) -> HyperDualArray:
        ev = np.exp(self.val)
        return HyperDualArray(
            ev,
            ev * self.d1,
            ev * self.d2,
            ev * (self.d12 + self.d1 * self.d2),
        )

    def _log(self) -> HyperDualArray:
        v = self.val
        v2 = v * v
        return HyperDualArray(
            np.log(v),
            self.d1 / v,
            self.d2 / v,
            self.d12 / v - (self.d1 * self.d2) / v2,
        )

    def __getitem__(self, idx: Any) -> HyperDualArray:
        return HyperDualArray(
            np.asarray(self.val[idx]),
            np.asarray(self.d1[idx]),
            np.asarray(self.d2[idx]),
            np.asarray(self.d12[idx]),
        )

    def sum(self, axis: Any = None, **kwargs: Any) -> HyperDualArray:
        keepdims = kwargs.get("keepdims", False)
        return HyperDualArray(
            np.asarray(np.sum(self.val, axis=axis, keepdims=keepdims)),
            np.asarray(np.sum(self.d1, axis=axis, keepdims=keepdims)),
            np.asarray(np.sum(self.d2, axis=axis, keepdims=keepdims)),
            np.asarray(np.sum(self.d12, axis=axis, keepdims=keepdims)),
        )

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        if method != "__call__":
            return NotImplemented
        if len(inputs) == 1:
            x = inputs[0]
            assert isinstance(x, HyperDualArray)
            v = x.val
            if ufunc is np.exp:
                return x._exp()
            if ufunc is np.log:
                return x._log()
            if ufunc is np.sqrt:
                return x ** 0.5
            if ufunc is np.square:
                return x * x
            if ufunc is np.sin:
                sv, cv = np.sin(v), np.cos(v)
                return HyperDualArray(sv, cv * x.d1, cv * x.d2, cv * x.d12 - sv * x.d1 * x.d2)
            if ufunc is np.cos:
                sv, cv = np.sin(v), np.cos(v)
                return HyperDualArray(cv, -sv * x.d1, -sv * x.d2, -sv * x.d12 - cv * x.d1 * x.d2)
            if ufunc is np.negative:
                return -x
        if len(inputs) == 2:
            if ufunc is np.matmul:
                return _hyper_dual_matmul(inputs[0], inputs[1])
            a = HyperDualArray.coerce(inputs[0])
            b = HyperDualArray.coerce(inputs[1])
            if ufunc is np.add:
                return a + b
            if ufunc is np.subtract:
                return a - b
            if ufunc is np.multiply:
                return a * b
            if ufunc in (np.true_divide, np.divide):
                return a / b
            if ufunc is np.power:
                return a ** b
        return NotImplemented

    def __array_function__(self, func: Any, types: Any, args: Any, kwargs: Any) -> Any:
        if func is np.sum:
            return self.sum(*args[1:], **kwargs)
        if func is np.matmul:
            return _hyper_dual_matmul(args[0], args[1])
        if func is np.dot:
            return _hyper_dual_matmul(args[0], args[1])
        return NotImplemented


def _hyper_dual_matmul(a: Any, b: Any) -> HyperDualArray:
    a_hd = HyperDualArray.coerce(a)
    b_hd = HyperDualArray.coerce(b)
    v = np.asarray(a_hd.val @ b_hd.val)
    d1 = np.asarray(a_hd.d1 @ b_hd.val + a_hd.val @ b_hd.d1)
    d2 = np.asarray(a_hd.d2 @ b_hd.val + a_hd.val @ b_hd.d2)
    d12 = np.asarray(
        a_hd.d12 @ b_hd.val
        + a_hd.d1 @ b_hd.d2
        + a_hd.d2 @ b_hd.d1
        + a_hd.val @ b_hd.d12
    )
    return HyperDualArray(v, d1, d2, d12)


def seed_dual_vector(x: Sequence[float] | np.ndarray) -> DualArray:
    """Create a 1D `DualArray` seeded with the identity tangent matrix $I_P$."""
    arr = np.asarray(x, dtype=np.float64).ravel()
    P = arr.size
    return DualArray(arr, np.eye(P, dtype=np.float64))


def jacobian_forward(
    fn: Callable[[Any], Any],
    x: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate `fn(x)` and its exact forward-mode Jacobian `df/dx` in a single pass.

    Parameters
    ----------
    fn : Callable
        Differentiable function composed of NumPy operations and/or `dual_solve`.
    x : array_like of shape (P,)
        Evaluation point.

    Returns
    -------
    val : np.ndarray
        Primal output `fn(x)`.
    jac : np.ndarray
        Exact Jacobian of shape `val.shape + (P,)`.
    """
    x_dual = seed_dual_vector(x)
    out = fn(x_dual)
    if _has_dual(out):
        out_dual = _coerce_dual(out, x_dual.n_params)
        return out_dual.val.copy(), out_dual.der.copy()
    arr = np.asarray(out, dtype=np.float64)
    return arr, np.zeros(arr.shape + (x_dual.n_params,), dtype=np.float64)


def value_and_grad(
    fn: Callable[[Any], Any],
    x: Sequence[float] | np.ndarray,
) -> tuple[float, np.ndarray]:
    """Evaluate scalar objective `fn(x)` and its exact gradient vector of shape `(P,)`."""
    val, jac = jacobian_forward(fn, x)
    if val.size != 1:
        raise ValueError(f"`value_and_grad` expects a scalar output, got shape {val.shape}")
    return float(val.reshape(())), jac.reshape(-1)


def hessian_forward(
    fn: Callable[[Any], Any],
    x: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate `fn(x)`, its gradient/Jacobian, and exact second-order Hessian via `HyperDualArray`.

    Returns
    -------
    val : np.ndarray
        Primal value `fn(x)`.
    jac : np.ndarray
        First derivative of shape `val.shape + (P,)`.
    hess : np.ndarray
        Second derivative of shape `val.shape + (P, P)`.
    """
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    P = x_arr.size
    val0, jac0 = jacobian_forward(fn, x_arr)
    out_shape = val0.shape
    hess = np.zeros(out_shape + (P, P), dtype=np.float64)

    for i in range(P):
        e_i = np.zeros(P, dtype=np.float64)
        e_i[i] = 1.0
        for j in range(i, P):
            e_j = np.zeros(P, dtype=np.float64)
            e_j[j] = 1.0
            hd_x = HyperDualArray(x_arr, e_i, e_j, np.zeros(P, dtype=np.float64))
            res = fn(hd_x)
            if isinstance(res, HyperDualArray):
                d12 = np.asarray(res.d12, dtype=np.float64).reshape(out_shape)
            else:
                d12 = np.zeros(out_shape, dtype=np.float64)
            hess[..., i, j] = d12
            hess[..., j, i] = d12

    return val0, jac0, hess


def dual_solve(A: Any, b: Any) -> DualArray | np.ndarray:
    """Solve linear system $A(\\theta) x(\\theta) = b(\\theta)$ with analytical tangent propagation.

    Uses a single LU factorization of primal $A$ to solve both the primal system
    and all $P$ tangent right-hand sides $db - (dA)\\,x$.
    """
    n_params = _infer_n_params((A, b))
    if n_params is None:
        return np.linalg.solve(np.asarray(A, dtype=np.float64), np.asarray(b, dtype=np.float64))

    Ad = _coerce_dual(A, n_params)
    bd = _coerce_dual(b, n_params)

    lu, piv = scipy.linalg.lu_factor(Ad.val)
    x_val = scipy.linalg.lu_solve((lu, piv), bd.val)

    # rhs_der = bd.der - Ad.der @ x_val
    if bd.val.ndim == 1:
        # Ad.der: (N, N, P), x_val: (N,) -> (N, P)
        rhs_der = bd.der - np.einsum("ijp,j->ip", Ad.der, x_val)
        x_der = scipy.linalg.lu_solve((lu, piv), rhs_der)
    elif bd.val.ndim == 2:
        # Ad.der: (N, N, P), x_val: (N, K) -> (N, K, P)
        rhs_der = bd.der - np.einsum("ijp,jk->ikp", Ad.der, x_val)
        N, K, P = rhs_der.shape
        x_der = scipy.linalg.lu_solve((lu, piv), rhs_der.reshape(N, K * P)).reshape(N, K, P)
    else:
        raise ValueError(f"Unsupported right-hand side shape {bd.val.shape} in `dual_solve`.")

    return DualArray(x_val, x_der)


def dual_sylvester_solve(
    A: Any,
    B: Any,
    C: Any,
    D: Any,
) -> DualArray | np.ndarray:
    """Solve generalized Sylvester equation $A X + B X C = D$ with exact `DualArray` tangents.

    Propagates
        $A (dX) + B (dX) C = dD - (dA) X - (dB) X C - B X (dC)$
    across all $P$ parameter directions in a single batched Schur solve.
    """
    from puremacro.dsge._gradients import solve_sylvester_generalized

    n_params = _infer_n_params((A, B, C, D))
    if n_params is None:
        return solve_sylvester_generalized(A, B, C, D)

    Ad = _coerce_dual(A, n_params)
    Bd = _coerce_dual(B, n_params)
    Cd = _coerce_dual(C, n_params)
    Dd = _coerce_dual(D, n_params)

    X_val = solve_sylvester_generalized(Ad.val, Bd.val, Cd.val, Dd.val)
    XC = X_val @ Cd.val
    BX = Bd.val @ X_val

    # Build batched RHS of shape (P, N, M)
    # rhs_p = dD_p - dA_p @ X - dB_p @ (X @ C) - (B @ X) @ dC_p
    rhs_der = (
        np.moveaxis(Dd.der, -1, 0)
        - np.einsum("ijp,jk->pik", Ad.der, X_val)
        - np.einsum("ijp,jk->pik", Bd.der, XC)
        - np.einsum("ij,jkp->pik", BX, Cd.der)
    )
    dX_batch = solve_sylvester_generalized(Ad.val, Bd.val, Cd.val, rhs_der)
    X_der = np.moveaxis(dX_batch, 0, -1)
    return DualArray(X_val, X_der)
