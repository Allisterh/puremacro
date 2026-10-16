"""Batched macro residuals and Jacobians of the consistent-accounting trade model.

The equilibrium of :func:`puremacro.trade.solve_trade_equilibrium` is solved
here on a condensed (macro) system. Given the factor returns and the fiscal
transfers, producer prices follow from the zero-profit conditions and gross
outputs from goods-market clearing, both linear (Leontief) systems whose
operators are inverted once per tariff schedule. What remains is a macro
system of size ``B``:

* full layout ``[log r; log w; T]``, ``B = 3*nc``;
* reduced layout ``[log w; T]`` with ``r == w``, ``B = 2*nc``, exact only when
  the capital share is the same in every sector of a country (verified), since
  capital-market clearing then follows from labour-market clearing.

The macro residuals are the first producer price minus one (the numeraire),
labour-market clearing in every country but the first (implied by Walras' law
once all goods markets clear), capital-market clearing (full layout) and the
government budgets. Foreign saving is exogenous, as in the full model, in
numeraire units or as a fixed share of world factor income
(``foreign_saving_units``). A solution of the macro system therefore solves
every equation of the full consistent-accounting system (Cobb-Douglas
intermediates, ``sigma = 0``).

Key capabilities:

1. One-time inversion of the Leontief price operator ``(I - B^T)`` and output
   operator ``(I - A)`` per tariff schedule.
2. Vectorized multi-RHS products ``(B, M) @ (M, M)`` for a batch of perturbed
   macro points, on NumPy, PyTorch (CPU, CUDA, MPS) or Apple MLX.
3. Forward-mode (JVP) and reverse-mode (VJP) automatic differentiation via
   ``torch.func`` / ``mx.jvp`` / ``mx.vjp``.

Precision policy
----------------
Finite-difference Jacobians divide residual differences by a step of about
1e-4, so float32 evaluation (Apple MPS, MLX GPU stream) gives large relative
errors in the Jacobian. The evaluator therefore defaults to float64 wherever
the device supports it: PyTorch on CPU/CUDA, and the MLX CPU stream (where
``mx.exp`` is only float32-accurate and is replaced by an exact ``e ** x``). A
float32 device is used only when explicitly requested through ``stream='gpu'``.
"""
from __future__ import annotations

import math
import warnings
from typing import Any, Literal

import numpy as np
import scipy.linalg as la

from puremacro.trade._results import TradeCalibrationResult
from puremacro.trade.gpu.backend import (
    _device_info,
    _load_mlx,
    _load_torch,
    _resolve_backend_device,
    to_numpy,
)

FOREIGN_SAVING_UNITS = ("numeraire", "world_income")


def _uniform_capital_share(calib: TradeCalibrationResult, rtol: float = 1e-8, atol: float = 1e-10) -> bool:
    """True if every sector of each country has the same capital share alpha.

    The reduced macro layout imposes r_c == w_c, which is only equivalent to the
    full system (labour *and* capital market clearing) when the capital share is
    identical across the sectors of a country.
    """
    alpha = np.asarray(calib.alpha, dtype=float).reshape(calib.n_sectors, calib.n_countries)
    for c in range(calib.n_countries):
        col = alpha[:, c]
        active = col[col > 0.0]
        if active.size > 1 and not np.allclose(active, active[0], rtol=rtol, atol=atol):
            return False
    return True


def _resolve_layout(
    calib: TradeCalibrationResult,
    batch_size: int | None,
    factor_equivalence: bool | None,
) -> tuple[bool, int]:
    """Resolve the macro layout: (factor_equivalence, macro dimension).

    ``batch_size`` must equal ``2*nc`` (reduced layout) or ``3*nc`` (full
    layout) for the calibration's ``nc``; any other value is rejected rather
    than silently selecting a layout. When neither argument is given, the
    reduced layout is used iff it is exact for the calibration (uniform capital
    shares).
    """
    nc = calib.n_countries
    n_reduced, n_full = 2 * nc, 3 * nc

    if batch_size is not None:
        if batch_size == n_reduced:
            from_batch = True
        elif batch_size == n_full:
            from_batch = False
        else:
            raise ValueError(
                f"batch_size={batch_size} is inconsistent with a {nc}-country calibration: "
                f"expected {n_reduced} (reduced layout, r == w) or {n_full} (full layout)."
            )
        if factor_equivalence is not None and bool(factor_equivalence) != from_batch:
            raise ValueError(
                f"batch_size={batch_size} implies factor_equivalence={from_batch} but "
                f"factor_equivalence={factor_equivalence} was requested."
            )
        factor_equivalence = from_batch

    uniform = _uniform_capital_share(calib)
    if factor_equivalence is None:
        factor_equivalence = uniform
    elif factor_equivalence and not uniform:
        raise ValueError(
            "factor_equivalence=True (reduced layout, r == w) is invalid for this calibration: "
            "capital shares differ across sectors within a country, so imposing r == w drops "
            "the capital-market clearing condition. Use factor_equivalence=False."
        )
    return bool(factor_equivalence), (n_reduced if factor_equivalence else n_full)


class _Ops:
    """The few array operations the batched residual needs, for one backend."""

    def __init__(self, backend: str, module: Any = None, exact_exp: bool = False) -> None:
        self.backend = backend
        self.m = module
        self.exact_exp = exact_exp

    def exp(self, x: Any) -> Any:
        if self.backend == "numpy":
            return np.exp(x)
        if self.backend == "torch":
            return self.m.exp(x)
        if self.exact_exp:
            mx = self.m
            return mx.power(mx.array(math.e, dtype=mx.float64), x)
        return self.m.exp(x)

    def sum(self, x: Any, axis: int) -> Any:
        if self.backend == "torch":
            return self.m.sum(x, dim=axis)
        if self.backend == "mlx":
            return self.m.sum(x, axis=axis)
        return np.sum(x, axis=axis)

    def concat(self, parts: list[Any]) -> Any:
        if self.backend == "torch":
            return self.m.cat(parts, dim=1)
        if self.backend == "mlx":
            return self.m.concatenate(parts, axis=1)
        return np.concatenate(parts, axis=1)


class BatchedJacobianEvaluator:
    """Batched macro residuals and Jacobians of the consistent-accounting trade model.

    Parameters
    ----------
    calib : TradeCalibrationResult
        Calibrated trade model.
    tau_a : np.ndarray
        Intermediate tariff multipliers ``1 + rate``, shape ``(ns*nc, ns, nc)``
        or ``(ns, nc, ns, nc)``.
    taufd_a : np.ndarray
        Final-demand tariff multipliers, shape ``(ns*nc, nfd, nc)`` or
        ``(ns, nc, nfd, nc)``.
    device : str | None
        Target device ('auto', 'cuda', 'cuda:N', 'mps', 'mlx', 'gpu', 'cpu').
    backend : str | None
        Target backend ('torch', 'mlx', 'numpy' or None for automatic).
    batch_size : int | None, default None
        Macro dimension B; must equal ``2*nc`` (reduced layout) or ``3*nc``
        (full layout) for the calibration. Prefer ``factor_equivalence``.
    factor_equivalence : bool | None, default None
        True selects the reduced layout ``[log w; T]`` with r == w (valid only
        when capital shares are uniform across sectors within each country,
        which is verified); False selects the full layout ``[log r; log w; T]``.
        None (default) selects the reduced layout iff it is exact.
    foreign_saving_units : {"numeraire", "world_income"}, default "numeraire"
        The foreign-saving closure of :func:`puremacro.trade.solve_trade_equilibrium`.

    Notes
    -----
    Only the tensors of the selected backend are materialised; torch and mlx are
    imported lazily on first use. The model is the consistent-accounting model
    with Cobb-Douglas intermediates and the lump-sum fiscal closure.
    """

    def __init__(
        self,
        calib: TradeCalibrationResult,
        tau_a: np.ndarray,
        taufd_a: np.ndarray,
        device: str | None = None,
        backend: str | None = None,
        batch_size: int | None = None,
        factor_equivalence: bool | None = None,
        *,
        foreign_saving_units: str = "numeraire",
    ) -> None:
        from puremacro.trade._accounting import _parameters

        if foreign_saving_units not in FOREIGN_SAVING_UNITS:
            raise ValueError("foreign_saving_units must be 'numeraire' or 'world_income'")
        self.calib = calib
        self.ns = calib.n_sectors
        self.nc = calib.n_countries
        self.nfd = calib.n_final_demand
        self.M = self.ns * self.nc
        self.foreign_saving_units = foreign_saving_units
        self.factor_equivalence, self.batch_size = _resolve_layout(calib, batch_size, factor_equivalence)
        self.n_factor_vars = self.nc if self.factor_equivalence else 2 * self.nc

        self.backend, self.device = _resolve_backend_device(device, backend)
        dev_info = _device_info(self.backend, self.device)
        self.supports_float64 = dev_info.supports_float64
        self._torch = _load_torch() if self.backend == "torch" else None
        self._mx = _load_mlx() if self.backend == "mlx" else None
        if self.backend == "torch":
            self.dtype = self._torch.float32 if self.device == "mps" else self._torch.float64
        else:
            self.dtype = None

        ns, nc, nfd, M = self.ns, self.nc, self.nfd, self.M
        # Final-use tax shares and active categories of the consistent accounting
        # (raises ValueError for a calibration that accounting cannot use).
        fd_tax, basic0, inv = _parameters(calib)
        self._inv_category = inv
        self.a_2d = np.asarray(calib.a, dtype=float).reshape((M, M), order="F")
        self.inv_Y_np = la.inv(np.eye(M) - self.a_2d)

        alpha = np.asarray(calib.alpha, dtype=float).ravel(order="F")
        beta = np.asarray(calib.beta, dtype=float).ravel(order="F")
        tax = np.asarray(calib.tax, dtype=float).ravel(order="F")
        cell_country = np.repeat(np.arange(nc), ns)
        fd_country = np.repeat(np.arange(nc), nfd)
        e_cell = np.zeros((nc, M))
        e_cell[cell_country, np.arange(M)] = 1.0
        e_fd = np.zeros((nc, nfd * nc))
        e_fd[fd_country, np.arange(nfd * nc)] = 1.0
        inv_mask = np.zeros(nfd * nc)
        inv_mask[inv + nfd * np.arange(nc)] = 1.0
        active_fd = (np.asarray(basic0).ravel(order="F") > 0).astype(float)
        base_world = float(np.sum(calib.l_endow) + np.sum(calib.k_endow))
        self._np_const: dict[str, np.ndarray] = {
            "alpha": alpha,
            "beta": beta,
            "one_minus_tax": 1.0 - tax,
            "tax": tax,
            "e_cell": e_cell,
            "e_fd": e_fd,
            "l_endow": np.asarray(calib.l_endow, dtype=float).ravel(),
            "k_endow": np.asarray(calib.k_endow, dtype=float).ravel(),
            "theta": np.asarray(calib.theta, dtype=float).ravel(order="F"),
            "fd_tax": np.asarray(fd_tax, dtype=float).ravel(order="F"),
            "active_fd": active_fd,
            "saving": inv_mask * (np.asarray(calib.invforT, dtype=float).ravel() @ e_fd),
            "inv_Y_T": self.inv_Y_np.T.copy(),
            "afd_T": np.asarray(calib.afd, dtype=float).reshape((M, nfd * nc), order="F").T.copy(),
            "world_scale": np.array([1.0 / base_world]),
        }
        # Value scale of the T unknowns (median baseline factor income), for steps.
        self._value_scale = float(np.median(self._np_const["l_endow"] + self._np_const["k_endow"]))
        self.update_tariffs(tau_a, taufd_a)

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------

    def _saving(self, w: np.ndarray, r: np.ndarray) -> np.ndarray:
        """Foreign saving B_c at factor returns ``w``, ``r`` (1-D, length nc)."""
        b = np.asarray(self.calib.invforT, dtype=float).ravel()
        if self.foreign_saving_units == "numeraire":
            return b.copy()
        c = self._np_const
        world = float(np.sum(w * c["l_endow"] + r * c["k_endow"]))
        return b * world * float(c["world_scale"][0])

    def _macro_from_x0(self, x0: np.ndarray | None) -> np.ndarray:
        """Build the macro vector of this evaluator's layout from an initial guess.

        ``x0`` may be None (baseline guess: unit factor returns and the
        calibrated transfers), a full state vector of length
        ``2*ns*nc + 4*nc - 1``, or a macro vector of either layout (``2*nc`` or
        ``3*nc``); r == w is assumed when a reduced vector seeds the full layout.
        """
        nc, M = self.nc, self.M
        n_reduced, n_full, n_state = 2 * nc, 3 * nc, 2 * M + 4 * nc - 1

        if x0 is None:
            log_w = np.zeros(nc)
            log_r = np.zeros(nc)
            if self.calib.T is not None:
                T = np.asarray(self.calib.T, dtype=float).ravel()
            else:
                from puremacro.trade.solver import build_initial_guess

                T = build_initial_guess(self.calib)[2 * M + 2 * nc: 2 * M + 3 * nc]
        else:
            x0 = np.asarray(x0, dtype=float).ravel()
            if len(x0) == n_state:
                log_r = x0[2 * M: 2 * M + nc]
                log_w = x0[2 * M + nc: 2 * M + 2 * nc]
                T = x0[2 * M + 2 * nc: 2 * M + 3 * nc]
            elif len(x0) == n_full:
                log_r, log_w, T = x0[:nc], x0[nc: 2 * nc], x0[2 * nc:]
            elif len(x0) == n_reduced:
                log_w, T = x0[:nc], x0[nc:]
                log_r = log_w
            else:
                raise ValueError(
                    f"Invalid initial guess dimension {len(x0)}. Expected {n_reduced} (reduced macro "
                    f"vector), {n_full} (full macro vector) or {n_state} (full state vector)."
                )

        if self.factor_equivalence:
            return np.concatenate([log_w, T]).astype(float, copy=True)
        return np.concatenate([log_r, log_w, T]).astype(float, copy=True)

    def _split(self, xm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(log r, log w, T)`` of a macro vector of this layout."""
        nc = self.nc
        if self.factor_equivalence:
            return xm[:nc], xm[:nc], xm[nc: 2 * nc]
        return xm[:nc], xm[nc: 2 * nc], xm[2 * nc: 3 * nc]

    def _full_state(self, xm: np.ndarray, p_vec: np.ndarray, y_vec: np.ndarray) -> np.ndarray:
        """The full state vector ``[log p; log y; log r; log w; T; XN]``."""
        log_r, log_w, T = self._split(np.asarray(xm, dtype=float))
        saving = self._saving(np.exp(log_w), np.exp(log_r))
        return np.concatenate([
            np.log(np.maximum(p_vec, 1e-300)),
            np.log(np.maximum(y_vec, 1e-300)),
            log_r,
            log_w,
            T,
            saving[: self.nc - 1],
        ])

    # ------------------------------------------------------------------
    # Tariff / operator updates
    # ------------------------------------------------------------------

    def update_tariffs(self, tau_a: np.ndarray, taufd_a: np.ndarray) -> None:
        """Set the tariff schedule and invert the Leontief price operator ``(I - B^T)``."""
        ns, nc, nfd, M = self.ns, self.nc, self.nfd, self.M
        tau_a = np.asarray(tau_a, dtype=float)
        taufd_a = np.asarray(taufd_a, dtype=float)
        if tau_a.ndim == 4:
            tau_a = tau_a.transpose(1, 0, 2, 3).reshape(M, ns, nc)
        if taufd_a.ndim == 4:
            taufd_a = taufd_a.transpose(1, 0, 2, 3).reshape(M, nfd, nc)
        for schedule, shape in ((tau_a, (M, ns, nc)), (taufd_a, (M, nfd, nc))):
            if schedule.shape != shape or not np.isfinite(schedule).all() or np.any(schedule <= 0):
                raise ValueError("Tariff multipliers must have the expected shape and be finite and positive")
            for c in range(nc):
                if not np.allclose(schedule[c * ns:(c + 1) * ns, :, c], 1.0, rtol=0, atol=1e-14):
                    raise ValueError("Import tariffs must be one on domestic transactions")
        self.tau_a = tau_a
        self.taufd_a = taufd_a

        a = np.asarray(self.calib.a, dtype=float)
        afd = np.asarray(self.calib.afd, dtype=float)
        a_eff = (a * tau_a).reshape((M, M), order="F")
        b_t = a_eff.T / self._np_const["one_minus_tax"][:, None]
        self.inv_P_np = la.inv(np.eye(M) - b_t)
        const = dict(self._np_const)
        const["inv_P_T"] = self.inv_P_np.T.copy()
        const["afd_tf"] = (afd * taufd_a).reshape((M, nfd * nc), order="F")
        const["afd_duty"] = (afd * (taufd_a - 1.0)).reshape((M, nfd * nc), order="F")
        const["a_duty"] = (a * (tau_a - 1.0)).reshape((M, M), order="F")
        self._const_np = const
        self._const_cache: dict[str, dict[str, Any]] = {}

    def _constants(self, kind: str) -> dict[str, Any]:
        """Backend copies of the constants: 'numpy', 'torch', 'mlx64' or 'mlx32'."""
        if kind == "numpy":
            return self._const_np
        if kind not in self._const_cache:
            if kind == "torch":
                torch = self._torch
                self._const_cache[kind] = {
                    k: torch.as_tensor(v, device=self.device, dtype=self.dtype) for k, v in self._const_np.items()
                }
            else:
                mx = self._mx
                dtype = mx.float64 if kind == "mlx64" else mx.float32
                with mx.stream(mx.cpu if kind == "mlx64" else mx.gpu):
                    self._const_cache[kind] = {k: mx.array(v, dtype=dtype) for k, v in self._const_np.items()}
        return self._const_cache[kind]

    # ------------------------------------------------------------------
    # Residual evaluation
    # ------------------------------------------------------------------

    def _macro_residuals(self, X: Any, c: dict[str, Any], ops: _Ops, details: bool = False) -> Any:
        """Macro residuals of a batch ``X`` of shape ``(B, n_macro)`` (any backend)."""
        nc = self.nc
        if self.factor_equivalence:
            w = ops.exp(X[:, :nc])
            r = w
            T = X[:, nc: 2 * nc]
        else:
            r = ops.exp(X[:, :nc])
            w = ops.exp(X[:, nc: 2 * nc])
            T = X[:, 2 * nc: 3 * nc]
        alpha, beta = c["alpha"], c["beta"]
        r_cell = r @ c["e_cell"]
        w_cell = w @ c["e_cell"]
        unit_va = (r_cell / alpha) ** alpha * (w_cell / (1.0 - alpha)) ** (1.0 - alpha) / beta
        p = (unit_va / c["one_minus_tax"]) @ c["inv_P_T"]

        q = p @ c["afd_tf"]
        q = q * c["active_fd"] + (1.0 - c["active_fd"])
        price = q / (1.0 - c["fd_tax"])
        factor_income = w * c["l_endow"] + r * c["k_endow"]
        income = factor_income + T
        saving = c["saving"]
        if self.foreign_saving_units == "world_income":
            world = ops.sum(factor_income, 1)[:, None] * c["world_scale"]
            saving = saving * world
        expenditure = c["theta"] * (income @ c["e_fd"]) - saving
        quantity = expenditure / price
        d = quantity @ c["afd_T"]
        y = d @ c["inv_Y_T"]

        production_tax = (c["tax"] * p * y) @ c["e_cell"].T
        final_tax = (c["fd_tax"] * expenditure) @ c["e_fd"].T
        duties_i = (y * (p @ c["a_duty"])) @ c["e_cell"].T
        duties_f = (quantity * (p @ c["afd_duty"])) @ c["e_fd"].T
        government = production_tax + final_tax + duties_i + duties_f

        labor = ((1.0 - alpha) * unit_va * y / w_cell) @ c["e_cell"].T
        labor_gap = c["l_endow"] - labor
        budget_gap = T - government
        parts = [p[:, :1] - 1.0, labor_gap[:, 1:]]
        if not self.factor_equivalence:
            capital = (alpha * unit_va * y / r_cell) @ c["e_cell"].T
            parts.append(c["k_endow"] - capital)
        parts.append(budget_gap)
        f = ops.concat(parts)
        if details:
            return f, p, y
        return f

    def eval_macro_single(
        self,
        xm: np.ndarray,
        compute_full: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate the macro residuals at one point, in float64 on the host.

        Returns ``(f_macro, f_full, p_vec, p, y_vec)``: the macro residuals,
        the residual vector of the full consistent-accounting system at the
        reconstructed state (``None`` unless ``compute_full``), the producer
        prices (flat and ``(1, ns, nc)``) and the gross outputs.
        """
        xm = np.asarray(xm, dtype=float).ravel()
        f, p, y = self._macro_residuals(xm[None, :], self._const_np, _Ops("numpy"), details=True)
        p_vec, y_vec = p[0], y[0]
        f_full = None
        if compute_full:
            f_full = self.full_residuals(self._full_state(xm, p_vec, y_vec))
        return f[0], f_full, p_vec, p_vec.reshape((1, self.ns, self.nc), order="F"), y_vec

    def full_residuals(self, x_full: np.ndarray) -> np.ndarray:
        """Residuals of the full consistent-accounting system (the NumPy solver's) at ``x_full``."""
        from puremacro.trade._accounting import evaluate

        blocks = evaluate(
            x_full, self.calib, self.tau_a, self.taufd_a, None, None,
            foreign_saving_units=self.foreign_saving_units,
        )
        return np.concatenate([blocks["residuals"], blocks["physical_residuals"][:1],
                               blocks["net_exports_gap"]])

    def eval_macro_batched_torch(self, X_batch: Any) -> Any:
        """Evaluate a batch of macro points on PyTorch (CUDA, MPS or CPU)."""
        return self._macro_residuals(X_batch, self._constants("torch"), _Ops("torch", self._torch))

    def _mlx_stream(self, stream: Any) -> Any:
        """Resolve an MLX stream argument: None -> CPU (float64), 'cpu'/'gpu' or mx stream."""
        mx = self._mx
        if stream is None:
            return mx.cpu
        if isinstance(stream, str):
            key = stream.lower().strip()
            if key == "cpu":
                return mx.cpu
            if key == "gpu":
                return mx.gpu
            raise ValueError(f"Unknown MLX stream {stream!r}; expected 'cpu' or 'gpu'.")
        return stream

    def eval_macro_batched_mlx(self, X_batch: Any, stream: Any = None) -> Any:
        """Evaluate a batch of macro points on Apple MLX.

        ``stream`` selects the execution stream: ``mx.cpu`` / ``'cpu'`` (float64,
        the default) or ``mx.gpu`` / ``'gpu'`` (float32). A float64 ``X_batch``
        always runs on the CPU stream because Metal has no float64.
        """
        mx = self._mx
        target_stream = self._mlx_stream(stream)
        use_f64 = (X_batch.dtype == mx.float64) or (target_stream == mx.cpu)
        if use_f64:
            target_stream = mx.cpu
        with mx.stream(target_stream):
            if use_f64 and X_batch.dtype != mx.float64:
                X_batch = X_batch.astype(mx.float64)
            const = self._constants("mlx64" if use_f64 else "mlx32")
            # mx.exp is only float32-accurate on the float64 CPU stream; e ** x is
            # exact to about one ulp and differentiable, so it is used there.
            return self._macro_residuals(X_batch, const, _Ops("mlx", mx, exact_exp=use_f64))

    # ------------------------------------------------------------------
    # Jacobian evaluation
    # ------------------------------------------------------------------

    def _steps(self, xm: np.ndarray, eps_fd: float) -> np.ndarray:
        """Per-unknown finite-difference steps (log blocks: eps; T: eps * max(|T|, scale))."""
        h = np.full(len(xm), eps_fd, dtype=float)
        k = self.n_factor_vars
        h[k:] = eps_fd * np.maximum(np.abs(xm[k:]), self._value_scale)
        return h

    def evaluate_batched_jacobian(
        self,
        xm: np.ndarray,
        f_base: np.ndarray | None = None,
        eps_fd: float = 1e-6,
        ad_mode: Literal["finite_diff", "forward", "vjp"] = "finite_diff",
        stream: Any = None,
    ) -> np.ndarray:
        """Evaluate the full (B, B) macro Jacobian by batched execution or AD.

        Parameters
        ----------
        xm : np.ndarray
            Current macro point (length ``2*nc`` or ``3*nc``, matching the layout).
        f_base : np.ndarray | None
            Unperturbed macro residual vector. If None, evaluated automatically.
        eps_fd : float, default 1e-6
            Relative finite-difference step.
        ad_mode : str, default 'finite_diff'
            Differentiation mode: 'finite_diff' (batched one-sided differences),
            'forward' (forward-mode AD: torch.func.jacfwd / mx.jvp) or 'vjp'
            (reverse-mode AD: torch.func.jacrev / mx.vjp). If an AD mode fails
            on the device, a RuntimeWarning is emitted and finite differences are
            used; the fallback is never silent.
        stream : Any, default None
            MLX execution stream: ``'cpu'`` / ``mx.cpu`` (float64, the default)
            or ``'gpu'`` / ``mx.gpu`` (float32). Ignored for other backends.

        Returns
        -------
        np.ndarray
            Jacobian matrix J of shape (B, B).
        """
        if ad_mode not in ("finite_diff", "forward", "vjp"):
            raise ValueError(f"Unknown ad_mode {ad_mode!r}; expected 'finite_diff', 'forward' or 'vjp'.")

        xm = np.asarray(xm, dtype=float).ravel()
        B = len(xm)
        if B != self.batch_size:
            raise ValueError(
                f"Macro vector of length {B} does not match the evaluator layout (expected {self.batch_size})."
            )

        if f_base is None:
            f_base = self.eval_macro_single(xm)[0]

        if ad_mode in ("forward", "vjp") and self.backend == "torch":
            torch = self._torch
            try:
                def f_call(x_tensor: Any) -> Any:
                    return self.eval_macro_batched_torch(x_tensor.unsqueeze(0)).squeeze(0)

                xm_t = torch.as_tensor(xm, device=self.device, dtype=self.dtype)
                jac_fn = torch.func.jacfwd if ad_mode == "forward" else torch.func.jacrev
                return to_numpy(jac_fn(f_call)(xm_t)).astype(float, copy=False)
            except Exception as exc:
                warnings.warn(
                    f"ad_mode={ad_mode!r} failed on {self.backend}:{self.device} ({exc!r}); "
                    "falling back to batched finite differences.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        if ad_mode in ("forward", "vjp") and self.backend == "mlx":
            mx = self._mx
            target_stream = self._mlx_stream(stream)
            mx_dtype = mx.float64 if target_stream == mx.cpu else mx.float32
            try:
                with mx.stream(target_stream):
                    def f_call_mlx(x_m: Any) -> Any:
                        return mx.squeeze(
                            self.eval_macro_batched_mlx(mx.expand_dims(x_m, 0), stream=target_stream), 0
                        )

                    xm_mx = mx.array(xm, dtype=mx_dtype)
                    eye = np.eye(B)
                    if ad_mode == "forward":
                        cols = [to_numpy(mx.jvp(f_call_mlx, (xm_mx,), (mx.array(eye[b], dtype=mx_dtype),))[1][0])
                                for b in range(B)]
                        return np.column_stack(cols).astype(float, copy=False)
                    rows = [to_numpy(mx.vjp(f_call_mlx, (xm_mx,), (mx.array(eye[i], dtype=mx_dtype),))[1][0])
                            for i in range(B)]
                    return np.vstack(rows).astype(float, copy=False)
            except Exception as exc:
                warnings.warn(
                    f"ad_mode={ad_mode!r} failed on mlx ({exc!r}); falling back to batched finite differences.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        if ad_mode != "finite_diff" and self.backend == "numpy":
            warnings.warn(
                f"ad_mode={ad_mode!r} requires the torch or mlx backend; the numpy backend uses finite differences.",
                RuntimeWarning,
                stacklevel=2,
            )

        h = self._steps(xm, eps_fd)
        X_batch_np = np.tile(xm, (B, 1)) + np.diag(h)

        if self.backend == "torch":
            torch = self._torch
            X_batch_t = torch.as_tensor(X_batch_np, device=self.device, dtype=self.dtype)
            F_batch = to_numpy(self.eval_macro_batched_torch(X_batch_t)).astype(float, copy=False)
        elif self.backend == "mlx":
            mx = self._mx
            target_stream = self._mlx_stream(stream)
            dtype = mx.float64 if target_stream == mx.cpu else mx.float32
            with mx.stream(target_stream):
                F_batch_mx = self.eval_macro_batched_mlx(mx.array(X_batch_np, dtype=dtype), stream=target_stream)
                mx.eval(F_batch_mx)
                F_batch = to_numpy(F_batch_mx).astype(float, copy=False)
        else:
            F_batch = self._macro_residuals(X_batch_np, self._const_np, _Ops("numpy"))

        return (F_batch - f_base[np.newaxis, :]).T / h[np.newaxis, :]

    def jvp(self, xm: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Directional derivative (Jacobian-vector product) ``J(xm) @ v``.

        Uses forward-mode AD in float64 where the backend supports it (PyTorch on
        CPU/CUDA, the MLX CPU stream); otherwise, with a RuntimeWarning on the
        torch and mlx backends, a float64 central difference on the host.
        """
        xm = np.asarray(xm, dtype=float).ravel()
        v = np.asarray(v, dtype=float).ravel()
        if self.backend == "torch":
            torch = self._torch
            try:
                def f_call(x_tensor: Any) -> Any:
                    return self.eval_macro_batched_torch(x_tensor.unsqueeze(0)).squeeze(0)

                xm_t = torch.as_tensor(xm, device=self.device, dtype=self.dtype)
                v_t = torch.as_tensor(v, device=self.device, dtype=self.dtype)
                _, tangent = torch.func.jvp(f_call, (xm_t,), (v_t,))
                return to_numpy(tangent).astype(float, copy=False)
            except Exception as exc:
                warnings.warn(
                    f"torch.func.jvp failed on {self.device} ({exc!r}); using a central-difference "
                    "directional derivative.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        if self.backend == "mlx":
            mx = self._mx
            try:
                with mx.stream(mx.cpu):
                    def f_call_mlx(x_m: Any) -> Any:
                        return mx.squeeze(self.eval_macro_batched_mlx(mx.expand_dims(x_m, 0), stream=mx.cpu), 0)

                    _, tangent = mx.jvp(f_call_mlx, (mx.array(xm, dtype=mx.float64),),
                                        (mx.array(v, dtype=mx.float64),))
                    return to_numpy(tangent[0]).astype(float, copy=False)
            except Exception as exc:
                warnings.warn(
                    f"mx.jvp failed ({exc!r}); using a central-difference directional derivative.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        eps = 1e-7 * max(1.0, float(np.max(np.abs(xm)))) / max(float(np.max(np.abs(v))), 1e-300)
        f_plus = self.eval_macro_single(xm + eps * v)[0]
        f_minus = self.eval_macro_single(xm - eps * v)[0]
        return (f_plus - f_minus) / (2.0 * eps)
