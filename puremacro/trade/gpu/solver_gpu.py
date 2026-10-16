"""GPU-accelerated solver of the consistent-accounting trade model.

Solves the equilibrium of :func:`puremacro.trade.solve_trade_equilibrium`
(Cobb-Douglas intermediates, lump-sum fiscal closure) on its condensed macro
system (:mod:`puremacro.trade.gpu.batched_jacobian`), with PyTorch (NVIDIA CUDA /
Apple Silicon MPS), Apple MLX or NumPy Jacobians and Levenberg-Marquardt
regularized normal equations:

    (J^T J + mu * I) Delta x_m = -J^T F_m

combined with two-sided diagonal equilibration (Jacobi scaling) and backtracking
Armijo line search on the merit function Phi(x_m) = 1/2 ||F_m(x_m)||_2^2.

Precision policy
----------------
The macro Jacobian is assembled from one-sided finite differences, which are only
usable in float64. Jacobians are therefore evaluated in float64 wherever the device
supports it (PyTorch on CPU/CUDA, the MLX CPU stream). When ``device='auto'`` selects
a float32-only device (Apple MPS, MLX GPU), the solver transparently uses the float64
host evaluator of the same backend instead. When a float32 device is requested
explicitly (``device='mps'`` / ``'mlx'``), a RuntimeWarning is emitted, the device is
used only for the coarse phase (residual >= 5e6 in the first iterations) and the
Jacobian is polished in float64 on the host; the devices actually used are recorded
in ``result.metadata['jacobian_evaluations']``.

The returned result is post-processed and audited exactly like a NumPy solve:
``converged`` requires every equation of the full system, including the
physical goods and foreign-balance checks, to be within ``tol``.
"""
from __future__ import annotations

from collections import Counter
import time
import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np
import scipy.linalg as la

from puremacro.trade.gpu.backend import (
    _device_info,
    _load_torch,
    _resolve_backend_device,
    get_memory_usage,
    reset_peak_memory,
)
from puremacro.trade.gpu.batched_jacobian import BatchedJacobianEvaluator
from puremacro.trade.postprocessing import postprocess_trade_equilibrium
from puremacro.trade.solver import _resolve_tariffs

if TYPE_CHECKING:
    from puremacro.trade._results import TradeCalibrationResult, TradeEquilibriumResult

#: Residual level above which an explicitly requested float32 device is used for the
#: coarse phase of the iteration; below it the Jacobian is polished in float64.
_COARSE_RESIDUAL_THRESHOLD = 5e6
_COARSE_MAX_ITER = 5


def _jacobian_devices(
    requested_backend: str,
    requested_device: str,
    explicit_device: bool,
) -> tuple[tuple[str, str], tuple[str, str] | None]:
    """Choose the float64 (polish) evaluator and, if any, the float32 coarse evaluator.

    Returns ``((polish_backend, polish_device), coarse)`` where ``coarse`` is
    ``None`` unless the user explicitly requested a float32-only device.
    """
    info = _device_info(requested_backend, requested_device)
    if info.supports_float64:
        return (requested_backend, requested_device), None

    host = ("torch", "cpu") if requested_backend == "torch" else ("mlx", "cpu")
    if not explicit_device:
        # auto-selected float32 device: finite differences need float64
        return host, None

    warnings.warn(
        f"Device {requested_backend}:{requested_device} only supports float32; finite-difference "
        "Jacobians are unusable in float32, so it is used for the coarse phase only and the "
        f"Jacobian is polished in float64 on {host[0]}:{host[1]} "
        "(see result.metadata['jacobian_evaluations']).",
        RuntimeWarning,
        stacklevel=3,
    )
    return host, (requested_backend, requested_device)


def solve_trade_equilibrium_gpu(
    calib: TradeCalibrationResult,
    tau: np.ndarray | None = None,
    tau_fd: np.ndarray | None = None,
    tauf: np.ndarray | None = None,
    tauf_fd: np.ndarray | None = None,
    x0: np.ndarray | None = None,
    device: str | None = None,
    backend: str | None = None,
    batch_size: int | None = None,
    max_iter: int = 50,
    tol: float = 2.5e-3,
    damping: float = 1e-4,
    ad_mode: Literal["finite_diff", "forward", "vjp"] = "finite_diff",
    base_result: TradeEquilibriumResult | None = None,
    verbose: bool = False,
    *,
    factor_equivalence: bool | None = None,
    foreign_saving_units: str = "numeraire",
) -> TradeEquilibriumResult:
    """Solve multi-country multi-sector CGE trade equilibrium using GPU acceleration.

    Tariff arguments follow exactly the conventions of
    :func:`puremacro.trade.solver.solve_trade_equilibrium` (bilateral
    multipliers ``1 + rate``, or rates by importing country), so every call
    solves the same model as the NumPy solver with ``sigma=0`` and the same
    ``foreign_saving_units``.

    Parameters
    ----------
    calib : TradeCalibrationResult
        Calibrated model structural parameters.
    tau : np.ndarray | None, default None
        Intermediate tariff multipliers (1 + rate), shape (ns*nc, ns, nc) or
        (ns, nc, ns, nc), or rates by importing country (scalar or (nc,)). If
        None, the tariff-free baseline (all ones).
    tau_fd : np.ndarray | None, default None
        Final-demand tariffs with the same rules.
    tauf : np.ndarray | None, default None
        Intermediate tariff rates by importing country (nc,), used when ``tau``
        is None.
    tauf_fd : np.ndarray | None, default None
        Final-demand tariff rates by importing country (nc,), used when
        ``tau_fd`` is None.
    x0 : np.ndarray | None, default None
        Initial guess: the full state vector (length ``2*ns*nc + 4*nc - 1``) or a
        macro vector of the reduced (``2*nc``) or full (``3*nc``) layout.
        If None, unit factor returns and the calibrated transfers.
    device : str | None, default None
        Compute device ('auto', 'cuda', 'cuda:N', 'mps', 'mlx', 'gpu', 'cpu').
        If None, automatically detected; see the module docstring for the
        float64 policy applied to float32-only devices.
    backend : str | None, default None
        Compute backend ('torch', 'mlx', 'numpy' or None for automatic).
    batch_size : int | None, default None
        Macro dimension; must equal ``2*nc`` (reduced layout, r == w) or
        ``3*nc`` (full layout) for the calibration. Prefer ``factor_equivalence``.
    max_iter : int, default 50
        Maximum Levenberg-Marquardt iterations.
    tol : float, default 2.5e-3
        Absolute infinity-norm tolerance in the calibration's value units, on the
        macro residuals during the iteration and on every equation of the full
        system for ``converged``.
    damping : float, default 1e-4
        Initial Levenberg-Marquardt regularized damping parameter mu.
    ad_mode : str, default 'finite_diff'
        Differentiation mode ('finite_diff', 'forward', 'vjp'); see
        :meth:`BatchedJacobianEvaluator.evaluate_batched_jacobian`.
    base_result : TradeEquilibriumResult | None, default None
        Baseline equilibrium for terms-of-trade and CPI deflation calculations.
    verbose : bool, default False
        Whether to print per-iteration convergence diagnostics.
    factor_equivalence : bool | None, keyword-only, default None
        True imposes r == w (reduced macro layout), which is verified to be exact
        for the calibration (uniform capital shares across a country's sectors)
        and raises ValueError otherwise; False solves the full layout with
        separate capital-market clearing. None selects the reduced layout iff it
        is exact.
    foreign_saving_units : {"numeraire", "world_income"}, keyword-only
        Foreign-saving closure, as in :func:`puremacro.trade.solve_trade_equilibrium`.

    Returns
    -------
    TradeEquilibriumResult
        Fully post-processed trade equilibrium dataclass container. The metadata
        records ``device`` / ``backend`` actually used for the Jacobian, the
        requested pair, and ``jacobian_evaluations`` (counts by backend, device
        and dtype).
    """
    t_start = time.perf_counter()

    requested_backend, requested_device = _resolve_backend_device(device, backend)
    explicit_device = device is not None and str(device).lower().strip() not in ("auto", "")
    (polish_backend, polish_device), coarse = _jacobian_devices(
        requested_backend, requested_device, explicit_device
    )
    reset_peak_memory(polish_device, polish_backend)

    # Tariffs: identical defaults and shape handling to the NumPy solver
    tau_arr = None if tau is None else np.asarray(tau, dtype=float)
    tau_fd_arr = None if tau_fd is None else np.asarray(tau_fd, dtype=float)
    tau_a, taufd_a, _, _ = _resolve_tariffs(
        calib, tau=tau_arr, tau_fd=tau_fd_arr, tauf=tauf, tauf_fd=tauf_fd
    )

    # Float64 evaluator (always used for the residuals and the polished Jacobian)
    evaluator = BatchedJacobianEvaluator(
        calib=calib,
        tau_a=tau_a,
        taufd_a=taufd_a,
        device=polish_device,
        backend=polish_backend,
        batch_size=batch_size,
        factor_equivalence=factor_equivalence,
        foreign_saving_units=foreign_saving_units,
    )

    # Optional float32 evaluator for the coarse phase of an explicitly requested device
    evaluator_coarse = None
    coarse_stream = None
    if coarse is not None:
        if coarse[0] == "mlx":
            evaluator_coarse = evaluator  # same arrays, GPU stream
            coarse_stream = "gpu"
        else:
            evaluator_coarse = BatchedJacobianEvaluator(
                calib=calib,
                tau_a=tau_a,
                taufd_a=taufd_a,
                device=coarse[1],
                backend=coarse[0],
                batch_size=batch_size,
                factor_equivalence=evaluator.factor_equivalence,
                foreign_saving_units=foreign_saving_units,
            )

    xm = evaluator._macro_from_x0(x0)
    n_m = len(xm)
    n_factor_vars = evaluator.n_factor_vars
    torch = _load_torch() if polish_backend == "torch" else None
    jac_log: Counter[str] = Counter()

    # Initial residual evaluation
    f_m, _, p_sol, _, y_sol = evaluator.eval_macro_single(xm)
    res_max = float(np.max(np.abs(f_m)))
    iters = 0
    def _accepted(xm_c: np.ndarray, p_c: np.ndarray, y_c: np.ndarray) -> bool:
        """The full system (not only the macro residuals) is within ``tol``."""
        f_c = evaluator.full_residuals(evaluator._full_state(xm_c, p_c, y_c))
        return bool(np.isfinite(f_c).all() and float(np.max(np.abs(f_c))) <= tol)

    conv = res_max <= tol and _accepted(xm, p_sol, y_sol)

    if conv:
        if verbose:
            print(f"[solve_trade_equilibrium_gpu] Initial point already converged: max_res = {res_max:.4e}")
    else:
        mu = damping

        for it in range(max_iter):
            iters = it + 1
            res_max = float(np.max(np.abs(f_m)))

            # Stop once the full system is within tol; a macro residual within tol
            # alone keeps iterating (the reconstructed accounts may lag slightly).
            if res_max <= tol and _accepted(xm, p_sol, y_sol):
                conv = True
                break

            # Evaluate batched parallel Jacobian: float32 device only in the coarse phase
            use_coarse = (
                evaluator_coarse is not None
                and res_max >= _COARSE_RESIDUAL_THRESHOLD
                and it < _COARSE_MAX_ITER
            )
            if use_coarse:
                J = evaluator_coarse.evaluate_batched_jacobian(
                    xm, f_base=f_m, ad_mode=ad_mode, stream=coarse_stream
                )
                jac_log[f"{coarse[0]}:{coarse[1]}:float32"] += 1
            else:
                J = evaluator.evaluate_batched_jacobian(xm, f_base=f_m, ad_mode=ad_mode, stream="cpu" if polish_backend == "mlx" else None)
                jac_log[f"{polish_backend}:{polish_device}:float64"] += 1

            # Two-sided diagonal equilibration (Jacobi scaling)
            c_norm = np.linalg.norm(J, axis=0)
            c_norm[c_norm == 0] = 1.0
            D_R = 1.0 / c_norm
            J_scaled = J * D_R[np.newaxis, :]

            r_norm = np.linalg.norm(J_scaled, axis=1)
            r_norm[r_norm == 0] = 1.0
            D_L = 1.0 / r_norm
            J_equil = D_L[:, np.newaxis] * J_scaled
            rhs_equil = -D_L * f_m

            # Levenberg-Marquardt normal equations solve
            JT_J = J_equil.T @ J_equil
            JT_rhs = J_equil.T @ rhs_equil
            sol_u = None
            if torch is not None:
                try:
                    torch_dev = polish_device if (polish_device == "cpu" or polish_device.startswith("cuda")) else "cpu"
                    A_t = torch.as_tensor(JT_J + mu * np.eye(n_m), device=torch_dev, dtype=torch.float64)
                    b_t = torch.as_tensor(JT_rhs[:, None], device=torch_dev, dtype=torch.float64)
                    sol_t = torch.linalg.solve(A_t, b_t)
                    sol_u = sol_t.cpu().numpy().ravel()
                except Exception:
                    sol_u = None
            if sol_u is None:
                try:
                    sol_u = la.solve(JT_J + mu * np.eye(n_m), JT_rhs)
                except Exception:
                    sol_u = la.lstsq(J_equil, rhs_equil, rcond=1e-12)[0]

            delta_m = D_R * sol_u

            # Uniform step size clamping on factor prices to prevent exponent explosion
            max_factor_step = float(np.max(np.abs(delta_m[:n_factor_vars])))
            if max_factor_step > 0.3:
                delta_m *= (0.3 / max_factor_step)

            # Backtracking Armijo line search on merit function ||D_L * f_m||_2
            alpha_step = 1.0
            norm_0 = res_max
            norm_2_0 = float(np.linalg.norm(D_L * f_m))

            xm_trial = xm + alpha_step * delta_m
            xm_trial[:n_factor_vars] = np.clip(xm_trial[:n_factor_vars], -5.0, 5.0)
            f_trial_m, _, p_trial, _, y_trial = evaluator.eval_macro_single(xm_trial)

            best_trial = (xm_trial, f_trial_m, p_trial, y_trial)
            best_res_m = float(np.max(np.abs(f_trial_m)))
            best_norm_2 = float(np.linalg.norm(D_L * f_trial_m))

            accepted = False
            for _ in range(12):
                res_trial_m = float(np.max(np.abs(f_trial_m)))
                norm_2_trial = float(np.linalg.norm(D_L * f_trial_m))

                if norm_2_trial < best_norm_2 or res_trial_m < best_res_m:
                    best_trial = (xm_trial, f_trial_m, p_trial, y_trial)
                    best_res_m = res_trial_m
                    best_norm_2 = norm_2_trial

                if res_trial_m <= tol or norm_2_trial < norm_2_0 or res_trial_m < norm_0:
                    accepted = True
                    break

                alpha_step *= 0.5
                xm_trial = xm + alpha_step * delta_m
                xm_trial[:n_factor_vars] = np.clip(xm_trial[:n_factor_vars], -5.0, 5.0)
                f_trial_m, _, p_trial, _, y_trial = evaluator.eval_macro_single(xm_trial)

            if accepted:
                xm = xm_trial
                f_m = f_trial_m
                p_sol = p_trial
                y_sol = y_trial
                mu = max(mu * 0.3, 1e-8)
            elif best_norm_2 < norm_2_0 or best_res_m < norm_0:
                xm, f_m, p_sol, y_sol = best_trial
                mu = max(mu * 0.5, 1e-8)
            else:
                mu = min(mu * 5.0, 1e2)

            if verbose:
                print(
                    f"[{polish_backend}:{polish_device}] Iter {it + 1:02d}: "
                    f"max_res = {np.max(np.abs(f_m)):.4e}, "
                    f"step = {alpha_step:.4f}, mu = {mu:.2e}"
                )

    # Full state vector, audited on the full system (as for a NumPy solve)
    x_full = evaluator._full_state(xm, p_sol, y_sol)
    f_full = evaluator.full_residuals(x_full)
    conv = bool(np.isfinite(f_full).all() and float(np.max(np.abs(f_full))) <= tol)

    t_end = time.perf_counter()
    mem_info = get_memory_usage(polish_device, polish_backend)

    return postprocess_trade_equilibrium(
        x_sol=x_full,
        calib=calib,
        tau=tau_a,
        tau_fd=taufd_a,
        converged=conv,
        iterations=iters,
        base_result=base_result,
        foreign_saving_units=foreign_saving_units,
        metadata={
            "method": "gpu_accelerated",
            "device": polish_device,
            "backend": polish_backend,
            "requested_device": requested_device,
            "requested_backend": requested_backend,
            "jacobian_device": f"{polish_backend}:{polish_device}",
            "jacobian_dtype": "float64",
            "jacobian_evaluations": dict(jac_log),
            "batch_size": evaluator.batch_size,
            "factor_equivalence": evaluator.factor_equivalence,
            "ad_mode": ad_mode,
            "tol": tol,
            "max_iter": max_iter,
            "macro_residual_max": float(np.max(np.abs(f_m))),
            "solve_duration_seconds": t_end - t_start,
            "memory_usage": mem_info,
        },
    )
