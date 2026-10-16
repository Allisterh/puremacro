"""Adaptive homotopy continuation for large tariff shocks.

Follows the path lambda in [0, 1] of tariff multipliers

    tau(lambda) = tau_0 + lambda * (tau_1 - tau_0)
    taufd(lambda) = taufd_0 + lambda * (taufd_1 - taufd_0)

with adaptive step sizing, warm starts and backtracking on non-convergence,
solving each step with :func:`puremacro.trade.gpu.solver_gpu.solve_trade_equilibrium_gpu`.
Large asymmetric tariff shocks, where a Newton solve from the tariff-free
equilibrium can fail, are reached through a sequence of nearby problems.
"""
from __future__ import annotations

import time
import warnings
from typing import TYPE_CHECKING, Callable

import numpy as np

from puremacro.trade.gpu.backend import _resolve_backend_device
from puremacro.trade.gpu.solver_gpu import solve_trade_equilibrium_gpu
from puremacro.trade.solver import _resolve_tariffs

if TYPE_CHECKING:
    from puremacro.trade._results import TradeCalibrationResult, TradeEquilibriumResult


def solve_homotopy_continuation(
    calib: TradeCalibrationResult,
    target_tau: np.ndarray,
    target_tau_fd: np.ndarray,
    base_tau: np.ndarray | None = None,
    base_tau_fd: np.ndarray | None = None,
    x0: np.ndarray | None = None,
    device: str | None = None,
    backend: str | None = None,
    batch_size: int | None = None,
    initial_step: float = 0.1,
    min_step: float = 1e-4,
    max_step: float = 0.5,
    tol: float = 2.5e-3,
    damping: float = 1e-4,
    max_iter_per_step: int = 30,
    base_result: TradeEquilibriumResult | None = None,
    verbose: bool = False,
    callback: Callable[[float, TradeEquilibriumResult], None] | None = None,
    *,
    factor_equivalence: bool | None = None,
    foreign_saving_units: str = "numeraire",
) -> TradeEquilibriumResult:
    """Solve the trade equilibrium along an adaptive homotopy path from base to target tariffs.

    Parameters
    ----------
    calib : TradeCalibrationResult
        Calibrated model structural parameters.
    target_tau : np.ndarray
        Target intermediate tariffs (tau_1), in any form
        :func:`puremacro.trade.solve_trade_equilibrium` accepts.
    target_tau_fd : np.ndarray
        Target final-demand tariffs (taufd_1), in the same forms.
    base_tau : np.ndarray | None, default None
        Base intermediate tariffs (tau_0). Defaults to no tariffs.
    base_tau_fd : np.ndarray | None, default None
        Base final-demand tariffs (taufd_0). Defaults to no tariffs.
    x0 : np.ndarray | None, default None
        Equilibrium state at the base tariffs. If None, it is solved first.
    device : str | None, default None
        Target compute device ('auto', 'cuda', 'cuda:N', 'mps', 'mlx', 'gpu', 'cpu');
        see :func:`solve_trade_equilibrium_gpu` for the float64 policy.
    backend : str | None, default None
        Target backend ('torch', 'mlx', 'numpy' or None for automatic).
    batch_size : int | None, default None
        Macro dimension; must equal ``2*nc`` or ``3*nc`` for the calibration.
        Prefer ``factor_equivalence``.
    initial_step : float, default 0.1
        Initial homotopy step size Delta lambda.
    min_step : float, default 1e-4
        Minimum allowable step size before aborting.
    max_step : float, default 0.5
        Maximum allowable step size Delta lambda.
    tol : float, default 2.5e-3
        Absolute convergence tolerance of each step's solve.
    damping : float, default 1e-4
        Levenberg-Marquardt regularized damping.
    max_iter_per_step : int, default 30
        Maximum Newton iterations per continuation step.
    base_result : TradeEquilibriumResult | None, default None
        Reference equilibrium for postprocessing metrics.
    verbose : bool, default False
        Whether to log continuation steps and convergence diagnostics.
    callback : callable | None, default None
        Optional hook called after each successful continuation step as callback(lambda_val, result).
    factor_equivalence : bool | None, keyword-only, default None
        Reduced (r == w) versus full macro layout; see :func:`solve_trade_equilibrium_gpu`.
    foreign_saving_units : {"numeraire", "world_income"}, keyword-only
        Foreign-saving closure, as in :func:`puremacro.trade.solve_trade_equilibrium`.

    Returns
    -------
    TradeEquilibriumResult
        Converged equilibrium result at the target tariff schedule (lambda = 1.0).
    """
    t_start = time.perf_counter()
    # Validate the device request once; the per-step solver receives the
    # original strings so its float64 policy (auto vs explicit device) applies.
    _resolve_backend_device(device, backend)
    target_backend = backend
    target_device = device

    t_tau, t_tau_fd, _, _ = _resolve_tariffs(calib, target_tau, target_tau_fd)
    b_tau, b_tau_fd, _, _ = _resolve_tariffs(calib, base_tau, base_tau_fd)

    # Current continuation state
    curr_lambda = 0.0
    step_size = initial_step
    curr_x = x0

    # Ensure baseline state is converged if not provided
    if curr_x is None:
        if verbose:
            print("[homotopy] Solving baseline equilibrium at lambda = 0.0 ...")
        res_base = solve_trade_equilibrium_gpu(
            calib=calib,
            tau=b_tau,
            tau_fd=b_tau_fd,
            x0=None,
            device=target_device,
            backend=target_backend,
            batch_size=batch_size,
            factor_equivalence=factor_equivalence,
            max_iter=max_iter_per_step,
            tol=tol,
            damping=damping,
            verbose=verbose,
            foreign_saving_units=foreign_saving_units,
        )
        if not res_base.converged:
            raise RuntimeError("Baseline equilibrium at lambda = 0.0 failed to converge.")
        curr_x = res_base.x_sol
        last_good_result = res_base
    else:
        last_good_result = None

    last_good_x = curr_x.copy()
    step_count = 0

    while curr_lambda < 1.0 - 1e-10:
        step_count += 1
        target_step_lambda = min(curr_lambda + step_size, 1.0)
        actual_delta = target_step_lambda - curr_lambda

        # Linear tariff interpolation along homotopy path
        lam = target_step_lambda
        lam_tau = (1.0 - lam) * b_tau + lam * t_tau
        lam_tau_fd = (1.0 - lam) * b_tau_fd + lam * t_tau_fd

        if verbose:
            print(
                f"[homotopy] Step {step_count:02d}: Attempting lambda = {lam:.4f} "
                f"(step size = {actual_delta:.4f}) ..."
            )

        with warnings.catch_warnings():
            if step_count > 1 or x0 is None:
                # the float32-device notice was already issued by the first solve
                warnings.filterwarnings(
                    "ignore", message="Device .* only supports float32", category=RuntimeWarning
                )
            step_res = solve_trade_equilibrium_gpu(
                calib=calib,
                tau=lam_tau,
                tau_fd=lam_tau_fd,
                x0=last_good_x,
                device=target_device,
                backend=target_backend,
                batch_size=batch_size,
                factor_equivalence=factor_equivalence,
                max_iter=max_iter_per_step,
                tol=tol,
                damping=damping,
                base_result=base_result,
                verbose=False,
                foreign_saving_units=foreign_saving_units,
            )

        if step_res.converged and step_res.max_residual <= tol:
            curr_lambda = target_step_lambda
            last_good_x = step_res.x_sol.copy()
            last_good_result = step_res

            if verbose:
                print(
                    f"[homotopy] Step {step_count:02d} SUCCESS: lambda = {curr_lambda:.4f}, "
                    f"iters = {step_res.iterations}, max_res = {step_res.max_residual:.4e}"
                )

            if callback is not None:
                callback(curr_lambda, step_res)

            # Adaptive step enlargement if solved rapidly
            if step_res.iterations <= 4:
                step_size = min(step_size * 1.5, max_step)
            elif step_res.iterations >= max_iter_per_step - 5:
                step_size = max(step_size * 0.75, min_step)
        else:
            # Backtrack and halve step size
            step_size *= 0.5
            if verbose:
                print(
                    f"[homotopy] Step {step_count:02d} FAILED at lambda = {target_step_lambda:.4f} "
                    f"(max_res = {step_res.max_residual:.4e}). Backtracking, new step = {step_size:.4e}"
                )

            if step_size < min_step:
                raise RuntimeError(
                    f"Homotopy continuation failed: step size {step_size:.6e} fell below "
                    f"minimum threshold {min_step:.6e} at lambda = {curr_lambda:.4f}. "
                    f"Last residual: {step_res.max_residual:.4e}."
                )

    if last_good_result is None:
        raise RuntimeError("Homotopy continuation terminated without producing a result.")

    t_total = time.perf_counter() - t_start
    if verbose:
        print(
            f"[homotopy] COMPLETED full path to lambda = 1.0000 in {t_total:.2f}s "
            f"across {step_count} adaptive steps."
        )

    return last_good_result
