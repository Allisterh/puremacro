"""Apple Silicon MLX General Equilibrium Solver for puremacro.trade.

Provides native Apple MLX acceleration utilizing Unified Memory Architecture (UMA)
for zero-copy tensor sharing. The Jacobian is assembled on the MLX **CPU stream in
float64** (Metal has no float64, and float32 finite differences are too
inaccurate for Newton steps); the float32 GPU stream is used only for the coarse
phase of the iteration (residual >= 5e6). :func:`solve_trade_equilibrium_mlx` shares the
Levenberg-Marquardt driver of :func:`puremacro.trade.gpu.solver_gpu.solve_trade_equilibrium_gpu`.
"""
from __future__ import annotations

from dataclasses import replace
import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np

from puremacro.trade.gpu.backend import has_mlx
from puremacro.trade.gpu.solver_gpu import solve_trade_equilibrium_gpu

if TYPE_CHECKING:
    from puremacro.trade._results import TradeCalibrationResult, TradeEquilibriumResult


def solve_trade_equilibrium_mlx(
    calib: TradeCalibrationResult,
    tau: np.ndarray | None = None,
    tau_fd: np.ndarray | None = None,
    tauf: np.ndarray | None = None,
    tauf_fd: np.ndarray | None = None,
    x0: np.ndarray | None = None,
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
    """Solve multi-country multi-sector CGE trade equilibrium using Apple MLX.

    Same model, arguments and result as
    :func:`puremacro.trade.gpu.solver_gpu.solve_trade_equilibrium_gpu` with
    ``device='mlx'``.

    Parameters
    ----------
    calib : TradeCalibrationResult
        Calibrated model structural parameters.
    tau, tau_fd : np.ndarray | None, default None
        Intermediate and final-demand tariffs (multipliers ``1 + rate`` or rates
        by importing country); None means no tariffs.
    tauf, tauf_fd : np.ndarray | None, default None
        Tariff rates by importing country, used when ``tau`` / ``tau_fd`` is None.
    x0 : np.ndarray | None, default None
        Initial guess: full state vector (``2*ns*nc + 4*nc - 1``) or a macro
        vector of the reduced (``2*nc``) or full (``3*nc``) layout.
    batch_size : int | None, default None
        Macro dimension; must equal ``2*nc`` or ``3*nc`` for the calibration.
        Prefer ``factor_equivalence``.
    max_iter : int, default 50
        Maximum Levenberg-Marquardt iterations.
    tol : float, default 2.5e-3
        Absolute convergence tolerance (see ``solve_trade_equilibrium_gpu``).
    damping : float, default 1e-4
        Initial Levenberg-Marquardt damping parameter mu.
    ad_mode : str, default 'finite_diff'
        Differentiation mode ('finite_diff', 'forward', 'vjp').
    base_result : TradeEquilibriumResult | None, default None
        Baseline equilibrium for terms-of-trade and CPI calculations.
    verbose : bool, default False
        Whether to print per-iteration convergence diagnostics.
    factor_equivalence : bool | None, keyword-only, default None
        Reduced (r == w) versus full macro layout; see
        :func:`solve_trade_equilibrium_gpu`.
    foreign_saving_units : {"numeraire", "world_income"}, keyword-only
        Foreign-saving closure, as in :func:`puremacro.trade.solve_trade_equilibrium`.

    Returns
    -------
    TradeEquilibriumResult
        Fully post-processed trade equilibrium dataclass container with
        ``metadata['method'] == 'mlx_accelerated'``.
    """
    if not has_mlx():
        raise RuntimeError("Apple MLX is not installed or not supported on this platform.")

    with warnings.catch_warnings():
        # The float32-device notice is implied by calling the MLX solver explicitly.
        warnings.filterwarnings(
            "ignore",
            message="Device mlx:gpu only supports float32",
            category=RuntimeWarning,
        )
        res = solve_trade_equilibrium_gpu(
            calib=calib,
            tau=tau,
            tau_fd=tau_fd,
            tauf=tauf,
            tauf_fd=tauf_fd,
            x0=x0,
            device="mlx",
            backend="mlx",
            batch_size=batch_size,
            max_iter=max_iter,
            tol=tol,
            damping=damping,
            ad_mode=ad_mode,
            base_result=base_result,
            verbose=verbose,
            factor_equivalence=factor_equivalence,
            foreign_saving_units=foreign_saving_units,
        )

    metadata = dict(res.metadata or {})
    metadata["method"] = "mlx_accelerated"
    return replace(res, metadata=metadata)
