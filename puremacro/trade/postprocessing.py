"""Post-processing and flow accounting for solved trade equilibria.

Rebuilds the bilateral intermediate and final-demand flows, purchaser prices,
tariff and tax receipts, GDP aggregates, a fixed-basket CPI and the terms of
trade from a solved state vector, and audits the accounting identities before a
result is reported as converged.

Pure NumPy, vectorised.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from puremacro.trade.equilibrium import _evaluate_equilibrium

if TYPE_CHECKING:
    from puremacro.trade._results import TradeCalibrationResult, TradeEquilibriumResult


def intermediate_matrix_to_tensor(matrix: np.ndarray, *, ns: int, nc: int) -> np.ndarray:
    """Convert an intermediate transaction matrix to a 4D bilateral tensor.

    Parameters
    ----------
    matrix : np.ndarray
        2D matrix of shape (ns*nc, ns*nc), sectors stacked within countries.
    ns, nc : int (keyword-only)
        Number of sectors and countries.

    Returns
    -------
    np.ndarray
        4D tensor of shape (ns, nc, ns, nc) indexed as
        (s_src, c_src, s_dst, c_dst).
    """
    return np.asarray(matrix, dtype=float).reshape((ns, nc, ns, nc), order="F")


def intermediate_tensor_to_matrix(tensor: np.ndarray, *, ns: int, nc: int) -> np.ndarray:
    """Convert a 4D bilateral intermediate tensor (ns, nc, ns, nc) to a (ns*nc, ns*nc) matrix."""
    return np.asarray(tensor, dtype=float).reshape((ns * nc, ns * nc), order="F")


def final_demand_matrix_to_tensor(matrix: np.ndarray, *, ns: int, nc: int, nfd: int) -> np.ndarray:
    """Convert a (ns*nc, nfd*nc) final-demand matrix to a 4D tensor.

    The tensor has shape (ns, nc, nfd, nc) and is indexed as
    (s_src, c_src, fd_dst, c_dst).
    """
    return np.asarray(matrix, dtype=float).reshape((ns, nc, nfd, nc), order="F")


def final_demand_tensor_to_matrix(tensor: np.ndarray, *, ns: int, nc: int, nfd: int) -> np.ndarray:
    """Convert a 4D final-demand tensor (ns, nc, nfd, nc) to a (ns*nc, nfd*nc) matrix."""
    return np.asarray(tensor, dtype=float).reshape((ns * nc, nfd * nc), order="F")


def compute_postprocessing_flows(
    x_sol: np.ndarray,
    calib: TradeCalibrationResult,
    tau: np.ndarray | None = None,
    tau_fd: np.ndarray | None = None,
    tauf: np.ndarray | None = None,
    tauf_fd: np.ndarray | None = None,
    base_result: TradeEquilibriumResult | None = None,
    **model_options: Any,
) -> dict[str, Any]:
    """Compute post-processing flows, transaction matrices, prices, and aggregates.

    Reports producer-price bilateral trade, purchaser-price expenditure,
    nominal tax/duty/factor-payment tables and a fixed-basket purchaser CPI. It
    returns the solved producer prices, and ``accounting_metadata`` records all
    account residuals and the price/foreign-balance closure.

    Parameters
    ----------
    x_sol : np.ndarray
        Solved equilibrium state vector of length 2*ns*nc + 3*nc + (nc - 1).
    calib : TradeCalibrationResult
        Calibrated model structural parameters.
    tau, tau_fd : np.ndarray, optional
        Intermediate and final-demand tariff multipliers (1 + tariff_rate).
    tauf, tauf_fd : np.ndarray, optional
        National tariff rates by importing country, shape (nc,).
    base_result : TradeEquilibriumResult, optional
        Baseline result whose final-use baskets weight the CPI.
    **model_options
        ``sigma``, ``accounting``, ``fiscal_closure`` and the other options of
        :func:`puremacro.trade.equilibrium.compute_equilibrium_residuals`.

    Returns
    -------
    dict[str, Any]
        Dictionary with reconstructed flows, prices, aggregates, and tables.
    """
    from ._accounting import postprocess

    blocks = _evaluate_equilibrium(x_sol, calib, tau, tau_fd, tauf, tauf_fd, **model_options)
    return postprocess(blocks, calib, base_result)


def postprocess_trade_equilibrium(
    x_sol: np.ndarray,
    calib: TradeCalibrationResult,
    tau: np.ndarray | None = None,
    tau_fd: np.ndarray | None = None,
    tauf: np.ndarray | None = None,
    tauf_fd: np.ndarray | None = None,
    converged: bool = True,
    iterations: int = 0,
    base_result: TradeEquilibriumResult | None = None,
    metadata: dict[str, Any] | None = None,
    **model_options: Any,
) -> TradeEquilibriumResult:
    """Post-process an equilibrium state and package it in a TradeEquilibriumResult.

    The ``converged`` flag passed in is kept only if the independent audit
    passes: all physical equations (including the goods equation dropped for
    Walras' law and the realised foreign balances) and the solved residuals are
    finite and within ``metadata["tol"]`` (1e-8 when absent), and final demand
    is feasible.

    Parameters
    ----------
    x_sol : np.ndarray
        Solved equilibrium state vector.
    calib : TradeCalibrationResult
        Calibrated model structural parameters.
    tau, tau_fd : np.ndarray, optional
        Intermediate and final-demand tariff multipliers.
    tauf, tauf_fd : np.ndarray, optional
        National tariff rates by importing country.
    converged : bool, default True
        Solver convergence flag.
    iterations : int, default 0
        Number of iterations executed.
    base_result : TradeEquilibriumResult, optional
        Baseline result for the CPI basket.
    metadata : dict[str, Any], optional
        Additional diagnostic and provenance metadata.
    **model_options
        Options of :func:`puremacro.trade.equilibrium.compute_equilibrium_residuals`.

    Returns
    -------
    TradeEquilibriumResult
        Frozen dataclass with all flow accounting, prices, and aggregates populated.
    """
    from puremacro.trade._results import TradeEquilibriumResult

    flows = compute_postprocessing_flows(
        x_sol=x_sol,
        calib=calib,
        tau=tau,
        tau_fd=tau_fd,
        tauf=tauf,
        tauf_fd=tauf_fd,
        base_result=base_result,
        **model_options,
    )

    meta = dict(metadata) if metadata is not None else {}
    meta.update(model_options)
    meta.update({
        "fiscal_tariffs": flows["fiscal_tariffs"],
        "bilateral_trade": flows["bilateral_trade"],
        "tariffs_interm": flows["tariffs_interm"],
        "tariffs_fd": flows["tariffs_fd"],
        "data_model_vf": flows["data_model_vf"],
        "data_tariff_vf": flows["data_tariff_vf"],
    })
    meta.update(flows.get("accounting_metadata", {}))

    # Independent acceptance audit. The Newton, sparse-LU, Broyden, Krylov and
    # Keller solvers stop on this same residual set (solver._consistent_acceptor),
    # so for them it can only flip the flag for demand-infeasible roots; SciPy
    # hybr/lm use their own termination tests and are audited here.
    tolerance = float(meta.get("tol", 1e-8))
    physical = meta["physical_residuals"]
    converged = bool(converged and meta["demand_feasible"] and np.isfinite(physical).all()
                     and np.max(np.abs(physical)) <= tolerance
                     and np.isfinite(flows["max_residual"]) and flows["max_residual"] <= tolerance)

    return TradeEquilibriumResult(
        x_sol=np.asarray(x_sol, dtype=float).ravel(),
        p_sol=flows["p_sol"],
        y_sol=flows["y_sol"],
        r_sol=flows["r_sol"],
        w_sol=flows["w_sol"],
        T_sol=flows["T_sol"],
        XN_sol=flows["XN_sol"],
        intermediate_flows=flows["intermediate_flows"],
        final_demand_flows=flows["final_demand_flows"],
        p_fd=flows["p_fd"],
        c_fd=flows["c_fd"],
        gdp=flows["gdp"],
        gdp_fc=flows["gdp_fc"],
        imports=flows["imports"],
        exports=flows["exports"],
        tariffs=flows["tariffs"],
        cpi=flows["cpi"],
        terms_of_trade=flows["terms_of_trade"],
        c_sol=flows["c_fd"],
        pfd_sol=flows["p_fd"],
        Pfd_final=flows["P_fd"],
        qxX0_sol=flows["qxX0"],
        qxFD0_sol=flows["qxFD0"],
        data_model_vf=flows["data_model_vf"],
        data_tariff_vf=flows["data_tariff_vf"],
        converged=converged,
        iterations=iterations,
        diff=flows["diff"],
        residual_norm=flows["residual_norm"],
        max_residual=flows["max_residual"],
        residuals=flows["residuals"],
        country_codes=calib.country_codes,
        sector_codes=calib.sector_codes,
        metadata=meta,
    )


__all__ = [
    "intermediate_matrix_to_tensor",
    "intermediate_tensor_to_matrix",
    "final_demand_matrix_to_tensor",
    "final_demand_tensor_to_matrix",
    "compute_postprocessing_flows",
    "postprocess_trade_equilibrium",
]
