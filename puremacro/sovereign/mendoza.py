"""Illustrative adjustment-cost economy with a collateral constraint.

Solves the small open economy with domestic capital accumulation and an
endogenous borrowing collateral constraint:
    b_{t+1} <= kappa * q_t * k_{t+1}
where q_t = 1 + phi * (k_{t+1} - k_t) / k_t is Tobin's q.

This planning problem is inspired by Mendoza (2010); it omits the published
competitive model's working-capital and imported-input mechanisms. Small-grid
Bellman checks do not establish a replication of its published crisis moments.
Amplification and crisis frequency are measurements, never imposed targets.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import norm

from puremacro.sovereign._results import SuddenStopResult

__all__ = [
    "solve_sudden_stop",
]


def _discretize_productivity(
    n_z: int,
    rho_z: float,
    sigma_z: float,
    m: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Discretize log productivity shock z via Tauchen (1986)."""
    if n_z < 2:
        return np.array([1.0]), np.array([[1.0]])

    sigma_unc = sigma_z / np.sqrt(1.0 - rho_z ** 2)
    z_max = m * sigma_unc
    log_z_grid = np.linspace(-z_max, z_max, n_z)
    step = log_z_grid[1] - log_z_grid[0]
    p_trans = np.zeros((n_z, n_z), dtype=np.float64)

    for i in range(n_z):
        mu_i = rho_z * log_z_grid[i]
        p_trans[i, 0] = norm.cdf((log_z_grid[0] - mu_i + step / 2.0) / sigma_z)
        p_trans[i, n_z - 1] = norm.sf((log_z_grid[n_z - 1] - mu_i - step / 2.0) / sigma_z)
        for j in range(1, n_z - 1):
            hi = norm.cdf((log_z_grid[j] - mu_i + step / 2.0) / sigma_z)
            lo = norm.cdf((log_z_grid[j] - mu_i - step / 2.0) / sigma_z)
            p_trans[i, j] = hi - lo

    z_grid = np.exp(log_z_grid)
    # Normalize mean to 1.0
    z_grid = z_grid / float(np.mean(z_grid))
    return z_grid, p_trans


def solve_sudden_stop(
    beta: float = 0.96,
    gamma: float = 2.0,
    r_star: float = 0.04,
    kappa_collateral: float = 0.30,
    *,
    alpha: float = 0.32,
    delta_k: float = 0.088,
    phi: float = 0.08,
    rho_z: float = 0.90,
    sigma_z: float = 0.025,
    n_b: int = 25,
    b_min: float = 0.0,
    b_max: float | None = None,
    n_k: int = 25,
    k_min: float | None = None,
    k_max: float | None = None,
    n_z: int = 3,
    m_tauchen: float = 3.0,
    tol: float = 1e-6,
    max_iter: int = 500,
    seed: int = 42,
) -> SuddenStopResult:
    """Solve an illustrative collateral-constrained small open economy.

    Parameters
    ----------
    beta : float, default 0.96
        Subjective annual/quarterly discount factor.
    gamma : float, default 2.0
        CRRA relative risk aversion parameter.
    r_star : float, default 0.04
        World real risk-free rate.
    kappa_collateral : float, default 0.30
        Collateral constraint parameter b' <= kappa * q * k'.
    alpha : float, default 0.32
        Capital share in production y = z * k^alpha.
    delta_k : float, default 0.088
        Capital depreciation rate.
    phi : float, default 0.08
        Capital adjustment cost parameter.
    rho_z : float, default 0.90
        Autoregressive persistence of TFP shocks.
    sigma_z : float, default 0.025
        Innovation standard deviation of TFP shocks.
    n_b : int, default 25
        Number of grid points for foreign debt.
    b_min : float, default 0.0
        Minimum debt on grid.
    b_max : float, optional
        Maximum debt on grid. Defaults to 1.05 * kappa * k_ss.
    n_k : int, default 25
        Number of grid points for physical capital.
    k_min : float, optional
        Minimum capital. Defaults to 0.82 * k_ss.
    k_max : float, optional
        Maximum capital. Defaults to 1.18 * k_ss.
    n_z : int, default 3
        Number of points for TFP shock.
    m_tauchen : float, default 3.0
        Standard deviation multiplier for Tauchen discretization.
    tol : float, default 1e-6
        Convergence tolerance on value function.
    max_iter : int, default 500
        Maximum value function iterations.
    seed : int, default 42
        Random seed for simulation diagnostics.

    Returns
    -------
    SuddenStopResult
        Result containing policy functions for debt, capital, consumption,
        Tobin's q, collateral multipliers, and crisis diagnostics.
    """
    if kappa_collateral <= 0.0:
        raise ValueError(f"kappa_collateral must be > 0; got {kappa_collateral}")
    if beta <= 0.0 or beta >= 1.0:
        raise ValueError(f"beta must be in (0, 1); got {beta}")
    if r_star <= 0.0:
        raise ValueError(f"r_star must be > 0; got {r_star}")
    if not np.all(np.isfinite([beta, gamma, r_star, kappa_collateral, alpha,
                               delta_k, phi, rho_z, sigma_z, m_tauchen, tol])):
        raise ValueError('parameters must be finite')
    if not (gamma > 0 and 0 < alpha < 1 and 0 <= delta_k <= 1 and phi >= 0
            and abs(rho_z) < 1 and sigma_z > 0 and m_tauchen > 0 and tol > 0):
        raise ValueError('invalid preference, production, productivity or tolerance parameter')
    if any(isinstance(n, bool) or not isinstance(n, (int, np.integer))
           for n in (n_b, n_k, n_z, max_iter)) or min(n_b, n_k) < 2 or n_z < 1 or max_iter < 1:
        raise ValueError('integer grids n_b,n_k >= 2, n_z >= 1 and max_iter >= 1 required')

    # Steady-state capital: alpha * k_ss^(alpha - 1) = r_star + delta_k
    k_ss = float(((r_star + delta_k) / alpha) ** (1.0 / (alpha - 1.0)))

    if k_min is None:
        k_min = 0.82 * k_ss
    if k_max is None:
        k_max = 1.18 * k_ss
    if b_max is None:
        b_max = 1.05 * kappa_collateral * k_ss
    if not np.all(np.isfinite([b_min, b_max, k_min, k_max])) or not (b_max > b_min and k_max > k_min > 0):
        raise ValueError('finite increasing debt and positive capital grid bounds required')

    k_grid = np.linspace(k_min, k_max, n_k)
    b_grid = np.linspace(b_min, b_max, n_b)
    z_grid, p_trans = _discretize_productivity(n_z, rho_z, sigma_z, m=m_tauchen)

    # Precalculate investment and Tobin's q matrices: shape (n_k, n_k)
    # Row: current k_i, Column: next k_j
    k_cur_mat = k_grid[:, None]
    k_next_mat = k_grid[None, :]
    i_mat = k_next_mat - (1.0 - delta_k) * k_cur_mat + 0.5 * phi * (k_next_mat - k_cur_mat) ** 2 / k_cur_mat
    q_mat = 1.0 + phi * (k_next_mat - k_cur_mat) / k_cur_mat
    collateral_limit = kappa_collateral * q_mat * k_next_mat  # (n_k, n_k)

    def u_func(c_arr: np.ndarray) -> np.ndarray:
        feasible = c_arr > 0
        val = np.full_like(c_arr, -np.inf)
        val[feasible] = (np.log(c_arr[feasible]) if gamma == 1 else
                         (c_arr[feasible] ** (1 - gamma) - 1) / (1 - gamma))
        return val

    def marg_u(c_arr: np.ndarray) -> np.ndarray:
        return c_arr ** (-gamma)

    # Flat choice grid over (b', k'): total n_choices = n_b * n_k
    n_choices = n_b * n_k
    b_next_flat = np.repeat(b_grid, n_k)
    k_next_flat = np.tile(k_grid, n_b)
    k_next_idx = np.tile(np.arange(n_k), n_b)
    b_next_idx = np.repeat(np.arange(n_b), n_k)

    def _solve_inner(enforce_collateral: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, int, float]:
        v_inner = np.zeros((n_b, n_k, n_z), dtype=np.float64)
        policy_best = np.zeros((n_b, n_k, n_z), dtype=int)
        converged_inner = False
        iters_taken = 0

        for it in range(max_iter):
            iters_taken = it + 1
            ev = v_inner @ p_trans.T  # (n_b, n_k, n_z)
            ev_flat = ev[b_next_idx, k_next_idx, :]  # (n_choices, n_z)
            v_new = np.zeros_like(v_inner)
            u_opt = np.zeros_like(v_inner)

            for iz in range(n_z):
                y_cur = z_grid[iz] * (k_grid ** alpha)
                for ik in range(n_k):
                    i_choices = i_mat[ik, k_next_idx]  # (n_choices,)
                    col_lim = collateral_limit[ik, k_next_idx]  # (n_choices,)

                    if enforce_collateral:
                        valid = (b_next_flat <= col_lim + 1e-8)[None, :]
                    else:
                        valid = np.ones((n_b, n_choices), dtype=bool)

                    # Vectorized across all debt states ib
                    c_cand = (y_cur[ik] - i_choices)[None, :] - (1.0 + r_star) * b_grid[:, None] + b_next_flat[None, :]
                    c_cand = np.where(valid, c_cand, -1.0)
                    if np.any(~np.any(c_cand > 0, axis=1)):
                        raise ValueError('grid contains a state with no feasible consumption and collateral choice')
                    w_cand = u_func(c_cand) + beta * ev_flat[None, :, iz]

                    best_idx = np.argmax(w_cand, axis=1)
                    v_new[:, ik, iz] = np.take_along_axis(w_cand, best_idx[:, None], axis=1).squeeze(axis=1)
                    policy_best[:, ik, iz] = best_idx
                    c_chosen = np.take_along_axis(c_cand, best_idx[:, None], axis=1).squeeze(axis=1)
                    u_opt[:, ik, iz] = u_func(c_chosen)

            diff = float(np.max(np.abs(v_new - v_inner)))
            v_inner = v_new
            if diff < tol:
                converged_inner = True
                break

            # Howard policy evaluation acceleration
            b_target = b_next_idx[policy_best]
            k_target = k_next_idx[policy_best]
            for _ in range(15):
                ev_h = v_inner @ p_trans.T
                ev_opt = ev_h[b_target, k_target, np.arange(n_z)[None, None, :]]
                v_inner = u_opt + beta * ev_opt

        pb = b_next_flat[policy_best]
        pk = k_next_flat[policy_best]
        pc = np.zeros((n_b, n_k, n_z), dtype=np.float64)
        pq = np.zeros((n_b, n_k, n_z), dtype=np.float64)

        for iz in range(n_z):
            y_cur = z_grid[iz] * (k_grid ** alpha)
            for ik in range(n_k):
                k_val = k_grid[ik]
                for ib in range(n_b):
                    k_next_val = pk[ib, ik, iz]
                    b_next_val = pb[ib, ik, iz]
                    inv = k_next_val - (1.0 - delta_k) * k_val + 0.5 * phi * (k_next_val - k_val) ** 2 / k_val
                    q_val = 1.0 + phi * (k_next_val - k_val) / k_val
                    c_val = y_cur[ik] - (1.0 + r_star) * b_grid[ib] + b_next_val - inv
                    pc[ib, ik, iz] = c_val
                    pq[ib, ik, iz] = q_val

        # Independent final Bellman evaluation at the returned value function;
        # Howard iteration counts alone do not certify convergence.
        residual = 0.
        ev_final = (v_inner @ p_trans.T)[b_next_idx, k_next_idx, :]
        for iz in range(n_z):
            for ik in range(n_k):
                c = (z_grid[iz] * k_grid[ik] ** alpha - i_mat[ik, k_next_idx])[None, :] \
                    - (1 + r_star) * b_grid[:, None] + b_next_flat[None, :]
                if enforce_collateral:
                    c = np.where(b_next_flat[None, :] <= collateral_limit[ik, k_next_idx][None, :] + 1e-8, c, -1.)
                best = np.max(u_func(c) + beta * ev_final[None, :, iz], axis=1)
                residual = max(residual, float(np.max(np.abs(best - v_inner[:, ik, iz]))))
        return pb, pk, pc, pq, v_inner, converged_inner and residual <= tol, iters_taken, residual

    # Solve constrained model
    pb_c, pk_c, pc_c, pq_c, v_c, conv_c, iters_c, residual_c = _solve_inner(enforce_collateral=True)

    # Solve unconstrained model for amplification benchmark
    pb_u, pk_u, pc_u, pq_u, v_u, conv_u, _, residual_u = _solve_inner(enforce_collateral=False)

    # On a discrete debt grid, binding means the next higher debt node is
    # excluded by collateral, at the chosen capital. The Euler residual is
    # only a grid approximation to a multiplier, not an imposed positive value.
    capacity = kappa_collateral * pq_c * pk_c
    debt_step = b_grid[1] - b_grid[0]
    binding = (pb_c + debt_step > capacity + 1e-8) & (capacity < b_grid[-1] + 1e-8)
    mu_lambda = np.zeros((n_b, n_k, n_z), dtype=np.float64)
    euler_residual = np.zeros_like(mu_lambda)
    u_prime_c = marg_u(pc_c)

    for iz in range(n_z):
        for ik in range(n_k):
            for ib in range(n_b):
                b_next_val = pb_c[ib, ik, iz]
                k_next_val = pk_c[ib, ik, iz]
                ib_next = int(np.argmin(np.abs(b_grid - b_next_val)))
                ik_next = int(np.argmin(np.abs(k_grid - k_next_val)))
                eu_prime_next = float(np.dot(p_trans[iz], u_prime_c[ib_next, ik_next, :]))
                lam = float(u_prime_c[ib, ik, iz] - beta * (1.0 + r_star) * eu_prime_next)
                euler_residual[ib, ik, iz] = lam
                mu_lambda[ib, ik, iz] = max(0., lam) if binding[ib, ik, iz] else 0.

    metadata: dict[str, Any] = {
        "k_ss": k_ss,
        "model_scope": "illustrative_adjustment_cost_collateral_planner",
        "value_function": v_c,
        "bellman_residual": residual_c,
        "unconstrained_bellman_residual": residual_u,
        "unconstrained_converged": conv_u,
        "grid_binding_mask": binding,
        "collateral_slack": capacity - pb_c,
        "debt_euler_residual": euler_residual,
        "multiplier_interpretation": "positive_debt_euler_residual_at_grid_binding_states",
        "unconstrained_policy_b": pb_u,
        "unconstrained_policy_c": pc_u,
        "simulation_periods": 2500, "simulation_burn": 500, "simulation_seed": seed,
        "crisis_definition": "grid collateral binding, falling consumption and increasing current account",
        "amplification_definition": "ratio of mean mid-to-low-productivity consumption drops at identical simulated debt/capital states",
        "amplification_status": "undefined_zero_benchmark_response",
        "parameters": dict(beta=beta, gamma=gamma, r_star=r_star, alpha=alpha,
                            delta_k=delta_k, phi=phi, rho_z=rho_z, sigma_z=sigma_z,
                            m_tauchen=m_tauchen, tol=tol, max_iter=max_iter),
    }

    result = SuddenStopResult(
        k_grid=k_grid,
        b_grid=b_grid,
        z_grid=z_grid,
        policy_k_prime=pk_c,
        policy_b_prime=pb_c,
        policy_c=pc_c,
        policy_q_asset=pq_c,
        collateral_binding_fraction=float(binding.mean()),
        multiplier_lambda=mu_lambda,
        policy_b=pb_c,
        policy_k=pk_c,
        asset_price_q=pq_c,
        kappa_collateral=kappa_collateral,
        crisis_probability=0.,
        amplification_ratio=float('nan'),
        converged=conv_c and conv_u,
        iterations=iters_c,
        p_trans=p_trans,
        metadata=metadata,
    )
    sample = _simulate_sudden_stop(result, 2500, seed).iloc[500:]
    crisis_probability = float(sample['crisis'].mean())
    mean_capacity = float(sample['collateral_capacity'].mean())
    mean_new_debt = float(sample['b_next'].mean())
    metadata.update(mean_debt=float(sample['b'].mean()), mean_new_debt=mean_new_debt,
                    mean_capital=float(sample['k'].mean()), mean_tobin_q=float(sample['tobin_q'].mean()),
                    mean_collateral_capacity=mean_capacity,
                    precautionary_buffer_pct=100 * (mean_capacity - mean_new_debt) / mean_capacity,
                    simulated_binding_frequency=float(sample['collateral_binding'].mean()))
    ib = np.searchsorted(b_grid, sample['b'].to_numpy())
    ik = np.searchsorted(k_grid, sample['k'].to_numpy())
    weights = np.bincount(ib * n_k + ik, minlength=n_b * n_k).reshape(n_b, n_k) / len(sample)
    mid, low = n_z // 2, 0
    drop_c = float(np.sum(weights * (pc_c[:, :, mid] - pc_c[:, :, low])))
    drop_u = float(np.sum(weights * (pc_u[:, :, mid] - pc_u[:, :, low])))
    metadata.update(amplification_weights=weights, constrained_consumption_drop=drop_c,
                    unconstrained_consumption_drop=drop_u,
                    amplification_productivity_states=[float(z_grid[mid]), float(z_grid[low])])
    amplification_ratio = float('nan')
    if drop_u > 1e-12 and result.converged:
        amplification_ratio = drop_c / drop_u
        metadata['amplification_status'] = 'computed'
    elif not result.converged:
        metadata['amplification_status'] = 'undefined_unconverged_model'
    elif drop_u < -1e-12:
        metadata['amplification_status'] = 'undefined_benchmark_consumption_increases'
    return replace(result, crisis_probability=crisis_probability, amplification_ratio=amplification_ratio)


def _simulate_sudden_stop(result: SuddenStopResult, periods: int, seed: int) -> pd.DataFrame:
    """Shared path engine: crises and grid binding are distinct observations."""
    if isinstance(periods, bool) or int(periods) != periods or periods < 1:
        raise ValueError('periods must be a positive integer')
    rng = np.random.default_rng(seed)
    nb, nk, nz = len(result.b_grid), len(result.k_grid), len(result.z_grid)
    ib, ik, iz = nb // 4, nk // 2, nz // 2
    binding = result.metadata.get('grid_binding_mask')
    previous_c, previous_ca = None, None
    rows = []
    for t in range(periods):
        iz = int(rng.choice(nz, p=result.p_trans[iz]))
        state = (ib, ik, iz)
        b, k = float(result.b_grid[ib]), float(result.k_grid[ik])
        bp, kp = float(result.policy_b[state]), float(result.policy_k[state])
        c, q, multiplier = float(result.policy_c[state]), float(result.asset_price_q[state]), float(result.multiplier_lambda[state])
        capacity = result.kappa_collateral * q * kp
        is_binding = bool(binding[state]) if binding is not None else bool(multiplier > 0)
        current_account = b - bp
        crisis = bool(is_binding and previous_c is not None and previous_ca is not None
                      and c < previous_c and current_account > previous_ca)
        rows.append(dict(period=t, z=float(result.z_grid[iz]), b=b, debt=b, b_next=bp,
                         k=k, k_next=kp, consumption=c, c=c, tobin_q=q,
                         multiplier=multiplier, multiplier_lambda=multiplier, crisis=crisis,
                         collateral_binding=is_binding, collateral_capacity=capacity,
                         collateral_slack=capacity - bp, current_account=current_account))
        previous_c, previous_ca = c, current_account
        ib, ik = int(np.searchsorted(result.b_grid, bp)), int(np.searchsorted(result.k_grid, kp))
    return pd.DataFrame(rows)
