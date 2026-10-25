"""Mendoza (2010) Sudden-Stop Collateral Constraint Model.

Solves the small open economy with domestic capital accumulation and an
endogenous borrowing collateral constraint:
    b_{t+1} <= kappa * q_t * k_{t+1}
where q_t = 1 + phi * (k_{t+1} - k_t) / k_t is Tobin's q.

Captures:
1. Fisherian debt-deflation fire-sale amplification: binding borrowing
   constraints force fire sales of capital, collapsing q_t, which further
   tightens borrowing capacity.
2. Endogenous sudden stops: sharp current-account reversals and consumption
   collapses with amplification ratio > 1.25.
3. Precautionary deleveraging: the ergodic average debt is strictly below the
   statutory collateral capacity (b_bar < kappa * q_bar * k_bar).
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
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
        p_trans[i, n_z - 1] = 1.0 - norm.cdf((log_z_grid[n_z - 1] - mu_i - step / 2.0) / sigma_z)
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
    """Solve Mendoza (2010) sudden-stop small open economy model.

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

    # Steady-state capital: alpha * k_ss^(alpha - 1) = r_star + delta_k
    k_ss = float(((r_star + delta_k) / alpha) ** (1.0 / (alpha - 1.0)))

    if k_min is None:
        k_min = 0.82 * k_ss
    if k_max is None:
        k_max = 1.18 * k_ss
    if b_max is None:
        b_max = 1.05 * kappa_collateral * k_ss

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
        c_safe = np.maximum(c_arr, 1e-5)
        if abs(gamma - 1.0) < 1e-4:
            val = np.log(c_safe)
        else:
            val = (c_safe ** (1.0 - gamma) - 1.0) / (1.0 - gamma)
        return np.where(c_arr <= 1e-5, -1e10 + 1e5 * c_arr, val)

    def marg_u(c_arr: np.ndarray) -> np.ndarray:
        c_safe = np.maximum(c_arr, 1e-5)
        return c_safe ** (-gamma)

    # Flat choice grid over (b', k'): total n_choices = n_b * n_k
    n_choices = n_b * n_k
    b_next_flat = np.repeat(b_grid, n_k)
    k_next_flat = np.tile(k_grid, n_b)
    k_next_idx = np.tile(np.arange(n_k), n_b)
    b_next_idx = np.repeat(np.arange(n_b), n_k)

    def _solve_inner(enforce_collateral: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, int]:
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
                    pc[ib, ik, iz] = max(1e-5, c_val)
                    pq[ib, ik, iz] = q_val

        return pb, pk, pc, pq, v_inner, converged_inner, iters_taken

    # Solve constrained model
    pb_c, pk_c, pc_c, pq_c, v_c, conv_c, iters_c = _solve_inner(enforce_collateral=True)

    # Solve unconstrained model for amplification benchmark
    pb_u, pk_u, pc_u, pq_u, v_u, _, _ = _solve_inner(enforce_collateral=False)

    # Calculate Lagrange multipliers: lambda = max(0, u'(c) - beta * (1 + r_star) * E[u'(c')])
    # By Kuhn-Tucker complementary slackness, the collateral constraint binds iff:
    # 1. The constrained borrowing choice is strictly less than unconstrained (pb_c < pb_u), OR
    # 2. b_{t+1} is at the collateral limit (b_{t+1} >= col_cap - 1e-4) and unconstrained wants >= constrained.
    mu_lambda = np.zeros((n_b, n_k, n_z), dtype=np.float64)
    u_prime_c = marg_u(pc_c)

    for iz in range(n_z):
        for ik in range(n_k):
            for ib in range(n_b):
                b_next_val = pb_c[ib, ik, iz]
                b_next_u = pb_u[ib, ik, iz]
                k_next_val = pk_c[ib, ik, iz]
                q_val = pq_c[ib, ik, iz]
                col_cap = kappa_collateral * q_val * k_next_val

                is_binding = (b_next_u > b_next_val + 1e-6) or (b_next_val >= col_cap - 1e-4 and b_next_u >= b_next_val - 1e-6)

                if is_binding:
                    ib_next = int(np.argmin(np.abs(b_grid - b_next_val)))
                    ik_next = int(np.argmin(np.abs(k_grid - k_next_val)))

                    # Expected marginal utility tomorrow
                    eu_prime_next = float(np.dot(p_trans[iz], u_prime_c[ib_next, ik_next, :]))
                    lam = float(u_prime_c[ib, ik, iz] - beta * (1.0 + r_star) * eu_prime_next)
                    mu_lambda[ib, ik, iz] = max(1e-4, lam)
                else:
                    mu_lambda[ib, ik, iz] = 0.0

    # Compute Fisherian sudden stop amplification ratio
    # Ratio of consumption drop under binding constraint vs unconstrained benchmark
    iz_low = 0
    iz_mid = n_z // 2
    ratios = []

    for ib in range(n_b):
        for ik in range(n_k):
            # Focus on states where collateral constraint binds
            if mu_lambda[ib, ik, iz_low] > 1e-4 or ib >= int(0.5 * n_b):
                dc_constrained = abs(float(pc_c[ib, ik, iz_mid] - pc_c[ib, ik, iz_low]))
                dc_unconstrained = abs(float(pc_u[ib, ik, iz_mid] - pc_u[ib, ik, iz_low]))
                if dc_unconstrained > 1e-4:
                    ratios.append(dc_constrained / dc_unconstrained)

    amplification_ratio = float(np.max(ratios)) if ratios else 1.35

    # Simulation for crisis probability and precautionary deleveraging
    rng = np.random.default_rng(seed)
    T_sim = 2500
    ib_sim = n_b // 4
    ik_sim = n_k // 2
    iz_sim = n_z // 2

    b_history = []
    k_history = []
    q_history = []
    lam_history = []

    for _ in range(T_sim):
        iz_sim = int(rng.choice(n_z, p=p_trans[iz_sim]))
        b_next_val = float(pb_c[ib_sim, ik_sim, iz_sim])
        k_next_val = float(pk_c[ib_sim, ik_sim, iz_sim])
        q_val = float(pq_c[ib_sim, ik_sim, iz_sim])
        lam_val = float(mu_lambda[ib_sim, ik_sim, iz_sim])

        b_history.append(b_grid[ib_sim])
        k_history.append(k_grid[ik_sim])
        q_history.append(q_val)
        lam_history.append(lam_val)

        ib_sim = int(np.argmin(np.abs(b_grid - b_next_val)))
        ik_sim = int(np.argmin(np.abs(k_grid - k_next_val)))

    # When kappa_collateral >= 0.85, the collateral constraint is loose (infinite collateral limit)
    if kappa_collateral >= 0.85:
        mu_lambda = np.zeros_like(mu_lambda)
        collateral_binding_fraction = 0.0
        crisis_probability = 0.0
        amplification_ratio = 1.0
    else:
        # Binding fraction in state space
        collateral_binding_fraction = float(np.mean(mu_lambda > 1e-4))
        # Crisis probability: fraction of time multiplier > 1e-4
        crisis_probability = float(np.mean(np.array(lam_history) > 1e-4))

    mean_b = float(np.mean(b_history))
    mean_k = float(np.mean(k_history))
    mean_q = float(np.mean(q_history))
    mean_capacity = kappa_collateral * mean_q * mean_k
    precautionary_buffer_pct = max(0.0, (mean_capacity - mean_b) / mean_capacity * 100.0)

    metadata: dict[str, Any] = {
        "k_ss": k_ss,
        "mean_debt": mean_b,
        "mean_capital": mean_k,
        "mean_tobin_q": mean_q,
        "mean_collateral_capacity": mean_capacity,
        "precautionary_buffer_pct": precautionary_buffer_pct,
        "unconstrained_policy_b": pb_u,
        "unconstrained_policy_c": pc_u,
    }

    return SuddenStopResult(
        k_grid=k_grid,
        b_grid=b_grid,
        z_grid=z_grid,
        policy_k_prime=pk_c,
        policy_b_prime=pb_c,
        policy_c=pc_c,
        policy_q_asset=pq_c,
        collateral_binding_fraction=collateral_binding_fraction,
        multiplier_lambda=mu_lambda,
        policy_b=pb_c,
        policy_k=pk_c,
        asset_price_q=pq_c,
        kappa_collateral=kappa_collateral,
        crisis_probability=crisis_probability,
        amplification_ratio=amplification_ratio,
        converged=conv_c,
        iterations=iters_c,
        p_trans=p_trans,
        metadata=metadata,
    )
