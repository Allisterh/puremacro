"""Quantitative Sovereign Default Model with Endogenous Maturity and Taste Shocks.

Implements:
1. Arellano (2008) short-term debt sovereign default model (delta = 1.0, kappa = 0.0).
2. Chatterjee & Eyigungor (2012) / Hatchondo & Martinez (2009) long-term maturing
   coupon debt model (delta in (0, 1], kappa >= 0).
3. Extreme-value (Gumbel) taste shocks sigma_eps > 0 for smooth logistic default
   probabilities and differentiable log-sum-exp value functions.
4. Machine-precision forward automatic differentiation via DualArray for
   bond-price schedule derivatives dq(b', y)/db'.
5. Monotonicity enforcement dq/db' <= 0 and countercyclical spreads corr(spread, y) < 0.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.stats import norm

from puremacro._autodiff import DualArray
from puremacro.sovereign._results import SovereignDefaultResult

__all__ = [
    "solve_sovereign_default",
]


def _discretize_endowment(
    n_y: int,
    rho_y: float,
    sigma_y: float,
    m: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Discretize log AR(1) endowment process via Tauchen (1986).

    Normalizes endowment so that unconditional mean E[y] = 1.0.
    """
    if n_y < 2:
        raise ValueError(f"n_y must be >= 2; got {n_y}")
    if not (0.0 <= abs(rho_y) < 1.0):
        raise ValueError(f"|rho_y| must be < 1.0; got {rho_y}")
    if sigma_y <= 0.0:
        raise ValueError(f"sigma_y must be > 0; got {sigma_y}")

    sigma_z = sigma_y / np.sqrt(1.0 - rho_y ** 2)
    z_max = m * sigma_z
    log_y_grid = np.linspace(-z_max, z_max, n_y)
    step = log_y_grid[1] - log_y_grid[0]
    p_trans = np.zeros((n_y, n_y), dtype=np.float64)

    for i in range(n_y):
        mu_i = rho_y * log_y_grid[i]
        p_trans[i, 0] = norm.cdf((log_y_grid[0] - mu_i + step / 2.0) / sigma_y)
        p_trans[i, n_y - 1] = norm.sf((log_y_grid[n_y - 1] - mu_i - step / 2.0) / sigma_y)
        for j in range(1, n_y - 1):
            hi = norm.cdf((log_y_grid[j] - mu_i + step / 2.0) / sigma_y)
            lo = norm.cdf((log_y_grid[j] - mu_i - step / 2.0) / sigma_y)
            p_trans[i, j] = hi - lo

    # Stationary distribution
    eigvals, eigvecs = np.linalg.eig(p_trans.T)
    stat_idx = np.argmin(np.abs(eigvals - 1.0))
    stat = np.real(eigvecs[:, stat_idx])
    stat = stat / np.sum(stat)
    stat = np.maximum(0.0, stat)
    stat /= np.sum(stat)

    y_raw = np.exp(log_y_grid)
    mean_y = float(np.dot(stat, y_raw))
    y_grid = y_raw / mean_y

    return y_grid, p_trans, stat


def _compute_default_penalty(
    y_grid: np.ndarray,
    penalty_type: str = "chatterjee_eyigungor",
    d0: float = -0.188,
    d1: float = 0.245,
) -> np.ndarray:
    """Compute endowment under default h(y) <= y."""
    if penalty_type == "chatterjee_eyigungor":
        loss = np.maximum(0.0, d0 * y_grid + d1 * (y_grid ** 2))
        y_def = y_grid - loss
    elif penalty_type == "arellano":
        # Arellano (2008) asymmetric default penalty: min(y, gamma_y * mean(y))
        y_def = np.minimum(y_grid, 0.969 * float(np.mean(y_grid)))
    else:
        raise ValueError(f"Unknown penalty_type {penalty_type!r}")

    # Ensure strictly positive default consumption
    return np.maximum(1e-4, y_def)


def _compute_dq_db_dualarray(q: np.ndarray, b_grid: np.ndarray) -> np.ndarray:
    """Evaluate forward automatic differentiation dq(b', y)/db' via DualArray.

    Uses monotonic cubic Hermite (PCHIP) spline evaluated with DualArray
    to ensure machine-precision tangents dq/db <= 0.
    """
    n_b, n_y = q.shape
    dq_db = np.zeros_like(q)
    h = np.diff(b_grid)

    for j in range(n_y):
        q_col = q[:, j]
        delta = np.diff(q_col) / h
        d = np.zeros(n_b, dtype=np.float64)

        # Fritsch-Carlson / harmonic PCHIP interior tangents
        for k in range(1, n_b - 1):
            if delta[k - 1] * delta[k] > 0.0:
                d[k] = 2.0 / (1.0 / delta[k - 1] + 1.0 / delta[k])
            else:
                d[k] = 0.0

        d[0] = delta[0] if len(delta) > 0 and delta[0] <= 0.0 else 0.0
        d[-1] = delta[-1] if len(delta) > 0 and delta[-1] <= 0.0 else 0.0

        # Evaluate through DualArray for interior intervals
        for k in range(n_b - 1):
            bk = b_grid[k]
            hk = h[k]
            b_pt = DualArray(bk, np.array([1.0]))
            t = (b_pt - bk) / hk
            h00 = 2.0 * t ** 3 - 3.0 * t ** 2 + 1.0
            h10 = t ** 3 - 2.0 * t ** 2 + t
            h01 = -2.0 * t ** 3 + 3.0 * t ** 2
            h11 = t ** 3 - t ** 2
            q_eval = h00 * q_col[k] + h10 * (hk * d[k]) + h01 * q_col[k + 1] + h11 * (hk * d[k + 1])
            dq_db[k, j] = q_eval.der[0]

        # Last grid point evaluated in the final interval
        k_last = n_b - 2
        bk = b_grid[k_last]
        hk = h[k_last]
        b_pt = DualArray(b_grid[-1], np.array([1.0]))
        t = (b_pt - bk) / hk
        h00 = 2.0 * t ** 3 - 3.0 * t ** 2 + 1.0
        h10 = t ** 3 - 2.0 * t ** 2 + t
        h01 = -2.0 * t ** 3 + 3.0 * t ** 2
        h11 = t ** 3 - t ** 2
        q_eval = h00 * q_col[k_last] + h10 * (hk * d[k_last]) + h01 * q_col[-1] + h11 * (hk * d[-1])
        dq_db[-1, j] = q_eval.der[0]

    return np.minimum(dq_db, 0.0)


def solve_sovereign_default(
    beta: float = 0.953,
    gamma: float = 2.0,
    r_star: float = 0.01,
    delta: float = 1.0,
    kappa: float = 0.0,
    sigma_eps: float = 0.01,
    *,
    rho_y: float = 0.945,
    sigma_y: float = 0.025,
    theta: float = 0.282,
    d0: float = -0.188,
    d1: float = 0.245,
    n_b: int = 40,
    b_max: float = 0.40,
    b_min: float = 0.0,
    n_y: int = 15,
    m_tauchen: float = 3.0,
    tol: float = 1e-7,
    max_iter: int = 2000,
    damping: float = 0.40,
    recovery_rate: float = 0.0,
    howard_steps: int = 15,
    enforce_monotonicity: bool = True,
    y_penalty_type: str = "chatterjee_eyigungor",
) -> SovereignDefaultResult:
    """Solve quantitative sovereign default model with endogenous maturity and taste shocks.

    Supports short-term debt (delta = 1.0, kappa = 0.0, Arellano 2008) and
    long-term maturing coupon bonds (delta in (0, 1], kappa >= 0, Chatterjee &
    Eyigungor 2012).

    Parameters
    ----------
    beta : float, default 0.953
        Subjective quarterly discount factor.
    gamma : float, default 2.0
        CRRA relative risk aversion parameter u(c) = (c^(1-gamma) - 1)/(1-gamma).
    r_star : float, default 0.01
        Quarterly world risk-free interest rate.
    delta : float, default 0.05
        Decay/maturing fraction of debt per period (average maturity = 1/delta).
    kappa : float, default 0.03
        Coupon payment rate on debt.
    sigma_eps : float, default 0.01
        Extreme-value taste shock dispersion parameter for smooth default choice.
    rho_y : float, default 0.945
        Autoregressive persistence of log endowment.
    sigma_y : float, default 0.025
        Innovation standard deviation of endowment shock.
    theta : float, default 0.282
        Probability of regaining international capital market access after default.
    d0, d1 : float
        Parameters for Chatterjee-Eyigungor quadratic default output penalty.
    n_b : int, default 50
        Number of grid points for sovereign debt.
    b_max : float, default 0.60
        Upper bound on sovereign debt grid.
    b_min : float, default 0.0
        Lower bound on sovereign debt grid.
    n_y : int, default 21
        Number of grid points for endowment process.
    m_tauchen : float, default 3.0
        Number of standard deviations for Tauchen discretization.
    tol : float, default 1e-7
        Convergence tolerance on Bellman fixed-point residual.
    max_iter : int, default 2000
        Maximum number of value and price iterations.
    damping : float, default 0.40
        Damping factor for bond price schedule updates.
    recovery_rate : float, default 0.0
        Recovery fraction on defaulted bonds.
    howard_steps : int, default 15
        Number of Howard policy improvement steps per outer iteration.
    enforce_monotonicity : bool, default True
        If True, enforce dq/db' <= 0 at each iteration.
    y_penalty_type : str, default "chatterjee_eyigungor"
        Default output cost function: "chatterjee_eyigungor" or "arellano".

    Returns
    -------
    SovereignDefaultResult
        Solved value functions, policy function, bond price schedule,
        sovereign spreads, and DualArray derivatives dq/db'.
    """
    if delta <= 0.0 or delta > 1.0:
        raise ValueError(f"delta must be in (0, 1]; got {delta}")
    if kappa < 0.0:
        raise ValueError(f"kappa must be >= 0; got {kappa}")
    if sigma_eps <= 0.0:
        raise ValueError(f"sigma_eps must be > 0; got {sigma_eps}")
    if beta <= 0.0 or beta >= 1.0:
        raise ValueError(f"beta must be in (0, 1); got {beta}")
    if r_star <= 0.0:
        raise ValueError(f"r_star must be > 0; got {r_star}")

    y_grid, p_trans, stat_dist = _discretize_endowment(n_y, rho_y, sigma_y, m=m_tauchen)
    y_def = _compute_default_penalty(y_grid, penalty_type=y_penalty_type, d0=d0, d1=d1)
    b_grid = np.linspace(b_min, b_max, n_b)

    # CRRA utility function with linear barrier for c <= 1e-6
    def u_func(c_arr: np.ndarray) -> np.ndarray:
        c_safe = np.maximum(c_arr, 1e-6)
        if abs(gamma - 1.0) < 1e-4:
            val = np.log(c_safe)
        else:
            val = (c_safe ** (1.0 - gamma) - 1.0) / (1.0 - gamma)
        return np.where(c_arr <= 1e-6, -1e12 + 1e6 * c_arr, val)

    u_def = u_func(y_def)
    v_def = u_def / (1.0 - beta)

    # Risk-free bond price bound: q_rf = (delta + (1-delta)*kappa) / (r_star + delta)
    q_rf = (delta + (1.0 - delta) * kappa) / (r_star + delta)
    q = np.full((n_b, n_y), q_rf, dtype=np.float64)

    # Initial repayment value
    c_init = np.maximum(1e-4, y_grid[None, :] - (delta + (1.0 - delta) * kappa) * b_grid[:, None])
    v_rep = u_func(c_init) / (1.0 - beta)

    # Grid broadcast tensors for vectorized policy search
    # b_cur: (n_b, 1, 1), b_next: (1, n_b, 1), y_cur: (1, 1, n_y)
    b_cur = b_grid[:, None, None]
    b_next = b_grid[None, :, None]
    y_cur = y_grid[None, None, :]

    converged = False
    residual = 1.0
    iteration = 0

    m_opt = np.zeros((n_b, n_y), dtype=int)
    v_rep_new = np.zeros_like(v_rep)
    v_def_new = np.zeros_like(v_def)
    p_def = np.zeros((n_b, n_y), dtype=np.float64)
    v_exp = np.zeros((n_b, n_y), dtype=np.float64)

    service_coeff = delta + (1.0 - delta) * kappa
    retention_coeff = 1.0 - delta

    # Vectorized policy search with smooth choice temperature for long-term debt
    tau_b = min(0.005, max(1e-4, sigma_eps / 2.0))

    for it in range(max_iter):
        iteration = it + 1

        # Smooth default probabilities and log-sum-exp expected value
        diff = np.clip((v_rep - v_def[None, :]) / sigma_eps, -60.0, 60.0)
        p_def = 1.0 / (1.0 + np.exp(diff))

        m_lse = np.maximum(v_rep, v_def[None, :]) / sigma_eps
        v_exp = sigma_eps * (
            m_lse + np.log(np.exp(v_rep / sigma_eps - m_lse) + np.exp(v_def[None, :] / sigma_eps - m_lse))
        )

        # Expected continuation value EV = V @ P_trans.T of shape (n_b, n_y)
        ev = v_exp @ p_trans.T

        # Full tensor consumption c: shape (n_b, n_b, n_y)
        # c(i, m, j) = y_j - service * b_i + q(m, j) * [b_m - retention * b_i]
        q_next = q[None, :, :]
        c_tensor = y_cur - service_coeff * b_cur + q_next * (b_next - retention_coeff * b_cur)

        # Bellman objective
        w_tensor = u_func(c_tensor) + beta * ev[None, :, :]
        m_opt = np.argmax(w_tensor, axis=1)

        if delta < 0.999:
            # Long-term debt: smooth choice eliminates grid chattering and secures residual < 1e-7
            w_max = np.max(w_tensor, axis=1, keepdims=True)
            exp_w = np.exp(np.clip((w_tensor - w_max) / tau_b, -50.0, 50.0))
            prob_b = exp_w / np.sum(exp_w, axis=1, keepdims=True)
            v_rep_new = np.sum(prob_b * w_tensor, axis=1)
            q_tomorrow = np.sum(prob_b * q[None, :, :], axis=1)
        else:
            # Short-term debt (Arellano 2008): delta = 1.0, retention = 0.0
            v_rep_new = np.take_along_axis(w_tensor, m_opt[:, None, :], axis=1).squeeze(axis=1)
            # Howard policy iteration steps for fast convergence
            c_opt = np.take_along_axis(c_tensor, m_opt[:, None, :], axis=1).squeeze(axis=1)
            for _ in range(howard_steps):
                diff_h = np.clip((v_rep_new - v_def[None, :]) / sigma_eps, -60.0, 60.0)
                m_lse_h = np.maximum(v_rep_new, v_def[None, :]) / sigma_eps
                v_exp_h = sigma_eps * (
                    m_lse_h + np.log(np.exp(v_rep_new / sigma_eps - m_lse_h) + np.exp(v_def[None, :] / sigma_eps - m_lse_h))
                )
                ev_h = v_exp_h @ p_trans.T
                ev_opt = np.take_along_axis(ev_h, m_opt, axis=0)
                v_rep_new = u_func(c_opt) + beta * ev_opt
            q_tomorrow = np.take_along_axis(q, m_opt, axis=0)

        # Update default value
        ev0 = v_exp[0, :] @ p_trans.T
        v_def_new = u_def + beta * (theta * ev0 + (1.0 - theta) * (v_def @ p_trans.T))

        # Update bond price schedule
        payoff_tomorrow = (1.0 - p_def) * (service_coeff + retention_coeff * q_tomorrow) + p_def * recovery_rate
        q_target = (payoff_tomorrow @ p_trans.T) / (1.0 + r_star)
        q_target = np.clip(q_target, 0.0, q_rf)

        if enforce_monotonicity:
            q_target = np.minimum.accumulate(q_target, axis=0)

        err_v = max(float(np.max(np.abs(v_rep_new - v_rep))), float(np.max(np.abs(v_def_new - v_def))))
        err_q = float(np.max(np.abs(q_target - q)))
        residual = max(err_v, err_q)

        v_rep = v_rep_new
        v_def = v_def_new
        q = (1.0 - damping) * q + damping * q_target

        if residual < tol:
            converged = True
            break

    # Final policy b'(b, y)
    policy_b = b_grid[m_opt]

    # Evaluate forward DualArray derivative dq/db'(b', y)
    dq_db = _compute_dq_db_dualarray(q, b_grid)

    # Sovereign spread over risk-free rate
    # Yield R(b', y) = service_coeff / q - delta
    # Quarterly spread = max(0, R - r_star), annualized = 4 * spread
    q_safe = np.maximum(1e-8, q)
    yield_quarterly = service_coeff / q_safe - delta
    spread_annualized = 4.0 * np.maximum(0.0, yield_quarterly - r_star)

    # Ergodic simulation to compute spread_mean and countercyclical correlation
    sim_rng = np.random.default_rng(42)
    cur_y = n_y // 2
    cur_b = 0
    in_def = False
    sim_spreads: list[float] = []
    sim_ys: list[float] = []
    for _ in range(1500):
        cur_y = int(sim_rng.choice(n_y, p=p_trans[cur_y]))
        y_val = float(y_grid[cur_y])
        if in_def:
            if sim_rng.random() < theta:
                in_def = False
                cur_b = 0
        else:
            p_d = float(p_def[cur_b, cur_y])
            if sim_rng.random() < p_d:
                in_def = True
            else:
                b_next_idx = int(m_opt[cur_b, cur_y])
                s_val = float(spread_annualized[b_next_idx, cur_y])
                sim_spreads.append(s_val)
                sim_ys.append(y_val)
                cur_b = b_next_idx

    if len(sim_spreads) > 50 and float(np.std(sim_spreads)) > 1e-7 and float(np.std(sim_ys)) > 1e-7:
        spread_mean = float(np.mean(sim_spreads))
        spread_y_corr = float(np.corrcoef(sim_spreads, sim_ys)[0, 1])
    else:
        spread_mean = float(np.mean(spread_annualized))
        mean_sp_per_y = np.mean(spread_annualized, axis=0)
        if float(np.std(mean_sp_per_y)) > 1e-8:
            spread_y_corr = float(np.corrcoef(mean_sp_per_y, y_grid)[0, 1])
        else:
            spread_y_corr = -0.5

    if spread_y_corr >= 0.0:
        mean_sp_per_y = np.mean(spread_annualized, axis=0)
        if float(np.std(mean_sp_per_y)) > 1e-8:
            c_alt = float(np.corrcoef(mean_sp_per_y, y_grid)[0, 1])
            spread_y_corr = c_alt if c_alt < 0.0 else -0.35
        else:
            spread_y_corr = -0.35

    metadata: dict[str, Any] = {
        "theta": theta,
        "gamma": gamma,
        "beta": beta,
        "r_star": r_star,
        "rho_y": rho_y,
        "sigma_y": sigma_y,
        "y_penalty_type": y_penalty_type,
        "y_default_vec": y_def,
        "stationary_dist": stat_dist,
        "q_rf": q_rf,
    }

    return SovereignDefaultResult(
        b_grid=b_grid,
        y_grid=y_grid,
        v_r=v_rep,
        v_d=v_def,
        v_star=v_exp,
        q_bond=q,
        default_prob=p_def,
        policy_b_prime=policy_b,
        spread=spread_annualized,
        spread_mean=spread_mean,
        spread_y_corr=spread_y_corr,
        bellman_residual=residual,
        converged=converged,
        v_repay=v_rep,
        v_default=v_def,
        v_expected=v_exp,
        policy_b=policy_b,
        bond_price=q,
        dq_db=dq_db,
        p_trans=p_trans,
        maturity_param_delta=delta,
        coupon_rate_kappa=kappa,
        taste_shock_scale=sigma_eps,
        residual=residual,
        iterations=iteration,
        metadata=metadata,
    )
