"""Independent small-grid equations for sovereign and collateral extensions.

These references use NumPy/SciPy only, never production solver helpers.
Long-term debt checks the explicitly smoothed policy operator implemented by
PureMacro, not a replication of Chatterjee--Eyigungor published moments.
The collateral reference solves the stated adjustment-cost planning problem,
not Mendoza's full competitive model with working-capital financing.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit, logsumexp


def collateral(b, k, z, transition, *, beta=.96, gamma=2., r=.04,
               alpha=.32, depreciation=.088, adjustment=.08, collateral_share=.3,
               constrained=True, tolerance=1e-11):
    """Exhaustive value iteration over (b', k'), without Howard acceleration."""
    states = [(ib, ik) for ib in range(len(b)) for ik in range(len(k))]
    next_b = np.array([b[ib] for ib, _ in states])
    next_k = np.array([k[ik] for _, ik in states])
    flow = np.empty((len(states), len(states), len(z)))
    consumption = np.empty_like(flow)
    for row, (ib, ik) in enumerate(states):
        investment = next_k - (1 - depreciation) * k[ik] + adjustment / 2 * (next_k - k[ik]) ** 2 / k[ik]
        price = 1 + adjustment * (next_k - k[ik]) / k[ik]
        allowed = next_b <= collateral_share * price * next_k + 1e-8 if constrained else np.ones(len(states), bool)
        for j, productivity in enumerate(z):
            c = productivity * k[ik] ** alpha - (1 + r) * b[ib] + next_b - investment
            consumption[row, :, j] = c
            feasible = (c > 0) & allowed
            if not np.any(feasible):
                raise ValueError('no feasible action on the reference grid')
            utility = np.full(len(states), -np.inf)
            utility[feasible] = (np.log(c[feasible]) if gamma == 1 else
                                  (c[feasible] ** (1 - gamma) - 1) / (1 - gamma))
            flow[row, :, j] = utility
    value = np.zeros((len(states), len(z)))
    for _ in range(5000):
        objective = flow + beta * (value @ transition.T)[None, :, :]
        new = objective.max(axis=1)
        if np.max(np.abs(new - value)) <= tolerance:
            value = new
            break
        value = new
    else:
        raise RuntimeError('reference collateral value iteration did not converge')
    objective = flow + beta * (value @ transition.T)[None, :, :]
    policy = objective.argmax(axis=1)
    chosen_c = np.take_along_axis(consumption, policy[:, None, :], axis=1)[:, 0, :]
    shape = (len(b), len(k), len(z))
    return dict(value=value.reshape(shape), policy_b=next_b[policy].reshape(shape),
                policy_k=next_k[policy].reshape(shape), policy_c=chosen_c.reshape(shape),
                bellman_residual=float(np.max(np.abs(objective.max(axis=1) - value))))


def long_term(b, y, transition, default_income, *, beta=.953, gamma=2., r=.01,
              delta=.2, coupon=.03, default_scale=.01, reentry=.282,
              recovery=0., tolerance=1e-11):
    """Separate fixed point for the legacy smoothed long-term debt operator.

    Debt choice uses Boltzmann weights at tau=min(.005,max(.0001,sigma/2));
    its value is the weighted Bellman objective, without a taste-shock bonus.
    This is a quantal-policy approximation, not the deterministic maximum or
    the inclusive value of an entropy-regularized optimal-choice model.
    """
    service = delta + (1 - delta) * coupon
    rf_price = service / (r + delta)
    q = np.full((len(b), len(y)), rf_price)
    vr = np.zeros_like(q)
    vd = np.zeros(len(y))
    tau = min(.005, max(.0001, default_scale / 2))

    def utility(c):
        if np.any(c <= 0):
            raise ValueError('reference test requires a feasible consumption grid')
        return np.log(c) if gamma == 1 else (c ** (1 - gamma) - 1) / (1 - gamma)

    def operator(vr, vd, q):
        pd = (vd[None, :] > vr).astype(float) if default_scale == 0 else expit((vd[None, :] - vr) / default_scale)
        total = (np.maximum(vr, vd[None, :]) if default_scale == 0 else
                 default_scale * np.logaddexp(vr / default_scale, vd[None, :] / default_scale))
        ev = total @ transition.T
        repaid = np.empty_like(vr)
        continuation_price = np.empty_like(q)
        probabilities = np.empty((len(b), len(b), len(y)))
        modal = np.empty_like(vr, dtype=int)
        exact = np.empty_like(vr)
        for i, debt in enumerate(b):
            for j, income in enumerate(y):
                c = income - service * debt + q[:, j] * (b - (1 - delta) * debt)
                objective = utility(c) + beta * ev[:, j]
                modal[i, j] = objective.argmax()
                exact[i, j] = objective.max()
                if delta < .999:
                    logweight = np.maximum((objective - objective.max()) / tau, -50.)
                    weights = np.exp(logweight - logsumexp(logweight))
                else:
                    weights = np.zeros(len(b))
                    weights[modal[i, j]] = 1
                probabilities[i, :, j] = weights
                repaid[i, j] = weights @ objective
                continuation_price[i, j] = weights @ q[:, j]
        zero = np.array([np.interp(0, b, total[:, j]) for j in range(len(y))])
        defaulted = utility(default_income) + beta * (reentry * zero + (1 - reentry) * vd) @ transition.T
        price = ((1 - pd) * (service + (1 - delta) * continuation_price) + pd * recovery) @ transition.T / (1 + r)
        return repaid, defaulted, np.clip(price, 0, rf_price), probabilities, modal, total, pd, exact

    for iteration in range(10000):
        out = operator(vr, vd, q)
        error = max(np.max(np.abs(out[0] - vr)), np.max(np.abs(out[1] - vd)), np.max(np.abs(out[2] - q)))
        vr, vd, q = out[0], out[1], .6 * q + .4 * out[2]
        if error < tolerance:
            break
    else:
        raise RuntimeError('reference long-term fixed point did not converge')
    out = operator(vr, vd, q)
    return dict(v_repay=vr, v_default=vd, q=q, probabilities=out[3], modal_policy=b[out[4]],
                total=out[5], default=out[6], residual=float(error), iterations=iteration + 1,
                deterministic_optimality_gap=float(np.max(np.abs(out[7] - vr))))
