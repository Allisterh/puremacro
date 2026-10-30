"""Verbatim 6.0.1 simulator, independent of the current result class.

Source: c7002ecd:puremacro/sovereign/_results.py, SovereignDefaultResult.simulate.
Only the top-level function name and indentation differ. The same-platform
comparison detects single-ulp behavioral changes without requiring bitwise
libm/BLAS agreement across operating systems or NumPy releases.
Original dedented method SHA256: 5461f5fbe23fc94214ce373993fd5f18a4b89282fb1827e01fa9421369f746fb
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def legacy_simulate(
    self,
    T: int | None = None,
    periods: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate Markov sample path of the sovereign economy.

    Parameters
    ----------
    T : int, optional
        Number of simulation quarters (synonym for ``periods``).
    periods : int, optional
        Number of simulation periods. Defaults to 1000 if both are None.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Simulated paths for endowment, debt, consumption, bond prices,
        spreads, and default indicators. The path starts at the zero-debt
        node (nearest node if the grid omits zero). Reentry returns to that
        node before the current period's default/repayment decision.
    """
    n_periods = 1000
    if periods is not None:
        n_periods = int(periods)
    elif T is not None:
        n_periods = int(T)

    rng = np.random.default_rng(seed)
    n_y = len(self.y_grid)
    delta = self.maturity_param_delta
    kappa = self.coupon_rate_kappa
    theta = float(self.metadata.get("theta", 0.282))
    y_def_vec = self.metadata.get("y_default_vec", 0.969 * self.y_grid)

    y_idx = n_y // 2
    zero_idx = int(np.argmin(np.abs(self.b_grid)))
    b_idx = zero_idx
    in_default = False

    records = []
    for t in range(n_periods):
        # Draw next endowment state
        y_idx = int(rng.choice(n_y, p=self.p_trans[y_idx]))
        y_val = float(self.y_grid[y_idx])

        if in_default:
            if rng.random() < theta:
                in_default = False
                b_idx = zero_idx
            else:
                c_val = float(y_def_vec[y_idx]) if hasattr(y_def_vec, "__getitem__") else float(y_val * 0.969)
                records.append({
                    "period": t,
                    "y": y_val,
                    "b": 0.0,
                    "debt": 0.0,
                    "b_next": 0.0,
                    "default": True,
                    "repaid": False,
                    "bond_price": 0.0,
                    "q": 0.0,
                    "spread": float(np.nan),
                    "spread_bps": float(np.nan),
                    "consumption": c_val,
                    "c": c_val,
                })
                continue

        # Sovereign chooses whether to default
        p_def = float(self.default_prob[b_idx, y_idx])
        if rng.random() < p_def:
            in_default = True
            c_val = float(y_def_vec[y_idx]) if hasattr(y_def_vec, "__getitem__") else float(y_val * 0.969)
            records.append({
                "period": t,
                "y": y_val,
                "b": float(self.b_grid[b_idx]),
                "debt": float(self.b_grid[b_idx]),
                "b_next": 0.0,
                "default": True,
                "repaid": False,
                "bond_price": 0.0,
                "q": 0.0,
                "spread": float(np.nan),
                "spread_bps": float(np.nan),
                "consumption": c_val,
                "c": c_val,
            })
            continue

        # Sovereign repays and issues new debt
        b_next_val = float(self.policy_b[b_idx, y_idx])
        b_cur_val = float(self.b_grid[b_idx])

        # Find nearest grid index for next debt
        b_next_idx = int(np.argmin(np.abs(self.b_grid - b_next_val)))
        q_val = float(self.bond_price[b_next_idx, y_idx])
        spread_val = float(self.spread[b_next_idx, y_idx])

        c_val = y_val - (delta + (1.0 - delta) * kappa) * b_cur_val + q_val * (b_next_val - (1.0 - delta) * b_cur_val)
        c_val = max(1e-5, c_val)

        records.append({
            "period": t,
            "y": y_val,
            "b": b_cur_val,
            "debt": b_cur_val,
            "b_next": b_next_val,
            "default": False,
            "repaid": True,
            "bond_price": q_val,
            "q": q_val,
            "spread": spread_val,
            "spread_bps": spread_val * 10000.0,
            "consumption": c_val,
            "c": c_val,
        })
        b_idx = b_next_idx

    return pd.DataFrame(records)
