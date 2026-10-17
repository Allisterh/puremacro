"""Abadie, Diamond & Hainmueller (2010) Synthetic Control Method."""
from __future__ import annotations

from typing import Dict, Any
import numpy as np
import scipy.optimize


def synthetic_control(
    Y_matrix: np.ndarray,
    treated_idx: int,
    treat_time: int,
) -> Dict[str, Any]:
    """Synthetic Control Method for micro/macro policy evaluation.

    Donor weights are fitted on the pre-treatment rows by SLSQP under
    ``w >= 0`` and ``sum(w) == 1``; the synthetic series is then carried
    across the whole sample and the post-treatment gap averaged.

    Indexing convention
    -------------------
    ``treat_time`` is the **0-indexed row of the first treated period** —
    the same integer you would use to inject an effect with
    ``Y_matrix[treat_time:, treated_idx] += tau``. The two windows
    partition the sample with no gap and no overlap:

    - pre-treatment (fits the weights): rows ``[0, treat_time)``
    - post-treatment (averaged into ``att_post``): rows ``[treat_time, T)``

    So ``treat_time`` counts the pre-treatment rows. The MATLAB port in
    ``matlab/+puremacro/+causal/synthetic_control.m`` is 1-indexed:
    MATLAB ``treat_time = t`` is this function's ``treat_time = t - 1``.

    Parameters
    ----------
    Y_matrix : ndarray, shape (T, N_total)
        Panel of outcomes, rows are periods and columns are units.
    treated_idx : int
        Column index of the treated unit. Every other column is a donor.
    treat_time : int
        0-indexed row of the first treated period, ``1 <= treat_time < T``.
        At ``treat_time = 1`` the weights are fitted on a single row and
        are not identified; the pre-window is empty below that.

    Returns
    -------
    dict
        Six keys:

        ``weights`` : ndarray, shape (N_donors,)
            Fitted donor weights, ordered as the non-treated columns of
            ``Y_matrix``. Non-negative and summing to 1.
        ``y_treated`` : ndarray, shape (T,)
            The treated unit's actual outcome path, all ``T`` rows.
        ``y_synthetic`` : ndarray, shape (T,)
            The counterfactual ``Y_donors @ weights``, all ``T`` rows.
        ``att_path`` : ndarray, shape (T,)
            ``y_treated - y_synthetic`` period by period. Pre-treatment
            entries are the fit residuals, not effects.
        ``att_post`` : float
            Mean of ``att_path[treat_time:]`` — the post-treatment ATT.
        ``treat_time`` : int
            Echoed back unchanged, for plotting the treatment line.

    References
    ----------
    Abadie, A., Diamond, A. and Hainmueller, J. (2010). Synthetic control
        methods for comparative case studies. JASA 105(490), 493-505.
    """
    Y_matrix = np.asarray(Y_matrix, dtype=float)
    T, N_total = Y_matrix.shape

    donor_mask = np.ones(N_total, dtype=bool)
    donor_mask[treated_idx] = False

    y_treated = Y_matrix[:, treated_idx]
    Y_donors = Y_matrix[:, donor_mask]
    N_donors = Y_donors.shape[1]

    # treat_time is the 0-indexed first treated row, so it is also the
    # count of pre-treatment rows: [0, treat_time) fits, [treat_time, T)
    # is scored. Subtracting one here (as the 1-indexed MATLAB port must)
    # drops row treat_time - 1 out of both windows.
    T_pre = treat_time
    y_pre = y_treated[:T_pre]
    Y_donors_pre = Y_donors[:T_pre, :]

    def objective(w):
        return np.sum((y_pre - Y_donors_pre @ w) ** 2)

    w0 = np.ones(N_donors) / N_donors
    bounds = [(0.0, 1.0) for _ in range(N_donors)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    res_opt = scipy.optimize.minimize(objective, w0, bounds=bounds, constraints=constraints, method="SLSQP")
    w_opt = res_opt.x

    y_synthetic = Y_donors @ w_opt
    att_path = y_treated - y_synthetic
    att_post = float(np.mean(att_path[treat_time:]))

    return {
        "weights": w_opt,
        "y_treated": y_treated,
        "y_synthetic": y_synthetic,
        "att_path": att_path,
        "att_post": att_post,
        "treat_time": treat_time,
    }
