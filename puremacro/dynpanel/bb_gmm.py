"""Blundell-Bond (1998) system GMM estimator.

Extends Arellano-Bond difference GMM by stacking a level equation
with additional moment conditions: lagged FIRST DIFFERENCES instrument
the level regressor in the level equation. This restores identification
when the autoregressive root approaches 1, where AB difference GMM
becomes weak (Blundell-Bond 1998 sections 3-4).

References
----------
Blundell, R. and Bond, S. (1998). Initial conditions and moment
    restrictions in dynamic panel data models. Journal of Econometrics
    87(1), 115-143.
Arellano, M. and Bover, O. (1995). Another look at the instrumental
    variable estimation of error-components models. Journal of
    Econometrics 68(1), 29-51.
Roodman, D. (2009). How to do xtabond2: an introduction to difference
    and system GMM in Stata. Stata Journal 9(1), 86-136.
Windmeijer, F. (2005). A finite sample correction for the variance of
    linear efficient two-step GMM estimators. Journal of Econometrics
    126, 25-51.
"""
from __future__ import annotations

import numpy as np

from ._results import GMMResult
from .ab_gmm import (
    _dropped_instrument_notes,
    _gmm_sandwich_cov,
    _gmm_step,
    _hansen_j_rank,
    _inv_weight,
    _rank_notes,
    _warn_rank_notes,
)
from .diagnostics import ar_test, windmeijer_correction
from .instruments import (
    _build_panel_records,
    _level_at,
    _prune_zero_columns,
    _y_level_at,
    build_diff_design,
    build_diff_instruments,
)


# ---------------------------------------------------------------------
# Level-equation block
# ---------------------------------------------------------------------


def _build_level_block(
    records: list,
    *,
    lag_dep_var: int,
    n_endog: int,
    n_pred: int,
    n_exog: int,
    collapse: bool,
    gmm_lag_window: tuple = (2, None),
) -> tuple:
    """Build the level-equation rows for system GMM:

    - level outcome ``y_{i,t}`` and level RHS ``[y_{i,t-p} for p=1..lag_dep_var,
      X_endog_{i,t}, X_pred_{i,t}, X_exog_{i,t}]``.
    - level-equation instrument matrix block: one column per regressor
      (Roodman 2009 BB formulation, collapsed), namely
      ``Δy_{i,t-p-off}`` for the p-th lag of y, ``Δw_{i,t-1-off}`` for
      an endogenous ``w``, ``Δw_{i,t-off}`` for a predetermined ``w``,
      and the LEVEL ``x_{i,t}`` for a strictly exogenous ``x``.

    The lag of each level instrument is one shorter than the lowest lag
    that regressor gets in the difference block (Roodman 2009 sec. 3:
    the levels-equation instrument is the difference at lag
    ``laglimit_min - 1``). With ``gmm_lag_window = (lo, hi)`` the
    difference block starts at ``lo`` for endogenous variables and at
    ``lo - 1`` for predetermined ones, so ``off = max(lo - 2, 0)``
    shifts every level instrument accordingly and is zero at the
    default window ``(2, None)``.

    Returns
    -------
    y_lvl : (n_lvl,)
    X_lvl : (n_lvl, k) — same column ordering as the difference design.
    Z_lvl : (n_lvl, m_lvl) — instrument block for the level equation.
    rows : list of dicts ``{"panel", "t", "row"}``.
    """
    k = lag_dep_var + n_endog + n_pred + n_exog

    # Shift of every level instrument relative to the default window.
    # Clamped at 0: lo=1 would ask for Δy_{i,t}, which contains ε_it and
    # is never a valid level instrument.
    lo = int(gmm_lag_window[0])
    off = max(lo - 2, 0)

    def _needed_times(t_int: int) -> set:
        """Time stamps that must be observed for the level row at ``t``."""
        need = {t_int}
        for p in range(1, lag_dep_var + 1):
            need.add(t_int - p)                       # RHS lag of y
            need |= {t_int - p - off, t_int - p - off - 1}   # its Δy instr
        if n_endog:
            need |= {t_int - 1 - off, t_int - 2 - off}       # Δw_{t-1-off}
        if n_pred:
            need |= {t_int - off, t_int - 1 - off}           # Δw_{t-off}
        return need

    rows_data = []  # (rec, t)
    for rec in records:
        tset = set(rec["t"].tolist())
        for t in rec["t"]:
            t_int = int(t)
            if not _needed_times(t_int) <= tset:
                continue
            rows_data.append((rec, t_int))

    if not rows_data:
        # Empty level block is allowed (BB collapses to AB); caller
        # will treat as "no level moments contribute".
        return (
            np.empty(0),
            np.empty((0, k)),
            np.empty((0, 0)),
            [],
        )

    # We need to know the level-instrument ncols up front. Plan:
    #   - 1 col per lag of y for Δy_{i,t-p-off}
    #   - n_endog cols for ΔX_endog_{i,t-1-off}
    #   - n_pred cols for ΔX_pred_{i,t-off}
    #   - n_exog cols for X_exog_{i,t} itself (its differences are
    #     valid in both equations)
    if collapse:
        # one column per category — collapsed
        m_lvl = lag_dep_var + n_endog + n_pred + n_exog
    else:
        # uncollapsed: one column per (category, time t). We'll mirror
        # the collapsed structure for simplicity here (typical xtabond2
        # behaviour: the level moment block is collapsed by default
        # because there's only one moment per regressor in the level
        # equation per time period — uncollapsing it adds nothing
        # economically).
        m_lvl = lag_dep_var + n_endog + n_pred + n_exog

    n_lvl = len(rows_data)
    y_lvl = np.zeros(n_lvl)
    X_lvl = np.zeros((n_lvl, k))
    Z_lvl = np.zeros((n_lvl, m_lvl))
    rows = []

    for r, (rec, t_int) in enumerate(rows_data):
        # outcome
        y_lvl[r] = _y_level_at(rec, t_int)
        # design
        col = 0
        for p in range(1, lag_dep_var + 1):
            X_lvl[r, col] = _y_level_at(rec, t_int - p)
            col += 1
        for j in range(n_endog):
            X_lvl[r, col] = _level_at(rec, t_int, "Xe", j)
            col += 1
        for j in range(n_pred):
            X_lvl[r, col] = _level_at(rec, t_int, "Xp", j)
            col += 1
        for j in range(n_exog):
            X_lvl[r, col] = _level_at(rec, t_int, "Xx", j)
            col += 1

        # level instruments (all endpoints guaranteed observed by the
        # admissibility pass above)
        col = 0
        # Δy_{t-p-off} as level instrument for L^p y on RHS
        for p in range(1, lag_dep_var + 1):
            y_a = _y_level_at(rec, t_int - p - off)
            y_b = _y_level_at(rec, t_int - p - off - 1)
            Z_lvl[r, col] = np.nan if (y_a is None or y_b is None) else (y_a - y_b)
            col += 1
        # ΔX_endog_{t-1-off}.  NOT Δw_{i,t}: for a regressor declared
        # ENDOGENOUS, E[w_it ε_it] ≠ 0 by definition, so Δw_{i,t} =
        # w_it - w_{i,t-1} is correlated with the level error and the
        # moment is false by construction (Roodman 2009 uses Δw_{i,t}
        # only for a PREdetermined regressor).
        for j in range(n_endog):
            v_t = _level_at(rec, t_int - 1 - off, "Xe", j)
            v_tm1 = _level_at(rec, t_int - 2 - off, "Xe", j)
            Z_lvl[r, col] = np.nan if (v_t is None or v_tm1 is None) else (v_t - v_tm1)
            col += 1
        # ΔX_pred_{t-off}
        for j in range(n_pred):
            v_t = _level_at(rec, t_int - off, "Xp", j)
            v_tm1 = _level_at(rec, t_int - 1 - off, "Xp", j)
            Z_lvl[r, col] = np.nan if (v_t is None or v_tm1 is None) else (v_t - v_tm1)
            col += 1
        # X_exog_t
        for j in range(n_exog):
            v_t = _level_at(rec, t_int, "Xx", j)
            if v_t is not None:
                Z_lvl[r, col] = v_t
            col += 1

        rows.append({"panel": rec["id"], "t": t_int, "row": r})

    # Drop any row where we couldn't compute the level instruments
    # (some y_t-p-1 missing, etc.)
    keep = np.all(np.isfinite(X_lvl), axis=1) & np.isfinite(y_lvl)
    if Z_lvl.shape[1] > 0:
        keep = keep & np.all(np.isfinite(Z_lvl), axis=1)
    y_lvl = y_lvl[keep]
    X_lvl = X_lvl[keep]
    Z_lvl = Z_lvl[keep]
    rows = [r for k_, r in zip(keep, rows) if k_]

    return y_lvl, X_lvl, Z_lvl, rows


def _W_step2_system(
    Z_full: np.ndarray,
    residuals: np.ndarray,
    rows_full: list,
) -> tuple[np.ndarray, int]:
    """Step-2 weight for system GMM, panel-clustered across the stacked
    diff + level equations, and its rank. A singular matrix gets the
    Moore-Penrose inverse (see :func:`puremacro.dynpanel.ab_gmm._inv_weight`).
    """
    m = Z_full.shape[1]
    S = np.zeros((m, m))
    panel_groups: dict[object, list[int]] = {}
    for r, drow in enumerate(rows_full):
        panel_groups.setdefault(drow["panel"], []).append(r)
    for pid, rows in panel_groups.items():
        Z_i = Z_full[rows]
        u_i = residuals[rows]
        Zu = Z_i.T @ u_i
        S += np.outer(Zu, Zu)
    return _inv_weight(S, name="bb_gmm step2 weight Σ Z_i' u_i u_i' Z_i")


def _W_step1_system(Z_full: np.ndarray, n_diff: int) -> tuple[np.ndarray, int]:
    """Step-1 weight for system GMM and its rank (Moore-Penrose inverse
    when ``Z'Z`` is singular).

    Block structure: difference block uses ``I`` (simplification of the
    AR-banded ``H``), level block uses ``I`` (under iid level errors).
    Off-diagonal blocks are zero (Roodman 2009 page 102). Two-step is
    the one that matters for SEs anyway; step 1 is a starting weight.
    """
    M = Z_full.T @ Z_full  # = Z'I Z
    return _inv_weight(M, name="bb_gmm step1 weight Z'Z")


def _build_system(
    y: np.ndarray,
    panel_id: np.ndarray,
    time_id: np.ndarray,
    *,
    lag_dep_var: int,
    X_endog: np.ndarray | None,
    X_pred: np.ndarray | None,
    X_exog: np.ndarray | None,
    gmm_lag_window: tuple,
    collapse: bool,
    constant: bool | None = None,
) -> dict:
    """Stack the difference and level equations of system GMM.

    Returns a dict with ``Z`` (block-diagonal: difference instruments on
    the difference rows, level instruments on the level rows), ``X``,
    ``y``, ``rows`` (difference rows first, then level rows), ``diff_rows``,
    ``n_diff``, ``names``, ``instr_labels`` and ``dropped_instr_labels``.

    Instrument columns that are identically zero on the estimation sample
    are dropped (see :func:`puremacro.dynpanel.instruments._prune_zero_columns`);
    they carry no moment condition but make the weight matrices singular.
    """
    # 1) Build the difference block (using the same machinery as ab_gmm)
    records = _build_panel_records(y, panel_id, time_id, X_endog, X_pred, X_exog)
    n_endog = records[0]["Xe"].shape[1] if records else 0
    n_pred = records[0]["Xp"].shape[1] if records else 0
    n_exog = records[0]["Xx"].shape[1] if records else 0

    Z_diff, diff_rows, instr_labels_diff = build_diff_instruments(
        records,
        lag_dep_var=lag_dep_var,
        n_endog=n_endog,
        n_pred=n_pred,
        n_exog=n_exog,
        gmm_lag_window=gmm_lag_window,
        collapse=collapse,
    )
    y_d, X_d, names_auto = build_diff_design(
        records,
        diff_rows,
        lag_dep_var=lag_dep_var,
        n_endog=n_endog,
        n_pred=n_pred,
        n_exog=n_exog,
    )

    # drop bad diff rows
    keep_d = np.isfinite(y_d) & np.all(np.isfinite(X_d), axis=1)
    Z_diff = Z_diff[keep_d]
    y_d = y_d[keep_d]
    X_d = X_d[keep_d]
    diff_rows = [d for k_, d in zip(keep_d, diff_rows) if k_]

    n_diff = len(diff_rows)
    if n_diff == 0:
        raise ValueError(
            "No usable difference rows after filtering. "
            "Cannot estimate BB system GMM."
        )

    # 2) Build the level block
    y_lvl, X_lvl, Z_lvl_block, lvl_rows = _build_level_block(
        records,
        lag_dep_var=lag_dep_var,
        n_endog=n_endog,
        n_pred=n_pred,
        n_exog=n_exog,
        collapse=collapse,
        gmm_lag_window=gmm_lag_window,
    )
    n_lvl = len(lvl_rows)

    # 3) Stack: full Z is block-diagonal with diff instruments in the
    # top-left and level instruments in the bottom-right.
    m_diff = Z_diff.shape[1]
    m_lvl = Z_lvl_block.shape[1]
    m = m_diff + m_lvl
    n_full = n_diff + n_lvl
    Z_full = np.zeros((n_full, m))
    Z_full[:n_diff, :m_diff] = Z_diff
    if n_lvl > 0 and m_lvl > 0:
        Z_full[n_diff:, m_diff:] = Z_lvl_block

    X_full = np.zeros((n_full, X_d.shape[1]))
    X_full[:n_diff] = X_d
    X_full[n_diff:] = X_lvl

    y_full = np.zeros(n_full)
    y_full[:n_diff] = y_d
    y_full[n_diff:] = y_lvl

    labels_lvl = (
        [f"lvl_dL{p}.y" for p in range(1, lag_dep_var + 1)]
        + [f"lvl_dXe{j}" for j in range(n_endog)]
        + [f"lvl_dXp{j}" for j in range(n_pred)]
        + [f"lvl_Xx{j}" for j in range(n_exog)]
    )[:m_lvl]

    use_constant = (
        (n_endog + n_pred + n_exog > 0) if constant is None else bool(constant)
    )
    if use_constant and n_lvl > 0:
        ones_lvl = np.ones((n_lvl, 1))
        already_spans_const = (
            X_lvl.shape[1] > 0
            and np.linalg.matrix_rank(np.hstack([X_lvl, ones_lvl]))
            == np.linalg.matrix_rank(X_lvl)
        )
        if not already_spans_const:
            const_col = np.zeros((n_full, 1))
            const_col[n_diff:, 0] = 1.0
            X_full = np.hstack([X_full, const_col])
            Z_full = np.hstack([Z_full, const_col])
            m += 1
            names_auto = list(names_auto) + ["const"]
            labels_lvl = labels_lvl + ["lvl_const"]

    rows_full = list(diff_rows) + list(lvl_rows)

    Z_full, instr_labels, dropped = _prune_zero_columns(
        Z_full, list(instr_labels_diff) + labels_lvl
    )

    return {
        "Z": Z_full,
        "X": X_full,
        "y": y_full,
        "rows": rows_full,
        "diff_rows": diff_rows,
        "n_diff": n_diff,
        "names": names_auto,
        "instr_labels": instr_labels,
        "dropped_instr_labels": dropped,
    }


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def bb_gmm(
    y: np.ndarray,
    panel_id: np.ndarray,
    time_id: np.ndarray,
    *,
    lag_dep_var: int = 1,
    lags: int | None = None,
    X_endog: np.ndarray | None = None,
    X_pred: np.ndarray | None = None,
    X_exog: np.ndarray | None = None,
    gmm_lag_window: tuple = (2, None),
    collapse: bool = True,
    two_step: bool = True,
    windmeijer: bool = True,
    constant: bool | None = None,
    names: list | None = None,
) -> GMMResult:
    """Blundell-Bond (1998) system GMM.

    Stacks Arellano-Bond difference moments with Blundell-Bond level
    moments. The level moments instrument the level regressors with
    lagged FIRST DIFFERENCES (so the ``α_i`` fixed effect drops
    asymptotically under mean-stationarity of the regressors).

    Same signature and semantics as :func:`ab_gmm` (without its one-step
    ``robust=False`` option), plus ``constant``. The level block is
    always collapsed (one moment per regressor per time t), regardless
    of ``collapse=True/False``: the BB level moment is intrinsically
    one-per-regressor and uncollapsing it is not standard practice.
    The ``collapse`` flag still applies to the AB difference moment
    block. Instrument columns that are identically zero on the
    estimation sample are dropped and listed in ``GMMResult.notes``.

    ``gmm_lag_window`` applies to BOTH blocks: raising its lower limit
    to ``lo`` (the usual remedy for MA(q) errors) also pushes each level
    instrument back by ``lo - 2`` periods, so the level moment stays
    valid alongside the difference moments it accompanies. At the
    default ``(2, None)`` the level instruments are unshifted.

    The Windmeijer correction differentiates the step-2 weight at the
    one-step estimate (see
    :func:`puremacro.dynpanel.diagnostics.windmeijer_correction`). The
    step-1 weight here is ``(Z'Z)^{-1}``, not the ``H``-weighted matrix of
    Stata ``xtdpdsys``, so one-step estimates (and hence the two-step
    ones) are not expected to match ``xtdpdsys`` numerically.

    Singular weight matrices (``Z'Z`` or the step-2 moment covariance, the
    latter common with year dummies) are inverted with the Moore-Penrose
    generalized inverse, as in :func:`ab_gmm`, with a note and a
    ``RuntimeWarning``.

    See :func:`ab_gmm` parameters for details.

    References
    ----------
    See module-level docstring.
    """
    if lags is not None:
        lag_dep_var = lags
    system = _build_system(
        y,
        panel_id,
        time_id,
        lag_dep_var=lag_dep_var,
        X_endog=X_endog,
        X_pred=X_pred,
        X_exog=X_exog,
        gmm_lag_window=gmm_lag_window,
        collapse=collapse,
        constant=constant,
    )
    Z_full = system["Z"]
    X_full = system["X"]
    y_full = system["y"]
    rows_full = system["rows"]
    diff_rows = system["diff_rows"]
    n_diff = system["n_diff"]
    names_auto = system["names"]
    notes = _dropped_instrument_notes(system["dropped_instr_labels"])

    m = Z_full.shape[1]
    n_full = len(rows_full)
    n_panels = len({r["panel"] for r in rows_full})
    k = X_full.shape[1]

    if n_full < k + 1:
        raise ValueError(
            f"Only {n_full} usable observations for {k} regressors. "
            "Panel is degenerate."
        )

    # Step 1
    W1, rank1 = _W_step1_system(Z_full, n_diff)
    beta1, _ = _gmm_step(Z_full, X_full, y_full, W1, name="bb_gmm step1")
    resid1 = y_full - X_full @ beta1

    if not two_step:
        # one-step robust
        S = np.zeros((m, m))
        panel_groups: dict[object, list[int]] = {}
        for r, drow in enumerate(rows_full):
            panel_groups.setdefault(drow["panel"], []).append(r)
        for pid, rows in panel_groups.items():
            Z_i = Z_full[rows]
            u_i = resid1[rows]
            Zu = Z_i.T @ u_i
            S += np.outer(Zu, Zu)
        bread_inv = _gmm_sandwich_cov(Z_full, X_full, W1, name="bb_gmm one-step bread")
        ZX = Z_full.T @ X_full
        meat = ZX.T @ W1 @ S @ W1 @ ZX
        cov = bread_inv @ meat @ bread_inv
        cov = 0.5 * (cov + cov.T)
        beta = beta1
        resid = resid1
        W_for_J = W1
        step = 1
        windmeijer_used = False
    else:
        W2, rank2 = _W_step2_system(Z_full, resid1, rows_full)
        beta2, _ = _gmm_step(Z_full, X_full, y_full, W2, name="bb_gmm step2")
        resid2 = y_full - X_full @ beta2
        cov_uncorr = _gmm_sandwich_cov(Z_full, X_full, W2, name="bb_gmm step2 cov")

        if windmeijer:
            # one-step robust covariance V_1
            S1 = np.zeros((m, m))
            panel_groups = {}
            for r, drow in enumerate(rows_full):
                panel_groups.setdefault(drow["panel"], []).append(r)
            for pid, rows in panel_groups.items():
                Z_i = Z_full[rows]
                u_i = resid1[rows]
                Zu_i = Z_i.T @ u_i
                S1 += np.outer(Zu_i, Zu_i)
            bread1 = _gmm_sandwich_cov(Z_full, X_full, W1, name="bb_gmm V1 bread")
            ZX_full = Z_full.T @ X_full
            meat1 = ZX_full.T @ W1 @ S1 @ W1 @ ZX_full
            V_1 = bread1 @ meat1 @ bread1
            V_1 = 0.5 * (V_1 + V_1.T)

            cov = windmeijer_correction(
                beta_hat=beta2,
                Z=Z_full,
                X_diff=X_full,
                y_diff=y_full,
                W2=W2,
                residuals_step2=resid2,
                cov_uncorrected=cov_uncorr,
                diff_rows=rows_full,
                cov_step1=V_1,
                residuals_step1=resid1,
            )
            windmeijer_used = True
        else:
            cov = cov_uncorr
            windmeijer_used = False

        beta = beta2
        resid = resid2
        W_for_J = W2
        step = 2

    se = np.sqrt(np.maximum(np.diag(cov), 0.0))

    # Hansen J on the full system
    if step == 2:
        J, J_p, J_df = _hansen_j_rank(
            Z_full, resid, W_for_J, n_regressors=k, n_independent=rank1
        )
    else:
        W2, rank2 = _W_step2_system(Z_full, resid1, rows_full)
        J, J_p, J_df = _hansen_j_rank(
            Z_full, resid1, W2, n_regressors=k, n_independent=rank1
        )
    rank_notes = _rank_notes(
        m=m, rank1=rank1, rank2=rank2, step1_label="Z'Z", two_step=two_step
    )
    _warn_rank_notes(rank_notes, "bb_gmm")
    notes = notes + rank_notes

    # AR tests on the DIFFERENCE residuals only (BB convention)
    # extract residuals corresponding to diff rows
    resid_diff_part = resid[:n_diff]
    ar1, ar1_p = ar_test(resid_diff_part, diff_rows, lag=1)
    ar2, ar2_p = ar_test(resid_diff_part, diff_rows, lag=2)

    if names is not None:
        has_const = len(names_auto) == k and names_auto[-1] == "const"
        if len(names) == k:
            coef_names = tuple(names)
        elif has_const and len(names) == k - 1:
            coef_names = tuple(names) + ("const",)
        else:
            want = f"{k}" if not has_const else f"{k} (or {k - 1} without 'const')"
            raise ValueError(
                f"names has length {len(names)} but there are {want} coefficients."
            )
    else:
        coef_names = tuple(names_auto)

    converged = bool(np.all(np.isfinite(beta)) and np.all(np.isfinite(se)))

    return GMMResult(
        coefs=beta,
        se=se,
        cov=cov,
        names=coef_names,
        hansen_j=float(J),
        hansen_j_p=float(J_p),
        hansen_j_df=int(J_df),
        ar1_p=float(ar1_p),
        ar2_p=float(ar2_p),
        n_instruments=int(rank1),
        n_obs=int(n_full),
        n_panels=int(n_panels),
        step=int(step),
        windmeijer=bool(windmeijer_used),
        estimator="bb",
        converged=converged,
        notes=notes,
    )


__all__ = ["bb_gmm"]
