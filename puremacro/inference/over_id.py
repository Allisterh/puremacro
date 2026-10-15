"""Over-identification: Hansen J test for 2SLS / IV-GMM, plus a small
Stock-Yogo critical-value lookup for weak-instrument decisions.

Hansen J (``robust=True``, the default):
    Two-step GMM with the heteroskedasticity-robust (HC0) weight matrix.
    Step 1 is 2SLS; its residuals ``u_1`` give
    ``S = (1/T) sum_t z_t z_t' u_1t^2``. Step 2 is the efficient GMM
    estimator ``b_2 = (X'Z S^{-1} Z'X)^{-1} X'Z S^{-1} Z'y`` and the
    statistic is

        J = T * gbar(b_2)' S^{-1} gbar(b_2),   gbar(b) = Z'(y - X b) / T,

    which is ``chi^2(l - k)`` under valid instruments and conditional
    heteroskedasticity of unknown form (Hansen 1982; Hayashi 2000, ch. 3;
    what ``ivreg2, gmm2s robust`` and ``linearmodels.IVGMM`` report).

Sargan (``robust=False``):
    The homoskedastic special case: regress the 2SLS residual on the
    instrument matrix (and the included controls); ``J = T * R^2``. This is
    what the function computed -- under the name Hansen J -- before 2.3.1.

Here ``l`` = number of excluded instruments and ``k`` = number of
endogenous regressors; the included exogenous controls ``W`` (constant
included) enter both the instrument set and the regressor set and do not
change the degrees of freedom.

Stock-Yogo critical values for the Cragg-Donald first-stage F, single
endogenous regressor (n = 1), 5% test level:

* TSLS relative bias (Stock and Yogo 2002, NBER TWP 284, Table 1), maximal
  bias relative to OLS b in {5, 10, 20, 30}%, for K2 = 3..30 instruments;
* TSLS size (Table 2), maximal rejection rate r in {10, 15, 20, 25}% of a
  nominal 5% Wald test, for K2 = 1..30 instruments.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from .._linalg import inv_xtx


def hansen_j(
    y: np.ndarray,
    X_endog: np.ndarray,
    Z: np.ndarray,
    W: np.ndarray | None = None,
    *,
    robust: bool = True,
) -> dict:
    """Hansen J (robust two-step GMM) or Sargan over-identification statistic.

    Parameters
    ----------
    y       : (T,) outcome.
    X_endog : (T, k) endogenous regressors.
    Z       : (T, l) excluded instruments (l > k for the test to be meaningful).
    W       : (T, m) included exogenous controls; a constant is prepended
              when the first column is not all ones. ``None`` means a
              constant only.
    robust  : if True (default) the heteroskedasticity-robust Hansen J
              from two-step GMM with the HC0 weight matrix; if False the
              homoskedastic Sargan statistic ``T * R^2`` of the 2SLS residual
              on the instruments.

    Returns
    -------
    dict with stat (J), p_value, df (= l - k), n_obs.
    ``p_value`` is NaN when df = 0 (just identified: J is identically 0).

    References
    ----------
    Hansen, L.P. (1982). Large sample properties of generalized method of
        moments estimators. Econometrica 50(4), 1029-1054.
    Sargan, J.D. (1958). The estimation of economic relationships using
        instrumental variables. Econometrica 26(3), 393-415.
    Hayashi, F. (2000). Econometrics. Princeton University Press, ch. 3.
    """
    y = np.asarray(y, dtype=float).ravel()
    X_endog = np.asarray(X_endog, dtype=float)
    Z = np.asarray(Z, dtype=float)
    X = X_endog.reshape(-1, 1) if X_endog.ndim == 1 else X_endog
    Z = Z.reshape(-1, 1) if Z.ndim == 1 else Z
    T = len(y)
    k = X.shape[1]
    l = Z.shape[1]
    if X.shape[0] != T or Z.shape[0] != T:
        raise ValueError(
            f"hansen_j: y has {T} rows but X_endog has {X.shape[0]} and Z has {Z.shape[0]}"
        )

    if W is None:
        W = np.ones((T, 1))
    else:
        W = np.asarray(W, dtype=float)
        W = W.reshape(-1, 1) if W.ndim == 1 else W
        # ensure a constant is present
        if not np.allclose(W[:, 0], 1.0):
            W = np.column_stack([np.ones(T), W])

    full_Z = np.column_stack([W, Z])       # (T, m + l) instrument set
    XW = np.column_stack([X, W])           # (T, k + m) regressors

    # Step 1: 2SLS.  X_hat = P_Z X without forming the T x T projection.
    ZtZ_inv = inv_xtx(full_Z, name="hansen_j (Z'Z)")
    X_hat = full_Z @ (ZtZ_inv @ (full_Z.T @ X))
    XW_hat = np.column_stack([X_hat, W])
    beta_2sls = np.linalg.solve(XW_hat.T @ XW, XW_hat.T @ y)
    u_1 = y - XW @ beta_2sls

    if not robust:
        # Sargan: T * R^2 of the 2SLS residual on the instruments.
        proj = full_Z @ (ZtZ_inv @ (full_Z.T @ u_1))
        rss = float(np.sum((u_1 - proj) ** 2))
        tss = float(np.sum((u_1 - u_1.mean()) ** 2))
        r2 = 1.0 - rss / tss if tss > 0 else 0.0
        J = T * r2
    else:
        # HC0 weight matrix from the first-step residuals.
        S = (full_Z.T * u_1 ** 2) @ full_Z / T
        try:
            S_inv_ZX = np.linalg.solve(S, full_Z.T @ XW)
            S_inv_Zy = np.linalg.solve(S, full_Z.T @ y)
        except np.linalg.LinAlgError as exc:
            raise np.linalg.LinAlgError(
                "hansen_j: the HC0 weight matrix S = Z' diag(u^2) Z / T is "
                "singular (collinear instruments, or residuals that are "
                "exactly zero)"
            ) from exc
        # Step 2: efficient GMM with weight S^{-1}.
        ZX = full_Z.T @ XW
        beta_gmm = np.linalg.solve(ZX.T @ S_inv_ZX, ZX.T @ S_inv_Zy)
        u_2 = y - XW @ beta_gmm
        gbar = full_Z.T @ u_2 / T
        J = float(T * gbar @ np.linalg.solve(S, gbar))

    df = l - k
    p_value = chi2.sf(J, df=df) if df > 0 else np.nan
    return {
        "stat": float(J),
        "p_value": float(p_value),
        "df": int(df),
        "n_obs": int(T),
    }


# Stock-Yogo critical values, one endogenous regressor (n = 1), 5% level.
# Source: Stock, J.H. and Yogo, M. (2002), "Testing for Weak Instruments in
# Linear IV Regression", NBER Technical Working Paper 284,
# https://www.nber.org/system/files/working_papers/t0284/t0284.pdf
# (published in Andrews and Stock, eds., Identification and Inference for
# Econometric Models, Cambridge University Press, 2005).
#
# Table 1 "Critical Values for the Weak Instrument Test Based on TSLS Bias",
# printed page 58, columns "n = 1, b = 0.05, 0.10, 0.20, 0.30".
# The bias is defined only for K2 >= n + 2 (section 3.5, printed p. 24),
# so the n = 1 rows start at K2 = 3.
# Rows: K2 -> (b = 5%, 10%, 20%, 30%).
_SY_BIAS_PCTS = (5, 10, 20, 30)
_SY_BIAS_ROWS_N1 = {
     3: (13.91, 9.08, 6.46, 5.39),
     4: (16.85, 10.27, 6.71, 5.34),
     5: (18.37, 10.83, 6.77, 5.25),
     6: (19.28, 11.12, 6.76, 5.15),
     7: (19.86, 11.29, 6.73, 5.07),
     8: (20.25, 11.39, 6.69, 4.99),
     9: (20.53, 11.46, 6.65, 4.92),
    10: (20.74, 11.49, 6.61, 4.86),
    11: (20.90, 11.51, 6.56, 4.80),
    12: (21.01, 11.52, 6.53, 4.75),
    13: (21.10, 11.52, 6.49, 4.71),
    14: (21.18, 11.52, 6.45, 4.67),
    15: (21.23, 11.51, 6.42, 4.63),
    16: (21.28, 11.50, 6.39, 4.59),
    17: (21.31, 11.49, 6.36, 4.56),
    18: (21.34, 11.48, 6.33, 4.53),
    19: (21.36, 11.46, 6.31, 4.51),
    20: (21.38, 11.45, 6.28, 4.48),
    21: (21.39, 11.44, 6.26, 4.46),
    22: (21.40, 11.42, 6.24, 4.43),
    23: (21.41, 11.41, 6.22, 4.41),
    24: (21.41, 11.40, 6.20, 4.39),
    25: (21.42, 11.38, 6.18, 4.37),
    26: (21.42, 11.37, 6.16, 4.35),
    27: (21.42, 11.36, 6.14, 4.34),
    28: (21.42, 11.34, 6.13, 4.32),
    29: (21.42, 11.33, 6.11, 4.31),
    30: (21.42, 11.32, 6.09, 4.29),
}

# Table 2 "Critical Values for the Weak Instrument Test Based on TSLS Size",
# printed page 59, columns "n = 1, r = 0.10, 0.15, 0.20, 0.25" (maximal
# rejection rate of a nominal 5% Wald test of beta = beta_0).
# Rows: K2 -> (r = 10%, 15%, 20%, 25%).
_SY_SIZE_PCTS = (10, 15, 20, 25)
_SY_SIZE_ROWS_N1 = {
     1: (16.38, 8.96, 6.66, 5.53),
     2: (19.93, 11.59, 8.75, 7.25),
     3: (22.30, 12.83, 9.54, 7.80),
     4: (24.58, 13.96, 10.26, 8.31),
     5: (26.87, 15.09, 10.98, 8.84),
     6: (29.18, 16.23, 11.72, 9.38),
     7: (31.50, 17.38, 12.48, 9.93),
     8: (33.84, 18.54, 13.24, 10.50),
     9: (36.19, 19.71, 14.01, 11.07),
    10: (38.54, 20.88, 14.78, 11.65),
    11: (40.90, 22.06, 15.56, 12.23),
    12: (43.27, 23.24, 16.35, 12.82),
    13: (45.64, 24.42, 17.14, 13.41),
    14: (48.01, 25.61, 17.93, 14.00),
    15: (50.39, 26.80, 18.72, 14.60),
    16: (52.77, 27.99, 19.51, 15.19),
    17: (55.15, 29.19, 20.31, 15.79),
    18: (57.53, 30.38, 21.10, 16.39),
    19: (59.92, 31.58, 21.90, 16.99),
    20: (62.30, 32.77, 22.70, 17.60),
    21: (64.69, 33.97, 23.50, 18.20),
    22: (67.07, 35.17, 24.30, 18.80),
    23: (69.46, 36.37, 25.10, 19.41),
    24: (71.85, 37.57, 25.90, 20.01),
    25: (74.24, 38.77, 26.71, 20.61),
    26: (76.62, 39.97, 27.51, 21.22),
    27: (79.01, 41.17, 28.31, 21.83),
    28: (81.40, 42.37, 29.12, 22.43),
    29: (83.79, 43.57, 29.92, 23.04),
    30: (86.17, 44.78, 30.72, 23.65),
}

# Keys: (n_instruments, threshold %) -> first-stage F critical value.
_STOCK_YOGO_BIAS_K1 = {
    (k2, pct): cv
    for k2, row in _SY_BIAS_ROWS_N1.items()
    for pct, cv in zip(_SY_BIAS_PCTS, row)
}
_STOCK_YOGO_SIZE_K1 = {
    (k2, pct): cv
    for k2, row in _SY_SIZE_ROWS_N1.items()
    for pct, cv in zip(_SY_SIZE_PCTS, row)
}
_STOCK_YOGO_TABLES = {"bias": _STOCK_YOGO_BIAS_K1, "size": _STOCK_YOGO_SIZE_K1}


def stock_yogo_cv(
    n_instruments: int,
    max_bias_pct: int = 10,
    *,
    criterion: str = "bias",
) -> float | None:
    """Look up the Stock-Yogo critical value for first-stage F.

    Compare the Cragg-Donald first-stage F (with one endogenous regressor,
    the usual first-stage F) against this value; reject "weak instruments"
    if F exceeds it. The tables are for one endogenous regressor and a 5%
    test level.

    Parameters
    ----------
    n_instruments : int
        Number of excluded instruments ``l`` (``K2`` in Stock-Yogo).
    max_bias_pct : int, default 10
        The tolerated weakness, in percent. With ``criterion="bias"`` it is
        the maximal TSLS bias relative to OLS (5, 10, 20 or 30). With
        ``criterion="size"`` it is the maximal rejection rate of a nominal
        5% Wald test (10, 15, 20 or 25).
    criterion : {"bias", "size"}, keyword-only, default "bias"
        ``"bias"``: Stock-Yogo Table 1 (TSLS relative bias), tabulated for
        ``l = 3..30``. ``"size"``: Table 2 (TSLS Wald-test size), tabulated
        for ``l = 1..30``.

    Returns
    -------
    Critical value (float), or None if ``(l, max_bias_pct)`` is not
    tabulated. The bias is undefined for fewer than three instruments
    (Stock-Yogo define it only for ``l >= n + 2``), so
    ``stock_yogo_cv(1)`` and ``stock_yogo_cv(2)`` return None; use
    ``criterion="size"`` for one or two instruments.

    Notes
    -----
    The Staiger-Stock rule of thumb "F > 10" is close to the 10% bias
    critical value, which is about 11 for most ``l`` (9.08 at ``l = 3``).
    By the size criterion it is far less demanding: with one instrument a
    5% Wald test with at most 10% actual size needs F > 16.38, and the
    critical value grows with ``l`` (86.17 at ``l = 30``).

    References
    ----------
    Stock, J.H. and Yogo, M. (2002). Testing for weak instruments in linear
        IV regression. NBER Technical Working Paper 284, Tables 1 and 2.
        Published in Andrews, D.W.K. and Stock, J.H. (eds) (2005),
        Identification and Inference for Econometric Models, Cambridge
        University Press.
    """
    try:
        table = _STOCK_YOGO_TABLES[criterion]
    except (KeyError, TypeError):
        raise ValueError(
            f"stock_yogo_cv: criterion must be 'bias' or 'size', got {criterion!r}"
        ) from None
    return table.get((int(n_instruments), int(max_bias_pct)))


__all__ = ["hansen_j", "stock_yogo_cv"]
