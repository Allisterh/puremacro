r"""Exact analytical likelihood score and parameter sensitivity engine for linear DSGE models.

Computes exact analytical gradients of the log-likelihood function:
    \nabla_\theta \ln L(Y \mid \theta)
via implicit state-space differentiation and single-pass forward Kalman score recursion,
strictly under the zero-dependency Pyodide four-package contract (numpy, scipy, pandas, matplotlib).

Mathematical Foundations
-------------------------
1. Parameter Sensitivity of Equilibrium Decision Rules:
   Given structural equations around steady state:
       A_+(\theta) G(\theta)^2 + A_0(\theta) G(\theta) + A_-(\theta) = 0
       (A_0(\theta) + A_+(\theta) G(\theta)) N(\theta) + B_u(\theta) = 0
   Differentiating with respect to parameter \theta_j yields the generalized Sylvester system:
       \hat{A} \frac{\partial G}{\partial \theta_j} + A_+ \frac{\partial G}{\partial \theta_j} G = D_j
   where \hat{A} = A_0 + A_+ G and D_j = - ( \frac{\partial A_+}{\partial \theta_j} G^2 + \frac{\partial A_0}{\partial \theta_j} G + \frac{\partial A_-}{\partial \theta_j} ).
   Solved via complex Schur triangular back-substitution in O(N^3) flops.

   The innovation loading derivative follows directly:
       \frac{\partial N}{\partial \theta_j} = - \hat{A}^{-1} [ ( \frac{\partial A_0}{\partial \theta_j} + \frac{\partial A_+}{\partial \theta_j} G + A_+ \frac{\partial G}{\partial \theta_j} ) N + \frac{\partial B_u}{\partial \theta_j} ]

2. Stationary Unconditional Covariance Sensitivity:
   Differentiating the discrete Lyapunov equation P_0 = T P_0 T' + R Q R':
       \frac{\partial P_0}{\partial \theta_j} - T \frac{\partial P_0}{\partial \theta_j} T' = \frac{\partial T}{\partial \theta_j} P_0 T' + T P_0 (\frac{\partial T}{\partial \theta_j})' + \frac{\partial (R Q R')}{\partial \theta_j}
   solved via scipy.linalg.solve_discrete_lyapunov.

3. Single-Pass Forward Kalman Score Recursion:
   For observations y_t with forecast errors v_t = y_t - Z a_{t|t-1} - d and covariances F_t = Z P_{t|t-1} Z' + H:
       \frac{\partial \ln L}{\partial \theta_j} = -0.5 \sum_{t=1}^{T_{obs}} [ \operatorname{tr}(F_t^{-1} \frac{\partial F_t}{\partial \theta_j}) + 2 v_t' F_t^{-1} \frac{\partial v_t}{\partial \theta_j} - (F_t^{-1} v_t)' \frac{\partial F_t}{\partial \theta_j} (F_t^{-1} v_t) ]
   propagating (\frac{\partial a_{t|t-1}}{\partial \theta_j}, \frac{\partial P_{t|t-1}}{\partial \theta_j}) forward alongside the filter.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.linalg

from puremacro.dsge.priors import grad_log_prior, log_prior
from puremacro.reports import _df_to_latex, _df_to_markdown, _df_to_typst
from puremacro.state_space import StateSpaceModel


# ---------------------------------------------------------------------------
# Generalized Sylvester Solver
# ---------------------------------------------------------------------------

def solve_sylvester_generalized(
    A_hat: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
) -> np.ndarray:
    r"""Solve the generalized Sylvester matrix equation \hat{A} X + B X C = D.

    Parameters
    ----------
    A_hat : np.ndarray of shape (N, N)
        Leading coefficient matrix, typically (A_0 + A_+ G).
    B : np.ndarray of shape (N, N)
        Intermediate coefficient matrix, typically A_+.
    C : np.ndarray of shape (M, M)
        Right multiplier matrix with stable eigenvalues (e.g. state transition G).
    D : np.ndarray of shape (N, M) or batched (K, N, M)
        Right-hand side matrix or batch of K right-hand side matrices.

    Returns
    -------
    X : np.ndarray of shape (N, M) or batched (K, N, M)
        Solution matrix satisfying \hat{A} X + B X C = D.

    Notes
    -----
    Computes the complex Schur decomposition of C:
        C = U_c T_c U_c^H
    where T_c is upper triangular and U_c is unitary.
    Transforming \tilde{X} = X U_c and \tilde{D} = D U_c leads to:
        \hat{A} \tilde{X} + B \tilde{X} T_c = \tilde{D}
    Because T_c is upper triangular, column k of \tilde{X} satisfies:
        (\hat{A} + (T_c)_{k, k} B) \tilde{X}_{:, k} = \tilde{D}_{:, k} - B \sum_{m < k} \tilde{X}_{:, m} (T_c)_{m, k}
    which is solved column-by-column by triangular back-substitution.
    """
    A_hat = np.asarray(A_hat, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    D = np.asarray(D, dtype=float)

    is_batched = (D.ndim == 3)
    if not is_batched:
        D_batch = D[np.newaxis, :, :]
    else:
        D_batch = D

    K_batch, N, M = D_batch.shape
    if M == 0 or N == 0:
        res = np.zeros_like(D)
        return res

    if np.all(C == 0.0):
        # Degenerates to A_hat X = D
        X_batch = np.zeros_like(D_batch)
        for k in range(K_batch):
            try:
                X_batch[k] = np.linalg.solve(A_hat, D_batch[k])
            except np.linalg.LinAlgError:
                X_batch[k] = np.linalg.lstsq(A_hat, D_batch[k], rcond=None)[0]
        return X_batch if is_batched else X_batch[0]

    # Complex Schur decomposition of C: C = U_c @ T_c @ U_c^H
    T_c, U_c = scipy.linalg.schur(C, output="complex")

    # Transform RHS: \tilde{D} = D @ U_c
    D_tilde = np.zeros((K_batch, N, M), dtype=complex)
    for b in range(K_batch):
        D_tilde[b] = D_batch[b] @ U_c

    X_tilde = np.zeros((K_batch, N, M), dtype=complex)

    # Solve column by column k = 0, ..., M - 1
    for k in range(M):
        diag_val = T_c[k, k]
        M_k = A_hat + diag_val * B

        # Factorize M_k once for all K batch elements
        try:
            lu, piv = scipy.linalg.lu_factor(M_k)
            use_lu = True
        except (np.linalg.LinAlgError, scipy.linalg.LinAlgError):
            use_lu = False

        for b in range(K_batch):
            rhs_k = D_tilde[b, :, k].copy()
            if k > 0:
                # Subtraction of lower column couplings
                prev_couplings = X_tilde[b, :, :k] @ T_c[:k, k]
                rhs_k -= B @ prev_couplings

            if use_lu:
                sol_k = scipy.linalg.lu_solve((lu, piv), rhs_k)
            else:
                try:
                    sol_k = np.linalg.solve(M_k, rhs_k)
                except np.linalg.LinAlgError:
                    sol_k = np.linalg.lstsq(M_k, rhs_k, rcond=None)[0]

            X_tilde[b, :, k] = sol_k

    # Back-transform: X = Re(\tilde{X} @ U_c^H)
    X_out = np.zeros((K_batch, N, M), dtype=float)
    U_c_H = U_c.conj().T
    for b in range(K_batch):
        X_out[b] = np.real(X_tilde[b] @ U_c_H)

    return X_out if is_batched else X_out[0]


def sylvester_order2_tangent(
    A_hat: np.ndarray,
    B: np.ndarray,
    G: np.ndarray,
    D_xx: np.ndarray,
    *,
    dA_hat: np.ndarray,
    dB: np.ndarray,
    dG: np.ndarray,
    dD_xx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Solve second-order Sylvester equation $A_{\text{hat}} X + B X (G \otimes G) = D_{xx}$ and its parameter tangents.

    Parameters
    ----------
    A_hat, B : np.ndarray of shape (N, N)
        Leading and forward Sylvester coefficient matrices.
    G : np.ndarray of shape (n_s, n_s)
        First-order state transition matrix.
    D_xx : np.ndarray of shape (N, n_s * n_s)
        Second-order right-hand side matrix.
    dA_hat, dB : np.ndarray of shape (P, N, N)
        Parameter sensitivities of `A_hat` and `B`.
    dG : np.ndarray of shape (P, n_s, n_s)
        Parameter sensitivities of `G`.
    dD_xx : np.ndarray of shape (P, N, n_s * n_s)
        Parameter sensitivities of `D_xx`.

    Returns
    -------
    X_xx : np.ndarray of shape (N, n_s * n_s)
        Primal second-order decision-rule matrix.
    dX_xx : np.ndarray of shape (P, N, n_s * n_s)
        Exact analytical parameter derivatives $\partial X_{xx} / \partial \theta$.
    """
    A_hat = np.asarray(A_hat, dtype=float)
    B = np.asarray(B, dtype=float)
    G = np.asarray(G, dtype=float)
    D_xx = np.asarray(D_xx, dtype=float)
    dA_hat = np.asarray(dA_hat, dtype=float)
    dB = np.asarray(dB, dtype=float)
    dG = np.asarray(dG, dtype=float)
    dD_xx = np.asarray(dD_xx, dtype=float)

    C = np.kron(G, G)
    X_xx = solve_sylvester_generalized(A_hat, B, C, D_xx)

    P = dG.shape[0]
    M = C.shape[0]
    N = A_hat.shape[0]
    XC = X_xx @ C
    BX = B @ X_xx

    rhs_batch = np.zeros((P, N, M), dtype=float)
    for p in range(P):
        dC_p = np.kron(dG[p], G) + np.kron(G, dG[p])
        rhs_batch[p] = (
            dD_xx[p]
            - dA_hat[p] @ X_xx
            - dB[p] @ XC
            - BX @ dC_p
        )

    dX_xx = solve_sylvester_generalized(A_hat, B, C, rhs_batch)
    if dX_xx.ndim == 2:
        dX_xx = dX_xx[np.newaxis, :, :]
    return X_xx, dX_xx



# ---------------------------------------------------------------------------
# StateSpaceSensitivity Container
# ---------------------------------------------------------------------------

@dataclass
class StateSpaceSensitivity:
    """Analytical parameter sensitivities of time-invariant state-space matrices."""

    dT: np.ndarray
    dZ: np.ndarray
    dR: np.ndarray
    dQ: np.ndarray
    dH: np.ndarray
    dc: np.ndarray
    dd: np.ndarray
    da0: np.ndarray | None = None
    dP0: np.ndarray | None = None


class StateSpaceSensitivities(dict):
    """``{parameter: StateSpaceSensitivity}``, with the state space they belong to.

    Returned by :func:`build_state_space_sensitivities`. It is a plain
    ``dict`` (iteration, indexing and ``kalman_score`` are unchanged) that
    also carries

    - ``state_space``: the :class:`~puremacro.state_space.StateSpaceModel`
      whose matrices the sensitivities differentiate, to be passed to
      :func:`kalman_score` with them;
    - ``model``: the solved model that state space was built from. For a
      ``build_dynare`` / ``load_mod`` model whose calibration sets a lag
      coefficient to 0 this is the model re-solved on the re-solve state set
      (:func:`puremacro.dsge.dynare._resolve_states`), which has more states
      than the model passed in; a state space built from the original model
      does not match the sensitivities.
    """

    state_space: StateSpaceModel | None = None
    model: Any = None


def _check_sensitivity_shapes(model: StateSpaceModel, sens_list: Sequence[Any], names: Sequence[str]) -> None:
    """Raise a named ``ValueError`` when a sensitivity does not match ``model``."""
    m = int(np.shape(model.T)[0])
    expected = {
        "dT": np.shape(model.T),
        "dZ": np.shape(model.Z),
        "dR": np.shape(model.R) if model.R is not None else (m, m),
        "dQ": np.shape(model.Q),
        "dH": np.shape(model.H),
    }
    for nm, s in zip(names, sens_list):
        for field_name, shape in expected.items():
            got = np.shape(getattr(s, field_name))
            if got != shape:
                raise ValueError(
                    f"kalman_score: the sensitivity of {nm!r} has {field_name} of shape "
                    f"{got}, but the state space has {field_name[1:]} of shape {shape} "
                    f"(m = {m} states). The sensitivities and the state space must come "
                    "from the same solved model: build_state_space_sensitivities "
                    "re-solves a build_dynare/load_mod model on its re-solve state set "
                    "(every variable that enters with a lag, including lags calibrated "
                    "to 0), which can have more states than the model passed in. Use "
                    "the state space it returns with them (sens.state_space), or build "
                    "the state space from sens.model."
                )
        for field_name, size in (("dc", m), ("dd", int(np.shape(model.Z)[0]))):
            got = np.size(getattr(s, field_name))
            if got != size:
                raise ValueError(
                    f"kalman_score: the sensitivity of {nm!r} has {field_name} of size "
                    f"{got}, but the state space needs {size}. The sensitivities and the "
                    "state space must come from the same solved model (see "
                    "build_state_space_sensitivities: sens.state_space / sens.model)."
                )


# ---------------------------------------------------------------------------
# Forward Kalman Score Recursion
# ---------------------------------------------------------------------------

def kalman_score(
    y: np.ndarray,
    model: StateSpaceModel,
    sensitivities: Mapping[str, StateSpaceSensitivity] | Sequence[StateSpaceSensitivity],
    a0: np.ndarray | None = None,
    P0: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    r"""Compute exact analytical log-likelihood and score vector \nabla_\theta \ln L.

    Parameters
    ----------
    y : np.ndarray of shape (T_obs, n_y)
        Observed series matrix. NaN entries indicate missing values.
    model : StateSpaceModel
        Baseline state space model containing T, Z, R, Q, H, c, d.
    sensitivities : Mapping[str, StateSpaceSensitivity] or Sequence[StateSpaceSensitivity]
        Parameter sensitivity derivatives of state-space matrices.
    a0 : np.ndarray of shape (m,), optional
        Initial state mean. If None, initialized from stationary mean.
    P0 : np.ndarray of shape (m, m), optional
        Initial state covariance. If None, initialized from Lyapunov solution.

    Returns
    -------
    loglik : float
        Log-likelihood value \ln L(Y \mid \theta).
    score : np.ndarray of shape (K,)
        Analytical gradient vector \nabla_\theta \ln L(Y \mid \theta).
    """
    y_mat = np.asarray(y, dtype=float)
    if y_mat.ndim != 2:
        raise ValueError(f"y must be a 2-D array of shape (T_obs, n_y); got {y_mat.shape}")

    T_obs, n_y = y_mat.shape
    T_mat = np.asarray(model.T, dtype=float)
    Z_mat = np.asarray(model.Z, dtype=float)
    R_mat = np.asarray(model.R, dtype=float)
    Q_mat = np.asarray(model.Q, dtype=float)
    H_mat = np.asarray(model.H, dtype=float)
    c_vec = np.asarray(model.c, dtype=float).ravel()
    d_vec = np.asarray(model.d, dtype=float).ravel()

    m = T_mat.shape[0]
    r_dim = R_mat.shape[1]

    if isinstance(sensitivities, Mapping):
        sens_list = list(sensitivities.values())
        sens_names = [str(k) for k in sensitivities.keys()]
    else:
        sens_list = list(sensitivities)
        sens_names = [f"#{i}" for i in range(len(sens_list))]
    _check_sensitivity_shapes(model, sens_list, sens_names)

    K = len(sens_list)
    score = np.zeros(K, dtype=float)

    # Initial unconditional covariance P0 and mean a0
    RQR = R_mat @ Q_mat @ R_mat.T
    if P0 is None:
        try:
            P_curr = scipy.linalg.solve_discrete_lyapunov(T_mat, RQR)
            P_curr = 0.5 * (P_curr + P_curr.T)
        except (np.linalg.LinAlgError, scipy.linalg.LinAlgError, ValueError, Exception):
            P_curr = np.eye(m) * 1e6
    else:
        P_curr = np.asarray(P0, dtype=float).copy()

    if a0 is None:
        if np.any(c_vec != 0.0):
            try:
                a_curr = np.linalg.solve(np.eye(m) - T_mat, c_vec)
            except np.linalg.LinAlgError:
                a_curr = np.zeros(m)
        else:
            a_curr = np.zeros(m)
    else:
        a_curr = np.asarray(a0, dtype=float).copy()

    # Initialize da0 and dP0 for each parameter
    da_curr = [np.zeros(m) for _ in range(K)]
    dP_curr = [np.zeros((m, m)) for _ in range(K)]

    for j, s in enumerate(sens_list):
        if s.da0 is not None:
            da_curr[j] = np.asarray(s.da0, dtype=float).copy()
        elif a0 is None:
            if np.any(s.dT != 0.0) or np.any(s.dc != 0.0):
                try:
                    da_curr[j] = np.linalg.solve(
                        np.eye(m) - T_mat, s.dT @ a_curr + s.dc
                    )
                except np.linalg.LinAlgError:
                    da_curr[j] = np.zeros(m)

        if s.dP0 is not None:
            dP_curr[j] = np.asarray(s.dP0, dtype=float).copy()
        elif P0 is None:
            # Differentiate Lyapunov equation: dP0 - T dP0 T' = dT P0 T' + T P0 dT' + d(RQR)
            dRQR = (
                s.dR @ Q_mat @ R_mat.T
                + R_mat @ s.dQ @ R_mat.T
                + R_mat @ Q_mat @ s.dR.T
            )
            rhs_dP = s.dT @ P_curr @ T_mat.T + T_mat @ P_curr @ s.dT.T + dRQR
            try:
                dP_sol = scipy.linalg.solve_discrete_lyapunov(T_mat, rhs_dP)
                dP_curr[j] = 0.5 * (dP_sol + dP_sol.T)
            except (np.linalg.LinAlgError, scipy.linalg.LinAlgError, ValueError, Exception):
                dP_curr[j] = np.zeros((m, m))

    log2pi = math.log(2.0 * math.pi)
    loglik = 0.0

    # Sequential forward filter and sensitivity recursion
    for t in range(T_obs):
        y_t = y_mat[t]
        obs = np.isfinite(y_t)
        n_obs = int(np.sum(obs))

        if n_obs == 0:
            # Completely missing period: propagate forward without update
            a_next = T_mat @ a_curr + c_vec
            P_next = T_mat @ P_curr @ T_mat.T + RQR
            P_next = 0.5 * (P_next + P_next.T)

            for j, s in enumerate(sens_list):
                dRQR_j = (
                    s.dR @ Q_mat @ R_mat.T
                    + R_mat @ s.dQ @ R_mat.T
                    + R_mat @ Q_mat @ s.dR.T
                )
                da_curr[j] = s.dT @ a_curr + T_mat @ da_curr[j] + s.dc
                dP_curr[j] = (
                    s.dT @ P_curr @ T_mat.T
                    + T_mat @ dP_curr[j] @ T_mat.T
                    + T_mat @ P_curr @ s.dT.T
                    + dRQR_j
                )
                dP_curr[j] = 0.5 * (dP_curr[j] + dP_curr[j].T)

            a_curr = a_next
            P_curr = P_next
            continue

        # Sub-select active observables for period t
        if n_obs == n_y:
            y_obs = y_t
            Z_t = Z_mat
            d_t = d_vec
            H_t = H_mat
        else:
            y_obs = y_t[obs]
            Z_t = Z_mat[obs, :]
            d_t = d_vec[obs]
            H_t = H_mat[np.ix_(obs, obs)]

        # Baseline forecast error and covariance
        v = y_obs - Z_t @ a_curr - d_t
        F = Z_t @ P_curr @ Z_t.T + H_t
        F = 0.5 * (F + F.T)

        try:
            F_chol = np.linalg.cholesky(F)
            log_det = 2.0 * np.sum(np.log(np.diag(F_chol)))
            F_inv_v = np.linalg.solve(F_chol.T, np.linalg.solve(F_chol, v))
            F_inv = np.linalg.inv(F)
        except np.linalg.LinAlgError:
            F = F + 1e-10 * np.eye(n_obs)
            F_chol = scipy.linalg.cholesky(F, lower=True)
            log_det = 2.0 * np.sum(np.log(np.diag(F_chol)))
            F_inv_v = np.linalg.solve(F_chol.T, np.linalg.solve(F_chol, v))
            F_inv = np.linalg.inv(F)

        quad_form = float(v @ F_inv_v)
        ll_t = -0.5 * (n_obs * log2pi + log_det + quad_form)
        loglik += ll_t

        M_t = T_mat @ P_curr @ Z_t.T
        K_t = M_t @ F_inv
        L_t = T_mat - K_t @ Z_t

        # Pre-allocate containers for step t+1 sensitivities
        da_next = [np.zeros(m) for _ in range(K)]
        dP_next = [np.zeros((m, m)) for _ in range(K)]

        # Parameter gradient updates
        for j, s in enumerate(sens_list):
            if n_obs == n_y:
                dZ_t = s.dZ
                dd_t = s.dd
                dH_t = s.dH
            else:
                dZ_t = s.dZ[obs, :]
                dd_t = s.dd[obs]
                dH_t = s.dH[np.ix_(obs, obs)]

            dv_j = -dZ_t @ a_curr - Z_t @ da_curr[j] - dd_t
            dF_j = dZ_t @ P_curr @ Z_t.T + Z_t @ dP_curr[j] @ Z_t.T + Z_t @ P_curr @ dZ_t.T + dH_t
            dF_j = 0.5 * (dF_j + dF_j.T)

            # Likelihood score contribution at t:
            # dll_t = -0.5 * [ tr(F^{-1} dF) + 2 dv' F^{-1} v - (F^{-1} v)' dF (F^{-1} v) ]
            tr_term = float(np.sum(F_inv * dF_j))  # equivalent to trace(F_inv @ dF_j)
            grad_v_term = 2.0 * float(dv_j @ F_inv_v)
            grad_F_term = float(F_inv_v @ dF_j @ F_inv_v)

            dll_j = -0.5 * (tr_term + grad_v_term - grad_F_term)
            score[j] += dll_j

            # Sensitivities of Kalman gain and closed-loop transition
            dM_j = s.dT @ P_curr @ Z_t.T + T_mat @ dP_curr[j] @ Z_t.T + T_mat @ P_curr @ dZ_t.T
            dK_j = (dM_j - K_t @ dF_j) @ F_inv
            dL_j = s.dT - dK_j @ Z_t - K_t @ dZ_t

            dRQR_j = (
                s.dR @ Q_mat @ R_mat.T
                + R_mat @ s.dQ @ R_mat.T
                + R_mat @ Q_mat @ s.dR.T
            )

            # Joseph-stabilized covariance sensitivity update
            da_next[j] = s.dT @ a_curr + T_mat @ da_curr[j] + s.dc + dK_j @ v + K_t @ dv_j
            dP_next[j] = (
                dL_j @ P_curr @ L_t.T
                + L_t @ dP_curr[j] @ L_t.T
                + L_t @ P_curr @ dL_j.T
                + dK_j @ H_t @ K_t.T
                + K_t @ dH_t @ K_t.T
                + K_t @ H_t @ dK_j.T
                + dRQR_j
            )
            dP_next[j] = 0.5 * (dP_next[j] + dP_next[j].T)

        # Update baseline filter state and covariance for t+1
        a_curr = T_mat @ a_curr + c_vec + K_t @ v
        P_curr = L_t @ P_curr @ L_t.T + K_t @ H_t @ K_t.T + RQR
        P_curr = 0.5 * (P_curr + P_curr.T)

        da_curr = da_next
        dP_curr = dP_next

    return float(loglik), score


# ---------------------------------------------------------------------------
# Decision Rules and State Space Sensitivity Builder
# ---------------------------------------------------------------------------

def _re_solve_model(model: Any, params: dict, previous_ss: Mapping[str, float] | None = None) -> Any:
    """Re-solve a DSGE model at perturbed parameter values.

    Goes through the shared re-solves (:func:`puremacro.dsge.dynare.
    _resolve_dynare_model`, :func:`puremacro.dsge.build._resolve_build_model`):
    the steady state follows the parameters (so its derivative reaches the
    Jacobians and the measurement intercept), the declared shock covariance
    and the model's linearisation are kept, and a ``build_dynare`` model is
    solved with a predetermined set that does not depend on the parameter
    vector, so the decision rules at theta +- h have the same shape.
    ``previous_ss`` is the starting point of the steady-state solve in place
    of the model's own steady state.
    """
    if getattr(model, "_dynare_equations", None) is not None:
        from puremacro.dsge.dynare import _resolve_dynare_model

        return _resolve_dynare_model(model, params, strict=False, previous_ss=previous_ss)
    elif getattr(model, "_equations", None) is not None:
        from puremacro.dsge.build import _resolve_build_model

        return _resolve_build_model(model, params, strict=False, previous_ss=previous_ss)
    return model


def _steady_state_derivative(
    model: Any, params: Mapping[str, float], param_name: str, h: float
) -> np.ndarray | None:
    r"""``d ys / d theta`` by the implicit function theorem, or ``None``.

    With ``f(ys, theta) = 0`` the static model (every lead and lag at the
    steady state, shocks at zero), ``d ys / d theta = -J^{-1} df/dtheta``,
    where ``J = df/dys`` is taken at the model's steady state (``A_+ + A_0 +
    A_-`` for a ``build_dynare`` model) and ``df/dtheta`` is a central
    difference in ``theta`` alone at that fixed steady state. Unlike the
    difference of two re-solved steady states it does not depend on the
    re-solve deciding whether the old steady state still solves the model: a
    parameter that moves the static residual by less than the re-solve's
    exactness tolerance over the step used to be read as not moving the
    steady state at all. ``None`` when ``J`` is singular (a unit root leaves
    the level free) or anything is not finite; the caller then differences
    the re-solved steady states.
    """
    from puremacro.dsge.build import _Vec, _jacobian

    variables = list(model.variables)
    shocks = list(model.shocks)
    ys = np.asarray(pd.Series(model.steady_state).reindex(variables), dtype=float)
    if not np.all(np.isfinite(ys)) or param_name not in params:
        return None
    dyn = getattr(model, "_dynare_equations", None)
    klein = getattr(model, "_equations", None)
    if dyn is None and klein is None:
        return None
    names = list(params)
    base_vals = [float(params[k]) for k in names]

    def static(y, par_values):
        dtype = np.asarray(y).dtype
        y_v = _Vec(variables, y)
        e_v = _Vec(shocks, np.zeros(len(shocks), dtype=dtype), "shock")
        p_v = _Vec(names, par_values, "parameter")
        if dyn is not None:
            return dyn(y_v, y_v, y_v, e_v, p_v)
        return klein(y_v, y_v, e_v, p_v)

    j = names.index(param_name)
    up = list(base_vals)
    up[j] += h
    dn = list(base_vals)
    dn[j] -= h
    with np.errstate(all="ignore"):
        f_theta = (np.asarray(static(ys, up), dtype=float)
                   - np.asarray(static(ys, dn), dtype=float)) / (2.0 * h)
    if f_theta.shape != (len(variables),) or not np.all(np.isfinite(f_theta)):
        return None
    if not np.any(f_theta):
        return np.zeros(len(variables))

    c0 = getattr(model, "_A_plus", None)
    c1 = getattr(model, "_A_0", None)
    c2 = getattr(model, "_A_minus", None)
    if dyn is not None and c0 is not None and c1 is not None and c2 is not None:
        J = np.asarray(c0 + c1 + c2, dtype=np.float64)
    else:
        with np.errstate(all="ignore"):
            J = np.asarray(_jacobian(lambda y: static(y, base_vals), ys, len(variables),
                                    getattr(model, "method", "complex")), dtype=np.float64)
    if J.shape != (len(variables), len(variables)) or not np.all(np.isfinite(J)):
        return None
    sv = np.linalg.svd(J, compute_uv=False)
    if sv.size == 0 or not (sv[-1] > 1e-10 * sv[0]):
        return None
    dys = -np.linalg.solve(J, f_theta)
    return dys if np.all(np.isfinite(dys)) else None


def _decision_rule_and_ss_derivatives(
    model: Any, param_name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Implicit decision-rule derivatives with finite-difference structural inputs.

    Computes:
    1. Steady-state derivative `dys` via the Implicit Function Theorem (_steady_state_derivative).
    2. Decision rule derivatives dF = dghx and dL = dghu by solving the Generalized Sylvester equation:
       \\hat{A} \\frac{\\partial F}{\\partial \\theta_j} + A_+ \\frac{\\partial F}{\\partial \\theta_j} G =
           - \\left( \\frac{\\partial A_+}{\\partial \\theta_j} F G + \\frac{\\partial A_0}{\\partial \\theta_j} F + \\frac{\\partial A_{-, s}}{\\partial \\theta_j} \\right)
       where \\hat{A} = A_0 + A_+ F P_s, and the shock loading derivative:
       \\hat{A} \\frac{\\partial L}{\\partial \\theta_j} =
           - \\left( \\frac{\\partial A_+}{\\partial \\theta_j} F N + A_+ \\frac{\\partial F}{\\partial \\theta_j} N + \\frac{\\partial A_0}{\\partial \\theta_j} L + \\frac{\\partial B_u}{\\partial \\theta_j} \\right)
    """
    params_dict = dict(getattr(model, "_params", {}) or getattr(model, "params", {}) or {})
    if getattr(model, "_dynare_equations", None) is not None:
        from puremacro.dsge.dynare import _resolve_states

        if tuple(model.states) != tuple(_resolve_states(model)):
            model = _re_solve_model(model, params_dict)
    theta_0 = float(params_dict.get(param_name, 0.0))
    h = max(1e-6, 1e-5 * abs(theta_0))

    variables = list(model.variables)
    states = list(model.states)
    shocks = list(model.shocks)
    state_idx = [variables.index(s) for s in states]

    # 1. Steady-state sensitivity via Implicit Function Theorem
    try:
        dys_ift = _steady_state_derivative(model, params_dict, param_name, h)
    except Exception:
        dys_ift = None
    dys = np.asarray(dys_ift, dtype=float) if dys_ift is not None else np.zeros(len(variables))

    # 2. Analytical Sylvester differentiation if lead-lag equations are available
    if (getattr(model, "_dynare_equations", None) is not None
            and dys_ift is not None
            and getattr(model, "method", "complex") == "complex"):
        from puremacro.dsge.occbin import _differentiate_dynare_eqs

        ys0 = np.asarray(pd.Series(model.steady_state).reindex(variables), dtype=float)
        p_plus = dict(params_dict); p_plus[param_name] = theta_0 + h
        p_minus = dict(params_dict); p_minus[param_name] = theta_0 - h
        ss_plus = ys0 + h * dys
        ss_minus = ys0 - h * dys

        Ap_p, A0_p, Am_p, Bu_p, _, _, _, _ = _differentiate_dynare_eqs(
            model._dynare_equations, variables, shocks, p_plus, ss_plus
        )
        Ap_m, A0_m, Am_m, Bu_m, _, _, _, _ = _differentiate_dynare_eqs(
            model._dynare_equations, variables, shocks, p_minus, ss_minus
        )

        dA_plus = (Ap_p - Ap_m) / (2.0 * h)
        dA_0 = (A0_p - A0_m) / (2.0 * h)
        dA_minus = (Am_p - Am_m) / (2.0 * h)
        dA_minus_s = dA_minus[:, state_idx]
        dB_u = (Bu_p - Bu_m) / (2.0 * h)

        A_plus = np.asarray(model._A_plus, dtype=float)
        A_0 = np.asarray(model._A_0, dtype=float)

        F = np.asarray(model.dynare_dr.ghx, dtype=float)
        L = np.asarray(model.dynare_dr.ghu, dtype=float)
        G = F[state_idx, :]
        N = L[state_idx, :]

        # State selector matrix P_s such that G = P_s @ F
        P_s = np.zeros((len(states), len(variables)))
        for i, s in enumerate(states):
            P_s[i, variables.index(s)] = 1.0

        A_hat = A_0 + A_plus @ F @ P_s
        B = A_plus
        C = G
        D = - (dA_plus @ F @ G + dA_0 @ F + dA_minus_s)

        # Solve Sylvester equation for dF = dghx
        dF = solve_sylvester_generalized(A_hat, B, C, D)

        # Solve linear system for dL = dghu
        R_u = - (dA_plus @ F @ N + A_plus @ dF @ N + dA_0 @ L + dB_u)
        try:
            dL = np.linalg.solve(A_hat, R_u)
        except np.linalg.LinAlgError:
            dL = np.linalg.lstsq(A_hat, R_u, rcond=None)[0]

        return dF, dL, dys

    # 3. Re-solve when the static derivative is unavailable, or when the
    # model requires finite differences (e.g. non-holomorphic equations).
    p_plus = dict(params_dict); p_plus[param_name] = theta_0 + h
    p_minus = dict(params_dict); p_minus[param_name] = theta_0 - h
    prev_plus = prev_minus = None
    if dys_ift is not None:
        ys0 = np.asarray(pd.Series(model.steady_state).reindex(variables), dtype=float)
        prev_plus = dict(zip(variables, ys0 + h * dys))
        prev_minus = dict(zip(variables, ys0 - h * dys))
    m_plus = _re_solve_model(model, p_plus, previous_ss=prev_plus)
    m_minus = _re_solve_model(model, p_minus, previous_ss=prev_minus)
    if dys_ift is None:
        dys = (
            np.asarray(pd.Series(m_plus.steady_state).reindex(variables), dtype=float)
            - np.asarray(pd.Series(m_minus.steady_state).reindex(variables), dtype=float)
        ) / (2.0 * h)
    if hasattr(m_plus, "dynare_dr") and m_plus.dynare_dr is not None:
        ghx_p = np.asarray(m_plus.dynare_dr.ghx, dtype=float)
        ghx_m = np.asarray(m_minus.dynare_dr.ghx, dtype=float)
        ghu_p = np.asarray(m_plus.dynare_dr.ghu, dtype=float)
        ghu_m = np.asarray(m_minus.dynare_dr.ghu, dtype=float)
        return (ghx_p - ghx_m) / (2.0 * h), (ghu_p - ghu_m) / (2.0 * h), dys

    if hasattr(m_plus, "solution") and hasattr(m_plus.solution, "G"):
        G_p = np.asarray(m_plus.solution.G, dtype=float)
        G_m = np.asarray(m_minus.solution.G, dtype=float)
        N_p = np.asarray(m_plus.solution.N, dtype=float)
        N_m = np.asarray(m_minus.solution.N, dtype=float)
        return (G_p - G_m) / (2.0 * h), (N_p - N_m) / (2.0 * h), dys

    raise ValueError("Model does not expose decision rules.")


def compute_decision_rule_derivatives(
    model: Any,
    param_name: str,
    h_eps: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Compute analytical parameter derivatives of decision rules \frac{\partial ghx}{\partial \theta} and \frac{\partial ghu}{\partial \theta}.

    Parameters
    ----------
    model : LinearModel
        The solved first-order DSGE model.
    param_name : str
        The structural parameter name to differentiate.
    h_eps : float, default 1e-7
        Probe step for differentiation.

    Returns
    -------
    dghx : np.ndarray of shape (N_var, N_states)
        Derivative \frac{\partial ghx}{\partial \theta}.
    dghu : np.ndarray of shape (N_var, N_shocks)
        Derivative \frac{\partial ghu}{\partial \theta}.

    Notes
    -----
    For regular complex-step models the policy derivative is obtained from
    a generalized Sylvester equation; its structural matrix derivatives
    use central parameter differences and an implicit steady-state derivative.
    Other models are re-solved at ``theta +- h``. A ``build_dynare`` /
    ``load_mod`` model uses the re-solve state set
    (:func:`puremacro.dsge.dynare._resolve_states`): the
    model's states plus every variable that enters with a lag. ``N_states``
    is the size of that set, which exceeds ``len(model.states)`` when a lag
    coefficient is 0 at the calibration; the derivative with respect to that
    coefficient is then non-zero, as it must be. ``h_eps`` is unused (the
    step is ``max(1e-6, 1e-5 |theta|)``) and kept for backward compatibility.
    """
    dghx, dghu, _ = _decision_rule_and_ss_derivatives(model, param_name)
    return dghx, dghu


def build_state_space_sensitivities(
    model: Any,
    varobs: Sequence[str],
    param_names: Sequence[str],
    shock_cov: np.ndarray | None = None,
    measurement_errors: Mapping[str, float] | None = None,
) -> dict[str, StateSpaceSensitivity]:
    r"""Build exact analytical StateSpaceSensitivity objects for all specified parameters.

    Parameters
    ----------
    model : LinearModel
        The solved DSGE model.
    varobs : Sequence[str]
        Observable variable names.
    param_names : Sequence[str]
        Names of structural parameters, shock standard errors (stderr e), or measurement errors.
    shock_cov : np.ndarray, optional
        Baseline covariance matrix of exogenous shocks.
    measurement_errors : Mapping[str, float], optional
        Baseline measurement error standard deviations for observables.

    Returns
    -------
    sensitivities : StateSpaceSensitivities
        Dictionary mapping each parameter name to its StateSpaceSensitivity.
        It also carries ``.state_space``, the state space those derivatives
        belong to, and ``.model``, the solved model it was built from; pass
        ``.state_space`` to :func:`kalman_score` with them. For a
        ``build_dynare`` / ``load_mod`` model whose calibration sets a lag
        coefficient to 0 (``rho = 0`` in ``y = rho*y(-1) + ...``) the
        derivatives are taken on the re-solve state set, which includes that
        lag, so a state space built from the original model has fewer states
        and does not match them (``kalman_score`` raises a ``ValueError``
        saying so).
    """
    from puremacro.dsge.observation import make_state_space_from_varobs

    if getattr(model, "_dynare_equations", None) is not None:
        # The perturbed solves use the re-solve state set (the model's states
        # plus every variable that enters with a lag); the base state space
        # must have the same one, so a model whose calibration hides a lag
        # (coefficient 0) is re-solved on it first. The likelihood is
        # unchanged by the inert extra states; only the derivatives gain them.
        from puremacro.dsge.dynare import _resolve_states

        if tuple(model.states) != tuple(_resolve_states(model)):
            model = _re_solve_model(model, dict(getattr(model, "_params", None) or {}))

    obs = list(varobs)
    n_obs = len(obs)
    shocks = list(model.shocks)
    n_e = len(shocks)
    states = list(model.states)
    n_s = len(states)
    variables = list(model.variables)
    m_dim = n_s + n_e

    base_ssm = make_state_space_from_varobs(
        model, obs, shock_cov=shock_cov, measurement_error=measurement_errors
    )

    sens_dict = StateSpaceSensitivities()
    sens_dict.state_space = base_ssm
    sens_dict.model = model

    for name in param_names:
        dT = np.zeros((m_dim, m_dim))
        dZ = np.zeros((n_obs, m_dim))
        dR = np.zeros((m_dim, n_e))
        dQ = np.zeros((n_e, n_e))
        dH = np.zeros((n_obs, n_obs))
        dc = np.zeros(m_dim)
        dd = np.zeros(n_obs)

        # Case 1: Measurement error standard deviation: stderr obs_name
        if name.startswith("stderr_") and name[7:] in obs:
            obs_nm = name[7:]
            k = obs.index(obs_nm)
            sigma_me = float(measurement_errors.get(obs_nm, 0.0)) if measurement_errors else 0.0
            dH[k, k] = 2.0 * sigma_me
            sens_dict[name] = StateSpaceSensitivity(
                dT=dT, dZ=dZ, dR=dR, dQ=dQ, dH=dH, dc=dc, dd=dd
            )
            continue

        # Case 2: Shock standard deviation: stderr shock_name
        if name.startswith("stderr_") and name[7:] in shocks:
            sh_nm = name[7:]
            k = shocks.index(sh_nm)
            sigma_s = math.sqrt(max(base_ssm.Q[k, k], 0.0))
            dQ[k, k] = 2.0 * sigma_s
            sens_dict[name] = StateSpaceSensitivity(
                dT=dT, dZ=dZ, dR=dR, dQ=dQ, dH=dH, dc=dc, dd=dd
            )
            continue

        # Case 3: Structural parameter
        dghx, dghu, dys = _decision_rule_and_ss_derivatives(model, name)

        state_indices = [variables.index(s) for s in states]
        obs_indices = [variables.index(o) for o in obs]

        dT[:n_s, :n_s] = dghx[state_indices, :]
        dT[:n_s, n_s:] = dghu[state_indices, :]

        dZ[:, :n_s] = dghx[obs_indices, :]
        dZ[:, n_s:] = dghu[obs_indices, :]

        # The measurement intercept is the observables' steady state (level
        # units), so it moves with any parameter that moves the steady state
        # (SW07's robs with csigma or constebeta). A log-unit observable's
        # intercept is log(ss): d log(ss) = d ss / ss.
        units = getattr(model, "units", {}) or {}
        ys = np.asarray(
            pd.Series(model.steady_state).reindex(variables), dtype=float
        )
        for k, (o, r) in enumerate(zip(obs, obs_indices)):
            if units.get(o, "level") == "log":
                dd[k] = dys[r] / ys[r] if ys[r] > 0 else 0.0
            else:
                dd[k] = dys[r]

        sens_dict[name] = StateSpaceSensitivity(
            dT=dT, dZ=dZ, dR=dR, dQ=dQ, dH=dH, dc=dc, dd=dd
        )

    return sens_dict


# ---------------------------------------------------------------------------
# Diagnostics and Result Containers
# ---------------------------------------------------------------------------

@dataclass
class ScoreDiagnosticsResult:
    """Diagnostic report for analytical likelihood gradients against numerical verification."""

    loglik: float
    gradient: np.ndarray
    param_names: tuple[str, ...]
    elapsed_sec: float
    verification_df: pd.DataFrame | None = None

    def verify_numerical_gradient(
        self,
        y: np.ndarray,
        ssm_func: Callable[[dict], StateSpaceModel],
        base_params: dict,
        h: float = 1e-6,
    ) -> pd.DataFrame:
        """Compare analytical gradient against 5-point central finite differences."""
        names = self.param_names
        K = len(names)
        records = []

        for i, nm in enumerate(names):
            val0 = float(base_params[nm])
            step = max(h, h * abs(val0))

            def eval_ll(delta: float) -> float:
                p = dict(base_params)
                p[nm] = val0 + delta
                ssm = ssm_func(p)
                from puremacro.state_space import kalman_filter
                R = np.asarray(ssm.R, dtype=float)
                RQR = R @ ssm.Q @ R.T
                P0 = scipy.linalg.solve_discrete_lyapunov(ssm.T, RQR)
                a0 = np.linalg.solve(np.eye(ssm.T.shape[0]) - ssm.T, np.asarray(ssm.c, dtype=float))
                out = kalman_filter(y, ssm, a0=a0, P0=P0)
                return float(out["loglik"])

            f_p2 = eval_ll(2 * step)
            f_p1 = eval_ll(step)
            f_m1 = eval_ll(-step)
            f_m2 = eval_ll(-2 * step)

            fd_score = (-f_p2 + 8 * f_p1 - 8 * f_m1 + f_m2) / (12 * step)
            an_score = float(self.gradient[i])

            abs_err = abs(an_score - fd_score)
            rel_err = abs_err / max(1.0, abs(fd_score))
            passed = rel_err < 1e-4 or abs_err < 1e-5

            records.append({
                "parameter": nm,
                "analytical": an_score,
                "numerical_fd": fd_score,
                "abs_error": abs_err,
                "rel_error": rel_err,
                "status": "PASS" if passed else "WARN",
            })

        df = pd.DataFrame(records)
        self.verification_df = df
        return df

    def to_markdown(self) -> str:
        """Format score diagnostics as a clean Markdown table."""
        if self.verification_df is not None:
            return _df_to_markdown(self.verification_df, digits=6)
        df = pd.DataFrame({
            "parameter": list(self.param_names),
            "score": list(self.gradient),
        })
        return _df_to_markdown(df, digits=6)

    def to_latex(self) -> str:
        """Format score diagnostics as LaTeX tabular."""
        df = self.verification_df if self.verification_df is not None else pd.DataFrame({
            "parameter": list(self.param_names),
            "score": list(self.gradient),
        })
        return _df_to_latex(df, digits=6)

    def to_typst(self) -> str:
        """Format score diagnostics as Typst table."""
        df = self.verification_df if self.verification_df is not None else pd.DataFrame({
            "parameter": list(self.param_names),
            "score": list(self.gradient),
        })
        return _df_to_typst(df, digits=6)


# ---------------------------------------------------------------------------
# Combined Posterior and Analytical Gradient Evaluator
# ---------------------------------------------------------------------------

def log_posterior_and_gradient(
    vec: np.ndarray,
    names: Sequence[str],
    model_template: Any,
    y_data: np.ndarray,
    varobs: Sequence[str],
    priors: dict,
    fixed_params: dict | None = None,
) -> tuple[float, np.ndarray]:
    r"""Evaluate combined log-posterior \ln p(\theta \mid Y) and analytical gradient \nabla_\theta \ln p(\theta \mid Y).

    Parameters
    ----------
    vec : np.ndarray of shape (K,)
        Estimated parameter vector.
    names : Sequence[str]
        Structural parameter names corresponding to elements of vec.
        Noise-scale and correlation parameters require the observation-equation
        likelihood used by ``model.estimate(..., method="nuts")`` instead.
    model_template : LinearModel or ParsedModelDAG
        Template model for re-evaluating structural matrices.
    y_data : np.ndarray of shape (T_obs, n_y)
        Observed series data.
    varobs : Sequence[str]
        Observable variable names.
    priors : dict
        Prior distributions dictionary.
    fixed_params : dict, optional
        Fixed parameters dictionary.

    Returns
    -------
    log_post : float
        Combined log-posterior value \ln L(Y \mid \theta) + \ln p(\theta).
    grad_post : np.ndarray of shape (K,)
        Combined analytical gradient vector \nabla_\theta \ln p(\theta \mid Y).
    """
    param_dict = dict(getattr(model_template, "_params", {}) or {})
    unsupported = sorted(set(names) - set(param_dict))
    if unsupported:
        raise ValueError(
            "log_posterior_and_gradient supports structural parameters only; "
            f"unsupported names: {unsupported}. Use model.estimate for noise "
            "scale or correlation parameters."
        )
    if fixed_params:
        param_dict.update(fixed_params)
    for nm, v in zip(names, vec):
        param_dict[nm] = float(v)

    # 1. Log-prior and prior gradient
    lp = log_prior(param_dict, priors)
    if not math.isfinite(lp):
        return -math.inf, np.zeros(len(names))

    g_prior = grad_log_prior(vec, priors, names)

    # 2. Re-solve model at current parameters
    from puremacro.dsge.observation import make_state_space_from_varobs

    try:
        # The shared re-solve (steady state at theta, declared shock
        # covariance, one predetermined set for theta and theta +- h). The
        # build() branch used to pass check_steady_state=, which build() does
        # not take, so every draw of a build() model raised and scored -inf.
        m_curr = _re_solve_model(model_template, param_dict)
    except (ValueError, ArithmeticError, np.linalg.LinAlgError, Exception):
        return -math.inf, np.zeros(len(names))

    # 3. Build state-space and sensitivities
    try:
        ssm = make_state_space_from_varobs(m_curr, varobs)
        sens = build_state_space_sensitivities(m_curr, varobs, names)
        ll, g_ll = kalman_score(y_data, ssm, sens)
    except (ValueError, ArithmeticError, np.linalg.LinAlgError, Exception):
        return -math.inf, np.zeros(len(names))

    if not math.isfinite(ll):
        return -math.inf, np.zeros(len(names))

    total_log_post = float(ll + lp)
    total_grad = g_ll + g_prior

    return total_log_post, total_grad
