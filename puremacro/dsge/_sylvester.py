r"""Generalized Real Schur / Bartels-Stewart Sylvester solver for DSGE perturbation models.

Solves the higher-order state curvature generalized Sylvester equations:
    Order 1: A_hat X + A_+ X h_x = -K_x_ss
    Order 2: A_hat X + A_+ X (h_x \otimes h_x) = -K_xx
    Order 3: A_hat X + A_+ X (h_x \otimes h_x \otimes h_x) = -K_xxx

using the Real Schur decomposition of the state transition matrix h_x:
    h_x = U_h T_h U_h^T

where U_h is orthogonal and T_h is quasi-upper-triangular with 1x1 and 2x2 diagonal blocks.
Because h_x^{\otimes k} = U_h^{\otimes k} T_h^{\otimes k} (U_h^{\otimes k})^T, the transformed
system is block upper-triangular with small diagonal blocks of size m \in {1, 2, 4} (order 2)
or m \in {1, 2, 4, 8} (order 3) and is solved recursively level-by-level in real arithmetic
via Bartels-Stewart block back-substitution.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
import scipy.linalg


def _schur_blocks(T: np.ndarray, tol: float = 1e-14) -> list[tuple[int, int]]:
    """Partition quasi-upper-triangular Real Schur matrix T into 1x1 and 2x2 diagonal blocks."""
    n = T.shape[0]
    blocks: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if i + 1 < n and abs(float(T[i + 1, i])) > tol:
            blocks.append((i, i + 2))
            i += 2
        else:
            blocks.append((i, i + 1))
            i += 1
    return blocks


def _solve_scalar_block(
    A_hat: np.ndarray,
    A_plus: np.ndarray,
    tau: float,
    rhs: np.ndarray,
    lu_cache: OrderedDict[float, tuple[Any, Any, int]],
    max_cache: int,
    dgetrf: Any,
    dgetrs: Any,
) -> np.ndarray:
    """Solve (A_hat + tau * A_plus) x = rhs in real arithmetic with bounded LU caching."""
    key = round(float(tau), 12)
    if key in lu_cache:
        lu, piv, info = lu_cache[key]
        lu_cache.move_to_end(key)
    else:
        Mj = A_hat + tau * A_plus
        lu, piv, info = dgetrf(Mj)
        if len(lu_cache) >= max_cache:
            lu_cache.popitem(last=False)
        lu_cache[key] = (lu, piv, info)

    if info == 0:
        sol, solve_info = dgetrs(lu, piv, rhs)
        if solve_info == 0 and np.isfinite(sol).all():
            return sol
    Mj = A_hat + tau * A_plus
    return np.linalg.lstsq(Mj, rhs, rcond=None)[0]


def _solve_quasi_block(
    A_hat: np.ndarray,
    A_plus: np.ndarray,
    S: np.ndarray,
    D_block: np.ndarray,
) -> np.ndarray:
    """Solve A_hat X_B + A_plus X_B S = D_block for a small quasi-triangular block S of size m <= 8."""
    N, m = D_block.shape
    if m == 1:
        Mj = A_hat + float(S[0, 0]) * A_plus
        try:
            sol = np.linalg.solve(Mj, D_block[:, 0])
            if np.isfinite(sol).all():
                return sol[:, None]
        except np.linalg.LinAlgError:
            pass
        return np.linalg.lstsq(Mj, D_block[:, 0], rcond=None)[0][:, None]

    # Form small (N*m, N*m) real block system in column-major order:
    # (I_m \otimes A_hat + S^T \otimes A_plus) vec_F(X_B) = vec_F(D_block)
    lhs = np.kron(np.eye(m), A_hat) + np.kron(S.T, A_plus)
    rhs = D_block.reshape(N * m, order="F")
    try:
        sol_vec = np.linalg.solve(lhs, rhs)
        if not np.isfinite(sol_vec).all():
            sol_vec = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    except np.linalg.LinAlgError:
        sol_vec = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return sol_vec.reshape((N, m), order="F")


def solve_generalized_sylvester_schur(
    A_hat: np.ndarray,
    A_plus: np.ndarray,
    h_x: np.ndarray,
    K_xx: np.ndarray,
    max_cache: int = 16,
) -> np.ndarray:
    r"""Solve A_hat X + A_+ X (h_x \otimes h_x) = -K_xx via Real Schur Bartels-Stewart recursion.

    Parameters
    ----------
    A_hat : np.ndarray
        Coefficient matrix on X of shape (N, N), typically (A_0 + A_+ g_x P_s).
    A_plus : np.ndarray
        Lead coefficient matrix of shape (N, N).
    h_x : np.ndarray
        First-order state transition matrix of shape (n_x, n_x).
    K_xx : np.ndarray
        Second-order right-hand side curvature matrix of shape (N, n_x^2).
    max_cache : int, default 16
        Maximum number of real LU factorizations to cache.

    Returns
    -------
    g_xx : np.ndarray
        Second-order policy curvature matrix of shape (N, n_x^2).
    """
    A_hat = np.asarray(A_hat, dtype=float)
    A_plus = np.asarray(A_plus, dtype=float)
    h_x = np.asarray(h_x, dtype=float)
    K_xx = np.asarray(K_xx, dtype=float)

    N = A_hat.shape[0]
    n_x = h_x.shape[0]
    M = n_x * n_x

    if n_x == 0 or M == 0:
        return np.zeros((N, 0), dtype=float)

    if np.all(K_xx == 0.0):
        return np.zeros((N, M), dtype=float)

    if np.all(h_x == 0.0):
        rhs = -K_xx
        try:
            sol = np.linalg.solve(A_hat, rhs)
            if not np.isfinite(sol).all():
                sol = np.linalg.lstsq(A_hat, rhs, rcond=None)[0]
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(A_hat, rhs, rcond=None)[0]
        return np.ascontiguousarray(np.real(sol), dtype=float)

    # 1. Real Schur decomposition: h_x = U_h @ T_h @ U_h^T
    T_h, U_h = scipy.linalg.schur(h_x, output="real")
    blocks = _schur_blocks(T_h)

    # 2. Transform RHS via mode contractions without forming U_h \otimes U_h:
    # \tilde{D} = -K_xx @ (U_h \otimes U_h)
    K_tensor = (-K_xx).reshape(N, n_x, n_x)
    temp = np.tensordot(K_tensor, U_h, axes=(2, 0))
    D_tilde = np.tensordot(temp, U_h, axes=(1, 0)).transpose(0, 2, 1)
    D_tilde = np.ascontiguousarray(D_tilde, dtype=float)
    del temp

    # 3. Block-triangular Bartels-Stewart sweep
    X_tilde = np.zeros((N, n_x, n_x), dtype=float)
    W = np.zeros((N, n_x, n_x), dtype=float)  # caches A_plus @ X_tilde
    dgetrf = scipy.linalg.lapack.dgetrf
    dgetrs = scipy.linalg.lapack.dgetrs
    lu_cache: OrderedDict[float, tuple[Any, Any, int]] = OrderedDict()

    for i1, e1 in blocks:
        m1 = e1 - i1
        T1 = T_h[i1:e1, i1:e1]

        # Subtract Region 1 couplings (k1 < i1, all k2) in one vectorized contraction
        if i1 > 0:
            T1_off = T_h[:i1, i1:e1]
            if np.any(T1_off != 0.0):
                tmp1 = np.tensordot(W[:, :i1, :], T1_off, axes=(1, 0))  # (N, n_x, m1)
                D_tilde[:, i1:e1, :] -= np.tensordot(tmp1, T_h, axes=(1, 0))  # (N, m1, n_x)

        for i2, e2 in blocks:
            m2 = e2 - i2
            T2 = T_h[i2:e2, i2:e2]

            # Subtract Region 2 couplings (k1 in [i1, e1), k2 < i2)
            if i2 > 0:
                T2_off = T_h[:i2, i2:e2]
                if np.any(T2_off != 0.0):
                    if m1 == 1 and m2 == 1:
                        D_tilde[:, i1, i2] -= float(T1[0, 0]) * (W[:, i1, :i2] @ T2_off[:, 0])
                    else:
                        tmp2 = np.tensordot(W[:, i1:e1, :i2], T1, axes=(1, 0))  # (N, i2, m1)
                        D_tilde[:, i1:e1, i2:e2] -= np.tensordot(tmp2, T2_off, axes=(1, 0))  # (N, m1, m2)

            # Solve diagonal block (b1, b2) of size m = m1 * m2 in {1, 2, 4}
            if m1 == 1 and m2 == 1:
                tau = float(T1[0, 0] * T2[0, 0])
                sol = _solve_scalar_block(
                    A_hat, A_plus, tau, D_tilde[:, i1, i2],
                    lu_cache, max_cache, dgetrf, dgetrs,
                )
                X_tilde[:, i1, i2] = sol
                W[:, i1, i2] = A_plus @ sol
            else:
                S = np.kron(T1, T2)
                D_blk = D_tilde[:, i1:e1, i2:e2].reshape(N, m1 * m2)
                sol_blk = _solve_quasi_block(A_hat, A_plus, S, D_blk)
                X_tilde[:, i1:e1, i2:e2] = sol_blk.reshape(N, m1, m2)
                W[:, i1:e1, i2:e2] = (A_plus @ sol_blk).reshape(N, m1, m2)

    lu_cache.clear()
    del lu_cache, D_tilde, W

    # 4. Back-transform: X = \tilde{X} @ (U_h \otimes U_h)^T
    U_h_T = U_h.T
    temp_back = np.tensordot(X_tilde, U_h_T, axes=(2, 0))
    g_xx = np.tensordot(temp_back, U_h_T, axes=(1, 0)).transpose(0, 2, 1).reshape(N, M)
    return np.ascontiguousarray(g_xx, dtype=float)


def solve_order3_sylvester_schur(
    A_hat: np.ndarray,
    A_plus: np.ndarray,
    h_x: np.ndarray,
    K_xxx: np.ndarray,
    max_cache: int = 64,
) -> np.ndarray:
    r"""Solve A_hat X + A_+ X (h_x \otimes h_x \otimes h_x) = -K_xxx via Real Schur Bartels-Stewart recursion.

    Parameters
    ----------
    A_hat : np.ndarray
        Coefficient matrix on X of shape (N, N), typically (A_0 + A_+ g_x P_s).
    A_plus : np.ndarray
        Lead coefficient matrix of shape (N, N).
    h_x : np.ndarray
        First-order state transition matrix of shape (n_x, n_x).
    K_xxx : np.ndarray
        Third-order right-hand side curvature matrix of shape (N, n_x^3).
    max_cache : int, default 64
        Maximum number of real LU factorizations to cache (< 2 MB memory).

    Returns
    -------
    g_xxx : np.ndarray
        Third-order policy curvature matrix of shape (N, n_x^3).
    """
    A_hat = np.asarray(A_hat, dtype=float)
    A_plus = np.asarray(A_plus, dtype=float)
    h_x = np.asarray(h_x, dtype=float)
    K_xxx = np.asarray(K_xxx, dtype=float)

    N = A_hat.shape[0]
    n_x = h_x.shape[0]
    M = n_x**3

    if n_x == 0 or M == 0:
        return np.zeros((N, 0), dtype=float)

    if np.all(K_xxx == 0.0):
        return np.zeros((N, M), dtype=float)

    if np.all(h_x == 0.0):
        rhs = -K_xxx
        try:
            sol = np.linalg.solve(A_hat, rhs)
            if not np.isfinite(sol).all():
                sol = np.linalg.lstsq(A_hat, rhs, rcond=None)[0]
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(A_hat, rhs, rcond=None)[0]
        return np.ascontiguousarray(np.real(sol), dtype=float)

    # 1. Real Schur decomposition: h_x = U_h @ T_h @ U_h^T
    T_h, U_h = scipy.linalg.schur(h_x, output="real")
    blocks = _schur_blocks(T_h)

    # 2. Transform RHS via 3 successive real tensordot contractions without dense Kronecker
    K_tensor = (-K_xxx).reshape(N, n_x, n_x, n_x)
    temp1 = np.tensordot(K_tensor, U_h, axes=(3, 0))
    temp2 = np.tensordot(temp1, U_h, axes=(2, 0))
    D_tilde = np.tensordot(temp2, U_h, axes=(1, 0)).transpose(0, 3, 2, 1)
    D_tilde = np.ascontiguousarray(D_tilde, dtype=float)
    del temp1, temp2, K_tensor

    # 3. 3-level recursive Bartels-Stewart back-substitution in real arithmetic
    X_tilde = np.zeros((N, n_x, n_x, n_x), dtype=float)
    W = np.zeros((N, n_x, n_x, n_x), dtype=float)  # caches A_plus @ X_tilde
    dgetrf = scipy.linalg.lapack.dgetrf
    dgetrs = scipy.linalg.lapack.dgetrs
    lu_cache: OrderedDict[float, tuple[Any, Any, int]] = OrderedDict()

    for i1, e1 in blocks:
        m1 = e1 - i1
        T1 = T_h[i1:e1, i1:e1]

        # Level 1 coupling: k1 < i1 (contracted across all k2, k3 once per b1)
        if i1 > 0:
            T1_off = T_h[:i1, i1:e1]
            if np.any(T1_off != 0.0):
                u1 = np.tensordot(W[:, :i1, :, :], T1_off, axes=(1, 0))     # (N, n_x, n_x, m1)
                u2 = np.tensordot(u1, T_h, axes=(1, 0))                      # (N, n_x, m1, n_x)
                u3 = np.tensordot(u2, T_h, axes=(1, 0))                      # (N, m1, n_x, n_x)
                D_tilde[:, i1:e1, :, :] -= u3

        for i2, e2 in blocks:
            m2 = e2 - i2
            T2 = T_h[i2:e2, i2:e2]

            # Level 2 coupling: k1 in [i1, e1), k2 < i2 (contracted across all k3 once per (b1, b2))
            if i2 > 0:
                T2_off = T_h[:i2, i2:e2]
                if np.any(T2_off != 0.0):
                    v1 = np.tensordot(W[:, i1:e1, :i2, :], T1, axes=(1, 0))  # (N, i2, n_x, m1)
                    v2 = np.tensordot(v1, T2_off, axes=(1, 0))               # (N, n_x, m1, m2)
                    v3 = np.tensordot(v2, T_h, axes=(1, 0))                  # (N, m1, m2, n_x)
                    D_tilde[:, i1:e1, i2:e2, :] -= v3

            T12 = np.kron(T1, T2) if (m1 > 1 or m2 > 1) else None
            t12_scalar = float(T1[0, 0] * T2[0, 0]) if (m1 == 1 and m2 == 1) else 0.0

            for i3, e3 in blocks:
                m3 = e3 - i3
                T3 = T_h[i3:e3, i3:e3]

                # Level 3 coupling: k1 in [i1, e1), k2 in [i2, e2), k3 < i3
                if i3 > 0:
                    T3_off = T_h[:i3, i3:e3]
                    if np.any(T3_off != 0.0):
                        if m1 == 1 and m2 == 1 and m3 == 1:
                            D_tilde[:, i1, i2, i3] -= t12_scalar * (W[:, i1, i2, :i3] @ T3_off[:, 0])
                        else:
                            w1 = np.tensordot(W[:, i1:e1, i2:e2, :i3], T1, axes=(1, 0))  # (N, m2, i3, m1)
                            w2 = np.tensordot(w1, T2, axes=(1, 0))                        # (N, i3, m1, m2)
                            w3 = np.tensordot(w2, T3_off, axes=(1, 0))                    # (N, m1, m2, m3)
                            D_tilde[:, i1:e1, i2:e2, i3:e3] -= w3

                # Solve diagonal block (b1, b2, b3) of size m = m1 * m2 * m3 in {1, 2, 4, 8}
                if m1 == 1 and m2 == 1 and m3 == 1:
                    tau = t12_scalar * float(T3[0, 0])
                    sol = _solve_scalar_block(
                        A_hat, A_plus, tau, D_tilde[:, i1, i2, i3],
                        lu_cache, max_cache, dgetrf, dgetrs,
                    )
                    X_tilde[:, i1, i2, i3] = sol
                    W[:, i1, i2, i3] = A_plus @ sol
                else:
                    S = np.kron(T12 if T12 is not None else T1, T3)
                    D_blk = D_tilde[:, i1:e1, i2:e2, i3:e3].reshape(N, m1 * m2 * m3)
                    sol_blk = _solve_quasi_block(A_hat, A_plus, S, D_blk)
                    X_tilde[:, i1:e1, i2:e2, i3:e3] = sol_blk.reshape(N, m1, m2, m3)
                    W[:, i1:e1, i2:e2, i3:e3] = (A_plus @ sol_blk).reshape(N, m1, m2, m3)

    lu_cache.clear()
    del lu_cache, D_tilde, W

    # 4. Back-transform: X = \tilde{X} @ (U_h \otimes U_h \otimes U_h)^T
    U_h_T = U_h.T
    t1_b = np.tensordot(X_tilde, U_h_T, axes=(3, 0))
    t2_b = np.tensordot(t1_b, U_h_T, axes=(2, 0))
    t3_b = np.tensordot(t2_b, U_h_T, axes=(1, 0)).transpose(0, 3, 2, 1)
    g_xxx = t3_b.reshape(N, M)
    return np.ascontiguousarray(g_xxx, dtype=float)


def solve_generalized_sylvester_kronecker(
    A_hat: np.ndarray,
    A_plus: np.ndarray,
    h_x: np.ndarray,
    K_xx: np.ndarray,
    max_cache: int = 8,
) -> np.ndarray:
    r"""Solve the generalized Sylvester equation A_hat X + A_+ X (h_x \otimes h_x) = -K_xx.

    Routes through :func:`solve_generalized_sylvester_schur` (Real Schur /
    Bartels-Stewart block-triangular solver).
    """
    return solve_generalized_sylvester_schur(
        A_hat, A_plus, h_x, K_xx, max_cache=max_cache,
    )


def solve_order3_sylvester_kronecker(
    A_hat: np.ndarray,
    A_plus: np.ndarray,
    h_x: np.ndarray,
    K_xxx: np.ndarray,
    max_cache: int = 64,
) -> np.ndarray:
    r"""Solve generalized Sylvester equation A_hat X + A_+ X (h_x \otimes h_x \otimes h_x) = -K_xxx.

    Routes through :func:`solve_order3_sylvester_schur` (Real Schur /
    Bartels-Stewart block-triangular solver).
    """
    return solve_order3_sylvester_schur(
        A_hat, A_plus, h_x, K_xxx, max_cache=max_cache,
    )


def solve_order1_sylvester(
    A_hat: np.ndarray,
    A_plus: np.ndarray,
    h_x: np.ndarray,
    K_x_ss: np.ndarray,
    max_cache: int = 64,
) -> np.ndarray:
    r"""Solve generalized Sylvester equation A_hat X + A_+ X h_x = -K_x_ss.

    Solves for cross-state risk correction tensors such as g_x_sigmasigma
    via Real Schur / Bartels-Stewart block back-substitution.
    """
    A_hat = np.asarray(A_hat, dtype=float)
    A_plus = np.asarray(A_plus, dtype=float)
    h_x = np.asarray(h_x, dtype=float)
    K_x_ss = np.asarray(K_x_ss, dtype=float)

    N = A_hat.shape[0]
    n_x = h_x.shape[0]

    if n_x == 0:
        return np.zeros((N, 0), dtype=float)

    if np.all(K_x_ss == 0.0):
        return np.zeros((N, n_x), dtype=float)

    if np.all(h_x == 0.0):
        rhs = -K_x_ss
        try:
            sol = np.linalg.solve(A_hat, rhs)
            if not np.isfinite(sol).all():
                sol = np.linalg.lstsq(A_hat, rhs, rcond=None)[0]
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(A_hat, rhs, rcond=None)[0]
        return np.ascontiguousarray(np.real(sol), dtype=float)

    T_h, U_h = scipy.linalg.schur(h_x, output="real")
    blocks = _schur_blocks(T_h)

    D_tilde = np.ascontiguousarray((-K_x_ss) @ U_h, dtype=float)
    X_tilde = np.zeros((N, n_x), dtype=float)

    dgetrf = scipy.linalg.lapack.dgetrf
    dgetrs = scipy.linalg.lapack.dgetrs
    lu_cache: OrderedDict[float, tuple[Any, Any, int]] = OrderedDict()

    for i1, e1 in blocks:
        m1 = e1 - i1
        T1 = T_h[i1:e1, i1:e1]
        if m1 == 1:
            tau = float(T1[0, 0])
            sol = _solve_scalar_block(
                A_hat, A_plus, tau, D_tilde[:, i1],
                lu_cache, max_cache, dgetrf, dgetrs,
            )
            X_tilde[:, i1] = sol
            if e1 < n_x:
                row = T_h[i1, e1:]
                nz = np.flatnonzero(row)
                if len(nz) > 0:
                    wj = A_plus @ sol
                    D_tilde[:, e1 + nz] -= np.outer(wj, row[nz])
        else:
            sol_blk = _solve_quasi_block(A_hat, A_plus, T1, D_tilde[:, i1:e1])
            X_tilde[:, i1:e1] = sol_blk
            if e1 < n_x:
                T_off = T_h[i1:e1, e1:]
                if np.any(T_off != 0.0):
                    D_tilde[:, e1:] -= (A_plus @ sol_blk) @ T_off

    lu_cache.clear()
    del lu_cache, D_tilde

    g_x_ss = X_tilde @ U_h.T
    return np.ascontiguousarray(g_x_ss, dtype=float)

