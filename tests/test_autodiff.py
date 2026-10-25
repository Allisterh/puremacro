"""Tests for puremacro._autodiff (DualArray, HyperDualArray, dual_solve, dual_sylvester_solve)."""
from __future__ import annotations

import numpy as np
import pytest

from puremacro._autodiff import (
    DualArray,
    HyperDualArray,
    dual_solve,
    dual_sylvester_solve,
    hessian_forward,
    jacobian_forward,
    seed_dual_vector,
    value_and_grad,
)
from puremacro.dsge._gradients import sylvester_order2_tangent
from puremacro.vfi.analytic_gradients import compute_ift_gradients


def _complex_step_jacobian(fn, x: np.ndarray, h: float = 1e-20) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    f0 = np.asarray(fn(x_arr), dtype=np.float64)
    P = x_arr.size
    jac = np.zeros(f0.shape + (P,), dtype=np.float64)
    for j in range(P):
        xc = x_arr.astype(np.complex128)
        xc[j] += 1j * h
        jac[..., j] = np.imag(fn(xc)) / h
    return jac


def test_dual_array_ufuncs_and_complex_step_parity():
    def macro_euler(theta):
        alpha = theta[0]
        beta = theta[1]
        gamma = theta[2]
        delta = theta[3]
        k = np.array([1.2, 2.5, 3.8])
        y = k ** alpha
        c = 0.65 * y
        kp = y + (1.0 - delta) * k - c
        mpk = alpha * (kp ** (alpha - 1.0)) + (1.0 - delta)
        cp = 0.65 * (kp ** alpha)
        lhs = c ** (-gamma)
        rhs = beta * (cp ** (-gamma)) * mpk
        return np.concatenate([lhs - rhs, np.array([np.sum(np.log(c) + np.exp(-0.1 * kp))])])

    x0 = np.array([0.33, 0.96, 2.0, 0.08])
    val, jac_ad = jacobian_forward(macro_euler, x0)
    jac_cs = _complex_step_jacobian(macro_euler, x0)

    assert np.allclose(val, macro_euler(x0), atol=1e-15)
    assert np.max(np.abs(jac_ad - jac_cs)) < 1e-14


def test_value_and_grad_and_matmul():
    W = np.array([[2.0, -0.5, 0.2], [-0.5, 1.5, 0.3], [0.2, 0.3, 1.2]])
    b = np.array([0.4, -0.7, 1.1])

    def obj(x):
        r = (W @ x) - b
        return 0.5 * (r @ r) + np.sum(np.sin(x) * np.cos(x))

    x0 = np.array([0.5, -0.3, 0.8])
    val, grad = value_and_grad(obj, x0)
    grad_cs = _complex_step_jacobian(obj, x0).reshape(-1)
    assert np.isclose(val, obj(x0), atol=1e-15)
    assert np.max(np.abs(grad - grad_cs)) < 1e-14


def test_hessian_forward_hyper_dual_exactness():
    # 1. Transcendental function vs 4th-order central finite differences
    def ces_welfare(theta):
        x, y, z = theta[0], theta[1], theta[2]
        return np.log(0.4 * (x ** 0.6) + 0.6 * (y ** 0.6)) / 0.6 + np.exp(0.3 * x * z) / (1.0 + y * z)

    x0 = np.array([1.4, 0.9, 0.5])
    val, grad, hess = hessian_forward(ces_welfare, x0)

    h = 1e-3
    P = len(x0)
    hess_4th = np.zeros((P, P))
    for j in range(P):
        e = np.zeros(P)
        e[j] = 1.0
        _, g_p2 = jacobian_forward(ces_welfare, x0 + 2 * h * e)
        _, g_p1 = jacobian_forward(ces_welfare, x0 + h * e)
        _, g_m1 = jacobian_forward(ces_welfare, x0 - h * e)
        _, g_m2 = jacobian_forward(ces_welfare, x0 - 2 * h * e)
        hess_4th[:, j] = (-g_p2 + 8.0 * g_p1 - 8.0 * g_m1 + g_m2) / (12.0 * h)

    assert np.max(np.abs(hess - hess_4th)) < 1e-10
    assert np.allclose(hess, hess.T, atol=1e-15)

    # 2. Quadratic form with matrix multiplication
    W = np.array([[2.0, -0.5, 0.2], [-0.5, 1.5, 0.3], [0.2, 0.3, 1.2]])
    b = np.array([0.4, -0.7, 1.1])

    def obj(x):
        r = (W @ x) - b
        return 0.5 * (r @ r) + np.sum(np.sin(x) * np.cos(x))

    x_quad = np.array([0.5, -0.3, 0.8])
    _, _, hess_q = hessian_forward(obj, x_quad)
    hess_exact = W.T @ W + np.diag(-2.0 * np.sin(2.0 * x_quad))
    assert np.allclose(hess_q, hess_exact, atol=1e-14)


def test_dual_solve_and_dual_sylvester_solve():
    def build_mats(theta):
        a, b = theta[0], theta[1]
        A = np.array([[2.0 + a, 0.3 * b], [-0.2 * a, 1.8 + b * b]])
        B = np.array([[0.4 * a, -0.1], [0.05, 0.3 * b]])
        C = np.array([[0.7 * a, 0.1], [0.0, 0.5 * b]])
        D = np.array([[1.0 + a * b, -0.5 + a], [0.3 - b, 0.8 + a * a]])
        return A, B, C, D

    theta0 = np.array([0.8, 0.6])

    # 1. dual_solve
    def solve_lin(theta):
        A, _, _, D = build_mats(theta)
        return dual_solve(A, D[:, 0])

    x_val, x_jac = jacobian_forward(solve_lin, theta0)
    h_cs = 1e-20
    for j in range(2):
        tc = theta0.astype(np.complex128)
        tc[j] += 1j * h_cs
        Ac, _, _, Dc = build_mats(tc)
        cs = np.imag(np.linalg.solve(Ac, Dc[:, 0])) / h_cs
        assert np.max(np.abs(x_jac[:, j] - cs)) < 1e-14

    # 2. dual_sylvester_solve & sylvester_order2_tangent
    def solve_sylv(theta):
        A, B, G, D = build_mats(theta)
        C_kron = np.stack([
            np.concatenate([G[0, 0] * G[0], G[0, 1] * G[0]]),
            np.concatenate([G[0, 0] * G[1], G[0, 1] * G[1]]),
            np.concatenate([G[1, 0] * G[0], G[1, 1] * G[0]]),
            np.concatenate([G[1, 0] * G[1], G[1, 1] * G[1]]),
        ], axis=0)
        D_xx = np.concatenate([D, 0.5 * D], axis=1)
        return dual_sylvester_solve(A, B, C_kron, D_xx)

    X_val, X_jac = jacobian_forward(solve_sylv, theta0)
    _, dA = jacobian_forward(lambda t: build_mats(t)[0], theta0)
    _, dB = jacobian_forward(lambda t: build_mats(t)[1], theta0)
    _, dG = jacobian_forward(lambda t: build_mats(t)[2], theta0)
    _, dD = jacobian_forward(lambda t: np.concatenate([build_mats(t)[3], 0.5 * build_mats(t)[3]], axis=1), theta0)
    A0, B0, G0, D0 = build_mats(theta0)
    D_xx0 = np.concatenate([D0, 0.5 * D0], axis=1)
    X_t, dX_t = sylvester_order2_tangent(
        A0,
        B0,
        G0,
        D_xx0,
        dA_hat=np.moveaxis(dA, -1, 0),
        dB=np.moveaxis(dB, -1, 0),
        dG=np.moveaxis(dG, -1, 0),
        dD_xx=np.moveaxis(dD, -1, 0),
    )
    assert np.allclose(X_val, X_t, atol=1e-13)
    assert np.allclose(np.moveaxis(X_jac, -1, 0), dX_t, atol=1e-13)


def test_vfi_ift_autodiff_matches_central():
    class _MockSolution:
        def __init__(self, coeffs):
            self.coefficients = coeffs

    class _MockProblem:
        def __init__(self):
            self.params = {"alpha": 0.35, "delta": 0.08}
            self.beta = 0.96

    def residual_fn(c, p):
        alpha = p["alpha"]
        delta = p["delta"]
        beta = p["beta"]
        nodes = np.array([1.0, 1.5, 2.0])
        return c - beta * (alpha * (nodes ** alpha) + (1.0 - delta) * nodes) * 0.5 + 0.1 * (c ** 2)

    prob = _MockProblem()
    # Solve R(c*; theta) = 0
    nodes = np.array([1.0, 1.5, 2.0])
    rhs = 0.96 * (0.35 * (nodes ** 0.35) + 0.92 * nodes) * 0.5
    c_star = (-1.0 + np.sqrt(1.0 + 0.4 * rhs)) / 0.2
    sol = _MockSolution(c_star)

    res_ad = compute_ift_gradients(
        sol, prob, params=["alpha", "beta", "delta"], residual_fn=residual_fn, diff_method="autodiff"
    )
    res_fd = compute_ift_gradients(
        sol, prob, params=["alpha", "beta", "delta"], residual_fn=residual_fn, diff_method="central"
    )
    assert np.max(np.abs(res_ad.grad_coefficients - res_fd.grad_coefficients)) < 1e-9
