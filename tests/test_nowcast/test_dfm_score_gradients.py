"""Unit tests for analytical state-space likelihood gradients (kalman_score)
applied to mixed-frequency dynamic factor and state-space models with missing
observations and ragged edges.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.dsge._gradients import StateSpaceSensitivity, kalman_score
from puremacro.nowcast.dfm import DynamicFactorModel
from puremacro.state_space import StateSpaceModel


def _build_synthetic_mixed_frequency_dfm_panel(
    T_obs: int = 40,
    n_monthly: int = 3,
    rho: float = 0.75,
    sigma_f: float = 0.5,
    seed: int = 123,
):
    """Generate synthetic mixed-frequency macroeconomic dataset.

    Variables:
    - n_monthly monthly indicators: y_{j,t} = Lambda_j * F_t + e_{j,t}
    - 1 quarterly aggregate: y^Q_t = (1/3) * Lambda_Q * (F_t + F_{t-1} + F_{t-2}) + e^Q_t
      (observed only at t=2, 5, 8, ... and NaN elsewhere)
    - Ragged edge at latest months (staggered publication delays).
    """
    rng = np.random.default_rng(seed)
    F = np.zeros(T_obs)
    for t in range(1, T_obs):
        F[t] = rho * F[t - 1] + rng.normal(scale=sigma_f)

    # Loadings
    lambda_m = np.array([1.2, 0.8, -0.6])[:n_monthly]
    lambda_q = 1.0

    H_m_diag = np.array([0.15, 0.20, 0.25])[:n_monthly]
    H_q_val = 0.08

    Y = np.zeros((T_obs, n_monthly + 1))
    for j in range(n_monthly):
        Y[:, j] = lambda_m[j] * F + rng.normal(scale=np.sqrt(H_m_diag[j]), size=T_obs)

    # Quarterly aggregate: Mariano-Murasawa 3-month triangular sum
    Y[:, n_monthly] = np.nan
    for t in range(2, T_obs):
        if (t + 1) % 3 == 0:
            Y[t, n_monthly] = lambda_q * np.mean(F[t - 2 : t + 1]) + rng.normal(
                scale=np.sqrt(H_q_val)
            )

    # Introduce realistic missing observations / ragged edge
    # Entire missing observation in month 8 for series 1
    Y[8, 1] = np.nan
    # Completely missing period across all monthly indicators at t=14
    Y[14, :n_monthly] = np.nan
    # Ragged edge at end of panel:
    # Month T-1: only series 0 published
    Y[-1, 1:] = np.nan
    # Month T-2: series 2 and quarterly missing
    Y[-2, 2:] = np.nan

    return Y, lambda_m, lambda_q, H_m_diag, H_q_val


def test_kalman_score_mixed_frequency_dfm_ragged_edge():
    """Verify analytical kalman_score on mixed-frequency DFM with ragged edges and
    missing observations matches numerical derivatives."""
    T_obs = 36
    n_monthly = 3
    rho = 0.70
    sigma_f = 0.4
    Y, lambda_m, lambda_q, H_m_diag, H_q_val = _build_synthetic_mixed_frequency_dfm_panel(
        T_obs=T_obs, n_monthly=n_monthly, rho=rho, sigma_f=sigma_f, seed=42
    )

    k = 1  # 1 factor
    p = 3  # 3 lags for quarterly aggregation companion state
    m = k * p
    n = n_monthly + 1

    # Companion transition matrix A:
    # F_t = rho * F_{t-1} + u_t
    # F_{t-1} = F_{t-1}
    # F_{t-2} = F_{t-2}
    A = np.zeros((m, m))
    A[0, 0] = rho
    A[1, 0] = 1.0
    A[2, 1] = 1.0

    # Shock covariance Q
    Q = np.zeros((m, m))
    Q[0, 0] = sigma_f ** 2

    # Observation loading matrix Z
    Z = np.zeros((n, m))
    for j in range(n_monthly):
        Z[j, 0] = lambda_m[j]
    # Quarterly average: (1/3) * lambda_q * (F_t + F_{t-1} + F_{t-2})
    Z[n_monthly, 0] = lambda_q / 3.0
    Z[n_monthly, 1] = lambda_q / 3.0
    Z[n_monthly, 2] = lambda_q / 3.0

    # Observation covariance H
    H = np.diag(np.concatenate([H_m_diag, [H_q_val]]))

    ssm = StateSpaceModel(T=A, Z=Z, Q=Q, H=H)

    # Structural sensitivities:
    # 1. wrt factor persistence rho
    dA_drho = np.zeros((m, m))
    dA_drho[0, 0] = 1.0

    # 2. wrt monthly factor loading lambda_m[0]
    dZ_dlam0 = np.zeros((n, m))
    dZ_dlam0[0, 0] = 1.0

    # 3. wrt quarterly factor loading lambda_q
    dZ_dlamq = np.zeros((n, m))
    dZ_dlamq[n_monthly, 0] = 1.0 / 3.0
    dZ_dlamq[n_monthly, 1] = 1.0 / 3.0
    dZ_dlamq[n_monthly, 2] = 1.0 / 3.0

    sens = {
        "rho": StateSpaceSensitivity(
            dT=dA_drho,
            dZ=np.zeros_like(Z),
            dR=np.zeros((m, m)),
            dQ=np.zeros_like(Q),
            dH=np.zeros_like(H),
            dc=np.zeros(m),
            dd=np.zeros(n),
        ),
        "lambda_m0": StateSpaceSensitivity(
            dT=np.zeros_like(A),
            dZ=dZ_dlam0,
            dR=np.zeros((m, m)),
            dQ=np.zeros_like(Q),
            dH=np.zeros_like(H),
            dc=np.zeros(m),
            dd=np.zeros(n),
        ),
        "lambda_q": StateSpaceSensitivity(
            dT=np.zeros_like(A),
            dZ=dZ_dlamq,
            dR=np.zeros((m, m)),
            dQ=np.zeros_like(Q),
            dH=np.zeros_like(H),
            dc=np.zeros(m),
            dd=np.zeros(n),
        ),
    }

    loglik, score = kalman_score(Y, ssm, sens)

    # 1. Assert finite likelihood and gradients despite missing values and ragged edges
    assert np.isfinite(loglik)
    assert len(score) == 3
    assert np.all(np.isfinite(score))

    # 2. Verify exact analytical scores against central numerical finite differences
    eps = 1e-6

    # Test score wrt rho
    ssm_rho_p = StateSpaceModel(T=A + eps * dA_drho, Z=Z, Q=Q, H=H)
    ssm_rho_m = StateSpaceModel(T=A - eps * dA_drho, Z=Z, Q=Q, H=H)
    ll_rp, _ = kalman_score(Y, ssm_rho_p, {})
    ll_rm, _ = kalman_score(Y, ssm_rho_m, {})
    score_num_rho = (ll_rp - ll_rm) / (2.0 * eps)
    rel_err_rho = abs(score[0] - score_num_rho) / max(1.0, abs(score_num_rho))
    assert rel_err_rho < 1e-5

    # Test score wrt lambda_m0
    ssm_lam0_p = StateSpaceModel(T=A, Z=Z + eps * dZ_dlam0, Q=Q, H=H)
    ssm_lam0_m = StateSpaceModel(T=A, Z=Z - eps * dZ_dlam0, Q=Q, H=H)
    ll_lp, _ = kalman_score(Y, ssm_lam0_p, {})
    ll_lm, _ = kalman_score(Y, ssm_lam0_m, {})
    score_num_lam0 = (ll_lp - ll_lm) / (2.0 * eps)
    rel_err_lam0 = abs(score[1] - score_num_lam0) / max(1.0, abs(score_num_lam0))
    assert rel_err_lam0 < 1e-5

    # Test score wrt lambda_q
    ssm_lamq_p = StateSpaceModel(T=A, Z=Z + eps * dZ_dlamq, Q=Q, H=H)
    ssm_lamq_m = StateSpaceModel(T=A, Z=Z - eps * dZ_dlamq, Q=Q, H=H)
    ll_qp, _ = kalman_score(Y, ssm_lamq_p, {})
    ll_qm, _ = kalman_score(Y, ssm_lamq_m, {})
    score_num_lamq = (ll_qp - ll_qm) / (2.0 * eps)
    rel_err_lamq = abs(score[2] - score_num_lamq) / max(1.0, abs(score_num_lamq))
    assert rel_err_lamq < 1e-5


def test_dfm_class_ssm_score_and_property():
    """Verify DynamicFactorModel exposes .ssm property and produces valid likelihood
    and analytical score gradients."""
    rng = np.random.default_rng(99)
    T = 40
    n = 5
    X = rng.normal(size=(T, n))
    # Add ragged edge
    X[-1, 2:] = np.nan
    X[-2, 3:] = np.nan
    X[5, 1] = np.nan

    model = DynamicFactorModel(n_factors=1, p=1, standardize=True)
    with pytest.raises(RuntimeError, match="must be fitted"):
        _ = model.ssm

    model.fit(X)
    assert model.ssm is not None
    assert isinstance(model.ssm, StateSpaceModel)
    assert model.ssm.T.shape == (1, 1)
    assert model.ssm.Z.shape == (n, 1)

    # Standardized observations
    Xs = (X - model.result_.means) / model.result_.stds

    # Sensitivities wrt factor persistence A[0, 0]
    sens = {
        "A": StateSpaceSensitivity(
            dT=np.ones((1, 1)),
            dZ=np.zeros((n, 1)),
            dR=np.zeros((1, 1)),
            dQ=np.zeros((1, 1)),
            dH=np.zeros((n, n)),
            dc=np.zeros(1),
            dd=np.zeros(n),
        )
    }

    ll, score = kalman_score(Xs, model.ssm, sens)
    assert np.isfinite(ll)
    assert np.isfinite(score[0])

    eps = 1e-6
    ssm_p = StateSpaceModel(T=model.ssm.T + eps, Z=model.ssm.Z, Q=model.ssm.Q, H=model.ssm.H)
    ssm_m = StateSpaceModel(T=model.ssm.T - eps, Z=model.ssm.Z, Q=model.ssm.Q, H=model.ssm.H)
    ll_p, _ = kalman_score(Xs, ssm_p, {})
    ll_m, _ = kalman_score(Xs, ssm_m, {})
    score_num = (ll_p - ll_m) / (2.0 * eps)
    assert abs(score[0] - score_num) / max(1.0, abs(score_num)) < 1e-5
