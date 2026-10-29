"""Native model likelihoods used by the protocol adapters.

The DSGE score differentiates the Kalman recursion using the existing
state-space sensitivities (structural derivatives may use finite differences).
The DAG score uses central differences of the native Whittle likelihood.
Neither path substitutes a demonstration model for the supplied model.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def parameter_vector(params: Any, names: Sequence[str], defaults: np.ndarray) -> np.ndarray:
    if isinstance(params, Mapping):
        unknown = set(params) - set(names)
        if unknown:
            raise ValueError(f"Unknown adapter parameter(s): {sorted(unknown)}; expected {tuple(names)}")
        theta = np.array([params.get(name, defaults[i]) for i, name in enumerate(names)], dtype=float)
    else:
        theta = defaults.copy() if params is None else np.asarray(params, dtype=float)
    if theta.shape != (len(names),) or not np.all(np.isfinite(theta)):
        raise ValueError(f"Expected {len(names)} finite parameters in order {tuple(names)}")
    return theta


def observations(data: Any, observables: Sequence[str] | None) -> tuple[np.ndarray, tuple[str, ...]]:
    if data is None:
        raise ValueError("Observed data is required for a native model likelihood")
    if observables is None:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Specify observables in the data column order")
        observables = tuple(data.columns)
    names = tuple(observables)
    y = data.loc[:, list(names)].to_numpy(dtype=float) if isinstance(data, pd.DataFrame) else np.asarray(data, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    if not names or len(set(names)) != len(names) or y.ndim != 2 or y.shape[1] != len(names) or not len(y):
        raise ValueError("data must have one column per unique observable and at least one row")
    if np.isinf(y).any():
        raise ValueError("data contains infinite observations")
    return y, names


def state_space_paths(model: Any, periods: int, *, shocks: Any = None,
                      seed: Any = 0, impulse: int | None = None,
                      size: float = 1.0) -> pd.DataFrame:
    """Simulate the supplied linear state space, preserving its loadings."""
    m, n_e = model.T.shape[0], model.Q.shape[0]
    if periods < 1:
        raise ValueError("periods must be positive")
    rng = np.random.default_rng(seed)
    if impulse is not None:
        if not 0 <= impulse < n_e:
            raise ValueError("Unknown state-space shock")
        innovations = np.zeros((periods, n_e)); innovations[0, impulse] = size
        errors = np.zeros((periods, model.Z.shape[0]))
    else:
        innovations = (rng.multivariate_normal(np.zeros(n_e), model.Q, size=periods)
                       if shocks is None else np.asarray(shocks, dtype=float))
        errors = rng.multivariate_normal(np.zeros(model.Z.shape[0]), model.H, size=periods)
    if innovations.shape != (periods, n_e):
        raise ValueError("shocks shape must be (periods, n_state_shocks)")
    state = np.zeros(m)
    out = np.zeros((periods, model.Z.shape[0]))
    for t in range(periods):
        state = model.T @ state + model.R @ innovations[t] + (0 if impulse is not None else model.c)
        out[t] = model.Z @ state + errors[t] + (0 if impulse is not None else model.d)
    names = ["y"] if out.shape[1] == 1 else [f"y{i}" for i in range(out.shape[1])]
    return pd.DataFrame(out, columns=names, index=pd.RangeIndex(periods, name="t"))


def dsge_score(model: Any, names: Sequence[str], theta: np.ndarray, data: Any,
               *, observables: Sequence[str] | None, shock_cov: Any = None,
               measurement_errors: Any = None) -> tuple[float, np.ndarray]:
    from puremacro.dsge.build import LinearModel
    from puremacro.dsge._gradients import (
        StateSpaceSensitivity, StateSpaceSensitivities, _re_solve_model,
        build_state_space_sensitivities, kalman_score,
    )
    from puremacro.state_space import StateSpaceModel

    if isinstance(model, StateSpaceModel):
        if model.T.shape != (1, 1) or model.Q.shape != (1, 1) or model.H.shape != (1, 1):
            raise NotImplementedError("StateSpaceModel score adapter currently supports scalar systems only")
        expected = ("rho", "sigma_eps", "sigma_meas")
        if tuple(names) != expected:
            raise ValueError(f"Scalar state-space parameter names must be {expected}")
        rho, sig_e, sig_m = theta
        if abs(rho) >= 1 or sig_e <= 0 or sig_m < 0:
            raise ValueError("Scalar state-space likelihood requires |rho| < 1, sigma_eps > 0, sigma_meas >= 0")
        y, _ = observations(data, observables or ("y",))
        ssm = replace(model, T=np.array([[rho]]), Q=np.array([[sig_e**2]]), H=np.array([[sig_m**2]]))
        sens = {}
        for i, name in enumerate(names):
            zeros = np.zeros((1, 1))
            sens[name] = StateSpaceSensitivity(
                dT=np.ones((1, 1)) if i == 0 else zeros,
                dQ=np.array([[2*sig_e]]) if i == 1 else zeros,
                dH=np.array([[2*sig_m]]) if i == 2 else zeros,
                dZ=zeros, dR=zeros, dc=np.zeros(1), dd=np.zeros(1),
            )
        return kalman_score(y, ssm, sens)

    if not isinstance(model, LinearModel):
        raise NotImplementedError("Native DSGE likelihood requires a solved first-order LinearModel")
    base = dict(model._params or {})
    if set(names) - set(base):
        raise ValueError("DSGE adapter parameter names must identify structural model parameters")
    if names and model._equations is None and model._dynare_equations is None:
        raise NotImplementedError("Structural scores require re-solvable model equations")
    y, obs = observations(data, observables or model._varobs)
    base.update(zip(names, theta))
    current = _re_solve_model(model, base)
    sens = build_state_space_sensitivities(current, obs, names,
                                          shock_cov=shock_cov, measurement_errors=measurement_errors)
    assert isinstance(sens, StateSpaceSensitivities)
    assert sens.state_space is not None
    ssm = sens.state_space
    if np.max(np.abs(np.linalg.eigvals(ssm.T))) >= 1:
        raise ValueError("Native DSGE score currently requires a stationary first-order state space")
    return kalman_score(y, ssm, sens)


def dag_score(model: Any, names: Sequence[str], theta: np.ndarray, data: Any,
              *, steady_state: Mapping[str, float], params: Mapping[str, float],
              observables: Sequence[str] | None, shock_cov: Any,
              measurement_cov: Any = None, T_ma: int = 128,
              demean: bool = True, exclude_zero_freq: bool = False) -> tuple[float, np.ndarray]:
    from puremacro.dsge.ssj_dag import BlockDAG
    from puremacro.dsge.hank_estimate import hank_loglik_whittle

    if not isinstance(model, BlockDAG):
        raise NotImplementedError("Native HANK likelihood requires a BlockDAG")
    if not steady_state:
        raise ValueError("Supply the calibrated steady_state for the BlockDAG")
    if set(names) - set(params):
        raise ValueError("HANK adapter parameter names must be present in params")
    if shock_cov is None:
        raise ValueError("Specify shock_cov in the DAG's exogenous-variable order")
    y, obs = observations(data, observables)
    if not np.all(np.isfinite(y)):
        raise ValueError("Whittle data must be finite; missing observations are unsupported")
    known = set(model.unknowns) | set(model.exogenous)
    for block in model.blocks:
        known.update(block.outputs)
    if set(obs) - known:
        raise ValueError("Unknown DAG observables")

    def evaluate(vector: np.ndarray) -> float:
        calibration = dict(params)
        calibration.update(zip(names, vector))
        return hank_loglik_whittle(model, y, shock_cov, steady_state=steady_state,
                                   params=calibration, observables=obs, measurement_cov=measurement_cov,
                                   T_ma=T_ma, demean=demean, exclude_zero_freq=exclude_zero_freq)

    ll = evaluate(theta)
    grad = np.empty(len(theta))
    for i, val in enumerate(theta):
        step = 1e-5 * max(1.0, abs(float(val)))
        up, down = theta.copy(), theta.copy()
        up[i] += step
        down[i] -= step
        grad[i] = (evaluate(up) - evaluate(down)) / (2*step)
    return ll, grad
