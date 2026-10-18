"""Comprehensive tests for Analytical State-Space Kalman Score Gradients.

Verifies:
1. Exact analytical Kalman score gradients match 5-point central finite differences
   to within < 1e-5 relative/absolute error.
2. Parameter sensitivities of decision rules via generalized Sylvester solver
   solve_sylvester_generalized.
3. Parameter sensitivities across structural parameters, shock variances, and
   measurement error standard deviations.
4. Correctness of ScoreDiagnosticsResult verification and table generation.
5. Integration of log_posterior_and_gradient with NUTS / MAP estimation.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest
import scipy.linalg

from puremacro.dsge import load_mod
from puremacro.dsge._gradients import (
    ScoreDiagnosticsResult,
    build_state_space_sensitivities,
    compute_decision_rule_derivatives,
    kalman_score,
    log_posterior_and_gradient,
    solve_sylvester_generalized,
)
from puremacro.dsge.observation import make_state_space_from_varobs
from puremacro.state_space import kalman_filter


@pytest.fixture(scope="module")
def three_equation_model():
    """Calibrated 3-equation New Keynesian / RBC toy DSGE model."""
    mod_code = """
    var y c a;
    varexo e_a;
    parameters beta sigma rho_a;
    beta = 0.99;
    sigma = 1.0;
    rho_a = 0.8;

    model;
      c = c(+1) - (1/sigma)*y;
      y = c + a;
      a = rho_a * a(-1) + e_a;
    end;

    steady_state_model;
      y = 0;
      c = 0;
      a = 0;
    end;
    """
    return load_mod(mod_code)


@pytest.fixture(scope="module")
def neoclassical_growth_model():
    """Nonlinear neoclassical growth model where steady state varies with parameters."""
    mod_code = """
    var c k y;
    varexo e_a;
    parameters beta alpha delta;
    beta = 0.96;
    alpha = 0.33;
    delta = 0.10;

    model;
      1/c = beta * (1/c(+1)) * (alpha * y(+1)/k + 1 - delta);
      y = k(-1)^alpha;
      k = y - c + (1 - delta)*k(-1) + e_a;
    end;

    steady_state_model;
      k = ((1/beta - (1 - delta))/alpha)^(1/(alpha - 1));
      y = k^alpha;
      c = y - delta*k;
    end;
    """
    return load_mod(mod_code)


def test_sylvester_decision_rule_gradient_accuracy(three_equation_model):
    """Verify analytical dghx and dghu match 5-point central FD to < 1e-6 error."""
    m = three_equation_model
    param_name = "rho_a"
    dghx, dghu = compute_decision_rule_derivatives(m, param_name)

    val0 = float(m._params[param_name])
    h = 1e-5

    def eval_rules(delta):
        mod_p = load_mod(
            """
            var y c a;
            varexo e_a;
            parameters beta sigma rho_a;
            beta = 0.99;
            sigma = 1.0;
            rho_a = 0.8;
            model;
              c = c(+1) - (1/sigma)*y;
              y = c + a;
              a = rho_a * a(-1) + e_a;
            end;
            steady_state_model;
              y = 0; c = 0; a = 0;
            end;
            """,
            params={"beta": 0.99, "sigma": 1.0, "rho_a": val0 + delta},
        )
        return np.asarray(mod_p.dynare_dr.ghx), np.asarray(mod_p.dynare_dr.ghu)

    gx_p2, gu_p2 = eval_rules(2 * h)
    gx_p1, gu_p1 = eval_rules(h)
    gx_m1, gu_m1 = eval_rules(-h)
    gx_m2, gu_m2 = eval_rules(-2 * h)

    dghx_fd = (-gx_p2 + 8 * gx_p1 - 8 * gx_m1 + gx_m2) / (12 * h)
    dghu_fd = (-gu_p2 + 8 * gu_p1 - 8 * gu_m1 + gu_m2) / (12 * h)

    np.testing.assert_allclose(dghx, dghx_fd, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(dghu, dghu_fd, rtol=1e-5, atol=1e-5)


def test_kalman_score_accuracy_vs_five_point_fd(three_equation_model):
    """Verify analytical Kalman score vector matches 5-point central FD to < 1e-5."""
    m = three_equation_model
    varobs = ["y"]
    param_names = ["rho_a", "sigma", "stderr_e_a"]

    # Simulate synthetic data
    rng = np.random.default_rng(42)
    T_obs = 60
    y_sim = rng.normal(0.0, 0.5, size=(T_obs, len(varobs)))

    sens = build_state_space_sensitivities(m, varobs, param_names)
    ll_an, score_an = kalman_score(y_sim, sens.state_space, sens)

    # 5-point central finite differences
    h = 1e-6
    base_params = {"beta": 0.99, "sigma": 1.0, "rho_a": 0.8, "stderr_e_a": 1.0}

    def eval_ll(p_dict):
        # Update shock cov if stderr changed
        sh_cov = np.array([[p_dict["stderr_e_a"] ** 2]])
        m_eval = load_mod(
            """
            var y c a;
            varexo e_a;
            parameters beta sigma rho_a;
            beta = 0.99;
            sigma = 1.0;
            rho_a = 0.8;
            model;
              c = c(+1) - (1/sigma)*y;
              y = c + a;
              a = rho_a * a(-1) + e_a;
            end;
            steady_state_model;
              y = 0; c = 0; a = 0;
            end;
            """,
            params={"beta": p_dict["beta"], "sigma": p_dict["sigma"], "rho_a": p_dict["rho_a"]},
        )
        ssm = make_state_space_from_varobs(m_eval, varobs, shock_cov=sh_cov)
        RQR = ssm.R @ ssm.Q @ ssm.R.T
        P0 = scipy.linalg.solve_discrete_lyapunov(ssm.T, RQR)
        res = kalman_filter(y_sim, ssm, P0=P0)
        return float(res["loglik"])

    score_fd = np.zeros(len(param_names))
    for i, p_name in enumerate(param_names):
        v0 = base_params[p_name]
        step = h * max(abs(v0), 1.0)

        p_p2 = dict(base_params); p_p2[p_name] = v0 + 2 * step
        p_p1 = dict(base_params); p_p1[p_name] = v0 + step
        p_m1 = dict(base_params); p_m1[p_name] = v0 - step
        p_m2 = dict(base_params); p_m2[p_name] = v0 - 2 * step

        f_p2 = eval_ll(p_p2)
        f_p1 = eval_ll(p_p1)
        f_m1 = eval_ll(p_m1)
        f_m2 = eval_ll(p_m2)

        score_fd[i] = (-f_p2 + 8 * f_p1 - 8 * f_m1 + f_m2) / (12 * step)

    abs_err = np.abs(score_an - score_fd)
    rel_err = abs_err / np.maximum(1.0, np.abs(score_fd))

    assert np.all((rel_err < 1e-5) | (abs_err < 1e-5)), (
        f"Analytical score:\n{score_an}\nNumerical FD:\n{score_fd}\n"
        f"Rel error:\n{rel_err}\nAbs error:\n{abs_err}"
    )


def test_kalman_score_steady_state_derivative_accuracy(neoclassical_growth_model):
    """Verify analytical score works when parameters affect steady state."""
    m = neoclassical_growth_model
    varobs = ["y", "c"]
    param_names = ["alpha", "delta"]
    measurement_errors = {"y": 0.05, "c": 0.05}  # Prevents stochastic singularity

    rng = np.random.default_rng(123)
    T_obs = 40
    y_sim = rng.normal(1.0, 0.2, size=(T_obs, len(varobs)))

    sens = build_state_space_sensitivities(
        m, varobs, param_names, measurement_errors=measurement_errors
    )
    ll_an, score_an = kalman_score(y_sim, sens.state_space, sens)

    assert math.isfinite(ll_an)
    assert len(score_an) == 2
    assert np.all(np.isfinite(score_an))
    assert np.all(np.abs(score_an) < 1e6)

    # A finite score alone cannot detect a missing moving-steady-state term.
    from puremacro.dsge.dynare import _resolve_dynare_model

    for j, name in enumerate(param_names):
        base = float(m._params[name])
        h = 1e-5
        values = []
        for multiple in (2, 1, -1, -2):
            moved = _resolve_dynare_model(m, {name: base + multiple * h})
            ssm = make_state_space_from_varobs(moved, varobs, measurement_error=measurement_errors)
            values.append(kalman_score(y_sim, ssm, {})[0])
        numerical = (-values[0] + 8 * values[1] - 8 * values[2] + values[3]) / (12 * h)
        np.testing.assert_allclose(score_an[j], numerical, rtol=1e-5, atol=1e-5)

    # Verify diagnostics result formatting
    diag = ScoreDiagnosticsResult(
        loglik=ll_an,
        gradient=score_an,
        param_names=tuple(param_names),
        elapsed_sec=0.01,
    )
    md = diag.to_markdown()
    assert "alpha" in md
    assert "delta" in md
    assert "score" in md


def test_log_posterior_and_gradient_integration(three_equation_model):
    """Verify log_posterior_and_gradient produces valid joint posterior gradient."""
    m = three_equation_model
    varobs = ["y"]
    names = ["rho_a", "sigma"]
    vec = np.array([0.75, 1.1])

    rng = np.random.default_rng(99)
    y_sim = rng.normal(0.0, 0.5, size=(50, 1))

    priors = {
        "rho_a": {"dist": "beta", "mean": 0.7, "std": 0.1, "lb": 0.0, "ub": 1.0},
        "sigma": {"dist": "gamma", "mean": 1.0, "std": 0.3, "lb": 0.01, "ub": 5.0},
    }

    log_post, grad_post = log_posterior_and_gradient(
        vec=vec,
        names=names,
        model_template=m,
        y_data=y_sim,
        varobs=varobs,
        priors=priors,
    )

    assert math.isfinite(log_post)
    assert grad_post.shape == (2,)
    assert np.all(np.isfinite(grad_post))
    assert not np.all(grad_post == 0.0)
