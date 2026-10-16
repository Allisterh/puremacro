"""Behavioral, duality and bridge tests for puremacro.trade.household.

The first block checks the demand systems on an analytic 4x2 fixture
(benchmark normalization, Engel repair, duality, Slutsky symmetry, complex-step
derivatives, admissibility and the supernumerary-share fit). The second block
adds invariants (analytic Slutsky matrices, primal expenditure-minimization
oracle, general-base welfare, currency scaling, reporting). The third block
checks the bridges to consistent-accounting trade calibrations and equilibria
and the parity with the flexible module's smoothed LES at benchmark income.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import minimize

from puremacro.trade import (
    calibrate_trade_model,
    compute_hicksian_welfare,
    solve_trade_equilibrium,
)
from puremacro.trade.data import synthetic_codes, synthetic_icio_table
from puremacro.trade.household import (
    HouseholdCalibrationResult,
    HouseholdDemandResult,
    HouseholdDomainError,
    HouseholdPreferences,
    HouseholdWelfareResult,
    SupernumeraryFitResult,
    calibrate_household,
    compute_household_welfare,
    compute_household_welfare_from_results,
    fit_supernumerary_share,
    household_expenditure_from_calibration,
    household_prices_from_result,
)

X0 = np.array([[25., 10.], [45., 60.], [30., 30.], [0., 0.]])
ETA = np.array([[.6, .5], [.9, 1.1], [1.3, 1.2], [np.nan, np.nan]])
P = np.array([[1.3, .8], [.9, 1.25], [1.1, 1.05], [2., 3.]])
M = np.array([105., 110.])
RULES = ["fixed_baskets", "cobb_douglas", "stone_geary", "ces"]


def model(rule, **kwargs):
    return calibrate_household(X0, rule, **kwargs)


def synthetic_2c_2s_calib():
    """Balanced 2-country 2-sector synthetic model (copied from the flexible test suite)."""
    nc, ns, nfd = 2, 2, 3
    data = np.zeros((ns * nc + 3, ns * nc + nfd * nc), dtype=float)
    data[:4, :4] = np.array([[10.0, 15.0, 5.0, 5.0], [15.0, 20.0, 10.0, 10.0],
                             [5.0, 5.0, 12.0, 18.0], [10.0, 10.0, 18.0, 22.0]])
    y = np.array([100.0, 150.0, 120.0, 180.0])
    va = y - data[:4, :4].sum(axis=0)
    taxes = 0.05 * y
    va_fac = va - taxes
    data[4, :4] = taxes
    data[5, :4] = (2.0 / 3.0) * va_fac
    data[6, :4] = (1.0 / 3.0) * va_fac
    fd_row_sums = y - data[:4, :4].sum(axis=1)
    for i in range(4):
        if i < 2:
            data[i, 4:10] = fd_row_sums[i] * np.array([0.50, 0.25, 0.05, 0.10, 0.08, 0.02])
        else:
            data[i, 4:10] = fd_row_sums[i] * np.array([0.10, 0.08, 0.02, 0.50, 0.25, 0.05])
    data[4, 4:] = 0.02 * data[:4, 4:].sum(axis=0)
    return calibrate_trade_model(data, ns=ns, nc=nc, nfd=nfd, validate=True)


def balanced_calib(nc, ns, seed, country_codes=None):
    """Balanced random (n+3, n+3*nc) table, country-major cells, home-biased final demand."""
    nfd = 3
    rng = np.random.default_rng(seed)
    n = nc * ns
    y = rng.uniform(80., 200., n)
    Z = rng.uniform(.02, .12, (n, n)) * y[None, :]
    va = y - Z.sum(0) - .04 * y
    data = np.zeros((n + 3, n + nfd * nc))
    data[:n, :n] = Z
    data[n, :n], data[n + 1, :n], data[n + 2, :n] = .04 * y, 2. / 3. * va, va / 3.
    frac = rng.uniform(.2, 1., (n, nfd * nc))
    for i in range(n):
        frac[i, (i // ns) * nfd:(i // ns + 1) * nfd] *= 4.
    frac /= frac.sum(1, keepdims=True)
    data[:n, n:] = (y - Z.sum(1))[:, None] * frac
    data[n, n:] = .02 * data[:n, n:].sum(0)
    kw = {} if country_codes is None else {"country_codes": country_codes}
    return calibrate_trade_model(data, ns=ns, nc=nc, nfd=nfd, validate=True, **kw)


@pytest.fixture(scope="module")
def calib2():
    return synthetic_2c_2s_calib()


@pytest.fixture(scope="module")
def calib32():
    return balanced_calib(3, 2, seed=32, country_codes=["AAA", "BBB", "CCC"])


@pytest.fixture(scope="module")
def consistent_pair(calib2):
    base = solve_trade_equilibrium(calib2, accounting="consistent", tol=1e-10)
    tau = np.array([.15, 0.])
    cf = solve_trade_equilibrium(calib2, tau=tau, tau_fd=tau, accounting="consistent", tol=1e-10)
    assert base.converged and cf.converged
    return base, cf, tau


# ===========================================================================
# 1. Demand systems on the analytic fixture (12 tests)
# ===========================================================================

@pytest.mark.parametrize("rule,sigma", [("fixed_baskets", 1.), ("cobb_douglas", 1.),
                                        ("stone_geary", 1.), ("ces", 0.),
                                        ("ces", .4), ("ces", 1.), ("ces", 2.)])
def test_benchmark_budget_homogeneity_and_zeros(rule, sigma):
    prefs = model(rule, ces_elasticity=sigma,
                  **({"expenditure_elasticities": ETA} if rule == "stone_geary" else {}))
    base = prefs.evaluate(np.ones_like(X0), X0.sum(axis=0), validate=True)
    assert_allclose(base.quantities, X0, atol=1e-13)
    assert_allclose(base.utility, 1., atol=3e-15)
    assert_allclose(base.cost_index, 1., atol=3e-15)
    state = prefs.evaluate(P, M, validate=True)
    assert_allclose(np.sum(P * state.quantities, axis=0), M, rtol=2e-15)
    assert_allclose(prefs.demand(P * 2.7, M * 2.7), state.quantities, rtol=2e-15)
    assert_allclose(prefs.expenditure(P * 2.7, state.utility), M * 2.7, rtol=2e-15)
    assert np.array_equal(state.quantities[-1], [0., 0.])
    assert np.array_equal(prefs.beta[-1], [0., 0.])
    assert np.array_equal(prefs.gamma[-1], [0., 0.])
    assert not prefs.x0.flags.writeable and not prefs.gamma.flags.writeable


def test_income_elasticities_repaired_and_matched_locally():
    prefs = model("stone_geary", expenditure_elasticities=ETA, supernumerary_share=[.4, .5])
    weighted_mean = np.sum(prefs.shares[:-1] * ETA[:-1], axis=0)
    assert_allclose(prefs.eta[:-1], ETA[:-1] / weighted_mean)
    assert_allclose(prefs.beta.sum(axis=0), 1.)
    assert_allclose((prefs.shares * prefs.eta).sum(axis=0), 1.)
    assert prefs.calibration.maximum_absolute_elasticity_adjustment > 0
    assert_allclose(prefs.calibration.input_weighted_elasticity_mean, weighted_mean)
    step = 1e-6
    qplus = prefs.demand(np.ones_like(X0), prefs.m0 * (1 + step))
    qminus = prefs.demand(np.ones_like(X0), prefs.m0 * (1 - step))
    observed = (qplus[:-1] - qminus[:-1]) / (2 * step * X0[:-1])
    assert_allclose(observed, prefs.eta[:-1], atol=2e-10)
    with pytest.raises(ValueError, match="adding-up"):
        model("stone_geary", expenditure_elasticities=ETA, repair_adding_up=False)


def test_sg_zero_subsistence_is_cobb_douglas():
    sg = model("stone_geary", supernumerary_share=1.)
    cd = model("cobb_douglas")
    assert_allclose(sg.gamma, 0.)
    assert_allclose(sg.demand(P, M), cd.demand(P, M), rtol=1e-15)
    assert_allclose(sg.expenditure(P, [.7, 1.3]), cd.expenditure(P, [.7, 1.3]), rtol=1e-15)
    assert_allclose(sg.welfare(P, M).ev, cd.welfare(P, M).ev, rtol=1e-15)


@pytest.mark.parametrize("rule", RULES)
def test_exact_welfare_duality_and_sign(rule):
    prefs = model(rule, ces_elasticity=.7,
                  **({"expenditure_elasticities": ETA} if rule == "stone_geary" else {}))
    state = prefs.evaluate(P, M, validate=True)
    welfare = prefs.welfare(P, M, validate=True)
    assert_allclose(prefs.expenditure(P, state.utility), M, atol=3e-14)
    assert_allclose(prefs.hicksian(P, state.utility), state.quantities, atol=3e-14)
    assert_allclose(welfare.ev, prefs.expenditure(np.ones_like(X0), state.utility) - prefs.m0, atol=3e-14)
    assert_allclose(welfare.cv, M - prefs.expenditure(P, 1.), atol=3e-14)
    assert_allclose(welfare.cv, state.price_index * welfare.ev, atol=3e-14)
    richer = prefs.welfare(np.ones_like(X0), prefs.m0 * 1.1)
    assert_allclose(richer.ev, prefs.m0 * .1, atol=3e-14)
    assert_allclose(richer.cv, richer.ev, atol=3e-14)
    assert_allclose(welfare.cost_index, prefs.expenditure(P, 1.) / prefs.m0)
    assert_allclose(welfare.ev_pct_consumption, 100 * welfare.ev / prefs.m0)
    assert welfare.metadata["positive_is_gain"] is True


@pytest.mark.parametrize("rule", RULES)
def test_shephard_and_slutsky_symmetry_negative_semidefinite(rule):
    prefs = calibrate_household(X0[:3, :1], rule, ces_elasticity=1.8,
                                **({"expenditure_elasticities": ETA[:3, :1]} if rule == "stone_geary" else {}))
    p = P[:3, :1].copy()
    utility = np.array([.9])
    budget = prefs.expenditure(p, utility)
    q = prefs.demand(p, budget)
    step = 2e-5
    hessian = np.empty((3, 3))
    marshallian = np.empty((3, 3))
    for j in range(3):
        plus, minus = p.copy(), p.copy()
        plus[j, 0] += step
        minus[j, 0] -= step
        derivative = (prefs.expenditure(plus, utility) - prefs.expenditure(minus, utility)) / (2 * step)
        assert_allclose(derivative, q[j], rtol=2e-9)
        hessian[:, j] = ((prefs.hicksian(plus, utility) - prefs.hicksian(minus, utility)) / (2 * step))[:, 0]
        marshallian[:, j] = ((prefs.demand(plus, budget) - prefs.demand(minus, budget)) / (2 * step))[:, 0]
    income_derivative = ((prefs.demand(p, budget + step) - prefs.demand(p, budget - step)) / (2 * step))[:, 0]
    slutsky = marshallian + np.outer(income_derivative, q[:, 0])
    assert_allclose(slutsky, hessian, atol=7e-8)
    assert_allclose(hessian, hessian.T, atol=7e-8)
    assert np.linalg.eigvalsh((hessian + hessian.T) / 2).max() < 8e-8
    assert_allclose(hessian @ p[:, 0], 0., atol=8e-8)
    # The analytic substitution matrix agrees with both finite-difference constructions.
    analytic = prefs.slutsky(p, budget)[:, :, 0]
    assert_allclose(analytic, hessian, atol=7e-8)
    assert_allclose(analytic, slutsky, atol=7e-8)


@pytest.mark.parametrize("rule", RULES)
def test_complex_step_derivatives_preserved(rule):
    prefs = model(rule, ces_elasticity=.7,
                  **({"expenditure_elasticities": ETA} if rule == "stone_geary" else {}))
    direction_p = np.array([[.1, -.2], [.2, .3], [-.1, .05], [.3, .3]])
    direction_m = np.array([2., -1.])
    cs = prefs.evaluate(P + 1e-25j * direction_p, M + 1e-25j * direction_m)
    plus = prefs.evaluate(P + 1e-5 * direction_p, M + 1e-5 * direction_m)
    minus = prefs.evaluate(P - 1e-5 * direction_p, M - 1e-5 * direction_m)
    for name in ("quantities", "utility", "price_index", "cost_index", "surplus"):
        assert np.iscomplexobj(getattr(cs, name))
        assert_allclose(getattr(cs, name).imag / 1e-25,
                        (getattr(plus, name) - getattr(minus, name)) / 2e-5, atol=2e-9)


def test_invalid_calibrations_and_infeasible_states_rejected():
    with pytest.raises(ValueError, match="nonempty"):
        calibrate_household([1., 2.])
    with pytest.raises(ValueError, match="cannot be negative"):
        calibrate_household([[-1.], [2.]])
    with pytest.raises(ValueError, match="every region"):
        calibrate_household([[0., 1.], [0., 2.]])
    with pytest.raises(ValueError, match="positive expenditure elasticities"):
        model("stone_geary", expenditure_elasticities=np.zeros_like(X0))
    with pytest.raises(ValueError, match="maximum admissible"):
        model("stone_geary", expenditure_elasticities=ETA, supernumerary_share=.99)
    for lam in (0., -.1, np.inf):
        with pytest.raises(ValueError):
            model("stone_geary", supernumerary_share=lam)
    with pytest.raises(ValueError, match="homothetic"):
        model("cobb_douglas", expenditure_elasticities=ETA)
    with pytest.raises(ValueError, match="unknown preference rule"):
        model("aids")
    with pytest.raises(ValueError, match="CES elasticity"):
        model("ces", ces_elasticity=-1.)
    with pytest.raises(ValueError, match="sector_codes"):
        model("cobb_douglas", sector_codes=("a", "b"))
    prefs = model("stone_geary", expenditure_elasticities=ETA)
    b = np.sum(P * prefs.gamma, axis=0)
    with pytest.raises(HouseholdDomainError, match="exceed subsistence"):
        prefs.demand(P, b, validate=True)
    with pytest.raises(HouseholdDomainError, match="exceed subsistence"):
        prefs.demand(P, b - 1., validate=True)
    with pytest.raises(HouseholdDomainError, match="strictly positive"):
        prefs.demand(-P, M, validate=True)
    with pytest.raises(HouseholdDomainError, match="finite"):
        prefs.demand(P, [np.nan, 100.], validate=True)
    with pytest.raises(HouseholdDomainError, match="must be real"):
        prefs.demand(P.astype(complex), M, validate=True)
    with pytest.raises(HouseholdDomainError, match="strictly positive"):
        prefs.expenditure(P, 0., validate=True)
    with pytest.raises(ValueError, match="shape"):
        prefs.demand(P[:2], M)
    with pytest.raises(ValueError, match="shape"):
        prefs.demand(P, [1., 2., 3.])
    assert issubclass(HouseholdDomainError, ValueError)


def test_calibration_admissibility_boundary_and_unidentified_unit_targets():
    initial = model("stone_geary", expenditure_elasticities=ETA)
    cap = 1. / initial.eta.max(axis=0)
    boundary = model("stone_geary", expenditure_elasticities=ETA, supernumerary_share=cap)
    assert np.all(boundary.gamma >= 0)
    assert_allclose(boundary.demand(np.ones_like(X0), boundary.m0), X0, atol=2e-14)
    unit_eta = model("stone_geary", supernumerary_share=.5)
    assert_allclose(unit_eta.eta[X0 > 0], 1.)
    assert_allclose(unit_eta.gamma, X0 * .5)
    # Unit benchmark Engel targets do not force global homotheticity at changed prices.
    assert not np.allclose(unit_eta.demand(P, M), model("cobb_douglas").demand(P, M))


def test_supernumerary_share_provenance_flag_tracks_the_share_argument():
    key = "supernumerary_share_default_is_assumption"
    assert model("stone_geary").metadata[key] is True
    assert_allclose(model("stone_geary").supernumerary_share, .5)
    assert model("stone_geary", expenditure_elasticities=ETA).metadata[key] is True
    supplied = model("stone_geary", supernumerary_share=[.4, .5])
    assert supplied.metadata[key] is False
    assert_allclose(supplied.supernumerary_share, [.4, .5])
    both = model("stone_geary", expenditure_elasticities=ETA, supernumerary_share=.5)
    assert both.metadata[key] is False
    for rule in ("cobb_douglas", "fixed_baskets", "ces"):
        prefs = model(rule)
        assert prefs.metadata[key] is False and np.all(prefs.supernumerary_share == 1.)


def test_ces_limits_match_fixed_baskets_and_cobb_douglas():
    assert_allclose(model("ces", ces_elasticity=0.).demand(P, M), model("fixed_baskets").demand(P, M))
    assert_allclose(model("ces", ces_elasticity=1.).demand(P, M), model("cobb_douglas").demand(P, M))
    assert_allclose(model("ces", ces_elasticity=1. + 1e-9).demand(P, M), model("cobb_douglas").demand(P, M), rtol=1e-9)


def test_fit_surplus_from_own_compensated_elasticities_recovers_truth():
    prefs = model("stone_geary", expenditure_elasticities=ETA, supernumerary_share=[.35, .55])
    own = -prefs.supernumerary_share * prefs.eta * (1. - prefs.beta)
    fit = fit_supernumerary_share(X0, ETA, own)
    assert isinstance(fit, SupernumeraryFitResult)
    assert_allclose(fit.fitted, [.35, .55], atol=2e-16)
    assert_allclose(fit.weighted_rmse, 0., atol=1e-16)
    assert not fit.clipped.any()
    # Verify the elasticity formula by directly differentiating compensated demand.
    for i in range(3):
        p = np.ones_like(X0, dtype=complex)
        p[i] += 1e-25j
        observed = prefs.hicksian(p, 1.)[i].imag / (1e-25 * X0[i])
        assert_allclose(observed, own[i], atol=3e-16)


def test_fit_surplus_reports_projection_and_binding_bounds():
    prefs = model("stone_geary", expenditure_elasticities=ETA)
    slope = prefs.eta * (1. - prefs.beta)
    target = -slope * 2.0
    target[0, 0] *= .5  # Targets are not exactly representable by one LES scalar.
    fit = fit_supernumerary_share(X0, ETA, target)
    assert_allclose(fit.fitted, 1. / prefs.eta.max(axis=0))
    assert fit.clipped.all() and fit.upper_bound_hit.all()
    assert np.all(fit.weighted_rmse > 0)
    assert fit.weighted_rmse_unconstrained[0] > 0
    fit_zero = fit_supernumerary_share(X0, ETA, np.zeros_like(X0))
    assert np.all(fit_zero.fitted > 0)
    assert fit_zero.lower_bound_hit.all()
    assert_allclose(fit_zero.unconstrained, 0.)
    with pytest.raises(ValueError, match="do not identify"):
        fit_supernumerary_share([[100.]], [[1.]], [[0.]])
    with pytest.raises(ValueError, match="nonpositive"):
        fit_supernumerary_share(X0, ETA, np.ones_like(X0))
    with pytest.raises(ValueError, match="minimum_relative_share"):
        fit_supernumerary_share(X0, ETA, target, minimum_relative_share=1.)
    with pytest.raises(ValueError, match="positive fit weight"):
        fit_supernumerary_share(X0, ETA, target, weights=np.zeros_like(X0))


def test_untargeted_positive_expenditure_category_has_zero_fit_influence():
    expenditures = np.array([[20., 10.], [30., 35.], [50., 55.]])
    eta = np.ones_like(expenditures)
    weights = expenditures.copy()
    weights[-1] = 0.  # A positive consumption category has no empirical price target.
    truth = np.array([.35, .6])
    beta = expenditures / expenditures.sum(0)
    targets = -truth * (1 - beta)
    targets[-1] = 0.  # Placeholder only.
    fit = fit_supernumerary_share(expenditures, eta, targets, weights=weights)
    assert_allclose(fit.fitted, truth, atol=2e-16)
    targets[-1] = -9.  # Arbitrary placeholder cannot influence identification.
    changed = fit_supernumerary_share(expenditures, eta, targets, weights=weights)
    assert_allclose(changed.fitted, fit.fitted, atol=2e-16)
    assert_allclose(fit.max_abs_residual, 0., atol=2e-16)
    assert_allclose(changed.max_abs_residual, 0., atol=2e-16)
    assert_allclose(changed.weighted_rmse, 0., atol=2e-16)
    assert_allclose(fit.normalized_weights[-1], 0.)
    assert_allclose(fit.normalized_weights.sum(0), 1.)


# ===========================================================================
# 2. Additional invariants
# ===========================================================================

@pytest.mark.parametrize("rule,sigma", [("fixed_baskets", 1.), ("cobb_douglas", 1.), ("stone_geary", 1.),
                                        ("ces", 0.), ("ces", .4), ("ces", 2.)])
def test_analytic_slutsky_matches_finite_differences_in_every_region(rule, sigma):
    prefs = model(rule, ces_elasticity=sigma,
                  **({"expenditure_elasticities": ETA, "supernumerary_share": [.4, .55]}
                     if rule == "stone_geary" else {}))
    analytic = prefs.slutsky(P, M, validate=True)
    assert analytic.shape == (4, 4, 2)
    state = prefs.evaluate(P, M)
    step = 1e-5
    numeric = np.empty_like(analytic)
    for j in range(4):
        plus, minus = P.copy(), P.copy()
        plus[j] += step
        minus[j] -= step
        numeric[:, j, :] = (prefs.hicksian(plus, state.utility) - prefs.hicksian(minus, state.utility)) / (2 * step)
    assert_allclose(analytic, numeric, atol=5e-8)
    assert_allclose(analytic, analytic.transpose(1, 0, 2), atol=1e-14)
    assert_allclose(np.einsum("ijk,jk->ik", analytic, P), 0., atol=1e-12)
    for k in range(2):
        assert np.linalg.eigvalsh(analytic[:, :, k]).max() < 1e-12
    assert np.all(analytic[-1] == 0.) and np.all(analytic[:, -1] == 0.)


def _direct_utility(prefs, q, region):
    """Primal utility of a quantity vector for one region, benchmark-normalized."""
    active = prefs.x0[:, region] > 0
    if prefs.rule == "ces" and prefs.ces_elasticity != 1.:
        s = prefs.ces_elasticity
        w = prefs.shares[active, region]
        return np.sum(w ** (1 / s) * q[active] ** ((s - 1) / s)) ** (s / (s - 1)) / prefs.m0[region]
    b, g = prefs.beta[active, region], prefs.gamma[active, region]
    return np.prod(((q[active] - g) / b) ** b) / prefs.surplus0[region]


@pytest.mark.parametrize("rule,sigma", [("stone_geary", 1.), ("cobb_douglas", 1.), ("ces", .7), ("ces", 1.8)])
def test_primal_expenditure_minimization_oracle(rule, sigma):
    prefs = model(rule, ces_elasticity=sigma,
                  **({"expenditure_elasticities": ETA, "supernumerary_share": [.4, .55]}
                     if rule == "stone_geary" else {}))
    target = np.array([.9, 1.15])
    for k in range(2):
        active = prefs.x0[:, k] > 0
        p = P[active, k]
        gamma = prefs.gamma[active, k]
        n = int(active.sum())
        start = prefs.x0[active, k] * (1.3 if k == 0 else .8)
        res = minimize(lambda q: float(p @ q), start, method="SLSQP",
                       constraints=[{"type": "ineq", "fun": lambda q, k=k, active=active:
                                     _direct_utility(prefs, _embed(q, active), k) - target[k]}],
                       bounds=[(gamma[i] + 1e-9, None) for i in range(n)],
                       options={"ftol": 1e-14, "maxiter": 500})
        # Status 8 is SLSQP stalling at ftol=1e-14 (seen with OpenBLAS); the value
        # checks below decide whether the oracle actually reached the minimum.
        assert res.success or res.status == 8, res.message
        assert res.fun == pytest.approx(prefs.expenditure(P, target)[k], rel=1e-6)
        assert_allclose(res.x, prefs.hicksian(P, target)[active, k], rtol=2e-4)


def _embed(q, active):
    out = np.zeros(active.shape)
    out[active] = q
    return out


def test_general_base_welfare_is_consistent_with_benchmark_forms():
    prefs = model("stone_geary", expenditure_elasticities=ETA, supernumerary_share=[.4, .55])
    against_benchmark = compute_household_welfare(prefs, P, M, base_prices=np.ones_like(X0), base_budget=prefs.m0)
    direct = prefs.welfare(P, M)
    for name in ("ev", "cv", "ev_pct_consumption", "cv_pct_consumption", "utility", "cost_index"):
        assert_allclose(getattr(against_benchmark, name), getattr(direct, name), atol=3e-14)
    identity = compute_household_welfare(prefs, P, M, base_prices=P, base_budget=M)
    assert np.all(identity.ev == 0.) and np.all(identity.cv == 0.)
    p2, m2 = P * np.array([[1.1, .9], [1., 1.2], [.95, 1.], [1., 1.]]), M * 1.05
    forward = compute_household_welfare(prefs, p2, m2, base_prices=P, base_budget=M)
    reverse = compute_household_welfare(prefs, P, M, base_prices=p2, base_budget=m2)
    assert_allclose(forward.ev, -reverse.cv, atol=3e-14)
    assert_allclose(forward.cv, -reverse.ev, atol=3e-14)
    assert_allclose(forward.ev, prefs.expenditure(P, forward.utility) - M, atol=3e-14)
    assert_allclose(forward.cv, m2 - prefs.expenditure(p2, forward.base_utility), atol=3e-14)
    assert_allclose(forward.ev_pct_consumption, 100 * forward.ev / M)
    with pytest.raises(ValueError, match="supplied together"):
        compute_household_welfare(prefs, P, M, base_prices=P)
    with pytest.raises(TypeError):
        compute_household_welfare(object(), P, M)
    with pytest.raises(HouseholdDomainError):
        compute_household_welfare(prefs, P, np.sum(P * prefs.gamma, axis=0), validate=True)


@pytest.mark.parametrize("scale", [1e-3, 1e3])
def test_currency_scaling_preserves_percentages_utility_and_indices(scale):
    kw = {"expenditure_elasticities": ETA, "supernumerary_share": [.4, .55]}
    reference = model("stone_geary", **kw)
    scaled = calibrate_household(X0 * scale, "stone_geary", **kw)
    a, b = reference.welfare(P, M), scaled.welfare(P, M * scale)
    assert_allclose(b.ev, a.ev * scale, rtol=1e-12)
    assert_allclose(b.cv, a.cv * scale, rtol=1e-12)
    assert_allclose(b.ev_pct_consumption, a.ev_pct_consumption, rtol=1e-12)
    assert_allclose(b.utility, a.utility, rtol=1e-12)
    assert_allclose(b.cost_index, a.cost_index, rtol=1e-12)
    assert_allclose(scaled.demand(P, M * scale), reference.demand(P, M) * scale, rtol=1e-12)
    assert_allclose(scaled.slutsky(P, M * scale), reference.slutsky(P, M) * scale, rtol=1e-12)


def test_result_objects_labels_and_exporters():
    prefs = model("stone_geary", expenditure_elasticities=ETA, supernumerary_share=[.4, .55],
                  sector_codes=["AGR", "MAN", "SRV", "NONE"], country_codes=["USA", "ROW"])
    assert prefs.sector_codes == ("AGR", "MAN", "SRV", "NONE") and prefs.country_codes == ("USA", "ROW")
    assert (prefs.n_sectors, prefs.n_countries) == (4, 2)
    frame = prefs.to_dataframe()
    assert list(frame.index.names) == ["sector", "country"] and frame.shape == (8, 6)
    assert frame.loc[("NONE", "USA"), "active"] == False  # noqa: E712
    assert "stone_geary" in prefs.summary() and "USA" in prefs.calibration.to_dataframe().index
    assert isinstance(prefs.calibration, HouseholdCalibrationResult)
    audit = prefs.calibration.as_dict()
    assert audit["identification"].startswith("Stone-Geary surplus shares are supplied assumptions")
    state = prefs.evaluate(P, M)
    assert isinstance(state, HouseholdDemandResult)
    assert list(state.to_dataframe().columns) == ["budget", "price_index", "cost_index", "surplus", "utility"]
    assert state.quantities_frame().shape == (4, 2) and "ROW" in state.quantities_frame().columns
    welfare = prefs.welfare(P, M)
    assert isinstance(welfare, HouseholdWelfareResult) and welfare.country_codes == ("USA", "ROW")
    fit = fit_supernumerary_share(X0, ETA, -prefs.supernumerary_share * prefs.eta * (1. - prefs.beta),
                                  country_codes=["USA", "ROW"])
    assert list(fit.to_dataframe().index) == ["USA", "ROW"]
    for obj in (prefs, prefs.calibration, state, welfare, fit):
        assert "|" in obj.to_markdown()
        assert "tabular" in obj.to_latex()
        assert "table" in obj.to_typst()
        assert isinstance(obj.summary(), str) and obj.summary()
    assert "positive values denote gains" in welfare.summary()


def test_array_methods_return_writable_copies_and_unlabelled_results_export():
    prefs = model("stone_geary", expenditure_elasticities=ETA)
    assert isinstance(prefs, HouseholdPreferences)
    state = prefs.evaluate(P, M)
    for out in (prefs.demand(P, M), prefs.hicksian(P, .9), prefs.expenditure(P, 1.), prefs.slutsky(P, M)):
        assert out.flags.writeable
    q = prefs.demand(P, M)
    q *= 2.  # in-place work on returned demand never touches the frozen state
    assert_allclose(prefs.demand(P, M), state.quantities)
    assert not state.quantities.flags.writeable
    assert np.iscomplexobj(prefs.demand(P + 1e-25j, M))
    # Results constructed directly (no codes) fall back to generated labels.
    bare = HouseholdDemandResult(np.ones((2, 2)), *[np.ones(2)] * 5)
    assert list(bare.to_dataframe().index) == ["C00", "C01"]
    assert list(bare.quantities_frame().index) == ["S00", "S01"]
    assert "|" in bare.to_markdown()


# ===========================================================================
# 3. Bridges to puremacro trade calibrations and equilibria
# ===========================================================================

def test_household_expenditure_from_calibration_matches_flexible_benchmark(calib2):
    from puremacro.trade.flexible import _extract_benchmark_household_data
    x0 = household_expenditure_from_calibration(calib2, 0)
    budget0, shares0, c_s0 = _extract_benchmark_household_data(calib2)
    # Same sector shares; the flexible budget is net of the uniform final-use tax.
    assert_allclose(x0 / x0.sum(0), shares0[0], rtol=1e-13)
    gross_up = 1. / (1. - calib2.tax_fd[0, 0])
    assert_allclose(x0.sum(0), budget0.ravel() * gross_up, rtol=1e-13)
    assert_allclose(x0, c_s0[0] * gross_up[None, :], rtol=1e-13)
    assert_allclose(x0.sum(0), calib2.theta[0, 0] * (calib2.l_endow + calib2.k_endow + calib2.T.reshape(1, -1)).ravel())
    other = household_expenditure_from_calibration(calib2, 2)
    assert other.shape == (2, 2) and np.all(other >= 0)
    with pytest.raises(ValueError, match="Investment"):
        household_expenditure_from_calibration(calib2, 1)
    with pytest.raises(ValueError, match="out of range"):
        household_expenditure_from_calibration(calib2, 3)
    with pytest.raises(TypeError):
        household_expenditure_from_calibration(calib2, 0.)
    with pytest.raises(TypeError):
        household_expenditure_from_calibration("calib", 0)
    negative = replace(calib2, theta=calib2.theta * np.array([[[-1.], [1.], [1.]]]))
    with pytest.raises(ValueError, match="negative"):
        household_expenditure_from_calibration(negative, 0)


def test_synthetic_table_category_zero_is_admissible():
    nc, ns = 6, 5
    countries, sectors = synthetic_codes(nc, ns)
    calib = calibrate_trade_model(synthetic_icio_table(nc, ns, seed=4), ns=ns, nc=nc, nfd=3, validate=True,
                                  country_codes=countries, sector_codes=sectors)
    x0 = household_expenditure_from_calibration(calib, 0)
    assert x0.shape == (ns, nc) and np.all(x0 > 0)
    prefs = calibrate_household(x0, "stone_geary", sector_codes=calib.sector_codes, country_codes=calib.country_codes)
    assert prefs.country_codes[0] == calib.country_codes[0] and prefs.calibration.structural_zero_count == 0
    assert_allclose(prefs.demand(np.ones_like(x0), prefs.m0), x0, rtol=1e-12)
    assert_allclose(prefs.beta.sum(0), 1.)
    with pytest.raises(ValueError, match="Investment"):
        household_expenditure_from_calibration(calib, 1)


def _flexible_demand(calib, prices, budget, mu):
    """Flexible-module Stone-Geary demand in the household's purchaser units."""
    from puremacro.trade.flexible import FlexiblePreferenceConfig, compute_stone_geary_final_demand
    gross_up = 1. / (1. - calib.tax_fd[0, 0])
    net = (np.asarray(budget, dtype=float) / gross_up).reshape(1, 1, -1)
    q = compute_stone_geary_final_demand(net, prices[None], calib, FlexiblePreferenceConfig(mu_s=mu))[0]
    return q * gross_up[None, :]


def test_fixed_baskets_equal_flexible_basket_without_subsistence(calib2):
    x0 = household_expenditure_from_calibration(calib2, 0)
    fixed = calibrate_household(x0, "fixed_baskets")
    rng = np.random.default_rng(3)
    for _ in range(4):
        prices = rng.uniform(.8, 1.3, x0.shape)
        budget = fixed.m0 * rng.uniform(.7, 1.3, 2)
        assert_allclose(fixed.demand(prices, budget), _flexible_demand(calib2, prices, budget, 0.),
                        rtol=1e-12, atol=1e-12)


def test_exact_les_and_flexible_les_agree_only_at_the_benchmark(calib2):
    # The flexible module's supernumerary spending buys the benchmark basket in fixed proportions;
    # the exact LES here spends it by marginal budget shares. Both reproduce the benchmark.
    x0 = household_expenditure_from_calibration(calib2, 0)
    prefs = calibrate_household(x0, "stone_geary", supernumerary_share=.75)  # subsistence share 0.25
    ones = np.ones_like(x0)
    assert_allclose(prefs.demand(ones, prefs.m0), x0, rtol=1e-12)
    assert_allclose(_flexible_demand(calib2, ones, prefs.m0, .25), x0, rtol=1e-12)
    prices = np.random.default_rng(3).uniform(.8, 1.3, x0.shape)
    assert np.max(np.abs(prefs.demand(prices, prefs.m0) - _flexible_demand(calib2, prices, prefs.m0, .25))) > 1e-3


@pytest.mark.parametrize("category", [0, 2])
def test_consistent_bridge_reproduces_hicksian_welfare_with_fixed_baskets(calib2, consistent_pair, category):
    base, cf, tau = consistent_pair
    x0 = household_expenditure_from_calibration(calib2, category)
    p0, m0 = household_prices_from_result(base, calib2, category=category)
    p1, m1 = household_prices_from_result(cf, calib2, category=category)
    assert_allclose(p0, 1., atol=1e-9)
    assert_allclose(m0, x0.sum(0), rtol=1e-9)
    assert np.all(p1 > 0) and not np.allclose(p1, 1.)
    # Budgets and category prices agree with the audited accounting block.
    from puremacro.trade._accounting import evaluate
    block = evaluate(cf.x_sol, calib2, cf.metadata["intermediate_tariff_multipliers"],
                     cf.metadata["final_tariff_multipliers"], None, None)
    assert_allclose(m1, block["expenditure"][0, category], rtol=1e-12)
    weights = x0 / x0.sum(0)
    assert_allclose(np.sum(weights * p1, axis=0), block["Q"][0, category], rtol=1e-12)
    prefs = calibrate_household(x0, "fixed_baskets", sector_codes=calib2.sector_codes, country_codes=calib2.country_codes)
    welfare = compute_household_welfare_from_results(prefs, calib2, cf, base_result=base, category=category)
    for c in range(2):
        hicks = compute_hicksian_welfare(calib2, cf, base_result=base, target_country=c,
                                         consumption_categories=(category,))
        assert welfare.ev[c] == pytest.approx(hicks.ev, rel=1e-10, abs=1e-12)
        assert welfare.cv[c] == pytest.approx(hicks.cv, rel=1e-10, abs=1e-12)
        assert welfare.ev_pct_consumption[c] == pytest.approx(hicks.ev_pct_consumption, rel=1e-10)
    assert welfare.metadata["accounting"] == "consistent" and welfare.metadata["category"] == category
    assert welfare.metadata["tariff_schedule_verification"]["eq_result"].startswith("consistent")
    identity = compute_household_welfare_from_results(prefs, calib2, base, base_result=base, category=category)
    assert np.all(identity.ev == 0.) and np.all(identity.cv == 0.)
    # Non-homothetic and CES households differ from the Leontief basket but keep the sign structure.
    for rule, kw in (("stone_geary", {"supernumerary_share": .6}), ("cobb_douglas", {}), ("ces", {"ces_elasticity": .5})):
        alt = calibrate_household(x0, rule, **kw)
        w = compute_household_welfare_from_results(alt, calib2, cf, base_result=base, category=category)
        assert np.isfinite(w.ev).all() and np.sign(w.ev).tolist() == np.sign(welfare.ev).tolist()
        assert_allclose(w.ev, alt.expenditure(p0, w.utility) - alt.expenditure(p0, w.base_utility), atol=1e-12)


def test_bridges_on_non_square_table_with_independent_flow_certificates(calib32):
    # 3 countries x 2 sectors: a transposed (ns, nc) reshape cannot hide behind square shapes.
    from puremacro.trade._accounting import evaluate
    from puremacro.trade.solver import _resolve_tariffs
    calib = calib32
    nc, ns, nfd = 3, 2, calib.n_final_demand
    n = nc * ns
    rates = np.array([.2, 0., .07])
    x0 = household_expenditure_from_calibration(calib, 0)
    assert x0.shape == (ns, nc)
    fb = calibrate_household(x0, "fixed_baskets", sector_codes=calib.sector_codes, country_codes=calib.country_codes)
    base = solve_trade_equilibrium(calib, accounting="consistent", tol=1e-10)
    cf = solve_trade_equilibrium(calib, tau=rates, tau_fd=rates, accounting="consistent", tol=1e-10)
    assert base.converged and cf.converged
    p1, m1 = household_prices_from_result(cf, calib)
    # Certificate: sector purchaser spending read from the model's final-demand flows equals P_s * q_s.
    block = evaluate(cf.x_sol, calib, cf.metadata["intermediate_tariff_multipliers"],
                     cf.metadata["final_tariff_multipliers"], None, None)
    flows = block["F"].reshape(n, nfd, nc, order="F")
    spending = ((block["p"].ravel(order="F")[:, None, None] * block["tf"] * flows)[:, 0, :]
                .reshape(nc, ns, nc).sum(0) / (1. - block["fd_tax"][0, 0][None, :]))
    assert_allclose(spending, p1 * fb.demand(p1, m1), rtol=1e-12)
    welfare = compute_household_welfare_from_results(fb, calib, cf, base_result=base)
    assert welfare.country_codes == ("AAA", "BBB", "CCC")
    for c in range(nc):
        hicks = compute_hicksian_welfare(calib, cf, base_result=base, target_country=c)
        assert welfare.ev[c] == pytest.approx(hicks.ev, rel=1e-10, abs=1e-12)
        assert welfare.cv[c] == pytest.approx(hicks.cv, rel=1e-10, abs=1e-12)
    # The supplied national rates resolve to the recorded schedule and change nothing.
    _, tf, _, _ = _resolve_tariffs(calib, None, rates)
    assert_allclose(tf, cf.metadata["final_tariff_multipliers"], rtol=0, atol=1e-14)
    assert np.array_equal(household_prices_from_result(cf, calib, tau_fd=rates)[0], p1)


def test_bridge_labels_and_error_messages_name_the_caller(calib32):
    calib = calib32
    x0 = household_expenditure_from_calibration(calib, 0)
    base = solve_trade_equilibrium(calib, accounting="consistent", tol=1e-10)
    rates = np.array([.2, 0., .07])
    cf = solve_trade_equilibrium(calib, tau=rates, tau_fd=rates, accounting="consistent", tol=1e-10)
    # Preferences with generated labels take the calibration's; mislabeled preferences raise.
    plain = calibrate_household(x0, "cobb_douglas")
    assert plain.country_codes == ("C00", "C01", "C02")
    assert compute_household_welfare_from_results(plain, calib, cf, base_result=base).country_codes == ("AAA", "BBB", "CCC")
    reordered = calibrate_household(x0, "cobb_douglas", country_codes=["CCC", "BBB", "AAA"])
    with pytest.raises(ValueError, match="country_codes"):
        compute_household_welfare_from_results(reordered, calib, cf, base_result=base)
    wrong_sectors = calibrate_household(x0, "cobb_douglas", sector_codes=["X", "Y"])
    with pytest.raises(ValueError, match="sector_codes"):
        compute_household_welfare_from_results(wrong_sectors, calib, cf, base_result=base)
    # Errors from the shared consistent audit name the household function and the failing state.
    stale = replace(cf, converged=False)
    with pytest.raises(ValueError, match="household_prices_from_result: .*converged.*shared with compute_hicksian"):
        household_prices_from_result(stale, calib)
    with pytest.raises(ValueError, match=r"compute_household_welfare_from_results \(base_result\): .*converged"):
        compute_household_welfare_from_results(plain, calib, cf, base_result=replace(base, converged=False))
    with pytest.raises(ValueError, match=r"\(eq_result\): .*converged"):
        compute_household_welfare_from_results(plain, calib, stale, base_result=base)


def test_bridge_rejections(calib2, consistent_pair):
    base, cf, tau = consistent_pair
    x0 = household_expenditure_from_calibration(calib2, 0)
    prefs = calibrate_household(x0, "cobb_douglas")
    with pytest.raises(ValueError, match="converged"):
        household_prices_from_result(replace(cf, converged=False), calib2)
    with pytest.raises(ValueError, match="Investment"):
        household_prices_from_result(cf, calib2, category=1)
    with pytest.raises(ValueError, match="disagrees with the tariff schedule"):
        household_prices_from_result(cf, calib2, tau_fd=np.array([.3, 0.]))
    with pytest.raises(ValueError, match="disagrees with the tariff schedule"):
        household_prices_from_result(base, calib2, tau_fd=tau)
    with pytest.raises(ValueError, match="shape"):
        household_prices_from_result(cf, calib2, tau_fd=np.ones((3, 3)))
    # The solve's own national rates and the recorded multiplier array are both accepted and change nothing.
    p_ref, m_ref = household_prices_from_result(cf, calib2)
    for schedule in (tau, cf.metadata["final_tariff_multipliers"]):
        p_chk, m_chk = household_prices_from_result(cf, calib2, tau_fd=schedule)
        assert np.array_equal(p_chk, p_ref) and np.array_equal(m_chk, m_ref)
    w_ref = compute_household_welfare_from_results(prefs, calib2, cf, base_result=base)
    w_chk = compute_household_welfare_from_results(prefs, calib2, cf, base_result=base, tau_fd=tau,
                                                   base_tau_fd=np.zeros(2))
    assert np.array_equal(w_chk.ev, w_ref.ev) and np.array_equal(w_chk.cv, w_ref.cv)
    with pytest.raises(ValueError, match="tol"):
        household_prices_from_result(cf, calib2, tol=0.)
    with pytest.raises(TypeError):
        household_prices_from_result("result", calib2)
    meta = dict(cf.metadata)
    meta.pop("final_tariff_multipliers")
    with pytest.raises(ValueError):
        household_prices_from_result(replace(cf, metadata=meta), calib2)
    meta = dict(cf.metadata)
    meta["fiscal_closure"] = "deficit_reduction"
    with pytest.raises(NotImplementedError):
        household_prices_from_result(replace(cf, metadata=meta), calib2)
    # Only consistent-accounting states are accepted; the message says what to pass instead.
    for mode in ("other", None):
        meta = dict(cf.metadata)
        meta["accounting"] = mode
        with pytest.raises(ValueError, match="consistent-accounting result"):
            household_prices_from_result(replace(cf, metadata=meta), calib2)
        with pytest.raises(ValueError, match=r"\(base_result\): household prices require a consistent"):
            compute_household_welfare_from_results(prefs, calib2, cf, base_result=replace(base, metadata=meta))
    with pytest.raises(TypeError, match="trust_tau_fd_detail"):
        household_prices_from_result(cf, calib2, trust_tau_fd_detail=True)
    wrong = calibrate_household(x0 * 1.01, "cobb_douglas")
    with pytest.raises(ValueError, match="not calibrated on this calibration"):
        compute_household_welfare_from_results(wrong, calib2, cf, base_result=base)
    with pytest.raises(TypeError):
        compute_household_welfare_from_results(object(), calib2, cf, base_result=base)
