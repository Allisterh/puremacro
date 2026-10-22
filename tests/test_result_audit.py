"""Numerical result audits: tampering controls and independent economic checks."""
from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest

from puremacro import dp
from puremacro.validation import audit_result, DerivativeCheck, RefinementCheck, ResidualCheck


def make_two_asset(na=30, nb=20, amax=40., bmax=10., reverse=False, deterministic=False):
    m = dp.Model('audit household')
    a = amax * np.linspace(0, 1, na) ** 1.2
    b = bmax * np.linspace(0, 1, nb) ** 1.2
    for name, grid in ([('b', b), ('a', a)] if reverse else [('a', a), ('b', b)]):
        m.state(name, grid)
    if not deterministic:
        m.exogenous('z', dp.Markov([.5, 1.5], [[.95, .05], [.1, .9]]))
    m.parameters(beta=.98, gamma=1., r_b=.005, r_a=.0125, w=1., chi_0=1., chi_1=1., a_bar=.25)
    m.local('d=a(+1)-(1+r_a)*a')
    m.local('chi=chi_0/(1+chi_1)*abs(d)*(abs(d)/(a+a_bar))^chi_1')
    income = 'w' if deterministic else 'w*z'
    m.local(f'c=(1+r_b)*b+(1+r_a)*a+{income}-a(+1)-b(+1)-chi')
    m.reward('crra(c,gamma)').subject_to('c>0')
    m.aggregate(C='c', A='a', B='b')
    return m


@pytest.fixture(scope='module')
def household():
    return make_two_asset().solve(method='egm', tol=1e-9)


@pytest.fixture(scope='module')
def one_asset():
    m = dp.Model('one-asset audit')
    m.state('wealth', 40 * np.linspace(0, 1, 80) ** 2)
    m.exogenous('z', dp.Markov([.5, 1.5], [[.95, .05], [.1, .9]]))
    m.parameters(beta=.96, r=.01, gamma=2., w=1.)
    m.local('c=(1+r)*wealth+w*z-wealth(+1)')
    m.reward('crra(c,gamma)').subject_to('c>0')
    return m.solve(method='egm', tol=1e-10)


def rows(report):
    return {row.check: row for row in report.records}


def test_two_asset_real_maps_and_measured_accuracy(household):
    report = audit_result(household)
    checks = rows(report)
    assert report.solver_converged is True
    assert checks['budget_residual'].value < 1e-12
    assert checks['stationarity_residual'].value < 1e-10
    # Measured on this fixed 30x20 calibration, independently of EGM tol=1e-9.
    assert 1e-4 < checks['euler_kkt_a'].value < .0021
    assert 1e-4 < checks['euler_kkt_b'].value < .0041
    assert checks['euler_kkt_a_weighted_p99'].value < .0018
    assert checks['euler_kkt_b_weighted_p99'].value < .0022
    assert checks['euler_kkt_a'].location == (1, 1, 0)
    assert checks['euler_kkt_b'].location == (0, 1, 0)
    assert checks['euler_kkt_a'].details['worst_coordinates']['a'] == household.grids['a'][1]
    assert report.maps['budget_residual'].shape == (30, 20, 2)
    assert not report.maps['budget_residual'].flags.writeable
    assert checks['refinement'].status == 'not_checked'
    assert checks['derivatives'].status == 'not_checked'


def test_one_asset_stationary_euler_and_tampered_consumption(one_asset):
    report = audit_result(one_asset)
    assert rows(report)['stationarity_residual'].value < 1e-10
    assert rows(report)['budget_residual'].value < 1e-12
    tampered = deepcopy(one_asset)
    tampered.raw.c[2, 0] *= 1.2
    bad = audit_result(tampered)
    assert bad.solver_converged is True
    assert rows(bad)['budget_residual'].status == 'failed'
    assert rows(bad)['euler_kkt_a'].status == 'failed'
    assert rows(bad)['budget_residual'].location == (2, 0)


def test_continuous_policy_expectation_and_kkt_against_scalar_oracle(household):
    from scipy.interpolate import RegularGridInterpolator
    s = household
    a, b, P, c = s.grids['a'], s.grids['b'], s.P, s.raw.c
    aa = a[:, None, None]
    deposit = s.raw.aprime - 1.0125 * aa
    cost = .5 * deposit ** 2 / (aa + .25)
    chi_d = deposit / (aa + .25)
    chi_a = -1.0125 * chi_d - cost / (aa + .25)
    va, vb = (1.0125 - chi_a) / c, 1.005 / c
    # Independent SciPy interpolation of each destination shock state.
    interpolators = [(RegularGridInterpolator((a, b), va[:, :, z]),
                      RegularGridInterpolator((a, b), vb[:, :, z])) for z in range(2)]
    report = audit_result(s)
    for ia, ib, iz in ((1, 1, 0), (5, 5, 0), (15, 10, 1), (29, 19, 1), (0, 0, 0)):
        ap, bp = s.raw.aprime[ia, ib, iz], s.raw.bprime[ia, ib, iz]
        e_a = sum(P[iz, z] * float(interpolators[z][0]([ap, bp])[0]) for z in range(2))
        e_b = sum(P[iz, z] * float(interpolators[z][1]([ap, bp])[0]) for z in range(2))
        uc = 1 / c[ia, ib, iz]
        gradients = [.98 * e_a / uc - (1 + chi_d[ia, ib, iz]), .98 * e_b / uc - 1]
        for name, g, choice, grid in zip(('a', 'b'), gradients, (ap, bp), (a, b)):
            expected = max(g, 0) if choice == grid[0] else min(g, 0) if choice == grid[-1] else g
            assert report.maps['euler_kkt_' + name][ia, ib, iz] == pytest.approx(expected, abs=2e-14)


def test_choice_and_envelope_derivatives_independent_finite_differences(household):
    from puremacro.models.hank_sequence_space import _two_asset_cost
    # Independent scalar budget, with a non-unit gross return and current/next timing.
    a, b, ap, bp, z = 3.2, 2.1, 3.4, 1.8, .5
    def utility(x):
        aa, bb, anext, bnext = x
        d = anext - 1.0125 * aa
        c = 1.005 * bb + 1.0125 * aa + z - anext - bnext - .5 * d*d/(aa+.25)
        return np.log(c)
    x = np.array([a, b, ap, bp])
    c = np.exp(utility(x))
    _, cd, ca, _ = _two_asset_cost(ap, a, .0125, 1., 1., .25)
    derivative = np.array([(1.0125-ca)/c, 1.005/c, -(1+cd)/c, -1/c])
    report = audit_result(household, checks=('derivatives',), derivatives=(
        DerivativeCheck('choice_and_state_budget_derivatives', utility, x, derivative, tolerance=1e-9),))
    assert rows(report)['choice_and_state_budget_derivatives'].status == 'passed'
    bad = derivative.copy()
    bad[0] -= .0125 / c  # wrong gross-return timing in the current asset envelope
    report = audit_result(household, checks=('derivatives',), derivatives=(
        DerivativeCheck('wrong_asset_timing', utility, x, bad, tolerance=1e-9),))
    assert rows(report)['wrong_asset_timing'].status == 'failed'


def test_independent_bellman_optimization_on_canonical_grid(household):
    from scipy.interpolate import RegularGridInterpolator
    from scipy.optimize import differential_evolution
    s = household
    # Independent objective using policy-evaluated V, not EGM or its marginals.
    # Linear V interpolation differs from interpolating envelope derivatives;
    # the objective gap quantifies that approximation rather than claiming equality.
    for ia, ib, iz in ((5, 5, 0), (15, 10, 1)):
        a, b, z = s.grids['a'][ia], s.grids['b'][ib], s.grids['z'][iz]
        continuation = RegularGridInterpolator((s.grids['a'], s.grids['b']), s.V @ s.P[iz])
        def loss(x):
            ap, bp = x
            d = ap - 1.0125*a
            c = 1.005*b + 1.0125*a + z - ap - bp - .5*d*d/(a+.25)
            return 1e8 + abs(c) if c <= 0 else -np.log(c)-.98*float(continuation(x)[0])
        egm = np.array([s.raw.aprime[ia, ib, iz], s.raw.bprime[ia, ib, iz]])
        optimum = differential_evolution(loss, [(0, 40), (0, 10)], seed=1, tol=1e-9,
                                         popsize=10, x0=egm)
        gap = loss(egm) - optimum.fun
        assert optimum.success
        assert -1e-8 < gap < .002
        assert np.linalg.norm(egm - optimum.x) > .01  # oracle really used a different policy


def test_kkt_lower_upper_signs_and_narrow_cells():
    from puremacro.validation._audit_dp import _kkt
    grid = np.array([0., 1e-8, 2e-8])
    choices = np.array([0., 0., 1e-8, 2e-8, 2e-8])
    np.testing.assert_array_equal(_kkt(np.array([-1., 1., -2., 1., -1.]), choices, grid),
                                  [0., 1., -2., 0., -1.])


def test_mass_preserving_vs_mass_breaking_and_missing_distribution(household):
    changed = deepcopy(household)
    changed.distribution = .99 * changed.distribution + .01 / changed.distribution.size
    report = rows(audit_result(changed))
    assert report['mass_error'].status == 'passed'
    assert report['stationarity_residual'].status == 'failed'
    changed.distribution *= .8
    report = rows(audit_result(changed))
    assert report['mass_error'].status == 'failed'
    assert report['stationarity_residual'].status == 'failed'
    assert report['euler_kkt_a_weighted_p95'].status == 'not_checked'
    changed.distribution = None
    report = rows(audit_result(changed))
    assert report['distribution'].status == 'not_checked'
    assert report['budget_residual'].status == 'passed'


def test_tampering_and_constraints_do_not_override_solver_record(household):
    changed = deepcopy(household)
    changed.raw.c[4, 3, 1] *= 1.1
    report = audit_result(changed)
    assert report.solver_converged is True
    assert rows(report)['budget_residual'].status == 'failed'
    changed = deepcopy(household)
    changed.raw.bprime[4, 3, 1] = 11.
    assert rows(audit_result(changed))['bounds_b'].status == 'failed'
    wrong = deepcopy(household._c.model)
    wrong._locals[-1] = ('c', wrong._locals[-1][1].replace('(1+r_a)*a', '(1+r_a)*a(+1)'))
    assert rows(audit_result(household, model=wrong))['model_specification'].status == 'failed'
    narrow = deepcopy(household._c.model)
    narrow.subject_to('abs(b(+1)-1.5192158245186125)>0.00001')
    assert rows(audit_result(household, model=narrow))['model_specification'].status == 'failed'


def test_reversed_unequal_axes_and_deterministic_productivity():
    for deterministic in (False, True):
        sol = make_two_asset(20, 12, reverse=True, deterministic=deterministic).solve(method='egm', tol=1e-9)
        report = audit_result(sol)
        assert report.metadata['dp_grid_order'] == ['a', 'b', 'productivity']
        assert report.maps['budget_residual'].shape == (20, 12, 1 if deterministic else 2)
        assert rows(report)['budget_residual'].status == 'passed'
        assert rows(report)['stationarity_residual'].status == 'passed'
        assert rows(report)['mass_at_a_max'].value == pytest.approx(sol.distribution[-1].sum())
        assert rows(report)['mass_at_b_max'].value == pytest.approx(sol.distribution[:, -1].sum())


def test_reports_are_deterministic_and_preserve_inputs(household):
    before = deepcopy(household)
    first, second = audit_result(household), audit_result(household)
    assert first.to_json() == second.to_json()
    assert first.to_markdown() == second.to_markdown()
    assert json.loads(first.to_json())['solver_converged'] is True
    assert set(first.to_dataframe().columns) == {'category','check','value','tolerance','units','status',
                                               'evidence_source','location','details'}
    assert first.to_dataframe().attrs['dp_grid_order'] == ['a', 'b', 'productivity']
    np.testing.assert_array_equal(before.raw.c, household.raw.c)
    np.testing.assert_array_equal(before.distribution, household.distribution)
    assert before.params == household.params
    assert first.to_json() == second.to_json(indent=2)


def test_no_hidden_solver_calls_or_unrequested_callbacks(household, monkeypatch):
    import puremacro.models.hank_sequence_space as hank
    def forbidden(*args, **kwargs):
        raise AssertionError('unexpected solve')
    monkeypatch.setattr(dp.Model, 'solve', forbidden)
    monkeypatch.setattr(hank, '_two_asset_step', forbidden)
    report = audit_result(household, derivatives=(DerivativeCheck('unused', forbidden, np.zeros(1), np.zeros(1)),),
                          refinements=(RefinementCheck('unused', forbidden),))
    assert rows(report)['budget_residual'].status == 'passed'
    assert rows(report)['refinement'].status == 'not_checked'


@pytest.mark.mechanism_control
def test_solver_blocker_control(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('unexpected solve')
    monkeypatch.setattr(dp.Model, 'solve', forbidden)
    with pytest.raises(AssertionError, match='unexpected solve'):
        dp.Model('control').solve()


def test_posterior_derivative_mismatch_and_nonlinear_equation_callbacks():
    point = np.array([.3, -.2])
    objective = lambda x: -.5 * x @ np.diag([2., 3.]) @ x
    correct = -np.array([2., 3.]) * point
    report = audit_result(object(), checks=('equations', 'derivatives'), derivatives=(
        DerivativeCheck('posterior_gradient', objective, point, correct + [.1, 0.]),),
        residuals=(ResidualCheck('nonlinear_occbin_equilibrium', lambda result, model: np.array([.2**2])),
                   ResidualCheck('portfolio_constraint', lambda result, model: np.maximum([.02], 0.)),))
    assert rows(report)['posterior_gradient'].status == 'failed'
    assert rows(report)['nonlinear_occbin_equilibrium'].status == 'failed'
    assert rows(report)['portfolio_constraint'].status == 'failed'
    assert report.solver_converged is None


def test_refinement_is_explicit_and_near_zero_uses_absolute_tolerance(household):
    calls = []
    def refined():
        calls.append(1)
        return {'C': 1.003, 'zero': 1e-10, 'J_C_w_0_0': .201}
    old = {'C': 1., 'zero': 0., 'J_C_w_0_0': .2}
    experiment = RefinementCheck('double_horizon', refined, metrics=dict, atol=1e-8,
                                 units={'C': 'goods', 'J_C_w_0_0': 'goods per wage'})
    report = audit_result(old, refinements=(experiment,))
    assert calls == []
    assert rows(report)['refinement'].status == 'not_checked'
    report = audit_result(old, checks=('refinement',), refinements=(experiment,))
    assert calls == [1]
    assert rows(report)['double_horizon:C'].status == 'passed'
    assert rows(report)['double_horizon:zero'].details['relative_change'] is None
    assert rows(report)['double_horizon:J_C_w_0_0'].units == 'goods per wage'
    bad = RefinementCheck('different_keys', lambda: {'C': 1.}, metrics=dict)
    assert rows(audit_result(old, checks=('refinement',), refinements=(bad,)))['different_keys'].status == 'failed'


def test_three_grid_experiments_report_moments_kkt_and_bounds(household):
    experiments = (
        RefinementCheck('refine_both', lambda: make_two_asset(45, 30).solve(method='egm', tol=1e-9)),
        RefinementCheck('expand_illiquid', lambda: make_two_asset(30, 20, amax=60.).solve(method='egm', tol=1e-9)),
        RefinementCheck('expand_liquid', lambda: make_two_asset(30, 20, bmax=15.).solve(method='egm', tol=1e-9)),
    )
    report = audit_result(household, checks=('refinement',), refinements=experiments)
    checks = rows(report)
    for label in ('refine_both', 'expand_illiquid', 'expand_liquid'):
        assert checks[label + ':consumption'].value is not None
        assert checks[label + ':mean_a'].value is not None
        assert checks[label + ':mean_b'].value is not None
        assert checks[label + ':euler_kkt_a'].value is not None
        assert checks[label + ':euler_kkt_b_weighted_p99'].value is not None
        assert checks[label + ':mass_at_a_max'].value is not None
        assert checks[label + ':mass_at_b_max'].value is not None
    # This coarse grid is not certified at 0.5% for all moments.
    assert any(row.status == 'failed' for row in report.records if row.category == 'refinement')


def test_first_order_dsge_recomputed_rules_derivatives_and_missing_context():
    from puremacro.dsge import build
    def equations(lead, now, eps, params):
        return np.array([lead.x - .9 * now.x - eps.e])
    model = build(equations, variables=['x'], states=['x'], shocks=['e'],
                  steady_state={'x': 0.}, linearize='level')
    report = audit_result(model, checks=('solver', 'equations', 'derivatives'))
    assert all(row.status == 'passed' for row in report.records if row.category in ('equations', 'derivatives'))
    bad = replace(model.solution, G=model.solution.G + .1)
    report = audit_result(bad, model=model)
    assert report.solver_converged is True
    assert rows(report)['decision_rule_residual'].status == 'failed'
    assert rows(audit_result(bad))['decision_rule_residual'].status == 'not_checked'
    mismatch = replace(model, A=2 * model.A)
    assert rows(audit_result(mismatch, checks=('derivatives',)))['jacobian_lead'].status == 'failed'


def test_dynamic_trade_certificate_preserved_and_recomputed():
    from puremacro.trade.dynamic import analytic_two_country_accounts, calibrate_dynamic, DynamicEconomy, solve_dynamic_steady_state
    economy = DynamicEconomy(calibrate_dynamic(analytic_two_country_accounts()), adjustment_cost=2, risk_aversion=2)
    result = solve_dynamic_steady_state(economy)
    report = audit_result(result)
    checks = rows(report)
    for name, value in result.certificate.items():
        assert checks['certificate:' + name].value == value
        assert checks['certificate:' + name].evidence_source.endswith('recorded_independent_certificate')
    assert checks['recomputed_certificate'].status == 'not_checked'
    assert report.metadata['original_metadata'] == result.metadata
    z = result.z.copy()
    z[-1] += .01
    bad = replace(result, z=z)
    report = audit_result(bad, model=economy)
    assert report.solver_converged is True
    assert any(row.status == 'failed' for row in report.records if row.check.startswith('recomputed_certificate:'))
    assert all(row.status == 'passed' for row in report.records if row.check.startswith('certificate:'))


def test_nonfinite_and_unsupported_are_visible(household):
    altered = deepcopy(household)
    altered.raw.c[2, 3, 0] = np.nan
    report = audit_result(altered)
    assert rows(report)['finite_values'].status == 'failed'
    assert rows(report)['budget_residual'].status == 'failed'
    assert 'NaN' in report.to_json()
    assert rows(audit_result(object()))['equations'].status == 'unsupported'
    with pytest.raises(ValueError):
        audit_result(object(), checks='solver')
    with pytest.raises(ValueError):
        audit_result(object(), tolerances={'euler': float('nan')})


@pytest.mark.parametrize('kind', ['residual', 'derivative', 'atol', 'rtol'])
@pytest.mark.parametrize('invalid', [np.nan, np.inf, -1.])
def test_invalid_experiment_tolerances_rejected(kind, invalid):
    if kind == 'residual':
        kwargs = {'residuals': (ResidualCheck('invalid', lambda r, m: 0, tolerance=invalid),)}
    elif kind == 'derivative':
        kwargs = {'derivatives': (DerivativeCheck('invalid', np.sum, np.zeros(1), np.ones(1), tolerance=invalid),)}
    else:
        kwargs = {'refinements': (RefinementCheck('invalid', lambda: None, **{kind: invalid}),)}
    with pytest.raises(ValueError, match='finite and nonnegative'):
        audit_result(object(), **kwargs)


def test_scalar_empty_and_weighted_quantile_records():
    from puremacro.validation.audit import _Collector
    out = _Collector(('equations',), None)
    out.residual('equations', 'empty', np.array([]), 1e-8, weights=np.array([]))
    out.residual('equations', 'scalar', np.array(.01), .02, weights=np.array(1.))
    out.residual('equations', 'weighted', np.array([9., 2., 1., 5.]), 10., weights=np.array([0., .04, .95, .01]))
    checks = {r.check: r for r in out.rows}
    assert checks['empty'].status == 'failed'
    assert checks['empty_weighted_p50'].status == 'not_checked'
    assert checks['scalar'].status == 'passed'
    assert checks['scalar'].location == ()
    assert checks['scalar_weighted_p99'].value == .01
    assert checks['weighted_weighted_p50'].value == 1.
    assert checks['weighted_weighted_p95'].value == 1.
    assert checks['weighted_weighted_p99'].value == 2.


def test_structural_budget_rejects_adversarial_function_before_solving(household):
    from puremacro.dp._expr import ModelSpecError
    model = deepcopy(household._c.model)
    # A smooth local bump can vanish at all original validation probes and at
    # a chosen solution while changing nearby optima. No finite sampling proof.
    model.functions(bump=lambda a, xp=np: xp.where((a > 6.123) & (a < 6.124), .1, 0.))
    model._locals[-1] = ('c', model._locals[-1][1] + '+bump(a)')
    with pytest.raises(ModelSpecError, match='structurally recognized'):
        model.solve(method='egm')


@pytest.mark.parametrize('cost_form', [
    'chi_0/(1+chi_1)*abs(d)*(abs(d)/(a+a_bar))^chi_1',
    'chi_0*abs(d)^(1+chi_1)/(1+chi_1)/(a+a_bar)^chi_1',
    'chi_0*abs(d)^(1+chi_1)/((1+chi_1)*(a+a_bar)^chi_1)',
    'chi_0/(1+chi_1)*(abs(d)/(a+a_bar))^(1+chi_1)*(a+a_bar)',
])
def test_structural_budget_forms_locals_and_reordering(cost_form):
    model = make_two_asset()
    model._locals[1] = ('chi', cost_form)
    model._locals.insert(2, ('cash', 'w*z+b+r_b*b+a+r_a*a'))
    model._locals[-1] = ('c', 'cash-(chi+b(+1)+a(+1))')
    solution = model.solve(method='egm', tol=1e-8)
    assert rows(audit_result(solution))['budget_residual'].status == 'passed'


def test_named_discount_curvature_assets_and_parameter_aliases():
    import re
    from puremacro.models.hank_sequence_space import dp_sequence_jacobian, dp_nonlinear_transition
    original = make_two_asset().solve(method='egm', tol=1e-9)
    model = make_two_asset()
    names = {'a': 'capital', 'b': 'bonds', 'r_a': 'r_a_ss', 'r_b': 'r_ss', 'w': 'w_ss',
             'gamma': 'risk_aversion', 'beta': 'patience'}
    rename = lambda text: re.sub(r'\b('+'|'.join(names)+r')\b', lambda m: names[m[0]], text)
    model._states = [(names[name], grid) for name, grid in model._states]
    model._params = {names.get(name, name): value for name, value in model._params.items()}
    model._locals = [(name, rename(expr)) for name, expr in model._locals]
    model.reward('crra(c,risk_aversion)').discount('patience')
    model.functions(twice=lambda x, xp=np: 2*x)
    model._aggregates = {'C': 'twice(c)', 'PRICE': 'w_ss*c', 'A': 'capital', 'B': 'bonds'}
    solution = model.solve(method='egm', tol=1e-9)
    np.testing.assert_allclose(solution.raw.c, original.raw.c, atol=1e-12)
    assert rows(audit_result(solution))['budget_residual'].status == 'passed'
    J = dp_sequence_jacobian(solution, inputs=['w_ss', 'r_ss', 'r_a_ss'], outputs=['C', 'PRICE'], T=4)
    shock = 1e-5 * .5**np.arange(4)
    up = dp_nonlinear_transition(solution, {'w_ss': shock}, outputs=['C', 'PRICE'])
    down = dp_nonlinear_transition(solution, {'w_ss': -shock}, outputs=['C', 'PRICE'])
    for output in ('C', 'PRICE'):
        np.testing.assert_allclose(J[output]['w_ss'] @ shock, (up[output]-down[output])/2, rtol=1e-4, atol=1e-9)


def test_budget_domain_cannot_be_proved_by_cancelling_singular_terms(household):
    from puremacro.dp._expr import ModelSpecError
    for extra in ('+a/(a-6.123)-a/(a-6.123)', '+(a-6.123)^chi_1-(a-6.123)^chi_1'):
        model = deepcopy(household._c.model)
        model._locals[-1] = ('c', model._locals[-1][1] + extra)
        with pytest.raises(ModelSpecError, match='domain|denominator'):
            model.solve(method='egm')
    model = deepcopy(household._c.model)
    model.functions(abs=lambda x, xp=np: xp.abs(x))
    with pytest.raises(ModelSpecError, match='registered arbitrary functions'):
        model.solve(method='egm')


def test_log_utility_does_not_use_unrelated_gamma_parameter(household):
    model = deepcopy(household._c.model)
    model.parameters(gamma=2.)
    model.reward('log(c)')
    result = model.solve(method='egm', tol=1e-9)
    assert result.raw.hh_block.gamma == 1.
    np.testing.assert_allclose(result.raw.c, household.raw.c, atol=1e-12)


def test_dynare_derivative_audit_holds_declared_steady_state_fixed():
    from puremacro.dsge import build_dynare
    def equations(lead, now, lag, shock, parameters, steady_state=None):
        ss = now if steady_state is None else steady_state
        return np.array([now.x - .8*lag.x - .2*ss.x - shock.e])
    # Current base tree build_dynare has the old derivative behavior; an explicit
    # correct first-order matrix makes this an audit regression independent of
    # track1's caller repair. The nonlinear equation still uses the context API.
    model = build_dynare(lambda lead, now, lag, shock, p: np.array([now.x-.8*lag.x-2.-shock.e]),
                         variables=['x'], shocks=['e'], steady_state={'x':10.})
    model = replace(model, _dynare_equations=equations)
    report = audit_result(model, checks=('derivatives',))
    assert rows(report)['jacobian_current'].status == 'passed'
    assert rows(report)['jacobian_lag'].status == 'passed'
    assert rows(report)['jacobian_shocks'].status == 'passed'


def test_periodic_reducible_distribution_remains_a_stationarity_failure():
    from puremacro.validation.audit import _Collector
    from puremacro.validation._audit_dp import _distribution
    transition = np.array([[0., 1., 1., 0.], [1., 0., 0., 0.],
                           [0., 0., 0., 0.], [0., 0., 0., 1.]])
    dist = np.array([[.25, .5], [0., .25]])
    original = dist.copy()
    out = _Collector(('distribution',), None)
    _distribution(out, dist, (2, 2), lambda mu: (transition @ mu.ravel()).reshape(2, 2),
                  [('wealth', np.array([0., 1.]))])
    checks = {r.check: r for r in out.rows}
    assert checks['mass_error'].status == 'passed'
    assert checks['minimum_probability'].status == 'passed'
    assert checks['stationarity_residual'].value == .25
    assert checks['stationarity_residual'].status == 'failed'
    np.testing.assert_array_equal(dist, original)


def test_static_dsge_has_no_dynamic_rule_but_shock_and_level_checks():
    import pandas as pd
    from puremacro.dsge.build import LinearModel
    from puremacro.dsge.klein import KleinSolution
    # The public container can represent a static solution even though build()
    # requires a predetermined state. Independent exact rule: x_t = 2 + e_t.
    rule = KleinSolution(G=np.empty((0, 0)), F=np.empty((1, 0)), N=np.empty((0, 1)),
                         L=np.ones((1, 1)), eu=(1, 1), eigenvalues=np.array([np.inf]))
    model = LinearModel(variables=('x',), states=(), controls=('x',), shocks=('e',),
                        steady_state=pd.Series({'x': 2.}), units={'x': 'level'}, solution=rule,
                        A=np.zeros((1, 1)), B=-np.ones((1, 1)), C=np.ones((1, 1)),
                        method='analytical', residual_norm=0.,
                        _equations=lambda lead, now, shock, p: np.array([now.x-2-shock.e]))
    report = audit_result(model, checks=('solver','equations','derivatives'))
    assert rows(report)['decision_rule_residual'].status == 'not_checked'
    assert rows(report)['shock_loading_residual'].status == 'passed'
    assert rows(report)['steady_state_residual'].status == 'passed'
    assert rows(report)['jacobian_current'].status == 'passed'


@pytest.mark.parametrize('with_shock', [False, True])
def test_dynare_compact_rule_audits_original_equations_and_supplied_solution(with_shock):
    from puremacro.dsge import build_dynare

    def equations(lead, now, lag, shock, parameters):
        innovation = shock.e if with_shock else 0.
        return np.array([now.x - .8*lag.x - innovation,
                         now.c - .5*lead.c - .3*now.x])

    # Analytically x_t=.8*x_{t-1}+e_t and c_t=.5*x_t. Put the
    # control before the state to exercise the original variable ordering.
    model = build_dynare(equations, variables=['c', 'x'], shocks=['e'] if with_shock else [],
                         steady_state={'c': 0., 'x': 0.})
    np.testing.assert_allclose(model.solution.G, [[.8]])
    np.testing.assert_allclose(model.solution.F, [[.4]])
    if with_shock:
        np.testing.assert_allclose(model.solution.N, [[1.]])
        np.testing.assert_allclose(model.solution.L, [[.5]])
    report = audit_result(model)
    assert rows(report)['decision_rule_residual'].status == 'passed'
    assert rows(report)['decision_rule_residual'].evidence_source == 'recomputed_dynare_lead_lag_equations'
    assert rows(report)['steady_state_residual'].status == 'passed'
    if with_shock:
        assert rows(report)['shock_loading_residual'].status == 'passed'
    corrupted = replace(model.solution, F=model.solution.F + .02)
    bad = audit_result(corrupted, model=model)
    assert rows(bad)['decision_rule_residual'].status == 'failed'
    # The model context remains correct; the bare result must actually be used.
    assert rows(audit_result(model))['decision_rule_residual'].status == 'passed'


def test_default_audit_does_not_access_network(household, monkeypatch):
    import socket
    def forbidden(*args, **kwargs):
        raise AssertionError('unexpected network')
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    assert rows(audit_result(household))['budget_residual'].status == 'passed'


@pytest.mark.mechanism_control
def test_network_blocker_control(monkeypatch):
    import socket
    def forbidden(*args, **kwargs):
        raise AssertionError('unexpected network')
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    with pytest.raises(AssertionError, match='unexpected network'):
        socket.create_connection(('example.invalid', 443))


def test_trade_horizon_scope_and_original_certificate_tolerances():
    from puremacro.trade.dynamic import HorizonComparisonResult
    comparison = HorizonComparisonResult(
        periods=4, short_horizon=8, long_horizon=16,
        max_log_difference=1e-7, tolerance=1e-6, window_passed=True,
        welfare_max_difference_pp=1e-6, welfare_tolerance=1e-5, welfare_passed=True,
        terminal_check='discounted_wealth', terminal_max_log_gap=.1, terminal_tolerance=1e-3,
        max_discounted_wealth=1e-8, discounted_wealth_tolerance=1e-6,
        terminal_passed=True, passed=True, status='verified_window_and_welfare',
        validation_scope='window and welfare only; discounted wealth terminal mode',
        metadata={'evidence': 'supplied horizon comparison'},
    )
    report = audit_result(comparison, checks=('refinement',))
    assert rows(report)['horizon_terminal_state'].status == 'not_checked'
    assert rows(report)['horizon_discounted_wealth'].status == 'passed'
    assert report.metadata['original_status'] == 'verified_window_and_welfare'
    assert report.metadata['validation_scope'] == comparison.validation_scope
    assert report.to_dataframe().attrs['original_metadata'] == comparison.metadata
