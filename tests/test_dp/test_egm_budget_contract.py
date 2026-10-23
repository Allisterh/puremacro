"""Continuous EGM policies need budget identities beyond the asset grid nodes."""
from __future__ import annotations

import numpy as np
import pytest

from puremacro import dp


def _model(budget="1 + (1+r)*a - a(+1)", *, horizon=None):
    model = dp.Model("budget contract").parameters(beta=.95, r=.02)
    model.state("a", [0., 1., 2.])
    model.local("c = " + budget).reward("log(c)").subject_to("c > 0")
    if horizon is not None:
        model.horizon(horizon)
    return model


@pytest.mark.parametrize("asset", ["a", "a(+1)"])
@pytest.mark.parametrize("horizon", [None, 2])
@pytest.mark.parametrize("frontend", ["python", "mod"])
def test_nonlinear_budgets_equal_on_grid_are_rejected(asset, horizon, frontend):
    # The extra term vanishes at every declared node, yet changes resources at
    # the continuous policies EGM computes. More probes cannot prove identity.
    budget = f"1 + (1+r)*a - a(+1) + .2*{asset}*({asset}-1)*({asset}-2)"
    if frontend == "python":
        run = lambda: _model(budget, horizon=horizon).solve("egm", distribution=False)
    else:
        lifetime = "" if horizon is None else f"horizon = {horizon};"
        source = f"""parameters beta r; beta=.95; r=.02;
        state a=[0,1,2]; {lifetime}
        model; c={budget}; reward=log(c); end;
        constraints; c>0; end; solve(method=egm);"""
        run = lambda: dp.solve_mod(source)
    with pytest.raises(dp.ModelSpecError, match="budget|linear"):
        run()


def test_custom_asset_budget_cannot_pass_by_agreeing_on_grid():
    model = _model("1 + (1+r)*a - saving_cost(a(+1))")
    model.functions(saving_cost=lambda x: x + .2*x*(x-1)*(x-2))
    with pytest.raises(dp.ModelSpecError, match="budget|linear"):
        model.solve("egm", distribution=False)


@pytest.mark.parametrize("denominator", [
    "1 + .2*a(+1)*(a(+1)-1)*(a(+1)-2)",
    "a(+1)-.5",
])
def test_asset_dependent_denominators_cannot_hide_off_grid_changes(denominator):
    # The second form would cancel algebraically, hiding a singularity at .5.
    extra = f"({denominator})/({denominator}) - 1" if denominator == "a(+1)-.5" else f"1/({denominator}) - 1"
    model = _model("1 + (1+r)*a - a(+1) + " + extra, horizon=2)
    with pytest.raises(dp.ModelSpecError, match="budget|linear"):
        model.solve("egm")


def test_extra_constraint_equal_on_grid_is_rejected_before_life_cycle_solve():
    model = _model(horizon=2)
    model.subject_to("a(+1)*(a(+1)-1)*(a(+1)-2) <= 0")
    # Previously this returned a solution with -inf values because the extra
    # constraint was invisible on nodes but excluded off-grid saving policies.
    with pytest.raises(dp.ModelSpecError, match="exactly the constraint"):
        model.solve("egm")


def test_large_income_cannot_hide_nonunit_saving_cost():
    # A relative tolerance scaled by income used to admit a price of two.
    model = _model("1e12 + (1+r)*a - 2*a(+1)", horizon=2)
    with pytest.raises(dp.ModelSpecError, match="coefficient|budget"):
        model.solve("egm")


@pytest.mark.parametrize("variation", ["age", "shock", "choice"])
def test_saving_coefficient_checked_beyond_first_age_shock_or_choice(variation):
    price = {"age": "price", "shock": "1+z", "choice": "1+h"}[variation]
    model = _model(f"1e12 + (1+r)*a - ({price})*a(+1)", horizon=2)
    if variation == "age":
        model.parameters(price=np.array([1., 2.]))
    elif variation == "shock":
        model.exogenous("z", dp.Markov([0., 1.], [[.8, .2], [.2, .8]]))
    else:
        model.discrete("h", [0., 1.])
    with pytest.raises(dp.ModelSpecError, match="coefficient"):
        model.solve("egm")


@pytest.mark.parametrize("budget", [
    "1 + (1+r)*a - a(+1)",
    "1 - (a(+1) - (1+r)*a)",
    "(2 + 2*(1+r)*a - 2*a(+1))/2",
    "1 + (1+r)*a^1 - a(+1)",
])
@pytest.mark.parametrize("horizon", [None, 2])
def test_equivalent_affine_budgets_preserve_continuous_policies(budget, horizon):
    reference = _model(horizon=horizon).solve("egm", distribution=False)
    solution = _model(budget, horizon=horizon).solve("egm", distribution=False)
    np.testing.assert_allclose(solution.raw.aprime, reference.raw.aprime, atol=1e-12)
    np.testing.assert_allclose(solution.raw.c, reference.raw.c, atol=1e-12)
    assert np.any((solution.raw.aprime > 0) & (solution.raw.aprime < 1))
    current = np.array([0., 1., 2.])[:, None]
    resources = np.broadcast_to(1 + 1.02*current, solution.raw.c.shape)
    np.testing.assert_allclose(solution.raw.c + solution.raw.aprime, resources)


def test_custom_income_and_intermediate_locals_remain_supported():
    model = dp.Model("taxed income").parameters(beta=.95, r=.02)
    model.state("wealth", [0., 1., 2.])
    model.exogenous("z", dp.Markov([.5, 1.5], [[.8, .2], [.2, .8]]))
    model.functions(tax=lambda y: .15*y + .10*np.maximum(y-1., 0.))
    model.local("income = z - tax(z)", "resources = (1+r)*wealth + income",
                "c = resources - wealth(+1)", "spending = c")
    model.reward("log(spending)").subject_to("0 < c")
    solution = model.solve("egm", distribution=False)
    z = np.array([.5, 1.5])
    income = z - .15*z - .10*np.maximum(z-1., 0.)
    np.testing.assert_allclose(solution.raw.c + solution.raw.aprime,
                               1.02*np.array([0., 1., 2.])[:, None] + income)


def test_vfi_still_accepts_nonlinear_resource_equations():
    solution = _model("1 + (1+r)*a - a(+1) - .01*a(+1)^2").solve("vfi")
    assert np.isfinite(solution.V).all()
