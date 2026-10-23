"""Bounded refinement execution, meaningful stopping evidence, and solver oracle."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from puremacro.dsge.perfect_foresight import solve_perfect_foresight
from puremacro.validation import RefinementLadder, RefinementLevel, audit_result


def rows(report):
    return {row.check: row for row in report.records}


def ladder(values, calls, **kwargs):
    levels = []
    for i, value in enumerate(values):
        def solve(i=i, value=value):
            calls.append(i)
            return {"consumption": value}
        levels.append(RefinementLevel(f"level{i}", solve, settings={"nodes": 10*(i+2)}))
    return RefinementLadder("grid", levels, metrics=dict, atol=.01, rtol=0., **kwargs)


def run(baseline, plan):
    return audit_result(baseline, checks=("refinement",), refinement_ladders=(plan,))


def test_stops_only_after_consecutive_agreement_and_preserves_every_attempt():
    calls = []
    plan = ladder([2., 2.005, 2.006, 100.], calls, max_solves=4)
    report = run({"consumption": 1.}, plan)
    assert calls == [0,1,2]
    final = rows(report)["grid"]
    assert final.status == "passed"
    assert final.details["stop_reason"] == "stable_metrics"
    assert final.details["attempted_solves"] == 3
    assert final.details["consecutive_passes"] == 2
    assert [x["status"] for x in final.details["attempts"]] == ["failed", "passed", "passed"]
    assert rows(report)["grid/level0:consumption"].status == "failed"
    assert report.to_dataframe().attrs["refinement_ladders"]["grid"]["max_solves"] == 4
    assert '"stop_reason": "stable_metrics"' in report.to_json()
    assert "grid/level0:consumption" in report.to_markdown()


def test_failed_comparison_resets_consecutive_streak():
    calls = []
    report = run({"consumption": 1.}, ladder([1.001, 2., 2.001], calls))
    final = rows(report)["grid"]
    assert final.status == "failed"
    assert final.details["consecutive_passes"] == 1
    assert final.details["stop_reason"] == "levels_exhausted"


@pytest.mark.parametrize("budget,expected_calls,reason", [
    ({"max_solves": 1}, [0], "solve_budget_exhausted"),
    ({"max_cost": 1.5}, [0], "cost_budget_exhausted"),
    ({"max_cost": 0.}, [], "cost_budget_exhausted"),
])
def test_declared_budgets_prevent_extra_solves(budget, expected_calls, reason):
    calls = []
    report = run({"consumption": 1.}, ladder([2.,3.,4.], calls, **budget))
    assert calls == expected_calls
    final = rows(report)["grid"]
    assert final.status == ("failed" if calls else "not_checked")
    assert final.details["stop_reason"] == reason
    assert final.details["declared_cost_used"] == len(calls)


def test_nonuniform_cost_and_failed_callback_are_charged():
    calls = []
    def broken():
        calls.append("broken")
        raise RuntimeError("failed nonlinear solve")
    plan = RefinementLadder("horizon", [RefinementLevel("T100",broken,cost=2.)], metrics=dict,
                            max_cost=2.)
    report = run({"C":1.}, plan)
    final = rows(report)["horizon"]
    assert calls == ["broken"]
    assert final.status == "failed"
    assert final.details["declared_cost_used"] == 2.
    assert final.details["attempted_solves"] == 1
    assert final.details["completed_comparisons"] == 0
    assert final.details["stop_reason"] == "callback_failed"
    assert "failed nonlinear solve" in final.details["attempts"][0]["error"]


@pytest.mark.parametrize("bad", [{}, {"C":np.nan}, {"C":np.inf}, {"C":[1.]}, {"other":1.}])
def test_invalid_refined_metrics_fail_without_false_stabilization(bad):
    plan = RefinementLadder("bad", [RefinementLevel("bad",lambda:bad)], metrics=dict,
                            consecutive_passes=1)
    check = rows(run({"C":1.}, plan))["bad"]
    assert check.status == "failed"
    assert check.details["stop_reason"] == "invalid_result_or_metrics"


def test_invalid_baseline_never_calls_solver_and_convergence_is_required():
    calls = []
    plan = ladder([1.], calls)
    assert rows(run({"consumption":np.nan},plan))["grid"].details["stop_reason"] == "baseline_invalid"
    assert calls == []
    plan = RefinementLadder("failed", [RefinementLevel("fine",lambda:SimpleNamespace(converged=np.bool_(False),value=1.))],
                            metrics=lambda r:{"C":r.value}, consecutive_passes=1)
    assert rows(run(SimpleNamespace(converged=True,value=1.),plan))["failed"].status == "failed"


def test_pair_validation_refuses_changed_experiment():
    def validate(old,new):
        if old["shock"] != new["shock"]:
            raise ValueError("shock changed")
    plan = RefinementLadder("grid",[RefinementLevel("fine",lambda:{"C":1.,"shock":2.})],
                            metrics=lambda r:{"C":r["C"]}, validate_pair=validate, consecutive_passes=1)
    final = rows(run({"C":1.,"shock":1.},plan))["grid"]
    assert final.status == "failed"
    assert final.details["stop_reason"] == "pair_validation_failed"
    assert final.details["pair_validation"] == "caller_supplied"


def test_absolute_floor_and_multiple_metrics():
    plan = RefinementLadder("tiny", [RefinementLevel("fine",lambda:{"near_zero":1e-12,"wealth":2.})],
                            metrics=dict, atol=1e-8, rtol=0., consecutive_passes=1,
                            units={"wealth":"assets"})
    report = run({"near_zero":0.,"wealth":1.},plan)
    assert rows(report)["tiny/fine:near_zero"].details["relative_change"] is None
    assert rows(report)["tiny/fine:wealth"].units == "assets"
    assert rows(report)["tiny"].status == "failed"


def test_default_audit_does_not_run_refinement_callbacks():
    calls=[]
    plan=ladder([1.,1.],calls)
    report=audit_result({"consumption":1.},refinement_ladders=[plan])
    assert calls == []
    assert rows(report)["refinement"].status == "not_checked"
    run({"consumption":1.},plan)
    assert calls == [0,1]  # positive control: the actual callbacks run on opt-in


@pytest.mark.parametrize("kwargs", [
    {"max_solves":0}, {"max_solves":True}, {"max_solves":1.5},
    {"consecutive_passes":0}, {"max_cost":-1.}, {"max_cost":np.inf},
    {"atol":-1.}, {"rtol":np.nan}, {"levels":[]}, {"metrics":None},
])
def test_invalid_plan_rejected_even_without_opt_in(kwargs):
    plan=ladder([1.],[])
    with pytest.raises(ValueError):
        audit_result({"consumption":1.},refinement_ladders=[replace(plan,**kwargs)])


def test_level_name_cost_and_plan_name_validation():
    plan=ladder([1.,1.],[])
    for bad in (replace(plan,levels=[plan.levels[0],plan.levels[0]]),
                replace(plan,levels=[replace(plan.levels[0],cost=-1.)]),
                replace(plan,levels=[replace(plan.levels[0],name="")])):
        with pytest.raises(ValueError):
            run({"consumption":1.},bad)
    with pytest.raises(ValueError):
        audit_result({},refinement_ladders=[plan,plan])


def test_horizon_ladder_against_analytic_geometric_response():
    def solve(T):
        shocks=np.zeros(T);shocks[0]=1.
        return solve_perfect_foresight(lambda p,c,l,e:c-.5*l-e,y_init=np.array([0.]),
                                      y_ss=np.array([0.]),shocks=shocks,periods=T)
    def metrics(result):
        return {"cumulative_response":float(result.path.iloc[:,0].sum())}
    baseline=solve(2)
    calls=[]
    def callback(T):
        calls.append(T)
        result=solve(T)
        np.testing.assert_allclose(result.path.iloc[:,0], .5**np.arange(T), atol=1e-13)
        return result
    plan=RefinementLadder("horizon",[RefinementLevel(f"T{T}",lambda T=T:callback(T),settings={"T":T})
                                     for T in (4,8,16,32,64)],metrics=metrics,max_solves=4,
                           atol=.01,rtol=0.,consecutive_passes=2)
    report=run(baseline,plan)
    assert calls == [4,8,16,32]
    final=rows(report)["horizon"]
    assert final.status == "passed"
    last=final.details["attempts"][-1]["metrics"]["cumulative_response"]
    assert last == pytest.approx(2*(1-.5**32),abs=1e-13)
    assert final.details["pair_validation"] == "not_checked"
