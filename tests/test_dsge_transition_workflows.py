"""Economic and numerical oracles for the shared transition workflow."""
from dataclasses import replace

import numpy as np
import pytest

from puremacro.dsge.model import read_mod
from puremacro.dsge.perfect_foresight import solve_perfect_foresight
from puremacro.dsge.extended_path import extended_path
from puremacro.dsge.transitions import ShockAnnouncement, solve_announced_path, solve_transition
from puremacro.validation import audit_result


LINEAR = """
var x; varexo e;
model; x = 0.2*x(+1)+0.3*x(-1)+e; end;
initval; x=0; e=0; end;
"""


@pytest.mark.parametrize("backend", ["direct", "krylov"])
def test_sparse_symbolic_derivatives_preserve_fixed_steady_reference(backend):
    model = read_mod("""
        var y x; varexo e;
        model; y=0.2*y(+1)+0.3*y(-1)+e; x=STEADY_STATE(y)*y; end;
        initval; y=1; x=1; e=0; end;
    """)
    options = dict(y_init=[1, 1], y_end=[2, 4], periods=20, shocks=np.ones((20, 1)), tol=1e-10)
    numerical = solve_perfect_foresight(model, method="central", **options)
    analytical = solve_perfect_foresight(model, linear_solver=backend, **options)
    assert analytical.converged and numerical.converged
    np.testing.assert_allclose(analytical.path, numerical.path, atol=1e-9)
    np.testing.assert_allclose(analytical.path.x, 2 * analytical.path.y, atol=1e-10)
    if backend == "direct":
        assert analytical.solver_diagnostics["derivatives"] == "analytic"
        assert analytical.solver_diagnostics["jacobian_nnz"] < (3 * 20 - 2) * 4
    else:
        assert analytical.solver_diagnostics["jvp_method"] == "analytic"


@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.pyodide_smoke
def test_mcp_tags_use_equation_row_instead_of_diagonal_loading(upper):
    tag = "r < bound" if upper else "r > bound"
    source = f"""
        var x r; varexo e; parameters bound; bound=0;
        model; [mcp='{tag}'] r=e; x=1-3*r; end;
        initval; x=1; r=0; e=0; end;
    """
    model = read_mod(source)
    result = solve_perfect_foresight(model, periods=4, shocks=np.full((4, 1), 2 if upper else -2), mcp=True)
    assert result.converged
    np.testing.assert_allclose(result.path.r, 0, atol=1e-8)
    np.testing.assert_allclose(result.path.x, 1, atol=1e-8)


def test_explicit_mcp_mapping_preserves_sign():
    def equations(yp, yc, yl, e):
        x, r = yc
        return [e[0] - r, x + 3*r - 1]
    result = solve_perfect_foresight(equations, y_init=[1, 0], y_end=[1, 0],
        periods=3, shocks=-np.ones((3, 1)), variable_names=["x", "r"],
        mcp_bounds={"r": (0, None)}, mcp_equations={"r": (0, -1)})
    assert result.converged
    np.testing.assert_allclose(result.path, np.tile([1, 0], (3, 1)), atol=1e-9)


@pytest.mark.pyodide_smoke
def test_terminal_policy_chain_rule_and_independent_audit():
    root = (1 - np.sqrt(1 - 4 * .25 * .4)) / (2 * .25)
    coefficient = np.array([root])
    policy = lambda last: coefficient * last
    def equations(yp, yc, yl, e):
        return yc - .25 * yp - .4 * yl - e
    result = solve_perfect_foresight(equations, y_init=[1], y_end=[0], periods=5,
        shocks=np.zeros(5), terminal_policy=policy, terminal_jacobian=lambda last: coefficient.reshape(1, 1), tol=1e-12)
    assert result.converged
    np.testing.assert_allclose(result.path.iloc[:, 0], root ** np.arange(1, 6), atol=1e-12)
    report = audit_result(result, checks=["equations"])
    assert np.max(np.abs(report.maps["path_equations"])) < 1e-10
    altered = result.path.copy()
    altered.iloc[-1, 0] += 0.1
    bad = audit_result(replace(result, path=altered), checks=["equations"])
    assert np.max(np.abs(bad.maps["path_equations"])) > 1e-3
    coefficient[:] = 0.1
    changed = audit_result(result, checks=["equations"])
    assert changed.records[0].status == "failed"
    assert "terminal_policy changed" in changed.records[0].details["error"]


def test_continuation_recovers_target_that_exceeds_per_solve_iteration_budget():
    def equations(yp, yc, yl, e):
        return yc**3 - 1 - e
    options = dict(y_init=[1], y_end=[1], shocks=np.full(3, 999.), max_iter=4, tol=1e-9)
    naive = solve_transition(equations, continuation=False, **options)
    continued = solve_transition(equations, max_solves=30, **options)
    assert not naive.converged
    assert continued.converged and continued.reason == "target_solved"
    np.testing.assert_allclose(continued.path, 10, atol=1e-9)
    assert any(not row["converged"] for row in continued.continuation_history)
    assert continued.continuation_history[-1]["fraction"] == 1


def test_parameter_continuation_retains_exact_target_and_audit_calibration():
    model = read_mod("var x; parameters theta; theta=999; model; x^3=1+theta; end; initval; x=1; end;")
    result = solve_transition(model, y_init=[1], y_end=[1], shocks=np.zeros((3, 0)),
                              baseline_params={"theta": 0}, max_iter=4, tol=1e-9)
    assert result.converged
    np.testing.assert_allclose(result.path, 10, atol=1e-9)
    assert result.continuation_history[-1]["parameters"] == {"theta": 999}
    assert model.parameter_values["theta"] == 999
    model.parameter_values["theta"] = 0
    audit = audit_result(result.solution, checks=["equations"])
    assert np.max(np.abs(audit.maps["path_equations"])) < 1e-9


def test_continuation_uses_exact_target_under_large_baseline_cancellation():
    def equations(yp, yc, yl, e):
        return yc - np.atleast_1d(e)
    result = solve_transition(equations, y_init=[1e16], y_end=[1.],
        shocks=np.ones((3, 1)), baseline_shocks=np.full((3, 1), 1e16), initial_step=1, max_solves=3)
    assert result.converged
    np.testing.assert_array_equal(result.path, 1.)
    np.testing.assert_array_equal(result.solution._audit_context["y_end"], [1.])
    np.testing.assert_array_equal(result.solution._audit_context["exogenous_path"], np.ones((3, 1)))
    model = read_mod("var x; parameters target; target=1; model; x=target; end; initval; x=1; end;")
    parameter_result = solve_transition(model, y_init=[1e16], y_end=[1.], shocks=np.zeros((3, 0)),
        baseline_params={"target": 1e16}, initial_step=1, max_solves=3)
    assert parameter_result.converged
    assert parameter_result.continuation_history[-1]["parameters"]["target"] == 1.
    np.testing.assert_array_equal(parameter_result.path, 1.)


@pytest.mark.parametrize("backend", ["direct", "krylov"])
@pytest.mark.pyodide_smoke
def test_positive_domain_is_respected_even_during_derivative_and_line_search_probes(backend):
    evaluated = []
    def equations(yp, yc, yl, e):
        assert np.all(np.real(yc) > 0)
        evaluated.append(np.real(yc).copy())
        return np.log(yc) - e
    result = solve_transition(equations, y_init=[1], y_end=[1], shocks=np.full(5, -10.),
                              variable_bounds={"y_0": (0, None)}, linear_solver=backend)
    assert result.converged
    np.testing.assert_allclose(result.path, np.exp(-10), rtol=1e-8)
    assert np.min(evaluated) > 0


def test_explicit_scales_control_each_equation_and_preserve_original_residuals():
    scales = np.array([1e12, 1e-12])
    target = np.array([2., 3.])
    def equations(yp, yc, yl, e):
        return scales * (yc - e * target)
    result = solve_transition(equations, y_init=[0, 0], y_end=[0, 0], shocks=np.ones(4),
        equation_scale=scales, variable_scale=[1e3, 1e-3], tol=1e-10)
    assert result.converged and result.scaled_residual_norm <= 1e-10
    np.testing.assert_allclose(result.path, np.tile(target, (4, 1)), atol=1e-10)
    assert result.solution.solver_diagnostics["original_equation_residual_norm"] == result.solution.residual_norm


def test_continuation_budget_never_reports_baseline_as_target():
    result = solve_transition(read_mod(LINEAR), y_init=[0], y_end=[0], shocks=np.ones(5), max_solves=1)
    assert not result.converged and result.reason == "solve_budget_exhausted"
    assert result.solves == 1 and result.continuation_history[-1]["fraction"] == 0
    np.testing.assert_array_equal(result.path, 0)


def test_transition_result_export_and_headless_plot_contract(monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = solve_transition(read_mod(LINEAR), y_init=[0], y_end=[0],
                              shocks=np.ones(4), continuation=False)
    frame = result.to_frame()
    np.testing.assert_array_equal(result.to_dataframe(), frame)
    frame.iloc[0, 0] = -999
    assert result.path.iloc[0, 0] != -999
    assert "x" in result.to_markdown(head=2, digits=3)
    assert "\\begin{tabular}" in result.to_latex(index=False)
    assert "#table(" in result.to_typst(head=2)
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))
    figure, axes = plt.subplots()
    try:
        assert result.plot(ax=axes, show=False) is figure
        np.testing.assert_allclose(axes.lines[0].get_ydata(), result.path.x)
        assert not shown
    finally:
        plt.close(figure)
    with pytest.raises(ValueError, match="No numerical solution"):
        replace(result, solution=None).plot()


def test_horizon_refinement_records_common_window_and_hard_budget():
    model = read_mod(LINEAR)
    options = dict(y_init=[0], y_end=[0], shocks=np.r_[1., np.zeros(7)],
                   continuation=False, compare_periods=3, shock_tail=0.)
    accepted = solve_transition(model, horizon_tolerance=1e-7, max_periods=32, **options)
    assert accepted.converged and accepted.reason == "horizon_accepted"
    assert accepted.horizon_history[-1]["accepted"]
    short = solve_transition(model, horizon_tolerance=1e-15, max_periods=9, **options)
    assert not short.converged and short.reason == "horizon_budget_exhausted"
    assert len(short.path) == 9
    with pytest.raises(ValueError, match="explicit max_periods and shock_tail"):
        solve_transition(model, y_init=[0], y_end=[0], shocks=np.zeros(4), horizon_tolerance=1e-4, compare_periods=3)


def test_linear_continuation_reuses_only_identical_preconditioner():
    result = solve_transition(read_mod(LINEAR), y_init=[0], y_end=[0], shocks=np.ones(6), linear_solver="krylov")
    assert result.converged and result.solves > 1
    assert result.preconditioner_builds == 1


def test_extended_path_reuses_exact_blocks_and_retains_diagnostics():
    model = read_mod(LINEAR)
    options = dict(periods=8, horizon=10, y_init=[0], y_ss=[0],
                   shocks=np.linspace(-.1, .1, 8)[:, None], linear_solver="krylov")
    cached = extended_path(model, **options)
    uncached = extended_path(model, reuse_preconditioner=False, **options)
    assert cached.converged and uncached.converged
    np.testing.assert_allclose(cached.path, uncached.path, atol=1e-10)
    assert cached.solver_diagnostics["preconditioner_builds"] == 1
    assert cached.solver_diagnostics["preconditioner_reuses"] > 1
    assert len(cached.solver_diagnostics["rolling_solves"]) == 8
    assert not uncached.solver_diagnostics["preconditioner_reuse"]


def test_explicit_no_preconditioner_is_not_overridden_by_cache():
    result = solve_transition(read_mod(LINEAR), y_init=[0], y_end=[0], shocks=np.ones(4),
                              linear_solver="krylov", krylov_options={"preconditioner": None})
    assert result.converged
    assert result.preconditioner_builds == 0


def test_invalid_mcp_mapping_is_not_silently_inferred():
    with pytest.raises(ValueError, match="map exactly"):
        solve_perfect_foresight(lambda yp, yc, yl, e: yc-e, y_init=[0], y_end=[0],
            periods=3, mcp_bounds={0: (0, None)}, mcp_equations={1: (0, 1)})


@pytest.mark.pyodide_smoke
def test_announcements_change_expectations_only_at_the_revelation_date():
    model = read_mod(LINEAR)
    shocks = np.zeros((20, 1))
    shocks[7] = 1
    anticipated = solve_announced_path(model, [ShockAnnouncement(1, shocks)], y_init=[0], y_end=[0])
    surprise = solve_announced_path(model, [ShockAnnouncement(1, np.zeros_like(shocks)),
        ShockAnnouncement(8, shocks)], y_init=[0], y_end=[0])
    assert anticipated.converged and surprise.converged
    assert np.max(np.abs(anticipated.path.iloc[:7])) > 0.1
    np.testing.assert_array_equal(surprise.path.iloc[:7], 0)
    conditional = solve_perfect_foresight(model, y_init=[0], y_end=[0], periods=13, shocks=shocks[7:])
    np.testing.assert_allclose(surprise.path.iloc[7:], conditional.path, atol=1e-12)
    rewritten_past = shocks.copy()
    rewritten_past[0] = 1
    with pytest.raises(ValueError, match="previously realized"):
        solve_announced_path(model, [ShockAnnouncement(1, shocks), ShockAnnouncement(8, rewritten_past)], y_init=[0], y_end=[0])
    # Passing the rolling-surprise option would silently replace this calendar's
    # information dates, so reject it even for a single announced experiment.
    with pytest.raises(ValueError, match="announcement calendar"):
        solve_announced_path(model, [ShockAnnouncement(1, shocks)], y_init=[0], y_end=[0], surprise=True)


def test_transition_mcp_acceptance_uses_complementarity_not_raw_zero_equation():
    def equations(yp, yc, yl, e):
        return yc - e
    result = solve_transition(equations, y_init=[0], y_end=[0], shocks=-np.ones(4),
                              mcp_bounds={0: (0, None)}, mcp_equations={0: (0, 1)})
    assert result.converged
    np.testing.assert_allclose(result.path, 0, atol=1e-8)


def test_transition_preserves_model_mcp_tags_and_parameter_dependent_bounds():
    model = read_mod("""var x; varexo e; parameters floor; floor=0.25;
        model; [mcp='x > floor'] x=e; end; initval; x=0.25; e=0.25; end;""")
    result = solve_transition(model, y_init=[.25], y_end=[.25], shocks=-np.ones((3, 1)),
                              baseline_params={"floor": 0}, mcp=True)
    assert result.converged
    np.testing.assert_allclose(result.path, .25, atol=1e-8)
    assert result.solution.binding_periods["x"] == [1, 2, 3]
    simple = read_mod("var x; varexo e; model; [mcp='x > 0'] x=e; end; initval; x=0; end;")
    direct = solve_transition(simple, y_init=[0], y_end=[0], shocks=[-1.], mcp=True, continuation=False)
    assert direct.converged
    np.testing.assert_allclose(direct.path, 0, atol=1e-8)
