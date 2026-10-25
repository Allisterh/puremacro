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
    krylov = solve_perfect_foresight(equations, y_init=[1], y_end=[0], periods=5,
        shocks=np.zeros(5), terminal_policy=policy, terminal_jacobian=lambda last: coefficient.reshape(1, 1),
        linear_solver="krylov", tol=1e-12)
    assert result.converged and krylov.converged
    np.testing.assert_allclose(result.path.iloc[:, 0], root ** np.arange(1, 6), atol=1e-12)
    np.testing.assert_allclose(krylov.path.iloc[:, 0], root ** np.arange(1, 6), atol=1e-10)
    report = audit_result(result, checks=["equations"])
    assert np.max(np.abs(report.maps["path_equations"])) < 1e-10
    report_k = audit_result(krylov, checks=["equations"])
    assert np.max(np.abs(report_k.maps["path_equations"])) < 1e-10
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
    krylov = solve_transition(simple, y_init=[0], y_end=[0], shocks=[-1.], mcp=True, continuation=False, linear_solver="krylov")
    assert direct.converged and krylov.converged
    np.testing.assert_allclose(direct.path, 0, atol=1e-8)
    np.testing.assert_allclose(krylov.path, 0, atol=1e-8)


def _nearest_neighbor_mod(n: int = 100, nonlinear: bool = False) -> str:
    vars_decl = " ".join(f"y_{i}" for i in range(n))
    eqs = []
    for i in range(n):
        terms = [f"2.0*(y_{i} - STEADY_STATE(y_{i}))", f"-0.2*(y_{i}(+1) - STEADY_STATE(y_{i}))",
                 f"-0.3*(y_{i}(-1) - STEADY_STATE(y_{i}))"]
        if i > 0:
            terms.append(f"-0.1*(y_{i - 1} - STEADY_STATE(y_{i - 1}))")
        if i + 1 < n:
            terms.append(f"-0.1*(y_{i + 1} - STEADY_STATE(y_{i + 1}))")
        if nonlinear:
            terms.append(f"0.05*(y_{i} - STEADY_STATE(y_{i}))^3")
        if i == 0:
            terms.append("-e")
        eqs.append(" + ".join(terms) + " = 0;")
    init_decl = " ".join(f"y_{i}=1.0;" for i in range(n))
    return f"""
    var {vars_decl};
    varexo e;
    parameters alpha_p;
    alpha_p = 1.0;
    model;
    {" ".join(eqs)}
    end;
    initval;
    {init_decl}
    e = 0.0;
    end;
    """


def test_sparse_coordinate_jvp_matches_complex_step_and_invalidates_across_stages():
    from puremacro.dsge._transition_derivatives import TrajectoryJacobian
    from puremacro.dsge.transitions import _Coordinates

    model = read_mod("""
        var y_0 y_1 y_2;
        varexo e;
        parameters kappa;
        kappa = 0.5;
        model;
        2.0*(y_0 - STEADY_STATE(y_0)) - 0.2*(y_0(+1) - STEADY_STATE(y_0)) - 0.3*(y_0(-1) - STEADY_STATE(y_0)) + kappa*(y_1 - STEADY_STATE(y_1))^2 - e = 0;
        2.0*(y_1 - STEADY_STATE(y_1)) - 0.25*(y_1(+1) - STEADY_STATE(y_1)) - 0.25*(y_1(-1) - STEADY_STATE(y_1)) - 0.1*(y_0 - STEADY_STATE(y_0)) = 0;
        2.0*(y_2 - STEADY_STATE(y_2)) - 0.15*(y_2(+1) - STEADY_STATE(y_2)) - 0.35*(y_2(-1) - STEADY_STATE(y_2)) + 0.05*(y_2 - STEADY_STATE(y_2))^3 - 0.1*(y_1 - STEADY_STATE(y_1)) = 0;
        end;
        initval;
        y_0 = 1.0; y_1 = 1.2; y_2 = 1.5; e = 0.0;
        end;
    """)
    names = model.variables
    bounds = {"y_0": (0.0, None), "y_1": (0.2, 2.5)}
    var_scale = np.array([1.2, 0.8, 1.5])
    phys_scale = np.array([1.0, 1.0, 2.0])
    eq_scale = np.array([2.0, 0.5, 4.0])
    coords = _Coordinates(names, bounds, var_scale, physical_scale=phys_scale)
    y_init = np.array([1.0, 1.2, 1.5])
    y_end = np.array([1.1, 1.3, 1.6])
    z0 = coords.encode(y_init)
    zT = coords.encode(y_end)
    T = 6
    exo = np.linspace(0.01, 0.06, T)[:, None]
    jac = TrajectoryJacobian(
        model._dag, dict(model.parameter_values), y_end, coords=coords, equation_scale=eq_scale,
    )
    rng = np.random.default_rng(42)
    Z = np.tile(0.5 * (z0 + zT), (T, 1)) + 0.05 * rng.standard_normal((T, 3))
    factory = jac.make_stacked_jvp_factory(z0, zT, exo)
    matvec = factory(Z.ravel())
    v = rng.standard_normal(T * 3)
    jvp_sparse = matvec(v)

    # Complex-step directional derivative of the transformed stacked system
    from puremacro.dsge.transitions import _equations
    eval_raw = _equations(model, names)

    def stacked_transformed(Z_in):
        Z_mat = Z_in.reshape(T, 3)
        out = np.empty((T, 3), dtype=Z_in.dtype)
        for t in range(T):
            zl = z0 if t == 0 else Z_mat[t - 1]
            zc = Z_mat[t]
            zp = zT if t == T - 1 else Z_mat[t + 1]
            yp, yc, yl = coords.decode(zp), coords.decode(zc), coords.decode(zl)
            out[t] = np.asarray(eval_raw(yp, yc, yl, exo[t], steady_state=y_end), dtype=Z_in.dtype) / eq_scale
        return out.ravel()

    h = 1e-20
    jvp_complex = stacked_transformed(Z.ravel().astype(complex) + 1j * h * v).imag / h
    np.testing.assert_allclose(jvp_sparse, jvp_complex, rtol=1e-10, atol=1e-10)
    assert jac.dense_blocks_allocated == 0
    assert jac.jvp_value_evaluations == T

    # Re-applying at the same iterate reuses the cached operator without re-evaluating derivatives
    _ = matvec(2.0 * v)
    _ = factory(Z.ravel())(v)
    assert jac.jvp_value_evaluations == T
    assert jac.operator_builds == 1
    assert jac.operator_reuses == 1

    # Stage snapshot invalidation when parameters, steady reference, or coordinates change
    jac_param = TrajectoryJacobian(
        model._dag, {"kappa": 0.75}, y_end, coords=coords, equation_scale=eq_scale,
    )
    jac_ref = TrajectoryJacobian(
        model._dag, dict(model.parameter_values), y_end + 0.1, coords=coords, equation_scale=eq_scale,
    )
    coords_alt = _Coordinates(names, {"y_0": (0.1, None), "y_1": (0.2, 2.5)}, var_scale, physical_scale=phys_scale)
    jac_coord = TrajectoryJacobian(
        model._dag, dict(model.parameter_values), y_end, coords=coords_alt, equation_scale=eq_scale,
    )
    assert len({jac.stage_key, jac_param.stage_key, jac_ref.stage_key, jac_coord.stage_key}) == 4


@pytest.mark.parametrize("variant", ["linear", "nonlinear", "transformed"])
def test_100x1000_sparse_transition_scaling_benchmark(variant):
    n, T = 100, 1000
    nonlinear = variant in ("nonlinear", "transformed")
    model = read_mod(_nearest_neighbor_mod(n=n, nonlinear=nonlinear))
    y_init = np.ones(n)
    y_end = np.ones(n)
    shocks = np.zeros((T, 1))
    shocks[0, 0] = 0.15
    shocks[1, 0] = 0.08
    bounds = None
    var_scale = None
    eq_scale = None
    if variant == "transformed":
        bounds = {f"y_{i}": (0.0, None) if i % 2 == 0 else (0.1, 3.0) for i in range(n)}
        var_scale = np.full(n, 1.1)
        eq_scale = np.full(n, 1.5)

    common = dict(
        y_init=y_init, y_end=y_end, shocks=shocks, continuation=False,
        variable_bounds=bounds, variable_scale=var_scale, equation_scale=eq_scale, tol=1e-10,
    )
    direct = solve_transition(model, linear_solver="direct", **common)
    krylov = solve_transition(model, linear_solver="krylov", **common)

    assert direct.converged and krylov.converged
    assert direct.scaled_residual_norm <= 1e-9
    assert krylov.scaled_residual_norm <= 1e-9
    assert direct.physical_residual_norm <= 1e-9
    assert krylov.physical_residual_norm <= 1e-9
    np.testing.assert_allclose(krylov.path.to_numpy(), direct.path.to_numpy(), rtol=1e-8, atol=1e-8)

    # Sparse incidence preserved through solve_transition (~500k entries instead of ~30M)
    direct_diag = direct.solution.solver_diagnostics
    assert direct_diag["jacobian_nnz"] < 600_000
    assert direct_diag["dense_jvp_blocks"] == 0

    # Cached sparse JVP factory diagnostics for Krylov
    krylov_diag = krylov.solution.solver_diagnostics
    assert krylov_diag["jvp_method"] == "analytic"
    assert krylov_diag["jvp_backend"] == "stacked_factory"
    assert krylov_diag["dense_jvp_blocks"] == 0
    K = krylov_diag["operator_builds"]
    assert K >= 1
    assert krylov_diag["jvp_value_evaluations"] == T * K
    assert krylov_diag["jvp_cache_bytes"] < 32 * 1024 * 1024
    assert krylov_diag["preconditioner_storage_bytes"] > 0
    assert krylov_diag["krylov_basis_bytes"] > 0
    assert krylov_diag["process_peak_rss_bytes"] > 0


def test_physical_tolerances_and_mcp_diagnostics_enforced_under_equation_scaling():
    def equations(yp, yc, yl, e):
        return (yc - 1.0)**3 - e

    # Large equation_scale makes scaled residual look small after 1 iteration while physical residual is large
    lax = solve_transition(
        equations, y_init=[1.0], y_end=[1.0], shocks=np.full(4, 8.0),
        continuation=False, equation_scale=[1e12], max_iter=1, tol=1e-6,
    )
    strict_phys = solve_transition(
        equations, y_init=[1.0], y_end=[1.0], shocks=np.full(4, 8.0),
        continuation=False, equation_scale=[1e12], max_iter=1, tol=1e-6, physical_tol=1e-6,
    )
    assert lax.converged
    assert lax.physical_residual_norm > 1e-2
    assert not strict_phys.converged and strict_phys.reason == "target_failed"

    # MCP physical feasibility and complementarity diagnostics
    mcp_model = read_mod("""
        var x r; varexo e; parameters bound; bound = 0.0;
        model; [mcp='r > bound'] r = e; x = 1.0 - 2.0 * r; end;
        initval; x = 1.0; r = 0.0; e = 0.0; end;
    """)
    mcp_res = solve_transition(
        mcp_model, y_init=[1.0, 0.0], y_end=[1.0, 0.0],
        shocks=np.array([[-1.5], [-0.5], [0.5], [1.0]]),
        continuation=False, mcp=True, equation_scale=[10.0, 10.0],
        mcp_feasibility_tol=1e-8, mcp_complementarity_tol=1e-8,
    )
    assert mcp_res.converged
    assert mcp_res.physical_mcp_feasibility <= 1e-8
    assert mcp_res.physical_complementarity_residual <= 1e-8
    assert mcp_res.solution.physical_mcp_feasibility <= 1e-8
    assert mcp_res.solution.physical_complementarity_residual <= 1e-8


def test_unit_change_invariance_across_bounds_and_normalized_horizon_acceptance():
    def make_system(unit_factors):
        u0, u1 = unit_factors

        def equations(yp, yc, yl, e):
            y0_p, y1_p = yp[0] / u0, yp[1] / u1
            y0_c, y1_c = yc[0] / u0, yc[1] / u1
            y0_l, y1_l = yl[0] / u0, yl[1] / u1
            r0 = u0 * (y0_c - 0.2 * y0_p - 0.5 * y0_l - 0.3 - e[0])
            r1 = u1 * (y1_c - 0.15 * y1_p - 0.45 * y1_l - 0.4 - 0.1 * (y0_c - 1.0))
            return np.array([r0, r1], dtype=float)

        return dict(
            equations=equations,
            y_init=np.array([1.0 * u0, 1.0 * u1]),
            y_end=np.array([1.0 * u0, 1.0 * u1]),
            bounds={0: (0.0, None), 1: (0.2 * u1, 2.5 * u1)},
            physical_scale=np.array([u0, u1]),
            equation_scale=np.array([u0, u1]),
        )

    shocks = np.zeros((8, 1))
    shocks[0, 0] = 0.12
    base_sys = make_system((1.0, 1.0))
    scaled_sys = make_system((1e3, 1e-3))

    res_base = solve_transition(
        base_sys["equations"], y_init=base_sys["y_init"], y_end=base_sys["y_end"],
        shocks=shocks, continuation=True, initial_step=0.5,
        variable_bounds=base_sys["bounds"], physical_scale=base_sys["physical_scale"],
        equation_scale=base_sys["equation_scale"],
        normalized_horizon_tolerance=1e-5, compare_periods=4, max_periods=32, shock_tail=0.0,
        tol=1e-10,
    )
    res_scaled = solve_transition(
        scaled_sys["equations"], y_init=scaled_sys["y_init"], y_end=scaled_sys["y_end"],
        shocks=shocks, continuation=True, initial_step=0.5,
        variable_bounds=scaled_sys["bounds"], physical_scale=scaled_sys["physical_scale"],
        equation_scale=scaled_sys["equation_scale"],
        normalized_horizon_tolerance=1e-5, compare_periods=4, max_periods=32, shock_tail=0.0,
        tol=1e-10,
    )

    assert res_base.converged and res_scaled.converged
    assert res_base.reason == res_scaled.reason == "horizon_accepted"
    assert res_base.solves == res_scaled.solves
    assert len(res_base.horizon_history) == len(res_scaled.horizon_history)
    np.testing.assert_allclose(
        res_base.normalized_horizon_error, res_scaled.normalized_horizon_error, rtol=1e-9, atol=1e-11,
    )
    unscaled_path = res_scaled.path.to_numpy() / scaled_sys["physical_scale"][None, :]
    np.testing.assert_allclose(unscaled_path, res_base.path.to_numpy(), rtol=1e-10, atol=1e-10)
