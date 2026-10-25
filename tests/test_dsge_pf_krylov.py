"""Front-door Krylov parity, boundary semantics and numerical failure controls."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from puremacro.dsge import extended_path, load_mod, solve_perfect_foresight
from puremacro.dsge._parser import parse_mod_to_dag
from puremacro.dsge.stacked_newton import StackedProblem, _resolve_preconditioner


FIXTURES = Path(__file__).parent / "fixtures/dynare_issue_177"


@pytest.mark.parametrize("name", ["endval_literal", "endval_solved", "endval_no_steady", "endval_with_steady"])
@pytest.mark.parametrize("method", ["central", "complex"])
def test_krylov_matches_native_terminal_and_steady_state_fixtures(name, method):
    model = load_mod((FIXTURES / f"{name}.mod").read_text(encoding="utf-8"))
    result = solve_perfect_foresight(model, periods=10, method=method, linear_solver="krylov")
    reference = pd.read_csv(FIXTURES / f"{name}_dynare_out.csv").iloc[1:11]
    assert result.converged
    assert result.linear_solver == "krylov"
    np.testing.assert_allclose(result.path, reference, rtol=0, atol=1e-9)
    assert result.solver_diagnostics["converged_by"] == "residual_and_step"
    assert result.solver_diagnostics["linear_history"]


@pytest.mark.parametrize("solver", ["direct", "krylov"])
def test_explicit_terminal_reference_padding_and_context_copy(solver):
    model = parse_mod_to_dag("""
        var y x; varexo e;
        model; y = 0.5 + 0.5*y(+1) + e; x = STEADY_STATE(y); end;
        initval; y=1; x=1; end; steady;
    """)
    end = np.array([3.0, 3.0])
    shocks = np.array([[99.0], [0.1], [0.2], [-0.1], [77.0]])
    result = solve_perfect_foresight(model, periods=3, y_ss=[1, 1], y_end=end,
                                     shocks=shocks, linear_solver=solver)
    y = np.empty(3)
    lead = 3.0
    for t in range(2, -1, -1):
        y[t] = 0.5 + 0.5 * lead + shocks[t + 1, 0]
        lead = y[t]
    assert result.converged
    np.testing.assert_allclose(result.path.y, y, atol=1e-10)
    np.testing.assert_allclose(result.path.x, 3.0, atol=1e-10)
    end[:] = -100
    shocks[:] = -100
    context = result._audit_context
    assert context["mode"] == "perfect_foresight"
    np.testing.assert_array_equal(context["y_end"], [3, 3])
    np.testing.assert_allclose(context["equations"]([3, 3], [2, 3], [1, 1], [0]), [0, 0])
    np.testing.assert_allclose(context["exogenous_path"].ravel(), [0.1, 0.2, -0.1])


def nonlinear(lead, current, lag, shock):
    return [current[0] ** 2 - 0.2 * lead[0] - 0.3 * lag[0] - 0.5 - np.asarray(shock).item()]


def test_nonlinear_path_matches_direct_and_preserves_dampening():
    kwargs = dict(y_init=[0.9], y_ss=[1.0], periods=12,
                  shocks=np.linspace(0.1, 0, 12), dampening=0.5, max_iter=90)
    direct = solve_perfect_foresight(nonlinear, **kwargs)
    krylov = solve_perfect_foresight(nonlinear, linear_solver="krylov", **kwargs)
    assert direct.converged and krylov.converged
    np.testing.assert_allclose(krylov.path, direct.path, atol=2e-8, rtol=0)
    steps = [r["step"] for r in krylov.solver_diagnostics["linear_history"]]
    assert any(0 < step <= 0.5 for step in steps)
    assert max(steps) <= 0.5


def test_krylov_failure_returns_last_path_and_never_falls_back():
    with pytest.warns(RuntimeWarning, match="krylov.*iteration limit"):
        result = solve_perfect_foresight(nonlinear, y_init=[1], y_ss=[1],
                                         shocks=np.full(4, 4.0), periods=4,
                                         max_iter=1, linear_solver="krylov")
    assert not result.converged
    assert result.solver_diagnostics["converged_by"] == "iteration_limit"
    assert result.residual_norm > 1e-8
    assert not np.allclose(result.path, 1.0)


@pytest.mark.parametrize("options,exception", [
    ({"linear_solver": "bad"}, ValueError),
    ({"krylov_options": {}}, ValueError),
    ({"linear_solver": "krylov", "mcp_equations": {0: 0}}, ValueError),
    ({"linear_solver": "krylov", "mcp_bounds": {0: (1.0, 0.0)}}, ValueError),
    ({"linear_solver": "krylov", "terminal_jacobian": lambda y: np.eye(1)}, ValueError),
    ({"linear_solver": "krylov", "krylov_options": {"step_tol": None}}, ValueError),
    ({"linear_solver": "krylov", "krylov_options": {"krylov_method": "bad"}}, ValueError),
    ({"linear_solver": "krylov", "krylov_options": {"preconditioner": "bad"}}, ValueError),
    ({"linear_solver": "krylov", "krylov_options": {"preconditioner": 0}}, TypeError),
    ({"linear_solver": "krylov", "krylov_options": {"inner_m": None}}, ValueError),
])
def test_invalid_options_rejected_before_model_evaluation(options, exception):
    calls = []
    def equations(lead, current, lag, shock):
        calls.append(True)
        return [current[0]]
    with pytest.raises(exception):
        solve_perfect_foresight(equations, y_init=[0], y_ss=[0], periods=3, **options)
    assert calls == []


@pytest.mark.parametrize("route", ["extended", "surprise"])
def test_rolling_krylov_forwarding_preserves_shocks_and_fixed_reference(route):
    def equations(lead, current, lag, shock, *, steady_state=None):
        return [current[0] - 0.4 * lag[0] - 0.6 * steady_state[0] - np.asarray(shock).item()]
    shocks = np.array([0.2, -0.1, 0.3])
    if route == "extended":
        result = extended_path(equations, periods=3, horizon=4, y_init=[2], y_ss=[2],
                               shocks=shocks[:, None], linear_solver="krylov", strict=True)
    else:
        result = solve_perfect_foresight(equations, periods=3, y_init=[2], y_ss=[2],
                                        shocks=shocks, surprise=True, linear_solver="krylov")
        assert result.linear_solver == "krylov"
        assert result._audit_context is None
    expected = []
    y = 2.0
    for e in shocks:
        y = 0.4 * y + 1.2 + e
        expected.append(y)
    assert result.converged
    np.testing.assert_allclose(result.path.iloc[:, 0], expected, atol=1e-9)


def test_current_preconditioner_reuses_only_exact_unchanged_blocks():
    coefficient = np.array([2.0])
    def eq(lead, current, lag, shock):
        return coefficient * current - shock
    def blocks(lead, current, lag, shock):
        return np.zeros((1, 1)), coefficient.reshape(1, 1), np.zeros((1, 1))
    problem = StackedProblem(eq, [0], [0], np.ones(3), blocks_fn=blocks)
    _, factory = _resolve_preconditioner(problem, "current_block_tridiagonal", None)
    first = factory(np.zeros(3))
    assert factory(np.ones(3)) is first
    np.testing.assert_allclose(first @ np.ones(3), 0.5)
    coefficient[:] = np.nextafter(2.0, 3.0)
    second = factory(np.ones(3))
    assert second is not first  # even sub-tolerance changes must invalidate
    coefficient[:] = 4.0
    third = factory(np.ones(3))
    assert third is not second
    np.testing.assert_allclose(third @ np.ones(3), 0.25)


def test_callback_typeerror_propagates_with_krylov():
    def equations(lead, current, lag, shock, *, steady_state=None):
        raise TypeError("callback body failure")
    with pytest.raises(TypeError, match="callback body failure"):
        solve_perfect_foresight(equations, y_init=[1], y_ss=[1], periods=2,
                                linear_solver="krylov", method="central")


def test_surprise_failure_is_not_reported_as_converged():
    with pytest.warns(RuntimeWarning, match="krylov"):
        result = solve_perfect_foresight(nonlinear, y_init=[1], y_ss=[1], periods=2,
                                        shocks=[4.0, 4.0], surprise=True,
                                        max_iter=1, linear_solver="krylov")
    assert not result.converged
    assert result.solver_diagnostics["mode"] == "surprise"


@pytest.mark.parametrize("with_analytic_jac", [False, True])
@pytest.mark.parametrize("method", ["auto", "central", "complex"])
def test_krylov_terminal_policy_parity_and_chain_rule_directional_derivative(with_analytic_jac, method):
    from puremacro.dsge import read_mod, solve_transition
    from puremacro.validation import audit_result

    model = read_mod("""
        var k c;
        varexo e;
        parameters rho alpha;
        rho = 0.6;
        alpha = 0.35;
        model;
            k = rho * k(-1) + alpha * c(+1) + e;
            c = 0.5 * k + 0.25 * c(+1) - 0.1 * k(-1);
        end;
        initval; k = 0.0; c = 0.0; e = 0.0; end;
    """)
    K_mat = np.array([[0.45, -0.12], [0.22, 0.30]], dtype=float)
    y_star = np.array([0.2, 0.1], dtype=float)

    def term_policy(y_last):
        y = np.asarray(y_last, dtype=float)
        return y_star + K_mat @ (y - y_star) + 0.05 * np.array([(y[0] - y_star[0]) ** 2, 0.0])

    def term_jac(y_last):
        y = np.asarray(y_last, dtype=float)
        J = K_mat.copy()
        J[0, 0] += 0.10 * (y[0] - y_star[0])
        return J

    shocks = np.zeros((8, 1))
    shocks[0, 0] = 0.15
    shocks[1, 0] = -0.05
    common = dict(
        y_init=np.array([0.0, 0.0]),
        y_end=y_star,
        periods=8,
        shocks=shocks,
        method=method,
        terminal_policy=term_policy,
        terminal_jacobian=term_jac if with_analytic_jac else None,
        tol=1e-11,
    )
    direct = solve_perfect_foresight(model, linear_solver="direct", **common)
    krylov = solve_perfect_foresight(model, linear_solver="krylov", **common)
    assert direct.converged and krylov.converged
    np.testing.assert_allclose(krylov.path.to_numpy(), direct.path.to_numpy(), atol=1e-9, rtol=0)
    assert krylov.solver_diagnostics["has_terminal_policy"] is True
    report = audit_result(krylov)
    assert report.solver_converged is True
    assert {r.check: r.status for r in report.records}["path_equations"] == "passed"

    # Also verify through solve_transition with coordinate scaling
    tr_direct = solve_transition(
        model, y_init=[0.0, 0.0], y_end=y_star, shocks=shocks, continuation=False,
        terminal_policy=term_policy, terminal_jacobian=term_jac if with_analytic_jac else None,
        variable_scale=[1.2, 0.8], equation_scale=[0.9, 1.3], linear_solver="direct", tol=1e-11,
    )
    tr_krylov = solve_transition(
        model, y_init=[0.0, 0.0], y_end=y_star, shocks=shocks, continuation=False,
        terminal_policy=term_policy, terminal_jacobian=term_jac if with_analytic_jac else None,
        variable_scale=[1.2, 0.8], equation_scale=[0.9, 1.3], linear_solver="krylov", tol=1e-11,
    )
    assert tr_direct.converged and tr_krylov.converged
    np.testing.assert_allclose(tr_krylov.path.to_numpy(), tr_direct.path.to_numpy(), atol=1e-9, rtol=0)

    # Independent directional-derivative check of the terminal chain rule in StackedProblem.jvp
    eq_fn = model.compile_equations()
    params = model._dag.parameter_values
    from puremacro.dsge.build import _Vec
    p_vec = _Vec(list(params.keys()), list(params.values()), what="parameter")

    def raw_eq(yp, yc, yl, eps):
        return np.asarray(
            eq_fn(
                _Vec(model.variables, yp),
                _Vec(model.variables, yc),
                _Vec(model.variables, yl),
                _Vec(model.shocks, np.atleast_1d(eps)),
                p_vec,
            ),
            dtype=float,
        )

    prob = StackedProblem(
        raw_eq, [0.0, 0.0], y_star, shocks,
        variable_names=model.variables,
        terminal_policy=term_policy,
        terminal_jacobian=term_jac if with_analytic_jac else None,
    )
    rng = np.random.default_rng(42)
    y_probe = krylov.path.to_numpy().ravel() + 0.03 * rng.standard_normal(prob.n)
    v_dir = rng.standard_normal(prob.n)
    jvp_val = prob.jvp(y_probe)(v_dir)
    eps_fd = 1e-7
    fd_val = (prob.residual(y_probe + eps_fd * v_dir) - prob.residual(y_probe - eps_fd * v_dir)) / (2.0 * eps_fd)
    np.testing.assert_allclose(jvp_val, fd_val, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("case", ["lower", "upper", "two_sided", "pinned", "reordered"])
def test_krylov_mcp_cases_match_direct_and_satisfy_complementarity(case):
    from puremacro.dsge import MCPResult, read_mod, solve_transition

    if case == "reordered":
        # Equation 0 is standard Euler/state, Equation 1 is the MCP equation for variable 0 ('r')
        # with negative orientation (-1.0) so that s_sign * f_1 >= 0 at lower bound r >= 0.
        def reordered_eq(yp, yc, yl, eps):
            r, x = yc[0], yc[1]
            e = np.asarray(eps).ravel()[0]
            eq0 = x - 0.5 * yl[1] + 0.8 * r - e
            eq1 = -(r - (0.4 * x + e))  # negative sign orientation on row 1 for variable 0 ('r')
            return np.asarray([eq0, eq1])

        shocks = np.array([-0.5, -0.4, 0.3, 0.2, 0.0, 0.0])[:, None]
        kwargs = dict(
            y_init=[0.05, 0.0], y_ss=[0.0, 0.0], periods=6, shocks=shocks,
            variable_names=["r", "x"], mcp=True,
            mcp_bounds={"r": (0.0, None)}, mcp_equations={"r": (1, -1.0)}, tol=1e-10,
        )
        direct = solve_perfect_foresight(reordered_eq, linear_solver="direct", **kwargs)
        krylov = solve_perfect_foresight(reordered_eq, linear_solver="krylov", **kwargs)
    else:
        model = read_mod("""
            var r x;
            varexo e;
            parameters rho;
            rho = 0.5;
            model;
                r = 0.3 * x(+1) + e;
                x = rho * x(-1) - 0.5 * r + e;
            end;
            initval; r = 0.0; x = 0.0; e = 0.0; end;
        """)
        if case == "lower":
            bounds = {"r": (-0.05, None)}
            shocks = np.array([-0.25, -0.15, 0.10, 0.05, 0.0, 0.0])[:, None]
        elif case == "upper":
            bounds = {"r": (None, 0.05)}
            shocks = np.array([0.25, 0.15, -0.10, -0.05, 0.0, 0.0])[:, None]
        elif case == "two_sided":
            bounds = {"r": (-0.05, 0.05)}
            shocks = np.array([-0.25, 0.25, -0.15, 0.15, 0.0, 0.0])[:, None]
        else:  # pinned
            bounds = {"r": (0.02, 0.02)}
            shocks = np.array([-0.15, 0.20, 0.0, 0.0, 0.0, 0.0])[:, None]

        direct = solve_transition(
            model, y_init=[0.0, 0.0], y_end=[0.0, 0.0], shocks=shocks,
            continuation=False, mcp=True, mcp_bounds=bounds, mcp_equations={"r": (0, 1.0)},
            linear_solver="direct", tol=1e-10,
        )
        krylov = solve_transition(
            model, y_init=[0.0, 0.0], y_end=[0.0, 0.0], shocks=shocks,
            continuation=False, mcp=True, mcp_bounds=bounds, mcp_equations={"r": (0, 1.0)},
            linear_solver="krylov", tol=1e-10,
        )
        assert direct.converged and krylov.converged
        direct = direct.solution
        krylov = krylov.solution

    assert isinstance(krylov, MCPResult)
    assert direct.converged and krylov.converged
    assert krylov.linear_solver == "krylov"
    np.testing.assert_allclose(krylov.path.to_numpy(), direct.path.to_numpy(), atol=1e-9, rtol=0)
    assert krylov.binding_periods == direct.binding_periods
    assert len(krylov.binding_periods["r"]) >= 1
    assert krylov.physical_mcp_feasibility <= 1e-10
    assert krylov.physical_complementarity_residual <= 1e-10
    assert krylov.solver_diagnostics["has_mcp"] is True
    assert krylov.solver_diagnostics["dense_jvp_blocks"] == 0


def test_krylov_mcp_combined_with_terminal_policy_and_mod_tags():
    from puremacro.dsge import read_mod

    model = read_mod("""
        var r x;
        varexo e;
        parameters floor;
        floor = 0.0;
        model;
            [mcp = 'r > floor']
            r = 0.25 * x(+1) + e;
            x = 0.5 * x(-1) - 0.4 * r + 0.2 * x(+1) + e;
        end;
        initval; r = 0.02; x = 0.0; e = 0.0; end;
    """)
    shocks = np.array([-0.15, -0.10, 0.05, 0.02, 0.0, 0.0])[:, None]
    term_pol = lambda y_last: 0.4 * np.asarray(y_last, dtype=float)
    term_jac = lambda _y_last: 0.4 * np.eye(2)

    direct = solve_perfect_foresight(
        model, y_init=[0.02, 0.0], y_end=[0.0, 0.0], periods=6, shocks=shocks,
        mcp=True, terminal_policy=term_pol, terminal_jacobian=term_jac,
        linear_solver="direct", tol=1e-10,
    )
    krylov = solve_perfect_foresight(
        model, y_init=[0.02, 0.0], y_end=[0.0, 0.0], periods=6, shocks=shocks,
        mcp=True, terminal_policy=term_pol, terminal_jacobian=term_jac,
        linear_solver="krylov", tol=1e-10,
    )
    assert direct.converged and krylov.converged
    np.testing.assert_allclose(krylov.path.to_numpy(), direct.path.to_numpy(), atol=1e-9, rtol=0)
    assert krylov.binding_periods == direct.binding_periods
    assert krylov.solver_diagnostics["has_mcp"] is True
    assert krylov.solver_diagnostics["has_terminal_policy"] is True


def test_krylov_mcp_active_set_invalidation_and_preconditioner_rebuild():
    from puremacro.dsge import read_mod, solve_transition

    # Start from a positive interior initial guess (r = 0.5, active set empty)
    # for a negative shock that drives r to the binding lower bound 0.0,
    # forcing an active-set transition between Newton steps.
    model = read_mod("""
        var r x;
        varexo e;
        parameters floor;
        floor = 0.0;
        model;
            [mcp = 'r > floor']
            r = 0.2 * x(+1) + e;
            x = 0.6 * x(-1) - 0.5 * r + 0.1 * x(+1) ^ 2 + e;
        end;
        initval; r = 0.1; x = 0.0; e = 0.0; end;
    """)
    shocks = np.array([-0.3, -0.25, -0.1, 0.05, 0.0, 0.0])[:, None]
    init_guess = np.full((6, 2), 0.5)
    res = solve_transition(
        model, y_init=[0.1, 0.0], y_end=[0.0, 0.0], shocks=shocks,
        initial_path=init_guess, continuation=False, mcp=True,
        linear_solver="krylov", tol=1e-10,
    )
    assert res.converged
    diag = res.solution.solver_diagnostics
    assert diag["active_set_changes"] >= 1
    assert diag["active_set_invalidations"] >= 1
    assert diag["preconditioner_invalidations"] >= 1
    assert diag["preconditioner_builds"] >= 2
    assert any(h.get("active_set_changed", False) for h in diag["linear_history"])


def test_krylov_constrained_matrix_free_memory_advantage():
    from puremacro.dsge import read_mod

    n, T = 30, 40
    var_names = [f"y_{i}" for i in range(n)]
    eq_lines = [
        "[mcp = 'y_0 > 0'] y_0 = 0.25 * y_1(+1) + e;"
    ]
    for i in range(1, n):
        left = f"y_{i - 1}(-1)"
        right = f"y_{(i + 1) % n}(+1)"
        eq_lines.append(f"y_{i} = 0.45 * y_{i}(-1) + 0.2 * {right} - 0.1 * {left} + 0.02 * e;")
    mod_src = f"""
        var {' '.join(var_names)};
        varexo e;
        model;
            {' '.join(eq_lines)}
        end;
        initval; {' '.join(f'{v}=0;' for v in var_names)} e=0; end;
    """
    model = read_mod(mod_src)
    shocks = np.zeros((T, 1))
    shocks[0, 0] = -0.2
    shocks[1, 0] = -0.1
    term_pol = lambda y_last: 0.3 * np.asarray(y_last, dtype=float)
    term_jac = lambda _y_last: 0.3 * np.eye(n)

    direct = solve_perfect_foresight(
        model, y_init=np.zeros(n), y_end=np.zeros(n), periods=T, shocks=shocks,
        mcp=True, terminal_policy=term_pol, terminal_jacobian=term_jac,
        linear_solver="direct", tol=1e-10,
    )
    krylov = solve_perfect_foresight(
        model, y_init=np.zeros(n), y_end=np.zeros(n), periods=T, shocks=shocks,
        mcp=True, terminal_policy=term_pol, terminal_jacobian=term_jac,
        linear_solver="krylov", tol=1e-10,
    )
    assert direct.converged and krylov.converged
    np.testing.assert_allclose(krylov.path.to_numpy(), direct.path.to_numpy(), atol=1e-9, rtol=0)
    assert krylov.solver_diagnostics["dense_jvp_blocks"] == 0
    direct_matrix_bytes = direct.solver_diagnostics["jacobian_nnz"] * 16  # float64 + int32 indices
    krylov_cache_bytes = krylov.solver_diagnostics["jvp_cache_bytes"]
    assert krylov_cache_bytes < direct_matrix_bytes / 4


def test_supporting_applications_rbc_labor_irreversible_investment_and_nk_zlb():
    """End-to-end regression coverage for RBC endogenous labor, investment irreversibility, and NK ZLB."""
    from puremacro.dsge import read_mod, solve_transition
    from puremacro.validation import audit_result

    # 1. RBC with endogenous labor + positive domain bounds + terminal policy
    rbc_labor = read_mod("""
        var k c l a;
        varexo e;
        parameters beta alpha delta chi rho;
        beta = 0.96;
        alpha = 0.33;
        delta = 0.10;
        chi = 1.0;
        rho = 0.75;
        model;
            1 / c = beta * (1 / c(+1)) * (alpha * exp(a(+1)) * (k ^ (alpha - 1)) * (l(+1) ^ (1 - alpha)) + 1 - delta);
            chi * l = (1 - alpha) * exp(a) * (k(-1) ^ alpha) * (l ^ (-alpha)) / c;
            k = exp(a) * (k(-1) ^ alpha) * (l ^ (1 - alpha)) + (1 - delta) * k(-1) - c;
            a = rho * a(-1) + e;
        end;
        steady_state_model;
            a = 0;
            l = ((1 - alpha) / (chi * (1 - (alpha * beta * delta) / (1 - beta * (1 - delta))))) ^ 0.5;
            k = ((alpha * beta / (1 - beta * (1 - delta))) ^ (1 / (1 - alpha))) * l;
            c = (k ^ alpha) * (l ^ (1 - alpha)) - delta * k;
        end;
    """)
    ss_rbc = rbc_labor._dag.steady_state
    y_ss_rbc = np.array([ss_rbc["k"], ss_rbc["c"], ss_rbc["l"], 0.0], dtype=float)
    y_init_rbc = np.array([0.85 * ss_rbc["k"], ss_rbc["c"], ss_rbc["l"], 0.0], dtype=float)
    shocks_rbc = np.zeros((20, 1))
    shocks_rbc[0, 0] = -0.04
    shocks_rbc[1, 0] = -0.02
    rbc_bounds = {"k": (0.0, None), "c": (0.0, None), "l": (0.0, 1.5)}
    rbc_term = lambda y_last: y_ss_rbc + 0.7 * (np.asarray(y_last, dtype=float) - y_ss_rbc)

    rbc_direct = solve_transition(
        rbc_labor, y_init=y_init_rbc, y_end=y_ss_rbc, shocks=shocks_rbc,
        variable_bounds=rbc_bounds, terminal_policy=rbc_term,
        continuation=False, linear_solver="direct", tol=1e-10,
    )
    rbc_krylov = solve_transition(
        rbc_labor, y_init=y_init_rbc, y_end=y_ss_rbc, shocks=shocks_rbc,
        variable_bounds=rbc_bounds, terminal_policy=rbc_term,
        continuation=False, linear_solver="krylov", tol=1e-10,
    )
    assert rbc_direct.converged and rbc_krylov.converged
    np.testing.assert_allclose(rbc_krylov.path.to_numpy(), rbc_direct.path.to_numpy(), atol=1e-9, rtol=0)
    assert {r.check: r.status for r in audit_result(rbc_krylov.solution).records}["path_equations"] == "passed"

    # 2. Investment irreversibility model (binding lower bound on investment inv >= i_floor)
    irrev_mod = read_mod("""
        var k c inv a;
        varexo e;
        parameters beta alpha delta i_floor rho;
        beta = 0.96;
        alpha = 0.33;
        delta = 0.08;
        i_floor = 0.22;
        rho = 0.70;
        model;
            [mcp = 'inv > i_floor']
            1 / c - beta * (1 / c(+1)) * (alpha * exp(a(+1)) * (k ^ (alpha - 1)) + 1 - delta) = 0;
            k = (1 - delta) * k(-1) + inv;
            c = exp(a) * (k(-1) ^ alpha) - inv;
            a = rho * a(-1) + e;
        end;
        steady_state_model;
            a = 0;
            k = (alpha * beta / (1 - beta * (1 - delta))) ^ (1 / (1 - alpha));
            inv = delta * k;
            c = (k ^ alpha) - inv;
        end;
    """)
    ss_irr = irrev_mod._dag.steady_state
    y_ss_irr = np.array([ss_irr["k"], ss_irr["c"], ss_irr["inv"], 0.0], dtype=float)
    y_init_irr = np.array([1.15 * ss_irr["k"], ss_irr["c"], ss_irr["inv"], 0.0], dtype=float)
    shocks_irr = np.zeros((16, 1))
    shocks_irr[:3, 0] = -0.18

    irr_direct = solve_transition(
        irrev_mod, y_init=y_init_irr, y_end=y_ss_irr, shocks=shocks_irr,
        mcp=True, continuation=False, linear_solver="direct", tol=1e-10,
    )
    irr_krylov = solve_transition(
        irrev_mod, y_init=y_init_irr, y_end=y_ss_irr, shocks=shocks_irr,
        mcp=True, continuation=False, linear_solver="krylov", tol=1e-10,
    )
    assert irr_direct.converged and irr_krylov.converged
    np.testing.assert_allclose(irr_krylov.path.to_numpy(), irr_direct.path.to_numpy(), atol=1e-9, rtol=0)
    assert irr_krylov.solution.binding_periods == irr_direct.solution.binding_periods
    assert len(irr_krylov.solution.binding_periods["inv"]) >= 2
    assert irr_krylov.physical_mcp_feasibility <= 1e-10
    assert irr_krylov.physical_complementarity_residual <= 1e-10

    # 3. New Keynesian lower-bound (ZLB) transition with endogenous liftoff
    nk_zlb = read_mod("""
        var x pi i;
        varexo r_shock;
        parameters beta kappa sigma phi_pi phi_x i_elb;
        beta = 0.99;
        kappa = 0.15;
        sigma = 1.0;
        phi_pi = 1.5;
        phi_x = 0.25;
        i_elb = -0.01;
        model;
            x = x(+1) - (1 / sigma) * (i - pi(+1) - r_shock);
            pi = beta * pi(+1) + kappa * x;
            [mcp = 'i > i_elb']
            i = phi_pi * pi + phi_x * x;
        end;
        initval; x = 0.0; pi = 0.0; i = 0.0; r_shock = 0.0; end;
    """)
    shocks_nk = np.zeros((18, 1))
    shocks_nk[:4, 0] = -0.025
    shocks_nk[4:7, 0] = -0.01

    nk_direct = solve_transition(
        nk_zlb, y_init=[0.0, 0.0, 0.0], y_end=[0.0, 0.0, 0.0], shocks=shocks_nk,
        mcp=True, continuation=False, linear_solver="direct", tol=1e-10,
    )
    nk_krylov = solve_transition(
        nk_zlb, y_init=[0.0, 0.0, 0.0], y_end=[0.0, 0.0, 0.0], shocks=shocks_nk,
        mcp=True, continuation=False, linear_solver="krylov", tol=1e-10,
    )
    assert nk_direct.converged and nk_krylov.converged
    np.testing.assert_allclose(nk_krylov.path.to_numpy(), nk_direct.path.to_numpy(), atol=1e-9, rtol=0)
    assert nk_krylov.solution.binding_periods == nk_direct.solution.binding_periods
    assert len(nk_krylov.solution.binding_periods["i"]) >= 3
    assert nk_krylov.physical_mcp_feasibility <= 1e-10
    assert nk_krylov.physical_complementarity_residual <= 1e-10
    assert nk_krylov.solution.solver_diagnostics["dense_jvp_blocks"] == 0


