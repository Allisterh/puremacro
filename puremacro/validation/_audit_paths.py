"""Re-evaluate supplied transition paths without solving for replacement paths."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict

import numpy as np

from ._audit_report import PathAuditContext


def _array(value, shape, name):
    arr = np.array(value, dtype=float, copy=True)
    if arr.shape != shape or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} requires finite shape {shape}, got {arr.shape}")
    return arr


def audit_hank_model(out, result, model):
    """Editable closure equations; no hidden nonlinear household response solve."""
    if "solver" in out.checks:
        out.add("solver", "solver_converged", bool(result.converged), source="hank_model_solver_record")
    if "distribution" in out.checks:
        for name in ("asset_distribution", "joint_distribution"):
            value = getattr(result, name, None)
            if value is None:
                continue
            distribution = np.asarray(value, dtype=float)
            out.residual("distribution", f"{name}:mass", np.sum(distribution) - 1,
                         out.tol["mass"], source="recomputed_retained_stationary_distribution_mass")
            out.residual("distribution", f"{name}:negative_mass", np.minimum(distribution, 0),
                         out.tol["negative_mass"], source="recomputed_retained_stationary_distribution_nonnegativity")
        out.add("distribution", "distribution_transport", status="not_checked",
                reason="closure result retains stationary summaries, not transition policies and distributions")
    if "equations" not in out.checks:
        return
    context = getattr(result, "_audit_context", None) if model is None else model
    if not isinstance(context, Mapping):
        out.add("equations", "hank_model_context", status="not_checked" if model is None else "unsupported",
                reason="editable HANK closure requires retained context or an explicit context mapping")
        return
    mode = context.get("mode")
    if mode not in ("hank_model_linear", "hank_model_nonlinear"):
        out.add("equations", "hank_model_context", status="unsupported", reason="unknown HANK closure context mode")
        return
    try:
        T = result.horizon
        if T != context["horizon"] or not isinstance(T, (int, np.integer)) or T < 1:
            raise ValueError("reported horizon differs from retained closure context")
        names = tuple(context["variable_names"])
        shocks = context["shock_paths"]
        candidate = {name: _array(result.transition_paths[name], (T,), name)
                     for name in (*names, *shocks)}
        for name, expected in shocks.items():
            out.residual("equations", f"shock_context:{name}", candidate[name] - _array(expected, (T,), name),
                         out.tol["equations"], source="recomputed_path_vs_retained_shock_context")
        if mode == "hank_model_linear":
            aggregate, household = context["residuals"](candidate)
            out.residual("equations", "aggregate_equations", aggregate, out.tol["equations"],
                         source="recomputed_original_hank_aggregate_equations", boundary="zero deviations outside horizon")
            out.residual("equations", "household_consistency", household, out.tol["equations"],
                         source="recomputed_retained_sequence_jacobian_response", meaning="linear household approximation")
        else:
            aggregate = context["aggregate_residuals_fn"](candidate)
            out.residual("equations", "aggregate_equations", aggregate, out.tol["equations"],
                         source="recomputed_original_hank_aggregate_equations", boundary="zero deviations outside horizon")
            unchanged = True
            for inp, variable in context["household_inputs"].items():
                retained = _array(context["household_input_paths"][inp], (T,), inp)
                error = candidate[variable] - retained
                out.residual("equations", f"household_input_context:{inp}", error, out.tol["equations"],
                             source="recomputed_path_vs_retained_household_inputs")
                identical = bool(np.array_equal(candidate[variable], retained))
                out.add("equations", f"household_input_identity:{inp}", identical,
                        source="recomputed_exact_household_input_identity",
                        reason="retained nonlinear response conditions on identical inputs, independent of equation tolerance")
                unchanged = unchanged and identical
            if unchanged:
                errors = np.array([candidate[variable] - _array(context["household_output_paths"][output], (T,), output)
                                   for output, variable in context["household_outputs"].items()])
                out.residual("equations", "household_consistency", errors, out.tol["equations"],
                             source="recomputed_path_vs_retained_nonlinear_household_outputs",
                             meaning="conditional on unchanged household input paths; no household solve rerun")
            else:
                out.add("equations", "household_consistency", status="not_checked",
                        reason="household inputs changed; retained nonlinear response no longer certifies this path")
        out.add("equations", "household_euler_kkt", status="not_checked",
                reason="editable closure context does not retain micro policies and marginal values")
        out.add("equations", "household_budget", status="not_checked",
                reason="editable closure context does not retain household policy paths")
        out.metadata["hank_model_context"] = {"mode": mode, "periods": T,
                                             "variable_names": list(names), "hidden_solves": 0}
    except Exception as exc:
        out.add("equations", "hank_model_equations", status="failed",
                source="recomputed_hank_model_context", error=f"{type(exc).__name__}: {exc}")


def audit_perfect_foresight(out, result, model):
    if "solver" in out.checks:
        out.add("solver", "solver_converged", bool(result.converged), source="perfect_foresight_solver_record")
        out.add("solver", "solver_residual", float(result.residual_norm), status="not_checked",
                source="perfect_foresight_solver_record", reason="recorded stopping diagnostic")
    if "equations" not in out.checks:
        return
    context = asdict(model) if isinstance(model, PathAuditContext) else getattr(result, "_audit_context", None)
    source = "caller_path_context" if isinstance(model, PathAuditContext) else "retained_solve_time_context"
    if model is not None and not isinstance(model, PathAuditContext):
        out.add("equations", "path_context", status="unsupported",
                reason="model override must be a PathAuditContext with explicit boundaries and shocks")
        return
    if not isinstance(context, Mapping):
        out.add("equations", "path_equations", status="not_checked",
                reason="missing solve-time context; provide model=PathAuditContext; surprise paths are not a single perfect-foresight experiment")
        return
    if context.get("mode") != "perfect_foresight":
        out.add("equations", "path_equations", status="unsupported", source=source,
                reason="only one announced perfect-foresight path is supported", mode=context.get("mode"))
        return
    try:
        from puremacro.dsge._equations import _steady_state_adapter

        path = np.array(result.path, dtype=float, copy=True)
        if path.ndim != 2 or not all(path.shape) or not np.all(np.isfinite(path)):
            raise ValueError("path must be a finite nonempty two-dimensional array")
        periods, n = path.shape
        names = tuple(context.get("variable_names", ())) or tuple(result.variable_names)
        if names and (len(names) != n or tuple(result.path.columns) != names):
            raise ValueError("path columns must match context variable order")
        initial = _array(context["y_init"], (n,), "y_init")
        terminal = _array(context["y_end"], (n,), "y_end")
        shocks = np.array(context["exogenous_path"], dtype=float, copy=True)
        if shocks.ndim not in (1, 2) or shocks.shape[0] != periods or not np.all(np.isfinite(shocks)):
            raise ValueError("exogenous_path must be finite with exactly one row per reported date")
        equation = _steady_state_adapter(context["equations"])
        terminal_lead = terminal.copy()
        terminal_policy = context.get("terminal_policy")
        if terminal_policy is not None:
            if not callable(terminal_policy):
                raise ValueError("terminal_policy must be callable")
            reference_input = _array(context["terminal_policy_input"], (n,), "terminal_policy_input")
            reference_output = _array(context["terminal_policy_output"], (n,), "terminal_policy_output")
            replayed = _array(terminal_policy(reference_input.copy()), (n,), "terminal_policy replay")
            if not np.array_equal(replayed, reference_output):
                raise ValueError("terminal_policy changed since the solve; retained reference no longer matches")
            terminal_lead = _array(terminal_policy(path[-1].copy()), (n,), "terminal_policy output")
        full = np.vstack((initial, path, terminal_lead))
        errors = np.empty_like(path)
        for t in range(periods):
            value = equation(full[t + 2].copy(), full[t + 1].copy(), full[t].copy(),
                             shocks[t].copy(), steady_state=terminal.copy())
            errors[t] = _array(value, (n,), f"equations at period {t + 1}")
        out.residual("equations", "path_equations", errors, out.tol["equations"],
                     source="recomputed_perfect_foresight_equations", variable_names=list(names),
                     coordinates=(("period", np.arange(1, periods + 1)), ("equation", np.arange(n))),
                     context_source=source, steady_state_reference=terminal)
        out.add("equations", "terminal_distance", float(np.max(np.abs(path[-1] - terminal))),
                status="not_checked", units="variable units", source="recomputed_terminal_distance",
                reason="distance at T from the boundary at T+1; requires a chosen horizon accuracy criterion")
        out.metadata["path_context"] = {"source": source, "mode": "perfect_foresight",
                                        "periods": periods, "variable_names": list(names),
                                        "initial_boundary": initial, "terminal_boundary": terminal}
    except Exception as exc:
        out.add("equations", "path_equations", status="failed", source=source,
                error=f"{type(exc).__name__}: {exc}")


def audit_nonlinear_hank(out, result, model):
    if "solver" in out.checks:
        out.add("solver", "solver_converged", bool(result.converged), source="nonlinear_hank_solver_record")
        out.residual("solver", "solver_residual", result.residuals, float(result.tol),
                     source="nonlinear_hank_solver_record", meaning="recorded solve-time residual only")
    if not out.checks.intersection(("equations", "distribution")):
        return
    context = getattr(result, "_audit_context", None) if model is None else model
    if context is not None and not isinstance(context, Mapping):
        for category in sorted(out.checks.intersection(("equations", "distribution"))):
            out.add(category, "hank_context", status="unsupported",
                    reason="a HANK model override must be an explicit retained-context mapping")
        return
    try:
        T = result.horizon
        if not isinstance(T, (int, np.integer)) or T < 1:
            raise ValueError("horizon must be a positive integer")
        output = _array(result.U, (T,), "U")
        consumption = _array(result.irf_consumption_nonlinear, (T,), "consumption path")
        inflation = _array(result.irf_inflation_nonlinear, (T,), "inflation path")
        rates = _array(result.irf_rate_nonlinear, (T,), "ex-ante rate path")
        shock = _array(result.shock_seq, (T,), "shock path")
        monetary = result.shock_var in ("r", "monetary", "interest_rate", "rate")
        fiscal = result.shock_var in ("g", "fiscal", "spending", "transfer")
        if not monetary and not fiscal:
            raise ValueError("unknown HANK shock kind")
        government = np.zeros(T) if monetary else shock
        if "equations" in out.checks:
            out.residual("equations", "goods_market", output - consumption - government,
                         out.tol["equations"], source="recomputed_reported_aggregate_paths")
            for name in ("nonlinear_path", "irf_output_nonlinear"):
                out.residual("equations", f"output_consistency:{name}",
                             _array(getattr(result, name), (T,), name) - output,
                             out.tol["equations"], source="recomputed_reported_path_consistency")
            if context is not None:
                phi, kappa, beta = (float(context[k]) for k in ("phi_pi", "kappa", "beta"))
                if not all(np.isfinite(v) for v in (phi, kappa, beta)):
                    raise ValueError("GE parameters must be finite")
                lead_pi = np.r_[inflation[1:], 0.]
                out.residual("equations", "nkpc", inflation - beta * lead_pi - kappa * output,
                             out.tol["equations"], source="recomputed_hank_nkpc", terminal_inflation=0.)
                out.residual("equations", "policy_rule", rates - phi * inflation + lead_pi
                             - (shock if monetary else 0.), out.tol["equations"],
                             source="recomputed_hank_taylor_fisher_rule", timing="ex-ante real rate")
            else:
                out.add("equations", "ge_parameters", status="not_checked",
                        reason="effective GE overrides were not retained on this legacy result")
        if context is None or context.get("paths") is None or context.get("household") is None:
            if "equations" in out.checks:
                out.add("equations", "household_budget", status="not_checked",
                        reason="solve with retain_audit_paths=True to retain the final household pass")
            if "distribution" in out.checks:
                out.add("distribution", "distribution_transport", status="not_checked",
                        reason="per-date distributions and household policies were not retained")
            return
        _hank_household(out, result, context, output, consumption, rates, government)
    except Exception as exc:
        for category in sorted(out.checks.intersection(("equations", "distribution"))):
            out.add(category, "hank_path_context", status="failed", source="retained_hank_paths",
                    error=f"{type(exc).__name__}: {exc}")


def _hank_household(out, result, context, output, consumption, rates, government):
    from puremacro.models.hank_sequence_space import _forward_step
    from ._audit_dp import _kkt

    paths, hh = context["paths"], context["household"]
    T = result.horizon
    grid = np.array(hh["asset_grid"], dtype=float, copy=True)
    productivity = np.array(hh["productivity"], dtype=float, copy=True)
    if (grid.ndim != 1 or len(grid) < 2 or not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0)
            or productivity.ndim != 1 or not len(productivity) or not np.all(np.isfinite(productivity))):
        raise ValueError("household state grids must be finite, with strictly increasing asset points")
    shape = (len(grid), len(productivity))
    transition = _array(hh["transition"], (shape[1], shape[1]), "productivity transition")
    if np.any(transition < 0) or np.max(np.abs(transition.sum(axis=1) - 1)) > out.tol["mass"]:
        raise ValueError("productivity transition must be nonnegative and row stochastic")
    c = _array(paths["consumption"], (T, *shape), "consumption policies")
    assets = _array(paths["assets"], (T, *shape), "asset policies")
    dist = _array(paths["distributions"], (T + 1, *shape), "distributions")
    rr = _array(paths["realized_returns"], (T + 1,), "realized returns")
    wages = _array(paths["gross_wages"], (T,), "gross wages")
    initial = _array(hh["initial_distribution"], shape, "initial distribution")
    beta, gamma, debt, labour, rss, wss, css = (
        float(value) for value in (context["beta"], hh["gamma"], hh["debt"], hh["labour"],
                                   context["r_ss"], context["w_ss"], context["C_ss"]))
    if not all(np.isfinite(v) for v in (beta, gamma, debt, labour, rss, wss, css)) or min(labour, css, gamma) <= 0:
        raise ValueError("household parameters must be finite, with positive labour, consumption and curvature")
    aggregate_c = np.sum(dist[:-1] * c, axis=(1, 2))
    axis = (("date", np.arange(T)), ("assets", grid), ("productivity", productivity))
    if "equations" in out.checks:
        out.residual("equations", "return_timing", rr - np.r_[rss, rss + rates], out.tol["equations"],
                     source="recomputed_ex_ante_to_realized_timing", units="return per period")
        out.residual("equations", "wage_rule", wages - wss * (1 + output / css), out.tol["equations"],
                     source="recomputed_hank_wage_rule", reason="detects activation of the solver's wage floor")
        net_wages = wages - rr[:-1] * debt / labour
        cash = (1 + rr[:-1, None, None]) * grid[None, :, None] + net_wages[:, None, None] * productivity
        out.residual("equations", "household_budget", c + assets - cash, out.tol["equations"],
                     source="recomputed_household_resource_constraint", coordinates=axis)
        out.residual("equations", "asset_bounds", np.maximum(grid[0] - assets, np.maximum(assets - grid[-1], 0.)),
                     out.tol["equations"], source="supplied_household_policies", coordinates=axis,
                     reason="upper bound is numerical truncation, not an economic constraint")
        out.add("equations", "positive_consumption", bool(np.all(c > 0)), source="supplied_household_policies")
        out.add("equations", "positive_gross_returns", bool(np.all(1 + rr > 0)), source="supplied_household_inputs")
        out.residual("equations", "consumption_aggregation", aggregate_c - (css + consumption),
                     out.tol["equations"], source="recomputed_distribution_weighted_consumption")
        out.residual("equations", "household_goods_market", css + output - aggregate_c - government,
                     out.tol["equations"], source="recomputed_household_aggregate_equilibrium")
        terminal_c = _array(paths["terminal_consumption"], shape, "terminal consumption")
        out.add("equations", "positive_terminal_consumption", bool(np.all(terminal_c > 0)),
                source="supplied_terminal_household_policy")
        if np.all(c > 0) and np.all(terminal_c > 0) and np.all(1 + rr > 0):
            following_c = np.concatenate((c[1:], terminal_c[None]))
            marginal = (1 + rr[1:, None, None]) * following_c ** (-gamma)
            expectation = marginal @ transition.T
            errors = np.empty_like(c)
            for t in range(T):
                expected = np.column_stack([np.interp(assets[t, :, j], grid, expectation[t, :, j])
                                            for j in range(shape[1])])
                gradient = beta * expected / c[t] ** (-gamma) - 1.
                errors[t] = _kkt(gradient, assets[t], grid)
            out.residual("equations", "household_euler_kkt", errors, out.tol["euler"],
                         units="current marginal utility", source="recomputed_continuous_policy_expectations",
                         coordinates=axis, weights=dist[:-1] / T,
                         weighting="equal dates, within-date supplied probability mass")
        else:
            out.add("equations", "household_euler_kkt", status="not_checked",
                    reason="Euler checks require positive consumption and gross returns")
    if "distribution" in out.checks:
        out.residual("distribution", "path_mass_error", dist.sum(axis=(1, 2)) - 1., out.tol["mass"],
                     source="supplied_distribution_path", units="probability")
        out.residual("distribution", "path_negative_mass", np.maximum(-dist, 0.), out.tol["negative_mass"],
                     source="supplied_distribution_path", units="probability")
        out.residual("distribution", "initial_distribution", dist[0] - initial, out.tol["mass"],
                     source="retained_initial_distribution", units="probability")
        transport = np.array([_forward_step(dist[t], assets[t], grid, transition) - dist[t + 1]
                              for t in range(T)])
        out.residual("distribution", "distribution_transport", transport, out.tol["stationarity"],
                     source="recomputed_policy_lottery", units="probability per transition", coordinates=axis)
        out.residual("distribution", "asset_lottery_accounting",
                     np.sum(dist[1:] * grid[None, :, None], axis=(1, 2))
                     - np.sum(dist[:-1] * assets, axis=(1, 2)), out.tol["equations"],
                     source="recomputed_lottery_asset_means", units="asset units")
        out.residual("distribution", "upper_boundary_mass", dist[:, -1].sum(axis=1), out.tol["boundary_mass"],
                     source="supplied_distribution_path", units="probability")
    out.metadata["hank_path_scope"] = {"household": "one-asset EGM with balanced interest tax",
                                        "dates": T, "distribution_dates": T + 1,
                                        "timing": "ex-ante rates realized one date later",
                                        "path_arrays_retained": True}
