"""First-order DSGE and existing dynamic-trade evidence adapters."""
from __future__ import annotations

import numpy as np

from ._audit_report import DerivativeCheck


def audit_dsge(out, result, model):
    from puremacro.dsge.build import LinearModel, _Vec
    from puremacro.dsge.diagnostics import resid
    from .audit import _derivative
    context = result if isinstance(result, LinearModel) else model
    sol = result.solution if isinstance(result, LinearModel) else result
    if "solver" in out.checks:
        out.add("solver", "solver_converged", tuple(sol.eu) == (1, 1), source="klein_existence_uniqueness",
                existence_uniqueness=list(sol.eu), meaning="unique stable first-order decision rule")
        out.add("solver", "solver_residual", float(sol.residual), units="row-scaled equation residual",
                source="klein_solver_record", status="not_checked", reason="recorded stopping diagnostic")
    if not isinstance(context, LinearModel):
        if "equations" in out.checks:
            out.add("equations", "decision_rule_residual", status="not_checked", reason="requires model=LinearModel")
        return
    if "equations" in out.checks:
        n = sol.G.shape[0]
        if context.timing == "dynare":
            # Dynare's public solution is reduced to original state/control
            # variables, while A/B/C retain the augmented lag-state pencil.
            # Rebuild the ORIGINAL lead/current/lag equations from this supplied
            # solution, including a possibly tampered bare KleinSolution.
            variables = list(context.variables)
            states = [variables.index(name) for name in context.states]
            controls = [variables.index(name) for name in context.controls]
            gx = np.empty((len(variables), n))
            gu = np.empty((len(variables), len(context.shocks)))
            gx[states], gx[controls] = sol.G, sol.F
            gu[states], gu[controls] = sol.N, sol.L
            lead, current, lag, shock = (
                context._A_plus, context._A_0, context._A_minus, context._B_u
            )
            scale = np.maximum(np.max(np.abs(lead), axis=1)
                               + np.max(np.abs(current), axis=1)
                               + np.max(np.abs(lag), axis=1), 1e-30)
            if controls:
                omitted_lags = lag[:, controls]
                # A nonzero lag cannot be discarded merely because the reduced
                # rule solves the remaining columns. This is structural coverage,
                # separate from numerical residual tolerance; retain coefficients.
                out.residual(
                    "equations", "unrepresented_lag_terms", omitted_lags, 0.,
                    units="original lag Jacobian coefficients",
                    source="recomputed_dynare_lag_coverage",
                    control_names=list(context.controls),
                    offending_controls=[name for j, name in enumerate(context.controls)
                                        if np.any(omitted_lags[:, j] != 0)],
                    reason="all nonzero lag columns must be represented in the state vector",
                )
            rule = lead @ gx @ sol.G + current @ gx + lag[:, states]
            impact = lead @ gx @ sol.N + current @ gu + shock
            equation_source = "recomputed_dynare_lead_lag_equations"
        else:
            A, B, C = context.A, context.B, context.C
            scale = np.maximum(np.max(np.abs(A), axis=1) + np.max(np.abs(B), axis=1), 1e-30)
            common = A[:, :n] + A[:, n:] @ sol.F
            rule = common @ sol.G - B[:, :n] - B[:, n:] @ sol.F
            impact = common @ sol.N - B[:, n:] @ sol.L - C
            equation_source = "recomputed_klein_equations"
        if rule.size:
            out.residual("equations", "decision_rule_residual", rule / scale[:, None], out.tol["equations"],
                         units="row-scaled equation residual", source=equation_source)
        else:
            out.add("equations", "decision_rule_residual", status="not_checked",
                    reason="no predetermined states; dynamic state block is empty")
        if impact.size:
            out.residual("equations", "shock_loading_residual", impact / scale[:, None], out.tol["equations"],
                         units="row-scaled equation residual", source=equation_source)
        if context._equations is not None or context._dynare_equations is not None:
            residual = resid(context)
            out.residual("equations", "steady_state_residual", residual.to_numpy(), out.tol["equations"],
                         source="dsge.diagnostics.resid", equation_labels=list(residual.index))
        else:
            out.add("equations", "steady_state_residual", status="not_checked",
                    reason="nonlinear equation function is unavailable; no zero fallback used")
    if "derivatives" in out.checks:
        variables, shocks = list(context.variables), list(context.shocks)
        ss = np.asarray(context.steady_state.loc[variables], dtype=float)
        pars = context._params or {}
        par = _Vec(list(pars), list(pars.values()), "parameter")
        zeros = np.zeros(len(ss))
        e0 = np.zeros(len(shocks))
        if context._equations is not None:
            from puremacro.dsge._equations import _steady_state_adapter
            equation = _steady_state_adapter(context._equations)
            fixed_ss = _Vec(variables, ss.copy())
            order = [variables.index(v) for v in (*context.states, *context.controls)]
            inverse = np.argsort(order)

            def levels(x):
                return np.array([ss[j] * np.exp(x[j]) if context.units[name] == "log" else ss[j] + x[j]
                                 for j, name in enumerate(variables)])

            def evaluate(lead, now, shock):
                return np.asarray(equation(_Vec(variables, levels(lead)), _Vec(variables, levels(now)),
                                           _Vec(shocks, shock, "shock"), par, steady_state=fixed_ss))

            experiments = [
                DerivativeCheck("jacobian_lead", lambda x: evaluate(x, zeros, e0), zeros, context.A[:, inverse]),
                DerivativeCheck("jacobian_current", lambda x: evaluate(zeros, x, e0), zeros, -context.B[:, inverse]),
            ]
            if len(shocks):
                experiments.append(DerivativeCheck("jacobian_shocks", lambda x: evaluate(zeros, zeros, x),
                                                   e0, -context.C))
        elif context._dynare_equations is not None:
            from puremacro.dsge._equations import _steady_state_adapter
            dynare = _steady_state_adapter(context._dynare_equations)
            ss_vec = _Vec(variables, ss.copy())

            def eval_dyn(lead, now, lag, shock):
                return np.asarray(dynare(_Vec(variables, lead), _Vec(variables, now), _Vec(variables, lag),
                                         _Vec(shocks, shock, "shock"), par, steady_state=ss_vec))

            experiments = [
                DerivativeCheck("jacobian_lead", lambda x: eval_dyn(x, ss, ss, e0), ss, context._A_plus),
                DerivativeCheck("jacobian_current", lambda x: eval_dyn(ss, x, ss, e0), ss, context._A_0),
                DerivativeCheck("jacobian_lag", lambda x: eval_dyn(ss, ss, x, e0), ss, context._A_minus),
            ]
            if len(shocks):
                experiments.append(DerivativeCheck("jacobian_shocks", lambda x: eval_dyn(ss, ss, ss, x),
                                                   e0, context._B_u))
        else:
            experiments = []
        for experiment in experiments:
            from dataclasses import replace
            _derivative(out, replace(experiment, tolerance=out.tol["derivatives"]))
    out.metadata["dsge_timing"] = context.timing
    out.metadata["dsge_scope"] = "first-order decision rule, deterministic steady state, optional local derivatives"


def audit_trade(out, result, model):
    from puremacro.trade.dynamic._results import DynamicSteadyStateResult, HorizonComparisonResult
    out.metadata["original_metadata"] = result.metadata
    if isinstance(result, HorizonComparisonResult):
        out.metadata.update({"validation_scope": result.validation_scope, "original_status": result.status,
                             "terminal_check": result.terminal_check})
        if "refinement" in out.checks:
            checks = [
                ("horizon_window", result.max_log_difference, result.tolerance, "log difference", True),
                ("horizon_welfare", result.welfare_max_difference_pp, result.welfare_tolerance, "percentage points", True),
                ("horizon_terminal_state", result.terminal_max_log_gap, result.terminal_tolerance,
                 "log difference", result.terminal_check == "state_gap"),
                ("horizon_discounted_wealth", result.max_discounted_wealth, result.discounted_wealth_tolerance,
                 "discounted wealth", result.terminal_check == "discounted_wealth"),
            ]
            for name, value, tolerance, units, selected in checks:
                out.add("refinement", name, value, tolerance, units, "trade.dynamic.compare_horizons_record",
                        status=None if selected else "not_checked", selected_terminal_mode=result.terminal_check)
        return
    if "solver" in out.checks:
        out.add("solver", "solver_converged", bool(result.converged), source="trade_solver_record")
        out.add("solver", "solver_residual", float(result.max_residual), float(result.tol),
                "scaled equation residual", "trade_solver_record", iterations=result.iterations)
    if "equations" in out.checks:
        for name, value in sorted(result.certificate.items()):
            out.add("equations", f"certificate:{name}", float(value), max(1e-8, result.tol),
                    "certificate normalized residual", "trade.dynamic.recorded_independent_certificate",
                    reason="certificate at solve time; not recomputed evidence about a modified result")
        if isinstance(result, DynamicSteadyStateResult) and model is not None:
            try:
                rebuilt = model.certificate(result.z, result.z, result.z, result.policy, result.policy)
                for name, value in sorted(rebuilt.items()):
                    out.add("equations", f"recomputed_certificate:{name}", float(value), max(1e-8, result.tol),
                            "certificate normalized residual", "trade.dynamic.recomputed_independent_certificate",
                            tolerance_scope="retains original trade certificate floor max(1e-8, result.tol)")
            except Exception as exc:
                out.add("equations", "recomputed_certificate", status="failed",
                        source="trade.dynamic.recomputed_independent_certificate", error=f"{type(exc).__name__}: {exc}")
        else:
            out.add("equations", "recomputed_certificate", status="not_checked",
                    reason="steady-state recomputation requires model=DynamicEconomy; transition adapter preserves recorded certificate")
