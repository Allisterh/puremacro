"""Stationary EGM audits evaluated at the supplied continuous policies."""
from __future__ import annotations

from copy import copy
import numpy as np


def _kkt(gradient, policy, grid):
    """Normalized maximization derivative, projected at actual grid bounds.

    At a lower bound g <= 0; at an artificial upper bound g >= 0. The
    classification tolerance follows the *local* cell width, including narrow
    grids, rather than declaring all choices within a fixed 1e-8 to be bound.
    """
    eps = 32 * np.finfo(float).eps * max(1., float(np.max(np.abs(grid))))
    lo_eps = min(eps, (grid[1] - grid[0]) * 1e-6)
    hi_eps = min(eps, (grid[-1] - grid[-2]) * 1e-6)
    return np.where(policy <= grid[0] + lo_eps, np.maximum(gradient, 0),
                    np.where(policy >= grid[-1] - hi_eps, np.minimum(gradient, 0), gradient))


def _distribution(out, mu, shape, push, grids):
    if "distribution" not in out.checks:
        return
    if mu is None:
        out.add("distribution", "distribution", status="not_checked", reason="no distribution supplied")
        return
    dist = np.asarray(mu, dtype=float)
    if dist.shape != shape:
        out.add("distribution", "distribution_shape", False, source="supplied_distribution",
                expected=shape, actual=dist.shape)
        return
    out.add("distribution", "distribution_finite", bool(np.all(np.isfinite(dist))), source="supplied_distribution")
    out.add("distribution", "mass_error", float(abs(dist.sum() - 1)), out.tol["mass"],
            "probability", "supplied_distribution")
    out.add("distribution", "negative_mass", float(max(0., -np.min(dist))), out.tol["negative_mass"],
            "probability", "supplied_distribution", minimum_probability=float(np.min(dist)))
    minimum = float(np.min(dist))
    out.add("distribution", "minimum_probability", minimum, -out.tol["negative_mass"],
            "probability", "supplied_distribution",
            status="passed" if np.isfinite(minimum) and minimum >= -out.tol["negative_mass"] else "failed",
            comparison="value >= tolerance")
    out.residual("distribution", "stationarity_residual", push(dist) - dist, out.tol["stationarity"],
                 units="probability per transition", source="recomputed_policy_lottery")
    for axis, (name, grid) in enumerate(grids):
        mass = float(np.take(dist, -1, axis=axis).sum())
        out.add("distribution", f"mass_at_{name}_max", mass, out.tol["boundary_mass"],
                "probability", "supplied_distribution", boundary=float(grid[-1]), artificial_upper_bound=True)
        out.add("distribution", f"mass_at_{name}_min", float(np.take(dist, 0, axis=axis).sum()),
                units="probability", source="supplied_distribution", status="not_checked",
                reason="reported borrowing-bound mass; no accuracy threshold", boundary=float(grid[0]))


def audit_dp(out, result, model):
    from puremacro.dp._egm import EGMHouseholdSolution, EGMPlan
    from puremacro.dp._egm_two_asset import TwoAssetHouseholdSolution
    from puremacro.dp._expr import ModelSpecError
    raw = result.raw
    if "solver" in out.checks:
        converged = getattr(raw, "converged", None)
        # One-asset EGM returns only after convergence, but does not store a flag.
        if converged is None and isinstance(raw, EGMHouseholdSolution) and raw.egm is not None:
            converged = True
        out.add("solver", "solver_converged", converged, source="solver_record",
                status="not_checked" if converged is None else None,
                meaning="solver termination declaration, not an equation-accuracy certificate")
        if hasattr(raw, "sup_norm"):
            out.add("solver", "policy_iteration_residual", float(raw.sup_norm),
                    units="policy units", source="solver_record", status="not_checked",
                    iterations=int(raw.n_iter), reason="reported stopping residual; independent Euler tolerance below")
    if result.method != "egm" or result._c.finite or result._c.model._discrete is not None:
        return
    if not isinstance(raw, (EGMHouseholdSolution, TwoAssetHouseholdSolution)):
        return
    if not out.checks.intersection({"equations", "distribution"}):
        return
    try:
        # TwoAssetEGMPlan installs hooks: use an isolated compiled shell.
        comp = copy(result._c if model is None else model.check())
        plan = EGMPlan(comp)
        params = dict(result.params) if model is None else comp.param_values()
        if isinstance(raw, TwoAssetHouseholdSolution):
            _two_asset(out, result, plan, params)
        else:
            _one_asset(out, result, plan, params)
    except (ModelSpecError, ValueError, IndexError, FloatingPointError) as exc:
        if "equations" in out.checks:
            out.add("equations", "model_specification", status="failed", source="declared_model",
                    error=f"{type(exc).__name__}: {exc}")


def _common(out, raw, declared_c, declared_u, policies, coordinates, dist):
    if "equations" not in out.checks:
        return
    finite = all(np.all(np.isfinite(getattr(raw, name)))
                 for name in ("V", "c", "Va", "Vb", "chi", "d") if hasattr(raw, name))
    finite = finite and all(np.all(np.isfinite(p)) for _, p, _ in policies)
    out.add("equations", "finite_values", bool(finite), source="supplied_policies_and_value")
    out.add("equations", "positive_consumption", bool(np.all(raw.c > 0)), source="supplied_consumption")
    out.add("equations", "declared_feasibility", bool(np.all(np.isfinite(declared_u))),
            source="declared_reward_and_constraints")
    out.residual("equations", "budget_residual", raw.c - declared_c, out.tol["equations"],
                 source="recomputed_declared_budget", coordinates=coordinates)
    for name, policy, grid in policies:
        violation = np.maximum(np.maximum(grid[0] - policy, policy - grid[-1]), 0.)
        out.residual("equations", f"bounds_{name}", violation, out.tol["equations"],
                     source="declared_grid_bounds", coordinates=coordinates)
        out.add("equations", f"policy_hits_{name}_max", float(np.mean(policy >= grid[-1])),
                units="fraction of grid states", source="supplied_policy", status="not_checked",
                reason="reported artificial upper-bound incidence; inspect with boundary mass and refinement")


def _one_asset(out, result, plan, params):
    from puremacro.vfi.distribution import lottery_push
    raw, comp = result.raw, plan.comp
    a, P, ap, c = plan.a, comp.P, raw.aprime, raw.c
    shape = (len(a), P.shape[0])
    if c.shape != shape or ap.shape != shape:
        raise ValueError(f"policy shape must be {shape}")
    R, y = plan.budget(params)
    coordinates = [(comp.state_names[0], a), ("shock_index", np.arange(P.shape[0]))]
    declared_c = plan._call(plan._c_fn, ap, a[:, None], 2, params, None)
    declared_u = plan._call(comp.household, ap, a[:, None], 2, params, None)
    _common(out, raw, declared_c, declared_u, [(comp.state_names[0], ap, a)], coordinates, result.distribution)
    if "equations" in out.checks:
        _, marginal, _ = plan._u_funcs(params, float(np.max(R * a[-1] + y) - a[0]))
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            uc = marginal(c)
            next_marginal = np.asarray(R) * uc
            expected = np.zeros(shape)
            for z in range(P.shape[0]):
                for zp in range(P.shape[0]):
                    expected[:, z] += P[z, zp] * np.interp(ap[:, z], a, next_marginal[:, zp])
            beta = float(params[f"p_{comp.model._beta}"])
            gradient = (beta * expected - uc) / uc
        out.residual("equations", "euler_kkt_a", _kkt(gradient, ap, a), out.tol["euler"],
                     units="current marginal utility", source="recomputed_continuous_policy_euler",
                     weights=result.distribution, coordinates=coordinates,
                     interpolation="linear interpolation of next-state marginal utility; next-state return",
                     budget_scope="declared affine budget checked on compiler probes; not a proof for arbitrary registered functions",
                     upper_bound="artificial grid truncation")
    _distribution(out, result.distribution, shape, lambda mu: lottery_push(mu, ap, a, P),
                  [(comp.state_names[0], a)])
    out.metadata["dp_grid_order"] = [comp.state_names[0], "shock_index"]


def _two_asset(out, result, plan, params):
    from puremacro.models.hank_sequence_space import (
        _two_asset_cost, _two_asset_expectation, _two_asset_forward,
    )
    raw = result.raw
    a, b, s, P = plan.a_grid, plan.b_grid, plan.s_grid, plan.pi_s
    p = plan._extract_params(params)
    shape = (len(a), len(b), len(s))
    c, ap, bp = raw.c, raw.aprime, raw.bprime
    if any(arr.shape != shape for arr in (c, ap, bp, raw.Va, raw.Vb)):
        raise ValueError(f"two-asset policy shape must be {shape}")
    cur_a, cur_b, cur_s = a[:, None, None], b[None, :, None], s[None, None, :]
    coordinates = [(plan.illiquid_name, a), (plan.liquid_name, b), ("productivity", s)]
    declared_c = plan._call(plan._c_fn, ap, bp, cur_a, cur_b, cur_s, params)
    declared_u = plan._call(plan.comp.household, ap, bp, cur_a, cur_b, cur_s, params)
    _common(out, raw, declared_c, declared_u,
            [(plan.illiquid_name, ap, a), (plan.liquid_name, bp, b)], coordinates, result.distribution)
    if "equations" in out.checks:
        cost, chi_d, chi_a, _ = _two_asset_cost(ap, cur_a, p["r_a"], p["chi_0"], p["chi_1"], p["a_bar"])
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            uc = c ** (-p["gamma"])
            va, vb = (1 + p["r_a"] - chi_a) * uc, (1 + p["r_b"]) * uc
            ea = p["beta"] * _two_asset_expectation(va, ap, bp, a, b, P)
            eb = p["beta"] * _two_asset_expectation(vb, ap, bp, a, b, P)
            ga, gb = (ea - (1 + chi_d) * uc) / uc, (eb - uc) / uc
        for name, gradient, policy, grid in (("a", ga, ap, a), ("b", gb, bp, b)):
            out.residual("equations", f"euler_kkt_{name}", _kkt(gradient, policy, grid), out.tol["euler"],
                         units="current marginal utility", source="recomputed_envelope_and_continuous_policy_euler",
                         weights=result.distribution, coordinates=coordinates,
                         interpolation="bilinear asset lottery; next productivity probabilities",
                         upper_bound="artificial grid truncation", asset_role="illiquid" if name == "a" else "liquid")
        for name, supplied, recomputed in (("a", raw.Va, va), ("b", raw.Vb, vb)):
            out.residual("equations", f"marginal_value_{name}", (supplied - recomputed) / uc,
                         out.tol["equations"], source="recomputed_envelope", units="current marginal utility")
        out.residual("equations", "adjustment_cost", raw.chi - cost, out.tol["equations"],
                     source="recomputed_canonical_adjustment_cost")
        out.residual("equations", "deposit_timing", raw.d - (ap - (1 + p["r_a"]) * cur_a),
                     out.tol["equations"], source="recomputed_current_asset_return_timing")
    _distribution(out, result.distribution, shape,
                  lambda mu: _two_asset_forward(mu, ap, bp, a, b, P),
                  [(plan.illiquid_name, a), (plan.liquid_name, b)])
    out.metadata["dp_grid_order"] = [plan.illiquid_name, plan.liquid_name, "productivity"]
