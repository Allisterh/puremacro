"""Two-Asset Endogenous Grid Method (EGM) Plan for puremacro.dp.

Compiles and solves two-asset household dynamic programming models with
one liquid asset (e.g. bonds b with gross return 1 + r_b) and one illiquid
asset (e.g. capital/equity a with return 1 + r_a and convex adjustment cost
chi(a', a)).

Supports:
- Declarative models declared via puremacro.dp.Model with two continuous states.
- Vectorized backward EGM step on marginal values (V_a, V_b).
- Stationary joint distribution mu(a, b, s) via 2D bilinear lottery matrix.
- Evaluation of arbitrary model locals, choices, and aggregates on the (a, b, s) grid.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp

from puremacro.dp._expr import ModelSpecError, parse
from puremacro.models.hank_sequence_space import (
    _TA_C_FLOOR,
    _TA_EGM_MAX_ITER,
    _TA_EGM_TOL,
    _build_transition_matrix_2d,
    _stationary_distribution_2d,
    _two_asset_cost,
    _two_asset_floored_states,
    _two_asset_step,
    _TwoAssetHouseholdBlock,
)


@dataclass
class TwoAssetHouseholdSolution:
    """Raw solution of a two-asset household dynamic program."""

    V: np.ndarray             # (n_a, n_b, n_s) value function
    Va: np.ndarray            # (n_a, n_b, n_s) marginal value of illiquid asset
    Vb: np.ndarray            # (n_a, n_b, n_s) marginal value of liquid asset
    c: np.ndarray             # (n_a, n_b, n_s) consumption policy
    aprime: np.ndarray        # (n_a, n_b, n_s) next illiquid asset policy
    bprime: np.ndarray        # (n_a, n_b, n_s) next liquid asset policy
    d: np.ndarray             # (n_a, n_b, n_s) net deposit into illiquid asset
    chi: np.ndarray           # (n_a, n_b, n_s) adjustment cost
    n_iter: int
    sup_norm: float
    r_a: float                # return on illiquid asset
    r_b: float                # return on liquid asset
    w: float                  # non-financial wage / income
    converged: bool = True
    hh_block: _TwoAssetHouseholdBlock | None = field(default=None, repr=False)
    policy_d: np.ndarray | None = None
    choice_prob: np.ndarray | None = None


class TwoAssetEGMPlan:
    """Validated Two-Asset EGM Plan for a two-state ``_Compiled`` model."""

    def __init__(self, comp: Any):
        m = comp.model
        if len(comp.state_names) != 2:
            raise ModelSpecError(
                f"TwoAssetEGMPlan requires exactly two continuous states; got {comp.state_names}"
            )
        if comp.finite or m._discrete is not None or m._prices or len(comp.z_grids) > 1:
            raise ModelSpecError(
                "two-asset EGM supports an infinite horizon, no discrete choice or "
                "equilibrium prices, and at most one productivity shock"
            )
        self.comp = comp
        from puremacro.dp._two_asset_spec import recognize
        self.cname = self._find_consumption_local()
        illiquid_name, liquid_name, self.parameter_bindings = recognize(comp, self.cname)

        self.illiquid_name = illiquid_name
        self.liquid_name = liquid_name
        self.illiquid_idx = comp.state_names.index(illiquid_name)
        self.liquid_idx = comp.state_names.index(liquid_name)

        self.a_grid = np.asarray(comp.state_grids[self.illiquid_idx], dtype=float)
        self.b_grid = np.asarray(comp.state_grids[self.liquid_idx], dtype=float)
        self.n_a = len(self.a_grid)
        self.n_b = len(self.b_grid)

        # Idiosyncratic productivity shock
        if comp.z_grids:
            self.s_grid = np.asarray(comp.z_grids[0], dtype=float)
        else:
            self.s_grid = np.ones(1, dtype=float)
        self.n_s = len(self.s_grid)
        self.pi_s = np.asarray(comp.P, dtype=float)
        self.n_z = self.n_s

        # Look for consumption local
        self.cname = self._find_consumption_local()
        positive = (parse(f"{self.cname} > 0"), parse(f"0 < {self.cname}"))
        if any(node not in positive for node in comp.constraints):
            raise ModelSpecError("two-asset EGM supports only the explicit consumption > 0 constraint")
        from puremacro.dsge._ast import Call, Const, Param

        reward = comp.reward
        if not (isinstance(reward, Call) and reward.args and reward.args[0] == Param(self.cname)
                and ((reward.func.lower() in ("log", "ln") and len(reward.args) == 1)
                     or (reward.func.lower() == "crra" and len(reward.args) == 2
                         and isinstance(reward.args[1], (Const, Param))))):
            raise ModelSpecError("two-asset EGM requires reward crra(consumption, gamma) or log(consumption)")
        self._c_fn = comp._household_function(
            "_dp_two_asset_c", parse(self.cname, comp.state_names)
        )
        self._validate_model(comp.param_values())

        # Wire policy and evaluation hooks into comp
        comp.evaluate = self.evaluate
        comp.policy_values = self.policy_values

    def _find_consumption_local(self) -> str:
        for name, _ in self.comp.locals:
            if name.lower() in ("c", "cons", "consumption"):
                return name
        raise ModelSpecError("two-asset EGM requires a consumption local named c, cons or consumption")

    def _extract_params(self, params: dict[str, Any]) -> dict[str, float]:
        # Normalize compiled p_ names and aliases once; use the same recognized
        # parameter bindings as the model equation and nonlinear/Jacobian bridge.
        normalized = {name.removeprefix("p_"): value for name, value in params.items()}
        missing = set(self.parameter_bindings.values()) | {self.comp.model._beta}
        missing -= normalized.keys()
        if missing:
            raise ModelSpecError(f"two-asset EGM missing parameter bindings: {sorted(missing)}")
        values = {role: float(normalized[name]) for role, name in self.parameter_bindings.items()}
        values["beta"] = float(normalized[self.comp.model._beta])
        from puremacro.dsge._ast import Const, Param
        reward = self.comp.reward
        if reward.func.lower() in ("log", "ln"):
            values["gamma"] = 1.0
        else:
            gamma = reward.args[1]
            if isinstance(gamma, Param) and gamma.name not in normalized:
                raise ModelSpecError(f"two-asset EGM curvature must bind a declared parameter: {gamma.name}")
            values["gamma"] = float(gamma.value if isinstance(gamma, Const) else normalized[gamma.name])
        return values

    def _call(self, fn, ap, bp, a, b, s, params):
        nxt, cur = ([ap, bp], [a, b]) if self.illiquid_idx == 0 else ([bp, ap], [b, a])
        return fn(*nxt, *cur, s, *(params[f"p_{k}"] for k in self.comp.param_names), xp=np)

    def _validate_model(self, params: dict[str, Any]) -> None:
        """Reject declarations outside the built-in CRRA/convex-cost problem.

        Structural recognition already established the resource/cost identity.
        These state/choice probes are a second numerical/domain check, not the
        basis for accepting arbitrary user budget functions.
        """
        p = self._extract_params(params)
        if not all(np.isfinite(v) for v in p.values()):
            raise ModelSpecError("two-asset EGM parameters must be finite")
        if not (0 < p["beta"] < 1 and p["gamma"] > 0 and p["r_b"] > -1
                and p["r_a"] > -1 and p["w"] > 0 and p["chi_0"] > 0
                and p["chi_1"] >= 1 and p["a_bar"] > 0):
            raise ModelSpecError("two-asset EGM requires 0 < beta < 1, positive gamma, w, chi_0, "
                                 "a_bar, chi_1 >= 1 and gross returns > 0")
        if self.a_grid[0] < 0 or min(self.n_a, self.n_b) < 2:
            raise ModelSpecError("two-asset EGM needs at least two grid nodes per asset and nonnegative illiquid assets")

        # Include endpoints, interior nodes, and off-grid points to distinguish
        # changes in costs, income, and extra portfolio constraints.
        def probes(grid):
            return np.unique(np.r_[grid[::max(1, len(grid) // 7)], grid[-1],
                                   np.linspace(grid[0], grid[-1], 7)])

        a, b, ap, bp, s = np.meshgrid(probes(self.a_grid), probes(self.b_grid),
                                     probes(self.a_grid), probes(self.b_grid), self.s_grid,
                                     indexing="ij", sparse=True)
        cost = _two_asset_cost(ap, a, p["r_a"], p["chi_0"], p["chi_1"], p["a_bar"])[0]
        expected_c = (1 + p["r_b"]) * b + (1 + p["r_a"]) * a + p["w"] * s - ap - bp - cost
        declared_c = np.broadcast_to(self._call(self._c_fn, ap, bp, a, b, s, params), expected_c.shape)
        if not np.allclose(declared_c, expected_c, rtol=1e-10, atol=1e-10):
            raise ModelSpecError("two-asset EGM requires c = (1+r_b)*b + (1+r_a)*a + w*s "
                                 "- a(+1) - b(+1) - chi with the supported convex adjustment cost")
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            u = np.broadcast_to(self._call(self.comp.household, ap, bp, a, b, s, params), expected_c.shape)
            expected_u = (np.log(expected_c) if p["gamma"] == 1 else
                          (expected_c ** (1 - p["gamma"]) - 1) / (1 - p["gamma"]))
        feasible = expected_c > 0
        if not np.array_equal(np.isfinite(u), feasible):
            raise ModelSpecError("two-asset EGM supports exactly the consumption > 0 constraint")
        if not np.allclose(u[feasible], expected_u[feasible], rtol=1e-9, atol=1e-9):
            raise ModelSpecError("two-asset EGM requires reward crra(consumption, gamma) or log(consumption)")

    def solve_household(
        self,
        params: dict[str, Any],
        tol: float = _TA_EGM_TOL,
        max_iter: int = _TA_EGM_MAX_ITER,
        c0: np.ndarray | None = None,
    ) -> TwoAssetHouseholdSolution:
        """Solve the two-asset stationary household problem on (a, b, s)."""
        if not np.isfinite(tol) or tol <= 0 or int(max_iter) != max_iter or max_iter < 1:
            raise ValueError("tol must be finite and positive and max_iter must be a positive integer")
        self._validate_model(params)
        p = self._extract_params(params)
        beta, gamma = p["beta"], p["gamma"]
        r_b, r_a, w = p["r_b"], p["r_a"], p["w"]
        chi_0, chi_1, a_bar = p["chi_0"], p["chi_1"], p["a_bar"]

        na, nb, ns = self.n_a, self.n_b, self.n_s
        a_grid, b_grid, s_grid = self.a_grid, self.b_grid, self.s_grid
        pi_s = self.pi_s

        # Initial guess for marginal values
        cash0 = (
            (1.0 + r_b) * b_grid[None, :, None]
            + (1.0 + r_a) * a_grid[:, None, None]
            + w * s_grid[None, None, :]
        )
        c_guess = np.maximum(0.05 * cash0, _TA_C_FLOOR) if c0 is None else np.asarray(c0, dtype=float)
        if c_guess.shape != (na, nb, ns) or not np.all(np.isfinite(c_guess) & (c_guess > 0)):
            raise ValueError(f"c0 must be positive and finite with shape {(na, nb, ns)}")
        uc0 = c_guess ** (-gamma)
        Vb = (1.0 + r_b) * uc0
        Va = (1.0 + r_a) * uc0

        converged = False
        resid = float("nan")
        c = ap = bp = d = chi = np.zeros((na, nb, ns))

        for it in range(1, int(max_iter) + 1):
            Va_new, Vb_new, c_new, ap_new, bp_new, d_new, chi_new = _two_asset_step(
                Va, Vb, a_grid, b_grid, s_grid, pi_s, beta, gamma, r_b, r_a, w, chi_0, chi_1, a_bar
            )
            step_c = np.max(np.abs(c_new - c))
            step_a = np.max(np.abs(ap_new - ap))
            step_b = np.max(np.abs(bp_new - bp))
            resid = float(max(step_c, step_a, step_b))

            Va, Vb, c, ap, bp, d, chi = (
                Va_new,
                Vb_new,
                c_new,
                ap_new,
                bp_new,
                d_new,
                chi_new,
            )
            if resid < tol:
                converged = True
                break

        if not converged:
            warnings.warn(f"two-asset EGM did not converge: policy residual {resid:.3e} "
                          f"after {it} iterations", RuntimeWarning, stacklevel=2)
        if _two_asset_floored_states(a_grid, b_grid, s_grid, r_b, r_a, w, ap, bp, chi):
            raise RuntimeError("two-asset EGM consumption floor binds: the budget is infeasible on this grid")

        # Stationary distribution
        Lam = _build_transition_matrix_2d(ap, bp, a_grid, b_grid, pi_s)
        D_ss = _stationary_distribution_2d(Lam).reshape((na, nb, ns))
        stat_ok = bool(np.all(np.isfinite(D_ss)) and abs(D_ss.sum() - 1.0) < 1e-10
                       and np.max(np.abs(Lam @ D_ss.ravel() - D_ss.ravel())) < 1e-10)
        converged = converged and stat_ok

        # Reconstruct total value function V by policy evaluation
        u_val = self._call(self.comp.household, ap, bp, a_grid[:, None, None],
                          b_grid[None, :, None], s_grid[None, None, :], params)
        declared_c = self._call(self._c_fn, ap, bp, a_grid[:, None, None],
                                b_grid[None, :, None], s_grid[None, None, :], params)
        if not np.all(np.isfinite(u_val)) or not np.allclose(declared_c, c, rtol=1e-10, atol=1e-10):
            raise ModelSpecError("two-asset EGM policy violates the declared budget or feasible set")
        # Solve (I - beta * Lam^T) V = u
        N = na * nb * ns
        eye_N = sp.eye(N, format="csc")
        T_mat = sp.csc_matrix(Lam.T)
        V_flat = sp.linalg.spsolve(eye_N - beta * T_mat, u_val.ravel())
        V = V_flat.reshape((na, nb, ns))

        hh = _TwoAssetHouseholdBlock(
            a_grid=a_grid,
            b_grid=b_grid,
            s_grid=s_grid,
            pi_s=pi_s,
            beta=beta,
            gamma=gamma,
            r_b_ss=r_b,
            r_a_ss=r_a,
            w_ss=w,
            chi_0=chi_0,
            chi_1=chi_1,
            a_bar=a_bar,
            Va_ss=Va,
            Vb_ss=Vb,
            c_ss=c,
            d_ss=d,
            a_ss=ap,
            b_ss=bp,
            chi_ss=chi,
            D_ss=D_ss,
            Lambda=Lam,
            converged=converged,
            iterations=it,
            residual=resid,
            tol=tol,
        )

        return TwoAssetHouseholdSolution(
            V=V,
            Va=Va,
            Vb=Vb,
            c=c,
            aprime=ap,
            bprime=bp,
            d=d,
            chi=chi,
            n_iter=it,
            sup_norm=resid,
            r_a=r_a,
            r_b=r_b,
            w=w,
            converged=converged,
            hh_block=hh,
        )

    def distribution(self, sol: TwoAssetHouseholdSolution) -> np.ndarray:
        """Stationary distribution mu(a, b, s)."""
        if sol.hh_block is not None:
            return sol.hh_block.D_ss
        Lam = _build_transition_matrix_2d(sol.aprime, sol.bprime, self.a_grid, self.b_grid, self.pi_s)
        return _stationary_distribution_2d(Lam).reshape((self.n_a, self.n_b, self.n_s))

    def evaluate(
        self,
        fn: Any,
        sol: TwoAssetHouseholdSolution,
        params: dict[str, Any],
        age: int | None = None,
    ) -> np.ndarray:
        """Evaluate a household function on the (n_a, n_b, n_s) state space."""
        na, nb, ns = self.n_a, self.n_b, self.n_s
        a_grid, b_grid, s_grid = self.a_grid, self.b_grid, self.s_grid

        # Current state coordinates
        cur_a = a_grid[:, None, None]
        cur_b = b_grid[None, :, None]
        cur_s = s_grid[None, None, :]

        # Policies
        ap = sol.aprime
        bp = sol.bprime

        # Build args in the order expected by comp:
        # [nxt_0, nxt_1, cur_0, cur_1, z, *params]
        if self.illiquid_idx == 0:
            nxt = [ap, bp]
            cur = [cur_a, cur_b]
        else:
            nxt = [bp, ap]
            cur = [cur_b, cur_a]

        zs = [cur_s]
        args = nxt + cur + zs + [params[f"p_{k}"] for k in self.comp.param_names]
        val = fn(*args, xp=np)
        return np.broadcast_to(np.asarray(val, dtype=float), (na, nb, ns)).copy()

    def policy_values(
        self,
        expr: str,
        sol: TwoAssetHouseholdSolution,
        params: dict[str, Any],
    ) -> np.ndarray:
        """Evaluate any named policy or expression on the (a, b, s) grid."""
        node = parse(expr, set(self.comp.state_names) | set(self.comp.shock_names), functions=self.comp.model._functions)
        fn = self.comp._household_function("_dp_two_asset_policy", node)
        return self.evaluate(fn, sol, params)
