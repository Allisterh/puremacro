"""Heterogeneous Agents (HANK) Sequence-Space Bridge in .mod files.

Bridges Dynare-style .mod specifications carrying a `hetagent_block` with the
Sequence-Space Jacobian (SSJ) engine from Auclert, Bardóczy, Rognlie & Straub
(2021, Econometrica).

Key Features
------------
1. Parses `hetagent_block; ... end;` in .mod files. The `model;` block is
   parsed and kept on `HANKModel.dag` for inspection but is not solved: the
   general-equilibrium closure is the built-in New Keynesian Phillips curve and
   Taylor rule on the declared `beta`, `kappa`, `phi_pi` (and `alpha_ab` for
   two-asset) parameters, see `_ge_matrices` in `models.hank_sequence_space`.
2. Computes microeconomic stationary distributions D*(a) and consumption Jacobians
   (J_C_r, J_C_Y) via the Fake-News Algorithm.
3. Solves coupled general equilibrium MIT transition paths in sequence space
   without curse-of-dimensionality state space explosion.
4. Provides HANKResult with transition paths, distribution diagnostics, summary
   tables, and publication-ready matplotlib figures.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from puremacro.dsge._parser import parse_mod_to_dag, ParsedModelDAG
from puremacro.dsge._results import HANKResult
from puremacro.models.hank_sequence_space import (
    solve_hank_sequence_space,
    solve_two_asset_hank_sequence_space,
    solve_nonlinear_transition,
    fake_news_algorithm,
    dp_sequence_jacobian,
    dp_nonlinear_transition,
    _prepare_dp_household_block,
    _solve_household_block,
    _solve_two_asset_household_block,
    _consumption_jacobians,
    _two_asset_jacobians,
    _two_asset_ge_solve,
    _two_asset_steady_state_dict,
    _ge_matrices,
    _local_mpc,
    _transaction_cost,
    TwoAssetSequenceSpaceHANKResult,
    _HouseholdBlock,
    _TwoAssetHouseholdBlock,
)

# Two-asset calibration defaults (used when the .mod file does not set them); they
# match solve_two_asset_hank_sequence_space and give beta (1 + r_a) < 1, so that the
# illiquid-wealth distribution is interior.
_TWO_ASSET_DEFAULTS = {
    "beta": 0.98,
    "r_b_ss": 0.005,
    "r_a_ss": 0.0125,
    "chi_0": 1.0,
    "chi_1": 1.0,
    "a_bar": 0.25,
    "n_a": 25,
    "n_b": 25,
    "a_max": 40.0,
    "b_max": 10.0,
    "b_min": 0.0,
    "alpha_ab": 0.5,
}


class HANKModel:
    """Coupled Heterogeneous-Agent DSGE Model representation."""

    def __init__(
        self,
        dag: ParsedModelDAG,
        params: dict[str, float] | None = None,
        household_model: Any = None,
    ) -> None:
        self.dag = dag
        self.hetagent_config = dict(dag.hetagent_block or {})
        self.household_model = household_model
        
        # Merge declared parameters with explicit overrides
        p = dict(dag.parameter_values)
        if params:
            p.update(params)
        self.params = p

        self.variables = list(dag.variables)
        self.shocks = list(dag.shocks)

        # Determine 1-asset vs 2-asset configuration
        assets_val = self.hetagent_config.get("assets", 1)
        model_type = str(self.hetagent_config.get("model", "")).lower()
        is_ta_cfg = (
            int(float(assets_val)) == 2
            or "two_asset" in model_type
            or "2_asset" in model_type
        )
        is_ta_model = False
        if self.household_model is not None:
            from puremacro.dp._results import DPSolution

            m_orig = (
                self.household_model._c.model
                if isinstance(self.household_model, DPSolution)
                else self.household_model
            )
            if hasattr(m_orig, "_states") and len(m_orig._states) == 2:
                is_ta_model = True
            if is_ta_cfg and not is_ta_model:
                raise ValueError("a two-asset hetagent_block requires a two-state household_model")
        self.is_two_asset = is_ta_cfg or is_ta_model

        # Baseline calibration parameters
        ta = _TWO_ASSET_DEFAULTS
        self.beta = float(self.params.get("beta", ta["beta"] if self.is_two_asset else 0.985))
        self.gamma = float(self.params.get("gamma", 1.0))
        self.r_ss = float(self.params.get("r_ss", self.params.get("r_b_ss", ta["r_b_ss"] if self.is_two_asset else 0.01)))
        self.r_b_ss = float(self.params.get("r_b_ss", self.r_ss))
        self.r_a_ss = float(self.params.get("r_a_ss", ta["r_a_ss"]))
        self.phi_pi = float(self.params.get("phi_pi", 1.5))
        self.kappa = float(self.params.get("kappa", 0.1))
        self.chi_0 = float(self.hetagent_config.get("chi_0", self.params.get("chi_0", ta["chi_0"])))
        self.chi_1 = float(self.hetagent_config.get("chi_1", self.params.get("chi_1", ta["chi_1"])))
        self.a_bar = float(self.hetagent_config.get("a_bar", self.params.get("a_bar", ta["a_bar"])))
        self.alpha_ab = float(self.params.get("alpha_ab", ta["alpha_ab"]))

        # Solve stationary distribution on initialization
        self.steady_state: dict[str, float] = {}
        self.asset_grid: np.ndarray = np.array([])
        self.liquid_asset_grid: np.ndarray | None = None
        self.asset_distribution: np.ndarray = np.array([])
        self.marginal_distribution_b: np.ndarray | None = None
        self.joint_distribution: np.ndarray | None = None
        self.deposit_distribution: np.ndarray | None = None
        self.mpc_distribution: np.ndarray | None = None
        self._hh_block: _HouseholdBlock | _TwoAssetHouseholdBlock | None = None
        self._dp_block: dict[str, Any] | None = None
        self._dp_inputs: list[str] = []
        self._solve_steady_state()

    def _resolve_dp_inputs(self) -> list[str]:
        """Identify rate and income/wage input names on ``self.household_model``."""
        from puremacro.dp._results import DPSolution

        m_orig = self.household_model._c.model if isinstance(self.household_model, DPSolution) else self.household_model
        avail = set(m_orig._params.keys()) | set(m_orig._prices.keys()) | {n for n, _ in m_orig._locals}
        inputs: list[str] = []
        for cand in ("r_b", "r_a", "rb", "ra", "r", "w", "y", "z"):
            if cand in avail:
                inputs.append(cand)
        if not inputs:
            inputs = list(m_orig._prices.keys()) or list(m_orig._params.keys())[:1]
        return inputs

    def _solve_steady_state(self) -> None:
        """Solve the microeconomic stationary distribution and aggregate steady state."""
        if self.is_two_asset:
            if self.household_model is not None:
                self._dp_inputs = self._resolve_dp_inputs()
                blk = _prepare_dp_household_block(self.household_model, inputs=self._dp_inputs, outputs=None)
                self._dp_block = blk
                hh_ta = blk["hh_block"]
            else:
                ta = _TWO_ASSET_DEFAULTS
                n_a = int(self.hetagent_config.get("n_a", ta["n_a"]))
                n_b = int(self.hetagent_config.get("n_b", ta["n_b"]))
                a_max = float(self.hetagent_config.get("a_max", ta["a_max"]))
                b_max = float(self.hetagent_config.get("b_max", ta["b_max"]))
                b_min = float(self.hetagent_config.get("b_min", ta["b_min"]))

                hh_ta = _solve_two_asset_household_block(
                    beta=self.beta,
                    gamma=self.gamma,
                    r_b_ss=self.r_b_ss,
                    r_a_ss=self.r_a_ss,
                    w_ss=1.0,
                    n_a=n_a,
                    n_b=n_b,
                    a_max=a_max,
                    b_max=b_max,
                    b_min=b_min,
                    chi_0=self.chi_0,
                    chi_1=self.chi_1,
                    a_bar=self.a_bar,
                )
            self._hh_block = hh_ta
            self.asset_grid = hh_ta.a_grid.copy()
            self.liquid_asset_grid = hh_ta.b_grid.copy()
            self.asset_distribution = hh_ta.marginal_distribution_a.copy()
            self.marginal_distribution_b = hh_ta.marginal_distribution_b.copy()
            self.joint_distribution = hh_ta.joint_distribution.copy()
            self.deposit_distribution = hh_ta.deposit_distribution.copy()

            # Quarterly MPC out of a lump-sum transfer, averaged over (b, s) at each illiquid grid point
            mpc_state = hh_ta.mpc()
            weight = hh_ta.D_ss.sum(axis=(1, 2))
            mpc_a = np.mean(mpc_state, axis=(1, 2))
            has_mass = weight > 1e-12
            mpc_a[has_mass] = np.sum(hh_ta.D_ss * mpc_state, axis=(1, 2))[has_mass] / weight[has_mass]
            self.mpc_distribution = mpc_a

            self.steady_state = _two_asset_steady_state_dict(hh_ta)
            self.steady_state["r"] = hh_ta.r_b_ss
            for v in self.variables:
                if v not in self.steady_state:
                    self.steady_state[v] = 0.0
            return

        if self.household_model is not None:
            self._dp_inputs = self._resolve_dp_inputs()
            blk = _prepare_dp_household_block(self.household_model, inputs=self._dp_inputs, outputs=None)
            self._dp_block = blk
            self.asset_grid = blk["a_grid"].copy()
            D_ss = blk["D_ss"]
            self.asset_distribution = np.sum(D_ss, axis=1)
            params_ss = blk["params_ss"]
            if "p_r" in params_ss:
                self.r_ss = float(params_ss["p_r"])
            w_ss = float(params_ss.get("p_w", params_ss.get("p_y", 1.0)))
            c_ss_grid = blk["c_ss"]
            n_a = len(self.asset_grid)
            mpc_mat = _local_mpc(c_ss_grid, self.asset_grid, self.r_ss)
            weight = np.sum(D_ss, axis=1)
            mpc_cond = np.zeros(n_a)
            mask = weight > 1e-12
            mpc_cond[mask] = np.sum(D_ss[mask] * mpc_mat[mask], axis=1) / weight[mask]
            mpc_cond[~mask] = np.mean(mpc_mat[~mask], axis=1)
            self.mpc_distribution = mpc_cond

            c_ss = float(np.sum(D_ss * c_ss_grid))
            b_ss = float(np.sum(D_ss * blk["aprime_ss"]))
            self.steady_state = {
                "Y": c_ss,
                "C": c_ss,
                "r": self.r_ss,
                "pi": 0.0,
                "i": self.r_ss,
                "w": w_ss,
                "B": b_ss,
                "A": b_ss,
            }
            for v in self.variables:
                if v not in self.steady_state:
                    self.steady_state[v] = 0.0
            return

        n_a = int(self.hetagent_config.get("n_a", 100))
        a_max = float(self.hetagent_config.get("a_max", 50.0))

        hh_sa = _solve_household_block(
            beta=self.beta,
            gamma=self.gamma,
            r_ss=self.r_ss,
            n_a=n_a,
            a_max=a_max,
        )
        self._hh_block = hh_sa
        self.asset_grid = hh_sa.a_grid.copy()
        self.asset_distribution = np.sum(hh_sa.D_ss, axis=1)
        mpc_mat = _local_mpc(hh_sa.c_ss, hh_sa.a_grid, hh_sa.r_ss)
        weight = np.sum(hh_sa.D_ss, axis=1)
        mpc_cond = np.zeros(n_a)
        mask = weight > 1e-12
        mpc_cond[mask] = np.sum(hh_sa.D_ss[mask] * mpc_mat[mask], axis=1) / weight[mask]
        mpc_cond[~mask] = np.mean(mpc_mat[~mask], axis=1)
        self.mpc_distribution = mpc_cond

        c_ss = float(hh_sa.C_ss)
        y_ss = c_ss  # Market clearing Y = C in baseline
        self.steady_state = {
            "Y": y_ss,
            "C": c_ss,
            "r": self.r_ss,
            "pi": 0.0,
            "i": self.r_ss,
            "w": 1.0,
            "B": float(hh_sa.B),
        }
        # Populate any other variables from .mod file with steady state
        for v in self.variables:
            if v not in self.steady_state:
                self.steady_state[v] = 0.0

    def compute_jacobians(self, T: int = 300) -> dict[str, np.ndarray]:
        """Compute sequence-space Jacobians via Fake-News Algorithm.

        Two-asset models return the Jacobians of the household sector closed by
        the income rule ``z_t N = Y_t - r^b_t B_{t-1} - r^a_t A_{t-1}``
        (``J_O_Y``, ``J_O_rb``, ``J_O_ra`` for O in C, D, A, B, CHI, ex-ante
        dated, plus ``J_z_*`` for non-financial income); see
        :func:`puremacro.models.hank_sequence_space._two_asset_jacobians`.
        """
        if self._hh_block is None and self._dp_block is None:
            self._solve_steady_state()
        if self.is_two_asset:
            assert isinstance(self._hh_block, _TwoAssetHouseholdBlock)
            return _two_asset_jacobians(self._hh_block, T=T)

        if self.household_model is not None:
            J_dp = dp_sequence_jacobian(self.household_model, inputs=self._dp_inputs, outputs=None, T=T)
            c_key = "C" if "C" in J_dp else ("c" if "c" in J_dp else next(iter(J_dp)))
            r_key = "r" if "r" in self._dp_inputs else self._dp_inputs[0]
            w_key = "w" if "w" in self._dp_inputs else ("y" if "y" in self._dp_inputs else None)
            J_C_r = J_dp[c_key][r_key]
            c_ss = max(float(self.steady_state.get("C", 1.0)), 1e-12)
            w_ss = float(self.steady_state.get("w", 1.0))
            if w_key is not None:
                # Apply balanced-budget fiscal closure (dw_net = dw - B * dr) when both r and w are present
                B_ss = float(self.steady_state.get("B", 0.0))
                J_C_w = J_dp[c_key][w_key]
                J_C_Y = J_C_w * (w_ss / c_ss)
                J_C_r_closed = J_C_r - B_ss * J_C_w
            else:
                J_C_Y = np.zeros((T, T), dtype=float)
                J_C_r_closed = J_C_r
            out: dict[str, np.ndarray] = {
                "J_C_r": J_C_r_closed,
                "J_C_Y": J_C_Y,
            }
            for y_name, x_map in J_dp.items():
                for x_name, mat in x_map.items():
                    out[f"J_{y_name}_{x_name}"] = mat
            return out

        if self._hh_block is None:
            self._solve_steady_state()
        assert isinstance(self._hh_block, _HouseholdBlock)
        J_C_r, J_C_Y = _consumption_jacobians(self._hh_block, T=T)
        return {
            "J_C_r": J_C_r,
            "J_C_Y": J_C_Y,
        }

    def simulate(
        self,
        shock: str = "eps_m",
        magnitude: float = -0.0025,
        rho: float = 0.5,
        horizon: int = 40,
        nonlinear: bool = False,
    ) -> HANKResult:
        """Simulate general equilibrium transition dynamics for an MIT shock.

        The closure is the built-in NKPC + Taylor rule on the declared ``beta``,
        ``kappa``, ``phi_pi`` (and ``alpha_ab`` for two-asset) parameters with the
        single monetary shock path ``magnitude * rho**t``; the ``model;`` block of
        the .mod file is not read, and ``shock`` only labels the result
        (``shock_name``, plus a ``transition_paths`` column holding the monetary
        path when the name is a declared ``varexo``). ``nonlinear=True`` (Broyden quasi-Newton)
        is implemented for the built-in one-asset household only: a two-asset
        ``hetagent_block`` or a dp ``household_model`` raises ``NotImplementedError``.
        """
        if nonlinear and self.household_model is not None:
            raise NotImplementedError("nonlinear general-equilibrium simulation for a dp household_model "
                                      "is not implemented; dp_nonlinear_transition computes household "
                                      "responses to specified price paths")
        if nonlinear and self.is_two_asset:
            raise NotImplementedError("nonlinear general-equilibrium simulation for the two-asset household "
                                      "is not implemented; two-asset models are solved by the linear "
                                      "sequence-space method (nonlinear=False)")
        shock_path = magnitude * (rho ** np.arange(horizon))

        if self.is_two_asset:
            # Linear GE on the steady state solved at construction (the same block the
            # reported steady state and distributions come from).
            assert isinstance(self._hh_block, _TwoAssetHouseholdBlock)
            res_2a = _two_asset_ge_solve(
                self._hh_block,
                int(horizon),
                self.phi_pi,
                self.kappa,
                shock_path,
                self.alpha_ab,
            )
            dY = res_2a.irf_output[:horizon]
            dC = res_2a.irf_consumption[:horizon]
            dD = res_2a.irf_deposit[:horizon]
            dr_b = res_2a.irf_rate_b[:horizon]
            dr_a = res_2a.irf_rate_a[:horizon]
            dpi = res_2a.irf_inflation[:horizon]
            di = self.phi_pi * dpi + shock_path

            path_data_2a: dict[str, np.ndarray] = {
                "Y": dY,
                "C": dC,
                "CHI": dY - dC,  # adjustment costs: goods market Y = C + CHI
                "D": dD,
                "r_b": dr_b,
                "r_a": dr_a,
                "r": dr_b,
                "pi": dpi,
                "i": di,
            }
            if shock in self.shocks:
                path_data_2a[shock] = shock_path
            for v in self.variables:
                if v not in path_data_2a:
                    path_data_2a[v] = np.zeros(horizon)

            df_paths = pd.DataFrame(path_data_2a)
            J_dict = {
                "J_C_rb": res_2a.jacobian_c_rb,
                "J_C_ra": res_2a.jacobian_c_ra,
                "J_C_Y": res_2a.jacobian_c_y,
                "J_D_rb": res_2a.jacobian_d_rb,
                "J_D_ra": res_2a.jacobian_d_ra,
                "J_D_Y": res_2a.jacobian_d_y,
                "J_C_r": res_2a.jacobian_c_rb,
            }

            return HANKResult(
                steady_state=self.steady_state,
                transition_paths=df_paths,
                jacobians=J_dict,
                asset_distribution=self.asset_distribution,
                asset_grid=self.asset_grid,
                liquid_asset_grid=self.liquid_asset_grid,
                joint_distribution=self.joint_distribution,
                marginal_distribution_b=self.marginal_distribution_b,
                deposit_distribution=self.deposit_distribution,
                mpc_distribution=self.mpc_distribution,
                shock_name=shock,
                horizon=horizon,
                model_name=self.hetagent_config.get("model", "hank_two_asset"),
                converged=res_2a.converged,
            )

        if self.household_model is not None:
            J_dict = self.compute_jacobians(T=horizon)
            J_C_r = J_dict["J_C_r"]
            J_C_Y = J_dict["J_C_Y"]
            K_pi, M_r_Y = _ge_matrices(horizon, float(self.beta), float(self.kappa), float(self.phi_pi))
            H_Y = np.eye(horizon) - J_C_Y - J_C_r @ M_r_Y
            rhs = J_C_r @ shock_path
            dY = np.linalg.solve(H_Y, rhs)
            dC = dY.copy()
            dr = M_r_Y @ dY + shock_path
            dpi = K_pi @ dY
            di = self.phi_pi * dpi + shock_path
            converged = bool(np.all(np.isfinite(dY)))

            path_data_dp: dict[str, np.ndarray] = {
                "Y": dY,
                "C": dC,
                "r": dr,
                "pi": dpi,
                "i": di,
            }
            if shock in self.shocks:
                path_data_dp[shock] = shock_path
            for v in self.variables:
                if v not in path_data_dp:
                    path_data_dp[v] = dY - dC if v == "I" else np.zeros(horizon)

            return HANKResult(
                steady_state=self.steady_state,
                transition_paths=pd.DataFrame(path_data_dp),
                jacobians=J_dict,
                asset_distribution=self.asset_distribution,
                asset_grid=self.asset_grid,
                mpc_distribution=self.mpc_distribution,
                shock_name=shock,
                horizon=horizon,
                model_name=self.hetagent_config.get("model", "hank_dp_bridge"),
                converged=converged,
            )

        if nonlinear:
            res_nonlin = solve_nonlinear_transition(
                beta=self.beta,
                gamma=self.gamma,
                r_ss=self.r_ss,
                phi_pi=self.phi_pi,
                kappa=self.kappa,
                horizon=horizon,
                shock_seq=shock_path,
                n_a=int(self.hetagent_config.get("n_a", 50)),
                a_max=float(self.hetagent_config.get("a_max", 30.0)),
            )
            dY = res_nonlin.irf_output_nonlinear[:horizon]
            dC = res_nonlin.irf_consumption_nonlinear[:horizon]
            dr = res_nonlin.irf_rate_nonlinear[:horizon]
            dpi = res_nonlin.irf_inflation_nonlinear[:horizon]
            converged = res_nonlin.converged
            J_dict = {
                "J_C_r": res_nonlin.jacobian_c_r,
                "J_C_Y": res_nonlin.jacobian_c_y,
            }
        else:
            res_ssj = solve_hank_sequence_space(
                T=horizon,
                beta=self.beta,
                gamma=self.gamma,
                r_ss=self.r_ss,
                phi_pi=self.phi_pi,
                kappa=self.kappa,
                shock_magnitude=magnitude,
                shock_rho=rho,
                n_a=int(self.hetagent_config.get("n_a", 50)),
                a_max=float(self.hetagent_config.get("a_max", 30.0)),
            )
            dY = res_ssj.irf_output[:horizon]
            dC = res_ssj.irf_consumption[:horizon]
            dr = res_ssj.irf_rate[:horizon]
            dpi = res_ssj.irf_inflation[:horizon]
            converged = True
            J_dict = {
                "J_C_r": res_ssj.jacobian_c_r,
                "J_C_Y": res_ssj.jacobian_c_y,
            }

        # Build paths DataFrame matching variables in the .mod file
        di = self.phi_pi * dpi + shock_path

        path_data: dict[str, np.ndarray] = {
            "Y": dY,
            "C": dC,
            "r": dr,
            "pi": dpi,
            "i": di,
        }
        if shock in self.shocks:
            path_data[shock] = shock_path

        # Add remaining declared variables
        for v in self.variables:
            if v not in path_data:
                if v == "w":
                    path_data[v] = np.zeros(horizon)
                elif v == "K":
                    path_data[v] = np.zeros(horizon)
                elif v == "I":
                    path_data[v] = dY - dC
                else:
                    path_data[v] = np.zeros(horizon)

        df_paths = pd.DataFrame(path_data)

        return HANKResult(
            steady_state=self.steady_state,
            transition_paths=df_paths,
            jacobians=J_dict,
            asset_distribution=self.asset_distribution,
            asset_grid=self.asset_grid,
            mpc_distribution=self.mpc_distribution,
            shock_name=shock,
            horizon=horizon,
            model_name=self.hetagent_config.get("model", "hank_sequence_space"),
            converged=converged,
        )


def load_hank_mod(
    source: str | Path,
    household_model: Any = None,
    **param_overrides: Any,
) -> HANKModel:
    """Load and parse a Dynare .mod file containing a hetagent_block."""
    if isinstance(source, Path) or (isinstance(source, str) and "\n" not in source and Path(source).exists()):
        text = Path(source).read_text(encoding="utf-8")
    else:
        text = str(source)

    dag = parse_mod_to_dag(text)
    if dag.hetagent_block is None and household_model is None:
        raise ValueError(
            "load_hank_mod: the supplied .mod file does not contain a hetagent_block; "
            "use load_mod for standard representative-agent DSGE models."
        )

    return HANKModel(dag, params=param_overrides, household_model=household_model)


def solve_hank_bridge(
    source: str | Path,
    shock: str = "eps_m",
    magnitude: float = -0.0025,
    rho: float = 0.5,
    horizon: int = 40,
    nonlinear: bool = False,
    household_model: Any = None,
    **param_overrides: Any,
) -> HANKResult:
    """High-level interface to solve a HANK .mod model and return transition dynamics."""
    model = load_hank_mod(source, household_model=household_model, **param_overrides)
    return model.simulate(
        shock=shock,
        magnitude=magnitude,
        rho=rho,
        horizon=horizon,
        nonlinear=nonlinear,
    )


__all__ = [
    "HANKModel",
    "HANKResult",
    "load_hank_mod",
    "solve_hank_bridge",
    "dp_sequence_jacobian",
    "dp_nonlinear_transition",
]

