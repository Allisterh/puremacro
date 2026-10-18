r"""Accelerated Semi-Analytic Gradients via the Implicit Function Theorem (IFT) for puremacro.vfi.

Provides fast semi-analytic Jacobians of continuous dynamic programming policy
functions and general equilibrium aggregates with respect to structural parameters
\theta = (\beta, \alpha, \delta, \sigma, \dots) on continuous Chebyshev collocation,
Finite Element Method (FEM), and cubic/Schumaker spline residual systems:

    R(c^*(\theta); \theta) = 0 \in \mathbb{R}^N

Applying the Implicit Function Theorem:

    \nabla_c R(c^*; \theta) \nabla_\theta c^*(\theta) + \nabla_\theta R(c^*; \theta) = 0
    \implies \nabla_\theta c^*(\theta) = - [\nabla_c R(c^*; \theta)]^{-1} \nabla_\theta R(c^*; \theta)

Computational & Algorithmic Advantages:
- Single LU Factorization: Factors J_c = \nabla_c R once in O(N^3) flops and solves for all p
  parameter columns in O(p N^2), avoiding repeated non-linear solves; speed depends on system size and conditioning.
- Fixed-Point Chatter Elimination: Evaluates the residual system directly at the converged
  equilibrium c^*, avoiding non-linear solver convergence chatter and inner-loop tolerance inconsistencies.
- Continuous Policy Gradients: Evaluates continuous sensitivities \nabla_\theta g(s) = \Phi(s) \nabla_\theta c^*
  at arbitrary continuous state coordinates s.
- Aggregate sensitivities: supported representative-agent formulas are available;
  heterogeneous-agent distribution and coupled GE sensitivities are not implemented.
- Robust Regularization Fallback: Automatically detects high condition numbers cond(J_c) > 10^12
  or singular systems, falling back to Tikhonov regularization or truncated SVD pseudoinverse.
- Uses NumPy/SciPy linear algebra; ships no compiled extension of its own.
- Full puremacro Presentation Contract: .summary(), .plot(), .to_frame(), .to_markdown(),
  .to_latex(), .to_typst().
"""
from __future__ import annotations

import inspect
import time
import warnings
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

if TYPE_CHECKING:
    import matplotlib.figure

import numpy as np
import pandas as pd
from scipy.linalg import lu_factor, lu_solve, svd
from scipy.optimize import brentq

from puremacro import _backend as _bk
from puremacro.reports import _df_to_latex, _df_to_markdown, _df_to_typst

_GREEK_LETTERS = {
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "mu", "nu", "xi", "pi", "rho", "sigma",
    "tau", "upsilon", "phi", "chi", "psi", "omega",
    "varepsilon", "vartheta", "varpi", "varrho", "varsigma", "varphi",
}


# ---------------------------------------------------------------------------
# AnalyticGradientResult: Presentation & Gradient Container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnalyticGradientResult:
    r"""Exact analytic gradients of continuous dynamic programming solutions via IFT.

    Attributes
    ----------
    grad_coefficients : np.ndarray
        Solved parameter Jacobian of policy coefficients \nabla_\theta c^*, shape (N, p).
    grad_aggregates : dict[str, np.ndarray]
        Macroeconomic aggregate parameter sensitivities:
        {"K": (p,), "C": (p,), "r": (p,), "w": (p,)}
    param_names : list[str]
        Ordered names of differentiated structural parameters, length p.
    jacobian_resid_c : np.ndarray
        Residual Jacobian with respect to coefficients J_c = \nabla_c R, shape (N, N).
    jacobian_resid_theta : np.ndarray
        Residual Jacobian with respect to parameters J_\theta = \nabla_\theta R, shape (N, p).
    condition_number : float
        Condition number of J_c in the 2-norm: cond(J_c) = \sigma_{max} / \sigma_{min}.
    elapsed_time : float
        Wall-clock runtime of the IFT evaluation in seconds.
    solution : Any
        Underlying solved solution object (CollocationSolution, FEMSolution, etc.).
    problem : Any
        Underlying economic problem specification.
    metadata : dict[str, Any]
        Diagnostic metadata, solver mode ('lu', 'tikhonov', 'svd'), and parameters.
    """

    grad_coefficients: np.ndarray
    grad_aggregates: Dict[str, np.ndarray]
    param_names: List[str]
    jacobian_resid_c: np.ndarray
    jacobian_resid_theta: np.ndarray
    condition_number: float
    elapsed_time: float
    solution: Any = None
    problem: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_params(self) -> int:
        """Number of structural parameters differentiated."""
        return len(self.param_names)

    @property
    def n_coefficients(self) -> int:
        """Number of basis polynomial or nodal degrees of freedom N."""
        return self.grad_coefficients.shape[0]

    def policy_gradient(
        self, s: Union[float, Sequence[float], np.ndarray]
    ) -> np.ndarray:
        r"""Evaluate the continuous policy gradient \nabla_\theta g(s) at state s.

        Parameters
        ----------
        s : float, Sequence[float], or np.ndarray
            Continuous state coordinates at which to evaluate the gradient.

        Returns
        -------
        grad : np.ndarray
            If s is a scalar float: 1D array of shape (p,).
            If s is a 1D sequence of length M: 2D array of shape (M, p).
        """
        arr_s = np.asarray(s, dtype=np.float64)
        is_scalar = arr_s.ndim == 0
        arr_s_1d = np.atleast_1d(arr_s)

        # 0. Heterogeneous-Agent General Equilibrium (AiyagariContinuousEquilibrium)
        if self.metadata.get("grad_policy_a") is not None and self.metadata.get("a_grid") is not None:
            a_grid = np.asarray(self.metadata["a_grid"], dtype=np.float64)
            grad_pa = np.asarray(self.metadata["grad_policy_a"], dtype=np.float64)  # (N_k, n_z, p)
            z_grid = self.metadata.get("z_grid", None)
            p = len(self.param_names)
            if arr_s.ndim == 2 and arr_s.shape[1] == 2 and z_grid is not None:
                M = len(arr_s)
                k_pts = arr_s[:, 0]
                z_pts = arr_s[:, 1]
                grad_eval = np.zeros((M, p), dtype=np.float64)
                for m, zm in enumerate(z_grid):
                    mask = np.isclose(z_pts, zm)
                    if np.any(mask):
                        for k in range(p):
                            grad_eval[mask, k] = np.interp(k_pts[mask], a_grid, grad_pa[:, m, k])
                return grad_eval
            M = len(arr_s_1d)
            grad_eval = np.zeros((M, p), dtype=np.float64)
            for k in range(p):
                grad_eval[:, k] = np.interp(arr_s_1d, a_grid, grad_pa[:, 0, k])
            return grad_eval[0] if is_scalar else grad_eval

        # 1. CollocationBasis (Chebyshev)
        if self.solution is not None and hasattr(self.solution, "basis") and hasattr(self.solution.basis, "n_dims"):
            basis = self.solution.basis
            backend = getattr(self.solution, "backend", "numpy")
            Phi = basis.evaluate(arr_s_1d, backend=backend)
            grad_eval = Phi @ self.grad_coefficients  # (M, N) @ (N, p) -> (M, p)

        # 2. FEMMesh
        elif self.solution is not None and hasattr(self.solution, "mesh"):
            mesh = self.solution.mesh
            dim_nodes = mesh.dim_nodes[0]
            M = len(arr_s_1d)
            p = len(self.param_names)
            grad_eval = np.zeros((M, p), dtype=np.float64)
            for k in range(p):
                grad_eval[:, k] = np.interp(arr_s_1d, dim_nodes, self.grad_coefficients[:, k])

        # 3. CubicBSplineBasis
        elif self.solution is not None and hasattr(self.solution, "basis") and hasattr(self.solution.basis, "interpolate"):
            basis = self.solution.basis
            backend = getattr(self.solution, "backend", "numpy")
            M = len(arr_s_1d)
            p = len(self.param_names)
            grad_eval = np.zeros((M, p), dtype=np.float64)
            for k in range(p):
                grad_eval[:, k] = basis.interpolate(self.grad_coefficients[:, k], arr_s_1d, deriv=0, backend=backend)

        # 4. SchumakerSpline or fallback
        elif self.solution is not None and hasattr(self.solution, "basis") and hasattr(self.solution.basis, "eval"):
            # Schumaker coefficients are nodal values
            spl = self.solution.basis
            nodes = spl.x if hasattr(spl, "x") else np.linspace(spl.domain[0], spl.domain[1], self.n_coefficients)
            M = len(arr_s_1d)
            p = len(self.param_names)
            grad_eval = np.zeros((M, p), dtype=np.float64)
            for k in range(p):
                grad_eval[:, k] = np.interp(arr_s_1d, nodes, self.grad_coefficients[:, k])

        # 5. Generic fallback: linear interpolation along uniform grid if nodes not found
        else:
            p = len(self.param_names)
            M = len(arr_s_1d)
            grad_eval = np.zeros((M, p), dtype=np.float64)
            domain = (arr_s_1d.min(), arr_s_1d.max()) if arr_s_1d.min() < arr_s_1d.max() else (0.0, 1.0)
            nodes = np.linspace(domain[0], domain[1], self.n_coefficients)
            for k in range(p):
                grad_eval[:, k] = np.interp(arr_s_1d, nodes, self.grad_coefficients[:, k])

        return grad_eval[0] if is_scalar else grad_eval

    def aggregate_gradient(self, name: str) -> np.ndarray:
        """Return the parameter gradient vector of a specific macroeconomic aggregate.

        Parameters
        ----------
        name : str
            Aggregate variable name ('K', 'C', 'r', 'w').

        Returns
        -------
        np.ndarray
            Gradient vector of shape (p,).
        """
        if name not in self.grad_aggregates:
            raise KeyError(f"Aggregate '{name}' not found; available: {list(self.grad_aggregates.keys())}")
        return self.grad_aggregates[name]

    def summary(self) -> pd.DataFrame:
        """Structured summary DataFrame of IFT gradient analysis."""
        solver_mode = self.metadata.get("solver_method", "LU Factorization")
        max_grad = float(np.max(np.abs(self.grad_coefficients)))
        frob_norm = float(np.linalg.norm(self.grad_coefficients, "fro"))

        rows = [
            {"Metric": "Method", "Value": "Implicit Function Theorem (IFT)"},
            {"Metric": "Linear Solver", "Value": str(solver_mode)},
            {"Metric": "Condition Number cond(J_c)", "Value": f"{self.condition_number:.4e}"},
            {"Metric": "Parameters Differentiated", "Value": ", ".join(self.param_names)},
            {"Metric": "Basis Dimension N", "Value": str(self.n_coefficients)},
            {"Metric": "Parameter Count p", "Value": str(self.n_params)},
            {"Metric": "Max |dc*/dtheta|", "Value": f"{max_grad:.4e}"},
            {"Metric": "Frobenius Norm ||dc*/dtheta||_F", "Value": f"{frob_norm:.4e}"},
            {"Metric": "Elapsed Time (s)", "Value": f"{self.elapsed_time:.4f}"},
        ]

        # Append aggregate sensitivities
        for agg_name, agg_vec in self.grad_aggregates.items():
            if agg_name != "mu" and agg_vec is not None and agg_vec.ndim == 1 and len(agg_vec) == self.n_params:
                vals_str = ", ".join(f"{p}:{v:.3e}" for p, v in zip(self.param_names, agg_vec))
                rows.append({"Metric": f"Aggregate Sensitivity d{agg_name}/dtheta", "Value": vals_str})

        return pd.DataFrame(rows).set_index("Metric")

    def to_frame(self) -> pd.DataFrame:
        """Tabulate parameter Jacobians of coefficients and macro aggregates.

        Returns
        -------
        pd.DataFrame
            DataFrame with differentiated parameters as columns and basis
            coefficients c_0 ... c_{N-1} plus macro aggregates (K, C, r, w) as rows.
        """
        row_names = [f"c_{i}" for i in range(self.n_coefficients)]
        data_rows = list(self.grad_coefficients)

        for agg_name, agg_vec in self.grad_aggregates.items():
            if agg_name != "mu" and agg_vec is not None and agg_vec.ndim == 1 and len(agg_vec) == self.n_params:
                row_names.append(f"agg_{agg_name}")
                data_rows.append(agg_vec)

        df = pd.DataFrame(data_rows, index=row_names, columns=self.param_names)
        return df

    def to_markdown(self, **kwargs: Any) -> str:
        """Render tabulated parameter Jacobians as Markdown."""
        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """Render tabulated parameter Jacobians as LaTeX tabular."""
        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Render tabulated parameter Jacobians as Typst table."""
        return _df_to_typst(self.to_frame(), **kwargs)

    def plot(
        self,
        figsize: Tuple[float, float] = (12, 4.5),
        show: bool = False,
        n_points: int = 200,
    ) -> matplotlib.figure.Figure:
        """Plot continuous policy parameter gradients and macroeconomic aggregate sensitivities.

        Parameters
        ----------
        figsize : tuple[float, float], default (12, 4.5)
            Matplotlib figure dimensions (width, height).
        show : bool, default False
            Whether to display the figure interactively.
        n_points : int, default 200
            Number of points for continuous policy gradient evaluation.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure instance containing the diagnostic subplots.
        """
        import matplotlib.pyplot as plt

        # Determine continuous state evaluation domain
        domain = (0.5, 2.0)
        if self.solution is not None:
            if hasattr(self.solution, "distribution") and hasattr(self.solution.distribution, "asset_grid"):
                ag = self.solution.distribution.asset_grid
                domain = (float(ag[0]), float(ag[-1]))
            elif hasattr(self.solution, "basis") and hasattr(self.solution.basis, "domain"):
                dom = self.solution.basis.domain
                if isinstance(dom[0], (tuple, list)):
                    domain = (float(dom[0][0]), float(dom[0][1]))
                else:
                    domain = (float(dom[0]), float(dom[1]))
            elif hasattr(self.solution, "mesh") and hasattr(self.solution.mesh, "domain"):
                dom = self.solution.mesh.domain
                domain = (float(dom[0][0]), float(dom[0][1]))
        elif self.problem is not None and hasattr(self.problem, "domain"):
            dom = self.problem.domain
            if isinstance(dom[0], (tuple, list)):
                domain = (float(dom[0][0]), float(dom[0][1]))
            else:
                domain = (float(dom[0]), float(dom[1]))

        s_grid = np.linspace(domain[0], domain[1], n_points)
        pol_grads = self.policy_gradient(s_grid)  # (n_points, p)

        fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

        # Panel 1: Continuous Policy Sensitivities \nabla_\theta g(s)
        ax1 = axes[0]
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
        for k, p_name in enumerate(self.param_names):
            color = colors[k % len(colors)]
            latex_sym = f"\\{p_name}" if p_name.lower() in _GREEK_LETTERS else p_name
            ax1.plot(s_grid, pol_grads[:, k], label=f"$\\partial g(s) / \\partial {latex_sym}$", color=color, lw=2.0)
        ax1.axhline(0.0, color="gray", ls="--", alpha=0.6)
        ax1.set_title("Continuous Policy Sensitivities $\\nabla_\\theta g(s)$")
        ax1.set_xlabel("Continuous State $s$")
        ax1.set_ylabel("Derivative $\\partial s' / \\partial \\theta$")
        ax1.legend(loc="best", frameon=True)
        ax1.grid(True, alpha=0.3)

        # Panel 2: Macroeconomic Aggregate Sensitivities
        ax2 = axes[1]
        agg_keys = [k for k in ("K", "C", "r", "w") if k in self.grad_aggregates]
        if agg_keys:
            n_aggs = len(agg_keys)
            p = len(self.param_names)
            bar_width = 0.8 / p
            indices = np.arange(n_aggs)

            for k, p_name in enumerate(self.param_names):
                color = colors[k % len(colors)]
                latex_sym = f"\\{p_name}" if p_name.lower() in _GREEK_LETTERS else p_name
                vals = [float(self.grad_aggregates[ak][k]) for ak in agg_keys]
                x_pos = indices - 0.4 + (k + 0.5) * bar_width
                ax2.bar(x_pos, vals, width=bar_width, label=f"$\\theta = {latex_sym}$", color=color, alpha=0.85)

            ax2.axhline(0.0, color="black", lw=0.8, ls="--")
            ax2.set_xticks(indices)
            ax2.set_xticklabels([f"Aggregate ${ak}^*$" for ak in agg_keys])
            ax2.set_title("General Equilibrium Sensitivities $\\nabla_\\theta \\mathbf{Y}^*$")
            ax2.set_ylabel("Elasticity / Sensitivity")
            ax2.legend(loc="best", frameon=True)
            ax2.grid(True, alpha=0.3, axis="y")
        else:
            # Fallback: Plot singular value spectrum of J_c
            try:
                s_vals = svd(self.jacobian_resid_c, compute_uv=False)
                ax2.semilogy(np.arange(1, len(s_vals) + 1), s_vals, "o-", color="tab:blue", lw=1.8)
                ax2.set_title(f"Singular Value Spectrum of $J_c$ (cond={self.condition_number:.1e})")
                ax2.set_xlabel("Singular Value Index")
                ax2.set_ylabel("Singular Value $\\sigma_i$")
                ax2.grid(True, alpha=0.3)
            except Exception:
                ax2.text(0.5, 0.5, "Aggregate sensitivities not available", ha="center", va="center")

        if show:
            plt.show()
        return fig


# ---------------------------------------------------------------------------
# Core Algorithmic Functions: Residual Evaluation & Numerical Jacobians
# ---------------------------------------------------------------------------

def _extract_coefficients(solution: Any) -> np.ndarray:
    """Extract candidate solution coefficients c* from solution object."""
    if hasattr(solution, "coefficients"):
        return np.asarray(solution.coefficients, dtype=np.float64).copy()
    if hasattr(solution, "nodal_values"):
        return np.asarray(solution.nodal_values, dtype=np.float64).copy()
    if hasattr(solution, "policy_c"):
        return np.asarray(solution.policy_c, dtype=np.float64).ravel().copy()
    if hasattr(solution, "household_solution"):
        return _extract_coefficients(solution.household_solution)
    raise ValueError(f"Cannot extract coefficients from solution of type {type(solution)}")


def _extract_param_names(problem: Any, params: Optional[Sequence[str]] = None, solution: Any = None) -> List[str]:
    """Determine ordered list of parameter names to differentiate."""
    if params is not None:
        return [str(p) for p in params]

    # Inspect problem parameters
    names = []
    if hasattr(problem, "params") and isinstance(problem.params, dict):
        for k in problem.params.keys():
            names.append(str(k))
    elif problem is not None and hasattr(problem, "alpha") and hasattr(problem, "gamma"):
        for k in ("alpha", "beta", "delta", "gamma"):
            if hasattr(problem, k):
                names.append(k)
    elif solution is not None and hasattr(solution, "metadata") and isinstance(solution.metadata, dict):
        meta_params = solution.metadata.get("params")
        if isinstance(meta_params, dict):
            for k in ("alpha", "beta", "delta", "gamma"):
                if k in meta_params and meta_params[k] is not None:
                    names.append(k)
    if hasattr(problem, "beta"):
        if "beta" not in names:
            names.append("beta")

    # Default fallback set for dynamic economic models
    default_order = ["alpha", "beta", "delta", "sigma", "gamma", "z", "A", "rho"]
    final_names = [p for p in default_order if p in names]
    for p in names:
        if p not in final_names:
            final_names.append(p)

    if not final_names:
        final_names = ["alpha", "beta", "delta", "sigma"]
    return final_names


def _evaluate_residual_system(
    solution: Any,
    problem: Any,
    coefficients: np.ndarray,
    params: Optional[Dict[str, float]] = None,
    beta: Optional[float] = None,
    residual_fn: Optional[Callable] = None,
    backend: str = "numpy",
) -> np.ndarray:
    """Evaluate continuous residual vector R(c; theta) at given coefficients and parameters."""
    if residual_fn is not None:
        p_dict = dict(params or {})
        if beta is not None:
            p_dict["beta"] = float(beta)
        try:
            return np.asarray(residual_fn(coefficients, p_dict), dtype=np.float64)
        except TypeError:
            return np.asarray(residual_fn(coefficients), dtype=np.float64)

    curr_params = dict(getattr(problem, "params", {}) if problem is not None else {})
    if params is not None:
        curr_params.update(params)

    curr_beta = float(beta if beta is not None else getattr(problem, "beta", 0.96))
    if "beta" in curr_params or beta is not None:
        curr_params["beta"] = curr_beta

    # 1. CollocationProblem (Chebyshev)
    if solution is not None and hasattr(solution, "basis") and hasattr(solution.basis, "n_dims"):
        from puremacro.vfi.collocation import _evaluate_euler_residual

        prob_eval: Any = replace(problem, params=curr_params, beta=curr_beta) if problem is not None else problem
        nodes = solution.basis.nodes()
        return _evaluate_euler_residual(prob_eval, solution.basis, coefficients, nodes, backend=backend)

    # 2. FEMProblem (Piecewise Linear Elements)
    if solution is not None and hasattr(solution, "mesh"):
        mesh = solution.mesh
        dim_nodes = mesh.dim_nodes[0]
        nodes = mesh.nodes
        n_elements = mesh.elements[0]
        n_nodes = mesh.n_nodes

        prob_eval = replace(problem, params=curr_params, beta=curr_beta) if problem is not None else problem
        eval_residual = prob_eval._create_residual_eval_fn()
        borrow_bound = prob_eval.borrowing_constraint
        comp_mode = str(prob_eval.options.get("complementarity", "fb"))
        quad_order = int(prob_eval.options.get("quad_order", 3))

        from puremacro.vfi.fem import gauss_legendre_quadrature

        xi, w = gauss_legendre_quadrature(quad_order)
        phi_left = (1.0 - xi) / 2.0
        phi_right = (1.0 + xi) / 2.0
        w_l = (w * phi_left)[None, :]
        w_r = (w * phi_right)[None, :]

        s_left = dim_nodes[:-1, None]
        s_right = dim_nodes[1:, None]
        he = s_right - s_left
        sq = (s_left + s_right) / 2.0 + (he / 2.0) * xi[None, :]

        y_clamped = np.clip(coefficients, dim_nodes[0], dim_nodes[-1])
        y_eff = np.maximum(y_clamped, float(borrow_bound)) if borrow_bound is not None else y_clamped

        if prob_eval.projection == "galerkin" and mesh.dim == 1:
            spq = y_eff[:-1, None] * phi_left[None, :] + y_eff[1:, None] * phi_right[None, :]
            if borrow_bound is not None:
                spq = np.maximum(spq, float(borrow_bound))
            snq = np.interp(spq.ravel(), dim_nodes, y_eff).reshape(n_elements, len(xi))
            if borrow_bound is not None:
                snq = np.maximum(snq, float(borrow_bound))

            res_q = eval_residual(sq, spq, snq)
            c_left = ((he / 2.0) * np.sum(res_q * w_l, axis=1, keepdims=True)).ravel()
            c_right = ((he / 2.0) * np.sum(res_q * w_r, axis=1, keepdims=True)).ravel()

            r = np.zeros(n_nodes, dtype=np.float64)
            np.add.at(r, np.arange(n_elements), c_left)
            np.add.at(r, np.arange(1, n_elements + 1), c_right)

        elif prob_eval.projection == "collocation" and mesh.dim == 1:
            sp = y_eff
            sn = np.interp(sp, dim_nodes, y_eff)
            r = eval_residual(dim_nodes, sp, sn)
        else:
            sp = y_eff
            sn = mesh.evaluate_basis(sp) @ y_eff
            r = eval_residual(mesh.nodes, sp, sn)

        if borrow_bound is not None:
            a = coefficients - float(borrow_bound)
            b_val = r
            if comp_mode == "fb":
                return a + b_val - np.sqrt(a**2 + b_val**2 + 1e-12)
            return np.minimum(a, b_val)
        return r

    # 3. SplineCollocationProblem (Cubic or Schumaker Spline)
    if solution is not None and hasattr(solution, "basis") and (
        hasattr(solution.basis, "nodes") or hasattr(solution.basis, "x")
    ):
        from puremacro.vfi.splines import _evaluate_euler_residual_spline

        prob_eval = replace(problem, params=curr_params, beta=curr_beta) if problem is not None else problem
        basis = solution.basis
        if hasattr(basis, "nodes"):
            nodes = basis.nodes()
        elif hasattr(basis, "x"):
            nodes = basis.x
        else:
            a, b = prob_eval.domain
            nodes = np.linspace(a, b, len(coefficients))
        return _evaluate_euler_residual_spline(prob_eval, basis, coefficients, nodes, backend=backend)

    raise TypeError(f"Unsupported problem/solution type for residual evaluation: {type(problem)}")


def _compute_residual_jacobian_c(
    solution: Any,
    problem: Any,
    c_star: np.ndarray,
    params: Dict[str, float],
    beta: float,
    residual_fn: Optional[Callable] = None,
    step: float = 1e-6,
    backend: str = "numpy",
) -> np.ndarray:
    """Evaluate J_c = \nabla_c R(c^*; \theta) using central finite differences."""
    N = len(c_star)
    J_c = np.zeros((N, N), dtype=np.float64)

    for j in range(N):
        c_plus = c_star.copy()
        c_minus = c_star.copy()
        c_plus[j] += step
        c_minus[j] -= step

        r_plus = _evaluate_residual_system(
            solution, problem, c_plus, params=params, beta=beta, residual_fn=residual_fn, backend=backend
        )
        r_minus = _evaluate_residual_system(
            solution, problem, c_minus, params=params, beta=beta, residual_fn=residual_fn, backend=backend
        )
        J_c[:, j] = (r_plus - r_minus) / (2.0 * step)

    return J_c


def _compute_residual_jacobian_theta(
    solution: Any,
    problem: Any,
    c_star: np.ndarray,
    param_names: List[str],
    base_params: Dict[str, float],
    base_beta: float,
    residual_fn: Optional[Callable] = None,
    step: float = 1e-5,
    backend: str = "numpy",
) -> np.ndarray:
    """Evaluate J_\theta = \nabla_\theta R(c^*; \theta) using central finite differences."""
    N = len(c_star)
    p = len(param_names)
    J_theta = np.zeros((N, p), dtype=np.float64)

    for k, p_name in enumerate(param_names):
        if p_name == "beta":
            r_plus = _evaluate_residual_system(
                solution, problem, c_star, params=base_params, beta=base_beta + step, residual_fn=residual_fn, backend=backend
            )
            r_minus = _evaluate_residual_system(
                solution, problem, c_star, params=base_params, beta=base_beta - step, residual_fn=residual_fn, backend=backend
            )
        else:
            p_plus = dict(base_params)
            p_minus = dict(base_params)
            p_plus[p_name] = p_plus.get(p_name, 1.0) + step
            p_minus[p_name] = p_minus.get(p_name, 1.0) - step

            r_plus = _evaluate_residual_system(
                solution, problem, c_star, params=p_plus, beta=base_beta, residual_fn=residual_fn, backend=backend
            )
            r_minus = _evaluate_residual_system(
                solution, problem, c_star, params=p_minus, beta=base_beta, residual_fn=residual_fn, backend=backend
            )

        J_theta[:, k] = (r_plus - r_minus) / (2.0 * step)

    return J_theta


def _solve_ift_linear_system(
    J_c: np.ndarray,
    J_theta: np.ndarray,
    cond_max: float = 1e12,
    rcond: float = 1e-12,
    regularization: float = 1e-8,
    solver: str = "auto",
) -> Tuple[np.ndarray, float, str]:
    r"""Solve J_c X = - J_\theta using single LU factorization with robust SVD/Tikhonov fallback.

    Parameters
    ----------
    J_c : np.ndarray
        Residual Jacobian with respect to coefficients, shape (N, N).
    J_theta : np.ndarray
        Residual Jacobian with respect to parameters, shape (N, p).
    cond_max : float, default 1e12
        Maximum allowed condition number before triggering regularized fallback.
    rcond : float, default 1e-12
        Cutoff for small singular values in SVD pseudoinverse.
    regularization : float, default 1e-8
        Tikhonov damping factor \lambda.
    solver : {"auto", "lu", "tikhonov", "svd"}, default "auto"
        Direct solver choice or automatic fallback hierarchy.

    Returns
    -------
    grad_c : np.ndarray
        Parameter Jacobian \nabla_\theta c^* of shape (N, p).
    cond_num : float
        Condition number of J_c.
    method : str
        Solver method utilized ('LU Factorization', 'Tikhonov Regularization', 'Truncated SVD').
    """
    try:
        cond_num = float(np.linalg.cond(J_c))
    except Exception:
        cond_num = np.inf

    rhs = -J_theta
    solver_mode = str(solver).strip().lower()

    # 1. Primary: Single LU Factorization (O(p N^2) solve)
    if solver_mode in ("auto", "lu") and cond_num < cond_max:
        try:
            lu, piv = lu_factor(J_c)
            grad_c = lu_solve((lu, piv), rhs)
            return grad_c, cond_num, "LU Factorization"
        except Exception:
            pass  # Fall through to regularization

    # 2. Secondary: Tikhonov Regularization (J_c^T J_c + lambda I) X = J_c^T rhs
    if solver_mode == "tikhonov" or (solver_mode == "auto" and cond_num >= cond_max and regularization > 0.0):
        try:
            N = J_c.shape[0]
            reg_mat = J_c.T @ J_c + regularization * np.eye(N)
            if np.linalg.cond(reg_mat) < 1e14 or solver_mode == "tikhonov":
                reg_rhs = J_c.T @ rhs
                lu_reg, piv_reg = lu_factor(reg_mat)
                grad_c = lu_solve((lu_reg, piv_reg), reg_rhs)
                return grad_c, cond_num, f"Tikhonov Regularization (lambda={regularization:.1e})"
        except Exception:
            pass  # Fall through to SVD

    # 3. Tertiary: Truncated SVD Pseudoinverse
    U, S, Vt = svd(J_c, full_matrices=False)
    cutoff = rcond * S[0]
    S_inv = np.where(S > cutoff, 1.0 / S, 0.0)
    grad_c = (Vt.T * S_inv) @ (U.T @ rhs)
    return grad_c, cond_num, f"Truncated SVD (cutoff={cutoff:.1e})"


# ---------------------------------------------------------------------------
# Adjoint Stationary Distribution & Macroeconomic Aggregate Sensitivities
# ---------------------------------------------------------------------------

def _compute_heterogeneous_ift_gradients(
    solution: Any,
    problem: Any = None,
    params: Optional[Sequence[str]] = None,
    h: float = 1e-5,
    step_c: float = 1e-6,
    cond_max: float = 1e12,
    backend: str = "numpy",
    **kwargs: Any,
) -> AnalyticGradientResult:
    """Compute coupled household-IFT, tangent-KFE, and GE market-clearing sensitivities."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    from puremacro.vfi.continuous_distribution import (
        build_continuous_transition_matrix,
        young_lottery_weights,
    )
    from puremacro.vfi.discretize import markov_stationary, tauchen

    t0 = time.perf_counter()
    param_names = _extract_param_names(problem, params, solution=solution)
    p = len(param_names)

    dist = solution.distribution
    hh = getattr(solution, "household_solution", None)
    K_hist = np.asarray(dist.asset_grid, dtype=np.float64)
    N_k = len(K_hist)

    # Case 1: Canonical Aiyagari Continuous Equilibrium with Endogenous Grid Method household
    if hh is not None and hasattr(hh, "policy_c") and hasattr(hh, "z_grid") and hasattr(hh, "P_z"):
        meta_params = dict(solution.metadata.get("params", {}) if hasattr(solution, "metadata") and isinstance(solution.metadata, dict) else {})
        base_p: Dict[str, float] = {
            "beta": float(meta_params.get("beta", getattr(problem, "beta", 0.96))),
            "gamma": float(meta_params.get("gamma", getattr(problem, "gamma", 2.0))),
            "alpha": float(meta_params.get("alpha", getattr(problem, "alpha", 0.36))),
            "delta": float(meta_params.get("delta", getattr(problem, "delta", 0.08))),
            "rho_z": float(meta_params.get("rho_z", getattr(problem, "rho_z", 0.90)) or 0.90),
            "sigma_z": float(meta_params.get("sigma_z", getattr(problem, "sigma_z", 0.20)) or 0.20),
        }
        if hasattr(problem, "params") and isinstance(problem.params, dict):
            for k_p, v_p in problem.params.items():
                if v_p is not None:
                    base_p[str(k_p)] = float(v_p)
            if "sigma" in problem.params and "gamma" not in problem.params:
                base_p["gamma"] = float(problem.params["sigma"])

        c_star = np.asarray(hh.policy_c, dtype=np.float64).copy()
        n_a, n_z = c_star.shape
        a_max = float(meta_params.get("a_max", K_hist[-1]))
        a_grid_dense = a_max * (np.linspace(0.0, 1.0, n_a) ** 1.5)
        z_base = np.asarray(hh.z_grid, dtype=np.float64)
        P_base = np.asarray(hh.P_z, dtype=np.float64)
        r_star = float(solution.r)
        w_star = float(solution.w)
        L_star = float(solution.L)

        def _coleman_step(
            c_mat: np.ndarray,
            r_val: float,
            b_val: float,
            g_val: float,
            w_val: float,
            z_g: np.ndarray,
            P_g: np.ndarray,
        ) -> np.ndarray:
            EMu = b_val * (1.0 + r_val) * (np.maximum(c_mat, 1e-12) ** (-g_val) @ P_g.T)
            c_endo = EMu ** (-1.0 / g_val)
            c_new = np.zeros_like(c_mat)
            for m in range(n_z):
                a_endo = (c_endo[:, m] + a_grid_dense - w_val * z_g[m]) / (1.0 + r_val)
                c_interp = np.interp(a_grid_dense, a_endo, c_endo[:, m])
                binds = a_grid_dense < a_endo[0]
                c_interp[binds] = (1.0 + r_val) * a_grid_dense[binds] + w_val * z_g[m]
                c_new[:, m] = c_interp
            return c_new

        # 1. Assemble household Coleman Jacobian J_c = \nabla_c (T_EGM(c) - c)
        N_c = c_star.size
        J_c = np.zeros((N_c, N_c), dtype=np.float64)
        c_flat = c_star.ravel()
        b0 = base_p["beta"]
        g0 = base_p["gamma"]
        for j in range(N_c):
            cp = c_flat.copy()
            cm = c_flat.copy()
            cp[j] += step_c
            cm[j] -= step_c
            rp = (_coleman_step(cp.reshape(n_a, n_z), r_star, b0, g0, w_star, z_base, P_base) - cp.reshape(n_a, n_z)).ravel()
            rm = (_coleman_step(cm.reshape(n_a, n_z), r_star, b0, g0, w_star, z_base, P_base) - cm.reshape(n_a, n_z)).ravel()
            J_c[:, j] = (rp - rm) / (2.0 * step_c)

        lu_c, piv_c = lu_factor(J_c)
        # Polish c_star to machine precision at (r_star, base_p)
        for _ in range(2):
            r0 = (_coleman_step(c_star, r_star, b0, g0, w_star, z_base, P_base) - c_star).ravel()
            if float(np.max(np.abs(r0))) < 1e-14:
                break
            c_star -= lu_solve((lu_c, piv_c), r0).reshape(n_a, n_z)

        # 2. Factor sparse KFE operator (I - T^T) once at equilibrium
        ap_dense = np.maximum((1.0 + r_star) * a_grid_dense[:, None] + w_star * z_base[None, :] - c_star, 0.0)
        ap_hist = np.column_stack([np.interp(K_hist, a_grid_dense, ap_dense[:, m]) for m in range(n_z)])
        T_mat = build_continuous_transition_matrix(ap_hist, K_hist, P_base)
        N_tot = N_k * n_z
        A_kfe = (sp.eye(N_tot, format="csc") - T_mat.T.tocsc()).tolil()
        A_kfe[0, :] = 1.0
        kfe_lu = spla.splu(A_kfe.tocsc())
        rhs_0 = np.zeros(N_tot, dtype=np.float64)
        rhs_0[0] = 1.0
        mu_mat = kfe_lu.solve(rhs_0).reshape(N_k, n_z)

        j_lo, w_lo, _ = young_lottery_weights(ap_hist, K_hist)
        dk_bracket = K_hist[j_lo + 1] - K_hist[j_lo]
        interior_hist = (ap_hist > K_hist[0]) & (ap_hist < K_hist[-1])
        unconstrained = ((1.0 + r_star) * a_grid_dense[:, None] + w_star * z_base[None, :] - c_star) > 0.0

        def _solve_dmu(dap_hist_val: np.ndarray, dP_z_val: Optional[np.ndarray] = None) -> np.ndarray:
            d_pdf_half = np.zeros((N_k, n_z), dtype=np.float64)
            dw_lo = np.where(interior_hist, -dap_hist_val / dk_bracket, 0.0)
            for m in range(n_z):
                np.add.at(d_pdf_half[:, m], j_lo[:, m], mu_mat[:, m] * dw_lo[:, m])
                np.add.at(d_pdf_half[:, m], j_lo[:, m] + 1, -mu_mat[:, m] * dw_lo[:, m])
            rhs_mat = d_pdf_half @ P_base
            if dP_z_val is not None:
                pdf_half = np.zeros((N_k, n_z), dtype=np.float64)
                for m in range(n_z):
                    np.add.at(pdf_half[:, m], j_lo[:, m], mu_mat[:, m] * w_lo[:, m])
                    np.add.at(pdf_half[:, m], j_lo[:, m] + 1, mu_mat[:, m] * (1.0 - w_lo[:, m]))
                rhs_mat += pdf_half @ dP_z_val
            rhs_vec = rhs_mat.ravel()
            rhs_vec[0] = 0.0
            return kfe_lu.solve(rhs_vec).reshape(N_k, n_z)

        def _eval_env(r_val: float, p_dict: Dict[str, float]):
            b_v = p_dict["beta"]
            g_v = p_dict["gamma"]
            a_v = p_dict["alpha"]
            d_v = p_dict["delta"]
            rz_v = p_dict["rho_z"]
            sz_v = p_dict["sigma_z"]
            if rz_v == base_p["rho_z"] and sz_v == base_p["sigma_z"]:
                z_g, P_g, L_g = z_base, P_base, L_star
            else:
                lz, P_g = tauchen(n_z, rz_v, sz_v)
                z_g = np.exp(lz)
                L_g = float(np.sum(markov_stationary(P_g) * z_g))
            kl = ((r_val + d_v) / a_v) ** (1.0 / (a_v - 1.0))
            w_v = (1.0 - a_v) * (kl ** a_v)
            Kd_v = L_g * kl
            R_v = (_coleman_step(c_star, r_val, b_v, g_v, w_v, z_g, P_g) - c_star).ravel()
            return R_v, w_v, Kd_v, L_g, z_g, P_g

        # 3. Partial derivatives with respect to interest rate r at r*
        Rp_r, wp_r, Kdp_r, _, _, _ = _eval_env(r_star + h, base_p)
        Rm_r, wm_r, Kdm_r, _, _, _ = _eval_env(r_star - h, base_p)
        dR_dr = (Rp_r - Rm_r) / (2.0 * h)
        dw_dr = (wp_r - wm_r) / (2.0 * h)
        dKd_dr = (Kdp_r - Kdm_r) / (2.0 * h)
        dc_dr = lu_solve((lu_c, piv_c), -dR_dr).reshape(n_a, n_z)
        dap_dense_dr = np.where(unconstrained, a_grid_dense[:, None] + dw_dr * z_base[None, :] - dc_dr, 0.0)
        dap_hist_dr = np.column_stack([np.interp(K_hist, a_grid_dense, dap_dense_dr[:, m]) for m in range(n_z)])
        dmu_dr = _solve_dmu(dap_hist_dr)
        dKs_dr = float(np.sum(dmu_dr.sum(axis=1) * K_hist))
        dE_dr = dKs_dr - dKd_dr
        if abs(dE_dr) < 1e-12:
            dE_dr = 1e-12 if dE_dr >= 0.0 else -1e-12

        # 4. Partial derivatives with respect to each parameter and total GE IFT coupling
        J_theta_total = np.zeros((N_c, p), dtype=np.float64)
        grad_K = np.zeros(p, dtype=np.float64)
        grad_C = np.zeros(p, dtype=np.float64)
        grad_r = np.zeros(p, dtype=np.float64)
        grad_w = np.zeros(p, dtype=np.float64)
        grad_mu = np.zeros((N_k, n_z, p), dtype=np.float64)
        grad_policy_a = np.zeros((N_k, n_z, p), dtype=np.float64)
        grad_policy_c = np.zeros((n_a, n_z, p), dtype=np.float64)

        K_star = float(np.sum(mu_mat.sum(axis=1) * K_hist))
        alpha_0 = base_p["alpha"]
        delta_0 = base_p["delta"]
        Y_star = (K_star ** alpha_0) * (L_star ** (1.0 - alpha_0))

        for k_idx, p_name in enumerate(param_names):
            canon_name = "gamma" if p_name == "sigma" else p_name
            pp = dict(base_p)
            pm = dict(base_p)
            pp[canon_name] = pp.get(canon_name, 1.0) + h
            pm[canon_name] = pm.get(canon_name, 1.0) - h

            Rp_t, wp_t, Kdp_t, Lp_t, zp_t, Pp_t = _eval_env(r_star, pp)
            Rm_t, wm_t, Kdm_t, Lm_t, zm_t, Pm_t = _eval_env(r_star, pm)
            dR_dt = (Rp_t - Rm_t) / (2.0 * h)
            dw_dt_part = (wp_t - wm_t) / (2.0 * h)
            dKd_dt_part = (Kdp_t - Kdm_t) / (2.0 * h)
            dL_dt = (Lp_t - Lm_t) / (2.0 * h)
            dz_dt = (zp_t - zm_t) / (2.0 * h)
            dP_dt = (Pp_t - Pm_t) / (2.0 * h) if not np.array_equal(Pp_t, Pm_t) else None

            dc_dt_part = lu_solve((lu_c, piv_c), -dR_dt).reshape(n_a, n_z)
            dap_dense_dt_part = np.where(
                unconstrained,
                dw_dt_part * z_base[None, :] + w_star * dz_dt[None, :] - dc_dt_part,
                0.0,
            )
            dap_hist_dt_part = np.column_stack(
                [np.interp(K_hist, a_grid_dense, dap_dense_dt_part[:, m]) for m in range(n_z)]
            )
            dmu_dt_part = _solve_dmu(dap_hist_dt_part, dP_z_val=dP_dt)
            dKs_dt_part = float(np.sum(dmu_dt_part.sum(axis=1) * K_hist))

            # General equilibrium IFT: dr*/dtheta_k = - (dKs/dtheta_k - dKd/dtheta_k) / (dKs/dr - dKd/dr)
            dr_dt = -(dKs_dt_part - dKd_dt_part) / dE_dr
            dK_dt = dKs_dt_part + dKs_dr * dr_dt
            dw_dt = dw_dt_part + dw_dr * dr_dt

            # Aggregate consumption C* = K*^alpha L*^(1-alpha) - delta K*
            dY_dt = (
                alpha_0 * (K_star ** (alpha_0 - 1.0)) * (L_star ** (1.0 - alpha_0)) * dK_dt
                + (1.0 - alpha_0) * (K_star ** alpha_0) * (L_star ** (-alpha_0)) * dL_dt
            )
            if canon_name == "alpha":
                dY_dt += Y_star * np.log(max(K_star / max(L_star, 1e-12), 1e-12))
            dC_dt = dY_dt - delta_0 * dK_dt
            if canon_name == "delta":
                dC_dt -= K_star

            grad_r[k_idx] = dr_dt
            grad_K[k_idx] = dK_dt
            grad_w[k_idx] = dw_dt
            grad_C[k_idx] = dC_dt
            grad_mu[:, :, k_idx] = dmu_dt_part + dmu_dr * dr_dt
            grad_policy_a[:, :, k_idx] = dap_hist_dt_part + dap_hist_dr * dr_dt
            grad_policy_c[:, :, k_idx] = dc_dt_part + dc_dr * dr_dt
            J_theta_total[:, k_idx] = dR_dt + dR_dr * dr_dt

        grad_c, cond_num, solver_method = _solve_ift_linear_system(
            J_c,
            J_theta_total,
            cond_max=cond_max,
            rcond=kwargs.get("rcond", 1e-12),
            regularization=kwargs.get("reg", 1e-8),
            solver=str(kwargs.get("solver", "auto")),
        )

        elapsed = time.perf_counter() - t0
        grad_aggs = {"K": grad_K, "C": grad_C, "r": grad_r, "w": grad_w, "mu": grad_mu}
        metadata = {
            "solver_method": solver_method,
            "step_c": step_c,
            "step_theta": h,
            "backend": backend,
            "a_grid": K_hist,
            "a_grid_dense": a_grid_dense,
            "z_grid": z_base,
            "grad_policy_a": grad_policy_a,
            "grad_policy_c": grad_policy_c,
            "grad_mu": grad_mu,
            "grad_r": grad_r,
            "grad_w": grad_w,
            "grad_K": grad_K,
            "grad_C": grad_C,
            **kwargs,
        }
        return AnalyticGradientResult(
            grad_coefficients=grad_c,
            grad_aggregates=grad_aggs,
            param_names=param_names,
            jacobian_resid_c=J_c,
            jacobian_resid_theta=J_theta_total,
            condition_number=cond_num,
            elapsed_time=elapsed,
            solution=solution,
            problem=problem,
            metadata=metadata,
        )

    # Case 2: Continuous Projection Household coupled with Stationary Distribution
    target_sol = hh if hh is not None else solution
    target_prob = problem if problem is not None else getattr(target_sol, "problem", None)
    c_star = _extract_coefficients(target_sol)
    base_params = dict(getattr(target_prob, "params", {}) if target_prob is not None else {})
    base_beta = float(getattr(target_prob, "beta", 0.96) if target_prob is not None else 0.96)

    J_c = _compute_residual_jacobian_c(
        target_sol, target_prob, c_star, params=base_params, beta=base_beta, step=step_c, backend=backend
    )
    J_theta = _compute_residual_jacobian_theta(
        target_sol,
        target_prob,
        c_star,
        param_names=param_names,
        base_params=base_params,
        base_beta=base_beta,
        step=h,
        backend=backend,
    )
    grad_c, cond_num, solver_method = _solve_ift_linear_system(
        J_c,
        J_theta,
        cond_max=cond_max,
        rcond=kwargs.get("rcond", 1e-12),
        regularization=kwargs.get("reg", 1e-8),
        solver=str(kwargs.get("solver", "auto")),
    )

    # Propagate policy Jacobian through the stationary distribution KFE
    temp_res = AnalyticGradientResult(
        grad_coefficients=grad_c,
        grad_aggregates={},
        param_names=param_names,
        jacobian_resid_c=J_c,
        jacobian_resid_theta=J_theta,
        condition_number=cond_num,
        elapsed_time=0.0,
        solution=target_sol,
        problem=target_prob,
    )
    P_base = dist.shock_transition
    z_base = dist.shock_grid
    n_z = P_base.shape[0] if P_base is not None else 1
    from puremacro.vfi.continuous_distribution import _evaluate_policy

    ap_hist = _evaluate_policy(target_sol, K_hist, shock_grid=z_base, shock_transition=P_base)
    if ap_hist.ndim == 1:
        ap_hist = ap_hist[:, None]
    T_mat = build_continuous_transition_matrix(
        ap_hist if P_base is not None else ap_hist.ravel(), K_hist, P_base
    )
    N_tot = N_k * n_z
    A_kfe = (sp.eye(N_tot, format="csc") - T_mat.T.tocsc()).tolil()
    A_kfe[0, :] = 1.0
    kfe_lu = spla.splu(A_kfe.tocsc())
    rhs_0 = np.zeros(N_tot, dtype=np.float64)
    rhs_0[0] = 1.0
    mu_mat = kfe_lu.solve(rhs_0).reshape(N_k, n_z)
    j_lo, _, _ = young_lottery_weights(ap_hist, K_hist)
    dk_bracket = K_hist[j_lo + 1] - K_hist[j_lo]
    interior_hist = (ap_hist > K_hist[0]) & (ap_hist < K_hist[-1])

    pol_grad_hist = temp_res.policy_gradient(K_hist)  # (N_k, p)
    grad_K = np.zeros(p, dtype=np.float64)
    grad_C = np.zeros(p, dtype=np.float64)
    grad_r = np.zeros(p, dtype=np.float64)
    grad_w = np.zeros(p, dtype=np.float64)
    grad_mu = np.zeros((N_k, n_z, p), dtype=np.float64)

    alpha_0 = float(base_params.get("alpha", 0.36))
    delta_0 = float(base_params.get("delta", 0.08))
    K_star = float(dist.mean())

    for k_idx, p_name in enumerate(param_names):
        dap_k = np.tile(pol_grad_hist[:, k_idx : k_idx + 1], (1, n_z))
        d_pdf_half = np.zeros((N_k, n_z), dtype=np.float64)
        dw_lo = np.where(interior_hist, -dap_k / dk_bracket, 0.0)
        for m in range(n_z):
            np.add.at(d_pdf_half[:, m], j_lo[:, m], mu_mat[:, m] * dw_lo[:, m])
            np.add.at(d_pdf_half[:, m], j_lo[:, m] + 1, -mu_mat[:, m] * dw_lo[:, m])
        rhs_mat = d_pdf_half @ P_base if P_base is not None else d_pdf_half
        rhs_vec = rhs_mat.ravel()
        rhs_vec[0] = 0.0
        dmu_k = kfe_lu.solve(rhs_vec).reshape(N_k, n_z)
        grad_mu[:, :, k_idx] = dmu_k
        dk_val = float(np.sum(dmu_k.sum(axis=1) * K_hist))
        grad_K[k_idx] = dk_val
        grad_C[k_idx] = (alpha_0 * (K_star ** (alpha_0 - 1.0)) - delta_0) * dk_val
        grad_r[k_idx] = alpha_0 * (alpha_0 - 1.0) * (K_star ** (alpha_0 - 2.0)) * dk_val
        grad_w[k_idx] = (1.0 - alpha_0) * alpha_0 * (K_star ** (alpha_0 - 1.0)) * dk_val

    elapsed = time.perf_counter() - t0
    grad_aggs = {"K": grad_K, "C": grad_C, "r": grad_r, "w": grad_w, "mu": grad_mu}
    return AnalyticGradientResult(
        grad_coefficients=grad_c,
        grad_aggregates=grad_aggs,
        param_names=param_names,
        jacobian_resid_c=J_c,
        jacobian_resid_theta=J_theta,
        condition_number=cond_num,
        elapsed_time=elapsed,
        solution=solution,
        problem=problem,
        metadata={"solver_method": solver_method, "grad_mu": grad_mu, **kwargs},
    )


def _compute_aggregate_gradients(
    solution: Any,
    problem: Any,
    grad_coefficients: np.ndarray,
    param_names: List[str],
    ift_result_proxy: Optional[Any] = None,
) -> Dict[str, Any]:
    """Compute representative-agent or heterogeneous-agent aggregate sensitivities."""
    p = len(param_names)
    grad_K = np.zeros(p, dtype=np.float64)
    grad_C = np.zeros(p, dtype=np.float64)
    grad_r = np.zeros(p, dtype=np.float64)
    grad_w = np.zeros(p, dtype=np.float64)
    grad_mu: Any = None

    # Case A: Heterogeneous-Agent General Equilibrium (AiyagariContinuousEquilibrium)
    if hasattr(solution, "distribution") and hasattr(solution.distribution, "asset_grid"):
        het_res = _compute_heterogeneous_ift_gradients(solution, problem=problem, params=param_names)
        return het_res.grad_aggregates

    # Case B: Representative Agent Continuous Projection Model
    if solution is not None and hasattr(solution, "policy"):
        params_dict = dict(getattr(problem, "params", {}) if problem is not None else {})
        alpha = float(params_dict.get("alpha", 0.36))
        delta = float(params_dict.get("delta", 1.0))
        z_raw = params_dict.get("z", params_dict.get("A", 1.0))
        z = float(1.0 if z_raw is None else z_raw)

        # Find continuous steady state k* where g(k*) = k*
        k_min = 0.1
        k_max = 5.0
        if problem is not None and hasattr(problem, "domain"):
            dom = problem.domain
            if isinstance(dom[0], (tuple, list)):
                k_min, k_max = float(dom[0][0]), float(dom[0][1])
            else:
                k_min, k_max = float(dom[0]), float(dom[1])

        # Midpoint bracket
        s_mid = 0.5 * (k_min + k_max)
        try:
            # Check signs at endpoints
            f_low = solution.policy(k_min + 1e-4) - (k_min + 1e-4)
            f_high = solution.policy(k_max - 1e-4) - (k_max - 1e-4)
            if f_low * f_high <= 0:
                k_star = float(brentq(lambda k: solution.policy(k) - k, k_min + 1e-4, k_max - 1e-4))
            else:
                k_star = float(s_mid)
        except Exception:
            k_star = float(s_mid)

        # Policy derivative g'(k*) via central difference
        eps_k = 1e-6
        gp_kstar = float((solution.policy(k_star + eps_k) - solution.policy(k_star - eps_k)) / (2.0 * eps_k))
        denom = 1.0 - gp_kstar
        if abs(denom) < 1e-5:
            denom = 1e-5 if denom >= 0 else -1e-5

        # Evaluate policy gradient at k*
        for k_idx, p_name in enumerate(param_names):
            if hasattr(solution, "basis") and hasattr(solution.basis, "evaluate"):
                Phi_kstar = solution.basis.evaluate(np.array([k_star]))
                dg_kstar = float((Phi_kstar @ grad_coefficients[:, k_idx]).ravel()[0])
            elif hasattr(solution, "mesh"):
                dim_nodes = solution.mesh.dim_nodes[0]
                dg_kstar = float(np.interp(k_star, dim_nodes, grad_coefficients[:, k_idx]))
            elif hasattr(solution, "basis") and hasattr(solution.basis, "interpolate"):
                dg_kstar = float(solution.basis.interpolate(grad_coefficients[:, k_idx], np.array([k_star]), deriv=0)[0])
            else:
                dg_kstar = float(grad_coefficients[0, k_idx])

            # Continuous steady state response: dk*/dtheta = (dg/dtheta) / (1 - g'(k*))
            dk_star = dg_kstar / denom
            grad_K[k_idx] = dk_star

            # Output Y* = z (k*)^alpha
            dY_dk = alpha * z * (k_star ** (alpha - 1.0))
            dY_direct = 0.0
            if p_name == "alpha":
                dY_direct = z * (k_star**alpha) * np.log(max(k_star, 1e-12))
            elif p_name in ("z", "A"):
                dY_direct = k_star**alpha

            # Consumption C* = z (k*)^alpha - delta k*
            dC_direct = dY_direct
            if p_name == "delta":
                dC_direct -= k_star
            grad_C[k_idx] = (dY_dk - delta) * dk_star + dC_direct

            # Real interest rate r* = alpha z (k*)^{alpha - 1} - delta
            dr_dk = alpha * (alpha - 1.0) * z * (k_star ** (alpha - 2.0))
            dr_direct = 0.0
            if p_name == "alpha":
                dr_direct = z * (k_star ** (alpha - 1.0)) * (1.0 + alpha * np.log(max(k_star, 1e-12)))
            elif p_name == "delta":
                dr_direct = -1.0
            elif p_name in ("z", "A"):
                dr_direct = alpha * (k_star ** (alpha - 1.0))
            grad_r[k_idx] = dr_dk * dk_star + dr_direct

            # Real wage w* = (1 - alpha) z (k*)^alpha
            dw_dk = (1.0 - alpha) * alpha * z * (k_star ** (alpha - 1.0))
            dw_direct = 0.0
            if p_name == "alpha":
                dw_direct = -z * (k_star**alpha) + (1.0 - alpha) * z * (k_star**alpha) * np.log(max(k_star, 1e-12))
            elif p_name in ("z", "A"):
                dw_direct = (1.0 - alpha) * (k_star**alpha)
            grad_w[k_idx] = dw_dk * dk_star + dw_direct

    return {"K": grad_K, "C": grad_C, "r": grad_r, "w": grad_w, "mu": grad_mu}


# ---------------------------------------------------------------------------
# Public API Entry Points
# ---------------------------------------------------------------------------

def compute_ift_gradients(
    solution: Any,
    problem: Any = None,
    params: Optional[Sequence[str]] = None,
    h: float = 1e-5,
    step_c: float = 1e-6,
    residual_fn: Optional[Callable] = None,
    cond_max: float = 1e12,
    backend: str = "numpy",
    **kwargs: Any,
) -> AnalyticGradientResult:
    r"""Compute semi-analytic policy Jacobians via IFT and finite-difference residual Jacobians.

    Formulates the Implicit Function Theorem on continuous projection or heterogeneous-agent
    general equilibrium systems:

        \nabla_\theta c^*(\theta) = - [\nabla_c R(c^*; \theta)]^{-1} \nabla_\theta R(c^*; \theta)

    Parameters
    ----------
    solution : CollocationSolution, FEMSolution, SplineCollocationSolution, or AiyagariContinuousEquilibrium
        Converged continuous solution or heterogeneous-agent equilibrium container.
    problem : CollocationProblem, FEMProblem, SplineCollocationProblem, or AiyagariContinuousModel, optional
        Dynamic programming or general equilibrium problem definition.
    params : Sequence[str], optional
        Structural parameters to differentiate, e.g. ['alpha', 'beta', 'delta', 'sigma'].
        If None, automatically inspects model parameters.
    h : float, default 1e-5
        Step size for parameter differentiation \nabla_\theta R.
    step_c : float, default 1e-6
        Step size for coefficient Jacobian J_c = \nabla_c R.
    residual_fn : Callable, optional
        Custom residual evaluator R(c, params) -> np.ndarray.
    cond_max : float, default 1e12
        Condition number threshold for triggering regularized linear solves.
    backend : str, default "numpy"
        Compute acceleration backend.

    Returns
    -------
    result : AnalyticGradientResult
        Result container with .grad_coefficients, .policy_gradient(s), .grad_aggregates,
        and full presentation methods.
    """
    t0 = time.perf_counter()

    if hasattr(solution, "distribution") and hasattr(solution.distribution, "asset_grid") and residual_fn is None:
        return _compute_heterogeneous_ift_gradients(
            solution,
            problem=problem,
            params=params,
            h=h,
            step_c=step_c,
            cond_max=cond_max,
            backend=backend,
            **kwargs,
        )

    c_star = _extract_coefficients(solution)
    param_names = _extract_param_names(problem, params, solution=solution)

    base_params = dict(getattr(problem, "params", {}) if problem is not None else {})
    base_beta = float(getattr(problem, "beta", 0.96) if problem is not None else 0.96)

    # 1. Compute J_c = \nabla_c R(c^*; \theta)
    J_c = _compute_residual_jacobian_c(
        solution,
        problem,
        c_star,
        params=base_params,
        beta=base_beta,
        residual_fn=residual_fn,
        step=step_c,
        backend=backend,
    )

    # 2. Compute J_\theta = \nabla_\theta R(c^*; \theta)
    J_theta = _compute_residual_jacobian_theta(
        solution,
        problem,
        c_star,
        param_names=param_names,
        base_params=base_params,
        base_beta=base_beta,
        residual_fn=residual_fn,
        step=h,
        backend=backend,
    )

    # 3. Solve J_c X = - J_\theta via single LU factorization (or robust fallback)
    grad_c, cond_num, solver_method = _solve_ift_linear_system(
        J_c,
        J_theta,
        cond_max=cond_max,
        rcond=kwargs.get("rcond", 1e-12),
        regularization=kwargs.get("reg", 1e-8),
        solver=str(kwargs.get("solver", "auto")),
    )

    # 4. Compute macroeconomic aggregate sensitivities
    grad_aggs = _compute_aggregate_gradients(solution, problem, grad_c, param_names)

    elapsed = time.perf_counter() - t0

    metadata = {
        "solver_method": solver_method,
        "step_c": step_c,
        "step_theta": h,
        "backend": backend,
        **kwargs,
    }

    return AnalyticGradientResult(
        grad_coefficients=grad_c,
        grad_aggregates=grad_aggs,
        param_names=param_names,
        jacobian_resid_c=J_c,
        jacobian_resid_theta=J_theta,
        condition_number=cond_num,
        elapsed_time=elapsed,
        solution=solution,
        problem=problem,
        metadata=metadata,
    )


def policy_parameter_jacobian(
    solution: Any,
    problem: Any = None,
    s: Optional[Union[float, Sequence[float], np.ndarray]] = None,
    params: Optional[Sequence[str]] = None,
    h: float = 1e-5,
    ift_result: Optional[AnalyticGradientResult] = None,
    **kwargs: Any,
) -> np.ndarray:
    r"""Evaluate continuous policy function sensitivities \nabla_\theta g(s).

    Parameters
    ----------
    solution : CollocationSolution, FEMSolution, SplineCollocationSolution, or AiyagariContinuousEquilibrium
        Converged solution container.
    problem : CollocationProblem, FEMProblem, SplineCollocationProblem, or AiyagariContinuousModel, optional
        Dynamic model specification.
    s : float or np.ndarray
        Evaluation state coordinates.
    params : Sequence[str], optional
        Parameter names.
    h : float, default 1e-5
        Differentiation step.
    ift_result : AnalyticGradientResult, optional
        Precomputed IFT result to avoid re-factoring J_c.

    Returns
    -------
    np.ndarray
        Policy gradient of shape (p,) for scalar s, or (M, p) for array s.
    """
    if s is None and isinstance(problem, (int, float, list, tuple, np.ndarray)):
        s = problem
        problem = None
    if s is None:
        raise ValueError("Evaluation state coordinates 's' must be provided.")
    if ift_result is None:
        ift_result = compute_ift_gradients(solution, problem, params=params, h=h, **kwargs)
    return ift_result.policy_gradient(s)


def equilibrium_parameter_jacobian(
    solution: Any,
    problem: Optional[Any] = None,
    params: Optional[Sequence[str]] = None,
    h: float = 1e-5,
    ift_result: Optional[AnalyticGradientResult] = None,
    **kwargs: Any,
) -> Dict[str, np.ndarray]:
    r"""Compute parameter sensitivities of general equilibrium macro aggregates.

    Parameters
    ----------
    solution : CollocationSolution, FEMSolution, SplineCollocationSolution, or AiyagariContinuousEquilibrium
        Converged representative-agent or heterogeneous-agent general equilibrium solution.
    problem : Any, optional
        Problem specification (if available).
    params : Sequence[str], optional
        Parameter names to differentiate.
    h : float, default 1e-5
        Step size.
    ift_result : AnalyticGradientResult, optional
        Precomputed IFT result.

    Returns
    -------
    dict[str, np.ndarray]
        Sensitivities {"K": (p,), "C": (p,), "r": (p,), "w": (p,), "mu": ...}.
    """
    if ift_result is None:
        if problem is None and not hasattr(solution, "distribution"):
            problem = getattr(solution, "problem", getattr(solution, "household_solution", None))
        ift_result = compute_ift_gradients(solution, problem, params=params, h=h, **kwargs)
    return ift_result.grad_aggregates


def gmm_objective_and_gradient(
    theta_vals: Sequence[float],
    problem: Any,
    empirical_moments: Sequence[float],
    moment_fn: Optional[Callable] = None,
    weighting_matrix: Optional[np.ndarray] = None,
    param_names: Optional[Sequence[str]] = None,
    backend: str = "numpy",
    **kwargs: Any,
) -> Tuple[float, np.ndarray]:
    r"""Fast analytical GMM/SMM objective value and exact gradient for structural estimation.

    Evaluates the weighted distance objective:

        Q(\theta) = (m(\theta) - \hat{m})^\top W (m(\theta) - \hat{m})

    and its exact analytical gradient via the IFT Jacobian G = \nabla_\theta m(\theta):

        \nabla_\theta Q(\theta) = 2 G^\top W (m(\theta) - \hat{m}) \in \mathbb{R}^p

    Parameters
    ----------
    theta_vals : Sequence[float]
        Candidate parameter vector of length p.
    problem : CollocationProblem, FEMProblem, SplineCollocationProblem, or AiyagariContinuousModel
        Base problem template.
    empirical_moments : Sequence[float]
        Target empirical moments \hat{m} of length m.
    moment_fn : Callable, optional
        User function moment_fn(solution, problem) -> np.ndarray returning m(\theta).
        If None, uses macro aggregates [K*, C*, r*] matching empirical moment length.
    weighting_matrix : np.ndarray, optional
        Weighting matrix W of shape (m, m). Defaults to identity matrix.
    param_names : Sequence[str], optional
        Ordered names of structural parameters corresponding to theta_vals.
    backend : str, default "numpy"
        Compute acceleration backend.

    Returns
    -------
    Q_val : float
        Objective value Q(\theta).
    grad_Q : np.ndarray
        Exact analytical gradient vector of shape (p,).
    """
    theta_arr = np.asarray(theta_vals, dtype=np.float64)
    m_target = np.asarray(empirical_moments, dtype=np.float64)
    p = len(theta_arr)
    m = len(m_target)

    p_names = list(param_names) if param_names is not None else _extract_param_names(problem)[:p]
    if len(p_names) != p:
        p_names = [f"theta_{i}" for i in range(p)]

    # Update problem parameters
    if hasattr(problem, "params") and isinstance(problem.params, dict):
        p_dict = dict(getattr(problem, "params", {}))
        beta_val = getattr(problem, "beta", 0.96)
        for name, val in zip(p_names, theta_arr):
            if name == "beta":
                beta_val = float(val)
            else:
                p_dict[name] = float(val)
        prob_eval = replace(problem, params=p_dict, beta=beta_val)
    else:
        updates = {}
        p_dict = {}
        for name, val in zip(p_names, theta_arr):
            canon = "gamma" if name == "sigma" else name
            if hasattr(problem, canon):
                updates[canon] = float(val)
            p_dict[canon] = float(val)
        prob_eval = replace(problem, **updates) if updates else problem
        for k_attr in ("alpha", "beta", "delta", "gamma"):
            if hasattr(prob_eval, k_attr):
                p_dict.setdefault(k_attr, float(getattr(prob_eval, k_attr)))

    sol_eval = prob_eval.solve(backend=backend)

    # Compute IFT gradient result
    ift_res = compute_ift_gradients(sol_eval, prob_eval, params=p_names, backend=backend, **kwargs)

    # Evaluate model moments m(theta) and Jacobian G = dm/dtheta
    if moment_fn is not None:
        # User moment function
        m_eval = np.asarray(moment_fn(sol_eval, prob_eval), dtype=np.float64)
        # Numerical propagation through policy gradient if user function is arbitrary
        eps_step = 1e-5
        G = np.zeros((m, p), dtype=np.float64)
        for k in range(p):
            # Perturb along IFT gradient direction
            c_pert = _extract_coefficients(sol_eval) + eps_step * ift_res.grad_coefficients[:, k]
            # Replace coefficients
            sol_pert = replace(sol_eval, coefficients=c_pert) if hasattr(sol_eval, "coefficients") else sol_eval
            m_pert = np.asarray(moment_fn(sol_pert, prob_eval), dtype=np.float64)
            G[:, k] = (m_pert - m_eval) / eps_step
    else:
        # Default moments from macro aggregates: [K*, C*, r*, w*] up to length m
        agg_names = ["K", "C", "r", "w"][:m]
        # Evaluate current aggregate gradients
        aggs = ift_res.grad_aggregates

        # Compute baseline aggregates from solution
        alpha_val = float(p_dict.get("alpha", 0.36))
        delta_val = float(p_dict.get("delta", 1.0))
        z_raw = p_dict.get("z", p_dict.get("A", 1.0))
        z_val = float(1.0 if z_raw is None else z_raw)

        if hasattr(sol_eval, "distribution") and hasattr(sol_eval.distribution, "asset_grid"):
            k_val = float(getattr(sol_eval, "K", np.sum(sol_eval.distribution.marginal_assets() * sol_eval.distribution.asset_grid)))
            l_val = float(getattr(sol_eval, "L", 1.0))
            y_val = (k_val ** alpha_val) * (l_val ** (1.0 - alpha_val))
            c_val = y_val - delta_val * k_val
            r_val = float(getattr(sol_eval, "r", alpha_val * (k_val ** (alpha_val - 1.0)) * (l_val ** (1.0 - alpha_val)) - delta_val))
            w_val = float(getattr(sol_eval, "w", (1.0 - alpha_val) * (k_val ** alpha_val) * (l_val ** (-alpha_val))))
        elif hasattr(sol_eval, "policy"):
            dom: Any = getattr(prob_eval, "domain", ((0.05, 5.0),))
            dom0: Any = dom[0] if (isinstance(dom, (tuple, list)) and len(dom) > 0 and isinstance(dom[0], (tuple, list))) else dom
            k_min = float(dom0[0])
            k_max = float(dom0[1])
            try:
                f_low = sol_eval.policy(k_min + 1e-4) - (k_min + 1e-4)
                f_high = sol_eval.policy(k_max - 1e-4) - (k_max - 1e-4)
                if f_low * f_high <= 0:
                    k_val = float(brentq(lambda k: sol_eval.policy(k) - k, k_min + 1e-4, k_max - 1e-4))
                else:
                    k_val = float(0.5 * (k_min + k_max))
            except Exception:
                k_val = float(0.5 * (k_min + k_max))
            y_val = z_val * (k_val ** alpha_val)
            c_val = y_val - delta_val * k_val
            r_val = alpha_val * z_val * (k_val ** (alpha_val - 1.0)) - delta_val
            w_val = (1.0 - alpha_val) * z_val * (k_val ** alpha_val)
        else:
            k_val = 1.0
            y_val = z_val * (k_val ** alpha_val)
            c_val = y_val - delta_val * k_val
            r_val = alpha_val * z_val * (k_val ** (alpha_val - 1.0)) - delta_val
            w_val = (1.0 - alpha_val) * z_val * (k_val ** alpha_val)

        model_aggs = {"K": k_val, "C": c_val, "r": r_val, "w": w_val}
        m_eval = np.array([model_aggs[name] for name in agg_names], dtype=np.float64)

        G = np.zeros((m, p), dtype=np.float64)
        for row_i, name in enumerate(agg_names):
            if name in aggs and aggs[name] is not None:
                G[row_i, :] = aggs[name]

    diff = m_eval - m_target
    W = np.eye(m, dtype=np.float64) if weighting_matrix is None else np.asarray(weighting_matrix, dtype=np.float64)

    Q_val = float(diff.T @ W @ diff)
    grad_Q = 2.0 * (G.T @ (W @ diff))

    return Q_val, grad_Q
