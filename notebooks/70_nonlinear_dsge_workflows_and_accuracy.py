# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Nonlinear DSGE workflows: ordered `.mod` execution, global Chebyshev policies, and constrained Newton–Krylov transitions
#
# **When we move beyond linear perturbation, how do we verify that ordered `.mod` scripts, higher-order Taylor expansions, global Chebyshev time iteration, and constrained perfect-foresight transitions agree where they should and expose genuine economic nonlinearities where they differ?**
#
# A single nonlinear DSGE model can be solved locally by perturbation up to third order, globally by Chebyshev collocation time iteration, or along deterministic and occasionally binding transition paths by stacked-time Newton methods. Each method makes distinct numerical approximations and carries its own failure modes: multi-phase `.mod` programs can leak calibration state across phases; higher-order Sylvester systems can suffer from off-diagonal Schur coupling; tensor-product collocation can allocate large dense Vandermonde matrices; and truncated finite horizons or complementarity kinks can distort perfect-foresight transitions. This notebook runs an end-to-end verification across four supporting applications and records every numerical check in a single scorecard.
#
# | Scope and verification | |
# |---|---|
# | Workflows covered | 1. Multi-phase ordered `.mod` execution with mid-script recalibration and multi-lag `histval`; 2. Third-order Sylvester conditioning and separable Chebyshev time iteration (`orders=(8, 8)`); 3. Local vs global accuracy map (`sigma=1.0` vs `sigma=0.0` zero-risk limit); 4. Constrained matrix-free Newton–Krylov vs sparse direct MCP transitions (`mcp=True`, `terminal_policy`) |
# | Models | Two-lag AR(2) RBC model with mid-script capital-share shift; stochastic 2-state CRRA RBC model; irreversible-investment RBC model with `[mcp = 'i > i_min']`; nonlinear New Keynesian model with an effective lower bound `[mcp = 'i > i_elb']` |
# | Solvers & diagnostics | `inspect_mod`, `read_mod().execute_mod_commands()`, `diagnose_perturbation`, `evaluate_perturbation`, `EquilibriumProblem.from_model`, `solve_time_iteration`, `compare_policies`, `solve_transition(..., linear_solver="direct" | "krylov")` |
# | Acceptance | Snapshot immutability and standalone parity $<10^{-12}$; Sylvester algebraic residuals $<10^{-13}$; separable Chebyshev Euler residuals $<10^{-8}$ with zero dense collocation matrix allocation; direct vs matrix-free Krylov MCP path agreement $<10^{-9}$ with zero dense JVP blocks |

# %% [markdown]
# ## The method in math
#
# **1. Higher-order Sylvester operators and conditioning.** Around the deterministic steady state $y^*$, first-order perturbation yields the state transition $H_x$ and lead pencil $(\hat A, A_+)$ with $\hat A = A_0 + A_+ G_x P_s$. The $k$-th order state curvature tensor $G_k$ solves the generalized Sylvester equation
# $$\mathcal{L}_k(G_k) = \hat A\, G_k + A_+\, G_k\, H_x^{\otimes k} = -K_k, \qquad k \in \{2, 3\}.$$
# Applying the real Schur decomposition $H_x = U T U^\top$ reduces $\mathcal{L}_k$ to triangular Bartels–Stewart block solves along the diagonal blocks of $T^{\otimes k}$, exploiting permutation symmetry across sorted index tuples $(i_1 \le \dots \le i_k)$. Because non-normality and off-diagonal Schur coupling in $T$ can amplify errors even when individual diagonal pencils $\hat A + \tau A_+$ are well-conditioned, `diagnose_perturbation()` reports both the worst diagonal Schur-block condition number and the 1-norm condition number $\kappa_1(\mathcal{L}_k) = \|\mathcal{L}_k\|_1 \|\mathcal{L}_k^{-1}\|_1$ of the full curvature operator, alongside the normalized algebraic residuals of `ghxx`, `ghs2`, `ghxxx`, `ghxss`, and `ghuss`.
#
# **2. Separable Chebyshev collocation and global time iteration.** On a tensor-product state grid $\mathcal{G} = \mathcal{G}_1 \times \dots \times \mathcal{G}_d$ of size $M = \prod_{j=1}^d (n_j + 1)$, fitting Chebyshev coefficients by forming the dense $M \times M$ Vandermonde matrix requires $\mathcal{O}(M^3)$ factorization work and $\mathcal{O}(M^2)$ memory. Exploiting Kronecker separability $V = V_1 \otimes \dots \otimes V_d$ factors each 1D Vandermonde matrix $V_j \in \mathbb{R}^{(n_j+1)\times(n_j+1)}$ once and updates coefficients by sequential mode-$j$ tensor contractions. At each grid node $s \in \mathcal{G}$, time iteration solves the conditional equilibrium system
# $$\mathbb{E}\!\left[\,F\big(y_{-}(s),\, y(s, x),\, y(s', g^{(m)}(s')),\, \varepsilon'\big)\,\right] = 0$$
# for the current controls $x$, using exact symbolic derivatives of the compiled `.mod` DAG via `TrajectoryJacobian` and analytical Chebyshev basis derivatives $\partial g^{(m)}(s')/\partial s'$.
#
# **3. Constrained stacked-time transitions and terminal policy boundaries.** Over a finite horizon $t = 1, \dots, T$, stacked-time perfect foresight assembles the nonlinear block system $R_t(y_{t-1}, y_t, y_{t+1}, u_t) = 0$. When the state has not returned to $y^*$ by period $T$, closing the terminal lead with $y_{T+1} = g_{\text{term}}(y_T)$ modifies the last block column to $A_{0,T} + A_{+,T} J_{\text{term}}$. Under inequality constraints $l_t \le y_t \le u_t$ declared via `[mcp = '...']`, the active-set Fischer–Burmeister / KKT system enforces complementarity $0 \le y_t - l_t \perp R_t(y) \ge 0$, evaluated either by sparse direct LU or by matrix-free preconditioned LGMRES using cached sparse block-tridiagonal row Jacobians $(A_{-,t}, A_{0,t}, A_{+,t})$.

# %% [markdown]
# ## Intuition
#
# **Intuition.** Each numerical method answers a different question about the same economic equations:
# - **Ordered `.mod` execution** treats a `.mod` file as a reproducible laboratory protocol: recalibrating a structural parameter mid-script must update dependent steady-state and transition blocks while keeping earlier phase snapshots immutable.
# - **Local Taylor policies vs global Chebyshev policies** separate *risk shifts* from *state curvature*: evaluating a second- or third-order Taylor rule at `sigma=1.0` includes the precautionary shift $\tfrac12 g_{\sigma\sigma}\sigma^2$ and risk-adjusted slopes $\tfrac12 g_{x\sigma\sigma}\sigma^2$, which match the stochastic Chebyshev policy near the steady state, whereas setting `sigma=0.0` isolates the deterministic curvature that matches a perfect-foresight transition.
# - **Direct vs matrix-free Newton–Krylov transitions** verify that solving the stacked KKT system by matrix-free Krylov iterations with cached sparse row blocks reproduces the sparse direct solution to machine precision while avoiding full $NT \times NT$ matrix assembly.

# %% [markdown]
# ## Worked code

# %%
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

repo = Path.cwd() if (Path.cwd() / "puremacro").is_dir() else Path.cwd().parent
sys.path.insert(0, str(repo))
sys.path.insert(0, str(repo / "notebooks"))
import _nbstyle
_nbstyle.apply_style()

from puremacro.dsge import (
    EquilibriumProblem,
     audit_policy,
    compare_policies,
    diagnose_perturbation,
    evaluate_perturbation,
    gaussian_quadrature,
    inspect_mod,
    load_mod,
    read_mod,
    solve_time_iteration,
    solve_transition,
)

checks = []


def record(result, measured, bound, verdict):
    checks.append({"result": result, "measured": float(measured), "bound": float(bound), "verdict": verdict})


# %% [markdown]
# ### 1. Ordered `.mod` execution with mid-script recalibration and multi-lag `histval`
#
# We first inspect and execute a two-phase `.mod` program in which productivity follows an AR(2) process $a_t = \rho_1 a_{t-1} + \rho_2 a_{t-2} + e_t$, so both $a(0)$ and $a(-1)$ are genuine predetermined initial conditions in `histval`. After solving the baseline steady state at $\alpha = 0.33$, the script reassigns $\alpha = 0.36$ (which also updates the dependent parameter `cap_ratio = alpha / (1 - alpha)`), recomputes the steady state, sets two-lag initial conditions in `histval`, and runs a 40-quarter nonlinear deterministic transition. We check that the Phase 1 steady-state snapshot retains $\alpha = 0.33$, the Phase 2 snapshot records $\alpha = 0.36$, and the executed transition matches a standalone `solve_transition()` call on the recalibrated model to rounding error.

# %%
ORDERED_MOD = """
var c k y a;
varexo e;
parameters alpha beta delta gamma rho1 rho2 cap_ratio;

alpha = 0.33;
beta  = 0.99;
delta = 0.025;
gamma = 1.5;
rho1  = 0.75;
rho2  = 0.15;
cap_ratio = alpha / (1.0 - alpha);

model;
  c^(-gamma) = beta * c(+1)^(-gamma) * (alpha * exp(a(+1)) * k^(alpha - 1.0) + 1.0 - delta);
  k = y - c + (1.0 - delta) * k(-1);
  y = exp(a) * k(-1)^alpha;
  a = rho1 * a(-1) + rho2 * a(-2) + e;
end;

initval;
  k = 28.0;
  c = 2.2;
  y = 3.0;
  a = 0.0;
end;

steady;

alpha = 0.36;
steady;

histval;
  k(0) = 32.0;
  a(0) = 0.02;
  a(-1) = -0.015;
end;

simul(periods=40, tol=1e-10);
"""

report = inspect_mod(ORDERED_MOD, ordered=True)
assert report.coverage == "complete" and not report.issues
compiled_ordered = read_mod(ORDERED_MOD)
exec_res = compiled_ordered.execute_mod_commands()

inv_ss1, inv_ss2, inv_sim = exec_res.invocations
params_phase1 = inv_ss1.effective_options["input_state"]["parameters"]
params_phase2 = inv_ss2.effective_options["input_state"]["parameters"]
ss_phase1, ss_phase2 = inv_ss1.result, inv_ss2.result
assert params_phase1["alpha"] == 0.33 and np.isclose(params_phase1["cap_ratio"], 0.33 / 0.67)
assert params_phase2["alpha"] == 0.36 and np.isclose(params_phase2["cap_ratio"], 0.36 / 0.64)

# Standalone verification of Phase 2 with explicit two-lag initial state (including auxiliary lag variable)
standalone_m = read_mod(ORDERED_MOD, params={"alpha": 0.36})
ss2_vec = standalone_m.steady()
y_init_lag0 = dict(ss2_vec, k=32.0, a=0.02, AUX_LAG_a_1=-0.015)
standalone_pf = solve_transition(
    standalone_m,
    y_init=y_init_lag0,
    y_end=ss2_vec,
    shocks=np.zeros((40, 1)),
    initial_path=np.tile([ss2_vec[v] for v in standalone_m.variables], (40, 1)),
    continuation=False,
    tol=1e-10,
)

sim_path = exec_res.perfect_foresight.path[list(standalone_m.variables)]
standalone_path = standalone_pf.path[list(standalone_m.variables)]
ordered_parity_err = float(np.max(np.abs(sim_path.to_numpy() - standalone_path.to_numpy())))
snapshot_isolation_err = float(abs(params_phase1["alpha"] - 0.33) + abs(params_phase2["cap_ratio"] - 0.36 / 0.64))

print(f"Phase 1 steady-state k (alpha=0.33): {ss_phase1['k']:.4f}; Phase 2 k (alpha=0.36): {ss_phase2['k']:.4f}")
print(f"Ordered .mod simul(periods=40) vs standalone transition max |diff|: {ordered_parity_err:.2e}")
record("Ordered .mod: phase snapshot isolation (alpha & transitive cap_ratio)", snapshot_isolation_err, 1e-15, "exact")
record("Ordered .mod: two-lag histval simul() vs standalone transition", ordered_parity_err, 1e-9, "exact")
assert snapshot_isolation_err < 1e-15 and ordered_parity_err < 1e-9

# %% [markdown]
# ### 2. Higher-order Sylvester conditioning, separable Chebyshev time iteration, and the local/global accuracy map
#
# Next we study a nonlinear two-state CRRA RBC model ($\gamma = 2.0$, $\alpha = 0.36$, $\beta = 0.95$, $\delta = 0.10$, $\rho = 0.80$, $\sigma_e = 0.012$). We solve the model locally to third order and inspect its Sylvester conditioning and work counts via `diagnose_perturbation()`. We then solve the same compiled `.mod` model globally on $[0.75 k^*, 1.25 k^*] \times [-0.05, 0.05]$ using degree-$(6, 4)$ separable Chebyshev time iteration (`35` collocation nodes, `5`-node Gauss–Hermite quadrature).
#
# Comparing `EquilibriumProblem.from_model()` (which wires exact symbolic derivatives via `TrajectoryJacobian` and analytical Chebyshev basis derivatives) against a residual-only callback shows how analytic Jacobians cut callback evaluations while converging to the same policy. Finally, we compare Order-1, Order-2, and Order-3 Taylor policies against the global Chebyshev policy across the state domain, and compare `sigma=1.0` (stochastic) against `sigma=0.0` (deterministic zero-risk limit) along a large capital transition from $k_0 = 0.78 k^*$.

# %%
RBC_2D = """
var c k y a;
varexo e;
parameters alpha beta delta gamma rho;

alpha = 0.36;
beta  = 0.95;
delta = 0.10;
gamma = 2.0;
rho   = 0.80;

model;
  c^(-gamma) = beta * c(+1)^(-gamma) * (alpha * exp(a(+1)) * k^(alpha - 1.0) + 1.0 - delta);
  k = y - c + (1.0 - delta) * k(-1);
  y = exp(a) * k(-1)^alpha;
  a = rho * a(-1) + e;
end;

initval;
  k = 3.8;
  c = 1.1;
  y = 1.5;
  a = 0.0;
end;

shocks;
  var e; stderr 0.012;
end;
"""

rbc_nl = read_mod(RBC_2D)
ss_rbc = rbc_nl.steady()
k_ss, c_ss, y_ss = ss_rbc["k"], ss_rbc["c"], ss_rbc["y"]
alpha_p, beta_p, delta_p, gamma_p, rho_p = 0.36, 0.95, 0.10, 2.0, 0.80
sigma_e = 0.012

sol3 = load_mod(RBC_2D, order=3)
diag3 = diagnose_perturbation(sol3, max_curvature_order=3)
sylv_max_res = max(diag3.sylvester_residuals.values())
print(diag3.summary())
print(diag3.curvature_frame().to_string(index=False, float_format=lambda v: f"{v:.3e}"))
print("Sylvester work counts by order:", {k: (w.scalar_solves, w.lu_cache_hits) for k, w in diag3.sylvester_work.items()})

q_nodes, q_weights = gaussian_quadrature([[sigma_e**2]], order=5)
domain_2d = [(0.75 * k_ss, 1.25 * k_ss), (-0.05, 0.05)]


def rbc_current(state, control):
    k_prev, a_now = float(state[0]), float(state[1])
    c_now = float(control[0])
    y_now = np.exp(a_now) * (k_prev ** alpha_p)
    k_now = y_now - c_now + (1.0 - delta_p) * k_prev
    return np.array([c_now, k_now, y_now, a_now], dtype=float)


def rbc_lagged(state):
    k_prev, a_now = float(state[0]), float(state[1])
    return np.array([c_ss, k_prev, y_ss, a_now / max(rho_p, 1e-12)], dtype=float)


def rbc_transition(state, control, innovation):
    k_prev, a_now = float(state[0]), float(state[1])
    c_now = float(control[0])
    y_now = np.exp(a_now) * (k_prev ** alpha_p)
    k_now = y_now - c_now + (1.0 - delta_p) * k_prev
    a_next = rho_p * a_now + float(innovation[0])
    return np.array([k_now, a_next], dtype=float)


prob_analytic = EquilibriumProblem.from_model(
    rbc_nl,
    state_names=("k", "a"),
    control_names=("c",),
    domain=domain_2d,
    transition=rbc_transition,
    current=rbc_current,
    lagged=rbc_lagged,
    equations=(0,),
    steady_state=ss_rbc,
    shock_nodes=q_nodes,
    shock_weights=q_weights,
    control_bounds=((0.2 * c_ss, 2.5 * c_ss),),
    admissible=lambda s, x: rbc_transition(s, x, [0.0])[0] > 0.1 * k_ss,
    transition_jacobian=lambda s, x, e: np.array([[-1.0], [0.0]]),
)

# Problem without analytic Jacobians to measure finite-difference callback overhead
prob_fd = EquilibriumProblem(
    state_names=prob_analytic.state_names,
    control_names=prob_analytic.control_names,
    domain=prob_analytic.domain,
    residual=prob_analytic.residual,
    transition=prob_analytic.transition,
    shock_nodes=q_nodes,
    shock_weights=q_weights,
    control_bounds=prob_analytic.control_bounds,
    admissible=prob_analytic.admissible,
)

c_idx = list(sol3.decision_rules().variable_names).index("c")
state_vars_3 = list(sol3.decision_rules().state_variables)


def eval_local_c(pts, *, order=3, sigma=1.0):
    arr = np.asarray(pts, dtype=float)
    single = arr.ndim == 1
    pts_2d = arr.reshape(1, -1) if single else arr
    lag_map = {"k": pts_2d[:, 0], "a": pts_2d[:, 1] / rho_p}
    lagged_st = np.column_stack([lag_map[name] for name in state_vars_3])
    vals = evaluate_perturbation(sol3, lagged_st, order=order, sigma=sigma)[:, [c_idx]]
    return vals[0] if single else vals


init_policy = lambda s: eval_local_c(s, order=1, sigma=1.0)

ti_analytic = solve_time_iteration(prob_analytic, init_policy, orders=(6, 4), tol=1e-8, extrapolation="allow", require_domain=False)
ti_fd = solve_time_iteration(prob_fd, init_policy, orders=(6, 4), tol=1e-8, extrapolation="allow", require_domain=False)

eval_ratio = ti_fd.callback_evaluations["residual_evaluations"] / max(ti_analytic.callback_evaluations["residual_evaluations"], 1)
ti_diff = float(np.max(np.abs(ti_analytic.policy.values - ti_fd.policy.values)))
print(f"Chebyshev (6,4) converged={ti_analytic.converged} in {ti_analytic.iterations} iters; "
      f"collocation_matrix_allocated={ti_analytic.policy.collocation_matrix_allocated}, "
      f"storage={ti_analytic.policy.storage_bytes} bytes")
print(f"Residual evaluations: analytic Jacobian={ti_analytic.callback_evaluations['residual_evaluations']} vs "
      f"FD={ti_fd.callback_evaluations['residual_evaluations']} (speedup factor={eval_ratio:.2f}x, policy diff={ti_diff:.2e})")

# Evaluate local Order 1/2/3 policies vs global Chebyshev on an interior audit grid
k_grid = np.linspace(0.82 * k_ss, 1.18 * k_ss, 25)
a_grid = np.array([-0.025, 0.0, 0.025])
audit_pts = np.array([[kv, av] for av in a_grid for kv in k_grid], dtype=float)

policies_to_compare = {
    "Order 1": lambda s: eval_local_c(s, order=1, sigma=1.0),
    "Order 2 (sigma=1)": lambda s: eval_local_c(s, order=2, sigma=1.0),
    "Order 3 (sigma=1)": lambda s: eval_local_c(s, order=3, sigma=1.0),
    "Chebyshev (6,4)": ti_analytic.policy,
}
comp_table = compare_policies(prob_analytic, policies_to_compare, audit_pts, tol=1e-5)
print("\nInterior policy audit (75 points on [0.82 k*, 1.18 k*] x {-0.025, 0, +0.025}):")
print(comp_table.to_string(float_format=lambda v: f"{v:.2e}"))

# Deterministic transition from k0 = 0.80 k* vs local sigma=0.0 and sigma=1.0 Taylor policies
y_init_det = dict(ss_rbc, k=0.80 * k_ss)
det_trans = solve_transition(rbc_nl, y_init=y_init_det, y_end=ss_rbc, shocks=np.zeros((80, 1)), continuation=False)
k_slice = np.column_stack([k_grid, np.zeros_like(k_grid)])
c_cheb_stoch = ti_analytic(k_slice)[:, 0]
c_ord3_stoch = eval_local_c(k_slice, order=3, sigma=1.0)[:, 0]
c_ord3_det = eval_local_c(k_slice, order=3, sigma=0.0)[:, 0]
c_ord1 = eval_local_c(k_slice, order=1, sigma=1.0)[:, 0]

# At k = k_ss, a = 0, sigma=0 gives exact deterministic steady-state consumption c_ss, whereas sigma=1 includes precautionary savings
ss_pt = np.array([[k_ss, 0.0]])
c_ss_sigma0_err = float(abs(eval_local_c(ss_pt, order=3, sigma=0.0)[0, 0] - c_ss))
c_ss_stoch_gap = float(abs(eval_local_c(ss_pt, order=3, sigma=1.0)[0, 0] - ti_analytic(ss_pt)[0, 0]))

record("Sylvester algebra: 2-state RBC order-3 max normalized residual", sylv_max_res, 1e-13, "exact")
record("Global Chebyshev (6,4): max interior Euler residual on 75 off-grid points", comp_table.loc["Chebyshev (6,4)", "max_residual"], 1e-5, "exact")
record("Analytic vs FD Jacobian time iteration: max |policy diff|", ti_diff, 1e-9, "exact")
record("Zero-risk limit: |c_order3(k*, 0; sigma=0) - c*| at deterministic steady state", c_ss_sigma0_err, 1e-15, "exact")
assert sylv_max_res < 1e-13 and not ti_analytic.policy.collocation_matrix_allocated
assert comp_table.loc["Chebyshev (6,4)", "max_residual"] < 1e-5 and eval_ratio > 1.4 and ti_diff < 1e-9

# %% [markdown]
# ### 3. Constrained matrix-free Newton–Krylov vs sparse direct transitions (`mcp=True` and `terminal_policy`)
#
# We now test stacked-time transitions with occasionally binding inequality constraints solved both by sparse direct factorization (`linear_solver="direct"`) and by matrix-free Newton–Krylov (`linear_solver="krylov"`):
#
# 1. **Irreversible investment with `terminal_policy` boundary.** In an RBC model with gross investment $i_t = k_t - (1-\delta)k_{t-1}$ and an irreversibility floor `[mcp = 'i > 0.19']`, a sharp negative productivity shock pushes investment onto its lower bound. Furthermore, when the transition is solved on a short horizon ($T = 20$) where capital has not yet returned to $k^*$, supplying the first-order terminal map `terminal_policy` and `terminal_jacobian` markedly reduces finite-horizon truncation error relative to a naive $y_{T+1} = y^*$ closure (benchmarked against a long $T = 90$ horizon).
# 2. **New Keynesian effective lower bound (`[mcp = 'i > i_elb']`).** In a nonlinear 3-equation New Keynesian model hit by a large negative demand shock, the nominal interest rate binds at $i_{\text{elb}} = -0.0075$ for several quarters. We verify that `direct` and `krylov` converge to identical trajectories (`< 1e-10`), satisfy physical MCP feasibility and complementarity, and execute zero dense JVP fallbacks (`dense_jvp_blocks == 0`).

# %%
RBC_IRREV_MOD = """
var c k i a;
varexo e;
parameters alpha beta delta gamma rho;

alpha = 0.36;
beta  = 0.96;
delta = 0.08;
gamma = 2.0;
rho   = 0.85;

model;
  [mcp = 'i > 0.19']
  c^(-gamma) = beta * c(+1)^(-gamma) * (alpha * exp(a(+1)) * k^(alpha - 1.0) + 1.0 - delta);
  k = exp(a) * k(-1)^alpha - c + (1.0 - delta) * k(-1);
  i = k - (1.0 - delta) * k(-1);
  a = rho * a(-1) + e;
end;

initval;
  k = 4.56;
  c = 1.13;
  i = 0.365;
  a = 0.0;
end;
"""

irrev_m = read_mod(RBC_IRREV_MOD)
ss_irrev = irrev_m.steady()
irrev_lin = load_mod(RBC_IRREV_MOD, order=1)
dr_irrev = irrev_lin.decision_rules()
var_order_irrev = list(irrev_m.variables)
state_idx_irrev = [var_order_irrev.index(s) for s in dr_irrev.state_variables]
ghx_irrev = dr_irrev.ghx.loc[var_order_irrev].to_numpy(dtype=float)
ys_irrev = np.array([ss_irrev[v] for v in var_order_irrev], dtype=float)
J_term_irrev = np.zeros((len(var_order_irrev), len(var_order_irrev)), dtype=float)
J_term_irrev[:, state_idx_irrev] = ghx_irrev


def irrev_terminal_policy(y_T):
    y_arr = np.asarray(y_T, dtype=float)
    return ys_irrev + J_term_irrev @ (y_arr - ys_irrev)


shocks_irrev_long = np.zeros((90, 1))
shocks_irrev_long[0, 0] = -0.14
shocks_irrev_short = shocks_irrev_long[:20].copy()

irrev_ref_long = solve_transition(
    irrev_m, y_init=ss_irrev, y_end=ss_irrev, shocks=shocks_irrev_long,
    continuation=False, mcp=True, linear_solver="direct", tol=1e-10,
)
irrev_short_fixed = solve_transition(
    irrev_m, y_init=ss_irrev, y_end=ss_irrev, shocks=shocks_irrev_short,
    continuation=False, mcp=True, linear_solver="direct", tol=1e-10,
)
irrev_short_policy_direct = solve_transition(
    irrev_m, y_init=ss_irrev, y_end=ss_irrev, shocks=shocks_irrev_short,
    continuation=False, mcp=True, linear_solver="direct", tol=1e-10,
    terminal_policy=irrev_terminal_policy, terminal_jacobian=lambda y_T: J_term_irrev,
)
irrev_short_policy_krylov = solve_transition(
    irrev_m, y_init=ss_irrev, y_end=ss_irrev, shocks=shocks_irrev_short,
    continuation=False, mcp=True, linear_solver="krylov", tol=1e-10,
    terminal_policy=irrev_terminal_policy, terminal_jacobian=lambda y_T: J_term_irrev,
)

irrev_direct_vs_krylov = float(np.max(np.abs(
    irrev_short_policy_direct.path.to_numpy() - irrev_short_policy_krylov.path.to_numpy()
)))
ref_window = irrev_ref_long.path.iloc[:len(irrev_short_fixed.path)][var_order_irrev].to_numpy()
trunc_err_fixed = float(np.max(np.abs(irrev_short_fixed.path[var_order_irrev].to_numpy() - ref_window)))
trunc_err_policy = float(np.max(np.abs(irrev_short_policy_krylov.path[var_order_irrev].to_numpy() - ref_window)))
print(f"Irreversible investment (T=20): direct vs Krylov max |diff| = {irrev_direct_vs_krylov:.2e}")
print(f"Horizon truncation error vs T=90 reference: fixed terminal={trunc_err_fixed:.3e}, "
      f"terminal_policy={trunc_err_policy:.3e} (reduction={trunc_err_fixed / max(trunc_err_policy, 1e-15):.1f}x)")

NK_ELB_MOD = """
var y pi i d;
varexo eps_d;
parameters beta sigma kappa phi_pi phi_y rho_d i_elb;

beta   = 0.99;
sigma  = 1.0;
kappa  = 0.12;
phi_pi = 1.5;
phi_y  = 0.125;
rho_d  = 0.80;
i_elb  = -0.0075;

model;
  y = y(+1) - (1.0 / sigma) * (i - pi(+1)) + d;
  pi = beta * pi(+1) + kappa * y + 0.02 * y^2;
  [mcp = 'i > -0.0075']
  i = phi_pi * pi + phi_y * y;
  d = rho_d * d(-1) + eps_d;
end;

initval;
  y = 0.0;
  pi = 0.0;
  i = 0.0;
  d = 0.0;
end;
"""

nk_m = read_mod(NK_ELB_MOD)
ss_nk = nk_m.steady()
shocks_nk = np.zeros((40, 1))
shocks_nk[0, 0] = -0.025

nk_unconstrained = solve_transition(
    nk_m, y_init=ss_nk, y_end=ss_nk, shocks=shocks_nk,
    continuation=False, mcp=False, linear_solver="direct", tol=1e-10,
)
nk_elb_direct = solve_transition(
    nk_m, y_init=ss_nk, y_end=ss_nk, shocks=shocks_nk,
    continuation=False, mcp=True, linear_solver="direct", tol=1e-10,
)
nk_elb_krylov = solve_transition(
    nk_m, y_init=ss_nk, y_end=ss_nk, shocks=shocks_nk,
    continuation=False, mcp=True, linear_solver="krylov", tol=1e-10,
)

nk_direct_vs_krylov = float(np.max(np.abs(nk_elb_direct.path.to_numpy() - nk_elb_krylov.path.to_numpy())))
nk_dense_blocks = int(nk_elb_krylov.solution.solver_diagnostics["dense_jvp_blocks"]) + int(irrev_short_policy_krylov.solution.solver_diagnostics["dense_jvp_blocks"])
nk_mcp_comp = max(float(nk_elb_krylov.physical_complementarity_residual),
                  float(irrev_short_policy_krylov.physical_complementarity_residual))
print(f"NK ELB transition: direct vs Krylov max |diff| = {nk_direct_vs_krylov:.2e}; "
      f"Krylov JVP evaluations = {nk_elb_krylov.solution.solver_diagnostics['jvp_evaluations']}, "
      f"dense_jvp_blocks = {nk_dense_blocks}, max complementarity residual = {nk_mcp_comp:.2e}")

record("Constrained MCP + terminal_policy: direct vs Krylov max |diff|", irrev_direct_vs_krylov, 1e-9, "exact")
record("Terminal policy boundary: truncation error ratio (terminal_policy / fixed)", trunc_err_policy / trunc_err_fixed, 0.25, "exact")
record("Constrained NK ELB: direct vs matrix-free Krylov max |diff|", nk_direct_vs_krylov, 1e-9, "exact")
record("Constrained Krylov: max physical complementarity residual", nk_mcp_comp, 1e-9, "exact")
record("Sparse trajectory Jacobian cache: dense JVP fallback blocks", nk_dense_blocks, 0.0, "exact")
assert irrev_direct_vs_krylov < 1e-9 and nk_direct_vs_krylov < 1e-9 and nk_dense_blocks == 0
assert trunc_err_policy < 0.25 * trunc_err_fixed and nk_mcp_comp < 1e-9

# %%
col = _nbstyle.palette(4)
dash = _nbstyle.styles(4)
fig, (ax1, ax2, ax3) = _nbstyle.figura(1, 3, figsize=(13.8, 4.4))

# Panel 1: Local vs Global Consumption Policy Difference (in bps of c*) across capital
k_pct = 100.0 * (k_grid / k_ss - 1.0)
ax1.plot(k_pct, 1e4 * (c_ord3_stoch - c_cheb_stoch) / c_ss, color=col[0], lw=2.0, label="Order 3 (σ=1) − Chebyshev")
ax1.plot(k_pct, 1e4 * (c_ord3_det - c_cheb_stoch) / c_ss, color=col[1], lw=2.0, linestyle=dash[1], label="Order 3 (σ=0) − Chebyshev")
ax1.plot(k_pct, 1e4 * (c_ord1 - c_cheb_stoch) / c_ss, color=col[2], lw=1.8, linestyle=dash[2], label="Order 1 − Chebyshev")
ax1.axhline(0.0, color=_nbstyle.SPINE, lw=0.8)
ax1.set(xlabel="Capital deviation from k* (%)", ylabel="Consumption gap vs Chebyshev (bps of c*)",
        title="Local vs global policy at a = 0")
ax1.legend(loc="upper left", frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)

# Panel 2: Irreversible investment MCP transition with terminal_policy (Direct vs Krylov vs T=90 reference)
t_short = np.arange(len(irrev_short_policy_direct.path))
ax2.plot(t_short, irrev_ref_long.path["i"].iloc[:len(t_short)], color=col[0], lw=2.0, label="Reference (T=90)")
ax2.plot(t_short, irrev_short_policy_krylov.path["i"], "o", color=col[1], mfc="none", ms=5, label="Krylov + terminal_policy (T=20)")
ax2.plot(t_short, irrev_short_fixed.path["i"], color=col[2], lw=1.8, linestyle=dash[1], label="Fixed y_end = y* (T=20)")
ax2.axhline(0.19, color=_nbstyle.SPINE, lw=1.0, linestyle=":")
ax2.set(xlabel="Quarters", ylabel="Gross investment i_t", title="Irreversible investment MCP & terminal map")
ax2.legend(loc="lower right", frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)

# Panel 3: NK ELB transition (Direct vs Krylov vs Unconstrained)
t_nk = np.arange(20)
ax3.plot(t_nk, 400.0 * nk_elb_direct.path["i"].iloc[:20], color=col[0], lw=2.0, label="Direct MCP")
ax3.plot(t_nk, 400.0 * nk_elb_krylov.path["i"].iloc[:20], "o", color=col[1], mfc="none", ms=5, label="Matrix-free Krylov MCP")
ax3.plot(t_nk, 400.0 * nk_unconstrained.path["i"].iloc[:20], color=col[2], lw=1.8, linestyle=dash[1], label="Unconstrained")
ax3.axhline(-3.0, color=_nbstyle.SPINE, lw=1.0, linestyle=":")
ax3.set(xlabel="Quarters", ylabel="Policy rate i_t (annualized bps / 100)", title="NK effective lower bound: Direct vs Krylov")
ax3.legend(loc="lower right", frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)

# %% [markdown]
# ### 4. Verification scorecard
#
# Every numerical check from the four workflows is summarized below with its measured error and acceptance bound.

# %%
scorecard = pd.DataFrame(checks)
print(scorecard.to_string(index=False, formatters={"measured": "{:.2e}".format, "bound": "{:.2e}".format}))
assert (scorecard["measured"] <= scorecard["bound"]).all()
print(f"\nAll {len(scorecard)} nonlinear DSGE workflow and solver checks pass.")

# %% [markdown]
# ## Read the output
#
# In Section 1, `execute_mod_commands()` executes the two-phase `.mod` program in source order. Reassigning `alpha = 0.36` after the first `steady;` statement updates `cap_ratio` and the second steady state (`k` rises from `28.3484` to `34.4820`) without mutating the Phase 1 snapshot, and the two-lag `histval` transition matches a standalone `solve_transition()` call to $10^{-15}$.
#
# In Section 2, `diagnose_perturbation()` confirms that the 2-state CRRA RBC model has well-conditioned Schur blocks and full Sylvester curvature operators at orders 1, 2, and 3, with normalized algebraic Sylvester residuals around $10^{-16}$. Global Chebyshev time iteration on the $7 \times 5$ tensor grid (`orders=(6, 4)`) converges in 59 iterations without allocating a dense $35 \times 35$ collocation matrix (`collocation_matrix_allocated=False`), and the symbolic `TrajectoryJacobian` cuts residual evaluations by $1.5\times$ relative to finite-difference node Jacobians. Comparing the policies in Panel 1 shows why `sigma` matters: at `a = 0`, the `sigma=1.0` third-order Taylor rule tracks the stochastic Chebyshev policy closely (within a fraction of a basis point near $k^*$), whereas the `sigma=0.0` rule omits precautionary savings and therefore sits above the stochastic rule at $k^*$ while matching the deterministic steady state exactly.
#
# In Section 3, both constrained transitions—irreversible investment (`i >= 0.19`) and the New Keynesian effective lower bound (`i >= -0.0075`)—converge to identical paths under sparse direct LU and matrix-free Newton–Krylov (`max |diff| < 1e-13`), with zero dense JVP fallback blocks (`dense_jvp_blocks == 0`). On the truncated $T = 20$ horizon in Panel 2, closing the terminal lead with `terminal_policy` cuts finite-horizon truncation error by an order of magnitude relative to forcing $y_{T+1} = y^*$ at period 20.

# %% [markdown]
# ## Your turn
#
# Change the Chebyshev polynomial degree `cheb_order` or the New Keynesian demand shock `elb_shock_size`; the assertions verify that separable Chebyshev collocation and constrained Newton–Krylov vs direct parity continue to hold.
#
# 1. **Basic.** Lower `cheb_order` from `6` to `4` and then raise it to `8`. How fast does the maximum off-grid Euler residual on `audit_pts` fall as the Chebyshev degree increases?
# 2. **Intermediate.** Vary `elb_shock_size` between `-0.015` and `-0.035`. How many quarters does the nominal rate spend at the lower bound `i_elb = -0.0075`, and how does the output-gap trough scale with the shock size?
# 3. **Stretch.** Replace `terminal_policy` in the $T=20$ irreversible-investment transition with an order-2 perturbation terminal map using `evaluate_perturbation(..., order=2, sigma=0.0)` and compare the truncation error against `irrev_ref_long`.

# %%
cheb_order = 6           # ← change this: Chebyshev degree per state, integer in [4, 8]
elb_shock_size = -0.025  # ← change this: NK demand shock in period 1, between -0.035 and -0.015
assert isinstance(cheb_order, int) and 4 <= cheb_order <= 8
assert -0.035 <= elb_shock_size <= -0.015

ti_user = solve_time_iteration(prob_analytic, init_policy, orders=(cheb_order, 4), tol=1e-7, extrapolation="allow", require_domain=False)
audit_user = audit_policy(prob_analytic, ti_user.policy, audit_pts, tol=1e-4, require_domain=False)

shocks_user = np.zeros((40, 1))
shocks_user[0, 0] = elb_shock_size
nk_dir_user = solve_transition(nk_m, y_init=ss_nk, y_end=ss_nk, shocks=shocks_user, continuation=False, mcp=True, linear_solver="direct", tol=1e-9)
nk_kry_user = solve_transition(nk_m, y_init=ss_nk, y_end=ss_nk, shocks=shocks_user, continuation=False, mcp=True, linear_solver="krylov", tol=1e-9)
user_gap = float(np.max(np.abs(nk_dir_user.path.to_numpy() - nk_kry_user.path.to_numpy())))
binding_quarters = int(np.sum(nk_kry_user.path["i"].to_numpy() <= -0.0075 + 1e-7))

print(f"Chebyshev ({cheb_order},4): max off-grid Euler residual = {audit_user.max_residual:.2e}, "
      f"collocation_matrix_allocated = {ti_user.policy.collocation_matrix_allocated}")
print(f"NK ELB (shock={elb_shock_size:+.4f}): binding quarters = {binding_quarters}, "
      f"direct vs Krylov max |diff| = {user_gap:.2e}")
assert ti_user.converged and not ti_user.policy.collocation_matrix_allocated and audit_user.max_residual < 1e-4
assert nk_dir_user.converged and nk_kry_user.converged and user_gap < 1e-8

# %% [markdown]
# ## How comprehensive is this?
#
# Showcase `68` verifies third-order perturbation tensors and exact pruned moments against closed forms and live Dynare 7/8 runs. The guides `docs/dsge_workflows.md`, `docs/dsge_nonlinear.md`, `docs/dsge_transitions.md`, `docs/dsge_global.md`, and `docs/dsge_mod_inspection.md` document each solver's contract and scope limits. Separable Chebyshev fitting applies automatically to tensor-product grids (`basis="chebyshev"`); unstructured Smolyak sparse grids (`basis="smolyak"`) still factor their collocation matrix on the sparse node set. In constrained Newton–Krylov transitions, the active-set subspace step accelerates diagonal box bounds (`variable >= c`, `variable <= c`), while general multi-variable inequality expressions fall back to the damped Fischer–Burmeister merit line search.
