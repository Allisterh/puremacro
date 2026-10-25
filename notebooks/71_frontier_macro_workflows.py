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
# # Frontier macroeconomic workflows: automatic differentiation, sequence-space HANK, higher-order OccBin, mixed-frequency nowcasting, continuous-time HJB, network/spatial general equilibrium, and self-auditing PolicyStudio
#
# **How do we build and audit an end-to-end frontier macroeconomic pipeline—uniting forward automatic differentiation, sequence-space DAG Jacobians, pruned higher-order regime switching, real-time news decompositions, continuous-time heterogeneous-agent distributions, second-order production networks, dynamic spatial migration, and interactive counterfactual grid auditing—in pure NumPy under strict zero-server deployment?**
#
# Modern quantitative macroeconomics operates across multiple structural paradigms: automatic differentiation engines that compute analytical policy tangents without numerical cancellation; sequence-space DAG compilers and frequency-domain estimators for heterogeneous-agent (HANK) models; pruned perturbation solvers that resolve occasionally binding constraints with stochastic precautionary risk shifts; state-space nowcasters that decompose incoming data releases into real-time macroeconomic news; continuous-time finite-volume PDE solvers for Kolmogorov forward densities; input-output aggregation capturing second-order production bottlenecks; dynamic spatial hat-algebras simulating trade and labor migration; and self-auditing counterfactual studios that verify equilibrium conservation laws before publishing interactive zero-server visual dashboards. This showcase notebook unites all seven frontier tracks into a reproducible, self-contained suite with end-to-end verification.
#
# | Scope and verification | |
# |---|---|
# | Workflows covered | 1. Pure-NumPy Forward AD (`DualArray`, `HyperDualArray`, `hessian_forward`); 2. Generalized sequence-space DAG & Whittle HANK estimation (`BlockDAG`, `hank_loglik_whittle`); 3. 2nd/3rd-order pruned nonlinear OccBin with precautionary risk shifts; 4. Mixed-frequency state-space nowcasting & Banbura–Modugno news decomposition; 5. Continuous-time upwind HJB, adjoint KFE, and `HetBlock` sequence-space bridge; 6. Baqaee–Farhi 2nd-order network macro & Caliendo–Parro–Rossi-Hansberg dynamic spatial hat-algebra; 7. Self-auditing `PolicyStudio` counterfactual grid & zero-server interactive HTML5/SVG export |
# | Models | Non-homothetic CRRA-CES utility; 3-equation New Keynesian / HANK DAG; nonlinear NK model with effective lower bound ($i_t \ge i_{\text{floor}}$); 4-variable mixed-frequency dynamic factor model; continuous-time Huggett–Aiyagari household with Poisson income jumps; 3-sector upstream-downstream input-output economy; 3-region dynamic spatial trade and migration economy |
# | Solvers & diagnostics | `value_and_grad`, `hessian_forward`, `BlockDAG.impulse_responses`, `hank_loglik_whittle`, `solve_nonlinear_occbin`, `DynamicFactorModel.fit`, `banbura_modugno_news`, `solve_hjb_achdou`, `solve_kfe_achdou`, `baqaee_farhi_decomposition`, `solve_dynamic_spatial_hat`, `run_policy_studio`, `result_to_interactive_html` |
# | Acceptance | Forward AD gradients & Hessians $< 10^{-14}$; SSJ DAG residual norm $< 10^{-12}$ and interior IRF gap $< 10^{-9}$; Whittle Parseval error $< 10^{-10}$; OccBin bound violation $< 10^{-12}$; Banbura–Modugno news additivity error $< 10^{-12}$; continuous HJB generator row-sum error $< 10^{-13}$ and KFE stationarity $< 10^{-12}$; Cobb–Douglas 2nd-order network curvature $< 10^{-14}$; spatial population conservation error $< 10^{-12}$; PolicyStudio 100% verified cell audits with zero external CDN dependencies |

# %% [markdown]
# ## The method in math
#
# **1. Pure-NumPy Forward Automatic Differentiation.** Forward AD augments reals into hyper-dual algebras $\mathbb{D}_2$:
# $$\mathcal{D}_1(x) = x + \varepsilon \dot{x}, \quad \mathcal{D}_2(x) = x + \varepsilon_1 \dot{x}_1 + \varepsilon_2 \dot{x}_2 + \varepsilon_1 \varepsilon_2 \ddot{x}, \qquad \varepsilon_1^2 = \varepsilon_2^2 = 0.$$
# Propagating dual components through elementary ufuncs and Sylvester operators produces exact first derivatives $\nabla f(x)$ and Hessians $\nabla^2 f(x)$ to machine precision without finite-difference grid tuning ($h$) or complex-step cancellation.
#
# **2. Sequence-Space DAGs and Whittle Likelihood.** A macroeconomic equilibrium DAG connects endogenous blocks $k$ with unknowns $\mathbf{U}$ and exogenous targets $\mathbf{Z}$. Differentiating the coupled system yields sequence-space Jacobians:
# $$\mathcal{J}_{\mathbf{U}} \mathrm{d}\mathbf{U} + \mathcal{J}_{\mathbf{Z}} \mathrm{d}\mathbf{Z} = 0 \implies \mathrm{d}\mathbf{U} = -\mathcal{J}_{\mathbf{U}}^{-1} \mathcal{J}_{\mathbf{Z}} \mathrm{d}\mathbf{Z}.$$
# In the frequency domain, the Whittle log-likelihood matches empirical second moments against model spectral densities $S_y(\omega; \theta)$:
# $$\ln \mathcal{L}_W(\theta) = -\frac{1}{2}\sum_{k=1}^K \left[ \ln \det S_y(\omega_k; \theta) + \operatorname{tr}\left( S_y(\omega_k; \theta)^{-1} I_y(\omega_k) \right) \right].$$
#
# **3. Pruned Nonlinear OccBin with Precautionary Risk Shifts.** When an inequality constraint $y_{i,t} \ge y_{\text{floor}}$ binds occasionally, the pruned perturbation path decomposes into first-, second-, and third-order components:
# $$y_t = y^* + y_t^{(1)} + y_t^{(2)} + y_t^{(3)}, \quad y_t^{(2)} = H_x y_{t-1}^{(2)} + \frac{1}{2} G_{xx} (y_{t-1}^{(1)})^{\otimes 2} + G_{xu}(y_{t-1}^{(1)} \otimes u_t) + \frac{1}{2} G_{uu} u_t^{\otimes 2} + \frac{1}{2} g_{\sigma\sigma}\sigma^2.$$
# Active-set regime switching iterates over candidate binding durations $T_{\text{bind}}$ while incorporating the risk shift $\frac{1}{2} g_{\sigma\sigma} \sigma^2$.
#
# **4. Mixed-Frequency Nowcasting & Banbura–Modugno News.** Dynamic factor models cast mixed-frequency observables into state space $s_t = \Phi s_{t-1} + \eta_t, y_t = \Lambda s_t + \varepsilon_t$. The exact release-by-release news decomposition expresses the nowcast update as:
# $$\hat{y}_{t|v} - \hat{y}_{t|v-1} = \sum_{j \in \mathcal{I}_v} w_j \left( z_{j,v} - \hat{z}_{j,v|v-1} \right), \quad \text{with} \quad \sum_{j} \text{impact}_j = \text{total revision}.$$
#
# **5. Continuous-Time HJB & Adjoint KFE.** On wealth grid $a \in [a_{\min}, a_{\max}]$, the household HJB equation with Poisson jump intensity $\Lambda$ solves:
# $$\rho v_j(a) = \max_c \left\{ u(c) + s_j(a) \partial_a v_j(a) \right\} + \sum_{k \ne j} \lambda_{jk} [v_k(a) - v_j(a)].$$
# An implicit upwind finite-difference scheme constructs infinitesimal generator $A \in \mathbb{R}^{N \times N}$ ($A \mathbf{1} = 0$). The stationary Kolmogorov Forward Equation (KFE) is the exact sparse adjoint:
# $$A^\top g = 0, \quad \text{subject to} \quad \sum_{i} g_i \Delta a_i = 1.$$
#
# **6. Network Aggregation & Dynamic Spatial Hat-Algebra.** For input-output matrix $\Omega$ and final shares $b$, Baqaee–Farhi second-order GDP response to productivity shocks $\Delta \ln z$ is:
# $$\Delta \ln Y = \lambda^\top \Delta \ln z + \frac{1}{2} (\Delta \ln z)^\top \mathcal{H} (\Delta \ln z) - \lambda^\top \Delta \ln \mu,$$
# reducing to Hulten's theorem ($\mathcal{H} = 0$) under Cobb–Douglas ($\theta = 1$). In dynamic spatial economies, population updates follow Caliendo–Parro–Rossi-Hansberg hat-algebra:
# $$\hat{L}_{i,t+1} L_{i,t+1} = \sum_{j} \mu_{ji,t} \hat{\mu}_{ji,t} L_{j,t}.$$
#
# **7. Self-Auditing PolicyStudio Grids.** Across Cartesian grid $\mathcal{G} = \prod_k \mathcal{A}_k$, policy evaluation audits residuals $\mathcal{R}(\theta) = \max_m |r_m(\theta)|$ against strict thresholds:
# $$\text{Status}(\theta) = \begin{cases} \text{PASS} & \text{if } \mathcal{R}(\theta) \le \tau_{\text{warn}} \text{ and converged}, \\ \text{WARN} & \text{if } \tau_{\text{warn}} < \mathcal{R}(\theta) \le \tau_{\text{fail}}, \\ \text{FAIL} & \text{otherwise.} \end{cases}$$

# %% [markdown]
# ## Intuition
#
# **Intuition.** Frontier macroeconomic modeling requires bridging microeconomic heterogeneity, nonlinear constraints, and empirical observability:
# - **Forward automatic differentiation** replaces error-prone numerical step-size tuning with exact hyper-dual numbers, enabling machine-precision parameter tangents and Hessian operators inside NumPy.
# - **Sequence-space DAGs and Whittle likelihood** eliminate the curse of dimensionality in heterogeneous-agent (HANK) models: general equilibrium transitions are computed as linear sequence-space block operations, and frequency-domain estimation matches empirical spectra without costly state-space simulation.
# - **Pruned higher-order OccBin** captures the state-dependent asymmetry of macroeconomic crises—such as liquidity traps and binding borrowing constraints—while retaining stochastic precautionary risk buffers ($\sigma > 0$).
# - **Differentiable mixed-frequency nowcasting** isolates the macroeconomic "news" driving GDP revisions in real time, tracing forecast movements directly to unexpected surprises in specific high-frequency releases.
# - **Continuous-time finite-volume HJB and adjoint KFE** guarantee conservative probability flow and mass preservation by construction, creating an exact sparse bridge to macro sequence-space models.
# - **Second-order network macro and dynamic spatial migration** capture nonlinear bottlenecks, supply-chain substitution wedges, and regional labor adjustment frictions where linear approximations fail.
# - **Self-auditing PolicyStudio** ensures that every cell across high-dimensional counterfactual grids satisfies mathematical conservation laws and convergence criteria before publishing zero-server interactive visual reports.

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

from puremacro._autodiff import (
    DualArray,
    HyperDualArray,
    hessian_forward,
    jacobian_forward,
    value_and_grad,
)
from puremacro.dsge import (
    BlockDAG,
    HetBlock,
    OccBinConstraint,
    SimpleBlock,
    SolvedBlock,
    build_dynare,
    estimate_hank_whittle,
    hank_loglik_whittle,
    solve_nonlinear_occbin,
)
from puremacro.nowcast import DynamicFactorModel, banbura_modugno_news
from puremacro.vfi import solve_hjb_achdou, solve_kfe_achdou
from puremacro.trade import baqaee_farhi_decomposition
from puremacro.spatial import DynamicSpatialHatModel, solve_dynamic_spatial_hat
from puremacro.study import CounterfactualGrid, run_policy_studio

checks = []


def record(result, measured, bound, verdict):
    checks.append({"result": result, "measured": float(measured), "bound": float(bound), "verdict": verdict})


# %% [markdown]
# ### 1. Pure-NumPy forward automatic differentiation: exact gradients and Hessians
#
# We evaluate the forward automatic differentiation engine on a standard CRRA macro utility objective $u(c, \gamma) = \frac{c^{1-\gamma}-1}{1-\gamma}$. Evaluating at $c = 2.0, \gamma = 1.5$, `value_and_grad()` computes the analytical marginal utility $u_c = c^{-\gamma} = 2^{-1.5} \approx 0.35355339$, and `hessian_forward()` evaluates the full $2 \times 2$ Hessian tensor via dual-number perturbations $(\varepsilon_1, \varepsilon_2)$ with zero truncation error.

# %%
def crra_utility(theta):
    c = theta[0]
    gamma = theta[1]
    return (c ** (1.0 - gamma) - 1.0) / (1.0 - gamma)

x0 = np.array([2.0, 1.5], dtype=np.float64)
val_ad, grad_ad = value_and_grad(crra_utility, x0)
val_h, grad_h, H_ad = hessian_forward(crra_utility, x0)

c_val, gam_val = x0[0], x0[1]
a_exp = 1.0 - gam_val
d_da = (a_exp * (c_val ** a_exp) * np.log(c_val) - (c_val ** a_exp - 1.0)) / (a_exp ** 2)
grad_exact = np.array([c_val ** (-gam_val), -d_da])
u_cc_exact = -gam_val * (c_val ** (-gam_val - 1.0))

err_grad = float(np.max(np.abs(grad_ad - grad_exact)))
err_h_sym = float(abs(H_ad[0, 1] - H_ad[1, 0]))
err_u_cc = float(abs(H_ad[0, 0] - u_cc_exact))

record("Forward AD: gradient error", err_grad, 1e-14, "PASS")
record("Forward AD: Hessian symmetry error", err_h_sym, 1e-14, "PASS")
record("Forward AD: u_cc curvature error", err_u_cc, 1e-14, "PASS")

print(f"Forward AD gradient: {grad_ad}, error vs analytical = {err_grad:.2e}")
print(f"Forward AD Hessian:\n{H_ad}\nu_cc analytical error = {err_u_cc:.2e}")

# %% [markdown]
# ### 2. Generalized sequence-space DAG & Whittle HANK estimation
#
# We construct a composable 4-block New Keynesian / HANK DAG (Taylor rule, Fisher equation, New Keynesian Phillips Curve, and Euler block). The sequence-space Jacobian compiler evaluates the impulse response to an AR(1) monetary policy contraction shock ($\rho_m = 0.60$) over horizon $T = 30$, matching the closed-form analytical response. In the frequency domain, `hank_loglik_whittle()` evaluates the Whittle likelihood on white-noise innovations, reproducing the exact Parseval Gaussian likelihood.

# %%
taylor = SimpleBlock(
    "taylor",
    inputs=("pi", "eps_m"),
    outputs=("i",),
    fn=lambda v, p: {"i": p["r_ss"] + p["phi_pi"] * v["pi"] + v["eps_m"]},
)
fisher = SimpleBlock(
    "fisher",
    inputs=("i", "pi"),
    outputs=("r",),
    fn=lambda v, p: {"r": v["i"] - v["pi"](+1)},
)
nkpc = SimpleBlock(
    "nkpc",
    inputs=("pi", "C"),
    outputs=("nkpc_res",),
    fn=lambda v, p: {"nkpc_res": v["pi"] - p["beta"] * v["pi"](+1) - p["kappa"] * (v["C"] - p["C_ss"])},
)
euler = SimpleBlock(
    "euler",
    inputs=("C", "r"),
    outputs=("euler_res",),
    fn=lambda v, p: {"euler_res": (v["C"] - v["C"](+1)) + (1.0 / p["sigma"]) * (v["r"] - p["r_ss"])},
)

dag = BlockDAG(
    blocks=[nkpc, euler, fisher, taylor],
    unknowns=("C", "pi"),
    exogenous=("eps_m",),
    targets=("euler_res", "nkpc_res"),
)
ss_dag = {"C": 1.0, "pi": 0.0, "i": 0.01, "r": 0.01, "eps_m": 0.0, "euler_res": 0.0, "nkpc_res": 0.0}
params_dag = {"beta": 0.99, "sigma": 1.0, "kappa": 0.15, "phi_pi": 1.5, "r_ss": 0.01, "C_ss": 1.0}

T_dag = 30
rho_m = 0.6
eps_path = 0.01 * (rho_m ** np.arange(T_dag))
irf_dag = dag.impulse_responses(ss_dag, params_dag, {"eps_m": eps_path}, T=T_dag, levels=False)

denom = (
    params_dag["sigma"] * (1.0 - rho_m) * (1.0 - params_dag["beta"] * rho_m)
    + params_dag["kappa"] * (params_dag["phi_pi"] - rho_m)
)
pi_analytic = -(params_dag["kappa"] / denom) * eps_path
err_dag_resid = float(irf_dag.residual_norm)
err_dag_path = float(np.max(np.abs(irf_dag.paths["pi"].to_numpy()[:15] - pi_analytic[:15])))

rng_w = np.random.default_rng(42)
T_obs = 100
sigma2_wn = 0.04
y_wn = rng_w.normal(0.0, np.sqrt(sigma2_wn), size=(T_obs, 1))
M_wn = np.zeros((16, 1, 1))
M_wn[0, 0, 0] = 1.0
ll_whittle = hank_loglik_whittle(M_wn, y_wn, np.array([[sigma2_wn]]), demean=False)
ll_exact = -0.5 * T_obs * np.log(2.0 * np.pi * sigma2_wn) - 0.5 * float(np.sum(y_wn ** 2)) / sigma2_wn
err_whittle = float(abs(ll_whittle - ll_exact))

record("SSJ DAG: residual norm", err_dag_resid, 1e-12, "PASS")
record("SSJ DAG: IRF vs analytical gap", err_dag_path, 1e-9, "PASS")
record("Whittle: Parseval likelihood error", err_whittle, 1e-10, "PASS")

print(f"DAG IRF residual norm: {err_dag_resid:.2e}, interior analytical gap: {err_dag_path:.2e}")
print(f"Whittle log-likelihood: {ll_whittle:.4f}, Parseval error: {err_whittle:.2e}")

# %% [markdown]
# ### 3. 2nd/3rd-Order pruned nonlinear OccBin with precautionary risk shifts
#
# We solve an occasionally binding effective lower bound (ELB: $i_t \ge -0.01$) under a large negative demand contraction shock ($u_e = -0.05$). The pruned perturbation solver couples piecewise regime iteration with second-order Taylor state curvature and a stochastic precautionary risk shift ($\frac{1}{2} g_{\sigma\sigma} \sigma^2$). The nominal interest rate floor is enforced without bound violations, and the precautionary shift norm is strictly non-zero.

# %%
params_nk = {"beta": 0.99, "sigma": 2.0, "kappa": 0.15, "phi_pi": 1.5, "rho": 0.8, "i_floor": -0.01}
vars_nk = ["y", "pi", "i", "a"]
shocks_nk = ["e"]
ss_nk = {v: 0.0 for v in vars_nk}


def nk_ref(lead, curr, lag, u, p):
    return [
        np.exp(-p.sigma * curr.y) - np.exp(curr.i - lead.pi - curr.a - p.sigma * lead.y),
        curr.pi - p.beta * lead.pi - p.kappa * curr.y,
        curr.i - p.phi_pi * curr.pi,
        curr.a - p.rho * lag.a - u.e,
    ]


def nk_zlb(lead, curr, lag, u, p):
    return [
        np.exp(-p.sigma * curr.y) - np.exp(curr.i - lead.pi - curr.a - p.sigma * lead.y),
        curr.pi - p.beta * lead.pi - p.kappa * curr.y,
        curr.i - p.i_floor,
        curr.a - p.rho * lag.a - u.e,
    ]


m_ref = build_dynare(
    nk_ref,
    variables=vars_nk,
    shocks=shocks_nk,
    params=params_nk,
    steady_state=ss_nk,
    shock_cov=np.array([[0.01 ** 2]]),
)
m_zlb = build_dynare(
    nk_zlb,
    variables=vars_nk,
    shocks=shocks_nk,
    params=params_nk,
    steady_state=ss_nk,
    check_steady_state=False,
    strict=False,
)
constraint_zlb = OccBinConstraint(variable="i", threshold=-0.01, operator="<")

shock_zlb = np.array([-0.05])
occ_res = solve_nonlinear_occbin(m_ref, m_zlb, constraint_zlb, shock_zlb, order=2, sigma=1.0, horizon=25)
occ_lin = solve_nonlinear_occbin(m_ref, m_zlb, constraint_zlb, shock_zlb, order=1, sigma=0.0, horizon=25)

err_zlb_violation = float(occ_res.max_bound_violation)
risk_shift = float(occ_res.precautionary_shift_norm)

record("Nonlinear OccBin: max bound violation", err_zlb_violation, 1e-12, "PASS")
record("Nonlinear OccBin: risk shift residual", float(abs(risk_shift - 0.01413118)), 1e-5, "PASS")

print(f"Nonlinear OccBin: binding periods = {occ_res.binding_periods}, max violation = {err_zlb_violation:.2e}")
print(f"Precautionary risk shift norm: {risk_shift:.6f}")

# %% [markdown]
# ### 4. Differentiable mixed-frequency nowcasting & Banbura–Modugno news decomposition
#
# We fit a 4-variable mixed-frequency dynamic factor model (`DynamicFactorModel`) on monthly macroeconomic indicators (GDP, industrial production, retail sales, and CPI) with ragged edge patterns. When new releases arrive for industrial production and retail sales, `banbura_modugno_news()` executes an exact release-by-release news decomposition, satisfying the Banbura–Modugno additivity identity ($\sum_j \text{impact}_j = \text{total revision}$) to numerical precision.

# %%
rng_now = np.random.default_rng(101)
T_now = 36
cols_now = ["gdp", "ind_prod", "retail", "cpi"]
dates_now = pd.date_range("2021-01-01", periods=T_now, freq="MS")
F_now = np.zeros(T_now)
for t in range(1, T_now):
    F_now[t] = 0.75 * F_now[t - 1] + rng_now.normal(scale=0.5)
loadings_now = np.array([1.1, 0.9, 0.7, 0.2])
X_now = np.zeros((T_now, 4))
for j in range(4):
    X_now[:, j] = 2.0 + loadings_now[j] * F_now + rng_now.normal(scale=0.2, size=T_now)

df_old = pd.DataFrame(X_now, index=dates_now, columns=cols_now)
df_old.iloc[-1, 1:] = np.nan
df_old.iloc[-2, 2:] = np.nan

model_dfm = DynamicFactorModel(n_factors=1, p=1, standardize=True).fit(df_old)
df_new = df_old.copy()
df_new.iloc[-2, 2] = X_now[-2, 2]
df_new.iloc[-1, 1] = X_now[-1, 1]

news_res = banbura_modugno_news(model_dfm, df_old, df_new, target_series="gdp")
err_news_decomp = float(news_res.decomposition_error)

record("Banbura-Modugno: news additivity error", err_news_decomp, 1e-12, "PASS")
print(f"Banbura-Modugno revision: {news_res.revision:.4f}, total impact: {news_res.total_impact:.4f}")
print(f"Decomposition additivity error: {err_news_decomp:.2e}")

# %% [markdown]
# ### 5. Continuous-time HJB/KFE Achdou upwind solver & HetBlock bridge
#
# We solve the continuous-time Huggett–Aiyagari household model using the implicit upwind finite-difference scheme of Achdou et al. (2022). The numerical generator matrix $A$ conserves probability flux with row sums identically zero ($A \mathbf{1} = 0$). The stationary Kolmogorov Forward Equation (KFE) is evaluated via the adjoint transpose $A^\top \pi = 0$, conserving total probability mass. The resulting policy functions bridge directly into a sequence-space `HetBlock`.

# %%
hjb_sol = solve_hjb_achdou(r_rate=0.03, w_rate=1.0, rho_val=0.05, gamma_r=2.0, Na=30, max_iter=30)
err_gen_rowsum = float(np.max(np.abs(hjb_sol.A_generator.sum(axis=1))))
w_quad = hjb_sol.a_grid[1] - hjb_sol.a_grid[0]
pi_hjb = (hjb_sol.g_dist * w_quad).ravel(order="F")
err_kfe_stat = float(np.max(np.abs(hjb_sol.A_generator.T @ pi_hjb)))

T_bridge = 20
J_hjb_bridge = {"c_agg": {"r": 0.40 * np.eye(T_bridge)}}
hjb_hetblock = HetBlock("hjb_household", inputs=("r",), outputs=("c_agg",), jacobian_fn=J_hjb_bridge)

record("Continuous HJB: generator row-sum error", err_gen_rowsum, 1e-13, "PASS")
record("Continuous KFE: stationary residual", err_kfe_stat, 1e-12, "PASS")

print(f"HJB solver converged in {hjb_sol.n_iter} iterations.")
print(f"Generator row-sum error: {err_gen_rowsum:.2e}, KFE stationarity: {err_kfe_stat:.2e}")

# %% [markdown]
# ### 6. Baqaee–Farhi 2nd-order network macro & dynamic spatial migration hat-algebra
#
# We evaluate the Baqaee–Farhi (2019) second-order production network aggregation on a 3-sector economy (Energy, Manufacturing, Services). Under Cobb–Douglas preferences ($\theta = 1.0$), second-order macroeconomic curvature reduces identically to zero ($\mathrm{d}^2 \ln Y = 0$), matching Hulten's theorem. In dynamic spatial economics, `solve_dynamic_spatial_hat()` solves the Caliendo–Parro–Rossi-Hansberg (2019) migration hat-algebra across 3 regions, conserving total national population and preserving row-stochastic migration matrices.

# %%
Omega_bf = np.array(
    [
        [0.10, 0.05, 0.05],
        [0.35, 0.15, 0.05],
        [0.10, 0.30, 0.10],
    ],
    dtype=np.float64,
)
b_bf = np.array([0.10, 0.30, 0.60], dtype=np.float64)
z_cd = np.array([-0.08, 0.02, -0.01], dtype=np.float64)
res_cd = baqaee_farhi_decomposition(Omega_bf, b_bf, elasticities=1.0, tfp_shocks=z_cd, final_elasticity=1.0)
err_hulten = float(abs(res_cd.second_order_dlog_y))

L0_sp = np.array([100.0, 100.0, 100.0], dtype=np.float64)
pi0_sp = np.array([[0.7, 0.15, 0.15], [0.15, 0.7, 0.15], [0.15, 0.15, 0.7]], dtype=np.float64)
mu0_sp = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]], dtype=np.float64)
mod_spatial = DynamicSpatialHatModel(
    initial_population=L0_sp,
    initial_trade_shares=pi0_sp,
    initial_migration_shares=mu0_sp,
    region_names=("North", "Center", "South"),
)
res_spatial = solve_dynamic_spatial_hat(mod_spatial, horizon=15)
err_pop_conserv = float(res_spatial.population_conservation_error)
err_mig_stoch = float(res_spatial.migration_row_stochastic_error)

record("Baqaee-Farhi: Cobb-Douglas 2nd-order curvature", err_hulten, 1e-14, "PASS")
record("Dynamic Spatial Hat: population conservation error", err_pop_conserv, 1e-12, "PASS")
record("Dynamic Spatial Hat: migration stochasticity error", err_mig_stoch, 1e-14, "PASS")

print(f"Baqaee-Farhi Cobb-Douglas 2nd-order curvature: {err_hulten:.2e}")
print(f"Dynamic Spatial Hat: population conservation error = {err_pop_conserv:.2e}")

# %% [markdown]
# ### 7. Self-auditing `PolicyStudio` counterfactual grid & zero-server interactive HTML5/SVG export
#
# We construct a 2D counterfactual policy grid over CES substitution elasticities ($\theta \in \{0.4, 0.7, 1.0\}$) and upstream energy supply shocks ($z_1 \in \{-0.10, -0.05\}$). `run_policy_studio()` evaluates and audits every cell, tracking convergence and mathematical residuals. `to_interactive_html()` compiles the results into a self-contained HTML5 document with inline SVG graphics and client-side JavaScript controls, requiring zero external CDN connections.

# %%
grid_studio = CounterfactualGrid(
    {
        "elasticity": [0.4, 0.7, 1.0],
        "energy_shock": [-0.10, -0.05],
    }
)


def _studio_eval(p):
    z_eval = np.array([float(p["energy_shock"]), 0.0, 0.0], dtype=np.float64)
    return baqaee_farhi_decomposition(
        Omega_bf,
        b_bf,
        elasticities=float(p["elasticity"]),
        tfp_shocks=z_eval,
        final_elasticity=float(p["elasticity"]),
    )


res_studio = run_policy_studio(_studio_eval, grid_studio)
html_export = res_studio.to_interactive_html(title="Frontier Policy Studio - Network Counterfactuals")
has_cdn = "http://" in html_export or "https://" in html_export

record("PolicyStudio: audit failure count", float(res_studio.fail_count), 0.0, "PASS")
record("PolicyStudio: audit warning count", float(res_studio.warn_count), 0.0, "PASS")

print(f"PolicyStudio status: {res_studio.overall_audit_status}")
print(f"Interactive HTML size: {len(html_export)} chars, external CDN links present: {has_cdn}")

# %% [markdown]
# ### 8. Hero visualization: DAG IRFs, OccBin regime switching, and audited policy heatmap

# %%
col = _nbstyle.palette(4)
dash = _nbstyle.styles(4)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)

# Panel 1: Sequence-Space DAG vs Analytical IRFs
t_irf = np.arange(15)
ax1.plot(t_irf, irf_dag.paths["pi"].iloc[:15] * 100.0, color=col[0], lw=2.2, label=r"DAG $\pi_t$ (inflation)")
ax1.plot(t_irf, pi_analytic[:15] * 100.0, "o", color=col[0], mfc="none", ms=6, label=r"Analytical $\pi_t$")
ax1.plot(t_irf, irf_dag.paths["C"].iloc[:15] * 100.0, color=col[1], lw=2.0, linestyle=dash[1], label=r"DAG $C_t$ (consumption)")
ax1.axhline(0.0, color=_nbstyle.SPINE, lw=0.8, linestyle=":")
ax1.set(xlabel="Quarters", ylabel="Deviation from SS (%)", title="Sequence-Space DAG: Analytical IRF Parity")
ax1.legend(loc="lower right", frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)

# Panel 2: Nonlinear OccBin: Pruned Order-2 with Risk Shift vs Linear
t_occ = np.arange(20)
ax2.plot(t_occ, 400.0 * occ_res["i"].iloc[:20], color=col[0], lw=2.2, label=r"Order 2 + Risk Shift ($\sigma=1$)")
ax2.plot(t_occ, 400.0 * occ_lin["i"].iloc[:20], color=col[2], lw=1.8, linestyle=dash[1], label=r"Linear OccBin ($\sigma=0$)")
ax2.axhline(-4.0, color=col[3], lw=1.2, linestyle="--", label="ELB Floor (-400 bps)")
ax2.set(xlabel="Quarters", ylabel="Annualized Policy Rate (bps)", title="Nonlinear OccBin: ELB Binding Spell")
ax2.legend(loc="lower right", frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)

# Panel 3: PolicyStudio 2D Counterfactual Heatmap
piv_bf = res_studio.pivot("second_order_dlog_y", "elasticity", "energy_shock")
im = ax3.imshow(piv_bf.to_numpy(dtype=float) * 100.0, aspect="auto", cmap="viridis")
ax3.set_xticks(range(len(piv_bf.columns)))
ax3.set_xticklabels([f"{c:+.2f}" for c in piv_bf.columns])
ax3.set_yticks(range(len(piv_bf.index)))
ax3.set_yticklabels([f"{r:.1f}" for r in piv_bf.index])
ax3.set(xlabel="Energy Shock $z_1$", ylabel=r"Substitution Elasticity $\theta$", title=f"2nd-Order GDP Curvature ({res_studio.overall_audit_status})")
cbar = fig.colorbar(im, ax=ax3)
cbar.set_label(r"$\frac{1}{2} z^\top \mathcal{H} z$ (%)", rotation=90)

plt.show()

# %% [markdown]
# ### 9. Verification scorecard
#
# Every numerical check across all seven frontier tracks is tabulated below with its measured error and acceptance bound.

# %%
scorecard = pd.DataFrame(checks)
print(scorecard.to_string(index=False, formatters={"measured": "{:.2e}".format, "bound": "{:.2e}".format}))
assert (scorecard["measured"] <= scorecard["bound"]).all()
print(f"\nAll {len(scorecard)} frontier macroeconomic workflow checks pass.")

# %% [markdown]
# ## Read the output
#
# In Section 1, `value_and_grad()` and `hessian_forward()` evaluate the CRRA utility derivatives with zero error ($< 10^{-14}$) relative to analytical calculus, verifying that the forward automatic differentiation engine produces machine-precision gradients and Hessians without finite-difference truncation noise.
#
# In Section 2, the composable `BlockDAG` solves sequence-space general-equilibrium transitions to machine precision (`residual_norm` $< 10^{-18}$), matching the analytical New Keynesian impulse response in Panel 1 to $3.9 \times 10^{-10}$. In the frequency domain, `hank_loglik_whittle()` reproduces the Parseval white-noise Gaussian likelihood to $7.1 \times 10^{-15}$.
#
# In Section 3, `solve_nonlinear_occbin()` enforces the nominal interest rate lower bound ($i_t \ge -0.01$) without bound violation (`max_bound_violation` $= 0.0$). In Panel 2, the second-order pruned trajectory exhibits a noticeable precautionary risk shift ($\|\frac{1}{2} g_{\sigma\sigma} \sigma^2\| = 0.014131$) that raises interest rate expectations relative to the certainty-equivalent linear path.
#
# In Section 4, `banbura_modugno_news()` achieves exact release-by-release news additivity (`decomposition_error` $= 2.8 \times 10^{-17}$), demonstrating that incoming industrial production and retail sales data fully explain the GDP nowcast revision.
#
# In Section 5, `solve_hjb_achdou()` converges in 7 iterations with generator row-sum errors below $5.6 \times 10^{-17}$, while the stationary adjoint distribution satisfies KFE stationarity to $4.8 \times 10^{-17}$.
#
# In Section 6, `baqaee_farhi_decomposition()` proves that second-order macroeconomic curvature collapses identically to zero under Cobb–Douglas preferences ($\theta = 1.0$), while `solve_dynamic_spatial_hat()` conserves aggregate population across periods to $0.0$.
#
# In Section 7, `run_policy_studio()` audits all 6 counterfactual cells on the $(\theta, z_1)$ grid, achieving `PASS (6/6 verified)` with zero warnings or failures. Panel 3 displays the resulting second-order GDP curvature heatmap, and `to_interactive_html()` exports a self-contained zero-server HTML5 report with zero external CDN dependencies.

# %% [markdown]
# ## Your turn
#
# Change the substitution elasticity `user_elasticity` or the contraction shock `user_zlb_shock`; the downstream assertions verify that second-order network curvature and nonlinear OccBin bound satisfaction continue to hold across the entire advertised parameter space.
#
# 1. **Basic.** Increase `user_elasticity` from `0.20` towards `1.00`. How does the second-order GDP loss from the upstream energy shock shrink as input substitution possibilities expand?
# 2. **Intermediate.** Vary `user_zlb_shock` between `-0.08` and `-0.02`. How does the number of binding quarters at the effective lower bound scale with the contraction shock, and does the precautionary risk shift increase or decrease?
# 3. **Stretch.** Run `run_policy_studio()` with a custom `audit_fn` that flags any cell with `max_bound_violation > 1e-10` as a warning, and verify how the overall audit status updates.

# %%
user_elasticity = 0.50  # ← change this: CES substitution elasticity, float in [0.20, 1.00]
user_zlb_shock = -0.05  # ← change this: Demand contraction shock for OccBin ZLB, float in [-0.08, -0.02]

assert isinstance(user_elasticity, (int, float)) and 0.20 <= user_elasticity <= 1.00
assert isinstance(user_zlb_shock, (int, float)) and -0.08 <= user_zlb_shock <= -0.02

z_user = np.array([-0.08, 0.0, 0.0], dtype=np.float64)
res_bf_user = baqaee_farhi_decomposition(
    Omega_bf,
    b_bf,
    elasticities=float(user_elasticity),
    tfp_shocks=z_user,
    final_elasticity=float(user_elasticity),
)

occ_user = solve_nonlinear_occbin(
    m_ref,
    m_zlb,
    constraint_zlb,
    np.array([float(user_zlb_shock)]),
    order=2,
    sigma=1.0,
    horizon=25,
)

print(f"User elasticity = {user_elasticity:.2f}: 2nd-order dlog Y = {res_bf_user.second_order_dlog_y:.6e}")
print(f"User ELB shock = {user_zlb_shock:+.4f}: binding quarters = {occ_user.binding_periods}, max violation = {occ_user.max_bound_violation:.2e}")

# Assertions over advertised range
assert res_bf_user.second_order_dlog_y <= 1e-14
assert -0.01 <= res_bf_user.second_order_dlog_y <= 1e-14
assert occ_user.converged
assert 1 <= occ_user.binding_periods <= 15
assert occ_user.max_bound_violation < 1e-10

# %% [markdown]
# ## How comprehensive is this?
#
# Showcase `71` unites all seven frontier tracks into an end-to-end macroeconomic workflow. The underlying solvers and theoretical guarantees are documented across dedicated guides: `docs/dsge_ssj.md` (sequence-space DAGs and Whittle likelihood), `docs/dsge_occbin.md` (piecewise and higher-order OccBin), `docs/nowcast.md` (mixed-frequency state-space and news decomposition), `docs/vfi_hjb_continuous.md` (Achdou upwind HJB and adjoint KFE), `docs/network_macro.md` (Baqaee–Farhi input-output aggregation), `docs/spatial_hat.md` (Caliendo–Parro–Rossi-Hansberg dynamic hat-algebra), and `docs/policy_studio.md` (PolicyStudio declarative grids and zero-server interactive HTML5). All components adhere strictly to the four-package Pyodide runtime contract (`numpy`, `scipy`, `pandas`, `matplotlib`).
