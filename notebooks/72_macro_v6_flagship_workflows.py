# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # PureMacro v6 workflows: testing the calculations and their scope
#
# This notebook exercises native DSGE likelihoods, a Gaussian SMC target, reduced HANK demonstrations, sovereign debt calculations, synthetic distributional LPs and a spatial climate example. Its 14 checks test specified numerical properties. They do not validate a single integrated economic model or replicate published policy results.
#
# **Status, 2026-10-26:** source v6.0.0 was tagged but its PyPI publication failed. Use the reliability development checkout for these corrected examples.

# %% [markdown]
# ## The method in math
#
# The native DSGE score is $\nabla_\theta\log p(Y\mid\theta)$ from a Kalman sensitivity recursion; structural state-space derivatives may use finite differences. The SMC example targets an analytic Gaussian likelihood. Distributional share responses satisfy $\sum_b\beta_{b,h}=0$ by projection. The aggregate-risk demonstration has a prescribed correction proportional to $\sigma^2$; quadratic scaling is an algebra check, not evidence of endogenous precautionary saving.

# %% [markdown]
# ## Intuition
#
# Ask what each check establishes. A small residual verifies the supplied equations; a derivative comparison verifies a local score; adding-up verifies accounting. Economic validation additionally requires household optimality, documented equilibrium closure, independent references and convergence as grids and horizons increase.


# %% [markdown]
# ## Worked code

# %%
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

repo = Path.cwd() if (Path.cwd() / "puremacro").is_dir() else Path.cwd().parent
sys.path.insert(0, str(repo))
sys.path.insert(0, str(repo / "notebooks"))
import _nbstyle
_nbstyle.apply_style()

from puremacro.core.protocol import DifferentiableModel, wrap_differentiable_model
from puremacro.inference.smc import smc_sample
from puremacro.dsge.hank_nonlinear import (
    solve_hank_nonlinear_transition,
    solve_hank_aggregate_risk_order2,
)
from puremacro.sovereign import (
    solve_sovereign_default,
    solve_sudden_stop,
    decompose_spread_term_structure,
    evaluate_debt_restructuring,
)
from puremacro.lp.distributional import (
    estimate_distributional_lp,
    bridge_hank_distributional_irf,
)
from puremacro.climate.spatial_iam import (
    SpatialClimateIAM,
    solve_spatial_climate_iam,
)
from puremacro.dsge import build_dynare

checks = []


def record(result, measured, bound, verdict):
    checks.append({"result": result, "measured": float(measured), "bound": float(bound), "verdict": verdict})


# %% [markdown]
# ### 1. Native DSGE likelihood and structural scores
#
# We solve a three-equation New Keynesian model, simulate explicit data and evaluate the likelihood using its actual equations and shock loadings. The selected parameters are beta, sigma and kappa. Scores use the native Kalman recursion, with structural sensitivities that may use finite differences. The derivative comparison below is a consistency check, not a claim of end-to-end DualArray differentiation.


# %%
params_proto = {
    "beta": 0.99,
    "sigma": 1.0,
    "kappa": 0.12,
    "phi_pi": 1.5,
    "phi_y": 0.125,
    "rho_r": 0.75,
    "rho_a": 0.85,
    "sigma_a": 0.01,
    "sigma_r": 0.005,
}
variables_proto = ["y", "pi", "i", "a"]
shocks_proto = ["e_a", "e_r"]
ss_proto = {v: 0.0 for v in variables_proto}


def equations_proto(lead, curr, lag, u, p):
    return [
        curr.y - lead.y + (1.0 / p.sigma) * (curr.i - lead.pi - curr.a),
        curr.pi - p.beta * lead.pi - p.kappa * curr.y,
        curr.i - p.rho_r * lag.i - (1.0 - p.rho_r) * (p.phi_pi * curr.pi + p.phi_y * curr.y) - u.e_r,
        curr.a - p.rho_a * lag.a - u.e_a,
    ]


dsge_raw = build_dynare(equations_proto, variables=variables_proto, shocks=shocks_proto, params=params_proto, steady_state=ss_proto)
wrapped_model = wrap_differentiable_model(dsge_raw, param_names=["beta", "sigma", "kappa"])
assert isinstance(wrapped_model, DifferentiableModel)

sim_data_proto = wrapped_model.simulate(periods=50, seed=42)[["y", "pi"]].values
param_vec_proto = np.array([0.99, 1.0, 0.12])
loglik_proto, grad_proto = wrapped_model.loglik_and_grad(param_vec_proto, sim_data_proto, observables=["y", "pi"])

h_step = 1e-5
fd_grad_proto = np.zeros_like(grad_proto)
for p_idx in range(len(param_vec_proto)):
    e_p = np.zeros_like(param_vec_proto)
    e_p[p_idx] = h_step
    ll_plus, _ = wrapped_model.loglik_and_grad(param_vec_proto + e_p, sim_data_proto, observables=["y", "pi"])
    ll_minus, _ = wrapped_model.loglik_and_grad(param_vec_proto - e_p, sim_data_proto, observables=["y", "pi"])
    fd_grad_proto[p_idx] = (ll_plus - ll_minus) / (2.0 * h_step)

err_proto_grad = float(np.max(np.abs(grad_proto - fd_grad_proto)))
err_tombstones = 0.0
try:
    import puremacro.lp.garch_utils  # noqa: F401
    err_tombstones += 1.0
except ImportError:
    pass

import puremacro.sigma.sigma_numpy as s_numpy
if hasattr(s_numpy, "SigmaObject"):
    err_tombstones += 1.0

record("Protocol: Native score gradient error", err_proto_grad, 1e-4, "PASS")
record("Protocol: v6.0.0 legacy deprecation tombstones", err_tombstones, 0.0, "PASS")

print(f"Protocol loglik: {loglik_proto:.4f}, Native score: {grad_proto}")
print(f"Gradient error vs central finite differences: {err_proto_grad:.2e}")

# %% [markdown]
# ### 2. Tempered SMC on a Gaussian reference target
#
# Adaptive tempering and MALA proposals are checked against an analytic marginal likelihood. This example tests inference machinery; it does not estimate HANK parameters. The log marginal-likelihood estimator is not generally unbiased.


# %%
def smc_target(theta):
    diff = theta - np.array([0.99, 1.0])
    return -0.5 * np.sum(diff ** 2), -diff


priors_smc = {
    "beta": {"dist": "normal", "mean": 0.99, "std": 0.05},
    "sigma": {"dist": "normal", "mean": 1.0, "std": 0.20},
}
res_smc = smc_sample(smc_target, priors=priors_smc, n_particles=100, mutation="mala", seed=42)
err_smc_mdd = float(abs(res_smc.log_mdd - 0.0))
err_smc_term = float(abs(res_smc.stage_tempering[-1] - 1.0))

record("SMC Inference: marginal data density error", err_smc_mdd, 0.15, "PASS")
record("SMC Inference: terminal tempering stage error", err_smc_term, 0.0, "PASS")

print(f"SMC stages: {len(res_smc.stage_tempering)}, terminal phi: {res_smc.stage_tempering[-1]:.4f}")
print(f"Estimated log MDD: {res_smc.log_mdd:.4f} (error vs analytical = {err_smc_mdd:.4f})")

# %% [markdown]
# ### 3. Reduced nonlinear and aggregate-risk demonstrations
#
# The default DAG uses prescribed MPC and constraint-fraction formulas. It does not solve household policies or a wealth distribution. The aggregate-risk example uses prescribed moments and matrices, and emits an explicit warning. Its fourfold response to doubled volatility follows from its quadratic formula. For the independently checked one-asset household model, see docs/hank_ssj_and_aggregate_risk.md.


# %%
res_nl = solve_hank_nonlinear_transition(model=None, shocks={"eps_m": 1e-4}, horizon=25, tol=1e-8)
max_gap = float(np.max(np.abs(res_nl.paths["Y"].values - res_nl.linear_paths["Y"].values)))

res_ar1 = solve_hank_aggregate_risk_order2(model=None, sigma_agg=0.01)
res_ar2 = solve_hank_aggregate_risk_order2(model=None, sigma_agg=0.02)
ratio_vol = float(res_ar2.precautionary_shift_wealth / res_ar1.precautionary_shift_wealth)
err_quad = float(abs(ratio_vol - 4.0))

record("Reduced DAG: transition residual norm", float(res_nl.residual_norm), 1e-8, "PASS")
record("Reduced DAG: small-shock linear SSJ parity gap", max_gap, 1e-4, "PASS")
record("Reduced risk algebra: quadratic volatility scaling error", err_quad, 1e-6, "PASS")

print(f"Nonlinear HANK residual: {res_nl.residual_norm:.2e}, linear SSJ gap: {max_gap:.2e}")
print(f"Precautionary wealth shift (sigma=0.01): {res_ar1.precautionary_shift_wealth:.6f}")
print(f"Quadratic volatility scaling ratio: {ratio_vol:.4f} (expected = 4.0000)")

# %% [markdown]
# ### 4. Sovereign debt and restructuring calculations
#
# These examples exercise price schedules, solver residuals and restructuring outputs. Simulated correlations are reported without sign adjustments. The separate deterministic short-debt benchmark checks Arellano equations on matched grids; long-term debt and sudden-stop examples are not certified replications of published calibrations.


# %%
res_sov = solve_sovereign_default(beta=0.95, gamma=2.0, r_star=0.015, n_b=30, n_y=9, max_iter=250)
q_max = 1.0 / (1.0 + 0.015) + 1e-8
price_violation = float(np.max(np.maximum(0.0, res_sov.bond_price - q_max)) + np.max(np.maximum(0.0, -res_sov.bond_price)))
mean_spread = np.mean(res_sov.spread, axis=0)
corr_y = float(np.corrcoef(mean_spread, res_sov.y_grid)[0, 1])

restruct = evaluate_debt_restructuring(res_sov, haircut=0.25, maturity_extension=0.0)

record("Sovereign Default: Bellman fixed-point residual", float(res_sov.residual), 1e-4, "PASS")
record("Sovereign Default: bond price schedule bounds violation", price_violation, 0.0, "PASS")
record("Sovereign Default: countercyclical spread-output correlation", corr_y, 0.0, "PASS")

print(f"Sovereign Bellman residual: {res_sov.residual:.2e}, bond price violation: {price_violation:.2e}")
print(f"Spread-output correlation: {corr_y:.4f}, haircut spread reduction: {restruct.spread_reduction_bps:.2f} bps")

# %% [markdown]
# ### 5. Distributional LPs on a synthetic panel
#
# The adding-up check tests share accounting. The bridge receives the same response matrix twice, so zero distance is an identity check only. It does not compare empirical estimates with independently solved HANK deciles.


# %%
rng_dist = np.random.default_rng(42)
T_lp, N_lp = 35, 40
records_lp = []
shock_ts = rng_dist.normal(0, 1, size=T_lp)
for t_idx in range(T_lp):
    for i_idx in range(N_lp):
        u_shock = shock_ts[t_idx] + 0.2 * rng_dist.normal(0, 1)
        s_base = rng_dist.dirichlet([2, 3, 5, 3, 2])
        s_val = s_base + u_shock * np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
        s_val = np.clip(s_val, 0.001, 0.999)
        s_val = s_val / np.sum(s_val)
        records_lp.append({"t": t_idx, "i": f"u_{i_idx}", "shock": u_shock, **{f"s{k}": s_val[k] for k in range(5)}})

df_lp = pd.DataFrame(records_lp)
res_lp = estimate_distributional_lp(
    df_lp,
    shock="shock",
    bin_cols=[f"s{k}" for k in range(5)],
    horizons=range(8),
    cov_type="driscoll_kraay",
    time_col="t",
    unit_col="i",
    enforce_adding_up=True,
)
bridge_res = bridge_hank_distributional_irf(res_lp, res_lp.irf_matrix)

record("Distributional LP: cross-sectional share adding-up error", float(res_lp.adding_up_max_error), 1e-12, "PASS")
record("Bridge identity: response-matrix L2 distance", float(bridge_res.distance_l2), 1.0, "PASS")

print(f"Distributional LP adding-up max error: {res_lp.adding_up_max_error:.2e}")
print(f"Bridge identity L2 distance: {bridge_res.distance_l2:.2e}, sign concordance: {bridge_res.sign_concordance:.4f}")

# %% [markdown]
# ### 6. Spatial climate example
#
# We inspect population conservation and carbon accounting under the selected calibration. Passing these identities does not independently validate migration behavior, damage coefficients or policy welfare effects.


# %%
model_iam = SpatialClimateIAM(
    initial_population=np.array([100.0, 150.0, 80.0]),
    initial_trade_shares=np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]),
    initial_migration_shares=np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]),
    initial_temperatures=np.array([14.0, 22.0, 28.0]),
    carbon_intensity=np.array([0.4, 0.6, 0.5]),
    damage_coefficients=np.array([0.005, 0.015, 0.025]),
    region_names=("Temperate", "Subtropical", "Tropical"),
)
res_cbam = solve_spatial_climate_iam(
    model_iam,
    horizon=15,
    policy_scenario="cbam",
    carbon_tax_path=np.linspace(20, 80, 15),
    coalition_regions=[0],
)

record("Spatial Climate IAM: global population conservation error", float(res_cbam.population_conservation_error), 1e-12, "PASS")
record("Spatial Climate IAM: atmospheric carbon mass balance error", float(res_cbam.carbon_mass_balance_error), 1e-12, "PASS")

print(f"Population conservation error: {res_cbam.population_conservation_error:.2e}")
print(f"Carbon mass balance error: {res_cbam.carbon_mass_balance_error:.2e}")
print(f"CBAM carbon leakage rate: {res_cbam.carbon_leakage_rate:.4f}")

# %% [markdown]
# ### 7. Displaying separate policy scenarios together
#
# The climate path is solved here. The LP and restructuring statistics printed alongside it come from separate earlier calculations. There is no joint climate–distribution–sovereign equilibrium or estimated feedback mechanism in this example.


# %%
scenario_tax_path = np.linspace(25.0, 95.0, 15)
res_scenario_iam = solve_spatial_climate_iam(
    model_iam,
    horizon=15,
    policy_scenario="cbam",
    carbon_tax_path=scenario_tax_path,
    coalition_regions=[0],
)

temp_warming = res_scenario_iam.temperature_path[-1] - res_scenario_iam.temperature_path[0]
pop_shift_tropical = res_scenario_iam.population_path[-1, 2] - res_scenario_iam.population_path[0, 2]
pop_shift_temperate = res_scenario_iam.population_path[-1, 0] - res_scenario_iam.population_path[0, 0]

print("Separate scenario outputs:")
print(f"  Temperate Coalition carbon tax: ${scenario_tax_path[0]:.1f} -> ${scenario_tax_path[-1]:.1f}/tCO2")
print(f"  Regional warming after 15 periods: {temp_warming[0]:+.2f}C (North), {temp_warming[2]:+.2f}C (South)")
print(f"  Net climate migration: Tropical {pop_shift_tropical:+.2f}M, Temperate {pop_shift_temperate:+.2f}M")
print(f"  Distributional LP adding-up satisfied: {res_lp.adding_up_max_error < 1e-12}")
print(f"  Sovereign debt restructuring headroom: {abs(restruct.spread_reduction_bps):.2f} bps")

# %% [markdown]
# ### 8. Comparing the component outputs
#
# The panels display separate distributional, climate and sovereign calculations with their stated assumptions.

# %%
col = _nbstyle.palette(4)
dash = _nbstyle.styles(4)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)

# Panel 1: Synthetic share IRFs
h_vec = np.arange(8)
ax1.plot(h_vec, res_lp.irf_matrix[:, 0] * 100.0, color=col[0], lw=2.2, label=r"Bin 1 ($s_1$)")
ax1.plot(h_vec, res_lp.irf_matrix[:, 1] * 100.0, color=col[1], lw=1.8, linestyle=dash[1], label=r"Bin 2 ($s_2$)")
ax1.plot(h_vec, res_lp.irf_matrix[:, 2] * 100.0, color=col[2], lw=1.5, linestyle=dash[2], label=r"Bin 3 ($s_3$)")
ax1.plot(h_vec, res_lp.irf_matrix[:, 4] * 100.0, color=col[3], lw=2.2, linestyle=dash[3], label=r"Bin 5 ($s_5$)")
ax1.axhline(0.0, color=_nbstyle.SPINE, lw=0.8, linestyle=":")
ax1.set(xlabel="Horizons (Quarters)", ylabel="Share Change (pp)", title="Distributional LP: Adding-Up")
ax1.legend(loc="upper right", frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)

# Panel 2: Spatial Climate IAM Population Migration
t_vec = np.arange(15)
ax2.plot(t_vec, res_scenario_iam.population_path[:, 0], color=col[0], lw=2.2, label="Temperate (North)")
ax2.plot(t_vec, res_scenario_iam.population_path[:, 1], color=col[1], lw=2.0, linestyle=dash[1], label="Subtropical (Center)")
ax2.plot(t_vec, res_scenario_iam.population_path[:, 2], color=col[2], lw=1.8, linestyle=dash[2], label="Tropical (South)")
ax2.set(xlabel="Years", ylabel="Regional Population", title="Spatial Climate IAM: Migration")
ax2.legend(loc="center right", frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)

# Panel 3: Sovereign Default Bond Price Schedules
b_axis = np.linspace(0.0, 1.0, 30)
q_high = res_sov.bond_price[:, -1]
q_mid = res_sov.bond_price[:, len(res_sov.y_grid) // 2]
q_low = res_sov.bond_price[:, 0]
ax3.plot(b_axis, q_high, color=col[0], lw=2.2, label="High Income State")
ax3.plot(b_axis, q_mid, color=col[1], lw=2.0, linestyle=dash[1], label="Mean Income State")
ax3.plot(b_axis, q_low, color=col[2], lw=1.8, linestyle=dash[2], label="Low Income State")
ax3.axhline(1.0 / (1.0 + 0.015), color=col[3], lw=1.0, linestyle="--", label="Risk-Free Upper Bound")
ax3.set(xlabel="Borrowing Choice b'", ylabel="Bond Price q(b', y)", title="Sovereign Default: Bond Pricing")
ax3.legend(loc="lower left", frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)

# %% [markdown]
# ### 9. Verification scorecard
#
# Each row reports a measured numerical property and its tolerance. Demonstration and identity checks retain the limited scope explained above.

# %%
scorecard = pd.DataFrame(checks)
print(scorecard.to_string(index=False, formatters={"measured": "{:.2e}".format, "bound": "{:.2e}".format}))
assert (scorecard["measured"] <= scorecard["bound"]).all()
print(f"\nAll {len(scorecard)} v6.0.0 flagship macroeconomic workflow checks pass.")

# %% [markdown]
# ## Read the output
#
# The native DSGE likelihood changes with the structural model. Small transition residuals refer to the reduced DAG. The aggregate-risk scaling and zero bridge distance follow by construction. The sovereign and climate outputs depend on their calibrations. Consult the independent reference manifests before interpreting any of these numbers as a research result.
# %% [markdown]
# ## Your turn
#
# Change user_carbon_tax within [10, 100] and user_debt_haircut within [0.05, 0.40]. Compare the separate scenarios and inspect the numerical assertions. These knobs do not create a jointly solved model. A useful next experiment is to vary sovereign grids and simulation seeds, then compare uncertainty in policy effects with numerical approximation error.

# %%
user_carbon_tax = 45.0  # ← change this: Carbon tax in coalition region ($/tCO2), float in [10.0, 100.0]
user_debt_haircut = 0.20  # ← change this: Sovereign debt restructuring haircut, float in [0.05, 0.40]

assert isinstance(user_carbon_tax, (int, float)) and 10.0 <= user_carbon_tax <= 100.0
assert isinstance(user_debt_haircut, (int, float)) and 0.05 <= user_debt_haircut <= 0.40

res_iam_user = solve_spatial_climate_iam(
    model_iam,
    horizon=12,
    policy_scenario="cbam",
    carbon_tax_path=np.full(12, float(user_carbon_tax)),
    coalition_regions=[0],
)

restruct_user = evaluate_debt_restructuring(
    res_sov,
    haircut=float(user_debt_haircut),
    maturity_extension=0.0,
)

print(f"User Carbon Tax = ${user_carbon_tax:.1f}/tCO2: Carbon Leakage Rate = {res_iam_user.carbon_leakage_rate:.4f}, Pop Conservation Error = {res_iam_user.population_conservation_error:.2e}")
print(f"User Restructuring Haircut = {user_debt_haircut * 100:.1f}%: Spread Reduction = {restruct_user.spread_reduction_bps:.2f} bps, Post Default Prob = {restruct_user.post_default_prob:.4f}")

# Assertions over advertised parameter range
assert 0.0 <= res_iam_user.carbon_leakage_rate <= 1.0
assert res_iam_user.population_conservation_error < 1e-12
assert res_iam_user.carbon_mass_balance_error < 1e-12
assert restruct_user.spread_reduction_bps <= 0.0
assert restruct_user.post_default_prob <= restruct_user.pre_default_prob + 1e-6

# %% [markdown]
# ## How comprehensive is this?
#
# This is a component smoke test. Research validation starts with tools/reference_validation/validate_arellano_reference.py and validate_hank_reference.py, then requires published calibrations, grid and horizon convergence, and explicit policy counterfactuals. The guides docs/autodiff_and_protocol.md and docs/hank_ssj_and_aggregate_risk.md describe supported and demonstration operations.

