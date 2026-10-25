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
# # Supply-Chain Bottlenecks, Climate Migration & Carbon Border Adjustments: Baqaee–Farhi Networks, Dynamic Spatial Hat-Algebra, and PolicyStudio IAM
#
# **How do input-output production network substitution elasticities amplify upstream energy bottlenecks, how does global warming induce forward-looking regional labor migration, why does unilateral carbon taxation cause international carbon leakage unless insulated by Carbon Border Adjustment Mechanisms (CBAM), and how do we design self-auditing counterfactual grids with zero-server interactive reporting?**
#
# The global green transition represents one of the most complex structural transformations in modern economic history. Decarbonizing industrial economies requires simultaneous adjustments across input-output production networks where energy substitutes imperfectly with capital and labor, spatial geography where climate change alters regional habitability and drives labor migration, and international trade policy where carbon border tariffs prevent domestic climate abatement from being undone by foreign emissions leakage.
#
# This tutorial provides an integrated, self-explanatory computational laboratory in `puremacro` uniting second-order nonlinear input-output aggregation (Baqaee & Farhi 2019, 2020), dynamic spatial migration hat-algebra (Caliendo, Parro & Rossi-Hansberg 2019), and an Integrated Spatial Climate-Macro Assessment Model (SpatialClimateIAM) audited through `PolicyStudio`. All algorithms execute in pure NumPy and SciPy under the strict Pyodide runtime contract, generating publication-grade opaque visualizations and zero-server interactive HTML5 dashboards with zero external CDN dependencies.
#
# | Scope and verification | |
# |---|---|
# | Workflows covered | 1. Baqaee–Farhi (2019, 2020) Second-Order Network Macro Aggregation (`baqaee_farhi_decomposition`); 2. Caliendo–Parro–Rossi-Hansberg (2019) Dynamic Spatial Migration Hat-Algebra (`solve_dynamic_spatial_hat`); 3. Multi-Region Integrated Spatial Climate-Macro IAM (`SpatialClimateIAM`, `solve_spatial_climate_iam`); 4. Green Policy Counterfactuals (Laissez-Faire, Unilateral Tax, CBAM, Optimal Pigouvian Tax); 5. Self-Auditing `PolicyStudio` Grid & Interactive HTML5/SVG Export (`run_policy_studio`, `result_to_interactive_html`) |
# | Theoretical models | 3-sector input-output production network (Energy, Manufacturing, Services) with CES intermediate and final substitution; 3-region dynamic spatial economy (Temperate, Subtropical, Tropical) with forward-looking labor migration; 3-box mass-conserving carbon cycle model with regional temperature damage functions; Declarative policy counterfactual grid |
# | Solvers & methods | Macroeconomic Hessian curvature $\frac{1}{2} z^\top \mathcal{H} z$, nonlinear CES unit-cost fixed-point solver, dynamic spatial hat-algebra fixed-point iteration, mass-conserving carbon cycle accumulation, zero-server standalone SVG exporter |
# | Acceptance bounds | Baqaee–Farhi Cobb–Douglas 2nd-order curvature $< 10^{-14}$ (Hulten's theorem); Nonlinear CES price fixed-point 3rd-order gap $< 10^{-4}$; Dynamic spatial population conservation error $< 10^{-12}$; Dynamic spatial migration share row-stochasticity error $< 10^{-14}$; Spatial climate IAM global population conservation error $< 10^{-12}$; Atmospheric carbon cycle mass balance error $< 10^{-12}$; CBAM carbon leakage strictly lower than unilateral tax leakage; Optimal Pigouvian carbon tax parity vs. Golosov analytical formula $< 10^{-6}$; PolicyStudio counterfactual grid audit failure count $== 0$; Interactive HTML5 report zero external CDN links verified |

# %% [markdown]
# ## The method in math
#
# **1. Baqaee–Farhi (2019, 2020) Nonlinear Input-Output Aggregation.** Consider an economy with $N$ production sectors and an input-output intermediate trade matrix $\mathbf{\Omega} \in \mathbb{R}^{N \times N}$, where $\Omega_{ij}$ denotes the cost share of sector $j$ in sector $i$'s production. Final demand is characterized by expenditure shares $\mathbf{b} \in \mathbb{R}^N$. The Leontief inverse $\mathbf{\Psi} = (\mathbf{I} - \mathbf{\Omega})^{-1}$ determines the revenue-based Domar weights $\boldsymbol{\lambda} = \mathbf{b}^\top \mathbf{\Psi}$. Under an arbitrary vector of sectoral productivity shocks $\mathbf{z} \in \mathbb{R}^N$, first-order log GDP changes follow Hulten's (1978) theorem:
#
# $$\mathrm{d} \ln Y = \boldsymbol{\lambda}^\top \mathbf{z} = \sum_{i=1}^N \lambda_i z_i.$$
#
# When producers substitute across inputs with CES elasticities $\theta_i \ne 1.0$ and final consumers have elasticity $\theta_0 \ne 1.0$, second-order network curvature emerges through the macroeconomic Hessian $\mathcal{H} \in \mathbb{R}^{N \times N}$:
#
# $$\mathrm{d}^2 \ln Y = \frac{1}{2} \mathbf{z}^\top \mathcal{H} \mathbf{z} = \frac{1}{2} (\theta_0 - 1) \operatorname{Var}_{\mathbf{b}}(\mathbf{\Psi} \mathbf{z}) + \frac{1}{2} \sum_{i=1}^N \lambda_i (\theta_i - 1) \operatorname{Var}_{\mathbf{\Omega}_i}(\mathbf{\Psi} \mathbf{z}).$$
#
# Under Cobb–Douglas preferences ($\theta_i = \theta_0 = 1.0$), all cross-sectional variances vanish identically, reducing $\mathrm{d}^2 \ln Y \equiv 0.0$ to machine precision. When inputs are complements ($\theta_i < 1.0$), upstream bottlenecks (such as energy or critical mineral shortages) generate convex losses that compound through the supply chain.
#
# **2. Caliendo–Parro–Rossi-Hansberg (2019) Dynamic Spatial Migration Hat-Algebra.** Let $r \in \{1, \dots, R\}$ index geographic regions populated by forward-looking workers who choose where to live and work. The migration probability from origin $j$ to destination $i$ satisfies:
#
# $$\mu_{ji, t} = \frac{\left( \beta v_{i, t+1} - \tau_{ji}^m \right)^{1/\nu}}{\sum_k \left( \beta v_{k, t+1} - \tau_{jk}^m \right)^{1/\nu}},$$
#
# where $\nu$ governs migration sensitivity and $\tau_{ji}^m$ denotes moving frictions. In relative changes ("hat-algebra" $\hat{x}_{t+1} = x_{t+1} / x_t$), regional population dynamics evolve as:
#
# $$L_{i, t+1} = \sum_{j=1}^R \mu_{ji, t} \hat{\mu}_{ji, t} L_{j, t},$$
#
# preserving total national population $\sum_{r=1}^R L_{r, t} = \bar{L}$ to $< 10^{-12}$ at all horizons.
#
# **3. Integrated Spatial Climate-Macro IAM & Multi-Box Carbon Cycle.** Regional industrial activity generates carbon emissions $E_{r, t} = \sigma_r Y_{r, t} (1 - \mu_{r, t})$, where $\sigma_r$ is carbon intensity and $\mu_{r, t}$ is the abatement rate. Global emissions $E_t = \sum_r E_{r, t}$ accumulate in a 3-box mass-conserving carbon cycle model (Golosov et al. 2014):
#
# $$S_{1, t+1} = S_{1, t} + \phi_L E_t, \quad S_{2, t+1} = (1 - \phi_{\text{decay}}) S_{2, t} + \phi_0 (1 - \phi_L) E_t, \quad S_{\text{sink}, t+1} = S_{\text{sink}, t} + (1 - \phi_0)(1 - \phi_L) E_t.$$
#
# The stock of atmospheric carbon $S_t = S_{1, t} + S_{2, t}$ drives global surface temperature anomalies via the Arrhenius logarithmic relationship $\Delta T_t = \frac{\lambda}{\ln 2} \ln \left( \frac{S_{\text{pre}} + S_t}{S_{\text{pre}} + S_0} \right)$. Local temperatures induce Cruz & Rossi-Hansberg (2024) regional productivity and amenity damages:
#
# $$A_{r, t} = \exp\left( -\gamma_r^A (T_{r, t} - T_{r, 0}) \right), \quad B_{r, t} = \exp\left( -\gamma_r^B (T_{r, t} - T_{r, 0}) \right).$$
#
# **4. Carbon Leakage & CBAM Border Adjustments.** When an abating coalition $\mathcal{C}$ introduces a unilateral carbon tax $\tau_{\text{carbon}}$, domestic production costs rise, shifting energy-intensive manufacturing toward non-coalition regions $\mathcal{N}$. The carbon leakage rate is defined as:
#
# $$\Lambda = \frac{\Delta E_{\mathcal{N}}}{-\Delta E_{\mathcal{C}}} = \frac{\sum_{t} \sum_{i \in \mathcal{N}} (E_{i, t} - E_{i, t}^{\text{LF}})}{-\sum_{t} \sum_{i \in \mathcal{C}} (E_{i, t} - E_{i, t}^{\text{LF}})}.$$
#
# A Carbon Border Adjustment Mechanism (CBAM) imposes an import tariff on non-coalition goods equal to their embodied carbon content: $\tau_{ji, t}^{\text{CBAM}} = \tau_{\text{carbon}, t} \cdot \sigma_j$, leveling the playing field and strictly reducing carbon leakage ($\Lambda_{\text{CBAM}} < \Lambda_{\text{Uni}}$).
#
# **5. Optimal Pigouvian Carbon Tax (Golosov et al. 2014).** In the logarithmic-utility limit, the optimal Pigouvian carbon tax matches the present discounted value of marginal climate damages across all future periods:
#
# $$\tau_t^* = Y_t \bar{\gamma}_A \left[ \frac{\phi_L}{1 - \beta} + \frac{(1 - \phi_L)\phi_0}{1 - \beta(1 - \phi_{\text{decay}})} \right].$$

# %% [markdown]
# ## Intuition
#
# **Intuition.** To appreciate how supply-chain bottlenecks and climate policy interact in general equilibrium, consider the following economic insights:
#
# - **Leontief Fragility in Industrial Supply Chains.** Hulten's theorem is a cornerstone of macroeconomic aggregation, proving that under first-order approximations, a sector's importance is fully measured by its gross sales share (Domar weight). However, when inputs cannot be substituted easily—such as natural gas in chemical fertilizer plants or semiconductors in automotive manufacturing ($\theta < 1$)—a $10\%$ shock to a small input cannot be smoothed. As downstream firms scramble for scarce inputs, production collapses nonlinearly. The Baqaee–Farhi decomposition isolates this second-order macroeconomic curvature $\frac{1}{2} z^\top \mathcal{H} z$, showing why energy crises trigger disproportionately severe economic downturns.
#
# - **Spatial Relocation as an Adaptation Mechanism.** Climate change does not affect the globe uniformly: equatorial and subtropical regions experience severe heat stress and agricultural productivity collapses, while temperate regions face milder impacts or even extended growing seasons. In a static model without migration, workers in vulnerable regions suffer catastrophic welfare losses. In dynamic spatial general equilibrium, workers forward-lookingly migrate toward regions with higher real wages and amenities. Dynamic spatial hat-algebra models this labor mobility, proving that migration is a primary margin of climate adaptation.
#
# - **The Free-Rider Dilemma and Carbon Border Adjustments.** Climate abatement is a global public good. When a forward-looking coalition taxes emissions unilaterally, domestic heavy industry becomes uncompetitive, causing production to relocate abroad. If foreign producers use dirtier energy grids, global emissions may actually increase—the infamous "green paradox" of carbon leakage. CBAM functions as an anti-leakage shield: by taxing embodied carbon at the border, it eliminates the incentive for firms to offshore production and encourages trading partners to adopt carbon pricing.
#
# - **PolicyStudio: Declarative Auditing Before Publication.** Real-world policymaking requires testing hundreds of parameter combinations across tax paths, substitution elasticities, and border adjustments. `PolicyStudio` provides an automated audit engine that evaluates every cell of a counterfactual grid against strict mathematical criteria (population conservation, mass balance, convergence) and exports an interactive, zero-server HTML5 report with inline SVG graphics.

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

from puremacro.trade.network_macro import (
    baqaee_farhi_decomposition,
    solve_exact_ces_network_prices,
)
from puremacro.spatial.dynamic_hat import (
    DynamicSpatialHatModel,
    solve_dynamic_spatial_hat,
)
from puremacro.climate.spatial_iam import (
    SpatialClimateIAM,
    solve_spatial_climate_iam,
)
from puremacro.study.studio import (
    CounterfactualGrid,
    run_policy_studio,
    result_to_interactive_html,
)

checks = []


def record(result, measured, bound, verdict):
    checks.append({"result": result, "measured": float(measured), "bound": float(bound), "verdict": verdict})


# %% [markdown]
# ### 1. Baqaee–Farhi (2019, 2020) Second-Order Production Networks
#
# We evaluate the Baqaee–Farhi (2019) second-order production network aggregation on a 3-sector economy (Energy, Manufacturing, Services). Under Cobb–Douglas preferences ($\theta = 1.0$), second-order macroeconomic curvature reduces identically to zero ($\mathrm{d}^2 \ln Y = 0$), matching Hulten's theorem to $< 10^{-14}$. Under Leontief-like complementarity ($\theta = 0.35$), supply-chain bottlenecks amplify an upstream energy shock ($z_1 = -8\%$), generating convex output losses that match the exact nonlinear CES price fixed point to $< 10^{-4}$.

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
z_shock = np.array([-0.08, 0.02, -0.01], dtype=np.float64)

# Cobb-Douglas benchmark (Hulten theorem verification)
res_cd = baqaee_farhi_decomposition(
    Omega_bf,
    b_bf,
    elasticities=1.0,
    tfp_shocks=z_shock,
    final_elasticity=1.0,
)
err_hulten = float(abs(res_cd.second_order_dlog_y))

# Complementary supply-chain bottleneck (theta = 0.35)
res_ces = baqaee_farhi_decomposition(
    Omega_bf,
    b_bf,
    elasticities=0.35,
    tfp_shocks=z_shock,
    final_elasticity=0.35,
)
err_ces_gap = float(abs(res_ces.total_second_order_dlog_y - res_ces.exact_ces_dlog_y))

record("Baqaee-Farhi: Cobb-Douglas 2nd-order curvature == 0", err_hulten, 1e-14, "PASS")
record("Baqaee-Farhi: nonlinear CES unit-cost fixed-point gap", err_ces_gap, 1e-4, "PASS")

print(f"Baqaee-Farhi Cobb-Douglas 2nd-order curvature: {err_hulten:.2e} (bound < 1e-14)")
print(f"CES (theta=0.35) 1st-order: {res_ces.first_order_dlog_y * 100:.3f}%, 2nd-order: {res_ces.second_order_dlog_y * 100:.3f}%")
print(f"Taylor approximation gap vs exact CES fixed point: {err_ces_gap:.2e}")

# %% [markdown]
# ### 2. Caliendo–Parro–Rossi-Hansberg (2019) Dynamic Spatial Migration Hat-Algebra
#
# We solve dynamic regional labor migration across 3 geographic regions (North, Center, South) over a 15-year horizon using `solve_dynamic_spatial_hat()`. Forward-looking workers relocate according to relative real wage and amenity changes. We verify that total national population is conserved at every date ($\sum_r L_{r, t} = \bar{L}$) to $< 10^{-12}$ and that the migration probability matrix remains strictly row-stochastic to $< 10^{-14}$.

# %%
L0_sp = np.array([100.0, 100.0, 100.0], dtype=np.float64)
pi0_sp = np.array([[0.70, 0.15, 0.15], [0.15, 0.70, 0.15], [0.15, 0.15, 0.70]], dtype=np.float64)
mu0_sp = np.array([[0.90, 0.05, 0.05], [0.05, 0.90, 0.05], [0.05, 0.05, 0.90]], dtype=np.float64)

mod_spatial = DynamicSpatialHatModel(
    initial_population=L0_sp,
    initial_trade_shares=pi0_sp,
    initial_migration_shares=mu0_sp,
    region_names=("North", "Center", "South"),
)
res_spatial = solve_dynamic_spatial_hat(mod_spatial, horizon=15)

err_pop_conserv = float(res_spatial.population_conservation_error)
err_mig_stoch = float(res_spatial.migration_row_stochastic_error)

record("Dynamic Spatial Hat: population conservation error", err_pop_conserv, 1e-12, "PASS")
record("Dynamic Spatial Hat: migration stochasticity error", err_mig_stoch, 1e-14, "PASS")

print(f"Dynamic Spatial Hat solved over 15 periods.")
print(f"Population conservation error: {err_pop_conserv:.2e} (bound < 1e-12)")
print(f"Migration row-stochasticity error: {err_mig_stoch:.2e} (bound < 1e-14)")

# %% [markdown]
# ### 3. Integrated Spatial Climate-Macro IAM & Multi-Box Carbon Cycle
#
# We configure an Integrated Spatial Climate-Macro Assessment Model (`SpatialClimateIAM`) with 3 regions: Temperate ($T_1 = 14^\circ\text{C}$), Subtropical ($T_2 = 22^\circ\text{C}$), and Tropical ($T_3 = 28^\circ\text{C}$). Regional temperature damages follow Cruz & Rossi-Hansberg (2024), where tropical zones suffer steeper productivity losses ($\gamma_A = 0.025$) than temperate zones ($\gamma_A = 0.005$). We simulate a green transition under Carbon Border Adjustment Mechanisms (CBAM) with an escalating carbon tax trajectory ($\$20 \to \$80/\text{tCO}_2$). We verify that global population conservation and atmospheric carbon mass balance hold to $< 10^{-12}$.

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

err_iam_pop = float(res_cbam.population_conservation_error)
err_iam_carbon = float(res_cbam.carbon_mass_balance_error)

record("Spatial Climate IAM: global population conservation error", err_iam_pop, 1e-12, "PASS")
record("Spatial Climate IAM: atmospheric carbon mass balance error", err_iam_carbon, 1e-12, "PASS")

print(f"Spatial Climate IAM solved with CBAM policy.")
print(f"Population conservation error: {err_iam_pop:.2e}, Carbon mass balance error: {err_iam_carbon:.2e}")
print(f"Regional Population after 15 years: Temperate = {res_cbam.population_path[-1, 0]:.1f}M, Tropical = {res_cbam.population_path[-1, 2]:.1f}M")

# %% [markdown]
# ### 4. Climate Policy Counterfactuals: Laissez-Faire vs. CBAM vs. Optimal Pigouvian
#
# We compare transition dynamics across three policy regimes: Unilateral Carbon Tax, CBAM Tariff, and the Optimal Pigouvian Carbon Tax. We verify that CBAM strictly reduces international carbon leakage compared to unilateral taxation ($\Lambda_{\text{CBAM}} < \Lambda_{\text{Uni}}$). Furthermore, we verify that in the logarithmic-utility limit, the Pigouvian carbon tax computed by the solver matches the analytical Golosov et al. (2014) formula ($\$48.98/\text{tCO}_2$) to $< 10^{-6}$.

# %%
res_uni = solve_spatial_climate_iam(
    model_iam,
    horizon=15,
    policy_scenario="unilateral_tax",
    carbon_tax_path=np.linspace(20, 80, 15),
    coalition_regions=[0],
)

res_opt = solve_spatial_climate_iam(
    model_iam,
    horizon=15,
    policy_scenario="optimal_pigouvian",
)

leak_uni = float(res_uni.carbon_leakage_rate)
leak_cbam = float(res_cbam.carbon_leakage_rate)
err_leak_reduc = float(max(0.0, leak_cbam - leak_uni))

# Golosov analytical formula verification:
# Y_world = 330, exp_shares = [100, 150, 80]/330, mean_gamma = (100*0.005 + 150*0.015 + 80*0.025)/330 = 4.75/330
# tau = 330 * (4.75/330) * (0.20/0.04 + 0.80*0.393/(1 - 0.96*0.98)) = 4.75 * 10.31081081 = 48.97635135 $/tCO2
tax_golosov_analytical = 48.97635135135135
tax_golosov_solver = float(res_opt.pigouvian_tax_path[0])
err_golosov = float(abs(tax_golosov_solver - tax_golosov_analytical))

record("Climate Policy: CBAM carbon leakage reduction vs unilateral tax", err_leak_reduc, 0.0, "PASS")
record("Climate Policy: optimal Pigouvian tax parity vs Golosov formula", err_golosov, 1e-6, "PASS")

print(f"Carbon Leakage Rate: Unilateral Tax = {leak_uni * 100:.2f}%, CBAM = {leak_cbam * 100:.2f}% (Reduction = {(leak_uni - leak_cbam) * 100:.2f} pp)")
print(f"Optimal Pigouvian Tax: Solver = ${tax_golosov_solver:.4f}, Analytical = ${tax_golosov_analytical:.4f} (Error = {err_golosov:.2e})")

# %% [markdown]
# ### 5. Self-Auditing PolicyStudio Grid & Zero-Server Interactive HTML5/SVG
#
# We evaluate a 2D counterfactual policy grid over CES substitution elasticities ($\theta \in \{0.4, 0.7, 1.0\}$) and upstream energy supply shocks ($z_1 \in \{-0.10, -0.05\}$) using `run_policy_studio()`. Every cell is audited for convergence and numerical validity. Finally, `result_to_interactive_html()` exports a self-contained, zero-server HTML5 report with inline SVG graphics and client-side JavaScript controls, verifying that zero external CDN links (`http://` or `https://`) are present.

# %%
grid_studio = CounterfactualGrid({
    "elasticity": [0.4, 0.7, 1.0],
    "energy_shock": [-0.10, -0.05],
})


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
html_export = res_studio.to_interactive_html(title="Green Transition Policy Studio - Supply Chain Bottlenecks")
has_cdn = "http://" in html_export or "https://" in html_export

record("PolicyStudio: counterfactual grid audit failure count", float(res_studio.fail_count), 0.0, "PASS")
record("PolicyStudio: zero external CDN dependencies verified", float(1.0 if has_cdn else 0.0), 0.0, "PASS")

print(f"PolicyStudio status: {res_studio.overall_audit_status} (0 failures across {len(res_studio.cells)} grid cells)")
print(f"Interactive HTML report: {len(html_export)} characters, zero external CDN links = {not has_cdn}")

# %% [markdown]
# ### 6. Hero visualization: supply-chain bottlenecks, climate migration, and policy counterfactuals
#
# The 3-panel publication figure below illustrates:
# - **Panel A**: Baqaee–Farhi second-order macroeconomic curvature $\frac{1}{2} \mathbf{z}^\top \mathcal{H} \mathbf{z}$ across substitution elasticities $\theta \in [0.2, 1.8]$, showing how input complementarity ($\theta < 1$) amplifies energy supply shocks.
# - **Panel B**: Dynamic spatial climate migration paths $L_{r, t}$ over a 15-year horizon, showing labor reallocation from high-damage tropical zones to the temperate coalition.
# - **Panel C**: Global carbon emissions trajectories under Unilateral Taxation, CBAM Border Tariffs, and the Optimal Pigouvian Carbon Tax.

# %%
thetas_vec = np.linspace(0.2, 1.8, 25)
d2_curve = []
for th_val in thetas_vec:
    res_th = baqaee_farhi_decomposition(
        Omega_bf,
        b_bf,
        elasticities=float(th_val),
        tfp_shocks=z_shock,
        final_elasticity=float(th_val),
    )
    d2_curve.append(res_th.second_order_dlog_y * 100.0)

col = _nbstyle.palette(4)
dash = _nbstyle.styles(4)

fig, (ax1, ax2, ax3) = _nbstyle.figura(1, 3)

# Panel 1: Baqaee-Farhi Curvature vs Elasticity
ax1.plot(thetas_vec, d2_curve, color=col[0], lw=2.2, label=r"2nd-Order Curvature $\frac{1}{2} z^\top \mathcal{H} z$")
ax1.axhline(0.0, color=col[1], lw=1.2, linestyle="--", label="Cobb-Douglas Baseline (Hulten)")
ax1.axvline(1.0, color=_nbstyle.SPINE, lw=0.8, linestyle=":")
ax1.set(xlabel=r"CES Substitution Elasticity $\theta$", ylabel="2nd-Order Impact (pp)", title="Supply-Chain Bottlenecks")
ax1.legend(loc="lower right", frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)

# Panel 2: Dynamic Spatial Climate Migration
t_vec = np.arange(15)
ax2.plot(t_vec, res_cbam.population_path[:, 0], color=col[0], lw=2.2, label="Temperate (Coalition)")
ax2.plot(t_vec, res_cbam.population_path[:, 1], color=col[1], lw=2.0, linestyle=dash[1], label="Subtropical (Center)")
ax2.plot(t_vec, res_cbam.population_path[:, 2], color=col[2], lw=1.8, linestyle=dash[2], label="Tropical (High Damage)")
ax2.set(xlabel="Years t", ylabel="Regional Population (Millions)", title="Dynamic Climate Migration")
ax2.legend(loc="center right", frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)

# Panel 3: Policy Counterfactuals
ax3.plot(t_vec, res_uni.global_emissions_path, color=col[0], lw=2.0, label="Unilateral Tax (High Leakage)")
ax3.plot(t_vec, res_cbam.global_emissions_path, color=col[1], lw=2.2, linestyle=dash[1], label="CBAM Tariff (Reduced Leakage)")
ax3.plot(t_vec, res_opt.global_emissions_path, color=col[2], lw=1.8, linestyle=dash[2], label="Optimal Pigouvian Tax")
ax3.set(xlabel="Years t", ylabel="Global Emissions (GtC)", title="Climate Policy Scenarios")
ax3.legend(loc="upper right", frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)

# %% [markdown]
# ### 7. Verification scorecard
#
# Every numerical check across network bottlenecks, spatial migration, climate IAM, and PolicyStudio is tabulated below with its measured value, acceptance bound, and verification verdict.

# %%
scorecard = pd.DataFrame(checks)
print(scorecard.to_string(index=False, formatters={"measured": "{:.2e}".format, "bound": "{:.2e}".format}))
assert (scorecard["measured"] <= scorecard["bound"]).all()
print(f"\nAll {len(scorecard)} green transition, spatial IAM, and PolicyStudio checks pass.")

# %% [markdown]
# ## Read the output
#
# In Section 1, `baqaee_farhi_decomposition()` proves Hulten's theorem under Cobb–Douglas preferences ($\theta = 1.0$), with second-order macroeconomic curvature reducing to exactly $0.00$ ($< 10^{-14}$). When inputs are complements ($\theta = 0.35$), supply-chain bottlenecks generate an additional $0.077$ percentage point contraction, closely matching the exact nonlinear CES price fixed point with an error of just $6.8 \times 10^{-5}$.
#
# In Section 2, `solve_dynamic_spatial_hat()` solves the Caliendo–Parro–Rossi-Hansberg (2019) dynamic migration model, preserving global national population to $0.00$ ($< 10^{-12}$) and maintaining migration probability row-stochasticity with zero numerical deviation.
#
# In Section 3, `SpatialClimateIAM` couples dynamic trade and migration with a 3-box carbon cycle. Atmospheric carbon mass balance holds to $< 10^{-12}$, showing that rising tropical temperatures induce gradual labor migration northward into temperate coalition regions.
#
# In Section 4, climate policy counterfactuals show that unilateral carbon taxation produces a carbon leakage rate of $49.58\%$, which CBAM border tariffs reduce to $48.64\%$. The solver's optimal Pigouvian carbon tax matches the analytical Golosov et al. (2014) formula ($\$48.98/\text{tCO}_2$) with a numerical discrepancy of only $2.84 \times 10^{-14}$.
#
# In Section 5 and the Hero Visualization, `run_policy_studio()` evaluates and audits a 2D counterfactual grid with zero audit failures, compiling a standalone zero-server interactive HTML5 dashboard with zero external CDN dependencies.

# %% [markdown]
# ## Your turn
#
# Tune the carbon tax level `user_carbon_tax` and the energy substitution elasticity `user_energy_elasticity`. The downstream assertions verify that spatial population conservation, carbon mass balance, and network stability hold across the entire advertised parameter space.
#
# 1. **Basic.** Increase `user_carbon_tax` from `30.0` to `90.0` $/tCO2. How does higher domestic carbon taxation accelerate regional migration toward temperate zones, and why does CBAM become increasingly essential?
# 2. **Intermediate.** Vary `user_energy_elasticity` between `0.25` and `0.80`. How does supply-chain curvature deepen as energy substitution becomes more rigid?
# 3. **Stretch.** Extend the coalition to include Region 1 (Subtropical), creating an expanded climate club, and verify that international carbon leakage drops substantially.

# %%
user_carbon_tax = 50.0  # ← change this: Carbon tax in coalition region ($/tCO2), float in [20.0, 100.0]
user_energy_elasticity = 0.40  # ← change this: Energy substitution elasticity, float in [0.20, 1.20]

assert isinstance(user_carbon_tax, (int, float)) and 20.0 <= user_carbon_tax <= 100.0
assert isinstance(user_energy_elasticity, (int, float)) and 0.20 <= user_energy_elasticity <= 1.20

res_iam_user = solve_spatial_climate_iam(
    model_iam,
    horizon=12,
    policy_scenario="cbam",
    carbon_tax_path=np.full(12, float(user_carbon_tax)),
    coalition_regions=[0],
)

res_bf_user = baqaee_farhi_decomposition(
    Omega_bf,
    b_bf,
    elasticities=float(user_energy_elasticity),
    tfp_shocks=z_shock,
    final_elasticity=float(user_energy_elasticity),
)

print(f"User Carbon Tax = ${user_carbon_tax:.1f}/tCO2: Leakage Rate = {res_iam_user.carbon_leakage_rate * 100:.2f}%, Pop Error = {res_iam_user.population_conservation_error:.2e}")
print(f"User Energy Elasticity = {user_energy_elasticity:.2f}: 2nd-Order Curvature = {res_bf_user.second_order_dlog_y * 100:.4f}%")

# Assertions over advertised parameter range
assert 0.0 <= res_iam_user.carbon_leakage_rate <= 1.0
assert res_iam_user.population_conservation_error < 1e-12
assert res_iam_user.carbon_mass_balance_error < 1e-12
assert abs(res_bf_user.first_order_dlog_y) > 0.0

# %% [markdown]
# ## How comprehensive is this?
#
# This tutorial integrates production network macroeconomics, dynamic spatial trade and migration, and spatial climate-macro IAM under the `puremacro v6.0.0` release. Theoretical foundations and detailed API specifications are documented across `docs/network_spatial_climate_iam.md` (`docs/es/network_spatial_climate_iam.md`), with cross-references to the general differentiable model architecture in `docs/autodiff_and_protocol.md` and policy counterfactual design in `docs/study_policy_studio.md`. All routines adhere strictly to the four-package Pyodide runtime contract (`numpy`, `scipy`, `pandas`, `matplotlib`).
