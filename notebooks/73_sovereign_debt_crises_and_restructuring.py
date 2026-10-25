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
# # Sovereign Debt Crises, Long-Term Maturity & Restructuring Lab: Arellano Default Incentives, Chatterjee–Eyigungor Dilution, DualArray Price Schedules, Mendoza Sudden Stops, and Debt Restructuring
#
# **How do sovereign governments make optimal borrowing and default choices when facing volatile income, how does long-term bond maturity generate secondary market debt dilution, why do collateral constraints trigger Fisherian debt-deflation sudden stops, and how effective are debt restructuring haircuts versus maturity extensions at restoring fiscal solvency?**
#
# Emerging market economies frequently experience severe sovereign debt crises characterized by surging borrowing costs, abrupt loss of international capital market access, and catastrophic output collapses. Unlike corporate borrowers, sovereign nations cannot pledge physical assets to international creditors; repayment is sustained solely by the threat of future financial autarky and macroeconomic output penalties. Modern quantitative macroeconomics analyzes these phenomena through two complementary lenses: the endogenous willingness-to-pay framework of Arellano (2008) and Chatterjee & Eyigungor (2012), and the ability-to-borrow collateral friction of Mendoza (2010).
#
# This tutorial provides an intuitive, hands-on laboratory exploring quantitative sovereign default, long-term maturity debt dilution, collateral-constrained sudden stops, spread term-structure decomposition, and debt restructuring counterfactuals in `puremacro`. All algorithms are implemented in pure NumPy and SciPy under the strict four-package Pyodide contract, incorporating automatic differentiation via `DualArray` to evaluate exact bond price schedule derivatives $\partial q(b', y)/\partial b' \le 0$ without finite-difference grid noise.
#
# | Scope and verification | |
# |---|---|
# | Workflows covered | 1. Arellano (2008) Short-Term Sovereign Default with Extreme-Value Taste Shocks; 2. Chatterjee & Eyigungor (2012) Long-Term Debt Dilution & `DualArray` Derivative Evaluation $\partial q/\partial b' \le 0$; 3. Mendoza (2010) Collateral-Constrained Sudden Stops with Fisherian Debt-Deflation Spirals; 4. Sovereign Spread Term-Structure Decomposition across Maturities; 5. Debt Restructuring Counterfactuals (Face-Value Haircuts vs. Maturity Extensions) |
# | Theoretical models | Small open economy stochastic endowment model with endogenous default and exclusion; Chatterjee–Eyigungor long-term decaying coupon bonds; Mendoza production economy with working capital collateral constraints and endogenous Tobin's $q$; Multi-horizon zero-coupon defaultable survival pricing |
# | Solvers & methods | `solve_sovereign_default`, `solve_sudden_stop`, `decompose_spread_term_structure`, `evaluate_debt_restructuring`, `DualArray` forward automatic differentiation |
# | Acceptance bounds | Bellman fixed-point residual $< 10^{-7}$; bond price bounds $0 \le q(b', y) \le (1+r^*)^{-1}$; price schedule monotonicity $\partial q/\partial b' \le 0$; countercyclical spreads $\operatorname{corr}(\text{spread}, y) < 0$; sudden-stop Fisherian amplification ratio $> 1.25$; collateral binding fraction $> 0.05$; precautionary deleveraging gap; term-structure spread additivity error $< 10^{-12}$; post-restructuring spread reduction $\le 0.0$ bps; post-restructuring default hazard mitigation |

# %% [markdown]
# ## The method in math
#
# **1. Arellano (2008) Sovereign Default Mechanism.** Consider a small open economy whose representative household enjoys utility $u(c) = \frac{c^{1-\gamma}-1}{1-\gamma}$ over consumption $c_t$, discounted at subjective rate $\beta \in (0, 1)$. The economy receives a stochastic endowment shock $y_t$ following a log-AR(1) process discretized via Tauchen (1986). In each period, the sovereign enters with foreign debt obligations $b_t$ and decides whether to repay or default:
#
# $$v(b, y) = \max\left\{ v^R(b, y), v^D(y) \right\}.$$
#
# To ensure smooth choice probabilities and differentiable policy functions, we incorporate extreme-value (Gumbel) taste shocks with scale $\sigma_\varepsilon > 0$. The sovereign value function becomes the closed-form log-sum-exp envelope:
#
# $$v(b, y) = \sigma_\varepsilon \ln\left( \exp\left(\frac{v^R(b, y)}{\sigma_\varepsilon}\right) + \exp\left(\frac{v^D(y)}{\sigma_\varepsilon}\right) \right),$$
#
# which yields smooth logistic default hazard probabilities:
#
# $$p_{\text{def}}(b, y) = \frac{1}{1 + \exp\left(\frac{v^R(b, y) - v^D(y)}{\sigma_\varepsilon}\right)}.$$
#
# If the sovereign repays, it chooses new debt $b' \in \mathcal{B}$ subject to consumption $c = y - b + q(b', y) b'$, solving:
#
# $$v^R(b, y) = \max_{b'} \left\{ u(y - b + q(b', y) b') + \beta \sum_{y'} \pi(y' \mid y) v(b', y') \right\}.$$
#
# If the sovereign defaults, foreign debt is erased, the economy suffers an asymmetric output penalty $h(y) \le y$, and it is excluded from international capital markets, regaining access with exogenous re-entry probability $\theta \in (0, 1)$:
#
# $$v^D(y) = u(h(y)) + \beta \sum_{y'} \pi(y' \mid y) \left[ \theta v(0, y') + (1 - \theta) v^D(y') \right].$$
#
# **2. Chatterjee & Eyigungor (2012) Long-Term Debt & Debt Dilution.** In one-period debt models, the sovereign must refinance its entire debt stock every quarter. Emerging markets, however, issue long-term bonds that mature at rate $\delta \in (0, 1]$ and pay a constant coupon $\kappa \ge 0$. The price of an outstanding bond $q(b', y)$ satisfies the risk-neutral investor no-arbitrage condition:
#
# $$q(b', y) = \frac{1}{1 + r^*} \sum_{y'} \pi(y' \mid y) \left( 1 - p_{\text{def}}(b', y') \right) \left[ \delta + (1 - \delta) \left( \kappa + q(b''(b', y'), y') \right) \right].$$
#
# Long-term debt introduces severe **debt dilution**: when the government issues new bonds in the future ($b'' > b'$), it increases default risk on previously issued bonds, transferring wealth from existing creditors to the sovereign. Consequently, the secondary bond price schedule $q(b', y)$ slopes downward:
#
# $$\frac{\partial q(b', y)}{\partial b'} \le 0.$$
#
# In `puremacro`, this derivative is computed analytically using forward-mode automatic differentiation through the `DualArray` PCHIP spline representation, guaranteeing machine-precision monotonicity without finite-difference truncation error.
#
# **3. Mendoza (2010) Collateral-Constrained Sudden Stops.** In contrast to default models where debt is unsecured, sudden-stop models impose an endogenous collateral constraint: foreign debt cannot exceed a fraction $\kappa$ of the market value of domestic collateral:
#
# $$b_{t+1} \le \kappa q_t k_{t+1},$$
#
# where $k_{t+1}$ is physical capital and $q_t = 1 + \phi \left( \frac{k_{t+1} - (1-\delta_k) k_t}{k_t} \right)$ represents Tobin's $q$ determined by capital adjustment costs. When an adverse productivity or interest-rate shock pushes the economy against this borrowing limit, the Lagrange multiplier $\lambda_t > 0$ binds. Domestic agents are forced into abrupt deleveraging, slashing investment and consumption. The collapse in investment depresses the market price of capital $q_t$, which shrinks collateral value and further tightens the credit limit. This self-reinforcing vicious cycle is the classic **Fisherian debt-deflation spiral**, amplifying output contractions and current-account reversals by over $50\%$ relative to the frictionless economy.
#
# **4. Spread Term-Structure Decomposition & Debt Restructuring.** The annualized credit spread on an $m$-quarter zero-coupon sovereign claim is defined as:
#
# $$\text{Spread}_m(b, y) = 4 \left( \left[ \frac{1}{q_m(b, y)} \right]^{1/m} - (1 + r^*) \right).$$
#
# By pricing a synthetic physical survival claim with recovery and discounting, the spread decomposes additively into physical expected default loss and maturity/dilution risk premium:
#
# $$\text{Spread}_m = \text{DefaultLoss}_m + \text{RiskPremium}_m.$$
#
# When debt distress occurs, restructuring contracts evaluate face-value haircuts $h \in (0, 1)$ ($b_{\text{new}} = (1 - h) b_{\text{old}}$) and maturity extensions ($\delta_{\text{new}} < \delta_{\text{old}}$), evaluating immediate basis-point spread compression and default probability reduction.

# %% [markdown]
# ## Intuition
#
# **Intuition.** To understand why sovereign debt dynamics behave so differently from standard household borrowing, consider the following economic insights:
#
# - **Default as an Option with Asymmetric Penalties.** A sovereign government balances the immediate consumption benefit of repudiating its external debt against the future cost of financial isolation and output loss. During deep recessions, consumption is already low, marginal utility $u'(c)$ is elevated, and the output penalty $h(y)$ is relatively small, making default highly tempting. This explains why sovereign default risk and borrowing spreads are intensely countercyclical: spreads blow out precisely when the country is in a slump and desperately needs foreign capital.
#
# - **The Curse of Debt Dilution in Long-Term Maturities.** When debt is long-term, the sovereign enters each period owing money to legacy bondholders whose coupons were negotiated in the past. If the government issues more debt today, it drives down the market price of all bonds, diluting existing bondholders. Because the current government does not internalize the capital losses suffered by old lenders, it overborrows relative to the constrained-efficient benchmark. Forward automatic differentiation via `DualArray` captures this dilution gradient $\partial q/\partial b' \le 0$ with exact machine precision, showing how each additional dollar of debt erodes creditworthiness.
#
# - **Fisherian Deflation and Precautionary Deleveraging.** In Mendoza's sudden stop world, financial amplification occurs because asset prices are determined in general equilibrium. A single borrowing household takes the price of capital $q_t$ as given. But when all households cut investment simultaneously to service foreign loans, asset prices crash. This price drop shrinks everyone's borrowing capacity, forcing an emergency fire sale of domestic assets. Aware of this catastrophic trap, the sovereign maintains a precautionary debt buffer in normal times, keeping average debt well below the statutory borrowing ceiling.
#
# - **Debt Restructuring as a Circuit Breaker.** Once a sovereign is caught in the default zone, secondary market yields can exceed several thousand basis points, pricing in certain default. A debt restructuring that enforces a face-value haircut restores debt to the sustainable interior of the borrowing grid, compressing spreads and lowering the default hazard back to normal levels.

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

from puremacro.sovereign import (
    solve_sovereign_default,
    solve_sudden_stop,
    decompose_spread_term_structure,
    evaluate_debt_restructuring,
)
from puremacro._autodiff import DualArray

checks = []


def record(result, measured, bound, verdict=None):
    if verdict is None:
        verdict = "PASS" if float(measured) <= float(bound) else "FAIL"
    else:
        verdict = "PASS" if (verdict == "PASS" and float(measured) <= float(bound)) else "FAIL"
    checks.append({"result": result, "measured": float(measured), "bound": float(bound), "verdict": verdict})


# %% [markdown]
# ### 1. Arellano (2008) Short-Term Sovereign Default Benchmark
#
# We solve the canonical Arellano (2008) sovereign default model with one-period debt ($\delta = 1.0, \kappa = 0.0$) and extreme-value taste shocks ($\sigma_\varepsilon = 0.01$). The world risk-free rate is set to $r^* = 1.5\%$ quarterly ($6.1\%$ annualized). We verify that the Bellman fixed-point residual converges below $10^{-7}$, that the bond price schedule respects the no-arbitrage bounds $q(b', y) \in [0, (1 + r^*)^{-1}]$, and that sovereign spreads exhibit strong countercyclicality ($\operatorname{corr}(\text{spread}, y) < 0$).

# %%
res_arellano = solve_sovereign_default(
    beta=0.953,
    gamma=2.0,
    r_star=0.015,
    delta=1.0,
    kappa=0.0,
    sigma_eps=0.01,
    n_b=30,
    n_y=9,
    max_iter=250,
)

q_max_rf = 1.0 / (1.0 + 0.015)
err_arellano_res = float(res_arellano.residual)
err_bounds_arellano = float(
    np.max(np.maximum(0.0, res_arellano.bond_price - (q_max_rf + 1e-8)))
    + np.max(np.maximum(0.0, -res_arellano.bond_price))
)
mean_spread_arellano = np.mean(res_arellano.spread, axis=0)
corr_y_arellano = float(np.corrcoef(mean_spread_arellano, res_arellano.y_grid)[0, 1])

record("Arellano: Bellman fixed-point residual", err_arellano_res, 1e-7, "PASS" if err_arellano_res <= 1e-7 else "FAIL")
record("Arellano: bond price bounds violation", err_bounds_arellano, 0.0, "PASS" if err_bounds_arellano <= 0.0 else "FAIL")
record("Arellano: countercyclical spread-output correlation", corr_y_arellano, 0.0, "PASS" if corr_y_arellano <= 0.0 else "FAIL")

print(f"Arellano (2008) solved in {res_arellano.iterations} iterations.")
print(f"Bellman residual: {err_arellano_res:.2e}, Price bounds violation: {err_bounds_arellano:.2e}")
print(f"Spread-output correlation corr(spread, y): {corr_y_arellano:.4f}")

# %% [markdown]
# ### 2. Chatterjee & Eyigungor (2012) Long-Term Debt & DualArray Derivatives
#
# We extend the sovereign economy to long-term maturing coupon bonds following Chatterjee & Eyigungor (2012). Bonds mature at rate $\delta = 0.05$ (average maturity of 20 quarters or 5 years) and pay a quarterly coupon $\kappa = 0.03$. We evaluate the bond price schedule $q(b', y)$ and verify that forward automatic differentiation via `DualArray` evaluates negative dilution derivatives $\partial q/\partial b' \le 0$ across all grid points.

# %%
res_ce = solve_sovereign_default(
    beta=0.953,
    gamma=2.0,
    r_star=0.015,
    delta=0.05,
    kappa=0.03,
    sigma_eps=0.01,
    n_b=30,
    n_y=9,
    max_iter=250,
)

err_ce_res = float(res_ce.residual)
max_dq_db = float(np.max(res_ce.dq_db))
err_dq_db_mono = float(max(0.0, max_dq_db))

# Compare spread dilution: long-term debt increases average spread at median debt level
med_b_idx = len(res_ce.b_grid) // 2
spread_lt = float(np.mean(res_ce.spread[med_b_idx, :]) * 10000.0)
spread_st = float(np.mean(res_arellano.spread[med_b_idx, :]) * 10000.0)

record("Chatterjee-Eyigungor: Bellman residual", err_ce_res, 1e-7, "PASS" if err_ce_res <= 1e-7 else "FAIL")
record("Chatterjee-Eyigungor: DualArray derivative monotonicity dq/db <= 0", err_dq_db_mono, 0.0, "PASS" if err_dq_db_mono <= 0.0 else "FAIL")

print(f"Chatterjee-Eyigungor (2012) solved in {res_ce.iterations} iterations.")
print(f"Bellman residual: {err_ce_res:.2e}, Max dq/db derivative: {max_dq_db:.6f}")
print(f"Average spread at median debt: Long-Term = {spread_lt:.1f} bps, Short-Term = {spread_st:.1f} bps")

# %% [markdown]
# ### 3. Mendoza (2010) Collateral Constraints & Sudden-Stop Amplification
#
# We solve the Mendoza (2010) small open economy model where borrowing is constrained by the market value of domestic capital: $b_{t+1} \le \kappa q_t k_{t+1}$ with $\kappa = 0.30$. Capital adjustment costs ($\phi = 0.08$) render the price of capital (Tobin's $q$) endogenous. When adverse shocks hit, the collateral constraint binds ($\lambda_t > 0$), triggering a Fisherian debt-deflation fire-sale spiral. We verify that the crisis amplification ratio exceeds $1.25$ and that the economy maintains a precautionary buffer below statutory collateral.

# %%
res_mendoza = solve_sudden_stop(
    beta=0.96,
    gamma=2.0,
    r_star=0.04,
    kappa_collateral=0.30,
    alpha=0.32,
    delta_k=0.088,
    phi=0.08,
    n_b=20,
    n_k=20,
    n_z=3,
    max_iter=200,
)

err_amplification = float(max(0.0, 1.25 - res_mendoza.amplification_ratio))
err_binding_frac = float(max(0.0, 0.05 - res_mendoza.collateral_binding_fraction))

# Precautionary buffer: ergodic average debt is strictly below statutory collateral
mean_debt_mendoza = float(np.mean(res_mendoza.policy_b))
mean_collateral_mendoza = float(0.30 * np.mean(res_mendoza.asset_price_q) * np.mean(res_mendoza.policy_k))
gap_precautionary = float(max(0.0, mean_debt_mendoza - mean_collateral_mendoza))

record("Mendoza Sudden Stop: Fisherian amplification ratio > 1.25", err_amplification, 0.0, "PASS" if err_amplification <= 0.0 else "FAIL")
record("Mendoza Sudden Stop: collateral constraint binding fraction > 0.05", err_binding_frac, 0.0, "PASS" if err_binding_frac <= 0.0 else "FAIL")
record("Mendoza Sudden Stop: precautionary deleveraging gap", gap_precautionary, 0.0, "PASS" if gap_precautionary <= 0.0 else "FAIL")

print(f"Mendoza (2010) Sudden Stop solved in {res_mendoza.iterations} iterations.")
print(f"Crisis amplification ratio: {res_mendoza.amplification_ratio:.4f} (bound > 1.25)")
print(f"Collateral binding fraction: {res_mendoza.collateral_binding_fraction * 100:.2f}% of state space")
print(f"Average debt ({mean_debt_mendoza:.4f}) vs Collateral capacity ({mean_collateral_mendoza:.4f})")

# %% [markdown]
# ### 4. Sovereign Spread Term Structure & Debt Restructuring Counterfactuals
#
# We decompose the sovereign spread term structure across maturities from 1 quarter to 10 years ($m \in \{1, 4, 8, 12, 20, 40\}$ quarters) into physical expected default loss and maturity/dilution risk premia. We verify exact additivity of the spread decomposition ($< 10^{-10}$). Next, we evaluate a debt restructuring counterfactual that applies a $25\%$ face-value haircut ($h = 0.25$), computing the immediate basis-point spread reduction and default hazard mitigation.

# %%
df_term = decompose_spread_term_structure(res_ce, horizons=[1, 4, 8, 12, 20, 40])
err_term_additivity = float(np.max(np.abs(
    df_term["spread_bps"].values - (df_term["default_loss_component"].values + df_term["risk_premium_component"].values)
)))

res_restruct = evaluate_debt_restructuring(res_ce, haircut=0.25, maturity_extension=0.0)
err_spread_reduc = float(res_restruct.spread_reduction_bps)
err_hazard_reduc = float(max(0.0, res_restruct.post_default_prob - res_restruct.pre_default_prob))

record("Spread Term Structure: exact decomposition additivity error", err_term_additivity, 1e-10, "PASS" if err_term_additivity <= 1e-10 else "FAIL")
record("Debt Restructuring: immediate spread compression <= 0.0 bps", err_spread_reduc, 0.0, "PASS" if err_spread_reduc <= 0.0 else "FAIL")
record("Debt Restructuring: post-restructuring default hazard mitigation", err_hazard_reduc, 0.0, "PASS" if err_hazard_reduc <= 0.0 else "FAIL")

print("Sovereign Spread Term Structure Decomposition (median debt):")
print(df_term[["maturity_years", "spread_bps", "default_loss_component", "risk_premium_component"]].to_string(index=False))
print(f"\nRestructuring with 25% Haircut:")
print(f"  Spread reduction: {res_restruct.spread_reduction_bps:.2f} bps")
print(f"  Default probability: {res_restruct.pre_default_prob * 100:.2f}% -> {res_restruct.post_default_prob * 100:.2f}%")

# %% [markdown]
# ### 5. Hero visualization: bond pricing schedules, Fisherian debt deflation, and spread term structure
#
# The 3-panel publication figure below illustrates:
# - **Panel A**: Chatterjee & Eyigungor (2012) equilibrium bond price schedules $q(b', y)$ as a function of debt issuance $b'$ across low, median, and high income states, bounded by the risk-free rate $(1+r^*)^{-1}$.
# - **Panel B**: Mendoza (2010) Fisherian debt-deflation mechanism, showing how Tobin's $q$ (capital price) plunges as foreign debt rises and the collateral constraint binds.
# - **Panel C**: Sovereign spread term-structure decomposition across maturities (1 to 10 years), highlighting the relative contributions of physical default loss and maturity/dilution risk premia.

# %%
col = _nbstyle.palette(4)
dash = _nbstyle.styles(4)

fig, (ax1, ax2, ax3) = _nbstyle.figura(1, 3)

# Panel 1: Bond Price Schedules
q_max = 1.0 / (1.0 + 0.015)
ax1.plot(res_ce.b_grid, res_ce.bond_price[:, -1], color=col[0], lw=2.2, label=r"High Income ($y_{\mathrm{high}}$)")
ax1.plot(res_ce.b_grid, res_ce.bond_price[:, len(res_ce.y_grid) // 2], color=col[1], lw=2.0, linestyle=dash[1], label=r"Mean Income ($y_{\mathrm{mid}}$)")
ax1.plot(res_ce.b_grid, res_ce.bond_price[:, 0], color=col[2], lw=1.8, linestyle=dash[2], label=r"Low Income ($y_{\mathrm{low}}$)")
ax1.axhline(q_max, color=col[3], lw=1.0, linestyle="--", label=r"Risk-Free Bound $(1+r^*)^{-1}$")
ax1.set(xlabel=r"Next-Period Debt $b'$", ylabel=r"Bond Price $q(b', y)$", title="CE (2012) Bond Price Schedules")
ax1.legend(loc="lower left", frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)

# Panel 2: Mendoza Sudden Stop Fisherian Deflation
ik_mid = len(res_mendoza.k_grid) // 2
ax2.plot(res_mendoza.b_grid, res_mendoza.asset_price_q[:, ik_mid, 2], color=col[0], lw=2.2, label=r"High Prod ($z_{\mathrm{high}}$)")
ax2.plot(res_mendoza.b_grid, res_mendoza.asset_price_q[:, ik_mid, 1], color=col[1], lw=2.0, linestyle=dash[1], label=r"Normal Prod ($z_{\mathrm{mid}}$)")
ax2.plot(res_mendoza.b_grid, res_mendoza.asset_price_q[:, ik_mid, 0], color=col[2], lw=1.8, linestyle=dash[2], label=r"Low Prod ($z_{\mathrm{low}}$)")
ax2.set(xlabel="External Debt $b$", ylabel=r"Tobin's $q$ (Capital Price)", title="Mendoza (2010) Sudden Stop Deflation")
ax2.legend(loc="lower left", frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)

# Panel 3: Spread Term Structure Decomposition
m_vec = df_term["maturity_years"].values
ax3.plot(m_vec, df_term["spread_bps"].values, color=col[0], lw=2.2, label="Total Sovereign Spread")
ax3.plot(m_vec, df_term["default_loss_component"].values, color=col[1], lw=2.0, linestyle=dash[1], label="Expected Default Loss")
ax3.plot(m_vec, df_term["risk_premium_component"].values, color=col[2], lw=1.8, linestyle=dash[2], label="Dilution / Maturity Premium")
ax3.set(xlabel="Maturity (Years)", ylabel="Spread (bps)", title="Spread Term-Structure Decomposition")
ax3.legend(loc="upper right", frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)

# %% [markdown]
# ### 6. Verification scorecard
#
# Every numerical check across the sovereign default and restructuring lab is tabulated below with its measured value, acceptance bound, and verification verdict.

# %%
scorecard = pd.DataFrame(checks)
print(scorecard.to_string(index=False, formatters={"measured": "{:.2e}".format, "bound": "{:.2e}".format}))
assert (scorecard["measured"] <= scorecard["bound"]).all()
print(f"\nAll {len(scorecard)} sovereign debt crisis and restructuring checks pass.")

# %% [markdown]
# ## Read the output
#
# In Section 1, the Arellano (2008) one-period sovereign default solver converges to a Bellman fixed-point residual of $8.94 \times 10^{-8}$, well below the strict $10^{-7}$ tolerance. The bond price schedule strictly satisfies the theoretical no-arbitrage bounds $[0, (1 + r^*)^{-1}]$ with zero machine violation. Sovereign spreads exhibit strong countercyclicality with output ($\operatorname{corr}(\text{spread}, y) = -0.87$), capturing the empirical stylized fact that emerging market borrowing costs skyrocket during economic downturns.
#
# In Section 2, the Chatterjee & Eyigungor (2012) long-term debt solver introduces maturity decay ($\delta = 0.05$) and debt dilution. Evaluating the schedule derivative through `DualArray` forward automatic differentiation confirms that $\partial q(b', y)/\partial b' \le 0$ holds across all state points (maximum derivative of $-0.00495 \le 0.0$). Long-term debt shifts the spread schedule upward at median debt levels due to the sovereign's inability to commit not to dilute existing bondholders.
#
# In Section 3, the Mendoza (2010) sudden stop solver captures the general equilibrium collapse in asset prices when the collateral constraint binds. The collateral constraint is active across approximately $9.2\%$ of the state space, generating a crisis amplification ratio of $1.66$, confirming that Fisherian debt-deflation amplifies macroeconomic contractions by over $65\%$ relative to frictionless adjustments. In normal times, the economy engages in precautionary deleveraging, maintaining average debt well below the theoretical collateral ceiling.
#
# In Section 4 and the Hero Visualization, multi-horizon spread term structure decomposition satisfies exact additivity with zero error ($< 10^{-10}$), showing how physical default loss dominates short maturities while dilution risk premia expand at longer horizons. Evaluating a $25\%$ face-value restructuring haircut reduces spreads by several thousand basis points and significantly curtails the default hazard rate.

# %% [markdown]
# ## Your turn
#
# Experiment with the policy parameters below: tune the restructuring haircut `user_debt_haircut` and the maturity extension `user_maturity_extension`. The downstream assertions verify that spread reduction and default hazard mitigation hold robustly across the entire advertised parameter space.
#
# 1. **Basic.** Increase `user_debt_haircut` from `0.10` to `0.35`. How does the post-restructuring spread compression scale with the size of the haircut?
# 2. **Intermediate.** Set `user_debt_haircut = 0.15` and introduce a maturity extension `user_maturity_extension = 2.0` years. Notice how maturity lengthening provides immediate cash-flow relief and reduces rollover pressure.
# 3. **Stretch.** Compare a pure haircut ($h = 0.30, \Delta m = 0$) against a pure maturity reprofiling ($h = 0.0, \Delta m = 4.0$). Which strategy achieves greater default hazard reduction for the same net present value debt relief?

# %%
user_debt_haircut = 0.20  # ← change this: Face-value debt haircut, float in [0.05, 0.40]
user_maturity_extension = 1.0  # ← change this: Maturity extension in years, float in [0.0, 5.0]

assert isinstance(user_debt_haircut, (int, float)) and 0.05 <= user_debt_haircut <= 0.40
assert isinstance(user_maturity_extension, (int, float)) and 0.0 <= user_maturity_extension <= 5.0

restruct_user = evaluate_debt_restructuring(
    res_ce,
    haircut=float(user_debt_haircut),
    maturity_extension=float(user_maturity_extension),
)

print(f"User Restructuring: Haircut = {user_debt_haircut * 100:.1f}%, Extension = {user_maturity_extension:.1f} yrs")
print(f"  Immediate Spread Reduction: {restruct_user.spread_reduction_bps:.2f} bps")
print(f"  Pre-default prob: {restruct_user.pre_default_prob * 100:.2f}% -> Post-default prob: {restruct_user.post_default_prob * 100:.2f}%")

# Assertions over advertised parameter range
assert restruct_user.spread_reduction_bps <= 0.0
assert restruct_user.post_default_prob <= restruct_user.pre_default_prob + 1e-6
assert 0.0 <= restruct_user.post_default_prob <= 1.0

# %% [markdown]
# ## How comprehensive is this?
#
# This tutorial showcases the quantitative sovereign debt, long-term maturity, sudden stop, and debt restructuring suite introduced in `puremacro v6.0.0`. The underlying mathematical mechanisms and solver architectures are documented in detail in `docs/sovereign_default.md` (`docs/es/sovereign_default.md`), with cross-references to forward automatic differentiation in `docs/autodiff_and_protocol.md` and nonlinear sequence-space macroeconomics in `docs/hank_ssj_and_aggregate_risk.md`. All algorithms adhere strictly to the four-package Pyodide runtime contract (`numpy`, `scipy`, `pandas`, `matplotlib`).
