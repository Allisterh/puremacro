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
# # Dynare-like Automated VFI — Declarative Structural Dynamic Programming
#
# **Can we write structural dynamic programming models in declarative syntax without low-level grid loops?** In macroeconomics, Dynare transformed DSGE modeling by replacing manual matrix derivations with a clean declarative language. `puremacro.dp` brings the same idea to **heterogeneous-agent dynamic programming**: you declare parameters, a state grid, a shock process, the budget and the reward as equations in Dynare expression syntax, and `Model.solve()` writes the solver call. There are no grid loops, return-function closures or Tauchen matrices to manage. This notebook builds an incomplete-markets precautionary savings model, solves it by value function iteration, reads the policy and value functions, and measures wealth inequality exactly from the stationary distribution and by simulating a panel of households.

# %% [markdown]
# ## The method in math — the Bellman equation
#
# The single-agent problem on state $(a, z)$ solves:
# $$ V(a, z) = \max_{a' \in \mathcal{A}} \left\{ u(c) + \beta \sum_{z'} P(z, z') V(a', z') \right\} $$
# subject to the budget constraint:
# $$ c + a' = e^{z} a^{\alpha} + (1-\delta) a, \qquad c > 0, $$
# where $u(c) = \frac{c^{1-\gamma}}{1-\gamma}$, $z$ follows an AR(1) process $z_t = \rho z_{t-1} + \varepsilon_t$, and $P(z, z')$ is the Markov transition matrix of its Tauchen discretization.
#
# **Intuition.** Each line of the declaration below is one piece of this problem. `puremacro.dp` parses the equations, discretizes the shock, evaluates the reward on the grid $\mathcal{A} \times \mathcal{Z}$ and calls the `puremacro.vfi` engine. Discrete VFI keeps $a'$ on the grid nodes. The budget $e^z a^\alpha$ is not of the form $R a + y(z)$, so the endogenous grid method does not apply and we solve with `method="vfi"`.

# %% [markdown]
# ## Setup — declarative model specification
#
# We create `dp.Model()`, set the parameters ($\beta=0.96, \gamma=2.0, \alpha=0.36, \delta=0.10$), define an asset grid of 60 points on $a \in [0.1, 15.0]$, and declare an AR(1) productivity shock $z$ with 5 Tauchen states.

# %%
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_cwd = Path.cwd()
sys.path.insert(0, str(_cwd if (_cwd / "_nbstyle.py").exists() else _cwd / "notebooks"))
import _nbstyle
_nbstyle.apply_style()

from puremacro import dp

# Declare the structural model
model = dp.Model("Precautionary Savings Model")
model.parameters(beta=0.96, gamma=2.0, alpha=0.36, delta=0.10)

# Shock process and state grid
model.exogenous("z", dp.AR1(rho=0.90, sigma=0.15, n=5))
model.state("a", np.linspace(0.1, 15.0, 60))

# Budget constraint and utility, as equations
model.local("c = exp(z)*a^alpha + (1 - delta)*a - a(+1)")
model.reward("c^(1 - gamma)/(1 - gamma)")
model.subject_to("c > 0")

# %% [markdown]
# ## Solving the model and measuring wealth inequality
#
# `model.solve(method="vfi")` iterates on the Bellman equation until the value function changes by less than `tol`, then computes the stationary distribution of households over the $(a, z)$ grid. `sol.mean("a")` and `sol.gini("a")` are computed exactly from that distribution, with no simulation. The Gini is the standard weighted Gini of assets, $1 - \sum_i w_i (L_i + L_{i-1})$ on the Lorenz curve of the grid values $a_i$ weighted by their stationary mass $w_i$.

# %%
sol = model.solve(method="vfi", tol=1e-6)

V = sol.V
a_grid = sol.grids["a"]
policy_a = sol.policy("a(+1)")
mean_assets = sol.mean("a")
gini_coef = sol.gini("a")
top_mass = sol.distribution[-1].sum()

print(f"Model: {sol.model_name} (method: {sol.method})")
print(f"Stationary Mean Assets: {mean_assets:.4f}")
print(f"Wealth Gini Coefficient: {gini_coef:.4f}")
print(f"Mass at the top grid point a = 15: {top_mass:.4f}")

# Headline assertions
assert abs(mean_assets - 6.5733) < 1e-3, f"Mean assets moved: {mean_assets}"
assert abs(gini_coef - 0.3494) < 1e-3, f"Gini coefficient moved: {gini_coef}"

# %% [markdown]
# Mean assets are 6.5733 and the Gini is 0.3494. Earlier versions of this notebook printed 6.6767 and 0.3924. Those came from a 2000-household, 100-period simulation, and that Gini was computed after subtracting the minimum asset holding, so it was not the standard Gini. About 8% of the mass sits at the top grid point, so the upper end of the grid binds: widen it before reading the top of the distribution.

# %% [markdown]
# ## Visualizing policy functions and value functions
#
# The hero figure shows the policy rules $a'(a, z)$ for the lowest and highest productivity states, and the value functions.

# %%
fig, (ax1, ax2) = _nbstyle.figura(1, 2, figsize=(10.5, 4.2))

# Left panel: Policy functions
ax1.plot(a_grid, a_grid, linestyle="--", color=_nbstyle.SPINE, alpha=0.7, label="45° Line (a' = a)")
ax1.plot(a_grid, policy_a[:, 0], color=_nbstyle.palette(2)[0], lw=2, label="Low Productivity (z_min)")
ax1.plot(a_grid, policy_a[:, -1], color=_nbstyle.palette(2)[1], lw=2, label="High Productivity (z_max)")
ax1.set_xlabel("Current Assets a", color=_nbstyle.TEXTO)
ax1.set_ylabel("Next Period Assets a'", color=_nbstyle.TEXTO)
ax1.set_title("Optimal Asset Policy Functions a'(a, z)")
ax1.legend(loc="upper left", frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)

# Right panel: Value function
ax2.plot(a_grid, V[:, 0], color=_nbstyle.palette(2)[0], lw=2, label="V(a, z_min)")
ax2.plot(a_grid, V[:, -1], color=_nbstyle.palette(2)[1], lw=2, label="V(a, z_max)")
ax2.set_xlabel("Assets a", color=_nbstyle.TEXTO)
ax2.set_ylabel("Value V(a, z)", color=_nbstyle.TEXTO)
ax2.set_title("Value Functions V(a, z)")
ax2.legend(loc="lower right", frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)

# %% [markdown]
# **Reading the output.** Low-productivity agents ($z_{\min}$) decumulate assets when wealth is high, while high-productivity agents ($z_{\max}$) save. Precautionary motives prevent wealth from collapsing to the bottom of the grid.

# %% [markdown]
# ## Simulating a panel of households
#
# `sol.simulate(n_agents, periods)` draws individual histories under the solved policy and returns a `dp.Panel`. Here every household starts at the bottom of the grid, $a = 0.1$, with its shock drawn from the stationary distribution of $z$. The cross-sectional Gini of the panel, `panel.gini("a", by="t")`, first rises as lucky households accumulate and then settles at the exact stationary value. The same seed gives the same panel.

# %%
panel = sol.simulate(n_agents=5000, periods=200, seed=0, init={"a": 0.1})
gini_t = panel.gini("a", by="t")

print(panel)
print(f"Panel Gini at t = 0, 10, 199: {gini_t.iloc[0]:.4f}, {gini_t.iloc[10]:.4f}, {gini_t.iloc[-1]:.4f}")
print(f"Exact stationary Gini:        {gini_coef:.4f}")

assert abs(gini_t.iloc[-1] - gini_coef) < 0.02, "The panel Gini should settle near the exact value"

fig, ax = _nbstyle.figura(1, 1, figsize=(8.0, 3.6))
ax.plot(gini_t.index, gini_t.to_numpy(), color=_nbstyle.palette(2)[0], lw=2, label="Panel Gini of a, by period")
ax.axhline(gini_coef, linestyle="--", color=_nbstyle.palette(2)[1], lw=1.5, label=f"Exact stationary Gini ({gini_coef:.4f})")
ax.set_xlabel("Period t", color=_nbstyle.TEXTO)
ax.set_ylabel("Gini of assets", color=_nbstyle.TEXTO)
ax.set_title("A simulated panel converges to the stationary distribution")
ax.legend(loc="lower right", frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)

# %% [markdown]
# ## Your turn — precautionary savings under higher risk aversion
#
# **Prompts.**
# 1. *Basic*: Increase risk aversion from $\gamma = 2.0$ to $\gamma = 3.5$ (`# ← change this ...`). Does mean stationary wealth increase due to stronger precautionary motives?
# 2. *Intermediate*: Lower the discount factor to $\beta = 0.94$. Verify that impatient households hold less capital.
# 3. *Stretch*: Increase the shock's standard deviation to $\sigma_z = 0.25$ and compare `sol_yt.gini("a")` with the baseline.

# %%
# Your turn code block
gamma_new = 3.0  # ← change this: try 3.0 or 3.5

m_yt = dp.Model("High Risk Aversion")
m_yt.parameters(beta=0.96, gamma=gamma_new, alpha=0.36, delta=0.10)
m_yt.exogenous("z", dp.AR1(rho=0.90, sigma=0.15, n=5))
m_yt.state("a", np.linspace(0.1, 15.0, 60))
m_yt.local("c = exp(z)*a^alpha + (1 - delta)*a - a(+1)")
m_yt.reward("c^(1 - gamma)/(1 - gamma)")
m_yt.subject_to("c > 0")

sol_yt = m_yt.solve(method="vfi")
print(f"New Risk Aversion gamma = {gamma_new:.1f}")
print(f"New Stationary Mean Assets: {sol_yt.mean('a'):.4f}")
print(f"New Wealth Gini: {sol_yt.gini('a'):.4f}")

# Assertion holding for default
assert sol_yt.mean("a") > 0.0, "Mean assets must be positive"

# %% [markdown]
# ## How comprehensive is this?
#
# `puremacro.dp` compiles the declaration to the `puremacro.vfi` engines; `sol.raw` holds the engine's result. The same declaration style covers life cycles, discrete choices with taste shocks, the endogenous grid method, continuous time and market-clearing prices (see `docs/dp.md`). `sol.top_share`, `sol.quantile` and `sol.lorenz` give the rest of the exact wealth distribution. This notebook used `puremacro.vfi.Model` before; that class is deprecated and will be removed in 6.0.
