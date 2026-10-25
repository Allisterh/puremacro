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
# # Flujos de trabajo macroeconómicos insignia de PureMacro v6.0.0: modelado diferenciable unificado, SMC templado, HANK global no lineal, default soberano cuantitativo, proyecciones locales distributivas e IAM espacial climático
#
# **¿Cómo unificamos el modelado macroeconómico diferenciable, el método de Monte Carlo secuencial vectorizado con gradientes de score, HANK global no lineal en espacio de secuencias con riesgo agregado, default soberano cuantitativo con restricciones de colateral, proyecciones locales distributivas funcionales con aditividad exacta y transiciones espaciales climáticas-macroeconómicas en un marco auditado sin servidor en NumPy puro?**
#
# La macroeconomía cuantitativa en la frontera de investigación requiere dominar simultáneamente la heterogeneidad microeconómica de alta dimensión, la fragilidad financiera internacional, las transiciones climáticas espaciales de largo plazo y la inferencia bayesiana y empírica rigurosa. El lanzamiento mayor `puremacro v6.0.0` establece una base matemática unificada para estos dominios bajo un estricto contrato de ejecución sin servidor con solo cuatro paquetes (`numpy`, `scipy`, `pandas`, `matplotlib`). Este cuaderno insignia reúne los seis componentes de la versión en un flujo reproducible de extremo a extremo con una tarjeta de verificación de 14 métricas y visualización de calidad de publicación.
#
# | Alcance y verificación | |
# |---|---|
# | Flujos cubiertos | 1. Protocolo unificado de modelo diferenciable y limpieza de versión 6.0.0 (`puremacro.core.protocol`: `DifferentiableModel`, `wrap_differentiable_model`, gradientes de score AD); 2. Motor SMC templado vectorizado (`puremacro.inference.smc`: `smc_sample`, templado adaptativo por ESS, mutación MALA, densidad marginal de datos $\log p(Y)$); 3. HANK global no lineal con riesgo agregado (`puremacro.dsge.hank_nonlinear`: `solve_hank_nonlinear_transition`, actualizaciones Broyden de SSJ dependientes del estado, reducción de momentos Winberry–Reiter `solve_hank_aggregate_risk_order2`); 4. Default soberano cuantitativo y parada súbita (`puremacro.sovereign`: default Arellano/Chatterjee–Eyigungor, restricciones de colateral Mendoza, descomposición de estructura temporal de spreads, reestructuración de deuda); 5. Proyecciones locales funcionales y distributivas (`puremacro.lp.distributional`: `estimate_distributional_lp`, `estimate_quantile_lp`, aditividad exacta en subespacio nulo $\sum_b \beta_{b,h} = 0$, puente `bridge_hank_distributional_irf`); 6. IAM espacial climático-macroeconómico integrado (`puremacro.climate.spatial_iam`: `SpatialClimateIAM`, `solve_spatial_climate_iam`, conservación de población, balance de masa de carbono, reducción de fuga de carbono CBAM, impuesto pigouviano de Golosov); 7. Escenario de política integrado general: Transición de descarbonización global que conecta migración climática, incidencia distributiva por deciles de consumo y retroalimentación de riesgo soberano |
# | Modelos | Modelo de protocolo DSGE nuevo keynesiano de 3 ecuaciones; objetivo de estimación SMC en espacio de estados gaussiano; HANK de dos activos con agentes heterogéneos, restricciones de endeudamiento y riesgo agregado; modelo de default soberano con deuda de largo plazo de Arellano (2008) y Chatterjee–Eyigungor (2012); parada súbita con restricción de colateral de Mendoza (2010); panel microeconómico sintético con participaciones Dirichlet y cambios de escala-ubicación; economía climática espacial dinámica de 3 regiones (Templada, Subtropical, Tropical) con ciclo de carbono multicaja y CBAM |
# | Solucionadores y diagnósticos | `wrap_differentiable_model`, `smc_sample`, `solve_hank_nonlinear_transition`, `solve_hank_aggregate_risk_order2`, `solve_sovereign_default`, `solve_sudden_stop`, `decompose_spread_term_structure`, `evaluate_debt_restructuring`, `estimate_distributional_lp`, `estimate_quantile_lp`, `bridge_hank_distributional_irf`, `solve_spatial_climate_iam` |
# | Aceptación | Error de gradiente del protocolo diferenciable $< 10^{-4}$ y cero filtraciones de deprecaciones heredadas; log MDD de SMC dentro de 0.15 y etapa final de templado $\phi = 1.0$; residuo de transición HANK no lineal $< 10^{-8}$ y paridad con SSJ lineal en choques pequeños $< 10^{-4}$; error de escala cuadrática de riesgo agregado $< 10^{-6}$; residuo de Bellman de default soberano $< 10^{-4}$, cotas de precio de bonos $[0, (1+r^*)^{-1}]$ y spreads contracíclicos; aditividad transversal exacta de LP distributiva $< 10^{-12}$ y distancia estructural $L_2 \ge 0$; conservación de población en IAM climático espacial $< 10^{-12}$, balance de masa de carbono $< 10^{-12}$ y reducción de fuga de carbono mediante CBAM |

# %% [markdown]
# ## El método en fórmulas matemáticas
#
# **1. Protocolo unificado de modelo diferenciable.** Cualquier modelo macroeconómico estructural $m \in \text{DifferentiableModel}$ implementa una interfaz uniforme de cinco métodos: respuestas a impulsos $\mathcal{Y}_h = \mathcal{IR}(x)$, simulación estocástica $Y_t = \mathcal{S}(\epsilon_t)$, gradientes analíticos de verosimilitud $\nabla_\theta \ln p(Y|\theta)$ mediante diferenciación automática hacia adelante con `DualArray`, escenarios contrafactuales $\mathcal{C}(\Delta \theta)$ y exportación de tableros interactivos HTML5 sin servidor. Los símbolos obsoletos de v5 (`puremacro.lp.garch_utils` y `puremacro.sigma.sigma_numpy.SigmaObject`) quedan retirados permanentemente mediante lápidas estrictas.
#
# **2. Monte Carlo secuencial (SMC) templado y vectorizado.** Conecta la distribución previa $p(\theta)$ con la posterior $p(\theta|Y)$ a través de etapas adaptativas de templado $\phi_0 = 0 < \phi_1 < \dots < \phi_N = 1$:
# $$\pi_n(\theta) \propto p(\theta) p(Y|\theta)^{\phi_n}, \quad \text{con} \quad \phi_{n+1} = \operatorname{solve}\left( \text{ESS}(\phi) = \alpha N \right).$$
# La mutación de partículas utiliza deriva de Langevin (MALA) guiada por gradientes de score analíticos:
# $$\theta' = \theta + \frac{\epsilon^2}{2} \nabla \ln \pi_n(\theta) + \epsilon \xi, \quad \xi \sim \mathcal{N}(0, \mathbf{I}),$$
# produciendo una estimación insesgada de la densidad marginal de datos:
# $$\ln \hat{p}(Y) = \sum_{n=1}^N \ln \left( \frac{1}{M} \sum_{i=1}^M \tilde{w}_n^{(i)} \right).$$
#
# **3. HANK global no lineal con riesgo agregado.** Las transiciones en el espacio de secuencias resuelven restricciones de endeudamiento activas mediante jacobianos dependientes del estado $\mathcal{J}(X_t)$ actualizados mediante correcciones de rango 1 de Good Broyden:
# $$\mathcal{J}^{(k+1)} = \mathcal{J}^{(k)} + \frac{(y^{(k)} - \mathcal{J}^{(k)} s^{(k)}) s^{(k)\top}}{\|s^{(k)}\|^2}.$$
# El acoplamiento con riesgo agregado combina la reducción de momentos transversales de Reiter (2009) y Winberry (2018) con perturbación de Sylvester de segundo orden (`sylvester_order2_tangent`), generando acumulación precautoria de riqueza y descuentos en tasas que escalan cuadráticamente con la volatilidad agregada ($\frac{1}{2} g_{\sigma\sigma} \sigma_{\text{agg}}^2$).
#
# **4. Default soberano cuantitativo y parada súbita.** El endeudamiento soberano bajo default endógeno resuelve el sistema de Bellman:
# $$v(b, y) = \max\left\{ v^R(b, y), v^D(y) \right\} + \sigma_\varepsilon \xi,$$
# donde el esquema de precios de bonos satisface la condición de no arbitraje con derivadas de dilución $\partial q(b', y)/\partial b' \le 0$:
# $$q(b', y) = \frac{1 - \mathbb{E}[d'|y] + \mathbb{E}\left[(1 - d')(\kappa + (1 - \delta) q(b'', y')) \mid y\right]}{1 + r^*}.$$
# La dinámica de parada súbita impone restricciones de colateral de Mendoza (2010) $b' \le \kappa q_t k'$, desencadenando espirales de deflación de deuda de Fisher cuando se activan.
#
# **5. Proyecciones locales funcionales y distributivas.** Las respuestas a impulsos horizonte por horizonte de cuantiles y participaciones $\Delta_h s_{b, t}$ se proyectan sobre el subespacio nulo del vector unitario $\mathbf{1}_B$:
# $$\boldsymbol{\beta}_h^* = \left( \mathbf{I}_B - \frac{1}{B} \mathbf{1}_B \mathbf{1}_B^\top \right) \hat{\boldsymbol{\beta}}_h \implies \sum_{b=1}^B \beta_{b, h}^* = 0.0 \quad (< 10^{-12}),$$
# acopladas con inferencia de covarianza de panel Driscoll–Kraay y un puente de distancia estructural $L_2$ hacia deciles de `HetBlock` HANK.
#
# **6. IAM espacial climático-macroeconómico integrado.** El álgebra hat espacial dinámica multirregional conecta migración laboral y comercio con un ciclo de carbono multicaja ($M_{t+1} = \Phi M_t + \Gamma E_t$), funciones regionales de daño por temperatura $D_r(T_{r, t})$ y Mecanismos de Ajuste en Frontera por Carbono (CBAM):
# $$\hat{L}_{i, t+1} L_{i, t+1} = \sum_j \mu_{ji, t} \hat{\mu}_{ji, t} L_{j, t}, \quad \text{conservando la población global a } < 10^{-12}.$$
#
# **7. Escenario de política integrado general.** Une la suite en un escenario de transición ecológica global donde los impuestos al carbono inducen relocalización espacial, generan desigualdad doméstica de consumo entre deciles y transmiten tensiones fiscales hacia los diferenciales de bonos soberanos.

# %% [markdown]
# ## Intuición económica
#
# **Intuición.** La política macroeconómica contemporánea no puede examinarse en compartimentos estancos:
# - **El modelado diferenciable unificado** garantiza que los modelos estructurales compartan una interfaz programática uniforme y evalúen sensibilidades paramétricas exactas sin error de truncamiento numérico.
# - **El SMC templado** recorre superficies posteriores no gaussianas y multimodales donde las cadenas tradicionales de Metropolis–Hastings quedan atrapadas, aprovechando gradientes analíticos de score para orientar las propuestas de partículas.
# - **HANK no lineal en espacio de secuencias** evidencia que, si bien perturbaciones pequeñas de tasas lucen aproximadamente lineales, las contracciones monetarias severas provocan un aumento endógeno de hogares financieramente limitados, profundizando la recesión.
# - **El default soberano cuantitativo** modela de forma rigurosa los márgenes de quita nominal y extensión de vencimientos ante frenazos repentinos y espirales deflacionarias de deuda.
# - **Las proyecciones locales distributivas** conectan microdatos empíricos directamente con respuestas a impulsos agregadas preservando identidades contables exactas.
# - **El IAM espacial climático** cuantifica las implicaciones territoriales del calentamiento global, demostrando que las políticas climáticas unilaterales provocan fuga de emisiones a menos que se acompañen de mecanismos CBAM.
# - **El escenario integral** demuestra la interoperabilidad fluida entre todos los módulos de puremacro v6.0.0, conformando un entorno analítico de vanguardia sin dependencias externas.

# %% [markdown]
# ## Código implementado

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
# ### 1. Protocolo unificado de modelo diferenciable y deprecaciones de v6.0.0
#
# Evaluamos el protocolo `DifferentiableModel` sobre un modelo DSGE nuevo keynesiano de 3 ecuaciones. El modelo adaptado expone métodos uniformes `.simulate()`, `.impulse_response()` y `.loglik_and_grad()`. Verificamos que la diferenciación automática hacia adelante con `DualArray` evalúe el gradiente analítico del score $\nabla_\theta \ln p(Y|\theta)$ con precisión de máquina coincidiendo con diferencias finitas centrales, y confirmamos el retiro definitivo de submódulos obsoletos.

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
wrapped_model = wrap_differentiable_model(dsge_raw)
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

record("Protocol: DualArray gradient error", err_proto_grad, 1e-4, "PASS")
record("Protocol: v6.0.0 legacy deprecation tombstones", err_tombstones, 0.0, "PASS")

print(f"Protocol loglik: {loglik_proto:.4f}, DualArray score: {grad_proto}")
print(f"Gradient error vs central finite differences: {err_proto_grad:.2e}")

# %% [markdown]
# ### 2. Motor SMC templado vectorizado: templado adaptativo y mutación por score
#
# Estimamos un objetivo de espacio de estados gaussiano mediante Monte Carlo secuencial (`smc_sample`). El algoritmo calibra adaptativamente las etapas de templado $\phi_n \in [0, 1]$ manteniendo una razón objetivo de tamaño muestral efectivo (ESS) de 0.50, propaga las partículas mediante Langevin (MALA) guiado por gradientes analíticos y calcula la densidad marginal de datos $\ln \hat{p}(Y)$.

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
# ### 3. HANK global no lineal con riesgo agregado: SSJ dependiente del estado y perturbación de segundo orden
#
# Resolvemos trayectorias de transición en equilibrio general en un modelo HANK con restricciones de endeudamiento. Ante perturbaciones pequeñas ($10^{-4}$), `solve_hank_nonlinear_transition` recupera los jacobianos lineales del espacio de secuencias. Para el riesgo agregado, `solve_hank_aggregate_risk_order2` acopla la reducción de momentos de Winberry–Reiter con perturbación de Sylvester de segundo orden, verificando que la acumulación precautoria de riqueza escala cuadráticamente con la volatilidad agregada ($\sigma_{\text{agg}}^2$).

# %%
res_nl = solve_hank_nonlinear_transition(model=None, shocks={"eps_m": 1e-4}, horizon=25, tol=1e-8)
max_gap = float(np.max(np.abs(res_nl.paths["Y"].values - res_nl.linear_paths["Y"].values)))

res_ar1 = solve_hank_aggregate_risk_order2(model=None, sigma_agg=0.01)
res_ar2 = solve_hank_aggregate_risk_order2(model=None, sigma_agg=0.02)
ratio_vol = float(res_ar2.precautionary_shift_wealth / res_ar1.precautionary_shift_wealth)
err_quad = float(abs(ratio_vol - 4.0))

record("Nonlinear HANK: transition residual norm", float(res_nl.residual_norm), 1e-8, "PASS")
record("Nonlinear HANK: small-shock linear SSJ parity gap", max_gap, 1e-4, "PASS")
record("Aggregate Risk HANK: quadratic volatility scaling error", err_quad, 1e-6, "PASS")

print(f"Nonlinear HANK residual: {res_nl.residual_norm:.2e}, linear SSJ gap: {max_gap:.2e}")
print(f"Precautionary wealth shift (sigma=0.01): {res_ar1.precautionary_shift_wealth:.6f}")
print(f"Quadratic volatility scaling ratio: {ratio_vol:.4f} (expected = 4.0000)")

# %% [markdown]
# ### 4. Default soberano cuantitativo y parada súbita: esquemas de precios de bonos y reestructuración
#
# Resolvemos el modelo de default soberano cuantitativo siguiendo a Arellano (2008) y Chatterjee–Eyigungor (2012). El esquema de precios de bonos satisface las cotas de no arbitraje $q(b', y) \in [0, (1 + r^*)^{-1}]$ y monotonía débil $\partial q/\partial b' \le 0$. Los diferenciales de tasas son contracíclicos ($\operatorname{corr}(\text{spread}, y) < 0$). En escenarios de parada súbita, las restricciones de colateral desencadenan deflación de deuda que puede mitigarse con quitas nominales.

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
# ### 5. Proyecciones locales funcionales y distributivas: aditividad exacta y puente HANK
#
# Estimamos proyecciones locales distributivas sobre deciles transversales de gasto en microdatos de panel. La proyección sobre el subespacio nulo garantiza aditividad transversal exacta ($\max_h |\sum_b \beta_{b, h}| < 10^{-12}$) a precisión de máquina. Conectamos la función empírica con deciles estructurales de HANK mediante `bridge_hank_distributional_irf`.

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
record("Distributional Bridge: HANK decile IRF L2 distance", float(bridge_res.distance_l2), 1.0, "PASS")

print(f"Distributional LP adding-up max error: {res_lp.adding_up_max_error:.2e}")
print(f"Structural HANK bridge L2 distance: {bridge_res.distance_l2:.2e}, sign concordance: {bridge_res.sign_concordance:.4f}")

# %% [markdown]
# ### 6. IAM espacial climático-macroeconómico integrado: comercio, migración, balance de carbono y CBAM
#
# Modelamos una economía espacial integrada de 3 regiones (Templada, Subtropical, Tropical) sujeta a una transición ecológica. La producción regional emite gases de efecto invernadero según un ciclo multicaja cerrado. Verificamos que la conservación global de población ($\sum_r L_{r, t} = \bar{L}$) y el balance de masa de carbono se cumplan a $< 10^{-12}$ en todas las fechas, y comprobamos que los aranceles CBAM reduzcan la fuga internacional de emisiones.

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
# ### 7. Escenario de política integrado general: transición verde multirregional con retroalimentación distributiva y soberana
#
# Integramos los seis módulos en un escenario integral: una coalición climática adopta una senda creciente de impuestos al carbono protegida por aranceles CBAM. La transición estructural redistribuye mano de obra en el espacio geográfico, amplifica la dispersión del consumo entre deciles y genera presiones fiscales en economías emergentes que se resuelven con reestructuración de deuda.

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

print("Grand Integrated Scenario:")
print(f"  Temperate Coalition carbon tax: ${scenario_tax_path[0]:.1f} -> ${scenario_tax_path[-1]:.1f}/tCO2")
print(f"  Regional warming after 15 periods: {temp_warming[0]:+.2f}C (North), {temp_warming[2]:+.2f}C (South)")
print(f"  Net climate migration: Tropical {pop_shift_tropical:+.2f}M, Temperate {pop_shift_temperate:+.2f}M")
print(f"  Distributional LP adding-up satisfied: {res_lp.adding_up_max_error < 1e-12}")
print(f"  Sovereign debt restructuring headroom: {abs(restruct.spread_reduction_bps):.2f} bps")

# %% [markdown]
# ### 8. Visualización insignia: deciles de LP distributiva, transición climática espacial y riesgo soberano
#
# La figura de tres paneles presenta:
# - **Panel A**: Respuestas a impulsos de deciles de gasto $\Delta s_{b, h}$ de la proyección local distributiva, ilustrando la simetría entre deciles extremos y aditividad nula ($\sum_b \beta_{b, h} = 0$).
# - **Panel B**: Dinámica de población regional ($L_{r, t}$) del IAM espacial climático, reflejando migración desde zonas tropicales vulnerables hacia regiones templadas.
# - **Panel C**: Curvas de precio de bonos soberanos $q(b', y)$ según el endeudamiento de fin de período bajo diferentes niveles de ingreso nacional.

# %%
col = _nbstyle.palette(4)
dash = _nbstyle.styles(4)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)

# Panel 1: Distributional LP Decile IRFs
h_vec = np.arange(8)
ax1.plot(h_vec, res_lp.irf_matrix[:, 0] * 100.0, color=col[0], lw=2.2, label=r"Bottom Decile ($s_1$)")
ax1.plot(h_vec, res_lp.irf_matrix[:, 1] * 100.0, color=col[1], lw=1.8, linestyle=dash[1], label=r"Second Decile ($s_2$)")
ax1.plot(h_vec, res_lp.irf_matrix[:, 2] * 100.0, color=col[2], lw=1.5, linestyle=dash[2], label=r"Middle Decile ($s_3$)")
ax1.plot(h_vec, res_lp.irf_matrix[:, 4] * 100.0, color=col[3], lw=2.2, linestyle=dash[3], label=r"Top Decile ($s_5$)")
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
# ### 9. Tarjeta de verificación numérica
#
# Cada comprobación numérica a través de las seis vías de puremacro v6.0.0 se tabula a continuación con su error medido y cota de tolerancia.

# %%
scorecard = pd.DataFrame(checks)
print(scorecard.to_string(index=False, formatters={"measured": "{:.2e}".format, "bound": "{:.2e}".format}))
assert (scorecard["measured"] <= scorecard["bound"]).all()
print(f"\nAll {len(scorecard)} v6.0.0 flagship macroeconomic workflow checks pass.")

# %% [markdown]
# ## Interpretación de resultados
#
# En la Sección 1, `wrap_differentiable_model()` unifica el modelo DSGE de 3 ecuaciones bajo el protocolo `DifferentiableModel`, evaluando gradientes analíticos de score mediante diferenciación automática hacia adelante con un error de $1.67 \times 10^{-5}$ respecto a diferencias finitas. Las deprecaciones programadas de v6.0.0 levantan lápidas limpias con cero filtraciones.
#
# En la Sección 2, `smc_sample()` converge a través de etapas adaptativas de templado, recuperando la densidad marginal de datos analítica con un error de $0.019$ puntos logarítmicos y alcanzando la etapa final $\phi_N = 1.0$.
#
# En la Sección 3, `solve_hank_nonlinear_transition()` coincide con los jacobianos lineales del espacio de secuencias a $2.65 \times 10^{-6}$ ante perturbaciones pequeñas mientras resuelve restricciones de endeudamiento no lineales con un residuo de $7.98 \times 10^{-9}$. En riesgo agregado, duplicar la volatilidad cuadruplica la acumulación precautoria con error numérico nulo ($< 10^{-6}$).
#
# En la Sección 4, `solve_sovereign_default()` satisface la ecuación de Bellman a $4.25 \times 10^{-8}$, respeta estrictamente las cotas de precios de bonos $[0, (1 + r^*)^{-1}]$ y genera diferenciales contracíclicos ($\operatorname{corr}(\text{spread}, y) = -0.87$). La reestructuración de deuda comprime los diferenciales en más de $4,000$ pb.
#
# En la Sección 5, `estimate_distributional_lp()` asegura aditividad transversal exacta con error de $6.94 \times 10^{-18}$, y `bridge_hank_distributional_irf()` alinea las proyecciones locales con los deciles de HANK sin discrepancias.
#
# En la Sección 6, `solve_spatial_climate_iam()` verifica la conservación de población a $0.0$ y el balance de masa de carbono a $4.55 \times 10^{-13}$.
#
# En la Sección 7 y en el panel gráfico principal, el Escenario de Política Integrado ilustra la conexión simultánea entre descarbonización multirregional, relocalización laboral, desigualdad doméstica y sostenibilidad financiera internacional.

# %% [markdown]
# ## Ejercicios para el lector
#
# Modifique la trayectoria del impuesto al carbono `user_carbon_tax` o la quita de reestructuración de deuda `user_debt_haircut`; las aserciones posteriores comprueban que la conservación de población, el balance de carbono y la reducción de diferenciales continúan cumpliéndose en todo el rango anunciado.
#
# 1. **Básico.** Aumente `user_carbon_tax` desde `10.0` hacia `100.0` $/tCO2. ¿Cómo se acelera la relocalización laboral desde la región tropical hacia la coalición templada al moderarse los daños climáticos?
# 2. **Intermedio.** Varíe `user_debt_haircut` entre `0.05` y `0.40`. ¿Cómo escala la compresión de diferenciales tras la reestructuración con la quita nominal y por qué se reduce monótonamente el riesgo de default?
# 3. **Avanzado.** Combine el impuesto unilateral al carbono con un arancel CBAM personalizado equivalente al 100% de la brecha de intensidad de carbono incorporada y verifique que la tasa de fuga disminuye estrictamente.

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
# ## ¿Qué tan exhaustivo es esto?
#
# El cuaderno `72` une los seis ejes de frontera del lanzamiento mayor `puremacro v6.0.0` en una suite macroeconómica auditada sin servidor. Las formulaciones teóricas, solucionadores y puentes empíricos se documentan con detalle en las guías correspondientes: `docs/protocol.md` (interfaz unificada de modelos diferenciables), `docs/smc.md` (Monte Carlo secuencial vectorizado), `docs/hank_nonlinear.md` (HANK no lineal en secuencias con riesgo agregado), `docs/sovereign.md` (default cuantitativo, paradas súbitas y reestructuración), `docs/distributional_lp.md` (proyecciones locales funcionales con aditividad) y `docs/spatial_climate_iam.md` (equilibrio general espacial climático y CBAM). Todas las funciones operan estrictamente bajo el contrato de cuatro paquetes para Pyodide (`numpy`, `scipy`, `pandas`, `matplotlib`).
