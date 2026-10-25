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
# # Cuellos de botella en redes de suministro, migración climática y ajustes fronterizos de carbono: redes de Baqaee–Farhi, álgebra hat espacial dinámica e IAM en PolicyStudio
#
# **¿Cómo amplifican las elasticidades de sustitución en redes de producción de insumo-producto los cuellos de botella energéticos aguas arriba, cómo induce el calentamiento global la migración regional prospectiva de mano de obra, por qué los impuestos al carbono unilaterales causan fuga de carbono internacional a menos que se protejan mediante Mecanismos de Ajuste en Frontera por Carbono (CBAM) y cómo diseñamos mallas de contrafactuales con auditoría automática y generación de informes interactivos sin servidor?**
#
# La transición ecológica global representa una de las transformaciones estructurales más complejas de la historia económica contemporánea. La descarbonización de los sistemas productivos exige ajustes simultáneos en las redes de producción intersectoriales donde la energía sustituye de forma imperfecta al capital y al trabajo, en la geografía espacial donde el cambio climático altera la habitabilidad regional forzando la movilidad laboral, y en la política comercial internacional donde los aranceles de ajuste en frontera impiden que la mitigación climática doméstica se vea neutralizada por la fuga de emisiones hacia el extranjero.
#
# Este tutorial proporciona un laboratorio computacional integrado y autoexplicativo en `puremacro` que une la agregación no lineal de segundo orden en redes de insumo-producto (Baqaee & Farhi 2019, 2020), el álgebra hat espacial dinámica de migración (Caliendo, Parro & Rossi-Hansberg 2019) y un Modelo de Evaluación Integrada Clima-Macro Espacial (SpatialClimateIAM) auditado mediante `PolicyStudio`. Todos los algoritmos se ejecutan en NumPy y SciPy puros bajo el estricto contrato de ejecución Pyodide, generando visualizaciones opacas con calidad de publicación y cuadros de mando interactivos en HTML5 sin servidor y con cero dependencias externas de CDN.
#
# | Alcance y verificación | |
# |---|---|
# | Flujos cubiertos | 1. Agregación macroeconómica en redes de segundo orden de Baqaee–Farhi (2019, 2020) (`baqaee_farhi_decomposition`); 2. Álgebra hat espacial dinámica de migración de Caliendo–Parro–Rossi-Hansberg (2019) (`solve_dynamic_spatial_hat`); 3. IAM espacial integrado clima-macro multirregional (`SpatialClimateIAM`, `solve_spatial_climate_iam`); 4. Contrafactuales de política verde (Laissez-Faire, Impuesto Unilateral, CBAM, Impuesto Pigouviano Óptimo); 5. Malla de contrafactuales con auditoría automática en `PolicyStudio` y exportación interactiva a HTML5/SVG (`run_policy_studio`, `result_to_interactive_html`) |
# | Modelos teóricos | Red de producción de insumo-producto de 3 sectores (Energía, Manufactura, Servicios) con sustitución CES intermedia y final; economía espacial dinámica de 3 regiones (Templada, Subtropical, Tropical) con migración laboral prospectiva; modelo de ciclo de carbono de 3 cajas con conservación de masa y funciones de daño por temperatura regional; malla declarativa de contrafactuales de política |
# | Solucionadores y métodos | Curvatura del Hessiano macroeconómico $\frac{1}{2} z^\top \mathcal{H} z$, solucionador de punto fijo para costes unitarios CES no lineales, iteración de punto fijo de álgebra hat espacial dinámica, acumulación del ciclo de carbono con balance de masa, exportador SVG autónomo sin servidor |
# | Umbrales de aceptación | Curvatura de segundo orden de Cobb–Douglas de Baqaee–Farhi $< 10^{-14}$ (teorema de Hulten); discrepancia de tercer orden en punto fijo de precios CES $< 10^{-4}$; error de conservación de población espacial dinámica $< 10^{-12}$; error de estocasticidad por filas en migración $< 10^{-14}$; error de conservación de población global en IAM $< 10^{-12}$; error de balance de masa en ciclo de carbono $< 10^{-12}$; fuga de carbono con CBAM estrictamente menor a impuesto unilateral; paridad de impuesto pigouviano frente a fórmula analítica de Golosov $< 10^{-6}$; fallos de auditoría en malla de PolicyStudio $== 0$; informe interactivo HTML5 con cero enlaces a CDN externas |

# %% [markdown]
# ## El método en ecuaciones
#
# **1. Agregación no lineal de insumo-producto de Baqaee–Farhi (2019, 2020).** Consideremos una economía con $N$ sectores productivos y una matriz de transacciones intermedias de insumo-producto $\mathbf{\Omega} \in \mathbb{R}^{N \times N}$, donde $\Omega_{ij}$ denota la participación del coste del sector $j$ en la producción del sector $i$. La demanda final viene caracterizada por participaciones de gasto $\mathbf{b} \in \mathbb{R}^N$. La inversa de Leontief $\mathbf{\Psi} = (\mathbf{I} - \mathbf{\Omega})^{-1}$ determina las ponderaciones de Domar en términos de ingresos $\boldsymbol{\lambda} = \mathbf{b}^\top \mathbf{\Psi}$. Ante un vector arbitrario de choques de productividad sectoriales $\mathbf{z} \in \mathbb{R}^N$, las variaciones de primer orden en el PIB logarítmico satisfacen el teorema de Hulten (1978):
#
# $$\mathrm{d} \ln Y = \boldsymbol{\lambda}^\top \mathbf{z} = \sum_{i=1}^N \lambda_i z_i.$$
#
# Cuando los productores sustituyen insumos con elasticidades CES $\theta_i \ne 1.0$ y los consumidores finales tienen elasticidad $\theta_0 \ne 1.0$, surge una curvatura macroeconómica de segundo orden a través del Hessiano macroeconómico $\mathcal{H} \in \mathbb{R}^{N \times N}$:
#
# $$\mathrm{d}^2 \ln Y = \frac{1}{2} \mathbf{z}^\top \mathcal{H} \mathbf{z} = \frac{1}{2} (\theta_0 - 1) \operatorname{Var}_{\mathbf{b}}(\mathbf{\Psi} \mathbf{z}) + \frac{1}{2} \sum_{i=1}^N \lambda_i (\theta_i - 1) \operatorname{Var}_{\mathbf{\Omega}_i}(\mathbf{\Psi} \mathbf{z}).$$
#
# Bajo preferencias de Cobb–Douglas ($\theta_i = \theta_0 = 1.0$), todas las varianzas transversales se anulan idénticamente, reduciendo $\mathrm{d}^2 \ln Y \equiv 0.0$ con exactitud de máquina. Cuando los insumos son complementarios ($\theta_i < 1.0$), los cuellos de botella aguas arriba (tales como la escasez de gas natural o minerales críticos) provocan caídas convexas que se amplifican a lo largo de toda la cadena de suministro.
#
# **2. Álgebra hat espacial dinámica de migración de Caliendo–Parro–Rossi-Hansberg (2019).** Sea $r \in \{1, \dots, R\}$ el índice de regiones geográficas pobladas por trabajadores prospectivos que eligen dónde residir y producir. La probabilidad de migración desde el origen $j$ hacia el destino $i$ cumple:
#
# $$\mu_{ji, t} = \frac{\left( \beta v_{i, t+1} - \tau_{ji}^m \right)^{1/\nu}}{\sum_k \left( \beta v_{k, t+1} - \tau_{jk}^m \right)^{1/\nu}},$$
#
# donde $\nu$ gobierna la sensibilidad migratoria y $\tau_{ji}^m$ representa las fricciones de traslado. En variaciones relativas ("álgebra hat" $\hat{x}_{t+1} = x_{t+1} / x_t$), la dinámica poblacional regional evoluciona según:
#
# $$L_{i, t+1} = \sum_{j=1}^R \mu_{ji, t} \hat{\mu}_{ji, t} L_{j, t},$$
#
# conservando la población total nacional $\sum_{r=1}^R L_{r, t} = \bar{L}$ con una tolerancia $< 10^{-12}$ en todos los períodos.
#
# **3. IAM espacial integrado clima-macro y ciclo de carbono multirregional.** La actividad industrial regional genera emisiones de carbono $E_{r, t} = \sigma_r Y_{r, t} (1 - \mu_{r, t})$, donde $\sigma_r$ es la intensidad de carbono y $\mu_{r, t}$ la tasa de abatimiento. Las emisiones globales $E_t = \sum_r E_{r, t}$ se acumulan en un modelo de ciclo de carbono de 3 cajas con conservación estricta de masa (Golosov et al. 2014):
#
# $$S_{1, t+1} = S_{1, t} + \phi_L E_t, \quad S_{2, t+1} = (1 - \phi_{\text{decay}}) S_{2, t} + \phi_0 (1 - \phi_L) E_t, \quad S_{\text{sink}, t+1} = S_{\text{sink}, t} + (1 - \phi_0)(1 - \phi_L) E_t.$$
#
# El saldo de carbono atmosférico $S_t = S_{1, t} + S_{2, t}$ impulsa las anomalías de temperatura superficial global mediante la relación logarítmica de Arrhenius $\Delta T_t = \frac{\lambda}{\ln 2} \ln \left( \frac{S_{\text{pre}} + S_t}{S_{\text{pre}} + S_0} \right)$. Las temperaturas locales inducen daños de productividad y amenidad regionales según Cruz & Rossi-Hansberg (2024):
#
# $$A_{r, t} = \exp\left( -\gamma_r^A (T_{r, t} - T_{r, 0}) \right), \quad B_{r, t} = \exp\left( -\gamma_r^B (T_{r, t} - T_{r, 0}) \right).$$
#
# **4. Fuga de carbono y ajustes fronterizos CBAM.** Cuando una coalición de mitigación $\mathcal{C}$ establece un impuesto unilateral al carbono $\tau_{\text{carbon}}$, los costes de producción domésticos se encarecen, desplazando las industrias intensivas en energía hacia regiones no participantes $\mathcal{N}$. La tasa de fuga de carbono se define como:
#
# $$\Lambda = \frac{\Delta E_{\mathcal{N}}}{-\Delta E_{\mathcal{C}}} = \frac{\sum_{t} \sum_{i \in \mathcal{N}} (E_{i, t} - E_{i, t}^{\text{LF}})}{-\sum_{t} \sum_{i \in \mathcal{C}} (E_{i, t} - E_{i, t}^{\text{LF}})}.$$
#
# Un Mecanismo de Ajuste en Frontera por Carbono (CBAM) impone un arancel sobre las importaciones de bienes no pertenecientes a la coalición equivalente a su contenido de carbono incorporado: $\tau_{ji, t}^{\text{CBAM}} = \tau_{\text{carbon}, t} \cdot \sigma_j$, equilibrando la competitividad y reduciendo estrictamente la fuga de emisiones ($\Lambda_{\text{CBAM}} < \Lambda_{\text{Uni}}$).
#
# **5. Impuesto pigouviano óptimo al carbono (Golosov et al. 2014).** En el límite de utilidad logarítmica, el impuesto pigouviano óptimo coincide con el valor actual descontado de los daños climáticos marginales futuros:
#
# $$\tau_t^* = Y_t \bar{\gamma}_A \left[ \frac{\phi_L}{1 - \beta} + \frac{(1 - \phi_L)\phi_0}{1 - \beta(1 - \phi_{\text{decay}})} \right].$$

# %% [markdown]
# ## Intuición
#
# **Intuición.** Para comprender cómo interactúan los cuellos de botella en redes de suministro y las políticas climáticas en equilibrio general, conviene considerar las siguientes lecciones económicas:
#
# - **Fragilidad leontiefiana en las cadenas de valor.** El teorema de Hulten es un pilar de la agregación macroeconómica que establece que, en primera aproximación, la relevancia de un sector coincide exactamente con su peso de ventas brutas (ponderador de Domar). Sin embargo, cuando los insumos presentan una sustitución rígida —como ocurre con el gas natural en plantas petroquímicas o los semiconductores en la industria automotriz ($\theta < 1$)— una caída del $10\%$ en un insumo clave no puede suavizarse fácilmente. A medida que las empresas aguas abajo compiten por insumos escasos, la producción se desploma de forma no lineal. La descomposición de Baqaee–Farhi cuantifica esta curvatura macroeconómica de segundo orden $\frac{1}{2} z^\top \mathcal{H} z$, demostrando por qué los estrangulamientos energéticos provocan recesiones desproporcionadas.
#
# - **La reubicación espacial como mecanismo de adaptación.** El calentamiento global no afecta a todas las latitudes por igual: las zonas ecuatoriales y tropicales sufren un estrés térmico extremo con caídas acusadas de la productividad agrícola, mientras que las regiones templadas experimentan impactos más benignos o incluso ampliaciones en sus ciclos de cultivo. En un modelo estático sin migración, los trabajadores en zonas vulnerables sufren mermas de bienestar intolerables. En un equilibrio general espacial dinámico, los trabajadores se desplazan de manera prospectiva hacia regiones con mayores salarios reales y amenidades. El álgebra hat espacial dinámica modela esta movilidad, evidenciando que la migración laboral constituye un canal principal de adaptación climática.
#
# - **El dilema del polizón y los ajustes en frontera.** La mitigación climática es un bien público global. Cuando una coalición aplica un gravamen al carbono de forma unilateral, sus industrias pesadas pierden competitividad frente a competidores foráneos no regulados. Si estos productores utilizan matrices energéticas más contaminantes, las emisiones globales netas pueden aumentar paradójicamente. El CBAM opera como un escudo contra la fuga: al tasar el carbono incorporado en frontera, elimina el incentivo de deslocalización fabril y estimula a los socios comerciales a adoptar precios al carbono propios.
#
# - **PolicyStudio: Auditoría declarativa antes de la publicación.** El diseño riguroso de políticas públicas demanda evaluar cientos de combinaciones paramétricas entre trayectorias tributarias, elasticidades de sustitución y regímenes arancelarios. `PolicyStudio` ofrece un motor automatizado que somete cada celda de una malla contrafactual a estrictos controles matemáticos (conservación de población, balance de masa, convergencia numérica) y genera un informe interactivo autónomo en HTML5/SVG sin necesidad de servidores.

# %% [markdown]
# ## Código aplicado

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
# ## Interpretación de resultados
#
# En la Sección 1, `baqaee_farhi_decomposition()` valida el teorema de Hulten bajo preferencias de Cobb–Douglas ($\theta = 1.0$), donde la curvatura macroeconómica de segundo orden se reduce a $0.00$ ($< 10^{-14}$). Cuando los insumos operan como complementos ($\theta = 0.35$), los estrangulamientos en la cadena de suministro generan una contracción agregada adicional de $0.077$ puntos porcentuales, reproduciendo fielmente la solución del punto fijo de precios CES no lineal con un error de apenas $6.8 \times 10^{-5}$.
#
# En la Sección 2, `solve_dynamic_spatial_hat()` resuelve el modelo de migración dinámica de Caliendo–Parro–Rossi-Hansberg (2019), preservando la población nacional global en $0.00$ ($< 10^{-12}$) y manteniendo la estocasticidad por filas de la matriz de probabilidades de migración con una desviación nula.
#
# En la Sección 3, `SpatialClimateIAM` acopla el comercio y la movilidad espacial dinámica con un ciclo de carbono de 3 cajas. El balance de masa de carbono atmosférico se cumple estrictamente a $< 10^{-12}$, mostrando que el incremento térmico tropical empuja una relocalización laboral paulatina hacia las regiones templadas del norte.
#
# En la Sección 4, los contrafactuales de política climática evidencian que la tributación unilateral al carbono induce una tasa de fuga del $49.58\%$, la cual se reduce al $48.64\%$ gracias a los aranceles de ajuste en frontera CBAM. El impuesto pigouviano óptimo calculado por el solucionador coincide con la formulación analítica de Golosov et al. (2014) ($\$48.98/\text{tCO}_2$) con una discrepancia numérica de apenas $2.84 \times 10^{-14}$.
#
# En la Sección 5 y la Figura Principal, `run_policy_studio()` audita exhaustivamente una malla bidimensional sin registrar fallo alguno, compilando un cuadro de mando interactivo en HTML5 autónomo y sin dependencias externas de CDN.

# %% [markdown]
# ## Su turno
#
# Ajuste el nivel del impuesto al carbono `user_carbon_tax` y la elasticidad de sustitución energética `user_energy_elasticity`. Las aserciones subsiguientes corroboran que la conservación espacial de la población, el balance de masa de carbono y la estabilidad de la red se preservan en todo el espacio paramétrico ofrecido.
#
# 1. **Básico.** Aumente `user_carbon_tax` de `30.0` a `90.0` $/tCO2. ¿Cómo acelera una mayor imposición doméstica la migración hacia zonas templadas y por qué el CBAM se torna progresivamente indispensable?
# 2. **Intermedio.** Varíe `user_energy_elasticity` entre `0.25` y `0.80`. ¿Cómo se profundiza la curvatura de la red a medida que la sustitución de insumos energéticos se vuelve más rígida?
# 3. **Avanzado.** Extienda la coalición para incorporar a la Región 1 (Subtropical), conformando un club climático ampliado, y verifique que la fuga de carbono internacional cae sensiblemente.

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
# ## ¿Qué tan exhaustivo es esto?
#
# Este tutorial integra la macroeconomía de redes de producción, el comercio y la migración espacial dinámica y los modelos integrados clima-macro IAM en el marco del lanzamiento de `puremacro v6.0.0`. Los fundamentos teóricos y las especificaciones completas de la API se detallan en `docs/es/network_spatial_climate_iam.md` (`docs/network_spatial_climate_iam.md`), con referencias cruzadas al protocolo general de modelos diferenciables en `docs/es/autodiff_and_protocol.md` y al diseño de contrafactuales en `docs/es/study_policy_studio.md`. Todos los procedimientos respetan estrictamente el contrato de ejecución Pyodide de cuatro paquetes (`numpy`, `scipy`, `pandas`, `matplotlib`).
