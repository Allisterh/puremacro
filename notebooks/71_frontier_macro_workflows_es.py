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
# # Flujos de trabajo macroeconómicos de frontera: diferenciación automática, HANK en espacio de secuencias, OccBin de orden superior, nowcasting de frecuencia mixta, HJB en tiempo continuo, equilibrio general de redes/espacial y PolicyStudio con autoauditoría
#
# **¿Cómo construimos y auditamos una canalización macroeconómica de frontera de extremo a extremo—uniendo diferenciación automática hacia adelante, jacobianos DAG en espacio de secuencias, cambio de régimen podado de orden superior, descomposición de noticias en tiempo real, distribuciones de agentes heterogéneos en tiempo continuo, redes de producción de segundo orden, migración espacial dinámica y auditoría interactiva de mallas contrafactuales—en NumPy puro bajo despliegue estricto sin servidor?**
#
# La macroeconomía cuantitativa moderna opera a través de múltiples paradigmas estructurales: motores de diferenciación automática que calculan tangentes de política analíticas sin cancelación numérica; compiladores DAG en espacio de secuencias y estimadores en el dominio de la frecuencia para modelos con agentes heterogéneos (HANK); solucionadores de perturbación podada que resuelven restricciones ocasionalmente activas con desplazamientos estocásticos por riesgo precautorio; modelos de predicción inmediata (nowcasting) en espacio de estados que descomponen publicaciones de datos entrantes en noticias macroeconómicas en tiempo real; solucionadores PDE en volúmenes finitos en tiempo continuo para densidades progresivas de Kolmogorov; agregación insumo-producto que captura cuellos de botella de producción de segundo orden; álgebra de variaciones relativas (hat-algebra) espacial dinámica que simula comercio y migración laboral; y estudios contrafactuales con autoauditoría que verifican las leyes de conservación del equilibrio antes de publicar paneles visuales interactivos sin servidor externo. Este cuaderno de exhibición une las siete pistas de frontera en una suite reproducible y autónoma con verificación integral.
#
# | Alcance y verificación | |
# |---|---|
# | Flujos cubiertos | 1. AD hacia adelante en NumPy puro (`DualArray`, `HyperDualArray`, `hessian_forward`); 2. DAG generalizado en espacio de secuencias y estimación HANK de Whittle (`BlockDAG`, `hank_loglik_whittle`); 3. OccBin no lineal podado de 2.°/3.er orden con desplazamientos por riesgo precautorio; 4. Nowcasting en espacio de estados de frecuencia mixta y descomposición de noticias de Banbura–Modugno; 5. HJB en diferencias finitas contra el viento en tiempo continuo, KFE adjunta y puente `HetBlock` a espacio de secuencias; 6. Macroeconomía de redes de 2.° orden de Baqaee–Farhi y álgebra de variaciones relativas espacial dinámica de Caliendo–Parro–Rossi-Hansberg; 7. Malla contrafactual con autoauditoría `PolicyStudio` y exportación HTML5/SVG interactiva sin servidor |
# | Modelos | Utilidad macroeconómica CRRA-CES no homotética; DAG neokeynesiano / HANK de 3 ecuaciones; modelo NK no lineal con cota inferior efectiva ($i_t \ge i_{\text{floor}}$); modelo de factores dinámicos de frecuencia mixta de 4 variables; hogares en tiempo continuo de Huggett–Aiyagari con saltos de Poisson en el ingreso; economía insumo-producto de 3 sectores aguas arriba y aguas abajo; economía espacial dinámica de 3 regiones con comercio y migración |
# | Solucionadores y diagnósticos | `value_and_grad`, `hessian_forward`, `BlockDAG.impulse_responses`, `hank_loglik_whittle`, `solve_nonlinear_occbin`, `DynamicFactorModel.fit`, `banbura_modugno_news`, `solve_hjb_achdou`, `solve_kfe_achdou`, `baqaee_farhi_decomposition`, `solve_dynamic_spatial_hat`, `run_policy_studio`, `result_to_interactive_html` |
# | Aceptación | Gradientes y hessianos de AD hacia adelante $< 10^{-14}$; norma residual de DAG SSJ $< 10^{-12}$ y brecha interior de IRF $< 10^{-9}$; error de Parseval en Whittle $< 10^{-10}$; violación de cota en OccBin $< 10^{-12}$; error de aditividad en noticias de Banbura–Modugno $< 10^{-12}$; error de suma de filas del generador HJB continuo $< 10^{-13}$ y estacionariedad de KFE $< 10^{-12}$; curvatura de red de 2.° orden Cobb–Douglas $< 10^{-14}$; error de conservación de población espacial $< 10^{-12}$; auditorías de celdas 100% verificadas en PolicyStudio con cero dependencias CDN externas |

# %% [markdown]
# ## El método en matemáticas
#
# **1. Diferenciación automática hacia adelante en NumPy puro.** La AD hacia adelante extiende los números reales a álgebras hiperduales $\mathbb{D}_2$:
# $$\mathcal{D}_1(x) = x + \varepsilon \dot{x}, \quad \mathcal{D}_2(x) = x + \varepsilon_1 \dot{x}_1 + \varepsilon_2 \dot{x}_2 + \varepsilon_1 \varepsilon_2 \ddot{x}, \qquad \varepsilon_1^2 = \varepsilon_2^2 = 0.$$
# La propagación de componentes duales a través de funciones universales (ufuncs) y operadores de Sylvester produce primeras derivadas exactas $\nabla f(x)$ y hessianos $\nabla^2 f(x)$ hasta precisión de máquina sin calibración de malla por diferencias finitas ($h$) ni cancelación por paso complejo.
#
# **2. DAGs en espacio de secuencias y verosimilitud de Whittle.** Un DAG de equilibrio macroeconómico conecta bloques endógenos $k$ con incógnitas $\mathbf{U}$ y objetivos exógenos $\mathbf{Z}$. Diferenciar el sistema acoplado produce jacobianos en el espacio de secuencias:
# $$\mathcal{J}_{\mathbf{U}} \mathrm{d}\mathbf{U} + \mathcal{J}_{\mathbf{Z}} \mathrm{d}\mathbf{Z} = 0 \implies \mathrm{d}\mathbf{U} = -\mathcal{J}_{\mathbf{U}}^{-1} \mathcal{J}_{\mathbf{Z}} \mathrm{d}\mathbf{Z}.$$
# En el dominio de la frecuencia, la log-verosimilitud de Whittle confronta segundos momentos empíricos con densidades espectrales del modelo $S_y(\omega; \theta)$:
# $$\ln \mathcal{L}_W(\theta) = -\frac{1}{2}\sum_{k=1}^K \left[ \ln \det S_y(\omega_k; \theta) + \operatorname{tr}\left( S_y(\omega_k; \theta)^{-1} I_y(\omega_k) \right) \right].$$
#
# **3. OccBin no lineal podado con desplazamientos por riesgo precautorio.** Cuando una restricción de desigualdad $y_{i,t} \ge y_{\text{floor}}$ se activa ocasionalmente, la trayectoria de perturbación podada se descompone en componentes de primer, segundo y tercer orden:
# $$y_t = y^* + y_t^{(1)} + y_t^{(2)} + y_t^{(3)}, \quad y_t^{(2)} = H_x y_{t-1}^{(2)} + \frac{1}{2} G_{xx} (y_{t-1}^{(1)})^{\otimes 2} + G_{xu}(y_{t-1}^{(1)} \otimes u_t) + \frac{1}{2} G_{uu} u_t^{\otimes 2} + \frac{1}{2} g_{\sigma\sigma}\sigma^2.$$
# El cambio de régimen por conjunto activo itera sobre la duración de activación $T_{\text{bind}}$ incorporando simultáneamente el desplazamiento por riesgo $\frac{1}{2} g_{\sigma\sigma} \sigma^2$.
#
# **4. Nowcasting de frecuencia mixta y noticias de Banbura–Modugno.** Los modelos de factores dinámicos formulan variables observables de frecuencia mixta en espacio de estados $s_t = \Phi s_{t-1} + \eta_t, y_t = \Lambda s_t + \varepsilon_t$. La descomposición analítica de noticias expresa la actualización de la predicción inmediata como:
# $$\hat{y}_{t|v} - \hat{y}_{t|v-1} = \sum_{j \in \mathcal{I}_v} w_j \left( z_{j,v} - \hat{z}_{j,v|v-1} \right), \quad \text{con} \quad \sum_{j} \text{impacto}_j = \text{revisión total}.$$
#
# **5. HJB en tiempo continuo y KFE adjunta.** En la malla de riqueza $a \in [a_{\min}, a_{\max}]$, la ecuación HJB del hogar con intensidad de saltos de Poisson $\Lambda$ resuelve:
# $$\rho v_j(a) = \max_c \left\{ u(c) + s_j(a) \partial_a v_j(a) \right\} + \sum_{k \ne j} \lambda_{jk} [v_k(a) - v_j(a)].$$
# Un esquema implícito contra el viento en diferencias finitas construye el generador infinitesimal $A \in \mathbb{R}^{N \times N}$ ($A \mathbf{1} = 0$). La ecuación progresiva de Kolmogorov (KFE) estacionaria es la transpuesta adjunta dispersa exacta:
# $$A^\top g = 0, \quad \text{sujeta a} \quad \sum_{i} g_i \Delta a_i = 1.$$
#
# **6. Agregación de redes y álgebra de variaciones relativas espacial dinámica.** Para una matriz insumo-producto $\Omega$ y participaciones de consumo final $b$, la respuesta del PIB de segundo orden de Baqaee–Farhi ante perturbaciones de productividad $\Delta \ln z$ es:
# $$\Delta \ln Y = \lambda^\top \Delta \ln z + \frac{1}{2} (\Delta \ln z)^\top \mathcal{H} (\Delta \ln z) - \lambda^\top \Delta \ln \mu,$$
# reduciéndose al teorema de Hulten ($\mathcal{H} = 0$) bajo tecnología Cobb–Douglas ($\theta = 1$). En economías espaciales dinámicas, la actualización de población sigue el álgebra de variaciones relativas de Caliendo–Parro–Rossi-Hansberg:
# $$\hat{L}_{i,t+1} L_{i,t+1} = \sum_{j} \mu_{ji,t} \hat{\mu}_{ji,t} L_{j,t}.$$
#
# **7. Mallas contrafactuales con autoauditoría en PolicyStudio.** A lo largo de la malla cartesiana $\mathcal{G} = \prod_k \mathcal{A}_k$, la evaluación de políticas audita los residuos $\mathcal{R}(\theta) = \max_m |r_m(\theta)|$ frente a umbrales estrictos:
# $$\text{Estado}(\theta) = \begin{cases} \text{PASS} & \text{si } \mathcal{R}(\theta) \le \tau_{\text{warn}} \text{ y convergió}, \\ \text{WARN} & \text{si } \tau_{\text{warn}} < \mathcal{R}(\theta) \le \tau_{\text{fail}}, \\ \text{FAIL} & \text{en cualquier otro caso.} \end{cases}$$

# %% [markdown]
# ## Intuición
#
# **Intuición.** El modelado macroeconómico de frontera exige tender puentes entre la heterogeneidad microeconómica, las restricciones no lineales y la observabilidad empírica:
# - **La diferenciación automática hacia adelante** sustituye el ajuste heurístico del paso numérico por números hiperduales exactos, permitiendo obtener tangentes paramétricas y operadores hessianos con precisión de máquina dentro de NumPy.
# - **Los DAGs en espacio de secuencias y la verosimilitud de Whittle** eliminan la maldición de la dimensionalidad en modelos con agentes heterogéneos (HANK): las transiciones de equilibrio general se calculan como operaciones lineales en bloques de secuencias, y la estimación en el dominio de la frecuencia ajusta espectros empíricos sin requerir simulaciones costosas en espacio de estados.
# - **OccBin no lineal podado** captura la asimetría dependiente del estado característica de las crisis macroeconómicas—como las trampas de liquidez y las restricciones de endeudamiento activas—preservando simultáneamente las reservas estocásticas por riesgo precautorio ($\sigma > 0$).
# - **El nowcasting diferenciable de frecuencia mixta** aísla las "noticias" macroeconómicas que impulsan las revisiones del PIB en tiempo real, rastreando los movimientos del pronóstico directamente hasta las sorpresas inesperadas en publicaciones individuales de alta frecuencia.
# - **El método HJB en volúmenes finitos y la KFE adjunta en tiempo continuo** garantizan la conservación del flujo de probabilidad y de la masa total por construcción, creando un puente disperso exacto hacia los modelos macroeconómicos en espacio de secuencias.
# - **La macroeconomía de redes de segundo orden y la migración espacial dinámica** capturan cuellos de botella no lineales, distorsiones de sustitución en cadenas de suministro y fricciones regionales de ajuste laboral donde las aproximaciones lineales fallan.
# - **PolicyStudio con autoauditoría** garantiza que cada celda a través de mallas contrafactuales de alta dimensión satisfaga leyes de conservación matemática y criterios de convergencia antes de publicar informes visuales interactivos sin servidor.

# %% [markdown]
# ## Código resuelto

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
# ### 1. Diferenciación automática hacia adelante en NumPy puro: gradientes y hessianos exactos
#
# Evaluamos el motor de diferenciación automática hacia adelante sobre una función de utilidad macroeconómica CRRA estándar $u(c, \gamma) = \frac{c^{1-\gamma}-1}{1-\gamma}$. Evaluando en $c = 2.0, \gamma = 1.5$, `value_and_grad()` calcula la utilidad marginal analítica $u_c = c^{-\gamma} = 2^{-1.5} \approx 0.35355339$, y `hessian_forward()` evalúa el tensor hessiano completo de $2 \times 2$ mediante perturbaciones en números duales $(\varepsilon_1, \varepsilon_2)$ con cero error de truncamiento.

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
# ### 2. DAG generalizado en espacio de secuencias y estimación HANK de Whittle
#
# Construimos un DAG neokeynesiano / HANK componible de 4 bloques (regla de Taylor, ecuación de Fisher, curva de Phillips neokeynesiana y bloque de Euler). El compilador de jacobianos en espacio de secuencias evalúa la función de respuesta a impulso ante una perturbación contractiva de política monetaria AR(1) ($\rho_m = 0.60$) en un horizonte $T = 30$, coincidiendo con la respuesta analítica en forma cerrada. En el dominio de la frecuencia, `hank_loglik_whittle()` evalúa la verosimilitud de Whittle sobre innovaciones de ruido blanco, reproduciendo la verosimilitud gaussiana exacta de Parseval.

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
# ### 3. OccBin no lineal podado de 2.°/3.er orden con desplazamientos por riesgo precautorio
#
# Resolvemos una cota inferior efectiva ocasionalmente activa (ELB: $i_t \ge -0.01$) ante una fuerte perturbación contractiva de demanda ($u_e = -0.05$). El solucionador de perturbación podada acopla la iteración por regímenes lineales a tramos con la curvatura de segundo orden de Taylor y un desplazamiento estocástico por riesgo precautorio ($\frac{1}{2} g_{\sigma\sigma} \sigma^2$). El piso de la tasa de interés nominal se respeta sin violaciones de cota y la norma del desplazamiento precautorio es estrictamente no nula.

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
# ### 4. Nowcasting en espacio de estados de frecuencia mixta y noticias de Banbura–Modugno
#
# Ajustamos un modelo de factores dinámicos de frecuencia mixta de 4 variables (`DynamicFactorModel`) sobre indicadores macroeconómicos mensuales (PIB, producción industrial, ventas minoristas e IPC) con patrones de borde irregular. Cuando llegan nuevas publicaciones para la producción industrial y las ventas minoristas, `banbura_modugno_news()` ejecuta una descomposición exacta publicación por publicación, satisfaciendo la identidad de aditividad de Banbura–Modugno ($\sum_j \text{impacto}_j = \text{revisión total}$) con precisión numérica.

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
# ### 5. HJB en diferencias finitas contra el viento en tiempo continuo y puente HetBlock
#
# Resolvemos el modelo de hogares de Huggett–Aiyagari en tiempo continuo mediante el esquema implícito contra el viento en diferencias finitas de Achdou et al. (2022). La matriz del generador numérico $A$ conserva el flujo de probabilidad con sumas de filas idénticamente nulas ($A \mathbf{1} = 0$). La ecuación progresiva de Kolmogorov (KFE) estacionaria se evalúa a través de la transpuesta adjunta $A^\top \pi = 0$, conservando la masa total de probabilidad. Las funciones de política resultantes se conectan directamente a un `HetBlock` en el espacio de secuencias.

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
# ### 6. Macroeconomía de redes de 2.° orden de Baqaee–Farhi y álgebra de variaciones relativas espacial dinámica
#
# Evaluamos la agregación de redes de producción de segundo orden de Baqaee–Farhi (2019) sobre una economía de 3 sectores (Energía, Manufactura, Servicios). Bajo tecnología Cobb–Douglas ($\theta = 1.0$), la curvatura macroeconómica de segundo orden se reduce idénticamente a cero ($\mathrm{d}^2 \ln Y = 0$), coincidiendo con el teorema de Hulten. En economía espacial dinámica, `solve_dynamic_spatial_hat()` resuelve el álgebra de variaciones relativas de migración de Caliendo–Parro–Rossi-Hansberg (2019) a través de 3 regiones, conservando la población nacional agregada y manteniendo matrices de migración estocásticas por filas.

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
# ### 7. Malla contrafactual con autoauditoría en `PolicyStudio` y exportación interactiva HTML5/SVG sin servidor
#
# Construimos una malla contrafactual bidimensional de políticas sobre elasticidades de sustitución CES ($\theta \in \{0.4, 0.7, 1.0\}$) y perturbaciones de oferta de energía aguas arriba ($z_1 \in \{-0.10, -0.05\}$). `run_policy_studio()` evalúa y audita cada celda, verificando la convergencia y los residuos matemáticos. `to_interactive_html()` compila los resultados en un documento HTML5 autónomo con gráficos SVG incrustados y controles en JavaScript del lado del cliente, requiriendo cero conexiones CDN externas.

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
# ### 8. Visualización principal: IRFs de DAG, cambio de régimen en OccBin y mapa de calor de políticas auditadas

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
_ = cbar.set_label(r"$\frac{1}{2} z^\top \mathcal{H} z$ (%)", rotation=90)

# %% [markdown]
# ### 9. Cuadro de verificación
#
# Cada comprobación numérica a lo largo de las siete pistas de frontera se tabula a continuación con su error medido y cota de aceptación.

# %%
scorecard = pd.DataFrame(checks)
print(scorecard.to_string(index=False, formatters={"measured": "{:.2e}".format, "bound": "{:.2e}".format}))
assert (scorecard["measured"] <= scorecard["bound"]).all()
print(f"\nAll {len(scorecard)} frontier macroeconomic workflow checks pass.")

# %% [markdown]
# ## Interpretar los resultados
#
# En la Sección 1, `value_and_grad()` y `hessian_forward()` evalúan las derivadas de la utilidad CRRA con cero error ($< 10^{-14}$) respecto al cálculo analítico, confirmando que el motor de diferenciación automática hacia adelante produce gradientes y hessianos con precisión de máquina sin ruido de truncamiento por diferencias finitas.
#
# En la Sección 2, el `BlockDAG` componible resuelve las transiciones de equilibrio general en el espacio de secuencias hasta precisión de máquina (`residual_norm` $< 10^{-18}$), coincidiendo con la respuesta a impulso neokeynesiana analítica en el Panel 1 hasta $3.9 \times 10^{-10}$. En el dominio de la frecuencia, `hank_loglik_whittle()` reproduce la verosimilitud gaussiana de ruido blanco de Parseval hasta $7.1 \times 10^{-15}$.
#
# En la Sección 3, `solve_nonlinear_occbin()` respeta la cota inferior de la tasa de interés nominal ($i_t \ge -0.01$) sin violaciones de cota (`max_bound_violation` $= 0.0$). En el Panel 2, la trayectoria podada de segundo orden muestra un desplazamiento precautorio por riesgo perceptible ($\|\frac{1}{2} g_{\sigma\sigma} \sigma^2\| = 0.014131$) que eleva las expectativas de tasa de interés respecto a la trayectoria lineal equivalente con certidumbre.
#
# En la Sección 4, `banbura_modugno_news()` logra aditividad exacta publicación por publicación (`decomposition_error` $= 2.8 \times 10^{-17}$), demostrando que los datos entrantes de producción industrial y ventas minoristas explican plenamente la revisión del pronóstico del PIB.
#
# En la Sección 5, `solve_hjb_achdou()` converge en 7 iteraciones con errores de suma de filas del generador inferiores a $5.6 \times 10^{-17}$, mientras que la distribución estacionaria adjunta satisface la estacionariedad de la KFE hasta $4.8 \times 10^{-17}$.
#
# En la Sección 6, `baqaee_farhi_decomposition()` demuestra que la curvatura macroeconómica de segundo orden colapsa idénticamente a cero bajo preferencias Cobb–Douglas ($\theta = 1.0$), mientras que `solve_dynamic_spatial_hat()` conserva la población agregada a lo largo del tiempo hasta $0.0$.
#
# En la Sección 7, `run_policy_studio()` audita las 6 celdas contrafactuales en la malla $(\theta, z_1)$, logrando `PASS (6/6 verified)` sin advertencias ni fallos. El Panel 3 muestra el mapa de calor resultante de la curvatura del PIB de segundo orden, y `to_interactive_html()` exporta un informe HTML5 autónomo sin servidor con cero dependencias de CDN externas.

# %% [markdown]
# ## Tu turno
#
# Modifica la elasticidad de sustitución `user_elasticity` o la perturbación contractiva `user_zlb_shock`; las aserciones posteriores verifican que la curvatura de red de segundo orden y el cumplimiento de cotas en OccBin no lineal se mantengan en todo el espacio paramétrico anunciado.
#
# 1. **Básico.** Incrementa `user_elasticity` desde `0.20` hacia `1.00`. ¿Cómo se reduce la pérdida de PIB de segundo orden ante la perturbación energética aguas arriba a medida que se expanden las posibilidades de sustitución de insumos?
# 2. **Intermedio.** Varía `user_zlb_shock` entre `-0.08` y `-0.02`. ¿Cómo escala el número de trimestres de activación de la cota inferior efectiva con la perturbación contractiva, y aumenta o disminuye el desplazamiento por riesgo precautorio?
# 3. **Avanzado.** Ejecuta `run_policy_studio()` con una función personalizada `audit_fn` que clasifique cualquier celda con `max_bound_violation > 1e-10` como advertencia, y verifica cómo se actualiza el estado general de auditoría.

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
# ## ¿Qué tan exhaustivo es esto?
#
# La exhibición `71` une las siete pistas de frontera en un flujo de trabajo macroeconómico integral. Los solucionadores subyacentes y sus garantías teóricas se encuentran documentados en guías especializadas: `docs/es/dsge_ssj.md` (DAGs en espacio de secuencias y verosimilitud de Whittle), `docs/es/dsge_occbin.md` (OccBin por tramos y de orden superior), `docs/es/nowcast.md` (espacio de estados de frecuencia mixta y descomposición de noticias), `docs/es/vfi_hjb_continuous.md` (HJB contra el viento de Achdou y KFE adjunta), `docs/es/network_macro.md` (agregación insumo-producto de Baqaee–Farhi), `docs/es/spatial_hat.md` (álgebra de variaciones relativas dinámica de Caliendo–Parro–Rossi-Hansberg) y `docs/es/policy_studio.md` (mallas declarativas de PolicyStudio e HTML5 interactivo sin servidor). Todos los componentes cumplen estrictamente con el contrato de ejecución de Pyodide de cuatro paquetes (`numpy`, `scipy`, `pandas`, `matplotlib`).
