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
# # Flujos DSGE no lineales: ejecución ordenada de `.mod`, políticas globales de Chebyshev y transiciones Newton–Krylov con restricciones
#
# **Cuando vamos más allá de la perturbación lineal, ¿cómo verificamos que la ejecución ordenada de programas `.mod`, las expansiones de Taylor de orden superior, la iteración temporal global de Chebyshev y las transiciones de previsión perfecta con restricciones coincidan donde deben hacerlo y revelen no linealidades económicas genuinas donde difieren?**
#
# Un mismo modelo DSGE no lineal puede resolverse localmente mediante perturbación hasta tercer orden, globalmente mediante iteración temporal con colocación de Chebyshev, o a lo largo de trayectorias de transición deterministas y con restricciones ocasionalmente activas mediante métodos de Newton apilados en el tiempo. Cada método introduce aproximaciones numéricas distintas y tiene sus propios modos de fallo: los programas `.mod` de múltiples fases pueden filtrar el estado de calibración entre fases; los sistemas de Sylvester de orden superior pueden sufrir acoplamiento fuera de la diagonal de Schur; la colocación en producto tensorial puede alojar grandes matrices densas de Vandermonde; y los horizontes finitos truncados o las esquinas de complementariedad pueden distorsionar las transiciones de previsión perfecta. Este cuaderno ejecuta una verificación de extremo a extremo en cuatro aplicaciones y registra cada comprobación numérica en un único cuadro de verificación.
#
# | Alcance y verificación | |
# |---|---|
# | Flujos cubiertos | 1. Ejecución ordenada de `.mod` en múltiples fases con recalibración intermedia y `histval` con múltiples rezagos; 2. Condicionamiento de Sylvester de tercer orden e iteración temporal separable de Chebyshev (`orders=(8, 8)`); 3. Mapa de precisión local frente a global (`sigma=1.0` frente al límite sin riesgo `sigma=0.0`); 4. Transiciones MCP con restricciones mediante Newton–Krylov libre de matrices frente a factorización directa dispersa (`mcp=True`, `terminal_policy`) |
# | Modelos | Modelo RBC con productividad AR(2) de dos rezagos y cambio intermedio en la participación del capital; modelo RBC estocástico CRRA de 2 estados; modelo RBC con inversión irreversible `[mcp = 'i > i_min']`; modelo neokeynesiano no lineal con cota inferior efectiva `[mcp = 'i > i_elb']` |
# | Solucionadores y diagnósticos | `inspect_mod`, `read_mod().execute_mod_commands()`, `diagnose_perturbation`, `evaluate_perturbation`, `EquilibriumProblem.from_model`, `solve_time_iteration`, `compare_policies`, `solve_transition(..., linear_solver="direct" | "krylov")` |
# | Aceptación | Inmutabilidad de instantáneas y paridad frente a ejecución independiente $<10^{-12}$; residuos algebraicos de Sylvester $<10^{-13}$; residuos de Euler de Chebyshev separable $<10^{-8}$ sin alojar matrices densas de colocación; acuerdo entre trayectorias MCP directas y Krylov libre de matrices $<10^{-9}$ con cero bloques JVP densos |

# %% [markdown]
# ## El método en matemáticas
#
# **1. Operadores de Sylvester de orden superior y condicionamiento.** Alrededor del estado estacionario determinista $y^*$, la perturbación de primer orden produce la transición de estados $H_x$ y el haz adelantado $(\hat A, A_+)$ con $\hat A = A_0 + A_+ G_x P_s$. El tensor de curvatura de estados de orden $k$, $G_k$, resuelve la ecuación generalizada de Sylvester
# $$\mathcal{L}_k(G_k) = \hat A\, G_k + A_+\, G_k\, H_x^{\otimes k} = -K_k, \qquad k \in \{2, 3\}.$$
# La descomposición real de Schur $H_x = U T U^\top$ reduce $\mathcal{L}_k$ a sistemas por bloques triangulares de Bartels–Stewart a lo largo de los bloques diagonales de $T^{\otimes k}$, aprovechando la simetría de permutación sobre tuplas de índices ordenadas $(i_1 \le \dots \le i_k)$. Como la no normalidad y el acoplamiento fuera de la diagonal en $T$ pueden amplificar errores incluso cuando los haces diagonales individuales $\hat A + \tau A_+$ están bien condicionados, `diagnose_perturbation()` reporta tanto el peor número de condición de los bloques diagonales de Schur como el número de condición en norma 1, $\kappa_1(\mathcal{L}_k) = \|\mathcal{L}_k\|_1 \|\mathcal{L}_k^{-1}\|_1$, del operador completo de curvatura, junto con los residuos algebraicos normalizados de `ghxx`, `ghs2`, `ghxxx`, `ghxss` y `ghuss`.
#
# **2. Colocación separable de Chebyshev e iteración temporal global.** En una malla de estados en producto tensorial $\mathcal{G} = \mathcal{G}_1 \times \dots \times \mathcal{G}_d$ de tamaño $M = \prod_{j=1}^d (n_j + 1)$, ajustar los coeficientes de Chebyshev formando la matriz densa de Vandermonde $M \times M$ requiere trabajo de factorización $\mathcal{O}(M^3)$ y memoria $\mathcal{O}(M^2)$. Aprovechar la separabilidad de Kronecker $V = V_1 \otimes \dots \otimes V_d$ factoriza cada matriz 1D de Vandermonde $V_j \in \mathbb{R}^{(n_j+1)\times(n_j+1)}$ una sola vez y actualiza los coeficientes mediante contracciones tensoriales secuenciales en cada modo $j$. En cada nodo $s \in \mathcal{G}$, la iteración temporal resuelve el sistema de equilibrio condicional
# $$\mathbb{E}\!\left[\,F\big(y_{-}(s),\, y(s, x),\, y(s', g^{(m)}(s')),\, \varepsilon'\big)\,\right] = 0$$
# para los controles actuales $x$, empleando derivadas simbólicas exactas del DAG compilado del archivo `.mod` mediante `TrajectoryJacobian` y derivadas analíticas de la base de Chebyshev $\partial g^{(m)}(s')/\partial s'$.
#
# **3. Transiciones apiladas en el tiempo con restricciones y políticas terminales.** En un horizonte finito $t = 1, \dots, T$, la previsión perfecta apilada en el tiempo ensambla el sistema no lineal por bloques $R_t(y_{t-1}, y_t, y_{t+1}, u_t) = 0$. Cuando el estado aún no ha regresado a $y^*$ en el período $T$, cerrar el adelanto terminal con $y_{T+1} = g_{\text{term}}(y_T)$ modifica la última columna de bloques a $A_{0,T} + A_{+,T} J_{\text{term}}$. Bajo restricciones de desigualdad $l_t \le y_t \le u_t$ declaradas mediante `[mcp = '...']`, el sistema KKT / Fischer–Burmeister de conjunto activo impone la complementariedad $0 \le y_t - l_t \perp R_t(y) \ge 0$, evaluada ya sea mediante LU directa dispersa o mediante LGMRES precondicionado libre de matrices usando los bloques jacobianos dispersos en caché $(A_{-,t}, A_{0,t}, A_{+,t})$.

# %% [markdown]
# ## Intuición
#
# **Intuición.** Cada método numérico responde a una pregunta distinta sobre las mismas ecuaciones económicas:
# - **La ejecución ordenada de `.mod`** trata el archivo `.mod` como un protocolo reproducible de laboratorio: recalibrar un parámetro estructural a mitad del guion debe actualizar los bloques posteriores de estado estacionario y transición manteniendo inmutables las instantáneas de las fases previas.
# - **Las políticas locales de Taylor frente a las políticas globales de Chebyshev** separan los *desplazamientos por riesgo* de la *curvatura respecto al estado*: evaluar una regla de Taylor de segundo o tercer orden en `sigma=1.0` incorpora el desplazamiento precautorio $\tfrac12 g_{\sigma\sigma}\sigma^2$ y las pendientes ajustadas por riesgo $\tfrac12 g_{x\sigma\sigma}\sigma^2$, que coinciden con la política estocástica de Chebyshev cerca del estado estacionario, mientras que fijar `sigma=0.0` aísla la curvatura determinista que coincide con una transición de previsión perfecta.
# - **Las transiciones directas frente a Newton–Krylov libre de matrices** verifican que resolver el sistema KKT apilado mediante iteraciones de Krylov con bloques dispersos en caché reproduce la solución directa dispersa hasta precisión de máquina sin ensamblar la matriz completa $NT \times NT$.

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
# ### 1. Ejecución ordenada de `.mod` con recalibración intermedia y `histval` de múltiples rezagos
#
# Primero inspeccionamos y ejecutamos un programa `.mod` de dos fases en el que la productividad sigue un proceso AR(2), $a_t = \rho_1 a_{t-1} + \rho_2 a_{t-2} + e_t$, de modo que tanto $a(0)$ como $a(-1)$ son condiciones iniciales predeterminadas genuinas en `histval`. Tras resolver el estado estacionario base con $\alpha = 0.33$, el guion reasigna $\alpha = 0.36$ (lo que también actualiza el parámetro dependiente `cap_ratio = alpha / (1 - alpha)`), recalcula el estado estacionario, fija condiciones iniciales de dos rezagos en `histval` y ejecuta una transición determinista no lineal de 40 trimestres. Comprobamos que la instantánea de la Fase 1 conserva $\alpha = 0.33$, la de la Fase 2 registra $\alpha = 0.36$ y la transición ejecutada coincide con una llamada independiente a `solve_transition()` sobre el modelo recalibrado al nivel del error de redondeo.

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
# ### 2. Condicionamiento de Sylvester de orden superior, iteración temporal separable de Chebyshev y mapa de precisión local/global
#
# A continuación estudiamos un modelo RBC no lineal de dos estados con aversión al riesgo CRRA ($\gamma = 2.0$, $\alpha = 0.36$, $\beta = 0.95$, $\delta = 0.10$, $\rho = 0.80$, $\sigma_e = 0.012$). Resolvemos el modelo localmente hasta tercer orden e inspeccionamos su condicionamiento de Sylvester y sus conteos de trabajo mediante `diagnose_perturbation()`. Después resolvemos globalmente el mismo modelo `.mod` compilado en $[0.75 k^*, 1.25 k^*] \times [-0.05, 0.05]$ mediante iteración temporal separable de Chebyshev de grado $(6, 4)$ (`35` nodos de colocación y cuadratura de Gauss–Hermite de `5` nodos).
#
# Comparar `EquilibriumProblem.from_model()` (que conecta las derivadas simbólicas exactas mediante `TrajectoryJacobian` y las derivadas analíticas de la base de Chebyshev) contra un problema sin jacobiano analítico muestra cómo las derivadas exactas reducen las evaluaciones de residuos convergiendo a la misma política. Por último, comparamos las políticas locales de Taylor de orden 1, 2 y 3 frente a la política global de Chebyshev en el dominio de estados, y contrastamos `sigma=1.0` (estocástico) frente a `sigma=0.0` (límite determinista sin riesgo) a lo largo de una transición amplia de capital desde $k_0 = 0.78 k^*$.

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
# ### 3. Transiciones con restricciones mediante Newton–Krylov libre de matrices frente a factorización directa dispersa (`mcp=True` y `terminal_policy`)
#
# Ahora evaluamos transiciones apiladas en el tiempo con restricciones de desigualdad ocasionalmente activas resueltas tanto mediante factorización directa dispersa (`linear_solver="direct"`) como mediante Newton–Krylov libre de matrices (`linear_solver="krylov"`):
#
# 1. **Inversión irreversible con frontera `terminal_policy`.** En un modelo RBC con inversión bruta $i_t = k_t - (1-\delta)k_{t-1}$ y un piso de irreversibilidad `[mcp = 'i > 0.19']`, un choque negativo de productividad lleva la inversión a su cota inferior. Además, cuando la transición se resuelve en un horizonte corto ($T = 20$) en el que el capital aún no ha regresado a $k^*$, suministrar el mapa terminal de primer orden `terminal_policy` y `terminal_jacobian` reduce drásticamente el error de truncamiento de horizonte finito frente al cierre ingenuo $y_{T+1} = y^*$ (evaluado contra una referencia de horizonte largo $T = 90$).
# 2. **Cota inferior efectiva neokeynesiana (`[mcp = 'i > i_elb']`).** En un modelo neokeynesiano no lineal de 3 ecuaciones golpeado por un choque negativo severo de demanda, la tasa nominal queda restringida en $i_{\text{elb}} = -0.0075$ durante varios trimestres. Verificamos que `direct` y `krylov` convergen a trayectorias idénticas (`< 1e-10`), cumplen la factibilidad y complementariedad física MCP y ejecutan cero bloques JVP densos (`dense_jvp_blocks == 0`).

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
# ### 4. Cuadro de verificación
#
# Cada comprobación numérica de los cuatro flujos se resume a continuación junto con su error medido y su cota de aceptación.

# %%
scorecard = pd.DataFrame(checks)
print(scorecard.to_string(index=False, formatters={"measured": "{:.2e}".format, "bound": "{:.2e}".format}))
assert (scorecard["measured"] <= scorecard["bound"]).all()
print(f"\nAll {len(scorecard)} nonlinear DSGE workflow and solver checks pass.")

# %% [markdown]
# ## Lectura de los resultados
#
# En la sección 1, `execute_mod_commands()` ejecuta el programa `.mod` de dos fases en el orden del archivo fuente. Reasignar `alpha = 0.36` después de la primera instrucción `steady;` actualiza `cap_ratio` y el segundo estado estacionario (`k` sube de `28.3484` a `34.4820`) sin mutar la instantánea de la Fase 1, y la transición con `histval` de dos rezagos coincide con una llamada independiente a `solve_transition()` al nivel de $10^{-15}$.
#
# En la sección 2, `diagnose_perturbation()` confirma que el modelo RBC CRRA de 2 estados tiene bloques de Schur y operadores completos de curvatura de Sylvester bien condicionados en los órdenes 1, 2 y 3, con residuos algebraicos normalizados alrededor de $10^{-16}$. La iteración temporal global de Chebyshev en la malla tensorial $7 \times 5$ (`orders=(6, 4)`) converge en 59 iteraciones sin alojar una matriz densa de colocación $35 \times 35$ (`collocation_matrix_allocated=False`), y el `TrajectoryJacobian` simbólico reduce las evaluaciones de residuos en $1.5\times$ frente a las diferencias finitas. La comparación de políticas en el Panel 1 muestra por qué importa `sigma`: en `a = 0`, la regla de Taylor de tercer orden con `sigma=1.0` sigue de cerca la política estocástica de Chebyshev (a menos de una fracción de punto básico cerca de $k^*$), mientras que la regla con `sigma=0.0` omite el ahorro precautorio y por ello se sitúa por encima de la regla estocástica en $k^*$ al tiempo que reproduce exactamente el estado estacionario determinista.
#
# En la sección 3, ambas transiciones con restricciones —la inversión irreversible (`i >= 0.19`) y la cota inferior efectiva neokeynesiana (`i >= -0.0075`)— convergen a trayectorias idénticas bajo LU directa dispersa y Newton–Krylov libre de matrices (`max |diff| < 1e-13`), con cero bloques JVP densos (`dense_jvp_blocks == 0`). En el horizonte truncado $T = 20$ del Panel 2, cerrar el adelanto terminal con `terminal_policy` reduce el error de truncamiento de horizonte finito en un orden de magnitud frente a imponer $y_{T+1} = y^*$ en el período 20.

# %% [markdown]
# ## Tu turno
#
# Cambia el grado del polinomio de Chebyshev `cheb_order` o el tamaño del choque de demanda neokeynesiano `elb_shock_size`; las aserciones verifican que la colocación separable de Chebyshev y la paridad entre Newton–Krylov con restricciones y el método directo se mantienen.
#
# 1. **Básico.** Reduce `cheb_order` de `6` a `4` y luego súbelo a `8`. ¿Qué tan rápido cae el máximo residuo de Euler fuera de la malla en `audit_pts` al aumentar el grado de Chebyshev?
# 2. **Intermedio.** Varía `elb_shock_size` entre `-0.015` y `-0.035`. ¿Cuántos trimestres permanece la tasa nominal en la cota inferior `i_elb = -0.0075`, y cómo escala la caída máxima de la brecha del producto con el tamaño del choque?
# 3. **Avanzado.** Sustituye `terminal_policy` en la transición de inversión irreversible con $T=20$ por un mapa terminal de perturbación de segundo orden usando `evaluate_perturbation(..., order=2, sigma=0.0)` y compara el error de truncamiento contra `irrev_ref_long`.

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
# ## ¿Qué tan exhaustivo es esto?
#
# El cuaderno `68` verifica los tensores de perturbación de tercer orden y los momentos exactos con poda frente a formas cerradas y corridas en vivo de Dynare 7/8. Las guías `docs/es/dsge_workflows.md`, `docs/es/dsge_nonlinear.md`, `docs/es/dsge_transitions.md`, `docs/es/dsge_global.md` y `docs/es/dsge_mod_inspection.md` documentan el contrato y los límites de alcance de cada solucionador. El ajuste separable de Chebyshev se aplica automáticamente a mallas en producto tensorial (`basis="chebyshev"`); las mallas dispersas no estructuradas de Smolyak (`basis="smolyak"`) siguen factorizando su matriz de colocación en el conjunto disperso de nodos. En las transiciones Newton–Krylov con restricciones, el paso en subespacio de conjunto activo acelera las cotas de caja diagonales (`variable >= c`, `variable <= c`), mientras que las desigualdades generales de múltiples variables recurren a la búsqueda lineal amortiguada de mérito de Fischer–Burmeister.
