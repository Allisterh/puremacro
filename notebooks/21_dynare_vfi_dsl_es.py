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
# # VFI Automatizado estilo Dynare — Programación Dinámica Estructural Declarativa
#
# **¿Podemos escribir modelos de programación dinámica estructural en sintaxis declarativa sin bucles de malla de bajo nivel?** En macroeconomía, Dynare transformó la modelización DSGE reemplazando derivaciones matriciales manuales con un lenguaje declarativo limpio. `puremacro.dp` lleva la misma idea a la **programación dinámica con agentes heterogéneos**: declaras parámetros, una malla de estado, un proceso de choques, la restricción presupuestaria y la recompensa como ecuaciones en sintaxis de expresiones de Dynare, y `Model.solve()` escribe la llamada al resolvedor. No hay bucles de malla, funciones de retorno ni matrices de Tauchen que gestionar. Este cuaderno construye un modelo de ahorro precautorio en mercados incompletos, lo resuelve por iteración de la función de valor, lee las funciones de política y de valor, y mide la desigualdad de riqueza de forma exacta a partir de la distribución estacionaria y simulando un panel de hogares.

# %% [markdown]
# ## El método en matemáticas — la ecuación de Bellman
#
# El problema de un solo agente sobre el estado $(a, z)$ resuelve:
# $$ V(a, z) = \max_{a' \in \mathcal{A}} \left\{ u(c) + \beta \sum_{z'} P(z, z') V(a', z') \right\} $$
# sujeto a la restricción presupuestaria:
# $$ c + a' = e^{z} a^{\alpha} + (1-\delta) a, \qquad c > 0, $$
# donde $u(c) = \frac{c^{1-\gamma}}{1-\gamma}$, $z$ sigue un proceso AR(1) $z_t = \rho z_{t-1} + \varepsilon_t$, y $P(z, z')$ es la matriz de transición de Markov de su discretización de Tauchen.
#
# **Intuición.** Cada línea de la declaración de abajo es una pieza de este problema. `puremacro.dp` analiza las ecuaciones, discretiza el choque, evalúa la recompensa en la malla $\mathcal{A} \times \mathcal{Z}$ y llama al motor de `puremacro.vfi`. La VFI discreta mantiene $a'$ en los nodos de la malla. El término $e^z a^\alpha$ no tiene la forma $R a + y(z)$, así que el método de la malla endógena no se aplica y resolvemos con `method="vfi"`.

# %% [markdown]
# ## Configuración — especificación declarativa del modelo
#
# Creamos `dp.Model()`, fijamos los parámetros ($\beta=0.96, \gamma=2.0, \alpha=0.36, \delta=0.10$), definimos una malla de activos de 60 puntos en $a \in [0.1, 15.0]$ y declaramos un choque de productividad AR(1) $z$ con 5 estados de Tauchen.

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

# Declarar el modelo estructural
model = dp.Model("Precautionary Savings Model")
model.parameters(beta=0.96, gamma=2.0, alpha=0.36, delta=0.10)

# Proceso de choques y malla de estado
model.exogenous("z", dp.AR1(rho=0.90, sigma=0.15, n=5))
model.state("a", np.linspace(0.1, 15.0, 60))

# Restricción presupuestaria y utilidad, como ecuaciones
model.local("c = exp(z)*a^alpha + (1 - delta)*a - a(+1)")
model.reward("c^(1 - gamma)/(1 - gamma)")
model.subject_to("c > 0")

# %% [markdown]
# ## Resolviendo el modelo y midiendo la desigualdad de riqueza
#
# `model.solve(method="vfi")` itera sobre la ecuación de Bellman hasta que la función de valor cambia menos que `tol` y después calcula la distribución estacionaria de los hogares sobre la malla $(a, z)$. `sol.mean("a")` y `sol.gini("a")` se calculan de forma exacta a partir de esa distribución, sin simulación. El Gini es el Gini ponderado estándar de los activos, $1 - \sum_i w_i (L_i + L_{i-1})$ sobre la curva de Lorenz de los valores de la malla $a_i$ ponderados por su masa estacionaria $w_i$.

# %%
sol = model.solve(method="vfi", tol=1e-6)

V = sol.V
a_grid = sol.grids["a"]
policy_a = sol.policy("a(+1)")
mean_assets = sol.mean("a")
gini_coef = sol.gini("a")
top_mass = sol.distribution[-1].sum()

print(f"Modelo: {sol.model_name} (método: {sol.method})")
print(f"Activos Medios Estacionarios: {mean_assets:.4f}")
print(f"Coeficiente de Gini de Riqueza: {gini_coef:.4f}")
print(f"Masa en el punto superior de la malla a = 15: {top_mass:.4f}")

# Afirmaciones clave
assert abs(mean_assets - 6.5733) < 1e-3, f"Los activos medios cambiaron: {mean_assets}"
assert abs(gini_coef - 0.3494) < 1e-3, f"El coeficiente de Gini cambió: {gini_coef}"

# %% [markdown]
# Los activos medios son 6.5733 y el Gini es 0.3494. Versiones anteriores de este cuaderno imprimían 6.6767 y 0.3924. Esos valores venían de una simulación de 2000 hogares y 100 periodos, y ese Gini se calculaba después de restar la tenencia mínima de activos, así que no era el Gini estándar. Alrededor del 8% de la masa está en el punto superior de la malla, así que el extremo superior de la malla es una restricción activa: amplía la malla antes de interpretar la parte alta de la distribución.

# %% [markdown]
# ## Visualizando funciones de política y funciones de valor
#
# La figura principal muestra las reglas de política $a'(a, z)$ para los estados de productividad más bajo y más alto, y las funciones de valor.

# %%
fig, (ax1, ax2) = _nbstyle.figura(1, 2, figsize=(10.5, 4.2))

# Panel izquierdo: Funciones de política
ax1.plot(a_grid, a_grid, linestyle="--", color=_nbstyle.SPINE, alpha=0.7, label="Línea de 45° (a' = a)")
ax1.plot(a_grid, policy_a[:, 0], color=_nbstyle.palette(2)[0], lw=2, label="Baja Productividad (z_min)")
ax1.plot(a_grid, policy_a[:, -1], color=_nbstyle.palette(2)[1], lw=2, label="Alta Productividad (z_max)")
ax1.set_xlabel("Activos Actuales a", color=_nbstyle.TEXTO)
ax1.set_ylabel("Activos Siguiente Periodo a'", color=_nbstyle.TEXTO)
ax1.set_title("Funciones de Política Óptimas a'(a, z)")
ax1.legend(loc="upper left", frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)

# Panel derecho: Función de valor
ax2.plot(a_grid, V[:, 0], color=_nbstyle.palette(2)[0], lw=2, label="V(a, z_min)")
ax2.plot(a_grid, V[:, -1], color=_nbstyle.palette(2)[1], lw=2, label="V(a, z_max)")
ax2.set_xlabel("Activos a", color=_nbstyle.TEXTO)
ax2.set_ylabel("Valor V(a, z)", color=_nbstyle.TEXTO)
ax2.set_title("Funciones de Valor V(a, z)")
ax2.legend(loc="lower right", frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)

# %% [markdown]
# **Leyendo el resultado.** Los agentes de baja productividad ($z_{\min}$) desacumulan activos cuando la riqueza es alta, mientras que los agentes de alta productividad ($z_{\max}$) ahorran. Los motivos precautorios evitan que la riqueza colapse al extremo inferior de la malla.

# %% [markdown]
# ## Simulando un panel de hogares
#
# `sol.simulate(n_agents, periods)` genera historias individuales bajo la política resuelta y devuelve un `dp.Panel`. Aquí todos los hogares empiezan en el extremo inferior de la malla, $a = 0.1$, con su choque extraído de la distribución estacionaria de $z$. El Gini de corte transversal del panel, `panel.gini("a", by="t")`, primero sube mientras los hogares con suerte acumulan y después se estabiliza en el valor estacionario exacto. La misma semilla da el mismo panel.

# %%
panel = sol.simulate(n_agents=5000, periods=200, seed=0, init={"a": 0.1})
gini_t = panel.gini("a", by="t")

print(panel)
print(f"Gini del panel en t = 0, 10, 199: {gini_t.iloc[0]:.4f}, {gini_t.iloc[10]:.4f}, {gini_t.iloc[-1]:.4f}")
print(f"Gini estacionario exacto:         {gini_coef:.4f}")

assert abs(gini_t.iloc[-1] - gini_coef) < 0.02, "El Gini del panel debe estabilizarse cerca del valor exacto"

fig, ax = _nbstyle.figura(1, 1, figsize=(8.0, 3.6))
ax.plot(gini_t.index, gini_t.to_numpy(), color=_nbstyle.palette(2)[0], lw=2, label="Gini de a en el panel, por periodo")
ax.axhline(gini_coef, linestyle="--", color=_nbstyle.palette(2)[1], lw=1.5, label=f"Gini estacionario exacto ({gini_coef:.4f})")
ax.set_xlabel("Periodo t", color=_nbstyle.TEXTO)
ax.set_ylabel("Gini de los activos", color=_nbstyle.TEXTO)
ax.set_title("Un panel simulado converge a la distribución estacionaria")
ax.legend(loc="lower right", frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)

# %% [markdown]
# ## Tu turno — ahorro precautorio bajo mayor aversión al riesgo
#
# **Ejercicios.**
# 1. *Básico*: Incrementa la aversión al riesgo de $\gamma = 2.0$ a $\gamma = 3.5$ (`# ← cambia esto ...`). ¿Aumenta la riqueza media estacionaria debido a motivos precautorios más fuertes?
# 2. *Intermedio*: Reduce el factor de descuento a $\beta = 0.94$. Verifica que los hogares impacientes mantienen menos capital.
# 3. *Avanzado*: Incrementa la desviación típica del choque a $\sigma_z = 0.25$ y compara `sol_yt.gini("a")` con el caso base.

# %%
# Bloque de código Tu turno
gamma_new = 3.0  # ← cambia esto: prueba 3.0 o 3.5

m_yt = dp.Model("Alta Aversión al Riesgo")
m_yt.parameters(beta=0.96, gamma=gamma_new, alpha=0.36, delta=0.10)
m_yt.exogenous("z", dp.AR1(rho=0.90, sigma=0.15, n=5))
m_yt.state("a", np.linspace(0.1, 15.0, 60))
m_yt.local("c = exp(z)*a^alpha + (1 - delta)*a - a(+1)")
m_yt.reward("c^(1 - gamma)/(1 - gamma)")
m_yt.subject_to("c > 0")

sol_yt = m_yt.solve(method="vfi")
print(f"Nueva Aversión al Riesgo gamma = {gamma_new:.1f}")
print(f"Nuevos Activos Medios Estacionarios: {sol_yt.mean('a'):.4f}")
print(f"Nuevo Gini de Riqueza: {sol_yt.gini('a'):.4f}")

# Afirmación válida para el valor por defecto
assert sol_yt.mean("a") > 0.0, "Los activos medios deben ser positivos"

# %% [markdown]
# ## ¿Qué tan exhaustivo es esto?
#
# `puremacro.dp` compila la declaración a los motores de `puremacro.vfi`; `sol.raw` contiene el resultado del motor. El mismo estilo declarativo cubre ciclos de vida, decisiones discretas con choques de preferencia, el método de la malla endógena, tiempo continuo y precios de equilibrio de mercado (véase `docs/es/dp.md`). `sol.top_share`, `sol.quantile` y `sol.lorenz` dan el resto de la distribución exacta de la riqueza. Este cuaderno usaba antes `puremacro.vfi.Model`; esa clase está obsoleta y se eliminará en la versión 6.0.
