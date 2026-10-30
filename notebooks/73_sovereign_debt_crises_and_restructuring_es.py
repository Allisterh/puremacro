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
# # Deuda soberana, restricciones de colateral y reestructuración
#
# Este laboratorio explora deuda soberana a corto y largo plazo, colateral,
# contabilidad de diferenciales y escenarios de reestructuración. Son ejemplos
# numéricos, no reproducciones de las calibraciones publicadas ni de los momentos
# empíricos de Arellano, Chatterjee–Eyigungor o Mendoza.
#
# | Cálculo | Qué examinar |
# |---|---|
# | Impago a corto plazo | Residuos, cotas de precios y estadísticas de malla frente a simuladas |
# | Deuda a largo plazo | Curvas de precios y derivadas de sus interpolantes |
# | Ejemplo de colateral | Políticas sobre la malla y diagnósticos experimentales |
# | Reestructuración | Cambios con signo en un estado dado, preservando la calibración |
#
# La referencia independiente de corto plazo determinista está documentada en
# `docs/es/research_reliability.md`. No certifica todas las extensiones siguientes.

# %% [markdown]
# ## El método en ecuaciones
#
# Con deuda positiva $b$, el reembolso a corto plazo resuelve
#
# $$v^R(b,y)=\max_{b'}\{u(y-b+q(b',y)b')+\beta E[v(b',y')\mid y]\}.$$
#
# El impago produce $h(y)$ y el reingreso ocurre con deuda cero:
#
# $$v^D(y)=u(h(y))+\beta E[\theta v(0,y')+(1-\theta)v^D(y')\mid y].$$
#
# La decisión determinista compara los dos valores. `sigma_eps=0` selecciona ese
# caso; una dispersión positiva produce un valor log-sum-exp y probabilidades
# logísticas de impago. Estos ejemplos usan dispersión positiva y la función de
# costos de Chatterjee–Eyigungor predeterminada.
#
# Los bonos largos tienen tasa de vencimiento $\delta$ y cupón $\kappa$. Su precio
# sin riesgo es $[\delta+(1-\delta)\kappa]/(r^*+\delta)$, distinto del precio a un
# período. `dq_db` deriva el interpolante de precios ajustado, no todo el equilibrio
# respecto de un parámetro. La proyección predeterminada impone monotonicidad.
#
# La deuda larga usa una política probabilística suavizada cuyo valor es el
# objetivo ponderado, no un valor inclusivo. Simulación y valoración usan las
# mismas probabilidades; la política de deuda mostrada es su moda.
# El colateral impone $b'\leq\kappa q k'$. La amplificación divide caídas medias
# de consumo en los mismos estados simulados de deuda y capital. Puede ser menor
# que uno o indefinida. `risk_premium_component` es un ajuste de capitalización
# discreta de derechos de supervivencia, no una prima económica identificada.

# %% [markdown]
# ## Intuición
#
# El impago intercambia alivio de deuda por costos de producto y exclusión. Su
# frecuencia y la ciclicidad de diferenciales dependen de la calibración, la malla
# y los estados visitados. Una correlación entre promedios de malla no es un
# momento simulado del ciclo económico.
#
# Una quita reduce el principal; cambiar el vencimiento también altera pagos e
# incentivos futuros. Hay que medir los efectos en el estado elegido. La interfaz
# no garantiza menores diferenciales ni mejoras de bienestar.

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
# ### 1. Ejemplo de impago a corto plazo
#
# `res_arellano` nombra el resultado a corto plazo utilizado en este laboratorio.
# El ejemplo usa choques de preferencia, el costo predeterminado y una malla gruesa;
# no es la calibración original publicada. Examine el residuo de iteración y las
# cotas de no arbitraje. La correlación de promedios de malla se imprime por
# separado de la estadística simulada y su estado. Una correlación simulada
# indefinida conserva el valor `NaN`.

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
    max_iter=2000,
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


print(f"Short-term example solved in {res_arellano.iterations} iterations.")
print(f"Bellman residual: {err_arellano_res:.2e}, Price bounds violation: {err_bounds_arellano:.2e}")
print(f"Unweighted grid-average spread/income correlation: {corr_y_arellano:.4f}")
print(f"Simulated spread/income correlation: {res_arellano.spread_y_corr}; status={res_arellano.metadata['spread_y_corr_status']}")

# %% [markdown]
# ### 2. Deuda a largo plazo y derivadas de precios
#
# La tasa de vencimiento es 0.05 y el cupón 0.03. Compare las curvas bajo estos
# supuestos. El control de derivadas describe el interpolante proyectado; no
# establece por sí solo efectos estructurales de dilución ni reproduce
# Chatterjee–Eyigungor (2012).

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
    max_iter=2000,
)

err_ce_res = float(res_ce.residual)
max_dq_db = float(np.max(res_ce.dq_db))
err_dq_db_mono = float(max(0.0, max_dq_db))

# Compare unweighted income-grid means at the middle debt-grid point.
med_b_idx = len(res_ce.b_grid) // 2
spread_lt = float(np.mean(res_ce.spread[med_b_idx, :]) * 10000.0)
spread_st = float(np.mean(res_arellano.spread[med_b_idx, :]) * 10000.0)

record("Long-term quantal operator: residual", err_ce_res, 1e-7, "PASS" if err_ce_res <= 1e-7 else "FAIL")
record("Long-term: projected interpolant derivative dq/db <= 0", err_dq_db_mono, 0.0, "PASS" if err_dq_db_mono <= 0.0 else "FAIL")

print(f"Long-term example solved in {res_ce.iterations} iterations.")
print(f"Bellman residual: {err_ce_res:.2e}, Max dq/db derivative: {max_dq_db:.6f}")
print(f"Average spread at median debt: Long-Term = {spread_lt:.1f} bps, Short-Term = {spread_st:.1f} bps")

# %% [markdown]
# ### 3. Restricciones de colateral: diagnósticos experimentales
#
# El planificador ilustrativo resuelve problemas con y sin restricción.
# Las políticas coinciden con una referencia exhaustiva, con estados vinculantes.
# La amplificación divide caídas medias de consumo en los mismos estados
# simulados, sin sustitutos ni límites inferiores impuestos. La fracción vinculante
# de malla y la frecuencia simulada de crisis son distintas. La capacidad promedia
# kappa*q*k_next. No es una replicación del modelo competitivo completo de Mendoza.

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

# Measurements over the post-burn simulation, not averages over the state grid.
sample_mendoza = res_mendoza.simulate(T=2500, seed=42).iloc[500:]
mean_debt_mendoza = float(sample_mendoza["b_next"].mean())
mean_collateral_mendoza = float(sample_mendoza["collateral_capacity"].mean())
gap_precautionary = float(max(0.0, mean_debt_mendoza - mean_collateral_mendoza))
err_crisis_frequency = abs(res_mendoza.crisis_probability - float(sample_mendoza["crisis"].mean()))

assert res_mendoza.converged
assert 0.0 <= res_mendoza.collateral_binding_fraction <= 1.0
record("Collateral: simulated debt/capacity gap", gap_precautionary, 1e-8, "PASS" if gap_precautionary <= 1e-8 else "FAIL")
record("Collateral: reproduced crisis frequency", err_crisis_frequency, 1e-12, "PASS" if err_crisis_frequency <= 1e-12 else "FAIL")

print(f"Experimental collateral example: {res_mendoza.iterations} iterations; converged={res_mendoza.converged}")
print(f"Weighted consumption-drop ratio: {res_mendoza.amplification_ratio:.4f}; status={res_mendoza.metadata['amplification_status']}")
print(f"Grid binding: {res_mendoza.collateral_binding_fraction * 100:.2f}%; simulated crises: {res_mendoza.crisis_probability * 100:.2f}%")
print(f"Average next debt ({mean_debt_mendoza:.4f}) vs average collateral capacity ({mean_collateral_mendoza:.4f})")

# %% [markdown]
# ### 4. Contabilidad de diferenciales y reestructuración
#
# Los componentes del diferencial suman por construcción en este ejemplo. El
# residuo es contable, no una prima de riesgo estimada independientemente. La
# quita sola cambia la deuda heredada; un cambio de vencimiento vuelve a resolver
# la calibración conservada. Precios de bonos cercanos a cero pueden producir rendimientos
# extremos: examine el estado de evaluación antes de interpretar el resultado.

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
# ### 5. Curvas de precios y componentes contables
#
# El panel A muestra precios de bonos largos con su precio sin riesgo. El panel B
# muestra precios del capital en un corte de la malla. El panel C separa pérdidas
# de supervivencia y residuos contables. Las figuras describen los ejemplos
# numéricos seleccionados.

# %%
col = _nbstyle.palette(4)
dash = _nbstyle.styles(4)

fig, (ax1, ax2, ax3) = _nbstyle.figura(1, 3)

# Panel 1: Bond Price Schedules
q_max = float(res_ce.metadata["q_rf"])
ax1.plot(res_ce.b_grid, res_ce.bond_price[:, -1], color=col[0], lw=2.2, label=r"High Income ($y_{\mathrm{high}}$)")
ax1.plot(res_ce.b_grid, res_ce.bond_price[:, len(res_ce.y_grid) // 2], color=col[1], lw=2.0, linestyle=dash[1], label=r"Middle Income ($y_{\mathrm{mid}}$)")
ax1.plot(res_ce.b_grid, res_ce.bond_price[:, 0], color=col[2], lw=1.8, linestyle=dash[2], label=r"Low Income ($y_{\mathrm{low}}$)")
ax1.axhline(q_max, color=col[3], lw=1.0, linestyle="--", label="Long-term risk-free price")
ax1.set(xlabel=r"Next-Period Debt $b'$", ylabel=r"Bond Price $q(b', y)$", title="Long-Term Bond Prices")
ax1.legend(loc="lower left", frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)

# Panel 2: Experimental collateral-model asset prices
ik_mid = len(res_mendoza.k_grid) // 2
ax2.plot(res_mendoza.b_grid, res_mendoza.asset_price_q[:, ik_mid, 2], color=col[0], lw=2.2, label=r"High Prod ($z_{\mathrm{high}}$)")
ax2.plot(res_mendoza.b_grid, res_mendoza.asset_price_q[:, ik_mid, 1], color=col[1], lw=2.0, linestyle=dash[1], label=r"Normal Prod ($z_{\mathrm{mid}}$)")
ax2.plot(res_mendoza.b_grid, res_mendoza.asset_price_q[:, ik_mid, 0], color=col[2], lw=1.8, linestyle=dash[2], label=r"Low Prod ($z_{\mathrm{low}}$)")
ax2.set(xlabel="External Debt $b$", ylabel=r"Tobin's $q$ (Capital Price)", title="Experimental Collateral Prices")
ax2.legend(loc="lower left", frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)

# Panel 3: Spread Term Structure Decomposition
m_vec = df_term["maturity_years"].values
ax3.plot(m_vec, df_term["spread_bps"].values, color=col[0], lw=2.2, label="Total Sovereign Spread")
ax3.plot(m_vec, df_term["default_loss_component"].values, color=col[1], lw=2.0, linestyle=dash[1], label="Expected Default Loss")
ax3.plot(m_vec, df_term["risk_premium_component"].values, color=col[2], lw=1.8, linestyle=dash[2], label="Residual Component")
ax3.set(xlabel="Maturity (Years)", ylabel="Spread (bps)", title="Spread Term-Structure Decomposition")
ax3.legend(loc="upper right", frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)

# %% [markdown]
# ### 6. Tabla de comprobaciones numéricas
#
# La tabla cubre residuos, cotas y comparaciones del ejemplo elegido. No replica
# momentos publicados ni garantiza resultados para todos los parámetros. La
# correlación de malla y la amplificación experimental se reportan sin convertir
# sus signos o magnitudes esperados en criterios de validación.

# %%
scorecard = pd.DataFrame(checks)
print(scorecard.to_string(index=False, formatters={"measured": "{:.2e}".format, "bound": "{:.2e}".format}))
assert (scorecard["measured"] <= scorecard["bound"]).all()
print(f"\nAll {len(scorecard)} sovereign debt crisis and restructuring checks pass.")

# %% [markdown]
# ## Interpretación de resultados
#
# Lea los valores de las celdas ejecutadas, no cifras fijas en la prosa. Compruebe
# primero convergencia y cotas de precios. Distinga una correlación de malla de una
# correlación temporal, y un promedio de malla de una estadística ergódica.
#
# El control de derivadas depende de una proyección monótona impuesta. La
# aditividad de diferenciales es una identidad. Los resultados de reestructuración
# son cambios con signo en un estado, no estimaciones de bienestar. Se necesitan
# referencias independientes, convergencia de malla y momentos empíricos antes
# de usar las demás extensiones para conclusiones cuantitativas de política.

# %% [markdown]
# ## Su turno
#
# Cambie la quita y el **aumento fraccional de duración**. La implementación usa
# `delta_new = delta / (1 + maturity_extension)`: 1.0 duplica la duración esperada;
# no agrega un año. El ejercicio comprueba resultados numéricos válidos e imprime
# efectos con signo; no garantiza compresión de diferenciales para todo parámetro.
#
# 1. Compare quitas 0.10 y 0.35 con el mismo vencimiento.
# 2. Compare aumentos de duración 0.0 y 1.0 con la misma quita.
# 3. Repita con valores explícitos de `b_distress` y `y_distress`. Examine
#    probabilidades de impago y precios antes de comparar bienestar.

# %%
user_debt_haircut = 0.20  # ← change this: Face-value debt haircut, float in [0.05, 0.40]
user_maturity_extension = 1.0  # ← change this: Fractional duration increase in [0.0, 5.0]

assert isinstance(user_debt_haircut, (int, float)) and 0.05 <= user_debt_haircut <= 0.40
assert isinstance(user_maturity_extension, (int, float)) and 0.0 <= user_maturity_extension <= 5.0

restruct_user = evaluate_debt_restructuring(
    res_ce,
    haircut=float(user_debt_haircut),
    maturity_extension=float(user_maturity_extension),
)

print(f"User Restructuring: Haircut = {user_debt_haircut * 100:.1f}%, Duration increase = {user_maturity_extension * 100:.1f}%")
print(f"  Immediate Spread Reduction: {restruct_user.spread_reduction_bps:.2f} bps")
print(f"  Pre-default prob: {restruct_user.pre_default_prob * 100:.2f}% -> Post-default prob: {restruct_user.post_default_prob * 100:.2f}%")

# Output validity for the selected inputs; report signed economic effects.
assert np.isfinite(restruct_user.spread_reduction_bps)
assert 0.0 <= restruct_user.post_default_prob <= 1.0

# %% [markdown]
# ## ¿Qué tan exhaustivo es esto?
#
# Es un recorrido exploratorio. Las referencias independientes contrastan deuda
# corta y los operadores declarados de deuda larga, colateral y reestructuración.
# No establecen todos los momentos publicados ni la convergencia económica de
# las extensiones. Consulte `docs/es/sovereign_extensions_validation.md`.
#
# La resolución de la malla forma parte de cualquier resultado de default
# soberano, no es un detalle. En la malla archivada de la autora, 200 × 21, el
# estudio independiente de Arellano (2008) se acerca a la volatilidad publicada
# del diferencial; refinar la misma calibración y reingreso a 797 × 321 la reduce
# a la mitad, como advirtieron Hatchondo, Martinez y Sapriza (2010) para mallas
# discretas equiespaciadas. El veredicto de convergencia de la calibración
# pública, con tolerancias prerregistradas, está en `docs/es/arellano_replication.md`.
# Las mallas gruesas de este cuaderno son ilustrativas.
