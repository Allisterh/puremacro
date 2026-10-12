> 🇬🇧 [English](../research_workflows.md) · 🇪🇸 Español

# Evidencia independiente y estimación estructural

El [flujo de pronósticos entre países](cross_country_forecasting.md) añade un
panel fijo de tres países, una evaluación con ventana creciente y procedencia
portátil. Usa una edición reciente de datos revisados; no es evidencia causal
ni de tiempo real. Está incluido en puremacro 4.8.1.

Estos flujos de investigación conectan validación independiente y estimación
estructural. Funcionan sin conexión desde el paquete
instalado con las dependencias numéricas estándar y conservan los supuestos,
las unidades y el alcance de la evidencia.

Los [estudios con datos reales](empirical_research.md) añaden estimación SW07
con momentos observados y diagnósticos condicionales a la calibración, y una
replicación base de Romer–Romer (2010) con datos originales, software independiente
y valores publicados redondeados. El ejemplo NK sintético sigue siendo una
prueba de recuperación; los nuevos estudios tienen datos, supuestos y
conclusiones propios.

| Tarea | API | Guía |
|---|---|---|
| Revisar evidencia numérica y de datos | `puremacro.validation.run_research_benchmarks` | [Benchmarks independientes](research_benchmarks.md) |
| Ajustar modelos a momentos empíricos correlacionados | `puremacro.structural.fit_structural` y `lp_moment_targets` | [Puente estructural](structural_bridge.md) |
| Comparar correcciones de momentos y ponderaciones SW07 | `puremacro.structural.run_sw07_estimator_experiment` | [Controles emparejados del estimador](sw07_estimator_experiment.md) |

## Benchmarks

```python
from puremacro.validation import run_research_benchmarks

report = run_research_benchmarks()
print(report.passed)  # False: falla el redondeo publicado del estadístico t RR2010.
report.write("research_output/benchmarks")
```

Los cinco casos distinguen soluciones analíticas, resultados externos
autenticados de Dynare y la especificación original de Romer–Romer con
software independiente y valores publicados.
El expediente JSON/Markdown conserva valores observados y de referencia,
unidades, fuentes, tolerancias, controles negativos y limitaciones. No certifica
todos los modelos ni presenta ejercicios sintéticos como replicaciones empíricas.
Pasan cuatro de cinco casos: el estadístico t RR2010 queda fuera del intervalo
de redondeo publicado por aproximadamente 0.000003925. El fallo se conserva;
véase la [guía empírica](empirical_research.md).

## De proyecciones locales a un modelo nuevo keynesiano

```bash
python -m puremacro.examples.structural_irf_matching --output research_output/structural
```

La aplicación simula una economía de tres ecuaciones desde una solución en
forma cerrada independiente, con perturbación monetaria persistente y errores
de medición. Estima respuestas LP de producto e inflación con **covarianza HAC
conjunta**, y ajusta aversión al riesgo, pendiente de Phillips y persistencia
resolviendo el modelo mediante `load_mod`. La innovación es 0.01 en la tasa
trimestral: un punto porcentual.

Los horizontes 0–4 entran al ajuste; 5–8 son comprobaciones descriptivas con la
misma muestra. Se exportan observaciones, momentos, covarianza, parámetros,
comparaciones, figura y manifiesto con modelo, semilla y hash de los datos.
Es recuperación con datos simulados, no replicación histórica. Para un estudio
empírico sustituya el simulador por datos transformados y shocks identificados,
o construya `MomentTargets` con estimaciones y su covarianza conjunta. Frecuencia,
unidades, transformación y normalización deben coincidir con el modelo.
