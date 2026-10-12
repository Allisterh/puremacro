> 🇬🇧 [English](../empirical_research.md) · 🇪🇸 Español

# Replicación publicada con datos reales

Este estudio amplía los [flujos de investigación](research_workflows.md)
con una especificación original publicada y sus datos originales.
Comprueba un resultado concreto de un artículo.

## Romer–Romer (2010): respuesta tributaria original

```bash
python -m puremacro.examples.romer_romer_2010_replication --output research_output/rr2010
```

Se comprueba un resultado base de
[Romer y Romer (2010), *The Macroeconomic Effects of Tax Changes*](https://www.aeaweb.org/articles?id=10.1257/aer.100.3.763).
El libro original y la especificación `EXOGERNR.RAT` proceden del
[archivo de los autores](https://eml.berkeley.edu/~dromer/papers/DataSet.zip).
Se conservan hashes de las fuentes, los datos derivados y la referencia.

```python
from puremacro.replication import estimate_rr2010_baseline

replication = estimate_rr2010_baseline()
print(replication.to_frame().loc[10])
# Conservar la covarianza completa al usar estas respuestas como momentos.
targets = replication.to_moment_targets()
print(targets.labels)
```

En 1950T1–2007T4 se explica `100 * delta log(PIB real)` con constante y
valores contemporáneos y doce rezagos de `100 * (DEFIC + LONGR) / NOMGDP`.
Las sumas acumuladas de los trece coeficientes tributarios producen la
respuesta del nivel del PIB ante un aumento tributario de uno por ciento
del PIB. Un choque positivo representa un aumento de impuestos.

La especificación original usa covarianza MCO convencional con ajuste por
grados de libertad residuales. La covarianza de respuestas acumuladas aplica
la misma transformación lineal a toda la covarianza de coeficientes. HAC y
proyecciones locales pueden ser extensiones útiles, pero cambian la
especificación publicada.

Se reportan dos comprobaciones distintas:

1. **Paridad numérica:** coeficientes, covarianzas y respuestas acumuladas
   contra una ejecución congelada de software independiente que no importa
   puremacro. La tolerancia solo permite redondeo numérico.
2. **Evidencia publicada:** el mínimo a horizonte diez, respuesta `-3.08` y
   estadístico t `-3.53`, en la página impresa 781. Solo el redondeo publicado
   justifica la tolerancia mayor; no se aplica a la paridad de software.

Los datos originales producen una respuesta a horizonte diez de
**-3.0808127631** y un estadístico t de **-3.5249960750**. La paridad de software
pasa, y coinciden la respuesta y el horizonte mínimo publicados. El estadístico
t queda fuera del intervalo de redondeo de -3.53 por **0.000003925**. La lectura
independiente de los bancos RATS originales corrobora este valor. La causa no
está resuelta; no se redondean valores intermedios ni amplían tolerancias para
forzar coincidencia. El caso publicado y la aplicación informan **FAIL**, con
código de salida distinto de cero. No se afirma coincidencia de todos los objetivos.

El alcance es esta especificación base con datos originales, no todas las
pruebas de robustez ni una validación independiente del supuesto de exclusión
narrativa. El caso anterior `regression.romer_romer_tax_multiplier_ols` es una
LP modificada a horizonte ocho, con otra muestra y datos revisados, y su
comparación gruesa se identifica como tal.

## Reproducibilidad

La aplicación funciona sin red con datos y referencias incluidos en el
paquete. Exporta definiciones, muestra, unidades, estimaciones, diagnósticos
y figuras. La conversión de fuentes y regeneración de referencias son tareas
de desarrollo separadas de la ejecución normal. El software de referencia
no es una dependencia de ejecución.

La [guía del puente estructural](structural_bridge.md) explica las convenciones
de covarianza y la [guía de benchmarks](research_benchmarks.md), las fuentes,
comparaciones estrictas y controles negativos.
