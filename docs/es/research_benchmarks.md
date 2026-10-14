> 🇬🇧 [English](../research_benchmarks.md) · 🇪🇸 Español

# Benchmarks de investigación independientes

Ejecute comparaciones independientes sin conexión y conserve sus evidencias:

```python
from puremacro.validation import run_research_benchmarks

report = run_research_benchmarks()
print(report.passed)
paths = report.write("research-benchmarks")
```

Funciona desde el paquete instalado, sin checkout, MATLAB, Dynare ni acceso a
internet. Desde el repositorio, `python tools/run_research_benchmarks.py --output
/tmp/benchmarks` genera el mismo expediente y devuelve un código distinto de cero
si falla alguna comparación. La opción repetible `--case` selecciona casos.

La suite actual informa **4/5 casos aprobados y resultado global FAIL**. Los
datos originales RR2010 reproducen la respuesta del PIB y su horizonte mínimo,
pero el estadístico t queda apenas fuera del intervalo de redondeo publicado.
Se conserva la discrepancia sin ampliar tolerancias; el comando por defecto
devuelve un código distinto de cero.

## Alcance de los cinco casos

| Caso | Tipo de evidencia | Referencia y alcance |
|---|---|---|
| `linear_minimum_distance` | Analítica | Coeficientes, covarianza y objetivo GLS en forma cerrada; cuatro momentos correlacionados y dos parámetros |
| `growth_analytical` | Analítica | Política exacta de crecimiento con utilidad logarítmica y depreciación completa; estado estacionario y derivadas |
| `dynare_rbc_order2` | Software externo | Exportación previa de Dynare 7.0; reglas de decisión de segundo orden y momentos de simulación de un modelo RBC |
| `rr2010_baseline_software` | Software externo | MCO independiente sobre los datos originales; coeficientes, covarianzas completas, respuestas acumuladas, errores estándar, estadísticos t y muestra |
| `rr2010_published_peak` | Empírica publicada | Página 781 de Romer–Romer (2010): mínimo de la Figura 4 en el trimestre diez, respuesta -3.08 y estadístico t -3.53 |

Los primeros dos casos son ejercicios numéricos sintéticos. El caso Dynare
utiliza innovaciones simuladas. Los últimos dos casos usan los datos empíricos
originales de Romer–Romer. La paridad de software y
los valores publicados redondeados tienen comparaciones y tolerancias separadas.

Un resultado satisfactorio respalda exclusivamente las métricas, parámetros y
tolerancias del expediente. No establece precisión global, cobertura inferencial,
validez externa ni credibilidad de los supuestos de una política.

## Expediente y controles

`benchmark_report.json` incluye todos los valores observados y de referencia,
dimensiones, unidades, errores, tolerancias, fuentes, transformaciones, muestreo,
limitaciones, fecha UTC y versiones del entorno. `benchmark_report.md` ofrece
tablas legibles y abrevia únicamente los arreglos largos.

La comparación elemento por elemento exige
`abs(observado-referencia) <= atol + rtol*abs(referencia)`. Rechaza claves faltantes,
arreglos vacíos, dimensiones distintas, valores no finitos, fallos de cálculo y
procedencia incompleta. No permite broadcasting ni convierte fallos en omisiones
exitosas. Una selección vacía o identificadores duplicados producen `ValueError`.

Cada caso aprobado incluye un control negativo: debe rechazar un valor de
referencia deliberadamente alterado que se presenta como observado. Las pruebas
también alteran los resultados efectivos de cada caso. Esto verifica que la
comparación puede fallar; no sustituye la independencia del oráculo.

El benchmark del [puente estructural](structural_bridge.md) utiliza la covarianza
del **vector estimado de momentos**. Ya incorpora la escala muestral, de modo
que no se vuelve a dividir entre el tamaño de muestra. Se verifica toda la
matriz de covarianza de los parámetros.

## Procedencia externa y empírica

La referencia Dynare se ejecutó con Dynare 7.0 y MATLAB R2026a el 2026-09-20.
Contiene 2,500 innovaciones comunes y descarta las primeras 500 observaciones.
La covarianza usa `ddof=1`; la covarianza de rezago uno centra por separado sus
extremos y divide entre `N-2`. Las variables, estados y shocks se alinean por
nombre. Se verifican hashes del modelo, las innovaciones y el NPZ, además del
identificador de ejecución y el hash fijado de la referencia. El benchmark sin
conexión no vuelve a ejecutar Dynare ni regenera la evidencia.

La [replicación Romer–Romer](empirical_research.md) conserva la versión original
de los datos y la especificación `EXOGERNR.RAT`: crecimiento trimestral del PIB
sobre una constante y rezagos cero a doce del cambio tributario exógeno,
1950T1–2007T4. Acumula los coeficientes y su covarianza MCO convencional con
ajuste por grados de libertad residuales. Una implementación independiente de
software genera la referencia congelada; el benchmark no ejecuta RATS.
Coeficientes, covarianzas completas, respuestas, errores estándar, estadísticos
t y tamaños de muestra usan `rtol=1e-10, atol=1e-11`.

El caso publicado comprueba por separado el horizonte diez, -3.08 log por ciento
y el estadístico t -3.53 de la página impresa 781. Su `rtol=0, atol=0.005`
admite únicamente el redondeo a dos decimales; el horizonte entero debe coincidir.
Se replica una especificación base, sin validar independientemente la exogeneidad
narrativa ni reproducir todos los ejercicios de robustez.

Los datos originales producen -3.0808127631 y t=-3.5249960750. La comparación
con -3.53 excede la tolerancia 0.005 por 0.000003925. Los bancos RATS originales
corroboran el resultado del libro. La paridad de software pasa; la comparación
publicada conserva el fallo. Las pruebas verifican los valores coincidentes y
la conservación del fallo: aprobar las pruebas no implica aprobar todos los
benchmarks científicos.

## Extensiones y fuentes

`ResearchBenchmark` permite funciones separadas `compute` y `reference`, unidades
por métrica, citas, procedencia, muestreo, transformaciones, limitaciones y
tolerancias. Ambas funciones deben devolver diccionarios no vacíos con las
mismas claves. Una referencia generada por la propia función evaluada es una
prueba de regresión, no evidencia independiente. Antes de declarar una
replicación empírica, documente la tabla publicada, especificación, muestra y
transformaciones. La [guía en inglés](../research_benchmarks.md) incluye un ejemplo
completo de extensión.

- [Hansen, *Econometrics*](https://users.ssc.wisc.edu/~bhansen/econometrics/): marco GLS y distancia mínima.
- [QuantEcon, modelo Cass-Koopmans](https://python.quantecon.org/cass_koopmans_1.html): contexto del modelo de crecimiento.
- [Manual de Dynare](https://www.dynare.org/manual/the-model-file.html#stochastic-solution-and-simulation): convenciones de perturbación y simulación.
