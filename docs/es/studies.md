> 🇬🇧 [English](../studies.md) · 🇪🇸 Español

# Estudios y comparaciones reproducibles

`puremacro.study` registra un estudio estadístico explícito: datos seleccionados,
transformaciones, unidades, supuestos de identificación, configuración del
estimador, resultados, diagnósticos, versiones y sumas de verificación. Reutiliza
los estimadores LP y VAR recursivo, los formatos de informes y el códec portátil
del runtime. No deduce una interpretación causal ni certifica un modelo económico.

## Estimar y comparar proyecciones locales

Este ejemplo completo funciona sin conexión y utiliza datos sintéticos.
La exogeneidad se declara como supuesto ilustrativo; el programa no la demuestra.

```python
from dataclasses import replace
import numpy as np
import pandas as pd
from puremacro.study import (
    VariableSpec, StudySpec, run_study, compare_studies,
    save_study, load_study,
)

rng = np.random.default_rng(173)
observations = rng.normal(size=(240, 3))
for t in range(1, len(observations)):
    observations[t] += 0.4 * observations[t - 1]
data = pd.DataFrame(
    observations,
    columns=["shock", "output", "price"],
    index=pd.period_range("1960Q1", periods=240, freq="Q"),
)
data.attrs = {"source": "demostración sintética", "data_seed": 173}

spec = StudySpec(
    name="LP, dos rezagos", method="lp",
    variables=(
        VariableSpec("shock", "puntos porcentuales"),
        VariableSpec("output", "puntos de índice"),
        VariableSpec("price", "puntos de índice"),
    ),
    responses=("output",), shock="shock", controls=("price",),
    frequency="Q", identification="observed_exogenous",
    lags=2, horizon=8, shock_size=0.25, ci=0.90,
)
baseline = run_study(data, spec)
alternative = run_study(data, replace(spec, name="LP, cuatro rezagos", lags=4))
comparison = compare_studies(baseline, alternative)
print(comparison.to_frame())
print(baseline.audits)

# El directorio debe ser nuevo: no se sobrescriben resultados existentes.
baseline.write_report("study_output/lp_dos_rezagos")
# También puede conservarse el registro completo en memoria, en un navegador.
payload = save_study(baseline)
restored = load_study(payload)
assert restored.manifest == baseline.manifest
```

`write_report` genera tablas Markdown, LaTeX, Typst y CSV, `manifest.json`
y el archivo sin pérdida `study.pmstudy`. Markdown y el manifiesto explican
métodos, supuestos, muestras efectivas y alcance de los diagnósticos. CSV es un
formato de presentación; el archivo conserva tipos de índices y metadatos
admitidos de `DataFrame.attrs`. `to_json()` contiene el manifiesto y las tablas
de salida, pero no los datos de entrada. `StudyResult` y `StudyComparison`
son registros inmutables; sus tablas y diccionarios devueltos son copias.

## Declarar transformaciones, unidades y frecuencia

Cada columna necesita un `VariableSpec`. Las transformaciones disponibles
son `identity`, `log`, `diff` y `logdiff`; `scale` multiplica la serie transformada.
`unit` describe esa cantidad final. Por ejemplo,
`VariableSpec("output", "crecimiento logarítmico porcentual", "logdiff", 100)`
calcula `100 * diff(log(output))` y requiere niveles positivos. No se convierte
a porcentajes ni se anualiza una variable a partir de su nombre.

El índice debe ser un `PeriodIndex` o `DatetimeIndex` único, creciente y
consecutivo. `frequency="Q"` indica periodos trimestrales; para fechas se usa
el desplazamiento de calendario correspondiente, como `"QS"` o `"MS"`.
Los alias de fin de trimestre/mes admiten la escritura compatible con pandas.
Un periodo ausente o un valor no finito genera un error: nunca se comprime el
tiempo eliminando filas incompletas. Las columnas no declaradas también son errores.

`sample_start` y `sample_end` deben identificar observaciones del índice.
La selección ocurre antes de transformar. Diferenciar elimina exactamente la
primera observación seleccionada de todas las series; no se toma un valor de
fuera de esa muestra. Se guardan los datos originales seleccionados, no las
filas externas a la muestra. `data.attrs` puede registrar fuente, versión de
datos y notas de descarga. Los objetos de metadatos no admitidos generan un
error; no se descartan silenciosamente.

## Interpretar los contratos de estimación

| Configuración | `method="lp"` | `method="var_cholesky"` |
|---|---|---|
| Identificación | `observed_exogenous`: exogeneidad del shock observado condicionada a los regresores, declarada como supuesto | `recursive`: orden contemporáneo recursivo declarado |
| Respuesta en horizonte h | `y[t+h] - y[t-1]` | Impulso-respuesta en niveles en h |
| Tamaño del shock | Múltiplo de la unidad transformada del shock observado | Impacto sobre la variable del shock, en su unidad transformada |
| `lags=p` | Rezagos 1–p de cada respuesta, shock y controles; controles contemporáneos y constante | Rezagos 1–p de todas las variables ordenadas y constante; p ≥ 1 |
| Muestra efectiva | Primera fecha del shock `max(1,p)`; última `T-h-1` | Fechas p a T−1 en el ajuste del VAR |
| Incertidumbre | Intervalos normales HAC puntuales; Bartlett con ancho h+1 | Intervalos percentiles puntuales mediante bootstrap residual opcional |

Los horizontes incluyen el extremo: `horizon=8` devuelve 0 a 8. La muestra
de LP cambia con el horizonte y puede cambiar al modificar rezagos. Cada
resultado presenta `n_obs`; `effective_by_horizon` registra fechas del shock
y los regresores. Igualdad de la ventana de entrada no significa igualdad de
todas las observaciones usadas en la estimación.

En LP, los nombres de respuestas, shock y controles deben ser distintos.
Con varias respuestas, cada regresión usa los rezagos de su propia respuesta;
las otras respuestas no se añaden como controles. LP exige `n_boot=0`.
En VAR, el orden contiene todas las variables exactamente una vez y `controls`
está vacío:

```python
recursive = replace(
    spec, name="VAR recursivo", method="var_cholesky",
    identification="recursive", controls=(),
    ordering=("price", "shock", "output"), n_boot=199, seed=71,
)
var_result = run_study(data, recursive)
```

El VAR multiplica las respuestas por `shock_size / P[k,k]`, donde `P` es el
factor de Cholesky estimado y `k` la posición del shock. Los intervalos usan
esta normalización **fija**; no se normaliza cada réplica a idéntico impacto.
Los shocks negativos invierten correctamente los extremos del intervalo.
`n_boot=0` omite las columnas de intervalos. Las réplicas fallidas y la
inestabilidad estimada permanecen visibles en los diagnósticos. Una estimación
o intervalo exportado no finito genera un error.

`compare_studies` requiere al menos dos estudios con nombres únicos y el
mismo método, identificación, orden, horizonte, definiciones de variables,
convención de respuesta, tamaño/unidad del shock, y datos/metadatos originales
y transformados idénticos. Pueden cambiar rezagos, confianza, semilla y número
de réplicas, que quedan registrados. Se rechazan comparaciones mixtas LP/VAR:
un shock observado con respuesta acumulada no equivale automáticamente a una
innovación identificada recursivamente con respuesta en niveles.

## Conservar y cargar evidencia

`save_study(result, "study.pmstudy")` crea un archivo nuevo;
`save_study(result)` devuelve bytes.
`load_study(path_or_bytes, max_bytes=64 * 1024 * 1024)` verifica el inventario
exacto, esquema, SHA-256 de tablas y manifiesto, unidades, tamaños muestrales
y valores numéricos finitos. Limita la expansión ZIP y las dimensiones NPY
antes de decodificar. Usa JSON y el códec NPZ versión 2 del runtime, incluidos
metadatos admitidos. No extrae rutas, deserializa pickle, importa callbacks
ni ejecuta estimadores. No requiere Parquet, una cuenta remota ni conexión.

Las sumas detectan cambios, pero no autentican al autor ni demuestran que
un resultado deliberadamente modificado y vuelto a firmar sea correcto.
Se registran semillas y versiones numéricas. La igualdad exacta al repetir
se limita al mismo entorno numérico; otras bibliotecas BLAS o versiones pueden
alterar el redondeo. Los estados son `passed`, `failed`, `not_checked` y
`unsupported`; la identificación permanece `not_checked`.

## Registrar un modelo o escenario ya evaluado

`ScenarioSpec`, `ScenarioResult` y `record_scenario` registran resultados de
solucionadores existentes: DSGE, programación dinámica, HANK, ajuste estructural
o escenarios estáticos. No ejecutan otro solucionador. El código fuente queda
como texto inerte; se conservan parámetros, resultados, entradas, diagnósticos
y tablas adicionales de evidencia.

Este ejemplo registra una trayectoria calibrada explícitamente, sin afirmar
que sus parámetros se estimaron a partir de las observaciones ilustrativas:

```python
from puremacro.study import ScenarioSpec, record_scenario

scenario_spec = ScenarioSpec(
    name="Respuesta calibrada ilustrativa", model="respuesta geométrica declarada",
    variables=spec.variables, responses=spec.responses, shock=spec.shock,
    frequency=spec.frequency, identification=spec.identification,
    response_convention="y[t+h]-y[t-1]",
    horizon=spec.horizon, shock_size=spec.shock_size,
)
outputs = pd.DataFrame({
    "response": ["output"] * (spec.horizon + 1),
    "horizon": range(spec.horizon + 1),
    "estimate": [0.1 * 0.7**h for h in range(spec.horizon + 1)],
})
scenario = record_scenario(
    outputs, scenario_spec,
    inputs=baseline.inputs, transformed_inputs=baseline.data,
    model_source="response[h] = 0.1 * persistence**h; pre-shock baseline = 0",
    parameters={"persistence": 0.7, "impact": 0.1},
    audits=pd.DataFrame([{
        "check": "economic_identification", "status": "not_checked",
        "detail": "La equivalencia de shocks externos se supone en esta ilustración.",
    }]),
    provenance={"purpose": "ilustración calibrada; parámetros no estimados"},
)
overlay = compare_studies(
    baseline, scenario,
    comparison_note="Se supone que la innovación del modelo corresponde al shock "
                    "exógeno declarado y a la misma base previa; sólo ilustrativo.",
)
scenario.write_report("study_output/respuesta_calibrada")
```

Las salidas contienen exactamente `response`, `horizon`, `estimate`, primero
por respuesta y después por horizonte 0..H. `variables` declara las unidades.
Los diagnósticos contienen `check`, `status`, `detail`; se conservan estados
fallidos, no admitidos o no comprobados. `parameters` contiene escalares reales
finitos. `evidence={"nombre": dataframe}` admite matrices de calibración,
covarianza conjunta real de momentos, objetivos usados/reservados para evaluación,
distribuciones, estados o registros de réplicas. Nunca se deduce una covarianza
a partir de bandas marginales de LP.

`inputs` puede ser una tabla estática del modelo. `transformed_inputs` es
obligatorio si hay transformaciones o escalas no identidad declaradas; no se
deducen ni recalculan transformaciones de datos del modelo. Para insumo-producto
estático son etiquetas explícitas apropiadas `horizon=0`,
`frequency="annual_static"`, `identification="calibrated_input_output_scenario"`
y `response_convention="static_level_change"`. Un escenario independiente no
requiere índice de fechas empírico.

Cualquier comparación con escenarios exige `comparison_note`, además de unidades,
transformaciones, tamaño de shock, identificación, frecuencia, horizonte y
convención idénticos, y entradas originales/transformadas exactamente iguales.
La nota no permite eludir un contrato incompatible. Estas comprobaciones validan
la comparabilidad declarada, no demuestran equivalencia económica. No se inventan
tamaños muestrales empíricos ni intervalos para escenarios.

`save_study`/`load_study` distinguen archivos empíricos y de escenarios. El
informe de escenario incluye `model_source.txt`; el archivo conserva evidencia
y procedencia sin pérdida. Se rechazan estructuras que el códec no conserva,
como nombres de niveles MultiIndex en columnas o atributos propios de subclases.
Conviértalas deliberadamente a tablas ordinarias, conservando la definición de
coordenadas en procedencia. Toda evidencia numérica debe ser finita; conserve
los estados de réplicas fallidas por separado, sin ocultarlas ni sustituirlas.

El flujo no automatiza identificación IV, estimación estructural ni selección
de modelos. Para construir objetivos empíricos correlacionados y ajustar un
callback, use el [puente estructural](structural_bridge.md) y los
[flujos de investigación](research_workflows.md), y registre sus salidas reales.
