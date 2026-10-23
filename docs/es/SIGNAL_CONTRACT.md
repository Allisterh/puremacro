> 🇬🇧 [English](../SIGNAL_CONTRACT.md) · 🇪🇸 Español

# Contrato de señal

> Los constructores existentes mantienen sus diagnósticos opcionales de cobertura.
> `puremacro.narrative.signals` añade réplicas documentales explícitas, reestimación
> conjunta, escenarios de sensibilidad y calibración fuera de muestra. No establece
> identificación causal ni incertidumbre total del estimador.

El contrato de señal es la estructura de datos por `RiskIndex` que permite
a un estimador LP / SVAR aguas abajo conocer la fiabilidad de una lectura
del índice y propagar trayectorias de medición explícitas a reestimaciones.

## Esquema (Slice 1)

`puremacro.narrative.types.RiskIndex` incorpora dos campos opcionales:

```python
@dataclass
class RiskIndex:
    name: str
    country: str
    series: pd.Series
    method: str
    corpus: str
    language: str
    normalization: str
    metadata: dict
    quality: SignalQualityReport | None = None   # 0.65.0+
    draws:   pd.DataFrame | None          = None
```

Ambos campos toman el valor `None` por defecto — todo código anterior a 0.65.0 funciona de manera idéntica.

## Activación (Slice 1)

```python
from puremacro.narrative.indices import lui

ri = lui(records, country="USA", language="en", with_quality=True)
ri.quality.summary()      # one-row DataFrame: mean_n_docs, mean_doc_length, n_coverage_gaps, ...
```

El mismo argumento de palabra clave `with_quality=False` está disponible en todos los índices
canónicos de `puremacro.narrative.indices.__all__`: `epu`, `mpu`, `gpr`, `tone`,
`wui`, `lui`, `ltui`, `ltui_up`, `ltui_down`, `lwui`, `lwui_wage`,
`bbui`, `cboui`, `ep_ui`, `erpui`, `eurlex_ui`, `sotuui`, `bluesky_ui`.

## `SignalQualityReport`

| Campo                 | Incorporado en | Descripción                                                        |
|-----------------------|----------------|--------------------------------------------------------------------|
| `n_docs_per_period`   | 0.65.0         | documentos por trimestre que alimentaron el kernel                 |
| `avg_doc_length`      | 0.65.0         | promedio de tokens por documento por trimestre                     |
| `coverage_gaps`       | 0.65.0         | trimestres sin documentos dentro del rango de fechas               |
| `kernel_agreement`    | Flujo explícito de señales | media de correlaciones temporales por pares entre referencia y kernels alternativos; se cuentan los pares definidos |
| `multilingual_parity` | No calculado por este flujo | requiere un análisis separado de comparabilidad lingüística |
| `doc_bootstrap_sd`    | Flujo explícito de señales | desviación estándar muestral por periodo entre réplicas documentales |
| `corpus_loo_max_swing`| Flujo explícito de señales | máximo cambio absoluto al omitir un corpus |
| `benchmark_scores`    | `calibrate_signal` | Pearson/Spearman/RMSE/MAE/sesgo fuera de muestra en unidades del objetivo |
| `event_panel`         | `evaluate_events` | correlación de rangos, aciertos del decil superior y AUC en fechas de evaluación declaradas |
| `survey_scores`       | `calibrate_signal(kind="survey")` | las mismas métricas frente a una encuesta |

## Validación

`RiskIndex.__post_init__` valida `draws` (cuando se especifica):
- `draws.index` debe coincidir con `series.index`.
- `draws.columns` debe ser un `pd.MultiIndex` de dos niveles denominado `['source', 'draw_id']`.
- Cada valor del nivel `source` debe pertenecer al conjunto `{'kernel', 'lexicon', 'doc', 'corpus'}`.

El flujo nuevo comprueba además valores finitos, coordenadas únicas, cobertura
del calendario, ponderaciones positivas declaradas y coordenadas de salida de
los callbacks. No cambia los valores por defecto de los constructores existentes.

## Réplicas documentales con significado explícito

```python
from puremacro.narrative.signals import score_documents, build_signal_draws
from puremacro.narrative.indices._kernels import keyword_count_kernel

# records: (fecha, texto, URL, metadatos), una entrada por documento.
scores = score_documents(
    records,
    lambda rows: keyword_count_kernel(rows, terms=frozenset({"incertidumbre"}), language="es"),
)
ri = build_signal_draws(
    scores, name="incertidumbre", country="MEX", language="es",
    method="keyword_count", corpus="corpus local declarado",
    resampling="within_period", n_draws=199, seed=71,
    normalization="raw",
)
```

El adaptador exige una puntuación fechada por documento, en el orden original.
Conserva longitud y metadatos completos de corpus, grupo persistente y peso.
También puede proporcionar directamente una tabla con `date` y `score`.
Precalcular puntuaciones mantiene los servicios remotos, la codificación humana
y la selección de modelos fuera del remuestreo. El callback es código de
confianza ejecutado por el usuario, nunca almacenado ni importado por un archivo.

La agregación es la media documental. Con `weight_column="weight"`, los pesos
positivos entran tanto al numerador como al denominador. Una columna `weight`
no puede ignorarse silenciosamente. El remuestreo documental es uniforme y
conserva los pesos de cada documento: es un bootstrap de la razón ponderada.
Otras reglas de agregación quedan fuera de este flujo.

La unidad de remuestreo debe elegirse explícitamente:

- `within_period` selecciona el mismo número de documentos de forma independiente
  en cada periodo. Mantiene las fechas, pero no estima dependencia temporal del
  error de medición entre periodos.
- `cluster`, con `cluster_column`, remuestrea grupos persistentes una vez por
  réplica y aplica sus multiplicidades a todas las fechas. Cada grupo debe estar
  presente en todos los periodos. El denominador por fecha recoge diferencias en
  números/pesos documentales. Se rechazan paneles incompletos de grupos; nunca se
  intercambian fechas ni se trasladan periodos.

Debe haber documentos en cada periodo de `frequency` (por defecto `"Q"`).
La salida usa fechas de inicio de periodo. `n_draws=0` permite sólo escenarios
de sensibilidad; una sola réplica bootstrap se rechaza. La semilla utiliza un
generador NumPy local.

Una normalización distinta de `raw` requiere
`normalization_window=("2000Q1", "2005Q4")` o `"full_sample"` explícito.
La media y desviación poblacional de referencia se fijan en la señal base y
se aplican a todas las trayectorias. No se recentra cada réplica. Una referencia
constante falla. `bbd_100` sigue la convención existente de media 100 y desviación
50. Para evaluación futura, use sólo fechas de entrenamiento; una normalización
de muestra completa elegida deliberadamente permanece identificada en metadatos.

## Alternativas de puntuación y corpus

Use columnas de puntuación adicionales con
`alternatives={"kernel": {"etiqueta": "columna"}, "lexicon": {...}}`.
`corpus_column="corpus"` genera trayectorias que omiten un corpus cada vez.
Todos los periodos deben conservar un denominador positivo. Todas las alternativas
usan la misma normalización base. Son escenarios de sensibilidad; no se mezclan
con réplicas probabilísticas para producir percentiles. Tampoco se infiere
equivalencia de medición entre idiomas.

## Reestimar con cada trayectoria completa

```python
import pandas as pd
from puremacro.narrative.signals import propagate_signal

def refit(signal):
    # Sustituya la ilustración por su estimador, manteniendo fechas,
    # identificación, unidades y etiquetas en cada llamada.
    return pd.Series({"media": signal.mean(), "cov_rezago1": signal.cov(signal.shift(1))})

propagation = propagate_signal(
    ri, refit, model_label="estimador posterior declarado",
    outcome_units={"media": "unidades de puntuación", "cov_rezago1": "unidades al cuadrado"},
    outcome_convention="media muestral y covarianza de primer rezago",
)
print(propagation.attempts)
percentiles_medicion = propagation.summary("doc", ci=0.90)
```

Cada callback recibe una trayectoria completa y devuelve la misma Serie ordenada
de estimaciones puntuales finitas. El resultado conserva réplicas conjuntas,
fuentes, metadatos y hashes de entrada. El callback controla selección muestral,
identificación y semilla. No devuelva extremos de intervalos como si fueran
estimaciones puntuales independientes.

Las réplicas fallidas conservan posición, tipo de error y mensaje, con columna
de resultados explícitamente ausente. **Cualquier fallo bloquea los resúmenes.**
No se sustituyen por la referencia ni se eliminan automáticamente.
`summary("doc")` muestra percentiles puntuales de medición condicional; otras
fuentes muestran envolventes mínimo/máximo etiquetadas como sensibilidad.
No se suman incertidumbre muestral, de calibración y de medición. Un diseño de
remuestreo conjunto de series temporales/medición exige una decisión metodológica
adicional.

## Entrenar calibración y evaluar en fechas posteriores

```python
from puremacro.narrative.signals import calibrate_signal, evaluate_events

calibrated = calibrate_signal(
    ri, benchmark_alineado, name="referencia externa", target_unit="puntos de referencia",
    train_window=("2000-01-01", "2005-10-01"),
    evaluation_window=("2006-01-01", "2010-10-01"),
)
evaluated = evaluate_events(
    calibrated, eventos_binarios_alineados,
    evaluation_window=("2006-01-01", "2010-10-01"), polarity=1,
)
```

Objetivos y eventos deben coincidir con todas las fechas de la señal: no se
hacen uniones internas, interpolaciones ni eliminaciones de valores faltantes.
Entrenamiento y evaluación necesitan al menos tres observaciones cada uno; la
evaluación debe seguir al entrenamiento, sin solaparse. El mapa afín se estima
sólo con entrenamiento y permanece fijo en todas las réplicas.
`mapping="none"` evalúa unidades ya alineadas sin ajuste. `kind="survey"`
adjunta un `SurveyScore`. La serie calibrada se etiqueta `raw` en unidades del
objetivo y conserva la normalización original en sus metadatos.

Las ventanas conocidas de entrenamiento, incluida la historia de calibraciones,
no pueden solaparse con evaluación de eventos. Sin procedencia de entrenamiento,
la independencia se etiqueta como no verificada; no se puede descartar selección
previa de modelos/puntuaciones con esas fechas.
Correlaciones/AUC indefinidas son `None`, no un cero o ajuste perfecto inventado.
La polaridad del evento es explícita; AUC usa rangos medios para empates y la
selección del decil superior incluye todos los empates en el umbral, informando
`top_n`. La calibración devuelve una copia, conserva metadatos y no sobrescribe
nombres existentes. Tampoco se sobrescribe evidencia previa del panel de eventos.
Son diagnósticos de concordancia/ordenación, no identificación
causal ni probabilidades de evento calibradas automáticamente.
