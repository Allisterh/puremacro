# Auditar un resultado calculado

`puremacro.validation.audit_result` inspecciona un resultado ya resuelto. Presenta
la convergencia del algoritmo por separado de los residuos del presupuesto,
Euler/KKT, distribución y aproximación. No produce una etiqueta universal de
“económicamente correcto”.

```python
from puremacro.validation import audit_result

informe = audit_result(sol)         # solución estacionaria dp, método EGM
print(informe.to_markdown())
tabla = informe.to_dataframe()
texto_json = informe.to_json()
errores = informe.maps["budget_residual"]
```

Cada fila conserva `category`, `check`, `value`, `tolerance`, `units`, `status`,
`evidence_source`, `location` y `details`. Los estados posibles son `passed`,
`failed`, `not_checked` y `unsupported`. La falta de evidencia nunca se convierte
en aprobación. Los informes incluyen metadatos y mapas de residuos; JSON y
Markdown son deterministas. JSON identifica los valores no finitos mediante
cadenas (`"NaN"`, `"Infinity"`).

Por defecto se evalúan `solver`, `equations` y `distribution`. Se recalculan
ecuaciones y un paso de la distribución suministrada; no se resuelve de nuevo,
no se refina, no se deriva, no se normaliza la masa, no se modifican los datos y
no se accede a la red. Las categorías `derivatives` y `refinement` requieren
selección explícita. Las funciones del usuario deben respetar el mismo contrato.

## Resultados admitidos

| Resultado | Alcance |
|---|---|
| `dp.DPSolution` estacionaria, EGM, un activo | Presupuesto y factibilidad declarados, Euler/KKT en políticas continuas, masa, estacionariedad y extremos de la malla. No se incluyen ciclo de vida, elección discreta, HJB o riesgo agregado. Las pruebas numéricas del presupuesto de un activo se identifican como tales; no demuestran identidad simbólica para funciones arbitrarias. |
| `dp.DPSolution` estacionaria, dos activos | Restricciones de ambos activos, presupuesto, valores marginales, costo de ajuste, fecha del depósito, mapas Euler/KKT, percentiles ponderados y diagnósticos de distribución separados. El contexto está en el resultado; `model=otro_modelo_dp` lo contrasta con otra declaración. |
| `dsge.LinearModel` o `KleinSolution` de primer orden | Ecuaciones reconstruidas de las reglas de decisión y cargas de choques, y residuos del estado estacionario. Una solución Klein aislada necesita `model=modelo_lineal`. Un bloque de estados vacío se marca `not_checked`. Las derivadas opcionales mantienen fijo el operador de estado estacionario. No se certifica precisión no lineal global. |
| Estado estacionario o transición de comercio dinámico | Conserva los valores, tolerancias y procedencia del certificado original de 17 pruebas. `model=economia_dinamica` permite reconstruir el certificado estacionario en el `z` suministrado. Los certificados de transición registrados describen el resultado al resolverlo; no verifican automáticamente trayectorias modificadas. |
| `HorizonComparisonResult` de comercio dinámico | Con `checks=("refinement",)`, conserva el alcance de ventana, bienestar y prueba terminal elegida, incluido el modo más débil de riqueza descontada. |

Los certificados de comercio conservan `max(1e-8, result.tol)`; la tolerancia
`equations` general no cambia la afirmación de un certificado heredado.
Otros métodos se identifican como no admitidos. Una función explícita permite
agregar un residuo no lineal o una restricción de cartera:

```python
from puremacro.validation import ResidualCheck

informe = audit_result(
    resultado_no_lineal, model=mi_modelo,
    residuals=[ResidualCheck("equilibrio_no_lineal", recalcular_residuos,
                             tolerance=1e-8, units="bienes")],
)
```

`recalcular_residuos(resultado, modelo)` devuelve residuos de igualdad o
violaciones no negativas de desigualdades. Si falla, queda una prueba fallida
identificada; no se sustituye por la bandera de convergencia.

## Precisión del hogar

En dos activos se reconstruyen los valores marginales de envolvente a partir del
consumo y del costo convexo reconocido; se interpolan en las **políticas continuas
reales** usando la lotería bilineal y probabilidades del modelo. Las derivadas son

\[
g_b=\beta E[V_b(a',b',s')]-u'(c),\qquad
 g_a=\beta E[V_a(a',b',s')]-(1+\chi_d)u'(c).
\]

Se dividen por la utilidad marginal actual. En el interior deben ser cero; en el
límite inferior se exige `g <= 0` y en el superior `g >= 0`. Se informa la parte
que viola estas condiciones. Los límites superiores son truncamientos artificiales.
La clasificación usa la separación local entre nodos, también en celdas estrechas.

Los arreglos siguen `(activo ilíquido, activo líquido, productividad)` aunque se
invierta el orden de declaración. Se muestran índices y coordenadas del peor
error, y percentiles 50/95/99 ponderados por masa usando rango más próximo.
Sin una distribución válida, los percentiles quedan `not_checked`. Masa total,
probabilidad mínima, estacionariedad y masa en ambos extremos superiores se
comprueban por separado. No se corrige ni normaliza la distribución suministrada.

Las tolerancias de evaluación inicial son Euler/KKT `0.01`, ecuaciones `1e-8`,
estacionariedad `1e-10`, error de masa total `1e-12`, masa negativa `1e-14` y masa
en el extremo superior `0.001`. Son decisiones de la aplicación, independientes
del criterio de parada de EGM:

```python
informe = audit_result(sol, tolerances={"euler": 0.002, "boundary_mass": 1e-4})
```

El caso sintético fijo de malla 30×20 en `tests/test_result_audit.py` presenta errores
máximos normalizados de aproximadamente `0.001965` y `0.003831`, con percentiles
99 ponderados de `0.001641` y `0.002088`. La tolerancia iterativa `1e-9` no implica
precisión Euler de `1e-9`. Las pruebas comparan expectativas con interpolación de
SciPy, derivadas con un esquema independiente de cinco puntos y dos políticas
con optimización independiente de Bellman. Las diferencias de objetivo son
aproximadamente `0.001745` y `0.000322`: interpolar valores y valores marginales
son aproximaciones distintas. Es evidencia numérica sintética, no validación
externa ni empírica.

## Experimentos explícitos

```python
from puremacro.validation import DerivativeCheck, RefinementCheck

informe = audit_result(
    sol,
    checks=("solver", "equations", "distribution", "derivatives", "refinement"),
    derivatives=[DerivativeCheck("gradiente_posterior", log_posterior, theta, gradiente)],
    refinements=[
        RefinementCheck("refinar_ambas_mallas", resolver_mallas_finas),
        RefinementCheck("ampliar_limite_iliquido", resolver_limite_iliquido_mayor),
        RefinementCheck("ampliar_limite_liquido", resolver_limite_liquido_mayor),
    ],
)
```

Las derivadas se comparan con diferencias centrales de cinco puntos sobre copias
del vector. Cada función de refinamiento, sin argumentos, devuelve otro resultado.
Los indicadores predeterminados incluyen consumo, activos actuales, máximos y
percentiles Euler/KKT y masas superiores. Se aplica
`abs(nuevo-anterior) <= atol + rtol*abs(nuevo)`, con `atol=1e-8` y `rtol=0.005`.
Cerca de cero se usa el umbral absoluto sin presentar porcentajes engañosos.
Algunos momentos del caso grueso no cumplen el objetivo de 0.5%; converger no
certifica refinamiento.

Para horizontes o entradas seleccionadas del Jacobiano, se proporciona
`RefinementCheck(..., metrics=extraer_indicadores, units={...})`. El extractor debe
devolver el mismo mapa no vacío de escalares para ambos resultados; puede incluir
consumo en la ventana económica, diferencias terminales y `J_C_w[0,0]`. Resolver
nuevamente y calcular Jacobianos pertenece siempre a funciones suministradas
explícitamente; nunca se ejecuta por inferencia.
