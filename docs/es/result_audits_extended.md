> 🇬🇧 [English](../result_audits_extended.md) · 🇪🇸 Español

# Auditorías de transiciones y refinamiento acotado

La [auditoría de resultados](result_audits.md) puede inspeccionar trayectorias
de previsión perfecta y del HANK no lineal de un activo incorporado. Evalúa
la trayectoria suministrada sin resolver otra. Un experimento de refinamiento
explícito puede ejecutar callbacks bajo un presupuesto de trabajo declarado.

También se admiten cierres editables `HANKResult` (`closure='model'`). Se vuelven
a evaluar las ecuaciones agregadas originales y la respuesta lineal del hogar
mediante sus Jacobianos sobre las trayectorias suministradas. En el cierre no lineal,
la comparación con la transición final retenida del hogar exige que sus entradas
sigan coincidiendo con ese experimento. Una entrada modificada falla su comprobación
y deja la consistencia del hogar como `not_checked`; no se resuelve otro hogar.
La masa estacionaria se comprueba por separado. La evidencia micro de Euler,
presupuesto y transporte distributivo ausente sigue como `not_checked`.

## Previsión perfecta

```python
from puremacro.validation import audit_result

result = model.solve_perfect_foresight(periods=80)
report = audit_result(result)
print(report.to_markdown())
errors = report.maps["path_equations"]
```

Los nuevos `PerfectForesightResult` conservan las condiciones inicial y terminal
resueltas, los choques, el orden de variables y la función de ecuaciones. La
auditoría evalúa las ecuaciones originales en cada fecha, incluidas ambas
fronteras. `STEADY_STATE()` usa la referencia fija de la solución. No vuelve a
resolver `endval` ni reconstruye los choques desde un modelo modificado.

El mapa tiene ejes `(periodo, ecuación)`, con periodos desde uno. Una trayectoria
alterada falla aunque conserve `converged=True` y un residual registrado igual
a cero. La distancia entre la última fecha simulada y la frontera terminal es
un diagnóstico de horizonte sin calificación: la frontera se impone en `T+1`.

Para resultados antiguos o construidos manualmente:

```python
from puremacro.validation import PathAuditContext

context = PathAuditContext(
    equations=equations,  # (adelanto, actual, rezago, choque) -> residuales
    y_init=initial_state, y_end=terminal_state,
    exogenous_path=shocks, variable_names=result.variable_names,
)
report = audit_result(result, model=context)
```

Los choques deben tener exactamente una fila por fecha. Si la función declara
explícitamente `steady_state`, recibe el estado terminal fijo. El usuario
vincula los parámetros de una función sin modelo. Sin contexto suficiente,
la ecuación queda `not_checked`. Los choques sorpresivos secuenciales y MCP
tienen contratos distintos y no están cubiertos por este adaptador; tampoco
se atribuye una distribución a un modelo de agente representativo.

## HANK no lineal

```python
from puremacro.models.hank_sequence_space import solve_nonlinear_transition

transition = solve_nonlinear_transition(
    steady_state, shock_seq=monetary_shocks, horizon=80,
    retain_audit_paths=True,
)
report = audit_result(transition)
```

Se conservan los `phi_pi`, `kappa` y parámetros del hogar efectivos, incluidas
las sustituciones de parámetros. Aun sin retener políticas por fecha, se
recalculan las identidades de bienes, Phillips y Taylor/Fisher a partir de
las trayectorias reportadas. La inflación terminal desviada es cero según
el cierre incorporado; no se está verificando un bloque `.mod` arbitrario.

`retain_audit_paths=True` guarda el último recorrido del hogar ya calculado:
consumo, ahorro, distribuciones de entrada y terminal, retornos realizados,
salarios y consumo de continuación. No agrega otra solución EGM. Los arreglos
principales requieren aproximadamente `8*(3*T+2)*n_activos*n_productividad`
bytes, además de los insumos. El valor predeterminado es `False`; la evidencia
faltante queda `not_checked`. Los mapas de auditoría también ocupan memoria
proporcional al tamaño de la trayectoria.

Con esos datos se comprueban:

- Presupuesto por fecha y estado, consumo positivo, dominio de activos y
  correspondencia entre retorno ex ante y retorno realizado una fecha después.
- Agregación independiente del consumo y equilibrio de bienes, incluido el
  gasto público y el impuesto que financia intereses de deuda.
- Euler/KKT mediante interpolación de utilidad marginal en el ahorro continuo
  observado. La última fecha usa el consumo terminal conservado. Los percentiles
  ponderan fechas por igual y usan la masa suministrada en cada fecha.
- Masa total y negativa, distribución inicial, transporte por lotería,
  conservación de activos medios y masa en la frontera superior artificial.

No se normaliza masa, recorta política ni resuelve otro hogar. El ejemplo
sintético de 25 puntos y ocho fechas tiene residuales de bienes próximos a
`2e-12`, error Euler alrededor de `0.0023` y masa superior de `0.0056`, que falla
el umbral predeterminado `0.001`. Son observaciones de esa calibración, no cotas
universales del error.

## Escaleras explícitas de refinamiento

`RefinementCheck` compara un candidato con la base. `RefinementLadder` compara
resoluciones consecutivas y conserva cada intento:

```python
from puremacro.validation import RefinementLadder, RefinementLevel

ladder = RefinementLadder(
    "horizonte",
    levels=[
        RefinementLevel("T160", solve_T160, settings={"horizon":160}, cost=2),
        RefinementLevel("T320", solve_T320, settings={"horizon":320}, cost=4),
        RefinementLevel("T640", solve_T640, settings={"horizon":640}, cost=8),
    ],
    metrics=extract_economic_metrics,
    max_solves=3, max_cost=14, consecutive_passes=2,
    rtol=0.005, atol=1e-8,
    validate_pair=check_same_economic_experiment,
)
report = audit_result(baseline, checks=("refinement",), refinement_ladders=[ladder])
print(report.metadata["refinement_ladders"]["horizonte"]["stop_reason"])
```

`metrics(result)` es obligatorio: devuelve un diccionario no vacío de escalares
finitos, con las mismas claves en cada resolución. Elija consumo en una ventana
económica fija, riqueza, bienestar, brecha terminal, errores Euler o entradas
del jacobiano. Para unidades distintas, normalice explícitamente las métricas
o use escaleras separadas. No se eligen automáticamente mallas ni horizontes.

Cada métrica debe satisfacer `abs(nuevo-anterior) <= atol + rtol*abs(nuevo)`.
Se exige el número indicado de comparaciones consecutivas aprobadas. Un cambio
excesivo reinicia la cuenta; los intentos anteriores fallidos siguen visibles.
Un callback fallido, no convergencia declarada, valores no finitos, claves
distintas o rechazo de la pareja detienen el experimento sin certificar
estabilidad. La estabilidad se refiere solamente a las métricas elegidas.

`validate_pair(anterior,nuevo)` puede rechazar cambios de especificación,
fronteras o choques. Sin este callback, la equivalencia del experimento se
registra como `not_checked`: `settings` es procedencia declarada por el usuario,
no una inspección de la función que resolvió el modelo.

`max_solves` cuenta callbacks intentados, incluso fallidos. Cada intento consume
su `cost` positivo declarado; no comienza si excedería `max_cost`. Estas unidades
no son segundos ni memoria y no ofrecen un límite temporal duro: no se puede
interrumpir un callback en curso. La parada distingue `stable_metrics`,
`solve_budget_exhausted`, `cost_budget_exhausted`, `levels_exhausted` y errores
nombrados. JSON, Markdown y los metadatos del DataFrame conservan presupuestos,
intentos y métricas. Si no se selecciona `refinement`, no se ejecuta ninguno de
los callbacks de métricas o solución.
