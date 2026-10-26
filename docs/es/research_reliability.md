# Fiabilidad de investigación y próximos hitos de PureMacro

El orden acordado es **fiabilidad y replicaciones**, después **modelos y
aplicaciones de política**, y finalmente **docencia y facilidad de uso**.

## Rama de correcciones actual

La etiqueta fuente v6.0.0 no llegó a PyPI. Según la comprobación del 2026-10-26,
la distribución publicada es 5.7.1. Estas correcciones requieren el código de
desarrollo; una nueva publicación aún necesita la matriz CI y sus controles.

Los adaptadores conservan ecuaciones y cargas del modelo suministrado.
DSGE usa sensibilidades de Kalman; DAG usa diferencias centrales de Whittle
nativo manteniendo fijo el estado estacionario. Las operaciones no compatibles
producen errores; `model=None` selecciona demostraciones con advertencia.
Consulte [el alcance del protocolo](autodiff_and_protocol.md).

El DAG HANK predeterminado prescribe respuestas de hogares; el ejemplo de riesgo
agregado prescribe momentos y matrices. Son demostraciones, no distribuciones
de hogares resueltas. `asymmetry_ratio` mide el pico no lineal respecto al lineal,
no dos soluciones con choques opuestos. Consulte [el alcance HANK](hank_ssj_and_aggregate_risk.md).

Las correcciones soberanas incluyen default determinista, costos basados en
ingreso medio estacionario, reingreso con deuda cero y estadísticas simuladas
sin ajustes de signo. La correlación de un diferencial constante es `NaN`, con
metadatos que explican por qué no está definida.

## Primeras pruebas

Desde la raíz del repositorio, con Python 3.11+ y `.[dev,notebooks]`:

```sh
python -m pytest tests/test_protocol_model_fidelity.py tests/test_research_reliability_references.py tests/test_sovereign/test_reference_reliability.py -q
python tools/release_check.py --no-tests
```

El cuaderno **72** ofrece una visita interactiva bilingüe. **73** explora deuda
soberana; **74** combina LP distributivas sintéticas, inferencia gaussiana y
transiciones reducidas. Sus cuadros verifican propiedades concretas, no toda
la validez económica del modelo.

## Evidencia independiente disponible

| Comparación | Resultado en las mallas probadas | Pendiente |
|---|---|---|
| Ecuaciones Arellano de deuda corta y default determinista | Valores coinciden dentro de 4e-11; también precios, decisiones de default y políticas de deuda | Calibración publicada, momentos, convergencia y extensión de deuda larga |
| Bloque de hogares de un activo | Políticas y consumo coinciden con los autores SSJ dentro de 5e-11; jacobianos de rendimientos dentro de 4e-4 | Convergencia económica de mallas, equilibrio publicado y transiciones no lineales |

Las referencias, hashes y procedencia están en
`tests/fixtures/research_reliability/`. Los generadores están en
`tools/reference_validation/`. SSJ externo no es dependencia de ejecución.
Coincidir en una malla común no demuestra precisión de discretización.

## Siguiente: un estudio soberano completo

1. Igualar una calibración original de Arellano y su proceso de ingreso.
   Publicar objetivos y resultados de deuda/producto, frecuencia de default,
   diferenciales, volatilidad del consumo y ciclicidad comercial. Explicitar
   condicionamiento de simulación y anualización; incluir intervalos muestrales.
2. Refinar mallas e incrementar la simulación. Revisar residuos de Bellman,
   precios, estabilidad de políticas y masa en los extremos. Conservar las
   correlaciones simuladas originales, incluso signos inesperados.
3. Estudiar volatilidad, persistencia del ingreso y probabilidad de reingreso
   en el modelo validado de deuda corta. Reportar bienestar e incertidumbre
   numérica antes de agregar vencimientos y reestructuración no contrastados.

Así se obtiene un resultado revisable desde ecuaciones hasta tablas de política.

Los diagnósticos de colateral y paradas repentinas requieren corrección y una
referencia independiente: la amplificación incluye un valor de sustitución
heredado y la rama de restricción holgada prescribe diagnósticos. El residuo de
la estructura temporal es contable, no una prima de riesgo identificada.
El cuaderno 73 y la [guía soberana](sovereign_default.md) explican estos límites.

## Después: incidencia sobre hogares y docencia

Completar una referencia publicada HANK de un activo igualando reglas fiscales,
fechas de rendimientos y cierre de equilibrio. Validar transiciones no lineales
antes de enlazar deciles de hogares con LP de datos observados. El riesgo
agregado general de orden superior viene después de esas comprobaciones.

Convertir el estudio validado en una ruta bilingüe breve: instalar un wheel
verificado, reproducir una base, cambiar un supuesto y revisar diagnósticos.
Mantener explícita la distinción entre demostraciones y resultados empíricos.
