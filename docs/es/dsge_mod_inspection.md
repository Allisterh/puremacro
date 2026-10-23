# Inspección y ejecución de programas `.mod`

`inspect_mod` describe la sintaxis y el soporte operativo de un archivo Dynare
sin evaluar la calibración o el estado estacionario, compilar residuos o
derivadas, resolver, estimar, simular, importar herramientas gráficas ni cambiar
el generador aleatorio. Lee el archivo y sus inclusiones locales explícitas.
El informe no certifica determinación, precisión numérica ni validez económica.

```python
from puremacro.dsge import inspect_mod
informe = inspect_mod("estudio.mod", ordered=True)
print(informe.summary())
print(informe.to_json())
```

```bash
puremacro-dynare inspect estudio.mod --ordered
puremacro-dynare inspect estudio.mod --ordered --format json
puremacro-dynare inspect estudio.mod --entry-point load_mod --output informe.txt
```

Solo `--output` solicita escribir un archivo. Los códigos de salida son `0`
para cobertura completa sin hallazgos, `1` para incompatibilidades o datos del
usuario pendientes, y `2` para inspección incompleta o entrada ilegible.
Texto y JSON presentan los mismos hallazgos; JSON incluye los hashes SHA-256,
las opciones y toda la procedencia. No se añaden fechas de ejecución.

El punto de entrada predeterminado es `execute_mod_commands`; también se pueden
seleccionar `load_mod` y `load_hank_mod`. Cargar el modelo no ejecuta sus
simulaciones, suavizados o descomposiciones. Una opción aceptada por el analizador
solo aparece como soportada si el punto de entrada la utiliza. Los scripts,
comandos desconocidos, opciones ignoradas, `EXPECTATION` y `STEADY_STATE` sobre
variables exógenas generan hallazgos explícitos. `hetagent_block` se identifica
como extensión de puremacro que requiere la ruta HANK. Su cargador usa el hogar
incorporado configurado salvo que se proporcione `household_model`; los campos
retenidos pero no utilizados por HANKModel generan hallazgos separados.

Cada entrada conserva posición original y expandida, fragmento, estado,
opciones explícitas y selección de variables. Las ecuaciones y variables
locales son entradas hijas del bloque `model`. También se inventarían los
comentarios. Las directivas macro, incluso las ramas inactivas, aparecen después
del inventario expandido con sus posiciones originales. Las inclusiones
conservan su cadena de origen; las iteraciones tienen identificadores distintos
aunque provengan de la misma línea. Si una interpolación impide recuperar la
columna exacta, esta es `null`; los fragmentos con varios orígenes conservan los
tramos individuales.

La expansión macro es la única evaluación de expresiones durante la inspección.
Reutiliza el motor existente con presupuesto `macro_limit=100_000` y límites de
2 MB para fuentes y expansión. Un límite excedido, inclusión fallida, bloque
abierto o sintaxis malformada produce cobertura incompleta. Las expresiones de
calibración permanecen sin evaluar: un informe sin incompatibilidades no prueba
que `log(-1)` tenga sentido ni que el modelo vaya a resolverse.

`data=` permite comprobar columnas, conversión numérica, duplicados e infinitos
sin evaluar la verosimilitud. Se permiten NaN. Los datos pendientes se reportan
como entrada requerida; `datafile` no carga archivos de forma implícita.

## Ejecución ordenada opcional

Para rechazar programas incompatibles antes de resolver:

```python
from puremacro.dsge import load_mod
modelo = load_mod("estudio.mod", command_mode="ordered")
resultados = modelo.execute_mod_commands(ordered=True, data=datos)
for ejecucion in resultados.invocations:
    print(ejecucion.command, ejecucion.explicit_options,
          ejecucion.effective_options, ejecucion.requested_variables)
    completo = ejecucion.result
    seleccionado = ejecucion.selected_result
```

Omita `data` si solo simula. La carga ordenada devuelve un `LinearModel` de primer
orden; cada comando elige su propio orden de perturbación. No pase `order=2` o
`order=3` al cargador ordenado. El modelo utiliza exactamente la fuente expandida
que se inspeccionó y conserva ese informe aunque luego cambie una inclusión.
Los programas macro requieren `command_mode="ordered"`. Para fuentes sin macros,
también puede usar `execute_mod_commands(ordered=True)` en un modelo ya cargado,
pero en ese caso la resolución de carga ya ocurrió.

El subconjunto estocástico admite `stoch_simul`, `calib_smoother` y
`shock_decomposition` repetidos con calibración fija. La extensión de desarrollo
5.7.0 también reproduce bloques de frontera, declaraciones de choques y comandos
repetidos de preparación/resolución de previsión perfecta, incluidos programas
deterministas y estocásticos combinados. Use
`read_mod(source).execute_mod_commands()` para empezar sin resolver implícitamente
el estado estacionario ni la perturbación; la [guía de modelos no lineales](dsge_nonlinear.md)
describe las opciones y las instantáneas admitidas. Las reasignaciones intercaladas
de parámetros y los comandos de política y HANK siguen fuera del soporte ordenado.
Las incompatibilidades conocidas, el esquema de datos y los intervalos muestrales
se validan antes del primer comando numérico.

| Comando | Opciones explícitas soportadas | Valores iniciales |
|---|---|---|
| `stoch_simul` | `order`, `irf`, `periods`, `ar`, `drop`, `hp_filter`, `pruning`, `nograph` | order 2, irf 40, periods 0, ar 5, drop 100, sin filtro; nograph debe estar activo. |
| `calib_smoother` | `prefilter`, `first_obs`, `nobs` | Sin prefiltrado; desde observación 1 hasta el final. |
| `shock_decomposition` | `first_obs`, `nobs` | Muestra solicitada y sistema de estados completo. |

Las opciones deben ser literales. Los valores simbólicos se reportan como no
resueltos. `first_obs` comienza en 1; `nobs` es una cantidad.
`execute_mod_commands(sigma=...)` aplica la sustitución de desviaciones estándar
a los tres métodos numéricos.

Las opciones soportadas persisten dentro del estudio, con dos excepciones:
**omitir `order` en cada `stoch_simul` lo restablece a 2**, y **omitir la lista de
variables selecciona todas las salidas disponibles de esa invocación**.
`ar`, `irf`, filtros y ajustes de simulación persisten. `first_obs` y `nobs` son
compartidos entre suavizado y descomposición; `prefilter` también persiste.
Como la descomposición nativa no admite prefiltrado, se rechaza su uso después
de `calib_smoother(prefilter=1)` hasta que un suavizado anterior lo restablezca
explícitamente a `prefilter=0`. Cada ejecución independiente inicia con opciones
nuevas.

```text
stoch_simul(order=1, irf=3, ar=1) y;
stoch_simul(order=1, irf=7) x;
stoch_simul(irf=9);
```

Genera tres entradas con `(order, irf, ar)` iguales a `(1,3,1)`, `(1,7,1)` y
`(2,9,1)`, para `y`, `x` y todas las variables. La presentación seleccionada
filtra momentos e IRF pero conserva las reglas de decisión completas. El
suavizado solo puede presentar estados y observables disponibles; otras
selecciones se rechazan antes de ejecutar. La descomposición filtra sus tablas
de componentes.

Cada `DynareCommandInvocation` conserva procedencia, opciones explícitas y
efectivas, selección, huella de calibración/datos, estado y resultados.
`.stoch_simul`, `.smoother` y `.shock_decomposition` mantienen el último resultado
completo exitoso de su tipo. `ModExecutionError.results` conserva las entradas
completadas y la fallida; `ModCompatibilityError.report` contiene el informe
previo. Los valores predeterminados heredados siguen agregando por categoría:
`load_mod(...)` y `execute_mod_commands()` no cambian de comportamiento.

## Evidencia y límites

`tests/fixtures/dynare_commands/` contiene la fuente, un fragmento de programa
MATLAB generado por Dynare 7.0 y su hash. Verifica el restablecimiento de `order`,
la persistencia de `ar`/`irf`, listas de variables nuevas y opciones muestrales
compartidas. Es evidencia del preprocesador, no una comparación numérica de
Dynare. Las pruebas comparan además cada invocación puremacro con su método
Python directo y cubren fallos parciales y secuencias incompatibles.

El [manual Dynare](https://www.dynare.org/manual/the-model-file.html#stoch_simul)
y las referencias de código descritas en la [versión inglesa](../dsge_mod_inspection.md)
explican por qué los valores predeterminados no implican restablecer todas las
opciones. No se promete paridad general de programas Dynare, compatibilidad con
el estado global MATLAB `options_`, ni ejecución de scripts/modelos mutables.
Después de resolver, use `check`, `resid` y `model_diagnostics` para los
diagnósticos numéricos y del modelo.
