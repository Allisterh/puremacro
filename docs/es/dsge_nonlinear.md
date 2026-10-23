> 🇬🇧 [English](../dsge_nonlinear.md) · 🇪🇸 Español

# Una especificación no lineal y métodos de solución explícitos

`read_mod` compila un archivo `.mod` en un `NonlinearModel` sin calcular un
estado estacionario ni resolver una aproximación local de expectativas
racionales. Después se elige el problema numérico. La misma especificación
proporciona residuos no lineales para previsión perfecta, perturbación y un
problema de equilibrio global construido explícitamente.

Para elegir métodos y seguir la secuencia del curso, comience con la
[guía de flujos DSGE](dsge_workflows.md).

| Tarea | Punto de entrada | Qué se calcula |
|---|---|---|
| Inspeccionar sintaxis y compatibilidad de comandos | `inspect_mod(source, ordered=True)` | Un informe; ninguna solución numérica |
| Conservar ecuaciones y calibración | `read_mod(source, params=...)` | Un `NonlinearModel` compilado; pueden evaluarse declaraciones |
| Encontrar un equilibrio estático | `model.steady(...)` | Un diccionario verificado de niveles endógenos |
| Obtener una regla local de política | `model.solve(order=1)`, `2` o `3` | La solución de perturbación existente |
| Resolver una trayectoria determinista especificada | `model.solve_perfect_foresight(...)` | Valores endógenos en las fechas 1 a T |
| Ejecutar comandos admitidos en orden | `model.execute_mod_commands()` | Un historial con copias conservadas de los resultados |
| Mantener el flujo habitual de carga y solución | `load_mod(source, order=...)` | Una solución local inmediata |

Un problema finito de frontera puede tener solución aunque el método de regla
local rechace el modelo. Por otra parte, compilar correctamente no demuestra
existencia, unicidad, determinación ni convergencia. La previsión perfecta sigue
requiriendo un sistema bien planteado, fronteras apropiadas y una trayectoria
numérica factible.

## Comenzar con una trayectoria determinista exacta

```python
import numpy as np
from puremacro.dsge import read_mod, NonlinearModel

model = read_mod("""
var x;
model; [name='accumulation'] x=0.5*x(-1); end;
initval; x=1; end;
""")
assert model.shocks == ()
assert model.equation_names == ("accumulation",)
steady = model.steady()
path = model.solve_perfect_foresight(y_init=[1], y_end=[0], periods=8)
assert path.converged
np.testing.assert_allclose(path.path.x, 0.5 ** np.arange(1, 9), atol=1e-10)
```

Es depreciación sin inversión: el acervo en la fecha cero es 1 y cada periodo
conserva la mitad del anterior. No hace falta declarar `varexo`. `path.path`
contiene ocho filas y excluye las fronteras suministradas en las fechas cero y
nueve. En este ejemplo retrospectivo el valor terminal no aparece en ninguna
ecuación; una ecuación prospectiva sí lo utilizaría.

`steady()` devuelve el equilibrio estático verificado, aquí `{"x": 0.0}`. No
modifica `model.guess` ni `model.steady_state`. Este último, cuando existe, es
una declaración analítica suministrada, no un certificado de convergencia
numérica. Pase los valores devueltos explícitamente cuando sean necesarios
como fronteras o referencia de un cálculo posterior.

Un problema finito de frontera puramente prospectivo tampoco requiere una
regla local previa. Aquí el precio terminal especificado es 1 en la fecha 5;
la sustitución hacia atrás determina los cuatro precios presentados. El valor
inicial no aparece en esta ecuación:

```python
forward_only = read_mod("var p; model; p=2*p(+1); end;")
finite_path = forward_only.solve_perfect_foresight(
    y_init=[0], y_end=[1], periods=4,
)
assert finite_path.converged
np.testing.assert_allclose(finite_path.path.p, [16, 8, 4, 2], atol=1e-10)
```

El ejemplo establece una trayectoria finita con el valor terminal suministrado;
no selecciona una solución económica de horizonte infinito.

## Migrar desde `load_mod` de forma explícita

Las llamadas existentes a `load_mod` conservan su comportamiento. Use
`read_mod` cuando la siguiente tarea requiera las ecuaciones originales sin
exigir primero una solución local. Solicite esta última explícitamente para
obtener reglas de decisión, respuestas a impulsos o momentos:

```python
from puremacro.dsge import load_mod

local_source = """
var x; varexo e; parameters rho; rho=0.7;
model; x=rho*x(-1)+e; end;
initval; x=0; end;
shocks; var e; stderr 0.1; end;
"""
legacy = load_mod(local_source, order=1)
specification = read_mod(local_source)
local = specification.solve(order=1)
np.testing.assert_allclose(local.decision_rules().ghx,
                           legacy.decision_rules().ghx, atol=1e-12)
assert specification.steady_state is None
```

Hay tres detalles que conviene conservar al migrar:

- `read_mod(...).solve()` usa **orden 1** de forma predeterminada. No selecciona
  el orden de un comando `stoch_simul`. Indique `order=2` o `3` cuando esa sea la
  aproximación deseada. `load_mod(order=None)` conserva su selección de orden
  a partir del comando.
- La ejecución ordenada es predeterminada en `NonlinearModel`. Para un modelo
  devuelto por `load_mod`, indique `execute_mod_commands(ordered=True)`;
  cargar con `command_mode="ordered"` realiza la inspección de compatibilidad
  antes de la solución local. En una tarea exclusivamente determinista,
  `read_mod` evita ese cálculo adicional.
- `read_mod(params=...)` vuelve a evaluar las expresiones de estado estacionario
  analítico, valores iniciales y covarianzas con la calibración combinada. Las
  expresiones de choques por fecha también usan esa calibración al ejecutarse.
  `load_mod` conserva su comportamiento previo para la covarianza; use su
  argumento `shock_cov=` para modificarla explícitamente.

Calibre todos los parámetros y sustituya únicamente nombres declarados. Una
sustitución no repite las asignaciones dependientes fuera de los bloques: si
el archivo contiene `r=0.04; beta=1/(1+r);`, sustituir solo `r` deja `beta` en su
valor calibrado. Sustituya ambos o use una expresión local `#` dependiente de
parámetros cuando una identidad deba permanecer simbólica. Para otra
calibración, cree otra especificación con `read_mod(source, params=...)`; esto
también mantiene clara la procedencia del experimento.

Un `Path` o una cadena sin punto y coma ni saltos de línea identifica un archivo.
El código fuente con inclusiones relativas de macros puede usar `base_dir=`.
La lectura conserva una copia del código expandido: cambiar después un archivo
incluido no modifica un programa ya leído. Vuelva a leerlo para usar el nuevo
contenido.

## Convención temporal de los arreglos y contrato de funciones

Con n variables endógenas compiladas y m choques, la interfaz de residuos es

$$F(y_{t+1},y_t,y_{t-1},e_t;\theta,\bar y)=0.$$

| Argumento o resultado | Dimensión | Significado |
|---|---|---|
| `lead` | `(n,)` | Todos los valores endógenos en t+1 |
| `current` | `(n,)` | Todos los valores endógenos en t |
| `lag` | `(n,)` | Todos los valores endógenos en t−1 |
| `shocks` | `(m,)` | Valores exógenos/innovaciones actuales, en el orden declarado |
| `steady_state` | `(n,)` si se proporciona | Referencia endógena fija para `STEADY_STATE()` |
| Residuo devuelto | `(n,)` | Un residuo por ecuación compilada, en el orden de ecuaciones |
| Trayectoria PF de `shocks` | `(T, m)` | Valores exógenos en las fechas 1 a T |
| PF `initial_path` | `(T, n)` | Punto inicial de Newton, separado de las fronteras |

`model.residuals(...)` acepta estos arreglos unidimensionales o diccionarios
con todos los nombres requeridos. Evalúa **una fecha**, no un lote de fechas.
Si no hay choques, pase `[]` o un arreglo de dimensión `(0,)`. Los residuos
conservan entradas complejas, de modo que las funciones suaves apropiadas
pueden utilizar diferenciación de paso complejo. Evite convertir sus entradas
a `float` o usar valores absolutos y bifurcaciones al solicitar ese método.

El modelo usa la convención temporal escrita en sus ecuaciones. Por ejemplo,
`y=exp(a)*k(-1)^alpha; k=(1-delta)*k(-1)+i;` define `k` como acervo al final
del periodo; la producción en t utiliza el acervo heredado. Pase el vector
endógeno completo en cada fecha, no solo los estados seleccionados por un
método de perturbación.

`from_equations` envuelve la misma interfaz. La función recibe cuatro arreglos
posicionales y puede declarar explícitamente `params` y/o `steady_state`; esos
argumentos opcionales se pasan solo cuando aparecen por nombre en su firma.
`params` es un diccionario y `steady_state` es un arreglo o `None`. La función
proporciona las ecuaciones ya expresadas como residuos iguales a cero:

```python
def accumulation(lead, current, lag, shocks, *, params):
    return current - params["rho"] * lag

callback = NonlinearModel.from_equations(
    accumulation, variables=["x"], params={"rho": 0.5},
    equation_names=["accumulation"], guess=[0],
)
other_path = callback.solve_perfect_foresight(
    y_init=[1], y_end=[0], periods=8, linear_solver="krylov",
)
np.testing.assert_allclose(other_path.path, path.path, atol=1e-10)
```

`variables` no puede estar vacío, y los nombres endógenos y de choques deben
ser distintos. `equation_names`, si se proporciona, contiene un nombre único
por variable. `timing` registra la convención declarada; no desplaza ni
reescribe una función Python. `guess`, `steady_state` y `shock_cov` añaden
metadatos explícitos de valores iniciales, referencia y covarianza. Un modelo
creado mediante una función no tiene código `.mod` que ejecutar; use sus
métodos numéricos directamente.

Un sistema retrospectivo/prospectivo muestra por qué importan los tres vectores
endógenos. Sea `z` un AR(1) y `q` su valor descontado determinista, con valor
terminal cero. La función y la especificación `.mod` siguientes producen la
misma trayectoria de horizonte finito:

```python
def valuation_equations(lead, current, lag, shocks, *, params):
    z, q = current
    return np.array([
        z - params["rho"] * lag[0] - shocks[0],
        q - z - params["beta"] * lead[1],
    ])

valuation = NonlinearModel.from_equations(
    valuation_equations, variables=["z", "q"], shocks=["e"],
    params={"rho": 0.5, "beta": 0.8},
    equation_names=["state", "valuation"],
    timing={"z": [-1, 0], "q": [0, 1]}, guess=[0, 0],
)
valuation_mod = read_mod("""
var z q; varexo e; parameters rho beta; rho=0.5; beta=0.8;
model;
[name='state'] z=rho*z(-1)+e;
[name='valuation'] q=z+beta*q(+1);
end;
""")
innovations = np.zeros((8, 1))
innovations[0, 0] = 1
arguments = dict(y_init=[0, 0], y_end=[0, 0], periods=8, shocks=innovations)
callback_path = valuation.solve_perfect_foresight(**arguments)
compiled_path = valuation_mod.solve_perfect_foresight(**arguments)
assert callback_path.converged and compiled_path.converged
np.testing.assert_allclose(callback_path.path, compiled_path.path, atol=1e-10)
expected_z = 0.5 ** np.arange(8)
expected_q = np.zeros(9)
for t in range(7, -1, -1):
    expected_q[t] = expected_z[t] + 0.8 * expected_q[t + 1]
np.testing.assert_allclose(compiled_path.path.z, expected_z, atol=1e-10)
np.testing.assert_allclose(compiled_path.path.q, expected_q[:-1], atol=1e-10)
```

### Nombres, referencias y ecuaciones ampliadas

| Atributo | Interpretación |
|---|---|
| `variables`, `shocks` | Orden de arreglos utilizado por la especificación compilada |
| `parameter_values` | Calibración fija actual |
| `equation_names` | Etiquetas como `[name='Euler']` o nombres generados |
| `timing` | Fechas endógenas presentes en las ecuaciones compiladas |
| `equation_sources` | Ubicaciones de las ecuaciones originales |
| `source_hashes` | Huellas que identifican el código y sus archivos incluidos |

El analizador puede ampliar los adelantos/rezagos mayores y las variables
exógenas fechadas mediante ecuaciones auxiliares. Por ello, `model.variables`
puede ser más largo que la lista `var` original. Revíselo antes de construir
arreglos; las filas auxiliares pueden carecer de una ubicación propia en las
ecuaciones originales. Para historias no constantes, proporcione por Python
un estado ampliado coherente. El subconjunto ordenado de fronteras acepta solo
valores sin índice o de fecha cero, no un calendario `histval` completo.

Para auditar residuos dinámicos, proporcione explícitamente una referencia
fija `steady_state` cuando las ecuaciones contengan `STEADY_STATE()`. Si se
omite, `residuals` usa el vector corriente para ese operador, apropiado en un
cálculo estático. El atributo `model.steady_state` no se sustituye implícitamente
en `residuals`. En una solución de previsión perfecta, `y_end` proporciona
tanto la frontera terminal como la referencia fija:

```python
reference_model = read_mod("""
var p;
model; p=0.5*p(+1)+0.5*STEADY_STATE(p); end;
""")
fixed = reference_model.residuals([3], [2], [0], [], steady_state=[1])
static_reference = reference_model.residuals([3], [2], [0], [])
np.testing.assert_allclose(fixed, [0])
np.testing.assert_allclose(static_reference, [-0.5])
```

## Estados estacionarios y dominios positivos

`model.steady(exo_values=..., guess=..., tol=...)` resuelve las ecuaciones
estáticas originales y verifica sus residuos con `tol`. `exo_values` es un
diccionario por nombre de choque o un vector en ese orden; fija los niveles
exógenos constantes del equilibrio. `guess` es un punto inicial, no una
restricción sobre la solución. Sin un punto explícito, el método utiliza una
declaración analítica y después `initval`, cuando están disponibles.

Para logaritmos, recíprocos o potencias fraccionarias, proporcione un punto
factible. En un programa ordenado, `steady_state_model` suministra el punto
analítico incluso si falta `initval`. Las ecuaciones se resuelven y verifican
con los valores exógenos activos. Así, un nuevo nivel exógeno terminal no puede
sustituirse silenciosamente por el equilibrio inicial.

Este modelo de nivel positivo tiene la relación estática exacta
`x=center*exp(e/0.8)`. El centro sustituido es 3 y el choque permanente terminal
aumenta el estado estacionario a 12:

```python
positive_program = read_mod("""
var x; varexo e; parameters center; center=2;
model; log(x)=0.2*log(x(-1))+0.8*log(center)+e; end;
steady_state_model; x=center; end;
steady;
endval; e=0.8*log(4); end;
steady;
perfect_foresight_setup(periods=3);
perfect_foresight_solver(tolf=1e-10);
""", params={"center": 3})
positive_results = positive_program.execute_mod_commands()
static_results = [entry.result for entry in positive_results.invocations
                  if entry.command == "steady"]
np.testing.assert_allclose([value.x for value in static_results], [3, 12])
expected = 3 * 4 ** (1 - 0.2 ** np.arange(1, 4))
np.testing.assert_allclose(positive_results.perfect_foresight.path.x,
                           expected, atol=1e-8)
```

`perfect_foresight_setup(endval_steady, periods=...)` puede solicitar la
solución estática terminal durante la preparación. Esta opción nativa prepara
una frontera verificada; no promete reproducir todo el algoritmo de homotopía
o de estado estacionario terminal de Dynare. Para continuación no lineal y
comprobaciones independientes de horizonte, use el
[flujo de transiciones](dsge_transitions.md).

## Ejecutar comandos deterministas en orden

Los comandos emplean conceptos descritos en las secciones de
[condiciones iniciales y terminales](https://www.dynare.org/manual/the-model-file.html#initial-and-terminal-conditions)
y [simulación determinista](https://www.dynare.org/manual/the-model-file.html#deterministic-simulation)
del manual de Dynare. Las tablas siguientes definen el subconjunto nativo de
puremacro. Dynare ofrece otras formas de comandos y algoritmos adicionales.

El primer experimento es transitorio. El segundo modifica el nivel exógeno
constante y calcula el estado estacionario terminal correspondiente. Un choque
redeclarado después del primer setup muestra deliberadamente el efecto de
conservar una copia de las entradas:

```python
program = read_mod("""
var x; varexo e; parameters rho; rho=0.5;
model; x=rho*x(-1)+e; end;
initval; x=0; e=0; end;
steady;
shocks; var e; periods 1; values 1; end;
perfect_foresight_setup(periods=4);
shocks; var e; periods 1; values 9; end;
perfect_foresight_solver(tolf=1e-10);
endval; x=8; e=1; end;
steady;
shocks(overwrite); end;
perfect_foresight_setup(periods=8);
perfect_foresight_solver;
""")
results = program.execute_mod_commands()
solves = [entry for entry in results.invocations
          if entry.command == "perfect_foresight_solver"]
np.testing.assert_allclose(solves[0].result.path.x, [1, .5, .25, .125], atol=1e-10)
np.testing.assert_allclose(solves[1].result.path.x,
                           2 * (1 - .5 ** np.arange(1, 9)), atol=1e-10)
print(results.to_frame())
```

La primera trayectoria sigue comenzando en 1: su setup se preparó antes de
declarar el valor 9. El segundo utiliza el equilibrio terminal actualizado y
las declaraciones por fecha ya borradas. `endval; x=8; e=1; end; steady;`
produce `x=2` terminal, pues `steady` resuelve `x=0.5*x+1`. Omitir ese `steady`
conservaría el valor terminal literal 8.

### Estado de las fronteras y los choques

| Sentencia | Efecto al alcanzarla |
|---|---|
| `initval` | Inicia un nuevo experimento inicial/terminal; asigna cero a valores endógenos y exógenos omitidos; borra asignaciones históricas y terminales |
| `steady` antes de `endval` | Sustituye valores endógenos iniciales por el equilibrio estático con los niveles exógenos iniciales activos |
| `histval` | Proporciona la frontera endógena histórica y sustituye allí valores iniciales; admite valores sin índice o de fecha cero |
| `endval` | Actualiza valores terminales; las omisiones heredan el contexto terminal previo o los valores iniciales |
| `steady` después de `endval` | Sustituye valores endógenos terminales por el equilibrio con los niveles exógenos terminales |
| `shocks` | Acumula asignaciones por fecha y declaraciones de covarianza; redeclarar una fecha sustituye solo esa fecha |
| `shocks(overwrite)` | Borra choques por fecha y covarianzas previas antes de leer el nuevo bloque |
| `perfect_foresight_setup` | Copia fronteras, trayectoria exógena y punto inicial para el horizonte elegido |
| `perfect_foresight_solver` | Resuelve el experimento preparado; repetirlo tras converger utiliza la trayectoria anterior como punto inicial |
| `simul` | Prepara y resuelve el experimento actual en un comando |

El ejecutor comienza con valores iniciales cero. En este subconjunto, un
`endval` aislado no fija la frontera histórica: use `initval` o un `steady`
inicial explícito. `initval` no borra las declaraciones de choques; use
`shocks(overwrite)` cuando el nuevo experimento deba descartarlas. El
subconjunto rechaza `histval` junto con `endval` dentro de un mismo experimento
inicial.

Un setup contiene `y_init` en la fecha 0 y `y_end` en T+1. La trayectoria inicial
de Newton repite el vector endógeno terminal en las fechas 1 a T. La trayectoria
exógena usa los niveles exógenos terminales, o los iniciales si no hay `endval`,
y las asignaciones por fecha sustituyen esos niveles donde corresponda. Son
**niveles/innovaciones exógenos tal como aparecen en las ecuaciones**; no hay
conversión automática a variaciones porcentuales.

Los valores por fecha pueden usar expresiones finitas de los parámetros
calibrados previamente. Las fechas son enteros positivos dentro del horizonte.
Un único valor puede llenar todas las fechas de un rango; en otro caso,
proporcione un valor por cada fecha expandida. Para facilitar la portabilidad,
evite combinar varios rangos con un valor distinto por rango; escriba
secuencias separadas de declaraciones de choques.

Una declaración `stderr` o de covarianza controla la varianza estocástica; no
crea una realización determinista. Un bloque con solo choques por fecha deja
en cero las varianzas estocásticas no declaradas y puede generar una
advertencia del analizador. Declare ambos tipos explícitamente en programas
que también usen `stoch_simul`. El ejecutor mixto exige referencias exógenas
cero para los comandos estocásticos; recentre el modelo antes de un experimento
estocástico alrededor de una referencia exógena no nula.

## Opciones admitidas y límites de compatibilidad

Los siguientes valores predeterminados pertenecen al **ejecutor ordenado de
puremacro**. Las opciones deben ser literales admitidos; la inspección de
compatibilidad no evalúa expresiones arbitrarias en las opciones de comandos.

| Comando | Opciones aceptadas | Valores predeterminados iniciales |
|---|---|---|
| `steady` en un programa determinista | `tol` o `tolf` | Tolerancia de residuos `1e-10` |
| `perfect_foresight_setup` | `periods`, `endval_steady` | `100`, falso |
| `perfect_foresight_solver` | `tol` o `tolf`; `maxit` o `max_iter`; `periods` coincidente; `stack_solve_algo`; `linear_solver`; `mcp` | `1e-8`, `50`, `100`, `0`, `'direct'`, falso |
| `simul` | Opciones de setup y solver juntas | Los mismos valores PF |

Las opciones PF persisten si se omiten después en el mismo programa. Un
`tol=` posterior sustituye el valor de `tolf=` anterior y viceversa; lo mismo
ocurre con los alias del límite de iteraciones. Se rechaza indicar ambos alias
en un comando. La tolerancia de `steady` es independiente de la del solver PF.
Cambiar `periods` en el solver sin preparar primero ese horizonte se rechaza.
Un nuevo setup sustituye la trayectoria preparada; repetir el solver sobre el
mismo setup reutiliza la última trayectoria convergida.

| Función | Soporte ordenado nativo o alternativa requerida |
|---|---|
| Newton apilado disperso directo | `stack_solve_algo=0` o `linear_solver='direct'` |
| Newton–Krylov | `linear_solver='krylov'`; omita `stack_solve_algo` explícito |
| Etiquetas de complementariedad | `mcp` nativo con método directo; cotas simples de variable/parámetros |
| Conjunto completo de algoritmos Dynare | Se rechaza `stack_solve_algo` distinto de 0; no se afirma equivalencia numérica |
| `tolx`, `lmmcp`, familias de opciones de solver/homotopía | Fuera del subconjunto ordenado; use controles nativos Python admitidos |
| Cambios de parámetros entre experimentos | Cree especificaciones distintas con `params=`; se rechazan reasignaciones repetidas |
| Calendarios históricos | Solo asignaciones sin índice/de fecha cero; proporcione un estado ampliado por Python |
| `varexo_det`, tendencias, regímenes y estimación | Pueden analizarse o registrarse, pero esta ruta ordenada no los ejecuta |
| Comandos de errores de expectativas, noticias posteriores y calendario | No implementados en esta ruta |
| Scripts MATLAB/Octave y ejecución de funciones externas | No se ejecutan; traslade la lógica del estudio a Python |
| Programas Ramsey/OSR o de agentes heterogéneos | Use sus APIs/puntos de entrada explícitos; quedan fuera de este subconjunto |

Las etiquetas MCP como `[mcp='r > floor']` conservan la orientación de la
ecuación, lado izquierdo menos lado derecho. En una cota inferior, el residuo
asociado debe tener el signo no negativo de la condición de complementariedad.
La etiqueta no infiere ni corrige la desigualdad económica. Se rechazan cotas
repetidas para una variable, expresiones de cota dinámicas y sintaxis no
admitida. MCP nativo no es un alias de `lmmcp` y no puede combinarse con Krylov.

La especificación compilada utiliza el analizador existente y el
preprocesamiento acotado de macros admitido. No es un intérprete general de
MATLAB. Las asignaciones de frontera deben poder evaluarse con la calibración
y el contexto de valores disponibles; las expresiones que dependan de una
solución numérica anterior pueden fallar ya durante la lectura. Para estos
estudios, resuelva en Python y pase los arreglos resultantes explícitamente.
Use el informe de compatibilidad del punto de entrada exacto como referencia:
reconocer una palabra clave no implica admitir su ejecución.

## Inspeccionar, diagnosticar y corregir

La inspección de compatibilidad está separada de la solución.
`coverage="complete"` significa que se recopiló toda la sintaxis;
`exit_code==0` indica que no hay hallazgos de compatibilidad para la ruta
seleccionada. Ninguno certifica una solución numérica. El informe identifica
ubicaciones, opciones no admitidas y correcciones sugeridas. `read_mod` puede
conservar una especificación cuyos comandos no se admiten; la ejecución
verifica la compatibilidad antes de comenzar los comandos numéricos.

```python
from puremacro.dsge import inspect_mod, ModCompatibilityError, ModExecutionError

unsupported_source = """
var x; model; x=0.5*x(-1); end;
initval; x=1; end;
simul(periods=4, tolx=1e-8);
"""
report = inspect_mod(unsupported_source, ordered=True)
assert report.coverage == "complete" and report.exit_code != 0
assert any(issue.code == "UNSUPPORTED_OPTION" for issue in report.issues)
try:
    read_mod(unsupported_source).execute_mod_commands()
except ModCompatibilityError as error:
    assert any(issue.code == "UNSUPPORTED_OPTION" for issue in error.report.issues)
else:
    raise AssertionError("An unsupported option must not be ignored")
```

Cuando comienza un comando numérico, un fallo lanza `ModExecutionError` con
`.report` y `.results`. Se conservan las entradas completadas y la fallida.
Un resultado PF fallido se retiene cuando el solver lo devolvió; un fallo
estático no necesariamente tiene resultado numérico. Esta ecuación estática
sin solución real muestra la diferencia:

```python
impossible = read_mod("""
var x; model; x^2+1=0; end;
initval; x=0; end;
steady;
simul(periods=3);
""")
try:
    impossible.execute_mod_commands()
except ModExecutionError as error:
    partial = error.results
    assert partial.invocations[-1].status == "failed"
    assert partial.invocations[-1].command == "steady"
    assert "SteadyStateError" in partial.invocations[-1].error
else:
    raise AssertionError("An unverified steady state must not be accepted")
```

| Síntoma | Revisión y corrección |
|---|---|
| Un fallo local de QZ/determinación impide el estudio determinista | Cargue con `read_mod`; compruebe por separado que el sistema finito de fronteras esté bien planteado |
| `ModelError` al leer | Revise declaraciones, parámetros, número de ecuaciones, inclusiones y cobertura; aumentar iteraciones no corrige sintaxis no admitida |
| `ModCompatibilityError` | Lea `.report`; sustituya el comando/opción o use su API Python explícita |
| Logaritmos, potencias o divisiones generan residuos no finitos | Revise positividad y denominadores en puntos iniciales, fronteras y trayectorias de prueba; use un punto analítico factible |
| `SteadyStateError` | Revise las ecuaciones estáticas y la calibración; una declaración analítica también debe satisfacerlas |
| PF no converge | Revise el resultado fallido y `solver_diagnostics`; compruebe fechas, independencia de ecuaciones, coherencia de fronteras y escalas antes de aumentar `maxit` |
| Un choque modificado parece no afectar | Revise la copia del setup; las declaraciones posteriores requieren preparar otro |
| Residuo pequeño pero transición dudosa | Recalcule residuos originales, compare horizontes mayores y revise factibilidad; converger no verifica la aproximación terminal |

El método PF directo de Python puede devolver `converged=False`; el ejecutor
ordenado convierte ese resultado en `ModExecutionError` y lo conserva.
Compruebe siempre la convergencia al llamar directamente a métodos numéricos.
Para controles de dominio, continuación, fronteras de política terminal y
estudios de horizonte, véanse los [controles de transiciones](dsge_transitions.md)
y las [auditorías de trayectorias](result_audits_extended.md).

## Conservar y exportar el experimento

`DynareCommandResults.invocations` contiene copias de comandos numéricos,
incluidos `steady` y setup. Los bloques de declaraciones actualizan las
entradas y no generan cada uno un resultado numérico. Cada llamada conserva
comando, ubicación, opciones explícitas/efectivas, huella de entradas, estado,
`result` completo y `selected_result`. Seleccionar variables reduce la
presentación; no resuelve un sistema de equilibrio de menor tamaño.

`effective_options["input_state"]` conserva calibración, valores iniciales y
terminales, historia, choques por fecha y covarianza. Las entradas del solver
conservan también los datos preparados. Cada `PerfectForesightSetup` expone
`periods`, `variable_names`, `shock_names`, `y_init`, `y_end`, `exogenous_path`,
`initial_path` y `parameters`. Sus arreglos tienen las dimensiones indicadas.

Las copias se conservan entre llamadas; las declaraciones y soluciones
posteriores no sobrescriben los resultados anteriores. El acceso directo
`results.perfect_foresight` es una copia separada del último resultado PF
convergido. Los arreglos y DataFrames internos siguen siendo objetos mutables
ordinarios: cree su propia copia antes de editar un experimento conservado.

```python
setups = [entry.result for entry in results.invocations
          if entry.command == "perfect_foresight_setup"]
first_setup = setups[0]
np.testing.assert_allclose(first_setup.to_frame().e, [1, 0, 0, 0])
assert first_setup.to_dataframe().equals(first_setup.to_frame())
print(first_setup.summary())
print(results.to_frame())
markdown_shocks = first_setup.to_markdown()
latex_shocks = first_setup.to_latex()
typst_shocks = first_setup.to_typst()
assert "e" in markdown_shocks and "tabular" in latex_shocks and "#table" in typst_shocks
```

| Resultado | Tablas e informes |
|---|---|
| `PerfectForesightSetup` | `summary()` describe dimensiones; `to_frame()`/`to_dataframe()` devuelven los choques por fecha; los exportadores Markdown/LaTeX/Typst presentan esa tabla |
| `DynareCommandResults` | `summary()` y `to_frame()` describen el historial; `to_markdown()`, `to_latex()`, `to_typst()` lo exportan |
| `PerfectForesightResult` | `path`/`to_frame()` contienen trayectorias endógenas; `summary()` incluye evidencia de convergencia; exportadores y `plot()` facilitan su presentación |
| `MCPResult` | Añade periodos con cotas activas y diagnósticos de complementariedad a la trayectoria y la información de convergencia |

`terminal_error` mide la distancia entre la última fila endógena simulada y el
vector terminal suministrado. No es una prueba de transversalidad ni una
estimación de todos los errores de truncamiento. Compare fechas tempranas
relevantes entre horizontes, junto con los residuos de las ecuaciones originales.

## MAV T02_A: una especificación para un cambio permanente de productividad

El archivo del curso utiliza productividad logarítmica
`a_t=rho_a*a_(t-1)+e_a`. Mantener la productividad en 1.02 requiere una primera
innovación `log(1.02)` y después `(1-rho_a)*log(1.02)`. Una sola innovación a un
AR(1) con `rho_a<1` se disipa. Las mismas ecuaciones fuente pueden conservarse
para los estudios locales y deterministas.

Ejecute el siguiente bloque desde la raíz de la copia correspondiente del
repositorio; el archivo del curso depende del código fuente y no de un archivo
de datos instalado con el paquete:

```python
# requires: matching source checkout with curso/notebooks/
from pathlib import Path

source = Path("curso/notebooks/modelos/neoclasico_prevision_perfecta.mod").read_text()
experiment = read_mod(source + """
endval; e_a=(1-rho_a)*log(1.02); end;
steady;
shocks; var e_a; periods 1; values log(1.02); end;
perfect_foresight_setup(periods=80);
perfect_foresight_solver(tolf=1e-10);
perfect_foresight_setup(periods=160);
perfect_foresight_solver;
""")
study = experiment.execute_mod_commands()
transitions = [entry.result for entry in study.invocations
               if entry.command == "perfect_foresight_solver"]
assert all(result.converged for result in transitions)
np.testing.assert_allclose(transitions[0].path.a, np.log(1.02), atol=1e-10)
early_difference = np.max(np.abs(
    transitions[0].path.iloc[:20].to_numpy() -
    transitions[1].path.iloc[:20].to_numpy()
))
print(early_difference)
```

El `stoch_simul` de primer orden del archivo original se ejecuta antes de los
comandos añadidos. Ambos pares setup/solver conservan sus horizontes y
fronteras por separado. Revise `early_difference` y la evidencia de residuos;
un residuo apilado pequeño no certifica una solución de horizonte infinito.

Para comparar métodos locales y globales, construya
`EquilibriumProblem.from_model` con asignaciones explícitas de estados,
controles, transición y expectativas; véanse los
[métodos de equilibrio global](dsge_global.md). Un archivo `.mod` no identifica
por sí solo esa formulación recursiva ni un objetivo de planificador. Para
mejorar la inicialización de trayectorias, la continuación y las auditorías
terminales, continúe con las [transiciones no lineales](dsge_transitions.md).
Estas rutas reutilizan las ecuaciones y explicitan sus distintos conjuntos
de información y supuestos de aproximación.
