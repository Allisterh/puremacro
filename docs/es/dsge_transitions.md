> 🇬🇧 [English](../dsge_transitions.md) · 🇪🇸 Español

# Transiciones no lineales, dominios y fechas de anuncio

Puremacro resuelve problemas deterministas no lineales de frontera definidos
mediante un modelo `.mod` admitido o residuos de equilibrio en Python. Las
ecuaciones se resuelven conjuntamente en el tiempo; no se exige una solución
previa por perturbación. Esto **no** implica que todo modelo tenga solución,
que sea única o que Newton la encuentre desde cualquier conjetura. El modelo
debe respetar las interfaces de ecuaciones y temporalidad admitidas. Las
fronteras, los dominios y los supuestos de información siguen siendo decisiones
económicas de quien especifica el experimento.

| Interfaz | Qué calcula |
| --- | --- |
| `solve_perfect_foresight` | Un problema determinista de frontera con horizonte finito. |
| `solve_transition` | El mismo problema con continuación explícita, transformaciones de dominio, escalas y aceptación opcional del horizonte. |
| `solve_announced_path` | Segmentos realizados de planes finitos revisados en fechas de información especificadas. |
| `extended_path` | Planes móviles con la innovación actual realizada y las futuras fijadas en cero; no integra de forma no lineal el riesgo futuro. |

Estas rutinas calculan trayectorias, no una función de política estocástica
global. Para ese problema distinto, consulte los
[métodos globales de equilibrio](dsge_global.md). La
[guía de flujos DSGE](dsge_workflows.md) compara los métodos disponibles y los
conecta con los cuadernos del curso. Para cargar modelos y ejecutar
comandos, consulte la [guía de modelos no lineales](dsge_nonlinear.md) y la
[guía de inspección de `.mod`](dsge_mod_inspection.md).

## El problema matemático de horizonte finito

Escriba el modelo con `n` variables endógenas y `m` entradas exógenas como

\[
f(y_{t+1},y_t,y_{t-1},\varepsilon_t;\theta,\bar y)=0,
\qquad t=1,\ldots,T.
\]

Cada evaluación devuelve `n` residuos. El vector desconocido es
\(Y=(y_1^\top,\ldots,y_T^\top)^\top\in\mathbb R^{nT}\).
El rezago inicial y el adelanto final se fijan:

\[
y_0=y_{\mathrm{init}},\qquad y_{T+1}=y_{\mathrm{end}}.
\]

El sistema contiene **T fechas interiores**, con `n*T` ecuaciones e incógnitas.
El solucionador no añade una ecuación que exija `y_T == y_end`. Un modelo sin
adelantos no usa la frontera final en sus residuos dinámicos; uno sin rezagos
no usa allí la frontera inicial. Aun así, proporcione un vector completo en el
orden declarado. Las ecuaciones determinan qué entradas son económicamente
relevantes; el solucionador no infiere una nueva especificación de fronteras
para variables de estado y de salto.

El orden de argumentos es exactamente `f(lead, current, lag, shock)`. En una
ecuación `.mod`, el residuo es lado izquierdo menos lado derecho. La interfaz
presentada tiene un adelanto, un valor actual y un rezago; un modelo con
historia más larga debe reducirse a esta forma mediante estados auxiliares
explícitos o transformaciones admitidas por el analizador. Esta guía no promete
soporte automático para cualquier construcción temporal.

`y_end` también es la referencia fija \(\bar y\) de `STEADY_STATE()` durante
una resolución. No se sustituye por el iterado actual. Una función que declare
explícitamente el argumento `steady_state` recibe la misma referencia. Un
vector terminal explícito es una asignación de frontera; no tiene por qué ser
un estado estacionario por llamarse `y_end`. El alias de bajo nivel `y_ss`
cumple la misma función. Una asignación `.mod` `endval` y una resolución
explícita del estado estacionario terminal son operaciones distintas; consulte
la guía del modelo para el orden de comandos.

### Convenciones de arreglos y fechas

| Objeto | Convención |
| --- | --- |
| `y_init`, `y_end` | Un valor físico por variable endógena. `solve_transition` también acepta diccionarios por nombre de variable. |
| `shocks` | Se recomienda forma `(T, m)`; la fila `0` corresponde a la fecha `1`. Para un choque se acepta un vector de longitud `T`. Sin choques se usa `(T, 0)`. Son entradas exógenas efectivas de las ecuaciones, sin logaritmos, conversiones porcentuales ni escalamiento por desviación estándar automáticos. |
| `initial_path` | Niveles físicos de forma `(T, n)`, sin filas de frontera. En continuación inicializa la resolución de la base; las trayectorias aceptadas inicializan las etapas siguientes. |
| `.path` devuelta | DataFrame con índice `1,...,T`, solo fechas interiores. El refinamiento puede devolver más filas que la trayectoria original de choques. |
| `terminal_error` | `max(abs(y_T - y_end))`; distancia descriptiva, no un residuo adicional de equilibrio ni una prueba automática de precisión del truncamiento. |

`solve_transition` obtiene `T` de la trayectoria de choques. No pase `periods`,
`n_periods`, `exogenous_path`, `y_ss` ni `surprise` a esta interfaz: entran en
conflicto con su experimento explícito. En la interfaz de menor nivel
`solve_perfect_foresight`, indique `periods=T`: omitirlo no deduce el horizonte
de `shocks`. Esa API también admite un arreglo heredado de `T+2` filas y
descarta la primera y la última, o trunca otros arreglos más largos.
Exactamente `T` filas evita esa ambigüedad.

## Comenzar con un modelo no lineal

`read_mod` construye una especificación no lineal sin resolver primero una
aproximación local. Este modelo pequeño permite ver la temporalidad. Ejecute
los bloques Python de la guía **en orden y en una misma sesión**; los bloques
posteriores reutilizan estas importaciones y el ejemplo lineal. Inglés y
español contienen bloques ejecutables idénticos. El gráfico final requiere
Matplotlib.

```python
import numpy as np
from puremacro.dsge import read_mod, solve_perfect_foresight, solve_transition

model = read_mod("""
var x; varexo e;
model; x = 0.25*x(+1) + 0.4*x(-1) + e; end;
initval; x=0; e=0; end;
""")
shocks = np.r_[0.1, np.zeros(39)][:, None]
transition = solve_transition(
    model, y_init=[0.0], y_end=[0.0], shocks=shocks,
    horizon_tolerance=1e-8, compare_periods=20,
    max_periods=160, shock_tail=[0.0], linear_solver="krylov",
)
assert transition.converged, transition.summary()
assert transition.reason == "horizon_accepted"
path = transition.path
print(transition.summary())
```

El destino es una innovación en la fecha 1 seguida de ceros. El flujo resuelve
ese experimento y comprueba después que duplicar el horizonte tenga un efecto
suficientemente pequeño sobre las primeras 20 fechas. `transition.path`
devuelve una copia; compruebe siempre `transition.converged` antes de tratarla
como un destino aceptado.

## Newton disperso directo o Newton–Krylov sin matriz

En una trayectoria actual defina

\[
A_t=\frac{\partial f_t}{\partial y_{t+1}},\quad
B_t=\frac{\partial f_t}{\partial y_t},\quad
C_t=\frac{\partial f_t}{\partial y_{t-1}}.
\]

Con fronteras fijas, el jacobiano apilado y su producto son

\[
J=\begin{bmatrix}
B_1&A_1&&\\
C_2&B_2&A_2&\\
&\ddots&\ddots&\ddots\\
&&C_T&B_T
\end{bmatrix},\qquad
(Jv)_t=A_tv_{t+1}+B_tv_t+C_tv_{t-1},
\quad v_0=v_{T+1}=0.
\]

Ambos métodos buscan una dirección de Newton de \(Jd=-F(Y)\) y usan búsqueda
de paso. `linear_solver="direct"` construye la matriz dispersa apilada y
resuelve directamente el sistema lineal. Para un modelo analizado pasado
directamente a `solve_perfect_foresight`, la incidencia simbólica limita las
entradas almacenadas al patrón estructural. Los bloques de funciones Python,
incluidas las transformadas por la interfaz de transición, pueden almacenar
un patrón más completo por fecha.

`linear_solver="krylov"` aplica productos con el jacobiano apilado y utiliza
LGMRES por defecto; GMRES reiniciado es una opción explícita. Resulta útil
cuando formar o factorizar el sistema completo cuesta mucho, siempre que la
iteración lineal tenga un precondicionador eficaz. Los productos sin matriz no
hacen pequeños todos los costos: los bloques por fecha y sus factores todavía
pueden ser densos. Ningún método será necesariamente más rápido en todo modelo.

`method="auto"` usa primeras derivadas simbólicas para modelos analizados,
conservando la referencia fija de `STEADY_STATE`. Para funciones Python prueba
compatibilidad con paso complejo y, de no detectarla, usa diferencias
centrales. Proporcione
`jacobian_fn(lead,current,lag,shock) -> (A,B,C)` para controlar las derivadas,
como abajo. `method="analytic"` exige un modelo analizado o esa función.
Siguen disponibles `method="central"` y `"complex"`; el paso complejo exige
que los residuos conserven las perturbaciones complejas y sean adecuados para
esa diferenciación. Una prueba automática no demuestra esa propiedad.

```python
def equilibrium(lead, current, lag, shock):
    return current - 0.25*lead - 0.4*lag - shock

def date_jacobian(lead, current, lag, shock):
    return np.array([[-0.25]]), np.array([[1.0]]), np.array([[-0.4]])

fixed_problem = dict(
    y_init=[0.0], y_end=[0.0], periods=len(shocks), shocks=shocks,
    variable_names=["x"], tol=1e-10,
)
direct = solve_perfect_foresight(model, **fixed_problem)
krylov = solve_perfect_foresight(
    equilibrium, jacobian_fn=date_jacobian,
    linear_solver="krylov", **fixed_problem,
)
assert direct.converged and krylov.converged
np.testing.assert_allclose(direct.path, krylov.path, atol=1e-9, rtol=0)
assert krylov.solver_diagnostics["jvp_method"] == "analytic"
```

### Precondicionadores y combinaciones admitidas

El precondicionador Krylov predeterminado factoriza un sistema tridiagonal por
bloques construido con jacobianos fijos de la referencia terminal. Entre los
nombres admitidos están `steady_block_tridiagonal`, `steady_splu` y
`current_block_tridiagonal`; este último se reconstruye con la trayectoria
actual. La [guía de Newton apilado](dsge_stacked_newton.md) explica el protocolo
de operadores, las opciones de Jacobi por bloques y los residuos lineales.

`solve_transition` y `extended_path` con Krylov reutilizan la factorización
estacionaria predeterminada solo si el horizonte, las formas, los tipos de
datos y las entradas numéricas de los bloques coinciden **exactamente**.
Cambiar parámetros, referencias u horizonte puede exigir reconstruirla.
`reuse_preconditioner=False` desactiva la caché. Una opción explícita
`krylov_options["preconditioner"]`, incluido `None`, o `time_block` tiene
prioridad sobre la reutilización automática. El contador de la caché registra
sus propias construcciones, no todas las factorizaciones de un operador externo.

| Combinación solicitada | Comportamiento actual |
| --- | --- |
| Directo + ecuaciones ordinarias, dominios/escalas o MCP | Admitida. |
| Directo + política terminal, incluida asignación MCP explícita | Admitida; se incluye la regla de la cadena terminal. |
| Krylov + ecuaciones ordinarias y transformaciones de dominio/escala | Admitida. |
| Krylov + cotas MCP o `mcp=True` | Rechazo explícito. |
| Krylov + `terminal_policy` | Rechazo explícito. |
| Directo + cualquier `krylov_options` distinto de `None` | Rechazo explícito. |
| `terminal_jacobian` sin `terminal_policy` | Rechazo explícito. |

Las claves de alto nivel de `krylov_options` son `preconditioner`, `forcing`,
`krylov_method`, `inner_m`, `outer_k`, `krylov_maxiter`, `side`, `max_log_move`,
`time_block`, `progress` y `checkpoint`. Otros controles del motor de menor
nivel no se aceptan automáticamente aquí. Krylov nunca cambia silenciosamente
el problema no lineal al método directo. Un precondicionador de Thomas por
bloques puede recurrir a su factorización SuperLU documentada si un pivote es
inadecuado; eso afecta la construcción del precondicionador, no la elección
del solucionador no lineal.

## Continuación, base y presupuesto de resoluciones

La continuación resuelve una familia explícita de experimentos. Con entradas
base \((E^0,b^0,\theta^0)\) y objetivo \((E^1,b^1,\theta^1)\),

\[
E^\lambda=(1-\lambda)E^0+\lambda E^1,\qquad
b^\lambda=(1-\lambda)b^0+\lambda b^1,\qquad
\theta^\lambda=(1-\lambda)\theta^0+\lambda\theta^1.
\]

La implementación copia exactamente los extremos para las fracciones cero y
uno. El rezago inicial `y_init` permanece fijo. La frontera terminal y su
referencia `STEADY_STATE` siguen a `b` en cada etapa. Una política terminal
suministrada se conserva tal cual; no se recalibra de manera oculta al cambiar
los parámetros.

La base predeterminada es choques cero y `baseline_end=y_init`. Es una base de
cálculo, no una afirmación de que el estado inicial sea estacionario. Indique
`baseline_shocks` con la misma forma del objetivo y `baseline_end` si otra
base corresponde al experimento económico. La base debe resolverse con éxito.
`continuation=False` solicita una resolución directa del objetivo.

Para `NonlinearModel`, `baseline_params` cambia solamente los parámetros
nombrados y ya calibrados; los demás conservan sus valores objetivo. El
diccionario del modelo original no se modifica. No se resuelven estados
estacionarios intermedios automáticamente. Las cotas MCP admitidas que dependen
de parámetros se evalúan con la calibración de cada etapa. Las funciones
de residuos, derivadas y política terminal suministradas por el usuario deben
ser coherentes con el experimento que representan.

El incremento inicial es `initial_step=0.25`. Una etapa fallida divide el
incremento entre dos; una aceptada lo multiplica por 1.5, sin superar la distancia
restante a uno. `min_step=1/1024` limita las reducciones. `max_iter` rige cada
resolución numérica; `max_solves=40` cuenta conjuntamente la base, los intentos
de continuación aceptados y fallidos, y los refinamientos de horizonte.
Son presupuestos de iteraciones y resoluciones, no plazos de tiempo real.

```python
calibrated = read_mod("""
var x; parameters theta; theta=999;
model; x^3 = 1 + theta; end;
initval; x=1; end;
""")
parameter_path = solve_transition(
    calibrated, y_init=[1.0], y_end=[1.0], shocks=np.zeros((3, 0)),
    baseline_params={"theta": 0.0}, max_iter=4, max_solves=40, tol=1e-9,
)
assert parameter_path.converged, parameter_path.summary()
np.testing.assert_allclose(parameter_path.path["x"], 10.0, atol=1e-9)
assert parameter_path.continuation_history[-1]["fraction"] == 1.0
assert calibrated.parameter_values["theta"] == 999.0

limited = solve_transition(
    model, y_init=[0.0], y_end=[0.0], shocks=shocks, max_solves=1,
)
assert not limited.converged
assert limited.reason == "solve_budget_exhausted"
```

El ejemplo cúbico no tiene variables exógenas, de ahí la forma `(3, 0)`. Su
objetivo exacto es `x=10`. La comprobación final muestra que una base resuelta
por sí sola nunca se presenta como éxito del objetivo. La continuación puede
ayudar a alcanzar una raíz; no elimina singularidades, no garantiza una rama
única ni asegura que todas las bases conduzcan a la misma solución objetivo.

## Dominios económicos y escalas de residuos

`variable_bounds` especifica **dominios abiertos**. Con
`variable_scale[j]=s_j` positivo, el solucionador usa coordenadas monótonas:

| Dominio de `y_j` | Valor físico decodificado de `z_j` |
| --- | --- |
| Sin cotas | \(y_j=s_jz_j\) |
| \(y_j>\ell_j\) | \(y_j=\ell_j+\exp(s_jz_j)\) |
| \(y_j<u_j\) | \(y_j=u_j-\exp(-s_jz_j)\) |
| \(\ell_j<y_j<u_j\) | \(y_j=\ell_j+(u_j-\ell_j)/(1+\exp(-s_jz_j))\) |

Las fronteras, conjeturas y salidas de la política terminal deben estar
estrictamente dentro del dominio. Durante las evaluaciones transformadas,
candidatos con desbordamiento o saturación numérica se rechazan antes de evaluar
los residuos económicos; no se recorta una política a un valor factible.
Esto protege logaritmos y
potencias definidos solo en el dominio, pero no corrige cualquier raíz mal
condicionada.

`equation_scale[i]=q_i>0` sustituye la fila `f_i` por `f_i/q_i`. Para una
transición ordinaria, la aceptación exige convergencia del método y

\[
\max_{t,i}\left|f_i(y_{t+1},y_t,y_{t-1},\varepsilon_t)/q_i\right|
\leq\texttt{tol}.
\]

Esta comprobación explícita por fila es más estricta que confiar solamente
en la bandera de un método cuyo criterio también puede escalar por una magnitud
del jacobiano. `variable_scale` no sustituye unidades económicas interpretables
en las ecuaciones. Las escalas pueden ser escalares positivos, vectores o
diccionarios completos. Un diccionario sigue el orden de residuos mediante los
nombres de variables declarados; use un vector cuando esos nombres oculten
un orden diferente de ecuaciones.

```python
def log_equation(lead, current, lag, shock):
    return 1000.0 * (np.log(current) - shock)

positive = solve_transition(
    log_equation, y_init=[1.0], y_end=[1.0],
    shocks=np.full(10, -3.0), variable_names=["c"],
    variable_bounds={"c": (0.0, None)},
    equation_scale=[1000.0], variable_scale=[2.0], tol=1e-9,
)
assert positive.converged, positive.summary()
np.testing.assert_allclose(positive.path["c"], np.exp(-3.0), rtol=1e-8)
assert positive.scaled_residual_norm <= 1e-9
physical_error = np.max(np.abs(1000.0*(np.log(positive.path["c"]) + 3.0)))
np.testing.assert_allclose(positive.solution.residual_norm, physical_error, atol=1e-12)
```

En un `PerfectForesightResult` ordinario dentro de `transition.solution`,
`residual_norm` e `initial_residual_norm` conservan las unidades originales de
las ecuaciones; `transition.scaled_residual_norm` informa el criterio escalado
por fila. En `MCPResult`, los residuos y diagnósticos de complementariedad
conservan las unidades transformadas/escaladas del solucionador, aunque la
trayectoria se devuelva en niveles físicos. Compruebe por separado la
factibilidad y complementariedad físicas cuando importen sus tolerancias
económicas.

Un dominio abierto no representa una restricción que deba poder ser activa.
Use MCP para esa condición. Si una variable tiene dominio y cota de
complementariedad, esta última debe estar estrictamente dentro del dominio
para tener coordenada transformada finita; un dominio con límite inferior cero
y una cota inferior activa en cero son incompatibles.

## Aceptación del horizonte y colas de choques permanentes

Un residuo apilado pequeño verifica un problema de frontera finita. No demuestra
que esa frontera aproxime adecuadamente un horizonte infinito. Con
`horizon_tolerance`, el flujo primero resuelve el objetivo exacto y después fija
\(T'=\min(2T,\texttt{max_periods})\). Mantiene condición inicial, prefijo de
choques, calibración objetivo y política/referencia terminal; amplía los
choques explícitamente e inicializa la resolución más larga con la anterior.

Para `P=compare_periods`, el estadístico de aceptación es

\[
\Delta_{T,T'}=\max_{1\leq t\leq P}\max_j
\left|y^{T'}_{t,j}-y^T_{t,j}\right|.
\]

Es una **diferencia absoluta en unidades físicas**. No se divide entre
`equation_scale` ni `variable_scale`. Elija la tolerancia según las unidades
reportadas. `compare_periods` no puede superar el horizonte inicial y
`max_periods` debe superarlo. El refinamiento termina con la primera comparación
aceptada, al llegar al máximo horizonte, ante una resolución ampliada fallida
o al agotarse el presupuesto compartido. La comparación verifica solamente la
ventana común declarada.

Se exige un `shock_tail` explícito. Puede ser un escalar/fila compatible con
las fechas nuevas, o una función `shock_tail(old_T, new_T)` que devuelva la
extensión. Sus argumentos son longitudes de horizonte, no fechas de choques
contadas desde uno. No se impone silenciosamente una interpretación permanente
o transitoria.

Por ejemplo, si la productividad logarítmica satisface
\(a_t=\rho a_{t-1}+u_t\), cambiar solo `y_end` no mantiene la productividad
en `a_star`. Desde `a_0=0`, un nivel permanente `a_star` que comienza en la
fecha 1 requiere `u_1=a_star` y después `u_t=(1-rho)*a_star`. La misma innovación
recurrente debe mantenerse al ampliar la cola:

```python
productivity = read_mod("""
var a; varexo innovation; parameters rho; rho=0.9;
model; a = rho*a(-1) + innovation; end;
initval; a=0; innovation=0; end;
""")
rho, target_a, initial_horizon = 0.9, 0.2, 12
permanent_innovations = np.full((initial_horizon, 1), (1.0-rho)*target_a)
permanent_innovations[0, 0] = target_a
permanent = solve_transition(
    productivity, y_init=[0.0], y_end=[target_a],
    shocks=permanent_innovations, continuation=False,
    horizon_tolerance=1e-10, compare_periods=12, max_periods=48,
    shock_tail=[(1.0-rho)*target_a],
)
assert permanent.converged, permanent.summary()
np.testing.assert_allclose(permanent.path["a"], target_a, atol=1e-10)
assert permanent.horizon_history[-1]["accepted"]
```

`horizon_history` registra los horizontes anterior/nuevo, la ventana, diferencia
máxima, tolerancia y aceptación. `terminal_error` puede ser positivo en una
solución finita válida y no sustituye esta comparación.

## Anuncios, sorpresas y revisiones de creencias

En una resolución de previsión perfecta se conoce desde el inicio toda la
trayectoria de choques. Un choque no nulo en la fecha 8 puede alterar las
fechas 1–7 mediante ecuaciones con adelantos. Una sorpresa conocida solo en la
fecha 8 es un experimento de información distinto, aunque finalmente el arreglo
exógeno realizado sea el mismo.

`ShockAnnouncement(date, shocks, y_end=None)` registra una nueva trayectoria
de creencias sobre el calendario completo. El primer anuncio debe ocurrir en
la fecha 1; los posteriores deben aumentar estrictamente dentro del mismo
calendario. Todos los arreglos tienen la misma forma. Las revisiones pueden
cambiar choques actuales/futuros y opcionalmente la frontera/referencia terminal,
pero todas las filas anteriores al anuncio deben coincidir con los choques ya
realizados. Si se omite `y_end`, persiste la frontera anunciada anteriormente.

```python
from puremacro.dsge import ShockAnnouncement, solve_announced_path

news = np.zeros((40, 1))
news[7, 0] = 0.1
anticipated = solve_announced_path(
    model, [ShockAnnouncement(1, news)], y_init=[0.0], y_end=[0.0],
)
surprise = solve_announced_path(
    model, [ShockAnnouncement(1, np.zeros_like(news)), ShockAnnouncement(8, news)],
    y_init=[0.0], y_end=[0.0],
)
assert anticipated.converged and surprise.converged
assert np.max(np.abs(anticipated.path.iloc[:7].to_numpy())) > 1e-4
np.testing.assert_allclose(surprise.path.iloc[:7], 0.0, atol=1e-12)
assert surprise.solver_diagnostics["completed_periods"] == 40
```

En la fecha de anuncio `d`, se resuelven las fechas `d,...,T` desde el estado
efectivamente alcanzado en `d-1`, y se realiza solo el segmento anterior al
siguiente anuncio. Las revisiones futuras nunca se revelan antes. El calendario
restante se reduce; no se sustituye de manera oculta por un horizonte móvil
constante. Un plan fallido detiene la realización y devuelve `converged=False`,
el prefijo completado y diagnósticos por anuncio. La realización empalmada no
tiene un único contexto de auditoría de previsión perfecta ex ante.

`solve_announced_path` acepta opciones del solucionador de previsión perfecta
de bajo nivel, no las opciones de continuación/dominio/horizonte de la
interfaz de transición. `surprise`, `periods`, `n_periods`, `exogenous_path`,
`y_ss` y una sola `initial_path` entran en conflicto con los planes fechados
y se rechazan explícitamente. No se ofrece aceptación automática de horizonte
para todo este calendario informativo; si hace falta, compare calendarios
más largos construidos explícitamente.

Para sorpresas repetidas de un periodo, `simulate_surprise_shocks` resuelve
en cambio un horizonte móvil fijo, realiza la primera fecha y repite. Su base
predeterminada de choques futuros es cero.
`solve_perfect_foresight(..., surprise=True)` utiliza ese experimento móvil.
`extended_path` puede recibir o generar la innovación actual y también fija
las futuras en cero. Esas trayectorias son una aproximación de equivalencia
cierta; sustituir la innovación media en ecuaciones no lineales no calcula por
sí solo las expectativas no lineales.

## Complementariedad: variables, filas y signos

Para una cota inferior de la variable `j`, la asignación explícita `(i,s)` significa

\[
y_{t,j}-\ell_j\geq0,\qquad s f_{t,i}\geq0,\qquad
(y_{t,j}-\ell_j)s f_{t,i}=0,\qquad s\in\{-1,+1\}.
\]

Para una cota superior se usa `u_j-y_j >= 0` y `s*f_i <= 0`. Con dos cotas
finitas, `s*f_i` es no negativo en la inferior, cero en el interior y no
positivo en la superior. Las demás filas siguen siendo igualdades. Por eso
el residuo de la igualdad original no tiene por qué anularse con una restricción
activa.

`mcp_bounds` asigna nombres o índices de variable desde cero a `(lower, upper)`;
`None` indica una cota ausente. `mcp_equations` asigna las mismas claves a una
fila de residuo desde cero o a `(row, sign)`. Una fila entera sola significa
signo `+1`. La asignación explícita debe cubrir exactamente las variables
acotadas, con filas distintas. Los órdenes de ecuaciones y variables pueden diferir.

La etiqueta admitida `[mcp='r > lower']` indica una cota inferior para `r`,
asociada con la ecuación etiquetada y su signo izquierdo menos derecho. Se
admiten `<`, `<=`, `>` y `>=` con una variable y una expresión finita de cota
que dependa solo de parámetros calibrados. La notación representa una cota
MCP cerrada, no un dominio abierto. Cotas dinámicas, etiquetas duplicadas para
una variable y desigualdades arbitrarias no forman un lenguaje general de
restricciones; las cotas bilaterales usan la API Python.

```python
bounded_model = read_mod("""
var x r; varexo e; parameters lower; lower=0;
model;
[mcp='r > lower'] r=e;
x=1-3*r;
end;
initval; x=1; r=0; e=0; end;
""")
bounded = solve_transition(
    bounded_model, y_init=[1.0, 0.0], y_end=[1.0, 0.0],
    shocks=-np.ones((5, 1)), mcp=True,
)
assert bounded.converged, bounded.summary()
np.testing.assert_allclose(bounded.path["r"], 0.0, atol=1e-8)

def reversed_residual(lead, current, lag, shock):
    x, r = current
    return np.array([shock[0]-r, x+3.0*r-1.0])

explicit = solve_perfect_foresight(
    reversed_residual, y_init=[1.0, 0.0], y_end=[1.0, 0.0],
    periods=5, shocks=-np.ones((5, 1)), variable_names=["x", "r"],
    mcp_bounds={"r": (0.0, None)}, mcp_equations={"r": (0, -1)},
)
assert explicit.converged
np.testing.assert_allclose(explicit.path, bounded.path, atol=1e-8)
gap = bounded.path["r"].to_numpy()
margin = gap + 1.0
assert np.min(gap) >= -1e-8 and np.min(margin) >= -1e-8
assert np.max(np.abs(np.hypot(gap, margin)-gap-margin)) < 1e-8
```

Aquí `r` es la segunda variable, pero su ecuación complementaria es la primera.
La etiqueta implica `mcp_equations={"r": (0, 1)}`; la función invierte esa
ecuación deliberadamente y por eso usa `(0, -1)`. Las comprobaciones finales
verifican brechas físicas no negativas y un residuo Fischer–Burmeister
independientemente de la bandera del solucionador.

Use `mcp=True` para activar las etiquetas admitidas. Cotas explícitas no vacías
también seleccionan MCP nativo. Sin asignaciones/etiquetas explícitas, la
interfaz Python anterior conserva su heurística de cargas para fila/signo;
especificar la asignación aclara las restricciones económicas de signo.
Se utiliza el solucionador semisuave nativo, sin afirmar equivalencia con
todos los algoritmos u opciones de complementariedad de Dynare. MCP sigue
requiriendo el método directo.

## Políticas terminales y último bloque del jacobiano

Si se conoce un mapa terminal fiable `g`, `terminal_policy(last_state)` fija
\(y_{T+1}=g(y_T)\). Debe devolver un vector finito en el mismo orden de
variables. `y_end` permanece como referencia fija de `STEADY_STATE`. El último
bloque diagonal pasa a ser

\[
\frac{d f_T}{d y_T}=B_T+A_T Dg(y_T).
\]

Proporcione `terminal_jacobian(last_state)` como matriz cuadrada o permita que
el solucionador directo diferencie el mapa con diferencias centrales. Con
dominios/escalas, `solve_transition` recibe estas funciones en coordenadas
físicas y compone su regla de la cadena con la transformación.

En el ejemplo lineal, una cola estable `x_t=r^t*x_0` cumple
`0.25*r**2-r+0.4=0`. Su raíz menor da una cola exacta para este modelo lineal
particular sin choques:

```python
stable_root = (1.0 - np.sqrt(1.0 - 4.0*0.25*0.4)) / (2.0*0.25)
terminal = solve_perfect_foresight(
    model, y_init=[1.0], y_end=[0.0], periods=5, shocks=np.zeros((5, 1)),
    terminal_policy=lambda last: stable_root * last,
    terminal_jacobian=lambda last: np.array([[stable_root]]), tol=1e-10,
)
assert terminal.converged
np.testing.assert_allclose(terminal.path["x"], stable_root**np.arange(1, 6), atol=1e-10)

from puremacro.validation import audit_result
terminal_audit = audit_result(terminal, checks=["equations"])
assert np.max(np.abs(terminal_audit.maps["path_equations"])) < 1e-9
```

Un mapa terminal lineal local en un modelo no lineal es una aproximación cuya
adecuación todavía requiere una comprobación de horizonte o económica. Krylov
rechaza políticas terminales. Las auditorías ordinarias conservadas vuelven a
evaluar la política y mantienen la referencia fija; además comparan su
entrada/salida terminal original para detectar cambios de estado de la función.
Mantenga las funciones sin cambios después de resolver. Esa repetición no
demuestra identidad matemática en todo el dominio de la política. Las
trayectorias MCP requieren sus propias comprobaciones físicas de complementariedad.

## Diagnósticos, fallos y presentación

`TransitionResult.converged` exige alcanzar el objetivo exacto y aprobar todas
las comprobaciones de horizonte solicitadas. `solution` conserva el último
resultado numérico disponible; en un fallo puede ser intermedio o fallido.
`path` está vacío si no se conservó un resultado. La convergencia de una
resolución interna no basta para declarar éxito del flujo completo.

| `reason` | Interpretación |
| --- | --- |
| `target_solved` | Objetivo exacto aceptado; no se solicitó comprobar el horizonte. |
| `horizon_accepted` | Objetivo exacto y comparación de horizonte en la ventana común aceptados. |
| `baseline_failed` | La base especificada no superó la aceptación numérica. |
| `target_failed` | Falló el intento directo con `continuation=False`. |
| `continuation_step_exhausted` | Seguir reduciendo violaría `min_step`. |
| `solve_budget_exhausted` | Terminó el presupuesto compartido antes de completar lo solicitado. |
| `horizon_solve_failed` | Falló numéricamente un problema objetivo ampliado. |
| `horizon_budget_exhausted` | Se alcanzó el máximo horizonte sin la concordancia requerida. |

`continuation_history` registra la fracción, periodos, parámetros y aceptación
de cada intento. Los intentos numéricos completados incluyen residuos,
iteraciones y advertencias; las excepciones conservan un registro de error.
Los errores de contrato de entrada se propagan, no se presentan como éxitos
parciales. `solves`, `scaled_residual_norm` y `preconditioner_builds` resumen
el flujo conservado; inspeccione los historiales para distinguir una resolución
difícil de ecuaciones de un horizonte inadecuado.

En resultados ordinarios, `solution.solver_diagnostics` incluye detalles del
método/derivadas, las escalas y la norma original del residuo. `linear_history`
de Krylov informa `true_linear_residual` de la dirección devuelta, tolerancia
lineal solicitada, estado de iteraciones, paso y retrocesos. Un estado positivo
de la iteración interna no implica convergencia no lineal; examine el resultado
final y los residuos. `extended_path.solver_diagnostics` registra además
resoluciones móviles y construcciones/reutilizaciones del precondicionador.

Tablas y gráficos muestran el candidato conservado, sin alterar su estado.
`to_frame()` y `to_dataframe()` devuelven copias. Markdown, LaTeX y Typst usan
el formato y escape compartidos de `puremacro.reports`.
`plot(ax=..., show=False)` respeta los ejes suministrados y no abre una ventana
por defecto.
Si no se conservó una solución numérica, solicitar un gráfico produce un error claro.

```python
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

assert transition.converged, transition.summary()
continuation_table = pd.DataFrame(transition.continuation_history)
horizon_table = pd.DataFrame(transition.horizon_history)
assert continuation_table.iloc[-1]["fraction"] == 1.0
assert bool(horizon_table.iloc[-1]["accepted"])
exported = transition.to_frame()
pd.testing.assert_frame_equal(exported, transition.to_dataframe())
print(transition.to_markdown(head=3, digits=6))
assert r"\begin{tabular}" in transition.to_latex(head=3)
assert "#table(" in transition.to_typst(head=3)
figure, axes = plt.subplots()
assert transition.plot(ax=axes, show=False) is figure
plt.close(figure)

equation_audit = audit_result(transition.solution, checks=["equations"])
assert np.max(np.abs(equation_audit.maps["path_equations"])) < 1e-8
```

La auditoría ordinaria comprueba el experimento finito guardado independientemente
de la bandera de convergencia. Un residuo pequeño no certifica unicidad,
precisión de horizonte infinito ni optimalidad estocástica. Los tests de
`tests/test_dsge_transition_workflows.py` cubren trayectorias lineales y de raíz
estable exactas, coincidencia de derivadas, presupuestos y extremos exactos de
continuación, dominios/escalas, fechas de información, signos MCP, auditorías
terminales, reutilización de precondicionadores y exportaciones. Establecen
esos contratos en los casos probados, no convergencia universal para cualquier modelo.
