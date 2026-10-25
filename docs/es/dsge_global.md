> 🇬🇧 [English](../dsge_global.md) · 🇪🇸 Español

# Políticas de equilibrio locales y globales

Una misma especificación no lineal permite realizar varios cálculos, cada uno
con una pregunta económica y comprobaciones de precisión distintas:

| Cálculo | Objeto calculado | Tratamiento del futuro | Evidencia que conviene examinar |
| --- | --- | --- | --- |
| `model.solve(order=1/2/3)` | Aproximación local de Taylor de las reglas de decisión | La covarianza entra en la perturbación; las innovaciones realizadas entran al evaluar la política | Determinación, condicionamiento y residuos no lineales cerca y lejos del punto de expansión |
| `solve_perfect_foresight` / `solve_transition` | Trayectoria finita para condiciones iniciales y un cierre terminal especificados | La trayectoria anunciada de choques es conocida dentro del experimento | Ecuaciones originales, factibilidad, información disponible y sensibilidad al cierre y horizonte |
| `solve_time_iteration` | Política recursiva interpolada sobre un dominio de estados declarado | Una regla condicional integra políticas futuras en estados sucesores posibles | Convergencia en nodos, residuos Euler/KKT fuera de la grilla, cobertura y sensibilidad a aproximación e integración |

Una trayectoria de previsión perfecta no es una política estocástica: sustituir
choques futuros desconocidos por cero generalmente modifica una esperanza no
lineal. Una política global tampoco implica resolver todos los estados posibles
ni demostrar unicidad global. Aquí «global» describe la aproximación sobre el
dominio elegido. Consulte [la carga de modelos no lineales](dsge_nonlinear.md),
[las transiciones no lineales](dsge_transitions.md) y
[la guía de flujos DSGE](dsge_workflows.md) para elegir métodos y organizar la enseñanza.

La interfaz global no infiere el objetivo de un planificador a partir de
ecuaciones arbitrarias. Requiere estados, controles, transiciones,
probabilidades condicionales, dominios y admisibilidad explícitos. No existe una
garantía general de existencia, unicidad o convergencia.

Todos los bloques Python siguientes se ejecutan **en orden en una misma sesión**
y utilizan la API pública. Los ejemplos son idénticos en ambas ediciones.

## El contrato del equilibrio recursivo

Escriba el problema en términos de estados conocidos al decidir, controles e
innovaciones futuras:

\[
x_t=\pi(s_t),\qquad s_{t+1}=g(s_t,x_t,\varepsilon_{t+1}),\qquad
E_t\!\left[F(s_t,x_t,s_{t+1},\pi(s_{t+1}),\varepsilon_{t+1})\right]=0.
\]

`EquilibriumProblem` describe este sistema. No es un objetivo de Bellman ni
infiere un planificador que maximice bienestar. Los estados deben contener la
información necesaria para que la transición y la distribución condicional
suministradas sean válidas. Aumentar el grado polinomial no recupera un rezago,
régimen o choque persistente omitido.

Sean `ns` el número de estados, `nx` el de controles, `ne` el de coordenadas de
innovación y `q` el de nodos de integración.

| Argumento o callback | Forma y significado requeridos |
| --- | --- |
| `state_names`, `control_names` | Nombres ordenados, de longitudes `ns` y `nx`; únicos dentro de cada grupo |
| `domain` | Cotas finitas y crecientes de forma `(ns, 2)`; rectángulo de aproximación |
| `transition(state, controls, innovation)` | Recibe `(ns,)`, `(nx,)`, `(ne,)`; devuelve un estado sucesor finito `(ns,)` |
| `residual(state, controls, next_state, next_controls, innovation)` | Devuelve un vector finito `(nx,)` antes de tomar esperanza; una ecuación por control |
| `shock_nodes`, `shock_weights` | Innovaciones futuras `(q, ne)` y probabilidades no negativas `(q,)` que suman uno |
| `shock_rule(state)` | Par condicional opcional `(nodes, weights)`; sustituye la regla fija y se comprueba en cada evaluación |
| `control_bounds` | Cotas ordenadas `(nx, 2)`, o callback que las devuelve a partir de `(ns,)`; admite extremos infinitos |
| `admissible(state, controls)` | Comprobación económica booleana opcional, por ejemplo consumo positivo |
| `residual_scale` | Escalas positivas y finitas `(nx,)`; por defecto, unos |

Los callbacks reciben **un punto a la vez**. Las políticas devueltas también
admiten lotes: `result(state)` devuelve `(nx,)`, mientras que `result(states)`
con `states.shape == (n, ns)` devuelve `(n, nx)`. Un problema determinista puede
omitir la regla de choques: por defecto hay una innovación de forma `(0,)` con
ponderación uno.

Con `EquilibriumProblem.from_model` se necesitan estos mapeos adicionales:

| Mapeo | Contrato |
| --- | --- |
| `current(state, controls)` | Vector endógeno actual completo en orden `model.variables`; el mismo mapeo en los sucesores construye el vector adelantado |
| `lagged(state)` | Vector endógeno rezagado completo en orden `model.variables` |
| `exogenous(state)` | Innovaciones **actuales** en orden `model.shocks`; por defecto, ceros |
| `equations` | Una ecuación fuente distinta por control, seleccionada por nombre o índice entero desde cero |
| `steady_state` | Vector o diccionario endógeno completo opcional para las expresiones `steady_state(...)` del modelo fuente |

Consulte `model.equation_names` para conocer los nombres disponibles. El puente
copia el modelo y su calibración. Las variables capturadas por los callbacks
siguen siendo responsabilidad del usuario: cambiar un parámetro utilizado por
`transition` o `current` cambia ese callback aunque la calibración del residuo
fuente se haya copiado. El puente no comprueba que todas las ecuaciones no
seleccionadas se satisfagan idénticamente por los mapeos; establezca esas
identidades al construir el problema.

## Un modelo y una temporización recursiva explícita

Considere utilidad logarítmica, depreciación completa, producción Cobb–Douglas
e innovaciones gaussianas independientes de productividad logarítmica:

\[
\max E_0\sum_{t\ge0}\beta^t\log c_t,\quad
y_t=e^{a_t}k_{t-1}^{\alpha},\quad c_t+k_t=y_t,\quad a_t=\varepsilon_t.
\]

Esta especialización tratable del crecimiento estocástico tiene la política
exacta \(k_t=\alpha\beta y_t\), con \(c_t=(1-\alpha\beta)y_t\). Sustituirla en
\(1/c_t=\beta E_t[\alpha y_{t+1}/(k_t c_{t+1})]\) verifica Euler: el producto
futuro se cancela y queda la misma utilidad marginal actual. Se obtiene así una
referencia económica independiente del algoritmo numérico. El marco de
crecimiento estocástico procede de
[Brock y Mirman (1972)](https://www.sciencedirect.com/science/article/abs/pii/0022053172901354).

El estado recursivo es `(inherited_capital, observed_log_productivity)` y el
control es el ahorro, también denominado capital al final del período actual
`k` en el archivo `.mod`. `read_mod` compila el modelo sin exigir una solución local.

```python
from dataclasses import replace
import numpy as np
from puremacro.dsge import (
    read_mod, EquilibriumProblem, gaussian_quadrature,
    solve_time_iteration, audit_policy, compare_policies,
    solve_adaptive_time_iteration, refine_time_iteration,
    evaluate_perturbation, diagnose_perturbation,
)

model = read_mod("""
var y c k a; varexo e;
parameters alpha beta;
alpha=0.33; beta=0.95;
model;
    [name='resources'] c+k=y;
    [name='production'] y=exp(a)*k(-1)^alpha;
    [name='Euler'] 1/c=beta*alpha*y(+1)/(k*c(+1));
    [name='productivity'] a=e;
end;
steady_state_model;
    k=(alpha*beta)^(1/(1-alpha));
    y=k^alpha; c=(1-alpha*beta)*y; a=0;
end;
shocks; var e; stderr 0.02; end;
""")
alpha, beta = (model.parameter_values[key] for key in ("alpha", "beta"))
names = tuple(model.variables)

def output(state):
    return np.exp(state[1]) * state[0] ** alpha

def current(state, control):
    y = output(state)
    values = dict(y=y, c=y-control[0], k=control[0], a=state[1])
    return np.array([values[name] for name in names])

def lagged(state):
    return np.array([state[0] if name == "k" else 0.0 for name in names])

def transition(state, control, innovation):
    return np.array([control[0], innovation[0]])

nodes, weights = gaussian_quadrature([[0.02**2]], order=3)
problem = EquilibriumProblem.from_model(
    model, state_names=("inherited_capital", "observed_log_productivity"),
    control_names=("savings",), domain=((0.1, 0.5), (-0.05, 0.05)),
    current=current, lagged=lagged, transition=transition,
    equations=("Euler",), exogenous=lambda state: state[1:2],
    shock_nodes=nodes, shock_weights=weights,
    control_bounds=lambda state: [(1e-6, output(state)-1e-6)],
    admissible=lambda state, control: 0 < control[0] < output(state),
)
initial = lambda state: np.array([0.26 * output(state)])
result = solve_time_iteration(problem, initial, orders=(7, 2), tol=1e-8)
assert result.converged
print(result.summary())
grid_values = result.to_frame()
assert len(grid_values) == (7 + 1) * (2 + 1)
```

El estado contiene el capital heredado y la productividad **actual observada**.
La transición utiliza una innovación **futura**. `current` construye los
vectores endógenos actuales y futuros e impone las identidades de recursos y
producción; `lagged` aporta el capital histórico. Solo se selecciona la ecuación
de Euler restante. La esperanza condicional se aplica al residuo seleccionado
tras evaluar la política futura. La utilidad marginal futura debe permanecer
dentro de esa esperanza: multiplicar la ecuación por consumo futuro aleatorio
cambiaría el modelo estocástico.

Multiplicar por un factor conocido en `t`, distinto de cero, conserva una
igualdad condicionada a esa información. Multiplicar por una variable futura
aleatoria generalmente no lo hace. Mantenga expresiones como
`return_next / consumption_next` dentro del residuo que se integra.

En un modelo de crecimiento con trabajo endógeno, una alternativa útil es
`controls=(consumption, labor)`: imponga producción y recursos en `current`,
transfiera el ahorro resultante mediante `transition` y seleccione las
condiciones intratemporal y de Euler. Los notebooks MAV T02_A/T02_B muestran
esta construcción. Cambiar los controles exige adaptar residuos, cotas y
transición; añadir controles sin ecuaciones independientes adicionales no
define un sistema resoluble.

## Qué resuelve la iteración temporal

Cada iteración mantiene fija la **política futura interpolada anterior**. En
cada nodo de estados resuelve las ecuaciones condicionales para los controles
actuales mediante mínimos cuadrados no lineales con cotas. Después amortigua
opcionalmente la actualización nodal, interpola los controles nuevos y vuelve a
evaluar las ecuaciones con la política actualizada. Es una iteración de políticas
basada en Euler, no una mejora de Howard para una función de valor de Bellman.
Los métodos de proyección representan funciones en una base finita y evalúan
residuos; [Judd (1992)](https://kenjudd.org/wp-content/uploads/2022/05/1992-Projection-Methods-For-Solving-Aggregate-Growth-Models.pdf)
desarrolla el marco numérico general.

| Opción | Significado |
| --- | --- |
| `tol=1e-7` | Máximo residuo escalado KKT/de ecuaciones y actualización absoluta de controles aceptados en nodos |
| `max_iter=200` | Presupuesto de actualizaciones externas de política |
| `inner_max_iter=100` | Presupuesto de evaluaciones de función no lineal por nodo |
| `damping=1.0` | Fracción de la actualización propuesta, en `(0, 1]` |
| `max_nodes=10000` | Límite de construcción de grilla; no es una meta de precisión |
| `strict=False` | Conserva el resultado fallido con advertencia; `True` lanza una excepción por falta de convergencia numérica |

`inner_max_iter` acota las iteraciones de Newton amortiguado por nodo de
colocación. El número de evaluaciones sigue las llamadas al residuo; las
aproximaciones numéricas del Jacobiano pueden añadir llamadas al residuo cuando
se omiten `transition_jacobian` o `residual_jacobian` (con
`EquilibriumProblem.from_model`, las derivadas compiladas de las ecuaciones se
reutilizan automáticamente mediante `TrajectoryJacobian`).
`result.callback_evaluations` registra por separado `"transition"`,
`"residual"`, `"transition_jacobian"`, `"residual_jacobian"`, `"admissible"`,
`"control_bounds"`, `"shock_rule"` y `"trial_rejections"`. Ni ese presupuesto ni
`max_nodes` garantizan un límite de tiempo o memoria total.

La política inicial puede ser un callable, un vector constante `(nx,)` o un
arreglo nodal que coincida con la grilla construida. Debe ser finita y cumplir
cotas y `admissible` en los nodos. El problema interno impone cotas por control
y utiliza búsqueda de paso factible segura en el dominio (`trial_rejections` en
`history` y `callback_evaluations`) si una actualización completa viola cotas o
`admissible` en los nodos o en estados sucesores de cuadratura; `admissible` es
una comprobación de factibilidad, **no** un optimizador general con
restricciones no lineales. El ajuste tensorial de Chebyshev utiliza
factorizaciones LU 1D separables por eje de estado en lugar de invertir una
matriz de Vandermonde densa multidimensional
(`policy.collocation_matrix_allocated` es `False`). Diseñe parametrización y
cotas para que el residuo sea finito en los controles de prueba relevantes.

`result.converged` exige que pase la auditoría nodal actualizada, que el cambio
de controles sea pequeño y que no fallen soluciones por nodo. Un gradiente de
mínimos cuadrados pequeño o una actualización pequeña no bastan. El historial
incluye `policy_change`, `max_kkt_residual`, `inner_failures`, evaluaciones y masa
de salida del dominio. Son condiciones numéricas sobre esa discretización;
compruebe por separado la precisión económica.

## Comprobar puntos que no se utilizaron para resolver

```python
rng = np.random.default_rng(824)
points = rng.uniform([0.12, -0.04], [0.47, 0.04], size=(64, 2))
exact = lambda state: np.array([alpha * beta * output(state)])
comparison = compare_policies(
    problem, {"time iteration": result, "analytic": exact}, points, tol=1e-3,
)
print(comparison)
audit = audit_policy(problem, result, points, tol=1e-3)
assert audit.passed
print(audit.diagnostics)

reference = np.array([exact(state) for state in points])
policy_error = float(np.max(np.abs(result(points) - reference)))
assert policy_error < 3e-5
assert audit_policy(problem, exact, points, tol=1e-12).passed

def direct_euler(state, policy):
    savings = policy(state)[0]
    consumption = output(state) - savings
    expected_return = 0.0
    for innovation, weight in zip(nodes, weights):
        next_state = np.array([savings, innovation[0]])
        next_output = output(next_state)
        next_consumption = next_output - policy(next_state)[0]
        expected_return += weight * alpha * next_output / (savings * next_consumption)
    return 1 / consumption - beta * expected_return

np.testing.assert_allclose(
    [direct_euler(state, result) for state in points],
    audit.residuals["Euler"].to_numpy(), atol=1e-12,
)
print("maximum savings error against exact policy:", policy_error)
```

`audit_policy` evalúa la política recibida; nunca resuelve una sustituta. Los
puntos aleatorios anteriores son independientes de los nodos de colocación y
el cálculo directo de Euler verifica el puente `.mod` de forma independiente.
Una muestra finita aporta cobertura, no una cota del error supremo. Añada puntos
cerca de fronteras relevantes y cambios de restricciones activas.

| Campo de auditoría | Interpretación |
| --- | --- |
| `residuals` | Residuos esperados condicionales en las unidades originales de las ecuaciones |
| `kkt_residuals` | Residuos divididos por `residual_scale`, reemplazando las filas de complementariedad por el mapa natural |
| `diagnostics.feasible`, `future_feasible` | Los controles actuales y sucesores de cuadratura cumplen cotas y predicado económico |
| `state_inside`, `next_state_outside_mass` | Pertenencia al dominio y suma de ponderaciones de sucesores fuera del rectángulo declarado |
| `failure` | Fallo numérico o de dominio registrado en ese punto |
| `passed` | Todos los puntos cumplen tolerancia, factibilidad y comprobaciones de dominio por defecto |

Las evaluaciones fallidas pueden dejar indicadores infinitos en residuos o
masa de salida. Examine `failure` antes de interpretar esa masa como probabilidad.

`max_residual` y `max_kkt_residual` resumen las dos primeras tablas. Una
restricción activa puede tener residuo original no nulo y residuo KKT válido.
No compare residuos en unidades de consumo, utilidad marginal o recursos como
si fueran el mismo error. `residual_scale` es explícito; el paquete no presenta
un residuo genérico como error equivalente en consumo o bienestar.
`compare_policies` aplica el mismo problema y puntos a todos los métodos y
utiliza por defecto una auditoría estricta del dominio.

## Integrar la ecuación condicional

En cada estado actual, el algoritmo calcula
\(\sum_{j=1}^{q}w_j F(s,x,g(s,x,\varepsilon_j),\pi(g(s,x,\varepsilon_j)),\varepsilon_j)\).
No sustituye una ecuación no lineal por su evaluación en choques o controles
futuros medios. `gaussian_quadrature(covariance, order=3)` genera una regla de
Gauss–Hermite de media cero cuya covarianza ya está incorporada en los nodos.
No vuelva a multiplicarlos por la desviación estándar de la innovación.

Se admiten covarianzas correlacionadas y semidefinidas positivas singulares.
La integración utiliza el subespacio de autovalores positivos; el trabajo crece
como `order ** rank`, y `max_nodes` puede rechazar una regla demasiado grande.
`shock_rule(state)` puede suministrar nodos y ponderaciones dependientes del
estado, incluidas probabilidades condicionales de transición de Markov. Deben
ser probabilidades condicionadas al estado actual, no frecuencias incondicionales
de regímenes. Se rechazan probabilidades inválidas; no se normalizan silenciosamente.

La integración gaussiana es una regla finita de cuadratura, no una afirmación
de que los choques tienen soporte acotado. Incluso una masa de salida nula se
refiere solo a los nodos suministrados. Aumente el orden y examine las colas
económicas por separado.

El siguiente ejemplo comprueba momentos gaussianos y resuelve en un dominio de
productividad mayor para auditar siete nodos sin extrapolación. Ambas auditorías
evalúan **la misma política final** con reglas de integración diferentes.

```python
covariance = np.array([[0.0004, 0.00015], [0.00015, 0.0009]])
joint_nodes, joint_weights = gaussian_quadrature(covariance, order=3)
np.testing.assert_allclose(joint_weights @ joint_nodes, 0, atol=1e-15)
np.testing.assert_allclose(
    joint_nodes.T @ (joint_nodes * joint_weights[:, None]), covariance, atol=1e-14,
)

fine_nodes, fine_weights = gaussian_quadrature([[0.02**2]], order=7)
wide_problem = replace(problem, domain=((0.1, 0.5), (-0.09, 0.09)),
                       shock_nodes=fine_nodes, shock_weights=fine_weights)
warm_start = replace(result.policy, extrapolation="allow")
wide_result = solve_time_iteration(wide_problem, warm_start, orders=(11, 4), tol=1e-8)
assert wide_result.converged
same_domain_rule3 = replace(wide_problem, shock_nodes=nodes, shock_weights=weights)
audit3 = audit_policy(same_domain_rule3, wide_result, points, tol=1e-4)
audit7 = audit_policy(wide_problem, wide_result, points, tol=1e-4)
assert audit3.passed and audit7.passed
print("three-node / seven-node residuals:", audit3.max_residual, audit7.max_residual)
```

La extrapolación explícita se usa solo para formar el punto inicial del dominio
ampliado; `wide_result` conserva los criterios estrictos por defecto de
convergencia y evaluación. Euler exacto de Brock–Mirman cancela el producto
futuro, por lo que este ejemplo es inusualmente insensible a la cuadratura.
Comprueba conexiones e interpolación, no que tres nodos sean suficientes para
otros modelos no lineales.

## Elegir la base y el dominio de estados

| `basis` | Interpretación de `orders` | Propiedades útiles y limitaciones |
| --- | --- | --- |
| `"chebyshev"` | Grado escalar común o un grado por estado | Nodos tensoriales de Chebyshev–Lobatto; `(grado + 1)` nodos por dimensión. Aproximación suave; el total de nodos es el producto entre dimensiones |
| `"smolyak"` | Un único nivel positivo de grilla dispersa | Utiliza la base polinomial Smolyak existente; reduce el crecimiento tensorial, pero conserva costos de construcción, interpolación y solución por nodo |
| `"linear"` | Un grado por estado que define el mismo número de nodos tensoriales | Interpolación multilineal por tramos; útil cerca de quiebres, sin oscilaciones polinomiales entre valores nodales factibles para cotas constantes |

El nivel Smolyak no es un grado polinomial por dimensión; cambiar `basis`
conservando el mismo entero `orders` no mantiene fijos costo o precisión. El
adaptador limita la construcción dispersa a 12 dimensiones de estado y nivel 8,
además de `max_nodes`; son límites de trabajo, no afirmaciones de precisión
económica a esos tamaños. Las herramientas VFI/EGM y de interpolación existentes
siguen siendo útiles cuando sus objetivos y supuestos corresponden al problema;
esta interfaz trata residuos de equilibrio suministrados explícitamente.

La selección del dominio y la aproximación anisotrópica importan además del
número de nodos; [Judd, Maliar, Maliar y Valero (2014)](https://kenjudd.org/wp-content/uploads/2021/11/2014-Smolyak-Method-For-Solving-Dynamic-Economic-Models.pdf)
estudian estas decisiones en modelos dinámicos. Esta implementación usa la
base dispersa existente y una heurística separada de refinamiento Chebyshev;
no pretende reproducir todos los algoritmos de ese artículo.

Elija un dominio que incluya estados relevantes y examine transiciones desde
su interior y bordes. Ampliar el rectángulo sin aumentar grados puede reducir
la resolución donde hay mayor curvatura. Aumentar grados tampoco impide
controles inviables entre nodos. Evalúe cotas y `admissible` fuera de la grilla;
una curva suave no certifica factibilidad.

Dos opciones controlan decisiones diferentes:

- `extrapolation="raise"` controla la evaluación de la política devuelta y es
  el valor por defecto. `"allow"` extiende explícitamente el interpolante
  polinomial o lineal; no recorta el estado a la frontera más cercana.
- `require_domain=True` controla la aceptación de convergencia nodal. Con
  `False` se acepta convergencia condicionada a la extensión y se conservan los
  diagnósticos de salidas. No se desactivan comprobaciones de factibilidad económica.

Los controles de prueba utilizan internamente extensión polinomial/lineal para
no alterar silenciosamente un paso no lineal mediante recorte. Por defecto, la
aceptación final sigue exigiendo cierre del dominio bajo la regla suministrada.
Para un proceso gaussiano, cierre bajo cuadratura finita no significa cierre
bajo toda la distribución de choques.

## Refinar la aproximación con un presupuesto de trabajo

```python
adaptive = solve_adaptive_time_iteration(
    problem, initial, points, orders=(3, 2), tol=1e-4, max_solves=6,
)
print(adaptive.history)
assert adaptive.accepted

ladder = refine_time_iteration(
    problem, initial,
    levels=[{"orders": (7, 2)}, {"orders": (11, 4)}],
    points=points, tol=1e-4, max_solves=2,
)
print(ladder.history)

limited = refine_time_iteration(
    problem, initial, levels=[{"orders": (7, 2)}],
    points=points, tol=1e-4, max_solves=1,
)
assert not limited.accepted
```

El refinamiento adaptativo aumenta el grado de Chebyshev en la dirección del
estado con la mayor cola de coeficientes de grado alto. Esa cola es una
heurística, no una cota de error. La aceptación exige también comprobar Euler y
KKT de forma independiente y verificar el acuerdo entre políticas consecutivas
en los mismos puntos. Agotar `max_solves` no produce una aprobación.

La escalera explícita puede cambiar también `domain`, `shock_nodes` y
`shock_weights`. Así se comprueban por separado dominio e integración con los
mismos puntos de auditoría. `basis="smolyak"` utiliza la base existente de grilla
dispersa y un `orders` escalar indica su nivel; el adaptador anisotrópico
automático utiliza actualmente solo Chebyshev. `basis="linear"` interpola por
tramos sobre nodos tensoriales y resulta útil cerca de quiebres. Todos los
métodos necesitan comprobaciones independientes y cobertura de estados adecuada.

`PolicyRefinementResult.accepted` exige más que `result.converged`: la auditoría
final fuera de la grilla debe pasar, la solución interna debe converger y el
mayor cambio absoluto de controles respecto del nivel anterior, en los mismos
puntos, debe ser menor o igual a la tolerancia de refinamiento. El primer nivel
no puede establecer aceptación. `max_solves` trunca la escalera; `max_nodes`
limita la construcción y lanza una excepción si se pide una grilla demasiado
grande. Ninguno certifica precisión al llegar al límite de presupuesto.

Para interpretar los cambios, varíe una elección numérica a la vez: resolución,
cuadratura sobre una política y dominio fijos, y tamaño del dominio manteniendo
resolución comparable. Una escalera puede cambiar las tres, pero habrá varias
causas posibles para una respuesta distinta. Conserve juntos historial, política
y puntos de auditoría. Los puntos fijos permiten reproducibilidad; después de
seleccionar un nivel conviene una nueva muestra de validación independiente.

## Comparar aproximaciones locales con la misma esperanza

```python
local_policies, local_solutions = {}, {}
for order in (1, 2, 3):
    solution = model.solve(order=order)
    local_solutions[order] = solution
    rules = solution.decision_rules()
    assert tuple(rules.state_variables) == ("k",)
    assert tuple(rules.shock_names) == ("e",)
    capital_index = list(rules.variable_names).index("k")

    def local_policy(state, solution=solution, capital_index=capital_index):
        values = evaluate_perturbation(solution, [state[0]], [state[1]])
        return values[[capital_index]]

    local_policies[f"order {order}"] = local_policy
local_policies["global"] = adaptive.result
local_comparison = compare_policies(problem, local_policies, points, tol=1e-4)
print(local_comparison)
local_diagnostics = diagnose_perturbation(local_solutions[3])
print(local_diagnostics.summary())

local_seeded = solve_time_iteration(problem, local_policies["order 2"], orders=(7, 2), tol=1e-8)
assert local_seeded.converged
np.testing.assert_allclose(local_seeded(points), result(points), atol=1e-7)

batch = evaluate_perturbation(local_solutions[3], points[:, :1], points[:, 1:2])
assert batch.shape == (len(points), len(model.variables))
```

El estado global incluye productividad actual observada; el estado predeterminado
de la regla local en este ejemplo i.i.d. contiene solo capital heredado. El
código introduce explícitamente la productividad actual como innovación actual.
En un modelo AR(1), suministre productividad rezagada e innovación actual de
forma coherente con su ley de movimiento; no confunda productividad observada
actual con su rezago.

Todas las políticas comparadas utilizan las mismas identidades estáticas
exactas y regla de integración. Se comparan en los mismos puntos, no en
simulaciones o varianzas de choques diferentes. Aumentar el orden local puede
mejorar un entorno y aun así fallar lejos o violar una cota. No imponga un orden
preferido de errores antes de observarlos.

La regla de segundo orden también proporciona una inicialización factible
alternativa para la iteración temporal. El acuerdo entre dos inicializaciones
aporta evidencia numérica, no una prueba de unicidad. En otros modelos o dominios
mayores, una política inicial de Taylor puede violar cotas y debe reconsiderarse.

### Niveles, orden de coeficientes y escala de riesgo

`evaluate_perturbation` evalúa coeficientes ya presentes en la solución. Su
opción `order` puede truncar la aproximación disponible; no genera coeficientes
de tercer orden a partir de una solución de primero. Las convenciones son:

| Cantidad | Convención |
| --- | --- |
| `lagged_states` | Niveles de estados del período anterior en orden `decision_rules().state_variables`; vector o lote por filas |
| `shocks` | Innovaciones realizadas actuales en orden `shock_names`; omitirlas implica cero y un vector se comparte entre filas |
| `deviations=True` | Solo cambia los estados de entrada a desviaciones; la salida conserva niveles endógenos |
| Arreglo devuelto | Niveles actuales en orden `variable_names`; `(n_variables,)` o `(n_points, n_variables)` |
| `sigma` | Multiplicador no negativo de términos de riesgo anticipado; no reescala las innovaciones suministradas |

Por ejemplo, mantenga fija la innovación realizada al aislar los términos de
riesgo retenidos:

```python
third_order = local_solutions[3]
ss = third_order.decision_rules().ys
lagged_capital = np.array([float(ss["k"])])
realized_innovation = np.array([0.01])
with_risk = evaluate_perturbation(third_order, lagged_capital, realized_innovation, sigma=1)
without_risk = evaluate_perturbation(third_order, lagged_capital, realized_innovation, sigma=0)
print("retained risk-term contribution:", with_risk - without_risk)
```

En notación de coeficientes, se incluye la constante de segundo orden
`ghs2 * sigma**2 / 2` y los términos mixtos de tercero
`(ghxss @ state_deviation + ghuss @ innovation) * sigma**2 / 2`, junto con los
términos lineales, cuadráticos y cúbicos y sus factoriales de Taylor. La
covarianza utilizada al resolver ya está incorporada en esos coeficientes.
`sigma=0` evalúa la expansión retenida sin esos términos de riesgo; no vuelve a
resolver un modelo estocástico recalibrado. Cambiar covarianzas relativas
requiere resolver de nuevo. Este ejemplo logarítmico con
depreciación completa no tiene respuesta de ahorro precautorio; sus
contribuciones pequeñas o nulas de riesgo no deben generalizarse a modelos RBC.

Los órdenes dos y tres usan contracciones tensoriales directas sin construir
vectores de Kronecker. Son evaluaciones de Taylor sin poda, útiles para
inicialización y residuos. Una simulación podada conserva componentes de orden
separados de la historia de estados y es un cálculo distinto.
`diagnose_perturbation` informa raíces de transición cercanas al círculo unitario
y rango/condicionamiento del Jacobiano dinámico original cuando se conserva.
No sustituye una clasificación de Blanchard–Kahn ni certifica unicidad no lineal.
El rango utiliza los bloques concatenados rezagado/actual/adelantado; el número
de condición depende de las unidades de ecuaciones y variables. Un Jacobiano
no conservado se informa como no comprobado. Una raíz cercana a uno indica
persistencia y sensibilidad, no demuestra por sí sola invalidez del equilibrio.

**Condicionamiento de Sylvester de orden superior y reutilización exacta de factorizaciones.**
El solucionador recursivo de Schur real (Bartels–Stewart) indexa su caché LU
mediante valores exactos en punto flotante IEEE-754 (`float` para bloques
$1\times 1$ y cuatro flotantes para bloques $2\times 2$), eliminando la colisión
por redondeo a 12 decimales reportada en 5.7.1 cuando productos de Schur
cercanos a uno reutilizaban una factorización.
`diagnose_perturbation(solution, max_curvature_order=3)` informa
`schur_block_condition` (cota inferior rápida por bloques diagonales de Schur
del número de condición de orden superior), `curvature_operator_condition` y
`curvature_coupling_amplification` (condicionamiento del operador completo de
Kronecker y amplificación por acoplamiento fuera de la diagonal, resumidos en
`curvature_conditioning`), `sylvester_residuals` (`ghxx`, `ghs2`, `ghxxx`,
`ghxss`, `ghuss`) y `sylvester_work` (`SylvesterWorkCounts`), distinguiendo el
condicionamiento del operador de orden superior del número de condición del
Jacobiano dinámico de primer orden.

La poda evita generar recursivamente términos de orden superior al retenido y
transporta componentes de estado separados. La diferencia entre una regla de
Taylor sin poda y una recursión podada se desarrolla en
[Andreasen, Fernández-Villaverde y Rubio-Ramírez](https://public.econ.duke.edu/~jfr23/Current/Pruning_20140120.pdf).
Auditar la regla sin poda no equivale a auditar cualquier historia simulada con
poda. Utilice los métodos de simulación podada de la solución para esas
trayectorias; no reconstruya la poda alimentando estados simulados totales
repetidamente a `evaluate_perturbation`.

## Complementariedad y un ejemplo con cota activa

Las cotas restringen controles admisibles; no convierten ecuaciones en
inecuaciones. Con `complementarity={equation_index: control_index}`, sean `x`
el control y `F` su residuo esperado condicional. La convención de signos es:

\[
x=\ell\Rightarrow F\ge0,\qquad
\ell<x<u\Rightarrow F=0,\qquad
x=u\Rightarrow F\le0.
\]

El residuo de mapa natural es
\(x-\operatorname{clip}(x-F/d,\ell,u)\), donde `d` es la escala positiva de la
ecuación. Sus ceros expresan esas condiciones KKT. Esta proyección es parte de
la ecuación de complementariedad; no recorta la política devuelta ni la
transición de estados. La factibilidad se comprueba por separado. El mapeo es
explícito porque el índice del control puede diferir del de su ecuación.

Aquí un ajuste no negativo `x` minimiza `(x-s)**2 / 2`. La rentabilidad futura
es `0.5*s` y no depende de la decisión, por lo que la política exacta es
`max(s, 0)`. Este pequeño problema económico con restricción aísla el contrato
de signos y cota activa sin resolver ocultamente un problema de referencia.

```python
constrained_problem = EquilibriumProblem(
    state_names=("profitability",), control_names=("adjustment",),
    domain=((-1, 1),), transition=lambda s, x, e: 0.5 * s,
    residual=lambda s, x, sp, xp, e: x - s,
    control_bounds=[(0, 2)], complementarity={0: 0},
)
constrained = solve_time_iteration(
    constrained_problem, [0.5], basis="linear", orders=2, tol=1e-7,
)
constraint_points = np.linspace(-0.98, 0.98, 41)[:, None]
constraint_audit = audit_policy(constrained_problem, constrained, constraint_points, tol=2e-6)
assert constrained.converged and constraint_audit.passed
np.testing.assert_allclose(
    constrained(constraint_points)[:, 0], np.maximum(constraint_points[:, 0], 0), atol=2e-6,
)
assert constraint_audit.max_residual > 0.9
assert constraint_audit.max_kkt_residual < 2e-6
```

El residuo original no nulo con rentabilidad negativa es la desigualdad correcta
en la cota inferior. Exigir que sea cero rechazaría una esquina económicamente
válida. La base lineal representa exactamente el quiebre en este ejemplo
especial; fronteras no lineales arbitrarias siguen exigiendo refinamiento y
comprobaciones fuera de la grilla. Las KKT de equilibrio son condiciones
necesarias suministradas por el modelador, no una garantía de óptimo global de
bienestar único.

## Exportar resultados y graficar una sección explícita de la política

Las auditorías de políticas, los resultados de iteración temporal y de
refinamiento, y los diagnósticos locales ofrecen `to_frame()` /
`to_dataframe()`, `to_markdown()`, `to_latex()` y `to_typst()` mediante los
formatos compartidos del paquete. Las tablas de refinamiento contienen el
historial de soluciones y los diagnósticos locales contienen raíces de transición.

```python
print(audit.to_markdown(index=False))
slice_points = np.column_stack((np.linspace(0.12, 0.47, 40), np.zeros(40)))
figure = result.plot(slice_points, state="inherited_capital", controls=["savings"], show=False)
axis = figure.axes[0]
assert adaptive.plot(slice_points, state=0, ax=axis) is axis
latex_table = local_diagnostics.to_latex(index=False)
typst_table = adaptive.to_typst(index=False)

import matplotlib.pyplot as plt
plt.close(figure)
```

Las gráficas requieren coordenadas completas del estado; aquí se fija la
productividad explícitamente en cero. `state` elige la coordenada horizontal y
`controls` selecciona controles; se conserva el orden de puntos. Al recibir
`ax`, el método devuelve ese eje; en otro caso devuelve una figura nueva.
Matplotlib solo se carga al graficar y no se muestra una figura salvo con
`show=True`. El refinamiento grafica su política final con los mismos puntos explícitos.

`result.to_frame()` contiene nodos de estado y controles nodales;
`audit.to_frame()` contiene diagnósticos por punto. Las tablas de residuos
detallados están disponibles por separado. Los exportadores devuelven cadenas;
no escriben archivos ni publican resultados.

## Salidas del dominio, fallos y límites

La convergencia por defecto exige residuos nodales, tolerancia del cambio de
política, factibilidad y cierre bajo la regla de integración suministrada. Las
políticas lanzan una excepción fuera de su dominio. Para un dominio truncado
deliberadamente de choques persistentes, `extrapolation="allow",
require_domain=False` permite resolver **condicionado a la extensión del
interpolante**, conservando diagnósticos de salidas. No valida la región
omitida. En particular, las salidas de cuadratura en los extremos del ejemplo
AR(1) de MAV siguen visibles aunque pase una auditoría interior estricta.
No se recortan choques o políticas silenciosamente.

Un AR(1) estacionario también puede abandonar un dominio finito. Aumentar
persistencia u orden de cuadratura puede incrementar salidas en los bordes sin
señalar un error algebraico. Amplíe el dominio, examine un interior justificado
o acepte extrapolación condicional explícitamente e informe esa elección.
Recortar choques altera la ley de integración y puede producir una aprobación
engañosa. `audit_policy` conserva el criterio estricto por defecto aunque se
haya resuelto con `require_domain=False`; una auditoría condicional exige
indicar también `require_domain=False` en su propia llamada.

Ante fallo numérico o límite de iteraciones se conserva `converged=False` con
advertencia; `strict=True` lanza una excepción. El resultado ordinario conserva
la última iteración e historial. Formas, probabilidades o mapeos inválidos y
grillas demasiado grandes son errores de entrada y pueden lanzar una
excepción inmediatamente.

Cuando falle una solución, use el historial para distinguir fallo por nodo,
estancamiento, salidas del dominio y controles futuros inviables. Verifique
primero temporización y signos; después examine escalas, inicialización
factible, amortiguación y un dominio pequeño bien resuelto. Más iteraciones no
reparan un estado omitido, una esperanza incorrecta o un mapeo incompatible de restricciones.

El método está orientado a equilibrios manejables con estados continuos.
Equilibrios múltiples, regímenes discontinuos, restricciones muy no lineales,
colas no acotadas y grandes espacios DSGE/HANK necesitan tratamiento específico.
Las comprobaciones finitas de residuos y refinamiento aportan evidencia sobre
la aproximación informada; no establecen existencia, unicidad, transversalidad
ni precisión en todos los estados.
