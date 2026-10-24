> 🇬🇧 [English](../dsge_stacked_newton.md) · 🇪🇸 Español

# Newton-Krylov apilado en el tiempo para modelos de previsión perfecta

`puremacro.dsge.stacked_newton` resuelve el problema determinista de frontera en dos puntos de [`solve_perfect_foresight`](dsge_build.md#8-simulacion-no-lineal-y-prevision-perfecta), con un estado inicial dado `y_init` y un estado terminal dado `y_end`,

```
f(y_{t+1}, y_t, y_{t-1}, eps_t) = 0,   t = 1, ..., T,
y_0 = y_init,   y_{T+1} = y_end,
```

sin ensamblar el jacobiano apilado para los productos jacobiano-vector. La dirección de Newton se obtiene a partir de estos productos mediante LGMRES, precondicionado con la inversa exacta del jacobiano tridiagonal por bloques en el tiempo evaluado en el estado estacionario. Es un port independiente del modelo de una implementación anterior, escrito con las convenciones de puremacro. Tanto `solve_perfect_foresight` como `extended_path` lo ofrecen mediante `linear_solver="krylov"`; su opción predeterminada es `linear_solver="direct"`. Algunos precondicionadores ensamblan una matriz apilada dispersa para factorizarla.

## Qué calcula

| Objeto | Función |
|---|---|
| `StackedProblem` | Residuo apilado `F(Y)` y el producto libre de matrices `J(Y) v` a partir de un `jvp_fn` analítico por fecha, diferencias centrales o un paso complejo; bloques por fecha `(A_t, B_t, C_t)` y bloques del estado estacionario |
| `BlockTridiagonalPreconditioner` | Inversa exacta por Thomas de bloques (o SuperLU) de `kron(I, B) + kron(S, A) + kron(S', C)`; `time_block=H` da Jacobi por bloques de H fechas |
| `StructuredBlockTridiagonalPreconditioner` | Protocolo para eliminaciones específicas del modelo: resoluciones diagonales por fecha con barridos Jacobi, hacia delante, hacia atrás o Gauss-Seidel simétrico, o una subclase que sobrescribe `apply` |
| `preconditioned_lgmres` | LGMRES o GMRES con reinicios con lado de precondicionamiento explícito y el residuo verdadero de la dirección devuelta |
| `solve_stacked_newton_krylov` | Armazón de Newton inexacto: término de forzamiento, paso acotado, búsqueda lineal de Armijo, historial lineal, puntos de control |
| `compare_horizons` | Aceptación por duplicación de horizonte `max_{t <= periods} abs(y^T_t - y^2T_t) <= tolerance`; exportada a nivel de paquete como `puremacro.dsge.compare_stacked_horizons` |
| `StackedNewtonResult`, `HorizonComparison` | Objetos de resultado inmutables con `to_dataframe`, `summary`, `to_markdown`, `to_latex`, `to_typst`; `StackedNewtonResult.to_perfect_foresight_result()` devuelve un `PerfectForesightResult` |

## El sistema apilado y la acción de su jacobiano

Con `A_t = df_t/dy_{t+1}`, `B_t = df_t/dy_t` y `C_t = df_t/dy_{t-1}`, el jacobiano apilado es tridiagonal por bloques en el tiempo y su acción sobre una dirección apilada `v` es

```
(J v)_t = A_t v_{t+1} + B_t v_t + C_t v_{t-1},    v_0 = v_{T+1} = 0.
```

`StackedProblem` evalúa este producto de una de cuatro formas:

- `jvp_fn(y_next, y_curr, y_prev, eps, v_next, v_curr, v_prev)`: producto analítico por fecha (exacto);
- `stacked_jvp_factory(Y) -> matvec`: un producto por lotes suministrado por el modelo;
- `jvp_method="central"` (por defecto): `(F(Y + h v) - F(Y - h v)) / 2h` con `h = fd_step * max(1, |Y|_inf) / |v|_inf`, dos evaluaciones del residuo por producto, precisión relativa de alrededor de 1e-9;
- `jvp_method="complex"`: `Im F(Y + i h v) / h` con `h = 1e-20`, una evaluación compleja, exacta a precisión de máquina, requiere un `equations_fn` compatible con números complejos (la misma restricción que `solve_perfect_foresight(method="complex")`); una función que descarta la parte imaginaria (`np.real`, `float()`, un dtype real) se detecta y se rechaza con `TypeError` en lugar de producir en silencio un producto nulo.

La convención de apilamiento es la de `solve_perfect_foresight`: la fecha `t` ocupa las filas `t*n_vars:(t+1)*n_vars`, `y_prev` en la primera fecha es `y_init` y `y_next` en la última fecha es `y_end`. Cada fila de `exogenous_path` es una fecha interior salvo que se indique `n_periods`: entonces una trayectoria de exactamente `T + 2` filas (el formato de `solve_perfect_foresight` con las dos filas de frontera) se recorta a las filas `1..T`, así que pase `n_periods` siempre que la trayectoria incluya filas de frontera; de lo contrario cada choque cae una fecha más tarde.

Un modelo cuyo residuo en la fecha `t` necesita también los valores exógenos de la fecha siguiente empaqueta ambos en cada fila, `exogenous_path[t] = (eps_t, eps_{t+1})`, con la última fila repitiendo `eps_T`. Empaquetar los propios valores, y no un índice de fecha, mantiene con sentido la regla de trayectoria exógena mantenida de `compare_horizons`.

## La iteración de Newton-Krylov inexacta

En un iterado `Y` con residuo `F`, la dirección `d` satisface `|J d + F|_2 <= eta |F|_2` con el término de forzamiento

```
eta = max(eta_min, min(eta_max, sqrt(|F|_inf / s))),    (eta_min, eta_max) = (1e-5, 0.1),
```

resuelto por el LGMRES de SciPy con `inner_m = 40`, `outer_k = 6` y como máximo `krylov_maxiter = 35` iteraciones externas. `krylov_method="gmres"` usa GMRES con reinicios; es una elección explícita, no un recurso automático. Una dirección devuelta al alcanzar el límite de Krylov no es un error: su estado y su residuo lineal verdadero se registran y se somete a la búsqueda lineal como cualquier otra dirección. `s` es la escala del residuo: por defecto la mayor entrada de los bloques del estado estacionario con suelo en uno; `jacobian_scale=1.0` da una regla absoluta del residuo. El movimiento está acotado en unidades de la trayectoria, `step = min(1, max_log_move * max(1, |Y|_inf) / |d|_inf)` con `max_log_move = 0.75`: siempre que `|Y|_inf <= 1` esto acota el movimiento absoluto en 0.75, y la forma relativa mantiene a un modelo escrito en niveles de orden 1e3 o 1e4 convergiendo en los mismos seis pasos de Newton que el modelo en escala unitaria (con una cota absoluta el mismo modelo necesita más de mil pasos); `max_log_move=None` desactiva la cota. El paso se acepta con la prueba de Armijo `|F(Y + step d)|_2^2 <= |F(Y)|_2^2 (1 - 1e-4 step)`, reduciendo a la mitad como máximo `max_backtracks = 23` veces (24 evaluaciones de prueba). Cada paso aceptado se registra en `linear_history` y, cuando se solicita, se pasa a `progress` y `checkpoint`.

La convergencia usa la regla consciente de la escala de `solve_perfect_foresight`: `|F|_inf <= tol * s` y la siguiente dirección de Newton por debajo de `step_tol * max(1, |Y|_inf)` con `step_tol = 1e-10`. Un sistema cuyas ecuaciones se multiplican por 1e-12 o 1e8 devuelve por tanto la misma trayectoria, y nunca se devuelve la conjetura inicial intacta. `step_tol=None` elimina la prueba del paso y acepta solo con el residuo, sin una resolución lineal final; con `jacobian_scale=1.0` la prueba es entonces `|F|_inf <= tol`. Cuando el residuo ya cumple `tol * s` pero la prueba del paso no puede cumplirse, típicamente porque el residuo está en un suelo de ruido (una resolución iterativa interna, ruido de diferencias finitas), la resolución lanza `StackedNewtonError` con `converged_by = "step_test_failure"` y un mensaje que nombra la causa y la opción `step_tol=None`. Todo fallo lanza `StackedNewtonError` con el último estado adjunto como `.result` (`converged=False`, `metadata["converged_by"]` nombrando la causa: `iteration_limit`, `line_search_failure`, `step_test_failure`, `linear_solve_failure` para un error o un valor no finito dentro de un producto jacobiano o de una aplicación del precondicionador, `nonfinite_direction` o `preconditioner_failure`); una resolución fallida nunca devuelve `converged=True`.

Hay una salvedad importante para quien lea registros de Krylov. El LGMRES de SciPy llama a su callback al inicio de cada iteración externa, antes de la comprobación del residuo. Cuando se agota el límite de iteraciones, la dirección devuelta nunca pasó por el callback, de modo que el último residuo registrado no la describe. Por eso `preconditioned_lgmres` recalcula `|J d + F|_2` para la dirección devuelta en coordenadas originales y el solucionador lo registra como `true_linear_residual` junto a `requested_tolerance`. La temporización del callback se verificó en SciPy 1.18; en SciPy 1.10 y 1.11 la tolerancia relativa se pasa como `tol` en lugar de `rtol`.

## Precondicionadores

**Inversa exacta del estado estacionario (por defecto).** Con bloques constantes `(A, B, C)` en `(y_end, y_end, y_end, eps_T)`, la recursión de Thomas por bloques

```
D_1 = B,  U_1 = D_1^{-1} A,  D_t = B - C U_{t-1},  U_t = D_t^{-1} A,
z_1 = D_1^{-1} r_1,  z_t = D_t^{-1} (r_t - C z_{t-1}),
x_T = z_T,  x_t = z_t - U_t x_{t+1},
```

invierte exactamente `J0 = kron(I_T, B) + kron(S, A) + kron(S', C)` (`S` es el desplazamiento superior). La preparación cuesta `O(T n^3)`, cada aplicación `O(T n^2)`, la memoria `O(T n^2)`: T factores LU densos de tamaño n más T-1 matrices de acoplamiento. Es la relajación de Laffargue-Boucekkine-Juillard escrita en forma matricial; la diferencia con `solve_perfect_foresight` es que la factorización se hace una sola vez, en el estado estacionario, y se reutiliza en cada iteración de Krylov de cada paso de Newton, mientras que el jacobiano actual solo actúa mediante productos. `method="splu"` factoriza la misma matriz dispersa con SuperLU. La recursión de Thomas por bloques necesita que todos los pivotes principales `D_t` sean no singulares, algo que una matriz apilada no singular no garantiza: una `B` de rango deficiente la rompe mientras SuperLU y `solve_perfect_foresight` tienen éxito. En coma flotante ese pivote rara vez es exactamente cero (el redondeo deja entradas del orden de 1e-17), así que un pivote se rechaza cuando la estimación recíproca de la condición de LAPACK cae por debajo de `n * eps`, y la factorización terminada se comprueba una vez con un lado derecho aleatorio fijo: un error hacia atrás normado `|J0 x - r|_inf / (|J0|_inf |x|_inf + |r|_inf)` por encima de 1e-10 también la rechaza (una resolución estable hacia atrás da alrededor de 1e-16 sea cual sea el condicionamiento de `J0`; los modelos regulares dan como máximo 2e-16). Los precondicionadores con nombre de `solve_stacked_newton_krylov` recurren entonces a SuperLU y lo registran en el nombre del precondicionador (`block_tridiagonal[splu, exact, thomas_pivot_fallback]`); solo una matriz apilada que SuperLU tampoco puede factorizar lanza `StackedNewtonError`, con el estado inicial adjunto.

**Jacobi por bloques temporales.** `time_block=H` elimina los enlaces entre bloques consecutivos de H fechas: una factorización de longitud H se reutiliza para cada bloque completo y un bloque residual más corto recibe la suya (etiquetas de bloque `repeat(arange(T)//H, n)`). `H = T` es la inversa exacta; `H = 1` (la opción con nombre `"time_block_jacobi"`) conserva solo `B^{-1}` por fecha.

**Bloques actuales.** `"current_block_tridiagonal"` refactoriza el precondicionador a partir de los bloques por fecha actuales en cada paso de Newton. La resolución de Krylov converge entonces en un ciclo interno, lo que lo convierte en un método de Newton exacto al coste de `3 n` llamadas a las ecuaciones por fecha y paso, el mismo perfil de coste que `solve_perfect_foresight`. Existe sobre todo como oráculo.

**Protocolo estructurado.** `StructuredBlockTridiagonalPreconditioner(n_periods, n_vars, diag_solve=, lower_apply=, upper_apply=, sweep=)` toma resoluciones diagonales por fecha `D_t^{-1} r_t` y las acciones fuera de la diagonal `C_t x` y `A_t x`, y aplica un barrido Jacobi por bloques, hacia delante `(D + L)^{-1}`, hacia atrás `(D + U)^{-1}` o simétrico `(D + U)^{-1} D (D + L)^{-1}`. Una subclase puede en cambio sobrescribir `apply` con una eliminación global, de modo que un solucionador de transiciones específico del modelo podría aportarla a través de ella; en esta versión ningún módulo de puremacro usa el protocolo. `solve_stacked_newton_krylov` acepta cualquier objeto con `as_linear_operator()`, cualquier `LinearOperator` de SciPy o una fábrica `f(Y) -> operator` evaluada en cada iterado de Newton.

### Por qué el acoplamiento temporal pertenece al precondicionador

Medido en la transición de crecimiento Cobb-Douglas de 20 sectores de `tests/test_dsge_stacked_newton.py` (`n_vars = 40`, `T = 100`, 4,000 incógnitas, `tol = 1e-9`, `krylov_maxiter = 40`, portátil de desarrollo, tiempos solo indicativos):

| Solucionador lineal | Pasos de Newton | Iteraciones externas de LGMRES por paso | Segundos | Diferencia máxima absoluta con `solve_perfect_foresight` |
|---|---|---|---|---|
| `solve_perfect_foresight` (bloques densos por diferencias finitas, SuperLU) | 5 | no aplicable | 1.22 | referencia |
| `"steady_block_tridiagonal"` | 5 | 2, 2, 2, 2, 2, 2 | 0.15 | 3.5e-11 |
| `"steady_splu"` | 5 | 2, 2, 2, 2, 2, 2 | 0.11 | 3.5e-11 |
| `"time_block_jacobi"` (H = 1) | 6 | 4, 5, 5, 6, 7, 10, 11 | 3.4 | 1.2e-12 |
| `"current_block_tridiagonal"` | 4 | 2, 2, 2, 2, 2 | 1.4 | 8.9e-15 |
| `None` | falla | 40, 40, 40, 40, 40, 40, 40 (límite) | 24 | residuo estancado en 0.31 |

Los recuentos de iteraciones externas son las iteraciones externas de LGMRES iniciadas (una más que los ciclos internos completados de una resolución convergida); cada lista incluye la resolución final de comprobación de convergencia. La inversa exacta estacionaria necesita un ciclo interno por paso de Newton; eliminar los enlaces temporales multiplica el trabajo de Krylov; eliminar el precondicionador por completo hace fallar a LGMRES en un sistema de 4,000 incógnitas: el acoplamiento temporal debe estar dentro del precondicionador.

## Ejemplo ejecutable

El modelo de Ramsey de `tests/test_dynare_perfect_foresight.py`, una transición desde la mitad del capital de estado estacionario:

```python
import numpy as np
from puremacro.dsge import solve_perfect_foresight
from puremacro.dsge.stacked_newton import StackedProblem, solve_stacked_newton_krylov, compare_horizons

alpha, beta, delta, sigma = 0.33, 0.96, 0.10, 1.0
r_ss = 1 / beta - (1 - delta)
k_ss = (alpha / r_ss) ** (1 / (1 - alpha))
c_ss = k_ss ** alpha - delta * k_ss

def equations_fn(y_plus, y_curr, y_lag, eps):
    c_p, k_p = y_plus
    c, k = y_curr
    c_m, k_m = y_lag
    A = float(eps)
    euler = c ** (-sigma) - beta * c_p ** (-sigma) * (alpha * A * k ** (alpha - 1) + 1 - delta)
    resource = k - (A * k_m ** alpha + (1 - delta) * k_m - c)
    return [euler, resource]

y_ss = np.array([c_ss, k_ss])
y_init = np.array([c_ss, 0.5 * k_ss])
T = 60
problem = StackedProblem(equations_fn, y_init, y_ss, np.ones(T), variable_names=["c", "k"])
result = solve_stacked_newton_krylov(problem, tol=1e-8)
print(result.summary())
print(result.krylov_outer_iterations())          # (2, 2, 2, 2, 2, 2, 2)

reference = solve_perfect_foresight(equations_fn, y_init=y_init, y_ss=y_ss,
                                    exogenous_path=np.ones(T), n_periods=T, tol=1e-8,
                                    variable_names=["c", "k"])
print(np.max(np.abs(result.to_numpy() - reference.path.to_numpy())))   # about 5e-11

longer = solve_stacked_newton_krylov(
    StackedProblem(equations_fn, y_init, y_ss, np.ones(2 * T), variable_names=["c", "k"]), tol=1e-8)
print(compare_horizons(result, longer, periods=20, tolerance=1e-5).summary())
```

La comparación de horizontes pasa con 60 frente a 120 fechas (`max_abs_difference` alrededor de 1e-7) y falla con 30 frente a 60 (3e-3): la condición terminal se siente dentro de las primeras veinte fechas de una resolución de 30 fechas. Pasar en un par de horizontes es evidencia de estabilidad de horizonte de las variables retenidas solo sobre la ventana; no es un certificado para el problema de horizonte infinito. Cuando ambas entradas son objetos `StackedNewtonResult`, `compare_horizons` exige entradas convergidas, fronteras idénticas y la misma trayectoria exógena mantenida más allá del horizonte corto; los arrays, DataFrames y entradas `PerfectForesightResult` no llevan fronteras ni trayectoria exógena, así que se comparan tal cual (un `PerfectForesightResult` debe seguir estando convergido).

## Cuándo usarlo y sus límites

- Úselo cuando el modelo ya tenga evaluaciones baratas del residuo y o bien un `jvp_fn` analítico o un `equations_fn` compatible con complejos, y `T * n_vars` sea lo bastante grande como para que ensamblar `3 n_vars` columnas por diferencias finitas por fecha y factorizar la matriz apilada en cada paso de Newton domine. Con productos por diferencias centrales la resolución lineal es precisa a aproximadamente 1e-9 relativo, suficiente para la iteración de Newton pero no para una identidad de inversa exacta a 1e-12; use `jvp_fn` o `jvp_method="complex"` para eso.
- La memoria de la inversa exacta estacionaria es `O(T n^2)`: `n_vars = 200`, `T = 500` necesita unos 320 MB; `n_vars = 1000`, `T = 200` unos 3.2 GB (`BlockTridiagonalPreconditioner.memory_bytes` da la cifra exacta). Los sistemas con miles de incógnitas por fecha quedan fuera del alcance de este operador genérico; necesitan una eliminación estructurada (la vía `StructuredBlockTridiagonalPreconditioner`), y no se incluye ninguna. La estructura por bloques por sí sola no establece la velocidad ni la memoria a esa escala.
- Este módulo no resuelve ni valida ninguna transición a gran escala, y no incluye ningún precondicionador específico de un modelo: solo las inversas tridiagonales por bloques y el protocolo estructurado anteriores. Las identidades lineales anteriores no garantizan por sí solas la precisión numérica ni la convergencia no lineal.
- La complementariedad mixta (`mcp=True`) y el modo rodante `surprise` de `solve_perfect_foresight` no están disponibles aquí.
- `direct_below` intenta una resolución previa con `hybr` de MINPACK para sistemas apilados de como máximo ese número de incógnitas y la conserva cuando reduce el residuo (el valor por defecto, 0, mantiene cada resolución en la ruta de Krylov para que `linear_history` esté siempre completo).

## Qué está validado y qué no

`tests/test_dsge_stacked_newton.py` comprueba, con tolerancias declaradas:

- oráculos de álgebra lineal: las inversas por Thomas de bloques y SuperLU igualan resoluciones densas a 1e-12 para bloques constantes y por fecha, el Jacobi por bloques temporales iguala la resolución diagonal por bloques enmascarada para bloques constantes y por fecha, linealidad e invarianza de escala, el operador copia sus bloques (modificar las entradas tras la construcción no cambia nada), SuperLU sobre un ensamblaje singular lanza `ValueError`, los cuatro barridos estructurados frente a las fórmulas densas con `D`, `L`, `U`;
- salvaguardas de pivote: una `B = U V` de rango deficiente cuyo pivote LU es de nivel de redondeo y no cero (dos semillas aleatorias, números de condición apilados 616 y 941) es rechazada por Thomas de bloques, invertida por SuperLU a 1e-12 y resuelta por el solucionador por defecto mediante el recurso a SuperLU hasta la trayectoria de `solve_perfect_foresight` a 1e-8; un pivote que supera la prueba de condición pero hace que la eliminación sin pivoteo pierda unos diez dígitos (`B = [[1, 1], [1, 1 + 1e-12]]`, número de condición apilado 1.3e3) es rechazado por la comprobación del error hacia atrás; la comprobación nunca salta en los modelos regulares (error hacia atrás como máximo 1e-14, observado 2e-16);
- productos apilados: las diferencias centrales coinciden con el jacobiano denso columna a columna construido con la propia regla de paso de `solve_perfect_foresight` a 1e-6, el paso complejo y el jacobiano analítico de Ramsey a 1e-7 frente a él (la propia referencia por diferencias finitas es precisa a unos 2e-9) y a 1e-12 entre sí; un `equations_fn` que descarta la parte imaginaria se rechaza bajo el paso complejo;
- la identidad de inversa exacta: un precondicionador a partir de los bloques actuales hace que LGMRES y GMRES converjan a 1e-12 relativo en un ciclo interno sobre lados derechos aleatorios, para ambos lados;
- la salvedad de la temporización del callback: un caso 4 por 4 en el que se agota el límite y el residuo recalculado difiere del último residuo del callback en más de 0.01; la firma de SciPy anterior a 1.12 (`tol` en lugar de `rtol`) se ejercita sustituyendo solucionadores con la firma antigua;
- paridad con `solve_perfect_foresight` en la transición de Ramsey, un choque de productividad anticipado en `t = 5`, una transición histval a endval y el modelo de crecimiento de 20 sectores, todo a 1e-8 (observado de 5e-11 a 7e-11 en Ramsey, 3.5e-11 en el modelo de crecimiento); el precondicionamiento por la derecha y por la izquierda y GMRES coinciden con él a 1e-9; invarianza al reescalado de ecuaciones a 1e-9 para escalas de 1e-12 a 1e8 (la prueba del paso acepta una dirección final de hasta `1e-10 * max(1, |Y|_inf)`, así que dos resoluciones convergidas pueden diferir en esa cantidad; observado 2.6e-10); invarianza al reescalado de variables: el modelo de Ramsey en unidades de 1e-3, 1, 1e3 y 1e4 converge con las opciones por defecto en como máximo diez pasos a la trayectoria en escala unitaria a 1e-9 relativo, con los mismos recuentos de Newton y de Krylov para las unidades grandes;
- el contrato de fallo: una búsqueda lineal fallida (24 evaluaciones de prueba por defecto, siendo el paso registrado el último probado), el límite de iteraciones, un error o un valor no finito dentro de una resolución lineal (un precondicionador estructurado que se vuelve no finito con precondicionamiento por la derecha y por la izquierda, un `jvp_fn` que lanza una excepción o devuelve NaN, un operador del usuario que lanza una excepción) y un residuo en un suelo de ruido que cumple la tolerancia mientras la prueba del paso no puede cumplirse lanzan cada uno `StackedNewtonError` con el último estado adjunto y la causa en `converged_by`; un límite de Krylov agotado por sí solo se registra y no es fatal (una resolución con el límite alcanzado sigue coincidiendo con `solve_perfect_foresight` a 1e-8); una `B` estacionaria de rango 1 con matriz apilada no singular converge mediante el recurso a SuperLU (y con `steady_splu` y `current_block_tridiagonal`) a la trayectoria de `solve_perfect_foresight`, mientras que una matriz estacionaria que no puede factorizarse lanza `StackedNewtonError` con el estado inicial adjunto; `step_tol=None` acepta solo con el residuo;
- objetos de resultado: la trayectoria almacenada y los registros del historial son de solo lectura, las copias son modificables, y los resultados sobreviven a pickle, a copias profundas y a la exportación JSON del historial;
- reglas de comparación de horizontes y el estadístico de 30/60/120 fechas;
- una regresión de clasificación de precondicionadores marcada como `slow`.

Estas pruebas no establecen convergencia para modelos arbitrarios ni rendimiento a cualquier escala. Véanse los [benchmarks de escalamiento](performance_scaling.md) para las cargas medidas y sus límites. Las versiones de SciPy anteriores a 1.18 no se validan aquí más allá de la sustitución de firma indicada.

## Relación con `solve_perfect_foresight` y `extended_path`

Desde 5.6.0, `solve_perfect_foresight(..., linear_solver="krylov")` construye un `StackedProblem` a partir de las ecuaciones, condiciones iniciales y terminales y trayectoria de choques resueltas. Devuelve un `PerfectForesightResult` con `linear_solver`, `solver_diagnostics` y el contexto conservado para auditar la trayectoria. `extended_path` pasa el mismo backend y `krylov_options` a cada resolución sucesiva. Las resoluciones sucesivas con sorpresas también admiten este backend. Las cotas MCP con Krylov están explícitamente fuera de soporte.

Los fallos numéricos de Krylov conservan la última trayectoria y los diagnósticos, emiten una advertencia y devuelven `converged=False`; no cambian silenciosamente al backend directo. `tests/test_dsge_pf_krylov.py` cubre paridad de trayectorias no lineales, referencias congeladas de condiciones terminales de Dynare, llamadas sucesivas de extended path y sorpresas, validación de opciones e información de fallos.

El adaptador introducido en 5.7.0 utiliza productos jacobiano-vector analíticos para ecuaciones compiladas; las funciones ordinarias conservan productos numéricos. Use `solve_stacked_newton_krylov` directamente para proporcionar un `StackedProblem`, `jvp_fn` o `stacked_jvp_factory` propio. El resultado independiente puede convertirse con `StackedNewtonResult.to_perfect_foresight_result()`. Consulte los [flujos de transición](dsge_transitions.md) para continuación, dominios y reutilización de precondicionadores con bloques idénticos.

## Documentación relacionada

- [Cuaderno de bocetos DSGE y Dynare](dsge_build.md), sección 8, para `solve_perfect_foresight`, transiciones MCP y `extended_path`.
- [Estado de validación estructural](STRUCTURAL_VALIDATION_STATUS.md) para el cálculo soportado y el contrato de aceptación de esta superficie.
- [Equilibrio general espacial cuantitativo](spatial_and_trade_ge.md) para el solucionador estático de Allen-Arkolakis.
