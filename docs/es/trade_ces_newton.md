> 🇬🇧 [English](../trade_ces_newton.md) · 🇪🇸 Español

# Newton por bloques exacto para CES sobre la contabilidad consistente

`puremacro.trade.ces_newton` resuelve el equilibrio de comercio de la
[contabilidad de comercio consistente](trade_accounting.md) con una tecnología
de producción CES anidada en tres niveles, usando el jacobiano exacto en lugar
de diferencias finitas. Cada paso de Newton cuesta dos factorizaciones LU densas
`M x M` más un pequeño complemento de Schur macro. En una tabla sintética de 64
países por 12 sectores (`M = 768`, 1791 incógnitas) un arancel del 10% de un
país con elasticidades suaves (0.1, 0.1, 0.5) se resuelve en 4 iteraciones de
Newton, unos 1.4 s en un portátil. Todo estado devuelto ha superado un
certificado de flujos independiente; no se devuelve nada más.

```python
import numpy as np
from puremacro.trade import calibrate_trade_model
from puremacro.trade.ces_newton import (NestedCESTechnology, solve_ces_block_newton,
                                        continue_tariff_homotopy, certify_ces_equilibrium)

# Two countries, one sector, two final uses (C, I): a hand-balanced table in the
# calibrate_trade_model layout (rows: deliveries, net taxes, labour, capital).
Z = np.array([[10., 12.], [8., 14.]])
F = np.array([[30., 8., 20., 20.], [15., 15., 50., 18.]])
production_tax = np.array([4., 6.]); final_tax = np.array([2., 1., 3., 2.])
output = Z.sum(1) + F.sum(1)
va = output - Z.sum(0) - production_tax
table = np.vstack([np.hstack([Z, F]), np.r_[production_tax, final_tax],
                   np.r_[2 * va / 3, np.zeros(4)], np.r_[va / 3, np.zeros(4)]])
calib = calibrate_trade_model(table, ns=1, nc=2, nfd=2, country_codes=["A", "B"])

tau = np.ones((2, 1, 2)); tau[1, 0, 0] = 1.2        # A taxes intermediate imports from B at 20%
tau_fd = np.ones((2, 2, 2)); tau_fd[1, :, 0] = 1.2  # and final imports from B at 20%
technology = NestedCESTechnology(sigma_va_materials=0.3, sigma_sectors=0.5,
                                 sigma_origins=1.5, rho_va=0.8)
result = solve_ces_block_newton(calib, tau, tau_fd, technology=technology)
print(result.summary())
print(result.equilibrium.w_sol.ravel(), result.certificate["current_account"])

path = continue_tariff_homotopy(calib, tau, tau_fd, technology=technology)
print(path.stages_frame())
audit = certify_ces_equilibrium(calib, path.equilibrium.x_sol, tau=tau, tau_fd=tau_fd,
                                technology=technology)
print(max(audit.values()))
```

`result.equilibrium` es un `TradeEquilibriumResult` ordinario con
`metadata["accounting"] == "consistent"` y
`metadata["effective_method"] == "ces_block_newton"`. Sirve cualquier tabla que
acepte la contabilidad consistente: la tabla OCDE incluida (`load_icio_data()`),
`synthetic_icio_table` de cualquier tamaño, o una tabla de proveedor condensada
en agregados de uso final C/I/X no negativos. Dentro del repositorio, el
fixture congelado OCDE 2019 de tres regiones que usan las pruebas se carga con
`tools.reference_validation.validate_oecd.load_fixture()` seguido del
auxiliar privado `puremacro.trade._oecd_icio.condense_final_demand` y de
`puremacro.trade.data.package_mrio_to_calibration_result`.

## Qué calcula

Las ecuaciones son exactamente las de la contabilidad consistente: valoración a
precios de productor y de comprador, costos factoriales homogéneos, un impuesto
a la producción sobre el ingreso por ventas, participaciones de gasto final con
impuestos, devoluciones de suma fija, ahorro externo fijo en unidades del
numerario y el primer precio de productor igual a uno. Solo cambia la
tecnología de los insumos intermedios. La celda `j` (país `c`, sector `s`)
compra a la celda de origen `i` a `r_ij = p_i tau_ij` y combina los insumos
mediante

```
h_sj = CES_so({r_ij}_{i in s}; u0_ij)            origins within a material sector
m_j  = CES_ss({h_sj}_s; eta0_sj)                 material sectors
v_j  = CES_rho({w_c, r_c}; (1 - alpha_j, alpha_j))   labour and capital
c_j  = CES_sv({v_j, m_j}; (v0_j, A0_j) / (1 - t_j))  value added and materials
```

con `CES_sigma(x; s) = (sum_i s_i x_i^(1 - sigma))^(1 / (1 - sigma))`, la media
geométrica en `sigma = 1` y el valor uno en un nido vacío. Las participaciones
de referencia son `u0_ij = a0_ij / sum_{k in s} a0_kj`,
`eta0_sj = sum_{i in s} a0_ij / A0_j`, `A0_j = sum_i a0_ij` y
`v0_j = (l_j + k_j) / y_j`; la identidad `A0_j + v0_j = 1 - t_j` se cumple celda
a celda en toda tabla de `calibrate_trade_model`. El lema de Shephard da los
requerimientos de insumos por unidad de producto

```
a_ij  = a0_ij (c_j / m_j)^sv (m_j / h_sj)^ss (h_sj / r_ij)^so
b_Lj  = b0_Lj (c_j / v_j)^sv (v_j / w_c)^rho,   b0_Lj = (1 - alpha_j) v0_j
b_Kj  = b0_Kj (c_j / v_j)^sv (v_j / r_c)^rho,   b0_Kj = alpha_j v0_j
(1 - t_j) c_j = sum_i a_ij r_ij + b_Lj w_c + b_Kj r_c
```

y la fila de beneficio cero es `c_j - p_j`. Todo lo demás (ingreso de los
hogares `w L + r K + T`, gasto igual a `theta` por el ingreso menos el ahorro
externo en la inversión, canastas de origen Leontief `afd`, aranceles sobre
valores de productor, el presupuesto público, el cierre de ahorro fijo y la
fila del numerario que sustituye a la primera ecuación de bienes) no cambia, de
modo que con **tecnología plana** (`sigma_va_materials = 0`,
`sigma_sectors = sigma_origins`, `rho_va = 1`) el residuo es igual, fila a
fila, al residuo de la contabilidad consistente con `sigma = sigma_origins`. La
CES plana de la contabilidad consistente usa pesos `a_ij / A_j` sobre los `M`
orígenes; una CES anidada con elasticidades iguales en los niveles de sectores
y de orígenes se reduce exactamente a ese índice. `NestedCESTechnology.is_flat`
informa de este caso y `NestedCESTechnology.from_flexible` traduce una
`FlexibleTechnologyConfig` (`sigma_y`, `sigma_inter`, `rho_va`). Todas las
elasticidades CES son insumos estilizados, no estimaciones.

## La acción exacta del jacobiano

Las incógnitas son `x = [log p (M); log y (M); log r (N); log w (N); T (N); XN (N - 1)]`
y las filas son bienes (la fila 0 es el numerario `p_0 - 1`), precios, trabajo,
capital, saldos `XN - B0` y presupuestos, en el orden y la escala en niveles del
residuo de la contabilidad consistente. Con las participaciones de costo
actuales `alpha_ij = a_ij r_ij / ((1 - t_j) c_j)`,
`beta_Lj = b_Lj w_c / ((1 - t_j) c_j)` y `beta_Kj = b_Kj r_c / ((1 - t_j) c_j)`
(Euler: suman uno), las participaciones de nido `u_ij`, `eta_sj`, `theta_Lj`
construidas a partir de ellas y las direcciones logarítmicas `l` (precios),
`lw`, `lr`:

```
dlog c_j = sum_i alpha_ij l_i + beta_Lj lw_c + beta_Kj lr_c
dlog h_sj = sum_{i in s} u_ij l_i,   dlog m_j = sum_s eta_sj dlog h_sj,   dlog v_j = theta_Lj lw_c + theta_Kj lr_c
da_ij = a_ij [sv dlog c_j + (ss - sv) dlog m_j + (so - ss) dlog h_{s(i)j} - so l_i]
db_Lj = b_Lj [sv dlog c_j + (rho - sv) dlog v_j - rho lw_c]      (db_Kj likewise with lr_c)
d[y_j sum_i (tau_ij - 1) p_i a_ij] = duty_j dlog y_j + y_j sum_i (tau_ij - 1) p_i (da_ij + a_ij l_i)
```

El cierre tiene dos factores, y el numerario sustituye a la primera ecuación de
bienes mientras todos los presupuestos nacionales siguen en el sistema.
`CESBlockJacobian.solve(r)` calcula `J^{-1} r`

1. leyendo directamente el bloque de saldos (`dXN = r_bal`, un bloque identidad);
2. resolviendo el bloque de precios `(diag(p) - diag(c) alpha^T) l = diag(c)(beta_L C lw + beta_K C lr) - r_p`
   una vez para el término forzante y una vez por cada dirección unitaria de
   precio factorial (`L_r`, `L_w`);
3. resolviendo el bloque de bienes `(I - a) diag(y) dlog y = (da) y + afd . dq + rhs`,
   donde la ecuación de bienes omitida lleva un valor desconocido `mu` en `rhs[0]`;
4. ensamblando el complemento de Schur `(3N + 1) x (3N + 1)` de las filas de
   trabajo, capital, presupuesto y numerario en `(log r, log w, T, mu)` empujando
   direcciones unitarias por `response` en bloques de `block_size` columnas,
   factorizándolo por LU y sustituyendo hacia atrás.

`apply(v)` evalúa `J v` con los mismos diferenciales; `as_preconditioner()` y
`as_operator()` envuelven ambos como `scipy.sparse.linalg.LinearOperator`.
`macro_condition` es el número de condición del bloque de Schur escalado por
filas y columnas (`3N + 1` filas en lugar de `4N - 1`: las `N - 1` filas de
saldos forman un bloque identidad y se eliminan primero). `CESBlockJacobian.n`
es el número de celdas `M`; `dim` (igual a `shape[0]`) es el número de
incógnitas `2M + 4N - 1`.

La invertibilidad es una condición, no un teorema. El bloque de precios
`diag(p) - diag(c) alpha^T` es una Z-matriz cuya fila `j` es dominante cuando
`p_j > c_j (1 - beta_Lj - beta_Kj)`, lo que se cumple cerca de cualquier raíz
(donde `p = c`) mientras toda celda pague algún ingreso factorial; allí es una
M-matriz y su inversa es la serie de Neumann convergente. `I - a` es invertible
cuando la matriz de insumos actual tiene radio espectral menor que uno
(Hawkins-Simon). El bloque de Schur se factoriza con pivoteo parcial y se
registra su número de condición. La eliminación es una identidad local
condicional, no un teorema de convergencia ni de equilibrio global.

## Contrato de aceptación y de fallo

`solve_ces_block_newton` es un Newton amortiguado: dirección `-J^{-1} R(x)`,
primer paso `min(1, 1 / max|dx_log|)`, descenso de Armijo del residuo escalado
al cuadrado con factor `1 - 1e-4 step`, como mucho `max_backtracks`
divisiones a la mitad, y un filtro de admisibilidad (precios, producciones,
precios factoriales e ingresos finitos y positivos; cantidades finales
positivas en las categorías activas y exactamente cero en las estructuralmente
ausentes).

El bucle tiene una única prueba de aceptación, la misma que aplican después el
posprocesamiento y `compute_hicksian_welfare`. Solo se detiene cuando (i) cada
fila impuesta del residuo dividida por su escala de referencia (producción,
dotación o ingreso) es como mucho `tol` (por defecto `2e-11`), y (ii) cada fila
en niveles es como mucho `tol * max(scale)`, incluidas las filas que ninguna
ecuación de Newton impone y que solo se cumplen por la ley de Walras: la
ecuación de bienes omitida y los saldos externos realizados. También exige que
ningún gasto final sea negativo. `equilibrium.metadata["tol"]` registra esa cota
absoluta. Cuando (i) se cumple pero (ii) no, el bucle da pasos de Newton de
pulido adicionales; normalmente basta uno. Cuando los flujos son tan grandes en
relación con las escalas de referencia que el redondeo por sí solo supera la
cota, el bucle lanza un error. Por ejemplo, un multiplicador de importación de
1e8 con canastas finales de origen fijas hace que los aranceles devueltos sean
unas 3e7 veces el ingreso de referencia; el error informa entonces de la
auditoría en niveles y de la cota, y el remedio es una `tol` más holgada.

Además, solo se devuelve un estado si `certify_ces_equilibrium` es como mucho
`certificate_tol` (por defecto `1e-8`). El certificado reconstruye los flujos
intermedios y factoriales a partir de la tecnología en `(p, r, w)` con bucles
explícitos por país. Las cantidades finales, el gasto final y los ingresos
arancelarios registrados salen del ensamblaje de la demanda del residuo, y el
certificado los comprueba mediante las claves de valor de bienes, ecuación
omitida, presupuesto, cuenta corriente, aranceles y PIB. Informa de beneficio
cero, adición de costos, valores de bienes, la ecuación de bienes omitida, la
cuenta corriente, los aranceles reconstruidos bilateralmente, el vaciado de los
factores, los presupuestos de hogares y gobierno, la identidad del PIB y el
residuo escalado de la raíz. Cada clave se prueba contra la violación que
nombra: perturbaciones de 1e-5 de cada incógnita, fallos inyectados en el
ensamblaje y una tecnología que rompe la identidad de Euler en 1e-6.

Los límites de iteraciones, los fallos de la búsqueda lineal, los iterados no
admisibles, los bloques macro singulares y los certificados fallidos lanzan
`CESNewtonError` con el `history` de iteraciones y el último
`macro_condition`. Un fallo de la búsqueda lineal dice qué prueba rechazó los
pasos de prueba. Si los rechazó la prueba de Armijo, el mensaje informa del
número de condición, porque un jacobiano casi singular es la causa habitual. Si
todos los pasos salieron del dominio admisible, el mensaje nombra la cantidad,
la categoría y el país que limitan en el paso de prueba más pequeño. Cada
registro del historial cuenta ambos tipos de rechazo (`rejected_armijo`,
`rejected_inadmissible`) y guarda la auditoría en niveles
(`level_audit_before`). Nunca se devuelve un estado no convergido con
`converged=True`.

`continue_tariff_homotopy` mueve linealmente los esquemas intermedio y final
desde un esquema inicial (libre comercio por defecto) hasta el objetivo con una
fracción adaptativa: primer paso `0.25`, crecimiento `1.5` con tope `0.4` tras
una etapa que necesitó como mucho `fast_iterations` pasos de Newton, reducción
a la mitad tras un fallo, aborto por debajo de `min_step`. Cada etapa aceptada
supera las mismas pruebas de residuo, auditoría en niveles y certificado que
una solución directa y se empaqueta al aceptarse (una etapa cuyo empaquetado
falla cuenta como intento fallido). Los intentos fallidos se registran en
`failed_stages`. Todo aborto lanza `CESNewtonError` con `stages`,
`failed_stages` y `last_result` (la última etapa aceptada, o `None` cuando el
propio esquema inicial no se resuelve). `progress` recibe
`{"fraction", "step"}` antes de cada etapa de prueba y los registros del
historial de Newton de cada etapa.

En una tabla sintética aleatoria de 20 por 11 con elasticidad de origen 4, un
Newton directo desde la referencia puede fallar en la búsqueda lineal: la
prueba de Armijo rechaza todos los pasos y el número de condición macro crece
hasta cerca de 1e11. La homotopía alcanza el objetivo en tres etapas; para eso
existe la herramienta. No todo fallo es una singularidad. Con otros esquemas
aleatorios el Newton directo falla porque el filtro de admisibilidad limita: el
iterado queda pegado a una inversión nula en un país, y el error dice que todos
los pasos salieron del dominio admisible. Detrás de un fallo así puede haber una
raíz cuyo gasto de inversión en un país es negativo, porque el ahorro externo
nominal fijo supera la parte de inversión del ingreso. Esa raíz no es admisible
con este cierre, y una homotopía arancelaria no puede cambiarlo.

Los puntos singulares son una propiedad del modelo, no del solucionador. En el
fixture congelado OCDE de tres regiones el determinante del jacobiano de
referencia cambia de signo dos veces entre las elasticidades de origen 0.5 y 1.
El menor valor singular del jacobiano escalado es 4.8e-6 en 0.99, frente a 5e-3
en 0 o 2. Las pruebas lo comprueban con diferencias centradas del residuo de la
contabilidad consistente, con independencia de este módulo. Con un arancel del
10% de una región y elasticidad de origen unitaria el Newton directo falla, y
también fallaron el Newton por diferencias finitas de puremacro y `hybr` cuando
se comprobó. La homotopía certifica el extremo (1,1,1,1), pero la trayectoria
(0,1,1,1) se estanca junto a la referencia. Ambos fallos se lanzan con el
crecimiento del número de condición en `history`. La homotopía no es una
garantía de recuperación. Las comprobaciones discretas en las etapas aceptadas
no certifican una rama continua entre etapas, y un procedimiento fallido no es
un certificado de que no exista equilibrio.

## Bienestar y resultados

`CESBlockNewtonResult` envuelve el `TradeEquilibriumResult` (`equilibrium`), la
tecnología, `iterations`, `calls`, `seconds`, el `residual_max` escalado, el
`certificate`, `macro_condition`, el `history` de Newton y las `stages` y
`failed_stages` de la homotopía; `to_dataframe()` muestra el historial y
`stages_frame()` y `certificate_frame()` los otros dos, con `to_markdown()`,
`to_latex()`, `to_typst()` y `summary()` como en el resto de puremacro.

Con tecnología plana el resultado es un equilibrio de contabilidad consistente
en todos los sentidos y [`compute_hicksian_welfare`](trade_welfare.md) lo
acepta sin cambios (`metadata["hicksian_welfare_supported"]` es verdadero y
`metadata["sigma"]` es `sigma_origins`). Con tecnología anidada
`metadata["hicksian_welfare_supported"]` es falso. El indicador es informativo:
`compute_hicksian_welfare` no lo lee. Su auditoría reevalúa el estado con la CES
plana de la contabilidad consistente, que rechaza (con un `ValueError`) todo
estado anidado cuya raíz difiera de la del modelo plano. Un estado anidado que
también resuelve el modelo plano se acepta. Un ejemplo es la tecnología
(0, .5, .5, .7) en el fixture OCDE de tres regiones, donde las participaciones
de capital uniformes hacen inerte el nido capital/trabajo (`w = r`). La EV que
devuelve sigue siendo correcta, porque la tecnología no cambia las preferencias
de consumo. Por la misma razón, el bienestar hicksiano del consumo para
resultados anidados requiere evaluar las funciones de gasto del hogar a los
precios de comprador del resultado (`Pfd_final`) en lugar de volver a
resolver; véase [preferencias del hogar](trade_household.md).

## Protección de memoria

El conjunto de trabajo se cuenta en arreglos densos `M x M` de doble precisión
(`8 M^2` bytes cada uno; 4.7 MB con `M = 768`). Medido con `tracemalloc`, los
datos del problema retienen unos 3.3 de esos arreglos, los flujos del estado
actual 5.9 y el propio operador 6.2 (dos factores LU, participaciones de costo y
de nido, flujos arancelarios): unos 15 arreglos en total. Una solución de Newton
completa llega a un máximo de unos 22 arreglos, porque la búsqueda lineal evalúa
un estado de prueba mientras el actual sigue vivo: unos 100 MB con `M = 768`,
2.2 GB con `M = 3500` y 2.8 GB con el valor por defecto `max_cells = 4000`.
`max_cells` rechaza tablas mayores antes de reservar memoria, y el mensaje de
error da ambas estimaciones. No se reserva nada al importar, así que el módulo
se puede importar en el entorno de prueba del navegador. Allí el límite de
memoria de WebAssembly (unos pocos GB como mucho) hace de la protección por
defecto una cota superior más que un tamaño de trabajo.

## Qué está validado y qué no

`tests/test_trade_ces_newton.py` comprueba, en una tabla no negativa de 3 por 2
construida a mano con una categoría ausente de compras en el exterior y
contraaranceles bilaterales, en su variante con X activa, en el fixture
congelado OCDE 2019 de tres regiones y en una tabla sintética con semilla de
20 por 11 (64 por 12 marcada `slow`):

- el oráculo de inversa exacta en un punto aleatorio fuera del equilibrio: el
  jacobiano por paso complejo del residuo del módulo cumple `J solve(r) = r` y
  `apply(solve(r)) = r` a `3e-10` en unidades escaladas para las tecnologías
  (0,0,0,1), (.1,.3,1.5,1), (1,1,1,1), (.5,1,4,.7), (.1,.1,.5,.7) y tres planas;
- las diferencias centradas del residuo de la contabilidad consistente con
  tecnología plana invierten el operador a `1e-6` y coinciden con el jacobiano
  por paso complejo;
- paridad fila a fila del residuo con el de la contabilidad consistente para
  `sigma` en {0, .5, 1, 2} a `1e-12` relativo, el límite de Leontief frente a
  `solve_trade_equilibrium` a `1e-10`, equilibrios CES planos y EV hicksiana
  frente a los solucionadores por diferencias finitas, y los flujos
  posprocesados frente al posprocesamiento de la contabilidad consistente;
- reproducción de la referencia (`a = a0`, `b = b0`, residuo por debajo de
  `1e-12`) para toda tecnología, la identidad de Euler, las derivadas de
  Shephard por diferencias centradas, participaciones que suman uno;
- certificado como mucho `1e-8` y el libro contable como mucho `1e-10` veces la
  mayor escala de referencia (producción, dotación o ingreso) en toda solución
  aceptada, toda fila en niveles dentro de `metadata["tol"]`, homogeneidad
  nominal, invariancia a la unidad monetaria de precios y porcentajes de
  bienestar, e invariancia al orden de los países dado el numerario;
- una única prueba de aceptación: un barrido de soluciones en las tablas hechas
  a mano, algunas de las cuales necesitan un paso de pulido, solo devuelve
  estados que superan la auditoría del posprocesamiento;
- sensibilidad del certificado: cada clave salta (por encima de `1e-7`) ante la
  violación que nombra, y se rechaza una tecnología que viola la identidad de
  Euler en `1e-6`;
- mensajes de fallo que nombran su causa (dirección del límite de
  admisibilidad, fallo de Armijo, piso de redondeo con multiplicadores
  prohibitivos), abortos de la homotopía que llevan su registro, y la casi
  singularidad OCDE por diferencias centradas del residuo de la contabilidad
  consistente;
- el contrato de fallo, la certificación de las etapas de la homotopía, el
  registro de fallos forzados, la reanudación desde un esquema parcial, la
  independencia del tamaño de bloque a `1e-12`, una raíz `hybr` densa
  independiente del mismo residuo y un dorado OCDE autogenerado.

No validado: la convergencia para cualquier perfil de elasticidades fuera de los
fixtures anteriores, el bienestar con tecnología anidada, la ejecución en GPU,
las preferencias LES, los márgenes, las penalizaciones de capacidad y los
cierres fiscales distintos de la suma fija (los conserva el módulo flexible).
Los umbrales del residuo y del certificado son comprobaciones en coma flotante,
no pruebas por intervalos.

Relacionado: [contabilidad de comercio consistente](trade_accounting.md),
[bienestar hicksiano del consumo](trade_welfare.md),
[política arancelaria hicksiana y recuperación del solucionador](trade_policy.md),
[continuación arancelaria](trade_continuation.md),
[estado de la validación estructural](STRUCTURAL_VALIDATION_STATUS.md).
