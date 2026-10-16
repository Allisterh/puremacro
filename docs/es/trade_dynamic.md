> 🇬🇧 [English](../trade_dynamic.md) · 🇪🇸 Español

# MRIO dinámico con capital específico por sector

`puremacro.trade.dynamic` es una extensión dinámica, determinista y con
previsión perfecta del modelo multipaís de insumo-producto con coeficientes
fijos, en la que cada celda país-industria de origen mantiene su propio stock
de capital instalado con costos de ajuste a la Tobin-q. Los resultados son
objetos congelados con `to_dataframe()`, `to_markdown()`, `to_latex()` y
`to_typst()`, los constructores validan su entrada, los errores son
estructurados y ninguna solución devuelve `converged=True` sin superar en cada
fecha un certificado contable independiente.

**No** es el CGE estático de `solve_trade_equilibrium`: no hay sustitución de
Armington, el trabajo es inelástico y móvil dentro de cada país, los países
están en autarquía financiera y el consumo del gobierno está dentro de la
canasta del hogar. Su medida de bienestar es un equivalente de consumo CRRA, no
la EV hicksiana de [`compute_hicksian_welfare`](trade_welfare.md). Lea las
limitaciones del final antes de interpretar cualquier número.

## Inicio rápido sobre una tabla sintética

```python
from puremacro.trade import calibrate_trade_model
from puremacro.trade.data import synthetic_codes, synthetic_icio_table
from puremacro.trade.dynamic import (
    DynamicAccounts, DynamicTariff, DynamicEconomy, calibrate_dynamic,
    solve_dynamic_steady_state, solve_dynamic_transition, tariff_path,
    stability_report,
)

# A seeded synthetic table: twelve countries, five sectors (S01 and S02 are merchandise).
countries, sectors = synthetic_codes(12, 5)
calib = calibrate_trade_model(synthetic_icio_table(12, 5, seed=2), country_codes=countries, sector_codes=sectors)
accounts = DynamicAccounts.from_trade_calibration(calib, merchandise_sectors=("S01", "S02"))
calibration = calibrate_dynamic(accounts, beta=.96, delta=.08)
print(calibration.summary())

economy = DynamicEconomy(calibration, adjustment_cost=2., risk_aversion=2.)
baseline = solve_dynamic_steady_state(economy)                     # the benchmark
tariff = DynamicTariff.uniform(calibration, "C01", .05, sectors="merchandise")
terminal = solve_dynamic_steady_state(economy, tariff)             # exact 2C condensation
print(terminal.to_dataframe().loc[["C01", "C02"]])

report = stability_report(economy, baseline)                       # Blanchard-Kahn count
print(report.summary())

run = solve_dynamic_transition(economy, tariff_path(economy.zero_policy(), tariff, horizon=40),
                               terminal=terminal)
print(run.summary())
print(run.welfare.to_dataframe().loc[["C01", "C02"]])
```

Los flujos sintéticos son extracciones aleatorias genéricas, así que los
números son salidas del software, no estimaciones. La calibración no necesita
ajustes (60 celdas activas, sin reparaciones factoriales ni de inversión), la
referencia es determinada (60 raíces estables para 60 stocks predeterminados)
y un arancel permanente del 5% de `C01` sobre todas sus importaciones de
mercancías se resuelve a 40 fechas en tres iteraciones de Newton con un
certificado cercano a `2e-11`. La trayectoria de 40 fechas es un horizonte
resuelto, no aceptado: use `run_horizon_ladder` (sección siguiente) antes de
leer sus cifras de bienestar.

## Un ejemplo pequeño que reproduce la batería de pruebas

Las cuentas analíticas de dos países y dos sectores son el fixture que usan
las pruebas de este repositorio. Son un fixture de software, nunca evidencia
empírica.

```python
import numpy as np
from puremacro.trade.dynamic import (
    analytic_two_country_accounts, calibrate_dynamic, DynamicEconomy,
    DynamicTariff, tariff_path, solve_dynamic_steady_state, run_horizon_ladder,
)

accounts = analytic_two_country_accounts()
calibration = calibrate_dynamic(accounts)
economy = DynamicEconomy(calibration)
zero = economy.zero_policy()
rates = np.zeros((calibration.n_cells, calibration.n_countries))
rates[calibration.country == 0, 1] = .06     # BBB taxes AAA's goods at 6%
rates[calibration.country == 1, 0] = .08     # AAA taxes BBB's goods at 8%
shock = DynamicTariff.build(rates, consumption_rates=rates * .7,
                            investment_rates=rates * 1.3, label="analytic tariff")
terminal = solve_dynamic_steady_state(economy, shock, tol=1e-11)

# Announced at date 0, implemented at date 2, permanent.
ladder = run_horizon_ladder(
    economy, lambda T: tariff_path(zero, shock, horizon=T, announcement=2),
    horizons=(80, 160, 320), terminal=terminal, tol=1e-10,
)
print(ladder.to_dataframe())
accepted = ladder.accepted            # the 320-date solution (status "verified")
print(accepted.welfare.summary())      # AAA -0.24%, BBB +0.11% of benchmark consumption
```

En este fixture la comparación de 80 a 160 coincide en las primeras 20 fechas
(3.6e-7 en logaritmos) pero el bienestar todavía se mueve 0.0055 puntos
porcentuales, así que se rechaza; la de 160 a 320 supera la ventana (2.7e-12),
el bienestar (4.3e-5 pp) y la distancia al estado terminal (4.3e-5). La raíz
estable más lenta ronda 0.98, por lo que hacen falta cientos de fechas.

## El modelo

Celda `j` en el país `c(j)`, fecha `t`, precios de referencia iguales a uno.
El valor agregado es Cobb-Douglas normalizado a la referencia en capital
específico del sector y trabajo nacional; los insumos intermedios y las
canastas finales tienen coeficientes fijos:

```
v[j,t] = w[c(j),t]^(1-alpha[j]) (R[j,t]/R0[j])^alpha[j]
(1-tax[j]) p[j,t] = sum_i A[i,j] (1+tau[i,c(j),t]) p[i,t] + b[j] v[j,t]
L[j,t] = (1-alpha[j]) b[j] v[j,t] y[j,t] / w[c(j),t]
K[j,t] = alpha[j] b[j] v[j,t] y[j,t] / R[j,t]
y = A y + omegaC C + omegaI I_national
PC[c] = sum_i (1+tauC[i,c]) omegaCtax[i,c] p[i] / (1-tC[c])   (exempt X part added)
PI[c] = sum_i (1+tauI[i,c]) omegaI[i,c] p[i] / (1-tI[c])
```

Capital, instalación y ecuación de Euler:

```
K[j,t+1] = (1-delta) K[j,t] + K[j,t] Phi(x[j,t]),   x = I/K
Phi(x) = x - phi/2 (x-delta)^2
q[j,t] = PI[c(j),t] / Phi'(x[j,t])
q[j,t] = m[c,t+1] (R[j,t+1] + q[j,t+1] G[j,t+1]),  G = 1-delta+Phi(x)-x Phi'(x)
m[c,t+1] = beta (C[c,t+1]/C[c,t])^(-sigma) PC[c,t]/PC[c,t+1]
```

El presupuesto del hogar paga exactamente el consumo y la inversión que entrega
el vaciado de bienes:

```
PC C + PI sum_{j in c} I[j] + XN0[c] w_last
    = w L0[c] + sum_{j in c} R K + product and production tax receipts + tariff receipts
```

No hay bono negociado. Las exportaciones netas de referencia `XN0` son
transferencias mundiales balanceadas fijas en unidades del salario del último
país (el numerario). Un presupuesto redundante (el del país con mayor escala
presupuestaria) se sustituye por la ecuación del numerario y se comprueba
después junto con todas las demás cuentas. Como el último país del registro es
a la vez el numerario salarial y el ancla de las transferencias, el orden del
registro de países tiene contenido económico: reordenar los países cambia el
cierre y por tanto las asignaciones reales cuando se mueven los salarios
relativos (aproximadamente en el cambio del salario relativo del ancla por
`XN0`). Reordenar los sectores es un mero cambio de etiquetas.

**Eliminación.** Las incógnitas retenidas por fecha son
`z_t = [log w (C), log C/C0 (C), log K_{t+1}/K0 (N)]`, es decir `N + 2C` por
fecha. La instalación se invierte en su rama creciente,
`x = delta + 2g/(1 + sqrt(1 - 2 phi g))` con `g = K'/K - 1`; el vaciado de
bienes usa respuestas de Leontief guardadas `YC = (I-A)^-1 omegaC`,
`YI = (I-A)^-1 omegaI`; los precios factoriales salen de las condiciones
Cobb-Douglas; los precios de productor resuelven un sistema lineal ajustado por
aranceles para cada política distinta (LU dispersa hasta 600 celdas; por encima
de 600 celdas, LU densa cuando la matriz de coeficientes tiene densidad mayor
que .25 y en otro caso BiCGSTAB precondicionado con respaldo GMRES). El residuo
retenido son `C` ecuaciones de vaciado del trabajo, `C` presupuestos con un
numerario y `N` ecuaciones de Euler. `DynamicEconomy.state_jvp`/`states_jvp`
dan productos jacobiano-vector analíticos exactos, por lotes para cada
política.

**Certificado.** Cada fecha aceptada supera 17 comprobaciones reconstruidas de
forma independiente (bienes, beneficio cero, trabajo, acumulación de capital,
pagos factoriales, demanda de capital y de trabajo, tecnología, condición de
primer orden de la inversión, Euler, recaudación arancelaria, presupuesto
público, todos los presupuestos incluido el omitido, cuenta corriente, balance
mundial de transferencias, valor de los bienes de inversión) a
`max(1e-8, tol)`.

**Estado estacionario.** `solve_dynamic_steady_state(method="condensed")`
resuelve una condensación exacta de `2C` en salarios nacionales y precios de
la inversión con un jacobiano analítico denso (por defecto por encima de 240
incógnitas); `"full"` resuelve las `N + 2C` ecuaciones estacionarias con un
Newton sin matriz. Ambos reevalúan el residuo completo y el certificado. Ambos
son métodos de Newton locales sin respaldo de continuación, y `method="auto"`
no reintenta con el otro método ante un fallo: un arancel grande puede
necesitar `start` desde el estado estacionario de un arancel menor, y un
arancel prohibitivo puede no tener estado estacionario interior (en el fixture
analítico un arancel uniforme del 100% lanza `DynamicSolveError`). Un `start`
no factible o una política cuyo sistema de precios ajustado por aranceles no
tiene solución positiva lanza `EconomicDomainError` (un `ValueError`) antes de
cualquier paso de Newton.

**Transición.** `solve_dynamic_transition` apila `T (N + 2C)` incógnitas con
capital inicial fijo y un estado estacionario terminal resuelto aparte bajo la
última política (mantenida más allá del horizonte), y ejecuta un Newton
inexacto sin matriz: `lgmres` (con respaldo `gmres`) precondicionado por una
tridiagonal aproximada capital-tiempo más bloques nacionales
(`preconditioner="capital"`), término forzante
`max(1e-5, min(.1, sqrt(||F||)))`, un tope de `.75` en los movimientos en
logaritmos y divisiones de Armijo a la mitad. Un estancamiento de la búsqueda
lineal lanza `DynamicSolveError` nombrando la ecuación, la fecha y la celda, la
menor tasa de instalación del último iterado aceptado (incluida la fecha límite
terminal) y lo que el paso de Newton completo le haría. La conjetura inicial
por defecto es una mezcla exponencial del estado inicial al terminal; cuando un
choque grande haría desinvertir a esa conjetura (tasa de instalación `x <= 0`)
el solucionador usa una mezcla lineal, y cuando ni siquiera la mezcla lineal es
factible ninguna trayectoria de ese horizonte puede mantener la inversión
interior, lo que se informa como un `DynamicSolveError` que nombra la celda
(`metadata["start"]` registra la conjetura usada). Los fallos de la búsqueda
lineal del estado estacionario nombran la ecuación con mayor residuo en
`location`. Por defecto la transición ejecuta primero `stability_report` sobre
el estado terminal y rechaza un cierre indeterminado
(`DynamicDeterminacyError`); `require_determinacy=False` resuelve de todos
modos la pila finita.

## Libro de la calibración estacionaria

Un año de insumo-producto no es un estado estacionario. `calibrate_dynamic`
deriva `R0 = PI0 (1/beta - 1 + delta)`, `K0 = alpha VA0 / R0`, la inversión de
reposición `delta K0`, reasigna el flujo de inversión estacionario dentro de la
absorción observada (recompone la canasta solo donde la composición observada
no puede financiar la reposición, con `investment_policy="reallocate"`), agrupa
la variación de existencias, los objetos valiosos y las compras exentas en el
exterior en la canasta de consumo, y concilia los impuestos sobre el uso final.
`factor_policy="reclassify_losses"` convierte los componentes factoriales
negativos en subsidios a la producción y recorta las participaciones a
`capital_share_bounds`. Todo queda registrado en `calibration.report` y en
`calibration.to_dataframe("countries" | "factors" | "investment")`.

Puentes: `DynamicAccounts.from_icio` acepta `ICIOData` (esquema procesado
C/I/Cx: una columna I que contiene variación de existencias puede tener celdas
negativas, y entonces `negative_investment` debe ser `"raise"` o
`"to_inventory"`; las participaciones del capital salen de las filas de trabajo
y capital de la tabla), `RawIOData` (las categorías de los cargadores como
HFCE/NPISH/GGFC/GFCF/INVNT/DPABR se asignan al contrato
`("C","G","X","V","VAL")`; sin detalle factorial se supone y registra la
separación 2/3-1/3) y cualquier objeto tabla de insumo-producto con `Z`, `F`,
`fd_codes`, `VA`, `TLS`, `TFD`, `output`, `country_codes` y `sector_codes`. La
columna `EXPORT` del cargador de EXIOBASE (exportaciones a regiones fuera de la
tabla) no tiene asignación por defecto porque no son compras de residentes en
el exterior; pase `fd_map={"EXPORT": "X"}` u otro destino explícitamente.
`from_arrays` acepta `F` como `(M, N, K)` o `(M, N*K)` y `TFD` como `(N, K)` o
plano `(N*K,)`; una tabla de impuestos transpuesta `(K, N)` se rechaza cuando
`N != K` (con tantos países como códigos de uso final no se puede detectar la
orientación, así que pase filas de destino y columnas de categoría). El valor
por defecto `merchandise_sectors="isic_a_c"` marca como bienes los 22 códigos
de bienes de las 45 industrias de la OCDE y los códigos que son la letra sola
A, B o C o empiezan por una de ellas seguida de un dígito, salvo el C33 de
FIGARO (reparación e instalación); un objeto tabla que lleva su propia
`merchandise_mask` la conserva, y otros registros, como los códigos `i01...` de
EXIOBASE o los códigos sintéticos `S01...`, necesitan una lista explícita de
sectores. La regla y la lista de bienes resultante se registran en
`metadata["merchandise_sectors"]`. `from_trade_calibration` lee
`calib.data_calibra`; su columna Cx pasa a ser X (compras de residentes en el
exterior) solo para una tabla sin asignación de usos finales registrada y para
la asignación `DPABR` de la OCDE, el Cx de los cargadores FIGARO, WIOD y Eora
(existencias y objetos valiosos con signo) pasa a ser V, y el Cx de EXIOBASE,
que incluye su columna de exportaciones, necesita un `cx_category` explícito.

## Políticas

`DynamicTariff` guarda tres tablas de tasas ad valorem `N x C` (entregas
intermedias, de consumo y de inversión) por celda de origen y país de destino;
las entregas domésticas y las compras de residentes en el exterior nunca pagan.
Los aranceles específicos por sector de destino no son representables. Las
tasas deben ser mayores que `-1`: cero es libre comercio y las tasas negativas
son subsidios a la importación.
`DynamicTariff.uniform(cal, "C01", .1, sectors="merchandise", exporters=("C02",), uses=("intermediate",))`,
`from_rates` y `build` validan las tablas. La huella (sha256 de las tres
tablas) siempre se deriva de las tablas guardadas, nunca se pasa, así que
`dataclasses.replace(policy, rates=...)` produce una huella nueva; las
políticas idénticas comparten una factorización de la red de precios, y una
política cuyas tablas ya no coinciden con su huella se rechaza. Una selección
sin ninguna celda de origen extranjera (por ejemplo `exporters=("C01",)` para el
importador `"C01"`) lanza `ValueError` en lugar de devolver una política nula
con etiqueta de arancel. `tariff_path(baseline, shock, horizon=T, announcement=2, duration=5)`
construye esquemas anticipados, permanentes o temporales.

## Aceptación del horizonte

Una pila finita es una aproximación a una trayectoria de horizonte infinito.
`compare_horizons(short, long)` exige: (1) que los primeros `periods=20` estados
retenidos coincidan dentro de `1e-5` en logaritmos; (2) que el bienestar en
equivalente de consumo coincida dentro de `1e-4` puntos porcentuales; (3) una
comprobación terminal sobre la solución larga. El valor por defecto
`terminal_check="state_gap"` exige `max |z_T - z*| <= 1e-4` y da el estado
`verified`. El modo opcional `"discounted_wealth"` exige que el valor del
capital terminal descontado

```
D[c] = beta^(T-1) (C[c,T-1]/C0[c])^(-sigma) / PC[c,T-1] * sum_{j in c} q[j,T-1] K[j,T] / C0[c]
```

sea como mucho `1e-6` y da el estado explícitamente más débil
`verified_window_and_welfare`: no certifica ni la convergencia completa al
estado terminal ni una cota rigurosa de la cola de utilidad. Ninguna de estas
tolerancias es una cota de error frente a la solución desconocida de horizonte
infinito. Ambas soluciones deben venir de la misma economía (el resultado de la
transición registra una huella de los arreglos de calibración, el costo de
ajuste, la aversión al riesgo y el numerario salarial en
`metadata["economy_fingerprint"]`), partir del mismo estado inicial, terminar
en el mismo estado terminal y seguir la misma política anunciada.
`run_horizon_ladder` resuelve horizontes crecientes, arrancando cada uno en
caliente desde la trayectoria anterior, y se detiene en la primera comparación
aceptada.

## Bienestar

`consumption_equivalent_welfare` devuelve, por país, el cambio proporcional
permanente en el consumo de referencia `e` con
`sum_t beta^t u(e) = sum_t beta^t u(C_t/C0)` para una `u` CRRA (la misma
`risk_aversion` que la ecuación de Euler; logarítmica en uno), la cola
estacionaria ponderada por `beta^T/(1-beta)`, el equivalente de gasto anual a
precios de referencia y su porcentaje del PIB de referencia. No hay agregación
entre países ni descomposición por canales.

## Diagnóstico de determinación

`stability_report(economy, steady)` linealiza las ecuaciones de cada fecha con
los JVP propios del modelo, forma el haz
`[[A_plus,0],[0,I]] s_{t+1} = [[-A_0,-A_minus_K],[E_K,0]] s_t` con
`s_t = (dz_t, dK_t)` y cuenta los valores propios generalizados: la
determinación exige exactamente `N` raíces estables para los `N` stocks
predeterminados. El resultado informa de la raíz estable más lenta, la raíz
inestable más pequeña y las participaciones por país de su vector propio. El
punto debe ser estacionario bajo la política pasada (o la que lleve un
resultado de estado estacionario): el informe rechaza, con un `ValueError`, un
estado cuyo residuo `F(z, z, z)` supera `stationarity_tol` (por defecto
`1e-8`). El problema denso tiene orden `n_vars + N` y su costo crece con el cubo
de ese orden (segundos con unos cientos de celdas, decenas de segundos cerca de
mil). `solve_dynamic_transition` guarda el informe en la economía por huella de
la política terminal, de modo que una escalera de horizontes lo paga una vez.

## Qué está validado aquí y qué no

Verificado por pruebas de este repositorio (`tests/test_trade_dynamic.py`), en
el fixture analítico, el fixture congelado OCDE de tres regiones y tablas
sintéticas con semilla:

* Estacionariedad de la referencia, identidades factoriales y tributarias,
  cuentas independientes con bucles escalares en estados estacionarios y a lo
  largo de transiciones, JVP analíticos frente a diferencias centradas (2e-6),
  la condensación exacta frente a la solución completa, el jacobiano condensado
  frente a diferencias finitas, la evaluación por lotes, el momento de la
  anticipación, las políticas temporales, las guardas de la comparación de
  horizontes, el diagnóstico del extremo descontado y las identidades del
  bienestar en equivalente de consumo.
* `puremacro.dsge.solve_perfect_foresight` sobre las mismas ecuaciones a 40
  fechas como oráculo independiente de Newton apilado; los backends iterativos
  de precios y bienes coinciden con la LU dispersa a 1e-10 en una tabla
  dispersa con semilla.
* Guardas: una política reconstruida con `dataclasses.replace` se resuelve como
  política nueva, las comparaciones de horizontes rechazan soluciones de
  economías distintas, `stability_report` rechaza puntos no estacionarios y la
  barrera de determinación lanza un error estructurado en un fixture
  indeterminado.
* Tablas sintéticas: estados estacionarios con y sin un arancel del 10% en una
  tabla de 5 por 4 y (lenta) una tabla determinada de 12 por 5 con una
  transición certificada de 40 fechas y una escalera de horizontes.

No validado aquí:

* Parámetros. Una sola tabla de insumo-producto no identifica los stocks de
  capital históricos, la depreciación, los costos de ajuste ni las preferencias
  intertemporales. Los ejemplos usan `beta=.96`, `delta=.08`, costo de ajuste 2
  y aversión al riesgo CRRA 2; la aceptación numérica no valida esas elecciones
  de parámetros económicos.
* Inversión. Es un modelo de inversión interior; no resuelve un problema de
  complementariedad con irreversibilidad de la inversión. El choque admisible
  puede depender del horizonte: cuando un horizonte largo lleva la instalación
  de alguna celda hacia cero en la fecha límite terminal, la búsqueda lineal se
  estanca y el error nombra la celda. Un fallo de la búsqueda lineal es
  evidencia, no prueba, de que no existe una trayectoria interior.
* Alcance. Abastecimiento fijo, valor agregado Cobb-Douglas, trabajo nacional
  fijo y autarquía financiera con transferencias balanceadas; no hay
  elasticidades de Armington ni de oferta de trabajo. La variación registrada
  de existencias y los objetos valiosos no identifican stocks de existencias, y
  el consumo incluye la demanda final del gobierno y de las instituciones sin
  fines de lucro en la canasta del hogar representativo.
* Horizontes. Los horizontes largos son intrínsecos (raíces estables más lentas
  cerca de 0.98 en los fixtures), y una escalera aceptada en una tabla no dice
  nada de otra.
* Determinación. La indeterminación local es una propiedad del cierre
  linealizado en una tabla dada, no un defecto del solucionador, y las tablas
  grandes pueden ser indeterminadas. Ejecute `stability_report` antes de
  confiar en `require_determinacy=False`. El criterio `state_gap` puede
  rechazar una escalera que `discounted_wealth` aceptaría; trate los
  resultados de `discounted_wealth` solo como afirmaciones sobre la ventana
  inicial.

## Relacionado

* [Contabilidad de comercio consistente](trade_accounting.md),
  [bienestar hicksiano del consumo](trade_welfare.md) y
  [política arancelaria](trade_policy.md) para el modelo estático que esta
  extensión no sustituye.
* `puremacro.dsge.solve_perfect_foresight` para el Newton apilado genérico que
  se usa como oráculo independiente.
* [Estado de la validación estructural](STRUCTURAL_VALIDATION_STATUS.md).
