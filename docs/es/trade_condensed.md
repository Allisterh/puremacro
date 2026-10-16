> 🇬🇧 [English](../trade_condensed.md) · 🇪🇸 Español

# Modelo arancelario de Leontief condensado con un factor

`puremacro.trade.condensed` resuelve un modelo arancelario estático multipaís
y multisectorial con coeficientes de insumo fijos, un factor compuesto por país
y aranceles de importación devueltos de suma fija; elimina precios y
producciones de forma exacta, certifica cada equilibrio informado a partir de
los flujos brutos de la tabla y solo informa de medidas que no dependen del
numerario. Es un modelo distinto de `solve_trade_equilibrium` (contabilidad
consistente); la tabla comparativa de abajo dice en qué se diferencian.

## Ejemplo ejecutable sobre una tabla sintética

```python
import numpy as np
from puremacro.trade import calibrate_trade_model
from puremacro.trade.data import synthetic_codes, synthetic_icio_table
from puremacro.trade.condensed import (
    BalancedIOTable, build_tariff_wedges, calibrate_condensed, compute_measures, solve_condensed,
)

# A seeded synthetic table: five countries, four sectors (the first two are merchandise).
countries, sectors = synthetic_codes(5, 4)
trade_calib = calibrate_trade_model(synthetic_icio_table(5, 4, seed=0),
                                    country_codes=countries, sector_codes=sectors)
table = BalancedIOTable.from_trade_calibration(trade_calib, merchandise_mask=[True, True, False, False])
calib = calibrate_condensed(table)

# C01 raises a 15% duty on merchandise from C02 and C03, and 5% on everyone else.
wedges = build_tariff_wedges(calib, {"C01": {"*": 0.05, "C02": 0.15, "C03": 0.15}})
print(wedges.to_dataframe().head())
result = solve_condensed(calib, wedges)                      # gate, Newton, certificate
print(result.summary())
measures = compute_measures(calib, wedges, result.state, country="C01", partners=("C02", "C03"))
print(measures.to_markdown())
```

Los códigos sectoriales sintéticos `S01`..`S04` no pertenecen a ningún registro
conocido, así que el puente recibe una `merchandise_mask` explícita; los
aranceles recaen entonces solo sobre los dos sectores de mercancías. La
ejecución da seis iteraciones de Newton y devuelve un certificado con bloque
máximo cercano a `3e-12` (por debajo de la barrera de `1e-10`). Para `C01` la
tasa efectiva sobre las importaciones gravables queda entre las tasas del 5% y
del 15% (cerca del 11.7%), la recaudación arancelaria ronda el 1.8% del PIB y
la variación equivalente se descompone exactamente en ingreso factorial real,
recaudación neta real y la transferencia de déficit e inventarios. Las cifras
salen de flujos aleatorios genéricos, no de ninguna economía. Hay ejemplos de
tres regiones y otros sintéticos en `tests/test_trade_condensed.py`.

## Qué calcula

Índices: países `k`, celdas `i = country * S + sector` (M = N S), categorías
finales `f` en {C, G, X} con participaciones Cobb-Douglas e inventarios
exógenos V. La referencia tiene todos los precios, salarios y multiplicadores
iguales a uno.

Calibración a partir de una tabla balanceada (`table_from_arrays` y luego
`calibrate_condensed`):

```
a_ij = Z_ij / y0_j     b_j = VA_j / y0_j     t_j = TLS_j / y0_j
omega^f_ik = F^f_ik / B^f_k        A0^f_k = B^f_k + TFD^f_k       t^f_k = TFD^f_k / A0^f_k
theta^f_k = A0^f_k / sum_f A0^f_k  qV_ik = F^V_ik                 TV_k = TFD^V_k
Fbar_k = sum_{j in k} VA_j         Y0_k = Fbar_k + sum_{j in k} TLS_j + sum_f TFD^f_k + TV_k
XN0_k = fob exports - imports      s_k = XN0_k / sum_l Fbar_l
```

Equilibrio en niveles (2M + 3N incógnitas, una ecuación redundante por la ley
de Walras):

```
(E1) (1 - t_j) p_j = b_j w_c(j) + sum_i a_ij tau_ij p_i                       zero profit
(E2) y_i = sum_j a_ij y_j + sum_k (sum_f omega^f_ik q^f_k + qV_ik)             goods clearing
(E3) sum_{j in k} b_j y_j = Fbar_k                                              factor market
(E4) Y_k = w_k Fbar_k + PT_k + TR_k                                             income
(E5) XN_k = s_k sum_l w_l Fbar_l                                                closure
(E6) sum_l w_l Fbar_l = sum_l Fbar_l                                            numeraire
P^f_k = sum_i omega^f_ik tau^f_ik p_i     E^f_k = theta^f_k (Y_k - XN_k - E^V_k)
q^f_k = (1 - t^f_k) E^f_k / P^f_k         E^V_k = sum_i tau^V_ik p_i qV_ik + TV_k P^G_k
```

`PT_k` reúne los impuestos sobre productos en la producción y en las compras
finales; `TR_k` reúne todos los aranceles (`(tau - 1) x valor de productor`)
sobre compras intermedias, finales y de inventarios, y se devuelve mediante
(E4). El cierre del saldo comercial (E5) mantiene el saldo de cada país como
una proporción fija del ingreso factorial mundial (Dekle, Eaton y Kortum, 2008).

Condensación: dado `z = (log w, Y / Y0)`, (E1) se resuelve exactamente con una
LU de `I - B_tau^T` con `B_tau,ij = a_ij tau_ij / (1 - t_j)`, y (E2) con una LU
de `I - a`. Los residuos son (E3) y (E4) escalados por cantidades base, con la
ecuación de ingreso del país de referencia sustituida por el numerario; esa
ecuación eliminada es la comprobación de Walras que se informa con cada estado.
El jacobiano usa diferencias hacia adelante (`h = 1e-7`) con los precios
guardados entre las columnas de ingreso.

Solucionadores (`solve_condensed`): Newton amortiguado con retroceso de Armijo,
un tope de `0.3` en los desplazamientos de log-salarios y la regla de parada
`max |r| <= 1e-12`, `|Walras| <= 1e-10`; continuación en el parámetro natural
de escala arancelaria (división a la mitad ante un fallo hasta `1/256`,
predictor secante); un respaldo anidado que resuelve el sistema afín de
ingresos dados los salarios; continuación opcional por pseudolongitud de arco
con monitores de pliegue (`dlambda/ds`, `sign det J`, `sigma_min / sigma_max`)
y una comprobación de llegada a `1e-10`; arranques múltiples cercanos
opcionales que lanzan `MultipleEquilibria` cuando un arranque convergido cae a
más de `1e-9`. Una barrera de existencia de Collatz-Wielandt
`rho(B_tau) < 1 - 1e-3` (`puremacro.trade.regularize.compute_spectral_radius`)
se ejecuta antes de cada solución y falla de forma cerrada: cotas que
certifican `rho < 1` pero no el margen lanzan `ProductivityUncertified`
("margin not met", con las cotas adjuntas), cotas que no certifican nada lo
lanzan como "inconclusive", y ninguna afirma la no existencia; las cuñas
construidas para otra tabla se rechazan con `CalibrationError`. Cuando la
continuación se estanca, la etiqueta de diagnóstico del respaldo anidado y su
registro de etapas (`continuation_lam_reached`) nombran el estancamiento.

Certificado (`certify_raw_flows`): diez bloques recalculados a partir de
`Z, F, VA, TLS, TFD` y las cuñas, nunca de `a, b, omega, theta` ni de las
factorizaciones: beneficio cero en las celdas activas, vaciado de bienes,
mercados de factores, presupuesto de los hogares, ingreso nacional, identidad
de la recaudación arancelaria, cierre, identidad de la cuenta corriente (`NX` a
partir de valores de productor bilaterales), adición mundial y el numerario.
Cada bloque debe ser como mucho `1e-10`; `solve_condensed` lanza
`CertificationFailure` en otro caso y nunca devuelve `converged=True` tras un
fallo. El certificado usa los precios y las cantidades de las canastas del
estado, de modo que es independiente de los coeficientes calibrados pero no del
sistema de demanda que los produjo. Los dos bloques por celda (beneficio cero,
vaciado de bienes) se escalan por `max(p_j y_j, cell_floor)` y
`max(y_i, cell_floor)` con `cell_floor = 1` en las unidades propias de la
tabla, así que las celdas por debajo de una unidad se juzgan con `1e-10`
absoluto; pase un `cell_floor` menor a `certify_raw_flows` para tablas en
unidades grandes. Las celdas fantasma (celdas vacías que la construcción llena
con flujos de `1e-6`) quedan fuera solo del bloque de beneficio cero; todos los
demás bloques usan su razón de producción verdadera `y_j / y0_j`, porque el
modelo deja moverse la producción fantasma y el valor agregado fantasma forma
parte de `Fbar`. `solve_condensed(..., certify=False)` omite el certificado
solo para diagnóstico: ese resultado está `converged` pero no `passed` ni
`certified`.

Medidas (`compute_measures`, todas independientes del numerario,
`P^U_k = prod_f (P^f_k / P0^f_k)^theta^f_k`):

| Medida | Definición |
|---|---|
| precios al consumidor relativos a los salarios | `P^C_k / (P0^C_k w_k) - 1`, el inverso del salario real de consumo (no un IPC) |
| variación equivalente | `Atil^1_k / P^U_k - Atil^0_k`, en porcentaje del PIB base, con la descomposición exacta en ingreso factorial real, recaudación neta real (aranceles y otros impuestos) y la transferencia de déficit e inventarios |
| ingreso nacional real | `(Y_k / P^U_k) / Y0_k - 1` |
| términos de intercambio de Fisher | fob, sin aranceles, `sqrt(L^X P^X / (L^M P^M)) - 1` a cantidades base |
| recaudación arancelaria | `TR_k / Y_k` y la tasa efectiva sobre importaciones gravables (los inventarios solo cuentan como gravables cuando el estado se resolvió con `inventories="tariffed"`) |
| PIB real a precios constantes | `sum (VA_j + TLS_j) yhat_j + sum_f TFD^f qhat^f + TFD^V`, solo un efecto de composición |
| cota de costo unitario de EP | `p^PE = (I - B_tau^T)^{-1} b / (1 - t)`, `ln(1 + Pi^C) = ln(1 + Pi^{C,PE}) + Delta^rel`, y la separación de primer orden directa/en cascada |
| incidencia sectorial | `sector_incidence(..., groups=...)` con dispersión dentro de cada grupo ponderada por la producción |

`separate_baseline_tariffs(calib, wedges)` reexpresa los aranceles observados
en la referencia como cuñas brutas explícitas con impuestos residuales sobre
productos, de modo que los incrementos contrafactuales se acumulan sobre
aranceles iniciales no nulos sin contar dos veces la recaudación aduanera de
referencia (barreras a `1e-11`). Es una separación contable, no una prueba de
que un arancel no observado sea cero.

## Contrato de datos

`table_from_arrays` (o `BalancedIOTable.from_arrays`) acepta el esquema
`Z (M, M)`, `F (M, N, K)`, `VA (M,)`, `TLS (M,)`, `TFD (N, K)`,
`country_codes`, `sector_codes` y `fd_codes` en cualquiera de los seis códigos
de la OCDE, `("C", "G", "X", "V")`, `("C", "G", "X", "V", "VAL")` (VAL se une a
V y se registra), `("C", "G", "X")`, el `("C", "I", "Cx")` de la tabla
procesada, o los vocabularios de demanda final de los cargadores de
`puremacro.trade.data`: FIGARO (`P3_S14`, `P3_S15`, `P3_S13` sumados en C,
`P51G` en G, `P5M` en V), EXIOBASE (`HFCE`, `NPISH`, `GGFC`, `GFCF`, `INVNT`,
`VALUABLES` unido a V, y una columna `EXPORT` que debe ser idénticamente cero
en una tabla cerrada), WIOD (`CONS_h`, `CONS_np`, `CONS_g`, `GFCF`, `INVT`) y
Eora (`HFCE`, `NPISH`, `GGFC`, `GFCF`, `INVNT`, `ACQ_VAL` unido a V). Los
códigos que se asignan a una misma categoría se suman; C y G son obligatorias,
X y V valen cero cuando faltan. Los esquemas sin columna de compras en el
exterior (todos salvo el de la OCDE) dan una canasta X vacía, así que
`calibrate_condensed` necesita `allow_empty_purchases_abroad=True` para ellos
(X copia la canasta C como precio auxiliar; su gasto es idénticamente cero).
Los esquemas de tres categorías y cualquier tabla con celdas G negativas
necesitan `negative_investment="raise"` (por defecto) o `"to_inventory"`.
`BalancedIOTable.from_io_table` acepta cualquier objeto con esos atributos
(incluida una `Z` dispersa), `from_raw` acepta un
`puremacro.trade.data.RawIOData` de cualquiera de los cinco cargadores, y
`from_trade_calibration` hace de puente desde un `TradeCalibrationResult`
(trabajo y capital se suman en el factor único; G contiene los inventarios; Cx
pasa a ser X para una tabla sin asignación de usos finales registrada y para la
asignación `DPABR` de la OCDE, mientras que el Cx de los cargadores FIGARO,
WIOD y Eora pasa a ser el V exógeno, así que esas tablas se calibran con
`allow_empty_purchases_abroad=True`, y el Cx de EXIOBASE necesita un
`cx_category` explícito). La construcción añade fantasmas de `1e-6` a las
celdas vacías, pone un piso de `1e-3` de la producción al valor agregado no
positivo, recalcula TLS como residuo de columna y pone una barrera al cambio
(`1e-4` del |TLS| mundial, `1e-2` de la producción de una celda), y asigna el
redondeo mundial de los saldos comerciales al país de referencia (el último
código por defecto). La máscara de mercancías (secciones A-C de la CIIU Rev.4;
los suministros, la construcción y los servicios no son mercancías) se infiere
para el registro de 45 industrias de la OCDE, las letras de sección de la CIIU
Rev.4 y el fixture congelado de tres sectores; los demás registros (los códigos sectoriales de FIGARO,
EXIOBASE, WIOD y Eora, y los códigos sintéticos) pasan `merchandise_mask`, por
ejemplo a partir de `puremacro.trade.mrio.goods_mask`.

Los aranceles son tasas ad valorem `r >= 0`; las transacciones domésticas nunca
se gravan y las compras de residentes en el exterior (X) están exentas. Los
multiplicadores menores que uno (subsidios a la importación) los rechaza
`TariffWedges.from_arrays`, lo que mantiene la matriz de costos no decreciente
en la escala arancelaria, de modo que un único certificado de existencia con el
arancel completo cubre todas las etapas de la continuación.
`build_tariff_wedges` acepta un escalar, una tabla `(N, S)` de origen por
sector para un importador, un arreglo `(N, N, S)`, un diccionario
`{importer: {origin | "*": rate}}` (represalias; un origen nombrado siempre
prevalece sobre `"*"`, sea cual sea el orden de las claves) o una función como
`TariffScenario.get_rate`. `wedges.to_dataframe()` enumera las tasas no nulas
por origen, importador y sector. La cobertura por defecto son las mercancías
(`coverage="all"` incluye los servicios); `fd_tariffed` elige cuáles de C, G, V
pagan el arancel. Con `coverage="goods"`, una especificación explícita por
sector (un arreglo con eje sectorial, un arreglo `(S,)` en un diccionario o una
función) que pone aranceles a sectores de servicios se recorta a las
mercancías con un `RuntimeWarning`, y los sectores recortados se registran en
`wedges.metadata["dropped_service_rates"]`; las tasas escalares significan
"todos los sectores cubiertos" y no avisan. Las concordancias y las tablas
agregadas pertenecen a `puremacro.trade.mrio` (sus tablas entran por
`from_io_table`).

## Dos modelos, no uno

| | `solve_trade_equilibrium` (contabilidad consistente) | `puremacro.trade.condensed` |
|---|---|---|
| Factores | trabajo y capital, valor agregado Cobb-Douglas | un factor compuesto por país |
| Demanda final | participaciones C, I, Cx del ingreso; ahorro externo fijo restado de I | C, G, X Cobb-Douglas sobre la absorción neta de inventarios exógenos V |
| Saldo comercial | ahorro externo base fijo (en unidades del numerario o como proporción del ingreso factorial mundial) | proporciones del ingreso factorial mundial (también del PIB mundial o del PIB propio) |
| Numerario | primer precio de productor = 1 | ingreso factorial mundial (o el salario de cualquier país) |
| Incógnitas | sistema completo (Newton, LU dispersa, Krylov, PAC de Keller) | condensación exacta de 2N `(log w, Y / Y0)` |
| Tolerancia y aceptación | residuo y libro contable a la tolerancia del solucionador | residuo de `1e-12`, Walras de `1e-10`, diez bloques de flujos brutos a `1e-10` |
| Bienestar | EV/CV hicksianas con atribución de Shapley entre extremos | EV con descomposición aditiva exacta, ingreso nacional real, ToT de Fisher |
| Aranceles de referencia | los flujos observados son flujos a precios básicos | `separate_baseline_tariffs` |

Los resultados no son intercambiables entre las dos columnas. Use el modelo
condensado cuando quiera una referencia de Leontief de un factor certificada y
con medidas independientes del numerario; use la contabilidad consistente
cuando necesite la estructura de dos factores o las herramientas de política
hicksianas.

## Qué está validado y qué no

Verificado en este repositorio (`tests/test_trade_condensed.py`): reproducción
de la referencia a `1e-14` para todos los cierres y numerarios; certificados de
`1e-10` o menos en una tabla sintética 3x4, el fixture congelado OCDE 3x3, una
tabla sintética 20x6 y una tabla sintética 5x4; el certificado reacciona a
perturbaciones de `1e-5` de `w, p, y, Y, XN, TR` en el bloque nombrado y a una
perturbación salarial de `1e-9`; certificados de tablas en unidades naturales
con de una a seis celdas vacías (fantasma); aditividad de la descomposición de
la EV a `1e-12`; invariancia al numerario de dieciocho medidas y componentes de
la descomposición a `1e-9`; un arancel sobre cantidades fijas de inventarios es
neutral con el cierre factorial y no con el cierre de PIB mundial;
homogeneidad al escalar la tabla por 1000; una tasa efectiva exacta del `10%`
para un arancel uniforme, también cuando los inventarios no pagan arancel; los
jacobianos hacia adelante y centrados coinciden a `1e-6` relativo; la cota de
EP queda por debajo del precio al consumidor de EG; las etapas de continuación,
la detección de estancamiento, el respaldo anidado, los monitores y la llegada
por longitud de arco, los arranques múltiples cercanos y la recuperación desde
un arranque lejano en la tabla 20x6; una frontera de existencia que falla de
forma cerrada en la economía sintética 3x4 (un arancel del 500% de su primer
país sobre las mercancías supera la barrera de existencia, pero el equilibrio
con un salario positivo en ese país deja de existir cerca de un arancel del
259%, `lambda* = 0.51773` de 500%: a lo largo de la rama `lambda` crece de
forma monótona mientras el salario tiende a cero; la continuación en el
parámetro natural se estanca en `lambda = 0.515625`, la pseudolongitud de arco
lanza `FoldDetected` donde el jacobiano es numéricamente singular, y
`solve_condensed` termina en `EquilibriumNotFound` nombrando el
estancamiento); y los cinco esquemas de cargadores de `puremacro.trade.data`
en tablas sintéticas 3x3.

No validado: no se afirma unicidad global, ni un umbral de pliegue universal,
ni una garantía global para Newton. Las comprobaciones numéricas del residuo no
establecen existencia, unicidad ni optimalidad global de un equilibrio, y los
umbrales del certificado son criterios de verificación en coma flotante, no
encierros de raíces por intervalos. La longitud de arco y los arranques
múltiples son evidencia, no pruebas. No se ha exhibido ningún punto de retorno
en un ejemplo económico: el monitor de pliegue se ejercita con un cambio de
signo sintético del determinante y con la frontera de existencia de arriba, así
que `FoldDetected` significa que la trayectoria no puede continuarse de forma
regular (un punto de retorno o una frontera de existencia), y la frontera no es
un umbral de pliegue universal. La EV estática incluye la inversión en el
compuesto de absorción. Las ejecuciones completas de 45 sectores sobre las
tablas de la OCDE usan factorizaciones LU densas de matrices `M x M` con `M` en
los miles y no son un objetivo para el navegador.

## Referencias

- Dekle, R., Eaton, J., & Kortum, S. (2008). "Global Rebalancing with Gravity: Measuring the Burden of Adjustment." *IMF Staff Papers*, 55(3), 511–540.

Relacionado: [contabilidad de comercio consistente](trade_accounting.md),
[bienestar hicksiano del consumo](trade_welfare.md),
[política arancelaria hicksiana](trade_policy.md),
[tablas MRIO](trade_mrio.md),
[estado de la validación estructural](STRUCTURAL_VALIDATION_STATUS.md).
