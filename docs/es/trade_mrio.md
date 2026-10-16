> 🇬🇧 [English](../trade_mrio.md) · 🇪🇸 Español

# Tablas MRIO nativas: procedencia, lectores y agregación

`puremacro.trade.mrio` lee bases de datos multirregionales de insumo-producto
publicadas en su resolución nativa, registra de dónde sale cada número, las
balancea con un contrato declarado y las agrega de forma exacta. Es la capa de
ingesta sobre la que se apoyan los solucionadores de comercio: una tabla se
convierte en matriz de calibración con `to_calibration_matrix`, y una
calibración vuelve a ser tabla con `MRIOTable.from_trade_calibration`.

```python
from puremacro.trade import calibrate_trade_model
from puremacro.trade.data import synthetic_codes, synthetic_icio_table
from puremacro.trade.mrio import Concordance, MRIOTable, aggregate_mrio, to_calibration_matrix

# A seeded synthetic table (six countries, four sectors) and its table view.
countries, sectors = synthetic_codes(6, 4)
calib = calibrate_trade_model(synthetic_icio_table(6, 4, seed=2), country_codes=countries, sector_codes=sectors)
table = MRIOTable.from_trade_calibration(calib)
print(table.summary())
print(table.accounting_report().to_markdown())

# Keep C01 and C02, merge the rest; two sector groups.
blocs = Concordance.from_mapping(
    "blocs", {c: c if c in ("C01", "C02") else "REST" for c in table.country_codes})
two = Concordance.from_mapping(
    "two", {s: "GOODS" if k < 2 else "SERVICES" for k, s in enumerate(table.sector_codes)})
coarse = aggregate_mrio(table, two, region_concordance=blocs)
print(coarse.to_markdown("countries"))
print(coarse.metadata["aggregation"]["aggregated_identity_rel"])   # row-column identity after aggregation

matrix = to_calibration_matrix(coarse)                             # (9, 15): 3 countries x 2 sectors
coarse_calib = calibrate_trade_model(matrix, country_codes=coarse.country_codes,
                                     sector_codes=coarse.sector_codes)
```

La tabla sintética está balanceada, así que el informe contable muestra las
identidades de filas, columnas y producción al nivel del redondeo, y la
agregación conserva todos los totales (el residuo de la identidad agregada es
de unos `3e-16` relativos). Los archivos nativos se leen del mismo modo:

```python
# requires: the native OECD ICIO 2023, Eurostat FIGARO 2026 and EXIOBASE 3.8.2 files (not shipped)
from puremacro.trade.mrio import aggregate_mrio, read_exiobase_native, read_figaro_native, read_oecd_native, regularize_table

oecd = read_oecd_native("2019_SML.csv", 2019)          # refuses the known-corrupted digest
figaro = read_figaro_native("matrix_eu-ic-io_ind-by-ind_26ed_2019.csv")
exio = read_exiobase_native("IOT_2019_ixi.zip")         # sparse Z, members checked against the archive CRCs
core = read_exiobase_native("exiobase_core", archive="IOT_2019_ixi.zip")  # extracted members checked too
balanced, report = regularize_table(oecd)
sections = aggregate_mrio(balanced, "isic_section")
print(report.to_markdown("gates"))
```

Un archivo FIGARO de dos países y un directorio EXIOBASE de dos regiones con los
esquemas oficiales se distribuyen en `tests/fixtures/trade/mrio/`, de modo que
`read_figaro_native("tests/fixtures/trade/mrio/figaro_2c2s_26ed.csv")` corre
desde una copia del repositorio (con un `RuntimeWarning` de que el fixture no
es la edición registrada).

## El contrato de la tabla

Un `MRIOTable` es una dataclass congelada cuyos arreglos son copias de solo
lectura. Las celdas van por país, `i = country_index * S + sector_index`.

| atributo | forma | significado |
|---|---|---|
| `Z` | `(M, M)` densa o `scipy.sparse.csr_matrix` | entregas intermedias a precios básicos; las filas venden, las columnas compran |
| `F` | `(M, N, K)` | entregas finales por celda vendedora, país de destino y categoría |
| `fd_codes` | `K` cadenas | subconjunto de `("C", "G", "X", "V", "VAL")` en ese orden |
| `VA`, `TLS`, `output` | `(M,)` | valor agregado tal como se registra, impuestos netos de subsidios sobre las compras intermedias del comprador, producción bruta |
| `TFD` | `(N, K)` | impuestos netos de subsidios sobre las compras finales |
| `production_taxes`, `labor_compensation`, `operating_surplus` | `(M,)` o `None` | detalle factorial donde la fuente lo tiene |
| `merchandise_mask` | `(S,)` bool o `None` | sectores de bienes (CIIU A-C) |
| `metadata` | dict | `sources` (tupla de `SourceRecord`), `transformations` (tupla de cadenas), convenciones |

C es el consumo de hogares, ISFLSH y gobierno; G es la formación bruta de
capital fijo; X son las compras de residentes en el exterior (la canasta DPABR
de la OCDE); V es la variación con signo de existencias; VAL son los objetos
valiosos. El constructor con validación es `MRIOTable.from_arrays`: `Z`, G y X
deben ser no negativos, todo debe ser finito, y trabajo y excedente de
explotación se suministran juntos. El consumo, las existencias, los objetos
valiosos, el valor agregado y los impuestos negativos son datos reales de la
fuente y se conservan. `accounting_report()` recalcula de forma independiente
las identidades de filas y columnas y cuenta las entradas negativas; allí no se
rebalancea nada. `net_exports()` mide saldos fob sobre todos los usos y asigna
el redondeo en coma flotante de su suma mundial a `reference_country`.
`to_raw("cix")` da un `RawIOData` con las columnas procesadas C, I = G + V + VAL,
Cx = X.

## Procedencia

`identify_source(path, expected_sha256=..., expected_md5=...)` lee el archivo
en bloques de 4 MiB y rechaza un archivo ausente o modificado.
`verify_zip_member(archive, member, extracted_path=None)` comprueba el tamaño y
el CRC-32 de un miembro contra el directorio del archivo comprimido. Cada lector
guarda los `SourceRecord` resultantes en `metadata["sources"]`, con un `status`
de `"verified"`, `"whitelisted"`, `"hashed"` o `"unknown_edition"`.

`check_oecd_source(path, year=...)` compara un CSV de la OCDE (o el miembro del
año del ZIP 2016-2020) con dos registros:

| registro | resumen | significado |
|---|---|---|
| `OECD_ICIO_MD5[2019]` | `28cba31491177955445051d459053744` | `2019_SML.csv` aceptado, edición 2023 de la OCDE ICIO |
| `OECD_ICIO_MD5[2020]` | `d3e0f4979d85d6c0bb7cf4c43e324287` | `2020_SML.csv` aceptado |
| `OECD_KNOWN_CORRUPTED_MD5` | `d1b887aaafa54ab3f28fde78fcd21cdf` | una copia del archivo SML de 2020 con puntos decimales perdidos: los valores con 3-4 decimales perdieron el punto decimal y los menores que 0.001 pasaron a 0; se rechaza |

Un resumen corrupto lanza `MRIOIntegrityError`; uno de la lista blanca se
registra; cualquier otro se lee con un `RuntimeWarning` que lo declara edición
no autenticada. El lector de FIGARO compara el CSV con
`FIGARO_2026ED_2019_SHA256` y el de EXIOBASE compara el archivo comprimido con
`EXIOBASE_382_2019_IXI_SHA256` del mismo modo; ambos exigen un
`expected_sha256` cuando se pasa, y un archivo registrado leído para un año
distinto de 2019 lanza un error. Los miembros de EXIOBASE leídos del archivo
comprimido se comprueban contra sus entradas CRC-32. Un directorio EXIOBASE
extraído solo se autentica cuando se pasa su archivo comprimido (`archive=`):
el archivo se resume y cada miembro extraído se comprueba contra el tamaño y el
CRC-32 de su entrada, de modo que un byte cambiado lanza un error. Leído solo,
los miembros de un directorio se resumen, se registran como
`"unknown_edition"` y se señalan con un `RuntimeWarning`;
`metadata["source_authenticated"]` dice qué caso se aplicó. Los lectores nunca
descargan datos ni generan un fixture sustituto (el auxiliar de descarga del
archivo de la OCDE de [las herramientas de la tabla incluida](spatial_and_trade_ge.md)
es un paso aparte y explícito).

## Lectores y convenciones de las fuentes

| | OCDE ICIO, edición 2023 | Eurostat FIGARO, edición 2026 | EXIOBASE 3.8.2 ixi |
|---|---|---|---|
| lector | `read_oecd_native(path, year)` | `read_figaro_native(path, year=2019)` | `read_exiobase_native(path_or_zip, year=2019, archive=None)` |
| esquema | CSV SML de 77 economías x 45 industrias con filas TLS/VA/OUT | cabecera `rowLabels`, celdas `{region}_{industry}`, cinco columnas de uso final por región, seis filas finales `W2_*`; dimensiones derivadas de las etiquetas | `Z.txt`, `Y.txt`, `x.txt`, `industries.txt`, `satellite/F.txt`, `satellite/F_Y.txt` con cabeceras de dos niveles región/sector |
| unidades | millones de USD, precios corrientes | millones de EUR, precios básicos corrientes | millones de EUR, precios básicos corrientes |
| usos finales | C = HFCE + NPISH + GGFC, G = GFCF, X = DPABR, V = INVNT | C = P3_S13 + P3_S14 + P3_S15, G = P51G, X = 0, V = P5M | C = y01 + y02.a + y02.b, G = y04, X = 0, V = y05.a, VAL = y05.b |
| consumo | consumo de residentes con una canasta DPABR separada; incluye al gobierno | consumo territorial; incluye al gobierno; los totales `W2_OP_RES` y `W2_OP_NRES` se guardan en los metadatos | consumo publicado; incluye al gobierno; sin canasta DPABR |
| existencias | INVNT es la variación con signo; no se observa el stock | P5M une variación de existencias y objetos valiosos; la fuente no los separa | existencias y objetos valiosos son flujos separados |
| detalle factorial | ninguno: la fila VA no tiene separación observada trabajo/capital | `W2_D1` trabajo, `W2_D29X39` otros impuestos netos sobre la producción, `W2_B2A3G` excedente bruto de explotación e ingreso mixto | tres filas de cualificación para el trabajo, cuatro filas de excedente de explotación (incluidas rentas de la tierra y de recursos), otros impuestos netos sobre la producción |
| producción | columna OUT del proveedor | total de fila de las transacciones publicadas | `x.txt` |
| año | 1995-2020 | 2019 autenticado | 2019, una estimación adelantada |
| almacenamiento | densa | densa (`sparse_matrix=True` opcional) | CSR leída en bloques de 128 filas |

Los códigos de región se asignan a ISO-3 mediante `FIGARO_ISO3` (`FIGW1` pasa
a ser `ROW`) y `EXIOBASE_ISO3` (las cinco regiones del resto del mundo
conservan nombres distintos). Las máscaras de mercancías siguen `goods_mask`:
secciones A-C de la CIIU para las industrias de la OCDE y los grupos de
secciones CIIU, NACE A-C salvo C33 (reparación e instalación, que no es un
envío de mercancías) para FIGARO, industrias i01-i37 salvo i01.w para EXIOBASE.

El lector de FIGARO aplica, y registra en `metadata["rounding_adjustments"]`,
dos reparaciones acotadas del redondeo: una celda vacía cuya producción es una
unidad de redondeo negativa (`-0.001 <= y < 0` con masa absoluta de insumos de
como mucho `0.002`) se pone a cero, y una entrada de consumo en `[-0.001, 0)` se
mueve a V para que los totales de fila y el gasto del país no cambien.
Cualquier cosa mayor lanza un error. `rounding_repair=False` conserva el
redondeo publicado. El lector de EXIOBASE rechaza una columna
`Exports: Total (fob)` no nula y filas factoriales de demanda final no nulas,
porque requerirían un modelo explícito de destino.

## Contrato de regularización

`regularize_table(table)` balancea una tabla bruta con el contrato de
construcción de `puremacro.trade.condensed.table_from_arrays`, para cualquier
tabla con categorías nombradas. Con `out` la columna de producción, `row` los
totales de fila y `active = out > 0`:

1. Guarda de identidad: `median_{active} |row_i - out_i| / out_i <= 1e-5`, sin
   producción negativa, y toda celda de producción nula completamente vacía.
2. Fantasmas: cada celda de producción nula recibe `F[i, own country, C] = VA_i = 1e-6`
   para que su fila y su columna coincidan exactamente (`inactive="mask"` las
   deja vacías).
3. Piso del valor agregado: `VA_i = 1e-3 * out_i` solo donde es activa y `VA_i <= 0`.
4. Los impuestos netos sobre productos se recalculan como el residuo que cierra
   cada columna, `tls_j = row_j - sum_i Z_ij - VA_j`. En las celdas sin piso
   el cambio debe cumplir `sum_j |tls_j - TLS_j| <= 1e-4 * sum_j |TLS_j|` y
   `max_j |tls_j - TLS_j| / max(out_j, 1) <= 1e-2` (con una holgura relativa de
   `1e-9` en la segunda barrera).
5. Los saldos comerciales se miden fob sobre todos los usos;
   `|sum_o xn_o| <= 1e-9 * world GDP` con `world GDP = sum VA + sum tls + sum TFD`.

La tabla devuelta tiene `output` igual a los totales de fila tras los
fantasmas, el `TLS` residual y el paso añadido a `metadata["transformations"]`;
el `MRIOBuildReport` lleva los fantasmas, los pisos, ambos totales del impuesto
residual, el redondeo mundial de los saldos comerciales, el PIB mundial y cada
barrera con su valor, tolerancia e indicador de aprobación, y nombra las celdas
problemáticas cuando una barrera trata de celdas (`"cells"`, `"n_cells"` y
`"worst_cell"` para la barrera por celda). Cada tolerancia es un argumento;
`strict=False` devuelve el informe con `passed=False` en lugar de lanzar un
error (la barrera por celda es estrecha, y una edición publicada puede quedar
justo en ella o apenas por encima). Una tabla así se marca con
`metadata["regularized"] = False` con las barreras fallidas en
`metadata["failed_gates"]` y en su cadena de transformación, y
`aggregate_mrio` y `to_calibration_matrix` avisan cuando la reciben. Con
`strict=False` solo reciben un fantasma las celdas vacías con producción
exactamente cero; las celdas con producción negativa o entradas sueltas
conservan sus datos y se enumeran en `report.skipped_phantoms`. El fantasma de
`1e-6` y la escala `max(out, 1)` de la barrera por celda son cantidades
absolutas que suponen una tabla en millones de unidades monetarias, así que
reescale `phantom` para una tabla en otras unidades. `spectral=True` añade las
cotas de Collatz-Wielandt de `Z / output` de
`puremacro.trade.regularize.compute_spectral_radius`. Es un contrato distinto
del de `regularize_mrio_table`, que pone a todo valor agregado de celda activa
el piso `max(1e-3 Y, min(1, 0.5 Y))` (4.3.0: `max(1e-3 Y, 1)`, que elevaba el
valor agregado por encima de la producción en celdas menores de 1 M USD) y
recalcula la producción a partir de las ventas por fila antes del residuo salvo
que se suministre `Y`; los dos no son intercambiables.

## Agregación

`aggregate_mrio(table, concordance)` calcula la agregación exacta con
`P = I_N kron C^T`, donde `C` es la matriz 0/1 `(S, G)` de una `Concordance`:

```
Z_G = P Z P^T,   F_G[:, n, k] = P F[:, n, k],   VA_G = P VA,   TLS_G = P TLS,   output_G = P output
```

`TFD` no cambia, el detalle factorial se suma del mismo modo y una
`region_concordance` une también orígenes y destinos. Ambas identidades
contables son lineales, así que la agregación conmuta con el cierre del
impuesto residual cuando no se aplica ningún piso, y todo total se conserva
hasta el redondeo en coma flotante. Cuando la tabla fina está balanceada, la
gruesa también debe estarlo (`max_j |row_j - col_j| / max(|col_j|, 1) <= 1e-8`);
el residuo se registra en `metadata["aggregation"]` en cualquier caso. Un grupo
grueso es de mercancías solo cuando lo son todos sus miembros; los grupos
mixtos se enumeran.

En `CONCORDANCES` hay registrado un mapa con nombre de las 45 industrias de la
OCDE:

| nombre | grupos | grupos de bienes |
|---|---|---|
| `"isic_section"` | `ISIC_SECTION_CODES`: las 20 secciones A-T de la CIIU Rev.4 | A, B, C |

Cualquier otra agrupación es una `Concordance` construida con
`Concordance.from_mapping`, `Concordance.build` o `Concordance.identity`, o un
diccionario pasado a `aggregate_mrio`; `compose` encadena dos mapas.
`goods_mask` decide la máscara de mercancías etiqueta por etiqueta, de modo que
cualquier orden o subconjunto de un registro da la misma respuesta por código,
y un código fuera del registro nombrado lanza `ValueError`.

## Puentes con la calibración

`to_calibration_matrix(table)` devuelve la matriz `(M + 3, M + 3N)` para
`calibrate_trade_model`: las filas `0..M-1` son `[Z | F]` con C,
I = G + V + VAL y Cx = X por destino, la fila `M` es TLS con los impuestos
finales condensados del mismo modo, y las filas `M + 1` y `M + 2` son trabajo y
capital. `calibrate_trade_model` reconstruye cada columna con una separación
Cobb-Douglas y solo la acepta cuando ambos factores son estrictamente positivos
(una participación del capital estrictamente entre 0 y 1) o ambos son cero, así
que las filas factoriales se construyen con ese criterio:

* Los otros impuestos netos sobre la producción pasan por defecto a la fila TLS
  (`production_taxes="to_tls"`), que la calibración modela como impuesto sobre
  la producción, salvo en celdas donde son al menos tan grandes como el valor
  agregado. Allí se quedan en el valor agregado y un `RuntimeWarning` nombra
  las celdas (las tablas publicadas de FIGARO y EXIOBASE tienen celdas así).
  `"to_factors"` deja todo impuesto sobre la producción dentro del valor
  agregado.
* Trabajo y capital se reparten el valor agregado factorial restante en las
  proporciones observadas donde ambos son estrictamente positivos, y según
  `labor_share` (2/3-1/3 por defecto) en el resto: celdas fantasma, celdas con
  un factor negativo y celdas con un factor exactamente cero. Sin detalle,
  todas las celdas usan la separación por defecto.
* Una celda con valor agregado factorial negativo lanza `MRIOIntegrityError`;
  `regularize_table` pone un piso al valor agregado donde no es positivo.

`return_details=True` devuelve además las celdas de cada caso de respaldo. Una
tabla bruta provoca un `RuntimeWarning` que recomienda `regularize_table`
primero, y también una tabla cuyas barreras de regularización fallaron.
`MRIOTable.from_trade_calibration(calib)` es el puente inverso desde
`calib.data_calibra`. Lee la columna Cx de la calibración como X (compras de
residentes en el exterior, exentas de aranceles) solo para una tabla sin
asignación de usos finales registrada y para la asignación `DPABR` de la OCDE
(`calib.metadata["final_use_mapping"]`); el Cx de los cargadores FIGARO, WIOD
y Eora (existencias y objetos valiosos con signo) pasa a ser V, y el Cx de
EXIOBASE, que incluye su columna de exportaciones, lanza `ValueError` salvo que
`cx_category` nombre la categoría. Una celda de inversión negativa movida con
`negative_investment="to_inventory"` se suma a V. Una tabla obtenida de una
calibración FIGARO, WIOD o Eora no tiene por tanto X, y `to_calibration_matrix`
la devuelve con una columna Cx vacía, que la contabilidad consistente trata como
categoría inactiva.

## Qué está validado y qué no

`tests/test_trade_mrio.py` comprueba, en fixtures diseñados a mano con arreglos
esperados derivados a mano: los auxiliares de procedencia frente a `hashlib` y
miembros corruptos o discordantes; el álgebra de concordancias (la proyección es
igual al producto de Kronecker explícito, composición, máscaras gruesas); el
contrato de la tabla; los lectores de FIGARO y EXIOBASE, incluidas las
reparaciones de redondeo, las rutas zip, directorio comprobado contra el
archivo comprimido y por bloques, un byte cambiado en un miembro extraído, y
cada rechazo; el lector de la OCDE en un archivo diminuto con esquema SML;
cada barrera de regularización en modo estricto y de informe, y que una barrera
fallida nunca da una tabla marcada como regularizada; conservación,
composición, agregación de regiones y la coherencia de bienes de las 45
industrias a las secciones CIIU; la calibración del fixture OCDE agregado con
una solución base a precios uno; la calibración de tablas con celdas de factor
nulo y con impuestos sobre la producción mayores que el valor agregado; y el
viaje de ida y vuelta a través de una calibración sintética.

No validado, y no afirmado:

* Equilibrio general a tamaño completo en FIGARO o EXIOBASE. Este módulo
  promete tablas en resolución nativa, no equilibrios en resolución nativa;
  EXIOBASE ni siquiera se calibra aquí, porque `to_calibration_matrix`
  densifica `Z`.
* La comparabilidad del bienestar entre bases de datos. FIGARO conserva el
  consumo territorial y no se ha validado un puente de residencia; la edición
  2019 de EXIOBASE es una estimación adelantada y no se implica una afirmación
  de medida más fuerte; `active_mask` significa producción informada positiva,
  no un certificado de viabilidad económica.
* Otras ediciones y años. Los resúmenes identifican archivos 2019 concretos;
  una edición distinta se lee con un aviso y sin afirmación.
* Un lector de GTAP. Los datos de GTAP tienen licencia y no pueden distribuirse
  ni siquiera como fixtures, y el único analizador HAR de código abierto
  completo tiene licencia GPL-3, así que no puede redistribuirse bajo la
  licencia MIT de puremacro.
* La barrera por celda del impuesto residual es frágil por construcción. Su
  tolerancia es un argumento y el informe se devuelve en cualquier caso.

## Documentos relacionados

* [Contabilidad de comercio consistente](trade_accounting.md) para el
  solucionador al que alimenta la matriz de calibración.
* [Bienestar hicksiano del consumo](trade_welfare.md): el C nativo de la OCDE
  incluye el consumo del gobierno.
* [Modelo de Leontief condensado con un factor](trade_condensed.md), que
  comparte el contrato de regularización.
* [Equilibrio general espacial y de comercio cuantitativo](spatial_and_trade_ge.md)
  para la tabla incluida y los adaptadores armonizados.
* [Estado de la validación estructural](STRUCTURAL_VALIDATION_STATUS.md).
