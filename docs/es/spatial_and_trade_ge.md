> 🇬🇧 [English](../spatial_and_trade_ge.md) · 🇪🇸 Español

# Economía espacial cuantitativa y equilibrio general de comercio con gravedad

`puremacro.trade` y `puremacro.spatial` implementan dos marcos de equilibrio general para el comercio internacional y la geografía económica:
1. El modelo de comercio ricardiano multisectorial y multipaís con eslabonamientos insumo-producto, bienes intermedios y aranceles desarrollado por **Caliendo y Parro (2015, *Review of Economic Studies*)**, resuelto mediante **Álgebra Exacta de Sombreros** (*Exact Hat Algebra*) (`puremacro.trade.CaliendoParroModel`).
2. El modelo continuo de equilibrio espacial geográfico con movilidad laboral, externalidades de aglomeración y fuerzas de congestión desarrollado por **Allen y Arkolakis (2014, *Quarterly Journal of Economics*)** (`puremacro.spatial.AllenArkolakisModel`).

Ambos cumplen de forma estricta el contrato de ejecución en Python puro / Pyodide (cero dependencias en C++, basado exclusivamente en NumPy/SciPy/Pandas/Matplotlib), permitiendo evaluaciones contrafactuales de política económica —tales como cambios arancelarios y represalias, inversiones regionales en infraestructura de transporte y perturbaciones ambientales localizadas. El CGE de comercio multipaís y multisectorial completo sobre tablas de insumo-producto (`calibrate_trade_model`, `solve_trade_equilibrium`) se documenta en [contabilidad de comercio consistente](trade_accounting.md); sus datos incluidos se describen en la sección 7.

---

## 1. Marco teórico y algorítmico

### 1.1 Modelo de comercio ricardiano multisectorial con eslabonamientos insumo-producto de Caliendo-Parro (2015)

Considérese una economía global con $N$ países ($n, i = 1, \dots, N$) y $J$ sectores ($j, k = 1, \dots, J$). La producción en el país $n$ y sector $j$ combina trabajo $L_n^j$ con insumos intermedios provenientes de todos los sectores $k = 1, \dots, J$ bajo rendimientos constantes a escala:

$$Y_n^j = z_n^j \left( L_n^j \right)^{\gamma_n^j} \prod_{k=1}^J \left( M_n^{j, k} \right)^{\gamma_n^{j, k}}$$

donde $\gamma_n^j > 0$ es la participación del valor agregado, $\gamma_n^{j, k} \ge 0$ es la participación del costo del insumo intermedio $k$ en la producción del bien $j$, y las participaciones de costos suman la unidad:

$$\gamma_n^j + \sum_{k=1}^J \gamma_n^{j, k} = 1, \quad \forall n, j$$

La minimización de costos determina el costo unitario de la canasta de insumos:

$$c_n^j = \Upsilon_n^j w_n^{\gamma_n^j} \prod_{k=1}^J \left( P_n^k \right)^{\gamma_n^{j, k}}$$

donde $w_n$ denota el salario competitivo en el país $n$, $P_n^k$ es el índice de precios compuesto del sector $k$, e $\Upsilon_n^j$ es una constante tecnológica.

### 1.2 Formulación de Álgebra Exacta de Sombreros

Siguiendo la metodología de **Álgebra Exacta de Sombreros** (*Exact Hat Algebra*) de Dekle, Eaton y Kortum (2007) generalizada por Caliendo y Parro (2015), las variaciones contrafactuales se expresan en términos de cambios proporcionales $\hat{x} \equiv x' / x$. Esto permite resolver equilibrios contrafactuales directamente a partir de cuotas observadas de comercio bilateral en el equilibrio base $\pi_{ni}^j$ y matrices insumo-producto, sin requerir la estimación de fundamentos inobservables (tales como niveles absolutos de tecnología $T_n^j$ o costos bilaterales tipo iceberg $d_{ni}^j$).

Dada una variación arancelaria contrafactual bruta $\hat{\tau}_{ni}^j \equiv (1 + t_{ni}'^j) / (1 + t_{ni}^j)$:

1. **Variación de costos unitarios**:
   $$\hat{c}_n^j = \hat{w}_n^{\gamma_n^j} \prod_{k=1}^J \left( \hat{P}_n^k \right)^{\gamma_n^{j, k}}$$
2. **Variación de cuotas de comercio bilateral (Gravedad de Eaton-Kortum)**:
   $$\hat{\pi}_{ni}^j = \left( \frac{\hat{c}_i^j \hat{\tau}_{ni}^j}{\hat{P}_n^j} \right)^{-\theta_j}$$
   donde $\theta_j > 0$ es la elasticidad comercial sectorial.
3. **Variación de índices de precios sectoriales**:
   $$\hat{P}_n^j = \left( \sum_{i=1}^N \pi_{ni}^j \left( \hat{c}_i^j \hat{\tau}_{ni}^j \right)^{-\theta_j} \right)^{-\frac{1}{\theta_j}}$$
4. **Gasto y vaciado de mercados de bienes**:
   El gasto contrafactual sectorial $X_n'^j$ y el producto bruto $Y_n'^j$ resuelven el balance insumo-producto leontiefiano global:
   $$X_n'^j = \alpha_n^j I_n' + \sum_{k=1}^J \gamma_n^{k, j} Y_n'^k, \quad Y_n'^j = \sum_{i=1}^N \frac{\pi_{in}'^j}{\tau_{in}'^j} X_i'^j$$
   donde $I_n' = w_n' L_n + R_n' + D_n'$ es el ingreso nacional compuesto por ingresos laborales, recaudación arancelaria $R_n'$ y desbalances comerciales exógenos $D_n'$.
5. **Vaciado de mercado laboral y tanteo salarial (*Tâtonnement*)**:
   Los salarios $\hat{w}_n$ se actualizan iterativamente hasta que el exceso de demanda laboral satisface la tolerancia requerida:
   $$\left| \sum_{j=1}^J \gamma_n^j Y_n'^j - w_n' L_n \right| < \text{tol}$$
6. **Variaciones de ingreso real y bienestar agregado**:
   $$\widehat{\mathcal{W}}_n = \frac{\hat{I}_n}{\hat{P}_n}, \quad \text{donde} \quad \hat{P}_n = \prod_{j=1}^J \left( \hat{P}_n^j \right)^{\alpha_n^j}$$

### 1.3 Modelo gravitacional espacial geográfico con movilidad laboral de Allen-Arkolakis (2014)

Allen y Arkolakis (2014) formulan un modelo de equilibrio general espacial continuo sobre $N$ regiones geográficas donde los trabajadores migran libremente hasta igualar el nivel de utilidad real en todas partes.

#### Flujos comerciales y acceso a mercados
El comercio bilateral entre el origen $i$ y el destino $j$ obedece a una estructura gravitacional:

$$\pi_{ij} = \frac{\left( \frac{w_i \tau_{ij}}{A_i} \right)^{-\theta}}{\sum_{k=1}^N \left( \frac{w_k \tau_{kj}}{A_k} \right)^{-\theta}} = \frac{\left( \frac{w_i \tau_{ij}}{A_i} \right)^{-\theta}}{P_j^{-\theta}}$$

donde $\tau_{ij} \ge 1$ representan costos comerciales tipo iceberg ($\tau_{ii} = 1$), $\theta = \sigma - 1$ es la elasticidad comercial y $P_j$ es el índice de acceso a mercado de los consumidores (*Consumer Market Access*):

$$P_j^{-\theta} = \text{CMA}_j = \sum_{i=1}^N \tau_{ij}^{-\theta} \left( \frac{w_i}{A_i} \right)^{-\theta}$$

El acceso a mercado de las empresas (*Firm Market Access*, $\text{FMA}_i$) captura la demanda agregada de exportación:

$$\text{FMA}_i = \sum_{j=1}^N \tau_{ij}^{-\theta} P_j^\theta w_j L_j$$

El vaciado del mercado de bienes iguala el ingreso laboral regional con el valor total de las ventas hacia todos los destinos:

$$w_i L_i = \sum_{j=1}^N \pi_{ij} w_j L_j = \left( \frac{w_i}{A_i} \right)^{-\theta} \text{FMA}_i$$

### 1.4 Aglomeración, congestión y equilibrio espacial

La utilidad de un trabajador representativo en la localización $i$ depende de los salarios reales, amenidades locales $a_i$ y la población endógena $L_i$:

$$u_i = a_i \frac{w_i}{P_i}$$

La productividad y las amenidades endógenas incorporan efectos espaciales de desbordamiento (*spillovers*):
- **Aglomeración de productividad**: $A_i = \bar{A}_i L_i^\alpha$, donde $\alpha > 0$ refleja economías de escala externas marshallianas.
- **Congestión de amenidades**: $a_i = \bar{a}_i L_i^\beta$, donde $\beta < 0$ captura la inelasticidad en la oferta de vivienda, tiempos de traslado y congestión urbana.

#### Ecualización espacial de utilidad
La libre movilidad laboral iguala la utilidad a lo largo de todas las regiones pobladas:

$$u_i = \bar{u}, \quad \forall i \quad \implies \quad L_i = \bar{L} \frac{\left( \frac{w_i \bar{a}_i}{P_i} \right)^{-\frac{1}{\beta}}}{\sum_k \left( \frac{w_k \bar{a}_k}{P_k} \right)^{-\frac{1}{\beta}}}$$

#### Teorema de unicidad
Allen y Arkolakis (2014, Teorema 2) demuestran que el equilibrio general espacial es **único y estrictamente estable** si y solo si las fuerzas de congestión dominan a las economías de aglomeración:

$$\alpha + \beta \le \frac{\theta}{1 + \theta} \quad \text{y} \quad \alpha \le \frac{1}{\theta}$$

`AllenArkolakisModel.is_unique` evalúa analíticamente esta condición de unicidad durante la inicialización del modelo.

---

## 2. Opciones metodológicas y comparativa de modelos

| Dimensión | Caliendo-Parro (2015) | Allen-Arkolakis (2014) |
|---|---|---|
| **Mecanismo económico central** | Comercio multisectorial con eslabonamientos insumo-producto y aranceles | Movilidad laboral espacial con aglomeración y congestión |
| **Enfoque de resolución** | Álgebra Exacta de Sombreros sobre cuotas comerciales observadas | Mapeo de contracción de punto fijo sobre $(w_i, L_i)$ |
| **Márgenes endógenos** | Precios sectoriales, salarios, cuotas de gasto intermedio y final | Distribución de población, salarios, acceso a mercados |
| **Movilidad laboral** | Inmóvil entre países; móvil entre sectores domésticos | Libre movilidad espacial continua entre regiones |
| **Casos de uso principales** | Tratados de libre comercio, cambios arancelarios, represalias | Trenes de alta velocidad, corredores carreteros, choques climáticos |
| **Tasa de convergencia** | 10–30 iteraciones ($< 0.01$ s) | 20–50 iteraciones ($< 0.005$ s) |

### 2.1 Backends de hardware (`backend=`, 3.4.0)

El solucionador de punto fijo de Allen-Arkolakis acepta un argumento `backend` — `"numpy"`
(el valor por omisión), `"mlx"` (Apple Silicon) o `"cupy"` (NVIDIA); con el modelo
`aa_model` de la sección 4:

```text
aa_res = aa_model.solve_equilibrium(backend="mlx")
cf_res = aa_model.solve_counterfactual(trade_costs_new=tau_new, backend="mlx")
```

Conviene conocer tres propiedades antes de recurrir a ellos:

- **NumPy es el oráculo.** Toda ruta acelerada se contrasta contra la
  implementación de referencia en NumPy, y nunca al revés. El backend `numpy`
  está siempre disponible y preserva intacto el contrato de Pyodide.
- **`tol` sigue significando `tol`.** Metal no admite float64, de modo que la
  contracción con MLX en `AllenArkolakisModel` se ejecuta en float32 hasta un
  piso de 1e-6; el resultado se refina después con la contracción float64 de
  NumPy, inicializada en la solución del dispositivo, de manera que `converged`
  se refiere a la tolerancia `tol` solicitada.
- **El fallo es explícito y no fatal.** Un backend no instalado, un espacio de
  nombres de arreglos que no puede importarse o una excepción en el dispositivo
  emiten un `RuntimeWarning` y recurren a NumPy. Un resultado acelerado que no
  convergió tampoco se devuelve nunca:
  `AllenArkolakisModel` remata la solución del dispositivo con la contracción
  float64 de NumPy descrita arriba — sin advertencia, porque entonces
  `converged` es el veredicto de NumPy a la tolerancia `tol` solicitada.

```python
from puremacro._backend import available_backends
available_backends()          # ('numpy', 'numba', 'mlx') on this machine
```

`CaliendoParroModel` y el CGE de comercio `solve_trade_equilibrium` corren solo
en NumPy (`solve_trade_equilibrium(..., backend=...)` acepta `"numpy"` y lanza
`NotImplementedError` con cualquier otro valor). Los contrafactuales
arancelarios en GPU —jacobianos por lotes, continuación homotópica y los
solucionadores opcionales `torch` / `mlx` de `puremacro.trade.gpu`— son una
superficie aparte documentada en [`docs/es/trade_gpu.md`](trade_gpu.md).

---

## 3. Calibración canónica y especificaciones espaciales

### 1. Caliendo-Parro: economía de 3 países y 2 sectores
- Países: tres economías estilizadas `HOM`, `FOR` y `ROW` con cuotas comerciales fijadas a mano (sin datos).
- Sectores: manufacturas (comerciables, $\theta = 5.0$) y servicios (no comerciables, $\theta = 4.0$).
- Participación del valor agregado: $\gamma_n^j = 0.50$; participación insumo-producto: $\gamma_n^{j, k} = 0.25$.
- Aranceles iniciales: libre comercio en el equilibrio base, $\tau_{ni}^j = 1.0$.

### 2. Allen-Arkolakis: Geografía urbana de 4 ciudades en EE. UU.
- Ciudades: Nueva York, Los Ángeles, Chicago, Houston (con coordenadas geográficas reales de latitud y longitud).
- Elasticidad comercial: $\theta = 4.0$.
- Elasticidad de aglomeración: $\alpha = 0.08$; elasticidad de congestión: $\beta = -0.35$.
- Costos de transporte iceberg: parametrizados mediante distancias ortodrómicas (*Haversine*): $\tau_{ij} = 1.0 + \delta \cdot d_{ij}^\gamma$.

---

## 4. Ejemplos prácticos ejecutables

Los siguientes scripts simulan un choque arancelario bilateral del $25\%$ en Caliendo-Parro y evalúan una inversión en infraestructura de transporte en Allen-Arkolakis:

```python
import numpy as np
from puremacro.trade import CaliendoParroModel

# 3-Country, 2-Sector Caliendo-Parro Trade Economy with Tariffs (hand-set shares)
N, J = 3, 2
trade_shares = np.zeros((J, N, N))
trade_shares[0] = np.array([
    [0.60, 0.20, 0.20],
    [0.20, 0.60, 0.20],
    [0.20, 0.20, 0.60],
])
trade_shares[1] = np.eye(N)  # Non-tradables

cp_model = CaliendoParroModel(
    trade_shares=trade_shares,
    gamma_va=np.full((N, J), 0.50),
    gamma_io=np.full((N, J, J), 0.25),
    alpha=np.full((N, J), 0.50),
    theta=np.array([5.0, 4.0]),
    labor_income=np.array([100.0, 100.0, 100.0]),
    nontradables=[1],
    country_codes=["HOM", "FOR", "ROW"],
    sector_codes=["Manufactures", "Services"],
)

# Simulate a 25% tariff by HOM on manufactures from FOR
cp_res = cp_model.simulate_tariff_shock("HOM", "FOR", sector="Manufactures", tariff_rate=0.25)
assert cp_res.converged
assert len(cp_res.welfare_pct) == 3
print(f"Caliendo-Parro HOM welfare change: {cp_res.welfare_pct[0]:+.2f}%")
```

```python
import numpy as np
from puremacro.spatial import AllenArkolakisModel

# 4-City Allen-Arkolakis Spatial GE Model with Labor Mobility
coords = np.array([
    [40.7128, -74.0060],  # NYC
    [34.0522, -118.2437], # LA
    [41.8781, -87.6298],  # Chicago
    [29.7604, -95.3698],  # Houston
])
names = ["NYC", "LA", "CHI", "HOU"]

aa_model = AllenArkolakisModel.from_coordinates(
    coords,
    region_names=names,
    theta=4.0,
    alpha=0.08,
    beta=-0.35,
    total_population=100.0,
)
aa_res = aa_model.solve_equilibrium()
assert aa_res.converged
assert np.isclose(np.sum(aa_res.population), 100.0)

# Simulate Regional Transport Infrastructure Shock (20% bilateral cost reduction between NYC and Chicago)
infra_res = aa_model.simulate_infrastructure_shock("NYC", "CHI", cost_reduction=0.20)
assert infra_res.converged

print(f"Allen-Arkolakis population in NYC: {aa_res.population[0]:.2f}, LA: {aa_res.population[1]:.2f}")
print(f"Infrastructure welfare change: {infra_res.welfare_pct:+.2f}%")
```

---

## 5. Especificación completa de la API

```text
CaliendoParroModel(
    trade_shares: np.ndarray,
    gamma_va: np.ndarray,
    gamma_io: np.ndarray,
    alpha: np.ndarray,
    theta: np.ndarray,
    labor_income: np.ndarray,
    deficits: np.ndarray | None = None,
    tariffs: np.ndarray | None = None,
    country_codes: Sequence[str] | None = None,
    sector_codes: Sequence[str] | None = None,
    nontradables: Sequence[int | str] | None = None,
)

CaliendoParroModel.solve_counterfactual(
    tariffs_new: np.ndarray | None = None,
    tau_hat: np.ndarray | None = None,
    d_hat: np.ndarray | None = None,
    deficits_new: np.ndarray | None = None,
    deficit_rule: str = "fixed",
    tol: float = 1e-8,
    max_iter: int = 1500,
    damping: float = 0.35,
) -> CaliendoParroResult

CaliendoParroModel.simulate_tariff_shock(
    importer: str | int,
    exporter: str | int,
    sector: str | int | None = None,
    tariff_rate: float = 0.10,
    **kwargs: Any,
) -> CaliendoParroResult

CaliendoParroModel.simulate_trade_war(
    coalition_a: Sequence[str | int],
    coalition_b: Sequence[str | int],
    tariff_rate: float = 0.25,
    **kwargs: Any,
) -> CaliendoParroResult

AllenArkolakisModel(
    trade_costs: np.ndarray,
    fundamental_productivity: np.ndarray,
    fundamental_amenity: np.ndarray,
    theta: float = 4.0,
    alpha: float = 0.10,
    beta: float = -0.30,
    total_population: float = 1.0,
    region_names: Sequence[str] | None = None,
    coordinates: np.ndarray | None = None,
)

AllenArkolakisModel.from_coordinates(
    coords: Any,
    fundamental_productivity: np.ndarray | None = None,
    fundamental_amenity: np.ndarray | None = None,
    distance_cost_param: float = 0.001,
    distance_exponent: float = 0.5,
    metric: str = "haversine",
    theta: float = 4.0,
    alpha: float = 0.10,
    beta: float = -0.30,
    total_population: float = 1.0,
    region_names: Sequence[str] | None = None,
) -> AllenArkolakisModel

AllenArkolakisModel.solve_equilibrium(
    tol: float = 1e-8,
    max_iter: int = 2500,
    damping: float = 0.35,
    backend: str = "numpy",          # 'numpy' | 'mlx' | 'cupy'
) -> AllenArkolakisResult

AllenArkolakisModel.solve_counterfactual(
    trade_costs_new: np.ndarray | None = None,
    productivity_new: np.ndarray | None = None,
    amenity_new: np.ndarray | None = None,
    tol: float = 1e-8,
    max_iter: int = 2500,
    damping: float = 0.35,
    backend: str = "numpy",
) -> AllenArkolakisResult

AllenArkolakisModel.simulate_infrastructure_shock(
    origin: str | int,
    destination: str | int,
    cost_reduction: float = 0.20,
    **kwargs: Any,
) -> AllenArkolakisResult

AllenArkolakisModel.simulate_climate_shock(
    productivity_shocks: Mapping[str | int, float] | np.ndarray | None = None,
    amenity_shocks: Mapping[str | int, float] | np.ndarray | None = None,
    **kwargs: Any,
) -> AllenArkolakisResult
```

---

## 6. Interfaz de resultados y contrafactuales de política

### `CaliendoParroResult`
- `res.w_hat`: Variación bruta de los salarios nominales $\hat{w}_n$.
- `res.P_index_hat`: Variación bruta de los índices de precios al consumidor $\hat{P}_n$.
- `res.real_wage_hat`: Variación bruta de los salarios reales $\hat{w}_n / \hat{P}_n$.
- `res.welfare_pct`: Variación porcentual del bienestar nacional $(\widehat{\mathcal{W}}_n - 1) \times 100$.
- `res.tariff_revenue_prime`: Recaudación arancelaria contrafactual por país.
- `res.summary() -> pd.DataFrame`: Resumen por país de bienestar, salarios, precios y recaudación.
- `res.sector_summary() -> pd.DataFrame`: Desglose por país y sector de producciones brutas y precios.
- `res.to_frame()`, `res.to_markdown()`, `res.to_latex()`, `res.to_typst()`: Formatos de exportación para publicación.
- `res.plot(kind="welfare") -> matplotlib.figure.Figure`: Gráfico de barras de los efectos sobre el bienestar por país.

### `AllenArkolakisResult`
- `res.wages`: Salarios nominales regionales de equilibrio $w_i$.
- `res.population`: Distribución espacial de la población $L_i$, conservando estrictamente $\sum L_i = \bar{L}$.
- `res.price_index`: Índice de precios CES espacial $P_i$.
- `res.real_wages`: Salarios reales regionales $w_i / P_i$.
- `res.welfare`: Nivel de utilidad espacial ecualizada $\bar{u}$.
- `res.consumer_market_access`: Índice de acceso a mercado de los consumidores $\text{CMA}_i$.
- `res.firm_market_access`: Índice de acceso a mercado de exportación de las empresas $\text{FMA}_i$.
- `res.welfare_pct`: Cambio porcentual contrafactual en la utilidad agregada $(\bar{u}' / \bar{u} - 1) \times 100$.
- `res.summary() -> pd.DataFrame`: Tabla resumen con resultados regionales.
- `res.plot(kind="spatial") -> matplotlib.figure.Figure`: Diagrama de dispersión geográfico escalado por población y coloreado por salarios reales.

---

## 7. Datos de calibración incluidos y atribución

El paquete incluye una tabla Inter-País de Insumo-Producto de la OCDE, de modo
que `load_icio_data()` no necesita ningún archivo fuera de su instalación: las
tablas ICIO de la OCDE, edición 2023, año de referencia 2019, en millones de
USD corrientes, agregadas a los 19 países miembros del G20 más `ROW` (las demás
economías de las tablas de la OCDE) y a las 20 secciones de la CIIU Rev.4 de la
`A` a la `T`, con los usos finales `C` (consumo final), `I` (formación bruta de
capital) y `Cx` (compras directas de residentes en el exterior). La agrupación
es la estructura publicada de secciones de la CIIU y la regla de países se
declara en la nota de datos; el valor agregado se reparte 2/3 trabajo y 1/3
capital, porque las tablas de la OCDE no lo separan.

```python
from puremacro.trade import calibrate_trade_model
from puremacro.trade.data import load_icio_data

icio = load_icio_data(return_structured=True)      # the bundled 2019 table
print(icio.matrix.shape, icio.country_codes[:3], icio.sector_codes[:3])   # (403, 460)
calib = calibrate_trade_model(icio.matrix, country_codes=icio.country_codes,
                              sector_codes=icio.sector_codes)
print(calib.nc, calib.ns)                          # 20 countries, 20 sectors
```

La tabla es una adaptación de datos de la OCDE, redistribuida según las
condiciones de uso de la OCDE, que permiten su reutilización y adaptación con
atribución. **Si publica resultados calculados con ella, cite a la OCDE:
"OECD (2023), OECD Inter-Country Input-Output Tables, http://oe.cd/icio"**, no a
puremacro; la OCDE no está afiliada a este proyecto ni lo respalda, y las
opiniones expresadas en trabajos basados en esta adaptación no deben
presentarse como la posición oficial de la OCDE ni de sus países miembros. La
licencia MIT de puremacro cubre su propio código, no estos datos de terceros.

La tabla se reconstruye byte a byte a partir de la edición pública con
`python tools/build_icio_isic.py --download`, que descarga el archivo de la
OCDE en `$PUREMACRO_OECD_ICIO_DIR` (por defecto `~/.cache/puremacro/oecd_icio`)
y rechaza un archivo fuente cuyo MD5 no sea el registrado; una copia que ya
esté en disco se puede pasar con `--source`. Otros años, ediciones o
agrupaciones pasan por los lectores nativos de
[`puremacro.trade.mrio`](trade_mrio.md) y de `puremacro.trade.data`
(`load_oecd_icio_granular`, `load_figaro`, ...). Para resolver tablas de tamaño
completo use `solve_trade_equilibrium(calib, method="equilibrated_newton")`,
que aplica Newton sobre el jacobiano equilibrado por filas y columnas y, si
falla, nombra en `metadata` el país cuya dirección de nivel de precios se ha
vuelto singular. La nota completa, con los resúmenes del archivo fuente y del
archivo incluido, la concordancia y el registro de construcción, está en
`puremacro/trade/_datafiles/SOURCES.md`, que se instala con el paquete.

---

## Referencias bibliográficas

- Allen, T., & Arkolakis, C. (2014). "Trade and the Topography of the Spatial Economy." *The Quarterly Journal of Economics*, 129(3), 1085–1140.
- Caliendo, L., & Parro, F. (2015). "Estimates of the Trade and Welfare Effects of NAFTA." *The Review of Economic Studies*, 82(1), 1–44.
- Dekle, R., Eaton, J., & Kortum, S. (2007). "Unbalanced Trade." *American Economic Review*, 97(2), 351–355.
- Eaton, J., & Kortum, S. (2002). "Technology, Geography, and Trade." *Econometrica*, 70(5), 1741–1779.
- OECD (2023). *OECD Inter-Country Input-Output Tables*. http://oe.cd/icio
- United Nations (2008). *International Standard Industrial Classification of All Economic Activities (ISIC), Revision 4*. Statistical Papers, Series M, No. 4/Rev.4.
