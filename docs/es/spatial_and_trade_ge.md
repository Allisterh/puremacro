> 🇬🇧 [English](../spatial_and_trade_ge.md) · 🇪🇸 Español

# Equilibrio general espacial cuantitativo

`puremacro.spatial` implementa el modelo continuo de equilibrio espacial geográfico con movilidad laboral, externalidades de aglomeración y fuerzas de congestión desarrollado por **Allen y Arkolakis (2014, *Quarterly Journal of Economics*)**.

El modelo cumple de forma estricta el contrato de ejecución en Python puro / Pyodide (cero dependencias en C++, basado exclusivamente en NumPy/SciPy/Pandas/Matplotlib), permitiendo evaluaciones contrafactuales de política económica —tales como inversiones regionales en infraestructura de transporte y perturbaciones ambientales localizadas.

---

## 1. Marco teórico y algorítmico

### 1.1 Modelo gravitacional espacial geográfico con movilidad laboral de Allen-Arkolakis (2014)

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

### 1.2 Aglomeración, congestión y equilibrio espacial

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

## 2. Opciones metodológicas y del modelo

| Dimensión | Allen-Arkolakis (2014) |
|---|---|
| **Mecanismo económico central** | Movilidad laboral espacial con aglomeración y congestión |
| **Enfoque de resolución** | Mapeo de contracción de punto fijo sobre $(w_i, L_i)$ |
| **Márgenes endógenos** | Distribución de población, salarios, acceso a mercados |
| **Movilidad laboral** | Libre movilidad espacial continua entre regiones |
| **Casos de uso principales** | Trenes de alta velocidad, corredores carreteros, choques climáticos |
| **Tasa de convergencia** | 20–50 iteraciones ($< 0.005$ s) |

### 2.1 Backends de hardware (`backend=`, 3.4.0)

El solucionador de punto fijo acepta un argumento `backend` — `"numpy"`
(el valor por omisión), `"mlx"` (Apple Silicon) o `"cupy"` (NVIDIA):

```python
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
available_backends()          # ('numpy', 'numba', 'mlx') en esta máquina
```

---

## 3. Calibración canónica y especificaciones espaciales

### 1. Allen-Arkolakis: Geografía urbana de 4 ciudades en EE. UU.
- Ciudades: Nueva York, Los Ángeles, Chicago, Houston (con coordenadas geográficas reales de latitud y longitud).
- Elasticidad comercial: $\theta = 4.0$.
- Elasticidad de aglomeración: $\alpha = 0.08$; elasticidad de congestión: $\beta = -0.35$.
- Costos de transporte iceberg: parametrizados mediante distancias ortodrómicas (*Haversine*): $\tau_{ij} = 1.0 + \delta \cdot d_{ij}^\gamma$.

---

## 4. Ejemplos prácticos ejecutables

El siguiente script evalúa una inversión en infraestructura de transporte en Allen-Arkolakis:

```python
import numpy as np
from puremacro.spatial import AllenArkolakisModel

# Modelo de equilibrio general espacial de 4 ciudades en Allen-Arkolakis
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

# Simular reducción del 20% en costos bilaterales de transporte entre NYC y Chicago
infra_res = aa_model.simulate_infrastructure_shock("NYC", "CHI", cost_reduction=0.20)
assert infra_res.converged

print(f"Allen-Arkolakis: Población en NYC: {aa_res.population[0]:.2f}, LA: {aa_res.population[1]:.2f}")
print(f"Impacto en bienestar por infraestructura: {infra_res.welfare_pct:+.2f}%")
```

---

## 5. Especificación completa de la API

```text
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

## Referencias bibliográficas

- Allen, T., & Arkolakis, C. (2014). "Trade and the Topography of the Spatial Economy." *The Quarterly Journal of Economics*, 129(3), 1085–1140.
