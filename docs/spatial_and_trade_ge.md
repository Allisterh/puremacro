> 🇬🇧 English · 🇪🇸 [Español](es/spatial_and_trade_ge.md)

# Quantitative Spatial General Equilibrium

`puremacro.spatial` implements the continuous geographic spatial equilibrium model with labor mobility, agglomeration externalities, and congestion forces developed by **Allen and Arkolakis (2014, *Quarterly Journal of Economics*)**.

The model conforms strictly to the pure Python / Pyodide runtime contract (zero C++ dependencies, pure NumPy/SciPy/Pandas/Matplotlib), enabling rapid counterfactual policy evaluations—such as regional transport infrastructure investments and localized environmental shocks.

---

## 1. Theoretical & Algorithmic Framework

### 1.1 Allen-Arkolakis (2014) Geographic Gravity Model with Labor Mobility

Allen and Arkolakis (2014) formulate a continuous spatial general equilibrium model over $N$ geographical regions where workers freely migrate until real utility is equalized everywhere.

#### Trade Flows and Market Access
Bilateral trade between origin $i$ and destination $j$ satisfies gravity:

$$\pi_{ij} = \frac{\left( \frac{w_i \tau_{ij}}{A_i} \right)^{-\theta}}{\sum_{k=1}^N \left( \frac{w_k \tau_{kj}}{A_k} \right)^{-\theta}} = \frac{\left( \frac{w_i \tau_{ij}}{A_i} \right)^{-\theta}}{P_j^{-\theta}}$$

where $\tau_{ij} \ge 1$ are iceberg trade costs ($\tau_{ii} = 1$), $\theta = \sigma - 1$ is the trade elasticity, and $P_j$ is the Consumer Market Access index:

$$P_j^{-\theta} = \text{CMA}_j = \sum_{i=1}^N \tau_{ij}^{-\theta} \left( \frac{w_i}{A_i} \right)^{-\theta}$$

Firm Market Access ($\text{FMA}_i$) captures export demand:

$$\text{FMA}_i = \sum_{j=1}^N \tau_{ij}^{-\theta} P_j^\theta w_j L_j$$

Goods market clearing equates regional labor income to total sales:

$$w_i L_i = \sum_{j=1}^N \pi_{ij} w_j L_j = \left( \frac{w_i}{A_i} \right)^{-\theta} \text{FMA}_i$$

### 1.2 Agglomeration, Congestion, and Spatial Equilibrium

Worker utility in location $i$ depends on real wages, local amenities $a_i$, and endogenous population $L_i$:

$$u_i = a_i \frac{w_i}{P_i}$$

Endogenous productivity and amenities incorporate spillover forces:
- **Productivity Agglomeration**: $A_i = \bar{A}_i L_i^\alpha$, where $\alpha > 0$ reflects Marshallian external scale economies.
- **Amenity Congestion**: $a_i = \bar{a}_i L_i^\beta$, where $\beta < 0$ captures housing supply inelasticity, commuting congestion, and pollution.

#### Spatial Utility Equalization
Free labor mobility equalizes utility across all populated regions:

$$u_i = \bar{u}, \quad \forall i \quad \implies \quad L_i = \bar{L} \frac{\left( \frac{w_i \bar{a}_i}{P_i} \right)^{-\frac{1}{\beta}}}{\sum_k \left( \frac{w_k \bar{a}_k}{P_k} \right)^{-\frac{1}{\beta}}}$$

#### Uniqueness Theorem
Allen and Arkolakis (2014, Theorem 2) prove that the spatial general equilibrium is **unique and strictly stable** if and only if congestion forces dominate agglomeration spillovers:

$$\alpha + \beta \le \frac{\theta}{1 + \theta} \quad \text{and} \quad \alpha \le \frac{1}{\theta}$$

`AllenArkolakisModel.is_unique` verifies this condition automatically during initialization.

---

## 2. Methodological & Model Options

| Dimension | Allen-Arkolakis (2014) |
|---|---|
| **Core Economic Mechanism** | Spatial labor mobility with agglomeration & congestion |
| **Solving Approach** | Contraction mapping fixed point on $(w_i, L_i)$ |
| **Endogenous Margins** | Population distribution, wages, market access |
| **Labor Mobility** | Free continuous spatial mobility across regions |
| **Primary Use Cases** | High-speed rail, highway corridors, climate shocks |
| **Convergence Rate** | 20–50 iterations ($< 0.005$ s) |

### 2.1 Hardware backends (`backend=`, 3.4.0)

The fixed-point solver accepts a `backend` keyword — `"numpy"`
(the default), `"mlx"` (Apple Silicon) or `"cupy"` (NVIDIA):

```python
aa_res = aa_model.solve_equilibrium(backend="mlx")
cf_res = aa_model.solve_counterfactual(trade_costs_new=tau_new, backend="mlx")
```

Three properties are worth knowing before you reach for one:

- **NumPy is the oracle.** Every accelerated path is checked against the
  NumPy reference, never the other way round. `numpy` is always
  available and keeps the Pyodide contract intact.
- **`tol` still means `tol`.** Metal has no float64, so the MLX
  contraction in `AllenArkolakisModel` runs in float32 down to a 1e-6
  floor; the result is then polished with the float64 NumPy contraction
  warm-started at the device solution, so `converged` refers to the
  `tol` you asked for.
- **Failure is loud and non-fatal.** A backend that is not installed, an
  array namespace that will not import, or an exception on the device each
  emit a `RuntimeWarning` and fall back to NumPy. An accelerated result
  that did not converge is never returned either: `AllenArkolakisModel`
  finishes the device solution with the float64 NumPy contraction above —
  no warning, because `converged` is then the NumPy verdict at your `tol`.

```python
from puremacro._backend import available_backends
available_backends()          # ('numpy', 'numba', 'mlx') on this machine
```

---

## 3. Canonical Calibration & Spatial Specifications

### 1. Allen-Arkolakis 4-City US Geography
- Cities: New York City, Los Angeles, Chicago, Houston (with actual latitude/longitude coordinates).
- Trade elasticity: $\theta = 4.0$.
- Agglomeration elasticity: $\alpha = 0.08$; congestion elasticity: $\beta = -0.35$.
- Iceberg costs: parametrized via Haversine great-circle distances: $\tau_{ij} = 1.0 + \delta \cdot d_{ij}^\gamma$.

---

## 4. Runnable Worked Examples

The following script evaluates a transport infrastructure investment in Allen-Arkolakis:

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

## 5. Full API Specification

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

## 6. Result Interface & Policy Counterfactuals

### `AllenArkolakisResult`
- `res.wages`: Equilibrium regional nominal wages $w_i$.
- `res.population`: Spatial population distribution $L_i$, strictly conserving $\sum L_i = \bar{L}$.
- `res.price_index`: Spatial CES price index $P_i$.
- `res.real_wages`: Spatial real wages $w_i / P_i$.
- `res.welfare`: Equalized spatial utility level $\bar{u}$.
- `res.consumer_market_access`: Regional consumer market access index $\text{CMA}_i$.
- `res.firm_market_access`: Regional firm export market access index $\text{FMA}_i$.
- `res.welfare_pct`: Counterfactual percentage utility change $(\bar{u}' / \bar{u} - 1) \times 100$.
- `res.summary() -> pd.DataFrame`: Regional summary table.
- `res.plot(kind="spatial") -> matplotlib.figure.Figure`: Geographic scatter plot sized by population and colored by real wages.

---

## References

- Allen, T., & Arkolakis, C. (2014). "Trade and the Topography of the Spatial Economy." *The Quarterly Journal of Economics*, 129(3), 1085–1140.
