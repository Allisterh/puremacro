> 🇬🇧 [English](../monetary_transmission.md) · 🇪🇸 Español

# Simulador de transmisión monetaria: HANK frente a RANK

`puremacro.models.MonetaryTransmissionSimulator` compara lado a lado modelos **Neokeynesianos de Agentes Heterogéneos (HANK)** y de **Agente Representativo (RANK)**, implementando la descomposición de canales directo e indirecto de **Kaplan, Moll y Violante (2018, *American Economic Review*)** a través de la Propensión Marginal a Consumir (PMC) por deciles de riqueza. Se apoya en `solve_hank_sequence_space` ([HANK en el espacio de secuencias](models.md)) y devuelve contenedores inmutables (*frozen dataclasses*, `MonetaryTransmissionResult`) con una suite de exportación multiformato (`.summary()`, `.plot()`, `.to_markdown()`, `.to_latex()`, `.to_typst()`). El sitio del curso incluye un laboratorio en el navegador para la misma comparación (sección 3).

---

## 1. Simulador de Transmisión Monetaria y Macroprudencial (`MonetaryTransmissionSimulator`)

### 1.1 Marco Teórico: HANK frente a RANK

Los modelos Neokeynesianos de Agente Representativo (RANK) tradicionales asumen que la política monetaria opera predominantemente mediante el **canal directo de sustitución intertemporal**: hogares idénticos sin restricciones de liquidez ajustan su consumo a lo largo de una única ecuación de Euler ($c_t = \mathbb{E}_t c_{t+1} - \frac{1}{\gamma} (i_t - \mathbb{E}_t \pi_{t+1})$).

Por el contrario, los modelos HANK (Kaplan, Moll y Violante 2018; Auclert et al. 2021) incorporan riesgo idiosincrásico no asegurable y restricciones crediticias ($a' \ge 0$). Una porción sustancial de los hogares vive al día (*hand-to-mouth*), registrando elevadas Propensiones Marginales a Consumir (PMC) respecto al ingreso corriente.

En consecuencia, en el modelo HANK cuantitativo de Kaplan, Moll y Violante (2018), el **canal indirecto de equilibrio general**—que se transmite a través de la demanda de trabajo, el producto agregado y los salarios netos—explica la mayor parte de la respuesta macroeconómica del consumo.

### 1.2 Descomposición de Kaplan-Moll-Violante (2018) en el Espacio de Secuencias

A través del álgebra de matrices jacobianas en el espacio de secuencias, el vector de respuesta del consumo agregado $d\mathbf{C} \in \mathbb{R}^T$ en un horizonte de $T$ trimestres se descompone en:

$$d\mathbf{C} = \underbrace{\mathbf{J}^{C, r} \, d\mathbf{r}}_{\text{Canal Directo (Sustitución)}} + \underbrace{\mathbf{J}^{C, Y} \, d\mathbf{Y}}_{\text{Canal Indirecto (Ingreso de Equilibrio General)}}$$

donde $\mathbf{J}^{C, r} \equiv \frac{\partial \mathbf{C}}{\partial \mathbf{r}}$ y $\mathbf{J}^{C, Y} \equiv \frac{\partial \mathbf{C}}{\partial \mathbf{Y}}$ son matrices cuadradas $T \times T$.

En equilibrio general, la curva de Phillips Neokeynesiana y la regla de Taylor en el espacio de secuencias determinan:
$$d\boldsymbol{\pi} = \mathbf{K}_\pi d\mathbf{Y}, \quad d\mathbf{r} = \mathbf{M}_{r, Y} d\mathbf{Y} + d\boldsymbol{\epsilon}$$

Sustituyendo en el bloque de hogares se obtiene el sistema lineal de equilibrio general:
$$\left( \mathbf{I} - \mathbf{J}^{C, Y} - \mathbf{J}^{C, r} \mathbf{M}_{r, Y} \right) d\mathbf{Y} = \mathbf{J}^{C, r} d\boldsymbol{\epsilon}$$

| Dimensión | Economía RANK | Economía HANK |
|---|---|---|
| **Estructura de Mercados** | Mercados de activos completos | Mercados incompletos, restricción crediticia $a' \ge 0$ |
| **Distribución de la PMC** | Uniforme en deciles: $1 - \beta \approx 0.015$ | Pendiente pronunciada: Decil 1 ($>0.40$) al Decil 10 ($<0.06$) |
| **Participación Canal Directo** | Idénticamente $100\%$ | $20\% - 40\%$ en Kaplan, Moll y Violante (2018); $85.0\%$ en este simulador (Receta 1) |
| **Participación Canal Indirecto** | Idénticamente $0\%$ | $60\% - 80\%$ en Kaplan, Moll y Violante (2018); $15.0\%$ en este simulador (Receta 1) |
| **Núcleo de Propagación** | Ecuación de Euler intertemporal | Multiplicador de ingreso laboral en equilibrio general |

Las participaciones de la literatura son las de Kaplan, Moll y Violante (2018): los efectos indirectos explican el 80 por ciento de la respuesta del consumo durante el primer año en su caso base y el 60 por ciento cuando la deuda pública absorbe la mayor parte del desequilibrio fiscal en el corto plazo (sus cuadros 7 y 8). No son resultados de este simulador. En la Receta 1, `MonetaryTransmissionSimulator` imprime `indirect_share_hank=15.0%`, la participación del canal indirecto en la respuesta del consumo en la fecha 0, de modo que el canal directo explica el 85.0 por ciento restante: la calibración estilizada del simulador da una participación indirecta menor que el modelo HANK cuantitativo de ese artículo.

### 1.3 Recetas Prácticas en Python

#### Receta 1: Alza de Tasa de Política Monetaria (25 bps)
```python
from puremacro.models import MonetaryTransmissionSimulator

# Instanciar simulador con parámetros de malla continua
sim_mon = MonetaryTransmissionSimulator(
    beta=0.985,
    gamma=1.0,
    r_ss=0.01,
    phi_pi=1.5,
    kappa=0.1,
    n_a=50,
    a_max=30.0,
)

# PMC agregada en estado estacionario
print(f"PMC Agregada HANK: {sim_mon.steady_state_mpc:.4f}")

# Simular alza de 25 bps con persistencia rho=0.7 en 40 trimestres
res_mon = sim_mon.simulate_rate_shock(magnitude=0.0025, rho=0.7, T=40)

print(repr(res_mon))
# Salida: MonetaryTransmissionResult(shock_type='rate', magnitude=+0.0025, T=40, mpc_hank=0.1027, mpc_rank=0.0150, indirect_share_hank=15.0%)

# Resumen comparativo en texto
print(res_mon.summary())

# Respuestas pico
picos = res_mon.peak_responses()
print(f"Caída máxima de producto en HANK: {picos['output_hank']*100:.3f}%")
print(f"Caída máxima de producto en RANK: {picos['output_rank']*100:.3f}%")

# Gráfica de 4 paneles para publicación (FIR, escalera de PMC, canales KMV)
fig = res_mon.plot(kind="all", figsize=(12.0, 8.5))
fig.savefig("transmision_monetaria_hank_rank.png", dpi=300)
```

#### Receta 2: Transferencias Focalizadas e Intervenciones de Hoja de Balance
```python
# Simular transferencia monetaria focalizada a hogares con restricciones crediticias
res_tf = sim_mon.simulate_balance_sheet_intervention(
    amount=1.0,
    target="borrowers",   # 25% más pobre de los hogares
    T=40,
)

print(f"Multiplicador inicial de consumo en HANK: {res_tf.irf_consumption_hank[0]:.4f}")
print(f"Multiplicador inicial de consumo en RANK: {res_tf.irf_consumption_rank[0]:.4f}")
# Salida: Multiplicador inicial de consumo en HANK: 0.3163
#         Multiplicador inicial de consumo en RANK: 0.0150
# En HANK, los hogares con alta PMC gastan más en el impacto (unas 21 veces la respuesta en RANK)
```

---

## 2. Referencia Completa de la API

### 2.1 `MonetaryTransmissionSimulator`

| Método / Propiedad | Firma | Descripción |
|---|---|---|
| `steady_state_mpc` | `float` *(propiedad)* | PMC agregada de estado estacionario en la economía HANK. |
| `simulate_rate_shock()` | `magnitude=0.0025, rho=0.7, T=40` | Resolver equilibrio en espacio de secuencias HANK vs. RANK ante choque de tasas. |
| `simulate_balance_sheet_intervention()` | `amount=1.0, target='borrowers', T=40` | Simular transferencia fiscal focalizada o inyección de liquidez. |
| `simulate_transmission()` | `shock_type='rate', shock_path=None, ...` | Punto de entrada general que admite trayectorias exógenas arbitrarias. |

### 2.2 `MonetaryTransmissionResult`

| Atributo / Método | Tipo / Firma | Descripción |
|---|---|---|
| `irf_output_hank / rank` | `np.ndarray (T,)` | Trayectorias de función de impulso-respuesta del producto $d\mathbf{Y}$. |
| `irf_consumption_hank / rank` | `np.ndarray (T,)` | Trayectorias de impulso-respuesta del consumo $d\mathbf{C}$. |
| `irf_inflation_hank / rank` | `np.ndarray (T,)` | Trayectorias de impulso-respuesta de la inflación trimestral $d\boldsymbol{\pi}$ (`plot()` las muestra anualizadas, $\times 400$). |
| `irf_rate_hank / rank` | `np.ndarray (T,)` | Trayectorias de la tasa de interés real *ex-ante* $d\mathbf{r}$. |
| `mpc_deciles_hank / rank` | `pd.Series (10,)` | Propensión marginal a consumir en 10 deciles de riqueza. |
| `aggregate_mpc_hank / rank` | `float` | Propensión marginal a consumir agregada trimestral. |
| `direct_channel_hank` | `np.ndarray (T,)` | Canal directo de sustitución intertemporal KMV $\mathbf{J}^{C, r} d\mathbf{r}$. |
| `indirect_channel_hank` | `np.ndarray (T,)` | Canal indirecto de ingreso laboral KMV $\mathbf{J}^{C, Y} d\mathbf{Y}$. |
| `indirect_share_hank` | `float` | Porcentaje de la respuesta inicial de consumo explicada por el canal indirecto. |
| `peak_responses()` | `-> dict[str, float]` | Respuestas pico de producto, inflación y consumo. |
| `summary()` | `-> str` | Reporte en texto ASCII comparando los mecanismos HANK y RANK. |
| `to_frame()` | `-> pd.DataFrame` | DataFrame ordenado de trayectorias trimestrales indexado por $t \in [0, T-1]$. |
| `plot(kind)` | `kind='all' \| 'irf' \| 'mpc' \| 'kmv'` | Gráficas comparativas de transmisión HANK vs. RANK. |

---

## 3. Laboratorio Interactivo en el Navegador con JavaScript (`curso/site/labs/`)

El sitio del curso en `curso/site/labs/` incluye un laboratorio en el navegador para este simulador: HTML y JavaScript estáticos, sin pasos de compilación.

### 3.1 `politica-monetaria-hank.html` & `.js` (Transmisión Monetaria HANK vs. RANK)

- **Ubicación**: `curso/site/labs/politica-monetaria-hank.html` y `curso/site/labs/politica-monetaria-hank.js`.
- **Características**:
  * Un preajuste de escenario de política (alza de 100 pb, recorte de 100 pb, choque persistente con $\rho = 0.85$, inyección de liquidez) y seis controles deslizantes: el choque de tasa de interés (de −200 a +200 pb, anualizado), su persistencia $\rho$, la transferencia a hogares restringidos (% del PIB), la respuesta de Taylor $\phi_\pi$, la pendiente de Phillips $\kappa$ y la aversión al riesgo $\gamma$.
  * Gráficas de líneas de las funciones impulso-respuesta del producto y de la inflación, HANK frente a RANK.
  * Una gráfica de líneas de la PMC trimestral por decil de riqueza: la escalera descendente de HANK frente a la línea plana de RANK en $1 - \beta$.
  * La descomposición de Kaplan-Moll-Violante del consumo de HANK en sus canales directo e indirecto, y recuadros con la PMC agregada, la caída del producto en el impacto, la participación del canal indirecto y el multiplicador de balance.
  * Los jacobianos de los hogares HANK y las PMC por decil ($T = 40$) son arreglos fijos en el script; mover un control vuelve a resolver en el navegador el sistema lineal de equilibrio general.

### 3.2 Ejecución Local

Para servir los laboratorios localmente (cada página carga d3 7.9.0 y KaTeX 0.16.11, con su extensión auto-render, desde la CDN jsDelivr, y la fuente Manrope desde Google Fonts, así que las gráficas necesitan acceso a la red):

```bash
# Desde la raíz del repositorio puremacro
cd curso/site
python3 -m http.server 8000
```
Abra su navegador en `http://localhost:8000/labs/comercio-aranceles.html` o `http://localhost:8000/labs/politica-monetaria-hank.html`.

---

## Referencias

1. **Auclert, A., Bardóczy, B., Rognlie, M., y Straub, L. (2021).** "Using the Sequence-Space Jacobian to Solve and Estimate Heterogeneous-Agent Models." *Econometrica*, 89(6), 3115–3148.
2. **Kaplan, G., Moll, B., y Violante, G. L. (2018).** "Monetary Policy According to HANK." *American Economic Review*, 108(3), 697–743.
