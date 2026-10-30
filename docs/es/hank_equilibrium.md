> 🇬🇧 [English](../hank_equilibrium.md) · 🇪🇸 Español

# Equilibrio HANK publicado de un activo

`calibrate_one_asset_hank` y `solve_one_asset_hank` implementan el modelo del
[cuaderno HANK de los autores](https://github.com/shade-econ/sequence-jacobian/blob/eb4ac4c3157357a4cacbea4b5a7a281f5b48cab9/notebooks/hank.ipynb),
asociado con [Auclert, Bardóczy, Rognlie y Straub (2021)](https://doi.org/10.3982/ECTA17434).
La implementación utiliza NumPy/SciPy. El paquete de los autores sólo se usa
en un entorno separado para generar las referencias independientes.

```python
from puremacro.models import calibrate_one_asset_hank, solve_one_asset_hank

steady = calibrate_one_asset_hank()  # 500 nodos de activos, 7 estados de ingreso
linear = solve_one_asset_hank(steady, horizon=300)
nonlinear = solve_one_asset_hank(steady, horizon=300, nonlinear=True)
assert steady.converged and linear.converged and nonlinear.converged
print(steady.summary())
print(nonlinear.diagnostics)
print(nonlinear.to_dataframe().head())
```

El choque reduce el intercepto de la regla de Taylor en `0.0025 * 0.61**t`.
`magnitude=0.0025` aplica el choque opuesto. Las trayectorias son **desviaciones
en niveles**, con tasas e inflación en unidades decimales. Una respuesta del
consumo de `0.001` equivale a 0.1% del consumo estacionario, normalizado a uno.
Los resultados ofrecen `summary`, `to_markdown`, `to_latex` y `to_typst`.

## Ecuaciones y fechas

Los hogares eligen consumo, horas y ahorro:

$$c_t+a_{t+1}=(1+r_t)a_t+w_t e_t n_t+(Div_t-Tax_t)e_t,\quad a_{t+1}\ge0.$$

La utilidad marginal es $c^{-1/\mathrm{EIS}}$ y la condición laboral es
$v_\phi n^{1/\mathrm{Frisch}}=wec^{-1/\mathrm{EIS}}$. EGM resuelve el problema
interior; en el límite de endeudamiento se resuelven conjuntamente el
presupuesto y la condición laboral. Dividendos y transferencias son
proporcionales a la habilidad, de media estacionaria uno.

La producción y productividad estacionarias son uno. Las empresas contratan
$L_t=Y_t$, distribuyen $Div_t=Y_t-w_tL_t-Adj_t$ y pagan

$$Adj_t=\frac{\mu}{\mu-1}\frac{\log(1+\pi_t)^2}{2\kappa}Y_t.$$

La deuda pública es constante, $B=5.6$, y $Tax_t=r_tB$. El rendimiento realizado
incluye la revaluación inicial de la deuda nominal emitida previamente:

$$1+r_t=\frac{1+r^\star_{t-1}+\phi\pi_{t-1}}{1+\pi_t}.$$

Se resuelven $(\pi_t,w_t,Y_t)$ mediante $A_t=B$, $NE_t=L_t$ y

$$0=\kappa(w_t-1/\mu)+\frac{Y_{t+1}}{Y_t}
\frac{\log(1+\pi_{t+1})}{1+r_{t+1}}-\log(1+\pi_t).$$

El mercado de bienes $Y_t=C_t+Adj_t$ y los presupuestos individuales se
comprueban por separado. La política nominal inicial está predeterminada.
El valor terminal del hogar y los agregados futuros son estacionarios;
refinar el horizonte mide el efecto de esta condición terminal.

## Calibración y comprobación independiente

Los valores iniciales son EIS = Frisch = 0.5, persistencia de log habilidad
0.966, desviación estándar incondicional 0.5, margen 1.2, coeficiente de Phillips
0.1, coeficiente de Taylor 1.5 y tasa trimestral estacionaria 0.005. Se calibran
beta y la desutilidad laboral para obtener activos 5.6 y trabajo efectivo uno.
Cumplir esos objetivos no demuestra por sí solo convergencia numérica.

La referencia de los autores compara ahorro, consumo, horas, distribución,
parámetros calibrados y nueve trayectorias agregadas. La tolerancia lineal es
`1e-6` por unidad del choque monetario; la no lineal, para ambos signos, es
`1e-7` en niveles. Los residuos de mercado deben quedar por debajo de `1e-8`.
El generador verifica los archivos instalados contra la revisión fijada y
registra hashes y versiones de dependencias.

```sh
python -m pytest tests/test_models/test_one_asset_hank.py -q
python -m tools.reference_validation.check_hank_equilibrium \
  --output /tmp/hank-equilibrium --convergence
```

El protocolo compara los primeros 40 trimestres: permite 1% del máximo de la
respuesta más fina, con un piso absoluto de `1e-6`; los cambios en beta y la
desutilidad laboral deben ser menores que `1e-4`. Se refinan por separado
activos, estados de ingreso, límite superior y horizonte. Los ingresos más
extremos requieren ampliar el límite de activos de 150 utilizado en el cuaderno.

El registro `tests/fixtures/hank_equilibrium/convergence.json` contiene:

| Comprobación | Resultado |
|---|---|
| Calibración publicada de 500 × 7 | beta 0.9822435541; desutilidad laboral 0.7864334303 |
| Diferencia máxima de trayectorias lineales frente a los autores | 9.23e-14 |
| Diferencia máxima no lineal, ambos signos | 9.67e-10 |
| Activos, 500 → 1,000 nodos | Pasa |
| Ingreso, 11 → 15 y 15 → 21 estados | No pasa; se conserva en el registro |
| Ingreso, 21 → 31 estados | Pasa; mayor cambio de respuesta 0.614% de su máximo |
| Activos, 500 → 1,000, con 31 estados de ingreso | Pasa |
| Horizonte, 300 → 600, con 31 estados de ingreso | Pasa |

La malla ampliada de ingreso utiliza activos hasta 600. La comparación de
límites 300 → 600 con 15 estados pasa; la mayor masa en el extremo superior
de todas las transiciones es 1.09e-29. La calibración original reproduce la
discretización de los autores; la ampliada cumple el criterio de refinamiento
declarado. Esto no certifica cada malla intermedia de ingreso.

Se valida una transición monetaria determinista con previsión perfecta alrededor
de un estado estacionario publicado. Esto no demuestra precisión para choques
grandes arbitrarios, un límite inferior vinculante de tasas, dos activos o
bienestar con riesgo agregado. `solve_hank_sequence_space` y los ejemplos DAG
existentes utilizan otros cierres; sus resultados no son esta referencia.
