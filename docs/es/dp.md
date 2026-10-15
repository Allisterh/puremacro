> 🇬🇧 [English](../dp.md) · 🇪🇸 Español

# Programación dinámica declarativa (`puremacro.dp`)

`puremacro.dp` hace con los programas dinámicos lo que un archivo `.mod` de
Dynare hace con los modelos DSGE: el modelo se declara como ecuaciones y la
biblioteca escribe la llamada al solucionador. Los estados, los choques, las
decisiones, la recompensa, las restricciones y una condición opcional de vaciado
de mercado son cadenas con la sintaxis de expresiones de Dynare, leídas por el
mismo analizador que los archivos `.mod`. `Model.solve()` las compila a los
motores de `puremacro.vfi`, de modo que nunca hay que escribir una función de
retorno como clausura, expandir mallas a mano ni pasar listas posicionales de
parámetros.

| `solve(method=...)` | motor | qué resuelve |
|---|---|---|
| `"vfi"` (por defecto sin deriva) | `VFIProblem`, `FiniteHorizonProblem` | cualquier modelo en tiempo discreto sobre mallas; varios estados |
| `"egm"` | `vfi.solve_egm`, un paso EGM por edad | un activo con restricción presupuestaria `c + a(+1) = R*a + y(z)`; ciclo de vida; una decisión discreta con o sin choques de preferencia; Krusell-Smith |
| `"hjb"` (por defecto con deriva) | `vfi.solve_hjb_achdou` | tiempo continuo: un estado cuya deriva es lineal en un control |

Con `prices(...)` y `clear(...)`, la solución del hogar queda dentro de una
búsqueda de raíz del precio que vacía el mercado. Con `aggregate_state(...)`
queda dentro del bucle de la regla de pronóstico de Krusell y Smith (1998).

## Una economía de Aiyagari

```python
import numpy as np
from puremacro import dp

m = dp.Model("aiyagari")
m.parameters(beta=0.96, gamma=1.0, alpha=0.36, delta=0.08)
m.prices(r=(0.005, 0.0396))                       # intervalo para la búsqueda de raíz
m.exogenous("z", dp.AR1(rho=0.9, sigma=0.2, n=5)) # Tauchen; method="rouwenhorst"
m.state("a", np.linspace(1e-4, 80.0, 150))
m.local("w = (1 - alpha)*(alpha/(r + delta))^(alpha/(1 - alpha))")
m.local("c = w*exp(z) + (1 + r)*a - a(+1)")
m.reward("crra(c, gamma)")
m.subject_to("c > 0")
m.aggregate(K="a", L="exp(z)")
m.clear("K - L*(alpha/(r + delta))^(1/(1 - alpha))")

sol = m.solve("egm")
print(sol.prices["r"], sol.aggregates["K"])       # alrededor de 0.0305 y 7.49
print(sol.summary())
```

`a` es el valor actual de un estado y `a(+1)` su valor en el periodo siguiente.
Un identificador que no es un estado, un choque ni la decisión discreta se
resuelve, en este orden, como una variable local, un parámetro, un precio, un
agregado (sólo dentro de `clear`) y `age` (sólo en modelos de horizonte finito).
`crra(c, gamma)` es `(c^(1-gamma) - 1)/(1 - gamma)`, y `log(c)` cuando
`gamma = 1`. Cualquier otra función de utilidad se puede escribir como
expresión: EGM obtiene la utilidad marginal de la derivada simbólica y la
invierte numéricamente.

El mismo modelo se resuelve por VFI discreta con `m.solve()` (r alrededor de
0.0303 en esta malla, en unos 8 s frente a cerca de 1 s con EGM). Los dos
resultados difieren porque VFI coloca `a(+1)` sobre los nodos de la malla y
EGM no.

### Leer una solución

`DPSolution` contiene la solución sea cual sea el motor:

- `sol.V`, `sol.distribution`: arreglos de forma (n_estados, n_choques), con un
  eje de edad al frente en un ciclo de vida;
- `sol.policy(expr)`: cualquier expresión del modelo evaluada en la decisión
  óptima, por ejemplo `sol.policy("c")` o `sol.policy("a(+1)")`;
- `sol.mean(expr)`: su media poblacional bajo la distribución estacionaria, un
  valor por edad en un ciclo de vida;
- `sol.prices`, `sol.aggregates`, `sol.equilibrium`, `sol.grids`, `sol.P`;
- `sol.raw`: el resultado del motor subyacente.

```python
c = sol.policy("c")                 # (150, 5)
print(sol.mean("c"), sol.mean("a")) # consumo y activos medios
```

## Ciclo de vida

`horizon(T, survival=..., terminal=..., newborns=...)` vuelve finito el
problema. `age` (de 0 a T-1) puede aparecer en cualquier ecuación, así que los
perfiles de ingreso, la jubilación y los rendimientos que varían con la edad se
escriben en el propio modelo. Una expresión relacional como `(age < T_ret)` vale
1 cuando se cumple y 0 en caso contrario.

```python
T = 60
lc = dp.Model("life cycle")
lc.parameters(beta=0.96, gamma=2.0, r=0.03, T_ret=45)
lc.exogenous("z", dp.AR1(rho=0.95, sigma=0.1, n=5))
lc.state("a", np.linspace(0.0, 30.0, 120))
lc.local("y = (age < T_ret)*exp(z + 0.04*age - 0.0008*age^2) + (age >= T_ret)*0.6")
lc.local("c = (1 + r)*a + y - a(+1)")
lc.reward("crra(c, gamma)")
lc.subject_to("c > 0")
lc.horizon(T, survival=np.full(T, 0.99))

life = lc.solve("egm")
assets = life.mean("a")             # activos medios por edad; máximo al jubilarse
print(int(assets.argmax()))         # 45
```

Cada cohorte tiene masa uno en `life.distribution`, y la supervivencia no se
aplica ahí: solo descuenta el futuro. `life.population_shares()` da la
proporción de la población en cada edad con la supervivencia aplicada, y las
estadísticas que agregan edades la usan (véase
[Paneles y desigualdad](#paneles-y-desigualdad)). Los recién nacidos empiezan
en el punto más bajo de la malla con choques tomados de su distribución
estacionaria, salvo que `newborns=` indique otra cosa.

## Decisiones discretas y choques de preferencia

`discrete("h", values)` añade una decisión discreta que se toma junto con
`a(+1)`. Con `method="egm"` cada opción se invierte con la ecuación de Euler y
una envolvente superior conserva los segmentos óptimos (Iskhakov, Jørgensen,
Rust y Schjerning 2017). Las esquinas en el límite de endeudamiento y en el
extremo superior de la malla sólo se admiten donde se cumple su condición de
Kuhn-Tucker. `taste_shocks(scale)` añade choques de valor extremo tipo I: los
valores pasan a ser log-sumas, las decisiones probabilidades logit, y
`mean("h")` es una tasa de participación.

```python
w = dp.Model("work or not")
w.parameters(beta=0.95, gamma=2.0, r=0.03, chi=0.5, b=0.3, sig=0.05)
w.exogenous("z", dp.AR1(rho=0.9, sigma=0.2, n=5))
w.state("a", np.linspace(0.0, 60.0, 200))
w.discrete("h", [0.0, 1.0])
w.local("c = (1 + r)*a + h*exp(z) + (1 - h)*b - a(+1)")
w.reward("crra(c, gamma) - chi*h")
w.subject_to("c > 0")
w.taste_shocks("sig")
w.aggregate(H="h", A="a")

work = w.solve("egm")
print(work.aggregates["H"])         # tasa de participación, alrededor de 0.63
```

La recompensa y el ingreso pueden depender de la opción; el rendimiento de los
activos no. Los agregados sobre una decisión discreta convergen despacio en el
número de puntos de malla con cualquiera de los dos métodos, porque la masa de
la distribución está en los nodos: compruébelos en una malla más fina antes de
reportarlos.

## Tiempo continuo

Se declaran un `control`, una `drift` (deriva) para el estado y un
`discount_rate`, con `dp.Jump` (un generador de Poisson) para los choques. El
compilador comprueba que la deriva es lineal en el control con pendiente -1 y
deriva u' y su inversa a partir de la recompensa. Los precios y `clear`
funcionan como en tiempo discreto, con los agregados integrados contra la
densidad de Kolmogorov hacia adelante.

```python
ct = dp.Model("continuous-time Aiyagari")
ct.parameters(rho=0.05, gamma=2.0, alpha=0.33, delta=0.05)
ct.prices(r=(0.005, 0.048))
ct.exogenous("e", dp.Jump([0.2, 1.0], [[-0.1, 0.1], [0.1, -0.1]]))
ct.state("a", np.linspace(0.0, 30.0, 100))
ct.local("w = (1 - alpha)*(alpha/(r + delta))^(alpha/(1 - alpha))")
ct.control("c")
ct.drift("a", "r*a + w*e - c")
ct.discount_rate("rho")
ct.reward("crra(c, gamma)")
ct.aggregate(K="a", L="e")
ct.clear("K - L*(alpha/(r + delta))^(1/(1 - alpha))")

hjb = ct.solve()                    # method="hjb" porque el modelo tiene deriva
print(hjb.prices["r"])              # alrededor de 0.0200
```

Varios choques `Jump` se combinan en su cadena producto. `sol.distribution` es
la masa en cada nodo (la densidad por los pesos de cuadratura).

## Riesgo agregado: Krusell y Smith

`aggregate_shock("Z", process)` y `aggregate_state("K", grid, mean="a")`
convierten el modelo en el punto fijo de Krusell y Smith (1998): los hogares
pronostican `K` con una regla log-lineal, se simula la economía y la regla se
reestima hasta que deja de moverse. Los precios y la tecnología son ecuaciones
del modelo, así que sirve cualquier empresa y cualquier recompensa del hogar.

```python
ks = dp.Model("krusell-smith")
ks.parameters(beta=0.99, gamma=1.0, alpha=0.36, delta=0.025, L=0.9)
ks.exogenous("e", dp.Markov([0.5, 1.1], [[0.6, 0.4], [0.2, 0.8]]))
ks.aggregate_shock("Z", dp.Markov([0.99, 1.01], [[0.875, 0.125], [0.125, 0.875]]))
ks.state("a", np.linspace(0.0, 150.0, 100))
ks.aggregate_state("K", np.linspace(25.0, 45.0, 5), mean="a")
ks.local("r = alpha*Z*(K/L)^(alpha - 1) - delta", "w = (1 - alpha)*Z*(K/L)^alpha")
ks.local("c = (1 + r)*a + w*e - a(+1)")
ks.reward("crra(c, gamma)")
ks.subject_to("c > 0")

agg = ks.solve("egm", ks=dict(T=1000, burn_in=200, max_outer=60))
rule = agg.equilibrium                 # log K' = b0 + b1 log K, una regla por Z
print(rule.b1, rule.r_squared, rule.converged)   # b1 alrededor de 0.96; converge en 8 rondas
```

`L` es la media de la cadena idiosincrática `e` (su distribución estacionaria
es (1/3, 2/3)). Una simulación larga (`T` de varios miles) da coeficientes de
pronóstico más precisos que esta corrida rápida.

**Use `method="egm"` aquí.** La VFI discreta coloca los activos del periodo
siguiente en los nodos de la malla, y en una malla gruesa el capital simulado
entonces nunca se mueve: la regresión de pronóstico devuelve `b1 = 0` con
R² = 1 y pasa la tolerancia. Tanto este bucle como `vfi.krusell_smith` emiten un
`RuntimeWarning` en ese caso y cuando se detienen en `max_outer` sin converger.
Revise `rule.converged` y `rule.r_squared` antes de usar la regla de pronóstico.

## Paneles y desigualdad

Las estadísticas de desigualdad se calculan de forma exacta a partir de
`sol.distribution`, sin simulación: cada nodo de la malla es un punto de masa
con su masa estacionaria. `sol.gini(expr)` es el Gini ponderado estándar,
`1 - sum_i w_i (L_i + L_{i-1})` sobre la curva de Lorenz, la misma función que
`MicroFrame.gini`. Admite valores negativos (el Gini puede entonces superar 1)
y es NaN cuando el total no es positivo. `sol.top_share(expr, p)`,
`sol.quantile(expr, q)` y `sol.lorenz(expr)` completan el cuadro.

```python
print(sol.gini("a"), sol.top_share("a", 0.10))   # alrededor de 0.62 y 0.41
print(sol.quantile("a", 0.5))                    # riqueza mediana, alrededor de 3.76
curve = sol.lorenz("a")                          # columnas pop_share, value_share
```

`sol.simulate(n_agents=10_000, periods=None, *, seed=0, init=None, burn_in=0,
record=None)` genera historias individuales bajo la política resuelta y
devuelve un `dp.Panel`. `periods=None` e `init=None` toman valores por defecto
según el horizonte: 200 periodos y `"stationary"` en horizonte infinito, las J
edades del horizonte y `"newborn"` en un ciclo de vida. En horizonte infinito
cada agente empieza en una celda (estado, choque) extraída de `sol.distribution`,
así que el panel está en la distribución estacionaria desde el primer periodo;
`init={"a": 0.1}` hace que todos empiecen en un estado dado, y `burn_in`
descarta los primeros periodos. Con VFI los estados se quedan en los nodos de
la malla; con EGM la política se interpola entre ellos. Una elección discreta
también se simula: la opción que maximiza sin choques de preferencia, una
extracción logit con ellos, registrada como el valor de la opción. El panel
registra los estados, los choques y los controles (`record=` admite cualquier expresión del
modelo, como `"a(+1)"` o `"w*exp(z)"`). La misma semilla da un panel idéntico
bit a bit.

```python
panel = sol.simulate(n_agents=10_000, periods=200, seed=1)
print(panel)                        # Panel(n_agents=10000, periods=200, columns=['a', 'z', 'c'])
gini_t = panel.gini("a", by="t")    # un Gini de corte transversal por periodo
print(gini_t.iloc[-1])              # alrededor de 0.62, el valor exacto salvo ruido muestral
df = panel.to_frame()               # formato largo: id, t, a, z, c
```

`panel.mean`, `panel.quantile`, `panel.gini` y `panel.top_share` agregan todos
los periodos, o agrupan con `by="t"` (o `by="age"` en un ciclo de vida).
`panel.cross_section(t)` y `panel.path(i)` devuelven un periodo o un agente, y
`panel.to_microframe()` pasa el panel a los estimadores de microdatos de
encuestas. Un `n_agents * periods` mayor que 5e8 lanza `ValueError` con la
memoria que necesitaría.

En un ciclo de vida el panel recorre las edades del horizonte. Los agentes
empiezan como recién nacidos y, con supervivencia declarada, un agente vivo a la
edad j llega a j+1 con probabilidad `survival[j]`. `panel.alive` lo registra.
Los fallecidos conservan sus filas pero quedan fuera de toda estadística. Las
estadísticas exactas agregan edades con `population_shares()` como pesos, o
toman una sola cohorte con `age=j`.

```python
shares = life.population_shares()   # supervivencia 0.99: 0.022 a la edad 0, 0.012 a la 59
print(life.gini("a"))               # todas las edades agregadas, alrededor de 0.73
print(life.gini("a", age=30))       # la cohorte de 30 años, alrededor de 0.50
lp = life.simulate(n_agents=5000, seed=0)
print(lp.alive[:, -1].mean())       # alrededor de 0.55 llegan a la edad 59
print(lp.mean("a", by="age")[45])   # alrededor de 7.7; life.mean("a")[45] es 7.68
```

Las soluciones en tiempo continuo (HJB) no se simulan: `sol.distribution`
contiene la densidad de Kolmogorov hacia adelante, y las estadísticas exactas
funcionan sobre ella. Las soluciones de Krusell-Smith no se simulan ni se
resumen (no tienen distribución estacionaria); `agg.aggregates` contiene la
trayectoria simulada del capital. Ambos casos lanzan un error que lo indica.

## Comprobar una especificación

`m.check()` compila el modelo sin resolverlo y lanza `ModelSpecError` con el
motivo, por ejemplo un identificador desconocido o un nombre declarado dos
veces. `solve("egm")` y `solve("hjb")` añaden sus propias comprobaciones y
lanzan el mismo error ante una restricción presupuestaria que no tiene la forma
`c + a(+1) = R*a + y(z)` o una deriva que no es lineal en el control.

## Lo que `dp` todavía no hace

- EGM con más de un activo (use `method="vfi"`, que admite varios estados);
- más de un precio en tiempo continuo;
- reglas de pronóstico de Krusell-Smith con más de un momento.

## Migrar desde vfi.Model

`vfi.Model` está obsoleto y se eliminará en la versión 6.0. Un `vfi.Model`
recibía funciones de Python para la utilidad y la restricción presupuestaria;
`dp` recibe expresiones. Cada lambda se convierte en su expresión como cadena,
la restricción presupuestaria pasa a ser un local para `c`, y `c > 0` sustituye
a la penalización de `-1e10`. Este es el modelo de ahorro precautorio del
cuaderno 21, primero tal como se escribía para `vfi.Model`:

```text
m = vfi.Model("Precautionary Savings Model")
m.set_param("beta", 0.96).set_param("gamma", 2.0).set_param("alpha", 0.36).set_param("delta", 0.10)
m.add_state("a", 0.1, 15.0, 60).add_shock("z", 0.90, 0.15, 5)
m.set_utility(lambda c, p: c**(1 - p["gamma"]) / (1 - p["gamma"]))
m.set_budget(lambda a, ap, z, p: np.exp(z) * a**p["alpha"] + (1 - p["delta"]) * a - ap)
res = m.solve(howard_steps=15, tol=1e-6)
```

y como modelo `dp`:

```python
pm = dp.Model("Precautionary Savings Model")
pm.parameters(beta=0.96, gamma=2.0, alpha=0.36, delta=0.10)
pm.exogenous("z", dp.AR1(rho=0.90, sigma=0.15, n=5))   # Tauchen, como add_shock
pm.state("a", np.linspace(0.1, 15.0, 60))
pm.local("c = exp(z)*a^alpha + (1 - delta)*a - a(+1)")
pm.reward("c^(1 - gamma)/(1 - gamma)")
pm.subject_to("c > 0")

ps = pm.solve(method="vfi", tol=1e-6)
print(ps.mean("a"), ps.gini("a"))   # 6.5733 y 0.3494
```

La restricción presupuestaria no tiene la forma `R*a + y(z)`, así que este
modelo usa `method="vfi"`. La solución de `dp` elige el mismo nodo de la malla
para el periodo siguiente que `vfi.Model` en todos los estados, y las funciones
de valor coinciden a 7e-7. `res["V"]`, `res["policy_a"]` y `res["a_grid"]` pasan
a ser `ps.V`, `ps.policy("a(+1)")` y `ps.grids["a"]`. `res["mean_assets"]` y
`res["gini"]` (6.6767 y 0.3924) venían de una simulación de 2000 hogares y 100
periodos, y ese Gini restaba primero la tenencia mínima de activos.
`ps.mean("a")` y `ps.gini("a")` son los valores exactos de la distribución
estacionaria, y `ps.simulate(...)` da el panel.

## Validación

Los modelos de Aiyagari, Huggett, ciclo de vida y dos activos escritos como
modelos `dp` reproducen exactamente `vfi.examples` (mismos precios, políticas y
funciones de valor). Hay además comprobaciones independientes que no pasan por
el compilador: los modelos de dos periodos con utilidad logarítmica coinciden
con su forma cerrada a 1e-12, el último periodo logit coincide con su forma
cerrada a 1e-12, un modelo de dos periodos con decisión discreta coincide con
una maximización por fuerza bruta sobre 200,001 puntos, y el equilibrio de
Aiyagari en tiempo continuo coincide con `vfi.solve_aiyagari_continuous_hjb`
(r* a 1e-9). Las pruebas están en `tests/test_dp/`.

## Ejemplo desarrollado

`python -m puremacro.examples.scf_aiyagari` calibra el factor de descuento de un
modelo de Aiyagari `dp` (EGM) a una razón riqueza-ingreso de la Survey of
Consumer Finances y compara los momentos no calibrados con los datos, con
errores estándar. Véase [Microdatos de encuestas](micro_data.md).

## Referencias

- Aiyagari, S. R. (1994). Uninsured idiosyncratic risk and aggregate saving. *Quarterly Journal of Economics*, 109(3), 659–684.
- Achdou, Y., Han, J., Lasry, J.-M., Lions, P.-L. y Moll, B. (2022). Income and wealth distribution in macroeconomics: a continuous-time approach. *Review of Economic Studies*, 89(1), 45–86.
- Carroll, C. D. (2006). The method of endogenous gridpoints for solving dynamic stochastic optimization problems. *Economics Letters*, 91(3), 312–320.
- Iskhakov, F., Jørgensen, T. H., Rust, J. y Schjerning, B. (2017). The endogenous grid method for discrete-continuous dynamic choice models with (or without) taste shocks. *Quantitative Economics*, 8(2), 317–365.
- Krusell, P. y Smith, A. A. (1998). Income and wealth heterogeneity in the macroeconomy. *Journal of Political Economy*, 106(5), 867–896.
