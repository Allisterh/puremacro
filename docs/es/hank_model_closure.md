> 🇬🇧 [English](../hank_model_closure.md) · 🇪🇸 Español

<!-- Language: es | Counterpart: ../hank_model_closure.md -->

# Ecuaciones agregadas HANK editables

El puente HANK de archivos `.mod` conserva su cierre neokeynesiano incorporado
por defecto. Use `closure="model"` para resolver las ecuaciones agregadas del
archivo. Este modo optativo combina ecuaciones afines homogéneas en
**desviaciones de niveles** con respuestas del hogar mapeadas explícitamente.
`model(linear)` expresa esa intención; el parser también reconoce bloques
`model` afines.

```python
import numpy as np
from puremacro.dsge.hank import solve_hank_bridge

source = """
var Y C r pi i;
varexo monetary government;
parameters beta gamma r_ss phi_pi kappa;
beta=.985; gamma=1; r_ss=.01; phi_pi=1.5; kappa=.1;
hetagent_block; model=one_asset_hank; n_a=30; a_max=30; end;
model(linear);
Y=C+government;
pi=beta*pi(+1)+kappa*Y;
i=.3*i(-1)+phi_pi*pi+monetary;
r=i-pi(+1);
end;
"""
T = 12
result = solve_hank_bridge(
    source, horizon=T, closure="model",
    household_inputs={"r": "r", "Y": "Y"},
    household_outputs={"C": "C"},
    shock_paths={"monetary": -.002*.6**np.arange(T),
                 "government": .001*.4**np.arange(T)},
)
assert result.converged
assert result.diagnostics["aggregate_residual_max_abs"] < 1e-9
print(result.transition_paths.head())
```

Cada mapeo va de un **nombre del hogar** a un nombre agregado declarado. Los
insumos pueden apuntar a variables endógenas o choques exógenos; los productos
deben apuntar a endógenas distintas. Al añadir una identidad de respuesta por
producto del hogar, debe haber exactamente una ecuación por variable endógena.
Nombres no admitidos, destinos duplicados, constantes no homogéneas y
expresiones no lineales producen errores explícitos. Las variables sin resolver
no se rellenan con ceros. Por ejemplo, para desviaciones escriba
`i=phi_pi*pi+monetary`, no `i=r_ss+phi_pi*pi+monetary`.

`shock_paths` contiene arreglos finitos de exactamente `horizon` observaciones;
los choques declarados que se omiten son cero. Sin este argumento, el `shock`
declarado recibe `magnitude*rho**arange(horizon)`. En este modo el nombre del
choque determina qué ecuación afecta. Las desviaciones endógenas y exógenas
antes de 0 y después de `T-1` son cero. Es una condición de frontera de horizonte
finito: amplíe el horizonte y compare trayectorias si importa el efecto terminal.

El solver comprueba rango completo y evalúa independientemente las ecuaciones
originales y las identidades del hogar después de resolver. Los diagnósticos
incluyen rango, dimensión, número de condición y ambos residuos máximos.
Rango completo significa **resolubilidad a horizonte finito**, no determinación
de Blanchard–Kahn, unicidad del equilibrio a horizonte infinito ni precisión de
la malla del hogar. `closure_tol` controla el umbral absoluto de aceptación
(por defecto `1e-9`). No se sustituye silenciosamente una pseudoinversa ni
regularización. Se ensambla una matriz densa de tamaño
`(variables*horizon)**2`; empiece con horizontes moderados.

## Unidades y fechas del hogar

Todos los insumos y productos son desviaciones de niveles; las tasas usan el
mismo período y unidades de su calibración. No son porcentajes ni desviaciones
logarítmicas. El mapeo cambia nombres sin reescalar coordenadas.

| Hogar | Insumos disponibles | Productos disponibles | Convención |
|---|---|---|---|
| Incorporado de un activo | `r`, `Y` | `C` | `r` es la tasa ex ante realizada el siguiente período; `Y` cambia el salario bruto por `w_ss/C_ss`, conservando el impuesto sobre intereses de presupuesto equilibrado. |
| Incorporado de dos activos | `r_b`, `r_a`, `Y` | `C`, `D`, `A`, `B`, `CHI`, `z` | Tasas ex ante; ingreso no financiero cerrado como producto menos retornos de activos heredados. `A` y `B` son decisiones de ahorro. |
| `dp.Model` o `DPSolution` declarativo | Insumos declarados admitidos | Expresiones agregadas declaradas y alias admitidos | Unidades y fechas originales de los precios del hogar; sin ecuación fiscal adicional ni reescalamiento salarial. |

En un hogar declarativo, `A="a"` mide activos al entrar al período y
`S="a(+1)"` mide ahorro elegido en ese período. Esas fechas se conservan al
mapear variables agregadas. Especifique explícitamente salarios, política
fiscal, mercados de activos y precios en el bloque agregado. Un `DPSolution`
ya resuelto fija además la calibración que genera los jacobianos.

Los cierres incorporados siguen disponibles para comparar. Las opciones
exclusivas de ecuaciones editables se rechazan con `closure="builtin"`.

## Equilibrio con hogares no lineales

Con los mismos mapeos, `nonlinear=True` reemplaza las identidades lineales del
hogar por transiciones exactas de políticas hacia atrás y distribución hacia
adelante. Las ecuaciones agregadas siguen siendo afines y homogéneas en
desviaciones. Se admiten hogares incorporados y declarativos de uno y dos
activos; no es un lenguaje de ecuaciones agregadas no lineales arbitrarias.

```python
nonlinear = solve_hank_bridge(
    source, horizon=T, closure="model", nonlinear=True,
    household_inputs={"r": "r", "Y": "Y"},
    household_outputs={"C": "C"},
    shock_paths={"monetary": -.002*.6**np.arange(T)},
    closure_tol=1e-9, closure_max_iter=100,
)
assert nonlinear.converged
assert nonlinear.diagnostics["household_budget_max_abs"] < 1e-8
assert nonlinear.diagnostics["distribution_mass_max_abs"] < 1e-10
```

La solución lineal inicializa iteraciones de Broyden amortiguadas. Si esa
aproximación inicial es inviable, se intenta con desviaciones endógenas cero.
Cada paso debe reducir la norma del residuo y conservar presupuestos factibles
y la masa de la distribución. Se rechazan trayectorias con piso de consumo
activo. La distribución inicial y la continuación terminal del hogar corresponden
al estado estacionario calibrado; las tasas ex ante incluyen el retorno realizado
en la fecha terminal. Los hogares declarativos conservan sus fechas originales
de realización de los insumos.

Los diagnósticos informan iteraciones, historia de residuos, errores finales de
presupuesto y masa, y motivo de terminación. Si falla la convergencia, se advierte
y devuelve la última trayectoria con `converged=False`, sin sustituirla por la
respuesta lineal. Si un insumo exógeno hace inviable incluso la trayectoria
inicial, se produce `ValueError`. La convergencia numérica no certifica unicidad
del equilibrio, convergencia de malla ni una auditoría independiente de Euler.
El contexto privado conserva las respuestas agregadas finales del hogar sin
ejecutar otro solver.

## Paneles de hogares con dos activos

Un hogar de dos activos resuelto por EGM admite la interfaz habitual de panel:

```python
# sol is a two-asset DPSolution, with declared assets a and b.
panel = sol.simulate(
    n_agents=1000, periods=40, seed=2026, burn_in=10,
    record=["a", "b", "a(+1)", "b(+1)", "c"],
)
microdata = panel.to_microframe(t=20)
```

Ambas decisiones de activos se interpolan bilinealmente y siguen siendo
continuas. El consumo y los costos de ajuste registrados se calculan con el
presupuesto y las decisiones efectivas, no interpolando aparte una tabla de
consumo. La productividad sigue la cadena de Markov resuelta. La misma semilla
reproduce exactamente el panel y su calentamiento; se admite declarar los
activos en orden inverso. Los valores iniciales pueden estar entre nodos; los
que exceden la malla se recortan con una advertencia.

La inicialización estacionaria muestrea la distribución conjunta resuelta.
Las políticas continuas posteriores no tienen por qué preservar exactamente
la distribución de la lotería discretizada: compare momentos simulados con
`sol.mean(...)` y refine la malla cuando esa aproximación importe. Son paneles
bajo políticas estacionarias, no paneles de una transición de equilibrio general.
