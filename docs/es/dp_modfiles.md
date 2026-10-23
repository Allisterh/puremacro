> 🇬🇧 [English](../dp_modfiles.md) · 🇪🇸 Español

# Programas dinámicos en un archivo `.mod`

Escriba el objetivo y las restricciones del hogar o de la empresa como
ecuaciones y resuelva el archivo con `puremacro.dp.solve_mod`. Las declaraciones
se compilan al mismo [`dp.Model`](dp.md): se reutilizan los solvers VFI, EGM y HJB
y el resultado es un `DPSolution` ordinario.

Es **el lenguaje de programación dinámica de Puremacro con sintaxis inspirada
en Dynare**. No es un sistema DSGE ejecutable por Dynare. Para archivos DSGE de
Dynare utilice `puremacro.dsge.load_mod`.

## Un modelo VFI completo

Guarde este texto como `growth.mod`:

```text
parameters beta alpha;
beta = 0.95;
alpha = 0.36;
state k = geomspace(0.01, 2.0, 300);
model(name='Crecimiento por iteración de la función de valor');
  c = k^alpha - k(+1);
  reward = log(c);
end;
constraints; c > 0; end;
solve(method=vfi, tol=1e-9, distribution=false);
```

```python
from puremacro import dp

sol = dp.solve_mod("growth.mod")
print(sol.summary())
consumo = sol.policy("c")
capital_siguiente = sol.policy("k(+1)")
```

`k` es el capital actual y `k(+1)` es la elección de capital del siguiente
periodo. El compilador construye el problema de Bellman. Con utilidad
logarítmica y depreciación completa, la política continua exacta es
`k_next = alpha*beta*k**alpha`; la malla discreta introduce error numérico.

```bash
puremacro-dp growth.mod --check
puremacro-dp growth.mod
python -m puremacro.dp growth.mod --param beta=0.97
```

`--check` valida y compila sin resolver. No certifica convergencia, precisión de
la malla ni validez económica.

## Ahorro con riesgo de ingresos

```text
parameters beta gamma r w;
beta = 0.96; gamma = 2; r = 0.02; w = 1;
state a = 40*linspace(0, 1, 120)^2;
exogenous z = rouwenhorst(rho=0.9, sigma=0.15, n=5);
model;
  c = (1+r)*a + w*exp(z) - a(+1);
  reward = crra(c, gamma);
end;
constraints; c > 0; end;
aggregates; A = a; C = c; end;
solve(method=egm, tol=1e-9);
```

EGM calcula políticas continuas entre nodos bajo las restricciones de presupuesto
y preferencias del compilador existente. `method=vfi` utiliza búsqueda discreta.
Varias declaraciones `state` y una elección `discrete` permiten los problemas
multidimensionales que admite VFI.

```python
modelo = dp.load_mod("household.mod")  # valida sin resolver
egm = dp.solve_mod("household.mod")
vfi = dp.solve_mod("household.mod", method="vfi", tol=1e-10)
alternativo = dp.solve_mod("household.mod", params={"r": 0.01})
```

Las opciones Python prevalecen sobre las del archivo. Los parámetros sustituidos
se aplican antes de evaluar calibraciones y mallas dependientes; deben estar
declarados en el archivo. `modelo.solve(...)` usa los valores predeterminados
de la API Python, mientras que `solve_mod` aplica explícitamente el comando del
archivo.

## Declaraciones

| Declaración | Significado |
|---|---|
| `parameters beta gamma; beta=.96; gamma=2;` | Declarar y calibrar una sola vez, en orden de dependencia. Se admiten vectores de parámetros por edad. |
| `state a = linspace(0,40,120);` | Estado endógeno con malla finita y creciente. Su valor siguiente es una elección. |
| `exogenous z = ar1(rho=.9,sigma=.2,n=5);` | AR(1) gaussiano; Tauchen por defecto. También `tauchen(...)` y `rouwenhorst(...)`. |
| `exogenous z = markov([.5,1.5],[.9,.1; .2,.8]);` | Cadena de Markov explícita con matriz de transición por filas. |
| `discrete h = [0,1];` | Una elección discreta contemporánea. |
| `taste_shocks = sigma_h;` | Choques EV1 para el solver EGM con elección discreta. |
| `model; c=...; reward=...; end;` | Expresiones locales y un único objetivo; también se admite `# c=...`. |
| `constraints; c > 0; end;` | Condiciones de factibilidad. |
| `aggregates; A=a; C=c; end;` | Integrales bajo la distribución calculada, respetando la temporalidad declarada. |
| `price r=[-.01,.04]; clear=A-K_demand;` | Intervalo de un precio de equilibrio y residuo escalar de vaciado de mercado. La demanda debe estar definida con parámetros y expresiones locales. |
| `discount patience;` | Usar `patience` como factor de descuento en lugar de `beta`. |
| `horizon=30; survival=full(30,.99);` | Horizonte finito y supervivencia opcional; `age` comienza en cero. |
| `terminal=...; newborns=...;` | Arreglos opcionales con las dimensiones de `Model.horizon`. |
| `control c; drift a=r*a+z-c; discount_rate rho;` | Tiempo continuo mediante HJB; los choques usan `jump(valores, tasas)`. |
| `solve(method=egm,tol=1e-9);` | Comando final opcional, ejecutado únicamente por `solve_mod`. |

La calibración admite aritmética, `^`, arreglos, parámetros anteriores,
`linspace`, `geomspace`, `logspace`, `full`, `exp`, `log`, `sqrt`, `abs` y
`min`/`max` binarios. Las columnas de matrices requieren comas; los puntos y
coma separan filas. También se admiten listas anidadas. El lector limita los
arreglos de calibración y las matrices AR(1) a un millón de elementos; la API
Python permite construir entradas mayores. El producto de varias cadenas
independientes puede ser grande.

`solve` admite `method`, `backend`, `tol`, `howard`, `max_iter`, `distribution`,
`xtol` y `max_evals`, según el solver seleccionado. Se rechazan opciones
explícitas que serían ignoradas: `backend` y `howard` con EGM/HJB; `tol`,
`max_iter` y `howard` en horizontes finitos; y `xtol`/`max_evals` sin precios de
equilibrio. Los horizontes finitos y los equilibrios requieren distribución,
por lo que rechazan `distribution=false`. Las opciones Python se validan
después de combinarlas con las del archivo; las omitidas conservan los valores
predeterminados habituales. Véase la [guía DP](dp.md).
Los modelos con riesgo agregado/Krusell–Smith siguen disponibles mediante la
API Python; no forman parte de esta gramática de archivo.

Los comentarios usan `//`, `%` o `/* ... */`. Cada declaración termina en punto
y coma. Los errores identifican el archivo y la línea de la declaración. No se
ejecutan scripts Python/MATLAB, macros, inclusiones ni comandos desconocidos.
Las funciones económicas propias se registran explícitamente mediante
`dp.load_mod(source, functions={"tax": mi_impuesto})`, con el contrato existente
de `Model.functions`; el archivo no puede definirlas ni importarlas.

## Ejemplos y evidencia

El paquete instalado incluye `growth.mod`, `household.mod` y `lifecycle.mod`:

```python
from importlib.resources import files
from puremacro import dp

texto = files("puremacro.dp").joinpath("_examples/household.mod").read_text()
sol = dp.solve_mod(texto)
```

Las pruebas comparan archivos con declaraciones Python en VFI, EGM y HJB,
enumeran un máximo de Bellman de dos estados de forma independiente, verifican
la decisión de la última edad y contrastan crecimiento con su solución cerrada.
Son comprobaciones numéricas y del compilador, no validación empírica. Consulte
las [auditorías de resultados](result_audits.md) y refine las mallas explícitamente.
