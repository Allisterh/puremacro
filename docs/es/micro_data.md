> 🇬🇧 [English](../micro_data.md) · 🇪🇸 Español

# Registro de fuentes de datos y microdatos de encuestas

Dos piezas se apoyan en los conectores existentes:

- `puremacro.fetch.registry`: una tabla de todas las fuentes que puremacro
  puede consultar (series, paneles, vintages en tiempo real, microdatos y los
  conectores de texto narrativo), con lo que cada una necesita antes de
  llamarla.
- `puremacro.fetch.micro`: encuestas de hogares y personas descargadas mediante
  API públicas o archivos de descarga, devueltas con su diseño muestral para que
  cada estimación venga con un error estándar basado en el diseño.

## El registro

El registro describe; no envuelve. Los conectores existentes conservan sus
propias firmas, y cada entrada registra dónde vive la función, si necesita
clave, los términos de uso y si los resultados se pueden redistribuir.
Importarlo no carga código de red, así que también funciona en el navegador.

```python
from puremacro.fetch import registry

micro = registry.sources("micro")
print(micro[["id", "auth", "credential", "redistributable"]])
print(registry.sources()["kind"].value_counts())   # series, panel, vintage, micro, text
print(registry.info("fed.scf").terms_url)
```

`auth` es uno de `none`, `key_optional`, `key`, `registration` (una cuenta
gratuita con términos aceptados por colección), `approval` (una propuesta de
proyecto revisada por el proveedor) o `unaudited`. Los conectores de texto
narrativo aparecen como `unaudited` y no redistribuibles: sus términos no se han
revisado uno por uno, así que conviene consultar los términos de cada fuente
antes de republicar su texto. `registry.load(id)` importa y devuelve el
conector; `registry.register(SourceInfo(...))` añade uno propio.

## Proveedores de microdatos

| función | encuesta | diseño de varianza |
|---|---|---|
| `fetch_acs_pums(year, variables, ...)` | American Community Survey PUMS del Census (1 o 5 años), personas u hogares | 80 pesos replicados por diferencias sucesivas |
| `fetch_cps_basic(year, month, variables, ...)` | Current Population Survey del Census, mensual básica | sólo pesos (no hay réplicas publicadas) |
| `fetch_scf(year, variables, ...)` | Survey of Consumer Finances de la Reserva Federal, extracto resumen | cinco imputaciones, 999 pesos replicados bootstrap, reglas de Rubin |
| `fetch_enigh(year, variables, ...)` | ENIGH del INEGI (México), desde 2016 | estratos y UPM, linealización de Taylor |

La API del Census necesita una clave gratuita: guárdela como se describe en
[Credenciales](CREDENTIALS.md) (servicio `census`) en lugar de pasarla en el
código. Las claves nunca llegan a los nombres de archivo de la caché, al
manifiesto ni a los mensajes de error. La SCF y la ENIGH no necesitan clave.

```python
# requires: network (Census API key)
from puremacro.fetch.micro import fetch_acs_pums

acs = fetch_acs_pums(2023, ["AGEP", "WAGP", "SEX"], geography="state:11",
                     where={"AGEP": (25, 64)})
acs.mean("WAGP", by="SEX")          # estimación, ee y n por grupo
```

```python
# requires: network (federalreserve.gov)
from puremacro.fetch.micro import fetch_scf

scf = fetch_scf(2025, ["networth", "income", "age"])
scf.quantile("networth", 0.5)                  # publicado: 215,900 dólares
scf.ratio("networth", "income", q=0.5)         # riqueza mediana / ingreso mediano
scf.gini("networth"), scf.top_share("networth", 0.01)
```

Las descargas pasan por la caché HTTP compartida, así que la segunda llamada no
usa la red.

### Lo que conviene saber antes de estimar

- **Códigos de no aplicable.** Códigos del Census como `WAGP == -1` (personas
  menores de 15 años) o `ESR == 0` se convierten en `NaN` por defecto, leídos
  del propio libro de códigos del endpoint, para que no entren como valores en
  medias y totales. `na_to_nan=False` conserva los códigos originales y
  `frame.codebook` lista todos los códigos etiquetados. La CPS envía algunos
  códigos de ingresos con decimales implícitos (`-1` llega como `-0.01`); esos
  también se tratan.
- **Dólares de la SCF.** La Reserva Federal publica todas las rondas del
  extracto resumen en dólares de la encuesta más reciente. Con la publicación de
  2025 (octubre de 2026) los niveles de 2022 subieron 9.0%; las razones, las
  participaciones y el Gini no cambian.
- **Imputaciones de la SCF.** La SCF imputa cinco veces las respuestas
  faltantes. Cada hogar tiene cinco filas; las estimaciones promedian sobre las
  imputaciones y el error estándar suma la varianza entre imputaciones (reglas
  de Rubin) a la varianza muestral de las réplicas de la imputación 1, como
  indica la documentación de la Reserva Federal.
- **La ENIGH** no tiene pesos replicados. Medias, totales y cuantiles usan
  linealización de Taylor con la varianza de conglomerados últimos; `ratio`,
  `gini` y `top_share` lanzan `NotImplementedError` en lugar de devolver una
  estimación sin su error estándar.

## Estimar con un `MicroFrame`

Cada proveedor devuelve un `MicroFrame`: los registros, un `SurveyDesign` que
indica qué columnas son pesos y cómo convertir las réplicas en varianzas, y la
consulta que los produjo. Los estimadores `mean`, `total`, `quantile`, `share`,
`ratio`, `gini` y `top_share` devuelven la estimación, su error estándar, el `n`
sin ponderar y el número de imputaciones, opcionalmente por grupos (`by`).

El bloque siguiente construye un marco a mano para que funcione sin red; un
marco descargado se comporta igual.

```python
import numpy as np
import pandas as pd
from puremacro.fetch.micro import MicroFrame, SurveyDesign

rng = np.random.default_rng(0)
n = 2000
w = rng.uniform(50, 150, n)
records = pd.DataFrame({"PWGTP": w, "WAGP": rng.lognormal(10.5, 0.8, n),
                        "SEX": rng.integers(1, 3, n)})
replicates = pd.DataFrame(w[:, None] * rng.uniform(0.7, 1.3, (n, 80)),
                          columns=[f"PWGTP{r}" for r in range(1, 81)])
frame = MicroFrame(pd.concat([records, replicates], axis=1),
                   SurveyDesign.acs_pums(), unit="person",
                   source="synthetic", vintage="demo")

print(frame.mean("WAGP", by="SEX"))
print(frame.quantile("WAGP", 0.5))
print(frame.gini("WAGP"))
print(frame.top_share("WAGP", 0.10))
```

`SurveyDesign(...)` describe cualquier otra encuesta: un peso principal, pesos
replicados con su método (`"sdr"`, `"bootstrap"`, `"jk1"`, ...), escala y
centrado, o estratos y UPM con `method="taylor"`.

## Ejemplo desarrollado: una economía de Aiyagari frente a la SCF

```text
python -m puremacro.examples.scf_aiyagari --output results/
```

El script descarga una ronda de la SCF, estima razones riqueza-ingreso y
momentos de concentración con errores estándar por réplicas, calibra el factor
de descuento de un modelo de Aiyagari [`dp`](dp.md) (resuelto por EGM) a una
razón, traslada el error muestral del objetivo a `beta` por el método delta, y
compara los momentos no calibrados con los datos. Con la SCF 2025 y `beta`
calibrado a la riqueza mediana sobre el ingreso mediano:

| momento | SCF (ee) | modelo (ee) |
|---|---|---|
| riqueza mediana / ingreso mediano | 2.628 (0.077) | calibrado |
| riqueza media / ingreso medio | 8.518 (0.175) | 3.912 (0.101) |
| Gini | 0.824 (0.004) | 0.591 (0.001) |
| participación del 1% superior | 0.313 (0.009) | 0.063 (0.000) |
| participación del 10% superior | 0.721 (0.007) | 0.379 (0.001) |
| participación del 50% inferior | 0.021 (0.001) | 0.088 (0.001) |

`beta` = 0.952 (ee 0.002), r = 4.08%, K/Y = 2.98. La brecha en la parte alta es
la falla conocida del modelo de un activo con riesgo de ingreso y límite de
endeudamiento; nada en el ejemplo está ajustado para cerrarla. Calibrar a la
razón de medias no es posible con `beta < 1`: el modelo llega como máximo a
7.55, con r < 0.
