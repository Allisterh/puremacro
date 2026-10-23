# Macroeconomía Avanzada (MAV) — cuadernos del curso

*English version below.*

Cuadernos de Jupyter del curso de Macroeconomía Avanzada (MAV) de la Maestría en
Economía del ITAM, otoño 2026. Todo el curso corre en Python con
[`puremacro`](https://github.com/jalonso1979/puremacro): cada cuaderno trae sus salidas ya
ejecutadas, así que se puede leer en GitHub sin correr nada.

La carpeta contiene los cuadernos en español (`*_es.ipynb`) y en inglés, los
auxiliares `_datos.py`, `_nbstyle.py`, `_tutor.py`, `_llm.py`, `_t01.py` y `_dsge_labs.py`, los datos congelados en
`data_curso/` y los modelos `.mod` en `modelos/`. Las diapositivas en PDF están
en `curso/site/diapositivas/`.

**Actualización DSGE del 23 de octubre de 2026 (5.7.0).** T02_A–D usan modelos no lineales sin resolución implícita, ejecución ordenada de comandos `.mod`, transiciones con continuación y diagnósticos comunes de soluciones locales y globales. T02_D conserva las lecciones de inversión irreversible y volatilidad estocástica añadidas el 22 de octubre. Los ocho cuadernos actualizados fijan el código fuente y los auxiliares del tag `v5.7.0`; el tag `course-2026A-5.5.0` conserva la edición original. Véase la [validación adicional](../validation/README.md).

Para ejecutar T02_A–D con Python 3.12 o posterior, clone el tag y use un entorno virtual nuevo:

```bash
git clone --branch v5.7.0 --depth 1 https://github.com/jalonso1979/puremacro.git puremacro-5.7.0
cd puremacro-5.7.0
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: ./.venv/Scripts/Activate.ps1
python -m pip install -e ".[notebooks]" jupyterlab
cd curso/notebooks
jupyter lab
```

Seleccione ese entorno como kernel. Mantenga `_dsge_labs.py`, `modelos/` y `data_curso/` junto a los cuadernos. La validación registrada usa Python local; Colab instala el código del tag `v5.7.0`. Esta evidencia no certifica una ejecución completa en Colab. Las instrucciones siguientes reproducen la edición original completa.

## Abrir un cuaderno en Google Colab

Haga clic en el enlace del cuaderno. La primera celda de código instala
`puremacro` y descarga los materiales correspondientes del repositorio (un clon parcial de los materiales
y datos congelados); no pide acceso a Google Drive. Ejecute las celdas en orden.

| Código | Tema | Español | Inglés |
|---|---|---|---|
| T00 | Syllabus interactivo del curso | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T00_syllabus_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T00_syllabus.ipynb) |
| T00_B | Configura tu modelo de lenguaje local (Qwen3-4B) | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T00_B_configura_tu_modelo_local_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T00_B_local_model_setup.ipynb) |
| T01_A | Hechos del ciclo económico desde el mercado de trabajo | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T01_A_data_panels_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T01_A_data_panels.ipynb) |
| T01_B | Contabilidad nacional y hechos estilizados internacionales | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T01_B_measurement_and_cycles_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T01_B_measurement_and_cycles.ipynb) |
| T01_C | El ciclo a frecuencia mensual, semanal y diaria | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T01_C_stylized_facts_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T01_C_stylized_facts.ipynb) |
| T02_A | Modelo neoclásico de crecimiento: log-linealización y aproximación local | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/v5.7.0/curso/notebooks/T02_A_neoclassical_growth_local_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/v5.7.0/curso/notebooks/T02_A_neoclassical_growth_local.ipynb) |
| T02_B | RBC canónico: simulación, momentos y el debate de las horas | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/v5.7.0/curso/notebooks/T02_B_rbc_dynare_local_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/v5.7.0/curso/notebooks/T02_B_rbc_dynare_local.ipynb) |
| T02_C | Precios rígidos y búsqueda: fricciones en bienes y trabajo | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/v5.7.0/curso/notebooks/T02_C_new_keynesian_local_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/v5.7.0/curso/notebooks/T02_C_new_keynesian_local.ipynb) |
| T02_D | OccBin, inversión irreversible y choques de incertidumbre | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/v5.7.0/curso/notebooks/T02_D_occbin_zlb_regimes_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/v5.7.0/curso/notebooks/T02_D_occbin_zlb_regimes.ipynb) |
| T02_E | Estimación bayesiana de modelos DSGE y comparación de modelos | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T02_E_bayesian_dsge_local_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T02_E_bayesian_dsge_local.ipynb) |
| T02_F | Fricciones macrofinancieras: Gertler y Karadi (2011) | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T02_F_gertler_karadi_macrofinance_local_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T02_F_gertler_karadi_macrofinance_local.ipynb) |
| T03_A | Choques agregados con mercados incompletos: Krusell–Smith y jacobianos | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T03_A_aggregate_shocks_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T03_A_aggregate_shocks.ipynb) |
| T03_B | Volatilidad: GARCH y efecto apalancamiento | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T03_B_garch_volatility_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T03_B_garch_volatility.ipynb) |
| T03_C | Crecimiento en riesgo (growth-at-risk) | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T03_C_growth_at_risk_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T03_C_growth_at_risk.ipynb) |
| T03_D | Incertidumbre de política económica a partir de texto | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T03_D_narrative_uncertainty_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T03_D_narrative_uncertainty.ipynb) |
| T03_E | Modelos de lenguaje locales como instrumento de medición | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T03_E_local_llm_uncertainty_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T03_E_local_llm_uncertainty.ipynb) |
| T03_F | Comunicación de bancos centrales y tono de la política monetaria | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T03_F_narrative_llm_central_banks_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T03_F_narrative_llm_central_banks.ipynb) |
| T04_A | El multiplicador de impuestos, de tres maneras | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T04_A_tax_multiplier_three_ways_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T04_A_tax_multiplier_three_ways.ipynb) |
| T04_C | Contabilidad del ciclo económico: las cuñas | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T04_C_contabilidad_ciclo_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T04_C_business_cycle_accounting.ipynb) |
| T05_A | Identificación de choques con SVAR | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T05_A_svar_identification_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T05_A_svar_identification.ipynb) |
| T05_B | Proyecciones locales | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T05_B_local_projections_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T05_B_local_projections.ipynb) |
| T05_C | Diferencias en diferencias escalonadas | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T05_C_staggered_did_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T05_C_staggered_did.ipynb) |
| T05_D | DiD con proyecciones locales (LP-DiD) | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T05_D_lp_did_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T05_D_lp_did.ipynb) |
| T05_E | Transmisión dependiente del estado y respuestas generalizadas | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T05_E_regime_girf_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T05_E_regime_girf.ipynb) |
| T05_F | SVAR de la inflación de 2021–2024 (Bernanke–Blanchard) | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T05_F_svar_post_covid_inflation_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T05_F_svar_post_covid_inflation.ipynb) |
| T05_G | Identificación narrativa de choques monetarios con un LLM local (Banxico y FOMC) | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T05_G_choques_narrativos_llm_banxico_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T05_G_narrative_llm_banxico.ipynb) |
| T06_A | ¿De dónde viene la desigualdad de riqueza? | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T06_A_wealth_inequality_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T06_A_wealth_inequality.ipynb) |
| T06_B | Ciclo de vida, demografía y pensiones | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T06_B_life_cycle_and_demographics_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T06_B_life_cycle_and_demographics.ipynb) |
| T06_C | Carteras, preferencias y la frontera HANK | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T06_C_portfolios_and_preferences_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T06_C_portfolios_and_preferences.ipynb) |
| T07_A | Dinámica empresarial y mala asignación de recursos | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T07_A_firm_dynamics_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T07_A_firm_dynamics.ipynb) |
| T08_A | Búsqueda laboral y flujos de trabajadores | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T08_A_labor_flows_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T08_A_labor_flows.ipynb) |
| T09_A | Galería de validación numérica | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T09_A_validation_gallery_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T09_A_validation_gallery.ipynb) |
| T09_B | Construye tu propio índice | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T09_B_build_your_own_index_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T09_B_build_your_own_index.ipynb) |

## Correr los cuadernos en su máquina

Requiere Python 3.12 o posterior.

```bash
git clone --branch course-2026A-5.5.0 --depth 1 --filter=blob:none --sparse https://github.com/jalonso1979/puremacro.git
cd puremacro
git sparse-checkout set curso/notebooks
python -m pip install "puremacro[io]==5.5.0" statsmodels arch scikit-learn jupyterlab
cd curso/notebooks
jupyter lab
```

La edición fija `puremacro==5.5.0` y el contenido del tag `course-2026A-5.5.0`: auxiliares,
datos, caché y modelos. `puremacro.dp` y `puremacro.text` son obligatorios y sus
secciones se ejecutan completas. Los 72 cuadernos incluyen las variantes históricas
de T01; la tabla muestra la ruta principal.

La instalación inicial requiere red. Después se usan los datos congelados de
`data_curso/` y, por omisión, `MAV_LLM_MODE=cache`: respuestas de referencia de
Qwen3-4B, sin buscar motores ni descargar modelos. T00_B también corre en ese modo;
la comparación con un modelo vivo y los refrescos de datos son opcionales.
Para usar un modelo local, siga T00_B (`pip install "puremacro[local-llm]==5.5.0"`
u Ollama) y elija explícitamente `MAV_LLM_MODE=auto`.

## Datos

Los archivos de `data_curso/` son copias congeladas de datos públicos de
terceros (FRED, OCDE, INEGI, ILOSTAT, Banco Mundial y otros). La licencia MIT de
`puremacro` cubre el código, **no** esos datos. La procedencia de cada archivo y
cómo citarlo están en [`data_curso/FUENTES.md`](data_curso/FUENTES.md). Si usa
estos datos, cite a los editores originales.

## Licencia

El código es MIT. El texto de los cuadernos y las diapositivas del curso están bajo
[Creative Commons Atribución 4.0 Internacional (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/deed.es):
puede reutilizarlos y adaptarlos, también con fines comerciales, citando
«Jorge Alonso-Ortiz, *Macroeconomía Avanzada*, ITAM». Los datos conservan los
términos de sus editores.

---

# Advanced Macroeconomics (MAV) — course notebooks

Jupyter notebooks for the Advanced Macroeconomics (MAV) course in ITAM's Master's
programme in Economics, fall 2026. The whole course runs in Python with
[`puremacro`](https://github.com/jalonso1979/puremacro). Every notebook is saved with its
executed outputs, so you can read it on GitHub without running anything.

This folder holds the notebooks in Spanish (`*_es.ipynb`) and English, the
helpers `_datos.py`, `_nbstyle.py`, `_tutor.py`, `_llm.py`, `_t01.py` and `_dsge_labs.py`, the frozen data in
`data_curso/` and the `.mod` models in `modelos/`. The PDF slides are in
`curso/site/diapositivas/` (in Spanish).

**23 October 2026 DSGE update (5.7.0).** T02_A–D use nonlinear models without implicit solving, ordered `.mod` commands, continuation for transitions and common local/global solution diagnostics. T02_D retains the irreversible-investment and stochastic-volatility lessons added on 22 October. The eight updated notebooks pin the source code and helpers to tag `v5.7.0`; tag `course-2026A-5.5.0` retains the original edition. See the [additional validation](../validation/README.md).

To run T02_A–D with Python 3.12 or later, clone the tag and use a fresh virtual environment:

```bash
git clone --branch v5.7.0 --depth 1 https://github.com/jalonso1979/puremacro.git puremacro-5.7.0
cd puremacro-5.7.0
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: ./.venv/Scripts/Activate.ps1
python -m pip install -e ".[notebooks]" jupyterlab
cd curso/notebooks
jupyter lab
```

Select that environment as the kernel. Keep `_dsge_labs.py`, `modelos/` and `data_curso/` beside the notebooks. Recorded validation uses local Python; Colab installs the source from tag `v5.7.0`. This evidence does not certify a full Colab execution. The following instructions reproduce the complete original edition.

## Open a notebook in Google Colab

Click the notebook's link. The first code cell installs `puremacro` and fetches
the corresponding materials from the repository (a partial clone of the materials
and frozen data); it does not ask for Google Drive access. Run the cells in order.

| Code | Topic | Spanish | English |
|---|---|---|---|
| T00 | Interactive course syllabus | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T00_syllabus_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T00_syllabus.ipynb) |
| T00_B | Set up your local language model (Qwen3-4B) | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T00_B_configura_tu_modelo_local_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T00_B_local_model_setup.ipynb) |
| T01_A | Business-cycle facts from the labour side | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T01_A_data_panels_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T01_A_data_panels.ipynb) |
| T01_B | National accounts and cross-country stylized facts | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T01_B_measurement_and_cycles_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T01_B_measurement_and_cycles.ipynb) |
| T01_C | The cycle at monthly, weekly and daily frequency | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T01_C_stylized_facts_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T01_C_stylized_facts.ipynb) |
| T02_A | Neoclassical growth model: log-linearization and local approximation | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/v5.7.0/curso/notebooks/T02_A_neoclassical_growth_local_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/v5.7.0/curso/notebooks/T02_A_neoclassical_growth_local.ipynb) |
| T02_B | Canonical RBC: simulation, moments and the hours debate | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/v5.7.0/curso/notebooks/T02_B_rbc_dynare_local_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/v5.7.0/curso/notebooks/T02_B_rbc_dynare_local.ipynb) |
| T02_C | Sticky prices and search: frictions in goods and labour markets | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/v5.7.0/curso/notebooks/T02_C_new_keynesian_local_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/v5.7.0/curso/notebooks/T02_C_new_keynesian_local.ipynb) |
| T02_D | OccBin, irreversible investment and uncertainty shocks | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/v5.7.0/curso/notebooks/T02_D_occbin_zlb_regimes_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/v5.7.0/curso/notebooks/T02_D_occbin_zlb_regimes.ipynb) |
| T02_E | Bayesian DSGE estimation and model comparison | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T02_E_bayesian_dsge_local_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T02_E_bayesian_dsge_local.ipynb) |
| T02_F | Macro-financial frictions: Gertler–Karadi (2011) | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T02_F_gertler_karadi_macrofinance_local_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T02_F_gertler_karadi_macrofinance_local.ipynb) |
| T03_A | Aggregate shocks with incomplete markets: Krusell–Smith and sequence-space Jacobians | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T03_A_aggregate_shocks_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T03_A_aggregate_shocks.ipynb) |
| T03_B | Volatility: GARCH and leverage effects | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T03_B_garch_volatility_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T03_B_garch_volatility.ipynb) |
| T03_C | Growth-at-risk | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T03_C_growth_at_risk_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T03_C_growth_at_risk.ipynb) |
| T03_D | Economic policy uncertainty from text | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T03_D_narrative_uncertainty_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T03_D_narrative_uncertainty.ipynb) |
| T03_E | Local language models as measurement tools | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T03_E_local_llm_uncertainty_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T03_E_local_llm_uncertainty.ipynb) |
| T03_F | Central-bank communication and monetary-policy tone | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T03_F_narrative_llm_central_banks_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T03_F_narrative_llm_central_banks.ipynb) |
| T04_A | The tax multiplier, three ways | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T04_A_tax_multiplier_three_ways_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T04_A_tax_multiplier_three_ways.ipynb) |
| T04_C | Business-cycle accounting: the wedges | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T04_C_contabilidad_ciclo_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T04_C_business_cycle_accounting.ipynb) |
| T05_A | Identifying shocks with SVARs | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T05_A_svar_identification_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T05_A_svar_identification.ipynb) |
| T05_B | Local projections | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T05_B_local_projections_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T05_B_local_projections.ipynb) |
| T05_C | Staggered difference-in-differences | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T05_C_staggered_did_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T05_C_staggered_did.ipynb) |
| T05_D | Difference-in-differences with local projections (LP-DiD) | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T05_D_lp_did_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T05_D_lp_did.ipynb) |
| T05_E | State-dependent transmission and generalized impulse responses | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T05_E_regime_girf_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T05_E_regime_girf.ipynb) |
| T05_F | SVAR of the 2021–2024 inflation (Bernanke–Blanchard) | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T05_F_svar_post_covid_inflation_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T05_F_svar_post_covid_inflation.ipynb) |
| T05_G | Narrative identification of monetary shocks with a local LLM (Banxico and FOMC) | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T05_G_choques_narrativos_llm_banxico_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T05_G_narrative_llm_banxico.ipynb) |
| T06_A | Where does wealth inequality come from? | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T06_A_wealth_inequality_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T06_A_wealth_inequality.ipynb) |
| T06_B | Life cycle, demographics and pensions | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T06_B_life_cycle_and_demographics_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T06_B_life_cycle_and_demographics.ipynb) |
| T06_C | Portfolios, preferences and the HANK frontier | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T06_C_portfolios_and_preferences_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T06_C_portfolios_and_preferences.ipynb) |
| T07_A | Firm dynamics and misallocation | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T07_A_firm_dynamics_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T07_A_firm_dynamics.ipynb) |
| T08_A | Labour search and worker flows | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T08_A_labor_flows_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T08_A_labor_flows.ipynb) |
| T09_A | Numerical validation gallery | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T09_A_validation_gallery_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T09_A_validation_gallery.ipynb) |
| T09_B | Build your own index | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T09_B_build_your_own_index_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/course-2026A-5.5.0/curso/notebooks/T09_B_build_your_own_index.ipynb) |

## Run the notebooks locally

Requires Python 3.12 or later.

```bash
git clone --branch course-2026A-5.5.0 --depth 1 --filter=blob:none --sparse https://github.com/jalonso1979/puremacro.git
cd puremacro
git sparse-checkout set curso/notebooks
python -m pip install "puremacro[io]==5.5.0" statsmodels arch scikit-learn jupyterlab
cd curso/notebooks
jupyter lab
```

This edition pins `puremacro==5.5.0` and tag `course-2026A-5.5.0` for its helpers, data,
cache and models. `puremacro.dp` and `puremacro.text` are required and their
sections execute in full. All 72 notebooks include the historical T01 variants;
the table shows the main route.

Initial installation needs the network. Afterwards, notebooks use frozen files
in `data_curso/` and default to `MAV_LLM_MODE=cache`: Qwen3-4B reference responses,
without engine discovery or model downloads. T00_B also runs in this mode;
comparison with a live model and live data refreshes are optional.
For a local model, follow T00_B (`pip install "puremacro[local-llm]==5.5.0"` or
Ollama) and explicitly select `MAV_LLM_MODE=auto`.

## Data

The files in `data_curso/` are frozen copies of public third-party data (FRED,
OECD, INEGI, ILOSTAT, World Bank and others). The MIT licence of `puremacro`
covers the code, **not** that data. Where each file comes from, and how to cite
it, is in [`data_curso/FUENTES.md`](data_curso/FUENTES.md). If you use this data,
cite the original publishers.

## Licence

The code is MIT. The notebook text and the course slides are licensed under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/):
you may reuse and adapt them, commercially too, crediting
"Jorge Alonso-Ortiz, *Macroeconomía Avanzada*, ITAM". The data keep their publishers' terms.
