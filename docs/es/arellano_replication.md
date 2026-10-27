# Arellano: código original y sensibilidad a las mallas

**Se reproduce el algoritmo archivado; aún no se establece la convergencia de
los momentos económicos.** La comparación conserva las convenciones de cada
modelo y no recalibra parámetros para acercarse a los objetivos publicados.

## Ejecutar localmente

Desde el código de desarrollo, con Python 3.11+ y las dependencias habituales:

```sh
python -m pytest tests/test_arellano_2008_study.py -q
MPLBACKEND=Agg python -m tools.reference_validation.run_arellano_study --output /tmp/arellano-study
```

Se generan figuras PNG/PDF, `moments.csv` y `results.json`, con semillas,
intervalos, episodios, residuos y hashes del código. La ejecución habitual usa
referencias congeladas y no requiere red, gfortran ni Octave. El estudio tarda
unos minutos en la máquina probada.

## Verificación independiente

El Fortran del autor converge tras 49 iteraciones de precios. La implementación
independiente reproduce las 4.200 elecciones de deuda y default. Los precios
coinciden exactamente y los valores difieren como máximo en 5,3e-11. Además,
el bloque MATLAB original de agregación, ejecutado en Octave con las mismas
series y ventanas, coincide en los 12 momentos dentro de 1,1e-14. Esto verifica
código y medición; no certifica una replicación cuantitativa completa.

Fuentes: [artículo AER](https://doi.org/10.1257/aer.98.3.690),
[borrador público de julio de 2007](https://mpra.ub.uni-muenchen.de/7867/1/MPRA_paper_7867.pdf)
y [archivo de replicación](https://doi.org/10.3886/E113248V1). El borrador también
contiene los objetivos de la Tabla 4. Los archivos originales, licencias,
hashes y procedimientos de regeneración están en el directorio local
`tests/fixtures/arellano_2008/`.

## Objetivos publicados y malla original

Se usan 16 semillas independientes, 100.000 trimestres por semilla y 1.000 de
descarte. Cada réplica promedia 100 ventanas de 74 trimestres anteriores al
default, sin interrupciones del acceso al mercado. Los intervalos miden error
Monte Carlo de la **media entre réplicas**, no incertidumbre paramétrica ni la
dispersión de una sola réplica de 100 ventanas; esta última está en el JSON.

| Momento | Modelo publicado | Malla original 200 × 21 | Intervalo MC 95% |
|---|---:|---:|---:|
| Deuda/producto (%) | 5,95 | 6,283 | [6,083; 6,482] |
| Diferencial anual medio (%) | 3,58 | 3,405 | [3,362; 3,448] |
| Frecuencia anual de default (%) | 3,00 | 2,822 | [2,756; 2,887] |
| DE del diferencial (puntos porcentuales) | 6,36 | 6,009 | [5,891; 6,126] |
| DE balanza comercial/producto (pp) | 1,50 | 1,485 | [1,463; 1,508] |
| DE log consumo (%) | 6,38 | 6,391 | [6,323; 6,459] |
| DE log producto (%) | 5,81 | 5,826 | [5,754; 5,898] |
| Correlación diferencial, producto | -0,29 | -0,251 | [-0,265; -0,236] |
| Correlación balanza comercial, producto | -0,25 | -0,245 | [-0,252; -0,239] |
| Correlación consumo, producto | 0,97 | 0,971 | [0,970; 0,971] |
| Correlación balanza comercial, diferencial | 0,43 | 0,432 | [0,423; 0,441] |
| Correlación consumo, diferencial | -0,36 | -0,331 | [-0,347; -0,316] |

Persisten diferencias respecto a la media estimada. No están disponibles los
sorteos originales y el script archivado examina candidatos a default 2:300,
mientras el artículo describe 100 ventanas. No se afirma una reproducción
exacta de la tabla publicada.

## La convergencia económica sigue pendiente

Las mallas de deuda son anidadas, con extremos fijos y cero exacto; las mallas
de ingreso mantienen sus extremos logarítmicos. Los residuos independientes de
Bellman son inferiores a 4e-12 y los de precios a 1e-8. Ninguna elección simulada
de deuda con repago alcanza los extremos. Aun así, los momentos cambian:

| Activos × estados de ingreso | Deuda/producto (%) | Diferencial medio (%) | Default anual (%) | DE diferencial (pp) |
|---|---:|---:|---:|---:|
| 200 × 21, Fortran original | 6,283 | 3,405 | 2,822 | 6,009 |
| 399 × 21 | 6,322 | 3,248 | 2,663 | 5,402 |
| 797 × 21 | 6,347 | 3,320 | 2,740 | 5,348 |
| 797 × 41 | 6,010 | 4,043 | 3,558 | 4,442 |
| 797 × 81 | 5,491 | 3,504 | 3,080 | 3,477 |
| 797 × 161 | 5,246 | 3,370 | 2,957 | 3,230 |
| 797 × 321 | 5,144 | 3,321 | 2,942 | 3,098 |

Las filas de 161 y 321 estados provienen de
`run_arellano_study --income-states 41 81 161 321`, con las mismas semillas,
ventanas, calibración y reingreso archivados. Residuos pequeños y ausencia de
masa en los extremos no establecen convergencia económica. De 81 a 321 estados
la desviación del diferencial sigue cayendo, la deuda/producto baja y la
correlación entre balanza comercial y diferencial se estabiliza cerca de 0,77
frente al 0,43 publicado. La volatilidad publicada del diferencial, 6,36 puntos
porcentuales, es por tanto la que produce la malla archivada de 200 × 21 con
esta calibración; en mallas finas el mismo modelo genera cerca de la mitad. Es
el patrón documentado por Hatchondo, Martinez y Sapriza (2010): la técnica de
espacio de estados discreto con mallas equiespaciadas necesita muchos puntos
para evitar movimientos espurios de la tasa de interés. Acercarse a un objetivo
aislado no demuestra convergencia conjunta, y aquí no se reajusta ningún
parámetro.

Hatchondo, J. C., L. Martinez y H. Sapriza (2010), "Quantitative properties of
sovereign default models: solution methods matter", *Review of Economic
Dynamics* 13(4), 919–933, [doi:10.1016/j.red.2010.03.001](https://doi.org/10.1016/j.red.2010.03.001).

## Convenciones del modelo y de medición

El archivo limita el ingreso en default a `10*.969`, aunque el ingreso medio
es aproximadamente 10,0307. PureMacro normaliza la media estacionaria y aplica
el límite a esa media. Los datos archivados coinciden con Tauchen a la precisión
impresa, aunque el borrador menciona Tauchen–Hussey. La malla de activos llega
a 1,53502537, no a su extremo nominal de 1,5.

El archivo impone un período adicional de exclusión después de entrar en
default. Cambiar únicamente esa convención por la ecuación canónica reduce
deuda/producto de 6,283% a 5,440% y eleva el diferencial medio de 3,405% a 4,184%.
La calibración separada de la API pública, con 321 × 21 puntos, produce 5,055%
y 4,329%, respectivamente. También cambia normalización, malla y precisión;
por tanto no aísla solamente el efecto del momento de reingreso.

El estudio usa diferenciales compuestos `q**-4 - Rf**4`, desviaciones muestrales
por ventana y correlaciones sin ajustes. La frecuencia comparable al archivo
usa períodos de repago como denominador; el JSON también presenta el riesgo
con períodos de entrada al default incluidos en la exposición. Numerador y
denominador usan la misma muestra posterior al descarte. No se usan los
diagnósticos simulados breves de la API pública. Su antiguo bucle conservaba
un período adicional respecto a la ecuación canónica de Bellman. Ese error ya
está corregido: el reingreso permite decidir en el mismo trimestre. Las dos
interfaces públicas restablecen deuda cero y coinciden con la misma semilla.
El simulador común del estudio respeta la ecuación de Bellman de cada caso.

## Próxima decisión

Las convenciones públicas están documentadas en la [guía soberana](sovereign_default.md).
Ahora resolver la sensibilidad
de malla con criterios documentados antes de recalibrar o interpretar efectos
de política. Los siguientes experimentos son reingreso, volatilidad y persistencia
del ingreso; vencimientos y reestructuración requieren validación adicional.

## Comprobar las mallas de la calibración pública

```sh
MPLBACKEND=Agg python -m tools.reference_validation.check_arellano_grids --output /tmp/arellano-grids
```

Se resuelven ecuaciones independientes, contrastadas con la API pública en una
malla común de 401 × 9. Calibración, extremos, momento de reingreso y medición
permanecen fijos. La ejecución predeterminada compara 401/801 activos y 321/641
estados de ingreso, con 16 semillas emparejadas por caso.

El cambio de cada momento porcentual debe ser como máximo 0,10 puntos
porcentuales; el de cada correlación, 0,02. Se suma el cambio medio absoluto a
la semiamplitud del intervalo Student-t de 95% de las diferencias emparejadas.
Deben aprobar tanto el refinamiento final de deuda como el de ingreso, con al
menos 797 activos y 321 estados de ingreso en la malla fina. Son tolerancias
numéricas, no ajustes a objetivos publicados. Un código de salida distinto de
cero distingue cambios grandes de precisión Monte Carlo insuficiente. Aprobar
la base no valida nuevos parámetros, políticas ni deuda a largo plazo.

El informe registra semillas, hashes, residuos y comparaciones. Las soluciones
verificadas se reutilizan con `--periods 500000 --windows 1000` para aumentar
precisión manteniendo semillas. Los resultados archivados anteriores conservan
su significado histórico para otra calibración y convención de reingreso.
