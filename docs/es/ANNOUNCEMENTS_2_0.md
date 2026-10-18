> 🇬🇧 [English](../ANNOUNCEMENTS_2_0.md) · 🇪🇸 Español

# Anuncios de Lanzamiento de puremacro 2.0.0 y Kit de Difusión Comunitaria

Este documento proporciona plantillas de anuncio de lanzamiento listas para publicar para **puremacro 2.0.0**, adaptadas a distintas comunidades académicas, redes sociales y desarrolladores.

---

## 1. Hilo para EconTwitter / X / Bluesky 🧵

### Publicación 1 (Gancho y Propuesta de Valor)
> 🚀 Nos entusiasma anunciar **puremacro 2.0.0** — ¡el kit de herramientas de macroeconomía empírica y series de tiempo 100% en Python puro y sin compiladores de C!
>
> 📦 `pip install puremacro`
> 🌐 Listo para WebAssembly y iPad / Juno
> 📚 Documentación: https://jalonso1979.github.io/puremacro/
>
> Desde Proyecciones Locales hasta Diferencias en Diferencias (DiD) escalonadas de frontera, esto es lo nuevo 👇 (1/6)

### Publicación 2 (Econometría Micro/Macro de Frontera)
> 🔬 **Identificación de Frontera y DiD**:
> • Callaway-Sant'Anna (2021) y Sun-Abraham (2021)
> • Borusyak-Jaravel-Spiess (2022) y DiD Sintético
> • **Análisis de Sensibilidad Honest DiD** (Rambachan y Roth 2023): evalúa violaciones de tendencias paralelas con cotas de magnitud relativa y suavidad + valores de ruptura $M^*$.
>
> Todo vectorizado en NumPy puro. (2/6)

### Publicación 3 (Series de Tiempo y Macro Estructural)
> 📈 **Series de Tiempo Macro e Identificación Estructural**:
> • Proyecciones Locales con aumento de rezagos de Montiel Olea-Plagborg-Møller (2021) y GARCH-IV
> • Proyecciones Locales dependientes del estado y de transición suave
> • VARs Aumentados por Factores (FAVAR — Bernanke, Boivin y Eliasz 2005)
> • **Perturbación DSGE de 2.º Orden con Poda (Pruning)** (Kim, Kim, Schaumburg y Sims 2008): ¡estrictamente estable, GIRFs asimétricas y medias ergódicas! (3/6)

### Publicación 4 (Cero Fricción y Movilidad en Tableta/Navegador)
> 📱 **Verdadera Movilidad Académica**:
> Sin Fortran, sin pasos de compilación en C++, sin errores de incompatibilidad de BLAS/LAPACK.
>
> Se ejecuta al 100% dentro de tu navegador (Pyodide / JupyterLite) y de forma nativa en iPad (Juno). ¿Necesitas ejecutar un MCMC de 10,000 extracciones? ¡Una sola línea genera un cuaderno autocontenido de Google Colab con autenticación en la nube y datos integrados! (4/6)

### Publicación 5 (Flujo de Publicación y Piedra Rosetta)
> 📑 **Del Código al Artículo en Segundos**:
> • Exportación instantánea de tablas a LaTeX (`\begin{tabular}`) y Typst con estrellas de significancia académica (`***`, `**`, `*`).
> • ¿Vienes de Stata o MATLAB? Incluimos una guía completa de **Piedra Rosetta** (`reg`, `reghdfe`, `var`, `csdid` $\to$ puremacro). (5/6)

### Publicación 6 (Cómo Empezar)
> Prueba hoy mismo el entorno interactivo en el navegador sin necesidad de instalación:
> 🔗 Playground: https://jalonso1979.github.io/puremacro/playground.html
> 💻 GitHub: https://github.com/jalonso1979/puremacro
>
> ¡Comentarios, reportes de errores y contribuciones son muy bienvenidos! #EconTwitter #Python #Econometrics #Macroeconomics (6/6)

---

## 2. Publicación para LinkedIn (Economistas Académicos y de Banca Central)

> 📊 **Anunciando puremacro 2.0.0: Macroeconomía Empírica Moderna en el Ecosistema de Python Puro**
>
> En la investigación macroeconómica y el análisis de políticas públicas, los flujos de trabajo empíricos suelen enfrentar fricciones de herramientas: dependencias binarias pesadas, obstáculos de compilación específicos de cada plataforma y bases de código fragmentadas entre MATLAB, Stata y scripts a medida.
>
> Hoy lanzamos **puremacro 2.0.0**, diseñado desde los primeros principios alrededor de una arquitectura sin dependencias en C. Si Python y NumPy funcionan en tu dispositivo, puremacro funciona de inmediato — en clústeres Linux, macOS, Windows, iPad (vía Juno) y navegadores del lado del cliente (vía WebAssembly / Pyodide).
>
> 🌟 **Aspectos Destacados de la Versión 2.0.0**:
> 1. **Inferencia Causal de Frontera y Diferencias en Diferencias (DiD)**:
>    - Conjunto completo de estimadores DiD escalonados robustos a la heterogeneidad: Callaway y Sant'Anna (2021), Sun y Abraham (2021), Borusyak, Jaravel y Spiess (2022), y DiD Sintético (Arkhangelsky et al. 2021).
>    - **Análisis de Sensibilidad Honest DiD** (Rambachan y Roth 2023): evaluación de desviaciones respecto a tendencias paralelas mediante restricciones de magnitud relativa y suavidad, calculando conjuntos de confianza robustos (Imbens y Manski 2004) y valores de ruptura $M^*$.
>
> 2. **Macroeconomía Estructural y Series de Tiempo**:
>    - Proyecciones Locales (Jordà 2005) con aumento de rezagos de Montiel Olea y Plagborg-Møller (2021) e inferencia HAC robusta.
>    - Proyecciones Locales dependientes del estado y VARs Aumentados por Factores (FAVAR, Bernanke et al. 2005).
>    - Solucionador de expectativas racionales QZ de Klein (2000) y **perturbación DSGE de segundo orden con poda (pruning)** (Kim, Kim, Schaumburg y Sims 2008), eliminando trayectorias cuadráticas explosivas y proporcionando funciones de impulso-respuesta generalizadas asimétricas.
>
> 3. **Reportes Listos para Publicación**:
>    - Exportación directa de resultados de estimación a tablas académicas en LaTeX y Typst con estrellas de significancia convencionales (`* p<0.10`, `** p<0.05`, `*** p<0.01`).
>
> 4. **Puente de Cómputo en la Nube**:
>    - Transición fluida desde cuadernos de trabajo en dispositivos móviles o tabletas hacia aceleradores en Google Colab sin fricción de carga de archivos mediante cuadernos autocontenidos con datos integrados.
>
> 🔗 **Playground Interactivo en el Navegador**: https://jalonso1979.github.io/puremacro/playground.html
> 📖 **Documentación Completa**: https://jalonso1979.github.io/puremacro/
> 📦 **PyPI**: `pip install puremacro`
> 💻 **GitHub**: https://github.com/jalonso1979/puremacro
>
> Esperamos con entusiasmo ver las aplicaciones de investigación empírica y docencia que esta versión hará posibles.

---

## 3. Publicación para Foros y Discourse de QuantEcon / PyData

**Título**: [ANN] puremacro 2.0.0 — Kit de Herramientas de Macroeconomía Empírica y DiD de Frontera en Python Puro Compatible con Pyodide

**Cuerpo**:
> Hola a todos:
>
> ¡Nos entusiasma anunciar el lanzamiento de **puremacro 2.0.0**!
>
> ### ¿Por qué puremacro?
> La mayoría de los paquetes macroeconómicos en Python dependen de complejas matrices de compilación en C/Fortran/Cython (que fallan en entornos ligeros, WebAssembly o tabletas) o carecen de los métodos modernos de identificación posteriores a 2020.
>
> `puremacro` fue construido para cubrir esta brecha con tres reglas fundamentales de diseño:
> 1. **Cero extensiones compiladas en C**: Únicamente biblioteca estándar, NumPy y SciPy.
> 2. **Compatibilidad estricta con Pyodide**: La suite completa de estimadores se ejecuta del lado del cliente en el navegador mediante WebAssembly sin necesidad de un servidor backend.
> 3. **Métodos econométricos de frontera**: Soporte de primera clase para técnicas modernas de macroeconomía y datos de panel.
>
> ### ¿Qué incluye la versión 2.0.0?
> - **Proyecciones Locales**: Jordà (2005), aumento de rezagos de Montiel Olea y Plagborg-Møller (2021), instrumentos de choques GARCH-IV y transición suave (STLP).
> - **Vectores Autorregresivos**: VAR(p), FAVAR (Bernanke, Boivin y Eliasz 2005), descomposición histórica y descomposición de la varianza del error de pronóstico (FEVD).
> - **DiD Escalonado Moderno**:
>   - Callaway y Sant'Anna (2021)
>   - Sun y Abraham (2021)
>   - Borusyak, Jaravel y Spiess (2022)
>   - Diferencias en Diferencias Sintéticas (SDID)
>   - **Sensibilidad Honest DiD** (Rambachan y Roth 2023) ante violaciones de tendencias paralelas
> - **Modelado DSGE**:
>   - Solucionador QZ de Klein (2000) con Jacobianos automáticos de paso complejo (¡sin derivación manual!)
>   - Perturbación de segundo orden con poda (Kim et al. 2008)
> - **Reportes y Flujos de Trabajo**:
>   - Exportación directa a tablas en LaTeX y Typst con estrellas de significancia (`***`, `**`, `*`).
>   - Guía Piedra Rosetta de Stata a puremacro para economistas que migran desde archivos `.do`.
>   - Puente de descarga a la nube hacia Google Colab para usuarios de iPad / Juno.
>
> ### Pruébalo
> - Playground Interactivo en WebAssembly: https://jalonso1979.github.io/puremacro/playground.html
> - Guía de Inicio Rápido: https://jalonso1979.github.io/puremacro/getting_started.html
> - Repositorio en GitHub: https://github.com/jalonso1979/puremacro
>
> ¡Nos encantaría conocer tus comentarios, reportes de errores e ideas para futuras extensiones!
