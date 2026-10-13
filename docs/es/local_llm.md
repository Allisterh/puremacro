> 🇬🇧 [English](../local_llm.md) · 🇪🇸 Español

# Modelos de lenguaje locales: iPad, Mac, Windows, Linux y Colab

`puremacro.llm` ejecuta un modelo de lenguaje pequeño en su propio dispositivo, sin clave de API
y sin red después de una única descarga. `complete(prompt)` devuelve la respuesta del modelo y
elige el motor por sí mismo: MLX, llama.cpp, Ollama o un servidor compatible con OpenAI. Cada
respuesta se guarda en una caché de completados que no depende del dispositivo ni del motor, de
modo que una respuesta calculada una vez, en una Mac, se vuelve a leer en un iPad, en Windows, en
Colab o en el navegador, donde no corre ningún modelo. Las funciones narrativas con LLM
(`score_llm`, `llm_prob_kernel`) pasan por la misma puerta de entrada.

```python
# requires: a local LLM engine (mlx-lm, llama-cpp-python, or a running Ollama / OpenAI-compatible server)
import puremacro.llm as llm

print(llm.complete("In one sentence, what is a local projection?", max_tokens=64))
print(llm.chat([{"role": "system", "content": "Answer in Spanish."},
                {"role": "user", "content": "What is a Taylor rule?"}], max_tokens=128))
```

`puremacro.llm` es posterior a puremacro 4.8.1 (véase el CHANGELOG, Unreleased): hace falta una
versión que lo incluya. `import puremacro.llm` no importa ningún paquete de inferencia ni abre
ninguna conexión; cada motor se importa solo cuando se usa.

## Qué corre dónde

| Plataforma | Motor | Usted instala | El modelo viene de |
|---|---|---|---|
| iPad, app Juno | MLX; llama.cpp como respaldo | la app Juno; puremacro con el gestor de paquetes de Juno | Hugging Face: mlx-lm lo descarga en el primer uso y queda en el contenedor de la app |
| Mac (Apple silicon) | MLX, si no Ollama | `pip install "puremacro[local-llm]"`, o la app de Ollama | lo mismo, o `ollama pull qwen3:4b` |
| Windows, Linux | Ollama, o llamafile | el instalador de Ollama, o un llamafile ejecutado como servidor local | la etiqueta de Ollama, o el GGUF que trae el llamafile |
| Colab | llama.cpp en la GPU, u Ollama dentro del cuaderno | nada; lo hace una celda de configuración | un archivo GGUF de Hugging Face |
| Navegador, JupyterLite | ninguno | nada | solo la caché distribuida |

`engine="auto"` (el valor por defecto) toma el primero disponible de: `mlx` (cuando `mlx` y
`mlx_lm` se pueden importar), `llamacpp` (cuando `llama_cpp` se puede importar), Ollama en
`http://localhost:11434` y luego servidores compatibles con OpenAI en `http://localhost:8080`
(llamafile, llama-server) y `http://localhost:1234` (LM Studio). Decide lo que está instalado,
nunca el sistema operativo ni la familia de procesador; por eso en el iPad se elige MLX. Un
servidor cuenta como disponible cuando su lista de modelos (`/api/tags` o `/v1/models`) responde
con JSON en menos de tres segundos. Pase `engine="mlx"`, `"llamacpp"`, `"ollama"` u `"openai"`
para fijar uno.

Lo comprobado, al 2026-10-12:

- En una Mac (M3 Pro), con el modelo Qwen3 1.7B: MLX, llama-cpp-python (compilado desde el código
  fuente con Metal) y el servidor compatible con OpenAI de llama-cpp-python respondieron sin
  rastro del razonamiento en el texto, para el prompt de índices y el de extracción de eventos
  (cuyas respuestas escriben `"sign": +1`, que `score_llm` ya interpreta).
- Ollama: la petición está lista (`"think": false`), pero aún no se ha probado con un modelo
  Qwen3 descargado.
- iPad (app Juno, iPad Pro M4, 16 GB): `mlx`, `mlx_lm`, `llama_cpp` y `huggingface_hub` se
  importan y Hugging Face es accesible; un completado con el mlx-lm de Juno **aún no se ha
  verificado en el dispositivo**.
- Colab: las recetas con la rueda CUDA de llama.cpp y con Ollama dentro del cuaderno **aún no se
  han verificado en el dispositivo**.

## Instalación por plataforma

**iPad.** Instale la app Juno y, dentro de ella, puremacro con el gestor de paquetes de Juno
(Juno no tiene `pip`). Juno incluye MLX y mlx-lm, así que no hace falta nada más. El primer
completado descarga el modelo; véase [En el iPad](#en-el-ipad).

**Mac.** O bien los motores de Python:

```bash
pip install "puremacro[local-llm]"     # mlx-lm (macOS only), llama-cpp-python and huggingface_hub
```

o bien la app de Ollama (https://ollama.com) y después `ollama pull qwen3:4b`. Si están los dos,
puremacro elige MLX. El extra `[local-llm]` compila llama-cpp-python desde el código fuente, lo
que requiere las herramientas de línea de comandos de Xcode. No use la rueda Metal precompilada
de llama-cpp-python 0.3.36 del índice adicional del proyecto: está corrupta (la mitad de sus
archivos falla la comprobación CRC). Para compilar llama.cpp con Metal de forma explícita:

```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --no-binary llama-cpp-python
```

Si esa compilación falla, prescinda de llama.cpp: `pip install puremacro mlx-lm` basta para MLX,
y Ollama no necesita ningún paquete de Python.

**Windows y Linux.** Instale Ollama (https://ollama.com) y ejecute `ollama pull qwen3:4b`. O
descargue un llamafile y ejecútelo como servidor local; escucha en el puerto 8080 con una API
compatible con OpenAI, que `engine="auto"` encuentra. `pip install "puremacro[local-llm]"`
también funciona donde llama-cpp-python compila (hacen falta un compilador de C y CMake).

**Colab.** Elija un entorno con GPU (Entorno de ejecución → Cambiar tipo de entorno → GPU T4) y,
en una sola celda:

```bash
# requires: Colab with a GPU runtime (not yet verified on device)
pip install -q --only-binary=llama-cpp-python llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
pip install -q huggingface_hub puremacro
```

La rueda CUDA pesa unos 1.8 GB; `--only-binary` hace que la celda falle pronto en lugar de
iniciar una larga compilación solo para CPU. El motor descarga todas las capas en la GPU
(`n_gpu_layers=-1`). La otra vía es Ollama dentro del cuaderno: instálelo con el script de
instalación de Ollama para Linux, arranque `ollama serve` en segundo plano y ejecute
`ollama pull qwen3:4b`; `engine="auto"` lo encuentra entonces en el puerto 11434.

**Navegador (JupyterLite, juno.sh).** Bajo Pyodide no corre ningún motor. Distribuya una caché
(véase [La caché y cómo distribuirla](#la-cache-y-como-distribuirla)) y use el modo `cache`.
Pyodide 0.28 añade `sqlite3` solo si se pide: ejecute `await pyodide.loadPackage('sqlite3')`
antes de importar puremacro.

## La escalera de modelos

Un único modelo de referencia para todo, `qwen3-4b`: las cachés distribuidas se construyen con
él y `default` lo designa. Los tres peldaños tienen licencia Apache-2.0.

| Alias | Nivel | Parámetros | Memoria, aprox. | Presupuesto mínimo | Licencia | Elegido automáticamente |
|---|---|---|---|---|---|---|
| `qwen3-1.7b` | `small` | 1.7 B | 2.0 GB | 3 GB | apache-2.0 | con menos de 5 GB de presupuesto |
| `qwen3-4b` | `default` | 4 B | 3.5 GB | 5 GB | apache-2.0 | desde 5 GB |
| `qwen3-8b` | `large` | 8 B | 6.0 GB | 10 GB | apache-2.0 | nunca; pídalo por su nombre |

Lo que carga cada motor (`puremacro.llm.resolve_model_id(alias, engine)`):

| Alias | MLX (Hugging Face) | llama.cpp (repositorio, archivo) | Etiqueta de Ollama |
|---|---|---|---|
| `qwen3-1.7b` | `mlx-community/Qwen3-1.7B-4bit` | `ggml-org/Qwen3-1.7B-GGUF`, `Qwen3-1.7B-Q4_K_M.gguf` | `qwen3:1.7b` |
| `qwen3-4b` | `mlx-community/Qwen3-4B-4bit` | `Qwen/Qwen3-4B-GGUF`, `Qwen3-4B-Q4_K_M.gguf` | `qwen3:4b` |
| `qwen3-8b` | `mlx-community/Qwen3-8B-4bit` | `Qwen/Qwen3-8B-GGUF`, `Qwen3-8B-Q4_K_M.gguf` | `qwen3:8b` |

Las descargas son de unos 1.0, 2.3 y 4.6 GB para MLX, y de unos 1.3, 2.5 y 5 GB para los
archivos GGUF y las etiquetas de Ollama. Los servidores compatibles con OpenAI reciben el propio
alias como `model` (`"qwen3-4b"`); llamafile y llama-server responden con el modelo con el que se
arrancaron.

- **El presupuesto** (`device_budget_gb()`) es la memoria que puede usar un modelo: el conjunto
  de trabajo que MLX recomienda para la GPU cuando mlx está instalado; si no, el 60 % de la
  memoria física; si no, 8 GB supuestos. `pick_model()` lo convierte en un modelo; `check_fits`
  rechaza un modelo cuyo presupuesto mínimo el dispositivo no alcanza, y `complete()` lanza
  entonces `ModelTooLarge` (con `alias`, `suggested` y `budget_gb`) antes de buscar ningún motor.
  Una respuesta en caché se sirve sea cual sea el dispositivo.
- **Nombres.** `model=` acepta un nivel, un alias o el identificador propio de un motor (una
  etiqueta de Ollama, una ruta `.gguf`, un repositorio de Hugging Face), que llega al motor sin
  cambios y no se compara con el presupuesto. `PUREMACRO_LLM_MODEL` cambia lo que significa
  `default`, y nada más.
- **Razonamiento.** Los modelos Qwen3 razonan antes de responder. Los motores lo desactivan (MLX
  y llama.cpp mediante la plantilla de chat, Ollama con `"think": false`, los servidores
  compatibles con OpenAI con `chat_template_kwargs` y un sufijo `/no_think`), y `strip_thinking`
  elimina cualquier bloque `<think>…</think>` que aún llegue. Un modelo que razona recibe 8 tokens
  más que `max_tokens` en el motor, espacio para el bloque de razonamiento vacío con el que puede
  empezar.
- **Los modelos de 4.8.1** siguen con sus nombres, para el código que los usa:
  `qwen2.5-3b-instruct` (licencia `qwen-research`), `gemma2-2b` (`gemma`), `llama3.2-3b`
  (`llama3.2`) y `phi3.5` (`mit`). Nunca se eligen automáticamente.

## Modos

El argumento `mode`; si no, la variable de entorno `PUREMACRO_LLM_MODE`; si no, `auto`
(`puremacro.llm.current_mode()` dice cuál se aplica):

| Modo | Lee la caché | Ejecuta un modelo | Escribe la caché | Si la caché no tiene la respuesta |
|---|---|---|---|---|
| `auto` | sí | si falta la respuesta | sí (respuestas no vacías) | comprueba el presupuesto, busca un motor, ejecuta el modelo |
| `cache` | sí | nunca | no | lanza `CacheMiss` |
| `off` | no | siempre | no | (no se usa la caché) |

En modo `cache` no se construye ningún motor, no se abre ningún socket y no se carga ningún
modelo. Una caché que no se puede leer en absoluto (sin `sqlite3`, un archivo que no es una
caché, como un puntero de Git LFS) lanza `CacheUnavailable`, un tipo de `CacheMiss` cuyo
`reason` explica el motivo. El modo `off` ni abre ni crea un archivo de caché. Sin ningún motor
disponible, `auto` y `off` lanzan `LocalLLMUnavailable` con instrucciones de instalación.

Las funciones narrativas siguen el modo:

- `LocalBackend` (para `score_llm`) y `LocalProvider` (para `llm_prob_kernel`) aceptan
  `model="default"`, `engine="auto"`, `base_url=None`, `timeout=120.0` y `mode=None`, no buscan
  ningún motor al construirse y envían cada llamada a través de `puremacro.llm.complete`. Así, un
  corpus puntuado una vez se vuelve a puntuar sin ningún motor instalado.
- `get_default_backend()` y `get_default_provider()` imprimen una línea de estado:

  ```text
  [puremacro] local LLM: mode=cache engine=cache model=qwen3-4b
  ```

  En modo `cache` devuelven el objeto local sin sondear nada. En los modos `auto` y `off`
  consultan `puremacro.llm.engines.available_engines(base_url=…)`; si no aparece ningún motor (o
  no el indicado), imprimen la indicación de instalación, muestran `engine=mock` y devuelven
  `MockBackend` (cero eventos) o `MockProvider`, para que las demostraciones siempre corran.
- En modo `cache`, `score_llm` lanza `CacheMiss` (o `CacheUnavailable`) para un registro que la
  caché no tiene, en lugar de descartarlo: un corpus que no se puntuó de antemano falla de forma
  visible en vez de devolver una lista más corta. `ModelTooLarge` también se propaga. Los demás
  fallos de la llamada de un registro (un tiempo de espera agotado, por ejemplo) se siguen
  contando como malformados y el bucle continúa.
- `OllamaBackend()` y `OllamaProvider()` usan por defecto el alias `qwen3-4b`: el servidor recibe
  su etiqueta `qwen3:4b` y las filas de caché son las de `LocalBackend("default")`. Una etiqueta
  explícita, `OllamaBackend("qwen3:4b")`, llega al servidor sin cambios y entonces indexa la caché
  por la etiqueta, de modo que sus filas no se comparten con el alias.

```python
from puremacro.narrative.scoring import get_default_backend

backend = get_default_backend(mode="cache")   # probes nothing; prints the status line
print(type(backend).__name__, backend.model)  # LocalBackend qwen3-4b
```

## La caché y cómo distribuirla

Un único archivo SQLite, `~/.cache/puremacro/llm_cache.sqlite`, o el que indique
`PUREMACRO_LLM_CACHE`. Cada fila se indexa (formato `v1`) por el alias canónico (`qwen3-4b`,
nunca el identificador de un motor), el prompt y el prompt de sistema tal como usted los
escribió (antes de cualquier plantilla de chat), `max_tokens`, `temperature` (con tres
decimales) y `json_mode`. El motor se registra solo como procedencia, así que una fila escrita
por MLX responde a la misma llamada hecha con llama.cpp u Ollama. La función de clave es fija
durante toda la vida del formato; las pruebas fijan su valor de referencia:
`cache_key("qwen3-4b", "hola", max_tokens=8, temperature=0.0, json_mode=False)` es
`c630a581ab2b323492ac1db08bfc8b8f5a72655db00282142eb5762eacccb586`.

Para distribuir una: ejecute los cuadernos o scripts una vez en modo `auto` en una máquina con el
modelo de referencia, exporte las filas y entregue el archivo. Del lado receptor, apunte
`PUREMACRO_LLM_CACHE` a él y fije `PUREMACRO_LLM_MODE=cache`, o combínelo con su propia caché.
Una caché en un volumen de solo lectura sigue sirviendo respuestas.

```python
from puremacro.llm import CacheMiss, CacheRow, CompletionCache, cache_key, complete
from puremacro.llm.cache import prompt_sha256

# A stand-in for a cache built on a machine that runs the model: one row written by hand
built = CompletionCache("built_on_a_mac.sqlite")
prompt = "Does 'the outlook remains highly uncertain' express uncertainty? Answer yes or no."
key = cache_key("qwen3-4b", prompt, max_tokens=8, temperature=0.0, json_mode=False)
built.set(CacheRow(key, "qwen3-4b", prompt_sha256(prompt), 8, 0.0, False, "yes",
                   engine="mlx", engine_model_id="mlx-community/Qwen3-4B-4bit"))

# Ship the rows of the reference model in a file of their own
print(built.export("course_cache.sqlite", model="qwen3-4b"))   # 1 row copied

# Read it anywhere, with no engine installed: cache mode never loads a model
shipped = CompletionCache("course_cache.sqlite")
print(complete(prompt, model="qwen3-4b", max_tokens=8, mode="cache", cache=shipped))   # yes
try:
    complete("A prompt nobody ran", model="qwen3-4b", max_tokens=8, mode="cache", cache=shipped)
except CacheMiss as exc:
    print(type(exc).__name__, exc.model)                        # CacheMiss qwen3-4b
print(shipped.stats()["by_model"])                              # {'qwen3-4b': 1}

# Or add the shipped rows to a cache of your own (rows it already has are kept)
print(CompletionCache("my_cache.sqlite").merge("course_cache.sqlite"))   # 1
```

- `export(dest, *, model=None, since=None)` copia filas a otro archivo y lo crea si hace falta;
  `since="2026-10-01"` conserva las filas creadas en esa fecha o después (UTC). `merge(src)`
  añade las filas de otro archivo que a este le faltan. Ambos devuelven el número de filas
  añadidas y lanzan una excepción si no pueden hacer su trabajo; `stats()` da `rows`,
  `by_model`, `bytes` y `path`.
- Una respuesta en caché exige la misma llamada: el mismo alias, el mismo prompt, `max_tokens`,
  `temperature` y `json_mode`. `PUREMACRO_LLM_MODEL` cambia el alias que representa `default`, y
  una etiqueta explícita de Ollama indexa las filas por la etiqueta.
- Una caché en la que no se puede escribir (un directorio de solo lectura, un archivo dañado)
  nunca detiene un completado: las respuestas se recalculan y un aviso por archivo explica por
  qué.
- `llm_prob_kernel` mantiene su propia caché de puntuaciones delante de esta
  (`~/.cache/puremacro/llm_scores.sqlite`, o `PUREMACRO_LLM_SCORES_CACHE`). Basta con distribuir
  la caché de completados: una puntuación que falta allí pasa a la caché de completados.

## Variables de entorno

| Variable | Efecto |
|---|---|
| `PUREMACRO_LLM_MODE` | `auto`, `cache` u `off` cuando no se pasa `mode=`; vacía cuenta como no fijada, cualquier otro valor lanza `ValueError` |
| `PUREMACRO_LLM_MODEL` | lo que significa `default`: un nivel, un alias o el identificador propio de un motor; no afecta a `small`, `large` ni a los alias explícitos |
| `PUREMACRO_LLM_CACHE` | el archivo de la caché de completados (por defecto `~/.cache/puremacro/llm_cache.sqlite`) |
| `PUREMACRO_LLM_BASE_URL` | un servidor que sustituye a los tres servidores locales por defecto; se prueba como Ollama y luego como compatible con OpenAI |
| `PUREMACRO_LLM_SCORES_CACHE` | la caché de puntuaciones de `llm_prob_kernel` (por defecto `~/.cache/puremacro/llm_scores.sqlite`) |
| `PUREMACRO_LLM_BUDGET` | `spend` elimina el tope de llamadas (`max_calls`) de `llm_prob_kernel` |
| `HF_HUB_OFFLINE` | `1` hace que llama.cpp cargue sin red un archivo GGUF ya descargado en la caché de Hugging Face |

A la URL de un servidor, venga de `base_url=` o de `PUREMACRO_LLM_BASE_URL`, se le quitan las
barras finales y luego un `/v1` final. Así, `http://localhost:1234/v1`, la forma que documentan
LM Studio y los clientes de OpenAI, designa el mismo servidor que `http://localhost:1234`.

## En el iPad

- **Mantenga Juno en primer plano** mientras se descarga un modelo y mientras se puntúa un
  corpus. iPadOS suspende las apps en segundo plano, y con ellas se detienen la descarga o el
  bucle de puntuación. Las puntuaciones son cortas; divida un corpus largo en partes, y la caché
  conserva cada respuesta ya calculada.
- **Los modelos persisten en el contenedor de la app.** El primer completado descarga el modelo
  de Hugging Face (unos 2.3 GB para `qwen3-4b` con MLX, cerca de 1 GB para `qwen3-1.7b`) al
  contenedor de la app, donde lo encuentran las sesiones posteriores: la descarga ocurre una vez.
- **El presupuesto.** En un iPad Pro M4 con 16 GB, MLX recomienda un conjunto de trabajo de unos
  11 GB, así que `pick_model()` elige `qwen3-4b`, y `qwen3-8b` cabe si se pide por su nombre. Los
  dispositivos con menos memoria pasan a `qwen3-1.7b`, cuyas respuestas pueden diferir de las del
  modelo de referencia con el que se construyen las cachés distribuidas.
- **Menos cómputo.** Un producto de matrices float32 (2048 × 2048) alcanzó 117 GFLOPS con MLX en
  el iPad Pro M4, frente a 430 en una Mac M3 Pro. Espere una generación más lenta que en una Mac
  y prefiera la caché distribuida para cualquier trabajo largo.
- **El navegador no es la app.** El kernel Pyodide de juno.sh (JupyterLite) no ejecuta ningún
  motor; allí, use el modo `cache`. Véase también la [guía de ejecución](tablet.md).

## Solución de problemas

`report()` dice qué ofrece esta máquina a un modelo local: anfitrión, dispositivo, memoria, el
presupuesto, los motores, el modelo que elegiría `pick_model`, el archivo de caché y el modo. No
imprime nada, no crea ningún archivo de caché y, con `probe_servers=False` o en modo `cache`, no
sondea ningún servidor.

```python
import puremacro.llm as llm

print(llm.report(probe_servers=False))
```

En una Mac con mlx-lm y llama-cpp-python instalados se lee así (un ejemplo: las cifras dependen
de la máquina y de la caché):

```text
puremacro.llm
  host    : cpython 3.13.12 (arm64)
  device  : workstation
  memory  : 36,864 MB
  budget  : 28.08 GB for a model
  engines : mlx, llamacpp (servers not probed)
  model   : qwen3-4b (about 3.5 GB)
  cache   : /Users/you/.cache/puremacro/llm_cache.sqlite (1,204 rows, 2,457,600 bytes)
  mode    : auto (the cache, then a local model, whose answer is stored)
```

| Síntoma | Causa y remedio |
|---|---|
| `LocalLLMUnavailable` | ningún motor puede servir el modelo: instale uno (véase arriba) o use el modo `cache` con una caché distribuida |
| `engines : none` | no se encontró nada instalado; para un servidor, compruebe que responde en `/api/tags` o `/v1/models` |
| `CacheMiss` en modo `cache` | la caché distribuida no tiene esta llamada exacta; compare el alias, `max_tokens`, `temperature` y `json_mode` con la llamada que la construyó |
| `CacheUnavailable` | la caché no se puede leer aquí; su `reason` explica por qué (en Pyodide, cargue antes `sqlite3`; un puntero de Git LFS necesita `git lfs pull`) |
| `ModelTooLarge` | el modelo necesita más memoria que el presupuesto; use el modelo `suggested`, o `small` |
| HTTP 404 de Ollama | el modelo nunca se descargó: ejecute `ollama pull qwen3:4b` |
| `engine=mock` en la línea de estado | `get_default_backend` / `get_default_provider` no encontraron ningún motor y devolvieron el simulado |
| la primera llamada es lenta | el modelo se está descargando y cargando; las llamadas siguientes lo reutilizan |
