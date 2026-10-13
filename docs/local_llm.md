> 🇬🇧 English · 🇪🇸 [Español](es/local_llm.md)

# Local language models: iPad, Mac, Windows, Linux and Colab

`puremacro.llm` runs a small language model on your own device, with no API key and no
network after a one-time download. `complete(prompt)` returns the model's answer, choosing the
engine on its own: MLX, llama.cpp, Ollama or an OpenAI-compatible server. Every answer is stored
in a completion cache that does not depend on the device or the engine, so an answer computed
once, on a Mac, is read back on an iPad, on Windows, in Colab, or in the browser, where no model
runs at all. The narrative LLM features (`score_llm`, `llm_prob_kernel`) go through the same
entry point.

```python
# requires: a local LLM engine (mlx-lm, llama-cpp-python, or a running Ollama / OpenAI-compatible server)
import puremacro.llm as llm

print(llm.complete("In one sentence, what is a local projection?", max_tokens=64))
print(llm.chat([{"role": "system", "content": "Answer in Spanish."},
                {"role": "user", "content": "What is a Taylor rule?"}], max_tokens=128))
```

`puremacro.llm` is new after puremacro 4.8.1 (see the CHANGELOG, Unreleased): a release
that includes it is needed. `import puremacro.llm` imports no inference package and opens no
connection; each engine is imported only when it is used.

## What runs where

| Platform | Engine | You install | The model comes from |
|---|---|---|---|
| iPad, Juno app | MLX; llama.cpp as backup | the Juno app; puremacro through Juno's package manager | Hugging Face, downloaded by mlx-lm on first use and kept in the app's container |
| Mac (Apple silicon) | MLX, else Ollama | `pip install "puremacro[local-llm]"`, or the Ollama app | the same, or `ollama pull qwen3:4b` |
| Windows, Linux | Ollama, or llamafile | the Ollama installer, or one llamafile run as a local server | the Ollama tag, or the GGUF inside the llamafile |
| Colab | llama.cpp on the GPU, or Ollama inside the notebook | nothing; a setup cell does it | a GGUF file from Hugging Face |
| Browser, JupyterLite | none | nothing | the shipped cache only |

`engine="auto"` (the default) takes the first available of: `mlx` (when `mlx` and `mlx_lm` can
be imported), `llamacpp` (when `llama_cpp` can be imported), Ollama on `http://localhost:11434`,
then OpenAI-compatible servers on `http://localhost:8080` (llamafile, llama-server) and
`http://localhost:1234` (LM Studio). What is installed decides, never the operating system or the
processor family, which is why MLX is chosen on the iPad. A server counts as available when its
model list (`/api/tags` or `/v1/models`) answers with JSON within three seconds. Pass
`engine="mlx"`, `"llamacpp"`, `"ollama"` or `"openai"` to name one.

What has been checked, as of 2026-10-12:

- On a Mac (M3 Pro), with the Qwen3 1.7B model: MLX, llama-cpp-python (built from source with
  Metal) and llama-cpp-python's OpenAI-compatible server all answered with no thinking left in
  the text, for the index prompt and the event-extraction prompt (whose answers write
  `"sign": +1`, which `score_llm` now parses).
- Ollama: the request is in place (`"think": false`) but has not yet been run against a pulled
  Qwen3 model.
- iPad (Juno app, iPad Pro M4, 16 GB): `mlx`, `mlx_lm`, `llama_cpp` and `huggingface_hub` import
  and Hugging Face is reachable; a completion through Juno's mlx-lm build is **not yet verified
  on device**.
- Colab: the llama.cpp CUDA wheel and the in-notebook Ollama recipes below are **not yet verified
  on device**.

## Install per platform

**iPad.** Install the Juno app and, in it, puremacro through Juno's package manager (Juno has
no `pip`). Juno ships MLX and mlx-lm, so nothing else is needed. The first completion downloads
the model; see [On the iPad](#on-the-ipad).

**Mac.** Either the Python engines:

```bash
pip install "puremacro[local-llm]"     # llama-cpp-python, huggingface_hub, and mlx-lm on Apple silicon
```

or the Ollama app (https://ollama.com), then `ollama pull qwen3:4b`. MLX is the one puremacro
picks when both are present. The extra installs mlx-lm on Apple silicon only; on an Intel Mac
it installs llama-cpp-python and huggingface_hub, and Ollama works there too. The `[local-llm]`
extra builds llama-cpp-python from source, which needs the Xcode command-line tools. Do not use
the prebuilt Metal wheel of llama-cpp-python 0.3.36 from the project's extra index: as of
2026-10-12 it is corrupt (half of its files fail their CRC check). To build llama.cpp with Metal
explicitly:

```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --no-binary llama-cpp-python
```

If that build fails, skip llama.cpp: `pip install puremacro mlx-lm` is enough for MLX, and
Ollama needs no Python package at all.

**Windows and Linux.** Install Ollama (https://ollama.com) and run `ollama pull qwen3:4b`.
Or download a llamafile and run it as a local server; it listens on port 8080 with an
OpenAI-compatible API, which `engine="auto"` finds. `pip install "puremacro[local-llm]"` also
works where llama-cpp-python builds (a C compiler and CMake are needed).

**Colab.** Choose a GPU runtime (Runtime → Change runtime type → T4 GPU), then paste into one
code cell:

```text
# requires: Colab with a GPU runtime (not yet verified on device)
%pip install -q --only-binary=llama-cpp-python llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
%pip install -q huggingface_hub puremacro
```

The CUDA wheel is about 1.8 GB; `--only-binary` makes the cell fail fast instead of starting a
long CPU-only build. The engine offloads every layer to the GPU (`n_gpu_layers=-1`). The other
route is Ollama inside the notebook. First install what its Linux install script needs,
`apt-get -qq update; apt-get -qq install -y zstd pciutils lshw` (`zstd` because newer Ollama
Linux bundles are `.tar.zst` archives, `pciutils` and `lshw` for the script's GPU detection);
then run `curl -fsSL https://ollama.com/install.sh | sh`, start `ollama serve` in the background
and run `ollama pull qwen3:4b`; `engine="auto"` then finds it on port 11434.

**Browser (JupyterLite, juno.sh).** No engine runs under Pyodide. Ship a cache (see
[The cache and how to ship one](#the-cache-and-how-to-ship-one)) and use `cache` mode. Pyodide
0.28 adds `sqlite3` only on request: run `await pyodide.loadPackage('sqlite3')` before importing
puremacro.

## The model ladder

One reference model for everything, `qwen3-4b`: shipped caches are built with it, and
`default` names it. All three rungs are Apache-2.0.

| Alias | Tier name | Parameters | Memory, approx. | Minimum budget | Licence | Picked automatically |
|---|---|---|---|---|---|---|
| `qwen3-1.7b` | `small` | 1.7 B | 2.0 GB | 3 GB | apache-2.0 | below a 5 GB budget |
| `qwen3-4b` | `default` | 4 B | 3.5 GB | 5 GB | apache-2.0 | from 5 GB up |
| `qwen3-8b` | `large` | 8 B | 6.0 GB | 10 GB | apache-2.0 | never; ask for it by name |

What each engine loads (`puremacro.llm.resolve_model_id(alias, engine)`):

| Alias | MLX (Hugging Face) | llama.cpp (repository, file) | Ollama tag |
|---|---|---|---|
| `qwen3-1.7b` | `mlx-community/Qwen3-1.7B-4bit` | `ggml-org/Qwen3-1.7B-GGUF`, `Qwen3-1.7B-Q4_K_M.gguf` | `qwen3:1.7b` |
| `qwen3-4b` | `mlx-community/Qwen3-4B-4bit` | `Qwen/Qwen3-4B-GGUF`, `Qwen3-4B-Q4_K_M.gguf` | `qwen3:4b` |
| `qwen3-8b` | `mlx-community/Qwen3-8B-4bit` | `Qwen/Qwen3-8B-GGUF`, `Qwen3-8B-Q4_K_M.gguf` | `qwen3:8b` |

The downloads are about 1.0, 2.3 and 4.6 GB for MLX, and about 1.3, 2.5 and 5 GB for the GGUF
files and the Ollama tags. OpenAI-compatible servers receive the alias itself as `model`
(`"qwen3-4b"`); llamafile and llama-server answer with the model they were started with.

- **The budget** (`device_budget_gb()`) is the memory a model may use: the working set MLX
  recommends for the GPU when mlx is installed, else 60 % of physical memory, else an assumed
  8 GB. `pick_model()` turns it into a model; `check_fits` refuses a model whose minimum budget
  the device does not reach, and `complete()` then raises `ModelTooLarge` (with `alias`,
  `suggested` and `budget_gb`) before any engine is looked for. A cached answer is served
  whatever the device. The budget is the local device's even when `PUREMACRO_LLM_BASE_URL` or
  `base_url=` names another machine: a registry alias above it is refused before that server is
  asked; the engine's own model id (`qwen3:8b`) bypasses the check, at the cost of a different
  cache key and no thinking switch.
- **Names.** `model=` takes a tier name, an alias, or an engine's own id (an Ollama tag, a
  `.gguf` path, a Hugging Face repository), which reaches the engine unchanged. Such an id has
  no entry in the model registry, so it is not checked against the budget and is not treated as
  a thinking model: the engine gets no thinking switch and no extra tokens (see Thinking), and
  only `strip_thinking` cleans the answer. Use the alias (`qwen3-4b`, not `qwen3:4b`) to get
  the switch. `PUREMACRO_LLM_MODEL` changes what `default` means, and nothing else.
- **Thinking.** The Qwen3 models think before they answer. The engines switch that off (MLX and
  llama.cpp through the chat template, Ollama with `"think": false`, OpenAI-compatible servers
  with `chat_template_kwargs` and a `/no_think` suffix), and `strip_thinking` removes any
  `<think>…</think>` block that still comes back. A thinking model gets 8 tokens beyond
  `max_tokens` at the engine, room for the empty think block it may open with.
- **The models of 4.8.1** stay under their names, for code that uses them:
  `qwen2.5-3b-instruct` (licence `qwen-research`), `gemma2-2b` (`gemma`), `llama3.2-3b`
  (`llama3.2`) and `phi3.5` (`mit`). They are never picked automatically.

## Modes

The `mode` argument, else the environment variable `PUREMACRO_LLM_MODE`, else `auto`
(`puremacro.llm.current_mode()` says which applies):

| Mode | Reads the cache | Runs a model | Writes the cache | On a cache miss |
|---|---|---|---|---|
| `auto` | yes | on a miss | yes (non-empty answers) | checks the budget, finds an engine, runs the model |
| `cache` | yes | never | no | raises `CacheMiss` |
| `off` | no | always | no | (the cache is not used) |

In `cache` mode no engine is constructed, no socket is opened and no model is loaded. A cache
that cannot be read at all (no `sqlite3`, a file that is not a cache, such as a Git LFS pointer)
raises `CacheUnavailable`, a kind of `CacheMiss` whose `reason` says why. `off` mode neither
opens nor creates a cache file. With no engine available, `auto` and `off` raise
`LocalLLMUnavailable` with install instructions.

The narrative features follow the mode:

- `LocalBackend` (for `score_llm`) and `LocalProvider` (for `llm_prob_kernel`) take
  `model="default"`, `engine="auto"`, `base_url=None`, `timeout=120.0` and `mode=None`, look for
  no engine when they are built, and send every call through `puremacro.llm.complete`. So a
  corpus scored once is scored again with no engine installed.
- `get_default_backend()` and `get_default_provider()` print one status line:

  ```text
  [puremacro] local LLM: mode=cache engine=cache model=qwen3-4b
  ```

  In `cache` mode they return the local object without probing anything. In `auto` and `off`
  modes they ask `puremacro.llm.engines.available_engines(base_url=…)`; when no engine (or not
  the one named) is listed, they print the install hint, report `engine=mock` and return
  `MockBackend` (zero events) or `MockProvider`, so demos always run.
- In `cache` mode `score_llm` raises `CacheMiss` (or `CacheUnavailable`) for a record the cache
  does not hold, instead of dropping it: a corpus that was not scored ahead of time fails loudly
  rather than returning a shorter list. `ModelTooLarge` propagates too. Other failures of one
  record's call (a timeout, say) are still counted as malformed and the loop goes on.
- `OllamaBackend()` and `OllamaProvider()` default to the alias `qwen3-4b`: the server receives
  its tag `qwen3:4b`, and the cache rows are those of `LocalBackend("default")`. An explicit tag,
  `OllamaBackend("qwen3:4b")`, reaches the server unchanged and then keys the cache on the tag,
  so its rows are not shared with the alias. Nor is it known to think: the request goes without
  `"think": false` and without the 8 extra tokens, and only `strip_thinking` cleans the answer.
  Keep the alias to get the switch.

```python
from puremacro.narrative.scoring import get_default_backend

backend = get_default_backend(mode="cache")   # probes nothing; prints the status line
print(type(backend).__name__, backend.model)  # LocalBackend qwen3-4b
```

## The cache and how to ship one

One SQLite file, `~/.cache/puremacro/llm_cache.sqlite`, or the file named by
`PUREMACRO_LLM_CACHE`. A row is keyed (format `v1`) on the canonical alias (`qwen3-4b`, never an
engine's id), the prompt and system prompt as you wrote them (before any chat template),
`max_tokens`, `temperature` (to three decimals) and `json_mode`. The engine is recorded for
provenance only, so a row written by MLX answers the same call made through llama.cpp or Ollama.
The key function is fixed for the life of the format; its golden value is pinned by the tests:
`cache_key("qwen3-4b", "hola", max_tokens=8, temperature=0.0, json_mode=False)` is
`c630a581ab2b323492ac1db08bfc8b8f5a72655db00282142eb5762eacccb586`.

To ship one: run the notebooks or scripts once in `auto` mode on a machine with the reference
model, export the rows, and hand over the file. On the receiving side, point
`PUREMACRO_LLM_CACHE` at it and set `PUREMACRO_LLM_MODE=cache`, or merge it into your own cache.
A cache on a read-only volume still serves hits.

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

- `export(dest, *, model=None, since=None)` copies rows into another file, creating it;
  `since="2026-10-01"` keeps rows created on or after that date (UTC). `merge(src)` adds the rows
  of another file that this one lacks. Both return the number of rows added and raise if they
  cannot do their work; `stats()` gives `rows`, `by_model`, `bytes` and `path`.
- A cache hit needs the same call: the same alias, the same prompt, `max_tokens`, `temperature`
  and `json_mode`. `PUREMACRO_LLM_MODEL` changes the alias that `default` stands for, and an
  explicit Ollama tag keys rows under the tag.
- A cache that cannot be written (a read-only directory, a corrupt file) never stops a
  completion: answers are recomputed instead, and one warning per file says why.
- `llm_prob_kernel` keeps its own score cache in front of this one
  (`~/.cache/puremacro/llm_scores.sqlite`, or `PUREMACRO_LLM_SCORES_CACHE`). Shipping the
  completion cache is enough: a score missing there falls through to the completion cache.

## Environment variables

| Variable | Effect |
|---|---|
| `PUREMACRO_LLM_MODE` | `auto`, `cache` or `off` when no `mode=` is given; blank counts as unset, anything else raises `ValueError` |
| `PUREMACRO_LLM_MODEL` | what `default` means: a tier name, an alias or an engine's own id; `small`, `large` and explicit aliases are not affected |
| `PUREMACRO_LLM_CACHE` | the completion cache file (default `~/.cache/puremacro/llm_cache.sqlite`) |
| `PUREMACRO_LLM_BASE_URL` | one server to use instead of the three default local ones, tried as Ollama and then as OpenAI-compatible |
| `PUREMACRO_LLM_SCORES_CACHE` | the score cache of `llm_prob_kernel` (default `~/.cache/puremacro/llm_scores.sqlite`) |
| `PUREMACRO_LLM_BUDGET` | `spend` removes the call cap (`max_calls`) of `llm_prob_kernel` |
| `HF_HUB_OFFLINE` | `1` makes llama.cpp load a downloaded GGUF file from the Hugging Face cache with no network |

A server URL, from `base_url=` or `PUREMACRO_LLM_BASE_URL`, loses its trailing slashes and then
one trailing `/v1`. So `http://localhost:1234/v1`, the form LM Studio and OpenAI clients
document, names the same server as `http://localhost:1234`.

## On the iPad

- **Keep Juno in front** while a model downloads and while a corpus is scored. iPadOS suspends
  an app in the background, and a download or a scoring loop stops with it. Scoring runs are
  short; split a long corpus into pieces, and the cache keeps every answer already computed.
- **Models persist in the app's container.** The first completion downloads the model from
  Hugging Face (about 2.3 GB for `qwen3-4b` with MLX, about 1 GB for `qwen3-1.7b`) into the
  app's container, where later sessions find it: the download happens once.
- **The budget.** On an iPad Pro M4 with 16 GB, MLX recommends a working set of about 11 GB, so
  `pick_model()` chooses `qwen3-4b`, and `qwen3-8b` fits when asked for by name. Devices with
  less memory fall to `qwen3-1.7b`, whose answers can differ from those of the reference model
  that shipped caches are built with.
- **Compute is smaller.** A float32 matrix product (2048 × 2048) ran at 117 GFLOPS through MLX
  on the iPad Pro M4, against 430 on an M3 Pro Mac. Expect slower generation than on a Mac, and
  prefer the shipped cache for anything long.
- **The browser is not the app.** The Pyodide kernel of juno.sh (JupyterLite) runs no engine;
  there, use `cache` mode. See also the [runtime guide](tablet.md).

## Troubleshooting

`report()` says what this machine offers a local model: host, device, memory, the budget, the
engines, the model `pick_model` would choose, the cache file and the mode. It prints nothing,
creates no cache file and, with `probe_servers=False` or in `cache` mode, probes no server.

```python
import puremacro.llm as llm

print(llm.report(probe_servers=False))
```

On a Mac with mlx-lm and llama-cpp-python installed it reads like this (an example: the numbers
depend on the machine and the cache):

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

| Symptom | Cause and remedy |
|---|---|
| `LocalLLMUnavailable` | no engine can serve the model: install one (see above) or use `cache` mode with a shipped cache |
| `engines : none` | nothing installed was found; for a server, check that it answers at `/api/tags` or `/v1/models` |
| `CacheMiss` in `cache` mode | the shipped cache lacks this exact call; compare the alias, `max_tokens`, `temperature` and `json_mode` with the call that built it |
| `CacheUnavailable` | the cache cannot be read here; its `reason` says why (in Pyodide, load `sqlite3` first; a Git LFS pointer needs `git lfs pull`) |
| `ModelTooLarge` | the model needs more memory than the budget; use the `suggested` model, or `small` |
| HTTP 404 from Ollama | the model was never pulled: run `ollama pull qwen3:4b` |
| `engine=mock` in the status line | `get_default_backend` / `get_default_provider` found no engine and returned the mock |
| the first call is slow | the model is being downloaded and loaded; later calls reuse it |
