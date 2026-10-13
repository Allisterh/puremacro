"""Local language models behind one entry point.

``complete(prompt)`` returns a local model's answer to ``prompt``, and ``chat(messages)`` does
the same for a list of chat turns. Both look in a completion cache first, so an answer computed
once, on any machine and with any engine, is read back everywhere, including where no model can
run.

* **Modes** (:data:`MODES`). The ``mode`` argument, else the environment variable
  ``PUREMACRO_LLM_MODE``, else ``auto``:

  - ``auto``: the cache first. On a miss the model is checked against the device's memory, an
    engine is found, the model runs, and its answer is stored (an empty answer never is);
  - ``cache``: the cache only. A miss raises :class:`CacheMiss`, and a cache that cannot be read
    here raises :class:`CacheUnavailable`; no engine is constructed, no socket is opened and no
    model is loaded;
  - ``off``: the model every time. The cache is neither read nor written.

* **Models.** ``model`` is a tier (``default``, ``small``, ``large``), an alias of
  :data:`MODELS` such as ``qwen3-4b`` (the reference model, which ``default`` names unless
  ``PUREMACRO_LLM_MODEL`` says otherwise), or an engine's own id, such as an Ollama tag, which
  reaches the engine unchanged. The cache is keyed on the canonical alias (:func:`canonical`),
  never on an engine's id, so the engine that produced an answer does not matter. A model of the
  registry that needs more memory than the device offers raises :class:`ModelTooLarge` before
  any engine is looked for; a cached answer is served whatever the device.
* **Engines.** ``engine="auto"`` takes the first available of MLX, llama.cpp, Ollama and the
  OpenAI-compatible servers (see :mod:`puremacro.llm.engines`); ``base_url``, else
  ``PUREMACRO_LLM_BASE_URL``, names one server instead. :func:`get_engine` returns the engine
  itself, :func:`available_engines` the names of those available, and
  :func:`validate_engine_name` says whether a name is one ``engine`` takes, without probing
  anything.
* **Thinking models.** The Qwen3 models think before they answer. The engines switch the
  thinking off, and :func:`strip_thinking` removes whatever thinking still comes back. Such a
  model gets 8 tokens more than ``max_tokens`` at the engine, room for the empty think block it
  may still open its answer with; the cache key keeps the ``max_tokens`` of the call.
* **Cache.** :class:`CompletionCache`, in ``~/.cache/puremacro/llm_cache.sqlite`` or the file
  named by ``PUREMACRO_LLM_CACHE``. A cache that cannot be written, or in ``auto`` mode one that
  cannot be read, makes ``complete`` recompute answers instead of failing.
* **Errors.** :class:`CacheMiss` (a ``LookupError``) and :class:`CacheUnavailable` (a
  :class:`CacheMiss` whose ``reason`` says why the cache cannot be read), :class:`ModelTooLarge`
  (a ``RuntimeError`` with ``alias``, ``suggested`` and ``budget_gb``) and
  :class:`LocalLLMUnavailable` (no engine can serve the model; a :class:`BackendUnavailable`).
  A ``mode`` or ``engine`` that is not a known name, and a ``temperature`` that is not a finite
  number of zero or more, are a ``ValueError`` raised before the cache is read, in every mode.
* **Checking a machine.** :func:`report` says what this machine offers a local model: memory,
  engines, the model that fits, the cache and the mode.

Beyond what ``import puremacro`` loads, importing the package loads its three modules, the
standard library and ``puremacro._optional_stdlib`` (through which the cache imports
``sqlite3``), and nothing else: the inference packages, the network helpers and
:mod:`puremacro.runtime` are imported inside the functions that use them. So
``import puremacro.llm`` works under Pyodide, where no engine is available and ``cache`` mode is
the one that answers. Pyodide 0.28 adds ``sqlite3`` only on request: run
``await pyodide.loadPackage('sqlite3')`` before importing puremacro, or ``cache`` mode raises
:class:`CacheUnavailable`, whose message gives that hint.
"""
from __future__ import annotations

import math
import numbers
import os
from typing import Any

from . import cache as _cache
from . import engines as _engines
from . import models as _models
from .cache import (
    CacheMiss,
    CacheRow,
    CacheUnavailable,
    CompletionCache,
    cache_key,
    default_cache_path,
)
from .engines import (
    BackendUnavailable,
    Engine,
    HTTPEngine,
    LlamaCppEngine,
    LocalLLMUnavailable,
    MLXEngine,
    strip_thinking,
    validate_engine_name,
)
from .models import (
    MODELS,
    ModelSpec,
    ModelTooLarge,
    canonical,
    check_fits,
    device_budget_gb,
    pick_model,
    resolve_model_id,
)

__all__ = [
    "BackendUnavailable",
    "CacheMiss",
    "CacheRow",
    "CacheUnavailable",
    "CompletionCache",
    "HTTPEngine",
    "LlamaCppEngine",
    "LocalLLMUnavailable",
    "MLXEngine",
    "MODELS",
    "MODES",
    "ModelSpec",
    "ModelTooLarge",
    "available_engines",
    "cache_key",
    "canonical",
    "chat",
    "check_fits",
    "complete",
    "current_mode",
    "default_cache_path",
    "device_budget_gb",
    "get_engine",
    "pick_model",
    "report",
    "resolve_model_id",
    "strip_thinking",
    "validate_engine_name",
]

MODES = ("auto", "cache", "off")
"""The modes of a completion: cache then model and store; cache only; model only."""

_ENV_MODE = "PUREMACRO_LLM_MODE"

_THINKING_ALLOWANCE = 8
"""Tokens a thinking model gets beyond ``max_tokens`` at the engine: room for the empty
``<think></think>`` block it may still open its answer with, which :func:`strip_thinking`
removes. The cache key keeps the caller's ``max_tokens``."""

_MODE_MEANING = {
    "auto": "the cache, then a local model, whose answer is stored",
    "cache": "the cache only; a miss raises CacheMiss",
    "off": "a local model every time; the cache is not used",
}

# The functions below reach the engines, the device budget and the cache through the module
# objects (``_engines.resolve_engine``, ``_models.device_budget_gb``...), never through names
# bound at import, so that replacing an attribute of puremacro.llm.engines or
# puremacro.llm.models (as tests do with monkeypatch) changes what the facade calls.


def current_mode(mode: str | None = None) -> str:
    """The mode a completion runs in: ``mode``, else ``PUREMACRO_LLM_MODE``, else ``"auto"``.

    Surrounding blanks and capitals are ignored, and a blank variable counts as unset.

    Raises
    ------
    ValueError
        ``mode``, or the variable when ``mode`` is None, is not one of :data:`MODES`.
    """
    if mode is not None:
        chosen = str(mode).strip().lower()
        if chosen not in MODES:
            raise ValueError(f"unknown mode {mode!r}; use 'auto', 'cache' or 'off'")
        return chosen
    raw = os.environ.get(_ENV_MODE, "").strip()
    if not raw:
        return "auto"
    if raw.lower() not in MODES:
        raise ValueError(f"{_ENV_MODE}={raw!r} is not a mode; use 'auto', 'cache' or 'off'")
    return raw.lower()


def _check_temperature(temperature: Any) -> Any:
    """``temperature`` once checked: a real number, not a bool, finite and zero or more.

    ``nan`` would be keyed as the text ``nan`` and never hit, no engine takes a negative or
    infinite temperature, and ``True`` or ``"0.5"`` would reach an engine as something it does
    not expect. ``-0.0`` comes back as ``0.0`` so both share one key; every other value is passed
    on as given, so the ``v1`` key function is untouched.
    """
    if isinstance(temperature, bool) or not isinstance(temperature, numbers.Real):
        raise ValueError(f"temperature must be a real number >= 0; got {temperature!r}")
    value = float(temperature)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"temperature must be a finite number >= 0; got {temperature!r}")
    return 0.0 if value == 0 else temperature


def complete(prompt: str, *, model: str = "default", engine: str = "auto",
             mode: str | None = None, system: str | None = None, max_tokens: int = 256,
             temperature: float = 0.0, json_mode: bool = False, timeout: float = 120.0,
             base_url: str | None = None, cache: CompletionCache | None = None) -> str:
    """A local model's answer to ``prompt``, from the cache when it has one.

    In ``auto`` mode the cache is read first. On a miss, a model of the registry is checked
    against :func:`device_budget_gb`, then an engine is found, the model runs, and a non-empty
    answer is stored. ``cache`` mode stops at the miss; ``off`` mode skips the cache entirely
    (see the module docstring).

    Parameters
    ----------
    prompt : str
        The user turn as written; the cache key hashes it before any chat template.
    model : str
        ``"default"`` (``qwen3-4b`` unless ``PUREMACRO_LLM_MODEL`` names another model),
        ``"small"``, ``"large"``, an alias of :data:`MODELS`, or an engine's own id (an Ollama
        tag, a ``.gguf`` path), which reaches the engine unchanged and is not checked against
        the device's memory.
    engine : str
        ``"auto"`` (the first available; see :func:`puremacro.llm.engines.resolve_engine`),
        ``"mlx"``, ``"llamacpp"``, ``"ollama"`` or ``"openai"``.
    mode : str, optional
        ``"auto"``, ``"cache"`` or ``"off"``; None reads ``PUREMACRO_LLM_MODE`` and defaults to
        ``"auto"`` (:func:`current_mode`).
    system : str, optional
        System prompt, sent as a leading system turn. Part of the cache key.
    max_tokens : int
        Longest answer, in tokens. A thinking model gets 8 more at the engine, room for the
        empty think block it may still open its answer with; the key keeps this value.
    temperature : float
        Sampling temperature (MLX decodes greedily whatever it is). Part of the key.
    json_mode : bool
        Ask the engine for a JSON object where it supports that. Part of the key.
    timeout : float
        Seconds an HTTP engine waits for the answer.
    base_url : str, optional
        One server to use instead of the default local ones; else ``PUREMACRO_LLM_BASE_URL``.
    cache : CompletionCache, optional
        The cache to read and write. Defaults to ``CompletionCache()``: the file named by
        ``PUREMACRO_LLM_CACHE``, else ``~/.cache/puremacro/llm_cache.sqlite``. ``off`` mode
        neither opens it nor creates it.

    Returns
    -------
    str
        The answer, without the model's thinking and without surrounding whitespace. An empty
        answer is returned but not stored.

    Raises
    ------
    CacheMiss
        ``cache`` mode and no cached answer.
    CacheUnavailable
        ``cache`` mode and a cache that cannot be read here: no ``sqlite3``, or a file that
        cannot be opened as a cache; its ``reason`` says which. It is a :class:`CacheMiss`, so
        ``except CacheMiss`` catches it too.
    ModelTooLarge
        A model of the registry needs more memory than the device offers; raised before any
        engine is looked for, and never for a cached answer.
    LocalLLMUnavailable
        No engine, or not the one named, can serve the model.
    ValueError
        ``mode``, ``PUREMACRO_LLM_MODE`` or ``engine`` is not a valid name, or ``temperature``
        is not a finite number of zero or more, or ``prompt`` or ``system`` is not valid Unicode
        text (a lone surrogate). All are raised before the cache is read, in every mode, so a
        typo does not pass unnoticed while the cache has the answer.

    Other failures of the engine, such as an answer that does not come within ``timeout`` or a
    request the server turns down, propagate as the engine raises them.
    """
    mode = current_mode(mode)
    engine = _engines.validate_engine_name(engine)
    temperature = _check_temperature(temperature)
    alias = _models.canonical(model)
    key = _cache.cache_key(alias, prompt, system=system, max_tokens=max_tokens,
                           temperature=temperature, json_mode=json_mode)
    # Off mode neither reads nor writes a cache, so it builds none (CompletionCache() creates the
    # directory, the file and the table) and leaves one passed as ``cache`` untouched.
    store: CompletionCache | None = None
    if mode != "off":
        store = cache if cache is not None else _cache.CompletionCache()
        hit = store.get(key)
        if hit is not None:
            return hit
        if mode == "cache":
            if not store.readable:  # the lookup could not read the cache, so this is not a miss
                raise _cache.CacheUnavailable(key, alias, store.reason)
            raise _cache.CacheMiss(key, alias)
    spec = _models.MODELS.get(alias)
    if spec is not None:            # before any engine, so a model that cannot fit probes nothing
        _models.check_fits(spec, _models.device_budget_gb())
    eng = _engines.resolve_engine(engine, base_url=base_url, timeout=timeout)
    thinking = spec is not None and spec.thinking
    model_id = _models.resolve_model_id(alias, eng.name)
    reply = eng.complete(model_id, prompt, system=system,
                         max_tokens=max_tokens + _THINKING_ALLOWANCE if thinking else max_tokens,
                         temperature=temperature, json_mode=json_mode, thinking=thinking)
    text = _engines.strip_thinking(reply)
    if store is not None and text:  # auto mode: cache mode has returned or raised above
        store.set(_cache.CacheRow(key, alias, _cache.prompt_sha256(prompt, system), max_tokens,
                                  temperature, json_mode, text, eng.name, str(model_id)))
    return text


def chat(messages: list[dict], **kw: Any) -> str:
    """:func:`complete` for a list of chat turns, flattened into one system prompt and one prompt.

    The ``content`` of the ``system`` turns, joined by newlines, becomes the system prompt; a
    ``system`` keyword counts as a first system turn. A single other turn that is a ``user``
    turn is the prompt as it stands, so ``chat([{"role": "user", "content": p}])`` and
    ``complete(p)`` share their cache entry. Otherwise the other turns become one prompt, each
    as ``Role: content`` (``User: …``, ``Assistant: …``), separated by blank lines. Every other
    keyword goes to :func:`complete`.

    Raises
    ------
    ValueError
        ``messages`` has no turn other than system turns.
    """
    extra = kw.pop("system", None)
    systems = ([extra] if extra else []) + [turn["content"] for turn in messages
                                            if turn["role"] == "system"]
    turns = [turn for turn in messages if turn["role"] != "system"]
    if not turns:
        raise ValueError("chat() needs at least one turn that is not a system turn")
    if len(turns) == 1 and turns[0]["role"] == "user":
        prompt = turns[0]["content"]
    else:
        prompt = "\n\n".join(f"{turn['role'].capitalize()}: {turn['content']}" for turn in turns)
    return complete(prompt, system="\n".join(systems) if systems else None, **kw)


def get_engine(engine: str = "auto", *, timeout: float = 120.0,
               base_url: str | None = None) -> Engine:
    """The engine :func:`complete` would use, from :func:`puremacro.llm.engines.resolve_engine`.

    Raises
    ------
    LocalLLMUnavailable
        No engine, or not the one named, is available.
    ValueError
        ``engine`` is not a known name (:func:`validate_engine_name`); nothing is probed first.
    """
    engine = _engines.validate_engine_name(engine)
    return _engines.resolve_engine(engine, base_url=base_url, timeout=timeout)


def available_engines(*, probe_servers: bool = True) -> list[str]:
    """Names of the engines available here, in the order ``engine="auto"`` tries them.

    With ``probe_servers=False`` no server is probed and only ``mlx`` and ``llamacpp`` can be
    listed (see :func:`puremacro.llm.engines.available_engines`).
    """
    return _engines.available_engines(probe_servers=probe_servers)


def _machine() -> tuple[str, str, str]:
    """Host, device and physical memory from :mod:`puremacro.runtime`, or ``unknown``."""
    try:
        from puremacro.runtime import _capabilities

        caps = _capabilities.capabilities()
        memory = "unknown" if caps.memory_mb is None else f"{caps.memory_mb:,} MB"
        return f"{caps.host} {caps.python} ({caps.machine})", str(caps.device), memory
    except Exception:
        return "unknown", "unknown", "unknown"


def _engines_line(probe: bool) -> str:
    try:
        found = _engines.available_engines(probe_servers=probe)
    except Exception:
        return "unknown"
    if not probe:
        return f"{', '.join(found) or 'none'} (servers not probed)"
    return ", ".join(found) if found else f"none. {_engines.INSTALL_HINT}"


def _model_line(budget: float | None) -> str:
    try:
        spec = _models.pick_model() if budget is None else _models.pick_model(budget_gb=budget)
        default = _models.canonical("default")
    except Exception:
        return "unknown"
    text = f"{spec.alias} (about {spec.approx_gb:.1f} GB)"
    return text if default == spec.alias else f"{text}; 'default' is {default}"


def _cache_line() -> str:
    """The cache file, its rows, its size; a file that does not exist yet is not created."""
    try:
        path = _cache.default_cache_path().absolute()
    except Exception:
        return "unknown"
    try:
        if not path.exists():
            return f"{path} (no file yet)"
        stats = _cache.CompletionCache(path).stats()
        rows = int(stats["rows"])
        return f"{path} ({rows:,} row{'' if rows == 1 else 's'}, {int(stats['bytes']):,} bytes)"
    except Exception:
        return f"{path} (unknown)"


def report(*, probe_servers: bool = True) -> str:
    """What this machine offers a local model, one item per line.

    The host, device and physical memory (from :mod:`puremacro.runtime`); the memory budget a
    model may use (:func:`device_budget_gb`); the engines available; the model
    :func:`pick_model` chooses for that budget, with its approximate size; the cache file, with
    its row count and size; and the mode. An item whose source fails reads ``unknown`` instead
    of raising, and an invalid ``PUREMACRO_LLM_MODE`` is named instead of raised.

    Parameters
    ----------
    probe_servers : bool
        Probe the local servers (Ollama, OpenAI-compatible) besides the installed engines. In
        ``cache`` mode no server is probed, whatever this says.

    Returns
    -------
    str
        The report; nothing is printed. A cache file that does not exist yet is reported as
        such, not created.
    """
    host, device, memory = _machine()
    try:
        budget: float | None = float(_models.device_budget_gb())
    except Exception:
        budget = None
    try:
        mode: str | None = current_mode()
        mode_text = f"{mode} ({_MODE_MEANING[str(mode)]})"
    except ValueError as exc:
        mode, mode_text = None, f"invalid: {exc}"
    items = (
        ("host", host),
        ("device", device),
        ("memory", memory),
        ("budget", "unknown" if budget is None else f"{budget:.2f} GB for a model"),
        ("engines", _engines_line(probe_servers and mode != "cache")),
        ("model", _model_line(budget)),
        ("cache", _cache_line()),
        ("mode", mode_text),
    )
    width = max(len(label) for label, _ in items)
    lines = [f"  {label:<{width}} : {value}" for label, value in items]
    return "\n".join(["puremacro.llm", *lines])
