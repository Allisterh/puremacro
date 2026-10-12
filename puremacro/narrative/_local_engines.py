# puremacro/narrative/_local_engines.py
"""Compatibility re-exports; the engines live in :mod:`puremacro.llm`.

Through 4.8.1 this module held the local inference engines of the narrative scorers
(``score_llm`` and ``llm_prob_kernel``). They moved to :mod:`puremacro.llm.engines` and
:mod:`puremacro.llm.models`, and the names that code imported from here stay:

* ``BackendUnavailable``, ``LocalLLMUnavailable``, ``MLXEngine``, ``LlamaCppEngine``,
  ``HTTPEngine``, ``resolve_engine`` and ``resolve_model_id`` are the objects of
  :mod:`puremacro.llm`, not copies, so an ``except BackendUnavailable`` written here catches the
  ``LocalLLMUnavailable`` that the engines raise.
* ``MODEL_ALIASES`` is the registry :data:`puremacro.llm.MODELS` laid out by engine, the way the
  table of 4.8.1 was.
* ``chat`` is :func:`puremacro.llm.complete` under the signature of 4.8.1: it reads the
  completion cache first, and its ``base_url`` defaults to None, so that
  ``PUREMACRO_LLM_BASE_URL`` and then the default local servers (Ollama on 11434, OpenAI-compatible
  servers on 8080 and 1234) decide where it goes, instead of the one Ollama URL of 4.8.1.

Nothing is imported here but :mod:`puremacro.llm`, which loads no inference package and opens no
connection, so the module imports where mlx, llama.cpp and sockets do not exist.
"""
from __future__ import annotations

from .. import llm as _llm
from ..llm.engines import (
    BackendUnavailable,
    HTTPEngine,
    LlamaCppEngine,
    LocalLLMUnavailable,
    MLXEngine,
    resolve_engine,
)
from ..llm.models import MODELS, resolve_model_id

__all__ = [
    "BackendUnavailable",
    "HTTPEngine",
    "LlamaCppEngine",
    "LocalLLMUnavailable",
    "MLXEngine",
    "MODEL_ALIASES",
    "chat",
    "resolve_engine",
    "resolve_model_id",
]

# Canonical model name -> per-engine model id, from the registry of puremacro.llm:
#   mlx:      a Hugging Face repository id of an MLX conversion
#   llamacpp: a (repository, exact file name) pair
#   ollama:   an `ollama pull` tag
#   openai:   the "model" field sent to OpenAI-compatible servers
# A name outside the table is not ours to interpret and reaches the engine unchanged.
MODEL_ALIASES: dict[str, dict[str, str | tuple[str, str]]] = {
    alias: {"mlx": spec.mlx, "llamacpp": spec.llamacpp, "ollama": spec.ollama,
            "openai": spec.openai}
    for alias, spec in MODELS.items()
}


def chat(model: str, prompt: str, *, engine: str = "auto", base_url: str | None = None,
         max_tokens: int = 1024, temperature: float = 0.0, json_mode: bool = False,
         timeout: float = 120.0) -> str:
    """One-shot completion: :func:`puremacro.llm.complete` under the signature of 4.8.1.

    The answer comes from the completion cache when it has one, and is stored when a local model
    produced it (see :mod:`puremacro.llm`). ``base_url`` is None unless given, so the server is
    ``PUREMACRO_LLM_BASE_URL`` or else the first default local server that answers.
    """
    return _llm.complete(prompt, model=model, engine=engine, base_url=base_url,
                         max_tokens=max_tokens, temperature=temperature,
                         json_mode=json_mode, timeout=timeout)
