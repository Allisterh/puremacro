"""LLM-prob scoring kernel (Cluster B, item B4).

Sends each text window (sentence or paragraph) to an LLM with a prompt
asking "what is the probability this window expresses <category>?" and
returns the average probability per document, in [0, 1].

This is the highest-quality kernel in the package: it handles
qualifiers, irony, discourse structure, paraphrase — anything a
lexicon or MNL misses. But it's also the most expensive: each window
is an API call, and a single corpus pass can cost $5–30.

Design priorities, in order:
  1. **Opt-in**: the entire module imports lazily and only triggers
     when ``llm_prob_kernel(...)`` is actually called.
  2. **Cached**: scores are keyed on
     ``(provider, model, prompt_hash, text_hash)`` in a SQLite store
     under ``~/.cache/puremacro/llm_scores.sqlite``, or the file named by
     ``PUREMACRO_LLM_SCORES_CACHE``. Repeat runs are instant and free.
     This is not the completion cache of :mod:`puremacro.llm`, which keeps
     its own file and reads ``PUREMACRO_LLM_CACHE``.
  3. **Cost-capped**: a per-call budget (max API calls) is enforced
     unless ``PUREMACRO_LLM_BUDGET=spend`` is exported.
  4. **Provider-agnostic**: Anthropic Claude / OpenAI GPT / local
     Ollama all expose a ``score_paragraph(text, category) -> float``
     method via the ``LLMProvider`` ABC.
"""
from __future__ import annotations

import hashlib
import os
from puremacro._optional_stdlib import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

from puremacro import credentials
from ._kernels import _split_paragraphs, _split_sentences


# ---------------------------------------------------------------------------
# Cache layer (SQLite)
# ---------------------------------------------------------------------------
def _cache_path() -> Path:
    """The score cache file: ``$PUREMACRO_LLM_SCORES_CACHE``, else the user cache directory.

    ``PUREMACRO_LLM_CACHE`` is deliberately not read. It names the completion
    cache of :mod:`puremacro.llm`, and two caches in one file would put this
    table inside a (possibly read-only, shipped) completion cache and let this
    kernel's open transaction block that cache's writes.
    """
    return Path(os.environ.get(
        "PUREMACRO_LLM_SCORES_CACHE",
        Path.home() / ".cache" / "puremacro" / "llm_scores.sqlite",
    ))


def _ensure_cache_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS llm_scores ("
        " key TEXT PRIMARY KEY,"
        " provider TEXT,"
        " model TEXT,"
        " prompt_hash TEXT,"
        " text_hash TEXT,"
        " score REAL,"
        " created_at TEXT DEFAULT CURRENT_TIMESTAMP"
        ")"
    )


def _cache_key(provider: str, model: str, prompt: str, text: str) -> str:
    h = hashlib.sha256()
    h.update(provider.encode())
    h.update(b"|")
    h.update(model.encode())
    h.update(b"|")
    h.update(hashlib.sha256(prompt.encode()).hexdigest().encode())
    h.update(b"|")
    h.update(hashlib.sha256(text.encode()).hexdigest().encode())
    return h.hexdigest()


def _cache_get(conn: sqlite3.Connection, key: str) -> float | None:
    row = conn.execute(
        "SELECT score FROM llm_scores WHERE key = ?", (key,),
    ).fetchone()
    return float(row[0]) if row else None


def _cache_set(
    conn: sqlite3.Connection, key: str,
    provider: str, model: str, prompt: str, text: str, score: float,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO llm_scores "
        " (key, provider, model, prompt_hash, text_hash, score)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (
            key, provider, model,
            hashlib.sha256(prompt.encode()).hexdigest(),
            hashlib.sha256(text.encode()).hexdigest(),
            float(score),
        ),
    )


# ---------------------------------------------------------------------------
# Provider abstraction
# ---------------------------------------------------------------------------
class LLMProvider(ABC):
    """Minimal interface every provider must implement."""

    name: str = "abstract"
    model: str = "abstract"

    @abstractmethod
    def score_paragraph(self, text: str, category: str) -> float:
        """Return P(category | text) ∈ [0, 1]."""


class MockProvider(LLMProvider):
    """Deterministic provider for tests and offline development.

    Returns 1.0 if any keyword in ``hot_keywords`` appears in the text,
    else 0.0. Useful for testing the cache + aggregation logic without
    making real API calls.
    """
    name = "mock"
    model = "mock-v0"

    def __init__(self, hot_keywords: list[str] | None = None):
        self.hot_keywords = [k.lower() for k in (hot_keywords or ["uncertain"])]

    def score_paragraph(self, text: str, category: str) -> float:
        return 1.0 if any(k in text.lower() for k in self.hot_keywords) else 0.0


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider. Lazy-imports ``anthropic`` at construction."""
    name = "anthropic"

    def __init__(self, *, model: str = "claude-haiku-4-5-20251001",
                 api_key: str | None = None):
        try:
            import anthropic  # type: ignore
        except ImportError as e:
            raise ImportError(
                "anthropic SDK not installed. Run `pip install puremacro[llm]` "
                "to enable Anthropic-backed scoring."
            ) from e
        self.model = model
        self._client = anthropic.Anthropic(
            api_key=credentials.require("anthropic", explicit=api_key),
        )

    def score_paragraph(self, text: str, category: str) -> float:
        import anthropic as _anthropic  # already imported at __init__; safe to re-import
        prompt = (
            f"Score how strongly this paragraph expresses '{category}' "
            f"on a scale from 0.0 (not at all) to 1.0 (strongly). "
            f"Reply with ONLY a single decimal number.\n\n"
            f"Paragraph: {text}"
        )
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=8,
            messages=[{"role": "user", "content": prompt}],
        )
        text_blocks = [b for b in resp.content if isinstance(b, _anthropic.types.TextBlock)]
        content = text_blocks[0].text.strip() if text_blocks else "0"
        try:
            return max(0.0, min(1.0, float(content)))
        except ValueError:
            return 0.0


class LocalProvider(LLMProvider):
    """Free local-LLM provider for ``llm_prob_kernel``, through :func:`puremacro.llm.complete`.

    No API key. Every ``score_paragraph`` goes through the facade, which reads the
    completion cache first and applies the mode (``mode``, else ``PUREMACRO_LLM_MODE``,
    else ``auto``); no engine is looked for at construction, so in ``cache`` mode a cached
    corpus is scored with no engine installed. ``name`` is ``"local"`` and ``model`` the
    canonical alias (:func:`puremacro.llm.canonical`), so the SQLite score cache rows of
    this kernel are shared by every engine. Robustly extracts the first float in the reply.
    """

    name = "local"

    def __init__(self, model: str = "default", *,
                 engine: str = "auto", base_url: str | None = None,
                 timeout: float = 120.0, mode: str | None = None):
        from puremacro.llm import canonical
        self.model = canonical(model)
        self.engine = engine
        self.base_url = base_url
        self.timeout = timeout
        self.mode = mode

    def score_paragraph(self, text: str, category: str) -> float:
        import re

        from puremacro.llm import complete
        prompt = (
            f"Score how strongly this paragraph expresses '{category}' "
            f"on a scale from 0.0 (not at all) to 1.0 (strongly). "
            f"Reply with ONLY a single decimal number.\n\nParagraph: {text}"
        )
        out = complete(prompt, model=self.model, engine=self.engine, mode=self.mode,
                       max_tokens=8, temperature=0.0, json_mode=False,
                       timeout=self.timeout, base_url=self.base_url)
        m = re.search(r"\d*\.?\d+", out or "")
        if not m:
            return 0.0
        try:
            return max(0.0, min(1.0, float(m.group())))
        except ValueError:
            return 0.0


class OllamaProvider(LocalProvider):
    """Preset: ``LocalProvider(engine="ollama")``.

    The default is the registry alias ``qwen3-4b``, so the completion cache is keyed as for
    ``LocalProvider("default")``; the server receives its own tag (``qwen3:4b``). An Ollama tag
    given as ``model`` reaches the server, and keys the cache, unchanged.
    """

    def __init__(self, model: str = "qwen3-4b", *,
                 base_url: str = "http://localhost:11434", **kw):
        super().__init__(model=model, engine="ollama", base_url=base_url, **kw)


def get_default_provider(model: str = "default", **kw) -> LLMProvider:
    """A :class:`LocalProvider`, or :class:`MockProvider` when no engine can serve it.

    In ``cache`` mode the local provider is returned without probing anything (it reads
    the completion cache). In ``auto`` and ``off`` modes it is returned when
    :func:`puremacro.llm.engines.available_engines` lists an engine (the one named by
    ``engine``, if given), else the install hint is printed and ``MockProvider`` is
    returned. Prints
    ``[puremacro] local LLM: mode=<mode> engine=<name|cache|mock> model=<alias>``.
    Keywords go to :class:`LocalProvider`.
    """
    from puremacro.llm import current_mode
    from ..scoring.llm import _local_engine_for, _print_local_choice
    p = LocalProvider(model=model, **kw)
    mode = current_mode(p.mode)
    chosen = _local_engine_for(mode=mode, engine=p.engine, base_url=p.base_url)
    _print_local_choice(mode=mode, engine=chosen, model=p.model)
    return p if chosen is not None else MockProvider()


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
def _prompt_for(category: str) -> str:
    return (
        f"Score the strength with which this paragraph expresses "
        f"'{category}' on a 0.0–1.0 scale. Reply with ONLY the decimal."
    )


def llm_prob_kernel(
    records: Iterable[tuple],
    *,
    provider: LLMProvider,
    category: str,
    window: str = "paragraph",
    language: str = "en",
    max_calls: int | None = 5000,
) -> Iterator[tuple]:
    """Yield ``(date, mean_P_category)`` per document.

    Score per window is provided by ``provider.score_paragraph(text, category)``;
    results are cached on disk by (provider, model, prompt_hash, text_hash).
    The mean over all windows in a doc is yielded.

    Parameters
    ----------
    max_calls : safety cap on API calls per kernel invocation. Set to
        None to disable. Default 5000 (≈ $5–30 at Haiku pricing).
        ``PUREMACRO_LLM_BUDGET=spend`` removes the cap entirely.
    """
    if os.environ.get("PUREMACRO_LLM_BUDGET") == "spend":
        max_calls = None
    if window not in ("paragraph", "sentence"):
        raise ValueError(f"window must be 'sentence' or 'paragraph'; got {window!r}")

    splitter = _split_paragraphs if window == "paragraph" else _split_sentences
    prompt = _prompt_for(category)

    cache_path = _cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(cache_path))
    _ensure_cache_table(conn)

    n_calls = 0
    try:
        for record in records:
            if len(record) == 4:
                date, text, _, meta = record
                lang = (meta or {}).get("language", language)
            else:
                date, text, _ = record
                lang = language
            windows = splitter(text or "", lang)
            if not windows:
                yield (pd.Timestamp(date), 0.0)
                continue
            scores = []
            for w in windows:
                key = _cache_key(provider.name, provider.model, prompt, w)
                cached = _cache_get(conn, key)
                if cached is not None:
                    scores.append(cached)
                    continue
                if max_calls is not None and n_calls >= max_calls:
                    scores.append(0.0)  # over budget — bail
                    continue
                s = provider.score_paragraph(w, category)
                n_calls += 1
                _cache_set(conn, key, provider.name, provider.model,
                           prompt, w, s)
                scores.append(s)
            conn.commit()
            yield (pd.Timestamp(date), float(sum(scores) / len(scores)))
    finally:
        conn.close()


__all__ = [
    "LLMProvider",
    "MockProvider",
    "AnthropicProvider",
    "LocalProvider",
    "OllamaProvider",
    "get_default_provider",
    "llm_prob_kernel",
]
