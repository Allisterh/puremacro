"""LLM-backed event scoring via direct HTTP calls.

Two backends share an interface so the rest of the package is
backend-agnostic:

    backend = AnthropicBackend(model="claude-sonnet-4-6")
    backend = OpenAIBackend(model="gpt-4o-mini")
    events = score_llm(text_iter, backend=backend, kind="fiscal", country="USA")

API keys are resolved via ``puremacro.credentials`` (env vars,
TOML config, or explicit kwarg). The HTTP layer uses ``urllib.request`` only —
works in Pyodide via the JS-fetch proxy with no SDK dependency.

Strict JSON-schema validation: malformed events are dropped (a warning
counter is kept) rather than raising — pipelines stay running.

The ``kind`` parameter (default ``"fiscal"``) selects one of five
prompt templates. The language preamble lets a single prompt extract
events from non-English text natively (Anthropic / OpenAI both handle
multilingual extraction).
"""
from __future__ import annotations

import json
from puremacro._optional_stdlib import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from puremacro import credentials
from puremacro.llm import CacheMiss
from .._local_engines import BackendUnavailable
from ..types import NarrativeEvent, VALID_SIGNS


_LANGUAGE_PREAMBLE = (
    "The text below may be in any language; the field 'language' is "
    "{language} (ISO-639-1). Extract events regardless of language. "
    "Return all field labels in English."
)


_PROMPTS = {
    "fiscal": """You are extracting narrative fiscal-policy events from
press text. From the input below, output a JSON array of zero or more
events with this exact schema (one JSON object per event):

  {{
    "magnitude_usd_bn": float,             // size in USD billions (0 if not stated)
    "target": "investment" | "consumption" | "both",
    "subtarget": string | null,            // e.g. "defense", "infra", "transfers"
    "sign": -1 | 0 | +1,                   // +1 expansionary, -1 contractionary
    "confidence": float in [0, 1],
    "excerpt": string,                     // <=240 chars from the source
    "implementation_date": "YYYY-MM-DD" | null,
    "horizon_quarters": int 0..16 | null
  }}

Only include events that announce a *discrete change* in government
spending (consumption or capital), not retrospective reporting or
forecasts. Output ONLY the JSON array, no surrounding prose.

Field rules for the timing extraction:

- ``implementation_date``: if the text states an effective or
  implementation date for the announced policy, return it as an ISO
  ``YYYY-MM-DD`` string. If multiple effective dates are mentioned,
  return the FIRST one. Return ``null`` if no effective date is mentioned.

- ``horizon_quarters``: only return this if ``implementation_date`` is
  null AND the text gives a vague horizon. Return an integer 0–16
  representing quarters between announcement and first effective quarter.

{language_preamble}

Country: {country}
Date: {date}

Text:
\"\"\"
{text}
\"\"\"
""",
    "monetary": """You are extracting narrative MONETARY-policy events
from central-bank text. From the input below, output a JSON array of
zero or more events with this exact schema:

  {{
    "magnitude_bps": float,                // policy-rate change in basis points (0 if not stated)
    "target": "policy_rate" | "asset_purchase" | "forward_guidance"
              | "fx_intervention" | "lending_facility",
    "subtarget": string | null,
    "sign": -1 | 0 | +1,                   // +1 hawkish/tighten, -1 dovish/ease, 0 neutral
    "hawkish_dovish_prob": float in [0, 1],
    "confidence": float in [0, 1],
    "excerpt": string,
    "implementation_date": "YYYY-MM-DD" | null,
    "horizon_quarters": int 0..16 | null
  }}

Only include discrete monetary-policy decisions or unambiguous
forward-guidance announcements. Skip retrospective discussion and
forecasts. Output ONLY the JSON array.

{language_preamble}

Country: {country}
Date: {date}

Text:
\"\"\"
{text}
\"\"\"
""",
    "macropru": """You are extracting MACROPRUDENTIAL-policy events.
JSON array, one object per event:

  {{
    "target": "capital_buffer" | "ltv_dsti" | "sector_limit" | "reserve_requirement",
    "magnitude_pct": float,
    "subtarget": string | null,
    "sign": -1 | 0 | +1,                   // +1 tightening, -1 loosening
    "confidence": float in [0, 1],
    "excerpt": string,
    "implementation_date": "YYYY-MM-DD" | null,
    "horizon_quarters": int 0..16 | null
  }}

Only include announced policy actions, not commentary. Output JSON
array only.

{language_preamble}

Country: {country}
Date: {date}

Text:
\"\"\"
{text}
\"\"\"
""",
    "fx": """You are extracting FX-INTERVENTION or peg-change events.
JSON array, one object per event:

  {{
    "target": "intervention" | "peg_change",
    "magnitude_usd_bn": float,
    "subtarget": string | null,
    "sign": -1 | 0 | +1,                   // +1 buying domestic / defending, -1 selling
    "confidence": float in [0, 1],
    "excerpt": string,
    "implementation_date": "YYYY-MM-DD" | null,
    "horizon_quarters": int 0..16 | null
  }}

Output JSON array only.

{language_preamble}

Country: {country}
Date: {date}

Text:
\"\"\"
{text}
\"\"\"
""",
    "structural": """You are extracting STRUCTURAL-REFORM events. JSON
array, one object per event:

  {{
    "target": "labor" | "product_market" | "trade" | "tax_admin",
    "magnitude_z": float,                  // qualitative scale, ~[0, 3]
    "subtarget": string | null,
    "sign": -1 | 0 | +1,                   // +1 liberalizing, -1 restrictive
    "confidence": float in [0, 1],
    "excerpt": string,
    "implementation_date": "YYYY-MM-DD" | null,
    "horizon_quarters": int 0..16 | null
  }}

Output JSON array only.

{language_preamble}

Country: {country}
Date: {date}

Text:
\"\"\"
{text}
\"\"\"
""",
}


_MAGNITUDE_KEY_BY_KIND = {
    "fiscal":     "magnitude_usd_bn",
    "monetary":   "magnitude_bps",
    "macropru":   "magnitude_pct",
    "fx":         "magnitude_usd_bn",
    "structural": "magnitude_z",
}

_MAGNITUDE_UNIT_BY_KIND = {
    "fiscal":     "USD_bn",
    "monetary":   "bps",
    "macropru":   "ratio",
    "fx":         "USD_bn",
    "structural": "z",
}


def _build_prompt(*, kind, language, country, date, text):
    if kind not in _PROMPTS:
        raise ValueError(f"kind {kind!r} not in {sorted(_PROMPTS)}")
    return _PROMPTS[kind].format(
        country=country,
        date=date,
        text=str(text)[:6000],
        language_preamble=_LANGUAGE_PREAMBLE.format(language=language),
    )


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------
@dataclass
class _BackendBase:
    model: str
    max_tokens: int = 1024
    temperature: float = 0.0

    def call(self, prompt: str) -> str:  # pragma: no cover (abstract)
        raise NotImplementedError


class AnthropicBackend(_BackendBase):
    """Anthropic Messages API client (urllib-based)."""

    def __init__(self, model: str = "claude-sonnet-4-6", *,
                 api_key: str | None = None, **kw):
        super().__init__(model=model, **kw)
        self.api_key = credentials.require("anthropic", explicit=api_key)

    def call(self, prompt: str) -> str:
        payload = json.dumps({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        for block in body.get("content", []):
            if block.get("type") == "text":
                return block["text"]
        return ""


class OpenAIBackend(_BackendBase):
    """OpenAI Chat Completions client (urllib-based)."""

    def __init__(self, model: str = "gpt-4o-mini", *,
                 api_key: str | None = None, **kw):
        super().__init__(model=model, **kw)
        self.api_key = credentials.require("openai", explicit=api_key)

    def call(self, prompt: str) -> str:
        payload = json.dumps({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "content-type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        choices = body.get("choices") or []
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "")


class LocalBackend(_BackendBase):
    """Free local-LLM backend for ``score_llm``, through :func:`puremacro.llm.complete`.

    No API key. Every ``call`` goes through the facade, which reads the completion cache
    first and applies the mode (``mode``, else ``PUREMACRO_LLM_MODE``, else ``auto``); no
    engine is looked for at construction, so in ``cache`` mode a cached corpus is scored
    with no engine installed. ``model`` is a tier (``"default"``, ``"small"``,
    ``"large"``), an alias of :data:`puremacro.llm.MODELS` or an engine's own id, and is
    kept as its canonical name (the alias the cache is keyed on). ``engine="auto"`` takes
    the first available engine; ``base_url`` names one HTTP server (Ollama / LM Studio /
    OpenAI-compatible), else ``PUREMACRO_LLM_BASE_URL`` or the default local servers.
    ``json_mode`` is off by default: the prompts ask for a JSON array, which an engine's
    object-only JSON mode would forbid.
    """

    def __init__(self, model: str = "default", *,
                 engine: str = "auto", base_url: str | None = None,
                 timeout: float = 120.0, json_mode: bool = False,
                 mode: str | None = None, **kw):
        from puremacro.llm import canonical
        super().__init__(model=canonical(model), **kw)
        self.engine = engine
        self.base_url = base_url
        self.timeout = timeout
        self.json_mode = json_mode
        self.mode = mode

    def call(self, prompt: str) -> str:
        from puremacro.llm import complete
        return complete(prompt, model=self.model, engine=self.engine, mode=self.mode,
                        max_tokens=self.max_tokens, temperature=self.temperature,
                        json_mode=self.json_mode, timeout=self.timeout,
                        base_url=self.base_url)


class OllamaBackend(LocalBackend):
    """Preset: ``LocalBackend(engine="ollama")`` for a running Ollama/LM Studio."""

    def __init__(self, model: str = "qwen3:4b", *,
                 base_url: str = "http://localhost:11434", **kw):
        super().__init__(model=model, engine="ollama", base_url=base_url, **kw)


class MockBackend(_BackendBase):
    """Offline fallback: returns an empty JSON array (zero events), so
    pipelines/notebooks run deterministically when no engine is installed."""

    def __init__(self, model: str = "mock", **kw):
        super().__init__(model=model, **kw)

    def call(self, prompt: str) -> str:
        return "[]"


def _local_engine_for(*, mode: str, engine: str, base_url: str | None) -> str | None:
    """The engine a local object would use, ``"cache"`` in cache mode, or None if none is
    available. Cache mode probes nothing; the other modes ask
    :func:`puremacro.llm.engines.available_engines`."""
    if mode == "cache":
        return "cache"
    from puremacro.llm import engines as _engines
    names = _engines.available_engines(base_url=base_url)
    if engine == "auto":
        return names[0] if names else None
    return engine if engine in names else None


def _print_local_choice(*, mode: str, engine: str | None, model: str) -> None:
    if engine is None:
        from puremacro.llm.engines import INSTALL_HINT
        print(f"[puremacro] {INSTALL_HINT}")
    print(f"[puremacro] local LLM: mode={mode} engine={engine or 'mock'} model={model}")


def get_default_backend(model: str = "default", **kw) -> _BackendBase:
    """A :class:`LocalBackend`, or :class:`MockBackend` when no engine can serve it.

    In ``cache`` mode the local backend is returned without probing anything (it reads the
    completion cache). In ``auto`` and ``off`` modes it is returned when
    :func:`puremacro.llm.engines.available_engines` lists an engine (the one named by
    ``engine``, if given), else the install hint is printed and ``MockBackend`` (zero
    events) is returned, so demos and notebooks always run. Prints
    ``[puremacro] local LLM: mode=<mode> engine=<name|cache|mock> model=<alias>``.
    Keywords go to :class:`LocalBackend`.
    """
    from puremacro.llm import current_mode
    be = LocalBackend(model=model, **kw)
    mode = current_mode(be.mode)
    chosen = _local_engine_for(mode=mode, engine=be.engine, base_url=be.base_url)
    _print_local_choice(mode=mode, engine=chosen, model=be.model)
    return be if chosen is not None else MockBackend()


# ---------------------------------------------------------------------------
# Top-level scoring loop
# ---------------------------------------------------------------------------
def _drop_plus_signs(text: str) -> str:
    """``text`` without the ``+`` that precedes a digit outside JSON strings.

    JSON has no unary plus, yet the prompts write ``"sign": -1 | 0 | +1`` and models copy
    it (``"sign": +1``). Characters inside double-quoted strings, escapes included, are
    left as they are.
    """
    out: list[str] = []
    in_string = escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "+" and i + 1 < len(text) and text[i + 1].isdigit():
            continue
        out.append(ch)
    return "".join(out)


def _parse_response(text: str) -> list[dict]:
    """Best-effort JSON-array extraction from an LLM response.

    After a surrounding code fence is removed, a ``+`` before a digit outside strings is
    dropped (``"sign": +1`` reads as ``1``). A JSON array is returned as it is; a JSON
    object is accepted when it carries an ``"events"`` list, which is returned. Otherwise
    the text from the first ``[`` to the last ``]`` is tried. Anything else gives ``[]``.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        parsed = json.loads(_drop_plus_signs(text))
    except json.JSONDecodeError:
        try:
            i = text.index("[")
            j = text.rindex("]")
            parsed = json.loads(_drop_plus_signs(text[i:j + 1]))
        except (ValueError, json.JSONDecodeError, Exception):
            return []
    if isinstance(parsed, dict) and isinstance(parsed.get("events"), list):
        return parsed["events"]
    if not isinstance(parsed, list):
        return []
    return parsed


def _validate_event_dict(d: dict, *, kind: str) -> bool:
    from ..types import VALID_TARGETS_BY_KIND
    if not isinstance(d, dict):
        return False
    if d.get("target") not in VALID_TARGETS_BY_KIND[kind]:
        return False
    try:
        if int(d.get("sign", 99)) not in VALID_SIGNS:
            return False
    except (ValueError, TypeError, Exception):
        return False
    mag_key = _MAGNITUDE_KEY_BY_KIND[kind]
    if not isinstance(d.get(mag_key, None), (int, float)):
        return False
    if not isinstance(d.get("confidence", None), (int, float)):
        return False
    return True


def _profile_from_timing_fields(
    ev_dict: dict,
    *,
    announcement,
) -> list[tuple[pd.Timestamp, float]]:
    """Build implementation_profile from the LLM's timing fields.

    Prefers ``implementation_date`` (ISO string) over ``horizon_quarters``
    (int 0–16 fallback), which is read only when there is no date. Both null or
    unparseable → empty list.

    An empty list means "implemented in the announcement quarter, at the
    announcement date" (``NarrativeEvent.effective_profile``). So a date in the
    announcement quarter, and a horizon of 0, give the empty list: the start of
    that quarter would predate the announcement, which ``NarrativeEvent``
    refuses. A date in a later quarter gives ``[(start of that quarter, 1.0)]``,
    a horizon ``h`` in 1..16 ``[(start of the h-th quarter after, 1.0)]``.

    Refuses to emit a profile that predates the announcement quarter.
    """
    impl_date_str = ev_dict.get("implementation_date")
    horizon_q = ev_dict.get("horizon_quarters")

    if impl_date_str:
        try:
            impl_q = pd.Period(pd.Timestamp(impl_date_str), freq="Q").to_timestamp()
            ann_q = pd.Period(pd.Timestamp(announcement), freq="Q").to_timestamp()
            if impl_q > ann_q:
                return [(impl_q, 1.0)]
        except (ValueError, TypeError, Exception):
            pass
        return []

    if horizon_q is not None:
        try:
            h = int(horizon_q)
            if 1 <= h <= 16:
                ann_q = pd.Period(pd.Timestamp(announcement), freq="Q")
                return [((ann_q + h).to_timestamp(), 1.0)]
        except (ValueError, TypeError, Exception):
            return []
    return []


def score_llm(
    text_iter: Iterable[tuple],
    *,
    backend: _BackendBase,
    kind: str = "fiscal",
    language: str = "en",
    country: str = "USA",
    dry_run: bool = False,
) -> list[NarrativeEvent]:
    """Score each record with an LLM, parameterized by ``kind`` and ``language``.

    ``text_iter`` accepts both 3-tuple ``(date, text, source_url)`` and
    4-tuple ``(date, text, source_url, metadata)`` records. When a 4-tuple
    is passed, ``metadata["language"]`` overrides the function-level
    ``language`` per record.

    With ``dry_run=True``, return an empty list and print an estimated
    cost (rough — based on character count → token approximation).

    A ``BackendUnavailable`` error (e.g. a local engine/server that is
    unreachable) propagates to the caller rather than being dropped as malformed.
    So does a :class:`puremacro.llm.CacheMiss` (including ``CacheUnavailable``),
    which a :class:`LocalBackend` raises in ``cache`` mode for a record the
    completion cache does not hold: the corpus was not scored ahead of time, and
    dropping the record would return a silently shorter list of events. Any other
    failure of a record's call (a model that cannot load, a timeout) is counted
    with the malformed events and the loop goes on.
    """
    if kind not in _PROMPTS:
        raise ValueError(f"kind {kind!r} not in {sorted(_PROMPTS)}")

    items = list(text_iter)
    if dry_run:
        chars = sum(len(r[1]) for r in items)
        approx_tokens = chars / 4.0
        print(f"[score_llm dry-run kind={kind} lang={language}] "
              f"{len(items)} items, ~{approx_tokens:,.0f} input tokens, "
              f"~{len(items) * 200:,.0f} output tokens.")
        return []

    out: list[NarrativeEvent] = []
    n_dropped_malformed = 0
    for record in items:
        if len(record) == 4:
            date, text, source_url, meta = record
            record_lang = (meta or {}).get("language", language)
        else:
            date, text, source_url = record
            record_lang = language
        prompt = _build_prompt(
            kind=kind, language=record_lang, country=country,
            date=str(date), text=text,
        )
        try:
            response = backend.call(prompt)
        except (BackendUnavailable, CacheMiss):
            raise
        except (ValueError, ArithmeticError, Exception):
            n_dropped_malformed += 1
            continue
        for ev_dict in _parse_response(response):
            if not _validate_event_dict(ev_dict, kind=kind):
                n_dropped_malformed += 1
                continue
            mag_key = _MAGNITUDE_KEY_BY_KIND[kind]
            magnitude_unit = _MAGNITUDE_UNIT_BY_KIND[kind]
            profile = _profile_from_timing_fields(ev_dict, announcement=date)
            try:
                event = NarrativeEvent(
                    date=pd.Timestamp(date),
                    country=country,
                    magnitude=float(ev_dict[mag_key]),
                    magnitude_unit=magnitude_unit,
                    target=ev_dict["target"],
                    subtarget=ev_dict.get("subtarget"),
                    sign=int(ev_dict["sign"]),
                    confidence=float(ev_dict["confidence"]),
                    source_text=str(ev_dict.get("excerpt", text))[:500],
                    source_url=str(source_url),
                    scoring_method="llm",
                    implementation_profile=profile,
                    kind=kind,
                    language=record_lang,
                )
            except ValueError:      # an event NarrativeEvent refuses is dropped, not raised
                n_dropped_malformed += 1
                continue
            out.append(event)
    if n_dropped_malformed:
        print(f"[score_llm] dropped {n_dropped_malformed} malformed events.")
    return out


__all__ = ["AnthropicBackend", "OpenAIBackend", "score_llm",
           "LocalBackend", "OllamaBackend", "MockBackend", "get_default_backend",
           "_PROMPTS", "_build_prompt"]
