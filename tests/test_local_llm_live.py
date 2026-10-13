# tests/test_local_llm_live.py
"""Opt-in live tests against real local engines. Each SKIPS if its engine
(or model) is unavailable, or if the reference model does not fit this
machine's memory, so CI without engines stays green.

Run locally after:  pip install -e ".[local-llm]"   (MLX needs Apple Silicon)
and, for the Ollama case, `ollama serve` + `ollama pull qwen3:4b`. Every test
uses the default model (qwen3-4b, the reference model of the course caches).
Run with `-m local_llm`.
"""
import pytest

from puremacro.llm import LocalLLMUnavailable, ModelTooLarge
from puremacro.llm.engines import resolve_engine

pytestmark = [pytest.mark.local_llm, pytest.mark.slow]

_CALM = "The committee kept policy unchanged; conditions were stable and as expected."
_UNCERTAIN = ("Officials warned of highly uncertain, unpredictable risks; the "
              "outlook is murky and could shift abruptly in either direction.")

# The facade raises these before or instead of an answer when this machine cannot serve the model:
# the engine has not got it (a server that has not pulled it), or the model needs more memory than
# the device offers. Either is a skip, not a failure.
_CANNOT_SERVE = (LocalLLMUnavailable, ModelTooLarge)


def _engine_or_skip(name):
    try:
        return resolve_engine(name, timeout=5)
    except LocalLLMUnavailable:
        pytest.skip(f"engine {name!r} not available")


@pytest.mark.parametrize("engine_name", ["mlx", "llamacpp", "ollama"])
def test_local_provider_directional(engine_name):
    _engine_or_skip(engine_name)
    from puremacro.narrative.indices import LocalProvider
    p = LocalProvider(engine=engine_name)
    try:                    # no engine is touched before the first call (puremacro.llm)
        hi = p.score_paragraph(_UNCERTAIN, "economic uncertainty")
        lo = p.score_paragraph(_CALM, "economic uncertainty")
    except _CANNOT_SERVE as exc:
        pytest.skip(f"{engine_name} unusable: {exc}")
    assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0
    assert hi >= lo  # uncertain text scores at least as high as calm text


@pytest.mark.parametrize("engine_name", ["mlx", "llamacpp", "ollama"])
def test_local_backend_event_json_parses(engine_name):
    _engine_or_skip(engine_name)
    from puremacro.narrative.scoring import LocalBackend, score_llm
    be = LocalBackend(engine=engine_name)
    rec = ("2020-03-15",
           "The government announced a $500 billion infrastructure spending package.",
           "http://example.test")
    try:                    # no engine is touched before the first call (puremacro.llm)
        events = score_llm([rec], backend=be, kind="fiscal")
    except _CANNOT_SERVE as exc:      # score_llm re-raises both (configuration, not malformed output)
        pytest.skip(f"{engine_name} unusable: {exc}")
    # We assert structure, not exact extraction (small models vary):
    assert isinstance(events, list)
    for ev in events:
        assert ev.kind == "fiscal"
        assert ev.sign in (-1, 0, 1)


# --- puremacro.llm facade -----------------------------------------------------------------------
_QUESTION_ES = ("Responde con un número entre 0 y 1: ¿cuánta incertidumbre expresa "
                "'el panorama es muy incierto'?")


def _available_or_skip(engine_name):
    from puremacro.llm import available_engines
    if engine_name not in available_engines():
        pytest.skip(f"engine {engine_name!r} not available")


@pytest.mark.parametrize("engine_name", ["mlx", "llamacpp", "ollama", "openai"])
def test_facade_live_thinking_off(engine_name):
    """The default model answers through the facade with thinking off: a bare number, no
    ``<think>`` block, within the 8 tokens a score needs (off mode: no cache read or write)."""
    _available_or_skip(engine_name)
    from puremacro.llm import complete
    try:
        out = complete(_QUESTION_ES, engine=engine_name, mode="off", max_tokens=8)
    except _CANNOT_SERVE as exc:
        pytest.skip(f"{engine_name} unusable: {exc}")
    assert "<think>" not in out and "</think>" not in out
    value = float(out.strip().rstrip(".").replace(",", "."))
    assert 0.0 <= value <= 1.0


@pytest.mark.parametrize("engine_name", ["mlx", "llamacpp"])
def test_facade_live_loads_offline_after_download(engine_name):
    """Once the weights are in the Hugging Face cache, the in-process engines load them with
    ``HF_HUB_OFFLINE=1`` (a fresh interpreter, so nothing is reused from this one)."""
    import os
    import subprocess
    import sys
    _available_or_skip(engine_name)
    from puremacro.llm import complete
    try:                                         # make sure the weights are downloaded
        complete(_QUESTION_ES, engine=engine_name, mode="off", max_tokens=8)
    except _CANNOT_SERVE as exc:
        pytest.skip(f"{engine_name} unusable: {exc}")
    code = ("from puremacro.llm import complete; "
            f"print(repr(complete({_QUESTION_ES!r}, engine={engine_name!r}, mode='off', max_tokens=8)))")
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    run = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                         timeout=600)
    assert run.returncode == 0, run.stderr[-2000:]
    assert run.stdout.strip()
