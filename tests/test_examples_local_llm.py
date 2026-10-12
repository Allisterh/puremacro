"""Smoke: the local-LLM example runs end-to-end on the Mock fallback.

The no-engine path is forced by monkeypatching ``puremacro.llm.engines``: with
``available_engines`` empty, ``get_default_backend`` / ``get_default_provider`` return
the Mock objects, and ``resolve_engine`` raises in case anything still asks for an
engine. On a machine with a real engine installed (e.g. mlx-lm on Apple Silicon, or a
running Ollama), the example would otherwise load — and on first use download — a real
model, which is a live-network action this offline smoke test must never take.
"""
import importlib

import puremacro.llm.engines as engines


def test_example_main_runs(capsys, monkeypatch):
    monkeypatch.setattr(engines, "available_engines", lambda **k: [])
    monkeypatch.setattr(
        engines, "resolve_engine",
        lambda *a, **k: (_ for _ in ()).throw(engines.LocalLLMUnavailable("forced offline")),
    )
    mod = importlib.import_module("puremacro.examples.narrative_local_llm")
    mod.main()  # no engine available -> get_default_* fall back to Mock
    out = capsys.readouterr().out
    assert "engine=mock" in out
    assert "model=qwen3-4b" in out          # the example asks for model "default"


def test_example_docstring_names_the_reference_model():
    mod = importlib.import_module("puremacro.examples.narrative_local_llm")
    assert "ollama pull qwen3:4b" in mod.__doc__
    assert "qwen2.5" not in mod.__doc__
