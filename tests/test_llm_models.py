# tests/test_llm_models.py
"""Model registry and tier picker of ``puremacro.llm`` (spec section 3.3).

The first block is the contract of the task brief. The blocks below pin the ids the
registry ships (Hugging Face file names checked on 2026-10-12), the edge cases of the
picker and of the budget probe, and the rule that keeps the module importable where
mlx, llama.cpp and sockets do not exist.
"""
import dataclasses
import importlib.util
import json
import os
import pickle
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

import puremacro
from puremacro.llm import models as M

_ENV = "PUREMACRO_LLM_MODEL"     # removed before every test by tests/conftest.py
GIB = 1024 ** 3


# ---------------------------------------------------------------- the brief's contract

def test_ladder_ids():
    assert M.MODELS["qwen3-4b"].mlx == "mlx-community/Qwen3-4B-4bit"
    assert M.MODELS["qwen3-4b"].llamacpp == ("Qwen/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf")
    assert M.MODELS["qwen3-4b"].ollama == "qwen3:4b"
    assert all(M.MODELS[a].thinking for a in ("qwen3-1.7b", "qwen3-4b", "qwen3-8b"))
    assert M.MODELS["qwen2.5-3b-instruct"].mlx == "mlx-community/Qwen2.5-3B-Instruct-4bit"


def test_canonical_and_env_override(monkeypatch):
    assert M.canonical("default") == "qwen3-4b" and M.canonical("small") == "qwen3-1.7b"
    assert M.canonical("qwen3:4b") == "qwen3:4b"
    monkeypatch.setenv("PUREMACRO_LLM_MODEL", "qwen3-1.7b")
    assert M.canonical("default") == "qwen3-1.7b"


def test_resolve_model_id():
    assert M.resolve_model_id("default", "ollama") == "qwen3:4b"
    assert M.resolve_model_id("qwen3-4b", "openai") == "qwen3-4b"
    assert M.resolve_model_id("my-local.gguf", "llamacpp") == "my-local.gguf"
    assert M.resolve_model_id("qwen3-4b", "fake") == "qwen3-4b"


@pytest.mark.parametrize("budget,alias", [(4.0, "qwen3-1.7b"), (5.0, "qwen3-4b"), (8.0, "qwen3-4b"), (12.0, "qwen3-4b")])
def test_pick_model_by_budget(budget, alias):
    assert M.pick_model(budget_gb=budget).alias == alias


def test_pick_model_explicit_tier():
    assert M.pick_model("large", budget_gb=4.0).alias == "qwen3-8b"


def test_check_fits_suggests_smaller():
    with pytest.raises(M.ModelTooLarge) as ei:
        M.check_fits(M.MODELS["qwen3-8b"], 8.0)
    assert ei.value.alias == "qwen3-8b" and ei.value.suggested == "qwen3-4b" and ei.value.budget_gb == 8.0
    M.check_fits(M.MODELS["qwen3-4b"], 5.0)          # no raise at the threshold
    with pytest.raises(M.ModelTooLarge) as ei:
        M.check_fits(M.MODELS["qwen3-1.7b"], 2.0)
    assert ei.value.suggested is None


def test_device_budget_without_mlx(monkeypatch):
    import importlib.util
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None if name == "mlx" else object())
    import puremacro.runtime._capabilities as cap
    monkeypatch.setattr(cap, "capabilities", lambda: type("C", (), {"memory_mb": 16384})())
    assert M.device_budget_gb() == pytest.approx(9.6)


# ---------------------------------------------------------------- the registry

# alias -> (params_b, approx_gb, min_budget_gb, thinking, license, mlx, llamacpp, ollama, openai)
# The three Qwen3 rungs, then the four models of the 4.8.1 table: the same mlx, ollama and
# openai ids as there, and the llama.cpp globs ("*Q4_K_M.gguf") replaced by the exact file.
_REGISTRY = {
    "qwen3-1.7b": (1.7, 2.0, 3.0, True, "apache-2.0", "mlx-community/Qwen3-1.7B-4bit",
                   ("ggml-org/Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q4_K_M.gguf"), "qwen3:1.7b", "qwen3-1.7b"),
    "qwen3-4b": (4.0, 3.5, 5.0, True, "apache-2.0", "mlx-community/Qwen3-4B-4bit",
                 ("Qwen/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf"), "qwen3:4b", "qwen3-4b"),
    "qwen3-8b": (8.0, 6.0, 10.0, True, "apache-2.0", "mlx-community/Qwen3-8B-4bit",
                 ("Qwen/Qwen3-8B-GGUF", "Qwen3-8B-Q4_K_M.gguf"), "qwen3:8b", "qwen3-8b"),
    "qwen2.5-3b-instruct": (3.0, 2.5, 3.0, False, "qwen-research", "mlx-community/Qwen2.5-3B-Instruct-4bit",
                            ("Qwen/Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-instruct-q4_k_m.gguf"),
                            "qwen2.5:3b", "qwen2.5-3b-instruct"),
    "gemma2-2b": (2.6, 2.0, 3.0, False, "gemma", "mlx-community/gemma-2-2b-it-4bit",
                  ("bartowski/gemma-2-2b-it-GGUF", "gemma-2-2b-it-Q4_K_M.gguf"), "gemma2:2b", "gemma2-2b"),
    "llama3.2-3b": (3.2, 2.5, 3.0, False, "llama3.2", "mlx-community/Llama-3.2-3B-Instruct-4bit",
                    ("bartowski/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
                    "llama3.2:3b", "llama3.2-3b"),
    "phi3.5": (3.8, 3.0, 3.0, False, "mit", "mlx-community/Phi-3.5-mini-instruct-4bit",
               ("bartowski/Phi-3.5-mini-instruct-GGUF", "Phi-3.5-mini-instruct-Q4_K_M.gguf"),
               "phi3.5", "phi3.5"),
}


def test_registry_holds_the_ladder_then_the_legacy_models_keyed_by_alias():
    assert list(M.MODELS) == list(_REGISTRY)
    assert all(key == spec.alias for key, spec in M.MODELS.items())


@pytest.mark.parametrize("alias", list(_REGISTRY))
def test_registry_entry(alias):
    assert dataclasses.astuple(M.MODELS[alias]) == (alias, *_REGISTRY[alias])


def test_llamacpp_ids_are_exact_file_names():
    # A glob can match no file or several in a repository: the official Qwen repository ships
    # no Q4_K_M file for the 1.7B, so that rung comes from ggml-org, and every entry names
    # the one file it means.
    for alias, spec in M.MODELS.items():
        repo, filename = spec.llamacpp
        assert not set("*?[") & set(filename), alias
        assert filename.endswith(".gguf") and "/" not in filename, alias
        assert repo.count("/") == 1, alias
    assert M.MODELS["qwen3-1.7b"].llamacpp[0] == "ggml-org/Qwen3-1.7B-GGUF"


def test_aliases_name_the_three_tiers():
    assert M.ALIASES == {"default": "qwen3-4b", "small": "qwen3-1.7b", "large": "qwen3-8b"}
    assert set(M.ALIASES.values()) <= set(M.MODELS)


def test_modelspec_is_a_frozen_record_in_the_documented_field_order():
    assert [f.name for f in dataclasses.fields(M.ModelSpec)] == [
        "alias", "params_b", "approx_gb", "min_budget_gb", "thinking", "license",
        "mlx", "llamacpp", "ollama", "openai"]
    spec = M.MODELS["qwen3-4b"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.alias = "other"
    assert hash(spec) == hash(dataclasses.replace(spec))


def test_every_spec_fits_inside_its_own_minimum_budget():
    for spec in M.MODELS.values():
        assert 0 < spec.approx_gb <= spec.min_budget_gb, spec.alias


# ---------------------------------------------------------------- canonical

@pytest.mark.parametrize("override,expected", [
    ("small", "qwen3-1.7b"), ("large", "qwen3-8b"), ("default", "qwen3-4b"),
    ("qwen3-8b", "qwen3-8b"), ("  qwen3-1.7b\n", "qwen3-1.7b"),
    ("qwen3:4b", "qwen3:4b"), ("my-local.gguf", "my-local.gguf"),
])
def test_override_may_name_a_tier_a_registry_alias_or_an_engine_id(monkeypatch, override, expected):
    monkeypatch.setenv(_ENV, override)
    assert M.canonical("default") == expected


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_override_is_no_override(monkeypatch, blank):
    monkeypatch.setenv(_ENV, blank)
    assert M.canonical("default") == "qwen3-4b"


def test_override_replaces_default_and_nothing_else(monkeypatch):
    monkeypatch.setenv(_ENV, "qwen3-8b")
    assert M.canonical("small") == "qwen3-1.7b" and M.canonical("large") == "qwen3-8b"
    assert M.canonical("qwen3-1.7b") == "qwen3-1.7b" and M.canonical("phi3.5") == "phi3.5"


def test_canonical_is_idempotent_on_known_names():
    for name in [*M.MODELS, *M.ALIASES]:
        assert M.canonical(M.canonical(name)) == M.canonical(name)


# ---------------------------------------------------------------- resolve_model_id

@pytest.mark.parametrize("alias", list(_REGISTRY))
def test_resolve_model_id_gives_each_engine_its_own_id(alias):
    spec = M.MODELS[alias]
    assert M.resolve_model_id(alias, "mlx") == spec.mlx
    assert M.resolve_model_id(alias, "llamacpp") == spec.llamacpp
    assert M.resolve_model_id(alias, "ollama") == spec.ollama
    assert M.resolve_model_id(alias, "openai") == spec.openai


def test_resolve_model_id_follows_tiers_and_the_override(monkeypatch):
    assert M.resolve_model_id("small", "mlx") == "mlx-community/Qwen3-1.7B-4bit"
    assert M.resolve_model_id("large", "llamacpp") == ("Qwen/Qwen3-8B-GGUF", "Qwen3-8B-Q4_K_M.gguf")
    # a tier handed to an engine the registry does not know (a test double): the canonical alias
    assert M.resolve_model_id("default", "fake") == "qwen3-4b"
    assert M.resolve_model_id("small", "fake") == "qwen3-1.7b"
    monkeypatch.setenv(_ENV, "large")
    assert M.resolve_model_id("default", "ollama") == "qwen3:8b"
    assert M.resolve_model_id("default", "fake") == "qwen3-8b"
    assert M.resolve_model_id("small", "ollama") == "qwen3:1.7b"     # an explicit tier ignores it


def test_an_override_outside_the_registry_reaches_every_engine_as_written(monkeypatch):
    monkeypatch.setenv(_ENV, "qwen3:4b")                             # an Ollama tag
    for engine in ("mlx", "llamacpp", "ollama", "openai", "fake"):
        assert M.resolve_model_id("default", engine) == "qwen3:4b"   # never the word "default"


@pytest.mark.parametrize("engine", ["alias", "license", "params_b", "thinking"])
def test_resolve_model_id_reads_the_four_engine_fields_and_no_other(engine):
    assert M.resolve_model_id("qwen3-4b", engine) == "qwen3-4b"


# ---------------------------------------------------------------- device_budget_gb

def _mlx_listed(monkeypatch, present):
    """Make ``importlib.util.find_spec("mlx")`` answer ``present``; other names are real."""
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: (
        (object() if present else None) if name == "mlx" else real(name, *a, **k)))


def _fake_mlx(monkeypatch, *, device_info=None, metal_device_info=None):
    """Install a stand-in ``mlx.core`` that offers only the given device_info functions."""
    _mlx_listed(monkeypatch, True)
    package, core = types.ModuleType("mlx"), types.ModuleType("mlx.core")
    package.__path__ = []
    package.core = core
    if device_info is not None:
        core.device_info = device_info
    if metal_device_info is not None:
        core.metal = types.SimpleNamespace(device_info=metal_device_info)
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", core)


def _memory_mb(monkeypatch, value):
    import puremacro.runtime._capabilities as cap
    monkeypatch.setattr(cap, "capabilities", lambda: types.SimpleNamespace(memory_mb=value))


def _raises(exc):
    def fail(*args, **kwargs):
        raise exc
    return fail


def test_device_budget_is_the_mlx_recommended_working_set(monkeypatch):
    calls = []

    def device_info():
        calls.append("device_info")
        return {"max_recommended_working_set_size": int(11.4 * GIB), "memory_size": 16 * GIB}

    _fake_mlx(monkeypatch, device_info=device_info,
              metal_device_info=lambda: calls.append("metal.device_info") or {})
    _memory_mb(monkeypatch, 4096)                    # would give 2.4: the MLX figure wins
    assert M.device_budget_gb() == pytest.approx(11.4)
    assert calls == ["device_info"]                  # the deprecated spelling is not touched


def test_device_budget_uses_mx_metal_device_info_when_mlx_has_no_device_info(monkeypatch):
    _fake_mlx(monkeypatch, metal_device_info=lambda: {"max_recommended_working_set_size": 6 * GIB})
    _memory_mb(monkeypatch, 65536)
    assert M.device_budget_gb() == pytest.approx(6.0)


@pytest.mark.parametrize("info", [
    _raises(RuntimeError("no Metal device")),
    lambda: {},
    lambda: None,
    lambda: {"max_recommended_working_set_size": 0},
    lambda: {"max_recommended_working_set_size": -1},
    lambda: {"max_recommended_working_set_size": float("nan")},
    lambda: {"max_recommended_working_set_size": float("inf")},
    lambda: {"max_recommended_working_set_size": "lots"},
], ids=["raises", "no-key", "not-a-dict", "zero", "negative", "nan", "inf", "not-a-number"])
def test_an_unusable_mlx_figure_falls_through_to_physical_memory(monkeypatch, info):
    _fake_mlx(monkeypatch, device_info=info)
    _memory_mb(monkeypatch, 16384)
    assert M.device_budget_gb() == pytest.approx(9.6)


def test_mlx_listed_but_not_importable_falls_through(monkeypatch):
    _mlx_listed(monkeypatch, True)
    monkeypatch.setitem(sys.modules, "mlx.core", None)      # `import mlx.core` raises ImportError
    _memory_mb(monkeypatch, 16384)
    assert M.device_budget_gb() == pytest.approx(9.6)


def test_a_find_spec_that_raises_counts_as_no_mlx(monkeypatch):
    real = importlib.util.find_spec

    def find_spec(name, *args, **kwargs):
        if name == "mlx":
            raise ValueError("mlx.__spec__ is None")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    _memory_mb(monkeypatch, 16384)
    assert M.device_budget_gb() == pytest.approx(9.6)


@pytest.mark.parametrize("memory_mb", [None, 0, -5])
def test_device_budget_assumes_eight_gb_when_memory_is_unknown(monkeypatch, memory_mb):
    _mlx_listed(monkeypatch, False)
    _memory_mb(monkeypatch, memory_mb)
    assert M.device_budget_gb() == 8.0


def test_device_budget_assumes_eight_gb_when_the_capability_probe_fails(monkeypatch):
    import puremacro.runtime._capabilities as cap
    _mlx_listed(monkeypatch, False)
    monkeypatch.setattr(cap, "capabilities", _raises(OSError("sysconf failed")))
    assert M.device_budget_gb() == 8.0


def test_budget_and_picker_never_ask_the_platform(monkeypatch):
    import platform
    asked = []
    for name in ("system", "machine", "platform", "processor", "uname", "mac_ver"):
        monkeypatch.setattr(platform, name, lambda *a, _name=name, **k: asked.append(_name) or "")
    _fake_mlx(monkeypatch, device_info=lambda: {"max_recommended_working_set_size": 12 * GIB})
    assert M.device_budget_gb() == pytest.approx(12.0)
    assert M.pick_model().alias == "qwen3-4b"
    _mlx_listed(monkeypatch, False)
    _memory_mb(monkeypatch, 4096)
    assert M.pick_model().alias == "qwen3-1.7b"
    assert asked == []


def test_device_budget_agrees_with_the_real_mlx_device_when_there_is_one():
    try:
        import mlx.core as mx
    except Exception as exc:                          # not installed, or no usable device
        pytest.skip(f"mlx is not usable here: {exc}")
    try:
        info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        expected = float(info["max_recommended_working_set_size"]) / GIB
    except Exception as exc:
        pytest.skip(f"this mlx build reports no working set: {exc}")
    assert M.device_budget_gb() == pytest.approx(expected)


# ---------------------------------------------------------------- pick_model

@pytest.mark.parametrize("budget,alias", [
    (0.5, "qwen3-1.7b"), (2.0, "qwen3-1.7b"), (4.99, "qwen3-1.7b"),
    (5.0, "qwen3-4b"), (10.0, "qwen3-4b"), (64.0, "qwen3-4b"),
])
def test_pick_model_thresholds_and_the_8b_is_never_automatic(budget, alias):
    assert M.pick_model(budget_gb=budget).alias == alias


@pytest.mark.parametrize("tier,alias", [
    ("small", "qwen3-1.7b"), ("default", "qwen3-4b"), ("large", "qwen3-8b"),
    ("qwen3-8b", "qwen3-8b"), ("phi3.5", "phi3.5"),
])
def test_pick_model_accepts_tiers_and_registry_aliases_whatever_the_budget(tier, alias):
    assert M.pick_model(tier, budget_gb=1.0).alias == alias


def test_pick_model_rejects_a_name_outside_the_registry():
    with pytest.raises(ValueError, match="gpt-5"):
        M.pick_model("gpt-5")


def test_pick_model_reads_the_device_budget_when_none_is_given(monkeypatch):
    monkeypatch.setattr(M, "device_budget_gb", lambda: 4.0)
    assert M.pick_model().alias == "qwen3-1.7b"
    monkeypatch.setattr(M, "device_budget_gb", lambda: 11.4)
    assert M.pick_model().alias == "qwen3-4b"


def test_pick_model_given_a_budget_or_a_tier_does_not_probe_the_device(monkeypatch):
    monkeypatch.setattr(M, "device_budget_gb", lambda: pytest.fail("probed the device"))
    assert M.pick_model(budget_gb=8.0).alias == "qwen3-4b"
    assert M.pick_model("large").alias == "qwen3-8b"


def test_override_picks_the_named_model_whatever_the_budget(monkeypatch):
    monkeypatch.setenv(_ENV, "qwen3-8b")
    assert M.pick_model(budget_gb=4.0).alias == "qwen3-8b"
    monkeypatch.setenv(_ENV, "small")
    assert M.pick_model(budget_gb=64.0).alias == "qwen3-1.7b"


def test_an_override_outside_the_registry_leaves_the_pick_to_the_budget(monkeypatch):
    monkeypatch.setenv(_ENV, "qwen3:4b")             # an Ollama tag: there is no ModelSpec for it
    assert M.pick_model(budget_gb=4.0).alias == "qwen3-1.7b"
    assert M.pick_model(budget_gb=8.0).alias == "qwen3-4b"


def test_the_default_tier_follows_the_override_and_the_others_do_not(monkeypatch):
    monkeypatch.setenv(_ENV, "qwen3-8b")
    assert M.pick_model("default").alias == "qwen3-8b"
    assert M.pick_model("small").alias == "qwen3-1.7b"
    monkeypatch.setenv(_ENV, "qwen3:4b")
    with pytest.raises(ValueError, match="qwen3:4b"):
        M.pick_model("default")


# ---------------------------------------------------------------- check_fits / ModelTooLarge

def test_check_fits_threshold_is_the_specs_own_minimum():
    for spec in M.MODELS.values():
        M.check_fits(spec, spec.min_budget_gb)                       # at the threshold: fits
        M.check_fits(spec, spec.min_budget_gb + 100)
        with pytest.raises(M.ModelTooLarge):
            M.check_fits(spec, spec.min_budget_gb - 0.01)


@pytest.mark.parametrize("alias,budget,suggested", [
    ("qwen3-8b", 9.99, "qwen3-4b"), ("qwen3-8b", 5.0, "qwen3-4b"), ("qwen3-8b", 4.99, "qwen3-1.7b"),
    ("qwen3-8b", 3.0, "qwen3-1.7b"), ("qwen3-8b", 2.99, None),
    ("qwen3-4b", 4.99, "qwen3-1.7b"), ("qwen3-4b", 3.0, "qwen3-1.7b"), ("qwen3-4b", 2.0, None),
    ("qwen3-1.7b", 2.99, None),
])
def test_check_fits_suggests_the_largest_qwen3_that_fits(alias, budget, suggested):
    with pytest.raises(M.ModelTooLarge) as ei:
        M.check_fits(M.MODELS[alias], budget)
    assert ei.value.suggested == suggested and ei.value.alias == alias and ei.value.budget_gb == budget


@pytest.mark.parametrize("alias", ["qwen2.5-3b-instruct", "gemma2-2b", "llama3.2-3b", "phi3.5"])
def test_a_legacy_model_that_does_not_fit_has_nothing_smaller_to_suggest(alias):
    with pytest.raises(M.ModelTooLarge) as ei:
        M.check_fits(M.MODELS[alias], 2.9)
    assert ei.value.suggested is None


def test_model_too_large_message_names_the_model_the_need_the_budget_and_the_way_out():
    with pytest.raises(M.ModelTooLarge) as ei:
        M.check_fits(M.MODELS["qwen3-8b"], 8.0)
    text = str(ei.value)
    assert "qwen3-8b" in text and "10.0" in text and "8.00" in text and "qwen3-4b" in text
    with pytest.raises(M.ModelTooLarge) as ei:
        M.check_fits(M.MODELS["qwen3-1.7b"], 2.0)
    text = str(ei.value)
    assert "qwen3-1.7b" in text and "3.0" in text and "2.00" in text and "qwen3-4b" not in text


def test_model_too_large_is_a_runtime_error_that_survives_pickling():
    exc = M.ModelTooLarge("qwen3-8b", "qwen3-4b", 8.0)
    assert isinstance(exc, RuntimeError)
    twin = pickle.loads(pickle.dumps(exc))
    assert (twin.alias, twin.suggested, twin.budget_gb) == ("qwen3-8b", "qwen3-4b", 8.0)
    assert str(twin) == str(exc)


def test_model_too_large_for_a_name_outside_the_registry_still_renders():
    text = str(M.ModelTooLarge("my-model", None, 1.0))
    assert "my-model" in text and "1.00" in text


# ---------------------------------------------------------------- import rule

_BLOCKED = ("mlx", "mlx_lm", "llama_cpp", "huggingface_hub", "transformers")

_BLOCKER = textwrap.dedent("""
    import json, sys
    _BLOCKED = set(json.loads(sys.argv[1]))
    class _Blocker:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in _BLOCKED:
                raise ModuleNotFoundError("blocked (simulated Pyodide): " + name)
            return None
    sys.meta_path.insert(0, _Blocker())
""")

_IMPORT_PROBE = _BLOCKER + textwrap.dedent("""
    import puremacro                      # whatever the package itself pulls in is not ours
    before = set(sys.modules)
    import puremacro.llm.models as m
    new = set(sys.modules) - before
    ours_or_stdlib = lambda n: n.startswith("puremacro.llm") or n.split(".")[0] in sys.stdlib_module_names
    print(json.dumps({
        "foreign": sorted(n for n in new if not ours_or_stdlib(n)),
        "blocked": sorted(n for n in sys.modules if n.split(".")[0] in _BLOCKED),
        "alias": m.pick_model(budget_gb=8.0).alias,
        "id": m.resolve_model_id("default", "ollama"),
        "budget": m.device_budget_gb(),
    }))
""")

_CONTROL_PROBE = _BLOCKER + textwrap.dedent("""
    import importlib, importlib.util
    refused = []
    for name in sorted(_BLOCKED):
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if "simulated" in str(exc):
                refused.append(name)
    try:
        importlib.util.find_spec("mlx")           # the very probe device_budget_gb makes
    except ModuleNotFoundError as exc:
        if "simulated" in str(exc):
            refused.append("find_spec")
    print(json.dumps(refused))
""")


def _probe(code):
    root = Path(puremacro.__file__).resolve().parent.parent          # the tree under test
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        p for p in (str(root), os.environ.get("PYTHONPATH", "")) if p))
    proc = subprocess.run([sys.executable, "-c", code, json.dumps(list(_BLOCKED))],
                          capture_output=True, text=True, encoding="utf-8", errors="replace",
                          env=env, cwd=root, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_the_module_imports_and_runs_with_every_inference_dependency_absent():
    out = _probe(_IMPORT_PROBE)
    assert out["blocked"] == []
    assert out["foreign"] == []            # nothing but puremacro.llm and the standard library
    assert out["alias"] == "qwen3-4b" and out["id"] == "qwen3:4b"
    assert isinstance(out["budget"], float) and out["budget"] > 0   # find_spec("mlx") raising is survived


@pytest.mark.mechanism_control
def test_the_import_blocker_actually_blocks():
    """Positive control: an inert blocker would make the test above pass on any host."""
    assert _probe(_CONTROL_PROBE) == [*sorted(_BLOCKED), "find_spec"]
