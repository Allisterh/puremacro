# tests/test_local_engines.py
"""``puremacro.narrative._local_engines``: the names of 4.8.1, now re-exports of ``puremacro.llm``.

The module keeps the import path that code written for 4.8.1 uses (``score_llm``,
``llm_prob_kernel``, the course tutor). Each class and function it exports is the object of the
same name in ``puremacro.llm``, not a copy (``test_a_reexport_is_the_object_of_puremacro_llm``),
so an ``except BackendUnavailable`` written against the old module catches what the engines
raise. The engines are tested in tests/test_llm_engines.py; here are the aliases, the engine
resolution as the old call sites reach it, ``chat``, and the engines driven the way 4.8.1 drove
them.

The stub servers are the ``llm_stub_server`` and ``llm_stub_404_server`` fixtures of
tests/conftest.py. No test loads a model, downloads one or talks to a server outside 127.0.0.1:
every ``available`` that would probe a default port is patched.
"""
import inspect
import sys
import types

import pytest

import puremacro.llm as L
import puremacro.llm.engines as E
from puremacro.narrative import _local_engines as le

_ENV_URL = "PUREMACRO_LLM_BASE_URL"     # removed before every test by tests/conftest.py


# ---------------------------------------------------------------- what the module exports

_EXPORTS = ["BackendUnavailable", "HTTPEngine", "LlamaCppEngine", "LocalLLMUnavailable",
            "MLXEngine", "MODEL_ALIASES", "chat", "resolve_engine", "resolve_model_id"]


def test_the_module_exports_the_names_of_4_8_1_that_callers_use():
    assert sorted(le.__all__) == sorted(_EXPORTS)
    assert all(hasattr(le, name) for name in _EXPORTS)


@pytest.mark.parametrize("name", ["BackendUnavailable", "LocalLLMUnavailable", "MLXEngine",
                                  "LlamaCppEngine", "HTTPEngine", "resolve_model_id"])
def test_a_reexport_is_the_object_of_puremacro_llm(name):
    assert getattr(le, name) is getattr(L, name)


def test_resolve_engine_is_the_function_of_puremacro_llm_engines():
    assert le.resolve_engine is E.resolve_engine


def test_the_narrative_scorers_import_the_very_same_error_classes():
    # score_llm and the index kernel import these names from the shim; an except clause that
    # names BackendUnavailable there must catch the LocalLLMUnavailable that puremacro.llm raises.
    from puremacro.narrative.scoring import llm as scoring_llm
    assert scoring_llm.BackendUnavailable is L.BackendUnavailable
    assert issubclass(L.LocalLLMUnavailable, scoring_llm.BackendUnavailable)


def test_score_llm_lets_the_error_of_the_facade_propagate():
    from puremacro.narrative.scoring import score_llm

    class _Down:
        model = "x"
        max_tokens = 8
        temperature = 0.0

        def call(self, prompt):
            raise L.LocalLLMUnavailable("no engine")      # the class puremacro.llm raises

    with pytest.raises(L.BackendUnavailable):
        score_llm([("2020-01-01", "text", "http://x")], backend=_Down(), kind="fiscal")


def test_backend_unavailable_is_base_of_local_unavailable():
    assert issubclass(le.LocalLLMUnavailable, le.BackendUnavailable)


# ---------------------------------------------------------------- aliases

def test_resolve_model_id_known_alias_per_engine():
    assert le.resolve_model_id("qwen2.5-3b-instruct", "ollama") == "qwen2.5:3b"
    assert le.resolve_model_id("qwen2.5-3b-instruct", "mlx") == \
        "mlx-community/Qwen2.5-3B-Instruct-4bit"
    repo, fname = le.resolve_model_id("qwen2.5-3b-instruct", "llamacpp")
    assert repo == "Qwen/Qwen2.5-3B-Instruct-GGUF" and fname.endswith(".gguf")


def test_resolve_model_id_unknown_passes_through():
    assert le.resolve_model_id("my-custom-model", "ollama") == "my-custom-model"
    assert le.resolve_model_id("my-custom-model", "mlx") == "my-custom-model"


def test_model_aliases_cover_qwen3():
    assert le.MODEL_ALIASES["qwen3-4b"]["ollama"] == "qwen3:4b"


def test_model_aliases_are_the_registry_by_engine():
    assert set(le.MODEL_ALIASES) == set(L.MODELS)
    for alias, spec in L.MODELS.items():
        assert le.MODEL_ALIASES[alias] == {"mlx": spec.mlx, "llamacpp": spec.llamacpp,
                                           "ollama": spec.ollama, "openai": spec.openai}


# ---------------------------------------------------------------- the engines, through the shim

def test_http_engine_ollama_available_and_complete(llm_stub_server):
    eng = le.HTTPEngine(base_url=llm_stub_server.url, api="ollama", timeout=5)
    assert eng.name == "ollama"
    assert eng.available() is True
    llm_stub_server.chat_content = '[{"x": 1}]'
    out = eng.complete("qwen2.5:3b", "hi", max_tokens=16,
                       temperature=0.0, json_mode=True)
    assert out == '[{"x": 1}]'


def test_http_engine_openai_complete(llm_stub_server):
    eng = le.HTTPEngine(base_url=llm_stub_server.url, api="openai", timeout=5)
    assert eng.name == "openai"
    llm_stub_server.score_content = "0.42"
    out = eng.complete("qwen2.5-3b", "hi", max_tokens=8,
                       temperature=0.0, json_mode=False)
    assert out == "0.42"


def test_http_engine_unavailable_when_no_server():
    # Port 1 is never an Ollama server -> available() False, complete() raises.
    eng = le.HTTPEngine(base_url="http://127.0.0.1:1", api="ollama", timeout=1)
    assert eng.available() is False
    with pytest.raises(le.LocalLLMUnavailable):
        eng.complete("qwen2.5:3b", "hi", max_tokens=8,
                     temperature=0.0, json_mode=False)


def test_mlx_engine_complete_with_fake_module(monkeypatch):
    captured = {}

    class _Tok:
        def apply_chat_template(self, messages, add_generation_prompt=True):
            captured["messages"] = messages
            return "PROMPT:" + messages[-1]["content"]

    fake = types.ModuleType("mlx_lm")
    fake.load = lambda model_id: ("MODEL", _Tok())
    fake.generate = lambda model, tok, prompt, max_tokens=256, verbose=False: \
        f"gen({prompt}|{max_tokens})"
    monkeypatch.setitem(sys.modules, "mlx_lm", fake)
    monkeypatch.setattr(le.MLXEngine, "available", lambda self: True)

    eng = le.MLXEngine()
    assert eng.name == "mlx"
    out = eng.complete("mlx-community/Qwen2.5-3B-Instruct-4bit", "hello",
                       max_tokens=32, temperature=0.0, json_mode=True)
    assert out == "gen(PROMPT:hello|32)"
    assert captured["messages"][-1]["content"] == "hello"


def test_llamacpp_engine_complete_with_fake_module(monkeypatch):
    # puremacro.llm loads the (repository, file) pair of the registry with
    # huggingface_hub.hf_hub_download, not with Llama.from_pretrained: the stand-in hub says what
    # was asked for and where the cached copy is, and Llama is built from that path.
    downloads, built = [], []

    class _Llama:
        def __init__(self, **kw):
            built.append(kw)

        def create_chat_completion(self, messages, temperature=0.0,
                                   max_tokens=256, **kw):
            txt = messages[-1]["content"]
            return {"choices": [{"message": {"content": f"cc({txt})"}}]}

    fake = types.ModuleType("llama_cpp")
    fake.Llama = _Llama
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda **kw: downloads.append(kw) or "/hf-cache/model.gguf"
    monkeypatch.setitem(sys.modules, "llama_cpp", fake)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setattr(le.LlamaCppEngine, "available", lambda self: True)

    eng = le.LlamaCppEngine()
    assert eng.name == "llamacpp"
    pair = le.resolve_model_id("qwen2.5-3b-instruct", "llamacpp")
    out = eng.complete(pair, "hi", max_tokens=16, temperature=0.0, json_mode=True)  # a 4.8.1-style call
    assert out == "cc(hi)"
    assert downloads == [{"repo_id": pair[0], "filename": pair[1]}]
    assert built[0]["model_path"] == "/hf-cache/model.gguf"


# ---------------------------------------------------------------- engine resolution

def _force_available(monkeypatch, available_names):
    """Make each engine's available() report membership in available_names."""
    monkeypatch.setattr(le.MLXEngine, "available",
                        lambda self: "mlx" in available_names)
    monkeypatch.setattr(le.LlamaCppEngine, "available",
                        lambda self: "llamacpp" in available_names)
    monkeypatch.setattr(le.HTTPEngine, "available",
                        lambda self: "ollama" in available_names)


def test_resolve_auto_prefers_mlx_when_importable(monkeypatch):
    _force_available(monkeypatch, {"mlx", "llamacpp", "ollama"})
    assert le.resolve_engine().name == "mlx"


def test_resolve_auto_falls_through_to_http(monkeypatch, llm_stub_server):
    monkeypatch.setattr(le.MLXEngine, "available", lambda self: False)
    monkeypatch.setattr(le.LlamaCppEngine, "available", lambda self: False)
    eng = le.resolve_engine("auto", base_url=llm_stub_server.url)    # the stub answers /api/tags
    assert eng.name == "ollama" and eng.base_url == llm_stub_server.url


def test_resolve_auto_none_raises(monkeypatch):
    _force_available(monkeypatch, set())
    with pytest.raises(le.LocalLLMUnavailable):
        le.resolve_engine("auto")


def test_an_open_port_that_is_not_an_llm_server_is_no_engine(monkeypatch, llm_stub_404_server):
    monkeypatch.setattr(le.MLXEngine, "available", lambda self: False)
    monkeypatch.setattr(le.LlamaCppEngine, "available", lambda self: False)
    with pytest.raises(le.LocalLLMUnavailable):
        le.resolve_engine("auto", base_url=llm_stub_404_server)      # every request answered with a 404


def test_resolve_explicit_unavailable_raises(monkeypatch):
    _force_available(monkeypatch, set())
    with pytest.raises(le.LocalLLMUnavailable):
        le.resolve_engine("ollama")


# ---------------------------------------------------------------- chat

class _FakeEngine:
    """An engine of puremacro.llm.engines that records what ``complete`` was given."""

    name = "fake"

    def __init__(self, reply="0.7"):
        self.reply, self.calls = reply, []

    def available(self):
        return True

    def complete(self, model_id, prompt, *, system=None, max_tokens, temperature, json_mode,
                 thinking=False):
        self.calls.append(dict(model_id=model_id, prompt=prompt, system=system,
                               max_tokens=max_tokens, temperature=temperature,
                               json_mode=json_mode, thinking=thinking))
        return self.reply


def test_chat_delegates_to_facade(monkeypatch):
    fake = _FakeEngine("hola, mundo")
    monkeypatch.setattr(E, "resolve_engine", lambda *a, **k: fake)
    monkeypatch.setattr(L.models, "device_budget_gb", lambda: 11.4)      # not this machine's memory
    assert le.chat("default", "hola", max_tokens=8) == "hola, mundo"
    # the facade ran it: "default" is the reference model, which thinks (8 tokens more at the engine)
    assert fake.calls == [dict(model_id="qwen3-4b", prompt="hola", system=None, max_tokens=16,
                               temperature=0.0, json_mode=False, thinking=True)]
    assert L.complete("hola", max_tokens=8, mode="cache") == "hola, mundo"   # and stored the answer


def test_chat_hands_its_arguments_to_the_facade(monkeypatch):
    seen = []
    fake = _FakeEngine("ok")
    monkeypatch.setattr(E, "resolve_engine", lambda *a, **k: seen.append((a, k)) or fake)
    assert le.chat("qwen3:4b", "hola", engine="ollama", base_url="http://h:1", max_tokens=5,
                   temperature=0.3, json_mode=True, timeout=7.0) == "ok"
    assert seen == [(("ollama",), {"base_url": "http://h:1", "timeout": 7.0})]
    assert fake.calls == [dict(model_id="qwen3:4b", prompt="hola", system=None, max_tokens=5,
                               temperature=0.3, json_mode=True, thinking=False)]


def test_chat_leaves_the_server_to_the_environment_by_default(monkeypatch):
    seen = []
    monkeypatch.setattr(E, "resolve_engine", lambda *a, **k: seen.append((a, k)) or _FakeEngine())
    le.chat("qwen3:4b", "hola")
    assert seen == [(("auto",), {"base_url": None, "timeout": 120.0})]    # not the 11434 of 4.8.1


def test_chat_reaches_the_server_the_environment_names(monkeypatch, llm_stub_server):
    monkeypatch.setenv(_ENV_URL, llm_stub_server.url)
    llm_stub_server.chat_content = "bonjour"
    assert le.chat("qwen3:4b", "salut", engine="ollama", max_tokens=8) == "bonjour"
    assert llm_stub_server.last_body["model"] == "qwen3:4b"
    assert llm_stub_server.last_body["options"]["num_predict"] == 8


def test_chat_keeps_the_signature_of_4_8_1_but_for_the_base_url_default():
    params = inspect.signature(le.chat).parameters
    assert list(params) == ["model", "prompt", "engine", "base_url", "max_tokens", "temperature",
                            "json_mode", "timeout"]
    assert [params[name].kind for name in list(params)[2:]] == \
        [inspect.Parameter.KEYWORD_ONLY] * 6
    assert {name: p.default for name, p in params.items() if p.default is not p.empty} == {
        "engine": "auto", "base_url": None, "max_tokens": 1024, "temperature": 0.0,
        "json_mode": False, "timeout": 120.0}
