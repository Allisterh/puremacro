# tests/test_llm_engines.py
"""Engines of ``puremacro.llm`` (spec section 3.2).

The first block is the contract of that section, with the decisions taken while it was built:
the stub servers are the shared ``llm_stub_server`` and ``llm_stub_404_server`` fixtures of
tests/conftest.py, and llama.cpp switches thinking off through the model's chat handler
(``test_llamacpp_wraps_chat_handler_when_thinking``) rather than with a ``/no_think`` suffix.
The blocks below pin the rest of those decisions (what each engine sends, the server order and
URLs, the probe timeout, the error paths) and the rule that keeps the module importable where
mlx, llama.cpp and sockets do not exist.

No test loads a model, downloads one or talks to a real server. The engines get stand-in
modules (``mlx_lm``, ``llama_cpp``, ``huggingface_hub``), and every probe is answered by a stub
on 127.0.0.1 or by a patched ``available``: an Ollama running on the developer's machine must
not decide a test.
"""
import http.client
import importlib.util
import io
import json
import os
import socket
import subprocess
import sys
import textwrap
import time
import types
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from _timing import assert_within

import puremacro
from puremacro.llm import engines as E

_ENV_URL = "PUREMACRO_LLM_BASE_URL"     # removed before every test by tests/conftest.py
_CALL = dict(max_tokens=8, temperature=0.0, json_mode=False)


# ---------------------------------------------------------------- the contract

def test_mlx_available_by_import_not_platform(monkeypatch):
    import importlib.util, platform, sys
    monkeypatch.setattr(sys, "platform", "ios")
    monkeypatch.setattr(platform, "system", lambda: "iPadOS")
    monkeypatch.setattr(platform, "machine", lambda: "iPad16,5")
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a, **k: object() if n in ("mlx", "mlx_lm") else None)
    assert E.MLXEngine().available() is True
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a, **k: None)
    assert E.MLXEngine().available() is False


def _avail(monkeypatch, mlx, llamacpp, http):
    monkeypatch.setattr(E.MLXEngine, "available", lambda self: mlx)
    monkeypatch.setattr(E.LlamaCppEngine, "available", lambda self: llamacpp)
    monkeypatch.setattr(E.HTTPEngine, "available", lambda self: http)


def test_resolve_auto_order(monkeypatch):
    _avail(monkeypatch, True, True, True);  assert E.resolve_engine().name == "mlx"
    _avail(monkeypatch, False, True, True); assert E.resolve_engine().name == "llamacpp"
    _avail(monkeypatch, False, False, True); assert E.resolve_engine().name == "ollama"


def test_resolve_auto_none_raises(monkeypatch):
    _avail(monkeypatch, False, False, False)
    with pytest.raises(E.LocalLLMUnavailable) as ei: E.resolve_engine()
    assert "qwen3:4b" in str(ei.value)


def test_probe_open_port_that_is_not_an_llm_server(llm_stub_404_server):   # every request answered with a 404
    t0 = time.perf_counter()
    assert E.HTTPEngine(api="openai", base_url=llm_stub_404_server).available() is False
    assert_within(time.perf_counter() - t0, 3.5, "probe of a non-LLM server (hang guard)")


def test_strip_thinking():
    assert E.strip_thinking("<think>a\nb</think>0.7") == "0.7"
    assert E.strip_thinking("<think>unterminated") == ""
    assert E.strip_thinking("plain") == "plain"


def test_mlx_complete_passes_enable_thinking_false_and_system(monkeypatch):
    seen = {}
    class _Tok:
        def apply_chat_template(self, messages, **kw): seen.update(messages=messages, kw=kw); return "P"
    fake = types.ModuleType("mlx_lm"); fake.load = lambda mid, **kw: ("M", _Tok()); fake.generate = lambda m, t, prompt, max_tokens, verbose: "<think>x</think>0.5"
    monkeypatch.setitem(sys.modules, "mlx_lm", fake); monkeypatch.setattr(E.MLXEngine, "available", lambda self: True)
    out = E.MLXEngine().complete("mid", "hola", system="S", max_tokens=8, temperature=0.0, json_mode=False, thinking=True)
    assert out == "0.5" and seen["kw"]["enable_thinking"] is False
    assert seen["messages"][0] == {"role": "system", "content": "S"} and seen["messages"][1]["content"] == "hola"


def _install_fake_llama_cpp(monkeypatch, route):
    """Install stand-ins for ``llama_cpp`` and ``huggingface_hub``; return what reaches them.

    The ``Llama`` stand-in finds a chat handler the way llama-cpp-python 0.3.36 does
    (``chat_handler``, then ``_chat_handlers[chat_format]``, then
    ``llama_chat_format.get_chat_completion_handler``) and, like the real one, refuses keywords
    ``create_chat_completion`` does not take; ``seen["built"]`` holds the keywords of each
    ``Llama``. It has no ``from_pretrained``, which lists the repository on the Hub at every
    load: the file of a ``(repository, file)`` pair has to come from
    ``huggingface_hub.hf_hub_download``, whose stand-in takes keywords only, records them in
    ``seen["downloads"]`` and returns ``/hf-cache/<repository>/<file>``.
    ``route`` says where the handler is: ``"attribute"``, ``"format"`` (what a GGUF file with a
    chat template gives, Qwen3's case), ``"registry"``, or nowhere: ``"none"`` (all three are
    None or raise) and ``"bare"`` (a build without that machinery: no such attributes and no
    ``llama_chat_format`` module).
    """
    seen = {"built": [], "requests": [], "handler": [], "downloads": []}

    def handler(*, llama, **request):
        seen["handler"].append(request)
        return {"choices": [{"message": {"role": "assistant", "content": "<think>\n\n</think>\n\n0.3"}}]}

    def registry_lookup(name):
        if route == "registry" and name == "chatml":
            return handler
        raise ValueError(f"Invalid chat handler: {name}")

    class Llama:
        def __init__(self, **options):
            seen["built"].append(options)
            if route == "bare":
                return
            self.chat_handler = handler if route == "attribute" else None
            self.chat_format = "chatml" if route == "registry" else "chat_template.default"
            self._chat_handlers = {"chat_template.default": handler} if route == "format" else {}

        def create_chat_completion(self, *, messages, temperature=0.2, max_tokens=None,
                                   response_format=None):
            request = {"messages": messages, "temperature": temperature,
                       "max_tokens": max_tokens, "response_format": response_format}
            seen["requests"].append(request)
            if route in ("none", "bare"):
                return {"choices": [{"message": {"role": "assistant", "content": "0.3"}}]}
            chosen = (self.chat_handler or self._chat_handlers.get(self.chat_format)
                      or registry_lookup(self.chat_format))
            return chosen(llama=self, **request)

    def hf_hub_download(**request):
        seen["downloads"].append(request)
        return f"/hf-cache/{request['repo_id']}/{request['filename']}"

    fake = types.ModuleType("llama_cpp")
    fake.Llama = Llama
    if route != "bare":
        fake.llama_chat_format = types.SimpleNamespace(get_chat_completion_handler=registry_lookup)
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = hf_hub_download
    monkeypatch.setitem(sys.modules, "llama_cpp", fake)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    return seen


def test_llamacpp_wraps_chat_handler_when_thinking(monkeypatch, tmp_path):
    seen = _install_fake_llama_cpp(monkeypatch, "format")
    eng = E.LlamaCppEngine()
    pair = ("Qwen/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf")
    # (a) thinking: the chat handler gets enable_thinking=False; the user turn stays as written
    assert eng.complete(pair, "hola", system=None, thinking=True, **_CALL) == "0.3"
    assert seen["handler"][-1]["enable_thinking"] is False
    assert seen["handler"][-1]["messages"] == [{"role": "user", "content": "hola"}]
    # (b) no thinking: nothing injected, so the wrapper of the call before did not stay behind
    eng.complete(pair, "hola", system=None, thinking=False, **_CALL)
    assert "enable_thinking" not in seen["handler"][-1]
    assert seen["handler"][-1]["messages"] == [{"role": "user", "content": "hola"}]
    # (d) every layer offloaded to the GPU, for a Hugging Face pair and for a file on disk; the
    #     pair's file is the path hf_hub_download gives (the cached copy, found offline)
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")
    eng.complete(str(gguf), "hola", system=None, thinking=True, **_CALL)
    assert seen["handler"][-1]["enable_thinking"] is False
    assert seen["downloads"] == [{"repo_id": pair[0], "filename": pair[1]}]
    assert seen["built"] == [
        {"model_path": f"/hf-cache/{pair[0]}/{pair[1]}", "n_ctx": 4096, "n_gpu_layers": -1,
         "verbose": False},
        {"model_path": str(gguf), "n_ctx": 4096, "n_gpu_layers": -1, "verbose": False},
    ]
    # (c) where no chat handler can be found, the /no_think suffix is the switch instead
    for route in ("none", "bare"):
        seen = _install_fake_llama_cpp(monkeypatch, route)
        bare = E.LlamaCppEngine()
        assert bare.complete(pair, "hola", system="S", thinking=True, **_CALL) == "0.3"
        assert seen["requests"][-1]["messages"] == [
            {"role": "system", "content": "S"}, {"role": "user", "content": "hola\n/no_think"}], route
        bare.complete(pair, "hola", system="S", thinking=False, **_CALL)
        assert seen["requests"][-1]["messages"][-1] == {"role": "user", "content": "hola"}, route


def test_ollama_payload_think_false(llm_stub_server):                  # the stub records the last JSON body
    eng = E.HTTPEngine(api="ollama", base_url=llm_stub_server.url)
    eng.complete("qwen3:4b", "hola", system="S", max_tokens=8, temperature=0.0, json_mode=False, thinking=True)
    body = llm_stub_server.last_body
    assert body["think"] is False and body["messages"][0] == {"role": "system", "content": "S"}
    assert body["messages"][1] == {"role": "user", "content": "hola"}   # Ollama gets no suffix
    eng.complete("qwen3:4b", "hola", system="S", max_tokens=8, temperature=0.0, json_mode=False, thinking=False)
    assert "think" not in llm_stub_server.last_body


@pytest.mark.parametrize("thinking", [True, False])
def test_openai_payload_disables_thinking_two_ways(llm_stub_server, thinking):
    eng = E.HTTPEngine(api="openai", base_url=llm_stub_server.url)
    out = eng.complete("qwen3-4b", "hola", system="S", max_tokens=8, temperature=0.0, json_mode=False, thinking=thinking)
    body = llm_stub_server.last_body
    assert out == "0.7" and body["messages"][0] == {"role": "system", "content": "S"}
    if thinking:
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert body["messages"][1] == {"role": "user", "content": "hola\n/no_think"}
    else:
        assert "chat_template_kwargs" not in body
        assert body["messages"][1] == {"role": "user", "content": "hola"}


def test_base_url_env_override(monkeypatch, llm_stub_server):
    _avail(monkeypatch, False, False, True)
    monkeypatch.setenv("PUREMACRO_LLM_BASE_URL", llm_stub_server.url)
    eng = E.resolve_engine()
    assert eng.name == "ollama" and eng.base_url == llm_stub_server.url


# ---------------------------------------------------------------- names and constants

def test_install_hint_text():
    assert E.INSTALL_HINT == (
        "No local LLM engine available. Install one with "
        '`pip install "puremacro[local-llm]"` (MLX on Apple silicon, llama-cpp-python anywhere), '
        "or start Ollama (https://ollama.com) and run `ollama pull qwen3:4b`.")


def test_default_servers():
    assert E.DEFAULT_SERVERS == (("ollama", "http://localhost:11434"),
                                 ("openai", "http://localhost:8080"),
                                 ("openai", "http://localhost:1234"))


def test_the_three_engines_satisfy_the_engine_protocol():
    mlx, llamacpp, http = E.MLXEngine(), E.LlamaCppEngine(), E.HTTPEngine(base_url="http://h:1/", api="openai")
    assert all(isinstance(engine, E.Engine) for engine in (mlx, llamacpp, http))
    assert (mlx.name, llamacpp.name, http.name) == ("mlx", "llamacpp", "openai")
    assert mlx.adapter_path is None and llamacpp.n_ctx == 4096
    assert (http.base_url, http.api, http.timeout, http.probe_timeout) == ("http://h:1", "openai", 120.0, 3.0)
    assert issubclass(E.LocalLLMUnavailable, E.BackendUnavailable)
    assert issubclass(E.BackendUnavailable, RuntimeError)


def test_the_protocol_defaults_system_and_thinking_like_the_engines():
    """A 4.8.1-style call, without ``system`` and ``thinking``, type-checks through ``Engine``
    as it runs on each engine (the callers behind the narrative shim make such calls)."""
    import inspect

    def shape(cls):
        return [(p.name, p.kind, p.default) for p in inspect.signature(cls.complete).parameters.values()]

    protocol = shape(E.Engine)
    defaults = {name: default for name, _, default in protocol}
    assert defaults["system"] is None and defaults["thinking"] is False
    assert [shape(cls) for cls in (E.MLXEngine, E.LlamaCppEngine, E.HTTPEngine)] == [protocol] * 3


def test_http_engine_rejects_an_unknown_api():
    with pytest.raises(ValueError, match="'vllm'"):
        E.HTTPEngine(base_url="http://h:1", api="vllm")


def test_http_engine_defaults_to_the_local_ollama_server_as_in_4_8_1():
    """``HTTPEngine()`` is a compatibility name (re-exported by puremacro.narrative._local_engines)
    and 4.8.1 built it without arguments: Ollama on the first default server."""
    from puremacro.narrative import _local_engines as le

    eng = E.HTTPEngine()
    assert (eng.base_url, eng.api, eng.name) == ("http://localhost:11434", "ollama", "ollama")
    assert eng.base_url == E.DEFAULT_SERVERS[0][1] and (eng.timeout, eng.probe_timeout) == (120.0, 3.0)
    assert le.HTTPEngine is E.HTTPEngine and le.HTTPEngine().base_url == "http://localhost:11434"
    assert E.HTTPEngine(api="openai").name == "openai"                 # each default stands alone
    assert E.HTTPEngine(base_url="http://h:1/").api == "ollama"


@pytest.mark.parametrize("name", ["auto", "mlx", "llamacpp", "ollama", "openai"])
def test_validate_engine_name_returns_auto_and_the_four_engine_names(name):
    assert E.validate_engine_name(name) == name
    assert "validate_engine_name" in E.__all__


@pytest.mark.parametrize("name", ["olama", "MLX", " mlx", "gpt", "", None, 3])
def test_validate_engine_name_rejects_the_rest_with_the_resolvers_message_and_probes_nothing(monkeypatch, name):
    """What ``resolve_engine`` would raise for the name, raised without constructing an engine, so
    that callers (the narrative backends) can refuse a typo when they are built."""
    for cls in (E.MLXEngine, E.LlamaCppEngine, E.HTTPEngine):
        monkeypatch.setattr(cls, "available", lambda self: pytest.fail("probed"))
    with pytest.raises(ValueError) as checked:
        E.validate_engine_name(name)
    with pytest.raises(ValueError) as resolved:
        E.resolve_engine(name)
    assert str(checked.value) == str(resolved.value)
    assert repr(name) in str(checked.value) and "'auto'" in str(checked.value)
    assert "['mlx', 'llamacpp', 'ollama', 'openai']" in str(checked.value)


@pytest.mark.parametrize("raw,expected", [
    ("<think>\n\n</think>\n\n0.3", "0.3"),                 # what /no_think leaves
    ("<think>a</think>x <think>b</think>y", "x y"),
    ("0.4 <think>cut off by the token limit", "0.4"),
    ("Use the </think> tag.", "Use the </think> tag."),    # a closing tag alone is not thinking
    ("a</think>b<think>c", "a</think>b"),
    ("<think>only thinking</think>", ""),
    ("  0.5\n", "0.5"),
    ("", ""),
])
def test_strip_thinking_cases(raw, expected):
    assert E.strip_thinking(raw) == expected


# ---------------------------------------------------------------- availability by import

def test_llamacpp_available_by_import_and_mlx_needs_both_packages(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a, **k: object() if n == "llama_cpp" else None)
    assert E.LlamaCppEngine().available() is True and E.MLXEngine().available() is False
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a, **k: object() if n == "mlx_lm" else None)
    assert E.MLXEngine().available() is False                    # mlx-lm without mlx is not enough
    assert E.LlamaCppEngine().available() is False


def test_a_find_spec_that_raises_counts_as_not_installed(monkeypatch):
    def find_spec(name, *args, **kwargs):
        raise ValueError(f"{name}.__spec__ is None")             # a module stubbed in sys.modules
    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    assert E.MLXEngine().available() is False and E.LlamaCppEngine().available() is False


def test_availability_never_asks_the_platform(monkeypatch):
    import platform
    asked = []
    for name in ("system", "machine", "platform", "processor", "uname", "mac_ver"):
        monkeypatch.setattr(platform, name, lambda *a, _name=name, **k: asked.append(_name) or "")
    monkeypatch.setattr(E.HTTPEngine, "available", lambda self: False)
    E.MLXEngine().available()
    E.LlamaCppEngine().available()
    E.available_engines()
    try:
        E.resolve_engine()
    except E.LocalLLMUnavailable:
        pass                                    # no engine installed here: also an answer
    assert asked == []


# ---------------------------------------------------------------- MLX

def _fake_mlx_lm(monkeypatch, *, template_returns="P", reply="0.5", load_error=None):
    """Install a stand-in ``mlx_lm`` and return what reaches it."""
    seen = {"load": [], "template": [], "generate": []}

    class _Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            seen["template"].append({"messages": messages, **kwargs})
            return template_returns

    def load(model_id, **kwargs):
        seen["load"].append((model_id, kwargs))
        if load_error is not None:
            raise load_error
        return ("MODEL", _Tokenizer())

    def generate(model, tokenizer, prompt, max_tokens, verbose):
        seen["generate"].append({"model": model, "prompt": prompt, "max_tokens": max_tokens,
                                 "verbose": verbose})
        return reply

    fake = types.ModuleType("mlx_lm")
    fake.load, fake.generate = load, generate
    monkeypatch.setitem(sys.modules, "mlx_lm", fake)
    return seen


def test_mlx_applies_the_template_with_thinking_off_whatever_the_flag(monkeypatch):
    """``enable_thinking=False`` goes to the template on every call, not only when the facade knows
    the model thinks: a template without the variable ignores the keyword, and a Qwen3 named by its
    repository id (which the registry does not know) would otherwise think its token budget away."""
    seen = _fake_mlx_lm(monkeypatch, reply=" 0.6 ")
    assert E.MLXEngine().complete("mid", "hola", **_CALL) == "0.6"       # a 4.8.1-style call
    assert seen["template"] == [{"messages": [{"role": "user", "content": "hola"}],
                                 "add_generation_prompt": True, "enable_thinking": False}]
    assert seen["generate"] == [{"model": "MODEL", "prompt": "P", "max_tokens": 8, "verbose": False}]
    E.MLXEngine().complete("mid", "hola", thinking=False, **_CALL)
    assert seen["template"][-1]["enable_thinking"] is False


def test_mlx_switches_thinking_off_for_a_pass_through_model_id_too(monkeypatch):
    """Through the facade: a model outside the registry reaches MLX under its own id with
    ``thinking=False`` (the facade cannot know it thinks), the template still gets
    ``enable_thinking=False``, the think block that comes back is stripped, and no token allowance
    is added (the key keeps the caller's ``max_tokens``)."""
    import puremacro.llm as L

    seen = _fake_mlx_lm(monkeypatch, reply="<think>\n\n</think>\n\n0.5")
    monkeypatch.setattr(E.MLXEngine, "available", lambda self: True)
    model_id = "mlx-community/Qwen3-4B-8bit"
    assert L.complete("hola", model=model_id, engine="mlx", mode="off", max_tokens=8) == "0.5"
    assert seen["load"] == [(model_id, {})]
    assert seen["template"] == [{"messages": [{"role": "user", "content": "hola"}],
                                 "add_generation_prompt": True, "enable_thinking": False}]
    assert seen["generate"][0]["max_tokens"] == 8


@pytest.mark.parametrize("system", [None, ""])
def test_no_system_text_means_no_system_turn(monkeypatch, system):
    seen = _fake_mlx_lm(monkeypatch)
    E.MLXEngine().complete("mid", "hola", system=system, thinking=True, **_CALL)
    assert seen["template"][0]["messages"] == [{"role": "user", "content": "hola"}]


@pytest.mark.parametrize("template_returns,fed", [
    ([151644, 872], [151644, 872]),                  # token ids: mlx-lm 0.32 on transformers 5
    ("<|im_start|>user", "<|im_start|>user"),        # a rendered prompt
    ({"input_ids": [1, 2, 3]}, [1, 2, 3]),           # a BatchEncoding, transformers 5's default
], ids=["ids", "text", "mapping"])
def test_mlx_feeds_generate_what_the_template_gives(monkeypatch, template_returns, fed):
    seen = _fake_mlx_lm(monkeypatch, template_returns=template_returns)
    E.MLXEngine().complete("mid", "hola", system=None, thinking=True, **_CALL)
    assert seen["generate"][0]["prompt"] == fed


def test_mlx_loads_each_model_once_and_passes_the_adapter_only_when_set(monkeypatch):
    seen = _fake_mlx_lm(monkeypatch)
    plain = E.MLXEngine()
    for model_id in ("a", "a", "b"):
        plain.complete(model_id, "hola", system=None, thinking=False, **_CALL)
    E.MLXEngine(adapter_path="adapters/v1").complete("a", "hola", system=None, thinking=False, **_CALL)
    assert seen["load"] == [("a", {}), ("b", {}), ("a", {"adapter_path": "adapters/v1"})]


def test_mlx_missing_package_is_local_llm_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx_lm", None)            # `import mlx_lm` raises ImportError
    with pytest.raises(E.LocalLLMUnavailable, match="qwen3:4b") as ei:
        E.MLXEngine().complete("mid", "hola", system=None, thinking=True, **_CALL)
    assert isinstance(ei.value.__cause__, ImportError)


def test_mlx_load_failure_propagates_as_raised(monkeypatch):
    failure = OSError("Repository Not Found: org/no-such-model")
    _fake_mlx_lm(monkeypatch, load_error=failure)
    with pytest.raises(OSError) as ei:                  # mlx-lm is there: not a LocalLLMUnavailable
        E.MLXEngine().complete("org/no-such-model", "hola", system=None, thinking=True, **_CALL)
    assert ei.value is failure


# ---------------------------------------------------------------- llama.cpp

@pytest.mark.parametrize("route", ["attribute", "registry"])
def test_llamacpp_finds_the_chat_handler_where_llama_cpp_does(monkeypatch, route):
    seen = _install_fake_llama_cpp(monkeypatch, route)
    eng = E.LlamaCppEngine()
    eng.complete(("r", "f.gguf"), "hola", system=None, thinking=True, **_CALL)
    assert seen["handler"][-1]["enable_thinking"] is False
    assert seen["handler"][-1]["messages"] == [{"role": "user", "content": "hola"}]
    eng.complete(("r", "f.gguf"), "hola", system=None, thinking=False, **_CALL)
    assert "enable_thinking" not in seen["handler"][-1]


def test_llamacpp_request_carries_system_json_mode_and_sampling(monkeypatch):
    seen = _install_fake_llama_cpp(monkeypatch, "format")
    out = E.LlamaCppEngine(n_ctx=8192).complete(("r", "f.gguf"), "hola", system="S", max_tokens=16,
                                                temperature=0.2, json_mode=True)   # a 4.8.1-style call
    assert out == "0.3"
    assert seen["requests"] == [{
        "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "hola"}],
        "temperature": 0.2, "max_tokens": 16, "response_format": {"type": "json_object"}}]
    assert seen["built"][0]["n_ctx"] == 8192


def test_llamacpp_keeps_each_model_loaded_once(monkeypatch):
    seen = _install_fake_llama_cpp(monkeypatch, "format")
    eng = E.LlamaCppEngine()
    for model_id in (("r", "a.gguf"), ["r", "a.gguf"], ("r", "a.gguf"), ("r", "b.gguf")):
        eng.complete(model_id, "hola", system=None, thinking=False, **_CALL)
    assert seen["downloads"] == [{"repo_id": "r", "filename": "a.gguf"},
                                 {"repo_id": "r", "filename": "b.gguf"}]
    assert [options["model_path"] for options in seen["built"]] == ["/hf-cache/r/a.gguf",
                                                                     "/hf-cache/r/b.gguf"]


@pytest.mark.parametrize("model_id", ["qwen3:4b", "no/such/file.gguf", ("only-one",), ("r", "f", "x"), ("r", 4), 42])
def test_llamacpp_rejects_an_id_it_cannot_load(monkeypatch, model_id):
    seen = _install_fake_llama_cpp(monkeypatch, "format")
    with pytest.raises(ValueError, match="neither"):    # a bad argument, not a missing engine
        E.LlamaCppEngine().complete(model_id, "hola", system=None, thinking=False, **_CALL)
    assert seen["built"] == [] and seen["downloads"] == []


@pytest.mark.parametrize("pair_type", [tuple, list])
@pytest.mark.parametrize("file", ["*Q4_K_M.gguf", "Qwen3-4B-Q4_?_M.gguf", "Qwen3-*-Q4_K_M.gguf"])
def test_llamacpp_rejects_a_pattern_in_place_of_a_file_name(monkeypatch, pair_type, file):
    # 4.8.1 wrote the file of a pair as a glob ("*Q4_K_M.gguf"); the registry now holds the exact
    # name, because a pattern can match no file or several. hf_hub_download takes a name, not a
    # pattern, so the pair is refused before it is asked for anything.
    seen = _install_fake_llama_cpp(monkeypatch, "format")
    pair = pair_type(("Qwen/Qwen3-4B-GGUF", file))
    with pytest.raises(ValueError) as ei:       # a bad argument, not a missing engine
        E.LlamaCppEngine().complete(pair, "hola", system=None, thinking=False, **_CALL)
    assert str(ei.value) == f"llama.cpp registry files must be exact file names, not patterns: {file}"
    assert seen["downloads"] == [] and seen["built"] == []     # nothing was downloaded or loaded


@pytest.mark.parametrize("source", ["pair", "path"])
def test_llamacpp_load_failure_propagates_as_raised(monkeypatch, tmp_path, source):
    _install_fake_llama_cpp(monkeypatch, "format")
    gguf = tmp_path / "broken.gguf"
    gguf.write_bytes(b"not a model")
    model_id = ("r", "f.gguf") if source == "pair" else str(gguf)
    path = "/hf-cache/r/f.gguf" if source == "pair" else str(gguf)
    failure = ValueError(f"Failed to load model from file: {path}")       # llama.cpp's words

    def fail(*args, **options):
        raise failure

    monkeypatch.setattr(sys.modules["llama_cpp"].Llama, "__init__", fail)
    with pytest.raises(ValueError) as ei:   # llama-cpp-python is there: no LocalLLMUnavailable
        E.LlamaCppEngine().complete(model_id, "hola", system=None, thinking=False, **_CALL)
    assert ei.value is failure


def test_llamacpp_missing_package_is_local_llm_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    with pytest.raises(E.LocalLLMUnavailable, match="qwen3:4b") as ei:
        E.LlamaCppEngine().complete(("r", "f.gguf"), "hola", system=None, thinking=False, **_CALL)
    assert isinstance(ei.value.__cause__, ImportError)


def test_llamacpp_pair_without_huggingface_hub_is_local_llm_unavailable(monkeypatch, tmp_path):
    seen = _install_fake_llama_cpp(monkeypatch, "format")
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)    # `import huggingface_hub` raises
    with pytest.raises(E.LocalLLMUnavailable, match="pip install huggingface_hub") as ei:
        E.LlamaCppEngine().complete(("r", "f.gguf"), "hola", system=None, thinking=False, **_CALL)
    assert isinstance(ei.value.__cause__, ImportError) and seen["built"] == []
    gguf = tmp_path / "model.gguf"                               # a file on disk needs no hub
    gguf.write_bytes(b"GGUF")
    assert E.LlamaCppEngine().complete(str(gguf), "hola", system=None, thinking=False, **_CALL) == "0.3"


class _LocalEntryNotFoundError(FileNotFoundError):
    """What huggingface_hub raises for a file it has not cached when the Hub cannot be reached."""


@pytest.mark.parametrize("failure", [
    _LocalEntryNotFoundError("Cannot find the requested files in the disk cache and outgoing "
                             "traffic has been disabled."),
    OSError("404 Client Error. Repository Not Found for url: https://huggingface.co/api/models/x"),
    ConnectionError("Failed to resolve 'huggingface.co'"),
], ids=["offline-not-cached", "not-on-the-hub", "no-connection"])
def test_llamacpp_pair_neither_cached_nor_downloadable_is_local_llm_unavailable(monkeypatch, failure):
    seen = _install_fake_llama_cpp(monkeypatch, "format")

    def hf_hub_download(**request):
        seen["downloads"].append(request)
        raise failure

    monkeypatch.setattr(sys.modules["huggingface_hub"], "hf_hub_download", hf_hub_download)
    pair = ("Qwen/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf")
    with pytest.raises(E.LocalLLMUnavailable) as ei:
        E.LlamaCppEngine().complete(pair, "hola", system=None, thinking=True, **_CALL)
    assert pair[0] in str(ei.value) and pair[1] in str(ei.value)
    assert ei.value.__cause__ is failure and seen["built"] == []
    assert seen["downloads"] == [{"repo_id": pair[0], "filename": pair[1]}]


# ---------------------------------------------------------------- HTTP servers

def test_http_engines_against_the_stub(llm_stub_server):
    ollama = E.HTTPEngine(base_url=llm_stub_server.url + "/", api="ollama", timeout=5)
    openai = E.HTTPEngine(base_url=llm_stub_server.url, api="openai", timeout=5)
    assert ollama.base_url == llm_stub_server.url
    assert ollama.available() is True and openai.available() is True
    llm_stub_server.chat_content = '[{"x": 1}]'
    assert ollama.complete("qwen3:4b", "hi", max_tokens=16, temperature=0.0, json_mode=True) == '[{"x": 1}]'
    assert openai.complete("qwen3-4b", "hi", **_CALL) == "0.7"                 # 4.8.1-style calls


@pytest.mark.parametrize("given,root", [
    ("http://h:1", "http://h:1"),
    ("http://h:1/", "http://h:1"),
    ("http://h:1/v1", "http://h:1"),                  # how LM Studio and OpenAI clients write it
    ("http://h:1/v1/", "http://h:1"),
    ("http://h:1//v1//", "http://h:1"),
    ("http://h:1/v1/v1", "http://h:1/v1"),            # one /v1 only
    ("http://h:1/proxy/v1", "http://h:1/proxy"),
    ("http://h:1/v10", "http://h:1/v10"),
])
@pytest.mark.parametrize("api", ["ollama", "openai"])
def test_a_base_url_loses_its_trailing_slashes_and_one_v1(api, given, root):
    assert E.HTTPEngine(base_url=given, api=api).base_url == root


@pytest.mark.parametrize("api,probe,chat,reply", [
    ("openai", "/v1/models", "/v1/chat/completions", "0.7"),
    ("ollama", "/api/tags", "/api/chat", "[]"),
])
def test_a_v1_base_url_reaches_the_server_root(monkeypatch, llm_stub_server, api, probe, chat, reply):
    urls = []
    real_urlopen = urllib.request.urlopen

    def urlopen(request, *args, **kwargs):             # the real one, recording each URL
        urls.append(request.full_url)
        return real_urlopen(request, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    eng = E.HTTPEngine(base_url=llm_stub_server.url + "/v1", api=api)
    assert eng.base_url == llm_stub_server.url
    assert eng.available() is True       # the stub answers /v1/v1/models and /v1/api/tags with 404
    assert eng.complete("m", "hola", system=None, thinking=False, **_CALL) == reply
    assert urls == [llm_stub_server.url + probe, llm_stub_server.url + chat]


def test_ollama_payload_shape(llm_stub_server):
    E.HTTPEngine(base_url=llm_stub_server.url, api="ollama").complete(
        "qwen3:4b", "hola", system=None, max_tokens=8, temperature=0.1, json_mode=True, thinking=False)
    assert llm_stub_server.last_body == {
        "model": "qwen3:4b", "messages": [{"role": "user", "content": "hola"}], "stream": False,
        "options": {"temperature": 0.1, "num_predict": 8}, "format": "json"}


def test_openai_payload_shape(llm_stub_server):
    E.HTTPEngine(base_url=llm_stub_server.url, api="openai").complete(
        "qwen3-4b", "hola", system=None, max_tokens=8, temperature=0.1, json_mode=True, thinking=False)
    assert llm_stub_server.last_body == {
        "model": "qwen3-4b", "messages": [{"role": "user", "content": "hola"}],
        "temperature": 0.1, "max_tokens": 8, "response_format": {"type": "json_object"}}


@pytest.mark.parametrize("api", ["ollama", "openai"])
@pytest.mark.parametrize("content,expected", [("<think>\n\n</think>\n\n 0.4 ", "0.4"), (None, "")],
                         ids=["think-block", "null"])
def test_http_replies_are_stripped_and_a_null_content_is_empty(llm_stub_server, api, content, expected):
    llm_stub_server.chat_content = llm_stub_server.score_content = content
    eng = E.HTTPEngine(base_url=llm_stub_server.url, api=api)
    assert eng.complete("m", "hola", system=None, thinking=True, **_CALL) == expected


@pytest.mark.parametrize("api", ["ollama", "openai"])
def test_an_unreachable_server_is_local_llm_unavailable(api):
    eng = E.HTTPEngine(base_url="http://127.0.0.1:1", api=api, timeout=1)    # nothing listens on port 1
    assert eng.available() is False
    with pytest.raises(E.LocalLLMUnavailable) as ei:
        eng.complete("m", "hola", system=None, thinking=False, **_CALL)
    assert "http://127.0.0.1:1" in str(ei.value) and "qwen3:4b" in str(ei.value)
    assert isinstance(ei.value.__cause__, urllib.error.URLError)


@pytest.mark.parametrize("api", ["ollama", "openai"])
def test_a_connection_that_times_out_is_local_llm_unavailable(monkeypatch, api):
    tried = []

    def urlopen(request, *args, **kwargs):
        tried.append(request.full_url)
        raise urllib.error.URLError(TimeoutError("timed out"))   # urllib's words for a connect timeout

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    with pytest.raises(E.LocalLLMUnavailable) as ei:
        E.HTTPEngine(base_url="http://h:1", api=api, timeout=0.5).complete(
            "m", "hola", system=None, thinking=False, **_CALL)
    message = str(ei.value)
    assert "http://h:1" in message and "timed out" in message and E.INSTALL_HINT in message
    assert tried and all(url.startswith("http://h:1/") for url in tried)


@pytest.mark.parametrize("api", ["ollama", "openai"])
def test_a_reply_that_never_comes_propagates_the_timeout(llm_stub_server, api):
    llm_stub_server.delay = 30.0                        # the request is read, the reply never sent
    eng = E.HTTPEngine(base_url=llm_stub_server.url, api=api, timeout=0.3)
    with pytest.raises(TimeoutError) as ei:             # the server was reached: not a URLError
        eng.complete("m", "hola", system=None, thinking=False, **_CALL)
    assert not isinstance(ei.value, (urllib.error.URLError, E.LocalLLMUnavailable))


@pytest.mark.parametrize("status,body,said", [
    (500, json.dumps({"error": "model runner has unexpectedly stopped"}).encode(), "unexpectedly stopped"),
    (503, json.dumps({"error": {"message": "Loading model"}}).encode(), "Loading model"),   # llama-server
    (307, b"", ""),                      # a redirect urllib does not follow for a POST: not a 4xx
], ids=["500", "503", "307"])
@pytest.mark.parametrize("api", ["ollama", "openai"])
def test_a_server_error_is_local_llm_unavailable(llm_stub_server, api, status, body, said):
    llm_stub_server.chat_reply = (status, "application/json", body)
    with pytest.raises(E.LocalLLMUnavailable) as ei:
        E.HTTPEngine(base_url=llm_stub_server.url, api=api).complete(
            "m", "hola", system=None, thinking=False, **_CALL)
    message = str(ei.value)
    assert llm_stub_server.url in message and f"HTTP {status}" in message and said in message
    assert E.INSTALL_HINT in message
    assert isinstance(ei.value.__cause__, urllib.error.HTTPError)


def test_a_404_from_ollama_names_the_model_and_how_to_pull_it(llm_stub_server, llm_stub_404_server):
    missing = json.dumps({"error": 'model "qwen3:8b" not found, try pulling it first'}).encode()
    llm_stub_server.chat_reply = (404, "application/json", missing)            # how Ollama says it
    messages = []
    for url in (llm_stub_server.url, llm_stub_404_server):                     # and a bare 404 page
        with pytest.raises(E.LocalLLMUnavailable) as ei:
            E.HTTPEngine(base_url=url, api="ollama").complete(
                "qwen3:8b", "hola", system=None, thinking=True, **_CALL)
        assert isinstance(ei.value.__cause__, urllib.error.HTTPError)
        messages.append(str(ei.value))
        assert url in messages[-1] and "HTTP 404" in messages[-1] and "'qwen3:8b'" in messages[-1]
        assert "`ollama pull qwen3:8b`" in messages[-1]
        assert E.INSTALL_HINT not in messages[-1] and "local-llm" not in messages[-1]   # no generic hint
    assert "try pulling it first" in messages[0]                               # what Ollama said


def test_a_404_from_an_openai_server_names_the_model(llm_stub_server, llm_stub_404_server):
    missing = json.dumps({"error": {"message": "The model `qwen3-8b` does not exist",
                                    "type": "invalid_request_error", "code": "model_not_found"}}).encode()
    llm_stub_server.chat_reply = (404, "application/json", missing)            # the OpenAI shape
    messages = []
    for url in (llm_stub_server.url, llm_stub_404_server):
        with pytest.raises(E.LocalLLMUnavailable) as ei:
            E.HTTPEngine(base_url=url, api="openai").complete(
                "qwen3-8b", "hola", system=None, thinking=True, **_CALL)
        assert isinstance(ei.value.__cause__, urllib.error.HTTPError)
        messages.append(str(ei.value))
        assert url in messages[-1] and "HTTP 404" in messages[-1] and "'qwen3-8b'" in messages[-1]
        assert "ollama pull" not in messages[-1] and "local-llm" not in messages[-1]   # neither remedy
    assert "does not exist" in messages[0]


@pytest.mark.parametrize("status", [400, 401, 413, 422])
@pytest.mark.parametrize("api", ["ollama", "openai"])
def test_any_other_4xx_is_a_runtime_error_for_that_request_alone(llm_stub_server, api, status):
    body = json.dumps({"error": {"message": "the request (" + "x" * 300 + ") exceeds the context size"}})
    llm_stub_server.chat_reply = (status, "application/json", body.encode())
    with pytest.raises(RuntimeError) as ei:
        E.HTTPEngine(base_url=llm_stub_server.url, api=api).complete(
            "m", "hola", system=None, thinking=False, **_CALL)
    assert type(ei.value) is RuntimeError      # not a BackendUnavailable: score_llm counts the record
    assert str(ei.value) == f"{llm_stub_server.url} rejected the request (HTTP {status}): {body[:200]}"
    assert isinstance(ei.value.__cause__, urllib.error.HTTPError)


def test_the_servers_own_error_message_is_passed_on(llm_stub_server):
    missing = json.dumps({"error": 'model "qwen3:8b" not found, try pulling it first'}).encode()
    llm_stub_server.chat_reply = (404, "application/json", missing)            # how Ollama says it
    with pytest.raises(E.LocalLLMUnavailable, match="try pulling it first"):
        E.HTTPEngine(base_url=llm_stub_server.url, api="ollama").complete(
            "qwen3:8b", "hola", system=None, thinking=True, **_CALL)
    too_long = json.dumps({"error": {"message": "the request exceeds the context size"}}).encode()
    llm_stub_server.chat_reply = (400, "application/json", too_long)           # the OpenAI shape
    with pytest.raises(RuntimeError, match="exceeds the context size") as ei:  # this prompt, not the server
        E.HTTPEngine(base_url=llm_stub_server.url, api="openai").complete(
            "qwen3-4b", "hola", system=None, thinking=True, **_CALL)
    assert not isinstance(ei.value, E.BackendUnavailable)


def test_a_servers_top_level_message_is_passed_on_when_there_is_no_error_field(llm_stub_server):
    """vLLM answers ``{"object": "error", "message": ..., "code": 404}``: neither Ollama's ``error``
    string nor the OpenAI ``error.message`` object. The message is passed on all the same."""
    vllm = {"object": "error", "message": "The model `qwen3-8b` does not exist.",
            "type": "NotFoundError", "code": 404}
    assert E._server_message(json.dumps(vllm)) == "The model `qwen3-8b` does not exist."
    assert E._server_message(json.dumps({"error": "e", "message": "m"})) == "e"    # `error` first
    assert E._server_message(json.dumps({"error": {"message": "e"}, "message": "m"})) == "e"
    assert E._server_message(json.dumps({"message": {"text": "m"}})) == ""         # not a string
    assert E._server_message(json.dumps({"code": 404})) == ""
    assert E._server_message(json.dumps(["message"])) == "" and E._server_message("") == ""
    llm_stub_server.chat_reply = (404, "application/json", json.dumps(vllm).encode())
    with pytest.raises(E.LocalLLMUnavailable, match="does not exist") as ei:
        E.HTTPEngine(base_url=llm_stub_server.url, api="openai").complete(
            "qwen3-8b", "hola", system=None, thinking=True, **_CALL)
    assert "HTTP 404" in str(ei.value) and "'qwen3-8b'" in str(ei.value)


class _UnreadableBody(io.BytesIO):
    """The body of an error reply that cannot be read: the server sent the status line and the
    headers, then the connection broke or went silent."""

    def __init__(self, error):
        super().__init__()
        self.error = error

    def read(self, *args):
        raise self.error


@pytest.mark.parametrize("lost", [
    ConnectionResetError(54, "Connection reset by peer"),
    http.client.IncompleteRead(b"", 120),          # an HTTPException, which `except OSError` misses
    TimeoutError("timed out"),                     # the socket timed out reading the body
], ids=["reset", "incomplete-read", "timeout"])
@pytest.mark.parametrize("status,expected", [
    (400, RuntimeError), (404, E.LocalLLMUnavailable), (503, E.LocalLLMUnavailable),
], ids=["400", "404", "503"])
def test_an_error_body_that_cannot_be_read_is_mapped_as_an_empty_one(monkeypatch, status,
                                                                      expected, lost):
    def answer(body):
        """What complete() raises when the server answers ``status`` with ``body``."""
        sent = []

        def urlopen(request, *args, **kwargs):
            sent.append(urllib.error.HTTPError(request.full_url, status, "error", {}, body))
            raise sent[-1]

        monkeypatch.setattr(urllib.request, "urlopen", urlopen)
        with pytest.raises(RuntimeError) as ei:    # the status decides, never the read error
            E.HTTPEngine(base_url="http://h:1", api="openai").complete(
                "m", "hola", system=None, thinking=False, **_CALL)
        assert ei.value.__cause__ is sent[-1]
        return ei.value

    unread, empty = answer(_UnreadableBody(lost)), answer(io.BytesIO(b""))
    assert type(unread) is type(empty) is expected and str(unread) == str(empty)


@pytest.mark.parametrize("reply", [
    (200, "text/html", b"<!doctype html><title>a web app</title>"),
    (200, "application/json", b""),
    (200, "application/json", b"not json"),
    (200, "application/json", b"[]"),
    (500, "application/json", b"{}"),
], ids=["html", "empty", "not-json", "array", "server-error"])
@pytest.mark.parametrize("api", ["ollama", "openai"])
def test_a_probe_needs_a_json_object(llm_stub_server, api, reply):
    llm_stub_server.probe_reply = reply
    assert E.HTTPEngine(base_url=llm_stub_server.url, api=api).available() is False
    llm_stub_server.probe_reply = (200, "application/json", b"{}")
    assert E.HTTPEngine(base_url=llm_stub_server.url, api=api).available() is True


def _record_probes(monkeypatch):
    """Replace ``urlopen`` with one that records ``(url, timeout)`` and fails."""
    seen = []

    def urlopen(request, timeout=None, **kwargs):
        seen.append((request.full_url, timeout))
        raise urllib.error.URLError("no network in this test")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return seen


def _no_local_packages(monkeypatch):
    monkeypatch.setattr(E.MLXEngine, "available", lambda self: False)
    monkeypatch.setattr(E.LlamaCppEngine, "available", lambda self: False)


def test_a_probe_waits_three_seconds_at_most_by_default(monkeypatch):
    seen = _record_probes(monkeypatch)
    assert E.HTTPEngine(base_url="http://h:1", api="ollama").available() is False
    assert E.HTTPEngine(base_url="http://h:1", api="openai", timeout=1.5).available() is False
    assert E.HTTPEngine(base_url="http://h:1", api="openai", probe_timeout=10.0).available() is False
    assert seen == [("http://h:1/api/tags", 3.0), ("http://h:1/v1/models", 1.5),
                    ("http://h:1/v1/models", 10.0)]


def test_resolve_and_list_hand_their_probe_timeout_on(monkeypatch):
    seen = _record_probes(monkeypatch)
    _no_local_packages(monkeypatch)
    with pytest.raises(E.LocalLLMUnavailable):
        E.resolve_engine(probe_timeout=0.5)
    assert seen == [("http://localhost:11434/api/tags", 0.5), ("http://localhost:8080/v1/models", 0.5),
                    ("http://localhost:1234/v1/models", 0.5)]
    seen.clear()
    assert E.available_engines(probe_timeout=0.25) == []
    assert [timeout for _, timeout in seen] == [0.25, 0.25, 0.25]


def test_a_server_that_never_answers_costs_one_probe_timeout(llm_stub_server):
    llm_stub_server.delay = 30.0
    eng = E.HTTPEngine(base_url=llm_stub_server.url, api="ollama", probe_timeout=0.3)
    t0 = time.perf_counter()
    assert eng.available() is False
    # hang guard: the stub stalls 30 s, so the scaled budget stays below the
    # stall for any PUREMACRO_TIMING_SCALE under 12 (CI uses 10)
    assert_within(time.perf_counter() - t0, 2.5, "probe of a silent server")


# ---------------------------------------------------------------- resolve_engine and available_engines

def test_auto_tries_the_packages_then_each_server_in_order(monkeypatch):
    order = []
    monkeypatch.setattr(E.MLXEngine, "available", lambda self: order.append("mlx") or False)
    monkeypatch.setattr(E.LlamaCppEngine, "available", lambda self: order.append("llamacpp") or False)
    monkeypatch.setattr(E.HTTPEngine, "available", lambda self: order.append((self.api, self.base_url)) or False)
    with pytest.raises(E.LocalLLMUnavailable) as ei:
        E.resolve_engine()
    assert order == ["mlx", "llamacpp", *E.DEFAULT_SERVERS]
    assert "openai at http://localhost:1234" in str(ei.value)        # the message says what was tried
    order.clear()
    with pytest.raises(E.LocalLLMUnavailable):
        E.resolve_engine(base_url="http://gpu-box:5000/")
    assert order == ["mlx", "llamacpp", ("ollama", "http://gpu-box:5000"), ("openai", "http://gpu-box:5000")]


def test_the_env_var_replaces_the_default_servers_and_base_url_beats_it(monkeypatch):
    probed = []
    _no_local_packages(monkeypatch)
    monkeypatch.setattr(E.HTTPEngine, "available", lambda self: probed.append((self.api, self.base_url)) or False)
    monkeypatch.setenv(_ENV_URL, " http://env-box:6000 ")
    for base_url in (None, "http://arg-box:7000"):
        with pytest.raises(E.LocalLLMUnavailable):
            E.resolve_engine(base_url=base_url)
    assert probed == [("ollama", "http://env-box:6000"), ("openai", "http://env-box:6000"),
                      ("ollama", "http://arg-box:7000"), ("openai", "http://arg-box:7000")]
    probed.clear()
    monkeypatch.setenv(_ENV_URL, "   ")                                 # blank counts as unset
    with pytest.raises(E.LocalLLMUnavailable):
        E.resolve_engine()
    assert probed == list(E.DEFAULT_SERVERS)


def test_auto_returns_the_first_server_that_answers_with_the_timeouts_asked_for(monkeypatch):
    _no_local_packages(monkeypatch)
    monkeypatch.setattr(E.HTTPEngine, "available", lambda self: self.base_url == "http://localhost:1234")
    eng = E.resolve_engine(timeout=7.0, probe_timeout=1.0)
    assert (eng.name, eng.base_url, eng.timeout, eng.probe_timeout) == ("openai", "http://localhost:1234", 7.0, 1.0)


def test_explicit_engines_and_their_server_urls(monkeypatch):
    _avail(monkeypatch, True, True, True)
    assert isinstance(E.resolve_engine("mlx"), E.MLXEngine)
    assert isinstance(E.resolve_engine("llamacpp"), E.LlamaCppEngine)
    assert E.resolve_engine("ollama").base_url == "http://localhost:11434"
    assert E.resolve_engine("openai").base_url == "http://localhost:8080"
    assert E.resolve_engine("openai", base_url="http://arg-box:7000").base_url == "http://arg-box:7000"
    monkeypatch.setenv(_ENV_URL, "http://env-box:6000")
    assert E.resolve_engine("ollama").base_url == "http://env-box:6000"
    assert E.resolve_engine("openai").base_url == "http://env-box:6000"
    assert E.resolve_engine("ollama", base_url="http://arg-box:7000").base_url == "http://arg-box:7000"


def test_explicit_openai_tries_both_default_ports(monkeypatch):
    monkeypatch.setattr(E.HTTPEngine, "available", lambda self: self.base_url == "http://localhost:1234")
    assert E.resolve_engine("openai").base_url == "http://localhost:1234"
    with pytest.raises(E.LocalLLMUnavailable, match="http://localhost:11434"):
        E.resolve_engine("ollama")


@pytest.mark.parametrize("name", ["mlx", "llamacpp", "ollama", "openai"])
def test_an_explicit_engine_that_is_not_available_raises(monkeypatch, name):
    _avail(monkeypatch, False, False, False)
    with pytest.raises(E.LocalLLMUnavailable) as ei:
        E.resolve_engine(name)
    assert repr(name) in str(ei.value) and "qwen3:4b" in str(ei.value)


def test_an_unknown_engine_is_a_value_error_and_probes_nothing(monkeypatch):
    for cls in (E.MLXEngine, E.LlamaCppEngine, E.HTTPEngine):
        monkeypatch.setattr(cls, "available", lambda self: pytest.fail("probed"))
    with pytest.raises(ValueError, match="'gpt'"):
        E.resolve_engine("gpt")


def test_available_engines_lists_unique_names_in_order(monkeypatch):
    probed = []
    monkeypatch.setattr(E.MLXEngine, "available", lambda self: True)
    monkeypatch.setattr(E.LlamaCppEngine, "available", lambda self: False)
    monkeypatch.setattr(E.HTTPEngine, "available", lambda self: probed.append(self.base_url) or self.api == "openai")
    assert E.available_engines() == ["mlx", "openai"]
    assert probed == ["http://localhost:11434", "http://localhost:8080"]    # one answer per name is enough
    probed.clear()
    assert E.available_engines(probe_servers=False) == ["mlx"] and probed == []
    _avail(monkeypatch, True, True, True)
    assert E.available_engines() == ["mlx", "llamacpp", "ollama", "openai"]


def test_available_engines_follows_base_url_and_the_env_var(monkeypatch):
    probed = []
    _no_local_packages(monkeypatch)
    monkeypatch.setattr(E.HTTPEngine, "available", lambda self: probed.append((self.api, self.base_url)) or False)
    assert E.available_engines(base_url="http://arg-box:7000") == []
    monkeypatch.setenv(_ENV_URL, "http://env-box:6000")
    assert E.available_engines() == []
    assert probed == [("ollama", "http://arg-box:7000"), ("openai", "http://arg-box:7000"),
                      ("ollama", "http://env-box:6000"), ("openai", "http://env-box:6000")]


def test_the_env_var_and_base_url_lose_a_trailing_v1_too(monkeypatch, llm_stub_server):
    _no_local_packages(monkeypatch)
    monkeypatch.setenv(_ENV_URL, f" {llm_stub_server.url}/v1/ ")
    eng = E.resolve_engine()
    assert (eng.name, eng.base_url) == ("ollama", llm_stub_server.url)
    assert E.resolve_engine("openai").base_url == llm_stub_server.url
    assert E.available_engines() == ["ollama", "openai"]
    monkeypatch.delenv(_ENV_URL)
    assert E.resolve_engine("openai", base_url=llm_stub_server.url + "/v1").base_url == llm_stub_server.url
    assert E.available_engines(base_url=llm_stub_server.url + "/v1") == ["ollama", "openai"]


def test_one_trailing_v1_is_stripped_once_whichever_path_builds_the_engine(monkeypatch):
    # resolve_engine and available_engines hand HTTPEngine the URL as it was given, and the engine
    # strips one /v1 (and trailing slashes) from it: a second strip on the way would turn
    # /v1/v1 into the root of the server and /proxy/v1/v1 into /proxy.
    _no_local_packages(monkeypatch)
    monkeypatch.setattr(E.HTTPEngine, "available", lambda self: True)
    for api in ("ollama", "openai", "auto"):
        assert E.resolve_engine(api, base_url="http://h:1/v1/v1").base_url == "http://h:1/v1", api
        assert E.resolve_engine(api, base_url=" http://h:1/v1/v1/ ").base_url == "http://h:1/v1", api
    monkeypatch.setenv(_ENV_URL, "http://h:1/proxy/v1/v1")
    for api in ("ollama", "openai", "auto"):
        assert E.resolve_engine(api).base_url == "http://h:1/proxy/v1", api
    probed = []
    monkeypatch.setattr(E.HTTPEngine, "available", lambda self: probed.append((self.api, self.base_url)) or False)
    assert E.available_engines() == [] and E.available_engines(base_url="http://h:1/v1/v1") == []
    assert probed == [("ollama", "http://h:1/proxy/v1"), ("openai", "http://h:1/proxy/v1"),
                      ("ollama", "http://h:1/v1"), ("openai", "http://h:1/v1")]


def test_a_v1_v1_base_url_is_probed_at_one_v1_below_the_model_list(monkeypatch):
    # the same rule on the wire: the root /v1 plus the /v1/models of the OpenAI API
    _no_local_packages(monkeypatch)
    seen = _record_probes(monkeypatch)
    with pytest.raises(E.LocalLLMUnavailable):
        E.resolve_engine("openai", base_url="http://h:1/v1/v1")
    with pytest.raises(E.LocalLLMUnavailable):
        E.resolve_engine("ollama", base_url="http://h:1/v1/v1")
    assert seen == [("http://h:1/v1/v1/models", 3.0), ("http://h:1/v1/api/tags", 3.0)]


def test_available_engines_against_the_stub(monkeypatch, llm_stub_server):
    _no_local_packages(monkeypatch)
    assert E.available_engines(base_url=llm_stub_server.url) == ["ollama", "openai"]


# ---------------------------------------------------------------- no network at construction

def _block_connections(monkeypatch):
    """Make every outgoing connection fail; return the addresses that were tried."""
    tried = []

    def connect(self, address, *args, **kwargs):
        tried.append(address)
        raise ConnectionRefusedError("outgoing connections are blocked by this test")

    monkeypatch.setattr(socket.socket, "connect", connect)
    return tried


def test_constructing_engines_opens_no_connection(monkeypatch):
    tried = _block_connections(monkeypatch)
    engines = [E.MLXEngine(), E.MLXEngine(adapter_path="adapters"), E.LlamaCppEngine(n_ctx=2048),
               *(E.HTTPEngine(base_url=url, api=api) for api, url in E.DEFAULT_SERVERS)]
    assert [engine.name for engine in engines] == ["mlx", "mlx", "llamacpp", "ollama", "openai", "openai"]
    E.available_engines(probe_servers=False)
    assert tried == []


@pytest.mark.mechanism_control
def test_the_connection_blocker_actually_blocks(monkeypatch, llm_stub_server):
    """Positive control: an inert blocker would let the test above pass while engines probe."""
    engine = E.HTTPEngine(base_url=llm_stub_server.url, api="ollama")
    assert engine.available() is True
    tried = _block_connections(monkeypatch)
    assert engine.available() is False
    assert [address[1] for address in tried] == [int(llm_stub_server.url.rsplit(":", 1)[1])]   # one attempt


# ---------------------------------------------------------------- import rule

_BLOCKED = ("mlx", "mlx_lm", "llama_cpp", "huggingface_hub", "transformers")

_NAMES = textwrap.dedent("""
    import json, sys
    _BLOCKED = set(json.loads(sys.argv[1]))
""")

_BLOCKER = _NAMES + textwrap.dedent("""
    class _Blocker:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in _BLOCKED:
                raise ModuleNotFoundError("blocked (simulated Pyodide): " + name)
            return None
    sys.meta_path.insert(0, _Blocker())
""")

_IMPORT_AND_RUN = textwrap.dedent("""
    import puremacro                      # whatever the package itself pulls in is not ours
    before = set(sys.modules)
    import puremacro.llm.engines as E
    new = set(sys.modules) - before
    # The package imports its cache, which imports sqlite3 through puremacro._optional_stdlib
    # (tests/test_pyodide_compat.py requires that route); that one module is allowed.
    ours_or_stdlib = lambda n: (n.startswith("puremacro.llm") or n == "puremacro._optional_stdlib"
                                or n.split(".")[0] in sys.stdlib_module_names)
    ran = {
        "mlx": E.MLXEngine().available(),
        "llamacpp": E.LlamaCppEngine().available(),
        "engines": E.available_engines(probe_servers=False),
        "strip": E.strip_thinking("<think>x</think>0.5"),
    }
    print(json.dumps({
        **ran,
        "foreign": sorted(n for n in new if not ours_or_stdlib(n)),
        "blocked": sorted(n for n in sys.modules if n.split(".")[0] in _BLOCKED),
        "http_helpers": "puremacro._http" in sys.modules,
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
        importlib.util.find_spec("mlx_lm")        # the very probe MLXEngine.available makes
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
    out = _probe(_BLOCKER + _IMPORT_AND_RUN)
    assert out["foreign"] == []            # nothing but puremacro.llm, its sqlite3 route and the standard library
    assert out["blocked"] == [] and out["http_helpers"] is False
    assert out["mlx"] is False and out["llamacpp"] is False    # a find_spec that raises: not installed
    assert out["engines"] == [] and out["strip"] == "0.5"


def test_the_module_imports_none_of_the_inference_packages_that_are_installed():
    """Without the blocker: an import guarded by ``try`` would pass the test above unnoticed,
    but here it would load whatever this host has installed (mlx, on a Mac with it)."""
    out = _probe(_NAMES + _IMPORT_AND_RUN)
    assert out["foreign"] == [] and out["blocked"] == [] and out["http_helpers"] is False
    assert out["strip"] == "0.5"


@pytest.mark.mechanism_control
def test_the_import_blocker_actually_blocks():
    """Positive control: an inert blocker would make the test above pass on any host."""
    assert _probe(_CONTROL_PROBE) == [*sorted(_BLOCKED), "find_spec"]
