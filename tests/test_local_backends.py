# tests/test_local_backends.py
"""LocalBackend/LocalProvider wiring at the two call sites, via a fake engine.

Both go through :func:`puremacro.llm.complete`, which reaches the engines through the module
``puremacro.llm.engines``; the fake engine is installed there (not on the
``puremacro.narrative._local_engines`` re-export, whose ``resolve_engine`` the facade never
calls). No test here reaches a real engine or a live server.
"""
import hashlib

import pytest

import puremacro.llm.engines
import puremacro.llm.models
from puremacro.llm import CacheMiss, CacheUnavailable, ModelTooLarge
from puremacro.narrative import _local_engines as le
from puremacro.narrative.scoring import (
    LocalBackend, OllamaBackend, MockBackend, get_default_backend, score_llm,
)
from puremacro.narrative.scoring.llm import _PROMPTS


class _FakeEngine:
    def __init__(self, reply="0.7", *, name="fake"):
        self.name = name
        self.reply = reply
        self.calls = []

    def available(self):
        return True

    def complete(self, model, prompt, *, system=None, max_tokens, temperature, json_mode,
                 thinking=False):
        self.calls.append({"model": model, "prompt": prompt, "system": system,
                           "max_tokens": max_tokens, "temperature": temperature,
                           "json_mode": json_mode, "thinking": thinking})
        if isinstance(self.reply, BaseException):
            raise self.reply
        return self.reply


@pytest.fixture(autouse=True)
def _roomy_device(monkeypatch):
    """Every registry model fits, so ``complete`` never stops at ModelTooLarge here."""
    monkeypatch.setattr(puremacro.llm.models, "device_budget_gb", lambda: 64.0)


def _no_engine(*a, **k):
    pytest.fail("an engine was resolved")


def test_local_backend_calls_engine(monkeypatch):
    fake = _FakeEngine("[]", name="mlx")
    monkeypatch.setattr(puremacro.llm.engines, "resolve_engine", lambda *a, **k: fake)
    be = LocalBackend("qwen2.5-3b-instruct", engine="auto")
    assert be.call("hello") == "[]"
    # canonical name resolved to the per-engine (mlx) id; JSON mode off by default:
    assert fake.calls[0]["model"] == "mlx-community/Qwen2.5-3B-Instruct-4bit"
    assert fake.calls[0]["json_mode"] is False
    assert fake.calls[0]["max_tokens"] == 1024


def test_local_backend_json_mode_defaults_to_false():
    # The prompts ask for a JSON array; an object-forcing JSON mode yields zero events.
    assert LocalBackend().json_mode is False


def test_local_backend_passes_engine_base_url_and_timeout(monkeypatch):
    seen = {}

    def resolve(engine, **k):
        seen.update(engine=engine, **k)
        return _FakeEngine("[]")

    monkeypatch.setattr(puremacro.llm.engines, "resolve_engine", resolve)
    LocalBackend("qwen2.5-3b-instruct", engine="ollama", base_url="http://h:1",
                 timeout=5.0).call("hello")
    assert seen == {"engine": "ollama", "base_url": "http://h:1", "timeout": 5.0}


def test_local_backend_and_provider_resolve_no_engine_at_construction(monkeypatch):
    monkeypatch.setattr(puremacro.llm.engines, "resolve_engine", _no_engine)
    monkeypatch.setattr(puremacro.llm.engines, "available_engines", _no_engine)
    be = LocalBackend("default")
    p = LocalProvider("default")
    assert be.model == "qwen3-4b" and p.model == "qwen3-4b"


def test_ollama_presets_default_to_qwen3_4b():
    # The registry alias, so the cache key is the reference model's; the server still gets its tag.
    # base_url is None, so PUREMACRO_LLM_BASE_URL, else the default Ollama URL, names the server.
    be = OllamaBackend()
    assert (be.model, be.engine, be.base_url) == ("qwen3-4b", "ollama", None)
    p = OllamaProvider()
    assert (p.model, p.engine, p.base_url) == ("qwen3-4b", "ollama", None)
    assert OllamaBackend(base_url="http://h:1").base_url == "http://h:1"
    assert OllamaProvider(base_url="http://h:1").base_url == "http://h:1"


def test_ollama_presets_follow_the_server_the_environment_names(monkeypatch, llm_stub_server):
    monkeypatch.setenv("PUREMACRO_LLM_BASE_URL", llm_stub_server.url)
    llm_stub_server.chat_content = "[]"
    assert OllamaBackend().call("score this") == "[]"
    assert llm_stub_server.last_body["model"] == "qwen3:4b"
    llm_stub_server.chat_content = "0.6"
    assert OllamaProvider().score_paragraph("An uncertain outlook.", "uncertainty") == 0.6


def test_ollama_presets_pass_an_explicit_tag_through():
    assert OllamaBackend("qwen3:4b").model == "qwen3:4b"
    assert OllamaProvider("qwen3:4b").model == "qwen3:4b"


def test_ollama_backend_sends_the_ollama_tag_and_shares_the_reference_cache_key(monkeypatch):
    fake = _FakeEngine("[]", name="ollama")
    monkeypatch.setattr(puremacro.llm.engines, "resolve_engine", lambda *a, **k: fake)
    assert OllamaBackend().call("score this") == "[]"
    assert fake.calls[0]["model"] == "qwen3:4b"
    fake.reply = "unused"                                   # a second engine call would show
    assert LocalBackend("default", mode="cache").call("score this") == "[]"
    assert len(fake.calls) == 1


def test_ollama_provider_sends_the_ollama_tag_and_shares_the_reference_cache_key(monkeypatch):
    fake = _FakeEngine("0.6", name="ollama")
    monkeypatch.setattr(puremacro.llm.engines, "resolve_engine", lambda *a, **k: fake)
    assert OllamaProvider().score_paragraph("An uncertain outlook.", "uncertainty") == 0.6
    assert fake.calls[0]["model"] == "qwen3:4b"
    monkeypatch.setattr(puremacro.llm.engines, "resolve_engine", _no_engine)
    assert LocalProvider("default", mode="cache").score_paragraph(
        "An uncertain outlook.", "uncertainty") == 0.6


class _TooLargeBackend:
    model = "qwen3-8b"
    max_tokens = 8
    temperature = 0.0

    def call(self, prompt):
        raise ModelTooLarge("qwen3-8b", "qwen3-4b", 4.0)


def test_score_llm_raises_model_too_large():
    # A configuration error, identical for every record: raised, not counted as malformed.
    with pytest.raises(ModelTooLarge):
        score_llm([("2020-01-01", "text", "http://x"), ("2020-02-01", "text", "http://y")],
                  backend=_TooLargeBackend(), kind="fiscal")


def test_score_llm_raises_model_too_large_through_local_backend(monkeypatch):
    monkeypatch.setattr(puremacro.llm.models, "device_budget_gb", lambda: 1.0)
    monkeypatch.setattr(puremacro.llm.engines, "resolve_engine", _no_engine)
    with pytest.raises(ModelTooLarge):
        score_llm([("2020-01-01", "text", "http://x")], backend=LocalBackend("large"),
                  kind="fiscal")



def test_mock_backend_returns_empty_array():
    assert MockBackend().call("anything") == "[]"


def test_get_default_backend_falls_back_to_mock(monkeypatch, capsys):
    monkeypatch.setattr(puremacro.llm.engines, "available_engines", lambda **k: [])
    be = get_default_backend()
    assert isinstance(be, MockBackend)
    assert "[puremacro] local LLM: mode=auto engine=mock model=qwen3-4b" in capsys.readouterr().out


def test_get_default_backend_returns_local_when_an_engine_is_available(monkeypatch, capsys):
    monkeypatch.setattr(puremacro.llm.engines, "available_engines", lambda **k: ["ollama"])
    monkeypatch.setattr(puremacro.llm.engines, "resolve_engine", _no_engine)
    be = get_default_backend()
    assert isinstance(be, LocalBackend) and be.model == "qwen3-4b"
    out = capsys.readouterr().out
    assert out.strip().splitlines()[-1] == "[puremacro] local LLM: mode=auto engine=ollama model=qwen3-4b"


def test_get_default_backend_in_cache_mode_is_local_without_probes(monkeypatch):
    monkeypatch.setenv("PUREMACRO_LLM_MODE", "cache")
    monkeypatch.setattr(puremacro.llm.engines, "available_engines", lambda **k: pytest.fail("probed"))
    assert isinstance(get_default_backend(), LocalBackend)


def test_get_default_backend_cache_mode_line(monkeypatch, capsys):
    monkeypatch.setattr(puremacro.llm.engines, "available_engines", _no_engine)
    get_default_backend(mode="cache")
    assert "[puremacro] local LLM: mode=cache engine=cache model=qwen3-4b" in capsys.readouterr().out


def test_score_llm_runs_with_mock_backend():
    # MockBackend -> "[]" -> zero events, but the loop completes cleanly.
    out = score_llm([("2020-01-01", "fiscal stimulus announced", "http://x")],
                    backend=MockBackend(), kind="fiscal")
    assert out == []


class _DownBackend:
    model = "x"
    max_tokens = 8
    temperature = 0.0

    def call(self, prompt):
        raise le.LocalLLMUnavailable("server down")


class _GarbageBackend:
    model = "x"
    max_tokens = 8
    temperature = 0.0

    def call(self, prompt):
        return "not json at all"


def test_score_llm_propagates_backend_unavailable():
    with pytest.raises(le.BackendUnavailable):
        score_llm([("2020-01-01", "text", "http://x")],
                  backend=_DownBackend(), kind="fiscal")


def test_score_llm_propagates_unavailable_engine_through_local_backend(monkeypatch):
    def down(*a, **k):
        raise puremacro.llm.engines.LocalLLMUnavailable("none here")

    monkeypatch.setattr(puremacro.llm.engines, "resolve_engine", down)
    with pytest.raises(le.BackendUnavailable):
        score_llm([("2020-01-01", "text", "http://x")], backend=LocalBackend(), kind="fiscal")


def test_score_llm_still_drops_parse_errors():
    # Garbage output is dropped (not raised); loop returns [] cleanly.
    out = score_llm([("2020-01-01", "text", "http://x")],
                    backend=_GarbageBackend(), kind="fiscal")
    assert out == []


def test_score_llm_counts_an_engine_timeout_as_malformed(monkeypatch, capsys):
    # 4.8.1 behaviour: a per-record failure other than BackendUnavailable is dropped.
    fake = _FakeEngine(TimeoutError("timed out"))
    monkeypatch.setattr(puremacro.llm.engines, "resolve_engine", lambda *a, **k: fake)
    out = score_llm([("2020-01-01", "text", "http://x"), ("2020-02-01", "text", "http://y")],
                    backend=LocalBackend(), kind="fiscal")
    assert out == [] and len(fake.calls) == 2
    assert "dropped 2 malformed" in capsys.readouterr().out


_EVENT = ('[{"magnitude_usd_bn": 500.0, "target": "investment", "subtarget": "infra", "sign": 1, '
          '"confidence": 0.9, "excerpt": "a $500 billion package", "implementation_date": null, "horizon_quarters": null}]')


def test_score_llm_reads_cache_without_engine(monkeypatch):
    corpus = [("2020-03-15", "The government announced a $500 billion infrastructure package.", "u")]
    monkeypatch.setattr(puremacro.llm.engines, "resolve_engine", lambda *a, **k: _FakeEngine(_EVENT))
    first = score_llm(corpus, backend=LocalBackend("default"), kind="fiscal")
    assert len(first) == 1
    monkeypatch.setattr(puremacro.llm.engines, "resolve_engine", lambda *a, **k: pytest.fail("engine used in cache mode"))
    monkeypatch.setenv("PUREMACRO_LLM_MODE", "cache")
    second = score_llm(corpus, backend=LocalBackend("default"), kind="fiscal")
    assert [(e.date, e.sign, e.magnitude) for e in second] == [(e.date, e.sign, e.magnitude) for e in first]


def test_score_llm_surfaces_a_cache_miss_in_cache_mode(monkeypatch):
    monkeypatch.setenv("PUREMACRO_LLM_MODE", "cache")
    monkeypatch.setattr(puremacro.llm.engines, "resolve_engine", _no_engine)
    with pytest.raises(CacheMiss):
        score_llm([("2020-03-15", "never scored", "u")], backend=LocalBackend(), kind="fiscal")


def test_score_llm_surfaces_an_unreadable_cache_in_cache_mode(monkeypatch, tmp_path):
    garbage = tmp_path / "not_a_cache.sqlite"
    garbage.write_bytes(b"this is not an SQLite file" * 64)
    monkeypatch.setenv("PUREMACRO_LLM_CACHE", str(garbage))
    monkeypatch.setattr(puremacro.llm.engines, "resolve_engine", _no_engine)
    with pytest.raises(CacheUnavailable):
        score_llm([("2020-03-15", "text", "u")], backend=LocalBackend(mode="cache"), kind="fiscal")


def test_local_backend_mode_argument_overrides_environment(monkeypatch):
    monkeypatch.setattr(puremacro.llm.engines, "resolve_engine", _no_engine)
    with pytest.raises(CacheMiss):
        LocalBackend("default", mode="cache").call("hello")


# ---------------------------------------------------------------- names checked at construction
# A misspelt engine or mode is a configuration error, identical for every record. Found at the
# first call it would be counted as a malformed event by score_llm and the corpus would come back
# empty; so the constructors check the names, as the engine resolution of 4.8.1 did.

_BAD_NAMES = [
    pytest.param({"engine": "foo"}, r"unknown engine 'foo'; expected 'auto' or one of \[", id="engine-foo"),
    pytest.param({"engine": "olama"}, r"unknown engine 'olama'", id="engine-olama"),
    pytest.param({"mode": "x"}, r"unknown mode 'x'; use 'auto', 'cache' or 'off'", id="mode-x"),
    pytest.param({"mode": "cahce"}, r"unknown mode 'cahce'", id="mode-cahce"),
]


@pytest.mark.parametrize("kw, message", _BAD_NAMES)
def test_local_backend_refuses_an_unknown_engine_or_mode_at_construction(kw, message, monkeypatch):
    monkeypatch.setattr(puremacro.llm.engines, "resolve_engine", _no_engine)
    monkeypatch.setattr(puremacro.llm.engines, "available_engines", _no_engine)
    with pytest.raises(ValueError, match=message):
        LocalBackend(**kw)


@pytest.mark.parametrize("kw, message", _BAD_NAMES)
def test_local_provider_refuses_an_unknown_engine_or_mode_at_construction(kw, message, monkeypatch):
    monkeypatch.setattr(puremacro.llm.engines, "resolve_engine", _no_engine)
    monkeypatch.setattr(puremacro.llm.engines, "available_engines", _no_engine)
    with pytest.raises(ValueError, match=message):
        LocalProvider(**kw)


def test_the_engine_names_and_the_modes_are_accepted_at_construction(monkeypatch):
    monkeypatch.setattr(puremacro.llm.engines, "resolve_engine", _no_engine)
    monkeypatch.setattr(puremacro.llm.engines, "available_engines", _no_engine)
    for name in ("auto", "mlx", "llamacpp", "ollama", "openai"):
        assert LocalBackend(engine=name).engine == name
        assert LocalProvider(engine=name).engine == name
    for mode in ("auto", "cache", "off", None):
        assert LocalBackend(mode=mode).mode == mode
        assert LocalProvider(mode=mode).mode == mode


def test_the_engine_name_check_is_the_public_validator_of_puremacro_llm(monkeypatch):
    # Both constructors hand the name to puremacro.llm.validate_engine_name, "auto" included, and
    # keep what it returns; nothing is probed.
    seen = []

    def validator(name):
        seen.append(name)
        if name == "foo":
            raise ValueError(f"the facade's validator saw {name!r}")
        return name

    monkeypatch.setattr(puremacro.llm, "validate_engine_name", validator)
    with pytest.raises(ValueError, match="the facade's validator saw 'foo'"):
        LocalBackend(engine="foo")
    assert LocalBackend(engine="auto").engine == "auto"
    with pytest.raises(ValueError, match="the facade's validator saw 'foo'"):
        LocalProvider(engine="foo")
    assert seen == ["foo", "auto", "foo"]
    monkeypatch.undo()                                           # the real validator again
    with pytest.raises(ValueError, match=r"unknown engine 'foo'; expected 'auto' or one of \["):
        LocalBackend(engine="foo")


def test_local_backend_refuses_a_misspelt_mode_variable_at_construction(monkeypatch):
    # mode=None defers to PUREMACRO_LLM_MODE at every call; a variable that names no mode is
    # found at construction, not swallowed per record.
    monkeypatch.setenv("PUREMACRO_LLM_MODE", "cahce")
    with pytest.raises(ValueError, match="PUREMACRO_LLM_MODE"):
        LocalBackend()
    with pytest.raises(ValueError, match="PUREMACRO_LLM_MODE"):
        LocalProvider()


def test_score_llm_raises_on_a_misspelt_mode_variable_instead_of_zero_events(monkeypatch, capsys):
    monkeypatch.setenv("PUREMACRO_LLM_MODE", "cahce")
    monkeypatch.setattr(puremacro.llm.engines, "resolve_engine", _no_engine)
    with pytest.raises(ValueError, match="PUREMACRO_LLM_MODE"):
        score_llm([("2020-03-15", "text", "u")], backend=LocalBackend(), kind="fiscal")
    assert "malformed" not in capsys.readouterr().out


def test_get_default_backend_and_provider_refuse_a_misspelt_engine(monkeypatch, capsys):
    # Not the Mock fallback: that is for "no engine found", not for a name that is no engine.
    monkeypatch.setattr(puremacro.llm.engines, "available_engines", lambda **k: [])
    with pytest.raises(ValueError, match="unknown engine 'olama'"):
        get_default_backend(engine="olama")
    with pytest.raises(ValueError, match="unknown engine 'olama'"):
        get_default_provider(engine="olama")
    assert capsys.readouterr().out == ""


def test_get_default_backend_and_provider_keep_the_mock_fallback_for_an_absent_engine(monkeypatch, capsys):
    monkeypatch.setattr(puremacro.llm.engines, "available_engines", lambda **k: ["mlx"])
    assert isinstance(get_default_backend(engine="ollama"), MockBackend)
    assert isinstance(get_default_provider(engine="ollama"), MockProvider)
    assert capsys.readouterr().out.count("engine=mock") == 2


def test_scoring_llm_no_longer_imports_ssl():
    # Nothing in the module used it: the unverified-context fallback it once served is gone.
    import puremacro.narrative.scoring.llm as scoring_llm
    assert not hasattr(scoring_llm, "ssl")


# The sha256 of each prompt template and of the paragraph prompt, as they are in 4.8.1
# (computed from the text at the commit this task started from, identical to v4.8.1).
# The completion cache is keyed on the prompt, so a change here invalidates every
# shipped cache: change these values only together with a new cache.
_PROMPT_SHA256_481 = {
    "fiscal": "2458bfbb95f8c0787ccd684bdd0f0c18602bbe0caa173f00a77ef3cec45935d5",
    "monetary": "e4e3263c977f9f52bed92c75933692685b2758cf032b220dd4280b92daec4869",
    "macropru": "bb6d154266ce5de43739bb4e08b8b7929ab77306c20af4974919ba0aea5eae50",
    "fx": "a024a2911867e0a4400e64ca6013db633b52db3b872be1bc48f432caa87b4d5b",
    "structural": "aac351ef53cde96a4361ac54de4e2ba039ae861d06c0f4d16f060cc1c7e23ee7",
}
_PARAGRAPH_PROMPT_SHA256_481 = "c2b8ad573d222ab541fd4dfc2738339563bcccc30a1b42872ea57b6bb79984c1"


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_prompt_templates_are_those_of_4_8_1():
    assert {kind: _sha(t) for kind, t in _PROMPTS.items()} == _PROMPT_SHA256_481


def test_paragraph_prompt_is_that_of_4_8_1(monkeypatch):
    fake = _FakeEngine("0.4")
    monkeypatch.setattr(puremacro.llm.engines, "resolve_engine", lambda *a, **k: fake)
    LocalProvider().score_paragraph("Inflation risks remain elevated.", "uncertainty")
    assert _sha(fake.calls[0]["prompt"]) == _PARAGRAPH_PROMPT_SHA256_481
    assert fake.calls[0]["system"] is None and fake.calls[0]["json_mode"] is False


from puremacro.narrative.indices import (
    LocalProvider, OllamaProvider, get_default_provider, MockProvider, llm_prob_kernel,
)


def test_local_provider_parses_float(monkeypatch):
    fake = _FakeEngine("The score is 0.73.", name="ollama")
    monkeypatch.setattr(puremacro.llm.engines, "resolve_engine", lambda *a, **k: fake)
    p = LocalProvider("qwen2.5-3b-instruct", engine="ollama")
    assert p.name == "local"                       # score cache rows hit across engines
    assert abs(p.score_paragraph("very uncertain outlook", "uncertainty")
               - 0.73) < 1e-9
    assert fake.calls[0]["model"] == "qwen2.5:3b"
    assert fake.calls[0]["max_tokens"] == 8 and fake.calls[0]["temperature"] == 0.0


def test_local_provider_name_is_local_and_model_canonical(monkeypatch):
    monkeypatch.setattr(puremacro.llm.engines, "resolve_engine", lambda *a, **k: _FakeEngine("0.4"))
    p = LocalProvider("default")
    assert p.name == "local" and p.model == "qwen3-4b"


def test_local_provider_clamps_and_defaults(monkeypatch):
    monkeypatch.setattr(puremacro.llm.engines, "resolve_engine",
                        lambda *a, **k: _FakeEngine("no number here"))
    p = LocalProvider(engine="ollama")
    assert p.score_paragraph("x", "uncertainty") == 0.0


def test_local_provider_clamps_out_of_range(monkeypatch):
    # A reply above 1.0 is clamped to 1.0 (the min() guard).
    monkeypatch.setattr(puremacro.llm.engines, "resolve_engine",
                        lambda *a, **k: _FakeEngine("1.8"))
    p = LocalProvider(engine="ollama")
    assert p.score_paragraph("x", "uncertainty") == 1.0


def test_local_provider_in_cache_mode_raises_cache_miss(monkeypatch):
    monkeypatch.setattr(puremacro.llm.engines, "resolve_engine", _no_engine)
    with pytest.raises(CacheMiss):
        list(llm_prob_kernel([("2020-01-01", "An uncertain outlook.", "u")],
                             provider=LocalProvider(mode="cache"), category="uncertainty"))


def test_get_default_provider_falls_back_to_mock(monkeypatch, capsys):
    monkeypatch.setattr(puremacro.llm.engines, "available_engines", lambda **k: [])
    assert isinstance(get_default_provider(), MockProvider)
    assert "[puremacro] local LLM: mode=auto engine=mock model=qwen3-4b" in capsys.readouterr().out


def test_get_default_provider_returns_local_when_an_engine_is_available(monkeypatch, capsys):
    monkeypatch.setattr(puremacro.llm.engines, "available_engines", lambda **k: ["mlx", "ollama"])
    p = get_default_provider("small")
    assert isinstance(p, LocalProvider) and p.model == "qwen3-1.7b"
    assert "[puremacro] local LLM: mode=auto engine=mlx model=qwen3-1.7b" in capsys.readouterr().out


def test_get_default_provider_in_cache_mode_is_local_without_probes(monkeypatch, capsys):
    monkeypatch.setenv("PUREMACRO_LLM_MODE", "cache")
    monkeypatch.setattr(puremacro.llm.engines, "available_engines", lambda **k: pytest.fail("probed"))
    assert isinstance(get_default_provider(), LocalProvider)
    assert "[puremacro] local LLM: mode=cache engine=cache model=qwen3-4b" in capsys.readouterr().out
