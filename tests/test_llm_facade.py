# tests/test_llm_facade.py
import pytest
import puremacro.llm as L
import puremacro.llm.engines as E

import contextlib
import sqlite3

from puremacro.llm.cache import prompt_sha256

class FakeEngine:
    name = "fake"
    def __init__(self, reply="0.7"): self.reply, self.calls = reply, []
    def available(self): return True
    def complete(self, model_id, prompt, *, system, max_tokens, temperature, json_mode, thinking):
        self.calls.append(dict(model_id=model_id, prompt=prompt, system=system, thinking=thinking)); return self.reply

@pytest.fixture
def fake(monkeypatch):
    f = FakeEngine()
    monkeypatch.setattr(E, "resolve_engine", lambda *a, **k: f)
    monkeypatch.setattr(E, "available_engines", lambda **k: ["fake"])
    monkeypatch.setattr(L.models, "device_budget_gb", lambda: 11.4)
    return f

def test_auto_runs_once_then_hits_cache(fake):
    assert L.complete("hola", max_tokens=8) == "0.7"
    assert L.complete("hola", max_tokens=8) == "0.7"
    assert len(fake.calls) == 1
    assert fake.calls[0]["model_id"] == "qwen3-4b" and fake.calls[0]["thinking"] is True

def test_cache_mode_raises_cachemiss_without_touching_engines(monkeypatch):
    monkeypatch.setattr(E, "resolve_engine", lambda *a, **k: pytest.fail("engine resolved in cache mode"))
    with pytest.raises(L.CacheMiss) as ei:
        L.complete("nunca visto", mode="cache", max_tokens=8)
    assert ei.value.model == "qwen3-4b"

def test_off_mode_never_reads_or_writes(fake):
    L.complete("hola", mode="off", max_tokens=8); L.complete("hola", mode="off", max_tokens=8)
    assert len(fake.calls) == 2
    with pytest.raises(L.CacheMiss): L.complete("hola", mode="cache", max_tokens=8)

def test_mode_from_env(monkeypatch, fake):
    monkeypatch.setenv("PUREMACRO_LLM_MODE", "cache")
    with pytest.raises(L.CacheMiss): L.complete("x", max_tokens=8)
    monkeypatch.setenv("PUREMACRO_LLM_MODE", "loud")
    with pytest.raises(ValueError): L.current_mode()

def test_empty_or_think_only_reply_is_not_cached(monkeypatch):
    f = FakeEngine(reply="<think>solo pensando</think>")
    monkeypatch.setattr(E, "resolve_engine", lambda *a, **k: f)
    monkeypatch.setattr(L.models, "device_budget_gb", lambda: 11.4)
    assert L.complete("p", max_tokens=8) == ""
    with pytest.raises(L.CacheMiss): L.complete("p", mode="cache", max_tokens=8)

def test_model_too_large_raised_before_load(monkeypatch, fake):
    monkeypatch.setattr(L.models, "device_budget_gb", lambda: 4.0)
    with pytest.raises(L.ModelTooLarge) as ei: L.complete("p", model="large", max_tokens=8)
    assert ei.value.suggested == "qwen3-1.7b" and fake.calls == []

def test_unknown_model_passes_through_without_fit_check(monkeypatch, fake):
    monkeypatch.setattr(L.models, "device_budget_gb", lambda: 1.0)
    assert L.complete("p", model="qwen3:4b", max_tokens=8) == "0.7"
    assert fake.calls[0]["model_id"] == "qwen3:4b" and fake.calls[0]["thinking"] is False

def test_chat_single_user_turn_keys_like_complete(fake):
    L.chat([{"role": "system", "content": "S"}, {"role": "user", "content": "hola"}], max_tokens=8)
    assert fake.calls[0]["system"] == "S" and fake.calls[0]["prompt"] == "hola"
    assert L.complete("hola", system="S", mode="cache", max_tokens=8) == "0.7"

def test_chat_multi_turn_role_prefixes(fake):
    L.chat([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}], max_tokens=8)
    assert fake.calls[0]["prompt"] == "User: a\n\nAssistant: b\n\nUser: c"

def test_unwritable_cache_still_returns_live_text(monkeypatch, fake, tmp_path):
    import os, stat
    ro = tmp_path / "ro"; ro.mkdir(); os.chmod(ro, stat.S_IRUSR | stat.S_IXUSR)
    monkeypatch.setenv("PUREMACRO_LLM_CACHE", str(ro / "c.sqlite"))
    assert L.complete("hola", max_tokens=8) == "0.7"
    os.chmod(ro, stat.S_IRWXU)

def test_report_mentions_mode_engines_model_and_cache(fake):
    r = L.report()
    for needle in ("mode", "fake", "qwen3-4b", "cache"):
        assert needle in r


# ---------------------------------------------------------------- beyond the eleven of the brief
#
# Each test below pins one ruling of the plan: the token allowance of thinking models, the
# fit check before any engine is looked for, the row the cache keeps, the mode rules, the
# routes to the engines module, report() and chat().

class RecordingEngine(FakeEngine):
    """A FakeEngine that also records the sampling arguments the facade hands on."""
    def complete(self, model_id, prompt, *, system, max_tokens, temperature, json_mode, thinking):
        self.calls.append(dict(model_id=model_id, prompt=prompt, system=system, max_tokens=max_tokens,
                               temperature=temperature, json_mode=json_mode, thinking=thinking))
        return self.reply


@pytest.fixture
def recorder(monkeypatch):
    f = RecordingEngine()
    monkeypatch.setattr(E, "resolve_engine", lambda *a, **k: f)
    monkeypatch.setattr(L.models, "device_budget_gb", lambda: 11.4)
    return f


def test_a_thinking_model_gets_eight_more_tokens_and_the_key_keeps_the_callers(recorder):
    assert L.complete("hola", max_tokens=8) == "0.7"                      # qwen3-4b thinks
    assert L.complete("hola", model="qwen3:4b", max_tokens=8) == "0.7"    # an engine's own id
    assert L.complete("hola", model="gemma2-2b", max_tokens=8) == "0.7"   # registry, no thinking
    assert [c["max_tokens"] for c in recorder.calls] == [16, 8, 8]
    assert [c["thinking"] for c in recorder.calls] == [True, False, False]
    assert L.complete("hola", mode="cache", max_tokens=8) == "0.7"        # keyed on 8, not 16
    with pytest.raises(L.CacheMiss):
        L.complete("hola", mode="cache", max_tokens=16)


def test_a_model_that_cannot_fit_is_refused_before_any_engine_is_looked_for(monkeypatch):
    monkeypatch.setattr(E, "resolve_engine",
                        lambda *a, **k: pytest.fail("engine resolved for a model that cannot fit"))
    monkeypatch.setattr(L.models, "device_budget_gb", lambda: 4.0)
    with pytest.raises(L.ModelTooLarge) as ei:
        L.complete("p", model="large", max_tokens=8)
    assert (ei.value.alias, ei.value.suggested, ei.value.budget_gb) == ("qwen3-8b", "qwen3-1.7b", 4.0)


def test_a_cache_hit_needs_neither_an_engine_nor_the_memory_to_run_the_model(monkeypatch, fake):
    assert L.complete("p", model="large", max_tokens=8) == "0.7"          # 11.4 GB: the 8B runs
    monkeypatch.setattr(L.models, "device_budget_gb", lambda: 1.0)
    monkeypatch.setattr(E, "resolve_engine", lambda *a, **k: pytest.fail("engine resolved on a hit"))
    assert L.complete("p", model="large", max_tokens=8) == "0.7"
    assert len(fake.calls) == 1


def test_the_stored_row_keeps_the_callers_parameters_and_records_the_engine(monkeypatch, recorder):
    recorder.name = "llamacpp"                     # an engine whose model id is a (repo, file) pair
    pair = L.MODELS["qwen3-4b"].llamacpp
    assert L.complete("hola", system="S", max_tokens=8, temperature=0.25, json_mode=True) == "0.7"
    call = recorder.calls[0]
    assert (call["model_id"], call["system"], call["temperature"], call["json_mode"]) == (pair, "S", 0.25, True)
    with contextlib.closing(sqlite3.connect(L.default_cache_path())) as conn:
        rows = conn.execute("SELECT key, model, prompt_sha256, max_tokens, temperature, json_mode, "
                            "completion, engine, engine_model_id FROM completions").fetchall()
    key = L.cache_key("qwen3-4b", "hola", system="S", max_tokens=8, temperature=0.25, json_mode=True)
    assert rows == [(key, "qwen3-4b", prompt_sha256("hola", "S"), 8, 0.25, 1, "0.7", "llamacpp", str(pair))]


def test_an_explicit_cache_is_used_instead_of_the_default_one(fake, tmp_path):
    mine = L.CompletionCache(tmp_path / "mine.sqlite")
    assert L.complete("hola", max_tokens=8, cache=mine) == "0.7"
    key = L.cache_key("qwen3-4b", "hola", max_tokens=8, temperature=0.0, json_mode=False)
    assert mine.get(key) == "0.7"
    assert L.CompletionCache().get(key) is None
    assert L.complete("hola", mode="cache", max_tokens=8, cache=mine) == "0.7"


def test_off_mode_leaves_no_cache_file_behind(fake):
    assert L.complete("hola", mode="off", max_tokens=8) == "0.7"
    assert not L.default_cache_path().exists()


def test_off_mode_builds_no_cache_and_leaves_a_given_one_untouched(monkeypatch, fake, tmp_path):
    mine = L.CompletionCache(tmp_path / "mine.sqlite")
    key = L.cache_key("qwen3-4b", "hola", max_tokens=8, temperature=0.0, json_mode=False)
    mine.set(L.CacheRow(key, "qwen3-4b", prompt_sha256("hola", None), 8, 0.0, False,
                        "cached", "fake", "qwen3-4b"))
    monkeypatch.setattr(L.cache, "CompletionCache",
                        lambda *a, **k: pytest.fail("off mode built a CompletionCache"))
    assert L.complete("hola", mode="off", max_tokens=8) == "0.7"
    assert L.complete("hola", mode="off", max_tokens=8, cache=mine) == "0.7"   # its row is not read
    assert L.complete("adios", mode="off", max_tokens=8, cache=mine) == "0.7"  # no row is written
    assert len(fake.calls) == 3 and mine.stats()["rows"] == 1


def test_the_mode_argument_wins_over_the_variable_and_both_are_checked(monkeypatch):
    assert L.MODES == ("auto", "cache", "off")
    assert L.current_mode() == "auto"
    monkeypatch.setenv("PUREMACRO_LLM_MODE", "cache")
    assert L.current_mode() == "cache" and L.current_mode("off") == "off"
    with pytest.raises(ValueError, match="loud"):
        L.current_mode("loud")
    with pytest.raises(ValueError, match="loud"):
        L.complete("x", mode="loud", max_tokens=8)
    monkeypatch.setenv("PUREMACRO_LLM_MODE", " Off ")
    assert L.current_mode() == "off"
    monkeypatch.setenv("PUREMACRO_LLM_MODE", "  ")             # blank counts as unset
    assert L.current_mode() == "auto"
    monkeypatch.setenv("PUREMACRO_LLM_MODE", "loud")
    with pytest.raises(ValueError, match="PUREMACRO_LLM_MODE"):
        L.current_mode()


def test_engine_name_base_url_and_timeout_reach_the_resolver(monkeypatch):
    seen, f = [], FakeEngine()
    monkeypatch.setattr(E, "resolve_engine", lambda *a, **k: seen.append((a, k)) or f)
    monkeypatch.setattr(L.models, "device_budget_gb", lambda: 11.4)
    assert L.get_engine("ollama", timeout=5.0, base_url="http://h:1") is f
    assert L.complete("hola", engine="openai", base_url="http://h:2", timeout=7.0, max_tokens=8) == "0.7"
    assert seen == [(("ollama",), dict(base_url="http://h:1", timeout=5.0)),
                    (("openai",), dict(base_url="http://h:2", timeout=7.0))]


def test_available_engines_asks_the_engines_module(monkeypatch):
    seen = []
    monkeypatch.setattr(E, "available_engines", lambda **k: seen.append(k) or ["mlx"])
    assert L.available_engines(probe_servers=False) == ["mlx"]
    assert L.available_engines() == ["mlx"]
    assert seen == [dict(probe_servers=False), dict(probe_servers=True)]


def _fields(text):
    """The ``label : value`` lines of a report, as a dict."""
    return {label.strip(): value.strip()
            for label, value in (line.split(":", 1) for line in text.splitlines()[1:])}


def test_report_lists_each_item_on_its_own_line(fake):
    assert L.complete("hola", max_tokens=8) == "0.7"
    fields = _fields(L.report())
    assert set(fields) == {"host", "device", "memory", "budget", "engines", "model", "cache", "mode"}
    assert fields["engines"] == "fake" and fields["budget"].startswith("11.40 GB")
    assert fields["model"].startswith("qwen3-4b (about 3.5 GB)")
    assert fields["cache"].startswith(str(L.default_cache_path())) and "1 row," in fields["cache"]
    assert fields["mode"].startswith("auto")


def test_report_passes_the_probe_switch_on_and_never_probes_servers_in_cache_mode(monkeypatch, fake):
    seen = []
    monkeypatch.setattr(E, "available_engines", lambda **k: seen.append(k["probe_servers"]) or ["fake"])
    L.report(probe_servers=False)
    L.report()
    monkeypatch.setenv("PUREMACRO_LLM_MODE", "cache")
    assert "not probed" in _fields(L.report())["engines"]
    assert seen == [False, True, False]


def test_report_says_unknown_where_capabilities_or_stats_fail(monkeypatch, fake):
    import puremacro.runtime._capabilities as caps

    def broken(*args, **kwargs):
        raise RuntimeError("probe failed")

    assert L.complete("hola", max_tokens=8) == "0.7"           # the cache file exists, so stats() runs
    monkeypatch.setattr(caps, "capabilities", broken)
    monkeypatch.setattr(L.CompletionCache, "stats", broken)
    fields = _fields(L.report())
    assert fields["host"] == fields["device"] == fields["memory"] == "unknown"
    assert fields["cache"].endswith("(unknown)")
    assert fields["engines"] == "fake" and fields["model"].startswith("qwen3-4b")


def test_report_does_not_create_the_cache_file(fake):
    assert _fields(L.report())["cache"] == f"{L.default_cache_path()} (no file yet)"
    assert not L.default_cache_path().exists()


def test_report_names_an_invalid_mode_instead_of_raising(monkeypatch, fake):
    monkeypatch.setenv("PUREMACRO_LLM_MODE", "loud")
    assert "loud" in _fields(L.report())["mode"]


def test_chat_takes_a_system_keyword_as_the_first_system_turn(fake):
    L.chat([{"role": "system", "content": "B"}, {"role": "user", "content": "hola"}], system="A", max_tokens=8)
    assert fake.calls[0]["system"] == "A\nB" and fake.calls[0]["prompt"] == "hola"


def test_chat_needs_a_turn_that_is_not_a_system_turn(fake):
    with pytest.raises(ValueError):
        L.chat([{"role": "system", "content": "S"}], max_tokens=8)
    with pytest.raises(ValueError):
        L.chat([], max_tokens=8)
    assert fake.calls == []
