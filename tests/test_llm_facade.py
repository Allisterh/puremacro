# tests/test_llm_facade.py
import pytest
import puremacro.llm as L
import puremacro.llm.engines as E

import contextlib
import json
import os
import sqlite3
import stat
import subprocess
import sys
from pathlib import Path

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
    ro = tmp_path / "ro"
    ro.mkdir()
    os.chmod(ro, stat.S_IRUSR | stat.S_IXUSR)
    try:
        if os.access(ro, os.W_OK):      # root, or Windows: directory permissions are not enforced
            pytest.skip("directory permissions do not restrict this account")
        monkeypatch.setenv("PUREMACRO_LLM_CACHE", str(ro / "c.sqlite"))
        with pytest.warns(UserWarning, match="cannot be written"):
            assert L.complete("hola", max_tokens=8) == "0.7"
    finally:
        os.chmod(ro, stat.S_IRWXU)

def test_report_mentions_mode_engines_model_and_cache(fake):
    r = L.report()
    for needle in ("mode", "fake", "qwen3-4b", "cache"):
        assert needle in r


# ---------------------------------------------------------------- beyond the eleven above
#
# Each test below pins one decision of the design: the token allowance of thinking models, the
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


@pytest.mark.parametrize("temperature", [float("nan"), float("inf"), -float("inf"), -0.1, -1,
                                         "hot", "0.5", True, None, [0.5]],
                         ids=["nan", "inf", "-inf", "-0.1", "-1", "str", "numstr", "bool", "None",
                              "list"])
def test_temperature_must_be_a_finite_number_that_is_not_negative(monkeypatch, fake, temperature):
    """Refused before the cache is opened, in every mode: ``nan`` would otherwise be keyed as the
    text ``nan`` and never hit, and no engine takes a negative temperature."""
    monkeypatch.setattr(E, "resolve_engine",
                        lambda *a, **k: pytest.fail("engine resolved for a bad temperature"))
    for mode in L.MODES:
        with pytest.raises(ValueError, match="temperature") as ei:
            L.complete("hola", mode=mode, max_tokens=8, temperature=temperature)
        assert type(ei.value) is ValueError and repr(temperature) in str(ei.value)
    assert not L.default_cache_path().exists()


def test_temperatures_that_are_valid_reach_the_engine_and_the_key_as_given(recorder):
    for temperature in (0.0, 0.7, 2):
        assert L.complete("hola", max_tokens=8, temperature=temperature) == "0.7"
    assert [c["temperature"] for c in recorder.calls] == [0.0, 0.7, 2]
    assert L.complete("hola", mode="cache", max_tokens=8, temperature=2.0) == "0.7"   # same key as 2
    assert L.complete("hola", mode="cache", max_tokens=8, temperature=-0.0) == "0.7"  # -0.0 keys as 0.0
    assert L.cache_key("qwen3-4b", "hola", max_tokens=8, temperature=0.0, json_mode=False) == \
        "c630a581ab2b323492ac1db08bfc8b8f5a72655db00282142eb5762eacccb586"        # v1 untouched


def test_an_unknown_engine_name_is_refused_before_the_cache_is_read(monkeypatch, fake):
    """A typo in ``engine`` is a ValueError at the call, whatever the mode, even when the cache has
    the answer: it would otherwise pass unnoticed until the first miss."""
    assert L.complete("hola", max_tokens=8) == "0.7"                           # cached now
    monkeypatch.setattr(E, "resolve_engine", lambda *a, **k: pytest.fail("resolved an unknown engine"))
    for mode in L.MODES:
        with pytest.raises(ValueError, match="'olama'"):
            L.complete("hola", mode=mode, max_tokens=8, engine="olama")
    with pytest.raises(ValueError, match="'olama'"):
        L.get_engine("olama")
    assert L.validate_engine_name is E.validate_engine_name and "validate_engine_name" in L.__all__
    assert L.validate_engine_name("auto") == "auto" and L.validate_engine_name("ollama") == "ollama"
    assert L.complete("hola", mode="cache", max_tokens=8, engine="ollama") == "0.7"   # a known name


def test_a_live_answer_that_is_not_valid_unicode_is_returned_and_not_stored(monkeypatch, recwarn):
    """A lone surrogate in a model's text cannot be stored as UTF-8; the answer is still returned (it
    used to raise UnicodeEncodeError after the model had answered) and the cache keeps writing."""
    f = FakeEngine(reply="half \ud83d")
    monkeypatch.setattr(E, "resolve_engine", lambda *a, **k: f)
    monkeypatch.setattr(L.models, "device_budget_gb", lambda: 11.4)
    assert L.complete("p", max_tokens=8) == "half \ud83d"
    with pytest.raises(L.CacheMiss):
        L.complete("p", mode="cache", max_tokens=8)
    f.reply = "0.7"
    assert L.complete("q", max_tokens=8) == "0.7"
    assert L.complete("q", mode="cache", max_tokens=8) == "0.7"
    assert not [w for w in recwarn if "cache" in str(w.message).lower()]


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


# ---------------------------------------------------------------- a cache that cannot be read
#
# In cache mode, a lookup that could not read the cache raises CacheUnavailable, a CacheMiss whose
# reason says why, instead of a miss that tells the user to run the model. Auto mode still degrades
# to no cache, and a file that does not exist yet is an empty cache: a plain miss.

# What a clone made without git-lfs leaves where the cache file should be.
_LFS_POINTER = (b"version https://git-lfs.github.com/spec/v1\n"
                b"oid sha256:" + b"0" * 64 + b"\nsize 24576\n")


def _no_engine_in_cache_mode(monkeypatch):
    monkeypatch.setattr(E, "resolve_engine",
                        lambda *a, **k: pytest.fail("engine resolved in cache mode"))


def test_cache_mode_without_sqlite3_says_the_cache_cannot_be_read_and_how_to_load_it(tmp_path):
    """Pyodide 0.28 before `await pyodide.loadPackage('sqlite3')`, simulated as test_pyodide_compat
    does: the file holds the answer, but without sqlite3 nothing can read it."""
    shipped = tmp_path / "shipped.sqlite"
    key = L.cache_key("qwen3-4b", "hola", max_tokens=8, temperature=0.0, json_mode=False)
    L.CompletionCache(shipped).set(L.CacheRow(key, "qwen3-4b", prompt_sha256("hola", None), 8, 0.0,
                                              False, "0.7", "fake", "qwen3-4b"))
    assert L.complete("hola", mode="cache", max_tokens=8, cache=L.CompletionCache(shipped)) == "0.7"
    code = "\n".join([
        "import json, sys",
        "sys.modules['sqlite3'] = None  # simulate a build without it",
        "import puremacro.llm as L",
        "try:",
        "    got = ['returned', L.complete('hola', mode='cache', max_tokens=8)]",
        "except L.CacheMiss as e:",
        "    kind = type(e)",
        "    got = [f'{kind.__module__}.{kind.__qualname__}', e.key, e.model,",
        "           getattr(e, 'reason', None), str(e)]",
        "print(json.dumps(got))",
    ])
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          cwd=str(Path(__file__).resolve().parents[1]),
                          env={**os.environ, "PUREMACRO_LLM_CACHE": str(shipped)})
    assert proc.returncode == 0, proc.stderr[-2000:]
    kind, got_key, model, reason, message = json.loads(proc.stdout.strip().splitlines()[-1])
    assert kind == "puremacro.llm.cache.CacheUnavailable"          # caught above as a CacheMiss
    assert (got_key, model) == (key, "qwen3-4b")
    assert reason.startswith("ImportError: ") and "sqlite3" in reason
    assert message.startswith("the completion cache cannot be read here")
    assert "await pyodide.loadPackage('sqlite3')" in message
    assert issubclass(L.CacheUnavailable, L.CacheMiss)


def test_cache_mode_names_a_cache_file_that_is_not_a_database(monkeypatch, tmp_path):
    """An LFS pointer where the cache file should be: cache mode says the cache cannot be read and
    why. The same holds for a cache passed in whose file was fine when the object was made."""
    _no_engine_in_cache_mode(monkeypatch)
    pointer = tmp_path / "llm_cache.sqlite"
    pointer.write_bytes(_LFS_POINTER)
    monkeypatch.setenv("PUREMACRO_LLM_CACHE", str(pointer))
    with pytest.raises(L.CacheUnavailable) as ei:
        L.complete("hola", mode="cache", max_tokens=8)
    assert isinstance(ei.value, L.CacheMiss) and ei.value.model == "qwen3-4b"
    assert "file is not a database" in ei.value.reason
    assert str(ei.value).startswith("the completion cache cannot be read here")
    assert "file is not a database" in str(ei.value)
    assert pointer.read_bytes() == _LFS_POINTER                        # left as it was
    mine = L.CompletionCache(tmp_path / "mine.sqlite")
    mine.path.write_bytes(_LFS_POINTER)
    with pytest.raises(L.CacheUnavailable, match="file is not a database"):
        L.complete("hola", mode="cache", max_tokens=8, cache=mine)


def test_cache_mode_says_the_cache_cannot_be_read_when_its_path_runs_through_a_file(monkeypatch, tmp_path):
    """PUREMACRO_LLM_CACHE names the child of a file: no cache can ever be there, so this is not an
    empty cache (a miss that blames the model) but a cache that cannot be read, and why."""
    _no_engine_in_cache_mode(monkeypatch)
    blocker = tmp_path / "llm_cache.sqlite"
    blocker.write_text("x")
    monkeypatch.setenv("PUREMACRO_LLM_CACHE", str(blocker / "c.sqlite"))
    with pytest.raises(L.CacheUnavailable) as ei:
        L.complete("hola", mode="cache", max_tokens=8)
    assert "NotADirectoryError" in ei.value.reason and blocker.read_text() == "x"


def test_cache_mode_with_no_cache_file_yet_is_a_plain_miss(monkeypatch, tmp_path):
    """Also where the file cannot even be created: nothing to read is an empty cache."""
    _no_engine_in_cache_mode(monkeypatch)
    with pytest.raises(L.CacheMiss) as ei:                             # the suite's fresh path
        L.complete("hola", mode="cache", max_tokens=8)
    assert type(ei.value) is L.CacheMiss
    ro = tmp_path / "ro"
    ro.mkdir()
    os.chmod(ro, stat.S_IRUSR | stat.S_IXUSR)
    try:
        if os.access(ro, os.W_OK):
            pytest.skip("directory permissions do not restrict this account")
        monkeypatch.setenv("PUREMACRO_LLM_CACHE", str(ro / "c.sqlite"))
        with pytest.raises(L.CacheMiss) as ei:
            L.complete("hola", mode="cache", max_tokens=8)
        assert type(ei.value) is L.CacheMiss and not (ro / "c.sqlite").exists()
    finally:
        os.chmod(ro, stat.S_IRWXU)


def test_auto_mode_still_degrades_to_no_cache_where_the_cache_cannot_be_read(monkeypatch, fake,
                                                                              tmp_path):
    pointer = tmp_path / "llm_cache.sqlite"
    pointer.write_bytes(_LFS_POINTER)
    monkeypatch.setenv("PUREMACRO_LLM_CACHE", str(pointer))
    with pytest.warns(UserWarning, match="cannot be written"):
        assert L.complete("hola", max_tokens=8) == "0.7"
    assert len(fake.calls) == 1 and pointer.read_bytes() == _LFS_POINTER


def test_report_surfaces_unreadable_cache_reason(monkeypatch, fake, tmp_path):
    pointer = tmp_path / "llm_cache.sqlite"
    pointer.write_bytes(_LFS_POINTER)
    monkeypatch.setenv("PUREMACRO_LLM_CACHE", str(pointer))
    cache_field = _fields(L.report(probe_servers=False))["cache"]
    assert "unreadable:" in cache_field and "file is not a database" in cache_field

