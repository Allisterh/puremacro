# tests/test_llm_cache.py
import contextlib
import hashlib
import os
import pickle
import sqlite3
import stat
from pathlib import Path

import pytest
from puremacro.llm.cache import (CACHE_FORMAT, CacheMiss, CacheRow, CompletionCache,
                                 cache_key, default_cache_path, prompt_sha256)

GOLDEN = "c630a581ab2b323492ac1db08bfc8b8f5a72655db00282142eb5762eacccb586"

def test_golden_key():
    assert CACHE_FORMAT == "v1"
    assert prompt_sha256("hola") == "3173f87dc99e79b2df6105b53594c5459bb363bd663c1b9d908195688481bc5f"
    assert cache_key("qwen3-4b", "hola", max_tokens=8, temperature=0.0, json_mode=False) == GOLDEN

def test_golden_key_non_ascii_with_system():
    k = cache_key("qwen3-4b", "¿Qué es el PIB?", system="Eres un asistente.",
                  max_tokens=16, temperature=0.0, json_mode=False)
    assert k == "0fb187545fac544435c8fec55bd98117e4504c10088a756d9b24bc44452d7f7d"

def test_key_ignores_engine_but_not_params():
    base = dict(max_tokens=8, temperature=0.0, json_mode=False)
    assert cache_key("qwen3-4b", "hola", **base) != cache_key("qwen3-4b", "hola", **{**base, "max_tokens": 9})
    assert cache_key("qwen3-4b", "hola", **base) != cache_key("qwen3-4b", "hola", **{**base, "json_mode": True})

def test_round_trip_and_stats(tmp_path):
    c = CompletionCache(tmp_path / "c.sqlite")
    k = cache_key("qwen3-4b", "hola", max_tokens=8, temperature=0.0, json_mode=False)
    assert c.get(k) is None
    c.set(CacheRow(k, "qwen3-4b", prompt_sha256("hola"), 8, 0.0, False, "¡Hola!", "mlx", "mlx-community/Qwen3-4B-4bit"))
    assert c.get(k) == "¡Hola!"
    s = c.stats()
    assert s["rows"] == 1 and s["by_model"] == {"qwen3-4b": 1} and s["bytes"] > 0

def test_export_filters_by_model_and_merge_ignores_duplicates(tmp_path):
    a = CompletionCache(tmp_path / "a.sqlite")
    for model, text in [("qwen3-4b", "x"), ("qwen3-1.7b", "y")]:
        k = cache_key(model, "p", max_tokens=8, temperature=0.0, json_mode=False)
        a.set(CacheRow(k, model, prompt_sha256("p"), 8, 0.0, False, text))
    assert a.export(tmp_path / "b.sqlite", model="qwen3-4b") == 1
    b = CompletionCache(tmp_path / "b.sqlite")
    assert b.stats()["rows"] == 1
    assert b.merge(tmp_path / "a.sqlite") == 1      # only the 1.7b row is new
    assert b.merge(tmp_path / "a.sqlite") == 0

def test_default_path_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("PUREMACRO_LLM_CACHE", str(tmp_path / "x.sqlite"))
    assert default_cache_path() == tmp_path / "x.sqlite"

def test_unwritable_path_degrades_to_no_cache(tmp_path, recwarn):
    ro = tmp_path / "ro"; ro.mkdir(); os.chmod(ro, stat.S_IRUSR | stat.S_IXUSR)
    if os.access(ro, os.W_OK):      # root, or Windows: directory permissions are not enforced
        os.chmod(ro, stat.S_IRWXU)
        pytest.skip("directory permissions do not restrict this account")
    c = CompletionCache(ro / "c.sqlite")
    assert c.writable is False
    k = cache_key("qwen3-4b", "hola", max_tokens=8, temperature=0.0, json_mode=False)
    c.set(CacheRow(k, "qwen3-4b", prompt_sha256("hola"), 8, 0.0, False, "t"))   # no raise
    assert c.get(k) is None
    assert sum("cache" in str(w.message).lower() for w in recwarn) == 1
    c.set(CacheRow(k, "qwen3-4b", prompt_sha256("hola"), 8, 0.0, False, "t"))
    assert sum("cache" in str(w.message).lower() for w in recwarn) == 1          # warned once
    os.chmod(ro, stat.S_IRWXU)

def test_cache_miss_carries_key_and_model():
    e = CacheMiss("k", "qwen3-4b")
    assert e.key == "k" and e.model == "qwen3-4b"


# ---------------------------------------------------------------------------
# Beyond the eight tests of the brief: behaviour that spec section 3.4 states
# (key format, export/merge, degrade-to-no-cache) or that the code decides.
# ---------------------------------------------------------------------------

def _k(prompt, model="qwen3-4b"):
    return cache_key(model, prompt, max_tokens=8, temperature=0.0, json_mode=False)


def _row(key, text="t", model="qwen3-4b"):
    return CacheRow(key, model, prompt_sha256("p"), 8, 0.0, False, text)


def _sql(path, statement, args=()):
    """Run one statement against a cache file with a throw-away connection."""
    with contextlib.closing(sqlite3.connect(path)) as conn:
        rows = conn.execute(statement, args).fetchall()
        conn.commit()
    return rows


def _n_cache_warnings(recwarn):
    return sum("cache" in str(w.message).lower() for w in recwarn)


def _reference_key(model, prompt, system, max_tokens, temperature, json_mode):
    """Spec 3.4 written out literally, independent of the module under test."""
    p = hashlib.sha256(((system or "") + "\x1f" + prompt).encode("utf-8")).hexdigest()
    s = "|".join(["v1", model, p, str(int(max_tokens)),
                  f"{float(temperature):.3f}", "1" if json_mode else "0"])
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("model, prompt, system, max_tokens, temperature, json_mode", [
    ("qwen3-4b", "hola", None, 8, 0.0, False),
    ("qwen3-1.7b", "¿Qué es el PIB?", "Eres un asistente.", 256, 0.7, True),
    ("qwen3-8b", "línea 1\nlínea 2 — 日本語", "", 1, 1.0, False),
    ("qwen3:4b", "x|y", "a|b", 512, 0.0004, True),
])
def test_key_follows_the_specification_formula(model, prompt, system, max_tokens, temperature, json_mode):
    got = cache_key(model, prompt, system=system, max_tokens=max_tokens,
                    temperature=temperature, json_mode=json_mode)
    assert got == _reference_key(model, prompt, system, max_tokens, temperature, json_mode)
    assert prompt_sha256(prompt, system) == hashlib.sha256(
        ((system or "") + "\x1f" + prompt).encode("utf-8")).hexdigest()


def test_key_depends_on_every_input_and_rounds_temperature_to_three_decimals():
    base = dict(system=None, max_tokens=8, temperature=0.0, json_mode=False)
    k0 = cache_key("qwen3-4b", "hola", **base)
    assert cache_key("qwen3-4b", "hola", **{**base, "system": ""}) == k0           # None is ""
    assert cache_key("qwen3-4b", "hola", **{**base, "temperature": 0.0004}) == k0  # 3 decimals
    for change in ({"system": "S"}, {"temperature": 0.001}, {"max_tokens": 9}, {"json_mode": True}):
        assert cache_key("qwen3-4b", "hola", **{**base, **change}) != k0, change
    assert cache_key("qwen3-1.7b", "hola", **base) != k0
    assert cache_key("qwen3-4b", "hol", **base) != k0
    # system and prompt are separated, not just concatenated
    assert cache_key("m", "b", **{**base, "system": "a"}) != cache_key("m", "ab", **base)


def test_default_path_without_env_is_under_the_home_cache_directory(monkeypatch):
    expected = Path.home() / ".cache" / "puremacro" / "llm_cache.sqlite"
    monkeypatch.delenv("PUREMACRO_LLM_CACHE", raising=False)
    assert default_cache_path() == expected
    monkeypatch.setenv("PUREMACRO_LLM_CACHE", "")
    assert default_cache_path() == expected


def test_env_path_expands_the_home_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("PUREMACRO_LLM_CACHE", "~/llm.sqlite")
    assert default_cache_path() == tmp_path / "llm.sqlite"


def test_cache_without_arguments_uses_the_default_path(monkeypatch, tmp_path):
    monkeypatch.setenv("PUREMACRO_LLM_CACHE", str(tmp_path / "sub" / "env.sqlite"))
    c = CompletionCache()
    assert c.path == tmp_path / "sub" / "env.sqlite"
    assert c.writable is True and c.path.is_file()       # parent directory created too


def test_relative_path_is_anchored_when_the_cache_is_created(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    c = CompletionCache("rel.sqlite")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)                          # a notebook that calls os.chdir
    c.set(_row(_k("p")))
    assert c.path.is_absolute() and c.path.resolve() == (tmp_path / "rel.sqlite").resolve()
    assert c.get(_k("p")) == "t" and not (elsewhere / "rel.sqlite").exists()


def test_paths_with_uri_special_characters_read_back_what_was_written(tmp_path):
    """Reads open the file through a SQLite URI and writes through the plain path: a
    name like `c#1 %41+é.sqlite` must mean the same file to both (`%41` is "A" if
    the URI were not percent-encoded)."""
    c = CompletionCache(tmp_path / "my cache" / "c#1 %41+é&'.sqlite")
    c.set(_row(_k("p"), text="ok"))
    assert c.get(_k("p")) == "ok" and c.stats()["rows"] == 1
    assert c.export(tmp_path / "out dir" / "x%20y.sqlite") == 1
    assert CompletionCache(tmp_path / "out dir" / "x%20y.sqlite").get(_k("p")) == "ok"
    assert sorted(p.name for p in (tmp_path / "my cache").iterdir()) == ["c#1 %41+é&'.sqlite"]


def test_stored_columns_and_types(tmp_path):
    c = CompletionCache(tmp_path / "c.sqlite")
    c.set(CacheRow("k1", "qwen3-4b", "sha", 8, 0.5, True, "ok", "mlx", "mlx-community/Qwen3-4B-4bit"))
    c.set(CacheRow("k2", "qwen3-4b", "sha", 8, 0.0, False, "no provenance"))
    rows = _sql(c.path, "SELECT key, model, prompt_sha256, max_tokens, temperature, json_mode, "
                        "completion, engine, engine_model_id, typeof(created_at) FROM completions ORDER BY key")
    assert rows == [
        ("k1", "qwen3-4b", "sha", 8, 0.5, 1, "ok", "mlx", "mlx-community/Qwen3-4B-4bit", "text"),
        ("k2", "qwen3-4b", "sha", 8, 0.0, 0, "no provenance", None, None, "text"),
    ]
    assert _sql(c.path, "SELECT name FROM sqlite_master WHERE type = 'index' AND name = 'completions_model'")


def test_set_keeps_the_first_completion_for_a_key(tmp_path):
    c = CompletionCache(tmp_path / "c.sqlite")
    c.set(_row(_k("p"), text="first"))
    c.set(_row(_k("p"), text="second"))
    assert c.get(_k("p")) == "first" and c.stats()["rows"] == 1


def test_stats_groups_by_model(tmp_path):
    c = CompletionCache(tmp_path / "c.sqlite")
    for i in range(3):
        c.set(_row(_k(f"p{i}"), model="qwen3-4b"))
    c.set(_row(_k("p0", "qwen3-1.7b"), model="qwen3-1.7b"))
    s = c.stats()
    assert s["rows"] == 4 and s["by_model"] == {"qwen3-1.7b": 1, "qwen3-4b": 3}
    assert s["path"] == str(c.path) and s["bytes"] == c.path.stat().st_size


def test_reads_do_not_recreate_a_deleted_file_but_writes_do(tmp_path):
    c = CompletionCache(tmp_path / "c.sqlite")
    c.path.unlink()
    assert c.get(_k("p")) is None
    assert c.stats() == {"rows": 0, "by_model": {}, "bytes": 0, "path": str(c.path)}
    assert not c.path.exists()
    c.set(_row(_k("p"), text="back"))
    assert c.get(_k("p")) == "back"


def test_export_and_merge_preserve_every_column_including_created_at(tmp_path):
    a = CompletionCache(tmp_path / "a.sqlite")
    a.set(CacheRow(_k("p"), "qwen3-4b", prompt_sha256("p"), 8, 0.0, True, "ok", "mlx", "mlx-community/Qwen3-4B-4bit"))
    _sql(a.path, "UPDATE completions SET created_at = '2026-02-01 10:00:00'")
    cols = ("key, model, prompt_sha256, max_tokens, temperature, json_mode, completion, "
            "engine, engine_model_id, created_at")
    assert a.export(tmp_path / "b.sqlite") == 1
    assert _sql(tmp_path / "b.sqlite", f"SELECT {cols} FROM completions") == _sql(a.path, f"SELECT {cols} FROM completions")
    c = CompletionCache(tmp_path / "c.sqlite")
    assert c.merge(tmp_path / "b.sqlite") == 1
    assert _sql(c.path, f"SELECT {cols} FROM completions") == _sql(a.path, f"SELECT {cols} FROM completions")


def test_export_since_compares_created_at_as_iso_text(tmp_path):
    a = CompletionCache(tmp_path / "a.sqlite")
    for prompt, stamp in [("old", "2026-01-31 23:59:59"), ("edge", "2026-02-01 00:00:00"),
                          ("new", "2026-03-15 12:00:00")]:
        a.set(_row(_k(prompt), text=prompt))
        _sql(a.path, "UPDATE completions SET created_at = ? WHERE key = ?", (stamp, _k(prompt)))
    assert a.export(tmp_path / "d1.sqlite", since="2026-02-01") == 2                  # a date
    assert a.export(tmp_path / "d2.sqlite", since="2026-02-01 00:00:00") == 2         # inclusive
    assert a.export(tmp_path / "d3.sqlite", since="2026-02-01T00:00:00") == 2         # ISO 'T'
    assert a.export(tmp_path / "d4.sqlite", since="2026-02-01 00:00:01") == 1
    assert a.export(tmp_path / "d5.sqlite", since="2026-03-16") == 0
    assert a.export(tmp_path / "d6.sqlite", since="2026-02-01", model="qwen3-1.7b") == 0
    assert a.export(tmp_path / "d7.sqlite", since="2026-02-01", model="qwen3-4b") == 2
    assert sorted(r[0] for r in _sql(tmp_path / "d1.sqlite", "SELECT completion FROM completions")) == ["edge", "new"]


def test_export_creates_missing_directories_and_an_empty_result_is_a_valid_cache(tmp_path):
    a = CompletionCache(tmp_path / "a.sqlite")
    a.set(_row(_k("p")))
    dest = tmp_path / "ship" / "course" / "llm_cache.sqlite"
    assert a.export(dest, model="no-such-model") == 0
    assert CompletionCache(dest).stats()["rows"] == 0
    assert a.export(dest) == 1 and CompletionCache(dest).get(_k("p")) == "t"


def test_export_to_itself_inserts_nothing(tmp_path):
    a = CompletionCache(tmp_path / "a.sqlite")
    a.set(_row(_k("p")))
    assert a.export(a.path) == 0 and a.merge(a.path) == 0
    assert a.stats()["rows"] == 1


def test_merge_and_export_from_a_missing_file_raise_and_create_nothing(tmp_path):
    a = CompletionCache(tmp_path / "a.sqlite")
    with pytest.raises(FileNotFoundError):
        a.merge(tmp_path / "nope.sqlite")
    assert not (tmp_path / "nope.sqlite").exists()
    a.path.unlink()
    with pytest.raises(FileNotFoundError):
        a.export(tmp_path / "out.sqlite")
    assert not (tmp_path / "out.sqlite").exists()


def test_corrupt_cache_file_degrades_to_no_cache(tmp_path, recwarn):
    p = tmp_path / "c.sqlite"
    p.write_bytes(b"this is not a sqlite database " * 40)
    c = CompletionCache(p)
    assert c.writable is False and c.get(_k("p")) is None
    assert c.stats()["rows"] == 0
    c.set(_row(_k("p")))                                  # no raise
    c.set(_row(_k("q")))
    assert _n_cache_warnings(recwarn) == 1


def test_missing_sqlite3_module_degrades_to_no_cache(tmp_path, monkeypatch, recwarn):
    """Pyodide 0.28 ships sqlite3 as a load-on-request package: the placeholder from
    puremacro._optional_stdlib raises ImportError on any attribute access."""
    import puremacro.llm.cache as cache_module

    class NoSqlite:
        def __getattr__(self, name):
            raise ImportError("the standard-library module 'sqlite3' is not available")

    monkeypatch.setattr(cache_module, "sqlite3", NoSqlite())
    c = CompletionCache(tmp_path / "c.sqlite")
    assert c.writable is False
    assert c.get(_k("p")) is None and c.stats()["rows"] == 0
    c.set(_row(_k("p")))
    c.set(_row(_k("q")))
    assert _n_cache_warnings(recwarn) == 1
    assert not (tmp_path / "c.sqlite").exists()


def test_existing_cache_in_a_read_only_directory_still_serves_hits(tmp_path, recwarn):
    shipped = tmp_path / "shipped"
    CompletionCache(shipped / "c.sqlite").set(_row(_k("hola"), text="¡Hola!"))
    os.chmod(shipped, stat.S_IRUSR | stat.S_IXUSR)
    try:
        if os.access(shipped, os.W_OK):
            pytest.skip("directory permissions do not restrict this account")
        c = CompletionCache(shipped / "c.sqlite")
        assert c.get(_k("hola")) == "¡Hola!" and c.stats()["rows"] == 1
        c.set(_row(_k("nuevo")))                          # the journal cannot be created: no raise
        assert c.writable is False and c.get(_k("nuevo")) is None
        assert c.get(_k("hola")) == "¡Hola!"
        assert _n_cache_warnings(recwarn) == 1
    finally:
        os.chmod(shipped, stat.S_IRWXU)


def test_one_warning_per_unusable_cache_file_not_per_object(tmp_path, recwarn):
    p = tmp_path / "bad.sqlite"
    p.write_bytes(b"garbage" * 100)
    CompletionCache(p).set(_row(_k("p")))
    CompletionCache(p).set(_row(_k("p")))                 # a new object for the same file
    assert _n_cache_warnings(recwarn) == 1
    q = tmp_path / "bad2.sqlite"
    q.write_bytes(b"garbage" * 100)
    CompletionCache(q).set(_row(_k("p")))                 # a different file warns again
    assert _n_cache_warnings(recwarn) == 2


def test_warning_names_the_file_and_the_remedy(tmp_path):
    p = tmp_path / "bad.sqlite"
    p.write_bytes(b"garbage" * 100)
    with pytest.warns(UserWarning, match="PUREMACRO_LLM_CACHE") as w:
        CompletionCache(p).set(_row(_k("p")))
    assert str(p) in str(w[0].message)


def test_every_method_closes_the_connection_it_opened(tmp_path, monkeypatch, recwarn):
    """One connection per call, closed in `finally`: nothing stays open, on the
    success paths and on the failure paths alike."""
    import puremacro.llm.cache as cache_module

    opened = []

    class Spy:
        def __getattr__(self, name):
            if name != "connect":
                return getattr(sqlite3, name)

            def connect(*args, **kwargs):
                conn = sqlite3.connect(*args, **kwargs)
                opened.append(conn)
                return conn
            return connect

    monkeypatch.setattr(cache_module, "sqlite3", Spy())
    a = CompletionCache(tmp_path / "a.sqlite")
    a.set(_row(_k("p")))
    a.get(_k("p"))
    a.get(_k("absent"))
    a.stats()
    a.export(tmp_path / "b.sqlite")
    CompletionCache(tmp_path / "b.sqlite").merge(a.path)
    with pytest.raises(sqlite3.Error):                    # the source has no `completions` table
        a.merge(_empty_database(tmp_path / "empty.sqlite"))
    bad = tmp_path / "bad.sqlite"
    bad.write_bytes(b"garbage" * 100)
    broken = CompletionCache(bad)                         # fails while preparing the schema
    broken.get(_k("p"))
    broken.stats()
    broken.set(_row(_k("p")))
    assert len(opened) >= 10
    for conn in opened:
        with pytest.raises(sqlite3.ProgrammingError):     # "Cannot operate on a closed database"
            conn.execute("SELECT 1")


def _empty_database(path):
    _sql(path, "CREATE TABLE unrelated (x)")
    return path


def test_cache_miss_is_a_lookup_error_that_survives_pickling():
    e = pickle.loads(pickle.dumps(CacheMiss("ab" * 32, "qwen3-4b")))
    assert isinstance(e, LookupError)
    assert (e.key, e.model) == ("ab" * 32, "qwen3-4b")
    assert "qwen3-4b" in str(e) and "abababababab" in str(e)


def test_cache_row_is_frozen_and_positional_order_is_fixed():
    r = CacheRow("k", "m", "sha", 8, 0.0, False, "text", "mlx", "id")
    assert (r.key, r.model, r.prompt_sha256, r.max_tokens, r.temperature, r.json_mode,
            r.completion, r.engine, r.engine_model_id) == ("k", "m", "sha", 8, 0.0, False, "text", "mlx", "id")
    assert CacheRow("k", "m", "sha", 8, 0.0, False, "text").engine is None
    with pytest.raises(AttributeError):
        r.completion = "changed"
