"""F2.1 — migrate_from_flat_files: idempotent + remove flag + warn-and-skip."""
from __future__ import annotations

import json
import time

import pytest


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(tmp_path))
    import puremacro._cache_db as M
    M.close_conn()
    yield tmp_path
    M.close_conn()


def _make_flat_entry(d, url: str, body: bytes, age_seconds: int = 0):
    """Create a flat-file cache pair in `d`."""
    import hashlib
    key = hashlib.sha256(url.encode()).hexdigest()
    (d / f"{key}.bin").write_bytes(body)
    (d / f"{key}.json").write_text(json.dumps({
        "url": url,
        "fetched_at": time.time() - age_seconds,
        "content_type": "text/plain",
    }), encoding="utf-8")


def test_migration_inserts_entries(fresh_db):
    from puremacro._cache_db import get_conn, migrate_from_flat_files
    _make_flat_entry(fresh_db, "https://example.com/a", b"alpha")
    _make_flat_entry(fresh_db, "https://example.com/b", b"beta")
    conn = get_conn(fresh_db / "cache.db")
    n = migrate_from_flat_files(conn, fresh_db, remove=False)
    assert n == 2
    rows = conn.execute("SELECT url, body FROM http_cache ORDER BY url").fetchall()
    assert rows == [("https://example.com/a", b"alpha"),
                    ("https://example.com/b", b"beta")]


def test_migration_idempotent(fresh_db):
    from puremacro._cache_db import get_conn, migrate_from_flat_files
    _make_flat_entry(fresh_db, "https://example.com/a", b"alpha")
    conn = get_conn(fresh_db / "cache.db")
    assert migrate_from_flat_files(conn, fresh_db, remove=False) == 1
    assert migrate_from_flat_files(conn, fresh_db, remove=False) == 0


def test_migration_with_remove_unlinks_files(fresh_db):
    from puremacro._cache_db import get_conn, migrate_from_flat_files
    _make_flat_entry(fresh_db, "https://example.com/a", b"alpha")
    conn = get_conn(fresh_db / "cache.db")
    assert migrate_from_flat_files(conn, fresh_db, remove=True) == 1
    assert list(fresh_db.glob("*.bin")) == []
    assert list(fresh_db.glob("*.json")) == []


def test_migration_skips_corrupt_sidecars(fresh_db):
    import hashlib, warnings
    from puremacro._cache_db import get_conn, migrate_from_flat_files
    # Good entry + corrupt sidecar.
    _make_flat_entry(fresh_db, "https://example.com/good", b"ok")
    bad_url = "https://example.com/bad"
    bad_key = hashlib.sha256(bad_url.encode()).hexdigest()
    (fresh_db / f"{bad_key}.bin").write_bytes(b"corrupt")
    (fresh_db / f"{bad_key}.json").write_text("{ not valid json", encoding="utf-8")
    conn = get_conn(fresh_db / "cache.db")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        n = migrate_from_flat_files(conn, fresh_db, remove=False)
    assert n == 1
    assert any(issubclass(w.category, UserWarning) for w in caught)


def test_migration_missing_dir_returns_zero(tmp_path):
    from puremacro._cache_db import get_conn, migrate_from_flat_files
    conn = get_conn(tmp_path / "cache.db")
    nonexistent = tmp_path / "does_not_exist"
    assert migrate_from_flat_files(conn, nonexistent) == 0


def test_existing_key_keeps_original_files_when_remove_requested(fresh_db):
    import hashlib
    from puremacro._cache_db import get_conn, migrate_from_flat_files

    url = "https://example.com/preserve"
    _make_flat_entry(fresh_db, url, b"old")
    key = hashlib.sha256(url.encode()).hexdigest()
    conn = get_conn(fresh_db / "cache.db")
    conn.execute("INSERT INTO http_cache VALUES (?, ?, ?, ?, ?)",
                 (key, url, 123, "text/plain", b"newer"))
    assert migrate_from_flat_files(conn, fresh_db, remove=True) == 0
    assert (fresh_db / f"{key}.bin").read_bytes() == b"old"
    assert (fresh_db / f"{key}.json").exists()
    assert conn.execute("SELECT body FROM http_cache").fetchone() == (b"newer",)


def test_migration_commit_failure_rolls_back_and_retains_files(tmp_path):
    import sqlite3
    from puremacro._cache_db import bootstrap_schema, migrate_from_flat_files

    class CommitFailure(sqlite3.Connection):
        fail = False

        def commit(self):
            if self.fail:
                raise sqlite3.OperationalError("injected commit failure")
            return super().commit()

    conn = sqlite3.connect(tmp_path / "cache.db", factory=CommitFailure)
    bootstrap_schema(conn)
    _make_flat_entry(tmp_path, "https://example.com/a", b"saved")
    conn.fail = True
    with pytest.raises(sqlite3.OperationalError, match="commit failure"):
        migrate_from_flat_files(conn, tmp_path, remove=True)
    assert conn.execute("SELECT COUNT(*) FROM http_cache").fetchone() == (0,)
    assert len(list(tmp_path.glob("*.bin"))) == len(list(tmp_path.glob("*.json"))) == 1
    assert not conn.in_transaction
    conn.close()


def test_migration_locks_before_reading_or_counting_files(fresh_db, monkeypatch):
    import sqlite3
    from pathlib import Path
    from puremacro._cache_db import get_conn, migrate_from_flat_files

    _make_flat_entry(fresh_db, "https://example.com/a", b"body")
    conn = get_conn(fresh_db / "cache.db")
    writer = sqlite3.connect(fresh_db / "cache.db", timeout=0)
    original_read = Path.read_bytes
    attempts = []

    def read_while_writer_competes(path):
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            writer.execute("INSERT INTO http_cache VALUES (?, ?, ?, ?, ?)",
                           (path.stem, "competitor", 123, "text/plain", b"competitor"))
        writer.rollback()
        attempts.append(path)
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", read_while_writer_competes)
    assert migrate_from_flat_files(conn, fresh_db, remove=True) == 1
    assert len(attempts) == 1
    assert conn.execute("SELECT body FROM http_cache").fetchone() == (b"body",)
    # Positive control: the same competing connection writes after the lock is released.
    writer.execute("INSERT INTO http_cache VALUES (?, ?, ?, ?, ?)",
                   ("control", "control", 123, "text/plain", b"control"))
    writer.commit()
    writer.close()


def test_migration_large_batch_is_idempotent_and_transactional(fresh_db):
    from puremacro._cache_db import get_conn, migrate_from_flat_files

    for i in range(260):
        _make_flat_entry(fresh_db, f"https://example.com/{i}", str(i).encode())
    conn = get_conn(fresh_db / "cache.db")
    assert migrate_from_flat_files(conn, fresh_db) == 260
    assert migrate_from_flat_files(conn, fresh_db, remove=True) == 0
    assert len(list(fresh_db.glob("*.bin"))) == 260
    rows = conn.execute("SELECT url, body FROM http_cache").fetchall()
    assert {url: body for url, body in rows} == {
        f"https://example.com/{i}": str(i).encode() for i in range(260)
    }


@pytest.mark.mechanism_control
def test_sqlite_failure_and_lock_controls(tmp_path):
    import sqlite3
    from puremacro._cache_db import bootstrap_schema

    a = sqlite3.connect(tmp_path / "control.db")
    bootstrap_schema(a)
    b = sqlite3.connect(tmp_path / "control.db", timeout=0)
    a.execute("BEGIN IMMEDIATE")
    with pytest.raises(sqlite3.OperationalError, match="locked"):
        b.execute("INSERT INTO http_cache VALUES ('control', 'u', 0, NULL, X'01')")
    a.rollback()
    b.execute("INSERT INTO http_cache VALUES ('control', 'u', 0, NULL, X'01')")
    b.commit()
    assert a.execute("SELECT COUNT(*) FROM http_cache").fetchone() == (1,)
    a.close()
    b.close()


def test_migration_does_not_materialize_all_bodies_before_writing(fresh_db, monkeypatch):
    from pathlib import Path
    from puremacro._cache_db import get_conn, migrate_from_flat_files

    for i in range(260):
        _make_flat_entry(fresh_db, f"https://example.com/{i}", b"payload")
    conn = get_conn(fresh_db / "cache.db")
    read = Path.read_bytes
    reads = []
    reads_at_insert = []

    def tracked_read(path):
        reads.append(path)
        return read(path)

    def trace(statement):
        if statement.startswith("INSERT") and "http_cache" in statement:
            reads_at_insert.append(len(reads))

    monkeypatch.setattr(Path, "read_bytes", tracked_read)
    conn.set_trace_callback(trace)
    assert migrate_from_flat_files(conn, fresh_db) == 260
    # These checks also establish that both boundary mechanisms were exercised.
    assert len(reads_at_insert) == len(reads) == 260
    assert 0 < reads_at_insert[0] <= 128
    assert reads_at_insert[-1] == 260
