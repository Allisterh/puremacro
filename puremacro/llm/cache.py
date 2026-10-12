"""Completion cache for :mod:`puremacro.llm`.

One SQLite file maps ``(model, prompt, system prompt, sampling parameters)`` to
the text a local model returned, so a completion computed once, on any machine,
answers the same call everywhere: a cache built on a laptop can be shipped with
a course and read by a tablet that cannot run the model.

* **Location.** ``~/.cache/puremacro/llm_cache.sqlite``, or the file named by the
  environment variable ``PUREMACRO_LLM_CACHE``. The narrative score cache of
  ``llm_prob_kernel`` is another cache in another file, named by
  ``PUREMACRO_LLM_SCORES_CACHE``; this variable never moves it.
* **Other tables are not ours.** The file is an ordinary SQLite file and may hold
  tables that belong to other code. This module reads and writes ``completions``
  (and its index) and nothing else: it creates its table beside whatever is
  there, never alters or drops another table, and :meth:`CompletionCache.export`
  and :meth:`CompletionCache.merge` copy ``completions`` rows only, so exporting
  a shared file ships the completions alone. ``stats()["bytes"]`` is the size of
  the whole file.
* **Key (format ``v1``).** The model is the *canonical alias* (``qwen3-4b``),
  never an engine-specific id, and the prompt is hashed as the caller wrote it,
  *before* any chat template, so the engine that produced a row does not matter.
  The engine and its model id are stored for provenance only. The key function
  is fixed for the life of the format: rows written by one release stay
  readable by every later one, and the golden values pinned in the test suite
  keep any other implementation of the algorithm compatible with this file.
* **Never an error.** A cache is an optimisation. A file or directory that
  cannot be written, a corrupt file, or a Python without ``sqlite3`` (in
  Pyodide 0.28, run ``await pyodide.loadPackage('sqlite3')`` before importing
  puremacro) turns the cache into a no-op: lookups find nothing, writes are
  skipped, and one warning per cache file says why. :meth:`CompletionCache.export`
  and :meth:`CompletionCache.merge` are explicit actions, so they raise instead.
  :attr:`CompletionCache.readable` tells a lookup that could not read the file
  from one that found no row, and :attr:`CompletionCache.reason` says why, so
  that ``cache`` mode in :mod:`puremacro.llm` raises :class:`CacheUnavailable`,
  not a plain :class:`CacheMiss`, for the first.
* **No shared state.** Every method opens its own connection and closes it
  before returning; nothing is kept at module level except the record of which
  files were already reported as unusable. A write that finds the file locked by
  another connection waits (five seconds, the default of :func:`sqlite3.connect`)
  and is then skipped like any other failed write, so code that shares the file
  must not keep a write transaction open on it while a completion is being made.
"""
from __future__ import annotations

import contextlib
import hashlib
import os
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from puremacro._optional_stdlib import sqlite3

__all__ = [
    "CACHE_FORMAT",
    "CacheMiss",
    "CacheRow",
    "CacheUnavailable",
    "CompletionCache",
    "cache_key",
    "default_cache_path",
    "prompt_sha256",
]

CACHE_FORMAT = "v1"
"""Version of the key function and of the table layout; it is part of every key."""

_ENV_PATH = "PUREMACRO_LLM_CACHE"

_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS completions (
  key            TEXT PRIMARY KEY,
  model          TEXT NOT NULL,    -- canonical alias
  prompt_sha256  TEXT NOT NULL,
  max_tokens     INTEGER NOT NULL,
  temperature    REAL NOT NULL,
  json_mode      INTEGER NOT NULL,
  completion     TEXT NOT NULL,
  engine         TEXT,             -- provenance only
  engine_model_id TEXT,            -- provenance only
  created_at     TEXT DEFAULT CURRENT_TIMESTAMP
)"""
_CREATE_INDEX = "CREATE INDEX IF NOT EXISTS completions_model ON completions(model)"

_COLUMNS = ("key, model, prompt_sha256, max_tokens, temperature, json_mode, "
            "completion, engine, engine_model_id, created_at")

# Cache files already reported as unusable, so each is reported once per process
# however many CompletionCache objects (the facade makes one per call) point at it.
_WARNED: set[str] = set()


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------

def prompt_sha256(prompt: str, system: str | None = None) -> str:
    """SHA-256 (hex) of the system prompt and the prompt, as the caller wrote them.

    The two are joined by the unit separator ``"\\x1f"``; a missing system
    prompt counts as the empty string. Nothing is templated first: templates
    differ between engines, and hashing after templating would defeat
    cross-engine cache hits.
    """
    return hashlib.sha256(((system or "") + "\x1f" + prompt).encode("utf-8")).hexdigest()


def cache_key(model: str, prompt: str, *, system: str | None = None,
              max_tokens: int, temperature: float, json_mode: bool) -> str:
    """Cache key (SHA-256, hex) of one completion request, format ``v1``.

    ``model`` is the canonical alias (``"qwen3-4b"``), not an engine-specific
    id. ``temperature`` is rounded to three decimals, ``max_tokens`` truncated
    to an integer, ``json_mode`` reduced to 0 or 1. The engine is not part of
    the key: the same request gives the same key on every engine.
    """
    fields = [CACHE_FORMAT, model, prompt_sha256(prompt, system), str(int(max_tokens)),
              f"{float(temperature):.3f}", "1" if json_mode else "0"]
    return hashlib.sha256("|".join(fields).encode("utf-8")).hexdigest()


def default_cache_path() -> Path:
    """Where the cache lives: ``$PUREMACRO_LLM_CACHE``, else the user cache directory.

    The default is ``~/.cache/puremacro/llm_cache.sqlite``.
    """
    env = os.environ.get(_ENV_PATH)
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "puremacro" / "llm_cache.sqlite"


# ---------------------------------------------------------------------------
# Rows and errors
# ---------------------------------------------------------------------------

class CacheMiss(LookupError):
    """No cached completion for a request made in ``cache`` mode.

    Attributes
    ----------
    key : str
        The cache key that was looked up.
    model : str
        The canonical model alias of the request.
    """

    def __init__(self, key: str, model: str) -> None:
        super().__init__(key, model)        # args round-trip through pickle
        self.key = key
        self.model = model

    def __str__(self) -> str:
        return (f"no cached completion for model {self.model!r} (key {self.key[:12]}...); "
                "run it once with a local model, or load a cache that has it")


class CacheUnavailable(CacheMiss):
    """A ``cache``-mode request whose lookup could not read the cache at all.

    Raised by :func:`puremacro.llm.complete` in ``cache`` mode instead of a plain
    :class:`CacheMiss` when :attr:`CompletionCache.readable` is ``False``: no
    ``sqlite3`` (Pyodide 0.28 without ``await pyodide.loadPackage('sqlite3')``
    run before puremacro was imported), or a file that exists but cannot be
    opened as a cache, such as a Git LFS pointer. Being a :class:`CacheMiss`,
    it is caught wherever a miss is.

    Attributes
    ----------
    key : str
        The cache key that was looked up.
    model : str
        The canonical model alias of the request.
    reason : str or None
        Why the cache cannot be read: :attr:`CompletionCache.reason`, which for a
        missing ``sqlite3`` carries the hint on loading it.
    """

    def __init__(self, key: str, model: str, reason: str | None) -> None:
        super().__init__(key, model)
        self.args = (key, model, reason)    # all three round-trip through pickle
        self.reason = reason

    def __str__(self) -> str:
        return (f"the completion cache cannot be read here, so the completion for model "
                f"{self.model!r} (key {self.key[:12]}...) could not be looked up: {self.reason}")


@dataclass(frozen=True)
class CacheRow:
    """One cached completion, in the order of the table's columns.

    ``engine`` and ``engine_model_id`` record which engine produced the text and
    under which id; they are provenance only and never part of the key.
    """

    key: str
    model: str
    prompt_sha256: str
    max_tokens: int
    temperature: float
    json_mode: bool
    completion: str
    engine: str | None = None
    engine_model_id: str | None = None


# ---------------------------------------------------------------------------
# SQLite plumbing
# ---------------------------------------------------------------------------

def _unusable() -> tuple[type[BaseException], ...]:
    """The exceptions that mean "this cache cannot be used here", not "this code is wrong".

    ``sqlite3`` may be the placeholder of :mod:`puremacro._optional_stdlib`,
    whose every attribute, ``Error`` included, raises ImportError, so the
    exception classes are looked up defensively.
    """
    try:
        return (sqlite3.Error, OSError, ImportError)
    except ImportError:
        return (OSError, ImportError)


@contextlib.contextmanager
def _connect(path: Path, *, write: bool) -> Iterator[Any]:
    """One connection for one operation, closed whatever happens.

    ``write=True`` opens read-write and first makes the parent directory, the file
    and the table, all idempotently. ``write=False`` opens read-only (SQLite URI
    ``mode=ro``): it never creates or modifies the file, it works in a directory
    the process cannot write, and a missing file raises ``OperationalError``.
    """
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
    else:
        conn = sqlite3.connect(path.absolute().as_uri() + "?mode=ro", uri=True)
    try:
        if write:
            conn.execute(_CREATE_TABLE)
            conn.execute(_CREATE_INDEX)
            conn.commit()
        yield conn
    finally:
        conn.close()


def _utc_text(moment: str) -> str:
    """An ISO-8601 date or date-time as ``created_at`` is stored: ``YYYY-MM-DD[ HH:MM:SS]``.

    SQLite writes ``CURRENT_TIMESTAMP`` (UTC) with a space between date and time,
    and the comparison is on text, so a ``T`` separator would sort after every
    time of that day and silently drop it.
    """
    text = moment.strip()
    if len(text) > 10 and text[10] == "T":
        return text[:10] + " " + text[11:]
    return text


def _copy_rows(src: Path, dst: Path, *, model: str | None = None, since: str | None = None) -> int:
    """Copy rows of the cache file ``src`` into ``dst``, creating ``dst``; return the number added.

    ``INSERT OR IGNORE``: a key ``dst`` already has is left as it is.
    """
    if not src.is_file():
        raise FileNotFoundError(f"no cache file at {src}")
    if dst.exists() and os.path.samefile(src, dst):
        return 0                                  # attaching a file to itself deadlocks on commit
    conditions: list[str] = []
    params: list[str] = []
    if model is not None:
        conditions.append("model = ?")
        params.append(model)
    if since is not None:
        conditions.append("created_at >= ?")
        params.append(_utc_text(since))
    where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
    with _connect(dst, write=True) as conn:
        conn.execute("ATTACH DATABASE ? AS src", (str(src),))
        cur = conn.execute(
            f"INSERT OR IGNORE INTO main.completions ({_COLUMNS}) "
            f"SELECT {_COLUMNS} FROM src.completions{where}", params)
        added = cur.rowcount
        conn.commit()
    return int(added)


# ---------------------------------------------------------------------------
# The cache
# ---------------------------------------------------------------------------

class CompletionCache:
    """A file of cached completions (see the module docstring).

    Parameters
    ----------
    path : str or Path, optional
        The SQLite file. Defaults to :func:`default_cache_path`. The parent
        directory, the file and the table are created if they do not exist; a
        relative path is anchored at the current directory once, here.

    Attributes
    ----------
    path : Path
        Absolute path of the cache file.
    writable : bool
        Whether writes are expected to work: ``False`` when the file could not
        be prepared (unwritable directory, corrupt file, no ``sqlite3``), and
        set to ``False`` by the first write that fails. Lookups still work on an
        existing, readable file, so a cache shipped on a read-only volume serves hits.
    readable : bool
        Whether a lookup can read the file (read-only property): ``False`` when
        ``sqlite3`` is unavailable or the file exists but cannot be opened as a
        cache; ``True`` otherwise, also when the file does not exist yet.
    reason : str or None
        Why the cache cannot be used, or ``None`` when it can be read and
        written (read-only property).
    """

    def __init__(self, path: str | Path | None = None) -> None:
        chosen = default_cache_path() if path is None else Path(path).expanduser()
        self.path: Path = chosen.absolute()
        self._reason = ""            # why the file could not be prepared, or a write failed
        self._read_error = ""        # why the last lookup could not read the file
        self.writable: bool = self._prepare()

    def __repr__(self) -> str:
        return f"CompletionCache({str(self.path)!r}, writable={self.writable})"

    @property
    def readable(self) -> bool:
        """Whether a lookup can read the cache.

        ``False`` when ``sqlite3`` is unavailable (in Pyodide 0.28, unless
        ``await pyodide.loadPackage('sqlite3')`` ran before puremacro was
        imported) or when the file exists but cannot be opened as a cache: not
        an SQLite file (a Git LFS pointer, a corrupt file), no ``completions``
        table, no permission to read it. ``True`` otherwise, also when the file
        does not exist yet: a lookup then finds no row. Found when the cache is
        created and again by every :meth:`get`, so after a lookup that returned
        ``None`` it says whether the row is absent (``True``) or the lookup
        could not be made (``False``).
        """
        return not self._read_error

    @property
    def reason(self) -> str | None:
        """Why the cache cannot be used, or ``None`` when it can be read and written.

        The error of the last lookup, when it could not read the file; else the
        error that stopped the cache from writing (see ``writable``).
        """
        return self._read_error or self._reason or None

    def _prepare(self) -> bool:
        try:
            with _connect(self.path, write=True):
                pass
        except _unusable() as exc:
            self._reason = f"{type(exc).__name__}: {exc}"
            self.get("")        # can lookups still read it? A cache on a read-only volume can.
            return False
        return True

    def _warn_once(self) -> None:
        name = str(self.path)
        if name in _WARNED:
            return
        _WARNED.add(name)
        warnings.warn(
            f"puremacro.llm: the completion cache at {name} cannot be written ({self._reason}); "
            "completions will be recomputed instead of cached. Point PUREMACRO_LLM_CACHE at "
            "a writable file to keep them.",
            UserWarning, stacklevel=3)

    def get(self, key: str) -> str | None:
        """The cached completion for ``key``, or ``None`` (also when the cache is unusable).

        Sets :attr:`readable`: ``False`` when the lookup could not read the file,
        with the error in :attr:`reason`; ``True`` when it could, or found no file.
        """
        try:
            with _connect(self.path, write=False) as conn:
                row = conn.execute(
                    "SELECT completion FROM completions WHERE key = ?", (key,)).fetchone()
        except _unusable() as exc:
            # No file is an empty cache: the lookup just finds nothing. Anything else (no
            # sqlite3, a file SQLite cannot read as a cache) means no lookup was made.
            no_file = not isinstance(exc, ImportError) and not os.path.exists(self.path)
            self._read_error = "" if no_file else f"{type(exc).__name__}: {exc}"
            return None
        self._read_error = ""
        return None if row is None else row[0]

    def set(self, row: CacheRow) -> None:
        """Store ``row``; the first completion stored for a key is kept (``INSERT OR IGNORE``).

        Never raises because of the cache: when the write cannot be done, the
        cache stops writing (``writable`` becomes ``False``) and one warning per
        cache file says why.
        """
        if not self.writable:
            self._warn_once()
            return
        try:
            with _connect(self.path, write=True) as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO completions "
                    "(key, model, prompt_sha256, max_tokens, temperature, json_mode, "
                    "completion, engine, engine_model_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (row.key, row.model, row.prompt_sha256, int(row.max_tokens),
                     float(row.temperature), 1 if row.json_mode else 0, row.completion,
                     row.engine, row.engine_model_id))
                conn.commit()
        except _unusable() as exc:
            self.writable = False
            self._reason = f"{type(exc).__name__}: {exc}"
            self._warn_once()

    def export(self, dest: str | Path, *, model: str | None = None,
               since: str | None = None) -> int:
        """Copy rows into the cache file ``dest``, creating it; return the number added.

        Parameters
        ----------
        dest : str or Path
            The target cache file; its directory and table are created if needed.
            Rows whose key it already has are left as they are.
        model : str, optional
            Only rows of this canonical model alias.
        since : str, optional
            Only rows with ``created_at >= since``, compared as ISO-8601 text in
            UTC (``"2026-10-01"`` or ``"2026-10-01 08:00:00"``; a ``T`` between
            date and time is accepted).

        Raises
        ------
        FileNotFoundError
            This cache has no file yet, so there is nothing to export.
        sqlite3.Error, OSError
            ``dest`` cannot be written; an export that did not happen is not hidden.
        """
        return _copy_rows(self.path, Path(dest).expanduser(), model=model, since=since)

    def merge(self, src: str | Path) -> int:
        """Add the rows of the cache file ``src`` that this cache lacks; return the number added.

        Rows whose key is already here are left as they are.

        Raises
        ------
        FileNotFoundError
            ``src`` does not exist (it is not created).
        sqlite3.Error, OSError
            This cache cannot be written, or ``src`` is not a completion cache.
        """
        return _copy_rows(Path(src).expanduser(), self.path)

    def stats(self) -> dict[str, Any]:
        """Row counts and file size; an unusable cache reports zero rows.

        Returns
        -------
        dict
            ``rows`` (int), ``by_model`` (canonical alias -> row count, sorted by
            alias; ``completions`` only), ``bytes`` (size of the whole file,
            whatever else it holds; 0 if absent) and ``path`` (str).
        """
        by_model: dict[str, int] = {}
        try:
            with _connect(self.path, write=False) as conn:
                counts = conn.execute(
                    "SELECT model, COUNT(*) FROM completions GROUP BY model ORDER BY model")
                by_model = dict(counts.fetchall())
        except _unusable():
            pass
        try:
            size = self.path.stat().st_size
        except OSError:
            size = 0
        return {"rows": sum(by_model.values()), "by_model": by_model,
                "bytes": size, "path": str(self.path)}
