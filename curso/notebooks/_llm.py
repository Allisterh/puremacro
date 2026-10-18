"""Modelo de lenguaje local para las lecciones del curso: una llamada, con caché y sin red.

``get_llm().try_complete(pregunta)`` devuelve la respuesta del modelo, o ``None``
con una nota de una línea, y nunca detiene una lección. Las respuestas salen de
la caché del curso, ``data_curso/llm_cache.sqlite`` junto a este archivo, así que
una lección corre sin ningún modelo instalado; con un modelo local (MLX,
llama.cpp, Ollama o un servidor compatible con OpenAI) lo que la caché no tiene
se responde en vivo y se guarda. La lección T00_B explica la instalación.

Dos implementaciones, un mismo comportamiento:

* **puremacro 5.0 o posterior** (existe ``puremacro.llm``): cada llamada va a la
  librería y ``CacheMiss`` es la de la librería. Al importar este módulo,
  ``PUREMACRO_LLM_CACHE`` pasa a apuntar a la caché del curso si no estaba
  definida, así que las funciones narrativas de puremacro leen el mismo archivo.
* **puremacro 4.3 a 4.8.1** (sin ``puremacro.llm``): una réplica con los mismos
  modos, la misma escalera de modelos y el mismo formato de caché, construida
  sobre los motores de ``puremacro.narrative._local_engines``. Aquí
  ``PUREMACRO_LLM_CACHE`` no se exporta: el núcleo narrativo de 4.8.1 lee esa
  variable como la ruta de su propia caché de puntuaciones y escribiría en la
  caché del curso.

Modos (argumento ``mode``; si no, ``MAV_LLM_MODE``; si no, ``PUREMACRO_LLM_MODE``;
si no, ``auto``):

* ``auto``: lee la caché; si falta la respuesta, ejecuta el modelo local y la guarda.
* ``cache``: sólo lee la caché; no busca motores, no abre sockets, no carga modelos.
* ``off``: ejecuta siempre el modelo local; no lee ni escribe la caché.

``get_llm(model=...)`` cambia el modelo de ese objeto, y sólo de ese:
``REFERENCE_MODEL`` (``qwen3-4b``) es el que calculó las respuestas de la caché del
curso. ``get_llm(lang="en")`` imprime en inglés las notas de ese objeto; si no, van
en español.

``LLM.index(registros, categoria)`` calcula con el modelo del curso el índice de
``llm_prob_kernel``, y ``LLM.backend()`` da a ``score_llm`` de puremacro 4.8.1 un
backend; los dos responden desde esta misma caché, con las mismas claves que
usará la librería desde la 5.0. Las lecciones llaman a ``llm.index(...)``, nunca a
``llm_prob_kernel(..., provider=llm.provider())``. El núcleo guarda cada
puntuación en su propia caché (en puremacro 4.8.1, el archivo de
``PUREMACRO_LLM_CACHE`` o ``~/.cache/puremacro/llm_scores.sqlite``; desde la 5.0,
el de ``PUREMACRO_LLM_SCORES_CACHE``), también el 0.0 de una respuesta que
faltaba, y lo seguiría dando después de instalar un modelo o de recibir una
caché del curso más completa; si además las dos cachés comparten archivo, su
transacción abierta bloquea la escritura de cada respuesta nueva. ``index`` le
da una caché de puntuaciones nueva en cada llamada y la borra al terminar, así
que sólo quedan las respuestas de verdad, en la caché del curso. Con
``puremacro.llm`` mueve ``PUREMACRO_LLM_SCORES_CACHE``, porque ``PUREMACRO_LLM_CACHE``
es entonces la caché de respuestas de la librería; sin ella no sabe cuál de las
dos lee el núcleo (el de 4.8.1 lee la primera; el de una 5.x cuya ``puremacro.llm``
no se puede usar, la segunda) y mueve las dos.

El archivo de caché tiene el formato ``v1`` de ``puremacro.llm.cache`` (la misma
tabla y la misma clave), así que cada implementación lee lo que escribió la otra.
Las copias de ``notebooks/`` y ``colab_es/`` deben ser idénticas byte a byte.
"""
from __future__ import annotations

from collections.abc import Sequence
import contextlib
import hashlib
import importlib
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

__all__ = [
    "MODE_ENV",
    "MODES",
    "MISSES",
    "REFERENCE_MODEL",
    "USER_CACHE_ENV",
    "CacheMiss",
    "CourseBackend",
    "CourseProvider",
    "LLM",
    "cache_path",
    "get_llm",
    "open_cache",
    "user_cache_path",
]

MODE_ENV = "MAV_LLM_MODE"
"""Variable del curso que fija el modo; gana sobre ``PUREMACRO_LLM_MODE``."""

USER_CACHE_ENV = "MAV_LLM_USER_CACHE"
"""Variable del curso para la caché escribible del alumno delante de la caché del curso."""

MODES = ("auto", "cache", "off")

REFERENCE_MODEL = "qwen3-4b"
"""El modelo de referencia: el que calculó las respuestas de la caché del curso.

``get_llm(mode="cache", model=REFERENCE_MODEL)`` lee esas respuestas aunque
``PUREMACRO_LLM_MODEL`` elija otro modelo."""

_LANGS = ("es", "en")        # the languages of the notes an LLM prints

MISSES: list[tuple[str, str]] = []
"""Cada petición que se quedó sin respuesta, en orden: (alias del modelo, sha256 del prompt).

El sha256 es ``_prompt_sha256(prompt, system)``; con él se localiza la petición
al reconstruir la caché del curso."""

_CACHE_ENV = "PUREMACRO_LLM_CACHE"
_FALLBACK_CACHE_ENV = "PUREMACRO_LLM_FALLBACK_CACHE"
_SCORES_CACHE_ENV = "PUREMACRO_LLM_SCORES_CACHE"        # 5.0's narrative score cache
_LIBRARY_MODE_ENV = "PUREMACRO_LLM_MODE"
_MODEL_ENV = "PUREMACRO_LLM_MODEL"
_BASE_URL_ENV = "PUREMACRO_LLM_BASE_URL"


def _say(lang: str, es: str, en: str) -> str:
    """El texto de una nota en el idioma ``lang``: ``en`` si es ``"en"``; si no, ``es``."""
    return en if lang == "en" else es


# ---------------------------------------------------------------------------
# Rutas: nada de lo que se imprime lleva una ruta absoluta
# ---------------------------------------------------------------------------

def _helper_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except (OSError, RuntimeError, ValueError):     # un entorno aislado que niega realpath
        return Path(__file__).absolute().parent


_HERE = _helper_dir()
_COURSE_DIR = _HERE.parent


def cache_path() -> Path:
    """La caché del curso: ``data_curso/llm_cache.sqlite`` junto a este archivo."""
    return _HERE / "data_curso" / "llm_cache.sqlite"


def user_cache_path() -> Path:
    """La caché escribible del alumno: ``$MAV_LLM_USER_CACHE`` o ``~/.cache/puremacro/llm_cache.sqlite``."""
    env = os.environ.get(USER_CACHE_ENV, "").strip()
    if env:
        return Path(env).expanduser().absolute()
    try:
        return (Path.home() / ".cache" / "puremacro" / "llm_cache.sqlite").absolute()
    except (RuntimeError, KeyError, OSError):
        return (_HERE / "data_curso" / "llm_user_cache.sqlite").absolute()


def _shown(path) -> str:
    """``path`` tal como se puede imprimir: relativa a la carpeta del curso, o sólo su nombre."""
    p = Path(path)
    try:
        return p.resolve().relative_to(_COURSE_DIR).as_posix()
    except (ValueError, OSError, RuntimeError):
        return p.name or "?"


def _scrub(text: str, path) -> str:
    """``text`` sin las rutas absolutas conocidas: la caché, la carpeta del curso y el home."""
    p = Path(path)
    candidates = {str(p)}
    try:
        candidates.add(str(p.resolve()))
    except (OSError, RuntimeError):
        pass
    for raw in sorted(candidates, key=len, reverse=True):
        text = text.replace(raw, _shown(p))
    text = text.replace(str(_COURSE_DIR) + os.sep, "")
    try:
        home = str(Path.home())
    except (RuntimeError, KeyError, OSError):
        home = ""
    if len(home) > 1:
        text = text.replace(home, "~")
    return text


# ---------------------------------------------------------------------------
# Formato de caché v1 (el de puremacro.llm.cache, byte a byte)
# ---------------------------------------------------------------------------

def _prompt_sha256(prompt: str, system: str | None = None) -> str:
    """SHA-256 (hex) del prompt de sistema y del prompt, unidos por ``"\\x1f"``.

    Se calcula sobre el texto tal como lo escribió quien llama, antes de
    cualquier plantilla de chat, así que no depende del motor."""
    return hashlib.sha256(((system or "") + "\x1f" + prompt).encode("utf-8")).hexdigest()


def _cache_key(model: str, prompt: str, *, system: str | None = None,
               max_tokens: int, temperature: float, json_mode: bool) -> str:
    """Clave de caché del formato ``v1``, idéntica a ``puremacro.llm.cache.cache_key``.

    ``model`` es el alias canónico (``"qwen3-4b"``), nunca el id de un motor."""
    fields = ["v1", model, _prompt_sha256(prompt, system), str(int(max_tokens)),
              f"{float(temperature):.3f}", "1" if json_mode else "0"]
    return hashlib.sha256("|".join(fields).encode("utf-8")).hexdigest()


# The DDL of puremacro.llm.cache, character for character, so that sqlite_master
# reads the same whichever implementation created the table.
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

_UNWRITABLE_REPORTED: set[str] = set()


def _dedup_fallback_paths(primary: Path, candidates) -> tuple[Path, ...]:
    out: list[Path] = []
    seen: set[Path] = {primary}
    for item in candidates:
        p = Path(item).expanduser().absolute()
        if p not in seen:
            seen.add(p)
            out.append(p)
    return tuple(out)


class _FallbackCache:
    """Un archivo de caché con el formato ``v1`` de ``puremacro.llm.cache``.

    ``get`` y ``stats`` abren el archivo (y cualquier ``fallback_paths``) en
    sólo lectura: nunca los crean ni los modifican, y sirven en un volumen de
    sólo lectura. ``set`` crea la carpeta, el archivo, la tabla y el índice de
    ``self.path`` la primera vez y nunca toca ``fallback_paths``. Ningún método
    lanza por culpa de la caché: un archivo ausente o corrupto, o un Python sin
    ``sqlite3``, es una caché vacía.
    """

    def __init__(self, path=None, fallback_paths=()) -> None:
        if isinstance(path, Sequence) and not isinstance(path, (str, bytes, bytearray, os.PathLike)):
            if len(path) == 0:
                raise ValueError("CompletionCache requires at least one path when a sequence is given")
            self.path = Path(path[0]).expanduser().absolute()
            combined = list(path[1:])
            if fallback_paths:
                if isinstance(fallback_paths, (str, bytes, os.PathLike)):
                    combined.append(fallback_paths)
                else:
                    combined.extend(fallback_paths)
            self.fallback_paths = _dedup_fallback_paths(self.path, combined)
        else:
            chosen = cache_path() if path is None else path
            self.path = Path(chosen).expanduser().absolute()
            if fallback_paths:
                raw_fb = (fallback_paths,) if isinstance(fallback_paths, (str, bytes, os.PathLike)) else fallback_paths
            else:
                env_fb = os.environ.get(_FALLBACK_CACHE_ENV, "").strip()
                raw_fb = [p.strip() for p in env_fb.split(os.pathsep) if p.strip()] if env_fb else ()
            self.fallback_paths = _dedup_fallback_paths(self.path, raw_fb)

    def __repr__(self) -> str:
        if self.fallback_paths:
            fbs = [_shown(p) for p in self.fallback_paths]
            return f"_FallbackCache({_shown(self.path)!r}, fallback_paths={fbs!r})"
        return f"_FallbackCache({_shown(self.path)!r})"

    def _read_only(self, target: Path | None = None):
        import sqlite3
        p = self.path if target is None else target
        return sqlite3.connect(p.as_uri() + "?mode=ro", uri=True)

    def get(self, key: str) -> str | None:
        """La respuesta guardada para ``key``, o ``None``."""
        for tier in (self.path, *self.fallback_paths):
            try:
                conn = self._read_only(tier)
            except Exception:
                continue
            try:
                row = conn.execute(
                    "SELECT completion FROM completions WHERE key = ?", (key,)
                ).fetchone()
            except Exception:
                row = None
            finally:
                conn.close()
            if row is not None:
                return row[0]
        return None

    def set(self, key: str, model: str, prompt_sha256: str, max_tokens: int,
            temperature: float, json_mode: bool, completion: str,
            engine: str | None = None, engine_model_id: str | None = None,
            lang: str = "es") -> bool:
        """Guarda una respuesta; si la clave ya está, se queda la primera (``INSERT OR IGNORE``).

        Devuelve ``True`` si la fila queda en el archivo. Si no se puede
        escribir, lo dice una vez por archivo, en el idioma ``lang``, y devuelve
        ``False``: nunca lanza."""
        try:
            import sqlite3
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.path)
            try:
                conn.execute(_CREATE_TABLE)
                conn.execute(_CREATE_INDEX)
                conn.execute(
                    "INSERT OR IGNORE INTO completions (key, model, prompt_sha256, max_tokens, "
                    "temperature, json_mode, completion, engine, engine_model_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (key, model, prompt_sha256, int(max_tokens), float(temperature),
                     1 if json_mode else 0, completion, engine, engine_model_id))
                conn.commit()
            finally:
                conn.close()
        except Exception as exc:
            name = str(self.path)
            if name not in _UNWRITABLE_REPORTED:
                _UNWRITABLE_REPORTED.add(name)
                print(_say(lang,
                           f"[modelo local] no se pudo escribir en la caché {_shown(self.path)} "
                           f"({type(exc).__name__}); la respuesta se usa, pero no se guarda.",
                           f"[local model] could not write to the cache {_shown(self.path)} "
                           f"({type(exc).__name__}); the answer is used, but not saved."))
            return False
        return True

    def stats(self) -> dict:
        """Filas por modelo y tamaño del archivo; una caché inservible tiene cero filas.

        Claves ``rows``, ``by_model`` (alias -> filas), ``bytes`` y ``path``, como
        ``puremacro.llm.CompletionCache.stats``."""
        by_model: dict[str, int] = {}
        if not self.fallback_paths:
            try:
                conn = self._read_only()
            except Exception:
                conn = None
            if conn is not None:
                try:
                    by_model = dict(conn.execute(
                        "SELECT model, COUNT(*) FROM completions GROUP BY model ORDER BY model").fetchall())
                except Exception:
                    pass
                finally:
                    conn.close()
        else:
            seen_keys: set[str] = set()
            counts_by_model: dict[str, int] = {}
            for tier in (self.path, *self.fallback_paths):
                try:
                    conn = self._read_only(tier)
                except Exception:
                    continue
                try:
                    for key, model in conn.execute("SELECT key, model FROM completions"):
                        if key not in seen_keys:
                            seen_keys.add(key)
                            counts_by_model[model] = counts_by_model.get(model, 0) + 1
                except Exception:
                    pass
                finally:
                    conn.close()
            by_model = dict(sorted(counts_by_model.items()))
        try:
            size = self.path.stat().st_size
        except OSError:
            size = 0
        out = {"rows": sum(by_model.values()), "by_model": by_model, "bytes": size,
               "path": str(self.path)}
        if self.fallback_paths:
            out["fallback_paths"] = [str(p) for p in self.fallback_paths]
        return out


class CacheMiss(LookupError):
    """La caché no tiene la respuesta y el modelo local no pudo darla.

    Atributos ``key`` (la clave buscada) y ``model`` (el alias canónico). Con
    puremacro 5.0 o posterior este nombre es ``puremacro.llm.CacheMiss``.
    """

    def __init__(self, key: str, model: str) -> None:
        super().__init__(key, model)
        self.key = key
        self.model = model

    def __str__(self) -> str:
        return (f"sin respuesta en caché para el modelo {self.model!r} (clave {self.key[:12]}…); "
                "ejecútalo una vez con un modelo local (lección T00_B) o usa una caché que la tenga")


# ---------------------------------------------------------------------------
# Modos, modelos y memoria (las reglas de puremacro.llm)
# ---------------------------------------------------------------------------

_MODE_TEXT = {                  # (es, en): what report() says of each mode, through _say
    "auto": ("primero la caché; si falta la respuesta, el modelo local",
             "the cache first; if the answer is missing, the local model"),
    "cache": ("sólo la caché del curso; sin motores ni red",
              "only the course cache; no engines, no network"),
    "off": ("siempre el modelo local; sin caché",
            "always the local model; no cache"),
}


def _resolve_lang(lang: str | None = None) -> str:
    """El idioma de las notas: ``lang`` (``"es"`` o ``"en"``), o ``"es"`` si es ``None``.

    Cualquier otro valor es un error del programa y lanza ``ValueError``."""
    if lang is None:
        return "es"
    if lang not in _LANGS:
        raise ValueError(f"idioma desconocido {lang!r} en el argumento lang; usa 'es' o 'en'")
    return lang


def _resolve_mode(mode: str | None = None, lang: str = "es") -> str:
    """``mode``; si no, ``MAV_LLM_MODE``; si no, ``PUREMACRO_LLM_MODE``; si no, ``auto``.

    Un ``mode`` desconocido es un error del programa y lanza ``ValueError``. Un valor
    desconocido en la variable que manda no detiene la lección: se avisa una vez, en el
    idioma ``lang``, y se usa ``auto``."""
    if mode is not None:
        chosen = str(mode).strip().lower()
        if chosen not in MODES:
            raise ValueError(f"modo desconocido {mode!r} en el argumento mode; "
                             "usa 'auto', 'cache' u 'off'")
        return chosen
    for var in (MODE_ENV, _LIBRARY_MODE_ENV):
        raw = os.environ.get(var, "").strip()
        if not raw:
            continue
        if raw.lower() in MODES:
            return raw.lower()
        text = _scrub(_say(lang,
                           f"[modelo local] {var}={raw!r} no es un modo válido "
                           "(usa 'auto', 'cache' u 'off'); se usa 'auto'.",
                           f"[local model] {var}={raw!r} is not a valid mode "
                           "(use 'auto', 'cache' or 'off'); using 'auto'."), cache_path())
        if text not in _NOTICES_SHOWN:
            _NOTICES_SHOWN.add(text)
            print(text)
        return "auto"
    return "auto"


_ALIASES = {"default": "qwen3-4b", "small": "qwen3-1.7b", "large": "qwen3-8b"}

# Qwen3 ids per engine, checked on Hugging Face (2026-10-12): the MLX 4-bit
# conversions, the exact Q4_K_M GGUF files, the Ollama tags, and the alias itself
# for OpenAI-compatible servers.
_MODEL_IDS: dict[str, dict[str, object]] = {
    "qwen3-1.7b": {"mlx": "mlx-community/Qwen3-1.7B-4bit",
                   "llamacpp": ("ggml-org/Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q4_K_M.gguf"),
                   "ollama": "qwen3:1.7b",
                   "openai": "qwen3-1.7b"},
    "qwen3-4b": {"mlx": "mlx-community/Qwen3-4B-4bit",
                 "llamacpp": ("Qwen/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf"),
                 "ollama": "qwen3:4b",
                 "openai": "qwen3-4b"},
    "qwen3-8b": {"mlx": "mlx-community/Qwen3-8B-4bit",
                 "llamacpp": ("Qwen/Qwen3-8B-GGUF", "Qwen3-8B-Q4_K_M.gguf"),
                 "ollama": "qwen3:8b",
                 "openai": "qwen3-8b"},
}
_THINKING = frozenset({"qwen3-1.7b", "qwen3-4b", "qwen3-8b"})
_MIN_BUDGET_GB = {"qwen3-1.7b": 3.0, "qwen3-4b": 5.0, "qwen3-8b": 10.0}
# 4.8.1's engines cannot pass enable_thinking=False to the chat template, so the
# fallback appends "/no_think"; Qwen3 then opens its answer with an empty
# <think></think> block of about four tokens, which these extra tokens pay for.
_THINKING_ALLOWANCE = 8


def _canonical(model: str = "default") -> str:
    """Alias canónico: ``default`` (o ``PUREMACRO_LLM_MODEL``), ``small`` y ``large`` se traducen;
    cualquier otro nombre pasa tal cual."""
    name = (model or "default").strip()
    if name == "default":
        name = os.environ.get(_MODEL_ENV, "").strip() or name
    return _ALIASES.get(name, name)


def _pick_alias(budget_gb: float) -> str:
    """El modelo que la regla de niveles elige para un presupuesto de memoria en GB.

    Menos de 5 GB: ``qwen3-1.7b``; si no, ``qwen3-4b``. El 8B nunca se elige
    solo: se pide por nombre y necesita 10 GB."""
    return "qwen3-1.7b" if budget_gb < 5.0 else "qwen3-4b"


def _largest_fitting(budget_gb: float) -> str | None:
    fitting = [alias for alias, gb in sorted(_MIN_BUDGET_GB.items(), key=lambda kv: kv[1])
               if gb <= budget_gb]
    return fitting[-1] if fitting else None


def _importable(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ValueError:              # in sys.modules without a __spec__ (a stand-in module)
        return sys.modules.get(name) is not None
    except Exception:               # a parent package that fails to import
        return False


def _memory_mb() -> float | None:
    try:
        page, pages = os.sysconf("SC_PAGE_SIZE"), os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return None
    if isinstance(page, int) and isinstance(pages, int) and page > 0 and pages > 0:
        return page * pages / (1024 * 1024)
    return None


def _device_budget_gb() -> float:
    """Memoria que puede usar un modelo aquí, en GB de 1024³ bytes, como
    ``device_budget_gb`` de ``puremacro.llm``.

    El conjunto de trabajo recomendado de MLX si MLX está instalado; si no, el
    60 % de la RAM; si no se sabe, 8 GB. Sólo se consulta fuera del modo
    ``cache``, y sin arrancar hilos."""
    if _importable("mlx"):
        try:
            import mlx.core as mx
            info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
            return float(info["max_recommended_working_set_size"]) / 1024 ** 3
        except Exception:
            pass
    mb = _memory_mb()
    return mb * 0.6 / 1024 if mb else 8.0


class _ModelTooLarge(RuntimeError):
    """El modelo pedido no cabe en el presupuesto de memoria de este equipo."""

    def __init__(self, alias: str, suggested: str | None, budget_gb: float) -> None:
        super().__init__(alias, suggested, budget_gb)
        self.alias = alias
        self.suggested = suggested
        self.budget_gb = budget_gb


def _check_fits(alias: str) -> None:
    """Lanza ``_ModelTooLarge`` si un modelo de la escalera no cabe; los demás no se comprueban."""
    need = _MIN_BUDGET_GB.get(alias)
    if need is None:
        return
    budget = _device_budget_gb()
    if budget < need:
        raise _ModelTooLarge(alias, _largest_fitting(budget), budget)


# ---------------------------------------------------------------------------
# Motores de puremacro 4.8.1 (sólo en los modos auto y off)
# ---------------------------------------------------------------------------

class _LocalLLMUnavailable(RuntimeError):
    """Ningún motor local: ni MLX, ni llama.cpp, ni un servidor que responda."""


class _EngineRefused(RuntimeError):
    """El servidor local respondió, pero con un error HTTP (p. ej., el modelo no está descargado)."""

    def __init__(self, engine: str, model_id, status: int) -> None:
        super().__init__(engine, model_id, status)
        self.engine = engine
        self.model_id = model_id
        self.status = status


_DEFAULT_SERVERS = (("ollama", "http://localhost:11434"),
                    ("openai", "http://localhost:8080"),
                    ("openai", "http://localhost:1234"))
# One engine object per server setting, so a loaded model stays loaded across
# get_llm() calls; a failed search is remembered for a minute, because probing
# three closed ports on every miss costs seconds on some systems.
_ENGINES: dict = {}
_LAST_FAILURE: dict = {}
_RETRY_AFTER_S = 60.0


def _servers() -> tuple:
    url = os.environ.get(_BASE_URL_ENV, "").strip()
    return (("ollama", url), ("openai", url)) if url else _DEFAULT_SERVERS


def _resolve_engine():
    """El primer motor usable: MLX, llama.cpp y luego los servidores locales.

    La disponibilidad se decide por importación (``find_spec``), nunca por la
    plataforma: así MLX también se elige en el iPad. Con
    ``PUREMACRO_LLM_BASE_URL`` sólo se prueba ese servidor, como Ollama y como
    servidor compatible con OpenAI. Lanza ``_LocalLLMUnavailable`` si no hay
    ninguno."""
    servers = _servers()
    engine = _ENGINES.get(servers)
    if engine is not None:
        return engine
    failed = _LAST_FAILURE.get(servers)
    if failed is not None and time.monotonic() - failed < _RETRY_AFTER_S:
        raise _LocalLLMUnavailable("sin motor local (comprobado hace menos de un minuto)")
    from puremacro.narrative import _local_engines as engines
    if _importable("mlx_lm") and _importable("mlx"):
        engine = engines.MLXEngine()
    elif _importable("llama_cpp"):
        engine = engines.LlamaCppEngine(n_ctx=4096)
    else:
        for api, url in servers:
            candidate = engines.HTTPEngine(base_url=url, api=api, timeout=120.0)
            try:
                up = candidate.available()
            except Exception:
                up = False
            if up:
                engine = candidate
                break
    if engine is None:
        _LAST_FAILURE[servers] = time.monotonic()
        raise _LocalLLMUnavailable("sin motor local")
    _ENGINES[servers] = engine
    _LAST_FAILURE.pop(servers, None)
    return engine


def _forget_engine(engine) -> None:
    for setting in [s for s, e in _ENGINES.items() if e is engine]:
        del _ENGINES[setting]


def _available_engines(lang: str = "es") -> list[str]:
    """Los motores que hay aquí, para el informe en el idioma ``lang`` (abre sockets: nunca
    en modo ``cache``)."""
    names = []
    if _importable("mlx_lm") and _importable("mlx"):
        names.append("mlx")
    if _importable("llama_cpp"):
        names.append("llamacpp")
    try:
        from puremacro.narrative import _local_engines as engines
    except Exception:
        return names
    for api, url in _servers():
        try:
            if engines.HTTPEngine(base_url=url, api=api, timeout=120.0).available():
                names.append(f"{api} {_say(lang, 'en', 'at')} {url}")
        except Exception:
            pass
    return names


def _engine_model_id(alias: str, engine_name: str):
    ids = _MODEL_IDS.get(alias)
    if ids is not None:
        return ids.get(engine_name, alias)
    from puremacro.narrative._local_engines import resolve_model_id
    return resolve_model_id(alias, engine_name)       # 4.8.1's names; unknown ones pass through


def _strip_thinking(text) -> str:
    """Quita los bloques ``<think>…</think>`` y lo que siga a un ``<think>`` sin cerrar."""
    s = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    if "<think>" in s:
        s = s[: s.index("<think>")]
    return s.strip()


_NOTICES_SHOWN: set[str] = set()


def _notice(exc: BaseException, lang: str = "es") -> None:
    """Dice una vez por sesión por qué no respondió el modelo local, sin rutas ni mensajes.

    ``lang`` (``"es"`` o ``"en"``) es el idioma del LLM que avisa: cada texto existe en los
    dos, y cada uno sale una sola vez por sesión."""
    kind = type(exc).__name__.lstrip("_")
    if kind == "LocalLLMUnavailable":
        text = _say(lang,
                    "[modelo local] no hay ningún motor local (MLX, llama.cpp, Ollama o un servidor "
                    "compatible con OpenAI); la lección T00_B explica cómo instalar uno.",
                    "[local model] no local engine found (MLX, llama.cpp, Ollama or an "
                    "OpenAI-compatible server); lesson T00_B explains how to install one.")
    elif kind == "EngineRefused":
        engine = getattr(exc, "engine", "?")
        model_id = getattr(exc, "model_id", "?")
        status = getattr(exc, "status", "?")
        if status != 404:
            hint = _say(lang, "revisa el servidor (memoria libre, modelo cargado)",
                        "check the server (free memory, model loaded)")
        elif engine == "ollama":
            hint = _say(lang, f"descarga el modelo con `ollama pull {model_id}`",
                        f"download the model with `ollama pull {model_id}`")
        else:
            hint = _say(lang, f"carga el modelo {model_id} en el servidor",
                        f"load the model {model_id} on the server")
        text = _say(lang,
                    f"[modelo local] el servidor {engine} respondió con el error HTTP {status}; "
                    f"{hint} (lección T00_B).",
                    f"[local model] the {engine} server answered with HTTP error {status}; "
                    f"{hint} (lesson T00_B).")
    elif kind == "ModelTooLarge":
        alias = getattr(exc, "alias", "?")
        budget = getattr(exc, "budget_gb", None)
        suggested = getattr(exc, "suggested", None)
        about = _say(lang, "unos", "about")
        memory = f" ({about} {budget:.1f} GB)" if isinstance(budget, (int, float)) else ""
        if suggested:
            advice = _say(lang, f"prueba con PUREMACRO_LLM_MODEL={suggested}",
                          f"try PUREMACRO_LLM_MODEL={suggested}")
        else:
            advice = _say(lang, "en este equipo usa el modo cache", "use cache mode on this device")
        text = _say(lang,
                    f"[modelo local] {alias} no cabe en la memoria de este equipo{memory}; "
                    f"{advice} (lección T00_B).",
                    f"[local model] {alias} does not fit in this device's memory{memory}; "
                    f"{advice} (lesson T00_B).")
    else:
        text = _say(lang, f"[modelo local] el modelo local falló ({kind}); revisa la lección T00_B.",
                    f"[local model] the local model failed ({kind}); see lesson T00_B.")
    if text not in _NOTICES_SHOWN:
        _NOTICES_SHOWN.add(text)
        print(text)


def _puremacro_version(lang: str = "es") -> str:
    version = getattr(sys.modules.get("puremacro"), "__version__", None)
    return str(version) if version else _say(lang, "no disponible", "not available")


# ---------------------------------------------------------------------------
# Adaptadores para las funciones narrativas de puremacro 4.8.1
# ---------------------------------------------------------------------------

def _index_prompt(text: str, category: str) -> str:
    """El prompt de ``LocalProvider.score_paragraph`` de puremacro, byte a byte.

    La clave de caché depende de él: cambiarlo dejaría sin respuesta las filas
    ya calculadas."""
    return (
        f"Score how strongly this paragraph expresses '{category}' "
        f"on a scale from 0.0 (not at all) to 1.0 (strongly). "
        f"Reply with ONLY a single decimal number.\n\nParagraph: {text}"
    )


_FIRST_NUMBER = re.compile(r"\d*\.?\d+")


def _drop_plus_signs(text: str) -> str:
    """``text`` sin el ``+`` que precede a un dígito fuera de las cadenas JSON.

    JSON no tiene un más unario, pero los prompts escriben ``"sign": -1 | 0 | +1`` y los
    modelos lo copian (``"sign": +1``). Lo que va entre comillas dobles, escapes incluidos,
    se deja tal cual (``"+75 pb"`` en un extracto). Es la función del mismo nombre de
    ``score_llm`` en puremacro 5.0, para que el curso y la librería lean lo mismo."""
    out: list[str] = []
    in_string = escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "+" and i + 1 < len(text) and text[i + 1].isdigit():
            continue
        out.append(ch)
    return "".join(out)


# The record's date in a prompt of 4.8.1's score_llm: every _PROMPTS template has its
# "Date: {date}" line before the text, so the first such line is the announcement. The value
# is whatever the corpus wrote (score_llm passes it on unchanged); _quarter reads it as
# score_llm does, with pandas, so any format pandas accepts counts and anything else gives no
# quarter to compare with.
_ANNOUNCEMENT_DATE = re.compile(r"^Date:[ \t]*(.+?)[ \t]*$", re.M)


def _quarter(value):
    """El trimestre ``(año, 1..4)`` de la fecha que ``pandas.Timestamp`` lee en ``value``, como
    la lee ``score_llm`` de puremacro 4.8.1; ``None`` si ``value`` está vacío o no es una fecha."""
    if not value:
        return None
    try:
        import pandas as pd
        stamp = pd.Timestamp(value)
        return None if pd.isna(stamp) else (stamp.year, stamp.quarter)
    except Exception:
        return None


def _reads_as_zero(value) -> bool:
    """Si ``score_llm`` de puremacro 4.8.1 lee ``value`` como un ``horizon_quarters`` de 0 (``int(value)``)."""
    if value is None:
        return False
    try:
        return int(value) == 0
    except (TypeError, ValueError, OverflowError):
        return False


def _null_same_quarter_timing(text: str, prompt: str) -> str:
    """``text`` con ``implementation_date`` a ``null`` en cada evento cuya fecha cae en el
    trimestre del anuncio, y con ``horizon_quarters`` 0 a ``null``.

    El anuncio es la primera línea ``Date:`` de ``prompt``, leída como la lee ``score_llm``
    (``pandas.Timestamp``, en cualquier formato que pandas acepte); si no es una fecha, no se
    anula nada, tampoco un ``horizon_quarters`` 0. ``score_llm`` de puremacro 4.8.1
    convierte esos dos valores en un perfil que empieza el primer día de ese trimestre, y
    ``NarrativeEvent`` lo rechaza si el anuncio cae más de 7 días después: la excepción
    detiene el corpus entero. Un perfil vacío ya significa «el trimestre del anuncio», así
    que el evento dice lo mismo. Cuando anula la fecha anula también ``horizon_quarters``:
    4.8.1 sólo lee el horizonte si no hay fecha, y resucitarlo movería el evento.

    Lee la lista como la lee ``_parse_response`` de 4.8.1: sin el bloque de código
    ```` ``` ```` que la envuelva y, si eso no es JSON, el texto entre el primer ``[`` y el
    último ``]`` (prosa alrededor, una nota tras el bloque, un ``<think>`` vacío). Si no hay
    lista JSON, si el prompt no tiene esa línea o si no cambia nada, devuelve ``text`` tal
    cual; si cambia algo, la lista sola con ``json.dumps(..., ensure_ascii=False)`` (4.8.1
    descarta el resto del texto). Nunca lanza."""
    try:
        match = _ANNOUNCEMENT_DATE.search(prompt or "")
        if match is None:
            return text
        body = text.strip()
        if body.startswith("```"):              # the fence rule of 4.8.1's _parse_response
            body = body.strip("`")
            if body.lower().startswith("json"):
                body = body[4:]
            body = body.strip()
        try:
            events = json.loads(body)
        except json.JSONDecodeError:            # its fallback: first "[" to last "]"
            events = json.loads(body[body.index("["):body.rindex("]") + 1])
        if not isinstance(events, list):
            return text
        announced = _quarter(match.group(1).strip())
        changed = False
        for event in events:
            if not isinstance(event, dict):
                continue
            if announced is None:                # no readable announcement: nothing is normalised
                continue
            if _quarter(event.get("implementation_date")) == announced:
                event["implementation_date"] = None
                event["horizon_quarters"] = None   # 4.8.1 would read it once the date is gone
                changed = True
            if _reads_as_zero(event.get("horizon_quarters")):
                event["horizon_quarters"] = None
                changed = True
        return json.dumps(events, ensure_ascii=False) if changed else text
    except Exception:                           # not JSON, or anything odd: as before
        return text


@contextlib.contextmanager
def _throwaway_score_cache(*variables: str):
    """Mientras dura el bloque, cada variable de ``variables`` apunta a la misma caché de
    puntuaciones nueva en una carpeta temporal. Al salir, también tras un error, cada una
    recupera su valor (o desaparece si no estaba) y la carpeta se borra."""
    import tempfile
    saved = {name: os.environ.get(name) for name in variables}
    with tempfile.TemporaryDirectory(prefix="mav_llm_scores_", ignore_cleanup_errors=True) as folder:
        path = os.path.join(folder, "llm_scores.sqlite")
        try:
            for name in variables:
                os.environ[name] = path
            yield
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


class CourseProvider:
    """Proveedor de ``llm_prob_kernel`` que responde con el modelo del curso.

    Imita a ``LocalProvider`` de puremacro (``name`` ``"local"``, ``model`` el
    alias, el mismo prompt, 8 tokens, temperatura 0): devuelve el primer número
    de la respuesta acotado a [0, 1], y 0.0 si no hay respuesta.
    """

    name = "local"

    def __init__(self, llm: "LLM") -> None:
        self._llm = llm
        self.model = llm.model_alias

    def __repr__(self) -> str:
        return f"CourseProvider(model={self.model!r}, mode={self._llm.mode!r})"

    def score_paragraph(self, text: str, category: str) -> float:
        """P(``category`` | ``text``) en [0, 1]; 0.0 si el modelo no responde."""
        out = self._llm.try_complete(_index_prompt(text, category), max_tokens=8,
                                     temperature=0.0, json_mode=False)
        match = _FIRST_NUMBER.search(out or "")
        if not match:
            return 0.0
        try:
            return max(0.0, min(1.0, float(match.group())))
        except ValueError:
            return 0.0


class CourseBackend:
    """Backend de ``score_llm`` que responde con el modelo del curso.

    ``call(prompt)`` devuelve la respuesta (1024 tokens, temperatura 0, sin modo
    JSON), o ``"[]"`` (ningún evento) si no hay respuesta, con dos arreglos para
    ``score_llm`` de puremacro 4.8.1. Se hacen al devolverla: la caché guarda la
    respuesta del modelo tal cual.

    * Quita el ``+`` que el modelo escribe delante de un dígito fuera de las cadenas
      JSON: ``"sign": +1`` no es JSON válido para su analizador. Dentro de una cadena
      (``"+75 pb"`` en un extracto) se conserva, como hace ``score_llm`` desde la 5.0.
    * En una lista JSON de eventos, pone a ``null`` la ``implementation_date`` que
      cae en el trimestre del anuncio (la línea ``Date:`` del prompt) y el
      ``horizon_quarters`` 0: ``score_llm`` los convertiría en un perfil que
      ``NarrativeEvent`` rechaza, y perdería todos los documentos. Un perfil vacío
      ya significa «el trimestre del anuncio».
    """

    def __init__(self, llm: "LLM") -> None:
        self._llm = llm
        self.model = llm.model_alias
        self.max_tokens = 1024
        self.temperature = 0.0
        self.json_mode = False

    def __repr__(self) -> str:
        return f"CourseBackend(model={self.model!r}, mode={self._llm.mode!r})"

    def call(self, prompt: str) -> str:
        """La respuesta del modelo a ``prompt`` con los dos arreglos, o ``"[]"`` si no la hay."""
        out = self._llm.try_complete(prompt, max_tokens=self.max_tokens,
                                     temperature=self.temperature, json_mode=self.json_mode)
        if out is None:
            return "[]"
        return _null_same_quarter_timing(_drop_plus_signs(out), prompt)


# ---------------------------------------------------------------------------
# El modelo del curso
# ---------------------------------------------------------------------------

class LLM:
    """El modelo del curso en un modo. Se obtiene con :func:`get_llm`.

    Atributos
    ---------
    mode : str
        ``auto``, ``cache`` u ``off``.
    model_alias : str
        Alias canónico del modelo: el de ``model`` si se dio (``get_llm(model=...)``);
        si no, ``qwen3-4b``, salvo que ``PUREMACRO_LLM_MODEL`` diga otro.
    lang : str
        Idioma de las notas que imprime este objeto: ``"es"`` (si no se dio otro) o
        ``"en"``.
    cache_path : Path
        El archivo de caché: ``PUREMACRO_LLM_CACHE`` si está definida; si no, la
        caché del curso (:func:`cache_path`). Se fija al crear el objeto.
    backend_name : str
        ``"puremacro.llm"`` (puremacro 5.0 o posterior) o ``"fallback"``
        (puremacro 4.8.1 y anteriores).
    """

    def __init__(self, mode: str | None = None, model: str | None = None,
                 lang: str | None = None, *, cache=None) -> None:
        self.lang = _resolve_lang(lang)
        self.mode = _resolve_mode(mode, self.lang)
        self._library = _LIBRARY
        self.backend_name = "fallback" if _LIBRARY is None else "puremacro.llm"
        name = model or "default"
        self.model_alias = _canonical(name) if _LIBRARY is None else _library_alias(name)
        # What the library's complete() receives: "default" when no model was given, as
        # before, so the library applies its own default; the canonical alias otherwise.
        self._library_model = self.model_alias if model else "default"
        self._explicit_cache = cache is not None
        if cache is not None:
            if hasattr(cache, "path"):
                self.cache_path = Path(cache.path).expanduser().absolute()
                self.fallback_paths: tuple[Path, ...] = _dedup_fallback_paths(
                    self.cache_path, getattr(cache, "fallback_paths", ())
                )
            elif isinstance(cache, Sequence) and not isinstance(
                cache, (str, bytes, bytearray, os.PathLike)
            ):
                resolved = _FallbackCache(cache)
                self.cache_path = resolved.path
                self.fallback_paths = resolved.fallback_paths
            else:
                self.cache_path = Path(cache).expanduser().absolute()
                self.fallback_paths = _dedup_fallback_paths(self.cache_path, (cache_path(),))
        else:
            env_user = os.environ.get(USER_CACHE_ENV, "").strip()
            env_cache = os.environ.get(_CACHE_ENV, "").strip()
            if env_user:
                self.cache_path = Path(env_user).expanduser().absolute()
                self.fallback_paths = _dedup_fallback_paths(
                    self.cache_path,
                    (Path(env_cache).expanduser().absolute() if env_cache else cache_path(),),
                )
            else:
                self.cache_path = Path(env_cache).expanduser().absolute() if env_cache else cache_path()
                env_fb = os.environ.get(_FALLBACK_CACHE_ENV, "").strip()
                raw_fb = [p.strip() for p in env_fb.split(os.pathsep) if p.strip()] if env_fb else ()
                self.fallback_paths = _dedup_fallback_paths(self.cache_path, raw_fb)

    def __repr__(self) -> str:
        return (f"LLM(mode={self.mode!r}, model_alias={self.model_alias!r}, "
                f"backend_name={self.backend_name!r}, cache={_shown(self.cache_path)!r})")

    def complete(self, prompt: str, *, system: str | None = None, max_tokens: int = 256,
                 temperature: float = 0.0, json_mode: bool = False) -> str:
        """La respuesta del modelo a ``prompt``; lanza :class:`CacheMiss` si no la hay.

        En modo ``auto`` una respuesta nueva se guarda en la caché, salvo si queda
        vacía; en ``cache`` sólo se lee la caché; en ``off`` siempre se ejecuta el
        modelo. Si el modelo local no está, no cabe en la memoria o falla, se avisa
        una vez, en el idioma :attr:`lang`, y se lanza :class:`CacheMiss`, la única
        excepción que sale de aquí.
        Cada fallo queda en :data:`MISSES`."""
        key = _cache_key(self.model_alias, prompt, system=system, max_tokens=max_tokens,
                         temperature=temperature, json_mode=json_mode)
        if self._library is not None:
            return self._complete_with_library(prompt, key, system=system, max_tokens=max_tokens,
                                               temperature=temperature, json_mode=json_mode)
        cache = _FallbackCache(self.cache_path, fallback_paths=self.fallback_paths)
        if self.mode != "off":
            hit = cache.get(key)
            if hit is not None:
                return hit
        if self.mode == "cache":
            raise self._miss(key, prompt, system)
        try:
            text, engine_name, model_id = self._run_live(prompt, system, max_tokens,
                                                         temperature, json_mode)
        except Exception as exc:
            _notice(exc, self.lang)
            raise self._miss(key, prompt, system) from exc
        if self.mode == "auto" and text:
            cache.set(key, self.model_alias, _prompt_sha256(prompt, system), max_tokens,
                      temperature, json_mode, text, engine=engine_name,
                      engine_model_id=str(model_id), lang=self.lang)
        return text

    def try_complete(self, prompt: str, **kw) -> str | None:
        """Como :meth:`complete`, pero si no hay respuesta devuelve ``None`` e imprime una
        línea, en el idioma :attr:`lang`; nunca lanza. Es lo que deben usar las celdas de
        las lecciones. Si lo que falla es la caché misma (``CacheUnavailable`` de
        ``puremacro.llm``: un puntero de LFS, una copia corrupta, Pyodide sin ``sqlite3``), la
        línea lo dice y pide descargar de nuevo ``data_curso/llm_cache.sqlite``."""
        try:
            return self.complete(prompt, **kw)
        except CacheMiss as exc:
            unexpected = None
            # puremacro.llm's CacheUnavailable is a CacheMiss whose ``reason`` says the cache
            # itself could not be read: not a missing answer, so not the model's fault.
            unreadable = getattr(exc, "reason", None) is not None
        except Exception as exc:                # complete raises only CacheMiss: a bug
            unexpected, unreadable = type(exc).__name__, False
        try:
            sha = _prompt_sha256(prompt, kw.get("system"))
        except Exception:
            sha = "?" * 64
        if unreadable:
            print(_say(self.lang,
                       f"[modelo local] la caché {_shown(self.cache_path)} no se puede leer; "
                       "descarga de nuevo data_curso/llm_cache.sqlite (lección T00_B)",
                       f"[local model] the cache {_shown(self.cache_path)} cannot be read; "
                       "download data_curso/llm_cache.sqlite again (lesson T00_B)"))
        elif unexpected is None:
            print(_say(self.lang,
                       f"[modelo local] sin respuesta en caché para {self.model_alias} {sha[:8]}; "
                       "abre T00_B para instalar un modelo",
                       f"[local model] no cached answer for {self.model_alias} {sha[:8]}; "
                       "open lesson T00_B to install a model"))
        else:
            MISSES.append((self.model_alias, sha))
            print(_say(self.lang,
                       f"[modelo local] error inesperado ({unexpected}) con {self.model_alias} "
                       f"{sha[:8]}; la lección sigue sin esta respuesta",
                       f"[local model] unexpected error ({unexpected}) with {self.model_alias} "
                       f"{sha[:8]}; the lesson continues without this answer"))
        return None

    def report(self) -> str:
        """Resumen para comprobar la instalación: modo, modelo, caché y sus filas,
        implementación y, salvo en modo ``cache``, los motores que hay. En el idioma
        :attr:`lang`. Sin rutas absolutas."""
        lang = self.lang

        def label(es: str, en: str) -> str:         # two spaces, the label, padded to 10
            return f"  {_say(lang, es, en):<10}"

        stats = _FallbackCache(self.cache_path, fallback_paths=self.fallback_paths).stats()
        rows = stats["rows"]
        state = (_say(lang, f"{rows} fila", f"{rows} row") if rows == 1
                 else _say(lang, f"{rows} filas", f"{rows} rows"))
        if not stats["bytes"]:
            state += _say(lang, "; el archivo todavía no existe", "; the file does not exist yet")
        detail = "" if self._library is not None else _say(lang, "; sin puremacro.llm",
                                                            "; without puremacro.llm")
        lines = [
            _say(lang, "Modelo local del curso", "Course local model"),
            f"{label('modo:', 'mode:')}{self.mode} ({_say(lang, *_MODE_TEXT[self.mode])})",
            f"{label('modelo:', 'model:')}{self.model_alias}",
            f"{label('caché:', 'cache:')}{_shown(self.cache_path)} ({state})",
            f"{label('backend:', 'backend:')}{self.backend_name} "
            f"(puremacro {_puremacro_version(lang)}{detail})",
        ]
        if self.mode == "cache":
            return "\n".join(lines)
        if self._library is not None:
            try:
                text = self._library.report()
            except Exception as exc:
                text = _say(lang, f"(puremacro.llm.report() falló: {type(exc).__name__})",
                            f"(puremacro.llm.report() failed: {type(exc).__name__})")
            return "\n".join(lines + ["", _scrub(str(text or ""), self.cache_path)])
        try:
            found = _available_engines(lang)
        except Exception:
            found = []
        try:
            budget = _device_budget_gb()
        except Exception:
            budget = 8.0
        lines.append(label("motores:", "engines:") + (
            ", ".join(found) if found
            else _say(lang, "ninguno; la lección T00_B explica cómo instalar uno",
                      "none; lesson T00_B explains how to install one")))
        lines.append(label("memoria:", "memory:") + _say(
            lang,
            f"unos {budget:.1f} GB para el modelo; el indicado para este equipo es {_pick_alias(budget)}",
            f"about {budget:.1f} GB for the model; the one suited to this device is {_pick_alias(budget)}"))
        need = _MIN_BUDGET_GB.get(self.model_alias)
        if need is not None and budget < need:
            fits = _largest_fitting(budget)
            advice = (_say(lang, f"usa PUREMACRO_LLM_MODEL={fits}", f"use PUREMACRO_LLM_MODEL={fits}")
                      if fits else _say(lang, "usa el modo cache", "use cache mode"))
            lines.append(label("aviso:", "notice:") + _say(
                lang, f"{self.model_alias} necesita {need:.0f} GB; {advice}",
                f"{self.model_alias} needs {need:.0f} GB; {advice}"))
        return "\n".join(lines)

    def provider(self) -> CourseProvider:
        """Proveedor para ``llm_prob_kernel`` (puremacro 4.8.1) con este modelo y esta caché.

        Las lecciones no se lo pasan a ``llm_prob_kernel``: llaman a :meth:`index`. El
        núcleo guardaría en su propia caché de puntuaciones el 0.0 que este proveedor da
        cuando falta una respuesta, y lo repetiría aunque luego la respuesta exista; con
        las dos cachés en un mismo archivo, además, su transacción abierta bloquea las
        escrituras de esta. :meth:`index` le da una caché de puntuaciones desechable en
        cada llamada."""
        return CourseProvider(self)

    def backend(self) -> CourseBackend:
        """Backend para ``score_llm`` (puremacro 4.8.1) con este modelo y esta caché."""
        return CourseBackend(self)

    def index(self, records, category, *, window="paragraph", language="en") -> list[tuple]:
        """La serie de ``llm_prob_kernel`` con el modelo del curso: ``[(fecha, P media), ...]``,
        una por documento. Las lecciones usan esto, nunca
        ``llm_prob_kernel(..., provider=llm.provider())``.

        ``records``, ``category``, ``window`` y ``language`` son los de ``llm_prob_kernel``.
        En cada llamada el núcleo recibe una caché de puntuaciones nueva, en una carpeta
        temporal que se borra al terminar: una ventana sin respuesta puntúa 0.0 sólo en
        esta llamada, y la siguiente usa la respuesta en cuanto la caché la tenga o haya
        un modelo instalado. Las respuestas se leen y se guardan en :attr:`cache_path`, la
        caché fijada al crear este objeto, también en el fallback, donde durante la llamada
        ``PUREMACRO_LLM_CACHE`` apunta a la caché de puntuaciones.

        Las variables que llevan al núcleo a esa carpeta dependen de :attr:`backend_name`.
        Con ``"puremacro.llm"`` (puremacro 5.0 o posterior) es sólo
        ``PUREMACRO_LLM_SCORES_CACHE``: ``PUREMACRO_LLM_CACHE`` es entonces la caché de
        respuestas de la librería y no se toca. Con ``"fallback"`` son las dos, a un mismo
        archivo: el núcleo de puremacro 4.8.1 lee ``PUREMACRO_LLM_CACHE``, el de una 5.x
        cuya ``puremacro.llm`` no se puede usar lee ``PUREMACRO_LLM_SCORES_CACHE``, y sin la
        librería no se sabe cuál es. Cada variable recupera su valor al salir, o desaparece
        si no estaba, también si hay un error."""
        from puremacro.narrative.indices import llm_prob_kernel
        # With puremacro.llm, PUREMACRO_LLM_CACHE is the library's completion cache: only the
        # score cache moves. Without it the kernel may be 4.8.1's (reads PUREMACRO_LLM_CACHE) or a
        # 5.x one whose puremacro.llm does not load (reads PUREMACRO_LLM_SCORES_CACHE): move both.
        variables = ((_SCORES_CACHE_ENV,) if self._library is not None
                     else (_CACHE_ENV, _SCORES_CACHE_ENV))
        with _throwaway_score_cache(*variables):
            return list(llm_prob_kernel(records, provider=self.provider(), category=category,
                                        window=window, language=language))

    # -- internals ------------------------------------------------------------

    def _miss(self, key: str, prompt: str, system: str | None) -> LookupError:
        MISSES.append((self.model_alias, _prompt_sha256(prompt, system)))
        return CacheMiss(key, self.model_alias)

    def _complete_with_library(self, prompt: str, key: str, **kw) -> str:
        library = self._library
        call_kw = dict(kw)
        if self.fallback_paths or self._explicit_cache:
            call_kw["cache"] = open_cache(self.cache_path, fallback_paths=self.fallback_paths)
        try:
            return library.complete(prompt, model=self._library_model, mode=self.mode, **call_kw)
        except library.CacheMiss as exc:
            # CacheUnavailable included: it keeps its ``reason``, and try_complete reports it as
            # an unreadable cache instead of a missing answer.
            model = getattr(exc, "model", None)
            MISSES.append((model if isinstance(model, str) and model else self.model_alias,
                           _prompt_sha256(prompt, kw.get("system"))))
            raise
        except Exception as exc:                # no engine, model too large, engine error
            _notice(exc, self.lang)
            raise self._miss(key, prompt, kw.get("system")) from exc

    def _run_live(self, prompt, system, max_tokens, temperature, json_mode):
        engine = _resolve_engine()
        alias = self.model_alias
        _check_fits(alias)
        model_id = _engine_model_id(alias, engine.name)
        # 4.8.1's engines take no system prompt; the key still hashes the two apart.
        text = f"{system}\n\n{prompt}" if system else prompt
        budget = int(max_tokens)
        if alias in _THINKING:
            text += "\n/no_think"
            budget += _THINKING_ALLOWANCE
        try:
            raw = engine.complete(model_id, text, max_tokens=budget,
                                  temperature=float(temperature), json_mode=bool(json_mode))
        except Exception as exc:
            if type(exc).__name__ == "LocalLLMUnavailable":
                # 4.8.1's HTTPEngine turns every URLError into LocalLLMUnavailable,
                # HTTPError included: a server that answers 404 is up but lacks the model.
                status = getattr(exc.__cause__, "code", None)
                if isinstance(status, int):
                    raise _EngineRefused(engine.name, model_id, status) from exc
                _forget_engine(engine)                            # the server went away
            raise
        return _strip_thinking(raw), engine.name, model_id


def open_cache(paths=None, *, fallback_paths=None):
    """Abre una caché de uno o dos niveles (primero la del alumno, luego la del curso en sólo lectura).

    Con ``open_cache([student_cache, cache_path()])`` las consultas miran primero ``student_cache``
    y después ``cache_path()`` sin crearla ni modificarla; las respuestas nuevas se guardan sólo
    en ``student_cache``."""
    if paths is None:
        env_user = os.environ.get(USER_CACHE_ENV, "").strip()
        if env_user and fallback_paths is None:
            paths = [user_cache_path(), cache_path()]
        else:
            paths = cache_path()
    if _LIBRARY is not None and callable(getattr(_LIBRARY, "open_cache", None)):
        return _LIBRARY.open_cache(paths, fallback_paths=fallback_paths)
    return _FallbackCache(paths, fallback_paths=fallback_paths or ())


def get_llm(mode: str | None = None, model: str | None = None, lang: str | None = None,
            *, cache=None) -> LLM:
    """El modelo del curso en ``mode`` (si no, ``MAV_LLM_MODE``, ``PUREMACRO_LLM_MODE`` o ``auto``).

    ``model`` (``"default"``, ``"small"``, ``"large"`` o un alias, como :data:`REFERENCE_MODEL`)
    cambia el modelo de este objeto, y sólo de este; sin él es ``qwen3-4b``, o el que diga
    ``PUREMACRO_LLM_MODEL``. ``lang`` es el idioma de sus notas, ``"es"`` (sin él) o ``"en"``;
    cualquier otro valor lanza ``ValueError``. ``cache`` permite pasar una caché de dos niveles
    (como ``open_cache([student_cache, cache_path()])``).

    Uso en una lección::

        import _llm
        respuesta = _llm.get_llm().try_complete("¿Qué mide el PIB?", max_tokens=64)
        referencia = _llm.get_llm(mode="cache", model=_llm.REFERENCE_MODEL).try_complete(
            "¿Qué mide el PIB?", max_tokens=64)
    """
    return LLM(mode, model=model, lang=lang, cache=cache)


# ---------------------------------------------------------------------------
# puremacro.llm (puremacro 5.0 o posterior), si está
# ---------------------------------------------------------------------------

def _load_library():
    """``puremacro.llm`` si se puede importar, o ``None`` (puremacro 4.8.1 o sin puremacro)."""
    try:
        found = importlib.util.find_spec("puremacro.llm") is not None
    except ValueError:              # in sys.modules without a __spec__ (a stand-in module)
        found = sys.modules.get("puremacro.llm") is not None
    except Exception:               # puremacro missing or broken
        return None
    if not found:
        return None
    try:
        library = importlib.import_module("puremacro.llm")
    except Exception:
        return None
    miss = getattr(library, "CacheMiss", None)
    if not (callable(getattr(library, "complete", None))
            and isinstance(miss, type) and issubclass(miss, BaseException)):
        return None
    return library


def _library_alias(model: str = "default") -> str:
    canonical = getattr(_LIBRARY, "canonical", None)
    if callable(canonical):
        try:
            return str(canonical(model))
        except Exception:
            pass
    return _canonical(model)


_LIBRARY = _load_library()
if _LIBRARY is not None:
    # The library and the course read the same file. Only here: 4.8.1's narrative
    # kernel reads PUREMACRO_LLM_CACHE as its score cache and would write into ours.
    os.environ.setdefault(_CACHE_ENV, str(cache_path()))
    CacheMiss = _LIBRARY.CacheMiss  # noqa: F811  (the library's own class)
