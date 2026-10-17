#!/usr/bin/env python3
"""Construye los corpus de comunicados de Banxico y del FOMC (2008-2026).

Descarga los comunicados de decisión de política monetaria del Banco de México
y de la Reserva Federal (FOMC) mediante ``iter_banxico_decision`` e
``iter_fed_decision`` de ``puremacro.narrative.sources``, extrae el texto y los
primeros 1 500 vocablos (``excerpt``) para el modelo local de 4B parámetros, y
escribe:

- ``notebooks/data_curso/bancos_centrales/banxico_comunicados.csv``
- ``notebooks/data_curso/bancos_centrales/fomc_statements.csv``
- ``notebooks/data_curso/bancos_centrales/manifest.json``
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date as _date
import hashlib
import inspect
import json
from pathlib import Path
import re
import sys

import pandas as pd

import puremacro._http as _pm_http
from puremacro.narrative.sources.banxico import (
    _DATE_RX as _BX_DATE_RX,
    _LISTING_URL as _BX_LISTING_URL,
    _PDF_RX as _BX_PDF_RX,
    _USER_AGENT as _BX_UA,
    _parse_date as _bx_parse_date,
    _strip as _bx_strip,
    iter_banxico_decision,
)
from puremacro.narrative.sources.fed_decision import (
    _LISTING_URL as _FED_LISTING_URL,
    _UA as _FED_UA,
    _parse_listing as _fed_parse_listing,
    iter_fed_decision,
)

RAIZ = Path(__file__).resolve().parent.parent
DESTINO = RAIZ / "notebooks" / "data_curso" / "bancos_centrales"
MAX_WORDS = 1500


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _excerpt(text: str, max_words: int = MAX_WORDS) -> str:
    words = text.split()
    return " ".join(words[:max_words])


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?:-->|<--)+", " ", text)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def _call_connector(fn, *, start: int, end: int, **kwargs):
    """Call a narrative source iterator, supporting both (start, end) and () signatures."""
    sig = inspect.signature(fn)
    call_kwargs = dict(kwargs)
    if "start" in sig.parameters:
        call_kwargs["start"] = start
    elif "start_year" in sig.parameters:
        call_kwargs["start_year"] = start
    if "end" in sig.parameters:
        call_kwargs["end"] = end
    elif "end_year" in sig.parameters:
        call_kwargs["end_year"] = end
    return fn(**call_kwargs)


def _unpack_item(item, default_bank: str, default_lang: str) -> tuple[pd.Timestamp, str, str, str, str]:
    """Unpack either a (date, text, url, meta) tuple or a SourceDocument."""
    if isinstance(item, tuple):
        dt, text, url, meta = item
        lang = (meta or {}).get("language", default_lang)
        return pd.Timestamp(dt), str(text or ""), str(url or ""), default_bank, str(lang)
    dt = pd.Timestamp(getattr(item, "date"))
    text = str(getattr(item, "text", "") or getattr(item, "title", "") or "")
    url = str(getattr(item, "url", "") or "")
    lang = str(getattr(item, "language", default_lang) or default_lang)
    return dt, text, url, default_bank, lang


def construir_corpus(*, start: int = 2008, end: int = 2026, workers: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Descarga y estructura los comunicados de Banxico y del FOMC en [start, end]."""
    orig_fetch = _pm_http._fetch

    # 1. Descargar listados y pre-cargar en paralelo los documentos dentro de [start, end]
    bx_listing_bytes = orig_fetch(_BX_LISTING_URL, 30.0, _BX_UA, None)
    # El listado HTML de Banxico viene en ISO-8859-1; recodificar a UTF-8 conserva acentos.
    bx_listing_utf8 = bx_listing_bytes.decode("latin-1", errors="ignore").encode("utf-8")
    bx_html = bx_listing_utf8.decode("utf-8", errors="ignore")

    bx_target_urls: set[str] = set()
    bx_titles_by_url: dict[str, str] = {}
    for m in _BX_PDF_RX.finditer(bx_html):
        href = m.group(1)
        ctx = _bx_strip(bx_html[max(0, m.start() - 600):m.start()])
        dm = _BX_DATE_RX.search(ctx)
        if not dm:
            continue
        dt = _bx_parse_date(dm.group(1))
        if dt is None or not (start <= dt.year <= end):
            continue
        full_url = "https://www.banxico.org.mx" + href
        title = re.sub(r"^.*?(\d{1,2}/\d{1,2}/\d{2,4})\s*", "", ctx[-300:].strip()).strip()
        title = re.sub(r"\s*<[aA]\b.*$", "", title).strip()
        bx_target_urls.add(full_url)
        bx_titles_by_url[full_url] = title

    fed_listing_bytes = orig_fetch(_FED_LISTING_URL, 30.0, _FED_UA, None)
    fed_target_urls = {
        url for dt, url in _fed_parse_listing(fed_listing_bytes)
        if start <= dt.year <= end
    }

    prefetched: dict[str, bytes] = {
        _BX_LISTING_URL: bx_listing_utf8,
        _FED_LISTING_URL: fed_listing_bytes,
    }

    def _fetch_one(args: tuple[str, str]) -> tuple[str, bytes]:
        url, ua = args
        try:
            return url, orig_fetch(url, 25.0, ua, None)
        except Exception:
            return url, b""

    tasks = [(u, _BX_UA) for u in sorted(bx_target_urls)] + [(u, _FED_UA) for u in sorted(fed_target_urls)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for url, data in pool.map(_fetch_one, tasks):
            prefetched[url] = data

    def _cached_fetch(url: str, timeout: float, user_agent: str | None, headers: dict | None) -> bytes:
        if url in prefetched:
            return prefetched[url]
        return b""

    _pm_http._fetch = _cached_fetch
    try:
        bx_rows = []
        for item in _call_connector(iter_banxico_decision, start=start, end=end, fetch_body=True):
            dt, text, url, bank, lang = _unpack_item(item, "Banxico", "es")
            if not (start <= dt.year <= end):
                continue
            cleaned = _clean_text(text)
            title = bx_titles_by_url.get(url, "")
            if title and title not in cleaned[:400]:
                cleaned = f"{title}\n\n{cleaned}" if cleaned else title
            if not cleaned:
                continue
            bx_rows.append({
                "date": dt.strftime("%Y-%m-%d"),
                "bank": bank,
                "url": url,
                "language": lang,
                "text": cleaned,
                "excerpt": _excerpt(cleaned),
            })

        fed_rows = []
        for item in _call_connector(iter_fed_decision, start=start, end=end):
            dt, text, url, bank, lang = _unpack_item(item, "Fed", "en")
            if not (start <= dt.year <= end):
                continue
            cleaned = _clean_text(text)
            if not cleaned:
                continue
            fed_rows.append({
                "date": dt.strftime("%Y-%m-%d"),
                "bank": bank,
                "url": url,
                "language": lang,
                "text": cleaned,
                "excerpt": _excerpt(cleaned),
            })
    finally:
        _pm_http._fetch = orig_fetch

    df_bx = (
        pd.DataFrame(bx_rows)
        .drop_duplicates(subset=["date"], keep="first")
        .sort_values("date")
        .reset_index(drop=True)
    )
    df_fed = (
        pd.DataFrame(fed_rows)
        .drop_duplicates(subset=["date"], keep="first")
        .sort_values("date")
        .reset_index(drop=True)
    )
    return df_bx, df_fed


def actualizar_manifiesto(carpeta: Path) -> dict:
    manifest_path = carpeta / "manifest.json"
    hoy = _date.today().isoformat()
    descripciones = {
        "minutas_banxico_muestra.csv": (
            "Fragmentos didácticos de comunicados y minutas de política monetaria (Banco de México)"
        ),
        "banxico_comunicados.csv": (
            "Comunicados de decisión de política monetaria del Banco de México (2008-2026)"
        ),
        "fomc_statements.csv": (
            "Comunicados de decisión de política monetaria del FOMC de la Reserva Federal (2008-2026)"
        ),
    }
    prev_entries: dict[str, dict] = {
        "minutas_banxico_muestra.csv": {
            "bytes": 6212,
            "created_at": "2026-10-12",
            "file": "minutas_banxico_muestra.csv",
            "rows": 3,
            "sha256": "da33ef56b5b49b078f03923eb2d6a6b89e7ac5898b2416f99f62467c724486b1",
            "source": descripciones["minutas_banxico_muestra.csv"],
        }
    }
    if manifest_path.is_file():
        try:
            prev = json.loads(manifest_path.read_text(encoding="utf-8"))
            for item in prev.get("archivos", []):
                if "file" in item:
                    prev_entries[item["file"]] = item
        except Exception:
            pass

    archivos = []
    for nombre, fuente in descripciones.items():
        p = carpeta / nombre
        if not p.is_file():
            if nombre in prev_entries:
                archivos.append(prev_entries[nombre])
            continue
        df = pd.read_csv(p)
        archivos.append({
            "bytes": p.stat().st_size,
            "created_at": prev_entries.get(nombre, {}).get("created_at", hoy),
            "file": nombre,
            "rows": int(len(df)),
            "sha256": _sha256(p),
            "source": fuente,
        })
    man = {
        "archivos": archivos,
        "generado_por": "scripts/construir_corpus_bancos_centrales.py",
        "version": 1,
    }
    manifest_path.write_text(json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return man


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, default=2008, help="año inicial (por omisión: 2008)")
    ap.add_argument("--end", type=int, default=2026, help="año final (por omisión: 2026)")
    ap.add_argument("--destino", type=Path, default=DESTINO, help="carpeta de salida")
    args = ap.parse_args(argv)

    args.destino.mkdir(parents=True, exist_ok=True)
    df_bx, df_fed = construir_corpus(start=args.start, end=args.end)

    ruta_bx = args.destino / "banxico_comunicados.csv"
    ruta_fed = args.destino / "fomc_statements.csv"
    df_bx.to_csv(ruta_bx, index=False)
    df_fed.to_csv(ruta_fed, index=False)

    man = actualizar_manifiesto(args.destino)
    print(f"banxico_comunicados.csv: {len(df_bx)} filas ({df_bx['date'].min()} a {df_bx['date'].max()})")
    print(f"fomc_statements.csv:     {len(df_fed)} filas ({df_fed['date'].min()} a {df_fed['date'].max()})")
    print(f"manifest.json:           {len(man['archivos'])} archivos registrados")
    return 0


if __name__ == "__main__":
    sys.exit(main())
