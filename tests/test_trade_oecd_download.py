"""Automatic download of the OECD ICIO archive (offline: the HTTP layer is patched)."""
import urllib.error

import pytest

import puremacro
from puremacro import _http
from puremacro.trade import _oecd_icio

ZIP_BYTES = b"PK\x03\x04" + b"\x00" * 60


def _forbidden(url):
    return urllib.error.HTTPError(url, 403, "Forbidden", hdrs=None, fp=None)


def test_download_falls_back_through_request_forms_and_caches(tmp_path, monkeypatch):
    calls = []

    def fake_get(url, timeout=0, *, user_agent=None, headers=None):
        calls.append((user_agent, headers))
        if not headers:  # the CDN refuses the plain scripted GET
            raise _forbidden(url)
        return ZIP_BYTES

    monkeypatch.setattr(_http, "safe_get_bytes", fake_get)
    path = _oecd_icio.download_archive(tmp_path)
    assert path == tmp_path / _oecd_icio.ARCHIVE_NAME
    assert path.read_bytes() == ZIP_BYTES
    assert [h for _, h in calls] == [None, {"Range": "bytes=0-"}]
    # Every request names puremacro; none imitates a browser.
    for user_agent, _ in calls:
        assert user_agent.startswith(f"puremacro/{puremacro.__version__} ")
        assert "Mozilla" not in user_agent and "Chrome" not in user_agent
    assert not list(tmp_path.glob("*.part"))

    calls.clear()
    assert _oecd_icio.download_archive(tmp_path) == path  # cached copy, no request
    assert calls == []
    _oecd_icio.download_archive(tmp_path, refresh=True)
    assert len(calls) == 2


def test_download_failure_names_the_url_and_the_manual_route(tmp_path, monkeypatch):
    def fake_get(url, timeout=0, *, user_agent=None, headers=None):
        raise _forbidden(url)

    monkeypatch.setattr(_http, "safe_get_bytes", fake_get)
    with pytest.raises(RuntimeError) as info:
        _oecd_icio.download_archive(tmp_path)
    message = str(info.value)
    assert _oecd_icio.ARCHIVE_URL in message
    assert "browser" in message and str(tmp_path / _oecd_icio.ARCHIVE_NAME) in message
    assert not (tmp_path / _oecd_icio.ARCHIVE_NAME).exists()


def test_download_rejects_a_non_zip_response(tmp_path, monkeypatch):
    monkeypatch.setattr(_http, "safe_get_bytes",
                        lambda url, timeout=0, *, user_agent=None, headers=None: b"<html>challenge</html>")
    with pytest.raises(RuntimeError, match="did not return a ZIP"):
        _oecd_icio.download_archive(tmp_path)
    assert not (tmp_path / _oecd_icio.ARCHIVE_NAME).exists()


def test_cache_dir_honours_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("PUREMACRO_OECD_ICIO_DIR", str(tmp_path / "icio"))
    assert _oecd_icio.archive_cache_dir() == tmp_path / "icio"
    monkeypatch.delenv("PUREMACRO_OECD_ICIO_DIR")
    assert _oecd_icio.archive_cache_dir().parts[-3:] == (".cache", "puremacro", "oecd_icio")
