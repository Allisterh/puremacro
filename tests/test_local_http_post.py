# tests/test_local_http_post.py
"""post_json: urllib JSON POST used by the local-LLM HTTP engine."""
import http.server
import json
import threading

import pytest

from puremacro._http import post_json


class _EchoHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        sent = json.loads(self.rfile.read(n) or b"{}")
        out = json.dumps({"echo": sent, "path": self.path}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


@pytest.fixture
def echo_server():
    srv = http.server.HTTPServer(("127.0.0.1", 0), _EchoHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_post_json_roundtrip(echo_server):
    out = post_json(f"{echo_server}/x", {"a": 1, "b": "hi"}, timeout=5)
    assert out["path"] == "/x"
    assert out["echo"] == {"a": 1, "b": "hi"}


def test_post_json_does_not_retry_timeouts(monkeypatch):
    import urllib.error
    import urllib.request

    calls = []

    def fake_urlopen(req, timeout=None, **kw):
        calls.append((req.full_url, timeout, kw.get("context")))
        raise urllib.error.URLError(TimeoutError("timed out"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(urllib.error.URLError):
        post_json("https://example.org/v1/chat/completions", {"a": 1}, timeout=120.0)
    assert len(calls) == 1


def test_post_json_https_ssl_fallback_caps_retry_timeout(monkeypatch):
    import io
    import ssl
    import urllib.request
    from puremacro._http import DEFAULT_TIMEOUT

    calls = []

    class _FakeResp:
        def __enter__(self):
            return io.BytesIO(b'{"ok": true}')

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None, **kw):
        calls.append((req.full_url, timeout, kw.get("context")))
        if len(calls) == 1:
            raise ssl.SSLError("certificate verify failed")
        return _FakeResp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    out = post_json("https://example.org/v1/chat/completions", {"a": 1}, timeout=120.0)
    assert out == {"ok": True}
    assert len(calls) == 2
    assert calls[0][1] == 120.0 and calls[0][2] is None
    assert calls[1][1] == DEFAULT_TIMEOUT and calls[1][2] is not None

