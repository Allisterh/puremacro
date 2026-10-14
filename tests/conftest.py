"""Shared toy datasets for puremacro tests."""
import uuid

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _isolated_llm_cache(monkeypatch, tmp_path_factory):
    """Keep every test off the user's real LLM caches, mode, model and server.

    There are two caches in two files. ``puremacro.llm`` reads and writes the
    completion cache ``$PUREMACRO_LLM_CACHE`` (default
    ``~/.cache/puremacro/llm_cache.sqlite``); ``llm_prob_kernel`` reads and
    writes the narrative score cache ``$PUREMACRO_LLM_SCORES_CACHE`` (default
    ``~/.cache/puremacro/llm_scores.sqlite``). Each gets a file of its own,
    different for every test. Without this a test that completes a prompt or
    scores a document would leave rows in a real cache, and a cache hit left
    there by an earlier run would stand in for the engine a later test means
    to exercise.

    Both variables are set whatever the shell exports. A developer who points
    one at a cache worth keeping (to build one to ship, say) would otherwise
    get a test's fake completions or scores in it, and later tests would find
    them there. ``monkeypatch`` puts the developer's values back when the test
    ends.

    The files sit under the session's base temp directory but not in
    ``tmp_path``. ``tmp_path`` makes a numbered directory for every test, and
    pytest scans the whole base directory for the next number each time, so
    asking for it here would make the cost of a run grow with the square of
    the number of tests. A random name keeps each test's files its own, and no
    directory is made: both caches create their parent directory when they
    first open their file (``CompletionCache`` in ``_connect``, the score
    kernel just before it connects), so only a test that uses a cache touches
    the disk.

    The other three variables are removed: ``PUREMACRO_LLM_MODE`` (auto, cache
    or off), ``PUREMACRO_LLM_MODEL`` (what the ``default`` model means) and
    ``PUREMACRO_LLM_BASE_URL`` (the server the HTTP engines talk to). A test
    that wants one sets it with ``monkeypatch.setenv``; a value exported in the
    developer's shell would otherwise change which model, engine or server a
    test exercises.
    """
    base = tmp_path_factory.getbasetemp() / "llm_cache"
    monkeypatch.setenv("PUREMACRO_LLM_CACHE", str(base / f"{uuid.uuid4().hex}.sqlite"))
    monkeypatch.setenv("PUREMACRO_LLM_SCORES_CACHE", str(base / f"{uuid.uuid4().hex}_scores.sqlite"))
    monkeypatch.delenv("PUREMACRO_LLM_MODE", raising=False)
    monkeypatch.delenv("PUREMACRO_LLM_MODEL", raising=False)
    monkeypatch.delenv("PUREMACRO_LLM_BASE_URL", raising=False)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def toy_var2(rng):
    """Stationary 2-variable VAR(1): y_t = A y_{t-1} + u_t."""
    T, n = 200, 2
    A = np.array([[0.5, 0.1], [0.0, 0.6]])
    Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
    L = np.linalg.cholesky(Sigma)
    Y = np.zeros((T, n))
    for t in range(1, T):
        Y[t] = A @ Y[t-1] + L @ rng.standard_normal(n)
    return pd.DataFrame(Y, columns=["y1", "y2"])


@pytest.fixture
def toy_panel(rng):
    """Toy panel: 5 entities x 60 periods, single LP-friendly outcome."""
    rows = []
    for c in ("A", "B", "C", "D", "E"):
        x = rng.standard_normal(60)
        y = 0.3 * np.roll(x, 1) + 0.5 * rng.standard_normal(60)
        for t, (xv, yv) in enumerate(zip(x, y)):
            rows.append({"code": c, "t": t, "x": xv, "y": yv})
    df = pd.DataFrame(rows)
    return df.set_index(["code", "t"]).sort_index()


# ---------------------------------------------------------------------------
# Shared HTTP mock for CB-connector tests (Slice 1 + Slice 3)
# ---------------------------------------------------------------------------
import importlib


# Modules whose `safe_get_bytes` / `safe_get_text` we patch in offline
# CB-connector tests. Slice 1 + Slice 3 banks combined.
_CB_PATCH_TARGETS = [
    "puremacro.narrative.sources._rss",
    "puremacro.narrative.sources._ratedoc",
    "puremacro.narrative.sources._speeches",
    "puremacro.narrative.sources.fed_decision",
    "puremacro.narrative.sources.fed_minutes",
    "puremacro.narrative.sources.fed_press_conf",
    "puremacro.narrative.sources.fed_speeches",
    "puremacro.narrative.sources.ecb_decision",
    "puremacro.narrative.sources.ecb_minutes",
    "puremacro.narrative.sources.ecb_press_conf",
    "puremacro.narrative.sources.ecb_speeches",
    "puremacro.narrative.sources.boe_decision",
    "puremacro.narrative.sources.boe_minutes",
    "puremacro.narrative.sources.boe_speeches",
    "puremacro.narrative.sources.boj_decision",
    "puremacro.narrative.sources.boj_speeches",
    # Slice 3
    "puremacro.narrative.sources.banxico",
    "puremacro.narrative.sources.bcb",
    "puremacro.narrative.sources.bccl",
    "puremacro.narrative.sources.bcra",
    "puremacro.narrative.sources.banrep",
    "puremacro.narrative.sources.rba",
    "puremacro.narrative.sources.rbnz",
    "puremacro.narrative.sources.riksbank",
    "puremacro.narrative.sources.norges",
    "puremacro.narrative.sources.sarb",
    "puremacro.narrative.sources.pboc",
    "puremacro.narrative.sources.rbi",
    "puremacro.narrative.sources.bok",
    "puremacro.narrative.sources.mas",
    "puremacro.narrative.sources.bot",
    "puremacro.narrative.sources.bis_speeches",
    # BLS state-panel fetchers (Task 2 — Notebook 29)
    "puremacro.fetch.bls_state_panel",
    # Beige Book connector (Stream 4 — BBUI district-level)
    "puremacro.narrative.sources.beige_book",
]


@pytest.fixture
def mock_http(monkeypatch):
    """Per-test in-memory HTTP mock. Use ``register(bytes_=..., text=...)``
    to register URL → payload mappings before invoking the connector.
    Unregistered URLs raise ``LookupError`` (loud failure on missing mocks).
    """
    by_url_bytes: dict[str, bytes] = {}
    by_url_text: dict[str, str] = {}

    def _fake_bytes(url, **_kw):
        if url in by_url_bytes:
            return by_url_bytes[url]
        raise LookupError(f"mock_http: no bytes registered for {url}")

    def _fake_text(url, **_kw):
        if url in by_url_text:
            return by_url_text[url]
        raise LookupError(f"mock_http: no text registered for {url}")

    for modname in _CB_PATCH_TARGETS:
        try:
            mod = importlib.import_module(modname)
        except ImportError:
            continue
        if hasattr(mod, "safe_get_bytes"):
            monkeypatch.setattr(mod, "safe_get_bytes", _fake_bytes)
        if hasattr(mod, "safe_get_text"):
            monkeypatch.setattr(mod, "safe_get_text", _fake_text)

    def register(*, bytes_=None, text=None):
        if bytes_:
            by_url_bytes.update(bytes_)
        if text:
            by_url_text.update(text)

    return register


# ---------------------------------------------------------------------------
# Stub local-LLM servers for the HTTP engine of puremacro.llm
# ---------------------------------------------------------------------------
import http.server
import json
import threading


class _LLMStub:
    """What a test sees of the server of ``llm_stub_server``.

    The server answers on 127.0.0.1 like Ollama and like an OpenAI-compatible
    server: ``GET /api/tags`` and ``GET /v1/models`` list one model,
    ``POST /api/chat`` replies with ``chat_content`` and
    ``POST /v1/chat/completions`` with ``score_content``; any other path gets a
    404. A test changes the attributes to change what the server does.

    Attributes
    ----------
    url : str
        ``http://127.0.0.1:<port>``, with no trailing slash.
    last_body : object
        The decoded JSON body of the last POST; None before the first.
    chat_content, score_content : str or None
        The ``content`` of the two chat replies (None is sent as JSON null).
        They start as ``"[]"`` and ``"0.7"``: an empty list of events and a score.
    probe_reply : tuple of (int, str, bytes) or None
        When set, the ``(status, content type, body)`` both model lists answer
        with instead: a server that is up but is not an LLM server.
    chat_reply : tuple of (int, str, bytes) or None
        When set, the ``(status, content type, body)`` both chat endpoints
        answer with instead: an error such as Ollama's 404 for a model that
        was never pulled.
    delay : float
        Seconds every request waits before it is answered, for a probe that
        has to give up. Teardown releases a request that is still waiting.
    """

    def __init__(self):
        self.url = ""
        self.last_body = None
        self.chat_content = "[]"
        self.score_content = "0.7"
        self.probe_reply = None
        self.chat_reply = None
        self.delay = 0.0
        self.release = threading.Event()


class _StubHTTPServer(http.server.ThreadingHTTPServer):
    """One thread per request, none waited for at teardown, and no traceback
    when a client gives up on a request (a probe that timed out)."""

    daemon_threads = True
    block_on_close = False

    def handle_error(self, request, client_address):
        pass


def _llm_stub_handler(stub):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _answer(self, status, content_type, body):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj):
            self._answer(200, "application/json", json.dumps(obj).encode("utf-8"))

        def do_GET(self):
            stub.release.wait(stub.delay)
            if self.path not in ("/api/tags", "/v1/models"):
                self.send_error(404)
            elif stub.probe_reply is not None:
                self._answer(*stub.probe_reply)
            elif self.path == "/api/tags":
                self._json({"models": [{"name": "qwen3:4b"}]})
            else:
                self._json({"object": "list", "data": [{"id": "qwen3-4b"}]})

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            stub.last_body = json.loads(raw) if raw else None
            stub.release.wait(stub.delay)
            if self.path not in ("/api/chat", "/v1/chat/completions"):
                self.send_error(404)
            elif stub.chat_reply is not None:
                self._answer(*stub.chat_reply)
            elif self.path == "/api/chat":
                self._json({"message": {"role": "assistant", "content": stub.chat_content}})
            else:
                self._json({"choices": [{"message": {"role": "assistant",
                                                     "content": stub.score_content}}]})

    return Handler


class _NotFoundHandler(http.server.BaseHTTPRequestHandler):
    """A 404 for every request: an open port that is not an LLM server."""

    def log_message(self, *args):
        pass

    def _not_found(self):
        self.send_error(404)

    do_GET = do_POST = do_HEAD = _not_found


def _serve(handler):
    """Serve ``handler`` on a free port of 127.0.0.1; return the server, its thread, its URL."""
    server = _StubHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05},
                              daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{server.server_address[1]}"


def _stop(server, thread):
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


@pytest.fixture
def llm_stub_server():
    """A stub Ollama and OpenAI-compatible server; ``_LLMStub`` says what it answers."""
    stub = _LLMStub()
    server, thread, stub.url = _serve(_llm_stub_handler(stub))
    try:
        yield stub
    finally:
        stub.release.set()
        _stop(server, thread)


@pytest.fixture
def llm_stub_404_server():
    """The URL of a server that answers every request with a 404."""
    server, thread, url = _serve(_NotFoundHandler)
    try:
        yield url
    finally:
        _stop(server, thread)
