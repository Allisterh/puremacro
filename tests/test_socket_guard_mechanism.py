"""Positive controls for the test suite's default network and credentials guards.

``tests/conftest.py`` installs two autouse guards on every test that is not
marked ``network``, ``replication`` or ``local_llm``:

* a socket guard: ``socket.socket.connect``/``connect_ex`` to a non-loopback
  address and ``socket.getaddrinfo`` of a non-local host raise an ``OSError``
  that names the guard; loopback and unix-domain sockets keep working;
* credentials isolation: ``PUREMACRO_CREDENTIALS_FILE`` points at a file that
  does not exist, the cached config of ``puremacro.credentials`` is dropped and
  every service's key variables are removed, so a developer's real keys never
  reach a test and a cache loaded by an earlier test cannot leak into a later
  one.

A guard that silently stopped working would let every offline test reach the
internet again with nothing failing, so these tests prove each guard bites and
that a marked test is left alone.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

_TESTS = Path(__file__).resolve().parent

#: TEST-NET-1 (RFC 5737): reserved for documentation, never routed. Even if the
#: guard were inert, a connect here only times out; it reaches no one.
_UNROUTABLE = ("192.0.2.1", 80)


@pytest.mark.mechanism_control
def test_unmarked_test_cannot_connect_to_a_non_loopback_address():
    with pytest.raises(OSError, match="socket guard") as info:
        socket.create_connection(_UNROUTABLE, timeout=0.5)
    assert not isinstance(info.value, TimeoutError)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(OSError, match="socket guard"):
            s.connect(_UNROUTABLE)
        with pytest.raises(OSError, match="socket guard"):
            s.connect_ex(_UNROUTABLE)
    finally:
        s.close()


@pytest.mark.mechanism_control
def test_unmarked_test_cannot_resolve_a_public_host():
    with pytest.raises(socket.gaierror, match="socket guard"):
        socket.getaddrinfo("example.com", 443)
    with pytest.raises(OSError, match="socket guard"):
        socket.create_connection(("example.com", 80), timeout=0.5)


@pytest.mark.mechanism_control
def test_blocked_attempts_are_seen_even_when_the_caller_swallows_them():
    """urllib wraps the guard's error in URLError; the message still names it."""
    import urllib.error
    import urllib.request

    with pytest.raises(urllib.error.URLError, match="socket guard"):
        urllib.request.urlopen("http://example.com/", timeout=0.5)


@pytest.mark.mechanism_control
def test_loopback_connections_still_work():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    accepted = []

    def accept():
        conn, _ = server.accept()
        accepted.append(conn.recv(4))
        conn.close()

    t = threading.Thread(target=accept, daemon=True)
    t.start()
    try:
        for host in ("127.0.0.1", "localhost"):
            assert socket.getaddrinfo(host, port)
        client = socket.create_connection(("127.0.0.1", port), timeout=5)
        client.sendall(b"ping")
        client.close()
        t.join(timeout=5)
        assert accepted == [b"ping"]
    finally:
        server.close()


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="no unix-domain sockets")
@pytest.mark.mechanism_control
def test_unix_domain_sockets_still_work():
    # A short directory: unix socket paths are limited to ~104 bytes on macOS.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="pm") as d:
        path = os.path.join(d, "s")
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(path)
        server.listen(1)
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            client.connect(path)
            conn, _ = server.accept()
            client.sendall(b"ok")
            assert conn.recv(2) == b"ok"
            conn.close()
        finally:
            client.close()
            server.close()


@pytest.mark.mechanism_control
def test_credentials_are_isolated_from_the_developer():
    from puremacro import credentials

    cfg = credentials.default_config_path()
    assert os.environ.get("PUREMACRO_CREDENTIALS_FILE") == str(cfg)
    assert not cfg.exists()
    for spec in credentials.SERVICES.values():
        for var in spec.env_vars:
            assert var not in os.environ, var
    for names in credentials.PASSWORD_ENV_VARS.values():
        for var in names:
            assert var not in os.environ, var
    assert credentials._CONFIG_CACHE in (None, {})
    assert credentials.get("census") is None
    assert credentials.get("fred") is None


@pytest.mark.mechanism_control
def test_a_cached_config_from_an_earlier_test_does_not_leak(tmp_path):
    """The order case: one test loads a config with keys, the next must not see them."""
    inner = tmp_path / "inner"
    inner.mkdir()
    (inner / "conftest.py").write_text(
        (_TESTS / "conftest.py").read_text(encoding="utf-8"), encoding="utf-8")
    cfg = tmp_path / "keys.toml"
    cfg.write_text('[census]\napi_key = "leaked"\n', encoding="utf-8")
    (inner / "test_order.py").write_text(textwrap.dedent(f"""
        from puremacro import credentials

        def test_a_loads_a_config_with_keys():
            credentials._CONFIG_CACHE = None
            import os
            os.environ["PUREMACRO_CREDENTIALS_FILE"] = {str(cfg)!r}
            assert credentials.get("census") == "leaked"
            # left behind on purpose: the cache still holds the key

        def test_b_does_not_see_them():
            assert credentials.get("census") is None
    """), encoding="utf-8")
    _run_inner(inner, tmp_path)


@pytest.mark.mechanism_control
def test_keys_are_removed_before_credentials_is_first_imported(tmp_path):
    """A test that imports puremacro for the first time must not find the shell's keys."""
    from puremacro import credentials

    names = sorted({v for spec in credentials.SERVICES.values() for v in spec.env_vars}
                   | {v for vs in credentials.PASSWORD_ENV_VARS.values() for v in vs})
    inner = tmp_path / "inner"
    inner.mkdir()
    (inner / "conftest.py").write_text(
        (_TESTS / "conftest.py").read_text(encoding="utf-8"), encoding="utf-8")
    (inner / "test_first_import.py").write_text(textwrap.dedent(f"""
        import os
        import sys

        def test_first_import_sees_no_keys():
            assert "puremacro.credentials" not in sys.modules
            leaked = [n for n in {names!r} if n in os.environ]
            assert not leaked, leaked
    """), encoding="utf-8")
    _run_inner(inner, tmp_path, extra_env={n: "leaked" for n in names})


@pytest.mark.mechanism_control
def test_marked_tests_are_not_guarded(tmp_path):
    """network, replication and local_llm tests get the real socket functions.

    Those marks are deselected by default, so the check runs in a pytest
    subprocess over a copy of the conftest. It compares identities only: no
    test here opens a connection.
    """
    inner = tmp_path / "inner"
    inner.mkdir()
    (inner / "conftest.py").write_text(
        (_TESTS / "conftest.py").read_text(encoding="utf-8"), encoding="utf-8")
    (inner / "test_marks.py").write_text(textwrap.dedent("""
        import os
        import socket
        import pytest

        REAL = (socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo)

        def _now():
            return (socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo)

        @pytest.mark.network
        def test_network():
            assert _now() == REAL

        @pytest.mark.replication
        def test_replication():
            assert _now() == REAL

        @pytest.mark.local_llm
        def test_local_llm():
            assert _now() == REAL

        def test_unmarked():
            assert all(a is not b for a, b in zip(_now(), REAL))
    """), encoding="utf-8")
    _run_inner(inner, tmp_path)


def _run_inner(inner: Path, tmp_path: Path, extra_env: dict | None = None) -> None:
    ini = inner / "pytest.ini"
    ini.write_text(
        "[pytest]\nmarkers =\n    network\n    replication\n    local_llm\n",
        encoding="utf-8")
    env = dict(os.environ)
    env.pop("PYTEST_ADDOPTS", None)
    env.update(extra_env or {})
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(inner), "-q", "-p", "no:cacheprovider",
         "-c", str(ini), "--rootdir", str(inner),
         "--basetemp", str(tmp_path / "basetemp")],
        capture_output=True, text=True, env=env, timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
