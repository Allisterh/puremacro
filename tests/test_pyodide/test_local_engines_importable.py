# tests/test_pyodide/test_local_engines_importable.py
"""Regression: the local-engine layer and the two LLM call sites import in a
Pyodide-like environment where the optional inference engines (mlx_lm,
llama_cpp) are ABSENT. The engines must load LAZILY (inside methods) so the
HTTP path and the rest of the package import browser-clean.

The probe runs in a fresh interpreter anchored to the tree under test: its
working directory is the repository root and that root leads ``PYTHONPATH``,
so an editable install that points at another checkout cannot stand in for
the modules being tested.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import puremacro

_PROBE = textwrap.dedent(
    """
    import sys
    _BLOCKED = {"mlx_lm", "llama_cpp", "mlx"}
    class _Blocker:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in _BLOCKED:
                raise ModuleNotFoundError("blocked (simulated Pyodide): " + name)
            return None
    sys.meta_path.insert(0, _Blocker())
    import importlib
    importlib.import_module(sys.argv[1])
    leaked = sorted(m for m in _BLOCKED if m in sys.modules)
    assert not leaked, "leaked engine deps on import path: " + repr(leaked)
    print("OK")
    """
)

_TARGETS = [
    "puremacro.narrative._local_engines",
    "puremacro.narrative.scoring",
    "puremacro.narrative.indices",
    "puremacro.llm",
    "puremacro.llm.engines",
    "puremacro.llm.models",
    "puremacro.llm.cache",
]


def _probe(code, target):
    """Run ``code`` with ``target`` as its argument in a fresh interpreter anchored to the repo."""
    root = Path(puremacro.__file__).resolve().parents[1]          # the tree under test
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        p for p in (str(root), os.environ.get("PYTHONPATH", "")) if p))
    return subprocess.run([sys.executable, "-c", code, target],
                          capture_output=True, text=True, encoding="utf-8", errors="replace",
                          env=env, cwd=root, timeout=300)


@pytest.mark.parametrize("target", _TARGETS)
def test_local_engine_imports_without_inference_deps(target):
    r = _probe(_PROBE, target)
    assert r.returncode == 0, (
        f"{target} failed to import with engines absent:\n"
        f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}"
    )
    assert r.stdout.strip().splitlines()[-1] == "OK"


@pytest.mark.mechanism_control
def test_the_engine_blocker_actually_blocks():
    """Positive control: if `_PROBE`'s finder stops biting, every assertion in
    this file passes on a host that simply has none of these engines installed,
    which proves nothing about the import guards."""
    probe = _PROBE.replace(
        "importlib.import_module(sys.argv[1])",
        "\n".join([
            "blocked = []",
            "for _m in sorted(_BLOCKED):",
            "    try:",
            "        importlib.import_module(_m)",
            "    except ModuleNotFoundError as e:",
            "        if 'simulated' in str(e):",
            "            blocked.append(_m)",
            "assert blocked == sorted(_BLOCKED), blocked",
        ]),
    )
    proc = _probe(probe, "sys")
    assert proc.returncode == 0, proc.stderr[-2000:]
