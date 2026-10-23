"""The released course is a complete, reproducible edition, including DP/text."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "curso" / "notebooks"
SPEC = importlib.util.spec_from_file_location("course_publication", ROOT / "curso/scripts/publicar_web.py")
assert SPEC and SPEC.loader
publication = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publication)


def _text(cell):
    return "".join(cell["source"])


@pytest.mark.parametrize("modern", [False, True])
@pytest.mark.parametrize("name", ["T03_B_garch.ipynb", "T03_B_garch_es.ipynb"])
def test_setup_rewrite_updates_both_generations_and_preserves_customizations(modern, name):
    source = '''import os
import sys
from pathlib import Path
from collections import (
    Counter,
    defaultdict,
)
import numpy as np
try:
    import math as custom_math
except ImportError:
    custom_math = None

def _safe_probe(p, is_dir=None):
    return True

if "google.colab" in sys.modules:
    !pip install -q "puremacro[io,local-llm]>=4.8.1,<5" "arch>=6" openpyxl
    # CLONE

try:
    import _nbstyle
    _nbstyle.apply_style()
except ImportError:
    pass

custom_tail = {"preserve": np.arange(4)}
'''.replace("# CLONE", "!git clone --filter=blob:none --sparse example" if modern else "# drive.mount")
    actual, warnings = publication.reescribir_celda1(source, name)
    assert not warnings
    assert "puremacro[io,local-llm]==5.5.0" in actual
    assert "course-2026A-5.5.0" in actual
    requirements = ast.literal_eval(next(line.split(" = ", 1)[1] for line in actual.splitlines()
                                        if line.startswith("_course_requirements = ")))
    assert requirements == ["puremacro[io,local-llm]==5.5.0", "arch>=6", "openpyxl"]
    assert "from collections import (\n    Counter,\n    defaultdict,\n)" in actual
    assert "import numpy as np" in actual
    assert "try:\n    import math as custom_math\nexcept ImportError:\n    custom_math = None" in actual
    assert actual.endswith('custom_tail = {"preserve": np.arange(4)}')
    assert "--branch" in actual and "course-2026A-5.5.0" in actual
    assert "check=True" in actual
    assert publication.reescribir_celda1(actual, name) == (actual, [])


def test_badges_and_setup_are_idempotent_for_every_notebook():
    paths = sorted(NOTEBOOKS.glob("*.ipynb"))
    assert len(paths) == 72
    assert len(publication.cuadernos_fuente()) == 72
    counts = {"dp": 0, "text": 0}
    for path in paths:
        nb = json.loads(path.read_text(encoding="utf-8"))
        assert nb["metadata"]["course_edition"] == publication.EDICION
        first, setup = map(_text, nb["cells"][:2])
        assert publication.reescribir_celda0(first, path.name) == first
        assert publication.reescribir_celda1(setup, path.name) == (setup, [])
        assert "puremacro==5.5.0" in setup
        ref = "main" if path.name in publication.ACTUALIZADOS else publication.EDICION
        assert f"/blob/{ref}/" in first
        assert "MAV_LLM_MODE" in setup and '"cache"' in setup
        for capability in nb["metadata"].get("course_required_capabilities", []):
            counts[capability] += 1
            required = [c for c in nb["cells"] if c.get("metadata", {}).get("course_capability") == capability]
            assert required, path.name
            assert all(c["cell_type"] == "code" for c in required)
        for cell in nb["cells"]:
            source = _text(cell)
            assert "4a726629aa8a35c1c3d762a9bbbb2b4036590252" not in source
            assert "HAY_DP" not in source and "HAY_TEXT" not in source
    assert counts == {"dp": 6, "text": 8}


def test_generated_readme_uses_same_edition_and_extras_pin(tmp_path, monkeypatch):
    monkeypatch.setattr(publication, "DEST_NB", tmp_path)
    assert publication.construir_leeme()
    assert not publication.construir_leeme()
    actual = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert actual == (NOTEBOOKS / "README.md").read_text(encoding="utf-8")
    assert "puremacro[io]==5.5.0" in actual
    assert "puremacro[local-llm]==5.5.0" in actual
    assert "git clone --branch course-2026A-5.5.0" in actual
    amended_links = set(re.findall(r"/blob/main/curso/notebooks/([^()\s]+\.ipynb)", actual))
    assert amended_links == publication.ACTUALIZADOS


def test_course_amendments_have_matching_execution_evidence():
    """A mutable lesson link requires its own source and executed-output record."""
    evidence = json.loads((ROOT / "curso/validation/rbc-uncertainty-2026-10-22.json").read_text(encoding="utf-8"))
    assert evidence["puremacro"] == "5.5.0"
    assert {row["notebook"] for row in evidence["notebooks"]} == publication.ACTUALIZADOS
    for row in evidence["notebooks"]:
        path = NOTEBOOKS / row["notebook"]
        nb = json.loads(path.read_text(encoding="utf-8"))
        assert nb["metadata"]["course_edition"] == publication.EDICION
        assert nb["metadata"]["course_amendment"] == "2026-10-22-rbc-uncertainty"
        source = [(c["cell_type"], _text(c)) for c in nb["cells"]]
        assert hashlib.sha256(json.dumps(source, ensure_ascii=False).encode()).hexdigest() == row["source_sha256"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["output_sha256"]
        assert row["ok"] and not row["errors"] and not row["warnings"]
        assert row["girf_tables_equal_5_3_0_reference"]
        code = [c for c in nb["cells"] if c["cell_type"] == "code"]
        assert len(code) == row["code_cells"] == 20
        assert all(c["execution_count"] is not None for c in code)


@pytest.mark.mechanism_control
def test_execution_network_blocker_rejects_actual_socket_operations():
    """The runner's audit hook bites before either DNS lookup or TCP connect."""
    import subprocess
    import sys

    spec = importlib.util.spec_from_file_location("course_runner", ROOT / "curso/scripts/ejecutar_edicion.py")
    assert spec and spec.loader
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    preamble = runner.PREAMBLE
    hook = preamble[preamble.index("_audit_network_attempts = []"):]
    control = "import sys as _audit_sys, socket\n" + hook + '''
for probe in (lambda: socket.getaddrinfo("localhost", 9),
              lambda: socket.socket().connect(("127.0.0.1", 9))):
    try:
        probe()
    except RuntimeError as error:
        assert "network is disabled" in str(error)
    else:
        raise AssertionError("Offline execution hook did not block a socket operation")
assert _audit_network_attempts == ["socket.getaddrinfo", "socket.connect"]
'''
    subprocess.run([sys.executable, "-c", control], check=True, capture_output=True, text=True)


@pytest.mark.slow
@pytest.mark.mechanism_control
def test_runner_rejects_a_notebook_cell_tagged_as_expected_error(tmp_path):
    """Even Jupyter's raises-exception tag cannot turn a failed cell into a pass."""
    import nbformat
    from nbclient.exceptions import CellExecutionError
    import sys

    spec = importlib.util.spec_from_file_location("course_runner", ROOT / "curso/scripts/ejecutar_edicion.py")
    assert spec and spec.loader
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    kernel = tmp_path / "kernels/course-wheel"
    kernel.mkdir(parents=True)
    (kernel / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Course negative control", "language": "python"}), encoding="utf-8")
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
        'raise RuntimeError("course-negative-control")', metadata={"tags": ["raises-exception"]})])
    client = runner.notebook_client(nb, tmp_path, tmp_path, 30)
    with pytest.raises(CellExecutionError, match="course-negative-control"):
        client.execute(cwd=str(tmp_path), cleanup_kc=True)


def test_setup_keeps_unrecognized_middle_code_instead_of_silently_dropping_it():
    path = NOTEBOOKS / "T03_A_aggregate_shocks.ipynb"
    source = _text(json.loads(path.read_text(encoding="utf-8"))["cells"][1])
    customized = source.replace("%matplotlib inline", "economic_seed = 1729\n%matplotlib inline")
    actual, warnings = publication.reescribir_celda1(customized, path.name)
    assert actual == customized
    assert any("economic_seed" in warning for warning in warnings)


@pytest.mark.mechanism_control
def test_runner_evidence_is_utf8_with_a_windows_default_encoding(tmp_path, monkeypatch):
    """Evidence containing economic symbols must survive a non-UTF-8 locale."""
    spec = importlib.util.spec_from_file_location("course_runner", ROOT / "curso/scripts/ejecutar_edicion.py")
    assert spec and spec.loader
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    original_write = Path.write_text

    def windows_default(path, data, encoding=None, errors=None, newline=None):
        return original_write(path, data, encoding=encoding or "cp1252", errors=errors, newline=newline)

    monkeypatch.setattr(Path, "write_text", windows_default)
    path = tmp_path / "evidence.json"
    # Positive control: the simulated host default cannot encode this payload.
    with pytest.raises(UnicodeEncodeError):
        path.write_text("π: 财富")
    payload = {"model": "Huggett", "check": "π: 财富"}
    runner.write_json(path, payload)
    assert json.loads(path.read_bytes().decode("utf-8")) == payload


@pytest.mark.mechanism_control
@pytest.mark.parametrize("name", ["T01_B_cross_country_facts.ipynb",
                                  "T01_B_hechos_estilizados_internacionales_es.ipynb"])
def test_qna_export_keeps_frozen_inputs_and_checks_numeric_round_trip(name, tmp_path, monkeypatch, capsys):
    """Execute the actual lesson export cell without touching the frozen edition."""
    import hashlib
    import numpy as np
    import pandas as pd
    from puremacro import fetch

    nb = json.loads((NOTEBOOKS / name).read_text(encoding="utf-8"))
    cells = [_text(c) for c in nb["cells"] if c["cell_type"] == "code" and ".to_csv(" in _text(c)]
    assert len(cells) == 1
    code = compile(cells[0], name + " export cell", "exec")
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data_curso" / "bundle_2026A"
    data.mkdir(parents=True)
    filenames = {"qna_gasto.csv", "qna_renta.csv", "qna_produccion.csv",
                 "qna_mercado_laboral.csv", "qna_meta.csv"}
    for filename in filenames:
        (data / filename).write_bytes(b"frozen course edition\n")
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in data.iterdir()}
    index = pd.MultiIndex.from_product(
        [["MEX", "USA"], pd.date_range("2020-01-01", periods=3, freq="QS")], names=["code", "date"])
    values = np.arange(len(index), dtype=float)
    panel = pd.DataFrame({"gdp": 100.0 + values, "gdp_defl": 100.0,
                          "gdp_real": 100.0 + values, "cons_hh": 50.0 + values,
                          "cons_hh_defl": 100.0, "va_total": 90.0 + values,
                          "va_total_defl": 100.0, "comp_emp": 30.0 + values}, index=index)
    namespace = {"Path": Path, "pd": pd, "np": np, "DATA": data, "panel": panel,
                 "lab": pd.DataFrame({"emp": 10.0 + values, "hours_week": 40.1,
                                      "self_emp_share": 0.25}, index=index),
                 "bev": index.to_frame(index=False).assign(urate=5.0, vrate=2.0, fuente="fixture"),
                 "meta": pd.DataFrame({"code": ["MEX", "USA"]})}
    namespace.update({key: getattr(fetch, key) for key in
                      ("QNA_COMPONENTS", "QNA_ASSETS", "QNA_DURABILITY", "QNA_INCOME", "QNA_ACTIVITIES")})
    exec(code, namespace)
    assert {p.name for p in (tmp_path / "generated/qna").iterdir()} == filenames
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in data.iterdir()} == before
    expected = "Ida y vuelta verificada" if name.endswith("_es.ipynb") else "Round trip verified"
    assert expected in capsys.readouterr().out
    # A broken nominal/real identity must fail the lesson's original assertion.
    # This control prevents a skipped export/readback branch from passing.
    namespace["panel"] = panel.assign(gdp_real=panel["gdp_real"] * 2.0)
    with pytest.raises(AssertionError):
        exec(code, namespace)
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in data.iterdir()} == before


@pytest.mark.mechanism_control
def test_cached_course_status_ignores_generated_exports_but_detects_frozen_edits(tmp_path):
    """A Colab rerun accepts its own exports and still rejects edited inputs."""
    import subprocess

    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args], check=True,
                              capture_output=True, text=True).stdout

    git("init", "-q")
    (tmp_path / ".gitignore").write_bytes((ROOT / ".gitignore").read_bytes())
    course = tmp_path / "curso/notebooks"
    course.mkdir(parents=True)
    frozen = course / "frozen.csv"
    frozen.write_text("original data\n", encoding="utf-8")
    git("add", ".")
    git("-c", "user.name=Course control", "-c", "user.email=course-control@example.invalid",
        "-c", f"core.hooksPath={tmp_path / 'empty-hooks'}", "-c", "commit.gpgsign=false",
        "commit", "-qm", "Frozen edition fixture")
    generated = course / "generated/qna"
    generated.mkdir(parents=True)
    (generated / "qna_gasto.csv").write_text("generated data\n", encoding="utf-8")
    status_args = ("status", "--porcelain", "--untracked-files=normal", "--", "curso/notebooks")
    assert git(*status_args) == ""
    frozen.write_text("modified data\n", encoding="utf-8")
    assert "frozen.csv" in git(*status_args)
