#!/usr/bin/env python3
"""Execute the complete course against an installed wheel, outside its source tree.

Install the candidate wheel and notebook dependencies in a dedicated environment,
then run this script with that environment's Python (PYTHONPATH must be unset):

    python curso/scripts/ejecutar_edicion.py --wheel /path/puremacro-5.5.0-py3-none-any.whl \
        --work-dir /tmp/course-5.5-validation --jobs 2 --write-back

The work directory must be new. All 72 notebooks, including historical names and
T00_B, are the default. --notebook accepts exact names for a documented retry.
The runner copies course files, preserves all notebook cells, uses a kernel from
the same environment, blocks network access, forces cached LLM/frozen data modes,
and records the wheel, installed import, input hashes and every cell's execution.
The optional live-data and live-model paths are deliberately not exercised.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import zipfile

VERSION = "5.5.0"
EDITION = "course-2026A-5.5.0"
MARKER = "COURSE_EXECUTION_EVIDENCE="


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def verify_installed_wheel(wheel: Path) -> dict:
    """Reject an editable install, a different wheel and source-tree shadowing."""
    if os.environ.get("PYTHONPATH"):
        raise RuntimeError("Unset PYTHONPATH: this audit must import the installed wheel.")
    import puremacro

    module = Path(puremacro.__file__).resolve()
    if not module.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(f"puremacro is not installed in this environment: {module}")
    dist = importlib.metadata.distribution("puremacro")
    if dist.version != VERSION or puremacro.__version__ != VERSION:
        raise RuntimeError(f"Expected puremacro=={VERSION}; imported {puremacro.__version__}")
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    expected = sha256(wheel)
    installed_hash = direct.get("archive_info", {}).get("hashes", {}).get("sha256")
    if installed_hash is not None and installed_hash != expected:
        raise RuntimeError("The installed wheel SHA-256 differs from the supplied archive.")
    if direct.get("dir_info", {}).get("editable"):
        raise RuntimeError("An editable installation cannot certify a wheel.")
    # uv may omit archive_info.hashes. Compare installed members with the wheel
    # itself, including package data and distribution metadata. RECORD changes
    # during installation and is the only excluded archive member.
    verified = 0
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            if member.is_dir() or member.filename.endswith(".dist-info/RECORD"):
                continue
            installed = Path(dist.locate_file(member.filename))
            if not installed.is_file() or installed.read_bytes() != archive.read(member):
                raise RuntimeError("Installed file differs from the supplied wheel: " + member.filename)
            verified += 1
    if not verified:
        raise RuntimeError("The supplied wheel contains no verifiable files.")
    return {"version": dist.version, "module": str(module), "wheel": wheel.name,
            "verified_installed_files": verified,
            "wheel_sha256": expected, "python": sys.version, "executable": sys.executable,
            "dependencies": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}}


PREAMBLE = '''import sys as _audit_sys, os as _audit_os, json as _audit_json
from pathlib import Path as _AuditPath
import puremacro as _audit_pm
assert _audit_pm.__version__ == "5.5.0"
assert _AuditPath(_audit_pm.__file__).resolve().is_relative_to(_AuditPath(_audit_sys.prefix).resolve())
assert not _audit_os.environ.get("PYTHONPATH")
_audit_network_attempts = []
def _audit_offline(event, args):
    if event in {"socket.connect", "socket.getaddrinfo"}:
        _audit_network_attempts.append(event)
        raise RuntimeError("Course validation uses frozen data and cached LLM responses; network is disabled.")
_audit_sys.addaudithook(_audit_offline)
'''
POSTAMBLE = '''_audit_capabilities = [name for name in ("dp", "text") if "puremacro." + name in _audit_sys.modules]
_audit_llm = _audit_sys.modules.get("_llm")
_audit_misses = list(getattr(_audit_llm, "MISSES", []))
_audit_optional = {name: bool(globals()[name]) for name in ("VIVO",) if name in globals()}
if "HAY_PIB" in globals():
    _audit_optional["countries_with_gdp"] = sorted(HAY_PIB)
if "df_kaldor" in globals():
    _audit_optional["raw_euklems"] = "labelled-representative-values"
elif "df_klems" in globals():
    _audit_optional["raw_euklems"] = "local-data"
_audit_evidence = {"module": _audit_pm.__file__, "version": _audit_pm.__version__,
                  "capabilities": _audit_capabilities, "llm_mode": _audit_os.environ["MAV_LLM_MODE"],
                  "llm_cache_misses": _audit_misses, "network_attempts": _audit_network_attempts,
                  "optional_paths": _audit_optional}
print("COURSE_EXECUTION_EVIDENCE=" + _audit_json.dumps(_audit_evidence))
assert not _audit_network_attempts, _audit_network_attempts
assert _audit_llm is None or getattr(_audit_llm, "_LIBRARY", None) is not None, "Course helper silently fell back from puremacro.llm"
assert not _audit_misses, "Missing cached reference responses: " + repr(_audit_misses)
assert _audit_os.environ["MAV_LLM_MODE"] == "cache"
'''


def notebook_client(nb, workdir: Path, run: Path, timeout: int):
    """Use this environment's kernel; no notebook metadata may allow errors."""
    from nbclient import NotebookClient
    from jupyter_client import AsyncKernelManager
    from jupyter_client.kernelspec import KernelSpecManager

    specs = workdir / "kernels"
    manager = AsyncKernelManager(kernel_name="course-wheel", kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(specs)]))
    return NotebookClient(nb, km=manager, timeout=timeout, allow_errors=False,
                          force_raise_errors=True,
                          resources={"metadata": {"path": str(run)}}, record_timing=True)


def execute_one(name: str, work: str, timeout: int) -> dict:
    import nbformat
    workdir = Path(work)
    if (workdir / "STOP").exists():
        return {"notebook": name, "status": "cancelled", "reason": "STOP requested before execution"}
    snapshot = workdir / "snapshot"
    run = workdir / "runs" / Path(name).stem
    run.mkdir(parents=True)
    # Helpers and model/data directories come from the copied, hashed snapshot.
    # Per-notebook working folders isolate scratch output and writable user caches.
    for path in snapshot.iterdir():
        if path.suffix == ".ipynb":
            continue
        target = run / path.name
        if path.is_dir():
            target.symlink_to(path, target_is_directory=True)
        else:
            shutil.copy2(path, target)
    os.environ.update({"MAV_LLM_MODE": "cache", "PUREMACRO_LLM_MODE": "cache",
                       "PUREMACRO_LLM_MODEL": "qwen3-4b", "MAV_T01_VIVO": "0", "MAV_QNA_VIVO": "0",
                       "MAV_LLM_USER_CACHE": str(run / "llm-user.sqlite"),
                       "PUREMACRO_LLM_CACHE": str(run / "llm-user.sqlite"),
                       "PUREMACRO_LLM_FALLBACK_CACHE": str(snapshot / "data_curso/llm_cache.sqlite"),
                       "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(run / "mpl"),
                       "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                       "VECLIB_MAXIMUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    nb = nbformat.read(snapshot / name, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    original_count = len(nb.cells)
    required = nb.metadata.get("course_required_capabilities", [])
    nb.cells.insert(0, nbformat.v4.new_code_cell(PREAMBLE))
    nb.cells.append(nbformat.v4.new_code_cell(POSTAMBLE))
    client = notebook_client(nb, workdir, run, timeout)
    record = {"notebook": name, "input_sha256": sha256(snapshot / name),
              "required_capabilities": required, "status": "failed"}
    started = time.monotonic()
    try:
        client.execute(cwd=str(run), cleanup_kc=True)
        evidence = next(output.text[len(MARKER):].strip() for output in nb.cells[-1].outputs
                        if output.output_type == "stream" and output.text.startswith(MARKER))
        record["runtime"] = json.loads(evidence)
        if not set(required) <= set(record["runtime"]["capabilities"]):
            raise AssertionError("A required DP/text capability did not execute")
        code_cells = [cell for cell in nb.cells[1:-1] if cell.cell_type == "code" and cell.source.strip()]
        if any(cell.execution_count is None for cell in code_cells):
            raise AssertionError("A nonempty code cell did not execute")
        record["status"] = "passed"
    except Exception:
        record["error"] = traceback.format_exc()
    finally:
        record["seconds"] = round(time.monotonic() - started, 3)
        record["cells"] = [{"index": i, "type": c.cell_type,
                             "execution_count": c.get("execution_count"),
                             "capability": c.metadata.get("course_capability"),
                             "optional_path": c.metadata.get("course_optional_path"),
                             "source_sha256": hashlib.sha256(c.source.encode()).hexdigest(),
                             "output_count": len(c.get("outputs", [])),
                             "execution": c.metadata.get("execution", {})}
                            for i, c in enumerate(nb.cells[1:-1])]
        nb.cells = nb.cells[1:-1]
        assert len(nb.cells) == original_count
        for cell in nb.cells:
            if cell.cell_type == "code" and cell.execution_count is not None:
                cell.execution_count -= 1
                for out in cell.outputs:
                    if out.output_type == "execute_result" and out.get("execution_count") is not None:
                        out.execution_count -= 1
        output = workdir / "executed" / name
        nbformat.write(nb, output)
        record["output_sha256"] = sha256(output)
        write_json(workdir / "evidence" / (Path(name).stem + ".json"), record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1] / "notebooks")
    parser.add_argument("--notebook", action="append", default=[], help="Exact filename; default all 72")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=1800, help="Seconds per cell; no economic tolerances changed")
    parser.add_argument("--write-back", action="store_true", help="Copy successful executed notebooks to --source")
    args = parser.parse_args()
    provenance = verify_installed_wheel(args.wheel.resolve())
    source = args.source.resolve()
    work = args.work_dir.resolve()
    if work.is_relative_to(source.parents[1]):
        raise ValueError("The execution directory must be outside the repository")
    work.mkdir(parents=True, exist_ok=False)
    snapshot = work / "snapshot"
    shutil.copytree(source, snapshot,
                    ignore=shutil.ignore_patterns("__pycache__", ".ipynb_checkpoints", "generated"))
    names = args.notebook or sorted(p.name for p in snapshot.glob("*.ipynb"))
    if not args.notebook and len(names) != 72:
        raise ValueError(f"Expected all 72 notebooks, found {len(names)}")
    for name in names:
        if Path(name).name != name or not (snapshot / name).is_file():
            raise ValueError(f"Invalid notebook name: {name}")
    manifest = {str(p.relative_to(snapshot)): sha256(p) for p in sorted(snapshot.rglob("*")) if p.is_file()}
    for subdir in ("executed", "evidence", "kernels/course-wheel"):
        (work / subdir).mkdir(parents=True)
    write_json(work / "kernels/course-wheel/kernel.json", {
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Course installed wheel", "language": "python"})
    write_json(work / "inputs.json", {"edition": EDITION, "runner_sha256": sha256(Path(__file__)), "installed": provenance, "files": manifest,
                                      "selected_notebooks": names, "mode": "frozen-data/cached-llm"})
    records = []
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(execute_one, name, str(work), args.timeout): name for name in names}
        for future in as_completed(futures):
            try:
                record = future.result()
            except Exception:
                record = {"notebook": futures[future], "status": "failed", "error": traceback.format_exc()}
            records.append(record)
            print(f"{record['status']}: {record['notebook']} ({record.get('seconds', 0):.1f}s)", flush=True)
            write_json(work / "summary.json", sorted(records, key=lambda r: r["notebook"]))
    after = {str(p.relative_to(snapshot)): sha256(p) for p in sorted(snapshot.rglob("*")) if p.is_file()}
    integrity = {"added": sorted(set(after) - set(manifest)),
                 "removed": sorted(set(manifest) - set(after)),
                 "modified": sorted(name for name in manifest.keys() & after.keys() if manifest[name] != after[name])}
    write_json(work / "snapshot_integrity.json", integrity)
    changed = any(integrity.values())
    if changed:
        print("Frozen input files changed: " + repr(integrity), file=sys.stderr)
    elif args.write_back and not (work / "STOP").exists():
        for record in records:
            if record["status"] == "passed":
                shutil.copy2(work / "executed" / record["notebook"], source / record["notebook"])
    return int(changed or any(r["status"] != "passed" for r in records))



if __name__ == "__main__":
    raise SystemExit(main())
