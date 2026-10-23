#!/usr/bin/env python3
"""Execute amended DSGE lessons in fresh offline IPython processes.

This records source-checkout execution, not a released-wheel certification.
No cell or economic assertion is skipped. Use --write-back only after every
selected lesson passes and the package source remains unchanged during the run.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "curso/notebooks"
STEMS = ("T02_A_neoclassical_growth_local", "T02_B_rbc_dynare_local",
         "T02_C_new_keynesian_local", "T02_D_occbin_zlb_regimes")
NAMES = tuple(f"{stem}{suffix}.ipynb" for stem in STEMS for suffix in ("", "_es"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_digest(notebook):
    content = [(cell["cell_type"], "".join(cell["source"])) for cell in notebook["cells"]]
    return hashlib.sha256(json.dumps(content, ensure_ascii=False).encode()).hexdigest()


def package_hashes():
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted((ROOT / "puremacro").rglob("*.py"))}


def relative_source_paths(text):
    """Keep diagnostics portable without publishing a developer's home path."""
    return text.replace(str(ROOT) + os.sep, "").replace(str(ROOT), "<source checkout>")


def execute_one(name, work):
    run = Path(work) / Path(name).stem
    run.mkdir()
    snapshot = Path(work) / "snapshot"
    for path in snapshot.iterdir():
        if path.suffix == ".ipynb":
            continue
        if path.is_dir():
            (run / path.name).symlink_to(path, target_is_directory=True)
        else:
            shutil.copy2(path, run / path.name)
    os.chdir(run)
    os.environ.update(MAV_LLM_MODE="cache", PUREMACRO_LLM_MODE="cache", MAV_T01_VIVO="0",
                      MAV_QNA_VIVO="0", MPLBACKEND="Agg", MPLCONFIGDIR=str(run / "mpl"),
                      IPYTHONDIR=str(run / "ipython"),
                      MAV_LLM_USER_CACHE=str(run / "llm-user.sqlite"),
                      PUREMACRO_LLM_CACHE=str(run / "llm-user.sqlite"),
                      PUREMACRO_LLM_FALLBACK_CACHE=str(snapshot / "data_curso/llm_cache.sqlite"),
                      OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      VECLIB_MAXIMUM_THREADS="1")
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(run))
    import puremacro
    from IPython.terminal.interactiveshell import TerminalInteractiveShell
    from IPython.utils.capture import capture_output
    shell = TerminalInteractiveShell.instance()
    shell.display_formatter.active_types = ["text/plain", "text/html", "image/png", "image/svg+xml"]
    shell.run_line_magic("matplotlib", "inline")
    network_attempts = []

    def offline(event, args):
        if event in {"socket.connect", "socket.getaddrinfo"}:
            network_attempts.append(event)
            raise RuntimeError("DSGE notebook validation uses frozen inputs; network is disabled")

    sys.addaudithook(offline)
    nb = json.loads((snapshot / name).read_text(encoding="utf-8"))
    record = dict(notebook=name, source_sha256=source_digest(nb), ok=False, errors=[],
                  warnings=[], figures=0, code_cells=0, seconds=0.0,
                  version=puremacro.__version__, module=str(Path(puremacro.__file__).resolve().relative_to(ROOT)), cells=[])
    started = time.perf_counter()
    for index, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        code = "".join(cell["source"])
        cell["outputs"] = []
        cell["execution_count"] = None
        before = time.perf_counter()
        with capture_output(stdout=True, stderr=True, display=True) as captured:
            result = shell.run_cell(code, store_history=False)
        for stream in ("stdout", "stderr"):
            text = relative_source_paths(getattr(captured, stream))
            if text:
                cell["outputs"].append(dict(output_type="stream", name=stream, text=text.splitlines(keepends=True)))
                if stream == "stderr":
                    record["warnings"].append(dict(cell=index, text=text))
        for output in captured.outputs:
            data = {kind: relative_source_paths(value) if kind.startswith("text/") and isinstance(value, str)
                    else value for kind, value in output.data.items()}
            cell["outputs"].append(dict(output_type="display_data", data=data, metadata=output.metadata))
            record["figures"] += int("image/png" in output.data)
        error = result.error_before_exec or result.error_in_exec
        record["cells"].append(dict(index=index, seconds=round(time.perf_counter() - before, 3), ok=error is None))
        (run / "progress.json").write_text(json.dumps(record["cells"]) + "\n")
        if error is not None:
            record["errors"].append(dict(cell=index, error=relative_source_paths(repr(error))))
            break
        record["code_cells"] += 1
        cell["execution_count"] = record["code_cells"]
    else:
        record["ok"] = not network_attempts
    record["network_attempts"] = network_attempts
    record["seconds"] = round(time.perf_counter() - started, 3)
    output_path = Path(work) / "executed" / name
    output_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    record["output_sha256"] = digest(output_path)
    (Path(work) / "evidence" / (Path(name).stem + ".json")).write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--notebook", action="append", choices=NAMES)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--write-back", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1 or args.jobs > 4:
        parser.error("--jobs must be between 1 and 4")
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    (work / "executed").mkdir()
    (work / "evidence").mkdir()
    snapshot = work / "snapshot"
    snapshot.mkdir()
    names = args.notebook or list(NAMES)
    for name in names:
        shutil.copy2(SOURCE / name, snapshot / name)
    for path in SOURCE.glob("_*.py"):
        shutil.copy2(path, snapshot / path.name)
    shutil.copytree(SOURCE / "modelos", snapshot / "modelos")
    # Immutable frozen inputs are linked; hashes before and after guard writes.
    (snapshot / "data_curso").symlink_to(SOURCE / "data_curso", target_is_directory=True)
    input_files = [*snapshot.glob("_*.py"), *sorted((snapshot / "modelos").rglob("*")),
                   *sorted((snapshot / "data_curso").rglob("*"))]
    input_hashes = {str(path.relative_to(snapshot)): digest(path) for path in input_files if path.is_file()}
    before = package_hashes()
    records = []
    import multiprocessing
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=multiprocessing.get_context("spawn"),
                             max_tasks_per_child=1) as pool:
        futures = {pool.submit(execute_one, name, str(work)): name for name in names}
        for future in as_completed(futures):
            try:
                record = future.result()
            except Exception:
                record = dict(notebook=futures[future], ok=False, errors=[relative_source_paths(traceback.format_exc())])
            records.append(record)
            print(json.dumps({key: record.get(key) for key in ("notebook", "ok", "seconds", "code_cells", "errors")}), flush=True)
    unchanged = before == package_hashes()
    unchanged_inputs = all(digest(snapshot / path) == expected and digest(SOURCE / path) == expected
                           for path, expected in input_hashes.items())
    unchanged_notebooks = all(
        source_digest(json.loads((SOURCE / name).read_text(encoding="utf-8"))) ==
        source_digest(json.loads((snapshot / name).read_text(encoding="utf-8")))
        for name in names
    )
    success = all(record["ok"] for record in records) and unchanged and unchanged_inputs and unchanged_notebooks
    manifest = dict(date="2026-10-23", scope="T02_A-D bilingual 5.7.0 release source execution",
                    puremacro="5.7.0", runtime_source="local source checkout, not a released wheel",
                    verified_platform="local Python, fresh offline IPython process per notebook",
                    python=sys.version,
                    dependencies={name: version(name) for name in
                                  ("numpy", "scipy", "pandas", "matplotlib", "ipython")},
                    package_sources=before, package_unchanged=unchanged,
                    inputs_unchanged=unchanged_inputs, input_files=input_hashes,
                    notebook_sources_unchanged=unchanged_notebooks,
                    ok=success, notebooks=sorted(records, key=lambda r: r["notebook"]))
    (work / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    if args.write_back and success:
        for name in names:
            shutil.copy2(work / "executed" / name, SOURCE / name)
    if not success:
        print(f"Execution not accepted: unchanged package={unchanged}, inputs={unchanged_inputs}, "
              f"notebook sources={unchanged_notebooks}", flush=True)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
