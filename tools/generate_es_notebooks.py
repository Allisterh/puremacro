"""Historical generator of the Spanish editions of showcase notebooks.

``generate_es_notebook`` rebuilds a ``*_es.py`` twin from its English Jupytext
source and a dictionary of translated markdown cells. The notebooks it was
written for are no longer part of this repository, so ``main`` has none to
generate. It refuses to overwrite an existing ``*_es.py`` unless
``force=True`` (``--force`` on the command line).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

def parse_jupytext(content: str) -> tuple[str, list[tuple[str, str]]]:
    cell_pattern = re.compile(r"^# %%(\s+\[markdown\])?", re.MULTILINE)
    splits = cell_pattern.split(content)
    header = splits[0]
    cells: list[tuple[str, str]] = []
    i = 1
    while i < len(splits):
        tag = splits[i]
        body = splits[i + 1] if i + 1 < len(splits) else ""
        cell_type = "markdown" if tag and "[markdown]" in tag else "code"
        cells.append((cell_type, body.strip()))
        i += 2
    return header, cells


def assemble_jupytext(header: str, cells: list[tuple[str, str]]) -> str:
    parts = [header.strip() + "\n\n"]
    for c_type, body in cells:
        if c_type == "markdown":
            parts.append(f"# %% [markdown]\n{body}\n\n")
        else:
            parts.append(f"# %%\n{body}\n\n")
    return "".join(parts).rstrip() + "\n"


#: Notebook stem -> (translated markdown cells by index, index of the exercise cell).
#: Empty: the notebooks this tool generated are no longer part of this repository.
NOTEBOOKS: dict[str, tuple[dict[int, str], int]] = {}


def generate_es_notebook(stem: str, md_dict: dict[int, str], knob_cell_idx: int,
                         *, force: bool = False) -> None:
    en_path = Path(f"notebooks/{stem}.py")
    es_path = Path(f"notebooks/{stem}_es.py")
    if es_path.exists() and not force:
        raise FileExistsError(
            f"{es_path} exists and is maintained by hand; this generator's text may be "
            "stale. Pass force=True (--force) to overwrite it.")

    header, cells = parse_jupytext(en_path.read_text(encoding="utf-8"))
    es_cells = []

    for idx, (c_type, body) in enumerate(cells):
        if c_type == "markdown":
            if idx in md_dict:
                es_cells.append(("markdown", md_dict[idx]))
            else:
                es_cells.append(("markdown", body))
        else:
            if idx == knob_cell_idx:
                # Prepend Tu turno comment
                knob_body = "# Tu turno: personaliza parámetros e inspecciona el comportamiento del modelo\n" + body
                es_cells.append(("code", knob_body))
            else:
                es_cells.append(("code", body))

    output = assemble_jupytext(header, es_cells)
    es_path.write_text(output, encoding="utf-8")
    print(f"Generated {es_path.name} with {len(es_cells)} cells ({len([c for c in es_cells if c[0] == 'code'])} code cells)")


def main(argv: list[str] | None = None) -> None:
    force = "--force" in (sys.argv[1:] if argv is None else argv)
    for stem, (md_dict, knob_cell_idx) in NOTEBOOKS.items():
        generate_es_notebook(stem, md_dict, knob_cell_idx=knob_cell_idx, force=force)
    print(f"Generated {len(NOTEBOOKS)} Spanish edition(s).")


if __name__ == "__main__":
    main()
