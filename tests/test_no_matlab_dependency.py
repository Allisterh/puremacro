"""puremacro must reproduce its documented results with Python alone.

Before 4.0.0 three separate things tied the project to MATLAB:

* the trade model read its calibration data from outside the distribution,
  so its suites silently skipped on every other machine, and CI reported
  errors rather than skips;
* ``puremacro.dsge`` read Dynare's ``*_results.mat`` through ``scipy.io``;
* a ``matlab/`` companion toolbox shipped a second, unaudited implementation.

All three are gone; the trade model reads NumPy tables only. These tests
fail if any of them comes back, because the failure mode is
quiet: a module grows a ``scipy.io`` import, or a loader
reaches outside the distribution, and nobody notices until a stranger tries to
reproduce a published number.
"""
from __future__ import annotations

import importlib
import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Subpackages that are shipped samples rather than library code.
_NON_LIBRARY = (".tests", ".examples", ".teaching")


def test_importing_every_module_never_loads_scipy_io():
    """No library module may import ``scipy.io``, directly or transitively."""
    code = "\n".join([
        "import importlib, pkgutil, sys, warnings",
        "warnings.simplefilter('ignore')",
        "import puremacro",
        f"skip = {_NON_LIBRARY!r}",
        "for m in pkgutil.walk_packages(puremacro.__path__, 'puremacro.'):",
        "    if any(s in m.name for s in skip):",
        "        continue",
        "    try:",
        "        importlib.import_module(m.name)",
        "    except Exception:",
        "        pass",
        "print('scipy.io' in sys.modules)",
    ])
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().endswith("False"), (
        "A puremacro module imports scipy.io again. puremacro reads no MATLAB "
        "files; a caller who has one loads it and passes the mapping.\n"
        f"stdout: {proc.stdout[-500:]}"
    )


def test_no_matlab_sources_are_distributed():
    """Runtime sources carry no MATLAB files; development references are explicit."""
    tracked = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    if tracked.returncode != 0:  # not a git checkout, e.g. an unpacked sdist
        pytest.skip("not a git checkout")
    # These external development references live outside the runtime package.
    # Ordinary tests use frozen exports; reproducing them may use MATLAB/Octave.
    # Keep an exact allowlist: a new runtime dependency must still fail here.
    development_references = {
        "tools/reference_validation/export_dynare.m",
        "tests/fixtures/arellano_2008/original/markovchain.m",
        "tests/fixtures/arellano_2008/original/policies_simulation.m",
        "curso/notebooks/modelos/rbc_mexico_dual/Output/rbc_mexico_dual_results.mat",
    }
    offenders = [
        line for line in tracked.stdout.split("\n")
        if line.endswith((".m", ".mat", ".mlx"))
        and line not in development_references
    ]
    assert not offenders, (
        "MATLAB files are tracked again: " + ", ".join(sorted(offenders)[:10])
    )


def test_dynare_helpers_take_a_mapping_not_a_path():
    """The Dynare bridge consumes an ``oo_`` mapping the caller loaded."""
    from puremacro.dsge import load_dynare_dr, load_dynare_moments

    for fn in (load_dynare_dr, load_dynare_moments):
        with pytest.raises(TypeError, match=r"(?i)no longer reads MATLAB|loadmat"):
            fn("some_results.mat")

    # A mapping without oo_ is the caller's mistake, and still a KeyError.
    with pytest.raises(KeyError, match=r"(?i)oo_"):
        load_dynare_dr({"unrelated": 1})


def test_removed_module_name_is_gone():
    """``puremacro.dsge.load_dynare`` was renamed; the old path must not resolve."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("puremacro.dsge.load_dynare")


def test_trade_mat_input_is_refused_with_a_migration_message():
    """A ``.mat`` path given to the trade loader raises and names what to do instead."""
    from puremacro.trade.data import load_icio_data

    with pytest.raises(ValueError, match=r"(?i)no longer supported|savez_compressed"):
        load_icio_data(path="anything.mat")
