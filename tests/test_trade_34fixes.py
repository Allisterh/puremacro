"""Regression tests for the 3.4.0 pre-release GPU review findings on the spatial model.

Fast and needs no private data. The MLX case skips when mlx is not installed or
cannot be imported.

Covered findings:
- ``import puremacro.spatial`` must not import torch or mlx (lazy accelerator loading).
- ``AllenArkolakisModel`` with ``backend='mlx'`` honours ``tol``.
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from puremacro._backend import backend_available, get_array_namespace

PYTHON = sys.executable


@pytest.fixture
def importable_mlx():
    """For the MLX-only test: skip when mlx is installed but cannot be imported.

    ``backend_available`` is a cheap ``find_spec`` probe, so a broken install passes it;
    importing the namespace here turns that case into a skip rather than a failure.
    """
    try:
        get_array_namespace("mlx")
    except Exception:  # a broken install raises more than ImportError (OSError, RuntimeError)
        pytest.skip("mlx is installed but cannot be imported in this environment")


# ---------------------------------------------------------------------------
# Lazy accelerator imports
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", ["puremacro.spatial"])
def test_import_does_not_load_torch_or_mlx(module: str):
    """Importing the spatial package must not pull torch or mlx into sys.modules."""
    code = (
        "import sys\n"
        f"import {module}\n"
        "loaded = sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'mlx'))\n"
        "print('LOADED=' + ','.join(loaded))\n"
    )
    proc = subprocess.run([PYTHON, "-c", code], capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("LOADED=")][-1]
    assert line == "LOADED=", f"{module} eagerly imported: {line[len('LOADED='):]}"


# ---------------------------------------------------------------------------
# Allen-Arkolakis backend tolerance
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not backend_available("mlx"), reason="Apple MLX is not available")
@pytest.mark.usefixtures("importable_mlx")
def test_allen_arkolakis_mlx_honours_requested_tolerance():
    from puremacro.spatial.allen_arkolakis import AllenArkolakisModel

    rng = np.random.default_rng(0)
    model = AllenArkolakisModel.from_coordinates(rng.uniform(0.0, 10.0, size=(40, 2)))
    res_np = model.solve_equilibrium(tol=1e-10, backend="numpy")
    res_mx = model.solve_equilibrium(tol=1e-10, backend="mlx")
    assert res_np.converged and res_mx.converged
    assert res_mx.max_residual < 1e-10, "converged=True must mean the requested tol was met"
    np.testing.assert_allclose(res_mx.wages, res_np.wages, atol=1e-8, rtol=0)
    np.testing.assert_allclose(res_mx.population, res_np.population, atol=1e-8, rtol=0)
    assert abs(res_mx.welfare - res_np.welfare) < 1e-10
