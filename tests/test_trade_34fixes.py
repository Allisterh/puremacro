"""Regression tests for the accelerator paths of the spatial and trade models.

Fast and needs no private data. The MLX case skips when mlx is not installed or
cannot be imported.

Pinned behaviours:
- ``import puremacro.spatial`` and ``import puremacro.trade`` must not import torch
  or mlx (lazy accelerator loading).
- ``AllenArkolakisModel`` with ``backend='mlx'`` honours ``tol``.
- ``compute_equilibrium_residuals`` rejects unknown keyword arguments.
- ``build_tariff_matrices`` honours the calibration's codes and matches the
  solver layout.
- ``puremacro.trade.gpu``: device and backend resolution (also with a torch or mlx
  that is installed but cannot be imported), cheap ``has_torch``/``has_mlx``
  probes, the GPU/MLX solvers against the NumPy solver, the macro layout
  (reduced r == w vs full) validated against the calibration, batched residuals
  equal to the serial ones, float64 Jacobians, and exact AD modes that never
  fall back silently. Accelerator cases skip when torch / mlx are missing or
  cannot be imported; the NumPy paths always run.
"""
from __future__ import annotations

import subprocess
import sys
import warnings

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

@pytest.mark.parametrize("module", ["puremacro.spatial", "puremacro.trade", "puremacro.trade.gpu"])
def test_import_does_not_load_torch_or_mlx(module: str):
    """Importing the spatial or trade packages must not pull torch or mlx into sys.modules."""
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


# ---------------------------------------------------------------------------
# Trade model: residual keywords and tariff tensors
# ---------------------------------------------------------------------------

def _make_io_table(nc: int, ns: int, nfd: int, seed: int = 0, labour_share: np.ndarray | None = None) -> np.ndarray:
    """Deterministic balanced IO table with heavier domestic blocks."""
    rng = np.random.default_rng(seed)
    M = ns * nc
    data = np.zeros((M + 3, M + nfd * nc), dtype=float)
    Z = rng.uniform(1.0, 10.0, size=(M, M))
    for c in range(nc):
        Z[c * ns : (c + 1) * ns, c * ns : (c + 1) * ns] *= 3.0
    data[:M, :M] = Z
    col = Z.sum(axis=0)
    row = Z.sum(axis=1)
    y = np.maximum(col, row) * 2.0 + rng.uniform(20.0, 60.0, size=M)
    va = y - col
    taxes = 0.05 * y
    va_fac = va - taxes
    share = np.full(M, 2.0 / 3.0) if labour_share is None else np.tile(np.asarray(labour_share, dtype=float), nc)
    data[M, :M] = taxes
    data[M + 1, :M] = share * va_fac
    data[M + 2, :M] = (1.0 - share) * va_fac
    fd_tot = y - row
    for i in range(M):
        oc = i // ns
        sh = rng.uniform(0.5, 1.5, size=nfd * nc)
        sh[oc * nfd : (oc + 1) * nfd] *= 4.0
        sh /= sh.sum()
        data[i, M:] = fd_tot[i] * sh
    data[M, M:] = 0.02 * data[:M, M:].sum(axis=0)
    return data


@pytest.fixture(scope="module")
def calib3():
    from puremacro.trade.calibration import calibrate_trade_model

    return calibrate_trade_model(_make_io_table(3, 3, 3), ns=3, nc=3, nfd=3, validate=True)


def test_compute_equilibrium_residuals_rejects_unknown_kwargs(calib3):
    from puremacro.trade.equilibrium import compute_equilibrium_residuals
    from puremacro.trade.solver import solve_trade_equilibrium

    x0 = solve_trade_equilibrium(calib3).x_sol
    with pytest.raises(TypeError, match="fiscal_closur"):
        compute_equilibrium_residuals(x0, calib3, fiscal_closur="labor_tax")
    with pytest.raises(TypeError, match="sigma_typo"):
        compute_equilibrium_residuals(x0, calib3, sigma_typo=0.5)


def test_tariff_matrices_match_solver_layout_for_the_calibration_codes():
    from puremacro.trade import TariffScenario, build_tariff_matrices
    from puremacro.trade.calibration import calibrate_trade_model
    from puremacro.trade.solver import _resolve_tariffs

    ns, nc, nfd = 2, 2, 3
    calib = calibrate_trade_model(_make_io_table(nc, ns, nfd), ns=ns, nc=nc, nfd=nfd, validate=True,
                                  country_codes=("AAA", "BBB"), sector_codes=("S1", "S2"))
    scen = TariffScenario("BBB", {("AAA", "*"): 0.25, ("AAA", "S2"): 0.5})
    tau, tau_fd = build_tariff_matrices(calib, scen)
    ref = tau.transpose(1, 0, 2, 3).reshape(ns * nc, ns, nc)
    ref_fd = tau_fd.transpose(1, 0, 2, 3).reshape(ns * nc, nfd, nc)
    # Two using sectors receive each of the two tariffed origin sectors.
    assert int((ref != 1.0).sum()) == 4
    np.testing.assert_array_equal(ref[:2, :, 1], [[1.25, 1.25], [1.5, 1.5]])
    got, got_fd, _, _ = _resolve_tariffs(calib, tau, tau_fd)
    np.testing.assert_array_equal(got, ref)
    np.testing.assert_array_equal(got_fd, ref_fd)
    # Codes are resolved in the calibration, never in a fixed roster.
    with pytest.raises(ValueError, match="Unknown country code"):
        build_tariff_matrices(calib, TariffScenario("CCC", {("AAA", "*"): 0.25}))


# ---------------------------------------------------------------------------
# GPU / MLX acceleration of the trade model
# ---------------------------------------------------------------------------

from puremacro.trade.gpu import (  # noqa: E402
    BatchedJacobianEvaluator,
    select_compute_device,
    solve_homotopy_continuation,
    solve_trade_equilibrium_gpu,
    solve_trade_equilibrium_mlx,
)
from puremacro.trade.gpu import backend as gpu_backend  # noqa: E402
from puremacro.trade.solver import solve_trade_equilibrium  # noqa: E402

# Solver comparisons use a tight absolute tolerance: the default 2.5e-3 (value units)
# leaves factor prices about 1e-5 apart between two solvers.
TOL = 1e-9

# Accelerator gating has two halves.
#
# 1. Collection: the parametrizations and ``skipif`` marks use the cheap ``find_spec``
#    probes ``has_torch()``/``has_mlx()``, so collecting this module never imports torch
#    or mlx (tests/test_adversarial_orch20_m3_2.py diffs ``sys.modules`` for torch
#    in-process, and an import at collection would blind it).
# 2. Run time: ``_accelerator_import_gate`` (autouse) and the ``importable_mlx`` fixture
#    skip a case whose library is installed but cannot be imported. A torch whose shared
#    library does not load passes ``find_spec``; before this gate it produced five hard
#    failures with the contradictory "device='cpu' is not a PyTorch device" error.
HAS_TORCH = gpu_backend.has_torch()
HAS_MLX = gpu_backend.has_mlx()

_ACCELERATOR_LOADERS = {"torch": "_load_torch", "mlx": "_load_mlx"}


def _skip_unless_importable(backend: str | None) -> None:
    """Skip, never fail, when ``backend`` is torch/mlx and the library cannot be imported."""
    if backend not in _ACCELERATOR_LOADERS:
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # _load_* warns once about a broken install
        available = getattr(gpu_backend, _ACCELERATOR_LOADERS[backend])() is not None
    if not available:
        pytest.skip(f"{backend} is installed but cannot be imported in this environment")


def _skip_unimportable_accelerator_case(node) -> None:
    """Apply ``_skip_unless_importable`` to a test case parametrized with ``backend``."""
    callspec = getattr(node, "callspec", None)
    if callspec is not None:
        _skip_unless_importable(callspec.params.get("backend"))


@pytest.fixture(autouse=True)
def _accelerator_import_gate(request):
    _skip_unimportable_accelerator_case(request.node)


@pytest.fixture(scope="module")
def calib3_nonuniform():
    """3-country calibration whose sectors have labour shares 0.8 / 0.5 / 0.3 (r == w invalid)."""
    from puremacro.trade.calibration import calibrate_trade_model

    data = _make_io_table(3, 3, 3, seed=1, labour_share=np.array([0.8, 0.5, 0.3]))
    return calibrate_trade_model(data, ns=3, nc=3, nfd=3, validate=True)


@pytest.fixture(scope="module")
def shock3():
    """Country 0 imposes 25% on country 1 and 10% on country 2 (4D multipliers)."""
    nc, ns, nfd = 3, 3, 3
    tau = np.ones((ns, nc, ns, nc))
    tau_fd = np.ones((ns, nc, nfd, nc))
    tau[:, 1, :, 0] = 1.25
    tau_fd[:, 1, :, 0] = 1.25
    tau[:, 2, :, 0] = 1.10
    tau_fd[:, 2, :, 0] = 1.10
    return tau, tau_fd


def _available_gpu_backends() -> list[tuple[str | None, str]]:
    out: list[tuple[str | None, str]] = [("cpu", "numpy")]
    if gpu_backend.has_torch():
        out.append(("cpu", "torch"))
    if gpu_backend.has_mlx():
        out.append(("mlx", "mlx"))
    return out


def test_installed_but_unimportable_accelerator_skips_instead_of_failing(monkeypatch, request):
    """A torch (or mlx) that ``find_spec`` sees but that cannot be imported is skipped.

    Pins the run-time gate: with the torch cases parametrized on
    ``has_torch()`` alone, a torch whose shared library does not load produced five hard
    failures. The configuration is simulated here, so the check runs on every machine.
    """
    from types import SimpleNamespace

    # The run-time half of the gate is active for every test of this module.
    assert "_accelerator_import_gate" in request.fixturenames

    monkeypatch.setattr(gpu_backend, "has_torch", lambda: True)
    monkeypatch.setattr(gpu_backend, "has_mlx", lambda: True)
    monkeypatch.setattr(gpu_backend, "_load_torch", lambda: None)
    monkeypatch.setattr(gpu_backend, "_load_mlx", lambda: None)

    # Collection still generates the cases (find_spec is True) ...
    assert ("cpu", "torch") in _available_gpu_backends()
    assert ("mlx", "mlx") in _available_gpu_backends()
    assert ("torch", "cpu") in _evaluator_backends()
    assert ("mlx", "cpu") in _evaluator_backends()

    # ... and the gate skips them at run time instead of letting them fail.
    def case(**params):
        return SimpleNamespace(callspec=SimpleNamespace(params=params))

    for params in (
        {"device": "cpu", "backend": "torch"},
        {"device": "mlx", "backend": "mlx"},
        {"backend": "torch", "device": "cpu", "factor_equivalence": True},
    ):
        with pytest.raises(pytest.skip.Exception, match="installed but cannot be imported"):
            _skip_unimportable_accelerator_case(case(**params))
    with pytest.raises(pytest.skip.Exception, match="installed but cannot be imported"):
        _skip_unless_importable("mlx")

    # NumPy cases and unparametrized tests are never gated.
    _skip_unimportable_accelerator_case(case(device="cpu", backend="numpy"))
    _skip_unimportable_accelerator_case(SimpleNamespace())

    # An importable library lets the case run.
    monkeypatch.setattr(gpu_backend, "_load_torch", lambda: object())
    _skip_unimportable_accelerator_case(case(device="cpu", backend="torch"))


def test_torch_backend_with_cpu_device_reports_unimportable_torch(monkeypatch):
    """``backend='torch', device='cpu'`` without an importable torch names the real cause."""
    monkeypatch.setattr(gpu_backend, "_load_torch", lambda: None)
    with pytest.raises(RuntimeError, match="PyTorch is not installed or could not be imported"):
        gpu_backend._resolve_backend_device("cpu", "torch")
    # The device-name error is reserved for strings that are not torch devices at all.
    with pytest.raises(ValueError, match="is not a PyTorch device"):
        gpu_backend._resolve_backend_device("mlx", "torch")
    # 'cpu' alone keeps degrading to the NumPy evaluator.
    assert gpu_backend._resolve_backend_device("cpu", None) == ("numpy", "cpu")
    # Every torch-capable device string reports the missing library, not the device name,
    # including the ones that resolve through ``select_compute_device('torch')``.
    for dev in ("cuda", "cuda:0", "mps", "gpu", None, "torch", "pytorch"):
        with pytest.raises(RuntimeError, match="not installed or could not be imported"):
            gpu_backend._resolve_backend_device(dev, "torch")
    with pytest.raises(RuntimeError, match="PyTorch is not installed or could not be imported"):
        select_compute_device("cuda")


def test_mlx_device_with_torch_backend_is_a_value_error_with_or_without_mlx(monkeypatch):
    """``device='mlx'``/``'apple_mlx'`` is never a PyTorch device.

    Previously this combination raised ``RuntimeError`` ("Apple MLX requested but
    'mlx' package is not installed") on machines without MLX and ``ValueError`` on machines
    with it; it is now ``ValueError`` everywhere, and the message lists the accepted strings.
    """
    monkeypatch.setattr(gpu_backend, "_load_torch", lambda: _fake_torch(cuda=False, mps=False))
    for mlx_installed in (False, True):
        monkeypatch.setattr(gpu_backend, "has_mlx", lambda: mlx_installed)
        monkeypatch.setattr(gpu_backend, "_load_mlx", lambda: object() if mlx_installed else None)
        for dev in ("mlx", "apple_mlx"):
            for be in ("torch", "pytorch"):
                with pytest.raises(ValueError, match="is not a PyTorch device") as info:
                    gpu_backend._resolve_backend_device(dev, be)
                for accepted in ("'cpu'", "'cuda'", "'cuda:N'", "'mps'", "'gpu'", "'torch'", "'pytorch'"):
                    assert accepted in str(info.value)


def _fake_torch(cuda: bool, mps: bool):
    """Minimal stand-in exposing only what ``select_compute_device`` queries."""
    from types import SimpleNamespace

    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda, device_count=lambda: int(cuda)),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )


def test_torch_backend_keeps_accepting_the_torch_device_strings(monkeypatch):
    """The unimportable-torch fix must not narrow the accepted ``backend='torch'`` devices.

    Before the fix ``device='torch'``/``'pytorch'`` and ``device='gpu'`` resolved through
    ``select_compute_device``; they must keep resolving to a PyTorch device.
    """
    monkeypatch.setattr(gpu_backend, "_load_torch", lambda: _fake_torch(cuda=False, mps=True))
    assert gpu_backend._resolve_backend_device("torch", "torch") == ("torch", "mps")
    assert gpu_backend._resolve_backend_device("pytorch", "pytorch") == ("torch", "mps")
    assert gpu_backend._resolve_backend_device("gpu", "torch") == ("torch", "mps")
    assert gpu_backend._resolve_backend_device("cpu", "torch") == ("torch", "cpu")
    assert gpu_backend._resolve_backend_device(None, "torch") == ("torch", "mps")

    # 'gpu' with neither CUDA nor MPS is a valid request the machine cannot serve.
    monkeypatch.setattr(gpu_backend, "_load_torch", lambda: _fake_torch(cuda=False, mps=False))
    monkeypatch.setattr(gpu_backend, "has_mlx", lambda: True)
    with pytest.raises(RuntimeError, match="neither CUDA nor MPS"):
        gpu_backend._resolve_backend_device("gpu", "torch")
    with pytest.raises(ValueError, match="is not a PyTorch device"):
        gpu_backend._resolve_backend_device("numpy", "torch")


def test_has_torch_has_mlx_are_cheap_probes(monkeypatch):
    """has_torch()/has_mlx() use find_spec and never import the libraries themselves."""
    import importlib.util

    calls: list[str] = []
    real = importlib.util.find_spec

    def spy(name, *args, **kwargs):
        calls.append(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", spy)
    assert isinstance(gpu_backend.has_torch(), bool)
    assert isinstance(gpu_backend.has_mlx(), bool)
    assert calls == ["torch", "mlx"]


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["bogus", "gpu0", "cuda0", "apple"])
def test_select_compute_device_rejects_unknown_strings(bad: str):
    with pytest.raises(ValueError, match="Unknown device/backend"):
        select_compute_device(bad)


def test_select_compute_device_numpy_and_cpu_without_torch(monkeypatch):
    assert select_compute_device("numpy") == ("numpy", "cpu")
    # Without PyTorch, 'cpu' must resolve to the NumPy serial evaluator instead of raising.
    monkeypatch.setattr(gpu_backend, "_load_torch", lambda: None)
    assert select_compute_device("cpu") == ("numpy", "cpu")
    with pytest.raises(RuntimeError, match="PyTorch is not installed"):
        select_compute_device("mps")


def test_gpu_solver_runs_on_numpy_backend_without_torch(calib3, shock3, monkeypatch):
    """The public GPU solver must reach the NumPy evaluator when torch/mlx are absent."""
    monkeypatch.setattr(gpu_backend, "_load_torch", lambda: None)
    monkeypatch.setattr(gpu_backend, "has_mlx", lambda: False)
    tau, tau_fd = shock3
    ref = solve_trade_equilibrium(calib3, tau=tau, tau_fd=tau_fd, tol=TOL)
    res = solve_trade_equilibrium_gpu(calib3, tau=tau, tau_fd=tau_fd, device="cpu", tol=TOL)
    assert res.converged
    assert res.metadata["backend"] == "numpy"
    assert set(res.metadata["jacobian_evaluations"]) == {"numpy:cpu:float64"}
    np.testing.assert_allclose(res.w_sol, ref.w_sol, atol=1e-5)


# ---------------------------------------------------------------------------
# Tariff defaults and layout
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device,backend", _available_gpu_backends())
def test_gpu_solver_default_call_matches_numpy_solver(calib3, device, backend):
    """With all defaults the GPU/MLX solver must solve the same baseline as the NumPy solver."""
    ref = solve_trade_equilibrium(calib3, tol=TOL)
    assert ref.converged and ref.iterations == 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = solve_trade_equilibrium_gpu(calib3, device=device, backend=backend, tol=TOL)
    assert res.converged
    assert res.iterations == 0, "the default call is the tariff-free baseline, like the NumPy solver"
    np.testing.assert_allclose(res.w_sol, ref.w_sol, atol=1e-12)
    np.testing.assert_allclose(res.T_sol, ref.T_sol, atol=1e-10)
    np.testing.assert_allclose(res.x_sol, ref.x_sol, atol=1e-10)


@pytest.mark.skipif(not HAS_MLX, reason="Apple MLX is not available")
@pytest.mark.usefixtures("importable_mlx")
def test_mlx_solver_default_call_matches_numpy_solver(calib3):
    ref = solve_trade_equilibrium(calib3, tol=TOL)
    res = solve_trade_equilibrium_mlx(calib3, tol=TOL)
    assert res.converged and res.iterations == 0
    assert res.metadata["method"] == "mlx_accelerated"
    np.testing.assert_allclose(res.w_sol, ref.w_sol, atol=1e-12)


@pytest.mark.parametrize("device,backend", _available_gpu_backends())
def test_gpu_solver_shock_matches_numpy_solver(calib3, shock3, device, backend):
    tau, tau_fd = shock3
    ref = solve_trade_equilibrium(calib3, tau=tau, tau_fd=tau_fd, tol=TOL)
    assert ref.converged and ref.iterations > 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = solve_trade_equilibrium_gpu(calib3, tau=tau, tau_fd=tau_fd, device=device, backend=backend, tol=TOL)
    assert res.converged and res.iterations > 0
    assert res.metadata["jacobian_dtype"] == "float64"
    assert all(key.endswith(":float64") for key in res.metadata["jacobian_evaluations"])
    np.testing.assert_allclose(res.w_sol, ref.w_sol, atol=1e-5)
    np.testing.assert_allclose(res.p_sol, ref.p_sol, atol=1e-4)


def test_batch_size_magic_number_is_rejected_on_small_calibration(calib3, shock3):
    tau, tau_fd = shock3
    with pytest.raises(ValueError, match="batch_size=100 is inconsistent with a 3-country calibration"):
        solve_trade_equilibrium_gpu(calib3, tau=tau, tau_fd=tau_fd, device="cpu", backend="numpy", batch_size=100)
    with pytest.raises(ValueError, match="implies factor_equivalence=True"):
        solve_trade_equilibrium_gpu(
            calib3, tau=tau, tau_fd=tau_fd, device="cpu", backend="numpy", batch_size=6, factor_equivalence=False
        )
    # The dimension-consistent values still select the layout
    r6 = solve_trade_equilibrium_gpu(calib3, tau=tau, tau_fd=tau_fd, device="cpu", backend="numpy", batch_size=6, tol=TOL)
    r9 = solve_trade_equilibrium_gpu(calib3, tau=tau, tau_fd=tau_fd, device="cpu", backend="numpy", batch_size=9, tol=TOL)
    assert (r6.metadata["factor_equivalence"], r6.metadata["batch_size"]) == (True, 6)
    assert (r9.metadata["factor_equivalence"], r9.metadata["batch_size"]) == (False, 9)
    assert r6.converged and r9.converged
    np.testing.assert_allclose(r6.w_sol, r9.w_sol, atol=1e-5)


def test_reduced_layout_is_refused_when_capital_shares_differ(calib3_nonuniform):
    nc, ns, nfd = 3, 3, 3
    tau = np.ones((ns, nc, ns, nc))
    tau_fd = np.ones((ns, nc, nfd, nc))
    tau[:, 1, :, 0] = 1.30
    tau_fd[:, 1, :, 0] = 1.30
    ref = solve_trade_equilibrium(calib3_nonuniform, tau=tau, tau_fd=tau_fd, tol=TOL)
    assert ref.converged
    assert np.max(np.abs(ref.r_sol / ref.w_sol - 1.0)) > 1e-5, "r != w in this calibration"

    with pytest.raises(ValueError, match="factor_equivalence=True .* is invalid"):
        solve_trade_equilibrium_gpu(
            calib3_nonuniform, tau=tau, tau_fd=tau_fd, device="cpu", backend="numpy", factor_equivalence=True
        )
    # Default: the full layout is selected automatically and matches the NumPy solver
    res = solve_trade_equilibrium_gpu(calib3_nonuniform, tau=tau, tau_fd=tau_fd, device="cpu", backend="numpy", tol=TOL)
    assert res.converged
    assert res.metadata["factor_equivalence"] is False
    np.testing.assert_allclose(res.w_sol, ref.w_sol, atol=1e-5)
    np.testing.assert_allclose(res.r_sol / res.w_sol, ref.r_sol / ref.w_sol, atol=1e-5)


def test_initial_guess_accepts_macro_vectors_of_either_layout(calib3, shock3):
    tau, tau_fd = shock3
    nc = 3
    ref = solve_trade_equilibrium(calib3, tau=tau, tau_fd=tau_fd, tol=TOL)
    kw = dict(tau=tau, tau_fd=tau_fd, device="cpu", backend="numpy", tol=TOL)
    xm_reduced = np.concatenate([np.zeros(nc), calib3.T.ravel()])
    xm_full = np.concatenate([np.zeros(2 * nc), calib3.T.ravel()])
    for x0 in (ref.x_sol, xm_reduced, xm_full):
        res = solve_trade_equilibrium_gpu(calib3, x0=x0, **kw)
        assert res.converged
        np.testing.assert_allclose(res.w_sol, ref.w_sol, atol=1e-5)
    with pytest.raises(ValueError, match="Expected 6 .* 9 .* 29"):
        solve_trade_equilibrium_gpu(calib3, x0=np.zeros(7), **kw)


def test_homotopy_continuation_on_numpy_backend(calib3):
    nc, ns, nfd = 3, 3, 3
    tau = np.ones((ns, nc, ns, nc))
    tau_fd = np.ones((ns, nc, nfd, nc))
    tau[:, 1, :, 0] = 2.45
    tau_fd[:, 1, :, 0] = 2.45
    ref = solve_trade_equilibrium(calib3, tau=tau, tau_fd=tau_fd, tol=TOL)
    res = solve_homotopy_continuation(calib3, tau, tau_fd, device="cpu", backend="numpy", tol=TOL)
    assert res.converged
    np.testing.assert_allclose(res.w_sol, ref.w_sol, atol=1e-5)


# ---------------------------------------------------------------------------
# Batched evaluator: residual parity, masking, Jacobian accuracy, AD modes
# ---------------------------------------------------------------------------

def _central_difference_jacobian(ev: BatchedJacobianEvaluator, xm: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    B = len(xm)
    J = np.empty((B, B))
    for j in range(B):
        e = np.zeros(B)
        e[j] = eps
        f_plus, _, _, _, _ = ev.eval_macro_single(xm + e)
        f_minus, _, _, _, _ = ev.eval_macro_single(xm - e)
        J[:, j] = (f_plus - f_minus) / (2.0 * eps)
    return J


def _evaluator_backends() -> list[tuple[str, str]]:
    out = []
    if gpu_backend.has_torch():
        out.append(("torch", "cpu"))
    if gpu_backend.has_mlx():
        out.append(("mlx", "cpu"))
    return out


def _batched_residual(ev: BatchedJacobianEvaluator, xm: np.ndarray) -> np.ndarray:
    if ev.backend == "torch":
        torch = ev._torch
        return ev.eval_macro_batched_torch(torch.as_tensor(xm[None, :], dtype=ev.dtype, device=ev.device)).cpu().numpy()[0]
    mx = ev._mx
    return np.asarray(ev.eval_macro_batched_mlx(mx.array(xm[None, :], dtype=mx.float64)))[0]


@pytest.mark.parametrize("backend,device", _evaluator_backends())
@pytest.mark.parametrize("factor_equivalence", [True, False])
def test_batched_residual_matches_serial(calib3, shock3, backend, device, factor_equivalence):
    tau, tau_fd = shock3
    ev = BatchedJacobianEvaluator(calib3, tau, tau_fd, device=device, backend=backend, factor_equivalence=factor_equivalence)
    xm = ev._macro_from_x0(None)
    f0, _, _, _, y = ev.eval_macro_single(xm)
    assert np.all(y > 0)
    assert np.max(np.abs(_batched_residual(ev, xm) - f0)) < 1e-11 * max(1.0, np.max(np.abs(f0)))

    # Away from the baseline as well.
    xm_far = xm + np.random.default_rng(3).uniform(-0.2, 0.2, size=xm.size) * np.maximum(np.abs(xm), 1.0)
    f_far = ev.eval_macro_single(xm_far)[0]
    assert np.max(np.abs(_batched_residual(ev, xm_far) - f_far)) < 1e-11 * np.max(np.abs(f_far))


@pytest.mark.parametrize("backend,device", _evaluator_backends())
def test_float64_jacobian_and_ad_modes_are_accurate(calib3, shock3, backend, device):
    tau, tau_fd = shock3
    ev = BatchedJacobianEvaluator(calib3, tau, tau_fd, device=device, backend=backend)
    xm = ev._macro_from_x0(None)
    xm[: ev.n_factor_vars] += 0.02
    f0, _, _, _, _ = ev.eval_macro_single(xm)
    J_ref = _central_difference_jacobian(ev, xm)
    scale = np.max(np.abs(J_ref))

    J_fd = ev.evaluate_batched_jacobian(xm, f_base=f0, ad_mode="finite_diff")
    assert np.max(np.abs(J_fd - J_ref)) / scale < 1e-3  # one-sided differences

    for mode in ("forward", "vjp"):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)  # a silent FD fallback would show up as FD error
            J_ad = ev.evaluate_batched_jacobian(xm, f_base=f0, ad_mode=mode)
        assert np.max(np.abs(J_ad - J_ref)) / scale < 1e-7, mode

    v = np.random.default_rng(0).standard_normal(len(xm))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        jv = ev.jvp(xm, v)
    assert np.max(np.abs(jv - J_ref @ v)) / np.max(np.abs(J_ref @ v)) < 1e-7

    with pytest.raises(ValueError, match="Unknown ad_mode"):
        ev.evaluate_batched_jacobian(xm, f_base=f0, ad_mode="reverse")


def test_numpy_evaluator_warns_when_ad_is_requested(calib3, shock3):
    tau, tau_fd = shock3
    ev = BatchedJacobianEvaluator(calib3, tau, tau_fd, device="cpu", backend="numpy")
    xm = ev._macro_from_x0(None)
    with pytest.warns(RuntimeWarning, match="requires the torch or mlx backend"):
        ev.evaluate_batched_jacobian(xm, ad_mode="forward")


@pytest.mark.skipif(not HAS_MLX, reason="Apple MLX is not available")
@pytest.mark.usefixtures("importable_mlx")
def test_mlx_batched_eval_defaults_to_float64_cpu_stream(calib3, shock3):
    import mlx.core as mx

    tau, tau_fd = shock3
    ev = BatchedJacobianEvaluator(calib3, tau, tau_fd, device="mlx", backend="mlx")
    assert (ev.backend, ev.device) == ("mlx", "gpu")
    xm = ev._macro_from_x0(None)
    f0, _, _, _, _ = ev.eval_macro_single(xm)
    # stream=None used to raise TypeError (mx.DeviceType == None) and silently disabled AD
    out = ev.eval_macro_batched_mlx(mx.array(np.tile(xm, (2, 1)), dtype=mx.float64))
    assert out.dtype == mx.float64
    np.testing.assert_allclose(np.asarray(out)[0], f0, atol=1e-11)
    # a float32 input (mx.array's default for NumPy float64) is upcast and evaluated on the CPU stream
    out32 = ev.eval_macro_batched_mlx(mx.array(np.tile(xm, (2, 1))))
    assert out32.dtype == mx.float64
    np.testing.assert_allclose(np.asarray(out32)[0], f0, atol=1e-4)
    # 'cpu' / 'gpu' string streams are accepted; the default Jacobian is the float64 one
    J_cpu = ev.evaluate_batched_jacobian(xm, f_base=f0, stream="cpu")
    J_default = ev.evaluate_batched_jacobian(xm, f_base=f0)
    np.testing.assert_allclose(J_default, J_cpu, rtol=0, atol=0)
    J_gpu = ev.evaluate_batched_jacobian(xm, f_base=f0, stream="gpu")
    assert J_gpu.shape == J_cpu.shape and np.all(np.isfinite(J_gpu))


@pytest.mark.skipif(not HAS_MLX, reason="Apple MLX is not available")
@pytest.mark.usefixtures("importable_mlx")
def test_explicit_float32_device_warns_and_polishes_in_float64(calib3, shock3):
    tau, tau_fd = shock3
    ref = solve_trade_equilibrium(calib3, tau=tau, tau_fd=tau_fd, tol=TOL)
    with pytest.warns(RuntimeWarning, match="only supports float32"):
        res = solve_trade_equilibrium_gpu(calib3, tau=tau, tau_fd=tau_fd, device="mlx", tol=TOL)
    assert res.converged
    assert res.metadata["requested_device"] == "gpu"
    assert res.metadata["jacobian_device"] == "mlx:cpu"
    assert set(res.metadata["jacobian_evaluations"]) == {"mlx:cpu:float64"}
    np.testing.assert_allclose(res.w_sol, ref.w_sol, atol=1e-5)


