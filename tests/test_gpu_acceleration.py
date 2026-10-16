"""GPU and Apple Silicon MLX acceleration of the trade equilibrium.

Verifies, on a seeded synthetic table (no private data):

1. Device detection and memory profiling.
2. The pre-inverted Leontief operators and the batched Jacobian of both macro
   layouts (reduced ``2*nc``, full ``3*nc``).
3. The directional derivative (JVP).
4. The GPU / MLX solvers against the NumPy solver of the same model.
5. Homotopy continuation to a large tariff.
6. Native float64 MLX linear solves on the CPU stream.

Accelerator cases skip, never fail, when torch / mlx are missing or cannot be
imported; the NumPy paths always run.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from puremacro.trade import TariffScenario, build_tariff_matrices, calibrate_trade_model, solve_trade_equilibrium
from puremacro.trade.data import synthetic_codes, synthetic_icio_table
from puremacro.trade.gpu import backend as gpu_backend
from puremacro.trade.gpu.backend import (
    detect_device,
    get_memory_usage,
    reset_peak_memory,
    select_compute_device,
    to_numpy,
    to_tensor,
)
from puremacro.trade.gpu.batched_jacobian import BatchedJacobianEvaluator
from puremacro.trade.gpu.homotopy import solve_homotopy_continuation
from puremacro.trade.gpu.mlx_solver import solve_trade_equilibrium_mlx
from puremacro.trade.gpu.solver_gpu import solve_trade_equilibrium_gpu

NC, NS = 8, 5
TOL = 1e-8


def _importable(loader) -> bool:
    """True when the accelerator actually imports (``has_*`` are find_spec probes only)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return loader() is not None


def _accel_backend() -> str | None:
    """An importable torch, else an importable mlx, else None (probed lazily, never at collection)."""
    if _importable(gpu_backend._load_torch):
        return "torch"
    if _importable(gpu_backend._load_mlx):
        return "mlx"
    return None


needs_accelerator = pytest.mark.skipif(
    "_accel_backend() is None", reason="needs an importable torch or mlx (pip install puremacro[accel])"
)
needs_mlx = pytest.mark.skipif(
    "not _importable(gpu_backend._load_mlx)", reason="Apple MLX is not available or cannot be imported"
)


@pytest.fixture(scope="module")
def calib():
    """An 8-country, 5-sector synthetic calibration."""
    cc, sc = synthetic_codes(NC, NS)
    return calibrate_trade_model(synthetic_icio_table(NC, NS, seed=3), ns=NS, nc=NC, country_codes=cc, sector_codes=sc)


@pytest.fixture(scope="module")
def shock(calib):
    """C01 raises a 30% tariff on C02 and 15% on C03."""
    scenario = TariffScenario(importer="C01", tariffs={("C02", "*"): 0.30, ("C03", "*"): 0.15})
    return build_tariff_matrices(calib, scenario)


@pytest.fixture(scope="module")
def reference(calib, shock):
    tau, tau_fd = shock
    ref = solve_trade_equilibrium(calib, tau=tau, tau_fd=tau_fd, tol=TOL)
    assert ref.converged
    return ref


class TestDeviceBackend:
    """Device selection, capability detection and memory profiling."""

    def test_detect_device_auto(self):
        dev = detect_device()
        assert dev.device in ("cuda", "mps", "gpu", "cpu")
        assert dev.backend in ("torch", "mlx", "numpy")
        assert isinstance(dev.device_name, str)
        assert isinstance(dev.supports_float64, bool)
        assert dev.total_memory_gb is None or isinstance(dev.total_memory_gb, float)

    def test_select_compute_device(self):
        backend, dev = select_compute_device("auto")
        assert backend in ("torch", "mlx", "numpy")
        assert dev in ("cuda", "mps", "gpu", "cpu")

    @needs_accelerator
    def test_tensor_conversion_roundtrip(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
        arr_back = to_numpy(to_tensor(arr, backend=_accel_backend()))
        np.testing.assert_allclose(arr, arr_back, rtol=1e-12, atol=1e-12)

    def test_memory_usage_and_reset(self):
        reset_peak_memory()
        mem = get_memory_usage()
        assert "allocated_mb" in mem
        assert "peak_mb" in mem
        assert mem["peak_mb"] >= 0.0


class TestBatchedJacobianEvaluator:
    """Pre-inversion, batched evaluation of both layouts and the JVP."""

    def _evaluator(self, calib, shock, **kw):
        tau, tau_fd = shock
        backend = kw.pop("backend", _accel_backend() or "numpy")
        return BatchedJacobianEvaluator(calib, tau, tau_fd, device="cpu", backend=backend, **kw)

    def test_pre_inversion_operators(self, calib, shock):
        ev = self._evaluator(calib, shock, backend="numpy")
        M = NS * NC
        assert ev.inv_P_np.shape == (M, M) and ev.inv_Y_np.shape == (M, M)
        np.testing.assert_allclose((np.eye(M) - ev.a_2d) @ ev.inv_Y_np, np.eye(M), atol=1e-12)

    @pytest.mark.parametrize("factor_equivalence,size", [(True, 2 * NC), (False, 3 * NC)])
    def test_batched_jacobian_of_each_layout(self, calib, shock, reference, factor_equivalence, size):
        ev = self._evaluator(calib, shock, factor_equivalence=factor_equivalence)
        assert ev.batch_size == size
        xm = ev._macro_from_x0(reference.x_sol)
        f0 = ev.eval_macro_single(xm)[0]
        assert float(np.max(np.abs(f0))) < 1e-6  # the NumPy solution solves the macro system
        J = ev.evaluate_batched_jacobian(xm, f_base=f0)
        assert J.shape == (size, size)
        assert np.all(np.isfinite(J))
        assert np.linalg.matrix_rank(J) == size

    @needs_accelerator
    def test_jvp_directional_derivative(self, calib, shock):
        ev = self._evaluator(calib, shock)
        xm = ev._macro_from_x0(None)
        v = np.random.default_rng(123).standard_normal(ev.batch_size)
        v /= np.linalg.norm(v)
        J = ev.evaluate_batched_jacobian(xm, ad_mode="forward")
        jv = ev.jvp(xm, v)
        assert jv.shape == (ev.batch_size,)
        np.testing.assert_allclose(jv, J @ v, rtol=1e-9, atol=1e-9 * np.max(np.abs(J @ v)))


class TestGPUSolverParity:
    """solve_trade_equilibrium_gpu / _mlx against the NumPy solver."""

    @pytest.mark.parametrize("backend", ["numpy", "accelerator"])
    def test_gpu_solver_convergence_and_parity(self, calib, shock, reference, backend):
        if backend == "accelerator":
            backend = _accel_backend()
            if backend is None:
                pytest.skip("needs an importable torch or mlx")
        tau, tau_fd = shock
        res = solve_trade_equilibrium_gpu(calib, tau=tau, tau_fd=tau_fd, device="cpu", backend=backend, tol=TOL)
        assert res.converged, f"did not converge, max residual {res.max_residual}"
        assert res.max_residual <= TOL
        assert res.x_sol.shape == reference.x_sol.shape
        np.testing.assert_allclose(res.x_sol, reference.x_sol, rtol=0, atol=1e-7)
        # Realised net exports equal the fixed foreign saving (Walras).
        assert abs(float(np.sum(res.exports - res.imports))) < 1e-6

    @needs_mlx
    def test_mlx_solver_convergence_and_parity(self, calib, shock, reference):
        tau, tau_fd = shock
        res = solve_trade_equilibrium_mlx(calib, tau=tau, tau_fd=tau_fd, tol=TOL)
        assert res.converged
        assert res.metadata["method"] == "mlx_accelerated"
        np.testing.assert_allclose(res.x_sol, reference.x_sol, rtol=0, atol=1e-7)

    @needs_mlx
    def test_mlx_native_cpu_stream_solve(self):
        """Apple MLX solves float64 linear systems on the mx.cpu stream."""
        import mlx.core as mx

        n = 200
        rng = np.random.default_rng(123)
        A_np = rng.standard_normal((n, n))
        A_spd_np = A_np.T @ A_np + 1.0 * np.eye(n)
        b_np = rng.standard_normal(n)
        x_ref = np.linalg.solve(A_spd_np, b_np)
        with mx.stream(mx.cpu):
            x_mx = mx.linalg.solve(mx.array(A_spd_np, dtype=mx.float64), mx.array(b_np[:, None], dtype=mx.float64),
                                   stream=mx.cpu)
            mx.eval(x_mx)
            x_sol = np.array(x_mx).ravel()
        assert np.max(np.abs(x_sol - x_ref)) < 1.0e-10


class TestHomotopyContinuation:
    """Adaptive homotopy continuation along the tariff path."""

    def test_homotopy_continuation_to_a_large_tariff(self, calib):
        scenario = TariffScenario(importer="C01", tariffs={("C02", "*"): 1.20, ("C03", "*"): 0.60})
        tau, tau_fd = build_tariff_matrices(calib, scenario)
        steps: list[float] = []
        res = solve_homotopy_continuation(
            calib, tau, tau_fd, device="cpu", backend="numpy", initial_step=0.25, tol=TOL,
            callback=lambda lam, _res: steps.append(lam),
        )
        assert res.converged and res.max_residual <= TOL
        assert steps[-1] == pytest.approx(1.0) and len(steps) >= 2
        ref = solve_trade_equilibrium(calib, tau=tau, tau_fd=tau_fd, tol=TOL)
        assert ref.converged
        np.testing.assert_allclose(res.x_sol, ref.x_sol, rtol=0, atol=1e-7)
