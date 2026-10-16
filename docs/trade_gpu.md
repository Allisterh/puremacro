> 🇬🇧 English · 🇪🇸 [Español](es/trade_gpu.md)

# Hardware Backends: GPU, Apple MLX & Batched Jacobians for Trade and Spatial GE

`puremacro._backend` and `puremacro.trade.gpu` add optional hardware acceleration to the general equilibrium solvers documented in [Quantitative Spatial Economics & Gravity Trade GE](spatial_and_trade_ge.md). Two layers are involved:

1. **The array-namespace layer** (`puremacro._backend`): `numpy`, `mlx` (Apple Silicon) and `cupy` (NVIDIA) share one NumPy-like array API. `AllenArkolakisModel.solve_equilibrium(..., backend=)` / `solve_counterfactual(..., backend=)` run their contraction mapping through it, with a `RuntimeWarning` and a NumPy fallback whenever the requested backend is missing or fails. The trade solver `solve_trade_equilibrium` is NumPy only: `backend="numpy"` is the one accepted value and any other raises `NotImplementedError`.
2. **The device layer** (`puremacro.trade.gpu`): a PyTorch (CUDA / Apple MPS / CPU) and native Apple MLX engine for the consistent-accounting multi-country, multi-sector trade model of `puremacro.trade` (Cobb-Douglas intermediates, lump-sum rebates of tariff revenue), built around pre-inverted Leontief operators, a batched Jacobian evaluated as one multi-right-hand-side GEMM, a Levenberg-Marquardt (1944, 1963) step with a backtracking line search, and adaptive homotopy continuation (Allgower & Georg, 1990) in the tariff schedule.

Both layers are optional. Nothing on this page is required to run the models: the NumPy path remains the reference implementation and the correctness oracle, and it is the only path available in Pyodide (iPad, browser). Every accelerated result on this page is compared against it.

---

## 1. Algorithmic Framework

### 1.1 Which backend runs what

`puremacro._backend.SUPPORTED` is `("numpy", "numba", "mlx", "cupy")`. `backend_available(name)` checks `importlib.util.find_spec`, `available_backends()` lists the installed ones (NumPy first), and `get_array_namespace(name)` returns `numpy`, `mlx.core` or `cupy` (`"numba"` is a compiled-kernel backend, not an array namespace, and raises `ValueError` here). `to_numpy(x)` brings any array home, routing CuPy through `cp.asnumpy` because CuPy blocks implicit host conversion.

- **`solve_trade_equilibrium(calib, ..., backend=)`** accepts `"numpy"` only. Accelerated trade solves go through the device layer below, which solves the same model and returns the same result type.
- **`AllenArkolakisModel.solve_equilibrium(backend=)`**: the whole damped contraction mapping on $(w_i, L_i)$ (kernel matrix $\tau_{ji}^{-\theta}$, market access, the utility-equalization softmax) runs on the device in `float32` for MLX and `float64` for CuPy. A float32 device can only certify the fixed point to its own floor, so whenever that floor is coarser than the `tol` you asked for (always on MLX, where the device floors are $\max(\text{tol}, 10^{-6})$ on the residual and $10^{-5}$ on the utility dispersion) the device loop is treated as a warm start and the iteration is **finished on the host in NumPy float64** at the requested `tol`. The requested tolerance is therefore honored and `converged` is truthful whatever the backend; Example 4.4 shows MLX and NumPy agreeing to $10^{-9}$ after the same 27 iterations. After convergence the population is renormalized to $\bar L$, the wages to unit mean, and the price index, market-access terms and real wages are recomputed in NumPy float64, which is why the labor-conservation residual is zero (to rounding) on both backends. `solve_counterfactual` passes `backend` to both the baseline and the counterfactual solve.

### 1.2 Device selection and `DeviceInfo`

`puremacro.trade.gpu.select_compute_device(preferred)` maps a name to a `(backend, device)` pair, where `backend` is `"torch"`, `"mlx"` or `"numpy"` and `device` is `"cuda"`, `"mps"`, `"gpu"` or `"cpu"`:

| `preferred` | Result | If unavailable |
|---|---|---|
| `None` / `"auto"` | first of: torch CUDA, torch MPS, MLX GPU, torch CPU, NumPy CPU | never fails |
| `"numpy"` | `("numpy", "cpu")`, the serial NumPy evaluator | never fails |
| `"cpu"` | `("torch", "cpu")` when PyTorch is installed, `("numpy", "cpu")` otherwise | never fails |
| `"cuda"`, `"cuda:N"` | that PyTorch CUDA device | `RuntimeError` (PyTorch missing, `torch.cuda.is_available()` is `False`, or index `N` out of range) |
| `"mps"` | `("torch", "mps")` | `RuntimeError` (PyTorch missing or `torch.backends.mps.is_available()` is `False`) |
| `"torch"` / `"pytorch"` | best PyTorch device: CUDA, then MPS, then CPU | `RuntimeError` if PyTorch is missing |
| `"mlx"` / `"apple_mlx"` | `("mlx", "gpu")` | `RuntimeError` if `mlx` is missing |
| `"gpu"` | best GPU of any backend: CUDA, then MPS, then MLX | `RuntimeError` if none of the three is available |

Every accepted string is listed above; anything else raises `ValueError: Unknown device/backend ...; expected one of (...)` rather than quietly resolving to `"auto"`. `"numpy"` is a real name, so you can ask for the serial evaluator explicitly, and `"cpu"` does not demand PyTorch: it degrades to `("numpy", "cpu")`, which is what every Pyodide kernel gets.

`detect_device(preferred)` wraps the same choice in a frozen dataclass:

| `DeviceInfo` field | Meaning |
|---|---|
| `backend` | `"torch"`, `"mlx"` or `"numpy"` |
| `device` | `"cuda"`, `"mps"`, `"gpu"` (MLX) or `"cpu"` |
| `device_name` | CUDA adapter name, `"Apple Metal (<cpu>)"`, `"Apple MLX (<cpu>)"` or the host processor |
| `total_memory_gb` | CUDA: `total_memory` of the device properties; Apple Silicon: `sysctl hw.memsize` (the unified pool; `None` if `sysctl` cannot be run); `None` on plain CPU |
| `supports_float64` | `True` for CUDA and CPU, `False` for MPS, `False` for the MLX GPU stream |
| `is_uma` | `True` on Apple Silicon (unified memory: host and GPU share one pool, so there is no device copy) |

The `repr` summarizes the precision situation, e.g. `f32-only` for MPS and `f32-gpu/f64-cpu-stream` for MLX.

### 1.3 Precision: float64 versus float32

A one-sided finite difference divides the rounding error of the residual by the step, so a float32 residual produces a large relative Jacobian error and a diverging Newton step. **The rule the device layer follows is therefore: assemble the Jacobian in float64 wherever the device can, and use a float32 device only when you ask for one explicitly, and only for the coarse phase.**

- **CUDA and PyTorch CPU** execute in float64 throughout.
- **MPS** (Apple Metal through PyTorch) has no float64. With `device="auto"` the solver silently substitutes the float64 PyTorch **CPU** evaluator. With `device="mps"` requested explicitly it emits a `RuntimeWarning` ("... only supports float32; finite-difference Jacobians are unusable in float32, so it is used for the coarse phase only and the Jacobian is polished in float64 on torch:cpu"), uses MPS while $\max_i |F_i| \ge 5 \times 10^6$ and only in the first five iterations, and polishes in float64 afterwards.
- **MLX** keeps float64 constants created under `mx.stream(mx.cpu)`, where MLX runs Accelerate/LAPACK (and where `mx.exp`, which is only float32-accurate, is replaced by an exact `e ** x`), and float32 copies for the GPU stream. `BatchedJacobianEvaluator` **defaults to the CPU stream**; the float32 GPU stream is reached only through `stream="gpu"` or the explicit-device coarse phase. `solve_trade_equilibrium_mlx` is `solve_trade_equilibrium_gpu(device="mlx")` and follows exactly the same policy.

Whichever device is used, `result.metadata["jacobian_evaluations"]` counts the Jacobians by `backend:device:dtype`, so the precision path a run actually took is on the result. The single-point residual `eval_macro_single`, the line-search trial points and the final residual check are NumPy float64 on the host, so `max_residual` and `converged` are always float64 quantities; only coarse-phase Jacobian columns can carry float32 rounding. Example 4.3 measures it: the float32 MPS Jacobian is unusable (its error exceeds the largest entry), while the PyTorch CPU and MLX CPU-stream Jacobians reproduce the serial NumPy one to rounding.

### 1.4 The batched Jacobian

The consistent-accounting system of `puremacro.trade` (see [Consistent trade accounting](trade_accounting.md)) is condensed onto a macro vector. With $N$ countries, $S$ sectors and $M = SN$ country-sector pairs, `BatchedJacobianEvaluator` works on

$$x_m = \begin{bmatrix} \log w \\ T \end{bmatrix} \in \mathbb{R}^{2N} \quad\text{or}\quad x_m = \begin{bmatrix} \log r \\ \log w \\ T \end{bmatrix} \in \mathbb{R}^{3N},$$

where $w$ are wages, $r$ capital rentals and $T$ the lump-sum transfers (government budgets). Foreign saving is exogenous, as in the full model, in numeraire units or as a fixed share of world factor income (`foreign_saving_units`), so it is not an unknown. The macro residuals are the first producer price minus one (the numeraire), labour-market clearing in every country but the first (implied by Walras' law once all goods markets clear), capital-market clearing (full layout only) and the government budgets. A solution of the macro system solves every equation of the full consistent-accounting system.

The first layout imposes $r_c = w_c$, which is exact only when the capital share is the same in every sector of a country: capital-market clearing then follows from labour-market clearing. The layout is chosen by the keyword-only `factor_equivalence`, and `batch_size` is the older spelling, *validated* against the calibration. `factor_equivalence=None` (the default) selects the reduced layout exactly when it is exact for the calibration and the full layout otherwise; `factor_equivalence=True` on a calibration where it is not exact raises `ValueError` instead of quietly dropping the capital-market clearing condition. If you pass `batch_size`, it must equal $2N$ or $3N$ for *your* $N$: on the 2-country toy calibration below that is 4 or 6, and `batch_size=5` raises `ValueError: batch_size=5 is inconsistent with a 2-country calibration: expected 4 (reduced layout, r == w) or 6 (full layout).` Passing both keywords with conflicting implications also raises.

Given $x_m$, prices and gross outputs follow from two Leontief systems,

$$p = (I - B^\top)^{-1}\, v(x_m), \qquad y = (I - A)^{-1}\, d(x_m, p),$$

with $B^\top = (A \circ \tau)^\top / (1 - \text{tax})$ the tariff-inclusive cost-share operator, $v$ the Cobb-Douglas unit value added and $A$ the input-output coefficients. Both inverses are formed **once** (`(I - A)^{-1}` at construction, `(I - B^\top)^{-1}` in `update_tariffs`, i.e. once per tariff schedule). A forward-difference Jacobian needs $F$ at the $B$ perturbed points $x_m + h_j e_j$, with $h_j = \varepsilon$ for the log factor prices and $h_j = \varepsilon \max(|T_j|, s)$ for the transfers, where $s$ is the median baseline factor income ($\varepsilon = 10^{-6}$ by default, `eps_fd`). Stacking the perturbed points as rows of $X \in \mathbb{R}^{B \times B}$, the price and output solves for the whole batch become two matrix products,

$$P = \left[(I - B^\top)^{-1} V^\top\right]^\top, \qquad Y = \left[(I - A)^{-1} D^\top\right]^\top, \qquad V, D \in \mathbb{R}^{B \times M},$$

and every account (factor demands, production and final-use taxes, duties) becomes a batched matrix product. One kernel launch replaces $B$ serial residual evaluations, and

$$J_{:,j} = \frac{F(x_m + h_j e_j) - F(x_m)}{h_j}.$$

`ad_mode="forward"` replaces the differences by forward-mode automatic differentiation (`torch.func.jacfwd` on PyTorch, `mx.jvp` column by column on MLX) and `ad_mode="vjp"` by reverse mode (`torch.func.jacrev`, `mx.vjp`); both remove the step-size sensitivity. On any failure the evaluator emits a `RuntimeWarning` and falls back to the finite-difference path (never silently), and on the `numpy` backend either AD mode warns and uses finite differences. A macro vector whose length does not match the layout raises `ValueError`, and so does an unknown `ad_mode`. The `jvp(xm, v)` method returns the directional derivative $J v$ the same way, with a float64 central difference as last resort.

### 1.5 Levenberg-Marquardt step, equilibration and line search

Each iteration of `solve_trade_equilibrium_gpu` / `solve_trade_equilibrium_mlx` scales the Jacobian on both sides (Jacobi equilibration), $\tilde J = D_L J D_R$ with $D_R = \operatorname{diag}(1/\lVert J_{:,j}\rVert_2)$ and $D_L = \operatorname{diag}(1/\lVert (J D_R)_{i,:}\rVert_2)$, and solves the regularized normal equations

$$\left(\tilde J^\top \tilde J + \mu I\right) u = -\tilde J^\top D_L F(x_m), \qquad \Delta x_m = D_R\, u,$$

with `torch.linalg.solve` in float64 whenever PyTorch is the evaluating backend (on the CUDA device when that is the polish device, on the CPU otherwise), then `scipy.linalg.solve`, then `scipy.linalg.lstsq`; the first that succeeds wins. The step is rescaled so that its factor-price block stays within $0.3$ in absolute value, and the log factor prices are clipped to $[-5, 5]$. A backtracking line search on the merit function $\Phi(x) = \lVert D_L F(x) \rVert_2$ tries $\alpha = 1, \tfrac12, \tfrac14, \dots$ (at most 12 trial points) and accepts the first trial with $\max_i|F_i| \le \text{tol}$, $\Phi(x + \alpha \Delta) < \Phi(x)$ or $\max_i |F_i(x + \alpha\Delta)| < \max_i |F_i(x)|$: a simple-decrease rule in the Armijo (1966) backtracking pattern, without a sufficient-decrease constant. The damping follows the classical Marquardt update: $\mu \leftarrow \max(0.3\mu, 10^{-8})$ after an accepted step, $\mu \leftarrow \max(0.5\mu, 10^{-8})$ when only the best trial improved the merit, and $\mu \leftarrow \min(5\mu, 10^{2})$ otherwise. The iteration stops only when the macro residual is within `tol` **and** every equation of the full system at the reconstructed state is within `tol`; the returned result is post-processed and audited exactly like a NumPy solve.

### 1.6 Homotopy continuation in the tariff schedule

Large asymmetric tariff shocks can defeat a Newton-type method started from the baseline. `solve_homotopy_continuation` embeds the target schedule in the path

$$\tau(\lambda) = (1-\lambda)\,\tau_0 + \lambda\,\tau_1, \qquad \tau^{fd}(\lambda) = (1-\lambda)\,\tau^{fd}_0 + \lambda\,\tau^{fd}_1, \qquad \lambda \in [0, 1],$$

and traces it with the predictor-free, warm-started continuation of Allgower & Georg (1990, ch. 1): solve at $\lambda_0 = 0$ (or accept `x0`), then repeatedly solve at $\min(\lambda + \Delta\lambda, 1)$ starting from the last converged state. Step control is adaptive: a step that converged in at most 4 iterations grows the next one by $1.5\times$ (capped at `max_step`), one that needed at least `max_iter_per_step - 5` shrinks it by $0.75$ (floored at `min_step`), and a failed step is discarded and retried with half the size. If the step falls below `min_step`, if the baseline itself does not converge, or if no step ever succeeds, the function **raises `RuntimeError`** (it does not return a partial result); the `callback(lambda, result)` hook, called after every successful step, is the way to keep intermediate equilibria. The target and base schedules take every form `solve_trade_equilibrium` accepts (multipliers or rates by importing country).

---

## 2. Methodological Options

| Option | Namespace layer (`backend=`, spatial solver) | `solve_trade_equilibrium_gpu` | `solve_trade_equilibrium_mlx` |
|---|---|---|---|
| **Accepted names** | `"numpy"`, `"mlx"`, `"cupy"` (`"numba"` warns and falls back; any other string raises `ValueError`) | `device` in `"auto"`, `"numpy"`, `"cpu"`, `"cuda"`, `"cuda:N"`, `"mps"`, `"mlx"`, `"gpu"`; `backend` in `"torch"`, `"mlx"`, `"numpy"` | none: MLX only, raises `RuntimeError` without it |
| **Precision** | float32 contraction on MLX, finished on the host in float64 | float64 Jacobian wherever the device allows; a float32 device only when requested explicitly, and only for the coarse phase | same policy, on the MLX CPU stream |
| **Jacobian** | not applicable (fixed-point iteration) | batched finite differences, `ad_mode="forward"` (jacfwd) or `"vjp"` (jacrev) | batched finite differences, `ad_mode="forward"` (jvp) or `"vjp"` (mx.vjp) |
| **Stopping rule** | the `tol` you asked for, finished on the host when the device floor is coarser | `tol=2.5e-3` on every equation of the full system, absolute in the calibration's value units | same |
| **Failure policy** | `RuntimeWarning`, then NumPy, for a missing backend, a raised exception *and* a non-converged accelerated solve | exceptions propagate; AD failures warn and fall back to finite differences | same |
| **Where tested** | a 5-region spatial model (`tests/test_trade_solver_accelerated.py`) | an 8-country, 5-sector synthetic calibration against the NumPy solver (`tests/test_gpu_acceleration.py`), skipped when no accelerator is importable | same |

The default trade tolerance is $2.5 \times 10^{-3}$ for the NumPy solver and the accelerated ones. It is absolute in the calibration's value units and loose; tighten it for parity work, as the examples below do.

---

## 3. Installing the Accelerators

None of the accelerators is part of the base install, and the base install runs everything on this page through the NumPy path.

| Accelerator | Install | Platforms | Provides |
|---|---|---|---|
| Apple MLX | `pip install "puremacro[backend]"` (the `[accel]` extra has the same contents: `numba` plus `mlx` on macOS) | Apple Silicon macOS only (`sys_platform == 'darwin'` marker) | `backend="mlx"` on the spatial solver, `solve_trade_equilibrium_mlx`, `device="mlx"` in the device layer |
| NVIDIA CuPy | `pip install "puremacro[cuda]"` (`cupy-cuda12x`; choose the wheel matching your CUDA toolkit) | NVIDIA GPUs with CUDA | `backend="cupy"` on the spatial solver |
| PyTorch | `pip install torch` (there is **no** puremacro extra for it; use the selector at pytorch.org for a CUDA build) | CUDA, Apple MPS, or CPU | `solve_trade_equilibrium_gpu`, `solve_homotopy_continuation`, `device="cuda"/"mps"/"cpu"` |

`puremacro.trade.gpu` imports `torch` and `mlx` **lazily**: `import puremacro.trade`, and even `from puremacro.trade.gpu import solve_trade_equilibrium_gpu`, leave both out of `sys.modules` on a machine where they are installed, and succeed on a machine where they are not. A call that genuinely needs an absent accelerator raises `RuntimeError` when it asks for it by name, while `device=None` / `"auto"` resolves to `("numpy", "cpu")` and runs the serial evaluator. `puremacro.runtime.capabilities().backends` lists the installed *namespace* backends (`numpy`, `numba`, `mlx`, `cupy`); PyTorch is not part of that tuple even when installed.

**Pyodide, iPad and the browser.** Neither PyTorch, MLX nor CuPy can be installed in a Pyodide kernel, so `capabilities().backends == ("numpy",)` there (see [Running anywhere](tablet.md)), `detect_device()` returns `numpy:cpu` by construction, and the device solvers run the NumPy evaluator. A script written against the NumPy path runs unchanged on a tablet.

---

## 4. Runnable Worked Examples

All examples run offline. They were executed on an Apple Silicon workstation with PyTorch (MPS) and MLX installed and no NVIDIA GPU; the printed lines reproduce that run. The examples continue from each other, as one session.

### 4.1 What hardware do I have?

```python
from puremacro._backend import available_backends
from puremacro.trade.gpu import detect_device, get_memory_usage, has_mlx, has_torch, select_compute_device

print("namespace backends:", available_backends())
print("torch installed:", has_torch(), "| mlx installed:", has_mlx())
print("auto selection   :", select_compute_device())
print(detect_device())
if has_mlx():
    print(detect_device("mlx"))
if has_torch():
    print(detect_device("cpu"))
try:
    select_compute_device("cuda")
except RuntimeError as exc:
    print("cuda:", exc)
print(sorted(get_memory_usage()))
```

```text
namespace backends: ('numpy', 'numba', 'mlx')
torch installed: True | mlx installed: True
auto selection   : ('torch', 'mps')
<DeviceInfo: torch:mps (Apple Metal (arm), 36.0 GB, f32-only, UMA)>
<DeviceInfo: mlx:gpu (Apple MLX (arm), 36.0 GB, f32-gpu/f64-cpu-stream, UMA)>
<DeviceInfo: torch:cpu (arm, f64)>
cuda: CUDA requested but torch.cuda.is_available() is False.
['allocated_mb', 'backend', 'device', 'driver_mb', 'peak_mb']
```

With PyTorch installed, `"auto"` prefers MPS over MLX on a Mac, and the MPS `DeviceInfo` reports `f32-only`. The memory dictionary always carries `backend`, `device`, `allocated_mb` and `peak_mb`; MPS adds `driver_mb`, CUDA and MLX add `cache_mb`, and the NumPy fallback reports the peak resident set size of the process from `resource.getrusage`.

### 4.2 The NumPy reference on a toy calibration

The calibration is the 2-country, 2-sector, 3-final-use table of `tests/test_trade_solver_accelerated.py`. At its baseline the table is already an equilibrium, so the example imposes a 25% tariff on everything country 1 imports from country 0 to make the solvers work. `solve_trade_equilibrium` gives the reference, and refuses a non-NumPy backend.

```python
import numpy as np

from puremacro.trade import calibrate_trade_model, solve_trade_equilibrium


def toy_calibration():
    """2-country, 2-sector, 3-final-use table (tests/test_trade_solver_accelerated.py)."""
    nc, ns, nfd = 2, 2, 3
    data = np.zeros((7, 10), dtype=float)
    data[:4, :4] = np.array([
        [20.0, 15.0, 5.0, 2.0],
        [10.0, 25.0, 2.0, 8.0],
        [5.0, 5.0, 12.0, 18.0],
        [10.0, 10.0, 18.0, 22.0],
    ])
    inter_col_sums = data[:4, :4].sum(axis=0)
    y = np.array([100.0, 150.0, 120.0, 180.0])
    va = y - inter_col_sums
    taxes = 0.05 * y
    va_fac = va - taxes
    data[4, :4] = taxes
    data[5, :4] = (2.0 / 3.0) * va_fac
    data[6, :4] = (1.0 / 3.0) * va_fac
    fd_row_sums = y - data[:4, :4].sum(axis=1)
    for i in range(4):
        shares = [0.50, 0.25, 0.05, 0.10, 0.08, 0.02] if i < 2 else [0.10, 0.08, 0.02, 0.50, 0.25, 0.05]
        data[i, 4:] = fd_row_sums[i] * np.array(shares)
    data[4, 4:] = 0.02 * data[:4, 4:].sum(axis=0)
    return calibrate_trade_model(data, ns=ns, nc=nc, nfd=nfd, validate=True)


calib = toy_calibration()
ns, nc, nfd = calib.n_sectors, calib.n_countries, calib.n_final_demand

# 25% tariff on everything country 1 imports from country 0 (intermediates and final demand)
tau = np.ones((ns, nc, ns, nc))
tau[:, 0, :, 1] = 1.25
tau_fd = np.ones((ns, nc, nfd, nc))
tau_fd[:, 0, :, 1] = 1.25

ref = solve_trade_equilibrium(calib, tau=tau, tau_fd=tau_fd, tol=1e-9)
print(f"numpy reference: converged={ref.converged} iterations={ref.iterations} w={np.round(ref.w_sol.ravel(), 6)}")
try:
    solve_trade_equilibrium(calib, tau=tau, tau_fd=tau_fd, backend="mlx")
except NotImplementedError as exc:
    print("NotImplementedError:", exc)
```

```text
numpy reference: converged=True iterations=6 w=[1.000255 0.977642]
NotImplementedError: solve_trade_equilibrium supports the NumPy backend only
```

### 4.3 The batched Jacobian on the same calibration

The evaluator accepts the 4-D tariff tensors directly and, with `batch_size` left at `None`, picks the reduced layout $[\log w; T]$ (exact here: the capital share is uniform) with $B = 2 \cdot 2 = 4$ entries. The macro point is the reference equilibrium, where the macro residual vanishes. The MLX entry is requested as `device="mlx"`, so the evaluator takes the float64 CPU stream; `stream="gpu"` would be the way to ask for float32.

```python
from puremacro.trade.gpu import BatchedJacobianEvaluator

x_m = np.concatenate([np.log(ref.w_sol.ravel()), ref.T_sol.ravel()])   # [log w; T] at the reference
ev_np = BatchedJacobianEvaluator(calib, tau, tau_fd, device="cpu", backend="numpy")
J_np = ev_np.evaluate_batched_jacobian(x_m)
print("layout: B =", ev_np.batch_size, "| factor_equivalence =", ev_np.factor_equivalence,
      "| max|J| =", f"{np.max(np.abs(J_np)):.2f}")
print("macro residual at the reference:", f"{np.max(np.abs(ev_np.eval_macro_single(x_m)[0])):.1e}")

for backend, device in [("torch", "cpu"), ("torch", "mps"), ("mlx", "mlx")]:
    try:
        ev = BatchedJacobianEvaluator(calib, tau, tau_fd, device=device, backend=backend)
    except RuntimeError as exc:            # accelerator absent on this machine
        print(f"{backend}:{device} unavailable ({exc})")
        continue
    J = ev.evaluate_batched_jacobian(x_m)
    print(f"{backend}:{device:4s} dtype={ev.dtype}  max|J - J_numpy| = {np.max(np.abs(J - J_np)):.1e}")

ev_t = BatchedJacobianEvaluator(calib, tau, tau_fd, device="cpu", backend="torch")
J_ad = ev_t.evaluate_batched_jacobian(x_m, ad_mode="forward")   # torch.func.jacfwd, no step size
print("forward-AD vs finite differences:", f"{np.max(np.abs(J_ad - J_np)):.1e}")
```

```text
layout: B = 4 | factor_equivalence = True | max|J| = 21.95
macro residual at the reference: 5.7e-14
torch:cpu  dtype=torch.float64  max|J - J_numpy| = 0.0e+00
torch:mps  dtype=torch.float32  max|J - J_numpy| = 3.5e+01
mlx:mlx  dtype=None  max|J - J_numpy| = 2.8e-08
forward-AD vs finite differences: 9.3e-06
```

The float64 PyTorch CPU Jacobian reproduces the serial NumPy one exactly, and MLX on its float64 CPU stream to $3 \times 10^{-8}$ (MLX's own float64 kernels). The one float32 device in the table, MPS, is off by more than the largest entry: float32 rounding divided by the relative step $10^{-6}$ swamps the derivative, which is exactly why the solvers never polish on it (section 1.3). The $9 \times 10^{-6}$ gap between forward-mode AD and finite differences is the truncation error of the differences; `ad_mode="vjp"` (reverse mode) reproduces the AD Jacobian. (`ev.dtype` is only set on the PyTorch path; MLX picks float32 or float64 per stream.)

### 4.4 Spatial equilibrium on NumPy and MLX

```python
import numpy as np

from puremacro.spatial import AllenArkolakisModel

coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]])
model = AllenArkolakisModel.from_coordinates(coords)

res_np = model.solve_equilibrium(backend="numpy")
res_mlx = model.solve_equilibrium(backend="mlx")
print("iterations: numpy", res_np.iterations, "| mlx", res_mlx.iterations)
print("population (numpy):", np.round(res_np.population, 6))
print("max |dw| =", f"{np.max(np.abs(res_np.wages - res_mlx.wages)):.1e}",
      "| max |dL| =", f"{np.max(np.abs(res_np.population - res_mlx.population)):.1e}",
      "| max |dP| =", f"{np.max(np.abs(res_np.price_index - res_mlx.price_index)):.1e}")
print("labor conservation:", res_np.labor_conservation_residual, res_mlx.labor_conservation_residual)

cf_np = model.solve_counterfactual(productivity_new=np.array([1.2, 1.0, 1.0, 1.0, 1.0]), backend="numpy")
cf_mlx = model.solve_counterfactual(productivity_new=np.array([1.2, 1.0, 1.0, 1.0, 1.0]), backend="mlx")
print(f"welfare change: numpy {cf_np.welfare_pct:+.5f}% | mlx {cf_mlx.welfare_pct:+.5f}%")
```

```text
iterations: numpy 27 | mlx 27
population (numpy): [0.199763 0.199763 0.199763 0.199763 0.200947]
max |dw| = 5.4e-10 | max |dL| = 2.7e-10 | max |dP| = 2.7e-12
labor conservation: 0.0 0.0
welfare change: numpy +4.39312% | mlx +4.39312%
```

Both backends report the same 27 iterations and agree to $10^{-9}$, because the MLX float32 loop hands over to the NumPy float64 loop as soon as it reaches its own floor (section 1.1): the run is certified against the `tol=1e-8` you asked for, not against the device's $10^{-6}$. The welfare effect of a 20% productivity gain in the first region agrees to five decimals. The population sums to exactly $\bar L$ on both backends because the renormalization happens in NumPy.

### 4.5 The device solver and the homotopy on the toy calibration

Continuing from 4.2:

```python
from puremacro.trade.gpu import solve_homotopy_continuation, solve_trade_equilibrium_gpu

res_gpu = solve_trade_equilibrium_gpu(calib, tau=tau, tau_fd=tau_fd, tol=1e-9)
print(f"gpu       converged={res_gpu.converged} iterations={res_gpu.iterations} "
      f"max_residual={res_gpu.max_residual:.1e} w={np.round(res_gpu.w_sol.ravel(), 6)}")
print("max |w_gpu - w_numpy| =", f"{np.max(np.abs(res_gpu.w_sol - ref.w_sol)):.1e}")

meta = res_gpu.metadata
print("device used   :", f"{meta['backend']}:{meta['device']}",
      "| requested:", f"{meta['requested_backend']}:{meta['requested_device']}")
print("layout        :", "B =", meta["batch_size"], "factor_equivalence =", meta["factor_equivalence"])
print("jacobians     :", dict(meta["jacobian_evaluations"]))

steps = []
hom = solve_homotopy_continuation(calib, target_tau=tau, target_tau_fd=tau_fd, tol=1e-9,
                                  initial_step=0.25, max_step=0.5,
                                  callback=lambda lam, r: steps.append((round(lam, 3), r.iterations)))
print("homotopy steps:", steps)
print(f"homotopy  converged={hom.converged} max_residual={hom.max_residual:.1e} "
      f"w={np.round(hom.w_sol.ravel(), 6)}")
```

```text
gpu       converged=True iterations=5 max_residual=8.5e-14 w=[1.000255 0.977642]
max |w_gpu - w_numpy| = 1.2e-14
device used   : torch:cpu | requested: torch:mps
layout        : B = 4 factor_equivalence = True
jacobians     : {'torch:cpu:float64': 4}
homotopy steps: [(0.25, 4), (0.625, 4), (1.0, 4)]
homotopy  converged=True max_residual=1.8e-10 w=[1.000255 0.977642]
```

The device solver reaches the NumPy wages to $10^{-14}$. The metadata says what happened: `"auto"` picked MPS, the float64 policy of section 1.3 silently moved the Jacobian to `torch:cpu`, the reduced layout was chosen because it is exact here ($B = 4$), and every Jacobian was float64. The homotopy took three steps: $0.25$, then $0.625$ after the $1.5\times$ growth of a step that converged in at most 4 iterations, then $1.0$.

### 4.6 A larger synthetic table

The batched GEMM pays off as $M = SN$ grows. The same calls run on any calibration; here an 8-country, 5-sector synthetic table (`synthetic_icio_table`, generic random flows) with a 20% tariff by its first country on all imports:

```python
from puremacro.trade.data import synthetic_icio_table

nc, ns = 8, 5
big = calibrate_trade_model(synthetic_icio_table(nc, ns, seed=3), ns=ns, nc=nc, nfd=3)
rates = np.zeros(nc)
rates[0] = 0.20                                   # country C01 taxes all its imports at 20 percent
ref_big = solve_trade_equilibrium(big, tau=rates, tau_fd=rates, tol=1e-6)
gpu_big = solve_trade_equilibrium_gpu(big, tau=rates, tau_fd=rates, tol=1e-6)
print("B =", gpu_big.metadata["batch_size"], "| converged:", ref_big.converged, gpu_big.converged,
      "| iterations:", ref_big.iterations, gpu_big.iterations)
print("max |w_gpu - w_numpy| =", f"{np.max(np.abs(gpu_big.w_sol - ref_big.w_sol)):.1e}")
```

```text
B = 16 | converged: True True | iterations: 5 4
max |w_gpu - w_numpy| = 1.0e-10
```

Tariff rates by importing country are accepted exactly as by `solve_trade_equilibrium`; `solve_trade_equilibrium_mlx(big, tau=rates, tau_fd=rates)` and `solve_homotopy_continuation(big, target_tau=rates, target_tau_fd=rates)` take the same arguments.

---

## 5. Full API Specification

```text
# puremacro._backend
SUPPORTED = ("numpy", "numba", "mlx", "cupy")
backend_available(name: str) -> bool                     # ValueError for unknown names
available_backends() -> tuple[str, ...]                  # installed, numpy first
get_array_namespace(name: str)                           # numpy | mlx.core | cupy; ImportError with the pip hint
to_numpy(x) -> np.ndarray

# backend= on the model solvers
solve_trade_equilibrium(calib, ..., *, backend="numpy", ...)   # NumPy only; other values raise NotImplementedError
AllenArkolakisModel.solve_equilibrium(tol=1e-8, max_iter=2500, damping=0.35,
                                      backend="numpy") -> AllenArkolakisResult
AllenArkolakisModel.solve_counterfactual(trade_costs_new=None, productivity_new=None, amenity_new=None,
                                         tol=1e-8, max_iter=2500, damping=0.35,
                                         backend="numpy") -> AllenArkolakisResult

# puremacro.trade.gpu.backend
DeviceInfo(backend: Literal["torch", "mlx", "numpy"], device: str, device_name: str,
           total_memory_gb: float | None = None, supports_float64: bool = True, is_uma: bool = False)
has_torch() -> bool
has_mlx() -> bool
select_compute_device(preferred: str | None = None) -> tuple[str, str]
detect_device(preferred: str | None = None) -> DeviceInfo
device_context(device: str | None = None, backend: str | None = None)   # context manager -> (backend, device)
get_memory_usage(device: str | None = None, backend: str | None = None) -> dict[str, float | str]
reset_peak_memory(device: str | None = None, backend: str | None = None) -> None
to_tensor(array, device: str | None = None, dtype=None, backend: str = "torch")
to_numpy(tensor) -> np.ndarray

# puremacro.trade.gpu.batched_jacobian
BatchedJacobianEvaluator(calib: TradeCalibrationResult, tau_a, taufd_a,
                         device: str | None = None, backend: str | None = None,
                         batch_size: int | None = None, factor_equivalence: bool | None = None,
                         *, foreign_saving_units: str = "numeraire")
    .update_tariffs(tau_a, taufd_a) -> None
    .eval_macro_single(xm, compute_full=False)
        -> (f_macro, f_full | None, p_vec, p, y_vec)
    .full_residuals(x_full) -> np.ndarray
    .evaluate_batched_jacobian(xm, f_base=None, eps_fd=1e-6,
                               ad_mode="finite_diff" | "forward" | "vjp", stream=None) -> np.ndarray (B, B)
    .jvp(xm, v) -> np.ndarray
    .eval_macro_batched_torch(X_batch) / .eval_macro_batched_mlx(X_batch, stream=None)

# puremacro.trade.gpu.solver_gpu
solve_trade_equilibrium_gpu(calib, tau=None, tau_fd=None, tauf=None, tauf_fd=None, x0=None,
                            device=None, backend=None, batch_size=None, max_iter=50, tol=2.5e-3,
                            damping=1e-4, ad_mode="finite_diff", base_result=None, verbose=False, *,
                            factor_equivalence=None, foreign_saving_units="numeraire") -> TradeEquilibriumResult

# puremacro.trade.gpu.mlx_solver
solve_trade_equilibrium_mlx(calib, tau=None, tau_fd=None, tauf=None, tauf_fd=None, x0=None,
                            batch_size=None, max_iter=50, tol=2.5e-3, damping=1e-4,
                            ad_mode="finite_diff", base_result=None, verbose=False, *,
                            factor_equivalence=None, foreign_saving_units="numeraire") -> TradeEquilibriumResult

# puremacro.trade.gpu.homotopy
solve_homotopy_continuation(calib, target_tau, target_tau_fd, base_tau=None, base_tau_fd=None, x0=None,
                            device=None, backend=None, batch_size=None,
                            initial_step=0.1, min_step=1e-4, max_step=0.5, tol=2.5e-3, damping=1e-4,
                            max_iter_per_step=30, base_result=None, verbose=False, callback=None, *,
                            factor_equivalence=None, foreign_saving_units="numeraire") -> TradeEquilibriumResult
```

Every name above is importable from `puremacro.trade.gpu`.

### Parameters shared by the accelerated solvers

- `calib`: a `TradeCalibrationResult` from `calibrate_trade_model`.
- `tau`, `tau_fd`: tariffs in any form `solve_trade_equilibrium` accepts: multipliers ($1 + $ rate) of shape `(ns, nc, ns, nc)` / `(ns, nc, nfd, nc)` or the internal `(M, ns, nc)` / `(M, nfd, nc)` layout, or rates by importing country (a scalar or a vector of length `nc`); `None` means no tariffs.
- `tauf`, `tauf_fd` (direct solvers only): tariff rates by importing country, used when `tau` / `tau_fd` is `None`, exactly as on `solve_trade_equilibrium`.
- `x0`: `None` (unit factor returns and the calibrated transfers), the full state vector of length $2M + 4N - 1$, or a macro vector of either layout, $2N$ or $3N$; a reduced vector seeding the full layout sets $r = w$. Any other length raises `ValueError` naming the three it expects.
- `device` / `backend`: explicit values are kept as given; `select_compute_device` fills in whichever of the two is missing. The pair that ends up evaluating the Jacobian may differ from the pair you asked for (the float64 policy of section 1.3); `metadata["device"]` / `["backend"]` record the effective pair and `metadata["requested_device"]` / `["requested_backend"]` the requested one. `solve_trade_equilibrium_mlx` has neither argument.
- `batch_size` / `factor_equivalence`: the layout switch of section 1.4. Prefer `factor_equivalence` (`None` = choose by exactness, `True` = reduced $[\log w; T]$, `False` = full $[\log r; \log w; T]$); `batch_size` must equal $2N$ or $3N$ for the calibration.
- `foreign_saving_units`: `"numeraire"` (default) or `"world_income"`, the foreign-saving closure of `solve_trade_equilibrium`.
- `tol`: absolute infinity-norm tolerance in the calibration's value units, on every equation of the full system for `converged`; `max_iter`: Levenberg-Marquardt iterations (`max_iter_per_step` per continuation step).
- `damping`: initial $\mu$.
- `ad_mode`: `"finite_diff"` (batched differences, default), `"forward"` (forward-mode AD: `torch.func.jacfwd` / `mx.jvp`) or `"vjp"` (reverse mode: `torch.func.jacrev` / `mx.vjp`). An AD mode that fails on the device warns and falls back to finite differences; an unknown value raises `ValueError`.
- `base_result`: baseline equilibrium used by the post-processing for CPI and terms-of-trade indices.
- `verbose`: print per-iteration `max_res`, step and $\mu$ (and, in the homotopy, the step log).
- `callback` (homotopy only): `callback(lambda_value, result)` after each successful step.

---

## 6. Result Interface

### `TradeEquilibriumResult` (accelerated solvers)

The accelerated solvers return the same frozen dataclass as `solve_trade_equilibrium`, post-processed and audited by the same code, so `x_sol`, `p_sol`, `y_sol`, `r_sol`, `w_sol`, `T_sol`, `XN_sol`, the bilateral flow tensors, `gdp`, `imports`, `exports`, `tariffs`, `cpi` and `terms_of_trade` are documented once in the [trade accounting page](trade_accounting.md). The fields the solvers on this page set themselves are:

- `converged`, `iterations`, `max_residual`, `diff` / `residual_norm` (L1 norm) and `residuals`: evaluated on the full system residual in float64.
- `metadata`: `method` (`"gpu_accelerated"` or `"mlx_accelerated"`), the effective `device` and `backend` together with `requested_device` and `requested_backend`, `factor_equivalence`, `batch_size`, `jacobian_evaluations` (a count of Jacobians by `backend:device:dtype`), `ad_mode`, `tol`, `max_iter`, `macro_residual_max`, `solve_duration_seconds` and `memory_usage` (the dictionary of `get_memory_usage`).
- `summary(detailed=False) -> pd.DataFrame`: the solver diagnostic table; `detailed=True` gives the country table (GDP, trade, tariffs, CPI, terms of trade).
- `to_frame(detailed=False)`, `to_markdown(detailed=False)`, `to_latex(detailed=False)`, `to_typst(detailed=False)`: the same table in each format.
- There is no `plot` method on `TradeEquilibriumResult`; figures come from `puremacro.trade.plot`.

### `AllenArkolakisResult` (spatial solver)

Unchanged by the backend: `wages`, `population`, `price_index`, `real_wages`, `welfare`, `consumer_market_access`, `firm_market_access`, `trade_shares`, and for counterfactuals `w_hat`, `L_hat`, `welfare_hat`, `welfare_pct`; diagnostics `converged`, `iterations`, `max_residual`, `labor_conservation_residual`, `spatial_utility_variance`; presentation `summary()`, `to_frame()`, `to_markdown()`, `to_latex()`, `to_typst()` and `plot(kind="spatial")`. Nothing in the result records which backend produced it.

### `DeviceInfo` and the memory dictionary

`DeviceInfo` is the frozen dataclass of section 1.2. `get_memory_usage()` returns `{"backend", "device", "allocated_mb", "peak_mb"}` plus `cache_mb` (CUDA, MLX) or `driver_mb` (MPS); `reset_peak_memory()` resets the peak counters (CUDA, MLX) and empties the device caches (CUDA, MPS, MLX), and `device_context(...)` calls it on exit and yields the `(backend, device)` pair without cross-checking the two, echoing the strings you passed.

---

## 7. Caveats and Limitations

- **Model scope.** The device layer solves the consistent-accounting model with Cobb-Douglas intermediates (`sigma = 0`) and lump-sum rebates of tariff revenue. For CES intermediates, other fiscal closures or the capacity extension use `solve_trade_equilibrium`.
- **Two layers, two vocabularies.** The namespace layer takes `"numpy"`, `"mlx"`, `"cupy"` (`"numba"` warns and falls back, and anything else, `"torch"` included, raises `ValueError: Unknown backend`). The device layer takes `"auto"`, `"numpy"`, `"cpu"`, `"cuda"`, `"cuda:N"`, `"mps"`, `"mlx"`, `"gpu"` and raises `ValueError` on anything else.
- **Precision.** MPS has no float64 and MLX's Metal stream is float32, so an explicitly requested float32 device is used for the coarse phase only and warns when it is. Expect agreement to rounding on the float64 paths and an unusable Jacobian on a float32 one (Example 4.3).
- **Tolerances.** `tol=2.5e-3` on both the NumPy and the accelerated trade solvers is absolute and loose; tighten it explicitly for parity work.
- **Import cost.** `import puremacro.trade` does not pull in PyTorch or MLX; both are imported lazily at the first call that needs them, which pays the framework import.
- **Unverified paths.** The CuPy branch of the spatial solver and the CUDA branch of the device layer were not executed while writing this page (no NVIDIA hardware); they are covered by the fallback logic and by tests that skip when the accelerator is absent.
- **No speedup on small problems.** All device paths add transfer and launch latency; the toy calibration and the 5-region model run slower on an accelerator than on NumPy. [Performance & Benchmarks](benchmarks.md) currently publishes CPU timings only.
- **Offline and deterministic.** Nothing here touches the network; the accelerated paths are only as deterministic as the device reductions, and the NumPy path is the reference for reproducibility claims.

---

## References

- Allen, T., & Arkolakis, C. (2014). "Trade and the Topography of the Spatial Economy." *The Quarterly Journal of Economics*, 129(3), 1085–1140.
- Allgower, E. L., & Georg, K. (1990). *Numerical Continuation Methods: An Introduction*. Springer Series in Computational Mathematics 13. Berlin: Springer.
- Armijo, L. (1966). "Minimization of functions having Lipschitz continuous first partial derivatives." *Pacific Journal of Mathematics*, 16(1), 1–3.
- Caliendo, L., & Parro, F. (2015). "Estimates of the Trade and Welfare Effects of NAFTA." *The Review of Economic Studies*, 82(1), 1–44.
- Levenberg, K. (1944). "A method for the solution of certain non-linear problems in least squares." *Quarterly of Applied Mathematics*, 2(2), 164–168.
- Marquardt, D. W. (1963). "An algorithm for least-squares estimation of nonlinear parameters." *Journal of the Society for Industrial and Applied Mathematics*, 11(2), 431–441.
