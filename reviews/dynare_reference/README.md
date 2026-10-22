# Dynare Reference Validation for Issue #177 (STEADY_STATE in Perfect Foresight)

## Overview
This directory contains the Dynare 8 reference models, MATLAB R2026a execution logs, simulation outputs, and automated parity verification scripts for GitHub Issue #177.

## Environment
- **Platform**: Apple Silicon macOS ARM64 (Darwin 27.2.0)
- **MATLAB**: MATLAB R2026a (`26.1.0.3346908 (R2026a) Update 5`) native arm64 headless execution
- **Dynare**: Dynare 8-unstable (`/Applications/Dynare/8-2026-05-26-1803/matlab`) native arm64 MEX
- **Execution Command**:
  ```bash
  /Applications/MATLAB_R2026a.app/bin/matlab -batch "addpath('/Applications/Dynare/8-2026-05-26-1803/matlab'); run_dynare_all;"
  ```

## Reference Models Evaluated

### 1. Economically Meaningful Model: RBC with Fiscal Share of Steady-State Output
- **Model File**: `rbc_steady_state.mod`
- **Resource Constraint**:
  $$k_t = y_t - c_t - g_t + (1 - \delta) k_{t-1}$$
  $$g_t = gy \cdot \text{STEADY\_STATE}(y)$$
  $$a_t = \rho a_{t-1} + e_t, \quad e_1 = 0.05$$
- **Dynare Behavior**:
  $g_t$ remains strictly constant at $gy \cdot \bar{y} \approx 0.60306554$ across all 40 periods.
- **puremacro Parity**:
  - Max absolute difference across all 5 variables $(c, k, y, a, g)$ and all 40 periods:
    $$\max |y_{\text{puremacro}} - y_{\text{dynare}}| = 7.105 \times 10^{-14} \le 10^{-8}$$
  - Variable-by-variable max difference:
    - $c$: $5.329 \times 10^{-15}$
    - $k$: $7.105 \times 10^{-14}$
    - $y$: $4.885 \times 10^{-15}$
    - $a$: $9.140 \times 10^{-17}$
    - $g$: $2.220 \times 10^{-16}$

### 2. Minimal Issue #177 Model
- **Model File**: `minimal_177.mod`
- **Equations**:
  $$y_t = 1 + \rho (y_{t-1} - 1) + e_t, \quad x_t = \text{STEADY\_STATE}(y), \quad e_1 = 0.5$$
- **Dynare Behavior**:
  $x_t$ is flat at $1.0$ for all $t=1 \dots 10$.
- **puremacro Parity**:
  - Max absolute difference: $0.000 \times 10^0$ (exact integer match).

### 3. Permanent Shock: `endval` WITH `steady`
- **Model File**: `endval_with_steady.mod`
- **Specification**:
  `initval; y=1; x=1; end; steady; endval; e=0.5; y=2; x=2; end; steady;`
- **Dynare Behavior**:
  `oo_.steady_state` is updated to $[2, 2]$. $x_t$ is flat at $2.0$ along the transition path $t=1 \dots 10$.
- **puremacro Parity**:
  - Max absolute difference: $0.000 \times 10^0$ (exact integer match).

### 4. Permanent Shock: `endval` WITHOUT `steady`
- **Model File**: `endval_no_steady.mod`
- **Specification**:
  `initval; y=1; x=1; end; steady; endval; e=0.5; y=2; x=2; end;`
- **Dynare Behavior**:
  `oo_.steady_state` receives the `endval` values $[2, 2]$. $x_t$ is flat at $2.0$ along the transition path $t=1 \dots 10$.
- **puremacro Parity**:
  - Max absolute difference: $0.000 \times 10^0$ (exact integer match).

## Verification
Run the automated verification script:
```bash
python reviews/dynare_reference/verify_dynare_parity.py
```
All 4 models pass parity assertion $\le 10^{-8}$.
