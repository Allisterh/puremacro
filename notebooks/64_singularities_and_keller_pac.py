# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Folds, singular Jacobians and Keller continuation
#
# **How can we tell whether an equilibrium solver fails at a fold, from a bad starting point, or because the equilibrium stops existing, as a policy parameter moves?**
#
# Four experiments, each on inputs that are labelled for what they are:
#
# | Experiment | Input | Oracle |
# |---|---|---|
# | 1. A textbook fold | the hand-written residual $F=(x_1^2-\lambda,\;x_2-x_1)$ | closed-form branches $x_1=\pm\sqrt\lambda$ (independent, derived by hand) |
# | 2. puremacro's trade model | a hand-built two-country, two-sector social accounting matrix; the numbers are synthetic | Newton against Keller continuation (internal: two puremacro routes); the Hawkins-Simon threshold against NumPy eigenvalues of a matrix built by hand (internal: same calibration) |
# | 3. The existence certificate at scale | a seeded synthetic input-output table, 60 countries by 12 sectors (`synthetic_icio_table(60, 12, seed=0)`); generic random draws, not a model of any economy | the Collatz-Wielandt bounds, which certify themselves (internal); NumPy eigenvalues of the same matrix (internal) |
# | 4. A stiff linear step | a random 60x60 matrix with one planted stiff coordinate (seed 42) | the residual of each returned step, recomputed (internal) |
#
# The exercise at the end uses the Laffer curve, whose peak $t^*=1/(1+\varepsilon)$ is known in closed form.

# %% [markdown]
# ## The method in math
#
# Write the equilibrium conditions as $F(x,\lambda)=0$, $x\in\mathbb R^n$, with a scalar parameter $\lambda$ (a tariff path, a revenue target). A **fold** (a turning point, or saddle-node bifurcation) is a solution where $F_x$ is singular and the branch turns back in $\lambda$. **Natural-parameter continuation** fixes $\lambda$ and runs Newton in $x$, $x^{k+1}=x^k-F_x^{-1}F(x^k,\lambda)$, so beyond a fold it has no root to find. **Keller's pseudo-arclength continuation (PAC)** makes $\lambda$ an unknown and moves a distance $\Delta s$ along the unit tangent $t=(t_x,t_\lambda)$:
#
# $$\underbrace{\begin{pmatrix}F_x & F_\lambda\\ t_{x}^{\top} & t_{\lambda}\end{pmatrix}}_{\text{bordered matrix}}t^{\text{new}}=\begin{pmatrix}0\\1\end{pmatrix},\qquad (\hat x,\hat\lambda)=(x_k,\lambda_k)+\Delta s\,t\quad\text{(predictor)},$$
#
# $$\text{corrector: Newton on}\quad G(x,\lambda)=\begin{pmatrix}F(x,\lambda)\\ t_x^{\top}(x-x_k)+t_\lambda(\lambda-\lambda_k)-\Delta s\end{pmatrix}=0,\ \text{whose Jacobian is the bordered matrix.}$$
#
# At a simple fold $F_x$ loses one direction but $F_\lambda$ lies outside its range, so the bordered matrix stays invertible while $t_\lambda$ passes through zero: **a fold is one sign change of $t_\lambda$**. For the toy, $\det F_x=2x_1$ and $\kappa(F_x)\propto 1/|x_1|$ near the fold at $\lambda=0$.
#
# **Existence.** With Leontief intermediates, unit-cost prices solve $p^\top=p^\top B_\tau+v^\top$ with $B_{\tau,ij}=a_{ij}\,m_{ij}/(1-t_j)$, where $m_{ij}=1+\tau$ on cross-border flows, $t_j$ is the production tax and $v>0$ collects unit factor costs. A positive price vector exists if and only if $\rho(B_\tau)<1$ (Hawkins-Simon); it then exists for every $v>0$, but nothing in the condition says that the factor prices an equilibrium needs stay positive. For any $u>0$, $\min_i (B_\tau u)_i/u_i\le\rho(B_\tau)\le\max_i (B_\tau u)_i/u_i$ (Collatz-Wielandt): an upper bound below one certifies productivity, a lower bound of at least one certifies failure, and an interval that straddles one proves nothing.

# %% [markdown]
# ## Intuition
#
# **Intuition.** Think of $\lambda$ as a policy dial and $x$ as the equilibrium it produces. Along a smooth branch a small turn of the dial moves the equilibrium a little, so warm-starting Newton from the last solution works. At a fold the branch turns back: past it there is no nearby equilibrium, and at it the Jacobian loses a direction, so Newton slows from quadratic to linear convergence. PAC stops treating the dial as given and walks along the curve of solutions; the column $F_\lambda$ supplies the direction that $F_x$ has lost. The Laffer curve is the economic prototype: no tax rate raises more than the peak revenue, and a path of rising revenue targets turns at the peak onto the high-tax side. Not every failure is a fold, though. A solver can fail because it starts too far from the answer (any continuation cures that), or because the equilibrium stops existing at a boundary where a price reaches zero. A fold monitor has to tell these apart, and a certificate such as Hawkins-Simon rules out only some of them.

# %% [markdown]
# ## Worked code
#
# The preamble loads the plotting style and the trade solvers. The random generator is used only by Experiment 4.

# %%
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.optimize import brentq

_cwd = Path.cwd()
sys.path.insert(0, str(_cwd if (_cwd / "_nbstyle.py").exists() else _cwd / "notebooks"))
import _nbstyle
_nbstyle.apply_style()

from puremacro.trade import calibrate_trade_model, compute_equilibrium_residuals, solve_trade_equilibrium
from puremacro.trade.data import synthetic_icio_table
from puremacro.trade.solver import (
    check_hawkins_simon_viability,
    clamp_wage_displacement,
    solve_fold_continuation_step,
    solve_keller_pac,
    svd_clamped_newton_step,
)

rng = np.random.default_rng(42)

# %% [markdown]
# ### Experiment 1: a textbook fold
#
# Two helpers serve the whole notebook: Newton at a fixed $\lambda$, and PAC. `pac` records the $\lambda$-component of the unit tangent at every node, including the start, so the fold test can look for a sign change rather than for a negative value.

# %%
def newton_fixed(F, Fx, x0, lam, max_iter=50, tol=1e-10):
    """Newton in x with lam held fixed. Returns the last iterate, success, and the path of x[0]."""
    x = np.array(x0, dtype=float)
    path = [x[0]]
    for _ in range(max_iter):
        f, J = F(x, lam), Fx(x, lam)
        if not (np.all(np.isfinite(f)) and np.all(np.isfinite(J))):
            return x, False, np.array(path)                      # the iterate left the domain of F
        if np.max(np.abs(f)) < tol:
            return x, True, np.array(path)
        try:
            x = x - la.solve(J, f)
        except la.LinAlgError:                                   # exactly singular Jacobian
            return x, False, np.array(path)
        path.append(x[0])
    return x, bool(np.max(np.abs(F(x, lam))) < tol), np.array(path)


def pac(F, Fx, Flam, x0, lam0, ds, n_steps, lam_direction, until=None, tol=1e-12):
    """Keller pseudo-arclength continuation of F(x, lam) = 0 from the solution (x0, lam0)."""
    def bordered(z, t):
        x, lam = z[:-1], z[-1]
        return np.block([[Fx(x, lam), Flam(x, lam)[:, None]], [t[None, :]]])

    z = np.r_[x0, lam0]
    # First tangent: the null vector of [F_x, F_lam], pointed the way lam should move.
    t = la.null_space(np.column_stack([Fx(z[:-1], z[-1]), Flam(z[:-1], z[-1])]))[:, 0]
    t = t if np.sign(t[-1]) == np.sign(lam_direction) else -t
    rec = {"z": [z.copy()], "t_lam": [t[-1]], "cond_Fx": [np.linalg.cond(Fx(z[:-1], z[-1]))],
           "cond_bordered": [np.linalg.cond(bordered(z, t))], "max_F": [np.max(np.abs(F(z[:-1], z[-1])))]}
    for _ in range(n_steps):
        zk = z + ds * t                                          # predictor
        for _ in range(20):                                      # corrector
            g = np.r_[F(zk[:-1], zk[-1]), t @ (zk - z) - ds]
            if np.max(np.abs(g)) < tol:
                break
            zk = zk - la.solve(bordered(zk, t), g)
        t = la.solve(bordered(zk, t), np.eye(len(z))[-1])     # new tangent, t_old . t_new > 0
        t /= np.linalg.norm(t)
        z = zk
        rec["z"].append(z.copy())
        rec["t_lam"].append(t[-1])
        rec["cond_Fx"].append(np.linalg.cond(Fx(z[:-1], z[-1])))
        rec["cond_bordered"].append(np.linalg.cond(bordered(z, t)))
        rec["max_F"].append(np.max(np.abs(F(z[:-1], z[-1]))))
        if until is not None and until(z):
            break
    return {key: np.array(val) for key, val in rec.items()}


def F_toy(x, lam):
    return np.array([x[0] ** 2 - lam, x[1] - x[0]])


def Fx_toy(x, lam):
    return np.array([[2.0 * x[0], 0.0], [-1.0, 1.0]])


def Flam_toy(x, lam):
    return np.array([-1.0, 0.0])


# Natural-parameter continuation: lower lam step by step, warm-starting Newton at the last solution.
natural = []
x_warm = np.array([0.2, 0.2])
for lam in (0.03, 0.02, 0.01, 0.0, -0.01):
    x_new, ok, path = newton_fixed(F_toy, Fx_toy, x_warm, lam)
    natural.append({"lam": lam, "ok": ok, "iters": len(path) - 1, "x1": x_new[0],
                    "max_F": np.max(np.abs(F_toy(x_new, lam))), "path": path})
    if ok:
        x_warm = x_new
    print(f"lam = {lam:+.2f}: converged {ok!s:5}  iterations {len(path) - 1:2d}  "
          f"x1 = {x_new[0]:+.2e}  max|F| = {natural[-1]['max_F']:.1e}")

halving = natural[3]["path"][1:] / natural[3]["path"][:-1]
print(f"At lam = 0 all {len(halving)} Newton steps multiply x1 by {halving.mean():.6f}; "
      f"the state error {natural[3]['x1']:.1e} is the square root of the residual {natural[3]['max_F']:.1e}.")

assert [r["ok"] for r in natural] == [True, True, True, True, False], "Newton must succeed down to the fold and fail past it"
assert np.allclose(halving, 0.5, atol=1e-12), "at the singular root Newton is only linear, with ratio 1/2"
assert max(r["iters"] for r in natural[:3]) < natural[3]["iters"]

# %%
# PAC from the same start, heading for the fold (lam decreasing).
toy = pac(F_toy, Fx_toy, Flam_toy, np.array([0.2, 0.2]), 0.04, ds=0.08, n_steps=14, lam_direction=-1.0)
x1_pac, lam_pac = toy["z"][:, 0], toy["z"][:, -1]
sign_t = np.sign(toy["t_lam"])
flips = np.flatnonzero(np.diff(sign_t))
lower = x1_pac < 0
branch_err = np.max(np.abs(x1_pac[lower] + np.sqrt(lam_pac[lower])))

print("t_lambda at the nodes:", np.array2string(toy["t_lam"], precision=3, max_line_width=120))
assert sign_t[0] < 0 < sign_t[-1] and len(flips) == 1, f"t_lambda must change sign exactly once, not {len(flips)} times"
print(f"Sign changes of t_lambda: {len(flips)}, between nodes {flips[0]} and {flips[0] + 1} "
      f"(x1 from {x1_pac[flips[0]]:+.3f} to {x1_pac[flips[0] + 1]:+.3f}); smallest lambda visited {lam_pac.min():.1e}")
print(f"{lower.sum()} nodes on the lower branch, max|x1 + sqrt(lambda)| = {branch_err:.1e}; max|F| on the path {toy['max_F'].max():.1e}")
print(f"cond(F_x) at the nodes peaks at {toy['cond_Fx'].max():.3g}; the bordered matrix stays between "
      f"{toy['cond_bordered'].min():.3g} and {toy['cond_bordered'].max():.3g}")

# The fold test: one sign change (checked above), at the crossing of x1 = 0, and the far side on the lower branch.
assert x1_pac[flips[0]] > 0 > x1_pac[flips[0] + 1], "the sign change must coincide with x1 crossing zero"
assert lower.sum() >= 5 and branch_err < 1e-10, "the nodes past the fold must lie on x1 = -sqrt(lambda)"
assert lam_pac.min() >= 0.0 and toy["max_F"].max() < 1e-10

# %%
# Figure 1. Conditioning along the analytic branch x = (s, s), lambda = s^2, on a grid that
# reaches |x1| = 1e-6: the 15 PAC nodes are too coarse to show the blow-up by themselves.
s_grid = np.r_[-np.logspace(np.log10(0.6), -6, 300), np.logspace(-6, np.log10(0.6), 300)]
cond_Fx_branch = np.array([np.linalg.cond(Fx_toy(np.array([s, s]), s * s)) for s in s_grid])
cond_bordered_branch = []
for s in s_grid:
    t_branch = np.array([1.0, 1.0, 2.0 * s]) / np.sqrt(2.0 + 4.0 * s * s)   # unit tangent of the branch
    J_b = np.block([[Fx_toy(np.array([s, s]), s * s), Flam_toy(None, None)[:, None]], [t_branch[None, :]]])
    cond_bordered_branch.append(np.linalg.cond(J_b))
cond_bordered_branch = np.array(cond_bordered_branch)
print(f"On the dense branch: max cond(F_x) = {cond_Fx_branch.max():.2e}, "
      f"bordered matrix between {cond_bordered_branch.min():.3g} and {cond_bordered_branch.max():.3g}")
assert cond_Fx_branch.max() > 1e5 and cond_bordered_branch.max() < 2.0

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
lam_line = np.linspace(0.0, lam_pac.max(), 300)
ax0.plot(lam_line, np.sqrt(lam_line), **_nbstyle.S1, label=r"upper branch $x_1=+\sqrt{\lambda}$")
ax0.plot(lam_line, -np.sqrt(lam_line), **_nbstyle.S2, label=r"lower branch $x_1=-\sqrt{\lambda}$")
ax0.plot(lam_pac, x1_pac, ls="none", marker="o", ms=5, mfc="none", mec=_nbstyle.TINTA, label="PAC nodes")
ok_runs = [r for r in natural if r["ok"]]
ax0.plot([r["lam"] for r in ok_runs], [r["x1"] for r in ok_runs], ls="none", marker="s", ms=5,
         color=_nbstyle.S3["color"], label=r"Newton at fixed $\lambda$")
failed = natural[-1]
ax0.plot(failed["lam"], failed["x1"], ls="none", marker="x", ms=8, mew=2, color=_nbstyle.TINTA,
         label=rf"last Newton iterate at $\lambda={failed['lam']:.2f}$ (no root)")
ax0.axvline(0.0, color=_nbstyle.SPINE, lw=0.8, ls=":")
ax0.set_xlabel(r"continuation parameter $\lambda$")
ax0.set_ylabel(r"$x_1$")
ax0.set_title("Solution branches of the toy fold")
ax0.set_ylim(-0.62, 1.05)
ax0.legend(loc="upper left", fontsize=8)

ax1.semilogy(s_grid, cond_Fx_branch, **_nbstyle.S1, label=r"$\kappa(F_x)$ on the branch")
ax1.semilogy(s_grid, cond_bordered_branch, **_nbstyle.S2, label=r"$\kappa$(bordered) on the branch")
ax1.plot(x1_pac, toy["cond_Fx"], ls="none", marker="o", ms=5, mfc="none", mec=_nbstyle.TINTA,
         label=r"$\kappa(F_x)$ at the PAC nodes")
ax1.set_xlabel(r"$x_1$ along the branch (fold at $x_1=0$)")
ax1.set_ylabel(r"2-norm condition number $\kappa$")
ax1.set_title("Conditioning near the fold")
ax1.legend(loc="upper right", fontsize=8)

# %% [markdown]
# ### Experiment 2: puremacro's trade model
#
# A hand-built social accounting matrix for two countries, A (`C00`) and B (`C01`), with two sectors each. The numbers are synthetic. The goods accounts clear; A imports more than it exports, and the gap is held fixed as foreign saving.

# %%
nc_cge, ns_cge, nfd_cge = 2, 2, 3
data_cge = np.zeros((ns_cge * nc_cge + 3, ns_cge * nc_cge + nfd_cge * nc_cge))
data_cge[:4, :4] = np.array([[20.0, 10.0, 5.0, 2.0],
                             [8.0, 25.0, 2.0, 4.0],
                             [5.0, 5.0, 12.0, 18.0],
                             [10.0, 10.0, 18.0, 22.0]])
y_cge = np.array([100.0, 150.0, 120.0, 180.0])            # gross output by country-sector
va_fac = y_cge - data_cge[:4, :4].sum(axis=0) - 0.05 * y_cge
data_cge[4, :4] = 0.05 * y_cge                             # production taxes
data_cge[5, :4] = (2.0 / 3.0) * va_fac                     # the two factor rows
data_cge[6, :4] = (1.0 / 3.0) * va_fac
fd_rows = y_cge - data_cge[:4, :4].sum(axis=1)
home_bias, abroad = np.array([0.50, 0.25, 0.05]), np.array([0.10, 0.08, 0.02])
for i in range(4):
    split = np.r_[home_bias, abroad] if i < 2 else np.r_[abroad, home_bias]
    data_cge[i, 4:10] = fd_rows[i] * split
data_cge[4, 4:] = 0.02 * data_cge[:4, 4:].sum(axis=0)
calib_cge = calibrate_trade_model(data_cge, ns=ns_cge, nc=nc_cge, nfd=nfd_cge, validate=True)

exports_A = data_cge[0:2, [2, 3, 7, 8, 9]].sum()
imports_A = data_cge[2:4, [0, 1, 4, 5, 6]].sum()
print(f"A exports {exports_A:.1f} and imports {imports_A:.1f}; baseline foreign balances {calib_cge.invforT.ravel()}")

# %% [markdown]
# **Where the equilibrium stops existing.** `solve_trade_equilibrium` uses consistent accounting ([docs/trade_accounting.md](../docs/trade_accounting.md)). Follow the equilibrium as the tariff rises, warm-starting Newton at the previous solution (natural-parameter continuation in the tariff), once with Leontief intermediates ($\sigma=0$) and once with unit elasticity ($\sigma=1$). Track A's factor price relative to B's, which does not depend on the numeraire, and the smallest singular value of the Jacobian $F_x$ at each solution. Then ask PAC for a tariff past the end of each path and read its fold monitor.

# %%
def smallest_singular_value(x, tau, sigma, h=1e-7):
    """sigma_min of F_x at x, by central differences of the public residual function."""
    F = lambda z: compute_equilibrium_residuals(z, calib_cge, tau=tau, tau_fd=tau, sigma=sigma)
    J = np.column_stack([(F(x + h * e) - F(x - h * e)) / (2 * h) for e in np.eye(x.size)])
    return np.linalg.svd(J, compute_uv=False)[-1]


def tariff_path(sigma, taus):
    """Warm-started Newton along a tariff grid; stops at the first failure.

    Returns the tariffs reached, w_A / w_B and sigma_min(F_x) at each solution."""
    x, rel, smin = None, [], []
    for tau in taus:
        res = solve_trade_equilibrium(calib_cge, tau=tau, tau_fd=tau, sigma=sigma, x0=x, tol=1e-9)
        if not res.converged:
            break
        x = res.x_sol
        rel.append(res.w_sol.ravel()[0] / res.w_sol.ravel()[1])
        smin.append(smallest_singular_value(x.ravel(), tau, sigma))
    return np.asarray(taus[: len(rel)]), np.asarray(rel), np.asarray(smin)


tau_leo, rel_leo, smin_leo = tariff_path(0.0, np.r_[np.arange(0.0, 4.2, 0.05), np.arange(4.2, 4.3, 0.005)])
tau_sub, rel_sub, smin_sub = tariff_path(1.0, np.r_[np.arange(0.0, 1.05, 0.05), np.arange(1.05, 1.1, 0.0025)])
tau_zero = np.polyval(np.polyfit(rel_leo[-4:], tau_leo[-4:], 1), 0.0)   # where w_A / w_B reaches 0

with warnings.catch_warnings():         # silence overflow warnings while log w_A heads to -infinity
    warnings.simplefilter("ignore", RuntimeWarning)
    stall = solve_keller_pac(calib_cge, tau_target=4.4, sigma=0.0, tol=1e-9)
tau_stall = 4.4 * stall.metadata["final_lambda"]
turn = solve_keller_pac(calib_cge, tau_target=1.1, sigma=1.0, tol=1e-9)
event = turn.metadata["fold_points"][0]

print(f"sigma = 0: w_A/w_B = {rel_leo[0]:.3f} at 0%, {rel_leo[np.argmin(np.abs(tau_leo - 4.0))]:.3f} at 400%, "
      f"{rel_leo[-1]:.5f} at {tau_leo[-1]:.1%}; linear extrapolation reaches zero at a tariff of {tau_zero:.4f}")
print(f"  sigma_min(F_x) from {smin_leo[0]:.1e} at 0% to {smin_leo[-1]:.1e} at the last solution")
print(f"  PAC to 440%: converged {stall.converged}, stalls at an effective tariff of {tau_stall:.4f}, "
      f"fold events {len(stall.metadata['fold_points'])}, raw detections {stall.metadata['fold_detections']}")
print(f"sigma = 1: last solution at {tau_sub[-1]:.4f} with w_A/w_B = {rel_sub[-1]:.3f}; "
      f"sigma_min(F_x) falls from {smin_sub[0]:.1e} to {smin_sub[-1]:.1e}")
print(f"  PAC to 110%: converged {turn.converged}; fold monitor: {len(turn.metadata['fold_points'])} event of kind "
      f"'{event['kind']}', first detected at an effective tariff of {1.1 * event['lambda']:.4f}, "
      f"lambda reversed: {event['lambda_reversed']}, smallest log factor price {event['min_log_factor_price']:.2f}")

assert (not stall.converged) and stall.metadata["fold_points"] == [], "a corner: no fold event"
assert abs(tau_zero - tau_stall) < 0.01, "the warm-started path and PAC must locate the same boundary"
assert rel_leo[-1] < 1e-3 and rel_sub[-1] > 0.1, "sigma = 0 ends at a zero factor price, sigma = 1 does not"
assert smin_sub[-1] < 0.05 * smin_sub[0], "sigma = 1: the Jacobian becomes nearly singular where the path ends"
assert (not turn.converged) and event["kind"] == "lambda_reversal" and event["lambda_reversed"]

# %% [markdown]
# **What the existence certificate says on this table.** Find the tariff at which the Hawkins-Simon certificate fails, by bisection on the public check (it raises `ValueError` when a schedule is not certified; the helper reads the verdict and the bounds from the message). Check it by a second route: NumPy's eigenvalues of $B_\tau$ built by hand from the calibration.

# %%
def hawkins_simon(calib, tau):
    """('viable' | 'violated' | 'unresolved' | 'error', lower bound, upper bound) from the public check."""
    try:
        cert = check_hawkins_simon_viability(calib, tau=tau)
        return "viable", cert.cw_lower, cert.cw_upper
    except ValueError as err:
        msg = str(err)
        status = "violated" if "violates" in msg else "unresolved" if "unresolved" in msg else "error"
        found = re.findall(r"\[([-+0-9.eE]+), ([-+0-9.eE]+)\]", msg)
        low, high = (float(found[0][0]), float(found[0][1])) if found else (np.nan, np.nan)
        return status, low, high


def hs_threshold(calib, lo, hi, tol):
    """Bisect between a certified-viable and a certified-violated tariff."""
    assert hawkins_simon(calib, lo)[0] == "viable" and hawkins_simon(calib, hi)[0] == "violated"
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        status = hawkins_simon(calib, mid)[0]
        assert status in ("viable", "violated"), f"certificate {status} at {mid}"
        lo, hi = (mid, hi) if status == "viable" else (lo, mid)
    return lo, hi


def B_by_hand(calib, tau):
    """B_tau = a * m / (1 - t) with m = 1 + tau across borders; taxes clipped as the public check does."""
    ns, nc = calib.n_sectors, calib.n_countries
    a = np.asarray(calib.a, dtype=float).reshape((ns * nc, ns * nc), order="F")
    tax = np.clip(np.asarray(calib.tax, dtype=float).flatten(order="F"), -0.9, 0.999)
    country = np.arange(ns * nc) // ns
    m = np.where(country[:, None] != country[None, :], 1.0 + tau, 1.0)
    return a * m / (1.0 - tax)[None, :]


hs_lo, hs_hi = hs_threshold(calib_cge, 0.0, 20.0, tol=1e-6)
tau_eig = brentq(lambda tau: np.max(np.abs(np.linalg.eigvals(B_by_hand(calib_cge, tau)))) - 1.0, 0.0, 20.0, xtol=1e-9)
status_stall, low_stall, high_stall = hawkins_simon(calib_cge, tau_stall)
taus_cert = np.linspace(0.0, 10.0, 101)
rho_upper = np.array([check_hawkins_simon_viability(calib_cge, tau=t).cw_upper if t < hs_lo
                      else np.nan for t in taus_cert])
print(f"Certificate threshold on the 2x2 table: between {hs_lo:.6f} and {hs_hi:.6f}; eigenvalue route {tau_eig:.6f}")
print(f"At {tau_stall:.0%}, where the sigma = 0 branch ends: {status_stall}, rho(B) in [{low_stall:.4f}, {high_stall:.4f}]")

assert abs(0.5 * (hs_lo + hs_hi) - tau_eig) < 1e-4, "the certificate and the eigenvalues must agree"
assert status_stall == "viable" and high_stall < 0.9, "Hawkins-Simon certifies prices where the branch ends"

# %%
# Figure 2. Left: A's relative factor price along the tariff path. Right: the certificate on the same table.
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
ax0.plot(tau_leo, rel_leo, **_nbstyle.S1, label=r"Leontief, $\sigma=0$")
ax0.plot(tau_sub, rel_sub, **_nbstyle.S2, label=r"unit elasticity, $\sigma=1$")
ax0.axvline(tau_stall, color=_nbstyle.SPINE, lw=1.0, ls=":", label=rf"PAC stalls ($\sigma=0$), {tau_stall:.2f}")
ax0.axvline(tau_sub[-1], color=_nbstyle.NOTA, lw=1.0, ls="--", label=rf"path turns ($\sigma=1$), {tau_sub[-1]:.2f}")
ax0.set_xlabel(r"tariff rate $\tau$ (1 = 100%)")
ax0.set_ylabel(r"factor price of A relative to B, $w_A/w_B$")
ax0.set_title("A's factor price along the tariff path")
ax0.set_ylim(bottom=0.0)
ax0.legend(loc="upper right", fontsize=8)

ax1.plot(taus_cert, rho_upper, **_nbstyle.S1, label=r"certified upper bound on $\rho(B_\tau)$")
ax1.axhline(1.0, color=_nbstyle.SPINE, lw=1.0, ls="-")
ax1.axvline(tau_stall, color=_nbstyle.SPINE, lw=1.0, ls=":")
ax1.axvline(tau_eig, color=_nbstyle.NOTA, lw=1.0, ls="--")
ax1.text(tau_stall + 0.15, 0.95, "equilibrium branch ends\n" rf"($\sigma=0$) at {tau_stall:.2f}",
         color=_nbstyle.TEXTO, fontsize=8, va="top")
ax1.text(tau_eig - 0.15, 0.55, f"certificate fails\nat {tau_eig:.2f}", color=_nbstyle.TEXTO, fontsize=8, ha="right")
ax1.set_xlabel(r"tariff rate $\tau$ (1 = 100%)")
ax1.set_ylabel(r"spectral radius bound")
ax1.set_title("Hawkins-Simon certificate, same 2x2 table")
ax1.legend(loc="lower right", fontsize=8)

# %% [markdown]
# ### Experiment 3: the certificate at scale
#
# A seeded synthetic table with 60 countries and 12 sectors gives a 720-row matrix $B_\tau$; its flows are generic random draws, so the numbers describe the generator, not any production network. The public check clips production tax rates to $[-0.9, 0.999]$, so the bounds are for that clipped matrix (on this table no rate is clipped). NumPy's eigenvalues of the hand-built matrix check the threshold.

# %%
nc_big, ns_big = 60, 12
calib_big = calibrate_trade_model(synthetic_icio_table(nc_big, ns_big, seed=0), ns=ns_big, nc=nc_big, nfd=3,
                                  validate=True)
codes = list(calib_big.country_codes)
n_clipped = int(np.sum(np.asarray(calib_big.tax) < -0.9))
print(f"{len(codes)} synthetic countries ({codes[0]} to {codes[-1]}), {calib_big.n_sectors * len(codes)} "
      f"country-sectors; {n_clipped} tax rates below -0.9 are clipped")
big = {tau: hawkins_simon(calib_big, tau) for tau in (0.0, 0.25, 8.0)}
for tau, (status, low, high) in big.items():
    print(f"tariff {tau:4.0%}: {status:9s} rho(B) in [{low:.6f}, {high:.6f}]")
big_lo, big_hi = hs_threshold(calib_big, 4.0, 8.0, tol=1e-5)
print(f"The certificate switches from viable to violated between tariffs of {big_lo:.5f} and {big_hi:.5f}")
tau_eig_big = brentq(lambda tau: np.max(np.abs(np.linalg.eigvals(B_by_hand(calib_big, tau)))) - 1.0,
                     big_lo - 0.01, big_hi + 0.01, xtol=1e-9)
print(f"NumPy's eigenvalues of the {B_by_hand(calib_big, 0.0).shape[0]}-row matrix put rho(B) = 1 at {tau_eig_big:.6f}")

assert big[0.25][0] == "viable" and big[0.25][2] < 1.0, "the upper bound, not an estimate, certifies"
assert big[8.0][0] == "violated" and big[8.0][1] >= 1.0, "800% must be a certified violation, not unresolved"
assert big_lo - 1e-3 < tau_eig_big < big_hi + 1e-3, "the certificate and the eigenvalues must agree"

# %% [markdown]
# ### Experiment 4: what does a clamped Newton step buy?
#
# `puremacro.trade.solver` ships three helpers that bound a Newton step for log factor prices (`svd_clamped_newton_step`, `clamp_wage_displacement`, `solve_fold_continuation_step`); the library's own solvers do not call them. A bound keeps one bad linearization from throwing the iterate far away. Here the linear system itself is the difficulty: a random symmetric matrix of the size of Experiment 3's country roster, with one row and column scaled down by $10^{-5}$. The stiff coordinate is the country `C09`; the matrix is random and says nothing about that country. `solve_fold_continuation_step` eliminates the other 59 coordinates exactly, scales the whole step down to `max_disp`, and reports the residual of the step it returns.

# %%
n = len(codes)
k_stiff = codes.index("C09")
A_rnd = rng.standard_normal((n, n))
S = A_rnd.T @ A_rnd + np.eye(n)
S[k_stiff, :] *= 1e-5
S[:, k_stiff] *= 1e-5
S[k_stiff, k_stiff] = 1e-6
rhs = rng.standard_normal(n)


def rel_residual(step):
    return np.linalg.norm(S @ step - rhs) / np.linalg.norm(rhs)


dw_exact = la.solve(S, rhs)
dw_block, res_block, conv_block, info = solve_fold_continuation_step(
    S, rhs, idx_stiff="C09", country_codes=codes, max_disp=0.30, return_info=True)
steps = {
    "exact solve": dw_exact,
    "SVD modal clamp, then scale to 0.30": svd_clamped_newton_step(S, -rhs, np.ones(n), np.ones(n), max_comp=20.0, max_disp=0.30),
    "block elimination, then scale to 0.30": dw_block,
    "clip each coordinate to 0.30": clamp_wage_displacement(dw_exact, 0.30),
    "no step": np.zeros(n),
}
print(f"cond(S) = {np.linalg.cond(S):.2e}; stiff coordinate {info['idx_stiff']} ({info['country']}); "
      f"the exact solve has relative residual {rel_residual(dw_exact):.1e}")
for name, step in steps.items():
    print(f"  {name:38s} max|step| = {np.max(np.abs(step)):9.3g}   ||S step - rhs|| / ||rhs|| = {rel_residual(step):.4f}")
cosine = dw_block @ dw_exact / (np.linalg.norm(dw_block) * np.linalg.norm(dw_exact))
print(f"solve_fold_continuation_step reports residual {res_block:.4g} and converged {conv_block}; "
      f"its step is the exact solve times {info['clamp_scale']:.3g} (cosine {cosine:.6f}); "
      f"before the clamp the residual was {info['direction_residual']:.1e}")
print(f"Capped at 0.30 per step, Newton would need about {np.max(np.abs(dw_exact)) / 0.30:.2e} steps to cover the exact step")

assert rel_residual(dw_exact) < 1e-12 and info["idx_stiff"] == k_stiff and info["country"] == "C09"
assert abs(res_block - np.max(np.abs(S @ dw_block - rhs))) < 1e-12 and not conv_block, "reported = actual residual"
assert cosine > 1 - 1e-12 and info["direction_converged"]
assert 0.9 < rel_residual(steps["SVD modal clamp, then scale to 0.30"]) < 1.0
assert rel_residual(dw_block) > 0.999 and rel_residual(steps["clip each coordinate to 0.30"]) > 1.0

# %% [markdown]
# ## Read the output
#
# - **The toy fold (Figure 1).** Newton at a fixed $\lambda$, warm-started from the previous solution, converges at $\lambda=0.03$, $0.02$ and $0.01$ in 3 or 4 iterations. At the fold, $\lambda=0$, it needs 14: each step multiplies $x_1$ by exactly 0.5, so the state is still off by 6.1e-06 when the residual is 3.7e-11. At $\lambda=-0.01$ it fails because there is no real root. PAC from the same start turns the fold: $t_\lambda$ changes sign once, between nodes 3 and 4, where $x_1$ crosses zero (+0.033 to -0.024), and the 11 nodes past it lie on $x_1=-\sqrt\lambda$ to 2.5e-14. The right panel shows why this works. Along the branch $\kappa(F_x)$ grows like $1/|x_1|$, to 1.00e+06 at $|x_1|=10^{-6}$, while the bordered matrix stays between 1.41 and 1.85. The PAC nodes never come closer to the fold than $|x_1|=0.024$, so $\kappa(F_x)$ at the nodes peaks at only 41.9: a coarse path can step over a singularity without seeing it, which is why the test is the sign of $t_\lambda$ and not the size of $\kappa$.
# - **Where the equilibrium ends (Figure 2, left).** Under consistent accounting, warm-started Newton follows both paths to their end. With $\sigma=0$, A's factor price relative to B's falls from 1.000 at 0% to 0.076 at 400% and 0.00032 at 425%. Extrapolated linearly it reaches zero at a tariff of 4.2510, and PAC, asked for 440%, stalls at 4.2510 with no fold event and no raw detection. The branch of equilibria ends where A's factor price reaches zero: a corner, not a fold. With substitution between inputs ($\sigma=1$) the path ends much earlier, at 107.75%, with $w_A/w_B$ still at 0.187: the factor price is far from zero, but the smallest singular value of $F_x$ falls from 3.1e-03 to 2.3e-05. PAC, asked for 110%, records one event of kind `lambda_reversal`, first detected at an effective tariff of 1.0369, and does not converge: $\lambda$ falls, as at a turning point, and past it this branch has no equilibrium. Both ends make $F_x$ nearly singular ($\sigma_{\min}$ also falls to 2.9e-05 at $\sigma=0$), so the size of $\sigma_{\min}$ alone does not tell a corner from a fold; the factor price and the sign of $t_\lambda$ do.
# - **What the certificate can and cannot say (Figure 2, right, and Experiment 3).** On the 2x2 table the Hawkins-Simon certificate holds up to a tariff between 8.653800 and 8.653801; NumPy's eigenvalues put $\rho(B_\tau)=1$ at 8.653814, just above, because the certificate demands an upper bound below $1-10^{-6}$. At 425%, where the $\sigma=0$ branch ends, it certifies $\rho(B)$ = 0.6638, and it holds at the $\sigma=1$ turn as well. The condition is necessary for positive Leontief prices; it is far from sufficient for an equilibrium with positive factor prices. On the 720-row synthetic table a 25% tariff moves the bounds from 0.445882 to 0.465500, the verdict switches between tariffs of 6.93586 and 6.93587 (NumPy's eigenvalues: 6.935878), and 800% is a certified violation (both bounds 1.085275), not an unresolved interval. These numbers describe a random table, not any production network.
# - **The stiff step.** The exact solve has a relative residual of 4.1e-14 but moves the stiff coordinate by 7.91e+05. Each step capped at 0.30 leaves nearly all of the right-hand side unexplained: the SVD clamp leaves a relative residual of 0.9852, block elimination followed by uniform scaling leaves 1.0000 (the same as no step, since that step is the exact solve times 3.79e-07), and clipping each coordinate raises it to 13.4200. `solve_fold_continuation_step` reports this: residual 2.418 and converged False for the step it returns, while the unclamped direction had residual 2.7e-13. A cap bounds a step; it does not solve a stiff system, and at 0.30 per step Newton would need about 2.64e+06 steps. A cap pays off only when the model is nonlinear and a long step would leave the region where the linearization holds, which this linear test cannot show.

# %% [markdown]
# ## Your turn
#
# Change the elasticity of the tax base $\varepsilon$. Revenue is $R(t)=t(1-t)^{\varepsilon}$ for a tax rate $t\in[0,1]$, so the Laffer peak is at $t^*=1/(1+\varepsilon)$, a closed form derived by hand. The cell continues the revenue target $\bar R$ upward from $t=0.05$ with the notebook's `pac`, and checks three things: the single turn of the path brackets $t^*$, no node asks for more than the peak revenue, and Newton for a target 1% above the peak finds no tax rate in $[0,1]$.
#
# 1. **Basic.** Before running, predict $t^*$ for $\varepsilon=1$ and $\varepsilon=4$, then run both. Why is the revenue peak a fold of the equation $R(t)=\bar R$, and what does the sign of $t_\lambda$ record along this path?
# 2. **Intermediate.** In a new cell, run `newton_fixed(F_laffer, Fx_laffer, np.array([0.5 * t_star]), 0.99 * R_max)`. Newton now converges: on which side of the peak, given that it starts at $t^*/2$ where $R$ is concave? Which tax rate raises the same revenue on the other side, and which part of the PAC path reaches it?
# 3. **Stretch.** Back in Experiment 2, run `tariff_path(0.5, np.arange(0.0, 4.5, 0.01))` and then `solve_keller_pac` with `sigma=0.5` and a target just past the last tariff the path reaches. Is the end of the branch at $\sigma=0.5$ a corner, like $\sigma=0$ (a factor price heading to zero, no fold event), or a turn, like $\sigma=1$ (a `lambda_reversal` event with factor prices bounded away from zero)? Which numbers in the output decide it?

# %%
eps_base = 2.0   # ← change this: elasticity of the tax base, 0.5 to 4
assert 0.5 <= eps_base <= 4.0, "stay inside the advertised range 0.5 to 4"


def revenue(t):
    """Laffer revenue R(t) = t (1 - t)^eps; tax rates live in [0, 1] and are undefined outside."""
    return t * (1.0 - t) ** eps_base if 0.0 <= t <= 1.0 else np.nan


def F_laffer(x, R):
    return np.array([revenue(x[0]) - R])


def Fx_laffer(x, R):
    t = x[0]
    slope = (1.0 - t) ** eps_base - eps_base * t * (1.0 - t) ** (eps_base - 1.0) if 0.0 <= t < 1.0 else np.nan
    return np.array([[slope]])


def Flam_laffer(x, R):
    return np.array([-1.0])


t_star = 1.0 / (1.0 + eps_base)          # closed form: R'(t) = 0
R_max = revenue(t_star)
laffer = pac(F_laffer, Fx_laffer, Flam_laffer, np.array([0.05]), revenue(0.05), ds=0.05, n_steps=60,
             lam_direction=+1.0, until=lambda z: z[0] > 0.9)
t_path, R_path = laffer["z"][:, 0], laffer["z"][:, 1]
turn = np.flatnonzero(np.diff(np.sign(laffer["t_lam"])))
assert len(turn) == 1, f"the path should turn exactly once, not {len(turn)} times"
_, above_ok, _ = newton_fixed(F_laffer, Fx_laffer, np.array([0.5 * t_star]), 1.01 * R_max)
print(f"Peak: t* = {t_star:.4f}, R_max = {R_max:.6f}; PAC turned between t = {t_path[turn[0]]:.3f} and {t_path[turn[0] + 1]:.3f}")
print(f"Largest revenue target on the path {R_path.max():.6f}; path ends at t = {t_path[-1]:.3f}, R = {R_path[-1]:.4f}")
print(f"Newton for a target 1% above the peak converged: {above_ok}")

assert t_path[turn[0]] < t_star < t_path[turn[0] + 1], "the turn must bracket the revenue peak"
assert R_path.max() <= R_max + 1e-12 and laffer["max_F"].max() < 1e-10
assert not above_ok, "no tax rate raises more than the peak revenue"

# %% [markdown]
# ## How comprehensive is this?
#
# The trade solvers used here also serve notebook 63 (`63_trade_wars_and_nash_tariffs`), where `solve_policy_equilibrium(method="auto")` ends its recovery ladder with Keller continuation (`docs/trade_policy.md`); `docs/trade_accounting.md` lists the solvers that share the same equations; the `solve_keller_pac` docstring documents its fold monitor and the event kinds. The "Keller PAC" and "MRIO productivity" rows of `docs/STRUCTURAL_VALIDATION_STATUS.md` state what is validated. This notebook shows PAC detecting a turn of a two-country equilibrium path, not traversing it to a converged solution on the other side, and its fold is a property of one synthetic table. `docs/ADVISORY.md` records the corrected fold monitor and step residuals (2026-09-30). For empirical input-output work, use the bundled OECD table (`load_icio_data`) or read the OECD tables with `load_oecd_icio_granular` or `puremacro.trade.mrio.read_oecd_native`.
