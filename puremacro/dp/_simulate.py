"""Panel simulation for puremacro.dp solutions (private; see ``DPSolution.simulate``).

Agents are advanced with each solution's own policy representation:

- VFI: an integer index into the (flattened) state grid, ``policy_aprime[a, z]``;
  several states are stepped on the flat index and unravelled over the grid
  shape only when values are recorded.
- EGM: a continuous asset value, the next-asset policy ``aprime[:, z]``
  interpolated linearly between grid nodes and clipped to the grid ends.

Exogenous shocks move by the solution's Markov matrix ``sol.P``, one uniform
per agent and period drawn by inverse CDF. One ``numpy.random.Generator``
(PCG64) draws everything, so the same seed gives bit-identical panels: the
initial cells first, then in each period the discrete-choice uniforms (taste
shocks only), the shock uniforms and, in a life cycle with declared survival,
the survival uniforms.

A discrete choice is the solver's argmax option, ``policy_d`` at the agent's
cell (VFI) or at the bracketing node nearer its asset (EGM), and the next
asset is that option's policy. With taste shocks the option is drawn from the
logit probabilities ``choice_prob`` interpolated at the agent's asset, and the
next asset is the drawn option's ``aprime_by_choice``. The choice is recorded
as the option value.

A life cycle steps age j with ``policy[j]`` and runs for the T ages of the
horizon. Newborns are drawn from the declared ``newborns`` (normalised) or by
the solver's rule: the lowest asset node, the shock from the stationary
distribution of ``P``. An agent alive at age j survives to j+1 with
probability ``survival[j]``; the dead keep stepping, so the arrays stay finite,
and are marked ``alive=False`` so that statistics ignore them.

Recorded columns are model expressions compiled once with the household
function signature and evaluated on (n_agents, periods) arrays: ``n_<state>``
is the state at t+1, ``s_<state>`` the state at t, ``x_<shock>`` the shock at t.
"""
from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from puremacro.dp._expr import ModelSpecError, parse
from puremacro.dp._panel import _RESERVED, Panel
from puremacro.vfi.discretize import markov_stationary

MAX_CELLS = 5e8                  # n_agents * periods above this is refused

HJB_REFUSAL = "continuous-time solutions are not simulated; sol.distribution holds the KFE density"
KS_REFUSAL = "aggregate-risk solutions are not simulated; sol.aggregates holds the capital path"


# ----------------------------------------------------------------------------- steppers
def _draw_markov(rng: np.random.Generator, P: np.ndarray, z_idx: np.ndarray) -> np.ndarray:
    """Next shock index of every agent, by inverse CDF on the row ``P[z_idx]``."""
    cdf = np.cumsum(P, axis=1)
    cdf[:, -1] = 1.0
    u = rng.random(z_idx.size)
    nxt = (u[:, None] >= cdf[z_idx]).sum(axis=1)
    return np.minimum(nxt, P.shape[1] - 1)


def _step_vfi(policy_aprime: np.ndarray, a_idx: np.ndarray, z_idx: np.ndarray) -> np.ndarray:
    """Next flat state index under an index policy."""
    return policy_aprime[a_idx, z_idx]


def _interp_by_shock(table: np.ndarray, a_grid: np.ndarray, a_val: np.ndarray,
                     z_idx: np.ndarray) -> np.ndarray:
    """``table[:, z]`` interpolated linearly at each agent's asset, agents grouped
    by their shock."""
    out = np.empty_like(a_val, dtype=float)
    for z in np.unique(z_idx):
        sel = z_idx == z
        out[sel] = np.interp(a_val[sel], a_grid, table[:, z])
    return out


def _step_egm(aprime: np.ndarray, a_grid: np.ndarray, a_val: np.ndarray,
              z_idx: np.ndarray) -> np.ndarray:
    """Next asset value: ``aprime[:, z]`` interpolated linearly in the asset,
    clipped to ``[a_grid[0], a_grid[-1]]``, agents grouped by their shock."""
    return np.clip(_interp_by_shock(aprime, a_grid, a_val, z_idx), a_grid[0], a_grid[-1])


def _draw_choice(rng: np.random.Generator, choice_prob: np.ndarray, a_grid: np.ndarray,
                 a_val: np.ndarray, z_idx: np.ndarray) -> np.ndarray:
    """Option index of every agent under taste shocks: the probabilities
    ``choice_prob[k, :, z]`` interpolated linearly at the agent's asset and
    renormalised, then one uniform per agent by inverse CDF."""
    n_d = choice_prob.shape[0]
    p = np.stack([_interp_by_shock(choice_prob[k], a_grid, a_val, z_idx) for k in range(n_d)],
                 axis=1)
    p = np.clip(p, 0.0, None)
    p /= p.sum(axis=1, keepdims=True)
    cdf = np.cumsum(p, axis=1)
    cdf[:, -1] = 1.0
    u = rng.random(a_val.size)
    return np.minimum((u[:, None] >= cdf).sum(axis=1), n_d - 1)


def _step_by_choice(aprime_by_choice: np.ndarray, a_grid: np.ndarray, a_val: np.ndarray,
                    z_idx: np.ndarray, d_idx: np.ndarray) -> np.ndarray:
    """Next asset value under each agent's chosen option ``aprime_by_choice[d]``."""
    out = np.empty_like(a_val, dtype=float)
    for k in np.unique(d_idx):
        sel = d_idx == k
        out[sel] = _step_egm(aprime_by_choice[k], a_grid, a_val[sel], z_idx[sel])
    return out


def _draw_cells(rng: np.random.Generator, mass: np.ndarray, n: int) -> np.ndarray:
    """``n`` draws from the cells of a (flattened) mass, by inverse CDF."""
    cdf = np.cumsum(mass)
    idx = np.searchsorted(cdf, rng.random(n) * cdf[-1], side="right")
    return np.minimum(idx, mass.size - 1)


# ----------------------------------------------------------------------------- set-up
def _is_egm(sol) -> bool:
    return hasattr(sol.raw, "aprime") or hasattr(sol.raw, "c_policy")


def _expm_transition(A: np.ndarray, dt: float) -> np.ndarray:
    """Discrete transition matrix exp(A * dt) for continuous-time generator A."""
    from scipy.linalg import expm

    P = np.clip(expm(np.asarray(A, dtype=float) * float(dt)), 0.0, None)
    row_sums = P.sum(axis=1, keepdims=True)
    return np.where(row_sums > 0, P / row_sums, 1.0 / P.shape[1])


def _state_shape(sol) -> tuple[int, ...]:
    return tuple(g.size for g in sol._c.state_grids)


def _shock_shape(sol) -> tuple[int, ...]:
    return tuple(g.size for g in sol._c.z_grids) or (1,)


def _nearest(grid: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Index into ``grid`` of the node nearest each value. The grid need not be
    sorted: a Markov shock grid may list its values in any order."""
    grid = np.asarray(grid, dtype=float)
    v = np.asarray(v, dtype=float)
    if grid.size == 1:
        return np.zeros(v.shape, dtype=np.intp)
    order = np.argsort(grid, kind="stable")
    s = grid[order]
    j = np.clip(np.searchsorted(s, v), 1, s.size - 1)
    return order[np.where(np.abs(v - s[j - 1]) <= np.abs(s[j] - v), j - 1, j)]


def _newborn_mass(sol) -> np.ndarray:
    """Age-0 mass over (state, shock) cells in a life cycle: the declared
    ``newborns`` normalised, or the solver's rule (lowest asset node, shock
    from the stationary distribution of ``P``)."""
    from puremacro.vfi.finite_horizon import _z_stationary

    n_a = int(np.prod(_state_shape(sol)))
    n_z = int(np.prod(_shock_shape(sol)))
    nb = sol._c.model._newborns
    if nb is None:
        mu0 = np.zeros((n_a, n_z))
        mu0[0] = _z_stationary(np.asarray(sol.P, dtype=float))
        return mu0
    mu0 = np.asarray(nb, dtype=float)
    if mu0.shape != (n_a, n_z):
        raise ValueError(f"newborns must have shape ({n_a}, {n_z}); got {mu0.shape}")
    return mu0 / mu0.sum()


def _initial_state(sol, rng: np.random.Generator, n_agents: int, init, *, P_ergodic=None):
    """Initial state (flat index for VFI, value for EGM/HJB) and shock index per agent."""
    c = sol._c
    egm = _is_egm(sol)
    n_z = int(np.prod(_shock_shape(sol)))
    if isinstance(init, str):
        allowed = "newborn" if c.finite else "stationary"
        if init != allowed:
            kind = "a life cycle" if c.finite else "an infinite horizon"
            raise ValueError(f"init must be {allowed!r} or a dict {{state: value}} in {kind}, got {init!r}")
        if c.finite:
            cells = _draw_cells(rng, _newborn_mass(sol).ravel(), n_agents)
            a_idx, z_idx = np.divmod(cells, n_z)
            if egm:
                return c.state_grids[0][a_idx].astype(float), z_idx
            return a_idx, z_idx
        if sol.distribution is None:
            raise ValueError("init='stationary' needs sol.distribution, which is None here; "
                             "solve with distribution=True or pass init={state: value, ...}")
        cells = _draw_cells(rng, np.asarray(sol.distribution, dtype=float).ravel(), n_agents)
        a_idx, z_idx = np.divmod(cells, n_z)
        if egm:
            return c.state_grids[0][a_idx].astype(float), z_idx
        return a_idx, z_idx
    if not isinstance(init, Mapping):
        raise ValueError(f"init must be a string or a dict {{state: value}}, got {type(init).__name__}")

    unknown = [k for k in init if k not in c.state_names and k not in c.shock_names]
    if unknown:
        raise ValueError(f"init names {unknown} are neither states {c.state_names} "
                         f"nor shocks {c.shock_names}")
    missing = [s for s in c.state_names if s not in init]
    if missing:
        raise ValueError(f"init must give every state; missing {missing}")

    def values(name):
        v = np.asarray(init[name], dtype=float)
        if v.ndim == 0:
            v = np.full(n_agents, float(v))
        if v.shape != (n_agents,):
            raise ValueError(f"init[{name!r}] must be a scalar or have shape ({n_agents},), got {v.shape}")
        if not np.all(np.isfinite(v)):
            raise ValueError(f"init[{name!r}] must be finite")
        return v

    if egm:
        g = c.state_grids[0]
        v = values(c.state_names[0])
        state = np.clip(v, g[0], g[-1])
        if np.any(state != v):
            warnings.warn(f"init: values of {c.state_names[0]!r} outside the grid "
                          f"[{g[0]:g}, {g[-1]:g}] were clipped to it", UserWarning, stacklevel=4)
    else:
        idx, off = [], []
        for name, g in zip(c.state_names, c.state_grids):
            v = values(name)
            j = _nearest(g, v)
            if not np.allclose(g[j], v, rtol=1e-12, atol=1e-12):
                off.append(name)
            idx.append(j)
        if off:
            warnings.warn(f"init: values of {off} are off the grid; VFI states live on grid "
                          "nodes, so they were snapped to the nearest node", UserWarning, stacklevel=4)
        state = np.ravel_multi_index(tuple(idx), _state_shape(sol))

    # shocks: given ones must lie on their grid; the rest are drawn from the ergodic chain
    P_stat = P_ergodic if P_ergodic is not None else c.P
    z_idx = _draw_cells(rng, markov_stationary(P_stat), n_agents)
    given = [s for s in c.shock_names if s in init]
    if given:
        comps = list(np.unravel_index(z_idx, _shock_shape(sol)))
        for k, (name, g) in enumerate(zip(c.shock_names, c.z_grids)):
            if name in init:
                v = values(name)
                j = _nearest(g, v)
                if not np.allclose(g[j], v, rtol=1e-9, atol=1e-12):
                    raise ValueError(f"init[{name!r}] must take values on the shock grid {g.tolist()}")
                comps[k] = j
        z_idx = np.ravel_multi_index(tuple(comps), _shock_shape(sol))
    return state, z_idx


def _default_record(sol) -> list[str]:
    """States, shocks, then the controls: the ``c`` local, any ``control()``
    and the discrete choice."""
    c = sol._c
    m = c.model
    out = list(c.state_names) + list(c.shock_names)
    if "c" in c.local_names:
        out.append("c")
    out += list(m._controls)
    if m._discrete is not None:
        out.append(m._discrete[0])
    return list(dict.fromkeys(out))


def _compile_record(sol, record) -> list[tuple[str, Any]]:
    """Validate and compile the recorded expressions; nothing is drawn yet."""
    c = sol._c
    if record is None:
        record = _default_record(sol)
    elif isinstance(record, str):
        record = [record]
    record = list(record)
    if not record:
        raise ValueError("record must name at least one expression")
    bad = [r for r in record if not isinstance(r, str)]
    if bad:
        raise ValueError(f"record takes model expressions as strings (no callables); got {bad}")
    dup = sorted({r for r in record if record.count(r) > 1})
    if dup:
        raise ValueError(f"record lists {dup} more than once")
    clash = [r for r in record if r in _RESERVED]
    if clash:
        raise ValueError(f"record names {clash} are reserved panel columns ({list(_RESERVED)})")
    timed = c.state_names + c.shock_names + ([c.model._discrete[0]] if c.model._discrete else [])
    out = []
    for expr in record:
        try:
            fn = c._household_function("_dp_record", parse(expr, timed, functions=c.model._functions))
        except ModelSpecError as exc:
            raise ModelSpecError(f"record {expr!r}: {exc}") from None
        out.append((expr, fn))
    return out


def _stepper(sol, rng: np.random.Generator):
    """``step(state, z_idx, j) -> (next_state, d_idx)``: one period under the
    solution's policy at age ``j`` (ignored in an infinite horizon). ``d_idx``
    is the option index of every agent, or None without a discrete choice.

    - VFI: the next flat state index is ``policy_aprime[a, z]`` (the chosen
      option's policy); the option is ``policy_d[a, z]``.
    - EGM without taste shocks: the option is ``policy_d`` at the bracketing
      node nearer the agent's asset; the next asset interpolates ``aprime``.
    - EGM with taste shocks: the option is drawn from ``choice_prob``
      interpolated at the agent's asset (one uniform per agent, drawn here,
      before the period's shock uniforms); the next asset interpolates that
      option's ``aprime_by_choice[k]``.
    """
    c, raw = sol._c, sol.raw
    finite = c.finite

    def at(arr, j):
        return arr[j] if finite else arr

    pol_d = getattr(raw, "policy_d", None)
    pol_d = None if pol_d is None or c.model._discrete is None else np.asarray(pol_d)
    if not _is_egm(sol):
        pol = np.asarray(raw.policy_aprime)
        if pol_d is None:
            return lambda s, zz, j: (_step_vfi(at(pol, j), s, zz), None)
        return lambda s, zz, j: (_step_vfi(at(pol, j), s, zz), at(pol_d, j)[s, zz])

    a_grid = c.state_grids[0]
    aprime = np.asarray(raw.aprime, dtype=float)
    prob = getattr(raw, "choice_prob", None)
    if pol_d is None:
        return lambda s, zz, j: (_step_egm(at(aprime, j), a_grid, s, zz), None)
    if prob is None:
        def step_dc(s, zz, j):
            return _step_egm(at(aprime, j), a_grid, s, zz), at(pol_d, j)[_nearest(a_grid, s), zz]
        return step_dc
    prob = np.asarray(prob, dtype=float)
    by_choice = np.asarray(raw.aprime_by_choice, dtype=float)

    def step_taste(s, zz, j):
        d = _draw_choice(rng, at(prob, j), a_grid, s, zz)
        return _step_by_choice(at(by_choice, j), a_grid, s, zz, d), d
    return step_taste


def _simulate_hjb(sol, n_agents: int, periods: int, *, rng: np.random.Generator,
                  init, burn_in: int, compiled) -> Panel:
    """Continuous-time Euler-Maruyama / Poisson-jump simulation for HJB solutions."""
    c = sol._c
    a_grid = c.state_grids[0]
    P_1 = _expm_transition(sol.P, 1.0)
    n_sub = 10
    dt = 1.0 / n_sub
    P_sub = _expm_transition(sol.P, dt)
    state, z = _initial_state(sol, rng, n_agents, init, P_ergodic=P_1)
    s_drift = np.asarray(sol.raw.s_drift, dtype=float)
    c_policy = np.asarray(sol.raw.c_policy, dtype=float)

    S = np.empty((n_agents, periods), dtype=float)
    Z = np.empty((n_agents, periods), dtype=np.intp)
    total = burn_in + periods
    for t in range(total):
        k = t - burn_in
        if k >= 0:
            S[:, k], Z[:, k] = state, z
        for s_idx in range(n_sub):
            drift = _interp_by_shock(s_drift, a_grid, state, z)
            state = np.clip(state + drift * dt, a_grid[0], a_grid[-1])
            if P_sub.shape[0] > 1 and not (t == total - 1 and s_idx == n_sub - 1):
                z = _draw_markov(rng, P_sub, z)

    c_val = _interp_by_shock(c_policy, a_grid, S.ravel(), Z.ravel()).reshape(n_agents, periods)
    if c.z_grids:
        zs = [g[i] for g, i in zip(c.z_grids, np.unravel_index(Z, _shock_shape(sol)))]
    else:
        zs = [np.zeros((n_agents, periods))]
    args = [c_val, S] + zs + list(sol.params.values())
    arrays = {}
    for name, fn in compiled:
        val = np.asarray(fn(*args, xp=np), dtype=float)
        arrays[name] = np.array(np.broadcast_to(val, (n_agents, periods)))
    return Panel(arrays, age=None, alive=None,
                 source=f"puremacro.dp simulation of {sol.model_name!r} ({sol.method})",
                 functions=c.model._functions)


def _simulate_ks(sol, n_agents: int, periods: int, *, rng: np.random.Generator,
                 init, burn_in: int, compiled) -> Panel:
    """Panel simulation for Krusell-Smith solutions (vfi+ks and egm+ks)."""
    c = sol._c
    egm = _is_egm(sol)
    a_grid = c.state_grids[0]
    n_a = a_grid.size
    P_idio = np.asarray(c.P_idio, dtype=float)
    P_Z = np.asarray(c.P_agg, dtype=float)
    K_grid = np.asarray(c.z_grids[-1], dtype=float)
    n_idio, n_Z, n_K = P_idio.shape[0], P_Z.shape[0], K_grid.size
    n_idio_shocks = len(c.model._shocks)
    n_agg_shocks = len(c.model._agg_shocks)

    if isinstance(init, str):
        if init != "stationary":
            raise ValueError(f"init must be 'stationary' or a dict {{state: value}} in an infinite horizon, got {init!r}")
        mu0 = getattr(sol, "_ks_mu0", None)
        if mu0 is None:
            mu0 = np.full((n_a, n_idio), 1.0 / (n_a * n_idio))
        cells = _draw_cells(rng, np.asarray(mu0, dtype=float).ravel(), n_agents)
        a_idx, z_idio = np.divmod(cells, n_idio)
        state = a_grid[a_idx].astype(float) if egm else a_idx
        iZ = int(_draw_cells(rng, markov_stationary(P_Z), 1)[0])
        K_init = None
    elif isinstance(init, Mapping):
        unknown = [k for k in init if k not in c.state_names and k not in c.shock_names]
        if unknown:
            raise ValueError(f"init names {unknown} are neither states {c.state_names} nor shocks {c.shock_names}")
        missing = [s for s in c.state_names if s not in init]
        if missing:
            raise ValueError(f"init must give every state; missing {missing}")

        def values(name):
            v = np.asarray(init[name], dtype=float)
            if v.ndim == 0:
                v = np.full(n_agents, float(v))
            if v.shape != (n_agents,):
                raise ValueError(f"init[{name!r}] must be a scalar or have shape ({n_agents},), got {v.shape}")
            if not np.all(np.isfinite(v)):
                raise ValueError(f"init[{name!r}] must be finite")
            return v

        v_a = values(c.state_names[0])
        if egm:
            state = np.clip(v_a, a_grid[0], a_grid[-1])
            if np.any(state != v_a):
                warnings.warn(f"init: values of {c.state_names[0]!r} outside the grid "
                              f"[{a_grid[0]:g}, {a_grid[-1]:g}] were clipped to it", UserWarning, stacklevel=4)
        else:
            j_a = _nearest(a_grid, v_a)
            if not np.allclose(a_grid[j_a], v_a, rtol=1e-12, atol=1e-12):
                warnings.warn(f"init: values of {[c.state_names[0]]} are off the grid; VFI states live on grid "
                              "nodes, so they were snapped to the nearest node", UserWarning, stacklevel=4)
            state = j_a

        z_idio = _draw_cells(rng, markov_stationary(P_idio), n_agents)
        idio_shape = tuple(g.size for g in c.z_grids[:n_idio_shocks]) or (1,)
        if any(s in init for s in c.shock_names[:n_idio_shocks]):
            comps = list(np.unravel_index(z_idio, idio_shape))
            for k, (name, g) in enumerate(zip(c.shock_names[:n_idio_shocks], c.z_grids[:n_idio_shocks])):
                if name in init:
                    v = values(name)
                    j_arr = _nearest(g, v)
                    if not np.allclose(g[j_arr], v, rtol=1e-9, atol=1e-12):
                        raise ValueError(f"init[{name!r}] must take values on the shock grid {g.tolist()}")
                    comps[k] = j_arr
            z_idio = np.ravel_multi_index(tuple(comps), idio_shape)

        iZ = int(_draw_cells(rng, markov_stationary(P_Z), 1)[0])
        agg_shape = tuple(g.size for g in c.z_grids[n_idio_shocks:-1]) or (1,)
        if any(s in init for s in c.shock_names[n_idio_shocks:-1]):
            comps_z: list[Any] = list(np.unravel_index(iZ, agg_shape))
            for k, (name, g) in enumerate(zip(c.shock_names[n_idio_shocks:-1], c.z_grids[n_idio_shocks:-1])):
                if name in init:
                    v = float(np.asarray(init[name], dtype=float).ravel()[0])
                    j = int(_nearest(g, np.array([v]))[0])
                    if not np.isclose(g[j], v, rtol=1e-9, atol=1e-12):
                        raise ValueError(f"init[{name!r}] must take values on the shock grid {g.tolist()}")
                    comps_z[k] = j
            iZ = int(np.ravel_multi_index(tuple(comps_z), agg_shape))

        kname = c.shock_names[-1]
        K_init = float(np.mean(np.asarray(init[kname], dtype=float))) if kname in init else None
    else:
        raise ValueError(f"init must be a string or a dict {{state: value}}, got {type(init).__name__}")

    if egm:
        pol4 = np.asarray(sol.raw.aprime, dtype=float).reshape(n_a, n_idio, n_Z, n_K)
    else:
        pol4_val = a_grid[np.asarray(sol.raw.policy_aprime).reshape(n_a, n_idio, n_Z, n_K)]

    S = np.empty((n_agents, periods + 1), dtype=state.dtype)
    Z_idio = np.empty((n_agents, periods), dtype=np.intp)
    Z_agg = np.empty(periods, dtype=np.intp)
    K_rec = np.empty(periods, dtype=float)
    total = burn_in + periods
    for t in range(total):
        k = t - burn_in
        a_vals = state if egm else a_grid[state]
        K_t = float(K_init) if (t == 0 and K_init is not None) else float(np.mean(a_vals))
        Kc = float(np.clip(K_t, K_grid[0], K_grid[-1]))
        j_k = int(np.clip(np.searchsorted(K_grid, Kc) - 1, 0, n_K - 2))
        w = (K_grid[j_k + 1] - Kc) / (K_grid[j_k + 1] - K_grid[j_k])
        if egm:
            ap_table = w * pol4[:, :, iZ, j_k] + (1.0 - w) * pol4[:, :, iZ, j_k + 1]
            nxt = _step_egm(ap_table, a_grid, state, z_idio)
        else:
            ap_val = w * pol4_val[state, z_idio, iZ, j_k] + (1.0 - w) * pol4_val[state, z_idio, iZ, j_k + 1]
            nxt = _nearest(a_grid, ap_val)
        if k >= 0:
            S[:, k], Z_idio[:, k], Z_agg[k], K_rec[k] = state, z_idio, iZ, K_t
        if t < total - 1:
            if n_idio > 1:
                z_idio = _draw_markov(rng, P_idio, z_idio)
            if n_Z > 1:
                iZ = int(_draw_markov(rng, P_Z, np.array([iZ]))[0])
        state = nxt
    S[:, periods] = state

    cur = [S[:, :-1]] if egm else [a_grid[S[:, :-1]]]
    nxt_arrs = [S[:, 1:]] if egm else [a_grid[S[:, 1:]]]
    zs = []
    if n_idio_shocks > 0:
        idio_shape = tuple(g.size for g in c.z_grids[:n_idio_shocks])
        zs += [g[i] for g, i in zip(c.z_grids[:n_idio_shocks], np.unravel_index(Z_idio, idio_shape))]
    if n_agg_shocks > 0:
        agg_shape = tuple(g.size for g in c.z_grids[n_idio_shocks:-1])
        zs += [np.broadcast_to(g[i][None, :], (n_agents, periods))
               for g, i in zip(c.z_grids[n_idio_shocks:-1], np.unravel_index(Z_agg, agg_shape))]
    zs.append(np.broadcast_to(K_rec[None, :], (n_agents, periods)))
    args = nxt_arrs + cur + zs + list(sol.params.values())
    arrays = {}
    for name, fn in compiled:
        val = np.asarray(fn(*args, xp=np), dtype=float)
        arrays[name] = np.array(np.broadcast_to(val, (n_agents, periods)))
    return Panel(arrays, age=None, alive=None,
                 source=f"puremacro.dp simulation of {sol.model_name!r} ({sol.method})",
                 functions=c.model._functions)


# ----------------------------------------------------------------------------- driver
def simulate(sol, n_agents: int, periods: int | None, *, seed, init, burn_in: int,
             record: Sequence[str] | None) -> Panel:
    """Body of ``DPSolution.simulate``; see its docstring."""
    c = sol._c
    J = int(sol.raw.horizon) if c.finite else 0
    periods_int: int = (J if c.finite else 200) if periods is None else int(periods)
    for name, v, lo in (("n_agents", n_agents, 1), ("periods", periods_int, 1), ("burn_in", burn_in, 0)):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < lo:
            raise ValueError(f"{name} must be an integer >= {lo}, got {v!r}")
    n_agents, periods_int, burn_in = int(n_agents), int(periods_int), int(burn_in)
    if init is None:
        init = "newborn" if c.finite else "stationary"
    if c.finite:
        if burn_in:
            raise ValueError("burn_in applies to infinite-horizon solutions only; "
                             "a life cycle starts with newborns")
        if periods_int != J:
            warnings.warn(f"periods={periods_int} is ignored in a life cycle: the panel runs over "
                          f"the {J} ages of the horizon", UserWarning, stacklevel=3)
            periods_int = J

    if sol.method == "egm" and len(c.state_names) == 2:
        raise NotImplementedError("two-asset EGM panel simulation is not implemented; "
                                  "use policy(), distribution, or dp_nonlinear_transition()")
    compiled = _compile_record(sol, record)
    if n_agents * periods_int > MAX_CELLS:
        nbytes = 8 * n_agents * periods_int * len(compiled)
        raise ValueError(f"n_agents * periods = {n_agents * periods_int:.3g} exceeds {MAX_CELLS:.0e}: "
                         f"{len(compiled)} recorded arrays would need {nbytes:,} bytes "
                         f"({nbytes / 1e9:.1f} GB); simulate fewer agents or periods")

    rng = np.random.Generator(np.random.PCG64(seed))
    if sol.method == "hjb":
        return _simulate_hjb(sol, n_agents, periods_int, rng=rng, init=init, burn_in=burn_in, compiled=compiled)
    if sol.method.endswith("+ks"):
        return _simulate_ks(sol, n_agents, periods_int, rng=rng, init=init, burn_in=burn_in, compiled=compiled)

    state, z = _initial_state(sol, rng, n_agents, init)
    P = np.asarray(sol.P, dtype=float)
    egm = _is_egm(sol)
    step = _stepper(sol, rng)
    survival = c.model._survival if c.finite else None
    alive = None
    if survival is not None:
        survival = np.asarray(survival, dtype=float)
        alive = np.ones((n_agents, periods_int), dtype=bool)

    S = np.empty((n_agents, periods_int + 1), dtype=state.dtype)     # state at t, plus t = periods
    Z = np.empty((n_agents, periods_int), dtype=np.intp)
    D = np.empty((n_agents, periods_int), dtype=np.intp) if c.model._discrete is not None else None
    total = burn_in + periods_int
    for t in range(total):
        k = t - burn_in                                          # period (age in a life cycle)
        nxt, d = step(state, z, k)
        if k >= 0:
            S[:, k], Z[:, k] = state, z
            if D is not None:
                D[:, k] = d
        if t < total - 1:
            z = _draw_markov(rng, P, z)
            if alive is not None and survival is not None:
                alive[:, k + 1] = alive[:, k] & (rng.random(n_agents) < survival[k])
        state = nxt
    S[:, periods_int] = state

    # values for the household-function arguments, each (n_agents, periods)
    if egm:
        cur, nxt = [S[:, :-1]], [S[:, 1:]]
    else:
        shape = _state_shape(sol)
        cur = [g[i] for g, i in zip(c.state_grids, np.unravel_index(np.asarray(S[:, :-1], dtype=np.intp), shape))]
        nxt = [g[i] for g, i in zip(c.state_grids, np.unravel_index(np.asarray(S[:, 1:], dtype=np.intp), shape))]
    if c.z_grids:
        zs = [g[i] for g, i in zip(c.z_grids, np.unravel_index(Z, _shock_shape(sol)))]
    else:
        zs = [np.zeros((n_agents, periods_int))]
    age = np.arange(periods_int) if c.finite else None
    ages = [np.broadcast_to(age, (n_agents, periods_int))] if (c.finite and age is not None) else []
    ds = [np.asarray(c.model._discrete[1], dtype=float)[D]] if D is not None else []
    args = ds + nxt + cur + zs + ages + list(sol.params.values())
    arrays = {}
    for name, fn in compiled:
        val = np.asarray(fn(*args, xp=np), dtype=float)
        arrays[name] = np.array(np.broadcast_to(val, (n_agents, periods_int)))
    return Panel(arrays, age=age, alive=alive,
                 source=f"puremacro.dp simulation of {sol.model_name!r} ({sol.method})",
                 functions=c.model._functions)
