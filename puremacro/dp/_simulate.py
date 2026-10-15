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
initial cells first, then in each period the discrete-choice uniforms (none
yet), the shock uniforms and, in a life cycle with declared survival, the
survival uniforms.

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


def _step_egm(aprime: np.ndarray, a_grid: np.ndarray, a_val: np.ndarray,
              z_idx: np.ndarray) -> np.ndarray:
    """Next asset value: ``aprime[:, z]`` interpolated linearly in the asset,
    clipped to ``[a_grid[0], a_grid[-1]]``, agents grouped by their shock."""
    out = np.empty_like(a_val)
    for z in np.unique(z_idx):
        sel = z_idx == z
        out[sel] = np.interp(a_val[sel], a_grid, aprime[:, z])
    return np.clip(out, a_grid[0], a_grid[-1])


def _draw_cells(rng: np.random.Generator, mass: np.ndarray, n: int) -> np.ndarray:
    """``n`` draws from the cells of a (flattened) mass, by inverse CDF."""
    cdf = np.cumsum(mass)
    idx = np.searchsorted(cdf, rng.random(n) * cdf[-1], side="right")
    return np.minimum(idx, mass.size - 1)


# ----------------------------------------------------------------------------- set-up
def _is_egm(sol) -> bool:
    return hasattr(sol.raw, "aprime")


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


def _initial_state(sol, rng: np.random.Generator, n_agents: int, init):
    """Initial state (flat index for VFI, value for EGM) and shock index per agent."""
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
    z_idx = _draw_cells(rng, markov_stationary(c.P), n_agents)
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


def _compile_record(sol, record) -> list[tuple[str, object]]:
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
            fn = c._household_function("_dp_record", parse(expr, timed))
        except ModelSpecError as exc:
            raise ModelSpecError(f"record {expr!r}: {exc}") from None
        out.append((expr, fn))
    return out


# ----------------------------------------------------------------------------- driver
def simulate(sol, n_agents: int, periods: int | None, *, seed, init, burn_in: int,
             record: Sequence[str] | None) -> Panel:
    """Body of ``DPSolution.simulate``; see its docstring."""
    c = sol._c
    if hasattr(c, "policy_values") or sol.method == "hjb":
        raise NotImplementedError(HJB_REFUSAL)
    if sol.method.endswith("+ks"):
        raise NotImplementedError(KS_REFUSAL)
    if c.model._discrete is not None:
        raise NotImplementedError("simulating models with a discrete choice is not supported yet")
    J = int(sol.raw.horizon) if c.finite else None
    if periods is None:
        periods = J if c.finite else 200
    for name, v, lo in (("n_agents", n_agents, 1), ("periods", periods, 1), ("burn_in", burn_in, 0)):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < lo:
            raise ValueError(f"{name} must be an integer >= {lo}, got {v!r}")
    n_agents, periods, burn_in = int(n_agents), int(periods), int(burn_in)
    if init is None:
        init = "newborn" if c.finite else "stationary"
    if c.finite:
        if burn_in:
            raise ValueError("burn_in applies to infinite-horizon solutions only; "
                             "a life cycle starts with newborns")
        if periods != J:
            warnings.warn(f"periods={periods} is ignored in a life cycle: the panel runs over "
                          f"the {J} ages of the horizon", UserWarning, stacklevel=3)
            periods = J

    compiled = _compile_record(sol, record)
    if n_agents * periods > MAX_CELLS:
        nbytes = 8 * n_agents * periods * len(compiled)
        raise ValueError(f"n_agents * periods = {n_agents * periods:.3g} exceeds {MAX_CELLS:.0e}: "
                         f"{len(compiled)} recorded arrays would need {nbytes:,} bytes "
                         f"({nbytes / 1e9:.1f} GB); simulate fewer agents or periods")

    rng = np.random.Generator(np.random.PCG64(seed))
    state, z = _initial_state(sol, rng, n_agents, init)
    P = np.asarray(sol.P, dtype=float)
    egm = _is_egm(sol)
    if egm:
        aprime, a_grid = np.asarray(sol.raw.aprime, dtype=float), c.state_grids[0]
        if c.finite:
            step = lambda s, zz, j: _step_egm(aprime[j], a_grid, s, zz)    # noqa: E731
        else:
            step = lambda s, zz, j: _step_egm(aprime, a_grid, s, zz)       # noqa: E731
    else:
        pol = np.asarray(sol.raw.policy_aprime)
        if c.finite:
            step = lambda s, zz, j: _step_vfi(pol[j], s, zz)               # noqa: E731
        else:
            step = lambda s, zz, j: _step_vfi(pol, s, zz)                  # noqa: E731

    survival = c.model._survival if c.finite else None
    alive = None
    if survival is not None:
        survival = np.asarray(survival, dtype=float)
        alive = np.ones((n_agents, periods), dtype=bool)

    S = np.empty((n_agents, periods + 1), dtype=state.dtype)     # state at t, plus t = periods
    Z = np.empty((n_agents, periods), dtype=np.intp)
    total = burn_in + periods
    for t in range(total):
        k = t - burn_in                                          # period (age in a life cycle)
        nxt = step(state, z, k)
        if k >= 0:
            S[:, k], Z[:, k] = state, z
        if t < total - 1:
            z = _draw_markov(rng, P, z)
            if alive is not None:
                alive[:, k + 1] = alive[:, k] & (rng.random(n_agents) < survival[k])
        state = nxt
    S[:, periods] = state

    # values for the household-function arguments, each (n_agents, periods)
    if egm:
        cur, nxt = [S[:, :-1]], [S[:, 1:]]
    else:
        shape = _state_shape(sol)
        cur = [g[i] for g, i in zip(c.state_grids, np.unravel_index(S[:, :-1], shape))]
        nxt = [g[i] for g, i in zip(c.state_grids, np.unravel_index(S[:, 1:], shape))]
    if c.z_grids:
        zs = [g[i] for g, i in zip(c.z_grids, np.unravel_index(Z, _shock_shape(sol)))]
    else:
        zs = [np.zeros((n_agents, periods))]
    age = np.arange(periods) if c.finite else None
    ages = [np.broadcast_to(age, (n_agents, periods))] if c.finite else []
    args = nxt + cur + zs + ages + list(sol.params.values())
    arrays = {}
    for name, fn in compiled:
        val = np.asarray(fn(*args, xp=np), dtype=float)
        arrays[name] = np.array(np.broadcast_to(val, (n_agents, periods)))
    return Panel(arrays, age=age, alive=alive,
                 source=f"puremacro.dp simulation of {sol.model_name!r} ({sol.method})")
