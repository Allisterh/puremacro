"""Simulated individual histories from a dp solution.

A :class:`Panel` holds one (n_agents, periods) array per recorded column. It is
built by ``DPSolution.simulate``, not by users. Its statistics go through the
survey estimators of :class:`puremacro.fetch.micro.MicroFrame` (equal weights),
so a simulated panel and survey microdata are summarised by the same code.

Statistics take expressions in Dynare syntax over the recorded columns (for
example ``"a + 2*z"`` or ``"c^2"``); a recorded column's own name, such as
``"y(z)"``, is also accepted as is.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from puremacro import _wstats
from puremacro.dp._expr import ModelSpecError, compile_function, emit, parse, symbols

_RESERVED = ("id", "t", "age", "alive", "_w")
_VALUE = "_value"          # column holding an evaluated expression in a MicroFrame


class Panel:
    """Individual histories: one (n_agents, periods) array per recorded column.

    ``age`` is the (periods,) age of each period in a life cycle and None for an
    infinite horizon. ``alive`` is a bool (n_agents, periods) array, or None
    when every agent is alive in every period. Dead rows are kept in the arrays
    (the panel stays rectangular) and in :meth:`to_frame`, and dropped from
    :meth:`to_microframe` and every statistic.
    """

    def __init__(self, arrays: Mapping[str, np.ndarray], *, age: np.ndarray | None,
                 alive: np.ndarray | None, source: str):
        if not arrays:
            raise ValueError("a Panel needs at least one recorded column")
        arrays = {str(k): np.asarray(v) for k, v in arrays.items()}
        shapes = {k: v.shape for k, v in arrays.items()}
        shape = next(iter(shapes.values()))
        if len(shape) != 2 or any(s != shape for s in shapes.values()):
            raise ValueError(f"every column must have the same (n_agents, periods) shape; got {shapes}")
        clash = [k for k in arrays if k in _RESERVED]
        if clash:
            raise ValueError(f"column names {clash} are reserved ({list(_RESERVED)})")
        n_agents, periods = shape
        if age is not None:
            age = np.asarray(age)
            if age.shape != (periods,):
                raise ValueError(f"age must have shape ({periods},), got {age.shape}")
        if alive is not None:
            alive = np.asarray(alive)
            if alive.dtype != bool or alive.shape != shape:
                raise ValueError(f"alive must be a bool array of shape {shape}, "
                                 f"got {alive.dtype} {alive.shape}")
        self._arrays = arrays
        self._age = age
        self._alive = alive
        self.source = str(source)

    # ----- access ----------------------------------------------------------
    @property
    def names(self) -> list[str]:
        """The recorded columns, in recording order."""
        return list(self._arrays)

    @property
    def n_agents(self) -> int:
        return next(iter(self._arrays.values())).shape[0]

    @property
    def periods(self) -> int:
        return next(iter(self._arrays.values())).shape[1]

    @property
    def age(self) -> np.ndarray | None:
        """Age of each period (life cycle), or None (infinite horizon)."""
        return self._age

    @property
    def alive(self) -> np.ndarray | None:
        """Bool (n_agents, periods) survival record, or None if all are alive."""
        return self._alive

    def __getitem__(self, name: str) -> np.ndarray:
        try:
            return self._arrays[name]
        except KeyError:
            raise KeyError(f"{name!r} is not recorded; the panel has {self.names}") from None

    def __repr__(self) -> str:
        return f"Panel(n_agents={self.n_agents}, periods={self.periods}, columns={self.names})"

    # ----- frames ----------------------------------------------------------
    def to_frame(self, columns: Sequence[str] | None = None) -> pd.DataFrame:
        """Long format, one row per agent and period, agent by agent: ``id``,
        ``t``, then ``age`` and ``alive`` when present, then the recorded
        columns. Dead rows are kept."""
        return self._long(self._select(columns))

    def to_microframe(self, columns: Sequence[str] | None = None, t: int | None = None):
        """A :class:`~puremacro.fetch.micro.MicroFrame` with one equally
        weighted row (``_w = 1``) per living agent and period, so ``by="t"``
        or ``by="age"`` works with its estimators. With ``t`` given, it holds
        that one cross-section. Dead rows are dropped."""
        return self._microframe(self._select(columns), t=t)

    def cross_section(self, t: int) -> pd.DataFrame:
        """Every agent at period ``t`` (dead rows included), as in :meth:`to_frame`."""
        return self._long(self._arrays, t=self._check_t(t))

    def path(self, i: int) -> pd.DataFrame:
        """Agent ``i``'s history over every period, as in :meth:`to_frame`."""
        if not 0 <= int(i) < self.n_agents:
            raise IndexError(f"agent {i} out of range [0, {self.n_agents})")
        return self._long(self._arrays, i=int(i))

    # ----- statistics ------------------------------------------------------
    def mean(self, expr: str, by: str | None = None):
        """Mean of ``expr`` over living agents (all periods pooled when ``by``
        is None; ``by`` is ``"t"`` or, in a life cycle, ``"age"``)."""
        return self._stat("mean", expr, by)

    def quantile(self, expr: str, q: float, by: str | None = None):
        """``q``-quantile of ``expr`` (inverse CDF), as :meth:`mean` groups it."""
        return self._stat("quantile", expr, by, q)

    def gini(self, expr: str, by: str | None = None):
        """Gini coefficient of ``expr``; negatives allowed, NaN when the total
        is <= 0. Grouped as :meth:`mean`."""
        return self._stat("gini", expr, by)

    def top_share(self, expr: str, p: float, by: str | None = None):
        """Share of the total of ``expr`` held by the top ``p`` of agents.
        Grouped as :meth:`mean`."""
        return self._stat("top_share", expr, by, p)

    def summary(self) -> pd.DataFrame:
        """Mean, standard deviation (population, ddof 0) and Gini of each
        recorded column over living agents: at the last period (infinite
        horizon, indexed by ``variable``) or at every age (life cycle,
        indexed by ``variable`` and ``age``)."""
        ts = range(self.periods) if self._age is not None else [self.periods - 1]
        rows, index = [], []
        for name, arr in self._arrays.items():
            for t in ts:
                x = np.asarray(arr[:, t], dtype=float)
                keep = ~np.isnan(x)
                if self._alive is not None:
                    keep &= self._alive[:, t]
                x = x[keep]
                if x.size:
                    g = float(_wstats.w_gini(x, np.ones((x.size, 1)))[0])
                    rows.append((float(x.mean()), float(x.std()), g))
                else:
                    rows.append((np.nan, np.nan, np.nan))
                index.append((name, self._age[t].item()) if self._age is not None else name)
        if self._age is not None:
            idx = pd.MultiIndex.from_tuples(index, names=["variable", "age"])
        else:
            idx = pd.Index(index, name="variable")
        return pd.DataFrame(rows, index=idx, columns=["mean", "std", "gini"])

    # ----- internals -------------------------------------------------------
    def _select(self, columns) -> dict[str, np.ndarray]:
        if columns is None:
            return self._arrays
        if isinstance(columns, str):
            columns = [columns]
        return {c: self[c] for c in columns}

    def _check_t(self, t) -> int:
        if not 0 <= int(t) < self.periods:
            raise IndexError(f"period {t} out of range [0, {self.periods})")
        return int(t)

    def _long(self, values: Mapping[str, np.ndarray], *, t: int | None = None,
              i: int | None = None, alive: bool = True) -> pd.DataFrame:
        agents = np.arange(self.n_agents) if i is None else np.array([i])
        ts = np.arange(self.periods) if t is None else np.array([t])
        sel = np.ix_(agents, ts)
        data = {"id": np.repeat(agents, ts.size), "t": np.tile(ts, agents.size)}
        if self._age is not None:
            data["age"] = np.tile(self._age[ts], agents.size)
        if alive and self._alive is not None:
            data["alive"] = self._alive[sel].ravel()
        for name, arr in values.items():
            data[name] = arr[sel].ravel()
        return pd.DataFrame(data)

    def _microframe(self, values: Mapping[str, np.ndarray], t: int | None = None):
        from puremacro.fetch.micro._design import MicroFrame, SurveyDesign

        t = None if t is None else self._check_t(t)
        df = self._long(values, t=t, alive=False)
        if self._alive is not None:
            agents = np.arange(self.n_agents)
            ts = np.arange(self.periods) if t is None else np.array([t])
            df = df[self._alive[np.ix_(agents, ts)].ravel()].reset_index(drop=True)
        df["_w"] = 1.0
        return MicroFrame(df, SurveyDesign.weights_only("_w"), unit="agent",
                          source=self.source, vintage="simulated")

    def _evaluate(self, expr: str) -> np.ndarray:
        """Values of ``expr`` on every agent and period, as (n_agents, periods)."""
        if expr in self._arrays:
            return self._arrays[expr]
        env = {k: v for k, v in self._arrays.items() if k.isidentifier()}
        shape = (self.n_agents, self.periods)
        env.setdefault("t", np.broadcast_to(np.arange(self.periods), shape))
        if self._age is not None:
            env.setdefault("age", np.broadcast_to(self._age, shape))
        node = parse(expr, variables=list(env))
        used = sorted(symbols(node))
        unknown = [n for n, _ in used if n not in env]
        if unknown:
            raise ModelSpecError(f"{expr!r} uses {sorted(set(unknown))}, which the panel does "
                                 f"not record; it has {self.names}")
        timed = [f"{n}({lead:+d})" for n, lead in used if lead != 0]
        if timed:
            raise ModelSpecError(f"{expr!r} uses timed values {timed}; record them when "
                                 "simulating (record=[...]) and refer to them by name")
        args = sorted({n for n, _ in used})
        ident = {n: f"_v{k}" for k, n in enumerate(args)}
        fn = compile_function("_panel_expr", [ident[n] for n in args], [],
                              emit(node, lambda n, lead: ident[n]))
        out = np.asarray(fn(*(env[n] for n in args)), dtype=float)
        return np.broadcast_to(out, shape)

    def _stat(self, kind: str, expr: str, by: str | None, arg=None):
        allowed = ("t", "age") if self._age is not None else ("t",)
        if by is not None and by not in allowed:
            raise ValueError(f"by must be None or one of {list(allowed)}, got {by!r}")
        mf = self._microframe({_VALUE: self._evaluate(expr)})
        method = getattr(mf, kind)
        res = method(_VALUE, arg, by=by) if arg is not None else method(_VALUE, by=by)
        if by is None:
            return float(res["estimate"].iloc[0])
        out = pd.Series(res["estimate"].to_numpy(), index=pd.Index(res[by], name=by), name=kind)
        # a period or age in which every agent is dead has no rows: report NaN for it
        keys = self._age if by == "age" else np.arange(self.periods)
        return out.reindex(pd.Index(keys, name=by))
