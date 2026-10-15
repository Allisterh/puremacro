"""Result object for puremacro.dp: one solution type whatever the back end."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from puremacro.reports import df_to_latex, df_to_markdown, df_to_typst


class DPSolution:
    """Solved dp model: value, policies, distribution, prices and aggregates.

    ``raw`` is the underlying ``VFISolution``, ``FiniteHorizonSolution`` or
    ``EGMHouseholdSolution`` (``method="egm"``; continuous ``c``/``aprime``) or
    ``HJBSolution`` (``method="hjb"``; ``distribution`` is then the mass at each
    node, the KFE density times the quadrature weights, and ``P`` the generator) and
    ``equilibrium`` the ``EquilibriumResult`` when prices were solved for.
    Arrays are (n_states, n_shocks), with a leading age axis for finite horizons.
    """

    def __init__(self, compiled, raw, *, params: dict, distribution=None,
                 aggregates: dict | None = None, prices: dict | None = None,
                 equilibrium=None, method: str | None = None):
        self._c = compiled
        self.raw = raw
        self.model_name = compiled.model.name
        self.method = method or ("vfi-finite" if compiled.finite else "vfi")
        self.params = dict(params)
        self.distribution = distribution
        self.aggregates = dict(aggregates or {})
        self.prices = dict(prices or {})
        self.equilibrium = equilibrium

    @property
    def V(self) -> np.ndarray:
        return self.raw.V

    @property
    def grids(self) -> dict[str, np.ndarray]:
        """State and shock grids by name."""
        out = dict(zip(self._c.state_names, self._c.state_grids))
        out.update(zip(self._c.shock_names, self._c.z_grids))
        return out

    @property
    def P(self) -> np.ndarray:
        """Transition matrix of the (product) exogenous chain."""
        return self._c.P

    def policy(self, expr: str) -> np.ndarray:
        """Any model expression evaluated at the optimal choice, e.g. ``policy("c")``
        or ``policy("a(+1)")``; shape (n_states, n_shocks) or (T, n_states, n_shocks)."""
        from puremacro.dp._expr import parse

        c = self._c
        if hasattr(c, "policy_values"):
            return c.policy_values(expr, self.raw, self.params)
        timed = c.state_names + c.shock_names + ([c.model._discrete[0]] if c.model._discrete else [])
        fn = c._household_function("_dp_policy", parse(expr, timed))
        if not c.finite:
            return c.evaluate(fn, self.raw, self.params)
        return np.stack([c.evaluate(fn, self.raw, self.params, age=j)
                         for j in range(self.raw.horizon)])

    def mean(self, expr: str):
        """Population mean of ``expr`` under the solved distribution: a float, or
        one value per age for a finite horizon (each cohort has mass 1)."""
        if self.distribution is None:
            raise ValueError("no distribution: solve with distribution=True")
        vals = self.policy(expr)
        if not self._c.finite:
            return float(np.sum(self.distribution * vals))
        return np.array([float(np.sum(self.distribution[j] * vals[j])) for j in range(vals.shape[0])])

    def simulate(self, n_agents: int = 10_000, periods: int | None = None, *,
                 seed: int | None = 0, init: str | dict | None = None, burn_in: int = 0,
                 record: Sequence[str] | None = None):
        """Simulate individual histories under the solved policy; returns a ``dp.Panel``.

        ``periods`` defaults to 200 for an infinite horizon. In a life cycle the
        panel runs over the T ages of the horizon (``panel.age`` is 0..T-1);
        a ``periods`` other than T is ignored with a ``UserWarning``.

        ``init`` defaults to ``"stationary"`` for an infinite horizon: each
        agent's (state, shock) cell is drawn from ``sol.distribution`` and the
        agent starts at that cell's grid node, so the panel is in the ergodic
        distribution from t = 0. In a life cycle it defaults to ``"newborn"``:
        age-0 cells are drawn from the declared ``newborns`` (normalised) or, by
        the solver's rule, the lowest asset node with the shock drawn from the
        stationary distribution of ``sol.P``. A dict ``{state: value or array}``
        fixes the initial states (every state must be given); a value off the
        grid is snapped to the nearest node under VFI, with one warning, and
        kept as is (clipped to the grid ends) under EGM. Shocks may be fixed the
        same way, on their grid; otherwise they are drawn from the ergodic
        distribution of ``sol.P``.

        In a life cycle with declared survival, an agent alive at age j lives to
        j+1 with probability ``survival[j]``; ``panel.alive`` records it. The dead
        keep their (finite) paths in the arrays and are left out of every panel
        statistic, which is NaN at an age nobody reaches.

        ``burn_in`` (infinite horizon only) periods are simulated first and
        dropped, so the panel holds ``periods`` periods. ``record`` lists model
        expressions to evaluate along the path, as ``sol.policy`` takes them
        (``"c"``, ``"a(+1)"``, ``"w*exp(z)"``, and ``age`` or age-varying
        parameters in a life cycle); by default the states, the shocks and the
        controls (the ``c`` local, any ``control()`` and the discrete choice).
        They are checked before anything is drawn. VFI states stay on grid nodes; EGM interpolates
        the next-asset policy linearly between nodes. Shocks move by ``sol.P``.

        The same ``seed`` gives bit-identical panels (one PCG64 generator).
        ``n_agents * periods`` above 5e8 raises ``ValueError``. Continuous-time
        (HJB) and aggregate-risk (Krusell-Smith) solutions raise
        ``NotImplementedError``.
        """
        from puremacro.dp._simulate import simulate

        return simulate(self, n_agents, periods, seed=seed, init=init, burn_in=burn_in,
                        record=record)

    # -- exact-distribution statistics -------------------------------------
    # Life-cycle pooling rule, shared by gini, lorenz, top_share and quantile:
    # ``sol.distribution`` gives every cohort mass 1 and ignores survival, so
    # with ``age=None`` cohort j is weighted by ``population_shares()[j]``;
    # ``age=j`` uses that cohort alone and ``age="all"`` returns one result per age.

    def population_shares(self) -> np.ndarray:
        """Share of the population at each age in a life cycle, shape (T,).

        Age j has relative size ``prod(survival[:j])`` (newborns have size 1),
        normalised to sum to 1; uniform when no survival was declared. This is
        the weight of each cohort when the statistics pool ages; it is not
        applied to ``sol.distribution``, where every cohort keeps mass 1.
        """
        if not self._c.finite:
            raise ValueError("population_shares() applies to life-cycle (finite-horizon) solutions only")
        J = int(self.raw.horizon)
        surv = self._c.model._survival
        if surv is None:
            return np.full(J, 1.0 / J)
        surv = np.asarray(surv, dtype=float)
        sizes = np.cumprod(np.concatenate([[1.0], surv[:-1]]))
        return sizes / sizes.sum()

    def _mass_points(self, expr: str, age):
        """Values and masses for the statistics: a list of (key, x, w) cross-sections."""
        if self.method.endswith("+ks"):
            raise ValueError("no stationary distribution: Krusell-Smith solutions have none")
        if self.distribution is None:
            raise ValueError("no distribution: solve with distribution=True")
        vals = np.asarray(self.policy(expr), dtype=float)
        dist = np.asarray(self.distribution, dtype=float)
        if not self._c.finite:
            if age is not None:
                raise ValueError("age applies to life-cycle (finite-horizon) solutions only")
            return [(None, vals.ravel(), dist.ravel())]
        J = vals.shape[0]
        if age is None:
            w = self.population_shares()[:, None, None] * dist
            return [(None, vals.ravel(), w.ravel())]
        if isinstance(age, str):
            if age != "all":
                raise ValueError(f"age must be None, an integer age or 'all', got {age!r}")
            return [(j, vals[j].ravel(), dist[j].ravel()) for j in range(J)]
        j = int(age)
        if not 0 <= j < J or j != age:
            raise ValueError(f"age must be an integer in 0..{J - 1}, got {age!r}")
        return [(None, vals[j].ravel(), dist[j].ravel())]

    @staticmethod
    def _positive(x: np.ndarray, w: np.ndarray):
        # nodes without mass carry no information (and would be the q=0 quantile)
        keep = w > 0
        return x[keep], w[keep][:, None]

    def _scalar_stat(self, expr, age, fn):
        pts = self._mass_points(expr, age)
        out = [fn(*self._positive(x, w)) for _, x, w in pts]
        if pts[0][0] is None:
            return out[0]
        return pd.Series(out, index=pd.Index([k for k, _, _ in pts], name="age"), name=expr)

    def gini(self, expr: str = "a", *, age: int | str | None = None):
        """Gini coefficient of ``expr`` under the solved distribution.

        The standard weighted Gini on the Lorenz curve, ``1 - sum_i w_i (L_i + L_{i-1})``
        with each grid node a mass point: the same function as ``MicroFrame.gini``.
        Negative values are allowed (it can then exceed 1); NaN when the total is
        <= 0. In a life cycle, ``age=None`` pools all ages with cohort j weighted by
        ``population_shares()[j]``, ``age=j`` uses cohort j alone and ``age="all"``
        returns a ``pd.Series`` indexed by age.
        """
        from puremacro._wstats import w_gini

        return self._scalar_stat(expr, age, lambda x, W: float(w_gini(x, W)[0]))

    def top_share(self, expr: str = "a", p: float = 0.01, *, age: int | str | None = None):
        """Share of the total of ``expr`` held by the top ``p`` of the population,
        ``1 - L(1 - p)``, splitting the node that straddles the cut (as
        ``MicroFrame.top_share``). Life cycle: ``age=None`` pools ages weighted by
        ``population_shares()``, ``age=j`` is one cohort, ``age="all"`` a Series by age.
        """
        from puremacro._wstats import w_top_share

        if not 0.0 < p < 1.0:
            raise ValueError(f"p must lie strictly between 0 and 1, got {p}")
        return self._scalar_stat(expr, age, lambda x, W: float(w_top_share(x, W, p)[0]))

    def quantile(self, expr: str = "a", q=0.5, *, age: int | str | None = None):
        """Quantile(s) of ``expr``: the smallest value whose cumulative mass reaches
        ``q`` (as ``MicroFrame.quantile``), over nodes with positive mass. A float
        for a scalar ``q``, an array for a sequence. Life cycle: ``age=None`` pools
        ages weighted by ``population_shares()``, ``age=j`` is one cohort and
        ``age="all"`` gives a Series (scalar ``q``) or a DataFrame (one column per q)
        indexed by age.
        """
        from puremacro._wstats import w_quantile

        qs = np.asarray(q, dtype=float)
        if qs.ndim > 1 or np.any((qs < 0) | (qs > 1)):
            raise ValueError(f"q must lie in [0, 1], got {q}")

        def fn(x, W):
            r = np.array([float(w_quantile(x, W, qi)[0]) for qi in np.atleast_1d(qs)])
            return float(r[0]) if qs.ndim == 0 else r

        out = self._scalar_stat(expr, age, fn)
        if isinstance(out, pd.Series) and qs.ndim == 1:
            return pd.DataFrame(np.vstack(out.to_numpy()), index=out.index, columns=list(qs))
        return out

    def lorenz(self, expr: str = "a", n: int = 101, *, age: int | str | None = None) -> pd.DataFrame:
        """Lorenz curve of ``expr`` at ``n`` evenly spaced population shares.

        Columns ``pop_share`` and ``value_share``; the curve is piecewise linear
        through the nodes sorted by value (each node a mass point), so
        ``1 - value_share`` at ``1 - p`` is ``top_share(expr, p)``. All NaN when the
        total is <= 0. Life cycle: ``age=None`` pools ages weighted by
        ``population_shares()``, ``age=j`` is one cohort, and ``age="all"`` stacks the
        curves of every age with an ``age`` column first.
        """
        from puremacro._wstats import lorenz

        if int(n) != n or n < 2:
            raise ValueError(f"n must be an integer >= 2, got {n}")
        grid = np.linspace(0.0, 1.0, int(n))
        frames = []
        for key, x, w in self._mass_points(expr, age):
            _, _, cw, S = lorenz(*self._positive(x, w))
            N, tot = (cw[-1, 0], S[-1, 0]) if cw.size else (0.0, 0.0)
            if N > 0 and tot > 0:
                share = np.interp(grid, np.concatenate([[0.0], cw[:, 0] / N]),
                                  np.concatenate([[0.0], S[:, 0] / tot]))
            else:
                share = np.full(grid.shape, np.nan)
            df = pd.DataFrame({"pop_share": grid, "value_share": share})
            if key is not None:
                df.insert(0, "age", key)
            frames.append(df)
        return frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)

    def summary(self) -> pd.DataFrame:
        rec = [("Model", self.model_name), ("Method", self.method),
               ("States", ", ".join(f"{k} ({g.size})" for k, g in zip(self._c.state_names, self._c.state_grids))),
               ("Shocks", ", ".join(f"{k} ({g.size})" for k, g in zip(self._c.shock_names, self._c.z_grids)) or "none")]
        if hasattr(self.raw, "n_iter"):
            rec.append(("Iterations", str(self.raw.n_iter)))
        for k, v in self.prices.items():
            rec.append((f"Price {k}", f"{v:.6g}"))
        for k, v in self.aggregates.items():
            rec.append((f"Aggregate {k}", f"{v:.6g}"))
        return pd.DataFrame(rec, columns=["Metric", "Value"]).set_index("Metric")

    def to_markdown(self, **kwargs: Any) -> str:
        return df_to_markdown(self.summary(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        return df_to_latex(self.summary(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        return df_to_typst(self.summary(), **kwargs)

    def __repr__(self) -> str:
        extra = "".join(f", {k}={v:.6g}" for k, v in {**self.prices, **self.aggregates}.items())
        return f"DPSolution({self.model_name!r}, method={self.method!r}{extra})"

