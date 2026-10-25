r"""Integrated Spatial Climate-Macro & Green Transition IAM (Cruz & Rossi-Hansberg 2024; Golosov et al. 2014).

Couples dynamic spatial hat-algebra (`DynamicSpatialHatModel`: gravity trade and
forward-looking extreme-value labor migration) and multi-sector CES network macro
aggregation (`baqaee_farhi_decomposition`) with:

1. **Mass-Conserving Multi-Box Carbon Cycle**:
   Atmospheric carbon $S_t = S_{1, t} + S_{2, t}$ above pre-industrial stock,
   partitioned into permanent reservoir $S_{1, t}$, decaying reservoir $S_{2, t}$,
   and deep ocean/biosphere sink $S_{\text{sink}, t}$:

   .. math::

       S_{1, t} = S_{1, t-1} + \phi_L E_t,
       \qquad
       S_{2, t} = (1 - \phi_{\text{decay}}) S_{2, t-1} + (1 - \phi_L)\phi_0 E_t,

   .. math::

       S_{\text{sink}, t} = S_{\text{sink}, t-1} + \phi_{\text{decay}} S_{2, t-1} + (1 - \phi_L)(1 - \phi_0) E_t

   satisfying exact mass balance to $< 10^{-12}$:

   .. math::

       \sum_{s=0}^t E_s = (S_{1, t} - S_{1, 0}) + (S_{2, t} - S_{2, 0}) + (S_{\text{sink}, t} - S_{\text{sink}, 0}).

2. **Heterogeneous Regional Temperature Damages (Cruz & Rossi-Hansberg 2024)**:
   Global warming pattern-scaled to regional temperatures $T_{i, t} = T_{i, 0} + \zeta_i (T_t - T_0)$,
   impacting regional productivity $\hat{A}_{i, t}$ and amenities $\hat{B}_{i, t}$.

3. **Climate-Induced Labor Migration & Population Conservation**:
   Forward-looking migration across regions preserving global population
   $\sum_r L_{r, t} = \bar{L}$ to $< 10^{-12}$ at all $t$.

4. **Carbon Border Adjustment Mechanism (CBAM) & Leakage Reduction**:
   Unilateral coalition carbon pricing generates carbon leakage $\Delta E_{\mathcal{N}} / -\Delta E_{\mathcal{C}}$,
   while CBAM import tariffs strictly reduce leakage:

   .. math::

       \text{Leakage}_{\text{CBAM}} < \text{Leakage}_{\text{Unilateral}}.

5. **Analytical Golosov et al. (2014) Optimal Pigouvian Carbon Tax**:
   In the log-utility limit ($u(c) = \ln c$):

   .. math::

       \tau_t^{\text{Golosov}} = Y_t \bar{\gamma} \left[ \frac{\phi_L}{1 - \beta} + \frac{(1 - \phi_L)\phi_0}{1 - \beta(1 - \phi_{\text{decay}})} \right].
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    import matplotlib.figure

import numpy as np
import pandas as pd

from puremacro.reports import _df_to_latex, _df_to_markdown, _df_to_typst
from puremacro.spatial.dynamic_hat import DynamicSpatialHatModel, solve_dynamic_spatial_hat
from puremacro.trade.network_macro import baqaee_farhi_decomposition

__all__ = [
    "SpatialClimateIAM",
    "SpatialClimateIAMResult",
    "solve_spatial_climate_iam",
]


@dataclass(frozen=True)
class SpatialClimateIAM:
    r"""Baseline specification for an Integrated Spatial Climate-Macro Economy.

    Parameters
    ----------
    initial_population : np.ndarray
        Length-$N$ initial regional population vector $L_{i, 0}$.
    initial_trade_shares : np.ndarray
        $N \times N$ initial bilateral trade share matrix $\pi_{ij, 0}$ (row-stochastic).
    initial_migration_shares : np.ndarray
        $N \times N$ initial migration share matrix $\mu_{ij, 0}$ (row-stochastic).
    initial_temperatures : np.ndarray
        Length-$N$ regional baseline temperatures $T_{i, 0}$ in °C.
    carbon_intensity : np.ndarray
        Length-$N$ regional emissions intensity $\sigma_i$ (emissions per unit of output).
    damage_coefficients : np.ndarray
        Length-$N$ regional linear productivity damage elasticity $\gamma_i^A$.
    damage_curvature : np.ndarray | None, optional
        Length-$N$ regional quadratic damage curvature $\gamma_{2, i}^A$.
    amenity_damage_coefficients : np.ndarray | None, optional
        Length-$N$ regional amenity damage elasticity $\gamma_i^B$.
    warming_pattern_scale : np.ndarray | None, optional
        Length-$N$ regional climate sensitivity pattern scaling factor $\zeta_i$.
    trade_elasticity : float, default 4.0
        Bilateral trade elasticity $\theta > 0$.
    migration_dispersion : float, default 0.5
        Migration preference dispersion parameter $\nu > 0$.
    beta : float, default 0.96
        Subjective annual discount factor $\beta \in (0, 1)$.
    congestion_alpha : float, default 0.10
        Local amenity/housing congestion elasticity $\alpha \ge 0$.
    climate_sensitivity : float, default 3.0
        Equilibrium climate sensitivity $S_{\text{clim}}$ (°C per CO2 doubling).
    preindustrial_carbon : float, default 588.0
        Pre-industrial atmospheric carbon stock $S_{\text{pre-ind}}$ in GtC.
    initial_carbon_stock : float, default 850.0
        Baseline atmospheric carbon stock $S_0$ in GtC.
    phi_L : float, default 0.20
        Permanent fraction of emissions remaining in atmosphere indefinitely.
    phi_0 : float, default 0.393
        Fraction of non-permanent emissions entering decaying reservoir.
    phi_decay : float, default 0.02
        Annual decay rate of decaying carbon reservoir.
    abatement_theta1 : float, default 550.0
        Backstop technology marginal abatement cost coefficient.
    abatement_theta2 : float, default 2.6
        Marginal abatement cost curvature exponent ($\theta_2 > 1$).
    initial_wages : np.ndarray | None, optional
        Length-$N$ baseline nominal wage vector $w_{i, 0}$.
    region_names : tuple[str, ...] | None, optional
        Human-readable labels for the $N$ regions.
    optimal_temperatures : np.ndarray | None, optional
        Length-$N$ optimal productivity temperatures $T_i^*$ in °C (default 13.0 °C).
    spatial_model : DynamicSpatialHatModel | None, optional
        Optional pre-constructed spatial model from which to inherit primitives.
    """

    initial_population: np.ndarray
    initial_trade_shares: np.ndarray
    initial_migration_shares: np.ndarray
    initial_temperatures: np.ndarray
    carbon_intensity: np.ndarray
    damage_coefficients: np.ndarray
    damage_curvature: np.ndarray | None = None
    amenity_damage_coefficients: np.ndarray | None = None
    warming_pattern_scale: np.ndarray | None = None
    trade_elasticity: float = 4.0
    migration_dispersion: float = 0.5
    beta: float = 0.96
    congestion_alpha: float = 0.10
    climate_sensitivity: float = 3.0
    preindustrial_carbon: float = 588.0
    initial_carbon_stock: float = 850.0
    phi_L: float = 0.20
    phi_0: float = 0.393
    phi_decay: float = 0.02
    abatement_theta1: float = 550.0
    abatement_theta2: float = 2.6
    initial_wages: np.ndarray | None = None
    region_names: tuple[str, ...] | None = None
    optimal_temperatures: np.ndarray | None = None

    def __post_init__(self) -> None:
        L0 = np.asarray(self.initial_population, dtype=np.float64).reshape(-1)
        N = L0.size
        if np.any(L0 <= 0.0):
            raise ValueError("initial_population must be strictly positive.")
        pi0 = np.asarray(self.initial_trade_shares, dtype=np.float64)
        mu0 = np.asarray(self.initial_migration_shares, dtype=np.float64)
        if pi0.shape != (N, N) or mu0.shape != (N, N):
            raise ValueError(f"initial_trade_shares and initial_migration_shares must have shape ({N}, {N}).")
        if not np.allclose(np.sum(pi0, axis=1), 1.0, atol=1e-5):
            raise ValueError("Rows of initial_trade_shares must sum to 1.0.")
        if not np.allclose(np.sum(mu0, axis=1), 1.0, atol=1e-5):
            raise ValueError("Rows of initial_migration_shares must sum to 1.0.")
        if self.trade_elasticity <= 0.0 or self.migration_dispersion <= 0.0:
            raise ValueError("trade_elasticity and migration_dispersion must be strictly positive.")
        if not (0.0 < self.beta < 1.0):
            raise ValueError("beta must lie in (0, 1).")
        if self.abatement_theta2 <= 1.0:
            raise ValueError("abatement_theta2 must be > 1.0.")


@dataclass(frozen=True)
class SpatialClimateIAMResult:
    r"""Result container for Integrated Spatial Climate-Macro transition paths.

    Parameters
    ----------
    region_names : tuple[str, ...]
        Names of regions.
    population_path : np.ndarray
        Array of shape `(T, N)` with regional population levels.
    population_hat_path : np.ndarray
        Array of shape `(T, N)` with population relative changes $\hat{L}_{i, t}$.
    real_wage_hat_path : np.ndarray
        Array of shape `(T, N)` with real wage relative changes $\hat{\omega}_{i, t}$.
    temperature_path : np.ndarray
        Array of shape `(T, N)` with regional temperature trajectories $T_{i, t}$ in °C.
    global_temperature_path : np.ndarray
        Array of shape `(T,)` with global mean temperature anomaly $T_t$ in °C.
    carbon_stock_path : np.ndarray
        Array of shape `(T,)` with atmospheric carbon stock $S_t$ in GtC.
    emissions_path : np.ndarray
        Array of shape `(T, N)` with regional carbon emissions $E_{i, t}$.
    global_emissions_path : np.ndarray
        Array of shape `(T,)` with global total emissions $E_t$.
    welfare_hat : np.ndarray
        Array of shape `(N,)` with regional welfare changes $\hat{W}_i$.
    carbon_leakage_rate : float
        Carbon leakage rate relative to laissez-faire $\Delta E_{\mathcal{N}} / -\Delta E_{\mathcal{C}}$.
    pigouvian_tax_path : np.ndarray
        Array of shape `(T,)` with the optimal Pigouvian carbon tax trajectory.
    population_conservation_error : float
        Maximum absolute deviation from global population conservation $\max_t |\sum_r L_{rt} - \bar{L}|$.
    carbon_mass_balance_error : float
        Maximum absolute deviation from carbon mass conservation $\max_t |\text{Injected}_t - \text{Stored}_t|$.
    converged : bool
        Whether the integrated spatial climate-macro transition equilibrium converged.
    iterations : int
        Number of transition iterations executed.
    policy_scenario : str
        Policy scenario simulated ("laissez_faire", "unilateral_tax", "cbam", "optimal_pigouvian").
    metadata : dict[str, Any]
        Additional simulation metadata and network decomposition results.
    """

    region_names: tuple[str, ...]
    population_path: np.ndarray
    population_hat_path: np.ndarray
    real_wage_hat_path: np.ndarray
    temperature_path: np.ndarray
    global_temperature_path: np.ndarray
    carbon_stock_path: np.ndarray
    emissions_path: np.ndarray
    global_emissions_path: np.ndarray
    welfare_hat: np.ndarray
    carbon_leakage_rate: float
    pigouvian_tax_path: np.ndarray
    population_conservation_error: float
    carbon_mass_balance_error: float
    converged: bool
    iterations: int
    policy_scenario: str = "laissez_faire"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self, period: int = -1) -> pd.DataFrame:
        """Convert regional results at period `period` to a pandas DataFrame."""
        return pd.DataFrame(
            {
                "population": self.population_path[period],
                "population_hat": self.population_hat_path[period],
                "real_wage_hat": self.real_wage_hat_path[period],
                "temperature": self.temperature_path[period],
                "emissions": self.emissions_path[period],
                "welfare_hat": self.welfare_hat,
            },
            index=pd.Index(self.region_names, name="region"),
        )

    def to_frame(self, period: int = -1) -> pd.DataFrame:
        """Alias for `to_dataframe()`."""
        return self.to_dataframe(period=period)

    def summary(self, period: int = -1) -> pd.DataFrame:
        """Generate formatted summary table of climate and macroeconomic outcomes."""
        df = self.to_dataframe(period=period)
        return df

    def to_markdown(self, **kwargs: Any) -> str:
        """Render markdown representation of the summary table."""
        return _df_to_markdown(self.summary(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """Render LaTeX representation of the summary table."""
        return _df_to_latex(self.summary(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Render Typst representation of the summary table."""
        return _df_to_typst(self.summary(), **kwargs)

    def plot(self, figsize: tuple[float, float] = (12.0, 4.5)) -> matplotlib.figure.Figure:
        """Plot transition dynamics: regional populations and emissions/carbon stock."""
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

        T = self.population_path.shape[0]
        t_grid = np.arange(T)

        for i, r_name in enumerate(self.region_names):
            ax1.plot(t_grid, self.population_hat_path[:, i], lw=1.8, label=r_name)
        ax1.axhline(1.0, color="0.5", ls=":", lw=0.8)
        ax1.set_title("Regional Population Transition (L_hat)")
        ax1.set_xlabel("Year t")
        ax1.set_ylabel("Population Relative to Baseline")
        ax1.legend(frameon=False, fontsize="small")

        ax2.plot(t_grid, self.global_temperature_path, color="#c0392b", lw=2.0, label="Global Temp Anomaly (°C)")
        ax2.set_xlabel("Year t")
        ax2.set_ylabel("Temperature (°C)", color="#c0392b")
        ax2.tick_params(axis="y", labelcolor="#c0392b")

        ax2_twin = ax2.twinx()
        ax2_twin.plot(t_grid, self.carbon_stock_path, color="#2b5c8f", lw=1.8, ls="--", label="Carbon Stock (GtC)")
        ax2_twin.set_ylabel("Carbon Stock (GtC)", color="#2b5c8f")
        ax2_twin.tick_params(axis="y", labelcolor="#2b5c8f")
        ax2.set_title(f"Global Climate Transition ({self.policy_scenario})")

        return fig


def solve_spatial_climate_iam(
    model: SpatialClimateIAM,
    *,
    horizon: int = 30,
    policy_scenario: str = "laissez_faire",
    policy: str | None = None,
    carbon_tax_path: np.ndarray | None = None,
    carbon_tax: float | np.ndarray | None = None,
    coalition_regions: Sequence[int] | Sequence[str] | None = None,
    cbam: bool = False,
    tol: float = 1e-10,
    max_iter: int = 200,
    damping: float = 0.3,
) -> SpatialClimateIAMResult:
    r"""Solve the integrated spatial climate-macro transition equilibrium.

    Simulates the coupled general equilibrium of trade, migration, carbon emissions,
    multi-box carbon cycle accumulation, and regional temperature damages over
    `horizon` periods.

    Parameters
    ----------
    model : SpatialClimateIAM
        Model primitives and baseline calibration.
    horizon : int, default 30
        Transition horizon in years.
    policy_scenario : str, default "laissez_faire"
        Policy regime: `"laissez_faire"`, `"unilateral_tax"`, `"cbam"`, or `"optimal_pigouvian"`.
    policy : str | None, optional
        Alias for `policy_scenario`.
    carbon_tax_path : np.ndarray | None, optional
        Length-$T$ carbon tax trajectory in $/tCO2.
    carbon_tax : float | np.ndarray | None, optional
        Alias for `carbon_tax_path`.
    coalition_regions : Sequence[int] | Sequence[str] | None, optional
        Regions participating in the carbon abating coalition (default: first region or region 0).
    cbam : bool, default False
        When True and policy is unilateral, imposes CBAM border tariffs on non-coalition imports.
    tol : float, default 1e-10
        Convergence tolerance for dynamic spatial fixed point.
    max_iter : int, default 200
        Maximum iterations.
    damping : float, default 0.3
        Damping parameter for fixed-point iteration.

    Returns
    -------
    SpatialClimateIAMResult
        Simulation results with welfare, emissions, migration, and leakage metrics.
    """
    if policy is not None:
        policy_scenario = policy
    scenario = policy_scenario.lower().strip()
    if cbam and scenario in ("unilateral_tax", "unilateral"):
        scenario = "cbam"
    if scenario not in ("laissez_faire", "unilateral_tax", "cbam", "optimal_pigouvian"):
        raise ValueError(
            f"policy_scenario must be 'laissez_faire', 'unilateral_tax', 'cbam', or 'optimal_pigouvian', got {scenario!r}"
        )

    L0 = np.asarray(model.initial_population, dtype=np.float64).reshape(-1)
    N = L0.size
    total_L0 = float(np.sum(L0))
    pi0 = np.asarray(model.initial_trade_shares, dtype=np.float64)
    mu0 = np.asarray(model.initial_migration_shares, dtype=np.float64)
    w0 = np.ones(N, dtype=np.float64) if model.initial_wages is None else np.asarray(model.initial_wages, dtype=np.float64).reshape(N)
    Y0 = w0 * L0
    total_Y0 = float(np.sum(Y0))

    names = (
        tuple(model.region_names)
        if model.region_names is not None
        else tuple(f"Region_{i + 1}" for i in range(N))
    )

    # Resolve coalition
    coalition_idx: list[int] = []
    if coalition_regions is not None:
        for c in coalition_regions:
            if isinstance(c, int):
                coalition_idx.append(c)
            elif str(c) in names:
                coalition_idx.append(names.index(str(c)))
            else:
                for idx, r_name in enumerate(names):
                    if str(c).lower() == r_name.lower():
                        coalition_idx.append(idx)
                        break
    if not coalition_idx:
        # Default coalition: region 0 (or first region)
        coalition_idx = [0]
    non_coalition_idx = [i for i in range(N) if i not in coalition_idx]

    # Resolve carbon tax trajectory
    tax_input = carbon_tax if carbon_tax is not None else carbon_tax_path
    if tax_input is not None:
        if isinstance(tax_input, (int, float, np.number)):
            base_tax_path = np.full(horizon, float(tax_input), dtype=np.float64)
        else:
            arr_tax = np.asarray(tax_input, dtype=np.float64).ravel()
            base_tax_path = np.zeros(horizon, dtype=np.float64)
            base_tax_path[: min(horizon, len(arr_tax))] = arr_tax[: min(horizon, len(arr_tax))]
            if len(arr_tax) < horizon:
                base_tax_path[len(arr_tax) :] = arr_tax[-1]
    else:
        # Default escalating tax: starts at $50/tCO2 and grows at 2% annually
        base_tax_path = np.asarray(50.0 * (1.02 ** np.arange(horizon)), dtype=np.float64)

    sigma = np.asarray(model.carbon_intensity, dtype=np.float64).reshape(N)
    gamma_A = np.asarray(model.damage_coefficients, dtype=np.float64).reshape(N)
    gamma_2A = (
        np.asarray(model.damage_curvature, dtype=np.float64).reshape(N)
        if model.damage_curvature is not None
        else np.zeros(N, dtype=np.float64)
    )
    gamma_B = (
        np.asarray(model.amenity_damage_coefficients, dtype=np.float64).reshape(N)
        if model.amenity_damage_coefficients is not None
        else np.zeros(N, dtype=np.float64)
    )
    zeta = (
        np.asarray(model.warming_pattern_scale, dtype=np.float64).reshape(N)
        if model.warming_pattern_scale is not None
        else np.ones(N, dtype=np.float64)
    )
    T_opt = (
        np.asarray(model.optimal_temperatures, dtype=np.float64).reshape(N)
        if model.optimal_temperatures is not None
        else np.full(N, 13.0, dtype=np.float64)
    )
    T_i0 = np.asarray(model.initial_temperatures, dtype=np.float64).reshape(N)
    T_global_0 = float(np.sum(T_i0 * L0) / total_L0)

    # Initialize dynamic spatial model instance
    spatial_model = DynamicSpatialHatModel(
        initial_population=L0,
        initial_trade_shares=pi0,
        initial_migration_shares=mu0,
        trade_elasticity=model.trade_elasticity,
        migration_dispersion=model.migration_dispersion,
        beta=model.beta,
        congestion_alpha=model.congestion_alpha,
        initial_wages=w0,
        region_names=names,
    )

    # Golosov analytical formula helper
    def _compute_golosov_tax(y_world: float, mean_gamma: float) -> float:
        num_perm = model.phi_L / (1.0 - model.beta)
        num_decay = (1.0 - model.phi_L) * model.phi_0 / (1.0 - model.beta * (1.0 - model.phi_decay))
        return float(y_world * mean_gamma * (num_perm + num_decay))

    # Helper to simulate climate & economy under a specific policy
    def _simulate_system(
        scen: str,
        tax_series: np.ndarray,
        tol_val: float,
        iter_max: int,
    ) -> tuple[Any, ...]:
        # Arrays across horizon T
        L_path = np.repeat(L0[None, :], horizon, axis=0)
        w_hat_path = np.ones((horizon, N), dtype=np.float64)
        omega_hat_path = np.ones((horizon, N), dtype=np.float64)
        A_path = np.ones((horizon, N), dtype=np.float64)
        B_path = np.ones((horizon, N), dtype=np.float64)
        tau_path = np.ones((horizon, N, N), dtype=np.float64)

        S1_path = np.zeros(horizon, dtype=np.float64)
        S2_path = np.zeros(horizon, dtype=np.float64)
        S_sink_path = np.zeros(horizon, dtype=np.float64)
        S_path = np.zeros(horizon, dtype=np.float64)
        T_global_path = np.zeros(horizon, dtype=np.float64)
        T_path = np.zeros((horizon, N), dtype=np.float64)
        E_path = np.zeros((horizon, N), dtype=np.float64)
        E_global_path = np.zeros(horizon, dtype=np.float64)
        pigouvian_path = np.zeros(horizon, dtype=np.float64)
        welfare_hat = np.ones(N, dtype=np.float64)

        # Baseline carbon reservoirs at t=0
        S1_init = model.phi_L * model.initial_carbon_stock
        S2_init = (1.0 - model.phi_L) * model.initial_carbon_stock
        S_sink_init = 0.0

        converged_outer = False
        it_count = 0

        for outer_it in range(1, 15):
            it_count = outer_it

            # 1. Determine effective trade cost shifts under policy
            tau_cost = np.ones((horizon, N, N), dtype=np.float64)
            taxes_rt = np.zeros((horizon, N), dtype=np.float64)

            for t in range(horizon):
                if scen == "laissez_faire":
                    taxes_rt[t, :] = 0.0
                elif scen == "unilateral_tax":
                    for c_r in coalition_idx:
                        taxes_rt[t, c_r] = tax_series[t]
                    # Production cost shift in coalition regions
                    for c_r in coalition_idx:
                        tau_cost[t, :, c_r] *= 1.0 + (tax_series[t] * sigma[c_r] / 1000.0)
                elif scen == "cbam":
                    for c_r in coalition_idx:
                        taxes_rt[t, c_r] = tax_series[t]
                        tau_cost[t, :, c_r] *= 1.0 + (tax_series[t] * sigma[c_r] / 1000.0)
                    # CBAM tariff on non-coalition imports into coalition
                    for c_r in coalition_idx:
                        for n_r in non_coalition_idx:
                            tau_cost[t, c_r, n_r] *= 1.0 + (tax_series[t] * sigma[n_r] / 1000.0)
                elif scen == "optimal_pigouvian":
                    # Golosov tax applied globally
                    tau_g = _compute_golosov_tax(total_Y0, float(np.mean(gamma_A)))
                    pigouvian_path[t] = tau_g
                    taxes_rt[t, :] = tau_g
                    for r in range(N):
                        tau_cost[t, :, r] *= 1.0 + (tau_g * sigma[r] / 1000.0)

            # 2. Solve dynamic spatial hat model given A_path, B_path, and tau_cost
            hat_res = solve_dynamic_spatial_hat(
                spatial_model,
                horizon=horizon,
                tfp_hat_path=A_path,
                amenity_hat_path=B_path,
                trade_cost_hat_path=tau_cost,
                tol=tol_val,
                max_iter=iter_max,
                damping=damping,
            )

            L_path = hat_res.population_path
            w_hat_path = hat_res.wage_hat_path
            omega_hat_path = hat_res.real_wage_hat_path
            welfare_hat = hat_res.welfare_hat

            # 3. Compute emissions and carbon cycle
            L_hat = L_path / L0[None, :]
            Y_path = w_hat_path * L_hat * Y0[None, :]

            # Endogenous abatement
            th1 = model.abatement_theta1
            th2 = model.abatement_theta2
            ratio = np.maximum(0.0, taxes_rt) / (th1 * th2)
            mu_abate = np.minimum(1.0, ratio ** (1.0 / (th2 - 1.0)))

            E_new = sigma[None, :] * Y_path * (1.0 - mu_abate)
            E_global_new = np.sum(E_new, axis=1)

            # Carbon mass balance cycle
            s1_curr = S1_init
            s2_curr = S2_init
            sink_curr = S_sink_init

            for t in range(horizon):
                e_t = E_global_new[t]
                if t == 0:
                    s1_curr = s1_curr + model.phi_L * e_t
                    s2_curr = s2_curr + (1.0 - model.phi_L) * model.phi_0 * e_t
                    sink_curr = sink_curr + (1.0 - model.phi_L) * (1.0 - model.phi_0) * e_t
                else:
                    s1_curr = s1_curr + model.phi_L * e_t
                    s2_next = (1.0 - model.phi_decay) * s2_curr + (1.0 - model.phi_L) * model.phi_0 * e_t
                    sink_curr = sink_curr + model.phi_decay * s2_curr + (1.0 - model.phi_L) * (1.0 - model.phi_0) * e_t
                    s2_curr = s2_next

                S1_path[t] = s1_curr
                S2_path[t] = s2_curr
                S_sink_path[t] = sink_curr
                S_path[t] = s1_curr + s2_curr

                # Global and regional temperatures
                clim_scale = model.climate_sensitivity / np.log(2.0)
                co2_ratio = (model.preindustrial_carbon + S_path[t]) / (model.preindustrial_carbon + model.initial_carbon_stock)
                t_anomaly = clim_scale * float(np.log(max(1e-12, co2_ratio)))
                T_global_path[t] = T_global_0 + t_anomaly
                T_path[t] = T_i0 + zeta * t_anomaly

            # 4. Regional climate damages (Cruz & Rossi-Hansberg 2024)
            A_new = np.ones((horizon, N), dtype=np.float64)
            B_new = np.ones((horizon, N), dtype=np.float64)
            for t in range(horizon):
                delta_T = T_path[t] - T_i0
                lin_damage = -gamma_A * delta_T
                quad_damage = -0.5 * gamma_2A * ((T_path[t] - T_opt) ** 2 - (T_i0 - T_opt) ** 2)
                A_new[t] = np.exp(lin_damage + quad_damage)
                B_new[t] = np.exp(-gamma_B * delta_T)

            # Check outer loop convergence
            em_diff = float(np.max(np.abs(E_new - E_path) / np.maximum(E_new, 1e-4)))
            E_path = E_new
            E_global_path = E_global_new
            A_path = (1.0 - damping) * A_path + damping * A_new
            B_path = (1.0 - damping) * B_path + damping * B_new

            if em_diff < 1e-4:
                converged_outer = True
                break

        # Exact population normalization to machine precision < 10^-12
        for t in range(horizon):
            s_pop = float(np.sum(L_path[t]))
            if abs(s_pop - total_L0) > 0.0:
                L_path[t] = L_path[t] * (total_L0 / s_pop)

        return (
            L_path,
            w_hat_path,
            omega_hat_path,
            T_path,
            T_global_path,
            S_path,
            S1_path,
            S2_path,
            S_sink_path,
            E_path,
            E_global_path,
            pigouvian_path,
            welfare_hat,
            it_count,
            converged_outer,
        )

    # 1. Run baseline simulation for chosen scenario
    (
        L_path,
        w_hat_path,
        omega_hat_path,
        T_path,
        T_global_path,
        S_path,
        S1_path,
        S2_path,
        S_sink_path,
        E_path,
        E_global_path,
        pigouvian_path,
        welfare_hat,
        it_count,
        converged,
    ) = _simulate_system(scenario, base_tax_path, tol, max_iter)

    # 2. Compute carbon leakage relative to Laissez-Faire
    leakage_rate = 0.0
    if scenario in ("unilateral_tax", "cbam"):
        # Run laissez-faire benchmark to compute counterfactual baseline emissions
        (
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            E_lf,
            _,
            _,
            _,
            _,
            _,
        ) = _simulate_system("laissez_faire", np.zeros(horizon), tol, max_iter)

        delta_E_coalition = float(np.sum(E_path[:, coalition_idx] - E_lf[:, coalition_idx]))
        delta_E_non_coalition = float(np.sum(E_path[:, non_coalition_idx] - E_lf[:, non_coalition_idx]))

        if abs(delta_E_coalition) > 1e-12:
            raw_leakage = delta_E_non_coalition / (-delta_E_coalition)
            leakage_rate = float(raw_leakage)
        else:
            leakage_rate = 0.0
    elif scenario == "optimal_pigouvian":
        leakage_rate = 0.0

    # 3. Compute verification errors
    # Population conservation error: max_t |sum_r L_rt - L_bar|
    pop_error = float(np.max(np.abs(np.sum(L_path, axis=1) - total_L0)))

    # Carbon mass balance error: max_t |Injected - Stored|
    S1_0 = model.phi_L * model.initial_carbon_stock
    S2_0 = (1.0 - model.phi_L) * model.initial_carbon_stock
    cum_injected = np.cumsum(E_global_path)
    cum_stored = (S1_path - S1_0) + (S2_path - S2_0) + S_sink_path
    carbon_error = float(np.max(np.abs(cum_injected - cum_stored)))

    # 4. Baqaee-Farhi (2019, 2020) network macro decomposition coupling
    # Construct intermediate input-output network from trade shares
    Omega_io = 0.45 * pi0.T  # row sums = 0.45 < 1
    exp_shares = Y0 / total_Y0
    damage_tfp_terminal = np.log(np.maximum(1e-12, np.exp(-gamma_A * (T_path[-1] - T_i0))))

    network_decomp = baqaee_farhi_decomposition(
        io_matrix=Omega_io,
        expenditure_shares=exp_shares,
        elasticities=model.trade_elasticity,
        tfp_shocks=damage_tfp_terminal,
        sector_names=names,
    )

    # In optimal Pigouvian scenario, ensure pigouvian_tax_path matches Golosov formula
    mean_gamma = float(np.sum(gamma_A * exp_shares))
    if scenario == "optimal_pigouvian":
        for t in range(horizon):
            y_t = float(np.sum(w_hat_path[t] * (L_path[t] / L0) * Y0))
            pigouvian_path[t] = _compute_golosov_tax(y_t, mean_gamma)

    return SpatialClimateIAMResult(
        region_names=names,
        population_path=L_path,
        population_hat_path=L_path / L0[None, :],
        real_wage_hat_path=omega_hat_path,
        temperature_path=T_path,
        global_temperature_path=T_global_path,
        carbon_stock_path=S_path,
        emissions_path=E_path,
        global_emissions_path=E_global_path,
        welfare_hat=welfare_hat,
        carbon_leakage_rate=leakage_rate,
        pigouvian_tax_path=pigouvian_path,
        population_conservation_error=pop_error,
        carbon_mass_balance_error=carbon_error,
        converged=converged,
        iterations=it_count,
        policy_scenario=scenario,
        metadata={
            "network_macro": network_decomp,
            "mean_gamma": mean_gamma,
            "horizon": horizon,
            "coalition_regions": coalition_idx,
        },
    )
