"""Master test suite for puremacro Python package features."""
import pytest
import numpy as np

from puremacro import dp
from puremacro.vfi.discretize import tauchen
from puremacro.vfi.solve import solve_vfi_howard
from puremacro.vfi.hjb_achdou import solve_hjb_achdou
from puremacro.vfi.aiyagari_endogenous import solve_aiyagari_endogenous
from puremacro.did.callaway_santanna import callaway_santanna
from puremacro.causal.synthetic_control import synthetic_control
from puremacro.connectedness.bk2018 import barunik_krehlik


def test_dynare_vfi_model():
    """Notebook 21's precautionary-savings model, declared with dp (vfi.Model's successor)."""
    m = dp.Model("Precautionary Savings Model")
    m.parameters(beta=0.96, gamma=2.0, alpha=0.36, delta=0.10)
    m.exogenous("z", dp.AR1(rho=0.90, sigma=0.15, n=5))
    m.state("a", np.linspace(0.1, 15.0, 60))
    m.local("c = exp(z)*a^alpha + (1 - delta)*a - a(+1)")
    m.reward("c^(1 - gamma)/(1 - gamma)")
    m.subject_to("c > 0")
    sol = m.solve(method="vfi", tol=1e-6)

    assert 0.05 < sol.gini("a") < 0.60
    assert sol.mean("a") == pytest.approx(6.5733, abs=1e-3)

    # the same problem solved directly by the vfi engine gives the same policy
    a_grid = np.linspace(0.1, 15.0, 60)
    z_grid, P = tauchen(5, 0.90, 0.15)

    def ret(a, ap, z):
        c = np.exp(z) * a ** 0.36 + 0.90 * a - ap
        return -1e10 if c <= 0 else c ** (1.0 - 2.0) / (1.0 - 2.0)

    direct = solve_vfi_howard(ret, a_grid, z_grid, P, 0.96, howard_steps=15, tol=1e-6)
    assert np.array_equal(sol.raw.policy_aprime, direct["policy_idx"])


def test_hjb_achdou():
    res = solve_hjb_achdou(Na=50)
    assert res["V"].shape == (50, 2)
    assert res["n_iter"] <= 500


def test_aiyagari_endogenous():
    res = solve_aiyagari_endogenous(Na=30, Ne=3)
    assert res["K_star"] > 0
    assert res["L_star"] > 0


def test_callaway_santanna():
    np.random.seed(42)
    N_units = 30
    T_periods = 5
    group_assignment = np.array([np.inf]*10 + [3]*10 + [4]*10)

    y_vec, g_vec, t_vec, u_vec = [], [], [], []
    for u in range(N_units):
        g_val = group_assignment[u]
        for t in range(1, T_periods + 1):
            y_val = 2.0 + (t >= g_val) * 1.5 + np.random.randn() * 0.2
            y_vec.append(y_val)
            g_vec.append(g_val)
            t_vec.append(t)
            u_vec.append(u)

    res = callaway_santanna(y_vec, g_vec, t_vec, u_vec)
    assert not np.isnan(res["overall_att"])


def test_synthetic_control():
    np.random.seed(44)
    Y_panel = 10.0 + np.random.randn(40, 8)
    Y_panel[20:, 0] += 3.0

    res = synthetic_control(Y_panel, 0, 20)
    assert len(res["weights"]) == 7
    assert res["att_post"] > 1.0


def test_barunik_krehlik():
    np.random.seed(55)
    ret = np.random.randn(100, 3)
    res = barunik_krehlik(ret, var_lags=2, fevd_horizon=50)
    assert 0 <= res["total_spillover"] <= 100
