"""Small solved dp models shared by the dp test modules (module-scoped)."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from puremacro import dp

LC_T = 10
LC_SURVIVAL = 0.9


def _household(a_grid, name="household"):
    m = dp.Model(name)
    m.parameters(beta=0.96, gamma=1.0, alpha=0.36, delta=0.08, r=0.03)
    m.exogenous("z", dp.AR1(rho=0.9, sigma=0.2, n=5))
    m.state("a", a_grid)
    m.local("w = (1 - alpha)*(alpha/(r + delta))^(alpha/(1 - alpha))")
    m.local("c = w*exp(z) + (1 + r)*a - a(+1)")
    m.reward("crra(c, gamma)")
    m.subject_to("c > 0")
    return m


@pytest.fixture(scope="module")
def aiyagari_sol():
    """Aiyagari household at a fixed interest rate, solved by EGM."""
    return _household(np.linspace(1e-4, 60.0, 120)).solve("egm", tol=1e-10)


@pytest.fixture(scope="module")
def borrowing_sol():
    """The same household with a borrowing limit of -2, so wealth can be negative."""
    return _household(np.linspace(-2.0, 60.0, 120), "borrowing").solve("egm", tol=1e-10)


@pytest.fixture(scope="module")
def lc_sol():
    """Ten-period life cycle with a constant survival probability of 0.9."""
    m = dp.Model("life cycle")
    m.parameters(beta=0.96, gamma=2.0, r=0.03, T_ret=7)
    m.exogenous("z", dp.AR1(rho=0.95, sigma=0.1, n=3))
    m.state("a", np.linspace(0.0, 10.0, 40))
    m.local("y = (age < T_ret)*exp(z + 0.04*age) + (age >= T_ret)*0.6")
    m.local("c = (1 + r)*a + y - a(+1)")
    m.reward("crra(c, gamma)")
    m.subject_to("c > 0")
    m.horizon(LC_T, survival=np.full(LC_T, LC_SURVIVAL))
    return m.solve("egm")


@pytest.fixture(scope="module")
def hjb_sol():
    """Continuous-time household with a two-state income jump."""
    m = dp.Model("ct household")
    m.parameters(rho=0.05, gamma=2.0, r=0.03, w=1.0)
    m.exogenous("e", dp.Jump([0.2, 1.0], np.array([[-0.1, 0.1], [0.1, -0.1]])))
    m.state("a", np.linspace(0.0, 30.0, 100))
    m.control("c")
    m.drift("a", "r*a + w*e - c")
    m.reward("crra(c, gamma)")
    return m.solve()


@pytest.fixture(scope="module")
def ks_sol():
    """A tiny Krusell-Smith solve (one outer iteration); only its type matters."""
    m = dp.Model("krusell-smith")
    m.parameters(beta=0.99, gamma=1.0, alpha=0.36, delta=0.025, L=0.9)
    m.exogenous("e", dp.Markov([0.5, 1.1], np.array([[0.6, 0.4], [0.2, 0.8]])))
    m.aggregate_shock("Z", dp.Markov([0.99, 1.01], np.array([[0.875, 0.125], [0.125, 0.875]])))
    m.state("a", np.linspace(0.0, 60.0, 30))
    m.aggregate_state("K", np.linspace(8.0, 14.0, 3), mean="a")
    m.local("r = alpha*Z*(K/L)^(alpha - 1) - delta", "w = (1 - alpha)*Z*(K/L)^alpha")
    m.local("c = (1 + r)*a + w*e - a(+1)")
    m.reward("crra(c, gamma)")
    m.subject_to("c > 0")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return m.solve("egm", ks=dict(T=100, burn_in=10, max_outer=1))
