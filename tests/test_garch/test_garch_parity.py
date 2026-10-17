import numpy as np
import pandas as pd
import pytest

from puremacro.garch.fit import garch11_fit
from puremacro.garch._results import GARCH11Result


def _simulate_garch(rng, T, omega, alpha, beta):
    eps = np.zeros(T)
    sigma2 = np.zeros(T)
    sigma2[0] = omega / max(1e-8, 1 - alpha - beta)
    for t in range(1, T):
        sigma2[t] = omega + alpha * eps[t-1] ** 2 + beta * sigma2[t-1]
        eps[t] = np.sqrt(sigma2[t]) * rng.standard_normal()
    return pd.Series(eps, index=pd.date_range("1990-01-01", periods=T, freq="MS"))


def test_garch11_recovers_known_params():
    rng = np.random.default_rng(23)
    omega, alpha, beta = 0.05, 0.10, 0.85
    s = _simulate_garch(rng, T=2000, omega=omega, alpha=alpha, beta=beta)
    out = garch11_fit(s)
    # Verify documented attributes
    assert isinstance(out, GARCH11Result)
    for attr in ("omega", "alpha", "beta", "sigma", "loglik", "converged", "persistence"):
        assert hasattr(out, attr)
    assert out.converged
    # Persistence should be close to alpha + beta
    assert abs(out.persistence - (alpha + beta)) < 0.05
    # Sigma is a Series of length T with positive values
    assert len(out.sigma) == len(s)
    assert (out.sigma > 0).all()


def test_garch11_parity_with_arch():
    """Within 5pp of arch.arch_model — both fit Gaussian MLE so they
    converge to the same parameters on long enough data."""
    rng = np.random.default_rng(31)
    omega, alpha, beta = 0.04, 0.08, 0.88
    s = _simulate_garch(rng, T=3000, omega=omega, alpha=alpha, beta=beta)
    out = garch11_fit(s)
    from arch import arch_model
    am = arch_model(s.values, mean="Zero", vol="GARCH", p=1, q=1, rescale=False).fit(disp="off")
    # Tolerances reflect numerical optimizer differences
    assert abs(out.omega - am.params["omega"]) < 0.05
    assert abs(out.alpha - am.params["alpha[1]"]) < 0.05
    assert abs(out.beta - am.params["beta[1]"]) < 0.05


def test_garch11_result_is_frozen():
    rng = np.random.default_rng(0)
    s = _simulate_garch(rng, T=500, omega=0.05, alpha=0.10, beta=0.85)
    out = garch11_fit(s)
    assert isinstance(out, GARCH11Result)
    with pytest.raises(Exception):
        out.alpha = 99.0


def test_garch11_summary_smoke():
    rng = np.random.default_rng(0)
    s = _simulate_garch(rng, T=400, omega=0.05, alpha=0.10, beta=0.85)
    out = garch11_fit(s)
    txt = out.summary()
    assert "GARCH" in txt


def test_dcc_fit_returns_result_dataclass():
    from puremacro.garch.dcc import dcc_fit
    from puremacro.garch._results import DCCResult

    rng = np.random.default_rng(0)
    T = 600
    n = 3
    R = np.array([[1.0, 0.3, 0.1],
                  [0.3, 1.0, 0.2],
                  [0.1, 0.2, 1.0]])
    L = np.linalg.cholesky(R)
    rets = pd.DataFrame(
        (L @ rng.standard_normal((n, T))).T,
        columns=["A", "B", "C"],
        index=pd.date_range("1990-01-01", periods=T, freq="D"),
    )
    out = dcc_fit(rets)
    assert isinstance(out, DCCResult)
    assert 0.0 <= out.a <= 1.0
    assert 0.0 <= out.b <= 1.0
    assert out.R.shape == (T, n, n)
    assert out.H.shape == (T, n, n)
    assert out.sigma.shape == (T, n)
    with pytest.raises(Exception):
        out.a = 0.0


def _gaussian_loglik(y, omega, alpha, beta):
    """ℓ = -½ Σ_t [log 2π + log σ²_t + y²_t/σ²_t], σ²₀ = var(y).

    Written out from Bollerslev (1986) rather than imported, so it is an
    independent evaluation of what `garch11_fit` claims to report.
    """
    y = np.asarray(y, dtype=float)
    sigma2 = np.empty(len(y))
    sigma2[0] = float(np.var(y))
    for t in range(1, len(y)):
        sigma2[t] = omega + alpha * y[t - 1] ** 2 + beta * sigma2[t - 1]
    sigma2 = np.maximum(sigma2, 1e-10)
    return float(-0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + y ** 2 / sigma2))


def test_neg_loglik_is_scale_equivariant():
    """Truth anchor for the test below (holds before and after the fix).

    σ²₀ = var(eps) scales as c², so by induction on the recursion the whole
    σ² path does; eps²/σ² is then invariant and only Σ log σ² shifts. Hence
    _neg_loglik((c²ω, α, β), c·eps, c²v₀) = _neg_loglik((ω, α, β), eps, v₀)
    + T·log c exactly. An additive constant cannot move an argmax, which is
    why α and β below must be invariant to c and ω must scale by c².
    """
    from puremacro.garch.fit import _neg_loglik

    rng = np.random.default_rng(5)
    y = _simulate_garch(rng, T=1500, omega=0.05, alpha=0.08, beta=0.90).values
    v0 = float(np.var(y))
    base = _neg_loglik((0.05, 0.08, 0.90), y, v0)
    for c in (1e-3, 1e-2, 1e2, 1e3):
        got = _neg_loglik((0.05 * c * c, 0.08, 0.90), y * c, v0 * c * c)
        assert abs(got - base - len(y) * np.log(c)) < 1e-8


def test_garch11_is_scale_invariant():
    """α and β must not depend on the units of `returns`, and ω must scale
    with their square — see the equivariance identity above.

    Before the fix the likelihood was maximised on the raw series, so
    L-BFGS-B's *relative* ftol stopped it early once var(returns) left
    roughly [1e-2, 1e2]: the same T=2000 series gave α = 0.0786 at c = 1 but
    0.1033 at c = 1e-3 and 0.1169 at c = 1e2, all reporting converged=True,
    and the c = 1e2 point sat 12.5 log-likelihood units below the maximiser
    of the module's own objective. Decimal-unit daily returns (var ~ 1e-4)
    are the common case that tripped it.
    """
    rng = np.random.default_rng(5)
    s = _simulate_garch(rng, T=2000, omega=0.05, alpha=0.08, beta=0.90)
    base = garch11_fit(s)
    for c in (1e-3, 1e-2, 1e2, 1e4):
        r = garch11_fit(s * c)
        assert abs(r.alpha - base.alpha) < 1e-6, (c, r.alpha, base.alpha)
        assert abs(r.beta - base.beta) < 1e-6, (c, r.beta, base.beta)
        assert abs(r.omega / (c * c) / base.omega - 1.0) < 1e-5, (c, r.omega)


def test_garch11_loglik_is_evaluated_at_the_returned_params():
    """`GARCH11Result.loglik` is documented as the log-likelihood at the MLE,
    so it must equal ℓ recomputed at the ω/α/β the same object reports.

    It used to be `-res.fun`, the objective at whatever point L-BFGS-B
    evaluated last. After an ABNORMAL exit that is a rejected line-search
    trial, not `res.x`: on this stress DGP several of these seeds paired a
    strictly feasible (α + β < 0.999) parameter vector with the flat -1e10
    infeasibility sentinel as their reported log-likelihood.
    """
    for seed in range(8):
        rng = np.random.default_rng(seed)
        s = _simulate_garch(rng, T=3000, omega=0.01, alpha=0.06, beta=0.94)
        r = garch11_fit(s)
        assert r.alpha + r.beta < 0.999, seed  # res.x itself is feasible
        expect = _gaussian_loglik(s.values, r.omega, r.alpha, r.beta)
        assert abs(r.loglik - expect) < 1e-8 * max(1.0, abs(expect)), (seed, r.loglik, expect)


def test_garch11_rejects_non_finite_returns():
    """A NaN makes var0 NaN, `var0 <= 0` is False for NaN, and the likelihood
    is NaN at every parameter value, so the optimiser never left x0 and the
    hard-coded start came back as alpha=0.10, beta=0.85, persistence=0.95.
    No MLE exists for non-finite data, so this must raise.
    """
    rng = np.random.default_rng(7)
    s = _simulate_garch(rng, T=500, omega=0.05, alpha=0.10, beta=0.85)
    for bad in (np.nan, np.inf, -np.inf):
        for pos in (0, 250, 499):
            v = s.copy()
            v.iloc[pos] = bad
            with pytest.raises(ValueError, match="non-finite"):
                garch11_fit(v)
            with pytest.raises(ValueError, match="non-finite"):
                garch11_fit(v.values)


def test_garch11_rejects_unknown_mean_spelling():
    """Every spelling but the exact "constant" — including arch's own
    "Constant", and this module replaces arch.arch_model — used to fall
    through to the zero-mean fit with converged=True. `dcc_fit` validates the
    same kwarg the same way.
    """
    rng = np.random.default_rng(7)
    s = _simulate_garch(rng, T=500, omega=0.05, alpha=0.10, beta=0.85)
    for bad in ("Constant", "CONSTANT", "constant ", "const", "Zero", "mean"):
        with pytest.raises(ValueError, match="mean must be"):
            garch11_fit(s, mean=bad)
    for good in ("zero", "constant"):
        garch11_fit(s, mean=good)


def test_garch11_constant_mean_is_a_different_estimand():
    """Pins why the spelling guard matters: on a shifted series the zero-mean
    likelihood has to absorb mu² into the variance process, so it converges to
    a different point. Truth (ω, α, β) = (0.05, 0.08, 0.90), mu = 3.
    """
    rng = np.random.default_rng(1)
    s = _simulate_garch(rng, T=8000, omega=0.05, alpha=0.08, beta=0.90) + 3.0
    c = garch11_fit(s, mean="constant")
    z = garch11_fit(s, mean="zero")
    assert abs(c.alpha - 0.08) < 0.02
    assert abs(c.omega - 0.05) < 0.02
    assert abs(z.alpha - 0.08) > 0.03      # zero-mean crushes alpha
    assert z.omega > 4 * c.omega           # and inflates omega
