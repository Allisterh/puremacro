"""Modern diagnostics: frozen independent ArviZ outputs and adverse chains."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from puremacro.mcmc import mcse_mean, multichain_ess, rank_normalized_rhat

FIXTURES = Path(__file__).parent / "fixtures" / "mcmc_rank"


def test_frozen_independent_arviz_diagnostics():
    record = json.loads((FIXTURES / "reference.json").read_text())
    data = FIXTURES / "draws.npz"
    assert hashlib.sha256(data.read_bytes()).hexdigest() == record["draws_sha256"]
    assert record["reference_version"] == "0.22.0"
    with np.load(data, allow_pickle=False) as arrays:
        assert set(arrays) == set(record["results"])
        for name, expected in record["results"].items():
            x = arrays[name]
            if expected["rhat"] is None:
                assert np.isnan(rank_normalized_rhat(x))
            else:
                assert rank_normalized_rhat(x) == pytest.approx(expected["rhat"], rel=2e-12)
            for method in ("bulk", "tail", "mean"):
                if name == "ties" and method == "tail":
                    # The 95% quantile is the maximum category. ArviZ assigns
                    # its constant indicator nominal ESS; Puremacro explicitly
                    # marks the unavailable upper-tail evidence instead.
                    assert np.all(x <= np.quantile(x, .95))
                    assert np.isnan(multichain_ess(x, method=method))
                    continue
                assert multichain_ess(x, method=method) == pytest.approx(expected[method], rel=2e-11)
            assert mcse_mean(x) == pytest.approx(expected["mcse_mean"], rel=2e-11)


@pytest.mark.pyodide_smoke
def test_between_chain_disagreement_reduces_ess():
    rng = np.random.default_rng(109)
    x = rng.normal(size=(4, 1000))
    stuck = x + np.array([-8, -3, 3, 8])[:, None]
    assert rank_normalized_rhat(x) < 1.01
    assert rank_normalized_rhat(stuck) > 1.7
    assert multichain_ess(stuck) < 10
    assert multichain_ess(x) > 2000


def test_folded_rhat_detects_scale_disagreement():
    rng = np.random.default_rng(111)
    x = rng.normal(size=(4, 2000))
    x[0] *= 10
    assert rank_normalized_rhat(x) > 1.1


@pytest.mark.parametrize("x", [np.ones((4, 100)), np.zeros((2, 4)),
                               np.ones((2, 3)), np.empty((0, 10)),
                               np.full((2, 10), np.nan), np.full((2, 10), np.inf)])
def test_uninformative_input_does_not_look_converged(x):
    assert np.isnan(rank_normalized_rhat(x))
    for method in ("bulk", "tail", "mean"):
        assert np.isnan(multichain_ess(x, method=method))
    assert np.isnan(mcse_mean(x))


def test_different_constant_chains_have_infinite_rhat():
    x = np.repeat([[0], [1]], 100, axis=1)
    assert np.isinf(rank_normalized_rhat(x))
    assert multichain_ess(x) < 5


def test_rank_diagnostics_are_invariant_to_monotone_transform():
    rng = np.random.default_rng(99)
    x = rng.normal(size=(3, 401))
    # Bulk rank normalization is invariant to monotone transformations; folding
    # concerns scale and is intentionally not covered by this statement.
    assert multichain_ess(np.exp(x)) == pytest.approx(multichain_ess(x), rel=1e-13)


def test_odd_draw_middle_is_excluded_from_split_bulk():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(3, 101))
    y = x.copy()
    y[:, 50] = 1e50
    assert rank_normalized_rhat(x) == rank_normalized_rhat(y)
    assert multichain_ess(x) == multichain_ess(y)


def test_invalid_shapes_and_method():
    for fn in (rank_normalized_rhat, multichain_ess, mcse_mean):
        with pytest.raises(ValueError, match="shape"):
            fn(np.ones((2, 10, 2)))
    with pytest.raises(ValueError, match="method"):
        multichain_ess(np.ones((2, 10)), method="other")
    with pytest.raises(ValueError, match="tail_probability"):
        multichain_ess(np.ones((2, 10)), tail_probability=.5)


def test_mean_diagnostics_are_stable_under_extreme_units():
    rng = np.random.default_rng(45)
    x = rng.normal(size=(2, 100))
    for scale in (1e-200, 1e200):
        assert multichain_ess(scale*x, method="mean") == pytest.approx(multichain_ess(x, method="mean"))
        assert mcse_mean(scale*x)/scale == pytest.approx(mcse_mean(x))


def test_mean_diagnostics_are_translation_invariant_for_represented_draws():
    x = np.random.default_rng(71).normal(size=(4, 1000)) + 1e15
    # Subtraction removes only the common offset; this preserves the exact
    # representable differences, including quantization from forming x.
    centered = x - x.flat[0]
    assert multichain_ess(x, method="mean") == pytest.approx(multichain_ess(centered, method="mean"), rel=1e-13)
    assert mcse_mean(x) == pytest.approx(mcse_mean(centered), rel=1e-13)
    from puremacro.dsge.nuts import compute_ebfmi
    assert compute_ebfmi(x[0]) == pytest.approx(compute_ebfmi(centered[0]), rel=1e-13)


def test_centering_opposite_extremes_does_not_overflow():
    from puremacro.dsge.nuts import compute_ebfmi
    x = np.array([[-1.7, 1.6, -.8, 1.2]*30, [1.5, -1.6, .9, -.7]*30])
    extreme = x*1e308
    with np.errstate(all="raise"):
        assert multichain_ess(extreme, method="mean") == pytest.approx(multichain_ess(x, method="mean"))
        assert mcse_mean(extreme)/1e308 == pytest.approx(mcse_mean(x))
        assert compute_ebfmi(extreme[0]) == pytest.approx(compute_ebfmi(x[0]))


def test_nuts_summary_uses_modern_diagnostics_without_changing_draws():
    from puremacro.dsge._results import NUTSResult
    rng = np.random.default_rng(11)
    x = rng.normal(size=(2, 100, 1)) + np.array([0, 8])[:, None, None]
    result = NUTSResult(x, ("beta",), np.zeros((2,100)), (.8,.8), (.1,.1),
                        np.ones((2,100)), np.zeros((2,100), dtype=bool),
                        rng.normal(size=(2,100)), np.ones(1))
    before = x.copy()
    row = result.summary().loc["beta"]
    assert row.r_hat == rank_normalized_rhat(x[:,:,0])
    assert row.ess_bulk == multichain_ess(x[:,:,0])
    assert row.mcse_mean == mcse_mean(x[:,:,0])
    np.testing.assert_array_equal(x, before)


def test_legacy_split_and_energy_helpers_do_not_report_degenerate_success():
    from puremacro.dsge.nuts import compute_split_rhat, compute_ebfmi
    assert np.isnan(compute_split_rhat(np.ones((2, 50))))
    assert np.isinf(compute_split_rhat(np.repeat([[0], [1]], 50, axis=1)))
    assert np.isnan(compute_split_rhat(np.ones((2, 3))))
    assert np.isnan(compute_ebfmi(np.ones(50)))
    assert np.isnan(compute_ebfmi(np.array([np.nan, 1.])))
    e = np.random.default_rng(17).normal(size=100)
    assert compute_ebfmi(e*1e-100) == pytest.approx(compute_ebfmi(e))
