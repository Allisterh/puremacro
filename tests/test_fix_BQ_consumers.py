"""Galí (1999) difference specification in the BQ consumers.

``puremacro/examples/empirical_benchmark_replications.py`` and showcase
notebook 35 (EN and ES) estimate a bivariate long-run (Blanchard-Quah) SVAR on
the bundled ``load_gali1999`` data. ``bq_svar`` cumulates every response row by
default, which is right only when every column is a first difference. Hours
per capita are stored as a *level*, so feeding them in levels turned the hours
response into a running sum (+12.2 at h=20).

The consumers now follow Galí's own specification: labour productivity growth
and the FIRST DIFFERENCE of log hours, both cumulated to levels. Galí,
"Technology, Employment, and the Business Cycle: Do Technology Shocks Explain
Aggregate Fluctuations?", NBER Working Paper 5721 (August 1996; published in
AER 89(1), 1999, 249-271),
https://www.nber.org/system/files/working_papers/w5721/w5721.pdf:

* PDF p. 17 (printed p. 16): {(x_t, n_t)} is I(1) "with a stationary VAR
  representation in first differences"; C(1) is lower triangular and is "the
  Choleski factor of E(1) Sigma E(1)'".
* PDF p. 20 (printed p. 19): after a positive technology shock labour
  productivity rises and there is a "short-lived, though persistent, decline
  in employment" (hours fall on impact).

The oracle below re-estimates the VAR by OLS with numpy, forms the long-run
multiplier A(1)^{-1} = (I - A_1 - ... - A_p)^{-1}, takes the numpy Cholesky
factor of A(1)^{-1} Sigma A(1)^{-1}', and builds the MA(infinity) responses by
the companion recursion. No puremacro estimation code is used.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from puremacro.datasets import load_gali1999

PROJ_ROOT = Path(__file__).resolve().parents[1]
NB_DIR = PROJ_ROOT / "notebooks"
STEM = "35_empirical_benchmark_replications"
NB_SOURCES = [
    NB_DIR / f"{STEM}.py",
    NB_DIR / f"{STEM}_es.py",
    NB_DIR / f"{STEM}.ipynb",
    NB_DIR / f"{STEM}_es.ipynb",
]
P, H = 4, 20


# --------------------------------------------------------------------------
# Independent oracle
# --------------------------------------------------------------------------
def _oracle_difference_spec():
    df = load_gali1999()
    dprod = df["dlprod"].to_numpy(dtype=float)
    hours = df["hours"].to_numpy(dtype=float)
    # Galí's specification: [productivity growth, first difference of hours].
    Y = np.column_stack([dprod[1:], np.diff(hours)])
    T, n = Y.shape
    X = np.column_stack(
        [np.ones(T - P)] + [Y[P - l - 1:T - l - 1] for l in range(P)]
    )
    coef, *_ = np.linalg.lstsq(X, Y[P:], rcond=None)
    resid = Y[P:] - X @ coef
    # OLS covariance with the usual degrees-of-freedom correction
    # (T - p observations, 1 + n p regressors per equation).
    Sigma = resid.T @ resid / ((T - P) - (1 + n * P))
    A = [coef[1 + l * n:1 + (l + 1) * n].T for l in range(P)]

    lr = np.linalg.inv(np.eye(n) - sum(A))          # A(1)^{-1}
    C1 = np.linalg.cholesky(lr @ Sigma @ lr.T)       # long-run impact, lower-tri
    B = np.linalg.solve(lr, C1)                      # impact matrix

    Psi = [np.eye(n)]
    for h in range(1, H + 1):
        Psi.append(sum(A[l] @ Psi[h - l - 1] for l in range(min(h, P))))
    raw = np.stack([Psi_h @ B for Psi_h in Psi])     # (H+1, n, n)
    return {"Y": Y, "raw": raw, "levels": np.cumsum(raw, axis=0), "lr": lr, "B": B, "Sigma": Sigma}


@pytest.fixture(scope="module")
def oracle():
    return _oracle_difference_spec()


@pytest.fixture(scope="module")
def example_result():
    from puremacro.examples.empirical_benchmark_replications import gali_difference_spec

    return gali_difference_spec(n_boot=0)


# --------------------------------------------------------------------------
# Oracle sanity (identification restrictions of the paper)
# --------------------------------------------------------------------------
def test_oracle_satisfies_long_run_restriction(oracle):
    C1 = oracle["lr"] @ oracle["B"]
    assert abs(C1[0, 1]) < 1e-10           # non-tech shock: no long-run productivity effect
    assert C1[0, 0] > 0                    # sign normalisation: tech shock raises productivity
    np.testing.assert_allclose(oracle["B"] @ oracle["B"].T, oracle["Sigma"], atol=1e-12)


def test_oracle_reproduces_galis_qualitative_result(oracle):
    lv = oracle["levels"]
    assert lv[0, 0, 0] > 0                 # productivity up on impact
    assert lv[0, 1, 0] < 0                 # hours fall on impact (Galí, printed p. 19)


# --------------------------------------------------------------------------
# Example module
# --------------------------------------------------------------------------
def test_example_uses_first_difference_of_hours(example_result, oracle):
    Z, _ = example_result
    np.testing.assert_allclose(Z, oracle["Y"], rtol=0, atol=1e-12)


def test_example_irf_matches_hand_computed_bq(example_result, oracle):
    _, res = example_result
    np.testing.assert_allclose(res.irf_point, oracle["levels"], rtol=0, atol=1e-9)


def test_example_hours_level_is_not_a_running_sum(example_result):
    """Regression guard: the levels-input + default-cumulate bug gave +12.2."""
    _, res = example_result
    assert abs(res.irf_point[H, 1, 0]) < 1.0
    assert res.irf_point[0, 1, 0] < 0


def test_example_bands_are_ordered():
    from puremacro.examples.empirical_benchmark_replications import gali_difference_spec

    _, res = gali_difference_spec(n_boot=50, seed=0)
    assert np.all(res.irf_lower <= res.irf_upper + 1e-12)


# --------------------------------------------------------------------------
# Notebook 35 (EN + ES): source and stored outputs
# --------------------------------------------------------------------------
def _cells(path: Path) -> list[dict]:
    if path.suffix == ".ipynb":
        return json.loads(path.read_text(encoding="utf-8"))["cells"]
    return [{"cell_type": "code", "source": path.read_text(encoding="utf-8")}]


def _text(path: Path) -> str:
    return "\n".join("".join(c["source"]) for c in _cells(path))


@pytest.mark.parametrize("path", NB_SOURCES, ids=lambda p: p.name)
def test_notebook_uses_difference_spec(path):
    src = _text(path)
    assert 'df_gali["hours"].diff()' in src
    assert 'df_gali[["dlprod", "hours"]].to_numpy' not in src


STALE_CLAIMS = [
    r"-0.4\%$ on impact",
    "six quarters",
    "caen $-0.4\\%$",
    "seis trimestres",
    r"(\Delta x_t, n_t)",
]


@pytest.mark.parametrize("path", NB_SOURCES, ids=lambda p: p.name)
def test_notebook_prose_drops_stale_claims(path):
    src = _text(path)
    for claim in STALE_CLAIMS:
        assert claim not in src, f"{path.name} still says {claim!r}"


HOURS_LINE = re.compile(
    r"(?:Hours per capita|Horas per cápita)[^\n]*h=0:\s*([+-]\d+\.\d+)[^\n]*h=20:\s*([+-]\d+\.\d+)"
)
PROD_LINE = re.compile(
    r"(?:Productivity \(level\)|Productividad \(nivel\))[^\n]*h=0:\s*([+-]\d+\.\d+)[^\n]*h=20:\s*([+-]\d+\.\d+)"
)


@pytest.mark.parametrize("path", NB_SOURCES[2:], ids=lambda p: p.name)
def test_notebook_stored_outputs_match_oracle(path, oracle):
    out = []
    for c in _cells(path):
        for o in c.get("outputs", []):
            if o.get("output_type") == "stream":
                out.append("".join(o.get("text", "")))
    text = "\n".join(out)
    m_h, m_p = HOURS_LINE.search(text), PROD_LINE.search(text)
    assert m_h and m_p, f"{path.name}: stored IRF summary lines not found; re-execute it"
    lv = oracle["levels"]
    assert float(m_h.group(1)) == pytest.approx(lv[0, 1, 0], abs=6e-4)
    assert float(m_h.group(2)) == pytest.approx(lv[H, 1, 0], abs=6e-4)
    assert float(m_p.group(1)) == pytest.approx(lv[0, 0, 0], abs=6e-4)
    assert float(m_p.group(2)) == pytest.approx(lv[H, 0, 0], abs=6e-4)


def test_notebook_en_es_same_cell_structure():
    en = _cells(NB_DIR / f"{STEM}.ipynb")
    es = _cells(NB_DIR / f"{STEM}_es.ipynb")
    assert [c["cell_type"] for c in en] == [c["cell_type"] for c in es]


# --------------------------------------------------------------------------
# The example script end to end
# --------------------------------------------------------------------------
EXAMPLE_HOURS_LINE = re.compile(
    r"Hours per capita \(level\) to a technology shock: "
    r"h=0: ([+-]\d+\.\d{3})  h=20: ([+-]\d+\.\d{3})$",
    re.MULTILINE,
)


def test_example_main_prints_difference_spec_hours(monkeypatch, tmp_path, capsys, oracle):
    """``main()`` runs headless, writes its figure (redirected to ``tmp_path``
    so the tracked PNG is untouched) and prints the difference-spec hours
    line: about -0.834 on impact and -0.402 at h=20, to 3 decimals."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from puremacro.examples import empirical_benchmark_replications as ex

    real_savefig = plt.savefig
    saved = []

    def savefig_to_tmp(fname, *args, **kwargs):
        target = tmp_path / Path(fname).name
        saved.append(target)
        return real_savefig(target, *args, **kwargs)

    monkeypatch.setattr(ex.plt, "savefig", savefig_to_tmp)
    ex.main()
    out = capsys.readouterr().out

    assert len(saved) == 1 and saved[0].exists() and saved[0].stat().st_size > 0
    m = EXAMPLE_HOURS_LINE.search(out)
    assert m, "difference-spec hours line not printed"
    h0, h20 = float(m.group(1)), float(m.group(2))
    assert h0 == pytest.approx(-0.834, abs=1.5e-3)
    assert h20 == pytest.approx(-0.402, abs=1.5e-3)
    lv = oracle["levels"]
    assert h0 == pytest.approx(lv[0, 1, 0], abs=6e-4)
    assert h20 == pytest.approx(lv[H, 1, 0], abs=6e-4)
    # The running-sum bug printed +12.2 at h=20.
    assert h20 < 0
