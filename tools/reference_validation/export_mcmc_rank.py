"""Freeze independently computed ArviZ diagnostics; not a runtime dependency.

Run with ArviZ 0.22.0 installed in a separate reference environment. This tool
does not import Puremacro or use its diagnostics to construct reference values.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
from pathlib import Path

import arviz as az
import numpy as np
import scipy


def cases():
    rng = np.random.default_rng(20261023)
    iid = rng.normal(size=(4, 401))
    shifted = iid + np.arange(4)[:, None]*2
    scaled = iid * np.array([.1, .4, 1, 4])[:, None]
    heavy = rng.standard_t(1, size=(4, 501))
    ties = rng.integers(0, 8, size=(3, 301)).astype(float)
    out = {"iid_odd": iid, "shifted": shifted, "different_scales": scaled,
           "cauchy": heavy, "ties": ties, "one_chain": iid[:1]}
    for phi in [.8, -.7]:
        draws = rng.normal(size=(4, 600))
        for t in range(1, draws.shape[1]):
            draws[:, t] += phi*draws[:, t-1]
        out["ar_positive" if phi > 0 else "ar_negative"] = draws[:, 100:]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if az.__version__ != "0.22.0":
        raise RuntimeError("reference exporter requires ArviZ 0.22.0")
    args.output.mkdir(parents=True, exist_ok=True)
    draws = cases()
    path = args.output / "draws.npz"
    np.savez_compressed(path, **draws)
    observed = {}
    for name, x in draws.items():
        observed[name] = {
            "rhat": float(az.rhat(x, method="rank")) if x.shape[0] >= 2 else None,
            "bulk": float(az.ess(x, method="bulk")),
            "tail": float(az.ess(x, method="tail")),
            "mean": float(az.ess(x, method="mean")),
            "mcse_mean": float(az.mcse(x, method="mean")),
        }
    source = Path(inspect.getfile(az.stats.diagnostics))
    record = {
        "reference": "ArviZ public rhat/ess/mcse, executed on recorded input arrays",
        "reference_version": az.__version__, "numpy": np.__version__,
        "scipy": scipy.__version__, "python": platform.python_version(),
        "diagnostics_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source": "https://github.com/arviz-devs/arviz/blob/v0.22.0/arviz/stats/diagnostics.py",
        "paper": "https://doi.org/10.1214/20-BA1221",
        "draws_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "scope": "Finite nondegenerate inputs; constant-tail/constant-draw policies tested separately",
        "results": observed,
    }
    (args.output / "reference.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    print(f"Recorded {len(observed)} independent cases with ArviZ {az.__version__}")


if __name__ == "__main__":
    main()
