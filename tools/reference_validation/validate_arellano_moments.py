"""Check moment definitions by executing the author's MATLAB block in Octave.

The same Python-generated path and 100 eligible windows are supplied to both
implementations. This checks aggregation, not random-number generator parity.
The archived formula block is extracted verbatim except for its episode range.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from scipy.io import savemat

from tools.reference_validation.arellano_2008 import (
    FIXTURE, GROSS_R, eligible_windows, load_original, simulate, summarize_path,
)


def run(output: Path) -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    octave = shutil.which("octave")
    if octave is None:
        raise RuntimeError("Octave is required only to regenerate this independent check")
    path = simulate(load_original(), seed=1979)
    ends = eligible_windows(path)[:100]
    savemat(output / "path.mat", dict(css=path["consumption"][:, None], yss=path["output"][:, None],
                bss=path["assets"][:, None], pss=path["price"][:, None],
                tbss=((path["output"]-path["consumption"])/path["output"])[:, None],
                rfss=np.full((len(path["price"]), 1), GROSS_R),
                countdef=path["entry"].astype(float)[:, None],
                defepirows=(ends+1).astype(float)[:, None], defepi=float(len(ends)),
                endtime=float(len(path["price"])), ndd=np.array([path["excluded"].sum()], dtype=float)))
    source = (FIXTURE / "original/policies_simulation.m").read_text()
    block = source[source.index("n=74;"):source.index("%%  Simulation Default Event")]
    block = block.replace("for k=2:300", "for k=1:defepi")
    names = ["debt_pct", "spread_mean_pct", "spread_std_pp", "tb_std_pp", "consumption_std_pct",
             "output_std_pct", "corr_spread_y", "corr_tb_y", "corr_c_y", "corr_tb_spread",
             "corr_c_spread", "annual_default_pct"]
    footer = """
moments = [-100*mean(bmean_def./ymean_def), 100*mean(spmean), 100*mean(spstd), ...
100*mean(tbstd), 100*mean(cstd), 100*mean(ystd), mean(ycorr(:,6)), ...
mean(ycorr(:,4)), mean(ycorr(:,2)), mean(pcorr(:,4)), mean(pcorr(:,2)), ...
100*(1-(1-sum(countdef)/(endtime-ndd(1)))^4)];
dlmwrite('moments.csv', moments, 'precision', '%.17g');
"""
    (output / "oracle.m").write_text("load('path.mat');\n"+block+footer)
    with (output / "octave.log").open("w") as log:
        subprocess.run([octave, "--no-gui", "--quiet", "oracle.m"], cwd=output, check=True,
                       stdout=log, stderr=subprocess.STDOUT)
    external = dict(zip(names, map(float, np.loadtxt(output / "moments.csv", delimiter=","))))
    actual = summarize_path(path)["moments"]
    errors = {key: abs(external[key]-actual[key]) for key in names}
    record = dict(scope="Author's MATLAB aggregation block on identical path/windows, executed in Octave",
                  source_sha256=hashlib.sha256((FIXTURE / "original/policies_simulation.m").read_bytes()).hexdigest(),
                  runtime=subprocess.check_output([octave, "--version"], text=True, stderr=subprocess.DEVNULL).splitlines()[0],
                  seed=1979, periods=100000, burn=1000, windows=100, window_quarters=74,
                  adaptations=["Path supplied as MAT variables instead of original file loaders",
                               "Loop over 100 supplied eligible windows instead of candidates 2:300",
                               "Use the Fortran gross risk-free rate (original MATLAB hardcodes 1.017)",
                               "Use a consistent post-burn denominator and default-entry count",
                               "Export moments as CSV"],
                  moments=external, max_abs_errors=errors, passed=max(errors.values()) < 1e-10)
    (output / "moments_reference.json").write_text(json.dumps(record, indent=2)+"\n")
    if not record["passed"]:
        raise RuntimeError(f"Moment comparison failed: {errors}")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))
