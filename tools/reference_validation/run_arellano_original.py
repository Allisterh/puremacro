"""Run Arellano's archived Fortran with a documented portability patch.

Requires gfortran and LAPACK (Accelerate on macOS); neither is a runtime
dependency of PureMacro. Economic equations, loops and literal precision are
unchanged. The original BSD-licensed sources and input hashes are retained.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tests/fixtures/arellano_2008/original"

LAPACK_SHIM = """subroutine dlinrg(n, a, lda, ainv, ldinv)
  implicit none
  integer, intent(in) :: n, lda, ldinv
  real(8), intent(in) :: a(lda,n)
  real(8), intent(out) :: ainv(ldinv,n)
  real(8) :: work(n,n)
  integer :: piv(n), info, i
  work = a(1:n,1:n)
  ainv = 0.0d0
  do i=1,n
    ainv(i,i) = 1.0d0
  end do
  call dgesv(n,n,work,n,piv,ainv,ldinv,info)
  if (info /= 0) stop 2
end subroutine dlinrg
"""


def run(output: Path, source: Path = SOURCE) -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    compiler = shutil.which("gfortran")
    if compiler is None:
        raise RuntimeError("gfortran is required to regenerate the external reference")
    manifest = json.loads((SOURCE.parent / "source_manifest.json").read_text())
    for name, digest in manifest["files_sha256"].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Archived source hash mismatch: {name}")
    patches = []
    for name in ("parameters.f90", "global.f90", "process.f90", "main.f90"):
        original = (source/name).read_text()
        adapted = original.replace("C:\\Arellano\\default\\", "")
        adapted = adapted.replace("include 'link_f90_dll.h'", "! IMSL inverse replaced by LAPACK DGESV in lapack_shim.f90")
        if name == "parameters.f90":
            adapted = adapted.replace("kkk=1", "kkk=0")  # Missing warm-start files.
        if name == "main.f90":
            adapted = adapted.replace("v(izero,i_ss)", "v(int(izero),i_ss)")
            adapted = adapted.replace("!! Now we have all the solutions", """open(unit=97,file='convergence.txt',status='replace')
write(97,*) kk, ite, gap
close(97)
!! Now we have all the solutions""")
        (output/name).write_text(adapted)
        patches.extend(difflib.unified_diff(original.splitlines(True), adapted.splitlines(True),
                                           fromfile="original/"+name, tofile="portable/"+name))
    (output/"portability.patch").write_text("".join(patches))
    (output/"lapack_shim.f90").write_text(LAPACK_SHIM)
    for name in ("YMAT.dat", "PIMAT.dat"):
        shutil.copyfile(source/name, output/name)
    command = [compiler, "-O3", "-std=legacy", "parameters.f90", "global.f90", "process.f90",
               "main.f90", "lapack_shim.f90", "-o", "arellano_original"]
    command += ["-framework", "Accelerate"] if platform.system() == "Darwin" else ["-llapack", "-lblas"]
    started = time.monotonic()
    with (output/"compile.log").open("w") as log:
        subprocess.run(command, cwd=output, stdout=log, stderr=subprocess.STDOUT, check=True)
    with (output/"solve.log").open("w") as log:
        subprocess.run([str(output/"arellano_original")], cwd=output, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    outer, inner, price_error = np.loadtxt(output/"convergence.txt")
    if outer > 500 or price_error >= 1e-8:
        raise RuntimeError(f"Original Fortran did not converge: {outer=}, {price_error=}")
    arrays = {}
    for name in ("v", "sav", "savnd", "p", "psav", "psavnd", "def", "cons", "consnd", "yy", "dprob", "i_s_star"):
        arrays[name] = np.fromstring((output/(name+".f90")).read_text(), sep=" ").reshape((200,21), order="F")
    arrays["b"] = np.fromstring((output/"b.f90").read_text(), sep=" ")
    arrays["conaut"] = np.fromstring((output/"conaut.f90").read_text(), sep=" ")
    arrays["zero_index"] = np.array(int(float((output/"izero.f90").read_text()))-1)
    np.savez_compressed(output/"reference.npz", **arrays)
    record = dict(source_doi=manifest["archive_doi"], source_hashes=manifest["files_sha256"],
                  compiler=subprocess.check_output([compiler,"--version"],text=True).splitlines()[0],
                  command=command, elapsed_seconds=time.monotonic()-started,
                  outer_iterations=int(outer), inner_iterations=int(inner), price_error=float(price_error),
                  modifications=["Local paths", "Cold start because archived guesses are absent",
                                 "Explicit integer conversion of zero index", "IMSL inverse replaced by LAPACK solve",
                                 "Write convergence diagnostic"],
                  literal_precision="Original unsuffixed Fortran literals retained; no default-real-8 flag",
                  reference_sha256=hashlib.sha256((output/"reference.npz").read_bytes()).hexdigest())
    (output/"manifest.json").write_text(json.dumps(record,indent=2)+"\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))
