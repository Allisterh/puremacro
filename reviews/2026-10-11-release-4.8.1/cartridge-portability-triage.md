# Cartridge verification across operating systems

The immutable `v4.8.0` tag points to
`29d187e95b0e56e5ed4f7a838f4d5f2b832a7d1d`. Its release workflow blocked PyPI
publication after Windows Python 3.12 reported two failures: loading/verifying
the original v1 cartridge fixture, and verifying that fixture before repacking
it with new metadata. The Windows log recorded 18,995 passed and two failed;
the failures occurred in `Cartridge.verify()`, after raw-file verification and
deserialization had succeeded. This evidence concerns the code defect; the
final 4.8.1 release gate is recorded separately.

## Cause and reproduction

`Cartridge.verify()` compared a newly serialized NPZ with the original
compressed-payload SHA256. NPZ is a ZIP container, and Python writes creator-OS
metadata into each ZIP central-directory entry: Windows uses `0`, Unix uses
`3`. Those bytes change even when every uncompressed array and schema member
is identical. The same portability defect affects v2 cartridges transferred
between systems, although the original failures used the legacy v1 fixture.

An exact local surrogate set `ZipInfo.create_system = 0` during serialization
of the unchanged Unix fixture. Both NPZs were 731 bytes. Only offsets 540, 592
and 644 changed, each from `3` to `0`:

| Encoding | SHA256 |
|---|---|
| Original Unix fixture | `bbe8ffc5be314bdfba3e97996875162517cf32c77453db32285ab583aab52fbb` |
| Simulated Windows encoding | `fa4127c493a47230ade08a419b84e74f2952df22a3c60bdae1761b134bfc0d23` |

The old implementation rejected the unchanged frame. With the fix, that
Windows-serialization surrogate verifies successfully and rejects an altered
observation. This surrogate is not a substitute for the subsequent Windows CI
run.

## Fix and checks

`pocket/_cartridge.py` now keeps two distinct digests for a loaded frame:

- The actual stored-payload digest, which must match the original manifest.
- A digest of the reader's own version-matched encoding at load time, against
  which later in-memory changes are checked.

Raw archive integrity remains mandatory when `verify()` is called, including
after `load(..., verify=False)`. V1 retains its original metadata semantics;
v2 detects changed attributes. The frame codec and on-disk format do not
change. Manually constructed cartridge objects retain their previous fallback
verification behavior.

Ten regression cases cover v1/v2 frames, Windows/Unix ZIP identifiers,
stored/deflated containers with changed timestamps, data and v2 metadata
mutations, and valid-but-altered payloads whose deferred checksum verification
must fail. Controls establish that container bytes changed while all
uncompressed contents remained equal.

The targeted runtime, pocket, metadata and forecast suite passed **192 tests**
with the release Python 3.12 environment. The captured rerun is in
`cartridge-targeted-tests.log`:

```bash
env MPLCONFIGDIR=/tmp/puremacro-release-mpl XDG_CACHE_HOME=/tmp/puremacro-release-cache PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/puremacro-release-4.8.0-venv/bin/python -m pytest tests/test_runtime_store_attrs.py tests/test_pocket.py tests/test_runtime.py tests/test_cross_country_forecasting.py -q --tb=short -o cache_dir=/tmp/puremacro-cartridge-portability-cache
```

No public tag was changed and no remote release action was taken during this
diagnosis. `RELEASING.md` requires fixing forward after a genuine tagged-release
failure; release preparation therefore proceeds as 4.8.1.
