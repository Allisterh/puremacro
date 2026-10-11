"""Keep research-project material out of the public repository and its artifacts.

The standing rule is that no research project may be available here. This guard
enforces it on the tracked tree and on a built sdist or wheel with three checks,
and it is written so that the code itself never names a project:

1. **Path check.** A path may not start with one of ``FORBIDDEN_PREFIXES``:
   generic names of the directories that held research records, and the stale
   playground build.
2. **Machine-path check.** A text file may not contain a path that exists on
   exactly one machine (``MACHINE_PATTERNS``): a mounted volume, a user's home
   directory, or the ``RESEARCH/`` root.
3. **Hashed-name check.** Project names are checked as SHA-256 digests
   (``TOKEN_DIGESTS``), so reading this file does not reveal them. A finding
   names the file and the first twelve hex characters of the digest, never the
   token. (When the token is in the *path* it is of course in the path that is
   printed: that is the file's own name.)

What a name is
--------------
A text is cut into maximal runs of ``[A-Za-z0-9_]``, lower-cased. A run is a
token, and so is every group of consecutive underscore-separated words inside
one: ``a_b_c`` offers ``a_b_c``, ``a_b``, ``b_c``, ``a``, ``b`` and ``c``. Names
of modules and tests are snake_case, and a research name usually turns up inside
a longer one (``test_<name>_x.py``), which a whole-run match would miss. A token
matches whole words only, never inside a word, and only groups of up to
``MAX_TOKEN_WORDS`` words are tried, which bounds the work on odd input.

Which files are read
--------------------
The path check and the token check on the *path* apply to every entry. The
machine-path check and the token check on the *content* apply to text only: no
NUL byte in the first 8 KB. The token check also skips files over
``MAX_TOKEN_SCAN_BYTES``. The two files in ``SELF_EXEMPT`` are exempt from the
content checks, because by construction they hold machine-path patterns, digests
and the plaintext sentinel; their paths are still checked.

Usage
-----
::

    python tools/research_guard.py --git      # tracked files of the repository you are in
    python tools/research_guard.py dist/*     # built sdists and wheels

One line per finding, ``<path>: <kind>: <detail>``. Exit status 0 when clean, 1
when there are findings, 2 when the scan could not run (including a scan that
saw no files, which would otherwise pass vacuously). Standard library only, so a
release job can run it before anything is installed.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import tarfile
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

#: Generic prefixes no tracked path may start with (matched case-insensitively,
#: with ``\`` read as ``/``). They name only the directories that held research
#: records and the stale playground build, never a project.
FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "puremacro/trade/",
    "reviews/",
    "research_output/",
    "techreport/",
    "docs/research/",
    "docs/plans/",
    "docs/specs/",
    "docs/superpowers/",
    "tools/run_main_street_",
    "playground/dist/",
)

#: Paths that exist on exactly one machine, searched in text content.
MACHINE_PATTERNS: dict[str, re.Pattern[str]] = {
    "volume path": re.compile(r"/Volumes/"),
    "user home": re.compile(r"/Users/[A-Za-z0-9._-]+/"),
    "research root": re.compile(r"\bRESEARCH/"),
}

#: The one plaintext token in this repository. Its digest is in ``TOKEN_DIGESTS``
#: so that the hashed check always has a token it is guaranteed to find, which
#: is what lets the tests tell "no research content" from "a check that matches
#: nothing". It is not a real name.
SENTINEL_TOKEN = "zzresearchguardsentinel"

#: Files larger than this are covered by the path and machine-path checks only.
MAX_TOKEN_SCAN_BYTES = 10 * 1024 * 1024

#: Longest group of underscore-separated words tried as a token.
MAX_TOKEN_WORDS = 10

#: The guard's own sources: exempt from the content checks, never from the path
#: check.
SELF_EXEMPT = ("tools/research_guard.py", "tests/test_no_research_content.py")

#: SHA-256 digests of the private token list. Generated, never edited by hand:
#: change the private list and regenerate.
TOKEN_DIGESTS = frozenset({
    "01b0111ab17f779b8f9091d69da14b2fa3d8a46e586d4c4008bb767fb1bff890",
    "0720e41556ec90b394a1b82c6a62586538a7411812e7d566cd720a1f28b4c357",
    "07f976b2d10a3f42d22bf46c8326c673144b319243ea052af18b2438dde42549",
    "0f55be1db2a1c0f530696bb71c36dd4bf8bd836131eaef4543dc91f0e6911363",
    "11e42621664ddeee6a848980c2c9993fc29d99388cf0908787028b6fbca594ea",
    "1f0811ba909f644fa27bb66459641cd73203b3d90cb9d6e66515dd8e73f6f058",
    "2caaa4cca3cb19fc9c21f695d1f6ecb5fc48569e93c151ac2423096a15e6e2fa",
    "32c44e046f68c1bba7fdfd587a055c0b88cc4f84dfe46aa3afd13d1192a6eee5",
    "375597b19447dd320f8004f94417497d2511c4ea60ca0f04194401b0afce9274",
    "40c843b488e0db7a5cef67aa93a543e5dba96f0891f208c09cec7d63c16521ba",
    "41d00671a14f804b428d11af31bfe4c7861685d4596c01417f8723a17293085d",
    "44653a9bbc12fd1c4df45a50efc805d775ea958e5ee37259460062bccb799c2c",
    "49c1e36ba6a5f2f1fb745093c59fa13c6c970b6dec5aa02270c37a0cb28cc43d",
    "576dd68bef9c3d32467c1aae09cd3acde9eeef3595f8b2d02e788ca6c0fe55ac",
    "6456da8f502c68e038081f3490792747ce2333227f4223826465b3f9b20e5713",
    "68ca9dd77979da0918037dcb9f802fd59d2266b6b301374fd1a7a113d33d6282",
    "7105df2cf3a17799b31c2892b3cae2325888a1a05974ebe3ca25b302949c5d57",
    "7f13ad64317bc4ae90d5f298d45dd4d20e9b500c8904037dd9bba9cf9c65d363",
    "802feaec8d157c5e6a4c89fb525e483b8eff9e7d539a577338044093f3369b21",
    "8c10d63c290a11c165a565bf048b0a1394dce99a8a41a8f2a1a2e10fbcddeef4",
    "8d67895b8084b26d21c7de76aba6a29df2c0405deb889d4fa5488fd746299815",
    "8f01bdaf101e4dc4c11a2531ce50d9a0296bb80160b2484617d6adb78f9988eb",
    "9090be64342fae69d48ade5f619061212fd2a6b0b48095dc1f487690c85433fd",
    "92f4f7a4bd09604a6d4b2b3073fcd9ff00c476fe885508b37a1ba10a32f0ba60",
    "9374a49227f1819e3e4b6aacd2cde946da3069fed3614bd88daf74048a8c73f2",
    "96a19e58960bee9185f4cad99c9f823cf06a09fc27c5bd0d8707b3e3245b58ad",
    "a284fbe9db3fa96f938f4e202ded0a85d3753b5db8bf9946b36a8230aa91769b",
    "a3b559babb29622014bd2c3327519c6131ec6639fedffc213ab3b7761e54b9e6",
    "a5096d53d912e9302a2bd648f8720588958c16f83dabee9a2996907373e5795f",
    "b37ae2636238b38e663b9a60cbec86c8a270b678d98b511a4179733440a8d96a",
    "b40ee3010429c73d92a7622b90d94e17224b298e9027c4d9d458deeed1d189bd",
    "c08d416c837a86383227a99dc051c30b696d724093a6266d3a6c5f4b641748ad",
    "cc5b406f42c17c2c0f18edadaa1cf97b59b3f36e8524d2effcf8e35b63b68a82",
    "cd11c0829edcf31ca6d2a5f007ae7d578047e0c1811fcd0143c2460f18300189",
    "ce38a4eb768f048075c2de881ae7a352aa1be949844943809883a1654745bddc",
    "e089a42436a164f18c3d139384eac03a9f90d4fe8f023e6bcf62895e7fc2bab4",
    "e1dd607cceec3a97a061d092f223428fff456ca4e27d73bb4142ab5e68fdafd9",
    "e4fd0ec2178a5b7eeb0999cd0af65732600617e439359146c8e093bc2e532759",
    "ec54ef69f587c8d810d06d29423b70d0376bbc8c3a152186c5e69d9e56001e12",
    "f23634b63ee8e8c45e29df091b7bc47bb540a2f517e03ac60b24ae52aea3857e",
    "fc6e65444045b983c3a9ce1cf0ac05d51c5dd6e5dbceaceafd329b1e0360dcf2",
})

_BINARY_SNIFF_BYTES = 8192
_RUN = re.compile(rb"[a-z0-9_]+")


@dataclass(frozen=True)
class Finding:
    """One thing the guard objects to.

    ``kind`` is ``"forbidden path"`` (``detail`` is the prefix matched),
    ``"machine path: <name>"`` (``detail`` is ``line N``) or ``"hashed token"``
    (``detail`` is the first 12 hex characters of the digest, never the token).
    """

    path: str
    kind: str
    detail: str

    def __str__(self) -> str:
        return f"{self.path}: {self.kind}: {self.detail}"


def _candidates(lowered: bytes) -> set[bytes]:
    """Every token ``lowered`` offers: its runs and the word groups inside them."""
    runs = set(_RUN.findall(lowered))
    found = set(runs)
    for run in runs:
        if b"_" not in run:
            continue
        words = run.split(b"_")
        for start in range(len(words)):
            for stop in range(start + 1, min(len(words), start + MAX_TOKEN_WORDS) + 1):
                found.add(b"_".join(words[start:stop]))
    return found


def _token_hits(lowered: bytes, wanted: frozenset[bytes]) -> set[bytes]:
    """The raw digests in ``wanted`` that some token of ``lowered`` hashes to.

    Nothing is cached across files: on a 5,000-file tree a digest cache held
    millions of one-off tokens and ran slower than hashing them again.
    """
    sha256 = hashlib.sha256
    return {d for token in _candidates(lowered) if (d := sha256(token).digest()) in wanted}


def _scan_entry(path: str, data: bytes, wanted: frozenset[bytes]) -> list[Finding]:
    findings: list[Finding] = []
    normal = path.replace("\\", "/")
    folded = normal.lower()
    for prefix in FORBIDDEN_PREFIXES:
        if folded.startswith(prefix):
            findings.append(Finding(path, "forbidden path", prefix))
            break
    hits = _token_hits(path.encode("utf-8", "surrogateescape").lower(), wanted)
    if normal not in SELF_EXEMPT and b"\0" not in data[:_BINARY_SNIFF_BYTES]:
        text = data.decode("utf-8", "replace")
        for name, pattern in MACHINE_PATTERNS.items():
            match = pattern.search(text)
            if match:
                line = text.count("\n", 0, match.start()) + 1
                findings.append(Finding(path, f"machine path: {name}", f"line {line}"))
        if len(data) <= MAX_TOKEN_SCAN_BYTES:
            hits |= _token_hits(data.lower(), wanted)
    findings.extend(Finding(path, "hashed token", d.hex()[:12]) for d in sorted(hits))
    return findings


def scan_entries(entries: Iterable[tuple[str, bytes]]) -> list[Finding]:
    """Check ``(path, content)`` pairs and return every finding, in input order.

    Within one entry the order is: the path prefix, the machine patterns in
    ``MACHINE_PATTERNS`` order, then the hashed tokens by digest, so the output
    is the same on every run. A token found in both the path and the content is
    reported once.
    """
    wanted = frozenset(bytes.fromhex(d) for d in TOKEN_DIGESTS)
    findings: list[Finding] = []
    for path, data in entries:
        findings.extend(_scan_entry(path, data, wanted))
    return findings


def _run_git(repo: Path, *args: str) -> bytes:
    try:
        return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True).stdout
    except subprocess.CalledProcessError as exc:
        reason = exc.stderr.decode(errors="replace").strip()
        raise OSError(f"git {args[0]} failed in {repo}: {reason}") from exc


def git_entries(repo: Path) -> Iterator[tuple[str, bytes]]:
    """Yield ``git ls-files`` paths of ``repo`` with their working-tree bytes.

    A tracked path that is missing from the working tree (deleted mid-edit) is
    skipped; a tracked directory (a submodule) is yielded with no content.
    """
    repo = Path(repo)
    for raw in _run_git(repo, "ls-files", "-z").split(b"\0"):
        if not raw:
            continue
        rel = os.fsdecode(raw)
        try:
            data = (repo / rel).read_bytes()
        except (FileNotFoundError, NotADirectoryError):
            continue
        except IsADirectoryError:
            data = b""
        yield rel, data


def archive_entries(path: Path) -> Iterator[tuple[str, bytes]]:
    """Yield the members of a wheel (zip) as named, or of an sdist (``.tar.gz``)
    with its top-level ``name-version/`` directory stripped. Directories are
    skipped."""
    path = Path(path)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as wheel:
            for info in wheel.infolist():
                if not info.is_dir():
                    yield info.filename, wheel.read(info)
    elif tarfile.is_tarfile(path):
        with tarfile.open(path) as sdist:
            for member in sdist:
                if member.isdir():
                    continue
                parts = [p for p in member.name.split("/") if p not in ("", ".")]
                data = sdist.extractfile(member).read() if member.isfile() else b""
                yield "/".join(parts[1:] or parts), data
    else:
        raise ValueError(f"{path}: neither a wheel (zip) nor an sdist (tar)")


def _scan_source(label: str, entries: Iterable[tuple[str, bytes]]) -> list[Finding]:
    """``scan_entries``, refusing a scan that saw nothing: an empty file list
    passes every check, so it must not be reported as clean."""
    seen = 0

    def counted() -> Iterator[tuple[str, bytes]]:
        nonlocal seen
        for entry in entries:
            seen += 1
            yield entry

    findings = scan_entries(counted())
    if not seen:
        raise ValueError(f"{label}: no files to scan")
    print(f"research_guard: {label}: {seen} files scanned", file=sys.stderr)
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="research_guard",
        description="Fail if the tracked tree or a built artifact carries research material.")
    parser.add_argument("--git", action="store_true",
                        help="scan the tracked files of the repository containing the working directory")
    parser.add_argument("artifacts", nargs="*", type=Path, metavar="ARTIFACT",
                        help="a built sdist (.tar.gz) or wheel (.whl) to scan")
    args = parser.parse_args(argv)
    if not (args.git or args.artifacts):
        parser.error("give --git, one or more built artifacts, or both")

    findings: list[Finding] = []
    try:
        if args.git:
            top = _run_git(Path.cwd(), "rev-parse", "--show-toplevel").decode().strip()
            findings += _scan_source("tracked tree", git_entries(Path(top)))
        for artifact in args.artifacts:
            findings += _scan_source(artifact.name, archive_entries(artifact))
    except (OSError, ValueError, zipfile.BadZipFile, tarfile.TarError) as exc:
        print(f"research_guard: {exc}", file=sys.stderr)
        return 2
    for finding in findings:
        print(finding)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
