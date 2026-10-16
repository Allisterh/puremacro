"""Keep research-project material out of the public repository and its artifacts.

The standing rule is that no research project may be available here. This guard
enforces it on the tracked tree and on a built sdist or wheel with three checks,
and it is written so that the code itself never names a project:

1. **Path check.** A path may not start with one of ``FORBIDDEN_PREFIXES``:
   generic names of the directories that held research records, and the stale
   playground build. Nor may it be one of the paths that moved out of this tree
   and must never come back: ``research_guard_paths.txt`` holds the SHA-256 of
   each exact path (``FORBIDDEN_PATH_DIGESTS``), generated from a private
   manifest, so this repository does not list what moved.
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
The path checks and the token check on the *path* apply to every entry. The
machine-path check and the token check on the *content* apply to text only: no
NUL byte in the first 8 KB. The token check also skips files over
``MAX_TOKEN_SCAN_BYTES``. The files in ``MACHINE_PATH_EXEMPT`` are exempt from
the machine-path check because they hold the patterns, or regex literals for
them. The two in ``SELF_EXEMPT`` also hold the plaintext sentinel, so its digest
is ignored in their content. Nothing else is exempt from anything: their paths
and every other token are checked like any file's.

Built artifacts
---------------
A wheel (``.whl``) is read as named. An sdist, a tar archive or a zip that is not
a wheel, has its single top-level ``name-version/`` directory stripped. Members
that do not share one directory, or a directory that is not named for the
archive, are refused: stripping would shift every path, and a record directory
taken for the wrapper would hide its own prefix. A link is published as its
target text, so that text is the content of a link member and of a tracked
symlink.

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
    "reviews/",
    "research_output/",
    "techreport/",
    "docs/research/",
    "docs/plans/",
    "docs/specs/",
    "docs/superpowers/",
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

#: The guard's own sources. They hold the plaintext sentinel by construction, so
#: its digest is ignored in their content. Every other digest is still found
#: there, and their paths are checked like any other.
SELF_EXEMPT = ("tools/research_guard.py", "tests/test_no_research_content.py")

#: Files exempt from the machine-path check: the guard's own sources, which hold
#: the patterns, and the older test that holds machine-path regex literals. Their
#: paths and their tokens are still checked.
MACHINE_PATH_EXEMPT = (*SELF_EXEMPT, "tests/test_no_machine_specific_paths.py")

#: A path that is guaranteed to be in ``FORBIDDEN_PATH_DIGESTS``, so that the
#: tests can tell "no moved path is back" from "a check that matches nothing".
#: It is not a real path.
SENTINEL_PATH = "zz/research-guard-sentinel-path.txt"

#: SHA-256 digests, one per line, of the exact repo-relative paths that moved out
#: of this tree and must never come back. Generated from a private manifest,
#: never edited by hand.
PATH_DIGESTS_FILE = Path(__file__).with_name("research_guard_paths.txt")

#: SHA-256 digests of the private token list. Generated, never edited by hand:
#: change the private list and regenerate.
TOKEN_DIGESTS = frozenset({
    "01b0111ab17f779b8f9091d69da14b2fa3d8a46e586d4c4008bb767fb1bff890",
    "0720e41556ec90b394a1b82c6a62586538a7411812e7d566cd720a1f28b4c357",
    "07f976b2d10a3f42d22bf46c8326c673144b319243ea052af18b2438dde42549",
    "07fa48f309b1bcd4ddffd707d8c837252271ec1e035cf4b1b6b0a825fe2fc957",
    "0bce2057efe582a337045cfa88de58e1feb63a211f996254b9e7f57fab1599b8",
    "0f55be1db2a1c0f530696bb71c36dd4bf8bd836131eaef4543dc91f0e6911363",
    "11e42621664ddeee6a848980c2c9993fc29d99388cf0908787028b6fbca594ea",
    "1d0f4bf293179e25e67861b78345b38c9bc41c16778231cf5a7391ac1868822b",
    "1f0811ba909f644fa27bb66459641cd73203b3d90cb9d6e66515dd8e73f6f058",
    "20aff4604646621c5361203404c3f080e700b9c1855dbc624c9940feb5122f96",
    "20c5e5ac4767f3e93c9941c2067c206b54ee4c67f90a8583001e7db7fb5a6ccd",
    "29b6e6d126b7ab9e582aa1a77f02cd0f359850861a5a18ca1175468997a03c5c",
    "2caaa4cca3cb19fc9c21f695d1f6ecb5fc48569e93c151ac2423096a15e6e2fa",
    "32c44e046f68c1bba7fdfd587a055c0b88cc4f84dfe46aa3afd13d1192a6eee5",
    "375597b19447dd320f8004f94417497d2511c4ea60ca0f04194401b0afce9274",
    "3a23b884450a6ee3d32bce5c25a0f14130360d852bf764bcf570b68486a2a440",
    "3c418473fa91d3776ac9980b9ba5eab80b02de035864d3432a1a2812b9a88c42",
    "3e3016f25bd841dfdedba9b0660c5c3e71516ec9a7bf6289de030503493925cd",
    "40c843b488e0db7a5cef67aa93a543e5dba96f0891f208c09cec7d63c16521ba",
    "41d00671a14f804b428d11af31bfe4c7861685d4596c01417f8723a17293085d",
    "44653a9bbc12fd1c4df45a50efc805d775ea958e5ee37259460062bccb799c2c",
    "49c1e36ba6a5f2f1fb745093c59fa13c6c970b6dec5aa02270c37a0cb28cc43d",
    "518ed232db58846fd9060c4b5a5f91288b2a87dc3a9073777a8845e537759aff",
    "547012872225cec784c05760481102689a19957eb27030b8c34e30219ab0ee61",
    "576dd68bef9c3d32467c1aae09cd3acde9eeef3595f8b2d02e788ca6c0fe55ac",
    "579cb08895323eec68be03209f86e6aa2cfee6a8121fee11d437592ae44f44a2",
    "61ccd95c2ef659ac118619ecb0f1210c92b701ca9d007fc3d730cfc016d0acaf",
    "6456da8f502c68e038081f3490792747ce2333227f4223826465b3f9b20e5713",
    "68ca9dd77979da0918037dcb9f802fd59d2266b6b301374fd1a7a113d33d6282",
    "6b62eec9cfaff23b4049ccb9879afc8e7432bcf434e7c00e0c2d730aeae2f05a",
    "710411863c5f6231c9f536d49848ed96189d222960bad5252bd2845de49f05af",
    "7105df2cf3a17799b31c2892b3cae2325888a1a05974ebe3ca25b302949c5d57",
    "7132d61c9157e502778a970a27da4f622007a6f8c5874e04c2abd308468ce04b",
    "7bbc9a0715af5371c0f25a240f8d69e5f6420b7da72e6e70a914b0bdac31260a",
    "7d79c95a7e65f52f324ead2cf449df324a7b417ca7feb2246a551f707794373d",
    "7f13ad64317bc4ae90d5f298d45dd4d20e9b500c8904037dd9bba9cf9c65d363",
    "7f57dabb9057c9f98b1fe805c4921e3ecc388fafcff19ead1f55aac885596f11",
    "802feaec8d157c5e6a4c89fb525e483b8eff9e7d539a577338044093f3369b21",
    "830591e01b671a2657b33c6cceace69cb03cf66d7b70f09f432356effaa92da5",
    "87190686a1591624989d843bf49e9bf19220f8d184baac47c5424646ccc429e1",
    "8c10d63c290a11c165a565bf048b0a1394dce99a8a41a8f2a1a2e10fbcddeef4",
    "8d67895b8084b26d21c7de76aba6a29df2c0405deb889d4fa5488fd746299815",
    "8d6ea08fdddb8a701dc3c903c95809c836acf2c4d83088d57fd400c6c3cfdd2c",
    "8f01bdaf101e4dc4c11a2531ce50d9a0296bb80160b2484617d6adb78f9988eb",
    "9090be64342fae69d48ade5f619061212fd2a6b0b48095dc1f487690c85433fd",
    "92f4f7a4bd09604a6d4b2b3073fcd9ff00c476fe885508b37a1ba10a32f0ba60",
    "9374a49227f1819e3e4b6aacd2cde946da3069fed3614bd88daf74048a8c73f2",
    "956d5a5bbff62d0a609ab7667faf7a20ab9bd6ce1b345ce916d120a45e13e1d2",
    "96a19e58960bee9185f4cad99c9f823cf06a09fc27c5bd0d8707b3e3245b58ad",
    "9f15168328ba09268d0edec3743a78f9c1912edc5b6363b7ce9bfc4043e0d7f4",
    "a284fbe9db3fa96f938f4e202ded0a85d3753b5db8bf9946b36a8230aa91769b",
    "a3b559babb29622014bd2c3327519c6131ec6639fedffc213ab3b7761e54b9e6",
    "a5096d53d912e9302a2bd648f8720588958c16f83dabee9a2996907373e5795f",
    "a518a4ab77d2e6fa138d519bf0ad4e342e87949b3bb453d300c4c3668800d5ea",
    "a8dc36fdeccca554d810e9d5c074b60c50807ed2e336d77b9d8a375bd0937cf6",
    "aeb18864fa2fab6027c5f9949f8241faa2d4f2e5bd9d7bf6d539da8f7f49d41d",
    "b178fdb009889bf3f975e71174a0555c005357133241473c2176822796cc6d56",
    "b37ae2636238b38e663b9a60cbec86c8a270b678d98b511a4179733440a8d96a",
    "b40ee3010429c73d92a7622b90d94e17224b298e9027c4d9d458deeed1d189bd",
    "b6cd32547cbf01cb3cd5bd533fa5a60cd19ca28019ee4539a9d88ebbb1c06aa5",
    "bb3237bfd8736215ac2c0c3b9416363165b35f4d1cc6bf6d0beb55f7025267da",
    "bd8f87664cd04a32db624a2272ace515ae941cc9499eb4c431736652fff4c039",
    "c08d416c837a86383227a99dc051c30b696d724093a6266d3a6c5f4b641748ad",
    "c59ead8e5daf90553797e54c3ca6d6d78898b084ef86b4466a60dd6c82638176",
    "cc5b406f42c17c2c0f18edadaa1cf97b59b3f36e8524d2effcf8e35b63b68a82",
    "cd11c0829edcf31ca6d2a5f007ae7d578047e0c1811fcd0143c2460f18300189",
    "ce38a4eb768f048075c2de881ae7a352aa1be949844943809883a1654745bddc",
    "db4b03eaf27d2876b1f367033f7545a743c681d21c8347e9c7f6530d861aa265",
    "de8295e460955e38b5b3ecbf3c625f7a60bdf8e0fbc08865ebf4dec8e4fbd811",
    "df3707a2f87861b77614ab64fc7b0399245fa6bd0ad5ab8e83ac8b040b66032f",
    "e089a42436a164f18c3d139384eac03a9f90d4fe8f023e6bcf62895e7fc2bab4",
    "e1dd607cceec3a97a061d092f223428fff456ca4e27d73bb4142ab5e68fdafd9",
    "e4fd0ec2178a5b7eeb0999cd0af65732600617e439359146c8e093bc2e532759",
    "ec54ef69f587c8d810d06d29423b70d0376bbc8c3a152186c5e69d9e56001e12",
    "ed7042862f738abea4fa636bd614cb03182f35aa755f5c0bfcd92c89680d49dd",
    "eff520430f776ecef90e4953c1b6fc10dd7b20cf08dff6bc70ef72a3b542f39e",
    "f23634b63ee8e8c45e29df091b7bc47bb540a2f517e03ac60b24ae52aea3857e",
    "f3b34aaa8acf28d987bb7ec5f59670f2d8944f3dd564050662dced0fc5cc402e",
    "f9d6fb2e8187ca3e9e60509cd2dee2b6c1aef6c8128560d63a26ace9500f42e6",
    "fc6e65444045b983c3a9ce1cf0ac05d51c5dd6e5dbceaceafd329b1e0360dcf2",
    "fda9f04c2ded017607d60770485b3f2eb5872e0f48340f2c55c5bdfcffe93602",
    "feb2858966d6eb945fa43ae80d6b5d0805bcf3f2a9c005f503edf68f57ecdd10",
})

_BINARY_SNIFF_BYTES = 8192
_RUN = re.compile(rb"[a-z0-9_]+")
_SENTINEL_DIGEST = hashlib.sha256(SENTINEL_TOKEN.encode()).digest()
_HEX_DIGEST = re.compile(r"[0-9a-f]{64}")


def _parse_digests(text: str) -> frozenset[str]:
    """The SHA-256 digests in ``text``, one per line. Anything else is a
    ``ValueError``, so a damaged file cannot quietly weaken the check."""
    digests = text.split()
    bad = [d for d in digests if not _HEX_DIGEST.fullmatch(d)]
    if bad:
        raise ValueError(f"{len(bad)} entries of the moved-path digest file are not SHA-256 digests")
    return frozenset(digests)


def _load_path_digests(path: Path) -> frozenset[str]:
    """The digests in ``path``, which must include the sentinel path's: an empty or
    truncated file would silently switch the moved-path check off."""
    digests = _parse_digests(path.read_text(encoding="ascii"))
    if hashlib.sha256(SENTINEL_PATH.encode()).hexdigest() not in digests:
        raise ValueError(f"{path.name} does not list the sentinel path: truncated, or not generated?")
    return digests


FORBIDDEN_PATH_DIGESTS: frozenset[str] = _load_path_digests(PATH_DIGESTS_FILE)


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


def _scan_entry(path: str, data: bytes, wanted: frozenset[bytes],
                moved: frozenset[str]) -> list[Finding]:
    findings: list[Finding] = []
    normal = path.replace("\\", "/")
    folded = normal.lower()
    prefix = next((p for p in FORBIDDEN_PREFIXES if folded.startswith(p)), None)
    if prefix is not None:
        findings.append(Finding(path, "forbidden path", prefix))
    elif hashlib.sha256(normal.encode("utf-8", "surrogateescape")).hexdigest() in moved:
        findings.append(Finding(path, "forbidden path", "moved path"))
    hits = _token_hits(path.encode("utf-8", "surrogateescape").lower(), wanted)
    if b"\0" not in data[:_BINARY_SNIFF_BYTES]:
        if normal not in MACHINE_PATH_EXEMPT:
            text = data.decode("utf-8", "replace")
            for name, pattern in MACHINE_PATTERNS.items():
                match = pattern.search(text)
                if match:
                    line = text.count("\n", 0, match.start()) + 1
                    findings.append(Finding(path, f"machine path: {name}", f"line {line}"))
        if len(data) <= MAX_TOKEN_SCAN_BYTES:
            in_content = _token_hits(data.lower(), wanted)
            if normal in SELF_EXEMPT:
                in_content.discard(_SENTINEL_DIGEST)
            hits |= in_content
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
    moved = FORBIDDEN_PATH_DIGESTS
    findings: list[Finding] = []
    for path, data in entries:
        findings.extend(_scan_entry(path, data, wanted, moved))
    return findings


def _run_git(repo: Path, *args: str) -> bytes:
    try:
        return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True).stdout
    except subprocess.CalledProcessError as exc:
        reason = exc.stderr.decode(errors="replace").strip()
        raise OSError(f"git {args[0]} failed in {repo}: {reason}") from exc


def git_entries(repo: Path) -> Iterator[tuple[str, bytes]]:
    """Yield the ``git ls-files`` paths of the repository containing ``repo``, with
    their working-tree bytes. Paths are relative to the top level whichever
    directory ``repo`` names: ``git ls-files`` run in a subdirectory lists only
    that directory, relative to it, and every prefix check would miss.

    A tracked symlink yields its link text, which is the blob git publishes,
    never the bytes of the file it points at (that may lie outside the tree, and
    a dangling link still has its path checked). A tracked path that is missing
    from the working tree (deleted mid-edit) is skipped; a tracked directory (a
    submodule) is yielded with no content.
    """
    top = Path(os.fsdecode(_run_git(Path(repo), "rev-parse", "--show-toplevel").rstrip(b"\r\n")))
    for raw in _run_git(top, "ls-files", "-z").split(b"\0"):
        if not raw:
            continue
        rel = os.fsdecode(raw)
        target = top / rel
        if target.is_symlink():
            data = os.fsencode(os.readlink(target))
        else:
            try:
                data = target.read_bytes()
            except (FileNotFoundError, NotADirectoryError):
                continue
            except IsADirectoryError:
                data = b""
        yield rel, data


def _components(name: str) -> list[str]:
    return [part for part in name.split("/") if part not in ("", ".")]


_ARCHIVE_SUFFIXES = (".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".tar", ".zip")


def _stem(path: Path) -> str:
    """The archive's file name without its suffix: ``name-version`` for an sdist."""
    for suffix in _ARCHIVE_SUFFIXES:
        if path.name.lower().endswith(suffix):
            return path.name[: -len(suffix)]
    return path.stem


def _require_one_root(path: Path, members: Iterable[tuple[str, bool]]) -> None:
    """``ValueError`` unless every ``(name, is_directory)`` member sits under one
    top-level directory, and that directory is the archive's own ``name-version``.
    Stripping the first component is only right then: anything else would shift
    every path and break the prefix checks, and a lone record directory taken for
    the wrapper would take its prefix with it. The messages count and name no
    member."""
    roots: set[str] = set()
    loose = False
    for name, is_directory in members:
        parts = _components(name)
        if parts:
            roots.add(parts[0])
            loose = loose or (len(parts) == 1 and not is_directory)
    if len(roots) > 1 or loose:
        raise ValueError(f"{path.name}: the members do not share one top-level directory "
                         f"({len(roots)} top-level names{', one of them a file' if loose else ''})")
    if roots and next(iter(roots)) != _stem(path):
        raise ValueError(f"{path.name}: the top-level directory does not match the file name")


def archive_entries(path: Path) -> Iterator[tuple[str, bytes]]:
    """Yield the members of a built artifact as ``(path, content)``.

    A wheel (``.whl``) is read as named. An sdist, a tar archive or a zip that is
    not a wheel, has its single top-level ``name-version/`` directory stripped;
    members that do not share one, or whose directory is not named for the
    archive, raise ``ValueError``. Directories are skipped.
    A link member yields its link name as content, as a link is published as its
    target text, which can name a machine.
    """
    path = Path(path)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            wheel = path.suffix.lower() == ".whl"
            if not wheel:
                _require_one_root(path, [(i.filename, i.is_dir()) for i in infos])
            for info in infos:
                if not info.is_dir():
                    name = info.filename if wheel else "/".join(_components(info.filename)[1:])
                    yield name, archive.read(info)
    elif tarfile.is_tarfile(path):
        with tarfile.open(path) as sdist:
            members = sdist.getmembers()
            _require_one_root(path, [(m.name, m.isdir()) for m in members])
            for member in members:
                if member.isdir():
                    continue
                if member.isfile():
                    data = sdist.extractfile(member).read()
                elif member.issym() or member.islnk():
                    data = member.linkname.encode("utf-8", "surrogateescape")
                else:
                    data = b""
                yield "/".join(_components(member.name)[1:]), data
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
            findings += _scan_source("tracked tree", git_entries(Path.cwd()))
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
