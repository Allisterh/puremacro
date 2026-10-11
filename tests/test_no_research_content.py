"""The research guard: the tracked tree and the built artifacts carry no research material.

The standing rule is that no research project may be available in the public
repository. ``tools/research_guard.py`` enforces it three ways, and this file is
the guard's own test suite plus the one test that runs it over the real tree:

1. **Path check.** No tracked path may start with one of a short list of generic
   prefixes (the records directories, the stale playground build, ...).
2. **Machine-path check.** No tracked text file may contain a path that exists
   on exactly one machine: ``/Volumes/``, ``/Users/<name>/`` or ``RESEARCH/``.
3. **Hashed-name check.** Project names are never written down in this
   repository: the guard holds only the SHA-256 digests of a private list of
   distinctive identifiers, and a match reports the file and the first twelve
   hex characters of the digest, never the token.

Almost everything below is a *positive control*. A guard that has silently
stopped matching passes every scan of every tree, which is indistinguishable
from a clean tree, so each behaviour is shown to fire on a hand-built input
before the real tree is trusted to be clean. The only plaintext token in this
repository is ``SENTINEL_TOKEN``, whose digest is in the frozen set so that the
hashed check has something it is guaranteed to find.
"""
from __future__ import annotations

import ast
import collections
import dataclasses
import hashlib
import io
import os
import re
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

import tools.research_guard as guard
from tools.research_guard import (
    FORBIDDEN_PREFIXES,
    MACHINE_PATTERNS,
    MAX_TOKEN_SCAN_BYTES,
    MAX_TOKEN_WORDS,
    SELF_EXEMPT,
    SENTINEL_TOKEN,
    TOKEN_DIGESTS,
    Finding,
    archive_entries,
    git_entries,
    main,
    scan_entries,
)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _no_inherited_git_environment(monkeypatch):
    """A git hook exports ``GIT_DIR`` / ``GIT_INDEX_FILE``, which would redirect
    every ``git`` call below (the scratch repositories included) to the wrong
    repository."""
    for name in [n for n in os.environ if n.startswith("GIT_")]:
        monkeypatch.delenv(name)


def _kinds(findings):
    return [f.kind for f in findings]


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


# ---------------------------------------------------------------------------
# helpers that build the inputs the guard reads
# ---------------------------------------------------------------------------

def _wheel(path: Path, members: dict[str, bytes], directories: tuple[str, ...] = ()) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name in directories:
            zf.writestr(name.rstrip("/") + "/", b"")
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def _sdist(path: Path, members: dict[str, bytes], directories: tuple[str, ...] = ()) -> Path:
    with tarfile.open(path, "w:gz") as tf:
        for name in directories:
            info = tarfile.TarInfo(name)
            info.type = tarfile.DIRTYPE
            tf.addfile(info)
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _scratch_repo(root: Path, files: dict[str, bytes]) -> Path:
    """A throwaway repository whose index holds ``files`` (nothing committed)."""
    root.mkdir()
    _git(root, "init", "-q")
    for name, data in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    _git(root, "add", "--", *files)
    return root


# ---------------------------------------------------------------------------
# 1. the path check
# ---------------------------------------------------------------------------

#: The generic prefixes the path check must forbid, written out here rather
#: than read back from the module so that a dropped entry fails a test instead
#: of quietly narrowing the guard.
_SPEC_PREFIXES = (
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


def test_a_finding_is_an_immutable_value():
    finding = Finding("a", "b", "c")
    assert finding == Finding("a", "b", "c") and hash(finding) == hash(Finding("a", "b", "c"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        finding.path = "d"


def test_forbidden_prefix_is_found():
    found = scan_entries([("reviews/x.md", b"")])
    assert [(f.path, f.kind) for f in found] == [("reviews/x.md", "forbidden path")]


@pytest.mark.parametrize("prefix", _SPEC_PREFIXES)
def test_every_spec_prefix_is_forbidden(prefix):
    assert prefix in FORBIDDEN_PREFIXES
    found = scan_entries([(prefix + "x.py", b"")])
    assert found == [Finding(prefix + "x.py", "forbidden path", prefix)]


@pytest.mark.parametrize("path", [
    "docs/reviews/x.md",          # a prefix only counts at the start of a path
    "puremacro/trade_utils.py",   # the trailing slash is part of the prefix
    "docs/researchers.md",
    "tools/run_all.py",
])
def test_paths_that_only_resemble_a_prefix_pass(path):
    assert scan_entries([(path, b"")]) == []


@pytest.mark.parametrize("path", ["Reviews/x.md", "REVIEWS/x.md", "reviews\\x.md"])
def test_prefix_check_ignores_case_and_separator_style(path):
    """macOS and Windows resolve these to the same directory as ``reviews/``."""
    assert _kinds(scan_entries([(path, b"")])) == ["forbidden path"]


def test_tracked_playground_build_is_forbidden():
    """The built playground holds copies of notebooks, so it may not be tracked."""
    found = scan_entries([("playground/dist/x.js", b"")])
    assert _kinds(found) == ["forbidden path"]


def test_playground_build_gets_no_token_scan_exclusion():
    """No token-scan exclusion: one for the build would hide the notebook copies in it."""
    found = scan_entries([("playground/dist/x.js", SENTINEL_TOKEN.encode())])
    assert sorted(_kinds(found)) == ["forbidden path", "hashed token"]
    assert not hasattr(guard, "TOKEN_SCAN_EXCLUDE")


# ---------------------------------------------------------------------------
# 2. the machine-path check
# ---------------------------------------------------------------------------

def test_machine_path_is_found():
    found = scan_entries([("docs/x.md", b"see /Volumes/DATA/x")])
    assert _kinds(found) == ["machine path: volume path"]


def test_the_machine_patterns_are_the_three_generic_ones():
    assert set(MACHINE_PATTERNS) == {"volume path", "user home", "research root"}


@pytest.mark.parametrize("name, text", [
    ("volume path", b"see /Volumes/DATA/x"),
    ("user home", b"open /Users/someone/notes.txt"),
    ("research root", b"cd ~/Documents/RESEARCH/proj"),
])
def test_each_machine_pattern_fires(name, text):
    assert _kinds(scan_entries([("docs/x.md", text)])) == [f"machine path: {name}"]


@pytest.mark.parametrize("text", [
    b"/Users/<name>/Documents",           # a placeholder names no machine
    b"see /Volume/x and /Volumes",        # singular, or no trailing slash
    b"SEARCH/x MYRESEARCH/x research/x",  # needs a word boundary and capitals
])
def test_text_that_is_not_a_machine_path_passes(text):
    assert scan_entries([("docs/x.md", text)]) == []


def test_machine_path_finding_names_the_line_and_not_the_path():
    text = b"ok\nok\nsee /Volumes/DATA/somewhere_private\n"
    assert scan_entries([("docs/x.md", text)]) == [
        Finding("docs/x.md", "machine path: volume path", "line 3")]


# ---------------------------------------------------------------------------
# 3. the hashed-name check
# ---------------------------------------------------------------------------

def test_sentinel_digest_is_frozen_into_the_digest_set():
    assert _digest(SENTINEL_TOKEN) in TOKEN_DIGESTS


def test_the_digest_set_is_well_formed():
    """A truncated or upper-case digest would never match anything."""
    assert all(re.fullmatch(r"[0-9a-f]{64}", d) for d in TOKEN_DIGESTS)
    assert len(TOKEN_DIGESTS) > 10, "regenerated from a truncated list?"


def test_hashed_token_in_content_is_found_without_revealing_it():
    found = scan_entries([("docs/x.md", f"uses {SENTINEL_TOKEN}.".encode())])
    assert _kinds(found) == ["hashed token"]
    assert SENTINEL_TOKEN not in found[0].detail
    assert found[0].detail == _digest(SENTINEL_TOKEN)[:12]
    assert SENTINEL_TOKEN not in str(found[0])


def test_hashed_token_in_a_path_is_found():
    found = scan_entries([(f"docs/{SENTINEL_TOKEN}_notes.md", b"")])
    assert _kinds(found) == ["hashed token"]
    assert SENTINEL_TOKEN not in found[0].detail


def test_a_token_in_a_mixed_case_path_is_found():
    """Tokens are lower-cased in paths as they are in content."""
    path = f"docs/{SENTINEL_TOKEN.capitalize()}-Notes.md"
    assert _kinds(scan_entries([(path, b"")])) == ["hashed token"]


@pytest.mark.parametrize("text", [
    f"def test_{SENTINEL_TOKEN}_x(): pass",    # inside a longer snake_case name
    f"{SENTINEL_TOKEN}_data = 1",
    f"x_{SENTINEL_TOKEN}",
    f"path/to/{SENTINEL_TOKEN}.py",
    f"{SENTINEL_TOKEN}-notes",                  # a hyphen ends the run
    SENTINEL_TOKEN.upper(),                     # tokens are lower-cased
    f"_{SENTINEL_TOKEN}_",
])
def test_a_token_is_found_as_a_whole_word_of_a_name(text):
    assert _kinds(scan_entries([("docs/x.md", text.encode())])) == ["hashed token"]


@pytest.mark.parametrize("text", [
    f"{SENTINEL_TOKEN}s",                  # a longer word is a different word
    f"x{SENTINEL_TOKEN}",
    f"{SENTINEL_TOKEN}1",
    SENTINEL_TOKEN[:-1],                   # a prefix of the token
])
def test_a_token_is_not_found_inside_a_longer_word(text):
    assert scan_entries([("docs/x.md", text.encode())]) == []


def test_findings_for_one_file_come_out_in_a_fixed_order():
    """Path, then the machine patterns in their declared order, then tokens."""
    path = "reviews/x.md"
    text = f"{SENTINEL_TOKEN} RESEARCH/a /Users/someone/b /Volumes/c".encode()
    assert _kinds(scan_entries([(path, text)])) == [
        "forbidden path",
        "machine path: volume path",
        "machine path: user home",
        "machine path: research root",
        "hashed token",
    ]
    assert list(MACHINE_PATTERNS) == ["volume path", "user home", "research root"]


def test_a_token_is_reported_once_per_file_however_often_it_occurs():
    entry = (f"docs/{SENTINEL_TOKEN}.md",
             f"{SENTINEL_TOKEN} {SENTINEL_TOKEN} test_{SENTINEL_TOKEN}".encode())
    assert len(scan_entries([entry])) == 1


def test_several_tokens_are_reported_in_digest_order_and_never_named(monkeypatch):
    """Six tokens, so an unsorted report matches by luck once in 720 runs."""
    words = ("alpha_beta", "gamma", "delta", "epsilon_zeta_eta", "theta", "iota")
    digests = frozenset(_digest(w) for w in words)
    monkeypatch.setattr(guard, "TOKEN_DIGESTS", digests)
    found = scan_entries([("docs/x.md", " then ".join(reversed(words)).encode())])
    assert [f.detail for f in found] == sorted(d[:12] for d in digests)
    assert not any(w in str(f) for f in found for w in words)


def test_a_token_longer_than_the_word_limit_is_found_only_as_a_whole_name(monkeypatch):
    word_group = "_".join(f"w{i}" for i in range(MAX_TOKEN_WORDS))
    longer = word_group + "_w_extra"
    monkeypatch.setattr(guard, "TOKEN_DIGESTS", frozenset({_digest(word_group), _digest(longer)}))
    embedded = scan_entries([("a.md", f"x_{word_group}_y".encode())])
    assert [f.detail for f in embedded] == [_digest(word_group)[:12]]
    whole = scan_entries([("a.md", longer.encode())])
    assert _digest(longer)[:12] in [f.detail for f in whole]
    # one word more than the limit, embedded in something longer: out of reach
    too_long = "_".join(f"w{i}" for i in range(MAX_TOKEN_WORDS + 1))
    monkeypatch.setattr(guard, "TOKEN_DIGESTS", frozenset({_digest(too_long)}))
    assert scan_entries([("a.md", f"x_{too_long}_y".encode())]) == []
    assert _kinds(scan_entries([("a.md", too_long.encode())])) == ["hashed token"]


# ---------------------------------------------------------------------------
# which files get the text checks
# ---------------------------------------------------------------------------

def test_binary_content_skips_text_checks():
    assert scan_entries([("x.bin", b"\0" + SENTINEL_TOKEN.encode())]) == []
    assert scan_entries([("y.bin", b"\0 /Volumes/A/")]) == []


def test_undecodable_bytes_in_a_path_or_in_content_do_not_stop_the_scan():
    latin_1 = b"caf\xe9 /Volumes/A/ " + SENTINEL_TOKEN.encode()
    assert _kinds(scan_entries([("docs/x.txt", latin_1)])) == [
        "machine path: volume path", "hashed token"]
    assert scan_entries([("docs/\udcff.md", b"")]) == []     # what os.fsdecode makes of a bad name


def test_only_a_nul_byte_in_the_first_8_kb_makes_a_file_binary():
    filler = b" " * 8191
    nul_inside = filler + b"\0" + b" /Volumes/A/"        # NUL at offset 8191
    nul_outside = filler + b" " + b"\0 /Volumes/A/"      # NUL at offset 8192
    assert scan_entries([("a.bin", nul_inside)]) == []
    assert _kinds(scan_entries([("b.txt", nul_outside)])) == ["machine path: volume path"]


def _padded(size: int) -> bytes:
    head = b"/Volumes/A/\n" + SENTINEL_TOKEN.encode() + b"\n"
    return head + b" " * (size - len(head))


def test_the_token_scan_stops_just_over_the_size_limit_but_the_machine_scan_does_not():
    """A file over 10 MB is covered by the path and machine-path checks only."""
    at_limit = scan_entries([("docs/a.txt", _padded(MAX_TOKEN_SCAN_BYTES))])
    over_limit = scan_entries([("docs/b.txt", _padded(MAX_TOKEN_SCAN_BYTES + 1))])
    assert _kinds(at_limit) == ["machine path: volume path", "hashed token"]
    assert _kinds(over_limit) == ["machine path: volume path"]


def test_guard_sources_are_exempt_from_machine_patterns():
    assert scan_entries([("tools/research_guard.py", b'r"/Volumes/"')]) == []


def test_guard_sources_are_exempt_from_token_scan():
    """By construction they hold the plaintext sentinel and the digests."""
    assert scan_entries([("tests/test_no_research_content.py", SENTINEL_TOKEN.encode())]) == []


def test_the_exemption_is_exactly_the_two_guard_files():
    assert SELF_EXEMPT == ("tools/research_guard.py", "tests/test_no_research_content.py")


def test_the_exemption_covers_content_and_never_the_path(monkeypatch):
    monkeypatch.setattr(guard, "SELF_EXEMPT", ("reviews/x.md",))
    found = scan_entries([("reviews/x.md", f"/Volumes/A/ {SENTINEL_TOKEN}".encode())])
    assert _kinds(found) == ["forbidden path"]


# ---------------------------------------------------------------------------
# built artifacts
# ---------------------------------------------------------------------------

def test_artifact_members_are_scanned(tmp_path):
    wheel = _wheel(tmp_path / "puremacro-5.0.0a1-py3-none-any.whl",
                   {"puremacro/reviews_x.py": b"/Volumes/A/"})
    found = scan_entries(archive_entries(wheel))
    assert [(f.path, f.kind) for f in found] == [
        ("puremacro/reviews_x.py", "machine path: volume path")]

    sdist = _sdist(tmp_path / "puremacro-5.0.0a1.tar.gz",
                   {"puremacro-5.0.0a1/reviews/a.md": b""})
    found = scan_entries(archive_entries(sdist))
    assert [(f.path, f.kind) for f in found] == [("reviews/a.md", "forbidden path")]


def test_wheel_members_are_read_as_they_are_and_directories_are_skipped(tmp_path):
    members = {"puremacro/__init__.py": b"A",
               "puremacro-5.0.0a1.dist-info/METADATA": b"B"}
    wheel = _wheel(tmp_path / "w.whl", members, directories=("puremacro/sub",))
    assert list(archive_entries(wheel)) == list(members.items())


def test_the_sdist_root_directory_is_stripped_and_directories_are_skipped(tmp_path):
    sdist = _sdist(
        tmp_path / "puremacro-5.0.0a1.tar.gz",
        {"puremacro-5.0.0a1/pyproject.toml": b"C",
         "puremacro-5.0.0a1/puremacro/__init__.py": b"D"},
        directories=("puremacro-5.0.0a1", "puremacro-5.0.0a1/puremacro"),
    )
    assert dict(archive_entries(sdist)) == {
        "pyproject.toml": b"C", "puremacro/__init__.py": b"D"}


def test_a_link_member_of_an_sdist_still_has_its_path_checked(tmp_path):
    sdist = tmp_path / "puremacro-5.0.0a1.tar.gz"
    with tarfile.open(sdist, "w:gz") as tf:
        info = tarfile.TarInfo("puremacro-5.0.0a1/reviews/link")
        info.type = tarfile.SYMTYPE
        info.linkname = "elsewhere"
        tf.addfile(info)
    assert list(archive_entries(sdist)) == [("reviews/link", b"")]
    assert _kinds(scan_entries(archive_entries(sdist))) == ["forbidden path"]


def test_something_that_is_neither_wheel_nor_sdist_is_rejected(tmp_path):
    junk = tmp_path / "x.txt"
    junk.write_bytes(b"just text")
    with pytest.raises(ValueError):
        list(archive_entries(junk))


# ---------------------------------------------------------------------------
# the tracked tree
# ---------------------------------------------------------------------------

def test_git_entries_yields_tracked_files_with_their_working_tree_bytes(tmp_path):
    repo = _scratch_repo(tmp_path / "r", {"a.txt": b"staged", "sub/b.txt": b"x"})
    (repo / "a.txt").write_bytes(b"edited after staging")
    (repo / "untracked.txt").write_bytes(b"never added")
    assert list(git_entries(repo)) == [
        ("a.txt", b"edited after staging"), ("sub/b.txt", b"x")]


def test_git_entries_skips_a_tracked_path_missing_from_the_working_tree(tmp_path):
    repo = _scratch_repo(tmp_path / "r", {"keep.txt": b"k", "gone.txt": b"g"})
    (repo / "gone.txt").unlink()
    assert list(git_entries(repo)) == [("keep.txt", b"k")]


def test_git_entries_raises_outside_a_repository(tmp_path, monkeypatch):
    """A failed ``git ls-files`` must not look like an empty, clean tree."""
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))   # never find a parent repo
    with pytest.raises(OSError, match="ls-files"):
        list(git_entries(tmp_path))


def _format(findings, limit=200):
    counts = collections.Counter(f.kind for f in findings)
    head = ", ".join(f"{n} x {kind}" for kind, n in sorted(counts.items()))
    lines = [str(f) for f in findings[:limit]]
    if len(findings) > limit:
        lines.append(f"... and {len(findings) - limit} more")
    return f"{len(findings)} research-guard finding(s): {head}\n" + "\n".join(lines)


@pytest.mark.xfail(strict=True, reason="green once Task 8 lands")
def test_the_tree_has_no_research_content():
    findings = scan_entries(git_entries(REPO))
    if findings:
        pytest.fail(_format(findings), pytrace=False)


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------

def test_main_prints_one_line_per_finding_and_returns_1(tmp_path, capsys):
    sdist = _sdist(tmp_path / "puremacro-5.0.0a1.tar.gz", {
        "puremacro-5.0.0a1/reviews/a.md": b"",
        "puremacro-5.0.0a1/docs/x.md": f"see /Volumes/A/ and {SENTINEL_TOKEN}".encode(),
    })
    assert main([str(sdist)]) == 1
    out = capsys.readouterr().out
    assert out.splitlines() == [
        "reviews/a.md: forbidden path: reviews/",
        "docs/x.md: machine path: volume path: line 1",
        f"docs/x.md: hashed token: {_digest(SENTINEL_TOKEN)[:12]}",
    ]
    assert SENTINEL_TOKEN not in out


def test_main_returns_0_and_prints_no_finding_for_a_clean_artifact(tmp_path, capsys):
    wheel = _wheel(tmp_path / "puremacro-5.0.0a1-py3-none-any.whl",
                   {"puremacro/__init__.py": b"__version__ = '5.0.0a1'\n"})
    assert main([str(wheel)]) == 0
    assert capsys.readouterr().out == ""


def test_main_git_scans_the_repository_of_the_working_directory(tmp_path, monkeypatch, capsys):
    repo = _scratch_repo(tmp_path / "r", {"ok/a.py": b"x = 1\n", "reviews/b.md": b""})
    monkeypatch.chdir(repo / "ok")          # a subdirectory: the whole repo is scanned
    assert main(["--git"]) == 1
    assert capsys.readouterr().out.splitlines() == ["reviews/b.md: forbidden path: reviews/"]


def test_main_scans_the_tree_and_artifacts_in_one_run(tmp_path, monkeypatch, capsys):
    repo = _scratch_repo(tmp_path / "r", {"ok.py": b"x = 1\n"})
    wheel = _wheel(tmp_path / "w.whl", {"techreport/a.md": b""})
    monkeypatch.chdir(repo)
    assert main(["--git", str(wheel)]) == 1
    assert capsys.readouterr().out.splitlines() == ["techreport/a.md: forbidden path: techreport/"]


def test_main_with_nothing_to_scan_is_a_usage_error():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_main_git_outside_a_repository_is_an_error_not_a_pass(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))   # never find a parent repo
    monkeypatch.chdir(tmp_path)
    assert main(["--git"]) == 2
    assert capsys.readouterr().out == ""


def test_main_refuses_to_pass_a_scan_that_saw_no_files(tmp_path, capsys):
    """An empty file list passes every check, so it must not count as clean."""
    empty = _wheel(tmp_path / "empty.whl", {})
    assert main([str(empty)]) == 2
    assert capsys.readouterr().out == ""


def test_main_reports_an_artifact_it_cannot_read(tmp_path, capsys):
    assert main([str(tmp_path / "missing.whl")]) == 2
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# every internal import still resolves
# ---------------------------------------------------------------------------
#
# Removing a module leaves its importers behind. The guard's job is to keep the
# tree free of research material, and the cheapest way a removal goes wrong is a
# surviving ``from puremacro.<removed> import ...`` that nothing imports until a
# user does. tests/ is deliberately not scanned: a test that imports a removed
# module already fails loudly at collection, and the IO parity tests import a
# vendored puremacro from an external workspace on purpose.

_IMPORT_SCAN_ROOTS = ("puremacro/", "tools/", "playground/")
_IMPORT_SCAN_SKIP = ("playground/dist/",)


def _tracked_paths(repo: Path) -> frozenset[str]:
    raw = subprocess.run(["git", "ls-files", "-z"], cwd=repo, check=True,
                         capture_output=True).stdout
    return frozenset(os.fsdecode(p) for p in raw.split(b"\0") if p)


def _is_module(dotted: str, tracked: frozenset[str]) -> bool:
    base = dotted.replace(".", "/")
    return f"{base}/__init__.py" in tracked or f"{base}.py" in tracked


def _package_of(path: str) -> list[str]:
    """The package a file under puremacro/ lives in; a package's __init__ is it."""
    return path[: -len(".py")].split("/")[:-1]


def _check_imports(path: str, source: bytes, tracked: frozenset[str]) -> tuple[list[str], int]:
    """The ``puremacro`` imports in ``source`` that name no tracked module, and how
    many it checked. ``from M import Y`` resolves when ``M`` is a module (``Y`` may
    be a name it defines) or when ``M.Y`` is."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"{path}:{exc.lineno}: does not parse ({exc.msg})"], 0
    package = _package_of(path) if path.startswith("puremacro/") else None
    problems: list[str] = []
    checked = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] != "puremacro":
                    continue
                checked += 1
                if not _is_module(alias.name, tracked):
                    problems.append(f"{path}:{node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            shown = "." * node.level + (node.module or "")
            if node.level == 0:
                if (node.module or "").split(".")[0] != "puremacro":
                    continue
                base = node.module
            elif package is None:
                continue               # relative imports are resolved inside puremacro/ only
            else:
                up = node.level - 1
                if up >= len(package):
                    checked += 1
                    problems.append(f"{path}:{node.lineno}: from {shown} import ... "
                                    "(above the top-level package)")
                    continue
                base = ".".join(package[: len(package) - up] + ([node.module] if node.module else []))
            checked += 1
            if not (_is_module(base, tracked)
                    or any(_is_module(f"{base}.{a.name}", tracked) for a in node.names)):
                problems.append(f"{path}:{node.lineno}: from {shown} import ...")
    return sorted(problems), checked


_FAKE_TREE = frozenset({
    "puremacro/__init__.py",
    "puremacro/a/__init__.py",
    "puremacro/a/b.py",
    "puremacro/a/c.py",
    "puremacro/top.py",
})


@pytest.mark.parametrize("path, source, expected", [
    # absolute imports
    ("tools/t.py", "import puremacro", []),
    ("tools/t.py", "import puremacro.a.b", []),
    ("tools/t.py", "import puremacro.a.gone", ["tools/t.py:1: import puremacro.a.gone"]),
    ("tools/t.py", "from puremacro.a import b", []),            # a submodule
    ("tools/t.py", "from puremacro.a import a_name", []),       # a name the package defines
    ("tools/t.py", "from puremacro.gone import x", ["tools/t.py:1: from puremacro.gone import ..."]),
    ("tools/t.py", "import numpy\nfrom os import path", []),    # only puremacro is checked
    ("tools/t.py", "def f():\n    try:\n        import puremacro.a.gone\n    except ImportError:\n        pass",
     ["tools/t.py:3: import puremacro.a.gone"]),                # nested imports count
    # relative imports inside puremacro/
    ("puremacro/a/c.py", "from . import b", []),
    ("puremacro/a/c.py", "from .b import f", []),
    ("puremacro/a/c.py", "from .gone import f", ["puremacro/a/c.py:1: from .gone import ..."]),
    ("puremacro/a/c.py", "from .. import top", []),
    ("puremacro/a/c.py", "from ..top import f", []),
    ("puremacro/a/c.py", "from ..gone import f", ["puremacro/a/c.py:1: from ..gone import ..."]),
    ("puremacro/a/c.py", "from ... import x",
     ["puremacro/a/c.py:1: from ... import ... (above the top-level package)"]),
    ("puremacro/a/__init__.py", "from .b import f", []),        # an __init__ is its own package
    ("puremacro/top.py", "from . import a", []),
    ("puremacro/top.py", "from .. import a",
     ["puremacro/top.py:1: from .. import ... (above the top-level package)"]),
    ("tools/t.py", "from . import x", []),                      # not resolved outside puremacro/
    # a file that does not parse is a finding, not a skip
    ("tools/t.py", "def (:", ["tools/t.py:1: does not parse (invalid syntax)"]),
])
def test_the_import_resolver_flags_exactly_the_unresolvable_imports(path, source, expected):
    problems, _ = _check_imports(path, source.encode(), _FAKE_TREE)
    assert problems == expected


def test_the_import_resolver_counts_what_it_checks():
    source = b"import puremacro.a.b\nfrom . import x\nimport numpy\n"
    assert _check_imports("puremacro/a/c.py", source, _FAKE_TREE)[1] == 2


def test_every_internal_import_resolves():
    tracked = _tracked_paths(REPO)
    sources = sorted(p for p in tracked
                     if p.endswith(".py") and p.startswith(_IMPORT_SCAN_ROOTS)
                     and not p.startswith(_IMPORT_SCAN_SKIP))
    problems: list[str] = []
    scanned = checked = 0
    for rel in sources:
        try:
            source = (REPO / rel).read_bytes()
        except FileNotFoundError:
            continue                    # tracked, but deleted from the working tree mid-edit
        found, n = _check_imports(rel, source, tracked)
        problems += found
        checked += n
        scanned += 1
    assert scanned > 300 and checked > 500, (
        f"the scan is vacuous: {scanned} files, {checked} imports checked")
    assert not problems, (
        f"{len(problems)} import(s) name a module that is not tracked:\n  "
        + "\n  ".join(problems))
