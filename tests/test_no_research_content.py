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
import functools
import hashlib
import io
import os
import re
import subprocess
import tarfile
import zipfile
from collections.abc import Iterator
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


def test_the_prefix_list_is_exactly_the_generic_one():
    """A prefix that named a project would put the name in public code. The
    ``tools/run_<name>_*.py`` scripts are covered by the token in their names."""
    assert sorted(FORBIDDEN_PREFIXES) == sorted(_SPEC_PREFIXES)


@pytest.mark.parametrize("script", [
    f"tools/run_{SENTINEL_TOKEN}_phase3.py",
    f"tools/run_{SENTINEL_TOKEN}_lp.py",
    f"tools/run_{SENTINEL_TOKEN}.py",
])
def test_a_run_script_is_caught_by_the_token_in_its_name(script):
    """Word-group matching finds a token inside ``run_<token>_<rest>``, so no
    prefix is needed to forbid the scripts."""
    found = scan_entries([(script, b"")])
    assert _kinds(found) == ["hashed token"]
    assert found[0].detail == _digest(SENTINEL_TOKEN)[:12]


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


def _relative_base(package: list[str], level: int, module: str | None) -> str | None:
    """The dotted module a relative import names, or ``None`` when it climbs above
    the top-level package."""
    up = level - 1
    if up >= len(package):
        return None
    return ".".join(package[: len(package) - up] + ([module] if module else []))


def _literal_strings(value: ast.expr) -> Iterator[str]:
    """The strings in a literal list, tuple or set, or in a ``+`` of them."""
    if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
        for element in value.elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                yield element.value
    elif isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add):
        yield from _literal_strings(value.left)
        yield from _literal_strings(value.right)


def _assigned_names(target: ast.expr) -> Iterator[str]:
    """The names an assignment target binds, unpacking included. An attribute or
    an item binds none."""
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            yield from _assigned_names(element)
    elif isinstance(target, ast.Starred):
        yield from _assigned_names(target.value)


def _provided_names(tree: ast.Module, package: list[str]) -> frozenset[str] | None:
    """The names ``from <package> import name`` can get from the package's
    ``__init__``, or ``None`` when it can serve any name (a star import).

    Counted: bindings at module level, which includes the bodies of ``if``,
    ``try``, ``with``, ``for`` and ``while`` (they run at import time) but not
    function or class bodies; the strings of a literal ``__all__``; and, when the
    file defines a module-level ``__getattr__``, every string literal in it, since
    a lazy package lists what it serves. An import of the package from inside its
    own ``__init__`` is not evidence: ``from . import x`` binds ``x`` only if ``x``
    exists, and a leftover re-export of a removed submodule would vouch for itself.
    """
    this = ".".join(package)
    names: set[str] = set()
    lazy = False
    stack: list[ast.AST] = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            lazy = lazy or node.name == "__getattr__"
            continue                                    # its body is not module level
        if isinstance(node, ast.Import):
            names.update(a.asname or a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            origin = node.module if node.level == 0 else _relative_base(package, node.level, node.module)
            if origin != this:
                for alias in node.names:
                    if alias.name == "*":
                        return None
                    names.add(alias.asname or alias.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id == "__all__" and node.value is not None:
                    names.update(_literal_strings(node.value))
                if node.value is not None and not isinstance(node, ast.AugAssign):
                    names.update(_assigned_names(target))
        stack.extend(child for child in ast.iter_child_nodes(node)
                     if isinstance(child, (ast.stmt, ast.excepthandler, ast.match_case)))
    if lazy:
        names.update(c.value for c in ast.walk(tree)
                     if isinstance(c, ast.Constant) and isinstance(c.value, str))
    return frozenset(names)


@functools.lru_cache(maxsize=None)
def _init_names(root: Path, init: str) -> frozenset[str] | None:
    """``_provided_names`` for the ``__init__`` at ``init`` under ``root``. ``None``
    when the file cannot be read or parsed: that says nothing about a name (a
    missing file is skipped, a broken one is reported where it is scanned)."""
    try:
        tree = ast.parse((root / init).read_bytes())
    except (OSError, SyntaxError):
        return None
    return _provided_names(tree, init[: -len("/__init__.py")].split("/"))


def _unresolved_names(base: str, names: list[str], tracked: frozenset[str],
                      root: Path) -> list[str] | None:
    """The ``names`` that ``from <base> import <name>`` cannot find, or ``None`` when
    ``base`` is no module at all.

    A plain module may define any name, and ``*`` always resolves. In a package a
    name must be a submodule or something its ``__init__`` provides.
    """
    folder = base.replace(".", "/")
    init = f"{folder}/__init__.py"
    if f"{folder}.py" in tracked and init not in tracked:
        return []
    submodules = {name for name in names if _is_module(f"{base}.{name}", tracked)}
    if init not in tracked:                         # a namespace package, or nothing at all
        return [n for n in names if n not in submodules] if submodules else None
    missing = []
    for name in names:
        if name == "*" or name in submodules:
            continue
        provided = _init_names(root, init)
        if provided is not None and name not in provided:
            missing.append(name)
    return missing


def _check_imports(path: str, source: bytes, tracked: frozenset[str],
                   root: Path) -> tuple[list[str], int]:
    """The ``puremacro`` imports in ``source`` that name no tracked module and no
    name their package provides, and how many it checked.

    ``from M import Y`` resolves when ``M`` is a plain module (it may define ``Y``),
    or, for a package ``M``, when ``M.Y`` is a tracked module or ``M/__init__.py``
    provides ``Y`` (see ``_provided_names``). Package ``__init__`` files are read
    from ``root``.
    """
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
                base = _relative_base(package, node.level, node.module)
                if base is None:
                    checked += 1
                    problems.append(f"{path}:{node.lineno}: from {shown} import ... "
                                    "(above the top-level package)")
                    continue
            checked += 1
            unresolved = _unresolved_names(base, [a.name for a in node.names], tracked, root)
            if unresolved is None:
                problems.append(f"{path}:{node.lineno}: from {shown} import ...")
            else:
                problems.extend(f"{path}:{node.lineno}: from {shown} import {name}"
                                for name in unresolved)
    return sorted(problems), checked


def _write_tree(root: Path, files: dict[str, str]) -> frozenset[str]:
    """Write ``files`` under ``root`` and return their paths: a tiny tree to resolve against."""
    for rel, text in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return frozenset(files)


_FAKE_FILES = {
    "puremacro/__init__.py": "",
    "puremacro/a/__init__.py": "a_name = 1\n",
    "puremacro/a/b.py": "",
    "puremacro/a/c.py": "",
    "puremacro/top.py": "",
}


@pytest.mark.parametrize("path, source, expected", [
    # absolute imports
    ("tools/t.py", "import puremacro", []),
    ("tools/t.py", "import puremacro.a.b", []),
    ("tools/t.py", "import puremacro.a.gone", ["tools/t.py:1: import puremacro.a.gone"]),
    ("tools/t.py", "from puremacro.a import b", []),                  # a submodule
    ("tools/t.py", "from puremacro.a import a_name", []),             # a name the package binds
    ("tools/t.py", "from puremacro.a import gone", ["tools/t.py:1: from puremacro.a import gone"]),
    ("tools/t.py", "from puremacro.a import b, gone, a_name",         # each name is judged alone
     ["tools/t.py:1: from puremacro.a import gone"]),
    ("tools/t.py", "from puremacro.a import *", []),
    ("tools/t.py", "from puremacro.a.b import anything", []),         # a plain module may define any name
    ("tools/t.py", "from puremacro.gone import x", ["tools/t.py:1: from puremacro.gone import ..."]),
    ("tools/t.py", "import numpy\nfrom os import path", []),          # only puremacro is checked
    ("tools/t.py", "def f():\n    try:\n        import puremacro.a.gone\n    except ImportError:\n        pass",
     ["tools/t.py:3: import puremacro.a.gone"]),                      # nested imports count
    # relative imports inside puremacro/
    ("puremacro/a/c.py", "from . import b", []),
    ("puremacro/a/c.py", "from . import a_name", []),
    ("puremacro/a/c.py", "from . import gone", ["puremacro/a/c.py:1: from . import gone"]),
    ("puremacro/a/c.py", "from .b import f", []),
    ("puremacro/a/c.py", "from .gone import f", ["puremacro/a/c.py:1: from .gone import ..."]),
    ("puremacro/a/c.py", "from .. import top", []),
    ("puremacro/a/c.py", "from .. import gone", ["puremacro/a/c.py:1: from .. import gone"]),
    ("puremacro/a/c.py", "from ..top import f", []),
    ("puremacro/a/c.py", "from ..gone import f", ["puremacro/a/c.py:1: from ..gone import ..."]),
    ("puremacro/a/c.py", "from ... import x",
     ["puremacro/a/c.py:1: from ... import ... (above the top-level package)"]),
    ("puremacro/a/__init__.py", "from .b import f", []),              # an __init__ is its own package
    ("puremacro/top.py", "from . import a", []),
    ("puremacro/top.py", "from .. import a",
     ["puremacro/top.py:1: from .. import ... (above the top-level package)"]),
    ("tools/t.py", "from . import x", []),                            # not resolved outside puremacro/
    # a file that does not parse is a finding, not a skip
    ("tools/t.py", "def (:", ["tools/t.py:1: does not parse (invalid syntax)"]),
])
def test_the_import_resolver_flags_exactly_the_unresolvable_imports(tmp_path, path, source, expected):
    tracked = _write_tree(tmp_path, _FAKE_FILES)
    problems, _ = _check_imports(path, source.encode(), tracked, tmp_path)
    assert problems == expected


def test_the_import_resolver_counts_what_it_checks(tmp_path):
    tracked = _write_tree(tmp_path, _FAKE_FILES)
    source = b"import puremacro.a.b\nfrom . import x\nimport numpy\n"
    assert _check_imports("puremacro/a/c.py", source, tracked, tmp_path)[1] == 2


# ``from package import name`` needs the name to exist. Accepting it whenever the
# package exists would let a removed submodule through for as long as the package
# survives, which is the usual shape of a leftover import. Each case below writes
# a tiny ``puremacro/pkg/__init__.py`` and asks whether
# ``from puremacro.pkg import wanted`` resolves.

_WANTED_PROBLEM = ["tools/t.py:1: from puremacro.pkg import wanted"]


def _problems_for(root: Path, init: str, statement: str = "from puremacro.pkg import wanted") -> list[str]:
    tracked = _write_tree(root, {"puremacro/__init__.py": "", "puremacro/pkg/__init__.py": init + "\n"})
    return _check_imports("tools/t.py", statement.encode(), tracked, root)[0]


def test_a_removed_submodule_fails_and_a_name_the_package_binds_passes(tmp_path):
    tracked = _write_tree(tmp_path, {
        "puremacro/__init__.py": "",
        "puremacro/pkg/__init__.py": "def helper():\n    return 1\n",
        "puremacro/pkg/alive.py": "",
    })

    def check(statement):
        return _check_imports("tools/t.py", statement.encode(), tracked, tmp_path)[0]

    assert check("from puremacro.pkg import alive") == []          # a submodule that exists
    assert check("from puremacro.pkg import helper") == []         # a name pkg/__init__.py binds
    assert check("from puremacro.pkg import removed") == [         # a submodule that was removed
        "tools/t.py:1: from puremacro.pkg import removed"]


@pytest.mark.parametrize("init", [
    "wanted = 1",
    "first, (second, *wanted) = 1, (2, 3)",
    "wanted: int = 1",
    "def wanted(): ...",
    "async def wanted(): ...",
    "class wanted: ...",
    "import os as wanted",
    "import wanted.sub",
    "from os import wanted",
    "from os import path as wanted",
    "from .impl import wanted",                         # a re-export from a submodule
    "__all__ = ['wanted']",
    "__all__ = ('wanted',)",
    "__all__ = ['other'] + ['wanted']",
    "__all__ = []\n__all__ += ['wanted']",
    "__all__: list = ['wanted']",
    "try:\n    from os import path as wanted\nexcept ImportError:\n    wanted = None",
    "try:\n    import os\nexcept ImportError:\n    wanted = None",     # bound only in a handler
    "if True:\n    def wanted(): ...",
])
def test_every_kind_of_module_level_binding_counts(tmp_path, init):
    assert _problems_for(tmp_path, init) == []


@pytest.mark.parametrize("init", [
    "def f():\n    wanted = 1",                  # a function body is not module level
    "class C:\n    wanted = 1",                  # nor is a class body
    "def f():\n    import wanted",
    "# wanted = 1",
    "'wanted'",                                  # a string alone serves nothing
    "obj.wanted = 1",                            # an attribute or item target binds no name
    "table['wanted'] = 1",
    "wanted: int",                               # an annotation alone binds nothing
    "wanted += 1",                               # an augmented assignment needs an earlier binding
    "other = ['wanted']",                        # a list that is not __all__
    "__all__ = sorted(['wanted'])",              # an __all__ that is not a literal
    "_TABLE = {'wanted': 'impl'}",               # a lazy table without a module __getattr__
    "def other(): ...\n_TABLE = {'wanted': 'impl'}",                        # only __getattr__ makes it lazy
    "class C:\n    def __getattr__(self, name): ...\n_TABLE = {'wanted': 'impl'}",   # not a module __getattr__
])
def test_what_is_not_a_module_level_binding_does_not_count(tmp_path, init):
    assert _problems_for(tmp_path, init) == _WANTED_PROBLEM


def test_a_lazy_package_serves_the_names_its_file_lists(tmp_path):
    """A package with a module ``__getattr__`` serves names from a table in its own
    file (the narrative sources do: ``__all__`` is computed, not a literal)."""
    init = ("_LAZY = {'served': 'impl'}\n__all__ = sorted(_LAZY)\n\n\n"
            "def __getattr__(name):\n    raise AttributeError(name)\n")
    assert _problems_for(tmp_path, init, "from puremacro.pkg import served") == []
    assert _problems_for(tmp_path / "again", init, "from puremacro.pkg import wanted") == _WANTED_PROBLEM


def test_a_package_that_star_imports_can_provide_any_name(tmp_path):
    assert _problems_for(tmp_path, "from .impl import *") == []


def test_a_package_init_that_cannot_be_read_gives_no_verdict(tmp_path):
    unparseable = _write_tree(tmp_path, {"puremacro/__init__.py": "", "puremacro/pkg/__init__.py": "def (:\n"})
    statement = b"from puremacro.pkg import wanted\n"
    assert _check_imports("tools/t.py", statement, unparseable, tmp_path)[0] == []
    # ... but the broken file is reported where it is scanned
    assert _check_imports("puremacro/pkg/__init__.py", b"def (:\n", unparseable, tmp_path)[0]
    # tracked, yet absent from the working tree (deleted mid-edit)
    absent = tmp_path / "absent"
    absent.mkdir()
    tracked = frozenset({"puremacro/__init__.py", "puremacro/pkg/__init__.py"})
    assert _check_imports("tools/t.py", statement, tracked, absent)[0] == []


@pytest.mark.parametrize("statement", [
    "from . import wanted",
    "from . import wanted as wanted",
    "from puremacro.pkg import wanted",
    "from . import wanted\nfrom . import wanted",       # two of them do not vouch for each other
])
def test_a_package_cannot_vouch_for_a_name_it_imports_from_itself(tmp_path, statement):
    """``from . import x`` in ``pkg/__init__.py`` binds ``x`` only if ``x`` exists, so it
    is not evidence that it does. This is the shape of an ``__init__`` that still
    re-exports a removed submodule."""
    tracked = _write_tree(tmp_path, {"puremacro/__init__.py": "", "puremacro/pkg/__init__.py": statement + "\n"})
    own, _ = _check_imports("puremacro/pkg/__init__.py", (statement + "\n").encode(), tracked, tmp_path)
    other, _ = _check_imports("tools/t.py", b"from puremacro.pkg import wanted\n", tracked, tmp_path)
    assert len(own) == statement.count("import")
    assert other == _WANTED_PROBLEM


def test_a_directory_without_an_init_offers_only_its_submodules(tmp_path):
    tracked = _write_tree(tmp_path, {"puremacro/__init__.py": "", "puremacro/ns/one.py": ""})

    def check(statement):
        return _check_imports("tools/t.py", statement.encode(), tracked, tmp_path)[0]

    assert check("from puremacro.ns import one") == []
    assert check("from puremacro.ns import one, two") == ["tools/t.py:1: from puremacro.ns import two"]
    assert check("from puremacro.ns import two") == ["tools/t.py:1: from puremacro.ns import ..."]


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
        found, n = _check_imports(rel, source, tracked, REPO)
        problems += found
        checked += n
        scanned += 1
    assert scanned > 300 and checked > 500, (
        f"the scan is vacuous: {scanned} files, {checked} imports checked")
    # positive control for the name rule: the real package __init__ was read
    provided = _init_names(REPO, "puremacro/__init__.py")
    assert provided is not None and "__version__" in provided, "package __init__ files were not read"
    assert not problems, (
        f"{len(problems)} import(s) name a module or a name that is not there:\n  "
        + "\n  ".join(problems))
