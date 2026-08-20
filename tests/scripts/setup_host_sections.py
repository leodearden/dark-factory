"""Shared harness for executing a SLICE of scripts/setup-host.sh under test.

Sibling-helper module, following the `systemd_unit_invariants.py` precedent that
tests/scripts/conftest.py explicitly supports: pyproject sets
``--import-mode=importlib``, under which pytest does NOT put a test file's own
directory on sys.path, so conftest.py inserts this directory for exactly this
kind of non-test helper.

WHY A SLICE. setup-host.sh is a 700-line host bootstrap that installs systemd
units, starts containers and writes into $HOME. Running it whole in a test is
not an option, and re-deriving "just the interesting `if`" per test file means
triplicating a bash-quoting harness three ways. So each test group names the
section it cares about by the COMMENTS that delimit it and gets back that text,
verbatim, to run in a hermetic tmp tree.

Endpoints are DERIVED from markers, never pinned line numbers, so a slice
follows a reflow of its block instead of silently shifting off it.

MARKERS ARE CODE, NOT COMMENT PROSE — enforced, not merely stated: every
marker is located on a non-comment line. Each parity block hoists a uniquely-named
`_<gate>_parity_script="$REPO_ROOT/scripts/check_<x>_unit_parity.py"`
assignment at its top, and that line is the anchor. Anchoring on the section
comment instead would make CI red for a reworded comment or a fixed typo — zero
behavioural change — and anchoring on the bare checker FILENAME is ambiguous,
because a name occurs more than once in this file (the assignment plus the warn
text naming the remediation command). The assignment line is unique per site
and is exactly what the structural sweep in test_check_dashboard_unit_parity.py
keys on, so both mechanisms share one anchor.

NOTHING HERE TOUCHES REAL SYSTEMD. `repo_root` and `unit_dir` are always
tmp_path trees supplied by the caller, and `systemctl` is always a PATH stub
that exits 0 — the sliced sections do call `systemctl --user enable`. That stub
also RECORDS its argv into tmp_path, readable via `systemctl_calls` /
`enabled_units`, so the enable half of an install is observable rather than
merely assumed. Always, with no opt-in flag: a caller that never reads the log
is unaffected, and a flag would give the harness two behaviours to reason about
while letting a future caller silently lose the observability.

Generalized from the reference implementation in
tests/scripts/test_check_orchestrator_unit_parity.py (task 3424) and migrated
onto this module by task 3909, so all four parity suites now share one slicer,
one preamble and one stub. The stub directory's own PATH literal is owned here
too, by `stub_bin_dir`: a caller that drops stubs of its own alongside the
harness's `systemctl` imports that accessor instead of re-deriving
``tmp_path / "stub-bin"`` and depending on this module's private choice by
string equality.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
from collections.abc import Iterable

REPO_ROOT = pathlib.Path(__file__).parents[2]
SETUP_HOST_PATH = REPO_ROOT / "scripts" / "setup-host.sh"

# The four logging shims, reduced to PLAIN TEXT so assertions can match on
# prefixes without ANSI escapes. Prefixes mirror the reference harness.
#
# Deliberately NOT a str.format() template: these bodies are bash brace groups,
# and every `{ printf ... }` in them would be read as a replacement field.
_SHIMS = (
    "info()  { printf '==> %s\\n' \"$*\"; }\n"
    "ok()    { printf 'OK %s\\n' \"$*\"; }\n"
    "warn()  { printf 'WARN %s\\n' \"$*\"; }\n"
    "fail()  { printf 'FAIL %s\\n' \"$*\"; }\n"
)


# Where the systemctl stub appends one line per invocation, relative to the
# caller's tmp_path.
SYSTEMCTL_LOG = "systemctl-calls.log"


# setup-host.sh defines `_parity_verdict` once, below the log shims and above
# every parity call site, so a sliced block that calls it needs it in scope.
_VERDICT_HELPER_START = "_parity_verdict() {"
_VERDICT_HELPER_END = "\n}\n"


def _preamble(repo_root: pathlib.Path, unit_dir: pathlib.Path) -> str:
    """setup-host.sh's own `set` flags and variables, the shims, and the verdict helper.

    The helper is SLICED LIVE out of setup-host.sh, never carried here as a
    hand-written copy — unlike the four log shims above, which are deliberately
    reduced to plain text. The shims are reduced for a stated reason (stripping
    ANSI so assertions can match on prefixes) and their bodies are trivial
    `printf`s with no logic to drift. `_parity_verdict` IS the logic under
    test: a copied body would let the version the suite exercises and the
    version setup-host.sh ships diverge silently — which is precisely the
    "reports green because it never ran" class this whole gate family exists to
    catch, reproduced one level up in its own harness.

    Slicing also fails LOUDLY (slice_section asserts, naming the marker) if the
    helper is ever renamed, rather than leaving the suite testing a helper the
    installer no longer has.
    """
    return (
        "set -euo pipefail\n"
        f'REPO_ROOT="{repo_root}"\n'
        f'UNIT_DIR="{unit_dir}"\n'
        'mkdir -p "$UNIT_DIR"\n'
    ) + _SHIMS + slice_section(_VERDICT_HELPER_START, _VERDICT_HELPER_END)


def setup_host_text() -> str:
    """The full text of scripts/setup-host.sh."""
    return SETUP_HOST_PATH.read_text(encoding="utf-8")


def stub_bin_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """The (created) directory `run_section` prepends to PATH for *tmp_path*.

    THE PATH LITERAL LIVES HERE, ONCE. Callers that need to drop their own
    stubs alongside the harness's `systemctl` previously re-derived
    ``tmp_path / "stub-bin"`` and depended on this module's private choice by
    string equality — a rename here would silently drop their stubs off PATH.
    Going through this accessor makes that coupling an import instead.
    """
    stub_bin = tmp_path / "stub-bin"
    stub_bin.mkdir(exist_ok=True)
    return stub_bin


def write_stub(stub_bin: pathlib.Path, name: str, body: str) -> pathlib.Path:
    """Drop an executable bash stub *name* carrying *body* into *stub_bin*."""
    path = stub_bin / name
    path.write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path


def _find_in_code(text: str, marker: str, *, start: int = 0) -> int:
    """Index of the first occurrence of *marker* on a NON-COMMENT line.

    Enforces this module's "MARKERS ARE CODE, NOT COMMENT PROSE" rule rather
    than merely stating it, and matches the discovery rule the structural sweep
    in test_check_dashboard_unit_parity.py::_parity_call_sites already applies
    (`line.lstrip().startswith("#")`).

    Not cosmetic. MEASURED before this existed: a plain `text.find` for
    `_orch_parity_script=` landed on setup-host.sh's own harness-constraint
    COMMENT, which quotes the anchor, and the resulting slice reached back over
    189 lines of real installer code — including an `install -m 0755` writing
    into `$HOME`. These slices are EXECUTED, so that is a test running against
    the developer's real home directory.

    Returns -1 when *marker* appears only in comments (or not at all), so the
    caller raises its own self-naming AssertionError.
    """
    pos = text.find(marker, start)
    while pos != -1:
        line_start = text.rfind("\n", 0, pos) + 1
        if not text[line_start:pos].lstrip().startswith("#"):
            return pos
        pos = text.find(marker, pos + 1)
    return -1


def slice_section(
    start_marker: str, end_marker: str, *, end_after: str | None = None
) -> str:
    """Return setup-host.sh from the line carrying *start_marker* through *end_marker*.

    The slice runs from the START of the line containing the first instance of
    *start_marker* through the END of the line containing the first
    *end_marker* at or after it — both endpoints derived, so the slice survives
    a reflow of the block.

    Every marker is located on a NON-COMMENT line (see `_find_in_code`): a
    comment that quotes an anchor is prose about the code, not the code.

    *end_after* is an optional THIRD anchor. When given, the search for
    *end_marker* begins at it rather than at *start_marker*, so a slice can be
    made to run THROUGH an inner construct that closes with the same token —
    the orchestrator installer slice must end at the column-0 `fi` closing the
    INSTALL construct, not at the gate's own, which is the first one after the
    start.

    Deliberately an ANCHOR rather than the counted `occurrence` parameter task
    3557 deleted as dead. "The second `fi`" is a number that shifts silently
    the moment the block is reflowed — re-pointing the slice at a region nobody
    chose — whereas a marker that moves out from under the slice fails loudly,
    which is the same reason 3557 removed the counted form.

    Raises AssertionError NAMING the missing marker when any is absent. That
    matters: the silent alternative is a slice of the wrong (or empty) region,
    which runs cleanly and produces a vacuously green test — the same
    "reported green because it never ran" failure these tests exist to catch.
    """
    text = setup_host_text()

    pos = _find_in_code(text, start_marker)
    assert pos != -1, (
        f"start_marker {start_marker!r} not found in {SETUP_HOST_PATH} on a "
        f"non-comment line. A renamed anchor must fail here, not slice an "
        f"empty region."
    )

    start = text.rfind("\n", 0, pos) + 1

    search_from = pos
    if end_after is not None:
        after_pos = _find_in_code(text, end_after, start=pos)
        assert after_pos != -1, (
            f"end_after {end_after!r} not found in {SETUP_HOST_PATH} on a "
            f"non-comment line at or after {start_marker!r}."
        )
        search_from = after_pos

    end_pos = text.find(end_marker, search_from)
    # Names whichever anchor the search actually started from, so the message
    # points at the region that was searched rather than at a marker that was
    # found.
    assert end_pos != -1, (
        f"end_marker {end_marker!r} not found in {SETUP_HOST_PATH} at or after "
        f"{end_after if end_after is not None else start_marker!r}."
    )
    # Search for the line end from the marker's LAST character, not its first.
    # An end_marker may itself span lines (`"\nfi\n"` is the natural way to name
    # a column-0 `fi` without also matching an indented inner one); starting the
    # search at end_pos would then land on the marker's own leading newline and
    # cut the slice one line short — dropping the very `fi` it was asked for.
    marker_last = end_pos + len(end_marker) - 1
    line_end = text.find("\n", marker_last)
    end = len(text) if line_end == -1 else line_end + 1

    return text[start:end]


def slice_shell_function(name: str) -> str:
    """Return setup-host.sh's ``name() {`` ... column-0 ``}`` definition, verbatim.

    A slice that CALLS a helper defined elsewhere in the file dies with exit
    127 under the preamble, which knows only the four logging shims. Prepending
    the REAL definition is what keeps such a section runnable WITHOUT giving up
    what these tests are for: defining a copy of the helper in `_preamble`
    instead would make every assertion downstream a claim about the harness's
    own bash, green no matter what the shipped helper does — the same
    "verdict manufactured by the mechanism" failure a behavioural test exists
    to catch.

    Both endpoints are derived, as in `slice_section`: the header line, and the
    first column-0 ``}`` at or after it. A helper whose body ever grew a
    column-0 ``}`` of its own would slice short and fail LOUDLY under `bash`,
    not silently.
    """
    return slice_section(f"{name}() {{", "\n}\n")


def run_section(
    tmp_path: pathlib.Path,
    section_text: str,
    *,
    repo_root: pathlib.Path,
    unit_dir: pathlib.Path,
    env_extra: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Execute *section_text* under bash with setup-host.sh's own preamble.

    A stub `systemctl` is written into a tmp dir and PREPENDED to PATH, so a
    slice containing `systemctl --user enable` neither touches the host nor
    fails under `set -e`. It RECORDS its argv (one call per line) into
    ``tmp_path / SYSTEMCTL_LOG`` before exiting 0 — see the module docstring
    for why that is unconditional.
    """
    stub_bin = stub_bin_dir(tmp_path)
    write_stub(
        stub_bin,
        "systemctl",
        f"printf '%s\\n' \"$*\" >> {tmp_path / SYSTEMCTL_LOG}\nexit 0\n",
    )

    script = tmp_path / "section.sh"
    script.write_text(
        _preamble(repo_root, unit_dir) + section_text,
        encoding="utf-8",
    )

    env = dict(os.environ)
    env["PATH"] = f"{stub_bin}:{env.get('PATH', '')}"
    env.update(env_extra or {})
    return subprocess.run(
        ["bash", str(script)], capture_output=True, text=True, env=env
    )


def systemctl_calls(tmp_path: pathlib.Path) -> list[list[str]]:
    """Every `systemctl` invocation the run made, as argv token lists."""
    log = tmp_path / SYSTEMCTL_LOG
    if not log.is_file():
        return []
    return [
        line.split() for line in log.read_text(encoding="utf-8").splitlines() if line
    ]


def enabled_units(tmp_path: pathlib.Path) -> list[str]:
    """The units passed to `systemctl ... enable <unit>` during the run.

    Token-matched rather than substring-matched: `enable` naming one unit must
    never be satisfied by a line naming a different one.
    """
    enabled: list[str] = []
    for argv in systemctl_calls(tmp_path):
        if "enable" in argv:
            enabled.extend(argv[argv.index("enable") + 1 :])
    return enabled


def usage_error_checker(script_name: str, usage_flags: str, rejected: str) -> str:
    """A stub checker body shaped like argparse rejecting a RENAMED flag.

    One of the two ways a parity checker exits 2 without having checked
    anything (the other is `python3` refusing to open a script that was renamed
    or moved). Its stderr deliberately carries bracketed tokens — `[-h]`,
    `[--fix]` — so that a gate matching brackets LOOSELY rather than matching
    its checker's specific `[<tag>]` would read those as a report and hand the
    gate a verdict the checker never gave.

    (That hazard used to be worded as "a marker match that is not
    line-anchored". No gate is line-anchored any more — all five now test
    containment of one specific tag — but the stub is still exactly the right
    imposter, for the reason above: it emits no tag at all.)
    """
    return (
        "import sys\n"
        f"sys.stderr.write('usage: {script_name} {usage_flags}\\n"
        f"error: unrecognized arguments: {rejected}\\n')\n"
        "sys.exit(2)\n"
    )


def write_checker(
    repo_root: pathlib.Path,
    filename: str,
    *,
    body: str | None = None,
    siblings: Iterable[str] = (),
) -> pathlib.Path:
    """Put a parity checker at ``repo_root/scripts/<filename>``.

    With *body* None the REAL checker is copied out of the repo (plus any
    *siblings* it imports), so the gate under test drives the real one and only
    the TREE is fake. With *body* set, that text is written instead — the stub
    path used to simulate a checker that exits without reporting.
    """
    scripts = repo_root / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    target = scripts / filename
    if body is not None:
        target.write_text(body, encoding="utf-8")
        return target
    for name in (filename, *siblings):
        (scripts / name).write_text(
            (REPO_ROOT / "scripts" / name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    return target


def checker_repo(
    tmp_path: pathlib.Path,
    filename: str,
    *,
    body: str | None = None,
    siblings: Iterable[str] = (),
    with_checker: bool = True,
) -> pathlib.Path:
    """A minimal tmp REPO_ROOT holding only ``scripts/`` and maybe the checker.

    Enough for any gate block whose only repo-side dependency is the checker
    itself. Callers needing more of the tree (committed units, a service
    template) build their own root and call `write_checker` on it.
    """
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True, exist_ok=True)
    if with_checker:
        write_checker(repo, filename, body=body, siblings=siblings)
    return repo
