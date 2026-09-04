"""Script-agnostic harness for executing a SLICE of any shell script under test.

Sibling-helper module, following the `systemd_unit_invariants.py` precedent that
tests/scripts/conftest.py explicitly supports: pyproject sets
``--import-mode=importlib``, under which pytest does NOT put a test file's own
directory on sys.path, so conftest.py inserts this directory for exactly this
kind of non-test helper.

WHY A SLICE. The scripts these tests cover are host-mutating installers and
migration tools — setup-host.sh installs systemd units and writes into $HOME;
export-data.sh / import-data.sh `systemctl stop` services, `docker compose stop`
containers, chown through a throwaway container and copy data trees. Running one
whole in a test is not an option, and re-deriving "just the interesting `if`"
per test file means duplicating a bash-quoting harness once per suite. So each
test group names the section it cares about by CODE anchors that delimit it and
gets back that text, verbatim, to run in a hermetic tmp tree.

Endpoints are DERIVED from markers, never pinned line numbers, so a slice
follows a reflow of its block instead of silently shifting off it.

MARKERS ARE CODE, NOT COMMENT PROSE — enforced, not merely stated: every marker
is located on a non-comment line (see `find_in_code`). Anchoring on a section
COMMENT instead would make CI red for a reworded comment or a fixed typo — zero
behavioural change — and a comment that quotes an anchor is prose about the
code, not the code.

WHY THIS MODULE EXISTS SEPARATELY from tests/scripts/setup_host_sections.py.
The slicer was generalized out of tests/scripts/test_check_orchestrator_unit_parity.py
(task 3424) and migrated onto setup_host_sections.py by task 3909, at which point
it was shared by four parity suites but still hardcoded SETUP_HOST_PATH. Task
4488 needed the same primitives for export-data.sh and import-data.sh, so the
script-agnostic core moved HERE parameterized by script path, and
setup_host_sections.py became a thin binder whose public API is unchanged — its
four consumer suites need no edits and act as the regression net for the move.
The alternative was a verbatim second copy of ~100 lines of slicer plus a second
copy of the `| grep -q` detector and its eleven-case guard, which is exactly the
drift class this repo keeps paying to remove.
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess

REPO_ROOT = pathlib.Path(__file__).parents[2]


def stub_bin_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """The (created) directory `run_with_preamble` prepends to PATH for *tmp_path*.

    THE PATH LITERAL LIVES HERE, ONCE. Callers that need to drop their own
    stubs alongside a harness's own previously re-derived ``tmp_path / "stub-bin"``
    and depended on this module's private choice by string equality — a rename
    here would silently drop their stubs off PATH. Going through this accessor
    makes that coupling an import instead.
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


def find_in_code(text: str, marker: str, *, start: int = 0) -> int:
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
    script_path: pathlib.Path,
    start_marker: str,
    end_marker: str,
    *,
    end_after: str | None = None,
) -> str:
    """Return *script_path* from the line carrying *start_marker* through *end_marker*.

    The slice runs from the START of the line containing the first instance of
    *start_marker* through the END of the line containing the first
    *end_marker* at or after it — both endpoints derived, so the slice survives
    a reflow of the block.

    Every marker is located on a NON-COMMENT line (see `find_in_code`): a
    comment that quotes an anchor is prose about the code, not the code.

    *end_after* is an optional THIRD anchor. When given, the search for
    *end_marker* begins at it rather than at *start_marker*, so a slice can be
    made to run THROUGH an inner construct that closes with the same token —
    the orchestrator installer slice must end at the column-0 `fi` closing the
    INSTALL construct, not at the gate's own, which is the first one after the
    start. (Task 4488 MEASURED the same hazard twice more: import-data.sh's
    section-1 slice stops at the `fi` of the preceding `systemctl is-active`
    block and never reaches the docker site without it.)

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
    text = script_path.read_text(encoding="utf-8")

    pos = find_in_code(text, start_marker)
    assert pos != -1, (
        f"start_marker {start_marker!r} not found in {script_path} on a "
        f"non-comment line. A renamed anchor must fail here, not slice an "
        f"empty region."
    )

    start = text.rfind("\n", 0, pos) + 1

    search_from = pos
    if end_after is not None:
        after_pos = find_in_code(text, end_after, start=pos)
        assert after_pos != -1, (
            f"end_after {end_after!r} not found in {script_path} on a "
            f"non-comment line at or after {start_marker!r}."
        )
        search_from = after_pos

    end_pos = text.find(end_marker, search_from)
    # Names whichever anchor the search actually started from, so the message
    # points at the region that was searched rather than at a marker that was
    # found.
    assert end_pos != -1, (
        f"end_marker {end_marker!r} not found in {script_path} at or after "
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


def slice_shell_function(script_path: pathlib.Path, name: str) -> str:
    """Return *script_path*'s ``name() {`` ... column-0 ``}`` definition, verbatim.

    A slice that CALLS a helper defined elsewhere in the file dies with exit
    127 under a preamble that knows only the logging shims. Prepending the REAL
    definition is what keeps such a section runnable WITHOUT giving up what
    these tests are for: defining a copy of the helper in the preamble instead
    would make every assertion downstream a claim about the harness's own bash,
    green no matter what the shipped helper does — the same "verdict
    manufactured by the mechanism" failure a behavioural test exists to catch.

    Both endpoints are derived, as in `slice_section`: the header line, and the
    first column-0 ``}`` at or after it. A helper whose body ever grew a
    column-0 ``}`` of its own would slice short and fail LOUDLY under `bash`,
    not silently.
    """
    return slice_section(script_path, f"{name}() {{", "\n}\n")


def run_with_preamble(
    tmp_path: pathlib.Path,
    preamble: str,
    section_text: str,
    *,
    env_extra: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Execute *preamble* + *section_text* under bash in a hermetic tmp tree.

    `stub_bin_dir(tmp_path)` is PREPENDED to PATH, so any stubs a caller wrote
    there shadow the real binaries. Writing those stubs is the CALLER's job —
    this function installs none of its own, because which commands a section
    calls (and what each must do to keep the slice runnable under `set -e`) is
    a property of the script being sliced, not of the runner.
    """
    stub_bin = stub_bin_dir(tmp_path)

    script = tmp_path / "section.sh"
    script.write_text(preamble + section_text, encoding="utf-8")

    env = dict(os.environ)
    env["PATH"] = f"{stub_bin}:{env.get('PATH', '')}"
    env.update(env_extra or {})
    return subprocess.run(
        ["bash", str(script)], capture_output=True, text=True, env=env
    )


# --- the `| grep -q` detector ----------------------------------------------
# Shared by every file-scoped sweep that forbids the construct (currently
# test_setup_host_probe_pipelines.py and test_script_probe_pipelines.py), so
# the three regexes below exist ONCE. Its guard-the-guard —
# test_setup_host_probe_pipelines.py::test_the_grep_q_sweep_detects_a_planted_pipeline,
# seven planted spellings plus four must-not-match cases — stays in that suite
# and now guards the copy BOTH consumers use.

# A grep on the receiving end of a pipe, plus its arguments up to the end of
# THAT command: `[^|;&)]*` stops at the next pipeline stage, at a `;` or `&&`,
# and at the close of a command substitution, so a `-q` belonging to some later
# command on the same line is never read as this grep's.
_GREP_PIPE = re.compile(r"\|\s*grep\s+(?P<args>[^|;&)]*)")

# Every spelling of "exit on the first match and close the read end": the short
# clusters (`-q`, `-qF`, `-Fq`, `-iq`) and GNU's long forms. Matched against
# whole TOKENS rather than positionally, which is what lets a flag taking an
# argument sit in between — `grep -e PONG --quiet` is the same defect as
# `grep -q PONG` and the sweep must see both. Deliberately does NOT match a
# bare `| grep -F`, which reads its input to the end and cannot SIGPIPE the
# producer.
_QUIET_FLAG = re.compile(r"-[A-Za-z]*q[A-Za-z]*|--quiet|--silent")


def _pipes_into_quiet_grep(line):
    """True when *line* feeds a producer into a grep that exits on first match."""
    return any(
        any(_QUIET_FLAG.fullmatch(token) for token in match.group("args").split())
        for match in _GREP_PIPE.finditer(line)
    )


def grep_q_offenders(source):
    """Every non-comment line of *source* piping a producer into a quiet grep."""
    return [
        (n, line)
        for n, line in enumerate(source.splitlines(), start=1)
        if not line.strip().startswith("#") and _pipes_into_quiet_grep(line)
    ]
