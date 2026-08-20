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

MARKERS ARE CODE, NOT COMMENT PROSE. Each parity block hoists a uniquely-named
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
that exits 0 — the sliced sections do call `systemctl --user enable`.

Generalized from the reference implementation at
tests/scripts/test_check_orchestrator_unit_parity.py:1044-1119 (task 3424).
That file deliberately still carries its own copy: migrating it needs an edit
to a file this task holds no lock on. See the amendment note in the commit that
introduced this paragraph.
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


def _preamble(repo_root: pathlib.Path, unit_dir: pathlib.Path) -> str:
    """setup-host.sh's own `set` flags and variables, plus the plain-text shims."""
    return (
        "set -euo pipefail\n"
        f'REPO_ROOT="{repo_root}"\n'
        f'UNIT_DIR="{unit_dir}"\n'
        'mkdir -p "$UNIT_DIR"\n'
    ) + _SHIMS


def setup_host_text() -> str:
    """The full text of scripts/setup-host.sh."""
    return SETUP_HOST_PATH.read_text(encoding="utf-8")


def slice_section(start_marker: str, end_marker: str) -> str:
    """Return setup-host.sh from the line carrying *start_marker* through *end_marker*.

    The slice runs from the START of the line containing the first instance of
    *start_marker* through the END of the line containing the first
    *end_marker* at or after it — both endpoints derived, so the slice survives
    a reflow of the block.

    Raises AssertionError NAMING the missing marker when either is absent. That
    matters: the silent alternative is a slice of the wrong (or empty) region,
    which runs cleanly and produces a vacuously green test — the same
    "reported green because it never ran" failure these tests exist to catch.
    """
    text = setup_host_text()

    pos = text.find(start_marker)
    assert pos != -1, (
        f"start_marker {start_marker!r} not found in {SETUP_HOST_PATH}. A "
        f"renamed anchor must fail here, not slice an empty region."
    )

    start = text.rfind("\n", 0, pos) + 1

    end_pos = text.find(end_marker, pos)
    assert end_pos != -1, (
        f"end_marker {end_marker!r} not found in {SETUP_HOST_PATH} at or after "
        f"{start_marker!r}."
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


def run_section(
    tmp_path: pathlib.Path,
    section_text: str,
    *,
    repo_root: pathlib.Path,
    unit_dir: pathlib.Path,
    env_extra: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Execute *section_text* under bash with setup-host.sh's own preamble.

    A stub `systemctl` that exits 0 is written into a tmp dir and PREPENDED to
    PATH, so a slice containing `systemctl --user enable` neither touches the
    host nor fails under `set -e`.
    """
    stub_bin = tmp_path / "stub-bin"
    stub_bin.mkdir(exist_ok=True)
    systemctl = stub_bin / "systemctl"
    systemctl.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    systemctl.chmod(0o755)

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


def usage_error_checker(script_name: str, usage_flags: str, rejected: str) -> str:
    """A stub checker body shaped like argparse rejecting a RENAMED flag.

    One of the two ways a parity checker exits 2 without having checked
    anything (the other is `python3` refusing to open a script that was renamed
    or moved). Its stderr deliberately carries bracketed tokens — `[-h]`,
    `[--fix]` — because a marker match that is not line-anchored would read
    those as a report and hand the gate a verdict the checker never gave.
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
