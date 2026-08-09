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
follows a reflow of its block instead of silently shifting off it. Markers are
deliberately the section COMMENTS rather than the checker filenames: a checker
name occurs more than once in this file (invocation + the warn text that names
the remediation command), so a filename-anchored slice is ambiguous, and each
gate fix adds a further occurrence via its `_..._script=` assignment.

NOTHING HERE TOUCHES REAL SYSTEMD. `repo_root` and `unit_dir` are always
tmp_path trees supplied by the caller, and `systemctl` is always a PATH stub
that exits 0 — the sliced sections do call `systemctl --user enable`.

Generalized from the reference implementation at
tests/scripts/test_check_orchestrator_unit_parity.py:1044-1119 (task 3424).
That file deliberately still carries its own copy: it is the just-landed
reference for this pattern, and rewriting it to import this module would be
churn and merge conflict for no behavioural gain.
"""

from __future__ import annotations

import os
import pathlib
import subprocess

SETUP_HOST_PATH = pathlib.Path(__file__).parents[2] / "scripts" / "setup-host.sh"

# The preamble every slice needs: setup-host.sh's own `set` flags and the four
# logging shims, reduced to PLAIN TEXT so assertions can match on prefixes
# without ANSI escapes. Prefixes mirror the reference harness.
_PREAMBLE = (
    "set -euo pipefail\n"
    'REPO_ROOT="{repo_root}"\n'
    'UNIT_DIR="{unit_dir}"\n'
    'mkdir -p "$UNIT_DIR"\n'
    "info()  { printf '==> %s\\n' \"$*\"; }\n"
    "ok()    { printf 'OK %s\\n' \"$*\"; }\n"
    "warn()  { printf 'WARN %s\\n' \"$*\"; }\n"
    "fail()  { printf 'FAIL %s\\n' \"$*\"; }\n"
)


def setup_host_text() -> str:
    """The full text of scripts/setup-host.sh."""
    return SETUP_HOST_PATH.read_text(encoding="utf-8")


def slice_section(
    start_marker: str, end_marker: str, *, occurrence: int = 0
) -> str:
    """Return setup-host.sh from the line carrying *start_marker* through *end_marker*.

    The slice runs from the START of the line containing the ``occurrence``-th
    (0-based) instance of *start_marker* through the END of the line containing
    the first *end_marker* at or after it — both endpoints derived, so the slice
    survives a reflow of the block.

    Raises AssertionError NAMING the missing marker when either is absent. That
    matters: the silent alternative is a slice of the wrong (or empty) region,
    which runs cleanly and produces a vacuously green test — the same
    "reported green because it never ran" failure these tests exist to catch.
    """
    text = setup_host_text()

    pos = -1
    for n in range(occurrence + 1):
        pos = text.find(start_marker, pos + 1)
        assert pos != -1, (
            f"start_marker {start_marker!r} (occurrence {occurrence}) not found "
            f"in {SETUP_HOST_PATH} — only {n} occurrence(s) present. A renamed "
            f"section comment must fail here, not slice an empty region."
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
        _PREAMBLE.format(repo_root=repo_root, unit_dir=unit_dir) + section_text,
        encoding="utf-8",
    )

    env = dict(os.environ)
    env["PATH"] = f"{stub_bin}:{env.get('PATH', '')}"
    env.update(env_extra or {})
    return subprocess.run(
        ["bash", str(script)], capture_output=True, text=True, env=env
    )
