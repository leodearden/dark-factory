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
one preamble and one stub.

THIS MODULE IS NOW A THIN BINDER. Task 4488 needed the same slicer for
export-data.sh and import-data.sh, so the script-agnostic core — `find_in_code`,
`slice_section`, `slice_shell_function`, `stub_bin_dir`, `write_stub` and the
generic runner — moved to tests/scripts/shell_sections.py parameterized by
script path. What stays here is exactly what is SETUP-HOST-SPECIFIC: the path
binding, the four log shims, the `_parity_verdict` preamble, the recording
`systemctl` stub, and the checker/unit helpers. The public API is unchanged
byte-for-byte, so this module's four consumer suites needed no edits and are
the regression net for that move.

`stub_bin_dir` and `write_stub` are RE-EXPORTED rather than re-implemented: the
stub directory's PATH literal is owned once, in shell_sections. A caller that
drops stubs of its own alongside the harness's `systemctl` imports that
accessor — from either module, they are the same function — instead of
re-deriving ``tmp_path / "stub-bin"`` and depending on a private choice by
string equality.
"""

from __future__ import annotations

import pathlib
import subprocess
from collections.abc import Iterable

from shell_sections import (
    REPO_ROOT,
    run_with_preamble,
    stub_bin_dir,
    write_stub,
)
from shell_sections import slice_section as _slice_section
from shell_sections import slice_shell_function as _slice_shell_function

SETUP_HOST_PATH = REPO_ROOT / "scripts" / "setup-host.sh"

# Re-exported so `from setup_host_sections import stub_bin_dir, write_stub`
# keeps resolving for this module's consumers; they are shell_sections' own
# functions, not copies.
__all__ = [
    "REPO_ROOT",
    "SETUP_HOST_PATH",
    "SYSTEMCTL_LOG",
    "checker_repo",
    "enabled_units",
    "run_section",
    "setup_host_text",
    "slice_section",
    "slice_shell_function",
    "stub_bin_dir",
    "systemctl_calls",
    "usage_error_checker",
    "write_checker",
    "write_stub",
]

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
# Named, not spelled out as start/end markers: `slice_shell_function` already
# derives both endpoints of a shell definition from the name, and a second
# hand-written pair here would be the same slice expressed twice.
_VERDICT_HELPER = "_parity_verdict"


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

    Slicing also fails LOUDLY (`slice_section`, under `slice_shell_function`,
    asserts naming the marker) if the helper is ever renamed, rather than
    leaving the suite testing a helper the installer no longer has.
    """
    return (
        "set -euo pipefail\n"
        f'REPO_ROOT="{repo_root}"\n'
        f'UNIT_DIR="{unit_dir}"\n'
        'mkdir -p "$UNIT_DIR"\n'
    ) + _SHIMS + slice_shell_function(_VERDICT_HELPER)


def setup_host_text() -> str:
    """The full text of scripts/setup-host.sh."""
    return SETUP_HOST_PATH.read_text(encoding="utf-8")


def slice_section(
    start_marker: str, end_marker: str, *, end_after: str | None = None
) -> str:
    """`shell_sections.slice_section` bound to setup-host.sh — see it for the rules.

    Endpoints derived from CODE anchors (never line numbers), *end_after* an
    optional third anchor, and a missing marker raises an AssertionError naming
    it rather than silently slicing the wrong region.
    """
    return _slice_section(
        SETUP_HOST_PATH, start_marker, end_marker, end_after=end_after
    )


def slice_shell_function(name: str) -> str:
    """`shell_sections.slice_shell_function` bound to setup-host.sh.

    Lifts the SHIPPED definition of ``name``, so a slice that calls it exercises
    the installer's own helper rather than a copy this harness wrote.
    """
    return _slice_shell_function(SETUP_HOST_PATH, name)


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
    write_stub(
        stub_bin_dir(tmp_path),
        "systemctl",
        f"printf '%s\\n' \"$*\" >> {tmp_path / SYSTEMCTL_LOG}\nexit 0\n",
    )
    return run_with_preamble(
        tmp_path,
        _preamble(repo_root, unit_dir),
        section_text,
        env_extra=env_extra,
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
