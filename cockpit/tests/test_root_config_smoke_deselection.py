"""Repo-root pytest config must deselect `smoke` and register its marker (PRD C-smoke).

WHY THIS EXISTS: pytest reads only ONE [tool.pytest.ini_options] -- the
rootdir's single inifile -- and never merges addopts/markers across
pyproject.toml files. A bare `pytest`/`pytest .` from the repo root sets
rootdir=repo root, so the ROOT pyproject.toml is the effective config and
cockpit/pyproject.toml's `-m 'not smoke'` + `smoke` marker are silently
ignored; on a live DISPLAY=:0 desktop that routine run would COLLECT AND RUN
the smoke tests and spawn real X11 windows/tmux sessions, and the autouse
`_require_live_host` skip (smoke/conftest.py) does NOT fire when
DISPLAY/binaries/tkinter are present.

This is a FAST, non-live meta-test (not @pytest.mark.smoke) that drives a
`--collect-only` subprocess against the ROOT config to prove both directions
of the deselect are wired: the smoke tests are excluded by default, and the
`smoke` marker is registered (no PytestUnknownMarkWarning). `--collect-only`
runs no fixtures, so it spawns zero windows/tmux and is safe to run on a live
host.

WHAT IS ASSERTED NOW (task 4060, review finding recovered from task 2300):
the deselect run must exit pytest.ExitCode.NO_TESTS_COLLECTED (5, NOT 0 --
every smoke test is deselected so nothing remains to run) AND print a
`(N deselected)` summary with N >= 1, on top of the two original substring
checks (`_deselection_problems`). Both original assertions were substring
checks against captured output, so ANY run where collection never happened
satisfied both -- e.g. running this file's own subprocess command under an
interpreter without pytest installed measured rc=1 with stderr
`No module named pytest`, and BOTH original assertions passed against that
output. That was a false green reporting the X11 safety net as wired when
collection never ran. The happy path, for contrast: rc=5 with
`no tests collected (3 deselected)`.

THE POSITIVE CONTROL (`_collection_problems`,
test_smoke_tests_are_collectible_when_root_addopts_is_overridden): the
deselection guard alone cannot distinguish "deselected" from "never
present" -- smoke tests that were deleted or renamed would satisfy it just
as well. The positive control re-runs the same target with the root
addopts marker filter overridden and requires the smoke tests to actually
appear.

STILL NON-LIVE: every subprocess is `--collect-only` (runs no fixtures),
and the two guard-rejection tests
(test_deselection_guard_rejects_a_run_where_collection_never_ran,
test_positive_control_guard_rejects_a_run_that_collected_nothing) feed
hand-built subprocess.CompletedProcess stubs and spawn no subprocess at
all, so the whole file remains safe to run on a live DISPLAY=:0 desktop.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# The probed smoke-test FUNCTION name. A function name (not the filename)
# rules out a hypothetical collection-ERROR line printing the path and
# producing a false pass. Shared between the deselection guard and the
# positive control so the two cannot drift apart.
SMOKE_TEST_NAME = 'test_wm_focus_raises_exactly_the_disposable_window'

_DESELECTED_RE = re.compile(r'\((\d+) deselected\)')


def _collect_only(*extra_args: str) -> subprocess.CompletedProcess[str]:
    """`pytest --collect-only` over cockpit's smoke dir, bound to the ROOT config."""
    return subprocess.run(
        [
            sys.executable,
            '-m',
            'pytest',
            '-c',
            str(REPO_ROOT / 'pyproject.toml'),
            str(REPO_ROOT / 'cockpit' / 'tests' / 'smoke'),
            '--collect-only',
            '-q',
            '-p',
            'no:cacheprovider',
            *extra_args,
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )


def _deselection_problems(result: subprocess.CompletedProcess[str]) -> list[str]:
    """Every reason `result` fails to prove a REAL collection deselected the smoke tests."""
    combined = result.stdout + result.stderr
    problems = []

    # (1) a real run that selected nothing. rc is 5 (NOT 0): all smoke tests
    # are deselected so pytest reports NO_TESTS_COLLECTED. A collection
    # error, a broken conftest, or an interpreter without pytest lands on
    # some other rc and is rejected here.
    if result.returncode != pytest.ExitCode.NO_TESTS_COLLECTED:
        problems.append(
            f'expected exit code {int(pytest.ExitCode.NO_TESTS_COLLECTED)} '
            f'(NO_TESTS_COLLECTED), got {result.returncode} -- collection '
            f'may never have run'
        )

    # (2) collection actually walked the smoke dir and found tests to
    # filter. Necessary alongside (1): an empty/missing smoke dir also
    # exits 5 but prints no `deselected` clause.
    m = _DESELECTED_RE.search(combined)
    if m is None or int(m.group(1)) < 1:
        problems.append(
            'expected a `(N deselected)` summary with N >= 1 in the collection '
            'output, found none -- collection may not have walked the smoke dir'
        )

    # (3) DESELECTED -- a smoke test FUNCTION name appears in --collect-only
    # output only when the test is actually SELECTED/collected.
    if SMOKE_TEST_NAME in combined:
        problems.append(
            'smoke tests were collected under the root pytest config -- '
            'not deselected by default'
        )

    # (4) MARKER REGISTERED -- no unknown-mark warning for @pytest.mark.smoke.
    if 'PytestUnknownMarkWarning' in combined or 'Unknown pytest.mark.smoke' in combined:
        problems.append('the `smoke` marker is not registered under the root pytest config')

    return problems


def _collection_problems(result: subprocess.CompletedProcess[str]) -> list[str]:
    """Every reason `result` fails to prove the smoke tests EXIST and are collectible."""
    combined = result.stdout + result.stderr
    problems = []
    if result.returncode != pytest.ExitCode.OK:
        problems.append(
            f'expected exit code {int(pytest.ExitCode.OK)} (OK), got '
            f'{result.returncode} -- collection did not complete cleanly'
        )
    if SMOKE_TEST_NAME not in combined:
        problems.append(
            f'{SMOKE_TEST_NAME!r} was not collected even with the root addopts marker '
            f'filter removed -- the smoke tests are MISSING, not merely deselected'
        )
    return problems


def test_repo_root_pytest_config_deselects_smoke_and_registers_marker() -> None:
    result = _collect_only()
    problems = _deselection_problems(result)
    assert not problems, (
        '\n'.join(problems) + f'\n\nfull output:\n{result.stdout + result.stderr}'
    )


def test_deselection_guard_rejects_a_run_where_collection_never_ran() -> None:
    """Regression test for task 4060 (review finding recovered from task 2300).

    The two substring assertions above are both satisfied by a run where
    pytest never executed at all -- e.g. invoking an interpreter with no
    pytest installed. Measured on this base: running the guard's own
    subprocess command under `/usr/bin/python3` (no pytest installed) exits
    1 with stderr `/usr/bin/python3: No module named pytest`, and BOTH
    original assertions ('test_wm_focus_...' not in output, and no
    PytestUnknownMarkWarning) PASS against that output -- a false green that
    reports the X11 safety net as wired when collection never ran. This test
    pins `_deselection_problems` (introduced to fix that hole) to REJECT
    that exact stub.
    """
    never_ran = subprocess.CompletedProcess(
        args=[],
        returncode=1,
        stdout='',
        stderr='/usr/bin/python3: No module named pytest\n',
    )

    problems = _deselection_problems(never_ran)
    assert problems, f'expected _deselection_problems to reject a run that never ran pytest, but it did not. Stub: {never_ran!r}'
    assert any('exit' in p.lower() or 'returncode' in p.lower() for p in problems), (
        f'expected a problem naming the exit/return code as the reason for rejection, got: {problems}'
    )


def test_positive_control_guard_rejects_a_run_that_collected_nothing() -> None:
    """Regression test for task 4060's positive control (the "never present" half).

    The deselection guard above proves the smoke tests are ABSENT from a
    default root-config collection -- but that is equally satisfied by
    smoke tests that do not exist at all (deleted, renamed, or an emptied
    directory). Nothing in that guard alone distinguishes "deselected" from
    "never present". The positive control (step-4) re-runs the same target
    with the root addopts marker filter removed and requires the smoke
    tests to actually appear; `_collection_problems` is what that control
    is built on.

    This stub reproduces the shape pytest prints when a target directory
    truly yields nothing under an overridden addopts -- exit
    NO_TESTS_COLLECTED (5) with no smoke-test name anywhere in the output
    -- and pins `_collection_problems` to reject it, reporting BOTH
    available reasons (the exit code, and the missing test name).
    """
    nothing_collected = subprocess.CompletedProcess(
        args=[],
        returncode=int(pytest.ExitCode.NO_TESTS_COLLECTED),
        stdout='no tests ran in 0.01s\n',
        stderr='',
    )

    problems = _collection_problems(nothing_collected)
    assert problems, (
        f'expected _collection_problems to reject a run that collected nothing, '
        f'but it did not. Stub: {nothing_collected!r}'
    )
    assert len(problems) == 2, (
        f'expected both reasons (exit code not OK, and smoke test name absent) to be '
        f'reported, got {len(problems)}: {problems}'
    )
    assert any('exit' in p.lower() or 'returncode' in p.lower() for p in problems), (
        f'expected a problem naming the exit/return code, got: {problems}'
    )
    assert any(SMOKE_TEST_NAME in p for p in problems), (
        f'expected a problem naming the missing smoke test, got: {problems}'
    )


def test_smoke_tests_are_collectible_when_root_addopts_is_overridden() -> None:
    """POSITIVE CONTROL for the deselection guard above.

    The sibling guard proves the smoke tests are absent from a default
    root-config collection -- but that alone is also satisfied by tests
    that do not exist. Overriding `addopts` drops the root
    `-m 'not smoke and not integration and not warm_lane_bash'` filter, so
    the same target under the same config must now collect the tests --
    which is what distinguishes "deselected" from "never present".

    `-o` splits on the FIRST `=`, so the value keeps its own `=`.
    `--import-mode=importlib` is re-supplied deliberately (see design
    decision): the bare `-o addopts=` form also works but silently drops
    import mode along with the marker filter, running the control under a
    different collection semantics than every real root-bound run.

    Still non-live: `--collect-only` runs no fixtures, so no Tk window and
    no tmux session is created even though the tests are now SELECTED.
    """
    result = _collect_only('-o', 'addopts=--import-mode=importlib')
    problems = _collection_problems(result)
    assert not problems, (
        '\n'.join(problems) + f'\n\nfull output:\n{result.stdout + result.stderr}'
    )
