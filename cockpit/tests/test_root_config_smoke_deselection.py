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
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_repo_root_pytest_config_deselects_smoke_and_registers_marker() -> None:
    result = subprocess.run(
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
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    combined = result.stdout + result.stderr

    # DESELECTED -- a smoke test FUNCTION name appears in --collect-only
    # output only when the test is actually SELECTED/collected; a function
    # name (not the filename) rules out a hypothetical collection-ERROR line
    # printing the path producing a false pass.
    assert 'test_wm_focus_raises_exactly_the_disposable_window' not in combined, (
        f'smoke tests were collected under the root pytest config -- '
        f'not deselected by default:\n{combined}'
    )

    # MARKER REGISTERED -- no unknown-mark warning for @pytest.mark.smoke.
    assert 'PytestUnknownMarkWarning' not in combined, (
        f'the `smoke` marker is not registered under the root pytest config:\n{combined}'
    )
    assert 'Unknown pytest.mark.smoke' not in combined, (
        f'the `smoke` marker is not registered under the root pytest config:\n{combined}'
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
