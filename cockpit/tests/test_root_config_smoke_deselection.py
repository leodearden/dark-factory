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
