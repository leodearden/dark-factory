"""Runs an orchestrator probe file in a nested serial pytest session from ORCH_DIR, under this subproject's inifile and conftest chain.

A probe session never uses the host's default pytest temp root, whose
exit-time cleanup of other sessions' dead dirs is unbounded and runs outside
pytest-timeout; it runs under the ``basetemp`` its caller names instead
(contract: test_probe_session.py).

Not every nested session goes through here:
``test_aiosqlite_leak_isolation.py::_run_probe_session`` is deliberately
separate. It compares two inifiles from its probe's own directory with
``--confcutdir`` pinned there, and its probe requests no ``tmp_path``, so it
never creates a dir in the host's temp root.
"""
from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from _orch_helpers import ORCH_DIR, ORCH_PYPROJECT, sanitized_probe_env


def run_probe_session(
    probe: Path, *, basetemp: Path, kill_secs: int, backstop_secs: float,
    extra: Sequence[str] = (),
) -> subprocess.CompletedProcess[str]:
    smuggled = [arg for arg in extra if arg == '--basetemp' or arg.startswith('--basetemp=')]
    if smuggled:
        raise ValueError(
            'a probe session runs under the basetemp passed as basetemp=, never '
            f'one from extra; refusing {smuggled!r}'
        )
    return subprocess.run(
        [
            sys.executable, '-m', 'pytest', str(probe),
            '-c', str(ORCH_PYPROJECT), '-o', 'addopts=', '-p', 'no:cacheprovider',
            f'--basetemp={basetemp}',
            f'--timeout={kill_secs}', '--timeout-method=thread', '-q',
            *extra,
        ],
        cwd=ORCH_DIR, env=sanitized_probe_env(), capture_output=True, text=True,
        timeout=backstop_secs,
    )
