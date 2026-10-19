"""The one way an orchestrator test runs a probe file in a nested serial pytest session under this subproject's inifile."""
from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from _orch_helpers import ORCH_DIR, ORCH_PYPROJECT, sanitized_probe_env


def run_probe_session(
    probe: Path, *, kill_secs: int, backstop_secs: float, extra: Sequence[str] = (),
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable, '-m', 'pytest', str(probe),
            '-c', str(ORCH_PYPROJECT), '-o', 'addopts=', '-p', 'no:cacheprovider',
            f'--timeout={kill_secs}', '--timeout-method=thread', '-q',
            *extra,
        ],
        cwd=ORCH_DIR, env=sanitized_probe_env(), capture_output=True, text=True,
        timeout=backstop_secs,
    )
