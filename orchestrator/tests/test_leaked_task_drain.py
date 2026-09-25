"""conftest.py's ``_drain_leaked_tasks`` keeps a leaked one-cancel-proof task
from hanging the session at loop teardown (task 5811).

Both arms run the SAME probe test in a REAL nested pytest session under this
subproject's inifile, serially, with pytest-timeout armed.  The control arm
disables conftest.py: the leftover task swallows loop teardown's single
cancel, the session hangs, and pytest-timeout kills it -- the banner is
visible because the session is serial.  The treatment arm is the shipped
configuration and must simply pass.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from _orch_helpers import ORCH_DIR, ORCH_PYPROJECT, sanitized_probe_env

_PROBE = Path(__file__).with_name('_leaked_task_drain_probe.py')
#: The control arm's whole run time: it hangs until pytest-timeout kills it.
_CONTROL_KILL_SECS = 15
#: The treatment arm can only fail by hanging, so its bound buys nothing tight;
#: it spends ~5s in the drain's bounded wait before Runner.close frees the task.
_TREATMENT_KILL_SECS = 60


def _nested_session(kill_secs: int, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable, '-m', 'pytest', str(_PROBE),
            '-c', str(ORCH_PYPROJECT), '-o', 'addopts=', '-p', 'no:cacheprovider',
            f'--timeout={kill_secs}', '--timeout-method=thread', '-q',
            *extra,
        ],
        cwd=ORCH_DIR, env=sanitized_probe_env(), capture_output=True, text=True,
        timeout=kill_secs * 8,
    )


def test_without_the_drain_the_leak_hangs_loop_teardown_until_killed() -> None:
    run = _nested_session(_CONTROL_KILL_SECS, '--noconftest')
    assert run.returncode != 0, run.stdout[-2000:]
    assert 'Timeout' in run.stdout and '_cancel_all_tasks' in run.stdout, (
        'the control arm must die of pytest-timeout INSIDE loop teardown, '
        f'proving the leak hangs there; got rc={run.returncode}\n'
        f'{run.stdout[-2000:]}{run.stderr[-2000:]}'
    )


def test_with_the_drain_the_same_leak_is_harmless() -> None:
    run = _nested_session(_TREATMENT_KILL_SECS)
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-2000:]
    assert '1 passed' in run.stdout, run.stdout[-500:]
