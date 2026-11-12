"""Contract tests for ``process_group_run``, the killable process-group runner.

Every test drives real processes and patches nothing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from process_group_run import run_process_group

from df_pytest_isolation import wait_pid_gone


def test_the_run_returns_the_leaders_exit_status_and_each_stream_separately(
    tmp_path: Path,
) -> None:
    result = run_process_group(
        [
            sys.executable, '-c',
            'import sys; print("to-out"); print("to-err", file=sys.stderr); sys.exit(3)',
        ],
        cwd=tmp_path,
        env=dict(os.environ),
        budget_secs=60,
    )

    assert result.returncode == 3, result
    assert result.stdout == 'to-out\n', result
    assert result.stderr == 'to-err\n', result


def test_the_run_ends_when_the_leader_exits_though_a_descendant_still_holds_its_output(
    tmp_path: Path,
) -> None:
    # The backgrounded sleep inherits the leader's stdout and stderr and keeps
    # them open after the leader exits; a run that waited for EOF would overrun.
    result = run_process_group(
        ['sh', '-c', 'sleep 600 & echo $!'],
        cwd=tmp_path,
        env=dict(os.environ),
        budget_secs=60,
    )

    assert result.returncode == 0, result
    descendant = int(result.stdout)
    assert wait_pid_gone(descendant, timeout=30), (
        f'the backgrounded sleep {descendant} is still running: the leader\'s '
        'process group must be killed when the run ends.'
    )
