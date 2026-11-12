"""Contract tests for ``process_group_run``, the killable process-group runner.

Every test drives real processes and patches nothing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from process_group_run import ProcessGroupOverrun, run_process_group

from df_pytest_isolation import wait_pid_gone, wait_pids_exited


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


def test_an_overrun_names_every_live_group_member_and_leaves_none_running(
    tmp_path: Path,
) -> None:
    with pytest.raises(ProcessGroupOverrun) as raised:
        run_process_group(
            ['sh', '-c', 'echo partial-out; echo partial-err >&2; sleep 600 & sleep 600'],
            cwd=tmp_path,
            env=dict(os.environ),
            budget_secs=5,
        )

    overrun = raised.value
    report = str(overrun)
    member_pids = {member.pid for member in overrun.members}
    assert overrun.budget_secs == 5, report
    assert 'partial-out' in overrun.stdout, report
    assert 'partial-err' in overrun.stderr, report
    assert overrun.leader_pid in member_pids, report
    assert member_pids - {overrun.leader_pid}, (
        f'the backgrounded sleep is missing from the live members.\n{report}'
    )
    for member in overrun.members:
        assert member.state.exited is False, report
        if member.pid != overrun.leader_pid:
            assert 'sleep' in member.cmdline, report
    assert f'{overrun.budget_secs}s' in report, report
    for pid in member_pids:
        assert str(pid) in report, report
    assert 'stdout:\npartial-out' in report, report
    assert 'stderr:\npartial-err' in report, report
    assert wait_pids_exited(member_pids, timeout=30) == {}, (
        f'a member of the overrun group is still running.\n{report}'
    )
