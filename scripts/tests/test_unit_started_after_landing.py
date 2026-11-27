"""scripts/unit_started_after_landing.py: did a systemd unit start after a task's merge landed on main?

systemctl is a fake_recorder on PATH; main is a real tmp git repo whose
landings come from throughput_cohort_fixtures.land_branch.
"""
from __future__ import annotations

import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import unit_started_after_landing as predicate
from committed_git_repo import committed_repo
from shared.testing_own_time import run_on_own_time
from throughput_cohort_fixtures import land_branch

UNIT = 'fused-memory.service'
LANDED = datetime(2026, 11, 20, 12, 0, tzinfo=UTC)
M = int(LANDED.timestamp())
STARTED_AFTER = 0
NOT_YET = 75
ERROR = 1
SYSTEMCTL_ARGV = [
    '--user', 'show', UNIT, '-p', 'ExecMainStartTimestamp', '--timestamp=unix', '--value',
]


@pytest.fixture
def repo(tmp_path) -> Path:
    """A repo whose main carries one landing of task/42, committed at M."""
    root = tmp_path / 'repo'
    committed_repo(root, {})
    land_branch(root, 42, LANDED)
    return root


@pytest.fixture
def systemctl(fake_recorder, monkeypatch):
    """Factory: a fake systemctl first on PATH, answering ExecMainStartTimestamp with *stdout*."""
    def _install(stdout: str, exit_code: int = 0):
        fake = fake_recorder('systemctl', replies={
            'ExecMainStartTimestamp': {'stdout': stdout, 'exit_code': exit_code},
        })
        monkeypatch.setenv('PATH', f'{fake.bin_dir}:{os.environ["PATH"]}')
        return fake

    return _install


def _started_at(epoch: int) -> str:
    return f'@{epoch}\n'


def _run(*argv: str) -> int:
    """The predicate's exit status, whether main returns it or its parser raises it."""
    try:
        return predicate.main(list(argv))
    except SystemExit as exc:
        return exc.code


def test_a_unit_started_after_the_landing_passes(repo, systemctl, capsys):
    fake = systemctl(_started_at(M + 60))

    assert _run(UNIT, '42', '--repo', str(repo)) == STARTED_AFTER

    assert fake.calls() == [SYSTEMCTL_ARGV]
    out = capsys.readouterr().out
    assert LANDED.isoformat() in out
    assert (LANDED + timedelta(seconds=60)).isoformat() in out


@pytest.mark.parametrize('start', [M - 60, M], ids=['before', 'same-second'])
def test_a_unit_started_before_or_in_the_landing_second_is_not_yet(repo, systemctl, start):
    systemctl(_started_at(start))

    assert _run(UNIT, '42', '--repo', str(repo)) == NOT_YET


@pytest.mark.parametrize('task_id', ['4', '999'])
def test_a_task_without_its_own_merge_on_main_is_not_yet(repo, systemctl, task_id):
    systemctl(_started_at(M + 60))

    assert _run(UNIT, task_id, '--repo', str(repo)) == NOT_YET


def test_the_latest_of_two_landings_is_the_one_compared(repo, systemctl):
    land_branch(repo, 42, LANDED + timedelta(hours=1))
    systemctl(_started_at(M + 60))

    assert _run(UNIT, '42', '--repo', str(repo)) == NOT_YET


def test_an_empty_start_value_is_an_error_naming_the_unit(repo, systemctl, capsys):
    systemctl('\n')

    assert _run(UNIT, '42', '--repo', str(repo)) == ERROR

    assert UNIT in capsys.readouterr().err


def test_a_failing_systemctl_is_an_error(repo, systemctl):
    systemctl(_started_at(M + 60), exit_code=1)

    assert _run(UNIT, '42', '--repo', str(repo)) == ERROR


def test_a_repo_path_that_is_not_a_repo_is_an_error(tmp_path, systemctl):
    systemctl(_started_at(M + 60))
    not_a_repo = tmp_path / 'plain'
    not_a_repo.mkdir()

    assert _run(UNIT, '42', '--repo', str(not_a_repo)) == ERROR


@pytest.mark.parametrize('argv', [(UNIT,), (UNIT, 'forty-two')], ids=['missing', 'non-integer'])
def test_a_bad_command_line_is_an_error(repo, systemctl, argv):
    systemctl(_started_at(M + 60))

    assert _run(*argv, '--repo', str(repo)) == ERROR


_CHILD_BUDGET_S = 60.0  # own time; the child needs well under a second


def test_a_real_child_process_passes_when_the_unit_started_after(repo, systemctl):
    systemctl(_started_at(M + 60))

    completed = run_on_own_time(
        [sys.executable, predicate.__file__, UNIT, '42', '--repo', str(repo)],
        budget=_CHILD_BUDGET_S,
    )

    assert completed.returncode == STARTED_AFTER, completed.stderr
