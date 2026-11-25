"""The ``orchestrator verify-host-preflight`` command's refusals: exit 2 and no record (task 11731)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from _git_fixtures import seed_repo
from click.testing import CliRunner

from orchestrator.cli import main

COMMAND = 'verify-host-preflight'
BOX = {
    'name': 'box', 'ssh_host': 'box', 'git_remote': 'box',
    'df_checkout_path': '/remote/dark-factory', 'enabled': False,
}


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A project repo that the loaded config names as project_root, declaring the disabled runner 'box'."""
    repo = seed_repo(tmp_path / 'repo')
    monkeypatch.setenv('ORCH_PROJECT_ROOT', str(repo))
    monkeypatch.setenv('ORCH_VERIFY_RUNNERS', json.dumps([BOX]))
    return repo


@pytest.fixture
def config_yaml(tmp_path: Path) -> Path:
    path = tmp_path / 'orch.yaml'
    path.write_text('git:\n  main_branch: main\n')
    return path


def invoke(*args: str) -> int:
    assert COMMAND in main.commands
    return CliRunner().invoke(main, [COMMAND, *args]).exit_code


def test_an_undeclared_runner_is_refused_with_exit_2_and_no_record(
    repo: Path, config_yaml: Path,
) -> None:
    assert invoke('--runner', 'nope', '--config', str(config_yaml)) == 2
    assert not (repo / 'data' / 'verify-logs' / 'preflight').exists()


def test_a_persistent_merge_worktree_project_is_refused_with_exit_2_and_no_record(
    repo: Path, config_yaml: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('ORCH_GIT__PERSISTENT_MERGE_WORKTREE', 'true')

    assert invoke('--runner', 'box', '--config', str(config_yaml)) == 2
    assert not (repo / 'data' / 'verify-logs' / 'preflight').exists()
