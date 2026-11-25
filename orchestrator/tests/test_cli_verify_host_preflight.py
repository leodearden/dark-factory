"""The ``orchestrator verify-host-preflight`` command: exit codes and the record it writes (task 11731)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from _git_fixtures import seed_repo
from _verify_host_drive import runner_double
from _verify_result_doubles import real_verify_result
from click.testing import CliRunner, Result

from orchestrator import verify_host_preflight as preflight
from orchestrator.cli import main
from orchestrator.config import VerifyRunnerConfig

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


def runners_answering(monkeypatch: pytest.MonkeyPatch, *, passed: bool) -> list[str]:
    """Make every runner the command builds a double answering a *passed* verdict; returns the names built."""
    built: list[str] = []

    def builder(_config: object, _deployed_sha: object) -> preflight.BuildRunner:
        def build(runner_config: VerifyRunnerConfig) -> MagicMock:
            built.append(runner_config.name)
            return runner_double(runner_config.name, is_local=False, result=real_verify_result(passed))
        return build

    monkeypatch.setattr(preflight, 'production_runner_builder', builder)
    return built


def invoke(*args: str) -> Result:
    assert COMMAND in main.commands
    return CliRunner().invoke(main, [COMMAND, *args])


def box_record(repo: Path) -> Path:
    main_sha = subprocess.run(
        ['git', '-C', str(repo), 'rev-parse', 'main'], check=True, capture_output=True, text=True,
    ).stdout.strip()
    return repo / 'data' / 'verify-logs' / 'preflight' / f'box-{main_sha}.json'


def test_a_green_runner_exits_0_and_writes_its_record(
    repo: Path, config_yaml: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = runners_answering(monkeypatch, passed=True)

    assert invoke('--runner', 'box', '--config', str(config_yaml)).exit_code == 0
    assert built == ['box']
    assert json.loads(box_record(repo).read_text())['passed'] is True


def test_a_red_runner_exits_1_and_writes_its_record(
    repo: Path, config_yaml: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runners_answering(monkeypatch, passed=False)

    assert invoke('--runner', 'box', '--config', str(config_yaml)).exit_code == 1
    assert json.loads(box_record(repo).read_text())['passed'] is False


def test_json_prints_the_record_it_writes(
    repo: Path, config_yaml: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runners_answering(monkeypatch, passed=True)

    result = invoke('--runner', 'box', '--json', '--config', str(config_yaml))

    assert result.exit_code == 0
    assert json.loads(result.stdout) == json.loads(box_record(repo).read_text())


def test_an_undeclared_runner_is_refused_with_exit_2_and_no_record(
    repo: Path, config_yaml: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = runners_answering(monkeypatch, passed=True)

    assert invoke('--runner', 'nope', '--config', str(config_yaml)).exit_code == 2
    assert built == []
    assert not (repo / 'data' / 'verify-logs' / 'preflight').exists()


def test_a_persistent_merge_worktree_project_is_refused_with_exit_2_and_no_record(
    repo: Path, config_yaml: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = runners_answering(monkeypatch, passed=True)
    monkeypatch.setenv('ORCH_GIT__PERSISTENT_MERGE_WORKTREE', 'true')

    assert invoke('--runner', 'box', '--config', str(config_yaml)).exit_code == 2
    assert built == []
    assert not (repo / 'data' / 'verify-logs' / 'preflight').exists()
