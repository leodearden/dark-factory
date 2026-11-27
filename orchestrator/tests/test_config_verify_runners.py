"""Where OrchestratorConfig.verify_runners comes from, and which names it may carry."""

import json
import re
import textwrap
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from orchestrator.config import OrchestratorConfig, VerifyRunnerConfig, load_config
from orchestrator.verify_host_roster_file import ROSTER_FILE_ENV, RosterFileError

PROJECT = 'p'

ROSTER = """\
hosts:
  box-a:
    ssh_host: verify-a
    df_checkout_path: /srv/a/dark-factory
    gate_checkouts_path: /srv/a/gates
    projects:
      p:
        git_remote: verify-a-p
        config_path: /srv/a/p-host.yaml
  box-b:
    ssh_host: verify-b
    df_checkout_path: /srv/b/dark-factory
    enabled: false
    projects:
      p:
        git_remote: verify-b-p
  box-c:
    ssh_host: verify-c
    df_checkout_path: /srv/c/dark-factory
    projects:
      q:
        git_remote: verify-c-q
"""

ROSTER_OF_OTHER_PROJECTS = """\
hosts:
  box-c:
    ssh_host: verify-c
    df_checkout_path: /srv/c/dark-factory
    projects:
      q:
        git_remote: verify-c-q
"""

INLINE_EQUIVALENT_OF_ROSTER = [
    {
        'name': 'box-a',
        'ssh_host': 'verify-a',
        'git_remote': 'verify-a-p',
        'config_path': '/srv/a/p-host.yaml',
        'df_checkout_path': '/srv/a/dark-factory',
        'gate_checkouts_path': '/srv/a/gates',
    },
    {
        'name': 'box-b',
        'ssh_host': 'verify-b',
        'git_remote': 'verify-b-p',
        'df_checkout_path': '/srv/b/dark-factory',
        'enabled': False,
    },
]

INLINE_LAPTOP = [
    {
        'name': 'laptop',
        'ssh_host': 'laptop-ssh',
        'git_remote': 'laptop',
        'df_checkout_path': '/home/me/dark-factory',
    },
]


def _runner(name: str, *, enabled: bool = True) -> dict[str, object]:
    return {
        'name': name,
        'ssh_host': f'{name}-ssh',
        'git_remote': f'{name}-remote',
        'df_checkout_path': '/df',
        'enabled': enabled,
    }


class TestRunnerNames:
    def test_two_runners_with_one_name_are_refused(self):
        with pytest.raises(ValidationError, match=r"(?s)must be unique.*'box'"):
            OrchestratorConfig(verify_runners=[_runner('box'), _runner('box', enabled=False)])

    def test_a_runner_named_after_the_local_slot_is_refused(self):
        with pytest.raises(ValidationError, match=r"(?s)none may be 'local'.*'local'"):
            OrchestratorConfig(verify_runners=[_runner('local')])

    def test_assigning_a_duplicate_list_is_refused_and_keeps_the_old_one(self):
        config = OrchestratorConfig(verify_runners=[_runner('box')])
        original = list(config.verify_runners)

        with pytest.raises(ValidationError, match='must be unique'):
            config.verify_runners = [_runner('dup'), _runner('dup')]

        assert config.verify_runners == original


def _write_yaml(tmp_path: Path, name: str, runners: list[dict[str, object]] | None = None) -> Path:
    document: dict[str, object] = {'fused_memory': {'project_id': PROJECT}}
    if runners is not None:
        document['verify_runners'] = runners
    path = tmp_path / name
    path.write_text(yaml.safe_dump(document))
    return path


def _write_roster(tmp_path: Path, monkeypatch, text: str) -> Path:
    path = tmp_path / 'roster.yaml'
    path.write_text(textwrap.dedent(text))
    monkeypatch.setenv(ROSTER_FILE_ENV, str(path))
    return path


def _point_at_an_absent_roster(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(ROSTER_FILE_ENV, str(tmp_path / 'absent-roster.yaml'))


def _declare_inline(source: str, tmp_path: Path, monkeypatch, runners: list[dict[str, object]]) -> Path:
    """Declare *runners* inline through *source* and return the yaml path to load."""
    if source == 'yaml':
        monkeypatch.delenv('ORCH_VERIFY_RUNNERS', raising=False)
        return _write_yaml(tmp_path, 'inline.yaml', runners)
    monkeypatch.setenv('ORCH_VERIFY_RUNNERS', json.dumps(runners))
    return _write_yaml(tmp_path, 'bare.yaml')


class TestRosterFileIsRead:
    @pytest.mark.parametrize('inline_source', ['yaml', 'env'])
    def test_a_roster_file_yields_the_same_runner_list_as_the_inline_block(
        self, inline_source, tmp_path, monkeypatch,
    ):
        _point_at_an_absent_roster(tmp_path, monkeypatch)
        inline_yaml = _declare_inline(inline_source, tmp_path, monkeypatch, INLINE_EQUIVALENT_OF_ROSTER)
        inline = load_config(inline_yaml).verify_runners

        monkeypatch.setenv('ORCH_VERIFY_RUNNERS', '[]')
        _write_roster(tmp_path, monkeypatch, ROSTER)
        from_roster = load_config(_write_yaml(tmp_path, 'roster-only.yaml')).verify_runners

        assert from_roster == inline
        assert [runner.name for runner in from_roster] == ['box-a', 'box-b']
        assert from_roster[1].gate_paths().root == '/srv/b/dark-factory-gate'

    def test_an_absent_roster_file_changes_nothing(self, tmp_path, monkeypatch):
        _point_at_an_absent_roster(tmp_path, monkeypatch)
        monkeypatch.delenv('ORCH_VERIFY_RUNNERS', raising=False)

        config = load_config(_write_yaml(tmp_path, 'inline.yaml', INLINE_LAPTOP))

        assert config.verify_runners == [VerifyRunnerConfig(**runner) for runner in INLINE_LAPTOP]

    def test_a_roster_naming_only_other_projects_changes_nothing(self, tmp_path, monkeypatch):
        _write_roster(tmp_path, monkeypatch, ROSTER_OF_OTHER_PROJECTS)
        monkeypatch.delenv('ORCH_VERIFY_RUNNERS', raising=False)

        config = load_config(_write_yaml(tmp_path, 'inline.yaml', INLINE_LAPTOP))

        assert config.verify_runners == [VerifyRunnerConfig(**runner) for runner in INLINE_LAPTOP]

    def test_an_empty_environment_list_lets_the_roster_supply_the_runners(self, tmp_path, monkeypatch):
        monkeypatch.setenv('ORCH_VERIFY_RUNNERS', '[]')
        _write_roster(tmp_path, monkeypatch, ROSTER)

        config = load_config(_write_yaml(tmp_path, 'shadowed-inline.yaml', INLINE_LAPTOP))

        assert config.verify_runners == [
            VerifyRunnerConfig(**runner) for runner in INLINE_EQUIVALENT_OF_ROSTER
        ]

    @pytest.mark.parametrize(
        'roster_text',
        [
            pytest.param('hosts: [unterminated\n', id='yaml-syntax-error'),
            pytest.param(
                """\
                hosts:
                  box-a:
                    ssh_host: verify-a
                    df_checkout_path: /srv/a/dark-factory
                    bogus_key: 1
                    projects:
                      p:
                        git_remote: verify-a-p
                """,
                id='unknown-host-key',
            ),
            pytest.param('', id='empty-document'),
        ],
    )
    def test_a_roster_file_that_fails_to_parse_fails_the_load(self, roster_text, tmp_path, monkeypatch):
        roster_path = _write_roster(tmp_path, monkeypatch, roster_text)

        with pytest.raises(RosterFileError, match=re.escape(str(roster_path))):
            load_config(_write_yaml(tmp_path, 'bare.yaml'))
