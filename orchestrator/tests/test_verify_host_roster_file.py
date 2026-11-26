"""The private verify-host roster file parser (PRD verify-host-roster §6.3, task α1)."""

import dataclasses
import re
import textwrap
from pathlib import Path

import pytest

from orchestrator.config import VerifyRunnerConfig
from orchestrator.verify_host_roster_file import (
    ROSTER_FILE_ENV,
    Roster,
    RosterFileError,
    RosterRunner,
    load_roster,
    roster_file_path,
)

TWO_HOSTS = """\
hosts:
  box-a:
    ssh_host: verify-a
    df_checkout_path: /srv/a/dark-factory
    gate_checkouts_path: /srv/a/gates
    wrapper_env:
      ORCH_WATCHDOG_HEARTBEAT_TIMEOUT_SECS: "600"
    projects:
      p:
        git_remote: verify-a-p
        config_path: /srv/a/p-host.yaml
        task_slots: 2
        trust: trusted
        disk_footprint_gib: 40
        host_config:
          project_root: /srv/a/p
          verify_cgroup_memory: {max_gib: 12}
      q:
        git_remote: verify-a-q
        host_config:
          project_root: /srv/a/q
  box-b:
    ssh_host: verify-b
    df_checkout_path: /srv/b/dark-factory
    enabled: false
    projects:
      p:
        git_remote: verify-b-p
        config_path: /srv/b/p-host.yaml
        host_config:
          project_root: /srv/b/p
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / 'verify-hosts.yaml'
    path.write_text(textwrap.dedent(text))
    return path


def _two_hosts(tmp_path: Path) -> Roster:
    return load_roster(_write(tmp_path, TWO_HOSTS))


class TestRosterFilePath:
    def test_the_override_variable_wins(self) -> None:
        env = {ROSTER_FILE_ENV: '/elsewhere/hosts.yaml', 'XDG_CONFIG_HOME': '/xdg', 'HOME': '/h'}
        assert roster_file_path(env) == Path('/elsewhere/hosts.yaml')

    def test_xdg_config_home_is_used_without_the_override(self) -> None:
        env = {'XDG_CONFIG_HOME': '/xdg', 'HOME': '/h'}
        assert roster_file_path(env) == Path('/xdg/orchestrator/verify-hosts.yaml')

    def test_home_config_is_the_fallback(self) -> None:
        assert roster_file_path({'HOME': '/h'}) == Path('/h/.config/orchestrator/verify-hosts.yaml')

    def test_the_override_variable_is_named_orch_verify_hosts_file(self) -> None:
        assert ROSTER_FILE_ENV == 'ORCH_VERIFY_HOSTS_FILE'


class TestAbsentFile:
    def test_an_absent_file_is_an_empty_roster(self, tmp_path: Path) -> None:
        roster = load_roster(tmp_path / 'missing.yaml')
        assert roster.hosts == {}
        assert roster.runners_for_project('p') == ()

    def test_a_directory_at_the_path_is_an_error_not_an_absent_file(self, tmp_path: Path) -> None:
        path = tmp_path / 'verify-hosts.yaml'
        path.mkdir()
        with pytest.raises(RosterFileError, match=re.escape(str(path))):
            load_roster(path)


class TestRunnersForProject:
    def test_each_host_naming_the_project_yields_one_runner_in_file_order(self, tmp_path: Path) -> None:
        runners = _two_hosts(tmp_path).runners_for_project('p')
        assert runners == (
            RosterRunner(
                name='box-a',
                ssh_host='verify-a',
                git_remote='verify-a-p',
                config_path='/srv/a/p-host.yaml',
                df_checkout_path='/srv/a/dark-factory',
                gate_checkouts_path='/srv/a/gates',
                enabled=True,
                task_slots=2,
                trust='trusted',
            ),
            RosterRunner(
                name='box-b',
                ssh_host='verify-b',
                git_remote='verify-b-p',
                config_path='/srv/b/p-host.yaml',
                df_checkout_path='/srv/b/dark-factory',
                gate_checkouts_path=None,
                enabled=False,
                task_slots=None,
                trust='probation',
            ),
        )

    def test_a_project_on_one_host_yields_only_that_host(self, tmp_path: Path) -> None:
        runners = _two_hosts(tmp_path).runners_for_project('q')
        assert [runner.name for runner in runners] == ['box-a']

    def test_an_unnamed_project_yields_no_runner(self, tmp_path: Path) -> None:
        assert _two_hosts(tmp_path).runners_for_project('unnamed') == ()

    def test_omitted_optional_keys_take_their_defaults(self, tmp_path: Path) -> None:
        roster = load_roster(_write(tmp_path, """\
            hosts:
              h:
                ssh_host: s
                df_checkout_path: /srv/df
                projects:
                  p:
                    git_remote: r
            """))
        (runner,) = roster.runners_for_project('p')
        assert runner.trust == 'probation'
        assert runner.task_slots is None
        assert runner.config_path is None
        assert runner.gate_checkouts_path is None
        assert runner.enabled is True


class TestHostSideKeys:
    def test_host_configs_maps_each_project_to_its_host_config(self, tmp_path: Path) -> None:
        assert _two_hosts(tmp_path).host_configs('box-a') == {
            'p': {'project_root': '/srv/a/p', 'verify_cgroup_memory': {'max_gib': 12}},
            'q': {'project_root': '/srv/a/q'},
        }

    def test_wrapper_env_and_disk_footprint_are_readable(self, tmp_path: Path) -> None:
        host = _two_hosts(tmp_path).host('box-a')
        assert host.wrapper_env == {'ORCH_WATCHDOG_HEARTBEAT_TIMEOUT_SECS': '600'}
        assert host.projects['p'].disk_footprint_gib == 40
        assert host.projects['q'].disk_footprint_gib is None

    def test_an_unknown_host_names_the_host_and_the_file(self, tmp_path: Path) -> None:
        path = _write(tmp_path, TWO_HOSTS)
        roster = load_roster(path)
        with pytest.raises(RosterFileError) as raised:
            roster.host('box-z')
        assert 'box-z' in str(raised.value)
        assert str(path) in str(raised.value)


class TestTheConfigSeam:
    """α2 turns each runner into a VerifyRunnerConfig; it must equal the inline block's."""

    INLINE = (
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
            'config_path': '/srv/b/p-host.yaml',
            'df_checkout_path': '/srv/b/dark-factory',
            'enabled': False,
        },
    )

    def test_the_roster_yields_the_inline_blocks_runner_list(self, tmp_path: Path) -> None:
        from_roster = [
            VerifyRunnerConfig.model_validate(dataclasses.asdict(runner))
            for runner in _two_hosts(tmp_path).runners_for_project('p')
        ]
        from_inline = [VerifyRunnerConfig.model_validate(entry) for entry in self.INLINE]
        assert from_roster == from_inline


MALFORMED = {
    'yaml_syntax_error': 'hosts: [unclosed\n',
    'empty_document': '',
    'top_level_list': '- hosts\n',
    'unknown_host_key': """\
        hosts:
          h: {ssh_host: s, df_checkout_path: /d, enabeld: true, projects: {}}
        """,
    'unknown_project_key': """\
        hosts:
          h:
            ssh_host: s
            df_checkout_path: /d
            projects:
              p: {git_remote: r, trusted: true}
        """,
    'task_slots_not_an_int': """\
        hosts:
          h:
            ssh_host: s
            df_checkout_path: /d
            projects:
              p: {git_remote: r, task_slots: two}
        """,
    'task_slots_negative': """\
        hosts:
          h:
            ssh_host: s
            df_checkout_path: /d
            projects:
              p: {git_remote: r, task_slots: -1}
        """,
    'trust_unknown': """\
        hosts:
          h:
            ssh_host: s
            df_checkout_path: /d
            projects:
              p: {git_remote: r, trust: maybe}
        """,
    'host_config_not_a_mapping': """\
        hosts:
          h:
            ssh_host: s
            df_checkout_path: /d
            projects:
              p: {git_remote: r, host_config: [1]}
        """,
    'duplicate_host': """\
        hosts:
          h: {ssh_host: s, df_checkout_path: /d, projects: {}}
          h: {ssh_host: t, df_checkout_path: /e, projects: {}}
        """,
    'duplicate_project': """\
        hosts:
          h:
            ssh_host: s
            df_checkout_path: /d
            projects:
              p: {git_remote: r}
              p: {git_remote: t}
        """,
    'wrapper_env_name_not_an_identifier': """\
        hosts:
          h:
            ssh_host: s
            df_checkout_path: /d
            wrapper_env: {'BAD-NAME': '1'}
            projects: {}
        """,
}


@pytest.mark.parametrize('text', MALFORMED.values(), ids=MALFORMED.keys())
def test_a_malformed_file_fails_loudly_naming_the_path(tmp_path: Path, text: str) -> None:
    path = _write(tmp_path, text)
    with pytest.raises(RosterFileError, match=re.escape(str(path))):
        load_roster(path)


EXAMPLE_ROSTER = Path(__file__).resolve().parents[2] / 'config' / 'verify-hosts.example.yaml'


def _example() -> Roster:
    assert EXAMPLE_ROSTER.is_file(), EXAMPLE_ROSTER
    return load_roster(EXAMPLE_ROSTER)


class TestTheTrackedExample:
    """config/verify-hosts.example.yaml is the documented shape; keep it executable."""

    def test_the_example_declares_a_host_with_a_project(self) -> None:
        roster = _example()
        assert roster.hosts
        assert all(host.projects for host in roster.hosts.values())

    def test_every_project_block_names_its_project_root(self) -> None:
        roster = _example()
        for name in roster.hosts:
            for project_id, host_config in roster.host_configs(name).items():
                assert 'project_root' in host_config, (name, project_id)

    def test_every_runner_validates_as_the_dispatcher_would(self) -> None:
        roster = _example()
        project_ids = {project_id for host in roster.hosts.values() for project_id in host.projects}
        for project_id in sorted(project_ids):
            for runner in roster.runners_for_project(project_id):
                paths = VerifyRunnerConfig.model_validate(dataclasses.asdict(runner)).gate_paths()
                assert Path(paths.source).is_absolute()
                assert Path(paths.root).is_absolute()
