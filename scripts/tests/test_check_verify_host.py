"""scripts/deploy/check_verify_host.py, driven end to end through a fake ssh that runs the host locally."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from cli_subprocess_timeout import cli_timeout_from_env
from fake_verify_host_ssh import write_fake_ssh
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

DEPLOY = Path(__file__).resolve().parents[1] / 'deploy'
CHECKER = DEPLOY / 'check_verify_host.py'
CONTRACT = DEPLOY / 'verify_host_contract.py'
CLI_TIMEOUT = cli_timeout_from_env('CHECK_VERIFY_HOST_TEST_TIMEOUT')
VERDICTS = {'ok', 'fail', 'warn'}


class Host:
    """A roster host whose "remote" side lives under tmp_path, reached through the fake ssh."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.remote = tmp_path / 'remote'
        self.wrapper = self.remote / 'bin' / 'orchestrator'
        self.roster = tmp_path / 'verify-hosts.yaml'
        self.roster.write_text(f"""\
hosts:
  box:
    ssh_host: box-alias
    df_checkout_path: {self.remote}/df
    projects:
      p:
        git_remote: box
        config_path: {self.remote}/config/p.yaml
        host_config: {{project_root: {self.remote}/p, verify_cgroup_memory: {{max_gib: 1}}}}
      q:
        git_remote: box
        config_path: {self.remote}/config/q.yaml
        host_config: {{project_root: {self.remote}/q}}
""")
        self.env = {
            'HOME': str(tmp_path),
            'XDG_STATE_HOME': str(tmp_path / 'state'),
            'PATH': os.environ['PATH'],
            'LANG': os.environ.get('LANG', 'C.UTF-8'),
            'VERIFY_HOST_SSH': str(write_fake_ssh(tmp_path)),
            'VERIFY_HOST_WRAPPER_PATH': str(self.wrapper),
            'FAKE_SSH_LOG': str(tmp_path / 'ssh.log'),
        }
        self._provision()

    def _git(self, *args: str) -> None:
        subprocess.run(['git', *args], check=True, capture_output=True, timeout=CLI_TIMEOUT)

    def _provision(self) -> None:
        rendered = self.tmp / 'rendered'
        proc = subprocess.run(
            [sys.executable, str(CONTRACT), 'render', 'box', '--roster', str(self.roster), '--out', str(rendered)],
            env=self.env, capture_output=True, text=True, timeout=CLI_TIMEOUT, check=False,
        )
        assert proc.returncode == 0, proc.stderr
        for name in ('df', 'p', 'q'):
            self._git('init', '-q', str(self.remote / name))
            self._git('-C', str(self.remote / name), 'config', 'receive.denyCurrentBranch', 'updateInstead')
        df_orchestrator = self.remote / 'df' / '.venv' / 'bin' / 'orchestrator'
        df_orchestrator.parent.mkdir(parents=True)
        df_orchestrator.write_text('#!/bin/sh\n')
        df_orchestrator.chmod(0o755)
        (self.remote / 'config').mkdir()
        for project in ('p', 'q'):
            shutil.copyfile(rendered / f'{project}.yaml', self.remote / 'config' / f'{project}.yaml')
        self.wrapper.parent.mkdir(parents=True)
        shutil.copyfile(rendered / 'wrapper.sh', self.wrapper)
        self.wrapper.chmod(0o755)

    def check(self, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(CHECKER), *args], env={**self.env, **env},
            capture_output=True, text=True, timeout=CLI_TIMEOUT, check=False,
        )

    def rows(self, **env: str) -> tuple[int, dict[str, dict[str, Any]]]:
        proc = self.check('box', '--roster', str(self.roster), '--json', **env)
        assert 'Traceback' not in proc.stderr, proc.stderr
        report = json.loads(proc.stdout)
        assert report['host'] == 'box'
        return proc.returncode, {row['requirement']: row for row in report['rows']}


@pytest.fixture
def host(tmp_path: Path) -> Host:
    return Host(tmp_path)


def _table_ids() -> list[str]:
    proc = subprocess.run(
        [sys.executable, str(CONTRACT), 'requirements'],
        capture_output=True, text=True, timeout=CLI_TIMEOUT, check=True,
    )
    return proc.stdout.splitlines()


def test_the_checkers_rows_are_the_contract_table(host: Host) -> None:
    proc = host.check('box', '--roster', str(host.roster), '--json')
    rows = json.loads(proc.stdout)['rows']
    assert [row['requirement'] for row in rows] == _table_ids()
    for row in rows:
        assert set(row) == {'requirement', 'verdict', 'observed', 'needed'}
        assert row['verdict'] in VERDICTS


def test_the_exit_code_is_zero_iff_no_row_fails(host: Host) -> None:
    returncode, rows = host.rows()
    assert returncode == (1 if any(row['verdict'] == 'fail' for row in rows.values()) else 0)


def test_a_per_host_yaml_equal_to_the_render_is_ok_and_one_edited_byte_fails(host: Host) -> None:
    assert host.rows()[1]['per_host_yaml']['verdict'] == 'ok'
    config = host.remote / 'config' / 'q.yaml'
    config.write_text(config.read_text().replace('project_root', 'project_rooT'))
    row = host.rows()[1]['per_host_yaml']
    assert row['verdict'] == 'fail'
    assert "'q'" in row['observed']


def test_an_agent_credential_file_fails_the_check(host: Host) -> None:
    (host.tmp / '.claude').mkdir()
    (host.tmp / '.claude' / '.credentials.json').write_text('{}')
    returncode, rows = host.rows()
    assert rows['no_credentials']['verdict'] == 'fail'
    assert returncode == 1


def test_a_checkout_refusing_pushes_to_its_branch_fails(host: Host) -> None:
    assert host.rows()[1]['checkouts']['verdict'] == 'ok'
    subprocess.run(
        ['git', '-C', str(host.remote / 'p'), 'config', '--unset', 'receive.denyCurrentBranch'],
        check=True, timeout=CLI_TIMEOUT,
    )
    row = host.rows()[1]['checkouts']
    assert row['verdict'] == 'fail'
    assert str(host.remote / 'p') in row['observed']
    assert 'receive.denyCurrentBranch' in row['observed']


def test_an_unreachable_host_fails_every_row_without_a_traceback(host: Host) -> None:
    returncode, rows = host.rows(FAKE_SSH_UNREACHABLE='1')
    assert returncode == 1
    assert rows.pop('network')['verdict'] == 'fail'
    for row_id, row in rows.items():
        assert row['verdict'] == 'fail', row_id
        assert row['observed'].startswith('not observed'), row_id


def test_an_unpinned_host_key_fails_the_network_row(host: Host) -> None:
    row = host.rows(FAKE_SSH_UNPINNED='1')[1]['network']
    assert row['verdict'] == 'fail'
    assert 'known_hosts' in row['observed']


def test_the_text_table_has_a_header_and_one_line_per_requirement(host: Host) -> None:
    proc = host.check('box', '--roster', str(host.roster))
    lines = proc.stdout.splitlines()
    assert len(lines) == 1 + len(_table_ids())
    assert [line.split()[0] for line in lines[1:]] == _table_ids()


def test_the_roster_defaults_to_orch_verify_hosts_file(host: Host) -> None:
    proc = host.check('box', '--json', ORCH_VERIFY_HOSTS_FILE=str(host.roster))
    assert json.loads(proc.stdout)['host'] == 'box'


def test_an_unknown_host_exits_2_with_the_roster_error(host: Host) -> None:
    proc = host.check('nobox', '--roster', str(host.roster))
    assert proc.returncode == 2
    assert 'nobox' in proc.stderr
    assert proc.stdout == ''


def test_no_way_to_locate_the_roster_exits_2_naming_the_variables(host: Host) -> None:
    proc = host.check('box', HOME='')
    assert proc.returncode == 2
    assert 'Traceback' not in proc.stderr
    assert 'ORCH_VERIFY_HOSTS_FILE' in proc.stderr


def test_a_malformed_roster_exits_2_rather_than_reporting_no_hosts(host: Host) -> None:
    host.roster.write_text('hosts: [unclosed\n')
    proc = host.check('box', '--roster', str(host.roster))
    assert proc.returncode == 2
    assert str(host.roster) in proc.stderr
    assert proc.stdout == ''
