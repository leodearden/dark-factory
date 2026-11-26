"""scripts/deploy/verify_host_contract.py: the one requirement table and the renders a host installs."""

import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
import yaml
from cli_subprocess_timeout import cli_timeout_from_env
from orchestrator.config import OrchestratorConfig, VerifyCgroupMemoryConfig
from orchestrator.verify_runner import GIT_PUSH_SSH_CONFIG, REMOTE_TOOL_PATH_PRELUDE
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY = REPO_ROOT / 'scripts' / 'deploy'
CONTRACT = DEPLOY / 'verify_host_contract.py'
CLI_TIMEOUT = cli_timeout_from_env('VERIFY_HOST_CONTRACT_TEST_TIMEOUT')

TABLE_IDS = [
    'network', 'os_and_user', 'time_zone_and_locale', 'toolchain', 'systemd', 'kernel',
    'memory', 'disk', 'checkouts', 'wrapper', 'per_host_yaml', 'no_credentials',
]

PORTABLE_ARTIFACTS = [
    DEPLOY / 'verify_host_observe.py',
    DEPLOY / 'verify_host_contract.py',
    DEPLOY / 'check_verify_host.py',
    DEPLOY / 'provision_verify_host.sh',
    DEPLOY / 'verify-host-wrapper.template.sh',
    REPO_ROOT / 'orchestrator' / 'src' / 'orchestrator' / 'verify_host_roster_file.py',
]


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    module: Any = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _contract() -> Any:
    return _load('verify_host_contract', CONTRACT)


def _run(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CONTRACT), *args],
        env=env, capture_output=True, text=True, timeout=CLI_TIMEOUT, check=False,
    )


P_HOST_CONFIG = """
    project_root: /srv/host/p
    verify_cgroup_memory: {max_gib: 12}
    git: {main_branch: main}
"""


def _roster(tmp_path: Path, *, df: str | None = None, p_host_config: str = P_HOST_CONFIG) -> Path:
    df = df if df is not None else str(tmp_path / 'host' / 'df')
    text = f"""\
hosts:
  box:
    ssh_host: box-alias
    df_checkout_path: '{df}'
    wrapper_env:
      ORCH_WATCHDOG_HEARTBEAT_TIMEOUT_SECS: '600'
      SPACED: 'a b'
    projects:
      p:
        git_remote: box
        config_path: /srv/host/p.yaml
        host_config:
{textwrap.indent(textwrap.dedent(p_host_config).strip(), ' ' * 10)}
      q:
        git_remote: box
        config_path: /srv/host/q.yaml
        host_config:
          project_root: /srv/host/q
"""
    path = tmp_path / 'verify-hosts.yaml'
    path.write_text(text)
    return path


def _render(tmp_path: Path, roster: Path, host: str = 'box', out: str = 'out') -> subprocess.CompletedProcess[str]:
    return _run('render', host, '--roster', str(roster), '--out', str(tmp_path / out))


def _rendered(tmp_path: Path, roster: Path, out: str = 'out') -> Path:
    proc = _render(tmp_path, roster, out=out)
    assert proc.returncode == 0, proc.stderr
    return tmp_path / out


def _sourced(env_file: Path, name: str) -> list[str]:
    proc = subprocess.run(
        ['bash', '-c', 'source "$1"; declare -n value="$2"; printf "%s\\0" "${value[@]}"', '_', str(env_file), name],
        capture_output=True, text=True, timeout=CLI_TIMEOUT, check=True,
    )
    return proc.stdout.split('\0')[:-1]


def test_requirements_prints_the_table_ids_in_execution_order() -> None:
    proc = _run('requirements')
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines() == TABLE_IDS


def test_the_table_in_process_matches_the_cli() -> None:
    assert [row.id for row in _contract().REQUIREMENTS] == TABLE_IDS


class TestRender:
    def test_each_project_gets_its_host_config_yaml(self, tmp_path: Path) -> None:
        out = _rendered(tmp_path, _roster(tmp_path))
        p_text = (out / 'p.yaml').read_text()
        header, body = p_text.split('\n', 1)
        assert header.startswith('#')
        assert yaml.safe_load(body) == {
            'project_root': '/srv/host/p',
            'verify_cgroup_memory': {'max_gib': 12},
            'git': {'main_branch': 'main'},
        }
        assert yaml.safe_load((out / 'q.yaml').read_text()) == {'project_root': '/srv/host/q'}

    def test_key_order_in_the_roster_does_not_change_the_bytes(self, tmp_path: Path) -> None:
        first = _rendered(tmp_path, _roster(tmp_path), out='first') / 'p.yaml'
        swapped = """
            git: {main_branch: main}
            verify_cgroup_memory: {max_gib: 12}
            project_root: /srv/host/p
        """
        second = _rendered(tmp_path, _roster(tmp_path, p_host_config=swapped), out='second') / 'p.yaml'
        assert first.read_bytes() == second.read_bytes()

    def test_host_env_sources_into_aligned_arrays(self, tmp_path: Path) -> None:
        env_file = _rendered(tmp_path, _roster(tmp_path)) / 'host.env'
        contract = _contract()
        assert _sourced(env_file, 'SSH_HOST') == ['box-alias']
        assert _sourced(env_file, 'DF_CHECKOUT_PATH') == [f'{tmp_path}/host/df']
        assert _sourced(env_file, 'WRAPPER_PATH') == ['/usr/local/bin/orchestrator']
        assert _sourced(env_file, 'SSH_BIN') == ['ssh']
        assert _sourced(env_file, 'SSH_OPTS') == list(contract.DISPATCHER_SSH_OPTS)
        assert _sourced(env_file, 'PROJECT_IDS') == ['p', 'q']
        assert _sourced(env_file, 'CONFIG_PATHS') == ['/srv/host/p.yaml', '/srv/host/q.yaml']
        assert _sourced(env_file, 'PROJECT_ROOTS') == ['/srv/host/p', '/srv/host/q']

    def test_the_wrapper_is_written(self, tmp_path: Path) -> None:
        assert (_rendered(tmp_path, _roster(tmp_path)) / 'wrapper.sh').is_file()


class TestTheRenderedWrapper:
    @pytest.fixture
    def wrapper(self, tmp_path: Path) -> Path:
        orchestrator = tmp_path / 'host' / 'df' / '.venv' / 'bin' / 'orchestrator'
        orchestrator.parent.mkdir(parents=True)
        orchestrator.write_text('#!/bin/sh\nexit 3\n')
        orchestrator.chmod(0o755)
        wrapper = _rendered(tmp_path, _roster(tmp_path)) / 'wrapper.sh'
        wrapper.chmod(0o755)
        return wrapper

    @staticmethod
    def _call(tmp_path: Path, wrapper: Path, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        base = {'HOME': str(tmp_path), 'XDG_STATE_HOME': str(tmp_path / 'state'), 'PATH': os.environ['PATH']}
        return subprocess.run(
            [str(wrapper), *args], env={**base, **env},
            capture_output=True, text=True, timeout=CLI_TIMEOUT, check=False,
        )

    def test_it_parses(self, wrapper: Path) -> None:
        subprocess.run(['bash', '-n', str(wrapper)], check=True, timeout=CLI_TIMEOUT)

    def test_it_runs_orch_gate_bin_when_set(self, tmp_path: Path, wrapper: Path) -> None:
        proc = self._call(tmp_path, wrapper, 'gate-identity', ORCH_GATE_BIN='/bin/echo')
        assert (proc.returncode, proc.stdout) == (0, 'gate-identity\n')

    def test_it_falls_back_to_the_df_checkouts_orchestrator_and_logs(self, tmp_path: Path, wrapper: Path) -> None:
        proc = self._call(tmp_path, wrapper, 'verify-merge')
        assert proc.returncode == 3
        log = (tmp_path / 'state' / 'orchestrator' / 'wrapper.log').read_text()
        assert ' START ' in log
        assert 'EXIT rc=3' in log

    def test_it_puts_the_dispatchers_tool_path_first(self, tmp_path: Path, wrapper: Path) -> None:
        proc = self._call(tmp_path, wrapper, ORCH_GATE_BIN='/usr/bin/env')
        path_line = next(line for line in proc.stdout.splitlines() if line.startswith('PATH='))
        assert path_line.startswith(f'PATH={tmp_path}/.local/bin:')

    def test_wrapper_env_entries_are_defaults_an_ambient_value_overrides(self, tmp_path: Path, wrapper: Path) -> None:
        defaults = self._call(tmp_path, wrapper, ORCH_GATE_BIN='/usr/bin/env').stdout.splitlines()
        assert 'ORCH_WATCHDOG_HEARTBEAT_TIMEOUT_SECS=600' in defaults
        assert 'SPACED=a b' in defaults
        overridden = self._call(
            tmp_path, wrapper, ORCH_GATE_BIN='/usr/bin/env', ORCH_WATCHDOG_HEARTBEAT_TIMEOUT_SECS='900',
        ).stdout.splitlines()
        assert 'ORCH_WATCHDOG_HEARTBEAT_TIMEOUT_SECS=900' in overridden


class TestRenderRefuses:
    def _refused(self, tmp_path: Path, roster: Path, host: str = 'box') -> str:
        proc = _render(tmp_path, roster, host)
        assert proc.returncode == 1, (proc.stdout, proc.stderr)
        assert not (tmp_path / 'out' / 'wrapper.sh').exists()
        return proc.stderr

    def test_a_host_config_without_project_root(self, tmp_path: Path) -> None:
        message = self._refused(tmp_path, _roster(tmp_path, p_host_config='git: {main_branch: main}'))
        assert all(word in message for word in ('box', "'p'", 'project_root'))

    @pytest.mark.parametrize('unsafe', ['"', '$'])
    def test_a_df_checkout_path_unsafe_inside_double_quotes(self, tmp_path: Path, unsafe: str) -> None:
        message = self._refused(tmp_path, _roster(tmp_path, df=f'/srv/d{unsafe}f'))
        assert all(word in message for word in ('box', 'df_checkout_path'))

    def test_a_relative_df_checkout_path(self, tmp_path: Path) -> None:
        message = self._refused(tmp_path, _roster(tmp_path, df='srv/df'))
        assert all(word in message for word in ('box', 'df_checkout_path'))

    def test_an_unknown_host(self, tmp_path: Path) -> None:
        assert 'nobox' in self._refused(tmp_path, _roster(tmp_path), host='nobox')

    def test_a_broken_roster_file(self, tmp_path: Path) -> None:
        roster = tmp_path / 'verify-hosts.yaml'
        roster.write_text('hosts: [unclosed\n')
        assert str(roster) in self._refused(tmp_path, roster)


class TestPins:
    def test_the_ssh_options_are_the_dispatchers(self) -> None:
        opts = _contract().DISPATCHER_SSH_OPTS
        assert ('-c', 'core.sshCommand=ssh ' + ' '.join(opts)) == GIT_PUSH_SSH_CONFIG

    def test_the_wrapper_path_line_is_the_dispatchers_prelude(self, tmp_path: Path) -> None:
        wrapper = (_rendered(tmp_path, _roster(tmp_path)) / 'wrapper.sh').read_text()
        assert f'export {REMOTE_TOOL_PATH_PRELUDE}' in wrapper.splitlines()

    def test_the_host_config_keys_it_reads_are_config_fields(self) -> None:
        contract = _contract()
        memory_key, max_gib_key = contract.MEMORY_MAX_GIB_KEYS
        assert contract.PROJECT_ROOT_KEY in OrchestratorConfig.model_fields
        assert memory_key in OrchestratorConfig.model_fields
        assert max_gib_key in VerifyCgroupMemoryConfig.model_fields


@pytest.mark.parametrize('artifact', PORTABLE_ARTIFACTS, ids=lambda path: path.name)
def test_no_artifact_names_a_home_directory(artifact: Path) -> None:
    text = artifact.read_text()
    assert '/home/' not in text
    assert str(Path.home()) not in text
