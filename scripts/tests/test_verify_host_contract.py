"""scripts/deploy/verify_host_contract.py: the one requirement table and the renders a host installs."""

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
import yaml
from cli_subprocess_timeout import cli_timeout_from_env
from orchestrator.config import GitConfig, OrchestratorConfig, VerifyCgroupMemoryConfig
from orchestrator.verify_host_roster_file import load_roster
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
    git: {main_branch: trunk}
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


def test_requirements_titles_pairs_each_id_with_its_title() -> None:
    proc = _run('requirements', '--titles')
    assert proc.returncode == 0, proc.stderr
    pairs = [line.split('\t') for line in proc.stdout.splitlines()]
    assert pairs == [[row.id, row.title] for row in _contract().REQUIREMENTS]


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
            'git': {'main_branch': 'trunk'},
        }
        assert yaml.safe_load((out / 'q.yaml').read_text()) == {'project_root': '/srv/host/q'}

    def test_key_order_in_the_roster_does_not_change_the_bytes(self, tmp_path: Path) -> None:
        first = _rendered(tmp_path, _roster(tmp_path), out='first') / 'p.yaml'
        swapped = """
            git: {main_branch: trunk}
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
        assert _sourced(env_file, 'TOOL_PATH_PRELUDE') == [REMOTE_TOOL_PATH_PRELUDE]
        assert _sourced(env_file, 'PROJECT_IDS') == ['p', 'q']
        assert _sourced(env_file, 'CONFIG_PATHS') == ['/srv/host/p.yaml', '/srv/host/q.yaml']
        assert _sourced(env_file, 'PROJECT_ROOTS') == ['/srv/host/p', '/srv/host/q']
        assert _sourced(env_file, 'PROJECT_MAIN_BRANCHES') == ['trunk', contract.DEFAULT_MAIN_BRANCH]
        assert _sourced(env_file, 'REQUIRED_OS_ID') == [contract.REQUIRED_OS[0]]
        assert _sourced(env_file, 'REQUIRED_OS_VERSION_ID') == [contract.REQUIRED_OS[1]]

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

    @pytest.mark.parametrize('git', ['{main_branch: [trunk]}', '{main_branch: ""}', 'trunk'],
                             ids=['branch_not_a_string', 'branch_empty', 'git_not_a_mapping'])
    def test_a_host_config_main_branch_that_is_not_a_branch_name(self, tmp_path: Path, git: str) -> None:
        host_config = f'project_root: /srv/host/p\ngit: {git}'
        message = self._refused(tmp_path, _roster(tmp_path, p_host_config=host_config))
        assert all(word in message for word in ('box', "'p'", 'git.main_branch'))

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
        git_key, main_branch_key = contract.MAIN_BRANCH_KEYS
        assert contract.PROJECT_ROOT_KEY in OrchestratorConfig.model_fields
        assert memory_key in OrchestratorConfig.model_fields
        assert max_gib_key in VerifyCgroupMemoryConfig.model_fields
        assert git_key in OrchestratorConfig.model_fields
        assert main_branch_key in GitConfig.model_fields


@pytest.mark.parametrize('artifact', PORTABLE_ARTIFACTS, ids=lambda path: path.name)
def test_no_artifact_names_a_home_directory(artifact: Path) -> None:
    text = artifact.read_text()
    assert '/home/' not in text
    assert str(Path.home()) not in text


GIB_IN_KIB = 1024 * 1024

VERDICT_ROSTER = """\
hosts:
  box:
    ssh_host: box-alias
    df_checkout_path: /srv/box/df
    projects:
      p:
        git_remote: box
        config_path: /srv/box/p.yaml
        disk_footprint_gib: 40
        host_config: {project_root: /srv/box/p, verify_cgroup_memory: {max_gib: 12}}
      q:
        git_remote: box
        config_path: /srv/box/q.yaml
        disk_footprint_gib: 10
        host_config: {project_root: /srv/box/q, verify_cgroup_memory: {max_gib: 8}}
"""

UNSTATED_ROSTER = """\
hosts:
  box:
    ssh_host: box-alias
    df_checkout_path: /srv/box/df
    projects:
      p:
        git_remote: box
        config_path: /srv/box/p.yaml
        host_config: {project_root: /srv/box/p}
"""

REFERENCE = {
    'time_zone': 'Europe/London',
    'locale': 'en_US.UTF-8',
    'locale_available': True,
    'tzdata_legacy_installed': True,
    'uv_version': '0.11.6',
    'node_version': 'v22.23.4',
}


class Scene:
    """One roster host, a reference workstation and a host observation where every row is ok."""

    def __init__(self, tmp_path: Path, roster_text: str = VERDICT_ROSTER) -> None:
        self.contract = _contract()
        self.observer = self.contract.verify_host_observe
        roster = tmp_path / 'verify-hosts.yaml'
        roster.write_text(roster_text)
        self.spec = self.contract.HostSpec.from_roster(load_roster(roster), 'box')
        self.reference = self.observer.EnvironmentFacts.from_json(json.dumps(REFERENCE))

    def baseline(self) -> dict[str, Any]:
        roots = [self.spec.df_checkout_path, *(project.project_root for project in self.spec.projects)]
        return {
            'environment': dict(REFERENCE),
            'os_id': 'ubuntu', 'os_version_id': '24.04', 'user': 'verify', 'uid': 1000,
            'git_user_name': 'Verify Host', 'git_user_email': 'verify@example.invalid',
            'git_version': 'git version 2.43.0', 'bash_version': 'GNU bash, version 5.2.21',
            'flock_version': 'flock from util-linux 2.39.3', 'npm_version': '10.9.9',
            'uv_python_313': '/opt/uv/python/cpython-3.13/bin/python3.13',
            'systemd_run_scope_rc': 0, 'linger': 'yes',
            'landlock_abi': 6, 'var_tmp_writable': True,
            'mem_available_kib': 32 * GIB_IN_KIB, 'swap_total_kib': 8 * GIB_IN_KIB,
            'disk_free_gib': 100, 'sync_write_ms': 2.0,
            'checkouts': {root: {'exists': True, 'receive_deny_current_branch': 'updateInstead'} for root in roots},
            'df_orchestrator_executable': True,
            'wrapper_text': self.contract.render_wrapper(self.spec),
            'wrapper_probe_stdout': 'gate-identity', 'wrapper_probe_rc': 0,
            'configs': {
                project.config_path: self.contract.render_host_config(project.host_config)
                for project in self.spec.projects
            },
            'credentials_present': False,
        }

    def observation(self, facts: dict[str, Any] | None = None, **network: Any) -> Any:
        host = self.observer.HostFacts.from_json(json.dumps(facts if facts is not None else self.baseline()))
        network_facts = self.contract.NetworkFacts(**{'reachable': True, 'host_key_pinned': True, 'detail': '', **network})
        return self.contract.Observation(network=network_facts, host=host, unobserved_reason='')

    def verdict(self, row_id: str, observation: Any = None, reference: dict[str, Any] | None = None) -> Any:
        (row,) = (row for row in self.contract.REQUIREMENTS if row.id == row_id)
        reference_facts = (
            self.reference if reference is None
            else self.observer.EnvironmentFacts.from_json(json.dumps(reference))
        )
        return row.evaluate(observation or self.observation(), reference_facts, self.spec)

    def changed(self, row_id: str, change: Any) -> str:
        facts = self.baseline()
        change(facts)
        return self.verdict(row_id, self.observation(facts)).verdict


@pytest.fixture
def scene(tmp_path: Path) -> Scene:
    return Scene(tmp_path)


def _set(path: str, value: Any) -> Any:
    *parents, leaf = path.split('.')

    def change(facts: dict[str, Any]) -> None:
        target = facts
        for key in parents:
            target = target[key]
        target[leaf] = value

    return change


class TestTheTable:
    def test_every_row_is_ok_on_the_baseline(self, scene: Scene) -> None:
        verdicts = {row.id: scene.verdict(row.id) for row in scene.contract.REQUIREMENTS}
        assert {row_id: verdict.verdict for row_id, verdict in verdicts.items()} == dict.fromkeys(TABLE_IDS, 'ok')

    def test_every_row_states_what_it_observed_and_needed(self, scene: Scene) -> None:
        for row in scene.contract.REQUIREMENTS:
            verdict = scene.verdict(row.id)
            assert verdict.observed and verdict.needed, row.id

    def test_an_unobserved_host_fails_every_host_row_but_not_the_network_row(self, scene: Scene) -> None:
        observation = scene.contract.Observation(
            network=scene.contract.NetworkFacts(reachable=True, host_key_pinned=True, detail=''),
            host=None, unobserved_reason='observer exited 1',
        )
        for row in scene.contract.REQUIREMENTS:
            verdict = scene.verdict(row.id, observation)
            if row.id == 'network':
                assert verdict.verdict == 'ok'
            else:
                assert verdict.verdict == 'fail', row.id
                assert verdict.observed == 'not observed: observer exited 1', row.id

    def test_an_unobserved_row_states_the_same_need_as_an_observed_one(self, scene: Scene) -> None:
        unobserved = scene.contract.Observation(
            network=scene.contract.NetworkFacts(reachable=True, host_key_pinned=True, detail=''),
            host=None, unobserved_reason='observer exited 1',
        )
        for row in scene.contract.REQUIREMENTS:
            assert scene.verdict(row.id, unobserved).needed == scene.verdict(row.id).needed, row.id


class TestNetwork:
    def test_unreachable_fails(self, scene: Scene) -> None:
        assert scene.verdict('network', scene.observation(reachable=False, detail='timed out')).verdict == 'fail'

    def test_reachable_but_the_host_key_not_pinned_fails(self, scene: Scene) -> None:
        assert scene.verdict('network', scene.observation(host_key_pinned=False)).verdict == 'fail'


class TestOsAndUser:
    @pytest.mark.parametrize('change', [
        _set('os_version_id', '22.04'), _set('os_id', 'debian'), _set('uid', 0),
        _set('git_user_name', None), _set('git_user_email', None),
    ], ids=['os_22_04', 'not_ubuntu', 'root', 'no_git_name', 'no_git_email'])
    def test_fails(self, scene: Scene, change: Any) -> None:
        assert scene.changed('os_and_user', change) == 'fail'


class TestTimeZoneAndLocale:
    @pytest.mark.parametrize('change', [
        _set('environment.time_zone', 'UTC'), _set('environment.locale', 'C.UTF-8'),
        _set('environment.locale_available', False), _set('environment.tzdata_legacy_installed', False),
    ], ids=['time_zone_differs', 'locale_differs', 'locale_unavailable', 'no_tzdata_legacy'])
    def test_fails(self, scene: Scene, change: Any) -> None:
        assert scene.changed('time_zone_and_locale', change) == 'fail'


class TestToolchain:
    @pytest.mark.parametrize('change', [
        _set('environment.uv_version', '0.11.5'), _set('environment.node_version', 'v20.11.1'),
        _set('npm_version', None), _set('git_version', None), _set('flock_version', None),
        _set('bash_version', None), _set('uv_python_313', None),
    ], ids=['uv_differs', 'node_major_differs', 'no_npm', 'no_git', 'no_flock', 'no_bash', 'no_python_313'])
    def test_fails(self, scene: Scene, change: Any) -> None:
        assert scene.changed('toolchain', change) == 'fail'

    def test_the_same_node_major_with_another_patch_is_ok(self, scene: Scene) -> None:
        assert scene.changed('toolchain', _set('environment.node_version', 'v22.1.0')) == 'ok'


class TestSystemd:
    @pytest.mark.parametrize('change', [_set('systemd_run_scope_rc', 1), _set('linger', 'no')],
                             ids=['scope_fails', 'no_linger'])
    def test_fails(self, scene: Scene, change: Any) -> None:
        assert scene.changed('systemd', change) == 'fail'


class TestKernel:
    @pytest.mark.parametrize('change', [
        _set('landlock_abi', 1), _set('landlock_abi', -1), _set('var_tmp_writable', False),
    ], ids=['landlock_abi_1', 'no_landlock', 'var_tmp_read_only'])
    def test_fails(self, scene: Scene, change: Any) -> None:
        assert scene.changed('kernel', change) == 'fail'

    def test_the_need_and_the_verdict_share_the_contract_minimum(self, scene: Scene) -> None:
        minimum = scene.contract.MIN_LANDLOCK_ABI
        assert scene.changed('kernel', _set('landlock_abi', minimum - 1)) == 'fail'
        assert scene.changed('kernel', _set('landlock_abi', minimum)) == 'ok'
        assert f'Landlock ABI {minimum}' in scene.verdict('kernel').needed


class TestMemory:
    def test_no_swap_fails(self, scene: Scene) -> None:
        assert scene.changed('memory', _set('swap_total_kib', 0)) == 'fail'

    def test_mem_available_below_the_summed_scope_caps_fails(self, scene: Scene) -> None:
        assert scene.changed('memory', _set('mem_available_kib', 19 * GIB_IN_KIB)) == 'fail'

    def test_no_stated_scope_cap_warns(self, tmp_path: Path) -> None:
        assert Scene(tmp_path, UNSTATED_ROSTER).verdict('memory').verdict == 'warn'


class TestDisk:
    def test_free_space_below_the_summed_footprints_fails(self, scene: Scene) -> None:
        assert scene.changed('disk', _set('disk_free_gib', 49)) == 'fail'

    def test_no_stated_footprint_warns(self, tmp_path: Path) -> None:
        assert Scene(tmp_path, UNSTATED_ROSTER).verdict('disk').verdict == 'warn'

    def test_a_slow_synchronous_write_warns(self, scene: Scene) -> None:
        assert scene.changed('disk', _set('sync_write_ms', 126.0)) == 'warn'

    def test_a_synchronous_write_under_the_threshold_is_ok(self, scene: Scene) -> None:
        fast = scene.contract.SYNC_WRITE_WARN_MS - 0.5
        assert scene.changed('disk', _set('sync_write_ms', fast)) == 'ok'


class TestCheckouts:
    def test_an_absent_project_checkout_fails(self, scene: Scene) -> None:
        change = _set('checkouts./srv/box/p', {'exists': False, 'receive_deny_current_branch': None})
        assert scene.changed('checkouts', change) == 'fail'

    def test_a_checkout_refusing_pushes_to_its_branch_fails(self, scene: Scene) -> None:
        change = _set('checkouts./srv/box/df', {'exists': True, 'receive_deny_current_branch': None})
        assert scene.changed('checkouts', change) == 'fail'

    def test_a_df_venv_without_its_orchestrator_fails(self, scene: Scene) -> None:
        assert scene.changed('checkouts', _set('df_orchestrator_executable', False)) == 'fail'


class TestWrapper:
    def test_an_installed_wrapper_differing_from_the_render_fails(self, scene: Scene) -> None:
        def change(facts: dict[str, Any]) -> None:
            facts['wrapper_text'] += '# hand edit\n'

        assert scene.changed('wrapper', change) == 'fail'

    def test_a_wrapper_ignoring_orch_gate_bin_fails(self, scene: Scene) -> None:
        assert scene.changed('wrapper', _set('wrapper_probe_stdout', 'usage: orchestrator')) == 'fail'


class TestPerHostYaml:
    def test_a_missing_config_file_fails_naming_the_project(self, scene: Scene) -> None:
        facts = scene.baseline()
        facts['configs']['/srv/box/q.yaml'] = None
        verdict = scene.verdict('per_host_yaml', scene.observation(facts))
        assert verdict.verdict == 'fail'
        assert "'q'" in verdict.observed

    def test_a_config_file_differing_from_the_render_fails_naming_the_project(self, scene: Scene) -> None:
        facts = scene.baseline()
        facts['configs']['/srv/box/p.yaml'] += 'concurrent_verify: true\n'
        verdict = scene.verdict('per_host_yaml', scene.observation(facts))
        assert verdict.verdict == 'fail'
        assert "'p'" in verdict.observed


class TestNoCredentials:
    def test_an_agent_credential_file_fails(self, scene: Scene) -> None:
        assert scene.changed('no_credentials', _set('credentials_present', True)) == 'fail'
