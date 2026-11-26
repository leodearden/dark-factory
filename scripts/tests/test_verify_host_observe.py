"""scripts/deploy/verify_host_observe.py: the stdlib observer the host checker ships over ssh."""

import dataclasses
import importlib.util
import json
import os
import pwd
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from cli_subprocess_timeout import cli_timeout_from_env
from orchestrator.agents.landlock import SYS_landlock_create_ruleset
from orchestrator.agents.landlock_exec import LANDLOCK_CREATE_RULESET_VERSION
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

OBSERVER = Path(__file__).resolve().parents[1] / 'deploy' / 'verify_host_observe.py'
OBSERVE_TIMEOUT = cli_timeout_from_env('VERIFY_HOST_OBSERVE_TEST_TIMEOUT')

ENVIRONMENT_FIELDS = {
    'time_zone', 'locale', 'locale_available', 'tzdata_legacy_installed', 'uv_version', 'node_version',
}


def _load_observer() -> Any:
    spec = importlib.util.spec_from_file_location('verify_host_observe_under_test', OBSERVER)
    assert spec is not None and spec.loader is not None, OBSERVER
    module: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _env(home: Path) -> dict[str, str]:
    return {
        'HOME': str(home),
        'XDG_STATE_HOME': str(home / 'state'),
        'PATH': os.environ['PATH'],
        'LANG': os.environ.get('LANG', 'C.UTF-8'),
    }


def _observe(tmp_path: Path, *args: str) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, '-I', str(OBSERVER), *args],
        env=_env(tmp_path), cwd=tmp_path, capture_output=True, text=True,
        timeout=OBSERVE_TIMEOUT, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def _git(*args: str) -> None:
    subprocess.run(['git', *args], check=True, capture_output=True, timeout=OBSERVE_TIMEOUT)


def _executable(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)
    return path


def _os_release() -> dict[str, str]:
    pairs = (line.split('=', 1) for line in Path('/etc/os-release').read_text().splitlines() if '=' in line)
    return {key: value.strip('"') for key, value in pairs}


def test_the_environment_facts_parse_back_strictly() -> None:
    observer = _load_observer()
    observed = dataclasses.asdict(observer.observe_environment())
    assert set(observed) == ENVIRONMENT_FIELDS
    facts = observer.EnvironmentFacts.from_json(json.dumps(observed))
    assert dataclasses.asdict(facts) == observed
    with pytest.raises(ValueError, match='surplus'):
        observer.EnvironmentFacts.from_json(json.dumps({**observed, 'surplus': 1}))
    with pytest.raises(ValueError, match='uv_version'):
        observer.EnvironmentFacts.from_json(json.dumps({k: v for k, v in observed.items() if k != 'uv_version'}))


class TestAFullObservation:
    @pytest.fixture
    def observed(self, tmp_path: Path) -> dict[str, Any]:
        df = tmp_path / 'df'
        project = tmp_path / 'project'
        _git('init', '-q', str(df))
        _git('-C', str(df), 'config', 'receive.denyCurrentBranch', 'updateInstead')
        _executable(df / '.venv' / 'bin' / 'orchestrator', '#!/bin/sh\n')
        _git('init', '-q', str(project))
        (tmp_path / 'p.yaml').write_text('project_root: /srv/p\n')
        (tmp_path / 'q.yaml').write_text('project_root: /srv/q\n')
        wrapper = _executable(tmp_path / 'wrapper', '#!/bin/bash\n"$ORCH_GATE_BIN" "$@"\n')
        (tmp_path / '.claude').mkdir()
        (tmp_path / '.claude' / '.credentials.json').write_text('{}')
        return _observe(
            tmp_path,
            '--df-checkout', str(df),
            '--project-checkout', str(project),
            '--config-path', str(tmp_path / 'p.yaml'),
            '--config-path', str(tmp_path / 'q.yaml'),
            '--config-path', str(tmp_path / 'absent.yaml'),
            '--wrapper-path', str(wrapper),
            '--disk-path', str(tmp_path / 'df'),
        )

    def test_the_environment_facts_are_nested(self, observed: dict[str, Any]) -> None:
        assert set(observed['environment']) == ENVIRONMENT_FIELDS

    def test_config_files_are_reported_verbatim(self, observed: dict[str, Any], tmp_path: Path) -> None:
        assert observed['configs'] == {
            str(tmp_path / 'p.yaml'): 'project_root: /srv/p\n',
            str(tmp_path / 'q.yaml'): 'project_root: /srv/q\n',
            str(tmp_path / 'absent.yaml'): None,
        }

    def test_each_checkouts_receive_setting_is_reported(self, observed: dict[str, Any], tmp_path: Path) -> None:
        assert observed['checkouts'] == {
            str(tmp_path / 'df'): {'exists': True, 'receive_deny_current_branch': 'updateInstead'},
            str(tmp_path / 'project'): {'exists': True, 'receive_deny_current_branch': None},
        }
        assert observed['df_orchestrator_executable'] is True

    def test_the_wrapper_is_read_and_probed_through_orch_gate_bin(
        self, observed: dict[str, Any], tmp_path: Path,
    ) -> None:
        assert observed['wrapper_text'] == (tmp_path / 'wrapper').read_text()
        assert observed['wrapper_probe_stdout'] == 'gate-identity'
        assert observed['wrapper_probe_rc'] == 0

    def test_an_agent_credential_file_is_detected(self, observed: dict[str, Any]) -> None:
        assert observed['credentials_present'] is True

    def test_capacity_facts_are_numbers(self, observed: dict[str, Any]) -> None:
        for key in ('mem_available_kib', 'swap_total_kib', 'disk_free_gib', 'landlock_abi'):
            assert type(observed[key]) is int, key
        assert isinstance(observed['sync_write_ms'], float)
        assert observed['sync_write_ms'] >= 0

    def test_identity_facts_match_this_process(self, observed: dict[str, Any]) -> None:
        assert observed['uid'] == os.getuid()
        assert observed['user'] == pwd.getpwuid(os.getuid()).pw_name
        os_release = _os_release()
        assert observed['os_id'] == os_release['ID']
        assert observed['os_version_id'] == os_release.get('VERSION_ID')

    def test_the_json_parses_back_into_host_facts(self, observed: dict[str, Any]) -> None:
        facts = _load_observer().HostFacts.from_json(json.dumps(observed))
        assert dataclasses.asdict(facts) == observed


def test_missing_paths_are_reported_not_raised(tmp_path: Path) -> None:
    gone = tmp_path / 'gone'
    observed = _observe(
        tmp_path,
        '--df-checkout', str(gone / 'df'),
        '--project-checkout', str(gone / 'project'),
        '--config-path', str(gone / 'p.yaml'),
        '--wrapper-path', str(gone / 'wrapper'),
        '--disk-path', str(gone / 'df'),
    )
    assert observed['checkouts'] == {
        str(gone / 'df'): {'exists': False, 'receive_deny_current_branch': None},
        str(gone / 'project'): {'exists': False, 'receive_deny_current_branch': None},
    }
    assert observed['configs'] == {str(gone / 'p.yaml'): None}
    assert observed['df_orchestrator_executable'] is False
    assert observed['wrapper_text'] is None
    assert observed['wrapper_probe_stdout'] is None
    assert observed['credentials_present'] is False
    assert type(observed['disk_free_gib']) is int


def test_the_landlock_probe_mirrors_the_orchestrators_constants() -> None:
    observer = _load_observer()
    assert observer.SYS_landlock_create_ruleset == SYS_landlock_create_ruleset
    assert observer.LANDLOCK_CREATE_RULESET_VERSION == LANDLOCK_CREATE_RULESET_VERSION
