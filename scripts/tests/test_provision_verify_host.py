"""scripts/deploy/provision_verify_host.sh, driven through a fake ssh that runs the host locally."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from cli_subprocess_timeout import cli_timeout_from_env
from fake_verify_host_ssh import write_fake_ssh
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

DEPLOY = Path(__file__).resolve().parents[1] / 'deploy'
PROVISION = DEPLOY / 'provision_verify_host.sh'
CONTRACT = DEPLOY / 'verify_host_contract.py'
CLI_TIMEOUT = cli_timeout_from_env('PROVISION_VERIFY_HOST_TEST_TIMEOUT')


class Scene:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.remote = tmp_path / 'remote'
        self.home = tmp_path / 'home'
        self.home.mkdir()
        (self.home / 'marker').write_text('untouched\n')
        self.ssh_log = tmp_path / 'ssh.log'
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
        host_config: {{project_root: {self.remote}/p, verify_cgroup_memory: {{max_gib: 4}}}}
      q:
        git_remote: box
        config_path: {self.remote}/config/q.yaml
        host_config: {{project_root: {self.remote}/q, git: {{main_branch: trunk}}}}
""")
        self.env = {
            'HOME': str(self.home),
            'XDG_STATE_HOME': str(tmp_path / 'state'),
            'PATH': os.environ['PATH'],
            'LANG': os.environ.get('LANG', 'C.UTF-8'),
            'VERIFY_HOST_PYTHON': sys.executable,
            'VERIFY_HOST_SSH': str(write_fake_ssh(tmp_path)),
            'VERIFY_HOST_WRAPPER_PATH': str(self.remote / 'bin' / 'orchestrator'),
            'FAKE_SSH_LOG': str(self.ssh_log),
        }

    def run(self, *args: str, script: Path = PROVISION) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(script), *args], env=self.env, capture_output=True, text=True, timeout=CLI_TIMEOUT, check=False,
        )

    def ssh_calls(self) -> list[dict]:
        if not self.ssh_log.exists():
            return []
        return [json.loads(line) for line in self.ssh_log.read_text().splitlines()]

    def home_snapshot(self) -> dict[str, bytes]:
        return {str(path): path.read_bytes() for path in sorted(self.home.rglob('*')) if path.is_file()}


@pytest.fixture
def scene(tmp_path: Path) -> Scene:
    return Scene(tmp_path)


def _table_ids() -> list[str]:
    proc = subprocess.run(
        [sys.executable, str(CONTRACT), 'requirements'],
        capture_output=True, text=True, timeout=CLI_TIMEOUT, check=True,
    )
    return proc.stdout.splitlines()


def test_the_scripts_steps_are_the_checkers_requirements(scene: Scene) -> None:
    proc = scene.run('--list-steps')
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines() == _table_ids()


def test_a_dry_run_names_each_step_and_touches_nothing(scene: Scene) -> None:
    proc = scene.run('box', '--roster', str(scene.roster), '--dry-run')
    assert proc.returncode == 0, proc.stderr
    step_lines = [line for line in proc.stdout.splitlines() if line.split(':')[0] in _table_ids()]
    assert [line.split(':')[0] for line in step_lines] == _table_ids()
    assert 'box-alias' in proc.stdout
    assert f'{scene.remote}/df' in proc.stdout
    assert f'checkout {scene.remote}/p on main ' in proc.stdout
    assert f'checkout {scene.remote}/q on trunk ' in proc.stdout
    assert scene.ssh_calls() == []
    assert not scene.remote.exists()


class TestSyncConfig:
    @pytest.fixture
    def rendered(self, scene: Scene) -> Path:
        out = scene.tmp / 'rendered'
        subprocess.run(
            [sys.executable, str(CONTRACT), 'render', 'box', '--roster', str(scene.roster), '--out', str(out)],
            env=scene.env, capture_output=True, timeout=CLI_TIMEOUT, check=True,
        )
        return out

    def test_it_installs_each_projects_render(self, scene: Scene, rendered: Path) -> None:
        proc = scene.run('box', '--roster', str(scene.roster), '--sync-config')
        assert proc.returncode == 0, proc.stderr
        for project in ('p', 'q'):
            assert (scene.remote / 'config' / f'{project}.yaml').read_bytes() == (rendered / f'{project}.yaml').read_bytes()

    def test_it_touches_nothing_else(self, scene: Scene) -> None:
        before = scene.home_snapshot()
        assert scene.run('box', '--roster', str(scene.roster), '--sync-config').returncode == 0
        assert sorted(path.name for path in scene.remote.iterdir()) == ['config']
        assert scene.home_snapshot() == before
        assert [call['step'] for call in scene.ssh_calls()] == ['per_host_yaml']

    def test_an_unchanged_render_is_not_rewritten(self, scene: Scene) -> None:
        configs = [scene.remote / 'config' / name for name in ('p.yaml', 'q.yaml')]
        assert scene.run('box', '--roster', str(scene.roster), '--sync-config').returncode == 0
        first = [(path.read_bytes(), path.stat().st_mtime_ns) for path in configs]
        assert scene.run('box', '--roster', str(scene.roster), '--sync-config').returncode == 0
        assert [(path.read_bytes(), path.stat().st_mtime_ns) for path in configs] == first


class TestRefusals:
    def test_an_unknown_host_makes_no_ssh_call(self, scene: Scene) -> None:
        proc = scene.run('nobox', '--roster', str(scene.roster), '--sync-config')
        assert proc.returncode != 0
        assert 'nobox' in proc.stderr
        assert scene.ssh_calls() == []

    def test_a_malformed_roster_makes_no_ssh_call(self, scene: Scene) -> None:
        scene.roster.write_text('hosts: [unclosed\n')
        proc = scene.run('box', '--roster', str(scene.roster), '--sync-config')
        assert proc.returncode != 0
        assert str(scene.roster) in proc.stderr
        assert scene.ssh_calls() == []

    @pytest.mark.parametrize('args', [(), ('box', '--roster', 'r.yaml', '--teleport')], ids=['no_host', 'unknown_flag'])
    def test_a_usage_error_exits_2(self, scene: Scene, args: tuple[str, ...]) -> None:
        proc = scene.run(*args)
        assert proc.returncode == 2
        assert scene.ssh_calls() == []


def test_a_step_list_out_of_step_with_the_table_is_refused(scene: Scene, tmp_path: Path) -> None:
    """The self-check, through its one input: a contract whose table drops 'network' and adds 'teleport'."""
    copy = tmp_path / 'deploy'
    copy.mkdir()
    script = copy / PROVISION.name
    script.write_text(PROVISION.read_text())
    script.chmod(0o755)
    ids = [row for row in _table_ids() if row != 'network'] + ['teleport']
    (copy / CONTRACT.name).write_text(f'print({chr(10).join(ids)!r})\n')
    proc = scene.run('--list-steps', script=script)
    assert proc.returncode == 2
    assert 'network' in proc.stderr
    assert 'teleport' in proc.stderr
