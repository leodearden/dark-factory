"""Tests for install_laptop_transcript_secret_sweep.sh and the laptop drop-in it installs.

The script is driven by subprocess against a fake laptop: the ssh shim drops
the host argument and runs the rest through `bash -c` from $HOME, the way a
real ssh login shell would, so the script's single-quoted heredoc payloads,
and the remote checkout's own install-transcript-secret-sweep-timer.sh, run on
this machine. The shim harness is a copy of the one in
scripts/tests/test_install_laptop_merge_verify_trim.py, under the house
duplicate-with-docstring convention. The fake remote checkout holds byte
copies of the committed sweep files, except that its service's
Environment=PATH= line points at the fake home's .local/bin, so whether uv
resolves there is the test's choice and not this host's. Real systemd, ssh
and the real laptop are never touched.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import transcript_secret_sweep as sweep
from shared.secret_env import is_secret_env_name
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
SCRIPT = SCRIPTS_DIR / 'deploy' / 'install_laptop_transcript_secret_sweep.sh'
DROP_IN = SCRIPTS_DIR / 'transcript-secret-sweep.laptop.conf'

# The drop-in is a byte copy naming the PRODUCTION checkout, so this is never
# derived from the repo root, which is .worktrees/<id> in a lane.
PRODUCTION_ROOT = '/home/leo/src/dark-factory'

UNIT = 'transcript-secret-sweep'
SWEEP_FILES = (
    'scripts/transcript-secret-sweep.sh',
    'scripts/install-transcript-secret-sweep-timer.sh',
    f'scripts/{UNIT}.service',
    f'scripts/{UNIT}.timer',
    f'scripts/{UNIT}.laptop.conf',
)
FAKE_ENV_VALUE = 'fake-laptop-token-' + 'L' * 30

SHIMS = {
    'ssh': 'shift\ncd "$HOME"\nexec bash -c "$*"\n',
    'systemctl': (
        f'dropin="${{XDG_CONFIG_HOME:-$HOME/.config}}/systemd/user/{UNIT}.service.d/laptop.conf"\n'
        'if [ -f "$dropin" ]; then has=dropin; else has=no-dropin; fi\n'
        'printf \'%s\\t%s\\n\' "$*" "$has" >>"$FAKE_STATE/systemctl.log"\n'
        'case "$*" in\n'
        '  "--user show-environment") [ -z "${FAKE_NO_BUS:-}" ] ;;\n'
        '  "--user enable --now "*) touch "$FAKE_STATE/enabled" ;;\n'
        '  "--user is-enabled "*)\n'
        '    if [ -f "$FAKE_STATE/enabled" ]; then echo enabled; else echo disabled; exit 1; fi ;;\n'
        f'  "--user list-timers --all") echo "Mon 02:30 {UNIT}.timer {UNIT}.service" ;;\n'
        '  *) exit 0 ;;\n'
        'esac\n'
    ),
}
READ_ONLY_SYSTEMCTL = ('--user show-environment', f'--user is-enabled {UNIT}.timer', '--user list-timers --all')

# /bin/sh, not `env bash`: the preflight runs uv under the service's PATH,
# which in the fake checkout is the fake home's .local/bin alone.
FAKE_UV = '#!/bin/sh\nprintf \'%s\\n\' "$*" >>"$FAKE_STATE/uv.log"\nexit 0\n'


def _directives(path: Path) -> dict[str, list[tuple[str, str]]]:
    """Parse a unit into ``{section: [(key, value), ...]}``: full-line comments skipped, duplicate keys kept.

    A copy of scripts/tests/test_install_transcript_secret_sweep_timer.py::_directives.
    """
    sections: dict[str, list[tuple[str, str]]] = {}
    current = ''
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line[0] in '#;':
            continue
        if line.startswith('[') and line.endswith(']'):
            current = line[1:-1].strip()
            sections.setdefault(current, [])
            continue
        if '=' not in line:
            continue
        key, value = line.split('=', 1)
        sections.setdefault(current, []).append((key.strip(), value.strip()))
    return sections


def _exec_starts(path: Path) -> list[str]:
    return [value for key, value in _directives(path).get('Service', []) if key == 'ExecStart']


def _here(production_path: str) -> Path:
    """*production_path*, a path under the production checkout, mapped onto this checkout."""
    assert production_path.startswith(f'{PRODUCTION_ROOT}/'), production_path
    return SCRIPTS_DIR.parent / production_path.removeprefix(f'{PRODUCTION_ROOT}/')


@pytest.fixture
def world(tmp_path: Path) -> dict[str, Path]:
    """A fake laptop: its dark-factory checkout, its home, and the shims."""
    home = tmp_path / 'home'
    (home / '.config' / 'orchestrator').mkdir(parents=True)
    uv = home / '.local' / 'bin' / 'uv'
    uv.parent.mkdir(parents=True)
    uv.write_text(FAKE_UV)
    uv.chmod(0o755)
    repo = tmp_path / 'remote-dark-factory'
    for rel in SWEEP_FILES:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SCRIPTS_DIR.parent / rel, repo / rel)
    service = repo / 'scripts' / f'{UNIT}.service'
    service.write_text(''.join(
        f'Environment=PATH={uv.parent}\n' if line.startswith('Environment=PATH=') else line
        for line in service.read_text().splitlines(keepends=True)
    ))
    (repo / '.env').write_text(f'CLAUDE_OAUTH_TOKEN_Z={FAKE_ENV_VALUE}\n')
    state = tmp_path / 'state'
    state.mkdir()
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    for name, body in SHIMS.items():
        shim = bin_dir / name
        shim.write_text('#!/usr/bin/env bash\n' + body)
        shim.chmod(0o755)
    return {'home': home, 'repo': repo, 'state': state, 'bin': bin_dir, 'uv': uv}


def run(world: dict[str, Path], *args: str, **env_overrides: str) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        'PATH': f'{world["bin"]}{os.pathsep}{os.environ["PATH"]}',
        'HOME': str(world['home']),
        'XDG_CONFIG_HOME': str(world['home'] / '.config'),
        'SSH': str(world['bin'] / 'ssh'),
        'LAPTOP_HOST': 'dummy-host',
        'REMOTE_REPO': str(world['repo']),
        'FAKE_STATE': str(world['state']),
        **env_overrides,
    }
    return subprocess.run([str(SCRIPT), *args], env=env, capture_output=True, text=True, timeout=60)


def unit_dir(world: dict[str, Path]) -> Path:
    return world['home'] / '.config' / 'systemd' / 'user'


def installed_drop_in(world: dict[str, Path]) -> Path:
    return unit_dir(world) / f'{UNIT}.service.d' / 'laptop.conf'


def systemctl_calls(world: dict[str, Path]) -> list[tuple[str, str]]:
    log = world['state'] / 'systemctl.log'
    lines = log.read_text().splitlines() if log.exists() else []
    return [(call, drop_in) for call, drop_in in (line.split('\t') for line in lines)]


def assert_nothing_installed(world: dict[str, Path]) -> None:
    assert not installed_drop_in(world).exists()
    assert not (unit_dir(world) / f'{UNIT}.timer').exists()
    assert not (unit_dir(world) / f'{UNIT}.service').exists()
    assert [call for call, _ in systemctl_calls(world) if call not in READ_ONLY_SYSTEMCTL] == []


# ── the committed drop-in ───────────────────────────────────────────────────


def test_the_drop_in_replaces_execstart_with_the_wrapper_and_the_laptop_flags():
    assert _exec_starts(DROP_IN) == [
        '',
        f'{PRODUCTION_ROOT}/scripts/transcript-secret-sweep.sh '
        '--host-configs-dir /home/leo/.config/orchestrator --no-escalation',
    ]


def test_the_drop_in_names_the_committed_executable_wrapper():
    wrapper = _here(_exec_starts(DROP_IN)[-1].split()[0])

    assert wrapper == SCRIPTS_DIR / 'transcript-secret-sweep.sh'
    assert os.access(wrapper, os.X_OK)


def test_the_sweep_accepts_the_drop_ins_arguments(monkeypatch, tmp_path):
    for name in list(os.environ):
        if is_secret_env_name(name):
            monkeypatch.delenv(name)
    monkeypatch.setenv('CLAUDE_OAUTH_TOKEN_A', 'sk-ant-oat01-' + 'Dr' * 17 + 'D')
    empty_units = tmp_path / 'units'
    empty_units.mkdir()
    drop_in_args = _exec_starts(DROP_IN)[-1].split()[1:]

    assert sweep.main([
        *drop_in_args,
        '--units-dir', str(empty_units),
        '--claude-projects-dir', str(tmp_path),
        '--pytest-tmp-dir', str(tmp_path),
        '--host-configs-dir', str(tmp_path),
        '--dry-run',
    ]) == 0


# ── the installer ───────────────────────────────────────────────────────────


def test_script_is_executable():
    assert os.access(SCRIPT, os.X_OK), f'chmod +x {SCRIPT}'


def test_apply_installs_the_drop_in_before_enabling_the_timer(world):
    result = run(world)

    assert result.returncode == 0, result.stderr
    assert installed_drop_in(world).read_bytes() == DROP_IN.read_bytes()
    for name in (f'{UNIT}.service', f'{UNIT}.timer'):
        assert (unit_dir(world) / name).read_bytes() == (world['repo'] / 'scripts' / name).read_bytes()
    calls = systemctl_calls(world)
    reloads = {drop_in for call, drop_in in calls if call == '--user daemon-reload'}
    assert reloads == {'dropin'}
    assert dict(calls)[f'--user enable --now {UNIT}.timer'] == 'dropin'


def test_the_preflight_imports_the_sweep_exactly_as_the_wrapper_runs_it(world):
    assert run(world, '--check').returncode == 0
    repo = world['repo']
    wrapper_env = {
        **{name: value for name, value in os.environ.items() if name != 'TRANSCRIPT_SECRET_SWEEP_CMD'},
        'PATH': f'{world["uv"].parent}{os.pathsep}{os.environ["PATH"]}',
        'REPO': str(repo),
        'FAKE_STATE': str(world['state']),
    }
    wrapper = subprocess.run(
        [str(repo / 'scripts' / f'{UNIT}.sh'), '--help'], env=wrapper_env, capture_output=True, text=True, timeout=60
    )

    assert wrapper.returncode == 0, wrapper.stderr
    preflight_call, wrapper_call = (world['state'] / 'uv.log').read_text().splitlines()
    assert preflight_call == wrapper_call
    assert preflight_call == f'run --no-sync --project {repo}/shared python {repo}/scripts/transcript_secret_sweep.py --help'


def test_a_second_apply_is_a_no_op_that_still_succeeds(world):
    assert run(world).returncode == 0
    first = installed_drop_in(world).read_bytes()

    result = run(world)

    assert result.returncode == 0, result.stderr
    assert installed_drop_in(world).read_bytes() == first


@pytest.mark.parametrize('flag', ['--check', '--dry-run'])
def test_check_mode_installs_nothing(world, flag):
    result = run(world, flag)

    assert result.returncode == 0, result.stderr
    assert_nothing_installed(world)


def test_no_run_is_ever_started(world):
    assert run(world).returncode == 0

    assert not [call for call, _ in systemctl_calls(world) if ' start ' in f'{call} ']


def test_the_env_value_is_never_printed(world):
    result = run(world)

    assert result.returncode == 0, result.stderr
    assert FAKE_ENV_VALUE not in result.stdout + result.stderr


def _without_drop_in(world):
    (world['repo'] / 'scripts' / f'{UNIT}.laptop.conf').unlink()


def _without_env(world):
    (world['repo'] / '.env').unlink()


def _without_uv(world):
    world['uv'].unlink()


def _without_orchestrator_configs(world):
    (world['home'] / '.config' / 'orchestrator').rmdir()


def _without_an_importable_sweep(world):
    world['uv'].write_text('#!/bin/sh\necho "transcript_secret_sweep.py: No module named \'yaml\'" >&2\nexit 2\n')


@pytest.mark.parametrize(
    'break_it',
    [_without_drop_in, _without_env, _without_uv, _without_orchestrator_configs, _without_an_importable_sweep],
    ids=[
        'checkout-predates-the-drop-in', 'no-env', 'uv-not-on-the-service-path', 'no-host-configs',
        'sweep-does-not-import',
    ],
)
def test_a_failed_preflight_refuses_and_installs_nothing(world, break_it):
    break_it(world)

    result = run(world)

    assert result.returncode != 0
    assert 'ERROR' in result.stderr
    assert_nothing_installed(world)


def test_no_systemd_user_bus_refuses(world):
    result = run(world, FAKE_NO_BUS='1')

    assert result.returncode != 0
    assert_nothing_installed(world)


def test_an_unreachable_host_refuses(world):
    result = run(world, SSH='false')

    assert result.returncode != 0
    assert 'ssh dummy-host failed' in result.stderr
    assert_nothing_installed(world)


def test_an_unexpected_argument_exits_2(world):
    result = run(world, '--bogus')

    assert result.returncode == 2
    assert 'unexpected argument' in result.stderr
    assert_nothing_installed(world)
