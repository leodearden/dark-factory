"""Tests for scripts/install-duplicate-audit-timer.sh and the two units it installs (task 3136).

Drives the installer via subprocess with `scripts/tests/conftest.py::fake_timer_systemctl`
first on PATH, and reads the committed units through
`scripts/tests/conftest.py::unit_values` rather than grepping their text. Real
systemd is never touched.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

SCRIPT = Path(__file__).parent.parent / 'install-duplicate-audit-timer.sh'
TEMPLATES_DIR = Path(__file__).parent.parent
AUDIT_CONFIG = TEMPLATES_DIR.parent / 'fused-memory' / 'config' / 'duplicate-audit.yaml'

# The installed unit is a byte copy naming the PRODUCTION checkout, so this is
# never derived from the repo root, which is .worktrees/<id> in a lane.
PRODUCTION_ROOT = '/home/leo/src/dark-factory'

SERVICE_NAME = 'fused-memory-duplicate-audit.service'
TIMER_NAME = 'fused-memory-duplicate-audit.timer'
WRAPPER_NAME = 'fused-memory-duplicate-audit.sh'
PATH_TWIN = 'fused-memory-flag-marker-sweep.service'


def _run_script(fake_timer_systemctl, *, env=None):
    fake_timer_systemctl.reset()
    full_env = dict(os.environ)
    full_env['PATH'] = f'{fake_timer_systemctl.bin_dir}{os.pathsep}{full_env["PATH"]}'
    if env:
        full_env.update(env)
    return subprocess.run(['bash', str(SCRIPT)], env=full_env, capture_output=True, text=True, timeout=60)


def _path_env(unit_values, name) -> list[str]:
    return [v for v in unit_values(TEMPLATES_DIR / name, 'Service', 'Environment') if v.startswith('PATH=')]


# ── the installer ───────────────────────────────────────────────────────────


@pytest.mark.parametrize('script', [SCRIPT.name, WRAPPER_NAME])
def test_both_shell_scripts_are_executable(script):
    path = TEMPLATES_DIR / script
    assert os.access(path, os.X_OK), f'chmod +x {path}'


def test_install_copies_both_units_and_enables_the_timer(fake_timer_systemctl, tmp_path):
    xdg_config = tmp_path / 'xdg-config'
    result = _run_script(fake_timer_systemctl, env={'XDG_CONFIG_HOME': str(xdg_config)})
    assert result.returncode == 0, f'stdout={result.stdout!r} stderr={result.stderr!r}'

    unit_dir = xdg_config / 'systemd' / 'user'
    for name in (SERVICE_NAME, TIMER_NAME):
        assert (unit_dir / name).read_bytes() == (TEMPLATES_DIR / name).read_bytes()
    calls = fake_timer_systemctl.calls()
    assert calls.index(['daemon-reload']) < calls.index(['enable', '--now', TIMER_NAME]), calls


def test_install_is_idempotent(fake_timer_systemctl, tmp_path):
    env = {'XDG_CONFIG_HOME': str(tmp_path / 'xdg-config')}
    first = _run_script(fake_timer_systemctl, env=env)
    first_calls = fake_timer_systemctl.calls()
    second = _run_script(fake_timer_systemctl, env=env)
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr

    assert fake_timer_systemctl.calls() == first_calls
    unit_dir = tmp_path / 'xdg-config' / 'systemd' / 'user'
    for name in (SERVICE_NAME, TIMER_NAME):
        assert (unit_dir / name).read_bytes() == (TEMPLATES_DIR / name).read_bytes()


def test_install_fails_loud_when_the_timer_is_absent_from_list_timers(fake_timer_systemctl, tmp_path):
    result = _run_script(fake_timer_systemctl, env={
        'XDG_CONFIG_HOME': str(tmp_path / 'xdg'),
        'FAKE_SYSTEMCTL_OMIT_LIST_TIMERS': '1',
    })
    assert result.returncode != 0, f'stdout={result.stdout!r} stderr={result.stderr!r}'
    assert 'ERROR' in result.stderr, result.stderr
    assert TIMER_NAME in result.stderr, result.stderr


def test_install_never_starts_the_service(fake_timer_systemctl, tmp_path):
    """The first run belongs on schedule, after the operator seeds the T0 baseline report."""
    result = _run_script(fake_timer_systemctl, env={'XDG_CONFIG_HOME': str(tmp_path / 'xdg')})
    assert result.returncode == 0, result.stderr
    for call in fake_timer_systemctl.calls():
        assert call[:1] != ['start'], f'unexpected immediate run: {call!r}'
        assert SERVICE_NAME not in call, f'the installer touched the service: {call!r}'


# ── the committed units ─────────────────────────────────────────────────────


def test_timer_fires_weekly_on_the_configured_cadence_catching_up_a_missed_week(unit_values):
    timer = TEMPLATES_DIR / TIMER_NAME
    declared = yaml.safe_load(AUDIT_CONFIG.read_text(encoding='utf-8'))['cadence']['on_calendar']
    assert unit_values(timer, 'Timer', 'OnCalendar') == [declared], (
        f'{TIMER_NAME} OnCalendar must equal cadence.on_calendar in {AUDIT_CONFIG.name}'
    )
    assert unit_values(timer, 'Timer', 'Persistent') == ['true']
    assert unit_values(timer, 'Timer', 'RandomizedDelaySec') == ['300']
    assert unit_values(timer, 'Install', 'WantedBy') == ['timers.target']


def test_timer_does_not_collide_with_an_occupied_slot(unit_values):
    """Parsed on both sides, so a commented-out slot can neither manufacture nor mask a clash."""
    ours = set(unit_values(TEMPLATES_DIR / TIMER_NAME, 'Timer', 'OnCalendar'))
    assert ours, 'the timer declares no OnCalendar at all'
    for other in sorted(TEMPLATES_DIR.glob('*.timer')):
        if other.name == TIMER_NAME:
            continue
        clash = ours & set(unit_values(other, 'Timer', 'OnCalendar'))
        assert not clash, (
            f'{TIMER_NAME} shares {sorted(clash)!r} with {other.name} — pick a free slot '
            f'and update the OPERATIONS.md nightly ladder table'
        )


def test_service_is_a_journalled_oneshot_around_the_wrapper(unit_values):
    service = TEMPLATES_DIR / SERVICE_NAME
    assert unit_values(service, 'Service', 'Type') == ['oneshot']
    assert unit_values(service, 'Service', 'ExecStart') == [f'{PRODUCTION_ROOT}/scripts/{WRAPPER_NAME}']
    assert unit_values(service, 'Service', 'WorkingDirectory') == [PRODUCTION_ROOT]
    assert unit_values(service, 'Service', 'StandardOutput') == ['journal']
    assert unit_values(service, 'Service', 'StandardError') == ['journal']


def test_service_path_reaches_uv_and_matches_the_fleet_spelling(unit_values):
    """A boot-catch-up run with the user manager's minimal PATH hit ``exec: uv: not found``."""
    ours = _path_env(unit_values, SERVICE_NAME)
    assert len(ours) == 1, ours
    assert '/home/leo/.local/bin' in ours[0].removeprefix('PATH=').split(':')
    assert ours == _path_env(unit_values, PATH_TWIN)


def test_service_documentation_links_resolve_in_the_repo(unit_values):
    docs = unit_values(TEMPLATES_DIR / SERVICE_NAME, 'Unit', 'Documentation')
    assert docs
    prefix = f'file://{PRODUCTION_ROOT}/'
    for link in docs:
        assert link.startswith(prefix), link
        assert (TEMPLATES_DIR.parent / link.removeprefix(prefix)).is_file(), link


def test_service_execstart_points_at_a_real_executable_wrapper(unit_values):
    named, = unit_values(TEMPLATES_DIR / SERVICE_NAME, 'Service', 'ExecStart')
    assert named.startswith(f'{PRODUCTION_ROOT}/scripts/'), named
    here = TEMPLATES_DIR / named.split('/scripts/', 1)[1]
    assert here.is_file(), here
    assert os.access(here, os.X_OK), here
