"""Tests for scripts/install-return-brief-timer.sh and the two systemd units it installs (task 5376).

Drives the installer via subprocess with `scripts/tests/conftest.py::fake_systemctl`
first on PATH, and reads the committed units through
`scripts/tests/conftest.py::unit_values` rather than grepping their text. Real
systemd is never touched.
"""
from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path

import pytest
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT
from sitting import nightly_prepare

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

SCRIPT = Path(__file__).parent.parent / 'install-return-brief-timer.sh'
TEMPLATES_DIR = Path(__file__).parent.parent

# The installed unit is a byte copy naming the PRODUCTION checkout, so this is
# never derived from the repo root, which is .worktrees/<id> in a lane.
PRODUCTION_ROOT = '/home/leo/src/dark-factory'

SERVICE_NAME = 'return-brief.service'
TIMER_NAME = 'return-brief.timer'
PATH_TWIN = 'fused-memory-flag-marker-sweep.service'


def _run_script(fake_systemctl, tmp_path, *, env=None, reset_state=True):
    if reset_state:
        fake_systemctl.reset()
    full_env = dict(os.environ)
    full_env['PATH'] = f'{fake_systemctl.bin_dir}{os.pathsep}{full_env["PATH"]}'
    if env:
        full_env.update(env)
    return subprocess.run(['bash', str(SCRIPT)], env=full_env, capture_output=True, text=True, timeout=60)


def _path_env(unit_values, name) -> list[str]:
    return [v for v in unit_values(TEMPLATES_DIR / name, 'Service', 'Environment') if v.startswith('PATH=')]


# ── the installer ───────────────────────────────────────────────────────────


def test_script_is_executable():
    assert os.access(SCRIPT, os.X_OK), f'chmod +x {SCRIPT}'


def test_install_copies_both_units_and_enables_the_timer(fake_systemctl, tmp_path):
    xdg_config = tmp_path / 'xdg-config'
    result = _run_script(fake_systemctl, tmp_path, env={'XDG_CONFIG_HOME': str(xdg_config)})
    assert result.returncode == 0, f'stdout={result.stdout!r} stderr={result.stderr!r}'

    unit_dir = xdg_config / 'systemd' / 'user'
    for name in (SERVICE_NAME, TIMER_NAME):
        assert (unit_dir / name).read_bytes() == (TEMPLATES_DIR / name).read_bytes()
    calls = fake_systemctl.calls()
    assert ['daemon-reload'] in calls, calls
    assert ['enable', '--now', TIMER_NAME] in calls, calls


def test_install_is_idempotent(fake_systemctl, tmp_path):
    env = {'XDG_CONFIG_HOME': str(tmp_path / 'xdg-config')}
    first = _run_script(fake_systemctl, tmp_path, env=env)
    second = _run_script(fake_systemctl, tmp_path, env=env, reset_state=False)
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr

    unit_dir = tmp_path / 'xdg-config' / 'systemd' / 'user'
    for name in (SERVICE_NAME, TIMER_NAME):
        assert (unit_dir / name).read_bytes() == (TEMPLATES_DIR / name).read_bytes()


def test_install_fails_loud_when_the_timer_is_absent_from_list_timers(fake_systemctl, tmp_path):
    result = _run_script(fake_systemctl, tmp_path, env={
        'XDG_CONFIG_HOME': str(tmp_path / 'xdg'),
        'FAKE_SYSTEMCTL_OMIT_LIST_TIMERS': '1',
    })
    assert result.returncode != 0, f'stdout={result.stdout!r} stderr={result.stderr!r}'
    assert TIMER_NAME in result.stderr, result.stderr


def test_install_does_not_kick_an_immediate_run(fake_systemctl, tmp_path):
    """An off-cadence first run would spend a Fable budget nobody scheduled."""
    result = _run_script(fake_systemctl, tmp_path, env={'XDG_CONFIG_HOME': str(tmp_path / 'xdg')})
    assert result.returncode == 0, result.stderr
    for call in fake_systemctl.calls():
        assert call[:1] != ['start'], f'unexpected immediate run: {call!r}'


# ── the committed units ─────────────────────────────────────────────────────


def test_timer_fires_nightly_at_0530_catching_up_a_missed_night(unit_values):
    timer = TEMPLATES_DIR / TIMER_NAME
    assert unit_values(timer, 'Timer', 'OnCalendar') == ['*-*-* 05:30:00']
    assert unit_values(timer, 'Timer', 'Persistent') == ['true']
    assert unit_values(timer, 'Timer', 'RandomizedDelaySec') == ['300']


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
            f'and update the OPERATIONS.md §12 ladder table'
        )


def test_both_units_install_into_timers_target(unit_values):
    for name in (SERVICE_NAME, TIMER_NAME):
        assert unit_values(TEMPLATES_DIR / name, 'Install', 'WantedBy') == ['timers.target'], name


def test_service_is_a_journalled_oneshot_around_the_wrapper(unit_values):
    service = TEMPLATES_DIR / SERVICE_NAME
    assert unit_values(service, 'Service', 'Type') == ['oneshot']
    assert unit_values(service, 'Service', 'ExecStart') == [f'{PRODUCTION_ROOT}/scripts/return-brief.sh']
    assert unit_values(service, 'Service', 'WorkingDirectory') == [PRODUCTION_ROOT]
    assert unit_values(service, 'Service', 'StandardOutput') == ['journal']
    assert unit_values(service, 'Service', 'StandardError') == ['journal']


def test_service_unsets_the_api_key(unit_values):
    """OPERATIONS.md §12's policy: the CLI prefers an API key over the pool's OAuth token."""
    assert unit_values(TEMPLATES_DIR / SERVICE_NAME, 'Service', 'UnsetEnvironment') == ['ANTHROPIC_API_KEY']


def test_service_path_reaches_uv_and_claude_and_matches_the_fleet_spelling(unit_values):
    """A boot-catch-up run with the user manager's minimal PATH hit ``exec: uv: not found``."""
    ours = _path_env(unit_values, SERVICE_NAME)
    assert len(ours) == 1, ours
    assert '/home/leo/.local/bin' in ours[0].removeprefix('PATH=').split(':')
    assert ours == _path_env(unit_values, PATH_TWIN)


def test_service_start_timeout_is_finite_and_outlasts_the_prepare_step(unit_values):
    """A oneshot has no start timeout by default; this one must still leave the render room to run."""
    timeout, = unit_values(TEMPLATES_DIR / SERVICE_NAME, 'Service', 'TimeoutStartSec')
    seconds = float(timeout)
    assert math.isfinite(seconds)
    assert seconds > nightly_prepare.DEFAULT_TIMEOUT_SECS


def test_service_documents_the_renderer_and_the_runbook(unit_values):
    docs = unit_values(TEMPLATES_DIR / SERVICE_NAME, 'Unit', 'Documentation')
    assert any(d.endswith('/scripts/sitting/return_brief.py') for d in docs), docs
    assert any(d.endswith('/OPERATIONS.md') for d in docs), docs


def test_service_execstart_points_at_a_real_executable_wrapper(unit_values):
    named, = unit_values(TEMPLATES_DIR / SERVICE_NAME, 'Service', 'ExecStart')
    assert named.startswith(f'{PRODUCTION_ROOT}/scripts/'), named
    here = TEMPLATES_DIR / named.split('/scripts/', 1)[1]
    assert here.is_file(), here
    assert os.access(here, os.X_OK), here
