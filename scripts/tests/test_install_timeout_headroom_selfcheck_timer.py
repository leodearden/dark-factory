"""The timeout-headroom self-check's systemd pair and its installer (task 10940).

Under test: scripts/install-timeout-headroom-selfcheck-timer.sh, driven by
subprocess with `scripts/tests/conftest.py::fake_timer_systemctl` first on
PATH, and the committed timeout-headroom-selfcheck.{service,timer}, read
through `scripts/tests/conftest.py::unit_values` rather than grepped.  Real
systemd is never touched.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
SCRIPT = SCRIPTS_DIR / 'install-timeout-headroom-selfcheck-timer.sh'
WRAPPER = SCRIPTS_DIR / 'timeout-headroom-selfcheck.sh'

# The installed unit is a byte copy naming the PRODUCTION checkout, so this is
# never derived from the repo root, which is .worktrees/<id> in a lane.
PRODUCTION_ROOT = '/home/leo/src/dark-factory'

SERVICE_NAME = 'timeout-headroom-selfcheck.service'
TIMER_NAME = 'timeout-headroom-selfcheck.timer'
PATH_TWIN = 'exception-owner-sweep.service'


def _run_script(fake_timer_systemctl, *, env=None, reset_state=True):
    if reset_state:
        fake_timer_systemctl.reset()
    full_env = dict(os.environ)
    full_env['PATH'] = f'{fake_timer_systemctl.bin_dir}{os.pathsep}{full_env["PATH"]}'
    if env:
        full_env.update(env)
    return subprocess.run(['bash', str(SCRIPT)], env=full_env, capture_output=True, text=True, timeout=60)


def _path_env(unit_values, name) -> list[str]:
    return [v for v in unit_values(SCRIPTS_DIR / name, 'Service', 'Environment') if v.startswith('PATH=')]


def _here(production_path: str) -> Path:
    assert production_path.startswith(f'{PRODUCTION_ROOT}/'), production_path
    return SCRIPTS_DIR.parent / production_path.removeprefix(f'{PRODUCTION_ROOT}/')


# ── the installer ───────────────────────────────────────────────────────────


def test_script_is_executable():
    assert os.access(SCRIPT, os.X_OK), f'chmod +x {SCRIPT}'


def test_install_copies_both_units_and_enables_the_timer(fake_timer_systemctl, tmp_path):
    xdg_config = tmp_path / 'xdg-config'
    result = _run_script(fake_timer_systemctl, env={'XDG_CONFIG_HOME': str(xdg_config)})
    assert result.returncode == 0, f'stdout={result.stdout!r} stderr={result.stderr!r}'

    unit_dir = xdg_config / 'systemd' / 'user'
    for name in (SERVICE_NAME, TIMER_NAME):
        assert (unit_dir / name).read_bytes() == (SCRIPTS_DIR / name).read_bytes()
    calls = fake_timer_systemctl.calls()
    assert ['daemon-reload'] in calls, calls
    assert ['enable', '--now', TIMER_NAME] in calls, calls
    assert calls.index(['daemon-reload']) < calls.index(['enable', '--now', TIMER_NAME]), calls


def test_install_is_idempotent(fake_timer_systemctl, tmp_path):
    env = {'XDG_CONFIG_HOME': str(tmp_path / 'xdg-config')}
    first = _run_script(fake_timer_systemctl, env=env)
    second = _run_script(fake_timer_systemctl, env=env, reset_state=False)
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr


def test_install_fails_loud_when_the_timer_is_absent_from_list_timers(fake_timer_systemctl, tmp_path):
    result = _run_script(fake_timer_systemctl, env={
        'XDG_CONFIG_HOME': str(tmp_path / 'xdg'),
        'FAKE_SYSTEMCTL_OMIT_LIST_TIMERS': '1',
    })
    assert result.returncode != 0, f'stdout={result.stdout!r} stderr={result.stderr!r}'
    assert TIMER_NAME in result.stderr, result.stderr


def test_install_does_not_kick_an_immediate_run(fake_timer_systemctl, tmp_path):
    """Activation and the first run belong to the deployment, not to the install."""
    result = _run_script(fake_timer_systemctl, env={'XDG_CONFIG_HOME': str(tmp_path / 'xdg')})
    assert result.returncode == 0, result.stderr
    for call in fake_timer_systemctl.calls():
        assert call[:1] != ['start'], f'unexpected immediate run: {call!r}'


# ── the committed units ─────────────────────────────────────────────────────


def test_the_timer_fires_nightly_at_one_thirty_and_catches_up(unit_values):
    timer = SCRIPTS_DIR / TIMER_NAME
    assert unit_values(timer, 'Timer', 'OnCalendar') == ['*-*-* 01:30:00']
    assert unit_values(timer, 'Timer', 'Persistent') == ['true']
    assert unit_values(timer, 'Timer', 'RandomizedDelaySec') == ['300']


def test_service_is_a_oneshot_whose_execstart_is_the_wrapper(unit_values):
    service = SCRIPTS_DIR / SERVICE_NAME
    assert unit_values(service, 'Service', 'Type') == ['oneshot']
    named, = unit_values(service, 'Service', 'ExecStart')
    assert _here(named) == WRAPPER, named
    assert WRAPPER.is_file() and os.access(WRAPPER, os.X_OK), WRAPPER
    assert unit_values(service, 'Service', 'WorkingDirectory') == [PRODUCTION_ROOT]


def test_both_units_install_into_timers_target(unit_values):
    for name in (SERVICE_NAME, TIMER_NAME):
        assert unit_values(SCRIPTS_DIR / name, 'Install', 'WantedBy') == ['timers.target'], name


def test_service_path_reaches_uv_and_matches_the_fleet_spelling(unit_values):
    """The wrapper runs ``uv``; the user manager's minimal PATH does not reach it."""
    ours = _path_env(unit_values, SERVICE_NAME)
    assert len(ours) == 1, ours
    assert ours == _path_env(unit_values, PATH_TWIN)
    assert shutil.which('uv', path=ours[0].removeprefix('PATH=')) is not None, ours


def test_timer_does_not_collide_with_an_occupied_slot(unit_values):
    ours = set(unit_values(SCRIPTS_DIR / TIMER_NAME, 'Timer', 'OnCalendar'))
    assert ours, 'the timer declares no OnCalendar at all'
    for other in sorted(SCRIPTS_DIR.rglob('*.timer')):
        if other.name == TIMER_NAME:
            continue
        clash = ours & set(unit_values(other, 'Timer', 'OnCalendar'))
        assert not clash, f'{TIMER_NAME} shares {sorted(clash)!r} with {other.name}'
