"""Tests for the transcript secret sweep's systemd pair, its wrapper and its installer (task 6820).

Under test: scripts/install-transcript-secret-sweep-timer.sh, driven by
subprocess with `scripts/tests/conftest.py::fake_timer_systemctl` first on PATH; the
committed transcript-secret-sweep.{service,timer}, read through
`scripts/tests/conftest.py::unit_values` rather than grepped; and
scripts/transcript-secret-sweep.sh, with a fake sweep recorder at its
TRANSCRIPT_SECRET_SWEEP_CMD seam.  Real systemd is never touched.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
SCRIPT = SCRIPTS_DIR / 'install-transcript-secret-sweep-timer.sh'
WRAPPER = SCRIPTS_DIR / 'transcript-secret-sweep.sh'

# The installed unit is a byte copy naming the PRODUCTION checkout, so this is
# never derived from the repo root, which is .worktrees/<id> in a lane.
PRODUCTION_ROOT = '/home/leo/src/dark-factory'

SERVICE_NAME = 'transcript-secret-sweep.service'
TIMER_NAME = 'transcript-secret-sweep.timer'
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
    """*production_path*, a path under the production checkout, mapped onto this checkout."""
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

    unit_dir = tmp_path / 'xdg-config' / 'systemd' / 'user'
    for name in (SERVICE_NAME, TIMER_NAME):
        assert (unit_dir / name).read_bytes() == (SCRIPTS_DIR / name).read_bytes()


def test_install_fails_loud_when_the_timer_is_absent_from_list_timers(fake_timer_systemctl, tmp_path):
    result = _run_script(fake_timer_systemctl, env={
        'XDG_CONFIG_HOME': str(tmp_path / 'xdg'),
        'FAKE_SYSTEMCTL_OMIT_LIST_TIMERS': '1',
    })
    assert result.returncode != 0, f'stdout={result.stdout!r} stderr={result.stderr!r}'
    assert TIMER_NAME in result.stderr, result.stderr


def test_install_does_not_kick_an_immediate_run(fake_timer_systemctl, tmp_path):
    """Activation and the first run belong to the deployment task, not to the install."""
    result = _run_script(fake_timer_systemctl, env={'XDG_CONFIG_HOME': str(tmp_path / 'xdg')})
    assert result.returncode == 0, result.stderr
    for call in fake_timer_systemctl.calls():
        assert call[:1] != ['start'], f'unexpected immediate run: {call!r}'


# ── the committed units ─────────────────────────────────────────────────────


def test_service_is_a_oneshot_whose_execstart_is_a_real_wrapper_in_its_working_directory(unit_values):
    service = SCRIPTS_DIR / SERVICE_NAME
    assert unit_values(service, 'Service', 'Type') == ['oneshot']
    named, = unit_values(service, 'Service', 'ExecStart')
    wrapper = _here(named)
    assert wrapper == WRAPPER, named
    assert wrapper.is_file(), wrapper
    assert os.access(wrapper, os.X_OK), wrapper
    assert unit_values(service, 'Service', 'WorkingDirectory') == [PRODUCTION_ROOT]


def test_both_units_install_into_timers_target(unit_values):
    for name in (SERVICE_NAME, TIMER_NAME):
        assert unit_values(SCRIPTS_DIR / name, 'Install', 'WantedBy') == ['timers.target'], name


def test_service_path_reaches_uv_and_matches_the_fleet_spelling(unit_values):
    """The wrapper runs ``uv``; the user manager's minimal PATH does not reach it."""
    ours = _path_env(unit_values, SERVICE_NAME)
    assert len(ours) == 1, ours
    assert ours == _path_env(unit_values, PATH_TWIN)
    uv = shutil.which('uv', path=ours[0].removeprefix('PATH='))
    assert uv is not None, ours


def test_the_timer_fires_nightly_at_its_ladder_slot_and_catches_up(unit_values):
    timer = SCRIPTS_DIR / TIMER_NAME
    assert unit_values(timer, 'Timer', 'OnCalendar') == ['*-*-* 02:30:00']
    assert unit_values(timer, 'Timer', 'Persistent') == ['true']


def test_timer_does_not_collide_with_an_occupied_slot(unit_values):
    """Parsed on both sides, so a commented-out slot can neither manufacture nor mask a clash."""
    ours = set(unit_values(SCRIPTS_DIR / TIMER_NAME, 'Timer', 'OnCalendar'))
    assert ours, 'the timer declares no OnCalendar at all'
    for other in sorted(SCRIPTS_DIR.rglob('*.timer')):
        if other.name == TIMER_NAME:
            continue
        clash = ours & set(unit_values(other, 'Timer', 'OnCalendar'))
        assert not clash, (
            f'{TIMER_NAME} shares {sorted(clash)!r} with {other.name} — pick a free slot '
            f'and update the OPERATIONS.md §12 ladder table'
        )


# ── the wrapper ─────────────────────────────────────────────────────────────

_FAKE_POOL_VAR = 'CLAUDE_OAUTH_TOKEN_Z'

_RECORDER_SRC = f'''#!{sys.executable}
"""Fake sweep: records its argv and WHETHER it saw the .env pool var (never its value), exits $FAKE_SWEEP_RC."""
import json
import os
import sys

with open(os.environ["FAKE_SWEEP_RECORD"], "w") as f:
    json.dump({{"argv": sys.argv[1:], "saw_pool_var": bool(os.environ.get("{_FAKE_POOL_VAR}"))}}, f)
sys.exit(int(os.environ.get("FAKE_SWEEP_RC", "0")))
'''


def _run_wrapper(tmp_path: Path, **env: str) -> tuple[subprocess.CompletedProcess[str], dict]:
    repo = tmp_path / 'repo'
    repo.mkdir(exist_ok=True)
    (repo / '.env').write_text(f'{_FAKE_POOL_VAR}=fake-wrapper-token-' + 'w' * 30 + '\n')
    recorder = tmp_path / 'fake-sweep'
    recorder.write_text(_RECORDER_SRC)
    recorder.chmod(0o755)
    record = tmp_path / 'sweep-record.json'
    full_env = {
        **{k: v for k, v in os.environ.items() if k != _FAKE_POOL_VAR},
        'REPO': str(repo),
        'TRANSCRIPT_SECRET_SWEEP_CMD': str(recorder),
        'FAKE_SWEEP_RECORD': str(record),
        **env,
    }
    completed = subprocess.run(['bash', str(WRAPPER)], env=full_env, capture_output=True, text=True, timeout=60)
    return completed, json.loads(record.read_text())


def test_the_wrapper_runs_the_sweep_from_its_repo(tmp_path):
    completed, recorded = _run_wrapper(tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert recorded['argv'] == [f'{tmp_path / "repo"}/scripts/transcript_secret_sweep.py']


def test_the_wrapper_sources_the_repo_env_so_the_sweep_sees_live_values(tmp_path):
    completed, recorded = _run_wrapper(tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert recorded['saw_pool_var'] is True
    assert 'fake-wrapper-token' not in completed.stdout + completed.stderr


def test_a_failed_sweep_is_narrated_and_still_exits_zero(tmp_path):
    completed, _ = _run_wrapper(tmp_path, FAKE_SWEEP_RC='2')

    assert completed.returncode == 0, completed.stderr
    assert 'sweep=2' in completed.stdout + completed.stderr
