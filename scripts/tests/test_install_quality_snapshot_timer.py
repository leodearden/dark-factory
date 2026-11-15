"""Tests for scripts/install-quality-snapshot-timer.sh and the three units it installs (task 8464).

Drives the installer via subprocess with `scripts/tests/conftest.py::fake_systemctl`
first on PATH, and reads the committed units through
`scripts/tests/conftest.py::unit_values` rather than grepping their text.
``_run_script`` passes the project root and points the installer's interpreter
at this one. Real systemd is never touched.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import quality_snapshot_alarm
import systemd_unit_parity
from committed_git_repo import committed_repo
from shared.project_id import project_id_for_root
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

SCRIPT = Path(__file__).parent.parent / 'install-quality-snapshot-timer.sh'
TEMPLATES_DIR = Path(__file__).parent.parent

# The installed unit is a byte copy naming the PRODUCTION checkout, so this is
# never derived from the repo root, which is .worktrees/<id> in a lane.
PRODUCTION_ROOT = '/home/leo/src/dark-factory'

SERVICE_NAME = 'quality-snapshot@.service'
TIMER_NAME = 'quality-snapshot@.timer'
ALARM_NAME = 'quality-snapshot-alarm@.service'
UNIT_NAMES = (SERVICE_NAME, TIMER_NAME, ALARM_NAME)
ENV_FILE = '%E/dark-factory/quality-snapshot/%i.env'


def _run_script(fake_systemctl, tmp_path, root, *, env=None, reset_state=True):
    if reset_state:
        fake_systemctl.reset()
    full_env = dict(os.environ)
    full_env['PATH'] = f'{fake_systemctl.bin_dir}{os.pathsep}{full_env["PATH"]}'
    full_env['INSTALL_QUALITY_SNAPSHOT_TIMER_PYTHON'] = sys.executable
    full_env['XDG_CONFIG_HOME'] = str(tmp_path / 'xdg')
    if env:
        full_env.update(env)
    argv = ['bash', str(SCRIPT)] + ([] if root is None else [str(root)])
    return subprocess.run(argv, env=full_env, capture_output=True, text=True, timeout=60)


@pytest.fixture
def project(tmp_path) -> Path:
    return Path(committed_repo(tmp_path / 'My-Proj', {})).resolve()


def _unit_dir(tmp_path) -> Path:
    return tmp_path / 'xdg' / 'systemd' / 'user'


def _env_dir(tmp_path) -> Path:
    return tmp_path / 'xdg' / 'dark-factory' / 'quality-snapshot'


def _nothing_installed(fake_systemctl, tmp_path) -> None:
    assert fake_systemctl.calls() == []
    assert not _unit_dir(tmp_path).exists() or not any(_unit_dir(tmp_path).iterdir())
    assert not _env_dir(tmp_path).exists() or not any(_env_dir(tmp_path).iterdir())


# ── the committed units ─────────────────────────────────────────────────────


def test_timer_fires_daily_at_0345_catching_up_a_missed_night(unit_values):
    timer = TEMPLATES_DIR / TIMER_NAME
    assert unit_values(timer, 'Timer', 'OnCalendar') == ['*-*-* 03:45:00']
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


def test_service_is_a_low_priority_journalled_oneshot_that_alarms_on_failure(unit_values):
    service = TEMPLATES_DIR / SERVICE_NAME
    assert unit_values(service, 'Service', 'Type') == ['oneshot']
    assert unit_values(service, 'Service', 'Nice') == ['19']
    assert unit_values(service, 'Service', 'CPUWeight') == ['10']
    assert unit_values(service, 'Service', 'StandardOutput') == ['journal']
    assert unit_values(service, 'Service', 'StandardError') == ['journal']
    assert unit_values(service, 'Service', 'TimeoutStartSec') == ['2h']
    assert unit_values(service, 'Unit', 'OnFailure') == ['quality-snapshot-alarm@%i.service']


def test_service_requires_the_installer_written_root_file(unit_values):
    """No ``-`` prefix: a missing root file fails the run, and so alarms, rather than measuring the wrong tree."""
    assert unit_values(TEMPLATES_DIR / SERVICE_NAME, 'Service', 'EnvironmentFile') == [ENV_FILE]


def test_service_runs_the_cadence_wrapper_with_the_factory_interpreter(unit_values):
    assert unit_values(TEMPLATES_DIR / SERVICE_NAME, 'Service', 'ExecStart') == [
        f'{PRODUCTION_ROOT}/.venv/bin/python {PRODUCTION_ROOT}/scripts/quality_snapshot_cadence.py '
        '--root ${QUALITY_SNAPSHOT_ROOT} --project-id %i',
    ]
    assert (TEMPLATES_DIR / 'quality_snapshot_cadence.py').is_file()


def test_the_wrapper_accepts_the_units_argv(unit_values, tmp_path):
    """An η flag rename fails here: argparse took both flags, so the failure is the wrapper's own."""
    (exec_start,) = unit_values(TEMPLATES_DIR / SERVICE_NAME, 'Service', 'ExecStart')
    not_a_checkout = tmp_path / 'not-a-checkout'
    not_a_checkout.mkdir()
    substitutes = {
        f'{PRODUCTION_ROOT}/.venv/bin/python': sys.executable,
        f'{PRODUCTION_ROOT}/scripts/quality_snapshot_cadence.py': str(TEMPLATES_DIR / 'quality_snapshot_cadence.py'),
        '${QUALITY_SNAPSHOT_ROOT}': str(not_a_checkout),
        '%i': 'proj_a',
    }
    argv = [substitutes.get(token, token) for token in shlex.split(exec_start)] + ['--python', sys.executable]

    result = subprocess.run(
        argv, capture_output=True, text=True, timeout=120,
        env={**os.environ, 'XDG_STATE_HOME': str(tmp_path / 'state')},
    )

    assert result.returncode == 2, f'stdout={result.stdout!r} stderr={result.stderr!r}'
    assert 'quality_snapshot_cadence:' in result.stderr, result.stderr
    assert 'usage:' not in result.stderr, result.stderr


def test_the_alarm_accepts_its_units_argv(unit_values, tmp_path):
    """An alarm flag rename fails here: argparse took the unit's argv, so the failure is the post's own."""
    (exec_start,) = unit_values(TEMPLATES_DIR / ALARM_NAME, 'Service', 'ExecStart')
    handler = str(TEMPLATES_DIR / 'quality_snapshot_alarm.py')
    substitutes = {
        f'{PRODUCTION_ROOT}/.venv/bin/python': sys.executable,
        f'{PRODUCTION_ROOT}/scripts/quality_snapshot_alarm.py': handler,
        '%i': 'proj_a',
    }
    argv = [substitutes.get(token, token) for token in shlex.split(exec_start)]
    assert argv[:2] == [sys.executable, handler], argv
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    journalctl = bin_dir / 'journalctl'
    journalctl.write_text('#!/bin/sh\necho "fake journal line"\n')
    journalctl.chmod(0o755)

    result = subprocess.run(
        [*argv, '--escalation-url', 'http://127.0.0.1:1/mcp'], capture_output=True, text=True, timeout=120,
        env={**os.environ, 'PATH': f'{bin_dir}{os.pathsep}{os.environ["PATH"]}'},
    )

    assert result.returncode == 2, f'stdout={result.stdout!r} stderr={result.stderr!r}'
    assert 'quality_snapshot_alarm:' in result.stderr, result.stderr
    assert 'usage:' not in result.stderr, result.stderr


def test_alarm_unit_is_a_journalled_oneshot_that_runs_without_the_root_file(unit_values):
    alarm = TEMPLATES_DIR / ALARM_NAME
    assert unit_values(alarm, 'Service', 'Type') == ['oneshot']
    assert unit_values(alarm, 'Service', 'EnvironmentFile') == [f'-{ENV_FILE}']
    assert unit_values(alarm, 'Service', 'ExecStart') == [
        f'{PRODUCTION_ROOT}/.venv/bin/python {PRODUCTION_ROOT}/scripts/quality_snapshot_alarm.py --project-id %i',
    ]
    assert unit_values(alarm, 'Service', 'StandardOutput') == ['journal']
    assert unit_values(alarm, 'Service', 'StandardError') == ['journal']
    assert 'Install' not in systemd_unit_parity.parse_unit_directives(alarm.read_text(encoding='utf-8'))
    handler = TEMPLATES_DIR / 'quality_snapshot_alarm.py'
    assert handler.is_file(), handler
    assert os.access(handler, os.X_OK), f'chmod +x {handler}'


# ── the installer ───────────────────────────────────────────────────────────


def test_script_is_executable():
    assert os.access(SCRIPT, os.X_OK), f'chmod +x {SCRIPT}'


def test_install_copies_the_units_writes_the_root_file_and_enables_the_instance(fake_systemctl, tmp_path, project):
    result = _run_script(fake_systemctl, tmp_path, project)
    assert result.returncode == 0, f'stdout={result.stdout!r} stderr={result.stderr!r}'

    for name in UNIT_NAMES:
        assert (_unit_dir(tmp_path) / name).read_bytes() == (TEMPLATES_DIR / name).read_bytes()
    project_id = project_id_for_root(project)
    assert project_id == 'my_proj'
    env_file = _env_dir(tmp_path) / f'{project_id}.env'
    assert env_file.read_text().splitlines() == [f'QUALITY_SNAPSHOT_ROOT={project}']
    calls = fake_systemctl.calls()
    assert calls.index(['daemon-reload']) < calls.index(
        ['enable', '--now', f'quality-snapshot@{project_id}.timer']
    ), calls


def test_the_units_read_the_root_file_the_installer_writes(unit_values, fake_systemctl, tmp_path, project):
    assert _run_script(fake_systemctl, tmp_path, project).returncode == 0
    project_id = project_id_for_root(project)
    written = _env_dir(tmp_path) / f'{project_id}.env'

    def resolved(spelling: str) -> Path:
        return Path(spelling.replace('%E', str(tmp_path / 'xdg')).replace('%i', project_id))

    (service_reads,) = unit_values(TEMPLATES_DIR / SERVICE_NAME, 'Service', 'EnvironmentFile')
    (alarm_reads,) = unit_values(TEMPLATES_DIR / ALARM_NAME, 'Service', 'EnvironmentFile')
    assert resolved(service_reads) == written
    assert resolved(alarm_reads.removeprefix('-')) == written
    alarm_names = quality_snapshot_alarm.FailedRun.from_environment(
        project_id, {'XDG_CONFIG_HOME': str(tmp_path / 'xdg')},
    ).env_file
    assert alarm_names == written


def test_install_is_idempotent(fake_systemctl, tmp_path, project):
    first = _run_script(fake_systemctl, tmp_path, project)
    first_calls = fake_systemctl.calls()
    env_file = _env_dir(tmp_path) / f'{project_id_for_root(project)}.env'
    first_env = env_file.read_bytes()
    second = _run_script(fake_systemctl, tmp_path, project)
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr

    assert fake_systemctl.calls() == first_calls
    assert env_file.read_bytes() == first_env
    assert sorted(p.name for p in _env_dir(tmp_path).iterdir()) == [env_file.name]
    for name in UNIT_NAMES:
        assert (_unit_dir(tmp_path) / name).read_bytes() == (TEMPLATES_DIR / name).read_bytes()


def test_a_second_root_with_the_same_id_is_refused_until_its_env_file_is_removed(fake_systemctl, tmp_path, project):
    assert _run_script(fake_systemctl, tmp_path, project).returncode == 0
    env_file = _env_dir(tmp_path) / f'{project_id_for_root(project)}.env'
    first_env = env_file.read_bytes()
    other = Path(committed_repo(tmp_path / 'elsewhere' / 'My-Proj', {})).resolve()

    refused = _run_script(fake_systemctl, tmp_path, other)

    assert refused.returncode != 0, f'stdout={refused.stdout!r} stderr={refused.stderr!r}'
    assert str(project) in refused.stderr, refused.stderr
    assert str(other) in refused.stderr, refused.stderr
    assert env_file.read_bytes() == first_env
    assert fake_systemctl.calls() == []

    env_file.unlink()
    moved = _run_script(fake_systemctl, tmp_path, other)

    assert moved.returncode == 0, moved.stderr
    assert env_file.read_text().splitlines() == [f'QUALITY_SNAPSHOT_ROOT={other}']


def test_a_root_whose_id_the_wrapper_refuses_is_refused_before_anything_is_installed(fake_systemctl, tmp_path):
    root = Path(committed_repo(tmp_path / '_scratch', {})).resolve()

    result = _run_script(fake_systemctl, tmp_path, root)

    assert result.returncode != 0, f'stdout={result.stdout!r} stderr={result.stderr!r}'
    assert "'_scratch' is not a project id" in result.stderr, result.stderr
    _nothing_installed(fake_systemctl, tmp_path)


def test_install_fails_loud_when_the_timer_is_absent_from_list_timers(fake_systemctl, tmp_path, project):
    result = _run_script(fake_systemctl, tmp_path, project, env={'FAKE_SYSTEMCTL_OMIT_LIST_TIMERS': '1'})
    assert result.returncode != 0, f'stdout={result.stdout!r} stderr={result.stderr!r}'
    assert f'quality-snapshot@{project_id_for_root(project)}.timer' in result.stderr, result.stderr


def test_install_does_not_kick_an_immediate_run(fake_systemctl, tmp_path, project):
    """The first run belongs on schedule: an install is not an operator's snapshot run."""
    result = _run_script(fake_systemctl, tmp_path, project)
    assert result.returncode == 0, result.stderr
    for call in fake_systemctl.calls():
        assert call[:1] != ['start'], f'unexpected immediate run: {call!r}'


@pytest.mark.parametrize('kind', ['plain-directory', 'checkout-without-main'])
def test_a_root_with_no_main_commit_is_refused_before_anything_is_installed(fake_systemctl, tmp_path, kind):
    root = tmp_path / 'Not-Main'
    if kind == 'plain-directory':
        root.mkdir()
    else:
        committed_repo(root, {})
        subprocess.run(['git', '-C', str(root), 'branch', '-m', 'main', 'trunk'], check=True, timeout=30)

    result = _run_script(fake_systemctl, tmp_path, root)

    assert result.returncode != 0, f'stdout={result.stdout!r} stderr={result.stderr!r}'
    assert str(root.resolve()) in result.stderr, result.stderr
    _nothing_installed(fake_systemctl, tmp_path)


def test_a_root_an_environment_file_cannot_carry_verbatim_is_refused(fake_systemctl, tmp_path):
    root = Path(committed_repo(tmp_path / 'My Proj', {})).resolve()

    result = _run_script(fake_systemctl, tmp_path, root)

    assert result.returncode != 0, f'stdout={result.stdout!r} stderr={result.stderr!r}'
    assert str(root) in result.stderr, result.stderr
    _nothing_installed(fake_systemctl, tmp_path)


def test_no_root_argument_prints_usage(fake_systemctl, tmp_path):
    result = _run_script(fake_systemctl, tmp_path, None)

    assert result.returncode != 0
    assert 'usage' in result.stderr.lower(), result.stderr
    _nothing_installed(fake_systemctl, tmp_path)
