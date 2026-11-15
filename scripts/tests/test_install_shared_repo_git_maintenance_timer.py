"""Tests for scripts/install-shared-repo-git-maintenance-timer.sh, the two units it
installs, and the wrapper the service runs (task 8076).

Drives the installer via subprocess with `scripts/tests/conftest.py::fake_systemctl`
first on PATH, and reads the committed units through
`scripts/tests/conftest.py::unit_values` rather than grepping their text. Real
systemd is never touched. The wrapper is driven end to end against a real tmp repo.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from committed_git_repo import committed_repo
from git_repo_probe import loose_count
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

SCRIPT = Path(__file__).parent.parent / 'install-shared-repo-git-maintenance-timer.sh'
TEMPLATES_DIR = Path(__file__).parent.parent
WRAPPER = TEMPLATES_DIR / 'shared-repo-git-maintenance.sh'
JOB = TEMPLATES_DIR / 'shared_repo_git_maintenance.py'
CHECKOUT_ROOT = Path(__file__).parents[2]

# The installed unit is a byte copy naming the PRODUCTION checkout, so this is
# never derived from the repo root, which is .worktrees/<id> in a lane.
PRODUCTION_ROOT = '/home/leo/src/dark-factory'

SERVICE_NAME = 'shared-repo-git-maintenance.service'
TIMER_NAME = 'shared-repo-git-maintenance.timer'


def _run_script(fake_systemctl, tmp_path, *, env=None, reset_state=True):
    if reset_state:
        fake_systemctl.reset()
    full_env = dict(os.environ)
    full_env['PATH'] = f'{fake_systemctl.bin_dir}{os.pathsep}{full_env["PATH"]}'
    if env:
        full_env.update(env)
    return subprocess.run(['bash', str(SCRIPT)], env=full_env, capture_output=True, text=True, timeout=60)


# ── the installer ───────────────────────────────────────────────────────────


def test_script_is_executable():
    assert os.access(SCRIPT, os.X_OK), f'chmod +x {SCRIPT}'


def test_install_copies_both_units_enables_the_timer_and_kicks_a_non_blocking_run(fake_systemctl, tmp_path):
    xdg_config = tmp_path / 'xdg-config'
    result = _run_script(fake_systemctl, tmp_path, env={'XDG_CONFIG_HOME': str(xdg_config)})
    assert result.returncode == 0, f'stdout={result.stdout!r} stderr={result.stderr!r}'

    unit_dir = xdg_config / 'systemd' / 'user'
    for name in (SERVICE_NAME, TIMER_NAME):
        assert (unit_dir / name).read_bytes() == (TEMPLATES_DIR / name).read_bytes()
    calls = fake_systemctl.calls()
    assert (
        calls.index(['daemon-reload'])
        < calls.index(['enable', '--now', TIMER_NAME])
        < calls.index(['start', '--no-block', SERVICE_NAME])
    ), calls


def test_install_is_idempotent(fake_systemctl, tmp_path):
    env = {'XDG_CONFIG_HOME': str(tmp_path / 'xdg-config')}
    first = _run_script(fake_systemctl, tmp_path, env=env)
    first_calls = fake_systemctl.calls()
    second = _run_script(fake_systemctl, tmp_path, env=env)
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr

    assert fake_systemctl.calls() == first_calls
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


# ── the committed units ─────────────────────────────────────────────────────


def test_timer_fires_daily_at_0200_catching_up_a_missed_day(unit_values):
    """02:00 is the slot the already-filed deploy task 8087 checks list-timers for."""
    timer = TEMPLATES_DIR / TIMER_NAME
    assert unit_values(timer, 'Timer', 'OnCalendar') == ['*-*-* 02:00:00']
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
            f'and update the OPERATIONS.md section 12 nightly ladder table'
        )


def test_service_is_a_journalled_oneshot_around_the_wrapper(unit_values):
    service = TEMPLATES_DIR / SERVICE_NAME
    assert unit_values(service, 'Service', 'Type') == ['oneshot']
    assert unit_values(service, 'Service', 'ExecStart') == [
        f'{PRODUCTION_ROOT}/scripts/shared-repo-git-maintenance.sh',
    ]
    assert unit_values(service, 'Service', 'WorkingDirectory') == [PRODUCTION_ROOT]
    assert unit_values(service, 'Service', 'StandardOutput') == ['journal']
    assert unit_values(service, 'Service', 'StandardError') == ['journal']


def test_service_documents_the_safety_envelope(unit_values):
    named, = unit_values(TEMPLATES_DIR / SERVICE_NAME, 'Unit', 'Documentation')
    prefix = f'file://{PRODUCTION_ROOT}/'
    assert named.startswith(prefix), named
    assert named.endswith('/docs/shared-repo-git-maintenance.md'), named
    here = CHECKOUT_ROOT / named.removeprefix(prefix)
    assert here.is_file(), here


def test_service_execstart_points_at_a_real_executable_wrapper(unit_values):
    named, = unit_values(TEMPLATES_DIR / SERVICE_NAME, 'Service', 'ExecStart')
    assert named.startswith(f'{PRODUCTION_ROOT}/scripts/'), named
    here = TEMPLATES_DIR / named.split('/scripts/', 1)[1]
    assert here.is_file(), here
    assert os.access(here, os.X_OK), here


# ── the wrapper ─────────────────────────────────────────────────────────────


def _hermetic_args(tmp_path: Path, repo: Path) -> list[str]:
    units = tmp_path / 'no-units'
    units.mkdir(exist_ok=True)
    return ['--project-root', str(repo), '--units-dir', str(units), '--lock-file', str(tmp_path / 'l')]


def _run_wrapper(tmp_path: Path, repo: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ['bash', str(WRAPPER), *_hermetic_args(tmp_path, repo)],
        env={**os.environ, 'REPO': str(CHECKOUT_ROOT)},
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_wrapper_is_executable():
    assert os.access(WRAPPER, os.X_OK), f'chmod +x {WRAPPER}'


def test_wrapper_maintains_a_real_repo_and_narrates_the_exit_code(tmp_path):
    repo = Path(committed_repo(tmp_path / 'repo', {'README.md': 'seed\n'}))
    assert loose_count(repo) > 0

    result = _run_wrapper(tmp_path, repo)

    assert result.returncode == 0, result.stderr
    assert loose_count(repo) == 0
    last = result.stdout.splitlines()[-1]
    assert 'maintenance=0' in last, result.stdout


def test_wrapper_exits_zero_even_when_a_repo_fails(tmp_path):
    if os.geteuid() == 0:
        pytest.skip('root ignores the read-only pack dir that makes repack fail')
    repo = Path(committed_repo(tmp_path / 'repo', {'README.md': 'seed\n'}))
    pack_dir = repo / '.git' / 'objects' / 'pack'
    pack_dir.chmod(0o555)
    try:
        result = _run_wrapper(tmp_path, repo)
    finally:
        pack_dir.chmod(0o755)

    assert result.returncode == 0, result.stderr
    assert 'maintenance=1' in result.stdout.splitlines()[-1], result.stdout


def test_the_job_runs_on_the_stdlib_alone(tmp_path):
    """The unit runs the host's bare ``python3`` with no uv, venv or workspace, so a
    first-party import in the job or its leaves would fail only in production.
    ``-S -E`` drops site-packages and PYTHONPATH, leaving the stdlib and scripts/."""
    repo = Path(committed_repo(tmp_path / 'repo', {'README.md': 'seed\n'}))

    result = subprocess.run(
        [sys.executable, '-S', '-E', str(JOB), *_hermetic_args(tmp_path, repo)],
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert result.returncode == 0, result.stderr
    [record] = [json.loads(line) for line in result.stdout.splitlines()]
    assert record['outcome'] == 'maintained', record
