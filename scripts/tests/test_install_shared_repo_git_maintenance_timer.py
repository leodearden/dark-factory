"""Tests for scripts/install-shared-repo-git-maintenance-timer.sh, the two units it
installs, and the wrapper the service runs (task 8076).

Drives the installer via subprocess with a FAKE ``systemctl`` first on PATH, and
parses the committed units with ``systemd_unit_parity.parse_unit_directives``
rather than grepping their text. The fake-systemctl harness is a copy of the one
in scripts/tests/test_install_memory_eval_timer.py, under the house
duplicate-with-docstring convention; its rationale lives in
scripts/tests/test_install_memory_metadata_coverage_census_timer.py. Real systemd
is never touched. The wrapper is driven end to end against a real tmp repo.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import systemd_unit_parity
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


_FAKE_SYSTEMCTL_SRC = '''#!/usr/bin/env python3
"""Fake `systemctl`, copied from scripts/tests/test_install_memory_eval_timer.py.

Records every invocation (minus `--user`) into $FAKE_SYSTEMCTL_STATE.
`enable --now <unit>` marks each non-flag arg enabled; `start` is recorded;
`list-timers` echoes one line per timer enabled so far, unless
FAKE_SYSTEMCTL_OMIT_LIST_TIMERS=1 simulates an enable that left no timer.
"""
import json
import os
import sys

STATE_PATH = os.environ["FAKE_SYSTEMCTL_STATE"]


def _load():
    with open(STATE_PATH) as f:
        return json.load(f)


def _save(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f)


def main(argv):
    args = [a for a in argv[1:] if a != "--user"]
    if not args:
        return 1
    verb, rest = args[0], args[1:]

    state = _load()
    state.setdefault("calls", []).append(args)

    if verb in ("daemon-reload", "start"):
        _save(state)
        return 0

    if verb == "enable":
        enabled = state.setdefault("enabled_timers", [])
        for unit in (a for a in rest if not a.startswith("-")):
            if unit not in enabled:
                enabled.append(unit)
        _save(state)
        return 0

    if verb == "list-timers":
        _save(state)
        if os.environ.get("FAKE_SYSTEMCTL_OMIT_LIST_TIMERS") == "1":
            print("0 timers listed.")
            return 0
        enabled = state.get("enabled_timers", [])
        for unit in enabled:
            print(f"Tue 2026-10-20 03:15:00 BST 11h left n/a n/a {unit} {unit.replace('.timer', '.service')}")
        print(f"{len(enabled)} timers listed.")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
'''


def _fake_systemctl(tmp_path):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir(exist_ok=True)
    fake = bin_dir / 'systemctl'
    fake.write_text(_FAKE_SYSTEMCTL_SRC)
    fake.chmod(0o755)
    state_path = tmp_path / 'systemctl_state.json'
    state_path.write_text(json.dumps({'calls': [], 'enabled_timers': []}))
    return bin_dir, state_path


def _systemctl_calls(tmp_path):
    state_path = tmp_path / 'systemctl_state.json'
    if not state_path.is_file():
        return []
    return json.loads(state_path.read_text())['calls']


def _run_script(tmp_path, *, env=None, reset_state=True):
    bin_dir, state_path = _fake_systemctl(tmp_path) if reset_state else (
        tmp_path / 'bin', tmp_path / 'systemctl_state.json')
    full_env = dict(os.environ)
    full_env['PATH'] = f'{bin_dir}{os.pathsep}{full_env["PATH"]}'
    full_env['FAKE_SYSTEMCTL_STATE'] = str(state_path)
    if env:
        full_env.update(env)
    return subprocess.run(['bash', str(SCRIPT)], env=full_env, capture_output=True, text=True, timeout=60)


def _unit(path: Path) -> dict[str, dict[str, list[str]]]:
    return systemd_unit_parity.parse_unit_directives(path.read_text(encoding='utf-8'))


def _values(path: Path, section: str, key: str) -> list[str]:
    """Every value for ``key`` under ``[section]``; ``== ['x']`` also pins that it is declared once."""
    return _unit(path).get(section, {}).get(key, [])


# ── the installer ───────────────────────────────────────────────────────────


def test_script_is_executable():
    assert os.access(SCRIPT, os.X_OK), f'chmod +x {SCRIPT}'


def test_install_copies_both_units_enables_the_timer_and_kicks_a_non_blocking_run(tmp_path):
    xdg_config = tmp_path / 'xdg-config'
    result = _run_script(tmp_path, env={'XDG_CONFIG_HOME': str(xdg_config)})
    assert result.returncode == 0, f'stdout={result.stdout!r} stderr={result.stderr!r}'

    unit_dir = xdg_config / 'systemd' / 'user'
    for name in (SERVICE_NAME, TIMER_NAME):
        assert (unit_dir / name).read_bytes() == (TEMPLATES_DIR / name).read_bytes()
    calls = _systemctl_calls(tmp_path)
    assert (
        calls.index(['daemon-reload'])
        < calls.index(['enable', '--now', TIMER_NAME])
        < calls.index(['start', '--no-block', SERVICE_NAME])
    ), calls


def test_install_is_idempotent(tmp_path):
    env = {'XDG_CONFIG_HOME': str(tmp_path / 'xdg-config')}
    first = _run_script(tmp_path, env=env)
    first_calls = _systemctl_calls(tmp_path)
    second = _run_script(tmp_path, env=env)
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr

    assert _systemctl_calls(tmp_path) == first_calls
    unit_dir = tmp_path / 'xdg-config' / 'systemd' / 'user'
    for name in (SERVICE_NAME, TIMER_NAME):
        assert (unit_dir / name).read_bytes() == (TEMPLATES_DIR / name).read_bytes()


def test_install_fails_loud_when_the_timer_is_absent_from_list_timers(tmp_path):
    result = _run_script(tmp_path, env={
        'XDG_CONFIG_HOME': str(tmp_path / 'xdg'),
        'FAKE_SYSTEMCTL_OMIT_LIST_TIMERS': '1',
    })
    assert result.returncode != 0, f'stdout={result.stdout!r} stderr={result.stderr!r}'
    assert TIMER_NAME in result.stderr, result.stderr


# ── the committed units ─────────────────────────────────────────────────────


def test_timer_fires_daily_at_0200_catching_up_a_missed_day():
    """02:00 is the slot the already-filed deploy task 8087 checks list-timers for."""
    timer = TEMPLATES_DIR / TIMER_NAME
    assert _values(timer, 'Timer', 'OnCalendar') == ['*-*-* 02:00:00']
    assert _values(timer, 'Timer', 'Persistent') == ['true']
    assert _values(timer, 'Timer', 'RandomizedDelaySec') == ['300']
    assert _values(timer, 'Install', 'WantedBy') == ['timers.target']


def test_timer_does_not_collide_with_an_occupied_slot():
    """Parsed on both sides, so a commented-out slot can neither manufacture nor mask a clash."""
    ours = set(_values(TEMPLATES_DIR / TIMER_NAME, 'Timer', 'OnCalendar'))
    assert ours, 'the timer declares no OnCalendar at all'
    for other in sorted(TEMPLATES_DIR.glob('*.timer')):
        if other.name == TIMER_NAME:
            continue
        clash = ours & set(_values(other, 'Timer', 'OnCalendar'))
        assert not clash, (
            f'{TIMER_NAME} shares {sorted(clash)!r} with {other.name} — pick a free slot '
            f'and update the OPERATIONS.md section 12 nightly ladder table'
        )


def test_service_is_a_journalled_oneshot_around_the_wrapper():
    service = TEMPLATES_DIR / SERVICE_NAME
    assert _values(service, 'Service', 'Type') == ['oneshot']
    assert _values(service, 'Service', 'ExecStart') == [
        f'{PRODUCTION_ROOT}/scripts/shared-repo-git-maintenance.sh',
    ]
    assert _values(service, 'Service', 'WorkingDirectory') == [PRODUCTION_ROOT]
    assert _values(service, 'Service', 'StandardOutput') == ['journal']
    assert _values(service, 'Service', 'StandardError') == ['journal']


def test_service_documents_the_safety_envelope():
    named, = _values(TEMPLATES_DIR / SERVICE_NAME, 'Unit', 'Documentation')
    prefix = f'file://{PRODUCTION_ROOT}/'
    assert named.startswith(prefix), named
    assert named.endswith('/docs/shared-repo-git-maintenance.md'), named
    here = CHECKOUT_ROOT / named.removeprefix(prefix)
    assert here.is_file(), here


def test_service_execstart_points_at_a_real_executable_wrapper():
    named, = _values(TEMPLATES_DIR / SERVICE_NAME, 'Service', 'ExecStart')
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
