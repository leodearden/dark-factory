"""Tests for scripts/install-quality-snapshot-timer.sh and the three units it installs (task 8464).

Drives the installer via subprocess with a FAKE ``systemctl`` first on PATH,
and parses the committed units rather than grepping their text. The harness
helpers (``_FAKE_SYSTEMCTL_SRC``, ``_fake_systemctl``, ``_systemctl_calls``,
``_run_script``, ``_directives``, ``_values``) are copies of the ones in
scripts/tests/test_install_link_heal_timer.py, under the house
duplicate-with-docstring convention; ``_run_script`` here also passes the
project root and points the installer's interpreter at this one. Real systemd
is never touched.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import quality_snapshot_alarm
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


_FAKE_SYSTEMCTL_SRC = '''#!/usr/bin/env python3
"""Fake `systemctl`, copied from scripts/tests/test_install_link_heal_timer.py.

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
            print(f"Tue 2026-10-20 02:15:00 BST 11h left n/a n/a {unit} {unit.replace('.timer', '.service')}")
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


def _run_script(tmp_path, root, *, env=None, reset_state=True):
    bin_dir, state_path = _fake_systemctl(tmp_path) if reset_state else (
        tmp_path / 'bin', tmp_path / 'systemctl_state.json')
    full_env = dict(os.environ)
    full_env['PATH'] = f'{bin_dir}{os.pathsep}{full_env["PATH"]}'
    full_env['FAKE_SYSTEMCTL_STATE'] = str(state_path)
    full_env['INSTALL_QUALITY_SNAPSHOT_TIMER_PYTHON'] = sys.executable
    full_env['XDG_CONFIG_HOME'] = str(tmp_path / 'xdg')
    if env:
        full_env.update(env)
    argv = ['bash', str(SCRIPT)] + ([] if root is None else [str(root)])
    return subprocess.run(argv, env=full_env, capture_output=True, text=True, timeout=60)


def _directives(name) -> dict[str, list[tuple[str, str]]]:
    """Parse a unit into ``{section: [(key, value), ...]}``: full-line comments skipped, duplicate keys kept."""
    path = name if isinstance(name, Path) else TEMPLATES_DIR / name
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


def _values(directives, section, key) -> list[str]:
    """Every value for ``key`` under ``[section]``; ``== ['x']`` also pins that it is declared once."""
    return [v for k, v in directives.get(section, []) if k == key]


@pytest.fixture
def project(tmp_path) -> Path:
    return Path(committed_repo(tmp_path / 'My-Proj', {})).resolve()


def _unit_dir(tmp_path) -> Path:
    return tmp_path / 'xdg' / 'systemd' / 'user'


def _env_dir(tmp_path) -> Path:
    return tmp_path / 'xdg' / 'dark-factory' / 'quality-snapshot'


def _nothing_installed(tmp_path) -> None:
    assert _systemctl_calls(tmp_path) == []
    assert not _unit_dir(tmp_path).exists() or not any(_unit_dir(tmp_path).iterdir())
    assert not _env_dir(tmp_path).exists() or not any(_env_dir(tmp_path).iterdir())


# ── the committed units ─────────────────────────────────────────────────────


def test_timer_fires_daily_at_0345_catching_up_a_missed_night():
    timer = _directives(TIMER_NAME)
    assert _values(timer, 'Timer', 'OnCalendar') == ['*-*-* 03:45:00']
    assert _values(timer, 'Timer', 'Persistent') == ['true']
    assert _values(timer, 'Timer', 'RandomizedDelaySec') == ['300']
    assert _values(timer, 'Install', 'WantedBy') == ['timers.target']


def test_timer_does_not_collide_with_an_occupied_slot():
    """Parsed on both sides, so a commented-out slot can neither manufacture nor mask a clash."""
    ours = set(_values(_directives(TIMER_NAME), 'Timer', 'OnCalendar'))
    assert ours, 'the timer declares no OnCalendar at all'
    for other in sorted(TEMPLATES_DIR.glob('*.timer')):
        if other.name == TIMER_NAME:
            continue
        clash = ours & set(_values(_directives(other), 'Timer', 'OnCalendar'))
        assert not clash, (
            f'{TIMER_NAME} shares {sorted(clash)!r} with {other.name} — pick a free slot '
            f'and update the OPERATIONS.md nightly ladder table'
        )


def test_service_is_a_low_priority_journalled_oneshot_that_alarms_on_failure():
    service = _directives(SERVICE_NAME)
    assert _values(service, 'Service', 'Type') == ['oneshot']
    assert _values(service, 'Service', 'Nice') == ['19']
    assert _values(service, 'Service', 'CPUWeight') == ['10']
    assert _values(service, 'Service', 'StandardOutput') == ['journal']
    assert _values(service, 'Service', 'StandardError') == ['journal']
    assert _values(service, 'Service', 'TimeoutStartSec') == ['2h']
    assert _values(service, 'Unit', 'OnFailure') == ['quality-snapshot-alarm@%i.service']


def test_service_requires_the_installer_written_root_file():
    """No ``-`` prefix: a missing root file fails the run, and so alarms, rather than measuring the wrong tree."""
    assert _values(_directives(SERVICE_NAME), 'Service', 'EnvironmentFile') == [ENV_FILE]


def test_service_runs_the_cadence_wrapper_with_the_factory_interpreter():
    assert _values(_directives(SERVICE_NAME), 'Service', 'ExecStart') == [
        f'{PRODUCTION_ROOT}/.venv/bin/python {PRODUCTION_ROOT}/scripts/quality_snapshot_cadence.py '
        '--root ${QUALITY_SNAPSHOT_ROOT} --project-id %i',
    ]
    assert (TEMPLATES_DIR / 'quality_snapshot_cadence.py').is_file()


def test_the_wrapper_accepts_the_units_argv(tmp_path):
    """An η flag rename fails here: argparse took both flags, so the failure is the wrapper's own."""
    (exec_start,) = _values(_directives(SERVICE_NAME), 'Service', 'ExecStart')
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


def test_the_alarm_accepts_its_units_argv(tmp_path):
    """An alarm flag rename fails here: argparse took the unit's argv, so the failure is the post's own."""
    (exec_start,) = _values(_directives(ALARM_NAME), 'Service', 'ExecStart')
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


def test_alarm_unit_is_a_journalled_oneshot_that_runs_without_the_root_file():
    alarm = _directives(ALARM_NAME)
    assert _values(alarm, 'Service', 'Type') == ['oneshot']
    assert _values(alarm, 'Service', 'EnvironmentFile') == [f'-{ENV_FILE}']
    assert _values(alarm, 'Service', 'ExecStart') == [
        f'{PRODUCTION_ROOT}/.venv/bin/python {PRODUCTION_ROOT}/scripts/quality_snapshot_alarm.py --project-id %i',
    ]
    assert _values(alarm, 'Service', 'StandardOutput') == ['journal']
    assert _values(alarm, 'Service', 'StandardError') == ['journal']
    assert 'Install' not in alarm
    handler = TEMPLATES_DIR / 'quality_snapshot_alarm.py'
    assert handler.is_file(), handler
    assert os.access(handler, os.X_OK), f'chmod +x {handler}'


# ── the installer ───────────────────────────────────────────────────────────


def test_script_is_executable():
    assert os.access(SCRIPT, os.X_OK), f'chmod +x {SCRIPT}'


def test_install_copies_the_units_writes_the_root_file_and_enables_the_instance(tmp_path, project):
    result = _run_script(tmp_path, project)
    assert result.returncode == 0, f'stdout={result.stdout!r} stderr={result.stderr!r}'

    for name in UNIT_NAMES:
        assert (_unit_dir(tmp_path) / name).read_bytes() == (TEMPLATES_DIR / name).read_bytes()
    project_id = project_id_for_root(project)
    assert project_id == 'my_proj'
    env_file = _env_dir(tmp_path) / f'{project_id}.env'
    assert env_file.read_text().splitlines() == [f'QUALITY_SNAPSHOT_ROOT={project}']
    calls = _systemctl_calls(tmp_path)
    assert calls.index(['daemon-reload']) < calls.index(
        ['enable', '--now', f'quality-snapshot@{project_id}.timer']
    ), calls


def test_the_units_read_the_root_file_the_installer_writes(tmp_path, project):
    assert _run_script(tmp_path, project).returncode == 0
    project_id = project_id_for_root(project)
    written = _env_dir(tmp_path) / f'{project_id}.env'

    def resolved(spelling: str) -> Path:
        return Path(spelling.replace('%E', str(tmp_path / 'xdg')).replace('%i', project_id))

    (service_reads,) = _values(_directives(SERVICE_NAME), 'Service', 'EnvironmentFile')
    (alarm_reads,) = _values(_directives(ALARM_NAME), 'Service', 'EnvironmentFile')
    assert resolved(service_reads) == written
    assert resolved(alarm_reads.removeprefix('-')) == written
    alarm_names = quality_snapshot_alarm.FailedRun.from_environment(
        project_id, {'XDG_CONFIG_HOME': str(tmp_path / 'xdg')},
    ).env_file
    assert alarm_names == written


def test_install_is_idempotent(tmp_path, project):
    first = _run_script(tmp_path, project)
    first_calls = _systemctl_calls(tmp_path)
    env_file = _env_dir(tmp_path) / f'{project_id_for_root(project)}.env'
    first_env = env_file.read_bytes()
    second = _run_script(tmp_path, project)
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr

    assert _systemctl_calls(tmp_path) == first_calls
    assert env_file.read_bytes() == first_env
    assert sorted(p.name for p in _env_dir(tmp_path).iterdir()) == [env_file.name]
    for name in UNIT_NAMES:
        assert (_unit_dir(tmp_path) / name).read_bytes() == (TEMPLATES_DIR / name).read_bytes()


def test_a_second_root_with_the_same_id_is_refused_until_its_env_file_is_removed(tmp_path, project):
    assert _run_script(tmp_path, project).returncode == 0
    env_file = _env_dir(tmp_path) / f'{project_id_for_root(project)}.env'
    first_env = env_file.read_bytes()
    other = Path(committed_repo(tmp_path / 'elsewhere' / 'My-Proj', {})).resolve()

    refused = _run_script(tmp_path, other)

    assert refused.returncode != 0, f'stdout={refused.stdout!r} stderr={refused.stderr!r}'
    assert str(project) in refused.stderr, refused.stderr
    assert str(other) in refused.stderr, refused.stderr
    assert env_file.read_bytes() == first_env
    assert _systemctl_calls(tmp_path) == []

    env_file.unlink()
    moved = _run_script(tmp_path, other)

    assert moved.returncode == 0, moved.stderr
    assert env_file.read_text().splitlines() == [f'QUALITY_SNAPSHOT_ROOT={other}']


def test_a_root_whose_id_the_wrapper_refuses_is_refused_before_anything_is_installed(tmp_path):
    root = Path(committed_repo(tmp_path / '_scratch', {})).resolve()

    result = _run_script(tmp_path, root)

    assert result.returncode != 0, f'stdout={result.stdout!r} stderr={result.stderr!r}'
    assert "'_scratch' is not a project id" in result.stderr, result.stderr
    _nothing_installed(tmp_path)


def test_install_fails_loud_when_the_timer_is_absent_from_list_timers(tmp_path, project):
    result = _run_script(tmp_path, project, env={'FAKE_SYSTEMCTL_OMIT_LIST_TIMERS': '1'})
    assert result.returncode != 0, f'stdout={result.stdout!r} stderr={result.stderr!r}'
    assert f'quality-snapshot@{project_id_for_root(project)}.timer' in result.stderr, result.stderr


def test_install_does_not_kick_an_immediate_run(tmp_path, project):
    """The first run belongs on schedule: an install is not an operator's snapshot run."""
    result = _run_script(tmp_path, project)
    assert result.returncode == 0, result.stderr
    for call in _systemctl_calls(tmp_path):
        assert call[:1] != ['start'], f'unexpected immediate run: {call!r}'


@pytest.mark.parametrize('kind', ['plain-directory', 'checkout-without-main'])
def test_a_root_with_no_main_commit_is_refused_before_anything_is_installed(tmp_path, kind):
    root = tmp_path / 'Not-Main'
    if kind == 'plain-directory':
        root.mkdir()
    else:
        committed_repo(root, {})
        subprocess.run(['git', '-C', str(root), 'branch', '-m', 'main', 'trunk'], check=True, timeout=30)

    result = _run_script(tmp_path, root)

    assert result.returncode != 0, f'stdout={result.stdout!r} stderr={result.stderr!r}'
    assert str(root.resolve()) in result.stderr, result.stderr
    _nothing_installed(tmp_path)


def test_a_root_an_environment_file_cannot_carry_verbatim_is_refused(tmp_path):
    root = Path(committed_repo(tmp_path / 'My Proj', {})).resolve()

    result = _run_script(tmp_path, root)

    assert result.returncode != 0, f'stdout={result.stdout!r} stderr={result.stderr!r}'
    assert str(root) in result.stderr, result.stderr
    _nothing_installed(tmp_path)


def test_no_root_argument_prints_usage(tmp_path):
    result = _run_script(tmp_path, None)

    assert result.returncode != 0
    assert 'usage' in result.stderr.lower(), result.stderr
    _nothing_installed(tmp_path)
