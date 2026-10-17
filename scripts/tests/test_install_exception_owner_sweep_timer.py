"""Tests for the INV-12 owner sweep's systemd pair, its wrapper and its installer (task 5608).

Under test: scripts/install-exception-owner-sweep-timer.sh, driven by
subprocess with a FAKE ``systemctl`` first on PATH; the committed
exception-owner-sweep.{service,timer}, parsed rather than grepped and held to
referential integrity, not to configuration literals (INV-10); and
scripts/exception-owner-sweep.sh, with a fake sweep recorder at its
EXCEPTION_OWNER_SWEEP_CMD seam.  The harness helpers are copies of the ones in
scripts/tests/test_install_return_brief_timer.py, under the house
duplicate-with-docstring convention; their rationale lives there.  Real
systemd is never touched, except ``systemd-analyze calendar``, which only
evaluates an expression.
"""
from __future__ import annotations

import datetime
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import exception_register

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
SCRIPT = SCRIPTS_DIR / 'install-exception-owner-sweep-timer.sh'
WRAPPER = SCRIPTS_DIR / 'exception-owner-sweep.sh'

# The installed unit is a byte copy naming the PRODUCTION checkout, so this is
# never derived from the repo root, which is .worktrees/<id> in a lane.
PRODUCTION_ROOT = '/home/leo/src/dark-factory'

SERVICE_NAME = 'exception-owner-sweep.service'
TIMER_NAME = 'exception-owner-sweep.timer'
PATH_TWIN = 'fused-memory-flag-marker-sweep.service'


_FAKE_SYSTEMCTL_SRC = '''#!/usr/bin/env python3
"""Fake `systemctl`, copied from scripts/tests/test_install_return_brief_timer.py.

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
            print(f"Sat 2026-10-17 06:00:00 BST 11h left n/a n/a {unit} {unit.replace('.timer', '.service')}")
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


def _directives(name) -> dict[str, list[tuple[str, str]]]:
    """Parse a unit into ``{section: [(key, value), ...]}``: full-line comments skipped, duplicate keys kept."""
    path = name if isinstance(name, Path) else SCRIPTS_DIR / name
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


def _path_env(name) -> list[str]:
    return [v for v in _values(_directives(name), 'Service', 'Environment') if v.startswith('PATH=')]


def _here(production_path: str) -> Path:
    """*production_path*, a path under the production checkout, mapped onto this checkout."""
    assert production_path.startswith(f'{PRODUCTION_ROOT}/'), production_path
    return SCRIPTS_DIR.parent / production_path.removeprefix(f'{PRODUCTION_ROOT}/')


# ── the installer ───────────────────────────────────────────────────────────


def test_script_is_executable():
    assert os.access(SCRIPT, os.X_OK), f'chmod +x {SCRIPT}'


def test_install_copies_both_units_and_enables_the_timer(tmp_path):
    xdg_config = tmp_path / 'xdg-config'
    result = _run_script(tmp_path, env={'XDG_CONFIG_HOME': str(xdg_config)})
    assert result.returncode == 0, f'stdout={result.stdout!r} stderr={result.stderr!r}'

    unit_dir = xdg_config / 'systemd' / 'user'
    for name in (SERVICE_NAME, TIMER_NAME):
        assert (unit_dir / name).read_bytes() == (SCRIPTS_DIR / name).read_bytes()
    calls = _systemctl_calls(tmp_path)
    assert ['daemon-reload'] in calls, calls
    assert ['enable', '--now', TIMER_NAME] in calls, calls
    assert calls.index(['daemon-reload']) < calls.index(['enable', '--now', TIMER_NAME]), calls


def test_install_is_idempotent(tmp_path):
    env = {'XDG_CONFIG_HOME': str(tmp_path / 'xdg-config')}
    first = _run_script(tmp_path, env=env)
    second = _run_script(tmp_path, env=env, reset_state=False)
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr

    unit_dir = tmp_path / 'xdg-config' / 'systemd' / 'user'
    for name in (SERVICE_NAME, TIMER_NAME):
        assert (unit_dir / name).read_bytes() == (SCRIPTS_DIR / name).read_bytes()


def test_install_fails_loud_when_the_timer_is_absent_from_list_timers(tmp_path):
    result = _run_script(tmp_path, env={
        'XDG_CONFIG_HOME': str(tmp_path / 'xdg'),
        'FAKE_SYSTEMCTL_OMIT_LIST_TIMERS': '1',
    })
    assert result.returncode != 0, f'stdout={result.stdout!r} stderr={result.stderr!r}'
    assert TIMER_NAME in result.stderr, result.stderr


def test_install_does_not_kick_an_immediate_run(tmp_path):
    """Activation and the first run belong to the deployment task, not to the install."""
    result = _run_script(tmp_path, env={'XDG_CONFIG_HOME': str(tmp_path / 'xdg')})
    assert result.returncode == 0, result.stderr
    for call in _systemctl_calls(tmp_path):
        assert call[:1] != ['start'], f'unexpected immediate run: {call!r}'


# ── the committed units ─────────────────────────────────────────────────────


def test_service_is_a_oneshot_whose_execstart_is_a_real_wrapper_in_its_working_directory():
    service = _directives(SERVICE_NAME)
    assert _values(service, 'Service', 'Type') == ['oneshot']
    named, = _values(service, 'Service', 'ExecStart')
    wrapper = _here(named)
    assert wrapper == WRAPPER, named
    assert wrapper.is_file(), wrapper
    assert os.access(wrapper, os.X_OK), wrapper
    assert _values(service, 'Service', 'WorkingDirectory') == [PRODUCTION_ROOT]


def test_both_units_install_into_timers_target():
    for name in (SERVICE_NAME, TIMER_NAME):
        assert _values(_directives(name), 'Install', 'WantedBy') == ['timers.target'], name


def test_service_path_reaches_uv_and_matches_the_fleet_spelling():
    """The wrapper runs ``uv``; the user manager's minimal PATH does not reach it."""
    ours = _path_env(SERVICE_NAME)
    assert len(ours) == 1, ours
    assert ours == _path_env(PATH_TWIN)
    uv = shutil.which('uv', path=ours[0].removeprefix('PATH='))
    assert uv is not None, ours


_ELAPSE = re.compile(r'^\s*(?:Next elapse|Iteration #\d+): \w+ (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\b', re.M)


def test_the_timer_fires_once_per_sweep_period():
    """``--report`` calls a stamp stale after two SWEEP_PERIODs, so the timer must fire once per one.

    Consecutive local elapses are differenced as wall-clock times, so a
    daylight-saving change between them does not read as a longer period.
    """
    analyze = shutil.which('systemd-analyze')
    assert analyze is not None, 'systemd-analyze is needed to evaluate OnCalendar'
    calendar, = _values(_directives(TIMER_NAME), 'Timer', 'OnCalendar')
    completed = subprocess.run(
        [analyze, 'calendar', '--iterations=3', calendar], capture_output=True, text=True, timeout=30, check=False
    )
    assert completed.returncode == 0, completed.stderr
    elapses = [datetime.datetime.fromisoformat(stamp) for stamp in _ELAPSE.findall(completed.stdout)]
    assert len(elapses) == 3, completed.stdout
    periods = {later - earlier for earlier, later in zip(elapses, elapses[1:], strict=False)}
    assert periods == {exception_register.SWEEP_PERIOD}, completed.stdout


def test_timer_does_not_collide_with_an_occupied_slot():
    """Parsed on both sides, so a commented-out slot can neither manufacture nor mask a clash."""
    ours = set(_values(_directives(TIMER_NAME), 'Timer', 'OnCalendar'))
    assert ours, 'the timer declares no OnCalendar at all'
    for other in sorted(SCRIPTS_DIR.rglob('*.timer')):
        if other.name == TIMER_NAME:
            continue
        clash = ours & set(_values(_directives(other), 'Timer', 'OnCalendar'))
        assert not clash, (
            f'{TIMER_NAME} shares {sorted(clash)!r} with {other.name} — pick a free slot '
            f'and update the OPERATIONS.md §12 ladder table'
        )


# ── the wrapper ─────────────────────────────────────────────────────────────

_RECORDER_SRC = f'''#!{sys.executable}
"""Fake sweep: records its argv and the git redirection it inherited, exits $FAKE_SWEEP_RC."""
import json
import os
import sys

with open(os.environ["FAKE_SWEEP_RECORD"], "w") as f:
    json.dump({{"argv": sys.argv[1:], "env": {{k: os.environ.get(k) for k in ("GIT_DIR", "GIT_WORK_TREE")}}}}, f)
sys.exit(int(os.environ.get("FAKE_SWEEP_RC", "0")))
'''


def _run_wrapper(tmp_path: Path, **env: str) -> tuple[subprocess.CompletedProcess[str], dict]:
    repo = tmp_path / 'repo'
    repo.mkdir(exist_ok=True)
    recorder = tmp_path / 'fake-sweep'
    recorder.write_text(_RECORDER_SRC)
    recorder.chmod(0o755)
    record = tmp_path / 'sweep-record.json'
    full_env = {
        **os.environ,
        'REPO': str(repo),
        'EXCEPTION_OWNER_SWEEP_CMD': str(recorder),
        'FAKE_SWEEP_RECORD': str(record),
        **env,
    }
    completed = subprocess.run(['bash', str(WRAPPER)], env=full_env, capture_output=True, text=True, timeout=60)
    return completed, json.loads(record.read_text())


def test_the_wrapper_runs_the_sweep_on_its_repo(tmp_path):
    completed, recorded = _run_wrapper(tmp_path)

    assert completed.returncode == 0, completed.stderr
    repo = str(tmp_path / 'repo')
    assert recorded['argv'] == [f'{repo}/scripts/exception_owner_sweep.py', '--project-root', repo]


def test_a_failed_sweep_is_narrated_and_still_exits_zero(tmp_path):
    completed, _ = _run_wrapper(tmp_path, FAKE_SWEEP_RC='2')

    assert completed.returncode == 0, completed.stderr
    assert 'sweep=2' in completed.stdout + completed.stderr


def test_the_wrapper_drops_an_inherited_git_redirection(tmp_path):
    _, recorded = _run_wrapper(tmp_path, GIT_DIR=str(tmp_path / 'elsewhere.git'), GIT_WORK_TREE=str(tmp_path))

    assert recorded['env'] == {'GIT_DIR': None, 'GIT_WORK_TREE': None}
