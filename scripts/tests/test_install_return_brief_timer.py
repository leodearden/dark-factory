"""Tests for scripts/install-return-brief-timer.sh and the two systemd units it installs (task 5376).

Drives the installer via subprocess with a FAKE ``systemctl`` first on PATH,
and parses the committed units rather than grepping their text. The harness
helpers are copies of the ones in
scripts/tests/test_install_memory_metadata_coverage_census_timer.py, under the
house duplicate-with-docstring convention; their rationale lives there. Real
systemd is never touched.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

from sitting import nightly_prepare

SCRIPT = Path(__file__).parent.parent / 'install-return-brief-timer.sh'
TEMPLATES_DIR = Path(__file__).parent.parent

# The installed unit is a byte copy naming the PRODUCTION checkout, so this is
# never derived from the repo root, which is .worktrees/<id> in a lane.
PRODUCTION_ROOT = '/home/leo/src/dark-factory'

SERVICE_NAME = 'return-brief.service'
TIMER_NAME = 'return-brief.timer'
PATH_TWIN = 'fused-memory-flag-marker-sweep.service'


_FAKE_SYSTEMCTL_SRC = '''#!/usr/bin/env python3
"""Fake `systemctl`, copied from
scripts/tests/test_install_memory_metadata_coverage_census_timer.py.

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
            print(f"Sun 2026-09-27 05:30:00 BST 11h left n/a n/a {unit} {unit.replace('.timer', '.service')}")
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


def _path_env(name) -> list[str]:
    return [v for v in _values(_directives(name), 'Service', 'Environment') if v.startswith('PATH=')]


# ── the installer ───────────────────────────────────────────────────────────


def test_script_is_executable():
    assert os.access(SCRIPT, os.X_OK), f'chmod +x {SCRIPT}'


def test_install_copies_both_units_and_enables_the_timer(tmp_path):
    xdg_config = tmp_path / 'xdg-config'
    result = _run_script(tmp_path, env={'XDG_CONFIG_HOME': str(xdg_config)})
    assert result.returncode == 0, f'stdout={result.stdout!r} stderr={result.stderr!r}'

    unit_dir = xdg_config / 'systemd' / 'user'
    for name in (SERVICE_NAME, TIMER_NAME):
        assert (unit_dir / name).read_bytes() == (TEMPLATES_DIR / name).read_bytes()
    calls = _systemctl_calls(tmp_path)
    assert ['daemon-reload'] in calls, calls
    assert ['enable', '--now', TIMER_NAME] in calls, calls


def test_install_is_idempotent(tmp_path):
    env = {'XDG_CONFIG_HOME': str(tmp_path / 'xdg-config')}
    first = _run_script(tmp_path, env=env)
    second = _run_script(tmp_path, env=env, reset_state=False)
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr

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


def test_install_does_not_kick_an_immediate_run(tmp_path):
    """An off-cadence first run would spend a Fable budget nobody scheduled."""
    result = _run_script(tmp_path, env={'XDG_CONFIG_HOME': str(tmp_path / 'xdg')})
    assert result.returncode == 0, result.stderr
    for call in _systemctl_calls(tmp_path):
        assert call[:1] != ['start'], f'unexpected immediate run: {call!r}'


# ── the committed units ─────────────────────────────────────────────────────


def test_timer_fires_nightly_at_0530_catching_up_a_missed_night():
    timer = _directives(TIMER_NAME)
    assert _values(timer, 'Timer', 'OnCalendar') == ['*-*-* 05:30:00']
    assert _values(timer, 'Timer', 'Persistent') == ['true']
    assert _values(timer, 'Timer', 'RandomizedDelaySec') == ['300']


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
            f'and update the OPERATIONS.md §12 ladder table'
        )


def test_both_units_install_into_timers_target():
    for name in (SERVICE_NAME, TIMER_NAME):
        assert _values(_directives(name), 'Install', 'WantedBy') == ['timers.target'], name


def test_service_is_a_journalled_oneshot_around_the_wrapper():
    service = _directives(SERVICE_NAME)
    assert _values(service, 'Service', 'Type') == ['oneshot']
    assert _values(service, 'Service', 'ExecStart') == [f'{PRODUCTION_ROOT}/scripts/return-brief.sh']
    assert _values(service, 'Service', 'WorkingDirectory') == [PRODUCTION_ROOT]
    assert _values(service, 'Service', 'StandardOutput') == ['journal']
    assert _values(service, 'Service', 'StandardError') == ['journal']


def test_service_unsets_the_api_key():
    """OPERATIONS.md §12's policy: the CLI prefers an API key over the pool's OAuth token."""
    assert _values(_directives(SERVICE_NAME), 'Service', 'UnsetEnvironment') == ['ANTHROPIC_API_KEY']


def test_service_path_reaches_uv_and_claude_and_matches_the_fleet_spelling():
    """A boot-catch-up run with the user manager's minimal PATH hit ``exec: uv: not found``."""
    ours = _path_env(SERVICE_NAME)
    assert len(ours) == 1, ours
    assert '/home/leo/.local/bin' in ours[0].removeprefix('PATH=').split(':')
    assert ours == _path_env(PATH_TWIN)


def test_service_start_timeout_is_finite_and_outlasts_the_prepare_step():
    """A oneshot has no start timeout by default; this one must still leave the render room to run."""
    timeout, = _values(_directives(SERVICE_NAME), 'Service', 'TimeoutStartSec')
    seconds = float(timeout)
    assert math.isfinite(seconds)
    assert seconds > nightly_prepare.DEFAULT_TIMEOUT_SECS


def test_service_documents_the_renderer_and_the_runbook():
    docs = _values(_directives(SERVICE_NAME), 'Unit', 'Documentation')
    assert any(d.endswith('/scripts/sitting/return_brief.py') for d in docs), docs
    assert any(d.endswith('/OPERATIONS.md') for d in docs), docs


def test_service_execstart_points_at_a_real_executable_wrapper():
    named, = _values(_directives(SERVICE_NAME), 'Service', 'ExecStart')
    assert named.startswith(f'{PRODUCTION_ROOT}/scripts/'), named
    here = TEMPLATES_DIR / named.split('/scripts/', 1)[1]
    assert here.is_file(), here
    assert os.access(here, os.X_OK), here
