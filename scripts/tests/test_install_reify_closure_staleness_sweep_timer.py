"""Tests for scripts/install-reify-closure-staleness-sweep-timer.sh and the
two systemd unit files it installs (task 3102).

Drives the installer via subprocess with a FAKE `systemctl` shimmed onto PATH
(records every invocation, minus `--user`, into a shared JSON state file) --
mirroring test_install_flag_marker_sweep_timer.py. Real systemd is never
touched. Like the flag-marker installer and unlike the trickle one, this job
is a single non-templated instance, so there is no project_id argument and no
config-resolution seam to stub.

Unlike BOTH of those installers, this one deliberately does NOT kick an
immediate run: the first execution against the live reify task store should be
an operator-run `--dry-run` (see OPERATIONS.md), not a surprise mutation at
install time.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / 'install-reify-closure-staleness-sweep-timer.sh'
TEMPLATES_DIR = Path(__file__).parent.parent
REPO_ROOT = TEMPLATES_DIR.parent

SERVICE_NAME = 'reify-closure-staleness-sweep.service'
TIMER_NAME = 'reify-closure-staleness-sweep.timer'


_FAKE_SYSTEMCTL_SRC = '''#!/usr/bin/env python3
"""Fake `systemctl` for testing install-reify-closure-staleness-sweep-timer.sh.

Records every invocation (minus `--user`) into a JSON state file at
$FAKE_SYSTEMCTL_STATE. `enable --now <unit>` marks each non-flag arg as an
enabled unit; `start <unit>` is recorded and always succeeds; `list-timers`
echoes one line per *.timer enabled so far THIS RUN, unless
FAKE_SYSTEMCTL_OMIT_LIST_TIMERS=1 -- simulating the self-verify failure where
`enable` nominally succeeded but the unit is absent from `list-timers`.
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

    if verb == "daemon-reload":
        _save(state)
        return 0

    if verb == "enable":
        units = [a for a in rest if not a.startswith("-")]
        enabled = state.setdefault("enabled_timers", [])
        for u in units:
            if u not in enabled:
                enabled.append(u)
        _save(state)
        return 0

    if verb == "start":
        _save(state)
        return 0

    if verb == "list-timers":
        _save(state)
        if os.environ.get("FAKE_SYSTEMCTL_OMIT_LIST_TIMERS") == "1":
            print("0 timers listed.")
            return 0
        enabled = state.get("enabled_timers", [])
        for unit in enabled:
            service = unit.replace(".timer", ".service")
            print(f"Mon 2026-07-31 04:30:00 UTC 8h left n/a n/a {unit} {service}")
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
    return subprocess.run(
        ['bash', str(SCRIPT)],
        env=full_env, capture_output=True, text=True, timeout=30,
    )


# ── the installer ───────────────────────────────────────────────────────────


def test_script_is_executable():
    assert os.access(SCRIPT, os.X_OK), (
        f'Expected {SCRIPT} to be executable (os.X_OK); run: chmod +x {SCRIPT}')


def test_install_copies_both_units_and_enables_the_timer(tmp_path):
    xdg_config = tmp_path / 'xdg-config'
    result = _run_script(tmp_path, env={'XDG_CONFIG_HOME': str(xdg_config)})
    assert result.returncode == 0, (
        f'stdout={result.stdout!r} stderr={result.stderr!r}')

    unit_dir = xdg_config / 'systemd' / 'user'
    for name in (SERVICE_NAME, TIMER_NAME):
        installed = unit_dir / name
        assert installed.is_file(), f'Expected {installed} to exist after install'
        assert installed.read_bytes() == (TEMPLATES_DIR / name).read_bytes()

    calls = _systemctl_calls(tmp_path)
    assert ['daemon-reload'] in calls, calls
    assert ['enable', '--now', TIMER_NAME] in calls, calls


def test_install_does_not_kick_an_immediate_run(tmp_path):
    """The first run against the live reify store should be an operator's
    `--dry-run`, not a surprise mutation at install time -- so unlike the
    flag-marker and reclaim installers, this one never `start`s the service."""
    result = _run_script(tmp_path, env={'XDG_CONFIG_HOME': str(tmp_path / 'xdg')})
    assert result.returncode == 0, result.stderr
    for call in _systemctl_calls(tmp_path):
        assert call[:1] != ['start'], f'unexpected immediate run: {call!r}'


def test_install_fails_loud_when_the_timer_is_absent_from_list_timers(tmp_path):
    """The self-verify catches "enable nominally succeeded but the unit is
    absent" -- the case a bare `enable` exit code cannot distinguish."""
    result = _run_script(tmp_path, env={
        'XDG_CONFIG_HOME': str(tmp_path / 'xdg'),
        'FAKE_SYSTEMCTL_OMIT_LIST_TIMERS': '1',
    })
    assert result.returncode != 0, (
        f'Expected non-zero on self-verify failure; stdout={result.stdout!r} '
        f'stderr={result.stderr!r}')
    assert TIMER_NAME in result.stderr, result.stderr


def test_install_is_idempotent(tmp_path):
    xdg_config = tmp_path / 'xdg-config'
    env = {'XDG_CONFIG_HOME': str(xdg_config)}
    first = _run_script(tmp_path, env=env)
    second = _run_script(tmp_path, env=env, reset_state=False)
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr

    unit_dir = xdg_config / 'systemd' / 'user'
    for name in (SERVICE_NAME, TIMER_NAME):
        assert (unit_dir / name).read_bytes() == (TEMPLATES_DIR / name).read_bytes()


# ── the committed unit files ────────────────────────────────────────────────


def _unit(name):
    return (TEMPLATES_DIR / name).read_text()


def test_timer_fires_at_the_next_free_nightly_slot():
    """04:30, after 03:00 legibility-trickle, 03:30 flag-marker-sweep and the
    already-double-booked 04:00 (reclaim-orphaned-worktrees +
    legibility-transcript-check). The stagger is deliberate: these jobs all
    touch the same machine and, in two cases, the same stores."""
    assert 'OnCalendar=*-*-* 04:30:00' in _unit(TIMER_NAME)


def test_timer_catches_up_a_missed_night_and_avoids_a_thundering_herd():
    text = _unit(TIMER_NAME)
    assert 'Persistent=true' in text
    assert 'RandomizedDelaySec=300' in text


def test_timer_is_installed_into_timers_target():
    assert 'WantedBy=timers.target' in _unit(TIMER_NAME)


def test_service_is_a_thin_oneshot_around_the_committed_wrapper():
    text = _unit(SERVICE_NAME)
    assert 'Type=oneshot' in text
    assert f'ExecStart={REPO_ROOT}/scripts/reify-closure-staleness-sweep.sh' in text
    assert f'WorkingDirectory={REPO_ROOT}' in text


def test_service_execstart_points_at_a_real_executable_wrapper():
    text = _unit(SERVICE_NAME)
    exec_line = next(ln for ln in text.splitlines() if ln.startswith('ExecStart='))
    wrapper = exec_line.split('=', 1)[1]
    assert os.path.isfile(wrapper), wrapper
    assert os.access(wrapper, os.X_OK), wrapper


def test_service_documents_where_the_normative_contract_lives():
    """The sweep script itself is normative, so the unit points a reader at it
    rather than at any dark-factory-side paraphrase."""
    text = _unit(SERVICE_NAME)
    assert 'Documentation=' in text
    assert 'deterministic-gate-closure-staleness-sweep.sh' in text


def test_no_cadence_knob_was_added_to_the_orchestrator_config():
    """Guards the central design decision: the cadence is the .timer's
    OnCalendar, NOT a dark-factory-orchestrator.yaml key. A knob there would be
    misplaced (the orchestrator process is not what is being scheduled) and
    would cost a cross-repo commit plus a fleet redeploy to change."""
    config = REPO_ROOT / 'dark-factory-orchestrator.yaml'
    if not config.is_file():
        return
    text = config.read_text()
    for token in ('closure_staleness', 'closure-staleness', 'redispatch_requests',
                  'redispatch-requests'):
        assert token not in text, (
            f'{token!r} found in {config} — the cadence belongs in the .timer '
            f'unit, not in orchestrator config')
