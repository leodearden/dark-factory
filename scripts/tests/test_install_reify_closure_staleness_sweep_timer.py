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

# The units name the PRODUCTION checkout absolutely, since systemd resolves
# ExecStart/WorkingDirectory absolutely and the installed unit is a byte copy
# of the committed one -- so these assertions must not be derived from
# REPO_ROOT, which is a .worktrees/<id> path when the suite runs in a lane.
PRODUCTION_ROOT = '/home/leo/src/dark-factory'

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


def _directives(name) -> dict[str, list[tuple[str, str]]]:
    """Parse a systemd unit into `{section: [(key, value), ...]}`.

    WHY THIS EXISTS. A raw-substring check on a unit's text cannot tell a live
    DIRECTIVE from the COMMENT that explains it -- and these units comment
    heavily by design. The demonstrated case is `Persistent=true`, which
    appears BOTH inside the `[Timer]` comment block explaining the stagger
    ladder AND as the real directive, in `reify-closure-staleness-sweep.timer`,
    so deleting the directive left the substring assertion here GREEN while
    the safeguard it names (catching up a night missed to a sleeping/offline
    laptop) was gone. That is the class of finding this closes (task 4305,
    ported from the `_directives` helper added for the sibling census timer's
    tests, task 4006).

    Every unit assertion in this file routes through here regardless of
    whether a given literal is currently shadowed, so the weaker and stronger
    cases are not left to be told apart by eye; no new raw-substring pins.

    WHY IT IS HAND-ROLLED. Duplicate keys are preserved as separate pairs,
    never collapsed -- the service legitimately carries TWO `Documentation=`
    lines, which `configparser` would collapse or reject even with
    `strict=False`.

    WHY NOT `systemd-analyze verify`. It cannot assert that a specific
    directive is SET, and it would make this suite depend on systemd being
    installed in every container/CI runner.

    Accepts a unit filename (resolved under TEMPLATES_DIR) or a Path.
    """
    path = name if isinstance(name, Path) else TEMPLATES_DIR / name
    sections: dict[str, list[tuple[str, str]]] = {}
    current = ''
    for raw in path.read_text().splitlines():
        line = raw.strip()
        # FULL-LINE comments only: systemd treats `#`/`;` as a comment lead-in
        # at the start of a line, and a directive's value may legitimately
        # contain either character (a Documentation= URL fragment, here).
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
    """Every value declared for `key` under `[section]`, in file order.

    A list, not a scalar: asserting `== ['x']` pins both the value AND that it
    is declared exactly once, so a second stray `OnCalendar=` (which systemd
    reads as an ADDITIONAL firing) cannot slip in unnoticed.
    """
    return [v for k, v in directives.get(section, []) if k == key]


def test_timer_fires_at_the_next_free_nightly_slot():
    """04:30, after 03:00 legibility-trickle, 03:30 flag-marker-sweep and the
    already-double-booked 04:00 (reclaim-orphaned-worktrees +
    legibility-transcript-check). The stagger is deliberate: these jobs all
    touch the same machine and, in two cases, the same stores."""
    assert _values(_directives(TIMER_NAME), 'Timer', 'OnCalendar') == [
        '*-*-* 04:30:00']


def test_timer_does_not_collide_with_an_occupied_slot():
    """Guards the ladder itself, not just this unit's own literal: a future
    edit that re-cadences this job onto a taken slot fails here rather than
    silently double-booking a third job.

    Parsed on BOTH sides, so a commented-out slot in a sibling unit can
    neither manufacture a phantom collision nor mask a real one. Ported from
    the sibling census timer's test of the same name (task 4006) -- none of
    the logic is specific to this unit's name.
    """
    ours = set(_values(_directives(TIMER_NAME), 'Timer', 'OnCalendar'))
    assert ours, 'the timer declares no OnCalendar at all'
    for other in sorted(TEMPLATES_DIR.glob('*.timer')):
        if other.name == TIMER_NAME:
            continue
        clash = ours & set(_values(_directives(other), 'Timer', 'OnCalendar'))
        if clash:
            raise AssertionError(
                f'{TIMER_NAME} shares {sorted(clash)!r} with {other.name} — '
                f'pick a free slot and update the nightly ladder table in '
                f'OPERATIONS.md')


def test_timer_catches_up_a_missed_night_and_avoids_a_thundering_herd():
    """A silently skipped night leaves stranded reify rows stranded for
    another day -- `Persistent=true` is what makes a night missed to a
    sleeping laptop get caught up on next boot or login."""
    timer = _directives(TIMER_NAME)
    assert _values(timer, 'Timer', 'Persistent') == ['true']
    assert _values(timer, 'Timer', 'RandomizedDelaySec') == ['300']


def test_timer_is_installed_into_timers_target():
    assert _values(_directives(TIMER_NAME), 'Install', 'WantedBy') == [
        'timers.target']


def test_service_is_a_thin_oneshot_around_the_committed_wrapper():
    """Paths are the PRODUCTION checkout's, not this test run's.

    systemd resolves ExecStart absolutely and the installed unit is a byte copy
    of the committed one, so the unit must name /home/leo/src/dark-factory even
    when these tests run from a worktree under .worktrees/.
    """
    service = _directives(SERVICE_NAME)
    assert _values(service, 'Service', 'Type') == ['oneshot']
    assert _values(service, 'Service', 'ExecStart') == [
        f'{PRODUCTION_ROOT}/scripts/reify-closure-staleness-sweep.sh']
    assert _values(service, 'Service', 'WorkingDirectory') == [PRODUCTION_ROOT]


def test_service_sends_both_streams_to_the_journal():
    """The wrapper this unit runs "always exits 0" (see the [Service] comment
    above) so a failed sweep never surfaces as systemd `failed` state -- the
    journal is therefore the ONLY place a failure is ever readable. Ported
    from the sibling census timer's test of the same name (task 4006); this
    unit needs the guard more, not less, for exactly the reason above."""
    service = _directives(SERVICE_NAME)
    assert _values(service, 'Service', 'StandardOutput') == ['journal']
    assert _values(service, 'Service', 'StandardError') == ['journal']


def test_service_execstart_points_at_a_real_executable_wrapper():
    """The wrapper named by ExecStart exists and is executable.

    Checked against THIS checkout (the production path's scripts/ tail mapped
    onto the tree under test) so the assertion is meaningful from a worktree
    too -- what it pins is that the unit does not name a wrapper that was
    renamed or never committed.
    """
    named, = _values(_directives(SERVICE_NAME), 'Service', 'ExecStart')
    assert named.startswith(f'{PRODUCTION_ROOT}/scripts/'), named
    here = TEMPLATES_DIR / named.split('/scripts/', 1)[1]
    assert here.is_file(), here
    assert os.access(here, os.X_OK), here


def test_service_documents_where_the_normative_contract_lives():
    """The sweep script itself is normative, so the unit points a reader at it
    rather than at any dark-factory-side paraphrase."""
    docs = _values(_directives(SERVICE_NAME), 'Unit', 'Documentation')
    assert docs, 'the unit points a reader at nothing'
    assert any(d.endswith('/deterministic-gate-closure-staleness-sweep.sh')
               for d in docs), docs


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
