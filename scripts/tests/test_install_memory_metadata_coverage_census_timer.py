"""Tests for scripts/install-memory-metadata-coverage-census-timer.sh and the two
systemd unit files it installs (task 4006). The wrapper their ExecStart names
is tested in test_memory_metadata_coverage_census_wrapper.py.

Drives the installer via subprocess with a FAKE `systemctl` shimmed onto PATH
(records every invocation, minus `--user`, into a shared JSON state file) --
mirroring test_install_reclaim_orphaned_worktrees_timer.py. Real systemd is
never touched.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

SCRIPT = (Path(__file__).parent.parent
          / 'install-memory-metadata-coverage-census-timer.sh')
TEMPLATES_DIR = Path(__file__).parent.parent
REPO_ROOT = TEMPLATES_DIR.parent

# The units name the PRODUCTION checkout absolutely, since systemd resolves
# ExecStart/WorkingDirectory absolutely and the installed unit is a byte copy
# of the committed one -- so these assertions must not be derived from
# REPO_ROOT, which is a .worktrees/<id> path when the suite runs in a lane.
PRODUCTION_ROOT = '/home/leo/src/dark-factory'

SERVICE_NAME = 'memory-metadata-coverage-census.service'
TIMER_NAME = 'memory-metadata-coverage-census.timer'


_FAKE_SYSTEMCTL_SRC = '''#!/usr/bin/env python3
"""Fake `systemctl` for testing install-memory-metadata-coverage-census-timer.sh.

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
            print(f"Mon 2026-08-17 05:00:00 UTC 8h left n/a n/a {unit} {service}")
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
        env=full_env, capture_output=True, text=True, timeout=60,
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
    """No surprise run at install time.

    Unlike the flag-marker and reclaim installers, this one never `start`s the
    service. The census is a paginated full scroll of both live collections and
    it APPENDS a row to the committed trend history; an install-time firing
    would put an unreviewed, off-cadence row into the series the nightly job
    owns. An operator who wants a look now runs the wrapper by hand, or the
    census with `--no-history`.
    """
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
    heavily by design. The demonstrated case is `Persistent=true`, which appears
    BOTH in prose at memory-metadata-coverage-census.timer:20 and as the real
    directive at :27, so deleting the directive left the substring assertion
    GREEN. That is the class of finding this closes.

    SCOPED HONESTLY (esc-4006-7). An earlier draft generalized this to "every
    literal these tests care about also appears in prose" and cited
    `census_memory_metadata.py` at memory-metadata-coverage-census.service:7 as a
    second instance. That was measured FALSE and is retracted: the filename
    occurs exactly ONCE in the service unit, at the real `Documentation=` line,
    and the comment at :7 never contains it -- so for that directive the old
    substring form did catch an outright deletion. Routing it through the parser
    anyway is a STRENGTHENING, not a fix; see
    `test_service_documents_where_the_normative_contract_lives` for the
    comment-out edit on which the two forms genuinely diverge.

    Every unit assertion in this file routes through here regardless, so the
    weaker and stronger cases are not left to be told apart by eye; no new
    raw-substring pins.

    WHY IT IS HAND-ROLLED. Duplicate keys are preserved as separate pairs, never
    collapsed -- the service legitimately carries THREE `Documentation=` lines,
    which `configparser` would collapse or reject even with `strict=False`.

    WHY NOT `systemd-analyze verify`. It IS available here (systemd 255) and was
    deliberately not used: it cannot assert that a specific directive is SET,
    and it would make this suite depend on systemd being installed in every
    container/CI runner.

    Accepts a unit filename (resolved under TEMPLATES_DIR) or a Path, so sibling
    `*.timer` units can be parsed for the collision check.
    """
    path = name if isinstance(name, Path) else TEMPLATES_DIR / name
    sections: dict[str, list[tuple[str, str]]] = {}
    current = ''
    for raw in path.read_text().splitlines():
        line = raw.strip()
        # FULL-LINE comments only: systemd treats `#`/`;` as a comment lead-in at
        # the start of a line, and a directive's value may legitimately contain
        # either character (a Documentation= URL fragment, for instance).
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
    """05:00, after 03:00 legibility-trickle, 03:30 flag-marker-sweep and the
    already-double-booked 04:00 (reclaim-orphaned-worktrees +
    legibility-transcript-check). The stagger is deliberate: these jobs all
    touch the same machine and, in several cases, the same backing stores --
    this one scrolls every point in both live Qdrant collections."""
    assert _values(_directives(TIMER_NAME), 'Timer', 'OnCalendar') == [
        '*-*-* 05:00:00']


def test_timer_does_not_collide_with_an_occupied_slot():
    """Guards the ladder itself, not just this unit's own literal: a future
    edit that re-cadences this job onto a taken slot fails here rather than
    silently double-booking a third job.

    Parsed on BOTH sides, so a commented-out slot in a sibling unit can neither
    manufacture a phantom collision nor mask a real one.
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
                f'pick a free slot and update the OPERATIONS.md §12 ladder '
                f'table')


def test_timer_catches_up_a_missed_night_and_avoids_a_thundering_herd():
    """A silently skipped night is a hole in the series that no later run can
    reconstruct -- this timer's output is a TREND. `Persistent=true` is what
    makes a night missed to a sleeping laptop get caught up on next login."""
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
        f'{PRODUCTION_ROOT}/scripts/memory-metadata-coverage-census.sh']
    assert _values(service, 'Service', 'WorkingDirectory') == [PRODUCTION_ROOT]


def test_service_pins_the_boot_catch_up_path():
    """A Persistent=true boot catch-up run gets the user manager's minimal
    PATH (journal excerpt in scripts/fused-memory-flag-marker-sweep.service).
    This line is the unit-level half of the fix; the wrapper's own uv
    resolution covers a stale installed unit that predates it."""
    environment = _values(_directives(SERVICE_NAME), 'Service', 'Environment')
    path_entries = [v for v in environment if v.startswith('PATH=')]
    assert len(path_entries) == 1, environment
    directories = path_entries[0].removeprefix('PATH=').split(':')
    # Production home, literally: the installed unit is a byte copy.
    assert '/home/leo/.local/bin' in directories, directories
    assert '/usr/bin' in directories, directories


# land_files can return up to ~190 s plus its git teardown past its ceiling
# (scripts/_merge_queue_landing.py::land_files); OPERATIONS.md §5's
# deploy-landing paragraph budgets 300.
_LANDING_OVERRUN_SECS = 300
# Everything before the landing's ceiling starts: the reclaim, the generation
# worktree, the census and the rehearsal. Whole runs took 38-64 s on
# 2026-11-10..14 (the unit's journal), so an hour is ample.
_MEASURE_BUDGET_SECS = 3600
_SECONDS_PER_DAY = 86400


def test_a_nights_landing_is_bounded_so_the_next_elapse_finds_the_unit_inactive():
    """A Type=oneshot has no start timeout by default and systemd never starts
    a second instance while one is activating, so this bound is what keeps one
    night's landing from swallowing the next night's run. The landing's
    ceiling starts only once the night has measured, so TimeoutStartSec must
    hold the measuring too, or a slow-queue night is stopped mid-wait."""
    service = _directives(SERVICE_NAME)
    ceilings = [v.removeprefix('CENSUS_LANDING_TIMEOUT_SECS=')
                for v in _values(service, 'Service', 'Environment')
                if v.startswith('CENSUS_LANDING_TIMEOUT_SECS=')]
    assert len(ceilings) == 1 and ceilings[0].isdigit(), ceilings
    (start_timeout,) = _values(service, 'Service', 'TimeoutStartSec')
    assert start_timeout.isdigit(), start_timeout
    (delay,) = _values(_directives(TIMER_NAME), 'Timer', 'RandomizedDelaySec')

    assert (_MEASURE_BUDGET_SECS + int(ceilings[0]) + _LANDING_OVERRUN_SECS
            <= int(start_timeout))
    assert int(start_timeout) + int(delay) < _SECONDS_PER_DAY


def test_service_sends_both_streams_to_the_journal():
    """The report is written to files, but the run's narration -- and any
    shortfall the census exits 1 on -- is only ever readable in the journal."""
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
    """The census script and the PRD are normative; the unit points a reader at
    them rather than restating any figure that would drift.

    Asserted as parsed `[Unit] Documentation=` VALUES rather than as a substring
    of the file text. CORRECTION (esc-4006-7): an earlier draft of this docstring
    claimed the bare filename "also appears in the unit's own comment at :7, so
    deleting the real directive left a substring check green". That was measured
    FALSE and is retracted -- `census_memory_metadata.py` occurs exactly ONCE in
    the unit, at the real `Documentation=` line, so the old substring form did
    catch an outright deletion.

    The parsed form is kept as a STRENGTHENING, not a fix, and this is what it
    buys: under the plausible comment-out edit (the directive prefixed with `#`
    rather than removed), the filename is still present in the file text, so a
    substring check stays green while the unit documents nothing. The parsed form
    fails, correctly. Do not "simplify" this back to `in text` on the grounds
    that the two forms are equivalent -- they diverge on exactly that edit.

    Both entries are checked, which additionally exercises the parser's
    duplicate-key preservation -- a dict-collapse would keep only the last of the
    three.
    """
    docs = _values(_directives(SERVICE_NAME), 'Unit', 'Documentation')
    assert docs, 'the unit points a reader at nothing'
    assert any(d.endswith('/census_memory_metadata.py') for d in docs), docs
    assert any(d.endswith('/memory-metadata-vocabulary.md') for d in docs), docs


# DELIBERATELY NOT TESTED HERE (review of 2026-08-16): that no cadence knob
# was added to `dark-factory-orchestrator.yaml`.  That test grepped another
# file's raw TEXT for four tokens, which is not a behaviour this code can
# regress on -- and worse, it opened with a bare `if not config.is_file():
# return`, so any checkout layout that lacks the file (a worktree, a
# differently-rooted run) reported it as a PASSING guard while it inspected
# nothing.  A guard that cannot distinguish "verified" from "never looked" is
# worse than no guard.  The design decision it meant to protect -- the cadence
# lives in the .timer's `OnCalendar`, not in orchestrator config -- is asserted
# directly and positively by the OnCalendar tests in this file.
