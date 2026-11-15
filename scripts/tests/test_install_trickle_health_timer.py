"""Tests for scripts/legibility/install-trickle-health-timer.sh, and for the two
systemd unit templates it installs (task 4514, GAP 2).

WHY THIS FILE IS THE REGRESSION TEST FOR GAP 2. The defect was not that either
trickle probe was wrong — `check_trickle_progress.py` and
`check_trickle_liveness.sh` both worked. It was that NOTHING RAN THEM; the full
account is OPERATIONS.md §"Legibility trickle health probe (04:30)". So the
assertions that a unit EXISTS and that its `ExecStart` NAMES
`check_trickle_health.py` are not boilerplate — they are the thing that keeps
the probes bound.

Template-content invariants live here rather than in a separate file, per
`test_install_trickle_timer.py::test_service_template_pins_claude_bin`'s
precedent; the installer-behaviour half is driven via subprocess with
`scripts/tests/conftest.py::fake_timer_systemctl` first on PATH, and unit directives
are read through `scripts/tests/conftest.py::unit_values` and
`scripts/tests/conftest.py::unit_environment`. Real systemd is never touched.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from legibility import trickle_state
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

SCRIPT = (Path(__file__).parent.parent
          / 'legibility' / 'install-trickle-health-timer.sh')
TEMPLATES_DIR = Path(__file__).parent.parent
SERVICE_NAME = 'legibility-trickle-health@.service'
TIMER_NAME = 'legibility-trickle-health@.timer'
SERVICE = TEMPLATES_DIR / SERVICE_NAME
TIMER = TEMPLATES_DIR / TIMER_NAME

# The unit templates name the PRODUCTION checkout, not this test run's
# worktree: systemd resolves these absolutely and the installed unit is a byte
# copy of the committed one, so they must say /home/leo/src/dark-factory even
# when these tests run from .worktrees/<id>.
PRODUCTION_ROOT = '/home/leo/src/dark-factory'

# The nightly trickle's own units, asserted against as a PAIR with ours in two
# places: the state-root absence (below) is only a real guarantee if it holds
# for the WRITER as well as this reader, and the cadence relation is meaningless
# read off one file alone.
TRICKLE_SERVICE_NAME = 'legibility-trickle@.service'
TRICKLE_TIMER_NAME = 'legibility-trickle@.timer'


def _live_directive_text(name) -> str:
    """Every non-comment line of a unit, joined.

    The right granularity for an ABSENCE assertion. Not the raw file, because
    the comments are allowed — and want — to name the variable they explain.
    Not the parsed `Environment=` dict either, because the shape being excluded
    includes an inline `ExecStart=/usr/bin/env VAR=... ` assignment, which no
    `Environment=` parser would ever see (the 2026-09-14 account-pin drop-in
    used exactly that shape).
    """
    path = name if isinstance(name, Path) else TEMPLATES_DIR / name
    return '\n'.join(
        line for line in path.read_text().splitlines()
        if not line.lstrip().startswith(('#', ';'))
    )


def _on_calendar_time(unit_values, timer_name) -> str:
    """The `HH:MM:SS` of a timer declaring exactly one `*-*-* HH:MM:SS` slot."""
    slots = unit_values(TEMPLATES_DIR / timer_name, 'Timer', 'OnCalendar')
    assert len(slots) == 1, f'{timer_name} declares OnCalendar={slots!r}'
    return slots[0].split()[-1]


def test_both_unit_templates_exist():
    """THE GAP-2 regression test, in its bluntest form.

    The finding was literally "no systemd unit, cron entry or config binds
    either probe". A committed unit naming the probe is what closes it, and a
    test that the unit exists is what keeps it closed — a future tidy-up that
    deletes either file must fail here rather than quietly returning the
    pipeline to the unobserved state it spent 2026-07-16..29 in.
    """
    assert SERVICE.is_file(), f'Expected the health probe service at {SERVICE}'
    assert TIMER.is_file(), f'Expected the health probe timer at {TIMER}'


def test_service_execstart_runs_the_health_probe_for_the_instance(unit_values):
    """`ExecStart` must name THIS probe and take its project from `%i`.

    systemd expands `%i` at instantiation, so the template is copied verbatim
    and never per-project customised — one committed file serves dark_factory,
    reify and every project after them.
    """
    exec_starts = unit_values(TEMPLATES_DIR / SERVICE_NAME, 'Service', 'ExecStart')
    assert len(exec_starts) == 1, f'ExecStart={exec_starts!r}'
    command = exec_starts[0]

    assert 'scripts/legibility/check_trickle_health.py' in command, (
        f'The health unit must run check_trickle_health.py — that binding IS '
        f'the fix for GAP 2; got ExecStart={command!r}'
    )
    assert '--project-id %i' in command, (
        f'The probe must take its project from the %i instance name, not a '
        f'baked-in literal; got ExecStart={command!r}'
    )
    # A copy-paste from either sibling unit that forgot to swap the script is
    # the realistic way this lands wrong, and it would leave every assertion
    # about Type/WorkingDirectory/journal green while running the wrong job.
    assert 'nightly.py' not in command, (
        f'This unit must not run the nightly pipeline: a second nightly run at '
        f'04:30 would re-digest the night; got ExecStart={command!r}'
    )
    assert 'check_transcript_persistence.py' not in command, (
        f'That is the transcript-check unit\'s job, not this one\'s; got '
        f'ExecStart={command!r}'
    )


def test_service_execstart_interpreter_is_an_absolute_path(unit_values):
    """Asserted independently of the literal interpreter, because the property
    that matters is absoluteness, not which binary.

    Root cause this guards (2026-08-18): a bare name in `ExecStart` is
    execvp-resolved against whatever PATH the `systemd --user` manager holds.
    That manager started at boot with linger and NO `~/.local/bin` on PATH —
    the graphical login that imports `~/.profile` came 84 minutes later — and
    the nightly's `Persistent=true` catch-up fired into exactly that window.
    Recorded in `scripts/legibility-trickle@.service`'s own comments; both
    sibling units pin an absolute `uv` for this reason.
    """
    command = unit_values(TEMPLATES_DIR / SERVICE_NAME, 'Service', 'ExecStart')[0]
    interpreter = shlex.split(command)[0]
    assert os.path.isabs(interpreter), (
        f'ExecStart must begin with an ABSOLUTE interpreter path: a bare name '
        f'is PATH-resolved against the systemd --user manager\'s environment, '
        f'which is the 2026-08-18 ENOENT failure; got {interpreter!r}'
    )


def test_service_is_a_journal_logging_oneshot_in_the_dark_factory_checkout(
        unit_values, unit_environment):
    """All generic legibility CODE lives in the dark-factory checkout (PRD §5);
    only the target project varies, via `%i`. `Type=oneshot` is what makes the
    unit's `Result` meaningful to `check_trickle_liveness.sh`, and journal
    routing is what makes a failed probe readable at all."""
    service = TEMPLATES_DIR / SERVICE_NAME
    assert unit_values(service, 'Service', 'Type') == ['oneshot']
    assert unit_values(service, 'Service', 'WorkingDirectory') == [PRODUCTION_ROOT]
    assert unit_values(service, 'Service', 'StandardOutput') == ['journal']
    assert unit_values(service, 'Service', 'StandardError') == ['journal']
    assert unit_values(service, 'Install', 'WantedBy') == ['timers.target']

    env = unit_environment(SERVICE, 'Service')
    assert env.get('PYTHONPATH') == f'{PRODUCTION_ROOT}/scripts', (
        f'`from legibility import ...` must resolve under a bare ExecStart; '
        f'parsed [Service] Environment={env!r}'
    )


def test_service_pins_no_claude_bin(unit_environment):
    """This probe never invokes the coder, so a `LEGIBILITY_CLAUDE_BIN` pin
    would be a needless copy of the nightly unit's — and a copy that rots.

    Its absence is worth asserting rather than merely omitting: the realistic
    way this unit gets written is by copying `legibility-trickle@.service`,
    which carries that pin for a load-bearing reason that does not apply here.
    """
    env = unit_environment(SERVICE, 'Service')
    assert 'LEGIBILITY_CLAUDE_BIN' not in env, (
        f'The health probe launches no CLI; parsed [Service] '
        f'Environment={env!r}'
    )


def test_neither_unit_pins_a_legibility_state_root():
    """THE GAP-3 regression test at the unit level, asserted over the PAIR.

    `scripts/legibility/trickle_state.py::trickle_state_path` is anchored to
    the invoking account's passwd home and reads no ambient environment, so the
    WRITER (`legibility-trickle@<project>.service`, under the `systemd --user`
    manager) and this READER agree on one file WITHOUT either unit pinning
    anything. Pinning a state root in ONE of the two units is precisely how the
    writer/reader divergence comes back — and it would come back silently,
    because each unit in isolation would look perfectly reasonable. So the
    absence is pinned for BOTH halves here, not just the new one.

    `XDG_STATE_HOME` and `HOME` are checked alongside the dedicated override
    because they were the two ambient levers the original defect rode in on.

    The var NAME is read from `trickle_state.STATE_ROOT_ENV` rather than
    hardcoded, mirroring `test_install_trickle_timer.py`'s reading of
    `coder._CLAUDE_BIN_ENV_VAR`: a rename there must fail HERE rather than
    leave this test guarding a string nothing reads.
    """
    forbidden = (trickle_state.STATE_ROOT_ENV, 'XDG_STATE_HOME', 'HOME=')
    for unit in (SERVICE_NAME, TRICKLE_SERVICE_NAME):
        live = _live_directive_text(unit)
        for name in forbidden:
            assert name not in live, (
                f'{unit} must set no state root: the writer and the reader '
                f'agree because trickle_state_path is anchored to the passwd '
                f'home, and pinning one unit re-opens GAP 3 invisibly. Found '
                f'{name!r} in a live directive. To RELOCATE state, set '
                f'{trickle_state.STATE_ROOT_ENV} for BOTH units at once.'
            )


def test_timer_fires_at_the_free_0430_slot(unit_values):
    """04:30 is the slot `OPERATIONS.md` §12 explicitly declared free.

    The rest of the nightly ladder is taken: 03:00 legibility-trickle, 03:30
    flag-marker sweep, 04:00 already DOUBLE-booked (reclaim-orphaned-worktrees
    + legibility-transcript-check), 05:00 memory-metadata-coverage-census. The
    stagger is deliberate — these jobs share one machine and, in several cases,
    the same backing stores.
    """
    assert unit_values(TEMPLATES_DIR / TIMER_NAME, 'Timer', 'OnCalendar') == [
        '*-*-* 04:30:00']


def test_timer_does_not_collide_with_an_occupied_slot(unit_values):
    """Guards the LADDER, not just this unit's own literal, so a future
    re-cadence onto a taken slot fails here instead of silently double-booking
    a third job.

    Ported from
    `test_install_memory_metadata_coverage_census_timer.py::test_timer_does_not_collide_with_an_occupied_slot`.
    Parsed on BOTH sides, so a commented-out slot in a sibling can neither
    manufacture a phantom collision nor mask a real one.

    Scoped to OUR slots only, deliberately: 04:00 is already legitimately
    double-booked between two existing units, so a global all-pairs-distinct
    assertion would fail on pre-existing state that is nobody's bug.
    """
    ours = set(unit_values(TEMPLATES_DIR / TIMER_NAME, 'Timer', 'OnCalendar'))
    assert ours, 'the timer declares no OnCalendar at all'
    for other in sorted(TEMPLATES_DIR.glob('*.timer')):
        if other.name == TIMER_NAME:
            continue
        clash = ours & set(unit_values(other, 'Timer', 'OnCalendar'))
        if clash:
            raise AssertionError(
                f'{TIMER_NAME} shares {sorted(clash)!r} with {other.name} — '
                f'pick a free slot and update the OPERATIONS.md §12 ladder '
                f'table')


def test_health_probe_runs_strictly_after_the_nightly_trickle(unit_values):
    """Asserted as a RELATION between the two parsed timers, not as a literal.

    The probe reads the state file the night's run WRITES. Ordering it before
    the trickle would make it report on the previous night forever — stale by
    exactly one day, and never wrong in a way that looks wrong. A literal pin
    on 04:30 cannot catch a future re-cadence of the TRICKLE past it; this can.
    """
    assert _on_calendar_time(unit_values, TIMER_NAME) > _on_calendar_time(unit_values, TRICKLE_TIMER_NAME), (
        f'the health probe ({_on_calendar_time(unit_values, TIMER_NAME)}) must fire after '
        f'the nightly trickle ({_on_calendar_time(unit_values, TRICKLE_TIMER_NAME)}) whose '
        f'state file it reads'
    )


def test_timer_catches_up_a_missed_night_and_avoids_a_thundering_herd(unit_values):
    """`Persistent=true` catches up a probe missed to a sleeping laptop instead
    of silently skipping it — and `max_age_hours=72` gives the freshness check
    enough slack that a catch-up firing BEFORE the nightly's own catch-up
    cannot false-alarm. `RandomizedDelaySec` keeps every project's timer from
    firing at the same instant."""
    timer = TEMPLATES_DIR / TIMER_NAME
    assert unit_values(timer, 'Timer', 'Persistent') == ['true']
    assert unit_values(timer, 'Timer', 'RandomizedDelaySec'), (
        'a randomized delay is what stops every project firing at once'
    )


def test_timer_is_installed_into_timers_target(unit_values):
    assert unit_values(TEMPLATES_DIR / TIMER_NAME, 'Install', 'WantedBy') == [
        'timers.target']


# ---------------------------------------------------------------------------
# Installer behaviour
# ---------------------------------------------------------------------------
#
# Driven via subprocess with `scripts/tests/conftest.py::fake_timer_systemctl` first
# on PATH plus a REAL invocation of `nightly.py resolve-config` -- no mock,
# because the whole point of the installer delegating to it is that this script
# never re-implements the project_id -> repo mapping. Real systemd is never
# touched.
#
# `XDG_CONFIG_HOME` is still the right lever for the unit DIRECTORY and is
# untouched by task 4514's state-path change -- that change scoped only the
# legibility STATE root, and every sibling installer test honours
# `XDG_CONFIG_HOME` the same way.


def _write_project_config(tmp_path, *, project_id, escalation_port=8199):
    """Write a minimal valid legibility.yaml for *project_id* under a fresh
    search root, returning that search root (the PARENT dir
    resolve_config_path globs one level down from)."""
    search_root = tmp_path / 'search-root'
    project_root = search_root / project_id
    legibility_dir = project_root / 'docs' / 'legibility'
    legibility_dir.mkdir(parents=True, exist_ok=True)
    config_path = legibility_dir / 'legibility.yaml'
    config_path.write_text(
        f'project_id: {project_id}\n'
        f'project_root: {project_root}\n'
        f'escalation_port: {escalation_port}\n'
        f'cwd_prefixes:\n'
        f' - {project_root}\n',
        encoding='utf-8',
    )
    return search_root


def _run_script(fake_timer_systemctl, project_id, *, env=None):
    """Run install-trickle-health-timer.sh <project_id> via subprocess.

    Puts *fake_timer_systemctl* first on PATH, reset on every call, and defaults
    INSTALL_TRICKLE_HEALTH_TIMER_PYTHON=sys.executable so the script's
    `nightly.py resolve-config` delegation runs the REAL resolver directly --
    no `uv run`/`--frozen` needed in-test. Callers supply XDG_CONFIG_HOME and
    LEGIBILITY_SEARCH_ROOTS via *env*; any ambient LEGIBILITY_SEARCH_ROOTS is
    popped so a developer's shell cannot make this resolve the real projects.
    """
    fake_timer_systemctl.reset()

    full_env = dict(os.environ)
    full_env['PATH'] = f"{fake_timer_systemctl.bin_dir}{os.pathsep}{full_env['PATH']}"
    full_env['INSTALL_TRICKLE_HEALTH_TIMER_PYTHON'] = sys.executable
    full_env.pop('LEGIBILITY_SEARCH_ROOTS', None)
    if env:
        full_env.update(env)
    return subprocess.run(
        ['bash', str(SCRIPT), project_id],
        env=full_env, capture_output=True, text=True, timeout=30,
    )


def test_script_is_executable():
    assert os.access(SCRIPT, os.X_OK), (
        f'Expected {SCRIPT} to be executable (os.X_OK); it is not. '
        f'Run: chmod +x {SCRIPT}'
    )


def test_install_copies_templates_and_enables_timer(fake_timer_systemctl, tmp_path):
    search_root = _write_project_config(tmp_path, project_id='proj_a')
    xdg_config = tmp_path / 'xdg-config'

    result = _run_script(
        fake_timer_systemctl, 'proj_a',
        env={'XDG_CONFIG_HOME': str(xdg_config),
             'LEGIBILITY_SEARCH_ROOTS': str(search_root)},
    )

    assert result.returncode == 0, (
        f'stdout={result.stdout!r} stderr={result.stderr!r}'
    )

    unit_dir = xdg_config / 'systemd' / 'user'
    service_path = unit_dir / SERVICE_NAME
    timer_path = unit_dir / TIMER_NAME
    assert service_path.is_file(), f'Expected {service_path} after install'
    assert timer_path.is_file(), f'Expected {timer_path} after install'
    # BYTE-identical, not merely similar: systemd expands %i itself, so the
    # templates are copied verbatim and never per-project edited. A copy that
    # diverged would mean the committed file is no longer what runs.
    assert service_path.read_bytes() == SERVICE.read_bytes()
    assert timer_path.read_bytes() == TIMER.read_bytes()

    calls = fake_timer_systemctl.calls()
    assert ['daemon-reload'] in calls, f'calls={calls!r}'
    assert ['enable', '--now', 'legibility-trickle-health@proj_a.timer'] in calls, (
        f'calls={calls!r}'
    )
    assert any(c[0] == 'list-timers' for c in calls), f'calls={calls!r}'


def test_install_does_not_kick_an_immediate_run(fake_timer_systemctl, tmp_path):
    """No surprise run at install time.

    The probe judges the state the night's trickle wrote, and escalates what
    it finds. Kicked at install, on a host whose first 03:00 trickle has not
    run yet, it would judge a night that has not happened. The installer arms
    the timer and leaves the run to it.
    """
    search_root = _write_project_config(tmp_path, project_id='proj_a')

    result = _run_script(
        fake_timer_systemctl, 'proj_a',
        env={'XDG_CONFIG_HOME': str(tmp_path / 'xdg-config'),
             'LEGIBILITY_SEARCH_ROOTS': str(search_root)},
    )

    assert result.returncode == 0, f'stdout={result.stdout!r} stderr={result.stderr!r}'
    for call in fake_timer_systemctl.calls():
        assert call[:1] != ['start'], f'unexpected immediate run: {call!r}'


def test_install_is_idempotent(fake_timer_systemctl, tmp_path):
    search_root = _write_project_config(tmp_path, project_id='proj_a')
    xdg_config = tmp_path / 'xdg-config'
    env = {'XDG_CONFIG_HOME': str(xdg_config),
           'LEGIBILITY_SEARCH_ROOTS': str(search_root)}

    result_1 = _run_script(fake_timer_systemctl, 'proj_a', env=env)
    assert result_1.returncode == 0, (
        f'stdout={result_1.stdout!r} stderr={result_1.stderr!r}'
    )

    result_2 = _run_script(fake_timer_systemctl, 'proj_a', env=env)
    assert result_2.returncode == 0, (
        f'Expected re-running the install to still exit 0; '
        f'stdout={result_2.stdout!r} stderr={result_2.stderr!r}'
    )


def test_install_fails_when_self_verify_omits_timer(fake_timer_systemctl, tmp_path):
    """The guard that stops `enable --now` nominally succeeding while nothing
    observable is installed.

    That gap is exactly how a probe ends up shipped-but-unbound, which IS
    GAP 2. An installer that reports success without confirming the timer is
    in `list-timers` would let this whole task's fix be silently absent on the
    host it was deployed to.
    """
    search_root = _write_project_config(tmp_path, project_id='proj_a')
    xdg_config = tmp_path / 'xdg-config'

    result = _run_script(
        fake_timer_systemctl, 'proj_a',
        env={
            'XDG_CONFIG_HOME': str(xdg_config),
            'LEGIBILITY_SEARCH_ROOTS': str(search_root),
            'FAKE_SYSTEMCTL_OMIT_LIST_TIMERS': '1',
        },
    )

    assert result.returncode != 0, (
        f'Expected a non-zero exit when list-timers omits the enabled timer '
        f'(self-verify failure); stdout={result.stdout!r} '
        f'stderr={result.stderr!r}'
    )


def test_install_fails_when_project_config_unresolvable(fake_timer_systemctl, tmp_path):
    """Config resolution happens FIRST, so an unresolvable project_id costs
    nothing: no systemctl invocation and no unit file left behind to be
    enabled by a later unrelated `daemon-reload`."""
    xdg_config = tmp_path / 'xdg-config'
    empty_search_root = tmp_path / 'empty-search-root'
    empty_search_root.mkdir()

    result = _run_script(
        fake_timer_systemctl, 'no_such_project',
        env={'XDG_CONFIG_HOME': str(xdg_config),
             'LEGIBILITY_SEARCH_ROOTS': str(empty_search_root)},
    )

    assert result.returncode != 0, (
        f'Expected a non-zero exit for an unresolvable project_id; '
        f'stdout={result.stdout!r} stderr={result.stderr!r}'
    )
    assert fake_timer_systemctl.calls() == [], (
        f'Expected NO systemctl invocation when config resolution fails '
        f'first; calls={fake_timer_systemctl.calls()!r}'
    )
    assert not (xdg_config / 'systemd' / 'user' / SERVICE_NAME).exists(), (
        'Expected no unit file to be installed when config resolution fails'
    )


def test_installer_does_not_touch_the_trickle_units(fake_timer_systemctl, tmp_path):
    """This installer must never disturb the LIVE nightly timer.

    The two jobs are siblings by design and their unit names differ by one
    word, so a copy-paste slip in the installer would plausibly re-copy or
    re-enable `legibility-trickle@.*` — and on the real host that unit is the
    thing actually producing the digests. Installing the health probe must be
    a strictly additive operation.
    """
    search_root = _write_project_config(tmp_path, project_id='proj_a')
    xdg_config = tmp_path / 'xdg-config'

    result = _run_script(
        fake_timer_systemctl, 'proj_a',
        env={'XDG_CONFIG_HOME': str(xdg_config),
             'LEGIBILITY_SEARCH_ROOTS': str(search_root)},
    )
    assert result.returncode == 0, (
        f'stdout={result.stdout!r} stderr={result.stderr!r}'
    )

    unit_dir = xdg_config / 'systemd' / 'user'
    strays = sorted(p.name for p in unit_dir.glob('legibility-trickle@.*'))
    assert strays == [], (
        f'the health installer wrote the nightly trickle\'s units: {strays!r}'
    )
    touched = [
        c for c in fake_timer_systemctl.calls()
        if any(TRICKLE_TIMER_NAME.replace('@.', '@') in a for a in c)
    ]
    assert touched == [], (
        f'the health installer invoked systemctl on the nightly trickle timer: '
        f'{touched!r}'
    )
