"""Tests for scripts/legibility/install-trickle-health-timer.sh, and for the two
systemd unit templates it installs (task 4514, GAP 2).

WHY THIS FILE IS THE REGRESSION TEST FOR GAP 2. The defect was not that either
trickle probe was wrong — `check_trickle_progress.py` and
`check_trickle_liveness.sh` both worked. It was that NOTHING RAN THEM: repo-wide,
every reference to either was prose, a docstring, a test or PRD text, and the
only bindings either ever had were the one-shot `before_done` milestone
predicates on tasks 2587/2615 (both `done`, and a completed milestone predicate
never runs again). A probe nobody invokes is documentation. So the assertions
that a unit EXISTS and that its `ExecStart` NAMES `check_trickle_health.py` are
not boilerplate — they are the thing that keeps the probes bound.

Template-content invariants live here rather than in a separate file, per
`test_install_trickle_timer.py::test_service_template_pins_claude_bin`'s
precedent; the installer-behaviour half is driven via subprocess with a FAKE
`systemctl` shimmed onto PATH. Real systemd is never touched.
"""
from __future__ import annotations

import os
import shlex
from pathlib import Path

from legibility import trickle_state

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


def _directives(name) -> dict[str, list[tuple[str, str]]]:
    """Parse a systemd unit into `{section: [(key, value), ...]}`.

    Copied from
    `test_install_memory_metadata_coverage_census_timer.py::_directives` per
    this directory's deliberate copy-not-share convention for unit/fake-systemctl
    helpers.

    WHY A PARSER AND NOT A SUBSTRING SCAN. These units comment heavily by
    design, and a raw-substring check cannot tell a live DIRECTIVE from the
    COMMENT that explains it. Both units here carry prose naming
    `Persistent=true`, `check_trickle_health.py` and the state-root variable,
    so every substring pin over the raw text would stay green through an
    outright deletion of the directive it thought it was checking.

    Duplicate keys are preserved as separate pairs, never collapsed:
    `configparser` would collapse or reject them, and a repeated `OnCalendar=`
    is an ADDITIONAL firing to systemd, not an override.

    Accepts a unit filename (resolved under TEMPLATES_DIR) or a Path, so
    sibling `*.timer` units can be parsed for the collision check.
    """
    path = name if isinstance(name, Path) else TEMPLATES_DIR / name
    sections: dict[str, list[tuple[str, str]]] = {}
    current = ''
    for raw in path.read_text().splitlines():
        line = raw.strip()
        # FULL-LINE comments only: systemd treats `#`/`;` as a comment lead-in
        # at the start of a line, and a directive's value may legitimately
        # contain either character (a Documentation= URL fragment, say).
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
    is declared exactly once.
    """
    return [v for k, v in directives.get(section, []) if k == key]


def _service_environment(service_text):
    """Parse the [Service] section's `Environment=` assignments into a dict.

    Copied from `test_install_trickle_timer.py::_service_environment`, same
    copy-not-share convention. shlex.split()s each directive's right-hand side
    before splitting every token on its FIRST `=`, so a future reflow of the
    unit onto systemd's legal space-separated multi-assignment form does not
    silently stop being checked.
    """
    env = {}
    in_service = False
    for raw_line in service_text.splitlines():
        line = raw_line.strip()
        if line.startswith('['):
            in_service = line == '[Service]'
            continue
        if not in_service or not line.startswith('Environment='):
            continue
        for token in shlex.split(line[len('Environment='):]):
            name, sep, value = token.partition('=')
            if sep:
                env[name] = value
    return env


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


def _on_calendar_time(timer_name) -> str:
    """The `HH:MM:SS` of a timer declaring exactly one `*-*-* HH:MM:SS` slot."""
    slots = _values(_directives(timer_name), 'Timer', 'OnCalendar')
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


def test_service_execstart_runs_the_health_probe_for_the_instance():
    """`ExecStart` must name THIS probe and take its project from `%i`.

    systemd expands `%i` at instantiation, so the template is copied verbatim
    and never per-project customised — one committed file serves dark_factory,
    reify and every project after them.
    """
    exec_starts = _values(_directives(SERVICE_NAME), 'Service', 'ExecStart')
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


def test_service_execstart_interpreter_is_an_absolute_path():
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
    command = _values(_directives(SERVICE_NAME), 'Service', 'ExecStart')[0]
    interpreter = shlex.split(command)[0]
    assert os.path.isabs(interpreter), (
        f'ExecStart must begin with an ABSOLUTE interpreter path: a bare name '
        f'is PATH-resolved against the systemd --user manager\'s environment, '
        f'which is the 2026-08-18 ENOENT failure; got {interpreter!r}'
    )


def test_service_is_a_journal_logging_oneshot_in_the_dark_factory_checkout():
    """All generic legibility CODE lives in the dark-factory checkout (PRD §5);
    only the target project varies, via `%i`. `Type=oneshot` is what makes the
    unit's `Result` meaningful to `check_trickle_liveness.sh`, and journal
    routing is what makes a failed probe readable at all."""
    service = _directives(SERVICE_NAME)
    assert _values(service, 'Service', 'Type') == ['oneshot']
    assert _values(service, 'Service', 'WorkingDirectory') == [PRODUCTION_ROOT]
    assert _values(service, 'Service', 'StandardOutput') == ['journal']
    assert _values(service, 'Service', 'StandardError') == ['journal']
    assert _values(service, 'Install', 'WantedBy') == ['timers.target']

    env = _service_environment(SERVICE.read_text())
    assert env.get('PYTHONPATH') == f'{PRODUCTION_ROOT}/scripts', (
        f'`from legibility import ...` must resolve under a bare ExecStart; '
        f'parsed [Service] Environment={env!r}'
    )


def test_service_pins_no_claude_bin():
    """This probe never invokes the coder, so a `LEGIBILITY_CLAUDE_BIN` pin
    would be a needless copy of the nightly unit's — and a copy that rots.

    Its absence is worth asserting rather than merely omitting: the realistic
    way this unit gets written is by copying `legibility-trickle@.service`,
    which carries that pin for a load-bearing reason that does not apply here.
    """
    env = _service_environment(SERVICE.read_text())
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


def test_timer_fires_at_the_free_0430_slot():
    """04:30 is the slot `OPERATIONS.md` §12 explicitly declared free.

    The rest of the nightly ladder is taken: 03:00 legibility-trickle, 03:30
    flag-marker sweep, 04:00 already DOUBLE-booked (reclaim-orphaned-worktrees
    + legibility-transcript-check), 05:00 memory-metadata-coverage-census. The
    stagger is deliberate — these jobs share one machine and, in several cases,
    the same backing stores.
    """
    assert _values(_directives(TIMER_NAME), 'Timer', 'OnCalendar') == [
        '*-*-* 04:30:00']


def test_timer_does_not_collide_with_an_occupied_slot():
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


def test_health_probe_runs_strictly_after_the_nightly_trickle():
    """Asserted as a RELATION between the two parsed timers, not as a literal.

    The probe reads the state file the night's run WRITES. Ordering it before
    the trickle would make it report on the previous night forever — stale by
    exactly one day, and never wrong in a way that looks wrong. A literal pin
    on 04:30 cannot catch a future re-cadence of the TRICKLE past it; this can.
    """
    assert _on_calendar_time(TIMER_NAME) > _on_calendar_time(TRICKLE_TIMER_NAME), (
        f'the health probe ({_on_calendar_time(TIMER_NAME)}) must fire after '
        f'the nightly trickle ({_on_calendar_time(TRICKLE_TIMER_NAME)}) whose '
        f'state file it reads'
    )


def test_timer_catches_up_a_missed_night_and_avoids_a_thundering_herd():
    """`Persistent=true` catches up a probe missed to a sleeping laptop instead
    of silently skipping it — and `max_age_hours=72` gives the freshness check
    enough slack that a catch-up firing BEFORE the nightly's own catch-up
    cannot false-alarm. `RandomizedDelaySec` keeps every project's timer from
    firing at the same instant."""
    timer = _directives(TIMER_NAME)
    assert _values(timer, 'Timer', 'Persistent') == ['true']
    assert _values(timer, 'Timer', 'RandomizedDelaySec'), (
        'a randomized delay is what stops every project firing at once'
    )


def test_timer_is_installed_into_timers_target():
    assert _values(_directives(TIMER_NAME), 'Install', 'WantedBy') == [
        'timers.target']
