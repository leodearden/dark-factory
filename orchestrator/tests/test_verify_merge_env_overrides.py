"""The real verify-merge CLI honours its flock-wait and watchdog env overrides.

ORCH_MERGE_VERIFY_FLOCK_WAIT_SECS and ORCH_WATCHDOG_HEARTBEAT_TIMEOUT_SECS /
ORCH_WATCHDOG_KILL_GRACE_SECS must take effect in a spawned
``orchestrator verify-merge``, measured by the child's own stopwatch: each
test runs the CLI under an instrumented bootstrap that wraps the production
gate callable and reports what the child's clock saw.  Those bootstraps and
their parsers live here because these two tests are their only consumers.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import re
import subprocess
from typing import NamedTuple

import pytest
from _orch_helpers import VERIFY_CLI_PER_TEST_TIMEOUT
from _verify_boundary_ceilings import ROW5_HOLDER_TEARDOWN_CEILING_SECS
from _verify_boundary_harness import (
    fast_spec,
    kill_holder_tree,
    sleeper_spec,
    spawn_verify_merge,
    stderr_tail,
    worktree_base_for,
    write_verify_config,
)
from test_cli import _setup_verify_repo

from orchestrator.verify_cancel import merge_verify_lock_path, write_lock_holder_pgid
from orchestrator.verify_runner import FLOCK_CONTENTION_CATEGORY, result_from_json

# ---------------------------------------------------------------------------
# Task 3369 -- in-child stopwatch for the flock GATE, replacing task 2921/2941's
# outer wall-clock subtraction in test_flock_wait_env_override_speeds_up_
# contention_result below.
#
# The quantity that test is about is the CLI's *bounded flock wait* -- the one
# thing ORCH_MERGE_VERIFY_FLOCK_WAIT_SECS controls.  Measuring it from OUTSIDE
# means measuring `subprocess wall time`, which is dominated by a term the
# override has nothing to do with: interpreter startup + `import
# orchestrator.cli`.  Task 2921 tried to cancel that term by subtracting a
# bare-import baseline, and task 2941 by subtracting a same-shape uncontended
# verify-merge -- but the term is not merely large, it is HIGHLY VARIABLE
# between two sequential subprocesses, so subtracting one sample of it from
# another does not cancel it.  Measured on this box with the override provably
# honored: two back-to-back children reported `import orchestrator.cli` at 1.9s
# and 4.0s, and a single-test isolated run left a 6.0s residue against a 5.0s
# ceiling -- i.e. green behaviour reported red, the same unsound-wall-clock-proxy
# defect fixed for TestB2PreTurn1Wedge in test_liveness_boundary_gate.py.
#
# So instrument the child instead and assert on the code-under-test's OWN clock.
# FLOCK_GATE_TIMING_BOOTSTRAP wraps orchestrator.cli.acquire_merge_verify_flock
# -- the production callable the gate actually invokes -- in a stopwatch and
# emits one line per acquire on STDERR (stdout is reserved for the VerifyResult
# JSON).  The CLI is otherwise untouched: same argv, same config, same env, same
# real fcntl.flock against a really-held lock.  Across every probe run the wait
# it reports for a 0.5s override was 0.507-0.509s, load notwithstanding, against
# the 10.0s MERGE_VERIFY_FLOCK_WAIT_SECS production default an un-wired override
# would leave -- a discriminant with ~20x margin instead of a coin flip.
# ---------------------------------------------------------------------------

#: Marker token the instrumented bootstrap prints, one line per gate acquire.
FLOCK_GATE_MARKER = '__FLOCK_GATE__'

#: printf-style line format shared between the bootstrap's child-side print
#: (embedded below via f-string substitution) and the parser self-test's
#: synthetic ``emitted`` fixture, mirroring WATCHDOG_GATE_LINE_FMT (amendment
#: pass) so a field rename is a rename in both places rather than two
#: hand-written copies that can silently drift apart -- see
#: test_flock_gate_timing_bootstrap_parses_its_own_marker_lines.
FLOCK_GATE_LINE_FMT = f'{FLOCK_GATE_MARKER} lock=%s timeout=%r waited=%.4f acquired=%r'

FLOCK_GATE_TIMING_BOOTSTRAP = f"""
import sys, time
import orchestrator.cli as _cli
_real_acquire = _cli.acquire_merge_verify_flock


def _timed_acquire(path, timeout_secs, **kwargs):
    _t0 = time.monotonic()
    fd = _real_acquire(path, timeout_secs, **kwargs)
    print(
        '{FLOCK_GATE_LINE_FMT}'
        % (path.name, timeout_secs, time.monotonic() - _t0, fd is not None),
        file=sys.stderr,
    )
    return fd


_cli.acquire_merge_verify_flock = _timed_acquire
_cli.main()
"""


class FlockGateWait(NamedTuple):
    """One instrumented ``acquire_merge_verify_flock`` call inside the child."""

    lock: str
    timeout_secs: float
    waited_secs: float
    acquired: bool


#: Task 5019 (amendment): ``waited`` is bounded to the exact shape its
#: ``%.4f`` format specifier always produces -- a run of digits, a dot, and
#: exactly 4 fraction digits, with no exponent notation possible regardless
#: of magnitude -- mirroring the fix to _WATCHDOG_GATE_RE.  ``timeout``
#: stays ``\S+``: it is emitted via ``%r`` of a float, whose repr switches
#: to exponent notation outside [1e-4, 1e16) and can render ``inf``/``nan``,
#: none of which a digit-shaped group would accept -- and unlike a TRAILING
#: field, ``timeout`` is an INTERIOR field already delimited by the literal
#: ``' waited='`` that follows it, so (like ``lock``, delimited by
#: ``' timeout='``) it was never reachable by the trailing-abutment defect
#: this task fixes; tightening it would only add a false-negative risk with
#: no corresponding safety gain.  ``acquired`` stays anchored to the literal
#: ``True|False`` alternation.
_FLOCK_GATE_RE = re.compile(
    re.escape(FLOCK_GATE_MARKER)
    + r' lock=(?P<lock>\S+) timeout=(?P<timeout>\S+) '
    r'waited=(?P<waited>-?\d+\.\d{4}) acquired=(?P<acquired>True|False)'
)


#: Task 5019 (amendment; tempered per step-4): matches just the marker plus
#: the rest of its line, tempered -- via the negative lookahead below -- to
#: stop at the next marker occurrence instead of running unbounded to
#: end-of-line, so :func:`parse_flock_gate_waits` can tell apart THREE cases
#: rather than two: (1) "marker present but its fields don't conform" (a
#: genuine corruption -- must fail loudly); (2) "marker's fields matched,
#: followed by an abutting co-writer's text" (tolerated: _FLOCK_GATE_RE's
#: bounded groups plus ``.match()`` not requiring end-of-line already handle
#: it); and (3) "marker's fields matched, followed immediately by a SECOND
#: marker record with no separating newline" (also tolerated, but only
#: because tempering stops the region before it can swallow the second
#: marker's text). An earlier, unbounded ``[^\n]*`` version of this regex
#: handled (1) and (2) but silently mishandled (3): it swallowed both
#: abutted records into a single finditer hit, so only the first record's
#: fields were ever offered to _FLOCK_GATE_RE and the second vanished
#: without a trace instead of being scanned on its own -- see
#: test_parse_flock_gate_waits_recovers_both_abutted_marker_records.
_FLOCK_GATE_LINE_RE = re.compile(
    re.escape(FLOCK_GATE_MARKER) + r'(?:(?!' + re.escape(FLOCK_GATE_MARKER) + r')[^\n])*'
)


def parse_flock_gate_waits(stderr: str) -> list[FlockGateWait]:
    """Parse the :data:`FLOCK_GATE_TIMING_BOOTSTRAP` lines out of a child's stderr.

    Every marker occurrence must have its fields conform to
    :data:`_FLOCK_GATE_RE` -- a marker line whose fields don't parse raises
    ``ValueError`` instead of being silently dropped, so a partial
    corruption can't quietly lower the ``max()`` the ceiling assertions in
    this module are computed over (task 5019 amendment; mirrors
    :func:`parse_watchdog_gate_fire_delays`). That guarantee depends on
    :data:`_FLOCK_GATE_LINE_RE` tempering its line region to stop at the
    next marker occurrence (task 5019 step-4): an earlier, unbounded version
    of that regex made the claim true only for non-marker trailing
    corruption -- a SECOND marker record abutting the first with no
    separating newline was swallowed into the first's line region and its
    fields were never even offered to :data:`_FLOCK_GATE_RE`, so it vanished
    silently instead of raising.
    """
    waits = []
    for line_match in _FLOCK_GATE_LINE_RE.finditer(stderr):
        line = line_match.group(0)
        m = _FLOCK_GATE_RE.match(line)
        if m is None:
            raise ValueError(
                'flock gate marker line does not match the expected '
                f'lock=/timeout=/waited=/acquired= field shape: {line!r}'
            )
        waits.append(
            FlockGateWait(
                lock=m.group('lock'),
                timeout_secs=float(m.group('timeout')),
                waited_secs=float(m.group('waited')),
                acquired=m.group('acquired') == 'True',
            )
        )
    return waits


def test_flock_gate_timing_bootstrap_parses_its_own_marker_lines():
    """parse_flock_gate_waits round-trips the bootstrap's marker format.

    The bootstrap emits its lines from inside a spawned child, so a drift
    between what it prints and what the parser accepts would silently return
    zero observations -- which would make the ceiling assertion in
    test_flock_wait_env_override_speeds_up_contention_result vacuous rather
    than red.  The emit format and this parser are therefore pinned together
    here, with no subprocess involved.

    ``emitted`` is built from FLOCK_GATE_LINE_FMT (amendment-pass fix,
    mirroring test_watchdog_gate_timing_bootstrap_parses_its_own_marker_lines)
    -- the SAME format string the bootstrap embeds into the child -- rather
    than a hand-copied literal, so a field rename in the bootstrap breaks
    this round-trip by construction instead of relying on two copies staying
    in sync by discipline.
    """
    emitted = (
        f'{FLOCK_GATE_LINE_FMT % ("_merge-verify.lock", 0.5, 0.0001, True)}\n'
        'some unrelated stderr chatter\n'
        f'{FLOCK_GATE_LINE_FMT % (".merge_verify.lock", 0.5, 0.5083, False)}\n'
    )

    waits = parse_flock_gate_waits(emitted)

    assert waits == [
        FlockGateWait('_merge-verify.lock', 0.5, 0.0001, True),
        FlockGateWait('.merge_verify.lock', 0.5, 0.5083, False),
    ]


def test_parse_flock_gate_waits_returns_empty_for_uninstrumented_stderr():
    """Stderr with no marker lines yields no observations.

    This is the case the ceiling assertion's non-vacuity guard exists to
    catch: if the CLI ever stops reaching the gate through the module-level
    ``orchestrator.cli.acquire_merge_verify_flock`` name the bootstrap
    patches, the child produces no marker lines at all, and an assertion over
    an empty list would pass by default.
    """
    assert parse_flock_gate_waits('Traceback (most recent call last):\nboom\n') == []


def test_parse_flock_gate_waits_survives_abutting_log_line():
    """An unrelated co-writer's stderr line landing with no separator must
    not corrupt the parse.

    Task 5019 amendment -- symmetric coverage for the flock twin of
    test_parse_watchdog_gate_fire_delays_survives_abutting_log_line.  Unlike
    the watchdog twin, the flock line's true last field (``acquired``) is
    already anchored to the ``True|False`` literal alternation, so an
    abutting line was never able to corrupt it via the exact mechanism
    observed in production for the watchdog twin (see the _FLOCK_GATE_RE
    docstring above) -- this test pins that already-correct behaviour so a
    future regex change can't silently break it.
    """
    corrupted = (
        f'{FLOCK_GATE_MARKER} lock=_merge-verify.lock timeout=0.5 '
        'waited=0.5083 acquired=True'
        '2026-09-01T00:00:00Z some unrelated interleaved log line\n'
    )

    assert parse_flock_gate_waits(corrupted) == [
        FlockGateWait('_merge-verify.lock', 0.5, 0.5083, True),
    ]


def test_parse_flock_gate_waits_raises_loudly_on_malformed_marker_line():
    """A marker line whose fields don't conform must raise, not vanish.

    Task 5019 amendment -- symmetric coverage for the flock twin of
    test_parse_watchdog_gate_fire_delays_raises_loudly_on_malformed_marker_line.
    Silently dropping a genuinely-malformed marker line would quietly lower
    the max() the ceiling assertion in
    test_flock_wait_env_override_speeds_up_contention_result is computed
    over instead of surfacing the corruption.
    """
    malformed = (
        f'{FLOCK_GATE_MARKER} lock=_merge-verify.lock timeout=0.5 '
        'waited=NOT_A_NUMBER acquired=True\n'
    )

    with pytest.raises(ValueError, match=re.escape(FLOCK_GATE_MARKER)):
        parse_flock_gate_waits(malformed)


def test_parse_flock_gate_waits_recovers_both_abutted_marker_records():
    """Two marker records landing back-to-back with no separator must both
    survive the parse.

    Task 5019 step-3 -- the marker-to-marker analogue of
    test_parse_flock_gate_waits_survives_abutting_log_line's marker-to-log
    abutment: the same shared stderr fd that lets an unrelated co-writer's
    text land immediately after a marker's trailing field can just as well
    let a SECOND marker record land there instead, with no intervening
    newline. _FLOCK_GATE_LINE_RE's unbounded ``[^\n]*`` line region greedily
    swallows the whole physical line -- including the second marker's text
    -- as one hit, so ``_FLOCK_GATE_RE.match()`` only ever sees (and
    returns) the FIRST record, and the second is silently dropped because
    finditer never revisits text its first match already consumed. Built
    from FLOCK_GATE_LINE_FMT (not hand-copied literals) so a field rename
    breaks this round-trip by construction.
    """
    abutted = (
        f'{FLOCK_GATE_LINE_FMT % ("a", 0.5, 0.0001, True)}'
        f'{FLOCK_GATE_LINE_FMT % ("b", 0.5, 9.9999, False)}\n'
    )

    waits = parse_flock_gate_waits(abutted)

    assert len(waits) == 2
    # The larger wait sits on the SECOND (currently-dropped) record -- if it
    # silently vanishes, the ceiling assertions this parser feeds would be
    # computed over a falsely-low max() instead of surfacing the corruption.
    assert max(w.waited_secs for w in waits) == 9.9999


# ---------------------------------------------------------------------------
# Task 4474 -- watchdog twin of the flock-gate timing bootstrap above.  Same
# defect class, same remedy: task 4248's verify attempt-1 observed
# test_watchdog_timeout_env_override_fires_fast_without_heartbeat's OUTER
# wall-clock fire_delay (pgid-file appearance to proc.wait() return) at
# 19.75s under a full-suite xdist storm -- past even the 15.0s production
# un-wired-override window, so no widened ceiling could both accommodate
# that and still discriminate a wired override from an un-wired one.  The
# outer measurement was never purely the watchdog's own window: it also
# carries scheduler latency between the watchdog thread actually firing and
# this PARENT process observing proc.wait() return, which balloons under
# CPU contention exactly like the import cost the old pgid-file wait was
# already compensating for (task 2921's own inline comment claiming
# otherwise is what the 19.75s observation falsifies).
#
# So, exactly as FLOCK_GATE_TIMING_BOOTSTRAP does for
# orchestrator.cli.acquire_merge_verify_flock, WATCHDOG_GATE_TIMING_BOOTSTRAP
# wraps the production callables cli.py invokes on this seam.  It patches
# TWO module-level names, because the quantity under test --
# ORCH_WATCHDOG_HEARTBEAT_TIMEOUT_SECS *and* ORCH_WATCHDOG_KILL_GRACE_SECS
# together -- spans two calls:
#
# * orchestrator.cli.start_stdin_watchdog is wrapped ONLY to capture t0
#   (time.monotonic(), armed the instant before the watchdog thread starts
#   its blocking select loop); its ``fire`` callback is passed through
#   untouched, so a caller that ever stopped passing ``fire=`` explicitly
#   would still get the real production default instead of a silently
#   disabled tree-kill (amendment-pass fix: an earlier draft wrapped
#   ``fire`` itself and dropped the kill in that branch).
# * orchestrator.cli.fire_watchdog_kill is wrapped to intercept TWO of its
#   injectable arguments:
#     - ``sleep``: replaced with a stopwatch wrapper that records how long
#       the SIGTERM -> SIGKILL grace pause ACTUALLY took, so
#       ORCH_WATCHDOG_KILL_GRACE_SECS is a MEASURED term rather than an
#       assumed one.  cli.py:773 calls
#       ``fire_watchdog_kill(pgid, grace_secs=grace_secs)`` with no
#       ``sleep=``, so this injection displaces no production argument (and
#       it falls back to the caller's own ``sleep`` if one is ever passed,
#       exactly as the ``exit_fn`` interception does).
#     - ``exit_fn``: the wrapped exit_fn prints the marker line to stderr
#       (flushed explicitly, since the real exit_fn is os._exit, which
#       skips normal buffer flushing), then delegates to the real exit_fn.
#
#   The reported fire_delay is RECONSTRUCTED from two measured terms rather
#   than read off a single clock:
#
#       fire_delay = (t_fire_entry - t0) + observed_grace_sleep
#
#   where t_fire_entry is monotonic() at ENTRY to the wrapped
#   fire_watchdog_kill -- which genuinely is the fire-callback entry,
#   because cli.py's ``_on_watchdog_fire`` (cli.py:771-773) resolves the
#   ``fire_watchdog_kill`` module global at CALL time and therefore lands in
#   the already-wrapped callable.  No wrapper around ``fire`` itself is
#   needed, so the warning above against wrapping ``fire`` stays honoured.
#
# THIRD CORRECTION (this pass -- task 4559 verify).  Commit aa3095175c
# ("amend: measure the full watchdog arm-to-kill window, not just
# fire-callback entry") moved the stop point from fire-callback ENTRY to
# post-kill inside exit_fn, precisely so ORCH_WATCHDOG_KILL_GRACE_SECS came
# under the ceiling assertion -- right in intent, but over-inclusive in
# extent.  Stopping the clock in exit_fn also swept in fire_watchdog_kill's
# TWO ppid_map_provider() calls (verify_cancel.py:774 before the SIGTERM
# sweep and verify_cancel.py:782 after the grace sleep; each is a full
# /proc iterdir plus one read_text() per pid -- read_ppid_map,
# verify_cancel.py:165).  Neither walk is an env-override quantity under
# test, and both scale with machine load: measured across a ~300x load
# swing the two walks moved 0.56s -> 2.60s while the heartbeat wait held at
# 0.513-0.530s and the grace sleep at 0.200-0.217s, carrying the total to
# 5.18s -- past the 4.0s ceiling -- on a tree where BOTH overrides were
# correctly wired.  That is the same load-sensitive-proxy defect as the
# 19.75s outer wall clock, merely relocated inside the child.
#
# The reconstruction above is the third position, and it keeps what each
# earlier one had to trade away: the heartbeat override stays asserted via
# (t_fire_entry - t0), as in the pre-amendment form; the kill-grace
# override stays asserted via observed_grace_sleep, as the amendment
# intended; and the /proc walks -- which appear in neither override -- are
# excluded from both terms.  The ceiling stays at 4.0s deliberately: it is
# the DISCRIMINANT (the nearest broken shape -- kill-grace regressed to its
# 5.0s production default with the heartbeat still wired -- lands at ~5.5s),
# so the MEASUREMENT was the thing to fix, never the threshold.
#
# Because observed_grace_sleep is measured rather than assumed, a silent
# failure of the ``sleep=`` injection would zero that term and quietly
# retire the kill-grace axis from the ceiling.  The marker line therefore
# carries ``grace=`` alongside ``fire_delay=``, and the ceiling test
# asserts the grace pause was actually OBSERVED -- see
# WATCHDOG_KILL_GRACE_OVERRIDE_SECS.  This class of test has now been
# mis-measured three times (2921's outer wall clock -> 4248 falsified it at
# 19.75s -> 4474 moved the clock inside the child but left the /proc walks
# in); that non-vacuity guard is what stops a fourth.
#
# The reported fire_delay is therefore immune both to outer scheduler noise
# and to the /proc-walk cost that machine load makes unbounded.
# ---------------------------------------------------------------------------

#: Marker token the instrumented bootstrap prints, one line per watchdog fire.
WATCHDOG_GATE_MARKER = '__WATCHDOG_GATE__'

#: printf-style line format shared between the bootstrap's child-side print
#: (embedded below via f-string substitution) and the parser self-test's
#: synthetic ``emitted`` fixture, so a field rename is a rename in both
#: places rather than two hand-written copies that can silently drift apart
#: (amendment-pass fix -- see test_watchdog_gate_timing_bootstrap_parses_
#: its_own_marker_lines).
WATCHDOG_GATE_LINE_FMT = f'{WATCHDOG_GATE_MARKER} fire_delay=%.4f grace=%.4f'

WATCHDOG_GATE_TIMING_BOOTSTRAP = f"""
import os, sys, time
import orchestrator.cli as _cli
_real_start_stdin_watchdog = _cli.start_stdin_watchdog
_real_fire_watchdog_kill = _cli.fire_watchdog_kill
_t0 = None


def _timed_start_stdin_watchdog(pgid, **kwargs):
    global _t0
    _t0 = time.monotonic()
    return _real_start_stdin_watchdog(pgid, **kwargs)


def _timed_fire_watchdog_kill(pgid, *, exit_fn=None, sleep=None, **kwargs):
    # Entry to THIS wrapper is the fire-callback entry: cli.py's
    # _on_watchdog_fire resolves the fire_watchdog_kill module global at call
    # time, so it lands here.  Snapshot before delegating, so the two
    # load-sensitive /proc walks inside the real body stay OUT of the
    # heartbeat term.
    _t_fire_entry = time.monotonic()
    _real_exit = exit_fn if exit_fn is not None else os._exit
    _real_sleep = sleep if sleep is not None else time.sleep
    _observed_grace = [0.0]

    def _timed_sleep(secs):
        _s0 = time.monotonic()
        try:
            return _real_sleep(secs)
        finally:
            _observed_grace[0] += time.monotonic() - _s0

    def _timed_exit(code):
        _elapsed = (_t_fire_entry - _t0) + _observed_grace[0]
        print(
            '{WATCHDOG_GATE_LINE_FMT}' % (_elapsed, _observed_grace[0]),
            file=sys.stderr,
        )
        sys.stderr.flush()
        _real_exit(code)

    return _real_fire_watchdog_kill(
        pgid, exit_fn=_timed_exit, sleep=_timed_sleep, **kwargs
    )


_cli.start_stdin_watchdog = _timed_start_stdin_watchdog
_cli.fire_watchdog_kill = _timed_fire_watchdog_kill
_cli.main()
"""


class WatchdogGateFire(NamedTuple):
    """One instrumented ``fire_watchdog_kill`` call inside the child.

    ``fire_delay_secs`` is the reconstructed
    ``(t_fire_entry - t0) + grace_secs`` window -- the heartbeat wait plus
    the observed SIGTERM->SIGKILL grace pause, deliberately EXCLUDING
    ``fire_watchdog_kill``'s two load-sensitive /proc walks (see the banner
    above).  ``grace_secs`` is broken out separately so the ceiling test can
    assert that term was actually observed rather than silently defaulted to
    zero by a failed ``sleep=`` injection.
    """

    fire_delay_secs: float
    grace_secs: float


#: Task 5019: the fraction groups are bounded to exactly 4 digits (rather
#: than an unbounded ``\S+``) because WATCHDOG_GATE_LINE_FMT's ``%.4f``
#: always renders exactly 4 fraction digits, and the child's stderr fd is
#: shared with unrelated writers -- an interleaved log line landing with no
#: separator right after the trailing field (observed in production,
#: mr-ccc80440: 'grace=0.2023' + '2026-09-01...' collapsing into
#: '0.20232026-09-01') must not be absorbed into the match.
_WATCHDOG_GATE_RE = re.compile(
    re.escape(WATCHDOG_GATE_MARKER)
    + r' fire_delay=(?P<fire_delay>-?\d+\.\d{4}) grace=(?P<grace>-?\d+\.\d{4})'
)

#: Task 5019 (amendment; tempered per step-4): matches just the marker plus
#: the rest of its line, tempered -- via the negative lookahead below -- to
#: stop at the next marker occurrence instead of running unbounded to
#: end-of-line, so :func:`parse_watchdog_gate_fire_delays` can tell apart
#: THREE cases rather than two: (1) "marker present but its fields don't
#: conform" (a genuine corruption -- must fail loudly); (2) "marker's
#: fields matched, followed by an abutting co-writer's text" (the exact
#: production defect this task fixes, tolerated because ``.match()``
#: doesn't require end-of-line -- see
#: test_parse_watchdog_gate_fire_delays_survives_abutting_log_line); and
#: (3) "marker's fields matched, followed immediately by a SECOND marker
#: record with no separating newline" (also tolerated, but only because
#: tempering stops the region before it can swallow the second marker's
#: text). An earlier, unbounded ``[^\n]*`` version of this regex handled
#: (1) and (2) but silently mishandled (3): it swallowed both abutted
#: records into a single finditer hit, so only the first record's fields
#: were ever offered to _WATCHDOG_GATE_RE and the second vanished without a
#: trace instead of being scanned on its own -- see
#: test_parse_watchdog_gate_fire_delays_recovers_both_abutted_marker_records.
_WATCHDOG_GATE_LINE_RE = re.compile(
    re.escape(WATCHDOG_GATE_MARKER) + r'(?:(?!' + re.escape(WATCHDOG_GATE_MARKER) + r')[^\n])*'
)


def parse_watchdog_gate_fire_delays(stderr: str) -> list[WatchdogGateFire]:
    """Parse the :data:`WATCHDOG_GATE_TIMING_BOOTSTRAP` lines out of a child's stderr.

    Every marker occurrence must have its fields conform to
    :data:`_WATCHDOG_GATE_RE` -- a marker line whose fields don't parse
    raises ``ValueError`` instead of being silently dropped, so a partial
    corruption can't quietly lower the ``max()`` the ceiling assertions in
    this module are computed over (task 5019 amendment; mirrors
    :func:`parse_flock_gate_waits`). That guarantee depends on
    :data:`_WATCHDOG_GATE_LINE_RE` tempering its line region to stop at the
    next marker occurrence (task 5019 step-4): an earlier, unbounded version
    of that regex made the claim true only for non-marker trailing
    corruption -- a SECOND marker record abutting the first with no
    separating newline was swallowed into the first's line region and its
    fields were never even offered to :data:`_WATCHDOG_GATE_RE`, so it
    vanished silently instead of raising.
    """
    fires = []
    for line_match in _WATCHDOG_GATE_LINE_RE.finditer(stderr):
        line = line_match.group(0)
        m = _WATCHDOG_GATE_RE.match(line)
        if m is None:
            raise ValueError(
                'watchdog gate marker line does not match the expected '
                f'fire_delay=/grace= field shape: {line!r}'
            )
        fires.append(
            WatchdogGateFire(
                fire_delay_secs=float(m.group('fire_delay')),
                grace_secs=float(m.group('grace')),
            )
        )
    return fires


def test_watchdog_gate_timing_bootstrap_parses_its_own_marker_lines():
    """parse_watchdog_gate_fire_delays round-trips the bootstrap's marker format.

    Mirrors test_flock_gate_timing_bootstrap_parses_its_own_marker_lines: the
    bootstrap emits its lines from inside a spawned child, so a drift between
    what it prints and what the parser accepts would silently return zero
    observations -- which would make the ceiling assertion in
    test_watchdog_timeout_env_override_fires_fast_without_heartbeat vacuous
    rather than red.  Unlike the flock twin (amendment-pass tightening for
    this test), ``emitted`` below is built from WATCHDOG_GATE_LINE_FMT -- the
    SAME format string the bootstrap embeds into the child -- rather than a
    hand-copied literal, so a field rename in the bootstrap breaks this
    round-trip by construction instead of relying on two copies staying in
    sync by discipline.
    """
    emitted = (
        f'{WATCHDOG_GATE_LINE_FMT % (0.6931, 0.2003)}\n'
        'some unrelated stderr chatter\n'
        f'{WATCHDOG_GATE_LINE_FMT % (14.9807, 5.0012)}\n'
    )

    assert parse_watchdog_gate_fire_delays(emitted) == [
        WatchdogGateFire(fire_delay_secs=0.6931, grace_secs=0.2003),
        WatchdogGateFire(fire_delay_secs=14.9807, grace_secs=5.0012),
    ]


def test_parse_watchdog_gate_fire_delays_returns_empty_for_uninstrumented_stderr():
    """Stderr with no marker lines yields no observations.

    Mirrors test_parse_flock_gate_waits_returns_empty_for_uninstrumented_stderr
    -- this is the case the ceiling assertion's non-vacuity guard exists to
    catch: if the CLI ever stops reaching the watchdog through the
    module-level ``orchestrator.cli.start_stdin_watchdog`` name the bootstrap
    patches, the child produces no marker lines at all, and an assertion over
    an empty list would pass by default.
    """
    assert parse_watchdog_gate_fire_delays('Traceback (most recent call last):\nboom\n') == []


def test_parse_watchdog_gate_fire_delays_survives_abutting_log_line():
    """An unrelated co-writer's stderr line landing with no separator must
    not corrupt the parse.

    Task 5019 -- observed in production (mr-ccc80440, task 4537's post-merge
    verify): the child's stderr fd is shared with anything else that writes
    to it, and under merge-verify load a timestamped log line landed
    immediately after the marker line's trailing ``grace=`` field with no
    intervening whitespace or newline, collapsing the real value (0.2023)
    and a '2026-09-01...' timestamp into one unbroken token,
    '0.20232026-09-01'. The old ``\\S+`` field regex swallowed that whole
    token and ``float()`` raised inside the parser, which the merge worker
    then mis-dispositioned as "branch_bug" against a branch that never
    touched this test.

    The fix bounds the numeric groups to the exact shape
    WATCHDOG_GATE_LINE_FMT's ``%.4f`` can produce -- a run of digits, a dot,
    and exactly 4 fraction digits -- so an abutting line's digits cannot be
    absorbed into the match. This test reproduces the exact corrupted shape
    and must fail against the unbounded ``\\S+`` regex (raises ValueError)
    and pass once the fraction groups are bounded.
    """
    corrupted = (
        f'{WATCHDOG_GATE_MARKER} fire_delay=0.6931 grace=0.2023'
        '2026-09-01T00:00:00Z some unrelated interleaved log line\n'
    )

    assert parse_watchdog_gate_fire_delays(corrupted) == [
        WatchdogGateFire(fire_delay_secs=0.6931, grace_secs=0.2023),
    ]


def test_parse_watchdog_gate_fire_delays_raises_loudly_on_malformed_marker_line():
    """A marker line whose fields don't conform must raise, not vanish.

    Task 5019 amendment: bounding the field groups (to reject anything that
    doesn't match WATCHDOG_GATE_LINE_FMT's exact shape) means a
    genuinely-malformed marker line -- as opposed to one merely followed by
    abutting unrelated text, which the abutment test above proves is still
    tolerated -- no longer matches _WATCHDOG_GATE_RE at all. Silently
    dropping such a line from the returned list would quietly lower the
    max() the ceiling assertion in
    test_watchdog_timeout_env_override_fires_fast_without_heartbeat is
    computed over instead of surfacing the corruption, cutting against the
    repo's no-silent-fail-soft / structured-facts-at-failure invariant.
    """
    malformed = f'{WATCHDOG_GATE_MARKER} fire_delay=NOT_A_NUMBER grace=0.2023\n'

    with pytest.raises(ValueError, match=re.escape(WATCHDOG_GATE_MARKER)):
        parse_watchdog_gate_fire_delays(malformed)


def test_parse_watchdog_gate_fire_delays_recovers_both_abutted_marker_records():
    """Two marker records landing back-to-back with no separator must both
    survive the parse.

    Task 5019 step-3 -- twin of
    test_parse_flock_gate_waits_recovers_both_abutted_marker_records; see
    that test's docstring for the mechanism
    (_WATCHDOG_GATE_LINE_RE's unbounded ``[^\n]*`` swallows the second
    marker's text into the first's line region, so finditer never revisits
    it and the second record is silently dropped). Built from
    WATCHDOG_GATE_LINE_FMT (not hand-copied literals) so a field rename
    breaks this round-trip by construction.
    """
    abutted = (
        f'{WATCHDOG_GATE_LINE_FMT % (0.6931, 0.2003)}'
        f'{WATCHDOG_GATE_LINE_FMT % (14.9807, 5.0012)}\n'
    )

    fires = parse_watchdog_gate_fire_delays(abutted)

    assert len(fires) == 2
    # The larger fire_delay sits on the SECOND (currently-dropped) record --
    # if it silently vanishes, the ceiling assertion this parser feeds would
    # be computed over a falsely-low max() instead of surfacing corruption.
    assert max(f.fire_delay_secs for f in fires) == 14.9807


# ---------------------------------------------------------------------------
# Task 2309 step-1 RED -- env-var test seams for remote-side timing constants
# (PRD SS11 Q1/Q2 tunability).  Production defaults are byte-identical when
# unset; these overrides exist ONLY so the SS9 boundary rows in
# test_laptop_warm_verify_boundary.py can run fast and deterministically
# instead of being wall-clock-bound on the 10s flock wait / 10s+5s watchdog
# window.  RED until step-2 wires them into cli.py.
# ---------------------------------------------------------------------------


#: Ceiling on the CLI's own measured flock wait.  Sits an order of magnitude
#: above the 0.5s override the test sets and half way below the 10.0s
#: MERGE_VERIFY_FLOCK_WAIT_SECS production default an un-wired override would
#: leave -- so it discriminates the two cases with a huge margin either side.
FLOCK_WAIT_CEILING_SECS = 5.0


@pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)  # task 3369: one subprocess again (the baseline probe is gone)
def test_flock_wait_env_override_speeds_up_contention_result(tmp_path):
    """ORCH_MERGE_VERIFY_FLOCK_WAIT_SECS overrides the flock bounded wait.

    Holds the real flock (+ writes the holder pgid) exactly as
    test_cli.py:1419 does, then spawns a real knob-on verify-merge with the
    override set small.  Today (RED) verify-merge ignores the env var and
    waits the full 10.0s production window; asserting the wait came in well
    under that fails until cli.py reads the override.

    task 3369 (de-flake): the assertion is on the wait the CLI's OWN clock
    measured, reported by :data:`FLOCK_GATE_TIMING_BOOTSTRAP` from inside the
    child, NOT on the child's wall-clock duration.

    Tasks 2376/2921/2941 successively widened, then baseline-subtracted, an
    outer wall-clock measurement, and it kept coming back red on green
    behaviour.  The reason is structural rather than a matter of tuning: the
    override governs one term of the child's runtime (the bounded flock wait),
    while its wall clock is dominated by a term the override has nothing to do
    with -- interpreter startup plus ``import orchestrator.cli`` -- and that
    term is not stable enough between two sequential subprocesses to cancel by
    subtracting one sample of it from another.  Probed on this box with the
    override provably honored (the gate itself reported waited=0.508s), two
    back-to-back children put that import at 1.9s and 4.0s, and a
    single-test, otherwise-idle run left a 6.0s residue against the old 5.0s
    ceiling.  Same defect and same remedy as TestB2PreTurn1Wedge in
    test_liveness_boundary_gate.py: stop using wall clock as a proxy for a
    quantity the code under test already measures itself.

    The child still runs the real CLI end to end -- real argv, real config,
    real env, real ``fcntl.flock`` against a really-held lock.  The bootstrap
    only wraps the production ``acquire_merge_verify_flock`` callable in a
    stopwatch, so an un-wired override still shows up as a ~10s wait.  Its
    one failure mode -- the CLI reaching the gate by some path other than the
    patched module-level name, leaving zero observations and a vacuously-true
    ceiling -- is closed by the non-vacuity assertion below.
    """
    repo, head_sha = _setup_verify_repo(tmp_path)
    worktree_base = worktree_base_for(repo)
    worktree_base.mkdir(parents=True, exist_ok=True)

    cfg_file = tmp_path / 'config.yaml'
    write_verify_config(cfg_file, repo, persistent_merge_worktree=True)

    write_lock_holder_pgid(worktree_base, 999999)
    lock_path = merge_verify_lock_path(worktree_base)
    held_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT)
    fcntl.flock(held_fd, fcntl.LOCK_EX)
    try:
        proc = spawn_verify_merge(
            sha=head_sha,
            spec=fast_spec(),
            cfg_file=cfg_file,
            extra_env={'ORCH_MERGE_VERIFY_FLOCK_WAIT_SECS': '0.5'},
            bootstrap=FLOCK_GATE_TIMING_BOOTSTRAP,
        )
        # task 2376: widened from 15s -- host oversubscription can delay
        # subprocess completion past a short deadline; the discriminating
        # invariant is the FLOCK_WAIT_CEILING_SECS assertion below (task
        # 3369), never this ceiling, which only bounds a wedged child.
        try:
            stdout, stderr = proc.communicate(timeout=60)
        finally:
            # task 4092/4946: kill_holder_tree so a communicate() timeout
            # here does not leak the child leader -- and any
            # start_new_session build command verify.py spawned under it --
            # while we still hold the flock; a child killed before the
            # unlock below can never win the lock and proceed.  Free on the
            # success path: communicate() has already reaped the child, so
            # the ALREADY-REAPED SHORT CIRCUIT returns at once.
            kill_holder_tree(proc, timeout=ROW5_HOLDER_TEARDOWN_CEILING_SECS)
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(held_fd, fcntl.LOCK_UN)
        os.close(held_fd)

    assert proc.returncode == 0, (
        f'expected exit 0 (contention result on stdout), got {proc.returncode}; '
        f'stderr={stderr_tail(stderr)!r}'
    )
    result = result_from_json(stdout.decode())
    assert result.category == FLOCK_CONTENTION_CATEGORY, (
        f'expected flock-contention result, got category={result.category!r} '
        f'stdout={stdout.decode()[:2000]!r}'
    )

    waits = parse_flock_gate_waits(stderr.decode())
    # Non-vacuity guard: the ceiling below is a max() over these observations,
    # so an empty list would pass by default.  Zero observations means the CLI
    # no longer calls the gate through the module-level
    # ``orchestrator.cli.acquire_merge_verify_flock`` name the bootstrap
    # patches -- a real signal that this test stopped measuring anything, not
    # a pass.
    assert waits, (
        'no instrumented flock-gate observation on the child stderr -- the '
        'timing bootstrap patches orchestrator.cli.acquire_merge_verify_flock, '
        'so zero observations means the gate is no longer reached through that '
        'name and the ceiling assertion below would be vacuous; '
        f'stderr={stderr_tail(stderr)!r}'
    )
    longest = max(w.waited_secs for w in waits)
    assert longest < FLOCK_WAIT_CEILING_SECS, (
        f'expected the CLI-measured flock wait to be well under the 10s '
        f'production wait (MERGE_VERIFY_FLOCK_WAIT_SECS) given the 0.5s env '
        f'override -- longest={longest:.2f}s over {len(waits)} gate acquire(s): '
        f'{waits}; the env override is not wired up yet'
    )


#: Ceiling on the CLI's own measured watchdog delay -- armed (immediately
#: before the watchdog thread's blocking select loop starts) to the
#: fire callback's entry, PLUS the observed SIGTERM->SIGKILL grace pause,
#: reported by the child's own clock via WATCHDOG_GATE_TIMING_BOOTSTRAP.
#: Those two terms are exactly the two env overrides under test; the two
#: /proc walks inside fire_watchdog_kill are load-sensitive and belong to
#: neither, so they are deliberately excluded (third correction -- see the
#: banner above WATCHDOG_GATE_TIMING_BOOTSTRAP). This is deliberately NOT just "halfway
#: between the fully-wired and fully-un-wired extremes": it also has to
#: catch either override being wired while the OTHER silently regresses to
#: its production default, e.g. only ORCH_WATCHDOG_KILL_GRACE_SECS stops
#: being threaded through (0.5s override + 5.0s production grace = ~5.5s) or
#: only ORCH_WATCHDOG_HEARTBEAT_TIMEOUT_SECS does (90.0s production timeout
#: + 0.2s override = ~90.2s; it was ~10.2s before task 4195 derived that
#: constant from the ssh transport, so this axis has only got easier to
#: discriminate). 4.0s sits comfortably under the nearer of
#: those two partial-regression cases (5.5s) with room to spare above the
#: fully-wired case (~0.7s: 0.5s heartbeat timeout + 0.2s kill grace) even
#: under load, so it discriminates all three broken shapes from the
#: fully-wired one -- the same discriminating-ceiling shape as
#: FLOCK_WAIT_CEILING_SECS, just against two failure axes instead of one.
WATCHDOG_FIRE_DELAY_CEILING_SECS = 4.0

#: The two env-override values the ceiling test threads into the child, named
#: once so the ``extra_env`` it spawns with and the assertions it makes about
#: what came back cannot drift apart.
WATCHDOG_HEARTBEAT_OVERRIDE_SECS = 0.5
WATCHDOG_KILL_GRACE_OVERRIDE_SECS = 0.2


@pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)  # task 4474: one subprocess, wedge-detector wait widened
def test_watchdog_timeout_env_override_fires_fast_without_heartbeat(tmp_path):
    """ORCH_WATCHDOG_HEARTBEAT_TIMEOUT_SECS/_KILL_GRACE_SECS override the watchdog window.

    Spawns a real verify-merge --request-id with stdin=PIPE and NEVER writes
    a heartbeat, asserting the watchdog fires and self-exits non-zero.

    task 4474 (de-flake): the assertion is on the delay the CLI's OWN clock
    measured -- armed to fire-callback entry, plus the observed kill grace,
    covering BOTH overrides under test and nothing else -- reported by
    :data:`WATCHDOG_GATE_TIMING_BOOTSTRAP` from inside the child, NOT on an
    outer wall-clock measurement.  (Task 4559 verify, third correction: the
    window deliberately EXCLUDES fire_watchdog_kill's two /proc walks, which
    belong to neither override and grow without bound under load -- see the
    banner above WATCHDOG_GATE_TIMING_BOOTSTRAP.)

    Task 2921 measured this as an OUTER wall-clock quantity -- pgid-file
    appearance to proc.wait() return -- reasoning that absorbing the
    load-sensitive ``from orchestrator.cli import main`` startup via the
    pgid-file wait would leave the remaining wait purely watchdog-bound.
    Task 4248's verify attempt-1 falsified that: under a full-suite xdist
    storm the outer fire_delay came in at 19.75s -- past even the 15.0s
    un-wired-override window production had at the time (90.0s+5.0s since
    task 4195) -- while an isolated rerun at the
    same HEAD passed in 6.51s. The outer measurement was never purely the
    watchdog's own window: it also carries scheduler latency between the
    watchdog thread actually firing and this PARENT process observing
    proc.wait() return, which balloons under CPU contention exactly like the
    import cost the pgid-file wait was already compensating for. Same defect
    and same remedy as test_flock_wait_env_override_speeds_up_contention_result
    (task 3369): stop using wall clock as a proxy for a quantity the code
    under test already measures itself.

    The child still runs the real CLI end to end -- real argv, real config,
    real env, a real watchdog thread racing a real select() on fd 0, real
    SIGTERM/SIGKILL signaling. The bootstrap only wraps the production
    ``start_stdin_watchdog``/``fire_watchdog_kill`` callables in a stopwatch,
    so an un-wired override still shows up as a ~95s delay (90.0s heartbeat
    timeout + 5.0s kill grace; it was ~15s before task 4195 derived the
    timeout from the ssh transport) -- see WATCHDOG_FIRE_DELAY_CEILING_SECS
    for the exact discriminant, including the two partial-regression shapes.
    Its two failure modes are both closed by non-vacuity assertions below:
    the CLI reaching the watchdog/kill by some path other than the patched
    module-level names (leaving zero observations and a vacuously-true
    ceiling), and the ``sleep=`` injection failing to land (leaving the
    reconstructed delay's grace term at 0.0, which would silently drop
    ORCH_WATCHDOG_KILL_GRACE_SECS out of the ceiling rather than fail it).
    """
    repo, head_sha = _setup_verify_repo(tmp_path)
    cfg_file = tmp_path / 'config.yaml'
    write_verify_config(cfg_file, repo, persistent_merge_worktree=False)

    REQUEST_ID = 'env-seam-watchdog-test'

    proc = spawn_verify_merge(
        sha=head_sha,
        spec=sleeper_spec(300.0),
        cfg_file=cfg_file,
        request_id=REQUEST_ID,
        extra_env={
            'ORCH_WATCHDOG_HEARTBEAT_TIMEOUT_SECS': str(WATCHDOG_HEARTBEAT_OVERRIDE_SECS),
            'ORCH_WATCHDOG_KILL_GRACE_SECS': str(WATCHDOG_KILL_GRACE_OVERRIDE_SECS),
        },
        bootstrap=WATCHDOG_GATE_TIMING_BOOTSTRAP,
    )
    try:
        # No heartbeat is ever written -- stdin stays open but silent, which
        # is sufficient to exercise the heartbeat-starvation timing path
        # (Rows 1-3 later distinguish EOF vs. timeout precisely). This
        # ceiling is a WEDGE DETECTOR, not a speed assertion (task 2376's
        # lesson for the flock twin): the discriminating invariant is the
        # WATCHDOG_FIRE_DELAY_CEILING_SECS assertion below, never this one,
        # which only bounds a genuinely wedged child and so costs zero
        # wall-clock on the success path. It must NOT be replaced by a bare
        # proc.communicate() here (amendment-pass fix): unlike the flock
        # twin, communicate() with no ``input=`` closes stdin as its FIRST
        # action, which would deliver EOF to the child and race the
        # heartbeat-starvation path this test exists to exercise.
        try:
            proc.wait(timeout=60.0)
        except subprocess.TimeoutExpired:
            pytest.fail(
                'verify-merge did not self-exit within 60s of startup -- '
                'the watchdog did not fire at all'
            )
        # Only NOW -- after the child has already self-exited on its own,
        # confirmed above without ever touching stdin -- is it safe to drain
        # both pipes via communicate(): it also drains stdout (never done
        # before; a pre-fire stdout write bigger than one pipe buffer used to
        # risk a wedge) and closes all three pipe fds, bounded by a short
        # timeout rather than an unbounded read (amendment-pass fix,
        # mirroring test_flock_wait_env_override_speeds_up_contention_
        # result's proc.communicate() as closely as this test's
        # stdin-must-stay-open-until-fire requirement allows).
        try:
            _, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pytest.fail(
                'verify-merge self-exited but its stdout/stderr pipes did '
                'not reach EOF within 5s afterwards -- a descendant may have '
                'survived fire_watchdog_kill and is still holding a pipe open'
            )
    finally:
        # Merge of two independent teardown fixes to this one block; BOTH are
        # load-bearing and neither subsumes the other.
        #
        # (task 4092) kill_holder_tree replaces the former leader-only
        # ``proc.kill()``.  The holder's build is ``sleep 300`` run by the
        # real CLI under ``start_new_session=True``, so it does NOT die with
        # the leader: a leader-only SIGKILL orphaned that sleep to init for
        # up to five minutes after every run.  kill_holder_tree sweeps the
        # descendant tree (plus a guarded killpg backstop) and reaps the
        # leader itself, so it also subsumes the old ``proc.wait(timeout=5)``.
        # Its ``poll()``-free contract is why no ``if proc.poll() is None``
        # guard is reintroduced here -- see its docstring's ALREADY-REAPED
        # SHORT CIRCUIT paragraph for the pid-recycling hazard that adds.
        #
        # (commit aa3095175c) closing the pipe fds on EVERY path, including
        # the wedge/failure path where the ``pytest.fail``s above fire before
        # proc.communicate() ever runs.  kill_holder_tree closes stdin on
        # both of its exits but deliberately leaves stdout/stderr alone (it
        # serves call sites that still read them), so the stdout/stderr
        # closes must stay HERE rather than migrate into the helper.
        kill_holder_tree(proc, timeout=ROW5_HOLDER_TEARDOWN_CEILING_SECS)
        if proc.stdout is not None:
            with contextlib.suppress(OSError):
                proc.stdout.close()
        if proc.stderr is not None:
            with contextlib.suppress(OSError):
                proc.stderr.close()

    assert proc.returncode != 0, (
        f'expected non-zero exit (watchdog self-kill), got {proc.returncode}'
    )

    fires = parse_watchdog_gate_fire_delays(stderr.decode(errors='replace'))
    # Non-vacuity guard #1: mirrors the flock twin's -- the ceiling assertion
    # below is a max() over these observations, so an empty list would pass
    # by default. Zero observations means the CLI no longer calls
    # start_stdin_watchdog/fire_watchdog_kill through the module-level
    # ``orchestrator.cli`` names the bootstrap patches -- a real signal that
    # this test stopped measuring anything, not a pass.
    assert fires, (
        'no instrumented watchdog-gate observation on the child stderr -- '
        'the timing bootstrap patches orchestrator.cli.start_stdin_watchdog '
        'and orchestrator.cli.fire_watchdog_kill, so zero observations means '
        'the watchdog/kill path is no longer reached through those names '
        f'and the ceiling assertion below would be vacuous; '
        f'stderr={stderr_tail(stderr)!r}'
    )
    # Non-vacuity guard #2 (task 4559 verify, the third correction -- see the
    # banner above WATCHDOG_GATE_TIMING_BOOTSTRAP). fire_delay is now
    # RECONSTRUCTED as (t_fire_entry - t0) + observed_grace_sleep so the two
    # load-sensitive /proc walks stay out of it. That makes the kill-grace
    # axis depend on the bootstrap's ``sleep=`` injection actually landing:
    # if fire_watchdog_kill ever stopped accepting ``sleep`` or stopped
    # sleeping, the grace term would silently collapse to 0.0, every
    # fire_delay would shrink, and the ceiling would keep passing while
    # having quietly stopped covering ORCH_WATCHDOG_KILL_GRACE_SECS at all.
    # A real time.sleep(grace) can only overshoot its argument, never
    # undershoot it, so requiring the observed pause to reach the override is
    # safe under any load -- and a grace that regressed to the 5.0s
    # production default is NOT filtered out here on purpose: it flows into
    # fire_delay and is caught by the ceiling below, which is the axis that
    # is supposed to catch it.
    unmeasured = [f for f in fires if f.grace_secs < WATCHDOG_KILL_GRACE_OVERRIDE_SECS]
    assert not unmeasured, (
        f'the SIGTERM->SIGKILL grace pause was not observed at its '
        f'{WATCHDOG_KILL_GRACE_OVERRIDE_SECS}s override on '
        f'{len(unmeasured)} of {len(fires)} fire(s): {unmeasured}. '
        f'fire_delay is reconstructed as (fire-entry - armed) + observed '
        f'grace, so an unobserved grace term silently retires '
        f'ORCH_WATCHDOG_KILL_GRACE_SECS from the ceiling assertion below '
        f'rather than failing it -- the bootstrap\'s sleep= injection into '
        f'fire_watchdog_kill is no longer taking effect; '
        f'stderr={stderr_tail(stderr)!r}'
    )
    longest = max(f.fire_delay_secs for f in fires)
    assert longest < WATCHDOG_FIRE_DELAY_CEILING_SECS, (
        f'expected the CLI-measured watchdog delay (heartbeat wait + kill '
        f'grace, excluding fire_watchdog_kill\'s /proc walks) to be well '
        f'under the 95s production window (90s heartbeat timeout + 5s kill '
        f'grace) given the '
        f'{WATCHDOG_HEARTBEAT_OVERRIDE_SECS}s+{WATCHDOG_KILL_GRACE_OVERRIDE_SECS}s '
        f'env overrides -- longest={longest:.2f}s over {len(fires)} fire(s): '
        f'{fires}; the watchdog env overrides are not fully wired up'
    )
