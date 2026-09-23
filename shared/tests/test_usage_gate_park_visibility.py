"""A fully-capped park must be VISIBLE, not silent (task 4945).

THE GAP THIS CLOSES, named by the code itself. ``orchestrator/evals/
runner.py`` documents that ``cap_wait_sanity_secs`` is consulted at exactly
one place — ``_check_cap_wait``, in the cap-hit branch AFTER an invocation
returned — so "the unbounded ``_open.wait()`` above is out of its reach",
and filed making a fully-capped park visible as a follow-up. This is that
follow-up.

MEASURED BEFORE THE FIX: ``UsageGate.before_invoke`` logged
``'All accounts capped — waiting for any to reopen'`` exactly ONCE at INFO
and then blocked on a bare ``await self._open.wait()`` with zero further
output, for however long the pool stayed frozen. A campaign parked on a
fully-capped pool was therefore indistinguishable from a hung run — and
with evals now permanently sharing the fleet pool rather than holding a
private reserve (ruling 2026-08-30, tasks 4741/4945), that park is a
routine outcome rather than an exotic one.

WHY THIS IS OBSERVABILITY ONLY. ``before_invoke`` is a hot fleet-wide
path. The heartbeat must not change WHAT it returns or WHEN it unblocks —
it still blocks until an account reopens, and it still never tight-spins.
``test_the_park_still_blocks_and_still_unblocks`` pins both halves, so a
future change to the wait loop cannot quietly turn a park into a
spin or into an early return.

SHAPE BORROWED, DELIBERATELY, from the proven ``cap_wait`` log in
``shared/cli_invoke.py::_check_cap_wait``: same ``json.dumps(...,
default=str)`` serialisation, same last-logged-at throttle against a
module-level interval, same WARNING level, same soonest-open field from
``gate.soonest_resets_at``. Fleet log consumers then parse one familiar
event schema rather than a second bespoke one, and the sibling defect that
shape already survived — ``test_cap_retry.py::
test_cap_wait_log_survives_non_serializable_soonest_resets_at`` — tells
this suite what it has to pin too.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest
from _usage_gate_test_helpers import make_gate

from shared import usage_gate as usage_gate_mod

LOGGER_NAME = 'shared.usage_gate'

PARK_EVENT = 'all_capped_park'

# The one-shot INFO line that predates this heartbeat. It fires once per
# pass through the selection loop, which makes it a direct COUNT of loop
# iterations — that is what lets the throttling test below prove the
# throttle is load-bearing rather than an artifact of the wait period.
ONE_SHOT_PARK_MESSAGE = 'All accounts capped'


def _park_records(caplog) -> list[tuple[logging.LogRecord, dict]]:
    """Every structured ``all_capped_park`` record, with its decoded payload."""
    found = []
    for rec in caplog.records:
        if rec.name != LOGGER_NAME:
            continue
        try:
            payload = json.loads(rec.getMessage())
        except (ValueError, TypeError):
            continue  # the co-occurring plain-string logs are expected
        if isinstance(payload, dict) and payload.get('event') == PARK_EVENT:
            found.append((rec, payload))
    return found


def _loop_passes(caplog) -> int:
    """How many times the selection loop reached the all-capped branch."""
    return sum(
        1
        for rec in caplog.records
        if rec.name == LOGGER_NAME and ONE_SHOT_PARK_MESSAGE in rec.getMessage()
    )


def _all_capped_gate(names: tuple[str, ...] = ('a', 'b')):
    """A gate whose every account is capped until well past any test's runtime."""
    gate = make_gate(list(names), wait_for_reset=False)
    for acct in gate._accounts:
        acct.capped = True
        acct.resets_at = datetime.now(UTC) + timedelta(hours=5)
    assert gate.is_paused, 'fixture must actually be parked, or every assertion is vacuous'
    return gate


async def _park_for(gate, seconds: float) -> None:
    """Drive ``before_invoke`` into the park and abandon it after *seconds*.

    The same shape as ``test_usage_gate_exhaustive.py::
    test_blocks_when_all_capped`` — driving a deliberately-unbounded wait
    under ``asyncio.wait_for`` inside ``pytest.raises`` is how this repo
    asserts on a park without hanging the suite. The TimeoutError IS the
    assertion that the park never returned early.
    """
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(gate.before_invoke(), timeout=seconds)


@pytest.fixture(autouse=True)
def _capture_gate_logs(caplog):
    """Capture at DEBUG so the level of each record can itself be asserted."""
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    return caplog


@pytest.fixture
def fast_heartbeat(monkeypatch):
    """Shrink the heartbeat interval so no test has to wait in real time.

    The interval MUST be a module-level constant for this to work — that is
    why step 8 declares one rather than inlining the number.
    """

    def _set(interval: float) -> float:
        monkeypatch.setattr(
            usage_gate_mod, '_ALL_CAPPED_PARK_LOG_INTERVAL_SECS', interval
        )
        return interval

    return _set


class TestAllCappedParkIsVisible:
    async def test_the_park_emits_a_structured_heartbeat(
        self, caplog, fast_heartbeat
    ) -> None:
        fast_heartbeat(0.02)
        gate = _all_capped_gate()

        await _park_for(gate, 0.2)

        records = _park_records(caplog)
        assert records, (
            'a fully-capped park emitted no all_capped_park record, so it is '
            'still indistinguishable from a hung run. Captured from '
            f'{LOGGER_NAME}: {[r.getMessage() for r in caplog.records]!r}'
        )
        _, payload = records[0]
        assert payload['account_count'] == 2, (
            'the heartbeat must say how many accounts are frozen — one capped '
            f'account and a whole frozen fleet read identically otherwise: {payload!r}'
        )
        assert isinstance(payload['elapsed_s'], (int, float)), (
            'the heartbeat must say how long the park has lasted; without it a '
            f'reader cannot tell a minute of waiting from a day: {payload!r}'
        )
        assert 'soonest_open_at' in payload, (
            'the heartbeat must say when the pool is expected to reopen — that '
            'is what turns "parked" into an actionable ETA (and it is legitimately '
            f'None when no capped account has published a reset time): {payload!r}'
        )

    async def test_the_heartbeat_is_a_warning_not_an_info(
        self, caplog, fast_heartbeat
    ) -> None:
        """INFO is what made the original park invisible in practice."""
        fast_heartbeat(0.02)
        gate = _all_capped_gate()

        await _park_for(gate, 0.1)

        records = _park_records(caplog)
        assert records, 'no heartbeat emitted — cannot assert its level'
        assert all(rec.levelno == logging.WARNING for rec, _ in records), (
            'the heartbeat must be WARNING so it survives default fleet log '
            'levels. The predecessor line it replaces was INFO, which is a '
            'large part of why a frozen pool went unnoticed: levels are '
            f'{[logging.getLevelName(rec.levelno) for rec, _ in records]!r}'
        )

    async def test_the_heartbeat_repeats_rather_than_firing_once(
        self, caplog, fast_heartbeat
    ) -> None:
        """One record at the start of a multi-hour park is not visibility.

        ALSO PINS THAT ``elapsed_s`` GROWS, which no other test in this file
        can see: the throttling test counts records and this one used to,
        and both stay green against a park whose clock is stuck. That is a
        reachable regression, not a hypothetical — returning
        ``_park_started_at`` to a local while leaving ``_park_last_logged_at``
        instance-scoped keeps the counts and the throttle exactly as they
        are while reporting ``elapsed_s: 0.0`` on every heartbeat, which
        destroys the half of the signal that separates a ten-minute park
        from a two-day one. The values are asserted NON-DECREASING rather
        than strictly increasing because the payload is rounded to 0.1s, so
        consecutive fast-interval heartbeats legitimately tie.
        """
        interval = fast_heartbeat(0.02)
        gate = _all_capped_gate()

        # Long enough that the clock's growth clears the 0.1s rounding
        # resolution several times over, so the assertion cannot pass or fail
        # on a rounding boundary.
        await _park_for(gate, interval * 25)

        records = _park_records(caplog)
        assert len(records) >= 2, (
            'the park emitted at most one heartbeat across the whole park, '
            'which is the ONE-SHOT behaviour this task replaces: a single line '
            'logged when the park began tells a reader nothing about whether '
            'it is still going now'
        )

        elapsed = [payload['elapsed_s'] for _, payload in records]
        assert elapsed == sorted(elapsed), (
            f'elapsed_s went backwards across one park: {elapsed!r}. It '
            'measures a single monotonic clock from the moment the pool '
            'stopped serving anyone, so it can only rise until the park ends'
        )
        assert elapsed[-1] > elapsed[0], (
            f'elapsed_s never advanced across the whole park: {elapsed!r}. A '
            'heartbeat that repeats a frozen number is a clock stuck at the '
            'start of the park — the reader still cannot tell a ten-minute '
            'freeze from a two-day one, which is the whole point of the field'
        )

    async def test_the_heartbeat_is_throttled_under_repeated_wakeups(
        self, caplog, fast_heartbeat
    ) -> None:
        """A woken-but-still-frozen loop must not turn the log into a flood.

        NOT a hypothetical: ``before_invoke``'s own comment records that
        out-of-scope callers mutate ``phase`` directly through the retained
        legacy shims, bypassing ``_transition`` and its ``_open`` recompute —
        so ``_open`` genuinely can be set while the fleet is still frozen.
        The loop then wakes, re-runs its sweep, finds nothing serviceable and
        re-parks. Without a throttle that is one log line per wakeup.

        Proven by COMPARING the heartbeat count against the loop-pass count,
        so the bound cannot be satisfied merely by the wait period being long:
        the loop demonstrably ran many more times than the heartbeat fired.
        """
        interval = fast_heartbeat(0.05)
        gate = _all_capped_gate()
        duration = 0.3

        async def _wake_repeatedly() -> None:
            while True:
                gate._open.set()
                await asyncio.sleep(0.001)

        waker = asyncio.create_task(_wake_repeatedly())
        try:
            await _park_for(gate, duration)
        finally:
            waker.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waker

        heartbeats = len(_park_records(caplog))
        passes = _loop_passes(caplog)

        assert passes > 10, (
            f'only {passes} loop passes occurred, so this test never created '
            'the flood it exists to bound and its throttling assertion would '
            'pass vacuously'
        )
        assert heartbeats <= math.ceil(duration / interval) + 2, (
            f'{heartbeats} heartbeats across {duration}s at a {interval}s '
            f'interval — the log rate must be bounded by the interval, not by '
            f'how often the loop happens to wake ({passes} passes)'
        )
        assert heartbeats < passes, (
            f'{heartbeats} heartbeats for {passes} loop passes: the throttle is '
            'not actually suppressing anything, so a frozen pool would emit one '
            'warning per wakeup and bury the signal it exists to raise'
        )

    async def test_a_serving_gate_emits_no_park_record(self, caplog) -> None:
        """The heartbeat must mean "frozen", or it means nothing."""
        gate = make_gate(['a', 'b'], wait_for_reset=False)

        lease = await asyncio.wait_for(gate.before_invoke(), timeout=1.0)

        assert lease is not None
        assert not _park_records(caplog), (
            'a gate that served a lease without parking emitted an '
            'all_capped_park record. A heartbeat that fires when nothing is '
            'wrong trains readers to ignore it: '
            f'{[p for _, p in _park_records(caplog)]!r}'
        )

    async def test_the_heartbeat_survives_a_non_serialisable_reset_time(
        self, caplog, fast_heartbeat, monkeypatch
    ) -> None:
        """``default=str`` is load-bearing, not decoration.

        The sibling path already took this defect once —
        ``test_cap_retry.py::
        test_cap_wait_log_survives_non_serializable_soonest_resets_at`` —
        where a non-serialisable ``soonest_resets_at`` raised TypeError out
        of ``json.dumps`` and aborted the caller. Here the blast radius
        would be worse: the raise would escape ``before_invoke`` itself, so
        an observability line would convert a survivable park into a hard
        failure on the fleet's hottest path.
        """
        fast_heartbeat(0.02)
        gate = _all_capped_gate()
        monkeypatch.setattr(
            type(gate), 'soonest_resets_at', property(lambda self: MagicMock())
        )

        await _park_for(gate, 0.1)

        records = _park_records(caplog)
        assert records, (
            'no heartbeat survived a non-serialisable soonest_resets_at — '
            'json.dumps must be called with default=str so a bad value is '
            'stringified rather than raised out of before_invoke'
        )


class TestAnAbandonedParkIsNotInherited:
    """A park that ends by cancellation must not bequeath its clock.

    ``before_invoke`` clears the heartbeat clock on exactly one path — the
    one where it finally serves a lease. A parked caller is routinely
    CANCELLED instead: callers wrap it in ``asyncio.wait_for``, which is
    what ``_park_for`` above does, and what the eval runner's own timeouts
    do. Left set, the timestamps make the NEXT park report an ``elapsed_s``
    measured from a park that ended long ago and suppress its first
    heartbeat for up to a full interval — reintroducing, on the
    cancellation path, both of the observability failures this task exists
    to fix.
    """

    async def test_an_abandoned_park_does_not_poison_the_next_one(
        self, caplog, fast_heartbeat
    ) -> None:
        interval = fast_heartbeat(0.05)
        gate = _all_capped_gate()

        # A park that runs for a couple of intervals and is then abandoned.
        await _park_for(gate, interval * 2.4)
        assert _park_records(caplog), (
            'the first park logged nothing, so this test cannot show that its '
            'clock was or was not inherited'
        )

        caplog.clear()
        await _park_for(gate, interval * 1.5)

        records = _park_records(caplog)
        assert records, (
            'the second park emitted no heartbeat at all: the abandoned park '
            "left `_park_last_logged_at` set, so the new park's first "
            'heartbeat was throttled against a park that had already ended'
        )
        first_elapsed = records[0][1]['elapsed_s']
        assert first_elapsed < interval, (
            f'the second park opened at elapsed_s={first_elapsed!r}, i.e. it '
            'inherited the abandoned park\'s start time. A pool frozen for '
            'zero seconds would be reported as frozen for as long as some '
            'earlier, unrelated park happened to last'
        )

    async def test_a_cancelled_sibling_does_not_reset_a_live_park(
        self, caplog, fast_heartbeat
    ) -> None:
        """Clearing the shared clock is for the LAST waiter out, not any.

        The clock is deliberately shared across concurrent callers so a
        frozen pool announces itself once per interval rather than once per
        waiter — which means clearing it on any cancellation would restart a
        LIVE park's elapsed_s every time one of its many siblings timed out.
        A fully-capped pool is exactly the situation with many waiters, so
        this is the common case, not the exotic one.
        """
        interval = fast_heartbeat(0.02)
        gate = _all_capped_gate()

        survivor = asyncio.create_task(gate.before_invoke())
        doomed = asyncio.create_task(gate.before_invoke())
        await asyncio.sleep(interval * 10)

        doomed.cancel()
        with pytest.raises(asyncio.CancelledError):
            await doomed

        before = [payload['elapsed_s'] for _, payload in _park_records(caplog)]
        assert before and before[-1] > 0, (
            f'the shared park clock never advanced before the cancellation '
            f'({before!r}), so a reset afterwards would be undetectable'
        )

        caplog.clear()
        try:
            await asyncio.sleep(interval * 10)
            after = [payload['elapsed_s'] for _, payload in _park_records(caplog)]
        finally:
            survivor.cancel()
            with pytest.raises(asyncio.CancelledError):
                await survivor

        assert after, 'the surviving park stopped emitting heartbeats entirely'
        assert after[0] >= before[-1], (
            f'the surviving park restarted its clock at {after[0]!r} after a '
            f'sibling was cancelled (it stood at {before[-1]!r}). The pool has '
            'been frozen continuously throughout; elapsed_s must keep '
            'measuring from when it froze, not from when some other waiter '
            'gave up'
        )


class TestParkSemanticsAreUnchanged:
    """The heartbeat is additive: same return, same blocking, no spin."""

    async def test_the_park_still_blocks_and_still_unblocks(
        self, fast_heartbeat
    ) -> None:
        interval = fast_heartbeat(0.02)
        gate = _all_capped_gate(('a',))

        # Blocks while frozen — several heartbeat intervals' worth.
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(gate.before_invoke(), timeout=interval * 5)

        # ...and still returns the account the moment one reopens.
        async def _reopen() -> None:
            await asyncio.sleep(interval)
            gate._accounts[0].capped = False
            gate._open.set()

        reopener = asyncio.create_task(_reopen())
        lease = await asyncio.wait_for(gate.before_invoke(), timeout=2.0)
        await reopener

        assert lease is not None and lease.token == 'fake-token-a', (
            'the park must still hand back the reopened account unchanged; the '
            f'heartbeat is observability only: {lease!r}'
        )

    async def test_the_park_does_not_tight_spin(self, caplog, fast_heartbeat) -> None:
        """A zero-timeout wait would burn a core for the whole freeze.

        Counted in selection-loop passes rather than wall clock, because
        that is the quantity a spin actually inflates: with the wait bounded
        by the heartbeat interval a 0.2s park makes a handful of passes,
        while an unbounded-timeout wait would make thousands.
        """
        interval = fast_heartbeat(0.05)
        duration = 0.2
        gate = _all_capped_gate()

        await _park_for(gate, duration)

        passes = _loop_passes(caplog)
        assert 0 < passes <= math.ceil(duration / interval) + 2, (
            f'the park made {passes} selection-loop passes in {duration}s at a '
            f'{interval}s heartbeat interval. Each pass re-runs the account '
            'sweep and awaits _refresh_capped_accounts, so an unbounded count '
            'is a busy-wait burning a core for the whole freeze — the wait must '
            'be bounded by the interval, never by zero'
        )
