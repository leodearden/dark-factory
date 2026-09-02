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
        """One record at the start of a multi-hour park is not visibility."""
        interval = fast_heartbeat(0.02)
        gate = _all_capped_gate()

        await _park_for(gate, interval * 10)

        assert len(_park_records(caplog)) >= 2, (
            'the park emitted at most one heartbeat across ten intervals, which '
            'is the ONE-SHOT behaviour this task replaces: a single line logged '
            'when the park began tells a reader nothing about whether it is '
            'still going now'
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
