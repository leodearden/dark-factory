"""Deterministic unit tests for ``_orch_helpers.wait_responsive`` (task 3980).

``wait_responsive`` exists because the late-arrival tests in
test_merge_speculation.py flake under host oversubscription: their
``asyncio.wait_for(req.result, timeout=MERGE_RESULT_TIMEOUT)`` deadlines are
charged in WALL CLOCK, so a descheduled xdist worker fails a test whose logic
completed correctly (measured: green path 4.55s/4.08s/3.79s in isolation,
observed failing at the 45s and 25s deadlines under load avg 246.59 on 32
cores -- a >=11x tail).

The fix charges the budget in EVENT-LOOP-RESPONSIVE time instead: each polling
slice contributes only ``min(observed, slice_s)``, so wall clock the worker
never got is not billed to the awaited work. The behaviours pinned here are the
whole contract:

  (a) a promptly-resolved awaitable returns its value unchanged, fast;
  (b) a REAL hang on a RESPONSIVE loop still reports red, loudly, by label --
      the fix must not be able to mask a genuine merge-pipeline deadlock;
  (c) under simulated starvation the budget is NOT charged in wall-clock terms,
      so a wait whose APPARENT wall clock far exceeds ``timeout`` still
      succeeds (the case below lifts ``max_wall_s`` to isolate the accounting;
      in production that stretch is hard-bounded -- see (d));
  (d) the ``max_wall_s`` cap still terminates a permanently-starved wait, so a
      stretched wait can never outlive its class ``@pytest.mark.timeout`` and
      let pytest-timeout's thread method ``os._exit()`` the xdist worker;
  (e) a failure of the awaited work propagates AS ITSELF, and a caller-owned
      Future is left untouched on give-up -- both promised by the helper's
      docstring and relied on by every late-arrival call site.

Everything here is hermetic and sub-second: no git, no merge worker, tiny
injected ``timeout``/``slice_s``/``max_wall_s`` values rather than the
production 45s, and a synthetic monotonic clock for the starvation cases.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any

import _orch_helpers
import pytest
from _orch_helpers import (
    MERGE_RESULT_TIMEOUT,
    RESPONSIVE_WAIT_STRETCH,
    RESPONSIVE_WAIT_WALL_CAP,
    wait_responsive,
)

# Tiny injected bounds: this whole module must run in ~a second.
SLICE = 0.02
BUDGET = 0.5
WALL_CAP = 15.0

# Starvation factor for the synthetic clock. Deliberately far larger than the
# ~7.7x oversubscription measured at failure time so cases (c)/(d) carry a wide
# margin and cannot themselves become load-flaky.
STARVE = 200.0


class _StarvationClock:
    """Monotonic clock that runs *factor* times faster than real time.

    Simulates a descheduled xdist worker: a nominal ``slice_s`` sleep appears
    to have consumed ``factor * slice_s``, which is exactly the evidence
    ``wait_responsive`` must refuse to charge against the budget.
    """

    def __init__(self, factor: float) -> None:
        self._factor = factor
        self._origin = time.monotonic()

    def __call__(self) -> float:
        return (time.monotonic() - self._origin) * self._factor


def _reported(message: str, field: str) -> float:
    """Extract ``<field>=<number>s`` from a wait_responsive failure message.

    Pins the diagnostic contract: a give-up must report the charged total, the
    true elapsed wall clock and the implied starvation ratio, so a starved run
    is self-diagnosing rather than an opaque timeout.
    """
    match = re.search(rf'{field}=([0-9.]+)', message)
    assert match is not None, f'failure message does not report {field}=: {message}'
    return float(match.group(1))


@pytest.mark.asyncio
class TestWaitResponsiveHappyPath:
    """(a) A promptly-resolved awaitable returns its value, well under budget."""

    async def test_pre_resolved_future_returns_value_unchanged(self) -> None:
        fut: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        fut.set_result('sentinel')

        started = time.monotonic()
        got = await wait_responsive(
            fut, timeout=BUDGET, label='happy-pre', slice_s=SLICE, max_wall_s=WALL_CAP
        )

        assert got == 'sentinel'
        assert time.monotonic() - started < WALL_CAP

    async def test_future_resolved_after_a_few_slices_returns_value(self) -> None:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[int] = loop.create_future()
        loop.call_later(SLICE * 3, lambda: None if fut.done() else fut.set_result(7))

        got = await wait_responsive(
            fut, timeout=BUDGET, label='happy-slices', slice_s=SLICE, max_wall_s=WALL_CAP
        )

        assert got == 7

    async def test_accepts_a_coroutine_and_returns_its_value(self) -> None:
        async def _work() -> str:
            await asyncio.sleep(SLICE)
            return 'from-coro'

        got = await wait_responsive(
            _work(), timeout=BUDGET, label='happy-coro', slice_s=SLICE, max_wall_s=WALL_CAP
        )

        assert got == 'from-coro'

    async def test_defaults_derive_from_the_shared_merge_constant(self) -> None:
        # Never a fresh literal: the cap derives from MERGE_RESULT_TIMEOUT, and
        # the ratio of exactly 2 is what the paired @pytest.mark.timeout
        # arithmetic in test_merge_speculation.py depends on -- this assertion
        # goes red if RESPONSIVE_WAIT_STRETCH is ever moved off 2.0, which is
        # precisely what would invalidate that arithmetic.
        #
        # A second assertion restating RESPONSIVE_WAIT_WALL_CAP's own
        # definition (`int(RESPONSIVE_WAIT_STRETCH * MERGE_RESULT_TIMEOUT)`)
        # used to sit here; it was deleted in the amendment pass because it
        # could not fail for ANY values of the three constants -- the same
        # vacuity class this task exists to eliminate.
        assert RESPONSIVE_WAIT_WALL_CAP == 2 * MERGE_RESULT_TIMEOUT


@pytest.mark.asyncio
class TestWaitResponsiveDefaultCapScalesWithNominal:
    """The default ``max_wall_s`` must SCALE with the caller's ``timeout``.

    Reviewer finding on esc-3980-3: while the default was the flat
    ``RESPONSIVE_WAIT_WALL_CAP`` (90s), a site passing
    ``timeout=MERGE_GATE_BARRIER_TIMEOUT`` (15s) could consume 90s of wall
    clock while ``_call_wait_budget`` in test_merge_queue_concurrent_verify.py
    billed it ``min(15 * RESPONSIVE_WAIT_STRETCH, 90)`` = 30s.  That made the
    AST auditor an UNDER-count rather than an upper bound, and the true worst
    case for TestLateArrivalCleanCAS 365s against a 300s
    ``@pytest.mark.timeout`` -- an ``os._exit()`` of the xdist worker.

    Of the three, exactly ONE is discriminating against the old flat default:
    ``test_small_nominal_gives_up_at_the_scaled_cap_not_the_flat_one``, verified
    by mutation (restore `cap = RESPONSIVE_WAIT_WALL_CAP` and it fails
    `assert 90.0 == 2.0 +- 2.0e-06`).  The other two CANNOT discriminate, by
    construction, and are here as regression guards on the paths the scaling
    default must not break: the ceiling test asserts `cap ==
    RESPONSIVE_WAIT_WALL_CAP`, which the flat default trivially satisfied, and
    the explicit-`max_wall_s` test exercises an override the flat default also
    honoured.  Stated precisely rather than claiming all three discriminate --
    an overclaimed guard is the same vacuity defect this task exists to remove.
    """

    async def test_small_nominal_gives_up_at_the_scaled_cap_not_the_flat_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = _StarvationClock(STARVE)
        monkeypatch.setattr(_orch_helpers, '_monotonic', clock)

        fut: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        nominal = 1.0  # far below RESPONSIVE_WAIT_WALL_CAP / RESPONSIVE_WAIT_STRETCH

        with pytest.raises(pytest.fail.Exception) as excinfo:
            await wait_responsive(
                fut,
                timeout=nominal,
                label='scaled-cap',
                slice_s=SLICE,
                # max_wall_s deliberately NOT passed: the default is under test.
            )
        fut.cancel()

        message = str(excinfo.value)
        expected_cap = RESPONSIVE_WAIT_STRETCH * nominal
        assert _reported(message, 'cap') == pytest.approx(expected_cap)
        # The wait ended at the SCALED cap, nowhere near the flat 90s ceiling.
        assert _reported(message, 'wall') < float(RESPONSIVE_WAIT_WALL_CAP)

    async def test_large_nominal_is_still_bounded_by_the_absolute_ceiling(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = _StarvationClock(STARVE)
        monkeypatch.setattr(_orch_helpers, '_monotonic', clock)

        fut: asyncio.Future[Any] = asyncio.get_running_loop().create_future()

        with pytest.raises(pytest.fail.Exception) as excinfo:
            await wait_responsive(
                fut,
                # A nominal whose stretch would blow past the absolute ceiling.
                timeout=10.0 * RESPONSIVE_WAIT_WALL_CAP,
                label='ceiling-bounded',
                slice_s=SLICE,
            )
        fut.cancel()

        message = str(excinfo.value)
        assert _reported(message, 'cap') == pytest.approx(float(RESPONSIVE_WAIT_WALL_CAP))

    async def test_explicit_max_wall_s_still_wins_over_the_scaled_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = _StarvationClock(STARVE)
        monkeypatch.setattr(_orch_helpers, '_monotonic', clock)

        fut: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        explicit = 7.5

        with pytest.raises(pytest.fail.Exception) as excinfo:
            await wait_responsive(
                fut,
                timeout=1000.0,
                label='explicit-cap',
                slice_s=SLICE,
                max_wall_s=explicit,
            )
        fut.cancel()

        assert _reported(str(excinfo.value), 'cap') == pytest.approx(explicit)


@pytest.mark.asyncio
class TestWaitResponsiveRealHangStillReportsRed:
    """(b) A responsive loop + a never-resolving future must still fail LOUDLY.

    This is the property that keeps the fix honest: charging in responsive time
    stretches the deadline only when the loop was actually starved. When the
    loop IS responsive, every slice is charged in full and the guard trips at
    ~``timeout``, so a genuine merge-pipeline hang cannot be masked into an
    unbounded wait.
    """

    async def test_never_resolving_future_fails_naming_the_label(self) -> None:
        fut: asyncio.Future[Any] = asyncio.get_running_loop().create_future()

        started = time.monotonic()
        with pytest.raises(pytest.fail.Exception) as excinfo:
            await wait_responsive(
                fut,
                timeout=BUDGET,
                label='late3-a MergeOutcome',
                slice_s=SLICE,
                max_wall_s=WALL_CAP,
            )
        elapsed = time.monotonic() - started
        fut.cancel()

        message = str(excinfo.value)
        assert 'late3-a MergeOutcome' in message
        # Gave up on the BUDGET, not on the wall cap -- the charged total
        # reached ~timeout because the loop was responsive throughout.
        assert _reported(message, 'charged') >= BUDGET
        assert _reported(message, 'wall') < WALL_CAP
        # A real hang is reported promptly, never left to run to the cap.
        assert elapsed < WALL_CAP

    async def test_give_up_cancels_a_wrapped_coroutine_rather_than_leaking_it(
        self,
    ) -> None:
        cancelled = asyncio.Event()

        async def _never() -> None:
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        with pytest.raises(pytest.fail.Exception):
            await wait_responsive(
                _never(),
                timeout=BUDGET,
                label='leak-check',
                slice_s=SLICE,
                max_wall_s=WALL_CAP,
            )

        await asyncio.sleep(0)
        assert cancelled.is_set(), (
            'wait_responsive leaked its wrapper task into pytest-asyncio '
            'teardown instead of cancelling it on give-up'
        )

    async def test_give_up_leaves_a_caller_owned_future_untouched(self) -> None:
        """The mirror image of the test above, and an equally load-bearing
        promise: a Future the CALLER created must survive a give-up intact.

        Every late-arrival call site passes ``req_a.result`` -- a Future owned
        by the MergeRequest, not by this helper -- and several of them go on to
        await it again or assert on it after an earlier wait. ``owned = task is
        not aw`` is what keeps the cancel confined to wrapper tasks; without
        this assertion a refactor of that line could start cancelling caller
        futures and the suite would stay green (the existing hang test merely
        calls ``fut.cancel()`` afterwards, which succeeds either way).
        """
        fut: asyncio.Future[Any] = asyncio.get_running_loop().create_future()

        with pytest.raises(pytest.fail.Exception):
            await wait_responsive(
                fut,
                timeout=BUDGET,
                label='caller-owned',
                slice_s=SLICE,
                max_wall_s=WALL_CAP,
            )

        assert not fut.cancelled(), (
            'wait_responsive cancelled a caller-owned Future on give-up; the '
            'late-arrival tests re-await and assert on req.result after an '
            'earlier wait, so cancelling it turns one red wait into a '
            'cascade of unrelated CancelledErrors'
        )
        assert not fut.done(), 'caller-owned Future must be left pending'
        fut.cancel()


class _SentinelError(RuntimeError):
    """Distinct type so the propagation assertions cannot pass by accident."""


@pytest.mark.asyncio
class TestWaitResponsiveExceptionPropagation:
    """A failure of the awaited work must surface AS ITSELF, not as a give-up.

    ``return task.result()`` re-raises whatever the awaitable failed with, and
    every late-arrival call site depends on it: ``MergeRequest.result`` can be
    resolved with an exception rather than a MergeOutcome, and a test that saw
    ``pytest.fail('... wait_responsive gave up ...')`` instead of the real
    exception would send a reader hunting a phantom timeout. Nothing pinned
    this before the amendment pass.
    """

    async def test_future_resolved_with_an_exception_propagates_it(self) -> None:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Any] = loop.create_future()
        loop.call_later(
            SLICE * 2,
            lambda: None if fut.done() else fut.set_exception(_SentinelError('boom')),
        )

        with pytest.raises(_SentinelError, match='boom'):
            await wait_responsive(
                fut, timeout=BUDGET, label='exc-future', slice_s=SLICE, max_wall_s=WALL_CAP
            )

    async def test_raising_coroutine_propagates_rather_than_failing_the_wait(
        self,
    ) -> None:
        async def _boom() -> None:
            await asyncio.sleep(SLICE)
            raise _SentinelError('from-coro')

        with pytest.raises(_SentinelError, match='from-coro'):
            await wait_responsive(
                _boom(), timeout=BUDGET, label='exc-coro', slice_s=SLICE, max_wall_s=WALL_CAP
            )


@pytest.mark.asyncio
class TestWaitResponsiveIsLoadIndependent:
    """(c) THE core property: the budget is charged in responsive time.

    With the clock running ``STARVE`` times faster than real time, a wait whose
    APPARENT elapsed wall clock is orders of magnitude beyond ``timeout`` must
    still return its value -- a bare ``asyncio.wait_for(timeout=BUDGET)`` on
    the same clock would have raised TimeoutError long before.
    """

    async def test_apparent_wall_clock_far_beyond_budget_still_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = _StarvationClock(STARVE)
        monkeypatch.setattr(_orch_helpers, '_monotonic', clock)

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[str] = loop.create_future()
        loop.call_later(SLICE * 5, lambda: None if fut.done() else fut.set_result('late'))

        apparent_start = clock()
        got = await wait_responsive(
            fut,
            timeout=BUDGET,
            label='starved-but-correct',
            slice_s=SLICE,
            max_wall_s=1_000_000.0,
        )
        apparent_elapsed = clock() - apparent_start

        assert got == 'late'
        # The discriminating assertion: apparent wall clock blew far past the
        # budget, yet the wait survived -- so the budget was NOT charged in
        # wall-clock terms.
        assert apparent_elapsed > BUDGET * 10, (
            f'test did not actually simulate starvation: apparent elapsed '
            f'{apparent_elapsed}s vs budget {BUDGET}s'
        )


@pytest.mark.asyncio
class TestWaitResponsiveHonoursWallCap:
    """(d) Permanent starvation + a never-resolving future still terminates.

    Without this cap a stretched wait could outlive its class-level
    ``@pytest.mark.timeout``; under ``timeout_method = "thread"`` with
    ``--max-worker-restart=0`` that is an ``os._exit()`` of the xdist worker,
    i.e. a worker death rather than a clean failure.
    """

    async def test_permanently_starved_hang_gives_up_at_the_wall_cap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = _StarvationClock(STARVE)
        monkeypatch.setattr(_orch_helpers, '_monotonic', clock)

        fut: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        apparent_cap = 20.0

        started = time.monotonic()
        with pytest.raises(pytest.fail.Exception) as excinfo:
            await wait_responsive(
                fut,
                # Budget deliberately unreachable so ONLY the wall cap can
                # end this wait.
                timeout=1000.0,
                label='starved-and-hung',
                slice_s=SLICE,
                max_wall_s=apparent_cap,
            )
        elapsed = time.monotonic() - started
        fut.cancel()

        message = str(excinfo.value)
        assert 'starved-and-hung' in message
        assert _reported(message, 'wall') >= apparent_cap
        assert _reported(message, 'charged') < 1000.0
        # Real time spent is a small multiple of a slice: the cap is reached in
        # APPARENT time, so this terminates regardless of how starved the host
        # actually is.
        assert elapsed < WALL_CAP
