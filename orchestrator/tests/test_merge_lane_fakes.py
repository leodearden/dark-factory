"""The shared merge-lane fakes' own contract (``_merge_lane_fakes``).

Dozens of merge-lane test files lean on these doubles, so what they record and
when is pinned here, through their public surface only.
"""
from __future__ import annotations

import asyncio
import dataclasses
import time
from pathlib import Path

import pytest
from _merge_lane_fakes import (
    FakeClock,
    FakeVerifier,
    ScopedVerifyCall,
    VerifyScript,
    fails,
    hangs_until,
    passes,
    raises,
)
from _orch_helpers import wait_responsive

from orchestrator.verify import VerifyResult


async def _verify(verifier: FakeVerifier, task_id: str) -> VerifyResult:
    return await verifier.run_scoped(Path(f'wt-{task_id}'), None, [], task_id=task_id)


class TestFakeVerifierRecordsEachScopedVerify:
    @pytest.mark.asyncio
    async def test_each_call_records_task_id_worktree_and_module_configs_in_order(self):
        verifier = FakeVerifier()
        m1, m2 = object(), object()

        await verifier.run_scoped(Path('wt-a'), None, [m1, m2], task_id='a')
        await verifier.run_scoped(Path('wt-b'), None, [], task_id='b')

        assert verifier.verify_calls == [
            ScopedVerifyCall(task_id='a', worktree=Path('wt-a'), module_configs=(m1, m2)),
            ScopedVerifyCall(task_id='b', worktree=Path('wt-b'), module_configs=()),
        ]
        assert verifier.verified == ['a', 'b']

    @pytest.mark.asyncio
    async def test_the_record_is_an_immutable_snapshot(self):
        verifier = FakeVerifier()
        m1, m2 = object(), object()
        handed = [m1]

        await verifier.run_scoped(Path('wt'), None, handed, task_id='a')
        handed.append(m2)

        record = verifier.verify_calls[0]
        assert isinstance(record.module_configs, tuple)
        assert record.module_configs == (m1,)
        with pytest.raises(dataclasses.FrozenInstanceError):
            record.module_configs = (m1, m2)  # type: ignore[misc]

    @pytest.mark.asyncio
    async def test_the_record_exists_once_await_entry_returns(self):
        release = asyncio.Event()
        verifier = FakeVerifier(scripts={'a': hangs_until(release)})
        parked = asyncio.create_task(
            verifier.run_scoped(Path('wt-parked'), None, [], task_id='a'),
        )
        try:
            await wait_responsive(
                verifier.await_entry(1), label='first scoped verify entered',
            )
            assert not parked.done()
            assert verifier.verify_calls[0].worktree == Path('wt-parked')

            release.set()
            result = await wait_responsive(parked, label='released scoped verify returned')
            assert result.passed
        finally:
            parked.cancel()
            await asyncio.gather(parked, return_exceptions=True)


class TestFakeVerifierOrderedScripts:
    @pytest.mark.asyncio
    async def test_sequence_is_consumed_in_call_order_then_the_standing_verdict_applies(self):
        verifier = FakeVerifier(
            default=fails(category='x', summary='standing'),
            sequence=[passes('first'), fails(category='x', summary='second')],
        )

        summaries = [(await _verify(verifier, 't')).summary for _ in range(3)]

        assert summaries == ['first', 'second', 'standing']

    @pytest.mark.asyncio
    async def test_sequence_takes_precedence_over_task_scripts_until_exhausted(self):
        verifier = FakeVerifier(
            scripts={'t': fails(category='x', summary='keyed')},
            sequence=[passes('ordered')],
        )

        assert (await _verify(verifier, 't')).summary == 'ordered'
        assert (await _verify(verifier, 't')).summary == 'keyed'
        assert (await _verify(verifier, 'u')).summary == 'fake verify passed'

    @pytest.mark.asyncio
    async def test_the_sequence_is_copied_at_construction(self):
        ordered = [passes('ordered')]
        verifier = FakeVerifier(sequence=ordered)
        ordered[:] = [fails(category='x', summary='swapped'), fails(category='x', summary='added')]

        assert (await _verify(verifier, 't')).summary == 'ordered'
        assert (await _verify(verifier, 't')).summary == 'fake verify passed'

    @pytest.mark.asyncio
    async def test_an_ordered_script_can_raise(self):
        verifier = FakeVerifier(sequence=[raises(RuntimeError('boom'))])

        with pytest.raises(RuntimeError, match='boom'):
            await _verify(verifier, 't')
        assert (await _verify(verifier, 't')).passed


class TestVerifyScriptEntered:
    @pytest.mark.asyncio
    async def test_hangs_until_sets_entered_before_waiting_for_release(self):
        release, entered = asyncio.Event(), asyncio.Event()
        verifier = FakeVerifier(sequence=[hangs_until(release, entered=entered)])
        parked = asyncio.create_task(_verify(verifier, 't'))
        try:
            await wait_responsive(entered.wait(), label='hung verify arrived')
            assert not parked.done()

            release.set()
            result = await wait_responsive(parked, label='released verify returned')
            assert result.summary == 'fake verify passed'
        finally:
            parked.cancel()
            await asyncio.gather(parked, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_entered_is_set_for_a_script_that_does_not_hang(self):
        entered = asyncio.Event()
        verifier = FakeVerifier(scripts={'t': dataclasses.replace(passes(), entered=entered)})

        await _verify(verifier, 't')

        assert entered.is_set()

    @pytest.mark.asyncio
    async def test_first_call_gate_as_a_constructor_call(self):
        release, entered = asyncio.Event(), asyncio.Event()
        verifier = FakeVerifier(sequence=[hangs_until(release, entered=entered)])
        first = asyncio.create_task(_verify(verifier, 'first'))
        try:
            await wait_responsive(entered.wait(), label='gated first verify arrived')

            assert (await _verify(verifier, 'second')).passed
            assert not first.done()

            release.set()
            assert (await wait_responsive(first, label='gated first verify returned')).passed
            assert verifier.verified == ['first', 'second']
        finally:
            first.cancel()
            await asyncio.gather(first, return_exceptions=True)


class TestNextScriptSeam:
    @pytest.mark.asyncio
    async def test_a_subclass_overriding_next_script_keeps_the_base_recording(self):
        class _PickingVerifier(FakeVerifier):
            def next_script(self, task_id: str | None) -> VerifyScript:
                if task_id == 'p':
                    return fails(category='x', summary='picked')
                return super().next_script(task_id)

        verifier = _PickingVerifier()

        assert (await _verify(verifier, 'p')).summary == 'picked'
        assert (await _verify(verifier, 'q')).summary == 'fake verify passed'
        assert [call.task_id for call in verifier.verify_calls] == ['p', 'q']
        assert verifier.verified == ['p', 'q']


class TestFakeClockSleeps:
    """``sleep`` on the fake every merge-lane test injects.

    A started lane's ``_heartbeat_loop`` and ``_reprobe_loop`` sleep on this
    clock, so a bare-yield sleep would spin them at event-loop speed.
    """

    @pytest.mark.asyncio
    async def test_one_sleep_really_waits_so_a_periodic_loop_cannot_spin(self) -> None:
        clock = FakeClock()
        started = time.monotonic()
        await clock.sleep(30.0)
        # A floor well under the fake's own cap, not the cap itself: what is
        # being pinned is that a sleep costs REAL time at all. A bare yield
        # takes about 0.00002s.
        assert time.monotonic() - started >= 0.01

    @pytest.mark.asyncio
    async def test_a_sleep_waits_no_longer_than_the_cap_not_the_requested_period(self) -> None:
        await asyncio.wait_for(FakeClock().sleep(3600.0), timeout=5.0)

    @pytest.mark.asyncio
    async def test_a_sleep_shorter_than_the_cap_waits_only_the_requested_period(self) -> None:
        await asyncio.wait_for(FakeClock(sleep_cap=60.0).sleep(0.01), timeout=5.0)

    @pytest.mark.asyncio
    async def test_a_sleep_advances_both_readings_by_the_full_request_and_records_it(self) -> None:
        clock = FakeClock()
        before_now, before_mono = clock.now(), clock.monotonic()

        await clock.sleep(30.0)
        await clock.sleep(120.0)

        assert clock.now() - before_now == 150.0
        assert clock.monotonic() - before_mono == 150.0
        assert clock.sleeps == [30.0, 120.0]

    @pytest.mark.asyncio
    async def test_sleep_cap_zero_is_a_bare_yield_for_a_production_sized_poll_count(self) -> None:
        clock = FakeClock(sleep_cap=0.0)

        async def _poll_a_thousand_times() -> None:
            for _ in range(1000):
                await clock.sleep(0.05)

        # At the default cap each of these polls would be a real wait.
        await asyncio.wait_for(_poll_a_thousand_times(), timeout=10.0)
        assert len(clock.sleeps) == 1000

    @pytest.mark.asyncio
    async def test_sleep_cap_leaves_wait_for_any_on_wait_cap(self) -> None:
        clock = FakeClock(sleep_cap=0.0)
        forever = asyncio.create_task(asyncio.Event().wait())
        try:
            started = time.monotonic()
            assert await clock.wait_for_any({forever}, timeout=600.0) == set()
            assert time.monotonic() - started >= 0.01
        finally:
            forever.cancel()
            await asyncio.gather(forever, return_exceptions=True)
