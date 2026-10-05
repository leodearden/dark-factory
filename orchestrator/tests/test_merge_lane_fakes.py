"""The shared merge-lane fakes' own contract (``_merge_lane_fakes``).

Dozens of merge-lane test files lean on these doubles, so what they record and
when is pinned here, through their public surface only.
"""
from __future__ import annotations

import asyncio
import dataclasses
from pathlib import Path

import pytest
from _merge_lane_fakes import (
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
    async def test_the_recorded_module_configs_are_a_snapshot(self):
        verifier = FakeVerifier()
        m1, m2 = object(), object()
        handed = [m1]

        await verifier.run_scoped(Path('wt'), None, handed, task_id='a')
        handed.append(m2)

        recorded = verifier.verify_calls[0].module_configs
        assert isinstance(recorded, tuple)
        assert recorded == (m1,)

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

    def test_a_frozen_record(self):
        record = ScopedVerifyCall(task_id='a', worktree=Path('wt'), module_configs=())

        with pytest.raises(dataclasses.FrozenInstanceError):
            record.task_id = 'b'  # type: ignore[misc]


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
