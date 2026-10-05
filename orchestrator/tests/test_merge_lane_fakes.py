"""The shared merge-lane fakes' own contract (``_merge_lane_fakes``).

Dozens of merge-lane test files lean on these doubles, so what they record and
when is pinned here, through their public surface only.
"""
from __future__ import annotations

import asyncio
import dataclasses
from pathlib import Path

import pytest
from _merge_lane_fakes import FakeVerifier, ScopedVerifyCall, hangs_until
from _orch_helpers import wait_responsive


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
