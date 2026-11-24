"""DurableWriteQueue parking (task 10922): a write the provider refuses for a
reason that clears only when a human acts is PARKED, not dead-lettered.

A parked row keeps its attempt count, never reaches 'dead', and is re-probed
at the queue's slowest retry cadence. The queue knows nothing of billing: an
injected ParkPolicy names which failures park.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from unittest.mock import AsyncMock

import pytest
from _fm_polling import poll_until, poll_until_dead
from _queued_write_fixtures import queued
from shared.testing_fake_clock import FakeClock

from fused_memory.services.durable_queue import DurableWriteQueue, ParkDecision, ParkedWriteEvent
from fused_memory.services.queued_writes import QueueCallback, QueuedWrite

_MAX_ATTEMPTS = 3
_PROBE_AHEAD_SECONDS = 600.0
_PROBE_BEHIND_SECONDS = 3600.0


class _Exhausted(Exception):
    """Stands in for a refusal that only a human can clear."""


@dataclass
class _RecordingParkPolicy:
    parked: list[ParkedWriteEvent] = field(default_factory=list)
    cleared: int = 0
    raise_on_parked: bool = False
    raise_on_cleared: bool = False
    retry_at: Callable[[], float | None] = lambda: None
    observe: Callable[[ParkedWriteEvent], None] = lambda event: None

    def park_decision(self, exc: BaseException) -> ParkDecision | None:
        if not isinstance(exc, _Exhausted):
            return None
        return ParkDecision('test-park', retry_at=self.retry_at())

    async def on_parked(self, event: ParkedWriteEvent) -> None:
        self.parked.append(event)
        self.observe(event)
        if self.raise_on_parked:
            raise RuntimeError('on_parked hook failed')

    async def on_cleared(self) -> None:
        self.cleared += 1
        if self.raise_on_cleared:
            raise RuntimeError('on_cleared hook failed')


def _queue(tmp_path, execute_write, **overrides) -> DurableWriteQueue:
    kwargs = {
        'data_dir': tmp_path / 'queue',
        'execute_write': execute_write,
        'workers_per_group': 1,
        'semaphore_limit': 5,
        'max_attempts': _MAX_ATTEMPTS,
        'retry_base_seconds': 0.01,
        'retry_max_delay_seconds': 0.3,
        'write_timeout_seconds': 2.0,
    }
    kwargs.update(overrides)
    return DurableWriteQueue(**kwargs)


def _write(content: str, group_id: str = 'proj1') -> QueuedWrite:
    return queued(
        'add_episode', {'content': content, 'group_id': group_id, 'name': 'ep'},
        group_id=group_id,
    )


def _row(tmp_path, item_id: int) -> dict:
    db_path = tmp_path / 'queue' / 'write_queue.db'
    with contextlib.closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute('SELECT * FROM write_queue WHERE id = ?', (item_id,)).fetchone()
    assert row is not None
    return dict(row)


def _status_is(tmp_path, item_id: int, status: str):
    return lambda: _row(tmp_path, item_id)['status'] == status


class TestParking:
    @pytest.mark.asyncio
    async def test_a_parkable_failure_parks_and_never_dies(self, tmp_path):
        policy = _RecordingParkPolicy()
        on_dead_letter = AsyncMock()
        execute = AsyncMock(side_effect=_Exhausted('no credits remaining'))
        q = _queue(tmp_path, execute, park_policy=policy, on_dead_letter=on_dead_letter)
        await q.initialize()
        try:
            item_id = await q.enqueue(_write('parks'))
            await poll_until(
                lambda: execute.await_count >= _MAX_ATTEMPTS + 3, timeout=20.0,
                message='a parked row was not re-probed past max_attempts',
            )
            await poll_until(_status_is(tmp_path, item_id, 'parked'), timeout=5.0)

            row = _row(tmp_path, item_id)
            assert row['attempts'] == 0
            assert row['error'].startswith('parked (test-park)')
            on_dead_letter.assert_not_awaited()
            assert (await q.get_stats())['counts'].get('dead', 0) == 0

            event = policy.parked[0]
            assert (event.item_id, event.group_id, event.operation, event.reason) == (
                item_id, 'proj1', 'add_episode', 'test-park',
            )
            assert isinstance(event.exc, _Exhausted)
            for event_field in dataclasses.fields(event):
                with pytest.raises(dataclasses.FrozenInstanceError):
                    setattr(event, event_field.name, 'other')
        finally:
            await q.close()

    @pytest.mark.asyncio
    async def test_parking_rearms_every_parked_row_to_one_horizon(self, tmp_path):
        policy = _RecordingParkPolicy()
        execute = AsyncMock(side_effect=_Exhausted('no credits remaining'))
        q = _queue(tmp_path, execute, park_policy=policy, retry_max_delay_seconds=30.0)
        await q.initialize()
        try:
            first = await q.enqueue(_write('first', group_id='proj1'))
            await poll_until(_status_is(tmp_path, first, 'parked'), timeout=10.0)
            first_horizon = _row(tmp_path, first)['next_retry_at']

            second = await q.enqueue(_write('second', group_id='proj2'))
            await poll_until(_status_is(tmp_path, second, 'parked'), timeout=10.0)

            assert _row(tmp_path, first)['next_retry_at'] == _row(tmp_path, second)['next_retry_at']
            assert _row(tmp_path, first)['next_retry_at'] > first_horizon
        finally:
            await q.close()

    @pytest.mark.asyncio
    async def test_parked_rows_are_counted_and_age_as_pending(self, tmp_path):
        policy = _RecordingParkPolicy()
        execute = AsyncMock(side_effect=_Exhausted('no credits remaining'))
        q = _queue(tmp_path, execute, park_policy=policy, retry_max_delay_seconds=30.0)
        await q.initialize()
        try:
            item_id = await q.enqueue(_write('parks'))
            await poll_until(_status_is(tmp_path, item_id, 'parked'), timeout=10.0)

            stats = await q.get_stats()
            assert stats['counts'].get('parked') == 1
            assert stats['oldest_pending_age_seconds'] is not None
        finally:
            await q.close()

    @pytest.mark.asyncio
    async def test_a_raising_on_parked_hook_does_not_stop_the_worker(self, tmp_path):
        policy = _RecordingParkPolicy(raise_on_parked=True)

        async def execute(value: QueuedWrite) -> dict:
            if getattr(value, 'content', '') == 'parks':
                raise _Exhausted('no credits remaining')
            return {'episode_uuid': 'ep-1'}

        q = _queue(tmp_path, execute, park_policy=policy, retry_max_delay_seconds=30.0)
        await q.initialize()
        try:
            parked = await q.enqueue(_write('parks'))
            healthy = await q.enqueue(_write('healthy'))

            await poll_until(_status_is(tmp_path, healthy, 'completed'), timeout=10.0)
            assert _row(tmp_path, parked)['status'] == 'parked'
            assert len(policy.parked) == 1
        finally:
            await q.close()


class TestTheProbeTimeAParkNames:
    @pytest.mark.asyncio
    async def test_a_named_probe_time_becomes_every_parked_rows_horizon(self, tmp_path):
        probe_at = time.time() + _PROBE_AHEAD_SECONDS
        policy = _RecordingParkPolicy()
        execute = AsyncMock(side_effect=_Exhausted('capped'))
        q = _queue(tmp_path, execute, park_policy=policy, retry_max_delay_seconds=30.0)
        await q.initialize()
        try:
            first = await q.enqueue(_write('first', group_id='proj1'))
            await poll_until(_status_is(tmp_path, first, 'parked'))
            policy.retry_at = lambda: probe_at

            second = await q.enqueue(_write('second', group_id='proj2'))
            await poll_until(_status_is(tmp_path, second, 'parked'))

            row = _row(tmp_path, second)
            assert row['next_retry_at'] == probe_at
            assert row['attempts'] == 0
            assert _row(tmp_path, first)['next_retry_at'] == probe_at
        finally:
            await q.close()

    @pytest.mark.asyncio
    async def test_a_probe_time_in_the_past_is_clamped_to_now(self, tmp_path):
        retry_ats = iter([time.time() - _PROBE_BEHIND_SECONDS])
        horizons: list[float] = []
        policy = _RecordingParkPolicy(
            retry_at=lambda: next(retry_ats, time.time() + _PROBE_AHEAD_SECONDS),
            observe=lambda event: horizons.append(_row(tmp_path, event.item_id)['next_retry_at']),
        )
        execute = AsyncMock(side_effect=_Exhausted('capped'))
        q = _queue(tmp_path, execute, park_policy=policy, retry_max_delay_seconds=30.0)
        await q.initialize()
        try:
            before = time.time()
            item_id = await q.enqueue(_write('clamped'))
            await poll_until(
                lambda: execute.await_count >= 2,
                message='a row parked with a past probe time was not claimable at once',
            )
            await poll_until(lambda: len(horizons) >= 2)

            assert before <= horizons[0] <= time.time()
            assert _row(tmp_path, item_id)['status'] == 'parked'
            assert _row(tmp_path, item_id)['attempts'] == 0
        finally:
            await q.close()


class TestTheQueuesClock:
    @pytest.mark.asyncio
    async def test_parking_and_landing_follow_the_injected_clock(self, tmp_path):
        start = 1_800_000_000.0
        clock = FakeClock(start=start)
        policy = _RecordingParkPolicy(retry_at=lambda: clock() + _PROBE_AHEAD_SECONDS)
        execute = AsyncMock(side_effect=[_Exhausted('capped'), None])
        q = _queue(tmp_path, execute, park_policy=policy, clock=clock)
        await q.initialize()
        try:
            item_id = await q.enqueue(_write('on the fake clock'))
            await poll_until(_status_is(tmp_path, item_id, 'parked'))

            row = _row(tmp_path, item_id)
            assert row['created_at'] == start
            assert row['next_retry_at'] == start + _PROBE_AHEAD_SECONDS

            clock.advance(_PROBE_AHEAD_SECONDS)
            await poll_until(_status_is(tmp_path, item_id, 'completed'))

            assert _row(tmp_path, item_id)['completed_at'] == start + _PROBE_AHEAD_SECONDS
            await poll_until(
                lambda: policy.cleared == 1,
                message='on_cleared runs after the completion commits and was never reported',
            )
        finally:
            await q.close()


class TestWhatDoesNotPark:
    @pytest.mark.asyncio
    async def test_a_failure_the_policy_does_not_name_dies_as_today(self, tmp_path):
        policy = _RecordingParkPolicy()
        on_dead_letter = AsyncMock()
        execute = AsyncMock(side_effect=RuntimeError('boom'))
        q = _queue(tmp_path, execute, park_policy=policy, on_dead_letter=on_dead_letter)
        await q.initialize()
        try:
            await q.enqueue(_write('dies'))
            await poll_until_dead(q, group_id='proj1', expected_dead=1)
            await poll_until(lambda: on_dead_letter.await_count >= 1, timeout=10.0)

            assert execute.await_count == _MAX_ATTEMPTS
            assert policy.parked == []
        finally:
            await q.close()

    @pytest.mark.asyncio
    async def test_a_post_execute_failure_is_never_parked(self, tmp_path):
        """A retry re-runs the write; parking one that already LANDED would
        duplicate it on every probe."""
        policy = _RecordingParkPolicy()
        execute = AsyncMock(return_value={'episode_uuid': 'ep-1'})

        async def exhausted_callback(_value, _result):
            raise _Exhausted('no credits remaining')

        q = _queue(tmp_path, execute, park_policy=policy)
        q.register_callback(QueueCallback.DUAL_WRITE_EPISODE, exhausted_callback)
        await q.initialize()
        try:
            await q.enqueue(_write('landed'), callback=QueueCallback.DUAL_WRITE_EPISODE)
            await poll_until_dead(q, group_id='proj1', expected_dead=1)

            assert execute.await_count == _MAX_ATTEMPTS
            assert policy.parked == []
        finally:
            await q.close()

    @pytest.mark.asyncio
    async def test_without_a_policy_the_same_failure_dies_as_today(self, tmp_path):
        execute = AsyncMock(side_effect=_Exhausted('no credits remaining'))
        q = _queue(tmp_path, execute)
        await q.initialize()
        try:
            await q.enqueue(_write('dies'))
            await poll_until_dead(q, group_id='proj1', expected_dead=1)

            assert execute.await_count == _MAX_ATTEMPTS
        finally:
            await q.close()


def _make_due(tmp_path, item_id: int) -> None:
    """Let *item_id*'s park period elapse, and only its own."""
    db_path = tmp_path / 'queue' / 'write_queue.db'
    with contextlib.closing(sqlite3.connect(db_path)) as conn:
        conn.execute('UPDATE write_queue SET next_retry_at = 0 WHERE id = ?', (item_id,))
        conn.commit()


class _Switchable:
    """An execute_write that refuses until switched on, recording each landing."""

    def __init__(self) -> None:
        self.succeeding = False
        self.landed: list[str] = []

    async def __call__(self, value: QueuedWrite) -> dict:
        if not self.succeeding:
            raise _Exhausted('no credits remaining')
        self.landed.append(value.written_text)
        return {'episode_uuid': 'ep-1'}


_REFUSE_RELEASE = (
    'CREATE TRIGGER refuse_release BEFORE UPDATE OF status ON write_queue '
    "WHEN OLD.status = 'parked' AND NEW.status = 'pending' "
    "BEGIN SELECT RAISE(ABORT, 'release refused'); END"
)


def _on_queue_db(tmp_path, statement: str) -> None:
    db_path = tmp_path / 'queue' / 'write_queue.db'
    with contextlib.closing(sqlite3.connect(db_path)) as conn:
        conn.execute(statement)
        conn.commit()


async def _park_three(tmp_path, q: DurableWriteQueue) -> tuple[int, int, int]:
    """Park one row in proj1 to probe first, then one more in each of two groups."""
    first = await q.enqueue(_write('first', group_id='proj1'))
    await poll_until(_status_is(tmp_path, first, 'parked'), timeout=10.0)
    same_group = await q.enqueue(_write('same group', group_id='proj1'))
    other_group = await q.enqueue(_write('other group', group_id='proj2'))
    for item_id in (same_group, other_group):
        await poll_until(_status_is(tmp_path, item_id, 'parked'), timeout=10.0)
    return first, same_group, other_group


class TestUnparking:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('on_cleared_raises', [False, True], ids=['hook-ok', 'hook-raises'])
    async def test_one_parked_write_landing_drains_every_parked_row(
        self, tmp_path, on_cleared_raises,
    ):
        """The probe that lands proves the cause is gone, so nothing else
        waits out another period. A long cadence makes the drain observable:
        only the first row's period is let elapse."""
        policy = _RecordingParkPolicy(raise_on_cleared=on_cleared_raises)
        execute = _Switchable()
        q = _queue(tmp_path, execute, park_policy=policy, retry_max_delay_seconds=30.0)
        await q.initialize()
        try:
            first, same_group, other_group = await _park_three(tmp_path, q)

            execute.succeeding = True
            _make_due(tmp_path, first)

            for item_id in (first, same_group, other_group):
                await poll_until(_status_is(tmp_path, item_id, 'completed'), timeout=10.0)
            assert [_row(tmp_path, i)['attempts'] for i in (same_group, other_group)] == [1, 1]
            assert policy.cleared == 1
        finally:
            await q.close()

    @pytest.mark.asyncio
    async def test_a_failed_release_never_reruns_the_landed_write(self, tmp_path):
        """The landed probe is settled before the release is attempted, so a
        release the database refuses only delays the other parked rows to
        their own horizon; it never re-runs the write that landed."""
        policy = _RecordingParkPolicy()
        execute = _Switchable()
        q = _queue(tmp_path, execute, park_policy=policy, retry_max_delay_seconds=30.0)
        await q.initialize()
        try:
            first, same_group, other_group = await _park_three(tmp_path, q)
            _on_queue_db(tmp_path, _REFUSE_RELEASE)

            execute.succeeding = True
            _make_due(tmp_path, first)
            await poll_until(_status_is(tmp_path, first, 'completed'), timeout=10.0)
            # ~10x the retry backoff: a wrongly rescheduled `first` is due by now,
            # and _claim_next takes it ahead of the later sentinel.
            await asyncio.sleep(0.2)
            sentinel = await q.enqueue(_write('sentinel', group_id='proj1'))
            await poll_until(_status_is(tmp_path, sentinel, 'completed'), timeout=10.0)

            assert execute.landed.count('first') == 1
            first_row = _row(tmp_path, first)
            assert (first_row['status'], first_row['attempts']) == ('completed', 1)
            assert 'release refused' not in (first_row['error'] or '')
            for item_id in (same_group, other_group):
                assert _row(tmp_path, item_id)['status'] == 'parked'
            assert policy.cleared == 1

            _on_queue_db(tmp_path, 'DROP TRIGGER refuse_release')
            _make_due(tmp_path, same_group)
            for item_id in (same_group, other_group):
                await poll_until(_status_is(tmp_path, item_id, 'completed'), timeout=10.0)
            assert [execute.landed.count(c) for c in ('same group', 'other group')] == [1, 1]
        finally:
            await q.close()

    @pytest.mark.asyncio
    async def test_a_never_parked_write_landing_clears_nothing(self, tmp_path):
        policy = _RecordingParkPolicy()
        q = _queue(tmp_path, AsyncMock(return_value={'episode_uuid': 'ep-1'}), park_policy=policy)
        await q.initialize()
        try:
            item_id = await q.enqueue(_write('healthy'))
            await poll_until(_status_is(tmp_path, item_id, 'completed'), timeout=10.0)

            assert policy.cleared == 0
        finally:
            await q.close()

    @pytest.mark.asyncio
    async def test_parked_rows_survive_a_restart_and_drain(self, tmp_path):
        execute = _Switchable()
        q = _queue(tmp_path, execute, park_policy=_RecordingParkPolicy())
        await q.initialize()
        try:
            parked = [
                await q.enqueue(_write('a', group_id='proj1')),
                await q.enqueue(_write('b', group_id='proj2')),
            ]
            for item_id in parked:
                await poll_until(_status_is(tmp_path, item_id, 'parked'), timeout=10.0)
        finally:
            await q.close()

        execute.succeeding = True
        restarted = _queue(tmp_path, execute, park_policy=_RecordingParkPolicy())
        await restarted.initialize()
        try:
            for item_id in parked:
                await poll_until(_status_is(tmp_path, item_id, 'completed'), timeout=10.0)
        finally:
            await restarted.close()

    @pytest.mark.asyncio
    async def test_a_dead_letter_from_before_parking_stays_replayable(self, tmp_path):
        failing = AsyncMock(side_effect=RuntimeError('boom'))
        q = _queue(tmp_path, failing, park_policy=_RecordingParkPolicy())
        await q.initialize()
        try:
            item_id = await q.enqueue(_write('dies'))
            await poll_until_dead(q, group_id='proj1', expected_dead=1)

            failing.side_effect = None
            failing.return_value = {'episode_uuid': 'ep-1'}
            assert await q.replay_dead() == 1
            await poll_until(_status_is(tmp_path, item_id, 'completed'), timeout=10.0)
        finally:
            await q.close()
