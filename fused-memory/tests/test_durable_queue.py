"""Tests for the durable write queue — real SQLite, mocked Graphiti."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from fused_memory.services.durable_queue import DurableWriteQueue


@pytest.fixture
def mock_execute():
    """A mock execute_write function simulating GraphitiBackend.add_episode."""
    fn = AsyncMock(return_value={'episode_uuid': 'ep-001'})
    return fn


@pytest_asyncio.fixture
async def queue(tmp_path, mock_execute):
    """A DurableWriteQueue wired to a temp SQLite DB with fast settings."""
    q = DurableWriteQueue(
        data_dir=tmp_path / 'queue',
        execute_write=mock_execute,
        workers_per_group=3,
        semaphore_limit=5,
        max_attempts=3,
        retry_base_seconds=0.05,
        write_timeout_seconds=2.0,
    )
    await q.initialize()
    yield q
    await q.close()


class TestEnqueue:
    @pytest.mark.asyncio
    async def test_enqueue_returns_id(self, queue):
        item_id = await queue.enqueue(
            group_id='proj1', operation='add_episode',
            payload={'content': 'hello', 'group_id': 'proj1', 'name': 'ep'},
        )
        assert isinstance(item_id, int)
        assert item_id > 0

    @pytest.mark.asyncio
    async def test_enqueue_batch(self, queue):
        items = [
            {'group_id': 'proj1', 'operation': 'add_episode',
             'payload': {'content': f'item {i}', 'group_id': 'proj1', 'name': f'ep{i}'}}
            for i in range(5)
        ]
        ids = await queue.enqueue_batch(items)
        assert len(ids) == 5
        assert len(set(ids)) == 5  # all unique


class TestWorkerProcessing:
    @pytest.mark.asyncio
    async def test_item_gets_processed(self, queue, mock_execute):
        await queue.enqueue(
            group_id='proj1', operation='add_episode',
            payload={'content': 'test', 'group_id': 'proj1', 'name': 'ep'},
        )
        # Give worker time to pick up and process
        await asyncio.sleep(0.3)
        mock_execute.assert_called_once()
        stats = await queue.get_stats()
        assert stats['counts'].get('completed', 0) == 1

    @pytest.mark.asyncio
    async def test_callback_invoked(self, queue, mock_execute):
        callback = AsyncMock()
        queue.register_callback('test_cb', callback)

        await queue.enqueue(
            group_id='proj1', operation='add_episode',
            payload={'content': 'test', 'group_id': 'proj1', 'name': 'ep'},
            callback_type='test_cb',
        )
        await asyncio.sleep(0.3)
        callback.assert_called_once()
        # Check callback received (callback_type, result, payload)
        args = callback.call_args[0]
        assert args[0] == 'test_cb'
        assert args[1] == {'episode_uuid': 'ep-001'}


class TestRetry:
    @pytest.mark.asyncio
    async def test_retries_on_failure(self, tmp_path):
        call_count = 0

        async def flaky_execute(op, payload):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError('Graphiti timeout')
            return {'ok': True}

        q = DurableWriteQueue(
            data_dir=tmp_path / 'queue',
            execute_write=flaky_execute,
            workers_per_group=1,
            semaphore_limit=5,
            max_attempts=5,
            retry_base_seconds=0.01,
            write_timeout_seconds=2.0,
        )
        await q.initialize()

        await q.enqueue(
            group_id='proj1', operation='add_episode',
            payload={'content': 'test', 'group_id': 'proj1', 'name': 'ep'},
        )
        # Wait enough for retries (0.5s poll + 0.01s/0.02s backoff)
        await asyncio.sleep(3.0)

        assert call_count == 3
        stats = await q.get_stats()
        assert stats['counts'].get('completed', 0) == 1
        await q.close()


class TestTimeout:
    @pytest.mark.asyncio
    async def test_slow_write_retried(self, tmp_path):
        call_count = 0

        async def slow_execute(op, payload):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                await asyncio.sleep(10)  # will be cancelled by timeout
            return {'ok': True}

        q = DurableWriteQueue(
            data_dir=tmp_path / 'queue',
            execute_write=slow_execute,
            workers_per_group=1,
            semaphore_limit=5,
            max_attempts=3,
            retry_base_seconds=0.01,
            write_timeout_seconds=0.1,  # very short timeout
        )
        await q.initialize()

        await q.enqueue(
            group_id='proj1', operation='add_episode',
            payload={'content': 'test', 'group_id': 'proj1', 'name': 'ep'},
        )
        await asyncio.sleep(1.0)

        assert call_count >= 2  # at least one retry after timeout
        stats = await q.get_stats()
        assert stats['counts'].get('completed', 0) == 1
        await q.close()


class TestDeadLetter:
    @pytest.mark.asyncio
    async def test_exhausted_attempts_dead_lettered(self, tmp_path):
        execute = AsyncMock(side_effect=RuntimeError('always fails'))
        q = DurableWriteQueue(
            data_dir=tmp_path / 'queue',
            execute_write=execute,
            workers_per_group=1,
            semaphore_limit=5,
            max_attempts=2,
            retry_base_seconds=0.01,
            write_timeout_seconds=2.0,
        )
        await q.initialize()

        await q.enqueue(
            group_id='proj1', operation='add_episode',
            payload={'content': 'test', 'group_id': 'proj1', 'name': 'ep'},
        )
        await asyncio.sleep(1.0)

        dead = await q.get_dead_items()
        assert len(dead) == 1
        assert 'always fails' in dead[0]['error']
        assert dead[0]['attempts'] == 2
        await q.close()


class TestReplayDead:
    @pytest.mark.asyncio
    async def test_replay_resets_dead_items(self, tmp_path):
        call_count = 0

        async def eventually_works(op, payload):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError('fail')
            return {'ok': True}

        q = DurableWriteQueue(
            data_dir=tmp_path / 'queue',
            execute_write=eventually_works,
            workers_per_group=1,
            semaphore_limit=5,
            max_attempts=2,
            retry_base_seconds=0.01,
            write_timeout_seconds=2.0,
        )
        await q.initialize()

        await q.enqueue(
            group_id='proj1', operation='add_episode',
            payload={'content': 'test', 'group_id': 'proj1', 'name': 'ep'},
        )
        await asyncio.sleep(1.0)

        # Should be dead-lettered
        dead = await q.get_dead_items()
        assert len(dead) == 1

        # Replay
        count = await q.replay_dead('proj1')
        assert count == 1

        await asyncio.sleep(0.5)

        # Now should be completed (call_count was 2 from failures, 3rd succeeds)
        stats = await q.get_stats()
        assert stats['counts'].get('completed', 0) == 1
        assert call_count == 3
        await q.close()


class TestConcurrency:
    @pytest.mark.asyncio
    async def test_global_semaphore_respected(self, tmp_path):
        max_concurrent = 0
        current = 0
        lock = asyncio.Lock()

        async def tracking_execute(op, payload):
            nonlocal max_concurrent, current
            async with lock:
                current += 1
                if current > max_concurrent:
                    max_concurrent = current
            await asyncio.sleep(0.05)
            async with lock:
                current -= 1
            return {'ok': True}

        q = DurableWriteQueue(
            data_dir=tmp_path / 'queue',
            execute_write=tracking_execute,
            workers_per_group=5,
            semaphore_limit=2,  # only 2 concurrent
            max_attempts=3,
            retry_base_seconds=0.01,
            write_timeout_seconds=2.0,
        )
        await q.initialize()

        # Enqueue 10 items
        for i in range(10):
            await q.enqueue(
                group_id='proj1', operation='add_episode',
                payload={'content': f'item {i}', 'group_id': 'proj1', 'name': f'ep{i}'},
            )

        await asyncio.sleep(2.0)

        assert max_concurrent <= 2
        stats = await q.get_stats()
        assert stats['counts'].get('completed', 0) == 10
        await q.close()


class TestRecovery:
    @pytest.mark.asyncio
    async def test_in_flight_recovered_on_init(self, tmp_path):
        execute = AsyncMock(return_value={'ok': True})
        q = DurableWriteQueue(
            data_dir=tmp_path / 'queue',
            execute_write=execute,
            workers_per_group=1,
            semaphore_limit=5,
            max_attempts=3,
            retry_base_seconds=0.01,
            write_timeout_seconds=2.0,
        )
        await q.initialize()

        # Manually insert an in_flight item (simulating crash)
        import time
        assert q._db is not None
        await q._db.execute(
            'INSERT INTO write_queue '
            '(group_id, operation, payload, status, attempts, max_attempts, '
            ' next_retry_at, created_at) '
            "VALUES (?, ?, ?, 'in_flight', 1, 3, 0, ?)",
            ('proj1', 'add_episode', '{"content":"crashed","group_id":"proj1","name":"ep"}',
             time.time()),
        )
        await q._db.commit()
        await q.close()

        # Re-open — should recover
        q2 = DurableWriteQueue(
            data_dir=tmp_path / 'queue',
            execute_write=execute,
            workers_per_group=1,
            semaphore_limit=5,
            max_attempts=3,
            retry_base_seconds=0.01,
            write_timeout_seconds=2.0,
        )
        await q2.initialize()
        await asyncio.sleep(0.5)

        # The recovered item should have been processed
        execute.assert_called()
        stats = await q2.get_stats()
        assert stats['counts'].get('completed', 0) >= 1
        await q2.close()


class TestMultipleGroups:
    @pytest.mark.asyncio
    async def test_independent_group_processing(self, tmp_path):
        groups_seen: dict[str, int] = {}
        lock = asyncio.Lock()

        async def group_tracking_execute(op, payload):
            gid = payload['group_id']
            async with lock:
                groups_seen[gid] = groups_seen.get(gid, 0) + 1
            await asyncio.sleep(0.02)
            return {'ok': True}

        q = DurableWriteQueue(
            data_dir=tmp_path / 'queue',
            execute_write=group_tracking_execute,
            workers_per_group=2,
            semaphore_limit=10,
            max_attempts=3,
            retry_base_seconds=0.01,
            write_timeout_seconds=2.0,
        )
        await q.initialize()

        for group in ['alpha', 'beta', 'gamma']:
            for i in range(3):
                await q.enqueue(
                    group_id=group, operation='add_episode',
                    payload={'content': f'{group}-{i}', 'group_id': group, 'name': f'ep{i}'},
                )

        await asyncio.sleep(1.5)

        assert groups_seen.get('alpha', 0) == 3
        assert groups_seen.get('beta', 0) == 3
        assert groups_seen.get('gamma', 0) == 3
        await q.close()


class TestCallbackFailure:
    @pytest.mark.asyncio
    async def test_callback_error_doesnt_affect_completion(self, queue, mock_execute):
        failing_callback = AsyncMock(side_effect=ValueError('callback boom'))
        queue.register_callback('bad_cb', failing_callback)

        await queue.enqueue(
            group_id='proj1', operation='add_episode',
            payload={'content': 'test', 'group_id': 'proj1', 'name': 'ep'},
            callback_type='bad_cb',
        )
        await asyncio.sleep(0.3)

        # Item should still be completed despite callback failure
        stats = await queue.get_stats()
        assert stats['counts'].get('completed', 0) == 1


class TestShutdown:
    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, tmp_path):
        execute = AsyncMock(return_value={'ok': True})
        q = DurableWriteQueue(
            data_dir=tmp_path / 'queue',
            execute_write=execute,
            workers_per_group=3,
            semaphore_limit=5,
            max_attempts=3,
            retry_base_seconds=0.01,
            write_timeout_seconds=2.0,
        )
        await q.initialize()

        await q.enqueue(
            group_id='proj1', operation='add_episode',
            payload={'content': 'test', 'group_id': 'proj1', 'name': 'ep'},
        )
        await asyncio.sleep(0.2)

        # Close should not raise
        await q.close()

        # Workers should be cleaned up
        assert len(q._worker_tasks) == 0
        assert q._db is None


class TestStats:
    @pytest.mark.asyncio
    async def test_stats_correct_counts(self, tmp_path):
        call_count = 0

        async def mixed_execute(op, payload):
            nonlocal call_count
            call_count += 1
            if payload.get('fail'):
                raise RuntimeError('nope')
            return {'ok': True}

        q = DurableWriteQueue(
            data_dir=tmp_path / 'queue',
            execute_write=mixed_execute,
            workers_per_group=1,
            semaphore_limit=5,
            max_attempts=1,  # fail immediately to dead
            retry_base_seconds=0.01,
            write_timeout_seconds=2.0,
        )
        await q.initialize()

        # 2 will succeed, 1 will dead-letter
        await q.enqueue(
            group_id='proj1', operation='add_episode',
            payload={'content': 'good1', 'group_id': 'proj1', 'name': 'ep1'},
        )
        await q.enqueue(
            group_id='proj1', operation='add_episode',
            payload={'content': 'bad', 'group_id': 'proj1', 'name': 'ep2', 'fail': True},
        )
        await q.enqueue(
            group_id='proj1', operation='add_episode',
            payload={'content': 'good2', 'group_id': 'proj1', 'name': 'ep3'},
        )
        await asyncio.sleep(1.0)

        stats = await q.get_stats()
        assert stats['counts'].get('completed', 0) == 2
        assert stats['counts'].get('dead', 0) == 1
        await q.close()
