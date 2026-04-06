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


class TestEnqueueBatchRollback:
    @pytest.mark.asyncio
    async def test_rollback_on_mid_batch_exception(self, queue):
        """Bug 4: enqueue_batch must roll back on a mid-batch exception.

        A batch where the 2nd item is missing 'operation' (KeyError) should
        raise, and the DB should contain 0 pending items (no partial inserts).
        """
        items = [
            {'group_id': 'proj1', 'operation': 'add_episode',
             'payload': {'content': 'good', 'group_id': 'proj1', 'name': 'ep1'}},
            # Missing 'operation' key — triggers KeyError mid-batch
            {'group_id': 'proj1',
             'payload': {'content': 'bad', 'group_id': 'proj1', 'name': 'ep2'}},
            {'group_id': 'proj1', 'operation': 'add_episode',
             'payload': {'content': 'also good', 'group_id': 'proj1', 'name': 'ep3'}},
        ]

        with pytest.raises(KeyError):
            await queue.enqueue_batch(items)

        # Verify rollback: no items should have been persisted
        assert queue._db is not None
        cursor = await queue._db.execute(
            "SELECT COUNT(*) FROM write_queue WHERE status='pending'"
        )
        row = await cursor.fetchone()
        assert row[0] == 0, (
            f'Expected 0 pending items after rollback, got {row[0]}'
        )


class TestCallbackFailure:
    @pytest.mark.asyncio
    async def test_callback_error_triggers_retry(self, queue, mock_execute):
        """Callback failure should retry the item, not silently complete it."""
        failing_callback = AsyncMock(side_effect=ValueError('callback boom'))
        queue.register_callback('bad_cb', failing_callback)

        await queue.enqueue(
            group_id='proj1', operation='add_episode',
            payload={'content': 'test', 'group_id': 'proj1', 'name': 'ep'},
            callback_type='bad_cb',
        )

        # Wait for all retries to exhaust (max_attempts=3 in fixture)
        await asyncio.sleep(2.0)

        stats = await queue.get_stats()
        # Item should be dead-lettered, not completed — callback kept failing
        assert stats['counts'].get('dead', 0) == 1
        assert stats['counts'].get('completed', 0) == 0

    @pytest.mark.asyncio
    async def test_callback_success_completes_item(self, queue, mock_execute):
        """Callback success should mark the item completed."""
        ok_callback = AsyncMock()
        queue.register_callback('ok_cb', ok_callback)

        await queue.enqueue(
            group_id='proj1', operation='add_episode',
            payload={'content': 'test', 'group_id': 'proj1', 'name': 'ep'},
            callback_type='ok_cb',
        )
        await asyncio.sleep(0.3)

        stats = await queue.get_stats()
        assert stats['counts'].get('completed', 0) == 1
        ok_callback.assert_called_once()


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


class TestPurgeDead:
    """Tests for DurableWriteQueue.purge_dead()."""

    @pytest.mark.asyncio
    async def test_purge_dead_by_ids(self, tmp_path):
        """purge_dead(ids=[...]) removes only the specified dead rows."""
        execute = AsyncMock(side_effect=RuntimeError('always fails'))
        q = DurableWriteQueue(
            data_dir=tmp_path / 'queue',
            execute_write=execute,
            workers_per_group=1,
            semaphore_limit=5,
            max_attempts=1,
            retry_base_seconds=0.01,
            write_timeout_seconds=2.0,
        )
        await q.initialize()

        id1 = await q.enqueue(
            group_id='proj1', operation='add_episode',
            payload={'content': 'item1', 'group_id': 'proj1', 'name': 'ep1'},
        )
        id2 = await q.enqueue(
            group_id='proj1', operation='add_episode',
            payload={'content': 'item2', 'group_id': 'proj1', 'name': 'ep2'},
        )
        id3 = await q.enqueue(
            group_id='proj1', operation='add_episode',
            payload={'content': 'item3', 'group_id': 'proj1', 'name': 'ep3'},
        )
        # Wait for all three to dead-letter (max_attempts=1)
        await asyncio.sleep(1.0)

        dead_before = await q.get_dead_items()
        assert len(dead_before) == 3

        purged = await q.purge_dead(ids=[id1, id2])
        assert purged == 2

        dead_after = await q.get_dead_items()
        assert len(dead_after) == 1
        assert dead_after[0]['id'] == id3

        # Verify rows are physically gone from DB
        assert q._db is not None
        cursor = await q._db.execute(
            'SELECT COUNT(*) FROM write_queue WHERE id IN (?, ?)',
            (id1, id2),
        )
        row = await cursor.fetchone()
        assert row[0] == 0, f'Expected 0 rows for purged ids, got {row[0]}'

        await q.close()

    @pytest.mark.asyncio
    async def test_purge_dead_by_error_pattern(self, tmp_path):
        """purge_dead(error_pattern=...) removes only dead rows matching the LIKE pattern."""
        call_count = 0

        async def patterned_execute(op, payload):
            nonlocal call_count
            call_count += 1
            name = payload.get('name', '')
            if name.startswith('node'):
                raise RuntimeError('NodeNotFoundError: node abc123 not found')
            raise RuntimeError('Query timed out')

        q = DurableWriteQueue(
            data_dir=tmp_path / 'queue',
            execute_write=patterned_execute,
            workers_per_group=1,
            semaphore_limit=5,
            max_attempts=1,
            retry_base_seconds=0.01,
            write_timeout_seconds=2.0,
        )
        await q.initialize()

        # 2 items with NodeNotFoundError, 3 items with Query timed out
        for i in range(2):
            await q.enqueue(
                group_id='proj1', operation='add_episode',
                payload={'content': f'node{i}', 'group_id': 'proj1', 'name': f'node{i}'},
            )
        for i in range(3):
            await q.enqueue(
                group_id='proj1', operation='add_episode',
                payload={'content': f'timeout{i}', 'group_id': 'proj1', 'name': f'timeout{i}'},
            )
        await asyncio.sleep(1.0)

        dead_before = await q.get_dead_items()
        assert len(dead_before) == 5

        purged = await q.purge_dead(error_pattern='%NodeNotFoundError%')
        assert purged == 2

        dead_after = await q.get_dead_items()
        assert len(dead_after) == 3
        for item in dead_after:
            assert 'Query timed out' in item['error']

        await q.close()

    @pytest.mark.asyncio
    async def test_purge_dead_by_group_id(self, tmp_path):
        """purge_dead(group_id=...) removes only dead rows for the given group."""
        execute = AsyncMock(side_effect=RuntimeError('always fails'))
        q = DurableWriteQueue(
            data_dir=tmp_path / 'queue',
            execute_write=execute,
            workers_per_group=2,
            semaphore_limit=10,
            max_attempts=1,
            retry_base_seconds=0.01,
            write_timeout_seconds=2.0,
        )
        await q.initialize()

        for i in range(2):
            await q.enqueue(
                group_id='alpha', operation='add_episode',
                payload={'content': f'alpha{i}', 'group_id': 'alpha', 'name': f'ep{i}'},
            )
        for i in range(3):
            await q.enqueue(
                group_id='beta', operation='add_episode',
                payload={'content': f'beta{i}', 'group_id': 'beta', 'name': f'ep{i}'},
            )
        await asyncio.sleep(1.0)

        dead_before = await q.get_dead_items()
        assert len(dead_before) == 5

        purged = await q.purge_dead(group_id='alpha')
        assert purged == 2

        dead_after = await q.get_dead_items()
        assert len(dead_after) == 3
        for item in dead_after:
            assert item['group_id'] == 'beta'

        await q.close()

    @pytest.mark.asyncio
    async def test_purge_dead_combined_filters(self, tmp_path):
        """purge_dead(group_id=..., error_pattern=...) ANDs the two filters."""
        call_count = 0

        async def patterned_execute(op, payload):
            nonlocal call_count
            call_count += 1
            name = payload.get('name', '')
            if name.startswith('node'):
                raise RuntimeError('NodeNotFoundError: node xyz not found')
            raise RuntimeError('Query timed out')

        q = DurableWriteQueue(
            data_dir=tmp_path / 'queue',
            execute_write=patterned_execute,
            workers_per_group=2,
            semaphore_limit=10,
            max_attempts=1,
            retry_base_seconds=0.01,
            write_timeout_seconds=2.0,
        )
        await q.initialize()

        # alpha: 1 NodeNotFoundError + 1 timeout
        await q.enqueue(
            group_id='alpha', operation='add_episode',
            payload={'content': 'node-alpha', 'group_id': 'alpha', 'name': 'node-alpha'},
        )
        await q.enqueue(
            group_id='alpha', operation='add_episode',
            payload={'content': 'timeout-alpha', 'group_id': 'alpha', 'name': 'timeout-alpha'},
        )
        # beta: 1 NodeNotFoundError + 1 timeout
        await q.enqueue(
            group_id='beta', operation='add_episode',
            payload={'content': 'node-beta', 'group_id': 'beta', 'name': 'node-beta'},
        )
        await q.enqueue(
            group_id='beta', operation='add_episode',
            payload={'content': 'timeout-beta', 'group_id': 'beta', 'name': 'timeout-beta'},
        )
        await asyncio.sleep(1.0)

        dead_before = await q.get_dead_items()
        assert len(dead_before) == 4

        # Purge only alpha NodeNotFoundError items
        purged = await q.purge_dead(group_id='alpha', error_pattern='%NodeNotFoundError%')
        assert purged == 1

        dead_after = await q.get_dead_items()
        assert len(dead_after) == 3
        # alpha timeout + both beta items remain
        remaining_groups = [item['group_id'] for item in dead_after]
        assert remaining_groups.count('alpha') == 1
        assert remaining_groups.count('beta') == 2

        await q.close()
