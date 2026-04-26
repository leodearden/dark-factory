"""Tests for the durable write queue — real SQLite, mocked Graphiti."""

from __future__ import annotations

import asyncio
import time
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

    @pytest.mark.asyncio
    async def test_get_dead_items_limit_and_order(self, tmp_path):
        """get_dead_items(limit=N) returns at most N items, newest-first."""
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

        # Enqueue 4 items that will all be dead-lettered.
        for i in range(4):
            await q.enqueue(
                group_id='proj1', operation='add_episode',
                payload={'content': f'item {i}', 'group_id': 'proj1', 'name': f'ep{i}'},
            )
        await asyncio.sleep(1.0)

        # All 4 should be dead.
        all_dead = await q.get_dead_items()
        assert len(all_dead) == 4, f'Expected 4 dead items, got {len(all_dead)}'

        # With limit=2, must return exactly 2 newest items (highest ids first).
        limited = await q.get_dead_items(limit=2)
        assert len(limited) == 2
        assert limited[0]['id'] > limited[1]['id'], 'Items must be newest-first (id DESC)'

        # The 2 returned must be the 2 highest-id items.
        all_ids = sorted([item['id'] for item in all_dead], reverse=True)
        limited_ids = [item['id'] for item in limited]
        assert limited_ids == all_ids[:2], (
            f'Limited result should be the top-2 newest: {limited_ids} vs {all_ids[:2]}'
        )

        # No-limit call returns all items (backward compat).
        all_again = await q.get_dead_items()
        assert len(all_again) == 4

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


class TestDeleteDead:
    """Tests for DurableWriteQueue.delete_dead(group_id, ids)."""

    async def _insert_dead_row(self, q, group_id: str = 'proj1') -> int:
        """Insert a dead row directly via SQL and return its id."""
        assert q._db is not None
        cursor = await q._db.execute(
            'INSERT INTO write_queue '
            '(group_id, operation, payload, status, attempts, max_attempts, '
            ' next_retry_at, created_at) '
            "VALUES (?, 'add_episode', '{\"content\":\"dead\"}', 'dead', 3, 3, 0, ?)",
            (group_id, time.time()),
        )
        await q._db.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    async def _insert_pending_row(self, q, group_id: str = 'proj1') -> int:
        """Insert a pending row directly via SQL and return its id."""
        assert q._db is not None
        cursor = await q._db.execute(
            'INSERT INTO write_queue '
            '(group_id, operation, payload, status, attempts, max_attempts, '
            ' next_retry_at, created_at) '
            "VALUES (?, 'add_episode', '{\"content\":\"pending\"}', 'pending', 0, 3, 0, ?)",
            (group_id, time.time()),
        )
        await q._db.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    @pytest.mark.asyncio
    async def test_bulk_delete_dead_rows(self, queue):
        """Bulk-deleting dead rows returns them in 'deleted' and removes them."""
        id1 = await self._insert_dead_row(queue, 'proj1')
        id2 = await self._insert_dead_row(queue, 'proj1')

        result = await queue.delete_dead(group_id='proj1', ids=[id1, id2])

        assert sorted(result['deleted']) == sorted([id1, id2])
        assert result['not_found'] == []

        # Rows must no longer appear via get_dead_items
        remaining = await queue.get_dead_items(group_id='proj1')
        remaining_ids = [item['id'] for item in remaining]
        assert id1 not in remaining_ids
        assert id2 not in remaining_ids

    @pytest.mark.asyncio
    async def test_missing_ids_in_not_found(self, queue):
        """Non-existent IDs appear in 'not_found'."""
        result = await queue.delete_dead(group_id='proj1', ids=[99999, 88888])

        assert result['deleted'] == []
        assert sorted(result['not_found']) == [88888, 99999]

    @pytest.mark.asyncio
    async def test_cross_project_ids_not_deleted(self, queue):
        """Dead rows from proj-b are not deleted when calling with proj-a."""
        id_b = await self._insert_dead_row(queue, 'proj-b')

        result = await queue.delete_dead(group_id='proj-a', ids=[id_b])

        assert result['deleted'] == []
        assert result['not_found'] == [id_b]

        # Row in proj-b must still be there
        dead_b = await queue.get_dead_items(group_id='proj-b')
        assert any(item['id'] == id_b for item in dead_b)

    @pytest.mark.asyncio
    async def test_non_dead_status_ids_not_deleted(self, queue):
        """Pending rows are silently skipped (land in 'not_found') and stay in table."""
        id_pending = await self._insert_pending_row(queue, 'proj1')

        result = await queue.delete_dead(group_id='proj1', ids=[id_pending])

        assert result['deleted'] == []
        assert result['not_found'] == [id_pending]

        # Row must still be present
        assert queue._db is not None
        cursor = await queue._db.execute(
            'SELECT id FROM write_queue WHERE id = ?', (id_pending,)
        )
        row = await cursor.fetchone()
        assert row is not None, 'Pending row must not be deleted'

    @pytest.mark.asyncio
    async def test_empty_ids_no_op(self, queue):
        """Empty ids list returns empty buckets without touching the DB."""
        result = await queue.delete_dead(group_id='proj1', ids=[])

        assert result == {'deleted': [], 'not_found': []}

    @pytest.mark.asyncio
    async def test_mixed_call_partitions_correctly(self, queue):
        """Mixed input: valid dead, cross-project dead, non-existent, non-dead partitions."""
        id_dead_proj1 = await self._insert_dead_row(queue, 'proj1')
        id_dead_proj2 = await self._insert_dead_row(queue, 'proj2')
        id_pending = await self._insert_pending_row(queue, 'proj1')
        id_nonexistent = 999999

        result = await queue.delete_dead(
            group_id='proj1',
            ids=[id_dead_proj1, id_dead_proj2, id_pending, id_nonexistent],
        )

        assert result['deleted'] == [id_dead_proj1]
        assert sorted(result['not_found']) == sorted([id_dead_proj2, id_pending, id_nonexistent])


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
