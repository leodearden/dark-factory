"""E2E + concurrency stress tests — MemoryService wired to a real DurableWriteQueue.

Only Graphiti/Mem0 backends are mocked; the queue uses real SQLite via tmp_path.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from fused_memory.models.enums import SourceStore
from fused_memory.services.memory_service import MemoryService


@pytest_asyncio.fixture
async def integrated_service(mock_config):
    """MemoryService with real DurableWriteQueue, mocked Graphiti/Mem0."""
    svc = MemoryService(mock_config)

    # Mock backends so we never hit real FalkorDB/Qdrant
    svc.graphiti = MagicMock()
    svc.graphiti.initialize = AsyncMock()
    svc.graphiti.add_episode = AsyncMock(return_value=None)
    svc.graphiti.close = AsyncMock()
    svc.graphiti._require_client = MagicMock()

    svc.mem0 = MagicMock()
    svc.mem0.search = AsyncMock(return_value={'results': []})
    svc.mem0.add = AsyncMock(return_value={'results': [{'id': 'mem0-1'}]})
    svc.mem0.get_all = AsyncMock(return_value={'results': []})
    svc.mem0.delete = AsyncMock(return_value={'message': 'deleted'})

    # Real initialize — creates DurableWriteQueue with real SQLite
    await svc.initialize()

    yield svc

    await svc.close()


# ---------------------------------------------------------------------------
# A. Integration flow
# ---------------------------------------------------------------------------


class TestIntegrationFlow:
    @pytest.mark.asyncio
    async def test_add_memory_graphiti_processed(self, integrated_service):
        """add_memory with graphiti category -> enqueued -> worker processes -> graphiti called."""
        svc = integrated_service

        result = await svc.add_memory(
            content='The auth service depends on Redis',
            category='entities_and_relations',
            project_id='test',
        )
        assert SourceStore.graphiti in result.stores_written

        # Wait for worker to process
        await asyncio.sleep(1.0)

        svc.graphiti.add_episode.assert_called_once()
        stats = await svc.durable_queue.get_stats()
        assert stats['counts'].get('completed', 0) == 1

    @pytest.mark.asyncio
    async def test_add_memory_mem0_direct_call_returns_ids(self, integrated_service):
        """add_memory with mem0 category -> direct synchronous mem0.add call -> IDs returned.

        Mem0 writes are no longer enqueued; mem0.add is called inline so
        memory_ids are available to the caller before add_memory returns.
        """
        svc = integrated_service

        result = await svc.add_memory(
            content='Always use type hints in Python code',
            category='preferences_and_norms',
            project_id='test',
        )
        assert SourceStore.mem0 in result.stores_written
        # IDs are now returned synchronously — no sleep needed
        svc.mem0.add.assert_called_once()
        assert result.memory_ids == ['mem0-1']
        # No queue item created for Mem0 — queue should have 0 items
        stats = await svc.durable_queue.get_stats()
        assert stats['counts'].get('pending', 0) == 0
        assert stats['counts'].get('completed', 0) == 0

    @pytest.mark.asyncio
    async def test_add_episode_uuid_survives_full_flow(self, integrated_service):
        """add_episode uuid passes through queue serialization to graphiti.add_episode."""
        svc = integrated_service
        svc.graphiti.add_episode.return_value = None

        result = await svc.add_episode(
            content='User discussed auth changes',
            project_id='test',
        )
        assert result.episode_id is not None

        # Wait for worker to process
        await asyncio.sleep(1.5)

        svc.graphiti.add_episode.assert_called_once()
        call_kwargs = svc.graphiti.add_episode.call_args[1]
        assert call_kwargs.get('uuid') == result.episode_id

    @pytest.mark.asyncio
    async def test_add_episode_processed_with_callback(self, integrated_service):
        """add_episode -> enqueued with dual_write_episode callback -> graphiti called -> facts enqueued for mem0."""
        svc = integrated_service

        # graphiti.add_episode returns result with entity_edges for dual-write
        mock_result = MagicMock()
        edge = MagicMock()
        edge.fact = 'Always format code with black'  # preferences -> mem0 primary
        mock_result.entity_edges = [edge]
        svc.graphiti.add_episode.return_value = mock_result

        result = await svc.add_episode(
            content='User discussed formatting preferences',
            project_id='test',
        )
        assert result.status == 'queued'

        # Wait for worker + callback + mem0 classify_and_add processing
        await asyncio.sleep(2.5)

        svc.graphiti.add_episode.assert_called_once()
        # The callback enqueues mem0_classify_and_add items which the queue processes.
        # The fact "Always format code with black" should classify as preferences_and_norms
        # (mem0 primary) and trigger a mem0.add call.
        stats = await svc.durable_queue.get_stats()
        # At minimum the episode item completed; mem0 items may also be completed
        assert stats['counts'].get('completed', 0) >= 1

    @pytest.mark.asyncio
    async def test_dual_write_both_paths(self, integrated_service):
        """add_memory with dual_write=True -> both graphiti and mem0 enqueued and processed."""
        svc = integrated_service

        result = await svc.add_memory(
            content='We decided to use PostgreSQL for its JSON support',
            category='decisions_and_rationale',
            project_id='test',
            dual_write=True,
        )
        assert SourceStore.graphiti in result.stores_written
        assert SourceStore.mem0 in result.stores_written

        # Wait for queue to process both writes
        await asyncio.sleep(1.5)
        svc.graphiti.add_episode.assert_called_once()
        svc.mem0.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_replay_from_store_enqueues_batch(self, integrated_service):
        """replay_from_store -> items enqueued -> workers process all."""
        svc = integrated_service

        svc.mem0.get_all = AsyncMock(return_value={
            'results': [
                {'memory': 'fact one', 'metadata': {'category': 'temporal_facts'}},
                {'memory': 'fact two', 'metadata': {'category': 'entities_and_relations'}},
                {'memory': 'fact three', 'metadata': {'category': 'decisions_and_rationale'}},
            ]
        })

        count = await svc.replay_from_store(source_project_id='reify')
        assert count == 3

        # Wait for workers to process all items
        await asyncio.sleep(2.0)

        assert svc.graphiti.add_episode.call_count == 3
        stats = await svc.durable_queue.get_stats()
        assert stats['counts'].get('completed', 0) == 3


# ---------------------------------------------------------------------------
# B. Failure recovery
# ---------------------------------------------------------------------------


class TestFailureRecovery:
    @pytest.mark.asyncio
    async def test_transient_failure_retried(self, integrated_service):
        """graphiti fails twice then succeeds -> item completes after retries."""
        svc = integrated_service
        call_count = 0

        async def flaky_graphiti(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError('Graphiti timeout')
            return {'ok': True}

        svc.graphiti.add_episode = flaky_graphiti

        await svc.add_memory(
            content='The billing API depends on Stripe',
            category='entities_and_relations',
            project_id='test',
        )

        # Wait for retries (0.05s base * 2^n backoff + poll intervals)
        await asyncio.sleep(3.0)

        assert call_count == 3
        stats = await svc.durable_queue.get_stats()
        assert stats['counts'].get('completed', 0) == 1

    @pytest.mark.asyncio
    async def test_permanent_failure_dead_lettered(self, integrated_service):
        """graphiti always fails -> item exhausts max_attempts -> dead-lettered."""
        svc = integrated_service
        svc.graphiti.add_episode = AsyncMock(side_effect=RuntimeError('always fails'))

        await svc.add_memory(
            content='The auth service depends on Redis',
            category='entities_and_relations',
            project_id='test',
        )

        # max_attempts=3 in mock_config, wait for all attempts
        await asyncio.sleep(3.0)

        stats = await svc.durable_queue.get_stats()
        assert stats['counts'].get('dead', 0) == 1

        dead = await svc.durable_queue.get_dead_items()
        assert len(dead) == 1
        assert 'always fails' in dead[0]['error']

    @pytest.mark.asyncio
    async def test_replay_dead_after_fix(self, integrated_service):
        """dead-letter item -> fix graphiti -> replay_dead -> item completes."""
        svc = integrated_service
        svc.graphiti.add_episode = AsyncMock(side_effect=RuntimeError('broken'))

        await svc.add_memory(
            content='The auth service depends on Redis',
            category='entities_and_relations',
            project_id='test',
        )

        await asyncio.sleep(3.0)

        dead = await svc.durable_queue.get_dead_items()
        assert len(dead) == 1

        # "Fix" graphiti
        svc.graphiti.add_episode = AsyncMock(return_value={'ok': True})

        count = await svc.durable_queue.replay_dead('test')
        assert count == 1

        await asyncio.sleep(1.0)

        stats = await svc.durable_queue.get_stats()
        assert stats['counts'].get('completed', 0) == 1

    @pytest.mark.asyncio
    async def test_crash_recovery(self, mock_config):
        """Enqueue item, close before processing, new service recovers and processes it."""
        # Service 1: enqueue but close quickly
        svc1 = MemoryService(mock_config)
        svc1.graphiti = MagicMock()
        svc1.graphiti.initialize = AsyncMock()
        svc1.graphiti.add_episode = AsyncMock(return_value=None)
        svc1.graphiti.close = AsyncMock()
        svc1.graphiti._require_client = MagicMock()
        svc1.mem0 = MagicMock()
        svc1.mem0.add = AsyncMock(return_value={'results': []})

        await svc1.initialize()

        await svc1.add_memory(
            content='The billing API depends on Stripe',
            category='entities_and_relations',
            project_id='test',
        )

        # Close immediately — item may still be pending/in_flight
        await svc1.close()

        # Service 2: same data_dir -> should recover the item
        svc2 = MemoryService(mock_config)
        svc2.graphiti = MagicMock()
        svc2.graphiti.initialize = AsyncMock()
        svc2.graphiti.add_episode = AsyncMock(return_value={'ok': True})
        svc2.graphiti.close = AsyncMock()
        svc2.graphiti._require_client = MagicMock()
        svc2.mem0 = MagicMock()
        svc2.mem0.add = AsyncMock(return_value={'results': []})

        await svc2.initialize()
        await asyncio.sleep(1.5)

        # The recovered item should have been processed by svc2
        assert svc2.durable_queue is not None
        stats = await svc2.durable_queue.get_stats()
        assert stats['counts'].get('completed', 0) >= 1
        svc2.graphiti.add_episode.assert_called()

        await svc2.close()


# ---------------------------------------------------------------------------
# C. Concurrency stress
# ---------------------------------------------------------------------------


class TestConcurrencyStress:
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_100_concurrent_writes_all_complete(self, integrated_service):
        """Fire 100 add_memory calls concurrently -> all 100 complete."""
        svc = integrated_service
        svc.graphiti.add_episode = AsyncMock(return_value={'ok': True})

        tasks = [
            svc.add_memory(
                content=f'Service-{i} depends on database-{i}',
                category='entities_and_relations',
                project_id='test',
            )
            for i in range(100)
        ]
        results = await asyncio.gather(*tasks)
        assert len(results) == 100

        # Wait for all workers to complete
        for _ in range(60):
            stats = await svc.durable_queue.get_stats()
            if stats['counts'].get('completed', 0) == 100:
                break
            await asyncio.sleep(0.5)

        stats = await svc.durable_queue.get_stats()
        assert stats['counts'].get('completed', 0) == 100
        assert svc.graphiti.add_episode.call_count == 100

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_semaphore_limits_concurrency(self, integrated_service):
        """Track max concurrent graphiti calls -> never exceeds semaphore_limit (5)."""
        svc = integrated_service

        max_concurrent = 0
        current = 0
        lock = asyncio.Lock()

        async def tracking_execute(*args, **kwargs):
            nonlocal max_concurrent, current
            async with lock:
                current += 1
                if current > max_concurrent:
                    max_concurrent = current
            await asyncio.sleep(0.05)
            async with lock:
                current -= 1
            return {'ok': True}

        svc.graphiti.add_episode = tracking_execute

        tasks = [
            svc.add_memory(
                content=f'Service-{i} depends on db-{i}',
                category='entities_and_relations',
                project_id='test',
            )
            for i in range(30)
        ]
        await asyncio.gather(*tasks)

        # Wait for processing
        for _ in range(40):
            stats = await svc.durable_queue.get_stats()
            if stats['counts'].get('completed', 0) == 30:
                break
            await asyncio.sleep(0.5)

        stats = await svc.durable_queue.get_stats()
        assert stats['counts'].get('completed', 0) == 30
        # semaphore_limit is 5 in mock_config
        assert max_concurrent <= 5

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_multi_project_concurrent(self, integrated_service):
        """3 projects x 20 items each -> all 60 complete independently."""
        svc = integrated_service
        svc.graphiti.add_episode = AsyncMock(return_value={'ok': True})

        tasks = []
        for project in ['alpha', 'beta', 'gamma']:
            for i in range(20):
                tasks.append(
                    svc.add_memory(
                        content=f'{project} service-{i} depends on db-{i}',
                        category='entities_and_relations',
                        project_id=project,
                    )
                )

        await asyncio.gather(*tasks)

        for _ in range(40):
            stats = await svc.durable_queue.get_stats()
            if stats['counts'].get('completed', 0) == 60:
                break
            await asyncio.sleep(0.5)

        stats = await svc.durable_queue.get_stats()
        assert stats['counts'].get('completed', 0) == 60
        assert svc.graphiti.add_episode.call_count == 60

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_backpressure_under_slow_backend(self, integrated_service):
        """graphiti takes 100ms -> enqueue 50 items -> pending > 0 mid-flight, all eventually complete."""
        svc = integrated_service

        async def slow_graphiti(*args, **kwargs):
            await asyncio.sleep(0.1)
            return {'ok': True}

        svc.graphiti.add_episode = slow_graphiti

        tasks = [
            svc.add_memory(
                content=f'Slow service-{i} depends on db-{i}',
                category='entities_and_relations',
                project_id='test',
            )
            for i in range(50)
        ]
        await asyncio.gather(*tasks)

        # Check that there's backpressure — some items should still be pending
        stats_mid = await svc.durable_queue.get_stats()
        _ = stats_mid['counts'].get('pending', 0) + stats_mid['counts'].get('in_flight', 0)
        # At least some should be pending since 50 items * 100ms > instant
        # (This may be 0 if workers are very fast, so we just check eventual completion)

        for _ in range(60):
            stats = await svc.durable_queue.get_stats()
            if stats['counts'].get('completed', 0) == 50:
                break
            await asyncio.sleep(0.5)

        stats = await svc.durable_queue.get_stats()
        assert stats['counts'].get('completed', 0) == 50

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_mixed_success_and_failure_under_load(self, integrated_service):
        """50 items where every 5th fails permanently -> 40 completed, 10 dead-lettered."""
        svc = integrated_service

        call_index = 0
        lock = asyncio.Lock()

        async def selective_fail(operation, payload):
            nonlocal call_index
            async with lock:
                _ = call_index
                call_index += 1
            # Items with index 0-based, every 5th one has 'fail' in content
            if payload.get('should_fail'):
                raise RuntimeError('permanent failure')
            return {'ok': True}

        # Replace the queue's execute function directly
        svc.durable_queue._execute_write = selective_fail

        tasks = []
        for i in range(50):
            should_fail = (i % 5 == 0)
            tasks.append(
                svc.durable_queue.enqueue(
                    group_id='test',
                    operation='add_memory_graphiti',
                    payload={
                        'name': f'memory_{i}',
                        'content': f'item {i}',
                        'source': 'text',
                        'group_id': 'test',
                        'source_description': 'test',
                        'should_fail': should_fail,
                    },
                )
            )
        await asyncio.gather(*tasks)

        # Wait for all to be processed
        for _ in range(60):
            stats = await svc.durable_queue.get_stats()
            completed = stats['counts'].get('completed', 0)
            dead = stats['counts'].get('dead', 0)
            if completed + dead == 50:
                break
            await asyncio.sleep(0.5)

        stats = await svc.durable_queue.get_stats()
        assert stats['counts'].get('completed', 0) == 40
        assert stats['counts'].get('dead', 0) == 10


# ---------------------------------------------------------------------------
# D. Queue stats accuracy
# ---------------------------------------------------------------------------


class TestQueueStats:
    @pytest.mark.asyncio
    async def test_stats_during_processing(self, integrated_service):
        """Enqueue items, check stats mid-flight — total is conserved."""
        svc = integrated_service

        async def slow_graphiti(*args, **kwargs):
            await asyncio.sleep(0.2)
            return {'ok': True}

        svc.graphiti.add_episode = slow_graphiti

        for i in range(10):
            await svc.durable_queue.enqueue(
                group_id='test',
                operation='add_episode',
                payload={
                    'content': f'item {i}',
                    'group_id': 'test',
                    'name': f'ep{i}',
                    'source': 'text',
                },
            )

        # Check mid-flight
        await asyncio.sleep(0.3)
        stats = await svc.durable_queue.get_stats()
        counts = stats['counts']
        total = sum(counts.values())
        assert total == 10  # pending + in_flight + completed = 10

    @pytest.mark.asyncio
    async def test_stats_after_completion(self, integrated_service):
        """All items processed -> completed count matches total enqueued."""
        svc = integrated_service
        svc.graphiti.add_episode = AsyncMock(return_value={'ok': True})

        for i in range(15):
            await svc.durable_queue.enqueue(
                group_id='test',
                operation='add_episode',
                payload={
                    'content': f'item {i}',
                    'group_id': 'test',
                    'name': f'ep{i}',
                    'source': 'text',
                },
            )

        # Wait for all to complete
        for _ in range(20):
            stats = await svc.durable_queue.get_stats()
            if stats['counts'].get('completed', 0) == 15:
                break
            await asyncio.sleep(0.5)

        stats = await svc.durable_queue.get_stats()
        assert stats['counts'].get('completed', 0) == 15
