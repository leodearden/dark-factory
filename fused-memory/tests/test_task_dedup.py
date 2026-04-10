"""Tests for task deduplication in the TaskInterceptor middleware.

Layer 1: in-memory title-hash cache (exact match, time-windowed)
Layer 2: Qdrant vector similarity via TaskDeduplicator (tested with mock embedder)
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from fused_memory.middleware.task_interceptor import _DEDUP_CACHE_TTL, TaskInterceptor
from fused_memory.reconciliation.event_buffer import EventBuffer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def taskmaster():
    tm = AsyncMock()
    tm.get_task = AsyncMock(return_value={'id': '1', 'status': 'pending'})
    tm.set_task_status = AsyncMock(return_value={'success': True})
    tm.get_tasks = AsyncMock(return_value={'tasks': []})
    tm.update_task = AsyncMock(return_value={'success': True})

    _next_id = [10]

    async def _add_task(**kwargs):
        tid = str(_next_id[0])
        _next_id[0] += 1
        return {'id': tid, 'title': kwargs.get('title', '')}

    tm.add_task = AsyncMock(side_effect=_add_task)
    return tm


@pytest_asyncio.fixture
async def event_buffer(tmp_path):
    buf = EventBuffer(db_path=tmp_path / 'dedup_test.db', buffer_size_threshold=100)
    await buf.initialize()
    yield buf
    await buf.close()


@pytest.fixture
def interceptor(taskmaster, event_buffer):
    """Interceptor with no config (Layer 2 disabled, Layer 1 only)."""
    return TaskInterceptor(taskmaster, None, event_buffer)


# ---------------------------------------------------------------------------
# Layer 1: In-memory title-hash cache
# ---------------------------------------------------------------------------

class TestTitleHashCache:

    @pytest.mark.asyncio
    async def test_exact_duplicate_returns_cached(self, interceptor, taskmaster):
        """Second add_task with identical title returns the first result."""
        r1 = await interceptor.add_task(
            project_root='/project', title='Fix auth bug', description='desc',
        )
        assert r1['id'] == '10'

        r2 = await interceptor.add_task(
            project_root='/project', title='Fix auth bug', description='desc',
        )
        # Should return cached result, NOT create a new task
        assert r2['id'] == '10'
        # Taskmaster.add_task should only have been called once
        assert taskmaster.add_task.call_count == 1

    @pytest.mark.asyncio
    async def test_different_titles_not_deduped(self, interceptor, taskmaster):
        """Different titles should create separate tasks."""
        r1 = await interceptor.add_task(
            project_root='/project', title='Fix auth bug', description='d1',
        )
        r2 = await interceptor.add_task(
            project_root='/project', title='Add logging to parser', description='d2',
        )
        assert r1['id'] != r2['id']
        assert taskmaster.add_task.call_count == 2

    @pytest.mark.asyncio
    async def test_case_insensitive_match(self, interceptor, taskmaster):
        """Title matching is case-insensitive."""
        await interceptor.add_task(
            project_root='/project', title='Fix Auth Bug', description='d',
        )
        r2 = await interceptor.add_task(
            project_root='/project', title='fix auth bug', description='d',
        )
        assert r2['id'] == '10'
        assert taskmaster.add_task.call_count == 1

    @pytest.mark.asyncio
    async def test_whitespace_normalized(self, interceptor, taskmaster):
        """Extra whitespace is collapsed before hashing."""
        await interceptor.add_task(
            project_root='/project', title='Fix  auth   bug', description='d',
        )
        r2 = await interceptor.add_task(
            project_root='/project', title='Fix auth bug', description='d',
        )
        assert r2['id'] == '10'
        assert taskmaster.add_task.call_count == 1

    @pytest.mark.asyncio
    async def test_cache_expires_after_ttl(self, interceptor, taskmaster):
        """After TTL expires, the same title should create a new task."""
        await interceptor.add_task(
            project_root='/project', title='Fix auth bug', description='d',
        )

        # Manually expire the cache entry
        for key in interceptor._dedup_cache:
            result, _ = interceptor._dedup_cache[key]
            interceptor._dedup_cache[key] = (result, time.monotonic() - _DEDUP_CACHE_TTL - 1)

        r2 = await interceptor.add_task(
            project_root='/project', title='Fix auth bug', description='d',
        )
        assert r2['id'] == '11'  # New task created
        assert taskmaster.add_task.call_count == 2

    @pytest.mark.asyncio
    async def test_no_title_skips_dedup(self, interceptor, taskmaster):
        """Tasks created via prompt (no title) skip dedup."""
        r1 = await interceptor.add_task(
            project_root='/project', prompt='do something',
        )
        r2 = await interceptor.add_task(
            project_root='/project', prompt='do something',
        )
        # prompt-based tasks use 'prompt' as the title key for dedup,
        # so they WILL be deduped if the prompt matches
        assert r1['id'] == r2['id']

    @pytest.mark.asyncio
    async def test_metadata_preserved_on_first_call(self, interceptor, taskmaster):
        """Metadata update_task is still called on the original creation."""
        await interceptor.add_task(
            project_root='/project',
            title='Fix bug',
            description='d',
            metadata={'source': 'test', 'modules': ['a/']},
        )
        taskmaster.update_task.assert_called_once()
        call_kwargs = taskmaster.update_task.call_args[1]
        assert call_kwargs['metadata'] == {'source': 'test', 'modules': ['a/']}

    @pytest.mark.asyncio
    async def test_metadata_skipped_on_dedup_hit(self, interceptor, taskmaster):
        """Deduped calls should NOT call update_task for metadata."""
        await interceptor.add_task(
            project_root='/project', title='Fix bug', description='d',
            metadata={'source': 'test'},
        )
        taskmaster.update_task.reset_mock()

        await interceptor.add_task(
            project_root='/project', title='Fix bug', description='d',
            metadata={'source': 'test2'},
        )
        taskmaster.update_task.assert_not_called()


# ---------------------------------------------------------------------------
# Layer 2: Qdrant vector similarity (via mock)
# ---------------------------------------------------------------------------

class TestVectorDedup:

    @pytest.fixture
    def mock_deduplicator(self):
        dedup = AsyncMock()
        dedup.find_duplicate = AsyncMock(return_value=None)
        dedup.record_task = AsyncMock()
        return dedup

    @pytest.fixture
    def interceptor_with_dedup(self, taskmaster, event_buffer, mock_deduplicator):
        """Interceptor with a mocked deduplicator injected."""
        interceptor = TaskInterceptor(taskmaster, None, event_buffer)
        interceptor._deduplicator = mock_deduplicator
        return interceptor

    @pytest.mark.asyncio
    async def test_vector_dedup_blocks_similar(
        self, interceptor_with_dedup, mock_deduplicator, taskmaster,
    ):
        """When vector dedup finds a match, no new task is created."""
        mock_deduplicator.find_duplicate.return_value = {
            'task_id': '42',
            'task_title': 'Harden field_calculus test helpers',
            'score': 0.95,
        }

        result = await interceptor_with_dedup.add_task(
            project_root='/project',
            title='Harden field_calculus_tests helpers',
            description='d',
        )

        assert result['id'] == '42'
        assert result['deduplicated'] is True
        assert result['similarity_score'] == 0.95
        taskmaster.add_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_vector_dedup_allows_different(
        self, interceptor_with_dedup, mock_deduplicator, taskmaster,
    ):
        """When vector dedup finds no match, task is created normally."""
        mock_deduplicator.find_duplicate.return_value = None

        result = await interceptor_with_dedup.add_task(
            project_root='/project', title='Add logging to parser', description='d',
        )

        assert result['id'] == '10'
        taskmaster.add_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_vector_records_after_creation(
        self, interceptor_with_dedup, mock_deduplicator,
    ):
        """After creating a task, the embedding is recorded for future checks."""
        mock_deduplicator.find_duplicate.return_value = None

        await interceptor_with_dedup.add_task(
            project_root='/project', title='New feature X', description='d',
        )
        # Let background tasks complete
        await asyncio.sleep(0.05)
        await interceptor_with_dedup.drain()

        mock_deduplicator.record_task.assert_called_once()
        call_args = mock_deduplicator.record_task.call_args
        assert call_args[0][1] == 'New feature X'  # title

    @pytest.mark.asyncio
    async def test_cache_takes_priority_over_vector(
        self, interceptor_with_dedup, mock_deduplicator, taskmaster,
    ):
        """Layer 1 (cache) fires before Layer 2 (vector) is consulted."""
        mock_deduplicator.find_duplicate.return_value = None

        # First call: creates task, populates cache
        await interceptor_with_dedup.add_task(
            project_root='/project', title='Fix bug Z', description='d',
        )
        mock_deduplicator.find_duplicate.reset_mock()

        # Second call: cache hit, vector check should NOT be called
        await interceptor_with_dedup.add_task(
            project_root='/project', title='Fix bug Z', description='d',
        )
        mock_deduplicator.find_duplicate.assert_not_called()
        assert taskmaster.add_task.call_count == 1

    @pytest.mark.asyncio
    async def test_vector_error_does_not_block(self, taskmaster, event_buffer):
        """If the deduplicator's internal Qdrant call raises, task creation proceeds.

        We test through the real TaskDeduplicator (not a mock) so the try/except
        in find_duplicate actually catches the error.
        """
        from fused_memory.middleware.task_dedup import TaskDeduplicator

        interceptor = TaskInterceptor(taskmaster, None, event_buffer)

        # Create a real deduplicator with a broken client
        dedup = TaskDeduplicator.__new__(TaskDeduplicator)
        dedup._config = MagicMock()
        dedup._initialized_collections = set()
        broken_client = AsyncMock()
        broken_client.collection_exists = AsyncMock(side_effect=RuntimeError('Qdrant down'))
        dedup._client = broken_client
        dedup._embedder = AsyncMock()
        dedup._embedder.create = AsyncMock(return_value=[0.1] * 10)
        interceptor._deduplicator = dedup

        result = await interceptor.add_task(
            project_root='/project', title='Some task', description='d',
        )
        assert result['id'] == '10'
        taskmaster.add_task.assert_called_once()


# ---------------------------------------------------------------------------
# TaskDeduplicator unit tests (with mocked Qdrant + embedder)
# ---------------------------------------------------------------------------

class TestTaskDeduplicatorUnit:

    @pytest.fixture
    def mock_config(self):
        cfg = MagicMock()
        cfg.mem0.qdrant_url = 'http://localhost:6333'
        cfg.embedder.provider = 'openai'
        cfg.embedder.model = 'text-embedding-3-small'
        cfg.embedder.dimensions = 1536
        cfg.embedder.providers.openai.api_key = 'test-key'
        cfg.embedder.providers.openai.api_url = None
        return cfg

    @pytest.mark.asyncio
    async def test_find_duplicate_no_match(self, mock_config):
        from fused_memory.middleware.task_dedup import TaskDeduplicator

        dedup = TaskDeduplicator(mock_config)

        mock_client = AsyncMock()
        mock_client.collection_exists = AsyncMock(return_value=True)
        mock_result = MagicMock()
        mock_result.points = []
        mock_client.query_points = AsyncMock(return_value=mock_result)
        dedup._client = mock_client

        mock_embedder = AsyncMock()
        mock_embedder.create = AsyncMock(return_value=[0.1] * 1536)
        dedup._embedder = mock_embedder

        result = await dedup.find_duplicate('New task', 'test_project')
        assert result is None

    @pytest.mark.asyncio
    async def test_find_duplicate_with_match(self, mock_config):
        from fused_memory.middleware.task_dedup import TaskDeduplicator

        dedup = TaskDeduplicator(mock_config)

        mock_point = MagicMock()
        mock_point.score = 0.96
        mock_point.payload = {'task_id': '42', 'task_title': 'Fix bug'}

        mock_client = AsyncMock()
        mock_client.collection_exists = AsyncMock(return_value=True)
        mock_result = MagicMock()
        mock_result.points = [mock_point]
        mock_client.query_points = AsyncMock(return_value=mock_result)
        dedup._client = mock_client

        mock_embedder = AsyncMock()
        mock_embedder.create = AsyncMock(return_value=[0.1] * 1536)
        dedup._embedder = mock_embedder

        result = await dedup.find_duplicate('Fix bugg', 'test_project')
        assert result is not None
        assert result['task_id'] == '42'
        assert result['score'] == 0.96

    @pytest.mark.asyncio
    async def test_record_task_upserts(self, mock_config):
        from fused_memory.middleware.task_dedup import TaskDeduplicator

        dedup = TaskDeduplicator(mock_config)

        mock_client = AsyncMock()
        mock_client.collection_exists = AsyncMock(return_value=True)
        mock_client.upsert = AsyncMock()
        dedup._client = mock_client

        mock_embedder = AsyncMock()
        mock_embedder.create = AsyncMock(return_value=[0.1] * 1536)
        dedup._embedder = mock_embedder

        await dedup.record_task('42', 'Fix bug', 'test_project')
        mock_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_creates_collection_on_first_use(self, mock_config):
        from fused_memory.middleware.task_dedup import TaskDeduplicator

        dedup = TaskDeduplicator(mock_config)

        mock_client = AsyncMock()
        mock_client.collection_exists = AsyncMock(return_value=False)
        mock_client.create_collection = AsyncMock(return_value=True)
        mock_result = MagicMock()
        mock_result.points = []
        mock_client.query_points = AsyncMock(return_value=mock_result)
        dedup._client = mock_client

        mock_embedder = AsyncMock()
        mock_embedder.create = AsyncMock(return_value=[0.1] * 1536)
        dedup._embedder = mock_embedder

        await dedup.find_duplicate('Test', 'my_project')
        mock_client.create_collection.assert_called_once()
        cname = mock_client.create_collection.call_args[1]['collection_name']
        assert cname == 'task_dedup_my_project'


# ---------------------------------------------------------------------------
# Escalation content fingerprint (hash-prefix stripping)
# ---------------------------------------------------------------------------

def _strip_hash_prefix(detail: str) -> str:
    """Local copy of the helper from orchestrator.steward for testing."""
    if detail.startswith('#hash:') and '#' in detail[6:]:
        return detail[detail.index('#', 6) + 1:]
    return detail


# ---------------------------------------------------------------------------
# Bulk-operation dedup (expand_task / parse_prd)
# ---------------------------------------------------------------------------

class TestBulkDedup:
    """Tests for post-hoc dedup after expand_task / parse_prd.

    The interceptor snapshots the task tree before each bulk call, diffs it
    against the tree after, and removes any newly-created tasks whose title
    matches a pre-existing one (exact hash in Layer 1, vector in Layer 2).
    """

    # ── shared fixtures ──────────────────────────────────────────────────

    @pytest.fixture
    def pre_snapshot(self):
        """Task tree BEFORE expand_task runs."""
        return {
            'tasks': [
                {
                    'id': '1',
                    'title': 'Parent task',
                    'status': 'done',
                    'subtasks': [
                        {'id': '1.1', 'title': 'Fix auth bug', 'status': 'done'},
                    ],
                }
            ]
        }

    @pytest.fixture
    def post_snapshot(self):
        """Task tree AFTER expand_task adds duplicates + a genuinely new subtask."""
        return {
            'tasks': [
                {
                    'id': '1',
                    'title': 'Parent task',
                    'status': 'done',
                    'subtasks': [
                        {'id': '1.1', 'title': 'Fix auth bug', 'status': 'done'},
                        {'id': '1.2', 'title': 'Fix auth bug', 'status': 'pending'},
                        {'id': '1.3', 'title': 'Add retries', 'status': 'pending'},
                    ],
                }
            ]
        }

    @pytest.fixture
    def bulk_taskmaster(self, pre_snapshot, post_snapshot):
        """Taskmaster mock whose get_tasks returns pre then post snapshot."""
        tm = AsyncMock()
        tm.expand_task = AsyncMock(return_value={'success': True})
        tm.parse_prd = AsyncMock(return_value={'success': True})
        tm.remove_task = AsyncMock(return_value={'success': True})
        # First call (pre-snapshot) → pre, second call (post-snapshot) → post
        tm.get_tasks = AsyncMock(side_effect=[pre_snapshot, post_snapshot])
        return tm

    @pytest.fixture
    def bulk_interceptor(self, bulk_taskmaster, event_buffer):
        return TaskInterceptor(bulk_taskmaster, None, event_buffer)

    # ── step-1 test ───────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_expand_task_removes_subtask_duplicating_existing_done_task(
        self, bulk_interceptor, bulk_taskmaster,
    ):
        """expand_task removes newly-created subtasks whose title exactly matches
        a pre-existing subtask — even done/cancelled ones — but keeps unique ones."""
        result = await bulk_interceptor.expand_task('1', project_root='/project')

        # 1.2 duplicates 'Fix auth bug' from done subtask 1.1 → exactly one removal
        assert bulk_taskmaster.remove_task.call_count == 1
        removed_ids = [c.args[0] for c in bulk_taskmaster.remove_task.call_args_list]
        assert '1.2' in removed_ids

        # 1.3 'Add retries' is unique — must NOT be removed
        assert '1.3' not in removed_ids

        # Result must carry dedup metadata
        assert 'dedup' in result
        assert any(
            entry.get('task_id') == '1.2'
            for entry in result['dedup']['removed']
        )

    # ── step-3 test ───────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_expand_task_skips_dedup_for_existing_subtasks(
        self, bulk_taskmaster, event_buffer, pre_snapshot,
    ):
        """When all post-call subtasks already existed in the pre-snapshot,
        no removal is attempted and dedup['removed'] is empty."""
        # Both calls to get_tasks return the same snapshot — no new tasks created
        bulk_taskmaster.get_tasks = AsyncMock(side_effect=[pre_snapshot, pre_snapshot])
        interceptor = TaskInterceptor(bulk_taskmaster, None, event_buffer)

        result = await interceptor.expand_task('1', project_root='/project')

        bulk_taskmaster.remove_task.assert_not_called()
        assert result['dedup']['removed'] == []

    # ── step-5 test ───────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_expand_task_vector_dedup_removes_similar_subtask(
        self, bulk_taskmaster, event_buffer, pre_snapshot,
    ):
        """Layer-2 (vector) dedup removes a new subtask that is semantically
        similar to a pre-existing one even when the title is not an exact match."""
        # post snapshot has one new subtask with a DIFFERENT title (Layer 1 misses)
        post = {
            'tasks': [
                {
                    'id': '1', 'title': 'Parent task', 'status': 'done',
                    'subtasks': [
                        {'id': '1.1', 'title': 'Fix auth bug', 'status': 'done'},
                        # new task — similar but not identical title
                        {'id': '1.2', 'title': 'Fix authentication vulnerability', 'status': 'pending'},
                    ],
                }
            ]
        }
        bulk_taskmaster.get_tasks = AsyncMock(side_effect=[pre_snapshot, post])

        mock_dedup = AsyncMock()
        mock_dedup.find_duplicate = AsyncMock(return_value={
            'task_id': '1.1',
            'task_title': 'Fix auth bug',
            'score': 0.94,
        })
        mock_dedup.record_task = AsyncMock()

        interceptor = TaskInterceptor(bulk_taskmaster, None, event_buffer)
        interceptor._deduplicator = mock_dedup

        result = await interceptor.expand_task('1', project_root='/project')

        # Layer-2 found a match → remove_task must be called for the new subtask
        mock_dedup.find_duplicate.assert_called_once()
        call_args = mock_dedup.find_duplicate.call_args
        assert 'fix authentication vulnerability' in call_args.args[0].lower() or \
               'Fix authentication vulnerability' in call_args.args[0]

        assert bulk_taskmaster.remove_task.call_count == 1
        removed_ids = [c.args[0] for c in bulk_taskmaster.remove_task.call_args_list]
        assert '1.2' in removed_ids

        assert 'dedup' in result
        assert any(e.get('task_id') == '1.2' for e in result['dedup']['removed'])

    # ── step-7 test ───────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_expand_task_records_surviving_subtasks_in_layer1_and_layer2(
        self, bulk_taskmaster, event_buffer,
    ):
        """Surviving (non-duplicate) new subtasks are recorded in Layer-1 cache
        and via a fire-and-forget Layer-2 record_task call."""
        pre: dict = {'tasks': []}
        post = {
            'tasks': [
                {'id': '10', 'title': 'Task Alpha', 'status': 'pending', 'subtasks': []},
                {'id': '11', 'title': 'Task Beta',  'status': 'pending', 'subtasks': []},
            ]
        }
        bulk_taskmaster.get_tasks = AsyncMock(side_effect=[pre, post])

        mock_dedup = AsyncMock()
        mock_dedup.find_duplicate = AsyncMock(return_value=None)
        mock_dedup.record_task = AsyncMock()

        interceptor = TaskInterceptor(bulk_taskmaster, None, event_buffer)
        interceptor._deduplicator = mock_dedup

        result = await interceptor.expand_task('1', project_root='/project')
        await asyncio.sleep(0.05)
        await interceptor.drain()

        # Layer-1 cache must contain both survivors
        h_alpha = interceptor._title_hash('Task Alpha')
        h_beta  = interceptor._title_hash('Task Beta')
        assert h_alpha in interceptor._dedup_cache
        assert h_beta  in interceptor._dedup_cache

        # Layer-2 record_task must have been called once per survivor.
        # Titles are normalized (lowercased) by _flatten_tasks before being
        # passed to record_task.
        assert mock_dedup.record_task.call_count == 2
        recorded_titles = {c.args[1] for c in mock_dedup.record_task.call_args_list}
        assert 'task alpha' in recorded_titles
        assert 'task beta'  in recorded_titles

        assert result['dedup']['removed'] == []
        assert len(result['dedup']['kept']) == 2

    # ── step-9 test ───────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_parse_prd_removes_top_level_task_duplicating_existing(
        self, bulk_taskmaster, event_buffer,
    ):
        """parse_prd removes new top-level tasks whose title matches a
        pre-existing task (even a done one)."""
        pre: dict = {
            'tasks': [
                {'id': '5', 'title': 'Implement parser', 'status': 'done', 'subtasks': []},
            ]
        }
        post: dict = {
            'tasks': [
                {'id': '5', 'title': 'Implement parser', 'status': 'done', 'subtasks': []},
                {'id': '42', 'title': 'Implement parser', 'status': 'pending', 'subtasks': []},
            ]
        }
        bulk_taskmaster.get_tasks = AsyncMock(side_effect=[pre, post])
        interceptor = TaskInterceptor(bulk_taskmaster, None, event_buffer)

        result = await interceptor.parse_prd('/some/prd.md', project_root='/project')

        assert bulk_taskmaster.remove_task.call_count == 1
        removed_ids = [c.args[0] for c in bulk_taskmaster.remove_task.call_args_list]
        assert '42' in removed_ids

        assert 'dedup' in result
        assert any(e.get('task_id') == '42' for e in result['dedup']['removed'])

    # ── step-11 test (regression) ─────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_expand_task_rerun_idempotent(self, bulk_taskmaster, event_buffer):
        """Regression: re-running expand_task on an already-expanded task must
        not create duplicate subtasks.

        First call: empty pre → two new subtasks 1.1, 1.2 created.
        Second call: pre = post from first call; Taskmaster (re-)generates 1.3
        and 1.4 with the same titles as 1.1/1.2 → both must be removed.
        """
        first_pre: dict = {'tasks': [{'id': '1', 'title': 'Parent', 'status': 'pending', 'subtasks': []}]}
        first_post: dict = {
            'tasks': [{
                'id': '1', 'title': 'Parent', 'status': 'pending',
                'subtasks': [
                    {'id': '1.1', 'title': 'Task T1', 'status': 'pending'},
                    {'id': '1.2', 'title': 'Task T2', 'status': 'pending'},
                ],
            }]
        }
        # Second call: pre = first_post; Taskmaster adds duplicates again
        second_post: dict = {
            'tasks': [{
                'id': '1', 'title': 'Parent', 'status': 'pending',
                'subtasks': [
                    {'id': '1.1', 'title': 'Task T1', 'status': 'pending'},
                    {'id': '1.2', 'title': 'Task T2', 'status': 'pending'},
                    {'id': '1.3', 'title': 'Task T1', 'status': 'pending'},
                    {'id': '1.4', 'title': 'Task T2', 'status': 'pending'},
                ],
            }]
        }
        # get_tasks call sequence: pre1, post1, pre2 (=first_post), post2
        bulk_taskmaster.get_tasks = AsyncMock(
            side_effect=[first_pre, first_post, first_post, second_post]
        )
        interceptor = TaskInterceptor(bulk_taskmaster, None, event_buffer)

        # First expand — should keep 1.1/1.2, nothing to remove
        r1 = await interceptor.expand_task('1', project_root='/project')
        assert bulk_taskmaster.remove_task.call_count == 0
        assert r1['dedup']['removed'] == []

        bulk_taskmaster.remove_task.reset_mock()

        # Second expand — must remove 1.3 and 1.4 (titles match 1.1/1.2)
        await interceptor.expand_task('1', project_root='/project')
        assert bulk_taskmaster.remove_task.call_count == 2
        removed_ids = {c.args[0] for c in bulk_taskmaster.remove_task.call_args_list}
        assert '1.3' in removed_ids
        assert '1.4' in removed_ids

    # ── step-13 test ─────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_bulk_dedup_tolerates_remove_task_failure(
        self, bulk_taskmaster, event_buffer, pre_snapshot, post_snapshot,
    ):
        """When remove_task raises, expand_task still succeeds and the failure
        is recorded in dedup['errors'] without propagating an exception."""
        bulk_taskmaster.get_tasks = AsyncMock(side_effect=[pre_snapshot, post_snapshot])
        bulk_taskmaster.remove_task = AsyncMock(
            side_effect=RuntimeError('taskmaster busy')
        )
        interceptor = TaskInterceptor(bulk_taskmaster, None, event_buffer)

        result = await interceptor.expand_task('1', project_root='/project')

        # expand_task must not raise
        assert 'dedup' in result
        # The removal was attempted but failed — should be in errors, not removed
        assert result['dedup']['removed'] == []
        assert any(
            e.get('task_id') == '1.2' for e in result['dedup']['errors']
        )

    # ── step-15 test ─────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_bulk_dedup_snapshot_failure_does_not_break_bulk_op(
        self, bulk_taskmaster, event_buffer,
    ):
        """When the pre-snapshot get_tasks call raises, expand_task still
        returns the Taskmaster result and marks dedup as skipped."""
        # First call (pre-snapshot) raises; subsequent calls would work but aren't reached
        bulk_taskmaster.get_tasks = AsyncMock(
            side_effect=RuntimeError('taskmaster unavailable')
        )
        interceptor = TaskInterceptor(bulk_taskmaster, None, event_buffer)

        result = await interceptor.expand_task('1', project_root='/project')

        # Taskmaster expand_task result is preserved
        assert result.get('success') is True
        # Dedup was skipped due to snapshot failure
        assert result['dedup'].get('skipped_reason') == 'pre_snapshot_failed'
        # remove_task was never called
        bulk_taskmaster.remove_task.assert_not_called()


class TestHashPrefixStripping:

    def test_strip_hash_prefix(self):
        raw = '#hash:abc123def45678#[{"location":"foo.rs"}]'
        assert _strip_hash_prefix(raw) == '[{"location":"foo.rs"}]'

    def test_no_prefix_passthrough(self):
        raw = '[{"location":"foo.rs"}]'
        assert _strip_hash_prefix(raw) == raw

    def test_malformed_prefix_passthrough(self):
        raw = '#hash:noclosedhash'
        assert _strip_hash_prefix(raw) == raw
