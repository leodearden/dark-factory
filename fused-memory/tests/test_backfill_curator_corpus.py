"""Tests for TaskCurator.backfill_corpus() and related backfill infrastructure."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fused_memory.config.schema import CuratorConfig, FusedMemoryConfig
from fused_memory.middleware.task_curator import (
    DEFAULT_PRIORITY,
    _PRIORITY_RANK,
    BackfillResult,
    TaskCurator,
)

if TYPE_CHECKING:
    from fused_memory.middleware.task_interceptor import TaskInterceptor


def _make_config() -> FusedMemoryConfig:
    cfg = FusedMemoryConfig()
    cfg.curator = CuratorConfig()
    return cfg


def _make_task(
    task_id: str,
    title: str = 'Fix something',
    description: str = 'A description',
    status: str = 'pending',
    files_to_modify: list[str] | None = None,
) -> dict:
    return {
        'id': task_id,
        'title': title,
        'description': description,
        'status': status,
        'files_to_modify': files_to_modify or [],
    }


def _expected_point_id(project_id: str, task_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f'{project_id}/{task_id}'))


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: backfill_corpus() calls upsert with all tasks
# ──────────────────────────────────────────────────────────────────────────────


class TestBackfillCorpus:
    @pytest.mark.asyncio
    async def test_backfill_upserts_all_tasks(self):
        """backfill_corpus() upserts 3 tasks in a single Qdrant call with correct point IDs."""
        config = _make_config()
        curator = TaskCurator(config=config, taskmaster=None)

        tasks = [
            _make_task('1', title='Task one', description='Desc one', status='pending'),
            _make_task('2', title='Task two', description='Desc two', status='done'),
            _make_task('3', title='Task three', description='Desc three', status='cancelled'),
        ]
        project_id = 'myproject'

        mock_client = AsyncMock()
        mock_embedder = AsyncMock()
        mock_embedder.create = AsyncMock(return_value=[0.1] * 10)

        with patch.object(curator, '_get_qdrant', return_value=mock_client), \
             patch.object(curator, '_get_embedder', return_value=mock_embedder), \
             patch.object(curator, '_ensure_collection', return_value=f'task_curator_{project_id}'):

            await curator.backfill_corpus(tasks, project_id)

        # upsert must have been called exactly once
        mock_client.upsert.assert_called_once()
        call_kwargs = mock_client.upsert.call_args

        # collection name
        assert call_kwargs.kwargs.get('collection_name') == f'task_curator_{project_id}'

        # 3 points
        points = call_kwargs.kwargs.get('points', [])
        assert len(points) == 3

        # IDs must match the deterministic uuid5 scheme
        expected_ids = {_expected_point_id(project_id, str(i)) for i in range(1, 4)}
        actual_ids = {str(p.id) for p in points}
        assert actual_ids == expected_ids

        # Each payload must contain required keys
        for point in points:
            payload = point.payload
            assert 'task_id' in payload
            assert 'title' in payload
            assert 'description' in payload
            assert 'project_id' in payload
            assert 'updated_at' in payload
            assert payload['project_id'] == project_id

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2: backfill_corpus() returns BackfillResult with counts
    # ──────────────────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_backfill_returns_result_counts(self):
        """backfill_corpus() returns BackfillResult(upserted=3, skipped=0, errors=0)."""
        config = _make_config()
        curator = TaskCurator(config=config, taskmaster=None)

        tasks = [
            _make_task('10', title='Alpha'),
            _make_task('11', title='Beta'),
            _make_task('12', title='Gamma'),
        ]
        project_id = 'proj'

        mock_client = AsyncMock()
        mock_embedder = AsyncMock()
        mock_embedder.create = AsyncMock(return_value=[0.1] * 10)

        with patch.object(curator, '_get_qdrant', return_value=mock_client), \
             patch.object(curator, '_get_embedder', return_value=mock_embedder), \
             patch.object(curator, '_ensure_collection', return_value=f'task_curator_{project_id}'):

            result = await curator.backfill_corpus(tasks, project_id)

        assert isinstance(result, BackfillResult)
        assert result.upserted == 3
        assert result.skipped == 0
        assert result.errors == 0


# ──────────────────────────────────────────────────────────────────────────────
# Step 4: edge case tests
# ──────────────────────────────────────────────────────────────────────────────


class TestBackfillEdgeCases:
    @pytest.mark.asyncio
    async def test_backfill_skips_tasks_without_title(self):
        """Tasks with empty/missing title are skipped; BackfillResult.skipped=1."""
        config = _make_config()
        curator = TaskCurator(config=config, taskmaster=None)

        tasks = [
            _make_task('20', title=''),           # empty title — skip
            _make_task('21', title='Has title'),  # good
        ]
        project_id = 'proj'

        mock_client = AsyncMock()
        mock_embedder = AsyncMock()
        mock_embedder.create = AsyncMock(return_value=[0.1] * 10)

        with patch.object(curator, '_get_qdrant', return_value=mock_client), \
             patch.object(curator, '_get_embedder', return_value=mock_embedder), \
             patch.object(curator, '_ensure_collection', return_value=f'task_curator_{project_id}'):

            result = await curator.backfill_corpus(tasks, project_id)

        assert result.upserted == 1
        assert result.skipped == 1
        assert result.errors == 0

        points = mock_client.upsert.call_args.kwargs.get('points', [])
        assert len(points) == 1

    @pytest.mark.asyncio
    async def test_backfill_empty_list(self):
        """Empty input returns BackfillResult(0, 0, 0) without calling Qdrant."""
        config = _make_config()
        curator = TaskCurator(config=config, taskmaster=None)

        mock_client = AsyncMock()
        mock_embedder = AsyncMock()

        with patch.object(curator, '_get_qdrant', return_value=mock_client), \
             patch.object(curator, '_get_embedder', return_value=mock_embedder), \
             patch.object(curator, '_ensure_collection', return_value='task_curator_proj'):

            result = await curator.backfill_corpus([], 'proj')

        assert result == BackfillResult(upserted=0, skipped=0, errors=0)
        mock_client.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_backfill_single_embed_error_continues(self):
        """One task's embedding raising is counted as error; remaining tasks still upserted."""
        config = _make_config()
        curator = TaskCurator(config=config, taskmaster=None)

        tasks = [
            _make_task('30', title='Fails'),
            _make_task('31', title='Succeeds A'),
            _make_task('32', title='Succeeds B'),
        ]
        project_id = 'proj'

        async def maybe_fail(text: str):
            # Fail deterministically based on the task content rather than call
            # order, so this test is not sensitive to asyncio.gather scheduling.
            if 'Fails' in text:
                raise RuntimeError('embed API down')
            return [0.1] * 10

        mock_client = AsyncMock()
        mock_embedder = AsyncMock()
        mock_embedder.create = maybe_fail

        with patch.object(curator, '_get_qdrant', return_value=mock_client), \
             patch.object(curator, '_get_embedder', return_value=mock_embedder), \
             patch.object(curator, '_ensure_collection', return_value=f'task_curator_{project_id}'):

            result = await curator.backfill_corpus(tasks, project_id)

        assert result.errors == 1
        assert result.upserted == 2
        assert result.skipped == 0


# ──────────────────────────────────────────────────────────────────────────────
# Step 6: same point ID as record_task (idempotent overlap)
# ──────────────────────────────────────────────────────────────────────────────


class TestBackfillPointIdConsistency:
    @pytest.mark.asyncio
    async def test_backfill_uses_same_point_id_as_record_task(self):
        """backfill_corpus() and record_task() produce identical UUID5 point IDs."""
        from fused_memory.middleware.task_curator import CandidateTask

        project_id = 'myproject'
        task_id = '42'

        config = _make_config()
        curator = TaskCurator(config=config, taskmaster=None)

        # ---------- record_task path ----------
        record_task_point_id: str | None = None

        async def capture_record_upsert(*, collection_name, points):
            nonlocal record_task_point_id
            record_task_point_id = str(points[0].id)

        mock_client_record = AsyncMock()
        mock_client_record.upsert = capture_record_upsert
        mock_embedder_record = AsyncMock()
        mock_embedder_record.create = AsyncMock(return_value=[0.2] * 10)

        with patch.object(curator, '_get_qdrant', return_value=mock_client_record), \
             patch.object(curator, '_get_embedder', return_value=mock_embedder_record), \
             patch.object(curator, '_ensure_collection', return_value=f'task_curator_{project_id}'):

            await curator.record_task(
                task_id=task_id,
                candidate=CandidateTask(title='Same title', description='same desc'),
                project_id=project_id,
            )

        assert record_task_point_id is not None

        # ---------- backfill_corpus path ----------
        backfill_point_id: str | None = None

        async def capture_backfill_upsert(*, collection_name, points):
            nonlocal backfill_point_id
            backfill_point_id = str(points[0].id)

        mock_client_backfill = AsyncMock()
        mock_client_backfill.upsert = capture_backfill_upsert
        mock_embedder_backfill = AsyncMock()
        mock_embedder_backfill.create = AsyncMock(return_value=[0.2] * 10)

        # Reset lazy-init state so we get fresh mock
        curator._qdrant_client = None
        curator._embedder = None
        curator._initialized_collections = set()

        with patch.object(curator, '_get_qdrant', return_value=mock_client_backfill), \
             patch.object(curator, '_get_embedder', return_value=mock_embedder_backfill), \
             patch.object(curator, '_ensure_collection', return_value=f'task_curator_{project_id}'):

            await curator.backfill_corpus(
                [{'id': task_id, 'title': 'Same title', 'description': 'same desc', 'status': 'pending'}],
                project_id,
            )

        assert backfill_point_id is not None
        assert record_task_point_id == backfill_point_id

    @pytest.mark.asyncio
    async def test_record_task_normalizes_none_fields(self):
        """record_task() with files_to_modify=None and priority=None stores [] and 'medium' in payload."""
        from fused_memory.middleware.task_curator import CandidateTask

        project_id = 'proj'
        task_id = '99'

        config = _make_config()
        curator = TaskCurator(config=config, taskmaster=None)

        captured_payload: dict | None = None

        async def capture_upsert(*, collection_name, points):
            nonlocal captured_payload
            captured_payload = points[0].payload

        mock_client = AsyncMock()
        mock_client.upsert = capture_upsert
        mock_embedder = AsyncMock()
        mock_embedder.create = AsyncMock(return_value=[0.1] * 10)

        candidate = CandidateTask(title='T', description='D')
        # Explicitly override to None, simulating a caller that omits the defaults.
        candidate.files_to_modify = None  # type: ignore[assignment]
        candidate.priority = None  # type: ignore[assignment]

        with patch.object(curator, '_get_qdrant', return_value=mock_client), \
             patch.object(curator, '_get_embedder', return_value=mock_embedder), \
             patch.object(curator, '_ensure_collection', return_value=f'task_curator_{project_id}'):

            await curator.record_task(
                task_id=task_id,
                candidate=candidate,
                project_id=project_id,
            )

        assert captured_payload is not None
        assert captured_payload['files_to_modify'] == [], (
            f"Expected [], got {captured_payload['files_to_modify']!r}"
        )
        assert captured_payload['priority'] == 'medium', (
            f"Expected 'medium', got {captured_payload['priority']!r}"
        )

    def test_default_priority_in_priority_rank(self):
        """DEFAULT_PRIORITY must be a key in _PRIORITY_RANK so sort-key fallback of 99 is never used."""
        assert DEFAULT_PRIORITY in _PRIORITY_RANK, (
            f"DEFAULT_PRIORITY {DEFAULT_PRIORITY!r} is not a key in _PRIORITY_RANK {_PRIORITY_RANK}"
        )

    @pytest.mark.asyncio
    async def test_record_task_and_backfill_payload_shapes_match(self):
        """record_task() and backfill_corpus() produce payloads with identical types."""
        from fused_memory.middleware.task_curator import CandidateTask

        project_id = 'proj'
        task_id = '77'

        config = _make_config()
        curator = TaskCurator(config=config, taskmaster=None)

        # --- record_task path ---
        record_payload: dict | None = None

        async def capture_record(*, collection_name, points):
            nonlocal record_payload
            record_payload = points[0].payload

        mock_client_r = AsyncMock()
        mock_client_r.upsert = capture_record
        mock_embedder_r = AsyncMock()
        mock_embedder_r.create = AsyncMock(return_value=[0.1] * 10)

        candidate = CandidateTask(
            title='My task',
            description='A desc',
            files_to_modify=['a.py'],
            priority='high',
        )

        with patch.object(curator, '_get_qdrant', return_value=mock_client_r), \
             patch.object(curator, '_get_embedder', return_value=mock_embedder_r), \
             patch.object(curator, '_ensure_collection', return_value=f'task_curator_{project_id}'):

            await curator.record_task(task_id=task_id, candidate=candidate, project_id=project_id)

        # --- backfill_corpus path ---
        backfill_payload: dict | None = None

        async def capture_backfill(*, collection_name, points):
            nonlocal backfill_payload
            backfill_payload = points[0].payload

        mock_client_b = AsyncMock()
        mock_client_b.upsert = capture_backfill
        mock_embedder_b = AsyncMock()
        mock_embedder_b.create = AsyncMock(return_value=[0.1] * 10)

        # Reset lazy-init state
        curator._qdrant_client = None
        curator._embedder = None
        curator._initialized_collections = set()

        with patch.object(curator, '_get_qdrant', return_value=mock_client_b), \
             patch.object(curator, '_get_embedder', return_value=mock_embedder_b), \
             patch.object(curator, '_ensure_collection', return_value=f'task_curator_{project_id}'):

            await curator.backfill_corpus(
                [{'id': task_id, 'title': 'My task', 'description': 'A desc',
                  'files_to_modify': ['a.py'], 'priority': 'high', 'status': 'pending'}],
                project_id,
            )

        assert record_payload is not None
        assert backfill_payload is not None

        # Keys must match
        assert set(record_payload.keys()) == set(backfill_payload.keys()), (
            f"Key mismatch: record={set(record_payload)}, backfill={set(backfill_payload)}"
        )

        # files_to_modify must be a list in both
        assert isinstance(record_payload['files_to_modify'], list), (
            f"record_task files_to_modify type: {type(record_payload['files_to_modify'])}"
        )
        assert isinstance(backfill_payload['files_to_modify'], list), (
            f"backfill files_to_modify type: {type(backfill_payload['files_to_modify'])}"
        )

        # priority must be a str in both
        assert isinstance(record_payload['priority'], str), (
            f"record_task priority type: {type(record_payload['priority'])}"
        )
        assert isinstance(backfill_payload['priority'], str), (
            f"backfill priority type: {type(backfill_payload['priority'])}"
        )

        # Key-set equality excluding updated_at (documents intent: updated_at may legitimately differ)
        assert set(record_payload.keys()) - {'updated_at'} == set(backfill_payload.keys()) - {'updated_at'}, (
            f"Key-set mismatch (excluding updated_at): "
            f"record={set(record_payload) - {'updated_at'}}, backfill={set(backfill_payload) - {'updated_at'}}"
        )

        # Value equality for all shared keys except 'updated_at' (may differ by milliseconds)
        shared_keys = set(record_payload.keys()) - {'updated_at'}
        for key in shared_keys:
            assert record_payload[key] == backfill_payload[key], (
                f"Value mismatch for key '{key}': "
                f"record={record_payload[key]!r}, backfill={backfill_payload[key]!r}"
            )

    @pytest.mark.asyncio
    async def test_default_priority_alignment(self):
        """record_task() with default CandidateTask and backfill_corpus() with no priority key both produce 'medium'."""
        from fused_memory.middleware.task_curator import CandidateTask

        project_id = 'proj'
        task_id = '50'

        config = _make_config()
        curator = TaskCurator(config=config, taskmaster=None)

        # --- record_task path: CandidateTask with default priority='medium' ---
        record_payload: dict | None = None

        async def capture_record(*, collection_name, points):
            nonlocal record_payload
            record_payload = points[0].payload

        mock_client_r = AsyncMock()
        mock_client_r.upsert = capture_record
        mock_embedder_r = AsyncMock()
        mock_embedder_r.create = AsyncMock(return_value=[0.1] * 10)

        # Use default priority (CandidateTask defaults to 'medium')
        candidate = CandidateTask(title='Test task', description='Some description')

        with patch.object(curator, '_get_qdrant', return_value=mock_client_r), \
             patch.object(curator, '_get_embedder', return_value=mock_embedder_r), \
             patch.object(curator, '_ensure_collection', return_value=f'task_curator_{project_id}'):

            await curator.record_task(task_id=task_id, candidate=candidate, project_id=project_id)

        assert record_payload is not None

        # --- backfill_corpus path: task dict with no priority key ---
        backfill_payload: dict | None = None

        async def capture_backfill(*, collection_name, points):
            nonlocal backfill_payload
            backfill_payload = points[0].payload

        mock_client_b = AsyncMock()
        mock_client_b.upsert = capture_backfill
        mock_embedder_b = AsyncMock()
        mock_embedder_b.create = AsyncMock(return_value=[0.1] * 10)

        # Reset lazy-init state so we get a fresh mock
        curator._qdrant_client = None
        curator._embedder = None
        curator._initialized_collections = set()

        with patch.object(curator, '_get_qdrant', return_value=mock_client_b), \
             patch.object(curator, '_get_embedder', return_value=mock_embedder_b), \
             patch.object(curator, '_ensure_collection', return_value=f'task_curator_{project_id}'):

            await curator.backfill_corpus(
                [{'id': task_id, 'title': 'Test task', 'description': 'Some description', 'status': 'pending'}],
                project_id,
            )

        assert backfill_payload is not None

        # Both paths must produce DEFAULT_PRIORITY as the canonical default priority
        assert record_payload['priority'] == DEFAULT_PRIORITY, (
            f"record_task default priority: expected {DEFAULT_PRIORITY!r}, got {record_payload['priority']!r}"
        )
        assert backfill_payload['priority'] == DEFAULT_PRIORITY, (
            f"backfill_corpus default priority: expected {DEFAULT_PRIORITY!r}, got {backfill_payload['priority']!r}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Step 8: run_backfill() orchestration
# ──────────────────────────────────────────────────────────────────────────────


class TestRunBackfill:
    @pytest.mark.asyncio
    async def test_run_backfill_orchestrates_correctly(self):
        """run_backfill() fetches tasks, flattens, then calls backfill_corpus()."""
        from fused_memory.maintenance.backfill_curator_corpus import run_backfill

        project_root = '/fake/project'
        project_id = 'project'

        canned_task_tree = {
            'tasks': [
                {'id': '100', 'title': 'Task A', 'description': 'Desc A', 'status': 'done'},
                {
                    'id': '101',
                    'title': 'Task B',
                    'description': 'Desc B',
                    'status': 'pending',
                    'subtasks': [
                        {'id': '101.1', 'title': 'Sub B1', 'description': '', 'status': 'pending'},
                    ],
                },
            ],
        }

        mock_backfill_result = BackfillResult(upserted=3, skipped=0, errors=0)

        mock_taskmaster = AsyncMock()
        mock_taskmaster.get_tasks = AsyncMock(return_value=canned_task_tree)

        mock_curator = AsyncMock()
        mock_curator.backfill_corpus = AsyncMock(return_value=mock_backfill_result)

        mock_config = _make_config()

        import contextlib

        @contextlib.asynccontextmanager
        async def fake_maintenance_service(config_path):
            yield mock_config, MagicMock()

        with patch(
            'fused_memory.maintenance.backfill_curator_corpus.maintenance_service',
            new=fake_maintenance_service,
        ), patch(
            'fused_memory.maintenance.backfill_curator_corpus.TaskmasterBackend',
        ) as mock_tm_cls, patch(
            'fused_memory.maintenance.backfill_curator_corpus.TaskCurator',
        ) as mock_curator_cls:
            mock_tm_instance = AsyncMock()
            mock_tm_instance.get_tasks = AsyncMock(return_value=canned_task_tree)
            mock_tm_cls.return_value = mock_tm_instance

            mock_curator_cls.return_value = mock_curator

            result = await run_backfill(
                config_path=None,
                project_root=project_root,
            )

        # get_tasks called with the project_root
        mock_tm_instance.get_tasks.assert_called_once_with(project_root)

        # backfill_corpus called with flattened list and correct project_id
        mock_curator.backfill_corpus.assert_called_once()
        call_args = mock_curator.backfill_corpus.call_args
        flat_tasks = call_args.args[0] if call_args.args else call_args.kwargs.get('tasks')
        passed_project_id = call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs.get('project_id')

        assert len(flat_tasks) == 3  # parent + parent + subtask
        assert passed_project_id == project_id

        # Result propagated
        assert result.upserted == 3


# ──────────────────────────────────────────────────────────────────────────────
# Steps 10, 11, 13: TaskInterceptor auto-backfill hook
# ──────────────────────────────────────────────────────────────────────────────


def _make_interceptor_config() -> FusedMemoryConfig:
    """Make a config with curator enabled."""
    cfg = _make_config()
    cfg.curator.enabled = True
    return cfg


class TestAutoBackfill:
    """Tests for the auto-backfill hook in TaskInterceptor._maybe_backfill_corpus()."""

    def _make_interceptor(self, mock_taskmaster) -> TaskInterceptor:
        from fused_memory.middleware.task_interceptor import TaskInterceptor
        config = _make_interceptor_config()
        return TaskInterceptor(
            taskmaster=mock_taskmaster,
            targeted_reconciler=None,
            event_buffer=MagicMock(),
            config=config,
        )

    @pytest.mark.asyncio
    async def test_interceptor_triggers_backfill_on_empty_collection(self):
        """_maybe_backfill_corpus() spawns a background backfill when count is 0."""
        import asyncio

        canned_task_tree = {
            'tasks': [
                {'id': '1', 'title': 'Task A', 'description': 'Desc A', 'status': 'done'},
            ],
        }
        mock_taskmaster = AsyncMock()
        mock_taskmaster.get_tasks = AsyncMock(return_value=canned_task_tree)

        interceptor = self._make_interceptor(mock_taskmaster)

        config = _make_interceptor_config()
        curator = TaskCurator(config=config, taskmaster=mock_taskmaster)

        # count() returns 0 → trigger backfill
        from qdrant_client.http.models import CountResult
        mock_qdrant = AsyncMock()
        mock_qdrant.collection_exists = AsyncMock(return_value=True)
        mock_qdrant.count = AsyncMock(return_value=CountResult(count=0))

        backfill_called_with: list[tuple] = []

        async def mock_backfill_corpus(tasks, project_id):
            backfill_called_with.append((tasks, project_id))
            return BackfillResult(upserted=len(tasks))

        with patch.object(curator, '_get_qdrant', AsyncMock(return_value=mock_qdrant)), \
             patch.object(curator, 'backfill_corpus', side_effect=mock_backfill_corpus):

            await interceptor._maybe_backfill_corpus(curator, project_root='/fake/project')

            # Give the background task a moment to run
            await asyncio.sleep(0.05)

        assert len(backfill_called_with) == 1
        tasks_arg, project_id_arg = backfill_called_with[0]
        assert len(tasks_arg) == 1
        assert project_id_arg == 'project'

    @pytest.mark.asyncio
    async def test_interceptor_skips_backfill_when_corpus_populated(self):
        """_maybe_backfill_corpus() does NOT call backfill_corpus() when count > 0."""
        import asyncio

        mock_taskmaster = AsyncMock()
        interceptor = self._make_interceptor(mock_taskmaster)

        config = _make_interceptor_config()
        curator = TaskCurator(config=config, taskmaster=mock_taskmaster)

        from qdrant_client.http.models import CountResult
        mock_qdrant = AsyncMock()
        mock_qdrant.collection_exists = AsyncMock(return_value=True)
        mock_qdrant.count = AsyncMock(return_value=CountResult(count=5))

        backfill_called = False

        async def mock_backfill_corpus(tasks, project_id):
            nonlocal backfill_called
            backfill_called = True
            return BackfillResult()

        with patch.object(curator, '_get_qdrant', AsyncMock(return_value=mock_qdrant)), \
             patch.object(curator, 'backfill_corpus', side_effect=mock_backfill_corpus):

            await interceptor._maybe_backfill_corpus(curator, project_root='/fake/project')
            await asyncio.sleep(0.05)

        assert not backfill_called

    @pytest.mark.asyncio
    async def test_backfill_failure_does_not_block_curator(self):
        """_maybe_backfill_corpus() failing silently allows _get_curator() to return."""
        import asyncio

        canned_task_tree = {'tasks': [{'id': '1', 'title': 'T'}]}
        mock_taskmaster = AsyncMock()
        mock_taskmaster.get_tasks = AsyncMock(return_value=canned_task_tree)

        interceptor = self._make_interceptor(mock_taskmaster)

        config = _make_interceptor_config()
        curator = TaskCurator(config=config, taskmaster=mock_taskmaster)

        from qdrant_client.http.models import CountResult
        mock_qdrant = AsyncMock()
        mock_qdrant.collection_exists = AsyncMock(return_value=True)
        mock_qdrant.count = AsyncMock(return_value=CountResult(count=0))

        async def failing_backfill(tasks, project_id):
            raise RuntimeError('embed is down')

        with patch.object(curator, '_get_qdrant', AsyncMock(return_value=mock_qdrant)), \
             patch.object(curator, 'backfill_corpus', side_effect=failing_backfill):

            # Must not raise — graceful degradation
            await interceptor._maybe_backfill_corpus(curator, project_root='/fake/project')
            await asyncio.sleep(0.05)  # let background task settle

    @pytest.mark.asyncio
    async def test_get_curator_triggers_backfill_on_first_call(self):
        """_get_curator() fires _maybe_backfill_corpus() as a background task on first construction.

        This integration test exercises the full _get_curator → _maybe_backfill_corpus
        path, confirming that the wiring is live (not just that _maybe_backfill_corpus
        works in isolation).
        """
        import asyncio

        canned_task_tree = {
            'tasks': [
                {'id': '1', 'title': 'Task A', 'description': 'Desc A', 'status': 'done'},
            ],
        }
        mock_taskmaster = AsyncMock()
        # Provide project_root so _get_curator() extracts it and calls _maybe_backfill_corpus.
        mock_taskmaster.config = MagicMock()
        mock_taskmaster.config.project_root = '/fake/project'
        mock_taskmaster.get_tasks = AsyncMock(return_value=canned_task_tree)

        interceptor = self._make_interceptor(mock_taskmaster)

        backfill_called_with: list[tuple] = []

        async def mock_backfill(tasks, project_id):
            backfill_called_with.append((tasks, project_id))
            return BackfillResult(upserted=len(tasks))

        with patch('fused_memory.middleware.task_interceptor.TaskCurator') as mock_curator_cls:
            mock_curator_instance = AsyncMock()
            mock_curator_instance.corpus_count = AsyncMock(return_value=0)
            mock_curator_instance.backfill_corpus = mock_backfill
            mock_curator_cls.return_value = mock_curator_instance

            curator = await interceptor._get_curator()

            # Allow the background tasks (backfill check + backfill) to complete.
            await asyncio.sleep(0.1)

        assert curator is mock_curator_instance
        assert len(backfill_called_with) == 1
        tasks_arg, project_id_arg = backfill_called_with[0]
        assert len(tasks_arg) == 1
        assert project_id_arg == 'project'
