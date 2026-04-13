"""Tests for TaskCurator.backfill_corpus() and related backfill infrastructure."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fused_memory.config.schema import CuratorConfig, FusedMemoryConfig
from fused_memory.middleware.task_curator import (
    BackfillResult,
    TaskCurator,
    _flatten_task_tree,
)


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

            result = await curator.backfill_corpus(tasks, project_id)

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

        call_count = 0

        async def maybe_fail(text: str):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
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
