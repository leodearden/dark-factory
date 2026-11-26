"""Tests for TaskCurator.backfill_corpus() and related backfill infrastructure."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import uuid
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from _curator_helpers import curator_double, in_memory_corpus_curator, make_config
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import PointsList, UpdateResult

from fused_memory.config.schema import CuratorConfig, FusedMemoryConfig
from fused_memory.middleware.curator_types import DEFAULT_PRIORITY
from fused_memory.middleware.task_curator import (
    _PRIORITY_RANK,
    BACKFILL_DEFAULT_CONCURRENCY,
    BACKFILL_UPSERT_BATCH_SIZE,
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


QDRANT_DEFAULT_MAX_REQUEST_BYTES = 32 * 2**20
"""Qdrant's default ``service.max_request_size_mb``, in bytes."""

_CURATOR_LOGGER = 'fused_memory.middleware.task_curator'


class RecordingQdrant(AsyncQdrantClient):
    """An in-memory Qdrant that records each upsert's point count and refuses some upserts.

    An upsert is refused with the 400 Qdrant answers to an oversized body when
    its serialised body exceeds *max_request_bytes*, or when its 0-based call
    index is in *fail_calls*. A refused upsert writes nothing.
    """

    def __init__(
        self,
        *,
        max_request_bytes: int | None = None,
        fail_calls: frozenset[int] = frozenset(),
    ) -> None:
        super().__init__(location=':memory:')
        self._max_request_bytes = max_request_bytes
        self._fail_calls = fail_calls
        self.upsert_sizes: list[int] = []

    async def upsert(
        self, collection_name: str, points: Any, *args: Any, **kwargs: Any,
    ) -> UpdateResult:
        call_index = len(self.upsert_sizes)
        point_list = list(points)
        self.upsert_sizes.append(len(point_list))
        body_bytes = len(PointsList(points=point_list).model_dump_json())
        oversized = self._max_request_bytes is not None and body_bytes > self._max_request_bytes
        if oversized or call_index in self._fail_calls:
            raise UnexpectedResponse(
                status_code=400,
                reason_phrase='Bad Request',
                content=f'JSON payload ({body_bytes} bytes) refused'.encode(),
                headers=httpx.Headers(),
            )
        return await super().upsert(collection_name, point_list, *args, **kwargs)


class RecordingEmbedder:
    """A corpus embedder that counts its calls and the most calls it ever had in flight.

    Each call yields to the event loop once while in flight, so concurrent
    calls overlap with no wall-clock wait. A text containing *fail_marker*
    raises. A text containing *abort_marker* raises ``asyncio.CancelledError``,
    as an operator's Ctrl-C under ``asyncio.run`` does. Every component has a
    worst-case-length float repr, so a point serialises as large as a real
    embedding's can.
    """

    def __init__(
        self, *, fail_marker: str | None = None, abort_marker: str | None = None,
    ) -> None:
        self._vector = [-0.0123456789012345] * make_config().embedder.dimensions
        self._fail_marker = fail_marker
        self._abort_marker = abort_marker
        self.calls = 0
        self.in_flight = 0
        self.peak = 0

    async def create(self, input_data: str) -> list[float]:
        self.calls += 1
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(0)
            if self._abort_marker is not None and self._abort_marker in input_data:
                raise asyncio.CancelledError(f'run interrupted at {self._abort_marker!r}')
            if self._fail_marker is not None and self._fail_marker in input_data:
                raise RuntimeError(f'embedder refused a text containing {self._fail_marker!r}')
            return self._vector
        finally:
            self.in_flight -= 1


def _titled_tasks(count: int, *, title_chars: int = 20, description_chars: int = 20) -> list[dict]:
    """*count* titled tasks with distinct ids and padded title and description texts."""
    return [
        _make_task(
            str(i),
            title=f'Task {i} '.ljust(title_chars, 't'),
            description=f'Description {i} '.ljust(description_chars, 'd'),
        )
        for i in range(count)
    ]


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


class TestBackfillBatching:
    @pytest.mark.asyncio
    async def test_project_over_qdrant_request_limit_lands_in_bounded_batches(self):
        """1,200 worst-case points are about 36.8 MiB as one body, over Qdrant's 32 MiB limit."""
        qdrant = RecordingQdrant(max_request_bytes=QDRANT_DEFAULT_MAX_REQUEST_BYTES)
        curator = in_memory_corpus_curator(RecordingEmbedder(), qdrant_client=qdrant)
        tasks = _titled_tasks(1200, title_chars=200, description_chars=1000)
        try:
            result = await curator.backfill_corpus(tasks, 'proj')

            assert result == BackfillResult(upserted=1200, skipped=0, errors=0)
            assert await curator.corpus_count('proj') == 1200
            assert len(qdrant.upsert_sizes) > 1
            assert max(qdrant.upsert_sizes) <= BACKFILL_UPSERT_BATCH_SIZE
        finally:
            await curator.close()

    @staticmethod
    async def _backfill_in_threes(
        tasks: list[dict], embedder: RecordingEmbedder, *, fail_calls: frozenset[int],
    ) -> tuple[BackfillResult, int, RecordingQdrant]:
        """Backfill *tasks* 3 points a batch, refusing the listed upsert calls.

        Returns the result, the persisted corpus count and the Qdrant double.
        """
        qdrant = RecordingQdrant(fail_calls=fail_calls)
        curator = in_memory_corpus_curator(embedder, qdrant_client=qdrant)
        try:
            result = await curator.backfill_corpus(tasks, 'proj', batch_size=3)
            return result, await curator.corpus_count('proj'), qdrant
        finally:
            await curator.close()

    @pytest.mark.asyncio
    async def test_refused_middle_batch_costs_only_its_own_points(self):
        embedder = RecordingEmbedder()
        result, persisted, qdrant = await self._backfill_in_threes(
            _titled_tasks(7), embedder, fail_calls=frozenset({1}),
        )

        assert result == BackfillResult(upserted=4, skipped=0, errors=3)
        assert persisted == 4
        assert qdrant.upsert_sizes == [3, 3, 1]
        assert embedder.calls == 7

    @pytest.mark.asyncio
    async def test_run_interrupted_in_the_last_batch_keeps_the_batches_before_it(self):
        tasks = [*_titled_tasks(6), _make_task('abort', title='ABORT-RUN task')]
        qdrant = RecordingQdrant()
        curator = in_memory_corpus_curator(
            RecordingEmbedder(abort_marker='ABORT-RUN'), qdrant_client=qdrant,
        )
        try:
            with pytest.raises(asyncio.CancelledError):
                await curator.backfill_corpus(tasks, 'proj', batch_size=3)

            assert await curator.corpus_count('proj') == 6
            assert qdrant.upsert_sizes == [3, 3]
        finally:
            await curator.close()

    @pytest.mark.asyncio
    async def test_batch_failure_counts_compose_with_skips_and_embed_failures(self):
        tasks = [
            _make_task('untitled', title=''),
            _make_task('refused', title='EMBED-REFUSED task'),
            *_titled_tasks(5),
        ]
        result, persisted, qdrant = await self._backfill_in_threes(
            tasks, RecordingEmbedder(fail_marker='EMBED-REFUSED'), fail_calls=frozenset({0}),
        )

        assert result.skipped == 1
        assert result.upserted + result.errors == 6
        assert result.errors == 1 + qdrant.upsert_sizes[0]
        assert persisted == result.upserted

    @pytest.mark.asyncio
    async def test_refused_batch_logs_one_warning_naming_it(self, caplog):
        with caplog.at_level(logging.WARNING, logger=_CURATOR_LOGGER):
            await self._backfill_in_threes(
                _titled_tasks(7), RecordingEmbedder(), fail_calls=frozenset({1}),
            )

        warnings = [
            record for record in caplog.records
            if record.name == _CURATOR_LOGGER and record.levelno == logging.WARNING
        ]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert 'project=proj' in message
        assert 'batch 2/3' in message
        assert '3 points' in message
        assert warnings[0].exc_info is not None


class TestBackfillConcurrency:
    @staticmethod
    async def _backfill(
        embedder: RecordingEmbedder, tasks: list[dict], **knobs: int,
    ) -> BackfillResult:
        curator = in_memory_corpus_curator(embedder)
        try:
            return await curator.backfill_corpus(tasks, 'proj', **knobs)
        finally:
            await curator.close()

    @pytest.mark.asyncio
    async def test_peak_in_flight_embeds_equal_the_concurrency(self):
        embedder = RecordingEmbedder()
        result = await self._backfill(embedder, _titled_tasks(10), concurrency=3)

        assert embedder.peak == 3
        assert result.upserted == 10

    @pytest.mark.asyncio
    async def test_default_concurrency_is_ten(self):
        embedder = RecordingEmbedder()
        await self._backfill(embedder, _titled_tasks(12))

        assert BACKFILL_DEFAULT_CONCURRENCY == 10
        assert embedder.peak == BACKFILL_DEFAULT_CONCURRENCY

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('knob', 'value'), [('concurrency', 0), ('concurrency', -1), ('batch_size', 0)],
    )
    async def test_non_positive_knob_is_refused_before_any_embed(self, knob: str, value: int):
        embedder = RecordingEmbedder()
        with pytest.raises(ValueError, match=f'{knob}.*{re.escape(str(value))}'):
            await self._backfill(embedder, _titled_tasks(3), **{knob: value})

        assert embedder.calls == 0


# ──────────────────────────────────────────────────────────────────────────────
# Step 6: same point ID as record_task (idempotent overlap)
# ──────────────────────────────────────────────────────────────────────────────


class TestBackfillPointIdConsistency:
    @pytest.mark.asyncio
    async def test_backfill_uses_same_point_id_as_record_task(self):
        """backfill_corpus() and record_task() produce identical UUID5 point IDs."""
        from fused_memory.middleware.curator_types import CandidateTask

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
        from fused_memory.middleware.curator_types import CandidateTask

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
        assert captured_payload['priority'] == DEFAULT_PRIORITY, (
            f"Expected {DEFAULT_PRIORITY!r}, got {captured_payload['priority']!r}"
        )

    def test_default_priority_in_priority_rank(self):
        """DEFAULT_PRIORITY must be a key in _PRIORITY_RANK so sort-key fallback of 99 is never used.

        NOTE: This test is intentionally redundant with the module-level ValueError guard in
        task_curator.py.  If that guard fires, the module won't import and *all* tests in this
        file will fail before this assertion is ever reached.  The test is kept for documentation
        purposes — it makes the invariant explicit and searchable in the test suite.
        """
        assert DEFAULT_PRIORITY in _PRIORITY_RANK, (
            f"DEFAULT_PRIORITY {DEFAULT_PRIORITY!r} is not a key in _PRIORITY_RANK {_PRIORITY_RANK}"
        )

    @pytest.mark.asyncio
    async def test_record_task_and_backfill_payload_shapes_match(self):
        """record_task() and backfill_corpus() produce payloads with identical types."""
        from fused_memory.middleware.curator_types import CandidateTask

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

        # Both payloads must always include updated_at
        assert 'updated_at' in record_payload, f"record_payload missing 'updated_at': {set(record_payload)}"
        assert 'updated_at' in backfill_payload, f"backfill_payload missing 'updated_at': {set(backfill_payload)}"

        # Key-set equality excluding updated_at (may legitimately differ)
        assert set(record_payload.keys()) - {'updated_at'} == set(backfill_payload.keys()) - {'updated_at'}, (
            f"Key-set mismatch (excluding updated_at): "
            f"record={set(record_payload) - {'updated_at'}}, backfill={set(backfill_payload) - {'updated_at'}}"
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
        from fused_memory.middleware.curator_types import CandidateTask

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


class TestBackfillManager:
    @pytest.mark.asyncio
    async def test_backfill_carries_concurrency_to_the_curator(self):
        from fused_memory.maintenance.backfill_curator_corpus import BackfillManager

        taskmaster = AsyncMock()
        taskmaster.get_tasks = AsyncMock(return_value={'tasks': _titled_tasks(8)})
        embedder = RecordingEmbedder()
        curator = in_memory_corpus_curator(embedder)
        manager = BackfillManager(config=make_config(), taskmaster=taskmaster, curator=curator)
        try:
            report = await manager.backfill('/fake/proj', concurrency=2)

            assert embedder.peak == 2
            assert report.upserted == 8
            assert report.errors == 0
            assert await curator.corpus_count(report.project_id) == 8
        finally:
            await curator.close()


# ──────────────────────────────────────────────────────────────────────────────
# Step 8: run_backfill() orchestration
# ──────────────────────────────────────────────────────────────────────────────


class TestRunBackfill:
    @contextlib.contextmanager
    def _run_backfill_context(self, *, assert_close: bool = True):
        """Yield ``(mock_curator, mock_tm_cls)`` with the common patch stack applied.

        Patches applied: ``maintenance_service``, ``SqliteTaskBackend``, ``TaskCurator``.
        ``TaskCurator`` is pre-wired to return ``mock_curator``; tests configure
        ``mock_tm_cls.return_value`` themselves.

        After the ``with`` block exits, asserts ``curator.close()`` was awaited
        exactly once unless *assert_close* is ``False``.
        """
        mock_config = _make_config()
        mock_curator = AsyncMock()

        @contextlib.asynccontextmanager
        async def fake_maintenance_service(config_path):
            yield mock_config, MagicMock()

        with patch(
            'fused_memory.maintenance.backfill_curator_corpus.maintenance_service',
            new=fake_maintenance_service,
        ), patch(
            'fused_memory.maintenance.backfill_curator_corpus.SqliteTaskBackend',
        ) as mock_tm_cls, patch(
            'fused_memory.maintenance.backfill_curator_corpus.TaskCurator',
        ) as mock_curator_cls:
            mock_curator_cls.return_value = mock_curator
            yield mock_curator, mock_tm_cls

        if assert_close:
            mock_curator.close.assert_awaited_once()

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

        with self._run_backfill_context() as (mock_curator, mock_tm_cls):
            mock_curator.backfill_corpus = AsyncMock(return_value=mock_backfill_result)
            mock_tm_instance = AsyncMock()
            mock_tm_instance.get_tasks = AsyncMock(return_value=canned_task_tree)
            mock_tm_cls.return_value = mock_tm_instance

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

    @pytest.mark.asyncio
    async def test_run_backfill_passes_concurrency_to_curator(self):
        from fused_memory.maintenance.backfill_curator_corpus import run_backfill

        with self._run_backfill_context() as (mock_curator, mock_tm_cls):
            mock_curator.backfill_corpus = AsyncMock(return_value=BackfillResult(upserted=1))
            mock_tm_instance = AsyncMock()
            mock_tm_instance.get_tasks = AsyncMock(return_value={'tasks': _titled_tasks(1)})
            mock_tm_cls.return_value = mock_tm_instance

            await run_backfill(config_path=None, project_root='/fake/project', concurrency=4)

        mock_curator.backfill_corpus.assert_awaited_once()
        assert mock_curator.backfill_corpus.call_args.kwargs['concurrency'] == 4

    @pytest.mark.asyncio
    async def test_run_backfill_closes_curator(self):
        """run_backfill() calls curator.close() after a successful backfill."""
        from fused_memory.maintenance.backfill_curator_corpus import run_backfill

        project_root = '/fake/project'
        canned_task_tree = {
            'tasks': [
                {'id': '100', 'title': 'Task A', 'description': 'Desc A', 'status': 'done'},
            ],
        }
        mock_backfill_result = BackfillResult(upserted=1, skipped=0, errors=0)

        with self._run_backfill_context() as (mock_curator, mock_tm_cls):
            mock_curator.backfill_corpus = AsyncMock(return_value=mock_backfill_result)
            mock_tm_instance = AsyncMock()
            mock_tm_instance.get_tasks = AsyncMock(return_value=canned_task_tree)
            mock_tm_cls.return_value = mock_tm_instance

            await run_backfill(config_path=None, project_root=project_root)

        mock_curator.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_backfill_closes_curator_on_error(self):
        """run_backfill() calls curator.close() even when manager.backfill() raises.

        The RuntimeError must propagate to the caller after close() is called.
        """
        from fused_memory.maintenance.backfill_curator_corpus import run_backfill

        project_root = '/fake/project'

        with self._run_backfill_context() as (mock_curator, mock_tm_cls), patch(
            'fused_memory.maintenance.backfill_curator_corpus.BackfillManager',
        ) as mock_manager_cls:
            mock_tm_cls.return_value = AsyncMock()
            mock_manager_instance = AsyncMock()
            mock_manager_instance.backfill = AsyncMock(side_effect=RuntimeError('backfill failed'))
            mock_manager_cls.return_value = mock_manager_instance

            with pytest.raises(RuntimeError, match='backfill failed'):
                await run_backfill(config_path=None, project_root=project_root)

        mock_curator.close.assert_awaited_once()


class TestBackfillCli:
    def test_concurrency_defaults_to_the_curator_default(self):
        from fused_memory.maintenance.backfill_curator_corpus import build_parser

        args = build_parser().parse_args(['--project-root', '/x'])

        assert args.concurrency == BACKFILL_DEFAULT_CONCURRENCY

    def test_concurrency_is_parsed_as_an_int(self):
        from fused_memory.maintenance.backfill_curator_corpus import build_parser

        args = build_parser().parse_args(['--project-root', '/x', '--concurrency', '4'])

        assert args.concurrency == 4

    @pytest.mark.parametrize('text', ['0', '-1', 'abc'])
    def test_non_positive_or_non_int_concurrency_is_a_usage_error(self, text: str, capsys):
        from fused_memory.maintenance.backfill_curator_corpus import build_parser

        with pytest.raises(SystemExit) as exit_info:
            build_parser().parse_args(['--project-root', '/x', '--concurrency', text])

        assert exit_info.value.code == 2
        assert '--concurrency' in capsys.readouterr().err

    def test_main_runs_the_backfill_with_the_parsed_concurrency(self, monkeypatch, capsys):
        from fused_memory.maintenance import backfill_curator_corpus

        received: list[dict[str, Any]] = []

        async def fake_run_backfill(**kwargs: Any) -> BackfillResult:
            received.append(kwargs)
            return BackfillResult(upserted=5, skipped=1, errors=0)

        monkeypatch.setattr(backfill_curator_corpus, 'run_backfill', fake_run_backfill)

        backfill_curator_corpus.main(['--project-root', '/x', '--concurrency', '3'])

        assert received == [{'config_path': None, 'project_root': '/x', 'concurrency': 3}]
        assert 'Backfill complete: upserted=5 skipped=1 errors=0' in capsys.readouterr().out


# ──────────────────────────────────────────────────────────────────────────────
# TestPruneDriver: BackfillManager.prune() / run_prune() orchestration
# (curator corpus RC1 — task 2520, Layer C driver)
# ──────────────────────────────────────────────────────────────────────────────


class TestPruneDriver:
    @pytest.mark.asyncio
    async def test_run_prune_reads_snapshot_and_prunes(self):
        """run_prune() reads the live task snapshot, flattens it (including
        subtasks), and calls curator.prune_orphans() with the project's live
        task-id set."""
        from fused_memory.maintenance.backfill_curator_corpus import run_prune
        from fused_memory.middleware.task_curator import PruneResult

        project_root = '/fake/project'
        project_id = 'project'

        canned_task_tree = {
            'tasks': [
                {'id': '1', 'title': 'Task A', 'description': 'Desc A', 'status': 'done'},
                {
                    'id': '2',
                    'title': 'Task B',
                    'description': 'Desc B',
                    'status': 'pending',
                    'subtasks': [
                        {'id': '2.1', 'title': 'Sub B1', 'description': '', 'status': 'pending'},
                    ],
                },
            ],
        }
        mock_prune_result = PruneResult(pruned=1, live=3)

        with TestRunBackfill()._run_backfill_context() as (mock_curator, mock_tm_cls):
            mock_curator.prune_orphans = AsyncMock(return_value=mock_prune_result)
            mock_tm_instance = AsyncMock()
            mock_tm_instance.list_tags = AsyncMock(return_value=['master'])
            mock_tm_instance.get_tasks = AsyncMock(return_value=canned_task_tree)
            mock_tm_cls.return_value = mock_tm_instance

            await run_prune(config_path=None, project_root=project_root)

        mock_tm_instance.get_tasks.assert_called_once_with(project_root, tag='master')

        mock_curator.prune_orphans.assert_called_once()
        call_args = mock_curator.prune_orphans.call_args
        passed_project_id = (
            call_args.args[0] if call_args.args else call_args.kwargs.get('project_id')
        )
        passed_live_ids = (
            call_args.args[1]
            if len(call_args.args) > 1
            else call_args.kwargs.get('live_task_ids')
        )

        assert passed_project_id == project_id
        assert set(passed_live_ids) == {'1', '2', '2.1'}

    @pytest.mark.asyncio
    async def test_run_prune_skips_on_read_failure(self):
        """A failed live-task read must never be treated as a legitimate empty
        set — run_prune() must skip the sweep entirely rather than calling
        prune_orphans() (which would otherwise diff against nothing and
        orphan every corpus point)."""
        from fused_memory.maintenance.backfill_curator_corpus import run_prune

        project_root = '/fake/project'

        with TestRunBackfill()._run_backfill_context() as (mock_curator, mock_tm_cls):
            mock_curator.prune_orphans = AsyncMock()
            mock_tm_instance = AsyncMock()
            mock_tm_instance.list_tags = AsyncMock(return_value=['master'])
            mock_tm_instance.get_tasks = AsyncMock(side_effect=RuntimeError('db read failed'))
            mock_tm_cls.return_value = mock_tm_instance

            report = await run_prune(config_path=None, project_root=project_root)

        mock_curator.prune_orphans.assert_not_called()
        assert report.skipped is True
        assert report.reason == 'read_failed'

    @pytest.mark.asyncio
    async def test_run_prune_aggregates_live_ids_across_all_tags(self):
        """run_prune() enumerates every tag via list_tags() and unions
        get_tasks(project_root, tag=t) across ALL of them before diffing
        against the corpus — a task filed under a non-default tag must not
        be false-pruned just because the snapshot only ever looked at
        'master' (task 2603)."""
        from fused_memory.maintenance.backfill_curator_corpus import run_prune
        from fused_memory.middleware.task_curator import PruneResult

        project_root = '/fake/project'
        project_id = 'project'

        tag_trees = {
            'master': {
                'tasks': [
                    {'id': '1', 'title': 'Task A', 'description': 'Desc A', 'status': 'done'},
                    {'id': '2', 'title': 'Task B', 'description': 'Desc B', 'status': 'pending'},
                ],
            },
            'feature-x': {
                'tasks': [
                    {'id': '2.1', 'title': 'Task C', 'description': 'Desc C', 'status': 'pending'},
                    {'id': '9', 'title': 'Task D', 'description': 'Desc D', 'status': 'done'},
                ],
            },
        }

        async def _get_tasks_side_effect(_project_root, tag=None):
            return tag_trees[tag or 'master']

        mock_prune_result = PruneResult(pruned=0, live=4)

        with TestRunBackfill()._run_backfill_context() as (mock_curator, mock_tm_cls):
            mock_curator.prune_orphans = AsyncMock(return_value=mock_prune_result)
            mock_tm_instance = AsyncMock()
            mock_tm_instance.list_tags = AsyncMock(return_value=['master', 'feature-x'])
            mock_tm_instance.get_tasks = AsyncMock(side_effect=_get_tasks_side_effect)
            mock_tm_cls.return_value = mock_tm_instance

            await run_prune(config_path=None, project_root=project_root)

        mock_tm_instance.list_tags.assert_called_once_with(project_root)

        mock_curator.prune_orphans.assert_called_once()
        call_args = mock_curator.prune_orphans.call_args
        passed_project_id = (
            call_args.args[0] if call_args.args else call_args.kwargs.get('project_id')
        )
        passed_live_ids = (
            call_args.args[1]
            if len(call_args.args) > 1
            else call_args.kwargs.get('live_task_ids')
        )

        assert passed_project_id == project_id
        assert set(passed_live_ids) == {'1', '2', '2.1', '9'}

    @pytest.mark.asyncio
    async def test_run_prune_skips_when_list_tags_fails(self):
        """A list_tags() failure must skip the whole sweep — get_tasks is
        wired to a valid tree so ONLY the list_tags failure can be
        responsible for the skip (task 2603)."""
        from fused_memory.maintenance.backfill_curator_corpus import run_prune

        project_root = '/fake/project'
        canned_task_tree = {
            'tasks': [{'id': '1', 'title': 'Task A', 'description': 'Desc A', 'status': 'done'}],
        }

        with TestRunBackfill()._run_backfill_context() as (mock_curator, mock_tm_cls):
            mock_curator.prune_orphans = AsyncMock()
            mock_tm_instance = AsyncMock()
            mock_tm_instance.list_tags = AsyncMock(side_effect=RuntimeError('list_tags failed'))
            mock_tm_instance.get_tasks = AsyncMock(return_value=canned_task_tree)
            mock_tm_cls.return_value = mock_tm_instance

            report = await run_prune(config_path=None, project_root=project_root)

        mock_curator.prune_orphans.assert_not_called()
        assert report.skipped is True
        assert report.reason == 'read_failed'

    @pytest.mark.asyncio
    async def test_run_prune_skips_when_any_tag_read_fails(self):
        """list_tags() succeeds but a single tag's get_tasks() raises — the
        sweep must skip entirely rather than prune against only the tags it
        managed to read; a partial snapshot must never prune (task 2603)."""
        from fused_memory.maintenance.backfill_curator_corpus import run_prune

        project_root = '/fake/project'
        canned_master_tree = {
            'tasks': [{'id': '1', 'title': 'Task A', 'description': 'Desc A', 'status': 'done'}],
        }

        async def _get_tasks_side_effect(_project_root, tag=None):
            effective_tag = tag or 'master'
            if effective_tag == 'master':
                return canned_master_tree
            raise RuntimeError(f'read failed for tag {effective_tag!r}')

        with TestRunBackfill()._run_backfill_context() as (mock_curator, mock_tm_cls):
            mock_curator.prune_orphans = AsyncMock()
            mock_tm_instance = AsyncMock()
            mock_tm_instance.list_tags = AsyncMock(return_value=['master', 'feature-x'])
            mock_tm_instance.get_tasks = AsyncMock(side_effect=_get_tasks_side_effect)
            mock_tm_cls.return_value = mock_tm_instance

            report = await run_prune(config_path=None, project_root=project_root)

        mock_curator.prune_orphans.assert_not_called()
        assert report.skipped is True
        assert report.reason == 'read_failed'


# ──────────────────────────────────────────────────────────────────────────────
# Steps 10, 11, 13: TaskInterceptor auto-backfill hook
# ──────────────────────────────────────────────────────────────────────────────


def _make_interceptor_config(project_root: str | None = None) -> FusedMemoryConfig:
    """Make a config with curator enabled.

    ``project_root`` is now read by the interceptor from its OWN config
    (post-Commit 1 fix to stop reaching into ``taskmaster.config``), so
    callers that need the curator's backfill path must wire it here.
    """
    from fused_memory.config.schema import TaskmasterConfig

    cfg = _make_config()
    cfg.curator.enabled = True
    if project_root is not None:
        cfg.taskmaster = TaskmasterConfig(project_root=project_root)
    return cfg


class TestAutoBackfill:
    """Tests for the auto-backfill hook in TaskInterceptor._maybe_backfill_corpus()."""

    def _make_interceptor(
        self, mock_taskmaster, project_root: str | None = None,
    ) -> TaskInterceptor:
        from fused_memory.middleware.task_interceptor import TaskInterceptor
        # ``project_root`` is now read from the interceptor's own config
        # (Commit 1 fix); pass it through here when a backfill path needs it.
        config = _make_interceptor_config(project_root=project_root)
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
    async def test_maybe_backfill_awaits_inline_no_nested_task(self):
        """_maybe_backfill_corpus() awaits backfill_corpus() inline — no nested background task.

        After the call returns (no sleep), backfill_corpus must have been called and
        interceptor._background_tasks must be empty (no curator-auto-backfill task was spawned).
        """
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

        backfill_called_with: list[tuple] = []

        async def mock_backfill_corpus(tasks, project_id):
            backfill_called_with.append((tasks, project_id))
            return BackfillResult(upserted=len(tasks))

        with patch.object(curator, 'corpus_count', AsyncMock(return_value=0)), \
             patch.object(curator, 'backfill_corpus', side_effect=mock_backfill_corpus):

            await interceptor._maybe_backfill_corpus(curator, project_root='/fake/project')
            # No sleep — backfill must complete inline (direct await, not nested create_task)

        # backfill_corpus was called immediately (inline await, not deferred)
        assert len(backfill_called_with) == 1
        tasks_arg, project_id_arg = backfill_called_with[0]
        assert len(tasks_arg) == 1
        assert project_id_arg == 'project'
        # No nested background tasks were spawned
        assert len(interceptor._background_tasks) == 0

    @pytest.mark.asyncio
    async def test_get_curator_triggers_backfill_on_first_call(self):
        """_get_curator() fires _maybe_backfill_corpus() as a background task on first construction.

        This integration test exercises the full _get_curator → _maybe_backfill_corpus
        path, confirming that the wiring is live (not just that _maybe_backfill_corpus
        works in isolation). It calls the private ``_get_curator`` directly because
        its lazy build is the subject.
        """
        import asyncio

        canned_task_tree = {
            'tasks': [
                {'id': '1', 'title': 'Task A', 'description': 'Desc A', 'status': 'done'},
            ],
        }
        mock_taskmaster = AsyncMock()
        mock_taskmaster.get_tasks = AsyncMock(return_value=canned_task_tree)
        # Project_root now lives on the interceptor's own config — see
        # Commit 1's curator fix. Wire it through ``_make_interceptor``.
        interceptor = self._make_interceptor(
            mock_taskmaster, project_root='/fake/project',
        )

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


# ──────────────────────────────────────────────────────────────────────────────
# TaskInterceptor.close() lifecycle
# ──────────────────────────────────────────────────────────────────────────────


class TestTaskInterceptorClose:
    """Tests for TaskInterceptor.close() resource release and re-use prevention."""

    def _make_interceptor(self, mock_taskmaster=None, curator=None) -> TaskInterceptor:
        from fused_memory.middleware.task_interceptor import TaskInterceptor
        config = _make_interceptor_config()
        return TaskInterceptor(
            taskmaster=mock_taskmaster,
            targeted_reconciler=None,
            event_buffer=MagicMock(),
            config=config,
            curator=curator,
        )

    @pytest.mark.asyncio
    async def test_close_awaits_curator_close_and_clears(self):
        """close() awaits curator.close() and releases the curator."""
        curator = curator_double()
        interceptor = self._make_interceptor(curator=curator)

        await interceptor.close()

        curator.close.assert_awaited_once()
        assert interceptor.curator_registry_health() == ()

    @pytest.mark.asyncio
    async def test_close_sets_closed_flag(self):
        """close() sets _closed = True."""
        interceptor = self._make_interceptor()
        assert not interceptor._closed
        await interceptor.close()
        assert interceptor._closed

    @pytest.mark.asyncio
    async def test_get_curator_returns_none_after_close(self):
        """_get_curator() returns None after close(), preventing resource re-creation.

        Calls the private ``_get_curator`` directly because its close guard is
        the subject.
        """
        interceptor = self._make_interceptor()
        await interceptor.close()

        result = await interceptor._get_curator()
        assert result is None

    @pytest.mark.asyncio
    async def test_close_drains_before_releasing_curator(self):
        """close() calls drain() before closing the curator."""
        mock_curator = curator_double()
        interceptor = self._make_interceptor(curator=mock_curator)

        call_order: list[str] = []
        original_drain = interceptor.drain

        async def tracking_drain():
            call_order.append('drain')
            await original_drain()

        mock_curator.close = AsyncMock(
            side_effect=lambda: call_order.append('curator_close'),
        )
        interceptor.drain = tracking_drain  # type: ignore[assignment]

        await interceptor.close()

        assert call_order == ['drain', 'curator_close']
