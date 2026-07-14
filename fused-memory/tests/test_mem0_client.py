"""Unit tests for Mem0Backend — filter construction and search delegation."""

import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _fm_helpers import QDRANT_URL, qdrant_skipif
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import VectorParams

from fused_memory.backends.mem0_client import Mem0Backend
from fused_memory.models.scope import Scope


@pytest.fixture
def backend(mock_config):
    """Mem0Backend using mock config (no real Qdrant/Mem0 needed)."""
    return Mem0Backend(mock_config)


class TestMem0BackendSearch:
    @pytest.mark.asyncio
    async def test_no_categories_omits_filters(self, backend):
        """When categories is not passed, filters=None must reach instance.search."""
        mock_instance = MagicMock()
        mock_instance.search = AsyncMock(return_value={'results': []})
        with patch.object(backend, '_get_instance', AsyncMock(return_value=mock_instance)):
            await backend.search(
                query='q',
                scope=Scope(project_id='p'),
                limit=5,
            )
        call_kwargs = mock_instance.search.call_args.kwargs
        # filters kwarg must be absent or explicitly None
        filters = call_kwargs.get('filters', None)
        assert filters is None, (
            f'Expected filters=None when no categories given, got filters={filters!r}'
        )

    @pytest.mark.asyncio
    async def test_single_category_builds_equality_filter(self, backend):
        """A single category must produce an equality filter dict passed to instance.search."""
        mock_instance = MagicMock()
        mock_instance.search = AsyncMock(return_value={'results': []})
        with patch.object(backend, '_get_instance', AsyncMock(return_value=mock_instance)):
            await backend.search(
                query='q',
                scope=Scope(project_id='p'),
                limit=5,
                categories=['preferences_and_norms'],
            )
        call_kwargs = mock_instance.search.call_args.kwargs
        assert call_kwargs.get('filters') == {'category': 'preferences_and_norms'}, (
            f"Expected filters={{'category': 'preferences_and_norms'}}, got {call_kwargs.get('filters')!r}"
        )

    @pytest.mark.asyncio
    async def test_multiple_categories_builds_in_filter(self, backend):
        """Multiple categories must produce an in-list filter dict passed to instance.search."""
        mock_instance = MagicMock()
        mock_instance.search = AsyncMock(return_value={'results': []})
        with patch.object(backend, '_get_instance', AsyncMock(return_value=mock_instance)):
            await backend.search(
                query='q',
                scope=Scope(project_id='p'),
                limit=5,
                categories=['preferences_and_norms', 'procedural_knowledge'],
            )
        call_kwargs = mock_instance.search.call_args.kwargs
        expected = {'category': {'in': ['preferences_and_norms', 'procedural_knowledge']}}
        assert call_kwargs.get('filters') == expected, (
            f'Expected filters={expected!r}, got {call_kwargs.get("filters")!r}'
        )


class TestMem0BackendScrollByMetadata:
    """scroll_by_metadata builds a Qdrant payload filter and returns normalised point dicts."""

    def _make_mock_point(self, point_id: str, created_at: str | None, extra_meta: dict | None = None):
        """Create a MagicMock that mimics a Qdrant Record/ScoredPoint."""
        payload = {}
        if created_at is not None:
            payload['created_at'] = created_at
        if extra_meta:
            payload.update(extra_meta)
        point = MagicMock()
        point.id = point_id
        point.payload = payload
        return point

    @pytest.mark.asyncio
    async def test_builds_filter_and_calls_scroll(self, backend):
        """scroll_by_metadata builds the correct Qdrant Filter and calls client.scroll."""
        from qdrant_client.http import models as qmodels

        mock_client = AsyncMock()
        # scroll returns a tuple (points, next_offset)
        mock_client.scroll = AsyncMock(return_value=([], None))

        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=mock_client)):
            await backend.scroll_by_metadata(
                scope=Scope(project_id='dark_factory'),
                filters={'recon_pool': 'stage2_cycle_summary'},
                limit=50,
            )

        assert mock_client.scroll.await_count == 1
        call_kwargs = mock_client.scroll.call_args.kwargs
        # collection_name must derive from scope
        collection_prefix = backend.config.mem0.collection_prefix
        expected_collection = f'{collection_prefix}_dark_factory'
        assert call_kwargs.get('collection_name') == expected_collection
        # with_payload must be True
        assert call_kwargs.get('with_payload') is True
        # limit must be forwarded
        assert call_kwargs.get('limit') == 50
        # scroll_filter must be a Qdrant Filter with one FieldCondition
        scroll_filter = call_kwargs.get('scroll_filter')
        assert isinstance(scroll_filter, qmodels.Filter)
        assert isinstance(scroll_filter.must, list)
        assert len(scroll_filter.must) == 1
        cond = scroll_filter.must[0]
        assert isinstance(cond, qmodels.FieldCondition)
        assert cond.key == 'recon_pool'
        assert isinstance(cond.match, qmodels.MatchValue)
        assert cond.match.value == 'stage2_cycle_summary'

    @pytest.mark.asyncio
    async def test_normalises_points_to_dicts(self, backend):
        """Scroll points are normalised to {'id', 'created_at', 'metadata'} dicts."""
        p1 = self._make_mock_point('id-1', '2026-01-01T00:00:00+00:00', {'recon_pool': 'stage2_cycle_summary'})
        p2 = self._make_mock_point('id-2', '2026-02-01T00:00:00+00:00', {'recon_pool': 'stage2_cycle_summary'})

        mock_client = AsyncMock()
        mock_client.scroll = AsyncMock(return_value=([p1, p2], None))

        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=mock_client)):
            result = await backend.scroll_by_metadata(
                scope=Scope(project_id='p'),
                filters={'recon_pool': 'stage2_cycle_summary'},
            )

        assert len(result) == 2
        # First point
        assert result[0]['id'] == 'id-1'
        assert result[0]['created_at'] == '2026-01-01T00:00:00+00:00'
        assert isinstance(result[0]['metadata'], dict)
        assert result[0]['metadata']['recon_pool'] == 'stage2_cycle_summary'
        # Second point
        assert result[1]['id'] == 'id-2'
        assert result[1]['created_at'] == '2026-02-01T00:00:00+00:00'

    @pytest.mark.asyncio
    async def test_empty_filters_raises_value_error(self, backend):
        """Empty filters dict must raise ValueError (mirrors count_by_metadata)."""
        with pytest.raises(ValueError, match='scroll_by_metadata requires at least one filter'):
            await backend.scroll_by_metadata(
                scope=Scope(project_id='p'),
                filters={},
            )

    @pytest.mark.asyncio
    async def test_empty_scroll_result_returns_empty_list(self, backend):
        """When Qdrant returns no points, the method returns an empty list."""
        mock_client = AsyncMock()
        mock_client.scroll = AsyncMock(return_value=([], None))

        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=mock_client)):
            result = await backend.scroll_by_metadata(
                scope=Scope(project_id='p'),
                filters={'recon_pool': 'stage2_cycle_summary'},
            )

        assert result == []

    @pytest.mark.asyncio
    async def test_timeout_logs_warning_and_returns_empty(self, backend, caplog):
        """On TimeoutError, logs WARNING and returns [] without raising."""
        import logging

        mock_client = AsyncMock()
        mock_client.scroll = AsyncMock(side_effect=TimeoutError('too slow'))

        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=mock_client)), \
                caplog.at_level(logging.WARNING, logger='fused_memory.backends.mem0_client'):
            result = await backend.scroll_by_metadata(
                scope=Scope(project_id='p'),
                filters={'recon_pool': 'stage2_cycle_summary'},
            )

        assert result == []
        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_records) >= 1

    @pytest.mark.asyncio
    async def test_limit_reached_logs_truncation_warning(self, backend, caplog):
        """When Qdrant returns exactly `limit` points, logs a WARNING about possible truncation.

        Per the no-silent-caps convention: if scroll returns exactly the requested
        limit it may have hit the page boundary and silently dropped members.  A
        WARNING makes this visible so operators know to investigate.
        """
        import logging

        # Three points returned for limit=3 — triggers the boundary warning.
        points = [
            self._make_mock_point(f'id-{i}', f'2026-0{i+1}-01T00:00:00+00:00')
            for i in range(3)
        ]
        mock_client = AsyncMock()
        mock_client.scroll = AsyncMock(return_value=(points, 'some-offset'))

        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=mock_client)), \
                caplog.at_level(logging.WARNING, logger='fused_memory.backends.mem0_client'):
            result = await backend.scroll_by_metadata(
                scope=Scope(project_id='p'),
                filters={'recon_pool': 'stage2_cycle_summary'},
                limit=3,
            )

        # All three points are still returned — the warning does not suppress results.
        assert len(result) == 3
        warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any('truncated' in m or 'limit' in m for m in warning_msgs), (
            'Expected a truncation WARNING when len(points)==limit; '
            f'got warning messages: {warning_msgs}'
        )


class TestMem0BackendAddSystemRecord:
    """Unit tests for the dedup-exempt system-write path (task 2222 / W5-δ).

    ``add_system_record`` must pin ``infer=False`` LOCALLY in this dedicated
    method (not inherited from ``Mem0Backend.add``) — see the method's
    docstring for the full rationale (task decision #2: a future
    re-enabling of Mem0 dedup on the general ``add()`` must not silently
    re-break recon's system writes).
    """

    @pytest.mark.asyncio
    async def test_pins_infer_false_and_maps_scope_and_metadata(self, backend):
        """instance.add must be awaited with infer=False and scope-mapped ids."""
        mock_instance = MagicMock()
        mock_instance.add = AsyncMock(return_value={'results': [{'id': 'sys-1'}]})
        scope = Scope(project_id='p', agent_id='recon-stage-x', session_id='s')

        with patch.object(backend, '_get_instance', AsyncMock(return_value=mock_instance)):
            result = await backend.add_system_record(
                content='sys',
                scope=scope,
                metadata={'kind': 'cycle_summary'},
            )

        call_kwargs = mock_instance.add.await_args.kwargs
        assert call_kwargs.get('infer') is False, (
            'add_system_record must pin infer=False (the dedup-exempt pin); '
            f"got infer={call_kwargs.get('infer')!r}"
        )
        assert call_kwargs.get('user_id') == scope.mem0_user_id, (
            f"expected user_id={scope.mem0_user_id!r}, got {call_kwargs.get('user_id')!r}"
        )
        assert call_kwargs.get('agent_id') == 'recon-stage-x', (
            f"expected agent_id='recon-stage-x', got {call_kwargs.get('agent_id')!r}"
        )
        assert call_kwargs.get('run_id') == 's', (
            f"expected run_id='s', got {call_kwargs.get('run_id')!r}"
        )
        assert call_kwargs.get('metadata') == {'kind': 'cycle_summary'}, (
            f"expected metadata={{'kind': 'cycle_summary'}}, got {call_kwargs.get('metadata')!r}"
        )
        assert result == {'results': [{'id': 'sys-1'}]}, (
            f'add_system_record must return the raw mem0 result dict, got {result!r}'
        )


class TestMem0BackendUpdate:
    """Unit tests for Mem0Backend.update's metadata-forwarding (task 2452).

    mem0's ``AsyncMemory._update_memory`` overwrites the whole Qdrant
    payload on update -- passing the full existing payload back as
    ``metadata=`` is what preserves custom keys (``src_project`` /
    ``dst_project`` / ``kind`` / ...) instead of wiping them.
    ``Mem0Backend.update`` previously forwarded no metadata at all (and had
    zero callers), which would have silently wiped such keys for any future
    caller relying on an in-place edit.
    """

    @pytest.mark.asyncio
    async def test_forwards_metadata_to_instance_update(self, backend):
        """metadata= must be forwarded so mem0's payload-overwriting update
        preserves the custom provenance keys instead of wiping them."""
        mock_instance = MagicMock()
        mock_instance.update = AsyncMock(return_value={'message': 'Memory updated successfully!'})
        metadata = {'kind': 'cgl_eta_cross_target_rehome', 'src_project': 'reify'}

        with patch.object(backend, '_get_instance', AsyncMock(return_value=mock_instance)):
            await backend.update(
                'mem-id', 'new content', Scope(project_id='p'), metadata=metadata,
            )

        call = mock_instance.update.await_args
        assert call.args == ('mem-id', 'new content'), (
            f'expected positional args (mem-id, new content), got {call.args!r}'
        )
        assert call.kwargs.get('metadata') == metadata, (
            f'expected metadata={metadata!r} forwarded, got {call.kwargs.get("metadata")!r}'
        )

    @pytest.mark.asyncio
    async def test_back_compat_no_metadata_arg(self, backend):
        """Calling update() with no metadata (the pre-task-2452 call shape)
        must still work and must not forward a non-None metadata kwarg."""
        mock_instance = MagicMock()
        mock_instance.update = AsyncMock(return_value={'message': 'Memory updated successfully!'})

        with patch.object(backend, '_get_instance', AsyncMock(return_value=mock_instance)):
            await backend.update('mem-id', 'new content', Scope(project_id='p'))

        call = mock_instance.update.await_args
        assert call.args == ('mem-id', 'new content'), (
            f'expected positional args (mem-id, new content), got {call.args!r}'
        )
        assert call.kwargs.get('metadata') is None, (
            'expected no non-None metadata kwarg when caller omits it, got '
            f'{call.kwargs.get("metadata")!r}'
        )


# ---------------------------------------------------------------------------
# Integration: add_system_record fresh-point-per-call (task 2222 / W5-δ, P4)
# ---------------------------------------------------------------------------
#
# Mirrors tests/test_recon_dedup_premise.py's real-Qdrant harness (task 2221 /
# W5-γ, which already empirically proved the underlying infer=False primitive
# never dedup-drops), adapted to probe add_system_record instead of add().
# This pins the SAME fresh-point-every-call guarantee for the new dedicated,
# dedup-exempt method.


@pytest.fixture
def _asr_scope(worker_id) -> Scope:
    """A recon-stage scope isolated per xdist worker for add_system_record tests."""
    return Scope(
        project_id=f'_test_add_system_record_{worker_id}',
        agent_id='recon-stage-task_knowledge_sync',
    )


@pytest.fixture
def _asr_clean_collection(_asr_scope, mock_config):
    """Delete the scope's Qdrant collection before AND after the test."""
    collection = _asr_scope.mem0_collection_name(mock_config.mem0.collection_prefix)
    client = QdrantClient(url=QDRANT_URL, timeout=10)
    with contextlib.suppress(ResponseHandlingException, UnexpectedResponse):
        client.delete_collection(collection)
    yield collection
    with contextlib.suppress(Exception):
        client.delete_collection(collection)
    client.close()


def _asr_collection_vector_size(collection: str) -> int:
    """Read back the vector dimension Qdrant actually created for *collection*."""
    client = QdrantClient(url=QDRANT_URL, timeout=10)
    try:
        info = client.get_collection(collection)
        vectors = info.config.params.vectors
        assert isinstance(vectors, VectorParams), (
            f'expected an unnamed VectorParams config, got {vectors!r}'
        )
        return vectors.size
    finally:
        client.close()


async def _build_asr_backend(mock_config, scope, monkeypatch) -> Mem0Backend:
    """Construct a real Mem0Backend with a stubbed constant-vector embedder.

    Mirrors test_recon_dedup_premise._build_recon_backend's hermetic
    (non-real-embedder) path: cosine=1.0 is the strongest-possible duplicate
    signal the infer=False path could ever see.
    """
    monkeypatch.setattr('mem0.memory.main.capture_event', lambda *a, **kw: None)

    config = mock_config.model_copy(deep=True)
    backend = Mem0Backend(config)
    inst = await backend._get_instance(scope)
    inst.db.add_history = lambda *a, **kw: None

    collection = scope.mem0_collection_name(config.mem0.collection_prefix)
    dim = _asr_collection_vector_size(collection)
    stub_vector = [0.1] * dim
    inst.embedding_model.embed = lambda *a, **kw: stub_vector

    return backend


class TestMem0BackendAddSystemRecordIntegration:
    """P4 boundary signal: N identical add_system_record calls -> N distinct points.

    Real Qdrant, isolated collection, stubbed constant-vector embedder (the
    worst-case cosine=1.0 duplicate signal) — mirrors
    test_recon_dedup_premise.py's harness for backend.add(), applied to the
    new dedup-exempt backend.add_system_record method.
    """

    pytestmark = [qdrant_skipif(), pytest.mark.timeout(60)]

    @pytest.mark.asyncio
    async def test_identical_calls_all_land_distinct(
        self, mock_config, _asr_scope, _asr_clean_collection, monkeypatch,
    ):
        backend = await _build_asr_backend(mock_config, _asr_scope, monkeypatch)
        run_id = 'task-2222-add-system-record-probe'
        metadata = {'kind': 'cycle_summary', 'run_id': run_id}
        n = 6
        try:
            ids = []
            for _ in range(n):
                response = await backend.add_system_record(
                    content=(
                        'Stage 2 cycle summary fixture for task 2222 '
                        'add_system_record fresh-point-per-call probe.'
                    ),
                    scope=_asr_scope,
                    metadata=metadata,
                )
                results = response.get('results') or []
                assert len(results) == 1, (
                    f'expected exactly one result under infer=False, got {results!r}'
                )
                assert 'id' in results[0]
                ids.append(results[0]['id'])

            assert len(set(ids)) == n, (
                f'expected {n} distinct ids (no dedup drop), got {len(set(ids))}: {ids!r}'
            )
            assert await backend.count_by_metadata(_asr_scope, {'run_id': run_id}) == n
        finally:
            await backend.close()
