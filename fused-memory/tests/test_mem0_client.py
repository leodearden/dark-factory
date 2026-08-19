"""Unit tests for Mem0Backend — filter construction and search delegation."""

import asyncio
import contextlib
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from _fm_helpers import QDRANT_URL, collection_vector_size, qdrant_skipif
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from fused_memory.backends.mem0_client import Mem0Backend, is_missing_collection_error
from fused_memory.models.scope import Scope

# Sentinel distinguishing "caller passed vector=None" (model a Qdrant point
# that genuinely has no vector) from "caller passed nothing" (leave the
# MagicMock attribute unset, so it auto-creates a truthy child mock).  A
# plain None default would collapse those two cases and make the
# missing-vector degradation test unable to fail.
_UNSET = object()


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

    @staticmethod
    def _make_mock_point(point_id: str, created_at: str | None, extra_meta: dict | None = None):
        """Create a MagicMock that mimics a Qdrant Record/ScoredPoint.

        A staticmethod so sibling suites can reuse it directly rather than
        borrowing it with a foreign ``self``.
        """
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
    async def test_timeout_propagates_not_swallowed(self, backend):
        """On TimeoutError, the exception propagates — it is NOT swallowed into [].

        Mirrors count_by_metadata's propagate-by-default contract (no
        try/except around asyncio.wait_for): a timed-out scroll must never be
        indistinguishable from a genuinely-empty result (no-silent-fail
        invariant).
        """
        mock_client = AsyncMock()
        mock_client.scroll = AsyncMock(side_effect=TimeoutError('too slow'))

        with (
            patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=mock_client)),
            pytest.raises(TimeoutError),
        ):
            await backend.scroll_by_metadata(
                scope=Scope(project_id='p'),
                filters={'recon_pool': 'stage2_cycle_summary'},
            )

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


class TestMem0BackendScrollByMetadataWithVectors:
    """scroll_by_metadata can optionally return each point's stored vector.

    Motivation (task 3210): the corpus-health detector generates ANN
    candidate pairs by querying Qdrant with each record's OWN stored vector,
    so it needs the vectors to come back on the enumeration pass.  Fetching
    them here means the detector makes ZERO embedding API calls at runtime
    and stays in the same metric space Mem0 itself wrote.

    ``with_vectors`` defaults to False so every existing caller
    (prune_recon_cycle_summaries, sweep_orphan_flag_markers, GC pool
    enumeration, ...) keeps today's payload-only contract and pays no extra
    bandwidth for vectors it does not read.

    Mocked AsyncQdrantClient throughout — no live Qdrant, so these stay in
    the unit lane alongside the sibling scroll_by_metadata tests above.
    """

    @staticmethod
    def _make_mock_point(
        point_id: str,
        created_at: str | None,
        extra_meta: dict | None = None,
        vector=_UNSET,
    ):
        """Qdrant Record/ScoredPoint double, optionally carrying a vector.

        ``vector`` defaults to the _UNSET sentinel meaning "leave the
        attribute unset", which on a MagicMock auto-creates a truthy child
        mock — exactly the trap the implementation must not fall into when
        with_vectors=False.  Pass ``vector=None`` to model Qdrant returning a
        point with no vector, or a list to model a real one.
        """
        payload = {}
        if created_at is not None:
            payload['created_at'] = created_at
        if extra_meta:
            payload.update(extra_meta)
        point = MagicMock()
        point.id = point_id
        point.payload = payload
        if vector is not _UNSET:
            point.vector = vector
        return point

    @pytest.mark.asyncio
    async def test_defaults_to_no_vectors(self, backend):
        """Default call forwards with_vectors=False and returns dicts with NO 'vector' key.

        Pins the back-compat contract: today's callers must not start seeing
        a new key (nor pay to transfer vectors they never asked for).
        """
        p1 = self._make_mock_point('id-1', '2026-01-01T00:00:00+00:00', {'category': 'procedural_knowledge'})

        mock_client = AsyncMock()
        mock_client.scroll = AsyncMock(return_value=([p1], None))

        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=mock_client)):
            result = await backend.scroll_by_metadata(
                scope=Scope(project_id='p'),
                filters={'category': 'procedural_knowledge'},
            )

        assert mock_client.scroll.call_args.kwargs.get('with_vectors') is False, (
            'Default scroll_by_metadata must forward with_vectors=False to client.scroll, got '
            f'{mock_client.scroll.call_args.kwargs.get("with_vectors")!r}'
        )
        assert len(result) == 1
        assert 'vector' not in result[0], (
            f"Default mode must not add a 'vector' key; got keys {sorted(result[0])}"
        )
        # Existing keys are untouched.
        assert result[0]['id'] == 'id-1'
        assert result[0]['created_at'] == '2026-01-01T00:00:00+00:00'
        assert result[0]['metadata']['category'] == 'procedural_knowledge'

    @pytest.mark.asyncio
    async def test_with_vectors_true_forwards_and_lifts_vector(self, backend):
        """with_vectors=True forwards the flag and lifts each point's vector onto the dict."""
        vec1 = [0.1, 0.2, 0.3]
        vec2 = [0.4, 0.5, 0.6]
        p1 = self._make_mock_point(
            'id-1', '2026-01-01T00:00:00+00:00', {'category': 'procedural_knowledge'}, vector=vec1,
        )
        p2 = self._make_mock_point(
            'id-2', '2026-02-01T00:00:00+00:00', {'category': 'procedural_knowledge'}, vector=vec2,
        )

        mock_client = AsyncMock()
        mock_client.scroll = AsyncMock(return_value=([p1, p2], None))

        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=mock_client)):
            result = await backend.scroll_by_metadata(
                scope=Scope(project_id='p'),
                filters={'category': 'procedural_knowledge'},
                with_vectors=True,
            )

        assert mock_client.scroll.call_args.kwargs.get('with_vectors') is True, (
            'with_vectors=True must be forwarded to client.scroll, got '
            f'{mock_client.scroll.call_args.kwargs.get("with_vectors")!r}'
        )
        assert len(result) == 2
        assert result[0]['vector'] == vec1
        assert result[1]['vector'] == vec2
        # Existing keys are unchanged in this mode too.
        assert result[0]['id'] == 'id-1'
        assert result[0]['created_at'] == '2026-01-01T00:00:00+00:00'
        assert result[0]['metadata']['category'] == 'procedural_knowledge'
        assert result[1]['id'] == 'id-2'
        assert result[1]['created_at'] == '2026-02-01T00:00:00+00:00'

    @pytest.mark.asyncio
    async def test_missing_vector_degrades_to_none(self, backend):
        """A point returned without a vector yields vector=None rather than raising.

        Qdrant can return a point with no vector (e.g. it was never stored, or
        the collection uses named vectors).  The detector COUNTS these as a
        ``missing_vector`` disclosure and skips them, so the backend must hand
        back a clean None instead of exploding mid-enumeration and losing the
        whole scan.
        """
        p_ok = self._make_mock_point('id-ok', '2026-01-01T00:00:00+00:00', vector=[0.1, 0.2])
        p_missing = self._make_mock_point('id-missing', '2026-02-01T00:00:00+00:00', vector=None)

        mock_client = AsyncMock()
        mock_client.scroll = AsyncMock(return_value=([p_ok, p_missing], None))

        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=mock_client)):
            result = await backend.scroll_by_metadata(
                scope=Scope(project_id='p'),
                filters={'category': 'procedural_knowledge'},
                with_vectors=True,
            )

        assert len(result) == 2, 'A vector-less point must still be returned, not dropped'
        assert result[0]['vector'] == [0.1, 0.2]
        assert result[1]['vector'] is None
        # The vector-less record keeps its identity so it can be counted/reported.
        assert result[1]['id'] == 'id-missing'
        assert result[1]['created_at'] == '2026-02-01T00:00:00+00:00'


class TestMem0BackendPayloadFilterSingleHome:
    """The Qdrant payload ``Filter`` has exactly ONE construction site (INV-5).

    Motivation (task 3225): the same ``Filter(must=[FieldCondition(key=k,
    match=MatchValue(value=v)) ...])`` was built independently inside
    ``count_by_metadata``, ``scroll_by_metadata`` and
    ``scripts/census_memory_metadata.scroll_all_payloads``.  That is not a
    tidiness problem: the census's coverage reconciliation cross-checks its
    SCROLL against ``count_by_metadata``'s COUNT, so if the two constructions
    ever drift the reconciliation silently compares two DIFFERENT point sets
    while still reporting ``coverage.complete == true``.

    A comment cannot fail CI, so the drift is pinned by an equality assertion
    at each entry point instead: the ``count_filter=`` kwarg the count API
    hands Qdrant must EQUAL the ``scroll_filter=`` kwarg the scroll API hands
    Qdrant must EQUAL ``_build_payload_filter(filters)``.  qdrant models are
    pydantic, so ``==`` is structural value equality.
    """

    @pytest.mark.asyncio
    async def test_builds_field_conditions_in_dict_order(self, backend):
        """_build_payload_filter maps each filter item to a FieldCondition/MatchValue."""
        from qdrant_client.http import models as qmodels

        built = backend._build_payload_filter({'category': 'x', 'kind': 'y'})

        assert isinstance(built, qmodels.Filter)
        assert isinstance(built.must, list)
        assert len(built.must) == 2
        # Narrow to FieldCondition up front: qdrant's `Condition` is a union
        # and only FieldCondition carries `.key`.  The re-length-check is what
        # keeps this as strong as asserting on the raw list -- a non-FieldCondition
        # would be filtered out here and caught by the count, not slip through.
        fields = [c for c in built.must if isinstance(c, qmodels.FieldCondition)]
        assert len(fields) == 2
        # Dict-insertion order is preserved so the filter is deterministic
        # (two callers passing the same dict produce byte-identical filters).
        assert [c.key for c in fields] == ['category', 'kind']
        for cond, expected in zip(fields, ['x', 'y'], strict=True):
            assert isinstance(cond.match, qmodels.MatchValue)
            assert cond.match.value == expected

    @pytest.mark.asyncio
    async def test_count_and_scroll_pass_the_same_filter_object_shape(self, backend):
        """ANTI-DRIFT: count_filter == scroll_filter == _build_payload_filter(filters).

        This is the whole point of the extraction.  If a future edit changes
        one construction site and not the other, the census's
        scroll-vs-count reconciliation starts comparing two different point
        sets with no error surface — this assertion is what fails first.
        """
        filters = {'category': 'procedural_knowledge', 'recon_pool': 'stage2_cycle_summary'}
        scope = Scope(project_id='dark_factory')

        count_client = AsyncMock()
        count_client.count = AsyncMock(return_value=MagicMock(count=0))
        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=count_client)):
            await backend.count_by_metadata(scope=scope, filters=filters)

        scroll_client = AsyncMock()
        scroll_client.scroll = AsyncMock(return_value=([], None))
        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=scroll_client)):
            await backend.scroll_by_metadata(scope=scope, filters=filters)

        count_filter = count_client.count.call_args.kwargs.get('count_filter')
        scroll_filter = scroll_client.scroll.call_args.kwargs.get('scroll_filter')
        built = backend._build_payload_filter(filters)

        assert count_filter == built, (
            f'count_by_metadata built {count_filter!r}, _build_payload_filter built {built!r}'
        )
        assert scroll_filter == built, (
            f'scroll_by_metadata built {scroll_filter!r}, _build_payload_filter built {built!r}'
        )
        assert count_filter == scroll_filter, (
            'count and scroll must select the same point set for the same filters; '
            f'got count_filter={count_filter!r} scroll_filter={scroll_filter!r}'
        )

    def test_empty_filters_raises_value_error(self, backend):
        """An empty filter dict is rejected at the single home too.

        Every caller already rejects it (an unfiltered Filter would select the
        WHOLE collection), so the shared builder must not become the hole that
        lets one through.
        """
        with pytest.raises(ValueError, match='at least one filter'):
            backend._build_payload_filter({})

    # -- the optional text-needle arm (task 3682) --------------------------
    #
    # ``scan_payload_text`` built its own ``Filter(must=[...], should=[...])``
    # inline, so the "one construction site" claim held for three of the four
    # metadata-addressed reads and quietly excluded the fourth.  Folding its
    # needle prefilter in as an optional ``should`` arm is what makes the
    # cross-method equality below provable at that entry point too.

    def test_text_needles_build_matchtext_should_conditions_in_order(self, backend):
        """One FieldCondition(key='data', MatchText) per needle, in call order.

        MatchText (not MatchValue) is load-bearing: MatchValue is exact
        equality on the whole field and can never find a fragment embedded in
        a longer memory.
        """
        from qdrant_client.http import models as qmodels

        built = backend._build_payload_filter({}, text_needles=['alpha', 'beta'])

        assert isinstance(built, qmodels.Filter)
        assert isinstance(built.should, list)
        assert len(built.should) == 2
        conds = [c for c in built.should if isinstance(c, qmodels.FieldCondition)]
        assert len(conds) == 2
        assert [c.key for c in conds] == ['data', 'data']
        for cond, needle in zip(conds, ['alpha', 'beta'], strict=True):
            assert isinstance(cond.match, qmodels.MatchText)
            assert cond.match.text == needle

    def test_filters_and_needles_populate_both_arms_of_one_filter(self, backend):
        """A narrowed prefilter scan is ONE Filter: must AND-ed, should OR-ed."""
        from qdrant_client.http import models as qmodels

        built = backend._build_payload_filter(
            {'category': 'procedural_knowledge'}, text_needles=['alpha']
        )

        assert isinstance(built.must, list) and len(built.must) == 1
        must_cond = built.must[0]
        assert isinstance(must_cond, qmodels.FieldCondition)
        assert must_cond.key == 'category'
        assert isinstance(must_cond.match, qmodels.MatchValue)
        assert must_cond.match.value == 'procedural_knowledge'

        assert isinstance(built.should, list) and len(built.should) == 1
        should_cond = built.should[0]
        assert isinstance(should_cond, qmodels.FieldCondition)
        assert should_cond.key == 'data'
        assert isinstance(should_cond.match, qmodels.MatchText)
        assert should_cond.match.text == 'alpha'

    def test_an_unused_arm_is_omitted_not_emitted_empty(self, backend):
        """OMIT an unused arm; never emit ``[]``.

        This is what makes the cross-method anti-drift equality provable.  A
        live probe (qdrant 1.17.1) measured that ``Filter(must=[c])`` and
        ``Filter(must=[c], should=[])`` select the SAME points — an empty arm
        is a server-side no-op, so this is not a live selection bug.  But
        qdrant models are pydantic, and ``Filter(must=[c]) ==
        Filter(must=[c], should=[])`` is **False** under structural equality.
        So a builder that emitted ``should=[]`` could never be asserted equal
        to what ``count_by_metadata`` hands Qdrant, and the single-home claim
        would stay untestable at the scan entry point.
        """
        filters_only = backend._build_payload_filter({'k': 'v'})
        assert filters_only.should is None, (
            'an unused should arm must be omitted, not emitted as []; '
            f'got {filters_only.should!r}'
        )

        needles_only = backend._build_payload_filter({}, text_needles=['alpha'])
        assert needles_only.must is None, (
            'an unused must arm must be omitted, not emitted as []; '
            f'got {needles_only.must!r}'
        )

    def test_needles_only_is_allowed_and_does_not_raise(self, backend):
        """A prefilter scan with no metadata narrowing is a legitimate call.

        The empty-input guard rejects "no filters AND no needles", not "no
        filters" — a ``should``-only Filter still selects a proper subset of
        the collection, so the guard's whole-collection rationale does not
        apply to it.
        """
        from qdrant_client.http import models as qmodels

        built = backend._build_payload_filter({}, text_needles=['alpha'])

        assert isinstance(built, qmodels.Filter)
        assert built.should and len(built.should) == 1

    @pytest.mark.parametrize(
        ('filters', 'needles'),
        [({}, None), ({}, []), (None, None), (None, [])],
        ids=['empty-none', 'empty-empty', 'none-none', 'none-empty'],
    )
    def test_both_arms_empty_still_raises_value_error(self, backend, filters, needles):
        """Two empty arms is still an unfiltered whole-collection select.

        The message keeps the ``at least one filter`` substring the existing
        ``test_empty_filters_raises_value_error`` matches on, so widening the
        guard cannot silently retire that assertion.
        """
        with pytest.raises(ValueError, match='at least one filter'):
            backend._build_payload_filter(filters, text_needles=needles)


def _paging_client(pages: list[tuple[list, object]]) -> AsyncMock:
    """AsyncMock Qdrant client whose scroll() replays a scripted sequence of
    ``(points, next_offset)`` pairs."""
    client = AsyncMock()
    client.scroll = AsyncMock(side_effect=list(pages))
    return client


async def _drain(agen) -> list:
    return [item async for item in agen]


class TestMem0BackendScrollCollectionPages:
    """The low-level pager: collection-addressed, raw points, real pagination.

    Ported from ``scripts/census_memory_metadata.py``'s ``scroll_all_payloads``
    (task 3225), whose whole reason to exist was that the backend discarded
    ``next_offset`` and so re-read the same head forever.  The loop now lives
    here — one home for the offset/next_offset walk, the per-page read bound
    and the page budget — with the census and
    ``scripts/consolidate_namespace_families.py`` both sitting on top of it.

    Collection-name-addressed (not Scope-addressed) and filter-optional on
    purpose: consolidate_namespace_families scrolls LEGACY mis-named
    collections (``reify_reify``, ``fused_dark-factory``) that a Scope
    structurally cannot produce, with no filter at all.
    """

    @staticmethod
    def _make_mock_point(point_id: str, created_at: str | None = None, extra_meta: dict | None = None):
        """Reuse the sibling suite's Qdrant Record/ScoredPoint double."""
        return TestMem0BackendScrollByMetadata._make_mock_point(point_id, created_at, extra_meta)

    @pytest.mark.asyncio
    async def test_yields_raw_points_across_all_pages_in_order(self, backend):
        """Raw point objects (not normalised dicts) stream out in page order.

        consolidate_namespace_families' merge_collection reads
        ``point.id``/``point.vector``/``point.payload`` to build PointStructs,
        so this layer must not normalise.
        """
        p1, p2, p3, p4 = (self._make_mock_point(f'id-{i}') for i in range(1, 5))
        client = _paging_client([([p1, p2], 'off-1'), ([p3], 'off-2'), ([p4], None)])

        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=client)):
            got = await _drain(backend.scroll_collection_pages('reify_reify', page_size=2))

        assert got == [p1, p2, p3, p4], 'points must stream in order, identity-preserved'

    @pytest.mark.asyncio
    async def test_passes_previous_next_offset_on_each_subsequent_call(self, backend):
        """THE defect being fixed: each page is requested at the prior next_offset.

        A non-paging implementation sends offset=None three times and re-reads
        the same head.
        """
        client = _paging_client([
            ([self._make_mock_point('a')], 'off-1'),
            ([self._make_mock_point('b')], 'off-2'),
            ([self._make_mock_point('c')], None),
        ])

        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=client)):
            await _drain(backend.scroll_collection_pages('c', page_size=1))

        offsets = [call.kwargs.get('offset') for call in client.scroll.await_args_list]
        assert offsets == [None, 'off-1', 'off-2']

    @pytest.mark.asyncio
    async def test_stops_when_next_offset_is_none(self, backend):
        client = _paging_client([([self._make_mock_point('a')], None)])

        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=client)):
            await _drain(backend.scroll_collection_pages('c', page_size=10))

        assert client.scroll.await_count == 1

    @pytest.mark.asyncio
    async def test_forwards_collection_name_verbatim_and_scroll_kwargs(self, backend):
        """collection_name is passed through UNCHANGED — never scope-derived.

        ``fused_dark-factory`` is a real legacy collection name from
        consolidate_namespace_families' COLLECTION_MERGES; prefixing or
        re-deriving it would address a collection that does not exist.
        """
        client = _paging_client([([], None)])

        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=client)):
            await _drain(backend.scroll_collection_pages('fused_dark-factory', page_size=25))

        kwargs = client.scroll.await_args_list[0].kwargs
        assert kwargs['collection_name'] == 'fused_dark-factory'
        assert kwargs['with_payload'] is True
        assert kwargs['limit'] == 25
        assert kwargs['with_vectors'] is False, 'vectors are opt-in, not paid for by default'
        assert kwargs['scroll_filter'] is None, 'no filter given ⇒ scroll the whole collection'

    @pytest.mark.asyncio
    async def test_forwards_with_vectors_and_scroll_filter_when_given(self, backend):
        """Both opt-in kwargs reach Qdrant verbatim."""
        from qdrant_client.http import models as qmodels

        given_filter = backend._build_payload_filter({'category': 'procedural_knowledge'})
        client = _paging_client([([], None)])

        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=client)):
            await _drain(backend.scroll_collection_pages(
                'c', scroll_filter=given_filter, page_size=10, with_vectors=True,
            ))

        kwargs = client.scroll.await_args_list[0].kwargs
        assert kwargs['with_vectors'] is True
        assert isinstance(kwargs['scroll_filter'], qmodels.Filter)
        assert kwargs['scroll_filter'] == given_filter

    @pytest.mark.asyncio
    async def test_empty_collection_yields_nothing(self, backend):
        client = _paging_client([([], None)])

        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=client)):
            got = await _drain(backend.scroll_collection_pages('c', page_size=10))

        assert got == []

    @pytest.mark.asyncio
    async def test_raises_when_page_budget_exhausted_with_live_next_offset(self, backend):
        """A truncated stream RAISES, never returns short (INV-2 no-silent-fail).

        A pager with no budget loops forever if Qdrant keeps handing back a
        live next_offset, so the budget travels with the loop — and exhausting
        it is an error, not a quiet early return.
        """
        from fused_memory.backends.mem0_client import ScrollPageBudgetExhausted

        client = _paging_client([
            ([self._make_mock_point('a')], 'off-1'),
            ([self._make_mock_point('b')], 'off-2'),
        ])

        with (
            patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=client)),
            pytest.raises(ScrollPageBudgetExhausted) as excinfo,
        ):
            await _drain(backend.scroll_collection_pages('fused_reify', page_size=1, max_pages=2))

        message = str(excinfo.value)
        assert 'fused_reify' in message, 'the message must name the collection'
        assert '2' in message, 'the message must name the exhausted page count'
        assert 'off-2' in message, 'the message must name the still-live offset'

    @pytest.mark.asyncio
    async def test_does_not_raise_when_budget_is_exactly_enough(self, backend):
        p1, p2 = self._make_mock_point('a'), self._make_mock_point('b')
        client = _paging_client([([p1], 'off-1'), ([p2], None)])

        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=client)):
            got = await _drain(backend.scroll_collection_pages('c', page_size=1, max_pages=2))

        assert got == [p1, p2]

    @pytest.mark.asyncio
    async def test_scroll_page_budget_exhausted_is_an_exception(self):
        from fused_memory.backends.mem0_client import ScrollPageBudgetExhausted

        assert issubclass(ScrollPageBudgetExhausted, Exception)

    @pytest.mark.asyncio
    async def test_scroll_point_budget_exhausted_is_an_exception(self):
        from fused_memory.backends.mem0_client import ScrollPointBudgetExhausted

        assert issubclass(ScrollPointBudgetExhausted, Exception)

    @pytest.mark.asyncio
    async def test_the_two_budget_exhaustions_are_unrelated_by_inheritance(self):
        """Neither budget exception may be a sub- or superclass of the other.

        The two are DIFFERENT events: ``max_points`` is a cap the caller asked
        for, ``max_pages`` is a safety backstop against an endless walk.  The
        whole point of splitting them is that a caller can catch exactly one
        and let the other propagate — which is how ``scan_payload_text`` keeps
        its flag-and-warn posture for its own ``limit`` while still failing
        loudly on the backstop.  An inheritance link would silently collapse
        that choice back into one.

        Load-bearing beyond this class:
        ``scripts/census_memory_metadata.CensusScanIncomplete`` is a
        module-level ALIAS of ScrollPageBudgetExhausted, so its ``except
        CensusScanIncomplete`` would start catching (or, in the other
        direction, leaking) a points-cap event the census has no cap for.
        ``scripts/consolidate_namespace_families`` catches the page budget
        directly and has the same exposure.
        """
        from fused_memory.backends.mem0_client import (
            ScrollPageBudgetExhausted,
            ScrollPointBudgetExhausted,
        )

        assert ScrollPointBudgetExhausted is not ScrollPageBudgetExhausted
        assert not issubclass(ScrollPointBudgetExhausted, ScrollPageBudgetExhausted), (
            'a points-cap event must not be catchable as a page-budget event; '
            'census_memory_metadata.CensusScanIncomplete aliases the latter'
        )
        assert not issubclass(ScrollPageBudgetExhausted, ScrollPointBudgetExhausted), (
            'a page-budget event must not be catchable as a points-cap event; '
            "scan_payload_text catches only the points cap and relies on the "
            'page budget propagating past it'
        )

    # -- the caller-supplied points cap (task 3682) ------------------------
    #
    # scan_payload_text carried a second copy of this walk purely because it
    # needed to stop after N POINTS rather than N pages.  Pushing that cap in
    # here is what lets the walk have one home; the tests below pin the three
    # properties the fold must not lose.

    @pytest.mark.asyncio
    async def test_max_points_shrinks_the_request_so_it_never_over_fetches(self, backend):
        """The request is shrunk to what is still wanted: ONE round-trip, limit=3.

        This is the property a naive ``async for ... break`` layering cannot
        have: to learn "there is more" it must pull a point PAST the cap,
        costing an extra scroll round-trip (~70-90 ms measured against live
        qdrant).  Owning the cap inside the pager makes the look-ahead free.
        """
        points = [self._make_mock_point(f'id-{i}') for i in range(3)]
        client = _paging_client([(points, None)])

        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=client)):
            got = await _drain(
                backend.scroll_collection_pages('c', page_size=256, max_points=3)
            )

        assert len(got) == 3
        assert client.scroll.await_count == 1
        assert client.scroll.call_args.kwargs.get('limit') == 3, (
            'the page request must shrink to the remaining budget, not ask for '
            f'the full page_size; got {client.scroll.call_args.kwargs.get("limit")!r}'
        )

    @pytest.mark.asyncio
    async def test_raises_when_the_points_cap_is_reached_with_a_live_next_offset(self, backend):
        """All capped points are yielded, THEN it raises — never a short return."""
        from fused_memory.backends.mem0_client import ScrollPointBudgetExhausted

        points = [self._make_mock_point(f'id-{i}') for i in range(2)]
        client = _paging_client([(points, 'off-1')])
        got: list = []

        with (
            patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=client)),
            pytest.raises(ScrollPointBudgetExhausted) as excinfo,
        ):
            async for point in backend.scroll_collection_pages(
                'fused_reify', page_size=256, max_points=2
            ):
                got.append(point)

        assert got == points, 'the capped points are yielded before the raise'
        message = str(excinfo.value)
        assert 'fused_reify' in message, 'the message must name the collection'
        assert '2' in message, 'the message must name the cap'
        assert 'off-1' in message, 'the message must name the still-live offset'

    @pytest.mark.asyncio
    async def test_a_clean_end_exactly_at_the_cap_does_not_raise(self, backend):
        """Reaching the cap and the end of the stream together is NOT a truncation.

        Nothing was left behind, so there is nothing to disclose.  Ordering the
        ``next_offset is None`` return ahead of the cap check is what makes an
        exhaustive-but-exactly-capped walk a clean result instead of a
        spurious error.
        """
        points = [self._make_mock_point(f'id-{i}') for i in range(3)]
        client = _paging_client([(points, None)])

        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=client)):
            got = await _drain(
                backend.scroll_collection_pages('c', page_size=256, max_points=3)
            )

        assert got == points

    @pytest.mark.asyncio
    async def test_the_default_points_cap_is_inert(self, backend):
        """max_points=None (the default) never caps and never raises.

        Pins that the existing consumers are structurally unaffected:
        scroll_all_by_metadata / _scroll_all_records,
        scripts/census_memory_metadata and
        scripts/consolidate_namespace_families all pass no cap, so none of
        them can ever see ScrollPointBudgetExhausted.
        """
        pages = [([self._make_mock_point(f'id-{i}')], f'off-{i}') for i in range(5)]
        pages.append(([self._make_mock_point('last')], None))
        client = _paging_client(pages)

        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=client)):
            got = await _drain(backend.scroll_collection_pages('c', page_size=1))

        assert len(got) == 6
        assert client.scroll.call_args.kwargs.get('limit') == 1, (
            'with no cap the request stays at the full page_size'
        )

    @pytest.mark.asyncio
    async def test_the_points_cap_wins_when_both_budgets_are_exhausted(self, backend):
        """Same page exhausts both: the caller's explicit cap is the event raised.

        max_pages is a backstop; reporting it when the caller's own cap
        explains the stop would misattribute an expected outcome to an
        internal safety limit.
        """
        from fused_memory.backends.mem0_client import ScrollPointBudgetExhausted

        client = _paging_client([([self._make_mock_point('a')], 'off-1')])

        with (
            patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=client)),
            pytest.raises(ScrollPointBudgetExhausted),
        ):
            await _drain(
                backend.scroll_collection_pages('c', page_size=1, max_pages=1, max_points=1)
            )

    @pytest.mark.asyncio
    async def test_a_server_over_return_stops_at_the_cap_rather_than_over_yielding(self, backend):
        """If the server hands back MORE than the shrunk request asked for, the
        cap still holds.

        The cap is enforced per-yield, not per-page, so a server that ignores
        the shrunk ``limit`` cannot walk a caller past the budget it set.
        """
        from fused_memory.backends.mem0_client import ScrollPointBudgetExhausted

        points = [self._make_mock_point(f'id-{i}') for i in range(5)]
        client = _paging_client([(points, 'off-1')])
        got: list = []

        with (
            patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=client)),
            pytest.raises(ScrollPointBudgetExhausted),
        ):
            async for point in backend.scroll_collection_pages(
                'c', page_size=256, max_points=2
            ):
                got.append(point)

        assert got == points[:2], f'must not yield past the cap; got {len(got)} points'

    @pytest.mark.asyncio
    async def test_a_hung_page_request_raises_instead_of_hanging(self, backend):
        """A wedged socket fails loudly rather than hanging a ~30-page scan forever.

        Same propagate-don't-swallow posture as count_by_metadata /
        scroll_by_metadata / get_point_by_id.
        """
        async def _hang(**kwargs):
            await asyncio.sleep(3600)

        client = AsyncMock()
        client.scroll = AsyncMock(side_effect=_hang)
        backend._read_timeout = 0.01

        with (
            patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=client)),
            pytest.raises(TimeoutError),
        ):
            await _drain(backend.scroll_collection_pages('c', page_size=10))

    @pytest.mark.asyncio
    async def test_timeout_applies_per_page_not_per_scan(self, backend):
        """Every page gets its OWN full-length bound, not a share of one budget.

        A per-scan bound would abort a long-but-healthy multi-page scan the
        moment the pages' cumulative time crossed the read timeout.  Asserted
        by spying on wait_for rather than by sleeping, so it cannot flake
        under parallel load.
        """
        p1, p2, p3 = (self._make_mock_point(c) for c in 'abc')
        client = _paging_client([([p1], 'off-1'), ([p2], 'off-2'), ([p3], None)])
        backend._read_timeout = 7.5

        real_wait_for = asyncio.wait_for
        seen_timeouts: list = []

        async def _spy(awaitable, timeout=None):
            seen_timeouts.append(timeout)
            return await real_wait_for(awaitable, timeout)

        with (
            patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=client)),
            patch('fused_memory.backends.mem0_client.asyncio.wait_for', _spy),
        ):
            got = await _drain(backend.scroll_collection_pages('c', page_size=1))

        assert got == [p1, p2, p3]
        assert seen_timeouts == [7.5, 7.5, 7.5], (
            'each of the 3 page requests must be wrapped in its own '
            f'wait_for(timeout=_read_timeout); got {seen_timeouts!r}'
        )


class TestMem0BackendScrollAllByMetadata:
    """The metadata-addressed full-enumeration generator.

    Same addressing as ``scroll_by_metadata`` (Scope + non-empty filters) and
    the same per-record shape, but it EXHAUSTS the match set by paging instead
    of returning one capped list.  ``scroll_by_metadata`` is untouched and
    keeps its one-shot semantics for callers that want them
    (``scripts/audit_duplicate_memories.py``).
    """

    @staticmethod
    def _make_mock_point(
        point_id: str, created_at: str | None = None, extra_meta: dict | None = None, vector=_UNSET,
    ):
        """Reuse the with-vectors suite's double (it models a missing vector)."""
        return TestMem0BackendScrollByMetadataWithVectors._make_mock_point(
            point_id, created_at, extra_meta, vector,
        )

    @pytest.mark.asyncio
    async def test_resolves_collection_name_from_scope_and_prefix(self, backend):
        """Collection resolution matches scroll_by_metadata's exactly."""
        client = _paging_client([([], None)])

        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=client)):
            await _drain(backend.scroll_all_by_metadata(
                Scope(project_id='dark_factory'), {'category': 'procedural_knowledge'},
            ))

        prefix = backend.config.mem0.collection_prefix
        assert client.scroll.call_args.kwargs['collection_name'] == f'{prefix}_dark_factory'

    @pytest.mark.asyncio
    async def test_scroll_filter_equals_the_count_filter_for_the_same_filters(self, backend):
        """ANTI-DRIFT at THIS entry point — the census's reconciliation depends on it.

        census_project COUNTs with count_by_metadata and SCROLLs with this
        API, then compares the two figures to decide coverage.complete.  If
        their filters drifted, the reconciliation would compare two different
        point sets and still report complete coverage.
        """
        filters = {'category': 'observations_and_summaries'}
        scope = Scope(project_id='dark_factory')

        count_client = AsyncMock()
        count_client.count = AsyncMock(return_value=MagicMock(count=0))
        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=count_client)):
            await backend.count_by_metadata(scope=scope, filters=filters)

        scroll_client = _paging_client([([], None)])
        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=scroll_client)):
            await _drain(backend.scroll_all_by_metadata(scope, filters))

        count_filter = count_client.count.call_args.kwargs.get('count_filter')
        scroll_filter = scroll_client.scroll.call_args.kwargs.get('scroll_filter')
        assert scroll_filter == count_filter == backend._build_payload_filter(filters)

    @pytest.mark.asyncio
    async def test_record_shape_is_identical_to_scroll_by_metadata(self, backend):
        """Element-wise equality between the list API and the stream API.

        Both normalise through the same helper, so the two shapes cannot
        drift — a caller migrating from one to the other sees byte-identical
        records.
        """
        def _points():
            return [
                self._make_mock_point('id-1', '2026-01-01T00:00:00+00:00', {'category': 'x'}),
                self._make_mock_point('id-2', None, {'category': 'x'}),
            ]

        filters = {'category': 'x'}
        scope = Scope(project_id='p')

        list_client = AsyncMock()
        list_client.scroll = AsyncMock(return_value=(_points(), None))
        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=list_client)):
            as_list = await backend.scroll_by_metadata(scope, filters)

        stream_client = _paging_client([(_points(), None)])
        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=stream_client)):
            as_stream = await _drain(backend.scroll_all_by_metadata(scope, filters))

        assert as_stream == as_list
        # 'metadata' is the FULL payload (created_at included); 'created_at'
        # is lifted out of it as a convenience, not moved.
        assert as_stream[0] == {
            'id': 'id-1',
            'created_at': '2026-01-01T00:00:00+00:00',
            'metadata': {'category': 'x', 'created_at': '2026-01-01T00:00:00+00:00'},
        }
        assert as_stream[1]['created_at'] is None, 'a payload with no created_at degrades to None'

    @pytest.mark.asyncio
    async def test_records_stream_across_page_boundaries_in_order(self, backend):
        """The whole point: the match set is exhausted, not capped at one page."""
        pages = [
            ([self._make_mock_point('id-1'), self._make_mock_point('id-2')], 'off-1'),
            ([self._make_mock_point('id-3')], None),
        ]
        client = _paging_client(pages)

        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=client)):
            got = await _drain(backend.scroll_all_by_metadata(
                Scope(project_id='p'), {'category': 'x'}, page_size=2,
            ))

        assert [r['id'] for r in got] == ['id-1', 'id-2', 'id-3']
        assert [c.kwargs.get('offset') for c in client.scroll.await_args_list] == [None, 'off-1']

    @pytest.mark.asyncio
    async def test_with_vectors_lifts_vectors_and_degrades_missing_ones(self, backend):
        """with_vectors=True adds 'vector'; a vector-less point degrades to None, not dropped."""
        p_ok = self._make_mock_point('id-ok', vector=[0.1, 0.2])
        p_missing = self._make_mock_point('id-missing', vector=None)
        client = _paging_client([([p_ok, p_missing], None)])

        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=client)):
            got = await _drain(backend.scroll_all_by_metadata(
                Scope(project_id='p'), {'category': 'x'}, with_vectors=True,
            ))

        assert client.scroll.call_args.kwargs['with_vectors'] is True
        assert len(got) == 2, 'a vector-less point must still be returned, not dropped'
        assert got[0]['vector'] == [0.1, 0.2]
        assert got[1]['vector'] is None

    @pytest.mark.asyncio
    async def test_no_vector_key_when_with_vectors_is_false(self, backend):
        """Default mode must not add a 'vector' key nor pay to transfer vectors."""
        client = _paging_client([([self._make_mock_point('id-1')], None)])

        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=client)):
            got = await _drain(backend.scroll_all_by_metadata(Scope(project_id='p'), {'category': 'x'}))

        assert client.scroll.call_args.kwargs['with_vectors'] is False
        assert 'vector' not in got[0], f'got keys {sorted(got[0])}'

    @pytest.mark.asyncio
    async def test_empty_filters_raises_value_error(self, backend):
        """Empty filters is rejected — it would enumerate the whole collection."""
        with pytest.raises(ValueError, match='scroll_all_by_metadata requires at least one filter'):
            await _drain(backend.scroll_all_by_metadata(Scope(project_id='p'), {}))

    @pytest.mark.asyncio
    async def test_empty_filters_raises_at_call_time_not_first_iteration(self, backend):
        """The empty-filters guard must fire when the method is CALLED.

        ``async def`` + ``yield`` makes this an async GENERATOR function, so
        calling it merely builds a generator object and runs no body at all --
        the ``if not filters`` guard does not execute until the first
        ``__anext__``. A caller that builds the generator and then
        conditionally never iterates gets NO error for a mistake the
        docstring documents as rejected, and the whole-collection
        enumeration this guard exists to prevent passes silently unflagged.

        The coroutine sibling ``scroll_by_metadata`` fails on await, so the
        two halves of the same addressing contract disagree about when a bad
        argument is a bad argument.

        No await and no _drain here: the bare call expression is the entire
        thing under test.
        """
        with pytest.raises(ValueError, match='scroll_all_by_metadata requires at least one filter'):
            backend.scroll_all_by_metadata(Scope(project_id='p'), {})

    @pytest.mark.asyncio
    async def test_still_streams_lazily_after_eager_validation(self, backend):
        """Validating eagerly must not make the SCROLL eager too.

        The argument check moves to call time; the paging stays lazy, so
        peak memory is still one page rather than the corpus. Mirrors
        test_streams_rather_than_accumulating, but additionally asserts that
        merely CONSTRUCTING the stream issues no Qdrant round-trip.
        """
        client = _paging_client([
            ([self._make_mock_point('id-1'), self._make_mock_point('id-2')], 'off-1'),
            ([self._make_mock_point('id-3')], None),
        ])

        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=client)):
            agen = backend.scroll_all_by_metadata(
                Scope(project_id='p'), {'category': 'x'}, page_size=2,
            )
            assert client.scroll.await_count == 0, (
                'building the stream must not fetch a page; '
                f'scroll was awaited {client.scroll.await_count} times'
            )
            first = await anext(agen)
            assert first['id'] == 'id-1'
            assert client.scroll.await_count == 1, (
                'consuming one record must fetch exactly one page; '
                f'scroll was awaited {client.scroll.await_count} times'
            )
            await agen.aclose()

    @pytest.mark.asyncio
    async def test_timeout_propagates_out_of_the_generator(self, backend):
        """A timed-out page must never be mistaken for the end of the stream."""
        client = AsyncMock()
        client.scroll = AsyncMock(side_effect=TimeoutError('too slow'))

        with (
            patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=client)),
            pytest.raises(TimeoutError),
        ):
            await _drain(backend.scroll_all_by_metadata(Scope(project_id='p'), {'category': 'x'}))

    @pytest.mark.asyncio
    async def test_page_budget_exhaustion_propagates_out_of_the_generator(self, backend):
        """A truncated enumeration raises rather than ending short."""
        from fused_memory.backends.mem0_client import ScrollPageBudgetExhausted

        client = _paging_client([
            ([self._make_mock_point('id-1')], 'off-1'),
            ([self._make_mock_point('id-2')], 'off-2'),
        ])

        with (
            patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=client)),
            pytest.raises(ScrollPageBudgetExhausted),
        ):
            await _drain(backend.scroll_all_by_metadata(
                Scope(project_id='p'), {'category': 'x'}, page_size=1, max_pages=2,
            ))

    @pytest.mark.asyncio
    async def test_streams_rather_than_accumulating(self, backend):
        """Consuming one record fetches ONE page — peak memory is a page, not the corpus.

        This is what a list-returning API structurally cannot offer, and the
        reason the census can fold a live corpus into counters safely.
        """
        client = _paging_client([
            ([self._make_mock_point('id-1'), self._make_mock_point('id-2')], 'off-1'),
            ([self._make_mock_point('id-3')], None),
        ])

        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=client)):
            agen = backend.scroll_all_by_metadata(
                Scope(project_id='p'), {'category': 'x'}, page_size=2,
            )
            first = await anext(agen)
            assert first['id'] == 'id-1'
            assert client.scroll.await_count == 1, (
                'consuming one record must not have drained page 2; '
                f'scroll was awaited {client.scroll.await_count} times'
            )
            await agen.aclose()


class TestMem0BackendGetPointById:
    """get_point_by_id fetches a single Qdrant point by id, returning its raw payload.

    Direct-to-Qdrant point-fetch (retrieve by id), non-semantic — mirrors the
    scroll_by_metadata / count_by_metadata timeout-propagation contract, which is
    exactly why it bypasses the timeout-swallowing Mem0Backend.get.
    """

    def _make_mock_point(self, point_id: str, payload: dict | None):
        """Create a MagicMock that mimics a Qdrant Record returned by retrieve()."""
        point = MagicMock()
        point.id = point_id
        point.payload = payload
        return point

    @pytest.mark.asyncio
    async def test_retrieves_raw_payload(self, backend):
        """get_point_by_id returns the point's full raw payload dict and calls
        client.retrieve once with the right collection/id and payload/vector flags."""
        uuid = '77a3f6bc-0000-0000-0000-000000000000'
        payload = {'data': 'txt', 'category': 'observations_and_summaries', 'agent_id': 'a'}
        point = self._make_mock_point(uuid, payload)

        mock_client = AsyncMock()
        mock_client.retrieve = AsyncMock(return_value=[point])

        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=mock_client)):
            result = await backend.get_point_by_id(uuid, Scope(project_id='dark_factory'))

        assert result == payload, f'expected raw payload {payload!r}, got {result!r}'
        assert mock_client.retrieve.await_count == 1
        call_kwargs = mock_client.retrieve.call_args.kwargs
        collection_prefix = backend.config.mem0.collection_prefix
        assert call_kwargs.get('collection_name') == f'{collection_prefix}_dark_factory'
        assert call_kwargs.get('ids') == [uuid]
        assert call_kwargs.get('with_payload') is True
        assert call_kwargs.get('with_vectors') is False

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self, backend):
        """An empty retrieve result -> None (genuine miss)."""
        mock_client = AsyncMock()
        mock_client.retrieve = AsyncMock(return_value=[])

        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=mock_client)):
            result = await backend.get_point_by_id(
                '00000000-0000-0000-0000-000000000000',
                Scope(project_id='dark_factory'),
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_timeout_propagates_not_swallowed(self, backend):
        """On TimeoutError the exception propagates — it is NOT swallowed into None.

        Mirrors scroll_by_metadata/count_by_metadata's propagate-by-default
        contract (no try/except around asyncio.wait_for): a timed-out point-fetch
        must never be indistinguishable from a genuine not-found (no-silent-fail
        invariant), which is exactly why this bypasses the timeout-swallowing
        Mem0Backend.get.
        """
        mock_client = AsyncMock()
        mock_client.retrieve = AsyncMock(side_effect=TimeoutError('too slow'))

        with (
            patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=mock_client)),
            pytest.raises(TimeoutError),
        ):
            await backend.get_point_by_id(
                '00000000-0000-0000-0000-000000000000',
                Scope(project_id='dark_factory'),
            )

    @pytest.mark.asyncio
    async def test_multiple_records_uses_first_and_warns(self, backend, caplog):
        """A single-id retrieve returning >1 record is unexpected: use records[0]'s
        payload and log a WARNING (defense-in-depth, mirroring scroll_by_metadata)."""
        uuid = '77a3f6bc-0000-0000-0000-000000000000'
        first_payload = {'data': 'first', 'category': 'observations_and_summaries'}
        second_payload = {'data': 'second', 'category': 'observations_and_summaries'}
        p1 = self._make_mock_point(uuid, first_payload)
        p2 = self._make_mock_point(uuid, second_payload)

        mock_client = AsyncMock()
        mock_client.retrieve = AsyncMock(return_value=[p1, p2])

        with (
            patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=mock_client)),
            caplog.at_level(logging.WARNING),
        ):
            result = await backend.get_point_by_id(uuid, Scope(project_id='dark_factory'))

        assert result == first_payload, f'expected the first record payload, got {result!r}'
        assert any(
            'retrieved 2 points for a single id' in rec.getMessage()
            for rec in caplog.records
        ), f'expected a >1-point WARNING, got {[r.getMessage() for r in caplog.records]!r}'


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


class TestMem0BackendPayloadPrimitives:
    """Payload-only writes that never re-embed (task 3088).

    ``Mem0Backend.update`` goes through mem0's ``AsyncMemory.update``, which
    re-embeds the content, rewrites ``updated_at`` and appends a mem0 history
    row. For a purely-cosmetic metadata patch (e.g. tagging a survivor with
    ``topic=``) all three are waste, so the metadata-only arms of
    ``update_memory`` route straight to Qdrant's payload APIs instead. Named
    1:1 after the Qdrant primitives they wrap so the decision doc's §5(b)
    routing table reads directly against the code.
    """

    UUID = '77a3f6bc-0000-0000-0000-000000000000'

    def _mocks(self, backend):
        mock_client = AsyncMock()
        mock_instance = MagicMock()
        mock_instance.update = AsyncMock()
        return mock_client, mock_instance

    @pytest.mark.asyncio
    async def test_set_payload_merges_via_qdrant(self, backend):
        mock_client, mock_instance = self._mocks(backend)
        get_instance = AsyncMock(return_value=mock_instance)

        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=mock_client)), \
                patch.object(backend, '_get_instance', get_instance):
            await backend.set_payload(
                self.UUID, {'topic': 'docs-prd-landing'}, Scope(project_id='dark_factory'),
            )

        assert mock_client.set_payload.await_count == 1
        kwargs = mock_client.set_payload.call_args.kwargs
        prefix = backend.config.mem0.collection_prefix
        assert kwargs.get('collection_name') == f'{prefix}_dark_factory'
        assert kwargs.get('payload') == {'topic': 'docs-prd-landing'}
        assert kwargs.get('points') == [self.UUID], (
            f'the point id must pass through unchanged, got {kwargs.get("points")!r}'
        )
        get_instance.assert_not_called()
        mock_instance.update.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_payload_removes_named_keys(self, backend):
        mock_client, mock_instance = self._mocks(backend)
        get_instance = AsyncMock(return_value=mock_instance)

        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=mock_client)), \
                patch.object(backend, '_get_instance', get_instance):
            await backend.delete_payload(
                self.UUID, ['topic', 'kind'], Scope(project_id='dark_factory'),
            )

        assert mock_client.delete_payload.await_count == 1
        kwargs = mock_client.delete_payload.call_args.kwargs
        prefix = backend.config.mem0.collection_prefix
        assert kwargs.get('collection_name') == f'{prefix}_dark_factory'
        assert kwargs.get('keys') == ['topic', 'kind']
        assert kwargs.get('points') == [self.UUID]
        get_instance.assert_not_called()
        mock_instance.update.assert_not_called()

    @pytest.mark.asyncio
    async def test_overwrite_payload_replaces_whole_payload(self, backend):
        mock_client, mock_instance = self._mocks(backend)
        get_instance = AsyncMock(return_value=mock_instance)
        full = {'data': 'txt', 'created_at': 'c', 'topic': 'new'}

        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=mock_client)), \
                patch.object(backend, '_get_instance', get_instance):
            await backend.overwrite_payload(
                self.UUID, full, Scope(project_id='dark_factory'),
            )

        assert mock_client.overwrite_payload.await_count == 1
        kwargs = mock_client.overwrite_payload.call_args.kwargs
        prefix = backend.config.mem0.collection_prefix
        assert kwargs.get('collection_name') == f'{prefix}_dark_factory'
        assert kwargs.get('payload') == full
        assert kwargs.get('points') == [self.UUID]
        get_instance.assert_not_called()
        mock_instance.update.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('method', 'args'),
        [
            ('set_payload', ({'topic': 't'},)),
            ('delete_payload', (['topic'],)),
            ('overwrite_payload', ({'topic': 't'},)),
        ],
    )
    async def test_timeout_propagates(self, backend, method, args):
        """A write timeout must PROPAGATE, never be swallowed into a falsy return
        — the house posture on this file (get_point_by_id), in deliberate
        contrast to get() which does swallow."""
        mock_client = AsyncMock()
        setattr(mock_client, method, AsyncMock(side_effect=TimeoutError))

        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=mock_client)), \
                pytest.raises(TimeoutError):
            await getattr(backend, method)(
                self.UUID, *args, Scope(project_id='dark_factory'),
            )


class TestMem0ManagedMetadataKeys:
    """The mem0-owned payload-key set, extracted to its single home (task 3088).

    Value-pins the key set mem0's ``AsyncMemory._update_memory`` recomputes or
    restores (mem0ai 1.0.11, ``mem0/memory/main.py:2461-2481``). Previously this
    lived privately in ``scripts/tag_cgl_eta_rehome_scope.py``; the in-place
    ``update_memory`` tool is a second consumer, so per decision doc
    ``plans/mem0-in-place-update-decision.md`` §6 it moves next to
    ``Mem0Backend.update``, whose docstring already documents the very
    constraint it encodes. One definition repo-wide (INV-5).
    """

    def test_is_frozenset_with_exact_membership(self):
        from fused_memory.backends.mem0_client import MEM0_MANAGED_METADATA_KEYS

        assert isinstance(MEM0_MANAGED_METADATA_KEYS, frozenset), (
            f'expected a frozenset, got {type(MEM0_MANAGED_METADATA_KEYS).__name__}'
        )
        assert set(MEM0_MANAGED_METADATA_KEYS) == {
            'data', 'hash', 'created_at', 'updated_at',
            'user_id', 'agent_id', 'run_id', 'actor_id', 'role',
        }, f'unexpected membership: {sorted(MEM0_MANAGED_METADATA_KEYS)}'

    def test_split_partitions_managed_from_custom(self):
        from fused_memory.backends.mem0_client import split_managed_metadata

        payload = {
            'data': 'content', 'hash': 'h', 'created_at': 'c', 'updated_at': 'u',
            'user_id': 'p', 'kind': 'canonical', 'src_project': 'reify', 'topic': 't',
        }
        managed, custom = split_managed_metadata(payload)
        assert managed == {
            'data': 'content', 'hash': 'h', 'created_at': 'c',
            'updated_at': 'u', 'user_id': 'p',
        }, f'unexpected managed subset: {managed!r}'
        assert custom == {
            'kind': 'canonical', 'src_project': 'reify', 'topic': 't',
        }, f'unexpected custom subset: {custom!r}'

    def test_unknown_keys_land_in_custom(self):
        from fused_memory.backends.mem0_client import split_managed_metadata

        managed, custom = split_managed_metadata({'wholly_novel_key': 1})
        assert managed == {}, f'expected no managed keys, got {managed!r}'
        assert custom == {'wholly_novel_key': 1}, (
            f'an unknown key must be treated as custom (preserved), got {custom!r}'
        )

    def test_empty_payload_yields_two_empty_dicts(self):
        from fused_memory.backends.mem0_client import split_managed_metadata

        assert split_managed_metadata({}) == ({}, {})

    def test_neither_result_aliases_the_input(self):
        """Callers mutate the custom subset in place (the merge/delete arms), so
        neither returned dict may alias the caller's payload."""
        from fused_memory.backends.mem0_client import split_managed_metadata

        payload = {'data': 'content', 'kind': 'canonical'}
        managed, custom = split_managed_metadata(payload)
        assert managed is not payload and custom is not payload
        custom['kind'] = 'mutated'
        managed['data'] = 'mutated'
        assert payload == {'data': 'content', 'kind': 'canonical'}, (
            f'mutating a returned subset must not touch the input, got {payload!r}'
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


class TestIsMissingCollectionError:
    """`is_missing_collection_error` tells a genuinely-absent Qdrant collection
    ("zero memories" — a true empty result) apart from a real backend failure
    that callers must still surface per no-silent-fail (task 2949).

    step-1 (RED): the predicate does not exist yet.
    """

    def test_missing_collection_404_is_true(self):
        """Qdrant's stable missing-collection 404 is the ONLY matching shape."""
        exc = UnexpectedResponse(
            404,
            'Not Found',
            b"Collection `fused_solar_challenge` doesn't exist!",
            httpx.Headers(),
        )
        assert is_missing_collection_error(exc) is True

    def test_non_404_unexpected_response_is_false(self):
        """A 500 is a real backend failure, not an empty collection."""
        exc = UnexpectedResponse(500, 'Internal Server Error', b'boom', httpx.Headers())
        assert is_missing_collection_error(exc) is False

    def test_unrelated_404_is_false(self):
        """The status code alone is NOT enough — the message must name a
        collection.

        Qdrant answers 404 for several absent resources, and only the missing
        COLLECTION means "zero memories".  Reading any other 404 as an empty
        result would be exactly the silent fail-soft this predicate exists to
        avoid, so widening the match to `status_code == 404` must fail here.
        """
        exc = UnexpectedResponse(
            404,
            'Not Found',
            b'{"status":{"error":"Not found: Point 42 does not exist"}}',
            httpx.Headers(),
        )
        assert is_missing_collection_error(exc) is False

    def test_missing_snapshot_404_is_false(self):
        """Boundary on the phrase itself: Qdrant words a missing SNAPSHOT with
        the same "doesn't exist!" wording, so the phrase alone cannot decide.
        """
        exc = UnexpectedResponse(
            404,
            'Not Found',
            b"Snapshot `daily-2026-08-09` doesn't exist!",
            httpx.Headers(),
        )
        assert is_missing_collection_error(exc) is False

    def test_str_content_is_decoded_leniently(self):
        """`.content` is typed bytes but assigned verbatim by qdrant_client, so
        an already-decoded str must still match rather than crash the caller's
        error path.
        """
        exc = UnexpectedResponse(
            404,
            'Not Found',
            "Collection `fused_solar_challenge` doesn't exist!",  # type: ignore[arg-type]
            httpx.Headers(),
        )
        assert is_missing_collection_error(exc) is True

    def test_none_content_is_false(self):
        """A 404 carrying no body proves nothing about a collection, and must
        not blow up the predicate on the way to returning False.
        """
        exc = UnexpectedResponse(404, 'Not Found', None, httpx.Headers())  # type: ignore[arg-type]
        assert is_missing_collection_error(exc) is False

    def test_generic_exception_is_false(self):
        assert is_missing_collection_error(RuntimeError('qdrant down')) is False

    def test_timeout_error_is_false(self):
        """A timeout must keep propagating — it is not evidence of 'no data'."""
        assert is_missing_collection_error(TimeoutError()) is False


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
    # Race-safe vector-dim read: under `-n auto` load the just-created
    # collection can transiently 404 before it is durably readable, so retry
    # the read via the shared bounded-poll helper rather than a single get.
    client = QdrantClient(url=QDRANT_URL, timeout=10)
    try:
        dim = await collection_vector_size(client, collection)
    finally:
        client.close()
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

    pytestmark = [qdrant_skipif(), pytest.mark.timeout(60), pytest.mark.integration]

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


# ---------------------------------------------------------------------------
# scan_payload_text — literal substring scan over Qdrant payload TEXT
# (task 3083, WORK b).
#
# The capability gap this closes: Mem0Backend could search SEMANTICALLY
# (`search`), match METADATA equality (`count_by_metadata`/`scroll_by_metadata`),
# and fetch a POINT BY ID (`get_point_by_id`) — but nothing could match payload
# TEXT. A leaked tool-call fragment carries almost no semantic signal, so the
# corpus was structurally unsweepable for it.
#
# Sentinel literals use the \x3c escape for "<" (task 3083 plan pre-1).
# ---------------------------------------------------------------------------

_SCAN_CLOSE_CONTENT = '\x3c/content>'
_SCAN_CLOSE_INVOKE = '\x3c/invoke>'
_SCAN_OPEN_CONTENT = '\x3cparameter name="content">'

_SCAN_BODY = 'A memory about the reconciliation grace window.'
# The c759c53b shape: body, closing content tag, real newline, bare closing
# invoke tag.
_SCAN_LEAK_FRAGMENT = _SCAN_CLOSE_CONTENT + '\n' + _SCAN_CLOSE_INVOKE
_SCAN_LEAK_TEXT = _SCAN_BODY + _SCAN_LEAK_FRAGMENT
# The 9f2d2ae6 shape: body, tag, newline, an opening content parameter, then a
# verbatim duplicate of the body.
_SCAN_DUP_TEXT = _SCAN_BODY + _SCAN_CLOSE_CONTENT + '\n' + _SCAN_OPEN_CONTENT + _SCAN_BODY


def _scan_point(point_id: str, payload: dict):
    """Create a MagicMock that mimics a Qdrant Record."""
    point = MagicMock()
    point.id = point_id
    point.payload = payload
    return point


def _stub_pager(points=(), raises=None):
    """Stand-in for ``scroll_collection_pages``: records its call args, yields
    *points*, then optionally raises *raises*.

    A plain factory rather than an AsyncMock because the real method is an
    async GENERATOR: calling it must return an async iterator, not a coroutine.
    """
    calls: list = []

    def _factory(*args, **kwargs):
        calls.append((args, kwargs))

        async def _gen():
            for point in points:
                yield point
            if raises is not None:
                raise raises

        return _gen()

    _factory.calls = calls
    return _factory


class TestMem0BackendScanPayloadText:
    """Literal payload-text scan: prefilter, mandatory re-verify, pagination."""

    @pytest.mark.asyncio
    async def test_prefilter_builds_should_matchtext_conditions(self, backend):
        """One FieldCondition per needle, OR-combined via `should`.

        MatchText (not MatchValue) is load-bearing: MatchValue is exact
        equality on the whole field, which can never find a fragment embedded
        in a longer memory.
        """
        from qdrant_client.http import models as qmodels

        mock_client = AsyncMock()
        mock_client.scroll = AsyncMock(return_value=([], None))

        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=mock_client)):
            await backend.scan_payload_text(
                scope=Scope(project_id='dark_factory'),
                needles=[_SCAN_CLOSE_CONTENT, _SCAN_CLOSE_INVOKE],
            )

        call_kwargs = mock_client.scroll.call_args.kwargs
        collection_prefix = backend.config.mem0.collection_prefix
        assert call_kwargs.get('collection_name') == f'{collection_prefix}_dark_factory'
        scroll_filter = call_kwargs.get('scroll_filter')
        assert isinstance(scroll_filter, qmodels.Filter)
        assert isinstance(scroll_filter.should, list)
        assert len(scroll_filter.should) == 2
        for cond, needle in zip(
            scroll_filter.should, [_SCAN_CLOSE_CONTENT, _SCAN_CLOSE_INVOKE], strict=True
        ):
            assert isinstance(cond, qmodels.FieldCondition)
            assert cond.key == 'data'
            assert isinstance(cond.match, qmodels.MatchText)
            assert cond.match.text == needle

    @pytest.mark.asyncio
    async def test_scan_filter_equals_the_count_filter_for_the_same_filters(self, backend):
        """ANTI-DRIFT (INV-5): scan's scroll_filter == count's count_filter.

        The fourth entry point onto ``_build_payload_filter``.  The same
        reconciliation hazard the other three are pinned against applies here
        with a sharper edge: ``scan_payload_text``'s whole purpose is a TRUE
        incidence rate, and an incidence rate is a ratio of a scan against a
        count.  If the two filter constructions ever selected different point
        sets the ratio would be wrong with no error surface at all.

        Exhaustive mode is the one that matters: it is the mode used to
        establish the true rate, and it is the one where scan previously
        emitted an empty ``should`` arm that made this equality false.
        """
        filters = {'category': 'procedural_knowledge', 'recon_pool': 'stage2_cycle_summary'}
        scope = Scope(project_id='dark_factory')

        count_client = AsyncMock()
        count_client.count = AsyncMock(return_value=MagicMock(count=0))
        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=count_client)):
            await backend.count_by_metadata(scope=scope, filters=filters)

        scan_client = AsyncMock()
        scan_client.scroll = AsyncMock(return_value=([], None))
        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=scan_client)):
            await backend.scan_payload_text(scope=scope, filters=filters, exhaustive=True)

        count_filter = count_client.count.call_args.kwargs.get('count_filter')
        scan_filter = scan_client.scroll.call_args.kwargs.get('scroll_filter')
        built = backend._build_payload_filter(filters)

        assert scan_filter == built, (
            f'scan_payload_text built {scan_filter!r}, _build_payload_filter built {built!r}'
        )
        assert scan_filter == count_filter, (
            'an exhaustive scan and the count it is divided by must select the same '
            f'point set; got scan_filter={scan_filter!r} count_filter={count_filter!r}'
        )

    @pytest.mark.asyncio
    async def test_prefilter_scan_filter_comes_from_the_shared_builder(self, backend):
        """The prefilter mode routes through the same home, needles and all."""
        filters = {'category': 'procedural_knowledge'}
        needles = [_SCAN_CLOSE_CONTENT, _SCAN_CLOSE_INVOKE]

        scan_client = AsyncMock()
        scan_client.scroll = AsyncMock(return_value=([], None))
        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=scan_client)):
            await backend.scan_payload_text(
                scope=Scope(project_id='p'), needles=needles, filters=filters
            )

        scan_filter = scan_client.scroll.call_args.kwargs.get('scroll_filter')
        built = backend._build_payload_filter(filters, text_needles=needles)

        assert scan_filter == built, (
            f'scan_payload_text built {scan_filter!r}, _build_payload_filter built {built!r}'
        )

    @pytest.mark.asyncio
    async def test_metadata_filters_are_anded_in_via_must(self, backend):
        """An optional `filters` dict narrows the scan, using the same `must`
        key-equality list count_by_metadata builds."""
        from qdrant_client.http import models as qmodels

        mock_client = AsyncMock()
        mock_client.scroll = AsyncMock(return_value=([], None))

        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=mock_client)):
            await backend.scan_payload_text(
                scope=Scope(project_id='p'),
                needles=[_SCAN_CLOSE_CONTENT],
                filters={'category': 'procedural_knowledge'},
            )

        scroll_filter = mock_client.scroll.call_args.kwargs.get('scroll_filter')
        assert isinstance(scroll_filter.must, list)
        assert len(scroll_filter.must) == 1
        cond = scroll_filter.must[0]
        assert isinstance(cond, qmodels.FieldCondition)
        assert cond.key == 'category'
        assert isinstance(cond.match, qmodels.MatchValue)
        assert cond.match.value == 'procedural_knowledge'
        # The needle prefilter is still OR-ed inside the same filter.
        assert scroll_filter.should and len(scroll_filter.should) == 1

    @pytest.mark.asyncio
    async def test_exhaustive_mode_passes_no_scroll_filter(self, backend):
        """exhaustive=True skips the prefilter entirely so the answer rests on
        nothing but the shared Python detector — the mode for establishing the
        true incidence rate."""
        mock_client = AsyncMock()
        mock_client.scroll = AsyncMock(return_value=([], None))

        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=mock_client)):
            await backend.scan_payload_text(scope=Scope(project_id='p'), exhaustive=True)

        assert mock_client.scroll.call_args.kwargs.get('scroll_filter') is None

    @pytest.mark.asyncio
    async def test_metadata_filters_still_apply_in_exhaustive_mode(self, backend):
        mock_client = AsyncMock()
        mock_client.scroll = AsyncMock(return_value=([], None))

        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=mock_client)):
            await backend.scan_payload_text(
                scope=Scope(project_id='p'), exhaustive=True, filters={'category': 'x'}
            )

        scroll_filter = mock_client.scroll.call_args.kwargs.get('scroll_filter')
        assert scroll_filter is not None
        assert len(scroll_filter.must) == 1
        assert not scroll_filter.should

    @pytest.mark.asyncio
    async def test_paginates_until_next_page_offset_is_exhausted(self, backend):
        """Mandatory in both modes: the point of this capability is the TRUE
        incidence rate, and a silently-capped scan answers that wrongly — the
        same silent-wrong-value class this whole task is about.

        scroll_by_metadata discards next_page_offset and caps at limit=1000;
        that single-shot behaviour is deliberately NOT reused here.
        """
        page1 = [_scan_point('id-1', {'data': _SCAN_LEAK_TEXT})]
        page2 = [_scan_point('id-2', {'data': _SCAN_DUP_TEXT})]
        mock_client = AsyncMock()
        mock_client.scroll = AsyncMock(side_effect=[(page1, 'cursor1'), (page2, None)])

        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=mock_client)):
            result = await backend.scan_payload_text(scope=Scope(project_id='p'))

        assert mock_client.scroll.await_count == 2
        first, second = mock_client.scroll.call_args_list
        assert first.kwargs.get('offset') is None
        assert second.kwargs.get('offset') == 'cursor1'
        assert [m['id'] for m in result['matches']] == ['id-1', 'id-2']
        assert result['scanned'] == 2
        assert result['truncated'] is False

    # -- the walk is delegated, not duplicated (task 3682) -----------------

    @pytest.mark.asyncio
    async def test_the_walk_is_delegated_to_the_shared_pager(self, backend):
        """SINGLE HOME: scan drives scroll_collection_pages, not its own loop.

        The caller's ``limit`` rides on the pager's ``max_points``, which is
        why the cap could be pushed down there rather than layered on top with
        a ``break``.  The raw-client assertion is the load-bearing half: it is
        what proves no second copy of the walk survives inside this method.
        """
        from fused_memory.utils.toolcall_xml_leak import PREFILTER_NEEDLES

        filters = {'category': 'procedural_knowledge'}
        pager = _stub_pager(points=[_scan_point('id-1', {'data': _SCAN_LEAK_TEXT})])
        raw_client = AsyncMock()
        raw_client.scroll = AsyncMock(return_value=([], None))

        with (
            patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=raw_client)),
            patch.object(backend, 'scroll_collection_pages', pager),
        ):
            result = await backend.scan_payload_text(
                scope=Scope(project_id='p'), filters=filters, page_size=64, limit=10
            )

        assert len(pager.calls) == 1, f'expected exactly one pager call, got {pager.calls!r}'
        args, kwargs = pager.calls[0]
        collection_prefix = backend.config.mem0.collection_prefix
        assert args[0] == f'{collection_prefix}_p'
        assert kwargs['page_size'] == 64
        assert kwargs['max_points'] == 10, (
            "the caller's limit must ride on the pager's points cap, not a "
            f'caller-side break; got {kwargs.get("max_points")!r}'
        )
        assert kwargs['with_vectors'] is False
        assert kwargs['scroll_filter'] == backend._build_payload_filter(
            filters, text_needles=list(PREFILTER_NEEDLES)
        )
        raw_client.scroll.assert_not_awaited()
        assert result['scanned'] == 1

    @pytest.mark.asyncio
    async def test_the_points_cap_becomes_a_truncated_flag_not_a_raise(self, backend, caplog):
        """POSTURE: scan converts the points cap into its own flag + WARNING.

        The primitive raises so it can never truncate silently; THIS caller
        chooses to report that as a normal capped result, because being
        stopped by a limit the caller itself passed is an expected outcome.
        The exception must never reach the caller.
        """
        from fused_memory.backends.mem0_client import ScrollPointBudgetExhausted

        points = [_scan_point(f'id-{i}', {'data': _SCAN_LEAK_TEXT}) for i in range(3)]
        pager = _stub_pager(points=points, raises=ScrollPointBudgetExhausted('capped'))

        with (
            patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=AsyncMock())),
            patch.object(backend, 'scroll_collection_pages', pager),
            caplog.at_level(logging.WARNING),
        ):
            result = await backend.scan_payload_text(scope=Scope(project_id='p'), limit=3)

        assert result['truncated'] is True
        assert result['scanned'] == 3, (
            'every point yielded before the raise is counted, so `scanned` stays '
            f'an exact incidence-rate denominator; got {result["scanned"]!r}'
        )
        assert len(result['matches']) == 3
        assert any('truncat' in r.message.lower() for r in caplog.records), caplog.text

    @pytest.mark.asyncio
    async def test_the_points_cap_handler_does_not_swallow_other_failures(self, backend):
        """NARROW EXCEPT: only the points cap is converted; nothing else.

        A broad ``except Exception`` around the walk would fold a timed-out or
        otherwise-failed scan into a clean-looking capped result — the exact
        silent-wrong-value class this scan exists to measure.  Pinned here
        against a sentinel the handler must not know about; the live timeout
        path is covered by test_timeout_propagates_not_swallowed.
        """
        class _Sentinel(RuntimeError):
            pass

        pager = _stub_pager(
            points=[_scan_point('id-1', {'data': _SCAN_LEAK_TEXT})], raises=_Sentinel('boom')
        )

        with (
            patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=AsyncMock())),
            patch.object(backend, 'scroll_collection_pages', pager),
            pytest.raises(_Sentinel),
        ):
            await backend.scan_payload_text(scope=Scope(project_id='p'))

    @pytest.mark.asyncio
    async def test_max_pages_is_accepted_and_forwarded_to_the_pager(self, backend):
        """The page budget scan inherits from the shared pager is steerable.

        At the default page_size=256 and max_pages=200 the ceiling is 51,200
        points against a ~19,321-point live corpus (2.6x headroom), so the
        default is not reachable today — but it is a real ceiling, and a
        caller that outgrows it needs an escape hatch that is not "copy the
        walk back out".
        """
        from fused_memory.backends.mem0_client import DEFAULT_SCROLL_MAX_PAGES

        pager = _stub_pager()
        with (
            patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=AsyncMock())),
            patch.object(backend, 'scroll_collection_pages', pager),
        ):
            await backend.scan_payload_text(scope=Scope(project_id='p'))
        assert pager.calls[0][1]['max_pages'] == DEFAULT_SCROLL_MAX_PAGES

        pager2 = _stub_pager()
        with (
            patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=AsyncMock())),
            patch.object(backend, 'scroll_collection_pages', pager2),
        ):
            await backend.scan_payload_text(scope=Scope(project_id='p'), max_pages=7)
        assert pager2.calls[0][1]['max_pages'] == 7

    @pytest.mark.asyncio
    async def test_page_budget_exhaustion_propagates_rather_than_flagging_truncated(
        self, backend
    ):
        """The two truncation events must stay DISTINGUISHABLE at this caller.

        A caller `limit` is something the caller asked for, so it is reported
        as a normal capped result.  The page budget is a safety backstop
        nobody asked for; reporting it the same way would hand a sweep a
        plausible-looking undercount carrying a `truncated` flag it was not
        told to expect — the exact silent-wrong-value class this scan exists
        to measure.  So it PROPAGATES.
        """
        from fused_memory.backends.mem0_client import ScrollPageBudgetExhausted

        # Pages that never exhaust: next_offset stays live forever.
        client = AsyncMock()
        client.scroll = AsyncMock(
            return_value=([_scan_point('id-1', {'data': _SCAN_LEAK_TEXT})], 'cursor-forever')
        )

        with (
            patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=client)),
            pytest.raises(ScrollPageBudgetExhausted),
        ):
            await backend.scan_payload_text(scope=Scope(project_id='p'), max_pages=3)

        assert client.scroll.await_count == 3

    @pytest.mark.asyncio
    async def test_a_limit_reached_first_is_the_quiet_flagged_form_not_a_raise(
        self, backend, caplog
    ):
        """Both budgets in play, `limit` reached first: flag, not raise.

        Pins the precedence chosen inside the pager — the caller's explicit
        cap outranks the backstop — as observed from this caller.
        """
        client = AsyncMock()
        client.scroll = AsyncMock(
            return_value=([_scan_point('id-1', {'data': _SCAN_LEAK_TEXT})], 'cursor-forever')
        )

        with (
            patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=client)),
            caplog.at_level(logging.WARNING),
        ):
            result = await backend.scan_payload_text(
                scope=Scope(project_id='p'), limit=1, max_pages=3
            )

        assert result['truncated'] is True
        assert result['scanned'] == 1
        assert client.scroll.await_count == 1
        assert any('truncat' in r.message.lower() for r in caplog.records), caplog.text

    @pytest.mark.asyncio
    async def test_page_size_is_forwarded_as_scroll_limit(self, backend):
        mock_client = AsyncMock()
        mock_client.scroll = AsyncMock(return_value=([], None))

        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=mock_client)):
            await backend.scan_payload_text(scope=Scope(project_id='p'), page_size=64)

        assert mock_client.scroll.call_args.kwargs.get('limit') == 64

    @pytest.mark.asyncio
    async def test_always_requests_payload_without_vectors(self, backend):
        """Vectors are dead weight for a text scan over ~19k points."""
        mock_client = AsyncMock()
        mock_client.scroll = AsyncMock(return_value=([], None))

        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=mock_client)):
            await backend.scan_payload_text(scope=Scope(project_id='p'))

        call_kwargs = mock_client.scroll.call_args.kwargs
        assert call_kwargs.get('with_payload') is True
        assert call_kwargs.get('with_vectors') is False

    @pytest.mark.asyncio
    async def test_prefilter_hits_are_re_verified_by_the_python_detector(self, backend):
        """THE authoritative verdict is the shared detector, never the prefilter.

        This fails SAFE if a text payload index is ever created on `data` and
        MatchText flips from substring to tokenized matching: tokenized is
        strictly MORE permissive for these needles, so the prefilter stays a
        superset and this re-verify still yields the exact answer.
        """
        genuine = _scan_point('real', {'data': _SCAN_LEAK_TEXT})
        # Contains the word "content" and even the bare closing tag, but no
        # leak: no continuation after real whitespace.
        decoy = _scan_point('decoy', {'data': 'Talking about content' + _SCAN_CLOSE_CONTENT})
        # The tasks 2938/2939 shape: quotes the leak with an ESCAPED newline.
        prose = _scan_point(
            'prose', {'data': 'Quoted: `' + _SCAN_CLOSE_CONTENT + '\\n' + _SCAN_CLOSE_INVOKE + '`.'}
        )
        mock_client = AsyncMock()
        mock_client.scroll = AsyncMock(return_value=([genuine, decoy, prose], None))

        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=mock_client)):
            result = await backend.scan_payload_text(scope=Scope(project_id='p'))

        assert [m['id'] for m in result['matches']] == ['real']
        # `scanned` counts everything walked, so the incidence rate has a
        # correct denominator even though only one record matched.
        assert result['scanned'] == 3

    @pytest.mark.asyncio
    @pytest.mark.parametrize('key', ['data', 'memory', 'content'])
    async def test_content_is_read_via_the_canonical_key_order(self, backend, key):
        mock_client = AsyncMock()
        mock_client.scroll = AsyncMock(return_value=([_scan_point('p1', {key: _SCAN_LEAK_TEXT})], None))

        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=mock_client)):
            result = await backend.scan_payload_text(scope=Scope(project_id='p'))

        assert [m['id'] for m in result['matches']] == ['p1']

    @pytest.mark.asyncio
    async def test_data_wins_over_memory_and_content(self, backend):
        """`data` is the canonical Qdrant payload key for memory text
        (memory_service._MEM0_CONTENT_KEYS)."""
        point = _scan_point('p1', {'data': 'clean text', 'memory': _SCAN_LEAK_TEXT})
        mock_client = AsyncMock()
        mock_client.scroll = AsyncMock(return_value=([point], None))

        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=mock_client)):
            result = await backend.scan_payload_text(scope=Scope(project_id='p'))

        assert result['matches'] == []

    @pytest.mark.asyncio
    async def test_hit_shape_carries_id_created_at_fragments_excerpt_metadata(self, backend):
        payload = {
            'data': _SCAN_LEAK_TEXT,
            'created_at': '2026-07-27T00:00:00+00:00',
            'category': 'observations_and_summaries',
            'agent_id': 'claude-interactive',
        }
        mock_client = AsyncMock()
        mock_client.scroll = AsyncMock(return_value=([_scan_point('id-9', payload)], None))

        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=mock_client)):
            result = await backend.scan_payload_text(scope=Scope(project_id='p'))

        hit = result['matches'][0]
        assert hit['id'] == 'id-9'
        assert hit['created_at'] == '2026-07-27T00:00:00+00:00'
        assert hit['matched_fragments'] == [_SCAN_LEAK_FRAGMENT]
        assert _SCAN_BODY[:20] in hit['excerpt']
        assert hit['metadata']['category'] == 'observations_and_summaries'
        assert hit['metadata']['agent_id'] == 'claude-interactive'

    @pytest.mark.asyncio
    async def test_payloadless_point_does_not_raise(self, backend):
        point = MagicMock()
        point.id = 'no-payload'
        point.payload = None
        mock_client = AsyncMock()
        mock_client.scroll = AsyncMock(return_value=([point], None))

        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=mock_client)):
            result = await backend.scan_payload_text(scope=Scope(project_id='p'))

        assert result['matches'] == []
        assert result['scanned'] == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize('needles', [None, []])
    async def test_absent_needles_default_to_the_shared_sentinels(self, backend, needles):
        """Defaulting to an empty needle list would scan for nothing and report
        a clean corpus — a silent false negative."""
        from qdrant_client.http import models as qmodels

        from fused_memory.utils.toolcall_xml_leak import PREFILTER_NEEDLES

        mock_client = AsyncMock()
        mock_client.scroll = AsyncMock(return_value=([], None))

        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=mock_client)):
            await backend.scan_payload_text(scope=Scope(project_id='p'), needles=needles)

        scroll_filter = mock_client.scroll.call_args.kwargs.get('scroll_filter')
        assert isinstance(scroll_filter, qmodels.Filter)
        assert isinstance(scroll_filter.should, list)
        assert len(scroll_filter.should) == len(PREFILTER_NEEDLES)
        for cond, needle in zip(scroll_filter.should, PREFILTER_NEEDLES, strict=True):
            assert isinstance(cond, qmodels.FieldCondition)
            assert isinstance(cond.match, qmodels.MatchText)
            assert cond.match.text == needle

    @pytest.mark.asyncio
    async def test_timeout_propagates_not_swallowed(self, backend):
        """The load-bearing raw-Qdrant invariant: no try/except around
        asyncio.wait_for, so a timed-out scan is never mistaken for a clean
        corpus (which is exactly the wrong answer for an incidence sweep)."""
        mock_client = AsyncMock()
        mock_client.scroll = AsyncMock(side_effect=TimeoutError('too slow'))

        with (
            patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=mock_client)),
            pytest.raises(TimeoutError),
        ):
            await backend.scan_payload_text(scope=Scope(project_id='p'))

    @pytest.mark.asyncio
    async def test_timeout_on_a_later_page_also_propagates(self, backend):
        page1 = [_scan_point('id-1', {'data': _SCAN_LEAK_TEXT})]
        mock_client = AsyncMock()
        mock_client.scroll = AsyncMock(side_effect=[(page1, 'cursor1'), TimeoutError('too slow')])

        with (
            patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=mock_client)),
            pytest.raises(TimeoutError),
        ):
            await backend.scan_payload_text(scope=Scope(project_id='p'))

    @pytest.mark.asyncio
    async def test_explicit_limit_truncates_loudly(self, backend, caplog):
        """No silent caps: a truncated walk must be self-disclosed in the
        result AND logged, because a silently-capped scan produces a
        plausible-looking undercount of the true incidence rate."""
        page1 = [_scan_point(f'id-{i}', {'data': _SCAN_LEAK_TEXT}) for i in range(3)]
        mock_client = AsyncMock()
        mock_client.scroll = AsyncMock(side_effect=[(page1, 'cursor1'), (page1, None)])

        with (
            patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=mock_client)),
            caplog.at_level(logging.WARNING),
        ):
            result = await backend.scan_payload_text(scope=Scope(project_id='p'), limit=3)

        assert result['truncated'] is True
        assert result['scanned'] == 3
        assert mock_client.scroll.await_count == 1
        assert any('truncat' in r.message.lower() for r in caplog.records), caplog.text

    @pytest.mark.asyncio
    async def test_untruncated_walk_reports_truncated_false_and_logs_nothing(self, backend, caplog):
        mock_client = AsyncMock()
        mock_client.scroll = AsyncMock(
            return_value=([_scan_point('id-1', {'data': _SCAN_LEAK_TEXT})], None)
        )

        with (
            patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=mock_client)),
            caplog.at_level(logging.WARNING),
        ):
            result = await backend.scan_payload_text(scope=Scope(project_id='p'), limit=1000)

        assert result['truncated'] is False
        assert not [r for r in caplog.records if 'truncat' in r.message.lower()]

    @pytest.mark.asyncio
    async def test_empty_corpus_is_a_clean_result_not_an_error(self, backend):
        mock_client = AsyncMock()
        mock_client.scroll = AsyncMock(return_value=([], None))

        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=mock_client)):
            result = await backend.scan_payload_text(scope=Scope(project_id='p'))

        assert result == {'matches': [], 'scanned': 0, 'truncated': False}

    @pytest.mark.parametrize('limit', [0, -1])
    @pytest.mark.asyncio
    async def test_a_non_positive_limit_raises_rather_than_walking_nothing(
        self, backend, limit
    ):
        """``limit=0`` would make ``min(page_size, limit - scanned)`` request 0
        points, so the walk returns ``{'matches': [], 'scanned': 0,
        'truncated': False}`` — INDISTINGUISHABLE from a genuinely clean
        corpus, and read by a caller's exit-code predicate as a complete,
        successful sweep. Same no-silent-wrong-value rule that makes a scan
        timeout propagate rather than collapse into an empty list.
        """
        mock_client = AsyncMock()
        mock_client.scroll = AsyncMock(return_value=([], None))

        with (
            patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=mock_client)),
            pytest.raises(ValueError, match='strictly positive'),
        ):
            await backend.scan_payload_text(scope=Scope(project_id='p'), limit=limit)

        mock_client.scroll.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_limit_of_one_still_scans(self, backend):
        """The boundary is rejected BELOW 1, not at it — a 1-record probe is a
        legitimate call and must not be caught by the guard."""
        mock_client = AsyncMock()
        mock_client.scroll = AsyncMock(
            return_value=([_scan_point('id-1', {'data': _SCAN_LEAK_TEXT})], None)
        )

        with patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=mock_client)):
            result = await backend.scan_payload_text(scope=Scope(project_id='p'), limit=1)

        assert result['scanned'] == 1
        assert len(result['matches']) == 1


# ---------------------------------------------------------------------------
# TRIPWIRE: the un-indexed-field MatchText semantics the prefilter relies on.
#
# A read-only live probe (qdrant 1.17.1, 19,321 points) measured that on an
# UN-INDEXED payload field, MatchText performs a LITERAL, case-sensitive,
# ORDER-PRESERVING substring match. Nothing in this repo calls
# create_payload_index today — but if a text index is ever added to `data`,
# MatchText SILENTLY flips to tokenized word-matching. A silent semantic flip
# is precisely the failure class this task exists to kill, so it must fail
# LOUDLY here rather than quietly narrowing the sweep.
#
# Correctness does not depend on this test passing: the mandatory Python
# re-verify keeps the result exact either way, because tokenized matching is
# strictly MORE permissive for these needles. Only speed would degrade.
# ---------------------------------------------------------------------------


@pytest.fixture
def _spt_scope(worker_id) -> Scope:
    return Scope(project_id=f'_test_scan_payload_text_{worker_id}')


@pytest.fixture
def _spt_clean_collection(_spt_scope, mock_config):
    collection = _spt_scope.mem0_collection_name(mock_config.mem0.collection_prefix)
    client = QdrantClient(url=QDRANT_URL, timeout=10)
    with contextlib.suppress(ResponseHandlingException, UnexpectedResponse):
        client.delete_collection(collection)
    yield collection
    with contextlib.suppress(Exception):
        client.delete_collection(collection)
    client.close()


class TestMem0BackendScanPayloadTextIntegration:
    """Pins the un-indexed MatchText substring semantics against live Qdrant."""

    pytestmark = [qdrant_skipif(), pytest.mark.timeout(60), pytest.mark.integration]

    @pytest.mark.asyncio
    async def test_matchtext_prefilter_is_a_literal_substring_match(
        self, mock_config, _spt_scope, _spt_clean_collection, monkeypatch,
    ):
        backend = await _build_asr_backend(mock_config, _spt_scope, monkeypatch)
        try:
            await backend.add_system_record(
                content='The reconciliation stage two judge halts on an empty backlog.',
                scope=_spt_scope,
                metadata={'kind': 'scan_payload_text_probe'},
            )

            # Mid-word substring: proves `contains`, not tokenization.
            midword = await backend.scan_payload_text(
                scope=_spt_scope, needles=['econciliatio'], exhaustive=False
            )
            assert midword['scanned'] == 1, (
                'un-indexed MatchText must substring-match mid-word; a text '
                'payload index on `data` would flip it to tokenized matching'
            )

            # Reversed word order must NOT match a substring engine.
            reversed_order = await backend.scan_payload_text(
                scope=_spt_scope, needles=['judge reconciliation'], exhaustive=False
            )
            assert reversed_order['scanned'] == 0, (
                'reversed word order matched — MatchText is tokenized here, so '
                'the prefilter semantics assumed by scan_payload_text changed'
            )

            # Sanity: exhaustive mode ignores the prefilter entirely, so it
            # still walks the record regardless of MatchText semantics.
            assert (await backend.scan_payload_text(
                scope=_spt_scope, exhaustive=True
            ))['scanned'] == 1
        finally:
            await backend.close()


class TestBuildConfigDictEndpointPlumbing:
    """_build_config_dict is a pure function of config — no Qdrant, no
    AsyncMemory, no _get_instance. Greenfield: nothing tested this dict before.
    """

    # --- default (shipped-config) behaviour: must stay byte-identical ---

    def test_default_omits_embedding_dims_entirely(self, backend):
        """The load-bearing byte-identical assertion.

        mem0/embeddings/openai.py sets
        ``_pass_dimensions_to_api = self.config.embedding_dims is not None``.
        We omit the key today, so ``dimensions`` is NEVER sent on an
        embeddings request. Emitting the key at all — even as 1536 — would
        start sending it on every request under the shipped config. Hence the
        emit-only-when-non-default rule; do not "simplify" it away.
        """
        config_dict = backend._build_config_dict('some_collection')

        assert backend.config.embedder.dimensions == 1536
        assert 'embedding_dims' not in config_dict['embedder']['config']

    def test_default_emits_incumbent_base_urls(self, backend):
        config_dict = backend._build_config_dict('some_collection')

        assert config_dict['llm']['config']['openai_base_url'] == 'https://api.openai.com/v1'
        assert (
            config_dict['embedder']['config']['openai_base_url']
            == 'https://api.openai.com/v1'
        )

    def test_default_vector_store_dims_match_mem0s_own_default(self, backend):
        """Emitting this key is a no-op at the shipped config — mem0's Qdrant
        default is already 1536 — but it is required for a collection to be
        CREATED at a non-default dimensionality."""
        from mem0.configs.vector_stores.qdrant import QdrantConfig

        config_dict = backend._build_config_dict('some_collection')

        assert config_dict['vector_store']['config']['embedding_model_dims'] == 1536
        assert (
            QdrantConfig.model_fields['embedding_model_dims'].default == 1536
        ), 'mem0 changed its Qdrant dims default — the no-op claim above no longer holds'

    def test_preexisting_keys_are_unchanged(self, backend):
        config_dict = backend._build_config_dict('some_collection')

        assert config_dict['version'] == 'v1.1'
        assert config_dict['vector_store']['provider'] == 'qdrant'
        assert config_dict['vector_store']['config']['url'] == backend.config.mem0.qdrant_url
        assert config_dict['vector_store']['config']['collection_name'] == 'some_collection'

        llm = config_dict['llm']['config']
        assert config_dict['llm']['provider'] == 'openai'
        assert llm['model'] == 'gpt-4o-mini'
        assert llm['temperature'] == 0.1
        assert llm['max_tokens'] == 4096
        assert llm['api_key'] == 'test-key'

        embedder = config_dict['embedder']['config']
        assert config_dict['embedder']['provider'] == 'openai'
        assert embedder['model'] == 'text-embedding-3-small'
        assert embedder['api_key'] == 'test-key'

    # --- local-endpoint behaviour ---

    def test_llm_and_embedder_endpoints_are_independent(self, mock_config):
        """The two provider blocks are read SEPARATELY — one is not reused for
        both, which is what makes independent LLM/embedder endpoints possible.
        """
        mock_config.llm.providers.openai.api_url = 'http://127.0.0.1:1234/v1'
        mock_config.embedder.providers.openai.api_url = 'http://127.0.0.1:5678/v1'
        config_dict = Mem0Backend(mock_config)._build_config_dict('c')

        assert config_dict['llm']['config']['openai_base_url'] == 'http://127.0.0.1:1234/v1'
        assert (
            config_dict['embedder']['config']['openai_base_url']
            == 'http://127.0.0.1:5678/v1'
        )

    def test_non_default_dims_emitted_under_both_key_names(self, mock_config):
        """BOTH keys are required for a non-1536 arm, and they are NOT the same
        key: the embedder key controls what dimensionality is requested from
        the API, the vector-store key controls what Qdrant creates the
        collection with. Setting only one silently yields a mismatch.
        """
        mock_config.embedder.dimensions = 768
        config_dict = Mem0Backend(mock_config)._build_config_dict('c')

        assert config_dict['embedder']['config']['embedding_dims'] == 768
        assert config_dict['vector_store']['config']['embedding_model_dims'] == 768

    def test_anthropic_llm_block_carries_no_openai_base_url(self, mock_config):
        """openai_base_url is an OpenAIConfig-only parameter; passing it to the
        anthropic config class would TypeError out of mem0's factory."""
        from fused_memory.config.schema import AnthropicProviderConfig

        mock_config.llm.provider = 'anthropic'
        mock_config.llm.providers.anthropic = AnthropicProviderConfig(api_key='ak')
        config_dict = Mem0Backend(mock_config)._build_config_dict('c')

        assert config_dict['llm']['provider'] == 'anthropic'
        assert 'openai_base_url' not in config_dict['llm']['config']

    # --- key-name guard ---

    def test_every_emitted_key_is_accepted_by_the_installed_mem0(self, mock_config):
        """The highest-value assertion here.

        An unknown key raises TypeError from mem0/utils/factory.py, and
        AsyncMemory.from_config only catches pydantic ValidationError — so a
        misspelling escapes uncaught at instance construction, in production,
        not here. The names are easy to get wrong: the EMBEDDER takes
        `embedding_dims`, while `embedding_model_dims` is the VECTOR STORE's
        key. Validate what we emit against the installed signatures.
        """
        import inspect

        from mem0.configs.embeddings.base import BaseEmbedderConfig
        from mem0.configs.llms.openai import OpenAIConfig
        from mem0.configs.vector_stores.qdrant import QdrantConfig

        # Non-default dims so every optional key is actually emitted.
        mock_config.embedder.dimensions = 768
        mock_config.llm.providers.openai.api_url = 'http://127.0.0.1:1234/v1'
        mock_config.embedder.providers.openai.api_url = 'http://127.0.0.1:5678/v1'
        config_dict = Mem0Backend(mock_config)._build_config_dict('c')

        embedder_params = set(inspect.signature(BaseEmbedderConfig.__init__).parameters)
        llm_params = set(inspect.signature(OpenAIConfig.__init__).parameters)
        vector_store_params = set(QdrantConfig.model_fields)

        for key in config_dict['embedder']['config']:
            assert key in embedder_params, (
                f'embedder key {key!r} is not a BaseEmbedderConfig parameter — '
                f'mem0 would TypeError. Did you mean one of {sorted(embedder_params)}?'
            )
        for key in config_dict['llm']['config']:
            assert key in llm_params, f'llm key {key!r} is not an OpenAIConfig parameter'
        for key in config_dict['vector_store']['config']:
            assert key in vector_store_params, (
                f'vector_store key {key!r} is not a QdrantConfig field'
            )

        # And pin the two easily-confused spellings explicitly.
        assert 'embedding_dims' in embedder_params
        assert 'embedding_model_dims' not in embedder_params
        assert 'embedding_model_dims' in vector_store_params


class TestNonDefaultDimsAgainstAnExistingCollection:
    """The migration hazard behind plumbing embedder.dimensions at all.

    Neither dims key was plumbed before, so a deployment already configured
    with ``embedder.dimensions != 1536`` was running INERT: mem0 created its
    Qdrant collection at its own 1536 default and requested 1536-wide vectors.
    Both now follow the config — but only for a collection that does not exist
    yet. Changing the knob on a live deployment therefore requires recreating
    and re-embedding the collection; it is not live-migratable.
    """

    @staticmethod
    def _fake_qdrant_client(existing: list[str]):
        collections = MagicMock()
        collections.collections = [MagicMock(name=n) for n in existing]
        # MagicMock(name=...) sets the mock's repr name, not a .name attribute.
        for mock_col, real_name in zip(collections.collections, existing, strict=True):
            mock_col.name = real_name

        client = MagicMock()
        client.get_collections.return_value = collections
        return client

    def test_existing_collection_is_left_at_its_old_dimensionality(self):
        """Upstream short-circuits, so the new dims never reach Qdrant.

        Measured against the installed mem0: ``Qdrant.create_col`` returns
        early when the collection is already listed ('Collection ... already
        exists. Skipping creation.'). The embedder meanwhile DOES start
        requesting N-wide vectors, so every subsequent upsert fails on a
        dimension mismatch at runtime. Pinning it here means a future mem0 that
        learns to migrate — or to fail loudly — turns this red instead of
        leaving a stale operator note in place.
        """
        from mem0.vector_stores.qdrant import Qdrant

        client = self._fake_qdrant_client(['fused_dark_factory'])

        Qdrant(
            collection_name='fused_dark_factory',
            embedding_model_dims=768,
            client=client,
        )

        client.create_collection.assert_not_called()

    def test_a_fresh_collection_is_created_at_the_configured_dimensionality(self):
        """Control: the plumbing does work — on a collection that does not yet
        exist, which is the only case where changing dimensions is safe."""
        from mem0.vector_stores.qdrant import Qdrant

        client = self._fake_qdrant_client([])

        Qdrant(collection_name='brand_new', embedding_model_dims=768, client=client)

        client.create_collection.assert_called_once()
        vectors_config = client.create_collection.call_args.kwargs['vectors_config']
        assert vectors_config.size == 768


class TestAmbientBaseUrlPrecedenceIsReported:
    """Config is now authoritative over OPENAI_BASE_URL / OPENAI_API_BASE.

    That is intended — a written-down endpoint should beat an inherited env
    var — but for a site that routed OpenAI traffic through a gateway by
    exporting OPENAI_BASE_URL without OPENAI_API_URL, it silently sends that
    traffic back to api.openai.com. An egress and billing change must be
    announced (docs/legibility/design-invariants.md, no-silent-fail-soft).
    """

    #: Every ambient var ``warn_if_ambient_base_url_is_overridden`` inspects.
    #: Both are live in the wild (mem0 still honours the deprecated
    #: ``OPENAI_API_BASE`` spelling), so a test that sets only one is at the
    #: mercy of whatever the host/CI runner exports for the other.
    AMBIENT_VARS = ('OPENAI_BASE_URL', 'OPENAI_API_BASE')

    @pytest.fixture(autouse=True)
    def _clear_warn_cache(self, monkeypatch):
        """Reset the once-per-process warn cache AND the ambient environment.

        Clearing both vars here (rather than per-test) means every test in
        this class starts from a known-empty ambient environment and then
        sets only the var it means to exercise — otherwise an inherited
        ``OPENAI_API_BASE`` on the runner adds a second pair of warnings and
        flips the count assertions red.
        """
        from fused_memory.config.env_precedence import reset_warning_cache

        for var in self.AMBIENT_VARS:
            monkeypatch.delenv(var, raising=False)
        reset_warning_cache()
        yield
        reset_warning_cache()

    @pytest.mark.parametrize('var', ['OPENAI_BASE_URL', 'OPENAI_API_BASE'])
    def test_disagreeing_env_var_is_reported(self, mock_config, monkeypatch, caplog, var):
        monkeypatch.setenv(var, 'https://gateway.example.invalid/v1')

        with caplog.at_level(logging.WARNING):
            Mem0Backend(mock_config)._build_config_dict('c')

        messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(
            var in m and 'https://api.openai.com/v1' in m for m in messages
        ), f'expected a warning naming {var} and the configured endpoint, got {messages}'

    def test_agreeing_env_var_is_silent(self, mock_config, monkeypatch, caplog):
        """No warning when the environment and the config already agree —
        otherwise the signal is noise on a correctly-configured deployment."""
        monkeypatch.setenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')

        with caplog.at_level(logging.WARNING):
            Mem0Backend(mock_config)._build_config_dict('c')

        assert not [r for r in caplog.records if 'OPENAI_BASE_URL' in r.message]

    def test_unset_env_is_silent(self, mock_config, caplog):
        # Both ambient vars are already cleared by the autouse fixture.
        with caplog.at_level(logging.WARNING):
            Mem0Backend(mock_config)._build_config_dict('c')

        assert not [r for r in caplog.records if 'takes precedence' in r.message]

    def test_repeated_builds_warn_only_once(self, mock_config, monkeypatch, caplog):
        """_build_config_dict runs once per project instance; an unguarded
        warning would repeat per project and read as a fresh incident."""
        monkeypatch.setenv('OPENAI_BASE_URL', 'https://gateway.example.invalid/v1')
        backend = Mem0Backend(mock_config)

        with caplog.at_level(logging.WARNING):
            backend._build_config_dict('c1')
            backend._build_config_dict('c2')

        hits = [r for r in caplog.records if 'takes precedence' in r.message]
        # One for the llm block, one for the embedder block — not four.
        assert len(hits) == 2, [r.message for r in hits]
