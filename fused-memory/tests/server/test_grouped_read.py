"""Unit tests for server-side grouped reads (task 3129, leaf δ).

step-1 (RED) / step-2 (GREEN):
  ``build_grouped_document`` turns a canonical id into ONE grouped block —
  bounded amendment digests plus an EXACT sighting count — over the three
  landed Mem0 read primitives (``count_memories_by_metadata`` /
  ``get_memories_by_metadata`` / ``get_memory_by_id``).

This file also pins the CHILD-RECORD SCHEMA as a contract: an amendment or
sighting is an ordinary Mem0 entry with its own UUID whose ``metadata``
carries ``parent_id`` (canonical full-UUID) plus ``kind`` in
``{'amendment', 'sighting'}`` — both already members of the landed
``memory_metadata.KIND_REGISTRY`` (task 3195).  The representation is
strictly ADD-ONLY: no code path in this task edits a canonical's text.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from fused_memory.backends.mem0_client import MEM0_MANAGED_METADATA_KEYS
from fused_memory.memory_metadata import (
    EXPERIMENTAL_KEY_PREFIX,
    KIND_REGISTRY,
    validate_memory_metadata,
)
from fused_memory.models.enums import SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.server.grouped_read import (
    _AMENDMENT_DIGEST_CAP,
    _DIGEST_CHARS,
    _DIGEST_ELLIPSIS,
    _MAX_CONCURRENT_GROUP_READS,
    AMENDMENT_KIND,
    CHILD_KINDS,
    CHILDREN_UNAVAILABLE_KEY,
    CONTESTED_METADATA_KEY,
    PARENT_UNAVAILABLE_KEY,
    PARENT_UNRESOLVED_KEY,
    SIGHTING_KIND,
    _suppress_child,
    build_grouped_document,
    group_memory_document,
    group_search_results,
    is_contested_child,
)

_PROJECT_ID = 'dark_factory'

# Canonical full-UUIDs (36-char dashed) — the shape memory_metadata's
# ``invalid_parent_id_shape`` check requires of every parent_id.
_CANONICAL_ID = '11111111-1111-4111-8111-111111111111'
_AMEND_1 = '22222222-2222-4222-8222-222222222221'
_AMEND_2 = '22222222-2222-4222-8222-222222222222'
_SIGHT_1 = '33333333-3333-4333-8333-333333333331'
_SIGHT_2 = '33333333-3333-4333-8333-333333333332'
_SIGHT_3 = '33333333-3333-4333-8333-333333333333'

# Sentinel key for the "no kind filter" (total children) count in _stub_service.
_TOTAL = '__total__'


def _child(
    memory_id: str,
    parent_id: str,
    kind: str,
    text: str,
    created_at: str | None = None,
    **extra_meta,
) -> dict:
    """A scrolled child row, shaped exactly as ``get_memories_by_metadata`` returns.

    ``metadata`` is the FULL raw Qdrant payload — so the body lives under the
    mem0 content key ``data`` (the first of ``_MEM0_CONTENT_KEYS``), not under
    a separate 'content' field.
    """
    return {
        'id': memory_id,
        'created_at': created_at,
        'metadata': {
            'data': text,
            'parent_id': parent_id,
            'kind': kind,
            **extra_meta,
        },
    }


def _stub_service(
    children: list[dict],
    *,
    counts: dict | None = None,
    parents: dict | None = None,
    scroll_error: Exception | None = None,
    count_errors: dict | None = None,
) -> AsyncMock:
    """AsyncMock service dispatching the three Mem0 read primitives over *children*.

    *counts* overrides the derived counts per ``kind`` filter value (or
    ``_TOTAL`` for the un-kinded total) so a test can model a scroll bounded
    below the exact count.  *count_errors* raises per count filter, keyed the
    same way, so a test can fault ONE of the counts.  *parents* maps memory id
    → raw record for ``get_memory_by_id``.
    """
    service = AsyncMock()

    def _matches(child: dict, filters: dict) -> bool:
        meta = child['metadata']
        return all(meta.get(key) == value for key, value in filters.items())

    def _count(*, project_id: str, filters: dict):
        key = filters.get('kind', _TOTAL)
        if count_errors is not None and key in count_errors:
            raise count_errors[key]
        if counts is not None and key in counts:
            return counts[key]
        return len([c for c in children if _matches(c, filters)])

    def _scroll(*, project_id: str, filters: dict, limit: int = 1000):
        matched = [c for c in children if _matches(c, filters)]
        return matched[:limit]

    def _get_by_id(*, project_id: str, memory_id: str):
        return (parents or {}).get(memory_id)

    service.count_memories_by_metadata = AsyncMock(side_effect=_count)
    service.get_memories_by_metadata = AsyncMock(side_effect=scroll_error or _scroll)
    service.get_memory_by_id = AsyncMock(side_effect=_get_by_id)
    return service


def _two_amendments_three_sightings() -> list[dict]:
    return [
        _child(_AMEND_1, _CANONICAL_ID, AMENDMENT_KIND, 'first correction', '2026-08-01T00:00:00+00:00'),
        _child(_AMEND_2, _CANONICAL_ID, AMENDMENT_KIND, 'second correction', '2026-08-02T00:00:00+00:00'),
        _child(_SIGHT_1, _CANONICAL_ID, SIGHTING_KIND, 'seen again', '2026-08-03T00:00:00+00:00'),
        _child(_SIGHT_2, _CANONICAL_ID, SIGHTING_KIND, 'seen again', '2026-08-04T00:00:00+00:00'),
        _child(_SIGHT_3, _CANONICAL_ID, SIGHTING_KIND, 'seen again', '2026-08-05T00:00:00+00:00'),
    ]


class TestBuildGroupedDocument:
    """The leaf's headline signal: one canonical → one grouped block."""

    @pytest.mark.asyncio
    async def test_two_amendments_and_three_sightings_collapse_to_one_block(self):
        """2 amendment + 3 sighting children → 2 digests and sighting_count == 3."""
        service = _stub_service(_two_amendments_three_sightings())

        block = await build_grouped_document(service, _PROJECT_ID, _CANONICAL_ID)

        assert block is not None, (
            'A canonical WITH children must produce a grouped block, got None. '
            'RED: build_grouped_document does not exist / returns nothing yet.'
        )
        assert len(block['amendments']) == 2, (
            f"Expected exactly 2 amendment digests, got {block.get('amendments')!r}"
        )
        assert block['sighting_count'] == 3, (
            f"Expected sighting_count == 3 (the EXACT count API), got {block!r}"
        )
        # Sightings are counted, never listed — a sighting body carries no
        # information a count does not, and listing them would reintroduce the
        # unbounded fan-out the digest cap exists to bound.
        digest_ids = {d['id'] for d in block['amendments']}
        assert digest_ids == {_AMEND_1, _AMEND_2}, (
            f'Digest list must contain the AMENDMENT children only, got {digest_ids!r}'
        )

    @pytest.mark.asyncio
    async def test_digest_entry_shape(self):
        """Each digest entry carries id / digest / created_at / kind."""
        service = _stub_service(_two_amendments_three_sightings())

        block = await build_grouped_document(service, _PROJECT_ID, _CANONICAL_ID)

        assert block is not None
        for entry in block['amendments']:
            assert set(entry) >= {'id', 'digest', 'created_at', 'kind'}, (
                f'Digest entry must carry id/digest/created_at/kind, got {entry!r}'
            )
            assert entry['kind'] == AMENDMENT_KIND, (
                f"Digest entry kind must be {AMENDMENT_KIND!r}, got {entry!r}"
            )
        first = block['amendments'][0]
        assert first['id'] == _AMEND_1
        assert first['digest'] == 'first correction', (
            f"A short body must pass through VERBATIM, got {first['digest']!r}"
        )
        assert first['created_at'] == '2026-08-01T00:00:00+00:00'

    @pytest.mark.asyncio
    async def test_long_body_truncated_to_named_cap_with_ellipsis(self):
        """A long amendment body is truncated to _DIGEST_CHARS + an ellipsis marker."""
        long_body = 'x' * (_DIGEST_CHARS + 500)
        short_body = 'y' * (_DIGEST_CHARS - 1)
        service = _stub_service([
            _child(_AMEND_1, _CANONICAL_ID, AMENDMENT_KIND, long_body, '2026-08-01T00:00:00+00:00'),
            _child(_AMEND_2, _CANONICAL_ID, AMENDMENT_KIND, short_body, '2026-08-02T00:00:00+00:00'),
        ])

        block = await build_grouped_document(service, _PROJECT_ID, _CANONICAL_ID)

        assert block is not None
        long_digest, short_digest = block['amendments'][0], block['amendments'][1]
        assert long_digest['digest'] == long_body[:_DIGEST_CHARS] + _DIGEST_ELLIPSIS, (
            'A body longer than _DIGEST_CHARS must be cut at the named cap and '
            f'marked with {_DIGEST_ELLIPSIS!r}, got len={len(long_digest["digest"])}'
        )
        assert short_digest['digest'] == short_body, (
            'A body at or under _DIGEST_CHARS must pass through verbatim with NO '
            f'ellipsis, got {short_digest["digest"][-5:]!r}'
        )
        assert not short_digest['digest'].endswith(_DIGEST_ELLIPSIS)

    @pytest.mark.asyncio
    async def test_digest_ordering_is_deterministic(self):
        """Digests sort by (created_at, id) so two runs over one corpus agree."""
        older = '2026-08-01T00:00:00+00:00'
        newer = '2026-08-09T00:00:00+00:00'
        # Deliberately scrolled out of order, and with a created_at TIE broken by id.
        service = _stub_service([
            _child(_AMEND_2, _CANONICAL_ID, AMENDMENT_KIND, 'b', newer),
            _child(_AMEND_1, _CANONICAL_ID, AMENDMENT_KIND, 'a', newer),
            _child('00000000-0000-4000-8000-000000000001', _CANONICAL_ID, AMENDMENT_KIND, 'c', older),
        ])

        first = await build_grouped_document(service, _PROJECT_ID, _CANONICAL_ID)
        second = await build_grouped_document(service, _PROJECT_ID, _CANONICAL_ID)

        assert first is not None and second is not None
        order = [d['id'] for d in first['amendments']]
        assert order == [
            '00000000-0000-4000-8000-000000000001',  # oldest created_at first
            _AMEND_1,  # created_at tie → lower id first
            _AMEND_2,
        ], f'Digests must sort by (created_at, id), got {order!r}'
        assert order == [d['id'] for d in second['amendments']], (
            'Two runs over the same corpus must produce the same digest order'
        )

    @pytest.mark.asyncio
    async def test_missing_created_at_does_not_crash_ordering(self):
        """A child with created_at None still sorts deterministically (empty-string key)."""
        service = _stub_service([
            _child(_AMEND_2, _CANONICAL_ID, AMENDMENT_KIND, 'b', '2026-08-02T00:00:00+00:00'),
            _child(_AMEND_1, _CANONICAL_ID, AMENDMENT_KIND, 'a', None),
        ])

        block = await build_grouped_document(service, _PROJECT_ID, _CANONICAL_ID)

        assert block is not None
        assert [d['id'] for d in block['amendments']] == [_AMEND_1, _AMEND_2], (
            'A None created_at must sort as the empty string, not raise a '
            'TypeError comparing None to str'
        )


def _result(
    memory_id: str,
    content: str,
    score: float,
    *,
    metadata: dict | None = None,
    created_at: str | None = None,
    source_store: SourceStore = SourceStore.mem0,
) -> MemoryResult:
    """A search hit, exactly as ``MemoryService.search`` returns them."""
    return MemoryResult(
        id=memory_id,
        content=content,
        source_store=source_store,
        relevance_score=score,
        metadata=metadata or {},
        created_at=created_at,
    )


def _child_result(memory_id: str, score: float, kind: str = AMENDMENT_KIND, **extra_meta) -> MemoryResult:
    return _result(
        memory_id,
        'a correction',
        score,
        metadata={'parent_id': _CANONICAL_ID, 'kind': kind, **extra_meta},
    )


def _parent_record(memory_id: str = _CANONICAL_ID, text: str = 'the canonical claim') -> dict:
    """A raw ``get_memory_by_id`` record for the parent."""
    return {
        'id': memory_id,
        'content': text,
        'metadata': {'data': text, 'kind': 'canonical', 'topic': 'merge-lane'},
    }


class TestGroupSearchResults:
    """List-level collapse: children fold into their parent's grouped hit."""

    @pytest.mark.asyncio
    async def test_child_is_dropped_when_its_parent_is_also_a_hit(self):
        """esc-5541: an untagged child must not outrank the canonical it amends."""
        service = _stub_service(_two_amendments_three_sightings())
        hits = [
            _result(_CANONICAL_ID, 'the canonical claim', 0.9, metadata={'kind': 'canonical'}),
            _child_result(_AMEND_1, 0.8),
        ]

        grouped = await group_search_results(service, _PROJECT_ID, hits)

        assert len(grouped) == 1, (
            f'The child must fold into its parent, leaving ONE hit, got {grouped!r}. '
            'RED: group_search_results does not exist yet.'
        )
        assert grouped[0]['id'] == _CANONICAL_ID
        assert grouped[0]['grouped']['sighting_count'] == 3
        assert len(grouped[0]['grouped']['amendments']) == 2

    @pytest.mark.asyncio
    async def test_child_only_match_resolves_upward_to_the_parent(self):
        """D6: a child's content is never unreachable — the group takes its rank."""
        service = _stub_service(
            _two_amendments_three_sightings(),
            parents={_CANONICAL_ID: _parent_record()},
        )
        hits = [_child_result(_AMEND_1, 0.77)]

        grouped = await group_search_results(service, _PROJECT_ID, hits)

        assert len(grouped) == 1, f'Expected the parent to take the hit slot, got {grouped!r}'
        promoted = grouped[0]
        assert promoted['id'] == _CANONICAL_ID, (
            f'A child-only match must resolve UPWARD to its parent, got {promoted!r}'
        )
        assert promoted['content'] == 'the canonical claim', (
            'The promoted hit must carry the PARENT\'s content, not the child\'s'
        )
        assert promoted['grouped']['sighting_count'] == 3
        assert promoted['relevance_score'] == pytest.approx(0.77), (
            'The promoted parent occupies the CHILD\'s rank position and score'
        )
        service.get_memory_by_id.assert_awaited_once_with(
            project_id=_PROJECT_ID, memory_id=_CANONICAL_ID
        )

    @pytest.mark.asyncio
    async def test_parent_reached_twice_is_emitted_once_with_the_higher_score(self):
        """Direct hit + upward resolution must not double-list the same parent."""
        service = _stub_service(
            _two_amendments_three_sightings(),
            parents={_CANONICAL_ID: _parent_record()},
        )
        hits = [
            _child_result(_AMEND_1, 0.95),
            _result(_CANONICAL_ID, 'the canonical claim', 0.4, metadata={'kind': 'canonical'}),
        ]

        grouped = await group_search_results(service, _PROJECT_ID, hits)

        assert len(grouped) == 1, f'The parent must be emitted EXACTLY once, got {grouped!r}'
        assert grouped[0]['id'] == _CANONICAL_ID
        assert grouped[0]['relevance_score'] == pytest.approx(0.95), (
            'Dedup must keep the HIGHER relevance_score of the two routes, got '
            f"{grouped[0]['relevance_score']}"
        )

    @pytest.mark.asyncio
    async def test_surviving_order_is_preserved_and_the_list_never_grows(self):
        """Relative rank order survives; grouping only ever shortens the list."""
        service = _stub_service(_two_amendments_three_sightings())
        hits = [
            _result('55555555-5555-4555-8555-555555555551', 'unrelated a', 0.99),
            _child_result(_AMEND_1, 0.85),
            _result(_CANONICAL_ID, 'the canonical claim', 0.8, metadata={'kind': 'canonical'}),
            _result('55555555-5555-4555-8555-555555555552', 'unrelated b', 0.7),
        ]

        grouped = await group_search_results(service, _PROJECT_ID, hits)

        assert len(grouped) <= len(hits), 'Grouping must never GROW the result list'
        assert [r['id'] for r in grouped] == [
            '55555555-5555-4555-8555-555555555551',
            _CANONICAL_ID,
            '55555555-5555-4555-8555-555555555552',
        ], (
            'Survivors must keep their relative order, with the group occupying the '
            f'best rank any of its members held, got {[r["id"] for r in grouped]!r}'
        )

    @pytest.mark.asyncio
    async def test_grouping_is_keyed_strictly_on_parent_id_not_topic(self):
        """D5: a shared `topic` is NOT a grouping key — only parent_id is."""
        service = _stub_service([])
        peer_a = '66666666-6666-4666-8666-666666666661'
        peer_b = '66666666-6666-4666-8666-666666666662'
        hits = [
            _result(peer_a, 'first note', 0.9, metadata={'topic': 'merge-lane'}),
            _result(peer_b, 'second note', 0.8, metadata={'topic': 'merge-lane'}),
        ]

        grouped = await group_search_results(service, _PROJECT_ID, hits)

        assert [r['id'] for r in grouped] == [peer_a, peer_b], (
            'Two entries sharing only a topic are independent peers and must both '
            f'survive as top-level hits, got {grouped!r}'
        )
        assert service.get_memory_by_id.await_count == 0, (
            'A topic is not a parent pointer — no upward resolution may be attempted'
        )

    @pytest.mark.asyncio
    async def test_result_dicts_keep_the_model_dump_shape(self):
        """Grouped output is model_dump()-shaped, so the wire contract is unchanged."""
        service = _stub_service([])
        hit = _result('77777777-7777-4777-8777-777777777771', 'plain', 0.5)

        grouped = await group_search_results(service, _PROJECT_ID, [hit])

        assert grouped[0] == hit.model_dump(), (
            'A childless, non-child result must dump EXACTLY as today — no grouped '
            f'key, no reshaping, got {grouped[0]!r}'
        )

    @pytest.mark.asyncio
    async def test_graphiti_results_cost_no_child_probe(self):
        """Only Mem0 points can be parents, so a graph edge is never probed.

        ``parent_id`` liveness resolves through ``Mem0Backend.get_point_by_id``
        (memory_service's write seam, task 3197), so a Graphiti edge uuid can
        never be a live parent — probing one would be a guaranteed-zero Qdrant
        count per graph hit on every search.
        """
        service = _stub_service([])
        hits = [
            _result(
                '88888888-8888-4888-8888-888888888881',
                'an edge fact',
                0.9,
                source_store=SourceStore.graphiti,
            )
        ]

        grouped = await group_search_results(service, _PROJECT_ID, hits)

        assert len(grouped) == 1
        assert 'grouped' not in grouped[0]
        assert service.count_memories_by_metadata.await_count == 0, (
            'A Graphiti-sourced hit must issue no child count at all, got '
            f'{service.count_memories_by_metadata.await_args_list!r}'
        )


class TestNeverSuppressCarveOuts:
    """Three cases where a child must survive as a top-level hit regardless."""

    @pytest.mark.asyncio
    async def test_contested_child_stays_top_level_beside_its_parent(self):
        """esc-5712: a correction must not be demoted to a digest under what it contests."""
        service = _stub_service([
            _child(
                _AMEND_1,
                _CANONICAL_ID,
                AMENDMENT_KIND,
                'this claim is WRONG',
                '2026-08-01T00:00:00+00:00',
                **{CONTESTED_METADATA_KEY: True},
            ),
            _child(_AMEND_2, _CANONICAL_ID, AMENDMENT_KIND, 'a routine addendum', '2026-08-02T00:00:00+00:00'),
        ])
        hits = [
            _result(_CANONICAL_ID, 'the canonical claim', 0.9, metadata={'kind': 'canonical'}),
            _child_result(_AMEND_1, 0.8, AMENDMENT_KIND, **{CONTESTED_METADATA_KEY: True}),
        ]

        grouped = await group_search_results(service, _PROJECT_ID, hits)

        assert [r['id'] for r in grouped] == [_CANONICAL_ID, _AMEND_1], (
            'A CONTESTED child must survive as a top-level hit in its own right '
            f'even when its parent is present, got {[r["id"] for r in grouped]!r}. '
            'RED: the contested carve-out does not exist yet.'
        )
        assert grouped[1]['content'] == 'a correction'

    @pytest.mark.asyncio
    async def test_contested_child_is_also_marked_in_the_parents_digests(self):
        """Surfacing it in BOTH places is strictly more informative than either."""
        service = _stub_service([
            _child(
                _AMEND_1,
                _CANONICAL_ID,
                AMENDMENT_KIND,
                'this claim is WRONG',
                '2026-08-01T00:00:00+00:00',
                **{CONTESTED_METADATA_KEY: True},
            ),
            _child(_AMEND_2, _CANONICAL_ID, AMENDMENT_KIND, 'a routine addendum', '2026-08-02T00:00:00+00:00'),
        ])

        block = await build_grouped_document(service, _PROJECT_ID, _CANONICAL_ID)

        assert block is not None
        by_id = {d['id']: d for d in block['amendments']}
        assert by_id[_AMEND_1].get('contested') is True, (
            f'The contested amendment must be marked in the digest list, got {by_id[_AMEND_1]!r}'
        )
        assert 'contested' not in by_id[_AMEND_2], (
            "An ordinary amendment's digest must OMIT the contested key entirely, "
            f'got {by_id[_AMEND_2]!r}'
        )

    @pytest.mark.asyncio
    async def test_child_with_an_unresolvable_parent_survives_and_is_marked(self):
        """A dangling parent pointer must never make a child's content vanish."""
        # parents={} → get_memory_by_id returns None: a dead/dangling pointer.
        service = _stub_service([], parents={})
        hits = [_child_result(_AMEND_1, 0.8)]

        grouped = await group_search_results(service, _PROJECT_ID, hits)

        assert len(grouped) == 1, (
            f'A child whose parent does not resolve must NOT be dropped, got {grouped!r}'
        )
        assert grouped[0]['id'] == _AMEND_1
        assert grouped[0]['content'] == 'a correction', (
            'The surviving child keeps its OWN content'
        )
        assert grouped[0].get('parent_unresolved') is True, (
            'The survivor must carry an explicit dangling-pointer marker rather '
            f'than silently looking like an ordinary hit, got {grouped[0]!r}'
        )

    @pytest.mark.asyncio
    async def test_resolvable_parent_does_not_stamp_parent_unresolved(self):
        """Fault-only loudness: the marker appears only on the fault path."""
        service = _stub_service(
            _two_amendments_three_sightings(),
            parents={_CANONICAL_ID: _parent_record()},
        )

        grouped = await group_search_results(service, _PROJECT_ID, [_child_result(_AMEND_1, 0.8)])

        assert 'parent_unresolved' not in grouped[0], (
            f'A resolvable parent must produce no marker, got {grouped[0]!r}'
        )

    @pytest.mark.asyncio
    async def test_failed_grouping_suppresses_nothing(self):
        """We never learned what that parent's children are — so we drop none of them."""
        service = _stub_service(
            _two_amendments_three_sightings(),
            count_errors={_TOTAL: TimeoutError('qdrant count timed out')},
        )
        hits = [
            _result(_CANONICAL_ID, 'the canonical claim', 0.9, metadata={'kind': 'canonical'}),
            _child_result(_AMEND_1, 0.8),
        ]

        grouped = await group_search_results(service, _PROJECT_ID, hits)

        assert [r['id'] for r in grouped] == [_CANONICAL_ID, _AMEND_1], (
            'A parent whose grouping FAILED must suppress nothing — suppressing on '
            'the strength of a grouping we never actually read would hide a child '
            f'behind a backend timeout, got {[r["id"] for r in grouped]!r}'
        )
        assert grouped[0]['grouped'] == {
            'children_unavailable': True,
            'error_type': 'TimeoutError',
        }, f'The parent must still carry the loud fault marker, got {grouped[0]!r}'

    # ---- fourth carve-out: NOT REPRESENTED -------------------------------
    #
    # ``_suppress_child`` claimed suppression happened "only when the child is
    # genuinely REPRESENTED by its parent's grouped document" but never checked
    # it.  These pin the claim as an ENFORCED precondition, behind the pinning
    # that makes it true on the normal path: a future edit to either mechanism
    # then degrades to keeping a child as a VISIBLE DUPLICATE rather than to
    # silently losing it as an invisible retrieval regression.

    @staticmethod
    def _resolvable_parent() -> MemoryResult:
        return _result(_CANONICAL_ID, 'the canonical claim', 0.9, metadata={'kind': 'canonical'})

    @staticmethod
    def _block(**overrides) -> dict:
        block = {'amendments': [], 'amendment_count': 0, 'sighting_count': 0}
        block.update(overrides)
        return block

    def test_child_listed_as_a_digest_is_represented(self):
        """The in-cap amendment case: present in `amendments` → suppressible."""
        block = self._block(
            amendments=[{'id': _AMEND_1, 'digest': 'a correction'}], amendment_count=1
        )

        assert _suppress_child(_AMEND_1, {}, self._resolvable_parent(), block) is True

    def test_child_absent_from_the_block_is_never_suppressed(self):
        """The truncated-list case: a cap-sized sample listing only OTHER ids."""
        block = self._block(
            amendments=[{'id': _AMEND_1, 'digest': 'a correction'}],
            amendment_count=50,
            truncated=True,
        )

        assert _suppress_child(_AMEND_2, {}, self._resolvable_parent(), block) is False, (
            'A child the block does not list must NOT be suppressed — dropping it '
            'would erase its text from the response entirely. '
            'RED: _suppress_child does not check representation yet.'
        )

    def test_a_pinned_child_counts_as_represented(self):
        """`matched_children` is representation too — that is what pinning buys."""
        block = self._block(
            matched_children=[{'id': _SIGHT_1, 'content': 'seen again', 'matched': True}],
            sighting_count=4,
        )

        assert _suppress_child(_SIGHT_1, {}, self._resolvable_parent(), block) is True
        assert _suppress_child(_SIGHT_2, {}, self._resolvable_parent(), block) is False, (
            'A matched_children list naming only OTHER ids represents nothing '
            'about this child'
        )

    def test_block_carrying_neither_key_represents_nothing(self):
        """A block with no `amendments` and no `matched_children` suppresses nothing."""
        block = {'amendment_count': 0, 'sighting_count': 3}

        assert _suppress_child(_SIGHT_1, {}, self._resolvable_parent(), block) is False

    def test_unusable_child_id_is_never_suppressed(self):
        """Neither pinnable nor verifiable → keep it, visibly."""
        block = self._block(
            amendments=[{'id': _AMEND_1, 'digest': 'a correction'}], amendment_count=1
        )

        for bad_id in ('', None, 123, {'id': _AMEND_1}):
            assert _suppress_child(bad_id, {}, self._resolvable_parent(), block) is False, (
                f'A child whose id is {bad_id!r} cannot be pinned into the block nor '
                'checked against it, so it must survive rather than vanish'
            )

    def test_the_three_existing_carve_outs_still_short_circuit_first(self):
        """Cheapest and most specific first — their behaviour is unchanged."""
        represented = self._block(
            amendments=[{'id': _AMEND_1, 'digest': 'a correction'}], amendment_count=1
        )

        assert _suppress_child(_AMEND_1, {}, None, represented) is False, (
            'UNRESOLVABLE PARENT still wins even over a block that represents the child'
        )
        assert (
            _suppress_child(
                _AMEND_1,
                {CONTESTED_METADATA_KEY: True},
                self._resolvable_parent(),
                represented,
            )
            is False
        ), 'CONTESTED still wins even over a block that represents the child'
        assert (
            _suppress_child(
                _AMEND_1,
                {},
                self._resolvable_parent(),
                {CHILDREN_UNAVAILABLE_KEY: True, 'error_type': 'TimeoutError'},
            )
            is False
        ), 'A failed grouping still suppresses nothing'

    @pytest.mark.asyncio
    async def test_unpinnable_child_survives_as_a_top_level_hit(self):
        """End-to-end: the precondition keeps it visible instead of losing it."""
        service = _stub_service(_two_amendments_three_sightings())
        # An empty id defeats pinning (nothing to key the entry on) AND
        # verification — the child must therefore stay a hit in its own right.
        unpinnable = _result(
            '',
            'an unkeyed correction',
            0.8,
            metadata={'parent_id': _CANONICAL_ID, 'kind': AMENDMENT_KIND},
        )
        hits = [
            _result(_CANONICAL_ID, 'the canonical claim', 0.9, metadata={'kind': 'canonical'}),
            unpinnable,
        ]

        grouped = await group_search_results(service, _PROJECT_ID, hits)

        assert len(grouped) == 2, (
            'A child that can be neither pinned nor verified must survive as a '
            f'TOP-LEVEL hit rather than vanishing, got {grouped!r}'
        )
        assert grouped[1]['content'] == 'an unkeyed correction'
        block = grouped[0]['grouped']
        assert not block.get('matched_children'), (
            f'An unkeyed child cannot be pinned, so nothing may be invented, got {block!r}'
        )


#: A matched child body deliberately LONGER than ``_DIGEST_CHARS``, so "the
#: pinned entry carries the FULL text" cannot pass by accident on a short body.
_MATCHED_TEXT = 'THE MATCHED TEXT — ' + ('the correction body says otherwise. ' * 20)


class TestSuppressionRequiresRepresentation:
    """A child is suppressed ONLY when its parent's block demonstrably represents it.

    Two live paths break the "represented" claim, and in BOTH the matched text
    and the matched id vanished from the response entirely — the
    "retrieval-regression branch of Option B" the PRD closes by decision:

    (a) TRUNCATED AMENDMENT LIST — the backend caps the scroll BEFORE this
        module sorts, so which cap-sized sample comes back is arbitrary and a
        matched amendment beyond the cap simply is not in ``amendments``;
    (b) SIGHTING-ONLY MATCH — ``build_grouped_document`` scrolls
        ``kind='amendment'`` only, so a matched sighting can never be listed.

    The fix keeps BOTH goals: the canonical still absorbs its children
    (esc-5541 stays closed) AND the matched text stays reachable (D6), by
    pinning the swallowed child — full body, from the hit already in hand, at
    zero extra backend reads — into a ``matched_children`` list on the block.
    """

    _FANOUT = 50
    _MATCHED_INDEX = 30  # beyond _AMENDMENT_DIGEST_CAP: excluded from the scroll

    @classmethod
    def _fifty_amendments(cls) -> list[dict]:
        return [
            _child(
                f'44444444-4444-4444-8444-{index:012d}',
                _CANONICAL_ID,
                AMENDMENT_KIND,
                _MATCHED_TEXT if index == cls._MATCHED_INDEX else f'correction {index}',
                f'2026-08-01T00:00:{index:02d}+00:00',
            )
            for index in range(cls._FANOUT)
        ]

    @classmethod
    def _matched_amendment_id(cls) -> str:
        return f'44444444-4444-4444-8444-{cls._MATCHED_INDEX:012d}'

    @staticmethod
    def _canonical_hit(score: float = 0.9) -> MemoryResult:
        return _result(_CANONICAL_ID, 'the canonical claim', score, metadata={'kind': 'canonical'})

    @staticmethod
    def _matched_hit(memory_id: str, kind: str, score: float, created_at: str) -> MemoryResult:
        return _result(
            memory_id,
            _MATCHED_TEXT,
            score,
            metadata={'parent_id': _CANONICAL_ID, 'kind': kind},
            created_at=created_at,
        )

    @staticmethod
    def _represented_ids(block: dict) -> set:
        """Every child id the parent's block demonstrably accounts for."""
        return {d.get('id') for d in block.get('amendments', [])} | {
            e.get('id') for e in block.get('matched_children', [])
        }

    @pytest.mark.asyncio
    async def test_matched_amendment_beyond_the_cap_is_pinned_not_swallowed(self):
        """(a) The cap-sized digest sample excludes the match — pin it, don't lose it."""
        service = _stub_service(self._fifty_amendments())
        matched_id = self._matched_amendment_id()
        hits = [
            self._canonical_hit(0.9),
            self._matched_hit(matched_id, AMENDMENT_KIND, 0.8, '2026-08-01T00:00:30+00:00'),
        ]

        grouped = await group_search_results(service, _PROJECT_ID, hits)

        # (i) The collapse STILL happens — esc-5541 stays closed.
        assert [r['id'] for r in grouped] == [_CANONICAL_ID], (
            'The canonical must still absorb its child and be emitted exactly '
            f'once, got {[r["id"] for r in grouped]!r}'
        )
        block = grouped[0]['grouped']
        assert matched_id not in {d['id'] for d in block['amendments']}, (
            'PRECONDITION: this test is only meaningful while the matched '
            'amendment is beyond the digest cap and therefore absent from the '
            f'sample the backend returned, got {block["amendments"]!r}'
        )
        # (ii) ...and the swallowed child is genuinely REPRESENTED.
        pinned = {e['id']: e for e in block.get('matched_children', [])}
        assert matched_id in pinned, (
            'A child suppressed into its parent must appear in the block that '
            f'replaced it, got matched_children={block.get("matched_children")!r}. '
            'RED: nothing pins the swallowed child yet, so its id AND its text '
            'vanish from the response entirely.'
        )
        entry = pinned[matched_id]
        assert entry['content'] == _MATCHED_TEXT, (
            'The pinned entry must carry the child FULL body: that text was in '
            'the response BEFORE grouping, so truncating it would itself be the '
            f'retrieval regression, got {entry!r}'
        )
        assert 'digest' not in entry, (
            f'A pinned match is a full body, not a _DIGEST_CHARS digest, got {entry!r}'
        )
        assert entry['kind'] == AMENDMENT_KIND
        assert entry['matched'] is True
        assert entry['created_at'] == '2026-08-01T00:00:30+00:00'

    @pytest.mark.asyncio
    async def test_pinning_an_amendment_does_not_corrupt_the_counts(self):
        """The exact-count API still owns amendment_count / sighting_count / truncated."""
        service = _stub_service(
            self._fifty_amendments()
            + [_child(_SIGHT_1, _CANONICAL_ID, SIGHTING_KIND, 'seen', '2026-08-02T00:00:00+00:00')]
        )
        hits = [
            self._canonical_hit(0.9),
            self._matched_hit(
                self._matched_amendment_id(), AMENDMENT_KIND, 0.8, '2026-08-01T00:00:30+00:00'
            ),
        ]

        block = (await group_search_results(service, _PROJECT_ID, hits))[0]['grouped']

        assert block['amendment_count'] == self._FANOUT, (
            'amendment_count must stay the EXACT count API value, not len(digests) '
            f'and not len(digests) + pins, got {block!r}'
        )
        assert len(block['amendments']) == _AMENDMENT_DIGEST_CAP, (
            f'Pinning must not widen the bounded digest list, got {block["amendments"]!r}'
        )
        assert block.get('truncated') is True, (
            'truncated must still reflect count-vs-digests — a pin is not a digest, '
            f'got {block!r}'
        )
        assert block['sighting_count'] == 1, (
            f'The exact sighting count is unaffected by amendment pinning, got {block!r}'
        )

    @pytest.mark.asyncio
    async def test_matched_sighting_is_pinned_though_the_scroll_is_amendment_only(self):
        """(b) A sighting can NEVER appear in an amendment-only digest scroll."""
        sightings = [_SIGHT_1, _SIGHT_2, _SIGHT_3, '33333333-3333-4333-8333-333333333334']
        service = _stub_service(
            [_child(_AMEND_1, _CANONICAL_ID, AMENDMENT_KIND, 'a routine addendum', '2026-08-01T00:00:00+00:00')]
            + [
                _child(sight_id, _CANONICAL_ID, SIGHTING_KIND, 'seen again', f'2026-08-0{n + 2}T00:00:00+00:00')
                for n, sight_id in enumerate(sightings)
            ]
        )
        hits = [
            self._canonical_hit(0.9),
            self._matched_hit(_SIGHT_2, SIGHTING_KIND, 0.8, '2026-08-03T00:00:00+00:00'),
        ]

        grouped = await group_search_results(service, _PROJECT_ID, hits)

        assert [r['id'] for r in grouped] == [_CANONICAL_ID], (
            f'The collapse must still happen for a sighting, got {[r["id"] for r in grouped]!r}'
        )
        block = grouped[0]['grouped']
        assert _SIGHT_2 not in {d['id'] for d in block['amendments']}, (
            'PRECONDITION: the digest scroll filters kind=amendment, so a sighting '
            f'is structurally unlistable there, got {block["amendments"]!r}'
        )
        pinned = {e['id']: e for e in block.get('matched_children', [])}
        assert _SIGHT_2 in pinned, (
            'A matched SIGHTING is swallowed with nothing to represent it — it must '
            f'be pinned, got matched_children={block.get("matched_children")!r}'
        )
        assert pinned[_SIGHT_2]['content'] == _MATCHED_TEXT
        assert pinned[_SIGHT_2]['kind'] == SIGHTING_KIND, (
            'A sighting must be recorded AS a sighting, never misfiled as an '
            f'amendment, got {pinned[_SIGHT_2]!r}'
        )
        assert pinned[_SIGHT_2]['matched'] is True
        assert block['amendment_count'] == 1, (
            f'A pinned sighting must never inflate amendment_count, got {block!r}'
        )
        assert block['sighting_count'] == len(sightings), (
            f'sighting_count stays the exact count API value, got {block!r}'
        )
        assert 'truncated' not in block, (
            'The single amendment listed completely — pinning a sighting must not '
            f'invent a truncation marker, got {block!r}'
        )

    @pytest.mark.asyncio
    async def test_in_cap_matched_amendment_is_marked_in_place_not_duplicated(self):
        """Already in `amendments` → mark it there; do not copy it into matched_children."""
        service = _stub_service(_two_amendments_three_sightings())
        hits = [self._canonical_hit(0.9), _child_result(_AMEND_1, 0.8)]

        block = (await group_search_results(service, _PROJECT_ID, hits))[0]['grouped']

        by_id = {d['id']: d for d in block['amendments']}
        assert by_id[_AMEND_1].get('matched') is True, (
            'An in-cap matched amendment must be marked matched: True IN PLACE, '
            f'got {by_id[_AMEND_1]!r}'
        )
        assert 'matched' not in by_id[_AMEND_2], (
            'Fault/signal-only loudness: an unmatched digest omits the key entirely, '
            f'got {by_id[_AMEND_2]!r}'
        )
        assert 'matched_children' not in block, (
            'A child already represented by a digest must NOT be duplicated into a '
            f'second list, got {block!r}'
        )

    @pytest.mark.asyncio
    async def test_the_same_child_hit_twice_is_pinned_once(self):
        """Pinning dedups by id — a duplicated hit must not duplicate the entry."""
        service = _stub_service(self._fifty_amendments())
        matched_id = self._matched_amendment_id()
        hits = [
            self._canonical_hit(0.9),
            self._matched_hit(matched_id, AMENDMENT_KIND, 0.8, '2026-08-01T00:00:30+00:00'),
            self._matched_hit(matched_id, AMENDMENT_KIND, 0.7, '2026-08-01T00:00:30+00:00'),
        ]

        block = (await group_search_results(service, _PROJECT_ID, hits))[0]['grouped']

        assert [e['id'] for e in block['matched_children']] == [matched_id], (
            f'The same swallowed child must be pinned exactly once, got '
            f'{block["matched_children"]!r}'
        )

    @pytest.mark.asyncio
    async def test_every_swallowed_child_appears_in_its_parents_block(self):
        """The general property both live cases instantiate."""
        matched_id = self._matched_amendment_id()
        service = _stub_service(
            self._fifty_amendments()
            + [
                _child(_SIGHT_1, _CANONICAL_ID, SIGHTING_KIND, 'seen', '2026-08-02T00:00:00+00:00'),
                _child(_SIGHT_2, _CANONICAL_ID, SIGHTING_KIND, 'seen', '2026-08-03T00:00:00+00:00'),
            ]
        )
        in_cap_id = '44444444-4444-4444-8444-000000000002'
        hits = [
            self._canonical_hit(0.95),
            self._matched_hit(matched_id, AMENDMENT_KIND, 0.9, '2026-08-01T00:00:30+00:00'),
            self._matched_hit(_SIGHT_2, SIGHTING_KIND, 0.85, '2026-08-03T00:00:00+00:00'),
            self._matched_hit(in_cap_id, AMENDMENT_KIND, 0.8, '2026-08-01T00:00:02+00:00'),
        ]

        grouped = await group_search_results(service, _PROJECT_ID, hits)

        by_id = {r['id']: r for r in grouped}
        surviving = set(by_id)
        for hit in hits:
            parent_id = (hit.metadata or {}).get('parent_id')
            if parent_id is None or hit.id in surviving:
                continue
            block = by_id[parent_id]['grouped']
            assert hit.id in self._represented_ids(block), (
                f'{hit.id} was suppressed but its parent block does not represent '
                f'it — that is a silent retrieval regression, not a grouping. Block: {block!r}'
            )


class TestIsContestedChild:
    """The single home of the contested predicate (leaf γ's one constant to stamp)."""

    def test_experimental_namespace_key(self):
        assert CONTESTED_METADATA_KEY == 'x_contested', (
            'The flag lives in the x_ experimental namespace so it needs no '
            "amendment to task 3195's closed five-key reserved set and makes no "
            'unknown-key census noise'
        )

    def test_truthy_flag_is_contested(self):
        assert is_contested_child({CONTESTED_METADATA_KEY: True}) is True
        assert is_contested_child({CONTESTED_METADATA_KEY: 'yes'}) is True

    def test_absent_or_falsy_flag_is_not_contested(self):
        assert is_contested_child({}) is False
        assert is_contested_child({CONTESTED_METADATA_KEY: False}) is False
        assert is_contested_child({CONTESTED_METADATA_KEY: None}) is False
        assert is_contested_child(None) is False


class TestChildReadFaultContainment:
    """A backend fault must never be reported as "this canonical has no children".

    ``Mem0Backend`` deliberately PROPAGATES a Qdrant read-timeout as
    ``TimeoutError`` rather than collapsing it to a falsy value, precisely so
    a caller can tell "genuinely absent" from "backend failed".  Collapsing it
    here would resurrect the silent-orphan class task 3197 exists to close.
    """

    @pytest.mark.asyncio
    async def test_total_probe_timeout_is_not_reported_as_childless(self):
        """A faulted TOTAL probe must not return None — None means "no children"."""
        service = _stub_service(
            _two_amendments_three_sightings(),
            count_errors={_TOTAL: TimeoutError('qdrant count timed out')},
        )

        block = await build_grouped_document(service, _PROJECT_ID, _CANONICAL_ID)

        assert block is not None, (
            'A faulted child read must NOT return None — that is the wire shape '
            'for "this canonical genuinely has no children". '
            'RED: the fault escapes or collapses to None.'
        )
        assert block == {'children_unavailable': True, 'error_type': 'TimeoutError'}, (
            f'Expected the explicit unavailability marker, got {block!r}'
        )

    @pytest.mark.asyncio
    async def test_sighting_count_timeout_does_not_emit_a_fabricated_zero(self):
        """A faulted sighting count must not silently become sighting_count: 0."""
        service = _stub_service(
            _two_amendments_three_sightings(),
            count_errors={SIGHTING_KIND: TimeoutError('qdrant count timed out')},
        )

        block = await build_grouped_document(service, _PROJECT_ID, _CANONICAL_ID)

        assert block is not None
        assert block.get('sighting_count') != 0, (
            f'A faulted sighting count must never be reported as 0, got {block!r}'
        )
        assert 'sighting_count' not in block, (
            'A fault block must carry NO count at all rather than a partial one a '
            f'consumer could read as complete, got {block!r}'
        )
        assert block['children_unavailable'] is True
        assert block['error_type'] == 'TimeoutError'

    @pytest.mark.asyncio
    async def test_amendment_scroll_timeout_yields_the_unavailability_marker(self):
        """A faulted digest scroll is loud, and carries no partial digest list."""
        service = _stub_service(
            _two_amendments_three_sightings(),
            scroll_error=TimeoutError('qdrant scroll timed out'),
        )

        block = await build_grouped_document(service, _PROJECT_ID, _CANONICAL_ID)

        assert block is not None
        assert block == {'children_unavailable': True, 'error_type': 'TimeoutError'}, (
            f'Expected the explicit unavailability marker, got {block!r}'
        )
        assert 'amendments' not in block, (
            'A fault block must not carry a partial digest list a consumer could '
            f'read as exhaustive, got {block!r}'
        )

    @pytest.mark.asyncio
    async def test_fault_does_not_propagate_out_of_the_read_path(self):
        """A grouping fault must not escape and break an otherwise-working read."""
        service = _stub_service(
            _two_amendments_three_sightings(),
            scroll_error=RuntimeError('qdrant exploded'),
        )

        block = await build_grouped_document(service, _PROJECT_ID, _CANONICAL_ID)

        assert block == {'children_unavailable': True, 'error_type': 'RuntimeError'}, (
            'Any backend exception — not just TimeoutError — must be contained '
            f'and NAMED by type, got {block!r}'
        )


class TestBoundedListingHonesty:
    """A bounded digest listing must never read as exhaustive.

    ``get_memories_by_metadata`` is a single un-cursored scroll with no in-band
    truncation flag, so deriving a count from ``len(rows)`` would silently
    under-report a wide fan-out — the same honesty task 3197's
    ``DescendantScan.truncated`` provides on the delete path.
    """

    @staticmethod
    def _wide_amendment_fanout(extra: int = 5) -> list[dict]:
        total = _AMENDMENT_DIGEST_CAP + extra
        return [
            _child(
                f'44444444-4444-4444-8444-{index:012d}',
                _CANONICAL_ID,
                AMENDMENT_KIND,
                f'correction {index}',
                f'2026-08-01T00:00:{index:02d}+00:00',
            )
            for index in range(total)
        ] + [
            _child(_SIGHT_1, _CANONICAL_ID, SIGHTING_KIND, 'seen', '2026-08-01T00:00:00+00:00'),
            _child(_SIGHT_2, _CANONICAL_ID, SIGHTING_KIND, 'seen', '2026-08-01T00:00:01+00:00'),
            _child(_SIGHT_3, _CANONICAL_ID, SIGHTING_KIND, 'seen', '2026-08-01T00:00:02+00:00'),
        ]

    @pytest.mark.asyncio
    async def test_wide_fanout_is_capped_and_marked_truncated(self):
        """More amendments than the cap → capped list + truncated + EXACT count."""
        service = _stub_service(self._wide_amendment_fanout(extra=5))

        block = await build_grouped_document(service, _PROJECT_ID, _CANONICAL_ID)

        assert block is not None
        assert len(block['amendments']) == _AMENDMENT_DIGEST_CAP, (
            f'The digest list must be bounded by _AMENDMENT_DIGEST_CAP '
            f"({_AMENDMENT_DIGEST_CAP}), got {len(block['amendments'])}"
        )
        assert block.get('truncated') is True, (
            'A capped listing must carry truncated: True so it is never read as '
            f'exhaustive, got {block!r}. RED: no truncation marker yet.'
        )
        assert block['amendment_count'] == _AMENDMENT_DIGEST_CAP + 5, (
            'amendment_count must be the EXACT count from the count API, not '
            f"len(digests) ({len(block['amendments'])}), got "
            f"{block['amendment_count']}"
        )

    @pytest.mark.asyncio
    async def test_sighting_count_stays_exact_when_amendments_truncate(self):
        """The sighting count is a count-API read and is unaffected by the cap."""
        service = _stub_service(self._wide_amendment_fanout(extra=5))

        block = await build_grouped_document(service, _PROJECT_ID, _CANONICAL_ID)

        assert block is not None
        assert block['sighting_count'] == 3, (
            f"sighting_count must stay exact under truncation, got {block!r}"
        )
        sighting_filters = {'parent_id': _CANONICAL_ID, 'kind': SIGHTING_KIND}
        issued = [c.kwargs['filters'] for c in service.count_memories_by_metadata.await_args_list]
        assert sighting_filters in issued, (
            f'sighting_count must come from count_memories_by_metadata, got {issued!r}'
        )

    @pytest.mark.asyncio
    async def test_concurrent_write_between_reads_is_marked_truncated(self):
        """count > len(scroll) below the cap (a race) is truncation too."""
        # Two amendment rows on disk, but the exact count says seven — the shape
        # a write landing between the count and the scroll produces.
        service = _stub_service(
            _two_amendments_three_sightings(),
            counts={AMENDMENT_KIND: 7},
        )

        block = await build_grouped_document(service, _PROJECT_ID, _CANONICAL_ID)

        assert block is not None
        assert len(block['amendments']) == 2
        assert block['amendment_count'] == 7, (
            f'The exact count must win over len(digests), got {block!r}'
        )
        assert block.get('truncated') is True, (
            'Any count > len(digests) — cap OR concurrent write — must be marked '
            f'truncated, got {block!r}'
        )

    @pytest.mark.asyncio
    async def test_complete_listing_omits_the_truncated_key(self):
        """Fault-only loudness: a complete listing carries no truncated key at all."""
        service = _stub_service(_two_amendments_three_sightings())

        block = await build_grouped_document(service, _PROJECT_ID, _CANONICAL_ID)

        assert block is not None
        assert 'truncated' not in block, (
            "'truncated' must be OMITTED (not False) when the listing is "
            f'complete, matching the search tool degraded-key convention, got {block!r}'
        )
        assert block['amendment_count'] == 2


class TestZeroChildCommonPath:
    """The whole live corpus today: a canonical with NO children.

    Leaf α measured ``metadata.parent_id`` at zero live footprint, so this is
    the branch every result on the current fleet takes.  It must cost exactly
    one cheap exact count and emit NO ``grouped`` key at all — which is what
    makes every existing search/get response byte-identical to today's.
    """

    @pytest.mark.asyncio
    async def test_childless_canonical_returns_none(self):
        """No children → None, so the caller emits no `grouped` key whatsoever."""
        service = _stub_service([])

        block = await build_grouped_document(service, _PROJECT_ID, _CANONICAL_ID)

        assert block is None, (
            'A childless canonical must return None (no grouped key, response '
            f'byte-identical to today), got {block!r}'
        )

    @pytest.mark.asyncio
    async def test_childless_canonical_costs_exactly_one_count_and_no_scroll(self):
        """Count first; scroll only when it buys something a count cannot."""
        service = _stub_service([])

        await build_grouped_document(service, _PROJECT_ID, _CANONICAL_ID)

        assert service.get_memories_by_metadata.await_count == 0, (
            'A childless canonical must issue NO scroll — the amendment bodies '
            'a scroll buys cannot exist. RED: build_grouped_document scrolls '
            'unconditionally.'
        )
        assert service.count_memories_by_metadata.await_count == 1, (
            'A childless canonical must cost EXACTLY ONE exact count '
            f"({{'parent_id': cid}}), got "
            f'{service.count_memories_by_metadata.await_count} count call(s): '
            f'{service.count_memories_by_metadata.await_args_list!r}'
        )
        (only_call,) = service.count_memories_by_metadata.await_args_list
        assert only_call.kwargs['filters'] == {'parent_id': _CANONICAL_ID}, (
            'The single probe must be the un-kinded TOTAL child count, not a '
            f"per-kind sub-count, got {only_call.kwargs['filters']!r}"
        )

    @pytest.mark.asyncio
    async def test_nonzero_total_proceeds_to_the_detail_reads(self):
        """A non-zero total is what unlocks the sighting count and the scroll."""
        service = _stub_service(_two_amendments_three_sightings())

        block = await build_grouped_document(service, _PROJECT_ID, _CANONICAL_ID)

        assert block is not None
        issued = [c.kwargs['filters'] for c in service.count_memories_by_metadata.await_args_list]
        assert issued[0] == {'parent_id': _CANONICAL_ID}, (
            f'The TOTAL child count must be the FIRST read issued, got {issued!r}'
        )
        assert {'parent_id': _CANONICAL_ID, 'kind': SIGHTING_KIND} in issued, (
            f'The exact sighting count must be issued for a parented canonical, got {issued!r}'
        )
        assert service.get_memories_by_metadata.await_count == 1


class TestChildRecordSchema:
    """The add-only child-record representation, pinned as a contract."""

    def test_child_kinds_are_landed_kind_registry_members(self):
        """'amendment' and 'sighting' are already KIND_REGISTRY members (task 3195)."""
        assert AMENDMENT_KIND == 'amendment'
        assert SIGHTING_KIND == 'sighting'
        assert frozenset({'amendment', 'sighting'}) == CHILD_KINDS
        for kind in CHILD_KINDS:
            assert kind in KIND_REGISTRY, (
                f'{kind!r} must be a landed KIND_REGISTRY member — this leaf adds '
                'no vocabulary of its own.'
            )

    def test_child_record_metadata_validates(self):
        """A child is an ordinary entry: parent_id (full UUID) + a child kind."""
        for kind in sorted(CHILD_KINDS):
            meta = {'parent_id': _CANONICAL_ID, 'kind': kind}
            violations = validate_memory_metadata(meta, enforce_kind_registry=True)
            fatal = [v for v in violations if v.fatal]
            assert not fatal, (
                f'A well-formed {kind!r} child record must produce NO fatal '
                f'metadata violation, got {fatal!r}'
            )

    def test_short_hex_parent_id_is_rejected_by_shape_check(self):
        """parent_id must be a canonical full-UUID — the shape check is the schema."""
        meta = {'parent_id': 'deadbeef', 'kind': AMENDMENT_KIND}
        violations = validate_memory_metadata(meta, enforce_kind_registry=True)
        codes = {v.code for v in violations if v.fatal}
        assert 'invalid_parent_id_shape' in codes, (
            f'A non-UUID parent_id must be a FATAL shape violation, got {violations!r}'
        )

    @pytest.mark.asyncio
    async def test_grouped_read_never_edits_a_canonical(self):
        """Add-only: the read path issues no write/update/delete against the service."""
        service = _stub_service(_two_amendments_three_sightings())

        await build_grouped_document(service, _PROJECT_ID, _CANONICAL_ID)

        for write_attr in ('add_memory', 'update_memory', 'delete_memory', 'add_episode'):
            called = getattr(service, write_attr).await_count
            assert called == 0, (
                f'build_grouped_document must never call {write_attr} — this leaf is '
                f'strictly add-only on the read path, got {called} call(s)'
            )

    # (Deliberately no test that _AMENDMENT_DIGEST_CAP / _DIGEST_CHARS are
    # positive ints: asserting a module's own private constants against
    # themselves exercises no behaviour and no regression can break it.  Their
    # real behaviour is pinned by TestBoundedListingHonesty's
    # test_wide_fanout_is_capped_and_marked_truncated and by
    # test_long_body_truncated_to_named_cap_with_ellipsis above.)


class TestIdLessHitsNeverCollapse:
    """An EMPTY id is an id-LESS hit, not a shared identity.

    ``MemoryService._search_mem0`` builds every hit with ``id=item.get('id',
    '')``, so ``''`` is reachable on the live path.  Keying entries on it would
    make two distinct id-less hits the same entry — silently dropping a result
    and its content, which is the very retrieval regression this module exists
    to prevent.
    """

    @pytest.mark.asyncio
    async def test_two_id_less_hits_both_survive_as_separate_entries(self):
        service = _stub_service([])
        hits = [
            _result('', 'first orphan text', 0.9),
            _result('', 'second orphan text', 0.8),
        ]

        grouped = await group_search_results(service, _PROJECT_ID, hits)

        assert len(grouped) == 2, (
            'Two hits carrying id="" are two DIFFERENT results, not one — '
            f'collapsing them silently drops a body, got {grouped!r}'
        )
        assert [r['content'] for r in grouped] == ['first orphan text', 'second orphan text'], (
            f'Both bodies must survive, in rank order, got {grouped!r}'
        )

    @pytest.mark.asyncio
    async def test_an_id_less_hit_costs_no_child_probe(self):
        """`''` cannot be a parent id, so probing it is a guaranteed-zero count."""
        service = _stub_service([])

        await group_search_results(service, _PROJECT_ID, [_result('', 'orphan', 0.9)])

        assert service.count_memories_by_metadata.await_count == 0, (
            "An empty id must never enter the groupable set — {'parent_id': ''} "
            'can match nothing, so the count is pure waste. Got '
            f'{service.count_memories_by_metadata.await_args_list!r}'
        )


class TestPromotedParentMetadataShape:
    """One canonical, ONE wire shape — however it was reached.

    A direct search hit's ``metadata`` is mem0-normalized
    (``MemoryService._search_mem0`` forwards ``item['metadata']``), while
    ``get_memory_by_id`` returns the FULL raw Qdrant payload.  An upward-resolved
    parent must not therefore leak mem0-owned keys (``hash``/``user_id``/``role``
    …) or duplicate the body inside ``metadata``.
    """

    _CUSTOM = {'category': 'observations_and_summaries', 'kind': 'canonical', 'topic': 'merge-lane'}

    def _raw_record(self) -> dict:
        """A parent record as ``get_memory_by_id`` really returns it."""
        return {
            'id': _CANONICAL_ID,
            'content': 'the canonical claim',
            'metadata': {
                'data': 'the canonical claim',
                'hash': 'deadbeef',
                'user_id': 'dark_factory',
                'created_at': '2026-08-01T00:00:00+00:00',
                'updated_at': '2026-08-02T00:00:00+00:00',
                **self._CUSTOM,
            },
        }

    @pytest.mark.asyncio
    async def test_promoted_parent_metadata_matches_the_direct_hit_shape(self):
        upward_service = _stub_service(
            _two_amendments_three_sightings(), parents={_CANONICAL_ID: self._raw_record()}
        )
        direct_service = _stub_service(_two_amendments_three_sightings())

        promoted = (
            await group_search_results(upward_service, _PROJECT_ID, [_child_result(_AMEND_1, 0.77)])
        )[0]
        # The SAME record arriving as a direct hit: _search_mem0 hands over
        # mem0's already-normalized metadata, i.e. the custom half.
        direct = (
            await group_search_results(
                direct_service,
                _PROJECT_ID,
                [_result(_CANONICAL_ID, 'the canonical claim', 0.77, metadata=dict(self._CUSTOM))],
            )
        )[0]

        assert promoted['metadata'] == direct['metadata'], (
            'An upward-resolved parent must carry the SAME metadata as the same '
            f'record arriving as a direct hit, got {promoted["metadata"]!r} vs '
            f'{direct["metadata"]!r}'
        )
        assert promoted['metadata'] == self._CUSTOM

    @pytest.mark.asyncio
    async def test_mem0_managed_keys_never_leak_to_the_wire(self):
        service = _stub_service(
            _two_amendments_three_sightings(), parents={_CANONICAL_ID: self._raw_record()}
        )

        promoted = (
            await group_search_results(service, _PROJECT_ID, [_child_result(_AMEND_1, 0.77)])
        )[0]

        leaked = MEM0_MANAGED_METADATA_KEYS & set(promoted['metadata'])
        assert not leaked, (
            'mem0-owned payload keys are internal and must be stripped by '
            f'split_managed_metadata, got {sorted(leaked)!r} in {promoted["metadata"]!r}'
        )
        assert promoted['content'] == 'the canonical claim'
        assert 'data' not in promoted['metadata'], (
            'The body must not be duplicated inside metadata — that is response '
            f'size proportional to content length, got {promoted["metadata"]!r}'
        )
        assert promoted['created_at'] == '2026-08-01T00:00:00+00:00', (
            'created_at lands in the MANAGED half and must still be read back '
            f'from it, got {promoted!r}'
        )


class TestFanOutIsBounded:
    """The hot read path must not burst one Qdrant request per hit, unbounded.

    The ``search`` tool clamps ``limit`` to 1000, so the size of this module's
    fan-out is chosen by the CALLER: without a ceiling a single grouped search
    could put ~1000 concurrent counts into one gather, contending with the very
    client pool the search itself used.
    """

    @staticmethod
    def _peak_tracking_service(peak: list[int], in_flight: list[int]) -> AsyncMock:
        async def _count(*, project_id: str, filters: dict):
            in_flight[0] += 1
            peak[0] = max(peak[0], in_flight[0])
            await asyncio.sleep(0.005)
            in_flight[0] -= 1
            return 0

        service = AsyncMock()
        service.count_memories_by_metadata = AsyncMock(side_effect=_count)
        return service

    @pytest.mark.asyncio
    async def test_child_probes_never_exceed_the_named_ceiling(self):
        peak, in_flight = [0], [0]
        service = self._peak_tracking_service(peak, in_flight)
        hits = [
            _result(f'55555555-5555-4555-8555-{index:012d}', f'hit {index}', 0.9)
            for index in range(_MAX_CONCURRENT_GROUP_READS * 3)
        ]

        grouped = await group_search_results(service, _PROJECT_ID, hits)

        assert len(grouped) == len(hits), 'Bounding concurrency must not drop results'
        assert peak[0] <= _MAX_CONCURRENT_GROUP_READS, (
            f'At most {_MAX_CONCURRENT_GROUP_READS} child probes may be in flight at '
            f'once, saw {peak[0]} — an unbounded gather over a 1000-hit search would '
            'saturate the Qdrant client pool'
        )
        assert peak[0] > 1, (
            f'The probes must still run CONCURRENTLY up to the ceiling, saw peak '
            f'{peak[0]} — a bound of one would serialize the whole search'
        )


class TestDetailReadsAreConcurrent:
    """The three detail reads share no data dependency, so they must not serialize."""

    @pytest.mark.asyncio
    async def test_sighting_count_amendment_count_and_scroll_overlap(self):
        # A 3-party barrier: it can only be passed if all three reads are in
        # flight AT ONCE. Serialized, the first read waits on two parties that
        # will never arrive, and wait_for turns that into a loud timeout.
        barrier = asyncio.Barrier(3)
        children = _two_amendments_three_sightings()

        async def _rendezvous():
            await asyncio.wait_for(barrier.wait(), timeout=5)

        async def _count(*, project_id: str, filters: dict):
            if 'kind' not in filters:
                return len(children)  # the TOTAL probe deliberately runs first
            await _rendezvous()
            kind = filters['kind']
            return len([c for c in children if c['metadata']['kind'] == kind])

        async def _scroll(*, project_id: str, filters: dict, limit: int = 1000):
            await _rendezvous()
            return [c for c in children if c['metadata']['kind'] == AMENDMENT_KIND][:limit]

        service = AsyncMock()
        service.count_memories_by_metadata = AsyncMock(side_effect=_count)
        service.get_memories_by_metadata = AsyncMock(side_effect=_scroll)

        block = await build_grouped_document(service, _PROJECT_ID, _CANONICAL_ID)

        assert block == {
            'amendments': [
                {
                    'id': _AMEND_1,
                    'digest': 'first correction',
                    'created_at': '2026-08-01T00:00:00+00:00',
                    'kind': AMENDMENT_KIND,
                },
                {
                    'id': _AMEND_2,
                    'digest': 'second correction',
                    'created_at': '2026-08-02T00:00:00+00:00',
                    'kind': AMENDMENT_KIND,
                },
            ],
            'amendment_count': 2,
            'sighting_count': 3,
        }, (
            'The two counts and the digest scroll must be issued CONCURRENTLY — '
            'a TimeoutError contained as children_unavailable here means they '
            f'were serialized, got {block!r}'
        )

    @pytest.mark.asyncio
    async def test_the_total_probe_still_gates_the_detail_reads(self):
        """Concurrency must not leak upward: zero children still costs ONE count."""
        service = _stub_service([])

        block = await build_grouped_document(service, _PROJECT_ID, _CANONICAL_ID)

        assert block is None
        assert service.count_memories_by_metadata.await_count == 1, (
            'The TOTAL probe short-circuits BEFORE the detail reads — gathering '
            'them must not turn the zero-child common path into three reads, got '
            f'{service.count_memories_by_metadata.await_args_list!r}'
        )
        assert service.get_memories_by_metadata.await_count == 0


class TestPointIdParentIsVerified:
    """``grouped.parent.id`` must never be an unverified pointer.

    ``group_search_results`` already stamps ``parent_unresolved`` for a dangling
    ``parent_id``; the point-id surface must not silently report the same
    dangling pointer as a real parent.  ``build_grouped_document`` cannot tell
    the difference on its own — "parent exists but is childless" and "parent does
    not exist" are both ``None`` — so existence is probed explicitly.
    """

    @staticmethod
    def _child_record() -> dict:
        return {
            'id': _AMEND_1,
            'content': 'a correction',
            'metadata': {'data': 'a correction', 'parent_id': _CANONICAL_ID, 'kind': AMENDMENT_KIND},
        }

    @pytest.mark.asyncio
    async def test_dangling_parent_is_marked_unresolved(self):
        # No parents registered → get_memory_by_id misses on the pointer.
        service = _stub_service(_two_amendments_three_sightings())

        grouped = await group_memory_document(
            service, _PROJECT_ID, _AMEND_1, self._child_record()
        )

        assert grouped is not None
        assert grouped[PARENT_UNRESOLVED_KEY] is True, (
            'A parent_id that resolves to nothing must be reported as UNRESOLVED, '
            f'not handed back as a real parent, got {grouped!r}'
        )
        assert grouped['parent']['id'] == _CANONICAL_ID, (
            'The pointer itself is still reported — marked, not hidden'
        )

    @pytest.mark.asyncio
    async def test_resolvable_parent_carries_no_marker(self):
        service = _stub_service(
            _two_amendments_three_sightings(), parents={_CANONICAL_ID: _parent_record()}
        )

        grouped = await group_memory_document(
            service, _PROJECT_ID, _AMEND_1, self._child_record()
        )

        assert grouped is not None
        assert PARENT_UNRESOLVED_KEY not in grouped, (
            f'Fault-only loudness: a live parent gets no marker, got {grouped!r}'
        )
        assert PARENT_UNAVAILABLE_KEY not in grouped
        assert grouped['parent']['sighting_count'] == 3, 'The group itself still comes through'

    @pytest.mark.asyncio
    async def test_failed_probe_is_unavailable_not_unresolved(self):
        """Not knowing is a different claim from knowing the parent is absent."""
        service = _stub_service(
            _two_amendments_three_sightings(), parents={_CANONICAL_ID: _parent_record()}
        )
        service.get_memory_by_id = AsyncMock(side_effect=TimeoutError('qdrant timed out'))

        grouped = await group_memory_document(
            service, _PROJECT_ID, _AMEND_1, self._child_record()
        )

        assert grouped is not None
        assert grouped[PARENT_UNAVAILABLE_KEY] is True, (
            f'A FAILED existence probe must be its own marker, got {grouped!r}'
        )
        assert grouped['error_type'] == 'TimeoutError'
        assert PARENT_UNRESOLVED_KEY not in grouped, (
            'A probe that failed must NOT be reported as a confirmed miss — that '
            f'would fabricate a dangling-pointer fault, got {grouped!r}'
        )

    @pytest.mark.asyncio
    async def test_a_canonical_read_costs_no_parent_probe(self):
        """The verification is on the CHILD branch only."""
        service = _stub_service(_two_amendments_three_sightings())
        record = {
            'id': _CANONICAL_ID,
            'content': 'the canonical claim',
            'metadata': {'data': 'the canonical claim', 'kind': 'canonical'},
        }

        await group_memory_document(service, _PROJECT_ID, _CANONICAL_ID, record)

        assert service.get_memory_by_id.await_count == 0, (
            'A canonical has no parent pointer to verify, so the point-id surface '
            f'must issue no extra read, got {service.get_memory_by_id.await_args_list!r}'
        )


class TestContestedKeyComposesTheExperimentalPrefix:
    """The flag is built from the vocabulary module's prefix, not a re-spelling."""

    def test_key_is_composed_not_hardcoded(self):
        composed = f'{EXPERIMENTAL_KEY_PREFIX}contested'
        assert composed == CONTESTED_METADATA_KEY, (
            'CONTESTED_METADATA_KEY must COMPOSE memory_metadata.'
            'EXPERIMENTAL_KEY_PREFIX so a prefix change cannot silently desync '
            f'the read side from the vocabulary module, got {CONTESTED_METADATA_KEY!r}'
        )
