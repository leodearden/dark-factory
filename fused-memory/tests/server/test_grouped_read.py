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

from unittest.mock import AsyncMock

import pytest

from fused_memory.memory_metadata import KIND_REGISTRY, validate_memory_metadata
from fused_memory.server.grouped_read import (
    AMENDMENT_KIND,
    CHILD_KINDS,
    SIGHTING_KIND,
    _AMENDMENT_DIGEST_CAP,
    _DIGEST_CHARS,
    _DIGEST_ELLIPSIS,
    build_grouped_document,
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
    count_error: Exception | None = None,
) -> AsyncMock:
    """AsyncMock service dispatching the three Mem0 read primitives over *children*.

    *counts* overrides the derived counts per ``kind`` filter value (or
    ``_TOTAL`` for the un-kinded total) so a test can model a scroll bounded
    below the exact count.  *parents* maps memory id → raw record for
    ``get_memory_by_id``.
    """
    service = AsyncMock()

    def _matches(child: dict, filters: dict) -> bool:
        meta = child['metadata']
        return all(meta.get(key) == value for key, value in filters.items())

    def _count(*, project_id: str, filters: dict):
        if counts is not None:
            key = filters.get('kind', _TOTAL)
            if key in counts:
                return counts[key]
        return len([c for c in children if _matches(c, filters)])

    def _scroll(*, project_id: str, filters: dict, limit: int = 1000):
        matched = [c for c in children if _matches(c, filters)]
        return matched[:limit]

    def _get_by_id(*, project_id: str, memory_id: str):
        return (parents or {}).get(memory_id)

    service.count_memories_by_metadata = AsyncMock(
        side_effect=count_error or _count
    )
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
        assert CHILD_KINDS == frozenset({'amendment', 'sighting'})
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

    @pytest.mark.asyncio
    async def test_amendment_digest_cap_is_a_named_constant(self):
        """The digest cap is a named module constant, not an inline literal."""
        assert isinstance(_AMENDMENT_DIGEST_CAP, int) and _AMENDMENT_DIGEST_CAP > 0
        assert isinstance(_DIGEST_CHARS, int) and _DIGEST_CHARS > 0
