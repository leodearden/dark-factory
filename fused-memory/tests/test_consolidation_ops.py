"""`services/consolidation_ops.py` — the retain arm of a consolidation (task 5238).

The arm `consolidate_memories` ran inline — mint the canonical, tag the
retained peers, list the topic's closure — lifted into one function that
needs no server, no tool closure and no `mcp.tool()` registration, so the
auto-consolidation executor can call it directly (PRD contract C3).

Harness per `tests/test_consolidate_memories_tool.py`: the mem0_update
leaves are a REAL `Mem0UpdateConfig`, not a bare Mock. An `AsyncMock`
makes every attribute a Mock, which the fail-closed authz resolver
rejects — so without the real leaf every case here would pass for the
wrong reason, reporting `Mem0UpdateNotAuthorized` instead of the property
under test.

The two reads this arm issues return DIFFERENT shapes and are modelled by
two deliberately separate row shapers, as the tool's suite does, so the
fake cannot re-cross what the real backends keep apart:

    get_memory_by_id         -> `created_at` NESTED in the raw payload
    get_memories_by_metadata -> `created_at` LIFTED flat by the scroll
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fused_memory.services.consolidation_ops import execute_retain_consolidation

from fused_memory.config.schema import Mem0UpdateConfig
from fused_memory.models.memory import AddMemoryResponse

PROJECT_ID = 'dark_factory'
# On the default allowlist for both mem0_update arms, so no case here can
# fail for an authorization reason and be misread as a result.
AGENT = 'recon-stage-memory_consolidator'
RUN_ID = 'run-abc'
TOPIC = 'memory-consolidation'
CONTENT = 'Consolidation folds a duplicate cluster into one canonical claim.'
CATEGORY = 'procedural_knowledge'
SESSION_ID = 'session-xyz'
CAUSATION_ID = 'causation-xyz'
SOURCE = 'full_recon'

CANONICAL = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'
M1 = 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb'
M2 = 'cccccccc-cccc-4ccc-8ccc-cccccccccccc'
INCUMBENT = 'dddddddd-dddd-4ddd-8ddd-dddddddddddd'

CREATED_AT = '2026-01-01T00:00:00+00:00'


def _point_row(memory_id, **metadata):
    """What `get_memory_by_id` returns: `created_at` NESTED in the payload."""
    return {
        'id': memory_id,
        'content': f'record {memory_id}',
        'metadata': {'topic': TOPIC, 'created_at': CREATED_AT, **metadata},
    }


def _scroll_row(memory_id, **metadata):
    """What `get_memories_by_metadata` returns: `created_at` LIFTED flat."""
    return {
        'id': memory_id,
        'content': f'record {memory_id}',
        'created_at': CREATED_AT,
        'metadata': {'topic': TOPIC, **metadata},
    }


def make_service(
    *,
    members=None,
    canonical_peers=(),
    read_errors=None,
    update_errors=None,
    update_raises=None,
    scroll_error=None,
    minted_ids=(CANONICAL,),
    topic_total=None,
):
    """A MemoryService mock modelling one topic under consolidation.

    *canonical_peers* is the set of ids whose payload already carries
    `canonical: True` — the peer the arm must refuse rather than demote.
    *read_errors* maps an id to an exception `get_memory_by_id` raises, the
    fail-closed case: a check that did not ANSWER is not a check that said
    "not canonical".

    *update_errors* and *update_raises* are the two halves of
    `update_memory`'s split contract: it REPORTS MemoryNotFound and its
    authorization refusals by RETURNING `{'error_type': ...}`, while every
    other failure goes through `_journaled_backend_call`, which logs and
    RE-RAISES. Both must collapse to the same per-id verdict.

    *minted_ids* is what the canonical write reports back; `()` models a
    write that landed nothing while raising nothing.
    """
    members = [M1, M2] if members is None else members
    canonical_peers = set(canonical_peers)
    read_errors = read_errors or {}
    update_errors = update_errors or {}
    update_raises = update_raises or {}

    svc = AsyncMock()
    svc.config.mem0_update = Mem0UpdateConfig()

    async def _add(**kwargs):
        return AddMemoryResponse(memory_ids=list(minted_ids), message='ok')

    svc.add_memory = AsyncMock(side_effect=_add)

    async def _get(project_id=None, memory_id=None, **_):
        assert isinstance(memory_id, str), 'the arm must name the id it reads'
        if memory_id in read_errors:
            raise read_errors[memory_id]
        if memory_id in canonical_peers:
            return _point_row(memory_id, canonical=True)
        return _point_row(memory_id)

    svc.get_memory_by_id = AsyncMock(side_effect=_get)

    async def _update(memory_id=None, **kwargs):
        assert isinstance(memory_id, str), 'the arm must name the id it patches'
        if memory_id in update_raises:
            raise update_raises[memory_id]
        if memory_id in update_errors:
            return update_errors[memory_id]
        return {
            'status': 'updated',
            'store': 'mem0',
            'id': memory_id,
            'metadata_patched': True,
        }

    svc.update_memory = AsyncMock(side_effect=_update)

    async def _scroll(**kwargs):
        if scroll_error is not None:
            raise scroll_error
        return [
            _scroll_row(m, **({'canonical': True} if m in canonical_peers else {}))
            for m in members
        ]

    svc.get_memories_by_metadata = AsyncMock(side_effect=_scroll)

    async def _count(**_):
        return len(members) if topic_total is None else topic_total

    svc.count_memories_by_metadata = AsyncMock(side_effect=_count)
    return svc


async def call_execute(svc, **overrides):
    args = {
        'project_id': PROJECT_ID,
        'topic': TOPIC,
        'canonical_content': CONTENT,
        'retain_ids': [M1, M2],
        'category': CATEGORY,
        'agent_id': AGENT,
        'run_id': RUN_ID,
        'extra_canonical_meta': {},
        'session_id': SESSION_ID,
        'causation_id': CAUSATION_ID,
        'source': SOURCE,
    }
    args.update(overrides)
    return await execute_retain_consolidation(svc, **args)


def _patched_ids(svc):
    return [c.kwargs['memory_id'] for c in svc.update_memory.await_args_list]


class TestTheCanonicalIsMinted:
    """The mint is the arm's first write and the one irreversible ordering
    constraint it inherits: no metadata patch may precede it, because a
    write-then-tag run whose write fails costs nothing, while a
    tag-then-write run whose write fails has already stamped peers for a
    topic that has no canonical."""

    @pytest.mark.asyncio
    async def test_the_canonical_carries_the_topic_and_the_canonical_flag(self):
        svc = make_service()

        await call_execute(svc)

        svc.add_memory.assert_awaited_once()
        meta = svc.add_memory.await_args.kwargs['metadata']
        assert meta['topic'] == TOPIC
        assert meta['canonical'] is True

    @pytest.mark.asyncio
    async def test_every_extra_canonical_meta_key_reaches_the_write(self):
        """`extra_canonical_meta` is how the tool passes the caller's own
        cleaned metadata AND the `supersedes` list it computed; dropping a
        key here would silently change what the canonical durably claims."""
        svc = make_service()

        await call_execute(
            svc,
            extra_canonical_meta={'supersedes': ['x', 'y'], 'kind': 'index'},
        )

        meta = svc.add_memory.await_args.kwargs['metadata']
        assert meta['supersedes'] == ['x', 'y']
        assert meta['kind'] == 'index'

    @pytest.mark.asyncio
    async def test_the_write_provenance_is_forwarded_verbatim(self):
        """Every one of these lands in the WriteJournal and the causation
        graph. A dropped `session_id`/`causation_id`/`_source` changes what
        the fold RECORDS without changing what it does, which is the class
        of drift no envelope assertion can see."""
        svc = make_service()

        await call_execute(svc)

        kwargs = svc.add_memory.await_args.kwargs
        assert kwargs['content'] == CONTENT
        assert kwargs['category'] == CATEGORY
        assert kwargs['project_id'] == PROJECT_ID
        assert kwargs['agent_id'] == AGENT
        assert kwargs['session_id'] == SESSION_ID
        assert kwargs['causation_id'] == CAUSATION_ID
        assert kwargs['_source'] == SOURCE

    @pytest.mark.asyncio
    async def test_a_mint_that_lands_nothing_refuses_and_patches_nothing(self):
        """`AddMemoryResponse.memory_ids` can come back EMPTY without
        raising. Carrying a None canonical id forward would tag peers into
        a topic whose canonical does not exist."""
        svc = make_service(minted_ids=())

        result = await call_execute(svc)

        assert result['error_type'] == 'CanonicalWriteFailed'
        assert result['topic'] == TOPIC
        svc.update_memory.assert_not_awaited()


class TestThePeersAreTaggedInPlace:
    """Each peer keeps its Qdrant point id, so every citation, parent
    pointer and supersedes edge already aimed at it stays valid. That id
    stability is the entire reason the arm exists."""

    @pytest.mark.asyncio
    async def test_one_patch_per_retained_peer(self):
        svc = make_service()

        await call_execute(svc)

        assert _patched_ids(svc) == [M1, M2]

    @pytest.mark.asyncio
    async def test_the_patch_is_the_topic_alone_and_merges(self):
        """`content=None` keeps the peer's vector still; `merge` keeps its
        own source/run_id/parent, which are not this arm's to discard.
        `canonical` would mint a second claimant for the topic and
        `parent_id` would make a peer a child of the canonical."""
        svc = make_service()

        await call_execute(svc)

        for call in svc.update_memory.await_args_list:
            assert call.kwargs['metadata_patch'] == {'topic': TOPIC}
            assert call.kwargs['content'] is None
            assert call.kwargs['metadata_mode'] == 'merge'
            assert call.kwargs.get('metadata_delete_keys') is None
            assert 'canonical' not in call.kwargs['metadata_patch']
            assert 'parent_id' not in call.kwargs['metadata_patch']


class TestAPeerThatCannotBeTaggedIsNamed:
    """One failure costs one peer, never the arm: the peers that CAN be
    tagged are, because re-running to catch the rest would re-write the
    canonical — the +1-per-pass ratchet the op exists to end."""

    @pytest.mark.asyncio
    async def test_a_canonical_peer_is_refused_not_demoted(self):
        """Patching `canonical: False` would also hold the uniqueness
        invariant, but it would silently rewrite a claim this arm was not
        asked to touch. A prior canonical in the retain list is usually an
        authoring mistake — that record belongs in `supersedes`."""
        svc = make_service(canonical_peers=[M1])

        result = await call_execute(svc)

        assert [f['error_type'] for f in result['retain_failures']] == [
            'RetainedPeerIsCanonical'
        ]
        assert result['retain_failures'][0]['id'] == M1
        assert M1 not in _patched_ids(svc)

    @pytest.mark.asyncio
    async def test_a_peer_that_cannot_be_read_fails_closed(self):
        """A check that did not ANSWER is not a check that said "not
        canonical". Refusing costs one peer's tag, which a caller can
        retry; tagging on an unproven check mints a duplicate canonical
        that nothing downstream will catch."""
        svc = make_service(read_errors={M1: TimeoutError('qdrant timeout')})

        result = await call_execute(svc)

        assert [f['error_type'] for f in result['retain_failures']] == [
            'RetainCheckFailed'
        ]
        assert M1 not in _patched_ids(svc)

    @pytest.mark.asyncio
    async def test_one_failure_does_not_abort_the_arm(self):
        svc = make_service(canonical_peers=[M1])

        result = await call_execute(svc)

        assert result['retained'] == [M2]
        assert _patched_ids(svc) == [M2]

    @pytest.mark.asyncio
    async def test_a_returned_refusal_and_a_raise_are_the_same_verdict(self):
        """`update_memory` reports MemoryNotFound by RETURNING a structured
        rejection and every backend failure by RAISING. Guarding only one
        shape records a refusal as a success, or lets one timeout flatten
        the whole envelope."""
        returned = make_service(
            update_errors={
                M1: {'error': 'gone', 'error_type': 'MemoryNotFound'},
            }
        )
        raised = make_service(update_raises={M1: TimeoutError('qdrant timeout')})

        by_return = await call_execute(returned)
        by_raise = await call_execute(raised)

        assert by_return['retained'] == [M2]
        assert by_raise['retained'] == [M2]
        assert by_return['retain_failures'][0]['error_type'] == 'MemoryNotFound'
        assert by_raise['retain_failures'][0]['error_type'] == 'TimeoutError'
        assert by_raise['retain_failures'][0]['id'] == M1


class TestTheClosureIsScrolledNeverSearched:
    """A ranked top-N read can silently omit the canonical this call just
    wrote — the exact failure that made the original incident's "re-derive
    via search" correction route dispatch back into the superseded members
    it was collapsing."""

    @pytest.mark.asyncio
    async def test_the_listing_comes_from_the_deterministic_scroll(self):
        svc = make_service()

        result = await call_execute(svc)

        svc.get_memories_by_metadata.assert_awaited_once()
        kwargs = svc.get_memories_by_metadata.await_args.kwargs
        assert kwargs['filters'] == {'topic': TOPIC}
        assert kwargs['project_id'] == PROJECT_ID
        assert [m['id'] for m in result['topic_members']] == [M1, M2]
        svc.search.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_an_unreadable_scroll_degrades_to_not_available(self):
        """`[]` alone reads as "this topic has no members" — the overclaim
        the op exists to eliminate, and doubly wrong on a call that just
        wrote a canonical into that very topic."""
        svc = make_service(scroll_error=TimeoutError('qdrant timeout'))

        result = await call_execute(svc)

        assert result['topic_members_available'] is False
        assert result['topic_members'] == []
        assert result['canonical_id'] == CANONICAL


class TestTheEnvelopeIsTheSharedBuilders:
    """One envelope definition, two callers: the tool keeps calling
    `build_consolidation_result` with its delete-arm data, and this arm
    calls it with those lists empty. No second status rule."""

    @pytest.mark.asyncio
    async def test_a_clean_retain_run_is_consolidated(self):
        svc = make_service()

        result = await call_execute(svc)

        assert result['status'] == 'consolidated'
        assert result['canonical_id'] == CANONICAL
        assert result['topic'] == TOPIC
        assert result['retained'] == [M1, M2]
        assert result['retain_failures'] == []

    @pytest.mark.asyncio
    async def test_every_delete_arm_disposition_is_present_and_empty(self):
        """Present rather than absent, so a caller reads
        `result['survivors']` without a membership test and cannot mistake
        "this arm performs no deletes" for "the deletes were not reported"."""
        svc = make_service()

        result = await call_execute(svc)

        assert result['deleted'] == []
        assert result['failed_deletes'] == []
        assert result['survivors'] == []
        assert result['survivor_check_failed'] == []

    @pytest.mark.asyncio
    async def test_a_failed_tag_makes_the_run_partial(self):
        svc = make_service(canonical_peers=[M1])

        result = await call_execute(svc)

        assert result['status'] == 'partial'
