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

from fused_memory.config.schema import Mem0UpdateConfig
from fused_memory.models.memory import AddMemoryResponse
from fused_memory.services.consolidation_ops import (
    TOPIC_MEMBER_LIMIT,
    RetainArmApplied,
    RetainArmRefused,
    apply_retain_arm,
    execute_retain_consolidation,
)

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
    closure_scroll_error=None,
    scroll_rows=None,
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
    write that landed nothing while raising nothing. *scroll_rows*
    replaces the topic's rows wholesale, for the shapes *members* cannot
    express — a topic naming no canonical, or naming two.

    The scroll honours `filters` and `limit` as the real one does.
    *scroll_error* fails every scroll; *closure_scroll_error* fails only
    the topic-wide listing, so a read that also names `canonical` still
    answers.
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

    # Patches LAND, so a read-back is a real question about the world
    # rather than a restatement of the row the shaper built. That is what
    # makes "the incumbent was not touched" checkable: a demotion would
    # merge `canonical: False` into its payload and the read-back would
    # see it.
    landed: dict[str, dict] = {}

    async def _get(project_id=None, memory_id=None, **_):
        assert isinstance(memory_id, str), 'the arm must name the id it reads'
        if memory_id in read_errors:
            raise read_errors[memory_id]
        canonical = {'canonical': True} if memory_id in canonical_peers else {}
        return _point_row(memory_id, **canonical, **landed.get(memory_id, {}))

    svc.get_memory_by_id = AsyncMock(side_effect=_get)

    async def _update(memory_id=None, **kwargs):
        assert isinstance(memory_id, str), 'the arm must name the id it patches'
        if memory_id in update_raises:
            raise update_raises[memory_id]
        if memory_id in update_errors:
            return update_errors[memory_id]
        # A server-side Qdrant payload MERGE, never a replace — the shape
        # that lets a peer KEEP a `canonical: True` it already carried.
        landed.setdefault(memory_id, {}).update(kwargs.get('metadata_patch') or {})
        return {
            'status': 'updated',
            'store': 'mem0',
            'id': memory_id,
            'metadata_patched': True,
        }

    svc.update_memory = AsyncMock(side_effect=_update)

    async def _scroll(*, filters=None, limit=None, **_):
        if scroll_error is not None:
            raise scroll_error
        if closure_scroll_error is not None and filters == {'topic': TOPIC}:
            raise closure_scroll_error
        rows = (
            list(scroll_rows)
            if scroll_rows is not None
            else [
                _scroll_row(m, **({'canonical': True} if m in canonical_peers else {}))
                for m in members
            ]
        )
        matching = [
            row
            for row in rows
            if all(
                row['metadata'].get(key) == value
                for key, value in (filters or {}).items()
            )
        ]
        return matching if limit is None else matching[:limit]

    svc.get_memories_by_metadata = AsyncMock(side_effect=_scroll)

    async def _count(**_):
        return len(members) if topic_total is None else topic_total

    svc.count_memories_by_metadata = AsyncMock(side_effect=_count)
    return svc


def _arm_args(**overrides):
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
    return args


async def call_execute(svc, **overrides):
    return await execute_retain_consolidation(svc, **_arm_args(**overrides))


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


class TestTheArmsWritesStandApartFromItsClosure:
    """`apply_retain_arm` is the arm's writes alone, for a caller with a
    delete arm of its own: it lists no closure, so that caller reads the
    closure once, after its fold, and has one source for it."""

    @pytest.mark.asyncio
    async def test_the_writes_report_a_typed_outcome_and_list_no_closure(self):
        svc = make_service()

        arm = await apply_retain_arm(svc, **_arm_args())

        assert arm == RetainArmApplied(
            canonical_id=CANONICAL,
            canonical_supersedes=[],
            retained=[M1, M2],
            retain_failures=[],
        )
        svc.get_memories_by_metadata.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_refusal_is_its_own_type_carrying_the_wire_shape(self):
        svc = make_service(minted_ids=())

        arm = await apply_retain_arm(svc, **_arm_args())

        assert isinstance(arm, RetainArmRefused)
        assert arm.response['error_type'] == 'CanonicalWriteFailed'
        svc.update_memory.assert_not_awaited()


class TestTagOnlyTouchesNothingOnTheIncumbent:
    """PRD D14: a topic that ALREADY has a canonical does not need another
    one. `canonical_content=None` skips the mint entirely and stamps the
    topic onto the unstamped members, which is the whole job when the
    incumbent's claim is still correct.

    Minting anyway would be the +1-per-pass ratchet the op exists to end,
    reached by the most natural reading of "consolidate this topic".

    Tag-only obeys the rule `TestTheCanonicalIsMinted` states — no metadata
    patch before a canonical is established — and it establishes one by
    RESOLVING the incumbent, so a failed resolution leaves nothing touched.
    """

    @pytest.mark.asyncio
    async def test_the_incumbent_is_resolved_and_nothing_is_minted(self):
        svc = make_service(members=[INCUMBENT, M1, M2], canonical_peers=[INCUMBENT])

        result = await call_execute(svc, canonical_content=None)

        assert result['canonical_id'] == INCUMBENT
        svc.add_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_incumbent_is_not_patched_and_reads_back_unchanged(self):
        """`content_amend=False` is load-bearing: this arm stamps metadata
        and writes new records, and refreshing the incumbent's text would
        make the deliberately wider metadata bar a route to a silent
        rewrite. Nothing is refreshed on it at all — not its content, not
        its metadata, not its `canonical` claim."""
        svc = make_service(members=[INCUMBENT, M1, M2], canonical_peers=[INCUMBENT])
        before = await svc.get_memory_by_id(
            project_id=PROJECT_ID, memory_id=INCUMBENT
        )

        await call_execute(svc, canonical_content=None)

        assert set(_patched_ids(svc)) == {M1, M2}
        assert INCUMBENT not in _patched_ids(svc)
        after = await svc.get_memory_by_id(
            project_id=PROJECT_ID, memory_id=INCUMBENT
        )
        assert after == before

    @pytest.mark.asyncio
    async def test_the_envelope_reports_a_clean_tag_only_fold(self):
        svc = make_service(members=[INCUMBENT, M1, M2], canonical_peers=[INCUMBENT])

        result = await call_execute(svc, canonical_content=None)

        assert result['status'] == 'consolidated'
        assert result['retained'] == [M1, M2]

    @pytest.mark.asyncio
    async def test_an_incumbent_passed_in_retain_is_refused_not_demoted(self):
        """The caller's predicate is meant to strip it and disclose the
        strip. When one slips through, the loop's existing
        `RetainedPeerIsCanonical` branch is what catches it — refusing a
        peer it cannot prove non-canonical, rather than quietly patching
        `canonical: False` onto a claim it was not asked to touch."""
        svc = make_service(members=[INCUMBENT, M1, M2], canonical_peers=[INCUMBENT])

        result = await call_execute(
            svc, canonical_content=None, retain_ids=[INCUMBENT, M1, M2]
        )

        assert [f['error_type'] for f in result['retain_failures']] == [
            'RetainedPeerIsCanonical'
        ]
        assert result['retain_failures'][0]['id'] == INCUMBENT
        assert INCUMBENT not in _patched_ids(svc)
        assert result['canonical_id'] == INCUMBENT

    @pytest.mark.asyncio
    async def test_the_most_recent_canonical_wins_when_two_exist(self):
        """More than one canonical per (project, topic) is REACHABLE, not
        theoretical: uniqueness ships in warn mode, so duplicates land
        through ordinary writes. The pick is `select_canonical_payload`'s
        total order — most recent, then lowest id — never an arbitrary
        scroll position."""
        older = dict(_scroll_row(M1, canonical=True), created_at='2026-01-01T00:00:00+00:00')
        newer = dict(_scroll_row(M2, canonical=True), created_at='2026-06-01T00:00:00+00:00')
        svc = make_service(scroll_rows=[older, newer])

        result = await call_execute(svc, canonical_content=None, retain_ids=[])

        assert result['canonical_id'] == M2

    @pytest.mark.asyncio
    async def test_a_topic_naming_no_canonical_fails_closed(self):
        """Never an envelope with a null `canonical_id`:
        `build_consolidation_result` types it `str`, and a result claiming
        a canonical that does not exist is the silent-fail-soft this op
        exists to surface."""
        svc = make_service(scroll_rows=[_scroll_row(M1), _scroll_row(M2)])

        result = await call_execute(svc, canonical_content=None)

        assert result['error_type'] == 'TagOnlyIncumbentNotFound'
        assert result['topic'] == TOPIC
        assert 'canonical_id' not in result

    @pytest.mark.asyncio
    async def test_an_unreadable_closure_fails_closed_too(self):
        """A scroll that did not ANSWER cannot show the topic has no
        canonical — the same fail-closed posture the peer check takes."""
        svc = make_service(scroll_error=TimeoutError('qdrant timeout'))

        result = await call_execute(svc, canonical_content=None)

        assert result['error_type'] == 'TagOnlyIncumbentNotFound'
        assert result['topic'] == TOPIC

    @pytest.mark.asyncio
    async def test_a_topic_naming_no_canonical_touches_nothing(self):
        """No peer is read or patched on behalf of a fold with no canonical."""
        svc = make_service(scroll_rows=[_scroll_row(M1), _scroll_row(M2)])

        result = await call_execute(svc, canonical_content=None)

        assert result['error_type'] == 'TagOnlyIncumbentNotFound'
        svc.update_memory.assert_not_awaited()
        svc.get_memory_by_id.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_an_unreadable_canonical_touches_nothing(self):
        svc = make_service(scroll_error=TimeoutError('qdrant timeout'))

        result = await call_execute(svc, canonical_content=None)

        assert result['error_type'] == 'TagOnlyIncumbentNotFound'
        svc.update_memory.assert_not_awaited()
        svc.get_memory_by_id.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_closure_that_fails_after_the_tags_still_reports_the_fold(self):
        """The fold landed, so it is reported in full — the same degradation
        the mint path takes in
        `test_an_unreadable_scroll_degrades_to_not_available`."""
        svc = make_service(
            members=[INCUMBENT, M1, M2],
            canonical_peers=[INCUMBENT],
            closure_scroll_error=TimeoutError('qdrant timeout'),
        )

        result = await call_execute(svc, canonical_content=None)

        assert 'error_type' not in result
        assert result['canonical_id'] == INCUMBENT
        assert result['retained'] == [M1, M2]
        assert result['topic_members_available'] is False

    @pytest.mark.asyncio
    async def test_the_incumbent_is_found_when_the_closure_listing_is_capped(self):
        """The capped closure listing can never contain this incumbent, so it
        must be resolved by a read that is not subject to that cap."""
        crowd = [
            _scroll_row(f'{i:08d}-0000-4000-8000-000000000000')
            for i in range(TOPIC_MEMBER_LIMIT)
        ]
        svc = make_service(
            scroll_rows=[*crowd, _scroll_row(INCUMBENT, canonical=True)],
            topic_total=TOPIC_MEMBER_LIMIT + 1,
        )

        result = await call_execute(svc, canonical_content=None, retain_ids=[])

        assert result.get('canonical_id') == INCUMBENT


class TestAuthorizationIsFailClosedAndPreWrite:
    """The arm authorizes ITSELF, above every read and every write.

    The tool in front of it already runs the same gate, so on that path it
    runs twice — which costs nothing measurable, the resolver being pure,
    synchronous and three `getattr` hops. The duplication is the point: the
    auto-consolidation executor calls this function with NO tool boundary
    in front of it, and a service-level write primitive that trusts its
    caller to have authorized is exactly the shape that leaks.

    An unauthorized caller is turned away before anything is done on its
    behalf and before it learns anything about the system.
    """

    @pytest.mark.asyncio
    async def test_a_caller_off_the_bar_is_refused(self):
        svc = make_service()

        result = await call_execute(svc, agent_id='stranger-session')

        assert result['error_type'] == 'Mem0UpdateNotAuthorized'
        assert result['agent_id'] == 'stranger-session'

    @pytest.mark.asyncio
    async def test_a_refusal_reads_nothing_and_writes_nothing(self):
        svc = make_service()

        await call_execute(svc, agent_id='stranger-session')

        svc.add_memory.assert_not_awaited()
        svc.update_memory.assert_not_awaited()
        svc.get_memory_by_id.assert_not_awaited()
        svc.get_memories_by_metadata.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_kill_switch_outranks_the_allowlist(self):
        """One knob an operator can reliably use to stop an in-flight
        incident, regardless of who is calling."""
        svc = make_service()
        svc.config.mem0_update.enabled = False

        result = await call_execute(svc)

        assert result['error_type'] == 'Mem0UpdateToolDisabled'
        svc.add_memory.assert_not_awaited()
        svc.update_memory.assert_not_awaited()


class TestLiveRead:
    """PRD C4: the arm must hold the LIVE `memory_service`, never a config
    captured at call entry or at construction.

    `mem0_update.*` is green-tier — hot-reloadable with no restart — and
    `reload_config` delivers that by mutating the SHARED config object in
    place. A helper that bound `memory_service.config`, or any leaf of it,
    to a local would make all five leaves restart-only in disguise while
    every other test in this file still passed.

    Mirrors `tests/server/test_update_memory_authz_gate.py::TestLiveRead`,
    reproduced at the new boundary rather than assumed to be covered by
    the old one.
    """

    @pytest.mark.asyncio
    async def test_a_prefix_added_in_place_takes_effect_on_the_next_call(self):
        svc = make_service()
        refused = await call_execute(svc, agent_id='stranger-session')
        assert refused['error_type'] == 'Mem0UpdateNotAuthorized'

        # Exactly what reload_config does: mutate the shared object.
        svc.config.mem0_update.metadata_patch_allowed_agent_prefixes.append(
            'stranger-'
        )

        allowed = await call_execute(svc, agent_id='stranger-session')

        assert allowed['status'] == 'consolidated'
        assert 'error_type' not in allowed


class TestTheMintBypassesTheToolLevelGuards:
    """The canonical is written through `MemoryService.add_memory`, which
    never meets the near-duplicate and topic-cluster guards — those are
    reached only from the `add_memory` TOOL body.

    That bypass is correct BY CONSTRUCTION: a canonical is near its peers
    by definition, being the claim they all make, and a topic under
    consolidation is exactly the "known-contradictory cluster" shape the
    topic guard bounces. Routing the mint through the tool would make the
    ratified index canonical unwritable.

    The pin is what stops a later edit quietly reintroducing the tool path
    and bouncing canonicals with
    `ProceduralKnowledgeKnownTopicClusterWriteRejected`: the canonical
    reaches the store as one await on the service method, carrying the
    content asked for.
    """

    @pytest.mark.asyncio
    async def test_the_canonical_is_written_through_the_service_method(self):
        svc = make_service()
        near_duplicate_of_its_peers = f'record {M1}'

        await call_execute(
            svc,
            canonical_content=near_duplicate_of_its_peers,
            category='procedural_knowledge',
        )

        svc.add_memory.assert_awaited_once()
        assert (
            svc.add_memory.await_args.kwargs['content']
            == near_duplicate_of_its_peers
        )

    @pytest.mark.asyncio
    async def test_the_mint_asks_for_no_near_duplicate_override(self):
        """`metadata={'allow_near_duplicate': True}` is the escape hatch a
        caller of the TOOL uses. This arm does not need one and must not
        learn to set one: an override travels into the record's durable
        metadata, and it would start claiming that a guard which never ran
        was deliberately waived."""
        svc = make_service()

        await call_execute(svc)

        assert 'allow_near_duplicate' not in svc.add_memory.await_args.kwargs
        assert 'allow_near_duplicate' not in (
            svc.add_memory.await_args.kwargs['metadata'] or {}
        )
