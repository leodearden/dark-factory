"""The `consolidate_memories` MCP tool (task 3133).

Stage-1's write-then-delete choreography is a RATCHET: a guard-exempt
canonical write plus unordered deletes with no verification nets +1 entry
per failed pass (17 of the 89 entries the reify curator deleted were the
consolidator's own prior canonicals and scaffolding). The cure is not a
better prompt — it is one op whose closure is CORROBORATED by a live
re-read, never inferred from "the delete call returned ok".

Harness per `tests/test_delete_memory_citation_guard.py`:
`create_mcp_server(mock_service)` + `_tool_manager.call_tool` +
`_parse_result` unwrapping FastMCP `TextContent`. The mem0_update leaves
are REAL `Mem0UpdateConfig` values rather than bare Mocks — a bare
`AsyncMock` makes every leaf a Mock, which the fail-closed authz resolver
rejects, so every case here would otherwise pass for the wrong reason
(`Mem0UpdateNotAuthorized` instead of the property under test). That
failure mode is recorded in `tests/test_update_memory_tool.py`.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import fused_memory.server.tools as tools_module
from fused_memory.config.schema import Mem0UpdateConfig
from fused_memory.models.memory import AddMemoryResponse
from fused_memory.server.tools import create_mcp_server
from fused_memory.services.memory_service import DescendantScan

PROJECT_ID = 'dark_factory'
# On the default allowlist for both mem0_update arms, so no case here can
# fail for an authorization reason and be misread as a result.
AGENT = 'recon-stage-memory_consolidator'
RUN_ID = 'run-abc'
TOPIC = 'memory-consolidation'
CONTENT = 'Consolidation folds a duplicate cluster into one canonical claim.'

# The cluster: three superseded duplicates, folded into one new canonical.
S1 = '11111111-1111-4111-8111-111111111111'
S2 = '22222222-2222-4222-8222-222222222222'
S3 = '33333333-3333-4333-8333-333333333333'
SUPERSEDES = [S1, S2, S3]
CANONICAL = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'

RETAIN_1 = 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb'
RETAIN_2 = 'cccccccc-cccc-4ccc-8ccc-cccccccccccc'

# Two direct children of S1. The orphan rule (PRD V3): they are re-homed
# onto the canonical BEFORE their parent is deleted, never after.
C1 = 'dddddddd-dddd-4ddd-8ddd-dddddddddddd'
C2 = 'eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee'

# The citation gate only runs for a mem0 record in a REGISTERED project with a
# task DB to scan (`_citation_gate_applies`), so these two are what switch it on.
PROJECT_ROOT = '/tmp/root'
KNOWN_PROJECTS = {PROJECT_ID: PROJECT_ROOT}


def _parse_result(result):
    """Parse a FastMCP call_tool result (list of TextContent) into a dict."""
    if isinstance(result, list):
        content = result[0].text if hasattr(result[0], 'text') else str(result[0])
        return json.loads(content)
    return result


# The two reads this op issues return DIFFERENT shapes, and one mock helper
# standing in for both is how a live bug stayed invisible. Verified on this
# tree:
#
#   get_memory_by_id      -> {'id', 'content', 'metadata': <FULL raw payload>}
#                            (memory_service.py: "metadata is the FULL
#                            unprocessed Qdrant payload") — NO top-level
#                            `created_at`; the timestamp is a payload key.
#   get_memories_by_metadata -> {'id', 'created_at', 'metadata'} — because
#                            `Mem0Backend.scroll_by_metadata` explicitly LIFTS
#                            it (`'created_at': payload.get('created_at')`),
#                            which `get_point_by_id` does not do.
#
# So they are kept as two helpers, deliberately named apart, so a later edit
# cannot re-cross them and re-hide the same class of defect.
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
    gone=SUPERSEDES,
    topic_members=None,
    topic_total=None,
    children=None,
    truncated_scans=(),
    stuck_children=(),
    delete_errors=None,
    update_errors=None,
    update_raises=None,
    canonical_peers=(),
    pre_read_errors=None,
    corroboration_errors=None,
    scroll_error=None,
    child_scan_errors=None,
    recount_errors=None,
    events=None,
):
    """A MemoryService mock modelling one consolidation cluster.

    *gone* is the set the post-delete re-read finds ABSENT — the ids that
    are genuinely closed. Anything in `SUPERSEDES` but not in *gone* still
    resolves, i.e. it is a survivor of a delete that claimed success.

    *children* maps a supersede id to its direct children; *delete_errors*
    maps a supersede id to an exception `delete_memory` raises for it.
    *update_errors* maps a memory id to the structured rejection
    `update_memory` RETURNS for it — that primitive reports
    `MemoryNotFound` as a return value rather than a raise, so a caller
    that only guarded against exceptions would read a refusal as a success.
    *update_raises* is the OTHER half of that primitive's contract and maps
    a memory id to an exception `update_memory` RAISES for it: only
    `MemoryNotFound` comes back structured, while every backend failure goes
    through `_journaled_backend_call`, which logs and re-raises. A caller
    that only handled the returned shape would let one timeout escape and
    flatten the whole envelope.

    *canonical_peers* is the set of ids whose payload already carries
    `canonical: True`. The retain patch is a server-side payload MERGE, so
    such a peer KEEPS that key and, once tagged with the cluster's topic,
    becomes a second canonical for it — and nothing downstream catches it,
    since `_apply_canonical_uniqueness` runs only on the `add_memory` path.

    *truncated_scans* is the set of parents whose child listing comes back
    `truncated=True`: a listing the op could not fully SEE is a set it
    cannot prove it moved. *stuck_children* are children whose reparent
    patch REPORTS success while the child stays put — the silent-fail-soft
    case that only a live re-count can catch.

    The READ-FAILURE knobs model the one thing every Qdrant read here does
    by documented contract: a timeout PROPAGATES rather than collapsing into
    a `None`/empty answer, so the caller can tell "genuinely absent" from
    "backend did not answer". They are split by PHASE because the op reads
    each supersede twice and the two reads mean different things:
    *pre_read_errors* fails the pre-delete victim capture (tombstone
    ENRICHMENT), *corroboration_errors* fails the post-delete re-read
    (closure PROOF). *scroll_error* fails the topic-closure listing,
    *child_scan_errors* fails `list_child_ids` for a parent, and
    *recount_errors* fails the live post-reparent `{'parent_id': ...}`
    re-count for a parent.

    *events* is a shared ordered log — the only way to assert the ORDERING
    that is this op's whole contract, since the steps land on different
    mocks (the service, then the task interceptor, then the service again).
    """
    gone = set(gone)
    children = children or {}
    truncated_scans = set(truncated_scans)
    stuck_children = set(stuck_children)
    delete_errors = delete_errors or {}
    update_errors = update_errors or {}
    update_raises = update_raises or {}
    canonical_peers = set(canonical_peers)
    pre_read_errors = pre_read_errors or {}
    corroboration_errors = corroboration_errors or {}
    child_scan_errors = child_scan_errors or {}
    recount_errors = recount_errors or {}
    members = SUPERSEDES if topic_members is None else topic_members
    log = events if events is not None else []

    svc = AsyncMock()
    svc.config.mem0_update = Mem0UpdateConfig()

    async def _add(**kwargs):
        log.append(('add_memory', None))
        return AddMemoryResponse(memory_ids=[CANONICAL], message='ok')

    svc.add_memory = AsyncMock(side_effect=_add)

    async def _delete(**kwargs):
        mid = kwargs.get('memory_id')
        log.append(('delete_memory', mid))
        if mid in delete_errors:
            raise delete_errors[mid]
        return {'status': 'deleted', 'store': 'mem0', 'id': mid}

    svc.delete_memory = AsyncMock(side_effect=_delete)

    # The op reads each supersede TWICE: once before its delete, to capture
    # the metadata/created_at a tombstone needs while they still exist, and
    # once after the delete loop to corroborate closure. `gone` describes
    # the SECOND answer — the record was there, then it was not — so a
    # first read always resolves and later reads apply `gone`.
    reads_seen: dict[str, int] = {}

    async def _get(project_id=None, memory_id=None, **_):
        # The double asserts its own contract: every read the op issues names
        # a concrete id. A read with no id would otherwise key this ledger on
        # `None` and quietly answer a question about the wrong record.
        assert isinstance(memory_id, str), 'the op must name the id it reads'
        log.append(('read', memory_id))
        seen = reads_seen.get(memory_id, 0)
        reads_seen[memory_id] = seen + 1
        # A timeout PROPAGATES out of this primitive by documented contract,
        # which is why the phase matters: the first read is enrichment, every
        # later one is the closure proof.
        phase_errors = corroboration_errors if seen else pre_read_errors
        if memory_id in phase_errors:
            raise phase_errors[memory_id]
        if seen and memory_id in gone:
            return None
        if memory_id in canonical_peers:
            return _point_row(memory_id, canonical=True)
        return _point_row(memory_id)

    svc.get_memory_by_id = AsyncMock(side_effect=_get)

    async def _list_child_ids(memory_id, *, project_id):
        log.append(('list_children', memory_id))
        if memory_id in child_scan_errors:
            raise child_scan_errors[memory_id]
        return DescendantScan(
            ids=list(children.get(memory_id, ())),
            truncated=memory_id in truncated_scans,
        )

    svc.list_child_ids = AsyncMock(side_effect=_list_child_ids)

    # Children that have actually been re-homed onto the canonical. The
    # live post-reparent re-count reads THIS, not the patch return values —
    # modelling the world's answer separately from the call's is what lets
    # a test express "every patch reported success and a child stayed put".
    moved: set[str] = set()

    async def _update(memory_id=None, **kwargs):
        # Same contract as `_get`: a patch with no id must fail loudly here
        # rather than register an un-named record as successfully re-homed.
        assert isinstance(memory_id, str), 'the op must name the id it patches'
        log.append(('update_memory', memory_id))
        if memory_id in update_raises:
            raise update_raises[memory_id]
        if memory_id in update_errors:
            return update_errors[memory_id]
        if (kwargs.get('metadata_patch') or {}).get('parent_id') and (
            memory_id not in stuck_children
        ):
            moved.add(memory_id)
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
        return [_scroll_row(m) for m in members]

    svc.get_memories_by_metadata = AsyncMock(side_effect=_scroll)

    async def _count(project_id=None, filters=None, **_):
        filters = filters or {}
        if 'parent_id' in filters:
            # The live child re-count: how many of this parent's children
            # are STILL its children right now.
            parent = filters['parent_id']
            if parent in recount_errors:
                raise recount_errors[parent]
            kids = children.get(parent, ())
            return len([k for k in kids if k not in moved])
        return len(members) if topic_total is None else topic_total

    svc.count_memories_by_metadata = AsyncMock(side_effect=_count)
    return svc


def _task(task_id, status, metadata):
    return {
        'id': task_id,
        'status': status,
        'title': f'task {task_id}',
        'metadata': metadata,
    }


def make_interceptor(tasks, *, scan_error=None, failing_task_ids=(), events=None):
    """A TaskInterceptor mock for the citation gate (per the 3108 harness).

    *scan_error* makes the task-DB read raise, which the gate must fail
    CLOSED on. *failing_task_ids* makes the repoint WRITE be refused for
    those tasks — interceptor gates refuse by RETURNING
    ``{'success': False}``, never by raising, so a truthy-dict check would
    let a failed repoint through.
    """
    failing = set(failing_task_ids)
    log = events if events is not None else []

    async def _get_tasks(project_root):
        if scan_error is not None:
            raise scan_error
        return {'tasks': list(tasks)}

    async def _update_task(task_id=None, **kwargs):
        log.append(('repoint', task_id))
        if task_id in failing:
            return {
                'success': False,
                'error': 'write refused',
                'error_type': 'ReconTerminalWriteRejected',
            }
        return {'success': True}

    interceptor = MagicMock()
    interceptor.get_tasks = AsyncMock(side_effect=_get_tasks)
    interceptor.update_task = AsyncMock(side_effect=_update_task)
    return interceptor


async def call_consolidate(
    svc, *, task_interceptor=None, known_projects=None, **overrides
):
    server = create_mcp_server(
        svc, task_interceptor=task_interceptor, known_projects=known_projects
    )
    args = {
        'canonical_content': CONTENT,
        'topic': TOPIC,
        'project_id': PROJECT_ID,
        'supersedes': list(SUPERSEDES),
        'run_id': RUN_ID,
        'agent_id': AGENT,
    }
    args.update(overrides)
    return _parse_result(
        await server._tool_manager.call_tool('consolidate_memories', args)
    )


class TestToolRegistration:
    def test_tool_is_registered(self):
        server = create_mcp_server(make_service())

        tools = server._tool_manager.list_tools()

        assert 'consolidate_memories' in {t.name for t in tools}


class TestHappyPath:
    """One canonical written, three supersedes folded, closure corroborated."""

    @pytest.mark.asyncio
    async def test_canonical_is_written_once_with_the_consolidation_metadata(self):
        svc = make_service()

        await call_consolidate(svc)

        svc.add_memory.assert_awaited_once()
        meta = svc.add_memory.await_args.kwargs['metadata']
        assert meta['topic'] == TOPIC
        assert meta['canonical'] is True
        assert list(meta['supersedes']) == SUPERSEDES
        assert svc.add_memory.await_args.kwargs['content'] == CONTENT

    @pytest.mark.asyncio
    async def test_every_supersede_is_deleted_from_mem0(self):
        svc = make_service()

        await call_consolidate(svc)

        deleted = [c.kwargs['memory_id'] for c in svc.delete_memory.await_args_list]
        assert deleted == SUPERSEDES
        assert {c.kwargs['store'] for c in svc.delete_memory.await_args_list} == {'mem0'}

    @pytest.mark.asyncio
    async def test_result_envelope(self):
        svc = make_service()

        result = await call_consolidate(svc)

        assert result['status'] == 'consolidated'
        assert result['canonical_id'] == CANONICAL
        assert result['topic'] == TOPIC
        assert result['deleted'] == SUPERSEDES
        assert result['survivors'] == []
        assert result['failed_deletes'] == []

    @pytest.mark.asyncio
    async def test_topic_closure_is_listed_from_the_deterministic_scroll(self):
        svc = make_service(topic_members=[CANONICAL, RETAIN_1])

        result = await call_consolidate(svc)

        assert [m['id'] for m in result['topic_members']] == [CANONICAL, RETAIN_1]
        assert result['topic_members_truncated'] is False
        call = svc.get_memories_by_metadata.await_args
        assert call.kwargs['filters'] == {'topic': TOPIC}
        assert call.kwargs['project_id'] == PROJECT_ID

    @pytest.mark.asyncio
    async def test_the_closure_listing_never_goes_through_search(self):
        """THE pinned negative.

        Semantic search is what made the original incident unfixable: run
        live, the re-derive query returned only superseded cluster members,
        routing dispatch back into the contradictory advice consolidation
        existed to collapse. A consolidation that reports its own result via
        a top-N ranked read can silently omit the record it just wrote.
        """
        svc = make_service()

        await call_consolidate(svc)

        svc.search.assert_not_called()


class TestValidationIsRefusedBeforeAnyWrite:
    """The validator's refusals reach the wire, and cost zero writes."""

    @pytest.mark.asyncio
    async def test_malformed_supersede_is_refused_and_nothing_is_written(self):
        svc = make_service()

        result = await call_consolidate(svc, supersedes=[S1, '873889a1'])

        assert result['error_type'] == 'ValidationError'
        assert '873889a1' in result['error']
        svc.add_memory.assert_not_awaited()
        svc.delete_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_hint_survives_the_tool_boundary(self):
        """Returned, never raised.

        `@mcp_tool_errors` flattens an exception to {'error', 'error_type'},
        which would drop the hint — the part that tells the caller what to
        do about it. PRESENCE is the contract; the hint's wording is
        documentation and lives in `consolidation.py`, fixed once.
        """
        svc = make_service()

        result = await call_consolidate(svc, topic='memory_consolidation')

        assert result.get('hint')

    @pytest.mark.asyncio
    async def test_delete_arm_without_run_id_is_refused_before_the_canonical(self):
        svc = make_service()

        result = await call_consolidate(svc, run_id=None)

        assert result['error_type'] == 'ValidationError'
        assert 'run_id' in result['error']
        svc.add_memory.assert_not_awaited()


class TestAuthorizationIsFailClosedAndPreWrite:
    """The gate is unconditional and precedes every write.

    Deliberately NOT conditioned on a non-empty `retain`: reparenting is
    only discovered after the canonical exists, so a gate that waited for
    the first metadata patch would deny mid-transaction — with a canonical
    already written and a cluster half-folded.
    """

    @pytest.mark.asyncio
    async def test_unauthorized_agent_is_denied_before_the_canonical_write(self):
        svc = make_service()

        result = await call_consolidate(svc, agent_id='claude-interactive')

        assert result['error_type'] == 'Mem0UpdateNotAuthorized'
        assert result['agent_id'] == 'claude-interactive'
        svc.add_memory.assert_not_called()
        svc.delete_memory.assert_not_called()
        svc.update_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_denial_fires_for_a_supersedes_only_call(self):
        """Proves the gate is unconditional, not keyed on the retain list."""
        svc = make_service()

        result = await call_consolidate(svc, agent_id='claude-interactive', retain=[])

        assert result['error_type'] == 'Mem0UpdateNotAuthorized'
        svc.add_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_kill_switch_denies_every_agent(self):
        svc = make_service()
        svc.config.mem0_update = Mem0UpdateConfig(enabled=False)

        result = await call_consolidate(svc)

        assert result['error_type'] == 'Mem0UpdateToolDisabled'
        svc.add_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_the_content_amend_arm_is_never_requested(self):
        """This op stamps metadata and writes new records — it amends nothing.

        Widening ONLY the content-amend allowlist must therefore not
        authorize the caller. Requesting an arm the op does not use would
        make the wider metadata bar a back door into a silent-rewrite
        primitive.
        """
        svc = make_service()
        svc.config.mem0_update = Mem0UpdateConfig(
            content_amend_allowed_agent_prefixes=['claude-interactive'],
            metadata_patch_allowed_agent_prefixes=['recon-stage-'],
        )

        result = await call_consolidate(svc, agent_id='claude-interactive')

        assert result['error_type'] == 'Mem0UpdateNotAuthorized'
        svc.add_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_an_authorized_recon_agent_proceeds(self):
        svc = make_service()
        svc.config.mem0_update = Mem0UpdateConfig(
            metadata_patch_allowed_agent_prefixes=['recon-stage-']
        )

        result = await call_consolidate(svc)

        assert result['status'] == 'consolidated'
        svc.add_memory.assert_awaited_once()


class TestCanonicalFirstOrdering:
    """A canonical that did not land is NEVER followed by deletions.

    The ordering IS the anti-ratchet property. Deleting first and writing
    after would net-LOSE content on a failed write — unrecoverably, since
    there is no read path to a deleted point. Writing first and deleting
    after can only ever net-ADD, which is reportable and re-runnable.
    """

    @pytest.mark.asyncio
    async def test_canonical_uniqueness_violation_destroys_nothing(self):
        from fused_memory.memory_metadata import CanonicalUniquenessViolation

        svc = make_service()
        svc.add_memory = AsyncMock(
            side_effect=CanonicalUniquenessViolation(
                project_id=PROJECT_ID, topic=TOPIC, incumbent_id=RETAIN_1
            )
        )

        result = await call_consolidate(svc)

        assert result['error_type'] == 'CanonicalUniquenessViolation'
        assert RETAIN_1 in result['error']
        svc.delete_memory.assert_not_called()
        svc.update_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_metadata_rejection_destroys_nothing(self):
        from fused_memory.memory_metadata import (
            MemoryMetadataValidationError,
            MetadataViolation,
        )

        svc = make_service()
        svc.add_memory = AsyncMock(
            side_effect=MemoryMetadataValidationError([
                MetadataViolation(
                    key='topic',
                    code='invalid_topic_slug',
                    message='topic is malformed',
                    fatal=True,
                )
            ])
        )

        result = await call_consolidate(svc)

        assert result['error_type'] == 'MemoryMetadataValidationError'
        svc.delete_memory.assert_not_called()
        svc.update_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_canonical_that_returned_no_id_refuses_the_whole_op(self):
        """The silent-fail-soft case: the write did not raise, and landed nothing.

        Proceeding here would delete a cluster in favour of a canonical that
        does not exist — exactly the net-loss the ordering exists to make
        impossible, arrived at by trusting a success that was not one.
        """
        svc = make_service()
        svc.add_memory = AsyncMock(return_value=AddMemoryResponse(memory_ids=[]))

        result = await call_consolidate(svc)

        assert result['error_type'] == 'CanonicalWriteFailed'
        svc.delete_memory.assert_not_called()
        svc.update_memory.assert_not_called()


class TestTheDeleteSetGoesThroughTheCitationGate:
    """The supersedes go THROUGH the existing gate, never under it.

    A consolidate op that called `MemoryService.delete_memory` directly
    would gate nothing and destroy N cited records while reporting success
    — the one-guarded-record-and-N-unguarded-ones failure task 3197
    documents. The gate lives at the tool layer because it needs the task
    interceptor and the known-projects registry, so this op reuses the
    same closures rather than growing a second scanner (INV-5).

    A load-bearing asymmetry versus `delete_memory`: this op ALWAYS has a
    concrete surviving replacement — the canonical it is about to write —
    so the gate's "you named no survivor" refusal is structurally
    unreachable here. What IS reachable is a scan the op could not
    complete, and an unknown citation state immediately before an
    irreversible delete must fail closed.
    """

    @pytest.mark.asyncio
    async def test_an_unreadable_task_db_refuses_the_whole_op(self):
        """(a) FAIL CLOSED, and before the canonical exists.

        `assert_not_called` on all three write paths is the proof that the
        pre-flight is genuinely non-mutating AND that it precedes the
        canonical write: a refused consolidation must leave the corpus
        byte-identical, not stranded with a canonical over an unfolded
        cluster.
        """
        svc = make_service()
        interceptor = make_interceptor([], scan_error=RuntimeError('task DB unreadable'))

        result = await call_consolidate(
            svc, task_interceptor=interceptor, known_projects=KNOWN_PROJECTS
        )

        assert result['error_type'] == 'ConsolidationCitationGateRejected'
        assert [b['error_type'] for b in result['blocked']] == ['CitationScanFailed'] * 3
        svc.add_memory.assert_not_called()
        svc.delete_memory.assert_not_called()
        svc.update_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_every_blocker_is_collected_into_one_refusal(self):
        """N unclearable ids cost ONE refusal, not N refuse-fix-retry trips.

        A consolidation set is a cluster by construction, so refusing at
        the first blocker would leave the caller to re-derive the rest by
        hand — the step that found 3 of 8 in the original incident.
        """
        svc = make_service()
        interceptor = make_interceptor([], scan_error=RuntimeError('task DB unreadable'))

        result = await call_consolidate(
            svc, task_interceptor=interceptor, known_projects=KNOWN_PROJECTS
        )

        assert [b['memory_id'] for b in result['blocked']] == SUPERSEDES

    @pytest.mark.asyncio
    async def test_live_citers_are_repointed_at_the_new_canonical(self):
        """(b) The canonical is written FIRST and is the repoint target.

        It satisfies the gate's "the replacement must not be a record this
        call destroys" rule by construction: it is the one record the op
        creates.
        """
        svc = make_service()
        interceptor = make_interceptor([
            _task('501', 'pending', {'mem0_canonical_entry': S1}),
            _task('502', 'pending', {'mem0_canonical_entry': S2}),
        ])

        result = await call_consolidate(
            svc, task_interceptor=interceptor, known_projects=KNOWN_PROJECTS
        )

        assert result['status'] == 'consolidated'
        patched = [
            json.loads(c.kwargs['metadata'])['mem0_canonical_entry']
            for c in interceptor.update_task.await_args_list
        ]
        assert patched == [CANONICAL, CANONICAL]

    @pytest.mark.asyncio
    async def test_the_repoint_stats_reach_the_caller_per_id(self):
        svc = make_service()
        interceptor = make_interceptor([
            _task('501', 'pending', {'mem0_canonical_entry': S1}),
        ])

        result = await call_consolidate(
            svc, task_interceptor=interceptor, known_projects=KNOWN_PROJECTS
        )

        entry = result['citation_repoint'][S1]
        assert entry['outcome'] == 'repointed'
        assert entry['citation_repoint']['stage1_citation_tasks_repointed'] == 1
        assert entry['citation_repoint']['stage1_citation_repoint_failures'] == 0

    @pytest.mark.asyncio
    async def test_citations_are_repointed_before_any_delete_lands(self):
        """No window exists in which the entry is gone but a task still cites it."""
        events: list[tuple[str, str | None]] = []
        svc = make_service(events=events)
        interceptor = make_interceptor(
            [_task('501', 'pending', {'mem0_canonical_entry': S3})], events=events
        )

        await call_consolidate(
            svc, task_interceptor=interceptor, known_projects=KNOWN_PROJECTS
        )

        kinds = [kind for kind, _ in events]
        assert kinds.index('add_memory') < kinds.index('repoint')
        assert kinds.index('repoint') < kinds.index('delete_memory')

    @pytest.mark.asyncio
    async def test_a_failed_repoint_suppresses_only_its_own_delete(self):
        """(c) One unrepointable citer costs ONE supersede, not the cluster.

        Deleting S2 now would strand exactly the pointer the gate exists to
        protect, so its delete is refused — while S1 and S3, whose citers
        were rewritten, still fold. The refused id is reported twice on
        purpose: in `failed_deletes` (what did not happen, and why) and in
        `survivors` (what is still in the corpus).
        """
        svc = make_service(gone=[S1, S3])
        interceptor = make_interceptor(
            [
                _task('501', 'pending', {'mem0_canonical_entry': S1}),
                _task('502', 'pending', {'mem0_canonical_entry': S2}),
                _task('503', 'pending', {'mem0_canonical_entry': S3}),
            ],
            failing_task_ids={'502'},
        )

        result = await call_consolidate(
            svc, task_interceptor=interceptor, known_projects=KNOWN_PROJECTS
        )

        assert result['status'] == 'partial'
        assert [f['id'] for f in result['failed_deletes']] == [S2]
        assert result['failed_deletes'][0]['error_type'] == 'CitationRepointFailed'
        assert result['deleted'] == [S1, S3]
        assert S2 in result['survivors']
        attempted = [c.kwargs['memory_id'] for c in svc.delete_memory.await_args_list]
        assert attempted == [S1, S3]

    @pytest.mark.asyncio
    async def test_an_unscannable_project_is_never_enumerated(self):
        """(d) The gate's own precondition short-circuits first.

        With no registered project there is no task DB to scan, so the op
        must not pay for an enumeration it would have to discard — and must
        not refuse on behalf of a gate that is not running.
        """
        svc = make_service()
        interceptor = make_interceptor([
            _task('501', 'pending', {'mem0_canonical_entry': S1}),
        ])

        result = await call_consolidate(
            svc, task_interceptor=interceptor, known_projects={}
        )

        assert result['status'] == 'consolidated'
        assert result['deleted'] == SUPERSEDES
        interceptor.get_tasks.assert_not_called()


def _delete_failures():
    """The realistic ways `delete_memory` refuses, as (label, exception).

    All three are RETURNED to the caller as data, not raised: the caller
    needs to know which ids are still in the corpus, and `@mcp_tool_errors`
    would flatten the whole envelope to {'error', 'error_type'} and destroy
    exactly that.
    """
    from fused_memory.memory_metadata import ParentHasChildrenError
    from fused_memory.utils.validation import InputValidationError

    return [
        pytest.param(RuntimeError('mem0 backend unreachable'), 'RuntimeError',
                     id='backend-error'),
        pytest.param(InputValidationError('memory_id is not a UUID'),
                     'InputValidationError', id='validation-error'),
        pytest.param(
            ParentHasChildrenError(parent_id=S2, child_ids=[RETAIN_1]),
            'ParentHasChildrenError', id='parent-has-children',
        ),
    ]


class TestPartialFailureIsTwoWay:
    """One failed delete costs one id, and is reported from BOTH sides.

    The signal this op exists to produce: `failed_deletes` says what the
    call could not do, `survivors` says what is still in the corpus, and
    the same id appearing in both is what tells a caller the cluster is
    still open. A raise would collapse all of it — `@mcp_tool_errors`
    flattens an exception to {'error', 'error_type'}, so the per-id
    dispositions, and with them the whole deliverable, would be gone.
    """

    @pytest.mark.parametrize('exc,error_type', _delete_failures())
    @pytest.mark.asyncio
    async def test_a_failed_delete_is_captured_structurally(self, exc, error_type):
        svc = make_service(gone=[S1, S3], delete_errors={S2: exc})

        result = await call_consolidate(svc)

        assert result['status'] == 'partial'
        assert result['failed_deletes'] == [
            {'id': S2, 'error': str(exc), 'error_type': error_type}
        ]

    @pytest.mark.asyncio
    async def test_the_same_id_is_named_on_both_sides(self):
        svc = make_service(gone=[S1, S3], delete_errors={S2: RuntimeError('boom')})

        result = await call_consolidate(svc)

        assert [f['id'] for f in result['failed_deletes']] == [S2]
        assert result['survivors'] == [S2]
        assert result['deleted'] == [S1, S3]

    @pytest.mark.asyncio
    async def test_one_failure_does_not_abort_the_rest_of_the_cluster(self):
        """S3 comes AFTER the failing id, so reaching it proves the loop ran on."""
        svc = make_service(gone=[S1, S3], delete_errors={S2: RuntimeError('boom')})

        await call_consolidate(svc)

        attempted = [c.kwargs['memory_id'] for c in svc.delete_memory.await_args_list]
        assert attempted == SUPERSEDES

    @pytest.mark.asyncio
    async def test_the_exception_never_propagates_out_of_the_tool(self):
        svc = make_service(gone=[S1, S3], delete_errors={S2: RuntimeError('boom')})

        result = await call_consolidate(svc)

        # Not the flattened {'error', 'error_type'} shape a raise would give.
        assert 'error' not in result
        assert result['canonical_id'] == CANONICAL


class TestClosureIsCorroboratedNeverClaimed:
    """`survivors` comes from the WORLD, not from what the deletes said.

    Both asymmetric cases are the point. A delete that reports
    {'status': 'deleted'} over an id that still resolves IS a survivor —
    that silent-fail-soft is what the ratchet was made of. An id whose
    delete raised but which no longer resolves is NOT one: reporting it as
    a leftover would send an operator chasing a record that is gone.
    """

    @pytest.mark.asyncio
    async def test_a_claimed_delete_that_did_not_take_is_a_survivor(self):
        """(a) The call succeeded and the world disagrees. The world wins."""
        svc = make_service(gone=[S1, S3])

        result = await call_consolidate(svc)

        assert result['survivors'] == [S2]
        assert result['status'] == 'partial'
        # NOT a failure: nothing refused, so there is nothing to report as
        # one. The disagreement is what `survivors` is for.
        assert result['failed_deletes'] == []

    @pytest.mark.asyncio
    async def test_a_raised_delete_over_an_absent_id_is_not_a_survivor(self):
        """(b) It refused, but the record is gone anyway.

        Listing it as a leftover would send an operator chasing nothing —
        the failure is real and reported, the survival is not.
        """
        svc = make_service(gone=SUPERSEDES, delete_errors={S2: RuntimeError('boom')})

        result = await call_consolidate(svc)

        assert [f['id'] for f in result['failed_deletes']] == [S2]
        assert result['survivors'] == []

    @pytest.mark.asyncio
    async def test_every_supersede_is_re_read_once_after_the_last_delete(self):
        """(c) The re-read is what produces `survivors` — nothing else can.

        Sliced at the LAST delete: a read taken before then answers a
        question about a record the op had not finished acting on.
        """
        events: list[tuple[str, str | None]] = []
        svc = make_service(events=events)

        await call_consolidate(svc)

        last_delete = max(
            i for i, (kind, _) in enumerate(events) if kind == 'delete_memory'
        )
        # An un-named read is dropped rather than compared, and the equality
        # below is what makes that loud: a missing id shortens the list.
        after = [
            mid
            for kind, mid in events[last_delete + 1:]
            if kind == 'read' and mid is not None
        ]
        assert sorted(after) == sorted(SUPERSEDES)

    @pytest.mark.asyncio
    async def test_a_truncated_closure_listing_says_so(self):
        """(d) A capped listing must never read as the whole closure."""
        from fused_memory.server.tools import _TOPIC_MEMBER_LIMIT

        svc = make_service(
            topic_members=[f'member-{i}' for i in range(_TOPIC_MEMBER_LIMIT)],
            topic_total=512,
        )

        result = await call_consolidate(svc)

        assert result['topic_members_truncated'] is True
        assert result['topic_members_total'] == 512
        assert len(result['topic_members']) == _TOPIC_MEMBER_LIMIT

    @pytest.mark.asyncio
    async def test_the_common_path_pays_for_no_count(self):
        """The extra round-trip is owed only when the cap was actually hit."""
        svc = make_service(topic_members=[CANONICAL, RETAIN_1])

        result = await call_consolidate(svc)

        assert result['topic_members_truncated'] is False
        assert result['topic_members_total'] == 2
        svc.count_memories_by_metadata.assert_not_called()


class TestRetainAndTagArm:
    """The RATIFIED default arm (gate 3200): peers are TAGGED, not deleted.

    Option C's write shape is short retained single-claim peers sharing
    `metadata.topic` with exactly one `canonical: true` per (project,
    topic). Retaining is what preserves the Qdrant POINT ID, so every
    citation, parent pointer and supersedes edge already aimed at a peer
    stays valid across the consolidation. Deleting-and-rewriting peers is
    what made the previous choreography lose content it had no path to
    restore — there is no write path to a deleted point id.
    """

    @pytest.mark.asyncio
    async def test_each_retained_peer_is_tagged_in_place(self):
        svc = make_service()

        await call_consolidate(svc, retain=[RETAIN_1, RETAIN_2])

        patched = {
            c.kwargs['memory_id']: c.kwargs for c in svc.update_memory.await_args_list
        }
        assert set(patched) == {RETAIN_1, RETAIN_2}
        for kwargs in patched.values():
            assert kwargs['metadata_patch'] == {'topic': TOPIC}
            # `content=None` is load-bearing, not incidental: passing the
            # text back would re-embed a record whose claim did not change,
            # moving its vector for no reason. A retained peer keeps its
            # text, its vector AND its id.
            assert kwargs.get('content') is None
            # Merge, and nothing deleted — the peer's own metadata (its
            # source, its run_id, its parent) is not this op's to discard.
            assert kwargs.get('metadata_mode') == 'merge'
            assert kwargs.get('metadata_delete_keys') is None

    @pytest.mark.asyncio
    async def test_no_retained_peer_is_stamped_canonical(self):
        """Exactly ONE `canonical: true` per (project, topic) — the invariant
        `CanonicalUniquenessViolation` exists to defend. A retained peer that
        also claimed canonical would make the cluster unresolvable and would
        make the NEXT consolidation of this topic refuse outright."""
        svc = make_service()

        await call_consolidate(svc, retain=[RETAIN_1, RETAIN_2])

        for call in svc.update_memory.await_args_list:
            assert 'canonical' not in (call.kwargs['metadata_patch'] or {})
            # Peers, not children: the retain arm never re-homes anything.
            assert 'parent_id' not in (call.kwargs['metadata_patch'] or {})

    @pytest.mark.asyncio
    async def test_a_peer_that_is_already_canonical_is_refused(self):
        """NOT SETTING `canonical` IS NOT THE SAME AS ENSURING IT IS UNSET.

        The patch is a server-side payload MERGE, so a peer already holding
        `canonical: True` keeps it and, paired with the cluster's topic,
        becomes a SECOND canonical for (project, T). Nothing downstream
        catches that: `_apply_canonical_uniqueness` is reached only from the
        `add_memory` path, and `update_memory` deliberately does not run
        `_apply_memory_metadata_validation`. The duplicate would surface one
        pass later, as a failed canonical write on the NEXT consolidation of
        this topic.

        Refused, not silently demoted: a prior canonical in the retain list
        is usually an authoring mistake — that record is what belonged in
        `supersedes`.
        """
        svc = make_service(canonical_peers={RETAIN_2})

        result = await call_consolidate(svc, retain=[RETAIN_1, RETAIN_2])

        [failure] = result['retain_failures']
        assert failure['id'] == RETAIN_2
        assert failure['error_type'] == 'RetainedPeerIsCanonical'
        assert result['status'] == 'partial'
        # THE PROPERTY: no patch was ever issued for it, so its payload is
        # untouched and no second canonical exists.
        patched = {c.kwargs['memory_id'] for c in svc.update_memory.await_args_list}
        assert RETAIN_2 not in patched
        # One refusal costs one peer, never the arm.
        assert result['retained'] == [RETAIN_1]
        assert RETAIN_1 in patched

    @pytest.mark.asyncio
    async def test_an_unreadable_peer_refuses_its_tag(self):
        """The check FAILS CLOSED, the same posture as the child listing and
        for the same reason: a check that did not ANSWER is not a check that
        said "not canonical". Refusing costs one tag, which a caller can
        retry; tagging on an unproven check mints a duplicate canonical that
        nothing downstream will catch."""
        svc = make_service(
            pre_read_errors={RETAIN_2: TimeoutError('qdrant read timed out')}
        )

        result = await call_consolidate(svc, retain=[RETAIN_1, RETAIN_2])

        [failure] = result['retain_failures']
        assert failure['id'] == RETAIN_2
        assert failure['error_type'] == 'RetainCheckFailed'
        assert result['retained'] == [RETAIN_1]
        assert result['status'] == 'partial'
        patched = {c.kwargs['memory_id'] for c in svc.update_memory.await_args_list}
        assert RETAIN_2 not in patched
        # The envelope survives and the delete arm still ran: a read failure
        # degrades one peer's claim, never the whole result.
        assert result['deleted'] == list(SUPERSEDES)

    @pytest.mark.asyncio
    async def test_retained_ids_are_echoed_unchanged(self):
        """Echoing the ids back UNCHANGED is the point-id-preservation proof:
        a caller can compare them to what it sent and see that tagging moved
        no record."""
        svc = make_service()

        result = await call_consolidate(svc, retain=[RETAIN_1, RETAIN_2])

        assert result['retained'] == [RETAIN_1, RETAIN_2]
        assert result['retain_failures'] == []
        assert result['status'] == 'consolidated'

    @pytest.mark.asyncio
    async def test_a_retained_peer_is_never_deleted(self):
        svc = make_service()

        await call_consolidate(svc, retain=[RETAIN_1, RETAIN_2])

        deleted = {c.kwargs['memory_id'] for c in svc.delete_memory.await_args_list}
        assert deleted == set(SUPERSEDES)
        assert RETAIN_1 not in deleted
        assert RETAIN_2 not in deleted

    @pytest.mark.asyncio
    async def test_the_canonical_supersedes_only_the_deleted_ids(self):
        """`metadata.supersedes` is a claim about what this canonical
        REPLACED. A retained peer was not replaced — it is still in the
        corpus stating its own claim — so listing it there would tell every
        future reader to prefer the canonical over a record that is still
        live and still correct."""
        svc = make_service()

        await call_consolidate(svc, retain=[RETAIN_1, RETAIN_2])

        meta = svc.add_memory.await_args.kwargs['metadata']
        assert list(meta['supersedes']) == SUPERSEDES
        assert RETAIN_1 not in meta['supersedes']
        assert RETAIN_2 not in meta['supersedes']

    @pytest.mark.asyncio
    async def test_a_retain_only_call_deletes_nothing(self):
        """The ratified DEFAULT shape: tag the cluster, delete none of it.
        It needs no `run_id` because it never deletes and so never has a
        deletion to attribute."""
        svc = make_service()

        result = await call_consolidate(
            svc, supersedes=[], run_id=None, retain=[RETAIN_1, RETAIN_2]
        )

        assert result['status'] == 'consolidated'
        assert result['deleted'] == []
        assert result['survivors'] == []
        assert result['retained'] == [RETAIN_1, RETAIN_2]
        svc.delete_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_failed_tag_is_captured_and_does_not_abort_the_arm(self):
        """`update_memory` reports `MemoryNotFound` by RETURNING a structured
        rejection, never by raising — so a caller checking only for
        exceptions would record a peer as tagged when it was refused."""
        svc = make_service(
            update_errors={
                RETAIN_2: {
                    'error': f'memory {RETAIN_2} not found in project {PROJECT_ID}',
                    'error_type': 'MemoryNotFound',
                }
            }
        )

        result = await call_consolidate(svc, retain=[RETAIN_1, RETAIN_2])

        assert result['retain_failures'] == [
            {
                'id': RETAIN_2,
                'error': f'memory {RETAIN_2} not found in project {PROJECT_ID}',
                'error_type': 'MemoryNotFound',
            }
        ]
        # Only the peer that actually took is claimed as retained.
        assert result['retained'] == [RETAIN_1]
        assert result['status'] == 'partial'
        # ...and the peer that COULD be tagged still was: one refusal costs
        # one peer, not the arm.
        tagged = {c.kwargs['memory_id'] for c in svc.update_memory.await_args_list}
        assert RETAIN_1 in tagged

    @pytest.mark.asyncio
    async def test_a_raising_tag_is_captured_and_does_not_flatten_the_envelope(self):
        """The OTHER half of `update_memory`'s contract. Only
        `MemoryNotFound` comes back as a structured return; every backend
        failure goes through `_journaled_backend_call`, which logs and
        RE-RAISES. An unguarded await would let one Qdrant timeout escape to
        `@mcp_tool_errors`, which flattens the whole result to
        {'error', 'error_type'} — destroying the per-id dispositions this op
        exists to produce, and skipping the delete arm entirely."""
        svc = make_service(
            update_raises={RETAIN_2: TimeoutError('qdrant set_payload timed out')}
        )

        result = await call_consolidate(svc, retain=[RETAIN_1, RETAIN_2])

        # The envelope SURVIVES: a raise is one peer's disposition, in the
        # same shape as a returned rejection, not the end of the call.
        assert result['retain_failures'] == [
            {
                'id': RETAIN_2,
                'error': 'qdrant set_payload timed out',
                'error_type': 'TimeoutError',
            }
        ]
        assert result['retained'] == [RETAIN_1]
        assert result['status'] == 'partial'
        # The delete arm still ran to completion — the whole reason the raise
        # must not propagate.
        assert result['deleted'] == list(SUPERSEDES)
        assert result['survivors'] == []


class TestChildrenAreReHomedBeforeTheirParentDies:
    """The orphan rule (PRD V3): reparent, THEN delete — never the reverse.

    A supersede's children encode a hierarchy the records themselves
    assert. Deleting the parent first would strand them pointing at an id
    that no longer resolves, and nothing could repair that afterwards: the
    children still name the dead parent, and there is no read path to a
    deleted point to learn what it was. So the pointer moves while both
    ends still exist.

    DIRECT children only. Re-pointing the top layer preserves the deeper
    subtree's internal structure, whereas flattening every descendant onto
    the canonical would destroy the hierarchy the records encode.
    """

    @pytest.mark.asyncio
    async def test_each_supersede_is_scanned_for_direct_children(self):
        svc = make_service(children={S1: [C1, C2]})

        await call_consolidate(svc)

        scanned = [c.args[0] for c in svc.list_child_ids.await_args_list]
        assert scanned == SUPERSEDES
        for call in svc.list_child_ids.await_args_list:
            assert call.kwargs['project_id'] == PROJECT_ID

    @pytest.mark.asyncio
    async def test_children_are_re_pointed_at_the_new_canonical(self):
        svc = make_service(children={S1: [C1, C2]})

        await call_consolidate(svc)

        patched = {
            c.kwargs['memory_id']: c.kwargs for c in svc.update_memory.await_args_list
        }
        assert set(patched) == {C1, C2}
        for kwargs in patched.values():
            assert kwargs['metadata_patch'] == {'parent_id': CANONICAL}
            # The child's own claim is untouched — only its parent pointer
            # moves, so no re-embed and no other metadata is disturbed.
            assert kwargs.get('content') is None
            assert kwargs.get('metadata_mode') == 'merge'

    @pytest.mark.asyncio
    async def test_every_patch_lands_before_the_parent_is_deleted(self):
        """THE ordering pin. A child re-pointed after its parent died was
        an orphan for the interval in between, and a crash inside that
        interval leaves it one permanently."""
        events = []
        svc = make_service(children={S1: [C1, C2]}, events=events)

        await call_consolidate(svc)

        assert ('update_memory', C1) in events
        assert ('update_memory', C2) in events
        parent_died = events.index(('delete_memory', S1))
        assert events.index(('update_memory', C1)) < parent_died
        assert events.index(('update_memory', C2)) < parent_died
        # ...and the scan that found them came before the patches.
        assert events.index(('list_children', S1)) < events.index(
            ('update_memory', C1)
        )

    @pytest.mark.asyncio
    async def test_the_reparenting_is_reported_per_child(self):
        svc = make_service(children={S1: [C1, C2]})

        result = await call_consolidate(svc)

        assert result['reparented'] == [
            {'child_id': C1, 'from': S1, 'to': CANONICAL},
            {'child_id': C2, 'from': S1, 'to': CANONICAL},
        ]
        assert result['reparent_failures'] == []
        # The re-homing did not cost the fold: S1 still died.
        assert result['status'] == 'consolidated'
        assert result['deleted'] == SUPERSEDES

    @pytest.mark.asyncio
    async def test_a_childless_supersede_costs_no_patch_and_no_recount(self):
        """The common path stays cheap. `list_child_ids` short-circuits on
        its own cheap count (pinned in `test_memory_service.py`), and a
        parent the scan found childless has nothing that could have failed
        to move — so re-counting would re-ask the same question in the same
        instant."""
        svc = make_service()

        result = await call_consolidate(svc)

        assert result['reparented'] == []
        assert result['reparent_failures'] == []
        svc.update_memory.assert_not_awaited()
        svc.count_memories_by_metadata.assert_not_awaited()


class TestAnIncompleteReparentRefusesTheDelete:
    """The op must not become the caller that routes around the orphan gate.

    `ParentHasChildrenError` only means something while every caller
    respects it. This op could trivially have forced its way past — it
    holds `cascade=True` — and deliberately does not: cascading would
    DESTROY the children the re-homing exists to preserve, turning a
    refusal into permanent data loss.

    So a delete happens only over a re-homing this call can PROVE
    complete. Failing that proof leaves the supersede alive and named,
    which is a recoverable outcome; deleting on an unproven re-homing is
    not recoverable at all.
    """

    @pytest.mark.asyncio
    async def test_a_child_that_did_not_move_saves_its_parent(self):
        svc = make_service(
            gone=[S2, S3],
            children={S1: [C1, C2]},
            update_errors={
                C2: {'error': 'memory not found', 'error_type': 'MemoryNotFound'}
            },
        )

        result = await call_consolidate(svc)

        attempted = [c.kwargs['memory_id'] for c in svc.delete_memory.await_args_list]
        assert S1 not in attempted
        [failure] = result['failed_deletes']
        assert failure['id'] == S1
        assert failure['error_type'] == 'ReparentIncomplete'
        # The stranded child is NAMED: an operator needs to know which
        # pointer to fix, not merely that one exists.
        assert C2 in failure['error']
        assert result['survivors'] == [S1]
        assert result['status'] == 'partial'

    @pytest.mark.asyncio
    async def test_the_rest_of_the_cluster_still_folds(self):
        """One unprovable re-homing costs one supersede, not the cluster."""
        svc = make_service(
            gone=[S2, S3],
            children={S1: [C1, C2]},
            update_errors={
                C2: {'error': 'memory not found', 'error_type': 'MemoryNotFound'}
            },
        )

        result = await call_consolidate(svc)

        assert result['deleted'] == [S2, S3]
        # C1 moved and stays reported as moved — a refusal to delete the
        # parent does not un-say the re-homing that did land.
        assert result['reparented'] == [
            {'child_id': C1, 'from': S1, 'to': CANONICAL}
        ]
        assert [f['child_id'] for f in result['reparent_failures']] == [C2]

    @pytest.mark.asyncio
    async def test_a_raising_child_patch_refuses_only_that_parent(self):
        """A RAISE IS THE SAME EVENT AS A RETURNED REJECTION: either way the
        child was not re-homed, so it strands and saves its parent.

        Letting it propagate would be strictly worse than any per-id failure
        — this loop runs INTERLEAVED with the deletes, so the raise would
        flatten the envelope and destroy the dispositions of supersedes
        ALREADY IRREVERSIBLY DELETED, and skip their tombstone write. Those
        records would be gone with neither a disposition nor a tombstone:
        indistinguishable from silent data loss, which is the exact hole the
        delete arm's tombstone contract exists to close.
        """
        svc = make_service(
            gone=[S2, S3],
            children={S1: [C1, C2]},
            update_raises={C2: TimeoutError('qdrant set_payload timed out')},
        )

        with patched_tombstone_writer() as writer:
            result = await call_consolidate(svc)

        # Only S1 is refused, and it is refused for the RIGHT reason: the
        # blocker is its stranded child, not the raw timeout.
        [failure] = result['failed_deletes']
        assert failure['id'] == S1
        assert failure['error_type'] == 'ReparentIncomplete'
        assert C2 in failure['error']
        assert result['survivors'] == [S1]
        # The raise is recorded in the same per-child shape as a rejection.
        assert result['reparent_failures'] == [
            {
                'child_id': C2,
                'from': S1,
                'to': CANONICAL,
                'error': 'qdrant set_payload timed out',
                'error_type': 'TimeoutError',
            }
        ]
        # The rest of the cluster still folds...
        assert result['deleted'] == [S2, S3]
        assert result['reparented'] == [
            {'child_id': C1, 'from': S1, 'to': CANONICAL}
        ]
        # ...and is still TOMBSTONED. This is the assertion the propagating
        # version could not satisfy: a flattened envelope never reaches the
        # tombstone step, leaving confirmed-gone records un-attributable.
        assert writer.await_args is not None
        assert sorted(v['id'] for v in writer.await_args.args[2]) == sorted([S2, S3])
        assert result['tombstones_written'] == 2

    @pytest.mark.asyncio
    async def test_a_truncated_child_listing_refuses_the_delete(self):
        """A listing the op could not fully SEE is a set it cannot prove it
        moved. The count it reports must read as a FLOOR, not as the whole
        set — an exhaustive-sounding number over a truncated scan is the
        overclaim this op exists to eliminate."""
        svc = make_service(
            gone=[S2, S3], children={S1: [C1]}, truncated_scans={S1}
        )

        result = await call_consolidate(svc)

        attempted = [c.kwargs['memory_id'] for c in svc.delete_memory.await_args_list]
        assert S1 not in attempted
        [failure] = result['failed_deletes']
        assert failure['id'] == S1
        assert failure['error_type'] == 'ReparentIncomplete'
        assert result['survivors'] == [S1]

    @pytest.mark.asyncio
    async def test_a_patch_that_reported_success_is_still_corroborated(self):
        """CORROBORATE AFTER ACTING, the same discipline 3197's cascade
        applies to its own child deletes. Every patch here returns
        `{'status': 'updated'}` and one child stays put anyway — trusting
        the return value would delete a parent that still has one."""
        svc = make_service(
            gone=[S2, S3], children={S1: [C1, C2]}, stuck_children={C2}
        )

        result = await call_consolidate(svc)

        attempted = [c.kwargs['memory_id'] for c in svc.delete_memory.await_args_list]
        assert S1 not in attempted
        [failure] = result['failed_deletes']
        assert failure['id'] == S1
        assert failure['error_type'] == 'ReparentIncomplete'
        assert result['survivors'] == [S1]
        assert result['status'] == 'partial'
        # Every patch reported success, so nothing lands in
        # `reparent_failures` — the refusal comes from the WORLD's answer.
        assert result['reparent_failures'] == []
        recounted = [
            c.kwargs['filters']
            for c in svc.count_memories_by_metadata.await_args_list
            if 'parent_id' in (c.kwargs.get('filters') or {})
        ]
        assert {'parent_id': S1} in recounted

    @pytest.mark.asyncio
    async def test_the_delete_is_never_forced_with_cascade(self):
        """THE pinned negative. `cascade=True` would delete the children
        instead of the refusal — converting a reported, recoverable refusal
        into exactly the silent orphaning (here: silent destruction) that
        PRD V3 forbids."""
        svc = make_service(gone=[S2, S3], children={S1: [C1, C2]})

        await call_consolidate(svc)

        for call in svc.delete_memory.await_args_list:
            assert call.kwargs.get('cascade') in (None, False)


@contextmanager
def patched_tombstone_writer(returns=None):
    """Patch the batch tombstone writer where `server/tools.py` imported it."""
    writer = AsyncMock(
        side_effect=(lambda *a, **k: len(a[2])) if returns is None
        else (lambda *a, **k: returns)
    )
    with patch.object(tools_module, 'record_mem0_deletion_tombstones', writer):
        yield writer


class TestTheDeleteArmStampsATombstone:
    """Every reaped supersede gets a task-3041 tombstone. IRREVERSIBLE means
    ATTRIBUTABLE.

    The motivating evidence is concrete: a consolidation survivor carried
    the forward pointer (`consolidated_from=<victim>`), but probing the
    DEAD id returned `{'found': false}` with the `tombstone` key ABSENT.
    Per `get_memory_by_id`'s own omitted-not-null contract, that absence is
    what proves a deletion was NOT deliberate — so from the victim id
    alone, which is all an auditor chasing a broken reference holds, that
    deletion was indistinguishable from silent data loss. The victim is
    permanently unfixable (no write path reaches a deleted point id), so
    the gap is only preventable going forward, which is why it belongs in
    this op's contract.
    """

    @pytest.mark.asyncio
    async def test_each_victim_is_read_before_its_delete(self):
        """There is no read path to a deleted point id, so the tombstone's
        metadata and created_at must be captured while the record still
        exists — the same reason `_sweep_stale_mem0_pool` keeps full member
        dicts rather than bare ids."""
        events: list[tuple[str, str | None]] = []
        svc = make_service(events=events)

        with patched_tombstone_writer():
            await call_consolidate(svc)

        for sid in SUPERSEDES:
            assert events.index(('read', sid)) < events.index(('delete_memory', sid))

    @pytest.mark.asyncio
    async def test_one_batch_call_carries_the_whole_cluster(self):
        """The BATCH form, as both prod sweeps use: one upsert_many, one
        commit, one fsync, all-or-nothing. A consolidation set is a cluster
        by construction, so it is exactly what the batch form is for."""
        svc = make_service()

        with patched_tombstone_writer() as writer:
            await call_consolidate(svc)

        writer.assert_awaited_once()
        assert writer.await_args is not None
        args, kwargs = writer.await_args
        assert args[1] == PROJECT_ID
        assert kwargs['deleter'] == 'consolidate_memories'
        # The run that KILLED these records, not the run that wrote them.
        assert kwargs['deleting_run_id'] == RUN_ID
        # The REVERSE pointer: where the content went.
        assert kwargs['absorbed_by'] == CANONICAL
        assert [v['id'] for v in args[2]] == SUPERSEDES
        for victim in args[2]:
            # Pinned end-to-end against the REAL read shape: `metadata` is the
            # full payload the point read returned, and `created_at` is the key
            # nested INSIDE it — not a top-level field, which is what
            # `get_memory_by_id` never returns. Recording `None` here would
            # silently drop one of the three fields of the very audit row this
            # change exists to make answerable.
            assert victim['metadata'] == {'topic': TOPIC, 'created_at': CREATED_AT}
            assert victim['created_at'] == CREATED_AT

    @pytest.mark.asyncio
    async def test_the_delete_is_attributed_to_the_same_source(self):
        """`_source` and `deleter` are the same string — the idiom both
        sweeps follow, so the write journal and the tombstone name the same
        actor and an auditor can join them."""
        svc = make_service()

        with patched_tombstone_writer():
            await call_consolidate(svc)

        for call in svc.delete_memory.await_args_list:
            assert call.kwargs['_source'] == 'consolidate_memories'
            assert call.kwargs['causation_id'] == RUN_ID

    @pytest.mark.asyncio
    async def test_only_confirmed_gone_ids_are_tombstoned(self):
        """THE property. A tombstone must never claim a record that is
        still alive: it is a durable 30-day audit row asserting the record
        is gone, so minting one over a live record manufactures exactly the
        false attribution the mechanism exists to prevent.

        S1 is confirmed gone. S2's delete RAISED. S3's delete reported
        success and S3 still resolves. Only S1 is a victim.
        """
        svc = make_service(
            gone=[S1], delete_errors={S2: RuntimeError('backend unreachable')}
        )

        with patched_tombstone_writer() as writer:
            result = await call_consolidate(svc)

        assert writer.await_args is not None
        [victim] = writer.await_args.args[2]
        assert victim['id'] == S1
        assert [f['id'] for f in result['failed_deletes']] == [S2]
        assert sorted(result['survivors']) == sorted([S2, S3])

    @pytest.mark.asyncio
    async def test_the_stamp_follows_the_corroborating_re_read(self):
        """Stamped from the world's answer, not the call's. This op holds
        strictly better evidence than the two sweeps — a deterministic
        post-delete re-read is already mandatory here — so "a tombstone
        exists" means "corroborated gone" rather than "the backend did not
        raise"."""
        events: list[tuple[str, str | None]] = []
        svc = make_service(events=events)

        async def _record(*args, **kwargs):
            events.append(('tombstone', None))
            return len(args[2])

        with patch.object(
            tools_module,
            'record_mem0_deletion_tombstones',
            AsyncMock(side_effect=_record),
        ):
            await call_consolidate(svc)

        stamped = events.index(('tombstone', None))
        last_read = max(i for i, (kind, _) in enumerate(events) if kind == 'read')
        assert stamped > last_read

    @pytest.mark.asyncio
    async def test_the_written_count_reaches_the_caller(self):
        svc = make_service()

        with patched_tombstone_writer():
            result = await call_consolidate(svc)

        assert result['tombstones_written'] == 3

    @pytest.mark.asyncio
    async def test_the_retain_arm_never_tombstones(self):
        """SCOPE BOUNDARY, pinned as a negative. Retained peers are never
        deleted, so nothing about them becomes un-attributable and there is
        no dead id to probe. A later refactor must not quietly add deletion
        bookkeeping to a path that deletes nothing."""
        svc = make_service()

        with patched_tombstone_writer() as writer:
            result = await call_consolidate(
                svc, supersedes=[], run_id=None, retain=[RETAIN_1, RETAIN_2]
            )

        writer.assert_not_called()
        assert result['retained'] == [RETAIN_1, RETAIN_2]


class TestATombstoneShortfallIsDisclosed:
    """The audit trail can fail without the consolidation having failed.

    Both are true at once and the envelope must say both: the records ARE
    gone (structured-facts), and the rows explaining why they are gone did
    NOT land (no-silent-fail-soft). Inferring the second from the first is
    exactly what a caller would do if the counts were absent.
    """

    @pytest.mark.asyncio
    async def test_a_writer_that_wrote_nothing_is_visible_in_the_envelope(self):
        """The fail-safe no-`recon_ledger` path — a real deployment running
        `recon_ledger_enabled=False`. It returns 0 without raising, so
        without the counters a caller would read a clean consolidation
        envelope and conclude the audit trail exists."""
        svc = make_service()

        with patched_tombstone_writer(returns=0):
            result = await call_consolidate(svc)

        assert result['tombstones_written'] == 0
        assert result['tombstones_expected'] == 3

    @pytest.mark.asyncio
    async def test_a_raising_writer_never_costs_the_envelope(self):
        """The writer is fail-safe by contract, so this should be
        unreachable — and is guarded anyway, because `@mcp_tool_errors`
        would flatten the result to {'error', 'error_type'} and destroy the
        per-id dispositions that ARE the deliverable. Losing the survivor
        list to a failed audit write would be a strictly worse trade than
        the gap it is reporting."""
        svc = make_service()
        writer = AsyncMock(side_effect=RuntimeError('ledger disk full'))

        with patch.object(tools_module, 'record_mem0_deletion_tombstones', writer):
            result = await call_consolidate(svc)

        assert result['canonical_id'] == CANONICAL
        assert result['deleted'] == SUPERSEDES
        assert result['survivors'] == []
        assert result['tombstones_written'] == 0
        assert result['tombstones_expected'] == 3

    @pytest.mark.asyncio
    async def test_a_shortfall_alone_does_not_make_the_op_partial(self):
        """`status` answers "is this cluster still open?", and it is not.
        Conflating an audit-trail gap with a consolidation failure would
        invite a caller to RETRY a completed merge — re-writing a canonical
        whose supersedes are already gone, which is precisely the
        +1-per-pass ratchet this op exists to end."""
        svc = make_service()

        with patched_tombstone_writer(returns=0):
            result = await call_consolidate(svc)

        assert result['status'] == 'consolidated'
        assert result['failed_deletes'] == []
        assert result['survivors'] == []

    @pytest.mark.asyncio
    async def test_the_happy_path_reports_a_complete_trail(self):
        svc = make_service()

        with patched_tombstone_writer():
            result = await call_consolidate(svc)

        assert result['tombstones_written'] == result['tombstones_expected'] == 3

    @pytest.mark.asyncio
    async def test_expected_counts_only_the_confirmed_gone(self):
        """`tombstones_expected` is a claim about what SHOULD have been
        stamped, so it must exclude the ids no tombstone was ever owed
        for — otherwise a correct run over a partial cluster would report a
        phantom shortfall."""
        svc = make_service(
            gone=[S1], delete_errors={S2: RuntimeError('backend unreachable')}
        )

        with patched_tombstone_writer() as writer:
            result = await call_consolidate(svc)

        assert result['tombstones_expected'] == 1
        assert result['tombstones_written'] == 1
        assert writer.await_args is not None
        assert len(writer.await_args.args[2]) == 1


class TestReadFailuresDoNotDestroyTheEnvelope:
    """A READ that fails after deletes have landed must never flatten the
    result.

    Premise, verified on this tree: `get_memory_by_id`,
    `get_memories_by_metadata`, `count_memories_by_metadata` and the
    `list_child_ids` primitives all PROPAGATE `TimeoutError` by documented
    contract — deliberately, so a caller can distinguish "genuinely absent"
    from "the backend did not answer". An unguarded call site therefore lets
    that exception escape the tool body, where `@mcp_tool_errors` flattens
    the whole envelope to {'error', 'error_type': 'TimeoutError'} —
    destroying `deleted`/`failed_deletes`/`survivors`/`reparented` for
    records that are ALREADY irreversibly gone, and skipping the tombstone
    step so nothing is attributable either. The reads are where this op
    LEARNS things; a learning failure must degrade what it can claim, never
    erase what it did.

    The two halves are asymmetric on purpose:

      an ENRICHMENT read (victim capture, closure listing) degrades — its
          failure costs detail, and detail must never veto a fold
      a PROOF read (child listing, post-reparent re-count) FAILS CLOSED —
          an unreadable answer is not "no children", and reading it as one
          would orphan exactly the records the op exists to preserve
    """

    @pytest.mark.asyncio
    async def test_a_failed_corroboration_read_is_named_not_swallowed(self):
        """An id whose closure could not be CHECKED is neither a survivor
        (never observed alive) nor confirmed gone (never proven dead). It
        gets its own list, because collapsing it into either one asserts
        something no read supports."""
        svc = make_service(
            corroboration_errors={S2: TimeoutError('qdrant read timed out')}
        )

        with patched_tombstone_writer() as writer:
            result = await call_consolidate(svc)

        assert result.get('error_type') != 'TimeoutError'
        assert result['canonical_id'] == CANONICAL
        assert result['deleted'] == SUPERSEDES
        assert result['survivor_check_failed'] == [
            {
                'id': S2,
                'error': 'qdrant read timed out',
                'error_type': 'TimeoutError',
            }
        ]
        assert S2 not in result['survivors']
        # NOT tombstoned: a tombstone means CORROBORATED gone, and this id
        # was never corroborated either way.
        assert writer.await_args is not None
        assert [v['id'] for v in writer.await_args.args[2]] == [S1, S3]
        assert result['tombstones_expected'] == 2
        # The op could not prove closure, which IS the deliverable — unlike a
        # tombstone shortfall, that is genuinely open business.
        assert result['status'] == 'partial'

    @pytest.mark.asyncio
    async def test_a_failed_closure_scroll_degrades_and_says_so(self):
        """An empty `topic_members` on a scroll that never answered would
        read as "this topic has no members" — the overclaim this op exists
        to eliminate. The listing is disclosed as UNAVAILABLE instead."""
        svc = make_service(scroll_error=TimeoutError('scroll timed out'))

        with patched_tombstone_writer() as writer:
            result = await call_consolidate(svc)

        assert result.get('error_type') != 'TimeoutError'
        assert result['canonical_id'] == CANONICAL
        assert result['deleted'] == SUPERSEDES
        assert result['topic_members'] == []
        assert result['topic_members_available'] is False
        assert result['topic_members_truncated'] is False
        # The deletes and their audit trail are unaffected: the listing is a
        # convenience, the closure is the contract.
        writer.assert_awaited_once()
        assert result['tombstones_written'] == 3

    @pytest.mark.asyncio
    async def test_a_failed_victim_read_still_folds_and_still_tombstones(self):
        """Tombstone ENRICHMENT, not tombstone existence. Losing the
        metadata/created_at of one victim costs those two fields; refusing
        the delete over it would cost the consolidation, and refusing the
        tombstone would leave the very dead id this change exists to make
        answerable un-attributable."""
        svc = make_service(
            pre_read_errors={S2: TimeoutError('point read timed out')}
        )

        with patched_tombstone_writer() as writer:
            result = await call_consolidate(svc)

        assert result.get('error_type') != 'TimeoutError'
        assert result['deleted'] == SUPERSEDES
        assert result['status'] == 'consolidated'
        assert writer.await_args is not None
        victims = {v['id']: v for v in writer.await_args.args[2]}
        # Degraded to exactly the shape a genuine not-found already yields.
        assert victims[S2]['metadata'] is None
        assert victims[S2]['created_at'] is None
        # One miss costs one victim's detail, never the batch's.
        assert victims[S1]['created_at'] == CREATED_AT
        assert victims[S3]['metadata'] == {'topic': TOPIC, 'created_at': CREATED_AT}

    @pytest.mark.asyncio
    async def test_an_unreadable_child_listing_refuses_the_delete(self):
        """FAIL CLOSED. A listing that did not answer is not a listing that
        said "no children" — and deleting on that reading would orphan the
        records the reparent pass exists to preserve, permanently, since no
        read path reaches a deleted parent to learn what it was."""
        svc = make_service(
            child_scan_errors={S1: TimeoutError('child scan timed out')}
        )

        with patched_tombstone_writer():
            result = await call_consolidate(svc)

        assert result.get('error_type') != 'TimeoutError'
        deleted_ids = [c.kwargs['memory_id'] for c in svc.delete_memory.await_args_list]
        assert S1 not in deleted_ids
        [failure] = [f for f in result['failed_deletes'] if f['id'] == S1]
        assert failure['error_type'] == 'ChildScanFailed'
        # One unreadable parent costs one fold: the rest of the cluster still
        # closes, because re-running to catch them would re-write the
        # canonical — the +1-per-pass ratchet.
        assert result['deleted'] == [S2, S3]
        assert result['status'] == 'partial'

    @pytest.mark.asyncio
    async def test_an_unreadable_recount_refuses_the_delete(self):
        """The re-count is the CORROBORATION that every reparent patch
        actually landed. A corroboration that could not be read is not a
        corroboration, so the delete stays unearned — the same refusal a
        non-zero count produces, distinguished only by its message."""
        svc = make_service(
            children={S1: [C1, C2]},
            recount_errors={S1: TimeoutError('count timed out')},
        )

        with patched_tombstone_writer():
            result = await call_consolidate(svc)

        assert result.get('error_type') != 'TimeoutError'
        deleted_ids = [c.kwargs['memory_id'] for c in svc.delete_memory.await_args_list]
        assert S1 not in deleted_ids
        [failure] = [f for f in result['failed_deletes'] if f['id'] == S1]
        assert failure['error_type'] == 'ReparentIncomplete'
        assert result['deleted'] == [S2, S3]

    @pytest.mark.asyncio
    async def test_survivor_check_failed_is_always_present(self):
        """Always-present-keys rule: a caller reads
        `result['survivor_check_failed']` without a membership test, and a
        later arm cannot quietly stop reporting by omitting its key."""
        svc = make_service()

        with patched_tombstone_writer():
            result = await call_consolidate(svc)

        assert result['survivor_check_failed'] == []
        assert result['topic_members_available'] is True
