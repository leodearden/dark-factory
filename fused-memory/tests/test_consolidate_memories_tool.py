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
from unittest.mock import AsyncMock, MagicMock

import pytest

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


def _row(memory_id, **metadata):
    return {
        'id': memory_id,
        'content': f'record {memory_id}',
        'created_at': '2026-01-01T00:00:00+00:00',
        'metadata': {'topic': TOPIC, **metadata},
    }


def make_service(
    *,
    gone=SUPERSEDES,
    topic_members=None,
    topic_total=None,
    children=None,
    delete_errors=None,
    events=None,
):
    """A MemoryService mock modelling one consolidation cluster.

    *gone* is the set the post-delete re-read finds ABSENT — the ids that
    are genuinely closed. Anything in `SUPERSEDES` but not in *gone* still
    resolves, i.e. it is a survivor of a delete that claimed success.

    *children* maps a supersede id to its direct children; *delete_errors*
    maps a supersede id to an exception `delete_memory` raises for it.

    *events* is a shared ordered log — the only way to assert the ORDERING
    that is this op's whole contract, since the steps land on different
    mocks (the service, then the task interceptor, then the service again).
    """
    gone = set(gone)
    children = children or {}
    delete_errors = delete_errors or {}
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

    async def _get(project_id=None, memory_id=None, **_):
        log.append(('read', memory_id))
        if memory_id in gone:
            return None
        return _row(memory_id)

    svc.get_memory_by_id = AsyncMock(side_effect=_get)

    async def _list_child_ids(memory_id, *, project_id):
        return DescendantScan(ids=list(children.get(memory_id, ())), truncated=False)

    svc.list_child_ids = AsyncMock(side_effect=_list_child_ids)
    svc.update_memory = AsyncMock(
        return_value={'status': 'updated', 'store': 'mem0', 'metadata_patched': True}
    )
    svc.get_memories_by_metadata = AsyncMock(
        return_value=[_row(m) for m in members]
    )
    svc.count_memories_by_metadata = AsyncMock(
        return_value=len(members) if topic_total is None else topic_total
    )
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
        do about it.
        """
        svc = make_service()

        result = await call_consolidate(svc, topic='memory_consolidation')

        assert 'fused_memory.topic_slug' in result.get('hint', '')

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
        assert 'metadata_patch' in result['error']
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
        assert 'canonical' in result['error'].lower()
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
        after = [mid for kind, mid in events[last_delete + 1:] if kind == 'read']
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
