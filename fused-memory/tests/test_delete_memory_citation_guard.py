"""Tests for delete_memory's pre-delete citation-repoint guard (task 3108).

A recon consolidation delete destroys one of a duplicate cluster in favour of a
survivor. Task metadata that still cites the doomed entry becomes a dangling
pointer the moment the delete lands — and dispatch follows those pointers.

The incident this guard closes had two failure modes:

1. The remediation enumerated citing tasks by hand and found 3 of 8. The 5 it
   missed included the pending/dispatchable ones — the ones that mattered.
2. The "correction" told dispatch to re-derive the canonical entry via
   ``search(query=...)``. Run live, that query returned only superseded cluster
   members, routing dispatch back into the contradictory advice consolidation
   existed to collapse.

So the guard runs BEFORE the irreversible delete: scan every task's metadata
mechanically, repoint live citers to a CONCRETE surviving UUID, and only then
delete. A delete that cannot satisfy that is refused, which closes the
dangling-pointer window rather than moving it.

Harness modelled on ``test_delete_memory_alias.py`` (the real delete_memory
MCP-tool test): ``create_mcp_server`` + ``_tool_manager.call_tool`` +
``_parse_result`` unwrapping FastMCP ``TextContent``.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.server.tools import create_mcp_server

# Doomed duplicate and its surviving replacement — both canonical 36-char UUIDs.
DOOMED = '2531b4d8-1111-4aaa-8bbb-000000000001'
SURVIVOR = '9f3ac071-3333-4eee-8fff-000000000003'

RECON_AGENT = 'recon-stage-memory_consolidator'
PROJECT_ROOT = '/tmp/root'
KNOWN_PROJECTS = {'dark_factory': PROJECT_ROOT}


def _parse_result(result):
    """Parse a FastMCP call_tool result (list of TextContent) into a dict."""
    if isinstance(result, list):
        content = result[0].text if hasattr(result[0], 'text') else str(result[0])
        return json.loads(content)
    return result


def _task(task_id, status, metadata):
    return {'id': task_id, 'status': status, 'title': f'task {task_id}', 'metadata': metadata}


def _make_service():
    """MemoryService whose delete_memory is an EXPLICIT AsyncMock.

    Required by tests/test_check_bare_magicmock_config.py and
    tests/test_check_asyncmock_assertion_style.py — an awaited child must be
    declared, not left as an auto-attribute.
    """
    svc = AsyncMock()
    svc.delete_memory = AsyncMock(
        return_value={'status': 'deleted', 'store': 'mem0', 'id': DOOMED},
    )
    return svc


def _make_interceptor(tasks, update_result=None):
    interceptor = MagicMock()
    interceptor.get_tasks = AsyncMock(return_value={'tasks': tasks})
    interceptor.update_task = AsyncMock(
        return_value=update_result if update_result is not None else {'success': True},
    )
    return interceptor


async def _call_delete(mcp_server, **overrides):
    args = {
        'memory_id': DOOMED,
        'store': 'mem0',
        'project_id': 'dark_factory',
        'agent_id': RECON_AGENT,
    }
    args.update(overrides)
    return _parse_result(
        await mcp_server._tool_manager.call_tool('delete_memory', args),
    )


class TestCitationRepointRequired:
    """A consolidation delete with live citers and no replacement is REFUSED."""

    @pytest.fixture
    def mock_service(self):
        return _make_service()

    @pytest.fixture
    def interceptor(self):
        return _make_interceptor([
            _task('101', 'pending', {'mem0_canonical_entry': DOOMED}),
            _task('102', 'pending', {
                'memory_hints': {'entities': [], 'queries': [f'advice {DOOMED} here']},
            }),
            _task('103', 'pending', {'unrelated': 'nothing to see'}),
        ])

    @pytest.fixture
    def mcp_server(self, mock_service, interceptor):
        return create_mcp_server(
            mock_service,
            task_interceptor=interceptor,
            known_projects=KNOWN_PROJECTS,
        )

    @pytest.mark.asyncio
    async def test_delete_refused_when_live_citers_and_no_replacement(
        self, mcp_server, mock_service,
    ):
        """No replacement id supplied -> the delete never runs."""
        result = await _call_delete(mcp_server)

        assert result['error_type'] == 'CitationRepointRequired'
        # The irreversible operation did NOT happen.
        mock_service.delete_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_refusal_names_the_citing_tasks_and_paths(self, mcp_server):
        """The rejection is actionable: it says exactly which tasks cite the
        doomed entry and where, so the caller is not left to re-derive the
        enumeration by hand — the step that found 3 of 8 in the incident."""
        result = await _call_delete(mcp_server)

        citers = {c['task_id']: c for c in result['citing_tasks']}
        assert set(citers) == {'101', '102'}
        assert citers['101']['paths'] == ['mem0_canonical_entry']
        assert citers['102']['paths'] == ['memory_hints.queries[0]']
        assert result['memory_id'] == DOOMED

    @pytest.mark.asyncio
    async def test_refusal_hint_demands_a_concrete_uuid_and_forbids_search(
        self, mcp_server,
    ):
        """The hint names the fix (a concrete surviving UUID) and explicitly
        rules out the incident's re-derive-via-search instruction."""
        result = await _call_delete(mcp_server)

        hint = result['hint']
        assert 'replacement_memory_id' in hint
        assert 'search(' in hint


class TestRepointThenDelete:
    """With a concrete replacement, citers are repointed and THEN deleted."""

    @pytest.fixture
    def mock_service(self):
        return _make_service()

    @pytest.fixture
    def interceptor(self):
        return _make_interceptor([
            _task('201', 'pending', {'mem0_canonical_entry': DOOMED}),
            _task('202', 'in-progress', {
                'memory_hints': {'entities': [], 'queries': [f'advice {DOOMED}']},
            }),
            _task('203', 'done', {'cited': DOOMED}),
            _task('204', 'pending', {'unrelated': 'nothing'}),
        ])

    @pytest.fixture
    def mcp_server(self, mock_service, interceptor):
        return create_mcp_server(
            mock_service,
            task_interceptor=interceptor,
            known_projects=KNOWN_PROJECTS,
        )

    @pytest.mark.asyncio
    async def test_delete_succeeds_and_reports_repoint_stats(
        self, mcp_server, mock_service,
    ):
        """(a) The delete runs, and its result carries the repoint stats."""
        result = await _call_delete(mcp_server, replacement_memory_id=SURVIVOR)

        assert result['status'] == 'deleted'
        mock_service.delete_memory.assert_awaited_once()

        repoint = result['citation_repoint']
        assert repoint['stage1_citation_tasks_repointed'] == 2
        assert repoint['stage1_citations_repointed'] == 2
        assert repoint['stage1_citation_repoint_failures'] == 0
        assert repoint['stage1_terminal_citations_reported'] == 1

    @pytest.mark.asyncio
    async def test_every_repoint_write_lands_before_the_delete(
        self, mcp_server, mock_service, interceptor,
    ):
        """(b) ORDERING is the guarantee. If the delete could interleave ahead
        of a repoint, the dangling-pointer window would still exist — it would
        just be narrower. Record both call streams and assert the boundary."""
        order: list[str] = []

        async def _record_update(**kwargs):
            order.append(f'repoint:{kwargs["task_id"]}')
            return {'success': True}

        async def _record_delete(**kwargs):
            order.append('delete')
            return {'status': 'deleted', 'store': 'mem0', 'id': DOOMED}

        interceptor.update_task = AsyncMock(side_effect=_record_update)
        mock_service.delete_memory = AsyncMock(side_effect=_record_delete)

        await _call_delete(mcp_server, replacement_memory_id=SURVIVOR)

        assert order.count('delete') == 1
        assert order[-1] == 'delete'
        # Both live citers were repointed strictly before it.
        assert set(order[:-1]) == {'repoint:201', 'repoint:202'}

    @pytest.mark.asyncio
    async def test_terminal_citer_is_reported_not_written(
        self, mcp_server, interceptor,
    ):
        """(c) A done citer is surfaced on the result and never rewritten —
        the terminal dangler is made visible rather than silenced."""
        result = await _call_delete(mcp_server, replacement_memory_id=SURVIVOR)

        terminal = result['citation_repoint']['terminal_citations']
        assert [t['task_id'] for t in terminal] == ['203']
        assert terminal[0]['status'] == 'done'
        assert terminal[0]['paths'] == ['cited']

        written = {c[1]['task_id'] for c in interceptor.update_task.call_args_list}
        assert '203' not in written

    @pytest.mark.asyncio
    async def test_no_citers_deletes_without_any_write(self, mock_service):
        """(d) Nothing cites the id -> the delete proceeds and no task is
        touched. The gate must not manufacture writes."""
        interceptor = _make_interceptor([
            _task('301', 'pending', {'cited': SURVIVOR}),
            _task('302', 'pending', {'unrelated': 'nothing'}),
        ])
        mcp_server = create_mcp_server(
            mock_service,
            task_interceptor=interceptor,
            known_projects=KNOWN_PROJECTS,
        )

        result = await _call_delete(mcp_server, replacement_memory_id=SURVIVOR)

        assert result['status'] == 'deleted'
        mock_service.delete_memory.assert_awaited_once()
        interceptor.update_task.assert_not_awaited()


# The literal string Stage 2 wrote as a "correction" during the incident.
INCIDENT_SEARCH_INSTRUCTION = 're-derive the current canonical entry via search(query=...)'


class TestCitationGateScopingAndFailures:
    """The gate is narrowly scoped, and every failure refuses the delete."""

    @pytest.fixture
    def mock_service(self):
        return _make_service()

    @pytest.fixture
    def interceptor(self):
        return _make_interceptor([_task('401', 'pending', {'cited': DOOMED})])

    @pytest.fixture
    def mcp_server(self, mock_service, interceptor):
        return create_mcp_server(
            mock_service,
            task_interceptor=interceptor,
            known_projects=KNOWN_PROJECTS,
        )

    @pytest.mark.asyncio
    async def test_non_recon_caller_bypasses_the_gate_entirely(
        self, mcp_server, mock_service, interceptor,
    ):
        """(a) An interactive caller pays no extra get_tasks read and is never
        blocked — the gate is scoped to the consolidation path the incident
        actually came from."""
        result = await _call_delete(mcp_server, agent_id='claude-interactive')

        assert result['status'] == 'deleted'
        mock_service.delete_memory.assert_awaited_once()
        interceptor.get_tasks.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_graphiti_store_bypasses_the_gate_entirely(
        self, mcp_server, mock_service, interceptor,
    ):
        """(a) The gate is mem0-scoped, mirroring verify_cited_memories."""
        result = await _call_delete(mcp_server, store='graphiti')

        assert result['status'] == 'deleted'
        mock_service.delete_memory.assert_awaited_once()
        interceptor.get_tasks.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_search_instruction_replacement_is_rejected(
        self, mcp_server, mock_service, interceptor,
    ):
        """(b) The incident's re-derive-via-search "correction" is refused
        mechanically. Running that query live returned only superseded cluster
        members, so accepting it would preserve the dangling pointer under a
        new name instead of closing it."""
        result = await _call_delete(
            mcp_server, replacement_memory_id=INCIDENT_SEARCH_INSTRUCTION,
        )

        assert result['error_type'] == 'CitationReplacementInvalid'
        mock_service.delete_memory.assert_not_awaited()
        # Nothing was rewritten to the bogus pointer either.
        interceptor.update_task.assert_not_awaited()
        assert 'search(' in result['hint']

    @pytest.mark.asyncio
    async def test_truncated_uuid_replacement_is_rejected(
        self, mcp_server, mock_service,
    ):
        """(b) An 8-char prefix is not a forwarding pointer — the same
        truncated-UUID hazard prompts/stage1.py already warns about."""
        result = await _call_delete(mcp_server, replacement_memory_id=SURVIVOR[:8])

        assert result['error_type'] == 'CitationReplacementInvalid'
        mock_service.delete_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_rejected_repoint_write_blocks_the_delete(self, mock_service):
        """(c) Delete only AFTER the repoint actually landed. Interceptor gates
        refuse by RETURNING {'success': False}, never by raising, so a
        truthy-dict check would have let this through."""
        interceptor = _make_interceptor(
            [_task('501', 'pending', {'cited': DOOMED})],
            update_result={
                'success': False,
                'error': 'write refused',
                'error_type': 'ReconTerminalWriteRejected',
            },
        )
        mcp_server = create_mcp_server(
            mock_service,
            task_interceptor=interceptor,
            known_projects=KNOWN_PROJECTS,
        )

        result = await _call_delete(mcp_server, replacement_memory_id=SURVIVOR)

        assert result['error_type'] == 'CitationRepointFailed'
        mock_service.delete_memory.assert_not_awaited()
        assert [u['task_id'] for u in result['unrepointed']] == ['501']

    @pytest.mark.asyncio
    async def test_scan_failure_fails_closed(self, mock_service):
        """(d) FAIL CLOSED. An unreadable task DB means 'unknown', and unknown
        must not be treated as 'no citations' when the next step destroys data
        irreversibly. A refused delete is retried next cycle; a silently
        permitted one manufactures the L2."""
        interceptor = _make_interceptor([])
        interceptor.get_tasks = AsyncMock(side_effect=RuntimeError('task db unreachable'))
        mcp_server = create_mcp_server(
            mock_service,
            task_interceptor=interceptor,
            known_projects=KNOWN_PROJECTS,
        )

        result = await _call_delete(mcp_server, replacement_memory_id=SURVIVOR)

        assert result['error_type'] == 'CitationScanFailed'
        mock_service.delete_memory.assert_not_awaited()
        assert result['memory_id'] == DOOMED

    @pytest.mark.asyncio
    async def test_no_task_interceptor_leaves_behaviour_unchanged(self, mock_service):
        """(e) The baseline test_delete_memory_alias.py construction — no
        interceptor, no known_projects registry — must keep working exactly as
        before, gate or no gate."""
        mcp_server = create_mcp_server(mock_service)

        result = await _call_delete(mcp_server, replacement_memory_id=SURVIVOR)

        assert result['status'] == 'deleted'
        mock_service.delete_memory.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_project_absent_from_registry_leaves_behaviour_unchanged(
        self, mock_service,
    ):
        """(e) An interceptor wired but the project unregistered: no live task
        DB to scan, so the pre-existing behaviour is preserved."""
        interceptor = _make_interceptor([_task('601', 'pending', {'cited': DOOMED})])
        mcp_server = create_mcp_server(
            mock_service,
            task_interceptor=interceptor,
            known_projects=None,
        )

        result = await _call_delete(mcp_server, replacement_memory_id=SURVIVOR)

        assert result['status'] == 'deleted'
        mock_service.delete_memory.assert_awaited_once()
        interceptor.get_tasks.assert_not_awaited()


TOMBSTONE_KEY = 'x_memory_citation_tombstones'


def _repointed_metadata(run_id='run-1'):
    """Metadata exactly as a completed repoint pass leaves it.

    ``cited`` now addresses the survivor; the only remaining mention of the
    doomed id lives in the labelled provenance ledger, whose
    ``superseded_memory_id`` names it BY DESIGN.
    """
    return {
        'cited': SURVIVOR,
        TOMBSTONE_KEY: [
            {
                'superseded_memory_id': DOOMED,
                'replacement_memory_id': SURVIVOR,
                'paths': ['cited'],
                'run_id': run_id,
            },
        ],
    }


class TestTombstonedTaskIsNotALiveCiter:
    """The gate and the sweep must agree that a tombstone is not a citation.

    ``_scan_task_citations`` feeds the gate's ``live_citers`` list. If it counts
    a tombstone's ``superseded_memory_id`` — which exists precisely to name the
    deleted id — then an already-repointed task looks like an outstanding one
    forever, the gate demands a repoint the sweep reports zero work for, and the
    retry ``_CITATION_REPOINT_FAILED_HINT`` instructs the caller to perform
    never terminates.
    """

    @pytest.fixture
    def mock_service(self):
        return _make_service()

    @pytest.mark.asyncio
    async def test_fully_repointed_task_does_not_require_a_replacement(
        self, mock_service,
    ):
        """(a) Nothing LIVE points at the doomed id, so the delete proceeds even
        with no replacement_memory_id supplied."""
        interceptor = _make_interceptor([
            _task('901', 'pending', _repointed_metadata()),
        ])
        mcp_server = create_mcp_server(
            mock_service,
            task_interceptor=interceptor,
            known_projects=KNOWN_PROJECTS,
        )

        result = await _call_delete(mcp_server)

        assert result.get('error_type') != 'CitationRepointRequired'
        assert result['status'] == 'deleted'
        mock_service.delete_memory.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fully_repointed_task_is_never_rewritten_again(
        self, mock_service,
    ):
        """(b) The same shape WITH a valid replacement issues no redundant
        write — a second rewrite would clobber superseded_memory_id and destroy
        the forwarding provenance."""
        interceptor = _make_interceptor([
            _task('902', 'pending', _repointed_metadata()),
        ])
        mcp_server = create_mcp_server(
            mock_service,
            task_interceptor=interceptor,
            known_projects=KNOWN_PROJECTS,
        )

        result = await _call_delete(mcp_server, replacement_memory_id=SURVIVOR)

        assert result['status'] == 'deleted'
        interceptor.update_task.assert_not_awaited()
        mock_service.delete_memory.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_retry_after_a_partial_failure_completes_the_delete(
        self, mock_service,
    ):
        """(c) END-TO-END RETRY — the exact sequence the failure hint prescribes.

        Pass 1: two pending citers, the second's write REJECTED -> the delete is
        refused. Pass 2: task one now carries pass-1's output (so it is no
        longer a citer) and task two's write lands -> only task two is written
        and the delete proceeds.
        """
        # ---- Pass 1: task 912's repoint write is rejected. ----
        interceptor_1 = MagicMock()
        interceptor_1.get_tasks = AsyncMock(return_value={'tasks': [
            _task('911', 'pending', {'cited': DOOMED}),
            _task('912', 'pending', {'cited': DOOMED}),
        ]})

        async def _reject_912(**kwargs):
            if kwargs['task_id'] == '912':
                return {'success': False, 'error': 'write lock contention'}
            return {'success': True}

        interceptor_1.update_task = AsyncMock(side_effect=_reject_912)
        server_1 = create_mcp_server(
            mock_service,
            task_interceptor=interceptor_1,
            known_projects=KNOWN_PROJECTS,
        )

        result_1 = await _call_delete(server_1, replacement_memory_id=SURVIVOR)

        assert result_1['error_type'] == 'CitationRepointFailed'
        mock_service.delete_memory.assert_not_awaited()

        # Task 911's write DID land; capture the metadata it now holds.
        payload_911 = json.loads(next(
            c[1]['metadata'] for c in interceptor_1.update_task.call_args_list
            if c[1]['task_id'] == '911'
        ))
        after_911 = {'cited': DOOMED}
        after_911.update(payload_911)  # shallow merge, as the backend applies it

        # ---- Pass 2: retry against the post-pass-1 snapshot. ----
        interceptor_2 = _make_interceptor([
            _task('911', 'pending', after_911),
            _task('912', 'pending', {'cited': DOOMED}),
        ])
        server_2 = create_mcp_server(
            mock_service,
            task_interceptor=interceptor_2,
            known_projects=KNOWN_PROJECTS,
        )

        result_2 = await _call_delete(server_2, replacement_memory_id=SURVIVOR)

        assert result_2['status'] == 'deleted'
        mock_service.delete_memory.assert_awaited_once()

        # Only the still-outstanding task was written on the retry.
        written = [c[1]['task_id'] for c in interceptor_2.update_task.call_args_list]
        assert written == ['912']

        # And task 911's provenance survived the second pass intact.
        assert after_911['cited'] == SURVIVOR
        ledger = after_911[TOMBSTONE_KEY]
        assert len(ledger) == 1
        assert ledger[0]['superseded_memory_id'] == DOOMED
        assert ledger[0]['replacement_memory_id'] == SURVIVOR
