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
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.server.tools import create_mcp_server
from fused_memory.services.memory_service import DescendantScan

# Doomed duplicate and its surviving replacement — both canonical 36-char UUIDs.
DOOMED = '2531b4d8-1111-4aaa-8bbb-000000000001'
SURVIVOR = '9f3ac071-3333-4eee-8fff-000000000003'
# A child of DOOMED: destroyed by `cascade=True`, so it has to pass the gate
# on its own account (task 3197). Before the pre-flight it never did — the
# cascade recursed inside the SERVICE, below the tool layer the gate lives at.
CHILD = '7a4e15c2-4444-4bbb-8ccc-000000000004'
GRANDCHILD = 'c0ffee11-5555-4ddd-8eee-000000000005'

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


def _make_service(resolvable=(SURVIVOR,), descendants=(), truncated=False):
    """MemoryService whose awaited children are EXPLICIT AsyncMocks.

    Required by tests/test_check_bare_magicmock_config.py and
    tests/test_check_asyncmock_assertion_style.py — an awaited child must be
    declared, not left as an auto-attribute.

    ``get_memory_by_id`` models the real Mem0 point read: the full record for a
    resolvable id, ``None`` on a genuine miss (memory_service.py:3339-3375).

    ``list_descendant_ids`` is the read-only enumeration the cascade
    pre-flight gates on (task 3197): what a ``cascade=True`` delete WOULD
    destroy, deepest-first, plus whether that answer is complete.
    """
    svc = AsyncMock()
    svc.delete_memory = AsyncMock(
        return_value={'status': 'deleted', 'store': 'mem0', 'id': DOOMED},
    )

    async def _get(project_id, memory_id):
        if memory_id in resolvable:
            return {'id': memory_id, 'content': 'surviving canonical advice', 'metadata': {}}
        return None

    svc.get_memory_by_id = AsyncMock(side_effect=_get)
    svc.list_descendant_ids = AsyncMock(
        return_value=DescendantScan(ids=list(descendants), truncated=truncated),
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


class TestGateAppliesToEveryCaller:
    """The gate keys on the RECORD, not on who is deleting it (task 3624).

    "Will this delete dangle a live pointer?" is a property of the entry and
    the task DB, not of the caller's agent_id. The original scoping bounded the
    gate to ``recon-stage-*`` callers, so the identical delete issued from an
    interactive session — the way the 25-gate consolidation batch of task 3524
    is actually driven — landed unguarded and stranded exactly the pointers the
    guard exists to protect.
    """

    @pytest.fixture
    def mock_service(self):
        return _make_service()

    @pytest.fixture
    def interceptor(self):
        return _make_interceptor([
            _task('1201', 'pending', {'mem0_canonical_entry': DOOMED}),
            _task('1202', 'pending', {'unrelated': 'nothing'}),
        ])

    @pytest.fixture
    def mcp_server(self, mock_service, interceptor):
        return create_mcp_server(
            mock_service,
            task_interceptor=interceptor,
            known_projects=KNOWN_PROJECTS,
        )

    @pytest.mark.asyncio
    async def test_interactive_caller_with_live_citers_is_refused(
        self, mcp_server, mock_service,
    ):
        """(a) THE LEAF SIGNAL. An interactive delete of a still-cited mem0
        entry is refused, exactly as a recon-stage one already was."""
        result = await _call_delete(mcp_server, agent_id='claude-interactive')

        assert result['error_type'] == 'CitationRepointRequired'
        assert [c['task_id'] for c in result['citing_tasks']] == ['1201']
        mock_service.delete_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_interactive_caller_with_no_live_citers_still_deletes(
        self, mock_service,
    ):
        """(b) REGRESSION for the common case. Broadening the gate must not
        turn every ordinary interactive delete into a refusal — the gate runs,
        finds nothing live, and gets out of the way."""
        interceptor = _make_interceptor([
            _task('1211', 'pending', {'cited': SURVIVOR}),
            _task('1212', 'pending', {'unrelated': 'nothing'}),
        ])
        mcp_server = create_mcp_server(
            mock_service,
            task_interceptor=interceptor,
            known_projects=KNOWN_PROJECTS,
        )

        result = await _call_delete(mcp_server, agent_id='claude-interactive')

        assert result['status'] == 'deleted'
        mock_service.delete_memory.assert_awaited_once()
        interceptor.update_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_interactive_caller_can_repoint_with_a_replacement(
        self, mcp_server, mock_service, interceptor,
    ):
        """(c) The remedy is available to every caller too: supplying a valid
        survivor gets the interactive caller the same repoint-then-delete path
        recon already had, not merely a refusal it cannot clear."""
        result = await _call_delete(
            mcp_server,
            agent_id='claude-interactive',
            replacement_memory_id=SURVIVOR,
        )

        assert result['status'] == 'deleted'
        mock_service.delete_memory.assert_awaited_once()
        interceptor.update_task.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_gate_runs_when_agent_id_is_absent(
        self, mcp_server, mock_service,
    ):
        """(d) The deliberate consequence of keying on the record: a delete
        with no agent_id at all is gated too. Under the old predicate an
        unidentified caller was the LEAST guarded one."""
        result = await _call_delete(mcp_server, agent_id=None)

        assert result['error_type'] == 'CitationRepointRequired'
        mock_service.delete_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_refusal_hint_names_the_allow_dangling_citations_escape(
        self, mcp_server,
    ):
        """A refusal with no reachable next action is a dead end, not a guard.

        ``hint`` is a field of the structured tool response, so this is
        behaviour rather than prose. Before broadening, the only exit the hint
        offered was "supply a surviving UUID" — which is unreachable for the
        caller the broadening newly captures: an operator dropping a record
        outright, with no survivor at all to repoint to. The escape is
        undiscoverable unless the refusal names it.
        """
        result = await _call_delete(mcp_server, agent_id='claude-interactive')

        assert result['error_type'] == 'CitationRepointRequired'
        assert 'allow_dangling_citations' in result['hint']

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'bad_replacement',
        ['search(query="canonical dispatch advice")', SURVIVOR[:8], DOOMED],
        ids=['search-instruction', 'truncated-uuid', 'self-repoint'],
    )
    async def test_a_bad_replacement_is_not_offered_the_escape(
        self, mcp_server, bad_replacement,
    ):
        """The escape is available to every caller but advertised to only one.

        A caller who reached CitationReplacementInvalid did so BY naming a
        survivor, so they demonstrably have one and their fix is to correct the
        value. Dangling a live pointer because a UUID was truncated reproduces
        the incident this gate closes, so the shared hint must keep saying
        "copy the correct UUID" and nothing else.
        """
        result = await _call_delete(
            mcp_server,
            agent_id='claude-interactive',
            replacement_memory_id=bad_replacement,
        )

        assert result['error_type'] == 'CitationReplacementInvalid'
        assert 'allow_dangling_citations' not in result['hint']
        assert 'replacement_memory_id' in result['hint']


class TestAllowDanglingCitationsEscape:
    """``metadata={'allow_dangling_citations': True}`` is the deliberate escape.

    Broadening the gate (task 3624) leaves one caller with no reachable next
    action: an operator dropping a record outright, with no survivor at all to
    repoint to. ``replacement_memory_id`` cannot help them, so the refusal would
    be a dead end. The escape closes that — knowingly, loudly, and only on a
    LITERAL ``True``.
    """

    @pytest.fixture
    def mock_service(self):
        return _make_service()

    @pytest.fixture
    def interceptor(self):
        return _make_interceptor([
            _task('1301', 'pending', {'mem0_canonical_entry': DOOMED}),
            _task('1302', 'pending', {'unrelated': 'nothing'}),
        ])

    @pytest.fixture
    def mcp_server(self, mock_service, interceptor):
        return create_mcp_server(
            mock_service,
            task_interceptor=interceptor,
            known_projects=KNOWN_PROJECTS,
        )

    @pytest.mark.asyncio
    async def test_override_permits_the_delete(
        self, mcp_server, mock_service, interceptor,
    ):
        """(a) The flag lets the delete through with no replacement supplied —
        and rewrites nothing. It dangles the citation knowingly; it does not
        silently invent a forwarding pointer.

        The response names what was dangled, in the same shape the refusal
        reports in ``citing_tasks``. A server-side WARNING alone would be
        invisible to the MCP caller this escape exists to serve, which is the
        same silent-override defect the placement of the check argues against.
        """
        result = await _call_delete(
            mcp_server,
            agent_id='claude-interactive',
            metadata={'allow_dangling_citations': True},
        )

        assert result['status'] == 'deleted'
        mock_service.delete_memory.assert_awaited_once()
        interceptor.update_task.assert_not_awaited()
        assert result['dangled_citation_count'] == 1
        assert [c['task_id'] for c in result['dangled_citations']] == ['1301']
        assert result['dangled_citations'][0]['paths'] == ['mem0_canonical_entry']
        # Nothing was dropped, so there is no ignored-argument noise to report.
        assert 'ignored_replacement_memory_id' not in result

    @pytest.mark.asyncio
    @pytest.mark.parametrize('value', ['yes', 1, 'true', None])
    async def test_only_a_literal_true_counts(
        self, mcp_server, mock_service, value,
    ):
        """(b) The ``is True`` half of the house override idiom
        (tools.py:2103-2104). A truthy ``'yes'`` or ``1`` must NOT unlock an
        irreversible delete — the same literal-boolean convention CHANGELOG.md
        records for ``allow_mcp_markup``.

        And it must not degrade SILENTLY: a dropped value plus a hint telling
        the caller to pass the flag they believe they just passed is a retry
        loop, not a guard. The refusal names the value it ignored.
        """
        result = await _call_delete(
            mcp_server,
            agent_id='claude-interactive',
            metadata={'allow_dangling_citations': value},
        )

        assert result['error_type'] == 'CitationRepointRequired', value
        mock_service.delete_memory.assert_not_awaited()
        assert result['ignored_override'] == {'allow_dangling_citations': value}
        assert 'literal boolean True' in result['hint']

    @pytest.mark.asyncio
    async def test_a_literal_false_is_honoured_without_being_reported(
        self, mcp_server, mock_service,
    ):
        """The ignored-value report is for a MALFORMED override, not for a
        deliberate one. ``False`` is a literal boolean saying "do not override";
        answering it with "your override was ignored" would be noise on a caller
        who did exactly the right thing."""
        result = await _call_delete(
            mcp_server,
            agent_id='claude-interactive',
            metadata={'allow_dangling_citations': False},
        )

        assert result['error_type'] == 'CitationRepointRequired'
        mock_service.delete_memory.assert_not_awaited()
        assert 'ignored_override' not in result
        assert 'literal boolean True' not in result['hint']

    @pytest.mark.asyncio
    async def test_override_does_not_defeat_the_fail_closed_scan_error(
        self, mock_service,
    ):
        """(c) PLACEMENT, pinned. The escape sits AFTER the scan, so an
        unreadable task DB still fails closed. The flag means "I accept
        dangling the citers you just showed me" — with nothing enumerated there
        is nothing to knowingly accept, so this is the right semantics rather
        than an accident of ordering."""
        interceptor = _make_interceptor([])
        interceptor.get_tasks = AsyncMock(side_effect=RuntimeError('task db unreachable'))
        mcp_server = create_mcp_server(
            mock_service,
            task_interceptor=interceptor,
            known_projects=KNOWN_PROJECTS,
        )

        result = await _call_delete(
            mcp_server,
            agent_id='claude-interactive',
            metadata={'allow_dangling_citations': True},
        )

        assert result['error_type'] == 'CitationScanFailed'
        mock_service.delete_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_override_is_never_forwarded_to_the_store(
        self, mcp_server, mock_service,
    ):
        """(d) The write-time flag must not reach persistence. It cannot today
        (delete_memory discards _extract_causation's cleaned dict and the
        service call takes no metadata parameter), so this guards the actual
        hazard: a future signature change that starts forwarding the
        envelope."""
        await _call_delete(
            mcp_server,
            agent_id='claude-interactive',
            metadata={'allow_dangling_citations': True},
        )

        kwargs = mock_service.delete_memory.await_args.kwargs
        assert 'metadata' not in kwargs
        assert 'allow_dangling_citations' not in kwargs

    @pytest.mark.asyncio
    async def test_override_applies_to_a_recon_caller_too(
        self, mcp_server, mock_service,
    ):
        """(e) The escape is a property of stated INTENT, not of identity —
        adding a second caller-identity check would reintroduce exactly the
        scoping this task just removed."""
        result = await _call_delete(
            mcp_server,
            agent_id=RECON_AGENT,
            metadata={'allow_dangling_citations': True},
        )

        assert result['status'] == 'deleted'
        mock_service.delete_memory.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_override_logs_a_warning_naming_the_memory_id_and_every_dangled_citer(
        self, mock_service, caplog,
    ):
        """The override must leave a TRACE. An escape that lands silently is the
        same class of defect as the gate that never ran: the operator's stated
        intent is recorded, but which pointers it stranded is not — which is
        exactly the enumerate-by-hand step the incident got wrong (3 of 8).

        The WARNING names ``live_citers``, the same list the rejection path
        reports in ``citing_tasks``: what is being knowingly dangled, not every
        task that ever mentioned the id. A citer that has already reached a
        terminal status is not dangled by this delete, so naming it would
        overstate the damage.
        """
        interceptor = _make_interceptor([
            _task('1311', 'pending', {'mem0_canonical_entry': DOOMED}),
            _task('1312', 'pending', {
                'memory_hints': {'entities': [], 'queries': [f'advice {DOOMED} here']},
            }),
            _task('1313', 'done', {'cited': DOOMED}),
            _task('1314', 'pending', {'unrelated': 'nothing'}),
        ])
        mcp_server = create_mcp_server(
            mock_service,
            task_interceptor=interceptor,
            known_projects=KNOWN_PROJECTS,
        )

        with caplog.at_level('WARNING'):
            result = await _call_delete(
                mcp_server,
                agent_id='claude-interactive',
                metadata={'allow_dangling_citations': True},
            )

        assert result['status'] == 'deleted'
        assert [r for r in caplog.records if r.levelname == 'WARNING']
        assert 'allow_dangling_citations' in caplog.text
        # The record being destroyed, and the caller who asked for it.
        assert DOOMED in caplog.text
        assert 'claude-interactive' in caplog.text
        # Every live citer this delete strands.
        assert '1311' in caplog.text
        assert '1312' in caplog.text
        # ...and only those: the terminal citer is not dangled by this delete.
        assert '1313' not in caplog.text

    @pytest.mark.asyncio
    async def test_override_warning_names_a_supplied_replacement_it_did_not_use(
        self, mcp_server, mock_service, interceptor, caplog,
    ):
        """Both arguments together are contradictory — "repoint them to this
        survivor" vs "dangle them knowingly" — so one must be dropped. The
        override wins, keeping a single code path, but the dropped value is
        REPORTED rather than silently discarded (the repo's
        loud-over-silent-degradation norm)."""
        with caplog.at_level('WARNING'):
            result = await _call_delete(
                mcp_server,
                agent_id='claude-interactive',
                replacement_memory_id=SURVIVOR,
                metadata={'allow_dangling_citations': True},
            )

        assert result['status'] == 'deleted'
        mock_service.delete_memory.assert_awaited_once()
        # The override wins: nothing was repointed to the survivor.
        interceptor.update_task.assert_not_awaited()
        # But the ignored argument is named, not dropped in silence — in the
        # log AND in the response, since the MCP caller who supplied it reads
        # only the latter.
        assert SURVIVOR in caplog.text
        assert result['ignored_replacement_memory_id'] == SURVIVOR

    @pytest.mark.asyncio
    async def test_scan_cost_is_reported_at_debug(
        self, mcp_server, mock_service, caplog,
    ):
        """Broadening the gate put a full task-tree read on EVERY mem0 delete,
        so a 25-delete batch pays it 25 times. The snapshot is deliberately not
        cached (a stale one would hide a citer that appeared after it was taken,
        which is a race on an irreversible delete), so the cost is made
        measurable instead. DEBUG, not INFO: this is now the common path."""
        with caplog.at_level('DEBUG'):
            await _call_delete(
                mcp_server,
                agent_id='claude-interactive',
                metadata={'allow_dangling_citations': True},
            )

        scan_lines = [
            r.getMessage() for r in caplog.records
            if r.levelname == 'DEBUG' and 'citation gate: scanned' in r.getMessage()
        ]
        assert len(scan_lines) == 1
        # The two numbers an operator needs: how much tree was walked, and how
        # long it took.
        assert 'scanned 2 task(s)' in scan_lines[0]
        assert ' ms' in scan_lines[0]

    @pytest.mark.asyncio
    async def test_no_warning_when_the_flag_is_absent(
        self, mcp_server, mock_service, caplog,
    ):
        """The trace belongs to the OVERRIDE, not to every gated delete. An
        ordinary refusal already returns its citers to the caller structurally,
        so logging there would be noise on the common path and would dilute the
        one line an operator needs to find."""
        with caplog.at_level('WARNING'):
            result = await _call_delete(mcp_server, agent_id='claude-interactive')

        assert result['error_type'] == 'CitationRepointRequired'
        mock_service.delete_memory.assert_not_awaited()
        assert 'allow_dangling_citations' not in caplog.text


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
        # ``dict[str, Any]``: post-merge the blob is heterogeneous — the
        # tombstone ledger is a ``list[dict]``, not a ``str``.
        after_911: dict[str, Any] = {'cited': DOOMED}
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


# Well-formed, canonical, and addresses nothing — the shape check cannot tell
# it apart from a real survivor.
HALLUCINATED = 'deadbeef-1111-4aaa-8bbb-000000000099'


class TestReplacementMustResolveAndDiffer:
    """A concrete-looking replacement is not yet a usable one.

    ``is_concrete_memory_id`` only rules out PROSE — its docstring claims
    nothing about existence — so the shape check alone accepts any canonical
    UUID string. That leaves two holes which reproduce exactly the harm this
    gate exists to prevent, with the original now destroyed and unrecoverable:

    - a hallucinated/typo'd id (root cause (1) named in ``verify_cited_memories``'
      own docstring) rewrites every live citer to point at nothing, and THEN
      lands the delete — the incident's dangling pointers, merely relocated;
    - ``replacement_memory_id == memory_id`` makes the rewrite a
      self-substitution that still reports ``count > 0`` and zero failures, so
      the gate declares a successful repoint while every citation still
      addresses the entry being destroyed.
    """

    @pytest.fixture
    def interceptor(self):
        return _make_interceptor([
            _task('1001', 'pending', {'mem0_canonical_entry': DOOMED}),
            _task('1002', 'pending', {'unrelated': 'nothing'}),
        ])

    def _server(self, service, interceptor):
        return create_mcp_server(
            service,
            task_interceptor=interceptor,
            known_projects=KNOWN_PROJECTS,
        )

    @pytest.mark.asyncio
    async def test_unresolvable_replacement_is_refused(self, interceptor):
        """(a) A well-formed id that addresses nothing must never become the
        destination of a live citation."""
        service = _make_service()
        mcp_server = self._server(service, interceptor)

        result = await _call_delete(mcp_server, replacement_memory_id=HALLUCINATED)

        assert result['error_type'] == 'CitationReplacementNotFound'
        assert result['replacement_memory_id'] == HALLUCINATED
        assert result['memory_id'] == DOOMED
        assert [c['task_id'] for c in result['citing_tasks']] == ['1001']
        # Neither half of the harm happened: nothing was repointed at a phantom,
        # and the original still exists.
        service.delete_memory.assert_not_awaited()
        interceptor.update_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_self_repoint_is_refused(self, interceptor):
        """(b) Repointing an id to ITSELF is a no-op dressed as a success."""
        service = _make_service()
        mcp_server = self._server(service, interceptor)

        result = await _call_delete(mcp_server, replacement_memory_id=DOOMED)

        assert result['error_type'] == 'CitationReplacementInvalid'
        assert result['replacement_memory_id'] == DOOMED
        assert 'itself' in result['error'] or 'same' in result['error']
        service.delete_memory.assert_not_awaited()
        interceptor.update_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_existence_check_failure_fails_closed(self, interceptor):
        """(c) A raised backend read is 'unknown', not 'resolves' — matching the
        scan's fail-closed posture ahead of an irreversible delete."""
        service = _make_service()
        service.get_memory_by_id = AsyncMock(side_effect=TimeoutError('qdrant timeout'))
        mcp_server = self._server(service, interceptor)

        result = await _call_delete(mcp_server, replacement_memory_id=SURVIVOR)

        assert result['error_type'] == 'CitationReplacementCheckFailed'
        service.delete_memory.assert_not_awaited()
        interceptor.update_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_resolvable_different_replacement_still_repoints_then_deletes(
        self, interceptor,
    ):
        """(d) The happy path is unchanged, and the existence read targets the
        REPLACEMENT — checking the doomed id would prove nothing."""
        service = _make_service()
        mcp_server = self._server(service, interceptor)

        result = await _call_delete(mcp_server, replacement_memory_id=SURVIVOR)

        assert result['status'] == 'deleted'
        service.delete_memory.assert_awaited_once()
        interceptor.update_task.assert_awaited_once()
        service.get_memory_by_id.assert_awaited_once_with('dark_factory', SURVIVOR)

    @pytest.mark.asyncio
    async def test_no_live_citers_pays_for_no_existence_read(self):
        """(e) The checks sit AFTER the no-live-citers early-out, so an ordinary
        uncited delete takes no extra store read."""
        service = _make_service()
        interceptor = _make_interceptor([_task('1101', 'pending', {'cited': SURVIVOR})])
        mcp_server = self._server(service, interceptor)

        result = await _call_delete(mcp_server, replacement_memory_id=HALLUCINATED)

        assert result['status'] == 'deleted'
        service.get_memory_by_id.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_inactive_gate_pays_for_no_existence_read(self, interceptor):
        """(e) An out-of-scope RECORD (a graphiti store) bypasses the gate
        entirely and reads nothing.

        Task 3624 narrowed this: the caller-identity arm
        (``{'agent_id': 'claude-interactive'}``) used to sit alongside the
        graphiti one, but the gate no longer keys on WHO is deleting — only on
        the record. That arm's inverse now lives in
        ``TestGateAppliesToEveryCaller``.
        """
        service = _make_service()
        mcp_server = self._server(service, interceptor)

        result = await _call_delete(
            mcp_server, replacement_memory_id=HALLUCINATED, store='graphiti',
        )

        assert result['status'] == 'deleted'
        service.get_memory_by_id.assert_not_awaited()
        interceptor.get_tasks.assert_not_awaited()


class TestCascadeCitationPreflight:
    """A cascaded child delete must pass the citation gate too (task 3197).

    The cascade recurses inside ``MemoryService.delete_memory`` — BELOW the
    tool layer where this gate lives — so before the pre-flight a
    ``cascade=True`` delete destroyed every descendant without any of them
    being scanned for live citers. One gated record, N ungated ones,
    reported as a success: precisely the dangling pointers this gate exists
    to prevent, reintroduced by the opt-in added to close the orphan half of
    the same lifecycle rule.
    """

    @pytest.fixture
    def mock_service(self):
        return _make_service(descendants=[CHILD])

    @pytest.fixture
    def interceptor(self):
        # The citer points at the CHILD, not at the delete's target. Nothing
        # in the single-record path would ever have looked at it.
        return _make_interceptor([
            _task('1401', 'pending', {'mem0_canonical_entry': CHILD}),
            _task('1402', 'pending', {'unrelated': 'nothing'}),
        ])

    @pytest.fixture
    def mcp_server(self, mock_service, interceptor):
        return create_mcp_server(
            mock_service,
            task_interceptor=interceptor,
            known_projects=KNOWN_PROJECTS,
        )

    @pytest.mark.asyncio
    async def test_cited_descendant_refuses_the_whole_cascade(
        self, mcp_server, mock_service,
    ):
        """(a) THE LEAF SIGNAL. Nothing is deleted — not the child, not the
        parent.

        A cascade is one operation with one stated intent. Deleting the
        uncited half and refusing the rest would leave the caller with a
        partially-destroyed subtree they never asked for and cannot infer
        the shape of.
        """
        result = await _call_delete(mcp_server, cascade=True)

        assert result['error_type'] == 'CascadeCitationGateRejected'
        mock_service.delete_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_refusal_names_the_blocking_descendant_and_its_citers(
        self, mcp_server,
    ):
        """(b) Actionable from the wire response alone.

        The caller has to learn WHICH descendant blocked and WHY without a
        second lookup — the gate's own principle ("never left to re-derive
        the enumeration by hand"), applied to the cascade set.
        """
        result = await _call_delete(mcp_server, cascade=True)

        blocked = result['blocked']
        assert [b['memory_id'] for b in blocked] == [CHILD]
        assert blocked[0]['error_type'] == 'CitationRepointRequired'
        citers = blocked[0]['citing_tasks']
        assert [c['task_id'] for c in citers] == ['1401']
        assert citers[0]['status'] == 'pending'
        assert citers[0]['paths'] == ['mem0_canonical_entry']

    @pytest.mark.asyncio
    async def test_refusal_names_the_cascade_target_and_size(self, mcp_server):
        """(c) The envelope says what operation was refused, not just what
        blocked it — the target id and how many records were in scope."""
        result = await _call_delete(mcp_server, cascade=True)

        assert result['memory_id'] == DOOMED
        assert result['cascade_size'] == 2  # the child plus the target itself

    @pytest.mark.asyncio
    async def test_uncited_cascade_proceeds(self, mock_service):
        """(d) The gate did not become a blanket cascade ban.

        Nothing in the set is cited, so the whole cascade runs — and the
        flag reaches the service verbatim.
        """
        interceptor = _make_interceptor([
            _task('1411', 'pending', {'cited': SURVIVOR}),
            _task('1412', 'pending', {'unrelated': 'nothing'}),
        ])
        mcp_server = create_mcp_server(
            mock_service,
            task_interceptor=interceptor,
            known_projects=KNOWN_PROJECTS,
        )

        result = await _call_delete(mcp_server, cascade=True)

        assert result['status'] == 'deleted'
        mock_service.delete_memory.assert_awaited_once()
        assert mock_service.delete_memory.call_args.kwargs['cascade'] is True
        interceptor.update_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_cascade_delete_keeps_the_single_record_envelope(
        self, mock_service,
    ):
        """(e) The aggregate shape is scoped to the cascade path.

        Every existing caller of the plain delete keeps today's exact
        envelope; the new error type is additive, not a rewrite of the wire
        contract.
        """
        interceptor = _make_interceptor([
            _task('1421', 'pending', {'mem0_canonical_entry': DOOMED}),
        ])
        mcp_server = create_mcp_server(
            mock_service,
            task_interceptor=interceptor,
            known_projects=KNOWN_PROJECTS,
        )

        result = await _call_delete(mcp_server)

        assert result['error_type'] == 'CitationRepointRequired'
        assert 'blocked' not in result
        assert [c['task_id'] for c in result['citing_tasks']] == ['1421']
        mock_service.delete_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_childless_cascade_still_reports_the_aggregate_shape(self):
        """(f) The error type keys on the FLAG the caller passed, not on
        corpus state they cannot see.

        Whether the tree happened to have children this time is invisible to
        the caller, so branching the wire contract on it would force every
        cascade error handler to handle both shapes anyway.
        """
        service = _make_service(descendants=[])
        interceptor = _make_interceptor([
            _task('1431', 'pending', {'mem0_canonical_entry': DOOMED}),
        ])
        mcp_server = create_mcp_server(
            service, task_interceptor=interceptor, known_projects=KNOWN_PROJECTS,
        )

        result = await _call_delete(mcp_server, cascade=True)

        assert result['error_type'] == 'CascadeCitationGateRejected'
        assert result['cascade_size'] == 1
        assert [b['memory_id'] for b in result['blocked']] == [DOOMED]
        service.delete_memory.assert_not_awaited()
