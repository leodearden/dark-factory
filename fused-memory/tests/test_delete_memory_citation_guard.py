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
