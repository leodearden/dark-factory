"""Integration tests for the premature-completion-claim write-gate in
add_memory and add_episode (task 2824).

A recon-stage- write that frames a specific task as COMPLETE in present/past
tense ("task N has landed and now enforces X") is soft-blocked when that
task's LIVE status is confirmed non-terminal; it passes when the task is
actually done/cancelled, and fails OPEN whenever the live status cannot be
resolved (task_interceptor unconfigured, project_id unknown, get_statuses
raises, or a status is unknown/absent).

Closes the gap left by task 2628 (live-status *fields*) and task 2624
(Stage-2's own *task* writes): neither cross-checked present-tense
completion-claim *phrasing* against the named task's live status.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.server.tools import create_mcp_server

# Frames task 5252 as complete in present tense ("has landed", "now enforces").
_COMPLETION_CONTENT = 'task 5252 has landed and now enforces the manifest gate'
_PROJECT_ID = 'dark_factory'
_KNOWN_PROJECTS = {'dark_factory': '/root'}


def _server_with_statuses(mock_service, statuses):
    """Build a server whose task_interceptor.get_statuses returns `statuses`.

    `statuses` is the string-keyed {id: status} map real get_statuses returns.
    """
    task_interceptor = MagicMock()
    task_interceptor.get_statuses = AsyncMock(return_value=statuses)
    return create_mcp_server(
        mock_service,
        task_interceptor=task_interceptor,
        known_projects=_KNOWN_PROJECTS,
    )


def _allowing_service():
    """An AsyncMock memory_service whose add_memory returns a dict-dumpable
    result (so the tool's `return result.model_dump()` yields a real dict).
    """
    mock_service = AsyncMock()
    _mem_result = MagicMock()
    _mem_result.model_dump.return_value = {'id': 'x'}
    mock_service.add_memory.return_value = _mem_result
    return mock_service


class TestAddMemoryPrematureCompletionGate:
    """Write-gate: recon-stage- agents must not write a present-tense
    completion claim about a task that is still LIVE/non-terminal via
    add_memory, for the gated categories (decisions_and_rationale /
    observations_and_summaries / entities_and_relations / temporal_facts).
    """

    @pytest.mark.asyncio
    async def test_rejects_completion_claim_for_in_progress_task(self):
        """Completion claim + gated category + recon-stage agent + the named
        task LIVE in-progress → rejected. Standard error dict, and
        memory_service.add_memory must NOT be called. (Acceptance case.)
        """
        mock_service = AsyncMock()
        server = _server_with_statuses(mock_service, {'5252': 'in-progress'})

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _COMPLETION_CONTENT,
                'category': 'observations_and_summaries',
                'agent_id': 'recon-stage-task_knowledge_sync',
                'project_id': _PROJECT_ID,
            },
        )

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert result.get('error') == 'premature_completion_claim_write_blocked', (
            f"Expected error='premature_completion_claim_write_blocked', got: {result!r}"
        )
        assert result.get('error_type') == 'ReconPrematureCompletionClaimWriteRejected', (
            f"Expected error_type='ReconPrematureCompletionClaimWriteRejected', got: {result!r}"
        )
        assert result.get('content_excerpt') == _COMPLETION_CONTENT[:200], (
            f'Expected content_excerpt=content[:200], got: {result!r}'
        )
        assert result.get('hint'), f'Expected a non-empty hint, got: {result!r}'
        mock_service.add_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_allows_completion_claim_for_done_task(self):
        """Same content + gated category + recon-stage agent but the named task
        is LIVE done (terminal) → gate does NOT fire; add_memory called once.
        (Acceptance case: a claim about an actually-complete task is allowed.)
        """
        mock_service = _allowing_service()
        server = _server_with_statuses(mock_service, {'5252': 'done'})

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _COMPLETION_CONTENT,
                'category': 'observations_and_summaries',
                'agent_id': 'recon-stage-task_knowledge_sync',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error') != 'premature_completion_claim_write_blocked', (
            f'Gate must not fire for a done (terminal) task; got: {result!r}'
        )
        mock_service.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_allows_non_recon_agent(self):
        """Same content but agent_id='claude-interactive' (non-recon) → not
        blocked even though task 5252 is in-progress. The gate is scoped to
        recon-stage- agents only.
        """
        mock_service = _allowing_service()
        server = _server_with_statuses(mock_service, {'5252': 'in-progress'})

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _COMPLETION_CONTENT,
                'category': 'observations_and_summaries',
                'agent_id': 'claude-interactive',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error') != 'premature_completion_claim_write_blocked', (
            f'Gate must not fire for a non-recon agent; got: {result!r}'
        )
        mock_service.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_allows_none_category(self):
        """Same content + category=None (auto-classify) → not blocked. Bypassing
        category=None mirrors the 2628 live-status precedent.
        """
        mock_service = _allowing_service()
        server = _server_with_statuses(mock_service, {'5252': 'in-progress'})

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _COMPLETION_CONTENT,
                'category': None,
                'agent_id': 'recon-stage-task_knowledge_sync',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error') != 'premature_completion_claim_write_blocked', (
            f'Gate must not fire when category=None (auto-classify path); got: {result!r}'
        )
        mock_service.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_allows_non_gated_procedural_knowledge_category(self):
        """Same content + category=procedural_knowledge (non-gated, where a
        norm ABOUT completion phrasing legitimately lives) → not blocked.
        """
        mock_service = _allowing_service()
        server = _server_with_statuses(mock_service, {'5252': 'in-progress'})

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _COMPLETION_CONTENT,
                'category': 'procedural_knowledge',
                'agent_id': 'recon-stage-task_knowledge_sync',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error') != 'premature_completion_claim_write_blocked', (
            f'Gate must not fire for non-gated category=procedural_knowledge; got: {result!r}'
        )
        mock_service.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_fails_open_when_task_interceptor_unconfigured(self):
        """Same content + gated category + recon-stage agent, but no
        task_interceptor is configured → the live status cannot be resolved, so
        the gate fails OPEN and the write proceeds (add_memory called once).
        """
        mock_service = _allowing_service()
        server = create_mcp_server(mock_service, known_projects=_KNOWN_PROJECTS)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _COMPLETION_CONTENT,
                'category': 'observations_and_summaries',
                'agent_id': 'recon-stage-task_knowledge_sync',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error') != 'premature_completion_claim_write_blocked', (
            f'Gate must fail open when task_interceptor is unconfigured; got: {result!r}'
        )
        mock_service.add_memory.assert_called_once()
