"""Regression tests verifying that tools.py delegates validation to the shared validators.

These tests replace the original vacuous tests that either called shared validators
directly (not through tools.py at all) or checked hasattr() on closures that were
never module-level attributes.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

import fused_memory.server.tools as tools_module
from fused_memory.server.tools import create_mcp_server
from fused_memory.utils.validation import validate_project_id, validate_project_root


def _parse_tool_result(result):
    """Extract the dict from a FastMCP TextContent result or pass-through dict."""
    if isinstance(result, list):
        content = result[0].text if hasattr(result[0], 'text') else str(result[0])
        return json.loads(content)
    return result


class TestToolsDelegateToSharedValidators:
    """Object identity checks proving tools.py imports and re-exports the shared validators.

    If tools.py ever re-introduces private closures for validation, these tests fail
    immediately — the module attribute would no longer be the same object as the
    shared validator imported from utils.validation.
    """

    def test_validate_project_id_is_shared_validator(self):
        """tools.validate_project_id must be the exact same object as utils.validation.validate_project_id.

        tools.py line 14: `from fused_memory.utils.validation import validate_project_id`
        This creates a module-level attribute that is identical (by identity) to the
        shared validator. Any re-implementation would break this check.
        """
        assert tools_module.validate_project_id is validate_project_id, (
            "tools.validate_project_id is not the shared validator from utils.validation. "
            "tools.py must delegate to the shared validator, not re-implement it."
        )

    def test_validate_project_root_is_shared_validator(self):
        """tools.validate_project_root must be the exact same object as utils.validation.validate_project_root."""
        assert tools_module.validate_project_root is validate_project_root, (
            "tools.validate_project_root is not the shared validator from utils.validation. "
            "tools.py must delegate to the shared validator, not re-implement it."
        )


class TestToolsValidationIntegration:
    """Integration tests that invoke MCP handlers through server._tool_manager.call_tool().

    These verify that validation errors propagate correctly through the MCP handler
    layer, not just that the validator functions themselves work. An error dict from
    the validator must flow back to the caller without being swallowed or transformed.
    """

    @pytest.mark.asyncio
    async def test_add_memory_rejects_whitespace_project_id(self):
        """add_memory handler rejects whitespace-only project_id before reaching the service.

        Whitespace-only was the specific bug that motivated the validator consolidation
        in task 253. The old private _validate_project_id used `if not project_id`
        (truthy check only), which passed whitespace-only inputs like '   '.
        This test verifies that the bugfix propagates through the MCP handler layer.
        """
        mock_service = AsyncMock()
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {'content': 'test content', 'project_id': '   '},
        )

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert 'error' in result, f'Expected error key in result: {result!r}'
        assert 'non-empty' in result['error'] or 'whitespace' in result['error'].lower(), (
            f"Expected error message to mention 'non-empty' or 'whitespace', got: {result['error']!r}"
        )
        assert result['error_type'] == 'ValidationError', (
            f"Expected error_type='ValidationError', got: {result['error_type']!r}"
        )
        # Validation must short-circuit before reaching the service
        mock_service.add_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_tasks_rejects_whitespace_project_root(self):
        """get_tasks handler rejects whitespace-only project_root before reaching the task interceptor.

        Validates that the project_root consolidation bugfix propagates through the
        MCP handler layer for task tools, mirroring the project_id test for memory tools.
        """
        mock_service = AsyncMock()
        mock_task_interceptor = AsyncMock()
        server = create_mcp_server(mock_service, task_interceptor=mock_task_interceptor)

        result = await server._tool_manager.call_tool(
            'get_tasks',
            {'project_root': '   '},
        )

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert 'error' in result, f'Expected error key in result: {result!r}'
        assert result['error_type'] == 'ValidationError', (
            f"Expected error_type='ValidationError', got: {result['error_type']!r}"
        )
        # Validation must short-circuit before reaching the task interceptor
        mock_task_interceptor.get_tasks.assert_not_called()


class TestSearchIncludePlannedMCPTool:
    """step-19: MCP search tool accepts and forwards include_planned parameter."""

    @pytest.mark.asyncio
    async def test_include_planned_forwarded_to_service(self):
        """search MCP tool passes include_planned=True to memory_service.search."""
        mock_service = AsyncMock()
        mock_service.search.return_value = []
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'search',
            {
                'query': 'test query',
                'project_id': 'test',
                'include_planned': True,
            },
        )

        assert isinstance(result, dict), f'Expected dict result, got {type(result)}: {result!r}'
        assert 'results' in result, f'Expected results key in response: {result!r}'
        mock_service.search.assert_called_once()
        call_kwargs = mock_service.search.call_args[1]
        assert call_kwargs.get('include_planned') is True, (
            'include_planned=True must be forwarded from MCP search tool to memory_service.search'
        )

    @pytest.mark.asyncio
    async def test_include_planned_defaults_to_false(self):
        """search MCP tool passes include_planned=False when not specified."""
        mock_service = AsyncMock()
        mock_service.search.return_value = []
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'search',
            {
                'query': 'test query',
                'project_id': 'test',
            },
        )

        assert isinstance(result, dict), f'Expected dict result, got {type(result)}: {result!r}'
        assert 'results' in result, f'Expected results key in response: {result!r}'
        mock_service.search.assert_called_once()
        call_kwargs = mock_service.search.call_args[1]
        assert call_kwargs.get('include_planned', False) is False, (
            'include_planned must default to False in MCP search tool'
        )

    @pytest.mark.asyncio
    async def test_include_planned_false_forwarded_to_service(self):
        """search MCP tool passes include_planned=False when explicitly set to False."""
        mock_service = AsyncMock()
        mock_service.search.return_value = []
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'search',
            {
                'query': 'test query',
                'project_id': 'test',
                'include_planned': False,
            },
        )

        assert isinstance(result, dict), f'Expected dict result, got {type(result)}: {result!r}'
        assert 'results' in result, f'Expected results key in response: {result!r}'
        mock_service.search.assert_called_once()
        call_kwargs = mock_service.search.call_args[1]
        assert call_kwargs.get('include_planned') is False, (
            'include_planned=False must be forwarded to memory_service.search'
        )


class TestKnownProjectRegistryGate:
    """MCP write tools gate on known_projects registry when provided (task 1549).

    Mirrors TestToolsValidationIntegration: uses create_mcp_server(AsyncMock()) +
    server._tool_manager.call_tool + assert_not_called() pattern.
    """

    @pytest.mark.asyncio
    async def test_add_memory_rejects_unknown_project_id(self):
        """add_memory is rejected at the MCP boundary when project_id is not in known_projects.

        The service must never be called — the gate short-circuits before any downstream
        logic including the backlog gate.
        """
        mock_service = AsyncMock()
        server = create_mcp_server(
            mock_service,
            known_projects={'dark_factory': '/home/leo/src/dark-factory'},
        )

        result = await server._tool_manager.call_tool(
            'add_memory',
            {'content': 'test content', 'project_id': 'know-live', 'metadata': {}},
        )

        assert isinstance(result, dict), f'Expected error dict, got {type(result)}: {result!r}'
        assert 'error' in result, f'Expected error key in result: {result!r}'
        assert 'DASHBOARD_KNOWN_PROJECT_ROOTS' in result['error'], (
            f"Error must mention DASHBOARD_KNOWN_PROJECT_ROOTS; got: {result['error']!r}"
        )
        assert result.get('error_type') == 'ValidationError'
        mock_service.add_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_memory_passes_for_known_project_id(self):
        """add_memory passes the gate and reaches the service for a known project_id."""
        mock_service = AsyncMock()
        server = create_mcp_server(
            mock_service,
            known_projects={'dark_factory': '/home/leo/src/dark-factory'},
        )

        await server._tool_manager.call_tool(
            'add_memory',
            {'content': 'test content', 'project_id': 'dark_factory', 'metadata': {}},
        )

        mock_service.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_memory_permissive_without_known_projects(self):
        """add_memory is permissive when no known_projects registry is provided."""
        mock_service = AsyncMock()
        server = create_mcp_server(mock_service)  # no known_projects

        await server._tool_manager.call_tool(
            'add_memory',
            {'content': 'test content', 'project_id': 'whatever', 'metadata': {}},
        )

        mock_service.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_episode_rejects_unknown_project_id(self):
        """add_episode is rejected at the MCP boundary for an unknown project_id."""
        mock_service = AsyncMock()
        server = create_mcp_server(
            mock_service,
            known_projects={'dark_factory': '/home/leo/src/dark-factory'},
        )

        result = await server._tool_manager.call_tool(
            'add_episode',
            {'content': 'test content', 'project_id': 'know-live'},
        )

        assert isinstance(result, dict), f'Expected error dict, got {type(result)}: {result!r}'
        assert 'error' in result
        assert 'DASHBOARD_KNOWN_PROJECT_ROOTS' in result['error'], (
            f"Error must mention DASHBOARD_KNOWN_PROJECT_ROOTS; got: {result['error']!r}"
        )
        mock_service.add_episode.assert_not_called()


class TestGetEntityEdgeLimitMCPTool:
    """task 2089 step-3: MCP get_entity tool accepts, validates, and forwards edge_limit."""

    @pytest.mark.asyncio
    async def test_edge_limit_forwarded_to_service(self):
        """get_entity MCP tool forwards edge_limit=25 to memory_service.get_entity."""
        mock_service = AsyncMock()
        mock_service.get_entity.return_value = {'nodes': [], 'edges': []}
        server = create_mcp_server(mock_service)

        await server._tool_manager.call_tool(
            'get_entity',
            {'name': 'entity', 'project_id': 'test', 'edge_limit': 25},
        )

        mock_service.get_entity.assert_called_once()
        call_kwargs = mock_service.get_entity.call_args[1]
        assert call_kwargs.get('edge_limit') == 25, (
            'edge_limit=25 must be forwarded from MCP get_entity tool to memory_service.get_entity'
        )

    @pytest.mark.asyncio
    async def test_edge_limit_defaults_to_ten(self):
        """get_entity MCP tool forwards edge_limit=10 to the service when not specified."""
        mock_service = AsyncMock()
        mock_service.get_entity.return_value = {'nodes': [], 'edges': []}
        server = create_mcp_server(mock_service)

        await server._tool_manager.call_tool(
            'get_entity',
            {'name': 'entity', 'project_id': 'test'},
        )

        mock_service.get_entity.assert_called_once()
        call_kwargs = mock_service.get_entity.call_args[1]
        assert call_kwargs.get('edge_limit') == 10, (
            'edge_limit must default to 10 in MCP get_entity tool'
        )

    @pytest.mark.asyncio
    async def test_edge_limit_zero_returns_validation_error(self):
        """edge_limit=0 is rejected at the MCP boundary before reaching the service."""
        mock_service = AsyncMock()
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'get_entity',
            {'name': 'entity', 'project_id': 'test', 'edge_limit': 0},
        )

        parsed = _parse_tool_result(result)
        assert 'error' in parsed, f'Expected error key in result: {parsed!r}'
        assert parsed.get('error_type') == 'ValidationError', (
            f"Expected error_type='ValidationError', got: {parsed!r}"
        )
        mock_service.get_entity.assert_not_called()

    @pytest.mark.asyncio
    async def test_edge_limit_above_1000_clamped(self):
        """edge_limit=5000 is clamped to 1000 before being forwarded to the service."""
        mock_service = AsyncMock()
        mock_service.get_entity.return_value = {'nodes': [], 'edges': []}
        server = create_mcp_server(mock_service)

        await server._tool_manager.call_tool(
            'get_entity',
            {'name': 'entity', 'project_id': 'test', 'edge_limit': 5000},
        )

        mock_service.get_entity.assert_called_once()
        call_kwargs = mock_service.get_entity.call_args[1]
        assert call_kwargs.get('edge_limit') == 1000, (
            f"edge_limit=5000 must be clamped to 1000, got: {call_kwargs.get('edge_limit')!r}"
        )


class TestSubmitTaskPremiseLintGuard:
    """submit_task premise-lint guard xi (task 2231/W5-xi) — boundary test.

    The unit-level invariant matrix for premise_lint_error lives in
    test_premise_lint_guard.py; these tests assert the guard is WIRED into
    the submit_task tool boundary, mirroring TestToolsValidationIntegration's
    create_mcp_server(mock_service, task_interceptor=mock_task_interceptor) +
    assert_not_called() pattern: a recon-stage caller whose description
    asserts a known-false premise about recon internals is rejected BEFORE
    the interceptor is reached, while a clean description from the same
    caller reaches the interceptor unchanged.
    """

    @pytest.mark.asyncio
    async def test_submit_task_recon_stage_false_premise_rejected(self, monkeypatch):
        """A recon-stage submit_task whose description asserts run_id
        persists across cycles is rejected with a ValidationError naming the
        violated invariant and never reaches the interceptor."""
        # Synthetic project_root ('/project') isn't a real git working tree;
        # stub resolve_main_checkout to pass it through unchanged, mirroring
        # test_task_tools.py's passthrough_main_checkout fixture.
        monkeypatch.setattr(
            'fused_memory.server.tools.resolve_main_checkout', lambda p: str(p),
        )
        mock_service = AsyncMock()
        mock_task_interceptor = AsyncMock()
        server = create_mcp_server(mock_service, task_interceptor=mock_task_interceptor)

        result = await server._tool_manager.call_tool(
            'submit_task',
            {
                'project_root': '/project',
                'prompt': 'Fix the ledger',
                'description': 'Fix the bug where run_id persists across cycles in the ledger.',
                'agent_id': 'recon-stage-task_knowledge_sync',
                'metadata': {'execution_class': 'code_tdd'},
            },
        )

        parsed = _parse_tool_result(result)
        assert parsed.get('error_type') == 'ValidationError', f'got {parsed!r}'
        assert 'run_id_is_fresh_per_run' in parsed.get('error', ''), f'got {parsed!r}'
        mock_task_interceptor.submit_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_submit_task_recon_stage_false_premise_in_details_rejected(self, monkeypatch):
        """A recon-stage submit_task whose `details` (not `description` or
        `prompt`) asserts a known-false premise is rejected before the
        interceptor is reached. `details` is submit_task's own
        docstring-named "Task details / implementation notes" field, which
        recon-authored remediation tasks commonly use to carry substantial
        mechanism prose — linting only description/prompt would leave this
        channel unchecked (reviewer_comprehensive: robustness_coverage_gap)."""
        monkeypatch.setattr(
            'fused_memory.server.tools.resolve_main_checkout', lambda p: str(p),
        )
        mock_service = AsyncMock()
        mock_task_interceptor = AsyncMock()
        server = create_mcp_server(mock_service, task_interceptor=mock_task_interceptor)

        result = await server._tool_manager.call_tool(
            'submit_task',
            {
                'project_root': '/project',
                'prompt': 'Reconcile task 7',
                'description': 'Reconcile task 7 status against the knowledge graph.',
                'details': 'Implementation note: run_id persists across cycles in this subsystem.',
                'agent_id': 'recon-stage-task_knowledge_sync',
                'metadata': {'execution_class': 'code_tdd'},
            },
        )

        parsed = _parse_tool_result(result)
        assert parsed.get('error_type') == 'ValidationError', f'got {parsed!r}'
        assert 'run_id_is_fresh_per_run' in parsed.get('error', ''), f'got {parsed!r}'
        mock_task_interceptor.submit_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_submit_task_recon_stage_false_premise_in_title_rejected(self, monkeypatch):
        """A recon-stage submit_task whose `title` asserts a known-false
        premise is rejected before the interceptor is reached."""
        monkeypatch.setattr(
            'fused_memory.server.tools.resolve_main_checkout', lambda p: str(p),
        )
        mock_service = AsyncMock()
        mock_task_interceptor = AsyncMock()
        server = create_mcp_server(mock_service, task_interceptor=mock_task_interceptor)

        result = await server._tool_manager.call_tool(
            'submit_task',
            {
                'project_root': '/project',
                'prompt': 'Reconcile task 7',
                'title': 'Bug: run_id persists across cycles',
                'description': 'Reconcile task 7 status against the knowledge graph.',
                'agent_id': 'recon-stage-task_knowledge_sync',
                'metadata': {'execution_class': 'code_tdd'},
            },
        )

        parsed = _parse_tool_result(result)
        assert parsed.get('error_type') == 'ValidationError', f'got {parsed!r}'
        assert 'run_id_is_fresh_per_run' in parsed.get('error', ''), f'got {parsed!r}'
        mock_task_interceptor.submit_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_submit_task_recon_stage_clean_description_accepted(self, monkeypatch):
        """A recon-stage submit_task with a clean description (no known
        false premise) is accepted and reaches the interceptor — premise-lint
        does not block clean recon submits."""
        monkeypatch.setattr(
            'fused_memory.server.tools.resolve_main_checkout', lambda p: str(p),
        )
        mock_service = AsyncMock()
        mock_task_interceptor = AsyncMock()
        mock_task_interceptor.submit_task.return_value = {'ticket': 'tkt_x'}
        server = create_mcp_server(mock_service, task_interceptor=mock_task_interceptor)

        result = await server._tool_manager.call_tool(
            'submit_task',
            {
                'project_root': '/project',
                'prompt': 'Reconcile task 7',
                'description': 'Reconcile task 7 status against the knowledge graph.',
                'agent_id': 'recon-stage-task_knowledge_sync',
                'metadata': {'execution_class': 'code_tdd'},
            },
        )

        parsed = _parse_tool_result(result)
        assert 'error' not in parsed, f'got {parsed!r}'
        mock_task_interceptor.submit_task.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_submit_task_recon_stage_false_premise_in_prompt_rejected(self, monkeypatch):
        """A recon-stage submit_task whose `prompt` (not `description`)
        asserts a known-false premise is rejected before the interceptor is
        reached. `prompt` is submit_task's own docstring-named primary
        "Task description for AI generation" field, so linting
        `description` alone would leave a false premise stated only here
        unchecked (the coverage gap this guard closes)."""
        monkeypatch.setattr(
            'fused_memory.server.tools.resolve_main_checkout', lambda p: str(p),
        )
        mock_service = AsyncMock()
        mock_task_interceptor = AsyncMock()
        server = create_mcp_server(mock_service, task_interceptor=mock_task_interceptor)

        result = await server._tool_manager.call_tool(
            'submit_task',
            {
                'project_root': '/project',
                'prompt': 'Fix the bug where run_id persists across cycles in the ledger.',
                'description': 'Reconcile task 7 status against the knowledge graph.',
                'agent_id': 'recon-stage-task_knowledge_sync',
                'metadata': {'execution_class': 'code_tdd'},
            },
        )

        parsed = _parse_tool_result(result)
        assert parsed.get('error_type') == 'ValidationError', f'got {parsed!r}'
        assert 'run_id_is_fresh_per_run' in parsed.get('error', ''), f'got {parsed!r}'
        mock_task_interceptor.submit_task.assert_not_called()
