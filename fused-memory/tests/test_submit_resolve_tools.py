"""Tests for the submit_task and resolve_ticket MCP tool registrations."""

import json
import logging
import os
from unittest.mock import AsyncMock

import pytest

from fused_memory.server.tools import create_mcp_server


@pytest.fixture(autouse=True)
def passthrough_main_checkout(monkeypatch):
    """Stub resolve_main_checkout to pass its argument through unchanged.

    These tests use synthetic project_root values like ``/project`` that
    aren't real git working trees.
    """
    monkeypatch.setattr(
        'fused_memory.server.tools.resolve_main_checkout', lambda p: str(p),
    )


@pytest.fixture
def task_interceptor():
    ti = AsyncMock()
    ti.submit_task = AsyncMock(return_value={'ticket': 'tkt_ABCDEFGHIJKLMNOPQRSTUVWXYZ'})
    ti.resolve_ticket = AsyncMock(return_value={'status': 'created', 'task_id': '5'})
    ti.cancel_ticket = AsyncMock(return_value={'status': 'cancelled', 'ticket_id': 'tkt_X'})
    return ti


@pytest.fixture
def mcp_server(task_interceptor):
    """MCP server with a mocked task interceptor."""
    mock_service = AsyncMock()
    return create_mcp_server(mock_service, task_interceptor=task_interceptor)


# ------------------------------------------------------------------
# submit_task MCP tool
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_task_mcp_tool_signature_and_forwarding(mcp_server, task_interceptor):
    """submit_task MCP tool forwards all documented args to interceptor.submit_task
    and the returned {ticket: ...} dict flows back unchanged.
    """
    result = await mcp_server._tool_manager.call_tool(
        'submit_task',
        {
            'project_root': '/project',
            'title': 'My Task',
            'description': 'A description',
            'details': 'Some details',
            'dependencies': '1,2',
            'priority': 'high',
            'metadata': {'escalation_id': 'e1'},
            'tag': 'mytag',
        },
    )

    task_interceptor.submit_task.assert_called_once()
    call_kwargs = task_interceptor.submit_task.call_args.kwargs
    assert call_kwargs['project_root'] == '/project'
    assert call_kwargs.get('title') == 'My Task'
    assert call_kwargs.get('description') == 'A description'
    assert call_kwargs.get('details') == 'Some details'
    assert call_kwargs.get('dependencies') == '1,2'
    assert call_kwargs.get('priority') == 'high'
    assert call_kwargs.get('tag') == 'mytag'

    # Return value flows back unchanged.
    assert result == {'ticket': 'tkt_ABCDEFGHIJKLMNOPQRSTUVWXYZ'}


@pytest.mark.asyncio
async def test_submit_task_routing_override_reason_forwarded(mcp_server, task_interceptor):
    """submit_task MCP tool forwards routing_override_reason to interceptor.submit_task.

    Two cases:
    - When supplied, the exact value is forwarded as a kwarg.
    - When omitted, an empty string '' is forwarded (the declared default).
    """
    # Case 1: routing_override_reason supplied
    task_interceptor.submit_task.reset_mock()
    await mcp_server._tool_manager.call_tool(
        'submit_task',
        {
            'project_root': '/project',
            'title': 'My Task',
            'routing_override_reason': 'owner asserted',
        },
    )
    call_kwargs = task_interceptor.submit_task.call_args.kwargs
    assert call_kwargs.get('routing_override_reason') == 'owner asserted', (
        f"Expected routing_override_reason='owner asserted', got: {call_kwargs.get('routing_override_reason')!r}"
    )

    # Case 2: routing_override_reason omitted → default '' forwarded
    task_interceptor.submit_task.reset_mock()
    await mcp_server._tool_manager.call_tool(
        'submit_task',
        {
            'project_root': '/project',
            'title': 'My Task',
        },
    )
    call_kwargs = task_interceptor.submit_task.call_args.kwargs
    assert call_kwargs.get('routing_override_reason') == '', (
        f"Expected routing_override_reason='' when omitted, got: {call_kwargs.get('routing_override_reason')!r}"
    )


# ------------------------------------------------------------------
# resolve_ticket MCP tool
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_ticket_mcp_tool_signature_and_forwarding(mcp_server, task_interceptor):
    """resolve_ticket MCP tool forwards args to interceptor.resolve_ticket
    and the returned {status, task_id} dict flows back unchanged.
    """
    result = await mcp_server._tool_manager.call_tool(
        'resolve_ticket',
        {
            'ticket': 'tkt_ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            'project_root': '/project',
            'timeout_seconds': 10,
        },
    )

    task_interceptor.resolve_ticket.assert_called_once()
    call_kwargs = task_interceptor.resolve_ticket.call_args.kwargs
    assert call_kwargs.get('ticket') == 'tkt_ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    assert call_kwargs.get('project_root') == '/project'
    assert call_kwargs.get('timeout_seconds') == 10

    # Return value flows back unchanged.
    assert result == {'status': 'created', 'task_id': '5'}


@pytest.mark.asyncio
async def test_resolve_ticket_mcp_tool_rejects_non_prefixed_id(mcp_server, task_interceptor):
    """resolve_ticket MCP tool returns ValidationError for inputs that are
    neither ``tkt_…`` tickets nor numeric task ids.

    Numeric inputs (``'42'``, ``42``) are intentionally short-circuited as
    ``idempotent_passthrough`` since planning_mode landed — see
    test_task_tools.py for that coverage. This test pins the rejection
    path for everything else.
    """
    result = await mcp_server._tool_manager.call_tool(
        'resolve_ticket',
        {
            'ticket': 'not-a-ticket',
            'project_root': '/project',
        },
    )

    assert result.get('error_type') == 'ValidationError', (
        f'Expected ValidationError, got: {result}'
    )
    assert 'tkt_' in result.get('error', ''), (
        f'Error message should mention tkt_: {result}'
    )
    # Interceptor must NOT be called when the ticket id is invalid.
    task_interceptor.resolve_ticket.assert_not_called()


# ------------------------------------------------------------------
# cancel_ticket MCP tool
# ------------------------------------------------------------------



# ------------------------------------------------------------------
# deterministic task_kind guard — MCP-boundary integration (B10)
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deterministic_no_before_done_no_always_escalates_rejects(mcp_server, task_interceptor, tmp_path):
    """B10: task_kind='deterministic' with no before_done and no always_escalates → validation error naming the invariant."""
    result = await mcp_server._tool_manager.call_tool(
        'submit_task',
        {
            'project_root': str(tmp_path),
            'title': 'Gate task',
            'task_kind': 'deterministic',
        },
    )
    assert 'error' in result, f'Expected error, got: {result!r}'
    assert result.get('error_type') == 'ValidationError', f'Expected ValidationError: {result!r}'
    msg = result['error']
    assert 'ill-formed no-op' in msg or 'must run an action' in msg or 'always escalate' in msg, (
        f'Error message should name the no-op invariant: {msg!r}'
    )
    task_interceptor.submit_task.assert_not_called()


@pytest.mark.asyncio
async def test_normal_with_before_done_in_metadata_rejects(mcp_server, task_interceptor, tmp_path):
    """task_kind='normal' + metadata.before_done → validation error (before_done only on deterministic)."""
    result = await mcp_server._tool_manager.call_tool(
        'submit_task',
        {
            'project_root': str(tmp_path),
            'title': 'Bad task',
            'task_kind': 'normal',
            'metadata': {'before_done': {'script': 'run.sh', 'timeout_secs': 60}},
        },
    )
    assert 'error' in result
    assert result.get('error_type') == 'ValidationError'
    assert 'before_done' in result['error']
    assert 'deterministic' in result['error']
    task_interceptor.submit_task.assert_not_called()


@pytest.mark.asyncio
async def test_deterministic_always_escalates_forwarded_with_task_kind_injected(mcp_server, task_interceptor, tmp_path):
    """task_kind='deterministic' + always_escalates=True → accepted; forwarded metadata has task_kind='deterministic'."""
    await mcp_server._tool_manager.call_tool(
        'submit_task',
        {
            'project_root': str(tmp_path),
            'title': 'Gate task',
            'task_kind': 'deterministic',
            'metadata': {'always_escalates': True},
        },
    )
    task_interceptor.submit_task.assert_called_once()
    forwarded_meta = task_interceptor.submit_task.call_args.kwargs['metadata']
    if isinstance(forwarded_meta, str):
        forwarded_meta = json.loads(forwarded_meta)
    assert forwarded_meta.get('task_kind') == 'deterministic'
    assert forwarded_meta.get('always_escalates') is True


@pytest.mark.asyncio
async def test_valid_deploy_before_done_forwarded(mcp_server, task_interceptor, tmp_path):
    """Valid deploy: real chmod+x script + timeout_secs → accepted; forwarded metadata preserves before_done and task_kind."""
    script = tmp_path / 'deploy.sh'
    script.write_text('#!/bin/sh\necho deploy\n')
    os.chmod(script, 0o755)

    await mcp_server._tool_manager.call_tool(
        'submit_task',
        {
            'project_root': str(tmp_path),
            'title': 'Deploy task',
            'task_kind': 'deterministic',
            'metadata': {'before_done': {'script': 'deploy.sh', 'timeout_secs': 60}},
        },
    )
    task_interceptor.submit_task.assert_called_once()
    forwarded_meta = task_interceptor.submit_task.call_args.kwargs['metadata']
    if isinstance(forwarded_meta, str):
        forwarded_meta = json.loads(forwarded_meta)
    assert forwarded_meta.get('task_kind') == 'deterministic'
    assert forwarded_meta.get('before_done', {}).get('script') == 'deploy.sh'


@pytest.mark.asyncio
async def test_default_task_kind_normal_injected(mcp_server, task_interceptor, tmp_path):
    """Default (task_kind omitted) → forwarded metadata has task_kind='normal'."""
    await mcp_server._tool_manager.call_tool(
        'submit_task',
        {
            'project_root': str(tmp_path),
            'title': 'Ordinary task',
        },
    )
    task_interceptor.submit_task.assert_called_once()
    forwarded_meta = task_interceptor.submit_task.call_args.kwargs['metadata']
    if isinstance(forwarded_meta, str):
        forwarded_meta = json.loads(forwarded_meta)
    if forwarded_meta is None:
        forwarded_meta = {}
    assert forwarded_meta.get('task_kind') == 'normal', (
        f"Expected task_kind='normal', got: {forwarded_meta.get('task_kind')!r}"
    )


# ------------------------------------------------------------------
# cancel_ticket MCP tool
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_ticket_mcp_tool_delegates_to_interceptor(mcp_server, task_interceptor):
    """cancel_ticket MCP tool forwards ticket_id to interceptor.cancel_ticket
    and the returned dict flows back unchanged.

    RED in step-7: the MCP tool is not yet registered — call_tool raises ToolNotFound.
    """
    result = await mcp_server._tool_manager.call_tool(
        'cancel_ticket',
        {'ticket_id': 'tkt_X'},
    )

    task_interceptor.cancel_ticket.assert_called_once_with(ticket_id='tkt_X')
    # Return value flows back unchanged.
    assert result == {'status': 'cancelled', 'ticket_id': 'tkt_X'}, (
        f'Expected cancelled dict, got: {result!r}'
    )


# ------------------------------------------------------------------
# routing-intent lint guard — MCP-boundary integration (task 2563)
#
# The unit-level detection/payload/flag matrix for routing_intent_finding /
# routing_intent_reject / routing_intent_warning / routing_intent_enforced
# lives in test_routing_intent_guard.py; these tests assert the guard is
# WIRED into the submit_task tool boundary, covering both the curator path
# and the planning_mode path (a single guard placement before the one
# task_interceptor.submit_task call covers both — see
# routing_intent_guard.py's module docstring). RED: the guard is not yet
# wired into tools.py.
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_routing_intent_warn_mode_flags_and_still_submits(
    mcp_server, task_interceptor, monkeypatch, caplog,
):
    """WARN mode (FUSED_ROUTING_INTENT_ENFORCE unset) + a marker-bearing
    task_kind='normal' submission -> result carries 'routing_intent_warning',
    the interceptor is still called (submission proceeds unblocked), and a
    'routing_intent_lint.flagged' WARNING is logged (the census line)."""
    monkeypatch.delenv('FUSED_ROUTING_INTENT_ENFORCE', raising=False)
    with caplog.at_level(logging.WARNING):
        result = await mcp_server._tool_manager.call_tool(
            'submit_task',
            {
                'project_root': '/project',
                'title': 'DO NOT IMPLEMENT this; escalate to a human instead of implementing.',
                'task_kind': 'normal',
            },
        )
    assert 'routing_intent_warning' in result, f'Expected warning payload, got: {result!r}'
    task_interceptor.submit_task.assert_called_once()
    assert any('routing_intent_lint.flagged' in rec.message for rec in caplog.records), (
        f'Expected a routing_intent_lint.flagged census WARNING, got records: {caplog.records!r}'
    )


@pytest.mark.asyncio
async def test_routing_intent_passing_mention_not_flagged(mcp_server, task_interceptor, monkeypatch):
    """A genuine code task whose prose merely mentions routing-adjacent
    terminology in passing (task-2408 shape) -> no 'routing_intent_warning',
    interceptor called normally (no false positive at the wiring level)."""
    monkeypatch.delenv('FUSED_ROUTING_INTENT_ENFORCE', raising=False)
    result = await mcp_server._tool_manager.call_tool(
        'submit_task',
        {
            'project_root': '/project',
            'title': 'Refactor the deterministic pure-gate helper',
            'description': 'Implement the fix in X.',
            'task_kind': 'normal',
        },
    )
    assert 'routing_intent_warning' not in result
    task_interceptor.submit_task.assert_called_once()


@pytest.mark.asyncio
async def test_routing_intent_reject_mode_blocks_submission(mcp_server, task_interceptor, monkeypatch):
    """REJECT mode (FUSED_ROUTING_INTENT_ENFORCE=1) + a marker-bearing
    task_kind='normal' submission -> ValidationError, interceptor never
    reached."""
    monkeypatch.setenv('FUSED_ROUTING_INTENT_ENFORCE', '1')
    result = await mcp_server._tool_manager.call_tool(
        'submit_task',
        {
            'project_root': '/project',
            'title': 'DO NOT IMPLEMENT this; escalate to a human instead of implementing.',
            'task_kind': 'normal',
        },
    )
    assert result.get('error_type') == 'ValidationError', f'Expected ValidationError, got: {result!r}'
    task_interceptor.submit_task.assert_not_called()


@pytest.mark.asyncio
async def test_routing_intent_warning_merged_into_planning_mode_result(
    mcp_server, task_interceptor, monkeypatch,
):
    """planning_mode=True + a marker-bearing submission in WARN mode ->
    'routing_intent_warning' is merged into the planning_mode result dict,
    proving the guard placement covers the planning_mode path (which
    bypasses the curator-side guards 1898/2225/2085) via the same single
    wiring point used for the curator path."""
    monkeypatch.delenv('FUSED_ROUTING_INTENT_ENFORCE', raising=False)
    task_interceptor.submit_task = AsyncMock(
        return_value={'task_id': '5', 'status': 'deferred', 'planning_mode': True}
    )
    result = await mcp_server._tool_manager.call_tool(
        'submit_task',
        {
            'project_root': '/project',
            'title': 'DO NOT IMPLEMENT this; escalate to a human instead of implementing.',
            'task_kind': 'normal',
            'planning_mode': True,
        },
    )
    assert result.get('task_id') == '5'
    assert result.get('status') == 'deferred'
    assert result.get('planning_mode') is True
    assert 'routing_intent_warning' in result, f'Expected warning payload, got: {result!r}'
