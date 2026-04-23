"""Tests for _StubMcpSession and _build_eval_scheduler in evals/runner.py."""

from __future__ import annotations

import json

import pytest

from orchestrator.evals.runner import _StubMcpSession


class TestStubMcpSessionSetTaskStatus:
    """Tests for _StubMcpSession.call_tool('set_task_status', ...)."""

    @pytest.mark.asyncio
    async def test_set_status_stores_status(self):
        """set_task_status stores the status for the given task_id."""
        stub = _StubMcpSession()
        await stub.call_tool('set_task_status', {'id': 'task-1', 'status': 'in-progress'})
        assert stub._statuses['task-1'] == 'in-progress'

    @pytest.mark.asyncio
    async def test_set_status_accepts_done_provenance(self):
        """set_task_status accepts optional done_provenance without raising."""
        stub = _StubMcpSession()
        await stub.call_tool(
            'set_task_status',
            {
                'id': 'task-2',
                'status': 'done',
                'done_provenance': {'commit': 'abc123'},
            },
        )
        assert stub._statuses['task-2'] == 'done'

    @pytest.mark.asyncio
    async def test_set_status_accepts_reopen_reason(self):
        """set_task_status accepts optional reopen_reason without raising."""
        stub = _StubMcpSession()
        await stub.call_tool(
            'set_task_status',
            {
                'id': 'task-3',
                'status': 'pending',
                'reopen_reason': 'un-defer script',
            },
        )
        assert stub._statuses['task-3'] == 'pending'

    @pytest.mark.asyncio
    async def test_set_status_returns_jsonrpc_envelope(self):
        """set_task_status returns a correctly-shaped JSON-RPC envelope."""
        stub = _StubMcpSession()
        result = await stub.call_tool(
            'set_task_status', {'id': 'task-4', 'status': 'done'}
        )
        # Top-level envelope keys
        assert result['jsonrpc'] == '2.0'
        assert isinstance(result['id'], int)
        assert 'result' in result

        # Content block structure
        content = result['result']['content']
        assert isinstance(content, list)
        assert len(content) >= 1
        block = content[0]
        assert block['type'] == 'text'
        assert isinstance(block['text'], str)

        # Decoded text carries {id, status}
        decoded = json.loads(block['text'])
        assert decoded['id'] == 'task-4'
        assert decoded['status'] == 'done'

    @pytest.mark.asyncio
    async def test_set_status_request_id_increments(self):
        """Each call_tool call increments the request ID."""
        stub = _StubMcpSession()
        r1 = await stub.call_tool('set_task_status', {'id': 'x', 'status': 'done'})
        r2 = await stub.call_tool('set_task_status', {'id': 'y', 'status': 'done'})
        assert r2['id'] > r1['id']


class TestStubMcpSessionGetTask:
    """Tests for _StubMcpSession.call_tool('get_task', ...)."""

    @pytest.mark.asyncio
    async def test_get_task_no_status_returns_no_status_key(self):
        """When no status has been set, decoded text has no 'status' key (or None).

        The Scheduler.get_status parses content[0].text via json.loads, then
        unwraps the Taskmaster envelope (data key), then reads .get('status').
        When status is absent, get_status must return None.
        """
        stub = _StubMcpSession()
        result = await stub.call_tool('get_task', {'id': 'unseen-task'})
        block = result['result']['content'][0]
        decoded = json.loads(block['text'])
        # Either no 'status' key or status is None
        assert decoded.get('status') is None

    @pytest.mark.asyncio
    async def test_get_task_after_set_returns_status(self):
        """After a prior set_task_status, get_task reports the stored status."""
        stub = _StubMcpSession()
        await stub.call_tool('set_task_status', {'id': 'task-10', 'status': 'in-progress'})
        result = await stub.call_tool('get_task', {'id': 'task-10'})
        block = result['result']['content'][0]
        decoded = json.loads(block['text'])
        assert decoded['id'] == 'task-10'
        assert decoded['status'] == 'in-progress'

    @pytest.mark.asyncio
    async def test_get_task_envelope_shape(self):
        """get_task returns a correctly-shaped JSON-RPC envelope."""
        stub = _StubMcpSession()
        result = await stub.call_tool('get_task', {'id': 'task-11'})
        assert result['jsonrpc'] == '2.0'
        assert isinstance(result['id'], int)
        content = result['result']['content']
        assert isinstance(content, list) and len(content) >= 1
        assert content[0]['type'] == 'text'
