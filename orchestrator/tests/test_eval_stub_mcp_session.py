"""Tests for _StubMcpSession and _build_eval_scheduler in evals/runner.py."""

from __future__ import annotations

import json

import pytest

from orchestrator.evals.runner import _StubMcpSession, _build_eval_scheduler


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


class TestStubMcpSessionGetTasks:
    """Tests for _StubMcpSession.call_tool('get_tasks', ...)."""

    @pytest.mark.asyncio
    async def test_get_tasks_returns_empty_list(self):
        """get_tasks returns an envelope whose decoded text has tasks: []."""
        stub = _StubMcpSession()
        result = await stub.call_tool('get_tasks', {})
        block = result['result']['content'][0]
        decoded = json.loads(block['text'])
        assert decoded['tasks'] == []

    @pytest.mark.asyncio
    async def test_get_tasks_envelope_shape(self):
        """get_tasks returns a correctly-shaped JSON-RPC envelope."""
        stub = _StubMcpSession()
        result = await stub.call_tool('get_tasks', {})
        assert result['jsonrpc'] == '2.0'
        assert isinstance(result['id'], int)
        content = result['result']['content']
        assert isinstance(content, list) and len(content) >= 1
        assert content[0]['type'] == 'text'


class TestStubMcpSessionUpdateTask:
    """Tests for _StubMcpSession.call_tool('update_task', ...)."""

    @pytest.mark.asyncio
    async def test_update_task_returns_non_error_envelope(self):
        """update_task returns a non-error envelope so Scheduler.update_task returns True.

        Scheduler.update_task checks result.get('result', result).get('isError').
        No isError key means success → True.
        """
        stub = _StubMcpSession()
        result = await stub.call_tool('update_task', {'id': 'task-20', 'metadata': '{}'})
        # The envelope must not have isError set
        content_wrapper = result.get('result', result)
        assert not content_wrapper.get('isError')
        # Decoded text echoes back the id
        block = result['result']['content'][0]
        decoded = json.loads(block['text'])
        assert decoded['id'] == 'task-20'

    @pytest.mark.asyncio
    async def test_update_task_envelope_shape(self):
        """update_task returns a correctly-shaped JSON-RPC envelope."""
        stub = _StubMcpSession()
        result = await stub.call_tool('update_task', {'id': 'task-21', 'metadata': '{}'})
        assert result['jsonrpc'] == '2.0'
        assert isinstance(result['id'], int)
        content = result['result']['content']
        assert isinstance(content, list) and len(content) >= 1
        assert content[0]['type'] == 'text'


class TestStubMcpSessionUnknownTool:
    """Tests for _StubMcpSession.call_tool with an unknown tool name."""

    @pytest.mark.asyncio
    async def test_unknown_tool_raises_not_implemented_error(self):
        """Calling an unknown tool name raises NotImplementedError."""
        stub = _StubMcpSession()
        with pytest.raises(NotImplementedError):
            await stub.call_tool('some_unknown_tool', {})

    @pytest.mark.asyncio
    async def test_unknown_tool_message_names_the_tool(self):
        """The NotImplementedError message names the unknown tool."""
        stub = _StubMcpSession()
        with pytest.raises(NotImplementedError, match='some_unknown_tool'):
            await stub.call_tool('some_unknown_tool', {})


# ---------------------------------------------------------------------------
# Tests for _build_eval_scheduler helper
# ---------------------------------------------------------------------------


class TestBuildEvalScheduler:
    """Tests for _build_eval_scheduler(orch_config, task_id, modules) helper."""

    def test_returns_scheduler_instance(self):
        """Returns a production Scheduler (not _EvalScheduler)."""
        from orchestrator.config import OrchestratorConfig
        from orchestrator.scheduler import Scheduler

        config = OrchestratorConfig()
        scheduler, _ = _build_eval_scheduler(config, 'task-99', ['some_module'])
        assert isinstance(scheduler, Scheduler)

    def test_returns_stub_mcp_session(self):
        """Returns a _StubMcpSession as the second element."""
        from orchestrator.config import OrchestratorConfig

        config = OrchestratorConfig()
        _, session = _build_eval_scheduler(config, 'task-99', ['some_module'])
        assert isinstance(session, _StubMcpSession)

    def test_di_wired_mcp_session(self):
        """scheduler._mcp_session is the returned session (DI wired)."""
        from orchestrator.config import OrchestratorConfig

        config = OrchestratorConfig()
        scheduler, session = _build_eval_scheduler(config, 'task-99', ['some_module'])
        assert scheduler._mcp_session is session

    def test_lock_preinstalled_for_task(self):
        """The module lock is pre-installed for task_id in lock_table._held."""
        from orchestrator.config import OrchestratorConfig
        from shared.locking import normalize_lock

        config = OrchestratorConfig()
        scheduler, _ = _build_eval_scheduler(config, 'task-99', ['some_module'])
        assert 'task-99' in scheduler.lock_table._held
        expected_module = normalize_lock('some_module', config.lock_depth)
        assert expected_module in scheduler.lock_table._held['task-99']

    @pytest.mark.asyncio
    async def test_handle_blast_radius_expansion_returns_true(self):
        """handle_blast_radius_expansion returns True without raising.

        This is the end-to-end sanity check: DI + lock-preinstall means the
        production code path for plan refinement works in eval mode.
        """
        from orchestrator.config import OrchestratorConfig

        config = OrchestratorConfig()
        scheduler, _ = _build_eval_scheduler(
            config, 'task-99', ['some_module']
        )
        result = await scheduler.handle_blast_radius_expansion(
            'task-99',
            ['some_module'],
            ['some_module', 'other_module'],
        )
        assert result is True
