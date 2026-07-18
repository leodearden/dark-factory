"""Tests for _StubMcpSession and _build_eval_scheduler in evals/runner.py."""

from __future__ import annotations

import json
import logging

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.evals.runner import _build_eval_scheduler, _StubMcpSession
from orchestrator.scheduler import extract_rejection
from shared.mcp_envelope import parse_tool_result


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


class TestStubMcpSessionSetTaskClaimant:
    """Tests for _StubMcpSession.call_tool('set_task_claimant', ...).

    ``set_task_claimant`` is a REAL dispatched tool: ``Scheduler.set_task_claimant``
    heartbeats via ``dispatch_tool('set_task_claimant', ...)`` every ~60s, so a
    >=90s eval run exercises it.  Without a stub branch each heartbeat raised
    ``NotImplementedError`` (swallowed + logged as a best-effort WARNING) — the
    B9 warning spam this branch removes.
    """

    @pytest.mark.asyncio
    async def test_set_task_claimant_returns_clean_envelope(self):
        """set_task_claimant returns a well-formed, non-rejection envelope.

        The response must (a) be a correctly-shaped JSON-RPC envelope and
        (b) parse to a clean payload so ``extract_rejection`` (the exact parse
        the Scheduler applies to the response) returns None — i.e. the heartbeat
        is treated as success, not a rejection.
        """
        stub = _StubMcpSession()
        resp = await stub.call_tool(
            'set_task_claimant',
            {
                'id': 't1',
                'project_root': '/p',
                'claimant_run_id': 'r1',
                'heartbeat_at': '2026-01-01T00:00:00Z',
            },
        )
        # Envelope shape.
        assert resp['jsonrpc'] == '2.0'
        assert isinstance(resp['id'], int)
        content = resp['result']['content']
        assert isinstance(content, list) and len(content) >= 1
        assert content[0]['type'] == 'text'
        # The Scheduler parses this exact response with extract_rejection;
        # a clean payload (no 'error'/'success:False') must yield None.
        assert extract_rejection(resp) is None

    @pytest.mark.asyncio
    async def test_scheduler_set_task_claimant_logs_no_warning(self, caplog):
        """B9: a real Scheduler.set_task_claimant heartbeat logs ZERO warnings.

        Builds a production Scheduler wired to the stub via _build_eval_scheduler,
        then invokes the same status-untouching heartbeat path the eval loop runs
        every ~60s.  With the stub branch present, dispatch succeeds and
        extract_rejection returns None, so no 'set_task_claimant'/'unknown tool'
        WARNING is emitted (the B9 heartbeat-spam regression).
        """
        scheduler, _ = _build_eval_scheduler(
            OrchestratorConfig(), 'task-99', ['some_module']
        )
        caplog.set_level(logging.WARNING)
        await scheduler.set_task_claimant(
            'task-99',
            claimant_run_id='r1',
            heartbeat_at='2026-01-01T00:00:00Z',
        )
        warnings = [
            r.getMessage()
            for r in caplog.records
            if r.levelno >= logging.WARNING
        ]
        offenders = [
            m for m in warnings
            if 'set_task_claimant' in m or 'unknown tool' in m
        ]
        assert not offenders, (
            'set_task_claimant heartbeat emitted WARNING(s) (B9 regression):\n'
            + '\n'.join(offenders)
        )


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


class TestStubMcpSessionGetStatuses:
    """Tests for _StubMcpSession.call_tool('get_statuses', ...)."""

    @pytest.mark.asyncio
    async def test_get_statuses_returns_empty_mapping_when_no_tasks_set(self):
        """Fresh stub: get_statuses returns envelope text that parses to {'statuses': {}}."""
        stub = _StubMcpSession()
        result = await stub.call_tool('get_statuses', {'project_root': '/p'})
        block = result['result']['content'][0]
        decoded = json.loads(block['text'])
        assert decoded == {'statuses': {}}

    @pytest.mark.asyncio
    async def test_get_statuses_reflects_prior_set_task_status(self):
        """After set_task_status('7', 'in-progress'), get_statuses returns {'statuses': {'7':'in-progress'}}."""
        stub = _StubMcpSession()
        await stub.call_tool('set_task_status', {'id': '7', 'status': 'in-progress'})
        result = await stub.call_tool('get_statuses', {'project_root': '/p'})
        block = result['result']['content'][0]
        decoded = json.loads(block['text'])
        assert decoded == {'statuses': {'7': 'in-progress'}}

    @pytest.mark.asyncio
    async def test_get_statuses_filters_by_ids_argument(self):
        """When ids=['a'] is passed, only 'a' appears in statuses (not other seeded ids)."""
        stub = _StubMcpSession()
        await stub.call_tool('set_task_status', {'id': 'a', 'status': 'done'})
        await stub.call_tool('set_task_status', {'id': 'b', 'status': 'pending'})
        result = await stub.call_tool('get_statuses', {'project_root': '/p', 'ids': ['a']})
        block = result['result']['content'][0]
        decoded = json.loads(block['text'])
        assert decoded == {'statuses': {'a': 'done'}}
        assert 'b' not in decoded['statuses']


class TestStubMcpSessionGetExternalStatuses:
    """Tests for _StubMcpSession.call_tool('get_external_statuses', ...).

    ``get_external_statuses`` is a REAL dispatched tool (cross-project dep
    resolution at scheduler.py; it landed AFTER the PRD was written).  The
    Scheduler parses the response via ``parse_tool_result(result, None, dict)``
    (whole-inner-dict mode): a FLAT ``{dep: status}`` dict, no 'statuses' wrapper,
    every requested dep key present.
    """

    @pytest.mark.asyncio
    async def test_get_external_statuses_keys_each_requested_dep(self):
        """Each requested dep is echoed with a status string (whole-inner-dict)."""
        stub = _StubMcpSession()
        resp = await stub.call_tool(
            'get_external_statuses', {'deps': ['dark_factory:42']}
        )
        # Parsed exactly as the Scheduler parses it (key=None whole-inner-dict).
        parsed, err = parse_tool_result(resp, None, dict)
        assert err is None
        assert isinstance(parsed, dict)
        assert 'dark_factory:42' in parsed
        assert isinstance(parsed['dark_factory:42'], str)

    @pytest.mark.asyncio
    async def test_get_external_statuses_no_deps_returns_empty_dict(self):
        """No 'deps' arg → the payload parses to a flat empty dict (not an error)."""
        stub = _StubMcpSession()
        resp = await stub.call_tool('get_external_statuses', {})
        parsed, err = parse_tool_result(resp, None, dict)
        assert err is None
        assert parsed == {}

    @pytest.mark.asyncio
    async def test_get_external_statuses_envelope_shape(self):
        """get_external_statuses returns a correctly-shaped JSON-RPC envelope."""
        stub = _StubMcpSession()
        resp = await stub.call_tool(
            'get_external_statuses', {'deps': ['p:1']}
        )
        assert resp['jsonrpc'] == '2.0'
        assert isinstance(resp['id'], int)
        content = resp['result']['content']
        assert isinstance(content, list) and len(content) >= 1
        assert content[0]['type'] == 'text'


class TestStubMcpSessionAddDependency:
    """Tests for _StubMcpSession.call_tool('add_dependency', ...).

    ``add_dependency`` is NOT currently dispatched by scheduler.py (so it is not
    covered by the B3 dispatch-literal tripwire); the branch is added defensively
    per task spec so a future dispatch site is already stubbed.  Covered here by
    a direct unit test.
    """

    @pytest.mark.asyncio
    async def test_add_dependency_returns_envelope_echoing_id(self):
        """add_dependency returns a well-formed envelope whose text echoes the id."""
        stub = _StubMcpSession()
        resp = await stub.call_tool(
            'add_dependency',
            {'id': 't1', 'depends_on': 't2', 'project_root': '/p'},
        )
        assert resp['jsonrpc'] == '2.0'
        assert isinstance(resp['id'], int)
        content = resp['result']['content']
        assert isinstance(content, list) and len(content) >= 1
        assert content[0]['type'] == 'text'
        decoded = json.loads(content[0]['text'])
        assert decoded['id'] == 't1'


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

    @pytest.mark.asyncio
    async def test_handle_blast_radius_expansion_returns_true(self):
        """handle_blast_radius_expansion returns True without raising.

        This is the end-to-end sanity check: DI + lock-preinstall means the
        production code path for plan refinement works in eval mode.  The test
        also implicitly verifies that the module lock was pre-installed: without
        the pre-install ``try_acquire_additional`` would raise ``KeyError`` when
        looking up ``task_id`` in ``_held``.
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

    @pytest.mark.asyncio
    async def test_handle_blast_radius_expansion_contended_returns_false(self):
        """handle_blast_radius_expansion returns False when lock is contended.

        The interesting path: a different task already holds ``other_module``
        at max capacity (default limit = 1), so ``try_acquire_additional``
        rejects the expansion and the method resets the task to pending.
        This is the whole rationale for pre-installing the module lock — to
        ensure the ``_held`` entry exists so the conflict-check works rather
        than raising ``KeyError``.
        """
        from orchestrator.config import OrchestratorConfig

        config = OrchestratorConfig()
        scheduler, stub = _build_eval_scheduler(
            config, 'task-99', ['some_module']
        )
        # A competing task grabs other_module at the limit (max_per_module=1).
        acquired = scheduler.lock_table.try_acquire('task-other', ['other_module'])
        assert acquired, 'pre-condition: task-other must successfully hold other_module'

        result = await scheduler.handle_blast_radius_expansion(
            'task-99',
            ['some_module'],
            ['some_module', 'other_module'],  # other_module is now contended
        )
        # Expansion fails → method returns False and resets task to pending.
        assert result is False
        assert stub._statuses.get('task-99') == 'pending'
