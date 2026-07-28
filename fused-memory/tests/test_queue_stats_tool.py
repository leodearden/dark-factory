"""MCP-level tests for the get_queue_stats tool."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from test_backlog_policy import _seed_buffered, _StubQueue

from fused_memory.reconciliation.backlog_policy import BacklogPolicy
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.server.tools import create_mcp_server
from fused_memory.services.durable_queue import DurableWriteQueue

# ── helpers ────────────────────────────────────────────────────────────────


def _make_mock_service(get_stats_return=None):
    """Build a minimal mock MemoryService with a durable_queue."""
    svc = AsyncMock()
    svc.durable_queue = MagicMock()
    svc.durable_queue.get_stats = AsyncMock(
        return_value=get_stats_return
        or {'counts': {'completed': 8648, 'dead': 3}, 'oldest_pending_age_seconds': None}
    )
    return svc


_FAKE_STATS = {'counts': {'completed': 8648, 'dead': 3}, 'oldest_pending_age_seconds': None}


# ── step-1 tests ───────────────────────────────────────────────────────────


class TestGetQueueStatsProjectScoping:
    """get_queue_stats forwards project_id through to durable_queue.get_stats."""

    @pytest.mark.asyncio
    async def test_forwards_project_id_as_group_id(self):
        """project_id is forwarded as group_id, and the tool returns get_stats' result."""
        svc = _make_mock_service(get_stats_return=_FAKE_STATS)
        server = create_mcp_server(svc)

        result = await server._tool_manager.call_tool(
            'get_queue_stats',
            {'project_id': 'proj1'},
        )

        svc.durable_queue.get_stats.assert_called_once_with(group_id='proj1')
        assert result == _FAKE_STATS

    @pytest.mark.asyncio
    async def test_no_project_id_is_global(self):
        """Omitting project_id preserves the backward-compatible global path."""
        svc = _make_mock_service(get_stats_return=_FAKE_STATS)
        server = create_mcp_server(svc)

        result = await server._tool_manager.call_tool(
            'get_queue_stats',
            {},
        )

        svc.durable_queue.get_stats.assert_called_once_with(group_id=None)
        assert result == _FAKE_STATS


# ── step-3: real-queue acceptance test ──────────────────────────────────────


async def _poll_until_dead(
    q: DurableWriteQueue,
    *,
    group_id: str | None = None,
    expected_dead: int,
    timeout: float = 20.0,
) -> None:
    """Poll queue stats until at least *expected_dead* items have status='dead'.

    Mirrors the helper in test_durable_queue.py: avoids a fixed wall-clock
    sleep, which was a source of flakiness under xdist load.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        stats = await q.get_stats(group_id=group_id)
        if stats['counts'].get('dead', 0) >= expected_dead:
            return
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            raise AssertionError(
                f'Timed out waiting for {expected_dead} dead item(s) '
                f'(group_id={group_id!r}); last counts={stats["counts"]}'
            )
        await asyncio.sleep(0.05)


class TestGetQueueStatsConsistencyWithDeadLetters:
    """Reproduces the reported symptom: get_queue_stats' dead count must agree
    with get_dead_letters for the same project, against a real durable queue.
    """

    @pytest.mark.asyncio
    async def test_scoped_dead_counts_agree_with_get_dead_letters(self, tmp_path):
        # group_ids are already underscore-canonical (proj_a/proj_b/proj_c), matching
        # what real writers actually enqueue: memory_service.add_episode/add_memory
        # always key the durable queue by scope.graphiti_group_id, which Scope's
        # field_validator (task 2267 / seam S2) canonicalizes at construction. A
        # hyphenated literal here would be synthetic data no production write path
        # can produce, and would falsely fail once get_queue_stats canonicalizes its
        # query project_id (task 2268 / seam S3) while the out-of-scope
        # get_dead_letters tool does not.
        execute = AsyncMock(side_effect=RuntimeError('fail'))
        q = DurableWriteQueue(
            data_dir=tmp_path / 'queue',
            execute_write=execute,
            workers_per_group=1,
            semaphore_limit=5,
            max_attempts=1,
            retry_base_seconds=0.01,
            write_timeout_seconds=2.0,
        )
        await q.initialize()
        try:
            for i in range(2):
                await q.enqueue(
                    group_id='proj_a', operation='add_episode',
                    payload={'content': f'a{i}', 'group_id': 'proj_a', 'name': f'a{i}'},
                )
            await q.enqueue(
                group_id='proj_b', operation='add_episode',
                payload={'content': 'b0', 'group_id': 'proj_b', 'name': 'b0'},
            )
            await _poll_until_dead(q, expected_dead=3)

            svc = AsyncMock()
            svc.durable_queue = q
            server = create_mcp_server(svc)

            stats_a = await server._tool_manager.call_tool(
                'get_queue_stats', {'project_id': 'proj_a'},
            )
            stats_b = await server._tool_manager.call_tool(
                'get_queue_stats', {'project_id': 'proj_b'},
            )
            stats_c = await server._tool_manager.call_tool(
                'get_queue_stats', {'project_id': 'proj_c'},
            )
            stats_global = await server._tool_manager.call_tool('get_queue_stats', {})

            assert stats_a['counts'].get('dead', 0) == 2
            assert stats_b['counts'].get('dead', 0) == 1
            assert stats_c['counts'].get('dead', 0) == 0
            assert stats_global['counts'].get('dead', 0) == 3

            dead_letters_a = await server._tool_manager.call_tool(
                'get_dead_letters', {'project_id': 'proj_a'},
            )
            dead_letters_b = await server._tool_manager.call_tool(
                'get_dead_letters', {'project_id': 'proj_b'},
            )
            dead_letters_c = await server._tool_manager.call_tool(
                'get_dead_letters', {'project_id': 'proj_c'},
            )

            assert stats_a['counts'].get('dead', 0) == dead_letters_a['counts']['durable_queue']
            assert stats_b['counts'].get('dead', 0) == dead_letters_b['counts']['durable_queue']
            assert stats_c['counts'].get('dead', 0) == dead_letters_c['counts']['durable_queue']
        finally:
            await q.close()


# ── deliverable (b): reconciliation_backlog probe on get_queue_stats ─────────


class TestGetQueueStatsReconciliationBacklog:
    """get_queue_stats must expose the reconciliation event backlog — the metric
    backlog escalations actually govern. The auto-watcher/L2 mis-triaged the
    2026-07-20 judge-halt incident for 2 days by reading this tool's
    durable-write-queue counts (a distinct subsystem, ~0) while the true
    reconciliation backlog grew to 1548 (task 2920 deliverable b)."""

    @pytest.mark.asyncio
    async def test_reconciliation_backlog_present_when_policy_wired(self, tmp_path):
        """When a BacklogPolicy is wired and a project_id is given, the result
        carries reconciliation_backlog == policy.current_backlog(project_id)
        (buffered events + queue depth + retries)."""
        buf = EventBuffer(db_path=tmp_path / 'qs_eb.db', buffer_size_threshold=100)
        await buf.initialize()
        try:
            await _seed_buffered(buf, 'proj', n=5)
            policy = BacklogPolicy(
                buf,
                _StubQueue(queue_depth=7, retry_in_flight=2),
                lambda _: False,
                hard_limit=500,
            )
            expected = await policy.current_backlog('proj')
            assert expected == 5 + 7 + 2  # fixture sanity

            svc = _make_mock_service(
                get_stats_return={'counts': {'completed': 1, 'dead': 0}},
            )
            server = create_mcp_server(svc, backlog_policy=policy)
            result = await server._tool_manager.call_tool(
                'get_queue_stats', {'project_id': 'proj'},
            )
            assert result['reconciliation_backlog'] == expected
            assert result['reconciliation_backlog'] == 14
        finally:
            await buf.close()

    @pytest.mark.asyncio
    async def test_reconciliation_backlog_absent_when_no_policy(self):
        """With no BacklogPolicy wired the field is NOT emitted — this keeps the
        existing exact-equality scoping tests byte-identical."""
        svc = _make_mock_service(get_stats_return=_FAKE_STATS)
        server = create_mcp_server(svc)
        result = await server._tool_manager.call_tool(
            'get_queue_stats', {'project_id': 'proj'},
        )
        assert 'reconciliation_backlog' not in result


# ── unhalt_reconciliation auto-closes the halt escalation (task 2998) ───────


class TestUnhaltReconciliationClosesEscalation:
    """End-to-end: clearing a halt must leave its escalation resolved.

    The task's user-observable signal — a halt escalation written by
    BacklogPolicy is closed when the operator runs unhalt_reconciliation,
    instead of staying pending forever.
    """

    @pytest.mark.asyncio
    async def test_unhalt_resolves_pending_halt_escalation(self, tmp_path):
        from escalation.queue import EscalationQueue

        from fused_memory.config.schema import (
            FusedMemoryConfig,
            ReconciliationConfig,
        )
        from fused_memory.reconciliation.harness import ReconciliationHarness
        from fused_memory.reconciliation.journal import ReconciliationJournal

        project_root = tmp_path / 'proj_root'
        project_root.mkdir()

        buf = EventBuffer(db_path=tmp_path / 'eb.db', buffer_size_threshold=100)
        await buf.initialize()
        journal = ReconciliationJournal(tmp_path / 'journal')
        await journal.initialize()
        try:
            policy = BacklogPolicy(buf, _StubQueue(), lambda _: True)
            config = FusedMemoryConfig(
                reconciliation=ReconciliationConfig(
                    enabled=True,
                    judge_enabled=True,
                    explore_codebase_root='/tmp/test',
                    agent_llm_provider='anthropic',
                    agent_llm_model='claude-sonnet-4-20250514',
                )
            )
            harness = ReconciliationHarness(
                memory_service=AsyncMock(),
                taskmaster=AsyncMock(),
                journal=journal,
                event_buffer=buf,
                config=config,
                backlog_policy=policy,
                known_projects={'proj': str(project_root)},
            )
            assert harness.judge is not None

            reason = 'Unparseable judge response in run 33581299'
            await harness.judge._apply_halt('proj', reason=reason)
            await harness._notify_judge_halt('proj', reason=reason)

            esc_dir = project_root / 'data' / 'escalations'
            queue = EscalationQueue(esc_dir)
            pending = [
                e for e in queue.get_pending()
                if e.id.startswith('esc-reconciliation-halt-')
            ]
            assert len(pending) == 1
            esc_id = pending[0].id

            svc = _make_mock_service()
            server = create_mcp_server(svc, reconciliation_harness=harness)
            result = await server._tool_manager.call_tool(
                'unhalt_reconciliation', {'project_id': 'proj'},
            )

            assert result['status'] == 'unhalted'
            assert esc_id in result['escalations_resolved']
            assert esc_id in result['message']

            fresh = EscalationQueue(esc_dir)
            assert esc_id not in {e.id for e in fresh.get_pending()}
            closed = fresh.get(esc_id)
            assert closed is not None
            assert closed.status == 'resolved'
        finally:
            await journal.close()
            await buf.close()
