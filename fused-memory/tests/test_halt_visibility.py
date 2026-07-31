"""MCP-level tests for reconciliation halt visibility (task 3050).

A halt used to be invisible outside harness logs: ``get_status`` said nothing,
``get_queue_stats`` reported a large ``reconciliation_backlog`` whose two causes
(HALTED vs cannot-keep-up) have opposite remedies, and
``trigger_reconciliation`` answered ``'requested'`` while the harness silently
skipped every cycle. One real halt therefore ran unnoticed for 48h.

These tests pin the read surface (``reconciliation_halt`` on ``get_status`` and
``get_queue_stats``) and the honest ``trigger_reconciliation`` outcome.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from test_backlog_policy import _seed_buffered, _StubQueue

from fused_memory.config.schema import FusedMemoryConfig, ReconciliationConfig
from fused_memory.reconciliation.backlog_policy import BacklogPolicy
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.reconciliation.harness import ReconciliationHarness
from fused_memory.reconciliation.journal import ReconciliationJournal
from fused_memory.server.tools import create_mcp_server

# ── helpers ────────────────────────────────────────────────────────────────


def _make_status_mock_service():
    """Mock MemoryService whose get_status returns the standard shape.

    Copied from test_dead_letters_tool.py's ``_make_status_mock_service`` so the
    legacy-shape assertions here and there stay comparable.
    """
    svc = AsyncMock()
    svc.get_status = AsyncMock(return_value={
        'graphiti': {'connected': True},
        'mem0': {'connected': True},
        'projects': {},
        'queue': {
            'counts': {'completed': 8648, 'dead': 3},
            'oldest_pending_age_seconds': None,
        },
    })
    svc.durable_queue = MagicMock()
    svc.durable_queue.get_stats = AsyncMock(return_value={
        'counts': {'completed': 8648, 'dead': 3},
        'oldest_pending_age_seconds': None,
    })
    svc.durable_queue.get_dead_items = AsyncMock(return_value=[])
    return svc


async def _make_harness(tmp_path, *, backlog_policy=None, event_buffer=None, **cfg):
    """Build a REAL ReconciliationHarness (real journal + event buffer).

    Mirrors test_queue_stats_tool.py's fixture. Returns
    ``(harness, journal, buffer)``; the caller closes journal/buffer.
    """
    project_root = tmp_path / 'proj_root'
    project_root.mkdir(exist_ok=True)

    buf = event_buffer
    owns_buf = buf is None
    if owns_buf:
        buf = EventBuffer(db_path=tmp_path / 'hv_eb.db', buffer_size_threshold=100)
        await buf.initialize()
    journal = ReconciliationJournal(tmp_path / 'hv_journal')
    await journal.initialize()

    config = FusedMemoryConfig(
        reconciliation=ReconciliationConfig(
            enabled=True,
            judge_enabled=True,
            explore_codebase_root='/tmp/test',
            agent_llm_provider='anthropic',
            agent_llm_model='claude-sonnet-4-20250514',
            **cfg,
        )
    )
    harness = ReconciliationHarness(
        memory_service=AsyncMock(),
        taskmaster=AsyncMock(),
        journal=journal,
        event_buffer=buf,
        config=config,
        backlog_policy=backlog_policy,
        known_projects={'proj': str(project_root)},
    )
    assert harness.judge is not None
    return harness, journal, buf if owns_buf else None


_REAL_REASON = 'Serious verdict in run 33581299'


# ── deliverable (A): get_status reports halt state ──────────────────────────


class TestGetStatusReconciliationHalt:
    """get_status must answer "is anything halted, and why" — the probe an
    operator reaches for first, and the one that said nothing for 48h."""

    @pytest.mark.asyncio
    async def test_halted_project_reports_reason_and_times(self, tmp_path):
        harness, journal, buf = await _make_harness(tmp_path)
        try:
            await harness.judge._apply_halt('proj', reason=_REAL_REASON)

            server = create_mcp_server(
                _make_status_mock_service(), reconciliation_harness=harness,
            )
            result = await server._tool_manager.call_tool(
                'get_status', {'project_id': 'proj'},
            )

            halt = result['reconciliation_halt']
            assert halt['halted'] is True
            assert halt['halt_reason'] == _REAL_REASON
            assert isinstance(halt['halted_at'], str)
            assert isinstance(halt['cooldown_until'], str)
            assert halt['cooldown_expired'] is False
            assert halt['halted_projects'] == ['proj']
            # The halt enrichment is top-level, NOT inside `queue` — `queue` is
            # the durable-write-queue subsystem and conflating the two is the
            # exact 2920 mis-triage.
            assert 'reconciliation_halt' not in result['queue']
        finally:
            await journal.close()
            if buf:
                await buf.close()

    @pytest.mark.asyncio
    async def test_non_halted_project_still_sees_the_fleet(self, tmp_path):
        """Asking about a healthy project must still surface that ANOTHER
        project is halted — nobody knew to go looking last time."""
        harness, journal, buf = await _make_harness(tmp_path)
        try:
            await harness.judge._apply_halt('proj', reason=_REAL_REASON)

            server = create_mcp_server(
                _make_status_mock_service(), reconciliation_harness=harness,
            )
            result = await server._tool_manager.call_tool(
                'get_status', {'project_id': 'other'},
            )

            halt = result['reconciliation_halt']
            assert halt['halted'] is False
            assert halt['halt_reason'] is None
            assert halt['halted_at'] is None
            assert halt['halted_projects'] == ['proj']
        finally:
            await journal.close()
            if buf:
                await buf.close()

    @pytest.mark.asyncio
    async def test_global_call_lists_halted_projects_without_per_project_claim(
        self, tmp_path,
    ):
        harness, journal, buf = await _make_harness(tmp_path)
        try:
            await harness.judge._apply_halt('proj', reason=_REAL_REASON)

            server = create_mcp_server(
                _make_status_mock_service(), reconciliation_harness=harness,
            )
            result = await server._tool_manager.call_tool('get_status', {})

            halt = result['reconciliation_halt']
            assert halt['halted_projects'] == ['proj']
            # No project was named, so no per-project claim is made at all.
            assert 'halted' not in halt
            assert 'halt_reason' not in halt
        finally:
            await journal.close()
            if buf:
                await buf.close()

    @pytest.mark.asyncio
    async def test_absent_when_no_harness_wired(self):
        """No harness → no key, so the legacy get_status shape (and
        test_dead_letters_tool.py's assertions) stay byte-identical."""
        server = create_mcp_server(_make_status_mock_service())
        result = await server._tool_manager.call_tool(
            'get_status', {'project_id': 'proj'},
        )
        assert 'reconciliation_halt' not in result

    @pytest.mark.asyncio
    async def test_absent_when_harness_has_no_judge(self, tmp_path):
        harness, journal, buf = await _make_harness(tmp_path)
        try:
            harness.judge = None
            server = create_mcp_server(
                _make_status_mock_service(), reconciliation_harness=harness,
            )
            result = await server._tool_manager.call_tool(
                'get_status', {'project_id': 'proj'},
            )
            assert 'reconciliation_halt' not in result
        finally:
            await journal.close()
            if buf:
                await buf.close()
