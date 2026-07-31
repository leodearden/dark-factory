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


# ── deliverable (A): get_queue_stats answers both questions in one probe ────


class TestGetQueueStatsReconciliationHalt:
    """The drain-confirmation probe must distinguish HALTED from cannot-keep-up.

    task 2920 gave this tool ``reconciliation_backlog``; on its own that number
    could not separate the two causes, which is how the 2026-07-20 halt was
    mis-triaged as a capacity problem for two days.
    """

    @pytest.mark.asyncio
    async def test_backlog_and_halt_in_one_call(self, tmp_path):
        buf = EventBuffer(db_path=tmp_path / 'hv_qs_eb.db', buffer_size_threshold=100)
        await buf.initialize()
        journal = None
        try:
            await _seed_buffered(buf, 'proj', n=5)
            policy = BacklogPolicy(
                buf,
                _StubQueue(queue_depth=7, retry_in_flight=2),
                lambda _: False,
                hard_limit=500,
            )
            harness, journal, _ = await _make_harness(
                tmp_path, backlog_policy=policy, event_buffer=buf,
            )
            await harness.judge._apply_halt('proj', reason=_REAL_REASON)

            server = create_mcp_server(
                _make_status_mock_service(),
                reconciliation_harness=harness,
                backlog_policy=policy,
            )
            result = await server._tool_manager.call_tool(
                'get_queue_stats', {'project_id': 'proj'},
            )

            # One probe, both answers.
            assert result['reconciliation_backlog'] == 14  # 5 buffered + 7 + 2
            halt = result['reconciliation_halt']
            assert halt['halted'] is True
            assert halt['halt_reason'] == _REAL_REASON
            assert halt['halted_projects'] == ['proj']
        finally:
            if journal:
                await journal.close()
            await buf.close()

    @pytest.mark.asyncio
    async def test_non_halted_project_reports_not_halted(self, tmp_path):
        """A big backlog with halted=False is the capacity case — a genuinely
        different remedy, so the field must say so rather than stay silent."""
        harness, journal, buf = await _make_harness(tmp_path)
        try:
            server = create_mcp_server(
                _make_status_mock_service(), reconciliation_harness=harness,
            )
            result = await server._tool_manager.call_tool(
                'get_queue_stats', {'project_id': 'proj'},
            )
            assert result['reconciliation_halt']['halted'] is False
            assert result['reconciliation_halt']['halt_reason'] is None
        finally:
            await journal.close()
            if buf:
                await buf.close()

    @pytest.mark.asyncio
    async def test_absent_when_no_harness_wired(self):
        """No harness → no key, so test_queue_stats_tool.py's
        ``result == _FAKE_STATS`` exact-equality assertions stay valid."""
        server = create_mcp_server(_make_status_mock_service())
        result = await server._tool_manager.call_tool(
            'get_queue_stats', {'project_id': 'proj'},
        )
        assert 'reconciliation_halt' not in result
        assert result == {
            'counts': {'completed': 8648, 'dead': 3},
            'oldest_pending_age_seconds': None,
        }

    @pytest.mark.asyncio
    async def test_global_call_shape_unchanged(self, tmp_path):
        """Per-project only, mirroring reconciliation_backlog's gating."""
        harness, journal, buf = await _make_harness(tmp_path)
        try:
            await harness.judge._apply_halt('proj', reason=_REAL_REASON)
            server = create_mcp_server(
                _make_status_mock_service(), reconciliation_harness=harness,
            )
            result = await server._tool_manager.call_tool('get_queue_stats', {})
            assert 'reconciliation_halt' not in result
        finally:
            await journal.close()
            if buf:
                await buf.close()


# ── deliverable (B): trigger_reconciliation stops lying under halt ──────────


def _make_interceptor():
    """Mock TaskInterceptor exposing an awaitable buffer.request_trigger."""
    interceptor = MagicMock()
    interceptor.buffer = MagicMock()
    interceptor.buffer.request_trigger = AsyncMock(return_value=None)
    return interceptor


def _expire_cooldown(judge, project_id='proj'):
    from datetime import UTC, datetime, timedelta

    judge._halt_cooldown_until[project_id] = datetime.now(UTC) - timedelta(hours=1)


class TestTriggerReconciliationUnderHalt:
    """trigger_reconciliation must not answer 'requested' while the harness is
    skipping every cycle for that project.

    That false success is the second half of the incident: an operator, having
    finally noticed the backlog, ran the trigger, was told reconciliation
    'will trigger within ~5 seconds', and waited. Nothing ran.
    """

    @pytest.mark.asyncio
    async def test_halted_with_live_cooldown_reports_halted_and_requests_nothing(
        self, tmp_path,
    ):
        harness, journal, buf = await _make_harness(tmp_path)
        try:
            await harness.judge._apply_halt('proj', reason=_REAL_REASON)
            interceptor = _make_interceptor()
            server = create_mcp_server(
                _make_status_mock_service(),
                task_interceptor=interceptor,
                reconciliation_harness=harness,
            )

            result = await server._tool_manager.call_tool(
                'trigger_reconciliation', {'project_id': 'proj'},
            )

            assert result['status'] == 'halted'
            assert result['status'] != 'requested'
            assert result['halted'] is True
            assert result['halt_reason'] == _REAL_REASON
            assert result['halted_at'] is not None
            assert result['cooldown_until'] is not None
            assert result['cooldown_expired'] is False
            assert result['trigger_requested'] is False
            assert 'unhalt_reconciliation' in result['remedy']
            # Nothing was requested, so nothing may be promised.
            interceptor.buffer.request_trigger.assert_not_awaited()
        finally:
            await journal.close()
            if buf:
                await buf.close()

    @pytest.mark.asyncio
    async def test_expired_cooldown_with_auto_unhalt_forwards_the_trigger(
        self, tmp_path,
    ):
        """The mirror-image lie is also forbidden: with auto-unhalt enabled and
        the cooldown expired, the next tick unhalts and then RUNS the cycle, so
        a manual trigger genuinely is consumed. Say 'halted', but forward it."""
        harness, journal, buf = await _make_harness(
            tmp_path, auto_unhalt_after_cooldown=True,
        )
        try:
            await harness.judge._apply_halt('proj', reason=_REAL_REASON)
            _expire_cooldown(harness.judge)
            interceptor = _make_interceptor()
            server = create_mcp_server(
                _make_status_mock_service(),
                task_interceptor=interceptor,
                reconciliation_harness=harness,
            )

            result = await server._tool_manager.call_tool(
                'trigger_reconciliation', {'project_id': 'proj'},
            )

            assert result['status'] == 'halted'
            assert result['cooldown_expired'] is True
            assert result['trigger_requested'] is True
            interceptor.buffer.request_trigger.assert_awaited_once_with('proj')
            assert 'auto' in result['message'].lower()
        finally:
            await journal.close()
            if buf:
                await buf.close()

    @pytest.mark.asyncio
    async def test_expired_cooldown_without_auto_unhalt_requests_nothing(
        self, tmp_path,
    ):
        """Cooldown expiry alone does not clear a halt when auto-unhalt is off —
        the project stays halted until someone acts."""
        harness, journal, buf = await _make_harness(
            tmp_path, auto_unhalt_after_cooldown=False,
        )
        try:
            await harness.judge._apply_halt('proj', reason=_REAL_REASON)
            _expire_cooldown(harness.judge)
            interceptor = _make_interceptor()
            server = create_mcp_server(
                _make_status_mock_service(),
                task_interceptor=interceptor,
                reconciliation_harness=harness,
            )

            result = await server._tool_manager.call_tool(
                'trigger_reconciliation', {'project_id': 'proj'},
            )

            assert result['status'] == 'halted'
            assert result['cooldown_expired'] is True
            assert result['trigger_requested'] is False
            assert 'unhalt_reconciliation' in result['remedy']
            interceptor.buffer.request_trigger.assert_not_awaited()
        finally:
            await journal.close()
            if buf:
                await buf.close()

    @pytest.mark.asyncio
    async def test_not_halted_keeps_legacy_requested_behaviour(self, tmp_path):
        harness, journal, buf = await _make_harness(tmp_path)
        try:
            interceptor = _make_interceptor()
            server = create_mcp_server(
                _make_status_mock_service(),
                task_interceptor=interceptor,
                reconciliation_harness=harness,
            )

            result = await server._tool_manager.call_tool(
                'trigger_reconciliation', {'project_id': 'proj'},
            )

            assert result['status'] == 'requested'
            assert result['project_id'] == 'proj'
            interceptor.buffer.request_trigger.assert_awaited_once_with('proj')
        finally:
            await journal.close()
            if buf:
                await buf.close()

    @pytest.mark.asyncio
    async def test_no_harness_wired_keeps_legacy_behaviour(self):
        """Missing observability must never block a trigger."""
        interceptor = _make_interceptor()
        server = create_mcp_server(
            _make_status_mock_service(), task_interceptor=interceptor,
        )

        result = await server._tool_manager.call_tool(
            'trigger_reconciliation', {'project_id': 'proj'},
        )

        assert result['status'] == 'requested'
        interceptor.buffer.request_trigger.assert_awaited_once_with('proj')

    @pytest.mark.asyncio
    async def test_harness_without_judge_keeps_legacy_behaviour(self, tmp_path):
        harness, journal, buf = await _make_harness(tmp_path)
        try:
            harness.judge = None
            interceptor = _make_interceptor()
            server = create_mcp_server(
                _make_status_mock_service(),
                task_interceptor=interceptor,
                reconciliation_harness=harness,
            )

            result = await server._tool_manager.call_tool(
                'trigger_reconciliation', {'project_id': 'proj'},
            )

            assert result['status'] == 'requested'
            interceptor.buffer.request_trigger.assert_awaited_once_with('proj')
        finally:
            await journal.close()
            if buf:
                await buf.close()

    @pytest.mark.asyncio
    async def test_taskmaster_unconfigured_still_returns_configuration_error(
        self, tmp_path,
    ):
        """The config check runs before the halt check — a halted project with
        no Taskmaster still reports the config failure, not a halt."""
        harness, journal, buf = await _make_harness(tmp_path)
        try:
            await harness.judge._apply_halt('proj', reason=_REAL_REASON)
            server = create_mcp_server(
                _make_status_mock_service(), reconciliation_harness=harness,
            )

            result = await server._tool_manager.call_tool(
                'trigger_reconciliation', {'project_id': 'proj'},
            )

            assert result['error_type'] == 'ConfigurationError'
        finally:
            await journal.close()
            if buf:
                await buf.close()
