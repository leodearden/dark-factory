"""Tests for PRD γ cutover: BaseStage MCP config wiring and ReconReportState lifecycle."""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import StageId, Watermark
from fused_memory.models.scope import ProjectId, ProjectRoot, ProjectScope
from fused_memory.reconciliation.cli_stage_runner import STAGE_REPORT_SCHEMA
from fused_memory.reconciliation.stages.base import BaseStage
from fused_memory.reconciliation.stages.memory_consolidator import MemoryConsolidator


class _StubStage(BaseStage):
    """Minimal BaseStage stub for testing BaseStage.run logic directly."""

    def get_disallowed_tools(self) -> list[str]:
        return []

    def get_system_prompt(self) -> str:
        return 'test prompt'

    def get_report_schema(self) -> dict:
        return STAGE_REPORT_SCHEMA

    async def assemble_payload(self, events, watermark, prior_reports) -> str:
        return 'test payload'


def _make_stage(
    recon_report_port: int = 8003,
    recon_report_state=None,
) -> _StubStage:
    """Build a _StubStage with minimal mock deps."""
    config = ReconciliationConfig()
    memory_mock = AsyncMock()

    stage = _StubStage(
        StageId.memory_consolidator,
        memory_mock,
        AsyncMock(),  # taskmaster
        AsyncMock(),  # journal
        config,
        scope=ProjectScope(ProjectId('test_project'), ProjectRoot('/tmp/test')),
        recon_report_port=recon_report_port,
        recon_report_state=recon_report_state,
    )
    return stage


def _make_consolidator(recon_report_port: int = 8003) -> MemoryConsolidator:
    """Build a MemoryConsolidator with mocked deps for _build_mcp_config tests."""
    config = ReconciliationConfig()
    memory_mock = AsyncMock()
    memory_mock.get_episodes = AsyncMock(return_value=[])
    memory_mock.mem0 = AsyncMock()
    memory_mock.mem0.get_all = AsyncMock(return_value={'results': []})
    memory_mock.get_status = AsyncMock(return_value={})

    stage = MemoryConsolidator(
        StageId.memory_consolidator,
        memory_mock,
        AsyncMock(),  # taskmaster
        AsyncMock(),  # journal
        config,
        scope=ProjectScope(ProjectId('test_project'), ProjectRoot('/tmp/test')),
        recon_report_port=recon_report_port,
    )
    stage.episode_limit = 5
    stage.memory_limit = 10
    return stage


class _FakeReconState:
    """Recording fake for ReconReportState — logs calls for assertion."""

    def __init__(self, assembled_report: dict | None = None):
        self.calls: list[tuple] = []
        self._assembled = assembled_report

    def start_report(self, run_id: str, stage: str, project_id: str) -> dict:
        self.calls.append(('start_report', run_id, stage, project_id))
        return {'run_id': run_id, 'stage': stage}

    def get_assembled_report(self, run_id: str, stage: str) -> dict | None:
        self.calls.append(('get_assembled_report', run_id, stage))
        return self._assembled


def _minimal_ctor_args():
    """Positional args shared by every BaseStage subclass constructor in this file."""
    config = ReconciliationConfig()
    memory_mock = AsyncMock()
    return (
        StageId.memory_consolidator,
        memory_mock,
        AsyncMock(),  # taskmaster
        AsyncMock(),  # journal
        config,
    )


class TestBaseStageScopeContract:
    """BaseStage requires a ProjectScope at construction (task 2146 / recon-project-scope
    batch β); project_id/project_root become read-only views over it.
    """

    def test_stub_stage_without_scope_raises_type_error(self):
        """scope is a required keyword-only argument — omitting it is a TypeError."""
        with pytest.raises(TypeError):
            _StubStage(*_minimal_ctor_args())  # type: ignore[call-arg]

    def test_memory_consolidator_without_scope_raises_type_error(self):
        """The contract propagates through MemoryConsolidator's *args/**kwargs forward."""
        with pytest.raises(TypeError):
            MemoryConsolidator(*_minimal_ctor_args())

    def test_scope_populates_project_id_and_project_root(self):
        scope = ProjectScope(ProjectId('p'), ProjectRoot('/root'))
        stage = _StubStage(*_minimal_ctor_args(), scope=scope)
        assert stage.project_id == 'p'
        assert stage.project_root == '/root'
        assert stage.scope is scope

    def test_project_id_has_no_setter(self):
        scope = ProjectScope(ProjectId('p'), ProjectRoot('/root'))
        stage = _StubStage(*_minimal_ctor_args(), scope=scope)
        with pytest.raises(AttributeError):
            # setattr, not a direct attribute assignment, so this stays pyright-clean
            # (a direct assignment to a read-only property is reportAttributeAccessIssue).
            setattr(stage, 'project_id', 'x')  # noqa: B010

    def test_project_root_has_no_setter(self):
        scope = ProjectScope(ProjectId('p'), ProjectRoot('/root'))
        stage = _StubStage(*_minimal_ctor_args(), scope=scope)
        with pytest.raises(AttributeError):
            # setattr, not a direct attribute assignment, so this stays pyright-clean
            # (a direct assignment to a read-only property is reportAttributeAccessIssue).
            setattr(stage, 'project_root', 'y')  # noqa: B010

    def test_known_projects_defaults_to_empty_dict(self):
        scope = ProjectScope(ProjectId('p'), ProjectRoot('/root'))
        stage = _StubStage(*_minimal_ctor_args(), scope=scope)
        assert stage.known_projects == {}

    def test_known_projects_stored_when_passed(self):
        scope = ProjectScope(ProjectId('p'), ProjectRoot('/root'))
        known_projects = {'p': '/root', 'q': '/other'}
        stage = _StubStage(*_minimal_ctor_args(), scope=scope, known_projects=known_projects)
        assert stage.known_projects == known_projects


class TestReconStateLifecycle:
    """BaseStage.run must call start_report before the CLI and get_assembled_report after."""

    @pytest.mark.asyncio
    async def test_start_report_called_before_cli(self):
        """start_report is called once with (run_id, stage_id.value, project_id)."""
        call_log: list[str] = []

        assembled = {
            'summary': 'ok',
            'stats': {'items_reviewed': 5},
            'flagged_items': [],
            'summary_warnings': [],
        }

        class _RecordingState(_FakeReconState):
            def start_report(self, run_id, stage, project_id):
                call_log.append('start_report')
                return super().start_report(run_id, stage, project_id)

            def get_assembled_report(self, run_id, stage):
                call_log.append('get_assembled_report')
                return super().get_assembled_report(run_id, stage)

        state = _RecordingState(assembled_report=assembled)
        stage = _make_stage(recon_report_state=state)
        watermark = Watermark(project_id='test_project')

        from fused_memory.reconciliation.cli_stage_runner import StageResult

        fake_result = StageResult(report={}, success=True)

        with patch(
            'fused_memory.reconciliation.stages.base.run_stage_via_cli',
            new=AsyncMock(return_value=fake_result),
        ):
            await stage.run([], watermark, [], run_id='run-001')

        # start_report must be called before the CLI (first in log)
        assert call_log[0] == 'start_report', 'start_report must be called before CLI'
        assert call_log[1] == 'get_assembled_report', 'get_assembled_report called after CLI'

    @pytest.mark.asyncio
    async def test_start_report_args(self):
        """start_report receives the correct run_id, stage_id.value, and project_id."""
        assembled = {'summary': '', 'stats': {}, 'flagged_items': [], 'summary_warnings': []}
        state = _FakeReconState(assembled_report=assembled)
        stage = _make_stage(recon_report_state=state)
        watermark = Watermark(project_id='test_project')

        from fused_memory.reconciliation.cli_stage_runner import StageResult

        with patch(
            'fused_memory.reconciliation.stages.base.run_stage_via_cli',
            new=AsyncMock(return_value=StageResult(report={}, success=True)),
        ):
            await stage.run([], watermark, [], run_id='run-abc')

        start_call = next(c for c in state.calls if c[0] == 'start_report')
        assert start_call[1] == 'run-abc'  # run_id
        assert start_call[2] == StageId.memory_consolidator.value  # stage
        assert start_call[3] == 'test_project'  # project_id

    @pytest.mark.asyncio
    async def test_assembled_report_overrides_stage_result(self):
        """StageReport.items_flagged and stats come from assembled report, not stage_result."""
        assembled = {
            'summary': 'test summary',
            'stats': {'items_reviewed': 42},
            'flagged_items': [{'description': 'test finding', 'severity': 'minor'}],
            'summary_warnings': [],
        }
        state = _FakeReconState(assembled_report=assembled)
        stage = _make_stage(recon_report_state=state)
        watermark = Watermark(project_id='test_project')

        from fused_memory.reconciliation.cli_stage_runner import StageResult

        # stage_result.report has different content — should be ignored
        dummy_result = StageResult(
            report={'flagged_items': [], 'stats': {}, 'summary': 'ignored'},
            success=True,
        )

        with patch(
            'fused_memory.reconciliation.stages.base.run_stage_via_cli',
            new=AsyncMock(return_value=dummy_result),
        ):
            report = await stage.run([], watermark, [], run_id='run-xyz')

        assert report.items_flagged == assembled['flagged_items']
        assert report.stats == assembled['stats']


class TestReconStateFallbackToEmpty:
    """When get_assembled_report returns None, BaseStage.run produces an empty StageReport."""

    @pytest.mark.asyncio
    async def test_none_assembled_produces_empty_stage_report(self):
        """get_assembled_report returning None → items_flagged=[], stats={}, no exception."""
        state = _FakeReconState(assembled_report=None)  # Simulates agent crash
        stage = _make_stage(recon_report_state=state)
        watermark = Watermark(project_id='test_project')

        from fused_memory.reconciliation.cli_stage_runner import StageResult

        with patch(
            'fused_memory.reconciliation.stages.base.run_stage_via_cli',
            new=AsyncMock(return_value=StageResult(report={}, success=True)),
        ):
            report = await stage.run([], watermark, [], run_id='run-none')

        assert report.items_flagged == [], 'items_flagged must be empty on None assembled'
        assert report.stats == {}, 'stats must be empty on None assembled'

    @pytest.mark.asyncio
    async def test_timestamps_still_set_on_none_assembled(self):
        """started_at and completed_at are populated even on None assembled."""
        state = _FakeReconState(assembled_report=None)
        stage = _make_stage(recon_report_state=state)
        watermark = Watermark(project_id='test_project')

        from fused_memory.reconciliation.cli_stage_runner import StageResult

        with patch(
            'fused_memory.reconciliation.stages.base.run_stage_via_cli',
            new=AsyncMock(return_value=StageResult(report={}, success=True)),
        ):
            report = await stage.run([], watermark, [], run_id='run-ts')

        assert report.started_at is not None
        assert report.completed_at is not None


class TestHarnessForwardsReconState:
    """ReconciliationHarness must accept recon_report_state and forward it to all stages."""

    def _make_harness(self, state):
        from unittest.mock import MagicMock

        from fused_memory.config.schema import FusedMemoryConfig
        from fused_memory.reconciliation.harness import ReconciliationHarness

        config = FusedMemoryConfig()
        memory_mock = AsyncMock()
        journal_mock = AsyncMock()
        event_buffer_mock = MagicMock()

        harness = ReconciliationHarness(
            memory_service=memory_mock,
            taskmaster=None,
            journal=journal_mock,
            event_buffer=event_buffer_mock,
            config=config,
            recon_report_state=state,
        )
        # Task 2146 β: the harness no longer pre-builds a long-lived `self.stages`
        # list at construction — every cycle builds fresh stages via
        # `_make_stages(scope)`. Build one here and stash it so this fixture's
        # callers keep their `harness.stages[N]` access pattern.
        harness.stages = harness._make_stages(
            ProjectScope(ProjectId('test_project'), ProjectRoot('/tmp/test'))
        )
        return harness

    def test_stages_receive_recon_report_state(self):
        """All three stages get _recon_report_state set to the injected state object."""
        state = _FakeReconState()
        harness = self._make_harness(state)
        for stage in harness.stages:
            assert stage._recon_report_state is state, (
                f'Stage {stage.stage_id} must have _recon_report_state set'
            )

    def test_stages_receive_recon_report_port_from_config(self):
        """All three stages get _recon_report_port from config.server.recon_report_port."""
        state = _FakeReconState()
        harness = self._make_harness(state)
        # Default ServerConfig has recon_report_port=8003
        for stage in harness.stages:
            assert stage._recon_report_port == 8003, (
                f'Stage {stage.stage_id} must have _recon_report_port=8003 from config'
            )


class TestMainWiringReconState:
    """Verify that run_server threads the recon_report state into ReconciliationHarness.

    Part 1 (functional): tests that _build_recon_report_components() returns a
    ReconReportState that can be passed to ReconciliationHarness and is preserved
    as harness._recon_report_state (identity check).

    Part 2 (structural / RED): tests that run_server's source actually contains the
    wiring call so the live process — not just a manual construction — threads the
    state. This assertion FAILS until step-12 changes run_server.
    """

    def test_build_recon_report_components_returns_recon_report_state(self):
        """_build_recon_report_components returns a ReconReportState instance."""
        from fused_memory.config.schema import FusedMemoryConfig, ServerConfig
        from fused_memory.server.main import _build_recon_report_components
        from fused_memory.server.recon_report import ReconReportState

        config = FusedMemoryConfig(server=ServerConfig(recon_report_port=8099))
        state, _, _ = _build_recon_report_components(config)
        assert isinstance(state, ReconReportState)

    def test_harness_preserves_state_identity(self):
        """ReconciliationHarness preserves the exact state object passed at construction."""
        from fused_memory.config.schema import FusedMemoryConfig, ServerConfig
        from fused_memory.reconciliation.harness import ReconciliationHarness
        from fused_memory.server.main import _build_recon_report_components

        config = FusedMemoryConfig(server=ServerConfig(recon_report_port=8099))
        state, _, _ = _build_recon_report_components(config)

        harness = ReconciliationHarness(
            memory_service=AsyncMock(),
            taskmaster=None,
            journal=AsyncMock(),
            event_buffer=MagicMock(),
            config=config,
            recon_report_state=state,
        )
        assert harness._recon_report_state is state


class TestBuildMcpConfigReconReport:
    """_build_mcp_config must inject a recon-report HTTP entry from the configured port."""

    def test_recon_report_entry_default_port(self):
        """recon-report server entry present with default port 8003."""
        stage = _make_consolidator(recon_report_port=8003)
        mcp_config = stage._build_mcp_config()
        servers = mcp_config['mcpServers']
        assert 'recon-report' in servers, 'recon-report must be in mcpServers'
        entry = servers['recon-report']
        assert entry == {'type': 'http', 'url': 'http://127.0.0.1:8003/mcp/'}

    def test_recon_report_entry_custom_port(self):
        """recon-report entry uses the port passed at construction, not a hard-coded value."""
        stage = _make_consolidator(recon_report_port=9999)
        mcp_config = stage._build_mcp_config()
        entry = mcp_config['mcpServers']['recon-report']
        assert entry == {'type': 'http', 'url': 'http://127.0.0.1:9999/mcp/'}

    def test_existing_entries_preserved(self):
        """fused-memory and jcodemunch entries still present after recon-report injection."""
        stage = _make_consolidator(recon_report_port=8003)
        servers = stage._build_mcp_config()['mcpServers']
        assert 'fused-memory' in servers, 'fused-memory entry must remain'
        assert 'jcodemunch' in servers, 'jcodemunch entry must remain'


class TestStartReportErrorHandling:
    """BaseStage.run must degrade gracefully when start_report raises (reviewer finding
    error_handling).  A transient start_report failure must not abort the stage.
    """

    @pytest.mark.asyncio
    async def test_start_report_raises_degrades_to_empty_report(self):
        """When start_report raises, the stage produces an empty StageReport (no exception)."""

        class _BrokenState(_FakeReconState):
            def start_report(self, run_id, stage, project_id):
                raise RuntimeError('duplicate run_id from crashed prior run')

        state = _BrokenState(assembled_report={'summary': 'should not appear', 'flagged_items': [{'description': 'x', 'severity': 'minor', 'cited_tasks': [{'task_id': '1', 'project_id': 'p', 'title': 't'}]}], 'stats': {}})
        stage = _make_stage(recon_report_state=state)
        watermark = Watermark(project_id='test_project')

        from fused_memory.reconciliation.cli_stage_runner import StageResult

        with patch(
            'fused_memory.reconciliation.stages.base.run_stage_via_cli',
            new=AsyncMock(return_value=StageResult(report={}, success=True)),
        ):
            report = await stage.run([], watermark, [], run_id='run-broken')

        # Degraded: empty items and stats (get_assembled_report was NOT called)
        assert report.items_flagged == [], (
            'Degraded stage must produce empty items_flagged, not the assembled report'
        )
        assert report.stats == {}
        # get_assembled_report must NOT have been called (no successful start)
        get_calls = [c for c in state.calls if c[0] == 'get_assembled_report']
        assert get_calls == [], (
            f'get_assembled_report must not be called after start_report failure, '
            f'but got: {get_calls!r}'
        )

    @pytest.mark.asyncio
    async def test_start_report_raises_stage_still_returns_stage_report(self):
        """Even with start_report failure, run() returns a StageReport (not an exception)."""

        class _BrokenState(_FakeReconState):
            def start_report(self, run_id, stage, project_id):
                raise ValueError('TTL-reaper race')

        state = _BrokenState(assembled_report=None)
        stage = _make_stage(recon_report_state=state)
        watermark = Watermark(project_id='test_project')

        from fused_memory.models.reconciliation import StageReport
        from fused_memory.reconciliation.cli_stage_runner import StageResult

        with patch(
            'fused_memory.reconciliation.stages.base.run_stage_via_cli',
            new=AsyncMock(return_value=StageResult(report={}, success=True)),
        ):
            result = await stage.run([], watermark, [], run_id='run-valuerr')

        assert isinstance(result, StageReport), 'run() must return StageReport on start_report failure'
        assert result.started_at is not None
        assert result.completed_at is not None

    @pytest.mark.asyncio
    async def test_start_report_raises_short_circuits_without_invoking_cli(self):
        """Degraded path must short-circuit — run_stage_via_cli is NOT called.

        The recon_report-mode prompts instruct the agent NOT to produce a
        structured JSON response.  Falling back to a JSON-schema CLI invocation
        in the degraded path would either time out or fail schema validation,
        burning a full agent budget on every recon_report outage.  Verify the
        short-circuit by asserting the CLI runner was never called and the
        returned StageReport reports zero llm_calls / tokens_used.
        """

        class _BrokenState(_FakeReconState):
            def start_report(self, run_id, stage, project_id):
                raise RuntimeError('start_report exploded')

        state = _BrokenState(assembled_report=None)
        stage = _make_stage(recon_report_state=state)
        watermark = Watermark(project_id='test_project')

        from fused_memory.reconciliation.cli_stage_runner import StageResult

        cli_mock = AsyncMock(return_value=StageResult(report={}, success=True))
        with patch(
            'fused_memory.reconciliation.stages.base.run_stage_via_cli',
            new=cli_mock,
        ):
            result = await stage.run([], watermark, [], run_id='run-shortcircuit')

        assert cli_mock.await_count == 0, (
            'run_stage_via_cli must NOT be invoked when start_report fails — '
            f'short-circuit broken, was awaited {cli_mock.await_count} time(s)'
        )
        assert result.items_flagged == []
        assert result.stats == {}
        assert result.llm_calls == 0
        assert result.tokens_used == 0


class TestSessionCaptureLifecycle:
    """BaseStage.run mints a CLI session, persists it (record_run_session) BEFORE
    launching the stage subprocess, forwards session_id + a per-run TaskConfigDir
    into run_stage_via_cli, and clears the session afterwards — even when the stage
    subprocess raises (finally path) — per task 2744."""

    def _make_stage_with_real_journal_dir(self, tmp_path):
        """_StubStage whose journal has a real data_dir + AsyncMock session methods."""
        stage = _make_stage(recon_report_state=None)
        journal = MagicMock()
        journal.data_dir = tmp_path
        journal.record_run_session = AsyncMock()
        journal.clear_run_session = AsyncMock()
        stage.journal = journal
        return stage, journal

    @pytest.mark.asyncio
    async def test_session_minted_persisted_and_forwarded(self, tmp_path):
        from shared.config_dir import TaskConfigDir

        from fused_memory.reconciliation.cli_stage_runner import StageResult

        stage, journal = self._make_stage_with_real_journal_dir(tmp_path)
        watermark = Watermark(project_id='test_project')

        call_log: list[str] = []
        captured: dict = {}

        async def _record(run_id, *, session_id, stage_cursor):
            call_log.append('record')

        async def _runner(**kwargs):
            call_log.append('runner')
            captured.update(kwargs)
            return StageResult(report={}, success=True)

        async def _clear(run_id):
            call_log.append('clear')

        journal.record_run_session.side_effect = _record
        journal.clear_run_session.side_effect = _clear

        with patch(
            'fused_memory.reconciliation.stages.base.run_stage_via_cli',
            new=AsyncMock(side_effect=_runner),
        ):
            await stage.run([], watermark, [], run_id='run-sc1')

        # (a) record_run_session BEFORE the runner; clear_run_session AFTER.
        assert call_log == ['record', 'runner', 'clear'], (
            f'expected mint-before-spawn / clear-after ordering, got {call_log!r}'
        )

        # record_run_session got (run_id, session_id=<uuid4>, stage_cursor=<stage>).
        journal.record_run_session.assert_awaited_once()
        rc = journal.record_run_session.await_args
        assert rc.args[0] == 'run-sc1'
        session_id = rc.kwargs['session_id']
        assert uuid.UUID(session_id).version == 4, 'session_id must be a uuid4 string'
        assert rc.kwargs['stage_cursor'] == StageId.memory_consolidator.value

        # (b) the runner received the SAME session_id and a per-run TaskConfigDir
        #     under <data_dir>/recon-config.
        assert captured['session_id'] == session_id
        cfg = captured['config_dir']
        assert isinstance(cfg, TaskConfigDir)
        assert cfg.path.parent == tmp_path / 'recon-config'
        assert cfg.path.name == 'claude-config-run-sc1'

        # (c) clear_run_session was awaited with the run_id.
        journal.clear_run_session.assert_awaited_once_with('run-sc1')

    @pytest.mark.asyncio
    async def test_clear_runs_even_when_runner_raises(self, tmp_path):
        stage, journal = self._make_stage_with_real_journal_dir(tmp_path)
        watermark = Watermark(project_id='test_project')

        with patch(
            'fused_memory.reconciliation.stages.base.run_stage_via_cli',
            new=AsyncMock(side_effect=RuntimeError('subprocess boom')),
        ), pytest.raises(RuntimeError):
            await stage.run([], watermark, [], run_id='run-sc2')

        # Mint-before-spawn happened, and clear ran in the finally despite the raise.
        journal.record_run_session.assert_awaited_once()
        journal.clear_run_session.assert_awaited_once_with('run-sc2')

    @pytest.mark.asyncio
    async def test_preserves_session_snapshot_on_cancel(self, tmp_path):
        """On asyncio.CancelledError from the runner (a restart interrupting an
        in-flight stage), the session snapshot is PRESERVED — clear_run_session
        is NOT called — so the interrupted run row retains session_id +
        stage_cursor for the startup adopt-and-resume pass (task σ).

        Contrast test_clear_runs_even_when_runner_raises: a NON-cancel exception
        still clears (only a genuine cancellation preserves the snapshot)."""
        stage, journal = self._make_stage_with_real_journal_dir(tmp_path)
        watermark = Watermark(project_id='test_project')

        with patch(
            'fused_memory.reconciliation.stages.base.run_stage_via_cli',
            new=AsyncMock(side_effect=asyncio.CancelledError()),
        ), pytest.raises(asyncio.CancelledError):
            await stage.run([], watermark, [], run_id='run-cancel')

        # Mint-before-spawn happened...
        journal.record_run_session.assert_awaited_once()
        # ...but clear_run_session was NOT called on cancellation — the snapshot
        # survives on the interrupted run row for the resume pass.
        journal.clear_run_session.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_clear_called_exactly_once_on_normal_return(self, tmp_path):
        """Complement to the cancel case: a normal runner return DOES clear the
        session snapshot, exactly once."""
        from fused_memory.reconciliation.cli_stage_runner import StageResult

        stage, journal = self._make_stage_with_real_journal_dir(tmp_path)
        watermark = Watermark(project_id='test_project')

        with patch(
            'fused_memory.reconciliation.stages.base.run_stage_via_cli',
            new=AsyncMock(return_value=StageResult(report={}, success=True)),
        ):
            await stage.run([], watermark, [], run_id='run-normal')

        journal.clear_run_session.assert_awaited_once_with('run-normal')
