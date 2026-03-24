"""Tests for the TaskSteward process steward."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.steward import StewardMetrics, TaskSteward


@pytest.fixture
def worktree(tmp_path: Path) -> Path:
    wt = tmp_path / 'worktree'
    wt.mkdir()
    task_dir = wt / '.task'
    task_dir.mkdir()
    (task_dir / 'metadata.json').write_text(json.dumps({
        'task_id': '42', 'title': 'Test', 'description': 'desc',
    }))
    (task_dir / 'plan.json').write_text(json.dumps({
        'task_id': '42', 'title': 'Test',
        'steps': [{'id': 'step-1', 'status': 'pending'}],
    }))
    return wt


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.project_root = Path('/tmp/fake-project')
    config.models.steward = 'opus'
    config.budgets.steward = 15.0
    config.max_turns.steward = 100
    config.effort.steward = 'max'
    config.backends.steward = 'claude'
    config.models.steward_triage = 'opus'
    config.budgets.steward_triage = 10.0
    config.max_turns.steward_triage = 50
    config.effort.steward_triage = 'high'
    config.backends.steward_triage = 'claude'
    config.escalation.host = 'localhost'
    config.escalation.port = 8102
    config.fused_memory.url = 'http://localhost:8002'
    config.fused_memory.project_id = 'dark_factory'
    return config


@pytest.fixture
def mock_queue(tmp_path: Path):
    queue = MagicMock()
    queue.queue_dir = tmp_path / 'escalations'
    queue.queue_dir.mkdir()
    queue.get_by_task.return_value = []
    return queue


@pytest.fixture
def mock_mcp():
    mcp = MagicMock()
    mcp.url = 'http://localhost:8002'
    mcp.mcp_config_json.return_value = {'mcpServers': {}}
    return mcp


@pytest.fixture
def mock_briefing():
    briefing = AsyncMock()
    briefing.build_steward_prompt.return_value = 'Handle this escalation.'
    briefing.build_steward_triage_prompt.return_value = 'Triage these suggestions.'
    return briefing


@pytest.fixture
def steward(worktree, mock_config, mock_queue, mock_mcp, mock_briefing):
    return TaskSteward(
        task_id='42',
        task={'id': '42', 'title': 'Test Task', 'description': 'A test'},
        worktree=worktree,
        config=mock_config,
        mcp=mock_mcp,
        escalation_queue=mock_queue,
        briefing=mock_briefing,
    )


class TestNextEscalation:
    @pytest.mark.asyncio
    async def test_returns_existing_pending(self, steward, mock_queue):
        """Should pick up an already-pending escalation without running the watcher."""
        from escalation.models import Escalation

        esc = Escalation(
            id='esc-42-1', task_id='42', agent_role='orchestrator',
            severity='blocking', category='limit_exhausted',
            summary='execute limit exhausted',
        )
        mock_queue.get_by_task.return_value = [esc]

        result = await steward._next_escalation()
        assert result is not None
        assert result.id == 'esc-42-1'

    @pytest.mark.asyncio
    async def test_returns_none_when_watcher_fails(self, steward, mock_queue):
        """When no pending and watcher exits with error, return None."""
        mock_queue.get_by_task.return_value = []

        with patch('orchestrator.steward.asyncio.create_subprocess_exec') as mock_exec:
            proc = AsyncMock()
            proc.returncode = 1
            proc.communicate.return_value = (b'', b'error')
            mock_exec.return_value = proc

            result = await steward._next_escalation()
            assert result is None


class TestHandleEscalation:
    @pytest.mark.asyncio
    async def test_invokes_agent_and_tracks_metrics(self, steward, mock_queue):
        from escalation.models import Escalation

        esc = Escalation(
            id='esc-42-1', task_id='42', agent_role='orchestrator',
            severity='blocking', category='limit_exhausted',
            summary='execute limit exhausted',
        )

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.cost_usd = 2.50
        mock_result.duration_ms = 30000
        mock_result.turns = 15

        with patch('orchestrator.steward.invoke_with_cap_retry', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = mock_result
            mock_queue.get_by_task.return_value = []  # resolved after invocation

            await steward._handle_escalation(esc)

            assert steward.metrics.invocations == 1
            assert steward.metrics.total_cost_usd == 2.50
            assert steward.metrics.escalations_handled == 1
            mock_invoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_passes_correct_role_config(self, steward, mock_config):
        from escalation.models import Escalation

        esc = Escalation(
            id='esc-42-1', task_id='42', agent_role='orchestrator',
            severity='blocking', category='limit_exhausted',
            summary='test',
        )

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.cost_usd = 0
        mock_result.duration_ms = 0
        mock_result.turns = 0

        with patch('orchestrator.steward.invoke_with_cap_retry', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = mock_result
            steward.escalation_queue.get_by_task.return_value = []

            await steward._handle_escalation(esc)

            call_kwargs = mock_invoke.call_args.kwargs
            assert call_kwargs['model'] == 'opus'
            assert call_kwargs['max_turns'] == 100
            assert call_kwargs['max_budget_usd'] == 15.0
            assert call_kwargs['effort'] == 'max'


class TestRunLoop:
    @pytest.mark.asyncio
    async def test_stops_when_stopped_flag_set(self, steward):
        """Loop should exit when _stopped is set."""
        steward._stopped = True
        await steward._run_loop()
        # Should exit immediately without error

    @pytest.mark.asyncio
    async def test_stops_when_no_escalation(self, steward):
        """Loop should exit when _next_escalation returns None."""
        with patch.object(steward, '_next_escalation', new_callable=AsyncMock) as mock_next:
            mock_next.return_value = None
            await steward._run_loop()
            mock_next.assert_called_once()


class TestTriageRouting:
    @pytest.mark.asyncio
    async def test_review_suggestions_uses_triage_role(self, steward, mock_config):
        """review_suggestions category should use STEWARD_TRIAGE role and config."""
        from escalation.models import Escalation

        suggestions = [
            {'reviewer': 'reuse_auditor', 'severity': 'suggestion',
             'location': 'src/foo.py:10', 'category': 'duplication',
             'description': 'Duplicate logic', 'suggested_fix': 'Extract helper'},
        ]
        esc = Escalation(
            id='esc-42-1', task_id='42', agent_role='orchestrator',
            severity='info', category='review_suggestions',
            summary='1 review suggestion(s) for triage',
            detail=json.dumps(suggestions),
        )

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.cost_usd = 5.0
        mock_result.duration_ms = 60000
        mock_result.turns = 20

        with patch('orchestrator.steward.invoke_with_cap_retry', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = mock_result
            steward.escalation_queue.get_by_task.return_value = []

            await steward._handle_escalation(esc)

            call_kwargs = mock_invoke.call_args.kwargs
            assert call_kwargs['model'] == 'opus'
            assert call_kwargs['max_budget_usd'] == 10.0
            assert call_kwargs['max_turns'] == 50
            assert call_kwargs['effort'] == 'high'
            assert 'triager' in call_kwargs['system_prompt'].lower()

    @pytest.mark.asyncio
    async def test_regular_escalation_uses_steward_role(self, steward, mock_config):
        """Non-suggestion escalation should use standard STEWARD role."""
        from escalation.models import Escalation

        esc = Escalation(
            id='esc-42-1', task_id='42', agent_role='orchestrator',
            severity='blocking', category='limit_exhausted',
            summary='test',
        )

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.cost_usd = 0
        mock_result.duration_ms = 0
        mock_result.turns = 0

        with patch('orchestrator.steward.invoke_with_cap_retry', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = mock_result
            steward.escalation_queue.get_by_task.return_value = []

            await steward._handle_escalation(esc)

            call_kwargs = mock_invoke.call_args.kwargs
            assert 'escalation handler' in call_kwargs['system_prompt'].lower()

    @pytest.mark.asyncio
    async def test_triage_runs_in_project_root(self, steward, mock_config, worktree):
        """Triage should run in project_root (post-merge), not worktree."""
        from escalation.models import Escalation

        esc = Escalation(
            id='esc-42-1', task_id='42', agent_role='orchestrator',
            severity='info', category='review_suggestions',
            summary='1 suggestion', detail='[]',
        )

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.cost_usd = 0
        mock_result.duration_ms = 0
        mock_result.turns = 0

        with patch('orchestrator.steward.invoke_with_cap_retry', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = mock_result
            steward.escalation_queue.get_by_task.return_value = []

            await steward._handle_escalation(esc)

            call_kwargs = mock_invoke.call_args.kwargs
            assert call_kwargs['cwd'] == mock_config.project_root

    @pytest.mark.asyncio
    async def test_triage_passes_parsed_suggestions(self, steward, mock_briefing):
        """Triage should pass parsed JSON suggestions to the briefing builder."""
        from escalation.models import Escalation

        suggestions = [
            {'reviewer': 'test_analyst', 'description': 'Missing edge case test'},
        ]
        esc = Escalation(
            id='esc-42-1', task_id='42', agent_role='orchestrator',
            severity='info', category='review_suggestions',
            summary='1 suggestion', detail=json.dumps(suggestions),
        )

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.cost_usd = 0
        mock_result.duration_ms = 0
        mock_result.turns = 0

        with patch('orchestrator.steward.invoke_with_cap_retry', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = mock_result
            steward.escalation_queue.get_by_task.return_value = []

            await steward._handle_escalation(esc)

            mock_briefing.build_steward_triage_prompt.assert_called_once()
            call_kwargs = mock_briefing.build_steward_triage_prompt.call_args.kwargs
            assert call_kwargs['suggestions'] == suggestions
            assert call_kwargs['escalation_id'] == 'esc-42-1'


class TestStewardMetrics:
    def test_initial_values(self):
        m = StewardMetrics()
        assert m.invocations == 0
        assert m.total_cost_usd == 0.0
        assert m.escalations_handled == 0
