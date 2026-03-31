"""Tests for the persistent TaskSteward."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from escalation.models import Escalation

from orchestrator.steward import StewardMetrics, TaskSteward

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def worktree(tmp_path: Path) -> Path:
    wt = tmp_path / 'worktree'
    wt.mkdir()
    task_dir = wt / '.task'
    task_dir.mkdir()
    (task_dir / 'metadata.json').write_text(json.dumps({'task_id': '42'}))
    (task_dir / 'plan.json').write_text(json.dumps({'steps': []}))
    return wt


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.project_root = Path('/tmp/fake-project')
    config.models.steward = 'opus'
    config.budgets.steward = 5.0
    config.max_turns.steward = 100
    config.effort.steward = 'high'
    config.backends.steward = 'claude'
    config.escalation.host = 'localhost'
    config.escalation.port = 8102
    config.fused_memory.url = 'http://localhost:8002'
    config.fused_memory.project_id = 'dark_factory'
    config.steward_lifetime_budget = 12.0
    config.steward_max_retries = 3
    config.steward_completion_timeout = 300.0
    config.suggestion_triage_threshold = 10
    return config


@pytest.fixture
def mock_queue(tmp_path: Path):
    queue = MagicMock()
    queue.queue_dir = tmp_path / 'escalations'
    queue.queue_dir.mkdir()
    queue.get_by_task.return_value = []
    queue.get.return_value = None
    queue.make_id.return_value = 'esc-42-99'
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
    briefing.build_steward_initial_prompt.return_value = 'Full steward briefing.'
    briefing.build_steward_continuation_prompt.return_value = 'New escalation details.'
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


def _make_result(cost=1.0, turns=5, session_id='sess-abc', success=True):
    from shared.cli_invoke import AgentResult
    return AgentResult(
        success=success,
        output='done',
        cost_usd=cost,
        duration_ms=5000,
        turns=turns,
        session_id=session_id,
    )


def _make_escalation(**overrides):  # type: ignore[no-untyped-def]
    defaults: dict = dict(
        id='esc-42-1',
        task_id='42',
        agent_role='orchestrator',
        severity='blocking',
        category='limit_exhausted',
        summary='execute limit exhausted',
    )
    defaults.update(overrides)
    return Escalation(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Session Persistence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStewardSessionPersistence:

    async def test_first_invocation_uses_initial_prompt(self, steward, mock_briefing):
        esc = _make_escalation()
        steward.escalation_queue.get.return_value = _make_escalation(
            status='resolved', resolution='fixed',
        )

        with patch('orchestrator.steward.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = _make_result(session_id='sess-first')
            await steward._handle_escalation(esc)

            mock_briefing.build_steward_initial_prompt.assert_called_once()
            mock_briefing.build_steward_continuation_prompt.assert_not_called()
            assert 'resume_session_id' not in mock_invoke.call_args.kwargs

    async def test_second_invocation_uses_resume(self, steward, mock_briefing):
        steward._session_id = 'sess-first'
        esc = _make_escalation(id='esc-42-2')
        steward.escalation_queue.get.return_value = _make_escalation(
            id='esc-42-2', status='resolved', resolution='fixed',
        )

        with patch('orchestrator.steward.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = _make_result(session_id='sess-second')
            await steward._handle_escalation(esc)

            mock_briefing.build_steward_continuation_prompt.assert_called_once()
            mock_briefing.build_steward_initial_prompt.assert_not_called()
            assert mock_invoke.call_args.kwargs['resume_session_id'] == 'sess-first'

    async def test_session_id_captured_from_result(self, steward):
        esc = _make_escalation()
        steward.escalation_queue.get.return_value = _make_escalation(
            status='resolved', resolution='fixed',
        )

        with patch('orchestrator.steward.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = _make_result(session_id='sess-captured')
            await steward._handle_escalation(esc)

        assert steward._session_id == 'sess-captured'


# ---------------------------------------------------------------------------
# Cap-Hit Backoff
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStewardCapHitBackoff:

    async def test_sleeps_before_prompt_rebuild_on_cap_hit(self, steward, mock_briefing):
        """Steward sleeps _CAP_HIT_COOLDOWN_SECS before retrying on cap hit."""
        from orchestrator.steward import _CAP_HIT_COOLDOWN_SECS

        esc = _make_escalation()
        steward.escalation_queue.get.return_value = _make_escalation(
            status='resolved', resolution='fixed',
        )

        call_count = 0

        def detect_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return call_count == 1  # cap hit on first call only

        gate = MagicMock()
        gate.before_invoke = AsyncMock(return_value='token-a')
        gate.detect_cap_hit = MagicMock(side_effect=detect_side_effect)
        gate.on_agent_complete = MagicMock()
        steward.usage_gate = gate

        with (
            patch('orchestrator.steward.invoke_agent', new_callable=AsyncMock) as mock_invoke,
            patch('orchestrator.steward.asyncio.sleep', new_callable=AsyncMock) as mock_sleep,
        ):
            mock_invoke.return_value = _make_result(session_id='sess-new')
            await steward._handle_escalation(esc)

            mock_sleep.assert_called_once_with(_CAP_HIT_COOLDOWN_SECS)
            assert mock_invoke.call_count == 2
            # Session was reset and recaptured
            assert steward._session_id == 'sess-new'

    async def test_no_sleep_when_no_cap_hit(self, steward):
        """No sleep when invocation succeeds without cap hit."""
        esc = _make_escalation()
        steward.escalation_queue.get.return_value = _make_escalation(
            status='resolved', resolution='fixed',
        )

        gate = MagicMock()
        gate.before_invoke = AsyncMock(return_value='token-a')
        gate.detect_cap_hit = MagicMock(return_value=False)
        gate.on_agent_complete = MagicMock()
        steward.usage_gate = gate

        with (
            patch('orchestrator.steward.invoke_agent', new_callable=AsyncMock) as mock_invoke,
            patch('orchestrator.steward.asyncio.sleep', new_callable=AsyncMock) as mock_sleep,
        ):
            mock_invoke.return_value = _make_result()
            await steward._handle_escalation(esc)

            mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# Account Name Threading
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStewardAccountName:

    async def test_account_name_set_on_result(self, steward, worktree, mock_mcp):
        """_invoke_with_session stamps account_name from usage_gate on the result."""
        gate = MagicMock()
        gate.before_invoke = AsyncMock(return_value='token-a')
        gate.active_account_name = 'max-d'
        gate.detect_cap_hit = MagicMock(return_value=False)
        gate.on_agent_complete = MagicMock()
        steward.usage_gate = gate

        esc = _make_escalation()
        mcp_config = mock_mcp.mcp_config_json()

        with patch('orchestrator.steward.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = _make_result(session_id='sess-x')
            result = await steward._invoke_with_session(
                prompt='test prompt',
                cwd=worktree,
                mcp_config=mcp_config,
                per_invocation_budget=2.0,
                escalation=esc,
            )

        assert result.account_name == 'max-d'

    async def test_account_name_reflects_failover_on_cap_hit(
        self, steward, worktree, mock_mcp, mock_briefing,
    ):
        """After cap hit + session reset, account_name reflects the retry account."""
        from unittest.mock import PropertyMock

        call_count = 0

        def detect_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return call_count == 1  # cap hit on first call only

        gate = MagicMock()
        gate.before_invoke = AsyncMock(return_value='token-a')
        gate.detect_cap_hit = MagicMock(side_effect=detect_side_effect)
        gate.on_agent_complete = MagicMock()
        # active_account_name is read twice in cap-hit path: once for capture (loop 1),
        # once inside continue path (loop 2 capture)
        type(gate).active_account_name = PropertyMock(
            side_effect=['max-d', 'max-c'],
        )
        steward.usage_gate = gate

        esc = _make_escalation()
        mcp_config = mock_mcp.mcp_config_json()

        with (
            patch('orchestrator.steward.invoke_agent', new_callable=AsyncMock) as mock_invoke,
            patch('orchestrator.steward.asyncio.sleep', new_callable=AsyncMock),
        ):
            mock_invoke.return_value = _make_result(session_id='sess-new')
            result = await steward._invoke_with_session(
                prompt='test prompt',
                cwd=worktree,
                mcp_config=mcp_config,
                per_invocation_budget=2.0,
                escalation=esc,
            )

        assert result.account_name == 'max-c'


# ---------------------------------------------------------------------------
# Retry Logic
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStewardRetryLogic:

    async def test_retry_count_increments_on_unresolved(self, steward):
        esc = _make_escalation()
        steward.escalation_queue.get.return_value = _make_escalation(status='pending')

        with patch('orchestrator.steward.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = _make_result()
            await steward._handle_escalation(esc)

        assert steward._retry_counts.get('esc-42-1') == 1

    async def test_auto_escalates_after_max_retries(self, steward, mock_config):
        mock_config.steward_max_retries = 2
        esc = _make_escalation()
        steward._retry_counts['esc-42-1'] = 2

        await steward._handle_escalation(esc)

        steward.escalation_queue.submit.assert_called_once()
        submitted = steward.escalation_queue.submit.call_args[0][0]
        assert submitted.level == 1
        assert 'Failed after 2 attempts' in submitted.summary

        steward.escalation_queue.resolve.assert_called_once()
        assert steward.escalation_queue.resolve.call_args[1].get('dismiss') is True

    async def test_different_escalations_have_independent_counts(self, steward):
        for esc_id in ('esc-42-1', 'esc-42-2'):
            esc = _make_escalation(id=esc_id)
            steward.escalation_queue.get.return_value = _make_escalation(
                id=esc_id, status='pending',
            )
            with patch('orchestrator.steward.invoke_agent', new_callable=AsyncMock) as mock_invoke:
                mock_invoke.return_value = _make_result()
                await steward._handle_escalation(esc)

        assert steward._retry_counts == {'esc-42-1': 1, 'esc-42-2': 1}


# ---------------------------------------------------------------------------
# Lifetime Budget
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStewardLifetimeBudget:

    async def test_tracks_cumulative_cost(self, steward):
        esc = _make_escalation()
        steward.escalation_queue.get.return_value = _make_escalation(
            status='resolved', resolution='fixed',
        )
        with patch('orchestrator.steward.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = _make_result(cost=2.5)
            await steward._handle_escalation(esc)

        assert steward.metrics.total_cost_usd == pytest.approx(2.5)

    async def test_auto_escalates_on_budget_exhaustion(self, steward, mock_config):
        mock_config.steward_lifetime_budget = 5.0
        steward.metrics.total_cost_usd = 6.0

        esc = _make_escalation()
        await steward._handle_escalation(esc)

        steward.escalation_queue.submit.assert_called_once()
        submitted = steward.escalation_queue.submit.call_args[0][0]
        assert submitted.level == 1
        assert 'budget exhausted' in submitted.summary.lower()

    async def test_per_invocation_budget_capped_by_remaining(self, steward, mock_config):
        mock_config.steward_lifetime_budget = 12.0
        mock_config.budgets.steward = 5.0
        steward.metrics.total_cost_usd = 10.0

        esc = _make_escalation()
        steward.escalation_queue.get.return_value = _make_escalation(
            status='resolved', resolution='fixed',
        )

        with patch('orchestrator.steward.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = _make_result(cost=1.5)
            await steward._handle_escalation(esc)
            assert mock_invoke.call_args.kwargs['max_budget_usd'] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Unified Role
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStewardUnifiedRole:

    async def test_blocking_escalation_uses_worktree_cwd(self, steward, worktree):
        esc = _make_escalation(category='limit_exhausted')
        steward.escalation_queue.get.return_value = _make_escalation(
            status='resolved', resolution='fixed',
        )
        with patch('orchestrator.steward.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = _make_result()
            await steward._handle_escalation(esc)
            assert mock_invoke.call_args.kwargs['cwd'] == worktree

    async def test_suggestions_use_project_root_cwd(self, steward, mock_config):
        esc = _make_escalation(category='review_suggestions', severity='info', detail='[]')
        steward.escalation_queue.get.return_value = _make_escalation(
            category='review_suggestions', status='resolved', resolution='triaged',
        )
        with patch('orchestrator.steward.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = _make_result()
            await steward._handle_escalation(esc)
            assert mock_invoke.call_args.kwargs['cwd'] == mock_config.project_root

    async def test_same_role_for_all_escalation_types(self, steward):
        from orchestrator.agents.roles import STEWARD
        for category in ('limit_exhausted', 'review_suggestions', 'review_issues'):
            esc = _make_escalation(category=category, severity='info')
            steward.escalation_queue.get.return_value = _make_escalation(
                category=category, status='resolved', resolution='done',
            )
            steward._session_id = None
            with patch('orchestrator.steward.invoke_agent', new_callable=AsyncMock) as mock_invoke:
                mock_invoke.return_value = _make_result()
                await steward._handle_escalation(esc)
                assert mock_invoke.call_args.kwargs['system_prompt'] == STEWARD.system_prompt


# ---------------------------------------------------------------------------
# Auto-Escalation
# ---------------------------------------------------------------------------


class TestStewardAutoEscalation:

    def test_creates_level1_with_diagnostic(self, steward):
        esc = _make_escalation()
        steward._auto_escalate_to_human(esc, 'test reason')

        submitted = steward.escalation_queue.submit.call_args[0][0]
        assert submitted.level == 1
        assert submitted.task_id == '42'
        assert 'test reason' in submitted.summary

    def test_dismisses_original(self, steward):
        esc = _make_escalation()
        steward._auto_escalate_to_human(esc, 'test reason')

        call_args = steward.escalation_queue.resolve.call_args
        assert call_args[0][0] == 'esc-42-1'
        assert call_args[1].get('dismiss') is True

    def test_tracks_reescalation_metric(self, steward):
        esc = _make_escalation()
        steward._auto_escalate_to_human(esc, 'test reason')
        assert steward.metrics.escalations_reescalated == 1


# ---------------------------------------------------------------------------
# Run Loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStewardRunLoop:

    async def test_stops_when_stopped_flag_set(self, steward):
        steward._stopped = True
        await steward._run_loop()

    async def test_continues_on_none_and_stops_via_flag(self, steward):
        """_run_loop does NOT exit on None — it retries until _stopped."""
        call_count = 0

        async def _next_esc():
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                steward._stopped = True
            return None

        with (
            patch.object(steward, '_next_escalation', side_effect=_next_esc),
            patch('orchestrator.steward.asyncio.sleep', new_callable=AsyncMock) as mock_sleep,
        ):
            await steward._run_loop()
            # Should have retried (slept) at least once before _stopped was set
            assert mock_sleep.call_count >= 1
            mock_sleep.assert_called_with(1)

    async def test_handles_multiple_sequential_escalations(self, steward):
        esc1 = _make_escalation(id='esc-42-1')
        esc2 = _make_escalation(id='esc-42-2')
        call_count = 0

        async def _next_esc():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return esc1
            elif call_count == 2:
                return esc2
            steward._stopped = True
            return None

        handle_mock = AsyncMock()
        with (
            patch.object(steward, '_next_escalation', side_effect=_next_esc),
            patch.object(steward, '_handle_escalation', handle_mock),
            patch('orchestrator.steward.asyncio.sleep', new_callable=AsyncMock),
        ):
            await steward._run_loop()
            assert handle_mock.call_count == 2

    async def test_continues_after_transient_none(self, steward):
        """A transient None doesn't prevent handling the next escalation."""
        esc = _make_escalation(id='esc-42-1')
        call_count = 0

        async def _next_esc():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return None  # transient failure
            if call_count == 2:
                return esc
            steward._stopped = True
            return None

        handle_mock = AsyncMock()
        with (
            patch.object(steward, '_next_escalation', side_effect=_next_esc),
            patch.object(steward, '_handle_escalation', handle_mock),
            patch('orchestrator.steward.asyncio.sleep', new_callable=AsyncMock),
        ):
            await steward._run_loop()
            handle_mock.assert_called_once_with(esc)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestStewardMetrics:

    def test_initial_values(self):
        m = StewardMetrics()
        assert m.invocations == 0
        assert m.total_cost_usd == 0.0
        assert m.escalations_reescalated == 0


# ---------------------------------------------------------------------------
# Next Escalation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestNextEscalation:

    async def test_returns_existing_pending(self, steward):
        esc = _make_escalation()
        steward.escalation_queue.get_by_task.return_value = [esc]
        result = await steward._next_escalation()
        assert result is esc

    async def test_returns_none_when_watcher_fails(self, steward):
        steward.escalation_queue.get_by_task.return_value = []
        with patch('asyncio.create_subprocess_exec', new_callable=AsyncMock) as mock_exec:
            proc = AsyncMock()
            proc.returncode = 1
            proc.communicate.return_value = (b'', b'error')
            mock_exec.return_value = proc
            result = await steward._next_escalation()
            assert result is None

    async def test_returns_none_for_non_level0_from_watcher(self, steward):
        """Defense-in-depth: discard watcher results that aren't level 0."""
        level1_esc = _make_escalation(level=1)
        steward.escalation_queue.get_by_task.return_value = []
        with patch('asyncio.create_subprocess_exec', new_callable=AsyncMock) as mock_exec:
            proc = AsyncMock()
            proc.returncode = 0
            proc.communicate.return_value = (
                level1_esc.to_json().encode(), b'',
            )
            mock_exec.return_value = proc
            result = await steward._next_escalation()
            assert result is None

    async def test_passes_level_filter_to_watcher(self, steward):
        """Steward spawns watcher with --level 0."""
        steward.escalation_queue.get_by_task.return_value = []
        with patch('asyncio.create_subprocess_exec', new_callable=AsyncMock) as mock_exec:
            proc = AsyncMock()
            proc.returncode = 1
            proc.communicate.return_value = (b'', b'')
            mock_exec.return_value = proc
            await steward._watch_for_escalation()
            cmd = mock_exec.call_args[0]
            assert '--level' in cmd
            level_idx = cmd.index('--level')
            assert cmd[level_idx + 1] == '0'


# ---------------------------------------------------------------------------
# Tests: CostStore constructor params
# ---------------------------------------------------------------------------


class TestStewardCostStoreConstructor:
    """TaskSteward accepts cost_store, run_id, and project_id as optional params."""

    def test_defaults_to_none_when_not_provided(
        self, worktree, mock_config, mock_queue, mock_mcp, mock_briefing
    ):
        """cost_store, run_id, project_id are None/'' by default."""
        s = TaskSteward(
            task_id='42',
            task={'id': '42', 'title': 'Test Task', 'description': 'A test'},
            worktree=worktree,
            config=mock_config,
            mcp=mock_mcp,
            escalation_queue=mock_queue,
            briefing=mock_briefing,
        )
        assert s._cost_store is None
        assert s._run_id == ''
        assert s._project_id == ''

    def test_stores_cost_store_as_attribute(
        self, worktree, mock_config, mock_queue, mock_mcp, mock_briefing
    ):
        """When cost_store is provided it is accessible as self._cost_store."""
        from unittest.mock import MagicMock
        mock_cost_store = MagicMock()
        s = TaskSteward(
            task_id='42',
            task={'id': '42', 'title': 'Test Task', 'description': 'A test'},
            worktree=worktree,
            config=mock_config,
            mcp=mock_mcp,
            escalation_queue=mock_queue,
            briefing=mock_briefing,
            cost_store=mock_cost_store,
            run_id='run-steward12345',
            project_id='dark_factory',
        )
        assert s._cost_store is mock_cost_store
        assert s._run_id == 'run-steward12345'
        assert s._project_id == 'dark_factory'

    def test_existing_steward_fixture_still_works(self, steward):
        """steward fixture (no cost_store args) still constructs correctly."""
        assert steward._cost_store is None
        assert steward._run_id == ''
        assert steward._project_id == ''


# ---------------------------------------------------------------------------
# Tests: CostStore save_invocation called after successful invocation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStewardSaveInvocation:
    """_invoke_with_session() records invocation to cost_store on success."""

    async def test_saves_invocation_after_success(
        self, worktree, mock_config, mock_queue, mock_mcp, mock_briefing
    ):
        """save_invocation is called after a successful invoke_agent call."""
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_cost_store = MagicMock()
        mock_cost_store.save_invocation = AsyncMock()

        s = TaskSteward(
            task_id='42',
            task={'id': '42', 'title': 'Test Task', 'description': 'A test'},
            worktree=worktree,
            config=mock_config,
            mcp=mock_mcp,
            escalation_queue=mock_queue,
            briefing=mock_briefing,
            cost_store=mock_cost_store,
            run_id='run-steward99',
            project_id='dark_factory',
        )

        esc = _make_escalation()
        from shared.cli_invoke import AgentResult as _AR
        invoke_result = _AR(
            success=True, output='done',
            cost_usd=0.75, duration_ms=3000, session_id='sess-x',
        )

        with patch('orchestrator.steward.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = invoke_result
            await s._invoke_with_session(
                prompt='Handle this',
                cwd=worktree,
                mcp_config={'mcpServers': {}},
                per_invocation_budget=5.0,
                escalation=esc,
            )

        mock_cost_store.save_invocation.assert_awaited_once()
        call_kwargs = mock_cost_store.save_invocation.call_args.kwargs
        assert call_kwargs['run_id'] == 'run-steward99'
        assert call_kwargs['task_id'] == '42'
        assert call_kwargs['project_id'] == 'dark_factory'
        assert call_kwargs['role'] == 'steward'
        assert call_kwargs['cost_usd'] == pytest.approx(0.75)
        assert call_kwargs['duration_ms'] == 3000
        assert call_kwargs['capped'] is False
        assert 'started_at' in call_kwargs
        assert 'completed_at' in call_kwargs
        assert 'model' in call_kwargs

    async def test_does_not_save_on_cap_hit_retry(
        self, worktree, mock_config, mock_queue, mock_mcp, mock_briefing
    ):
        """save_invocation is NOT called when a cap-hit triggers retry."""
        from unittest.mock import AsyncMock, MagicMock, patch, call

        mock_cost_store = MagicMock()
        mock_cost_store.save_invocation = AsyncMock()
        mock_usage_gate = MagicMock()
        mock_usage_gate.before_invoke = AsyncMock(return_value='tok')
        mock_usage_gate.active_account_name = 'acct-A'
        # First call: cap hit; second call: success
        mock_usage_gate.detect_cap_hit = MagicMock(side_effect=[True, False])
        mock_usage_gate.on_agent_complete = MagicMock()

        s = TaskSteward(
            task_id='42',
            task={'id': '42', 'title': 'Test Task', 'description': 'A test'},
            worktree=worktree,
            config=mock_config,
            mcp=mock_mcp,
            escalation_queue=mock_queue,
            briefing=mock_briefing,
            usage_gate=mock_usage_gate,
            cost_store=mock_cost_store,
            run_id='run-captest',
            project_id='dark_factory',
        )

        esc = _make_escalation()
        cap_result = _make_result(cost=0.0, session_id=None)
        success_result = _make_result(cost=1.0, session_id='sess-after-cap')

        with patch('orchestrator.steward.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            with patch('asyncio.sleep', new_callable=AsyncMock):
                mock_invoke.side_effect = [cap_result, success_result]
                await s._invoke_with_session(
                    prompt='Handle',
                    cwd=worktree,
                    mcp_config={'mcpServers': {}},
                    per_invocation_budget=5.0,
                    escalation=esc,
                )

        # save_invocation called exactly once (after success, not on cap-hit)
        mock_cost_store.save_invocation.assert_awaited_once()

    async def test_no_save_when_cost_store_is_none(
        self, worktree, mock_config, mock_queue, mock_mcp, mock_briefing
    ):
        """When cost_store is None, invocation recording is silently skipped."""
        from unittest.mock import AsyncMock, patch

        s = TaskSteward(
            task_id='42',
            task={'id': '42', 'title': 'Test Task', 'description': 'A test'},
            worktree=worktree,
            config=mock_config,
            mcp=mock_mcp,
            escalation_queue=mock_queue,
            briefing=mock_briefing,
            cost_store=None,
        )

        esc = _make_escalation()

        with patch('orchestrator.steward.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = _make_result()
            # Should not raise even with no cost_store
            result = await s._invoke_with_session(
                prompt='Handle',
                cwd=worktree,
                mcp_config={'mcpServers': {}},
                per_invocation_budget=5.0,
                escalation=esc,
            )

        assert result is not None
