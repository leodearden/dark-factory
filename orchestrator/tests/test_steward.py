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
    config.timeouts.steward = 900.0
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

    async def test_sleeps_before_retry_on_cap_hit(self, steward, mock_briefing):
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
            assert steward._session_id == 'sess-new'

    async def test_preserves_session_on_cap_hit(self, steward, mock_briefing):
        """Cap hit with session_id → session preserved, resume on next account."""
        # Set an existing session so _handle_escalation takes the continuation path
        steward._session_id = 'sess-existing'
        esc = _make_escalation()
        steward.escalation_queue.get.return_value = _make_escalation(
            status='resolved', resolution='fixed',
        )

        call_count = 0

        def detect_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return call_count == 1

        gate = MagicMock()
        gate.before_invoke = AsyncMock(side_effect=['token-a', 'token-b'])
        gate.detect_cap_hit = MagicMock(side_effect=detect_side_effect)
        gate.on_agent_complete = MagicMock()
        steward.usage_gate = gate

        with (
            patch('orchestrator.steward.invoke_agent', new_callable=AsyncMock) as mock_invoke,
            patch('orchestrator.steward.asyncio.sleep', new_callable=AsyncMock),
        ):
            mock_invoke.return_value = _make_result(session_id='sess-capped')
            await steward._handle_escalation(esc)

            # Second call should resume with the capped session
            second_call = mock_invoke.call_args_list[1]
            assert second_call.kwargs.get('resume_session_id') == 'sess-capped'
            # Briefing should NOT have been rebuilt (was continuation, not initial)
            mock_briefing.build_steward_initial_prompt.assert_not_called()

    async def test_rebuilds_prompt_only_when_no_session(self, steward, mock_briefing):
        """Cap hit with empty session_id → full prompt rebuild in _invoke_with_session."""
        esc = _make_escalation()
        steward.escalation_queue.get.return_value = _make_escalation(
            status='resolved', resolution='fixed',
        )

        call_count = 0

        def detect_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return call_count == 1

        gate = MagicMock()
        gate.before_invoke = AsyncMock(side_effect=['token-a', 'token-b'])
        gate.detect_cap_hit = MagicMock(side_effect=detect_side_effect)
        gate.on_agent_complete = MagicMock()
        steward.usage_gate = gate

        with (
            patch('orchestrator.steward.invoke_agent', new_callable=AsyncMock) as mock_invoke,
            patch('orchestrator.steward.asyncio.sleep', new_callable=AsyncMock),
        ):
            # First call: cap hit with no session_id
            mock_invoke.side_effect = [
                _make_result(session_id=''),
                _make_result(session_id='sess-new'),
            ]
            await steward._handle_escalation(esc)

            # Called twice: once in _handle_escalation (initial prompt, _session_id=None)
            # and once in _invoke_with_session (cap hit rebuild, no session to resume)
            assert mock_briefing.build_steward_initial_prompt.call_count == 2
            # Second invoke call should NOT have resume_session_id
            second_call = mock_invoke.call_args_list[1]
            assert 'resume_session_id' not in second_call.kwargs

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

    async def test_auto_escalates_after_one_attempt_with_default_retries(
        self, steward, mock_config,
    ):
        mock_config.steward_max_retries = 1
        esc = _make_escalation()
        steward._retry_counts['esc-42-1'] = 1

        await steward._handle_escalation(esc)

        steward.escalation_queue.submit.assert_called_once()
        submitted = steward.escalation_queue.submit.call_args[0][0]
        assert submitted.level == 1
        assert 'Failed after 1 attempts' in submitted.summary

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
# Timeout Passthrough
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStewardTimeoutPassthrough:

    async def test_invoke_with_session_passes_timeout_seconds(self, steward, mock_config):
        """_invoke_with_session must forward config.timeouts.steward as timeout_seconds."""
        mock_config.timeouts.steward = 900.0

        with patch('orchestrator.steward.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = _make_result(session_id='sess-t')
            await steward._invoke_with_session(
                prompt='do work',
                cwd=steward.worktree,
                mcp_config={},
                per_invocation_budget=5.0,
                escalation=_make_escalation(),
            )

        assert mock_invoke.call_args.kwargs['timeout_seconds'] == pytest.approx(900.0)


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

    async def test_suggestions_use_worktree_cwd(self, steward, worktree):
        esc = _make_escalation(category='review_suggestions', severity='info', detail='[]')
        steward.escalation_queue.get.return_value = _make_escalation(
            category='review_suggestions', status='resolved', resolution='triaged',
        )
        with patch('orchestrator.steward.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = _make_result()
            await steward._handle_escalation(esc)
            assert mock_invoke.call_args.kwargs['cwd'] == worktree

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
# Config Defaults
# ---------------------------------------------------------------------------


class TestStewardDefaultConfig:

    def test_default_steward_max_retries_is_one(self, monkeypatch, tmp_path):
        """steward_max_retries default must be 1 (one attempt, zero retries)."""
        from orchestrator.config import OrchestratorConfig

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = OrchestratorConfig()
        assert config.steward_max_retries == 1
