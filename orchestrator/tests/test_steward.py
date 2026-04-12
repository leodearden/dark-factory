"""Tests for the persistent TaskSteward."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from escalation.models import Escalation

from orchestrator.steward import StewardMetrics, TaskSteward, _is_timeout_kill

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
    config.steward_max_attempts = 3
    config.steward_completion_timeout = 300.0
    config.steward_max_timeouts_per_escalation = 3
    config.timeouts.steward = 1800.0
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


def _make_result(
    cost=1.0, turns=5, session_id='sess-abc', success=True,
    duration_ms=5000, stderr='', timed_out=False,
):
    from shared.cli_invoke import AgentResult
    return AgentResult(
        success=success,
        output='done',
        stderr=stderr,
        cost_usd=cost,
        duration_ms=duration_ms,
        turns=turns,
        session_id=session_id,
        timed_out=timed_out,
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

    async def test_auto_escalates_after_max_attempts(self, steward, mock_config):
        mock_config.steward_max_attempts = 2
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
        mock_config.steward_max_attempts = 1
        esc = _make_escalation()
        steward._retry_counts['esc-42-1'] = 1

        await steward._handle_escalation(esc)

        steward.escalation_queue.submit.assert_called_once()
        submitted = steward.escalation_queue.submit.call_args[0][0]
        assert submitted.level == 1
        assert 'Failed after 1 attempt:' in submitted.summary

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

    async def test_timeout_seconds_forwarded_across_cap_hit_retry(
        self, steward, mock_config,
    ):
        """Both the initial and cap-hit-recovery invocations must carry timeout_seconds."""
        mock_config.timeouts.steward = 900.0

        call_count = 0

        def detect_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return call_count == 1  # cap hit on first call only

        gate = MagicMock()
        gate.before_invoke = AsyncMock(return_value='token-a')
        gate.detect_cap_hit = MagicMock(side_effect=detect_side_effect)
        gate.on_agent_complete = MagicMock()
        gate.confirm_account_ok = MagicMock()
        steward.usage_gate = gate

        with (
            patch('orchestrator.steward.invoke_agent', new_callable=AsyncMock) as mock_invoke,
            patch('orchestrator.steward.asyncio.sleep', new_callable=AsyncMock),
        ):
            mock_invoke.return_value = _make_result(session_id='sess-t')
            await steward._invoke_with_session(
                prompt='do work',
                cwd=steward.worktree,
                mcp_config={},
                per_invocation_budget=5.0,
                escalation=_make_escalation(),
            )

        assert mock_invoke.call_count == 2
        for call in mock_invoke.call_args_list:
            assert call.kwargs['timeout_seconds'] == pytest.approx(900.0)


# ---------------------------------------------------------------------------
# _is_timeout_kill helper — unit test each branch directly
# ---------------------------------------------------------------------------


class TestIsTimeoutKill:
    """Pin each branch of the _is_timeout_kill helper with direct calls."""

    def test_success_true_short_circuits_to_false(self):
        """success=True must return False regardless of timed_out or stderr."""
        result = _make_result(success=True, timed_out=True, stderr='Process killed after 900.0s timeout')
        assert _is_timeout_kill(result) is False

    def test_timed_out_true_returns_true_via_structured_path(self):
        """timed_out=True with empty stderr must return True (primary signal)."""
        result = _make_result(success=False, timed_out=True, stderr='')
        assert _is_timeout_kill(result) is True

    def test_stderr_fallback_marker_and_token_matches(self):
        """Marker phrase + 'timeout' token in stderr → fallback returns True."""
        result = _make_result(
            success=False, timed_out=False,
            stderr='Process killed after 900.0s timeout (SIGTERM+SIGKILL)',
        )
        assert _is_timeout_kill(result) is True

    def test_partial_stderr_missing_timeout_token_does_not_match(self):
        """Marker present but 'timeout' token absent and timed_out=False → False."""
        result = _make_result(
            success=False, timed_out=False,
            stderr='Process killed after foo',  # no 'timeout' token
        )
        assert _is_timeout_kill(result) is False


# ---------------------------------------------------------------------------
# Timeout-kill recovery
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStewardTimeoutKillRecovery:
    """Timeout-killed invocations must NOT consume the retry budget."""

    async def test_timeout_kill_does_not_increment_retry_count(self, steward, mock_config):
        """A SIGTERM+SIGKILL timeout must leave _retry_counts unchanged."""
        mock_config.steward_max_attempts = 2
        mock_config.timeouts.steward = 900.0
        esc = _make_escalation(id='esc-42-1')
        # Queue returns pending after the invocation (not resolved)
        steward.escalation_queue.get.return_value = _make_escalation(
            id='esc-42-1', status='pending',
        )

        with patch('orchestrator.steward.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = _make_result(
                success=False,
                cost=1.5,
                turns=3,
                session_id='sess-killed',
                stderr='Process killed after 900.0s timeout (SIGTERM+SIGKILL)',
            )
            await steward._handle_escalation(esc)

        assert steward._retry_counts.get('esc-42-1', 0) == 0
        steward.escalation_queue.submit.assert_not_called()

    async def test_timeout_kill_increments_timeouts_recovered_metric(
        self, steward, mock_config,
    ):
        """timeouts_recovered counter must tick; invocations==1, handled==0."""
        mock_config.steward_max_attempts = 2
        mock_config.timeouts.steward = 900.0
        esc = _make_escalation(id='esc-42-1')
        steward.escalation_queue.get.return_value = _make_escalation(
            id='esc-42-1', status='pending',
        )

        with patch('orchestrator.steward.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = _make_result(
                success=False, cost=1.5, turns=3, session_id='sess-killed',
                stderr='Process killed after 900.0s timeout (SIGTERM+SIGKILL)',
            )
            await steward._handle_escalation(esc)

        assert steward.metrics.timeouts_recovered == 1
        assert steward.metrics.invocations == 1
        assert steward.metrics.escalations_handled == 0

    async def test_timeout_kill_matches_terminated_stderr_pattern(
        self, steward, mock_config,
    ):
        """'Process terminated after …' pattern (SIGTERM; stream closed) must match."""
        mock_config.steward_max_attempts = 2
        mock_config.timeouts.steward = 900.0
        esc = _make_escalation(id='esc-42-1')
        steward.escalation_queue.get.return_value = _make_escalation(
            id='esc-42-1', status='pending',
        )

        with patch('orchestrator.steward.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = _make_result(
                success=False, cost=1.5, turns=3, session_id='sess-term',
                stderr='Process terminated after 900.0s timeout (SIGTERM); stream closed',
            )
            await steward._handle_escalation(esc)

        assert steward._retry_counts.get('esc-42-1', 0) == 0
        assert steward.metrics.timeouts_recovered == 1

    async def test_non_timeout_failure_still_increments_retry_count(
        self, steward, mock_config,
    ):
        """Non-timeout failures must still consume retry budget (regression guard)."""
        mock_config.steward_max_attempts = 2
        mock_config.timeouts.steward = 900.0
        esc = _make_escalation(id='esc-42-1')
        steward.escalation_queue.get.return_value = _make_escalation(
            id='esc-42-1', status='pending',
        )

        with patch('orchestrator.steward.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = _make_result(
                success=False, cost=0.5,
                stderr='Some other CLI error (not a timeout)',
            )
            await steward._handle_escalation(esc)

        assert steward._retry_counts.get('esc-42-1') == 1
        assert steward.metrics.timeouts_recovered == 0

    async def test_timeout_kill_still_tracks_cost_and_duration(
        self, steward, mock_config,
    ):
        """Cost and duration from killed invocation must flow into lifetime metrics."""
        mock_config.steward_max_attempts = 2
        mock_config.timeouts.steward = 900.0
        esc = _make_escalation(id='esc-42-1')
        steward.escalation_queue.get.return_value = _make_escalation(
            id='esc-42-1', status='pending',
        )

        with patch('orchestrator.steward.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = _make_result(
                success=False, cost=2.25, turns=7, duration_ms=900000,
                stderr='Process killed after 900.0s timeout (SIGTERM+SIGKILL)',
            )
            await steward._handle_escalation(esc)

        assert steward.metrics.total_cost_usd == pytest.approx(2.25)
        assert steward.metrics.total_duration_ms == 900000
        assert steward.metrics.invocations == 1

    async def test_timeout_kill_does_not_auto_escalate_even_at_retry_cap(
        self, steward, mock_config,
    ):
        """Timeout-kill on first attempt must NOT auto-escalate, even with max_retries=1."""
        mock_config.steward_max_attempts = 1
        mock_config.timeouts.steward = 900.0
        esc = _make_escalation(id='esc-42-77')
        # Fresh escalation — retry count starts at 0
        steward.escalation_queue.get.return_value = _make_escalation(
            id='esc-42-77', status='pending',
        )

        with patch('orchestrator.steward.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = _make_result(
                success=False, cost=1.0, turns=5,
                stderr='Process killed after 900.0s timeout (SIGTERM+SIGKILL)',
            )
            await steward._handle_escalation(esc)

        steward.escalation_queue.submit.assert_not_called()
        assert steward._retry_counts.get('esc-42-77', 0) == 0
        assert steward.metrics.timeouts_recovered == 1

    async def test_structured_timed_out_flag_triggers_recovery(
        self, steward, mock_config,
    ):
        """timed_out=True with empty stderr must still trigger recovery (primary signal)."""
        mock_config.steward_max_attempts = 2
        mock_config.timeouts.steward = 900.0
        esc = _make_escalation(id='esc-42-1')
        steward.escalation_queue.get.return_value = _make_escalation(
            id='esc-42-1', status='pending',
        )

        with patch('orchestrator.steward.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = _make_result(
                success=False,
                stderr='',  # empty — only structured field can drive recovery
                timed_out=True,
                cost=1.5,
                turns=3,
                session_id='sess-killed',
            )
            await steward._handle_escalation(esc)

        assert steward._retry_counts.get('esc-42-1', 0) == 0
        assert steward.metrics.timeouts_recovered == 1
        assert steward._timeout_counts.get('esc-42-1') == 1

    async def test_partial_stderr_match_is_not_treated_as_timeout(
        self, steward, mock_config,
    ):
        """Contains 'Process killed after' but lacks 'timeout' token and timed_out=False.

        Pins the AND-discipline of the stderr fallback: a future loosening
        (e.g. removing the 'timeout' token check) would fail this test.
        """
        mock_config.steward_max_attempts = 2
        mock_config.timeouts.steward = 900.0
        esc = _make_escalation(id='esc-42-1')
        steward.escalation_queue.get.return_value = _make_escalation(
            id='esc-42-1', status='pending',
        )

        with patch('orchestrator.steward.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = _make_result(
                success=False,
                stderr='note: previously Process killed after foo',
                timed_out=False,
                cost=0.5,
            )
            await steward._handle_escalation(esc)

        # Consumed a retry (not treated as timeout)
        assert steward._retry_counts.get('esc-42-1', 0) == 1
        assert steward.metrics.timeouts_recovered == 0
        assert steward._timeout_counts.get('esc-42-1', 0) == 0


# ---------------------------------------------------------------------------
# Timeout Cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStewardTimeoutCap:
    """_timeout_counts tracks consecutive timeout-kills; cap triggers auto-escalation."""

    def test_timeout_counts_dict_initialized_empty(self, steward):
        """_timeout_counts must be an empty dict right after construction."""
        assert steward._timeout_counts == {}

    async def test_timeout_kill_increments_timeout_count(
        self, steward, mock_config, caplog,
    ):
        """A SIGTERM+SIGKILL timeout must increment _timeout_counts but NOT _retry_counts.

        Also verifies the log message includes the 'timeout_count: N/M' suffix so
        operators can see the cap approaching.
        """
        mock_config.steward_max_timeouts_per_escalation = 3
        mock_config.steward_max_attempts = 2
        mock_config.timeouts.steward = 900.0
        esc = _make_escalation(id='esc-42-1')
        steward.escalation_queue.get.return_value = _make_escalation(
            id='esc-42-1', status='pending',
        )

        with patch('orchestrator.steward.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = _make_result(
                success=False,
                cost=0.0,
                turns=0,
                session_id='sess-killed',
                stderr='Process killed after 900.0s timeout (SIGTERM+SIGKILL)',
            )
            with caplog.at_level(logging.WARNING):
                await steward._handle_escalation(esc)

        assert steward._timeout_counts.get('esc-42-1') == 1
        assert steward._retry_counts.get('esc-42-1', 0) == 0
        assert 'timeout_count: 1/3' in caplog.text

    async def test_non_timeout_failure_does_not_increment_timeout_count(
        self, steward, mock_config,
    ):
        """Non-timeout failures must NOT touch _timeout_counts (selectivity guard)."""
        mock_config.steward_max_timeouts_per_escalation = 3
        mock_config.steward_max_attempts = 2
        mock_config.timeouts.steward = 900.0
        esc = _make_escalation(id='esc-42-1')
        steward.escalation_queue.get.return_value = _make_escalation(
            id='esc-42-1', status='pending',
        )

        with patch('orchestrator.steward.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = _make_result(
                success=False,
                cost=1.0,
                turns=3,
                session_id='sess-normal-fail',
                stderr='some other error',
            )
            await steward._handle_escalation(esc)

        assert steward._timeout_counts.get('esc-42-1', 0) == 0
        assert steward._retry_counts.get('esc-42-1') == 1

    async def test_auto_escalates_when_timeout_count_at_cap(
        self, steward, mock_config,
    ):
        """When _timeout_counts[id] >= cap, guard must fire BEFORE invoke_agent."""
        mock_config.steward_max_timeouts_per_escalation = 2
        esc = _make_escalation(id='esc-42-1')
        # Pre-seed timeout count at cap
        steward._timeout_counts['esc-42-1'] = 2

        with patch('orchestrator.steward.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            await steward._handle_escalation(esc)

        # Guard fires before any invocation
        mock_invoke.assert_not_called()
        # Level-1 re-escalation was submitted
        steward.escalation_queue.submit.assert_called_once()
        submitted_esc = steward.escalation_queue.submit.call_args[0][0]
        assert submitted_esc.level == 1
        assert 'repeatedly timed out' in submitted_esc.summary.lower()
        # Detail carries the timed-out reason so reviewers can diagnose downstream
        assert 'timed out' in submitted_esc.detail.lower()
        # Metrics counter was incremented by _auto_escalate_to_human
        assert steward.metrics.escalations_reescalated == 1
        # Original escalation was dismissed; reason includes count/cap for observability
        steward.escalation_queue.resolve.assert_called_once_with(
            esc.id,
            'Auto-dismissed: re-escalated to level 1 — Invocation repeatedly timed out (2/2)',
            dismiss=True,
            resolved_by='steward',
        )

    async def test_timeout_cap_pops_counters(self, steward, mock_config):
        """When the timeout cap fires via _handle_escalation both counters are popped.

        Integration test through the timeout cap guard path (L231-236 in steward.py).
        After cap-fire the dicts must not retain stale entries for the escalation id.
        """
        mock_config.steward_max_timeouts_per_escalation = 2
        esc = _make_escalation(id='esc-42-1')
        # Pre-seed at cap so the guard fires immediately
        steward._timeout_counts['esc-42-1'] = 2
        steward._retry_counts['esc-42-1'] = 1

        with patch('orchestrator.steward.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            await steward._handle_escalation(esc)

        # Guard fires before any invocation
        mock_invoke.assert_not_called()
        # Auto-escalation was issued (level-1 submitted)
        steward.escalation_queue.submit.assert_called_once()
        submitted = steward.escalation_queue.submit.call_args[0][0]
        assert submitted.level == 1
        # Both counters popped — no stale entries remain
        assert 'esc-42-1' not in steward._timeout_counts
        assert 'esc-42-1' not in steward._retry_counts

    async def test_different_escalations_have_independent_timeout_counts(
        self, steward, mock_config,
    ):
        """Timeout counts for distinct escalation ids must not bleed across."""
        mock_config.steward_max_timeouts_per_escalation = 3
        mock_config.steward_max_attempts = 5
        mock_config.timeouts.steward = 900.0

        timeout_stderr = 'Process killed after 900.0s timeout (SIGTERM+SIGKILL)'

        esc1 = _make_escalation(id='esc-42-1')
        steward.escalation_queue.get.return_value = _make_escalation(
            id='esc-42-1', status='pending',
        )
        with patch('orchestrator.steward.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = _make_result(
                success=False, cost=0.0, turns=0, session_id='sess-k1',
                stderr=timeout_stderr,
            )
            await steward._handle_escalation(esc1)

        esc2 = _make_escalation(id='esc-42-2')
        steward.escalation_queue.get.return_value = _make_escalation(
            id='esc-42-2', status='pending',
        )
        with patch('orchestrator.steward.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = _make_result(
                success=False, cost=0.0, turns=0, session_id='sess-k2',
                stderr=timeout_stderr,
            )
            await steward._handle_escalation(esc2)

        assert steward._timeout_counts == {'esc-42-1': 1, 'esc-42-2': 1}

    async def test_repeated_timeout_kills_eventually_terminate(
        self, steward, mock_config,
    ):
        """Headline acceptance test: after cap timeouts the steward stops invoking."""
        mock_config.steward_max_timeouts_per_escalation = 3
        mock_config.steward_max_attempts = 10  # retry guard must not fire first
        mock_config.timeouts.steward = 900.0
        esc = _make_escalation(id='esc-42-inf')
        # Queue always returns pending (never resolved)
        steward.escalation_queue.get.return_value = _make_escalation(
            id='esc-42-inf', status='pending',
        )

        with patch('orchestrator.steward.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = _make_result(
                success=False,
                cost=0.0,           # realistic: streaming cut off mid-turn
                turns=0,
                duration_ms=900000,
                session_id='sess-inf',
                stderr='Process killed after 900.0s timeout (SIGTERM+SIGKILL)',
            )
            for _ in range(5):
                await steward._handle_escalation(esc)

        # invoke_agent called exactly 3 times (cap=3); calls 4 and 5 are blocked
        assert mock_invoke.call_count == 3
        # Level-1 re-escalation was submitted at least once
        steward.escalation_queue.submit.assert_called()
        first_submit = steward.escalation_queue.submit.call_args_list[0][0][0]
        assert first_submit.level == 1
        assert 'repeatedly timed out' in first_submit.summary.lower()
        # Timeout metric reflects the 3 actual invocations
        assert steward.metrics.timeouts_recovered == 3
        assert steward._timeout_counts['esc-42-inf'] == 3


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

    def test_pops_per_escalation_counters(self, steward):
        """_auto_escalate_to_human must pop _retry_counts and _timeout_counts for the id.

        Prevents slow memory leak: counters accumulate forever if not cleaned up
        on cap-fire paths (the success path already pops at L351-352).
        """
        esc = _make_escalation(id='esc-42-1')
        steward._retry_counts['esc-42-1'] = 2
        steward._timeout_counts['esc-42-1'] = 1

        steward._auto_escalate_to_human(esc, 'cap fired')

        assert 'esc-42-1' not in steward._retry_counts
        assert 'esc-42-1' not in steward._timeout_counts


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

    def test_timeouts_recovered_initial_value(self):
        m = StewardMetrics()
        assert m.timeouts_recovered == 0


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

    def test_default_steward_max_attempts_is_one(self, monkeypatch, tmp_path):
        """steward_max_attempts default must be 1 (the renamed field)."""
        from orchestrator.config import OrchestratorConfig

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = OrchestratorConfig()
        assert config.steward_max_attempts == 1

    def test_default_steward_wall_clock_timeout_is_1800(self, monkeypatch, tmp_path):
        """timeouts.steward default must be 1800s (per-invocation wall-clock)."""
        from orchestrator.config import OrchestratorConfig

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = OrchestratorConfig()
        assert config.timeouts.steward == 1800.0

    def test_default_steward_max_timeouts_per_escalation_is_3(
        self, monkeypatch, tmp_path,
    ):
        """steward_max_timeouts_per_escalation default must be 3."""
        from orchestrator.config import OrchestratorConfig

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = OrchestratorConfig()
        assert config.steward_max_timeouts_per_escalation == 3

    def test_timeout_cap_default_respects_policy_window(self, monkeypatch, tmp_path):
        """Default steward_max_timeouts_per_escalation must be in the policy range [2, 5]."""
        from orchestrator.config import OrchestratorConfig

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = OrchestratorConfig()
        assert 2 <= config.steward_max_timeouts_per_escalation <= 5
