"""Tests for PathScopeAdjudicator — LLM Stage-2 gate on path-scope-guard heuristic HITs.

Test layout (mirrors the plan):
- step-1: success-path verdict parsing (allow / reject / uncertain)
- step-3: fail-safe on LLM trouble (timeout, empty-output, malformed, exception)
- step-5: consecutive-ZOT circuit breaker
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from fused_memory.config.schema import PathScopeAdjudicatorConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adjudicator(
    config: PathScopeAdjudicatorConfig | None = None,
    *,
    tmp_path=None,
    escalator=None,
):
    """Construct a PathScopeAdjudicator with minimal wiring."""
    from fused_memory.middleware.path_scope_adjudicator import PathScopeAdjudicator

    cfg = config or PathScopeAdjudicatorConfig()
    return PathScopeAdjudicator(
        config=cfg,
        usage_gate=None,
        cwd=tmp_path,
        scope_violation_escalator=escalator,
    )


def _agent_result(
    success: bool = True,
    structured_output=None,
    output: str = '',
    timed_out: bool = False,
    subtype: str | None = None,
    cost_usd: float = 0.0,
    turns: int = 0,
):
    """Build a minimal AgentResult for patching."""
    from shared.cli_invoke import AgentResult

    return AgentResult(
        success=success,
        output=output,
        structured_output=structured_output,
        timed_out=timed_out,
        subtype=subtype or '',
        cost_usd=cost_usd,
        turns=turns,
    )


async def _do_adjudicate(adj, **kwargs):
    """Call adjudicate with sensible defaults filled in."""
    defaults = dict(
        title='Update orchestrator/harness.py to fix a bug',
        description='Fix a crash in the harness',
        matched_paths=('orchestrator/',),
        project_id='dark_factory',
        suggested_project='orchestrator',
        project_root='/some/project',
    )
    defaults.update(kwargs)
    return await adj.adjudicate(**defaults)


# ===========================================================================
# Step-1: success-path verdict parsing
# ===========================================================================


_INVOKE_PATH = 'fused_memory.middleware.path_scope_adjudicator.invoke_with_cap_retry'


@pytest.mark.asyncio
class TestSuccessPathParsing:
    """Adjudicator parses structured LLM output into AdjudicationVerdict correctly."""

    async def test_allow_verdict_should_allow_creation(self, tmp_path):
        """Clean allow → verdict.verdict=='allow', should_allow_creation is True."""
        adj = _make_adjudicator(tmp_path=tmp_path)
        result = _agent_result(
            success=True,
            structured_output={'verdict': 'allow', 'reason': 'incidental example mention'},
        )
        with patch(_INVOKE_PATH, new=AsyncMock(return_value=result)) as mock_invoke:
            verdict = await _do_adjudicate(adj)

        assert verdict.verdict == 'allow'
        assert verdict.should_allow_creation is True
        assert 'incidental example mention' in verdict.reason
        # invoke was called
        mock_invoke.assert_awaited_once()
        assert mock_invoke.await_args is not None
        call_kwargs = mock_invoke.await_args.kwargs
        # model from config
        assert call_kwargs.get('model') == PathScopeAdjudicatorConfig().model
        # output_schema set
        assert call_kwargs.get('output_schema') is not None
        # disallowed_tools=['*']
        assert call_kwargs.get('disallowed_tools') == ['*']
        # bounded timeout
        assert call_kwargs.get('timeout_seconds') is not None
        assert call_kwargs['timeout_seconds'] <= PathScopeAdjudicatorConfig().timeout_seconds

    async def test_reject_verdict_blocks_creation(self, tmp_path):
        """verdict='reject' → should_allow_creation is False."""
        adj = _make_adjudicator(tmp_path=tmp_path)
        result = _agent_result(
            success=True,
            structured_output={'verdict': 'reject', 'reason': 'genuine misroute'},
        )
        with patch(_INVOKE_PATH, new=AsyncMock(return_value=result)):
            verdict = await _do_adjudicate(adj)

        assert verdict.verdict == 'reject'
        assert verdict.should_allow_creation is False

    async def test_uncertain_verdict_blocks_creation(self, tmp_path):
        """verdict='uncertain' → should_allow_creation is False."""
        adj = _make_adjudicator(tmp_path=tmp_path)
        result = _agent_result(
            success=True,
            structured_output={'verdict': 'uncertain', 'reason': 'cannot determine'},
        )
        with patch(_INVOKE_PATH, new=AsyncMock(return_value=result)):
            verdict = await _do_adjudicate(adj)

        assert verdict.verdict == 'uncertain'
        assert verdict.should_allow_creation is False

    async def test_invoke_called_with_permission_mode_bypass(self, tmp_path):
        """invoke_with_cap_retry is called with permission_mode='bypassPermissions'."""
        adj = _make_adjudicator(tmp_path=tmp_path)
        result = _agent_result(
            success=True,
            structured_output={'verdict': 'allow', 'reason': 'ok'},
        )
        with patch(_INVOKE_PATH, new=AsyncMock(return_value=result)) as mock_invoke:
            await _do_adjudicate(adj)

        assert mock_invoke.await_args is not None
        call_kwargs = mock_invoke.await_args.kwargs
        assert call_kwargs.get('permission_mode') == 'bypassPermissions'

    async def test_llm_used_flag_set_on_success(self, tmp_path):
        """AdjudicationVerdict.llm_used is True when invoke was called and succeeded."""
        adj = _make_adjudicator(tmp_path=tmp_path)
        result = _agent_result(
            success=True,
            structured_output={'verdict': 'allow', 'reason': 'ok'},
        )
        with patch(_INVOKE_PATH, new=AsyncMock(return_value=result)):
            verdict = await _do_adjudicate(adj)

        assert verdict.llm_used is True


# ===========================================================================
# Step-3: fail-safe on LLM trouble (hard constraint — never silently allow)
# ===========================================================================


@pytest.mark.asyncio
class TestFailSafeOnLLMTrouble:
    """Adjudicator never allows creation when the LLM fails in any way."""

    async def test_timeout_returns_fail_safe(self, tmp_path):
        """AgentResult with timed_out=True → fail-safe verdict (no creation allowed)."""
        adj = _make_adjudicator(tmp_path=tmp_path)
        result = _agent_result(
            success=False, output='', timed_out=True, subtype='error',
        )
        with patch(_INVOKE_PATH, new=AsyncMock(return_value=result)):
            verdict = await _do_adjudicate(adj)

        assert verdict.should_allow_creation is False
        assert verdict.failed is True

    async def test_zero_output_returns_fail_safe(self, tmp_path):
        """Zero-output AgentResult → fail-safe verdict, failed=True."""
        adj = _make_adjudicator(tmp_path=tmp_path)
        # Zero-output means timed out with no turns (is_zero_output_timeout).
        result = _agent_result(
            success=False, output='', timed_out=True, subtype='error',
        )
        with patch(_INVOKE_PATH, new=AsyncMock(return_value=result)):
            verdict = await _do_adjudicate(adj)

        assert verdict.should_allow_creation is False
        assert verdict.failed is True

    async def test_success_with_none_structured_output_returns_fail_safe(self, tmp_path):
        """success=True but structured_output=None → treated as uncertain, fails safe."""
        adj = _make_adjudicator(tmp_path=tmp_path)
        result = _agent_result(success=True, structured_output=None)
        with patch(_INVOKE_PATH, new=AsyncMock(return_value=result)):
            verdict = await _do_adjudicate(adj)

        assert verdict.should_allow_creation is False

    async def test_malformed_output_missing_verdict_returns_fail_safe(self, tmp_path):
        """success=True but structured_output missing 'verdict' → fail-safe."""
        adj = _make_adjudicator(tmp_path=tmp_path)
        result = _agent_result(success=True, structured_output={'reason': 'no verdict key'})
        with patch(_INVOKE_PATH, new=AsyncMock(return_value=result)):
            verdict = await _do_adjudicate(adj)

        assert verdict.should_allow_creation is False

    async def test_runtime_error_returns_fail_safe_no_propagation(self, tmp_path):
        """invoke_with_cap_retry raises RuntimeError → fail-safe verdict, no propagation."""
        adj = _make_adjudicator(tmp_path=tmp_path)
        with patch(_INVOKE_PATH, side_effect=RuntimeError('boom')):
            verdict = await _do_adjudicate(adj)  # must not raise

        assert verdict.should_allow_creation is False

    async def test_all_accounts_capped_returns_fail_safe_no_propagation(self, tmp_path):
        """AllAccountsCappedException → fail-safe verdict (not allow), no propagation."""
        from shared.cli_invoke import AllAccountsCappedException

        adj = _make_adjudicator(tmp_path=tmp_path)
        exc = AllAccountsCappedException(retries=3, elapsed_secs=300.0, label='test')
        with patch(_INVOKE_PATH, side_effect=exc):
            verdict = await _do_adjudicate(adj)  # must not raise

        assert verdict.should_allow_creation is False


# ===========================================================================
# Step-5: consecutive-ZOT circuit breaker
# ===========================================================================


@pytest.mark.asyncio
class TestZeroOutputBreakerCircuit:
    """Circuit breaker short-circuits the LLM after consecutive ZOT failures."""

    async def test_breaker_open_skips_llm_call(self, tmp_path):
        """When the breaker is open, adjudicate returns fail-safe WITHOUT calling the LLM."""
        cfg = PathScopeAdjudicatorConfig(zero_output_breaker_threshold=1)
        adj = _make_adjudicator(config=cfg, tmp_path=tmp_path)

        # A real ZOT: timed_out=True, turns=0, cost_usd=0.0.
        zot_result = _agent_result(
            success=False, output='', timed_out=True, subtype='error',
        )
        # First call opens the breaker (threshold=1).
        with patch(_INVOKE_PATH, new=AsyncMock(return_value=zot_result)) as mock_invoke:
            first = await _do_adjudicate(adj)
            assert first.should_allow_creation is False
            assert mock_invoke.await_count == 1

        # Second call: breaker is open; LLM must NOT be called.
        with patch(_INVOKE_PATH, new=AsyncMock(return_value=zot_result)) as mock_invoke2:
            second = await _do_adjudicate(adj)

        assert second.should_allow_creation is False
        assert 'breaker' in second.reason.lower()
        assert mock_invoke2.await_count == 0, 'LLM must not be called when breaker is open'

    async def test_threshold_opens_breaker(self, tmp_path):
        """After zero_output_breaker_threshold consecutive ZOTs the breaker trips."""
        threshold = 2
        cfg = PathScopeAdjudicatorConfig(zero_output_breaker_threshold=threshold)
        adj = _make_adjudicator(config=cfg, tmp_path=tmp_path)

        # A real ZOT: timed_out=True, turns=0, cost_usd=0.0.
        zot_result = _agent_result(
            success=False, output='', timed_out=True, subtype='error',
        )
        with patch(_INVOKE_PATH, new=AsyncMock(return_value=zot_result)):
            for _ in range(threshold):
                await _do_adjudicate(adj)

        # After threshold ZOTs, next call should short-circuit without calling LLM.
        with patch(_INVOKE_PATH, new=AsyncMock(return_value=zot_result)) as mock_probe:
            after = await _do_adjudicate(adj)

        assert after.should_allow_creation is False
        assert mock_probe.await_count == 0, 'Breaker must be open after threshold ZOTs'

    async def test_successful_call_resets_breaker(self, tmp_path):
        """A successful LLM call resets the ZOT counter so a subsequent ZOT doesn't trip the breaker.

        Test setup (threshold=2):
          1. Drive ONE ZOT → counter becomes 1, breaker stays CLOSED.
          2. Successful call → counter resets to 0  (the code path under test).
          3. Drive ONE more ZOT → counter becomes 1 again, still < threshold.
          4. Assert the mock was reached (breaker did NOT open), proving the
             success in step 2 actually cleared the accumulated count.
        """
        threshold = 2
        cfg = PathScopeAdjudicatorConfig(
            zero_output_breaker_threshold=threshold,
            zero_output_breaker_cooldown_seconds=9999,
        )
        adj = _make_adjudicator(config=cfg, tmp_path=tmp_path)

        zot_result = _agent_result(
            success=False, output='', timed_out=True, subtype='error',
        )
        ok_result = _agent_result(
            success=True, structured_output={'verdict': 'allow', 'reason': 'ok'},
        )

        # Step 1: one ZOT — counter=1, breaker CLOSED (threshold=2).
        with patch(_INVOKE_PATH, new=AsyncMock(return_value=zot_result)):
            first = await _do_adjudicate(adj)
        assert first.should_allow_creation is False
        assert adj._consecutive_zero_output_timeouts == 1

        # Step 2: successful call — should reset counter to 0.
        with patch(_INVOKE_PATH, new=AsyncMock(return_value=ok_result)):
            good = await _do_adjudicate(adj)
        assert good.should_allow_creation is True
        assert adj._consecutive_zero_output_timeouts == 0, (
            'Successful call must reset the ZOT counter'
        )

        # Step 3 + 4: one more ZOT → counter=1, still < threshold; LLM IS reached.
        with patch(_INVOKE_PATH, new=AsyncMock(return_value=zot_result)) as mock_probe:
            after = await _do_adjudicate(adj)
        assert after.should_allow_creation is False
        assert mock_probe.await_count == 1, (
            'Breaker must remain CLOSED after one ZOT when success reset the counter'
        )
        assert adj._consecutive_zero_output_timeouts == 1


# ===========================================================================
# Step-3: budget-misconfig signal (cost>0 error_max_budget_usd → loud escalation)
# ===========================================================================


@pytest.mark.asyncio
class TestBudgetMisconfigSignal:
    """Adjudicator emits a loud config-defect signal for cost>0 error_max_budget_usd.

    Key invariants:
    - fail-safe preserved: should_allow_creation=False, failed=True
    - escalator.report_budget_misconfig called exactly once with cost_usd/turns/model
    - ZOT breaker counter NOT incremented (not a hang)
    - DISTINCT warning logged (names budget-misconfiguration, not the generic 'LLM call failed')
    """

    async def test_budget_misconfig_calls_escalator_once(self, tmp_path):
        """error_max_budget_usd + cost_usd>0 → escalator.report_budget_misconfig called once."""
        from unittest.mock import Mock

        mock_escalator = Mock()
        mock_escalator.report_budget_misconfig = Mock(return_value='esc-fake-1')
        adj = _make_adjudicator(tmp_path=tmp_path, escalator=mock_escalator)
        result = _agent_result(
            success=False, subtype='error_max_budget_usd', cost_usd=0.11, turns=2,
        )
        with patch(_INVOKE_PATH, new=AsyncMock(return_value=result)):
            verdict = await _do_adjudicate(adj)

        mock_escalator.report_budget_misconfig.assert_called_once()
        call_kwargs = mock_escalator.report_budget_misconfig.call_args.kwargs
        assert call_kwargs['cost_usd'] == 0.11
        assert call_kwargs['turns'] == 2
        # Carries model and max_budget_usd from config.
        from fused_memory.config.schema import PathScopeAdjudicatorConfig
        assert call_kwargs['model'] == PathScopeAdjudicatorConfig().model
        assert call_kwargs['max_budget_usd'] == PathScopeAdjudicatorConfig().max_budget_usd
        # Fail-safe preserved.
        assert verdict.should_allow_creation is False
        assert verdict.failed is True

    async def test_budget_misconfig_reason_names_config_defect(self, tmp_path):
        """Verdict reason explicitly names the budget-misconfiguration."""
        from unittest.mock import Mock

        adj = _make_adjudicator(tmp_path=tmp_path, escalator=Mock())
        result = _agent_result(
            success=False, subtype='error_max_budget_usd', cost_usd=0.11, turns=2,
        )
        with patch(_INVOKE_PATH, new=AsyncMock(return_value=result)):
            verdict = await _do_adjudicate(adj)

        assert 'budget' in verdict.reason.lower() or 'misconfig' in verdict.reason.lower()

    async def test_budget_misconfig_does_not_increment_zot_counter(self, tmp_path):
        """Budget-misconfig path must NOT increment the ZOT breaker counter."""
        from unittest.mock import Mock

        adj = _make_adjudicator(tmp_path=tmp_path, escalator=Mock())
        result = _agent_result(
            success=False, subtype='error_max_budget_usd', cost_usd=0.11, turns=2,
        )
        with patch(_INVOKE_PATH, new=AsyncMock(return_value=result)):
            await _do_adjudicate(adj)

        assert adj._consecutive_zero_output_timeouts == 0, (
            'ZOT breaker must not be incremented for a deterministic budget-config defect'
        )

    async def test_budget_misconfig_warning_logged(self, tmp_path, caplog):
        """A DISTINCT WARNING is logged naming the budget-misconfiguration."""
        import logging
        from unittest.mock import Mock

        adj = _make_adjudicator(tmp_path=tmp_path, escalator=Mock())
        result = _agent_result(
            success=False, subtype='error_max_budget_usd', cost_usd=0.11, turns=2,
        )
        with caplog.at_level(logging.WARNING), patch(_INVOKE_PATH, new=AsyncMock(return_value=result)):
            await _do_adjudicate(adj)

        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            'budget' in m.lower() or 'misconfig' in m.lower() for m in warning_msgs
        ), f'expected a budget-misconfig WARNING; got: {warning_msgs}'

    async def test_genuine_zot_does_not_call_escalator(self, tmp_path):
        """Negative: genuine ZOT (timed_out=True, cost=0) does NOT call report_budget_misconfig."""
        from unittest.mock import Mock

        mock_escalator = Mock()
        mock_escalator.report_budget_misconfig = Mock()
        adj = _make_adjudicator(tmp_path=tmp_path, escalator=mock_escalator)
        # Real ZOT: timed_out=True, cost=0 → is_zero_output_timeout returns True.
        result = _agent_result(
            success=False, subtype='error', cost_usd=0.0, turns=0, timed_out=True,
        )
        with patch(_INVOKE_PATH, new=AsyncMock(return_value=result)):
            await _do_adjudicate(adj)

        mock_escalator.report_budget_misconfig.assert_not_called()
        # ZOT counter IS incremented for a genuine hang.
        assert adj._consecutive_zero_output_timeouts == 1

    async def test_zero_cost_budget_subtype_does_not_call_escalator(self, tmp_path):
        """Edge: subtype='error_max_budget_usd' with cost_usd=0.0 → NOT a config defect."""
        from unittest.mock import Mock

        mock_escalator = Mock()
        mock_escalator.report_budget_misconfig = Mock()
        adj = _make_adjudicator(tmp_path=tmp_path, escalator=mock_escalator)
        # cost_usd=0.0 → guard is not met; falls through to generic failure handling.
        result = _agent_result(
            success=False, subtype='error_max_budget_usd', cost_usd=0.0, turns=0,
        )
        with patch(_INVOKE_PATH, new=AsyncMock(return_value=result)):
            await _do_adjudicate(adj)

        mock_escalator.report_budget_misconfig.assert_not_called()

    async def test_no_escalator_budget_misconfig_still_returns_fail_safe(self, tmp_path):
        """No escalator wired (escalator=None) → fail-safe still returned, no exception."""
        adj = _make_adjudicator(tmp_path=tmp_path, escalator=None)
        result = _agent_result(
            success=False, subtype='error_max_budget_usd', cost_usd=0.11, turns=2,
        )
        with patch(_INVOKE_PATH, new=AsyncMock(return_value=result)):
            verdict = await _do_adjudicate(adj)  # must not raise

        assert verdict.should_allow_creation is False
        assert verdict.failed is True
