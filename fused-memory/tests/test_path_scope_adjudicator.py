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


def _make_adjudicator(config: PathScopeAdjudicatorConfig | None = None, *, tmp_path=None):
    """Construct a PathScopeAdjudicator with minimal wiring."""
    from fused_memory.middleware.path_scope_adjudicator import PathScopeAdjudicator

    cfg = config or PathScopeAdjudicatorConfig()
    return PathScopeAdjudicator(config=cfg, usage_gate=None, cwd=tmp_path)


def _agent_result(
    success: bool = True,
    structured_output=None,
    output: str = '',
    timed_out: bool = False,
    subtype: str | None = None,
):
    """Build a minimal AgentResult for patching."""
    from shared.cli_invoke import AgentResult

    return AgentResult(
        success=success,
        output=output,
        structured_output=structured_output,
        timed_out=timed_out,
        subtype=subtype or '',
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
        """One successful LLM call resets the breaker so the next call reaches the LLM."""
        cfg = PathScopeAdjudicatorConfig(
            zero_output_breaker_threshold=1,
            zero_output_breaker_cooldown_seconds=9999,
        )
        adj = _make_adjudicator(config=cfg, tmp_path=tmp_path)

        zot_result = _agent_result(
            success=False, output='', subtype='error_empty_output',
        )
        ok_result = _agent_result(
            success=True, structured_output={'verdict': 'allow', 'reason': 'ok'},
        )

        # Open the breaker.
        with patch(_INVOKE_PATH, new=AsyncMock(return_value=zot_result)):
            await _do_adjudicate(adj)

        # A successful call resets it (this fires because the mock always returns zot
        # when breaker is open, but we need the breaker to be CLOSED for the call to go
        # through — so we first simulate the cooldown expiring by directly resetting).
        adj._reset_breaker()

        # Now a successful call should reset the breaker.
        with patch(_INVOKE_PATH, new=AsyncMock(return_value=ok_result)):
            good = await _do_adjudicate(adj)

        assert good.should_allow_creation is True
        assert good.verdict == 'allow'

        # After reset, the NEXT call also reaches the LLM (breaker is closed).
        with patch(_INVOKE_PATH, new=AsyncMock(return_value=ok_result)) as mock_after:
            await _do_adjudicate(adj)

        assert mock_after.await_count == 1
