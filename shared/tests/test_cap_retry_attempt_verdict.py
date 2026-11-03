"""Each cap-retry attempt carries exactly one verdict."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _usage_gate_test_helpers import make_gate, make_mock_cost_store

from shared.cli_invoke import AgentResult, invoke_with_cap_retry
from shared.invocation_outcome import CapHit
from shared.testing import make_gate_mock

_CAP_STDERR = "You've hit your usage limit resets in 3h"


def _report_cap_then_confirm(slot) -> None:
    slot.report(CapHit(resets_at=None, reason='cap'))
    slot.confirm(1.0)


def _detect_cap_then_confirm(slot) -> None:
    assert slot.detect_cap_hit(_CAP_STDERR, '')
    slot.confirm(1.0)


_SETTLE_THEN_CONFIRM = pytest.mark.parametrize(
    'drive',
    [_report_cap_then_confirm, _detect_cap_then_confirm],
    ids=['report', 'detect_cap_hit'],
)


async def test_confirm_after_report_adds_no_cost_and_keeps_the_cap():
    gate = make_gate(['a'])
    try:
        async with gate.invoke_slot() as slot:
            _report_cap_then_confirm(slot)
        assert gate.cumulative_cost == 0.0
        assert gate.active_account_name is None
    finally:
        await gate.shutdown()


async def test_confirm_on_settled_slot_logs_the_dropped_cost(caplog):
    gate = make_gate(['a'])
    try:
        with caplog.at_level(logging.DEBUG, logger='shared.usage_gate'):
            async with gate.invoke_slot() as slot:
                _report_cap_then_confirm(slot)
        assert any(
            r.levelno == logging.DEBUG and 'already-settled' in r.getMessage()
            and '$1.0000' in r.getMessage()
            for r in caplog.records
        )
    finally:
        await gate.shutdown()


async def test_confirm_on_unsettled_slot_accumulates_cost():
    gate = make_gate(['a'])
    try:
        async with gate.invoke_slot() as slot:
            slot.confirm(0.75)
        assert gate.cumulative_cost == pytest.approx(0.75)
    finally:
        await gate.shutdown()


async def _production_verdict_calls(drive) -> tuple[int, int]:
    gate = make_gate(['a'])
    try:
        with (
            patch.object(
                gate, 'on_agent_complete', wraps=gate.on_agent_complete,
            ) as on_agent_complete,
            patch.object(
                gate, 'confirm_account_ok', wraps=gate.confirm_account_ok,
            ) as confirm_account_ok,
        ):
            async with gate.invoke_slot() as slot:
                drive(slot)
        return on_agent_complete.call_count, confirm_account_ok.call_count
    finally:
        await gate.shutdown()


async def _double_verdict_calls(drive) -> tuple[int, int]:
    gate = make_gate_mock(detect_cap_hit=MagicMock(return_value=True))
    async with gate.invoke_slot() as slot:
        drive(slot)
    return gate.on_agent_complete.call_count, gate.confirm_account_ok.call_count


@_SETTLE_THEN_CONFIRM
async def test_gate_double_matches_production_on_confirm_after_settling(drive):
    """(on_agent_complete calls, confirm_account_ok calls) per party."""
    parties = {
        'production': await _production_verdict_calls(drive),
        'make_gate_mock': await _double_verdict_calls(drive),
    }
    assert parties == dict.fromkeys(parties, (0, 0))


def _stale_lease_then_success(gate):
    """invoke_fn whose first attempt loses its lease to a sibling's cap."""
    calls: list[dict] = []

    async def invoke_fn(**kw):
        calls.append(dict(kw))
        if len(calls) == 1:
            gate.detect_cap_hit(_CAP_STDERR, '', oauth_token=kw['oauth_token'])
            return AgentResult(
                success=False, output='', stderr='', cost_usd=0.0, turns=1, duration_ms=100,
            )
        return AgentResult(success=True, output='done', cost_usd=1.23)

    return invoke_fn, calls


async def test_success_after_unattributed_resume_fallback_is_confirmed_and_uncapped():
    gate = make_gate(['a', 'b'])
    cost_store = make_mock_cost_store()
    invoke_fn, calls = _stale_lease_then_success(gate)
    try:
        result = await invoke_with_cap_retry(
            gate,
            'lbl',
            invoke_fn=invoke_fn,
            cost_store=cost_store,
            prompt='real task',
            resume_session_id='sess-crashed',
        )
        assert (result.success, result.resume_fallbacks, result.account_name) == (True, 1, 'b')
        assert calls[0]['resume_session_id'] == 'sess-crashed'
        assert 'resume_session_id' not in calls[1]

        assert gate.cumulative_cost == pytest.approx(1.23)
        cost_store.save_invocation.assert_awaited_once()
        row = cost_store.save_invocation.await_args.kwargs
        assert (row['capped'], row['capped_reason'], row['account_name']) == (False, None, 'b')
    finally:
        await gate.shutdown()


async def test_declined_retry_after_unattributed_attempt_keeps_that_attempts_verdict():
    gate = make_gate(['a', 'b'])
    cost_store = make_mock_cost_store()
    invoke_fn, calls = _stale_lease_then_success(gate)
    try:
        result = await invoke_with_cap_retry(
            gate,
            'lbl',
            invoke_fn=invoke_fn,
            cost_store=cost_store,
            prompt='real task',
            resume_session_id='sess-crashed',
            rebuild_prompt=AsyncMock(return_value=None),
        )
        assert result.retry_aborted is True
        assert len(calls) == 1
        cost_store.save_invocation.assert_awaited_once()
        row = cost_store.save_invocation.await_args.kwargs
        assert (row['capped'], row['capped_reason']) == (True, 'account')
    finally:
        await gate.shutdown()
