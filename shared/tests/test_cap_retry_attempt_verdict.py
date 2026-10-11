"""Each cap-retry attempt carries exactly one verdict."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from _usage_gate_test_helpers import make_gate

from shared.invocation_outcome import CapHit
from shared.testing import make_gate_mock


def _report_cap_then_confirm(slot) -> None:
    slot.report(CapHit(resets_at=None, reason='cap'))
    slot.confirm(1.0)


async def test_confirm_after_report_adds_no_cost_and_keeps_the_cap():
    gate = make_gate(['a'])
    try:
        async with gate.invoke_slot() as slot:
            _report_cap_then_confirm(slot)
        assert gate.cumulative_cost == 0.0
        assert gate.active_account_name is None
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
    gate = make_gate_mock()
    async with gate.invoke_slot() as slot:
        drive(slot)
    return gate.on_agent_complete.call_count, gate.confirm_account_ok.call_count


async def test_gate_double_matches_production_on_confirm_after_report():
    """(on_agent_complete calls, confirm_account_ok calls) per party."""
    parties = {
        'production': await _production_verdict_calls(_report_cap_then_confirm),
        'make_gate_mock': await _double_verdict_calls(_report_cap_then_confirm),
    }
    assert parties == dict.fromkeys(parties, (0, 0))
