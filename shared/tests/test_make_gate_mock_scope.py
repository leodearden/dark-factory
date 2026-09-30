"""Contract tests for scope-forwarding fidelity in ``shared.testing.make_gate_mock``.

Pins that the ``invoke_slot()`` double forwards the ``scope`` supplied to
``invoke_slot(scope=...)`` into every gate call production
``UsageGate.invoke_slot`` / ``InvokeSlot`` forward it into —
``before_invoke``, ``report()``'s CapHit/NearCap dispatch arms (task 4234),
and ``detect_cap_hit`` (task 4969).

Without this pin, a scope-forwarding regression could silently pass through
the double: every other ``make_gate_mock``-based suite either asserts only
``assert_called_once()`` / ``assert_not_called()`` on the affected gate
methods or, where it compares exact ``call_args``, drives only unscoped
invocations — so none of them checks what ``scope`` a scoped invocation
forwards.

Unscoped invocations must still forward ``scope=None`` explicitly:
production passes the ``scope`` kwarg unconditionally, never omitting it, so
the double must match that shape rather than silently dropping the kwarg
when it happens to be falsy.
"""

from __future__ import annotations

from datetime import UTC, datetime

from shared.invocation_outcome import CapHit, NearCap
from shared.testing import make_gate_mock

SCOPE = 'claude-fable-5'


async def test_report_cap_hit_forwards_scope_to_handle_cap_detected():
    gate = make_gate_mock()
    resets_at = datetime(2026, 7, 8, 12, 0, tzinfo=UTC)

    async with gate.invoke_slot(scope=SCOPE) as slot:
        slot.report(CapHit(resets_at=resets_at, reason='cap'))

    assert gate._handle_cap_detected.call_args.kwargs['scope'] == SCOPE


async def test_report_cap_hit_forwards_none_scope_when_unscoped():
    gate = make_gate_mock()

    async with gate.invoke_slot() as slot:
        slot.report(CapHit(resets_at=None, reason='cap'))

    assert gate._handle_cap_detected.call_args.kwargs['scope'] is None


async def test_report_near_cap_forwards_scope_to_handle_near_cap_warning():
    gate = make_gate_mock()

    async with gate.invoke_slot(scope=SCOPE) as slot:
        slot.report(NearCap(reason='close to limit'))

    assert gate._handle_near_cap_warning.call_args.kwargs['scope'] == SCOPE


async def test_invoke_slot_forwards_scope_to_before_invoke():
    gate = make_gate_mock()

    async with gate.invoke_slot(scope=SCOPE) as slot:
        slot.confirm()

    assert gate.before_invoke.call_args.kwargs['scope'] == SCOPE


async def test_detect_cap_hit_forwards_scope_to_gate():
    gate = make_gate_mock()

    async with gate.invoke_slot(scope=SCOPE) as slot:
        slot.detect_cap_hit('err', 'out')

    assert gate.detect_cap_hit.call_args.kwargs['scope'] == SCOPE


async def test_detect_cap_hit_forwards_none_scope_when_unscoped():
    gate = make_gate_mock()

    async with gate.invoke_slot() as slot:
        slot.detect_cap_hit('err', 'out')

    assert gate.detect_cap_hit.call_args.kwargs['scope'] is None
