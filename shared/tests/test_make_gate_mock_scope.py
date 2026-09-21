"""Contract tests for scope-forwarding fidelity in ``shared.testing.make_gate_mock``.

Pins that the ``invoke_slot()`` double forwards the ``scope`` supplied to
``invoke_slot(scope=...)`` into the same gate calls production
``UsageGate.invoke_slot`` / ``InvokeSlot.report`` forward it into —
``before_invoke`` and ``report()``'s CapHit/NearCap dispatch arms (task
4234). The sibling ``detect_cap_hit`` proxy call is a deliberate, documented
exception (see the comment in ``_slot_detect_cap_hit``,
``shared/src/shared/testing.py``) and is NOT pinned here — tracked as a
follow-up instead.

Without this pin, a scope-forwarding regression could silently pass through
the double: every existing ``make_gate_mock``-based suite only asserts
``assert_called_once()`` / ``assert_not_called()`` on the affected gate
methods, never exact ``call_args``, so neither gaining nor losing the
``scope`` kwarg would fail any of them.
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
    """Unscoped invocations must still forward ``scope=None`` explicitly.

    Production ``InvokeSlot.report`` always passes ``scope=self.scope``
    unconditionally, never omitting the kwarg — the double must match that
    shape rather than silently dropping the kwarg when it happens to be
    falsy.
    """
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
