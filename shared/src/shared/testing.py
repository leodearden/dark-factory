"""Test helpers for :mod:`shared.usage_gate`.

``make_gate_mock()`` is the canonical way to build a mock ``UsageGate`` for
tests. Use it anywhere a real gate would be overkill.

**Why a helper and not bare ``MagicMock(spec=UsageGate)``.**  Since
commit 065e95a4c9, ``shared.cli_invoke.invoke_with_cap_retry`` routes every
invocation through ``async with usage_gate.invoke_slot() as slot:`` and then
calls ``slot.detect_cap_hit(...)``, ``slot.confirm(...)``, ``slot.settle()``.
A bare ``MagicMock()`` (or even ``MagicMock(spec=UsageGate)``) leaves
``slot.detect_cap_hit`` auto-wired to a child ``MagicMock`` that returns a
truthy ``MagicMock`` on every call. The retry loop therefore believes every
invocation is a cap hit and runs ~19 real ``asyncio.sleep`` calls with
exponential backoff (5→10→20→40→80→160→300, capped) — about 45 minutes
before raising ``AllAccountsCappedException``. This helper wires the slot
correctly so ``detect_cap_hit`` delegates to the configurable
``gate.detect_cap_hit`` mock (default: returns ``False``).

``spec=UsageGate`` on the outer mock also catches attribute-name drift
(e.g. typoing ``gate.active_account`` for ``gate.active_account_name``).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from shared.usage_gate import InvokeSlot, UsageGate

__all__ = ['make_gate_mock']


def make_gate_mock(**overrides) -> MagicMock:
    """Return a ``MagicMock(spec=UsageGate)`` wired for ``invoke_slot()``.

    Defaults:
      - ``account_count = 1``
      - ``before_invoke`` returns ``'tok'``
      - ``detect_cap_hit`` returns ``False``
      - ``active_account_name = 'acct'``
      - ``on_agent_complete``, ``confirm_account_ok``, ``release_probe_slot``
        are plain ``MagicMock`` instances (assertable).

    Pass any of these (or other ``UsageGate`` attrs) as keyword arguments to
    override. ``before_invoke`` is typically ``AsyncMock(side_effect=[...])``
    when a test needs token rotation across retries.

    The returned mock's ``invoke_slot()`` is a callable that produces a fresh
    async-context-manager per call. ``__aenter__`` yields a slot whose
    ``detect_cap_hit``, ``confirm``, ``settle`` methods proxy back to the gate
    — so tests can still assert on ``gate.detect_cap_hit.call_args``,
    ``gate.confirm_account_ok.assert_called_with(...)``, etc. ``__aexit__``
    calls ``gate.release_probe_slot(slot.token)`` unless the slot was
    settled, matching the real ``UsageGate.invoke_slot`` behaviour.
    """
    gate = MagicMock(spec=UsageGate)
    gate.account_count = overrides.pop('account_count', 1)
    gate.before_invoke = overrides.pop('before_invoke', AsyncMock(return_value='tok'))
    gate.detect_cap_hit = overrides.pop('detect_cap_hit', MagicMock(return_value=False))
    gate.active_account_name = overrides.pop('active_account_name', 'acct')
    gate.on_agent_complete = overrides.pop('on_agent_complete', MagicMock())
    gate.confirm_account_ok = overrides.pop('confirm_account_ok', MagicMock())
    gate.release_probe_slot = overrides.pop('release_probe_slot', MagicMock())
    for k, v in overrides.items():
        setattr(gate, k, v)

    def _make_invoke_slot_cm():
        holder: dict = {'slot': None}

        async def _aenter_impl(*_args, **_kw):
            token = await gate.before_invoke()
            slot = MagicMock(spec=InvokeSlot)
            slot.token = token
            # Mirror InvokeSlot.__init__: None → '' so tests can rely on
            # the same coercion as production code.
            slot.account_name = gate.active_account_name or ''
            slot._settled = False

            def _slot_detect_cap_hit(stderr, output, backend='claude'):
                hit = gate.detect_cap_hit(
                    stderr, output, backend, oauth_token=slot.token,
                )
                if hit:
                    slot._settled = True
                return hit

            def _slot_confirm(cost_usd=0.0):
                gate.confirm_account_ok(slot.token)
                gate.on_agent_complete(cost_usd)
                slot._settled = True

            def _slot_settle():
                slot._settled = True

            # Plain (sync) MagicMocks — InvokeSlot.detect_cap_hit/confirm/
            # settle are plain methods. Using AsyncMock here would break
            # prod's unawaited `if slot.detect_cap_hit(...):` call.
            slot.detect_cap_hit = MagicMock(side_effect=_slot_detect_cap_hit)
            slot.confirm = MagicMock(side_effect=_slot_confirm)
            slot.settle = MagicMock(side_effect=_slot_settle)
            holder['slot'] = slot
            return slot

        async def _aexit_impl(*_args, **_kw):
            slot = holder['slot']
            if slot is not None and not slot._settled:
                gate.release_probe_slot(slot.token)
            return None

        cm = MagicMock()
        cm.__aenter__ = AsyncMock(side_effect=_aenter_impl)
        cm.__aexit__ = AsyncMock(side_effect=_aexit_impl)
        return cm

    gate.invoke_slot = MagicMock(side_effect=_make_invoke_slot_cm)
    return gate
