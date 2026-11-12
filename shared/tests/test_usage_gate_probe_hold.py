"""A resumed account's probe claim is bounded by UsageCapConfig.probe_slot_hold_secs.

The claim (PROBING -> PROBE_IN_FLIGHT) lets one caller test a resumed account
before the pool piles onto it. Its verdict arrives only when the holder's
session exits, so an unbounded claim parks every other caller for the
holder's whole lifetime (task 9327). These tests drive public seams, with the
probe subprocess stubbed by ``make_gate``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from datetime import UTC, datetime, timedelta

import pydantic
import pytest
from _usage_gate_test_helpers import make_gate, ok_result, scripted_invoke

from shared.cli_invoke import invoke_with_cap_retry
from shared.config_models import UsageCapConfig
from shared.invocation_outcome import CapHit

_SERVED_WITHIN_SECS = 5.0
_STILL_BLOCKED_AFTER_SECS = 0.1


def _an_hour_from_now() -> datetime:
    return datetime.now(UTC) + timedelta(hours=1)


def _a_minute_ago() -> datetime:
    return datetime.now(UTC) - timedelta(minutes=1)


def _cap_hit(resets_at: datetime) -> CapHit:
    return CapHit(resets_at=resets_at, reason='usage limit reached')


async def _report_cap(gate, resets_at: datetime) -> None:
    async with gate.invoke_slot() as slot:
        slot.report(_cap_hit(resets_at))


async def _eventually(predicate) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout=_SERVED_WITHIN_SECS)


async def _cancel(task: asyncio.Task) -> None:
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def _until_holding(holder: asyncio.Task, holding: asyncio.Event) -> None:
    holding_seen = asyncio.ensure_future(holding.wait())
    try:
        await asyncio.wait({holding_seen, holder}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        await _cancel(holding_seen)
    assert holding.is_set(), f'the holder exited before it held the slot: {holder!r}'


async def test_a_long_running_probe_holder_does_not_starve_a_second_caller_after_resume():
    gate = make_gate(
        ['max-a', 'max-c'],
        wait_for_reset=True,
        probe_interval_secs=3600,
        probe_slot_hold_secs=0.2,
    )
    holding = asyncio.Event()
    may_exit = asyncio.Event()

    async def watcher_session(**_kwargs):
        holding.set()
        await may_exit.wait()
        return ok_result('watched')

    watcher = None
    try:
        await _report_cap(gate, _an_hour_from_now())
        await _report_cap(gate, _a_minute_ago())
        await _eventually(lambda: gate.capped_account_names == ('max-a',))

        watcher = asyncio.create_task(
            invoke_with_cap_retry(
                gate,
                'Escalation watcher (auto)',
                invoke_fn=watcher_session,
                prompt='watch',
                model='sonnet',
            )
        )
        await _until_holding(watcher, holding)

        implementer = await asyncio.wait_for(
            invoke_with_cap_retry(
                gate,
                'Task 464 [implementer]',
                invoke_fn=scripted_invoke(ok_result()),
                prompt='implement',
                model='opus',
            ),
            timeout=_SERVED_WITHIN_SECS,
        )

        assert implementer.success
        assert implementer.account_name == 'max-c'
        assert not watcher.done()

        may_exit.set()
        watched = await asyncio.wait_for(watcher, timeout=_SERVED_WITHIN_SECS)
        assert watched.success
    finally:
        may_exit.set()
        if watcher is not None and not watcher.done():
            await _cancel(watcher)
        await gate.shutdown()


async def test_a_verdict_reported_after_the_hold_expired_still_caps_the_account():
    gate = make_gate(['max-a'], probe_slot_hold_secs=0.05)
    try:
        await _report_cap(gate, _a_minute_ago())
        async with gate.invoke_slot() as holder:
            second = await asyncio.wait_for(
                gate.before_invoke(), timeout=_SERVED_WITHIN_SECS,
            )
            assert second is not None and second.name == 'max-a'

            holder.report(_cap_hit(_an_hour_from_now()))

        assert gate.capped_account_names == ('max-a',)
    finally:
        await gate.shutdown()


async def test_the_probe_slot_still_blocks_a_second_caller_inside_the_hold():
    gate = make_gate(['max-a'], probe_slot_hold_secs=60)
    second = None
    try:
        await _report_cap(gate, _a_minute_ago())
        async with gate.invoke_slot() as holder:
            second = asyncio.create_task(gate.before_invoke())
            await asyncio.sleep(_STILL_BLOCKED_AFTER_SECS)
            assert not second.done()

            holder.report(_cap_hit(_an_hour_from_now()))

        assert gate.capped_account_names == ('max-a',)
        await asyncio.sleep(_STILL_BLOCKED_AFTER_SECS)
        assert not second.done()
    finally:
        if second is not None:
            await _cancel(second)
        await gate.shutdown()


@contextlib.contextmanager
def _recorded_loop_errors():
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()
    errors: list[dict] = []
    loop.set_exception_handler(lambda _loop, context: errors.append(context))
    try:
        yield errors
    finally:
        loop.set_exception_handler(previous)


async def test_a_confirm_inside_the_hold_is_never_overridden_by_it():
    hold_secs = 0.05
    gate = make_gate(['max-a'], probe_slot_hold_secs=hold_secs)
    try:
        with _recorded_loop_errors() as loop_errors:
            await _report_cap(gate, _a_minute_ago())
            async with gate.invoke_slot() as holder:
                holder.confirm()
            lease = await asyncio.wait_for(
                gate.before_invoke(), timeout=_SERVED_WITHIN_SECS,
            )
            assert lease is not None

            await asyncio.sleep(hold_secs * 3)

            assert gate.lease_is_current(lease)
        assert loop_errors == []
    finally:
        await gate.shutdown()


async def test_a_cap_verdict_inside_the_hold_is_never_overridden_by_it():
    hold_secs = 0.05
    gate = make_gate(['max-a'], probe_slot_hold_secs=hold_secs)
    try:
        with _recorded_loop_errors() as loop_errors:
            await _report_cap(gate, _a_minute_ago())
            async with gate.invoke_slot() as holder:
                holder.report(_cap_hit(_an_hour_from_now()))

            await asyncio.sleep(hold_secs * 3)

            assert gate.capped_account_names == ('max-a',)
        assert loop_errors == []
    finally:
        await gate.shutdown()


def _park_payloads(caplog) -> list[dict]:
    payloads = []
    for record in caplog.records:
        try:
            payload = json.loads(record.getMessage())
        except ValueError:
            continue
        if isinstance(payload, dict) and payload.get('event') == 'all_capped_park':
            payloads.append(payload)
    return payloads


async def test_a_park_behind_a_held_claim_names_the_account_awaiting_its_verdict(caplog):
    caplog.set_level(logging.WARNING, logger='shared.usage_gate')
    gate = make_gate(['max-a'], probe_slot_hold_secs=60)
    parked = None
    try:
        await _report_cap(gate, _a_minute_ago())
        async with gate.invoke_slot():
            parked = asyncio.create_task(gate.before_invoke())
            await _eventually(lambda: _park_payloads(caplog))

        payload = _park_payloads(caplog)[0]
        assert payload['probe_in_flight_accounts'] == ['max-a'], payload
        assert payload['capped_accounts'] == [], payload
    finally:
        if parked is not None:
            await _cancel(parked)
        await gate.shutdown()


async def test_shutdown_disarms_a_pending_hold():
    hold_secs = 0.05
    gate = make_gate(['max-a'], probe_slot_hold_secs=hold_secs)
    try:
        await _report_cap(gate, _a_minute_ago())
        async with gate.invoke_slot():
            await gate.shutdown()
            await asyncio.sleep(hold_secs * 3)

            assert gate.try_lease() is None
    finally:
        await gate.shutdown()


@pytest.mark.parametrize('hold_secs', [0, -1.0])
def test_a_non_positive_hold_is_rejected(hold_secs):
    with pytest.raises(pydantic.ValidationError):
        UsageCapConfig(probe_slot_hold_secs=hold_secs)
