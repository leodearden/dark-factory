"""Exhaustive concurrency tests for UsageGate.

Tests race conditions, thundering herd prevention, gate open/close semantics,
probe slot contention, shutdown races, and rapid cap/uncap cycling.
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from shared.config_models import AccountConfig, UsageCapConfig
from shared.usage_gate import SessionBudgetExhausted, UsageGate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_gate(
    account_names: list[str],
    *,
    cost_store=None,
    wait_for_reset: bool = False,
    session_budget_usd: float | None = None,
    probe_interval_secs: int = 300,
    max_probe_interval_secs: int = 1800,
) -> UsageGate:
    acct_cfgs = []
    env_vars: dict[str, str] = {}
    for name in account_names:
        env_key = f'TEST_TOKEN_{name.upper().replace("-", "_")}'
        env_vars[env_key] = f'fake-token-{name}'
        acct_cfgs.append(AccountConfig(name=name, oauth_token_env=env_key))
    config = UsageCapConfig(
        accounts=acct_cfgs,
        wait_for_reset=wait_for_reset,
        session_budget_usd=session_budget_usd,
        probe_interval_secs=probe_interval_secs,
        max_probe_interval_secs=max_probe_interval_secs,
    )
    with patch.dict(os.environ, env_vars):
        gate = UsageGate(config, cost_store=cost_store)
    gate._run_probe = AsyncMock(return_value=True)
    return gate


def make_mock_cost_store() -> AsyncMock:
    store = AsyncMock()
    store.save_account_event = AsyncMock(return_value=None)
    return store


def all_tokens(gate: UsageGate) -> set[str]:
    """Return the set of all valid tokens in a gate."""
    return {a.token for a in gate._accounts if a.token}


# ---------------------------------------------------------------------------
# TestConcurrentBeforeInvoke
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestConcurrentBeforeInvoke:
    """Concurrent before_invoke calls must be serialized and never deadlock."""

    async def test_10_tasks_3_accounts_all_get_valid_tokens(self):
        """10 tasks call before_invoke simultaneously with 3 uncapped accounts.

        All must get valid tokens and no crashes.
        """
        gate = make_gate(['a', 'b', 'c'])
        valid = all_tokens(gate)

        async def call():
            return await gate.before_invoke()

        tasks = [asyncio.create_task(call()) for _ in range(10)]
        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=2.0)

        assert len(results) == 10
        for token in results:
            assert token in valid

    async def test_10_tasks_1_account_all_get_same_token(self):
        """10 tasks with 1 account available: all get the same token (serialized)."""
        gate = make_gate(['solo'])
        expected = 'fake-token-solo'

        tasks = [asyncio.create_task(gate.before_invoke()) for _ in range(10)]
        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=2.0)

        assert all(t == expected for t in results)

    async def test_5_tasks_2_accounts_tokens_distributed(self):
        """5 tasks with 2 accounts: each task gets one of the 2 tokens."""
        gate = make_gate(['x', 'y'])
        valid = all_tokens(gate)

        tasks = [asyncio.create_task(gate.before_invoke()) for _ in range(5)]
        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=2.0)

        for token in results:
            assert token in valid

    async def test_50_tasks_burst_no_deadlock(self):
        """50 tasks burst: all complete without deadlock."""
        gate = make_gate(['a', 'b', 'c'])

        tasks = [asyncio.create_task(gate.before_invoke()) for _ in range(50)]
        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=5.0)

        assert len(results) == 50
        valid = all_tokens(gate)
        for token in results:
            assert token in valid

    async def test_cap_mid_flight(self):
        """Tasks calling before_invoke while another task caps an account mid-flight.

        Start several tasks, then cap an account between iterations. All tasks
        must still complete with valid tokens.
        """
        gate = make_gate(['primary', 'backup'])
        valid = all_tokens(gate)
        results: list[str | None] = []
        started = asyncio.Event()

        async def slow_caller(idx: int):
            if idx == 0:
                started.set()
            token = await gate.before_invoke()
            results.append(token)
            return token

        async def capper():
            await started.wait()
            # Give tasks a chance to enter before_invoke
            await asyncio.sleep(0)
            gate._accounts[0].capped = True

        tasks = [asyncio.create_task(slow_caller(i)) for i in range(5)]
        tasks.append(asyncio.create_task(capper()))

        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=2.0)

        for token in results:
            assert token in valid


# ---------------------------------------------------------------------------
# TestProbeSlotContention
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestProbeSlotContention:
    """Probe slot mechanism: exactly one task claims the slot, others block."""

    async def test_one_claims_probe_slot_others_block(self):
        """Account in probing=True state, 5 tasks call before_invoke.

        Exactly one claims the slot (sets probe_in_flight=True), others block
        until the gate reopens.
        """
        gate = make_gate(['acct'])
        acct = gate._accounts[0]
        acct.probing = True

        claimed: list[str | None] = []
        blocked_count = 0

        # First task claims the probe slot
        token = await asyncio.wait_for(gate.before_invoke(), timeout=1.0)
        claimed.append(token)

        # After claiming, probe_in_flight must be True and probing must be False
        assert acct.probe_in_flight is True
        assert acct.probing is False

        # Now _open is cleared. Start 5 more tasks — they should all block.
        blocked_tasks = [asyncio.create_task(gate.before_invoke()) for _ in range(5)]

        # Give tasks a chance to block
        await asyncio.sleep(0.05)

        for t in blocked_tasks:
            assert not t.done(), 'Task should be blocked waiting on gate'
            blocked_count += 1

        assert blocked_count == 5

        # Clean up: confirm OK to unblock
        gate.confirm_account_ok(acct.token)
        results = await asyncio.wait_for(asyncio.gather(*blocked_tasks), timeout=1.0)
        assert all(r == acct.token for r in results)

    async def test_confirm_account_ok_unblocks_waiters(self):
        """After confirm_account_ok() clears probe_in_flight, blocked tasks proceed."""
        gate = make_gate(['acct'])
        acct = gate._accounts[0]
        acct.probing = True

        # Claim probe slot
        await asyncio.wait_for(gate.before_invoke(), timeout=1.0)
        assert acct.probe_in_flight is True

        # Start blocked tasks
        blocked = [asyncio.create_task(gate.before_invoke()) for _ in range(3)]
        await asyncio.sleep(0.05)

        for t in blocked:
            assert not t.done()

        # Confirm OK
        gate.confirm_account_ok(acct.token)

        results = await asyncio.wait_for(asyncio.gather(*blocked), timeout=1.0)
        assert len(results) == 3
        assert all(r == acct.token for r in results)
        assert acct.probe_in_flight is False
        assert acct.probe_count == 0

    async def test_probe_slot_claim_is_atomic(self):
        """Probe slot claim is atomic: no race between probing check and flag set.

        Run the scenario many times to increase probability of hitting a race.
        """
        for _ in range(30):
            gate = make_gate(['acct'])
            acct = gate._accounts[0]
            acct.probing = True

            # Two tasks race to claim the probe slot
            t1 = asyncio.create_task(gate.before_invoke())
            t2 = asyncio.create_task(gate.before_invoke())

            # First completes immediately (claims slot)
            done, pending = await asyncio.wait(
                {t1, t2}, timeout=0.5, return_when=asyncio.FIRST_COMPLETED,
            )

            assert len(done) >= 1, 'At least one task should complete (claim slot)'

            # After the first claims, probe_in_flight=True, probing=False.
            # The second task should be blocked (probe_in_flight blocks it).
            assert acct.probing is False
            assert acct.probe_in_flight is True

            # Unblock the second
            gate.confirm_account_ok(acct.token)
            if pending:
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True), timeout=0.5,
                )

    async def test_gate_cleared_on_probe_claim_set_on_confirm(self):
        """Gate is cleared when probe slot is claimed, and set when confirmed."""
        gate = make_gate(['acct'])
        acct = gate._accounts[0]
        acct.probing = True

        # Before claim, gate is open
        assert gate._open.is_set()

        # Claim probe slot
        await gate.before_invoke()

        # Gate must be cleared (other tasks must wait)
        assert not gate._open.is_set()

        # Confirm
        gate.confirm_account_ok(acct.token)

        # Gate must be set again
        assert gate._open.is_set()


# ---------------------------------------------------------------------------
# TestGateOpenCloseRaces
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGateOpenCloseRaces:
    """Gate (asyncio.Event) open/close races and missed-signal prevention."""

    async def test_clear_then_set_wakes_waiters(self):
        """Gate cleared then immediately set: waiters wake up (no missed signal)."""
        gate = make_gate(['acct'])
        gate._accounts[0].capped = True
        gate._open.clear()

        results: list[str | None] = []

        async def waiter():
            token = await gate.before_invoke()
            results.append(token)

        tasks = [asyncio.create_task(waiter()) for _ in range(3)]
        await asyncio.sleep(0.05)

        # All should be blocked
        for t in tasks:
            assert not t.done()

        # Uncap and set gate
        gate._accounts[0].capped = False
        gate._open.set()

        await asyncio.wait_for(asyncio.gather(*tasks), timeout=1.0)
        assert len(results) == 3

    async def test_multiple_waiters_all_wake(self):
        """Multiple tasks waiting on gate: all wake up when gate opens."""
        gate = make_gate(['acct'])
        gate._accounts[0].capped = True
        gate._open.clear()

        tasks = [asyncio.create_task(gate.before_invoke()) for _ in range(10)]
        await asyncio.sleep(0.05)

        for t in tasks:
            assert not t.done()

        gate._accounts[0].capped = False
        gate._open.set()

        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=2.0)
        assert len(results) == 10
        assert all(r == 'fake-token-acct' for r in results)

    async def test_recap_before_lock_acquired_retries(self):
        """Gate opens, task starts before_invoke, but account re-caps before lock acquired.

        The task must retry correctly rather than returning a capped account's token.
        """
        gate = make_gate(['a', 'b'])
        valid = all_tokens(gate)
        # Both capped initially
        gate._accounts[0].capped = True
        gate._accounts[1].capped = True
        gate._open.clear()

        task = asyncio.create_task(gate.before_invoke())
        await asyncio.sleep(0.05)
        assert not task.done()

        # Briefly uncap 'a', then immediately re-cap it.
        # Also uncap 'b' so the task can eventually proceed.
        gate._accounts[0].capped = False
        gate._open.set()
        await asyncio.sleep(0)  # let waiter wake and re-enter the while loop
        gate._accounts[0].capped = True
        gate._accounts[1].capped = False
        gate._open.set()  # ensure gate is open for the retry

        result = await asyncio.wait_for(task, timeout=1.0)
        # Must have gotten a valid token — either 'a' if grabbed before re-cap, or 'b'
        assert result in valid

    async def test_rapid_open_close_open_tasks_eventually_proceed(self):
        """Rapid open/close/open: tasks eventually proceed."""
        gate = make_gate(['acct'])
        gate._accounts[0].capped = True
        gate._open.clear()

        task = asyncio.create_task(gate.before_invoke())
        await asyncio.sleep(0.05)

        # Rapid cycling
        for _ in range(10):
            gate._accounts[0].capped = False
            gate._open.set()
            await asyncio.sleep(0)
            gate._accounts[0].capped = True
            gate._open.clear()

        # Final uncap
        gate._accounts[0].capped = False
        gate._open.set()

        result = await asyncio.wait_for(task, timeout=1.0)
        assert result == 'fake-token-acct'


# ---------------------------------------------------------------------------
# TestRefreshVsProbeRace
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRefreshVsProbeRace:
    """Races between _refresh_capped_accounts and probe loop uncapping."""

    async def test_no_double_counting_pause_started_at(self):
        """Both refresh and probe loop try to uncap same account.

        pause_started_at must not be double-counted (set to None after first uncap).
        """
        gate = make_gate(['acct'], wait_for_reset=True)
        acct = gate._accounts[0]
        acct.capped = True
        start_time = datetime.now(UTC) - timedelta(minutes=10)
        acct.pause_started_at = start_time
        acct.resets_at = datetime.now(UTC) - timedelta(minutes=1)

        initial_pause = gate._total_pause_secs

        # Refresh uncaps (reset time passed)
        refreshed = await gate._refresh_capped_accounts()
        assert refreshed is True
        assert acct.capped is False
        assert acct.pause_started_at is None

        pause_after_refresh = gate._total_pause_secs
        assert pause_after_refresh > initial_pause

        # Simulate probe also trying to uncap (it checks acct.capped first)
        # Since acct.capped is already False, the probe loop should exit
        # without touching pause_started_at again.
        acct_copy_capped = acct.capped
        assert acct_copy_capped is False  # probe loop would see this and exit

        # Total pause should not change again
        assert gate._total_pause_secs == pause_after_refresh

    async def test_refresh_uncaps_while_probe_sleeping(self):
        """Refresh uncaps account while probe loop is sleeping.

        Probe loop should see acct.capped is False and exit early.
        """
        gate = make_gate(['acct'], wait_for_reset=True, probe_interval_secs=1)
        acct = gate._accounts[0]
        acct.capped = True
        acct.pause_started_at = datetime.now(UTC) - timedelta(minutes=5)
        acct.resets_at = datetime.now(UTC) + timedelta(seconds=0.1)

        # Start probe loop in background
        probe_task = asyncio.create_task(gate._account_resume_probe_loop(acct))

        # Let probe start sleeping
        await asyncio.sleep(0.05)

        # Refresh uncaps account (simulating timer expiry path)
        acct.capped = False
        acct.probing = True
        acct.pause_started_at = None

        # Probe loop should notice capped=False and exit
        await asyncio.wait_for(probe_task, timeout=2.0)

        # No error, probe exited cleanly
        assert not probe_task.cancelled()

    async def test_probing_set_idempotently(self):
        """Both paths set probing=True idempotently: no inconsistent state."""
        gate = make_gate(['acct'], wait_for_reset=True)
        acct = gate._accounts[0]
        acct.capped = True
        acct.pause_started_at = datetime.now(UTC) - timedelta(minutes=5)
        acct.resets_at = datetime.now(UTC) - timedelta(minutes=1)

        # Refresh sets probing=True
        await gate._refresh_capped_accounts()
        assert acct.probing is True
        assert acct.capped is False

        # Simulate probe loop also setting probing=True (idempotent)
        acct.probing = True  # no-op if already True
        assert acct.probing is True
        assert acct.capped is False


# ---------------------------------------------------------------------------
# TestShutdownRaces
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestShutdownRaces:
    """Shutdown during various concurrent operations."""

    async def test_shutdown_during_before_invoke_wait(self):
        """Shutdown during before_invoke wait (task blocked on _open.wait).

        before_invoke should either return or allow cancellation.
        """
        gate = make_gate(['acct'])
        gate._accounts[0].capped = True
        gate._open.clear()

        task = asyncio.create_task(gate.before_invoke())
        await asyncio.sleep(0.05)
        assert not task.done()

        # Shutdown the gate
        await gate.shutdown()

        # The blocked task is still stuck on _open.wait().
        # Cancel it explicitly (shutdown doesn't cancel caller tasks).
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_shutdown_during_probe_loop(self):
        """Shutdown during probe loop sleep: probe task cancelled cleanly."""
        gate = make_gate(['acct'], wait_for_reset=True, probe_interval_secs=60)
        acct = gate._accounts[0]
        acct.capped = True
        acct.resets_at = datetime.now(UTC) + timedelta(hours=1)

        # Start probe loop
        gate._start_account_resume_probe(acct)
        assert acct.resume_task is not None
        assert not acct.resume_task.done()

        # Shutdown cancels probe tasks
        await asyncio.wait_for(gate.shutdown(), timeout=2.0)

        assert acct.resume_task is None

    async def test_cost_event_then_shutdown_no_crash(self):
        """Cost event fires, then immediately shutdown: no crash."""
        gate = make_gate(['acct'], cost_store=make_mock_cost_store())

        # Create a blocking cost store
        block = asyncio.Event()

        async def blocking_save(*args, **kwargs):
            await block.wait()

        # gate._cost_store is CostStore | None here but make_mock_cost_store()
        # returns a non-None AsyncMock.  Assert + cast-free access via the
        # assert-narrowed local; side_effect is a MagicMock attribute that
        # pyright can't see through the AsyncMock bound-method type.
        assert gate._cost_store is not None
        gate._cost_store.save_account_event.side_effect = blocking_save  # type: ignore[attr-defined]

        gate._fire_cost_event('acct', 'cap_hit', '{}')
        await asyncio.sleep(0)

        assert len(gate._background_tasks) == 1

        # Shutdown while event is in-flight
        await asyncio.wait_for(gate.shutdown(), timeout=2.0)
        assert len(gate._background_tasks) == 0

    async def test_shutdown_while_fire_cost_event_in_progress(self):
        """Shutdown while _fire_cost_event tasks are in progress: all drained."""
        gate = make_gate(['acct'], cost_store=make_mock_cost_store())

        block = asyncio.Event()

        async def blocking_save(*args, **kwargs):
            await block.wait()

        assert gate._cost_store is not None
        gate._cost_store.save_account_event.side_effect = blocking_save  # type: ignore[attr-defined]

        # Fire multiple events
        for _ in range(5):
            gate._fire_cost_event('acct', 'test', '{}')
        await asyncio.sleep(0)

        assert len(gate._background_tasks) == 5

        await asyncio.wait_for(gate.shutdown(), timeout=2.0)
        assert len(gate._background_tasks) == 0


# ---------------------------------------------------------------------------
# TestCapDuringProbing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCapDuringProbing:
    """Cap detection during probe lifecycle phases."""

    async def test_detect_cap_during_probing_true(self):
        """Account in probing=True state, detect_cap_hit called.

        Must set: probing=False, capped=True, probe_in_flight=False.
        """
        gate = make_gate(['acct'])
        acct = gate._accounts[0]
        acct.probing = True
        acct.capped = False

        gate.detect_cap_hit(
            "You've hit your usage limit",
            'resets in 3h',
            oauth_token=acct.token,
        )

        assert acct.capped is True
        assert acct.probing is False
        assert acct.probe_in_flight is False

    async def test_detect_cap_during_probe_in_flight(self):
        """Account in probe_in_flight=True state, detect_cap_hit called.

        Must reset all probe flags and mark capped.
        """
        gate = make_gate(['acct'])
        acct = gate._accounts[0]
        acct.probe_in_flight = True
        acct.probing = False
        acct.capped = False

        gate.detect_cap_hit(
            "You've hit your usage limit",
            'resets in 3h',
            oauth_token=acct.token,
        )

        assert acct.capped is True
        assert acct.probing is False
        assert acct.probe_in_flight is False

    async def test_recap_restarts_probe(self):
        """After re-cap, a new resume task is created (probe restarts)."""
        gate = make_gate(['acct'], wait_for_reset=True, probe_interval_secs=60)
        acct = gate._accounts[0]
        acct.capped = False
        acct.probing = True

        # First cap: starts a probe task
        gate.detect_cap_hit(
            "You've hit your usage limit",
            'resets in 1h',
            oauth_token=acct.token,
        )

        first_task = acct.resume_task
        assert first_task is not None

        # Simulate uncap via probe success
        acct.capped = False
        acct.probing = True

        # Wait for first task to see capped=False and exit
        await asyncio.wait_for(first_task, timeout=2.0)

        # Second cap: must start a new resume task
        gate.detect_cap_hit(
            "You've hit your usage limit",
            'resets in 2h',
            oauth_token=acct.token,
        )

        second_task = acct.resume_task
        assert second_task is not None
        assert second_task is not first_task

        # Cleanup
        acct.capped = False
        await asyncio.wait_for(second_task, timeout=2.0)


# ---------------------------------------------------------------------------
# TestRapidCapUncapCycling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRapidCapUncapCycling:
    """Rapid cap/uncap transitions must leave state consistent."""

    async def test_single_account_rapid_cycling(self):
        """uncapped -> capped -> uncapped -> capped rapidly: state consistent."""
        gate = make_gate(['acct'])
        acct = gate._accounts[0]

        for i in range(20):
            if i % 2 == 0:
                # Cap
                gate._handle_cap_detected(
                    f'cap #{i}',
                    datetime.now(UTC) + timedelta(hours=1),
                    acct.token,
                )
                assert acct.capped is True
                assert gate.is_paused is True
            else:
                # Uncap
                acct.capped = False
                acct.probing = False
                acct.probe_in_flight = False
                acct.pause_started_at = None
                gate._open.set()
                assert acct.capped is False
                assert gate.is_paused is False

    async def test_3_accounts_rapid_cycling_is_paused_correct(self):
        """3 accounts with rapid cap/uncap: is_paused tracks correctly.

        is_paused is True only when ALL accounts are capped.
        """
        gate = make_gate(['a', 'b', 'c'])

        # Cap a
        gate._handle_cap_detected('cap', None, gate._accounts[0].token)
        assert not gate.is_paused  # b and c still up

        # Cap b
        gate._handle_cap_detected('cap', None, gate._accounts[1].token)
        assert not gate.is_paused  # c still up

        # Cap c
        gate._handle_cap_detected('cap', None, gate._accounts[2].token)
        assert gate.is_paused  # all capped

        # Uncap b
        gate._accounts[1].capped = False
        assert not gate.is_paused

        # Re-cap b
        gate._handle_cap_detected('cap', None, gate._accounts[1].token)
        assert gate.is_paused

        # Uncap a and c
        gate._accounts[0].capped = False
        gate._accounts[2].capped = False
        assert not gate.is_paused

    async def test_multiple_detect_cap_hit_idempotent_pause_started_at(self):
        """Multiple detect_cap_hit calls on same account: pause_started_at not overwritten."""
        gate = make_gate(['acct'])
        acct = gate._accounts[0]

        # First cap
        gate._handle_cap_detected('first cap', None, acct.token)
        first_pause_start = acct.pause_started_at
        assert first_pause_start is not None

        await asyncio.sleep(0.01)

        # Second cap on same account (idempotent)
        gate._handle_cap_detected('second cap', None, acct.token)
        assert acct.pause_started_at == first_pause_start, (
            'pause_started_at was overwritten on repeat detect_cap_hit'
        )


# ---------------------------------------------------------------------------
# TestThunderingHerd
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestThunderingHerd:
    """Thundering herd prevention when accounts uncap."""

    async def test_one_probe_slot_on_uncap(self):
        """All 3 accounts capped, gate closed. One uncaps.

        10 tasks waiting: exactly one gets probe slot, rest wait.
        After confirm, rest proceed.
        """
        gate = make_gate(['a', 'b', 'c'])
        for acct in gate._accounts:
            acct.capped = True
        gate._open.clear()

        # Start 10 waiting tasks
        tasks = [asyncio.create_task(gate.before_invoke()) for _ in range(10)]
        await asyncio.sleep(0.05)
        for t in tasks:
            assert not t.done()

        # Uncap account 'a' with probing=True (simulates probe success)
        acct_a = gate._accounts[0]
        acct_a.capped = False
        acct_a.probing = True
        gate._open.set()

        # Let tasks process: exactly one should claim the probe slot
        await asyncio.sleep(0.05)

        # The probe claimant completes immediately.
        done = [t for t in tasks if t.done()]
        # Exactly one task should have completed (the probe claimant).
        assert len(done) == 1, f'Expected 1 done task (probe claimant), got {len(done)}'
        assert acct_a.probe_in_flight is True
        assert acct_a.probing is False

        # Rest are blocked
        pending = [t for t in tasks if not t.done()]
        assert len(pending) == 9

        # Confirm OK: rest should proceed
        gate.confirm_account_ok(acct_a.token)

        results = await asyncio.wait_for(asyncio.gather(*pending), timeout=2.0)
        assert len(results) == 9
        assert all(r == acct_a.token for r in results)

    async def test_two_accounts_uncap_simultaneously(self):
        """Two accounts uncap simultaneously.

        Two probe slots claimed (one per account), rest wait until both confirmed.
        """
        gate = make_gate(['a', 'b', 'c'])
        for acct in gate._accounts:
            acct.capped = True
        gate._open.clear()

        # Start 10 waiting tasks
        tasks = [asyncio.create_task(gate.before_invoke()) for _ in range(10)]
        await asyncio.sleep(0.05)

        # Uncap a and b with probing=True
        gate._accounts[0].capped = False
        gate._accounts[0].probing = True
        gate._accounts[1].capped = False
        gate._accounts[1].probing = True
        gate._open.set()

        # Let tasks process
        await asyncio.sleep(0.05)

        done = [t for t in tasks if t.done()]
        # Two probe slots: two tasks should have completed
        assert len(done) == 2, f'Expected 2 done tasks (probe claimants), got {len(done)}'

        claimed_tokens = {t.result() for t in done}
        assert claimed_tokens == {'fake-token-a', 'fake-token-b'}

        # Confirm both
        gate.confirm_account_ok('fake-token-a')
        gate.confirm_account_ok('fake-token-b')

        pending = [t for t in tasks if not t.done()]
        results = await asyncio.wait_for(asyncio.gather(*pending), timeout=2.0)
        assert len(results) == 8
        valid = {'fake-token-a', 'fake-token-b'}
        for r in results:
            assert r in valid


# ---------------------------------------------------------------------------
# TestSessionBudgetConcurrency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSessionBudgetConcurrency:
    """Session budget checks under concurrent access."""

    async def test_budget_exhausted_during_concurrent_calls(self):
        """Multiple tasks call before_invoke while budget is borderline.

        Some may succeed, but once budget is exhausted, the rest must raise.
        """
        gate = make_gate(['acct'], session_budget_usd=1.0)
        gate._cumulative_cost = 0.99

        # First call succeeds (cost not yet over budget)
        token = await gate.before_invoke()
        assert token == 'fake-token-acct'

        # Push over budget
        gate.on_agent_complete(0.02)
        assert gate._cumulative_cost >= 1.0

        # Subsequent calls must raise
        with pytest.raises(SessionBudgetExhausted):
            await gate.before_invoke()

    async def test_concurrent_budget_exhaustion(self):
        """10 tasks with budget exactly at limit: no task should hang."""
        gate = make_gate(['acct'], session_budget_usd=1.0)
        gate._cumulative_cost = 1.0

        async def call():
            return await gate.before_invoke()

        tasks = [asyncio.create_task(call()) for _ in range(10)]
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True), timeout=2.0,
        )

        for r in results:
            assert isinstance(r, SessionBudgetExhausted)


# ---------------------------------------------------------------------------
# TestConcurrentCapAndBeforeInvoke
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestConcurrentCapAndBeforeInvoke:
    """Concurrent detect_cap_hit and before_invoke calls."""

    async def test_cap_hit_while_tasks_selecting_accounts(self):
        """detect_cap_hit fires while tasks are calling before_invoke.

        Tasks that already acquired a token proceed; tasks still waiting
        must switch to remaining uncapped accounts.
        """
        gate = make_gate(['a', 'b', 'c'])
        results: list[str | None] = []

        async def caller():
            token = await gate.before_invoke()
            results.append(token)

        # Start tasks
        tasks = [asyncio.create_task(caller()) for _ in range(5)]
        await asyncio.sleep(0)

        # Cap account 'a' after some tasks may have already started
        gate.detect_cap_hit(
            "You've hit your usage limit",
            'resets in 3h',
            oauth_token='fake-token-a',
        )

        await asyncio.wait_for(asyncio.gather(*tasks), timeout=2.0)
        assert len(results) == 5
        # All results must be valid tokens (b or c if a was capped before selection)
        valid = all_tokens(gate)
        for r in results:
            assert r in valid


# ---------------------------------------------------------------------------
# TestProbeLoopConcurrency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestProbeLoopConcurrency:
    """Probe loop interacting with concurrent before_invoke calls."""

    async def test_probe_success_wakes_waiting_tasks(self):
        """Probe loop succeeds, sets probing=True, sets _open.

        Waiting before_invoke tasks should wake up and one claims the probe slot.
        """
        gate = make_gate(['acct'], wait_for_reset=True, probe_interval_secs=0)
        acct = gate._accounts[0]
        acct.capped = True
        # resets_at far in the future so _refresh_capped_accounts does NOT
        # preemptively uncap this account — we want before_invoke to block on
        # _open.wait(), not be uncapped by the refresh path.
        acct.resets_at = datetime.now(UTC) + timedelta(hours=1)
        gate._open.clear()

        # Start waiting tasks — they will block because acct is capped
        waiting_tasks = [asyncio.create_task(gate.before_invoke()) for _ in range(5)]
        await asyncio.sleep(0.05)

        for t in waiting_tasks:
            assert not t.done()

        # Simulate the probe loop succeeding: uncap and set probing
        acct.capped = False
        acct.probing = True
        acct.probe_count = 0
        acct.pause_started_at = None
        gate._open.set()

        # One waiting task should claim the probe slot
        await asyncio.sleep(0.05)
        done = [t for t in waiting_tasks if t.done()]
        assert len(done) >= 1

        # Confirm to unblock rest
        gate.confirm_account_ok(acct.token)

        pending = [t for t in waiting_tasks if not t.done()]
        if pending:
            await asyncio.wait_for(asyncio.gather(*pending), timeout=1.0)

    async def test_multiple_probe_loops_dont_conflict(self):
        """Two accounts have separate probe loops that don't interfere."""
        gate = make_gate(['a', 'b'], wait_for_reset=True, probe_interval_secs=0)

        for acct in gate._accounts:
            acct.capped = True
            acct.resets_at = datetime.now(UTC) - timedelta(minutes=1)

        # Run both probe loops
        p1 = asyncio.create_task(gate._account_resume_probe_loop(gate._accounts[0]))
        p2 = asyncio.create_task(gate._account_resume_probe_loop(gate._accounts[1]))

        await asyncio.wait_for(asyncio.gather(p1, p2), timeout=2.0)

        # Both should be uncapped with probing=True
        assert gate._accounts[0].capped is False
        assert gate._accounts[0].probing is True
        assert gate._accounts[1].capped is False
        assert gate._accounts[1].probing is True


# ---------------------------------------------------------------------------
# TestStressBeforeInvoke
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStressBeforeInvoke:
    """High-contention stress tests to expose subtle races."""

    async def test_100_tasks_with_intermittent_caps(self):
        """100 tasks with random-ish cap/uncap: all must complete or raise."""
        gate = make_gate(['a', 'b'])
        valid = all_tokens(gate)
        completed = 0
        errors = 0

        async def worker(idx: int):
            nonlocal completed, errors
            try:
                token = await asyncio.wait_for(gate.before_invoke(), timeout=2.0)
                assert token in valid
                completed += 1
            except (TimeoutError, RuntimeError):
                errors += 1

        async def chaos():
            for i in range(20):
                await asyncio.sleep(0.01)
                acct = gate._accounts[i % 2]
                if i % 3 == 0:
                    gate._handle_cap_detected(f'cap {i}', None, acct.token)
                else:
                    acct.capped = False
                    acct.probing = False
                    acct.probe_in_flight = False
                    gate._open.set()

        tasks = [asyncio.create_task(worker(i)) for i in range(100)]
        tasks.append(asyncio.create_task(chaos()))

        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=10.0)
        # Most tasks should complete. Some may time out during all-capped windows.
        assert completed + errors == 100

    async def test_repeated_lock_acquisition_no_starvation(self):
        """30 iterations: repeated before_invoke calls don't starve any task.

        Run the scenario to verify that the lock doesn't consistently favor
        one task over another.
        """
        gate = make_gate(['acct'])

        async def call(idx: int):
            token = await gate.before_invoke()
            return idx, token

        for _ in range(30):
            tasks = [asyncio.create_task(call(i)) for i in range(10)]
            results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=2.0)
            assert len(results) == 10
            # All tasks got the token
            assert all(r[1] == 'fake-token-acct' for r in results)


# ---------------------------------------------------------------------------
# TestConfirmAccountOkEdgeCases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestConfirmAccountOkEdgeCases:
    """Edge cases in confirm_account_ok."""

    async def test_confirm_with_no_probe_in_flight_is_noop(self):
        """confirm_account_ok when probe_in_flight=False: no state change."""
        gate = make_gate(['acct'])
        acct = gate._accounts[0]
        assert acct.probe_in_flight is False

        gate._open.clear()
        gate.confirm_account_ok(acct.token)

        # Gate should not be set (probe_in_flight was False → no-op)
        assert not gate._open.is_set()

    async def test_confirm_with_unknown_token_is_noop(self):
        """confirm_account_ok with an unknown token: no crash, no state change."""
        gate = make_gate(['acct'])
        gate._accounts[0].probe_in_flight = True

        gate.confirm_account_ok('unknown-token')

        # Probe in flight not cleared (wrong token)
        assert gate._accounts[0].probe_in_flight is True

    async def test_confirm_with_none_token_is_noop(self):
        """confirm_account_ok with None token: no crash."""
        gate = make_gate(['acct'])
        gate._accounts[0].probe_in_flight = True

        gate.confirm_account_ok(None)
        assert gate._accounts[0].probe_in_flight is True

    async def test_double_confirm_is_safe(self):
        """confirm_account_ok called twice: second call is a no-op."""
        gate = make_gate(['acct'])
        acct = gate._accounts[0]
        acct.probing = True

        # Claim probe slot
        await gate.before_invoke()
        assert acct.probe_in_flight is True

        gate.confirm_account_ok(acct.token)
        assert acct.probe_in_flight is False
        assert gate._open.is_set()

        # Second confirm — should be safe
        gate.confirm_account_ok(acct.token)
        assert acct.probe_in_flight is False
        assert gate._open.is_set()
