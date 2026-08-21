"""Exhaustive tests for UsageGate._account_resume_probe_loop and _run_probe."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.cli_invoke import AgentResult
from shared.config_models import AccountConfig, UsageCapConfig
from shared.invocation_outcome import (
    CAP_HIT_PREFIXES,
    NEAR_CAP_PREFIXES,
    CapHit,
    CliLocalError,
    NearCap,
    classify_invocation,
)
from shared.usage_gate import (
    _SPAWN_FAULT_THRESHOLD,
    AccountPhase,
    AccountState,
    IllegalTransitionError,
    ProbeSpawnError,
    UsageGate,
)

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
    return gate


def make_mock_cost_store() -> AsyncMock:
    store = AsyncMock()
    store.save_account_event = AsyncMock(return_value=None)
    return store


def _capped_account(
    gate: UsageGate,
    *,
    resets_at: datetime | None = None,
    probe_count: int = 0,
    pause_started_at: datetime | None = None,
) -> AccountState:
    """Return the first account on *gate*, marked as capped with the given fields."""
    acct = gate._accounts[0]
    acct.capped = True
    acct.probe_count = probe_count
    acct.resets_at = resets_at
    acct.pause_started_at = pause_started_at
    return acct


# ---------------------------------------------------------------------------
# TestProbeLoopBackoff
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestProbeLoopBackoff:
    """Backoff interval calculations in _account_resume_probe_loop."""

    async def _run_single_iteration(
        self,
        gate: UsageGate,
        acct: AccountState,
    ) -> float:
        """Run exactly one iteration of the probe loop, capturing sleep_for.

        Mocks asyncio.sleep to capture the argument, then uncaps the account
        so the loop exits on the capped check after sleep.
        """
        captured_sleep: list[float] = []

        original_sleep = asyncio.sleep

        async def capture_sleep(duration: float) -> None:
            captured_sleep.append(duration)
            # Uncap after capturing so the loop exits
            acct.capped = False
            await original_sleep(0)

        # _run_probe should not be reached because we uncap during sleep
        gate._run_probe = AsyncMock(return_value=False)

        with patch('asyncio.sleep', side_effect=capture_sleep):
            await asyncio.wait_for(
                gate._account_resume_probe_loop(acct), timeout=5,
            )

        if captured_sleep:
            return captured_sleep[0]
        return 0.0

    async def test_probe_count_0_interval_equals_base(self):
        """probe_count=0 -> interval = base * 2^0 = base."""
        gate = make_gate(['a'], probe_interval_secs=300, max_probe_interval_secs=1800)
        acct = _capped_account(
            gate,
            resets_at=datetime.now(UTC) + timedelta(hours=2),
            probe_count=0,
        )
        slept = await self._run_single_iteration(gate, acct)
        assert slept == pytest.approx(300, abs=2)

    async def test_probe_count_1_interval_doubles(self):
        """probe_count=1 -> interval = base * 2."""
        gate = make_gate(['a'], probe_interval_secs=300, max_probe_interval_secs=1800)
        acct = _capped_account(
            gate,
            resets_at=datetime.now(UTC) + timedelta(hours=2),
            probe_count=1,
        )
        slept = await self._run_single_iteration(gate, acct)
        assert slept == pytest.approx(600, abs=2)

    async def test_probe_count_2_interval_quadruples(self):
        """probe_count=2 -> interval = base * 4."""
        gate = make_gate(['a'], probe_interval_secs=300, max_probe_interval_secs=1800)
        acct = _capped_account(
            gate,
            resets_at=datetime.now(UTC) + timedelta(hours=2),
            probe_count=2,
        )
        slept = await self._run_single_iteration(gate, acct)
        assert slept == pytest.approx(1200, abs=2)

    async def test_probe_count_5_interval_base_times_32(self):
        """probe_count=5 -> interval = base * 32."""
        gate = make_gate(['a'], probe_interval_secs=10, max_probe_interval_secs=100000)
        acct = _capped_account(
            gate,
            resets_at=datetime.now(UTC) + timedelta(hours=2),
            probe_count=5,
        )
        slept = await self._run_single_iteration(gate, acct)
        assert slept == pytest.approx(320, abs=2)

    async def test_probe_count_10_capped_at_ceiling(self):
        """probe_count=10 -> interval capped at ceiling (1800), not 300*1024=307200."""
        gate = make_gate(['a'], probe_interval_secs=300, max_probe_interval_secs=1800)
        acct = _capped_account(
            gate,
            resets_at=datetime.now(UTC) + timedelta(hours=2),
            probe_count=10,
        )
        slept = await self._run_single_iteration(gate, acct)
        assert slept == pytest.approx(1800, abs=2)

    async def test_ceiling_enforcement(self):
        """base=300, ceiling=1800, count=10 -> 1800 not 307200."""
        gate = make_gate(['a'], probe_interval_secs=300, max_probe_interval_secs=1800)
        acct = _capped_account(
            gate,
            resets_at=datetime.now(UTC) + timedelta(hours=10),
            probe_count=10,
        )
        slept = await self._run_single_iteration(gate, acct)
        assert slept == 1800

    async def test_sleep_bounded_by_resets_at(self):
        """When resets_at is closer than interval, sleep = remaining."""
        gate = make_gate(['a'], probe_interval_secs=3600, max_probe_interval_secs=7200)
        acct = _capped_account(
            gate,
            resets_at=datetime.now(UTC) + timedelta(seconds=60),
            probe_count=0,
        )
        slept = await self._run_single_iteration(gate, acct)
        assert slept == pytest.approx(60, abs=3)

    async def test_sleep_zero_when_resets_at_in_past(self):
        """resets_at in the past -> remaining=0 -> sleep_for=0."""
        gate = make_gate(['a'], probe_interval_secs=300, max_probe_interval_secs=1800)
        acct = _capped_account(
            gate,
            resets_at=datetime.now(UTC) - timedelta(minutes=10),
            probe_count=0,
        )

        # sleep_for=0 means asyncio.sleep is NOT called. The loop proceeds
        # directly to probe. Mock _run_probe to succeed so the loop exits.
        gate._run_probe = AsyncMock(return_value=True)

        await asyncio.wait_for(
            gate._account_resume_probe_loop(acct), timeout=5,
        )

        assert not acct.capped

    async def test_very_high_probe_count_no_overflow(self):
        """probe_count=50 -> no overflow, just ceiling."""
        gate = make_gate(['a'], probe_interval_secs=300, max_probe_interval_secs=1800)
        acct = _capped_account(
            gate,
            resets_at=datetime.now(UTC) + timedelta(hours=10),
            probe_count=50,
        )
        slept = await self._run_single_iteration(gate, acct)
        assert slept == 1800


# ---------------------------------------------------------------------------
# TestProbeLoopLifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestProbeLoopLifecycle:
    """Lifecycle behavior of _account_resume_probe_loop."""

    async def test_probe_succeeds_first_try(self):
        """Probe succeeds first try -> acct uncapped, probing=True, probe_count=0, gate opens."""
        gate = make_gate(
            ['a'],
            probe_interval_secs=0,
            max_probe_interval_secs=0,
        )
        acct = _capped_account(
            gate,
            resets_at=datetime.now(UTC) - timedelta(minutes=1),
            probe_count=0,
        )

        gate._run_probe = AsyncMock(return_value=True)

        await asyncio.wait_for(
            gate._account_resume_probe_loop(acct), timeout=5,
        )

        assert not acct.capped
        assert acct.probing is True
        assert acct.probe_count == 0
        assert gate._open.is_set()

    async def test_probe_fails_once_then_succeeds(self):
        """Probe fails once then succeeds -> probe_count=0 on success, acct uncapped."""
        gate = make_gate(
            ['a'],
            probe_interval_secs=0,
            max_probe_interval_secs=0,
        )
        acct = _capped_account(
            gate,
            resets_at=datetime.now(UTC) - timedelta(minutes=1),
            probe_count=0,
        )

        gate._run_probe = AsyncMock(side_effect=[False, True])

        await asyncio.wait_for(
            gate._account_resume_probe_loop(acct), timeout=5,
        )

        assert not acct.capped
        assert acct.probe_count == 0
        gate._run_probe.assert_awaited()
        assert gate._run_probe.await_count == 2

    async def test_multiple_failures_then_success(self):
        """3 failures then success -> verify backoff progression."""
        gate = make_gate(
            ['a'],
            probe_interval_secs=0,
            max_probe_interval_secs=0,
        )
        acct = _capped_account(
            gate,
            resets_at=datetime.now(UTC) - timedelta(minutes=1),
            probe_count=0,
        )

        gate._run_probe = AsyncMock(side_effect=[False, False, False, True])

        await asyncio.wait_for(
            gate._account_resume_probe_loop(acct), timeout=5,
        )

        assert not acct.capped
        assert acct.probe_count == 0
        assert gate._run_probe.await_count == 4

    async def test_already_uncapped_by_refresh_exits_early(self):
        """acct.capped set to False externally during sleep -> early exit, no probe fired."""
        gate = make_gate(
            ['a'],
            probe_interval_secs=1,
            max_probe_interval_secs=10,
        )
        acct = _capped_account(
            gate,
            resets_at=datetime.now(UTC) + timedelta(hours=1),
            probe_count=0,
        )

        async def uncap_during_sleep(duration: float) -> None:
            acct.capped = False  # simulate external uncap

        gate._run_probe = AsyncMock(return_value=True)

        with patch('asyncio.sleep', side_effect=uncap_during_sleep):
            await asyncio.wait_for(
                gate._account_resume_probe_loop(acct), timeout=5,
            )

        # _run_probe should NOT have been called — loop exits after checking capped
        gate._run_probe.assert_not_awaited()

    async def test_cancellation_during_sleep(self):
        """CancelledError during asyncio.sleep -> returns cleanly, acct stays capped."""
        gate = make_gate(
            ['a'],
            probe_interval_secs=1,
            max_probe_interval_secs=10,
        )
        acct = _capped_account(
            gate,
            resets_at=datetime.now(UTC) + timedelta(hours=1),
            probe_count=0,
        )

        async def cancel_during_sleep(duration: float) -> None:
            raise asyncio.CancelledError()

        gate._run_probe = AsyncMock(return_value=True)

        with patch('asyncio.sleep', side_effect=cancel_during_sleep):
            await asyncio.wait_for(
                gate._account_resume_probe_loop(acct), timeout=5,
            )

        assert acct.capped is True
        gate._run_probe.assert_not_awaited()

    async def test_no_resets_at_defaults_to_1h(self):
        """No resets_at -> defaults target to 1h from now, sleep doesn't complete quickly."""
        gate = make_gate(
            ['a'],
            probe_interval_secs=60,
            max_probe_interval_secs=3600,
        )
        acct = _capped_account(
            gate,
            resets_at=None,
            probe_count=0,
        )

        captured_sleep: list[float] = []

        async def capture_and_uncap(duration: float) -> None:
            captured_sleep.append(duration)
            acct.capped = False  # exit the loop

        gate._run_probe = AsyncMock(return_value=True)

        with patch('asyncio.sleep', side_effect=capture_and_uncap):
            await asyncio.wait_for(
                gate._account_resume_probe_loop(acct), timeout=5,
            )

        assert len(captured_sleep) == 1
        # base=60, ceiling=3600, count=0 -> interval=60
        # remaining ~ 3600 (1h default), so sleep_for = min(60, ~3600) = 60
        assert captured_sleep[0] == pytest.approx(60, abs=2)

    async def test_pause_duration_tracked(self):
        """pause_started_at consumed, _total_pause_secs updated on success."""
        gate = make_gate(
            ['a'],
            probe_interval_secs=0,
            max_probe_interval_secs=0,
        )
        start_time = datetime.now(UTC) - timedelta(seconds=120)
        acct = _capped_account(
            gate,
            resets_at=datetime.now(UTC) - timedelta(minutes=1),
            probe_count=0,
            pause_started_at=start_time,
        )

        gate._run_probe = AsyncMock(return_value=True)
        gate._total_pause_secs = 0.0
        # step-29/30: _total_pause_secs is gate-level-only. _capped_account
        # builds CAPPED state without calling _transition, so the gate clock
        # is never stamped — do it here to match the per-account backdate.
        gate._pause_started_at = start_time

        await asyncio.wait_for(
            gate._account_resume_probe_loop(acct), timeout=5,
        )

        assert acct.pause_started_at is None
        assert gate._total_pause_secs >= 119  # at least 119s of 120s pause

    async def test_pause_started_at_none_no_crash(self):
        """pause_started_at is None -> no duration tracking crash."""
        gate = make_gate(
            ['a'],
            probe_interval_secs=0,
            max_probe_interval_secs=0,
        )
        acct = _capped_account(
            gate,
            resets_at=datetime.now(UTC) - timedelta(minutes=1),
            probe_count=0,
            pause_started_at=None,
        )

        gate._run_probe = AsyncMock(return_value=True)
        gate._total_pause_secs = 0.0

        await asyncio.wait_for(
            gate._account_resume_probe_loop(acct), timeout=5,
        )

        assert gate._total_pause_secs == 0.0  # unchanged
        assert acct.pause_started_at is None

    async def test_gate_opened_on_successful_probe(self):
        """self._open.set() called on successful probe."""
        gate = make_gate(
            ['a'],
            probe_interval_secs=0,
            max_probe_interval_secs=0,
        )
        acct = _capped_account(
            gate,
            resets_at=datetime.now(UTC) - timedelta(minutes=1),
            probe_count=0,
        )

        # Close the gate first
        gate._open.clear()
        assert not gate._open.is_set()

        gate._run_probe = AsyncMock(return_value=True)

        await asyncio.wait_for(
            gate._account_resume_probe_loop(acct), timeout=5,
        )

        assert gate._open.is_set()


# ---------------------------------------------------------------------------
# TestProbeLoopCostEvents
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestProbeLoopCostEvents:
    """Cost event emission in _account_resume_probe_loop."""

    async def test_emits_resumed_event_on_success(self):
        """Emits 'resumed' event via _write_cost_event on success when cost_store is set."""
        store = make_mock_cost_store()
        gate = make_gate(
            ['a'],
            cost_store=store,
            probe_interval_secs=0,
            max_probe_interval_secs=0,
        )
        acct = _capped_account(
            gate,
            resets_at=datetime.now(UTC) - timedelta(minutes=1),
            probe_count=0,
        )

        gate._run_probe = AsyncMock(return_value=True)

        with patch.object(
            gate, '_write_cost_event', new_callable=AsyncMock,
        ) as mock_write:
            await asyncio.wait_for(
                gate._account_resume_probe_loop(acct), timeout=5,
            )

        mock_write.assert_awaited_once()
        args = mock_write.call_args[0]
        assert args[0] == 'a'        # account_name
        assert args[1] == 'resumed'  # event_type

    async def test_event_details_include_label_key(self):
        """Event details JSON includes 'label' key."""
        store = make_mock_cost_store()
        gate = make_gate(
            ['a'],
            cost_store=store,
            probe_interval_secs=0,
            max_probe_interval_secs=0,
        )
        acct = _capped_account(
            gate,
            resets_at=datetime.now(UTC) - timedelta(minutes=1),
            probe_count=0,
        )

        gate._run_probe = AsyncMock(return_value=True)

        with patch.object(
            gate, '_write_cost_event', new_callable=AsyncMock,
        ) as mock_write:
            await asyncio.wait_for(
                gate._account_resume_probe_loop(acct), timeout=5,
            )

        details = json.loads(mock_write.call_args[0][2])
        assert 'label' in details

    async def test_no_write_cost_event_when_cost_store_none(self):
        """No _write_cost_event call when cost_store=None."""
        gate = make_gate(
            ['a'],
            cost_store=None,
            probe_interval_secs=0,
            max_probe_interval_secs=0,
        )
        acct = _capped_account(
            gate,
            resets_at=datetime.now(UTC) - timedelta(minutes=1),
            probe_count=0,
        )

        gate._run_probe = AsyncMock(return_value=True)

        with patch.object(
            gate, '_write_cost_event', new_callable=AsyncMock,
        ) as mock_write:
            await asyncio.wait_for(
                gate._account_resume_probe_loop(acct), timeout=5,
            )

        mock_write.assert_not_awaited()

    async def test_no_event_on_probe_failure(self):
        """No event emitted on probe failure."""
        store = make_mock_cost_store()
        gate = make_gate(
            ['a'],
            cost_store=store,
            probe_interval_secs=0,
            max_probe_interval_secs=0,
        )
        acct = _capped_account(
            gate,
            resets_at=datetime.now(UTC) - timedelta(minutes=1),
            probe_count=0,
        )

        # Fail once, then succeed to exit loop
        gate._run_probe = AsyncMock(side_effect=[False, True])

        with patch.object(
            gate, '_write_cost_event', new_callable=AsyncMock,
        ) as mock_write:
            await asyncio.wait_for(
                gate._account_resume_probe_loop(acct), timeout=5,
            )

        # Only the 'resumed' event on success, not a 'failed' event
        assert mock_write.await_count == 1
        assert mock_write.call_args[0][1] == 'resumed'

    async def test_label_reflects_probe_count_on_first_try(self):
        """After a successful first-try probe, the label should reflect the probe number that confirmed (1, not 0)."""
        store = make_mock_cost_store()
        gate = make_gate(
            ['a'],
            cost_store=store,
            probe_interval_secs=0,
            max_probe_interval_secs=0,
        )
        acct = _capped_account(
            gate,
            resets_at=datetime.now(UTC) - timedelta(minutes=1),
            probe_count=0,
        )

        gate._run_probe = AsyncMock(return_value=True)

        with patch.object(
            gate, '_write_cost_event', new_callable=AsyncMock,
        ) as mock_write:
            await asyncio.wait_for(
                gate._account_resume_probe_loop(acct), timeout=5,
            )

        details = json.loads(mock_write.call_args[0][2])
        assert details['label'] == 'probe #1 confirmed'

    async def test_label_reflects_actual_probe_count_after_multiple_failures(self):
        """With probe_count=0 and [False, False, True] side_effect: iteration 1 increments to 1
        then fails, iteration 2 increments to 2 then fails, iteration 3 increments to 3 then
        succeeds — label should read 'probe #3 confirmed'."""
        store = make_mock_cost_store()
        gate = make_gate(
            ['a'],
            cost_store=store,
            probe_interval_secs=0,
            max_probe_interval_secs=0,
        )
        acct = _capped_account(
            gate,
            resets_at=datetime.now(UTC) - timedelta(minutes=1),
            probe_count=0,
        )

        gate._run_probe = AsyncMock(side_effect=[False, False, True])

        with patch.object(
            gate, '_write_cost_event', new_callable=AsyncMock,
        ) as mock_write:
            await asyncio.wait_for(
                gate._account_resume_probe_loop(acct), timeout=5,
            )

        assert mock_write.await_count == 1
        details = json.loads(mock_write.call_args[0][2])
        assert details['label'] == 'probe #3 confirmed'
        assert gate._run_probe.await_count == 3


# ---------------------------------------------------------------------------
# TestRunProbe — mock asyncio.create_subprocess_exec
# ---------------------------------------------------------------------------


def _make_mock_proc(
    returncode: int = 0,
    stdout: bytes = b'',
    stderr: bytes = b'',
) -> MagicMock:
    """Return a mock process whose communicate() returns (stdout, stderr)."""
    proc = MagicMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    return proc


@pytest.mark.asyncio
class TestRunProbe:
    """Tests for _run_probe — mock asyncio.create_subprocess_exec."""

    async def _make_probing_gate(self) -> tuple[UsageGate, AccountState]:
        gate = make_gate(['a'])
        acct = gate._accounts[0]
        acct.capped = True
        return gate, acct

    async def test_success_exit_0_no_cap_patterns(self):
        """Exit code 0, no cap patterns -> returns True."""
        gate, acct = await self._make_probing_gate()
        proc = _make_mock_proc(returncode=0, stdout=b'ok')

        with patch('asyncio.create_subprocess_exec', return_value=proc):
            result = await gate._run_probe(acct)

        assert result is True

    async def test_cap_hit_pattern_in_stderr(self):
        """Cap pattern 'You've hit your' in stderr -> returns False."""
        gate, acct = await self._make_probing_gate()
        proc = _make_mock_proc(
            returncode=0,
            stderr=b"You've hit your usage limit",
        )

        with patch('asyncio.create_subprocess_exec', return_value=proc):
            result = await gate._run_probe(acct)

        assert result is False

    async def test_cap_pattern_in_stdout(self):
        """Cap pattern 'You're close to' in stdout -> returns False."""
        gate, acct = await self._make_probing_gate()
        proc = _make_mock_proc(
            returncode=0,
            stdout=b"You're close to your usage limit",
        )

        with patch('asyncio.create_subprocess_exec', return_value=proc):
            result = await gate._run_probe(acct)

        assert result is False

    async def test_near_cap_pattern_in_combined(self):
        """Near-cap pattern in combined output -> returns False."""
        gate, acct = await self._make_probing_gate()
        proc = _make_mock_proc(
            returncode=0,
            stderr=b'Some info',
            stdout=b"You're close to your usage limit for this billing period",
        )

        with patch('asyncio.create_subprocess_exec', return_value=proc):
            result = await gate._run_probe(acct)

        assert result is False

    async def test_nonzero_exit_code_no_cap_pattern(self):
        """Non-zero exit code (no cap pattern) -> returns False."""
        gate, acct = await self._make_probing_gate()
        proc = _make_mock_proc(returncode=1, stdout=b'error')

        with patch('asyncio.create_subprocess_exec', return_value=proc):
            result = await gate._run_probe(acct)

        assert result is False

    @pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
    async def test_timeout_returns_false(self):
        """Timeout (asyncio.wait_for raises TimeoutError) -> returns False."""
        gate, acct = await self._make_probing_gate()
        proc = _make_mock_proc(returncode=0, stdout=b'ok')

        async def fake_wait_for(coro, *args, **kwargs):
            coro.close()
            raise TimeoutError()

        with (
            patch('asyncio.create_subprocess_exec', return_value=proc),
            patch('shared.usage_gate.asyncio.wait_for', side_effect=fake_wait_for),
        ):
            result = await gate._run_probe(acct)

        assert result is False

    async def test_cancelled_error_propagates(self):
        """CancelledError must propagate so shutdown() can drain the probe task.

        Previously _run_probe swallowed the cancel and returned False, which
        left ``_account_resume_probe_loop`` looping forever and made
        ``UsageGate.shutdown()`` hang waiting for the task to finish.
        """
        gate, acct = await self._make_probing_gate()

        async def cancel_exec(*args, **kwargs):
            raise asyncio.CancelledError()

        with (
            patch('asyncio.create_subprocess_exec', side_effect=cancel_exec),
            pytest.raises(asyncio.CancelledError),
        ):
            await gate._run_probe(acct)

    async def test_general_exception_returns_false(self):
        """A non-OSError exception from the spawn -> still returns False.

        The generic ``except Exception`` arm is unchanged. Its EXAMPLE changed
        (task 4512): this test used to raise ``FileNotFoundError`` here and
        assert ``False``, which is precisely the defect — a missing binary is
        an infrastructure fault, and it now raises ``ProbeSpawnError`` (see
        ``TestRunProbeSpawnFault``). ``RuntimeError`` is not an ``OSError``, so
        it still falls through to the generic arm and keeps the old verdict:
        only the OSError-at-spawn case was reclassified, not everything.
        """
        gate, acct = await self._make_probing_gate()

        with patch(
            'asyncio.create_subprocess_exec',
            side_effect=RuntimeError('something unexpected'),
        ):
            result = await gate._run_probe(acct)

        assert result is False

    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    async def test_credentials_written_before_subprocess(self):
        """Credentials written to probe_config_dir before subprocess call."""
        gate, acct = await self._make_probing_gate()
        proc = _make_mock_proc(returncode=0, stdout=b'ok')

        call_order: list[str] = []

        original_write = gate._probe_config_dir.write_credentials

        def track_write(token: str) -> None:
            call_order.append('write_creds')
            original_write(token)

        async def track_exec(*args, **kwargs):
            call_order.append('subprocess')
            return proc

        # Test-only monkeypatch of a method — pyright flags the assign.
        gate._probe_config_dir.write_credentials = track_write  # type: ignore[method-assign]

        with patch('asyncio.create_subprocess_exec', side_effect=track_exec):
            await gate._run_probe(acct)

        assert call_order == ['write_creds', 'subprocess']

    async def test_env_oauth_token_set(self):
        """Env var CLAUDE_CODE_OAUTH_TOKEN set to acct.token."""
        gate, acct = await self._make_probing_gate()
        proc = _make_mock_proc(returncode=0, stdout=b'ok')
        captured_env: list[dict] = []

        async def capture_exec(*args, **kwargs):
            captured_env.append(kwargs.get('env', {}))
            return proc

        with patch('asyncio.create_subprocess_exec', side_effect=capture_exec):
            await gate._run_probe(acct)

        assert captured_env[0]['CLAUDE_CODE_OAUTH_TOKEN'] == acct.token

    async def test_env_config_dir_set(self):
        """Env var CLAUDE_CONFIG_DIR set to probe_config_dir.path."""
        gate, acct = await self._make_probing_gate()
        proc = _make_mock_proc(returncode=0, stdout=b'ok')
        captured_env: list[dict] = []

        async def capture_exec(*args, **kwargs):
            captured_env.append(kwargs.get('env', {}))
            return proc

        with patch('asyncio.create_subprocess_exec', side_effect=capture_exec):
            await gate._run_probe(acct)

        assert captured_env[0]['CLAUDE_CONFIG_DIR'] == str(gate._probe_config_dir.path)

    async def test_anthropic_api_key_stripped(self):
        """ANTHROPIC_API_KEY stripped from env."""
        gate, acct = await self._make_probing_gate()
        proc = _make_mock_proc(returncode=0, stdout=b'ok')
        captured_env: list[dict] = []

        async def capture_exec(*args, **kwargs):
            captured_env.append(kwargs.get('env', {}))
            return proc

        with (
            patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'secret-key'}),
            patch('asyncio.create_subprocess_exec', side_effect=capture_exec),
        ):
            await gate._run_probe(acct)

        assert 'ANTHROPIC_API_KEY' not in captured_env[0]

    async def test_command_includes_expected_args(self):
        """Command includes 'haiku', '--max-turns', '1', etc."""
        gate, acct = await self._make_probing_gate()
        proc = _make_mock_proc(returncode=0, stdout=b'ok')
        captured_args: list[tuple] = []

        async def capture_exec(*args, **kwargs):
            captured_args.append(args)
            return proc

        with patch('asyncio.create_subprocess_exec', side_effect=capture_exec):
            await gate._run_probe(acct)

        cmd = captured_args[0]
        assert 'claude' in cmd
        assert '--print' in cmd
        assert '--output-format' in cmd
        assert 'json' in cmd
        assert '--model' in cmd
        assert 'haiku' in cmd
        assert '--max-turns' in cmd
        assert '1' in cmd
        assert '--max-budget-usd' in cmd
        assert '0.01' in cmd
        assert '--permission-mode' in cmd
        assert 'bypassPermissions' in cmd
        assert 'Say ok' in cmd

    async def test_empty_stdout_and_stderr_returns_true(self):
        """Empty stdout and stderr -> returns True (exit code 0)."""
        gate, acct = await self._make_probing_gate()
        proc = _make_mock_proc(returncode=0, stdout=b'', stderr=b'')

        with patch('asyncio.create_subprocess_exec', return_value=proc):
            result = await gate._run_probe(acct)

        assert result is True

    async def test_nonzero_exit_with_cap_pattern_returns_false(self):
        """Non-zero exit code WITH cap pattern -> returns False.

        Cap patterns are checked before return code in the source, so this
        should hit the cap pattern branch. Either way the result is False.
        """
        gate, acct = await self._make_probing_gate()
        proc = _make_mock_proc(
            returncode=1,
            stderr=b"You've hit your usage limit",
        )

        with patch('asyncio.create_subprocess_exec', return_value=proc):
            result = await gate._run_probe(acct)

        assert result is False

    async def test_all_cap_hit_prefixes_detected(self):
        """Every prefix in CAP_HIT_PREFIXES triggers False."""
        gate, acct = await self._make_probing_gate()

        for prefix in CAP_HIT_PREFIXES:
            proc = _make_mock_proc(returncode=0, stderr=prefix.encode())
            with patch('asyncio.create_subprocess_exec', return_value=proc):
                result = await gate._run_probe(acct)
            assert result is False, f'CAP_HIT prefix {prefix!r} not detected'

    async def test_all_near_cap_prefixes_detected(self):
        """Every prefix in NEAR_CAP_PREFIXES triggers False."""
        gate, acct = await self._make_probing_gate()

        for prefix in NEAR_CAP_PREFIXES:
            proc = _make_mock_proc(returncode=0, stdout=prefix.encode())
            with patch('asyncio.create_subprocess_exec', return_value=proc):
                result = await gate._run_probe(acct)
            assert result is False, f'NEAR_CAP prefix {prefix!r} not detected'

    async def test_case_insensitive_cap_pattern_match(self):
        """Cap pattern matching is case-insensitive."""
        gate, acct = await self._make_probing_gate()
        proc = _make_mock_proc(
            returncode=0,
            stderr=b"YOU'VE HIT YOUR USAGE LIMIT",
        )

        with patch('asyncio.create_subprocess_exec', return_value=proc):
            result = await gate._run_probe(acct)

        assert result is False

    async def test_probe_prefix_only_without_confirm_keyword_still_returns_false(self):
        """Probe returns False on a bare CAP_HIT prefix with no CAP_CONFIRM_KEYWORDS keyword.

        Deliberate asymmetry with detect_cap_hit:
        - detect_cap_hit requires BOTH a prefix AND a confirm keyword ('resets', 'usage
          limit', 'upgrade your plan') to avoid false positives on generic phrases.
        - _run_probe intentionally does NOT apply the confirm-keyword guard.  The probe
          runs only while an account is already capped; any whiff of a cap prefix in the
          probe output means the account is still capped and we must NOT unpause it.
          Being conservative here avoids the far worse outcome of unpausing a capped
          account and burning quota.

        DO NOT 'fix' this asymmetry by adding the confirm-keyword guard to _run_probe.
        If you think the asymmetry is a bug, read the inline comment above the prefix
        loop in _run_probe and this docstring — then escalate rather than silently change
        the behavior.
        """
        gate, acct = await self._make_probing_gate()
        prefix = CAP_HIT_PREFIXES[0]  # e.g. "You've hit your"
        # Deliberately no 'resets', 'usage limit', or 'upgrade your plan' in the string.
        stderr_content = f'{prefix} quota'.encode()
        proc = _make_mock_proc(returncode=0, stderr=stderr_content)

        with patch('asyncio.create_subprocess_exec', return_value=proc):
            result = await gate._run_probe(acct)

        assert result is False, (
            '_run_probe must return False on a bare cap prefix even without a confirm keyword; '
            'see docstring for the deliberate asymmetry with detect_cap_hit'
        )


# ---------------------------------------------------------------------------
# task 2129 (W4-beta) step-3: _run_probe's cap detection must route through
# classify_invocation(strict_confirm=False) — the DD-2 probe regime — so
# CliLocalError precedence (reify-3604) applies here too, not just at
# detect_cap_hit/cli_invoke.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRunProbeSpawnFault:
    """_run_probe must distinguish "could not spawn" from "ran, still capped".

    Task 4512. Both used to be ``return False``, which made a permanent host
    fault (unresolvable ``claude`` binary) indistinguishable from a usage cap
    all the way up the stack. The spawn fault now raises ``ProbeSpawnError``.

    Half of this class is the RAISES side; the other half is the ORDERING TRAP
    that the fix's shape exists to avoid. On py3.13
    ``issubclass(TimeoutError, OSError)`` is True and ``asyncio.TimeoutError is
    TimeoutError`` — so an ``except OSError`` arm added to the EXISTING
    combined try (or placed before the current ``except TimeoutError``) would
    silently reclassify every 30-second probe timeout as a missing-binary host
    fault. That is the same misclassification this task removes, pointed the
    other way. The fix therefore SPLITS the try so the OSError arm can only
    ever see the spawn; these tests pin that boundary from both sides.
    """

    async def _make_probing_gate(self) -> tuple[UsageGate, AccountState]:
        gate = make_gate(['a'])
        acct = gate._accounts[0]
        acct.capped = True
        return gate, acct

    # --- RAISES: every OSError from the spawn is an infrastructure fault ---

    async def test_missing_binary_raises_probe_spawn_error(self, caplog):
        """The observed incident: `claude` is not on PATH."""
        gate, acct = await self._make_probing_gate()
        cause = FileNotFoundError(2, 'No such file or directory', 'claude')

        with (
            caplog.at_level(logging.ERROR, logger='shared.usage_gate'),
            patch('asyncio.create_subprocess_exec', side_effect=cause),
            pytest.raises(ProbeSpawnError) as excinfo,
        ):
            await gate._run_probe(acct)

        assert excinfo.value.binary == 'claude'
        assert isinstance(excinfo.value.cause, FileNotFoundError)

        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert errors, (
            'a spawn fault must be logged at ERROR, not the WARNING an '
            'ordinary probe failure gets — it is an operator action item'
        )
        rendered = ' '.join(r.getMessage() for r in errors)
        assert acct.name in rendered, rendered
        assert 'No such file or directory' in rendered, rendered

    @pytest.mark.parametrize(
        'cause',
        [
            PermissionError(13, 'Permission denied', 'claude'),
            NotADirectoryError(20, 'Not a directory', 'claude'),
        ],
        ids=['permission-denied', 'not-a-directory'],
    )
    async def test_every_oserror_at_spawn_is_a_spawn_fault(self, cause):
        """Not just ENOENT.

        A non-executable binary (chmod loss) and a broken self-update symlink
        whose parent is a file both surface as other OSError subclasses, and
        both mean the same thing: the probe never ran, so nothing was learned
        about the account's capacity. Enumerating errnos would leave the next
        variant silently classified as a cap.
        """
        gate, acct = await self._make_probing_gate()

        with (
            patch('asyncio.create_subprocess_exec', side_effect=cause),
            pytest.raises(ProbeSpawnError) as excinfo,
        ):
            await gate._run_probe(acct)

        assert excinfo.value.cause is cause
        assert excinfo.value.binary == 'claude'

    # --- DOES NOT RAISE: the probe ran, so its bool verdict stands ---

    @pytest.mark.filterwarnings('error::pytest.PytestUnraisableExceptionWarning')
    async def test_timeout_is_not_a_spawn_fault(self):
        """THE ORDERING TRAP. A 30s probe timeout still returns False.

        `TimeoutError` IS an `OSError` subclass, so this is the single test
        that distinguishes the split-try fix from the tempting one-line
        `except OSError` added to the combined try. If it ever goes red with a
        ProbeSpawnError, the OSError arm has drifted onto the read.
        """
        gate, acct = await self._make_probing_gate()
        proc = _make_mock_proc(returncode=0, stdout=b'ok')

        async def fake_wait_for(coro, *args, **kwargs):
            coro.close()
            raise TimeoutError()

        with (
            patch('asyncio.create_subprocess_exec', return_value=proc),
            patch('shared.usage_gate.asyncio.wait_for', side_effect=fake_wait_for),
        ):
            result = await gate._run_probe(acct)

        assert result is False

    async def test_broken_pipe_during_read_is_not_a_spawn_fault(self):
        """A mid-read BrokenPipeError (also an OSError) still returns False.

        The spawn SUCCEEDED here, so the probe ran; whatever broke afterwards
        is not evidence that the host cannot start `claude`. Latching the
        gate unhealthy on this would make a flaky pipe look like a missing
        binary — and, unlike a missing binary, no operator action fixes it.
        """
        gate, acct = await self._make_probing_gate()
        proc = MagicMock()
        proc.returncode = 0
        proc.communicate = AsyncMock(
            side_effect=BrokenPipeError(32, 'Broken pipe'),
        )

        with patch('asyncio.create_subprocess_exec', return_value=proc):
            result = await gate._run_probe(acct)

        assert result is False

    async def test_non_oserror_at_spawn_is_not_a_spawn_fault(self):
        """A RuntimeError from the spawn still returns False.

        The reclassification is scoped to OSError specifically. A caller-side
        bug or a mocking accident must not masquerade as a host fault and
        latch the gate unhealthy.
        """
        gate, acct = await self._make_probing_gate()

        with patch(
            'asyncio.create_subprocess_exec',
            side_effect=RuntimeError('not an OSError'),
        ):
            result = await gate._run_probe(acct)

        assert result is False

    async def test_cancelled_error_still_propagates(self):
        """CancelledError must keep propagating past the new OSError arm.

        `asyncio.CancelledError` derives from BaseException, so it cannot be
        caught by `except OSError` — but the spawn's new arm sits ahead of
        every existing handler, so the shutdown-drain contract
        (`UsageGate.shutdown()` waits on this task) is re-pinned here rather
        than assumed.
        """
        gate, acct = await self._make_probing_gate()

        async def cancel_exec(*args, **kwargs):
            raise asyncio.CancelledError()

        with (
            patch('asyncio.create_subprocess_exec', side_effect=cancel_exec),
            pytest.raises(asyncio.CancelledError),
        ):
            await gate._run_probe(acct)


async def _drive_loop(
    gate: UsageGate,
    acct: AccountState,
    *,
    max_sleeps: int,
) -> list[float]:
    """Run `_account_resume_probe_loop` for a bounded number of iterations.

    Same fake-clock shape as `TestProbeLoopBackoff._run_single_iteration`:
    `asyncio.sleep` is replaced by a capture that never really waits, and the
    account is uncapped on the last one so the loop's own `if not acct.capped:
    return` terminates it. A persistently spawn-faulting probe would otherwise
    loop forever — which is precisely the production symptom under test.

    Note the off-by-one this shape implies: the uncapping sleep is followed by
    the capped-check and a return, so N sleeps drive N-1 probes.
    """
    captured: list[float] = []
    original_sleep = asyncio.sleep

    async def capture_sleep(duration: float) -> None:
        captured.append(duration)
        if len(captured) >= max_sleeps:
            acct.capped = False
        await original_sleep(0)

    with patch('asyncio.sleep', side_effect=capture_sleep):
        await asyncio.wait_for(gate._account_resume_probe_loop(acct), timeout=5)
    return captured


def _count_matching(caplog, needle: str) -> int:
    return sum(1 for r in caplog.records if needle in r.getMessage())


async def _drain_cost_events(gate: UsageGate) -> None:
    """Let `_fire_cost_event`'s fire-and-forget tasks reach the writer.

    `_fire_cost_event` schedules `_write_cost_event` via `loop.create_task`
    and returns immediately, so asserting on the writer straight after the
    loop would race it — and race it in the direction that makes a missing
    event look like a correctly-suppressed one.
    """
    for _ in range(5):
        if not gate._background_tasks:
            return
        await asyncio.gather(*list(gate._background_tasks))


def _events_of(mock_write, event_type: str) -> list[tuple]:
    return [c[0] for c in mock_write.call_args_list if c[0][1] == event_type]


def _spawn_fault(binary: str = 'claude') -> ProbeSpawnError:
    return ProbeSpawnError(
        binary, FileNotFoundError(2, 'No such file or directory', binary),
    )


class TestProbeInfraFaultLatch:
    """Gate-level accounting for probes that could not be spawned (task 4512).

    `shared` cannot import `escalation` (the dependency runs the other way), so
    the gate does what its same-layer peer `ApiHealthGate` does: it EXPOSES the
    condition and leaves the escalation lifecycle to a consumer that may
    legally file one. `probe_infra_fault` is that surface.

    The latch is deliberately consecutive-counted rather than windowed. A
    spawn error immediately followed by a probe that RAN is genuinely
    transient and must clear; a window counter would still latch on three
    errors interleaved with successes, which is a flaky host, not one that
    cannot start the binary at all.
    """

    def _fresh(self) -> tuple[UsageGate, AccountState]:
        gate = make_gate(['a'])
        return gate, gate._accounts[0]

    def test_fresh_gate_reports_no_fault(self):
        gate = make_gate(['a', 'b'])

        assert gate.probe_infra_fault is None
        assert [a.probe_spawn_failures for a in gate._accounts] == [0, 0]

    def test_below_threshold_counts_and_logs_but_does_not_latch(self, caplog):
        """Loud immediately; latched only at the threshold.

        The two are separate on purpose. Every spawn fault is an ERROR the
        moment it happens, so a transient one is never silent — but the LATCH
        is the durable "an operator must fix this host" claim, and staking that
        on a single failure would flap.
        """
        gate, acct = self._fresh()

        with caplog.at_level(logging.ERROR, logger='shared.usage_gate'):
            for expected_count in range(1, _SPAWN_FAULT_THRESHOLD):
                gate._note_probe_spawn_failure(acct, _spawn_fault())
                assert acct.probe_spawn_failures == expected_count
                assert gate.probe_infra_fault is None, (
                    f'latched after {expected_count} failure(s); the threshold '
                    f'is {_SPAWN_FAULT_THRESHOLD}'
                )

        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(errors) == _SPAWN_FAULT_THRESHOLD - 1, (
            'every spawn fault must log at ERROR, not only the latching one'
        )

    def test_threshold_arms_the_latch_naming_binary_and_cause(self, caplog):
        gate, acct = self._fresh()

        with caplog.at_level(logging.ERROR, logger='shared.usage_gate'):
            for _ in range(_SPAWN_FAULT_THRESHOLD):
                gate._note_probe_spawn_failure(acct, _spawn_fault())

        fault = gate.probe_infra_fault
        assert isinstance(fault, str) and fault, (
            f'expected a latched description after {_SPAWN_FAULT_THRESHOLD} '
            f'consecutive spawn failures, got {fault!r}'
        )
        assert 'claude' in fault, fault
        assert 'No such file or directory' in fault, fault

    def test_a_probe_that_ran_clears_the_counter_and_the_latch(self):
        """Recovery is proof-based: the binary is resolvable again.

        A probe that returned at all — True or False — spawned, so the host
        fault is over. Nothing else clears the latch; it is explicitly NOT
        time-based, because no amount of waiting fixes a missing binary.
        """
        gate, acct = self._fresh()
        for _ in range(_SPAWN_FAULT_THRESHOLD):
            gate._note_probe_spawn_failure(acct, _spawn_fault())
        assert gate.probe_infra_fault is not None

        gate._clear_probe_spawn_failures(acct)

        assert acct.probe_spawn_failures == 0
        assert gate.probe_infra_fault is None

    def test_non_consecutive_failures_never_latch(self):
        """note, note, clear, note, note -> still None.

        Four failures total, more than the threshold, but never three in a
        row. This is what makes the counter CONSECUTIVE rather than windowed.
        """
        gate, acct = self._fresh()

        gate._note_probe_spawn_failure(acct, _spawn_fault())
        gate._note_probe_spawn_failure(acct, _spawn_fault())
        gate._clear_probe_spawn_failures(acct)
        gate._note_probe_spawn_failure(acct, _spawn_fault())
        gate._note_probe_spawn_failure(acct, _spawn_fault())

        assert gate.probe_infra_fault is None
        assert acct.probe_spawn_failures == _SPAWN_FAULT_THRESHOLD - 1

    def test_counters_are_per_account(self):
        """Matching the observed incident, where all six accounts fault.

        Each account runs its own probe loop, so a shared counter would reach
        the threshold in a third of the time on a six-account fleet and make
        the threshold mean something different per deployment size.
        """
        gate = make_gate(['a', 'b'])
        acct_a, acct_b = gate._accounts

        gate._note_probe_spawn_failure(acct_a, _spawn_fault())
        gate._note_probe_spawn_failure(acct_a, _spawn_fault())
        gate._note_probe_spawn_failure(acct_b, _spawn_fault())

        assert acct_a.probe_spawn_failures == 2
        assert acct_b.probe_spawn_failures == 1
        assert gate.probe_infra_fault is None

    def test_new_built_gate_never_raises_attribute_error(self):
        """A gate built via `UsageGate.__new__`, as orchestrator tests build one.

        orchestrator/tests/_orch_helpers.py:1142 assigns ~15 fields by hand and
        never runs `__init__`, so it cannot know about a field added here. The
        module already has an idiom for exactly this
        (`getattr(self, '_shutting_down', False)`), and this test is what
        forces the new attribute to use it rather than a bare `self._...` read
        that would AttributeError deep inside an unrelated orchestrator test.
        """
        gate = UsageGate.__new__(UsageGate)
        gate._accounts = [AccountState(name='a', token='tok')]
        gate._cost_store = None
        gate._background_tasks = set()
        acct = gate._accounts[0]

        assert gate.probe_infra_fault is None

        for _ in range(_SPAWN_FAULT_THRESHOLD):
            gate._note_probe_spawn_failure(acct, _spawn_fault())

        assert acct.probe_spawn_failures == _SPAWN_FAULT_THRESHOLD
        assert gate.probe_infra_fault is not None

        gate._clear_probe_spawn_failures(acct)
        assert gate.probe_infra_fault is None


class TestStaleSpawnFailureCounter:
    """A spawn-failure counter must not outlive the account's probing phase.

    Task 4512 review finding. `probe_spawn_failures` counts CONSECUTIVE
    failures to SPAWN a probe, and only CAPPED/AUTH_FAILED accounts spawn
    probes at all (`_account_resume_probe_loop` and `_reprobe_account`
    respectively). But it was zeroed ONLY by `_clear_probe_spawn_failures` —
    i.e. by a probe that actually RAN for that same account — and an account
    can leave the probing phases without ever running another probe:

    - `_refresh_capped_accounts` takes CAPPED -> PROBING purely on
      `now >= acct.resets_at` (a `resets_at` written by the ORIGINAL genuine
      cap detection, which the spawn-fault branch deliberately never
      touches); `_account_resume_probe_loop` then exits at its
      `if not acct.capped: return` without clearing anything.
    - `_on_sighup_async` force-resets EVERY account to AVAILABLE.

    The counter left behind is stale forever, because the account will never
    probe again to zero it, and it breaks the latch in BOTH directions:

    1. it pins `probe_infra_fault` non-None long after the host is fixed,
       contradicting that property's documented contract ("None when probes
       can spawn"); and
    2. it lets a single later transient spawn error trip a threshold whose
       own comment promises `_SPAWN_FAULT_THRESHOLD` CONSECUTIVE failures.
    """

    def _capped_gate(self, names: list[str]) -> tuple[UsageGate, AccountState]:
        gate = make_gate(names)
        return gate, _capped_account(gate)

    def test_recovery_edge_resets_the_counter(self):
        """CAPPED -> PROBING is recovery: the pending run of failures is over.

        This is the exact edge `_refresh_capped_accounts` takes when
        `resets_at` elapses, and the one that stranded the counter.
        """
        gate, acct = self._capped_gate(['a'])

        for _ in range(_SPAWN_FAULT_THRESHOLD - 1):
            gate._note_probe_spawn_failure(acct, _spawn_fault())
        assert acct.probe_spawn_failures == _SPAWN_FAULT_THRESHOLD - 1

        gate._transition(acct, AccountPhase.PROBING)

        assert acct.probe_spawn_failures == 0, (
            'an account that left the probing phases cannot produce another '
            'probe to zero this counter, so leaving it set strands it forever'
        )

    def test_a_stale_counter_must_not_latch_a_later_transient_error(self):
        """Consequence 2, driven through the real recovery path.

        The count is deliberately reached by `_note_probe_spawn_failure` and
        cleared by a real `_transition`, never by assigning the field: the
        bug is that the REAL edges leave it behind.
        """
        gate, acct = self._capped_gate(['a'])

        for _ in range(_SPAWN_FAULT_THRESHOLD - 1):
            gate._note_probe_spawn_failure(acct, _spawn_fault())
        gate._transition(acct, AccountPhase.PROBING)  # recovery
        gate._transition(acct, AccountPhase.CAPPED)  # a fresh, unrelated cap much later

        gate._note_probe_spawn_failure(acct, _spawn_fault())

        assert acct.probe_spawn_failures == 1
        assert gate.probe_infra_fault is None, (
            f'one spawn error after a recovery latched the gate; '
            f'_SPAWN_FAULT_THRESHOLD is {_SPAWN_FAULT_THRESHOLD} CONSECUTIVE '
            f'failures, and the run before the recovery is not consecutive '
            f'with this one'
        )

    def test_a_stale_counter_must_not_pin_the_latch_unhealthy(self):
        """Consequence 1: the gate reports UNHEALTHY after the host is fixed."""
        gate, acct_a = self._capped_gate(['a', 'b'])
        acct_b = gate._accounts[1]

        for _ in range(_SPAWN_FAULT_THRESHOLD):
            gate._note_probe_spawn_failure(acct_a, _spawn_fault())
        assert gate.probe_infra_fault is not None

        # `a` leaves CAPPED the way _refresh_capped_accounts does — on the
        # clock, without ever running another probe.
        gate._transition(acct_a, AccountPhase.PROBING)

        # `b`'s probe spawned fine, so the binary is demonstrably resolvable.
        gate._clear_probe_spawn_failures(acct_b)

        assert gate.probe_infra_fault is None, (
            "the only account still probing has proven it can spawn; the "
            "latch is held open solely by a non-probing account's stale count"
        )

    def test_counter_survives_edges_inside_the_blocked_set(self):
        """AUTH_FAILED -> CAPPED preserves it: both phases still spawn probes.

        The guard against fixing this too broadly. Recovery is "left the
        blocked SET", not "left CAPPED" — an account moving between the two
        blocked phases keeps producing probe evidence, so the run of
        consecutive failures must keep accumulating across that edge.
        """
        gate = make_gate(['a'])
        acct = gate._accounts[0]
        acct.auth_failed = True

        for _ in range(_SPAWN_FAULT_THRESHOLD - 1):
            gate._note_probe_spawn_failure(acct, _spawn_fault())

        gate._transition(acct, AccountPhase.CAPPED)

        assert acct.probe_spawn_failures == _SPAWN_FAULT_THRESHOLD - 1
        assert acct.phase is AccountPhase.CAPPED

    def test_capped_to_auth_failed_is_not_a_reachable_edge(self):
        """Why the test above can only exercise one direction of that pair.

        `_LEGAL_TRANSITIONS[CAPPED]` is `{PROBING}` alone, so the CAPPED ->
        AUTH_FAILED direction is unreachable through the sole phase writer
        and needs no counter behaviour of its own. Pinned here so a later
        widening of the transition table is forced back through this class.
        """
        gate, acct = self._capped_gate(['a'])
        gate._note_probe_spawn_failure(acct, _spawn_fault())

        with pytest.raises(IllegalTransitionError):
            gate._transition(acct, AccountPhase.AUTH_FAILED)

        assert acct.phase is AccountPhase.CAPPED
        assert acct.probe_spawn_failures == 1

    def test_a_still_blocked_account_with_a_live_counter_blocks_the_clear(self):
        """Regression control: recovery must not become "any good probe wins".

        `a` genuinely still cannot spawn, so one good probe on `b` says
        nothing about `a` and the latch must hold. This pins that the fix
        does not overshoot into clearing on the first successful probe
        anywhere on the fleet.
        """
        gate, acct_a = self._capped_gate(['a', 'b'])
        acct_b = gate._accounts[1]
        acct_b.capped = True

        for _ in range(_SPAWN_FAULT_THRESHOLD):
            gate._note_probe_spawn_failure(acct_a, _spawn_fault())
        assert gate.probe_infra_fault is not None
        gate._note_probe_spawn_failure(acct_b, _spawn_fault())

        gate._clear_probe_spawn_failures(acct_b)

        assert acct_b.probe_spawn_failures == 0
        assert acct_a.probe_spawn_failures == _SPAWN_FAULT_THRESHOLD
        assert gate.probe_infra_fault is not None

    async def test_sighup_hard_reset_clears_every_counter_and_the_latch(self):
        """An operator "refresh everything" must not stay latched on evidence
        it just discarded.

        `_reprobe_account` is what is stubbed here, NOT `_run_probe`: a probe
        that RAN would zero the counters through `_clear_probe_spawn_failures`
        and mask the thing under test. What must hold is that the hard reset
        ITSELF discards the counts, since after it no account is in a phase
        that would ever produce another probe to clear them.
        """
        gate = make_gate(['a', 'b'])
        acct_a, acct_b = gate._accounts
        acct_a.capped = True  # exits the blocked set on the SIGHUP edge
        # acct_b stays AVAILABLE: a force=True self-loop, which the
        # edge-specific bookkeeping does not cover.

        for _ in range(_SPAWN_FAULT_THRESHOLD):
            gate._note_probe_spawn_failure(acct_a, _spawn_fault())
        gate._note_probe_spawn_failure(acct_b, _spawn_fault())
        assert gate.probe_infra_fault is not None

        gate._reprobe_account = AsyncMock(return_value=None)
        with patch('shared.usage_gate.load_dotenv'):
            await gate._on_sighup_async()

        assert [a.probe_spawn_failures for a in gate._accounts] == [0, 0]
        assert gate.probe_infra_fault is None

    async def test_sighup_against_a_still_broken_host_restarts_the_count(self):
        """The same SIGHUP end to end, with the real `_reprobe_account`.

        The operator refreshes while the binary is still missing. Every
        account's run of failures restarts from the reset, so the fault has
        to be re-established from scratch rather than re-latching instantly
        on counts the SIGHUP already invalidated.
        """
        gate = make_gate(['a', 'b'])
        acct_a, acct_b = gate._accounts
        acct_a.capped = True

        for _ in range(_SPAWN_FAULT_THRESHOLD):
            gate._note_probe_spawn_failure(acct_a, _spawn_fault())
        for _ in range(_SPAWN_FAULT_THRESHOLD - 1):
            gate._note_probe_spawn_failure(acct_b, _spawn_fault())
        assert gate.probe_infra_fault is not None

        gate._run_probe = AsyncMock(side_effect=_spawn_fault())
        with patch('shared.usage_gate.load_dotenv'):
            await gate._on_sighup_async()

        assert [a.probe_spawn_failures for a in gate._accounts] == [1, 1]
        assert gate.probe_infra_fault is None, (
            'a single post-reset spawn failure must not re-arm a latch the '
            'operator-driven reset had just cleared'
        )
    # --- Only accounts that can still PRODUCE evidence may veto a clear ----
    #
    # Defence in depth ALONGSIDE the reset above, not instead of it. The two
    # fix different halves and neither subsumes the other. The reset stops
    # stale counters being CREATED — and is the only thing that can fix the
    # premature-latch half, since at `_note_probe_spawn_failure` time the
    # account genuinely IS blocked, so no guard narrowing could help there.
    # The narrowed guard stops a stale counter that exists ANYWAY — reached
    # through a future exit path that bypasses `_transition`, or a
    # `UsageGate.__new__` fixture that assigns its fields by hand — from
    # pinning the latch UNHEALTHY forever.

    def _orphan_counter_gate(
        self, orphan_phase: AccountPhase
    ) -> tuple[UsageGate, AccountState, AccountState]:
        """Gate where `a` holds an orphaned counter in *orphan_phase* and `b`,
        CAPPED, has just armed the latch.

        `a`'s count is assigned DIRECTLY, deliberately bypassing `_transition`:
        it stands in for a counter that exists anyway, which is exactly the
        case the reset in the sole phase writer cannot police.
        """
        gate = make_gate(['a', 'b'])
        acct_a, acct_b = gate._accounts
        acct_a.phase = orphan_phase
        acct_a.probe_spawn_failures = _SPAWN_FAULT_THRESHOLD - 1

        acct_b.capped = True
        for _ in range(_SPAWN_FAULT_THRESHOLD):
            gate._note_probe_spawn_failure(acct_b, _spawn_fault())
        assert gate.probe_infra_fault is not None
        return gate, acct_a, acct_b

    def test_a_non_probing_account_cannot_veto_the_clear(self):
        """The only account that can still spawn probes has just proven it can.

        `a` is AVAILABLE: it runs neither a resume loop nor an auth re-probe,
        so it will never spawn a probe to zero its own counter. Letting it
        vote is what makes the latch unclearable. `probe_infra_fault`'s
        contract is "None whenever probes CAN spawn", and an account that
        does not probe says nothing at all about that.
        """
        gate, _acct_a, acct_b = self._orphan_counter_gate(AccountPhase.AVAILABLE)

        gate._clear_probe_spawn_failures(acct_b)

        assert gate.probe_infra_fault is None, (
            'a non-probing account is holding the latch open on a count it '
            'can never clear, because it will never probe again'
        )

    def test_narrowing_the_vote_mutates_nobody_else(self):
        """Who gets a vote changes; nobody else's counter does.

        The guard must not quietly become a fleet-wide zeroing pass — an
        orphaned count stays visible for diagnosis, it just stops voting.
        """
        gate, acct_a, acct_b = self._orphan_counter_gate(AccountPhase.AVAILABLE)

        gate._clear_probe_spawn_failures(acct_b)

        assert acct_b.probe_spawn_failures == 0
        assert acct_a.probe_spawn_failures == _SPAWN_FAULT_THRESHOLD - 1

    def test_an_auth_failed_account_does_get_a_vote(self):
        """AUTH_FAILED still probes, so its pending failures are still live.

        `_reprobe_account` -> `_run_probe` runs for an auth-failed account
        exactly as the resume loop does for a capped one. This pins the
        narrowing at `capped or auth_failed` rather than `capped` alone —
        dropping the auth path would take it back out of the accounting
        surface it was deliberately joined to.
        """
        gate, _acct_a, acct_b = self._orphan_counter_gate(AccountPhase.AUTH_FAILED)

        gate._clear_probe_spawn_failures(acct_b)

        assert gate.probe_infra_fault is not None

    def test_a_capped_account_still_gets_a_vote_after_the_narrowing(self):
        """Belt and braces that the reset and the narrowing compose.

        The regression control above, restated through the narrowed guard:
        `a` is still CAPPED, still running its resume loop, and still
        genuinely cannot spawn — so one good probe on `b` must not clear it.
        """
        gate, _acct_a, acct_b = self._orphan_counter_gate(AccountPhase.CAPPED)

        gate._clear_probe_spawn_failures(acct_b)

        assert gate.probe_infra_fault is not None



@pytest.mark.asyncio
class TestProbeInfraFaultCostEvents:
    """The `probe_infra_fault` account event is edge-triggered and re-armable.

    Task 4512. Forensics for an operator who reads the CostStore rather than
    the journal — but only useful if the row rate reflects EVENTS, not
    elapsed outage. The observed incident ran ~9 hours across six accounts;
    a per-probe write would have produced thousands of identical rows and
    buried the one fact that matters.
    """

    def _faulting_gate(self, store) -> tuple[UsageGate, AccountState]:
        gate = make_gate(
            ['a'], cost_store=store, probe_interval_secs=1, max_probe_interval_secs=8,
        )
        return gate, _capped_account(gate, resets_at=None)

    async def test_exactly_one_event_for_a_long_outage(self):
        store = make_mock_cost_store()
        gate, acct = self._faulting_gate(store)
        gate._run_probe = AsyncMock(side_effect=_spawn_fault())

        with patch.object(gate, '_write_cost_event', new_callable=AsyncMock) as mock_write:
            await _drive_loop(gate, acct, max_sleeps=10)
            await _drain_cost_events(gate)

        assert gate._run_probe.await_count == 9, 'harness drove the wrong depth'
        assert len(_events_of(mock_write, 'probe_infra_fault')) == 1, (
            'the latch must be edge-triggered: 9 faulted probes, one row'
        )

    async def test_event_details_are_diagnosable_without_the_log(self):
        store = make_mock_cost_store()
        gate, acct = self._faulting_gate(store)
        gate._run_probe = AsyncMock(side_effect=_spawn_fault())

        with patch.object(gate, '_write_cost_event', new_callable=AsyncMock) as mock_write:
            await _drive_loop(gate, acct, max_sleeps=10)
            await _drain_cost_events(gate)

        (event,) = _events_of(mock_write, 'probe_infra_fault')
        assert event[0] == 'a'
        details = json.loads(event[2])
        assert details['binary'] == 'claude'
        assert 'No such file or directory' in details['cause']
        assert details['consecutive'] == _SPAWN_FAULT_THRESHOLD

    async def test_no_cap_hit_event_is_written_on_the_spawn_fault_path(self):
        """The defect at the data layer.

        A host fault recorded as `cap_hit` would be absorbed by every cap
        dashboard and cap-history query as genuine load — the same
        indistinguishable-from-a-real-cap failure this task removes, one
        layer below the logs.
        """
        store = make_mock_cost_store()
        gate, acct = self._faulting_gate(store)
        gate._run_probe = AsyncMock(side_effect=_spawn_fault())

        with patch.object(gate, '_write_cost_event', new_callable=AsyncMock) as mock_write:
            await _drive_loop(gate, acct, max_sleeps=10)
            await _drain_cost_events(gate)

        assert _events_of(mock_write, 'cap_hit') == []
        assert _events_of(mock_write, 'resumed') == []

    async def test_a_recovered_then_broken_host_is_reported_twice(self):
        """Re-latching, not deduped forever.

        Three faults latch and fire; a probe that RUNS (returns False, so the
        account is still capped) clears the latch; three more faults latch and
        fire again. Without the clear re-arming the event, a host that broke,
        healed, and broke again would be silently reported once.
        """
        store = make_mock_cost_store()
        gate, acct = self._faulting_gate(store)
        gate._run_probe = AsyncMock(
            side_effect=[
                *[_spawn_fault()] * _SPAWN_FAULT_THRESHOLD,
                False,
                *[_spawn_fault()] * _SPAWN_FAULT_THRESHOLD,
            ],
        )

        with patch.object(gate, '_write_cost_event', new_callable=AsyncMock) as mock_write:
            await _drive_loop(gate, acct, max_sleeps=2 * _SPAWN_FAULT_THRESHOLD + 2)
            await _drain_cost_events(gate)

        assert len(_events_of(mock_write, 'probe_infra_fault')) == 2

    async def test_store_less_gate_takes_the_same_path_without_raising(self):
        """`_fire_cost_event` no-ops without a store; the latch still arms."""
        gate = make_gate(
            ['a'], cost_store=None, probe_interval_secs=1, max_probe_interval_secs=8,
        )
        acct = _capped_account(gate, resets_at=None)
        gate._run_probe = AsyncMock(side_effect=_spawn_fault())

        with patch.object(gate, '_write_cost_event', new_callable=AsyncMock) as mock_write:
            await _drive_loop(gate, acct, max_sleeps=10)
            await _drain_cost_events(gate)

        mock_write.assert_not_awaited()
        assert gate.probe_infra_fault is not None


@pytest.mark.asyncio
class TestProbeLoopSpawnFault:
    """The resume loop must stop inventing a reset clock it cannot know.

    Task 4512, the user-observable half. With `claude` unresolvable, every
    probe failed to spawn and returned False; the loop read that as "still
    capped", found `resets_at is None`, fabricated `now + 1h`, and logged
    "no resets_at - defaulting to 1h" plus a synthetic "resets in 3600s"
    countdown — forever. A permanent host fault presented as a self-inflicted
    usage cap, indefinitely, with a confident-looking clock attached.

    The second half of this class is the CONTROL: a genuine cap must not get
    noisier or behave differently. Those tests are green before AND after the
    fix, which is what makes them worth having — they pin that the non-fault
    branch stayed byte-identical rather than merely still passing.
    """

    # --- the fault path -------------------------------------------------

    async def test_spawn_fault_stops_repeating_the_fabricated_clock(self, caplog):
        """The misleading lines fire once, pre-evidence, and never again.

        Exactly once, not never. On the FIRST iteration the loop has not yet
        run a probe, so it cannot know a spawn fault is coming — and it must
        behave identically to a genuine cap at that point, or the control
        tests below would be describing a different code path. What the fix
        removes is the REPETITION: every iteration after the first knows, and
        must not keep asserting a reset time it has no evidence for.

        Driven well past the threshold (7 probes) so a still-repeating loop
        would show 7 occurrences, not 1.
        """
        gate = make_gate(['a'], probe_interval_secs=1, max_probe_interval_secs=8)
        acct = _capped_account(gate, resets_at=None)
        gate._run_probe = AsyncMock(side_effect=_spawn_fault())

        with caplog.at_level(logging.DEBUG, logger='shared.usage_gate'):
            await _drive_loop(gate, acct, max_sleeps=8)

        assert gate._run_probe.await_count == 7, 'test harness drove the wrong depth'
        assert _count_matching(caplog, 'no resets_at') == 1, (
            'the fabricated-1h default must not repeat once the loop knows '
            'the probe cannot spawn'
        )
        assert _count_matching(caplog, 'resets in') == 1, (
            'a synthetic countdown must not be reported for an account whose '
            'reset time is genuinely unknown'
        )

    async def test_spawn_fault_never_writes_a_reset_time(self):
        """`acct.resets_at` stays None — the unknown is preserved as unknown."""
        gate = make_gate(['a'], probe_interval_secs=1, max_probe_interval_secs=8)
        acct = _capped_account(gate, resets_at=None)
        gate._run_probe = AsyncMock(side_effect=_spawn_fault())

        await _drive_loop(gate, acct, max_sleeps=8)

        assert acct.resets_at is None

    async def test_spawn_fault_is_reported_as_infrastructure_and_latches(self, caplog):
        gate = make_gate(['a'], probe_interval_secs=1, max_probe_interval_secs=8)
        acct = _capped_account(gate, resets_at=None)
        gate._run_probe = AsyncMock(side_effect=_spawn_fault())

        with caplog.at_level(logging.DEBUG, logger='shared.usage_gate'):
            await _drive_loop(gate, acct, max_sleeps=_SPAWN_FAULT_THRESHOLD + 1)

        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert errors, 'a host fault must reach ERROR level'
        rendered = ' '.join(r.getMessage() for r in errors)
        assert 'claude' in rendered, rendered
        assert gate.probe_infra_fault is not None, (
            f'{_SPAWN_FAULT_THRESHOLD} consecutive spawn faults must latch'
        )

    async def test_spawn_fault_still_backs_off_instead_of_hot_spinning(self):
        """Retry cadence is preserved, so a missing binary is not a busy loop.

        `probe_count` is still incremented for a faulted attempt — deliberately
        — so the interval still doubles toward the ceiling. What the fix drops
        is only the fabricated `resets_at`, not the backoff.
        """
        gate = make_gate(['a'], probe_interval_secs=1, max_probe_interval_secs=8)
        acct = _capped_account(gate, resets_at=None)
        gate._run_probe = AsyncMock(side_effect=_spawn_fault())

        sleeps = await _drive_loop(gate, acct, max_sleeps=8)

        assert sleeps, 'the loop must sleep between spawn attempts'
        assert all(s > 0 for s in sleeps), sleeps
        assert all(s <= 8 for s in sleeps), (
            f'a sleep exceeded max_probe_interval_secs: {sleeps}'
        )
        assert sleeps[-1] > sleeps[0], f'backoff did not grow: {sleeps}'

    async def test_spawn_fault_leaves_the_account_capped(self):
        """Fail-visible, not fail-open.

        We could not verify capacity, and an invocation would die at the same
        missing binary anyway — so the account stays CAPPED. The honest
        difference from today is that its reset time stays unknown rather than
        invented.
        """
        gate = make_gate(['a'], probe_interval_secs=1, max_probe_interval_secs=8)
        acct = _capped_account(gate, resets_at=None)
        gate._run_probe = AsyncMock(side_effect=_spawn_fault())

        captured: list[bool] = []
        original_sleep = asyncio.sleep

        async def capture_sleep(duration: float) -> None:
            captured.append(acct.capped)
            if len(captured) >= 5:
                acct.capped = False
            await original_sleep(0)

        with patch('asyncio.sleep', side_effect=capture_sleep):
            await asyncio.wait_for(
                gate._account_resume_probe_loop(acct), timeout=5,
            )

        assert all(captured), (
            'the account was uncapped mid-fault: a probe that never ran is '
            'not evidence of capacity'
        )

    async def test_recovery_clears_the_latch_and_then_resumes(self):
        """fault, fault, ran-and-capped, ran-and-free.

        The `False` is the load-bearing one: it clears BOTH the counter and
        the latch even though the account is still capped, because a probe
        that returned at all proves the binary spawns.
        """
        gate = make_gate(['a'], probe_interval_secs=1, max_probe_interval_secs=8)
        acct = _capped_account(gate, resets_at=None)
        gate._run_probe = AsyncMock(
            side_effect=[_spawn_fault(), _spawn_fault(), False, True],
        )

        await _drive_loop(gate, acct, max_sleeps=20)

        assert gate._run_probe.await_count == 4
        assert acct.probe_spawn_failures == 0
        assert gate.probe_infra_fault is None
        assert not acct.capped, 'the final successful probe must still resume'

    # --- the control: a genuine cap must not get noisier ----------------

    async def test_genuine_cap_keeps_every_existing_log_line(self, caplog):
        """The non-fault branch is byte-identical, and this is what says so.

        A fix that made real caps noisier — or quieter — would be a
        regression in the opposite direction, and nothing else in the suite
        would catch it: every other cap test asserts on state, not on the
        operator-facing narration.
        """
        gate = make_gate(['a'], probe_interval_secs=1, max_probe_interval_secs=8)
        acct = _capped_account(gate, resets_at=None)
        gate._run_probe = AsyncMock(side_effect=[False, True])

        with caplog.at_level(logging.DEBUG, logger='shared.usage_gate'):
            await _drive_loop(gate, acct, max_sleeps=20)

        assert _count_matching(caplog, 'no resets_at') >= 1
        assert _count_matching(caplog, 'resets in') >= 1
        assert _count_matching(caplog, 'retrying after backoff') == 1
        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and 'no resets_at' in r.getMessage()
        ]
        assert warnings, 'the 1h-default line must stay a WARNING'

    async def test_genuine_cap_is_not_an_infrastructure_fault(self):
        """A probe that RAN and found the account capped latches nothing."""
        gate = make_gate(['a'], probe_interval_secs=1, max_probe_interval_secs=8)
        acct = _capped_account(gate, resets_at=None)
        gate._run_probe = AsyncMock(side_effect=[False, True])

        await _drive_loop(gate, acct, max_sleeps=20)

        assert gate.probe_infra_fault is None
        assert acct.probe_spawn_failures == 0

    async def test_genuine_cap_still_resumes_and_fires_one_resumed_event(self):
        store = make_mock_cost_store()
        gate = make_gate(
            ['a'], cost_store=store, probe_interval_secs=1, max_probe_interval_secs=8,
        )
        acct = _capped_account(gate, resets_at=None)
        gate._run_probe = AsyncMock(side_effect=[False, True])

        with patch.object(gate, '_write_cost_event', new_callable=AsyncMock) as mock_write:
            await _drive_loop(gate, acct, max_sleeps=20)

        assert not acct.capped
        assert mock_write.await_count == 1
        assert mock_write.call_args[0][1] == 'resumed'


@pytest.mark.asyncio
class TestRunProbeClassifyInvocationConsistency:
    """_run_probe's verdicts must agree with classify_invocation(strict_confirm=False).

    ``test_non_cap_marker_is_not_misread_as_still_capped`` is RED against
    today's _run_probe: it scans CAP_HIT_PREFIXES/NEAR_CAP_PREFIXES directly
    with no notion of NON_CAP_CLI_ERROR_MARKERS, so a message that contains
    both a cap-like prefix and a local-CLI-error marker is misread as "still
    capped" (returns False) today. Once _run_probe is rewired onto
    classify_invocation (step-4), CliLocalError precedence applies uniformly
    and this goes green. The other test pins the already-correct prefix-only
    (no confirm keyword) behavior against the classifier as a regression guard.
    """

    async def _make_probing_gate(self) -> tuple[UsageGate, AccountState]:
        gate = make_gate(['a'])
        acct = gate._accounts[0]
        acct.capped = True
        return gate, acct

    async def test_prefix_only_without_confirm_keyword_agrees_with_classifier(self):
        """DD-2: the probe regime (strict_confirm=False) accepts a bare prefix
        with no CAP_CONFIRM_KEYWORDS keyword — still "capped", still False."""
        gate, acct = await self._make_probing_gate()
        prefix = CAP_HIT_PREFIXES[0]  # e.g. "You've hit your"
        text = f'{prefix} quota'  # deliberately no confirm keyword
        outcome = classify_invocation(
            AgentResult(success=False, output=text), strict_confirm=False,
        )
        assert isinstance(outcome, (CapHit, NearCap))

        proc = _make_mock_proc(returncode=0, stderr=text.encode())
        with patch('asyncio.create_subprocess_exec', return_value=proc):
            result = await gate._run_probe(acct)
        assert result is False

    async def test_non_cap_marker_is_not_misread_as_still_capped(self):
        """reify-3604 applied to _run_probe: a local CLI/usage error occurring
        alongside cap-like text must not be treated as "still capped"."""
        gate, acct = await self._make_probing_gate()
        text = (
            f'{CAP_HIT_PREFIXES[0]} usage limit. Your plan resets in 3h. '
            'permission denied: /tmp/x'
        )
        outcome = classify_invocation(
            AgentResult(success=False, output=text), strict_confirm=False,
        )
        assert isinstance(outcome, CliLocalError)

        proc = _make_mock_proc(returncode=0, stderr=text.encode())
        with patch('asyncio.create_subprocess_exec', return_value=proc):
            result = await gate._run_probe(acct)
        assert result is True, (
            '_run_probe must not misread a local CLI error co-occurring with '
            'cap-like text as "still capped" (CliLocalError precedence, reify-3604)'
        )


# ---------------------------------------------------------------------------
# TestProbeEdgeCases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestProbeEdgeCases:
    """Edge cases in probe loop behavior."""

    async def test_probe_count_100_no_overflow(self):
        """probe_count very high (100) -> no integer overflow, uses ceiling."""
        gate = make_gate(['a'], probe_interval_secs=300, max_probe_interval_secs=1800)
        acct = _capped_account(
            gate,
            resets_at=datetime.now(UTC) + timedelta(hours=10),
            probe_count=100,
        )

        captured_sleep: list[float] = []

        async def capture_and_uncap(duration: float) -> None:
            captured_sleep.append(duration)
            acct.capped = False

        gate._run_probe = AsyncMock(return_value=False)

        with patch('asyncio.sleep', side_effect=capture_and_uncap):
            await asyncio.wait_for(
                gate._account_resume_probe_loop(acct), timeout=5,
            )

        assert captured_sleep[0] == 1800

    async def test_resets_at_exactly_now(self):
        """resets_at exactly equal to now -> remaining=0, sleep_for=0."""
        gate = make_gate(
            ['a'],
            probe_interval_secs=0,
            max_probe_interval_secs=0,
        )
        acct = _capped_account(
            gate,
            resets_at=datetime.now(UTC),
            probe_count=0,
        )

        gate._run_probe = AsyncMock(return_value=True)

        await asyncio.wait_for(
            gate._account_resume_probe_loop(acct), timeout=5,
        )

        assert not acct.capped

    async def test_resets_at_changes_mid_loop(self):
        """resets_at changes mid-loop (external code updates it) -> next iteration uses new value."""
        gate = make_gate(
            ['a'],
            probe_interval_secs=1,
            max_probe_interval_secs=100,
        )
        acct = _capped_account(
            gate,
            resets_at=datetime.now(UTC) + timedelta(hours=1),
            probe_count=0,
        )

        call_count = 0
        captured_sleeps: list[float] = []

        async def capture_sleep(duration: float) -> None:
            nonlocal call_count
            captured_sleeps.append(duration)
            call_count += 1
            if call_count == 1:
                # Simulate external code changing resets_at to something very close
                acct.resets_at = datetime.now(UTC) + timedelta(seconds=0.2)
            elif call_count >= 2:
                acct.capped = False

        gate._run_probe = AsyncMock(return_value=False)

        with patch('asyncio.sleep', side_effect=capture_sleep):
            await asyncio.wait_for(
                gate._account_resume_probe_loop(acct), timeout=5,
            )

        # First sleep used the original resets_at (far future), interval=1
        assert captured_sleeps[0] == pytest.approx(1, abs=0.5)
        # Second sleep should use the new resets_at (very close), so remaining is small
        # interval = 1 * 2^1 = 2, but remaining ~0.2 => sleep_for ~ 0.2
        assert captured_sleeps[1] < 1.0

    async def test_uncapped_account_immediate_return(self):
        """Probe loop called on already-uncapped account -> immediate return."""
        gate = make_gate(
            ['a'],
            probe_interval_secs=0,
            max_probe_interval_secs=0,
        )
        acct = gate._accounts[0]
        acct.capped = False  # NOT capped

        gate._run_probe = AsyncMock(return_value=True)

        await asyncio.wait_for(
            gate._account_resume_probe_loop(acct), timeout=5,
        )

        gate._run_probe.assert_not_awaited()

    async def test_external_uncap_during_probe_execution(self):
        """Account uncapped externally while _run_probe is running.

        If acct.capped becomes False BEFORE _run_probe returns (e.g.
        _refresh_capped_accounts ran concurrently), the loop should still
        handle this gracefully. Since the check is `while acct.capped`
        at loop top and `if not acct.capped` after sleep, but probe runs
        after that check, the probe result still applies.
        """
        gate = make_gate(
            ['a'],
            probe_interval_secs=0,
            max_probe_interval_secs=0,
        )
        acct = _capped_account(
            gate,
            resets_at=datetime.now(UTC) - timedelta(minutes=1),
            probe_count=0,
        )

        async def probe_that_uncaps(a: AccountState) -> bool:
            # Simulate external uncap during probe execution
            # The probe still returns False, but acct is uncapped
            a.capped = False
            return False

        gate._run_probe = AsyncMock(side_effect=probe_that_uncaps)

        await asyncio.wait_for(
            gate._account_resume_probe_loop(acct), timeout=5,
        )

        # Loop should exit on the `while acct.capped` check at next iteration
        assert not acct.capped
