"""Exhaustive tests for UsageGate._account_resume_probe_loop and _run_probe."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.config_models import AccountConfig, UsageCapConfig
from shared.usage_gate import (
    CAP_HIT_PREFIXES,
    NEAR_CAP_PREFIXES,
    AccountState,
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

    async def test_timeout_returns_false(self):
        """Timeout (asyncio.wait_for raises TimeoutError) -> returns False."""
        gate, acct = await self._make_probing_gate()
        proc = _make_mock_proc(returncode=0, stdout=b'ok')

        with (
            patch('asyncio.create_subprocess_exec', return_value=proc),
            patch('shared.usage_gate.asyncio.wait_for', side_effect=TimeoutError),
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
        """General exception (e.g., FileNotFoundError) -> returns False."""
        gate, acct = await self._make_probing_gate()

        with patch(
            'asyncio.create_subprocess_exec',
            side_effect=FileNotFoundError('claude not found'),
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

        gate._probe_config_dir.write_credentials = track_write

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
