"""Tests for orchestrator.systemd_inspect (task 2119).

Hoists coverage for the task-2091-hardened `systemctl --user show` inspector
out of test_deterministic_runner.py::TestInspectUnitTimeoutHardening and
test_harness_deterministic_recon_sweep.py::TestReconInspectUnit into one
place, now that the implementation itself lives in one module and both
DeterministicRunner and harness delegate to it.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from shared.deploy_state import VerifyBaseline

from orchestrator.systemd_inspect import (
    _INSPECT_TIMEOUT_SECS,
    _deterministic_deploy_health_verdict,
    _empty_baseline_fresh,
    inspect_systemd_unit,
)


def test_inspect_timeout_secs_constant() -> None:
    """Module-constant contract relied on by DeterministicRunner's __init__ default."""
    assert _INSPECT_TIMEOUT_SECS == 10.0


class TestInspectSystemdUnit:
    """inspect_systemd_unit: task-2091 timeout hardening + output parsing."""

    @pytest.mark.asyncio
    async def test_wedged_call_returns_mainpid_zero_sentinel(self, caplog) -> None:
        """A wedged systemctl call must time out and return the MainPID=0
        sentinel directly, killing the stuck subprocess."""
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        never_resolves = asyncio.Event()
        mock_proc.communicate = AsyncMock(side_effect=never_resolves.wait)
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock(return_value=None)

        with (
            patch('asyncio.create_subprocess_exec', AsyncMock(return_value=mock_proc)),
            caplog.at_level(logging.WARNING, logger='orchestrator.systemd_inspect'),
        ):
            # Hang tripwire: if the fix regresses, fail loudly instead of
            # stalling the suite.
            result = await asyncio.wait_for(
                inspect_systemd_unit(
                    'orchestrator-reify.service',
                    timeout_secs=0.05,
                    reap_grace_secs=0.05,
                ),
                timeout=5,
            )

        assert result == {
            'MainPID': 0,
            'ActiveState': '',
            'ActiveEnterTimestamp': '',
            'ActiveEnterTimestampMonotonic': 0,
        }
        mock_proc.kill.assert_called_once()
        assert 'MainPID=0 sentinel' in caplog.text

    @pytest.mark.asyncio
    async def test_parses_healthy_unit_output_and_coerces_ints(self) -> None:
        stdout = (
            b'MainPID=1234\n'
            b'ActiveState=active\n'
            b'ActiveEnterTimestamp=Sat 2026-07-04 12:00:00 UTC\n'
            b'ActiveEnterTimestampMonotonic=123456789\n'
        )
        proc = AsyncMock()
        proc.communicate = AsyncMock(return_value=(stdout, b''))
        with patch('asyncio.create_subprocess_exec', AsyncMock(return_value=proc)):
            result = await inspect_systemd_unit('fused-memory.service', timeout_secs=5.0)

        assert result['MainPID'] == 1234
        assert isinstance(result['MainPID'], int)
        assert result['ActiveState'] == 'active'
        assert result['ActiveEnterTimestampMonotonic'] == 123456789
        assert isinstance(result['ActiveEnterTimestampMonotonic'], int)

    @pytest.mark.asyncio
    async def test_malformed_numeric_field_coerces_to_zero_not_raise(self) -> None:
        """A non-numeric MainPID (unexpected systemctl output shape) must
        coerce to 0 rather than raising."""
        stdout = b'MainPID=not-a-number\nActiveState=failed\n'
        proc = AsyncMock()
        proc.communicate = AsyncMock(return_value=(stdout, b''))
        with patch('asyncio.create_subprocess_exec', AsyncMock(return_value=proc)):
            result = await inspect_systemd_unit('unknown.service', timeout_secs=5.0)

        assert result['MainPID'] == 0
        assert isinstance(result['MainPID'], int)


class TestEmptyBaselineFresh:
    """_empty_baseline_fresh (task 2611) — the shared pure predicate used by
    both _deterministic_deploy_health_verdict below (with an empty baseline)
    and proc_supervision.RestartPlan._execute_cross_unit_blocking. Directly
    unit-tests the allowlist contract so both call sites can rely on a single
    source of truth instead of re-verifying it themselves (amendment: a
    reviewer noted the blocklist-vs-allowlist distinction and the None
    ActiveState hole matter enough to lock in explicitly)."""

    def test_fresh_when_monotonic_advanced_and_active(self) -> None:
        assert _empty_baseline_fresh(0, 500, 'active') is True

    def test_fresh_when_monotonic_advanced_and_inactive(self) -> None:
        """A plain Type=oneshot service (no RemainAfterExit) settles back to
        'inactive' after a successful run — this must count as fresh."""
        assert _empty_baseline_fresh(0, 500, 'inactive') is True

    def test_not_fresh_when_monotonic_not_advanced(self) -> None:
        assert _empty_baseline_fresh(500, 500, 'active') is False

    def test_not_fresh_when_monotonic_regressed(self) -> None:
        assert _empty_baseline_fresh(500, 100, 'active') is False

    def test_not_fresh_when_failed(self) -> None:
        assert _empty_baseline_fresh(0, 500, 'failed') is False

    def test_not_fresh_when_wedged_sentinel(self) -> None:
        assert _empty_baseline_fresh(0, 500, '') is False

    def test_not_fresh_when_active_state_missing(self) -> None:
        """A missing ActiveState key (surfaced as None by dict.get) must be
        treated conservatively as unconfirmed, not waved through by an
        accidental `None not in (...)` blocklist gap."""
        assert _empty_baseline_fresh(0, 500, None) is False

    @pytest.mark.parametrize('transient_state', ['activating', 'deactivating', 'reloading'])
    def test_not_fresh_when_active_state_transient(self, transient_state: str) -> None:
        """A mid-transition ActiveState must not be reported fresh even if
        the activation clock has already advanced (e.g. a race mid-restart)
        — only the terminal 'active'/'inactive' states are trusted."""
        assert _empty_baseline_fresh(0, 500, transient_state) is False

    def test_not_fresh_when_live_monotonic_not_int(self) -> None:
        assert _empty_baseline_fresh(0, None, 'active') is False


class TestDeterministicDeployHealthVerdict:
    """_deterministic_deploy_health_verdict pure classifier (relocated from harness.py)."""

    def test_healthy_when_pid_positive_and_active(self) -> None:
        result = _deterministic_deploy_health_verdict({'MainPID': 1234, 'ActiveState': 'active'})
        assert result == 'healthy'

    def test_unconfirmed_when_pid_zero(self) -> None:
        result = _deterministic_deploy_health_verdict({'MainPID': 0, 'ActiveState': 'active'})
        assert result == 'unconfirmed'

    def test_unconfirmed_when_not_active(self) -> None:
        result = _deterministic_deploy_health_verdict({'MainPID': 1234, 'ActiveState': 'failed'})
        assert result == 'unconfirmed'

    def test_unconfirmed_when_inspect_result_is_none(self) -> None:
        assert _deterministic_deploy_health_verdict(None) == 'unconfirmed'

    def test_unconfirmed_when_inspect_result_is_empty(self) -> None:
        assert _deterministic_deploy_health_verdict({}) == 'unconfirmed'

    # --- ζ/task 2240 DS-3: freshness branch (verify_baseline present) ------

    def test_healthy_with_baseline_when_monotonic_advanced_and_pid_positive(self) -> None:
        result = {
            'MainPID': 1234, 'ActiveState': 'active',
            'ActiveEnterTimestampMonotonic': 500,
        }
        baseline = {'active_enter_timestamp_monotonic': 100, 'main_pid': 999}
        assert _deterministic_deploy_health_verdict(result, verify_baseline=baseline) == 'healthy'

    def test_healthy_with_baseline_accepts_verifybaseline_instance(self) -> None:
        result = {
            'MainPID': 1234, 'ActiveState': 'active',
            'ActiveEnterTimestampMonotonic': 500,
        }
        baseline = VerifyBaseline(active_enter_timestamp_monotonic=100, main_pid=999)
        assert _deterministic_deploy_health_verdict(result, verify_baseline=baseline) == 'healthy'

    def test_healthy_with_baseline_even_when_active_state_not_active(self) -> None:
        """The freshness branch does not require ActiveState=='active' — only
        MainPID>0 and monotonic advancement past baseline (real freshness
        supersedes the liveness proxy)."""
        result = {
            'MainPID': 1234, 'ActiveState': 'deactivating',
            'ActiveEnterTimestampMonotonic': 500,
        }
        baseline = {'active_enter_timestamp_monotonic': 100, 'main_pid': 999}
        assert _deterministic_deploy_health_verdict(result, verify_baseline=baseline) == 'healthy'

    def test_unconfirmed_with_baseline_when_monotonic_not_advanced(self) -> None:
        """Stale — the always-on-unit ambiguity the liveness-only CAVEAT
        described: the unit is active, but the live monotonic equals the
        pre-deploy baseline, so this restart demonstrably did NOT happen."""
        result = {
            'MainPID': 1234, 'ActiveState': 'active',
            'ActiveEnterTimestampMonotonic': 100,
        }
        baseline = {'active_enter_timestamp_monotonic': 100, 'main_pid': 999}
        assert _deterministic_deploy_health_verdict(result, verify_baseline=baseline) == 'unconfirmed'

    def test_unconfirmed_with_baseline_when_monotonic_regressed(self) -> None:
        result = {
            'MainPID': 1234, 'ActiveState': 'active',
            'ActiveEnterTimestampMonotonic': 50,
        }
        baseline = {'active_enter_timestamp_monotonic': 100, 'main_pid': 999}
        assert _deterministic_deploy_health_verdict(result, verify_baseline=baseline) == 'unconfirmed'

    def test_unconfirmed_with_baseline_when_pid_zero(self) -> None:
        result = {
            'MainPID': 0, 'ActiveState': 'active',
            'ActiveEnterTimestampMonotonic': 500,
        }
        baseline = {'active_enter_timestamp_monotonic': 100, 'main_pid': 999}
        assert _deterministic_deploy_health_verdict(result, verify_baseline=baseline) == 'unconfirmed'

    # --- task 2611 (esc-2584-1): empty-baseline (install-fresh oneshot/timer)
    # freshness — a .timer unit / Type=oneshot service never reports a live
    # MainPID, even once genuinely active, so an EMPTY baseline (main_pid=0)
    # must not be gated by the unconditional pid>0 check above. -------------

    def test_healthy_with_empty_baseline_when_monotonic_advanced(self) -> None:
        result = {
            'MainPID': 0, 'ActiveState': 'active',
            'ActiveEnterTimestampMonotonic': 500,
        }
        baseline = {'active_enter_timestamp_monotonic': 0, 'main_pid': 0}
        assert _deterministic_deploy_health_verdict(result, verify_baseline=baseline) == 'healthy'

    def test_healthy_with_empty_baseline_accepts_verifybaseline_instance(self) -> None:
        """VerifyBaseline-instance variant of the mapping case above — locks
        both the mapping and the dataclass main_pid-reading paths."""
        result = {
            'MainPID': 0, 'ActiveState': 'active',
            'ActiveEnterTimestampMonotonic': 500,
        }
        baseline = VerifyBaseline(active_enter_timestamp_monotonic=0, main_pid=0)
        assert _deterministic_deploy_health_verdict(result, verify_baseline=baseline) == 'healthy'

    def test_unconfirmed_with_empty_baseline_when_activated_then_failed(self) -> None:
        """An activated-then-FAILED oneshot (monotonic advanced, but
        ActiveState ended up 'failed') must not be reported healthy —
        the only new hole the empty-baseline branch must close."""
        result = {
            'MainPID': 0, 'ActiveState': 'failed',
            'ActiveEnterTimestampMonotonic': 500,
        }
        baseline = {'active_enter_timestamp_monotonic': 0, 'main_pid': 0}
        assert _deterministic_deploy_health_verdict(result, verify_baseline=baseline) == 'unconfirmed'

    def test_unconfirmed_with_empty_baseline_when_no_activation(self) -> None:
        """No activation at all (monotonic unchanged from the empty
        baseline) must still be unconfirmed — an empty baseline is not a
        free pass."""
        result = {
            'MainPID': 0, 'ActiveState': 'inactive',
            'ActiveEnterTimestampMonotonic': 0,
        }
        baseline = {'active_enter_timestamp_monotonic': 0, 'main_pid': 0}
        assert _deterministic_deploy_health_verdict(result, verify_baseline=baseline) == 'unconfirmed'

    def test_unconfirmed_with_empty_baseline_when_active_state_transient(self) -> None:
        """Amendment (task 2611 review): a mid-transition ActiveState
        ('activating') with an advanced monotonic must not read 'healthy' —
        wiring-level guard that the classifier really delegates to
        _empty_baseline_fresh's allowlist rather than a wider blocklist."""
        result = {
            'MainPID': 0, 'ActiveState': 'activating',
            'ActiveEnterTimestampMonotonic': 500,
        }
        baseline = {'active_enter_timestamp_monotonic': 0, 'main_pid': 0}
        assert _deterministic_deploy_health_verdict(result, verify_baseline=baseline) == 'unconfirmed'

    def test_unconfirmed_with_empty_baseline_when_active_state_missing(self) -> None:
        """Amendment (task 2611 review): a malformed/partial inspect result
        missing the ActiveState key entirely must read 'unconfirmed', not
        slip through a `None not in (...)` blocklist gap."""
        result = {
            'MainPID': 0,
            'ActiveEnterTimestampMonotonic': 500,
        }
        baseline = {'active_enter_timestamp_monotonic': 0, 'main_pid': 0}
        assert _deterministic_deploy_health_verdict(result, verify_baseline=baseline) == 'unconfirmed'

    def test_no_baseline_preserves_liveness_only_verdict(self) -> None:
        """verify_baseline=None (explicit) is identical to omitting it — the
        exact pre-existing liveness-only branch (backward compat)."""
        result = {
            'MainPID': 1234, 'ActiveState': 'active',
            'ActiveEnterTimestampMonotonic': 5,
        }
        assert _deterministic_deploy_health_verdict(result, verify_baseline=None) == 'healthy'
        assert _deterministic_deploy_health_verdict(result) == 'healthy'
