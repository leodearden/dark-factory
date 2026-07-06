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

from orchestrator.systemd_inspect import (
    _INSPECT_TIMEOUT_SECS,
    _deterministic_deploy_health_verdict,
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

        with patch('asyncio.create_subprocess_exec', AsyncMock(return_value=mock_proc)):
            with caplog.at_level(logging.WARNING, logger='orchestrator.systemd_inspect'):
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
