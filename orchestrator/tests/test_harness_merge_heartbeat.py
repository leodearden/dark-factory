"""Tests for Harness._write_merge_heartbeat (task 2395, α of the fleet-redeploy PRD).

Every orchestrator unit writes a tiny JSON heartbeat to a fleet-common
directory (see ``orchestrator.fleet_heartbeat``) on each run-loop tick.
This method gathers the live state — ORCH_UNIT, ``_merge_pipeline_idle()``
(the authoritative drain-gate truth source, task 1973 U2), and the
``queue_empty``/``depth`` diagnostics — and delegates the on-disk contract
to the pure ``fleet_heartbeat`` module (pinned separately in
``test_fleet_heartbeat.py``).

Covers:
  - An idle tick (worker snapshot depth=0, empty queue) writes
    merge_idle:True.
  - A busy tick (worker snapshot depth>0) writes merge_idle:False with the
    diagnostic depth preserved.
  - A disk write failure is swallowed (fail-open), never propagated.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from orchestrator.harness import Harness

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def harness(tmp_path: Path, mock_orch_config):
    """Minimal harness with mocked heavy deps (mirrors test_harness_service_restart.py:34-68)."""
    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        h = Harness(mock_orch_config)

    h.scheduler = MagicMock()
    h.scheduler._dispatched = set()
    h.event_store = MagicMock()
    return h


# ---------------------------------------------------------------------------
# Harness._write_merge_heartbeat
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestWriteMergeHeartbeat:
    """_write_merge_heartbeat() gathers live state and writes via fleet_heartbeat."""

    async def test_idle_tick_writes_heartbeat_with_merge_idle_true(
        self, harness: Harness, tmp_path: Path, monkeypatch
    ):
        """(a) snapshot depth=0, empty queue -> merge_idle:True, depth:0, queue_empty:True."""
        fleet_dir = tmp_path / 'fleet'
        monkeypatch.setenv('ORCH_UNIT', 'orchestrator-reify.service')
        monkeypatch.setenv('ORCH_FLEET_DIR', str(fleet_dir))
        worker = MagicMock()
        worker.snapshot.return_value = {'depth': 0}
        harness._merge_worker = worker
        assert harness._merge_queue.empty()

        await harness._write_merge_heartbeat()

        path = fleet_dir / 'orchestrator-reify.service.json'
        assert path.exists()
        payload = json.loads(path.read_text())
        assert payload == {
            'unit': 'orchestrator-reify.service',
            'merge_idle': True,
            'depth': 0,
            'queue_empty': True,
            'ts_epoch': payload['ts_epoch'],
        }
        assert isinstance(payload['ts_epoch'], float)

    async def test_busy_tick_writes_heartbeat_with_merge_idle_false(
        self, harness: Harness, tmp_path: Path, monkeypatch
    ):
        """(b) snapshot depth=2 -> merge_idle:False, depth:2."""
        fleet_dir = tmp_path / 'fleet'
        monkeypatch.setenv('ORCH_UNIT', 'orchestrator-reify.service')
        monkeypatch.setenv('ORCH_FLEET_DIR', str(fleet_dir))
        worker = MagicMock()
        worker.snapshot.return_value = {'depth': 2}
        harness._merge_worker = worker

        await harness._write_merge_heartbeat()

        path = fleet_dir / 'orchestrator-reify.service.json'
        payload = json.loads(path.read_text())
        assert payload['unit'] == 'orchestrator-reify.service'
        assert payload['merge_idle'] is False
        assert payload['depth'] == 2
        assert isinstance(payload['ts_epoch'], float)


# ---------------------------------------------------------------------------
# Harness._write_merge_heartbeat — fail-open
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestWriteMergeHeartbeatFailOpen:
    """A heartbeat write must never crash or stall the run loop."""

    async def test_disk_write_failure_does_not_propagate(
        self, harness: Harness, monkeypatch, caplog
    ):
        """write_heartbeat raising must be swallowed (logged), not propagated."""
        monkeypatch.setenv('ORCH_UNIT', 'orchestrator-reify.service')

        with patch(
            'orchestrator.harness.write_heartbeat',
            side_effect=RuntimeError('disk full'),
        ), caplog.at_level(logging.WARNING):
            await harness._write_merge_heartbeat()  # must not raise

        assert 'merge heartbeat write failed' in caplog.text

    async def test_empty_unit_writes_nothing_and_is_logged_not_raised(
        self, harness: Harness, tmp_path: Path, monkeypatch, caplog
    ):
        """An unset ORCH_UNIT is refused LOUDLY but never stops the run loop.

        The producer-level half of the behaviour pinned at unit level by
        ``test_fleet_heartbeat.py``'s ``TestMalformedUnitIsRefused``, exercised
        through the REAL ``write_heartbeat`` (no patch) so the two halves cannot
        drift.  All three properties matter and are asserted together:

        1. it does NOT raise — harness.py's ``except Exception`` fail-open is a
           documented invariant, and propagating would trade a corrupt heartbeat
           for a dead orchestrator, which is strictly worse;
        2. the existing ``merge heartbeat write failed`` WARNING fires, which is
           where the loudness actually lands at the production call site; and
        3. nothing is created — the substantive half, which holds regardless of
           what the caller does with the exception.  Asserted UNCONDITIONALLY on
           the fleet dir itself: ``write_heartbeat`` refuses before any
           filesystem work, so the directory must not exist AT ALL.  An
           ``if fleet_dir.exists():``-guarded emptiness check would be dead code
           here and would still pass if the refusal were ever moved after
           ``safe_io.atomic_write_text(..., mkdir=True)`` created the parent —
           precisely the regression worth catching.  One line also subsumes the
           retired ``unknown-unit.json`` name and any ``.json.tmp`` residue.

        ``delenv`` is REQUIRED, not defensive: ORCH_UNIT is AMBIENT-SET in an
        orchestrator-dispatched session (measured: ``orchestrator-dark-factory.service``,
        inherited from the systemd unit), so a test relying on it being absent
        would pass vacuously on a developer box and write a REAL unit's
        heartbeat name on the factory.
        """
        fleet_dir = tmp_path / 'fleet'
        monkeypatch.delenv('ORCH_UNIT', raising=False)
        monkeypatch.setenv('ORCH_FLEET_DIR', str(fleet_dir))
        worker = MagicMock()
        worker.snapshot.return_value = {'depth': 0}
        harness._merge_worker = worker

        with caplog.at_level(logging.WARNING):
            await harness._write_merge_heartbeat()  # must not raise

        assert 'merge heartbeat write failed' in caplog.text
        assert not fleet_dir.exists(), (
            f'{fleet_dir} exists, so the refusal ran AFTER filesystem work had '
            'already begun. write_heartbeat must reject a blank unit name before '
            'safe_io.atomic_write_text(..., mkdir=True) can create the parent.'
        )
