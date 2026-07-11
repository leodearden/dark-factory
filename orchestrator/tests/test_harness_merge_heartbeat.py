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
"""

from __future__ import annotations

import json
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
