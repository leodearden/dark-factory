"""Tests for Harness._write_merge_heartbeat (task 2395, α of the fleet-redeploy PRD).

Every orchestrator unit writes a tiny JSON heartbeat to a fleet-common
directory (see ``orchestrator.fleet_heartbeat``) from its merge-heartbeat
background service, whatever the dispatch loop is doing (task 5344).
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
  - A disk write failure is swallowed (fail-open), never propagated, and
    its WARNING carries no ORCH_UNIT hint when the unit name is valid.
  - A blank ORCH_UNIT is refused without raising and without creating
    anything, and the WARNING names ORCH_UNIT's value and expected shape,
    both where the drain participant refuses it first and where
    write_heartbeat does.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator import fleet_heartbeat
from orchestrator.fleet_drain import DrainParticipant, DrainPass
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


def _fail_open_warning(caplog) -> str:
    """The fail-open WARNING's own message, without the exc_info traceback.

    ``caplog.text`` includes the traceback, and the refusing ValueError
    already quotes ``orchestrator-dark-factory.service`` as its example, so
    asserting on it could not catch the hint going missing.
    """
    return next(m for m in caplog.messages if m.startswith('merge heartbeat write failed'))


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
            'drain': None,
            'verifies_in_flight': [],
            'deployed_sha': None,
        }
        assert isinstance(payload['ts_epoch'], float)

    async def test_heartbeat_carries_the_deployed_sha(
        self, harness: Harness, tmp_path: Path, monkeypatch
    ):
        """Task 6565 D1: the unit's deployed sha and its flags reach the heartbeat."""
        from orchestrator.deployed_sha import DeployedSha

        fleet_dir = tmp_path / 'fleet'
        monkeypatch.setenv('ORCH_UNIT', 'orchestrator-reify.service')
        monkeypatch.setenv('ORCH_FLEET_DIR', str(fleet_dir))
        worker = MagicMock()
        worker.snapshot.return_value = {'depth': 0}
        harness._merge_worker = worker
        deployed = DeployedSha(sha='9' * 40, indeterminate=False, dirty_wire_path=True)
        harness._deployed_sha = deployed

        await harness._write_merge_heartbeat()

        payload = json.loads((fleet_dir / 'orchestrator-reify.service.json').read_text())
        assert payload['deployed_sha'] == deployed.to_wire()

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
        """write_heartbeat raising must be swallowed (logged), not propagated.

        The unit name is valid, so the WARNING must not send the operator to
        ORCH_UNIT for an unrelated failure.
        """
        monkeypatch.setenv('ORCH_UNIT', 'orchestrator-reify.service')

        with patch(
            'orchestrator.harness.write_heartbeat',
            side_effect=RuntimeError('disk full'),
        ), caplog.at_level(logging.WARNING):
            await harness._write_merge_heartbeat()  # must not raise

        assert 'ORCH_UNIT' not in _fail_open_warning(caplog)

    async def test_empty_unit_writes_nothing_and_is_logged_not_raised(
        self, harness: Harness, tmp_path: Path, monkeypatch, caplog
    ):
        """An unset ORCH_UNIT is refused LOUDLY but never stops the run loop.

        Runs the real drain participant and the real write path (no patch).
        The drain participant runs first, so today the refusal comes from
        ``fleet_drain.drain_request_path`` and ``write_heartbeat`` is never
        reached; ``test_blank_unit_past_the_drain_participant_is_refused_by_write_heartbeat``
        covers that refusal at this seam.  Four properties are asserted together:

        1. it does NOT raise — harness.py's ``except Exception`` fail-open is a
           documented invariant, and propagating would trade a corrupt heartbeat
           for a dead orchestrator, which is strictly worse;
        2. the ``merge heartbeat write failed`` WARNING fires, which is where
           the loudness actually lands at the production call site;
        3. that WARNING's own message, not the traceback logged with it, names
           ORCH_UNIT's value and the expected shape of a unit name; and
        4. nothing is created in the fleet dir, whichever check refuses first.

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

        warning = _fail_open_warning(caplog)
        assert "ORCH_UNIT was ''" in warning
        assert 'orchestrator-dark-factory.service' in warning
        assert not fleet_dir.exists()

    async def test_blank_unit_past_the_drain_participant_is_refused_by_write_heartbeat(
        self, harness: Harness, tmp_path: Path, monkeypatch, caplog
    ):
        """The REAL ``write_heartbeat`` refuses a blank unit before any filesystem work.

        The drain participant is stubbed to return a reading, so the blank
        unit reaches ``write_heartbeat`` itself: the producer-level half of
        ``test_fleet_heartbeat.py``'s ``TestMalformedUnitIsRefused``.  The fleet
        dir is asserted absent UNCONDITIONALLY: an ``if fleet_dir.exists():``
        emptiness check would still pass if the refusal were ever moved after
        ``safe_io.atomic_write_text(..., mkdir=True)`` created the parent.
        """
        fleet_dir = tmp_path / 'fleet'
        monkeypatch.delenv('ORCH_UNIT', raising=False)
        monkeypatch.setenv('ORCH_FLEET_DIR', str(fleet_dir))
        reading = DrainPass(drain=None, verifies_in_flight=[], depth=0)

        with patch.object(
            DrainParticipant, 'on_heartbeat_pass', AsyncMock(return_value=reading),
        ), patch(
            'orchestrator.harness.write_heartbeat',
            wraps=fleet_heartbeat.write_heartbeat,
        ) as write, caplog.at_level(logging.WARNING):
            await harness._write_merge_heartbeat()  # must not raise

        assert write.call_count == 1
        assert "ORCH_UNIT was ''" in _fail_open_warning(caplog)
        assert not fleet_dir.exists(), (
            f'{fleet_dir} exists, so the refusal ran AFTER filesystem work had '
            'already begun. write_heartbeat must reject a blank unit name before '
            'safe_io.atomic_write_text(..., mkdir=True) can create the parent.'
        )
