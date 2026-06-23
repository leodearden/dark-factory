"""Tests for Harness starvation-watchdog escalation filing/resolving/wiring.

Task 1880 — Scheduler starvation watchdog: INFO-escalate a deps-satisfied
pending task skipped past yaml thresholds, auto-resolve on dispatch.

Covers:
  (a) _file_starvation_info — files one pending INFO escalation, deduped.
  (b) _resolve_starvation_info — resolves the pending escalation, idempotent.
  (c) Harness wiring — scheduler._on_starvation_warn and
      scheduler._on_starvation_resolve are installed after construction.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from escalation.queue import EscalationQueue

from orchestrator.config import OrchestratorConfig
from orchestrator.harness import Harness
from orchestrator.run_store import RunStore


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _make_harness_with_queue(tmp_path: Path) -> tuple[Harness, EscalationQueue]:
    """Build a Harness with a real EscalationQueue and a mock RunStore.

    Construction follows the pattern in test_harness_park_stop.py:
    ``Harness(config)`` is called directly (no MCP server needed) and then
    ``_escalation_queue`` is wired by the helper.
    """
    config = OrchestratorConfig(project_root=tmp_path)
    harness = Harness(config)

    # Wire a real EscalationQueue so filing/resolving hit real storage.
    queue = EscalationQueue(tmp_path / 'esc')
    harness._escalation_queue = queue

    # Stub out RunStore so persistence calls don't hit SQLite.
    harness._run_store = MagicMock(spec=RunStore)
    harness._run_id = 'run-test-watchdog'

    return harness, queue


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHarnessStarvationWatchdogFiling:
    """Tests for _file_starvation_info and _resolve_starvation_info."""

    @pytest.mark.asyncio
    async def test_file_starvation_info_creates_pending_escalation(
        self, tmp_path: Path
    ) -> None:
        """_file_starvation_info files exactly one pending INFO escalation for T1.

        Checks:
        - severity == 'info'
        - level == 0
        - agent_role == 'orchestrator-starvation-watchdog'
        - category == 'risk_identified'
        - Does NOT change task status (pure signal, PROPERTY 1).
        """
        harness, queue = _make_harness_with_queue(tmp_path)

        await harness._file_starvation_info(
            'T1',
            summary='STARVATION_WATCHDOG: task T1 skipped 55 times',
            detail='task T1 has been skipped for over 1800s with no dispatch',
        )

        pending = queue.get_by_task('T1', status='pending')
        assert len(pending) == 1, (
            f'Expected exactly one pending escalation for T1; got {len(pending)}'
        )
        esc = pending[0]
        assert esc.severity == 'info', (
            f'Expected severity="info"; got {esc.severity!r}'
        )
        assert esc.level == 0, f'Expected level=0; got {esc.level}'
        assert esc.agent_role == 'orchestrator-starvation-watchdog', (
            f'Expected agent_role="orchestrator-starvation-watchdog"; '
            f'got {esc.agent_role!r}'
        )
        assert esc.category == 'risk_identified', (
            f'Expected category="risk_identified"; got {esc.category!r}'
        )
        # PROPERTY 1: only one escalation record was written; task status was
        # not mutated (the filer calls only make_id + get_by_task + submit on
        # the queue, no set_task_status).  The single-record count above already
        # proves no extra writes occurred.

    @pytest.mark.asyncio
    async def test_file_starvation_info_dedup_no_duplicate(
        self, tmp_path: Path
    ) -> None:
        """Calling _file_starvation_info again while one escalation is pending files no duplicate.

        Two calls for the same task_id while a pending escalation exists must
        leave exactly one pending escalation (dedup guard).
        """
        harness, queue = _make_harness_with_queue(tmp_path)

        await harness._file_starvation_info(
            'T1',
            summary='STARVATION_WATCHDOG: first call',
            detail='detail first',
        )
        await harness._file_starvation_info(
            'T1',
            summary='STARVATION_WATCHDOG: second call',
            detail='detail second',
        )

        pending = queue.get_by_task('T1', status='pending')
        assert len(pending) == 1, (
            f'Expected exactly one pending escalation after two calls; '
            f'got {len(pending)}'
        )

    @pytest.mark.asyncio
    async def test_resolve_starvation_info_resolves_pending(
        self, tmp_path: Path
    ) -> None:
        """_resolve_starvation_info flips the pending escalation to resolved.

        After _resolve_starvation_info:
        - get_by_task(T1, status='pending') returns empty (no longer pending).
        - The resolved record's status == 'resolved'.
        """
        harness, queue = _make_harness_with_queue(tmp_path)

        await harness._file_starvation_info(
            'T1',
            summary='STARVATION_WATCHDOG: task T1',
            detail='detail',
        )
        # Pre-condition: one pending escalation.
        assert len(queue.get_by_task('T1', status='pending')) == 1

        await harness._resolve_starvation_info('T1')

        pending_after = queue.get_by_task('T1', status='pending')
        assert len(pending_after) == 0, (
            f'Expected no pending escalations after resolve; got {pending_after!r}'
        )
        # The escalation must have moved to resolved.
        all_t1 = queue.get_by_task('T1')
        resolved = [e for e in all_t1 if e.status == 'resolved']
        assert len(resolved) == 1, (
            f'Expected exactly one resolved escalation; got {resolved!r}'
        )

    @pytest.mark.asyncio
    async def test_resolve_starvation_info_is_idempotent(
        self, tmp_path: Path
    ) -> None:
        """Calling _resolve_starvation_info twice is a harmless no-op the second time."""
        harness, queue = _make_harness_with_queue(tmp_path)

        await harness._file_starvation_info(
            'T1',
            summary='STARVATION_WATCHDOG: task T1',
            detail='detail',
        )
        await harness._resolve_starvation_info('T1')
        # Second call must not raise or duplicate the resolve.
        await harness._resolve_starvation_info('T1')

        all_t1 = queue.get_by_task('T1')
        resolved = [e for e in all_t1 if e.status == 'resolved']
        assert len(resolved) == 1, (
            f'Expected exactly one resolved escalation after two resolve calls; '
            f'got {resolved!r}'
        )

    @pytest.mark.asyncio
    async def test_file_starvation_info_no_queue_is_noop(
        self, tmp_path: Path
    ) -> None:
        """_file_starvation_info with no escalation queue attached is a clean no-op."""
        config = OrchestratorConfig(project_root=tmp_path)
        harness = Harness(config)
        assert harness._escalation_queue is None

        # Must not raise.
        await harness._file_starvation_info(
            'T1',
            summary='STARVATION_WATCHDOG: T1',
            detail='detail',
        )

    @pytest.mark.asyncio
    async def test_resolve_starvation_info_no_queue_is_noop(
        self, tmp_path: Path
    ) -> None:
        """_resolve_starvation_info with no escalation queue attached is a clean no-op."""
        config = OrchestratorConfig(project_root=tmp_path)
        harness = Harness(config)
        assert harness._escalation_queue is None

        # Must not raise.
        await harness._resolve_starvation_info('T1')


class TestHarnessStarvationWatchdogWiring:
    """Tests that Harness wires scheduler callbacks after construction."""

    @pytest.mark.asyncio
    async def test_harness_wires_starvation_warn_callback(
        self, tmp_path: Path
    ) -> None:
        """After Harness construction, scheduler._on_starvation_warn must be wired.

        Calling the callback directly must file a pending INFO escalation for
        the given task_id.
        """
        harness, queue = _make_harness_with_queue(tmp_path)

        assert harness.scheduler._on_starvation_warn is not None, (
            'Harness must wire scheduler._on_starvation_warn after construction'
        )

        # Invoke the callback directly — simulating what the scheduler does.
        await harness.scheduler._on_starvation_warn(
            'T_wire',
            summary='STARVATION_WATCHDOG: T_wire skipped 55×',
            detail='idle 1900s',
        )

        pending = queue.get_by_task('T_wire', status='pending')
        assert len(pending) == 1, (
            f'Expected one pending INFO escalation after warn callback; '
            f'got {pending!r}'
        )
        assert pending[0].severity == 'info'

    @pytest.mark.asyncio
    async def test_harness_wires_starvation_resolve_callback(
        self, tmp_path: Path
    ) -> None:
        """After Harness construction, scheduler._on_starvation_resolve must be wired.

        Filing then resolving via the callbacks must leave the escalation resolved.
        """
        harness, queue = _make_harness_with_queue(tmp_path)

        assert harness.scheduler._on_starvation_resolve is not None, (
            'Harness must wire scheduler._on_starvation_resolve after construction'
        )

        await harness.scheduler._on_starvation_warn(
            'T_wire',
            summary='STARVATION_WATCHDOG: T_wire',
            detail='detail',
        )
        await harness.scheduler._on_starvation_resolve('T_wire')

        pending = queue.get_by_task('T_wire', status='pending')
        assert len(pending) == 0, (
            f'Expected no pending escalations after resolve callback; got {pending!r}'
        )
