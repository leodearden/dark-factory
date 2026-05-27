"""Tests for Harness._on_escalation_resolved halt-owner predicate.

These tests target the phantom-L1 bug: previously, ANY `wip_conflict` resolve
triggered `unhalt_wip()` — so an unrelated escalation's resolution released
the halt while the real blocker stayed pending. The fix keys the un-halt on
the escalation ID that owns the halt (MergeWorker._halt_owner_esc_id),
registered by the workflow handler right after submitting the escalation.

These are regression guards — the test setup deliberately reproduces the
original failure shape (two wip_conflict escalations; only one owns the halt).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.harness import Harness
from orchestrator.merge_queue import SpeculativeMergeWorker


@pytest.fixture
def harness(tmp_path: Path, mock_orch_config) -> Harness:
    """Real Harness with a real EscalationQueue; other internals mocked."""
    mock_orch_config.orphan_l0_reaper_enabled = False

    with (
        patch('orchestrator.harness.McpLifecycle'),
        patch('orchestrator.harness.Scheduler'),
        patch('orchestrator.harness.BriefingAssembler'),
    ):
        h = Harness(mock_orch_config)

    h._escalation_queue = EscalationQueue(tmp_path / 'escalations')
    h._escalation_queue.set_resolve_callback(h._on_escalation_resolved)
    return h


def _make_wip_esc(
    queue: EscalationQueue,
    task_id: str,
    *,
    category: str = 'wip_conflict',
    timestamp: str | None = None,
) -> Escalation:
    kwargs: dict = dict(
        id=queue.make_id(task_id),
        task_id=task_id,
        agent_role='orchestrator',
        severity='blocking',
        category=category,
        summary='test wip escalation',
        detail='detail',
        suggested_action='manual_intervention',
        level=1,
    )
    if timestamp is not None:
        kwargs['timestamp'] = timestamp
    esc = Escalation(**kwargs)
    queue.submit(esc)
    return esc


class _FakeMergeWorker:
    """Minimal halt-owner state machine — same contract as MergeWorker."""

    def __init__(self) -> None:
        self._halted = False
        self._owner: str | None = None

    @property
    def is_wip_halted(self) -> bool:
        return self._halted

    @property
    def halt_owner_esc_id(self) -> str | None:
        return self._owner

    def halt_for_wip(self, reason: str) -> None:
        self._halted = True
        self._owner = None

    def set_halt_owner(self, esc_id: str) -> None:
        assert self._owner is None
        self._owner = esc_id

    def is_halt_owner(self, esc_id: str) -> bool:
        return self._owner is not None and self._owner == esc_id

    def unhalt_wip(self) -> None:
        self._halted = False
        self._owner = None


class TestHaltOwnerUnhaltPredicate:
    """Harness._on_escalation_resolved un-halts only for the owning escalation."""

    def test_resolving_non_owner_does_not_unhalt(self, harness: Harness):
        """Regression guard for esc-1888-57: resolving esc-B must NOT un-halt
        when esc-A owns the halt. Prior code matched on category alone and
        released the halt prematurely.
        """
        worker = _FakeMergeWorker()
        harness._merge_worker = worker  # type: ignore[assignment]
        queue = harness._escalation_queue
        assert queue is not None

        esc_a = _make_wip_esc(queue, '1888')  # will own the halt
        esc_b = _make_wip_esc(queue, '9999')  # unrelated

        worker.halt_for_wip('pop_conflict_no_advance')
        worker.set_halt_owner(esc_a.id)
        assert worker.is_wip_halted

        # Resolve the non-owner — un-halt must NOT fire.
        queue.resolve(esc_b.id, 'unrelated cleanup', resolved_by='test')
        assert worker.is_wip_halted, (
            'Resolving a non-owning wip_conflict must not release the halt'
        )
        assert worker.is_halt_owner(esc_a.id), (
            'Owner pointer must still point at esc_a'
        )

        # Resolve the owner — un-halt fires.
        queue.resolve(esc_a.id, 'user cleaned up', resolved_by='test')
        assert not worker.is_wip_halted
        assert not worker.is_halt_owner(esc_a.id)

    def test_resolving_unmerged_state_owner_unhalts(self, harness: Harness):
        """Category is irrelevant — any owning category un-halts. unmerged_state
        owns the halt in the new handler path; resolving it must release.
        """
        worker = _FakeMergeWorker()
        harness._merge_worker = worker  # type: ignore[assignment]
        queue = harness._escalation_queue
        assert queue is not None

        esc = _make_wip_esc(queue, '55', category='unmerged_state')

        worker.halt_for_wip('unmerged_state')
        worker.set_halt_owner(esc.id)

        queue.resolve(esc.id, 'user cleared UU markers', resolved_by='test')
        assert not worker.is_wip_halted

    def test_resolving_when_no_owner_is_safe(self, harness: Harness):
        """Gap window: halt set but owner not yet registered — resolve
        should leave the halt in place (no premature un-halt, no crash).
        """
        worker = _FakeMergeWorker()
        harness._merge_worker = worker  # type: ignore[assignment]
        queue = harness._escalation_queue
        assert queue is not None

        worker.halt_for_wip('simulated gap window')
        assert worker.is_wip_halted
        assert worker._owner is None

        esc = _make_wip_esc(queue, '42')
        # NO set_halt_owner — simulate the gap between halt_for_wip and
        # the workflow's set_halt_owner registration.
        queue.resolve(esc.id, 'fake', resolved_by='test')

        assert worker.is_wip_halted, (
            'With no owner registered, resolve must leave the halt alone'
        )

    def test_resolving_when_not_halted_does_nothing(self, harness: Harness):
        """No halt in effect: resolve callback must be a no-op on the worker."""
        worker = _FakeMergeWorker()
        harness._merge_worker = worker  # type: ignore[assignment]
        queue = harness._escalation_queue
        assert queue is not None

        esc = _make_wip_esc(queue, '7')
        queue.resolve(esc.id, 'any reason', resolved_by='test')

        assert not worker.is_wip_halted


class TestRehydrateMergeHalt:
    """Harness._rehydrate_merge_halt restores halt+owner from preserved L1s on restart."""

    def test_rehydrate_restores_halt_and_owner_from_preserved_wip_conflict(
        self, harness: Harness
    ):
        """Core regression: after restart with a preserved wip_conflict L1,
        the halt and owner must be restored.  Simulates restart by assigning
        a fresh (un-halted, owner=None) _FakeMergeWorker and calling
        _rehydrate_merge_halt() directly.
        """
        worker = _FakeMergeWorker()
        harness._merge_worker = worker  # type: ignore[assignment]
        queue = harness._escalation_queue
        assert queue is not None

        esc = _make_wip_esc(queue, '1111')  # level-1, wip_conflict (preserved L1)

        result = harness._rehydrate_merge_halt()

        assert worker.is_wip_halted is True
        assert worker.halt_owner_esc_id == esc.id
        assert result == esc.id

    def test_rehydrate_restores_from_unmerged_state(self, harness: Harness):
        """Single preserved level-1 unmerged_state L1 restores halt+owner."""
        worker = _FakeMergeWorker()
        harness._merge_worker = worker  # type: ignore[assignment]
        queue = harness._escalation_queue
        assert queue is not None

        esc = _make_wip_esc(queue, '2222', category='unmerged_state')

        result = harness._rehydrate_merge_halt()

        assert worker.is_wip_halted is True
        assert worker.halt_owner_esc_id == esc.id
        assert result == esc.id

    @pytest.mark.parametrize('setup', [
        'empty_queue',
        'wrong_category',
        'wrong_level',
    ])
    def test_rehydrate_noop_when_no_relevant_l1(
        self, harness: Harness, setup: str
    ):
        """No qualifying L1 means no-op: worker stays un-halted, result is None."""
        worker = _FakeMergeWorker()
        harness._merge_worker = worker  # type: ignore[assignment]
        queue = harness._escalation_queue
        assert queue is not None

        if setup == 'empty_queue':
            pass  # nothing submitted
        elif setup == 'wrong_category':
            # level-1 but irrelevant category
            _make_wip_esc(queue, '3333', category='infra_issue')
        elif setup == 'wrong_level':
            # wip_conflict but level-0 (not a preserved L1)
            esc = Escalation(
                id=queue.make_id('4444'),
                task_id='4444',
                agent_role='orchestrator',
                severity='blocking',
                category='wip_conflict',
                summary='test wip escalation',
                detail='detail',
                suggested_action='manual_intervention',
                level=0,
            )
            queue.submit(esc)

        result = harness._rehydrate_merge_halt()

        assert worker.is_wip_halted is False
        assert worker.halt_owner_esc_id is None
        assert result is None

    def test_rehydrate_picks_most_recent_l1(self, harness: Harness):
        """When two preserved L1s qualify, the most recent by timestamp wins."""
        worker = _FakeMergeWorker()
        harness._merge_worker = worker  # type: ignore[assignment]
        queue = harness._escalation_queue
        assert queue is not None

        esc_older = _make_wip_esc(
            queue, '5555', timestamp='2026-05-27T10:00:00+00:00'
        )
        esc_newer = _make_wip_esc(
            queue, '6666', timestamp='2026-05-27T11:00:00+00:00'
        )

        result = harness._rehydrate_merge_halt()

        # The most recent escalation must own the halt
        assert worker.halt_owner_esc_id == esc_newer.id
        assert result == esc_newer.id
        _ = esc_older  # referenced for clarity

    def test_rehydrate_logs_warning_for_multiple_l1s(
        self, harness: Harness, caplog: pytest.LogCaptureFixture
    ):
        """When multiple qualifying L1s exist, a warning pinning the premature-
        resume risk must be logged alongside the normal halt-restore warning.
        """
        worker = _FakeMergeWorker()
        harness._merge_worker = worker  # type: ignore[assignment]
        queue = harness._escalation_queue
        assert queue is not None

        _make_wip_esc(queue, '7001', timestamp='2026-05-27T09:00:00+00:00')
        _make_wip_esc(queue, '7002', timestamp='2026-05-27T10:00:00+00:00')
        _make_wip_esc(queue, '7003', timestamp='2026-05-27T11:00:00+00:00')

        with caplog.at_level(logging.WARNING, logger='orchestrator.harness'):
            harness._rehydrate_merge_halt()

        # The multi-L1 advisory warning must appear in the log output
        advisory_msgs = [
            r for r in caplog.records
            if 'qualifying L1s found' in r.message
        ]
        assert advisory_msgs, (
            'Expected a warning about multiple qualifying L1s; got none. '
            f'Log records: {[r.message for r in caplog.records]}'
        )
        # Sanity: the halt is set and owner is the most recent
        assert worker.is_wip_halted
        assert worker.halt_owner_esc_id is not None

    def test_rehydrate_against_real_speculative_merge_worker(
        self, harness: Harness
    ):
        """Integration guard: _rehydrate_merge_halt works with the real
        SpeculativeMergeWorker, ensuring _FakeMergeWorker can't drift from
        the real implementation's halt-owner contract.
        """
        real_worker = SpeculativeMergeWorker(
            git_ops=MagicMock(),
            queue=asyncio.Queue(),
        )
        harness._merge_worker = real_worker  # type: ignore[assignment]
        queue = harness._escalation_queue
        assert queue is not None

        # Initial state must match the post-restart shape: un-halted, no owner
        assert real_worker.is_wip_halted is False
        assert real_worker.halt_owner_esc_id is None

        esc = _make_wip_esc(queue, '8888')

        result = harness._rehydrate_merge_halt()

        assert real_worker.is_wip_halted is True, (
            'Real SpeculativeMergeWorker must be halted after rehydration'
        )
        assert real_worker.halt_owner_esc_id == esc.id, (
            'Real SpeculativeMergeWorker owner must be the preserved L1 id'
        )
        assert result == esc.id


class TestForceUnhaltMergeQueue:
    """Operator escape hatch for orphan halts (no escalation owns the halt).

    Regression for the 2026-05-04 know-live incident: workflow soft-cancel
    raced merge submission, so the queue halted on a request whose
    workflow had already exited.  No escalation existed to resolve — the
    only recovery was an orchestrator restart.  ``force_unhalt_merge_queue``
    closes that gap while preserving the legitimate
    ``resolve_issue → _on_escalation_resolved`` un-halt path.
    """

    def test_force_unhalt_when_no_owner_succeeds(self, harness: Harness):
        worker = _FakeMergeWorker()
        harness._merge_worker = worker  # type: ignore[assignment]

        worker.halt_for_wip('orphan halt — no owner registered')
        assert worker.is_wip_halted
        assert worker.halt_owner_esc_id is None

        result = harness.force_unhalt_merge_queue('orphan recovery')
        assert result['unhalted'] is True
        assert result['prior_owner'] is None
        assert result['reason'] == 'orphan recovery'
        assert not worker.is_wip_halted

    def test_force_unhalt_when_owner_resolved_succeeds(
        self, harness: Harness,
    ):
        """If the owner escalation is already resolved, force-unhalt may proceed.

        Idempotent against the legitimate path: resolving the owner
        normally fires ``_on_escalation_resolved`` which un-halts; if that
        ran (worker not halted), force-unhalt reports ``queue not halted``.
        If for some reason it didn't run, force-unhalt completes the job.
        """
        worker = _FakeMergeWorker()
        harness._merge_worker = worker  # type: ignore[assignment]
        queue = harness._escalation_queue
        assert queue is not None

        esc = _make_wip_esc(queue, '321')
        worker.halt_for_wip('halt with owner')
        worker.set_halt_owner(esc.id)
        # Resolve the owning escalation directly via the queue — this
        # fires the resolve callback which un-halts the worker.
        queue.resolve(esc.id, 'cleanup done', resolved_by='test')
        # Worker should have been unhalted by the callback path.
        assert not worker.is_wip_halted

        # Calling force-unhalt now is a no-op.
        result = harness.force_unhalt_merge_queue('belt and braces')
        assert result['unhalted'] is False
        assert result.get('reason') == 'queue not halted'

    def test_force_unhalt_when_active_owner_refused(self, harness: Harness):
        """If the halt has an active owning escalation, refuse force-unhalt."""
        worker = _FakeMergeWorker()
        harness._merge_worker = worker  # type: ignore[assignment]
        queue = harness._escalation_queue
        assert queue is not None

        esc = _make_wip_esc(queue, '4242')
        worker.halt_for_wip('legit halt with active owner')
        worker.set_halt_owner(esc.id)

        result = harness.force_unhalt_merge_queue('try anyway')
        assert result['unhalted'] is False
        assert result.get('owner_esc_id') == esc.id
        assert 'resolve_issue' in result.get('error', '')
        # Worker still halted, owner still set.
        assert worker.is_wip_halted
        assert worker.is_halt_owner(esc.id)

    def test_force_unhalt_when_not_halted_noop(self, harness: Harness):
        worker = _FakeMergeWorker()
        harness._merge_worker = worker  # type: ignore[assignment]

        result = harness.force_unhalt_merge_queue('nothing to do')
        assert result['unhalted'] is False
        assert result.get('reason') == 'queue not halted'

    def test_force_unhalt_when_no_merge_worker(self, harness: Harness):
        """Bare harness (no merge worker wired) reports the wiring problem."""
        harness._merge_worker = None
        result = harness.force_unhalt_merge_queue('test')
        assert result['unhalted'] is False
        assert 'merge worker' in result.get('error', '').lower()
