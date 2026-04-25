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

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.config import GitConfig
from orchestrator.harness import Harness


@pytest.fixture
def git_config() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
    )


@pytest.fixture
def harness(tmp_path: Path, git_config: GitConfig) -> Harness:
    """Real Harness with a real EscalationQueue; other internals mocked."""
    config = MagicMock()
    config.git = git_config
    config.project_root = tmp_path
    config.usage_cap.enabled = False
    config.review.enabled = False
    config.orphan_l0_reaper_enabled = False
    config.sandbox.backend = 'auto'

    with (
        patch('orchestrator.harness.McpLifecycle'),
        patch('orchestrator.harness.Scheduler'),
        patch('orchestrator.harness.BriefingAssembler'),
    ):
        h = Harness(config)

    h._escalation_queue = EscalationQueue(tmp_path / 'escalations')
    h._escalation_queue.set_resolve_callback(h._on_escalation_resolved)
    return h


def _make_wip_esc(
    queue: EscalationQueue, task_id: str, *, category: str = 'wip_conflict',
) -> Escalation:
    esc = Escalation(
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
