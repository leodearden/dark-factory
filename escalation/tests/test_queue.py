"""Tests for EscalationQueue — focusing on dismiss_all_pending()."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from escalation.models import Escalation
from escalation.queue import EscalationQueue


def _make_escalation(esc_id: str, task_id: str = '1', status: str = 'pending') -> Escalation:
    esc = Escalation(
        id=esc_id,
        task_id=task_id,
        agent_role='orchestrator',
        severity='blocking',
        category='task_failure',
        summary='Something failed',
    )
    esc.status = status
    return esc


def _submit_escalation(queue: EscalationQueue, esc: Escalation) -> None:
    """Write an escalation directly, bypassing the callback."""
    queue.submit(esc)


class TestDismissAllPending:
    """EscalationQueue.dismiss_all_pending() bulk-dismisses pending escalations."""

    def test_empty_queue_returns_zero(self, tmp_path: Path):
        """Empty queue: no-op, returns 0."""
        queue = EscalationQueue(tmp_path / 'queue')
        count = queue.dismiss_all_pending('Stale from prior run')
        assert count == 0

    def test_single_pending_dismissed(self, tmp_path: Path):
        """Single pending escalation is dismissed and count returned is 1."""
        queue = EscalationQueue(tmp_path / 'queue')
        esc = _make_escalation('esc-1-1')
        queue.submit(esc)

        count = queue.dismiss_all_pending('Stale from prior run')

        assert count == 1
        updated = queue.get('esc-1-1')
        assert updated is not None
        assert updated.status == 'dismissed'
        assert updated.resolution == 'Stale from prior run'

    def test_multiple_pending_all_dismissed(self, tmp_path: Path):
        """Multiple pending escalations are all dismissed; count matches."""
        queue = EscalationQueue(tmp_path / 'queue')
        for i in range(3):
            queue.submit(_make_escalation(f'esc-{i}-1', task_id=str(i)))

        count = queue.dismiss_all_pending('Orchestrator restarted')

        assert count == 3
        for i in range(3):
            esc = queue.get(f'esc-{i}-1')
            assert esc is not None
            assert esc.status == 'dismissed'

    def test_resolved_escalation_not_touched(self, tmp_path: Path):
        """Already-resolved escalations are not re-dismissed."""
        queue = EscalationQueue(tmp_path / 'queue')

        pending = _make_escalation('esc-1-1')
        queue.submit(pending)

        already_resolved = _make_escalation('esc-2-1')
        queue.submit(already_resolved)
        queue.resolve('esc-2-1', 'Fixed manually', dismiss=False)

        count = queue.dismiss_all_pending('Stale from prior run')

        assert count == 1  # only the pending one

        resolved_esc = queue.get('esc-2-1')
        assert resolved_esc is not None
        assert resolved_esc.status == 'resolved'
        assert resolved_esc.resolution == 'Fixed manually'

    def test_dismissed_escalation_not_touched(self, tmp_path: Path):
        """Already-dismissed escalations are not re-dismissed."""
        queue = EscalationQueue(tmp_path / 'queue')

        pending = _make_escalation('esc-1-1')
        queue.submit(pending)

        already_dismissed = _make_escalation('esc-2-1')
        queue.submit(already_dismissed)
        queue.resolve('esc-2-1', 'User dismissed earlier', dismiss=True)

        count = queue.dismiss_all_pending('Stale from prior run')

        assert count == 1  # only the pending one

        dismissed_esc = queue.get('esc-2-1')
        assert dismissed_esc is not None
        assert dismissed_esc.status == 'dismissed'
        assert dismissed_esc.resolution == 'User dismissed earlier'  # unchanged

    def test_resolution_message_preserved(self, tmp_path: Path):
        """Resolution message is preserved on dismissed escalations."""
        queue = EscalationQueue(tmp_path / 'queue')
        queue.submit(_make_escalation('esc-1-1'))

        msg = 'Auto-dismissed: orchestrator restarted, stale from prior run'
        queue.dismiss_all_pending(msg)

        esc = queue.get('esc-1-1')
        assert esc is not None
        assert esc.resolution == msg

    def test_mixed_statuses_only_pending_dismissed(self, tmp_path: Path):
        """With a mix of pending/resolved/dismissed, only pending ones are dismissed."""
        queue = EscalationQueue(tmp_path / 'queue')

        queue.submit(_make_escalation('esc-1-1'))  # pending
        queue.submit(_make_escalation('esc-2-1'))  # pending

        queue.submit(_make_escalation('esc-3-1'))
        queue.resolve('esc-3-1', 'resolved already', dismiss=False)

        queue.submit(_make_escalation('esc-4-1'))
        queue.resolve('esc-4-1', 'dismissed already', dismiss=True)

        count = queue.dismiss_all_pending('Stale from prior run')

        assert count == 2  # esc-1-1 and esc-2-1

        assert queue.get('esc-1-1').status == 'dismissed'  # type: ignore[union-attr]
        assert queue.get('esc-2-1').status == 'dismissed'  # type: ignore[union-attr]
        assert queue.get('esc-3-1').status == 'resolved'  # type: ignore[union-attr]
        assert queue.get('esc-4-1').status == 'dismissed'  # type: ignore[union-attr]
        assert queue.get('esc-4-1').resolution == 'dismissed already'  # type: ignore[union-attr]


class TestDismissAllPendingResilience:
    """EscalationQueue.dismiss_all_pending() is resilient to per-item resolve() failures."""

    def test_resolve_failure_does_not_abort_loop(self, tmp_path: Path):
        """If resolve() raises on one item, the remaining items are still dismissed."""
        queue = EscalationQueue(tmp_path / 'queue')
        queue.submit(_make_escalation('esc-1-1', task_id='1'))
        queue.submit(_make_escalation('esc-2-1', task_id='2'))
        queue.submit(_make_escalation('esc-3-1', task_id='3'))

        # Patch resolve() so it raises OSError for esc-2-1, succeeds for others
        original_resolve = queue.resolve

        def patched_resolve(esc_id: str, resolution: str, dismiss: bool = False):
            if esc_id == 'esc-2-1':
                raise OSError('disk full')
            return original_resolve(esc_id, resolution, dismiss=dismiss)

        with patch.object(queue, 'resolve', side_effect=patched_resolve):
            count = queue.dismiss_all_pending('Stale from prior run')

        # Only 2 of 3 succeeded (esc-2-1 failed)
        assert count == 2

    def test_resolve_failure_count_reflects_successes_only(self, tmp_path: Path):
        """Returned count only reflects successfully dismissed escalations."""
        queue = EscalationQueue(tmp_path / 'queue')
        queue.submit(_make_escalation('esc-1-1', task_id='1'))
        queue.submit(_make_escalation('esc-2-1', task_id='2'))

        original_resolve = queue.resolve

        def patched_resolve(esc_id: str, resolution: str, dismiss: bool = False):
            if esc_id == 'esc-1-1':
                raise OSError('permission denied')
            return original_resolve(esc_id, resolution, dismiss=dismiss)

        with patch.object(queue, 'resolve', side_effect=patched_resolve):
            count = queue.dismiss_all_pending('Stale from prior run')

        assert count == 1  # only esc-2-1 succeeded

    def test_resolve_failure_does_not_propagate(self, tmp_path: Path):
        """An OSError from resolve() does not propagate to the caller."""
        queue = EscalationQueue(tmp_path / 'queue')
        queue.submit(_make_escalation('esc-1-1', task_id='1'))

        def always_fail(esc_id: str, resolution: str, dismiss: bool = False):
            raise OSError('disk full')

        with patch.object(queue, 'resolve', side_effect=always_fail):
            # Must not raise
            count = queue.dismiss_all_pending('Stale from prior run')

        assert count == 0
