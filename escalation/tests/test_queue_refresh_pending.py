"""``EscalationQueue.refresh_pending``: the locked refresh of a pending record in the queue root (task 9411)."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from _escalation_seed import seed_escalation

from escalation.models import Escalation
from escalation.queue import EscalationQueue, escalation_id_lock

_SEEDED_UPDATED_AT = '2026-11-01T00:00:00+00:00'


@pytest.fixture
def queue(tmp_path: Path) -> EscalationQueue:
    return EscalationQueue(tmp_path / 'esc')


def _seed_pending(queue: EscalationQueue, *, level: int = 1) -> Escalation:
    return seed_escalation(
        queue, level=level, task_id='9411', category='infra_issue',
        suggested_action='investigate', filing_claimant_run_id='run-a',
        updated_at=_SEEDED_UPDATED_AT,
    )


class TestRefreshPending:
    def test_refreshes_summary_detail_timestamp_and_counts_the_repeat(self, queue):
        seeded = _seed_pending(queue)

        refreshed = queue.refresh_pending(seeded.id, level=1, summary='s2', detail='d2')

        assert refreshed is not None
        assert refreshed.id == seeded.id
        assert refreshed.summary == 's2'
        assert refreshed.detail == 'd2'
        assert refreshed.dedupe_count == seeded.dedupe_count + 1
        assert refreshed.timestamp >= seeded.timestamp
        assert refreshed.level == seeded.level
        assert refreshed.agent_role == seeded.agent_role
        assert refreshed.severity == seeded.severity
        assert refreshed.category == seeded.category
        assert refreshed.suggested_action == seeded.suggested_action
        assert refreshed.filing_claimant_run_id == seeded.filing_claimant_run_id
        assert refreshed.updated_at == seeded.updated_at
        assert queue.get(seeded.id) == refreshed

        again = queue.refresh_pending(seeded.id, level=1, summary='s3', detail='d3')

        assert again is not None
        assert again.dedupe_count == seeded.dedupe_count + 2

    def test_a_resolved_record_is_not_refreshed_or_resurrected(self, queue):
        seeded = _seed_pending(queue)
        queue.resolve(seeded.id, 'fixed', resolved_by='test')

        assert queue.refresh_pending(seeded.id, level=1, summary='s2', detail='d2') is None

        stored = queue.get(seeded.id)
        assert stored is not None
        assert stored.status == 'resolved'
        assert stored.summary == seeded.summary
        assert seeded.id not in [esc.id for esc in queue.get_pending()]

    def test_an_unknown_id_returns_none(self, queue):
        assert queue.refresh_pending('esc-nope-1', level=1, summary='s', detail='d') is None

    def test_a_record_no_longer_at_the_level_is_not_refreshed(self, queue):
        seeded = _seed_pending(queue, level=1)
        queue.park(seeded.id, 'parked', resolved_by='test')

        assert queue.refresh_pending(seeded.id, level=1, summary='s2', detail='d2') is None

        stored = queue.get(seeded.id)
        assert stored is not None
        assert stored.status == 'pending'
        assert stored.level == 2
        assert stored.dedupe_count == 0
        assert stored.summary == seeded.summary


class TestRefreshPendingHoldsTheIdLock:
    @pytest.mark.timeout(60)
    def test_a_refresh_waits_for_the_record_lock(self, queue):
        seeded = _seed_pending(queue)
        refreshed: list[Escalation | None] = []
        worker = threading.Thread(
            target=lambda: refreshed.append(
                queue.refresh_pending(seeded.id, level=1, summary='s2', detail='d2'),
            ),
        )

        with escalation_id_lock(queue.queue_dir, seeded.id):
            worker.start()
            worker.join(timeout=0.5)
            assert worker.is_alive()
            held = queue.get(seeded.id)
            assert held is not None
            assert held.dedupe_count == 0

        worker.join(timeout=30)
        assert not worker.is_alive()
        stored = queue.get(seeded.id)
        assert stored is not None
        assert stored.dedupe_count == 1
        assert refreshed == [stored]


class TestRefreshPendingNotify:
    def test_notify_fires_once_with_the_refreshed_record(self, queue):
        seeded = _seed_pending(queue)
        notified: list[Escalation] = []
        queue.set_notify_callback(notified.append)

        refreshed = queue.refresh_pending(seeded.id, level=1, summary='s2', detail='d2')

        assert notified == [refreshed]

        queue.resolve(seeded.id, 'fixed', resolved_by='test')
        queue.refresh_pending(seeded.id, level=1, summary='s3', detail='d3')

        assert notified == [refreshed]

    @pytest.mark.timeout(60)
    def test_a_notify_callback_that_reenters_the_same_id_does_not_deadlock(self, queue):
        seeded = _seed_pending(queue)

        def reenter(esc: Escalation) -> None:
            queue.stamp_triage(esc.id, triaged_by='cb')

        queue.set_notify_callback(reenter)

        queue.refresh_pending(seeded.id, level=1, summary='s2', detail='d2')

        stored = queue.get(seeded.id)
        assert stored is not None
        assert stored.triaged_by == 'cb'
