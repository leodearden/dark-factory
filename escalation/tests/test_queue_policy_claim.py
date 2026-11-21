"""The policy-class claim on an escalation record and the queue write that sets it.

``EscalationQueue.claim_policy_class`` is the sole writer of ``policy_class``,
``claimed_at`` and ``claimed_by`` (task 8422; PRD
``plans/decision-stream-prd.md`` §3 decision 4). It is an annotation write in
the shape of ``stamp_triage``: pending records in the queue root only, no
status or ``updated_at`` change, and a refusal callback evaluated on the record
as read under the per-id lock.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from _escalation_seed import seed_escalation

from escalation.models import Escalation
from escalation.queue import ClaimResult, EscalationQueue

_EXECUTOR = 'orchestrator-escalation-executor'


def _never_refuse(_record: Escalation) -> None:
    return None


def _seed_pending(queue: EscalationQueue) -> Escalation:
    return seed_escalation(
        queue, level=2, task_id='8422', category='provenance_unattributed',
    )


class TestClaimFieldsZeroMigration:
    def test_legacy_record_has_no_claim(self) -> None:
        legacy = {
            'id': 'esc-1-1', 'task_id': '1', 'agent_role': 'implementer',
            'severity': 'blocking', 'category': 'task_failure', 'summary': 's',
        }
        record = Escalation.from_dict(legacy)
        assert record.policy_class is None
        assert record.claimed_at is None
        assert record.claimed_by is None

    def test_claim_fields_round_trip(self) -> None:
        record = Escalation(
            id='esc-1-1', task_id='1', agent_role='implementer', severity='blocking',
            category='task_failure', summary='s', policy_class='found_on_main',
            claimed_at='2026-11-04T10:00:00+00:00', claimed_by=_EXECUTOR,
        )
        again = Escalation.from_dict(record.to_dict())
        assert (again.policy_class, again.claimed_at, again.claimed_by) == (
            'found_on_main', '2026-11-04T10:00:00+00:00', _EXECUTOR,
        )


class TestClaimPolicyClass:
    def test_claim_writes_the_three_fields_and_nothing_else(self, tmp_path: Path) -> None:
        queue = EscalationQueue(tmp_path / 'esc')
        seeded = _seed_pending(queue)
        queue.stamp_triage(seeded.id, triaged_by='watcher', triage_note='predicate P')
        before = queue.get(seeded.id)
        assert before is not None

        result = queue.claim_policy_class(
            seeded.id, policy_class='landing_provenance', claimed_by=_EXECUTOR,
            refuse=_never_refuse,
        )

        assert result.refusal is None
        assert result.record is not None
        assert result.record.policy_class == 'landing_provenance'
        assert result.record.claimed_by == _EXECUTOR
        assert result.record.claimed_at is not None
        claimed_at = datetime.fromisoformat(result.record.claimed_at)
        assert claimed_at.tzinfo is not None
        assert abs(datetime.now(UTC) - claimed_at) < timedelta(seconds=30)

        stored = queue.get(seeded.id)
        assert stored is not None
        assert (stored.policy_class, stored.claimed_at, stored.claimed_by) == (
            'landing_provenance', result.record.claimed_at, _EXECUTOR,
        )
        for name in ('status', 'level', 'updated_at', 'triaged_at', 'triage_note'):
            assert getattr(stored, name) == getattr(before, name), name

    def test_refusal_is_evaluated_on_the_freshly_read_record(self, tmp_path: Path) -> None:
        queue = EscalationQueue(tmp_path / 'esc')
        seeded = _seed_pending(queue)
        sentinel = object()
        seen: list[Escalation] = []

        def refuse_if_claimed(record: Escalation) -> object | None:
            seen.append(record)
            return sentinel if record.claimed_at is not None else None

        first = queue.claim_policy_class(
            seeded.id, policy_class='landing_provenance', claimed_by=_EXECUTOR,
            refuse=refuse_if_claimed,
        )
        assert first.record is not None
        second = queue.claim_policy_class(
            seeded.id, policy_class='found_on_main', claimed_by=_EXECUTOR,
            refuse=refuse_if_claimed,
        )

        assert second == ClaimResult(record=None, refusal=sentinel)
        assert seen[-1].claimed_at == first.record.claimed_at
        stored = queue.get(seeded.id)
        assert stored is not None
        assert stored.policy_class == 'landing_provenance'
        assert stored.claimed_at == first.record.claimed_at

    def test_missing_id_is_not_found(self, tmp_path: Path) -> None:
        queue = EscalationQueue(tmp_path / 'esc')
        result = queue.claim_policy_class(
            'esc-does-not-exist-1', policy_class='found_on_main', claimed_by=_EXECUTOR,
            refuse=_never_refuse,
        )
        assert result == ClaimResult(record=None, refusal=None)

    def test_archived_record_is_refused_and_not_resurrected(self, tmp_path: Path) -> None:
        queue = EscalationQueue(tmp_path / 'esc')
        seeded = _seed_pending(queue)
        queue.resolve(seeded.id, 'Fixed')

        result = queue.claim_policy_class(
            seeded.id, policy_class='found_on_main', claimed_by=_EXECUTOR,
            refuse=_never_refuse,
        )

        assert result == ClaimResult(record=None, refusal=None)
        assert not (queue.queue_dir / f'{seeded.id}.json').exists()
        archived = queue.get(seeded.id)
        assert archived is not None
        assert archived.policy_class is None
