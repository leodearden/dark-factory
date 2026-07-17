"""Tests for ProvenanceConflictSink (task 2677 step-3/step-4).

Covers the shared sink that converts a ``StaleEvidenceRejection`` into a
deduped, born-at-L2 ``provenance_conflict`` escalation plus an in-memory
"terminal-for-this-tick" memo (``should_skip``). See
plans/found-on-main-provenance-integrity-prd.md §Contract + §R5.
"""

from __future__ import annotations

import json

from escalation.queue import EscalationQueue


def _record(sink, *, task_id='42', evidence_commit='abc', gate_source='test'):
    return sink.record(
        task_id=task_id,
        evidence_commit=evidence_commit,
        evidence_committed_at='2026-07-10T00:00:00+00:00',
        reopen_at='2026-07-15T00:00:00+00:00',
        agent_id='claude-recon-x',
        gate_source=gate_source,
    )


class TestProvenanceConflictSinkRecord:
    def test_record_files_one_pending_l2_escalation(self, tmp_path):
        from orchestrator.provenance_conflict import (
            ProvenanceConflictSink,
            stale_evidence_fingerprint,
        )

        queue = EscalationQueue(tmp_path)
        sink = ProvenanceConflictSink(escalation_queue=queue)

        _record(sink)

        pending = queue.get_by_task('42', status='pending')
        assert len(pending) == 1, f'Expected 1 pending escalation, got {len(pending)}'
        esc = pending[0]
        assert esc.category == 'provenance_conflict'
        assert esc.level == 2
        assert esc.severity == 'urgent'
        assert esc.agent_role.startswith('orchestrator-')
        assert esc.dedupe_fingerprint == stale_evidence_fingerprint('42', 'abc')

        detail = json.loads(esc.detail)
        assert detail['task_id'] == '42'
        assert detail['evidence_commit'] == 'abc'
        assert detail['evidence_committed_at'] == '2026-07-10T00:00:00+00:00'
        assert detail['reopen_at'] == '2026-07-15T00:00:00+00:00'
        assert detail['gate_source'] == 'test'

    def test_repeat_rejection_same_sha_folds_to_dedupe_count(self, tmp_path):
        from orchestrator.provenance_conflict import ProvenanceConflictSink

        queue = EscalationQueue(tmp_path)
        sink = ProvenanceConflictSink(escalation_queue=queue)

        _record(sink)
        _record(sink)

        pending = queue.get_by_task('42', status='pending')
        assert len(pending) == 1, (
            f'Expected exactly one pending record after a repeat rejection, '
            f'got {len(pending)}'
        )
        assert pending[0].dedupe_count == 1

    def test_different_sha_files_a_separate_record(self, tmp_path):
        from orchestrator.provenance_conflict import ProvenanceConflictSink

        queue = EscalationQueue(tmp_path)
        sink = ProvenanceConflictSink(escalation_queue=queue)

        _record(sink, evidence_commit='abc')
        _record(sink, evidence_commit='def')

        pending = queue.get_by_task('42', status='pending')
        assert len(pending) == 2, (
            f'A distinct evidence_commit must file a separate record, '
            f'got {len(pending)}'
        )

    def test_none_queue_sink_is_noop_but_memoizes(self):
        """Queue-less sink: record() must not raise and returns None."""
        from orchestrator.provenance_conflict import ProvenanceConflictSink

        sink = ProvenanceConflictSink(escalation_queue=None)

        result = _record(sink)

        assert result is None


class TestProvenanceConflictSinkShouldSkip:
    def test_should_skip_true_immediately_after_record(self, tmp_path):
        from orchestrator.provenance_conflict import ProvenanceConflictSink

        queue = EscalationQueue(tmp_path)
        sink = ProvenanceConflictSink(escalation_queue=queue)

        _record(sink)

        assert sink.should_skip('42') is True

    def test_should_skip_false_after_escalation_resolved(self, tmp_path):
        from orchestrator.provenance_conflict import ProvenanceConflictSink

        queue = EscalationQueue(tmp_path)
        sink = ProvenanceConflictSink(escalation_queue=queue)

        escalation_id = _record(sink)
        assert escalation_id is not None
        queue.resolve(escalation_id, 'operator resolved the provenance conflict')

        assert sink.should_skip('42') is False

    def test_should_skip_false_when_reopen_at_differs(self, tmp_path):
        from orchestrator.provenance_conflict import ProvenanceConflictSink

        queue = EscalationQueue(tmp_path)
        sink = ProvenanceConflictSink(escalation_queue=queue)

        _record(sink)

        assert sink.should_skip('42', reopen_at='2026-07-16T00:00:00+00:00') is False

    def test_should_skip_false_for_unknown_task(self, tmp_path):
        from orchestrator.provenance_conflict import ProvenanceConflictSink

        queue = EscalationQueue(tmp_path)
        sink = ProvenanceConflictSink(escalation_queue=queue)

        assert sink.should_skip('does-not-exist') is False


class TestProvenanceConflictSinkRecordFromRejection:
    def test_record_from_rejection_adapts_stale_evidence_rejection(self, tmp_path):
        from orchestrator.provenance_conflict import ProvenanceConflictSink
        from orchestrator.scheduler import StaleEvidenceRejection

        queue = EscalationQueue(tmp_path)
        sink = ProvenanceConflictSink(escalation_queue=queue)

        exc = StaleEvidenceRejection(
            task_id='42',
            evidence_commit='abc',
            evidence_committed_at='2026-07-10T00:00:00+00:00',
            reopen_at='2026-07-15T00:00:00+00:00',
            agent_id='claude-recon-x',
            raw="success=False payload={'error': 'done_evidence_stale'}",
        )

        escalation_id = sink.record_from_rejection(exc, gate_source='dispatch-gate')
        assert escalation_id is not None

        esc = queue.get(escalation_id)
        assert esc is not None
        detail = json.loads(esc.detail)
        assert detail['gate_source'] == 'dispatch-gate'
        assert detail['evidence_commit'] == 'abc'
        assert sink.should_skip('42') is True
