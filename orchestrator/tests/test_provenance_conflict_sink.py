"""Tests for ProvenanceConflictSink (task 2677 step-3/step-4).

Covers the shared sink that converts a ``StaleEvidenceRejection`` into a
deduped, born-at-L2 ``provenance_conflict`` escalation plus an in-memory
"terminal-for-this-tick" memo (``should_skip``). See
plans/found-on-main-provenance-integrity-prd.md §Contract + §R5.
"""

from __future__ import annotations

import json
import logging
from typing import Any

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

    def test_none_queue_record_logs_warning_for_visibility(self, caplog):
        """reviewer_comprehensive amendment (task 2677): recording before the
        queue is late-bound memoizes escalation_id=None, which gates
        should_skip until a queue is bound (see
        test_none_queue_memo_self_heals_once_queue_is_bound for the
        self-heal once that happens). That interim window must not be a
        SILENT condition — record() logs a WARNING naming the task so the
        risk is observable even though no escalation can be filed yet.
        """
        from orchestrator.provenance_conflict import ProvenanceConflictSink

        sink = ProvenanceConflictSink(escalation_queue=None)

        with caplog.at_level(logging.WARNING, logger='orchestrator.provenance_conflict'):
            _record(sink, task_id='42', evidence_commit='abc')

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any('42' in r.getMessage() for r in warnings), (
            f'Expected a WARNING naming the unbound-queue task; '
            f'got: {[r.getMessage() for r in warnings]!r}'
        )

    def test_none_queue_memo_self_heals_once_queue_is_bound(self, tmp_path):
        """reviewer_comprehensive amendment (task 2677,
        robustness_silent_degradation): a memo recorded with no queue bound
        must NOT gate the task forever. Once a queue is later bound (the
        harness's late-bind-by-reference), should_skip must invalidate the
        stale None-escalation_id memo so the next caller retries the write
        instead of being silently gated for the rest of the process
        lifetime with no escalation ever filed.
        """
        from orchestrator.provenance_conflict import ProvenanceConflictSink

        sink = ProvenanceConflictSink(escalation_queue=None)
        _record(sink)
        assert sink.should_skip('42') is True, (
            'must gate while the queue remains unbound'
        )

        sink.escalation_queue = EscalationQueue(tmp_path)

        assert sink.should_skip('42') is False, (
            'a None-escalation_id memo must self-heal the instant a queue '
            'is bound, so the next attempt retries and files a real '
            'escalation'
        )


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

    def test_should_skip_prunes_memo_once_escalation_resolved(self, tmp_path):
        """reviewer_comprehensive amendment (task 2677, resource_management):
        self._memo was otherwise never pruned — a resolved conflict's entry
        would linger in the dict for the rest of the process lifetime,
        growing monotonically with the number of distinct tasks that ever
        hit a stale-evidence conflict. should_skip must evict the memo
        entry the moment it observes the escalation is no longer pending.
        """
        from orchestrator.provenance_conflict import ProvenanceConflictSink

        queue = EscalationQueue(tmp_path)
        sink = ProvenanceConflictSink(escalation_queue=queue)

        escalation_id = _record(sink)
        assert escalation_id is not None
        queue.resolve(escalation_id, 'operator resolved the provenance conflict')

        assert sink.should_skip('42') is False
        assert '42' not in sink._memo, (
            'a resolved conflict must not leave a permanent memo entry behind'
        )

    def test_should_skip_false_when_reopen_at_differs(self, tmp_path):
        from orchestrator.provenance_conflict import ProvenanceConflictSink

        queue = EscalationQueue(tmp_path)
        sink = ProvenanceConflictSink(escalation_queue=queue)

        _record(sink)

        assert sink.should_skip('42', reopen_at='2026-07-16T00:00:00+00:00') is False

    def test_should_skip_prunes_memo_when_reopen_at_differs(self, tmp_path):
        """reviewer_comprehensive amendment (task 2677, resource_management):
        a reopen_at mismatch proves the memo entry stale (the task was
        reopened again since the conflict was recorded). It must be pruned
        immediately rather than left for the next record() to overwrite —
        the common self-heal case is that the retried write, now carrying
        the fresh reopen_at, succeeds outright, so no subsequent record()
        call ever arrives to overwrite a left-in-place entry and it would
        otherwise linger in self._memo for the rest of the process
        lifetime.
        """
        from orchestrator.provenance_conflict import ProvenanceConflictSink

        queue = EscalationQueue(tmp_path)
        sink = ProvenanceConflictSink(escalation_queue=queue)

        _record(sink)

        assert sink.should_skip('42', reopen_at='2026-07-16T00:00:00+00:00') is False
        assert '42' not in sink._memo, (
            'a reopen_at mismatch must not leave a permanent memo entry behind'
        )

    def test_should_skip_true_when_reopen_at_differs_only_in_iso8601_formatting(
        self, tmp_path,
    ):
        """reviewer_comprehensive amendment (task 2677, correctness_robustness):
        the memo'd reopen_at (sourced from the fused-memory interceptor's
        rejection payload) and a caller-supplied reopen_at (e.g. harness.py's
        live metadata.get('reopen_at') read) are stamped from the same
        underlying metadata.reopen_at field and expected to be byte-identical
        in practice — but should_skip must not be defeated by incidental
        ISO-8601 formatting drift between the two read paths. The fixture
        memoizes '2026-07-15T00:00:00+00:00'; a 'Z'-suffixed spelling of the
        identical instant must still be treated as a match.
        """
        from orchestrator.provenance_conflict import ProvenanceConflictSink

        queue = EscalationQueue(tmp_path)
        sink = ProvenanceConflictSink(escalation_queue=queue)

        _record(sink)

        assert sink.should_skip('42', reopen_at='2026-07-15T00:00:00Z') is True

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


class TestArbitrationPendingReopenAtArm:
    """``arbitration_pending`` inherits ``should_skip``'s invalidation arm.

    ``should_skip`` deliberately RELEASES when the task's current
    ``reopen_at`` no longer matches the recorded one — "the task was
    reopened again since the conflict was recorded — a fresh write attempt
    is warranted" — and ``merge_queue.py`` names that arm as its self-heal.
    A reopen_at-agnostic durable hold would OVERRIDE it and wedge a
    re-reopened task forever (no dispatch, no flip) until a human resolved
    the L2, i.e. fixing a restart bug by installing a permanent one.

    The data for doing it durably is already on the record: ``record()``
    serialises ``reopen_at`` into the escalation's ``detail`` JSON.  Records
    here are built through the real ``record()`` against a real
    ``EscalationQueue`` so the ``detail`` shape is never hand-faked.

    Deliberate asymmetry, pinned below: EVERY ambiguity resolves to HOLD.
    Releasing is the action that lets a contested task move, so it must
    require positive evidence that the arbitration is stale; the absence of
    evidence is never enough.
    """

    _R1 = '2026-07-15T00:00:00+00:00'
    _R2 = '2026-07-20T00:00:00+00:00'

    def _sink_with_conflict(self, tmp_path):
        from orchestrator.provenance_conflict import ProvenanceConflictSink

        queue = EscalationQueue(tmp_path / 'esc')
        sink = ProvenanceConflictSink(escalation_queue=queue)
        _record(sink)
        return sink, queue.get_by_task('42', status='pending')

    def test_moved_on_reopen_at_releases_the_hold(self, tmp_path):
        """The task was reopened after the conflict — the hold must release."""
        sink, rows = self._sink_with_conflict(tmp_path)

        assert sink.arbitration_pending(rows, reopen_at=self._R2) is False

    def test_same_reopen_at_holds(self, tmp_path):
        sink, rows = self._sink_with_conflict(tmp_path)

        assert sink.arbitration_pending(rows, reopen_at=self._R1) is True

    def test_iso8601_drift_still_holds(self, tmp_path):
        """``...Z`` vs ``...+00:00`` is the drift ``_reopen_at_matches``
        exists to absorb — a naive ``==`` would wrongly RELEASE here.
        """
        sink, rows = self._sink_with_conflict(tmp_path)

        assert sink.arbitration_pending(
            rows, reopen_at='2026-07-15T00:00:00Z',
        ) is True

    def test_omitted_reopen_at_holds(self, tmp_path):
        """No ``reopen_at`` kwarg (the ``_UNKNOWN_REOPEN`` sentinel) — the
        same "caller has no metadata handy" arm ``should_skip`` already has.
        """
        sink, rows = self._sink_with_conflict(tmp_path)

        assert sink.arbitration_pending(rows) is True

    def test_unparseable_detail_holds(self, tmp_path):
        """A record whose ``detail`` cannot yield a recorded ``reopen_at``
        carries no evidence that the arbitration is stale, so it HOLDS.
        """
        sink, rows = self._sink_with_conflict(tmp_path)

        # ``Escalation.detail`` is declared ``str``, so ``None`` is deliberately
        # OFF-CONTRACT here: it pins the sink's ``getattr(record, 'detail', None)
        # or '{}'`` defensive arm, which exists for records rehydrated from a
        # store that serialized ``detail`` as JSON ``null``.  Annotated ``Any``
        # so the type checker permits the off-contract assignment.
        bad_detail: Any
        for bad_detail in ('not json', None):
            rows[0].detail = bad_detail
            assert sink.arbitration_pending(
                rows, reopen_at=self._R2,
            ) is True, f'detail={bad_detail!r} must hold'

    def test_non_provenance_records_never_hold(self, tmp_path):
        """A pending blocking L2 in another category is the ANY-LEVEL veto's
        business, not the arbitration hold's.
        """
        from escalation.models import Escalation

        from orchestrator.provenance_conflict import ProvenanceConflictSink

        sink = ProvenanceConflictSink(
            escalation_queue=EscalationQueue(tmp_path / 'esc'),
        )
        rows = [
            Escalation(
                id='esc-42-1',
                task_id='42',
                agent_role='implementer',
                severity='blocking',
                category='design_concern',
                summary='an open handoff on task 42',
                level=2,
            ),
        ]

        assert sink.arbitration_pending(rows) is False

    def test_none_records_do_not_hold(self, tmp_path):
        """An unreadable store is the CALLER's disposition (store_unavailable),
        not something this method may quietly convert into a hold.
        """
        from orchestrator.provenance_conflict import ProvenanceConflictSink

        sink = ProvenanceConflictSink(
            escalation_queue=EscalationQueue(tmp_path / 'esc'),
        )

        assert sink.arbitration_pending(None) is False

    def test_resolved_conflict_row_does_not_hold(self, tmp_path):
        """A RESOLVED conflict released the task — mirrors ``should_skip``'s
        ``_escalation_pending`` arm (amendment, reviewer_comprehensive
        robustness_api_contract).

        The parameter is named ``pending_records`` and the only in-tree
        caller passes ``get_by_task(tid, status='pending')``, but the method
        accepts an arbitrary iterable and the sink is shared with the
        merge-queue writer sites.  A caller handing over an UNFILTERED
        ``get_by_task(tid)`` (which also scans the archive tier, so it yields
        long-resolved rows) must not thereby wedge the task forever: the
        documented release path — an operator resolves the L2 — has to keep
        releasing.
        """
        from orchestrator.provenance_conflict import ProvenanceConflictSink

        queue = EscalationQueue(tmp_path / 'esc')
        sink = ProvenanceConflictSink(escalation_queue=queue)
        escalation_id = _record(sink)
        assert escalation_id is not None
        assert queue.resolve(escalation_id, 'operator arbitrated') is not None

        # The unfiltered read: the resolved row now lives in the archive tier.
        unfiltered = queue.get_by_task('42')
        assert [e.status for e in unfiltered] == ['resolved'], unfiltered

        assert sink.arbitration_pending(unfiltered, reopen_at=self._R1) is False
        assert sink.arbitration_pending(unfiltered) is False


class TestArbitrationHoldLogStreak:
    """``note_hold_observed`` / ``clear_hold_observed`` keep the hold's log
    line loud on a TRANSITION and quiet on the repeats (amendment,
    reviewer_comprehensive observability_log_volume).

    The hold re-fires on every dispatch tick for as long as the arbitration
    stands — and on a cold-memo process it short-circuits before
    ``record()``, so nothing re-warms ``should_skip`` to quiet it.  Logging
    it unconditionally at INFO would emit one line per tick until a human
    resolved the L2, which is the log spam INV-4 forbids.
    """

    _R1 = '2026-07-15T00:00:00+00:00'
    _R2 = '2026-07-20T00:00:00+00:00'

    def _sink(self, tmp_path):
        from orchestrator.provenance_conflict import ProvenanceConflictSink

        return ProvenanceConflictSink(
            escalation_queue=EscalationQueue(tmp_path / 'esc'),
        )

    def test_first_observation_is_loud_then_quiet(self, tmp_path):
        sink = self._sink(tmp_path)

        assert sink.note_hold_observed('42', reopen_at=self._R1) is True
        assert sink.note_hold_observed('42', reopen_at=self._R1) is False
        assert sink.note_hold_observed('42', reopen_at=self._R1) is False

    def test_moved_on_reopen_at_is_a_new_transition(self, tmp_path):
        """A re-reopen is genuinely new hold state — an operator wants it."""
        sink = self._sink(tmp_path)

        assert sink.note_hold_observed('42', reopen_at=self._R1) is True
        assert sink.note_hold_observed('42', reopen_at=self._R2) is True
        assert sink.note_hold_observed('42', reopen_at=self._R2) is False

    def test_absent_reopen_at_dedupes_too(self, tmp_path):
        """A caller with no task metadata handy still gets ONE loud line."""
        sink = self._sink(tmp_path)

        assert sink.note_hold_observed('42') is True
        assert sink.note_hold_observed('42') is False
        assert sink.note_hold_observed('42', reopen_at=None) is False

    def test_streaks_are_per_task(self, tmp_path):
        sink = self._sink(tmp_path)

        assert sink.note_hold_observed('42', reopen_at=self._R1) is True
        assert sink.note_hold_observed('43', reopen_at=self._R1) is True

    def test_release_makes_a_returning_hold_loud_again(self, tmp_path):
        """Also the pruning arm: the dict stays bounded by CURRENTLY-held
        tasks rather than every task ever contested in this process.
        """
        sink = self._sink(tmp_path)

        assert sink.note_hold_observed('42', reopen_at=self._R1) is True
        sink.clear_hold_observed('42')
        assert sink._hold_logged == {}

        assert sink.note_hold_observed('42', reopen_at=self._R1) is True

    def test_clearing_an_unheld_task_is_a_no_op(self, tmp_path):
        sink = self._sink(tmp_path)

        sink.clear_hold_observed('42')  # must not raise
        assert sink._hold_logged == {}
