"""Tests for scripts/backfill_gate_backlog_summaries.py (task 4314).

The script rewrites LEGACY (pre-3520) ``reconciliation_stale_gate_backlog``
summaries — the relative-age phrasing ``'... for 48.7h'`` — into the absolute
``since <ISO>`` anchor that
``fused_memory/reconciliation/stage1_stall_detector.py::maybe_escalate_stalled_gate_backlog``
has emitted since task 3520.

The script lives under ``scripts/``, which is not a package and not on
PYTHONPATH, so it is loaded through the SHARED ``load_script_module`` helper
rather than a local ``spec_from_file_location`` copy — see the mandate in
``tests/conftest.py`` (tasks 3738 / 3895).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from _fm_helpers import load_script_module
from escalation.dedupe import compute_content_fingerprint, gate_backlog_fingerprint_key
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from fused_memory.reconciliation.stage1_stall_detector import (
    STAGE1_GATE_BACKLOG_STALL_THRESHOLD_SECS,
    maybe_escalate_stalled_gate_backlog,
)

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'backfill_gate_backlog_summaries.py'

_mod = load_script_module(SCRIPT_PATH, mod_name='backfill_gate_backlog_summaries')

GATE_BACKLOG_CATEGORY = _mod.GATE_BACKLOG_CATEGORY
is_legacy_gate_backlog_record = _mod.is_legacy_gate_backlog_record
extract_gate_escalated_at = _mod.extract_gate_escalated_at
rebuild_summary = _mod.rebuild_summary
rebuild_detail = _mod.rebuild_detail
plan_rewrites = _mod.plan_rewrites
Rewrite = _mod.Rewrite
BackfillPlan = _mod.BackfillPlan
apply_rewrites = _mod.apply_rewrites
run = _mod.run
main = _mod.main


# ---------------------------------------------------------------------------
# Fixture data — a realistic pre-3520 record, modelled line-for-line on a live
# one (esc-5943-1 on the production queue).
# ---------------------------------------------------------------------------

ANCHOR = '2026-08-01T18:18:50.294218+00:00'
RUN_ID = '62e9b073-a070-47dc-b179-03608db93bef'

LEGACY_SUMMARY = 'Gate task 166 has awaited a human decision for 48.7h'
ANCHORED_SUMMARY = (
    'Gate task 166 has awaited a human decision since '
    f'{ANCHOR} (past the 48h gate-backlog threshold)'
)
FALLBACK_SUMMARY = (
    'Gate task 166 has awaited a human decision beyond the 48h gate-backlog threshold'
)

# The multi-line ``description:`` block, carrying blank lines and embedded
# quotes — the shape that makes a naive line-count or strip-based rewrite fail.
_DESCRIPTION_LINES = [
    'description: Mem0 observations_and_summaries memory bbaa46e5 reads:',
    '',
    '"Velocity IS registered (dimension.rs:550) so \\"Velocity alias\\" framing is stale."',
    '',
    'DECIDE ONE OF (per the Source-Completion conservative rule):',
    '(a) Edit bbaa46e5 to drop only the now-false clause.',
    '(b) Leave bbaa46e5 as a historical record and write a superseding memory.',
]

LEGACY_DETAIL_LINES = [
    'project_id: dark_factory',
    f'run_id: {RUN_ID}',
    'task_id: 166',
    f'gate_escalated_at: {ANCHOR}',
    'age_hours: 48.7',
    "title: Human curator gate: correct a stale 'Torque' clause in memory bbaa46e5",
    *_DESCRIPTION_LINES,
]
LEGACY_DETAIL = '\n'.join(LEGACY_DETAIL_LINES)


def _legacy_esc(**kwargs) -> Escalation:
    """Build a realistic pre-3520 gate-backlog record; *kwargs* override fields."""
    fields: dict = {
        'id': 'esc-166-1',
        'task_id': '166',
        'agent_role': 'reconciliation-stage1',
        'severity': 'blocking',
        'category': GATE_BACKLOG_CATEGORY,
        'summary': LEGACY_SUMMARY,
        'detail': LEGACY_DETAIL,
        'level': 1,
        'status': 'pending',
        'dedupe_fingerprint': None,
    }
    fields.update(kwargs)
    return Escalation(**fields)


# ---------------------------------------------------------------------------
# is_legacy_gate_backlog_record
# ---------------------------------------------------------------------------


class TestIsLegacyGateBacklogRecord:
    """Selection predicate: the record's own OLD-SUMMARY SHAPE, nothing else."""

    def test_category_constant_matches_the_emitter(self):
        assert GATE_BACKLOG_CATEGORY == 'reconciliation_stale_gate_backlog'

    def test_legacy_record_is_selected(self):
        assert is_legacy_gate_backlog_record(_legacy_esc()) is True

    def test_3520_anchored_summary_is_not_selected(self):
        """The measured 11-record already-anchored-but-unstamped population."""
        esc = _legacy_esc(summary=ANCHORED_SUMMARY)
        assert is_legacy_gate_backlog_record(esc) is False

    def test_3520_fallback_summary_is_not_selected(self):
        esc = _legacy_esc(summary=FALLBACK_SUMMARY)
        assert is_legacy_gate_backlog_record(esc) is False

    def test_other_category_is_not_selected(self):
        esc = _legacy_esc(category='recon_integrity_issue')
        assert is_legacy_gate_backlog_record(esc) is False

    def test_resolved_record_is_not_selected(self):
        esc = _legacy_esc(status='resolved')
        assert is_legacy_gate_backlog_record(esc) is False

    def test_trailing_prose_after_the_hours_is_not_selected(self):
        """The regex must be fully anchored at both ends."""
        esc = _legacy_esc(summary=f'{LEGACY_SUMMARY} and counting')
        assert is_legacy_gate_backlog_record(esc) is False

    def test_leading_prose_before_the_summary_is_not_selected(self):
        esc = _legacy_esc(summary=f'FYI: {LEGACY_SUMMARY}')
        assert is_legacy_gate_backlog_record(esc) is False

    def test_empty_summary_is_not_selected(self):
        esc = _legacy_esc(summary='')
        assert is_legacy_gate_backlog_record(esc) is False


# ---------------------------------------------------------------------------
# extract_gate_escalated_at
# ---------------------------------------------------------------------------


class TestExtractGateEscalatedAt:
    """Anchor recovery: a fail-closed prose parser that is NEVER a reformatter."""

    def test_returns_the_anchor_byte_identically(self):
        """Assert with ``==`` on the STRING, not a date comparison."""
        assert extract_gate_escalated_at(LEGACY_DETAIL) == ANCHOR

    def test_z_suffixed_value_round_trips_verbatim(self):
        """A ``Z`` suffix stays a ``Z`` — the parse is a gate, not a normaliser."""
        detail = LEGACY_DETAIL.replace(
            f'gate_escalated_at: {ANCHOR}', 'gate_escalated_at: 2026-08-01T18:18:50Z'
        )
        assert extract_gate_escalated_at(detail) == '2026-08-01T18:18:50Z'

    def test_crlf_detail_strips_only_the_trailing_cr(self):
        detail = LEGACY_DETAIL.replace('\n', '\r\n')
        assert extract_gate_escalated_at(detail) == ANCHOR

    def test_missing_line_returns_none(self):
        detail = '\n'.join(
            line for line in LEGACY_DETAIL_LINES
            if not line.startswith('gate_escalated_at: ')
        )
        assert extract_gate_escalated_at(detail) is None

    def test_empty_value_returns_none(self):
        detail = LEGACY_DETAIL.replace(
            f'gate_escalated_at: {ANCHOR}', 'gate_escalated_at: '
        )
        assert extract_gate_escalated_at(detail) is None

    def test_literal_none_token_returns_none(self):
        """The emitter writes ``f'gate_escalated_at: {gate_escalated_at}'`` unguarded."""
        detail = LEGACY_DETAIL.replace(
            f'gate_escalated_at: {ANCHOR}', 'gate_escalated_at: None'
        )
        assert extract_gate_escalated_at(detail) is None

    def test_unparseable_value_returns_none(self):
        detail = LEGACY_DETAIL.replace(
            f'gate_escalated_at: {ANCHOR}', 'gate_escalated_at: not-a-timestamp'
        )
        assert extract_gate_escalated_at(detail) is None

    def test_empty_detail_returns_none(self):
        assert extract_gate_escalated_at('') is None

    def test_prefix_is_matched_at_line_start_and_first_wins(self):
        """The anti-false-positive case that matters on real data.

        A record whose multi-line ``description:`` block itself quotes a
        ``gate_escalated_at: ...`` line must still yield the value from the real
        line 4 — proving the parser scans for a LINE-START prefix and takes the
        FIRST match, not a substring search anywhere in the blob.
        """
        decoy = 'gate_escalated_at: 1999-01-01T00:00:00+00:00'
        detail = '\n'.join([*LEGACY_DETAIL_LINES, decoy])
        assert extract_gate_escalated_at(detail) == ANCHOR

    def test_mid_line_occurrence_is_not_matched(self):
        """A prose line MENTIONING the key mid-sentence is not a match."""
        detail = '\n'.join([
            'project_id: dark_factory',
            'task_id: 166',
            'description: the record had gate_escalated_at: 1999-01-01T00:00:00+00:00 set',
        ])
        assert extract_gate_escalated_at(detail) is None


# ---------------------------------------------------------------------------
# rebuild_summary — EMITTER PARITY
# ---------------------------------------------------------------------------


class TestRebuildSummary:
    """The rebuilt summary must be byte-identical to what the live emitter writes."""

    @pytest.mark.asyncio
    async def test_matches_a_freshly_minted_record_byte_for_byte(self, tmp_path: Path):
        """Parity is pinned to the EMITTER, not to a copied literal.

        Mint a record by actually calling
        ``stage1_stall_detector.maybe_escalate_stalled_gate_backlog`` against a
        real ``EscalationQueue``, then assert ``rebuild_summary`` reproduces its
        summary exactly.  A future edit to either f-string breaks this test.
        """
        queue = EscalationQueue(tmp_path)
        now = datetime.fromisoformat(ANCHOR) + timedelta(hours=48.7)
        task_by_id = {
            '166': {
                'id': '166',
                'status': 'blocked',
                'title': 'Gate task 166',
                'metadata': {'operational_mode': 'gate', 'gate_escalated_at': ANCHOR},
            }
        }

        escalated = await maybe_escalate_stalled_gate_backlog(
            queue,
            project_id='dark_factory',
            run_id='r1',
            stalled_task_ids=['166'],
            task_by_id=task_by_id,
            now=now,
        )
        assert escalated == ['166'], 'the emitter must have filed a NEW record'

        pending = queue.get_pending()
        assert len(pending) == 1
        minted = pending[0]

        assert rebuild_summary('166', ANCHOR) == minted.summary

    def test_fallback_branch_when_the_anchor_is_unrecoverable(self):
        assert rebuild_summary('166', None) == FALLBACK_SUMMARY

    def test_anchored_branch_shape(self):
        assert rebuild_summary('166', ANCHOR) == ANCHORED_SUMMARY

    def test_threshold_renders_from_the_shared_constant_not_a_literal(self):
        assert 'past the 72h gate-backlog threshold' in rebuild_summary(
            '166', ANCHOR, threshold_secs=72 * 3600
        )

    def test_default_threshold_is_the_emitters_constant(self):
        """The default must BE the shared constant, not a copy that can drift."""
        assert rebuild_summary('166', ANCHOR) == rebuild_summary(
            '166', ANCHOR, threshold_secs=STAGE1_GATE_BACKLOG_STALL_THRESHOLD_SECS
        )

    def test_anchor_is_interpolated_verbatim(self):
        """A ``Z``-suffixed anchor is not silently normalised on the way in."""
        assert rebuild_summary('166', '2026-08-01T18:18:50Z') == (
            'Gate task 166 has awaited a human decision since '
            '2026-08-01T18:18:50Z (past the 48h gate-backlog threshold)'
        )


# ---------------------------------------------------------------------------
# rebuild_detail
# ---------------------------------------------------------------------------


class TestRebuildDetail:
    """Rename ONE key on ONE line; every other byte of the detail is preserved."""

    def test_only_the_age_line_changes(self):
        """Line-by-line byte identity except index 4 — same value, same position."""
        before = LEGACY_DETAIL.split('\n')
        after = rebuild_detail(LEGACY_DETAIL).split('\n')

        assert len(after) == len(before), 'line count must not change'
        assert after[4] == 'age_hours_at_filing: 48.7'
        assert before[4] == 'age_hours: 48.7'
        for i, (b, a) in enumerate(zip(before, after, strict=True)):
            if i == 4:
                continue
            assert a == b, f'line {i} must be byte-identical: {b!r} != {a!r}'

    def test_line_zero_is_untouched(self):
        """Line 0 is gate_backlog_fingerprint_key's ONLY recovery site."""
        assert rebuild_detail(LEGACY_DETAIL).split('\n')[0] == 'project_id: dark_factory'

    def test_multiline_description_block_is_preserved_verbatim(self):
        after = rebuild_detail(LEGACY_DETAIL)
        assert '\n'.join(_DESCRIPTION_LINES) in after

    def test_already_canonical_detail_round_trips_unchanged(self):
        """Idempotence: a second pass over a rewritten detail is a true no-op."""
        once = rebuild_detail(LEGACY_DETAIL)
        assert rebuild_detail(once) == once

    def test_detail_with_no_age_line_round_trips_unchanged(self):
        detail = '\n'.join(
            line for line in LEGACY_DETAIL_LINES if not line.startswith('age_hours: ')
        )
        assert rebuild_detail(detail) == detail

    def test_empty_detail_round_trips_unchanged(self):
        assert rebuild_detail('') == ''

    def test_only_the_first_age_line_is_renamed(self):
        detail = '\n'.join([*LEGACY_DETAIL_LINES, 'age_hours: 99.9'])
        after = rebuild_detail(detail).split('\n')
        assert after[4] == 'age_hours_at_filing: 48.7'
        assert after[-1] == 'age_hours: 99.9'

    def test_mid_prose_occurrence_is_not_renamed(self):
        """Line-start prefix match only — a description mentioning it stays put."""
        detail = '\n'.join([
            'project_id: dark_factory',
            'task_id: 166',
            'description: the record said age_hours: 3.2 at filing time',
        ])
        assert rebuild_detail(detail) == detail

    def test_canonical_prefix_is_never_double_prefixed(self):
        """``age_hours_at_filing: `` must not be read as ``age_hours: `` + junk."""
        detail = 'project_id: dark_factory\nage_hours_at_filing: 1.0'
        after = rebuild_detail(detail)
        assert 'age_hours_at_filing_at_filing:' not in after
        assert after == detail

    def test_canonical_line_does_not_shield_a_later_legacy_line(self):
        """A canonical line earlier in the blob must not stop the legacy rename."""
        detail = '\n'.join([
            'project_id: dark_factory',
            'age_hours_at_filing: 1.0',
            'age_hours: 48.7',
        ])
        assert rebuild_detail(detail).split('\n') == [
            'project_id: dark_factory',
            'age_hours_at_filing: 1.0',
            'age_hours_at_filing: 48.7',
        ]


# ---------------------------------------------------------------------------
# plan_rewrites
# ---------------------------------------------------------------------------


def _anchored_esc(**kwargs) -> Escalation:
    """A post-3520 record whose summary is ALREADY correct."""
    detail = LEGACY_DETAIL.replace('age_hours: 48.7', 'age_hours_at_filing: 48.7')
    return _legacy_esc(summary=ANCHORED_SUMMARY, detail=detail, **kwargs)


def _mixed_pending() -> list[Escalation]:
    """The live queue's shape in miniature: 2 legacy + 3 records to leave alone."""
    stamped = compute_content_fingerprint(
        GATE_BACKLOG_CATEGORY, '', ['dark_factory:169'], ''
    )
    return [
        _legacy_esc(id='esc-166-1', task_id='166'),
        _legacy_esc(
            id='esc-167-1',
            task_id='167',
            summary='Gate task 167 has awaited a human decision for 91.2h',
            detail=LEGACY_DETAIL.replace('task_id: 166', 'task_id: 167'),
        ),
        # The measured 11-record population: unstamped, but already anchored.
        _anchored_esc(id='esc-168-1', task_id='168'),
        # The measured 54-record population: anchored AND stamped.
        _anchored_esc(id='esc-169-1', task_id='169', dedupe_fingerprint=stamped),
        _legacy_esc(id='esc-170-1', task_id='170', category='recon_integrity_issue'),
    ]


class TestPlanRewrites:
    """Pure planning: which records get rewritten, and to what."""

    def test_selects_only_the_legacy_records(self):
        plan = plan_rewrites(_mixed_pending())

        assert isinstance(plan, BackfillPlan)
        assert [r.escalation_id for r in plan.rewrites] == ['esc-166-1', 'esc-167-1']
        assert plan.pending_total == 5
        assert plan.legacy_total == 2
        assert plan.skipped_fingerprint_drift == 0

    def test_already_anchored_unstamped_record_is_never_touched(self):
        """The 11 records filed between 3520 landing and 3522's stamp landing."""
        plan = plan_rewrites(_mixed_pending())
        assert 'esc-168-1' not in {r.escalation_id for r in plan.rewrites}

    def test_rewrite_carries_the_before_and_after_of_both_fields(self):
        plan = plan_rewrites(_mixed_pending())
        rw = plan.rewrites[0]

        assert isinstance(rw, Rewrite)
        assert rw.escalation_id == 'esc-166-1'
        assert rw.task_id == '166'
        assert rw.old_summary == LEGACY_SUMMARY
        assert rw.new_summary == rebuild_summary('166', ANCHOR)
        assert rw.old_detail == LEGACY_DETAIL
        assert rw.new_detail == rebuild_detail(LEGACY_DETAIL)
        assert rw.anchored is True

    def test_counters_split_anchored_from_fallback(self):
        plan = plan_rewrites(_mixed_pending())
        assert plan.anchored == 2
        assert plan.fallback == 0

    def test_unrecoverable_anchor_falls_back_and_is_counted(self):
        esc = _legacy_esc(
            detail=LEGACY_DETAIL.replace(
                f'gate_escalated_at: {ANCHOR}', 'gate_escalated_at: None'
            )
        )
        plan = plan_rewrites([esc])

        assert plan.legacy_total == 1
        assert plan.anchored == 0
        assert plan.fallback == 1
        assert len(plan.rewrites) == 1
        assert plan.rewrites[0].anchored is False
        assert plan.rewrites[0].new_summary == FALLBACK_SUMMARY

    def test_fingerprint_is_preserved_across_every_emitted_rewrite(self):
        """The positive form of the machine-checked invariant."""
        pending = _mixed_pending()
        by_id = {e.id: e for e in pending}
        plan = plan_rewrites(pending)

        assert plan.rewrites, 'expected at least one rewrite to check'
        for rw in plan.rewrites:
            before = by_id[rw.escalation_id]
            after = _legacy_esc(
                id=before.id,
                task_id=before.task_id,
                summary=rw.new_summary,
                detail=rw.new_detail,
                dedupe_fingerprint=before.dedupe_fingerprint,
            )
            key_before = gate_backlog_fingerprint_key(before)
            key_after = gate_backlog_fingerprint_key(after)
            assert key_before is not None
            assert key_after is not None
            assert key_before == key_after

    def test_fails_closed_when_the_rewrite_would_move_the_fingerprint(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """A rebuild_detail that disturbs line 0 must emit NO rewrite at all.

        The failure mode this guards is a permanently non-folding parent that
        mints a duplicate every Stage-1 cycle — so the guard fails CLOSED
        (record keeps its stale summary, a strictly recoverable outcome) rather
        than rewriting and hoping.
        """
        monkeypatch.setattr(
            _mod,
            'rebuild_detail',
            lambda detail: detail.replace('project_id: dark_factory', 'project_id: reify'),
        )
        plan = plan_rewrites([_legacy_esc()])

        assert plan.legacy_total == 1
        assert plan.rewrites == []
        assert plan.skipped_fingerprint_drift == 1

    def test_empty_pending_list_yields_an_empty_plan(self):
        plan = plan_rewrites([])
        assert plan.rewrites == []
        assert plan.pending_total == 0
        assert plan.legacy_total == 0


# ---------------------------------------------------------------------------
# apply_rewrites — the FROZEN-FIELD contract
# ---------------------------------------------------------------------------

FROZEN_TIMESTAMP = '2026-08-01T18:18:50.294218+00:00'
FROZEN_UPDATED_AT = '2026-08-02T00:00:00+00:00'


def _seeded_queue(tmp_path: Path, *escalations: Escalation) -> EscalationQueue:
    queue = EscalationQueue(tmp_path)
    for esc in escalations:
        queue.submit(esc)
    return queue


def _folded_legacy_esc(**kwargs) -> Escalation:
    """A legacy record carrying non-default values on every FROZEN field."""
    fields: dict = {
        'dedupe_count': 7,
        'dedupe_children': ['esc-166-2', 'esc-166-3'],
        'dedupe_fingerprint': None,
        'level': 1,
        'severity': 'blocking',
        'timestamp': FROZEN_TIMESTAMP,
        'updated_at': FROZEN_UPDATED_AT,
    }
    fields.update(kwargs)
    return _legacy_esc(**fields)


class TestApplyRewritesFrozenFields:
    """Only ``summary`` and ``detail`` may change.  Everything else is frozen."""

    def test_every_other_field_is_byte_identical_on_disk(self, tmp_path: Path):
        queue = _seeded_queue(tmp_path, _folded_legacy_esc())
        raw = tmp_path / 'esc-166-1.json'
        before = json.loads(raw.read_text())

        plan = plan_rewrites(queue.get_pending())
        report = apply_rewrites(queue, plan)

        after = json.loads(raw.read_text())
        assert after['summary'] == rebuild_summary('166', ANCHOR)
        assert after['detail'] == rebuild_detail(LEGACY_DETAIL)
        assert report['rewritten'] == 1

        # TOTAL assertion: pop the two mutable fields and require the rest to be
        # identical, so a field added to Escalation later is covered with no
        # edit here.
        for d in (before, after):
            d.pop('summary')
            d.pop('detail')
        assert after == before

    def test_dedupe_fingerprint_is_not_opportunistically_stamped(self, tmp_path: Path):
        """3522's tested steady state is UNSTAMPED — recomputed deterministically."""
        queue = _seeded_queue(tmp_path, _folded_legacy_esc())
        apply_rewrites(queue, plan_rewrites(queue.get_pending()))

        after = json.loads((tmp_path / 'esc-166-1.json').read_text())
        assert after['dedupe_fingerprint'] is None

    def test_updated_at_is_not_bumped(self, tmp_path: Path):
        """A display-only correction is not a substantive change (models.py:392).

        Bumping it on 67 records would mass-invalidate the watcher's triage
        stamps via the ``updated_at > triaged_at`` re-verify rule and force a
        full re-drain of the backlog.
        """
        queue = _seeded_queue(tmp_path, _folded_legacy_esc())
        apply_rewrites(queue, plan_rewrites(queue.get_pending()))

        after = json.loads((tmp_path / 'esc-166-1.json').read_text())
        assert after['updated_at'] == FROZEN_UPDATED_AT

    def test_fold_state_survives(self, tmp_path: Path):
        queue = _seeded_queue(tmp_path, _folded_legacy_esc())
        apply_rewrites(queue, plan_rewrites(queue.get_pending()))

        after = json.loads((tmp_path / 'esc-166-1.json').read_text())
        assert after['dedupe_count'] == 7
        assert after['dedupe_children'] == ['esc-166-2', 'esc-166-3']

    def test_record_archived_between_planning_and_applying_is_skipped(
        self, tmp_path: Path
    ):
        """Root only, never the archive — mirrors attach_dedupe_child's guard."""
        queue = _seeded_queue(tmp_path, _folded_legacy_esc())
        plan = plan_rewrites(queue.get_pending())

        # A steward resolved it; sweep relocated the file out of the root.
        (tmp_path / 'esc-166-1.json').unlink()

        report = apply_rewrites(queue, plan)
        assert report['rewritten'] == 0
        assert report['skipped_missing'] == 1


# ---------------------------------------------------------------------------
# apply_rewrites — LOCK AND RE-READ (safety against live Stage-1 folds)
# ---------------------------------------------------------------------------


class TestApplyRewritesLockDiscipline:
    """These records are LIVE fold targets; every RMW must be lock-scoped."""

    def test_takes_the_per_id_sidecar_lock(self, tmp_path: Path):
        """Mirrors escalation/tests/test_sweep.py::TestSweepRelocationLock.

        The sidecar is removed after seeding — ``queue.submit`` takes the same
        lock, so leaving it in place would make this assertion vacuous.
        """
        queue = _seeded_queue(tmp_path, _folded_legacy_esc())
        plan = plan_rewrites(queue.get_pending())
        (tmp_path / 'esc-166-1.json.lock').unlink()

        apply_rewrites(queue, plan)

        assert (tmp_path / 'esc-166-1.json.lock').exists(), (
            'Expected sidecar lock file esc-166-1.json.lock in the queue root — '
            'apply_rewrites must take escalation_id_lock around every '
            'read-modify-write, because these records are live Stage-1 fold targets'
        )

    def test_a_concurrent_fold_is_not_reverted(self, tmp_path: Path):
        """A fold landing after planning must survive the rewrite intact.

        A lock-free ``get()`` -> mutate-the-planned-copy -> ``submit()`` would
        silently revert ``dedupe_count``/``dedupe_children``/``updated_at`` —
        exactly the mutation this backfill is forbidden to make.
        """
        queue = _seeded_queue(tmp_path, _folded_legacy_esc())
        plan = plan_rewrites(queue.get_pending())

        # A Stage-1 cycle folds a repeat filing into this parent.
        queue.attach_dedupe_child('esc-166-1', 'esc-166-9')
        folded = json.loads((tmp_path / 'esc-166-1.json').read_text())

        apply_rewrites(queue, plan)

        after = json.loads((tmp_path / 'esc-166-1.json').read_text())
        assert after['summary'] == rebuild_summary('166', ANCHOR)
        assert after['dedupe_count'] == folded['dedupe_count'] == 8
        assert after['dedupe_children'] == ['esc-166-2', 'esc-166-3', 'esc-166-9']
        assert after['updated_at'] == folded['updated_at'] != FROZEN_UPDATED_AT

    def test_record_no_longer_legacy_at_apply_time_is_skipped(self, tmp_path: Path):
        """TOCTOU: re-check the predicate INSIDE the lock, as sweep does."""
        queue = _seeded_queue(tmp_path, _folded_legacy_esc())
        plan = plan_rewrites(queue.get_pending())

        raw = tmp_path / 'esc-166-1.json'
        record = json.loads(raw.read_text())
        record['summary'] = ANCHORED_SUMMARY
        raw.write_text(json.dumps(record))

        report = apply_rewrites(queue, plan)

        assert report['rewritten'] == 0
        assert report['skipped_no_longer_legacy'] == 1
        assert json.loads(raw.read_text())['summary'] == ANCHORED_SUMMARY

    def test_a_concurrent_detail_change_is_not_clobbered_by_stale_text(
        self, tmp_path: Path
    ):
        """The rebuild must run on the FRESHLY READ record, not the planned string."""
        queue = _seeded_queue(tmp_path, _folded_legacy_esc())
        plan = plan_rewrites(queue.get_pending())

        raw = tmp_path / 'esc-166-1.json'
        record = json.loads(raw.read_text())
        fresh_detail = f'{LEGACY_DETAIL}\nnote: appended after planning'
        record['detail'] = fresh_detail
        raw.write_text(json.dumps(record))

        apply_rewrites(queue, plan)

        after = json.loads(raw.read_text())
        assert after['detail'] == rebuild_detail(fresh_detail)
        assert after['detail'].endswith('\nnote: appended after planning')


# ---------------------------------------------------------------------------
# run() / main() — end to end
# ---------------------------------------------------------------------------


def _snapshot(tmp_path: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(tmp_path.glob('*.json'))}


class TestRunAndMain:
    """Dry-run is the default and is TOTAL; --apply is exact and idempotent."""

    def test_dry_run_is_the_default_and_writes_nothing(self, tmp_path: Path):
        _seeded_queue(tmp_path, *_mixed_pending())
        before = _snapshot(tmp_path)

        report = run(tmp_path)

        assert report['dry_run'] is True
        assert report['pending_total'] == 5
        assert report['legacy_total'] == 2
        assert _snapshot(tmp_path) == before, 'a dry run must not write ANY file'

    def test_report_carries_the_planning_counters(self, tmp_path: Path):
        _seeded_queue(tmp_path, *_mixed_pending())
        report = run(tmp_path)

        assert report['queue_dir'] == str(tmp_path)
        assert report['anchored'] == 2
        assert report['fallback'] == 0
        assert report['skipped_fingerprint_drift'] == 0
        # Apply-only counters must be absent from a dry run's report.
        assert 'rewritten' not in report
        assert 'skipped_missing' not in report
        assert 'skipped_no_longer_legacy' not in report

    def test_apply_rewrites_exactly_the_legacy_records(self, tmp_path: Path):
        _seeded_queue(tmp_path, *_mixed_pending())

        report = run(tmp_path, apply=True)

        assert report['dry_run'] is False
        assert report['rewritten'] == 2
        assert report['skipped_missing'] == 0
        assert report['skipped_no_longer_legacy'] == 0
        after = json.loads((tmp_path / 'esc-166-1.json').read_text())
        assert after['summary'] == rebuild_summary('166', ANCHOR)

    def test_non_targets_are_byte_identical_after_an_apply(self, tmp_path: Path):
        _seeded_queue(tmp_path, *_mixed_pending())
        before = _snapshot(tmp_path)

        run(tmp_path, apply=True)

        after = _snapshot(tmp_path)
        for name in ('esc-168-1.json', 'esc-169-1.json', 'esc-170-1.json'):
            assert after[name] == before[name], f'{name} must not be touched'
        for name in ('esc-166-1.json', 'esc-167-1.json'):
            assert after[name] != before[name]

    def test_second_apply_is_a_structural_no_op(self, tmp_path: Path):
        _seeded_queue(tmp_path, *_mixed_pending())
        run(tmp_path, apply=True)
        after_first = _snapshot(tmp_path)

        report = run(tmp_path, apply=True)

        assert report['legacy_total'] == 0
        assert report['rewritten'] == 0
        assert _snapshot(tmp_path) == after_first

    def test_main_dry_run_returns_0_prints_json_and_writes_nothing(
        self, tmp_path: Path, capsys
    ):
        _seeded_queue(tmp_path, *_mixed_pending())
        before = _snapshot(tmp_path)

        assert main(['--queue-dir', str(tmp_path)]) == 0

        report = json.loads(capsys.readouterr().out)
        assert report['dry_run'] is True
        assert report['legacy_total'] == 2
        assert _snapshot(tmp_path) == before

    def test_main_apply_performs_the_writes(self, tmp_path: Path, capsys):
        _seeded_queue(tmp_path, *_mixed_pending())

        assert main(['--queue-dir', str(tmp_path), '--apply']) == 0

        report = json.loads(capsys.readouterr().out)
        assert report['dry_run'] is False
        assert report['rewritten'] == 2
