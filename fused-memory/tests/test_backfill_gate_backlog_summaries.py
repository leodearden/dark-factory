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

from datetime import datetime, timedelta
from pathlib import Path

import pytest
from _fm_helpers import load_script_module
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
