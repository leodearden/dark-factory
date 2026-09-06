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

from pathlib import Path

from _fm_helpers import load_script_module
from escalation.models import Escalation

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'backfill_gate_backlog_summaries.py'

_mod = load_script_module(SCRIPT_PATH, mod_name='backfill_gate_backlog_summaries')

GATE_BACKLOG_CATEGORY = _mod.GATE_BACKLOG_CATEGORY
is_legacy_gate_backlog_record = _mod.is_legacy_gate_backlog_record


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
