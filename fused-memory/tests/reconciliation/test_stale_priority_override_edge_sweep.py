"""Tests for the stale priority-override / pin-queue Graphiti edge sweep (task 2781).

Reconciliation Stage 1 (MemoryConsolidator) has no deterministic sweep that
keeps VALID (invalid_at IS NULL) priority-override / pin-queue Graphiti
temporal_facts edges (boost tier, TTL, pin order) in sync with live
scheduler-override state. When a task's override is consumed (task dispatched
-> row cleared by clear_terminal) or expires (clear_expired), the task drops
out of the live override table but its "task N has boost/TTL/pin override"
edge silently persists as a stale valid_at edge until a human catches it
(this required TWO one-time manual backfills — run 2d59c7de, finding
3852bd07). This module adds a small deterministic post-processor mirroring
task 2613's stale_status_snapshot_edge_sweep: enumerate valid edges,
lexically extract each edge's single subject task_id, read LIVE override
state directly from scheduler_overrides.db (the full table, not
get_pin_queue's pinned=1 projection), and invalidate any edge whose task is
absent from the live table OR whose (TTL) edge's live ttl_until has elapsed.

Covers:
- extract_priority_override_task_id: pure lexical extractor — returns the
  single subject task_id a priority-override edge asserts, gated on the
  "priority override" phrase; None when the gate fails or the fact has no
  single distinct task-id subject (multi/no subject).
- is_ttl_override_fact: pure classifier — True iff BOTH the "priority
  override" phrase AND a "TTL" token are present (either order,
  separator-tolerant).
- select_stale_priority_override_edges: pure decision core — selects edges
  whose task is absent from the live override map, or whose TTL edge's live
  ttl_until has elapsed.
- read_live_override_state: async reader over scheduler_overrides.db (full
  table read).
- sweep_stale_priority_override_edges: async orchestrator — enumerates,
  reads live state, and invalidates, best-effort throughout.
"""

from __future__ import annotations

from fused_memory.reconciliation.stale_priority_override_edge_sweep import (
    extract_priority_override_task_id,
)


class TestExtractPriorityOverrideTaskId:
    """extract_priority_override_task_id(fact) returns the single subject
    task_id a priority-override edge asserts, or None.

    Gated on the "priority override" phrase: a fact without it is never a
    priority-override edge this sweep concerns itself with. Returns None when
    the fact has no single distinct extractable subject (a multi-subject
    event record or a count-only/no-subject fact), mirroring 2613's
    count-only exclusion.
    """

    def test_set_override_audit_form(self):
        """"Set priority override for task 5166: {...}" -> 5166."""
        assert (
            extract_priority_override_task_id(
                "Set priority override for task 5166: {'boost_tier': 'high'}"
            )
            == 5166
        )

    def test_pin_order_form(self):
        """"Task 4079 priority override pin order 3" -> 4079 (the '3' is not
        a task reference)."""
        assert (
            extract_priority_override_task_id('Task 4079 priority override pin order 3')
            == 4079
        )

    def test_ttl_form(self):
        """"Task 4940 priority override with a TTL of 3600s" -> 4940 (the
        '3600' is not a task reference)."""
        assert (
            extract_priority_override_task_id(
                'Task 4940 priority override with a TTL of 3600s'
            )
            == 4940
        )

    def test_non_override_fact_gated_out(self):
        """"Task 5 is done" -> None: no "priority override" phrase, so the
        gate fails before any id is extracted."""
        assert extract_priority_override_task_id('Task 5 is done') is None

    def test_count_only_no_subject_returns_none(self):
        """"There are 8 tasks in progress" -> None (no gate phrase, no
        single subject)."""
        assert extract_priority_override_task_id('There are 8 tasks in progress') is None

    def test_multi_subject_event_record_returns_none(self):
        """"Reordered pin queue: [1, 2, 3]" -> None: a multi-subject event
        record with no single "priority override" subject."""
        assert extract_priority_override_task_id('Reordered pin queue: [1, 2, 3]') is None

    def test_multiple_distinct_subjects_with_phrase_returns_none(self):
        """Even WITH the gate phrase, two distinct task ids -> None: the
        extractor returns a subject only when exactly one distinct id is
        present."""
        assert (
            extract_priority_override_task_id(
                'priority override reordered for task 1, task 2 and task 3'
            )
            is None
        )

    def test_incidental_date_digits_not_swept_in(self):
        """"Task 5166 priority override boost set 2026-07-19" -> 5166: the
        date's digits are not task references and must not defeat the
        single-subject rule."""
        assert (
            extract_priority_override_task_id(
                'Task 5166 priority override boost set 2026-07-19'
            )
            == 5166
        )
