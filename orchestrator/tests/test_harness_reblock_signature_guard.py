"""Tests for Harness signature-aware re-block guard (task δ / PRD contract C5).

Guard summary
-------------
Placed inside ``_cascade_unblock_member`` (after the ``status != 'blocked'``
early-exit, before ``set_task_status('pending')``), the guard counts
same-signature re-pends cross-incarnation.  After 3 same-signature flips the
4th is WITHHELD and a born-at-L2 human escalation is filed.

Tests are grouped by plan step:
  TestSignatureDerivation     — step-1 (RED) / step-2 (GREEN)
  TestBelowThresholdFlips     — step-3 (RED) / step-4 (GREEN)
  TestSignatureReset          — step-5 (RED) / step-6 (GREEN)
  TestThresholdTrip           — step-7 (RED) / step-8 (GREEN)
  TestDedupAndHumanReset      — step-9 (RED) / step-10 (GREEN)
  TestMetadataClobberBoundary — step-11 (RED) / step-12 (GREEN)
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from escalation.models import Escalation

from orchestrator.config import OrchestratorConfig
from orchestrator.harness import Harness
from orchestrator.scheduler import Scheduler

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def harness(mock_orch_config) -> Harness:
    """Harness with mocked internals for reblock-guard unit testing."""
    with (
        patch('orchestrator.harness.McpLifecycle'),
        patch('orchestrator.harness.OverrideStore'),
        patch('orchestrator.harness.Scheduler'),
        patch('orchestrator.harness.BriefingAssembler'),
    ):
        h = Harness(mock_orch_config)

    # Replace scheduler with async mocks
    h.scheduler = MagicMock()
    h.scheduler.get_status = AsyncMock(return_value='blocked')
    h.scheduler.set_task_status = AsyncMock()
    h.scheduler.get_task = AsyncMock(return_value={'id': '42', 'metadata': {}})
    h.scheduler.update_task = AsyncMock(return_value=True)

    # No escalation queue by default (bare-harness style)
    h._escalation_queue = None

    return h


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_l1_esc(
    task_id: str = '42',
    category: str = 'infra_issue',
    summary: str = 'something went wrong',
    status: str = 'resolved',
    resolved_by: str = 'l2-cascade:esc-100-1',
    level: int = 1,
) -> Escalation:
    return Escalation(
        id=f'esc-{task_id}-1',
        task_id=task_id,
        agent_role='workflow',
        severity='blocking',
        category=category,
        summary=summary,
        level=level,
        status=status,
        resolved_by=resolved_by,
    )


# ---------------------------------------------------------------------------
# Step-1 / Step-2: Signature derivation
# ---------------------------------------------------------------------------


class TestSignatureDerivation:
    """Harness._reblock_signature derives category:normalize(summary)[:120]."""

    def test_simple_category_and_summary(self):
        """(a) Simple category + summary — expected format."""
        esc = _make_l1_esc(category='infra_issue', summary='disk full')
        sig = Harness._reblock_signature(esc)
        assert sig == 'infra_issue:disk full'

    def test_mixed_case_and_whitespace_normalized(self):
        """(b) Mixed case + runs of whitespace/newlines → collapsed & lowercased."""
        raw_summary = '  Worker  CRASHED\n\twith   error  CODE 42  '
        esc = _make_l1_esc(category='task_failure', summary=raw_summary)
        sig = Harness._reblock_signature(esc)
        # normalize = ' '.join(s.split()).lower() → all whitespace runs collapsed to
        # a single space and the result lowercased.
        expected_norm = ' '.join(raw_summary.split()).lower()
        assert sig == f'task_failure:{expected_norm}'
        # Confirm the whitespace really was collapsed (no double-spaces in result)
        suffix = sig[len('task_failure:'):]
        assert '  ' not in suffix, 'Consecutive spaces should have been collapsed'

    def test_long_summary_truncated_to_120_chars_of_summary(self):
        """(c) >120-char summary → normalized summary truncated to 120 chars (category not counted)."""
        long_summary = 'A' * 50 + ' ' + 'B' * 80  # 131 chars total; after collapse: 131
        esc = _make_l1_esc(category='infra_issue', summary=long_summary)
        sig = Harness._reblock_signature(esc)
        normalized = ' '.join(long_summary.split()).lower()
        # The summary part must be truncated to 120 chars
        assert sig == f'infra_issue:{normalized[:120]}'
        # Sanity: the suffix part (after 'infra_issue:') is exactly 120 chars
        suffix = sig[len('infra_issue:'):]
        assert len(suffix) == 120

    def test_none_summary_treated_as_empty(self):
        """Edge case: summary=None → treat as empty string (no crash)."""
        # Monkeypatch summary to None to simulate missing field
        esc_no_summary = Escalation(
            id='esc-1-1',
            task_id='1',
            agent_role='workflow',
            severity='blocking',
            category='infra_issue',
            summary=None,  # type: ignore[arg-type]
            level=1,
        )
        sig = Harness._reblock_signature(esc_no_summary)
        assert sig == 'infra_issue:'


# ---------------------------------------------------------------------------
# Shared state helpers for fake in-memory task backend
# ---------------------------------------------------------------------------


def _make_fake_scheduler(harness, initial_metadata: dict | None = None):
    """Wire harness.scheduler with a simple in-memory task backend.

    Returns (persisted_metadata, call_order) so tests can inspect state.
    The backend simulates:
      - get_task: returns the current persisted_metadata
      - update_task: applies update (with optional append/merge semantics)
      - set_task_status: records the call in call_order
    """
    persisted_metadata: dict = dict(initial_metadata or {})
    call_order: list[str] = []

    async def fake_get_task(tid):
        return {'id': tid, 'metadata': dict(persisted_metadata)}

    async def fake_update_task(tid, md, *, append=False, metadata_mode=None):
        if isinstance(md, dict):
            persisted_metadata.update(md)
        call_order.append('update_task')
        return True

    async def fake_set_task_status(tid, status, **kwargs):
        call_order.append('set_task_status')

    harness.scheduler.get_task = fake_get_task
    harness.scheduler.update_task = fake_update_task
    harness.scheduler.set_task_status = fake_set_task_status

    return persisted_metadata, call_order


# ---------------------------------------------------------------------------
# Step-3 / Step-4: Below-threshold flips increment counter and proceed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBelowThresholdFlips:
    """Three same-signature re-pends proceed; counter persisted before each flip."""

    async def test_three_same_signature_flips_increment_counter_and_proceed(
        self, harness: Harness
    ):
        """Counter increments 1→2→3 across three flips; all flip to 'pending'.

        Verifies:
        - After each flip the persisted reblock_guard = {count: N, signature: sig}
        - set_task_status('pending') awaited on all three flips
        - update_task called BEFORE set_task_status on each flip (crash-safe ordering)
        """
        task_id = '42'
        esc = _make_l1_esc(
            task_id=task_id,
            category='infra_issue',
            summary='disk full',
            resolved_by='l2-cascade:esc-100-1',
        )
        expected_sig = Harness._reblock_signature(esc)

        persisted_metadata, call_order = _make_fake_scheduler(harness)

        # Drive three flips via _on_escalation_resolved
        for expected_count in range(1, 4):
            call_order.clear()
            harness._on_escalation_resolved(esc)
            await asyncio.gather(*list(harness._background_tasks))

            # Counter should have been persisted
            guard = persisted_metadata.get('reblock_guard')
            assert guard is not None, f'flip {expected_count}: reblock_guard not persisted'
            assert guard['count'] == expected_count, (
                f'flip {expected_count}: expected count={expected_count}, got {guard["count"]}'
            )
            assert guard['signature'] == expected_sig, (
                f'flip {expected_count}: signature mismatch'
            )

            # Ordering: update_task must precede set_task_status (crash-safe)
            assert call_order == ['update_task', 'set_task_status'], (
                f'flip {expected_count}: expected update_task before set_task_status, '
                f'got {call_order}'
            )


# ---------------------------------------------------------------------------
# Step-5 / Step-6: Signature change resets counter to 1
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSignatureReset:
    """A different-signature escalation resets the counter to 1 (not 0, not 3)."""

    async def test_signature_change_resets_count_to_1_and_proceeds(
        self, harness: Harness
    ):
        """Seed count=2 with old signature; drive with a different-signature esc.

        Assert: persisted count becomes 1 (reset, NOT 3) with the NEW signature,
        and set_task_status('pending') is awaited (flip proceeds).
        """
        task_id = '42'
        old_sig = 'infra_issue:old error'
        new_esc = _make_l1_esc(
            task_id=task_id,
            category='task_failure',  # different category → different signature
            summary='completely different failure',
            resolved_by='l2-cascade:esc-100-1',
        )
        new_sig = Harness._reblock_signature(new_esc)
        assert new_sig != old_sig, 'test requires signatures to differ'

        # Seed persisted state: count=2 with OLD signature
        persisted_metadata, call_order = _make_fake_scheduler(
            harness,
            initial_metadata={'reblock_guard': {'count': 2, 'signature': old_sig}},
        )
        set_task_status_calls: list[str] = []

        original_fake_set = harness.scheduler.set_task_status

        async def recording_set_task_status(task_id, status, **kwargs):
            set_task_status_calls.append(status)
            await original_fake_set(task_id, status, **kwargs)

        harness.scheduler.set_task_status = recording_set_task_status

        harness._on_escalation_resolved(new_esc)
        await asyncio.gather(*list(harness._background_tasks))

        guard = persisted_metadata.get('reblock_guard')
        assert guard is not None, 'reblock_guard should be persisted'
        assert guard['count'] == 1, (
            f'Expected count=1 (reset on signature change), got {guard["count"]}'
        )
        assert guard['signature'] == new_sig, 'Signature should be updated to new_sig'

        # Flip must proceed (different signature → always proceed)
        assert 'pending' in set_task_status_calls, (
            'set_task_status(pending) should have been called (flip proceeds on reset)'
        )


# ---------------------------------------------------------------------------
# Step-7 / Step-8: Threshold trip — 4th same-signature flip withheld + L2
# ---------------------------------------------------------------------------


def _make_mock_queue(*, pending_l2_root_cause: str | None = None):
    """Build a minimal mock EscalationQueue for threshold tests.

    Exposes make_id, submit, find_pending_l2_by_root_cause, get.
    submitted_escalations is a list of submitted Escalation objects.

    q.get() returns None (no parent L2) so _resolve_escalation_action falls
    back to the legacy mapping (resolved → 'resume').
    """
    submitted: list = []

    def make_id(task_id: str) -> str:
        return f'esc-{task_id}-L2-1'

    def submit(esc):
        submitted.append(esc)

    def find_pending_l2_by_root_cause(root_cause: str) -> str | None:
        return pending_l2_root_cause

    q = MagicMock()
    q.make_id = make_id
    q.submit = submit
    q.find_pending_l2_by_root_cause = find_pending_l2_by_root_cause
    q._submitted = submitted
    # get() must return None so _resolve_escalation_action falls back to the
    # legacy resolved→resume mapping (not an unrecognised MagicMock action).
    q.get = MagicMock(return_value=None)
    return q


@pytest.mark.asyncio
class TestThresholdTrip:
    """4th same-signature flip is withheld; a born-at-L2 escalation is filed."""

    async def test_fourth_flip_withheld_and_l2_filed(
        self, harness: Harness, caplog
    ):
        """Seed count=3 + same signature → withhold + born-at-L2.

        Asserts:
        - set_task_status('pending') is NOT called
        - Exactly one L2 submitted with severity='urgent', category='task_failure',
          level=2, agent_role='harness-reblock-guard'
        - Summary == 'persistent re-block: 3 redispatches, signature <sig>'
        - WARNING logged
        """
        task_id = '42'
        esc = _make_l1_esc(
            task_id=task_id,
            category='infra_issue',
            summary='disk full',
            resolved_by='l2-cascade:esc-100-1',
        )
        sig = Harness._reblock_signature(esc)

        # Wire mock queue so the L2 is filed
        q = _make_mock_queue()
        harness._escalation_queue = q

        # Seed: count=3 with same signature (prev_count >= threshold)
        persisted_metadata, _ = _make_fake_scheduler(
            harness,
            initial_metadata={'reblock_guard': {'count': 3, 'signature': sig}},
        )
        set_status_calls: list = []

        original_set = harness.scheduler.set_task_status

        async def spy_set(task_id, status, **kw):
            set_status_calls.append(status)
            await original_set(task_id, status, **kw)

        harness.scheduler.set_task_status = spy_set

        with caplog.at_level(logging.WARNING):
            harness._on_escalation_resolved(esc)
            await asyncio.gather(*list(harness._background_tasks))

        # Flip must be WITHHELD
        assert set_status_calls == [], (
            f'set_task_status should NOT be called at threshold; got {set_status_calls}'
        )

        # Exactly one L2 filed
        assert len(q._submitted) == 1, (
            f'Expected 1 L2 submitted, got {len(q._submitted)}'
        )
        filed = q._submitted[0]
        assert filed.severity == 'urgent'
        assert filed.category == 'task_failure'
        assert filed.level == 2
        assert filed.agent_role == 'harness-reblock-guard'
        expected_summary = f'persistent re-block: 3 redispatches, signature {sig}'
        assert filed.summary == expected_summary, (
            f'Expected summary {expected_summary!r}, got {filed.summary!r}'
        )

        # WARNING should be logged
        assert any(r.levelno >= logging.WARNING for r in caplog.records), (
            'Expected a WARNING record for threshold trip'
        )


# ---------------------------------------------------------------------------
# Step-9 / Step-10: Born-at-L2 dedup + human reset
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDedupAndHumanReset:
    """Dedup prevents double-filing; clearing the guard allows the loop to resume."""

    async def test_second_withhold_does_not_file_duplicate_l2(
        self, harness: Harness
    ):
        """(a) Dedup: with a pending L2 already existing for this task, no second L2 filed.

        Configure mock queue with find_pending_l2_by_root_cause returning an existing
        id (simulating a pending L2), then drive a second withhold and assert NO
        second escalation is submitted.
        """
        task_id = '42'
        esc = _make_l1_esc(
            task_id=task_id,
            category='infra_issue',
            summary='disk full',
            resolved_by='l2-cascade:esc-100-1',
        )
        sig = Harness._reblock_signature(esc)

        # Queue already has a pending L2 for this task
        q = _make_mock_queue(pending_l2_root_cause='esc-42-L2-1')
        harness._escalation_queue = q

        # Seed: count=3 (at threshold)
        _make_fake_scheduler(
            harness,
            initial_metadata={'reblock_guard': {'count': 3, 'signature': sig}},
        )

        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        # No second L2 should be submitted
        assert len(q._submitted) == 0, (
            f'Expected 0 new L2 submissions (dedup), got {len(q._submitted)}'
        )

    async def test_human_reset_allows_loop_to_resume(self, harness: Harness):
        """(b) Human reset: clearing reblock_guard from metadata → next flip proceeds.

        Seed metadata with no reblock_guard (simulating human clearance) and
        drive a flip with the previously-tripping signature. Assert:
        - persisted reblock_guard becomes {count: 1}
        - set_task_status('pending') is called (loop resumes)
        """
        task_id = '42'
        esc = _make_l1_esc(
            task_id=task_id,
            category='infra_issue',
            summary='disk full',
            resolved_by='l2-cascade:esc-100-1',
        )

        # No queue (no L2 should be filed — this is just a normal flip)
        harness._escalation_queue = None

        # Seed: no reblock_guard (human cleared it)
        persisted_metadata, call_order = _make_fake_scheduler(
            harness,
            initial_metadata={},
        )
        set_status_calls: list = []
        original_set = harness.scheduler.set_task_status

        async def recording_set(task_id, status, **kw):
            set_status_calls.append(status)
            await original_set(task_id, status, **kw)

        harness.scheduler.set_task_status = recording_set

        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        guard = persisted_metadata.get('reblock_guard')
        assert guard is not None, 'reblock_guard should be persisted'
        assert guard['count'] == 1, (
            f'Expected count=1 (reset after human clear), got {guard["count"]}'
        )

        # Flip must proceed
        assert 'pending' in set_status_calls, (
            'set_task_status(pending) should be called after human reset'
        )


# ---------------------------------------------------------------------------
# Step-11 / Step-12: Metadata-clobber boundary (C3.4)
# ---------------------------------------------------------------------------


def _make_clobber_fake_scheduler(harness, initial_metadata: dict):
    """Wire harness.scheduler with a fake backend that distinguishes append semantics.

    This backend simulates real database behaviour:
      - update_task(tid, md, append=False)  → REPLACE whole metadata blob with md
        (shallow replace — siblings are clobbered; this is the bad path that
        step-11 must catch when append=True is absent).
      - update_task(tid, md, append=True)   → RECURSIVE MERGE: only the keys in
        md are merged into the existing blob; siblings survive.
      - set_task_status                     → AUDIT MERGE: merges in an extra
        'reopen_reason' key (like the real interceptor) and updates status,
        preserving all other existing keys.

    Returns (task_state, call_order) where task_state is a mutable dict holding
    ``{'metadata': {...}, 'status': 'blocked'}`` so tests can inspect both.
    """

    def _deep_merge(base: dict, overlay: dict) -> dict:
        """Recursive merge: overlay keys replace / extend base keys."""
        result = dict(base)
        for k, v in overlay.items():
            if isinstance(v, dict) and isinstance(result.get(k), dict):
                result[k] = _deep_merge(result[k], v)
            else:
                result[k] = v
        return result

    task_state: dict = {
        'id': '42',
        'status': 'blocked',
        'metadata': dict(initial_metadata),
    }
    call_order: list[str] = []

    async def fake_get_task(tid):
        return {
            'id': tid,
            'status': task_state['status'],
            'metadata': dict(task_state['metadata']),
        }

    async def fake_get_status(tid):
        return task_state['status']

    async def fake_update_task(tid, md, *, append=False, metadata_mode=None):
        """Simulates the #1827 metadata_mode contract.

        metadata_mode='merge': shallow last-write-wins, siblings preserved.
        metadata_mode='additive' / append=True: recursive deep merge.
        metadata_mode='replace' / default: whole-blob replace (clobber hazard).
        """
        # Resolve effective mode: metadata_mode > append True→additive > default
        if metadata_mode == 'merge':
            task_state['metadata'] = {**task_state['metadata'], **md}
        elif metadata_mode == 'additive' or (metadata_mode is None and append):
            task_state['metadata'] = _deep_merge(task_state['metadata'], md)
        else:
            # replace: simulate the clobber hazard (metadata_mode='replace' or default)
            task_state['metadata'] = dict(md)
        call_order.append('update_task')
        return True

    async def fake_set_task_status(tid, status, **kwargs):
        # Simulate the reopen_reason audit-merge the real interceptor performs.
        # It merges extra audit keys into the existing metadata (preserves siblings).
        existing = dict(task_state['metadata'])
        if status == 'pending' and task_state['status'] == 'blocked':
            existing.setdefault('reopen_reason', 'guard-test')
        task_state['metadata'] = existing
        task_state['status'] = status
        call_order.append('set_task_status')

    harness.scheduler.get_task = fake_get_task
    harness.scheduler.get_status = fake_get_status
    harness.scheduler.update_task = fake_update_task
    harness.scheduler.set_task_status = fake_set_task_status

    return task_state, call_order


@pytest.mark.asyncio
class TestMetadataClobberBoundary:
    """C3.4: counter survives a full blocked→pending→in-progress→blocked cycle.

    This test FAILS if _check_reblock_guard calls update_task WITHOUT append=True
    because the fake backend then replaces the whole metadata blob with just
    {'reblock_guard': {...}}, clobbering sibling keys ('files', 'memory_hints').

    After step-12 adds append=True the recursive-merge path preserves siblings
    and the test passes.
    """

    async def test_counter_survives_full_cycle_and_siblings_preserved(
        self, harness: Harness
    ):
        """Counter continues 1→2 across incarnations; sibling metadata not clobbered.

        Scenario:
          1. Seed: metadata={'files':['foo.py'], 'memory_hints':['hint1']}
          2. 1st flip (blocked→pending): counter persisted as count=1
          3. Simulate workflow cycle: status in-progress → blocked
             (the flip's set_task_status + subsequent in-progress/blocked writes
              must not clobber reblock_guard or sibling keys)
          4. 2nd flip (blocked→pending): counter persisted as count=2

        Asserts:
          - reblock_guard.count == 2 (counter continues, not reset/lost)
          - sibling keys 'files' and 'memory_hints' still present
        """
        task_id = '42'
        esc = _make_l1_esc(
            task_id=task_id,
            category='infra_issue',
            summary='disk full',
            resolved_by='l2-cascade:esc-100-1',
        )
        expected_sig = Harness._reblock_signature(esc)

        task_state, call_order = _make_clobber_fake_scheduler(
            harness,
            initial_metadata={
                'files': ['foo.py', 'bar.py'],
                'memory_hints': ['hint1', 'hint2'],
            },
        )

        # ── 1st flip: blocked → pending ──────────────────────────────────────
        call_order.clear()
        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        # Counter persisted
        guard = task_state['metadata'].get('reblock_guard')
        assert guard is not None, '1st flip: reblock_guard not persisted'
        assert guard['count'] == 1, f'1st flip: expected count=1, got {guard["count"]}'
        assert guard['signature'] == expected_sig, '1st flip: signature mismatch'

        # Flip proceeded
        assert task_state['status'] == 'pending', (
            f'1st flip: expected status=pending, got {task_state["status"]}'
        )

        # ── Simulate workflow cycle: pending → in-progress → blocked ─────────
        # The harness/workflow engine would call set_task_status to mark progress
        # and then re-block when a new failure occurs.
        task_state['status'] = 'in-progress'
        # Simulate a new blocking event: status goes back to blocked.
        task_state['status'] = 'blocked'
        # The scheduler's get_status must reflect the new blocked status.
        # (fake_get_status already reads from task_state)

        # ── 2nd flip: blocked → pending ──────────────────────────────────────
        call_order.clear()
        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        guard = task_state['metadata'].get('reblock_guard')
        assert guard is not None, '2nd flip: reblock_guard missing after cycle'
        assert guard['count'] == 2, (
            f'2nd flip: expected count=2 (counter continues), got {guard["count"]}'
        )
        assert guard['signature'] == expected_sig, '2nd flip: signature mismatch'

        # Flip proceeded again
        assert task_state['status'] == 'pending', (
            f'2nd flip: expected status=pending, got {task_state["status"]}'
        )

        # ── Critical: sibling metadata keys must survive the full cycle ───────
        meta = task_state['metadata']
        assert 'files' in meta, (
            "Sibling key 'files' was clobbered — update_task must use append=True"
        )
        assert meta['files'] == ['foo.py', 'bar.py'], (
            f"'files' value corrupted: {meta['files']!r}"
        )
        assert 'memory_hints' in meta, (
            "Sibling key 'memory_hints' was clobbered — update_task must use append=True"
        )
        assert meta['memory_hints'] == ['hint1', 'hint2'], (
            f"'memory_hints' value corrupted: {meta['memory_hints']!r}"
        )


# ---------------------------------------------------------------------------
# Amendment 3a: Orphan/direct re-pend branch exercises the same guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestOrphanBranchGuard:
    """Guard is exercised identically via the orphan/direct re-pend branch.

    The orphan path in _on_escalation_resolved fires when:
      - escalation.level >= 1
      - resolved_by does NOT start with 'l2-cascade:'
      - escalation.task_id NOT in _escalation_events

    Both the l2-cascade and orphan paths call _cascade_unblock_member, so
    the guard must behave identically regardless of which branch delivered
    the escalation.
    """

    async def test_orphan_branch_increments_counter_and_proceeds(
        self, harness: Harness
    ):
        """Orphan escalation (not l2-cascade) increments guard and flips to pending."""
        task_id = '42'
        # resolved_by does NOT start with 'l2-cascade:' → orphan/direct path
        esc = _make_l1_esc(
            task_id=task_id,
            category='infra_issue',
            summary='disk full',
            resolved_by='operator',      # orphan — not a cascade member
            level=1,
        )
        expected_sig = Harness._reblock_signature(esc)

        # task_id must not be in _escalation_events (orphan condition)
        assert task_id not in harness._escalation_events

        persisted_metadata, call_order = _make_fake_scheduler(harness)
        set_status_calls: list[str] = []
        original_set = harness.scheduler.set_task_status

        async def spy_set(task_id, status, **kw):
            set_status_calls.append(status)
            await original_set(task_id, status, **kw)

        harness.scheduler.set_task_status = spy_set

        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        # Counter persisted with count=1
        guard = persisted_metadata.get('reblock_guard')
        assert guard is not None, 'orphan path: reblock_guard not persisted'
        assert guard['count'] == 1, (
            f'orphan path: expected count=1, got {guard["count"]}'
        )
        assert guard['signature'] == expected_sig

        # Flip must proceed
        assert 'pending' in set_status_calls, (
            'orphan path: set_task_status(pending) should have been called'
        )

    async def test_orphan_branch_withholds_at_threshold(
        self, harness: Harness, caplog
    ):
        """Orphan escalation is withheld at threshold just like cascade path."""
        task_id = '42'
        esc = _make_l1_esc(
            task_id=task_id,
            category='infra_issue',
            summary='disk full',
            resolved_by='sweep-watcher',  # not l2-cascade → orphan path
            level=1,
        )
        sig = Harness._reblock_signature(esc)

        q = _make_mock_queue()
        harness._escalation_queue = q

        # Seed: count=3 (at threshold)
        _make_fake_scheduler(
            harness,
            initial_metadata={'reblock_guard': {'count': 3, 'signature': sig}},
        )
        set_status_calls: list = []

        async def spy_set(task_id, status, **kw):
            set_status_calls.append(status)

        harness.scheduler.set_task_status = spy_set

        with caplog.at_level(logging.WARNING):
            harness._on_escalation_resolved(esc)
            await asyncio.gather(*list(harness._background_tasks))

        # Flip must be withheld
        assert set_status_calls == [], (
            f'orphan path: flip should be withheld at threshold; got {set_status_calls}'
        )
        # L2 must be filed
        assert len(q._submitted) == 1, (
            f'orphan path: expected 1 L2 submitted, got {len(q._submitted)}'
        )


# ---------------------------------------------------------------------------
# Amendment 3b: Corrupt-metadata robustness (non-dict reblock_guard)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCorruptMetadataRobustness:
    """A corrupt (non-dict) reblock_guard degrades gracefully — no AttributeError.

    If metadata.reblock_guard is a truthy non-dict value (e.g. a string from
    a bad write or manual edit), the guard must treat it as absent (count=0)
    rather than propagating an AttributeError that would abort the flip path.
    """

    async def test_string_reblock_guard_resets_to_count_1_and_proceeds(
        self, harness: Harness
    ):
        """Non-dict reblock_guard value → treated as absent → count resets to 1."""
        task_id = '42'
        esc = _make_l1_esc(
            task_id=task_id,
            category='infra_issue',
            summary='disk full',
            resolved_by='l2-cascade:esc-100-1',
        )

        # Seed a corrupt string value for reblock_guard
        persisted_metadata, call_order = _make_fake_scheduler(
            harness,
            initial_metadata={'reblock_guard': 'corrupt-string-value'},
        )
        set_status_calls: list[str] = []
        original_set = harness.scheduler.set_task_status

        async def spy_set(task_id, status, **kw):
            set_status_calls.append(status)
            await original_set(task_id, status, **kw)

        harness.scheduler.set_task_status = spy_set

        # Must not raise; should proceed with count=1
        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        guard = persisted_metadata.get('reblock_guard')
        assert guard is not None, 'reblock_guard should be re-persisted after corrupt value'
        assert isinstance(guard, dict), (
            f'Expected dict, got {type(guard).__name__}: {guard!r}'
        )
        assert guard['count'] == 1, (
            f'Corrupt guard should reset to count=1, got {guard["count"]}'
        )

        # Flip must proceed (corrupt counter = absent = start from 0)
        assert 'pending' in set_status_calls, (
            'Flip should proceed when guard was corrupt (treated as absent)'
        )

    async def test_integer_reblock_guard_resets_and_proceeds(
        self, harness: Harness
    ):
        """An integer (not a dict) reblock_guard is also treated as absent."""
        task_id = '42'
        esc = _make_l1_esc(
            task_id=task_id,
            category='task_failure',
            summary='worker crashed',
            resolved_by='l2-cascade:esc-200-1',
        )

        persisted_metadata, _ = _make_fake_scheduler(
            harness,
            initial_metadata={'reblock_guard': 99},  # integer, not a dict
        )
        set_status_calls: list[str] = []
        original_set = harness.scheduler.set_task_status

        async def spy_set(task_id, status, **kw):
            set_status_calls.append(status)
            await original_set(task_id, status, **kw)

        harness.scheduler.set_task_status = spy_set

        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        guard = persisted_metadata.get('reblock_guard')
        assert isinstance(guard, dict), (
            f'Guard should be reset to a dict; got {type(guard).__name__}: {guard!r}'
        )
        assert guard['count'] == 1, (
            f'Integer guard should reset to count=1, got {guard["count"]}'
        )
        assert 'pending' in set_status_calls, (
            'Flip should proceed when guard was a plain integer (treated as absent)'
        )


# ---------------------------------------------------------------------------
# step-6 (1828): _check_reblock_guard wire-mode assertion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReblockGuardWireMode:
    """Guard update_task wire call must forward metadata_mode='merge'.

    The guard writes {'reblock_guard': {count, signature}}, a full sub-dict.
    Under additive (_merge_values: recursive, scalar OLD-wins), the count
    collision across incarnations would keep the OLD count — freezing the
    counter so the re-block threshold never trips.  Shallow merge overwrites
    the whole reblock_guard key wholesale (new count wins) while preserving
    sibling metadata keys.

    RED today: _check_reblock_guard calls update_task(append=True) →
    after step-5, wrapper maps that to metadata_mode='additive', not 'merge'.
    GREEN after step-7 impl (switch append=True → metadata_mode='merge').
    """

    async def test_reblock_guard_increment_forwards_merge_mode(
        self, harness: Harness, monkeypatch
    ):
        """Guard increment path must forward metadata_mode='merge' on the wire."""
        task_id = '42'
        esc = _make_l1_esc(
            task_id=task_id,
            category='infra_issue',
            summary='disk full',
        )

        # Seed get_task to return an existing guard + sibling keys so the
        # increment path (prev_count + 1) is exercised.
        existing_guard = {'count': 1, 'signature': Harness._reblock_signature(esc)}
        harness.scheduler.get_task = AsyncMock(return_value={
            'id': task_id,
            'metadata': {
                'reblock_guard': existing_guard,
                'files': ['foo.py'],
                'memory_hints': ['hint1'],
            },
        })

        # Capture update_task wire calls via a REAL Scheduler + monkeypatched mcp_call.
        captured_payloads: list[dict] = []

        async def mock_mcp_call(url, method, payload, **kwargs):
            captured_payloads.append(payload)
            return {}

        real_scheduler = Scheduler(OrchestratorConfig(max_per_module=1))
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock_mcp_call)

        # Replace only update_task; get_task/get_status/set_task_status stay on mock.
        harness.scheduler.update_task = real_scheduler.update_task

        result = await harness._check_reblock_guard(esc, task_id)

        assert result is True, 'Guard should allow the flip (count=2 < threshold=3)'

        update_wire_calls = [
            p for p in captured_payloads if p.get('name') == 'update_task'
        ]
        assert len(update_wire_calls) == 1, (
            f'Expected 1 update_task wire call; got {len(update_wire_calls)} '
            f'(all: {[p.get("name") for p in captured_payloads]})'
        )
        arguments = update_wire_calls[0]['arguments']
        assert arguments.get('metadata_mode') == 'merge', (
            f"_check_reblock_guard must forward metadata_mode='merge' so the "
            f"whole reblock_guard sub-dict is overwritten wholesale (new count "
            f"wins) and sibling keys are preserved; got: {arguments}"
        )
        assert 'append' not in arguments, (
            f"'append' key must not appear on the wire; got: {arguments}"
        )
