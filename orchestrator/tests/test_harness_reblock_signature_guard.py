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
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from escalation.models import Escalation

from orchestrator.harness import Harness

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
        esc = _make_l1_esc(category='infra_issue', summary='x')
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

    async def fake_update_task(tid, md, *, append=False):
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

        async def recording_set_task_status(tid, status, **kwargs):
            set_task_status_calls.append(status)
            await original_fake_set(tid, status, **kwargs)

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

        async def spy_set(tid, status, **kw):
            set_status_calls.append(status)
            await original_set(tid, status, **kw)

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
