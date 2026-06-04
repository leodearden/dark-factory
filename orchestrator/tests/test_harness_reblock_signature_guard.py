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

        # Simulate a fake in-memory metadata store that tracks the latest
        # reblock_guard after each update_task call.
        persisted_metadata: dict = {}
        call_order: list[str] = []

        async def fake_get_task(tid):
            return {'id': tid, 'metadata': dict(persisted_metadata)}

        async def fake_update_task(tid, md, *, append=False):
            if append and isinstance(md, dict):
                # Recursive-merge (step-12 semantics; for now just update)
                persisted_metadata.update(md)
            else:
                persisted_metadata.update(md)
            call_order.append('update_task')
            return True

        async def fake_set_task_status(tid, status, **kwargs):
            call_order.append('set_task_status')

        harness.scheduler.get_task = fake_get_task
        harness.scheduler.update_task = fake_update_task
        harness.scheduler.set_task_status = fake_set_task_status

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
