"""Tests for ReviewCheckpoint.should_run_full — the run-forever full-review ceiling.

Step 2 of the run-forever orchestrator: the per-idle-cycle full review is gated by
a true AND of a wall-clock interval and a completed-task count, so a queue that
drains and refills repeatedly can't trigger a fresh (expensive) full review on
every idle cycle.  ``run_full()`` records both trackers so the gate re-closes.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import pydantic_spec

from orchestrator.config import OrchestratorConfig
from orchestrator.review_checkpoint import ReviewCheckpoint, ReviewReport
from orchestrator.verify import VerifyResult

_PHASE1 = VerifyResult(
    passed=True, summary='OK', test_output='', lint_output='', type_output=''
)


def _make_checkpoint(
    min_interval: float = 86400.0, min_tasks: int = 20
) -> ReviewCheckpoint:
    config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    config.project_root = Path('/tmp')
    config.review.full_review_min_interval_secs = min_interval
    config.review.full_review_min_tasks = min_tasks
    mcp = MagicMock()
    mcp.mcp_config_json.return_value = {'mcpServers': {}}
    return ReviewCheckpoint(config, mcp=mcp, usage_gate=None)


class TestShouldRunFullGate:
    def test_first_review_time_gate_open_but_task_gate_requires_20(self):
        """last_time=None opens the time gate, but the 20-task gate still blocks."""
        cp = _make_checkpoint()
        assert cp._last_full_review_time is None

        # 0 merges → task gate closed even though time gate is open.
        assert cp.should_run_full(now=1000.0) is False

        # 19 merges → still one short.
        cp._merge_count = 19
        assert cp.should_run_full(now=1000.0) is False

        # 20 merges → both gates open.
        cp._merge_count = 20
        assert cp.should_run_full(now=1000.0) is True

    def test_time_gate_blocks_until_interval_elapses(self):
        """After a full review, the time gate stays closed until the interval passes."""
        cp = _make_checkpoint(min_interval=86400.0, min_tasks=20)
        # Simulate a prior full review at t=1000 after 20 merges, then 25 more.
        cp._last_full_review_time = 1000.0
        cp._last_full_review_at_merge = 20
        cp._merge_count = 45  # 25 merges since → task gate wide open

        # Only 100s elapsed → time gate closed → AND is False despite task gate.
        assert cp.should_run_full(now=1100.0) is False

        # Exactly the interval elapsed → time gate opens → both open.
        assert cp.should_run_full(now=1000.0 + 86400.0) is True

    def test_task_gate_blocks_even_when_time_gate_open(self):
        """Time gate open (interval long past) but <20 merges since → still False."""
        cp = _make_checkpoint(min_interval=86400.0, min_tasks=20)
        cp._last_full_review_time = 1000.0
        cp._last_full_review_at_merge = 20
        cp._merge_count = 30  # only 10 merges since

        # A full day later — time gate open, task gate closed (10 < 20).
        assert cp.should_run_full(now=1000.0 + 200000.0) is False


@pytest.mark.asyncio
class TestRunFullRecordsTrackers:
    async def test_run_full_records_time_and_merge(self, monkeypatch):
        """run_full() stamps both full-review trackers (separate from focused)."""
        cp = _make_checkpoint()
        cp._merge_count = 25

        cp._run_review = AsyncMock(  # type: ignore[method-assign]
            return_value=ReviewReport(
                review_id='x', mode='full', changed_modules=[], phase1=_PHASE1,
            )
        )

        monkeypatch.setattr(
            'orchestrator.review_checkpoint.time.monotonic', lambda: 5000.0
        )

        await cp.run_full()

        # Full-review trackers recorded.
        assert cp._last_full_review_time == 5000.0
        assert cp._last_full_review_at_merge == 25
        # Existing focused-cadence reset is preserved (not perturbed).
        assert cp._last_review_at_merge == 25

        # The gate now re-closes: no new merges, no elapsed time → False.
        assert cp.should_run_full(now=5000.0) is False
