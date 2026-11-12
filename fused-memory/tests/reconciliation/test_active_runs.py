"""RED tests for ActiveRunRegistry (fused_memory.reconciliation.active_runs).

The registry is the in-process source of the /health ``recon_busy`` signal
(task 2703 δ). ``run_full_cycle`` wraps its stage loop in
``with self._active_runs.track(run_id, project_id, started_at) as active:``
so a full reconciliation cycle in flight is visible as a structured,
machine-readable entry — and, critically, is cleared on EVERY exit path
(normal return, ``Exception``, and the ``CancelledError`` that a
timeout/shutdown raises mid-stage; survey E1), so a restart cancelled
mid-cycle cannot leak a phantom-busy entry.
"""
from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from fused_memory.models.reconciliation import StageReport
from fused_memory.reconciliation.active_runs import ActiveRunRegistry
from fused_memory.reconciliation.stages.memory_consolidator import MemoryConsolidator
from reconciliation.harness_fixtures import make_harness_config, open_harness, stub_stage_runs

RUN_ID = "run-abc"
PROJECT_ID = "dark_factory"
STARTED_AT = "2026-07-18T06:00:00+00:00"


def test_empty_registry_snapshot_is_empty_list():
    reg = ActiveRunRegistry()
    assert reg.snapshot() == []


def test_enter_adds_entry_with_stage_none():
    reg = ActiveRunRegistry()
    with reg.track(RUN_ID, PROJECT_ID, STARTED_AT):
        snap = reg.snapshot()
        assert len(snap) == 1
        assert snap[0] == {
            "project_id": PROJECT_ID,
            "run_id": RUN_ID,
            "stage": None,
            "started_at": STARTED_AT,
        }


def test_handle_stage_updates_entry_stage():
    reg = ActiveRunRegistry()
    with reg.track(RUN_ID, PROJECT_ID, STARTED_AT) as active:
        active.stage("stage1_memory_consolidation")
        assert reg.snapshot()[0]["stage"] == "stage1_memory_consolidation"
        active.stage("stage2_task_knowledge_sync")
        assert reg.snapshot()[0]["stage"] == "stage2_task_knowledge_sync"


def test_exit_removes_entry():
    reg = ActiveRunRegistry()
    with reg.track(RUN_ID, PROJECT_ID, STARTED_AT):
        assert len(reg.snapshot()) == 1
    assert reg.snapshot() == []


def test_exit_clears_entry_on_exception_and_propagates():
    reg = ActiveRunRegistry()
    with pytest.raises(ValueError), reg.track(RUN_ID, PROJECT_ID, STARTED_AT):
        assert len(reg.snapshot()) == 1
        raise ValueError("boom")
    assert reg.snapshot() == []


def test_exit_clears_entry_on_cancelled_error_and_propagates():
    """The timeout/shutdown path raises ``asyncio.CancelledError`` (a
    ``BaseException``, NOT an ``Exception``) mid-stage; the entry must still
    clear so a restart cancelled mid-cycle cannot leak a phantom-busy entry
    (survey E1)."""
    reg = ActiveRunRegistry()
    with pytest.raises(asyncio.CancelledError), reg.track(RUN_ID, PROJECT_ID, STARTED_AT):
        assert len(reg.snapshot()) == 1
        raise asyncio.CancelledError()
    assert reg.snapshot() == []


def test_two_concurrent_runs_appear_and_clear_independently():
    reg = ActiveRunRegistry()
    with reg.track("run-1", "proj-1", STARTED_AT):
        with reg.track("run-2", "proj-2", STARTED_AT):
            snap = reg.snapshot()
            # Both present, in insertion order.
            assert [e["run_id"] for e in snap] == ["run-1", "run-2"]
        # Inner exited: only the outer run remains.
        assert [e["run_id"] for e in reg.snapshot()] == ["run-1"]
    assert reg.snapshot() == []


def test_snapshot_returns_json_serializable_independent_copies():
    reg = ActiveRunRegistry()
    with reg.track(RUN_ID, PROJECT_ID, STARTED_AT) as active:
        active.stage("stage1_memory_consolidation")
        snap = reg.snapshot()
        # Plain JSON-serializable dicts (no datetime/objects).
        json.dumps(snap)
        # Independent copies: mutating a returned dict must not alter state.
        snap[0]["stage"] = "MUTATED"
        snap[0]["extra"] = "x"
        again = reg.snapshot()
        assert again[0]["stage"] == "stage1_memory_consolidation"
        assert "extra" not in again[0]


# ---------------------------------------------------------------------------
# Integration: the REAL ReconciliationHarness must open the registry while
# run_full_cycle runs and clear it on every exit path.
#
# The unit tests above exercise ActiveRunRegistry in isolation, and
# tests/test_health_endpoint.py checks /health's recon_busy field against a
# stub harness. Neither drives the actual seam — that run_full_cycle wraps its
# stage loop in ``with self._active_runs.track(...)`` and calls
# ``_active.stage(current_stage_name)`` as it advances — so a regression that
# dropped the wrapper or the per-stage update would pass every other test.
# These two tests guard that seam end to end (task 2703 δ amendment). The judge
# is configured off: recon_busy scopes to the stage pipeline only, so the
# post-cycle judge task is out of scope here.
# ---------------------------------------------------------------------------


def _open_cycle_harness(tmp_path):
    return open_harness(
        tmp_path,
        taskmaster=AsyncMock(),
        config=make_harness_config(judge_enabled=False),
        known_projects={"test-project": "/tmp/test-project"},
    )


def _empty_report(stage_id) -> StageReport:
    now = datetime.now(UTC)
    return StageReport(stage=stage_id, started_at=now, completed_at=now)


@pytest.mark.asyncio
async def test_run_full_cycle_populates_registry_per_stage_and_clears_on_return(
    tmp_path, monkeypatch
):
    """run_full_cycle surfaces exactly the in-flight run via
    recon_busy_snapshot() with the correct run_id/project_id/started_at, the
    stage advances through the pipeline, and the entry is gone once the cycle
    returns."""
    async with _open_cycle_harness(tmp_path) as harness:
        snapshots: list[list[dict]] = []

        async def capture(
            self, events, watermark, prior_reports, run_id, model=None, resume_session_id=None,
        ):
            snapshots.append(harness.recon_busy_snapshot())
            return _empty_report(self.stage_id)

        stub_stage_runs(monkeypatch, capture)

        run = await harness.run_full_cycle("test-project", "test-trigger")

        # One snapshot captured at the top of each of the three stages.
        assert len(snapshots) == 3
        for snap in snapshots:
            assert len(snap) == 1, "exactly the one in-flight full cycle is tracked"
            assert snap[0]["run_id"] == run.id
            assert snap[0]["project_id"] == "test-project"
            assert snap[0]["started_at"] == run.started_at.isoformat()
        # _active.stage(current_stage_name) advances as the pipeline walks stages.
        assert [snap[0]["stage"] for snap in snapshots] == [
            "memory_consolidator",
            "task_knowledge_sync",
            "integrity_check",
        ]
        # Cleared once the cycle returns — no phantom-busy entry lingers.
        assert harness.recon_busy_snapshot() == []


@pytest.mark.asyncio
async def test_run_full_cycle_clears_registry_when_cancelled_mid_stage(tmp_path, monkeypatch):
    """A cycle cancelled mid-stage (the timeout/shutdown path, survey E1) must
    still clear its registry entry, so a restart cancelled mid-cycle cannot
    leak a phantom-busy run."""
    async with _open_cycle_harness(tmp_path) as harness:
        mid_run: list[list[dict]] = []

        async def capture_then_cancel_stage_1(
            self, events, watermark, prior_reports, run_id, model=None, resume_session_id=None,
        ):
            mid_run.append(harness.recon_busy_snapshot())
            if isinstance(self, MemoryConsolidator):
                raise asyncio.CancelledError()
            return _empty_report(self.stage_id)

        stub_stage_runs(monkeypatch, capture_then_cancel_stage_1)

        # Patch the finally-block Stage-1 backstop to a no-op so the cancelled path
        # stays fast and side-effect-free; we are asserting registry cleanup only.
        # return_value=True mirrors the real bool return, which the backstop stamps
        # onto run.stage_reports['_error'] before that dict is JSON-persisted.
        with patch(
            "fused_memory.reconciliation.harness.write_stage1_cycle_summary",
            AsyncMock(return_value=True),
        ), pytest.raises(asyncio.CancelledError):
            await harness.run_full_cycle("test-project", "test-trigger")

        # The entry WAS live mid-cycle (Stage 1)...
        assert len(mid_run) == 1
        assert mid_run[0][0]["stage"] == "memory_consolidator"
        # ...and is cleared despite the mid-stage cancellation.
        assert harness.recon_busy_snapshot() == []
