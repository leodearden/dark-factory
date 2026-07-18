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

import pytest

from fused_memory.reconciliation.active_runs import ActiveRunRegistry

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
