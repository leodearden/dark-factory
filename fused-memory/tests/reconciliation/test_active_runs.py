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
import pytest_asyncio

from fused_memory.models.reconciliation import StageReport
from fused_memory.models.scope import ProjectId, ProjectRoot, ProjectScope
from fused_memory.reconciliation.active_runs import ActiveRunRegistry
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.reconciliation.journal import ReconciliationJournal

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
# These two tests guard that seam end to end (task 2703 δ amendment). The
# harness scaffolding mirrors tests/test_harness.py::_make_test_harness /
# _mock_stage_run (re-inlined here because ``tests`` is not an importable
# package — there is no tests/__init__.py).
# ---------------------------------------------------------------------------


def _scope(project_id: str, project_root: str) -> ProjectScope:
    return ProjectScope(ProjectId(project_id), ProjectRoot(project_root))


def _rescope(stages: list, scope: ProjectScope) -> list:
    """Re-scope pinned stage instances in place so the ``_make_stages`` shim
    honors the scope production passes while keeping the mocked ``.run``."""
    for s in stages:
        s.scope = scope
    return stages


@pytest_asyncio.fixture
async def harness_journal(tmp_path):
    j = ReconciliationJournal(tmp_path / "active_runs_journal")
    await j.initialize()
    yield j
    await j.close()


@pytest_asyncio.fixture
async def harness_event_buffer(tmp_path):
    buf = EventBuffer(
        db_path=tmp_path / "active_runs_eb.db",
        buffer_size_threshold=2,
        max_staleness_seconds=3600,
    )
    await buf.initialize()
    yield buf
    await buf.close()


@pytest.fixture
def harness_memory_service():
    svc = AsyncMock()
    svc.search = AsyncMock(return_value=[])
    svc.get_episodes = AsyncMock(return_value=[])
    svc.get_status = AsyncMock(
        return_value={
            "graphiti": {"connected": True},
            "mem0": {"connected": True},
            "projects": {},
        }
    )
    svc.get_entity = AsyncMock(return_value={"nodes": [], "edges": []})
    svc.get_memories_by_metadata = AsyncMock(return_value=[])
    svc.mem0 = AsyncMock()
    svc.mem0.get_all = AsyncMock(return_value={"results": []})
    return svc


def _make_harness(journal, event_buffer, mock_memory_service):
    """Build a ReconciliationHarness on test fixtures with pinned, mockable
    stages. ``judge`` is forced off so run_full_cycle never spawns the
    fire-and-forget judge task — the recon_busy signal deliberately scopes to
    the stage pipeline only, so a post-cycle judge is out of scope here."""
    from fused_memory.config.schema import FusedMemoryConfig, ReconciliationConfig
    from fused_memory.reconciliation.harness import ReconciliationHarness

    config = FusedMemoryConfig(
        reconciliation=ReconciliationConfig(
            enabled=True,
            explore_codebase_root="/tmp/test",
            agent_llm_provider="anthropic",
            agent_llm_model="claude-sonnet-4-20250514",
        )
    )
    harness = ReconciliationHarness(
        memory_service=mock_memory_service,
        taskmaster=AsyncMock(),
        journal=journal,
        event_buffer=event_buffer,
        config=config,
    )
    harness.judge = None
    # task 1143: inject known project so run_full_cycle pre-flight does not raise.
    harness._known_projects = {"test-project": "/tmp/test-project"}
    # Pin stages so mocked .run methods stick across the cycle; the shim
    # re-scopes (rather than rebuilds) on each _make_stages(scope) call.
    harness.stages = harness._make_stages(_scope("test-project", "/tmp/test-project"))
    harness._make_stages = lambda scope, **k: _rescope(harness.stages, scope)
    return harness


def _stub_stage_run(stage, on_run):
    """Replace ``stage.run`` with a mock that awaits ``on_run(stage)`` then
    returns an empty StageReport — lets a test observe registry state at the
    moment each stage fires."""

    async def mock_run(events, watermark, prior_reports, run_id, model=None, _s=stage):
        await on_run(_s)
        return StageReport(
            stage=_s.stage_id,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items_flagged=[],
            stats={},
            llm_calls=0,
            tokens_used=0,
        )

    stage.run = mock_run


@pytest.mark.asyncio
async def test_run_full_cycle_populates_registry_per_stage_and_clears_on_return(
    harness_journal, harness_event_buffer, harness_memory_service
):
    """run_full_cycle surfaces exactly the in-flight run via
    recon_busy_snapshot() with the correct run_id/project_id/started_at, the
    stage advances through the pipeline, and the entry is gone once the cycle
    returns."""
    harness = _make_harness(harness_journal, harness_event_buffer, harness_memory_service)

    snapshots: list[list[dict]] = []

    async def capture(_stage):
        snapshots.append(harness.recon_busy_snapshot())

    for stage in harness.stages:
        _stub_stage_run(stage, capture)

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
async def test_run_full_cycle_clears_registry_when_cancelled_mid_stage(
    harness_journal, harness_event_buffer, harness_memory_service
):
    """A cycle cancelled mid-stage (the timeout/shutdown path, survey E1) must
    still clear its registry entry, so a restart cancelled mid-cycle cannot
    leak a phantom-busy run."""
    harness = _make_harness(harness_journal, harness_event_buffer, harness_memory_service)

    mid_run: list[list[dict]] = []

    async def capture_then_cancel(_stage):
        mid_run.append(harness.recon_busy_snapshot())
        raise asyncio.CancelledError()

    _stub_stage_run(harness.stages[0], capture_then_cancel)

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
