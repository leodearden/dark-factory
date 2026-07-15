"""Tests for the recon_lifecycle_filer adapter and ReconReportState.active_run_for_stage
(task 2624, step-7).

``build_lifecycle_reset_filer`` is the injected ``FileFindingFn`` collaborator
wired into ``TaskInterceptor.set_lifecycle_reset_filer`` (see
``test_live_task_write_guard.py``'s interceptor-boundary tests, which exercise
it via a mock). This module tests the REAL adapter against a REAL
``ReconReportState``: it must resolve the currently in-progress Stage-2
(``task_knowledge_sync``) run for the finding's project via the new
``active_run_for_stage`` accessor, then land a ``task_lifecycle_reset_detected``
finding through ``add_finding`` — or silently no-op when no such run is
active, since filing must never raise on the write path.
"""

from __future__ import annotations

import pytest

from fused_memory.middleware.live_task_write_guard import (
    LIFECYCLE_RESET_FLAG_TYPE,
    LifecycleResetFinding,
    LiveTaskSnapshot,
)
from fused_memory.server.recon_report import ReconReportState

STAGE = 'task_knowledge_sync'
PROJECT_ID = 'dark_factory'


def _make_state() -> ReconReportState:
    t = [0.0]
    return ReconReportState(ttl_seconds=300, clock=lambda: t[0])


def _finding(**overrides) -> LifecycleResetFinding:
    kwargs: dict = dict(
        task_id='42',
        project_id=PROJECT_ID,
        op='update_task',
        before=LiveTaskSnapshot(
            status='in-progress',
            claimant_run_id='run-abc/session-1/pid=1',
            heartbeat_at='2026-07-15T12:00:00+00:00',
        ),
        after=LiveTaskSnapshot(status='pending', claimant_run_id=None, heartbeat_at=None),
        diverged_fields=('claimant_run_id', 'status'),
    )
    kwargs.update(overrides)
    return LifecycleResetFinding(**kwargs)


# ---------------------------------------------------------------------------
# (a) ReconReportState.active_run_for_stage
# ---------------------------------------------------------------------------


class TestActiveRunForStage:
    def test_returns_in_progress_run_id(self):
        state = _make_state()
        state.start_report(run_id='r1', stage=STAGE, project_id=PROJECT_ID)
        assert state.active_run_for_stage(STAGE, PROJECT_ID) == 'r1'

    def test_none_when_nothing_active(self):
        state = _make_state()
        assert state.active_run_for_stage(STAGE, PROJECT_ID) is None

    def test_none_for_other_project(self):
        state = _make_state()
        state.start_report(run_id='r1', stage=STAGE, project_id='other_project')
        assert state.active_run_for_stage(STAGE, PROJECT_ID) is None

    def test_none_for_other_stage(self):
        state = _make_state()
        state.start_report(run_id='r1', stage='memory_consolidator', project_id=PROJECT_ID)
        assert state.active_run_for_stage(STAGE, PROJECT_ID) is None

    def test_none_once_run_moved_past_the_stage(self):
        # r1 started stage 'task_knowledge_sync' and has since moved on to a
        # later stage — the run_id's CURRENT stage is no longer STAGE, even
        # though a (r1, STAGE) entry still sits in _state.
        state = _make_state()
        state.start_report(run_id='r1', stage=STAGE, project_id=PROJECT_ID)
        state.start_report(run_id='r1', stage='later_stage', project_id=PROJECT_ID)
        assert state.active_run_for_stage(STAGE, PROJECT_ID) is None

    def test_none_once_stage_completed(self):
        state = _make_state()
        state.start_report(run_id='r1', stage=STAGE, project_id=PROJECT_ID)
        state.complete('r1', 'summary')
        assert state.active_run_for_stage(STAGE, PROJECT_ID) is None


# ---------------------------------------------------------------------------
# (b) / (c) build_lifecycle_reset_filer
# ---------------------------------------------------------------------------


class TestBuildLifecycleResetFiler:
    @pytest.mark.asyncio
    async def test_files_finding_against_the_active_run(self):
        from fused_memory.server.recon_lifecycle_filer import build_lifecycle_reset_filer

        state = _make_state()
        state.start_report(run_id='run-1', stage=STAGE, project_id=PROJECT_ID)
        filer = build_lifecycle_reset_filer(state)

        await filer(_finding())

        rows = state.get_findings_for_run('run-1')
        assert len(rows) == 1
        row = rows[0]
        assert row['flag_type'] == LIFECYCLE_RESET_FLAG_TYPE
        assert row['task_id'] == '42'
        assert row['description']
        assert row['suggested_action']
        assert row['severity'] == 'serious'
        assert row['category'] == 'task_memory_mismatch'
        assert row['actionable'] is True

    @pytest.mark.asyncio
    async def test_no_op_when_no_active_run(self):
        from fused_memory.server.recon_lifecycle_filer import build_lifecycle_reset_filer

        state = _make_state()
        filer = build_lifecycle_reset_filer(state)

        # Must not raise even though no Stage-2 run is active for this project.
        await filer(_finding())

        assert state._state == {}

    @pytest.mark.asyncio
    async def test_no_op_when_active_run_is_for_a_different_project(self):
        from fused_memory.server.recon_lifecycle_filer import build_lifecycle_reset_filer

        state = _make_state()
        state.start_report(run_id='run-1', stage=STAGE, project_id='other_project')
        filer = build_lifecycle_reset_filer(state)

        await filer(_finding(project_id=PROJECT_ID))

        assert state.get_findings_for_run('run-1') == []
