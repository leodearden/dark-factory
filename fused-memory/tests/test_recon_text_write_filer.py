"""Tests for the recon_text_write_filer adapter (task 4650).

``build_text_write_unconfirmed_filer`` lands a
``task_text_write_unconfirmed`` finding, raised by
``middleware/task_text_write_guard.py::verify_text_write_landed``, on the
active Stage-2 run of a REAL ``ReconReportState``, or is a no-op when no such
run is active for the finding's project.
"""

from __future__ import annotations

import inspect

import pytest

from fused_memory.middleware.task_text_write_guard import (
    TEXT_WRITE_UNCONFIRMED_FLAG_TYPE,
    TextWriteUnconfirmedFinding,
)
from fused_memory.server.recon_report import ReconReportState

STAGE = 'task_knowledge_sync'
PROJECT_ID = 'dark_factory'


def _make_state() -> ReconReportState:
    t = [0.0]
    return ReconReportState(ttl_seconds=300, clock=lambda: t[0])


def _filer(state: ReconReportState):
    from fused_memory.server.recon_text_write_filer import build_text_write_unconfirmed_filer

    return build_text_write_unconfirmed_filer(state)


def _finding(**overrides) -> TextWriteUnconfirmedFinding:
    kwargs: dict = dict(
        task_id='3708',
        project_id=PROJECT_ID,
        op='update_task',
        unconfirmed_fields=('description', 'details'),
    )
    kwargs.update(overrides)
    return TextWriteUnconfirmedFinding(**kwargs)


class TestBuildTextWriteUnconfirmedFiler:
    def test_builds_an_awaitable_callback(self):
        filer = _filer(_make_state())

        assert inspect.iscoroutinefunction(filer)

    @pytest.mark.asyncio
    async def test_files_one_finding_against_the_active_stage2_run(self):
        state = _make_state()
        state.start_report(run_id='run-1', stage=STAGE, project_id=PROJECT_ID)

        await _filer(state)(_finding())

        (row,) = state.get_findings_for_run('run-1')
        assert row['flag_type'] == TEXT_WRITE_UNCONFIRMED_FLAG_TYPE
        assert row['task_id'] == '3708'
        assert 'description' in row['description']
        assert 'details' in row['description']
        assert 'expected_text_fp' in row['suggested_action']
        assert 'get_task' in row['suggested_action']
        assert row['severity'] == 'serious'
        assert row['category'] == 'task_memory_mismatch'
        assert row['actionable'] is True

    @pytest.mark.asyncio
    async def test_no_op_when_no_run_is_active(self):
        state = _make_state()

        await _filer(state)(_finding())

        assert state.active_run_for_stage(STAGE, PROJECT_ID) is None

    @pytest.mark.asyncio
    async def test_no_op_when_the_active_run_is_for_another_project(self):
        state = _make_state()
        state.start_report(run_id='run-1', stage=STAGE, project_id='other_project')

        await _filer(state)(_finding())

        assert state.get_findings_for_run('run-1') == []
