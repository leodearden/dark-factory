"""Tests for the recon_text_write_filer adapter (task 4650).

``build_text_write_unconfirmed_filer`` lands a
``task_text_write_unconfirmed`` finding, raised by
``middleware/task_text_write_guard.py::verify_text_write_landed``, on the
active Stage-2 run of a REAL ``ReconReportState``, or drops it with a warning
when no such run is active for the finding's project.
"""

from __future__ import annotations

import inspect
import logging

import pytest

from fused_memory.middleware.task_text_write_guard import (
    TASK_TEXT_FP_KEY,
    TEXT_VERIFIED_KEY,
    TEXT_WRITE_UNCONFIRMED_FLAG_TYPE,
    TextFieldVerification,
    TextWriteUnconfirmedFinding,
)
from fused_memory.server.recon_report import ReconReportState
from fused_memory.server.recon_text_write_filer import build_text_write_unconfirmed_filer

STAGE = 'task_knowledge_sync'
PROJECT_ID = 'dark_factory'


def _make_state() -> ReconReportState:
    t = [0.0]
    return ReconReportState(ttl_seconds=300, clock=lambda: t[0])


def _filer(state: ReconReportState):
    return build_text_write_unconfirmed_filer(state)


REPLACED_DESCRIPTION = TextFieldVerification(field='description', confirmed=False, appended=False)
APPENDED_DETAILS = TextFieldVerification(field='details', confirmed=False, appended=True)


def _finding(**overrides) -> TextWriteUnconfirmedFinding:
    kwargs: dict = dict(
        task_id='3708',
        project_id=PROJECT_ID,
        op='update_task',
        unconfirmed=(REPLACED_DESCRIPTION, APPENDED_DETAILS),
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
        assert 'expected_text_fp' in row['suggested_action']
        assert TASK_TEXT_FP_KEY in row['suggested_action']
        assert TEXT_VERIFIED_KEY in row['suggested_action']
        assert 'get_task' in row['suggested_action']
        assert row['severity'] == 'serious'
        assert row['category'] == 'task_memory_mismatch'
        assert row['actionable'] is True

    @pytest.mark.asyncio
    async def test_the_description_states_each_field_s_unmet_invariant_by_mode(self):
        state = _make_state()
        state.start_report(run_id='run-1', stage=STAGE, project_id=PROJECT_ID)

        await _filer(state)(_finding())

        (row,) = state.get_findings_for_run('run-1')
        assert 'description (replace: the stored text does not equal' in row['description']
        assert 'details (append: the stored text does not end with' in row['description']
        assert 'contain' not in row['description']

    @pytest.mark.asyncio
    async def test_the_description_says_a_failed_read_left_the_landing_unknown(self):
        state = _make_state()
        state.start_report(run_id='run-1', stage=STAGE, project_id=PROJECT_ID)
        unread = TextFieldVerification(
            field='details', confirmed=False, appended=False, readback_error=True,
        )

        await _filer(state)(_finding(unconfirmed=(unread,)))

        (row,) = state.get_findings_for_run('run-1')
        assert 'details (the read failed' in row['description']

    @pytest.mark.asyncio
    async def test_drops_with_a_warning_when_no_run_is_active(self, caplog):
        state = _make_state()

        with caplog.at_level(logging.WARNING, logger='fused_memory.server.recon_text_write_filer'):
            await _filer(state)(_finding())

        (record,) = caplog.records
        assert record.levelno == logging.WARNING
        assert TEXT_WRITE_UNCONFIRMED_FLAG_TYPE in record.getMessage()
        assert '3708' in record.getMessage()
        state.start_report(run_id='run-1', stage=STAGE, project_id=PROJECT_ID)
        assert state.get_findings_for_run('run-1') == []

    @pytest.mark.asyncio
    async def test_no_op_when_the_active_run_is_for_another_project(self):
        state = _make_state()
        state.start_report(run_id='run-1', stage=STAGE, project_id='other_project')

        await _filer(state)(_finding())

        assert state.get_findings_for_run('run-1') == []
