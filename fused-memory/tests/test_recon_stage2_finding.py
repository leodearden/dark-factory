"""Tests for ``server/recon_stage2_finding.py::file_stage2_finding``.

The helper lands a code-detected finding on the project's active Stage-2 run
of a REAL ``ReconReportState``, or drops it with one warning when no such run
is active.
"""

from __future__ import annotations

import logging

from fused_memory.models.reconciliation import StageId
from fused_memory.server.recon_report import ReconReportState
from fused_memory.server.recon_stage2_finding import file_stage2_finding

PROJECT_ID = 'dark_factory'


def _make_state() -> ReconReportState:
    t = [0.0]
    return ReconReportState(ttl_seconds=300, clock=lambda: t[0])


def _file(state: ReconReportState, *, project_id: str = PROJECT_ID) -> None:
    file_stage2_finding(
        state,
        project_id=project_id,
        task_id='42',
        flag_type='some_flag',
        description='D',
        suggested_action='S',
    )


def test_files_one_finding_on_the_active_stage2_run():
    state = _make_state()
    state.start_report(run_id='run-1', stage=StageId.task_knowledge_sync, project_id=PROJECT_ID)

    _file(state)

    (row,) = state.get_findings_for_run('run-1')
    assert row['task_id'] == '42'
    assert row['flag_type'] == 'some_flag'
    assert row['description'] == 'D'
    assert row['suggested_action'] == 'S'
    assert row['severity'] == 'serious'
    assert row['category'] == 'task_memory_mismatch'
    assert row['actionable'] is True


def test_drops_with_a_warning_when_no_run_is_active(caplog):
    state = _make_state()

    with caplog.at_level(logging.WARNING):
        _file(state)

    (record,) = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert 'some_flag' in record.getMessage()
    assert '42' in record.getMessage()
    assert PROJECT_ID in record.getMessage()
    state.start_report(run_id='run-1', stage=StageId.task_knowledge_sync, project_id=PROJECT_ID)
    assert state.get_findings_for_run('run-1') == []


def test_ignores_a_run_active_on_another_stage():
    state = _make_state()
    state.start_report(run_id='run-1', stage=StageId.memory_consolidator, project_id=PROJECT_ID)

    _file(state)

    assert state.get_findings_for_run('run-1') == []


def test_ignores_the_active_run_of_another_project():
    state = _make_state()
    state.start_report(
        run_id='run-1', stage=StageId.task_knowledge_sync, project_id='other_project',
    )

    _file(state, project_id=PROJECT_ID)

    assert state.get_findings_for_run('run-1') == []
