"""Tests for ``server/recon_stage2_finding.py::file_stage2_finding``.

The helper lands a code-detected finding on the project's active Stage-2 run
of a REAL ``ReconReportState``, or drops it with one warning when no such run
is active.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import pytest

from fused_memory.models.reconciliation import StageId
from fused_memory.server.recon_report import ReconReportState
from fused_memory.server.recon_stage2_finding import file_stage2_finding

PROJECT_ID = 'dark_factory'


def _make_state() -> ReconReportState:
    return ReconReportState(ttl_seconds=300, clock=lambda: 0.0)


def _file(state: ReconReportState) -> None:
    file_stage2_finding(
        state,
        project_id=PROJECT_ID,
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


def _start_no_run(state: ReconReportState) -> None:
    pass


def _start_a_run_on_another_stage(state: ReconReportState) -> None:
    state.start_report(run_id='run-1', stage=StageId.memory_consolidator, project_id=PROJECT_ID)


def _start_the_stage2_run_of_another_project(state: ReconReportState) -> None:
    state.start_report(
        run_id='run-1', stage=StageId.task_knowledge_sync, project_id='other_project',
    )


@pytest.mark.parametrize(
    'start_runs',
    [_start_no_run, _start_a_run_on_another_stage, _start_the_stage2_run_of_another_project],
    ids=['no_run', 'another_stage', 'another_project'],
)
def test_drops_with_one_warning_when_the_project_has_no_active_stage2_run(
    caplog, start_runs: Callable[[ReconReportState], None],
):
    state = _make_state()
    start_runs(state)

    with caplog.at_level(logging.WARNING):
        _file(state)

    (record,) = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert 'some_flag' in record.getMessage()
    assert '42' in record.getMessage()
    assert PROJECT_ID in record.getMessage()
    assert state.get_findings_for_run('run-1') == []
