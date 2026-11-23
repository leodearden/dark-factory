"""What one architect phase concluded: plan confirmation, exit reports, and their conflict (task 6814)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from orchestrator.architect_phase_verdict import (
    ExitReport,
    PlanExitConflict,
    clear_exit_reports,
    find_plan_exit_conflict,
    is_confirmed,
    newly_confirmed,
    read_exit_reports,
)
from orchestrator.artifacts import Plan, TaskArtifacts

T0 = '2026-11-23T10:00:00+00:00'
T1 = '2026-11-23T11:00:00+00:00'
STEPS = [{'id': 'step-1', 'type': 'impl', 'description': 'do it', 'status': 'pending', 'commit': None}]


def _confirmed_plan(stamp: str = T0) -> dict:
    return {'title': 't', 'files': ['a.py'], 'steps': list(STEPS), '_finalized_at': stamp}


def _model(document: dict) -> Plan:
    return Plan.from_json(document)


UNCONFIRMED = _model({'title': 't', 'steps': list(STEPS)})


@pytest.fixture
def artifacts(tmp_path: Path) -> TaskArtifacts:
    a = TaskArtifacts(tmp_path)
    a.init('6814', 'title', 'description')
    return a


_WRITERS: dict[ExitReport, Callable[[TaskArtifacts], None]] = {
    ExitReport.UNACTIONABLE_TASK: lambda a: a.write_unactionable_task('spec unworkable', 'ev-u'),
    ExitReport.FALSE_PREMISE: lambda a: a.write_false_premise('false', 'the premise', 'ev-f', 'respec'),
    ExitReport.ALREADY_DONE: lambda a: a.write_already_done('abc123', 'ev-a'),
    ExitReport.READY_TO_MERGE: lambda a: a.write_ready_to_merge('def456', 'ev-r'),
    ExitReport.BLOCKING_DEPENDENCY: lambda a: a.write_blocking_dependency('5763', 'needs it', 'sha0'),
}


class TestIsConfirmed:
    @pytest.mark.parametrize('plan', [
        None,
        _model({}),
        _model({'title': 't', 'steps': [], '_finalized_at': T0}),
        UNCONFIRMED,
    ], ids=['absent', 'empty', 'no-steps', 'unconfirmed'])
    def test_false_for_a_plan_not_confirmed(self, plan: Plan | None) -> None:
        assert is_confirmed(plan) is False

    @pytest.mark.parametrize('document', [
        _confirmed_plan(),
        {**_confirmed_plan(), '_session_id': '6814-abc'},
    ], ids=['confirmed', 'stamped'])
    def test_true_for_a_confirmed_or_stamped_plan(self, document: dict) -> None:
        assert is_confirmed(_model(document)) is True


class TestNewlyConfirmed:
    @pytest.mark.parametrize('plan', [None, UNCONFIRMED], ids=['absent', 'unconfirmed'])
    def test_false_when_not_confirmed(self, plan: Plan | None) -> None:
        assert newly_confirmed(plan, None) is False

    def test_false_when_the_earlier_plan_carried_the_same_stamp(self) -> None:
        assert newly_confirmed(_model(_confirmed_plan(T0)), _model(_confirmed_plan(T0))) is False

    @pytest.mark.parametrize('before', [
        None, UNCONFIRMED, _model(_confirmed_plan(T0)),
    ], ids=['absent', 'unconfirmed', 'other-stamp'])
    def test_true_when_confirmed_with_a_new_stamp(self, before: Plan | None) -> None:
        assert newly_confirmed(_model(_confirmed_plan(T1)), before) is True


class TestReadExitReports:
    def test_empty_when_no_report(self, artifacts: TaskArtifacts) -> None:
        assert read_exit_reports(artifacts) == {}

    @pytest.mark.parametrize('report', list(_WRITERS), ids=lambda r: r.value)
    def test_returns_each_written_report(self, artifacts: TaskArtifacts, report: ExitReport) -> None:
        _WRITERS[report](artifacts)

        reports = read_exit_reports(artifacts)

        assert list(reports) == [report]
        assert reports[report] == getattr(artifacts, f'read_{report.value}')()

    def test_every_member_has_a_writer(self) -> None:
        assert set(_WRITERS) == set(ExitReport)


class TestFindPlanExitConflict:
    def _reports(self) -> dict:
        return {
            ExitReport.UNACTIONABLE_TASK: {'reason': 'spec unworkable', 'evidence': 'ev-u'},
            ExitReport.BLOCKING_DEPENDENCY: {'depends_on_task_id': '5763', 'reason': 'needs it'},
        }

    def test_none_without_reports(self) -> None:
        assert find_plan_exit_conflict(_model(_confirmed_plan(T1)), None, {}) is None

    @pytest.mark.parametrize('plan,before', [
        (UNCONFIRMED, None),
        (_model(_confirmed_plan(T0)), _model(_confirmed_plan(T0))),
    ], ids=['unconfirmed', 'confirmed-before-phase'])
    def test_none_when_not_newly_confirmed(self, plan: Plan, before: Plan | None) -> None:
        assert find_plan_exit_conflict(plan, before, self._reports()) is None

    def test_conflict_carries_both_sides(self) -> None:
        reports = self._reports()
        plan = _model(_confirmed_plan(T1))

        conflict = find_plan_exit_conflict(plan, _model(_confirmed_plan(T0)), reports)

        assert isinstance(conflict, PlanExitConflict)
        assert conflict.plan == plan
        assert dict(conflict.reports) == reports

    def test_summary_and_detail_name_every_report_and_the_plan(self) -> None:
        conflict = find_plan_exit_conflict(_model(_confirmed_plan(T1)), None, self._reports())
        assert conflict is not None

        summary, detail = conflict.summary(), conflict.detail()

        for text in (summary, detail):
            assert 'unactionable_task' in text
            assert 'blocking_dependency' in text
        assert T1 in detail
        assert 'step-1: do it' in detail
        assert '\n' not in summary
        assert len(summary) < 200
        assert len(detail) <= 2000

    def test_detail_stays_bounded_for_huge_payloads_and_plans(self) -> None:
        reports = {r: {'evidence': 'x' * 5000} for r in ExitReport}
        steps = [{'id': f'step-{n}', 'description': 'y' * 3000} for n in range(1, 21)]

        conflict = find_plan_exit_conflict(_model({'steps': steps, '_finalized_at': T1}), None, reports)
        assert conflict is not None

        detail = conflict.detail()
        assert len(detail) <= 2000
        assert T1 in detail
        for report in ExitReport:
            assert report.value in detail
        for step in steps:
            assert f"{step['id']}: " in detail


class TestClearExitReports:
    def test_clears_only_the_named_reports(self, artifacts: TaskArtifacts) -> None:
        artifacts.write_plan(_confirmed_plan())
        artifacts.write_unactionable_task('spec unworkable', 'ev-u')
        artifacts.write_blocking_dependency('5763', 'needs it', 'sha0')
        artifacts.write_already_done('abc123', 'ev-a')

        clear_exit_reports(artifacts, [ExitReport.UNACTIONABLE_TASK, ExitReport.BLOCKING_DEPENDENCY])

        assert artifacts.read_unactionable_task() is None
        assert artifacts.read_blocking_dependency() is None
        assert artifacts.read_already_done() is not None
        assert artifacts.read_plan()['_finalized_at'] == T0
