"""The C5 implementer-shape report over shadow_cells rows (evals/shadow_report.py).

Driven through ``read_shadow_report`` against a real runs.db, real result JSONs
and real production invocations, all seeded through ``_shadow_report_seed``.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from _shadow_report_seed import (
    CANDIDATE,
    OPENED_AT,
    PROJECT_ID,
    SeededProject,
    make_project,
    seed_cell,
    seed_production_cost,
    seed_scored_cell,
)
from shared.project_layout import runs_db

from orchestrator.evals.results import ResultStore
from orchestrator.evals.shadow_report import (
    LineTag,
    ReportLine,
    RunsDbMissingError,
    ShadowReport,
    read_shadow_report,
)
from orchestrator.shadow_types import CostBasis, ReferenceKind, ShadowCellState, ShadowShape

_OTHER = 'kimi-k2-endpoint'
_AFTER_OPEN = OPENED_AT + timedelta(hours=1)


@pytest.fixture
def project(tmp_path: Path) -> SeededProject:
    return make_project(tmp_path)


async def _report(
    project: SeededProject, *, candidate: str | None = None, n_min: int = 12,
) -> ShadowReport:
    return await read_shadow_report(
        project.db_path,
        project_id=PROJECT_ID,
        shape=ShadowShape.IMPLEMENTER,
        candidate=candidate,
        n_min=n_min,
    )


def _line(
    report: ShadowReport, candidate: str, cost_basis: CostBasis = CostBasis.METERED,
) -> ReportLine:
    [line] = [
        line for line in report.lines
        if (line.key.candidate, line.key.cost_basis) == (candidate, cost_basis)
    ]
    return line


class TestCounts:
    @pytest.mark.asyncio
    async def test_settled_cells_split_by_reference_and_ended_cells_by_state(self, project):
        seed_scored_cell(project)
        seed_scored_cell(project)
        seed_scored_cell(project, reference_kind=ReferenceKind.NONE)
        for state in (
            ShadowCellState.CAP_EXCLUDED,
            ShadowCellState.EXPIRED,
            ShadowCellState.FAILED,
            ShadowCellState.OPENED,
            ShadowCellState.RUNNING,
            ShadowCellState.AWAITING_REFERENCE,
        ):
            seed_cell(project.store, state=state)

        line = _line(await _report(project), CANDIDATE)

        assert (line.pairs, line.unpaired) == (2, 1)
        assert (line.cap_excluded, line.expired, line.failed) == (1, 1, 1)

    @pytest.mark.asyncio
    async def test_a_candidate_with_only_in_flight_cells_still_gets_a_line(self, project):
        seed_cell(project.store, candidate=_OTHER, state=ShadowCellState.OPENED)

        line = _line(await _report(project), _OTHER)

        assert (line.pairs, line.unpaired, line.primary.n) == (0, 0, 0)

    @pytest.mark.asyncio
    async def test_cells_of_another_shape_or_project_are_not_read(self, project):
        seed_scored_cell(project)
        seed_scored_cell(project, shape=ShadowShape.ARCHITECT)
        seed_scored_cell(project, project_id='other-project')

        report = await _report(project)

        assert [line.key.shape for line in report.lines] == [ShadowShape.IMPLEMENTER]
        assert report.lines[0].pairs == 1


class TestPrimaryOutcome:
    @pytest.mark.asyncio
    async def test_the_paired_difference_is_candidate_usable_minus_a_landed_production(
        self, project,
    ):
        for _ in range(3):
            seed_scored_cell(project)
        seed_scored_cell(project, metrics={'review_blocking_issues': 2})

        report = await _report(project)
        line = _line(report, CANDIDATE)

        assert (line.primary.n, line.primary.mean) == (4, -0.25)
        assert line.primary.ci_low is not None and line.primary.ci_high is not None
        assert report.complete is True

    @pytest.mark.asyncio
    async def test_an_unmeasured_gate_is_not_usable(self, project):
        seed_scored_cell(project)
        seed_scored_cell(project, metrics={'tests_pass': None})

        line = _line(await _report(project), CANDIDATE)

        assert line.primary.mean == -0.5


class TestCostPerUsable:
    @pytest.mark.asyncio
    async def test_both_sides_are_costed_over_pairs_only(self, project):
        pairs = [
            seed_scored_cell(project, cost_usd=1.0),
            seed_scored_cell(project, cost_usd=2.0),
            seed_scored_cell(project, cost_usd=3.0, metrics={'lint_clean': False}),
        ]
        unpaired = seed_scored_cell(project, reference_kind=ReferenceKind.NONE, cost_usd=7.0)
        failed = seed_cell(project.store, state=ShadowCellState.FAILED, cost_usd=50.0)
        for cell in pairs:
            await seed_production_cost(project.db_path, PROJECT_ID, cell.task_id, 5.0, _AFTER_OPEN)
        await seed_production_cost(
            project.db_path, PROJECT_ID, pairs[0].task_id, 100.0, OPENED_AT - timedelta(hours=1),
        )
        for cell in (unpaired, failed):
            await seed_production_cost(project.db_path, PROJECT_ID, cell.task_id, 1000.0, _AFTER_OPEN)

        line = _line(await _report(project), CANDIDATE)

        assert line.candidate_cost.usd_per_usable == 3.0
        assert line.incumbent_cost.usd_per_usable == 5.0

    @pytest.mark.asyncio
    async def test_no_usable_outcome_has_no_cost_per_usable(self, project):
        seed_scored_cell(project, cost_usd=2.0, metrics={'tests_pass': False})
        seed_cell(project.store, candidate=_OTHER, state=ShadowCellState.RUNNING)

        report = await _report(project)

        assert _line(report, CANDIDATE).candidate_cost.usd_per_usable is None
        idle = _line(report, _OTHER)
        assert idle.candidate_cost.usd_per_usable is None
        assert idle.incumbent_cost.usd_per_usable is None


class TestCostBasis:
    @pytest.mark.asyncio
    async def test_each_basis_is_its_own_line_and_only_a_subscription_is_imputed(self, project):
        subscription = {'cost_basis': CostBasis.SUBSCRIPTION, 'cost_usd': 1.0}
        seed_scored_cell(project, **subscription, cost_credits=4.0)
        seed_scored_cell(project, **subscription, cost_credits=2.0)
        seed_scored_cell(project, cost_basis=CostBasis.METERED, cost_usd=1.0)

        report = await _report(project)
        imputed = _line(report, CANDIDATE, CostBasis.SUBSCRIPTION)
        metered = _line(report, CANDIDATE, CostBasis.METERED)

        assert (len(report.lines), imputed.pairs, metered.pairs) == (2, 2, 1)
        assert imputed.candidate_cost.usd_imputed is True
        assert imputed.candidate_cost.credits_per_usable == 3.0
        assert metered.candidate_cost.usd_imputed is False
        assert metered.candidate_cost.credits_per_usable is None
        for line in report.lines:
            incumbent = line.incumbent_cost
            assert incumbent.cost_basis is CostBasis.METERED
            assert (incumbent.usd_imputed, incumbent.credits_per_usable) == (False, None)


class TestUnderpowered:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(('n_min', 'underpowered'), [(5, True), (4, False)])
    async def test_a_line_below_n_min_pairs_is_tagged(self, project, n_min, underpowered):
        for _ in range(4):
            seed_scored_cell(project)

        line = _line(await _report(project, n_min=n_min), CANDIDATE)

        assert (LineTag.UNDERPOWERED in line.tags) is underpowered


@pytest.mark.asyncio
async def test_terminal_kinds_split_every_readable_settled_cell(project):
    seed_scored_cell(project)
    seed_scored_cell(project, metrics={'terminal_kind': 'planned'})
    seed_scored_cell(project, reference_kind=ReferenceKind.NONE, metrics={'terminal_kind': 'planned'})

    line = _line(await _report(project), CANDIDATE)

    assert line.terminal_kinds == (('planned', 2), ('unmeasured', 1))


class TestResultMissing:
    @pytest.mark.asyncio
    async def test_an_unreadable_result_marks_its_line_and_the_report_incomplete(self, project):
        deleted = seed_scored_cell(project)
        assert deleted.result_path is not None
        deleted.result_path.unlink()
        pathless = seed_cell(project.store)
        corrupt = seed_scored_cell(project)
        assert corrupt.result_path is not None
        corrupt.result_path.write_text('{"task_id": ')
        seed_scored_cell(project)
        seed_scored_cell(project, candidate=_OTHER)

        report = await _report(project)
        line = _line(report, CANDIDATE)

        missing = {gap.cell_id: gap for gap in line.result_missing}
        assert set(missing) == {deleted.cell_id, pathless.cell_id, corrupt.cell_id}
        assert missing[deleted.cell_id].result_path == deleted.result_path
        assert missing[deleted.cell_id].task_id == deleted.task_id
        assert missing[pathless.cell_id].result_path is None
        assert all(gap.detail.strip() for gap in line.result_missing)
        assert LineTag.RESULT_MISSING in line.tags
        assert (line.pairs, line.unpaired) == (1, 0)
        assert report.complete is False
        untouched = _line(report, _OTHER)
        assert (untouched.result_missing, untouched.pairs) == ((), 1)
        assert LineTag.RESULT_MISSING not in untouched.tags

    @pytest.mark.asyncio
    async def test_a_result_whose_metrics_do_not_read_back_is_missing(self, project):
        cell = seed_scored_cell(project)
        assert cell.result_path is not None

        def unknown_reference_kind(raw: dict) -> dict:
            raw['metrics']['reference_kind'] = 'merged'
            return raw

        ResultStore(project.results_dir).rewrite(cell.result_path, unknown_reference_kind)

        report = await _report(project)

        [gap] = _line(report, CANDIDATE).result_missing
        assert (gap.cell_id, 'merged' in gap.detail) == (cell.cell_id, True)
        assert report.complete is False


@pytest.mark.asyncio
async def test_a_candidate_filter_reads_only_that_candidates_lines(project):
    seed_scored_cell(project)
    seed_scored_cell(project, candidate=_OTHER)

    report = await _report(project, candidate=_OTHER)

    assert [line.key.candidate for line in report.lines] == [_OTHER]


@pytest.mark.asyncio
async def test_an_absent_runs_db_is_refused_and_not_created(tmp_path):
    db_path = runs_db(tmp_path / 'mistyped-project')

    with pytest.raises(RunsDbMissingError) as excinfo:
        await read_shadow_report(
            db_path, project_id=PROJECT_ID, shape=ShadowShape.IMPLEMENTER,
            candidate=None, n_min=12,
        )

    assert str(db_path) in str(excinfo.value)
    assert not db_path.parent.exists()


@pytest.mark.asyncio
async def test_a_shape_without_a_report_rule_is_refused_naming_the_supported_shapes(project):
    seed_scored_cell(project, shape=ShadowShape.ARCHITECT)

    with pytest.raises(ValueError, match=ShadowShape.IMPLEMENTER.value):
        await read_shadow_report(
            project.db_path, project_id=PROJECT_ID, shape=ShadowShape.ARCHITECT,
            candidate=None, n_min=12,
        )


@pytest.mark.asyncio
async def test_a_settled_row_with_an_incoherent_reference_is_refused_naming_the_cell(project):
    incoherent = seed_cell(project.store, reference_kind=ReferenceKind.LANDED, reference_sha=None)

    with pytest.raises(ValueError, match=incoherent.cell_id):
        await _report(project)
