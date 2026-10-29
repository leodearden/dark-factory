"""The C5 implementer-shape report over shadow_cells rows (evals/shadow_report.py).

Driven through ``read_shadow_report`` against a real runs.db, real result JSONs
and real production invocations, all seeded through ``_shadow_report_seed``.
"""

from __future__ import annotations

import json
import re
from datetime import timedelta
from pathlib import Path

import pytest
from _shadow_report_seed import (
    CANDIDATE,
    INCUMBENT,
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
    render_text,
)
from orchestrator.run_store import ShadowCell
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


# The contracted C5 columns, spelled out here as the pin rather than imported.
_COLUMNS = (
    'shape', 'candidate', 'incumbent', 'pairs', 'unpaired', 'cap_excluded', 'expired',
    'failed', 'usable \u0394 [95% CI]', '$/usable cand', '$/usable inc',
    'credits/usable cand', 'credits/usable inc', 'terminal_kind', 'tags',
)
_N_MIN = 3


def _seed_two_lines(project: SeededProject) -> ShadowCell:
    """A metered line of four pairs and an under-n_min subscription line of two."""
    metered = [seed_scored_cell(project, cost_usd=1.0) for _ in range(4)]
    for credits in (4.0, 2.0):
        seed_scored_cell(
            project, candidate=_OTHER, cost_basis=CostBasis.SUBSCRIPTION,
            cost_usd=0.5, cost_credits=credits,
        )
    return metered[0]


def _cells_of(row: str) -> list[str]:
    return re.split(r'\s{2,}', row.strip())


def _table(text: str) -> list[dict[str, str]]:
    lines = text.splitlines()
    [start] = [i for i, line in enumerate(lines) if _cells_of(line)[0] == 'shape']
    rows = []
    for line in lines[start + 1:]:
        if not line.strip():
            break
        rows.append(dict(zip(_cells_of(lines[start]), _cells_of(line), strict=True)))
    return rows


def _row(text: str, candidate: str) -> dict[str, str]:
    [row] = [row for row in _table(text) if row['candidate'] == candidate]
    return row


class TestRenderText:
    @pytest.mark.asyncio
    async def test_the_table_header_is_exactly_the_contracted_columns(self, project):
        _seed_two_lines(project)

        text = render_text(await _report(project, n_min=_N_MIN))

        [header] = [line for line in text.splitlines() if _cells_of(line)[0] == 'shape']
        assert tuple(_cells_of(header)) == _COLUMNS

    @pytest.mark.asyncio
    async def test_the_preamble_names_the_store_shape_and_n_min_basis(self, project):
        _seed_two_lines(project)

        text = render_text(await _report(project, n_min=_N_MIN))
        preamble = text.split('\n\n')[0]

        for fact in (PROJECT_ID, str(project.db_path), 'implementer', f'n_min {_N_MIN}'):
            assert fact in preamble
        assert 'provisional' in preamble
        assert 'not a power calculation' in preamble

    @pytest.mark.asyncio
    async def test_an_underpowered_line_prints_tagged_never_hidden(self, project):
        _seed_two_lines(project)

        text = render_text(await _report(project, n_min=_N_MIN))

        assert 'UNDERPOWERED' in _row(text, _OTHER)['tags']
        assert _row(text, CANDIDATE)['tags'] == '-'

    @pytest.mark.asyncio
    async def test_only_an_imputed_dollar_figure_is_tagged_imputed(self, project):
        _seed_two_lines(project)

        text = render_text(await _report(project, n_min=_N_MIN))

        assert 'imputed' in _row(text, _OTHER)['$/usable cand']
        assert 'imputed' not in _row(text, CANDIDATE)['$/usable cand']
        assert all('imputed' not in row['$/usable inc'] for row in _table(text))

    @pytest.mark.asyncio
    async def test_a_figure_that_does_not_exist_prints_a_dash(self, project):
        _seed_two_lines(project)

        text = render_text(await _report(project, n_min=_N_MIN))

        assert _row(text, CANDIDATE)['credits/usable cand'] == '-'
        assert {row['credits/usable inc'] for row in _table(text)} == {'-'}
        assert _row(text, _OTHER)['credits/usable cand'] != '-'

    @pytest.mark.asyncio
    async def test_a_missing_result_is_tagged_and_named_and_the_report_says_incomplete(
        self, project,
    ):
        gone = _seed_two_lines(project)
        assert gone.result_path is not None
        gone.result_path.unlink()

        text = render_text(await _report(project, n_min=_N_MIN))

        assert 'result_missing' in _row(text, CANDIDATE)['tags']
        [detail] = [line for line in text.splitlines() if line.startswith('result_missing:')]
        for fact in (gone.cell_id, gone.task_id, str(gone.result_path)):
            assert fact in detail
        assert text.rstrip().splitlines()[-1].startswith('INCOMPLETE')

    @pytest.mark.asyncio
    async def test_a_report_with_no_lines_says_so(self, project):
        text = render_text(await _report(project))

        assert 'no shadow cells match' in text
        assert 'INCOMPLETE' not in text


_LINE_KEYS = {
    'shape', 'candidate', 'incumbent', 'cost_basis', 'pairs', 'unpaired', 'cap_excluded',
    'expired', 'failed', 'primary_outcome', 'candidate_cost', 'incumbent_cost',
    'terminal_kind', 'tags', 'result_missing',
}
_COST_KEYS = {'usd_per_usable', 'usd_imputed', 'credits_per_usable', 'cost_basis'}


class TestToDict:
    @pytest.mark.asyncio
    async def test_the_json_form_carries_every_contracted_field(self, project):
        _seed_two_lines(project)

        payload = json.loads(json.dumps((await _report(project, n_min=_N_MIN)).to_dict()))

        assert {'project_id', 'shape', 'n_min', 'complete', 'lines'} <= set(payload)
        assert (payload['project_id'], payload['shape']) == (PROJECT_ID, 'implementer')
        assert (payload['n_min'], payload['complete']) == (_N_MIN, True)
        for line in payload['lines']:
            assert set(line) == _LINE_KEYS
            assert set(line['primary_outcome']) == {
                'name', 'n', 'mean_difference', 'ci95_low', 'ci95_high',
            }
            assert line['primary_outcome']['name'] == 'usable'
            assert set(line['candidate_cost']) == _COST_KEYS
            assert set(line['incumbent_cost']) == _COST_KEYS
            assert line['incumbent'] == INCUMBENT
            assert line['incumbent_cost']['cost_basis'] == 'metered'
        imputed = [line for line in payload['lines'] if line['candidate'] == _OTHER]
        assert [line['candidate_cost']['usd_imputed'] for line in imputed] == [True]
        assert [line['cost_basis'] for line in imputed] == ['subscription']

    @pytest.mark.asyncio
    async def test_a_missing_result_appears_in_the_json_and_marks_it_incomplete(self, project):
        gone = _seed_two_lines(project)
        assert gone.result_path is not None
        gone.result_path.unlink()

        payload = json.loads(json.dumps((await _report(project, n_min=_N_MIN)).to_dict()))

        assert payload['complete'] is False
        [line] = [line for line in payload['lines'] if line['candidate'] == CANDIDATE]
        [gap] = line['result_missing']
        assert (gap['cell_id'], gap['task_id']) == (gone.cell_id, gone.task_id)
        assert gap['result_path'] == str(gone.result_path)
        assert gap['detail']
        assert 'result_missing' in line['tags']

    @pytest.mark.asyncio
    async def test_the_json_lines_and_the_table_rows_are_the_same_rows(self, project):
        gone = _seed_two_lines(project)
        assert gone.result_path is not None
        gone.result_path.unlink()
        report = await _report(project, n_min=_N_MIN)

        from_json = [
            (line['candidate'], line['incumbent'], line['pairs'], line['tags'])
            for line in report.to_dict()['lines']
        ]
        from_text = [
            (row['candidate'], row['incumbent'], int(row['pairs']),
             [] if row['tags'] == '-' else row['tags'].split(','))
            for row in _table(render_text(report))
        ]
        assert from_json == from_text
