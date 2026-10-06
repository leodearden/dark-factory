"""census_outcomes folds a stream of typed evidence into the never-failed x cost census."""
from __future__ import annotations

from collections.abc import Iterator, Sequence

import pytest
import suite_census_outcomes as oc

PASSED, FAILED, SKIPPED = oc.Outcome.PASSED, oc.Outcome.FAILED, oc.Outcome.SKIPPED

JUNIT = oc.SourceWindow(
    source='junit', pattern='data/*.junit-*.xml.gz', present=True, artefacts=3,
    first='20260901T000000Z', last='20260930T235959Z',
)
LOGS = oc.SourceWindow(
    source='logs', pattern='data/*.test-*.log', present=True, artefacts=1,
    first='20260910T101010Z', last='20260910T101010Z',
)


def _test(name: str, package: str = 'p') -> oc.TestId:
    return oc.TestId(package=package, name=name)


def _obs(
    run: str, test: oc.TestId, outcome: oc.Outcome, seconds: float | None,
    source: str = 'junit',
) -> oc.Observation:
    return oc.Observation(source=source, run=run, test=test, outcome=outcome, seconds=seconds)


def _evidence(
    records: Sequence[oc.Observation | oc.Unresolved],
    *, windows: Sequence[oc.SourceWindow] = (JUNIT, LOGS),
    uncosted: frozenset[oc.TestId] = frozenset(),
) -> oc.Evidence:
    return oc.Evidence(
        windows=tuple(windows), records=lambda: iter(records), uncosted_universe=uncosted,
    )


def _census(records, **kwargs) -> oc.OutcomeCensus:
    return oc.census_outcomes(_evidence(records, **kwargs))


def _cost(census: oc.OutcomeCensus, test: oc.TestId) -> oc.TestCost:
    (cost,) = [cost for cost in census.ranking if cost.test == test]
    return cost


class TestPerRunCost:
    def test_params_of_one_function_sum_into_one_run(self):
        a = _test('t.py::test_a')
        census = _census([_obs('r1', a, PASSED, 2.0), _obs('r1', a, PASSED, 3.0)])
        cost = _cost(census, a)
        assert (cost.runs, cost.median_s, cost.total_s) == (1, 5.0, 5.0)

    def test_any_failed_observation_in_a_run_marks_the_test_failed(self):
        b = _test('t.py::test_b')
        census = _census([_obs('r1', b, PASSED, 1.0), _obs('r1', b, FAILED, 0.5)])
        assert b in census.failed_floor
        assert census.ranking == ()


class TestFailedFloor:
    def test_failure_without_seconds_joins_the_floor_without_a_cost(self):
        c = _test('t.py::test_c')
        census = _census([_obs('log1', c, FAILED, None, source='logs')])
        assert c in census.failed_floor
        assert census.failed_without_cost == (c,)
        assert census.ranking == ()

    def test_floor_is_the_union_across_sources(self):
        d, e = _test('t.py::test_d'), _test('t.py::test_e')
        census = _census([
            _obs('r1', d, PASSED, 1.0),
            _obs('r1', e, FAILED, 1.0),
            _obs('log1', d, FAILED, None, source='logs'),
        ])
        assert census.failed_floor == frozenset({d, e})
        assert census.failed_without_cost == ()


class TestRanking:
    def test_order_is_median_then_total_then_name(self):
        w, x, y, z = (_test(f't.py::test_{n}') for n in 'wxyz')
        census = _census([
            _obs('r1', w, PASSED, 10.0),
            _obs('r1', x, PASSED, 3.0),
            _obs('r1', z, PASSED, 4.0),
            _obs('r1', y, PASSED, 4.0),
            _obs('r2', x, PASSED, 5.0),
        ])
        assert [cost.test for cost in census.ranking] == [w, x, y, z]

    def test_cost_carries_runs_median_max_and_total(self):
        x = _test('t.py::test_x')
        census = _census([
            _obs('r1', x, PASSED, 3.0), _obs('r2', x, PASSED, 5.0), _obs('r3', x, PASSED, 1.0),
        ])
        assert _cost(census, x) == oc.TestCost(
            test=x, runs=3, median_s=3.0, max_s=5.0, total_s=9.0,
        )

    def test_a_test_that_only_ever_skipped_is_not_ranked(self):
        s = _test('t.py::test_s')
        census = _census([_obs('r1', s, SKIPPED, 0.01), _obs('r2', s, SKIPPED, 0.01)])
        assert census.ranking == ()

    def test_skipped_params_do_not_add_to_a_run_cost(self):
        a = _test('t.py::test_a')
        census = _census([_obs('r1', a, PASSED, 2.0), _obs('r1', a, SKIPPED, 7.0)])
        assert _cost(census, a).total_s == 2.0


class TestPackageSummary:
    def test_hand_computed_three_test_package(self):
        census = _census([
            _obs('r1', _test('t.py::test_big'), PASSED, 10.0),
            _obs('r1', _test('t.py::test_one'), PASSED, 1.0),
            _obs('r1', _test('t.py::test_two'), PASSED, 1.0),
            _obs('r1', _test('q.py::test_q', package='q'), PASSED, 1.0),
            _obs('log1', _test('t.py::test_gone'), FAILED, None, source='logs'),
        ])
        assert [summary.package for summary in census.packages] == ['p', 'q']
        summary = census.packages[0]
        assert (summary.observed, summary.failed, summary.never_failed) == (3, 1, 3)
        assert summary.p50_s == 1.0
        assert summary.top1_share == pytest.approx(10 / 12)


class TestSourceTallies:
    def test_runs_red_runs_and_unresolved_samples(self):
        a = _test('t.py::test_a')
        unresolved = [oc.Unresolved(source='junit', raw_id=f'gone{i}') for i in range(7)]
        census = _census([
            _obs('r1', a, PASSED, 1.0),
            *unresolved,
            _obs('r2', a, FAILED, 1.0),
            _obs('log1', a, FAILED, None, source='logs'),
        ])
        tallies = {tally.source: tally for tally in census.sources}
        junit = tallies['junit']
        assert (junit.runs, junit.red_runs, junit.unresolved) == (2, 1, 7)
        assert junit.unresolved_samples == ('gone0', 'gone1', 'gone2', 'gone3', 'gone4')
        assert (tallies['logs'].runs, tallies['logs'].red_runs) == (1, 1)


def test_uncosted_universe_minus_the_floor_is_never_failed_uncosted():
    alpha = _test('tests/infra/test_alpha.sh', package='tests/infra')
    beta = _test('tests/infra/test_beta.sh', package='tests/infra')
    census = _census(
        [_obs('ledger1', beta, FAILED, None, source='logs')],
        uncosted=frozenset({alpha, beta}),
    )
    assert census.never_failed_uncosted == (alpha,)
    (infra,) = [summary for summary in census.packages if summary.package == 'tests/infra']
    assert infra.uncosted_never_failed == 1


class TestStreaming:
    def test_records_is_called_once_and_may_be_a_generator(self):
        a = _test('t.py::test_a')
        calls = []

        def records() -> Iterator[oc.Observation | oc.Unresolved]:
            calls.append(1)
            yield _obs('r1', a, PASSED, 1.0)
            yield _obs('r2', a, PASSED, 3.0)

        census = oc.census_outcomes(
            oc.Evidence(windows=(JUNIT,), records=records, uncosted_universe=frozenset())
        )
        assert calls == [1]
        assert _cost(census, a).runs == 2

    def test_a_run_reappearing_after_another_started_is_refused(self):
        a = _test('t.py::test_a')
        with pytest.raises(ValueError, match='r1'):
            _census([
                _obs('r1', a, PASSED, 1.0), _obs('r2', a, PASSED, 1.0), _obs('r1', a, PASSED, 1.0),
            ])

    def test_a_record_from_a_source_without_a_window_is_refused(self):
        with pytest.raises(ValueError, match='mystery'):
            _census([_obs('r1', _test('t.py::test_a'), PASSED, 1.0, source='mystery')])


def _table_rows(text: str) -> list[list[str]]:
    return [
        [cell.strip() for cell in line.strip().strip('|').split('|')]
        for line in text.splitlines()
        if line.startswith('|')
    ]


def _rows_containing(text: str, needle: str) -> list[list[str]]:
    return [row for row in _table_rows(text) if any(needle in cell for cell in row)]


class TestRenderMarkdown:
    @pytest.fixture
    def census(self) -> oc.OutcomeCensus:
        return _census([
            _obs('r1', _test('t.py::test_slow'), PASSED, 9.0),
            _obs('r1', _test('t.py::test_mid'), PASSED, 5.0),
            _obs('r1', _test('t.py::test_fast'), PASSED, 1.0),
            _obs('r1', _test('t.py::test_bad'), FAILED, 1.0),
            _obs('log1', _test('t.py::test_logged'), FAILED, None, source='logs'),
        ])

    def test_one_window_row_per_source(self, census):
        text = oc.render_markdown(census, top=10)
        for window in (JUNIT, LOGS):
            (row,) = _rows_containing(text, window.pattern)
            assert str(window.artefacts) in row
            assert window.first in row and window.last in row

    def test_ranked_rows_in_order_truncated_to_top(self, census):
        text = oc.render_markdown(census, top=2)
        positions = [text.index(name) for name in ('test_slow', 'test_mid')]
        assert positions == sorted(positions)
        assert _rows_containing(text, 'test_slow') and _rows_containing(text, 'test_mid')
        assert not _rows_containing(text, 'test_fast')

    def test_per_package_row_and_floor_members(self, census):
        text = oc.render_markdown(census, top=10)
        assert [row for row in _table_rows(text) if row[0] == 'p']
        (floor_block,) = [
            block for block in text.split('<details>') if 'test_bad' in block
        ]
        items = [line for line in floor_block.splitlines() if line.startswith('- ')]
        assert len(items) == len(census.failed_floor) == 2
        assert any('test_logged' in item for item in items)

    def test_rendering_is_deterministic(self, census):
        assert oc.render_markdown(census, top=10) == oc.render_markdown(census, top=10)


_THIN_PREFIX = f'Ranked on fewer than {oc.THIN_SAMPLE_RUNS} runs: '


def _thin_ranks(text: str) -> str:
    (line,) = [line for line in text.splitlines() if line.startswith(_THIN_PREFIX)]
    return line.removeprefix(_THIN_PREFIX).split('.')[0]


class TestThinSampleNote:
    @pytest.fixture
    def census(self) -> oc.OutcomeCensus:
        steady, once, also_once = (
            _test(f't.py::test_{name}') for name in ('steady', 'once', 'also_once')
        )
        records = [_obs('r0', once, PASSED, 9.0), _obs('r0', also_once, PASSED, 1.0)]
        records += [_obs(f'r{i}', steady, PASSED, 5.0) for i in range(oc.THIN_SAMPLE_RUNS)]
        return _census(records)

    def test_shown_rows_under_the_threshold_are_named_by_rank(self, census):
        assert [cost.runs for cost in census.ranking] == [1, oc.THIN_SAMPLE_RUNS, 1]
        assert _thin_ranks(oc.render_markdown(census, top=10)) == '1, 3'

    def test_rows_beyond_top_are_not_named(self, census):
        assert _thin_ranks(oc.render_markdown(census, top=2)) == '1'

    def test_a_ranking_without_thin_rows_says_none(self, census):
        assert _thin_ranks(oc.render_markdown(census, top=0)) == 'none'
