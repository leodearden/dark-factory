"""Scoring the link adjudicator against blind-rated pairs, and the gate on it (task 7159).

The population is the write-triage soak's rated pairs: a ratings file keyed by
``child_intent_id`` joined to an items file holding the rated texts. The
fixtures here are synthetic and shaped like the soak's own two files: two
projects of ten items, one of two, and six rows per ten-item project rated by
a second rater, so the raters' own kappa is 0.8 on both binaries.

A calibration run asks a :class:`FakeAdjudicator` about the rated texts, into
a real ledger on ``tmp_path``, and escapes through a :class:`RecordingFiler`.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from _link_heal_harness import (
    FAKE_ADJUDICATOR_MODEL,
    FAKE_PROMPT_SHA256,
    Answer,
    FakeAdjudicator,
    RecordingFiler,
)

from fused_memory.maintenance.link_adjudicator import AdjudicationFailure
from fused_memory.maintenance.link_heal import MISFILE_VERDICTS, Verdict
from fused_memory.maintenance.link_heal_calibration import (
    POOLED,
    Binary,
    CalibrationInputError,
    CalibrationItem,
    CalibrationSettings,
    KappaBelowFloor,
    TooManyUnanswered,
    assess_calibration,
    binary_agreement,
    load_soak_population,
    rater_agreement,
    run_calibration,
)
from fused_memory.maintenance.link_heal_executor import CALIBRATION_ANCHOR
from fused_memory.maintenance.link_heal_ledger import LinkHealLedger, RunKind, RunSource
from fused_memory.maintenance.link_heal_store import text_sha256

FIXTURES = Path(__file__).resolve().parent / 'fixtures'
RATINGS = FIXTURES / 'link_heal_calibration_ratings.jsonl'
ITEMS = FIXTURES / 'link_heal_calibration_items.json'

ALPHA = 'proj_alpha'
BETA = 'proj_beta'
GAMMA = 'proj_gamma'

#: The fixture raters' kappa (0.8 on both binaries) less the default margin.
FLOOR = 0.7
MARGIN = 0.10
GATE_SIZE = 10


def _population() -> tuple[CalibrationItem, ...]:
    return load_soak_population(RATINGS, ITEMS)


def _perfect(items: tuple[CalibrationItem, ...]) -> dict[str, Verdict]:
    return {item.key: item.final for item in items}


def _assess(items, verdicts, **overrides):
    settings = {'margin': MARGIN, 'min_project_items': GATE_SIZE, **overrides}
    return assess_calibration(items, verdicts, **settings)


def _write_soak(tmp_path: Path, ratings: list, items: object) -> tuple[Path, Path]:
    ratings_path = tmp_path / 'adjudications.jsonl'
    items_path = tmp_path / 'items_all.json'
    ratings_path.write_text(
        ''.join(row if isinstance(row, str) else json.dumps(row) + '\n' for row in ratings),
        encoding='utf-8',
    )
    items_path.write_text(items if isinstance(items, str) else json.dumps(items), encoding='utf-8')
    return ratings_path, items_path


def _fixture_rows() -> tuple[list[dict], list[dict]]:
    ratings = [json.loads(line) for line in RATINGS.read_text(encoding='utf-8').splitlines()]
    items = json.loads(ITEMS.read_text(encoding='utf-8'))
    return ratings, items


class TestLoadSoakPopulation:
    def test_every_rating_joins_its_item_in_ratings_order(self):
        ratings, items = _fixture_rows()
        by_intent = {item['intent_id']: item for item in items}

        population = _population()

        assert [item.key for item in population] == [row['child_intent_id'] for row in ratings]
        first, row = population[0], ratings[0]
        source = by_intent[row['child_intent_id']]
        assert first.project == row['project']
        assert first.child_id == row['child_memory_id']
        assert first.parent_id == row['parent_id']
        assert first.child_text == source['child_text']
        assert first.parent_text == source['parent_text']
        assert first.final is Verdict(row['rater_verdict'])
        assert first.primary is Verdict(row['primary_verdict'])
        assert first.second is Verdict(row['second_verdict'])

    def test_a_row_with_no_second_rating_carries_none(self):
        population = _population()

        assert sum(item.second is not None for item in population) == 12
        assert population[6].second is None

    def test_a_rating_with_no_item_names_its_line_and_key(self, tmp_path):
        ratings, items = _fixture_rows()
        orphan = {**ratings[1], 'child_intent_id': 'no-such-intent'}
        paths = _write_soak(tmp_path, [ratings[0], orphan], items)

        with pytest.raises(CalibrationInputError, match=r'line 2\b.*no-such-intent'):
            load_soak_population(*paths)

    @pytest.mark.parametrize('field', ['rater_verdict', 'primary_verdict', 'second_verdict'])
    def test_a_word_outside_the_brief_names_its_line_field_and_value(self, tmp_path, field):
        ratings, items = _fixture_rows()
        bad = {**ratings[0], field: 'MAYBE'}
        paths = _write_soak(tmp_path, [ratings[1], bad], items)

        with pytest.raises(CalibrationInputError, match=rf'line 2\b.*{field}.*MAYBE'):
            load_soak_population(*paths)

    def test_a_ratings_line_that_is_not_json_names_its_line(self, tmp_path):
        ratings, items = _fixture_rows()
        paths = _write_soak(tmp_path, [ratings[0], '{not json\n'], items)

        with pytest.raises(CalibrationInputError, match=r'line 2\b.*not JSON'):
            load_soak_population(*paths)

    def test_an_items_file_that_is_not_json_names_the_file(self, tmp_path):
        ratings, _items = _fixture_rows()
        paths = _write_soak(tmp_path, ratings, '[{"intent_id": ')

        with pytest.raises(CalibrationInputError, match=r'items_all\.json.*not JSON'):
            load_soak_population(*paths)


def _pairs(both_yes: int, adjudicator_only: int, rater_only: int, both_no: int):
    return (
        [(True, True)] * both_yes
        + [(True, False)] * adjudicator_only
        + [(False, True)] * rater_only
        + [(False, False)] * both_no
    )


class TestBinaryAgreement:
    def test_the_confusion_and_kappa_of_a_hand_computed_table(self):
        agreement = binary_agreement(_pairs(40, 10, 5, 45))

        assert agreement.n == 100
        assert (
            agreement.both_yes,
            agreement.adjudicator_only,
            agreement.rater_only,
            agreement.both_no,
        ) == (40, 10, 5, 45)
        assert agreement.kappa == pytest.approx(0.70)

    @pytest.mark.parametrize('pairs', [_pairs(12, 0, 0, 0), _pairs(0, 0, 0, 12), []])
    def test_kappa_is_none_when_chance_agreement_is_one(self, pairs):
        assert binary_agreement(pairs).kappa is None


class TestBinaries:
    @pytest.mark.parametrize('verdict', list(Verdict))
    def test_misfile_is_the_table_s_misfile_verdicts(self, verdict):
        assert Binary.MISFILE.holds(verdict) is (verdict in MISFILE_VERDICTS)

    @pytest.mark.parametrize('verdict', list(Verdict))
    def test_contradiction_is_corrects(self, verdict):
        assert Binary.CONTRADICTION.holds(verdict) is (verdict is Verdict.CORRECTS)

    def test_rater_agreement_is_the_second_rater_against_the_primary_on_the_overlap(self):
        agreement = rater_agreement(_population())

        misfile = agreement[Binary.MISFILE]
        assert (misfile.both_yes, misfile.adjudicator_only, misfile.rater_only, misfile.both_no) == (
            3, 1, 0, 8,
        )
        assert misfile.kappa == pytest.approx(0.8)
        contradiction = agreement[Binary.CONTRADICTION]
        assert (
            contradiction.both_yes,
            contradiction.adjudicator_only,
            contradiction.rater_only,
            contradiction.both_no,
        ) == (3, 0, 1, 8)
        assert contradiction.kappa == pytest.approx(0.8)


def _misfiled_in(items, project: str) -> list[CalibrationItem]:
    return [item for item in items if item.project == project and Binary.MISFILE.holds(item.final)]


class TestAssessCalibration:
    def test_an_adjudicator_agreeing_with_every_rating_passes(self):
        population = _population()

        result = _assess(population, _perfect(population))

        assert result.passed
        assert result.failures == ()
        assert result.unanswered_share == 0.0
        misfile = result.pooled.binaries[Binary.MISFILE]
        assert misfile.agreement.kappa == pytest.approx(1.0)
        assert (misfile.agreement.both_yes, misfile.agreement.both_no) == (7, 15)
        assert misfile.floor == pytest.approx(FLOOR)
        assert result.pooled.scope == POOLED
        assert result.pooled.gated

    def test_every_project_has_an_entry_and_only_the_large_ones_are_gated(self):
        population = _population()

        result = _assess(population, _perfect(population))

        assert {project: score.gated for project, score in result.projects.items()} == {
            ALPHA: True, BETA: True, GAMMA: False,
        }
        assert result.projects[GAMMA].items == 2
        for score in result.projects.values():
            for binary in Binary:
                assert score.binaries[binary].floor == pytest.approx(FLOOR)

    def test_one_gated_project_below_its_floor_fails_while_the_pool_passes(self):
        population = _population()
        verdicts = _perfect(population)
        for item in _misfiled_in(population, ALPHA)[:2]:
            verdicts[item.key] = Verdict.EXTENDS

        result = _assess(population, verdicts)

        assert not result.passed
        assert result.pooled.binaries[Binary.MISFILE].clears_floor
        [failure] = result.failures
        assert isinstance(failure, KappaBelowFloor)
        assert (failure.binary, failure.scope) == (Binary.MISFILE, ALPHA)
        assert failure.kappa == pytest.approx(0.14 / 0.34)
        assert failure.floor == pytest.approx(FLOOR)

    def test_a_project_below_the_minimum_is_reported_but_never_gated(self):
        population = _population()
        verdicts = _perfect(population)
        for item in population:
            if item.project == GAMMA:
                verdicts[item.key] = Verdict.EXTENDS

        result = _assess(population, verdicts)

        assert result.passed
        assert not result.projects[GAMMA].binaries[Binary.MISFILE].clears_floor

    def test_a_kappa_of_none_fails_closed(self):
        population = _population()
        delta = tuple(
            CalibrationItem(
                key=f'delta-{index}', project='proj_delta', child_id=f'c{index}',
                parent_id=f'p{index}', child_text=f'delta child {index}',
                parent_text=f'delta parent {index}', final=verdict, primary=verdict, second=None,
            )
            for index, verdict in enumerate((Verdict.EXTENDS, Verdict.SAME, Verdict.CORRECTS))
        )
        items = population + delta

        result = _assess(items, _perfect(items), min_project_items=3)

        assert not result.passed
        [failure] = result.failures
        assert isinstance(failure, KappaBelowFloor)
        assert (failure.binary, failure.scope, failure.kappa) == (
            Binary.MISFILE, 'proj_delta', None,
        )

    def test_more_than_the_unanswered_bound_fails(self):
        population = _population()
        verdicts = _perfect(population)
        for item in [item for item in population if item.final is Verdict.EXTENDS][:2]:
            del verdicts[item.key]

        result = _assess(population, verdicts)

        assert not result.passed
        assert result.unanswered_share == pytest.approx(2 / 22)
        [failure] = result.failures
        assert isinstance(failure, TooManyUnanswered)
        assert failure.share == pytest.approx(2 / 22)
        assert failure.bound == 0.05

    def test_unanswered_pairs_within_the_bound_pass_and_are_left_out_of_kappa(self):
        population = _population()
        verdicts = _perfect(population)
        dropped = next(item for item in population if item.final is Verdict.EXTENDS)
        del verdicts[dropped.key]

        result = _assess(population, verdicts)

        assert result.passed
        assert result.pooled.answered == 21
        assert result.pooled.binaries[Binary.MISFILE].agreement.n == 21

    def test_a_verdict_for_an_unrated_key_is_refused(self):
        population = _population()

        with pytest.raises(ValueError, match='not-a-rated-key'):
            _assess(population, {**_perfect(population), 'not-a-rated-key': Verdict.SAME})

    def test_the_json_is_the_same_whatever_order_the_items_arrive_in(self):
        population = _population()
        verdicts = _perfect(population)
        verdicts[_misfiled_in(population, BETA)[0].key] = Verdict.SAME

        forward = _assess(population, verdicts).as_json()
        backward = _assess(tuple(reversed(population)), verdicts).as_json()

        assert json.dumps(forward, sort_keys=True) == json.dumps(backward, sort_keys=True)
        assert forward['passed'] is True
        assert [entry['scope'] for entry in forward['projects']] == [ALPHA, BETA, GAMMA]

    def test_the_adjudicator_is_scored_against_the_final_verdict_not_the_primary(self):
        population = _population()
        verdicts = _perfect(population)
        assert population[0].primary is Verdict.RELATED
        overruled = replace(population[0], final=Verdict.SAME)

        result = _assess((overruled, *population[1:]), verdicts)

        assert result.pooled.binaries[Binary.MISFILE].agreement.adjudicator_only == 1


POPULATION_SHA = 'f' * 64
SETTINGS = CalibrationSettings(margin=MARGIN, min_project_items=GATE_SIZE)


@pytest.fixture
def ledger(tmp_path: Path):
    opened = LinkHealLedger(tmp_path / 'link_heal.db')
    yield opened
    opened.close()


def _answering(population, overrides: dict[str, Answer] | None = None) -> FakeAdjudicator:
    """A fake adjudicator answering each child text with its final verdict, unless overridden."""
    answers: dict[str, Answer] = {item.child_text: item.final for item in population}
    answers.update(overrides or {})
    return FakeAdjudicator(answers.__getitem__)


def _failing_alpha(population) -> dict[str, Answer]:
    return {item.child_text: Verdict.EXTENDS for item in _misfiled_in(population, ALPHA)[:2]}


#: Late on 3 November in UTC, already 4 November east of it.
CALIBRATION_NOW = datetime(2026, 11, 4, 1, 30, tzinfo=timezone(timedelta(hours=5)))


async def _calibrate(population, adjudicate, ledger, tmp_path, filer=None):
    return await run_calibration(
        population,
        adjudicate=adjudicate,
        ledger=ledger,
        settings=SETTINGS,
        filer=filer if filer is not None else RecordingFiler(),
        reports_dir=tmp_path / 'link-heal-reports',
        population_sha256=POPULATION_SHA,
        now=CALIBRATION_NOW,
    )


class TestRunCalibration:
    @pytest.mark.asyncio
    async def test_it_is_a_finished_non_writing_calibration_run(self, ledger, tmp_path):
        population = _population()

        report = await _calibrate(population, _answering(population), ledger, tmp_path)

        run = ledger.resolve_run(report.run_id)
        assert run.source is RunSource.CALIBRATION
        assert not run.writes
        assert run.kind is RunKind.PLAN
        assert run.finished_at is not None

    @pytest.mark.asyncio
    async def test_it_asks_about_the_rated_texts_only(self, ledger, tmp_path):
        population = _population()
        adjudicate = _answering(population)

        await _calibrate(population, adjudicate, ledger, tmp_path)

        assert sorted((pair.child_text, pair.parent_text) for pair in adjudicate.pairs) == sorted(
            (item.child_text, item.parent_text) for item in population
        )

    @pytest.mark.asyncio
    async def test_each_answered_pair_is_ledgered_at_its_rated_texts(self, ledger, tmp_path):
        population = _population()
        unanswered = population[4]
        adjudicate = _answering(
            population, {unanswered.child_text: AdjudicationFailure.PARSE_FAILURE},
        )

        report = await _calibrate(population, adjudicate, ledger, tmp_path)

        for item in population:
            row = ledger.adjudication_at(
                item.project, item.child_id, item.parent_id,
                text_sha256(item.child_text), text_sha256(item.parent_text),
                model=FAKE_ADJUDICATOR_MODEL, prompt_sha256=FAKE_PROMPT_SHA256,
            )
            if item is unanswered:
                assert row is None
                continue
            assert row is not None, item.key
            assert row.run_id == report.run_id
            assert row.record.verdict is item.final
            assert row.record.reason == f'scripted {item.final}'
            assert row.record.model == FAKE_ADJUDICATOR_MODEL
            assert row.record.prompt_sha256 == FAKE_PROMPT_SHA256

    @pytest.mark.asyncio
    async def test_the_calibration_is_recorded_for_its_model_and_prompt(self, ledger, tmp_path):
        population = _population()

        report = await _calibrate(population, _answering(population), ledger, tmp_path)

        row = ledger.latest_calibration(FAKE_ADJUDICATOR_MODEL, FAKE_PROMPT_SHA256)
        assert row is not None
        assert row.passed
        assert row.run_id == report.run_id
        assert row.population_sha256 == POPULATION_SHA
        assert row.report['run_id'] == report.run_id
        assert row.report['passed'] is True
        assert ledger.latest_calibration('another-model', FAKE_PROMPT_SHA256) is None
        assert ledger.latest_calibration(FAKE_ADJUDICATOR_MODEL, 'e' * 64) is None

    @pytest.mark.asyncio
    async def test_the_newest_calibration_wins(self, ledger, tmp_path):
        population = _population()
        await _calibrate(population, _answering(population), ledger, tmp_path)

        failed = await _calibrate(
            population, _answering(population, _failing_alpha(population)), ledger, tmp_path,
        )

        row = ledger.latest_calibration(FAKE_ADJUDICATOR_MODEL, FAKE_PROMPT_SHA256)
        assert row is not None
        assert (row.run_id, row.passed) == (failed.run_id, False)

    @pytest.mark.asyncio
    async def test_it_writes_a_json_and_a_markdown_report(self, ledger, tmp_path):
        population = _population()

        report = await _calibrate(population, _answering(population), ledger, tmp_path)

        reports_dir = tmp_path / 'link-heal-reports'
        stem = f'link-heal-calibration-2026-11-03-{report.run_id[:8]}'
        assert report.json_path == reports_dir / f'{stem}.json'
        assert report.markdown_path == reports_dir / f'{stem}.md'
        row = ledger.latest_calibration(FAKE_ADJUDICATOR_MODEL, FAKE_PROMPT_SHA256)
        assert row is not None
        assert json.loads(report.json_path.read_text(encoding='utf-8')) == row.report
        markdown = report.markdown_path.read_text(encoding='utf-8')
        for expected in (POOLED, ALPHA, BETA, GAMMA, 'misfile', 'contradiction', '0.700', 'PASSED'):
            assert expected in markdown, expected
        assert 'opus' in markdown

    @pytest.mark.asyncio
    async def test_a_naive_now_is_refused_before_any_question(self, ledger, tmp_path):
        population = _population()
        adjudicate = _answering(population)

        with pytest.raises(ValueError, match='timezone-aware'):
            await run_calibration(
                population, adjudicate=adjudicate, ledger=ledger, settings=SETTINGS,
                filer=RecordingFiler(), reports_dir=tmp_path / 'link-heal-reports',
                population_sha256=POPULATION_SHA, now=datetime(2026, 11, 3, 12, 0),
            )

        assert adjudicate.pairs == []
        assert ledger.recent_runs(10) == []

    @pytest.mark.asyncio
    async def test_a_passing_calibration_files_nothing(self, ledger, tmp_path):
        population = _population()
        filer = RecordingFiler()

        report = await _calibrate(population, _answering(population), ledger, tmp_path, filer)

        assert report.passed
        assert filer.escapes == []

    @pytest.mark.asyncio
    async def test_a_failing_calibration_files_one_escape_naming_each_failure(
        self, ledger, tmp_path,
    ):
        population = _population()
        filer = RecordingFiler()

        report = await _calibrate(
            population, _answering(population, _failing_alpha(population)), ledger, tmp_path, filer,
        )

        assert not report.passed
        [escape] = filer.escapes
        assert escape.anchor == CALIBRATION_ANCHOR == 'link-heal-calibration'
        [failure] = escape.detail['failures']
        assert 'misfile' in failure and ALPHA in failure
        assert escape.detail['report'] == str(report.markdown_path)
        assert 'FAILED' in report.markdown_path.read_text(encoding='utf-8')
