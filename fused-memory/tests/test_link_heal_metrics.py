"""One link-heal sweep run's metric series and the rolling summary gate θ reads (task 6188).

Section 1 drives ``link_heal_metrics`` at its own seam: a :class:`SweepRunView`
built by hand becomes one M1 metric series, and the series on disk roll up into
``summary.json``. Section 2 drives :func:`run_sweep` over the in-process server
in ``_link_heal_harness``, so what is under test there is what a real sweep
leaves for θ.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import pytest_asyncio
from _link_heal_harness import (
    ALL_LINK_HEAL_PREFIXES,
    DF,
    FAKE_ADJUDICATOR_MODEL,
    FAKE_PROMPT_SHA256,
    PROJECTS,
    REIFY,
    FakeAdjudicator,
    LinkHealHarness,
    RecordingFiler,
    build_harness,
)
from shared.memory_eval_metrics import (
    Corpus,
    Metric,
    MetricSeries,
    format_stamp,
    load_metric_series,
    metrics_artifact_path,
    write_metric_series,
)

from fused_memory.maintenance.link_adjudicator import AdjudicationFailure
from fused_memory.maintenance.link_heal import HealAction, RunCounts, Verdict
from fused_memory.maintenance.link_heal_executor import (
    BACKLOG_ANCHOR,
    MISFILE_SHARE_ANCHOR,
    RUN_REPORT_ANCHOR,
    RunLimits,
    run_adjudicator_plan,
)
from fused_memory.maintenance.link_heal_ledger import LinkHealLedger, RunSource
from fused_memory.maintenance.link_heal_metrics import (
    EVAL_ID,
    SUMMED_METRICS,
    SweepMetrics,
    SweepMetricsTarget,
    summarize_series,
    summary_path,
    sweep_metric_series,
)
from fused_memory.maintenance.link_heal_report import (
    CalibrationStatus,
    LedgerRun,
    SweepRunView,
    WritesWithheld,
)
from fused_memory.maintenance.link_heal_store import CreatedWindow
from fused_memory.maintenance.link_heal_sweep import SweepReport, SweepSettings, run_sweep
from fused_memory.server.grouped_read import SIGHTING_KIND

NOW = datetime(2026, 11, 15, 2, 15, tzinfo=UTC)
DAY = timedelta(days=1)
STAMP = format_stamp(NOW)
THETA_KEYS = (
    'runs_7d', 'writing_runs_7d', 'would_escape_7d', 'adjudication_failure_rate_7d',
    'actions_planned_7d', 'skipped_cap_7d',
)
LIMITS = RunLimits(
    max_actions_per_run=25, backlog_multiplier=5, write_failure_streak=3,
    misfile_share_ceiling=0.25, corrects_share_ceiling=0.60,
)
GATE_SCRIPT = Path(__file__).resolve().parents[2] / 'scripts' / 'check_write_triage_readiness_gate.py'
#: Gate θ's three summary bounds, copied verbatim from task 6191's metadata.before_done.args.
THETA_SUMMARY_REQUIRES = (
    ('s', 'runs_7d', '>=', '6'),
    ('s', 'would_escape_7d', '<=', '0'),
    ('s', 'adjudication_failure_rate_7d', '<=', '0.05'),
)


def _id(prefix: str, n: int) -> str:
    return f'{prefix}-0000-4000-8000-{n:012d}'


def _run(n: int, **counts) -> LedgerRun:
    return LedgerRun(run_id=f'{n:032x}', counts=RunCounts(**counts))


def _view(plan: LedgerRun, apply: LedgerRun | None = None, records=()) -> SweepRunView:
    return SweepRunView(
        plan=plan,
        apply=apply,
        records=records,
        window=CreatedWindow(since=NOW - DAY, until=NOW),
        calibration=CalibrationStatus(FAKE_ADJUDICATOR_MODEL, FAKE_PROMPT_SHA256, None, False),
        withheld=WritesWithheld.WRITES_OFF if apply is None else None,
        ledger_path=Path('link_heal.db'),
    )


def _values(series: MetricSeries) -> dict[str, float]:
    return {metric.metric_id: metric.value for metric in series.metrics}


def _series(stamp: str, **values: int) -> MetricSeries:
    """A link-heal series stamped *stamp*: every summed metric 0 unless named."""
    return MetricSeries(
        schema_version=1,
        eval_id=EVAL_ID,
        run_stamp=stamp,
        corpus=Corpus(project_id=DF),
        metrics=[
            Metric(metric_id=metric_id, kind='scalar', value=values.get(metric_id, 0), n=1)
            for metric_id in SUMMED_METRICS
        ],
    )


def assert_theta_keys(summary: dict) -> None:
    for key in THETA_KEYS:
        assert key in summary, key
        assert isinstance(summary[key], int | float), (key, summary[key])
        assert not isinstance(summary[key], bool), key


class TestOneRunsSeries:
    def test_the_series_is_the_link_heal_eval_stamped_with_the_run(self):
        series = sweep_metric_series(_view(_run(1)), project_id=DF, stamp=STAMP)

        assert EVAL_ID == 'link-heal'
        assert series.eval_id == EVAL_ID
        assert series.run_stamp == STAMP
        assert series.corpus.project_id == DF

    def test_every_summed_metric_is_one_run_wide(self):
        series = sweep_metric_series(_view(_run(1)), project_id=DF, stamp=STAMP)

        assert [metric.metric_id for metric in series.metrics] == list(SUMMED_METRICS)
        assert {metric.n for metric in series.metrics} == {1}

    def test_a_report_mode_run_is_not_a_writing_run(self):
        series = sweep_metric_series(_view(_run(1)), project_id=DF, stamp=STAMP)

        assert _values(series)['writing_runs'] == 0

    def test_a_run_with_an_apply_run_is_a_writing_run(self):
        series = sweep_metric_series(_view(_run(1), _run(2)), project_id=DF, stamp=STAMP)

        assert _values(series)['writing_runs'] == 1

    def test_escapes_and_caps_sum_over_the_plan_and_apply_runs(self):
        plan = _run(1, would_escape=(MISFILE_SHARE_ANCHOR,), skipped_cap=0)
        apply = _run(2, would_escape=(BACKLOG_ANCHOR,), skipped_cap=4)

        values = _values(sweep_metric_series(_view(plan, apply), project_id=DF, stamp=STAMP))

        assert values['would_escape'] == 2
        assert values['skipped_cap'] == 4

    def test_the_adjudication_counts_are_the_plan_runs(self):
        plan = _run(
            1, planned=6, adjudications_asked=20, adjudication_failed=1, adjudication_capped=3,
        )
        apply = _run(2, planned=6, adjudications_asked=99)

        values = _values(sweep_metric_series(_view(plan, apply), project_id=DF, stamp=STAMP))

        assert values['actions_planned'] == 6
        assert values['adjudications_asked'] == 20
        assert values['adjudication_failed'] == 1
        assert values['adjudication_capped'] == 3

    def test_a_rise_in_an_escape_cap_or_failure_is_the_alarmed_direction(self):
        series = sweep_metric_series(_view(_run(1)), project_id=DF, stamp=STAMP)

        alarmed = {
            metric.metric_id for metric in series.metrics
            if metric.kind == 'count' and metric.direction == 'higher_is_worse'
        }
        reported = {metric.metric_id for metric in series.metrics if metric.kind == 'scalar'}
        assert alarmed == {'would_escape', 'skipped_cap', 'adjudication_failed', 'adjudication_capped'}
        assert reported == {'writing_runs', 'actions_planned', 'adjudications_asked'}

    def test_a_view_with_no_records_counts_no_project(self):
        series = sweep_metric_series(_view(_run(1)), project_id=DF, stamp=STAMP)

        assert series.corpus.counts == {}


class TestTheSummary:
    def test_the_window_holds_the_last_seven_days_inclusive(self, tmp_path):
        for moment in (NOW, NOW - 3 * DAY, NOW - 7 * DAY):
            write_metric_series(_series(format_stamp(moment)), tmp_path)
        for moment in (NOW - 7 * DAY - timedelta(seconds=1), NOW + timedelta(seconds=1)):
            write_metric_series(_series(format_stamp(moment), writing_runs=1), tmp_path)

        summary = summarize_series(tmp_path, now=NOW)

        assert summary['runs_7d'] == 3
        assert summary['writing_runs_7d'] == 0
        assert summary['window'] == {
            'since': format_stamp(NOW - 7 * DAY), 'until': format_stamp(NOW),
        }
        assert summary['eval_id'] == EVAL_ID

    def test_each_summed_metric_is_its_id_with_a_7d_suffix(self, tmp_path):
        write_metric_series(
            _series(STAMP, would_escape=1, actions_planned=4, skipped_cap=2, writing_runs=1),
            tmp_path,
        )
        write_metric_series(
            _series(format_stamp(NOW - DAY), actions_planned=3, adjudication_capped=5), tmp_path,
        )

        summary = summarize_series(tmp_path, now=NOW)

        assert {f'{metric_id}_7d' for metric_id in SUMMED_METRICS} <= set(summary)
        assert summary['would_escape_7d'] == 1
        assert summary['actions_planned_7d'] == 7
        assert summary['skipped_cap_7d'] == 2
        assert summary['writing_runs_7d'] == 1
        assert summary['adjudication_capped_7d'] == 5
        assert all(type(summary[f'{metric_id}_7d']) is int for metric_id in SUMMED_METRICS)
        assert_theta_keys(summary)

    def test_the_failure_rate_is_failed_over_asked(self, tmp_path):
        write_metric_series(_series(STAMP, adjudications_asked=12, adjudication_failed=1), tmp_path)
        write_metric_series(_series(format_stamp(NOW - DAY), adjudications_asked=8), tmp_path)

        summary = summarize_series(tmp_path, now=NOW)

        assert summary['adjudications_asked_7d'] == 20
        assert summary['adjudication_failed_7d'] == 1
        assert summary['adjudication_failure_rate_7d'] == pytest.approx(0.05)

    def test_with_nothing_asked_the_rate_is_zero(self, tmp_path):
        write_metric_series(_series(STAMP), tmp_path)

        summary = summarize_series(tmp_path, now=NOW)

        assert summary['adjudications_asked_7d'] == 0
        assert summary['adjudication_failure_rate_7d'] == 0.0
        assert_theta_keys(summary)

    @pytest.mark.parametrize('make_dir', [False, True], ids=['absent', 'empty'])
    def test_no_series_still_writes_every_key(self, tmp_path, make_dir):
        if make_dir:
            (tmp_path / EVAL_ID).mkdir()

        summary = summarize_series(tmp_path, now=NOW)

        assert summary['runs_7d'] == 0
        assert_theta_keys(summary)


class TestTheTarget:
    def test_it_writes_the_series_and_the_summary(self, tmp_path):
        root = tmp_path / 'memory-evals'

        emitted = SweepMetricsTarget(root, DF).write(_view(_run(1)), NOW)

        assert emitted == SweepMetrics(
            metrics_path=metrics_artifact_path(root, EVAL_ID, STAMP),
            summary_path=summary_path(root),
        )
        assert emitted.summary_path == root / EVAL_ID / 'summary.json'
        assert load_metric_series(emitted.metrics_path).eval_id == EVAL_ID
        assert emitted.unwritten is None
        assert emitted.as_detail() == {
            'metrics_json': str(emitted.metrics_path), 'summary_json': str(emitted.summary_path),
        }

    def test_a_root_under_a_regular_file_is_reported_not_raised(self, tmp_path):
        (tmp_path / 'a-file').write_text('not a directory')

        emitted = SweepMetricsTarget(tmp_path / 'a-file' / 'memory-evals', DF).write(
            _view(_run(1)), NOW,
        )

        assert emitted.unwritten is not None
        assert emitted.unwritten.startswith('NotADirectoryError')
        assert emitted.as_detail()['metrics_unwritten'] == emitted.unwritten

    def test_an_unparseable_earlier_series_is_reported_and_no_summary_is_written(self, tmp_path):
        root = tmp_path / 'memory-evals'
        broken = metrics_artifact_path(root, EVAL_ID, format_stamp(NOW - DAY))
        broken.parent.mkdir(parents=True)
        broken.write_text('{not json')

        emitted = SweepMetricsTarget(root, DF).write(_view(_run(1)), NOW)

        assert emitted.unwritten is not None
        assert emitted.unwritten.startswith('JSONDecodeError')
        assert not summary_path(root).exists()


@pytest_asyncio.fixture
async def harness(mock_config, tmp_path):
    built = await build_harness(
        mock_config, tmp_path, metadata_patch_prefixes=ALL_LINK_HEAL_PREFIXES,
    )
    yield built
    await built.journal.close()


@pytest.fixture
def ledger(tmp_path: Path):
    opened = LinkHealLedger(tmp_path / 'link_heal.db')
    yield opened
    opened.close()


def _seed_sightings(harness: LinkHealHarness, count: int, *, project: str = DF, first: int = 0):
    for n in range(first, first + count):
        harness.seed_link(
            kind=SIGHTING_KIND, child=_id('aaaaaaaa', n), parent=_id('bbbbbbbb', n),
            project=project, child_text=f'sighting {n}', parent_text=f'the parent of {n}',
        )


class TestTheCorpusCountsExaminedLinksPerProject:
    @pytest.mark.asyncio
    async def test_a_recorded_run_counts_each_projects_examined_links(
        self, harness, ledger, tmp_path,
    ):
        _seed_sightings(harness, 3)
        _seed_sightings(harness, 2, project=REIFY, first=10)
        plan = await run_adjudicator_plan(
            adjudicate=FakeAdjudicator(lambda _text: Verdict.SAME),
            store=harness.store(), census=harness.mem0, ledger=ledger, limits=LIMITS,
            projects=PROJECTS, plan_path_for=lambda _run_id: tmp_path / 'p.json',
            source=RunSource.SWEEP,
        )
        view = _view(
            LedgerRun(plan.run_id, plan.counts), records=tuple(ledger.run_records(plan.run_id)),
        )

        series = sweep_metric_series(view, project_id=DF, stamp=STAMP)

        assert series.corpus.counts == {DF: 3, REIFY: 2}


INSIDE = (NOW - timedelta(hours=12)).isoformat()


def _sweep_settings(*, writes: bool = False, max_adjudications: int = 200) -> SweepSettings:
    return SweepSettings(
        projects=(DF,), writes=writes, dry_run=False,
        actions=frozenset({HealAction.DETACH, HealAction.RELABEL}), window_hours=24,
        max_adjudications=max_adjudications,
    )


def _seed_in_window(harness: LinkHealHarness, texts: list[str]) -> None:
    for n, text in enumerate(texts):
        harness.seed_link(
            kind=SIGHTING_KIND, child=_id('cccccccc', n), parent=_id('dddddddd', n),
            child_text=text, parent_text=f'the parent of {text}', meta={'created_at': INSIDE},
        )


def _calibrate(ledger: LinkHealLedger) -> None:
    run_id = ledger.start_run(RunSource.CALIBRATION, writes=False)
    ledger.add_calibration(
        run_id, model=FAKE_ADJUDICATOR_MODEL, prompt_sha256=FAKE_PROMPT_SHA256,
        population_sha256='0' * 64, passed=True, report={'passed': True},
    )


async def _sweep(
    harness: LinkHealHarness,
    ledger: LinkHealLedger,
    tmp_path: Path,
    adjudicate: FakeAdjudicator,
    *,
    settings: SweepSettings | None = None,
    filer: RecordingFiler | None = None,
    limits: RunLimits = LIMITS,
    now: datetime = NOW,
    metrics_root: Path | None = None,
) -> SweepReport:
    return await run_sweep(
        adjudicate=adjudicate,
        store=harness.store(),
        census=harness.mem0,
        ledger=ledger,
        limits=limits,
        settings=settings or _sweep_settings(),
        filer=filer or RecordingFiler(),
        reports_dir=tmp_path / 'reports',
        now=now,
        metrics=SweepMetricsTarget(metrics_root or tmp_path / 'memory-evals', DF),
    )


def _summary(tmp_path: Path) -> dict:
    return json.loads(summary_path(tmp_path / 'memory-evals').read_text())


def _run_report_detail(filer: RecordingFiler) -> dict:
    (escape,) = [escape for escape in filer.escapes if escape.anchor == RUN_REPORT_ANCHOR]
    return dict(escape.detail)


class TestAReportModeSweepLeavesItsSeriesAndTheSummary:
    @pytest.mark.asyncio
    async def test_one_non_writing_run_leaves_a_series_and_a_summary_theta_can_read(
        self, harness, ledger, tmp_path,
    ):
        _seed_in_window(harness, ['an extending sighting', 'another extending sighting'])
        filer = RecordingFiler()

        report = await _sweep(
            harness, ledger, tmp_path, FakeAdjudicator(lambda _text: Verdict.EXTENDS), filer=filer,
        )

        (run,) = ledger.recent_runs(50)
        assert (run.source, run.writes) == (RunSource.SWEEP, False)
        root = tmp_path / 'memory-evals'
        series = load_metric_series(metrics_artifact_path(root, EVAL_ID, STAMP))
        assert series.eval_id == 'link-heal'
        summary = _summary(tmp_path)
        assert_theta_keys(summary)
        assert (summary['runs_7d'], summary['writing_runs_7d']) == (1, 0)
        assert report.metrics == SweepMetrics(
            metrics_path=metrics_artifact_path(root, EVAL_ID, STAMP),
            summary_path=summary_path(root),
        )
        assert _run_report_detail(filer)['summary_json'] == str(summary_path(root))

    @pytest.mark.asyncio
    async def test_a_backlog_the_plan_forecasts_is_counted_and_never_filed(
        self, harness, ledger, tmp_path,
    ):
        _seed_in_window(harness, [f'an extending sighting {n}' for n in range(3)])
        filer = RecordingFiler()
        limits = RunLimits(
            max_actions_per_run=1, backlog_multiplier=1, write_failure_streak=3,
            misfile_share_ceiling=0.25, corrects_share_ceiling=0.60,
        )

        report = await _sweep(
            harness, ledger, tmp_path, FakeAdjudicator(lambda _text: Verdict.EXTENDS),
            filer=filer, limits=limits,
        )

        assert BACKLOG_ANCHOR in report.view.plan.counts.would_escape
        assert BACKLOG_ANCHOR not in {escape.anchor for escape in filer.escapes}
        assert _summary(tmp_path)['would_escape_7d'] == 1

    @pytest.mark.asyncio
    async def test_a_misfile_share_the_plan_forecasts_is_counted_and_writes_nothing(
        self, harness, ledger, tmp_path,
    ):
        _seed_in_window(
            harness,
            [f'misfiled sighting {n}' if n < 8 else f'belonging sighting {n}' for n in range(20)],
        )
        filer = RecordingFiler()

        report = await _sweep(
            harness, ledger, tmp_path,
            FakeAdjudicator(
                lambda text: Verdict.RELATED if text.startswith('misfiled') else Verdict.SAME,
            ),
            filer=filer,
        )

        assert MISFILE_SHARE_ANCHOR in report.view.plan.counts.would_escape
        assert MISFILE_SHARE_ANCHOR not in {escape.anchor for escape in filer.escapes}
        assert harness.mem0.write_count == 0
        assert _summary(tmp_path)['would_escape_7d'] == 1


class TestTheAdjudicationFiguresReachTheSummary:
    @pytest.mark.asyncio
    async def test_the_failure_rate_is_over_the_pairs_asked(self, harness, ledger, tmp_path):
        _seed_in_window(harness, [f'a sighting {n}' for n in range(4)])
        answers = iter([AdjudicationFailure.PARSE_FAILURE])

        await _sweep(
            harness, ledger, tmp_path, FakeAdjudicator(lambda _text: next(answers, Verdict.SAME)),
        )

        summary = _summary(tmp_path)
        assert summary['adjudications_asked_7d'] == 4
        assert summary['adjudication_failure_rate_7d'] == pytest.approx(0.25)

    @pytest.mark.asyncio
    async def test_the_cap_is_counted(self, harness, ledger, tmp_path):
        _seed_in_window(harness, [f'a sighting {n}' for n in range(5)])

        await _sweep(
            harness, ledger, tmp_path, FakeAdjudicator(lambda _text: Verdict.SAME),
            settings=_sweep_settings(max_adjudications=2),
        )

        summary = _summary(tmp_path)
        assert summary['adjudication_capped_7d'] == 3
        assert summary['adjudications_asked_7d'] == 2


class TestWritingRunsAndNights:
    @pytest.mark.asyncio
    async def test_a_writing_sweep_counts_as_a_writing_run(self, harness, ledger, tmp_path):
        _seed_in_window(harness, ['an extending sighting'])
        _calibrate(ledger)

        report = await _sweep(
            harness, ledger, tmp_path, FakeAdjudicator(lambda _text: Verdict.EXTENDS),
            settings=_sweep_settings(writes=True),
        )

        assert report.apply_run_id is not None
        assert _summary(tmp_path)['writing_runs_7d'] == 1

    @pytest.mark.asyncio
    async def test_two_nights_are_two_runs(self, harness, ledger, tmp_path):
        _seed_in_window(harness, ['a sighting'])
        same = FakeAdjudicator(lambda _text: Verdict.SAME)

        await _sweep(harness, ledger, tmp_path, same)
        await _sweep(harness, ledger, tmp_path, same, now=NOW + DAY)

        assert _summary(tmp_path)['runs_7d'] == 2


class TestMetricsThatCannotBeWritten:
    @pytest.mark.asyncio
    async def test_are_escalated_and_the_report_is_still_written(self, harness, ledger, tmp_path):
        (tmp_path / 'a-file').write_text('not a directory')
        filer = RecordingFiler()

        report = await _sweep(
            harness, ledger, tmp_path, FakeAdjudicator(lambda _text: Verdict.SAME),
            filer=filer, metrics_root=tmp_path / 'a-file' / 'memory-evals',
        )

        assert report.report_path.exists()
        assert report.metrics.unwritten is not None
        assert report.metrics.unwritten.startswith('NotADirectoryError')
        assert _run_report_detail(filer)['metrics_unwritten'] == report.metrics.unwritten


class TestThetaReadsTheSummary:
    @pytest.mark.asyncio
    async def test_every_theta_summary_bound_resolves_to_a_number(
        self, harness, ledger, tmp_path,
    ):
        _seed_in_window(harness, ['a sighting'])
        await _sweep(harness, ledger, tmp_path, FakeAdjudicator(lambda _text: Verdict.SAME))
        argv = [
            sys.executable, str(GATE_SCRIPT), '--gate', 'theta',
            '--report', 's', str(summary_path(tmp_path / 'memory-evals')),
        ]
        for require in THETA_SUMMARY_REQUIRES:
            argv += ['--require', *require]

        completed = subprocess.run(argv, capture_output=True, text=True, timeout=60)

        verdict = json.loads(completed.stdout.strip().splitlines()[-2])
        assert len(verdict['checks']) == 3
        for check in verdict['checks']:
            assert check['note'] is None, check
            assert isinstance(check['actual'], int | float), check
            assert not isinstance(check['actual'], bool), check
        assert [check['check'] for check in verdict['checks'] if not check['ok']] == [
            's:runs_7d >= 6',
        ]
