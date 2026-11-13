"""The scheduled link-heal sweep's records (task 7159).

Every case runs against the in-process server in ``_link_heal_harness`` with a
real ledger on ``tmp_path`` and a :class:`FakeAdjudicator` answering by child
text. At the executor seam a sweep is an adjudicator plan of its own source:
only the links whose child was created inside the run window are examined,
only the heals the sweep is enabled for are staged, and a writing sweep files
the adjudicator storm escape rather than only forecasting it.

End to end, :func:`run_sweep` resolves its window from the ledger, gates its
writes on the adjudicator's calibration, and leaves one report on disk and one
info escalation naming it, whether or not it wrote.
"""

from __future__ import annotations

import json
from collections.abc import Callable
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
    Answer,
    FakeAdjudicator,
    LinkHealHarness,
    RecordingFiler,
    build_harness,
)

from fused_memory.config.schema import LinkHealConfig
from fused_memory.maintenance.link_adjudicator import AdjudicationFailure
from fused_memory.maintenance.link_heal import BasisSource, HealAction, LinkImage, Report, Verdict
from fused_memory.maintenance.link_heal_executor import (
    ADJUDICATOR_ANCHOR,
    MISFILE_SHARE_ANCHOR,
    RUN_REPORT_ANCHOR,
    EscapeFiler,
    RunLimits,
    RunReport,
    pending_plan_sha256,
    run_adjudicator_plan,
    run_apply,
    run_undo,
)
from fused_memory.maintenance.link_heal_ledger import (
    NO_ACTION_TAKEN,
    ActionState,
    Disposition,
    LinkHealLedger,
    RunSource,
)
from fused_memory.maintenance.link_heal_report import WritesWithheld
from fused_memory.maintenance.link_heal_store import CreatedWindow
from fused_memory.maintenance.link_heal_sweep import SweepReport, SweepSettings, run_sweep
from fused_memory.models.scope import Scope
from fused_memory.server.grouped_read import (
    AMENDMENT_KIND,
    CONTESTED_METADATA_KEY,
    PARENT_ID_KEY,
    SIGHTING_KIND,
)

LIMITS = RunLimits(
    max_actions_per_run=25, backlog_multiplier=5, write_failure_streak=3,
    misfile_share_ceiling=0.25, corrects_share_ceiling=0.60,
)
SWEEP_ACTIONS = frozenset({HealAction.DETACH, HealAction.RELABEL})
WINDOW = CreatedWindow(
    since=datetime(2026, 11, 1, tzinfo=UTC), until=datetime(2026, 11, 2, tzinfo=UTC),
)
INSIDE = '2026-11-01T12:00:00+00:00'
BEFORE = '2026-10-01T12:00:00+00:00'
STORM = {'count': 3, 'threshold': 3, 'labels': ['cli_failed'], 'model': 'opus'}


def _id(prefix: str, n: int) -> str:
    return f'{prefix}-0000-4000-8000-{n:012d}'


MISFILED = _id('aaaaaaaa', 1)
EXTENDS = _id('aaaaaaaa', 2)
AMENDMENT_CORRECTS = _id('aaaaaaaa', 3)
SIGHTING_CORRECTS = _id('aaaaaaaa', 4)
OLD = _id('aaaaaaaa', 5)
TRIAGE_CONTESTED = _id('aaaaaaaa', 6)
HAND_CONTESTED = _id('aaaaaaaa', 7)

ANSWERS: dict[str, Answer] = {
    'misfiled sighting': Verdict.RELATED,
    'extending sighting': Verdict.EXTENDS,
    'correcting amendment': Verdict.CORRECTS,
    'correcting sighting': Verdict.CORRECTS,
    'an old misfiled sighting': Verdict.RELATED,
    'a triage-contested sighting': Verdict.RELATED,
    'a hand-contested sighting': Verdict.RELATED,
}


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


def _seed(harness: LinkHealHarness) -> None:
    links = (
        (MISFILED, 'misfiled sighting', SIGHTING_KIND, INSIDE, False, False),
        (EXTENDS, 'extending sighting', SIGHTING_KIND, INSIDE, False, False),
        (AMENDMENT_CORRECTS, 'correcting amendment', AMENDMENT_KIND, INSIDE, False, False),
        (SIGHTING_CORRECTS, 'correcting sighting', SIGHTING_KIND, INSIDE, False, False),
        (OLD, 'an old misfiled sighting', SIGHTING_KIND, BEFORE, False, False),
        (TRIAGE_CONTESTED, 'a triage-contested sighting', SIGHTING_KIND, INSIDE, False, True),
        (HAND_CONTESTED, 'a hand-contested sighting', SIGHTING_KIND, INSIDE, True, False),
    )
    for n, (child, text, kind, created_at, by_hand, by_triage) in enumerate(links):
        harness.seed_link(
            kind=kind, child=child, parent=_id('bbbbbbbb', n), child_text=text,
            parent_text=f'the parent of {text}', meta={'created_at': created_at},
            contested=by_hand, contested_by_triage=by_triage,
        )


async def _sweep_plan(
    harness: LinkHealHarness,
    ledger: LinkHealLedger,
    adjudicate: FakeAdjudicator,
    tmp_path: Path,
    *,
    filer: EscapeFiler | None = None,
) -> RunReport:
    return await run_adjudicator_plan(
        adjudicate=adjudicate,
        store=harness.store(),
        census=harness.mem0,
        ledger=ledger,
        limits=LIMITS,
        projects=PROJECTS,
        plan_path_for=lambda _run_id: tmp_path / 'sweep-plan.json',
        source=RunSource.SWEEP,
        enabled_actions=SWEEP_ACTIONS,
        created_in=WINDOW,
        filer=filer,
    )


def _answering() -> FakeAdjudicator:
    return FakeAdjudicator(ANSWERS.__getitem__)


class TestTheSweepPlan:
    @pytest.mark.asyncio
    async def test_the_run_is_a_sweep_run(self, harness, ledger, tmp_path):
        _seed(harness)

        report = await _sweep_plan(harness, ledger, _answering(), tmp_path)

        run = ledger.resolve_run(report.run_id)
        assert run.source is RunSource.SWEEP
        assert not run.writes

    @pytest.mark.asyncio
    async def test_only_children_created_in_the_window_are_examined_or_asked(
        self, harness, ledger, tmp_path,
    ):
        _seed(harness)
        adjudicate = _answering()

        report = await _sweep_plan(harness, ledger, adjudicate, tmp_path)

        examined = {record.child_id for record in ledger.run_records(report.run_id)}
        assert OLD not in examined
        assert examined == {
            MISFILED, EXTENDS, AMENDMENT_CORRECTS, SIGHTING_CORRECTS,
            TRIAGE_CONTESTED, HAND_CONTESTED,
        }
        asked = {pair.child_text for pair in adjudicate.pairs}
        assert 'an old misfiled sighting' not in asked
        assert 'a hand-contested sighting' not in asked

    @pytest.mark.asyncio
    async def test_a_heal_the_sweep_is_not_enabled_for_is_recorded_never_pending(
        self, harness, ledger, tmp_path,
    ):
        _seed(harness)

        report = await _sweep_plan(harness, ledger, _answering(), tmp_path)

        records = {record.child_id: record for record in ledger.run_records(report.run_id)}
        for child, decision in (
            (AMENDMENT_CORRECTS, HealAction.FLAG),
            (SIGHTING_CORRECTS, HealAction.RELABEL_FLAG),
        ):
            assert records[child].decision is decision
            assert records[child].disposition is Disposition.NOT_ENABLED
            assert records[child].action_id is None
        assert report.counts.not_enabled == 2
        assert report.counts.planned == 3
        assert dict(report.counts.planned_by_action) == {'detach': 2, 'relabel': 1}

    @pytest.mark.asyncio
    async def test_only_detach_and_relabel_heals_are_pending_and_only_under_the_sweep(
        self, harness, ledger, tmp_path,
    ):
        _seed(harness)

        await _sweep_plan(harness, ledger, _answering(), tmp_path)

        pending = {
            (row.planned.child_id, row.planned.action) for row in ledger.pending_actions(RunSource.SWEEP)
        }
        assert pending == {
            (MISFILED, HealAction.DETACH),
            (EXTENDS, HealAction.RELABEL),
            (TRIAGE_CONTESTED, HealAction.DETACH),
        }
        assert ledger.pending_actions(RunSource.ADJUDICATOR) == []

    @pytest.mark.asyncio
    async def test_contests_carry_their_6533_outcomes_through_the_sweep(
        self, harness, ledger, tmp_path,
    ):
        _seed(harness)

        report = await _sweep_plan(harness, ledger, _answering(), tmp_path)

        records = {record.child_id: record for record in ledger.run_records(report.run_id)}
        triage = records[TRIAGE_CONTESTED]
        assert (triage.decision, triage.disposition) == (HealAction.DETACH, Disposition.STAGED)
        hand = records[HAND_CONTESTED]
        assert (hand.decision, hand.disposition) == (
            Report.CONTESTED_REPORTED, Disposition.REPORTED,
        )
        assert harness.mem0.payload(DF, HAND_CONTESTED)[CONTESTED_METADATA_KEY] is True


class TestTheSweepFilesItsStorm:
    @pytest.mark.asyncio
    async def test_a_writing_sweep_files_the_storm_once_and_discloses_it(
        self, harness, ledger, tmp_path,
    ):
        _seed(harness)
        filer = RecordingFiler()

        report = await _sweep_plan(
            harness, ledger, FakeAdjudicator(ANSWERS.__getitem__, storm=STORM), tmp_path,
            filer=filer,
        )

        assert [escape.anchor for escape in filer.escapes] == [ADJUDICATOR_ANCHOR]
        assert [dict(entry) for entry in report.counts.escaped] == [
            {'anchor': ADJUDICATOR_ANCHOR, 'escalation_id': 'esc-recorded-1'},
        ]
        assert ADJUDICATOR_ANCHOR not in report.counts.would_escape

    @pytest.mark.asyncio
    async def test_without_a_filer_the_storm_is_only_forecast(self, harness, ledger, tmp_path):
        _seed(harness)

        report = await _sweep_plan(
            harness, ledger, FakeAdjudicator(ANSWERS.__getitem__, storm=STORM), tmp_path,
        )

        assert ADJUDICATOR_ANCHOR in report.counts.would_escape
        assert report.counts.escaped == ()


NOW = datetime(2026, 11, 2, tzinfo=UTC)
DAY = timedelta(hours=24)


def _settings(
    *, writes: bool = False, dry_run: bool = False, projects: tuple[str, ...] = PROJECTS,
) -> SweepSettings:
    return SweepSettings(
        projects=projects, writes=writes, dry_run=dry_run, actions=SWEEP_ACTIONS, window_hours=24,
    )


def _calibrate(ledger: LinkHealLedger, *, passed: bool, prompt_sha256: str = FAKE_PROMPT_SHA256) -> str:
    run_id = ledger.start_run(RunSource.CALIBRATION, writes=False)
    ledger.add_calibration(
        run_id, model=FAKE_ADJUDICATOR_MODEL, prompt_sha256=prompt_sha256,
        population_sha256='0' * 64, passed=passed, report={'passed': passed},
    )
    return run_id


async def _sweep(
    harness: LinkHealHarness,
    ledger: LinkHealLedger,
    tmp_path: Path,
    *,
    settings: SweepSettings,
    filer: RecordingFiler,
    adjudicate: FakeAdjudicator | None = None,
    now: datetime = NOW,
    since: datetime | None = None,
    limits: RunLimits = LIMITS,
    reports_dir: Path | None = None,
    plan_path_for: Callable[[str], Path] | None = None,
) -> SweepReport:
    return await run_sweep(
        adjudicate=adjudicate or _answering(),
        store=harness.store(),
        census=harness.mem0,
        ledger=ledger,
        limits=limits,
        settings=settings,
        filer=filer,
        reports_dir=tmp_path / 'reports' if reports_dir is None else reports_dir,
        now=now,
        since=since,
        plan_path_for=plan_path_for,
    )


def _sweep_runs(ledger: LinkHealLedger, *, writes: bool) -> list[str]:
    return [
        run.run_id for run in ledger.recent_runs(50)
        if run.source is RunSource.SWEEP and run.writes is writes
    ]


def _report_json(report: SweepReport) -> dict:
    return json.loads(report.json_path.read_text())


class TestADryRunSweep:
    @pytest.mark.asyncio
    async def test_it_plans_once_and_never_applies(self, harness, ledger, tmp_path):
        _seed(harness)

        report = await _sweep(harness, ledger, tmp_path, settings=_settings(), filer=RecordingFiler())

        assert _sweep_runs(ledger, writes=False) == [report.plan_run_id]
        assert _sweep_runs(ledger, writes=True) == []
        assert report.apply_run_id is None
        assert harness.mem0.write_count == 0

    @pytest.mark.asyncio
    async def test_every_record_took_no_action_and_kept_its_before_state(
        self, harness, ledger, tmp_path,
    ):
        _seed(harness)

        report = await _sweep(harness, ledger, tmp_path, settings=_settings(), filer=RecordingFiler())

        records = ledger.run_records(report.plan_run_id)
        assert len(records) == 6
        for record in records:
            assert record.action_taken == NO_ACTION_TAKEN
            assert record.after_child == record.before_child

    @pytest.mark.asyncio
    async def test_it_leaves_a_report_and_one_info_escalation_naming_it(
        self, harness, ledger, tmp_path,
    ):
        _seed(harness)
        filer = RecordingFiler()

        report = await _sweep(harness, ledger, tmp_path, settings=_settings(), filer=filer)

        assert report.json_path.parent == tmp_path / 'reports'
        assert report.json_path.is_file()
        assert report.report_path.suffix == '.md'
        assert report.report_path.is_file()
        run_reports = [escape for escape in filer.escapes if escape.anchor == RUN_REPORT_ANCHOR]
        assert len(run_reports) == 1
        assert str(report.report_path) in run_reports[0].summary
        assert report.escalation_id == f'esc-recorded-{len(filer.escapes)}'

    @pytest.mark.asyncio
    async def test_the_report_says_dry_run_and_names_no_undo(self, harness, ledger, tmp_path):
        _seed(harness)

        report = await _sweep(harness, ledger, tmp_path, settings=_settings(), filer=RecordingFiler())

        document = _report_json(report)
        assert document['writes'] == {'mode': 'dry_run', 'withheld': WritesWithheld.WRITES_OFF.value}
        assert document['undo_run_id'] is None
        assert document['plan_run_id'] == report.plan_run_id
        assert '- Writes: dry-run' in report.report_path.read_text()

    @pytest.mark.asyncio
    async def test_a_dry_run_forecasts_the_storm_rather_than_filing_it(
        self, harness, ledger, tmp_path,
    ):
        _seed(harness)
        filer = RecordingFiler()

        await _sweep(
            harness, ledger, tmp_path, settings=_settings(), filer=filer,
            adjudicate=FakeAdjudicator(ANSWERS.__getitem__, storm=STORM),
        )

        assert ADJUDICATOR_ANCHOR not in [escape.anchor for escape in filer.escapes]


class TestWithheldWrites:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('calibration', 'reason'),
        [
            (None, WritesWithheld.UNCALIBRATED),
            ('failed', WritesWithheld.CALIBRATION_FAILED),
            ('another prompt', WritesWithheld.UNCALIBRATED),
        ],
    )
    async def test_writes_wait_for_a_passing_calibration_of_this_adjudicator(
        self, harness, ledger, tmp_path, calibration, reason,
    ):
        _seed(harness)
        if calibration == 'failed':
            _calibrate(ledger, passed=False)
        elif calibration == 'another prompt':
            _calibrate(ledger, passed=True, prompt_sha256='f' * 64)

        report = await _sweep(
            harness, ledger, tmp_path, settings=_settings(writes=True), filer=RecordingFiler(),
        )

        assert report.apply_run_id is None
        assert _sweep_runs(ledger, writes=True) == []
        assert harness.mem0.write_count == 0
        document = _report_json(report)
        assert document['writes'] == {'mode': 'withheld', 'withheld': reason.value}
        assert f'- Writes: withheld ({reason.value})' in report.report_path.read_text()

    @pytest.mark.asyncio
    async def test_the_dry_run_override_withholds_a_calibrated_writing_sweep(
        self, harness, ledger, tmp_path,
    ):
        _seed(harness)
        _calibrate(ledger, passed=True)

        report = await _sweep(
            harness, ledger, tmp_path, settings=_settings(writes=True, dry_run=True),
            filer=RecordingFiler(),
        )

        assert report.apply_run_id is None
        assert _sweep_runs(ledger, writes=True) == []
        assert harness.mem0.write_count == 0
        assert _report_json(report)['writes'] == {
            'mode': 'dry_run', 'withheld': WritesWithheld.DRY_RUN.value,
        }


class TestAllowedWrites:
    @pytest.mark.asyncio
    async def test_one_apply_run_applies_the_staged_heals(self, harness, ledger, tmp_path):
        _seed(harness)
        calibration_run = _calibrate(ledger, passed=True)

        report = await _sweep(
            harness, ledger, tmp_path, settings=_settings(writes=True), filer=RecordingFiler(),
        )

        assert _sweep_runs(ledger, writes=True) == [report.apply_run_id]
        applied = {
            (row.planned.child_id, row.planned.action)
            for row in ledger.applied_actions(report.apply_run_id)
        }
        assert applied == {
            (MISFILED, HealAction.DETACH),
            (EXTENDS, HealAction.RELABEL),
            (TRIAGE_CONTESTED, HealAction.DETACH),
        }
        document = _report_json(report)
        assert document['undo_run_id'] == report.apply_run_id
        assert document['calibration']['calibration_run_id'] == calibration_run
        assert f'link_heal.py undo --run {report.apply_run_id}' in report.report_path.read_text()

    @pytest.mark.asyncio
    async def test_the_records_show_each_heal_and_its_after_state(self, harness, ledger, tmp_path):
        _seed(harness)
        _calibrate(ledger, passed=True)

        report = await _sweep(
            harness, ledger, tmp_path, settings=_settings(writes=True), filer=RecordingFiler(),
        )

        post_images = {
            row.action_id: row.planned.post_image
            for row in ledger.applied_actions(report.apply_run_id)
        }
        records = {record.child_id: record for record in ledger.run_records(report.plan_run_id)}
        for child, action in (
            (MISFILED, 'detach'), (EXTENDS, 'relabel'), (TRIAGE_CONTESTED, 'detach'),
        ):
            record = records[child]
            assert record.action_taken == action
            assert record.after_child is not None
            assert record.after_child.link_image() == post_images[record.action_id]
        assert records[AMENDMENT_CORRECTS].action_taken == NO_ACTION_TAKEN

    @pytest.mark.asyncio
    async def test_undoing_the_apply_run_restores_every_before_state(
        self, harness, ledger, tmp_path,
    ):
        _seed(harness)
        _calibrate(ledger, passed=True)
        report = await _sweep(
            harness, ledger, tmp_path, settings=_settings(writes=True), filer=RecordingFiler(),
        )
        assert report.apply_run_id is not None

        await run_undo(
            ledger.undo_target(ledger.resolve_run(report.apply_run_id)),
            store=harness.store(), ledger=ledger, limits=LIMITS,
        )

        for record in ledger.run_records(report.plan_run_id):
            restored = LinkImage.from_metadata(harness.mem0.payload(DF, record.child_id))
            assert restored == record.before_child.link_image()

    @pytest.mark.asyncio
    async def test_an_undone_heal_keeps_its_action_and_after_state_and_names_its_undo(
        self, harness, ledger, tmp_path,
    ):
        _seed(harness)
        _calibrate(ledger, passed=True)
        report = await _sweep(
            harness, ledger, tmp_path, settings=_settings(writes=True), filer=RecordingFiler(),
        )
        assert report.apply_run_id is not None

        undo = await run_undo(
            ledger.undo_target(ledger.resolve_run(report.apply_run_id)),
            store=harness.store(), ledger=ledger, limits=LIMITS,
        )

        post_images = {
            row.action_id: row.planned.post_image for row in ledger.run_actions(report.plan_run_id)
        }
        records = {record.child_id: record for record in ledger.run_records(report.plan_run_id)}
        for child, action in (
            (MISFILED, 'detach'), (EXTENDS, 'relabel'), (TRIAGE_CONTESTED, 'detach'),
        ):
            record = records[child]
            assert record.action_taken == action
            assert record.after_child is not None
            assert record.after_child.link_image() == post_images[record.action_id]
            assert record.undone_run_id == undo.run_id
        assert records[AMENDMENT_CORRECTS].action_taken == NO_ACTION_TAKEN
        assert records[AMENDMENT_CORRECTS].undone_run_id is None
        for record in records.values():
            restored = LinkImage.from_metadata(harness.mem0.payload(DF, record.child_id))
            assert restored == record.before_child.link_image()

    @pytest.mark.asyncio
    async def test_a_writing_sweep_files_its_storm(self, harness, ledger, tmp_path):
        _seed(harness)
        _calibrate(ledger, passed=True)
        filer = RecordingFiler()

        await _sweep(
            harness, ledger, tmp_path, settings=_settings(writes=True), filer=filer,
            adjudicate=FakeAdjudicator(ANSWERS.__getitem__, storm=STORM),
        )

        assert [escape.anchor for escape in filer.escapes] == [
            ADJUDICATOR_ANCHOR, RUN_REPORT_ANCHOR,
        ]


SECOND_PROMPT_SHA256 = 'e' * 64
DANGLING = _id('aaaaaaaa', 8)
STAGED_HEALS = {
    (MISFILED, HealAction.DETACH),
    (EXTENDS, HealAction.RELABEL),
    (TRIAGE_CONTESTED, HealAction.DETACH),
}


class TestAWritingSweepAfterDryRunNights:
    @pytest.mark.asyncio
    async def test_its_report_counts_the_heals_earlier_plans_staged(
        self, harness, ledger, tmp_path,
    ):
        _seed(harness)
        await _sweep(harness, ledger, tmp_path, settings=_settings(), filer=RecordingFiler())
        _calibrate(ledger, passed=True)

        report = await _sweep(
            harness, ledger, tmp_path, settings=_settings(writes=True), filer=RecordingFiler(),
            now=NOW + DAY,
        )

        assert ledger.run_records(report.plan_run_id) == []
        counts = _report_json(report)['counts']
        assert counts['examined'] == 0
        assert counts['carried_from_earlier_plans'] == 3
        assert counts['by_action'] == {'detach': 2, 'relabel': 1}
        assert counts['by_verdict'] == {'RELATED': 2, 'EXTENDS': 1}
        assert counts['by_project'] == {DF: {'detach': 2, 'relabel': 1}}
        assert _report_json(report)['records_ref']['apply_run_id'] == report.apply_run_id
        assert 'No links were examined.' not in report.report_path.read_text()


class TestAWriteThatLandedButFailedVerification:
    @pytest.mark.asyncio
    async def test_its_record_shows_the_verification_read_and_the_failed_state(
        self, harness, ledger, tmp_path,
    ):
        _seed(harness)
        _calibrate(ledger, passed=True)
        _tamper_every_write(harness)

        report = await _sweep(
            harness, ledger, tmp_path, settings=_settings(writes=True), filer=RecordingFiler(),
        )

        record = {r.child_id: r for r in ledger.run_records(report.plan_run_id)}[MISFILED]
        assert record.action_state is ActionState.FAILED
        assert record.action_taken == NO_ACTION_TAKEN
        assert record.after_child is not None
        assert record.after_child.metadata.get(TAMPER_KEY) is True
        assert TAMPER_KEY not in record.before_child.metadata
        assert _report_json(report)['counts']['by_action_state'][ActionState.FAILED.value] == 3


TAMPER_KEY = 'x_tampered_after_write'


def _tamper_every_write(harness: LinkHealHarness) -> None:
    """Land each write, then put the record back as it was plus a marker, so the
    write lands and its verification read differs from both the before- and the
    intended after-state."""
    mem0 = harness.mem0
    for route in ('set_payload', 'delete_payload'):
        original = getattr(mem0, route)

        async def tampered(
            memory_id: str, change: object, scope: Scope, _original=original,
        ) -> None:
            key = (scope.project_id, memory_id)
            before = dict(mem0.points[key])
            await _original(memory_id, change, scope)
            mem0.points[key] = {**before, TAMPER_KEY: True}

        setattr(mem0, route, tampered)


class TestAParentHostedInAProjectTheSweepDoesNotExamine:
    @pytest.mark.asyncio
    async def test_is_reported_cross_project_and_never_detached(self, harness, ledger, tmp_path):
        child, parent = _id('cccccccc', 1), _id('dddddddd', 1)
        harness.seed(REIFY, parent, 'a parent hosted in reify')
        harness.seed_link(
            kind=SIGHTING_KIND, child=child, parent=parent, project=DF, parent_text=None,
            child_text='a cross-project sighting', meta={'created_at': INSIDE},
        )
        _calibrate(ledger, passed=True)
        settings = SweepSettings(
            projects=(DF,), presence_projects=(REIFY,), writes=True, dry_run=False,
            actions=SWEEP_ACTIONS, window_hours=24,
        )

        report = await _sweep(
            harness, ledger, tmp_path, settings=settings, filer=RecordingFiler(),
            adjudicate=FakeAdjudicator(lambda _text: Verdict.RELATED),
        )

        (record,) = ledger.run_records(report.plan_run_id)
        assert record.decision is Report.CROSS_PROJECT_REPORTED
        assert record.action_state is None
        assert ledger.applied_actions(report.apply_run_id) == []
        assert harness.mem0.payload(DF, child)[PARENT_ID_KEY] == parent


class TestTheWriteTimeCalibrationGate:
    @pytest.mark.asyncio
    async def test_heals_staged_under_a_failed_calibration_are_withheld_after_another_prompt_passes(
        self, harness, ledger, tmp_path,
    ):
        _seed(harness)
        failed = _calibrate(ledger, passed=False)
        first = await _sweep(
            harness, ledger, tmp_path, settings=_settings(writes=True), filer=RecordingFiler(),
        )
        assert first.apply_run_id is None
        assert _report_json(first)['writes']['withheld'] == WritesWithheld.CALIBRATION_FAILED.value
        _calibrate(ledger, passed=True, prompt_sha256=SECOND_PROMPT_SHA256)

        second = await _sweep(
            harness, ledger, tmp_path, settings=_settings(writes=True), filer=RecordingFiler(),
            adjudicate=FakeAdjudicator(ANSWERS.__getitem__, prompt_sha256=SECOND_PROMPT_SHA256),
            now=NOW + DAY,
        )

        assert second.apply_run_id is not None
        assert harness.mem0.write_count == 0
        staged = ledger.run_actions(first.plan_run_id)
        assert {(row.planned.child_id, row.planned.action) for row in staged} == STAGED_HEALS
        for row in staged:
            assert row.state is ActionState.WITHHELD_UNCALIBRATED
            assert row.executed_run_id == second.plan_run_id
            assert row.detail == {
                'model': FAKE_ADJUDICATOR_MODEL,
                'prompt_sha256': FAKE_PROMPT_SHA256,
                'calibration': 'failed',
                'calibration_run_id': failed,
            }
        assert ledger.pending_actions(RunSource.SWEEP) == []
        assert second.view.plan.counts.withheld_uncalibrated == len(staged)
        assert f'withheld {len(staged)} heal(s)' in second.report_path.read_text()

    @pytest.mark.asyncio
    async def test_a_recalibrated_adjudicator_s_verdicts_heal_links_a_stale_basis_had_staged(
        self, harness, ledger, tmp_path,
    ):
        _seed(harness)
        _calibrate(ledger, passed=False)
        first = await _sweep(
            harness, ledger, tmp_path, settings=_settings(writes=True), filer=RecordingFiler(),
        )
        assert first.apply_run_id is None
        _calibrate(ledger, passed=True, prompt_sha256=SECOND_PROMPT_SHA256)

        second = await _sweep(
            harness, ledger, tmp_path, settings=_settings(writes=True), filer=RecordingFiler(),
            adjudicate=FakeAdjudicator(ANSWERS.__getitem__, prompt_sha256=SECOND_PROMPT_SHA256),
            now=NOW + DAY, since=WINDOW.since,
        )

        assert second.apply_run_id is not None
        applied = {
            (row.planned.child_id, row.planned.action)
            for row in ledger.applied_actions(second.apply_run_id)
        }
        assert applied == STAGED_HEALS
        assert all(
            row.state is ActionState.WITHHELD_UNCALIBRATED
            for row in ledger.run_actions(first.plan_run_id)
        )
        records = {r.child_id: r for r in ledger.run_records(second.plan_run_id)}
        assert records[MISFILED].prompt_sha256 == SECOND_PROMPT_SHA256
        assert records[MISFILED].disposition is Disposition.STAGED
        assert records[MISFILED].action_state is ActionState.APPLIED

    @pytest.mark.asyncio
    async def test_staged_heals_apply_once_their_own_adjudicator_passes(
        self, harness, ledger, tmp_path,
    ):
        _seed(harness)
        first = await _sweep(
            harness, ledger, tmp_path, settings=_settings(writes=True), filer=RecordingFiler(),
        )
        assert first.apply_run_id is None
        assert _report_json(first)['writes']['withheld'] == WritesWithheld.UNCALIBRATED.value
        _calibrate(ledger, passed=True)

        second = await _sweep(
            harness, ledger, tmp_path, settings=_settings(writes=True), filer=RecordingFiler(),
            now=NOW + DAY,
        )

        assert second.apply_run_id is not None
        applied = {
            (row.planned.child_id, row.planned.action)
            for row in ledger.applied_actions(second.apply_run_id)
        }
        assert applied == STAGED_HEALS
        assert second.view.apply is not None
        assert second.view.apply.counts.withheld_uncalibrated == 0

    @pytest.mark.asyncio
    async def test_an_attended_sweep_apply_withholds_uncalibrated_heals_but_not_deterministic_ones(
        self, harness, ledger, tmp_path,
    ):
        _seed(harness)
        harness.seed_link(
            child=DANGLING, parent=_id('bbbbbbbb', 8), child_text='a dangling amendment',
            parent_text=None, meta={'created_at': INSIDE},
        )
        planned = await _sweep(
            harness, ledger, tmp_path, settings=_settings(dry_run=True), filer=RecordingFiler(),
        )

        report = await run_apply(
            store=harness.store(), ledger=ledger, limits=LIMITS, filer=RecordingFiler(),
            source=RunSource.SWEEP,
            approved_plan_sha256=pending_plan_sha256(ledger.pending_actions(RunSource.SWEEP)),
        )

        rows = ledger.run_actions(planned.plan_run_id)
        adjudicated = [row for row in rows if row.planned.basis_source is BasisSource.ADJUDICATOR]
        assert {(row.planned.child_id, row.planned.action) for row in adjudicated} == STAGED_HEALS
        for row in adjudicated:
            assert row.state is ActionState.WITHHELD_UNCALIBRATED
            assert row.executed_run_id == report.run_id
        (deterministic,) = [row for row in rows if row.planned.basis_source is not BasisSource.ADJUDICATOR]
        assert (deterministic.planned.child_id, deterministic.state) == (DANGLING, ActionState.APPLIED)
        assert harness.mem0.write_count == 1
        assert report.counts.withheld_uncalibrated == len(adjudicated)
        assert report.counts.applied == 1


class TestThePlanTimeCalibrationGate:
    """A plan run that files its escapes is one about to be applied, so for a
    calibration-gated source it withholds stale-basis heals before staging its own."""

    @pytest.mark.asyncio
    async def test_a_gated_plan_about_to_be_applied_withholds_stale_basis_heals_first(
        self, harness, ledger, tmp_path,
    ):
        _seed(harness)
        first = await _sweep_plan(harness, ledger, _answering(), tmp_path)
        _calibrate(ledger, passed=True, prompt_sha256=SECOND_PROMPT_SHA256)

        second = await _sweep_plan(
            harness, ledger,
            FakeAdjudicator(ANSWERS.__getitem__, prompt_sha256=SECOND_PROMPT_SHA256),
            tmp_path, filer=RecordingFiler(),
        )

        staged = ledger.run_actions(first.run_id)
        assert {(row.planned.child_id, row.planned.action) for row in staged} == STAGED_HEALS
        for row in staged:
            assert row.state is ActionState.WITHHELD_UNCALIBRATED
            assert row.executed_run_id == second.run_id
        assert second.counts.withheld_uncalibrated == len(staged)
        records = {r.child_id: r for r in ledger.run_records(second.run_id)}
        assert records[MISFILED].disposition is Disposition.STAGED

    @pytest.mark.asyncio
    async def test_an_ungated_plan_never_withholds(self, harness, ledger, tmp_path):
        _seed(harness)
        first = await run_adjudicator_plan(
            adjudicate=_answering(), store=harness.store(), census=harness.mem0, ledger=ledger,
            limits=LIMITS, projects=PROJECTS, plan_path_for=lambda _run_id: tmp_path / 'a.json',
        )

        second = await run_adjudicator_plan(
            adjudicate=FakeAdjudicator(ANSWERS.__getitem__, prompt_sha256=SECOND_PROMPT_SHA256),
            store=harness.store(), census=harness.mem0, ledger=ledger, limits=LIMITS,
            projects=PROJECTS, plan_path_for=lambda _run_id: tmp_path / 'b.json',
            filer=RecordingFiler(),
        )

        assert second.counts.withheld_uncalibrated == 0
        assert all(row.state is ActionState.PLANNED for row in ledger.run_actions(first.run_id))


def _second_prompt_answering(misfiled: Answer) -> FakeAdjudicator:
    """The second calibrated prompt: it agrees with the first on every link but
    MISFILED's, on which it answers *misfiled*."""
    return FakeAdjudicator(
        lambda text: misfiled if text == 'misfiled sighting' else ANSWERS[text],
        prompt_sha256=SECOND_PROMPT_SHA256,
    )


class TestANewerRulingAtTheSameTexts:
    """Two calibrated prompts disagree on MISFILED at the texts it shows: the first
    staged a detach on a dry night, the second rules otherwise on a writing night."""

    async def _two_nights(
        self, harness, ledger, tmp_path, misfiled: Answer,
    ) -> tuple[SweepReport, SweepReport]:
        _seed(harness)
        _calibrate(ledger, passed=True)
        first = await _sweep(harness, ledger, tmp_path, settings=_settings(), filer=RecordingFiler())
        _calibrate(ledger, passed=True, prompt_sha256=SECOND_PROMPT_SHA256)
        second = await _sweep(
            harness, ledger, tmp_path, settings=_settings(writes=True), filer=RecordingFiler(),
            adjudicate=_second_prompt_answering(misfiled), now=NOW + DAY, since=WINDOW.since,
        )
        return first, second

    @staticmethod
    def _first_detach(ledger: LinkHealLedger, first: SweepReport):
        (detach,) = [
            row for row in ledger.run_actions(first.plan_run_id) if row.planned.child_id == MISFILED
        ]
        return detach

    @pytest.mark.asyncio
    async def test_a_ruling_that_plans_no_heal_supersedes_the_pending_one(
        self, harness, ledger, tmp_path,
    ):
        first, second = await self._two_nights(harness, ledger, tmp_path, Verdict.SAME)

        detach = self._first_detach(ledger, first)
        assert detach.state is ActionState.SUPERSEDED
        assert detach.executed_run_id == second.plan_run_id
        record = {r.child_id: r for r in ledger.run_records(second.plan_run_id)}[MISFILED]
        assert (record.verdict, record.decision) == (Verdict.SAME, Report.NO_ACTION)
        assert detach.detail is not None
        assert detach.detail['superseded_by_run'] == second.plan_run_id
        assert detach.detail['basis_key'] == str(record.adjudication_id)
        assert detach.detail['decision'] == Report.NO_ACTION.value
        assert second.apply_run_id is not None
        applied = {row.planned.child_id for row in ledger.applied_actions(second.apply_run_id)}
        assert MISFILED not in applied
        assert applied == {EXTENDS, TRIAGE_CONTESTED}
        assert harness.mem0.payload(DF, MISFILED)[PARENT_ID_KEY] == _id('bbbbbbbb', 0)
        assert second.view.plan.counts.superseded == 1

    @pytest.mark.asyncio
    async def test_a_ruling_that_plans_another_heal_supersedes_the_pending_one_and_stages_its_own(
        self, harness, ledger, tmp_path,
    ):
        first, second = await self._two_nights(harness, ledger, tmp_path, Verdict.EXTENDS)

        assert self._first_detach(ledger, first).state is ActionState.SUPERSEDED
        assert second.apply_run_id is not None
        applied = {
            (row.planned.child_id, row.planned.action)
            for row in ledger.applied_actions(second.apply_run_id)
        }
        assert (MISFILED, HealAction.RELABEL) in applied
        assert (MISFILED, HealAction.DETACH) not in applied

    @pytest.mark.asyncio
    async def test_no_verdict_rules_on_nothing_so_the_pending_heal_stands(
        self, harness, ledger, tmp_path,
    ):
        first, second = await self._two_nights(
            harness, ledger, tmp_path, AdjudicationFailure.CLI_FAILED,
        )

        detach = self._first_detach(ledger, first)
        assert detach.state is ActionState.APPLIED
        assert detach.executed_run_id == second.apply_run_id
        assert second.view.plan.counts.superseded == 0


class TestASweepWhosePlanStopped:
    @pytest.mark.asyncio
    async def test_writes_nothing_not_even_earlier_pending_heals_and_says_why(
        self, harness, ledger, tmp_path,
    ):
        _seed(harness)
        _calibrate(ledger, passed=True)
        await _sweep(harness, ledger, tmp_path, settings=_settings(), filer=RecordingFiler())
        assert ledger.pending_actions(RunSource.SWEEP)

        report = await _sweep(
            harness, ledger, tmp_path, settings=_settings(writes=True), filer=RecordingFiler(),
            now=NOW + DAY, plan_path_for=lambda _run_id: tmp_path / 'missing' / 'plan.json',
        )

        assert report.view.plan.counts.stopped_by is not None
        assert report.apply_run_id is None
        assert _sweep_runs(ledger, writes=True) == []
        assert harness.mem0.write_count == 0
        assert report.view.withheld is WritesWithheld.PLAN_STOPPED
        assert _report_json(report)['writes'] == {
            'mode': 'withheld', 'withheld': WritesWithheld.PLAN_STOPPED.value,
        }


class TestASweepReportThatCannotBeWritten:
    @pytest.mark.asyncio
    async def test_is_still_escalated_naming_the_failure(self, harness, ledger, tmp_path):
        _seed(harness)
        blocker = tmp_path / 'a-regular-file'
        blocker.write_text('')
        filer = RecordingFiler()

        report = await _sweep(
            harness, ledger, tmp_path, settings=_settings(), filer=filer,
            reports_dir=blocker / 'reports',
        )

        assert report.report_unwritten
        assert not report.report_path.exists()
        (escape,) = [escape for escape in filer.escapes if escape.anchor == RUN_REPORT_ANCHOR]
        assert escape.detail['report_unwritten'] == report.report_unwritten
        assert 'could not be written' in escape.summary
        assert report.escalation_id == f'esc-recorded-{len(filer.escapes)}'


class TestTheReportNamesEveryHealTheRunActedOn:
    @pytest.mark.asyncio
    async def test_heals_withheld_at_plan_time_are_counted_with_their_oldest_child(
        self, harness, ledger, tmp_path,
    ):
        _seed(harness)
        _calibrate(ledger, passed=False)
        await _sweep(harness, ledger, tmp_path, settings=_settings(), filer=RecordingFiler())
        _calibrate(ledger, passed=True, prompt_sha256=SECOND_PROMPT_SHA256)

        report = await _sweep(
            harness, ledger, tmp_path, settings=_settings(writes=True), filer=RecordingFiler(),
            adjudicate=FakeAdjudicator(ANSWERS.__getitem__, prompt_sha256=SECOND_PROMPT_SHA256),
            now=NOW + DAY,
        )

        document = _report_json(report)
        assert document['counts']['examined'] == 0
        assert document['counts']['by_action_state'] == {
            ActionState.WITHHELD_UNCALIBRATED.value: len(STAGED_HEALS),
        }
        assert document['withheld_heals'] == {
            'count': len(STAGED_HEALS), 'oldest_child_created_at': INSIDE,
        }
        assert f'oldest child created {INSIDE}' in report.report_path.read_text()

    @pytest.mark.asyncio
    async def test_a_dry_run_that_supersedes_a_heal_counts_it(self, harness, ledger, tmp_path):
        _seed(harness)
        await _sweep(harness, ledger, tmp_path, settings=_settings(), filer=RecordingFiler())

        report = await _sweep(
            harness, ledger, tmp_path, settings=_settings(), filer=RecordingFiler(),
            adjudicate=_second_prompt_answering(Verdict.SAME), now=NOW + DAY, since=WINDOW.since,
        )

        counts = _report_json(report)['counts']
        assert counts['by_action_state'][ActionState.SUPERSEDED.value] == 1
        assert counts['carried_from_earlier_plans'] == 1


def _seed_misfile_heavy(harness: LinkHealHarness) -> None:
    """Twenty in-window sighting links, eight of them misfiled: a 0.40 misfile share."""
    for n in range(20):
        text = f'misfiled sighting {n}' if n < 8 else f'belonging sighting {n}'
        harness.seed_link(
            kind=SIGHTING_KIND, child=_id('cccccccc', n), parent=_id('dddddddd', n),
            child_text=text, parent_text=f'the parent of {text}', meta={'created_at': INSIDE},
        )


class TestTheShareEscape:
    @pytest.mark.asyncio
    async def test_a_misfile_heavy_sweep_files_the_share_escape_and_writes_nothing(
        self, harness, ledger, tmp_path,
    ):
        _seed_misfile_heavy(harness)
        _calibrate(ledger, passed=True)
        filer = RecordingFiler()
        adjudicate = FakeAdjudicator(
            lambda text: Verdict.RELATED if text.startswith('misfiled') else Verdict.SAME,
        )

        report = await _sweep(
            harness, ledger, tmp_path, settings=_settings(writes=True), filer=filer,
            adjudicate=adjudicate,
        )

        assert MISFILE_SHARE_ANCHOR in [escape.anchor for escape in filer.escapes]
        assert harness.mem0.write_count == 0
        assert report.apply_run_id is not None
        records = ledger.run_records(report.plan_run_id)
        assert {record.action_taken for record in records} == {NO_ACTION_TAKEN}
        assert _report_json(report)['undo_run_id'] is None


def _seed_one(
    harness: LinkHealHarness, n: int, text: str, created_at: str, *, project: str = DF,
) -> str:
    child = _id('eeeeeeee', n)
    harness.seed_link(
        kind=SIGHTING_KIND, child=child, parent=_id('ffffffff', n), project=project,
        child_text=text, parent_text=f'the parent of {text}', meta={'created_at': created_at},
    )
    return child


def _window(report: SweepReport) -> tuple[datetime, datetime]:
    document = _report_json(report)
    return (
        datetime.fromisoformat(document['window']['since']),
        datetime.fromisoformat(document['window']['until']),
    )


class TestTheWindow:
    @pytest.mark.asyncio
    async def test_the_first_sweep_looks_back_its_window_hours(self, harness, ledger, tmp_path):
        report = await _sweep(harness, ledger, tmp_path, settings=_settings(), filer=RecordingFiler())

        assert _window(report) == (NOW - DAY, NOW)

    @pytest.mark.asyncio
    async def test_the_next_sweep_starts_where_the_last_one_ended(self, harness, ledger, tmp_path):
        await _sweep(harness, ledger, tmp_path, settings=_settings(), filer=RecordingFiler())
        later = NOW + timedelta(hours=6)

        report = await _sweep(
            harness, ledger, tmp_path, settings=_settings(), filer=RecordingFiler(), now=later,
        )

        assert _window(report) == (NOW, later)

    @pytest.mark.asyncio
    async def test_an_explicit_since_overrides_the_watermark(self, harness, ledger, tmp_path):
        await _sweep(harness, ledger, tmp_path, settings=_settings(), filer=RecordingFiler())
        backlog = datetime(2026, 1, 1, tzinfo=UTC)
        later = NOW + DAY

        report = await _sweep(
            harness, ledger, tmp_path, settings=_settings(), filer=RecordingFiler(),
            now=later, since=backlog,
        )

        assert _window(report) == (backlog, later)

    @pytest.mark.asyncio
    async def test_an_explicit_since_reaches_children_the_window_would_miss(
        self, harness, ledger, tmp_path,
    ):
        _seed(harness)

        report = await _sweep(
            harness, ledger, tmp_path, settings=_settings(), filer=RecordingFiler(),
            since=datetime(2026, 9, 1, tzinfo=UTC),
        )

        assert OLD in {record.child_id for record in ledger.run_records(report.plan_run_id)}

    @pytest.mark.asyncio
    @pytest.mark.parametrize('failure', ['adjudication', 'read'])
    async def test_an_incomplete_sweep_does_not_advance_the_watermark(
        self, harness, ledger, tmp_path, failure,
    ):
        await _sweep(harness, ledger, tmp_path, settings=_settings(), filer=RecordingFiler())
        second = NOW + DAY
        child = _seed_one(harness, 1, 'an unjudgeable sighting', '2026-11-02T12:00:00+00:00')
        if failure == 'read':
            harness.mem0.read_failures.add(child)
        failing = FakeAdjudicator(lambda _text: AdjudicationFailure.CLI_FAILED)

        incomplete = await _sweep(
            harness, ledger, tmp_path, settings=_settings(), filer=RecordingFiler(),
            adjudicate=failing, now=second,
        )
        assert not incomplete.complete
        third = second + DAY

        report = await _sweep(
            harness, ledger, tmp_path, settings=_settings(), filer=RecordingFiler(),
            adjudicate=FakeAdjudicator(lambda _text: Verdict.SAME), now=third,
        )

        assert _window(report) == (NOW, third)

    @pytest.mark.asyncio
    async def test_an_incomplete_first_sweep_leaves_its_whole_window_for_the_next(
        self, harness, ledger, tmp_path,
    ):
        child = _seed_one(harness, 1, 'an unjudgeable sighting', INSIDE)
        failing = FakeAdjudicator(lambda _text: AdjudicationFailure.CLI_FAILED)
        first = await _sweep(
            harness, ledger, tmp_path, settings=_settings(), filer=RecordingFiler(),
            adjudicate=failing,
        )
        assert not first.complete
        second = NOW + DAY

        report = await _sweep(
            harness, ledger, tmp_path, settings=_settings(), filer=RecordingFiler(),
            adjudicate=FakeAdjudicator(lambda _text: Verdict.SAME), now=second,
        )

        assert _window(report) == (NOW - DAY, second)
        assert child in {record.child_id for record in ledger.run_records(report.plan_run_id)}

    @pytest.mark.asyncio
    async def test_a_one_project_sweep_leaves_the_other_projects_window_open(
        self, harness, ledger, tmp_path,
    ):
        same = FakeAdjudicator(lambda _text: Verdict.SAME)
        await _sweep(harness, ledger, tmp_path, settings=_settings(), filer=RecordingFiler())
        child = _seed_one(
            harness, 1, 'a reify sighting', (NOW + timedelta(hours=2)).isoformat(), project=REIFY,
        )
        await _sweep(
            harness, ledger, tmp_path, settings=_settings(projects=(DF,)), filer=RecordingFiler(),
            adjudicate=same, now=NOW + timedelta(hours=6),
        )
        last = NOW + timedelta(hours=12)

        report = await _sweep(
            harness, ledger, tmp_path, settings=_settings(), filer=RecordingFiler(),
            adjudicate=same, now=last,
        )

        assert _window(report) == (NOW, last)
        assert child in {record.child_id for record in ledger.run_records(report.plan_run_id)}

    @pytest.mark.asyncio
    async def test_a_later_since_does_not_skip_the_gap_before_it(self, harness, ledger, tmp_path):
        same = FakeAdjudicator(lambda _text: Verdict.SAME)
        await _sweep(harness, ledger, tmp_path, settings=_settings(), filer=RecordingFiler())
        child = _seed_one(harness, 1, 'a gap sighting', (NOW + timedelta(hours=2)).isoformat())
        await _sweep(
            harness, ledger, tmp_path, settings=_settings(), filer=RecordingFiler(),
            adjudicate=same, now=NOW + timedelta(hours=6), since=NOW + timedelta(hours=4),
        )
        nightly = NOW + timedelta(hours=12)

        report = await _sweep(
            harness, ledger, tmp_path, settings=_settings(), filer=RecordingFiler(),
            adjudicate=same, now=nightly,
        )

        assert _window(report) == (NOW, nightly)
        assert child in {record.child_id for record in ledger.run_records(report.plan_run_id)}

    @pytest.mark.asyncio
    async def test_a_complete_sweep_says_so(self, harness, ledger, tmp_path):
        _seed(harness)

        report = await _sweep(harness, ledger, tmp_path, settings=_settings(), filer=RecordingFiler())

        assert report.complete


class TestSweepSettings:
    def test_it_reads_the_config_leaves(self):
        config = LinkHealConfig(projects=[DF], writes=True, sweep_window_hours=6)

        settings = SweepSettings.from_config(config, dry_run=True)

        assert settings == SweepSettings(
            projects=(DF,), writes=True, dry_run=True, actions=SWEEP_ACTIONS, window_hours=6,
        )

    def test_a_projects_override_replaces_the_configured_projects(self):
        config = LinkHealConfig(projects=[DF])

        assert SweepSettings.from_config(config, projects=['reify']).projects == ('reify',)

    def test_a_projects_override_still_looks_for_parents_in_the_configured_projects(self):
        config = LinkHealConfig(projects=[DF, REIFY])

        settings = SweepSettings.from_config(config, projects=[REIFY])

        assert settings.presence_projects == (REIFY, DF)

    def test_a_sweep_over_no_project_is_refused(self):
        with pytest.raises(ValueError, match='projects'):
            SweepSettings.from_config(LinkHealConfig())

    def test_an_unknown_sweep_action_is_refused_by_name(self):
        config = LinkHealConfig(projects=[DF], sweep_actions=['detach', 'explode'])

        with pytest.raises(ValueError, match='explode') as raised:
            SweepSettings.from_config(config)

        for action in HealAction:
            assert action.value in str(raised.value)
