"""The scheduled link-heal sweep's records (task 7159).

Every case runs against the in-process server in ``_link_heal_harness`` with a
real ledger on ``tmp_path`` and a :class:`FakeAdjudicator` answering by child
text. At the executor seam a sweep is an adjudicator plan of its own source:
only the links whose child was created inside the run window are examined,
only the heals the sweep is enabled for are staged, and a writing sweep files
the adjudicator storm escape rather than only forecasting it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
import pytest_asyncio
from _link_heal_harness import (
    ALL_LINK_HEAL_PREFIXES,
    DF,
    PROJECTS,
    Answer,
    FakeAdjudicator,
    LinkHealHarness,
    RecordingFiler,
    build_harness,
)

from fused_memory.maintenance.link_heal import HealAction, Report, Verdict
from fused_memory.maintenance.link_heal_executor import (
    ADJUDICATOR_ANCHOR,
    EscapeFiler,
    RunLimits,
    RunReport,
    run_adjudicator_plan,
)
from fused_memory.maintenance.link_heal_ledger import Disposition, LinkHealLedger, RunSource
from fused_memory.maintenance.link_heal_store import CreatedWindow
from fused_memory.server.grouped_read import AMENDMENT_KIND, CONTESTED_METADATA_KEY, SIGHTING_KIND

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
