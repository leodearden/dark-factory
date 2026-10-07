"""Heals planned from the link adjudicator's verdicts (task 6184, PRD H1/H2).

Every case runs against the in-process server in ``_link_heal_harness`` with a
real ledger on ``tmp_path``. The adjudicator is a :class:`FakeAdjudicator`, so
what is under test is which links reach it, what is ledgered from its answers,
and which escapes a run records.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
from pathlib import Path

import pytest
import pytest_asyncio
from _link_heal_harness import (
    ALL_LINK_HEAL_PREFIXES,
    CHILD,
    CHILD_TEXT,
    DF,
    FAKE_ADJUDICATOR_MODEL,
    PARENT,
    PARENT_TEXT,
    PROJECTS,
    REIFY,
    Answer,
    FakeAdjudicator,
    LinkHealHarness,
    build_harness,
)

from fused_memory.maintenance.link_adjudicator import AdjudicationFailure, LinkPair
from fused_memory.maintenance.link_heal import BasisSource, HealAction, Verdict
from fused_memory.maintenance.link_heal_executor import (
    ADJUDICATOR_ANCHOR,
    CORRECTS_SHARE_ANCHOR,
    MISFILE_SHARE_ANCHOR,
    RunLimits,
    RunReport,
    run_adjudicator_plan,
)
from fused_memory.maintenance.link_heal_ledger import LinkHealLedger, RunSource
from fused_memory.maintenance.link_heal_store import text_sha256
from fused_memory.server.grouped_read import SIGHTING_KIND

LIMITS = RunLimits(
    max_actions_per_run=25, backlog_multiplier=5, write_failure_streak=3,
    misfile_share_ceiling=0.25, corrects_share_ceiling=0.60,
)

STORM = {'count': 3, 'threshold': 3, 'labels': ['cli_failed'], 'model': 'opus'}


def _id(prefix: str, n: int = 0) -> str:
    return f'{prefix}-0000-4000-8000-{n:012d}'


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


def always(answer: Answer) -> FakeAdjudicator:
    return FakeAdjudicator(lambda _text: answer)


async def adjudicated_plan(
    harness: LinkHealHarness,
    ledger: LinkHealLedger,
    adjudicate: FakeAdjudicator,
    plan_path: Path,
    limits: RunLimits = LIMITS,
) -> RunReport:
    return await run_adjudicator_plan(
        adjudicate=adjudicate,
        store=harness.store(),
        census=harness.mem0,
        ledger=ledger,
        limits=limits,
        projects=PROJECTS,
        plan_path_for=lambda _run_id: plan_path,
    )


def seed_sightings(harness: LinkHealHarness, count: int) -> list[str]:
    """*count* sightings of PARENT, child texts ``sighting <n>``; their ids, in order."""
    children = [_id('aaaaaaaa', n) for n in range(count)]
    for n, child in enumerate(children):
        harness.seed_link(kind=SIGHTING_KIND, child=child, child_text=f'sighting {n}')
    return children


def scripted(words: dict[int, Verdict], default: Verdict = Verdict.EXTENDS) -> FakeAdjudicator:
    """Answer ``words[n]`` for ``sighting <n>``, else *default*."""
    return FakeAdjudicator(lambda text: words.get(int(text.split()[-1]), default))


class TestAnAdjudicatedVerdictPlansAHeal:
    @pytest.mark.asyncio
    async def test_an_extends_sighting_plans_one_relabel_on_a_ledgered_adjudication(
        self, harness, ledger, tmp_path,
    ):
        harness.seed_link(kind=SIGHTING_KIND)

        report = await adjudicated_plan(harness, ledger, always(Verdict.EXTENDS), tmp_path / 'p.json')

        (row,) = ledger.run_actions(report.run_id)
        assert row.planned.action is HealAction.RELABEL
        assert row.planned.basis_source is BasisSource.ADJUDICATOR
        adjudication = ledger.adjudication_at(
            DF, CHILD, PARENT, text_sha256(CHILD_TEXT), text_sha256(PARENT_TEXT),
        )
        assert adjudication is not None
        assert row.planned.basis_key == str(adjudication.adjudication_id)
        assert adjudication.run_id == report.run_id
        assert adjudication.record.verdict is Verdict.EXTENDS
        assert adjudication.record.reason == 'scripted EXTENDS'
        assert adjudication.record.model == FAKE_ADJUDICATOR_MODEL
        assert report.counts.adjudicated == 1
        assert report.counts.planned == 1

    @pytest.mark.asyncio
    async def test_the_run_is_a_non_writing_adjudicator_run(self, harness, ledger, tmp_path):
        harness.seed_link(kind=SIGHTING_KIND)

        report = await adjudicated_plan(harness, ledger, always(Verdict.EXTENDS), tmp_path / 'p.json')

        (run,) = ledger.recent_runs(1)
        assert run.run_id == report.run_id
        assert run.source is RunSource.ADJUDICATOR
        assert run.writes is False
        assert report.plan_path == tmp_path / 'p.json'
        assert report.plan_sha256 is not None


class TestOnlyVerdictRowLinksAreSent:
    @pytest.mark.asyncio
    async def test_deterministic_links_never_reach_the_adjudicator(self, harness, ledger, tmp_path):
        harness.seed_link(kind=SIGHTING_KIND)
        harness.seed_link(child=_id('bbbbbbbb', 1), contested=True)
        harness.seed_link(child=_id('bbbbbbbb', 2), parent=_id('dddddddd', 2), parent_text=None)
        harness.seed(DF, _id('eeeeeeee', 0), 'the root of a chain')
        harness.seed_link(
            child=_id('eeeeeeee', 1), parent=_id('eeeeeeee', 0), parent_text=None, contested=True,
        )
        harness.seed_link(child=_id('bbbbbbbb', 3), parent=_id('eeeeeeee', 1), parent_text=None)
        harness.seed(REIFY, _id('ffffffff', 0), 'a parent in another project')
        harness.seed_link(child=_id('bbbbbbbb', 4), parent=_id('ffffffff', 0), parent_text=None)
        fake = always(Verdict.EXTENDS)

        report = await adjudicated_plan(harness, ledger, fake, tmp_path / 'p.json')

        assert fake.pairs == [
            LinkPair(key=f'{DF}:{CHILD}', child_text=CHILD_TEXT, parent_text=PARENT_TEXT),
        ]
        assert [item.name for item in dataclasses.fields(LinkPair)] == [
            'key', 'child_text', 'parent_text',
        ]
        counts = report.counts
        assert (counts.contested_reported, counts.chain_reported, counts.cross_project_reported) == (
            2, 1, 1,
        )
        assert dict(counts.planned_by_action) == {
            HealAction.RELABEL.value: 1, HealAction.DETACH.value: 1,
        }


class TestJudgedPairsAreReused:
    @pytest.mark.asyncio
    async def test_a_pair_judged_at_its_current_hashes_is_not_sent_again(
        self, harness, ledger, tmp_path,
    ):
        harness.seed_link(kind=SIGHTING_KIND)
        await adjudicated_plan(harness, ledger, always(Verdict.EXTENDS), tmp_path / 'p.json')
        fake = always(Verdict.RELATED)

        second = await adjudicated_plan(harness, ledger, fake, tmp_path / 'p.json')

        assert fake.pairs == []
        assert second.counts.adjudications_reused == 1
        assert second.counts.adjudicated == 1
        assert second.counts.already_pending == 1
        assert dict(second.counts.planned_by_action) == {HealAction.RELABEL.value: 1}

    @pytest.mark.asyncio
    async def test_an_edited_child_is_adjudicated_again(self, harness, ledger, tmp_path):
        harness.seed_link(kind=SIGHTING_KIND)
        await adjudicated_plan(harness, ledger, always(Verdict.EXTENDS), tmp_path / 'p.json')
        harness.mem0.payload(DF, CHILD)['data'] = 'the child note, edited'
        fake = always(Verdict.RELATED)

        report = await adjudicated_plan(harness, ledger, fake, tmp_path / 'p.json')

        assert [pair.child_text for pair in fake.pairs] == ['the child note, edited']
        assert report.counts.adjudications_reused == 0
        assert dict(report.counts.planned_by_action) == {HealAction.DETACH.value: 1}


class TestAFailedAdjudication:
    @pytest.mark.asyncio
    async def test_is_counted_and_never_ledgered(self, harness, ledger, tmp_path):
        harness.seed_link(kind=SIGHTING_KIND)

        report = await adjudicated_plan(
            harness, ledger, always(AdjudicationFailure.PARSE_FAILURE), tmp_path / 'p.json',
        )

        assert ledger.adjudication_at(
            DF, CHILD, PARENT, text_sha256(CHILD_TEXT), text_sha256(PARENT_TEXT),
        ) is None
        assert ledger.adjudication_verdicts({report.run_id}) == []
        assert report.counts.adjudication_failed == 1
        assert report.counts.unexamined == 1
        assert report.counts.planned == 0
        assert report.counts.complete is False


class TestAPlanRunRecordsEscapesAndFilesNone:
    def test_a_plan_run_has_no_filer(self):
        assert 'filer' not in inspect.signature(run_adjudicator_plan).parameters

    @pytest.mark.asyncio
    async def test_an_adjudicator_storm_would_escape(self, harness, ledger, tmp_path):
        harness.seed_link(kind=SIGHTING_KIND)
        fake = FakeAdjudicator(lambda _text: AdjudicationFailure.CLI_FAILED, storm=STORM)

        report = await adjudicated_plan(harness, ledger, fake, tmp_path / 'p.json')

        assert report.counts.would_escape == (ADJUDICATOR_ANCHOR,)
        assert report.counts.escaped == ()

    @pytest.mark.asyncio
    async def test_a_misfile_share_over_its_ceiling_would_escape_and_still_plans(
        self, harness, ledger, tmp_path,
    ):
        seed_sightings(harness, 30)
        fake = scripted({n: Verdict.RELATED for n in range(12)})

        report = await adjudicated_plan(harness, ledger, fake, tmp_path / 'p.json')

        assert report.counts.would_escape == (MISFILE_SHARE_ANCHOR,)
        assert report.counts.escaped == ()
        assert report.counts.adjudicated == 30
        assert dict(report.counts.planned_by_action) == {
            HealAction.DETACH.value: 12, HealAction.RELABEL.value: 18,
        }
        document = json.loads((tmp_path / 'p.json').read_text(encoding='utf-8'))
        assert len(document['actions']) == 30

    @pytest.mark.asyncio
    async def test_fewer_than_twenty_adjudications_never_trip_a_share(
        self, harness, ledger, tmp_path,
    ):
        seed_sightings(harness, 19)
        fake = scripted({n: Verdict.RELATED for n in range(10)})

        report = await adjudicated_plan(harness, ledger, fake, tmp_path / 'p.json')

        assert report.counts.would_escape == ()

    @pytest.mark.asyncio
    async def test_a_corrects_share_over_its_ceiling_would_escape(self, harness, ledger, tmp_path):
        seed_sightings(harness, 30)
        fake = scripted({n: Verdict.CORRECTS for n in range(19)})

        report = await adjudicated_plan(harness, ledger, fake, tmp_path / 'p.json')

        assert report.counts.would_escape == (CORRECTS_SHARE_ANCHOR,)
