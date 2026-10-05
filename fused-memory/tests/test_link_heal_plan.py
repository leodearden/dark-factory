"""Planning heals over a live store (task 6181, plans/write-triage-link-healing-prd.md H1).

Every case runs ``build_plan`` against the in-process server in
``_link_heal_harness``: the census enumerates the linked ids, and every value
the table reads comes back through the server's own tools. The last section
runs ``run_plan``, the non-writing run that ledgers the plan and writes its
document.
"""

from __future__ import annotations

import hashlib
import json

import pytest
import pytest_asyncio
from _link_heal_harness import (
    ALL_LINK_HEAL_PREFIXES,
    CHILD,
    CHILD_TEXT,
    DEFAULT_LIMITS,
    DF,
    PARENT,
    PARENT_TEXT,
    PROJECTS,
    REIFY,
    FailingCensus,
    LinkHealHarness,
    build_harness,
    corpus_basis,
    run_corpus_plan,
)

from fused_memory.maintenance.link_heal import (
    BasisSource,
    HealAction,
    LinkBasis,
    LinkImage,
    Plan,
    RunCounts,
    build_plan,
)
from fused_memory.maintenance.link_heal_executor import (
    BACKLOG_ANCHOR,
    PLAN_DOCUMENT_STOP,
    WRITE_FAILURE_ANCHOR,
    RunLimits,
    RunReport,
    render_plan_document,
    run_plan,
)
from fused_memory.maintenance.link_heal_ledger import ActionState, LinkHealLedger, RunSource
from fused_memory.maintenance.link_heal_store import (
    COUNT_TOOL,
    CensusFailed,
    LinkHealStore,
    text_sha256,
)
from fused_memory.server.grouped_read import (
    AMENDMENT_KIND,
    PARENT_ID_KEY,
    SIGHTING_KIND,
)

OTHER_PARENT = '33333333-3333-4333-8333-333333333333'
GRANDPARENT = '44444444-4444-4444-8444-444444444444'
GRANDCHILD = '55555555-5555-4555-8555-555555555555'
SECOND_CHILD = '66666666-6666-4666-8666-666666666666'


@pytest_asyncio.fixture
async def harness(mock_config, tmp_path):
    built = await build_harness(
        mock_config, tmp_path, metadata_patch_prefixes=ALL_LINK_HEAL_PREFIXES,
    )
    yield built
    await built.journal.close()


async def plan(harness: LinkHealHarness, *bases: LinkBasis) -> Plan:
    return await build_plan(
        bases, store=harness.store(), census=harness.mem0, projects=PROJECTS,
    )


class TestPlannedHeals:
    @pytest.mark.asyncio
    async def test_a_corpus_rated_misfiled_amendment_plans_one_detach(self, harness):
        harness.seed_link()
        basis = corpus_basis('RELATED')

        result = await plan(harness, basis)

        (action,) = result.actions
        assert action.project_id == DF
        assert action.child_id == CHILD
        assert action.action is HealAction.DETACH
        assert action.pre_image == LinkImage(parent_id=PARENT, kind=AMENDMENT_KIND)
        assert action.post_image == LinkImage(parent_id=None)
        assert action.basis_source is BasisSource.CORPUS
        assert action.basis_key == 'H001'
        assert action.child_sha256 == basis.child_sha256
        assert action.parent_sha256 == basis.parent_sha256
        counts = result.counts
        assert (counts.links_total, counts.examined, counts.adjudicated) == (1, 1, 1)
        assert counts.planned == 1
        assert dict(counts.planned_by_action) == {HealAction.DETACH.value: 1}

    @pytest.mark.asyncio
    async def test_an_unrated_link_to_a_deleted_parent_is_a_deterministic_detach(self, harness):
        harness.seed_link(parent_text=None)

        result = await plan(harness)

        (action,) = result.actions
        assert action.action is HealAction.DETACH
        assert action.basis_source is BasisSource.DETERMINISTIC
        assert action.child_sha256 == text_sha256(CHILD_TEXT)
        assert action.parent_sha256 is None
        assert result.counts.planned == 1
        assert result.counts.adjudicated == 0
        assert result.counts.unexamined == 0


class TestReportedLinks:
    @pytest.mark.asyncio
    async def test_a_parent_present_only_in_another_run_project_is_reported(self, harness):
        harness.seed_link(parent_text=None)
        harness.seed(REIFY, PARENT, PARENT_TEXT)

        result = await plan(harness, corpus_basis('RELATED'))

        assert result.actions == ()
        assert result.counts.cross_project_reported == 1
        assert result.counts.planned == 0

    @pytest.mark.asyncio
    async def test_a_parent_that_is_itself_a_link_is_a_chain(self, harness):
        harness.seed(DF, GRANDPARENT, 'the grandparent note')
        harness.seed_link(parent_text=None)
        harness.seed(
            DF, PARENT, PARENT_TEXT, **{PARENT_ID_KEY: GRANDPARENT, 'kind': AMENDMENT_KIND},
        )

        result = await plan(harness, corpus_basis('RELATED'))

        assert result.actions == ()
        assert result.counts.chain_reported == 1
        assert result.counts.unexamined == 1  # the parent's own, unrated link

    @pytest.mark.asyncio
    async def test_a_contested_child_rated_related_is_reported_not_detached(self, harness):
        harness.seed_link(contested=True)

        result = await plan(harness, corpus_basis('RELATED'))

        assert result.actions == ()
        assert result.counts.contested_reported == 1

    @pytest.mark.asyncio
    async def test_a_child_edited_since_export_is_a_stale_rating(self, harness):
        harness.seed_link(child_text='the child note, edited after export')

        result = await plan(harness, corpus_basis('RELATED'))

        assert result.actions == ()
        assert result.counts.stale_rating == 1
        assert result.counts.adjudicated == 0

    @pytest.mark.asyncio
    async def test_a_corpus_row_whose_child_moved_parent_is_unlinked(self, harness):
        harness.seed(DF, PARENT, PARENT_TEXT)
        harness.seed_link(parent=OTHER_PARENT, parent_text='another parent note')

        result = await plan(harness, corpus_basis('RELATED'))

        assert result.actions == ()
        assert result.counts.unexamined == 1
        assert result.counts.bases_unlinked == 1
        assert result.counts.adjudicated == 0

    @pytest.mark.asyncio
    async def test_an_unrated_live_link_is_unexamined(self, harness):
        harness.seed_link()

        result = await plan(harness)

        assert result.actions == ()
        assert result.counts.unexamined == 1
        assert result.counts.bases_unlinked == 0

    @pytest.mark.asyncio
    async def test_a_half_link_with_a_child_rated_extends_has_children(self, harness):
        harness.seed_link(kind=None)
        harness.seed(
            DF, GRANDCHILD, 'a note filed under the child',
            **{PARENT_ID_KEY: CHILD, 'kind': SIGHTING_KIND},
        )

        result = await plan(harness, corpus_basis('EXTENDS'))

        assert result.actions == ()
        assert result.counts.has_children == 1
        assert result.counts.chain_reported == 1  # the grandchild's link onto a link


async def plan_counting_children(harness: LinkHealHarness, basis: LinkBasis) -> tuple[Plan, int]:
    """The plan for *basis*, and how many children counts the server was asked for."""
    calls: list[tuple[str, dict]] = []
    result = await build_plan(
        (basis,),
        store=LinkHealStore(harness.recording_tool_caller(calls)),
        census=harness.mem0,
        projects=PROJECTS,
    )
    return result, sum(1 for tool, _arguments in calls if tool == COUNT_TOOL)


class TestChildrenAreCountedOnlyForAHalfLink:
    @pytest.mark.asyncio
    async def test_a_sighting_is_planned_without_a_children_count(self, harness):
        harness.seed_link(kind=SIGHTING_KIND)

        result, counted = await plan_counting_children(harness, corpus_basis('EXTENDS'))

        assert dict(result.counts.planned_by_action) == {HealAction.RELABEL.value: 1}
        assert counted == 0

    @pytest.mark.asyncio
    async def test_a_half_link_counts_its_children_once(self, harness):
        harness.seed_link(kind=None)

        result, counted = await plan_counting_children(harness, corpus_basis('EXTENDS'))

        assert dict(result.counts.planned_by_action) == {
            HealAction.COMPLETE_AMENDMENT.value: 1,
        }
        assert counted == 1


class TestReadFailures:
    @pytest.mark.asyncio
    async def test_a_parent_read_timeout_is_read_failed_and_plans_nothing(self, harness):
        harness.seed_link()
        harness.mem0.read_failures.add(PARENT)

        result = await plan(harness, corpus_basis('RELATED'))

        assert result.actions == ()
        assert result.counts.read_failed == 1
        assert result.counts.examined == 0
        assert result.counts.links_total == 1
        assert result.counts.bases_unlinked == 0


REPORT_COUNTERS = (
    'stale_rating', 'contested_reported', 'chain_reported', 'peer_reported',
    'has_children', 'cross_project_reported', 'unclear', 'no_action', 'unexamined',
)


class TestDisclosureIdentities:
    async def _mixed_store(self, harness: LinkHealHarness) -> Plan:
        harness.seed_link()
        harness.seed_link(child=SECOND_CHILD, parent=OTHER_PARENT, kind=SIGHTING_KIND,
            child_text='a second child', parent_text='a second parent',
        )
        harness.seed(
            DF, GRANDCHILD, 'an orphaned note',
            **{PARENT_ID_KEY: GRANDPARENT, 'kind': SIGHTING_KIND},
        )
        harness.seed(DF, '77777777-7777-4777-8777-777777777777', 'an unread note',
                     **{PARENT_ID_KEY: '88888888-8888-4888-8888-888888888888'})
        harness.mem0.read_failures.add('77777777-7777-4777-8777-777777777777')
        return await plan(
            harness,
            corpus_basis('RELATED'),
            corpus_basis(
                'EXTENDS', child=SECOND_CHILD, parent=OTHER_PARENT, key='H002',
                child_text='a second child', parent_text='a second parent',
            ),
            corpus_basis('SAME', child=GRANDPARENT, key='H003'),
        )

    @pytest.mark.asyncio
    async def test_every_link_is_examined_or_read_failed(self, harness):
        counts = (await self._mixed_store(harness)).counts

        assert counts.links_total == 4
        assert counts.links_total == counts.examined + counts.read_failed
        assert counts.read_failed == 1

    @pytest.mark.asyncio
    async def test_every_examined_link_has_exactly_one_disposition(self, harness):
        counts = (await self._mixed_store(harness)).counts

        reported = sum(getattr(counts, name) for name in REPORT_COUNTERS)
        assert counts.examined == counts.planned + reported

    @pytest.mark.asyncio
    async def test_adjudicated_counts_the_links_with_a_current_basis(self, harness):
        counts = (await self._mixed_store(harness)).counts

        assert counts.adjudicated == 2
        assert counts.bases_unlinked == 1

    @pytest.mark.asyncio
    async def test_planned_matches_the_actions_by_action(self, harness):
        result = await self._mixed_store(harness)

        assert result.counts.planned == len(result.actions) == 3
        by_action: dict[str, int] = {}
        for action in result.actions:
            by_action[action.action.value] = by_action.get(action.action.value, 0) + 1
        assert dict(result.counts.planned_by_action) == by_action == {
            HealAction.DETACH.value: 2, HealAction.RELABEL.value: 1,
        }

    @pytest.mark.asyncio
    async def test_planning_writes_nothing(self, harness):
        await self._mixed_store(harness)

        assert harness.mem0.write_count == 0
        assert harness.journal_rows() == []


class TestRunCounts:
    def test_a_fresh_disclosure_is_complete(self):
        assert RunCounts().complete is True

    @pytest.mark.parametrize(
        'partial',
        [
            {'read_failed': 1},
            {'failed': 1},
            {'skipped_cap': 1},
            {'not_attempted': 1},
            {'stopped_by': 'write_failure_streak'},
        ],
        ids=lambda partial: next(iter(partial)),
    )
    def test_a_partial_run_is_never_complete(self, partial):
        assert RunCounts(**partial).complete is False

    def test_skipped_stale_alone_is_still_complete(self):
        assert RunCounts(skipped_stale=3, applied=1).complete is True

    def test_as_json_is_plain_json_carrying_the_completeness(self):
        counts = RunCounts(
            planned=2,
            planned_by_action={HealAction.DETACH.value: 2},
            would_escape=('link-heal-backlog',),
            escaped=({'anchor': 'link-heal-backlog', 'escalation_id': 'esc-1'},),
            caps_bit=('max_actions_per_run',),
            skipped_cap=1,
        )

        document = json.loads(json.dumps(counts.as_json()))

        assert document['planned_by_action'] == {'detach': 2}
        assert document['would_escape'] == ['link-heal-backlog']
        assert document['escaped'] == [
            {'anchor': 'link-heal-backlog', 'escalation_id': 'esc-1'},
        ]
        assert document['caps_bit'] == ['max_actions_per_run']
        assert document['complete'] is False
        assert document['stopped_by'] is None


DANGLING_CHILD = '99999999-9999-4999-8999-999999999999'
DELETED_PARENT = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'
SECOND_TEXT, SECOND_PARENT_TEXT = 'a second child', 'a second parent'


@pytest.fixture
def ledger(tmp_path):
    opened = LinkHealLedger(tmp_path / 'link_heal.db')
    yield opened
    opened.close()


def seed_three_heals(harness: LinkHealHarness) -> tuple[LinkBasis, ...]:
    """A misfiled amendment, an EXTENDS sighting and a dangling link: three heals."""
    harness.seed_link()
    harness.seed_link(child=SECOND_CHILD, parent=OTHER_PARENT, kind=SIGHTING_KIND,
        child_text=SECOND_TEXT, parent_text=SECOND_PARENT_TEXT,
    )
    harness.seed_link(child=DANGLING_CHILD, parent=DELETED_PARENT, parent_text=None,
        child_text='a note whose parent was deleted',
    )
    return (
        corpus_basis('RELATED'),
        corpus_basis(
            'EXTENDS', child=SECOND_CHILD, parent=OTHER_PARENT, key='H002',
            child_text=SECOND_TEXT, parent_text=SECOND_PARENT_TEXT,
        ),
    )


class TestRunPlanLedgersANonWritingRun:
    @pytest.mark.asyncio
    async def test_records_exactly_one_finished_non_writing_run(self, harness, ledger, tmp_path):
        report = await run_corpus_plan(
            harness, ledger, tmp_path / 'plan.json', seed_three_heals(harness),
        )

        (run,) = ledger.recent_runs(10)
        assert run.run_id == report.run_id
        assert run.source is RunSource.CORPUS
        assert run.writes is False
        assert run.finished_at is not None
        assert run.counts == report.counts.as_json()
        assert run.plan_sha256 == report.plan_sha256

    @pytest.mark.asyncio
    async def test_every_planned_action_is_an_actions_row_in_state_planned(
        self, harness, ledger, tmp_path,
    ):
        bases = seed_three_heals(harness)
        expected = await plan(harness, *bases)

        report = await run_corpus_plan(harness, ledger, tmp_path / 'plan.json', bases)

        rows = ledger.run_actions(report.run_id)
        assert {row.state for row in rows} == {ActionState.PLANNED}
        assert [row.planned for row in rows] == list(expected.actions)
        assert report.counts.planned == 3

    @pytest.mark.asyncio
    async def test_the_plan_document_is_written_and_its_sha_ledgered(
        self, harness, ledger, tmp_path,
    ):
        plan_path = tmp_path / 'plan.json'

        report = await run_corpus_plan(harness, ledger, plan_path, seed_three_heals(harness))

        assert report.plan_path == plan_path
        written = plan_path.read_bytes()
        assert hashlib.sha256(written).hexdigest() == report.plan_sha256
        assert ledger.resolve_run(report.run_id).plan_sha256 == report.plan_sha256

    @pytest.mark.asyncio
    async def test_the_plan_document_is_the_rendering_of_the_pending_rows(
        self, harness, ledger, tmp_path,
    ):
        plan_path = tmp_path / 'plan.json'
        await run_corpus_plan(harness, ledger, plan_path, seed_three_heals(harness))

        rendered = render_plan_document(ledger.pending_actions(RunSource.CORPUS))

        assert rendered == plan_path.read_bytes()
        document = json.loads(rendered)
        assert document['format'] == 'link-heal-plan/1'
        assert len(document['actions']) == 3
        assert rendered.endswith(b'\n')

    @pytest.mark.asyncio
    async def test_a_plan_run_writes_nothing_to_the_store(self, harness, ledger, tmp_path):
        await run_corpus_plan(harness, ledger, tmp_path / 'plan.json', seed_three_heals(harness))

        assert harness.mem0.write_count == 0
        assert harness.journal_rows() == []


class TestRunPlanLeavesNothingHalfDone:
    @pytest.mark.asyncio
    async def test_a_failed_census_raises_before_any_run_row(self, harness, ledger, tmp_path):
        with pytest.raises(CensusFailed):
            await run_plan(
                seed_three_heals(harness),
                store=harness.store(),
                census=FailingCensus(),
                ledger=ledger,
                limits=DEFAULT_LIMITS,
                projects=PROJECTS,
                source=RunSource.CORPUS,
                plan_path=tmp_path / 'plan.json',
            )

        assert ledger.recent_runs(10) == []

    @pytest.mark.asyncio
    async def test_an_unwritable_document_rolls_back_the_new_heals(
        self, harness, ledger, tmp_path,
    ):
        report = await run_corpus_plan(
            harness, ledger, tmp_path / 'missing' / 'plan.json', seed_three_heals(harness),
        )

        assert ledger.pending_actions(RunSource.CORPUS) == []
        assert report.counts.stopped_by == PLAN_DOCUMENT_STOP
        assert report.counts.complete is False
        assert (report.plan_sha256, report.plan_path) == (None, None)
        (run,) = ledger.recent_runs(10)
        assert run.finished_at is not None
        assert run.plan_sha256 is None
        assert run.counts == report.counts.as_json()

    @pytest.mark.asyncio
    async def test_heals_already_pending_survive_an_unwritable_document(
        self, harness, ledger, tmp_path,
    ):
        bases = seed_three_heals(harness)
        first = await run_corpus_plan(harness, ledger, tmp_path / 'plan.json', bases)

        await run_corpus_plan(harness, ledger, tmp_path / 'missing' / 'plan.json', bases)

        assert ledger.pending_actions(RunSource.CORPUS) == ledger.run_actions(first.run_id)


class TestRePlanning:
    @pytest.mark.asyncio
    async def test_re_planning_an_unchanged_store_inserts_no_new_rows(
        self, harness, ledger, tmp_path,
    ):
        bases = seed_three_heals(harness)
        first = await run_corpus_plan(harness, ledger, tmp_path / 'first.json', bases)

        second = await run_corpus_plan(harness, ledger, tmp_path / 'second.json', bases)

        assert len(ledger.pending_actions(RunSource.CORPUS)) == 3
        assert ledger.run_actions(second.run_id) == []
        assert second.counts.already_pending == 3
        assert second.plan_sha256 == first.plan_sha256

    def _undo(self, ledger: LinkHealLedger, report: RunReport) -> None:
        """Mark every heal *report* planned as applied, then undone."""
        writer = ledger.start_run(RunSource.CORPUS, writes=True)
        undoer = ledger.start_run(RunSource.UNDO, writes=True)
        for row in ledger.run_actions(report.run_id):
            ledger.set_outcome(row.action_id, ActionState.APPLIED, writer, None)
            ledger.mark_undone(row.action_id, undoer)

    @pytest.mark.asyncio
    async def test_an_undone_heal_at_the_same_hashes_is_not_re_planned(
        self, harness, ledger, tmp_path,
    ):
        harness.seed_link()
        bases = (corpus_basis('RELATED'),)
        self._undo(ledger, await run_corpus_plan(harness, ledger, tmp_path / 'a.json', bases))

        again = await run_corpus_plan(harness, ledger, tmp_path / 'b.json', bases)

        assert again.counts.undo_suppressed == 1
        assert again.counts.planned == 0
        assert ledger.pending_actions(RunSource.CORPUS) == []

    @pytest.mark.asyncio
    async def test_editing_the_child_re_opens_a_corpus_heal_as_a_stale_rating(
        self, harness, ledger, tmp_path,
    ):
        harness.seed_link()
        bases = (corpus_basis('RELATED'),)
        self._undo(ledger, await run_corpus_plan(harness, ledger, tmp_path / 'a.json', bases))
        harness.mem0.payload(DF, CHILD)['data'] = 'the child note, edited after the undo'

        again = await run_corpus_plan(harness, ledger, tmp_path / 'b.json', bases)

        assert again.counts.undo_suppressed == 0
        assert again.counts.stale_rating == 1

    @pytest.mark.asyncio
    async def test_editing_the_child_re_opens_a_deterministic_detach(
        self, harness, ledger, tmp_path,
    ):
        harness.seed_link(parent_text=None)
        self._undo(ledger, await run_corpus_plan(harness, ledger, tmp_path / 'a.json', ()))
        harness.mem0.payload(DF, CHILD)['data'] = 'the child note, edited after the undo'

        again = await run_corpus_plan(harness, ledger, tmp_path / 'b.json', ())

        assert again.counts.undo_suppressed == 0
        (row,) = ledger.pending_actions(RunSource.CORPUS)
        assert row.planned.basis_source is BasisSource.DETERMINISTIC
        assert row.planned.child_sha256 == text_sha256('the child note, edited after the undo')


class TestReportModeEscapes:
    @pytest.mark.asyncio
    async def test_a_backlog_over_cap_times_multiplier_would_escape_and_files_nothing(
        self, harness, ledger, tmp_path,
    ):
        limits = RunLimits(max_actions_per_run=2, backlog_multiplier=1, write_failure_streak=3)

        report = await run_corpus_plan(
            harness, ledger, tmp_path / 'plan.json', seed_three_heals(harness), limits,
        )

        assert report.counts.would_escape == (BACKLOG_ANCHOR,)
        assert report.counts.escaped == ()

    @pytest.mark.asyncio
    async def test_a_backlog_at_cap_times_multiplier_would_not_escape(
        self, harness, ledger, tmp_path,
    ):
        limits = RunLimits(max_actions_per_run=3, backlog_multiplier=1, write_failure_streak=3)

        report = await run_corpus_plan(
            harness, ledger, tmp_path / 'plan.json', seed_three_heals(harness), limits,
        )

        assert report.counts.would_escape == ()

    @pytest.mark.asyncio
    async def test_a_non_writing_run_never_would_escape_a_write_failure(
        self, harness, ledger, tmp_path,
    ):
        limits = RunLimits(max_actions_per_run=0, backlog_multiplier=1, write_failure_streak=1)

        report = await run_corpus_plan(
            harness, ledger, tmp_path / 'plan.json', seed_three_heals(harness), limits,
        )

        assert WRITE_FAILURE_ANCHOR not in report.counts.would_escape
        assert report.counts.would_escape == (BACKLOG_ANCHOR,)
