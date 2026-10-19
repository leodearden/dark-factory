"""The completion-record row, end to end (task 6961, plans/write-triage-link-healing-prd.md
H1 amendment 2026-10-19).

Write triage attached orchestrator completion records as children of other
records. A completion record is never a child, so the first deterministic row
detaches every linked one, whatever its kind and whether or not triage
contested it, and an agent-written child is never matched, whatever its
``task_id``. Each case runs against the in-process server in
``_link_heal_harness`` and observes the store payload, grouped reads, the write
journal and the ledger.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import pytest_asyncio
from _link_heal_harness import (
    ALL_LINK_HEAL_PREFIXES,
    CHILD,
    DEFAULT_LIMITS,
    DF,
    PARENT,
    PROJECTS,
    REIFY,
    FakeAdjudicator,
    LinkHealHarness,
    assert_store_invariants,
    build_harness,
    run_deterministic_apply,
    run_deterministic_plan,
)

from fused_memory.maintenance.link_heal import BasisSource, HealAction, LinkImage, Verdict
from fused_memory.maintenance.link_heal_executor import (
    RunReport,
    link_heal_agent_id,
    run_adjudicator_plan,
    run_undo,
)
from fused_memory.maintenance.link_heal_ledger import (
    ActionRow,
    ActionState,
    LinkHealLedger,
    RunSource,
)
from fused_memory.memory_metadata import ORCHESTRATOR_COMPLETION_SOURCE
from fused_memory.models.enums import MemoryCategory
from fused_memory.server.grouped_read import (
    AMENDMENT_KIND,
    CONTESTED_METADATA_KEY,
    PARENT_ID_KEY,
    SIGHTING_KIND,
)

COMPLETION_META = {'source': ORCHESTRATOR_COMPLETION_SOURCE, 'task_id': '4242'}
COMPLETION_TEXT = 'task 4242 completed: the link healer shipped'

REIFY_CHILD = '66666666-6666-4666-8666-666666666666'
REIFY_PARENT = '77777777-7777-4777-8777-777777777777'
AGENT_CHILD = '88888888-8888-4888-8888-888888888888'
AGENT_PARENT = '99999999-9999-4999-8999-999999999999'


@pytest_asyncio.fixture
async def harness(mock_config, tmp_path):
    built = await build_harness(
        mock_config, tmp_path, metadata_patch_prefixes=ALL_LINK_HEAL_PREFIXES,
    )
    yield built
    await built.journal.close()
    assert_store_invariants(built)


@pytest.fixture
def ledger(tmp_path):
    opened = LinkHealLedger(tmp_path / 'link_heal.db')
    yield opened
    opened.close()


def seed_contested_completion_child(harness: LinkHealHarness) -> None:
    """A completion record in DF that triage filed as a contested amendment of PARENT."""
    harness.seed_link(child_text=COMPLETION_TEXT, contested=True, meta=COMPLETION_META)


def seed_agent_cross_task_child(harness: LinkHealHarness) -> None:
    """An agent-written amendment in DF, filed under a parent from another task."""
    harness.seed(DF, AGENT_PARENT, 'an agent note written during task eight', task_id='8')
    harness.seed_link(
        child=AGENT_CHILD, parent=AGENT_PARENT, parent_text=None,
        child_text='an agent note written during task seven', meta={'task_id': '7'},
    )


def seed_store(harness: LinkHealHarness) -> None:
    seed_contested_completion_child(harness)
    harness.seed_link(
        kind=SIGHTING_KIND, child=REIFY_CHILD, parent=REIFY_PARENT, project=REIFY,
        child_text='task 4243 completed: reify shipped', parent_text='a reify note',
        meta={**COMPLETION_META, 'task_id': '4243'},
    )
    seed_agent_cross_task_child(harness)


async def plan_then_apply(
    harness: LinkHealHarness,
    ledger: LinkHealLedger,
    tmp_path: Path,
    *,
    between: Callable[[], None] | None = None,
) -> RunReport:
    await run_deterministic_plan(harness, ledger, tmp_path / 'plan.json')
    if between is not None:
        between()
    return await run_deterministic_apply(harness, ledger)


def only_outcome(ledger: LinkHealLedger, report: RunReport) -> ActionRow:
    (row,) = [
        row
        for run in ledger.recent_runs(10)
        for row in ledger.run_actions(run.run_id)
        if row.executed_run_id == report.run_id
    ]
    return row


def link_keys(harness: LinkHealHarness, project_id: str, memory_id: str) -> dict:
    return LinkImage.from_metadata(harness.mem0.payload(project_id, memory_id)).as_dict()


class TestTheDeterministicPlan:
    @pytest.mark.asyncio
    async def test_plans_exactly_the_completion_detaches_and_writes_nothing(
        self, harness, ledger, tmp_path,
    ):
        seed_store(harness)

        report = await run_deterministic_plan(harness, ledger, tmp_path / 'plan.json')

        rows = ledger.run_actions(report.run_id)
        assert sorted((row.planned.project_id, row.planned.child_id) for row in rows) == [
            (DF, CHILD), (REIFY, REIFY_CHILD),
        ]
        for row in rows:
            assert row.planned.action is HealAction.DETACH
            assert row.planned.basis_source is BasisSource.COMPLETION_RECORD
            assert row.planned.parent_sha256 is None
        assert dict(report.counts.planned_by_action) == {HealAction.DETACH.value: 2}
        assert report.counts.unexamined == 1
        assert report.counts.contested_reported == 0
        (run,) = ledger.recent_runs(10)
        assert (run.source, run.writes) == (RunSource.DETERMINISTIC, False)
        assert harness.mem0.write_count == 0


class TestApplyingTheDetach:
    @pytest.mark.asyncio
    async def test_the_contested_child_loses_every_link_key_and_keeps_its_provenance(
        self, harness, ledger, tmp_path,
    ):
        seed_store(harness)
        agent_before = dict(harness.mem0.payload(DF, AGENT_CHILD))

        report = await plan_then_apply(harness, ledger, tmp_path)

        payload = harness.mem0.payload(DF, CHILD)
        for key in (PARENT_ID_KEY, 'kind', CONTESTED_METADATA_KEY):
            assert key not in payload
        assert payload['task_id'] == '4242'
        assert payload['source'] == ORCHESTRATOR_COMPLETION_SOURCE
        assert payload['category'] == MemoryCategory.observations_and_summaries.value
        assert link_keys(harness, REIFY, REIFY_CHILD) == {}
        assert harness.mem0.payload(DF, AGENT_CHILD) == agent_before
        assert report.counts.applied == 2

    @pytest.mark.asyncio
    async def test_the_detached_record_surfaces_on_its_own(self, harness, ledger, tmp_path):
        seed_store(harness)

        await plan_then_apply(harness, ledger, tmp_path)

        search = await harness.call('search', query='link healer shipped', project_id=DF)
        assert [hit['id'] for hit in search['results']] == [CHILD]
        assert 'parent_unresolved' not in search['results'][0]
        parent = await harness.call('get_memory_by_id', project_id=DF, memory_id=PARENT)
        assert 'grouped' not in parent

    @pytest.mark.asyncio
    async def test_each_heal_is_one_attributed_delete_only_journal_row(
        self, harness, ledger, tmp_path,
    ):
        seed_store(harness)

        report = await plan_then_apply(harness, ledger, tmp_path)

        rows = harness.journal_rows()
        assert len(rows) == 2
        for row in rows:
            assert row['agent_id'] == link_heal_agent_id(report.run_id)
            assert row['causation_id'] == report.run_id
            assert 'metadata_patch' not in row['params']
        deletes = {row['params']['memory_id']: row['params']['metadata_delete_keys'] for row in rows}
        assert deletes == {
            CHILD: [PARENT_ID_KEY, 'kind', CONTESTED_METADATA_KEY],
            REIFY_CHILD: [PARENT_ID_KEY, 'kind'],
        }


class TestCorroboration:
    @pytest.mark.asyncio
    async def test_a_parent_deleted_since_the_plan_does_not_stop_the_detach(
        self, harness, ledger, tmp_path,
    ):
        seed_contested_completion_child(harness)

        report = await plan_then_apply(
            harness, ledger, tmp_path, between=lambda: harness.mem0.points.pop((DF, PARENT)),
        )

        assert only_outcome(ledger, report).state is ActionState.APPLIED
        assert link_keys(harness, DF, CHILD) == {}

    @pytest.mark.asyncio
    async def test_the_parent_is_never_read(self, harness, ledger, tmp_path):
        seed_contested_completion_child(harness)

        report = await plan_then_apply(
            harness, ledger, tmp_path, between=lambda: harness.mem0.read_failures.add(PARENT),
        )

        assert only_outcome(ledger, report).state is ActionState.APPLIED
        assert link_keys(harness, DF, CHILD) == {}

    @pytest.mark.asyncio
    async def test_a_record_no_longer_a_completion_record_is_skipped_stale(
        self, harness, ledger, tmp_path,
    ):
        seed_contested_completion_child(harness)

        def overwrite_source() -> None:
            harness.mem0.payload(DF, CHILD)['source'] = 'agent'

        report = await plan_then_apply(harness, ledger, tmp_path, between=overwrite_source)

        row = only_outcome(ledger, report)
        assert row.state is ActionState.SKIPPED_STALE
        assert row.detail == {'stale_field': 'source'}
        assert harness.mem0.write_count == 0


class TestUndo:
    @pytest.mark.asyncio
    async def test_undo_restores_the_triage_stamped_keys_and_a_replan_proposes_nothing(
        self, harness, ledger, tmp_path,
    ):
        seed_contested_completion_child(harness)
        applied = await plan_then_apply(harness, ledger, tmp_path)

        undone = await run_undo(
            ledger.undo_target(ledger.resolve_run(applied.run_id)),
            store=harness.store(), ledger=ledger, limits=DEFAULT_LIMITS,
        )

        assert undone.counts.applied == 1
        assert link_keys(harness, DF, CHILD) == {
            PARENT_ID_KEY: PARENT, 'kind': AMENDMENT_KIND, CONTESTED_METADATA_KEY: True,
        }
        replan = await run_deterministic_plan(harness, ledger, tmp_path / 'replan.json')
        assert replan.counts.undo_suppressed == 1
        assert ledger.run_actions(replan.run_id) == []


class TestAnAdjudicatorPlan:
    @pytest.mark.asyncio
    async def test_asks_only_about_the_agent_link_and_plans_the_completion_detach(
        self, harness, ledger, tmp_path,
    ):
        seed_contested_completion_child(harness)
        seed_agent_cross_task_child(harness)
        adjudicator = FakeAdjudicator(lambda _text: Verdict.SAME)

        report = await run_adjudicator_plan(
            adjudicate=adjudicator,
            store=harness.store(),
            census=harness.mem0,
            ledger=ledger,
            limits=DEFAULT_LIMITS,
            projects=PROJECTS,
            plan_path_for=lambda _run_id: tmp_path / 'plan.json',
        )

        assert [pair.key for pair in adjudicator.pairs] == [f'{DF}:{AGENT_CHILD}']
        (row,) = ledger.run_actions(report.run_id)
        assert row.planned.child_id == CHILD
        assert row.planned.basis_source is BasisSource.COMPLETION_RECORD
