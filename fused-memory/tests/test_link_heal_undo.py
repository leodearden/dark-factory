"""Undoing an apply run (task 6181, plans/write-triage-link-healing-prd.md H1 undo).

Each case plans, applies and undoes against the in-process server in
``_link_heal_harness``. An undo takes every heal the target run applied back
to its pre-image, newest first, one corroborated and verified write per step,
and owns the undone row that keeps a re-plan from proposing the heal again.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio
from _link_heal_harness import (
    ALL_LINK_HEAL_PREFIXES,
    CHILD,
    DEFAULT_LIMITS,
    DF,
    PARENT,
    LinkHealHarness,
    assert_store_invariants,
    build_harness,
    corpus_basis,
    run_corpus_apply,
    run_corpus_plan,
)

from fused_memory.maintenance.link_heal import BasisSource, LinkBasis, LinkImage
from fused_memory.maintenance.link_heal_executor import RunReport, run_undo
from fused_memory.maintenance.link_heal_ledger import (
    ActionRow,
    ActionState,
    LinkHealLedger,
    RunSource,
    UnknownRun,
)
from fused_memory.server.grouped_read import (
    AMENDMENT_KIND,
    CONTESTED_METADATA_KEY,
    PARENT_ID_KEY,
    SIGHTING_KIND,
)


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


async def plan_and_apply(
    harness: LinkHealHarness, ledger: LinkHealLedger, tmp_path: Path, *bases: LinkBasis,
) -> tuple[RunReport, RunReport]:
    planned = await run_corpus_plan(harness, ledger, tmp_path / 'plan.json', bases)
    applied = await run_corpus_apply(harness, ledger)
    assert applied.counts.applied == len(ledger.applied_actions(applied.run_id)) > 0
    return planned, applied


async def undo(harness: LinkHealHarness, ledger: LinkHealLedger, run_ref: str) -> RunReport:
    return await run_undo(
        ledger.resolve_run(run_ref), store=harness.store(), ledger=ledger, limits=DEFAULT_LIMITS,
    )


def the_heal(ledger: LinkHealLedger, plan: RunReport) -> ActionRow:
    (row,) = ledger.run_actions(plan.run_id)
    return row


def link_keys(harness: LinkHealHarness) -> dict:
    return LinkImage.from_metadata(harness.mem0.payload(DF, CHILD)).as_dict()


class TestUndoAHalfLinkCompletion:
    async def _undone(self, harness, ledger, tmp_path):
        harness.seed_link(kind=None)
        plan, applied = await plan_and_apply(harness, ledger, tmp_path, corpus_basis('SAME'))
        writes_before = harness.mem0.write_count
        report = await undo(harness, ledger, applied.run_id)
        return plan, report, writes_before

    @pytest.mark.asyncio
    async def test_one_delete_only_write_restores_the_pre_image(self, harness, ledger, tmp_path):
        _plan, report, writes_before = await self._undone(harness, ledger, tmp_path)

        assert harness.mem0.write_count == writes_before + 1
        assert harness.mem0.writes['delete_payload'] == 1
        assert link_keys(harness) == {PARENT_ID_KEY: PARENT}
        assert 'kind' not in harness.mem0.payload(DF, CHILD)
        undo_row = harness.journal_rows()[-1]
        assert undo_row['params']['metadata_delete_keys'] == ['kind']
        assert 'metadata_patch' not in undo_row['params']
        assert undo_row['agent_id'] == f'link-heal-{report.run_id[:8]}'
        assert report.counts.applied == 1

    @pytest.mark.asyncio
    async def test_the_heal_is_marked_undone_by_the_undo_run(self, harness, ledger, tmp_path):
        plan, report, _ = await self._undone(harness, ledger, tmp_path)

        heal = the_heal(ledger, plan)
        assert heal.state is ActionState.UNDONE
        assert heal.undone_run_id == report.run_id

    @pytest.mark.asyncio
    async def test_the_undo_run_ledgers_its_own_applied_step(self, harness, ledger, tmp_path):
        plan, report, _ = await self._undone(harness, ledger, tmp_path)

        run = ledger.resolve_run(report.run_id)
        assert run.source is RunSource.UNDO
        assert run.writes is True
        (step,) = ledger.undo_steps(report.run_id)
        assert step.state is ActionState.APPLIED
        assert step.basis_source is BasisSource.UNDO
        assert step.basis_key == str(the_heal(ledger, plan).action_id)
        assert step.original_action_id == the_heal(ledger, plan).action_id
        assert step.before == LinkImage(parent_id=PARENT, kind=SIGHTING_KIND)
        assert step.after == LinkImage(parent_id=PARENT)


class TestUndoADetach:
    async def _undone(self, harness, ledger, tmp_path):
        harness.seed_link()
        plan, applied = await plan_and_apply(harness, ledger, tmp_path, corpus_basis('RELATED'))
        report = await undo(harness, ledger, applied.run_id)
        return plan, applied, report

    @pytest.mark.asyncio
    async def test_one_patch_only_write_restores_parent_and_kind(self, harness, ledger, tmp_path):
        await self._undone(harness, ledger, tmp_path)

        assert link_keys(harness) == {PARENT_ID_KEY: PARENT, 'kind': AMENDMENT_KIND}
        undo_row = harness.journal_rows()[-1]
        assert undo_row['params']['metadata_patch'] == {
            PARENT_ID_KEY: PARENT, 'kind': AMENDMENT_KIND,
        }
        assert 'metadata_delete_keys' not in undo_row['params']
        child = await harness.call('get_memory_by_id', project_id=DF, memory_id=CHILD)
        assert child['metadata'][PARENT_ID_KEY] == PARENT

    @pytest.mark.asyncio
    async def test_undoing_the_same_run_again_writes_nothing(self, harness, ledger, tmp_path):
        _plan, applied, _first = await self._undone(harness, ledger, tmp_path)
        writes_before = harness.mem0.write_count

        again = await undo(harness, ledger, applied.run_id)

        assert harness.mem0.write_count == writes_before
        assert again.counts.applied == 0

    @pytest.mark.asyncio
    async def test_a_re_plan_does_not_propose_the_undone_detach(self, harness, ledger, tmp_path):
        await self._undone(harness, ledger, tmp_path)

        replan = await run_corpus_plan(
            harness, ledger, tmp_path / 'replan.json', (corpus_basis('RELATED'),),
        )

        assert replan.counts.undo_suppressed == 1
        assert ledger.pending_actions(RunSource.CORPUS) == []

    @pytest.mark.asyncio
    async def test_the_target_run_resolves_from_its_eight_hex_prefix(
        self, harness, ledger, tmp_path,
    ):
        harness.seed_link()
        _plan, applied = await plan_and_apply(harness, ledger, tmp_path, corpus_basis('RELATED'))

        report = await undo(harness, ledger, applied.run_id[:8])

        assert report.counts.applied == 1
        assert link_keys(harness) == {PARENT_ID_KEY: PARENT, 'kind': AMENDMENT_KIND}


class TestUndoARelabelAndFlag:
    @pytest.mark.asyncio
    async def test_deletes_the_flag_first_then_patches_the_kind_back(
        self, harness, ledger, tmp_path,
    ):
        harness.seed_link(kind=SIGHTING_KIND)
        _plan, applied = await plan_and_apply(harness, ledger, tmp_path, corpus_basis('CORRECTS'))
        assert harness.mem0.payload(DF, CHILD)[CONTESTED_METADATA_KEY] is True

        report = await undo(harness, ledger, applied.run_id)

        delete_row, patch_row = harness.journal_rows()[-2:]
        assert delete_row['params']['metadata_delete_keys'] == [CONTESTED_METADATA_KEY]
        assert patch_row['params']['metadata_patch'] == {'kind': SIGHTING_KIND}
        steps = ledger.undo_steps(report.run_id)
        assert [step.state for step in steps] == [ActionState.APPLIED, ActionState.APPLIED]
        assert steps[0].after == steps[1].before
        assert link_keys(harness) == {PARENT_ID_KEY: PARENT, 'kind': SIGHTING_KIND}
        assert (report.counts.planned, report.counts.applied) == (1, 1)


class TestResumingAPartUndoneHeal:
    """A relabel+flag undo whose second step (the kind patch) fails once."""

    async def _part_undone(self, harness, ledger, tmp_path):
        harness.seed_link(kind=SIGHTING_KIND)
        plan, applied = await plan_and_apply(harness, ledger, tmp_path, corpus_basis('CORRECTS'))
        harness.mem0.write_failures['set_payload'] = 1
        first = await undo(harness, ledger, applied.run_id)
        return plan, applied, first

    @pytest.mark.asyncio
    async def test_the_heal_counts_once_as_failed_and_stays_applied(
        self, harness, ledger, tmp_path,
    ):
        plan, _applied, first = await self._part_undone(harness, ledger, tmp_path)

        steps = ledger.undo_steps(first.run_id)
        assert [step.state for step in steps] == [ActionState.APPLIED, ActionState.FAILED]
        assert (first.counts.applied, first.counts.failed) == (0, 1)
        assert the_heal(ledger, plan).state is ActionState.APPLIED
        assert link_keys(harness) == {PARENT_ID_KEY: PARENT, 'kind': AMENDMENT_KIND}

    @pytest.mark.asyncio
    async def test_undoing_again_finishes_from_the_last_applied_step(
        self, harness, ledger, tmp_path,
    ):
        plan, applied, _first = await self._part_undone(harness, ledger, tmp_path)
        writes_before = harness.mem0.write_count

        again = await undo(harness, ledger, applied.run_id)

        (step,) = ledger.undo_steps(again.run_id)
        assert step.state is ActionState.APPLIED
        assert step.before == LinkImage(parent_id=PARENT, kind=AMENDMENT_KIND)
        assert step.after == LinkImage(parent_id=PARENT, kind=SIGHTING_KIND)
        assert harness.mem0.write_count == writes_before + 1
        assert link_keys(harness) == {PARENT_ID_KEY: PARENT, 'kind': SIGHTING_KIND}
        assert the_heal(ledger, plan).state is ActionState.UNDONE
        assert (again.counts.applied, again.counts.skipped_stale) == (1, 0)


class TestUndoCorroboration:
    @pytest.mark.asyncio
    async def test_a_record_re_edited_since_the_apply_is_skipped_stale(
        self, harness, ledger, tmp_path,
    ):
        harness.seed_link(kind=SIGHTING_KIND)
        plan, applied = await plan_and_apply(harness, ledger, tmp_path, corpus_basis('EXTENDS'))
        harness.mem0.payload(DF, CHILD)['kind'] = 'correction'
        writes_before = harness.mem0.write_count

        report = await undo(harness, ledger, applied.run_id)

        (step,) = ledger.undo_steps(report.run_id)
        assert step.state is ActionState.SKIPPED_STALE
        assert step.detail == {'stale_field': 'kind'}
        assert harness.mem0.write_count == writes_before
        assert the_heal(ledger, plan).state is ActionState.APPLIED
        assert report.counts.skipped_stale == 1

    @pytest.mark.asyncio
    async def test_an_unknown_run_is_refused_before_any_run_row(self, harness, ledger):
        with pytest.raises(UnknownRun):
            await undo(harness, ledger, 'f' * 32)

        assert ledger.recent_runs(10) == []
