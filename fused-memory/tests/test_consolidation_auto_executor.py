"""The auto-consolidation executor post-step (task 5240, PRD C4).

``reconciliation/consolidation_auto_executor.py::run_auto_consolidation`` runs
once per full Stage-1 cycle: it reads the proposal rows, ranks them under the
shared flood policy, reads each selected cluster deterministically, rules with
the pure predicate and executes, gates or records the verdict. The stores are
the stateful fakes in ``tests/_auto_consolidation_fakes.py``; the ledger and
the escalation queue are real.
"""

from __future__ import annotations

import json

import pytest
import pytest_asyncio
from _auto_consolidation_fakes import (
    NOW,
    PROJECT_ID,
    VALID_CLAIM,
    FakeMemoryService,
    FakeTaskBackend,
    enabled_config,
    propose,
)
from escalation.queue import EscalationQueue
from shared.task_metadata import HUMAN_CURATOR_GATE_KEY

from fused_memory.config.schema import Mem0UpdateConfig
from fused_memory.models.reconciliation import StageId, StageReport
from fused_memory.reconciliation.consolidation_auto import build_auto_canonical
from fused_memory.reconciliation.consolidation_auto_executor import (
    CANONICAL_SLUG_LIMIT,
    auto_consolidation_initial_stats,
    run_auto_consolidation,
)
from fused_memory.reconciliation.consolidation_gate import GATE_METADATA_KEY
from fused_memory.reconciliation.consolidation_proposal import (
    RECORD_KIND_CONSOLIDATION_PROPOSAL,
    outcome_from_record,
)
from fused_memory.reconciliation.recon_ledger import ReconLedgerStore

TOPIC = 'auto-topic-alpha'
EXEC_RUN = 'run-exec'


@pytest_asyncio.fixture
async def ledger(tmp_path):
    store = ReconLedgerStore(tmp_path / 'reconciliation.db')
    await store.initialize()
    yield store
    await store.close()


@pytest.fixture
def queue(tmp_path):
    return EscalationQueue(tmp_path / 'esc')


@pytest.fixture
def svc(ledger):
    return FakeMemoryService(ledger)


def _report():
    return StageReport(stage=StageId.memory_consolidator, started_at=NOW, completed_at=NOW)


async def run(
    svc,
    *,
    queue=None,
    taskmaster=None,
    project_id=PROJECT_ID,
    run_id=EXEC_RUN,
    active_tasks=(),
    tmp_root='/tmp/project-root',
):
    report = _report()
    await run_auto_consolidation(
        svc,
        taskmaster if taskmaster is not None else FakeTaskBackend(),
        queue,
        project_id=project_id,
        project_root=tmp_root,
        run_id=run_id,
        config=svc.config.consolidation_auto,
        active_tasks=None if active_tasks is None else list(active_tasks),
        report=report,
        now=NOW,
    )
    return report


async def _row(ledger, topic=TOPIC, run_id='run-p1', project_id=PROJECT_ID):
    return await ledger.get_by_identity(
        project_id=project_id,
        record_kind=RECORD_KIND_CONSOLIDATION_PROPOSAL,
        task_id='',
        flag_type=topic,
        run_id=run_id,
    )


def _assert_stats_complete(report):
    assert set(auto_consolidation_initial_stats()) <= set(report.stats)


class TestFreshClusterPasses:
    """PRD B1."""

    @pytest.mark.asyncio
    async def test_members_are_stamped_and_one_canonical_is_minted(self, svc, ledger, queue):
        members = svc.seed_many(5)
        await propose(ledger, topic=TOPIC, member_ids=members)
        taskmaster = FakeTaskBackend()

        report = await run(svc, queue=queue, taskmaster=taskmaster)

        for member in members:
            assert svc.metadata_of(member)['topic'] == TOPIC
            assert svc.metadata_of(member)['x_auto_consolidated_run'] == EXEC_RUN
        (canonical_id,) = svc.canonical_ids()
        canonical = svc.records[canonical_id]
        assert canonical['content'] == build_auto_canonical(VALID_CLAIM, TOPIC, 5, EXEC_RUN)
        assert canonical['metadata']['topic'] == TOPIC
        assert canonical['metadata']['category'] == 'procedural_knowledge'
        assert canonical['metadata']['x_auto_consolidated'] == {
            'run_id': EXEC_RUN,
            'predicate_version': '1',
            'members_count': 5,
            'at': NOW.isoformat(),
        }
        assert taskmaster.calls == []
        assert queue.get_pending() == []
        svc.search.assert_not_awaited()
        assert report.stats['auto_consolidations'] == 1
        assert report.stats['auto_consolidation_pointers'] == [
            {'topic': TOPIC, 'canonical_id': canonical_id, 'run_id': EXEC_RUN},
        ]
        _assert_stats_complete(report)

    @pytest.mark.asyncio
    async def test_the_system_record_nests_its_facts(self, svc, ledger, queue):
        members = svc.seed_many(5)
        await propose(ledger, topic=TOPIC, member_ids=members)

        await run(svc, queue=queue)

        (record,) = svc.system_records
        meta = record['metadata']
        assert meta['record_kind'] == 'auto_consolidation'
        assert 'topic' not in meta
        assert 'run_id' not in meta
        (canonical_id,) = svc.canonical_ids()
        assert meta['x_auto_consolidation'] == {
            'topic': TOPIC,
            'canonical_id': canonical_id,
            'run_id': EXEC_RUN,
            'member_ids': members,
            'tagged_ids': members,
        }

    @pytest.mark.asyncio
    async def test_the_ledger_row_is_addressed_and_never_expires(self, svc, ledger, queue):
        members = svc.seed_many(5)
        await propose(ledger, topic=TOPIC, member_ids=members)

        await run(svc, queue=queue)

        row = await _row(ledger)
        assert row.state == 'addressed'
        assert row.expires_at is None
        outcome = outcome_from_record(row)
        (canonical_id,) = svc.canonical_ids()
        assert outcome.canonical_id == canonical_id
        assert outcome.canonical_minted is True
        assert outcome.tagged_ids == tuple(members)
        assert outcome.acted_run_id == EXEC_RUN


class TestFullyConsolidatedReEmission:
    """PRD B3."""

    @pytest.mark.asyncio
    async def test_a_second_identical_proposal_is_a_noop_with_zero_writes(
        self, svc, ledger, queue,
    ):
        members = svc.seed_many(5)
        await propose(ledger, topic=TOPIC, member_ids=members)
        await run(svc, queue=queue)
        await propose(ledger, topic=TOPIC, member_ids=members, run_id='run-p2')
        writes_before = svc.write_awaits()

        report = await run(svc, queue=queue, run_id='run-exec-2')

        assert svc.write_awaits() == writes_before
        row = await _row(ledger, run_id='run-p2')
        assert row.state == 'addressed'
        assert outcome_from_record(row).reason_codes == ('already_consolidated',)
        assert report.stats['auto_consolidations_noop'] == 1
        _assert_stats_complete(report)


class TestProjectScoping:
    """PRD B13."""

    @pytest.mark.asyncio
    async def test_a_project_not_enabled_is_skipped_with_no_reads(self, svc, ledger, queue):
        members = svc.seed_many(3)
        await propose(ledger, topic=TOPIC, member_ids=members, project_id='reify')

        report = await run(svc, queue=queue, project_id='reify')

        assert report.stats['auto_consolidations_skipped_project'] == 1
        svc.get_memory_by_id.assert_not_awaited()
        svc.count_memories_by_metadata.assert_not_awaited()
        svc.get_memories_by_metadata.assert_not_awaited()
        assert (await _row(ledger, project_id='reify')).state == 'proposed'
        _assert_stats_complete(report)


class TestSupervisedDryRun:
    """PRD B8, pass side."""

    @pytest.mark.asyncio
    async def test_disabled_records_dry_run_and_writes_nothing(self, ledger, queue):
        svc = FakeMemoryService(ledger, consolidation_auto=enabled_config(enabled=False))
        members = svc.seed_many(5)
        await propose(ledger, topic=TOPIC, member_ids=members)
        taskmaster = FakeTaskBackend()

        report = await run(svc, queue=queue, taskmaster=taskmaster)

        assert svc.write_awaits() == (0, 0, 0, 0)
        assert taskmaster.calls == []
        assert (await _row(ledger)).state == 'dry_run'
        assert report.stats['auto_consolidations_dry_run'] == 1
        assert report.stats['auto_consolidations'] == 0
        _assert_stats_complete(report)


class TestFailClosedInputs:
    async def _seed(self, svc, ledger):
        members = svc.seed_many(3)
        await propose(ledger, topic=TOPIC, member_ids=members)

    def _assert_nothing_done(self, svc, report, name):
        assert report.stats['auto_consolidation_skipped_unavailable'] == name
        svc.get_memory_by_id.assert_not_awaited()
        assert svc.write_awaits() == (0, 0, 0, 0)
        _assert_stats_complete(report)

    @pytest.mark.asyncio
    async def test_no_ledger(self, svc, ledger, queue):
        await self._seed(svc, ledger)
        svc.recon_ledger = None

        report = await run(svc, queue=queue)

        self._assert_nothing_done(svc, report, 'ledger')

    @pytest.mark.asyncio
    async def test_no_task_tree(self, svc, ledger, queue):
        await self._seed(svc, ledger)

        report = await run(svc, queue=queue, active_tasks=None)

        self._assert_nothing_done(svc, report, 'task_tree')
        assert (await _row(ledger)).state == 'proposed'

    @pytest.mark.asyncio
    async def test_canonical_slug_scroll_raising(self, svc, ledger, queue):
        await self._seed(svc, ledger)
        svc.scroll_error = TimeoutError('qdrant down')

        report = await run(svc, queue=queue)

        self._assert_nothing_done(svc, report, 'canonical_slugs')

    @pytest.mark.asyncio
    async def test_canonical_slug_scroll_returning_a_full_page(self, svc, ledger, queue):
        await self._seed(svc, ledger)
        for n in range(CANONICAL_SLUG_LIMIT):
            svc.seed(topic=f'filler-topic-{n}', canonical=True)

        report = await run(svc, queue=queue)

        self._assert_nothing_done(svc, report, 'canonical_slugs')


class TestUnreadableMember:
    @pytest.mark.asyncio
    async def test_a_member_read_error_writes_nothing(self, svc, ledger, queue):
        members = svc.seed_many(4)
        svc.read_errors[members[1]] = TimeoutError('read timed out')
        await propose(ledger, topic=TOPIC, member_ids=members)

        report = await run(svc, queue=queue)

        assert svc.write_awaits() == (0, 0, 0, 0)
        assert report.stats['auto_consolidations'] == 0
        _assert_stats_complete(report)


GATE_TOPIC_TASK_ID = 55


def _open_gate(topic=TOPIC, task_id=GATE_TOPIC_TASK_ID):
    return {
        'id': task_id,
        'status': 'blocked',
        'metadata': {'operational_mode': 'gate', GATE_METADATA_KEY: {'topic': topic}},
    }


def _seed_consolidated(svc, topic=TOPIC, n=5, **canonical_meta):
    canonical = svc.seed(topic=topic, canonical=True, **canonical_meta)
    stamped = svc.seed_many(n, topic=topic)
    return canonical, stamped


class TestRegrowthIsTaggedOnly:
    """PRD B2."""

    @pytest.mark.asyncio
    async def test_only_the_new_members_are_stamped(self, svc, ledger, queue):
        canonical, stamped = _seed_consolidated(svc)
        fresh = svc.seed_many(3)
        before = svc.snapshot(canonical)
        await propose(ledger, topic=TOPIC, member_ids=[*stamped, *fresh, canonical])

        report = await run(svc, queue=queue)

        patched = [c.kwargs['memory_id'] for c in svc.update_memory.await_args_list]
        assert patched == fresh
        for call in svc.update_memory.await_args_list:
            assert call.kwargs['metadata_patch'] == {
                'topic': TOPIC,
                'x_auto_consolidated_run': EXEC_RUN,
            }
        assert svc.snapshot(canonical) == before
        svc.add_memory.assert_not_awaited()
        svc.delete_memory.assert_not_awaited()
        outcome = outcome_from_record(await _row(ledger))
        assert outcome.tagged_ids == tuple(fresh)
        assert outcome.stripped_ids == (canonical,)
        assert outcome.canonical_id == canonical
        assert outcome.canonical_minted is False
        assert report.stats['auto_consolidations_tag_only'] == 1
        assert report.stats['auto_consolidations'] == 0


class TestHazardGate:
    """PRD B4."""

    async def _seed_hazard(self, svc, ledger):
        canonical = svc.seed(topic=TOPIC, canonical=True, superseded_by='x')
        members = svc.seed_many(3)
        member_ids = [*members, canonical]
        await propose(ledger, topic=TOPIC, member_ids=member_ids)
        return member_ids

    @pytest.mark.asyncio
    async def test_a_corrected_incumbent_files_one_curator_gate(self, svc, ledger, queue):
        member_ids = await self._seed_hazard(svc, ledger)
        taskmaster = FakeTaskBackend()

        report = await run(svc, queue=queue, taskmaster=taskmaster, tmp_root='/proj')

        assert svc.write_awaits() == (0, 0, 0, 0)
        (call,) = taskmaster.calls
        assert call['project_root'] == '/proj'
        assert isinstance(call['metadata'], str)
        meta = json.loads(call['metadata'])
        assert meta['task_kind'] == 'deterministic'
        assert meta['always_escalates'] is True
        assert meta['operational_mode'] == 'gate'
        assert meta[HUMAN_CURATOR_GATE_KEY] is True
        block = meta[GATE_METADATA_KEY]
        assert block['topic'] == TOPIC
        assert block['provenance']['observed_members'] == member_ids
        assert block['provenance']['authoritative'] is False
        assert 'canonical_carries_correction' in block['auto_refusal']['reason_codes']
        assert 'canonical_carries_correction' in call['description']
        row = await _row(ledger)
        assert row.state == 'gated'
        assert outcome_from_record(row).gate_task_id == '1001'
        assert report.stats['auto_consolidation_gates_filed'] == 1

    @pytest.mark.asyncio
    async def test_a_hazard_still_files_in_the_supervised_cycle(self, ledger, queue):
        svc = FakeMemoryService(ledger, consolidation_auto=enabled_config(enabled=False))
        await self._seed_hazard(svc, ledger)
        taskmaster = FakeTaskBackend()

        await run(svc, queue=queue, taskmaster=taskmaster)

        assert len(taskmaster.calls) == 1
        assert (await _row(ledger)).state == 'gated'

    @pytest.mark.asyncio
    async def test_a_duplicate_candidate_key_reuses_the_existing_gate(self, svc, ledger, queue):
        await self._seed_hazard(svc, ledger)

        await run(svc, queue=queue, taskmaster=FakeTaskBackend(duplicate_of=77))

        row = await _row(ledger)
        assert row.state == 'gated'
        assert outcome_from_record(row).gate_task_id == '77'

    @pytest.mark.asyncio
    async def test_a_zero_filing_budget_leaves_the_row_proposed(self, ledger, queue):
        svc = FakeMemoryService(
            ledger, consolidation_auto=enabled_config(max_gate_filings_per_cycle=0),
        )
        await self._seed_hazard(svc, ledger)
        taskmaster = FakeTaskBackend()

        report = await run(svc, queue=queue, taskmaster=taskmaster)

        assert taskmaster.calls == []
        assert (await _row(ledger)).state == 'proposed'
        assert report.stats['auto_consolidation_unfiled'] == 1

    @pytest.mark.asyncio
    async def test_no_task_backend_counts_as_unfiled(self, svc, ledger, queue):
        await self._seed_hazard(svc, ledger)
        report = _report()

        await run_auto_consolidation(
            svc,
            None,
            queue,
            project_id=PROJECT_ID,
            project_root='/proj',
            run_id=EXEC_RUN,
            config=svc.config.consolidation_auto,
            active_tasks=[],
            report=report,
            now=NOW,
        )

        assert (await _row(ledger)).state == 'proposed'
        assert report.stats['auto_consolidation_unfiled'] == 1


class TestAlreadyGated:
    """PRD B5."""

    @pytest.mark.asyncio
    async def test_an_open_gate_makes_it_a_noop_naming_the_gate(self, svc, ledger, queue):
        canonical = svc.seed(topic=TOPIC, canonical=True, superseded_by='x')
        members = svc.seed_many(3)
        await propose(ledger, topic=TOPIC, member_ids=[*members, canonical])
        taskmaster = FakeTaskBackend()

        await run(svc, queue=queue, taskmaster=taskmaster, active_tasks=[_open_gate()])

        assert taskmaster.calls == []
        assert svc.write_awaits() == (0, 0, 0, 0)
        row = await _row(ledger)
        assert row.state == 'addressed'
        outcome = outcome_from_record(row)
        assert outcome.reason_codes == ('already_gated',)
        assert outcome.gate_task_id == str(GATE_TOPIC_TASK_ID)


class TestExecutionBudget:
    @pytest.mark.asyncio
    async def test_one_execution_per_cycle_leaves_the_second_proposed(self, ledger, queue):
        svc = FakeMemoryService(ledger, consolidation_auto=enabled_config(max_auto_per_cycle=1))
        await propose(ledger, topic='auto-topic-one', member_ids=svc.seed_many(5))
        await propose(ledger, topic='auto-topic-two', member_ids=svc.seed_many(4))

        report = await run(svc, queue=queue)

        states = {
            topic: (await _row(ledger, topic=topic)).state
            for topic in ('auto-topic-one', 'auto-topic-two')
        }
        assert states == {'auto-topic-one': 'addressed', 'auto-topic-two': 'proposed'}
        assert report.stats['auto_consolidations'] == 1
        assert report.stats['auto_consolidation_unexecuted'] == 1


class TestClosureFailure:
    @pytest.mark.asyncio
    async def test_a_member_that_could_not_be_tagged_files_a_gate_past_the_budget(
        self, ledger, queue,
    ):
        svc = FakeMemoryService(
            ledger, consolidation_auto=enabled_config(max_gate_filings_per_cycle=0),
        )
        members = svc.seed_many(4)
        svc.update_returns[members[1]] = {'error': 'gone', 'error_type': 'MemoryNotFound'}
        await propose(ledger, topic=TOPIC, member_ids=members)
        taskmaster = FakeTaskBackend()

        report = await run(svc, queue=queue, taskmaster=taskmaster)

        (call,) = taskmaster.calls
        reason_codes = json.loads(call['metadata'])[GATE_METADATA_KEY]['auto_refusal'][
            'reason_codes'
        ]
        assert reason_codes['unstamped_cluster_member'] == [members[1]]
        assert len(svc.system_records) == 1
        row = await _row(ledger)
        assert row.state == 'gated'
        assert row.expires_at is None
        outcome = outcome_from_record(row)
        (canonical_id,) = svc.canonical_ids()
        assert outcome.canonical_id == canonical_id
        assert outcome.tagged_ids == (members[0], members[2], members[3])
        assert outcome.gate_task_id == '1001'
        assert report.stats['auto_consolidation_closure_failures'] == 1
        assert report.stats['auto_consolidation_gates_filed'] == 1


class TestExecutionRefusal:
    @pytest.mark.asyncio
    async def test_an_unauthorized_executor_writes_nothing_and_is_refused(self, ledger, queue):
        svc = FakeMemoryService(
            ledger,
            mem0_update=Mem0UpdateConfig(metadata_patch_allowed_agent_prefixes=['curator-']),
        )
        await propose(ledger, topic=TOPIC, member_ids=svc.seed_many(4))

        report = await run(svc, queue=queue)

        assert svc.write_awaits() == (0, 0, 0, 0)
        row = await _row(ledger)
        assert row.state == 'refused'
        assert outcome_from_record(row).reason_codes == ('execution_refused',)
        assert report.stats['auto_consolidation_refusals'] == {'execution_refused': 1}
