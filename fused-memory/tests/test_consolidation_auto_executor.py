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
from datetime import timedelta

import pytest
import pytest_asyncio
from _auto_consolidation_fakes import (
    NOW,
    PROJECT_ID,
    PROPOSED_AT,
    VALID_CLAIM,
    FakeMemoryService,
    FakeTaskBackend,
    enabled_config,
    new_id,
    propose,
)
from escalation.queue import EscalationQueue
from shared.task_metadata import HUMAN_CURATOR_GATE_KEY

from fused_memory.config.schema import Mem0UpdateConfig
from fused_memory.models.reconciliation import StageId, StageReport
from fused_memory.reconciliation.consolidation_auto import build_auto_canonical
from fused_memory.reconciliation.consolidation_auto_executor import (
    BACKLOG_CATEGORY,
    CANONICAL_SLUG_LIMIT,
    REFUSED_CATEGORY,
    RevertNotAuthorized,
    auto_consolidation_initial_stats,
    revert_auto_consolidation,
    run_auto_consolidation,
)
from fused_memory.reconciliation.consolidation_gate import GATE_METADATA_KEY
from fused_memory.reconciliation.consolidation_proposal import (
    RECORD_KIND_CONSOLIDATION_PROPOSAL,
    ProposalOutcome,
    ProposalState,
    outcome_from_record,
    record_outcome,
)
from fused_memory.reconciliation.recon_ledger import ReconLedgerRecord, ReconLedgerStore

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


def _outcome(row) -> ProposalOutcome:
    outcome = outcome_from_record(row)
    assert outcome is not None, 'the executor never recorded an outcome on this row'
    return outcome


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
        outcome = _outcome(row)
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
        assert _outcome(row).reason_codes == ('already_consolidated',)
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
        outcome = _outcome(await _row(ledger))
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
        assert _outcome(row).gate_task_id == '1001'
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
        assert _outcome(row).gate_task_id == '77'

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
        outcome = _outcome(row)
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
        outcome = _outcome(row)
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
        assert _outcome(row).reason_codes == ('execution_refused',)
        assert report.stats['auto_consolidation_refusals'] == {'execution_refused': 1}


def _pending(queue, category=REFUSED_CATEGORY):
    return [e for e in queue.get_pending() if e.category == category]


def _new_ids(n):
    return [new_id() for _ in range(n)]


class TestBenignRefusal:
    """PRD B7."""

    @pytest.mark.asyncio
    async def test_an_aged_oversized_row_is_refused_with_one_info_record(
        self, svc, ledger, queue,
    ):
        await propose(ledger, topic=TOPIC, member_ids=_new_ids(21))

        report = await run(svc, queue=queue)

        (esc,) = _pending(queue)
        assert esc.level == 1
        assert esc.severity == 'info'
        assert esc.summary == f'{TOPIC}: member_count_out_of_range'
        assert esc.project_id == PROJECT_ID
        assert 'run-p1' in esc.detail
        assert 'member_count_out_of_range' in esc.detail
        row = await _row(ledger)
        assert row.state == 'refused'
        assert _outcome(row).reason_codes == ('member_count_out_of_range',)
        assert report.stats['auto_consolidation_refusals'] == {'member_count_out_of_range': 1}
        svc.get_memory_by_id.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_second_refusal_folds_into_the_pending_record(self, svc, ledger, queue):
        await propose(ledger, topic=TOPIC, member_ids=_new_ids(21))
        await run(svc, queue=queue)
        await propose(ledger, topic=TOPIC, member_ids=_new_ids(21), run_id='run-p2')

        await run(svc, queue=queue, run_id='run-exec-2')

        (esc,) = _pending(queue)
        assert esc.dedupe_count == 1

    @pytest.mark.asyncio
    async def test_the_streak_escape_files_a_separate_blocking_record(self, ledger, queue):
        svc = FakeMemoryService(
            ledger, consolidation_auto=enabled_config(refusal_streak_threshold=2),
        )
        for n in range(3):
            await propose(ledger, topic=TOPIC, member_ids=_new_ids(21), run_id=f'run-p{n}')
            await run(svc, queue=queue, run_id=f'run-exec-{n}')

        records = _pending(queue)
        assert sorted(e.severity for e in records) == ['blocking', 'info']

    @pytest.mark.asyncio
    async def test_an_addressed_row_between_refusals_resets_the_streak(self, ledger, queue):
        svc = FakeMemoryService(
            ledger, consolidation_auto=enabled_config(refusal_streak_threshold=1),
        )
        await propose(ledger, topic=TOPIC, member_ids=_new_ids(21), run_id='run-p0')
        await run(svc, queue=queue, run_id='run-exec-0')
        await propose(ledger, topic=TOPIC, member_ids=_new_ids(3), run_id='run-p1')
        addressed = await _row(ledger, run_id='run-p1')
        await record_outcome(
            ledger,
            addressed,
            ProposalOutcome(
                state=ProposalState.ADDRESSED, acted_run_id='x', acted_at=NOW.isoformat(),
            ),
        )
        await propose(ledger, topic=TOPIC, member_ids=_new_ids(21), run_id='run-p2')

        await run(svc, queue=queue, run_id='run-exec-2')

        assert [e.severity for e in _pending(queue)] == ['info']

    @pytest.mark.asyncio
    async def test_an_undecodable_row_is_refused(self, svc, ledger, queue):
        await ledger.upsert(
            ReconLedgerRecord(
                project_id=PROJECT_ID,
                record_kind=RECORD_KIND_CONSOLIDATION_PROPOSAL,
                flag_type=TOPIC,
                run_id='run-p1',
                payload_json='not json',
                state='proposed',
                created_at=NOW.isoformat(),
                expires_at='2099-01-01T00:00:00+00:00',
            ),
        )

        report = await run(svc, queue=queue)

        assert (await _row(ledger)).state == 'refused'
        assert report.stats['auto_consolidation_refusals'] == {'proposal_undecodable': 1}
        (esc,) = _pending(queue)
        assert esc.summary == f'{TOPIC}: proposal_undecodable'

    @pytest.mark.asyncio
    async def test_a_category_without_flood_weight_is_refused(self, svc, ledger, queue):
        await propose(
            ledger, topic=TOPIC, member_ids=svc.seed_many(3), category='temporal_facts',
        )

        report = await run(svc, queue=queue)

        assert (await _row(ledger)).state == 'refused'
        assert report.stats['auto_consolidation_refusals'] == {
            'category_not_consolidatable': 1,
        }
        svc.get_memory_by_id.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_execution_refused_files_its_info_record(self, ledger, queue):
        svc = FakeMemoryService(
            ledger,
            mem0_update=Mem0UpdateConfig(metadata_patch_allowed_agent_prefixes=['curator-']),
        )
        await propose(ledger, topic=TOPIC, member_ids=svc.seed_many(4))

        await run(svc, queue=queue)

        (esc,) = _pending(queue)
        assert esc.severity == 'info'
        assert esc.summary == f'{TOPIC}: execution_refused'

    @pytest.mark.asyncio
    async def test_no_queue_still_records_the_refusal(self, svc, ledger):
        await propose(ledger, topic=TOPIC, member_ids=_new_ids(21))

        report = await run(svc, queue=None)

        assert (await _row(ledger)).state == 'refused'
        assert report.stats['auto_consolidation_refusals'] == {'member_count_out_of_range': 1}


class TestBacklog:
    """PRD B9."""

    def _svc(self, ledger):
        return FakeMemoryService(
            ledger,
            consolidation_auto=enabled_config(
                max_auto_per_cycle=1, max_gate_filings_per_cycle=0, backlog_multiplier=2,
            ),
        )

    @pytest.mark.asyncio
    async def test_a_tripped_backlog_executes_nothing_and_escalates_once(self, ledger, queue):
        svc = self._svc(ledger)
        topics = ['auto-topic-one', 'auto-topic-two', 'auto-topic-three']
        for topic in topics:
            await propose(ledger, topic=topic, member_ids=svc.seed_many(3))
        taskmaster = FakeTaskBackend()

        report = await run(svc, queue=queue, taskmaster=taskmaster)

        svc.get_memory_by_id.assert_not_awaited()
        svc.count_memories_by_metadata.assert_not_awaited()
        assert taskmaster.calls == []
        for topic in topics:
            assert (await _row(ledger, topic=topic)).state == 'proposed'
        (esc,) = _pending(queue, BACKLOG_CATEGORY)
        assert esc.severity == 'info'
        assert report.stats['auto_consolidation_backlog_tripped'] is True
        assert report.stats['auto_consolidation_unexecuted'] == 3
        assert report.stats['auto_consolidation_applied_cap'] == 1

    @pytest.mark.asyncio
    async def test_a_second_tripped_cycle_folds(self, ledger, queue):
        svc = self._svc(ledger)
        for topic in ['auto-topic-one', 'auto-topic-two', 'auto-topic-three']:
            await propose(ledger, topic=topic, member_ids=svc.seed_many(3))

        await run(svc, queue=queue)
        await run(svc, queue=queue, run_id='run-exec-2')

        (esc,) = _pending(queue, BACKLOG_CATEGORY)
        assert esc.dedupe_count == 1


class TestRowSelection:
    @pytest.mark.asyncio
    async def test_an_older_row_for_the_same_topic_is_superseded(self, svc, ledger, queue):
        await propose(
            ledger,
            topic=TOPIC,
            member_ids=svc.seed_many(3),
            run_id='run-r1',
            now=PROPOSED_AT - timedelta(hours=1),
        )
        await propose(ledger, topic=TOPIC, member_ids=svc.seed_many(3), run_id='run-r2')

        report = await run(svc, queue=queue)

        assert (await _row(ledger, run_id='run-r1')).state == 'superseded'
        assert (await _row(ledger, run_id='run-r2')).state == 'addressed'
        assert report.stats['auto_consolidation_proposals_superseded'] == 1

    @pytest.mark.asyncio
    async def test_an_expired_row_is_neither_evaluated_nor_rewritten(self, svc, ledger, queue):
        members = svc.seed_many(3)
        await propose(ledger, topic=TOPIC, member_ids=members, ttl_hours=1)
        before = await _row(ledger)

        report = await run(svc, queue=queue)

        assert await _row(ledger) == before
        svc.get_memory_by_id.assert_not_awaited()
        assert report.stats['auto_consolidation_proposals_expired'] == 1


class TestVocabulary:
    def test_every_filed_category_is_registered(self):
        server = pytest.importorskip('escalation.server')
        assert REFUSED_CATEGORY in server.CATEGORIES
        assert BACKLOG_CATEGORY in server.CATEGORIES


REVERT_AGENT = 'recon-stage-memory_consolidator'


class TestReversal:
    """PRD B10."""

    @pytest.mark.asyncio
    async def test_a_minted_run_reverts_to_no_orphan(self, svc, ledger, queue):
        members = svc.seed_many(5)
        await propose(ledger, topic=TOPIC, member_ids=members)
        await run(svc, queue=queue)
        (canonical_id,) = svc.canonical_ids()
        row = await _row(ledger)

        result = await revert_auto_consolidation(svc, row, agent_id=REVERT_AGENT)

        assert result.reverted is True
        assert result.canonical_deleted == canonical_id
        assert result.untagged == tuple(members)
        assert result.failures == ()
        svc.delete_memory.assert_awaited_once()
        assert svc.delete_memory.await_args.args == (canonical_id, 'mem0')
        for member in members:
            assert 'topic' not in svc.metadata_of(member)
            assert 'x_auto_consolidated_run' not in svc.metadata_of(member)
        untag_calls = [
            c for c in svc.update_memory.await_args_list
            if c.kwargs.get('metadata_delete_keys')
        ]
        assert {c.kwargs['memory_id'] for c in untag_calls} == set(members)
        for call in untag_calls:
            assert call.kwargs['metadata_delete_keys'] == ['topic', 'x_auto_consolidated_run']
        assert await svc.get_memories_by_metadata(PROJECT_ID, {'topic': TOPIC}) == []
        assert (await _row(ledger)).state == 'reverted'

    @pytest.mark.asyncio
    async def test_a_tag_only_run_untags_only_its_own_members(self, svc, ledger, queue):
        canonical, stamped = _seed_consolidated(svc)
        fresh = svc.seed_many(3)
        await propose(ledger, topic=TOPIC, member_ids=[*stamped, *fresh, canonical])
        await run(svc, queue=queue)
        before = svc.snapshot(canonical)

        result = await revert_auto_consolidation(svc, await _row(ledger), agent_id=REVERT_AGENT)

        assert result.reverted is True
        assert result.canonical_deleted is None
        assert result.untagged == tuple(fresh)
        for member in fresh:
            assert 'topic' not in svc.metadata_of(member)
        for member in stamped:
            assert svc.metadata_of(member)['topic'] == TOPIC
        assert svc.snapshot(canonical) == before
        svc.delete_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_members_are_untagged_before_a_failing_delete(self, svc, ledger, queue):
        members = svc.seed_many(3)
        await propose(ledger, topic=TOPIC, member_ids=members)
        await run(svc, queue=queue)
        (canonical_id,) = svc.canonical_ids()
        svc.delete_errors[canonical_id] = TimeoutError('delete timed out')

        result = await revert_auto_consolidation(svc, await _row(ledger), agent_id=REVERT_AGENT)

        for member in members:
            assert 'topic' not in svc.metadata_of(member)
        assert result.reverted is False
        assert [f['id'] for f in result.failures] == [canonical_id]
        assert canonical_id in svc.records
        assert (await _row(ledger)).state == 'addressed'

    @pytest.mark.parametrize('state', ['refused', 'dry_run', 'proposed'])
    @pytest.mark.asyncio
    async def test_a_row_with_no_execution_cannot_be_reverted(self, svc, ledger, state):
        await propose(ledger, topic=TOPIC, member_ids=_new_ids(3))
        row = await _row(ledger)
        if state != 'proposed':
            await record_outcome(
                ledger,
                row,
                ProposalOutcome(
                    state=ProposalState(state), acted_run_id='x', acted_at=NOW.isoformat(),
                ),
            )
            row = await _row(ledger)

        with pytest.raises(ValueError, match=state):
            await revert_auto_consolidation(svc, row, agent_id=REVERT_AGENT)

    @pytest.mark.asyncio
    async def test_an_unauthorized_agent_is_refused_before_any_write(self, svc, ledger, queue):
        await propose(ledger, topic=TOPIC, member_ids=svc.seed_many(3))
        await run(svc, queue=queue)
        writes_before = svc.write_awaits()

        with pytest.raises(RevertNotAuthorized):
            await revert_auto_consolidation(svc, await _row(ledger), agent_id='someone-else')

        assert svc.write_awaits() == writes_before
