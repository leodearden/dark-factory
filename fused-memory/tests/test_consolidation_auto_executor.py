"""The auto-consolidation executor post-step (task 5240, PRD C4).

``reconciliation/consolidation_auto_executor.py::run_auto_consolidation`` runs
once per full Stage-1 cycle: it reads the proposal rows, ranks them under the
shared flood policy, reads each selected cluster deterministically, rules with
the pure predicate and executes, gates or records the verdict. The stores are
the stateful fakes in ``tests/_auto_consolidation_fakes.py``; the ledger and
the escalation queue are real.
"""

from __future__ import annotations

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
from fused_memory.reconciliation.consolidation_auto_executor import (
    CANONICAL_SLUG_LIMIT,
    auto_consolidation_initial_stats,
    run_auto_consolidation,
)

from fused_memory.models.reconciliation import StageId, StageReport
from fused_memory.reconciliation.consolidation_auto import build_auto_canonical
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
