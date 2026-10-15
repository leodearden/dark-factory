"""A stage1_flag_marker row names its subject task unambiguously (task 5614).

The ledger keys a marker row on (project_id = the RUNNING project, kind,
task_id, flag_type, '').  A task id cited from ANOTHER project must not share a
row with the same-numbered task of the running project, or recurrence,
cross-project suppression counters, acknowledgement and terminal-task GC all
conflate two different subjects.

Every test drives public seams only: ``dedup_flags``,
``compute_marker_signature`` and the ledger's own read API.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from fused_memory.models.memory import AddMemoryResponse
from fused_memory.models.reconciliation import StageId, StageReport, Watermark
from fused_memory.reconciliation.flag_dedup import (
    compute_content_fingerprint_signature,
    dedup_flags,
)
from fused_memory.reconciliation.recon_ledger import ReconLedgerRecord, ReconLedgerStore
from fused_memory.reconciliation.stages.base import BaseStage
from reconciliation.consolidator_fixtures import make_consolidator

RUNNING = 'dark_factory'
FT = 'some_recurring_gap'
MARKER = 'stage1_flag_marker'

LOCAL = {'task_id': '650', 'flag_type': FT, 'description': 'local subject'}
FOREIGN = {
    'task_id': None,
    'flag_type': FT,
    'description': 'foreign subject',
    'cited_tasks': [{'project_id': 'reify', 'task_id': '650', 'title': 't'}],
}
LOCAL_CITING_ITSELF = {
    'task_id': '650',
    'flag_type': FT,
    'cited_tasks': [{'project_id': RUNNING, 'task_id': '650'}],
}
MIXED_UNION = {
    'task_id': None,
    'flag_type': FT,
    'cited_tasks': [
        {'project_id': RUNNING, 'task_id': '12'},
        {'project_id': 'reify', 'task_id': '650'},
        {'project_id': 'autopilot_video', 'task_id': '655'},
    ],
}


@pytest_asyncio.fixture
async def memory_service(tmp_path):
    ledger = ReconLedgerStore(tmp_path / 'reconciliation.db')
    await ledger.initialize()
    service = AsyncMock()
    service.recon_ledger = ledger
    service.add_memory = AsyncMock(return_value=AddMemoryResponse(memory_ids=['mirror-id']))
    try:
        yield service
    finally:
        await ledger.close()


async def _seed_marker(
    ledger: ReconLedgerStore,
    task_id: str,
    *,
    project_id: str = RUNNING,
    flag_type: str = FT,
    run_id: str = 'legacy',
) -> None:
    await ledger.upsert(ReconLedgerRecord(
        project_id=project_id,
        record_kind=MARKER,
        payload_json=json.dumps({
            'source': MARKER,
            'kind': MARKER,
            'task_id': task_id,
            'flag_type': flag_type,
            'run_id': run_id,
            'last_seen_run_id': run_id,
        }),
        state='active',
        created_at='2026-01-01T00:00:00+00:00',
        task_id=task_id,
        flag_type=flag_type,
        run_id='',
        expires_at='2099-01-01T00:00:00+00:00',
    ))


def _by_description(flags: list[dict], description: str) -> dict:
    (match,) = [f for f in flags if f.get('description') == description]
    return match


@pytest.mark.asyncio
async def test_local_and_foreign_same_number_get_distinct_rows(memory_service):
    out = await dedup_flags(memory_service, RUNNING, 'r1', [LOCAL, FOREIGN])

    assert len(await memory_service.recon_ledger.marker_task_ids(RUNNING)) == 2
    assert 'persisted_from_run' not in _by_description(out, 'foreign subject')


@pytest.mark.asyncio
async def test_foreign_cycle_does_not_make_local_subject_recur(memory_service):
    await dedup_flags(memory_service, RUNNING, 'r1', [FOREIGN])
    (local,) = await dedup_flags(memory_service, RUNNING, 'r2', [LOCAL])

    assert 'persisted_from_run' not in local


@pytest.mark.asyncio
async def test_same_foreign_subject_still_recurs(memory_service):
    await dedup_flags(memory_service, RUNNING, 'r1', [FOREIGN])
    (foreign,) = await dedup_flags(memory_service, RUNNING, 'r2', [FOREIGN])

    assert foreign['persisted_from_run'] == 'r1'


@pytest.mark.asyncio
async def test_legacy_all_local_row_is_still_read(memory_service):
    await _seed_marker(memory_service.recon_ledger, '650')

    (flag,) = await dedup_flags(memory_service, RUNNING, 'r1', [LOCAL_CITING_ITSELF])

    assert flag['persisted_from_run'] == 'legacy'


@pytest.mark.asyncio
async def test_legacy_bare_row_for_foreign_subject_misses_once_then_self_heals(memory_service):
    """The bare legacy row cannot be told apart from a local subject's row.

    No offline migration can recover a bare component's project: since task
    4712 the marker payload holds only task_id/flag_type/run_id.  A fallback
    read of the bare key would re-import exactly the conflated history this
    change removes.  So the foreign subject misses once, re-keys, and recurs
    from the next cycle on.
    """
    await _seed_marker(memory_service.recon_ledger, '650')

    (first,) = await dedup_flags(memory_service, RUNNING, 'r1', [FOREIGN])
    (second,) = await dedup_flags(memory_service, RUNNING, 'r2', [FOREIGN])

    assert 'persisted_from_run' not in first
    assert second['persisted_from_run'] == 'r1'


@pytest.mark.asyncio
async def test_local_task_going_terminal_reaps_only_the_local_row(memory_service):
    ledger = memory_service.recon_ledger
    await dedup_flags(memory_service, RUNNING, 'r1', [LOCAL, FOREIGN])

    await ledger.gc(RUNNING, datetime.now(UTC).isoformat(), ['650'])

    assert await ledger.get_by_identity(RUNNING, MARKER, '650', FT, '') is None
    assert len(await ledger.marker_task_ids(RUNNING)) == 1


def test_foreign_citation_is_project_qualified():
    from fused_memory.reconciliation.flag_dedup import compute_marker_signature

    assert compute_marker_signature(FOREIGN, project_id=RUNNING) == ('reify:650', FT)


def test_local_citation_collapses_to_the_bare_top_level_id():
    from fused_memory.reconciliation.flag_dedup import compute_marker_signature

    assert compute_marker_signature(LOCAL_CITING_ITSELF, project_id=RUNNING) == ('650', FT)


def test_project_less_citations_stay_bare():
    from fused_memory.reconciliation.flag_dedup import compute_marker_signature

    flag = {
        'task_id': None,
        'flag_type': FT,
        'cited_tasks': [{'task_id': '12'}, {'task_id': '15'}],
    }

    assert compute_marker_signature(flag, project_id=RUNNING) == ('12,15', FT)


def test_mixed_union_qualifies_only_foreign_components():
    from fused_memory.reconciliation.flag_dedup import compute_marker_signature

    sig = compute_marker_signature(MIXED_UNION, project_id=RUNNING)

    assert sig is not None
    assert set(sig[0].split(',')) == {'12', 'reify:650', 'autopilot_video:655'}
    assert sig[1] == FT


@pytest.mark.asyncio
async def test_mixed_union_key_round_trips_through_the_ledger(memory_service):
    from fused_memory.reconciliation.flag_dedup import compute_marker_signature

    sig = compute_marker_signature(MIXED_UNION, project_id=RUNNING)
    assert sig is not None

    await dedup_flags(memory_service, RUNNING, 'r1', [MIXED_UNION])

    assert await memory_service.recon_ledger.get_by_identity(RUNNING, MARKER, *sig, '') is not None


@pytest.mark.asyncio
async def test_acknowledging_the_foreign_subject_addresses_only_its_row(memory_service):
    from fused_memory.reconciliation.flag_dedup import (
        acknowledge_resolved_flags,
        compute_marker_signature,
    )

    ledger = memory_service.recon_ledger
    await dedup_flags(memory_service, RUNNING, 'r1', [LOCAL, FOREIGN])
    foreign_sig = compute_marker_signature(FOREIGN, project_id=RUNNING)
    assert foreign_sig is not None

    acknowledged = await acknowledge_resolved_flags(memory_service, RUNNING, 'r2', [FOREIGN])

    assert acknowledged == 1
    foreign_row = await ledger.get_by_identity(RUNNING, MARKER, *foreign_sig, '')
    local_row = await ledger.get_by_identity(RUNNING, MARKER, '650', FT, '')
    assert foreign_row is not None and foreign_row.state == 'addressed'
    assert local_row is not None and local_row.state == 'active'


def test_anchorless_flag_falls_back_to_content_fingerprint():
    from fused_memory.reconciliation.flag_dedup import compute_marker_signature

    flag = {'task_id': None, 'flag_type': FT, 'description': 'no anchor at all'}

    assert compute_marker_signature(flag, project_id=RUNNING) == (
        compute_content_fingerprint_signature(flag)
    )


class TestConsolidatorAcknowledgesByMarkerIdentity:
    """MemoryConsolidator.run() tells a moot drop from a suppression by
    signature diff; that diff must compare marker identities, or a foreign
    subject's suppression shields a same-numbered local moot flag."""

    PROJECT_ID = 'test_project'
    PROJECT_ROOT = '/tmp/test_project'
    FOREIGN_ROOT = '/reify'
    ST = 'stale_metadata'

    SUPPRESSED = {
        'task_id': None,
        'flag_type': ST,
        'description': 'd1',
        'cited_tasks': [{'project_id': 'reify', 'task_id': '777', 'title': 'Fix'}],
    }
    MOOT = {'task_id': '777', 'flag_type': ST, 'description': 'd2'}

    async def _get_task(self, task_id, project_root):
        if str(task_id) == '777' and project_root == self.FOREIGN_ROOT:
            return {'id': 777, 'title': 'Fix', 'status': 'pending'}
        if str(task_id) == '777' and project_root == self.PROJECT_ROOT:
            return {'id': 777, 'status': 'done', 'title': 'x'}
        return {'error': f'No tasks found for ID(s): {task_id}', 'error_type': 'TaskmasterError'}

    @pytest.mark.asyncio
    async def test_moot_local_drop_is_reclaimed_while_foreign_suppression_is_kept(
        self, memory_service
    ):
        from fused_memory.reconciliation.flag_dedup import compute_marker_signature

        ledger = memory_service.recon_ledger
        suppressed_sig = compute_marker_signature(self.SUPPRESSED, project_id=self.PROJECT_ID)
        assert suppressed_sig is not None
        await _seed_marker(ledger, suppressed_sig[0], project_id=self.PROJECT_ID, flag_type=self.ST)
        await _seed_marker(ledger, '777', project_id=self.PROJECT_ID, flag_type=self.ST)

        stage = make_consolidator(project_root=self.PROJECT_ROOT)
        assert stage.memory is not None and stage.taskmaster is not None
        stage.memory.recon_ledger = ledger
        stage.known_projects = {'reify': self.FOREIGN_ROOT}
        stage.taskmaster.get_task = AsyncMock(side_effect=self._get_task)
        base_report = StageReport(
            stage=StageId.memory_consolidator,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items_flagged=[dict(self.SUPPRESSED), dict(self.MOOT)],
            stats={},
        )

        with patch.object(BaseStage, 'run', new=AsyncMock(return_value=base_report)):
            report = await stage.run(
                events=[],
                watermark=Watermark(project_id=self.PROJECT_ID),
                prior_reports=[],
                run_id='r2',
            )

        assert not [f for f in report.items_flagged if f.get('description') in ('d1', 'd2')]
        moot_row = await ledger.get_by_identity(self.PROJECT_ID, MARKER, '777', self.ST, '')
        suppressed_row = await ledger.get_by_identity(
            self.PROJECT_ID, MARKER, *suppressed_sig, '',
        )
        assert moot_row is not None and moot_row.state == 'addressed'
        assert suppressed_row is not None and suppressed_row.state == 'active'
