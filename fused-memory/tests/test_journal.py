"""Tests for reconciliation journal (SQLite persistence)."""

import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
import pytest_asyncio

from fused_memory.models.reconciliation import (
    JournalEntry,
    JudgeVerdict,
    ReconciliationRun,
    StageId,
    StageReport,
    Watermark,
)
from fused_memory.reconciliation.journal import ReconciliationJournal


@pytest_asyncio.fixture
async def journal(tmp_path):
    j = ReconciliationJournal(tmp_path / 'test_recon')
    await j.initialize()
    yield j
    await j.close()


@pytest.mark.asyncio
async def test_watermark_default(journal):
    wm = await journal.get_watermark('test-project')
    assert wm.project_id == 'test-project'
    assert wm.last_full_run_id is None
    assert wm.last_full_run_completed is None


@pytest.mark.asyncio
async def test_watermark_roundtrip(journal):
    now = datetime.now(timezone.utc)
    wm = Watermark(
        project_id='test-project',
        last_full_run_id='run-123',
        last_full_run_completed=now,
        last_episode_timestamp=now,
    )
    await journal.update_watermark(wm)

    loaded = await journal.get_watermark('test-project')
    assert loaded.last_full_run_id == 'run-123'
    assert loaded.last_full_run_completed is not None


@pytest.mark.asyncio
async def test_run_lifecycle(journal):
    run_id = str(uuid.uuid4())
    run = ReconciliationRun(
        id=run_id,
        project_id='test-project',
        run_type='full',
        trigger_reason='buffer_size:10',
        started_at=datetime.now(timezone.utc),
        events_processed=10,
        status='running',
    )
    await journal.start_run(run)

    # Is active
    assert await journal.is_run_active('test-project')

    # Complete
    await journal.complete_run(run_id, 'completed')

    # Load
    loaded = await journal.get_run(run_id)
    assert loaded is not None
    assert loaded.status == 'completed'
    assert loaded.completed_at is not None

    # No longer active
    assert not await journal.is_run_active('test-project')


@pytest.mark.asyncio
async def test_journal_entries(journal):
    run_id = str(uuid.uuid4())
    entry = JournalEntry(
        id=str(uuid.uuid4()),
        run_id=run_id,
        stage=StageId.memory_consolidator,
        timestamp=datetime.now(timezone.utc),
        operation='delete_memory',
        target_system='mem0',
        before_state={'id': '123', 'content': 'old'},
        after_state=None,
        reasoning='Duplicate found',
        evidence=[{'query': 'test', 'match': True}],
    )
    await journal.add_entry(entry)

    entries = await journal.get_entries(run_id)
    assert len(entries) == 1
    assert entries[0].operation == 'delete_memory'
    assert entries[0].before_state == {'id': '123', 'content': 'old'}
    assert entries[0].evidence == [{'query': 'test', 'match': True}]


@pytest.mark.asyncio
async def test_judge_verdict(journal):
    # Need a run first
    run_id = str(uuid.uuid4())
    run = ReconciliationRun(
        id=run_id,
        project_id='test-project',
        run_type='full',
        trigger_reason='test',
        started_at=datetime.now(timezone.utc),
        status='completed',
    )
    await journal.start_run(run)

    verdict = JudgeVerdict(
        run_id=run_id,
        reviewed_at=datetime.now(timezone.utc),
        severity='ok',
        findings=[],
        action_taken='none',
    )
    await journal.add_verdict(verdict)

    verdicts = await journal.get_recent_verdicts('test-project', limit=5)
    assert len(verdicts) == 1
    assert verdicts[0].severity == 'ok'


@pytest.mark.asyncio
async def test_recent_runs(journal):
    for i in range(3):
        run = ReconciliationRun(
            id=str(uuid.uuid4()),
            project_id='test-project',
            run_type='full',
            trigger_reason=f'test_{i}',
            started_at=datetime.now(timezone.utc),
            status='running',
        )
        await journal.start_run(run)

    runs = await journal.get_recent_runs('test-project', limit=10)
    assert len(runs) == 3


@pytest.mark.asyncio
async def test_stats(journal):
    run_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    run = ReconciliationRun(
        id=run_id,
        project_id='test-project',
        run_type='full',
        trigger_reason='test',
        started_at=now,
        status='running',
    )
    await journal.start_run(run)
    await journal.complete_run(run_id, 'completed')

    from datetime import timedelta
    since = now - timedelta(hours=1)
    stats = await journal.get_stats('test-project', since)
    assert stats['runs_count'] == 1


@pytest.mark.asyncio
async def test_get_nonexistent_run(journal):
    result = await journal.get_run('nonexistent-id')
    assert result is None


@pytest.mark.asyncio
async def test_stage_reports_roundtrip(journal):
    run_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    run = ReconciliationRun(
        id=run_id,
        project_id='test-project',
        run_type='full',
        trigger_reason='test',
        started_at=now,
        status='running',
    )
    await journal.start_run(run)

    report = StageReport(
        stage=StageId.memory_consolidator,
        started_at=now,
        completed_at=now,
        stats={'created': 1, 'deleted': 2},
        llm_calls=3,
        tokens_used=500,
    )
    await journal.update_run_stage_reports(run_id, {'memory_consolidator': report})

    loaded = await journal.get_run(run_id)
    assert 'memory_consolidator' in loaded.stage_reports
    assert loaded.stage_reports['memory_consolidator'].llm_calls == 3
