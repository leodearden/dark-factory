"""Tests for reconciliation journal (SQLite persistence)."""

import uuid
from datetime import UTC, datetime

import pytest
import pytest_asyncio

from fused_memory.models.reconciliation import (
    JournalEntry,
    JudgeVerdict,
    ReconciliationRun,
    RunStatus,
    RunType,
    StageId,
    StageReport,
    VerdictAction,
    VerdictSeverity,
    Watermark,
)
from fused_memory.reconciliation.journal import ReconciliationJournal


@pytest_asyncio.fixture
async def journal(tmp_path):
    j = ReconciliationJournal(tmp_path / 'test_recon')
    await j.open()
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
    now = datetime.now(UTC)
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
        run_type=RunType.full,
        trigger_reason='buffer_size:10',
        started_at=datetime.now(UTC),
        events_processed=10,
        status=RunStatus.running,
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
        timestamp=datetime.now(UTC),
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
        run_type=RunType.full,
        trigger_reason='test',
        started_at=datetime.now(UTC),
        status=RunStatus.completed,
    )
    await journal.start_run(run)

    verdict = JudgeVerdict(
        run_id=run_id,
        reviewed_at=datetime.now(UTC),
        severity=VerdictSeverity.ok,
        findings=[],
        action_taken=VerdictAction.none,
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
            run_type=RunType.full,
            trigger_reason=f'test_{i}',
            started_at=datetime.now(UTC),
            status=RunStatus.running,
        )
        await journal.start_run(run)

    runs = await journal.get_recent_runs('test-project', limit=10)
    assert len(runs) == 3


@pytest.mark.asyncio
async def test_stats(journal):
    run_id = str(uuid.uuid4())
    now = datetime.now(UTC)
    run = ReconciliationRun(
        id=run_id,
        project_id='test-project',
        run_type=RunType.full,
        trigger_reason='test',
        started_at=now,
        status=RunStatus.running,
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
    now = datetime.now(UTC)
    run = ReconciliationRun(
        id=run_id,
        project_id='test-project',
        run_type=RunType.full,
        trigger_reason='test',
        started_at=now,
        status=RunStatus.running,
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


@pytest.mark.asyncio
async def test_add_run_action(journal):
    run_id = str(uuid.uuid4())
    await journal.add_run_action(
        run_id, 'write', 'memory', 'add_memory',
        {'task_id': '1', 'type': 'completion'},
        causation_id=run_id,
    )
    actions = await journal.get_run_actions(run_id)
    assert len(actions) == 1
    assert actions[0]['action_type'] == 'write'
    assert actions[0]['target'] == 'memory'
    assert actions[0]['operation'] == 'add_memory'
    assert actions[0]['causation_id'] == run_id
    assert actions[0]['detail']['task_id'] == '1'


@pytest.mark.asyncio
async def test_get_run_actions_multiple(journal):
    run_id = str(uuid.uuid4())
    await journal.add_run_action(run_id, 'write', 'memory', 'add_memory', {'n': 1})
    await journal.add_run_action(run_id, 'read', 'search', 'search', {'n': 2})
    await journal.add_run_action(run_id, 'write', 'taskmaster', 'update_task', {'n': 3})

    actions = await journal.get_run_actions(run_id)
    assert len(actions) == 3
    assert {a['operation'] for a in actions} == {'add_memory', 'search', 'update_task'}


@pytest.mark.asyncio
async def test_get_run_actions_combined_without_write_journal(journal):
    """Without a write journal, combined returns only run_actions."""
    run_id = str(uuid.uuid4())
    await journal.add_run_action(run_id, 'write', 'memory', 'add_memory')
    combined = await journal.get_run_actions_combined(run_id)
    assert len(combined) == 1
    assert combined[0]['source'] == 'run_actions'


@pytest.mark.asyncio
async def test_get_run_actions_combined_with_write_journal(journal, tmp_path):
    """Combined returns actions from both run_actions and write journal."""
    from fused_memory.services.write_journal import WriteJournal

    wj = WriteJournal(tmp_path / 'combined_wj')
    await wj.open()
    journal.set_write_journal(wj)

    run_id = str(uuid.uuid4())
    await journal.add_run_action(run_id, 'write', 'memory', 'add_memory')
    await wj.log_write_op(
        write_op_id=str(uuid.uuid4()),
        causation_id=run_id,
        operation='add_memory',
        project_id='test',
    )

    combined = await journal.get_run_actions_combined(run_id)
    sources = {a['source'] for a in combined}
    assert 'run_actions' in sources
    assert 'write_journal' in sources
    assert len(combined) == 2

    await wj.close()


@pytest.mark.asyncio
async def test_combined_extracts_target_from_write_journal(journal, tmp_path):
    """Write journal ops get target extracted from result_summary."""
    from fused_memory.services.write_journal import WriteJournal

    wj = WriteJournal(tmp_path / 'target_wj')
    await wj.open()
    journal.set_write_journal(wj)

    run_id = str(uuid.uuid4())
    mem_id = 'abc-123-def'
    await wj.log_write_op(
        write_op_id=str(uuid.uuid4()),
        causation_id=run_id,
        operation='delete_memory',
        project_id='test',
        result_summary={'status': 'deleted', 'store': 'mem0', 'id': mem_id},
    )

    combined = await journal.get_run_actions_combined(run_id)
    assert len(combined) == 1
    assert combined[0]['target'] == mem_id

    await wj.close()


class TestExtractTarget:
    """Unit tests for _extract_target helper."""

    def test_delete_memory_id_from_result(self):
        from fused_memory.reconciliation.journal import _extract_target

        op = {'result_summary': '{"status":"deleted","id":"mem-42"}'}
        assert _extract_target(op) == 'mem-42'

    def test_add_memory_single_id(self):
        from fused_memory.reconciliation.journal import _extract_target

        op = {'result_summary': '{"memory_ids":["ep-1"],"stores":["mem0"]}'}
        assert _extract_target(op) == 'ep-1'

    def test_add_memory_multiple_ids(self):
        from fused_memory.reconciliation.journal import _extract_target

        op = {'result_summary': '{"memory_ids":["a","b","c"],"stores":["mem0","graphiti"]}'}
        assert _extract_target(op) == '3 memories'

    def test_add_episode_id(self):
        from fused_memory.reconciliation.journal import _extract_target

        op = {'result_summary': '{"episode_id":"ep-99","status":"queued"}'}
        assert _extract_target(op) == 'ep-99'

    def test_fallback_to_params_memory_id(self):
        from fused_memory.reconciliation.journal import _extract_target

        op = {'result_summary': '{"status":"deleted"}', 'params': '{"memory_id":"m-7"}'}
        assert _extract_target(op) == 'm-7'

    def test_fallback_to_params_query(self):
        from fused_memory.reconciliation.journal import _extract_target

        op = {'result_summary': '{"count":5}', 'params': '{"query":"recent decisions","limit":10}'}
        assert _extract_target(op) == 'recent decisions'

    def test_dict_inputs(self):
        from fused_memory.reconciliation.journal import _extract_target

        op = {'result_summary': {'id': 'direct-dict'}}
        assert _extract_target(op) == 'direct-dict'

    def test_missing_fields_returns_question_mark(self):
        from fused_memory.reconciliation.journal import _extract_target

        assert _extract_target({}) == '?'
        assert _extract_target({'result_summary': '{}'}) == '?'

    def test_malformed_json_returns_question_mark(self):
        from fused_memory.reconciliation.journal import _extract_target

        assert _extract_target({'result_summary': 'not json'}) == '?'
