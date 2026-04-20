"""Tests for the reconciliation stats verifier."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio

from fused_memory.models.reconciliation import StageId, StageReport
from fused_memory.reconciliation.stats_verifier import (
    _OP_TO_STAT,
    _bucket_ops_by_stage,
    _count_add_memory,
    verify_and_rewrite_stats,
)
from fused_memory.services.write_journal import WriteJournal


@pytest_asyncio.fixture
async def journal(tmp_path):
    j = WriteJournal(tmp_path / 'wj')
    await j.initialize()
    yield j
    await j.close()


async def _log_write(journal: WriteJournal, *, causation_id: str, operation: str,
                     result_summary: dict | None = None, success: bool = True) -> None:
    await journal.log_write_op(
        write_op_id=str(uuid.uuid4()),
        causation_id=causation_id,
        source='mcp_tool',
        operation=operation,
        project_id='test',
        result_summary=result_summary,
        success=success,
    )


def _stage_report(stage_id: StageId, started: datetime, completed: datetime,
                  stats: dict | None = None) -> StageReport:
    return StageReport(
        stage=stage_id,
        started_at=started,
        completed_at=completed,
        stats=stats or {},
    )


# ── _count_add_memory ───────────────────────────────────────────────────


def test_count_add_memory_empty_memory_ids_no_stores_is_false():
    op = {'success': 1, 'result_summary': {'memory_ids': [], 'stores': []}}
    assert _count_add_memory(op) is False


def test_count_add_memory_nonempty_memory_ids_is_true():
    op = {'success': 1, 'result_summary': {'memory_ids': ['m1'], 'stores': []}}
    assert _count_add_memory(op) is True


def test_count_add_memory_graphiti_enqueued_counts_even_if_mem0_deduped():
    op = {'success': 1, 'result_summary': {'memory_ids': [], 'stores': ['graphiti']}}
    assert _count_add_memory(op) is True


def test_count_add_memory_failure_is_false():
    op = {'success': 0, 'result_summary': {'memory_ids': ['m1']}}
    assert _count_add_memory(op) is False


def test_count_add_memory_handles_json_string_result_summary():
    op = {'success': 1, 'result_summary': '{"memory_ids": ["m1"], "stores": ["mem0"]}'}
    assert _count_add_memory(op) is True


# ── _bucket_ops_by_stage ────────────────────────────────────────────────


def test_bucket_ops_by_stage_assigns_to_matching_window():
    now = datetime(2026, 4, 20, 9, 0, 0, tzinfo=UTC)
    s1 = _stage_report(StageId.memory_consolidator, now, now + timedelta(minutes=2))
    s2 = _stage_report(
        StageId.task_knowledge_sync,
        now + timedelta(minutes=2, seconds=1),
        now + timedelta(minutes=4),
    )
    reports: dict[str, StageReport | dict] = {
        'memory_consolidator': s1,
        'task_knowledge_sync': s2,
    }
    ops = [
        {'layer': 'write_op', 'operation': 'add_memory',
         'created_at': (now + timedelta(minutes=1)).isoformat()},
        {'layer': 'write_op', 'operation': 'delete_memory',
         'created_at': (now + timedelta(minutes=3)).isoformat()},
        {'layer': 'write_op', 'operation': 'add_memory',
         'created_at': (now + timedelta(minutes=5)).isoformat()},  # after both stages
    ]
    buckets = _bucket_ops_by_stage(ops, reports)
    assert len(buckets['memory_consolidator']) == 1
    assert len(buckets['task_knowledge_sync']) == 1
    assert len(buckets['_unbucketed']) == 1


# ── verify_and_rewrite_stats ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_verify_rewrites_inflated_stats(journal):
    """Stage claims 3 memories_added; only 1 write_op actually stored one."""
    run_id = str(uuid.uuid4())
    now = datetime.now(UTC)
    stage_start = now - timedelta(minutes=1)
    stage_end = now + timedelta(minutes=1)

    # One successful add_memory with non-empty memory_ids.
    await _log_write(
        journal, causation_id=run_id, operation='add_memory',
        result_summary={'memory_ids': ['m1'], 'stores': ['mem0']},
    )
    # Two silently-deduped add_memory calls: empty memory_ids AND empty stores.
    for _ in range(2):
        await _log_write(
            journal, causation_id=run_id, operation='add_memory',
            result_summary={'memory_ids': [], 'stores': []},
        )

    reports: dict[str, StageReport | dict] = {
        'memory_consolidator': _stage_report(
            StageId.memory_consolidator, stage_start, stage_end,
            stats={'memories_added': 3, 'llm_calls_extra': 42},
        ),
    }

    observed = await verify_and_rewrite_stats(run_id, reports, journal)

    # Observed reflects the 1 real add; the 2 dedup'd attempts are no-ops.
    assert observed['memory_consolidator']['memories_added'] == 1

    stats = reports['memory_consolidator'].stats  # type: ignore[union-attr]
    assert stats['memories_added'] == 1
    # Original inflated value preserved for divergence visibility.
    assert stats['_reported']['memories_added'] == 3
    # Unrelated stats untouched.
    assert stats['llm_calls_extra'] == 42


@pytest.mark.asyncio
async def test_verify_adds_zero_when_stage_claimed_none(journal):
    """Unknown stat keys stay absent; known keys are always written as observed."""
    run_id = str(uuid.uuid4())
    now = datetime.now(UTC)
    reports: dict[str, StageReport | dict] = {
        'integrity_check': _stage_report(
            StageId.integrity_check,
            now - timedelta(minutes=1),
            now + timedelta(minutes=1),
            stats={'tasks_checked': 5},
        ),
    }

    await verify_and_rewrite_stats(run_id, reports, journal)

    stats = reports['integrity_check'].stats  # type: ignore[union-attr]
    # Known stat-keys present as 0 (no ops occurred).
    assert stats['memories_added'] == 0
    assert stats['memories_deleted'] == 0
    # Unrelated stat untouched.
    assert stats['tasks_checked'] == 5
    # No _reported snapshot because stage didn't report any known keys.
    assert '_reported' not in stats


@pytest.mark.asyncio
async def test_verify_tallies_delete_and_update_edge(journal):
    run_id = str(uuid.uuid4())
    now = datetime.now(UTC)

    await _log_write(journal, causation_id=run_id, operation='delete_memory',
                     result_summary={'status': 'deleted'})
    await _log_write(journal, causation_id=run_id, operation='delete_memory',
                     result_summary={'status': 'deleted'})
    await _log_write(journal, causation_id=run_id, operation='update_edge',
                     result_summary={'status': 'updated'})

    reports: dict[str, StageReport | dict] = {
        'memory_consolidator': _stage_report(
            StageId.memory_consolidator,
            now - timedelta(minutes=1),
            now + timedelta(minutes=1),
            stats={'memories_deleted': 5, 'edges_updated': 0},
        ),
    }

    await verify_and_rewrite_stats(run_id, reports, journal)

    stats = reports['memory_consolidator'].stats  # type: ignore[union-attr]
    assert stats['memories_deleted'] == 2
    assert stats['edges_updated'] == 1
    assert stats['_reported']['memories_deleted'] == 5
    assert stats['_reported']['edges_updated'] == 0


@pytest.mark.asyncio
async def test_verify_returns_empty_when_no_write_journal():
    run_id = str(uuid.uuid4())
    now = datetime.now(UTC)
    reports: dict[str, StageReport | dict] = {
        'memory_consolidator': _stage_report(
            StageId.memory_consolidator,
            now - timedelta(minutes=1),
            now + timedelta(minutes=1),
            stats={'memories_added': 2},
        ),
    }

    observed = await verify_and_rewrite_stats(run_id, reports, None)

    assert observed == {}
    # Stage reports untouched.
    assert reports['memory_consolidator'].stats == {'memories_added': 2}  # type: ignore[union-attr]


def test_op_to_stat_mapping_covers_core_memory_operations():
    """Guard against accidental deletions of key op-to-stat mappings."""
    for key in ('add_memory', 'delete_memory', 'update_edge', 'add_episode'):
        assert key in _OP_TO_STAT
