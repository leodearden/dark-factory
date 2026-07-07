"""Tests for the reconciliation stats verifier."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio

from fused_memory.models.reconciliation import StageId, StageReport
from fused_memory.reconciliation.stage_stats import _COMPUTED_STAT_KEYS
from fused_memory.reconciliation.stats_verifier import (
    _OP_TO_STAT,
    _STAT_ALIASES,
    _bucket_ops_by_stage,
    _count_add_memory,
    _count_graphiti_queued,
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
                     agent_id: str = 'recon-stage-memory_consolidator',
                     result_summary: dict | None = None, success: bool = True) -> None:
    await journal.log_write_op(
        write_op_id=str(uuid.uuid4()),
        causation_id=causation_id,
        source='mcp_tool',
        operation=operation,
        project_id='test',
        agent_id=agent_id,
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


def test_count_add_memory_graphiti_enqueued_without_memory_ids_is_false():
    op = {'success': 1, 'result_summary': {'memory_ids': [], 'stores': ['graphiti']}}
    assert _count_add_memory(op) is False


def test_count_add_memory_failure_is_false():
    op = {'success': 0, 'result_summary': {'memory_ids': ['m1']}}
    assert _count_add_memory(op) is False


def test_count_add_memory_handles_json_string_result_summary():
    op = {'success': 1, 'result_summary': '{"memory_ids": ["m1"], "stores": ["mem0"]}'}
    assert _count_add_memory(op) is True


# ── _count_graphiti_queued ──────────────────────────────────────────────


def test_count_graphiti_queued_true_when_empty_ids_and_graphiti_store():
    op = {'success': 1, 'result_summary': {'memory_ids': [], 'stores': ['graphiti']}}
    assert _count_graphiti_queued(op) is True


def test_count_graphiti_queued_false_when_nonempty_memory_ids():
    # Already counted as memories_added — not a graphiti-only enqueue.
    op = {'success': 1, 'result_summary': {'memory_ids': ['m1'], 'stores': ['graphiti']}}
    assert _count_graphiti_queued(op) is False


def test_count_graphiti_queued_false_when_mem0_only():
    op = {'success': 1, 'result_summary': {'memory_ids': [], 'stores': ['mem0']}}
    assert _count_graphiti_queued(op) is False


def test_count_graphiti_queued_false_when_failed():
    op = {'success': 0, 'result_summary': {'memory_ids': [], 'stores': ['graphiti']}}
    assert _count_graphiti_queued(op) is False


def test_count_graphiti_queued_true_for_stores_written_alias():
    op = {'success': 1, 'result_summary': {'memory_ids': [], 'stores_written': ['graphiti']}}
    assert _count_graphiti_queued(op) is True


def test_count_graphiti_queued_true_for_json_string_result_summary():
    import json
    rs = json.dumps({'memory_ids': [], 'stores': ['graphiti']})
    op = {'success': 1, 'result_summary': rs}
    assert _count_graphiti_queued(op) is True


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
async def test_verify_boundary_s1_computed_stats_override_llm_report(journal):
    """Boundary signal S1 (task 2226): the write journal, not the LLM's
    self-report, is the source of truth for a stage's write counters.

    The LLM reports memories_added=5 and memories_written=5 (both inflated);
    the journal shows only 2 real adds. After verification: memories_added is
    overridden to the observed count (2); both originals are preserved under
    _reported as the judge's divergence signal; memories_written — recognised
    as an LLM counter but no longer computed now that the alias mirror is gone
    — is dropped from the top-level stats entirely (its claim still visible
    under _reported); and a code-computed non-counter stat
    (graphiti_queue_health) is left untouched.
    """
    run_id = str(uuid.uuid4())
    now = datetime.now(UTC)

    for i in range(2):
        await _log_write(
            journal, causation_id=run_id, operation='add_memory',
            result_summary={'memory_ids': [f'm{i}'], 'stores': ['mem0']},
        )

    reports: dict[str, StageReport | dict] = {
        'memory_consolidator': _stage_report(
            StageId.memory_consolidator,
            now - timedelta(minutes=1), now + timedelta(minutes=1),
            stats={
                'memories_added': 5,
                'memories_written': 5,
                'graphiti_queue_health': {'depth': 3},
            },
        ),
    }

    await verify_and_rewrite_stats(run_id, reports, journal)

    stats = reports['memory_consolidator'].stats  # type: ignore[union-attr]
    assert stats['memories_added'] == 2
    assert stats['_reported']['memories_added'] == 5
    assert stats['_reported']['memories_written'] == 5
    assert 'memories_written' not in stats
    # Code-computed non-counter stat survives the override untouched.
    assert stats['graphiti_queue_health'] == {'depth': 3}


def test_stats_verifier_has_no_alias_map():
    """_STAT_ALIASES is retired — memories_written no longer mirrors a
    canonical key; it's either dropped (S1) or absent entirely."""
    import fused_memory.reconciliation.stats_verifier as sv  # noqa: PLC0415

    assert not hasattr(sv, '_STAT_ALIASES')


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
    # 'graphiti_writes_queued' was not in original stats — must not appear in _reported.
    assert 'graphiti_writes_queued' not in stats['_reported']
    # Unrelated stats untouched.
    assert stats['llm_calls_extra'] == 42


@pytest.mark.asyncio
async def test_verify_adds_zero_when_stage_claimed_none(journal):
    """Unknown stat keys stay absent; every canonical key is always written as
    observed, 0-default, even when no ops occurred."""
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
    # Every canonical stat-key present as 0 (no ops occurred).
    for key in _COMPUTED_STAT_KEYS:
        assert stats[key] == 0
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
                     result_summary={'status': 'updated', 'verified': True})

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
async def test_verify_tracks_graphiti_writes_queued_separately(journal):
    """Three add_memory ops: 2 with memory_ids, 1 graphiti-only enqueue.

    The verifier must attribute:
    - memories_added = 2  (ops a and c which returned non-empty memory_ids)
    - graphiti_writes_queued = 1  (op b: memory_ids=[], stores=['graphiti'])

    The LLM-emitted originals (5, 5, 0) must be preserved under _reported.
    """
    run_id = str(uuid.uuid4())
    now = datetime.now(UTC)
    stage_start = now - timedelta(minutes=1)
    stage_end = now + timedelta(minutes=1)

    # (a) mem0-only successful write — counts as memories_added.
    await _log_write(
        journal, causation_id=run_id, operation='add_memory',
        result_summary={'memory_ids': ['m1'], 'stores': ['mem0']},
    )
    # (b) graphiti-only enqueue — must NOT count as memories_added,
    #     must count as graphiti_writes_queued.
    await _log_write(
        journal, causation_id=run_id, operation='add_memory',
        result_summary={'memory_ids': [], 'stores': ['graphiti']},
    )
    # (c) mixed write: both stores, IDs returned — counts as memories_added.
    await _log_write(
        journal, causation_id=run_id, operation='add_memory',
        result_summary={'memory_ids': ['m2'], 'stores': ['mem0', 'graphiti']},
    )

    reports: dict[str, StageReport | dict] = {
        'memory_consolidator': _stage_report(
            StageId.memory_consolidator, stage_start, stage_end,
            stats={'memories_added': 5, 'memories_written': 5, 'graphiti_writes_queued': 0},
        ),
    }

    observed = await verify_and_rewrite_stats(run_id, reports, journal)

    assert observed['memory_consolidator']['memories_added'] == 2
    assert observed['memory_consolidator']['graphiti_writes_queued'] == 1

    stats = reports['memory_consolidator'].stats  # type: ignore[union-attr]
    assert stats['memories_added'] == 2
    assert stats['graphiti_writes_queued'] == 1
    # LLM-emitted originals preserved for divergence visibility.
    assert stats['_reported']['memories_added'] == 5
    assert stats['_reported']['graphiti_writes_queued'] == 0


@pytest.mark.asyncio
async def test_verify_overrides_all_computed_keys_with_divergent_originals(journal):
    """LLM reports inflated memories_added=9, memories_written=9, graphiti_writes_queued=7.

    Observed: 1 mem0-id op + 1 graphiti-only enqueue. Verifier overwrites both
    computed keys (memories_added, graphiti_writes_queued) to 1 and snapshots
    all three originals under _reported; memories_written is a recognised LLM
    counter that is no longer computed, so it is dropped from the top level
    entirely rather than mirrored.
    """
    run_id = str(uuid.uuid4())
    now = datetime.now(UTC)
    stage_start = now - timedelta(minutes=1)
    stage_end = now + timedelta(minutes=1)

    # (a) One mem0-id success — counts as memories_added.
    await _log_write(
        journal, causation_id=run_id, operation='add_memory',
        result_summary={'memory_ids': ['m1'], 'stores': ['mem0']},
    )
    # (b) One graphiti-only enqueue — counts as graphiti_writes_queued.
    await _log_write(
        journal, causation_id=run_id, operation='add_memory',
        result_summary={'memory_ids': [], 'stores': ['graphiti']},
    )

    reports: dict[str, StageReport | dict] = {
        'memory_consolidator': _stage_report(
            StageId.memory_consolidator, stage_start, stage_end,
            stats={'memories_added': 9, 'memories_written': 9, 'graphiti_writes_queued': 7},
        ),
    }

    observed = await verify_and_rewrite_stats(run_id, reports, journal)

    assert observed['memory_consolidator']['memories_added'] == 1
    assert observed['memory_consolidator']['graphiti_writes_queued'] == 1

    stats = reports['memory_consolidator'].stats  # type: ignore[union-attr]
    # Both computed keys overwritten with observed values.
    assert stats['memories_added'] == 1
    assert stats['graphiti_writes_queued'] == 1
    # memories_written is recognised but not computed — dropped from the top level.
    assert 'memories_written' not in stats
    # All three originals snapshotted under _reported.
    assert stats['_reported']['memories_added'] == 9
    assert stats['_reported']['memories_written'] == 9
    assert stats['_reported']['graphiti_writes_queued'] == 7


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


@pytest.mark.asyncio
async def test_verify_handles_plain_dict_report_without_stats_key(journal):
    """A stage report given as a plain dict (not a StageReport), with no
    pre-existing 'stats' key, gets one created and populated from observed
    computed counters — same override behaviour as the StageReport branch."""
    run_id = str(uuid.uuid4())

    await _log_write(
        journal, causation_id=run_id, operation='add_memory',
        result_summary={'memory_ids': ['m1'], 'stores': ['mem0']},
    )

    reports: dict[str, StageReport | dict] = {'memory_consolidator': {}}

    await verify_and_rewrite_stats(run_id, reports, journal)

    stats = reports['memory_consolidator']['stats']  # type: ignore[index]
    assert stats['memories_added'] == 1
    assert '_reported' not in stats


@pytest.mark.asyncio
async def test_verify_per_stage_isolation_across_two_stages(journal):
    """Ops stamped for one stage's agent_id must not count toward a sibling
    stage processed in the same verify_and_rewrite_stats call — agent_id
    bucketing is exact, unlike the old timestamp-window bucketing it replaced."""
    run_id = str(uuid.uuid4())
    now = datetime.now(UTC)
    stage_start = now - timedelta(minutes=1)
    stage_end = now + timedelta(minutes=1)

    await _log_write(
        journal, causation_id=run_id, operation='add_memory',
        agent_id='recon-stage-memory_consolidator',
        result_summary={'memory_ids': ['m1'], 'stores': ['mem0']},
    )
    await _log_write(
        journal, causation_id=run_id, operation='delete_memory',
        agent_id='recon-stage-task_knowledge_sync',
        result_summary={'status': 'deleted'},
    )

    reports: dict[str, StageReport | dict] = {
        'memory_consolidator': _stage_report(
            StageId.memory_consolidator, stage_start, stage_end,
        ),
        'task_knowledge_sync': _stage_report(
            StageId.task_knowledge_sync, stage_start, stage_end,
        ),
    }

    await verify_and_rewrite_stats(run_id, reports, journal)

    mc_stats = reports['memory_consolidator'].stats  # type: ignore[union-attr]
    tks_stats = reports['task_knowledge_sync'].stats  # type: ignore[union-attr]

    assert mc_stats['memories_added'] == 1
    assert mc_stats['memories_deleted'] == 0
    assert tks_stats['memories_added'] == 0
    assert tks_stats['memories_deleted'] == 1


@pytest.mark.asyncio
async def test_verify_skips_non_stage_id_keys_like_error(journal):
    """A non-StageId key such as '_error' (injected by the harness/judge for a
    crashed stage) is skipped entirely — no counter keys are ever injected."""
    run_id = str(uuid.uuid4())

    reports: dict[str, StageReport | dict] = {
        '_error': {'message': 'stage crashed', 'stats': {}},
    }

    observed = await verify_and_rewrite_stats(run_id, reports, journal)

    assert '_error' not in observed
    assert reports['_error'] == {'message': 'stage crashed', 'stats': {}}


@pytest.mark.asyncio
async def test_verify_aliases_memories_written_to_memories_added(journal):
    """memories_written must mirror memories_added after verification.

    The LLM uses both 'memories_added' and 'memories_written' interchangeably.
    The verifier must write both keys to the same observed count and snapshot
    both original LLM-emitted values under _reported.
    """
    run_id = str(uuid.uuid4())
    now = datetime.now(UTC)
    stage_start = now - timedelta(minutes=1)
    stage_end = now + timedelta(minutes=1)

    # One successful add_memory with non-empty memory_ids.
    await _log_write(
        journal, causation_id=run_id, operation='add_memory',
        result_summary={'memory_ids': ['m1'], 'stores': ['mem0']},
    )

    # Stage reports both keys with the same (inflated) value.
    reports: dict[str, StageReport | dict] = {
        'memory_consolidator': _stage_report(
            StageId.memory_consolidator, stage_start, stage_end,
            stats={'memories_added': 4, 'memories_written': 4},
        ),
    }

    await verify_and_rewrite_stats(run_id, reports, journal)

    stats = reports['memory_consolidator'].stats  # type: ignore[union-attr]
    # Both keys overwritten with the observed value (1).
    assert stats['memories_added'] == 1
    assert stats['memories_written'] == 1
    # Both originals preserved under _reported.
    assert stats['_reported']['memories_added'] == 4
    assert stats['_reported']['memories_written'] == 4


@pytest.mark.asyncio
async def test_verify_aliases_memories_written_when_only_written_key_present(journal):
    """When stage only reports memories_written (no memories_added), verifier
    writes both keys and snapshots only memories_written in _reported.
    """
    run_id = str(uuid.uuid4())
    now = datetime.now(UTC)
    stage_start = now - timedelta(minutes=1)
    stage_end = now + timedelta(minutes=1)

    await _log_write(
        journal, causation_id=run_id, operation='add_memory',
        result_summary={'memory_ids': ['m1'], 'stores': ['mem0']},
    )

    reports: dict[str, StageReport | dict] = {
        'memory_consolidator': _stage_report(
            StageId.memory_consolidator, stage_start, stage_end,
            stats={'memories_written': 3},
        ),
    }

    await verify_and_rewrite_stats(run_id, reports, journal)

    stats = reports['memory_consolidator'].stats  # type: ignore[union-attr]
    # Both keys written with observed value.
    assert stats['memories_added'] == 1
    assert stats['memories_written'] == 1
    # Only memories_written was originally present — memories_added was absent.
    assert 'memories_written' in stats['_reported']
    assert stats['_reported']['memories_written'] == 3
    # memories_added was not in the original stats, so NOT in _reported.
    assert 'memories_added' not in stats['_reported']


def test_op_to_stat_mapping_covers_core_memory_operations():
    """Guard against accidental deletions of key op-to-stat mappings."""
    for key in ('add_memory', 'delete_memory', 'update_edge', 'add_episode'):
        assert key in _OP_TO_STAT


def test_stat_aliases_values_are_canonical_keys():
    """Guard: _STAT_ALIASES values must all resolve to keys in the canonical set.

    Prevents chained aliases (a → b where b is itself an alias key) from
    silently propagating stale values in _apply_observed. The canonical set is
    _OP_TO_STAT.values() ∪ {'graphiti_writes_queued'}.
    """
    canonical = frozenset(_OP_TO_STAT.values()) | {'graphiti_writes_queued'}
    for alias_key, canonical_key in _STAT_ALIASES.items():
        assert canonical_key in canonical, (
            f"_STAT_ALIASES['{alias_key}'] = '{canonical_key}' is not a canonical key. "
            f"Canonical keys: {sorted(canonical)}. "
            "Chained aliases silently propagate stale values in _apply_observed."
        )


class TestUpdateEdgeVerifiedFilter:
    """Only verified update_edge ops should count toward edges_updated (step-17 / step-19)."""

    @pytest.mark.asyncio
    async def test_unverified_update_edge_op_excluded_from_edges_updated(self, journal):
        """Two update_edge ops with same causation_id: one verified=True, one verified=False.
        Only the verified op must count.
        """
        run_id = str(uuid.uuid4())
        now = datetime.now(UTC)

        await _log_write(
            journal, causation_id=run_id, operation='update_edge',
            result_summary={'verified': True, 'status': 'updated', 'store': 'graphiti'},
        )
        await _log_write(
            journal, causation_id=run_id, operation='update_edge',
            result_summary={'verified': False, 'status': 'updated', 'store': 'graphiti'},
        )

        reports: dict[str, StageReport | dict] = {
            'memory_consolidator': _stage_report(
                StageId.memory_consolidator,
                now - timedelta(minutes=1),
                now + timedelta(minutes=1),
                stats={'edges_updated': 2},
            ),
        }

        await verify_and_rewrite_stats(run_id, reports, journal)

        stats = reports['memory_consolidator'].stats  # type: ignore[union-attr]
        assert stats['edges_updated'] == 1, (
            'Only the verified=True update_edge op should count; verified=False must be excluded'
        )

    @pytest.mark.asyncio
    async def test_legacy_update_edge_op_without_verified_field_is_not_counted(self, journal):
        """A legacy update_edge op with no 'verified' key in result_summary must NOT count.

        Strict interpretation: missing verified field is treated as 'not verified',
        not 'trust the legacy entry'. Post-rollout cycles cannot accidentally count
        un-readback ops.
        """
        run_id = str(uuid.uuid4())
        now = datetime.now(UTC)

        await _log_write(
            journal, causation_id=run_id, operation='update_edge',
            result_summary={'status': 'updated', 'store': 'graphiti'},  # no 'verified' key
        )

        reports: dict[str, StageReport | dict] = {
            'memory_consolidator': _stage_report(
                StageId.memory_consolidator,
                now - timedelta(minutes=1),
                now + timedelta(minutes=1),
            ),
        }

        await verify_and_rewrite_stats(run_id, reports, journal)

        stats = reports['memory_consolidator'].stats  # type: ignore[union-attr]
        assert stats['edges_updated'] == 0, (
            'Legacy update_edge op without verified key must not count toward edges_updated'
        )

    @pytest.mark.asyncio
    async def test_truthy_non_true_verified_value_does_not_count(self, journal):
        """Truthy-but-not-True values like 'true' (string) or 1 (int) must NOT count.

        The strict ``is True`` identity check in ``_count_update_edge`` prevents
        coerced / loosely-typed values from slipping through.  This test pins
        the string-'true' case explicitly.
        """
        run_id = str(uuid.uuid4())
        now = datetime.now(UTC)

        await _log_write(
            journal, causation_id=run_id, operation='update_edge',
            result_summary={'verified': 'true', 'status': 'updated', 'store': 'graphiti'},
        )

        reports: dict[str, StageReport | dict] = {
            'memory_consolidator': _stage_report(
                StageId.memory_consolidator,
                now - timedelta(minutes=1),
                now + timedelta(minutes=1),
            ),
        }

        await verify_and_rewrite_stats(run_id, reports, journal)

        stats = reports['memory_consolidator'].stats  # type: ignore[union-attr]
        assert stats['edges_updated'] == 0, (
            "Truthy-but-not-True 'verified' (e.g. string 'true') must not count as verified"
        )


def test_alias_pass_reads_from_observed_for_order_independence(monkeypatch):
    """Alias pass reads from observed, not stats — verifies order-independence under a chained alias."""
    import fused_memory.reconciliation.stats_verifier as sv  # noqa: PLC0415

    # Inject a chained alias AFTER the existing 'memories_written' entry so
    # the old code produces the wrong result via insertion-order luck
    # (memories_written is processed first, setting stats['memories_written']=1,
    # then foo reads that stale stats value and incorrectly becomes 1 too).
    chained_aliases = {'memories_written': 'memories_added', 'foo': 'memories_written'}
    monkeypatch.setattr(sv, '_STAT_ALIASES', chained_aliases)
    monkeypatch.setattr(sv, '_TRACKED_STAT_KEYS',
                        sv._TRACKED_STAT_KEYS | frozenset({'foo'}))

    # One successful add_memory op → observed = {'memories_added': 1}
    observed = {'memories_added': 1}
    now = datetime.now(UTC)
    report = _stage_report(StageId.memory_consolidator, now - timedelta(minutes=1), now + timedelta(minutes=1))
    sv._apply_observed(report, observed)

    stats = report.stats

    # Single-level alias: memories_written -> memories_added (canonical, in observed).
    assert stats['memories_added'] == 1
    assert stats['memories_written'] == 1

    # Chained alias: foo -> memories_written (alias key, NOT in observed).
    # Fixed behavior:  observed.get('memories_written', 0) = 0  (deterministic).
    # Old behavior:    stats['memories_written'] = 1             (insertion-order accident).
    assert stats['foo'] == 0
