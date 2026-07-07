"""Tests for pure stage-stats computation from write-journal ops.

``derive_stage_stats`` is the write-journal-derived source of truth for a
stage's write counters. It filters ops down to the stage's OWN write_ops
(``layer == 'write_op'`` and ``agent_id == stage_agent_id``) and tallies them
via the same op→stat mapping the LLM-counter verifier used to own.
"""

from __future__ import annotations

import json
import uuid

import pytest
import pytest_asyncio

from fused_memory.reconciliation.stage_stats import (
    _COMPUTED_STAT_KEYS,
    _OP_TO_STAT,
    _count_add_memory,
    _count_graphiti_queued,
    derive_stage_stats,
)
from fused_memory.services.write_journal import WriteJournal

_STAGE_AGENT_ID = 'recon-stage-memory_consolidator'
_OTHER_STAGE_AGENT_ID = 'recon-stage-task_knowledge_sync'


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
    rs = json.dumps({'memory_ids': [], 'stores': ['graphiti']})
    op = {'success': 1, 'result_summary': rs}
    assert _count_graphiti_queued(op) is True


def test_op_to_stat_mapping_covers_core_memory_operations():
    """Guard against accidental deletions of key op-to-stat mappings."""
    for key in ('add_memory', 'delete_memory', 'update_edge', 'add_episode'):
        assert key in _OP_TO_STAT


@pytest_asyncio.fixture
async def journal(tmp_path):
    j = WriteJournal(tmp_path / 'wj')
    await j.initialize()
    yield j
    await j.close()


async def _log_write(
    journal: WriteJournal,
    *,
    causation_id: str,
    operation: str,
    agent_id: str = _STAGE_AGENT_ID,
    result_summary: dict | str | None = None,
    success: bool = True,
) -> None:
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


def _expected(**overrides: int) -> dict[str, int]:
    """Every canonical counter key at 0, with the given keys overridden.

    Mirrors derive_stage_stats' own 0-default seeding so tests assert exact
    per-key counts rather than just the keys a scenario happens to touch.
    """
    expected = dict.fromkeys(_COMPUTED_STAT_KEYS, 0)
    expected.update(overrides)
    return expected


@pytest.mark.asyncio
async def test_derive_stage_stats_tallies_own_stage_write_ops(journal):
    """add_memory / delete_memory / add_episode ops stamped with the stage's own
    agent_id are tallied via the op-to-stat mapping."""
    run_id = str(uuid.uuid4())

    await _log_write(
        journal, causation_id=run_id, operation='add_memory',
        result_summary={'memory_ids': ['m1'], 'stores': ['mem0']},
    )
    await _log_write(
        journal, causation_id=run_id, operation='delete_memory',
        result_summary={'status': 'deleted'},
    )
    await _log_write(
        journal, causation_id=run_id, operation='add_episode',
        result_summary={'status': 'added'},
    )

    ops = await journal.get_ops_by_causation(run_id)
    observed = derive_stage_stats(ops, _STAGE_AGENT_ID)

    assert observed['memories_added'] == 1
    assert observed['memories_deleted'] == 1
    assert observed['episodes_added'] == 1


@pytest.mark.asyncio
async def test_derive_stage_stats_returns_all_canonical_keys_zero_default_with_no_ops(
    journal,
):
    """Even with zero matching ops, every canonical counter key is present at 0
    — downstream override logic depends on a complete, deterministic key set."""
    run_id = str(uuid.uuid4())

    ops = await journal.get_ops_by_causation(run_id)
    observed = derive_stage_stats(ops, _STAGE_AGENT_ID)

    assert set(_COMPUTED_STAT_KEYS) <= set(observed)
    for key in _COMPUTED_STAT_KEYS:
        assert observed[key] == 0


@pytest.mark.asyncio
async def test_derive_stage_stats_excludes_other_stage_agent_id(journal):
    """An op stamped with a different stage's agent_id must not be tallied —
    agent_id bucketing is exact, unlike the old timestamp-window bucketing."""
    run_id = str(uuid.uuid4())

    await _log_write(
        journal, causation_id=run_id, operation='add_memory',
        agent_id=_OTHER_STAGE_AGENT_ID,
        result_summary={'memory_ids': ['m1'], 'stores': ['mem0']},
    )

    ops = await journal.get_ops_by_causation(run_id)
    observed = derive_stage_stats(ops, _STAGE_AGENT_ID)

    assert observed['memories_added'] == 0


@pytest.mark.asyncio
async def test_derive_stage_stats_dedup_add_memory_counts_toward_neither_key(journal):
    """A silently-deduped add_memory (empty memory_ids, no stores) is a no-op —
    it must not inflate memories_added nor graphiti_writes_queued."""
    run_id = str(uuid.uuid4())
    await _log_write(
        journal, causation_id=run_id, operation='add_memory',
        result_summary={'memory_ids': [], 'stores': []},
    )

    ops = await journal.get_ops_by_causation(run_id)
    observed = derive_stage_stats(ops, _STAGE_AGENT_ID)

    assert observed == _expected()


@pytest.mark.asyncio
async def test_derive_stage_stats_graphiti_only_enqueue_counts_queued_not_added(journal):
    run_id = str(uuid.uuid4())
    await _log_write(
        journal, causation_id=run_id, operation='add_memory',
        result_summary={'memory_ids': [], 'stores': ['graphiti']},
    )

    ops = await journal.get_ops_by_causation(run_id)
    observed = derive_stage_stats(ops, _STAGE_AGENT_ID)

    assert observed == _expected(graphiti_writes_queued=1)


@pytest.mark.asyncio
async def test_derive_stage_stats_graphiti_only_enqueue_stores_written_alias(journal):
    """The 'stores_written' key is accepted as an alias for 'stores'."""
    run_id = str(uuid.uuid4())
    await _log_write(
        journal, causation_id=run_id, operation='add_memory',
        result_summary={'memory_ids': [], 'stores_written': ['graphiti']},
    )

    ops = await journal.get_ops_by_causation(run_id)
    observed = derive_stage_stats(ops, _STAGE_AGENT_ID)

    assert observed == _expected(graphiti_writes_queued=1)


@pytest.mark.asyncio
async def test_derive_stage_stats_update_edge_counts_only_when_verified_true(journal):
    """Two update_edge ops, same stage: one verified, one not. Only the
    verified op counts."""
    run_id = str(uuid.uuid4())
    await _log_write(
        journal, causation_id=run_id, operation='update_edge',
        result_summary={'verified': True, 'status': 'updated'},
    )
    await _log_write(
        journal, causation_id=run_id, operation='update_edge',
        result_summary={'verified': False, 'status': 'updated'},
    )

    ops = await journal.get_ops_by_causation(run_id)
    observed = derive_stage_stats(ops, _STAGE_AGENT_ID)

    assert observed == _expected(edges_updated=1)


@pytest.mark.asyncio
async def test_derive_stage_stats_update_edge_missing_verified_key_excluded(journal):
    """A legacy update_edge op with no 'verified' key must not count — missing
    means 'not verified', not 'trust the legacy entry'."""
    run_id = str(uuid.uuid4())
    await _log_write(
        journal, causation_id=run_id, operation='update_edge',
        result_summary={'status': 'updated'},
    )

    ops = await journal.get_ops_by_causation(run_id)
    observed = derive_stage_stats(ops, _STAGE_AGENT_ID)

    assert observed == _expected()


@pytest.mark.asyncio
async def test_derive_stage_stats_update_edge_truthy_non_true_verified_excluded(journal):
    """Truthy-but-not-True values like the string 'true' must not count — the
    strict `is True` identity check rejects loosely-typed values."""
    run_id = str(uuid.uuid4())
    await _log_write(
        journal, causation_id=run_id, operation='update_edge',
        result_summary={'verified': 'true', 'status': 'updated'},
    )

    ops = await journal.get_ops_by_causation(run_id)
    observed = derive_stage_stats(ops, _STAGE_AGENT_ID)

    assert observed == _expected()


@pytest.mark.asyncio
async def test_derive_stage_stats_excludes_failed_ops(journal):
    """A generic-counter op (delete_memory) with success=0 must not count."""
    run_id = str(uuid.uuid4())
    await _log_write(
        journal, causation_id=run_id, operation='delete_memory',
        result_summary={'status': 'deleted'}, success=False,
    )

    ops = await journal.get_ops_by_causation(run_id)
    observed = derive_stage_stats(ops, _STAGE_AGENT_ID)

    assert observed == _expected()


@pytest.mark.asyncio
async def test_derive_stage_stats_parses_json_string_result_summary(journal):
    """result_summary logged as a JSON string (rather than a dict) is parsed."""
    run_id = str(uuid.uuid4())
    await _log_write(
        journal, causation_id=run_id, operation='add_memory',
        result_summary=json.dumps({'memory_ids': ['m1'], 'stores': ['mem0']}),
    )

    ops = await journal.get_ops_by_causation(run_id)
    observed = derive_stage_stats(ops, _STAGE_AGENT_ID)

    assert observed == _expected(memories_added=1)


@pytest.mark.asyncio
async def test_derive_stage_stats_excludes_backend_op_layer(journal):
    """backend_op-layer entries are a second audit layer of the same write and
    would double-count if tallied alongside the write_op."""
    run_id = str(uuid.uuid4())

    await journal.log_backend_op(
        causation_id=run_id,
        backend='graphiti',
        operation='add_memory',
        result_summary={'memory_ids': ['m1'], 'stores': ['graphiti']},
    )

    ops = await journal.get_ops_by_causation(run_id)
    observed = derive_stage_stats(ops, _STAGE_AGENT_ID)

    assert observed['memories_added'] == 0
