"""Tests for pure stage-stats computation from write-journal ops.

``derive_stage_stats`` is the write-journal-derived source of truth for a
stage's write counters. It filters ops down to the stage's OWN write_ops
(``layer == 'write_op'`` and ``agent_id == stage_agent_id``) and tallies them
via the same op→stat mapping the LLM-counter verifier used to own.
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio

from fused_memory.reconciliation.stage_stats import (
    _COMPUTED_STAT_KEYS,
    derive_stage_stats,
)
from fused_memory.services.write_journal import WriteJournal

_STAGE_AGENT_ID = 'recon-stage-memory_consolidator'
_OTHER_STAGE_AGENT_ID = 'recon-stage-task_knowledge_sync'


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
