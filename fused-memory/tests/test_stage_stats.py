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
    _landed,
    derive_stage_stats,
)
from fused_memory.services.durable_queue import POST_EXECUTE_DEAD_PREFIX
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
) -> str:
    """Journal one Layer-1 write_op and return its ``write_op_id``.

    The id is returned (rather than discarded inline) so a test can hand it to
    :func:`_stamp_terminal` afterwards. Keyword-only signature and defaults are
    unchanged, so existing call sites that ignore the return value still work.
    """
    write_op_id = str(uuid.uuid4())
    await journal.log_write_op(
        write_op_id=write_op_id,
        causation_id=causation_id,
        source='mcp_tool',
        operation=operation,
        project_id='test',
        agent_id=agent_id,
        result_summary=result_summary,
        success=success,
    )
    return write_op_id


async def _stamp_terminal(
    journal: WriteJournal,
    write_op_id: str,
    *,
    status: str,
    error: str | None = None,
) -> None:
    """Stamp a durable-queue terminal outcome onto an already-journalled op.

    ORDER IS LOAD-BEARING: ``_log_write`` must be awaited for ``write_op_id``
    BEFORE this call. The row then already carries ``causation_id`` /
    ``agent_id`` / ``operation``, so ``record_terminal_outcome``'s UPSERT takes
    its ``ON CONFLICT(id) DO UPDATE`` path — which deliberately touches only
    the three terminal columns. Stamping first would instead INSERT a
    ``source='durable_queue'`` row with a NULL ``causation_id``, which
    ``get_ops_by_causation`` never returns.
    """
    await journal.record_terminal_outcome(
        write_op_id=write_op_id,
        terminal_status=status,
        terminal_error=error,
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


# ── _landed ─────────────────────────────────────────────────────────────


def test_landed_false_for_dead_terminal_status():
    """The durable queue gave up on this write — it did not land."""
    assert _landed({'terminal_status': 'dead'}) is False


def test_landed_true_for_completed_terminal_status():
    assert _landed({'terminal_status': 'completed'}) is True


def test_landed_true_for_null_terminal_status():
    """NULL is the state of every pre-3582 row and of every operation that
    never touches the durable queue — it must keep counting as it does today."""
    assert _landed({'terminal_status': None}) is True


def test_landed_true_when_terminal_status_key_absent():
    """A denylist on the single literal 'dead': anything else keeps its
    current counting behaviour, so a schema surprise can never silently zero
    a stage's real counters."""
    assert _landed({}) is True


def test_landed_true_for_post_execute_dead_letter():
    """The ONE case where 'dead' still counts: the backend write LANDED and
    only the callback / completion-commit kept failing, which the durable
    queue marks by prefixing terminal_error with POST_EXECUTE_DEAD_PREFIX.

    Excluding these would UNDERCOUNT writes that genuinely happened — the
    opposite error from the over-counting this task fixes.
    """
    op = {
        'terminal_status': 'dead',
        'terminal_error': POST_EXECUTE_DEAD_PREFIX + 'callback blew up',
    }
    assert _landed(op) is True


def test_landed_false_for_dead_with_ordinary_error():
    """A dead-letter whose error carries no post-execute prefix never reached
    the backend."""
    assert _landed({'terminal_status': 'dead', 'terminal_error': 'boom'}) is False


def test_landed_false_for_dead_with_null_error():
    """record_terminal_outcome's terminal_error is `str | None = None`, so a
    dead-letter carrying no error string is reachable. It is NOT assumed to
    have landed — and the isinstance guard means it does not raise."""
    assert _landed({'terminal_status': 'dead', 'terminal_error': None}) is False


# ── derive_stage_stats: dead-lettered writes must not count as landed ────


@pytest.mark.asyncio
async def test_derive_stage_stats_dead_graphiti_enqueue_not_counted(journal):
    """THE HEADLINE CASE: a graphiti-only enqueue that the durable queue later
    dead-lettered never landed, so graphiti_writes_queued must be 0.

    Before this fix, `success=1` (the enqueue was ACCEPTED) was the only gate,
    so an operation with a 0% landing rate reported as fully green.
    """
    run_id = str(uuid.uuid4())
    op_id = await _log_write(
        journal, causation_id=run_id, operation='add_memory',
        result_summary={'memory_ids': [], 'stores': ['graphiti']},
    )
    await _stamp_terminal(journal, op_id, status='dead', error='backend refused')

    ops = await journal.get_ops_by_causation(run_id)
    observed = derive_stage_stats(ops, _STAGE_AGENT_ID)

    assert observed == _expected()


@pytest.mark.asyncio
async def test_derive_stage_stats_dead_add_episode_not_counted(journal):
    """The generic-counter branch also inherits the landed gate."""
    run_id = str(uuid.uuid4())
    op_id = await _log_write(
        journal, causation_id=run_id, operation='add_episode',
        result_summary={'status': 'added'},
    )
    await _stamp_terminal(journal, op_id, status='dead', error='boom')

    ops = await journal.get_ops_by_causation(run_id)
    observed = derive_stage_stats(ops, _STAGE_AGENT_ID)

    assert observed == _expected()


@pytest.mark.asyncio
async def test_derive_stage_stats_dead_add_memory_with_ids_not_counted(journal):
    """The _count_add_memory branch also inherits the landed gate."""
    run_id = str(uuid.uuid4())
    op_id = await _log_write(
        journal, causation_id=run_id, operation='add_memory',
        result_summary={'memory_ids': ['m1'], 'stores': ['mem0']},
    )
    await _stamp_terminal(journal, op_id, status='dead', error='boom')

    ops = await journal.get_ops_by_causation(run_id)
    observed = derive_stage_stats(ops, _STAGE_AGENT_ID)

    assert observed == _expected()


@pytest.mark.asyncio
async def test_derive_stage_stats_post_execute_dead_graphiti_enqueue_still_counts(journal):
    """A post-execute dead-letter DID land: the backend write succeeded and
    only the callback kept failing. It must still count toward its landed
    counter, or the fix trades over-counting for under-counting."""
    run_id = str(uuid.uuid4())
    op_id = await _log_write(
        journal, causation_id=run_id, operation='add_memory',
        result_summary={'memory_ids': [], 'stores': ['graphiti']},
    )
    await _stamp_terminal(
        journal, op_id, status='dead',
        error=POST_EXECUTE_DEAD_PREFIX + 'callback blew up',
    )

    ops = await journal.get_ops_by_causation(run_id)
    observed = derive_stage_stats(ops, _STAGE_AGENT_ID)

    assert observed['graphiti_writes_queued'] == 1


@pytest.mark.asyncio
async def test_derive_stage_stats_post_execute_dead_episode_still_counts(journal):
    """The generic-counter branch honours the post-execute exception too."""
    run_id = str(uuid.uuid4())
    op_id = await _log_write(
        journal, causation_id=run_id, operation='add_episode',
        result_summary={'status': 'added'},
    )
    await _stamp_terminal(
        journal, op_id, status='dead',
        error=POST_EXECUTE_DEAD_PREFIX + 'completion commit failed',
    )

    ops = await journal.get_ops_by_causation(run_id)
    observed = derive_stage_stats(ops, _STAGE_AGENT_ID)

    assert observed['episodes_added'] == 1


# ── writes_dead_lettered ────────────────────────────────────────────────


def test_writes_dead_lettered_is_a_canonical_computed_key():
    """Exclusion alone makes the numbers truthful but SILENT: a stage whose
    every write died reports graphiti_writes_queued: 0, indistinguishable from
    a stage that wrote nothing. This counter is what makes it diagnosable, and
    membership in the frozenset is what propagates it to the judge."""
    assert 'writes_dead_lettered' in _COMPUTED_STAT_KEYS


@pytest.mark.asyncio
async def test_writes_dead_lettered_present_at_zero_with_no_ops(journal):
    run_id = str(uuid.uuid4())

    ops = await journal.get_ops_by_causation(run_id)
    observed = derive_stage_stats(ops, _STAGE_AGENT_ID)

    assert observed['writes_dead_lettered'] == 0


@pytest.mark.asyncio
async def test_dead_graphiti_enqueue_counts_as_dead_lettered_not_queued(journal):
    """The pair that makes the meaning change legible rather than silent:
    graphiti_writes_queued == 0 AND writes_dead_lettered == 1."""
    run_id = str(uuid.uuid4())
    op_id = await _log_write(
        journal, causation_id=run_id, operation='add_memory',
        result_summary={'memory_ids': [], 'stores': ['graphiti']},
    )
    await _stamp_terminal(journal, op_id, status='dead', error='backend refused')

    ops = await journal.get_ops_by_causation(run_id)
    observed = derive_stage_stats(ops, _STAGE_AGENT_ID)

    assert observed == _expected(writes_dead_lettered=1)


@pytest.mark.asyncio
async def test_post_execute_dead_counts_in_both_keys(journal):
    """The DELIBERATE overlap: a post-execute dead-letter both landed (the
    backend write succeeded) and dead-lettered (it still needs operator
    attention and must not be blind-replayed). The two keys answer different
    questions, and this op is honestly BOTH."""
    run_id = str(uuid.uuid4())
    op_id = await _log_write(
        journal, causation_id=run_id, operation='add_memory',
        result_summary={'memory_ids': [], 'stores': ['graphiti']},
    )
    await _stamp_terminal(
        journal, op_id, status='dead',
        error=POST_EXECUTE_DEAD_PREFIX + 'callback blew up',
    )

    ops = await journal.get_ops_by_causation(run_id)
    observed = derive_stage_stats(ops, _STAGE_AGENT_ID)

    assert observed == _expected(graphiti_writes_queued=1, writes_dead_lettered=1)


@pytest.mark.asyncio
async def test_writes_dead_lettered_zero_for_completed_and_null_ops(journal):
    """Only a dead-letter counts — a confirmed-landed write and an unstamped
    one (NULL) must not inflate the counter."""
    run_id = str(uuid.uuid4())

    completed_id = await _log_write(
        journal, causation_id=run_id, operation='add_memory',
        result_summary={'memory_ids': ['m1'], 'stores': ['mem0']},
    )
    await _stamp_terminal(journal, completed_id, status='completed')
    await _log_write(
        journal, causation_id=run_id, operation='add_episode',
        result_summary={'status': 'added'},
    )

    ops = await journal.get_ops_by_causation(run_id)
    observed = derive_stage_stats(ops, _STAGE_AGENT_ID)

    assert observed == _expected(memories_added=1, episodes_added=1)


@pytest.mark.asyncio
async def test_writes_dead_lettered_respects_stage_scoping(journal):
    """A dead op belonging to a sibling stage must not be tallied here —
    the counter sits behind the same agent_id filter as every other."""
    run_id = str(uuid.uuid4())
    op_id = await _log_write(
        journal, causation_id=run_id, operation='add_memory',
        agent_id=_OTHER_STAGE_AGENT_ID,
        result_summary={'memory_ids': [], 'stores': ['graphiti']},
    )
    await _stamp_terminal(journal, op_id, status='dead', error='boom')

    ops = await journal.get_ops_by_causation(run_id)
    observed = derive_stage_stats(ops, _STAGE_AGENT_ID)

    assert observed == _expected()


@pytest.mark.asyncio
async def test_writes_dead_lettered_scoped_to_mapped_operations(journal):
    """A dead op whose operation is not in _OP_TO_STAT is not counted — the
    counter stays scoped to operations this module actually knows about."""
    run_id = str(uuid.uuid4())
    assert 'get_status' not in _OP_TO_STAT
    op_id = await _log_write(
        journal, causation_id=run_id, operation='get_status',
        result_summary={'status': 'ok'},
    )
    await _stamp_terminal(journal, op_id, status='dead', error='boom')

    ops = await journal.get_ops_by_causation(run_id)
    observed = derive_stage_stats(ops, _STAGE_AGENT_ID)

    assert observed == _expected()


@pytest.mark.asyncio
async def test_derive_stage_stats_completed_ops_still_count(journal):
    """NON-REGRESSION: terminal_status='completed' is the durable queue
    confirming the write landed — all three shapes count normally."""
    run_id = str(uuid.uuid4())

    graphiti_id = await _log_write(
        journal, causation_id=run_id, operation='add_memory',
        result_summary={'memory_ids': [], 'stores': ['graphiti']},
    )
    episode_id = await _log_write(
        journal, causation_id=run_id, operation='add_episode',
        result_summary={'status': 'added'},
    )
    memory_id = await _log_write(
        journal, causation_id=run_id, operation='add_memory',
        result_summary={'memory_ids': ['m1'], 'stores': ['mem0']},
    )
    for op_id in (graphiti_id, episode_id, memory_id):
        await _stamp_terminal(journal, op_id, status='completed')

    ops = await journal.get_ops_by_causation(run_id)
    observed = derive_stage_stats(ops, _STAGE_AGENT_ID)

    assert observed == _expected(
        graphiti_writes_queued=1, episodes_added=1, memories_added=1,
    )


@pytest.mark.asyncio
async def test_derive_stage_stats_unstamped_ops_still_count(journal):
    """NON-REGRESSION: terminal_status IS NULL — the state of every pre-3582
    row and of every operation that never touches the durable queue (e.g.
    delete_memory, update_edge). These must keep counting exactly as today."""
    run_id = str(uuid.uuid4())

    await _log_write(
        journal, causation_id=run_id, operation='add_memory',
        result_summary={'memory_ids': [], 'stores': ['graphiti']},
    )
    await _log_write(
        journal, causation_id=run_id, operation='add_episode',
        result_summary={'status': 'added'},
    )
    await _log_write(
        journal, causation_id=run_id, operation='add_memory',
        result_summary={'memory_ids': ['m1'], 'stores': ['mem0']},
    )

    ops = await journal.get_ops_by_causation(run_id)
    observed = derive_stage_stats(ops, _STAGE_AGENT_ID)

    assert observed == _expected(
        graphiti_writes_queued=1, episodes_added=1, memories_added=1,
    )


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
