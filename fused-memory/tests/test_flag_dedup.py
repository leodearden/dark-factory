"""Tests for fused_memory.reconciliation.flag_dedup module.

Tests cover compute_flag_signature, dedup_flags, and error-handling behavior.
"""
from __future__ import annotations

import uuid as _uuid_mod
from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.models.memory import AddMemoryResponse
from fused_memory.reconciliation.flag_dedup import build_suppression_payload

_STUB_ADD_MEMORY_RESPONSE = AddMemoryResponse(memory_ids=['stub-id'])

# ---------------------------------------------------------------------------
# compute_flag_signature tests (step-1)
# ---------------------------------------------------------------------------


class TestComputeFlagSignature:
    """Tests for compute_flag_signature(flag) -> tuple[str, str] | None."""

    def test_returns_tuple_with_both_fields_int_task_id(self):
        """Returns ('123', 'missing_deliverable') when task_id is int and flag_type is str."""
        from fused_memory.reconciliation.flag_dedup import compute_flag_signature

        flag = {'task_id': 123, 'flag_type': 'missing_deliverable'}
        result = compute_flag_signature(flag)
        assert result == ('123', 'missing_deliverable')

    def test_returns_tuple_with_both_fields_str_task_id(self):
        """Returns ('123', 'missing_deliverable') when task_id is str."""
        from fused_memory.reconciliation.flag_dedup import compute_flag_signature

        flag = {'task_id': '123', 'flag_type': 'missing_deliverable'}
        result = compute_flag_signature(flag)
        assert result == ('123', 'missing_deliverable')

    def test_returns_none_when_task_id_missing(self):
        """Returns None when task_id is absent."""
        from fused_memory.reconciliation.flag_dedup import compute_flag_signature

        flag = {'flag_type': 'missing_deliverable'}
        assert compute_flag_signature(flag) is None

    def test_returns_none_when_flag_type_missing(self):
        """Returns None when flag_type is absent."""
        from fused_memory.reconciliation.flag_dedup import compute_flag_signature

        flag = {'task_id': '42'}
        assert compute_flag_signature(flag) is None

    def test_returns_none_when_both_missing(self):
        """Returns None when both task_id and flag_type are absent."""
        from fused_memory.reconciliation.flag_dedup import compute_flag_signature

        flag = {'description': 'some flag'}
        assert compute_flag_signature(flag) is None

    def test_returns_tuple_when_task_id_is_zero(self):
        """task_id=0 is a falsy-but-valid value; should produce a signature, not None."""
        from fused_memory.reconciliation.flag_dedup import compute_flag_signature

        flag = {'task_id': 0, 'flag_type': 'missing_deliverable'}
        result = compute_flag_signature(flag)
        assert result == ('0', 'missing_deliverable'), (
            'task_id=0 must not be silently discarded by a falsy check'
        )


# ---------------------------------------------------------------------------
# dedup_flags tests — no-signature path (step-3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dedup_flags_no_signature_flags_pass_through_unchanged():
    """Flags without task_id/flag_type pass through with exactly one I/O call (suppression filter); add_memory never called."""
    from fused_memory.reconciliation.flag_dedup import dedup_flags

    memory_service = AsyncMock()
    flags = [
        {'description': 'some flag without task_id'},
        {'description': 'another flag without flag_type', 'task_id': '42'},
        {'description': 'flag without task_id', 'flag_type': 'missing_deliverable'},
        {},
    ]
    original_flags = [dict(f) for f in flags]

    result = await dedup_flags(
        memory_service=memory_service,
        project_id='p',
        run_id='r1',
        flags=flags,
    )

    # All flags returned unchanged
    assert result == original_flags
    # filter_suppressed issues exactly one project-scoped suppression search;
    # no per-flag searches because no flags have computable signatures.
    assert memory_service.search.call_count == 1
    # add_memory never called — no-signature flags never reach the marker write path
    memory_service.add_memory.assert_not_called()


# ---------------------------------------------------------------------------
# dedup_flags — prior marker found path (step-5)
# ---------------------------------------------------------------------------


def _make_memory_result(metadata: dict | None) -> MagicMock:
    """Build a minimal mock MemoryResult; metadata may be None to model a malformed memory record."""
    r = MagicMock()
    r.metadata = metadata
    r.content = 'Stage 1 flag marker'
    return r


def _assert_valid_stage1_marker(
    call_kwargs: dict,
    *,
    task_id: str,
    flag_type: str,
    run_id: str,
) -> None:
    """Assert that an add_memory call_kwargs encodes a well-formed stage1_flag_marker.

    Centralises the marker-shape contract so that a schema change only needs to
    be updated here rather than in every test that writes or inspects a marker.
    """
    assert call_kwargs.get('category') == 'observations_and_summaries'
    meta = call_kwargs.get('metadata', {})
    assert meta.get('source') == 'stage1_flag_marker'
    assert meta.get('task_id') == task_id
    assert meta.get('flag_type') == flag_type
    assert meta.get('run_id') == run_id
    assert meta.get('last_seen_run_id') == run_id


@pytest.mark.asyncio
async def test_dedup_flags_prior_marker_found_annotates_flag_no_write():
    """When a prior stage1_flag_marker exists the flag gets persisted_from_run/last_seen_run_id,
    a replacement marker is written, and the prior is deleted (atomic-replacement contract).

    Updated for task-1146: the old \"no write on HIT\" contract is replaced by
    atomic-replacement (write new marker → delete prior).
    """
    from fused_memory.reconciliation.flag_dedup import dedup_flags

    prior_marker = _make_memory_result({
        'source': 'stage1_flag_marker',
        'task_id': '42',
        'flag_type': 'missing_deliverable',
        'run_id': 'r0',
        'last_seen_run_id': 'r0',
    })
    prior_marker.id = 'prior-42'

    memory_service = AsyncMock()
    # task-1400: supply side_effect list for the post-write confirmation search.
    # [suppression=[], pre-write HIT=[prior], confirmation=[prior]] — confirmation
    # finds the prior (same metadata) → confirmed_id = prior_marker.id → delete proceeds.
    memory_service.search = AsyncMock(side_effect=[[], [prior_marker], [prior_marker]])
    memory_service.add_memory = AsyncMock(return_value=_STUB_ADD_MEMORY_RESPONSE)
    memory_service.delete_memory = AsyncMock(return_value=None)

    flags = [{'task_id': 42, 'flag_type': 'missing_deliverable', 'description': 'foo'}]

    result = await dedup_flags(
        memory_service=memory_service,
        project_id='p',
        run_id='r1',
        flags=flags,
    )

    # (a) Flag is annotated with persisted_from_run and last_seen_run_id
    assert len(result) == 1
    assert result[0]['persisted_from_run'] == 'r0'
    assert result[0]['last_seen_run_id'] == 'r1'

    # (b) search was called 3 times: suppression filter + per-flag prior-marker + confirmation.
    #     call_args refers to the LAST call (confirmation search), which must mention
    #     task_id+flag_type and use project_id='p'.
    assert memory_service.search.call_count == 3  # 1 suppression + 1 per-flag + 1 confirmation
    # project_id must be passed as a kwarg (production code uses kwargs throughout)
    assert memory_service.search.call_args.kwargs['project_id'] == 'p'
    # query must strictly mention both the task_id and the flag_type (no permissive 'or')
    query = memory_service.search.call_args.kwargs.get('query', '')
    assert '42' in query and 'missing_deliverable' in query

    # (c) Replacement marker written and prior deleted (atomic-replacement)
    memory_service.add_memory.assert_called_once()
    memory_service.delete_memory.assert_called_once()


@pytest.mark.asyncio
async def test_dedup_flags_metadata_predicate_filters_non_matching_results():
    """When Mem0 search returns rows matching task_id but with wrong source or wrong
    flag_type, the metadata predicate filters them all out, so the flag is treated as
    fresh: not annotated, and a new stage1_flag_marker is written.

    Regression coverage for flag_dedup.py:77-86 — without this test, dropping the
    source/flag_type guards from the kind dict passed to find_prior_memory would
    silently start treating cross-source rows as prior markers.
    """
    from fused_memory.reconciliation.flag_dedup import dedup_flags

    # Three rows whose task_id matches but whose source and/or flag_type do not.
    # They exercise both clauses of the kind conjunction independently:
    wrong_source = _make_memory_result({
        'source': 'targeted_reconciliation',  # wrong source
        'task_id': '42',
        'flag_type': 'missing_deliverable',
    })
    wrong_flag_type = _make_memory_result({
        'source': 'stage1_flag_marker',
        'task_id': '42',
        'flag_type': 'stale_metadata',  # wrong flag_type
    })
    both_wrong = _make_memory_result({
        'source': 'other',  # wrong source
        'task_id': '42',
        'flag_type': 'unrelated',  # wrong flag_type
    })

    memory_service = AsyncMock()
    # task-1400: supply side_effect list to satisfy the post-write confirmation search.
    # The wrong-metadata rows are not stage1_flag_markers, so confirmation also misses.
    # suppression filter, per-flag pre-write, confirmation first, confirmation retry:
    memory_service.search = AsyncMock(side_effect=[
        [wrong_source, wrong_flag_type, both_wrong],  # suppression filter (kind mismatch → no suppression)
        [wrong_source, wrong_flag_type, both_wrong],  # per-flag search (all filtered → MISS)
        [],  # confirmation search (miss — no stage1_flag_marker match)
        [],  # confirmation retry (miss)
    ])
    memory_service.add_memory = AsyncMock(return_value=_STUB_ADD_MEMORY_RESPONSE)

    flags = [{'task_id': 42, 'flag_type': 'missing_deliverable', 'description': 'foo'}]

    result = await dedup_flags(
        memory_service=memory_service,
        project_id='p',
        run_id='r1',
        flags=flags,
    )

    # Predicate rejected all rows — flag NOT annotated as prior-seen
    assert len(result) == 1
    assert 'persisted_from_run' not in result[0], (
        'Predicate should have filtered all non-matching rows; flag must not be annotated'
    )

    # No-prior-marker write path exercised: a new marker is written
    memory_service.add_memory.assert_called_once()
    _assert_valid_stage1_marker(
        memory_service.add_memory.call_args.kwargs,
        task_id='42', flag_type='missing_deliverable', run_id='r1',
    )

    # Four search calls: 1 suppression + 1 per-flag + 2 confirmation (miss+retry).
    assert memory_service.search.call_count == 4


# ---------------------------------------------------------------------------
# dedup_flags — no prior marker (fresh flag) path (step-7)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dedup_flags_no_prior_marker_writes_new_marker():
    """When no prior stage1_flag_marker exists, the flag is not annotated and a new marker
    is written to Mem0.
    """
    from fused_memory.reconciliation.flag_dedup import dedup_flags

    memory_service = AsyncMock()
    memory_service.search = AsyncMock(return_value=[])  # empty — no prior marker
    memory_service.add_memory = AsyncMock(return_value=_STUB_ADD_MEMORY_RESPONSE)

    flags = [{'task_id': '99', 'flag_type': 'stale_metadata', 'description': 'bar'}]

    result = await dedup_flags(
        memory_service=memory_service,
        project_id='p',
        run_id='r1',
        flags=flags,
    )

    # (a) Flag has NO persisted_from_run field — it's a fresh finding
    assert len(result) == 1
    assert 'persisted_from_run' not in result[0]

    # (b) add_memory called exactly once with the expected marker metadata
    memory_service.add_memory.assert_called_once()
    _assert_valid_stage1_marker(
        memory_service.add_memory.call_args.kwargs,
        task_id='99', flag_type='stale_metadata', run_id='r1',
    )


# ---------------------------------------------------------------------------
# dedup_flags — exception handling (step-9)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dedup_flags_search_exception_does_not_raise_and_warns(caplog):
    """When memory_service.search raises, dedup_flags does not raise, returns flags unchanged,
    and logs a WARNING.

    Also pins the post-refactor marker-growth behavior: when search fails,
    find_prior_memory swallows the exception and returns None, so dedup_flags
    falls into the else-branch and writes one new marker per flag (was: zero in
    the prior wrap-both pattern where both search and add_memory shared a single
    try/except).  See the marker-growth caveat comment in the else-branch of
    dedup_flags for details on monotonic marker growth during a sustained outage.

    Two flags are passed so the 'one write per flag' contract is verified, not
    merely 'exactly one write ever'.  The assertions (d) and (e) below lock down
    this contract so a future refactor cannot silently flip the count in either
    direction (zero if the wrap-both pattern is restored, or >N if an extra
    refresh write is added per flag).
    """
    import logging

    from fused_memory.reconciliation.flag_dedup import dedup_flags

    memory_service = AsyncMock()
    memory_service.search = AsyncMock(side_effect=RuntimeError('Mem0 down'))
    memory_service.add_memory = AsyncMock(return_value=_STUB_ADD_MEMORY_RESPONSE)

    flags = [
        {'task_id': '55', 'flag_type': 'stale_metadata', 'description': 'test'},
        {'task_id': '66', 'flag_type': 'missing_deliverable', 'description': 'test2'},
    ]

    with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
        result = await dedup_flags(
            memory_service=memory_service,
            project_id='p',
            run_id='r1',
            flags=flags,
        )

    # (a) Does NOT raise
    # (b) Returns both flags unchanged (no persisted_from_run)
    assert len(result) == 2
    assert 'persisted_from_run' not in result[0]
    assert 'persisted_from_run' not in result[1]
    # (c) WARNING log mentions the failure for both task_ids
    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any('55' in m for m in warning_messages)
    assert any('66' in m for m in warning_messages)
    # (d) Exactly one marker write per flag when search fails — verifies the 'per flag'
    #     contract (two flags → two writes).  Catches: zero writes (wrap-both try/except
    #     restored) or multiple writes per flag (extra refresh write added).
    assert memory_service.add_memory.call_count == 2
    # (e) Each marker is correctly shaped on the failure path.
    _assert_valid_stage1_marker(
        memory_service.add_memory.call_args_list[0].kwargs,
        task_id='55', flag_type='stale_metadata', run_id='r1',
    )
    _assert_valid_stage1_marker(
        memory_service.add_memory.call_args_list[1].kwargs,
        task_id='66', flag_type='missing_deliverable', run_id='r1',
    )


# ---------------------------------------------------------------------------
# dedup_flags — malformed prior-marker run_id uses sentinel (step-1)
# ---------------------------------------------------------------------------

_VALID_FILTER_META = {
    'source': 'stage1_flag_marker',
    'task_id': '42',
    'flag_type': 'missing_deliverable',
}


@pytest.mark.parametrize(
    'prior_metadata',
    [
        # (a) 'run_id' key absent — .get('run_id', run_id) silently returns run_id, not 'unknown'
        pytest.param(
            {**_VALID_FILTER_META},
            id='run_id-key-absent',
        ),
        # (c) 'run_id' key present but value is None — .get returns None (not 'unknown')
        pytest.param(
            {**_VALID_FILTER_META, 'run_id': None},
            id='run_id-is-None',
        ),
        # (d) 'run_id' key present but value is '' — .get returns '' (not 'unknown')
        pytest.param(
            {**_VALID_FILTER_META, 'run_id': ''},
            id='run_id-is-empty-string',
        ),
    ],
)
@pytest.mark.asyncio
async def test_dedup_flags_prior_marker_with_malformed_run_id_uses_sentinel(
    prior_metadata, caplog
):
    """When a prior stage1_flag_marker exists but has a missing/falsy run_id, dedup_flags
    must annotate the flag with persisted_from_run='unknown' — not the current run_id.

    Three malformed shapes parametrised:
    (a) 'run_id' key absent        → .get(key, run_id) silently returns 'r1' ≠ 'unknown'
    (c) run_id=None                → .get returns None ≠ 'unknown'
    (d) run_id=''                  → .get returns '' ≠ 'unknown'
    All three must produce persisted_from_run='unknown' with the sentinel fix applied.

    Also asserts that the sentinel-collapse path emits a DEBUG log for each malformed
    shape — this folds in what was previously a standalone test, giving observability
    coverage across all three shapes in a single parametrized suite.

    Note: the case where prior.metadata is None is intentionally omitted.  When
    metadata is None, the candidate filter ((r.metadata or {}).get('source') etc.)
    returns falsy values for all three checks, so the candidate is never selected
    as ``prior`` — the code under test (run_id extraction) is never reached.
    """
    import logging

    from fused_memory.reconciliation.flag_dedup import dedup_flags

    prior_marker = _make_memory_result(prior_metadata)
    prior_marker.id = 'prior-malformed'

    memory_service = AsyncMock()
    memory_service.search = AsyncMock(return_value=[prior_marker])
    memory_service.add_memory = AsyncMock(return_value=_STUB_ADD_MEMORY_RESPONSE)
    memory_service.delete_memory = AsyncMock(return_value=None)

    flags = [{'task_id': 42, 'flag_type': 'missing_deliverable', 'description': 'foo'}]

    with caplog.at_level(logging.DEBUG, logger='fused_memory.reconciliation.flag_dedup'):
        result = await dedup_flags(
            memory_service=memory_service,
            project_id='p',
            run_id='r1',
            flags=flags,
        )

    assert len(result) == 1
    assert result[0]['persisted_from_run'] == 'unknown', (
        f"persisted_from_run must fall back to sentinel 'unknown' for any falsy run_id "
        f"in prior marker metadata, but got {result[0].get('persisted_from_run')!r}."
    )
    assert result[0]['last_seen_run_id'] == 'r1'
    # Atomic-replacement: replacement marker is written and prior is deleted even
    # when run_id was malformed (annotation sentinel does not suppress the write).
    memory_service.add_memory.assert_called_once()
    memory_service.delete_memory.assert_called_once()

    # Sentinel-collapse path emits a DEBUG log — covers all three malformed shapes.
    # Loose enough to tolerate minor wording changes; strict enough to lock in the
    # observability intent (dashboards can grep for 'unknown'/'malformed').
    assert any(
        '42' in record.message
        and 'missing_deliverable' in record.message
        and ('unknown' in record.message or 'malformed' in record.message)
        and record.levelno == logging.DEBUG
        for record in caplog.records
    ), (
        f"Expected a DEBUG log mentioning task_id='42', flag_type='missing_deliverable', "
        f"and 'unknown'/'malformed', but got records: {[r.message for r in caplog.records]}"
    )


@pytest.mark.asyncio
async def test_dedup_flags_prior_marker_None_metadata_writes_new_marker():
    """Locks in the candidate-filter rejection behavior — the `meta = r.metadata or {}`
    guard in find_prior_memory short-circuits when metadata is None, so a refactor that
    drops the `or {}` guard (e.g., direct attribute access on r.metadata) would silently
    regress this path. Test passes against current impl. Companion to
    test_dedup_flags_prior_marker_with_malformed_run_id_uses_sentinel, which intentionally
    omits this case (see its docstring).
    """
    from fused_memory.reconciliation.flag_dedup import dedup_flags

    # Prior result with metadata=None — survives the search but is rejected by the
    # candidate filter because (r.metadata or {}).get('task_id', '') is empty and
    # does not match '42'; the task_id check runs first so the source/flag_type
    # checks are never evaluated.
    prior_result = _make_memory_result(None)

    memory_service = AsyncMock()
    memory_service.search = AsyncMock(return_value=[prior_result])
    memory_service.add_memory = AsyncMock(return_value=_STUB_ADD_MEMORY_RESPONSE)

    result = await dedup_flags(
        memory_service=memory_service,
        project_id='p',
        run_id='r1',
        flags=[{'task_id': 42, 'flag_type': 'missing_deliverable', 'description': 'foo'}],
    )

    # Candidate was filtered out — flag is NOT annotated with persisted_from_run
    assert len(result) == 1
    assert 'persisted_from_run' not in result[0], (
        f"Expected no persisted_from_run annotation but got {result[0].get('persisted_from_run')!r}"
    )

    # The dedup logic took the else-branch (novel flag) and wrote a new marker
    memory_service.add_memory.assert_called_once()

    # New marker has a well-formed stage1_flag_marker shape
    _assert_valid_stage1_marker(
        memory_service.add_memory.call_args.kwargs,
        task_id='42',
        flag_type='missing_deliverable',
        run_id='r1',
    )


# ---------------------------------------------------------------------------
# Step 3 (task-1146) — best-effort replace: HIT writes new marker then deletes prior
# NOTE: test names below use the historical 'atomic_replace' prefix; the contract
# was renamed 'best-effort replacement' in task-1165 (concurrent duplicates self-heal).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dedup_flags_prior_marker_atomic_replace_writes_new_and_deletes_prior():
    """On HIT, dedup_flags writes a new marker and deletes the prior, in that order.

    Pins the atomic-replacement contract introduced by task-1146:
    (a) flag is annotated with persisted_from_run='r0' and last_seen_run_id='r1'
    (b) add_memory called exactly once with a well-formed stage1_flag_marker for run_id='r1'
    (c) delete_memory called exactly once with memory_id='prior-123', store='mem0'
    (d) ordering invariant: add_memory call precedes delete_memory call in mock_calls
    """
    from fused_memory.reconciliation.flag_dedup import dedup_flags

    prior_marker = _make_memory_result({
        'source': 'stage1_flag_marker',
        'task_id': '42',
        'flag_type': 'missing_deliverable',
        'run_id': 'r0',
        'last_seen_run_id': 'r0',
    })
    prior_marker.id = 'prior-123'

    memory_service = AsyncMock()
    memory_service.search = AsyncMock(return_value=[prior_marker])
    memory_service.add_memory = AsyncMock(return_value=_STUB_ADD_MEMORY_RESPONSE)
    memory_service.delete_memory = AsyncMock(return_value=None)

    flags = [{'task_id': 42, 'flag_type': 'missing_deliverable', 'description': 'foo'}]

    result = await dedup_flags(
        memory_service=memory_service,
        project_id='p',
        run_id='r1',
        flags=flags,
    )

    # (a) annotation
    assert len(result) == 1
    assert result[0]['persisted_from_run'] == 'r0'
    assert result[0]['last_seen_run_id'] == 'r1'

    # (b) add_memory called once with valid marker for run_id='r1'
    memory_service.add_memory.assert_called_once()
    _assert_valid_stage1_marker(
        memory_service.add_memory.call_args.kwargs,
        task_id='42', flag_type='missing_deliverable', run_id='r1',
    )

    # (c) delete_memory called once for the prior
    memory_service.delete_memory.assert_called_once()
    del_kwargs = memory_service.delete_memory.call_args.kwargs
    assert del_kwargs.get('memory_id') == 'prior-123'
    assert del_kwargs.get('store') == 'mem0'
    assert del_kwargs.get('project_id') == 'p'

    # (d) ordering: add_memory before delete_memory.
    # Use method_calls (records only actual method invocations, no attribute
    # access) with exact name matching via call[0] so the check is immune to
    # future child-mock interactions whose repr happens to contain 'add_memory'.
    method_names = [c[0] for c in memory_service.method_calls]
    add_idx = method_names.index('add_memory')
    del_idx = method_names.index('delete_memory')
    assert add_idx < del_idx, (
        f'add_memory (idx {add_idx}) must precede delete_memory (idx {del_idx})'
    )


# ---------------------------------------------------------------------------
# Step 5 (task-1146) — atomic-replace collapses multiple predecessors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dedup_flags_atomic_replace_handles_multiple_predecessors():
    """On HIT with multiple prior markers (past leakage), all priors are deleted.

    Simulates N=3 prior markers for the same (task_id, flag_type) that exist
    due to prior search-failure or top-N rank-eviction leakage.  dedup_flags
    must write exactly ONE replacement marker and delete all THREE priors.

    (a) add_memory called exactly once
    (b) delete_memory called exactly three times covering ids {p-1, p-2, p-3}
    (c) flag annotation uses the lowest-id-lex prior's run_id
    """
    from fused_memory.reconciliation.flag_dedup import dedup_flags

    def _prior(id_: str, run_id: str) -> MagicMock:
        r = _make_memory_result({
            'source': 'stage1_flag_marker',
            'task_id': '42',
            'flag_type': 'missing_deliverable',
            'run_id': run_id,
            'last_seen_run_id': run_id,
        })
        r.id = id_
        return r

    prior1 = _prior('p-1', 'r0')   # lex-lowest id — annotation comes from this one
    prior2 = _prior('p-2', 'r-prev')
    prior3 = _prior('p-3', 'r-earlier')

    memory_service = AsyncMock()
    # Return [prior2, prior3, prior1] to exercise lex-sort — p-1 is NOT first in this order
    memory_service.search = AsyncMock(return_value=[prior2, prior3, prior1])
    memory_service.add_memory = AsyncMock(return_value=_STUB_ADD_MEMORY_RESPONSE)
    memory_service.delete_memory = AsyncMock(return_value=None)

    flags = [{'task_id': 42, 'flag_type': 'missing_deliverable', 'description': 'foo'}]

    result = await dedup_flags(
        memory_service=memory_service,
        project_id='p',
        run_id='r1',
        flags=flags,
    )

    # (a) exactly one write
    memory_service.add_memory.assert_called_once()

    # (b) exactly three deletes covering all prior ids
    assert memory_service.delete_memory.call_count == 3, (
        f'Expected 3 delete_memory calls but got {memory_service.delete_memory.call_count}'
    )
    deleted_ids = {
        call.kwargs.get('memory_id')
        for call in memory_service.delete_memory.call_args_list
    }
    assert deleted_ids == {'p-1', 'p-2', 'p-3'}, (
        f'Expected all three prior ids deleted but got {deleted_ids}'
    )

    # (c) annotation from lowest-id-lex prior (p-1 → run_id 'r0')
    assert result[0]['persisted_from_run'] == 'r0'


# ---------------------------------------------------------------------------
# Step 7 (task-1146) — write-failure skips delete (predecessor preserved)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dedup_flags_atomic_replace_skips_delete_if_write_fails(caplog):
    """When add_memory raises on the HIT path, delete_memory is never called.

    Pins the write-first ordering guarantee: if the replacement write fails,
    all priors must remain intact so the next cycle still has dedup state.

    (a) dedup_flags does NOT raise
    (b) delete_memory was NEVER called
    (c) flag IS annotated (annotation extracted BEFORE write attempt)
    (d) WARNING log mentions task_id, flag_type, and the failure
    """
    import logging as _logging

    from fused_memory.reconciliation.flag_dedup import dedup_flags

    prior_marker = _make_memory_result({
        'source': 'stage1_flag_marker',
        'task_id': '42',
        'flag_type': 'missing_deliverable',
        'run_id': 'r0',
        'last_seen_run_id': 'r0',
    })
    prior_marker.id = 'prior-write-fail'

    memory_service = AsyncMock()
    memory_service.search = AsyncMock(return_value=[prior_marker])
    memory_service.add_memory = AsyncMock(side_effect=RuntimeError('write failed'))
    memory_service.delete_memory = AsyncMock(return_value=None)

    flags = [{'task_id': 42, 'flag_type': 'missing_deliverable', 'description': 'foo'}]

    with caplog.at_level(_logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
        result = await dedup_flags(
            memory_service=memory_service,
            project_id='p',
            run_id='r1',
            flags=flags,
        )

    # (a) does not raise
    # (b) delete_memory never called — prior preserved
    memory_service.delete_memory.assert_not_called()

    # (c) flag IS annotated (annotation extracted before write attempt)
    assert len(result) == 1
    assert result[0].get('persisted_from_run') == 'r0'
    assert result[0].get('last_seen_run_id') == 'r1'

    # (d) WARNING log mentions task_id and flag_type
    warning_messages = [r.message for r in caplog.records if r.levelno >= _logging.WARNING]
    assert any('42' in m for m in warning_messages), (
        f'Expected WARNING mentioning task 42 but got: {warning_messages}'
    )
    assert any('missing_deliverable' in m for m in warning_messages), (
        f'Expected WARNING mentioning flag_type but got: {warning_messages}'
    )


# ---------------------------------------------------------------------------
# Step 9 (task-1146) — per-prior delete failure logs warning and continues
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dedup_flags_atomic_replace_per_prior_delete_failure_logs_warning_and_continues(caplog):
    """One failing delete does not abort the batch; both priors get a delete attempt.

    Configures two priors ('p-1', 'p-2'); delete_memory raises for 'p-1' but
    succeeds for 'p-2'.  Pins the contract that a per-prior delete error:
    (a) does NOT cause dedup_flags to raise
    (b) add_memory is called exactly once (write succeeded)
    (c) delete_memory is called exactly twice — both priors attempted
    (d) WARNING log mentions 'p-1' and the failure
    (e) flag is annotated normally
    """
    import logging as _logging

    from fused_memory.reconciliation.flag_dedup import dedup_flags

    def _prior(id_: str, run_id: str) -> MagicMock:
        r = _make_memory_result({
            'source': 'stage1_flag_marker',
            'task_id': '42',
            'flag_type': 'missing_deliverable',
            'run_id': run_id,
            'last_seen_run_id': run_id,
        })
        r.id = id_
        return r

    prior1 = _prior('p-1', 'r0')
    prior2 = _prior('p-2', 'r-prev')

    def _delete_side_effect(**kwargs):
        if kwargs.get('memory_id') == 'p-1':
            raise RuntimeError('delete p-1 failed')
        # p-2 succeeds — return None (AsyncMock awaitable returns None by default)

    memory_service = AsyncMock()
    memory_service.search = AsyncMock(return_value=[prior1, prior2])
    memory_service.add_memory = AsyncMock(return_value=_STUB_ADD_MEMORY_RESPONSE)
    memory_service.delete_memory = AsyncMock(side_effect=_delete_side_effect)

    flags = [{'task_id': 42, 'flag_type': 'missing_deliverable', 'description': 'foo'}]

    with caplog.at_level(_logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
        result = await dedup_flags(
            memory_service=memory_service,
            project_id='p',
            run_id='r1',
            flags=flags,
        )

    # (a) does not raise
    # (b) add_memory called once
    memory_service.add_memory.assert_called_once()

    # (c) delete_memory called twice — both priors attempted
    assert memory_service.delete_memory.call_count == 2, (
        f'Expected 2 delete calls but got {memory_service.delete_memory.call_count}'
    )

    # (d) WARNING log mentions 'p-1'
    warning_messages = [r.message for r in caplog.records if r.levelno >= _logging.WARNING]
    assert any('p-1' in m for m in warning_messages), (
        f'Expected WARNING mentioning p-1 but got: {warning_messages}'
    )

    # (e) flag annotated normally
    assert len(result) == 1
    assert result[0].get('persisted_from_run') == 'r0'
    assert result[0].get('last_seen_run_id') == 'r1'


@pytest.mark.asyncio
async def test_dedup_flags_add_memory_exception_does_not_raise_and_warns(caplog):
    """When memory_service.add_memory raises, dedup_flags does not raise, returns flag unchanged,
    and logs a WARNING.
    """
    import logging

    from fused_memory.reconciliation.flag_dedup import dedup_flags

    memory_service = AsyncMock()
    memory_service.search = AsyncMock(return_value=[])  # no prior marker
    memory_service.add_memory = AsyncMock(side_effect=RuntimeError('write failed'))

    flags = [{'task_id': '66', 'flag_type': 'missing_deliverable', 'description': 'test'}]

    with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
        result = await dedup_flags(
            memory_service=memory_service,
            project_id='p',
            run_id='r1',
            flags=flags,
        )

    # (a) Does NOT raise
    # (b) Returns flag unchanged
    assert len(result) == 1
    assert 'persisted_from_run' not in result[0]
    # (c) WARNING log mentions failure and task_id
    assert any(
        '66' in record.message and record.levelno >= logging.WARNING
        for record in caplog.records
    )


# ---------------------------------------------------------------------------
# Step 11 (task-1146) — two-consecutive-runs no-accumulation regression test
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# task-1165 step-5 — HIT path: deterministic prior-marker selection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dedup_flags_hit_prior_selection_is_deterministic_across_search_orders():
    """HIT path: prior selection must be deterministic regardless of search return order.

    Builds three priors with ids 'aaa', 'mmm', 'zzz' and distinct run_ids.
    Runs dedup_flags twice with different search return orders:
    - Run 1: search returns [zzz, aaa, mmm]
    - Run 2: search returns [mmm, zzz, aaa]

    Both runs must:
    - annotate persisted_from_run='r-aaa' (lowest-id-lex prior wins)
    - have the FIRST delete_memory call target memory_id='aaa'

    Today this fails because priors[0] reflects search order.
    """

    def _prior(id_: str, run_id: str) -> MagicMock:
        r = _make_memory_result({
            'source': 'stage1_flag_marker',
            'task_id': '99',
            'flag_type': 'stale_metadata',
            'run_id': run_id,
            'last_seen_run_id': run_id,
        })
        r.id = id_
        return r

    prior_aaa = _prior('aaa', 'r-aaa')
    prior_mmm = _prior('mmm', 'r-mmm')
    prior_zzz = _prior('zzz', 'r-zzz')

    from fused_memory.reconciliation.flag_dedup import dedup_flags

    flags = [{'task_id': '99', 'flag_type': 'stale_metadata', 'description': 'test'}]

    # Run 1: search returns [zzz, aaa, mmm] — aaa is NOT first
    memory_service_1 = AsyncMock()
    memory_service_1.search = AsyncMock(return_value=[prior_zzz, prior_aaa, prior_mmm])
    memory_service_1.add_memory = AsyncMock(return_value=_STUB_ADD_MEMORY_RESPONSE)
    memory_service_1.delete_memory = AsyncMock(return_value=None)

    result_1 = await dedup_flags(
        memory_service=memory_service_1,
        project_id='p',
        run_id='r1',
        flags=flags,
    )

    # Run 2: search returns [mmm, zzz, aaa] — different order
    memory_service_2 = AsyncMock()
    memory_service_2.search = AsyncMock(return_value=[prior_mmm, prior_zzz, prior_aaa])
    memory_service_2.add_memory = AsyncMock(return_value=_STUB_ADD_MEMORY_RESPONSE)
    memory_service_2.delete_memory = AsyncMock(return_value=None)

    result_2 = await dedup_flags(
        memory_service=memory_service_2,
        project_id='p',
        run_id='r2',
        flags=flags,
    )

    # Both runs should annotate with 'r-aaa' (lowest lex id = 'aaa')
    assert result_1[0]['persisted_from_run'] == 'r-aaa', (
        f"Run 1: expected persisted_from_run='r-aaa' but got {result_1[0].get('persisted_from_run')!r}"
    )
    assert result_2[0]['persisted_from_run'] == 'r-aaa', (
        f"Run 2: expected persisted_from_run='r-aaa' but got {result_2[0].get('persisted_from_run')!r}"
    )

    # First delete in each run must target 'aaa'
    first_delete_1 = memory_service_1.delete_memory.call_args_list[0].kwargs.get('memory_id')
    assert first_delete_1 == 'aaa', (
        f"Run 1: expected first delete to target 'aaa' but got {first_delete_1!r}"
    )
    first_delete_2 = memory_service_2.delete_memory.call_args_list[0].kwargs.get('memory_id')
    assert first_delete_2 == 'aaa', (
        f"Run 2: expected first delete to target 'aaa' but got {first_delete_2!r}"
    )


# ---------------------------------------------------------------------------
# task-1165 step-3 — MISS path: respects add_memory response memory_ids
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'add_memory_response,expect_noop_warning',
    [
        pytest.param(
            'empty',  # AddMemoryResponse(memory_ids=[])
            True,
            id='empty-memory_ids-warns',
        ),
        pytest.param(
            'non_empty',  # AddMemoryResponse(memory_ids=['new-marker-id'])
            False,
            id='non-empty-memory_ids-no-warn',
        ),
    ],
)
@pytest.mark.asyncio
async def test_dedup_flags_miss_respects_add_memory_response_memory_ids(
    add_memory_response, expect_noop_warning, caplog
):
    """MISS path: dedup_flags must inspect add_memory's return value.

    When add_memory returns an empty memory_ids list on MISS:
    - a WARNING must be emitted containing task_id and flag_type
      (the flag won't be detectable next cycle)

    When add_memory returns a non-empty memory_ids list on MISS:
    - no no-op WARNING should be emitted

    In both cases:
    - flag returns unannotated (no persisted_from_run)
    - add_memory is called exactly once
    """
    import logging

    from fused_memory.reconciliation.flag_dedup import dedup_flags

    if add_memory_response == 'empty':
        response = AddMemoryResponse(memory_ids=[])
    else:
        response = AddMemoryResponse(memory_ids=['new-marker-id'])

    memory_service = AsyncMock()
    memory_service.search = AsyncMock(return_value=[])  # MISS
    memory_service.add_memory = AsyncMock(return_value=response)

    flags = [{'task_id': '77', 'flag_type': 'stale_metadata', 'description': 'test'}]

    with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
        result = await dedup_flags(
            memory_service=memory_service,
            project_id='p',
            run_id='r1',
            flags=flags,
        )

    # Flag is NOT annotated on MISS path regardless of write outcome
    assert len(result) == 1
    assert 'persisted_from_run' not in result[0]

    # add_memory called exactly once
    memory_service.add_memory.assert_called_once()

    # Check WARNING for no-op case
    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    if expect_noop_warning:
        assert any('77' in m for m in warning_messages), (
            f'Expected WARNING mentioning task_id=77 but got: {warning_messages}'
        )
        assert any('stale_metadata' in m for m in warning_messages), (
            f'Expected WARNING mentioning flag_type but got: {warning_messages}'
        )
    else:
        noop_warnings = [m for m in warning_messages if 'no memory_ids' in m or 'returned no memory_ids' in m]
        assert not noop_warnings, (
            f'Unexpected no-op WARNING on non-empty memory_ids MISS path: {noop_warnings}'
        )


# ---------------------------------------------------------------------------
# task-1165 step-1 — HIT path: respects add_memory response memory_ids
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'add_memory_response,expect_delete,expect_noop_warning',
    [
        pytest.param(
            'empty',  # AddMemoryResponse(memory_ids=[])
            False,
            True,
            id='empty-memory_ids-skips-delete-and-warns',
        ),
        pytest.param(
            'non_empty',  # AddMemoryResponse(memory_ids=['new-marker-id'])
            True,
            False,
            id='non-empty-memory_ids-proceeds-to-delete',
        ),
    ],
)
@pytest.mark.asyncio
async def test_dedup_flags_hit_respects_add_memory_response_memory_ids(
    add_memory_response, expect_delete, expect_noop_warning, caplog
):
    """HIT path: dedup_flags must inspect add_memory's return value.

    When add_memory returns an empty memory_ids list:
    - delete_memory must NOT be called (priors preserved for next cycle)
    - a WARNING must be emitted containing task_id and flag_type

    When add_memory returns a non-empty memory_ids list:
    - delete_memory MUST be called once for the prior
    - no no-op WARNING should be emitted
    """
    import logging

    from fused_memory.reconciliation.flag_dedup import dedup_flags

    prior_marker = _make_memory_result({
        'source': 'stage1_flag_marker',
        'task_id': '42',
        'flag_type': 'missing_deliverable',
        'run_id': 'r0',
        'last_seen_run_id': 'r0',
    })
    prior_marker.id = 'prior-hit-resp-test'

    if add_memory_response == 'empty':
        response = AddMemoryResponse(memory_ids=[])
    else:
        response = AddMemoryResponse(memory_ids=['new-marker-id'])

    # task-1400: switch to side_effect so confirmation search is explicit.
    # 'empty': add_memory is a no-op → confirmation misses (no new findable marker)
    #          → confirmed_id=None → write_succeeded=False → delete skipped.
    # 'non_empty': add_memory wrote a marker → confirmation finds the prior marker
    #              (same metadata still present) → confirmed_id set → delete proceeds.
    if add_memory_response == 'empty':
        search_side_effect = [[], [prior_marker], [], []]  # conf miss + retry miss
    else:
        search_side_effect = [[], [prior_marker], [prior_marker]]  # conf finds marker

    memory_service = AsyncMock()
    memory_service.search = AsyncMock(side_effect=search_side_effect)
    memory_service.add_memory = AsyncMock(return_value=response)
    memory_service.delete_memory = AsyncMock(return_value=None)

    flags = [{'task_id': 42, 'flag_type': 'missing_deliverable', 'description': 'foo'}]

    with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
        result = await dedup_flags(
            memory_service=memory_service,
            project_id='p',
            run_id='r1',
            flags=flags,
        )

    # Flag is still annotated with persisted_from_run regardless of write outcome
    assert len(result) == 1
    assert result[0]['persisted_from_run'] == 'r0'
    assert result[0]['last_seen_run_id'] == 'r1'

    # add_memory always called once
    memory_service.add_memory.assert_called_once()

    if expect_delete:
        # Confirmation succeeded: delete should proceed
        memory_service.delete_memory.assert_called_once()
        del_kwargs = memory_service.delete_memory.call_args.kwargs
        assert del_kwargs.get('memory_id') == 'prior-hit-resp-test'
    else:
        # Confirmation missed (no findable marker): delete must be skipped
        memory_service.delete_memory.assert_not_called()

    # Check WARNING for no-op case (task-1400: WARNING now comes from confirmation failure)
    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    if expect_noop_warning:
        assert any('42' in m for m in warning_messages), (
            f'Expected WARNING mentioning task_id=42 but got: {warning_messages}'
        )
        assert any('missing_deliverable' in m for m in warning_messages), (
            f'Expected WARNING mentioning flag_type but got: {warning_messages}'
        )
    else:
        # No no-op warning expected when confirmation succeeded
        noop_warnings = [m for m in warning_messages if 'no memory_ids' in m or 'returned no memory_ids' in m]
        assert not noop_warnings, (
            f'Unexpected no-op WARNING on confirmed path: {noop_warnings}'
        )


# ---------------------------------------------------------------------------
# TestFilterSuppressed (task-1186 step-1) — filter_suppressed behavior
# ---------------------------------------------------------------------------


class TestFilterSuppressed:
    """Tests for filter_suppressed(memory_service, project_id, flags).

    All tests import filter_suppressed from flag_dedup; they will fail with
    ImportError until the implementation is added in step-2.
    """

    @pytest.mark.asyncio
    async def test_empty_flags_returns_empty_no_io(self):
        """(a) Empty flags input → empty result + zero I/O."""
        from fused_memory.reconciliation.flag_dedup import filter_suppressed

        memory_service = AsyncMock()
        result = await filter_suppressed(memory_service, 'p', [])
        assert result == []
        memory_service.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_search_results_all_flags_pass_through(self):
        """(b) Empty search results → all flags pass through unchanged."""
        from fused_memory.reconciliation.flag_dedup import filter_suppressed

        memory_service = AsyncMock()
        memory_service.search = AsyncMock(return_value=[])
        flags = [
            {'task_id': 42, 'flag_type': 'missing_deliverable'},
            {'task_id': 99, 'flag_type': 'stale_metadata'},
        ]
        result = await filter_suppressed(memory_service, 'p', flags)
        assert result == flags

    @pytest.mark.asyncio
    async def test_search_called_with_canonical_kwargs(self):
        """(c) search called with canonical kwargs (project_id, categories, stores, limit, query).

        limit=501 is used internally (not 500) so that genuine overflow can be
        detected without false positives at the boundary — see filter_suppressed
        docstring for details.
        """
        from fused_memory.reconciliation.flag_dedup import filter_suppressed

        memory_service = AsyncMock()
        memory_service.search = AsyncMock(return_value=[])
        flags = [{'task_id': 1, 'flag_type': 'missing_deliverable'}]
        await filter_suppressed(memory_service, 'p', flags)

        memory_service.search.assert_called_once()
        kwargs = memory_service.search.call_args.kwargs
        assert kwargs.get('project_id') == 'p'
        assert kwargs.get('categories') == ['observations_and_summaries']
        assert kwargs.get('stores') == ['mem0']
        assert kwargs.get('limit') == 501  # sentinel: request one extra to detect overflow
        assert 'stage1_flag_suppression' in kwargs.get('query', '')

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'flag_task_id,record_task_id',
        [
            pytest.param(42, 42, id='int-int'),
            pytest.param(42, '42', id='int-str'),
            pytest.param('42', 42, id='str-int'),
            pytest.param('42', '42', id='str-str'),
        ],
    )
    async def test_suppressed_flag_dropped_and_unrelated_kept(
        self, flag_task_id, record_task_id
    ):
        """(d) suppression record with matching task_id drops the flag; unrelated flags kept.

        Parametrized: int-int, int-str, str-int, str-str coercion combinations
        so that symmetric str() coercion is verified on both sides.
        """
        from fused_memory.reconciliation.flag_dedup import filter_suppressed

        suppression_record = _make_memory_result({
            'kind': 'stage1_flag_suppression',
            'task_id': record_task_id,
        })
        memory_service = AsyncMock()
        memory_service.search = AsyncMock(return_value=[suppression_record])

        suppressed_flag = {'task_id': flag_task_id, 'flag_type': 'missing_deliverable'}
        unrelated_flag = {'task_id': 99, 'flag_type': 'stale_metadata'}
        flags = [suppressed_flag, unrelated_flag]

        result = await filter_suppressed(memory_service, 'p', flags)
        assert len(result) == 1
        assert result[0] == unrelated_flag

    @pytest.mark.asyncio
    async def test_wrong_kind_not_treated_as_suppression(self):
        """(e) result with metadata.kind != 'stage1_flag_suppression' is NOT treated as suppression.

        Canonical-schema enforcement: vector-search near-miss must be rejected.
        The flag must pass through unchanged.
        """
        from fused_memory.reconciliation.flag_dedup import filter_suppressed

        near_miss_record = _make_memory_result({
            'kind': 'some_other_kind',
            'task_id': 42,
        })
        memory_service = AsyncMock()
        memory_service.search = AsyncMock(return_value=[near_miss_record])

        flags = [{'task_id': 42, 'flag_type': 'missing_deliverable'}]
        result = await filter_suppressed(memory_service, 'p', flags)
        assert len(result) == 1
        assert result[0] == flags[0]

    @pytest.mark.asyncio
    async def test_correct_kind_but_no_task_id_key_ignored(self):
        """(f) result with metadata.kind correct but no task_id key → ignored, flag passes through."""
        from fused_memory.reconciliation.flag_dedup import filter_suppressed

        no_task_id_record = _make_memory_result({
            'kind': 'stage1_flag_suppression',
            # 'task_id' key intentionally absent
        })
        memory_service = AsyncMock()
        memory_service.search = AsyncMock(return_value=[no_task_id_record])

        flags = [{'task_id': 42, 'flag_type': 'missing_deliverable'}]
        result = await filter_suppressed(memory_service, 'p', flags)
        assert len(result) == 1
        assert result[0] == flags[0]

    @pytest.mark.asyncio
    async def test_metadata_none_guard_does_not_crash(self):
        """(g) result with metadata=None → defensive (r.metadata or {}) guard, doesn't crash, flag passes through."""
        from fused_memory.reconciliation.flag_dedup import filter_suppressed

        null_metadata_record = _make_memory_result(None)
        memory_service = AsyncMock()
        memory_service.search = AsyncMock(return_value=[null_metadata_record])

        flags = [{'task_id': 42, 'flag_type': 'missing_deliverable'}]
        result = await filter_suppressed(memory_service, 'p', flags)
        assert len(result) == 1
        assert result[0] == flags[0]

    @pytest.mark.asyncio
    @pytest.mark.parametrize('suppression_records,label', [
        ([], 'empty_set'),
        (
            [_make_memory_result({'kind': 'stage1_flag_suppression', 'task_id': 'None'})],
            'none_string_in_set',
        ),
        (
            [_make_memory_result({'kind': 'stage1_flag_suppression', 'task_id': 42})],
            'other_valid_id_in_set',
        ),
    ], ids=['empty_set', 'none_string_in_set', 'other_valid_id_in_set'])
    async def test_flag_with_none_task_id_never_dropped(self, suppression_records, label):
        """(h) flag with task_id=None is never suppressed regardless of suppression set contents.

        The consumer-side guard short-circuits to 'keep' when flag_tid is None,
        regardless of what is in the suppressed_task_ids set.  This is symmetric
        with the producer-side suppression-record guard that skips None/empty
        task_ids when building the suppressed set.

        Three cases are parametrized to express the invariant directly:
        - empty_set: trivially preserved (no suppression records at all)
        - none_string_in_set: 'None' as a literal string passes the producer
          guard; without the consumer guard, str(None) == 'None' would drop
          the flag — this is the key regression scenario.
        - other_valid_id_in_set: a real suppression id (42) must not suppress
          a flag whose task_id is None.
        """
        from fused_memory.reconciliation.flag_dedup import filter_suppressed

        memory_service = AsyncMock()
        memory_service.search = AsyncMock(return_value=suppression_records)

        # Flag whose task_id key is present but set to Python None.
        flag_with_None_task_id = {'task_id': None, 'flag_type': 'missing_deliverable'}
        flags = [flag_with_None_task_id]

        result = await filter_suppressed(memory_service, 'p', flags)

        # The flag must always be preserved — task_id=None is not a valid suppression target.
        assert len(result) == 1, (
            f'[{label}] Expected flag with task_id=None to be preserved but result was: {result}'
        )
        assert result[0] == flag_with_None_task_id, (
            f'[{label}] Preserved flag differs from input: {result[0]}'
        )


# ---------------------------------------------------------------------------
# task-1186 step-3 — filter_suppressed: search exception → pass-through + WARNING
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_filter_suppressed_search_exception_returns_flags_unchanged_and_warns(caplog):
    """Search exception in filter_suppressed → conservative pass-through + WARNING.

    When memory_service.search raises, filter_suppressed must:
    (a) NOT raise
    (b) return both flags unchanged
    (c) emit at least one WARNING log mentioning the failure and the function name
    """
    import logging

    from fused_memory.reconciliation.flag_dedup import filter_suppressed

    memory_service = AsyncMock()
    memory_service.search = AsyncMock(side_effect=RuntimeError('Mem0 down'))

    flags = [
        {'task_id': '55', 'flag_type': 'stale_metadata'},
        {'task_id': '66', 'flag_type': 'missing_deliverable'},
    ]

    with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
        result = await filter_suppressed(memory_service, 'p', flags)

    # (a) Does NOT raise
    # (b) Returns both flags unchanged
    assert result == flags
    # (c) At least one WARNING log mentions failure and filter_suppressed
    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        'filter_suppressed' in m
        for m in warning_messages
    ), f'Expected WARNING mentioning filter_suppressed but got: {warning_messages}'


# ---------------------------------------------------------------------------
# task-1188 step-3 — filter_suppressed: saturation WARNING at limit=500
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_filter_suppressed_warns_when_search_results_saturate_limit(caplog):
    """filter_suppressed emits WARNING when search returns > 500 records (true overflow).

    The search uses limit=501 internally so that exactly-500-result sets are
    not false-positive warned.  501 results confirm the real set is larger than
    500 and genuine truncation is occurring.  Operators need a WARNING so
    dashboards can alert on incomplete coverage.

    Asserts:
    (a) At least one WARNING is emitted.
    (b) The WARNING message mentions the project_id ('proj_saturation').
    (c) The WARNING message contains '500', 'saturat', or 'truncat' to signal the
        saturation condition without pinning exact wording.
    (d) The non-suppressed flag (task_id=9999) is preserved in the result — a
        regression where saturation swallowed all flags would be caught here.
    """
    import logging

    from fused_memory.reconciliation.flag_dedup import filter_suppressed

    records_501 = [
        _make_memory_result({'kind': 'stage1_flag_suppression', 'task_id': i})
        for i in range(501)
    ]
    memory_service = AsyncMock()
    memory_service.search = AsyncMock(return_value=records_501)

    flags = [{'task_id': 9999, 'flag_type': 'missing_deliverable'}]

    with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
        result = await filter_suppressed(memory_service, 'proj_saturation', flags)

    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        'proj_saturation' in m for m in warning_messages
    ), f'Expected WARNING mentioning project_id but got: {warning_messages}'
    assert any(
        ('500' in m or 'saturat' in m or 'truncat' in m)
        for m in warning_messages
    ), f'Expected WARNING signalling saturation condition but got: {warning_messages}'
    # (d) Non-suppressed flag must pass through even at the saturation boundary.
    assert result == flags, (
        f'Expected non-suppressed flag to be preserved at saturation but got: {result}'
    )


@pytest.mark.asyncio
async def test_filter_suppressed_does_not_warn_when_search_results_below_limit(caplog):
    """filter_suppressed does NOT emit saturation WARNING when results count is <= 500.

    Negative test: prevents accidental always-on logging.  500 results (the
    effective business limit) must NOT trigger the saturation signal — only
    501+ (genuine overflow) should warn.  This eliminates the false-positive
    boundary case that would otherwise fire whenever a project has exactly 500
    active suppression records with no truncation occurring.
    """
    import logging

    from fused_memory.reconciliation.flag_dedup import filter_suppressed

    records_500 = [
        _make_memory_result({'kind': 'stage1_flag_suppression', 'task_id': i})
        for i in range(500)
    ]
    memory_service = AsyncMock()
    memory_service.search = AsyncMock(return_value=records_500)

    flags = [{'task_id': 9999, 'flag_type': 'missing_deliverable'}]

    with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
        await filter_suppressed(memory_service, 'proj_below', flags)

    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert not any(
        ('500' in m or 'saturat' in m or 'truncat' in m)
        for m in warning_messages
    ), f'Unexpected saturation WARNING with 500 results: {warning_messages}'


# ---------------------------------------------------------------------------
# task-1186 step-5 — integration: dedup_flags calls filter_suppressed FIRST
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dedup_flags_calls_filter_suppressed_before_signature_dedup():
    """Integration: dedup_flags calls filter_suppressed BEFORE the signature-dedup loop.

    The mock search side_effect tracks call order:
    - Call 1 (suppression query): returns a record suppressing task_id=42.
    - Call 2+ (per-flag prior-marker queries): returns [].

    Two flags: task_id=42 (suppressed) and task_id=99 (not suppressed), both
    with flag_type='missing_deliverable'.

    Expected outcomes:
    (a) Result has exactly one item — the task_id=99 flag.
    (b) task_id=42 flag was dropped.
    (c) add_memory called exactly once (MISS path for task_id=99 only).
    (d) search called exactly 4 times: 1 suppression + 1 per-flag-marker for task_id=99
        + 2 confirmation searches (first + retry, both miss — no stage1_flag_marker written
        yet from the perspective of the confirmation search).
    """
    from fused_memory.reconciliation.flag_dedup import dedup_flags

    suppression_record = _make_memory_result({
        'kind': 'stage1_flag_suppression',
        'task_id': 42,
    })

    call_count = [0]

    async def _search_side_effect(**kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # First call is the filter_suppressed project-scoped query
            return [suppression_record]
        # Subsequent calls: per-flag prior-marker (MISS) and confirmation searches
        return []

    memory_service = AsyncMock()
    memory_service.search = AsyncMock(side_effect=_search_side_effect)
    memory_service.add_memory = AsyncMock(return_value=_STUB_ADD_MEMORY_RESPONSE)

    flags = [
        {'task_id': 42, 'flag_type': 'missing_deliverable', 'description': 'suppressed'},
        {'task_id': 99, 'flag_type': 'missing_deliverable', 'description': 'survivor'},
    ]

    result = await dedup_flags(
        memory_service=memory_service,
        project_id='p',
        run_id='r1',
        flags=flags,
    )

    # (a) Only one flag survived
    assert len(result) == 1
    # (b) task_id=42 was dropped
    assert result[0].get('task_id') == 99, (
        f"Expected surviving flag task_id=99 but got {result[0].get('task_id')!r}"
    )
    # (c) add_memory called exactly once — only for task_id=99 MISS path
    assert memory_service.add_memory.call_count == 1, (
        f'Expected exactly 1 add_memory call (task_id=99 MISS) but got '
        f'{memory_service.add_memory.call_count}'
    )
    # (d) search called exactly 4 times: 1 suppression + 1 per-flag-marker + 2 confirmation
    assert memory_service.search.call_count == 4, (
        f'Expected 4 search calls (1 suppression + 1 per-flag + 2 confirmation) but got '
        f'{memory_service.search.call_count}'
    )


@pytest.mark.asyncio
async def test_dedup_flags_two_consecutive_runs_no_predecessor_accumulation():
    """Regression: two successive dedup_flags calls for the same flag leave exactly 1 marker.

    Uses a FakeMem0 in-memory store (self-contained, no external mocks) to
    exercise the full add/search/delete cycle deterministically.

    Run 1 (run_id='r1'): no prior exists → MISS branch → 1 marker stored.
    Run 2 (run_id='r2'): prior found → HIT branch → write replacement, delete prior
                         → still exactly 1 marker stored, with run_id='r2'.
    """
    import uuid as _uuid

    from fused_memory.reconciliation.flag_dedup import dedup_flags

    class _MemoryRecord:
        """Minimal stand-in for a Mem0 MemoryResult."""
        def __init__(self, id_: str, metadata: dict) -> None:
            self.id = id_
            self.metadata = metadata
            self.content = (
                f"Stage 1 flag marker: task={metadata.get('task_id')} "
                f"type={metadata.get('flag_type')} from run={metadata.get('run_id')}"
            )

    class FakeMem0:
        """In-memory Mem0 stub with add_memory, search, and delete_memory."""

        def __init__(self) -> None:
            self._store: dict[str, _MemoryRecord] = {}

        async def add_memory(self, *, metadata: dict, **_kwargs) -> AddMemoryResponse:
            id_ = str(_uuid.uuid4())
            self._store[id_] = _MemoryRecord(id_, metadata)
            return AddMemoryResponse(memory_ids=[id_])

        async def search(self, *, query: str = '', **_kwargs) -> list[_MemoryRecord]:
            # Return all stored records (deterministic; query not used for filtering)
            return list(self._store.values())

        async def delete_memory(self, *, memory_id: str, **_kwargs) -> None:
            self._store.pop(memory_id, None)

        def count(self) -> int:
            return len(self._store)

        def latest_run_id(self) -> str | None:
            """Return the run_id of the single stored marker (for assertion)."""
            records = list(self._store.values())
            if not records:
                return None
            return records[0].metadata.get('run_id')

    fake = FakeMem0()
    flag = {'task_id': '42', 'flag_type': 'missing_deliverable', 'description': 'foo'}

    # Run 1 — MISS branch: no prior, writes one marker
    await dedup_flags(
        memory_service=fake,
        project_id='p',
        run_id='r1',
        flags=[flag],
    )
    assert fake.count() == 1, (
        f'After run 1: expected 1 marker but got {fake.count()}'
    )

    # Run 2 — HIT branch: prior found, replacement written, prior deleted
    await dedup_flags(
        memory_service=fake,
        project_id='p',
        run_id='r2',
        flags=[flag],
    )
    assert fake.count() == 1, (
        f'After run 2: expected 1 marker (no accumulation) but got {fake.count()}'
    )
    assert fake.latest_run_id() == 'r2', (
        f"Surviving marker must have run_id='r2' but got {fake.latest_run_id()!r}"
    )


# ---------------------------------------------------------------------------
# build_suppression_payload tests (task-1185 step-1)
# ---------------------------------------------------------------------------


class TestBuildSuppressionPayload:
    """Unit tests for build_suppression_payload(task_id) -> SuppressionPayload.

    Pure dict assertions — no I/O, no async.
    """

    _CANONICAL = {
        'content': 'STAGE 1 FLAG SUPPRESSION task_id=42',
        'category': 'observations_and_summaries',
        'metadata': {'kind': 'stage1_flag_suppression', 'task_id': 42},
    }

    def test_full_payload_for_int_task_id(self):
        """Full payload dict equals the canonical schema literal for int input.

        Implicitly asserts: content, category, metadata.kind, metadata.task_id
        (int), and absence of project_id (not in canonical schema).
        """
        assert build_suppression_payload(42) == self._CANONICAL

    def test_coerces_str_task_id_to_int(self):
        """str task_id is coerced to int; resulting payload equals canonical schema."""
        assert build_suppression_payload('42') == self._CANONICAL

    def test_invalid_task_id_raises_descriptive_value_error(self):
        """Non-numeric task_id raises ValueError with function-name and bad-value context.

        The bare int() raises a context-free ValueError; the hardened implementation
        must wrap it with:
        - 'build_suppression_payload' in the error message (function-name context)
        - the bad value itself in the error message (bad-value context)
        - __cause__ set to the original ValueError/TypeError (from e chaining)
        """
        with pytest.raises(ValueError) as exc_info:
            build_suppression_payload('abc')

        error_message = str(exc_info.value)
        assert 'build_suppression_payload' in error_message, (
            f"Expected 'build_suppression_payload' in error message but got: {error_message!r}"
        )
        assert 'abc' in error_message, (
            f"Expected bad value 'abc' in error message but got: {error_message!r}"
        )
        assert exc_info.value.__cause__ is not None, (
            'Expected __cause__ to be set (from e chaining) but it was None'
        )
        assert isinstance(exc_info.value.__cause__, (ValueError, TypeError)), (
            f'Expected __cause__ to be ValueError or TypeError but got: {type(exc_info.value.__cause__)}'
        )


# ---------------------------------------------------------------------------
# Round-trip schema validation tests (task-1185 step-5)
#
# FakeMemoryService: in-test stub generalising the FakeMem0 pattern already
# used in test_dedup_flags_two_consecutive_runs_no_predecessor_accumulation.
# Accepts writer-supplied content and metadata so it works for arbitrary
# writers (flag-marker writes AND suppression-record writes).
# Kept module-private — promoting to conftest is a future refactor.
# ---------------------------------------------------------------------------

class _MemoryResultStub:
    """Minimal stand-in for a Mem0 MemoryResult, accepts explicit content."""

    def __init__(self, id_: str, content: str, metadata: dict) -> None:
        self.id = id_
        self.content = content
        self.metadata = metadata


class _FakeMemoryService:
    """In-memory memory service stub: add_memory, search, delete_memory.

    Stores records keyed by UUID.  search() applies minimal filtering:

    * When ``query`` is provided it is matched against ``metadata['kind']``
      exactly — records whose kind does not equal query are excluded.  This
      catches drift between the producer's ``metadata.kind`` and the reader's
      search query (e.g. filter_suppressed passes ``query='stage1_flag_suppression'``
      which must match the kind the producer wrote).
    * When ``categories`` is provided, only records whose stored category is in
      the list are returned.  ``categories=None`` (kwarg absent) means no category
      filter — all records pass.  This enforces the producer→reader category
      contract so that a category mismatch is caught by tests using this fake.
    * ``limit`` truncates the result list when provided.
    * All other kwargs (``stores``, ``project_id``) are accepted and ignored.

    This mirrors the existing FakeMem0 in the regression test but accepts
    writer-supplied content and metadata.
    """

    def __init__(self) -> None:
        self._store: dict[str, _MemoryResultStub] = {}
        self._categories: dict[str, str] = {}

    async def add_memory(
        self, *, content: str, category: str, metadata: dict, **_kwargs
    ) -> AddMemoryResponse:
        id_ = str(_uuid_mod.uuid4())
        self._store[id_] = _MemoryResultStub(id_, content, metadata)
        self._categories[id_] = category
        return AddMemoryResponse(memory_ids=[id_])

    async def search(
        self,
        *,
        query: str = '',
        limit: int | None = None,
        categories: list[str] | None = None,
        **_kwargs,
    ) -> list[_MemoryResultStub]:
        results = list(self._store.values())
        if categories is not None:
            results = [r for r in results if self._categories.get(r.id) in categories]
        if query:
            results = [r for r in results if r.metadata.get('kind') == query]
        if limit is not None:
            results = results[:limit]
        return results

    async def delete_memory(self, *, memory_id: str, **_kwargs) -> None:
        self._store.pop(memory_id, None)
        self._categories.pop(memory_id, None)

    def count(self) -> int:
        return len(self._store)


@pytest.mark.asyncio
async def test_fake_memory_service_search_filters_by_categories_kwarg():
    """_FakeMemoryService enforces category routing when categories kwarg is provided.

    Asserts:
    (a) A record stored with category='observations_and_summaries' is found when
        searching with categories=['observations_and_summaries'] (matching category).
    (b) The same record is NOT found when searching with categories=['preferences_and_norms']
        (category mismatch → 0 results).
    (c) The same record IS found when no categories kwarg is provided (absent kwarg =
        no category filter).

    Pins the contract that _FakeMemoryService applies the categories filter
    (case (b) would return 1 instead of 0 if the filter were dropped).
    """
    fake = _FakeMemoryService()
    await fake.add_memory(
        content='x',
        category='observations_and_summaries',
        metadata={'kind': 'stage1_flag_suppression', 'task_id': 1},
    )

    # (a) Matching category → 1 result
    results_matching = await fake.search(
        query='stage1_flag_suppression',
        categories=['observations_and_summaries'],
    )
    assert len(results_matching) == 1, (
        f'Expected 1 result for matching category but got {len(results_matching)}'
    )

    # (b) Mismatched category → 0 results
    results_mismatch = await fake.search(
        query='stage1_flag_suppression',
        categories=['preferences_and_norms'],
    )
    assert len(results_mismatch) == 0, (
        f'Expected 0 results for mismatched category but got {len(results_mismatch)}'
    )

    # (c) No categories kwarg → no filter, 1 result
    results_no_filter = await fake.search(query='stage1_flag_suppression')
    assert len(results_no_filter) == 1, (
        f'Expected 1 result with no categories filter but got {len(results_no_filter)}'
    )


@pytest.mark.asyncio
async def test_suppression_record_round_trips_via_producer():
    """Round-trip schema contract (producer path): write_suppression_record writes
    exactly what filter_suppressed's search finds.

    Exercises ONLY the canonical producer path (int task_id → write_suppression_record).
    A regression in write_suppression_record's int(task_id) coercion would be caught here.

    Canonical schema (four-line contract):
      1. metadata.kind == 'stage1_flag_suppression'
      2. metadata.task_id == 42 (int)
      3. content == 'STAGE 1 FLAG SUPPRESSION task_id=42'
      4. category == 'observations_and_summaries'

    The search kwargs match filter_suppressed's actual call shape exactly so
    that any future drift between the producer and the search-call contract is
    caught here first.
    """
    from fused_memory.reconciliation.flag_dedup import write_suppression_record

    raw_task_id = 42
    expected_content = f'STAGE 1 FLAG SUPPRESSION task_id={raw_task_id}'
    fake = _FakeMemoryService()

    # Write via the canonical producer (coerces to int).
    await write_suppression_record(fake, project_id='dark_factory', task_id=raw_task_id)

    # Search with the kwargs filter_suppressed actually uses (task-1186)
    results = await fake.search(
        query='stage1_flag_suppression',
        project_id='dark_factory',
        categories=['observations_and_summaries'],
        stores=['mem0'],
        limit=500,
    )

    # (4) Exactly one result
    assert len(results) == 1, f'Expected 1 result but got {len(results)}: {results}'

    # (5) Canonical kind key
    assert results[0].metadata['kind'] == 'stage1_flag_suppression', (
        f'metadata.kind mismatch: {results[0].metadata}'
    )

    # (6) task_id is stored as int by the producer
    assert results[0].metadata['task_id'] == raw_task_id, (
        f'metadata.task_id mismatch: {results[0].metadata["task_id"]!r}'
    )

    # (7) Canonical content
    assert results[0].content == expected_content, (
        f'content mismatch: {results[0].content!r}'
    )


@pytest.mark.asyncio
async def test_legacy_str_suppression_record_round_trips_through_consumer_search():
    """Round-trip schema contract (legacy-str consumer path): a hand-authored or
    legacy record with str task_id is found by filter_suppressed's search.

    Exercises ONLY the legacy-str path: task_id stored as str '42' via
    fake.add_memory directly, modelling records written before int coercion was
    enforced.  The reader's str-coercion makes the record visible to filter_suppressed.

    Canonical schema assertions use str-coercion so the test accurately reflects
    what a str-task_id record actually looks like in Mem0.
    """
    raw_task_id = '42'
    tid_int = int(raw_task_id)
    expected_content = f'STAGE 1 FLAG SUPPRESSION task_id={tid_int}'
    fake = _FakeMemoryService()

    # Model the legacy/hand-authored str-task_id storage shape directly.
    await fake.add_memory(
        content=expected_content,
        category='observations_and_summaries',
        project_id='dark_factory',
        metadata={'kind': 'stage1_flag_suppression', 'task_id': raw_task_id},
    )

    # Search with the kwargs filter_suppressed actually uses (task-1186)
    results = await fake.search(
        query='stage1_flag_suppression',
        project_id='dark_factory',
        categories=['observations_and_summaries'],
        stores=['mem0'],
        limit=500,
    )

    # (4) Exactly one result
    assert len(results) == 1, f'Expected 1 result but got {len(results)}: {results}'

    # (5) Canonical kind key
    assert results[0].metadata['kind'] == 'stage1_flag_suppression', (
        f'metadata.kind mismatch: {results[0].metadata}'
    )

    # (6) task_id round-trips via str-coercion (legacy records store as str)
    assert str(results[0].metadata['task_id']) == str(tid_int), (
        f'metadata.task_id mismatch: {results[0].metadata["task_id"]!r}'
    )

    # (7) Canonical content
    assert results[0].content == expected_content, (
        f'content mismatch: {results[0].content!r}'
    )


@pytest.mark.asyncio
async def test_write_suppression_record_propagates_invalid_task_id():
    """Pins that ValueError from build_suppression_payload propagates out of the
    public write_suppression_record entry point unchanged.

    Regression pin: a future producer change that wraps build_suppression_payload
    in a swallow-all try/except would silently break invalid-task_id detection.
    The existing test_invalid_task_id_raises_descriptive_value_error covers
    build_suppression_payload directly; this test closes the public-API gap.
    """
    from fused_memory.reconciliation.flag_dedup import write_suppression_record

    fake = _FakeMemoryService()
    with pytest.raises(ValueError):
        await write_suppression_record(fake, project_id='p', task_id='abc')


# ---------------------------------------------------------------------------
# filter_suppressed end-to-end test (task-1185 step-6)
#
# Closes the producer→reader contract end-to-end for both int and str
# task_id storage variants.  filter_suppressed is already on main from
# sibling task-1186; this test pins that what the producer writes is exactly
# what the reader correctly acts on.
#
# This test would have caught a producer that wrote metadata.kind under a
# different key, or metadata.task_id under a non-str-coercible type, because
# filter_suppressed requires BOTH fields to be present and correct before
# adding task_id to the suppressed set.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_filter_suppressed_drops_flag_written_by_producer():
    """Producer→reader contract (producer path): flag written by write_suppression_record
    is dropped by filter_suppressed.

    Exercises ONLY the canonical producer path (int task_id → write_suppression_record).
    A regression in write_suppression_record's int(task_id) coercion would be caught here.

    Test body:
      (1) build a FakeMemoryService;
      (2) write the suppression record via write_suppression_record (int task_id);
      (3) call filter_suppressed with a matching flag and an unrelated flag;
      (4) assert the suppressed flag is dropped;
      (5) assert the unrelated flag is preserved.
    """
    from fused_memory.reconciliation.flag_dedup import filter_suppressed, write_suppression_record

    task_id = 42
    fake = _FakeMemoryService()
    await write_suppression_record(fake, project_id='dark_factory', task_id=task_id)

    flags = [
        {'task_id': task_id, 'flag_type': 'missing_deliverable'},  # should be dropped
        {'task_id': 99, 'flag_type': 'stale_metadata'},             # should be kept
    ]

    result = await filter_suppressed(fake, 'dark_factory', flags)

    # (4) Suppressed flag is dropped
    assert len(result) == 1, (
        f'Expected 1 surviving flag but got {len(result)}: {result}'
    )
    # (5) Unrelated flag is preserved
    assert result[0]['task_id'] == 99, (
        f"Expected surviving flag task_id=99 but got {result[0]['task_id']!r}"
    )


@pytest.mark.asyncio
async def test_filter_suppressed_drops_flag_for_legacy_str_consumer_record():
    """Producer→reader contract (legacy-str consumer path): a hand-authored record
    with str task_id still causes filter_suppressed to drop the matching flag.

    Exercises ONLY the legacy-str path: task_id stored as str '42' via
    fake.add_memory directly.  The reader's str-coercion handles both int and str
    suppression records, so a legacy str record must suppress the corresponding flag.

    Test body:
      (1) build a FakeMemoryService;
      (2) write a legacy str-task_id record directly via fake.add_memory;
      (3) call filter_suppressed with a matching flag and an unrelated flag;
      (4) assert the suppressed flag is dropped;
      (5) assert the unrelated flag is preserved.
    """
    from fused_memory.reconciliation.flag_dedup import filter_suppressed

    raw_task_id = '42'
    tid_int = int(raw_task_id)
    fake = _FakeMemoryService()

    # Model the legacy/hand-authored str-task_id storage shape directly.
    await fake.add_memory(
        content=f'STAGE 1 FLAG SUPPRESSION task_id={tid_int}',
        category='observations_and_summaries',
        project_id='dark_factory',
        metadata={'kind': 'stage1_flag_suppression', 'task_id': raw_task_id},
    )

    flags = [
        {'task_id': tid_int, 'flag_type': 'missing_deliverable'},  # should be dropped
        {'task_id': 99, 'flag_type': 'stale_metadata'},             # should be kept
    ]

    result = await filter_suppressed(fake, 'dark_factory', flags)

    # (4) Suppressed flag is dropped
    assert len(result) == 1, (
        f'Expected 1 surviving flag but got {len(result)}: {result}'
    )
    # (5) Unrelated flag is preserved
    assert result[0]['task_id'] == 99, (
        f"Expected surviving flag task_id=99 but got {result[0]['task_id']!r}"
    )


# ---------------------------------------------------------------------------
# write_suppression_record tests (task-1185 step-3)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_memory_service():
    """AsyncMock memory_service with add_memory returning a stub AddMemoryResponse."""
    svc = AsyncMock()
    svc.add_memory = AsyncMock(return_value=AddMemoryResponse(memory_ids=['supp-1']))
    return svc


class TestWriteSuppressionRecord:
    """Async tests for write_suppression_record(memory_service, *, project_id, task_id, causation_id).

    Uses the ``mock_memory_service`` fixture to avoid per-test boilerplate.
    The canonical-payload, project_id, _source, and default-causation_id
    assertions are consolidated into one test that inspects the full kwargs
    dict; separate small tests cover the causation_id and str-coercion variants.
    """

    @pytest.mark.asyncio
    async def test_canonical_call_kwargs(self, mock_memory_service):
        """add_memory called once with the full canonical kwargs dict.

        Asserts content, category, metadata (kind + int task_id), project_id,
        _source sentinel, and causation_id=None (default) in a single call.
        """
        from fused_memory.reconciliation.flag_dedup import write_suppression_record

        result = await write_suppression_record(
            mock_memory_service, project_id='autopilot_video', task_id=42
        )

        mock_memory_service.add_memory.assert_called_once_with(
            content='STAGE 1 FLAG SUPPRESSION task_id=42',
            category='observations_and_summaries',
            metadata={'kind': 'stage1_flag_suppression', 'task_id': 42},
            project_id='autopilot_video',
            causation_id=None,
            _source='stage1_flag_suppression',
        )
        assert result.memory_ids == ['supp-1']

    @pytest.mark.asyncio
    async def test_passes_causation_id_when_provided(self, mock_memory_service):
        """causation_id is forwarded to add_memory when explicitly provided."""
        from fused_memory.reconciliation.flag_dedup import write_suppression_record

        await write_suppression_record(
            mock_memory_service, project_id='p', task_id=42, causation_id='recon-run-99'
        )

        kwargs = mock_memory_service.add_memory.call_args.kwargs
        assert kwargs['causation_id'] == 'recon-run-99'

    @pytest.mark.asyncio
    async def test_coerces_str_task_id(self, mock_memory_service):
        """passing task_id='42' produces metadata.task_id == 42 (int)."""
        from fused_memory.reconciliation.flag_dedup import write_suppression_record

        await write_suppression_record(mock_memory_service, project_id='p', task_id='42')

        kwargs = mock_memory_service.add_memory.call_args.kwargs
        assert kwargs['metadata']['task_id'] == 42
        assert isinstance(kwargs['metadata']['task_id'], int)


def test_write_suppression_record_importable_from_canonical_path():
    """Smoke test: write_suppression_record is importable from the path stage1.py advertises.

    If the helper is ever moved or renamed, this test fails CI rather than
    silently drifting the prompt's operator instructions
    (see STAGE1_SYSTEM_PROMPT ## Flag Suppression Check section).
    """
    from fused_memory.reconciliation.flag_dedup import write_suppression_record  # noqa: F401


# ---------------------------------------------------------------------------
# TestConfirmMarkerPersisted (task-1400 step-1+) — post-write confirmation helper
# ---------------------------------------------------------------------------


class TestConfirmMarkerPersisted:
    """Tests for confirm_marker_persisted(memory_service, *, project_id, task_id,
    flag_type, run_id, log) — the canonical-id confirmation helper.

    step-1: test_returns_canonical_id_from_search_not_response
    step-3: test_miss_then_retry_finds_marker
    step-5: test_miss_after_retry_returns_none_and_warns_never_raises
    """

    @pytest.mark.asyncio
    async def test_returns_canonical_id_from_search_not_response(self):
        """Returns MemoryResult.id from the search (canonical id), NOT any add_memory response id.

        Step-1 RED: confirm_marker_persisted does not exist yet → ImportError.
        When the confirmation search finds a matching marker, the returned id is
        the MemoryResult.id ('canonical-XYZ'), not any id from an add_memory
        response (which is the root-cause vector for the ID-mismatch bug).
        """
        import logging

        from fused_memory.reconciliation.flag_dedup import (
            confirm_marker_persisted,  # step-1: ImportError
        )

        confirmed_marker = _make_memory_result({
            'source': 'stage1_flag_marker',
            'task_id': '42',
            'flag_type': 'x',
            'run_id': 'r1',
        })
        confirmed_marker.id = 'canonical-XYZ'

        memory_service = AsyncMock()
        memory_service.search = AsyncMock(return_value=[confirmed_marker])

        result = await confirm_marker_persisted(
            memory_service,
            project_id='p',
            task_id='42',
            flag_type='x',
            run_id='r1',
            log=logging.getLogger('fused_memory.reconciliation.flag_dedup'),
        )

        assert result == 'canonical-XYZ', (
            f"confirm_marker_persisted must return the canonical MemoryResult.id "
            f"('canonical-XYZ'), not any add_memory response id; got {result!r}"
        )

    @pytest.mark.asyncio
    async def test_miss_then_retry_finds_marker(self, caplog):
        """On a first-search miss, one WARNING is emitted and a retry search is performed.

        Step-3 RED: current impl only does one search → on miss returns None immediately,
        no WARNING emitted, no retry.

        search side_effect: [[] (miss), [marker with id='retry-canon']] (retry hit)

        Asserts:
        (a) Returns the canonical id from the retry result ('retry-canon').
        (b) memory_service.search was called exactly 2 times.
        (c) Exactly one WARNING emitted BEFORE the retry containing task_id AND flag_type.
        """
        import logging

        from fused_memory.reconciliation.flag_dedup import confirm_marker_persisted

        retry_marker = _make_memory_result({
            'source': 'stage1_flag_marker',
            'task_id': '42',
            'flag_type': 'x',
            'run_id': 'r1',
        })
        retry_marker.id = 'retry-canon'

        memory_service = AsyncMock()
        # First call: miss ([]); second call (retry): hit
        memory_service.search = AsyncMock(side_effect=[[], [retry_marker]])

        with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
            result = await confirm_marker_persisted(
                memory_service,
                project_id='p',
                task_id='42',
                flag_type='x',
                run_id='r1',
                log=logging.getLogger('fused_memory.reconciliation.flag_dedup'),
            )

        # (a) Returns canonical id from retry result
        assert result == 'retry-canon', (
            f"Expected 'retry-canon' from retry but got {result!r}"
        )
        # (b) Exactly 2 search calls (initial + 1 retry)
        assert memory_service.search.call_count == 2, (
            f"Expected 2 search calls but got {memory_service.search.call_count}"
        )
        # (c) One WARNING mentioning task_id AND flag_type before the retry
        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any('42' in m and 'x' in m for m in warning_messages), (
            f"Expected WARNING containing task_id '42' and flag_type 'x' but got: {warning_messages}"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize('search_side_effect,label', [
        pytest.param([[], []], 'both_miss', id='both_miss'),
        pytest.param(RuntimeError('search exploded'), 'exception', id='exception'),
    ])
    async def test_miss_after_retry_returns_none_and_warns_never_raises(
        self, search_side_effect, label, caplog
    ):
        """On double-miss (or search exception), returns None without raising; final WARNING emitted.

        Step-5 RED: current impl already handles both-miss → None path (should pass),
        but the exception path (side_effect=RuntimeError) needs verification that
        find_prior_memories degrades it to [] so helper still returns None not raises.

        Two parametrizations:
        - both_miss:  search returns [] twice → final WARNING + None
        - exception:  search raises RuntimeError → find_prior_memories catches it → []
                      → first miss → WARNING + retry → [] → final WARNING + None

        Asserts:
        (a) Returns None.
        (b) Does NOT raise.
        (c) A final WARNING is emitted containing task_id AND flag_type
            with phrasing like 'could not confirm' (ops-greppable).
        """
        import logging

        from fused_memory.reconciliation.flag_dedup import confirm_marker_persisted

        memory_service = AsyncMock()
        if isinstance(search_side_effect, list):
            memory_service.search = AsyncMock(side_effect=search_side_effect)
        else:
            # RuntimeError — find_prior_memories catches this and returns []
            memory_service.search = AsyncMock(side_effect=search_side_effect)

        with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
            result = await confirm_marker_persisted(
                memory_service,
                project_id='p',
                task_id='55',
                flag_type='stale_metadata',
                run_id='r2',
                log=logging.getLogger('fused_memory.reconciliation.flag_dedup'),
            )

        # (a) Returns None
        assert result is None, f"[{label}] Expected None but got {result!r}"

        # (b) Does NOT raise (no exception propagated — verified by reaching here)

        # (c) Final WARNING mentioning task_id AND flag_type with 'could not confirm'
        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(
            '55' in m and 'stale_metadata' in m and 'could not confirm' in m
            for m in warning_messages
        ), (
            f"[{label}] Expected final WARNING with 'could not confirm' + task_id + flag_type "
            f"but got: {warning_messages}"
        )

    @pytest.mark.asyncio
    async def test_ignores_stale_prior_with_different_run_id(self, caplog):
        """Confirmation search must NOT accept a stale prior whose run_id differs from the current run.

        The confirmation's job is 'did MY write for THIS run land?': a prior from an
        earlier run must not masquerade as confirmation of the current write.

        Setup: memory_service.search returns ONLY a stale prior with run_id='r0' on
        BOTH the first call and the retry (side_effect=[[stale_prior],[stale_prior]]).
        Call confirm_marker_persisted with run_id='r1'.

        Asserts:
        (a) Returns None — the prior's run_id 'r0' != current 'r1', must NOT be accepted.
        (b) memory_service.search.call_count == 2 (miss → retry → miss).
        (c) Final 'could not confirm' WARNING fires containing task_id + flag_type.

        Fails until step-15 adds run_id to the confirmation kind filter so stale priors
        are not incorrectly matched.
        """
        import logging

        from fused_memory.reconciliation.flag_dedup import confirm_marker_persisted

        stale_prior = _make_memory_result({
            'source': 'stage1_flag_marker',
            'task_id': '42',
            'flag_type': 'x',
            'run_id': 'r0',  # OLD run — NOT the current run 'r1'
        })
        stale_prior.id = 'stale-prior'

        memory_service = AsyncMock()
        # Both calls return the stale prior (old run_id 'r0')
        memory_service.search = AsyncMock(side_effect=[[stale_prior], [stale_prior]])

        with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
            result = await confirm_marker_persisted(
                memory_service,
                project_id='p',
                task_id='42',
                flag_type='x',
                run_id='r1',
                log=logging.getLogger('fused_memory.reconciliation.flag_dedup'),
            )

        # (a) Returns None — stale prior must NOT be accepted as confirmation of 'r1' write
        assert result is None, (
            f"Expected None (stale prior run_id='r0' must not confirm run_id='r1') "
            f"but got {result!r}"
        )
        # (b) Both attempts made: miss → retry → miss
        assert memory_service.search.call_count == 2, (
            f"Expected 2 search calls (initial + 1 retry) but got {memory_service.search.call_count}"
        )
        # (c) Final 'could not confirm' WARNING fires containing task_id + flag_type
        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(
            '42' in m and 'x' in m and 'could not confirm' in m
            for m in warning_messages
        ), (
            f"Expected final WARNING with 'could not confirm' + task_id '42' + flag_type 'x' "
            f"but got: {warning_messages}"
        )


# ---------------------------------------------------------------------------
# task-1400 step-9 — dedup_flags HIT path: deletes priors only when confirmed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dedup_flags_hit_path_no_delete_when_confirmation_misses(caplog):
    """HIT path: when confirmation misses (marker not findable), priors are NOT deleted
    and a WARNING is emitted.

    Prior: [prior id='aaa']; add_memory returns memory_ids=['returned-id'];
    confirmation: search side_effect = [[], []] (both miss — not findable).

    Asserts:
    (a) delete_memory NOT called (priors preserved — self-healing intact).
    (b) WARNING mentioning task_id AND flag_type that the replacement could not be confirmed.
    (c) Flag IS annotated with persisted_from_run (annotation extracted before write).

    Fails until step-10 switches HIT gate from memory_ids to confirmed_id.
    """
    import logging

    from fused_memory.reconciliation.flag_dedup import dedup_flags

    prior_marker = _make_memory_result({
        'source': 'stage1_flag_marker',
        'task_id': '42',
        'flag_type': 'missing_deliverable',
        'run_id': 'r0',
        'last_seen_run_id': 'r0',
    })
    prior_marker.id = 'aaa'

    memory_service = AsyncMock()
    # suppression=[], pre-write HIT=[prior], confirmation miss=[], confirmation retry=[]
    memory_service.search = AsyncMock(side_effect=[[], [prior_marker], [], []])
    memory_service.add_memory = AsyncMock(return_value=AddMemoryResponse(memory_ids=['returned-id']))
    memory_service.delete_memory = AsyncMock(return_value=None)

    flags = [{'task_id': 42, 'flag_type': 'missing_deliverable', 'description': 'foo'}]

    with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
        result = await dedup_flags(
            memory_service=memory_service,
            project_id='p',
            run_id='r1',
            flags=flags,
        )

    # (a) delete_memory NOT called — priors preserved when replacement unfindable
    memory_service.delete_memory.assert_not_called()

    # (b) WARNING that replacement could not be confirmed
    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        ('42' in m or 'missing_deliverable' in m) and
        ('could not confirm' in m or 'cannot confirm' in m or 'not confirmed' in m
         or 'skipping' in m or 'skip' in m)
        for m in warning_messages
    ), (
        f"Expected WARNING about unconfirmed replacement for task 42 but got: {warning_messages}"
    )

    # (c) Flag IS annotated (annotation extracted before write attempt)
    assert len(result) == 1
    assert result[0].get('persisted_from_run') == 'r0'
    assert result[0].get('last_seen_run_id') == 'r1'


@pytest.mark.asyncio
async def test_dedup_flags_hit_path_deletes_when_confirmed(caplog):
    """HIT path: when confirmation succeeds (marker findable), all priors are deleted.

    Converse of the above: add_memory returns memory_ids=['returned-id'] (different
    from canonical); confirmation search returns [new_marker id='canon-new'].
    Priors must be deleted exactly as before (existing atomic-replace contract).

    Fails until step-10 switches HIT gate from memory_ids to confirmed_id.
    """
    from fused_memory.reconciliation.flag_dedup import dedup_flags

    prior_marker = _make_memory_result({
        'source': 'stage1_flag_marker',
        'task_id': '42',
        'flag_type': 'missing_deliverable',
        'run_id': 'r0',
        'last_seen_run_id': 'r0',
    })
    prior_marker.id = 'aaa'

    new_confirmed_marker = _make_memory_result({
        'source': 'stage1_flag_marker',
        'task_id': '42',
        'flag_type': 'missing_deliverable',
        'run_id': 'r1',
    })
    new_confirmed_marker.id = 'canon-new'

    memory_service = AsyncMock()
    # suppression=[], pre-write HIT=[prior], confirmation hit=[new_marker]
    memory_service.search = AsyncMock(side_effect=[[], [prior_marker], [new_confirmed_marker]])
    memory_service.add_memory = AsyncMock(return_value=AddMemoryResponse(memory_ids=['returned-DIFFERENT']))
    memory_service.delete_memory = AsyncMock(return_value=None)

    flags = [{'task_id': 42, 'flag_type': 'missing_deliverable', 'description': 'foo'}]

    result = await dedup_flags(
        memory_service=memory_service,
        project_id='p',
        run_id='r1',
        flags=flags,
    )

    # Prior deleted (atomic-replace contract holds when confirmed)
    memory_service.delete_memory.assert_called_once()
    del_kwargs = memory_service.delete_memory.call_args.kwargs
    assert del_kwargs.get('memory_id') == 'aaa'

    # Flag annotated
    assert result[0].get('persisted_from_run') == 'r0'
    assert result[0].get('last_seen_run_id') == 'r1'


# ---------------------------------------------------------------------------
# task-1400 step-11 — end-to-end integration: confirmation wired, full call sequence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dedup_flags_end_to_end_confirmation_wired():
    """Integration regression: full call sequence with one MISS-happy and one HIT-happy flag.

    Verifies the complete wired path so a later refactor cannot silently desync
    the search-call sequence.

    Flags:
    - flag_A: no task_id/flag_type → no-signature, pass-through unchanged
    - flag_B: MISS-happy (no prior); confirmation search finds the new marker
    - flag_C: HIT-happy (prior exists); confirmation search confirms replacement

    Search call sequence:
    1. suppression sweep (filter_suppressed) → []
    2. flag_B pre-write search → [] (MISS)
    3. flag_B confirmation → [b_marker] (found)
    4. flag_C pre-write search → [c_prior] (HIT)
    5. flag_C confirmation → [c_new_marker] (found)

    Asserts:
    (a) flag_A returned unchanged (no-signature pass-through)
    (b) flag_B NOT annotated (MISS → no persisted_from_run)
    (c) flag_C annotated with persisted_from_run='r0' and last_seen_run_id='r1'
    (d) add_memory called exactly twice (once for MISS, once for HIT replacement)
    (e) delete_memory called exactly once (for C's prior)
    (f) search called exactly 5 times (suppression + 2 pre-write + 2 confirmation)
    (g) filter_suppressed still issues exactly one project-scoped sweep
    """
    from fused_memory.reconciliation.flag_dedup import dedup_flags

    b_marker = _make_memory_result({
        'source': 'stage1_flag_marker',
        'task_id': '10',
        'flag_type': 'stale_metadata',
        'run_id': 'r1',
    })
    b_marker.id = 'b-canon'

    c_prior = _make_memory_result({
        'source': 'stage1_flag_marker',
        'task_id': '20',
        'flag_type': 'missing_deliverable',
        'run_id': 'r0',
        'last_seen_run_id': 'r0',
    })
    c_prior.id = 'c-prior'

    c_new_marker = _make_memory_result({
        'source': 'stage1_flag_marker',
        'task_id': '20',
        'flag_type': 'missing_deliverable',
        'run_id': 'r1',
    })
    c_new_marker.id = 'c-canon'

    memory_service = AsyncMock()
    memory_service.search = AsyncMock(side_effect=[
        [],          # (1) suppression sweep
        [],          # (2) flag_B pre-write MISS
        [b_marker],  # (3) flag_B confirmation hit
        [c_prior],   # (4) flag_C pre-write HIT
        [c_new_marker],  # (5) flag_C confirmation hit
    ])
    memory_service.add_memory = AsyncMock(return_value=_STUB_ADD_MEMORY_RESPONSE)
    memory_service.delete_memory = AsyncMock(return_value=None)

    flag_A = {'description': 'no-signature flag'}
    flag_B = {'task_id': '10', 'flag_type': 'stale_metadata', 'description': 'B'}
    flag_C = {'task_id': '20', 'flag_type': 'missing_deliverable', 'description': 'C'}

    result = await dedup_flags(
        memory_service=memory_service,
        project_id='p',
        run_id='r1',
        flags=[flag_A, flag_B, flag_C],
    )

    # (a) flag_A unchanged (no-signature)
    assert len(result) == 3
    assert result[0] == flag_A

    # (b) flag_B not annotated (MISS)
    assert 'persisted_from_run' not in result[1]

    # (c) flag_C annotated (HIT)
    assert result[2].get('persisted_from_run') == 'r0'
    assert result[2].get('last_seen_run_id') == 'r1'

    # (d) add_memory called twice
    assert memory_service.add_memory.call_count == 2, (
        f"Expected 2 add_memory calls but got {memory_service.add_memory.call_count}"
    )

    # (e) delete_memory called once (for C's prior)
    memory_service.delete_memory.assert_called_once()
    assert memory_service.delete_memory.call_args.kwargs.get('memory_id') == 'c-prior'

    # (f) search called exactly 5 times
    assert memory_service.search.call_count == 5, (
        f"Expected 5 search calls but got {memory_service.search.call_count}"
    )


# ---------------------------------------------------------------------------
# task-1400 step-7 — dedup_flags MISS path: confirmation called after write
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dedup_flags_miss_path_confirmed_marker_no_noop_warning(caplog):
    """MISS path: when confirmation succeeds (even if returned id != add_memory response id),
    no 'will not be detected next cycle' WARNING is emitted.

    search side_effect:
      [1] [] → suppression filter (no suppression)
      [2] [] → per-flag pre-write MISS (no prior)
      [3] [confirmation_marker id='canon-1'] → post-write confirmation hit

    add_memory returns memory_ids=['returned-DIFFERENT'] (different from canonical id).

    Asserts:
    (a) add_memory called once.
    (b) search called 3 times (suppression + pre-write + confirmation).
    (c) NO 'recurring flag will not be detected next cycle' WARNING (marker confirmed findable).
    (d) Flag is NOT annotated (MISS path → no persisted_from_run).

    Fails until step-8 wires confirmation into MISS branch.
    """
    import logging

    from fused_memory.reconciliation.flag_dedup import dedup_flags

    confirmation_marker = _make_memory_result({
        'source': 'stage1_flag_marker',
        'task_id': '77',
        'flag_type': 'stale_metadata',
        'run_id': 'r1',
    })
    confirmation_marker.id = 'canon-1'

    memory_service = AsyncMock()
    # [suppression filter=[], pre-write miss=[], confirmation hit=[marker]]
    memory_service.search = AsyncMock(side_effect=[[], [], [confirmation_marker]])
    memory_service.add_memory = AsyncMock(
        return_value=AddMemoryResponse(memory_ids=['returned-DIFFERENT'])
    )

    flags = [{'task_id': '77', 'flag_type': 'stale_metadata', 'description': 'test'}]

    with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
        result = await dedup_flags(
            memory_service=memory_service,
            project_id='p',
            run_id='r1',
            flags=flags,
        )

    # (a) add_memory called once
    memory_service.add_memory.assert_called_once()

    # (b) search called 3 times: suppression + pre-write + confirmation
    assert memory_service.search.call_count == 3, (
        f"Expected 3 search calls (suppression+pre-write+confirmation) but got "
        f"{memory_service.search.call_count}"
    )

    # (c) No 'will not be detected next cycle' WARNING (marker IS confirmed)
    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert not any('will not be detected next cycle' in m for m in warning_messages), (
        f"Expected no 'will not be detected next cycle' WARNING when confirmation "
        f"succeeds, but got: {warning_messages}"
    )

    # (d) Flag not annotated (MISS path)
    assert len(result) == 1
    assert 'persisted_from_run' not in result[0]


@pytest.mark.asyncio
async def test_dedup_flags_miss_path_confirmation_miss_emits_noop_warning(caplog):
    """MISS path: when confirmation misses (double-miss), the 'will not be detected next cycle'
    WARNING fires — driven off confirmation, not memory_ids.

    search side_effect:
      [1] [] → suppression filter
      [2] [] → pre-write MISS
      [3] [] → confirmation miss
      [4] [] → confirmation retry miss

    add_memory returns memory_ids=['stub-id'] (non-empty — old guard would have
    suppressed the WARNING; confirmation-driven guard must still emit it).

    Fails until step-8 wires confirmation into MISS branch.
    """
    import logging

    from fused_memory.reconciliation.flag_dedup import dedup_flags

    memory_service = AsyncMock()
    # suppression + pre-write miss + confirmation miss + confirmation retry miss
    memory_service.search = AsyncMock(side_effect=[[], [], [], []])
    memory_service.add_memory = AsyncMock(return_value=AddMemoryResponse(memory_ids=['stub-id']))

    flags = [{'task_id': '88', 'flag_type': 'missing_deliverable', 'description': 'test'}]

    with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
        result = await dedup_flags(
            memory_service=memory_service,
            project_id='p',
            run_id='r1',
            flags=flags,
        )

    # add_memory called once
    memory_service.add_memory.assert_called_once()

    # WARNING driven by confirmation failure (not just empty memory_ids)
    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        ('88' in m or 'missing_deliverable' in m) and
        ('detected next cycle' in m or 'could not confirm' in m or 'will not be detected' in m)
        for m in warning_messages
    ), (
        f"Expected WARNING about confirmation failure for task 88 but got: {warning_messages}"
    )

    # Flag not annotated (MISS path)
    assert len(result) == 1
    assert 'persisted_from_run' not in result[0]


# ---------------------------------------------------------------------------
# task-1400 step-14(B) — HIT path silent no-op does not wipe priors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dedup_flags_hit_path_silent_noop_does_not_wipe_priors(caplog):
    """HIT path: when add_memory is a silent no-op the stale prior must NOT
    masquerade as confirmation, and priors must be preserved.

    Models the deceptive no-op: add_memory returns non-empty memory_ids (as if
    the write succeeded), but the new marker was never indexed by Mem0.  The
    confirmation search can only find the surviving stale prior (run_id='r0').

    With run_id scoping in the confirmation kind filter (step-15):
    - confirmation finds only the prior (run_id='r0', != 'r1') → miss
    - confirmed_id = None → write_succeeded = False → priors NOT deleted
    - flag annotated with persisted_from_run='r0' and last_seen_run_id='r1'

    search side_effect: [[], [prior], [prior], [prior]]
      [1] suppression sweep        → []
      [2] per-flag pre-write HIT   → [prior run_id='r0']
      [3] confirmation first       → [prior run_id='r0'] (stale only, no new marker)
      [4] confirmation retry       → [prior run_id='r0'] (still stale only)

    Asserts:
    (a) delete_memory NOT called (priors preserved — silent no-op protection).
    (b) WARNING fires that the replacement could not be confirmed.
    (c) Flag annotated with persisted_from_run='r0' and last_seen_run_id='r1'
        (annotation extracted before write attempt).

    Fails until step-15 adds run_id to the confirmation kind filter so that
    the stale prior with run_id='r0' is not wrongly accepted as proof that the
    run_id='r1' write landed.
    """
    import logging

    from fused_memory.reconciliation.flag_dedup import dedup_flags

    prior_marker = _make_memory_result({
        'source': 'stage1_flag_marker',
        'task_id': '42',
        'flag_type': 'missing_deliverable',
        'run_id': 'r0',
        'last_seen_run_id': 'r0',
    })
    prior_marker.id = 'aaa'

    memory_service = AsyncMock()
    # suppression=[], pre-write HIT=[prior], confirmation=[prior], retry=[prior]
    memory_service.search = AsyncMock(
        side_effect=[[], [prior_marker], [prior_marker], [prior_marker]]
    )
    memory_service.add_memory = AsyncMock(
        return_value=AddMemoryResponse(memory_ids=['returned-id'])
    )
    memory_service.delete_memory = AsyncMock(return_value=None)

    flags = [{'task_id': 42, 'flag_type': 'missing_deliverable', 'description': 'foo'}]

    with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
        result = await dedup_flags(
            memory_service=memory_service,
            project_id='p',
            run_id='r1',
            flags=flags,
        )

    # (a) delete_memory NOT called — priors preserved (silent no-op protection)
    memory_service.delete_memory.assert_not_called()

    # (b) WARNING fires that replacement could not be confirmed
    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        ('42' in m or 'missing_deliverable' in m) and
        ('could not confirm' in m or 'cannot confirm' in m or 'not confirmed' in m
         or 'skipping' in m or 'skip' in m)
        for m in warning_messages
    ), (
        f"Expected WARNING about unconfirmed replacement but got: {warning_messages}"
    )

    # (c) Flag annotated (annotation extracted before write attempt, from the prior)
    assert len(result) == 1
    assert result[0].get('persisted_from_run') == 'r0'
    assert result[0].get('last_seen_run_id') == 'r1'
