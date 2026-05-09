"""Tests for fused_memory.reconciliation.flag_dedup module.

Tests cover compute_flag_signature, dedup_flags, and error-handling behavior.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.models.memory import AddMemoryResponse

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
    """Flags without task_id/flag_type pass through unchanged with zero I/O calls."""
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
    # Zero I/O calls
    memory_service.search.assert_not_called()
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

    # (a) Flag is annotated with persisted_from_run and last_seen_run_id
    assert len(result) == 1
    assert result[0]['persisted_from_run'] == 'r0'
    assert result[0]['last_seen_run_id'] == 'r1'

    # (b) search was called once with project_id='p' and a query mentioning task_id and flag_type
    memory_service.search.assert_called_once()
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
    memory_service.search = AsyncMock(return_value=[wrong_source, wrong_flag_type, both_wrong])
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

    # Exactly one search per flag
    memory_service.search.assert_called_once()


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

    memory_service = AsyncMock()
    memory_service.search = AsyncMock(return_value=[prior_marker])
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
        # Non-empty memory_ids: delete should proceed
        memory_service.delete_memory.assert_called_once()
        del_kwargs = memory_service.delete_memory.call_args.kwargs
        assert del_kwargs.get('memory_id') == 'prior-hit-resp-test'
    else:
        # Empty memory_ids: delete must be skipped
        memory_service.delete_memory.assert_not_called()

    # Check WARNING for no-op case
    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    if expect_noop_warning:
        assert any('42' in m for m in warning_messages), (
            f'Expected WARNING mentioning task_id=42 but got: {warning_messages}'
        )
        assert any('missing_deliverable' in m for m in warning_messages), (
            f'Expected WARNING mentioning flag_type but got: {warning_messages}'
        )
    else:
        # No no-op warning expected when memory_ids is non-empty
        noop_warnings = [m for m in warning_messages if 'no memory_ids' in m or 'returned no memory_ids' in m]
        assert not noop_warnings, (
            f'Unexpected no-op WARNING on non-empty memory_ids path: {noop_warnings}'
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
        """(c) search called with canonical kwargs (project_id, categories, stores, limit, query)."""
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
        assert kwargs.get('limit') == 50
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
