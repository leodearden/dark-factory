"""Tests for fused_memory.reconciliation.flag_dedup module.

Tests cover compute_flag_signature, dedup_flags, and error-handling behavior.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

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
    memory_service.add_memory = AsyncMock(return_value=None)
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
    memory_service.add_memory = AsyncMock(return_value=None)

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
    memory_service.add_memory = AsyncMock(return_value=None)

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
    memory_service.add_memory = AsyncMock(return_value=None)

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
    memory_service.add_memory = AsyncMock(return_value=None)
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
    memory_service.add_memory = AsyncMock(return_value=None)

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
# Step 3 (task-1146) — atomic-replace: HIT writes new marker then deletes prior
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
    memory_service.add_memory = AsyncMock(return_value=None)
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

    # (d) ordering: add_memory before delete_memory in mock_calls
    call_names = [str(c) for c in memory_service.mock_calls]
    add_idx = next(i for i, c in enumerate(call_names) if 'add_memory' in c)
    del_idx = next(i for i, c in enumerate(call_names) if 'delete_memory' in c)
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
    (c) flag annotation uses the FIRST prior's run_id
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

    prior1 = _prior('p-1', 'r0')  # first found — annotation should come from this one
    prior2 = _prior('p-2', 'r-prev')
    prior3 = _prior('p-3', 'r-earlier')

    memory_service = AsyncMock()
    memory_service.search = AsyncMock(return_value=[prior1, prior2, prior3])
    memory_service.add_memory = AsyncMock(return_value=None)
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

    # (c) annotation from first prior
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
