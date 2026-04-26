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


def _make_memory_result(metadata: dict) -> MagicMock:
    """Build a minimal mock MemoryResult with .metadata and .content."""
    r = MagicMock()
    r.metadata = metadata
    r.content = 'Stage 1 flag marker'
    return r


@pytest.mark.asyncio
async def test_dedup_flags_prior_marker_found_annotates_flag_no_write():
    """When a prior stage1_flag_marker exists the flag gets persisted_from_run/last_seen_run_id
    and NO additional marker write is made (avoids monotonic marker accumulation).
    """
    from fused_memory.reconciliation.flag_dedup import dedup_flags

    prior_marker = _make_memory_result({
        'source': 'stage1_flag_marker',
        'task_id': '42',
        'flag_type': 'missing_deliverable',
        'run_id': 'r0',
        'last_seen_run_id': 'r0',
    })

    memory_service = AsyncMock()
    memory_service.search = AsyncMock(return_value=[prior_marker])
    memory_service.add_memory = AsyncMock(return_value=None)

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

    # (c) No refresh write — the prior marker is sufficient; skipping avoids N*M accumulation
    memory_service.add_memory.assert_not_called()


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
    add_kwargs = memory_service.add_memory.call_args.kwargs
    meta = add_kwargs.get('metadata', {})
    assert meta.get('source') == 'stage1_flag_marker'
    assert meta.get('task_id') == '42'
    assert meta.get('flag_type') == 'missing_deliverable'
    assert meta.get('run_id') == 'r1'
    assert meta.get('last_seen_run_id') == 'r1'

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
    add_call_kwargs = memory_service.add_memory.call_args.kwargs
    assert add_call_kwargs.get('category') == 'observations_and_summaries'
    meta = add_call_kwargs.get('metadata', {})
    assert meta.get('source') == 'stage1_flag_marker'
    assert meta.get('task_id') == '99'
    assert meta.get('flag_type') == 'stale_metadata'
    assert meta.get('run_id') == 'r1'
    assert meta.get('last_seen_run_id') == 'r1'


# ---------------------------------------------------------------------------
# dedup_flags — exception handling (step-9)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dedup_flags_search_exception_does_not_raise_and_warns(caplog):
    """When memory_service.search raises, dedup_flags does not raise, returns flags unchanged,
    and logs a WARNING.
    """
    import logging

    from fused_memory.reconciliation.flag_dedup import dedup_flags

    memory_service = AsyncMock()
    memory_service.search = AsyncMock(side_effect=RuntimeError('Mem0 down'))
    memory_service.add_memory = AsyncMock(return_value=None)

    flags = [{'task_id': '55', 'flag_type': 'stale_metadata', 'description': 'test'}]

    with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
        result = await dedup_flags(
            memory_service=memory_service,
            project_id='p',
            run_id='r1',
            flags=flags,
        )

    # (a) Does NOT raise
    # (b) Returns flag unchanged (no persisted_from_run)
    assert len(result) == 1
    assert 'persisted_from_run' not in result[0]
    # (c) WARNING log mentions the failure and task_id
    assert any(
        '55' in record.message and record.levelno >= logging.WARNING
        for record in caplog.records
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
            {'source': 'stage1_flag_marker', 'task_id': '42', 'flag_type': 'missing_deliverable'},
            id='run_id-key-absent',
        ),
        # (c) 'run_id' key present but value is None — .get returns None (not 'unknown')
        pytest.param(
            {
                'source': 'stage1_flag_marker',
                'task_id': '42',
                'flag_type': 'missing_deliverable',
                'run_id': None,
            },
            id='run_id-is-None',
        ),
        # (d) 'run_id' key present but value is '' — .get returns '' (not 'unknown')
        pytest.param(
            {
                'source': 'stage1_flag_marker',
                'task_id': '42',
                'flag_type': 'missing_deliverable',
                'run_id': '',
            },
            id='run_id-is-empty-string',
        ),
    ],
)
@pytest.mark.asyncio
async def test_dedup_flags_prior_marker_with_malformed_run_id_uses_sentinel(prior_metadata):
    """When a prior stage1_flag_marker exists but has a missing/falsy run_id, dedup_flags
    must annotate the flag with persisted_from_run='unknown' — not the current run_id.

    Three malformed shapes parametrised:
    (a) 'run_id' key absent        → .get(key, run_id) silently returns 'r1' ≠ 'unknown'
    (c) run_id=None                → .get returns None ≠ 'unknown'
    (d) run_id=''                  → .get returns '' ≠ 'unknown'
    All three must produce persisted_from_run='unknown' with the sentinel fix applied.

    Note: the case where prior.metadata is None is intentionally omitted.  When
    metadata is None, the candidate filter ((r.metadata or {}).get('source') etc.)
    returns falsy values for all three checks, so the candidate is never selected
    as ``prior`` — the code under test (run_id extraction) is never reached.
    """
    from fused_memory.reconciliation.flag_dedup import dedup_flags

    prior_marker = _make_memory_result(prior_metadata)

    memory_service = AsyncMock()
    memory_service.search = AsyncMock(return_value=[prior_marker])
    memory_service.add_memory = AsyncMock(return_value=None)

    flags = [{'task_id': 42, 'flag_type': 'missing_deliverable', 'description': 'foo'}]

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
    # No new marker write — prior was found, no accumulation
    memory_service.add_memory.assert_not_called()


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
