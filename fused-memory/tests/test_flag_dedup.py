"""Tests for fused_memory.reconciliation.flag_dedup module.

Tests cover compute_flag_signature, dedup_flags, and error-handling behavior.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, call


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
async def test_dedup_flags_prior_marker_found_annotates_flag_and_refreshes():
    """When a prior stage1_flag_marker exists the flag gets persisted_from_run/last_seen_run_id
    and a refresh marker is written.
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
    search_call_kwargs = memory_service.search.call_args
    # Accept both positional and keyword call styles
    call_kwargs = search_call_kwargs.kwargs if search_call_kwargs.kwargs else {}
    call_args = search_call_kwargs.args if search_call_kwargs.args else ()
    # project_id should be 'p'
    assert call_kwargs.get('project_id') == 'p' or (len(call_args) >= 2 and 'p' in call_args)
    # query should mention 'task' and '42' and 'missing_deliverable'
    query = call_kwargs.get('query') or (call_args[0] if call_args else '')
    assert 'task' in query.lower() or '42' in query
    assert '42' in query or 'missing_deliverable' in query

    # (c) A refresh marker write was made via add_memory
    memory_service.add_memory.assert_called_once()


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
