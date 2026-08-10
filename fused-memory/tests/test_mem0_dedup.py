"""Tests for fused_memory.reconciliation.mem0_dedup module.

Tests cover find_prior_memory: match with symmetric str coercion, no-match
paths, search exception handling, search kwargs forwarding, and multi-key
kind discriminator.
"""
from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_memory_result(metadata: dict) -> MagicMock:
    """Build a minimal mock MemoryResult with .metadata and .content."""
    r = MagicMock()
    r.metadata = metadata
    r.content = 'some memory content'
    return r


# ---------------------------------------------------------------------------
# Step 5 — match with symmetric str coercion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_returns_match_with_symmetric_str_coercion():
    """find_prior_memory returns a match when prior metadata has int task_id=42
    and the caller passes task_id='42' (str) — both sides coerced to str.
    """
    from fused_memory.reconciliation.mem0_dedup import find_prior_memory

    prior = _make_memory_result({'task_id': 42, 'flag_type': 'foo'})

    memory_service = MagicMock()
    memory_service.search = AsyncMock(return_value=[prior])

    result = await find_prior_memory(
        memory_service,
        project_id='p',
        task_id='42',
        kind={'flag_type': 'foo'},
        query='q',
    )

    assert result is prior, (
        f'Expected find_prior_memory to return the matching result '
        f'but got {result!r}'
    )


# ---------------------------------------------------------------------------
# Step 7 — no-match paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_returns_none_when_no_match():
    """Returns None when search returns rows that do not satisfy the kind discriminator."""
    from fused_memory.reconciliation.mem0_dedup import find_prior_memory

    wrong_type = _make_memory_result({'task_id': '42', 'flag_type': 'wrong_type'})

    memory_service = MagicMock()
    memory_service.search = AsyncMock(return_value=[wrong_type])

    result = await find_prior_memory(
        memory_service,
        project_id='p',
        task_id='42',
        kind={'flag_type': 'correct_type'},
        query='q',
    )

    assert result is None, (
        f'Expected None when kind discriminator does not match but got {result!r}'
    )


@pytest.mark.asyncio
async def test_returns_none_on_empty_search_result():
    """Returns None when search returns an empty list."""
    from fused_memory.reconciliation.mem0_dedup import find_prior_memory

    memory_service = MagicMock()
    memory_service.search = AsyncMock(return_value=[])

    result = await find_prior_memory(
        memory_service,
        project_id='p',
        task_id='99',
        kind={'flag_type': 'some_type'},
        query='q',
    )

    assert result is None


# ---------------------------------------------------------------------------
# Step 9 — search exception returns None and logs WARNING
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_exception_returns_none_and_logs_warning(caplog):
    """When memory_service.search raises, find_prior_memory returns None and logs a WARNING.

    The caller must never see the exception — search failures degrade gracefully.
    """
    from fused_memory.reconciliation.mem0_dedup import find_prior_memory

    memory_service = MagicMock()
    memory_service.search = AsyncMock(side_effect=RuntimeError('Mem0 down'))

    with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.mem0_dedup'):
        result = await find_prior_memory(
            memory_service,
            project_id='p',
            task_id='55',
            kind={'flag_type': 'foo'},
            query='q',
        )

    # (a) Does NOT raise
    # (b) Returns None
    assert result is None

    # (c) At least one WARNING record mentions the task_id
    assert any(
        '55' in record.message and record.levelno >= logging.WARNING
        for record in caplog.records
    ), (
        f'Expected a WARNING mentioning task 55 but got: '
        f'{[r.message for r in caplog.records]}'
    )


# ---------------------------------------------------------------------------
# Step 11 — search kwargs forwarding
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_kwargs_forwarded():
    """find_prior_memory forwards query, project_id, categories, and limit as kwargs."""
    from fused_memory.reconciliation.mem0_dedup import find_prior_memory

    memory_service = MagicMock()
    memory_service.search = AsyncMock(return_value=[])

    await find_prior_memory(
        memory_service,
        project_id='p',
        task_id='42',
        kind={'flag_type': 'foo'},
        query='Q',
        categories=['observations_and_summaries'],
        limit=37,
    )

    memory_service.search.assert_called_once()
    call_kwargs = memory_service.search.call_args.kwargs
    assert call_kwargs.get('query') == 'Q'
    assert call_kwargs.get('project_id') == 'p'
    assert call_kwargs.get('categories') == ['observations_and_summaries']
    assert call_kwargs.get('limit') == 37


# ---------------------------------------------------------------------------
# Step 13 — multi-key kind discriminator: ALL keys must match
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kind_dict_requires_all_keys_to_match():
    """Returns None when task_id and one kind key match but another kind key does not.

    Locks down AND semantics: all kind keys must match simultaneously.
    flag_dedup depends on this with kind={'source': 'stage1_flag_marker', 'flag_type': ftype}.
    """
    from fused_memory.reconciliation.mem0_dedup import find_prior_memory

    # task_id and flag_type match, but source does NOT
    candidate = _make_memory_result({
        'task_id': '42',
        'flag_type': 'foo',
        'source': 'wrong_source',
    })

    memory_service = MagicMock()
    memory_service.search = AsyncMock(return_value=[candidate])

    result = await find_prior_memory(
        memory_service,
        project_id='p',
        task_id='42',
        kind={'flag_type': 'foo', 'source': 'stage1_flag_marker'},
        query='q',
    )

    assert result is None, (
        f'Expected None when source key fails to match but got {result!r}'
    )


# --- Step 1132 — empty-string kind-value boundary ---


@pytest.mark.asyncio
async def test_empty_string_kind_value_does_not_match_missing_key():
    """find_prior_memory returns None when the row's metadata lacks a kind key entirely.

    A kind value of '' should only match rows that explicitly store that key
    with an empty-string value — a missing key must NOT satisfy the filter.
    This guards against the meta.get(k, '') default coercing a missing key to ''.
    """
    from fused_memory.reconciliation.mem0_dedup import find_prior_memory

    # metadata has task_id but no 'flag_type' key at all
    row = _make_memory_result({'task_id': '42'})

    memory_service = MagicMock()
    memory_service.search = AsyncMock(return_value=[row])

    result = await find_prior_memory(
        memory_service,
        project_id='p',
        task_id='42',
        kind={'flag_type': ''},
        query='q',
    )

    assert result is None, (
        f'Expected None when kind key is absent from metadata but got {result!r}'
    )


@pytest.mark.asyncio
async def test_empty_string_kind_value_matches_explicit_empty_string():
    """find_prior_memory returns the row when metadata contains flag_type='' explicitly.

    Regression guard: after the missing-key fix, rows that explicitly store an
    empty-string value for a kind key must still be matched (no over-correction).
    """
    from fused_memory.reconciliation.mem0_dedup import find_prior_memory

    # metadata has task_id and flag_type explicitly set to ''
    row = _make_memory_result({'task_id': '42', 'flag_type': ''})

    memory_service = MagicMock()
    memory_service.search = AsyncMock(return_value=[row])

    result = await find_prior_memory(
        memory_service,
        project_id='p',
        task_id='42',
        kind={'flag_type': ''},
        query='q',
    )

    assert result is row, (
        f'Expected the row when flag_type is explicitly empty string but got {result!r}'
    )


# ---------------------------------------------------------------------------
# Step 1 (task-1146) — find_prior_memories (plural) returns ALL matching rows
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_find_prior_memories_returns_all_matching():
    """find_prior_memories returns ALL rows matching task_id+kind, in order.

    Given 4 search results where 3 match (task_id='42', flag_type='foo') and 1
    does not (wrong task_id), the function returns a list of exactly those 3
    matching rows, in the order they appeared in the search result.
    """
    from fused_memory.reconciliation.mem0_dedup import find_prior_memories

    match1 = _make_memory_result({'task_id': '42', 'flag_type': 'foo'})
    match2 = _make_memory_result({'task_id': 42, 'flag_type': 'foo'})   # int task_id → str-coerced
    match3 = _make_memory_result({'task_id': '42', 'flag_type': 'foo'})
    no_match = _make_memory_result({'task_id': '99', 'flag_type': 'foo'})  # wrong task_id

    memory_service = MagicMock()
    memory_service.search = AsyncMock(return_value=[match1, no_match, match2, match3])

    results = await find_prior_memories(
        memory_service,
        project_id='p',
        task_id='42',
        kind={'flag_type': 'foo'},
        query='q',
    )

    assert results == [match1, match2, match3], (
        f'Expected the 3 matching rows in order but got {results!r}'
    )


@pytest.mark.asyncio
async def test_find_prior_memories_search_failure_returns_empty_list_and_warns(caplog):
    """When memory_service.search raises, find_prior_memories returns [] (not None) and logs WARNING.

    The returned value must be a list so callers can safely iterate without
    checking for None.  The WARNING is logged under the caller's logger when one
    is provided (mirrors find_prior_memory behaviour).
    """
    from fused_memory.reconciliation.mem0_dedup import find_prior_memories

    memory_service = MagicMock()
    memory_service.search = AsyncMock(side_effect=RuntimeError('Mem0 down'))

    caller_logger = logging.getLogger('test.caller_logger')
    with caplog.at_level(logging.WARNING, logger='test.caller_logger'):
        results = await find_prior_memories(
            memory_service,
            project_id='p',
            task_id='77',
            kind={'flag_type': 'foo'},
            query='q',
            log=caller_logger,
        )

    # Returns empty list, not None
    assert results == [], f'Expected [] on search failure but got {results!r}'

    # At least one WARNING record under the caller's logger mentions task_id
    assert any(
        '77' in record.message and record.levelno >= logging.WARNING
        for record in caplog.records
    ), (
        f'Expected a WARNING mentioning task 77 but got: '
        f'{[r.message for r in caplog.records]}'
    )


# ---------------------------------------------------------------------------
# Task 3129 (leaf δ) — the child-suppression filter must NOT leak into
# MemoryService.search (task detail item 4 / docs/prds/memory-metadata-vocabulary.md).
#
# find_prior_memories post-filters RAW service results on task_id + kind.  If
# grouped_read's suppression ever moved down into the service layer, amendment
# and sighting rows would vanish before that loop ran — hiding duplicates from
# the recon dedup detector and candidates from the near-duplicate write guard.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_child_records_survive_to_the_per_record_post_filter():
    """Amendment/sighting rows must reach find_prior_memories' task_id/kind filter."""
    from fused_memory.reconciliation.mem0_dedup import find_prior_memories

    amendment = _make_memory_result(
        {'task_id': '42', 'kind': 'amendment', 'parent_id': '11111111-1111-4111-8111-111111111111'}
    )
    sighting = _make_memory_result(
        {'task_id': '42', 'kind': 'sighting', 'parent_id': '11111111-1111-4111-8111-111111111111'}
    )

    memory_service = MagicMock()
    memory_service.search = AsyncMock(return_value=[amendment, sighting])

    amendments = await find_prior_memories(
        memory_service,
        project_id='p',
        task_id='42',
        kind={'kind': 'amendment'},
        query='q',
    )
    sightings = await find_prior_memories(
        memory_service,
        project_id='p',
        task_id='42',
        kind={'kind': 'sighting'},
        query='q',
    )

    assert amendments == [amendment], (
        f'A child record must survive to the per-record post-filter, got {amendments!r}'
    )
    assert sightings == [sighting], (
        f'A child record must survive to the per-record post-filter, got {sightings!r}'
    )


def test_memory_service_search_does_not_import_the_suppression_filter():
    """The suppression filter has exactly ONE home: server/grouped_read.py.

    A structural guard, so a future refactor that pushes grouping down a layer
    fails here rather than silently hiding child records from every raw
    ``MemoryService.search`` consumer.
    """
    import inspect

    from fused_memory.services import memory_service as memory_service_module

    source = inspect.getsource(memory_service_module)
    assert 'grouped_read' not in source, (
        'services/memory_service.py must not reference server/grouped_read — '
        'grouping belongs at the MCP tool layer only (task 3129 / PRD V2 bullet 4)'
    )
    assert 'group_search_results' not in source, (
        'MemoryService.search must return RAW results; grouping is applied by the '
        'MCP tool wrappers only'
    )
