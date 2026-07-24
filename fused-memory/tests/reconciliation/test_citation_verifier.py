"""Tests for ``reconciliation.citation_verifier.verify_cited_memories``.

``verify_cited_memories`` is the Stage-1 post-assembly citation-verification
pass (task 2978): for each finding's ``cited_memories`` it resolves the
``store == 'mem0'`` ids via ``memory_service.get_memory_by_id`` and drops
phantom (genuinely not-found) ids while keeping resolved ones, so a finding's
claim is never silently backed by an id that does not exist. See the task
plan for the full three-way contract (keep / drop+mark / keep+mark).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, call

import pytest

from fused_memory.reconciliation.citation_verifier import verify_cited_memories


@pytest.mark.asyncio
async def test_mem0_keeps_resolved_drops_phantom():
    """A resolving mem0 id is kept; a genuinely not-found one is dropped and
    recorded on the finding, with matching stats."""
    finding = {
        'description': 'a finding',
        'cited_memories': [
            {'memory_id': 'A', 'store': 'mem0'},
            {'memory_id': 'B', 'store': 'mem0'},
        ],
    }

    async def _get(project_id, memory_id):
        # 'A' exists (raw point read returns a payload dict); 'B' is a phantom.
        if memory_id == 'A':
            return {'id': 'A', 'content': 'x', 'metadata': {}}
        return None

    memory_service = AsyncMock()
    memory_service.get_memory_by_id = AsyncMock(side_effect=_get)

    stats = await verify_cited_memories([finding], memory_service, 'test_project')

    # 'A' resolves and stays; 'B' is a phantom and is stripped.
    assert [c['memory_id'] for c in finding['cited_memories']] == ['A']

    # The dropped phantom is recorded verbatim (exactly one entry).
    assert finding['citation_failures'] == [
        {'memory_id': 'B', 'store': 'mem0', 'reason': 'memory_not_found'},
    ]

    # Both ids were resolved against the raw point-id read, scoped to project_id.
    memory_service.get_memory_by_id.assert_has_awaits(
        [call('test_project', 'A'), call('test_project', 'B')],
        any_order=True,
    )

    # Stats reflect one verified + one dropped, no errors.
    assert stats['stage1_citations_verified'] == 1
    assert stats['stage1_phantom_citations_dropped'] == 1
    assert stats['stage1_citation_verification_errors'] == 0


@pytest.mark.asyncio
async def test_backend_error_keeps_citation_and_marks_it():
    """A backend error (e.g. Qdrant timeout) is 'unknown', not 'absent': the
    citation is KEPT and marked verification_error — never dropped, never
    propagated (dropping-on-unknown would itself be a silent-fail)."""
    finding = {
        'description': 'a finding',
        'cited_memories': [{'memory_id': 'A', 'store': 'mem0'}],
    }

    memory_service = AsyncMock()
    memory_service.get_memory_by_id = AsyncMock(
        side_effect=TimeoutError('qdrant timeout'),
    )

    # The exception must NOT propagate out of the verifier.
    stats = await verify_cited_memories([finding], memory_service, 'test_project')

    # The citation is kept (unknown != absent).
    assert [c['memory_id'] for c in finding['cited_memories']] == ['A']

    # The uncertainty is surfaced via a verification_error marker.
    assert finding['citation_failures'] == [
        {
            'memory_id': 'A',
            'store': 'mem0',
            'reason': 'verification_error',
            'error_type': 'TimeoutError',
        },
    ]

    # Stats: one error, nothing dropped, nothing verified.
    assert stats['stage1_citation_verification_errors'] == 1
    assert stats['stage1_phantom_citations_dropped'] == 0
    assert stats['stage1_citations_verified'] == 0


@pytest.mark.asyncio
async def test_graphiti_citation_left_untouched_and_never_looked_up():
    """A store=='graphiti' citation is preserved verbatim and NEVER resolved
    via get_memory_by_id (a Mem0/Qdrant-only read that would false-flag every
    graphiti edge uuid as a phantom)."""
    finding = {
        'description': 'a finding',
        'cited_memories': [
            {'memory_id': 'm1', 'store': 'mem0'},
            {'memory_id': 'g1', 'store': 'graphiti'},
        ],
    }
    memory_service = AsyncMock()
    memory_service.get_memory_by_id = AsyncMock(
        return_value={'id': 'm1', 'content': 'x', 'metadata': {}},
    )

    stats = await verify_cited_memories([finding], memory_service, 'test_project')

    # Both citations remain; the graphiti one is untouched.
    assert finding['cited_memories'] == [
        {'memory_id': 'm1', 'store': 'mem0'},
        {'memory_id': 'g1', 'store': 'graphiti'},
    ]
    assert 'citation_failures' not in finding

    # Only the mem0 id was resolved — the graphiti uuid was never looked up.
    memory_service.get_memory_by_id.assert_awaited_once_with('test_project', 'm1')

    assert stats['stage1_citations_verified'] == 1
    assert stats['stage1_phantom_citations_dropped'] == 0
    assert stats['stage1_citation_verification_errors'] == 0


@pytest.mark.asyncio
async def test_malformed_entries_skipped_without_error():
    """Non-dict entries and dicts with a missing/empty memory_id are left in
    place without a lookup and without erroring."""
    finding = {
        'description': 'a finding',
        'cited_memories': [
            'not-a-dict',
            {'store': 'mem0'},  # missing memory_id
            {'memory_id': '', 'store': 'mem0'},  # empty memory_id
            {'memory_id': 'm1', 'store': 'mem0'},
        ],
    }
    memory_service = AsyncMock()
    memory_service.get_memory_by_id = AsyncMock(
        return_value={'id': 'm1', 'content': 'x', 'metadata': {}},
    )

    stats = await verify_cited_memories([finding], memory_service, 'test_project')

    # Every malformed entry is preserved verbatim; only the real mem0 id resolves.
    assert finding['cited_memories'] == [
        'not-a-dict',
        {'store': 'mem0'},
        {'memory_id': '', 'store': 'mem0'},
        {'memory_id': 'm1', 'store': 'mem0'},
    ]
    assert 'citation_failures' not in finding

    memory_service.get_memory_by_id.assert_awaited_once_with('test_project', 'm1')

    assert stats['stage1_citations_verified'] == 1
    assert stats['stage1_phantom_citations_dropped'] == 0
    assert stats['stage1_citation_verification_errors'] == 0


@pytest.mark.asyncio
async def test_finding_without_cited_memories_is_noop():
    """A finding with no 'cited_memories' key adds no citation_failures and
    triggers no lookup."""
    finding = {'description': 'a finding with no citations'}
    memory_service = AsyncMock()
    memory_service.get_memory_by_id = AsyncMock()

    stats = await verify_cited_memories([finding], memory_service, 'test_project')

    assert 'citation_failures' not in finding
    memory_service.get_memory_by_id.assert_not_awaited()
    assert stats == {
        'stage1_phantom_citations_dropped': 0,
        'stage1_citations_verified': 0,
        'stage1_citation_verification_errors': 0,
    }


@pytest.mark.asyncio
async def test_multiple_findings_processed_independently():
    """Each finding in one call is verified independently — a phantom in one
    finding never leaks its marker onto another."""
    finding_a = {
        'description': 'finding A',
        'cited_memories': [{'memory_id': 'good', 'store': 'mem0'}],
    }
    finding_b = {
        'description': 'finding B',
        'cited_memories': [{'memory_id': 'phantom', 'store': 'mem0'}],
    }

    async def _get(project_id, memory_id):
        return {'id': memory_id} if memory_id == 'good' else None

    memory_service = AsyncMock()
    memory_service.get_memory_by_id = AsyncMock(side_effect=_get)

    stats = await verify_cited_memories(
        [finding_a, finding_b], memory_service, 'test_project',
    )

    # A's good citation stays and is unmarked.
    assert [c['memory_id'] for c in finding_a['cited_memories']] == ['good']
    assert 'citation_failures' not in finding_a

    # B's phantom is dropped and marked on B alone.
    assert finding_b['cited_memories'] == []
    assert finding_b['citation_failures'] == [
        {'memory_id': 'phantom', 'store': 'mem0', 'reason': 'memory_not_found'},
    ]

    assert stats['stage1_citations_verified'] == 1
    assert stats['stage1_phantom_citations_dropped'] == 1
    assert stats['stage1_citation_verification_errors'] == 0
