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
