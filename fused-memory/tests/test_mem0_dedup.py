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
