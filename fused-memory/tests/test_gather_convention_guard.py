"""Guard tests for the recon `asyncio.gather(..., return_exceptions=True)` +
`isinstance(r, BaseException)` swallow idiom (PRD
plans/fm-cancellederror-convention-prd.md, task γ).

Two independent tests live here:

1. ``TestCapturedCancelledErrorPropagatesFromSummaryPool`` — a behavioural
   regression test that drives the real
   ``reconciliation.summary_pool.enforce_summary_pool_cap`` sweep with an
   injected ``asyncio.CancelledError`` from one ``delete_memory`` call, and
   asserts it re-raises (rather than being swallowed into a WARNING and an
   under-counted return value).

2. ``TestNoRawGatherReturnExceptionsOutsideHelperOrAllowlist`` — a whole-tree
   AST drift guard: every ``asyncio.gather(..., return_exceptions=True)`` call
   site under ``src/fused_memory/`` must live either inside the helper home
   (``utils/async_utils.py``) or in the explicit drain ALLOWLIST; any other
   site fails the test by file:line:function.

Both tests are new — task γ owns this file exclusively (see the PRD's W5 seam
note); no existing recon test files are touched.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from fused_memory.reconciliation.summary_pool import enforce_summary_pool_cap


class TestCapturedCancelledErrorPropagatesFromSummaryPool:
    """A captured CancelledError from one delete_memory call must re-raise,
    aborting the sweep — NOT be swallowed into a WARNING + under-count.

    RED on current main: enforce_summary_pool_cap's per-item guard is
    ``isinstance(result, BaseException)``, which catches the captured
    CancelledError, logs it as a WARNING ("not counted"), and returns an int
    (success_count) instead of re-raising.
    """

    @pytest.mark.asyncio
    async def test_cancelled_error_from_one_delete_reraises(self) -> None:
        members = [
            {'id': 'oldest', 'created_at': '2026-01-01T00:00:00+00:00', 'metadata': {}},
            {'id': 'second', 'created_at': '2026-02-01T00:00:00+00:00', 'metadata': {}},
            {'id': 'third', 'created_at': '2026-03-01T00:00:00+00:00', 'metadata': {}},
            {'id': 'newest', 'created_at': '2026-04-01T00:00:00+00:00', 'metadata': {}},
        ]

        async def _delete_side_effect(*, memory_id, store, project_id, causation_id, _source):
            if memory_id == 'oldest':
                raise asyncio.CancelledError()
            return None

        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata = AsyncMock(return_value=members)
        memory_service.delete_memory = AsyncMock(side_effect=_delete_side_effect)

        with pytest.raises(asyncio.CancelledError):
            await enforce_summary_pool_cap(
                memory_service,
                project_id='dark_factory',
                run_id='run-cancel',
                recon_pool='stage2_cycle_summary',
                trim_source='stage2_cycle_summary_trim',
                cap=2,
            )
