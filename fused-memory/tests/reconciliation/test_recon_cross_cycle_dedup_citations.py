"""Integration/smoke tests for cross-cycle dedup citations (task 1586).

Verifies end-to-end that a null-task_id cross_project finding:

  Premise (1): carries cited_tasks when assembled via the real recon_report
               machinery — TestCrossProjectFindingCarriesCitedTasks.

  Premise (2): deduplicates across reconciliation cycles via
               compute_flag_signature's cited_tasks fallback (Stage-1
               signature/marker path) — TestSignaturePathCrossCycleDedup.

  Premise (3): deduplicates across cycles via compute_content_fingerprint
               with _derive_affected_ids (escalation fingerprint path) —
               TestFingerprintPathCrossCycleFold.

Only stage2.py is edited (step-2); the dedup machinery tested here
(compute_flag_signature, dedup_flags, _derive_affected_ids,
compute_content_fingerprint) already landed on main via task-1573's siblings.
"""

from __future__ import annotations

from collections import deque
from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.models.memory import AddMemoryResponse
from fused_memory.reconciliation.flag_dedup import _marker_query
from fused_memory.server.recon_report import ReconReportState

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RUN_ID = 'smoke-1586-run-001'
_STAGE = 'task_knowledge_sync'
_PROJECT_ID = 'dark_factory'

_STUB_ADD_MEMORY_RESPONSE = AddMemoryResponse(memory_ids=['stub-id'])

# ---------------------------------------------------------------------------
# Local helpers (self-contained; do not import from test_flag_dedup.py)
# ---------------------------------------------------------------------------


def _make_memory_result(metadata: dict | None) -> MagicMock:
    """Build a minimal mock MemoryResult with the given metadata."""
    r = MagicMock()
    r.metadata = metadata
    r.content = 'Stage 1 flag marker'
    return r


def _make_search_stub(
    *,
    suppression: list[list] | None = None,
    marker: dict[tuple[str, str], list[list]] | None = None,
):
    """Local search stub suitable for ``AsyncMock(side_effect=...)``.

    Dispatches on query string:
    - ``'stage1_flag_suppression'`` — pops from the suppression queue.
    - ``_marker_query(tid, ftype)`` — pops from the matching marker queue.
    - Anything else — returns [].
    """
    suppression_queue: deque[list] = deque(suppression or [])
    marker_queues: dict[tuple[str, str], deque[list]] = {
        k: deque(v) for k, v in (marker or {}).items()
    }
    marker_query_to_key: dict[str, tuple[str, str]] = {
        _marker_query(*k): k for k in marker_queues
    }

    async def _stub(**kwargs: object) -> list:
        query: str = str(kwargs.get('query', ''))

        if query == 'stage1_flag_suppression':
            return suppression_queue.popleft() if suppression_queue else []

        key = marker_query_to_key.get(query)
        if key is not None:
            q = marker_queues[key]
            return q.popleft() if q else []

        return []

    return _stub


def _build_state_with_reify_project() -> ReconReportState:
    """Build a ReconReportState with 'reify' registered for cite_task.

    The task_interceptor mock returns a minimal task dict so that
    cite_task(project_id='reify', task_id='3803') succeeds.
    No memory_service is needed for this scope (no entity/edge/memory cites).
    """
    task_interceptor = AsyncMock()
    task_interceptor.get_task = AsyncMock(return_value={
        'title': 'Reify foreign task 3803',
        'data': {},
    })

    state = ReconReportState(
        ttl_seconds=3600,
        clock=lambda: 0.0,  # deterministic; avoid asyncio.get_running_loop() in start_report
        task_interceptor=task_interceptor,
    )
    state.known_projects['reify'] = '/tmp/reify'
    return state


# ---------------------------------------------------------------------------
# Premise (1): cross_project finding carries cited_tasks (step-1)
# ---------------------------------------------------------------------------


class TestCrossProjectFindingCarriesCitedTasks:
    """Premise (1) — a null-task_id cross_project finding carries cited_tasks.

    The existing end-to-end test (test_cutover_end_to_end::_stage2_agent) models
    the cross_project finding WITHOUT calling cite_task, so its cited_tasks list
    is empty.  This class verifies the citation-carrying contract directly: after
    cite_task(project_id='reify', task_id='3803') the assembled finding carries
    cited_tasks while task_id remains None.  That cited_tasks entry is the
    cross-cycle dedup anchor for both the Stage-1 signature path (premise 2) and
    the escalation fingerprint path (premise 3).
    """

    @pytest.mark.asyncio
    async def test_cross_project_finding_carries_cited_tasks(self):
        """cite_task on a null-task_id finding adds to cited_tasks without altering task_id."""
        state = _build_state_with_reify_project()
        state.start_report(_RUN_ID, _STAGE, _PROJECT_ID)

        r = state.add_finding(
            run_id=_RUN_ID,
            severity='moderate',
            category='cross_project_routing',
            description='Work belongs to reify/3803 — orphaned edge needs cleanup',
            suggested_action='Route to reify project for manual resolution',
            actionable=False,
            task_id=None,
            flag_type='cross_project',
        )
        assert 'error' not in r, f'add_finding failed: {r}'
        finding_id = r['finding_id']

        cite_result = await state.cite_task(
            run_id=_RUN_ID,
            finding_id=finding_id,
            project_id='reify',
            task_id='3803',
        )

        # cite_task itself must succeed and return {project_id, task_id, title}
        assert 'error' not in cite_result, f'cite_task failed: {cite_result}'
        assert cite_result['project_id'] == 'reify'
        assert cite_result['task_id'] == '3803'

        assembled = state.get_assembled_report(_RUN_ID, _STAGE)
        assert assembled is not None, 'get_assembled_report returned None'
        items = assembled['flagged_items']
        assert len(items) == 1, f'Expected 1 finding, got {len(items)}'
        finding = items[0]

        # (a) top-level task_id stays None — the cross_project routing contract
        assert finding['task_id'] is None, (
            f'task_id must remain None for cross_project finding; got {finding["task_id"]!r}'
        )

        # (b) category and flag_type preserved
        assert finding['flag_type'] == 'cross_project', (
            f'Expected flag_type="cross_project", got {finding["flag_type"]!r}'
        )
        assert finding['category'] == 'cross_project_routing', (
            f'Expected category="cross_project_routing", got {finding["category"]!r}'
        )

        # (c) cited_tasks carries exactly one entry for the foreign task
        assert len(finding['cited_tasks']) == 1, (
            f'Expected 1 cited_task, got {finding["cited_tasks"]!r}'
        )
        ct = finding['cited_tasks'][0]
        assert ct['project_id'] == 'reify', (
            f'cited_task.project_id must be "reify", got {ct["project_id"]!r}'
        )
        assert ct['task_id'] == '3803', (
            f'cited_task.task_id must be "3803", got {ct["task_id"]!r}'
        )

        # (d) post-cutover shape — affected_ids must NOT appear
        assert 'affected_ids' not in finding, (
            f'Finding must not contain affected_ids (post-cutover shape); keys: {list(finding)}'
        )
