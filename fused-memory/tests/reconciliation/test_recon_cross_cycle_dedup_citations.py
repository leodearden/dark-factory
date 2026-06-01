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

  Review fix (step-5): the UNKNOWN-project branch cannot carry a dedup anchor
               and 'appears in Known Projects' ↔ 'in known_projects' ↔
               'cite_task succeeds' — TestUnknownProjectBranchHasNoAnchor.

Only stage2.py is edited (step-2, step-6); the dedup machinery tested here
(compute_flag_signature, dedup_flags, _derive_affected_ids,
compute_content_fingerprint) already landed on main via task-1573's siblings.

NOTE — Coverage scope
---------------------
The only production change in this diff is the stage2.py prompt edit.  These
tests do NOT assert the prompt's text; they are contract/characterization tests
that pin the dedup machinery the prompt directs the LLM to use.
TestUnknownProjectBranchHasNoAnchor in particular locks in the
known-projects ↔ cite_task-success equivalence that motivated step-6's
correction.  Prompt-behaviour verification lives in the LLM eval/cutover
end-to-end suite (test_cutover_end_to_end.py), not here.
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


def _build_state(register_reify: bool = True) -> ReconReportState:
    """Build a ReconReportState for cite_task tests.

    When *register_reify* is True (default) the 'reify' project is added to
    ``known_projects`` so that ``cite_task(project_id='reify', ...)`` succeeds.
    When False, 'reify' is NOT registered and cite_task returns
    ``unknown_project`` without attaching any anchor.

    The task_interceptor mock is always non-None so tests can assert whether
    ``get_task`` was (not) awaited.  No memory_service is needed for this
    scope (no entity/edge/memory cites).
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
    if register_reify:
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
        state = _build_state()
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


# ---------------------------------------------------------------------------
# Premise (2): Stage-1 signature path dedup across cycles (step-3)
# ---------------------------------------------------------------------------


class TestSignaturePathCrossCycleDedup:
    """Premise (2) — the same null-task_id cross_project finding does NOT
    re-appear the next cycle via the Stage-1 flag-marker path.

    compute_flag_signature's cited_tasks fallback (introduced by task-1573)
    derives a deterministic (task_id, flag_type) tuple from cited_tasks when
    the top-level task_id is None.  dedup_flags uses this signature to key
    the stage1_flag_marker in Mem0, so cycle 2 finds the prior marker and
    annotates the flag with persisted_from_run instead of re-escalating.

    Existing dedup_flags tests only exercise flags with a non-None top-level
    task_id.  These tests are the first to drive the full MISS→write and
    HIT→annotate flow for a null-task_id flag whose signature comes from
    cited_tasks.
    """

    def test_compute_flag_signature_cited_tasks_fallback_returns_cited_task_id(self):
        """Null top-level task_id + cited_tasks → signature derived from cited task id."""
        from fused_memory.reconciliation.flag_dedup import compute_flag_signature

        flag = {
            'flag_type': 'cross_project',
            'task_id': None,
            'cited_tasks': [{'project_id': 'reify', 'task_id': '3803'}],
        }
        result = compute_flag_signature(flag)
        assert result == ('3803', 'cross_project'), (
            f'Expected ("3803", "cross_project") via cited_tasks fallback, got {result!r}'
        )

    @pytest.mark.asyncio
    async def test_dedup_flags_cycle1_miss_writes_marker_keyed_by_cited_task(self):
        """Cycle 1 (no prior marker): MISS — flag not annotated; marker written with
        cited-tasks-derived signature (task_id='3803', flag_type='cross_project')."""
        from fused_memory.reconciliation.flag_dedup import dedup_flags

        memory_service = AsyncMock()
        # All searches return empty: no suppression entries, no prior markers.
        memory_service.search = AsyncMock(return_value=[])
        memory_service.add_memory = AsyncMock(return_value=_STUB_ADD_MEMORY_RESPONSE)

        flag = {
            'task_id': None,
            'flag_type': 'cross_project',
            'cited_tasks': [{'project_id': 'reify', 'task_id': '3803'}],
            'description': 'cycle-1: reify/3803 needs routing',
        }

        result = await dedup_flags(
            memory_service=memory_service,
            project_id='p',
            run_id='r1',
            flags=[flag],
        )

        # Flag NOT annotated — fresh finding (MISS)
        assert len(result) == 1
        assert 'persisted_from_run' not in result[0], (
            f'Fresh flag must not have persisted_from_run; got {result[0]}'
        )

        # Marker was written (cycle-1 MISS always writes a new marker)
        memory_service.add_memory.assert_called_once()
        marker_kwargs = memory_service.add_memory.call_args.kwargs
        meta = marker_kwargs.get('metadata', {})
        assert meta.get('source') == 'stage1_flag_marker', (
            f'Marker source must be "stage1_flag_marker"; got {meta.get("source")!r}'
        )
        assert meta.get('task_id') == '3803', (
            f'Marker task_id must be "3803" (cited-tasks signature); got {meta.get("task_id")!r}'
        )
        assert meta.get('flag_type') == 'cross_project', (
            f'Marker flag_type must be "cross_project"; got {meta.get("flag_type")!r}'
        )
        assert meta.get('run_id') == 'r1', (
            f'Marker run_id must be "r1"; got {meta.get("run_id")!r}'
        )

    @pytest.mark.asyncio
    async def test_dedup_flags_cycle2_hit_annotates_persisted_from_run(self):
        """Cycle 2 (prior marker present): HIT — flag annotated with
        persisted_from_run='r0' and last_seen_run_id='r1'."""
        from fused_memory.reconciliation.flag_dedup import dedup_flags

        prior_marker = _make_memory_result({
            'source': 'stage1_flag_marker',
            'task_id': '3803',
            'flag_type': 'cross_project',
            'run_id': 'r0',
            'last_seen_run_id': 'r0',
        })
        prior_marker.id = 'prior-3803-r0'

        # Confirmation search must find a marker with the CURRENT run_id='r1'
        # (the prior has run_id='r0' and is excluded by the run_id-scoped filter).
        new_marker_r1 = _make_memory_result({
            'source': 'stage1_flag_marker',
            'task_id': '3803',
            'flag_type': 'cross_project',
            'run_id': 'r1',
            'last_seen_run_id': 'r1',
        })
        new_marker_r1.id = 'new-3803-r1'

        memory_service = AsyncMock()
        memory_service.search = AsyncMock(side_effect=_make_search_stub(
            suppression=[[]],
            marker={('3803', 'cross_project'): [[prior_marker], [new_marker_r1]]},
        ))
        memory_service.add_memory = AsyncMock(return_value=_STUB_ADD_MEMORY_RESPONSE)
        memory_service.delete_memory = AsyncMock(return_value=None)

        flag = {
            'task_id': None,
            'flag_type': 'cross_project',
            'cited_tasks': [{'project_id': 'reify', 'task_id': '3803'}],
            'description': 'cycle-2: reify/3803 still unrouted (prose may drift)',
        }

        result = await dedup_flags(
            memory_service=memory_service,
            project_id='p',
            run_id='r1',
            flags=[flag],
        )

        # Flag annotated with persisted_from_run (HIT)
        assert len(result) == 1
        assert result[0].get('persisted_from_run') == 'r0', (
            f'Expected persisted_from_run="r0", got {result[0].get("persisted_from_run")!r}'
        )
        assert result[0].get('last_seen_run_id') == 'r1', (
            f'Expected last_seen_run_id="r1", got {result[0].get("last_seen_run_id")!r}'
        )


# ---------------------------------------------------------------------------
# Premise (3): escalation fingerprint path folds on cited_tasks (step-4)
# ---------------------------------------------------------------------------


class TestFingerprintPathCrossCycleFold:
    """Premise (3) — harness._derive_affected_ids folds on cited_tasks across
    cycles so compute_content_fingerprint is stable despite prose drift.

    These tests guard on importorskip('escalation.dedupe'), mirroring harness.py's
    conditional HAS_ESCALATION import.  If the escalation package is absent the
    tests skip cleanly rather than ImportError.

    Design contract verified here:
    - When affected_ids is NON-EMPTY (because _derive_affected_ids extracts
      task_id '3803' from cited_tasks), compute_content_fingerprint ignores the
      description → SAME fingerprint across cycles despite prose drift.
    - When affected_ids is EMPTY (citation-less finding), the fingerprint falls
      back to hashing the normalised description → DIFFERENT fingerprints when
      descriptions differ.  This is the failure mode that stage2.py's edit fixes.
    """

    def test_derive_affected_ids_extracts_task_id_from_cited_tasks(self):
        """_derive_affected_ids extracts task_id from cited_tasks (cited_tasks first)."""
        from fused_memory.reconciliation.harness import _derive_affected_ids

        finding = {
            'cited_tasks': [{'project_id': 'reify', 'task_id': '3803'}],
        }
        result = _derive_affected_ids(finding)
        assert result == ['3803'], (
            f'_derive_affected_ids must return ["3803"] from cited_tasks; got {result!r}'
        )

    def test_fingerprint_identical_across_prose_drift_when_cited_tasks_match(self):
        """Same cited_tasks + different description → SAME fingerprint.

        When affected_ids is non-empty, compute_content_fingerprint ignores the
        description.  Two cycle instances of the same cross_project finding
        (same cited_tasks, drifted prose) must produce an identical fingerprint
        so the escalation dedup system folds them instead of re-escalating.
        """
        escalation_dedupe = pytest.importorskip('escalation.dedupe')
        compute_content_fingerprint = escalation_dedupe.compute_content_fingerprint
        from fused_memory.reconciliation.harness import _derive_affected_ids

        finding_cycle1 = {
            'category': 'cross_project_routing',
            'cited_tasks': [{'project_id': 'reify', 'task_id': '3803'}],
            'description': 'Cycle 1: reify/3803 — orphaned edge needs operator routing',
        }
        finding_cycle2 = {
            'category': 'cross_project_routing',
            'cited_tasks': [{'project_id': 'reify', 'task_id': '3803'}],
            'description': 'Cycle 2: reify/3803 remains unrouted (summary has drifted)',
        }

        fp1 = compute_content_fingerprint(
            'recon_integrity_issue',
            finding_cycle1.get('category', ''),
            _derive_affected_ids(finding_cycle1),
            finding_cycle1.get('description', ''),
        )
        fp2 = compute_content_fingerprint(
            'recon_integrity_issue',
            finding_cycle2.get('category', ''),
            _derive_affected_ids(finding_cycle2),
            finding_cycle2.get('description', ''),
        )

        assert fp1 == fp2, (
            'Same cited_tasks but different prose must produce IDENTICAL fingerprints '
            f'(non-empty affected_ids ignores description); fp1={fp1!r}, fp2={fp2!r}'
        )

    def test_fingerprint_differs_when_citation_less_descriptions_drift(self):
        """Empty cited_tasks + different description → DIFFERENT fingerprints.

        This is the failure mode that step-2's stage2.py edit fixes:
        without cite_task, cited_tasks=[] → _derive_affected_ids=[] →
        fingerprint falls back to hashing the normalised description →
        two cycles with drifted prose produce different fingerprints →
        the finding re-escalates every cycle.
        """
        escalation_dedupe = pytest.importorskip('escalation.dedupe')
        compute_content_fingerprint = escalation_dedupe.compute_content_fingerprint
        from fused_memory.reconciliation.harness import _derive_affected_ids

        finding_no_cite_1 = {
            'category': 'cross_project_routing',
            'cited_tasks': [],
            'description': 'Cycle 1: cross-project issue alpha — unique prose',
        }
        finding_no_cite_2 = {
            'category': 'cross_project_routing',
            'cited_tasks': [],
            'description': 'Cycle 2: cross-project issue beta — description has drifted',
        }

        fp1 = compute_content_fingerprint(
            'recon_integrity_issue',
            finding_no_cite_1.get('category', ''),
            _derive_affected_ids(finding_no_cite_1),
            finding_no_cite_1.get('description', ''),
        )
        fp2 = compute_content_fingerprint(
            'recon_integrity_issue',
            finding_no_cite_2.get('category', ''),
            _derive_affected_ids(finding_no_cite_2),
            finding_no_cite_2.get('description', ''),
        )

        assert fp1 != fp2, (
            'Citation-less findings with different descriptions must produce DIFFERENT '
            f'fingerprints (description-hash fallback); fp1={fp1!r}, fp2={fp2!r}'
        )


# ---------------------------------------------------------------------------
# Review fix: UNKNOWN-project branch has no anchor (step-5)
# ---------------------------------------------------------------------------


class TestUnknownProjectBranchHasNoAnchor:
    """Regression: the UNKNOWN-project branch cannot carry a cross-cycle dedup anchor.

    Root cause (caught by code review): step-2 placed the cite_task call inside
    the stage2.py bullet that fires when 'no project_root in Known Projects matches
    the scope' (target project NOT registered in known_projects). However,
    ReconReportState.cite_task returns 'unknown_project' (and appends nothing) when
    project_id not in self.known_projects (recon_report.py:551). The branch
    precondition and cite_task's guard are mutually exclusive — the anchor never
    materializes in that branch.

    server/main.py builds ONE known_projects map and feeds it to BOTH
    state.known_projects (gating cite_task) AND the payload's Known Projects section
    (rendered by _format_known_projects_section). So:

        'appears in Known Projects' ↔ 'in known_projects' ↔ 'cite_task succeeds'

    The corrected step-6 prompt must key cite_task on EXACTLY this condition
    (KNOWN-project branch only). These characterization tests lock in the
    invariant and guard against re-introducing the doomed unknown-project call.
    """

    @pytest.mark.asyncio
    async def test_cite_task_unknown_project_attaches_no_anchor(self):
        """cite_task on an unregistered project returns unknown_project and
        leaves cited_tasks empty — no anchor is attached in the unknown branch."""
        state = _build_state(register_reify=False)
        state.start_report(_RUN_ID, _STAGE, _PROJECT_ID)

        r = state.add_finding(
            run_id=_RUN_ID,
            severity='moderate',
            category='cross_project_routing',
            description='Work belongs to reify/3803 — target project NOT in known_projects',
            suggested_action='Operator must register reify and route manually',
            actionable=False,
            task_id=None,
            flag_type='cross_project',
        )
        assert 'error' not in r, f'add_finding failed: {r}'
        finding_id = r['finding_id']

        # cite_task must fail because 'reify' is not in known_projects
        cite_result = await state.cite_task(
            run_id=_RUN_ID,
            finding_id=finding_id,
            project_id='reify',
            task_id='3803',
        )

        # (1) cite_task returns the unknown_project sentinel — matches _ERR_UNKNOWN_PROJECT
        assert cite_result.get('error') == 'unknown_project', (
            f'Expected error="unknown_project" when project not registered; got {cite_result!r}'
        )

        # (2) finding's cited_tasks is EMPTY — no anchor was attached
        assembled = state.get_assembled_report(_RUN_ID, _STAGE)
        assert assembled is not None, 'get_assembled_report returned None'
        items = assembled['flagged_items']
        assert len(items) == 1
        finding = items[0]
        assert finding.get('cited_tasks') == [], (
            f'cited_tasks must be [] when project is unregistered; got {finding.get("cited_tasks")!r}'
        )

        # (3) the known_projects guard at L551 short-circuits BEFORE the get_task
        # service call at L558 — so the mock is never awaited
        state._task_interceptor.get_task.assert_not_awaited()

    def test_known_projects_section_lists_project_iff_registered(self):
        """_format_known_projects_section renders 'reify' only when it is in known_projects.

        Together with test_cross_project_finding_carries_cited_tasks (step-1, where
        'reify' IS registered and cite_task succeeds), this locks in the full invariant:

            'appears in Known Projects payload section'
            ↔ 'project_id in state.known_projects'
            ↔ 'cite_task returns success and appends to cited_tasks'

        The corrected step-6 prompt must key the cite_task directive on EXACTLY this
        condition — it must appear in the KNOWN-project branch, not the unknown-project one.
        """
        from fused_memory.reconciliation.stages.task_knowledge_sync import (
            TaskKnowledgeSync,
        )

        # Two-project case: 'reify' registered → section renders and contains 'reify'
        stage = TaskKnowledgeSync.__new__(TaskKnowledgeSync)
        stage.project_id = 'dark_factory'
        stage.known_projects = {'dark_factory': '/a', 'reify': '/b'}
        section = stage._format_known_projects_section()
        assert 'reify' in section, (
            f"'reify' must appear in Known Projects section when registered; got:\n{section!r}"
        )

        # Single-project case: < 2 projects guard (L1927) → empty string
        # 'reify' is absent from the payload — the LLM's Known Projects section
        # does NOT list it, consistent with cite_task failing for an unregistered project
        stage_solo = TaskKnowledgeSync.__new__(TaskKnowledgeSync)
        stage_solo.project_id = 'dark_factory'
        stage_solo.known_projects = {'dark_factory': '/a'}
        section_solo = stage_solo._format_known_projects_section()
        assert section_solo == '', (
            f'Section must be empty when fewer than 2 projects known; got:\n{section_solo!r}'
        )
