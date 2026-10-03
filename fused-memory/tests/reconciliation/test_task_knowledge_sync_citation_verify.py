"""Stage-2 and Stage-3 phantom-citation protection from the shared path (task 2979).

Task 2978 added ``verify_cited_memories`` to ``MemoryConsolidator.run()`` — a
PER-STAGE override. Stage 2 (``TaskKnowledgeSync``) and Stage 3
(``IntegrityCheck``, the ``STAGE3_REPORT_SCHEMA`` stage) therefore silently
inherited ZERO phantom-citation protection. Task 2979 hoists the pass into
``BaseStage.run()``'s shared ``items_flagged`` assembly, which every stage
funnels through, so no stage can be missed and no future stage can skip it by
forgetting to call it.

**These tests patch ``base.run_stage_via_cli``, NOT ``BaseStage.run``.**
That is load-bearing, not stylistic. The prevailing idiom elsewhere in the
suite — ``patch.object(BaseStage, 'run', new=AsyncMock(...))`` — stubs out the
exact code this task adds, so a test written that way could never go green.
Patching one layer lower keeps the real assembly path executing while still
hermetically controlling the stage's "LLM" output. Template: the five
``patch('...stages.base.run_stage_via_cli')`` sites in
``test_base_stage_cutover.py``.

Stages are constructed with ``recon_report_state=None`` on purpose: that is the
structured-output JSON fallback branch, where the model's ``flagged_items``
reach ``items_flagged`` without ever passing ``recon_report.cite_memory``'s
cite-time existence check — i.e. precisely the unchecked path this task closes.
"""

from __future__ import annotations

import logging
from contextlib import ExitStack
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import StageId, Watermark
from fused_memory.models.scope import ProjectId, ProjectRoot, ProjectScope
from fused_memory.reconciliation.cli_stage_runner import StageResult
from fused_memory.reconciliation.stages.task_knowledge_sync import (
    IntegrityCheck,
    TaskKnowledgeSync,
)

# The Stage-2 module-level post-run helpers. run()'s tail calls each of these
# after super().run(); they are irrelevant to citation verification and would
# otherwise drag real Mem0/ledger sweeps into the test. Patched to inert
# returns exactly as tests/test_stages.py already does.
_STAGE2_POST_RUN_HELPERS = (
    'write_cycle_summary',
    '_write_task_count_snapshot',
    '_verify_task_count_snapshot_written',
    '_gc_recon_markers',
    '_sweep_stale_persistence_markers',
    '_sweep_stale_mem0_flag_markers',
    '_sweep_stale_mem0_flag_for_stage2_markers',
)

# The Stage-3 finding-side filters. Each is a pass-through for the
# 'test_project' scope used here, but they are patched to the identity so a
# future policy change to one of them cannot silently turn these tests green
# by dropping the finding under test instead of verifying its citations.
_STAGE3_FINDING_FILTERS = (
    'filter_blocked_snapshot_findings',
    'filter_contamination_ceiling_findings',
)


def _deps() -> dict:
    """Keyword deps for constructing a stage — mirrors test_stages._mock_stage_deps."""
    memory_service = AsyncMock()
    memory_service.search = AsyncMock(return_value=[])
    memory_service.count_memories_by_metadata = AsyncMock(return_value=0)
    taskmaster = AsyncMock()
    taskmaster.get_tasks = AsyncMock(return_value={'tasks': []})
    return {
        'memory_service': memory_service,
        'taskmaster': taskmaster,
        'journal': AsyncMock(),
        'config': ReconciliationConfig(enabled=True, explore_codebase_root='/tmp/test'),
        'scope': ProjectScope(ProjectId('test_project'), ProjectRoot('/tmp/test')),
    }


def _resolving_memory_service(deps: dict, resolving: set[str]) -> AsyncMock:
    """Wire get_memory_by_id so ids in *resolving* exist and all others are phantoms."""

    async def _get(project_id: str, memory_id: str):
        return {'id': memory_id, 'content': 'x'} if memory_id in resolving else None

    deps['memory_service'].get_memory_by_id = AsyncMock(side_effect=_get)
    return deps['memory_service']


def _stage_result(flagged: list[dict[str, Any]]) -> StageResult:
    """A successful CLI stage result carrying *flagged* as the model's findings."""
    return StageResult(
        report={'summary': '', 'stats': {}, 'flagged_items': flagged},
        llm_calls=1,
        tokens_used=10,
        cost_usd=0.0,
        model='test-model',
        success=True,
        error='',
    )


async def _run_stage2(deps: dict, flagged: list[dict[str, Any]]):
    """Drive TaskKnowledgeSync.run() through the real BaseStage assembly path."""
    stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **deps, recon_report_state=None)
    stage._current_run_id = 'run-stage2-cite'

    stack = [
        patch(
            'fused_memory.reconciliation.stages.base.run_stage_via_cli',
            new=AsyncMock(return_value=_stage_result(flagged)),
        ),
        patch.object(stage, 'assemble_payload', new=AsyncMock(return_value='payload')),
        patch.object(
            stage, '_maybe_queue_briefing_refresh_tasks', new=AsyncMock(return_value=None)
        ),
        patch.object(stage, '_apply_post_flight_guards', new=AsyncMock(return_value=None)),
        patch.object(
            stage, '_run_entity_standing_decision_growth_sweep', new=AsyncMock(return_value=None)
        ),
    ]
    stack += [
        patch(
            f'fused_memory.reconciliation.stages.task_knowledge_sync.{name}',
            new=AsyncMock(return_value=0),
        )
        for name in _STAGE2_POST_RUN_HELPERS
    ]

    # ExitStack, not a hand-rolled nested-enter loop: it unwinds the managers
    # it already entered if a later __enter__ raises. A loop that just enters
    # in sequence leaks every patch it had installed when one blows up, and a
    # leaked mock.patch stays installed for the rest of the pytest session.
    with ExitStack() as patches:
        for manager in stack:
            patches.enter_context(manager)
        return await stage.run(
            [], Watermark(project_id='test_project'), [], run_id='run-stage2-cite'
        )


async def _run_stage3(deps: dict, flagged: list[dict[str, Any]]):
    """Drive IntegrityCheck.run() (the STAGE3_REPORT_SCHEMA stage) the same way."""
    stage = IntegrityCheck(StageId.integrity_check, **deps, recon_report_state=None)

    stack = [
        patch(
            'fused_memory.reconciliation.stages.base.run_stage_via_cli',
            new=AsyncMock(return_value=_stage_result(flagged)),
        ),
        patch.object(stage, 'assemble_payload', new=AsyncMock(return_value='payload')),
        patch(
            'fused_memory.reconciliation.stages.task_knowledge_sync'
            '.filter_false_phantom_task_creation_flags',
            new=AsyncMock(side_effect=lambda *, taskmaster, known_projects, flags: flags),
        ),
    ]
    stack += [
        patch(
            f'fused_memory.reconciliation.stages.task_knowledge_sync.{name}',
            side_effect=lambda items, **_kw: items,
        )
        for name in _STAGE3_FINDING_FILTERS
    ]

    # ExitStack, not a hand-rolled nested-enter loop: it unwinds the managers
    # it already entered if a later __enter__ raises. A loop that just enters
    # in sequence leaks every patch it had installed when one blows up, and a
    # leaked mock.patch stays installed for the rest of the pytest session.
    with ExitStack() as patches:
        for manager in stack:
            patches.enter_context(manager)
        return await stage.run(
            [], Watermark(project_id='test_project'), [], run_id='run-stage3-cite'
        )


@pytest.mark.asyncio
async def test_stage2_drops_phantom_citation_and_counts_it():
    """TaskKnowledgeSync.run() strips a phantom mem0 citation, marks it, and
    reports the drop under the stage2_ prefix."""
    deps = _deps()
    _resolving_memory_service(deps, resolving={'real'})
    flagged = [
        {
            'description': 'a stage-2 finding',
            'task_id': '42',
            'cited_memories': [
                {'memory_id': 'real', 'store': 'mem0'},
                {'memory_id': 'phantom', 'store': 'mem0'},
            ],
        },
    ]

    report = await _run_stage2(deps, flagged)

    finding = report.items_flagged[0]
    assert [c['memory_id'] for c in finding['cited_memories']] == ['real']
    assert finding['citation_failures'] == [
        {'memory_id': 'phantom', 'store': 'mem0', 'reason': 'memory_not_found'},
    ]

    assert report.stats['stage2_phantom_citations_dropped'] == 1
    assert report.stats['stage2_citations_verified'] == 1
    assert report.stats['stage2_citation_verification_errors'] == 0
    # Stage 2 must not be counted under another stage's key.
    assert 'stage1_phantom_citations_dropped' not in report.stats
    assert 'stage3_phantom_citations_dropped' not in report.stats


@pytest.mark.asyncio
async def test_stage3_drops_phantom_citation_and_counts_it():
    """IntegrityCheck.run() — the STAGE3_REPORT_SCHEMA stage — gets the same
    protection, under the stage3_ prefix. Confirms the fix reaches every stage
    funnelling through BaseStage.run(), not just the two it was written for."""
    deps = _deps()
    _resolving_memory_service(deps, resolving={'real'})
    flagged = [
        {
            'description': 'a stage-3 finding',
            'task_id': '99',
            'cited_memories': [
                {'memory_id': 'real', 'store': 'mem0'},
                {'memory_id': 'phantom', 'store': 'mem0'},
            ],
        },
    ]

    report = await _run_stage3(deps, flagged)

    finding = report.items_flagged[0]
    assert [c['memory_id'] for c in finding['cited_memories']] == ['real']
    assert finding['citation_failures'] == [
        {'memory_id': 'phantom', 'store': 'mem0', 'reason': 'memory_not_found'},
    ]

    assert report.stats['stage3_phantom_citations_dropped'] == 1
    assert report.stats['stage3_citations_verified'] == 1
    assert report.stats['stage3_citation_verification_errors'] == 0
    assert 'stage1_phantom_citations_dropped' not in report.stats
    assert 'stage2_phantom_citations_dropped' not in report.stats


@pytest.mark.asyncio
async def test_stage2_finding_without_citations_is_untouched_but_stats_present():
    """A finding carrying no cited_memories key is returned EXACTLY as the model
    emitted it — no empty cited_memories key bolted on, no citation_failures —
    while the three stage2_ counters are still present at zero (the
    explicit-zero convention Stage 2 already follows everywhere else)."""
    deps = _deps()
    memory_service = _resolving_memory_service(deps, resolving=set())
    flagged = [{'description': 'a citation-less finding', 'task_id': '7'}]

    report = await _run_stage2(deps, flagged)

    assert report.items_flagged[0] == {
        'description': 'a citation-less finding',
        'task_id': '7',
    }
    assert 'cited_memories' not in report.items_flagged[0]
    assert 'citation_failures' not in report.items_flagged[0]
    memory_service.get_memory_by_id.assert_not_awaited()

    assert report.stats['stage2_phantom_citations_dropped'] == 0
    assert report.stats['stage2_citations_verified'] == 0
    assert report.stats['stage2_citation_verification_errors'] == 0


@pytest.mark.asyncio
async def test_stage2_graphiti_citation_preserved_and_never_resolved():
    """A store=='graphiti' citation is kept verbatim and NEVER looked up:
    get_memory_by_id is a Mem0/Qdrant-only point read, so resolving a graphiti
    edge uuid through it would false-flag every graph citation as a phantom."""
    deps = _deps()
    memory_service = _resolving_memory_service(deps, resolving={'m1'})
    flagged = [
        {
            'description': 'a finding citing the graph',
            'task_id': '13',
            'cited_memories': [
                {'memory_id': 'm1', 'store': 'mem0'},
                {'memory_id': 'edge-uuid', 'store': 'graphiti'},
            ],
        },
    ]

    report = await _run_stage2(deps, flagged)

    finding = report.items_flagged[0]
    assert [c['memory_id'] for c in finding['cited_memories']] == ['m1', 'edge-uuid']
    assert 'citation_failures' not in finding

    awaited_ids = [c.args[1] for c in memory_service.get_memory_by_id.await_args_list]
    assert awaited_ids == ['m1']
    assert 'edge-uuid' not in awaited_ids

    assert report.stats['stage2_citations_verified'] == 1
    assert report.stats['stage2_phantom_citations_dropped'] == 0


# ---------------------------------------------------------------------------
# The two BaseStage branches this task adds that the tests above cannot reach:
# the write-back through an ACTIVE ReconReportState from a NON-Stage-1 run, and
# the write-back's degradation path. Everything above constructs the stage with
# recon_report_state=None (the JSON-fallback branch), so without these the
# write-back would only ever be covered for Stage 1.
# ---------------------------------------------------------------------------


class _CiteTimeMemoryService:
    """recon_report.cite_memory's cite-time existence check — passes for EVERY
    id, exactly as it does in production.

    Load-bearing for these tests: the phantom must survive cite time and be
    caught only by the later ``get_memory_by_id`` re-resolution, which is the
    TOCTOU a cite-time-only check structurally cannot catch. A stub that
    rejected the phantom here would never let it reach verification at all.
    """

    async def get_memory(self, memory_id, project_id, store):
        return {'category': 'observations_and_summaries', 'agent_id': 'x', 'created_at': 'n'}


def _make_rrs():
    from fused_memory.server.recon_report import ReconReportState

    return ReconReportState(
        ttl_seconds=300, clock=lambda: 0.0, memory_service=_CiteTimeMemoryService()
    )


_RRS_RUN_ID = 'run-stage2-rrs-cite'
_GOOD_ID = 'aaaaaaaa-1111-4111-8111-111111111111'
_PHANTOM_ID = 'bbbbbbbb-2222-4222-8222-222222222222'


async def _run_stage2_with_rrs(deps: dict, state, cited: tuple[str, ...]):
    """Drive TaskKnowledgeSync.run() with *state* ACTIVE (the RRS path).

    The mocked CLI files the finding and its citations into recon_report state
    and completes — the real ordering, all strictly BEFORE BaseStage.run()
    assembles and verifies.
    """
    from fused_memory.reconciliation.cli_stage_runner import StageResult

    stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **deps, recon_report_state=state)
    stage._current_run_id = _RRS_RUN_ID

    async def _fake_cli(**_kwargs):
        finding_id = state.add_finding(
            run_id=_RRS_RUN_ID,
            severity='moderate',
            category='task_knowledge_drift',
            description='a stage-2 finding citing memories',
            suggested_action='reconcile them',
            actionable=True,
            task_id='42',
            flag_type='knowledge_drift',
        )['finding_id']
        for memory_id in cited:
            await state.cite_memory(_RRS_RUN_ID, finding_id, memory_id, 'mem0')
        state.complete(_RRS_RUN_ID, summary='s')
        return StageResult(report={}, success=True)

    stack = [
        patch(
            'fused_memory.reconciliation.stages.base.run_stage_via_cli',
            new=AsyncMock(side_effect=_fake_cli),
        ),
        patch.object(stage, 'assemble_payload', new=AsyncMock(return_value='payload')),
        patch.object(
            stage, '_maybe_queue_briefing_refresh_tasks', new=AsyncMock(return_value=None)
        ),
        patch.object(stage, '_apply_post_flight_guards', new=AsyncMock(return_value=None)),
        patch.object(
            stage, '_run_entity_standing_decision_growth_sweep', new=AsyncMock(return_value=None)
        ),
    ]
    stack += [
        patch(
            f'fused_memory.reconciliation.stages.task_knowledge_sync.{name}',
            new=AsyncMock(return_value=0),
        )
        for name in _STAGE2_POST_RUN_HELPERS
    ]

    with ExitStack() as patches:
        for manager in stack:
            patches.enter_context(manager)
        return await stage.run(
            [], Watermark(project_id='test_project'), [], run_id=_RRS_RUN_ID
        )


@pytest.mark.asyncio
async def test_stage2_writeback_corrects_the_durable_recon_report_finding():
    """A Stage-2 run with an ACTIVE ReconReportState corrects the AUTHORITATIVE
    finding, not just the throwaway projection.

    ``get_assembled_report`` builds a fresh dict per finding carrying a NEW
    ``cited_memories`` list, so verification alone leaves the durable _Finding
    (and its persisted row) holding the phantom, and the two stores disagree
    about the same finding forever. This pins that the write-back reaches a
    NON-Stage-1 run — ``apply_citation_verification`` resolves the finding via
    ``_resolve_finding``, whose cross-stage lookup every other test here leaves
    uncovered because they all build the stage with recon_report_state=None.
    """
    deps = _deps()
    _resolving_memory_service(deps, resolving={_GOOD_ID})
    state = _make_rrs()

    report = await _run_stage2_with_rrs(deps, state, (_GOOD_ID, _PHANTOM_ID))

    # The returned report is corrected...
    assert [c['memory_id'] for c in report.items_flagged[0]['cited_memories']] == [_GOOD_ID]
    assert report.stats['stage2_phantom_citations_dropped'] == 1
    assert report.stats['stage2_citations_verified'] == 1

    # ...and so is the authoritative record the harness leaves behind.
    durable = state.get_findings_for_run(_RRS_RUN_ID)
    assert len(durable) == 1
    assert [c['memory_id'] for c in durable[0]['cited_memories']] == [_GOOD_ID], (
        'the authoritative _Finding still carries the phantom — verification '
        'corrected only the projection'
    )
    assert durable[0]['citation_failures'] == [
        {'memory_id': _PHANTOM_ID, 'store': 'mem0', 'reason': 'memory_not_found'},
    ]


@pytest.mark.asyncio
async def test_stage2_writeback_failure_degrades_without_failing_the_stage(caplog):
    """When ``apply_citation_verification`` raises, run() still returns the
    CORRECTED report and logs the failure loudly — it never fails the stage.

    The whole point of the write-back's try/except is that a citation-hygiene
    miss degrades to "report is correct, durable record is stale" rather than
    to a failed reconciliation stage. Nothing else pins that, so a refactor
    that let the exception propagate would go undetected.
    """
    deps = _deps()
    _resolving_memory_service(deps, resolving={_GOOD_ID})
    state = _make_rrs()

    with (
        patch.object(
            state,
            'apply_citation_verification',
            side_effect=RuntimeError('sqlite is on fire'),
        ),
        caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.stages.base'),
    ):
        report = await _run_stage2_with_rrs(deps, state, (_GOOD_ID, _PHANTOM_ID))

    # The stage completed and its report is still corrected.
    assert [c['memory_id'] for c in report.items_flagged[0]['cited_memories']] == [_GOOD_ID]
    assert report.stats['stage2_phantom_citations_dropped'] == 1

    # Loud, not silent: the structured warning names the run and the stage.
    records = [r for r in caplog.records if r.msg == 'reconciliation.citation_writeback_failed']
    assert len(records) == 1, (
        f'expected exactly one citation_writeback_failed warning, got {caplog.records!r}'
    )
    assert records[0].run_id == _RRS_RUN_ID
    assert records[0].stage == StageId.task_knowledge_sync.value
    assert records[0].finding_ids, 'the warning must name the affected finding_ids'
    assert 'sqlite is on fire' in records[0].error


@pytest.mark.asyncio
async def test_stage2_clean_pass_never_touches_the_durable_record():
    """A pass with zero drops and zero verification errors does NOT call
    ``apply_citation_verification`` at all.

    ``apply_citation_verification``'s ``_persist_run`` re-serialises and
    upserts EVERY entry of the run, and with nothing dropped the corrections
    would be byte-identical rewrites. This is the common case, so the skip is
    the difference between one wasted full-run write per stage per cycle and
    none.
    """
    deps = _deps()
    _resolving_memory_service(deps, resolving={_GOOD_ID})
    state = _make_rrs()

    with patch.object(
        state, 'apply_citation_verification', wraps=state.apply_citation_verification
    ) as writeback:
        report = await _run_stage2_with_rrs(deps, state, (_GOOD_ID,))

    assert report.stats['stage2_citations_verified'] == 1
    assert report.stats['stage2_phantom_citations_dropped'] == 0
    assert writeback.call_count == 0, (
        'a clean verification pass must not drive a full-run re-persist'
    )
    # And the durable record is untouched and still correct.
    durable = state.get_findings_for_run(_RRS_RUN_ID)
    assert [c['memory_id'] for c in durable[0]['cited_memories']] == [_GOOD_ID]
    assert durable[0]['citation_failures'] == []
