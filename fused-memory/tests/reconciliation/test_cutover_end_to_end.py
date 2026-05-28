"""Integration tests for PRD γ cutover: Stage 1→2→3 end-to-end cycle with recon_report state."""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import StageId, Watermark
from fused_memory.reconciliation.cli_stage_runner import STAGE_REPORT_SCHEMA, StageResult
from fused_memory.reconciliation.stages.base import BaseStage
from fused_memory.server.recon_report import ReconReportState


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_RUN_ID = 'e2e-test-run-001'
_PROJECT_ID = 'dark_factory'
_PROJECT_ROOT = '/tmp/test'
_EDGE_UUID = '12345678-1234-1234-1234-123456789abc'
_MEMORY_UUID = 'abcdef01-abcd-abcd-abcd-abcdef012345'


# ---------------------------------------------------------------------------
# Stub stage
# ---------------------------------------------------------------------------


class _StubStage(BaseStage):
    """Minimal BaseStage stub for end-to-end pipeline testing."""

    def get_disallowed_tools(self) -> list[str]:
        return []

    def get_system_prompt(self) -> str:
        return 'test prompt'

    def get_report_schema(self) -> dict:
        return STAGE_REPORT_SCHEMA

    async def assemble_payload(self, events, watermark, prior_reports) -> str:
        return 'test payload'


def _make_stub_stage(
    stage_id: StageId,
    recon_report_state: ReconReportState,
) -> _StubStage:
    config = ReconciliationConfig()
    stage = _StubStage(
        stage_id,
        AsyncMock(),  # memory_service
        AsyncMock(),  # taskmaster
        AsyncMock(),  # journal
        config,
        recon_report_port=8003,
        recon_report_state=recon_report_state,
    )
    stage.project_id = _PROJECT_ID
    stage.project_root = _PROJECT_ROOT
    return stage


# ---------------------------------------------------------------------------
# Fake service mocks for cite_* tools
# ---------------------------------------------------------------------------


def _build_state_with_service_mocks() -> ReconReportState:
    """Build a ReconReportState with mocked service deps for cite_* tools."""
    memory_service = AsyncMock()
    # cite_entity: returns a nodes list with one node
    memory_service.get_entity = AsyncMock(return_value={
        'nodes': [
            {'uuid': 'ent-uuid-1111-1111-111111111111', 'name': 'TaskEntity'},
        ]
    })
    # cite_edge: returns a fact dict
    memory_service.get_edge = AsyncMock(return_value={
        'fact': 'task was completed on 2026-01-01',
    })
    # cite_memory: returns a metadata fingerprint string
    memory_service.get_memory = AsyncMock(return_value='fingerprint-abc123')

    task_interceptor = AsyncMock()
    # cite_task: returns a task dict with title
    task_interceptor.get_task = AsyncMock(return_value={
        'title': 'Refactor reconciliation',
        'data': {},
    })

    state = ReconReportState(
        ttl_seconds=3600,
        memory_service=memory_service,
        task_interceptor=task_interceptor,
    )
    # Register the project so cite_task can resolve it
    state.known_projects[_PROJECT_ID] = _PROJECT_ROOT
    return state


# ---------------------------------------------------------------------------
# _FakeCliAgent — simulates the CLI agent's recon_report tool calls
# ---------------------------------------------------------------------------


def _stage1_agent(state: ReconReportState):
    """Return an async StageResult-compatible callable simulating Stage 1 agent."""

    async def _run(**_kwargs) -> StageResult:
        # Simulated Stage 1 CLI agent: add a memory-task mismatch finding,
        # cite an entity, set a stat, then complete.
        r = state.add_finding(
            run_id=_RUN_ID,
            severity='moderate',
            category='memory_task_mismatch',
            description='Memory contradicts task state',
            suggested_action='Update memory to match task',
            actionable=True,
            task_id='42',
            flag_type='memory_task_mismatch',
        )
        finding_id = r['finding_id']
        await state.cite_entity(
            run_id=_RUN_ID, finding_id=finding_id, name='TaskEntity'
        )
        state.set_stat(run_id=_RUN_ID, key='items_reviewed', value=10)
        state.complete(run_id=_RUN_ID, summary='Stage 1 complete — 1 finding')
        return StageResult(success=True)

    return _run


def _stage2_agent(state: ReconReportState):
    """Return an async StageResult-compatible callable simulating Stage 2 agent."""

    async def _run(**_kwargs) -> StageResult:
        # Normal task finding
        r1 = state.add_finding(
            run_id=_RUN_ID,
            severity='minor',
            category='missing_knowledge',
            description='Done task lacks memory',
            suggested_action='Write completion memory',
            actionable=True,
            task_id='99',
            flag_type='missing_completion_memory',
        )
        await state.cite_task(
            run_id=_RUN_ID,
            finding_id=r1['finding_id'],
            project_id=_PROJECT_ID,
            task_id='99',
        )

        # Cross-project routing finding (PRD γ §OQ5 — category-coded)
        state.add_finding(
            run_id=_RUN_ID,
            severity='moderate',
            category='cross_project_routing',
            description='Work belongs to sibling_project — orphaned edge',
            suggested_action='Route to sibling_project owner for cleanup',
            actionable=False,
            task_id=None,
            flag_type='cross_project',
        )

        state.complete(
            run_id=_RUN_ID,
            summary='Stage 2 complete — 1 task finding + 1 cross-project routing',
        )
        return StageResult(success=True)

    return _run


def _stage3_agent(state: ReconReportState):
    """Return an async StageResult-compatible callable simulating Stage 3 agent."""

    async def _run(**_kwargs) -> StageResult:
        # Finding with all four typed citation lists (PRD §9.3)
        r = state.add_finding(
            run_id=_RUN_ID,
            severity='serious',
            category='cross_store_inconsistency',
            description='Entity edge contradicts task record and a mem0 entry',
            suggested_action='Invalidate stale edge and delete duplicate memory',
            actionable=True,
            task_id='7',
            flag_type='cross_store_inconsistency',
        )
        finding_id = r['finding_id']

        await state.cite_entity(
            run_id=_RUN_ID, finding_id=finding_id, name='TaskEntity'
        )
        await state.cite_edge(
            run_id=_RUN_ID, finding_id=finding_id, edge_uuid=_EDGE_UUID
        )
        await state.cite_task(
            run_id=_RUN_ID,
            finding_id=finding_id,
            project_id=_PROJECT_ID,
            task_id='7',
        )
        await state.cite_memory(
            run_id=_RUN_ID,
            finding_id=finding_id,
            memory_id=_MEMORY_UUID,
            store='mem0',
        )

        state.set_stat(run_id=_RUN_ID, key='items_checked', value=5)
        state.complete(run_id=_RUN_ID, summary='Stage 3 complete — 1 serious inconsistency')
        return StageResult(success=True)

    return _run


# ---------------------------------------------------------------------------
# Helpers for assert_assembled_shape
# ---------------------------------------------------------------------------


def _assert_prd_93_shape(assembled: dict, stage_label: str) -> None:
    """Assert assembled report matches PRD §9.3 shape."""
    assert 'summary' in assembled, f'{stage_label}: missing summary'
    assert 'stats' in assembled, f'{stage_label}: missing stats'
    assert 'flagged_items' in assembled, f'{stage_label}: missing flagged_items'
    assert 'summary_warnings' in assembled, f'{stage_label}: missing summary_warnings'
    assert isinstance(assembled['flagged_items'], list), f'{stage_label}: flagged_items must be list'


def _assert_finding_shape(item: dict, label: str) -> None:
    """Assert a single finding dict matches PRD §9.3 shape."""
    for field in ('finding_id', 'severity', 'category', 'description',
                  'suggested_action', 'actionable', 'cited_entities',
                  'cited_edges', 'cited_tasks', 'cited_memories'):
        assert field in item, f'{label}: finding missing field {field!r}'
    # Verify no legacy affected_ids key
    assert 'affected_ids' not in item, f'{label}: finding must not contain affected_ids'


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCutoverEndToEnd:
    """End-to-end cycle: Stage 1 → Stage 2 → Stage 3 with recon_report state."""

    @pytest.mark.asyncio
    async def test_stage1_assembled_report_shape(self):
        """Stage 1 StageReport.items_flagged matches PRD §9.3 shape with typed citations."""
        state = _build_state_with_service_mocks()
        stage = _make_stub_stage(StageId.memory_consolidator, state)
        watermark = Watermark.model_construct(project_id=_PROJECT_ID)

        with patch(
            'fused_memory.reconciliation.stages.base.run_stage_via_cli',
            side_effect=_stage1_agent(state),
        ):
            report = await stage.run([], watermark, [], run_id=_RUN_ID)

        assert isinstance(report.items_flagged, list)
        assert len(report.items_flagged) == 1, (
            f'Expected 1 finding, got {len(report.items_flagged)}'
        )
        item = report.items_flagged[0]
        _assert_finding_shape(item, 'Stage1')

        # Verify entity citation populated
        assert len(item['cited_entities']) == 1
        assert item['cited_entities'][0]['canonical_name'] == 'TaskEntity'
        assert 'entity_uuid' in item['cited_entities'][0]

        # Verify top-level task_id/flag_type preserved
        assert item['task_id'] == '42'
        assert item['flag_type'] == 'memory_task_mismatch'

    @pytest.mark.asyncio
    async def test_stage2_cross_project_finding(self):
        """Stage 2 cross-project finding has category=cross_project_routing, flag_type=cross_project."""
        state = _build_state_with_service_mocks()
        stage = _make_stub_stage(StageId.task_knowledge_sync, state)
        watermark = Watermark.model_construct(project_id=_PROJECT_ID)

        with patch(
            'fused_memory.reconciliation.stages.base.run_stage_via_cli',
            side_effect=_stage2_agent(state),
        ):
            report = await stage.run([], watermark, [], run_id=_RUN_ID)

        assert len(report.items_flagged) == 2

        cross_project_items = [
            item for item in report.items_flagged
            if item.get('category') == 'cross_project_routing'
        ]
        assert len(cross_project_items) == 1, (
            f'Expected 1 cross_project_routing finding, got {len(cross_project_items)}'
        )
        cp = cross_project_items[0]
        assert cp['flag_type'] == 'cross_project'
        assert cp['actionable'] is False
        assert 'affected_ids' not in cp

    @pytest.mark.asyncio
    async def test_stage3_all_four_citation_lists(self):
        """Stage 3 finding carries all four typed citation lists with no affected_ids."""
        state = _build_state_with_service_mocks()
        stage = _make_stub_stage(StageId.integrity_check, state)
        watermark = Watermark.model_construct(project_id=_PROJECT_ID)

        with patch(
            'fused_memory.reconciliation.stages.base.run_stage_via_cli',
            side_effect=_stage3_agent(state),
        ):
            report = await stage.run([], watermark, [], run_id=_RUN_ID)

        assert len(report.items_flagged) == 1
        item = report.items_flagged[0]
        _assert_finding_shape(item, 'Stage3')

        assert len(item['cited_entities']) == 1, 'cited_entities must have 1 entry'
        assert len(item['cited_edges']) == 1, 'cited_edges must have 1 entry'
        assert len(item['cited_tasks']) == 1, 'cited_tasks must have 1 entry'
        assert len(item['cited_memories']) == 1, 'cited_memories must have 1 entry'

        # Verify shapes
        assert item['cited_edges'][0]['edge_uuid'] == _EDGE_UUID
        assert item['cited_tasks'][0]['project_id'] == _PROJECT_ID
        assert item['cited_tasks'][0]['task_id'] == '7'
        assert item['cited_memories'][0]['memory_id'] == _MEMORY_UUID
        assert item['cited_memories'][0]['store'] == 'mem0'

    @pytest.mark.asyncio
    async def test_get_assembled_report_matches_prd_93(self):
        """get_assembled_report returns summary + stats + flagged_items + summary_warnings."""
        state = _build_state_with_service_mocks()
        stage = _make_stub_stage(StageId.memory_consolidator, state)
        watermark = Watermark.model_construct(project_id=_PROJECT_ID)

        with patch(
            'fused_memory.reconciliation.stages.base.run_stage_via_cli',
            side_effect=_stage1_agent(state),
        ):
            await stage.run([], watermark, [], run_id=_RUN_ID)

        assembled = state.get_assembled_report(_RUN_ID, StageId.memory_consolidator.value)
        assert assembled is not None
        _assert_prd_93_shape(assembled, 'assembled Stage1')

        assert assembled['summary'] == 'Stage 1 complete — 1 finding'
        assert assembled['stats'].get('items_reviewed') == 10

    @pytest.mark.asyncio
    async def test_no_affected_ids_anywhere_in_stage3(self):
        """Stage 3 report contains no 'affected_ids' key at any level."""
        state = _build_state_with_service_mocks()
        stage = _make_stub_stage(StageId.integrity_check, state)
        watermark = Watermark.model_construct(project_id=_PROJECT_ID)

        with patch(
            'fused_memory.reconciliation.stages.base.run_stage_via_cli',
            side_effect=_stage3_agent(state),
        ):
            report = await stage.run([], watermark, [], run_id=_RUN_ID)

        import json
        serialized = json.dumps(report.items_flagged)
        assert 'affected_ids' not in serialized, (
            f'"affected_ids" must not appear anywhere in Stage 3 report items:\n{serialized}'
        )

    @pytest.mark.asyncio
    async def test_stage_timestamps_populated(self):
        """StageReport.started_at and completed_at are populated regardless of recon_report state."""
        state = _build_state_with_service_mocks()
        stage = _make_stub_stage(StageId.memory_consolidator, state)
        watermark = Watermark.model_construct(project_id=_PROJECT_ID)

        with patch(
            'fused_memory.reconciliation.stages.base.run_stage_via_cli',
            side_effect=_stage1_agent(state),
        ):
            report = await stage.run([], watermark, [], run_id=_RUN_ID)

        assert report.started_at is not None
        assert report.completed_at is not None
        assert report.completed_at >= report.started_at
