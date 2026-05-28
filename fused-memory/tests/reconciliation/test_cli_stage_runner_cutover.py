"""Tests for PRD γ cutover: cli_stage_runner schema migration and output_schema passthrough."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from fused_memory.reconciliation.cli_stage_runner import (
    FINDING_ITEM_SCHEMA,
    STAGE3_REPORT_SCHEMA,
    STAGE_REPORT_SCHEMA,
    _extract_report,
)


class TestFindingItemSchemaShape:
    """FINDING_ITEM_SCHEMA must carry four typed citation arrays per PRD §9.3."""

    def test_has_four_citation_arrays(self):
        """cited_entities, cited_edges, cited_tasks, cited_memories all present."""
        props = FINDING_ITEM_SCHEMA['properties']
        for key in ('cited_entities', 'cited_edges', 'cited_tasks', 'cited_memories'):
            assert key in props, f'Expected {key!r} in FINDING_ITEM_SCHEMA.properties'

    def test_no_affected_ids(self):
        """affected_ids must be removed from FINDING_ITEM_SCHEMA."""
        props = FINDING_ITEM_SCHEMA['properties']
        assert 'affected_ids' not in props, 'affected_ids must be retired from FINDING_ITEM_SCHEMA'

    def test_cited_entities_item_shape(self):
        """cited_entities items carry entity_uuid and canonical_name."""
        item_schema = FINDING_ITEM_SCHEMA['properties']['cited_entities']['items']
        props = item_schema.get('properties', {})
        assert 'entity_uuid' in props, 'cited_entities items must have entity_uuid'
        assert 'canonical_name' in props, 'cited_entities items must have canonical_name'

    def test_cited_edges_item_shape(self):
        """cited_edges items carry edge_uuid and fact_text_snapshot."""
        item_schema = FINDING_ITEM_SCHEMA['properties']['cited_edges']['items']
        props = item_schema.get('properties', {})
        assert 'edge_uuid' in props, 'cited_edges items must have edge_uuid'
        assert 'fact_text_snapshot' in props, 'cited_edges items must have fact_text_snapshot'

    def test_cited_tasks_item_shape(self):
        """cited_tasks items carry project_id, task_id, and title."""
        item_schema = FINDING_ITEM_SCHEMA['properties']['cited_tasks']['items']
        props = item_schema.get('properties', {})
        assert 'project_id' in props, 'cited_tasks items must have project_id'
        assert 'task_id' in props, 'cited_tasks items must have task_id'
        assert 'title' in props, 'cited_tasks items must have title'

    def test_cited_memories_item_shape(self):
        """cited_memories items carry memory_id, store, and metadata_fingerprint."""
        item_schema = FINDING_ITEM_SCHEMA['properties']['cited_memories']['items']
        props = item_schema.get('properties', {})
        assert 'memory_id' in props, 'cited_memories items must have memory_id'
        assert 'store' in props, 'cited_memories items must have store'
        assert 'metadata_fingerprint' in props, 'cited_memories items must have metadata_fingerprint'

    def test_description_and_severity_still_required(self):
        """description and severity remain required fields."""
        required = FINDING_ITEM_SCHEMA.get('required', [])
        assert 'description' in required, 'description must stay required'
        assert 'severity' in required, 'severity must stay required'

    def test_optional_top_level_fields_present(self):
        """task_id, flag_type, and actionable are still defined as top-level properties."""
        props = FINDING_ITEM_SCHEMA['properties']
        for key in ('task_id', 'flag_type', 'actionable'):
            assert key in props, f'Expected {key!r} still in FINDING_ITEM_SCHEMA.properties'


class TestStageReportSchemaShape:
    """STAGE_REPORT_SCHEMA must expose flagged_items, stats, summary with no affected_ids leak."""

    def test_has_required_top_level_keys(self):
        """flagged_items, stats, and summary are all present."""
        props = STAGE_REPORT_SCHEMA['properties']
        for key in ('flagged_items', 'stats', 'summary'):
            assert key in props, f'Expected {key!r} in STAGE_REPORT_SCHEMA.properties'

    def test_no_affected_ids_at_top_level(self):
        """affected_ids must not appear at the top level of STAGE_REPORT_SCHEMA."""
        props = STAGE_REPORT_SCHEMA['properties']
        assert 'affected_ids' not in props, 'affected_ids must not be at STAGE_REPORT_SCHEMA top level'


class TestStage3ReportSchemaShape:
    """STAGE3_REPORT_SCHEMA must mirror STAGE_REPORT_SCHEMA with structured finding items."""

    def test_has_required_top_level_keys(self):
        """flagged_items, stats, and summary are all present."""
        props = STAGE3_REPORT_SCHEMA['properties']
        for key in ('flagged_items', 'stats', 'summary'):
            assert key in props, f'Expected {key!r} in STAGE3_REPORT_SCHEMA.properties'

    def test_no_affected_ids_at_top_level(self):
        """affected_ids must not appear at the top level of STAGE3_REPORT_SCHEMA."""
        props = STAGE3_REPORT_SCHEMA['properties']
        assert 'affected_ids' not in props, 'affected_ids must not be at STAGE3_REPORT_SCHEMA top level'

    def test_flagged_items_uses_finding_item_schema(self):
        """STAGE3_REPORT_SCHEMA.flagged_items.items should be FINDING_ITEM_SCHEMA."""
        items_schema = STAGE3_REPORT_SCHEMA['properties']['flagged_items']['items']
        # The items schema must have the four citation arrays (same as FINDING_ITEM_SCHEMA)
        props = items_schema.get('properties', {})
        for key in ('cited_entities', 'cited_edges', 'cited_tasks', 'cited_memories'):
            assert key in props, f'STAGE3_REPORT_SCHEMA.flagged_items.items missing {key!r}'


class TestOutputSchemaNonePassthrough:
    """run_stage_via_cli must pass output_schema=None straight through (PRD γ step-14).

    When recon_report_state is active, the harness passes output_schema=None to
    run_stage_via_cli so the CLI agent can terminate with a short text turn after
    recon_report.complete instead of re-emitting a JSON report.

    RED: currently run_stage_via_cli replaces None with STAGE_REPORT_SCHEMA
    (line 255: output_schema=output_schema if output_schema is not None else STAGE_REPORT_SCHEMA).
    Step-14 changes this so None is passed straight through.
    """

    @pytest.mark.asyncio
    async def test_none_schema_reaches_invoke_with_cap_retry(self):
        """Passing output_schema=None must send output_schema=None to invoke_with_cap_retry.

        RED: Currently, None is replaced by STAGE_REPORT_SCHEMA before invoke.
        """
        from fused_memory.config.schema import ReconciliationConfig
        from fused_memory.reconciliation.cli_stage_runner import run_stage_via_cli

        captured_kwargs: dict = {}

        async def _capture_invoke(**kwargs):
            captured_kwargs.update(kwargs)
            # Return a minimal AgentResult
            result = MagicMock()
            result.structured_output = None
            result.output = None
            result.turns = 0
            result.cost_usd = 0.0
            result.success = True
            return result

        config = ReconciliationConfig()
        mcp_config = {'mcpServers': {}}

        with patch(
            'fused_memory.reconciliation.cli_stage_runner.invoke_with_cap_retry',
            new=_capture_invoke,
        ):
            await run_stage_via_cli(
                system_prompt='test',
                payload='test payload',
                disallowed_tools=[],
                config=config,
                mcp_config=mcp_config,
                output_schema=None,
            )

        assert captured_kwargs.get('output_schema') is None, (
            'output_schema=None must be passed straight through to invoke_with_cap_retry, '
            'not replaced by STAGE_REPORT_SCHEMA'
        )

    def test_extract_report_returns_empty_dict_when_no_output(self):
        """_extract_report must return {} (not a placeholder) when structured_output and output are empty.

        RED: Currently _extract_report returns {'summary': 'No output from agent'} as a placeholder.
        After step-14, it returns {} so BaseStage.run gets an empty dict that triggers the
        recon_report fallback path.
        """
        from shared.cli_invoke import AgentResult

        result = AgentResult(
            output='',
            structured_output=None,
            turns=0,
            cost_usd=0.0,
            success=True,
        )
        report = _extract_report(result)
        assert report == {}, (
            f'_extract_report must return {{}} when output is empty, got {report!r}'
        )
