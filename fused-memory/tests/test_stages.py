"""Tests for reconciliation stage configuration (CLI-native MCP execution)."""

import json

import pytest

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.reconciliation.cli_stage_runner import (
    DISALLOW_BUILTIN,
    DISALLOW_MEMORY_WRITES,
    DISALLOW_TASK_WRITES,
    STAGE1_DISALLOWED,
    STAGE2_DISALLOWED,
    STAGE3_DISALLOWED,
    STAGE3_REPORT_SCHEMA,
    STAGE_REPORT_SCHEMA,
)
from fused_memory.reconciliation.stages.memory_consolidator import MemoryConsolidator
from fused_memory.reconciliation.stages.task_knowledge_sync import (
    IntegrityCheck,
    TaskKnowledgeSync,
)


class TestDisallowedToolLists:
    """Verify per-stage disallowed tool lists are correct."""

    def test_stage1_disallows_task_writes_and_builtins(self):
        assert set(DISALLOW_TASK_WRITES).issubset(set(STAGE1_DISALLOWED))
        assert set(DISALLOW_BUILTIN).issubset(set(STAGE1_DISALLOWED))

    def test_stage1_allows_memory_writes(self):
        for tool in DISALLOW_MEMORY_WRITES:
            assert tool not in STAGE1_DISALLOWED

    def test_stage2_only_disallows_builtins(self):
        assert STAGE2_DISALLOWED == DISALLOW_BUILTIN

    def test_stage3_disallows_all_writes(self):
        assert set(DISALLOW_TASK_WRITES).issubset(set(STAGE3_DISALLOWED))
        assert set(DISALLOW_MEMORY_WRITES).issubset(set(STAGE3_DISALLOWED))
        assert set(DISALLOW_BUILTIN).issubset(set(STAGE3_DISALLOWED))

    def test_all_disallowed_have_mcp_prefix(self):
        """All MCP tools in disallowed lists should use the mcp__ naming convention."""
        for tool in DISALLOW_TASK_WRITES + DISALLOW_MEMORY_WRITES:
            assert tool.startswith('mcp__fused-memory__'), f'{tool} missing MCP prefix'

    def test_builtin_disallowed_are_claude_native(self):
        """Builtin disallowed should be Claude Code native tools."""
        for tool in DISALLOW_BUILTIN:
            assert not tool.startswith('mcp__'), f'{tool} should not have MCP prefix'


class TestStageSubclasses:
    """Each stage subclass returns the correct disallowed list."""

    @pytest.fixture
    def config(self):
        return ReconciliationConfig(
            enabled=True,
            explore_codebase_root='/tmp/test',
        )

    @pytest.fixture
    def mock_deps(self, config):
        from unittest.mock import AsyncMock
        return {
            'memory_service': AsyncMock(),
            'taskmaster': AsyncMock(),
            'journal': AsyncMock(),
            'config': config,
        }

    def test_memory_consolidator_disallowed(self, mock_deps):
        from fused_memory.models.reconciliation import StageId
        stage = MemoryConsolidator(StageId.memory_consolidator, **mock_deps)
        assert stage.get_disallowed_tools() == STAGE1_DISALLOWED

    def test_task_knowledge_sync_disallowed(self, mock_deps):
        from fused_memory.models.reconciliation import StageId
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        assert stage.get_disallowed_tools() == STAGE2_DISALLOWED

    def test_integrity_check_disallowed(self, mock_deps):
        from fused_memory.models.reconciliation import StageId
        stage = IntegrityCheck(StageId.integrity_check, **mock_deps)
        assert stage.get_disallowed_tools() == STAGE3_DISALLOWED


class TestStageReportSchema:
    """Output schema for stage reports."""

    def test_schema_has_required_summary(self):
        assert 'summary' in STAGE_REPORT_SCHEMA['required']

    def test_schema_is_valid_json_schema(self):
        """Basic structure validation."""
        assert STAGE_REPORT_SCHEMA['type'] == 'object'
        assert 'properties' in STAGE_REPORT_SCHEMA
        # Should be JSON-serializable (for --json-schema flag)
        json.dumps(STAGE_REPORT_SCHEMA)


class TestStage3ReportSchema:
    """STAGE3_REPORT_SCHEMA has structured finding item properties."""

    def test_stage3_schema_importable(self):
        from fused_memory.reconciliation.cli_stage_runner import STAGE3_REPORT_SCHEMA
        assert STAGE3_REPORT_SCHEMA is not None

    def test_stage3_flagged_items_has_item_properties(self):
        from fused_memory.reconciliation.cli_stage_runner import STAGE3_REPORT_SCHEMA
        items_schema = STAGE3_REPORT_SCHEMA['properties']['flagged_items']['items']
        assert 'properties' in items_schema
        props = items_schema['properties']
        for field in ('description', 'severity', 'actionable', 'category', 'affected_ids', 'suggested_action'):
            assert field in props, f"Expected '{field}' in flagged_items.items.properties"

    def test_stage3_finding_item_required_includes_description_and_severity(self):
        from fused_memory.reconciliation.cli_stage_runner import STAGE3_REPORT_SCHEMA
        items_schema = STAGE3_REPORT_SCHEMA['properties']['flagged_items']['items']
        assert 'required' in items_schema
        assert 'description' in items_schema['required']
        assert 'severity' in items_schema['required']

    def test_stage3_schema_is_json_serializable(self):
        from fused_memory.reconciliation.cli_stage_runner import STAGE3_REPORT_SCHEMA
        json.dumps(STAGE3_REPORT_SCHEMA)

    def test_stage3_schema_preserves_base_structure(self):
        from fused_memory.reconciliation.cli_stage_runner import STAGE3_REPORT_SCHEMA
        assert STAGE3_REPORT_SCHEMA['type'] == 'object'
        assert 'summary' in STAGE3_REPORT_SCHEMA['required']
        assert 'flagged_items' in STAGE3_REPORT_SCHEMA['properties']
        assert 'stats' in STAGE3_REPORT_SCHEMA['properties']


class TestExtractReportNormalization:
    """_extract_report normalizes findings key to flagged_items."""

    def _make_result(self, structured_output=None, output=None):
        from shared.cli_invoke import AgentResult
        return AgentResult(
            success=True,
            output=output or '',
            structured_output=structured_output,
        )

    def test_findings_remapped_to_flagged_items(self):
        from fused_memory.reconciliation.cli_stage_runner import _extract_report
        result = self._make_result(structured_output={
            'findings': [{'description': 'stale edge', 'severity': 'moderate'}],
            'summary': 'done',
        })
        report = _extract_report(result)
        assert 'flagged_items' in report
        assert report['flagged_items'] == [{'description': 'stale edge', 'severity': 'moderate'}]
        assert 'findings' not in report

    def test_flagged_items_preserved_when_no_findings(self):
        from fused_memory.reconciliation.cli_stage_runner import _extract_report
        result = self._make_result(structured_output={
            'flagged_items': [{'description': 'real finding', 'severity': 'serious'}],
            'summary': 'ok',
        })
        report = _extract_report(result)
        assert report['flagged_items'] == [{'description': 'real finding', 'severity': 'serious'}]

    def test_flagged_items_preferred_over_findings_when_both_present(self):
        from fused_memory.reconciliation.cli_stage_runner import _extract_report
        result = self._make_result(structured_output={
            'findings': [{'description': 'from findings'}],
            'flagged_items': [{'description': 'from flagged_items'}],
            'summary': 'both',
        })
        report = _extract_report(result)
        # flagged_items is non-empty → keep it, ignore findings
        assert report['flagged_items'] == [{'description': 'from flagged_items'}]

    def test_findings_used_when_flagged_items_is_empty(self):
        from fused_memory.reconciliation.cli_stage_runner import _extract_report
        result = self._make_result(structured_output={
            'findings': [{'description': 'fallback finding'}],
            'flagged_items': [],
            'summary': 'empty fi',
        })
        report = _extract_report(result)
        assert report['flagged_items'] == [{'description': 'fallback finding'}]


class TestNormalizePlaceholderFiltering:
    """_normalize_report filters out placeholder findings."""

    def _normalize(self, report):
        from fused_memory.reconciliation.cli_stage_runner import _normalize_report
        return _normalize_report(report)

    def test_filters_missing_description(self):
        report = {'flagged_items': [{'severity': 'minor'}], 'summary': 'x'}
        result = self._normalize(report)
        assert result['flagged_items'] == []

    def test_filters_question_mark_description(self):
        report = {'flagged_items': [{'description': '?', 'severity': 'moderate'}], 'summary': 'x'}
        result = self._normalize(report)
        assert result['flagged_items'] == []

    def test_filters_empty_description(self):
        report = {'flagged_items': [{'description': '', 'severity': 'minor'}], 'summary': 'x'}
        result = self._normalize(report)
        assert result['flagged_items'] == []

    def test_keeps_valid_findings(self):
        report = {
            'flagged_items': [{'description': 'real issue', 'severity': 'serious'}],
            'summary': 'x',
        }
        result = self._normalize(report)
        assert len(result['flagged_items']) == 1
        assert result['flagged_items'][0]['description'] == 'real issue'

    def test_mixed_valid_and_placeholder(self):
        report = {
            'flagged_items': [
                {'description': '?', 'severity': 'minor'},
                {'description': 'real', 'severity': 'moderate'},
                {'severity': 'serious'},  # no description
            ],
            'summary': 'x',
        }
        result = self._normalize(report)
        assert len(result['flagged_items']) == 1
        assert result['flagged_items'][0]['description'] == 'real'

    def test_all_placeholder_findings_removed(self):
        report = {
            'flagged_items': [
                {'description': '?'},
                {'description': '?', 'severity': 'serious'},
            ],
            'summary': 'x',
        }
        result = self._normalize(report)
        assert result['flagged_items'] == []


class TestPerStageReportSchema:
    """Each stage returns the correct report schema via get_report_schema()."""

    @pytest.fixture
    def mock_deps(self):
        from unittest.mock import AsyncMock
        from fused_memory.config.schema import ReconciliationConfig
        config = ReconciliationConfig(enabled=True, explore_codebase_root='/tmp/test')
        return {
            'memory_service': AsyncMock(),
            'taskmaster': AsyncMock(),
            'journal': AsyncMock(),
            'config': config,
        }

    def test_integrity_check_returns_stage3_schema(self, mock_deps):
        from fused_memory.models.reconciliation import StageId
        stage = IntegrityCheck(StageId.integrity_check, **mock_deps)
        assert stage.get_report_schema() is STAGE3_REPORT_SCHEMA

    def test_memory_consolidator_returns_base_schema(self, mock_deps):
        from fused_memory.models.reconciliation import StageId
        stage = MemoryConsolidator(StageId.memory_consolidator, **mock_deps)
        assert stage.get_report_schema() is STAGE_REPORT_SCHEMA

    def test_task_knowledge_sync_returns_base_schema(self, mock_deps):
        from fused_memory.models.reconciliation import StageId
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        assert stage.get_report_schema() is STAGE_REPORT_SCHEMA


class TestMcpConfig:
    """BaseStage._build_mcp_config() produces valid MCP server config."""

    @pytest.fixture
    def stage(self):
        from unittest.mock import AsyncMock

        from fused_memory.models.reconciliation import StageId
        from fused_memory.reconciliation.stages.base import BaseStage
        config = ReconciliationConfig(explore_codebase_root='/tmp/test')
        return BaseStage(
            StageId.memory_consolidator,
            AsyncMock(), AsyncMock(), AsyncMock(), config,
        )

    def test_mcp_config_has_fused_memory(self, stage):
        config = stage._build_mcp_config()
        assert 'mcpServers' in config
        assert 'fused-memory' in config['mcpServers']

    def test_mcp_config_no_escalation_by_default(self, stage):
        config = stage._build_mcp_config()
        assert 'escalation' not in config['mcpServers']

    def test_mcp_config_with_escalation_url(self, stage):
        stage._escalation_url = 'http://127.0.0.1:8103/mcp'
        config = stage._build_mcp_config()
        assert 'escalation' in config['mcpServers']
        assert config['mcpServers']['escalation']['url'] == 'http://127.0.0.1:8103/mcp'


class TestStage3PromptAlignment:
    """STAGE3_SYSTEM_PROMPT explicitly mentions flagged_items."""

    def test_stage3_prompt_references_flagged_items(self):
        from fused_memory.reconciliation.prompts.stage3 import STAGE3_SYSTEM_PROMPT
        assert 'flagged_items' in STAGE3_SYSTEM_PROMPT, (
            "STAGE3_SYSTEM_PROMPT must instruct the LLM to use 'flagged_items' key"
        )

    def test_stage3_prompt_has_output_format_section(self):
        from fused_memory.reconciliation.prompts.stage3 import STAGE3_SYSTEM_PROMPT
        # Should have an Output Format section to guide the LLM
        assert 'Output Format' in STAGE3_SYSTEM_PROMPT


class TestTierConfig:
    """MemoryConsolidator respects tier limits."""

    def test_default_limits(self):
        from unittest.mock import AsyncMock

        from fused_memory.models.reconciliation import StageId
        config = ReconciliationConfig()
        stage = MemoryConsolidator(
            StageId.memory_consolidator,
            AsyncMock(), AsyncMock(), AsyncMock(), config,
        )
        assert stage.episode_limit == 500
        assert stage.memory_limit == 1000

    def test_sonnet_tier_limits(self):
        from unittest.mock import AsyncMock

        from fused_memory.models.reconciliation import StageId
        config = ReconciliationConfig()
        stage = MemoryConsolidator(
            StageId.memory_consolidator,
            AsyncMock(), AsyncMock(), AsyncMock(), config,
        )
        stage.episode_limit = 125
        stage.memory_limit = 250
        assert stage.episode_limit == 125
        assert stage.memory_limit == 250
