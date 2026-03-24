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


class TestProjectIdValidation:
    """BaseStage.run() validates project_id before executing."""

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

    @pytest.mark.asyncio
    async def test_run_raises_on_empty_project_id(self, mock_deps):
        from fused_memory.models.reconciliation import StageId, Watermark
        stage = MemoryConsolidator(StageId.memory_consolidator, **mock_deps)
        # project_id defaults to '' — should raise before executing
        watermark = Watermark()
        with pytest.raises(ValueError, match='project_id'):
            await stage.run(
                events=[],
                watermark=watermark,
                prior_reports=[],
                run_id='test-run-001',
            )

    @pytest.mark.asyncio
    async def test_run_raises_on_whitespace_project_id(self, mock_deps):
        from fused_memory.models.reconciliation import StageId, Watermark
        stage = MemoryConsolidator(StageId.memory_consolidator, **mock_deps)
        # Whitespace-only project_id is truthy but semantically empty
        stage.project_id = '   '
        watermark = Watermark(project_id='dark_factory')
        with pytest.raises(ValueError, match='project_id'):
            await stage.run(
                events=[],
                watermark=watermark,
                prior_reports=[],
                run_id='test-run-001',
            )

    @pytest.mark.asyncio
    async def test_run_raises_on_watermark_project_id_mismatch(self, mock_deps):
        from fused_memory.models.reconciliation import StageId, Watermark
        stage = MemoryConsolidator(StageId.memory_consolidator, **mock_deps)
        # stage and watermark have contradictory project_ids
        stage.project_id = 'project_a'
        watermark = Watermark(project_id='project_b')
        with pytest.raises(ValueError, match='project_a'):
            await stage.run(
                events=[],
                watermark=watermark,
                prior_reports=[],
                run_id='test-run-001',
            )

    @pytest.mark.asyncio
    async def test_run_allows_matching_watermark_project_id(self, mock_deps):
        from unittest.mock import patch

        from fused_memory.models.reconciliation import StageId, Watermark
        stage = MemoryConsolidator(StageId.memory_consolidator, **mock_deps)
        stage.project_id = 'dark_factory'
        watermark = Watermark(project_id='dark_factory')

        async def fake_assemble_payload(events, wm, prior_reports):
            return '## Base Payload\nsome context'

        async def fake_run_stage_via_cli(**kwargs):
            from fused_memory.reconciliation.cli_stage_runner import StageResult
            return StageResult(success=True, report={'summary': 'ok'})

        with (
            patch.object(stage, 'assemble_payload', new=fake_assemble_payload),
            patch(
                'fused_memory.reconciliation.stages.base.run_stage_via_cli',
                new=fake_run_stage_via_cli,
            ),
        ):
            # Should not raise — matching project_ids are accepted
            await stage.run(
                events=[],
                watermark=watermark,
                prior_reports=[],
                run_id='test-run-001',
            )

    @pytest.mark.asyncio
    async def test_run_allows_none_watermark_project_id(self, mock_deps):
        from unittest.mock import patch

        from fused_memory.models.reconciliation import StageId, Watermark
        stage = MemoryConsolidator(StageId.memory_consolidator, **mock_deps)
        stage.project_id = 'dark_factory'
        # None watermark project_id skips the mismatch check
        watermark = Watermark()

        async def fake_assemble_payload(events, wm, prior_reports):
            return '## Base Payload\nsome context'

        async def fake_run_stage_via_cli(**kwargs):
            from fused_memory.reconciliation.cli_stage_runner import StageResult
            return StageResult(success=True, report={'summary': 'ok'})

        with (
            patch.object(stage, 'assemble_payload', new=fake_assemble_payload),
            patch(
                'fused_memory.reconciliation.stages.base.run_stage_via_cli',
                new=fake_run_stage_via_cli,
            ),
        ):
            # Should not raise — None watermark project_id bypasses mismatch check
            await stage.run(
                events=[],
                watermark=watermark,
                prior_reports=[],
                run_id='test-run-001',
            )

    @pytest.mark.asyncio
    async def test_run_allows_whitespace_only_watermark_project_id(self, mock_deps):
        from unittest.mock import patch

        from fused_memory.models.reconciliation import StageId, Watermark
        stage = MemoryConsolidator(StageId.memory_consolidator, **mock_deps)
        stage.project_id = 'dark_factory'
        # Whitespace-only watermark project_id normalizes to None and bypasses mismatch check
        watermark = Watermark(project_id='   ')

        async def fake_assemble_payload(events, wm, prior_reports):
            return '## Base Payload\nsome context'

        async def fake_run_stage_via_cli(**kwargs):
            from fused_memory.reconciliation.cli_stage_runner import StageResult
            return StageResult(success=True, report={'summary': 'ok'})

        with (
            patch.object(stage, 'assemble_payload', new=fake_assemble_payload),
            patch(
                'fused_memory.reconciliation.stages.base.run_stage_via_cli',
                new=fake_run_stage_via_cli,
            ),
        ):
            # Should not raise — whitespace-only watermark project_id normalizes to None
            await stage.run(
                events=[],
                watermark=watermark,
                prior_reports=[],
                run_id='test-run-001',
            )

    @pytest.mark.asyncio
    async def test_recon_context_includes_project_id(self, mock_deps):
        from unittest.mock import patch

        from fused_memory.models.reconciliation import StageId, Watermark
        stage = MemoryConsolidator(StageId.memory_consolidator, **mock_deps)
        stage.project_id = 'dark_factory'
        watermark = Watermark(project_id='dark_factory')

        captured_payload = {}

        async def fake_assemble_payload(events, wm, prior_reports):
            return '## Base Payload\nsome context'

        async def fake_run_stage_via_cli(**kwargs):
            captured_payload['payload'] = kwargs.get('payload', '')
            from fused_memory.reconciliation.cli_stage_runner import StageResult
            return StageResult(
                success=True,
                report={'summary': 'ok'},
            )

        with (
            patch.object(stage, 'assemble_payload', new=fake_assemble_payload),
            patch(
                'fused_memory.reconciliation.stages.base.run_stage_via_cli',
                new=fake_run_stage_via_cli,
            ),
        ):
            await stage.run(
                events=[],
                watermark=watermark,
                prior_reports=[],
                run_id='test-run-001',
            )

        payload = captured_payload.get('payload', '')
        assert 'project_id' in payload, 'recon_context must mention project_id'
        assert 'dark_factory' in payload, 'recon_context must include the actual project_id value'


class TestSystemPromptsProjectId:
    """All three stage system prompts mention project_id in their Guidelines."""

    def test_stage1_prompt_mentions_project_id(self):
        from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT
        assert 'project_id' in STAGE1_SYSTEM_PROMPT, (
            'STAGE1_SYSTEM_PROMPT must instruct the agent to include project_id in MCP calls'
        )

    def test_stage2_prompt_mentions_project_id(self):
        from fused_memory.reconciliation.prompts.stage2 import STAGE2_SYSTEM_PROMPT
        assert 'project_id' in STAGE2_SYSTEM_PROMPT, (
            'STAGE2_SYSTEM_PROMPT must instruct the agent to include project_id in MCP calls'
        )

    def test_stage3_prompt_mentions_project_id(self):
        from fused_memory.reconciliation.prompts.stage3 import STAGE3_SYSTEM_PROMPT
        assert 'project_id' in STAGE3_SYSTEM_PROMPT, (
            'STAGE3_SYSTEM_PROMPT must instruct the agent to include project_id in MCP calls'
        )


class TestPayloadProjectId:
    """Regression tests: assemble_payload output contains project_id for all stages."""

    @pytest.fixture
    def mock_deps(self):
        from unittest.mock import AsyncMock, MagicMock

        from fused_memory.config.schema import ReconciliationConfig
        config = ReconciliationConfig(enabled=True, explore_codebase_root='/tmp/test')

        # memory_service mock with nested mem0
        memory_service = AsyncMock()
        memory_service.get_episodes.return_value = []
        memory_service.get_status.return_value = {}
        mem0_mock = MagicMock()
        mem0_mock.get_all = AsyncMock(return_value={'results': []})
        memory_service.mem0 = mem0_mock

        taskmaster = AsyncMock()
        taskmaster.get_tasks.return_value = {}

        return {
            'memory_service': memory_service,
            'taskmaster': taskmaster,
            'journal': AsyncMock(),
            'config': config,
        }

    @pytest.mark.asyncio
    async def test_stage1_payload_contains_project_id(self, mock_deps):
        from fused_memory.models.reconciliation import StageId, Watermark
        stage = MemoryConsolidator(StageId.memory_consolidator, **mock_deps)
        stage.project_id = 'dark_factory'
        watermark = Watermark(project_id='dark_factory')
        payload = await stage.assemble_payload(events=[], watermark=watermark, prior_reports=[])
        assert 'dark_factory' in payload, 'Stage 1 payload must contain project_id value'
        assert 'project_id' in payload, 'Stage 1 payload must mention project_id'

    @pytest.mark.asyncio
    async def test_stage1_remediation_payload_contains_project_id(self, mock_deps):
        from fused_memory.models.reconciliation import StageId, Watermark
        stage = MemoryConsolidator(StageId.memory_consolidator, **mock_deps)
        stage.project_id = 'dark_factory'
        stage.remediation_findings = []
        watermark = Watermark(project_id='dark_factory')
        payload = await stage.assemble_payload(events=[], watermark=watermark, prior_reports=[])
        assert 'dark_factory' in payload, 'Stage 1 remediation payload must contain project_id value'
        assert 'project_id' in payload, 'Stage 1 remediation payload must mention project_id'

    @pytest.mark.asyncio
    async def test_stage2_payload_contains_project_id(self, mock_deps):
        from fused_memory.models.reconciliation import StageId, Watermark
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'dark_factory'
        watermark = Watermark(project_id='dark_factory')
        payload = await stage.assemble_payload(events=[], watermark=watermark, prior_reports=[])
        assert 'dark_factory' in payload, 'Stage 2 payload must contain project_id value'
        assert 'project_id' in payload, 'Stage 2 payload must mention project_id'

    @pytest.mark.asyncio
    async def test_stage3_payload_contains_project_id(self, mock_deps):
        from fused_memory.models.reconciliation import StageId, Watermark
        stage = IntegrityCheck(StageId.integrity_check, **mock_deps)
        stage.project_id = 'dark_factory'
        watermark = Watermark(project_id='dark_factory')
        payload = await stage.assemble_payload(events=[], watermark=watermark, prior_reports=[])
        assert 'dark_factory' in payload, 'Stage 3 payload must contain project_id value'
        assert 'project_id' in payload, 'Stage 3 payload must mention project_id'


class TestPaddedStageProjectId:
    """BaseStage.run() strips whitespace from stage.project_id before validation."""

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

    @pytest.mark.asyncio
    async def test_run_strips_padded_stage_project_id(self, mock_deps):
        from unittest.mock import patch

        from fused_memory.models.reconciliation import StageId, Watermark
        stage = MemoryConsolidator(StageId.memory_consolidator, **mock_deps)
        stage.project_id = 'dark_factory  '
        watermark = Watermark(project_id='dark_factory')

        async def fake_assemble_payload(events, wm, prior_reports):
            return '## Base Payload\nsome context'

        async def fake_run_stage_via_cli(**kwargs):
            from fused_memory.reconciliation.cli_stage_runner import StageResult
            return StageResult(success=True, report={'summary': 'ok'})

        with (
            patch.object(stage, 'assemble_payload', new=fake_assemble_payload),
            patch(
                'fused_memory.reconciliation.stages.base.run_stage_via_cli',
                new=fake_run_stage_via_cli,
            ),
        ):
            # Should not raise — padded stage project_id strips to 'dark_factory'
            await stage.run(
                events=[],
                watermark=watermark,
                prior_reports=[],
                run_id='test-run-001',
            )

    @pytest.mark.asyncio
    async def test_run_uses_stripped_project_id_in_payload(self, mock_deps):
        from unittest.mock import patch

        from fused_memory.models.reconciliation import StageId, Watermark
        stage = MemoryConsolidator(StageId.memory_consolidator, **mock_deps)
        stage.project_id = '  dark_factory  '
        watermark = Watermark(project_id='dark_factory')

        captured = {}

        async def fake_assemble_payload(events, wm, prior_reports):
            return '## Base Payload\nsome context'

        async def fake_run_stage_via_cli(**kwargs):
            captured['payload'] = kwargs.get('payload', '')
            from fused_memory.reconciliation.cli_stage_runner import StageResult
            return StageResult(success=True, report={'summary': 'ok'})

        with (
            patch.object(stage, 'assemble_payload', new=fake_assemble_payload),
            patch(
                'fused_memory.reconciliation.stages.base.run_stage_via_cli',
                new=fake_run_stage_via_cli,
            ),
        ):
            await stage.run(
                events=[],
                watermark=watermark,
                prior_reports=[],
                run_id='test-run-001',
            )

        payload = captured.get('payload', '')
        # Stripped value must appear in payload; padded value must not
        assert 'dark_factory' in payload
        assert '  dark_factory  ' not in payload


class TestWatermarkProjectIdNormalization:
    """Watermark.project_id field_validator normalizes whitespace and converts empty to None."""

    def test_watermark_default_project_id_is_none(self):
        from fused_memory.models.reconciliation import Watermark
        wm = Watermark()
        assert wm.project_id is None

    def test_watermark_empty_string_normalizes_to_none(self):
        from fused_memory.models.reconciliation import Watermark
        wm = Watermark(project_id='')
        assert wm.project_id is None

    def test_watermark_whitespace_only_normalizes_to_none(self):
        from fused_memory.models.reconciliation import Watermark
        wm = Watermark(project_id='   ')
        assert wm.project_id is None

    def test_watermark_valid_project_id_unchanged(self):
        from fused_memory.models.reconciliation import Watermark
        wm = Watermark(project_id='dark_factory')
        assert wm.project_id == 'dark_factory'

    def test_watermark_padded_project_id_stripped(self):
        from fused_memory.models.reconciliation import Watermark
        wm = Watermark(project_id=' dark_factory ')
        assert wm.project_id == 'dark_factory'

    def test_watermark_rejects_non_string_project_id(self):
        from pydantic import ValidationError

        from fused_memory.models.reconciliation import Watermark
        with pytest.raises(ValidationError):
            Watermark(project_id=123)  # type: ignore[arg-type]


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
