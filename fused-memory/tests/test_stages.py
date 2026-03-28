"""Tests for reconciliation stage configuration (CLI-native MCP execution)."""

import json
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from shared.cli_invoke import AgentResult

import fused_memory.reconciliation.stages.base as base_module
from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import StageId, Watermark
from fused_memory.reconciliation.cli_stage_runner import (
    DISALLOW_BUILTIN,
    DISALLOW_MEMORY_WRITES,
    DISALLOW_TASK_WRITES,
    STAGE1_DISALLOWED,
    STAGE2_DISALLOWED,
    STAGE3_DISALLOWED,
    STAGE3_REPORT_SCHEMA,
    STAGE_REPORT_SCHEMA,
    StageResult,
    _extract_report,
    _normalize_report,
)
from fused_memory.reconciliation.prompts.stage3 import STAGE3_SYSTEM_PROMPT
from fused_memory.reconciliation.stages.base import BaseStage
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
        return {
            'memory_service': AsyncMock(),
            'taskmaster': AsyncMock(),
            'journal': AsyncMock(),
            'config': config,
        }

    def test_memory_consolidator_disallowed(self, mock_deps):
        stage = MemoryConsolidator(StageId.memory_consolidator, **mock_deps)
        assert stage.get_disallowed_tools() == STAGE1_DISALLOWED

    def test_task_knowledge_sync_disallowed(self, mock_deps):
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        assert stage.get_disallowed_tools() == STAGE2_DISALLOWED

    def test_integrity_check_disallowed(self, mock_deps):
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
        assert STAGE3_REPORT_SCHEMA is not None

    def test_stage3_flagged_items_has_item_properties(self):
        items_schema = STAGE3_REPORT_SCHEMA['properties']['flagged_items']['items']
        assert 'properties' in items_schema
        props = items_schema['properties']
        for field in ('description', 'severity', 'actionable', 'category', 'affected_ids', 'suggested_action'):
            assert field in props, f"Expected '{field}' in flagged_items.items.properties"

    def test_stage3_finding_item_required_includes_description_and_severity(self):
        items_schema = STAGE3_REPORT_SCHEMA['properties']['flagged_items']['items']
        assert 'required' in items_schema
        assert 'description' in items_schema['required']
        assert 'severity' in items_schema['required']

    def test_stage3_schema_is_json_serializable(self):
        json.dumps(STAGE3_REPORT_SCHEMA)

    def test_stage3_schema_preserves_base_structure(self):
        assert STAGE3_REPORT_SCHEMA['type'] == 'object'
        assert 'summary' in STAGE3_REPORT_SCHEMA['required']
        assert 'flagged_items' in STAGE3_REPORT_SCHEMA['properties']
        assert 'stats' in STAGE3_REPORT_SCHEMA['properties']


class TestExtractReportNormalization:
    """_extract_report normalizes findings key to flagged_items."""

    def _make_result(self, structured_output=None, output=None):
        return AgentResult(
            success=True,
            output=output or '',
            structured_output=structured_output,
        )

    def test_findings_remapped_to_flagged_items(self):
        result = self._make_result(structured_output={
            'findings': [{'description': 'stale edge', 'severity': 'moderate'}],
            'summary': 'done',
        })
        report = _extract_report(result)
        assert 'flagged_items' in report
        assert report['flagged_items'] == [{'description': 'stale edge', 'severity': 'moderate'}]
        assert 'findings' not in report

    def test_flagged_items_preserved_when_no_findings(self):
        result = self._make_result(structured_output={
            'flagged_items': [{'description': 'real finding', 'severity': 'serious'}],
            'summary': 'ok',
        })
        report = _extract_report(result)
        assert report['flagged_items'] == [{'description': 'real finding', 'severity': 'serious'}]

    def test_flagged_items_preferred_over_findings_when_both_present(self):
        result = self._make_result(structured_output={
            'findings': [{'description': 'from findings'}],
            'flagged_items': [{'description': 'from flagged_items'}],
            'summary': 'both',
        })
        report = _extract_report(result)
        # flagged_items is non-empty → keep it, ignore findings
        assert report['flagged_items'] == [{'description': 'from flagged_items'}]

    def test_findings_used_when_flagged_items_is_empty(self):
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
        config = ReconciliationConfig(enabled=True, explore_codebase_root='/tmp/test')
        return {
            'memory_service': AsyncMock(),
            'taskmaster': AsyncMock(),
            'journal': AsyncMock(),
            'config': config,
        }

    def test_integrity_check_returns_stage3_schema(self, mock_deps):
        stage = IntegrityCheck(StageId.integrity_check, **mock_deps)
        assert stage.get_report_schema() is STAGE3_REPORT_SCHEMA

    def test_memory_consolidator_returns_base_schema(self, mock_deps):
        stage = MemoryConsolidator(StageId.memory_consolidator, **mock_deps)
        assert stage.get_report_schema() is STAGE_REPORT_SCHEMA

    def test_task_knowledge_sync_returns_base_schema(self, mock_deps):
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        assert stage.get_report_schema() is STAGE_REPORT_SCHEMA


class TestMcpConfig:
    """BaseStage._build_mcp_config() produces valid MCP server config."""

    @pytest.fixture
    def stage(self):
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
        assert 'flagged_items' in STAGE3_SYSTEM_PROMPT, (
            "STAGE3_SYSTEM_PROMPT must instruct the LLM to use 'flagged_items' key"
        )

    def test_stage3_prompt_has_output_format_section(self):
        # Should have an Output Format section to guide the LLM
        assert 'Output Format' in STAGE3_SYSTEM_PROMPT


class TestTaskKnowledgeSyncPayload:
    """TaskKnowledgeSync.assemble_payload() uses correct project attributes."""

    @pytest.fixture
    def mock_deps(self):
        config = ReconciliationConfig(enabled=True, explore_codebase_root='/tmp/test')
        return {
            'memory_service': AsyncMock(),
            'taskmaster': AsyncMock(),
            'journal': AsyncMock(),
            'config': config,
        }

    @pytest.fixture
    def watermark(self):
        return Watermark(project_id='reify')

    @pytest.mark.asyncio
    async def test_get_tasks_uses_project_root_not_project_id(self, mock_deps, watermark):
        """assemble_payload() must pass self.project_root (not self.project_id) to get_tasks."""
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'reify'
        stage.project_root = '/home/leo/src/reify'
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}

        await stage.assemble_payload([], watermark, [])

        mock_deps['taskmaster'].get_tasks.assert_called_once_with(
            project_root='/home/leo/src/reify'
        )

    @pytest.mark.asyncio
    async def test_payload_uses_dynamic_project_root_in_instructions(self, mock_deps, watermark):
        """assemble_payload() instruction text must use self.project_root, not hardcoded path."""
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'reify'
        stage.project_root = '/home/leo/src/reify'
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}

        payload = await stage.assemble_payload([], watermark, [])

        assert 'project_root="/home/leo/src/reify"' in payload
        assert 'project_root="/home/leo/src/dark-factory"' not in payload

    @pytest.mark.asyncio
    async def test_payload_dark_factory_project_still_works(self, mock_deps, watermark):
        """When project_root IS dark-factory, payload still contains the correct path."""
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'dark_factory'
        stage.project_root = '/home/leo/src/dark-factory'
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}
        wm = Watermark(project_id='dark_factory')

        payload = await stage.assemble_payload([], wm, [])

        assert 'project_root="/home/leo/src/dark-factory"' in payload

    @pytest.mark.asyncio
    async def test_payload_contains_project_id_for_memory_tools(self, mock_deps, watermark):
        """assemble_payload() instruction text still uses self.project_id for fused-memory calls."""
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'reify'
        stage.project_root = '/home/leo/src/reify'
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}

        payload = await stage.assemble_payload([], watermark, [])

        # The project_id should appear in the memory tools instruction (line 98)
        assert 'project_id="reify"' in payload


class BaseStageValidationTest:
    """Shared infrastructure for stage validation test classes.

    Both TestProjectIdValidation and TestRunIdValidation inherit from this base
    to avoid duplicating _fake_assemble_payload, _fake_run_stage_via_cli,
    mock_deps, and _patch_stage.
    """

    @staticmethod
    async def _fake_assemble_payload(
        events,
        watermark,
        prior_reports,
    ) -> str:
        return 'fake payload'

    @staticmethod
    async def _fake_run_stage_via_cli(**kwargs):
        return StageResult(
            success=True,
            report={'summary': 'ok'},
        )

    @pytest.fixture
    def mock_deps(self):
        config = ReconciliationConfig(enabled=True, explore_codebase_root='/tmp/test')
        return {
            'memory_service': AsyncMock(),
            'taskmaster': AsyncMock(),
            'journal': AsyncMock(),
            'config': config,
        }

    def _patch_stage(self, stage, cli_side_effect=None):
        """Return a context manager that patches assemble_payload and run_stage_via_cli.

        Args:
            stage: The stage instance to patch.
            cli_side_effect: Optional async callable for run_stage_via_cli side_effect.
                Defaults to self._fake_run_stage_via_cli.
        """
        effective_cli_side_effect = cli_side_effect if cli_side_effect is not None else self._fake_run_stage_via_cli

        @contextmanager
        def _ctx():
            with (
                patch.object(stage, 'assemble_payload', side_effect=self._fake_assemble_payload),
                patch(
                    'fused_memory.reconciliation.stages.base.run_stage_via_cli',
                    side_effect=effective_cli_side_effect,
                ),
            ):
                yield

        return _ctx()


class TestProjectIdValidation(BaseStageValidationTest):
    """BaseStage.run() validates project_id and watermark.project_id."""

    @pytest.mark.asyncio
    async def test_run_raises_on_empty_project_id(self, mock_deps):

        stage = MemoryConsolidator(StageId.memory_consolidator, **mock_deps)
        stage.project_id = ''

        with self._patch_stage(stage), pytest.raises(ValueError, match='project_id'):
            await stage.run(
                events=[],
                watermark=Watermark(project_id=''),
                prior_reports=[],
                run_id='test-run-1',
            )

    @pytest.mark.asyncio
    async def test_run_raises_on_whitespace_project_id(self, mock_deps):

        stage = MemoryConsolidator(StageId.memory_consolidator, **mock_deps)
        stage.project_id = '   '

        with self._patch_stage(stage), pytest.raises(ValueError, match='project_id'):
            await stage.run(
                events=[],
                watermark=Watermark(project_id='some_project'),
                prior_reports=[],
                run_id='test-run-2',
            )

    @pytest.mark.asyncio
    async def test_run_raises_on_watermark_project_id_mismatch(self, mock_deps):

        stage = MemoryConsolidator(StageId.memory_consolidator, **mock_deps)
        stage.project_id = 'project_a'

        with self._patch_stage(stage), pytest.raises(ValueError) as exc_info:
            await stage.run(
                events=[],
                watermark=Watermark(project_id='project_b'),
                prior_reports=[],
                run_id='test-run-3',
            )
        error_msg = str(exc_info.value)
        assert 'project_a' in error_msg
        assert 'project_b' in error_msg

    @pytest.mark.asyncio
    async def test_run_allows_matching_watermark_project_id(self, mock_deps):

        stage = MemoryConsolidator(StageId.memory_consolidator, **mock_deps)
        stage.project_id = 'dark_factory'

        with self._patch_stage(stage):
            result = await stage.run(
                events=[],
                watermark=Watermark(project_id='dark_factory'),
                prior_reports=[],
                run_id='test-run-4',
            )
        assert result is not None
        assert result.stage == StageId.memory_consolidator
        assert result.completed_at is not None
        assert result.items_flagged == []
        assert result.stats == {}
        assert result.started_at is not None
        assert result.started_at <= result.completed_at

    @pytest.mark.asyncio
    async def test_run_allows_empty_watermark_project_id(self, mock_deps):

        stage = MemoryConsolidator(StageId.memory_consolidator, **mock_deps)
        stage.project_id = 'dark_factory'

        with self._patch_stage(stage):
            result = await stage.run(
                events=[],
                watermark=Watermark(project_id=''),
                prior_reports=[],
                run_id='test-run-5',
            )
        assert result is not None
        assert result.stage == StageId.memory_consolidator
        assert result.completed_at is not None
        assert result.items_flagged == []
        assert result.stats == {}
        assert result.started_at is not None
        assert result.started_at <= result.completed_at

    @pytest.mark.asyncio
    async def test_recon_context_includes_project_id(self, mock_deps):

        stage = MemoryConsolidator(StageId.memory_consolidator, **mock_deps)
        stage.project_id = 'dark_factory'

        captured_kwargs = {}

        async def capture_run_stage_via_cli(**kwargs):
            captured_kwargs.update(kwargs)
            return StageResult(success=True, report={'summary': 'ok'})

        with self._patch_stage(stage, cli_side_effect=capture_run_stage_via_cli):
            await stage.run(
                events=[],
                watermark=Watermark(project_id='dark_factory'),
                prior_reports=[],
                run_id='test-run-6',
            )
        assert 'payload' in captured_kwargs
        assert '`project_id`: "dark_factory"' in captured_kwargs['payload']

    @pytest.mark.asyncio
    async def test_whitespace_watermark_project_id_treated_as_empty(self, mock_deps, caplog):
        """Whitespace-only watermark project_id is stripped to '' by field_validator,
        guard skips mismatch check, and emits a DEBUG log about skipping."""
        import logging


        stage = MemoryConsolidator(StageId.memory_consolidator, **mock_deps)
        stage.project_id = 'dark_factory'

        with self._patch_stage(stage), caplog.at_level(logging.DEBUG, logger='fused_memory.reconciliation.stages.base'):
            result = await stage.run(
                events=[],
                watermark=Watermark(project_id='   '),
                prior_reports=[],
                run_id='test-run-7',
            )

        # (a) run succeeds — whitespace stripped to '', guard treats as empty watermark
        assert result is not None
        # (b) result.stage is correct
        assert result.stage == StageId.memory_consolidator
        # (c) DEBUG log emitted about skipping mismatch check
        assert any(
            ('no project_id' in rec.message.lower() or 'skipping' in rec.message.lower())
            for rec in caplog.records
            if rec.levelno == logging.DEBUG
        )

    def test_patch_stage_patches_assemble_payload_and_run_stage(self, mock_deps):
        """_patch_stage replaces both assemble_payload and run_stage_via_cli with mocks."""

        stage = MemoryConsolidator(StageId.memory_consolidator, **mock_deps)
        original_run_stage_via_cli = base_module.run_stage_via_cli
        original_assemble_payload = stage.assemble_payload

        with self._patch_stage(stage):
            # (a) assemble_payload is replaced with a mock instance
            assert isinstance(stage.assemble_payload, (AsyncMock, MagicMock))
            # (b) run_stage_via_cli in the base module is no longer the original function
            assert base_module.run_stage_via_cli is not original_run_stage_via_cli

        # Postconditions: context manager must restore original state on exit
        # (a) run_stage_via_cli is the original function again
        assert base_module.run_stage_via_cli is original_run_stage_via_cli
        # (b) assemble_payload is no longer a mock
        assert not isinstance(stage.assemble_payload, (AsyncMock, MagicMock))
        # (c) assemble_payload is exactly the original method reference
        assert stage.assemble_payload == original_assemble_payload

    def test_patch_stage_accepts_cli_side_effect(self, mock_deps):
        """_patch_stage wires a custom cli_side_effect onto the run_stage_via_cli mock."""

        stage = MemoryConsolidator(StageId.memory_consolidator, **mock_deps)

        async def custom_cli(**kwargs):
            return StageResult(success=False, report={'summary': 'custom'})

        with self._patch_stage(stage, cli_side_effect=custom_cli):
            # The patched run_stage_via_cli should have custom_cli as its side_effect
            assert base_module.run_stage_via_cli.side_effect is custom_cli  # type: ignore[reportFunctionMemberAccess]
            # Cross-assert: assemble_payload is also patched regardless of which parameter path is taken
            assert isinstance(stage.assemble_payload, (AsyncMock, MagicMock))


class TestRunIdValidation(BaseStageValidationTest):
    """BaseStage.run() validates run_id before prompt interpolation."""

    @pytest.mark.asyncio
    async def test_run_raises_on_empty_run_id(self, mock_deps):

        stage = MemoryConsolidator(StageId.memory_consolidator, **mock_deps)
        stage.project_id = 'dark_factory'

        with self._patch_stage(stage), pytest.raises(ValueError, match='run_id'):
            await stage.run(
                events=[],
                watermark=Watermark(project_id='dark_factory'),
                prior_reports=[],
                run_id='',
            )

    @pytest.mark.asyncio
    async def test_run_raises_on_whitespace_run_id(self, mock_deps):

        stage = MemoryConsolidator(StageId.memory_consolidator, **mock_deps)
        stage.project_id = 'dark_factory'

        with self._patch_stage(stage), pytest.raises(ValueError, match='run_id'):
            await stage.run(
                events=[],
                watermark=Watermark(project_id='dark_factory'),
                prior_reports=[],
                run_id='   ',
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize('bad_run_id,vector_name', [
        ('run\nid', 'newline'),
        ('run`id', 'backtick'),
        ('run;id', 'semicolon'),
    ])
    async def test_run_raises_on_injection_run_id(self, mock_deps, bad_run_id, vector_name):

        stage = MemoryConsolidator(StageId.memory_consolidator, **mock_deps)
        stage.project_id = 'dark_factory'

        with self._patch_stage(stage), pytest.raises(ValueError, match='run_id'):
            await stage.run(
                events=[],
                watermark=Watermark(project_id='dark_factory'),
                prior_reports=[],
                run_id=bad_run_id,
            )

    @pytest.mark.asyncio
    async def test_run_allows_valid_uuid_run_id(self, mock_deps):

        stage = MemoryConsolidator(StageId.memory_consolidator, **mock_deps)
        stage.project_id = 'dark_factory'

        valid_uuid = '550e8400-e29b-41d4-a716-446655440000'

        with self._patch_stage(stage):
            result = await stage.run(
                events=[],
                watermark=Watermark(project_id='dark_factory'),
                prior_reports=[],
                run_id=valid_uuid,
            )
        assert result is not None
        assert result.stage == StageId.memory_consolidator
        assert result.completed_at is not None

    @pytest.mark.asyncio
    async def test_recon_context_includes_run_id(self, mock_deps):

        stage = MemoryConsolidator(StageId.memory_consolidator, **mock_deps)
        stage.project_id = 'dark_factory'

        captured_kwargs = {}
        run_id_value = 'test-run-abc123'

        async def capture_run_stage_via_cli(**kwargs):
            captured_kwargs.update(kwargs)
            return StageResult(success=True, report={'summary': 'ok'})

        with self._patch_stage(stage, cli_side_effect=capture_run_stage_via_cli):
            await stage.run(
                events=[],
                watermark=Watermark(project_id='dark_factory'),
                prior_reports=[],
                run_id=run_id_value,
            )
        assert 'payload' in captured_kwargs
        assert f'run_id: {run_id_value}' in captured_kwargs['payload']


class TestBaseStageValidationContract:
    """Verify that BaseStageValidationTest exists and provides the four shared members."""

    def test_base_class_exists(self):
        assert BaseStageValidationTest is not None, "BaseStageValidationTest must be defined in test_stages.py"

    def test_base_class_has_fake_assemble_payload(self):
        assert hasattr(BaseStageValidationTest, '_fake_assemble_payload'), "missing _fake_assemble_payload"

    def test_base_class_has_fake_run_stage_via_cli(self):
        assert hasattr(BaseStageValidationTest, '_fake_run_stage_via_cli'), "missing _fake_run_stage_via_cli"

    def test_base_class_has_mock_deps_fixture(self):
        assert hasattr(BaseStageValidationTest, 'mock_deps'), "missing mock_deps fixture"

    def test_base_class_has_patch_stage_helper(self):
        assert hasattr(BaseStageValidationTest, '_patch_stage'), "missing _patch_stage helper"


class TestTierConfig:
    """MemoryConsolidator respects tier limits."""

    def test_default_limits(self):
        config = ReconciliationConfig()
        stage = MemoryConsolidator(
            StageId.memory_consolidator,
            AsyncMock(), AsyncMock(), AsyncMock(), config,
        )
        assert stage.episode_limit == 500
        assert stage.memory_limit == 1000

    def test_limits_are_writable(self):
        config = ReconciliationConfig()
        stage = MemoryConsolidator(
            StageId.memory_consolidator,
            AsyncMock(), AsyncMock(), AsyncMock(), config,
        )
        assert stage.episode_limit == 500
        assert stage.memory_limit == 1000
        stage.episode_limit = 125
        stage.memory_limit = 250
        assert stage.episode_limit == 125
        assert stage.memory_limit == 250
