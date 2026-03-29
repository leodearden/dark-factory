"""Tests for reconciliation stage configuration (CLI-native MCP execution)."""

import json
import logging
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
    _select_proactive_sample,
)

_MOCK_TYPES = (AsyncMock, MagicMock)


class TestMockTypesConstant:
    """Validate the _MOCK_TYPES constant that TestProjectIdValidation depends on."""

    def test_mock_types_constant_defined(self):
        assert isinstance(_MOCK_TYPES, tuple)
        assert AsyncMock in _MOCK_TYPES
        assert MagicMock in _MOCK_TYPES


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
    @pytest.mark.parametrize(
        'watermark_pid,run_id',
        [
            ('', 'test-run-5'),
            ('   ', 'test-run-whitespace-wm'),
        ],
    )
    async def test_run_succeeds_and_logs_debug_when_watermark_project_id_falsy(
        self, mock_deps, caplog, watermark_pid, run_id
    ):
        """Empty or whitespace-only watermark project_id: mismatch check is skipped,
        a DEBUG log is emitted, and the run succeeds with full results."""
        stage = MemoryConsolidator(StageId.memory_consolidator, **mock_deps)
        stage.project_id = 'dark_factory'

        with self._patch_stage(stage), caplog.at_level(
            logging.DEBUG, logger='fused_memory.reconciliation.stages.base'
        ):
            result = await stage.run(
                events=[],
                watermark=Watermark(project_id=watermark_pid),
                prior_reports=[],
                run_id=run_id,
            )
        assert result is not None
        assert result.stage == StageId.memory_consolidator
        assert result.completed_at is not None
        assert result.items_flagged == []
        assert result.stats == {}
        assert result.started_at is not None
        assert result.started_at <= result.completed_at
        assert any(
            ('no project_id' in rec.message.lower() or 'skipping' in rec.message.lower())
            for rec in caplog.records
            if rec.name == 'fused_memory.reconciliation.stages.base'
            and rec.levelno == logging.DEBUG
        )

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

    def test_watermark_rejects_none_project_id(self):
        """Watermark(project_id=None) raises Pydantic ValidationError — None is not a valid string."""
        import pydantic

        with pytest.raises(pydantic.ValidationError, match='project_id'):
            Watermark(project_id=None)  # type: ignore[arg-type]

    def test_patch_stage_patches_assemble_payload_and_run_stage(self, mock_deps):
        """_patch_stage replaces both assemble_payload and run_stage_via_cli with mocks."""

        stage = MemoryConsolidator(StageId.memory_consolidator, **mock_deps)
        original_run_stage_via_cli = base_module.run_stage_via_cli
        original_assemble_payload = stage.assemble_payload

        with self._patch_stage(stage):
            # (a) assemble_payload is replaced with a mock instance
            assert isinstance(stage.assemble_payload, _MOCK_TYPES)
            # (b) run_stage_via_cli in the base module is no longer the original function
            assert base_module.run_stage_via_cli is not original_run_stage_via_cli

        # Postconditions: context manager must restore original state on exit
        # (a) run_stage_via_cli is the original function again
        assert base_module.run_stage_via_cli is original_run_stage_via_cli
        # (b) assemble_payload is no longer a mock
        assert not isinstance(stage.assemble_payload, _MOCK_TYPES)
        # (c) assemble_payload is exactly the original method reference
        assert stage.assemble_payload == original_assemble_payload

    def test_patch_stage_accepts_cli_side_effect(self, mock_deps):
        """_patch_stage wires a custom cli_side_effect onto the run_stage_via_cli mock."""

        stage = MemoryConsolidator(StageId.memory_consolidator, **mock_deps)
        original_run_stage_via_cli = base_module.run_stage_via_cli
        original_assemble_payload = stage.assemble_payload

        async def custom_cli(**kwargs):
            return StageResult(success=False, report={'summary': 'custom'})

        with self._patch_stage(stage, cli_side_effect=custom_cli):
            # The patched run_stage_via_cli should have custom_cli as its side_effect
            assert base_module.run_stage_via_cli.side_effect is custom_cli  # type: ignore[reportFunctionMemberAccess]
            # Cross-assert: assemble_payload is also patched regardless of which parameter path is taken
            assert isinstance(stage.assemble_payload, _MOCK_TYPES)

        # Postconditions: context manager must restore original state on exit
        # (a) run_stage_via_cli is the original function again
        assert base_module.run_stage_via_cli is original_run_stage_via_cli
        # (b) assemble_payload is no longer a mock
        assert not isinstance(stage.assemble_payload, _MOCK_TYPES)
        # (c) assemble_payload is exactly the original method reference
        assert stage.assemble_payload == original_assemble_payload

    def test_patch_stage_restores_on_exception(self, mock_deps):
        """_patch_stage restores originals even when an exception is raised inside the with block."""

        stage = MemoryConsolidator(StageId.memory_consolidator, **mock_deps)
        original_run_stage_via_cli = base_module.run_stage_via_cli
        original_assemble_payload = stage.assemble_payload

        with pytest.raises(RuntimeError, match='boom'), self._patch_stage(stage):
            raise RuntimeError('boom')

        # Postconditions: context manager must restore original state on abnormal exit
        # (a) run_stage_via_cli is the original function again
        assert base_module.run_stage_via_cli is original_run_stage_via_cli
        # (b) assemble_payload is no longer a mock
        assert not isinstance(stage.assemble_payload, _MOCK_TYPES)
        # (c) assemble_payload is exactly the original method reference
        assert stage.assemble_payload == original_assemble_payload


class TestProactiveSampling:
    """Tests for _select_proactive_sample helper and proactive sample payload section."""

    # --- Fixtures ---

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
        return Watermark(project_id='test_project')

    def _make_task(self, tid: int, status: str) -> dict:
        return {'id': tid, 'title': f'Task {tid}', 'status': status, 'dependencies': []}

    # --- Step 1: payload contains proactive sample section ---

    @pytest.mark.asyncio
    async def test_proactive_sample_section_present_in_payload(self, mock_deps, watermark):
        """assemble_payload with active tasks and 0 flagged items produces payload
        containing '### Proactive Task Sample' section header."""
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'test_project'
        stage.project_root = '/tmp/test_project'
        mock_deps['taskmaster'].get_tasks.return_value = {
            'tasks': [
                self._make_task(1, 'in-progress'),
                self._make_task(2, 'pending'),
                self._make_task(3, 'done'),
            ]
        }

        payload = await stage.assemble_payload([], watermark, [])

        assert '### Proactive Task Sample' in payload

    # --- Step 2: in-progress and blocked tasks appear first ---

    def test_proactive_sample_prioritizes_in_progress_and_blocked(self):
        """Given tasks with mixed statuses, _select_proactive_sample returns
        in-progress and blocked tasks before review, pending, and done tasks."""
        tasks = [
            self._make_task(1, 'done'),
            self._make_task(2, 'pending'),
            self._make_task(3, 'review'),
            self._make_task(4, 'blocked'),
            self._make_task(5, 'in-progress'),
        ]
        result = _select_proactive_sample(tasks, 5)
        statuses = [t['status'] for t in result]
        # in-progress and blocked must come before review, pending, done
        high_priority = {'in-progress', 'blocked'}
        low_priority = {'review', 'pending', 'done'}
        last_high = max(
            (i for i, t in enumerate(result) if t['status'] in high_priority),
            default=-1,
        )
        first_low = min(
            (i for i, t in enumerate(result) if t['status'] in low_priority),
            default=len(result),
        )
        assert last_high < first_low, (
            f'High-priority tasks should appear before low-priority tasks. Got: {statuses}'
        )

    # --- Step 3: sample capped at MIN_TASK_SAMPLE ---

    def test_proactive_sample_capped_at_min_task_sample(self):
        """Given more than 5 eligible tasks, _select_proactive_sample returns exactly 5."""
        tasks = [self._make_task(i, 'pending') for i in range(1, 12)]
        result = _select_proactive_sample(tasks, 5)
        assert len(result) == 5

    # --- Step 4: all tasks returned when fewer than floor ---

    def test_proactive_sample_includes_all_when_fewer_than_floor(self):
        """Given fewer than 5 total tasks, _select_proactive_sample returns all of them."""
        tasks = [
            self._make_task(1, 'in-progress'),
            self._make_task(2, 'pending'),
            self._make_task(3, 'done'),
        ]
        result = _select_proactive_sample(tasks, 5)
        assert len(result) == 3

    # --- Step 6: remediation mode skips proactive sample ---

    @pytest.mark.asyncio
    async def test_proactive_sample_skipped_in_remediation_mode(self, mock_deps, watermark):
        """When stage.remediation_mode=True, payload does NOT contain '### Proactive Task Sample'."""
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'test_project'
        stage.project_root = '/tmp/test_project'
        stage.remediation_mode = True
        mock_deps['taskmaster'].get_tasks.return_value = {
            'tasks': [
                self._make_task(1, 'in-progress'),
                self._make_task(2, 'pending'),
            ]
        }

        payload = await stage.assemble_payload([], watermark, [])

        assert '### Proactive Task Sample' not in payload

    # --- Step 8: system prompt includes proactive spot-check guideline ---

    def test_system_prompt_includes_proactive_spot_check_guideline(self):
        """STAGE2_SYSTEM_PROMPT contains instruction about reviewing the proactive task sample."""
        from fused_memory.reconciliation.prompts.stage2 import STAGE2_SYSTEM_PROMPT

        # The prompt should mention proactive sample review
        assert 'Proactive Task Sample' in STAGE2_SYSTEM_PROMPT, (
            "STAGE2_SYSTEM_PROMPT must contain a guideline about reviewing the Proactive Task Sample"
        )

    # --- Step 12: ID descending as recency proxy ---

    def test_select_proactive_sample_uses_id_descending_as_recency_proxy(self):
        """Given tasks with same status but different IDs, higher-ID tasks appear first."""
        tasks = [
            self._make_task(10, 'pending'),
            self._make_task(3, 'pending'),
            self._make_task(7, 'pending'),
            self._make_task(1, 'pending'),
            self._make_task(5, 'pending'),
        ]
        result = _select_proactive_sample(tasks, 5)
        ids = [t['id'] for t in result]
        assert ids == sorted(ids, reverse=True), (
            f'Tasks with same status should be ordered by ID descending. Got: {ids}'
        )

    # --- Step 13: empty task tree handled gracefully ---

    @pytest.mark.asyncio
    async def test_proactive_sample_empty_task_tree(self, mock_deps, watermark):
        """When taskmaster returns 0 tasks, proactive sample section shows 'No tasks.'."""
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'test_project'
        stage.project_root = '/tmp/test_project'
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}

        payload = await stage.assemble_payload([], watermark, [])

        assert '### Proactive Task Sample' in payload
        # Section should contain 'No tasks.' for empty list
        proactive_idx = payload.index('### Proactive Task Sample')
        section_text = payload[proactive_idx:proactive_idx + 200]
        assert 'No tasks.' in section_text, (
            f'Empty task tree should show "No tasks." in proactive sample. Got: {section_text!r}'
        )

    # --- Step 10: 'Your Task' section includes proactive step ---

    @pytest.mark.asyncio
    async def test_payload_your_task_includes_proactive_step(self, mock_deps, watermark):
        """The 'Your Task' section in the payload includes a proactive spot-check instruction."""
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'test_project'
        stage.project_root = '/tmp/test_project'
        mock_deps['taskmaster'].get_tasks.return_value = {
            'tasks': [
                self._make_task(1, 'in-progress'),
                self._make_task(2, 'blocked'),
            ]
        }

        payload = await stage.assemble_payload([], watermark, [])

        # The 'Your Task' section should instruct the agent to review the proactive sample
        assert 'Proactive Task Sample' in payload
        # Specifically in the Your Task instruction steps (not just the section header)
        your_task_idx = payload.index('## Your Task')
        proactive_step_count = payload[your_task_idx:].count('Proactive Task Sample')
        assert proactive_step_count >= 1, (
            "The 'Your Task' instruction section should reference the Proactive Task Sample"
        )


    # --- Step 14: non-dict elements filtered without error ---

    def test_select_proactive_sample_filters_non_dict_elements(self):
        """_select_proactive_sample with mixed non-dict elements returns only valid dict tasks
        without raising AttributeError or TypeError."""
        valid_task_1 = self._make_task(5, 'in-progress')
        valid_task_2 = self._make_task(3, 'pending')
        mixed_input = [
            valid_task_1,
            'a plain string',
            42,
            None,
            ['nested', 'list'],
            valid_task_2,
        ]

        # Should not raise, and should return only the dict tasks
        result = _select_proactive_sample(mixed_input, 10)

        result_ids = {t['id'] for t in result}
        assert result_ids == {5, 3}, (
            f'Only dict tasks should appear in result. Got ids: {result_ids}'
        )
        assert len(result) == 2


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
    @pytest.mark.parametrize('bad_run_id', [
        'run\nid',
        'run`id',
        'run;id',
    ], ids=['newline', 'backtick', 'semicolon'])
    async def test_run_raises_on_injection_run_id(self, mock_deps, bad_run_id):

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


class TestTierConfig:
    """MemoryConsolidator respects tier limits."""

    def test_default_limits(self):
        config = ReconciliationConfig()
        stage = MemoryConsolidator(
            StageId.memory_consolidator,
            AsyncMock(), AsyncMock(), AsyncMock(), config,
        )
        assert stage.episode_limit is None
        assert stage.memory_limit is None

    def test_limits_are_writable(self):
        config = ReconciliationConfig()
        stage = MemoryConsolidator(
            StageId.memory_consolidator,
            AsyncMock(), AsyncMock(), AsyncMock(), config,
        )
        assert stage.episode_limit is None
        assert stage.memory_limit is None
        stage.episode_limit = 125
        stage.memory_limit = 250
        assert stage.episode_limit == 125
        assert stage.memory_limit == 250

    @pytest.mark.asyncio
    async def test_assemble_payload_raises_without_limits(self):
        config = ReconciliationConfig()
        stage = MemoryConsolidator(
            StageId.memory_consolidator,
            AsyncMock(), AsyncMock(), AsyncMock(), config,
        )
        stage.project_id = 'test_project'
        watermark = Watermark(project_id='test_project')
        with pytest.raises(ValueError, match='episode_limit and memory_limit must be explicitly set'):
            await stage.assemble_payload(events=[], watermark=watermark, prior_reports=[])

    @pytest.mark.asyncio
    async def test_assemble_payload_succeeds_with_limits_set(self):
        config = ReconciliationConfig()
        memory_mock = AsyncMock()
        memory_mock.get_episodes = AsyncMock(return_value=[])
        memory_mock.mem0 = AsyncMock()
        memory_mock.mem0.get_all = AsyncMock(return_value={'results': []})
        memory_mock.get_status = AsyncMock(return_value={})
        stage = MemoryConsolidator(
            StageId.memory_consolidator,
            memory_mock, AsyncMock(), AsyncMock(), config,
        )
        stage.project_id = 'test_project'
        stage.episode_limit = 125
        stage.memory_limit = 250
        watermark = Watermark(project_id='test_project')
        # Should not raise
        result = await stage.assemble_payload(events=[], watermark=watermark, prior_reports=[])
        assert isinstance(result, str)
        assert 'Stage 1' in result

    @pytest.mark.asyncio
    async def test_remediation_path_also_validates_limits(self):
        config = ReconciliationConfig()
        stage = MemoryConsolidator(
            StageId.memory_consolidator,
            AsyncMock(), AsyncMock(), AsyncMock(), config,
        )
        stage.project_id = 'test_project'
        # Set remediation findings but leave limits as None
        stage.remediation_findings = [{'description': 'test finding'}]
        watermark = Watermark(project_id='test_project')
        with pytest.raises(ValueError, match='episode_limit and memory_limit must be explicitly set'):
            await stage.assemble_payload(events=[], watermark=watermark, prior_reports=[])


class TestProjectIdGuidelineConstants:
    """_PROJECT_ID_GUIDELINE template and per-stage constants in prompts/__init__.py."""

    def test_template_exists(self):
        """_PROJECT_ID_GUIDELINE exists in prompts/__init__.py."""
        from fused_memory.reconciliation.prompts import _PROJECT_ID_GUIDELINE
        assert isinstance(_PROJECT_ID_GUIDELINE, str)

    def test_template_has_tools_placeholder(self):
        """_PROJECT_ID_GUIDELINE contains a {tools} placeholder."""
        from fused_memory.reconciliation.prompts import _PROJECT_ID_GUIDELINE
        assert '{tools}' in _PROJECT_ID_GUIDELINE

    def test_template_has_double_brace_project_id(self):
        """_PROJECT_ID_GUIDELINE escapes project_id as {{project_id}} so it survives .format(tools=...)."""
        from fused_memory.reconciliation.prompts import _PROJECT_ID_GUIDELINE
        # After formatting with tools, the {project_id} placeholder must survive
        formatted = _PROJECT_ID_GUIDELINE.format(tools='search')
        assert '{project_id}' in formatted

    def test_stage1_constant_exists(self):
        """_STAGE1_PROJECT_ID_GUIDELINE exists in prompts/__init__.py."""
        from fused_memory.reconciliation.prompts import _STAGE1_PROJECT_ID_GUIDELINE
        assert isinstance(_STAGE1_PROJECT_ID_GUIDELINE, str)

    def test_stage2_constant_exists(self):
        """_STAGE2_PROJECT_ID_GUIDELINE exists in prompts/__init__.py."""
        from fused_memory.reconciliation.prompts import _STAGE2_PROJECT_ID_GUIDELINE
        assert isinstance(_STAGE2_PROJECT_ID_GUIDELINE, str)

    def test_stage3_constant_exists(self):
        """_STAGE3_PROJECT_ID_GUIDELINE exists in prompts/__init__.py."""
        from fused_memory.reconciliation.prompts import _STAGE3_PROJECT_ID_GUIDELINE
        assert isinstance(_STAGE3_PROJECT_ID_GUIDELINE, str)

    def test_stage1_constant_has_project_id_placeholder(self):
        """_STAGE1_PROJECT_ID_GUIDELINE contains {project_id} placeholder."""
        from fused_memory.reconciliation.prompts import _STAGE1_PROJECT_ID_GUIDELINE
        assert '{project_id}' in _STAGE1_PROJECT_ID_GUIDELINE

    def test_stage2_constant_has_project_id_placeholder(self):
        """_STAGE2_PROJECT_ID_GUIDELINE contains {project_id} placeholder."""
        from fused_memory.reconciliation.prompts import _STAGE2_PROJECT_ID_GUIDELINE
        assert '{project_id}' in _STAGE2_PROJECT_ID_GUIDELINE

    def test_stage3_constant_has_project_id_placeholder(self):
        """_STAGE3_PROJECT_ID_GUIDELINE contains {project_id} placeholder."""
        from fused_memory.reconciliation.prompts import _STAGE3_PROJECT_ID_GUIDELINE
        assert '{project_id}' in _STAGE3_PROJECT_ID_GUIDELINE

    def test_stage2_constant_includes_expand_task(self):
        """Stage 2 guideline includes expand_task (full MCP access)."""
        from fused_memory.reconciliation.prompts import _STAGE2_PROJECT_ID_GUIDELINE
        assert 'expand_task' in _STAGE2_PROJECT_ID_GUIDELINE

    def test_stage2_constant_includes_parse_prd(self):
        """Stage 2 guideline includes parse_prd (full MCP access)."""
        from fused_memory.reconciliation.prompts import _STAGE2_PROJECT_ID_GUIDELINE
        assert 'parse_prd' in _STAGE2_PROJECT_ID_GUIDELINE

    def test_stage1_does_not_include_task_tools(self):
        """Stage 1 guideline does not include task write tools (Stage 1 is memory-only)."""
        from fused_memory.reconciliation.prompts import _STAGE1_PROJECT_ID_GUIDELINE
        assert 'get_tasks' not in _STAGE1_PROJECT_ID_GUIDELINE
        assert 'set_task_status' not in _STAGE1_PROJECT_ID_GUIDELINE
        assert 'add_task' not in _STAGE1_PROJECT_ID_GUIDELINE

    def test_stage3_does_not_include_write_tools(self):
        """Stage 3 guideline does not include write tools (Stage 3 is read-only)."""
        from fused_memory.reconciliation.prompts import _STAGE3_PROJECT_ID_GUIDELINE
        assert 'add_memory' not in _STAGE3_PROJECT_ID_GUIDELINE
        assert 'delete_memory' not in _STAGE3_PROJECT_ID_GUIDELINE
        assert 'set_task_status' not in _STAGE3_PROJECT_ID_GUIDELINE
        assert 'add_task' not in _STAGE3_PROJECT_ID_GUIDELINE


class TestStagePayloadProjectIdGuideline:
    """All three stages include the per-stage project_id guideline in their assembled payload."""

    @pytest.fixture
    def memory_mock(self):
        m = AsyncMock()
        m.get_episodes = AsyncMock(return_value=[])
        m.mem0 = AsyncMock()
        m.mem0.get_all = AsyncMock(return_value={'results': []})
        m.get_status = AsyncMock(return_value={})
        return m

    @pytest.fixture
    def mock_deps_for_stage(self, memory_mock):
        config = ReconciliationConfig(enabled=True, explore_codebase_root='/tmp/test')
        return {
            'memory_service': memory_mock,
            'taskmaster': AsyncMock(),
            'journal': AsyncMock(),
            'config': config,
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'stage_class,stage_id,expected_guideline_import,extra_setup,expected_tools,excluded_tools',
        [
            (
                'MemoryConsolidator',
                StageId.memory_consolidator,
                '_STAGE1_PROJECT_ID_GUIDELINE',
                'limits',
                ['add_memory'],                          # Stage 1 has memory write access
                ['get_tasks', 'set_task_status', 'add_task'],  # Stage 1 has no task tools
            ),
            (
                'TaskKnowledgeSync',
                StageId.task_knowledge_sync,
                '_STAGE2_PROJECT_ID_GUIDELINE',
                'taskmaster',
                ['expand_task', 'parse_prd'],            # Stage 2 has full MCP access
                [],
            ),
            (
                'IntegrityCheck',
                StageId.integrity_check,
                '_STAGE3_PROJECT_ID_GUIDELINE',
                None,
                ['get_tasks'],                           # Stage 3 reads tasks
                ['add_memory', 'delete_memory', 'set_task_status', 'add_task'],  # read-only
            ),
        ],
    )
    async def test_stage_payload_contains_project_id_guideline(
        self,
        mock_deps_for_stage,
        stage_class,
        stage_id,
        expected_guideline_import,
        extra_setup,
        expected_tools,
        excluded_tools,
    ):
        """Each stage's assembled payload contains the per-stage project_id guideline
        with the project_id correctly interpolated, and with the correct tool list."""
        from fused_memory.reconciliation import prompts as prompts_module

        guideline_template = getattr(prompts_module, expected_guideline_import)
        project_id = 'test_proj'
        expected_guideline = guideline_template.format(project_id=project_id)

        # Build stage instance
        cls_map = {
            'MemoryConsolidator': MemoryConsolidator,
            'TaskKnowledgeSync': TaskKnowledgeSync,
            'IntegrityCheck': IntegrityCheck,
        }
        stage = cls_map[stage_class](stage_id, **mock_deps_for_stage)
        stage.project_id = project_id

        if extra_setup == 'limits':
            stage.episode_limit = 125
            stage.memory_limit = 250
        elif extra_setup == 'taskmaster':
            stage.project_root = '/home/leo/src/test_proj'
            mock_deps_for_stage['taskmaster'].get_tasks.return_value = {'tasks': []}

        watermark = Watermark(project_id=project_id)
        payload = await stage.assemble_payload([], watermark, [])

        assert expected_guideline in payload, (
            f'{stage_class} payload missing per-stage project_id guideline.\n'
            f'Expected: {expected_guideline!r}\n'
            f'Payload snippet: {payload[-500:]!r}'
        )

        # Verify stage-specific tool names appear in the guideline within the payload
        for tool in expected_tools:
            assert tool in payload, (
                f'{stage_class} payload guideline missing expected tool {tool!r}. '
                f'Payload: {payload[-300:]!r}'
            )

        # Verify excluded tools are not in the guideline portion of the payload
        # (they may appear elsewhere in the payload's task/memory data, so we check
        # that the guideline constant itself doesn't include them)
        for tool in excluded_tools:
            assert tool not in expected_guideline, (
                f'{stage_class} guideline should NOT include {tool!r} but does: '
                f'{expected_guideline!r}'
            )

    @pytest.mark.asyncio
    async def test_stage2_payload_contains_project_root_instruction(
        self, mock_deps_for_stage
    ):
        """Stage 2 payload additionally includes the project_root= instruction."""
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps_for_stage)
        stage.project_id = 'test_proj'
        stage.project_root = '/home/leo/src/test_proj'
        mock_deps_for_stage['taskmaster'].get_tasks.return_value = {'tasks': []}
        watermark = Watermark(project_id='test_proj')

        payload = await stage.assemble_payload([], watermark, [])

        assert 'project_root="/home/leo/src/test_proj"' in payload

