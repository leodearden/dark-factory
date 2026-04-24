"""Tests for reconciliation stage configuration (CLI-native MCP execution)."""

import json
import logging
from contextlib import contextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from shared.cli_invoke import AgentResult, AllAccountsCappedException

import fused_memory.reconciliation.stages.base as base_module
from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import StageId, StageReport, Watermark
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
    run_stage_via_cli,
)
from fused_memory.reconciliation.prompts.stage2 import STAGE2_SYSTEM_PROMPT
from fused_memory.reconciliation.prompts.stage3 import STAGE3_SYSTEM_PROMPT
from fused_memory.reconciliation.stages.base import BaseStage
from fused_memory.reconciliation.stages.memory_consolidator import MemoryConsolidator
from fused_memory.reconciliation.stages.task_knowledge_sync import (
    _FLAGGED_ITEMS_CHAR_BUDGET,
    IntegrityCheck,
    TaskKnowledgeSync,
    _format_flagged,
    _select_proactive_sample,
)
from fused_memory.reconciliation.task_filter import (
    MAX_CANCELLED_TASKS_RETAINED,
    MAX_DONE_TASKS_RETAINED,
    FilteredTaskTree,
    _id_key,
    filter_task_tree,
)

_MOCK_TYPES = (AsyncMock, MagicMock)


def _extract_section(payload: str, header: str) -> str:
    """Return the body of *header* up to the next '\\n#' boundary, or '' if absent.

    Locates *header* in *payload*, then slices from that position to the start
    of the next markdown header (any level) or end-of-string, whichever comes first.
    """
    start = payload.find(header)
    if start == -1:
        return ''
    end = payload.find('\n#', start + 1)
    if end == -1:
        end = len(payload)
    return payload[start:end]


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

    def test_submit_task_in_disallow_task_writes(self):
        """submit_task must be blocked in Stage 1/3 (only Stage 2 may create tasks)."""
        assert 'mcp__fused-memory__submit_task' in DISALLOW_TASK_WRITES

    def test_resolve_ticket_in_disallow_task_writes(self):
        """resolve_ticket must be blocked in Stage 1/3 (only Stage 2 may create tasks)."""
        assert 'mcp__fused-memory__resolve_ticket' in DISALLOW_TASK_WRITES

    def test_add_task_still_in_disallow_task_writes(self):
        """Deprecated add_task facade must still be blocked (defence-in-depth)."""
        assert 'mcp__fused-memory__add_task' in DISALLOW_TASK_WRITES

    def test_submit_task_not_in_stage2_disallowed(self):
        """Stage 2 must be allowed to call submit_task (it creates tasks)."""
        assert 'mcp__fused-memory__submit_task' not in STAGE2_DISALLOWED

    def test_resolve_ticket_not_in_stage2_disallowed(self):
        """Stage 2 must be allowed to call resolve_ticket (it finalises task creation)."""
        assert 'mcp__fused-memory__resolve_ticket' not in STAGE2_DISALLOWED

    def test_submit_task_in_stage1_disallowed(self):
        """Stage 1 must not be able to call submit_task."""
        assert 'mcp__fused-memory__submit_task' in STAGE1_DISALLOWED

    def test_resolve_ticket_in_stage1_disallowed(self):
        """Stage 1 must not be able to call resolve_ticket."""
        assert 'mcp__fused-memory__resolve_ticket' in STAGE1_DISALLOWED

    def test_submit_task_in_stage3_disallowed(self):
        """Stage 3 must not be able to call submit_task."""
        assert 'mcp__fused-memory__submit_task' in STAGE3_DISALLOWED

    def test_resolve_ticket_in_stage3_disallowed(self):
        """Stage 3 must not be able to call resolve_ticket."""
        assert 'mcp__fused-memory__resolve_ticket' in STAGE3_DISALLOWED


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


class TestDoneProvenanceSection:
    """_render_done_provenance_section and Stage 2 briefing integration."""

    @pytest.fixture
    def mock_deps(self, tmp_path):
        config = ReconciliationConfig(enabled=True, explore_codebase_root=str(tmp_path))
        return {
            'memory_service': AsyncMock(),
            'taskmaster': AsyncMock(),
            'journal': AsyncMock(),
            'config': config,
        }

    @staticmethod
    def _init_repo(path):
        import subprocess
        subprocess.run(['git', 'init', '-q', '-b', 'main', str(path)], check=True)
        subprocess.run(
            ['git', '-C', str(path), 'config', 'user.email', 't@e.example'], check=True,
        )
        subprocess.run(
            ['git', '-C', str(path), 'config', 'user.name', 'T'], check=True,
        )
        (path / 'a.txt').write_text('a\n')
        (path / 'b.txt').write_text('b\n')
        subprocess.run(['git', '-C', str(path), 'add', '-A'], check=True)
        subprocess.run(
            ['git', '-C', str(path), 'commit', '-q', '-m', 'feat: ship a + b'],
            check=True,
        )
        return subprocess.run(
            ['git', '-C', str(path), 'rev-parse', 'HEAD'],
            check=True, capture_output=True, text=True,
        ).stdout.strip()

    @pytest.mark.asyncio
    async def test_commit_provenance_renders_file_list(self, mock_deps, tmp_path):
        """Task with commit provenance → git show file list injected."""
        sha = self._init_repo(tmp_path)
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'p'
        stage.project_root = str(tmp_path)
        mock_deps['taskmaster'].get_tasks.return_value = {
            'tasks': [{
                'id': 7, 'status': 'done', 'title': 'Ship A+B',
                'metadata': {'done_provenance': {'commit': sha}},
            }],
        }

        payload = await stage.assemble_payload([], Watermark(project_id='p'), [])

        assert '### Done-task Provenance' in payload
        assert f'commit: {sha}' in payload
        assert 'a.txt' in payload
        assert 'b.txt' in payload

    @pytest.mark.asyncio
    async def test_note_only_provenance_renders_note_verbatim(self, mock_deps, tmp_path):
        """Note-only provenance → quoted verbatim, no git call."""
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'p'
        stage.project_root = str(tmp_path)
        mock_deps['taskmaster'].get_tasks.return_value = {
            'tasks': [{
                'id': 9, 'status': 'done', 'title': 'Covered by sibling',
                'metadata': {
                    'done_provenance': {'note': 'implementation landed under task 7'},
                },
            }],
        }

        payload = await stage.assemble_payload([], Watermark(project_id='p'), [])

        assert '### Done-task Provenance' in payload
        assert 'note: implementation landed under task 7' in payload
        assert 'commit:' not in _extract_section(payload, '### Done-task Provenance')

    @pytest.mark.asyncio
    async def test_missing_provenance_marked_legacy(self, mock_deps, tmp_path):
        """Done task without metadata.done_provenance → 'provenance: unknown (legacy)'."""
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'p'
        stage.project_root = str(tmp_path)
        mock_deps['taskmaster'].get_tasks.return_value = {
            'tasks': [{'id': 11, 'status': 'done', 'title': 'Legacy'}],
        }

        payload = await stage.assemble_payload([], Watermark(project_id='p'), [])

        section = _extract_section(payload, '### Done-task Provenance')
        assert 'provenance: unknown (legacy)' in section

    @pytest.mark.asyncio
    async def test_no_done_tasks_omits_section(self, mock_deps, tmp_path):
        """Empty done_tasks → provenance section is not injected."""
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'p'
        stage.project_root = str(tmp_path)
        mock_deps['taskmaster'].get_tasks.return_value = {
            'tasks': [{'id': 1, 'status': 'pending', 'title': 'WIP'}],
        }

        payload = await stage.assemble_payload([], Watermark(project_id='p'), [])

        assert '### Done-task Provenance' not in payload

    @pytest.mark.asyncio
    async def test_invalid_commit_gracefully_omits_file_list(self, mock_deps, tmp_path):
        """Unresolvable commit → section header emitted, no file list, no exception."""
        self._init_repo(tmp_path)
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'p'
        stage.project_root = str(tmp_path)
        bad = 'deadbeef' * 5  # 40 chars but not a real SHA
        mock_deps['taskmaster'].get_tasks.return_value = {
            'tasks': [{
                'id': 3, 'status': 'done', 'title': 'Bad ref',
                'metadata': {'done_provenance': {'commit': bad}},
            }],
        }

        payload = await stage.assemble_payload([], Watermark(project_id='p'), [])

        section = _extract_section(payload, '### Done-task Provenance')
        assert f'commit: {bad}' in section
        # git show failed → no files line
        assert 'files:' not in section


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
        assert isinstance(result, StageReport)
        assert result.stage == StageId.memory_consolidator
        assert result.completed_at is not None
        assert result.items_flagged == []
        assert result.stats == {}
        assert result.started_at is not None
        assert result.started_at <= result.completed_at

    @pytest.mark.asyncio
    async def test_run_handles_model_construct_watermark_with_padded_project_id(self, mock_deps):
        """model_construct() bypasses the Pydantic field_validator, so watermark.project_id
        may carry un-stripped whitespace.  BaseStage.run() must not raise a mismatch error
        in this situation."""

        stage = MemoryConsolidator(StageId.memory_consolidator, **mock_deps)
        stage.project_id = 'dark_factory'

        # Build a Watermark that bypasses the field_validator — project_id is NOT stripped.
        padded_watermark = Watermark.model_construct(project_id=' dark_factory ')

        with self._patch_stage(stage):
            result = await stage.run(
                events=[],
                watermark=padded_watermark,
                prior_reports=[],
                run_id='test-run-model-construct',
            )
        assert isinstance(result, StageReport)
        assert result.stage == StageId.memory_consolidator

    @pytest.mark.asyncio
    async def test_run_handles_model_construct_watermark_with_none_project_id(
        self, mock_deps, caplog
    ):
        """model_construct() can produce a Watermark with project_id=None (bypassing
        validators).  BaseStage.run() must not raise AttributeError when it encounters
        None — it should treat None the same as an empty project_id and skip the mismatch
        check with a DEBUG log."""

        stage = MemoryConsolidator(StageId.memory_consolidator, **mock_deps)
        stage.project_id = 'dark_factory'

        # Build a Watermark that bypasses the field_validator — project_id is None.
        none_watermark = Watermark.model_construct(project_id=None)

        with self._patch_stage(stage), caplog.at_level(
            logging.DEBUG, logger='fused_memory.reconciliation.stages.base'
        ):
            result = await stage.run(
                events=[],
                watermark=none_watermark,
                prior_reports=[],
                run_id='test-run-model-construct-none',
            )
        assert isinstance(result, StageReport)
        assert result.stage == StageId.memory_consolidator
        assert any(
            ('no project_id' in rec.message.lower() or 'skipping' in rec.message.lower())
            for rec in caplog.records
            if rec.name == 'fused_memory.reconciliation.stages.base'
            and rec.levelno == logging.DEBUG
        )

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
        assert isinstance(result, StageReport)
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

    def test_select_proactive_sample_non_int_ids_sort_equivalent_to_id_key(self):
        """_select_proactive_sample sorts non-parseable string ids identically to _id_key fallback=0.

        Non-int ids map to 0 via _id_key, so they sort last (after all positive-int ids)
        within the same status bucket. This documents the expected behaviour and acts as
        a regression guard before the inline sort_key is replaced with _id_key in step-4.
        """
        tasks = [
            {'id': 'abc', 'title': 'Task abc', 'status': 'pending', 'dependencies': []},
            {'id': 5, 'title': 'Task 5', 'status': 'pending', 'dependencies': []},
            {'id': 'xyz', 'title': 'Task xyz', 'status': 'pending', 'dependencies': []},
            {'id': 2, 'title': 'Task 2', 'status': 'pending', 'dependencies': []},
        ]
        result = _select_proactive_sample(tasks, 4)
        ids = [t['id'] for t in result]

        # int ids (5, 2) must precede non-parseable string ids ('abc', 'xyz')
        # because _id_key('abc') == _id_key('xyz') == 0 < 2 < 5, sorted descending
        int_ids = [i for i in ids if isinstance(i, int)]
        str_ids = [i for i in ids if isinstance(i, str)]
        assert int_ids == [5, 2], f'Int ids should be [5, 2] descending. Got: {int_ids}'
        # string ids appear after all int ids
        last_int_pos = max(ids.index(i) for i in int_ids)
        first_str_pos = min(ids.index(s) for s in str_ids)
        assert first_str_pos > last_int_pos, (
            f'Non-int ids (fallback key=0) must sort after int ids. '
            f'int_ids at positions {[ids.index(i) for i in int_ids]}, '
            f'str_ids at positions {[ids.index(s) for s in str_ids]}'
        )
        # Verify _id_key agrees: all non-int ids yield 0
        for t in tasks:
            if isinstance(t['id'], str):
                assert _id_key(t) == 0, f'_id_key should return 0 for non-int id {t["id"]!r}'

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

    # --- Step: lazy iterable acceptance (task-709) ---

    def test_select_proactive_sample_accepts_lazy_iterable(self):
        """_select_proactive_sample works with a generator (lazy Iterable[dict]), not a list.

        Verifies the Iterable[dict] type hint is accurate: heapq.nsmallest accepts any
        iterable, so a generator can be passed directly without first materialising a list.
        Priority ordering (in-progress/blocked before pending/done) is preserved.
        """

        def task_generator():
            yield self._make_task(1, 'done')
            yield self._make_task(2, 'pending')
            yield self._make_task(3, 'in-progress')
            yield self._make_task(4, 'done')
            yield self._make_task(5, 'blocked')
            yield self._make_task(6, 'pending')

        result = _select_proactive_sample(task_generator(), 3)

        assert len(result) == 3

        # Verify the correct 3 tasks were selected (highest-priority ones)
        # Generator: done, pending, in-progress, done, blocked, pending
        # Top-3 by priority: in-progress(0) > blocked(1) > pending(3)
        assert {t['status'] for t in result} == {'in-progress', 'blocked', 'pending'}, (
            f'Expected top-priority tasks {{in-progress, blocked, pending}}, got: '
            f'{[t["status"] for t in result]}'
        )

        high_priority = {'in-progress', 'blocked'}
        low_priority = {'pending', 'done'}
        statuses = [t['status'] for t in result]
        last_high = max(
            (i for i, t in enumerate(result) if t['status'] in high_priority),
            default=-1,
        )
        first_low = min(
            (i for i, t in enumerate(result) if t['status'] in low_priority),
            default=len(result),
        )
        assert last_high < first_low, (
            f'In-progress/blocked tasks must appear before pending/done. Got: {statuses}'
        )

    # --- Step: empty iterable (task-709 amendment) ---

    def test_select_proactive_sample_empty_iterable_returns_empty_list(self):
        """_select_proactive_sample with an empty iterable returns [] without error.

        Explicit edge-case guard: heapq.nsmallest handles empty input correctly, and
        this test ensures the Iterable[dict] signature doesn't introduce any early
        access that would fail on empty generators.
        """
        assert _select_proactive_sample(iter([]), 5) == []
        assert _select_proactive_sample(iter([]), 0) == []


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
        assert isinstance(result, StageReport)
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

    def test_stage2_guideline_includes_submit_task(self):
        """Stage 2 guideline must list submit_task (new two-phase task creation)."""
        from fused_memory.reconciliation.prompts import _STAGE2_PROJECT_ID_GUIDELINE
        assert 'submit_task' in _STAGE2_PROJECT_ID_GUIDELINE

    def test_stage2_guideline_includes_resolve_ticket(self):
        """Stage 2 guideline must list resolve_ticket (new two-phase task creation)."""
        from fused_memory.reconciliation.prompts import _STAGE2_PROJECT_ID_GUIDELINE
        assert 'resolve_ticket' in _STAGE2_PROJECT_ID_GUIDELINE

    def test_stage2_guideline_does_not_include_add_task(self):
        """Stage 2 guideline must not list deprecated add_task."""
        from fused_memory.reconciliation.prompts import _STAGE2_PROJECT_ID_GUIDELINE
        assert 'add_task' not in _STAGE2_PROJECT_ID_GUIDELINE

    def test_stage1_guideline_does_not_include_submit_task(self):
        """Stage 1 guideline must not mention submit_task (Stage 1 has no task tools)."""
        from fused_memory.reconciliation.prompts import _STAGE1_PROJECT_ID_GUIDELINE
        assert 'submit_task' not in _STAGE1_PROJECT_ID_GUIDELINE

    def test_stage1_guideline_does_not_include_resolve_ticket(self):
        """Stage 1 guideline must not mention resolve_ticket (Stage 1 has no task tools)."""
        from fused_memory.reconciliation.prompts import _STAGE1_PROJECT_ID_GUIDELINE
        assert 'resolve_ticket' not in _STAGE1_PROJECT_ID_GUIDELINE

    def test_stage3_guideline_does_not_include_submit_task(self):
        """Stage 3 guideline must not mention submit_task (Stage 3 is read-only)."""
        from fused_memory.reconciliation.prompts import _STAGE3_PROJECT_ID_GUIDELINE
        assert 'submit_task' not in _STAGE3_PROJECT_ID_GUIDELINE

    def test_stage3_guideline_does_not_include_resolve_ticket(self):
        """Stage 3 guideline must not mention resolve_ticket (Stage 3 is read-only)."""
        from fused_memory.reconciliation.prompts import _STAGE3_PROJECT_ID_GUIDELINE
        assert 'resolve_ticket' not in _STAGE3_PROJECT_ID_GUIDELINE


class TestStage2SystemPromptTaskCreationSurface:
    """STAGE2_SYSTEM_PROMPT correctly advertises the two-phase task-creation API."""

    def test_stage2_prompt_includes_submit_task(self):
        """STAGE2_SYSTEM_PROMPT must reference mcp__fused-memory__submit_task."""
        assert 'mcp__fused-memory__submit_task' in STAGE2_SYSTEM_PROMPT

    def test_stage2_prompt_includes_resolve_ticket(self):
        """STAGE2_SYSTEM_PROMPT must reference mcp__fused-memory__resolve_ticket."""
        assert 'mcp__fused-memory__resolve_ticket' in STAGE2_SYSTEM_PROMPT

    def test_stage2_prompt_does_not_include_add_task(self):
        """STAGE2_SYSTEM_PROMPT must not reference deprecated mcp__fused-memory__add_task."""
        assert 'mcp__fused-memory__add_task' not in STAGE2_SYSTEM_PROMPT

    def test_stage2_prompt_documents_created_status(self):
        """STAGE2_SYSTEM_PROMPT must document the 'created' resolve_ticket status."""
        assert 'created' in STAGE2_SYSTEM_PROMPT

    def test_stage2_prompt_documents_combined_status(self):
        """STAGE2_SYSTEM_PROMPT must document the 'combined' resolve_ticket status."""
        assert 'combined' in STAGE2_SYSTEM_PROMPT

    def test_stage2_prompt_documents_failed_status(self):
        """STAGE2_SYSTEM_PROMPT must document the 'failed' resolve_ticket status."""
        assert 'failed' in STAGE2_SYSTEM_PROMPT


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
                ['get_tasks', 'set_task_status', 'add_task', 'submit_task', 'resolve_ticket'],  # Stage 1 has no task tools
            ),
            (
                'TaskKnowledgeSync',
                StageId.task_knowledge_sync,
                '_STAGE2_PROJECT_ID_GUIDELINE',
                'taskmaster',
                ['expand_task', 'parse_prd', 'submit_task', 'resolve_ticket'],  # Stage 2 has full MCP access
                [],
            ),
            (
                'IntegrityCheck',
                StageId.integrity_check,
                '_STAGE3_PROJECT_ID_GUIDELINE',
                None,
                ['get_tasks'],                           # Stage 3 reads tasks
                ['add_memory', 'delete_memory', 'set_task_status', 'add_task', 'submit_task', 'resolve_ticket'],  # read-only
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


class TestTaskKnowledgeSyncDeduplication:
    """Module-introspection tests: task_knowledge_sync must not define symbols owned by task_filter."""

    def test_no_local_status_priority(self):
        """task_knowledge_sync must NOT define _STATUS_PRIORITY at module level.
        task_filter._STATUS_PRIORITY is the single source of truth.
        """
        import fused_memory.reconciliation.stages.task_knowledge_sync as mod
        assert not hasattr(mod, '_STATUS_PRIORITY'), (
            'task_knowledge_sync._STATUS_PRIORITY must be removed after step-8; '
            'import from task_filter instead'
        )

    def test_no_local_format_tasks(self):
        """task_knowledge_sync must NOT define _format_tasks at module level.
        Use task_filter._render_task_line / format_task_list instead.
        """
        import fused_memory.reconciliation.stages.task_knowledge_sync as mod
        assert not hasattr(mod, '_format_tasks'), (
            'task_knowledge_sync._format_tasks must be removed after step-8; '
            'use task_filter.format_task_list instead'
        )


class TestTaskKnowledgeSyncUsesFilterTaskTree:
    """Integration tests: assemble_payload delegates active-tree logic to filter_task_tree."""

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

    def _make_task(self, tid: int, status: str, title: str | None = None) -> dict:
        return {
            'id': tid,
            'title': title or f'Task {tid} ({status})',
            'status': status,
            'dependencies': [],
        }

    @pytest.mark.asyncio
    async def test_payload_active_task_tree_uses_filter_task_tree(self, mock_deps, watermark):
        """assemble_payload uses filter_task_tree: payload contains em-dash summary, 'shown'
        parenthetical, blocked task, and deferred task in the Active Task Tree section."""
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'test_project'
        stage.project_root = '/tmp/test_project'
        mock_deps['taskmaster'].get_tasks.return_value = {
            'tasks': [
                self._make_task(1, 'pending'),
                self._make_task(2, 'in-progress'),
                self._make_task(3, 'blocked', 'Blocked Task'),
                self._make_task(4, 'deferred', 'Deferred Task'),
                self._make_task(5, 'review'),
                self._make_task(6, 'done'),
                self._make_task(7, 'cancelled'),
                self._make_task(8, 'done'),
            ]
        }

        payload = await stage.assemble_payload([], watermark, [])

        # (a) em-dash summary line produced by format_filtered_task_tree
        assert '\u2014 omitted' in payload, (
            'Payload missing em-dash summary line from format_filtered_task_tree'
        )

        # (b) 'shown' parenthetical from format_filtered_task_tree header
        assert 'active shown' in payload, (
            "Payload missing 'active shown' parenthetical from filter_task_tree header"
        )

        # (c) blocked task appears in the Active Task Tree section
        assert 'Blocked Task' in payload, (
            'Blocked task title not found in payload; active set may not have been widened'
        )

        # (d) deferred task appears in the Active Task Tree section
        assert 'Deferred Task' in payload, (
            'Deferred task title not found in payload; active set may not have been widened'
        )

    @pytest.mark.asyncio
    async def test_payload_recently_completed_tasks_sorted_desc(self, mock_deps, watermark):
        """assemble_payload sorts recently completed tasks by id descending."""
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'test_project'
        stage.project_root = '/tmp/test_project'
        mock_deps['taskmaster'].get_tasks.return_value = {
            'tasks': [
                self._make_task(1, 'in-progress'),
                self._make_task(5, 'done', 'Done Five'),
                self._make_task(10, 'done', 'Done Ten'),
                self._make_task(3, 'done', 'Done Three'),
                self._make_task(8, 'done', 'Done Eight'),
            ]
        }

        payload = await stage.assemble_payload([], watermark, [])

        # (a) Recently Completed Tasks header present
        assert '### Recently Completed Tasks' in payload, (
            "Payload missing '### Recently Completed Tasks' header"
        )

        # (b) done tasks appear in descending id order: 10, 8, 5, 3
        section_text = _extract_section(payload, '### Recently Completed Tasks')

        pos_10 = section_text.find('[10]')
        pos_8 = section_text.find('[8]')
        pos_5 = section_text.find('[5]')
        pos_3 = section_text.find('[3]')

        assert pos_10 != -1, "Done task id=10 not found in Recently Completed section"
        assert pos_8 != -1, "Done task id=8 not found in Recently Completed section"
        assert pos_5 != -1, "Done task id=5 not found in Recently Completed section"
        assert pos_3 != -1, "Done task id=3 not found in Recently Completed section"

        assert pos_10 < pos_8 < pos_5 < pos_3, (
            f'Recently Completed Tasks not sorted by id desc. '
            f'positions: [10]={pos_10}, [8]={pos_8}, [5]={pos_5}, [3]={pos_3}'
        )

    @pytest.mark.asyncio
    async def test_payload_done_tasks_older_than_30_dropped_from_recently_completed(
        self, mock_deps, watermark
    ):
        """filter_task_tree caps done_tasks at MAX_DONE_TASKS_RETAINED; overflow tasks are dropped."""
        # Derive task count and boundary ids symbolically so the test fails loudly
        # if MAX_DONE_TASKS_RETAINED is ever changed.
        n_tasks = MAX_DONE_TASKS_RETAINED + 5
        lowest_retained = n_tasks - MAX_DONE_TASKS_RETAINED + 1  # = 6 when cap=30
        highest_dropped = n_tasks - MAX_DONE_TASKS_RETAINED       # = 5 when cap=30

        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'test_project'
        stage.project_root = '/tmp/test_project'
        mock_deps['taskmaster'].get_tasks.return_value = {
            'tasks': [self._make_task(i, 'done') for i in range(1, n_tasks + 1)]
        }

        payload = await stage.assemble_payload([], watermark, [])

        section = _extract_section(payload, '### Recently Completed Tasks')
        assert section, "Payload missing '### Recently Completed Tasks' section"

        # (a) Newest task (highest id) must appear — always retained
        assert f'- [{n_tasks}] ' in section, (
            f"Task id={n_tasks} not found in Recently Completed section; "
            f"it should be the first retained entry (highest id).\n"
            f"Section content:\n{section}"
        )

        # (b) Oldest task (id=1) must NOT appear — dropped by MAX_DONE_TASKS_RETAINED cap
        assert '- [1] ' not in section, (
            f"Task id=1 should be dropped by the MAX_DONE_TASKS_RETAINED={MAX_DONE_TASKS_RETAINED} cap "
            f"(retained ids are {n_tasks}..{lowest_retained}, dropped ids are {highest_dropped}..1).\n"
            f"Section content:\n{section}"
        )

        # (c) lowest_retained is at the cap boundary — pins exact cutoff
        assert f'- [{lowest_retained}] ' in section, (
            f"Task id={lowest_retained} should be retained "
            f"(it is at the cap boundary; ids {n_tasks}..{lowest_retained} are kept).\n"
            f"Section content:\n{section}"
        )

        # (d) highest_dropped is one above the cutoff — off-by-one regression guard
        assert f'- [{highest_dropped}] ' not in section, (
            f"Task id={highest_dropped} should be dropped "
            f"(ids {highest_dropped}..1 are cut by MAX_DONE_TASKS_RETAINED={MAX_DONE_TASKS_RETAINED}).\n"
            f"Section content:\n{section}"
        )

    @pytest.mark.asyncio
    async def test_stage_does_not_apply_second_slice_on_done_tasks(
        self, mock_deps, watermark
    ):
        """Stage must NOT apply a second done_tasks slice on top of filter_task_tree's cap.

        Two assertions:
        (1) Source-level guard: assemble_payload source must not contain a slice on
            done_tasks (e.g. ``done_tasks[:30]``).  This is a tripwire — it fires the
            moment someone re-introduces a hardcoded re-slice that duplicates the cap
            already enforced by filter_task_tree.
        (2) Behavioral guard: exactly MAX_DONE_TASKS_RETAINED done tasks must ALL appear
            in the Recently Completed section — the stage must not silently trim them.
        """
        import inspect
        import re

        # (1) Source-level tripwire: assemble_payload must not slice done_tasks.
        source = inspect.getsource(TaskKnowledgeSync.assemble_payload)
        assert not re.search(r'done_tasks\[.*:.*\]', source), (
            "assemble_payload contains a slice on done_tasks (e.g. done_tasks[:30]). "
            "This is dead code — filter_task_tree already caps done_tasks at "
            f"MAX_DONE_TASKS_RETAINED={MAX_DONE_TASKS_RETAINED}. "
            "Remove the slice; filter_task_tree is the single source of truth."
        )

        # (2) Behavioral guard: all MAX_DONE_TASKS_RETAINED tasks pass through uncut.
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'test_project'
        stage.project_root = '/tmp/test_project'
        mock_deps['taskmaster'].get_tasks.return_value = {
            'tasks': [
                self._make_task(i, 'done')
                for i in range(1, MAX_DONE_TASKS_RETAINED + 1)
            ]
        }

        payload = await stage.assemble_payload([], watermark, [])

        section = _extract_section(payload, '### Recently Completed Tasks')
        assert section, "Payload missing '### Recently Completed Tasks' section"

        # All MAX_DONE_TASKS_RETAINED tasks must appear — no second slice should trim them.
        missing = [
            tid
            for tid in range(1, MAX_DONE_TASKS_RETAINED + 1)
            if f'- [{tid}] ' not in section
        ]
        assert not missing, (
            f"Tasks {missing} missing from Recently Completed section. "
            f"The stage may be applying a redundant slice that trims "
            f"filter_task_tree's already-capped output.\n"
            f"Section content:\n{section}"
        )

    # --- Step: other-status exclusion from proactive pool (task-709) ---

    @pytest.mark.asyncio
    async def test_proactive_sample_pool_excludes_other_status_tasks(self, mock_deps, watermark):
        """filter_task_tree drops unknown-status tasks before they reach the proactive sample
        pool; such tasks must not appear in '### Proactive Task Sample'.

        The inline comment at task_knowledge_sync.py:94-95 documents this narrowing:
        filter_task_tree increments other_count for unknown statuses without appending to
        any list, so they never enter the itertools.chain pool.
        """
        mystery_title = 'Mystery Status Task'
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'test_project'
        stage.project_root = '/tmp/test_project'
        mock_deps['taskmaster'].get_tasks.return_value = {
            'tasks': [
                self._make_task(1, 'in-progress', 'Active Task'),
                self._make_task(2, 'pending', 'Pending Task'),
                self._make_task(3, 'done', 'Done Task'),
                self._make_task(4, 'mystery', mystery_title),  # unknown status -> other_count only
            ]
        }

        payload = await stage.assemble_payload([], watermark, [])

        proactive_section = _extract_section(payload, '### Proactive Task Sample')
        assert proactive_section, "Payload must contain '### Proactive Task Sample' section"
        assert mystery_title not in proactive_section, (
            f"Other-status task '{mystery_title}' must not appear in proactive sample pool; "
            f"filter_task_tree drops unknown-status tasks before the pool is built."
        )
        # Sanity: at least one known-status task must be in the pool
        assert 'Active Task' in proactive_section or 'Pending Task' in proactive_section, (
            "At least one known-status task must appear in the proactive sample section"
        )


# ── Tests for task 455: MemoryConsolidator filtered task tree injection ─────────


class TestMemoryConsolidatorFilteredTaskTree:
    """MemoryConsolidator includes/omits '### Active Task Tree' based on filtered_task_tree."""

    @pytest.fixture
    def mock_memory(self):
        svc = AsyncMock()
        svc.get_episodes = AsyncMock(return_value=[])
        svc.get_status = AsyncMock(return_value={})
        svc.mem0 = AsyncMock()
        svc.mem0.get_all = AsyncMock(return_value={'results': []})
        return svc

    @pytest.fixture
    def watermark(self):
        return Watermark(project_id='test_project')

    def _make_active_tree(self, count: int = 3):
        from fused_memory.reconciliation.task_filter import FilteredTaskTree
        active = [
            {'id': i, 'title': f'Active task {i}', 'status': 'pending', 'dependencies': []}
            for i in range(1, count + 1)
        ]
        return FilteredTaskTree(
            active_tasks=active,
            done_count=0,
            cancelled_count=2,
            other_count=0,
            total_count=count + 2,  # count active + 0 done + 2 cancelled
        )

    @pytest.mark.asyncio
    async def test_payload_includes_active_task_tree_section_when_set(
        self, mock_memory, watermark,
    ):
        """assemble_payload includes '### Active Task Tree' when filtered_task_tree is set."""
        stage = MemoryConsolidator(
            StageId.memory_consolidator, mock_memory, None, AsyncMock(), AsyncMock(),
        )
        stage.project_id = 'test_project'
        stage.episode_limit = 100
        stage.memory_limit = 200
        stage.filtered_task_tree = self._make_active_tree(3)

        payload = await stage.assemble_payload([], watermark, [])

        assert '### Active Task Tree' in payload
        assert 'Active task 1' in payload

    @pytest.mark.asyncio
    async def test_payload_omits_section_when_tree_none(self, mock_memory, watermark):
        """assemble_payload does NOT include '### Active Task Tree' when filtered_task_tree is None."""
        stage = MemoryConsolidator(
            StageId.memory_consolidator, mock_memory, None, AsyncMock(), AsyncMock(),
        )
        stage.project_id = 'test_project'
        stage.episode_limit = 100
        stage.memory_limit = 200
        stage.filtered_task_tree = None

        payload = await stage.assemble_payload([], watermark, [])

        assert '### Active Task Tree' not in payload

    @pytest.mark.asyncio
    async def test_format_assembled_payload_includes_tree_when_set(
        self, mock_memory, watermark,
    ):
        """_format_assembled_payload includes '### Active Task Tree' when filtered_task_tree is set."""
        from fused_memory.models.reconciliation import AssembledPayload

        ap = AssembledPayload(
            events=[],
            context_items={},
            total_tokens=0,
            events_remaining=0,
        )
        stage = MemoryConsolidator(
            StageId.memory_consolidator, mock_memory, None, AsyncMock(), AsyncMock(),
        )
        stage.project_id = 'test_project'
        stage.episode_limit = 100
        stage.memory_limit = 200
        stage.assembled_payload = ap
        stage.filtered_task_tree = self._make_active_tree(2)

        payload = await stage._format_assembled_payload(watermark)

        assert '### Active Task Tree' in payload
        assert 'Active task 1' in payload

    def test_make_active_tree_summary_line_has_consistent_total(self):
        """_make_active_tree(3) total_count must equal 3 active + 0 done + 2 cancelled = 5."""
        from fused_memory.reconciliation.task_filter import format_filtered_task_tree
        tree = self._make_active_tree(3)
        rendered = format_filtered_task_tree(tree)
        assert '5 total' in rendered, (
            f'Expected total_count=5 (3 active + 0 done + 2 cancelled) '
            f'but rendered: {rendered!r}'
        )


# ── Tests for task 455: TaskKnowledgeSync filtered task tree injection ─────────


class TestTaskKnowledgeSyncFilteredTaskTree:
    """TaskKnowledgeSync prefers harness-provided filtered_task_tree over self-fetch."""

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

    def _make_tree(self, tasks: list[dict], done_count: int = 0, cancelled_count: int = 0,
                   done_tasks: list[dict] | None = None,
                   cancelled_tasks: list[dict] | None = None):
        from fused_memory.reconciliation.task_filter import FilteredTaskTree
        return FilteredTaskTree(
            active_tasks=tasks,
            done_tasks=done_tasks or [],
            cancelled_tasks=cancelled_tasks or [],
            done_count=done_count,
            cancelled_count=cancelled_count,
            other_count=0,
            total_count=len(tasks) + done_count + cancelled_count,
        )

    @pytest.mark.asyncio
    async def test_uses_harness_filtered_tree_when_set(self, mock_deps, watermark):
        """When filtered_task_tree is set, assemble_payload uses it and skips get_tasks."""
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'test_project'
        stage.project_root = '/tmp/test_project'
        stage.filtered_task_tree = self._make_tree(
            [self._make_task(10, 'in-progress'), self._make_task(20, 'pending')],
            done_count=0,
        )

        payload = await stage.assemble_payload([], watermark, [])

        # get_tasks must NOT be called
        mock_deps['taskmaster'].get_tasks.assert_not_called()
        # Payload must contain the Active Task Tree section
        assert '### Active Task Tree' in payload
        assert 'Task 10' in payload
        assert 'Task 20' in payload
        # Recently Completed: done_count=0 and done_tasks=[] → 'No tasks.'
        recently_section = _extract_section(payload, '### Recently Completed Tasks')
        assert '### Recently Completed Tasks' in payload
        assert 'No tasks.' in recently_section, (
            f"Expected 'No tasks.' in Recently Completed section, got: {recently_section!r}"
        )

    @pytest.mark.asyncio
    async def test_fallback_self_fetch_uses_shared_filter(self, mock_deps, watermark):
        """When filtered_task_tree is None, fallback fetch includes blocked/deferred tasks."""
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'test_project'
        stage.project_root = '/tmp/test_project'
        stage.filtered_task_tree = None  # no harness-provided tree

        mock_deps['taskmaster'].get_tasks.return_value = {
            'tasks': [
                self._make_task(1, 'blocked'),
                self._make_task(2, 'deferred'),
                self._make_task(3, 'pending'),
                self._make_task(4, 'done'),
            ]
        }

        payload = await stage.assemble_payload([], watermark, [])

        # Active section must include blocked and deferred tasks
        assert '### Active Task Tree' in payload
        assert 'Task 1' in payload
        assert 'Task 2' in payload

    @pytest.mark.asyncio
    async def test_proactive_sample_derived_from_filtered_tree(self, mock_deps, watermark):
        """With filtered_task_tree set, proactive sample is drawn from active_tasks, not a self-fetch."""
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'test_project'
        stage.project_root = '/tmp/test_project'
        stage.filtered_task_tree = self._make_tree(
            [
                self._make_task(1, 'in-progress'),
                self._make_task(2, 'blocked'),
                self._make_task(3, 'pending'),
                self._make_task(4, 'pending'),
                self._make_task(5, 'pending'),
                self._make_task(6, 'pending'),
            ],
            done_count=0,
        )

        payload = await stage.assemble_payload([], watermark, [])

        # get_tasks must NOT be called
        mock_deps['taskmaster'].get_tasks.assert_not_called()
        # Proactive Task Sample section must be present
        assert '### Proactive Task Sample' in payload

    @pytest.mark.asyncio
    async def test_proactive_sample_includes_done_and_cancelled_via_harness_path(
        self, mock_deps, watermark,
    ):
        """When filtered_task_tree has done_tasks and cancelled_tasks, proactive sample
        can include tasks from those lists (not just active_tasks)."""
        done_tasks = [self._make_task(tid, 'done') for tid in range(2, 7)]   # ids 2-6
        cancelled_task = self._make_task(7, 'cancelled')
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'test_project'
        stage.project_root = '/tmp/test_project'
        stage.filtered_task_tree = self._make_tree(
            [self._make_task(1, 'in-progress')],  # only 1 active task
            done_count=5,
            cancelled_count=1,
            done_tasks=done_tasks,
            cancelled_tasks=[cancelled_task],
        )

        payload = await stage.assemble_payload([], watermark, [])

        mock_deps['taskmaster'].get_tasks.assert_not_called()
        proactive_section = _extract_section(payload, '### Proactive Task Sample')
        assert proactive_section, "### Proactive Task Sample section must be present"
        # At least one done task id (2-6) must appear inside the proactive sample section —
        # demonstrating that done tasks are reachable via the unified pool.
        done_ids_present = any(f'[{tid}]' in proactive_section for tid in range(2, 7))
        assert done_ids_present, (
            f'Expected at least one done task id (2-6) in proactive sample section. '
            f'Section was:\n{proactive_section!r}'
        )

    @pytest.mark.asyncio
    async def test_recently_completed_shows_done_tasks_from_harness_tree(
        self, mock_deps, watermark,
    ):
        """When filtered_task_tree has done_tasks populated, Recently Completed renders them."""
        done_task = self._make_task(99, 'done')
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'test_project'
        stage.project_root = '/tmp/test_project'
        stage.filtered_task_tree = self._make_tree(
            [self._make_task(10, 'in-progress')],
            done_count=1,
            done_tasks=[done_task],
        )

        payload = await stage.assemble_payload([], watermark, [])

        mock_deps['taskmaster'].get_tasks.assert_not_called()
        assert '### Recently Completed Tasks' in payload
        # Done task title must appear in the recently completed section
        assert 'Task 99' in payload

    @pytest.mark.asyncio
    async def test_recently_completed_renders_done_titles_via_primary_path(
        self, mock_deps, watermark,
    ):
        """Primary if-branch: done_tasks populated → all done task titles appear in Recently Completed."""
        done_tasks = [
            self._make_task(101, 'done'),
            self._make_task(102, 'done'),
            self._make_task(103, 'done'),
        ]
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'test_project'
        stage.project_root = '/tmp/test_project'
        stage.filtered_task_tree = self._make_tree(
            [self._make_task(10, 'in-progress')],
            done_count=3,
            done_tasks=done_tasks,
        )

        payload = await stage.assemble_payload([], watermark, [])

        # (a) get_tasks must NOT be called
        mock_deps['taskmaster'].get_tasks.assert_not_called()
        # (b) Recently Completed section must be present
        assert '### Recently Completed Tasks' in payload
        # (c) Each done task title must appear in the Recently Completed section
        recently_section = _extract_section(payload, '### Recently Completed Tasks')
        assert 'Task 101' in recently_section, "Task 101 not found in Recently Completed section"
        assert 'Task 102' in recently_section, "Task 102 not found in Recently Completed section"
        assert 'Task 103' in recently_section, "Task 103 not found in Recently Completed section"

    @pytest.mark.asyncio
    async def test_recently_completed_populated_on_fallback(self, mock_deps, watermark):
        """When filtered_task_tree is None, fallback path populates recently completed tasks."""
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'test_project'
        stage.project_root = '/tmp/test_project'
        stage.filtered_task_tree = None  # no harness-provided tree

        mock_deps['taskmaster'].get_tasks.return_value = {
            'tasks': [
                self._make_task(1, 'done'),
                self._make_task(2, 'done'),
                self._make_task(3, 'done'),
                self._make_task(4, 'pending'),
                self._make_task(5, 'in-progress'),
            ]
        }

        payload = await stage.assemble_payload([], watermark, [])

        assert '### Recently Completed Tasks' in payload
        # At least one done task title must appear
        assert 'Task 1' in payload or 'Task 2' in payload or 'Task 3' in payload

    @pytest.mark.asyncio
    async def test_fallback_renders_done_tasks_in_recently_completed_section(
        self, mock_deps, watermark,
    ):
        """Fallback path: done tasks appear inside Recently Completed section (scoped assertion)."""
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'test_project'
        stage.project_root = '/tmp/test_project'
        stage.filtered_task_tree = None  # trigger fallback self-fetch

        mock_deps['taskmaster'].get_tasks.return_value = {
            'tasks': [
                self._make_task(1, 'in-progress'),
                self._make_task(2, 'pending'),
                self._make_task(10, 'done'),
                self._make_task(11, 'done'),
                self._make_task(12, 'done'),
            ]
        }

        payload = await stage.assemble_payload([], watermark, [])

        # (a) Recently Completed Tasks header must be present in fallback path
        assert '### Recently Completed Tasks' in payload, (
            "Payload missing '### Recently Completed Tasks' header in fallback path"
        )

        # (b) Extract the Recently Completed section body
        recently_section = _extract_section(payload, '### Recently Completed Tasks')

        # (c) Done-task ids must appear INSIDE the Recently Completed section (scoped)
        assert '[10]' in recently_section, "Done task id=10 not found in Recently Completed section"
        assert '[11]' in recently_section, "Done task id=11 not found in Recently Completed section"
        assert '[12]' in recently_section, "Done task id=12 not found in Recently Completed section"

        # (d) Active-task ids must NOT appear inside the Recently Completed section —
        #     cross-validates that section extraction is correctly bounded.
        #     Anchored to the rendered task-line prefix '- [N] ' (matches
        #     _render_task_line format '- [{tid}] ({status}) {title} deps=...')
        #     to avoid false negatives from 'deps=[1]' or similar substrings.
        assert '- [1] ' not in recently_section, (
            "Active task id=1 should NOT be in Recently Completed section"
        )
        assert '- [2] ' not in recently_section, (
            "Active task id=2 should NOT be in Recently Completed section"
        )

        # (e) Symmetric cross-boundary check: done-task ids must NOT leak into
        #     the Active Task Tree section. This is the counterpart to (d) above
        #     and converts this test from a partial duplicate of
        #     test_payload_recently_completed_tasks_sorted_desc into a genuine
        #     cross-section-boundary assertion.
        active_section = _extract_section(payload, '### Active Task Tree')
        assert '- [10] ' not in active_section, (
            "Done task id=10 should NOT appear in Active Task Tree section"
        )
        assert '- [11] ' not in active_section, (
            "Done task id=11 should NOT appear in Active Task Tree section"
        )
        assert '- [12] ' not in active_section, (
            "Done task id=12 should NOT appear in Active Task Tree section"
        )


class TestExtractSectionHelper:
    """Unit tests for the _extract_section module-level helper."""

    def test_extracts_section_bounded_by_next_header(self):
        """Helper returns content from header up to (not including) the next '\\n#' boundary."""
        payload = '### First Section\nline one\nline two\n### Second Section\nother content'
        result = _extract_section(payload, '### First Section')
        assert result == '### First Section\nline one\nline two'
        assert '### Second Section' not in result

    def test_extracts_section_to_eof_when_no_next_header(self):
        """When no subsequent '#' header exists, helper returns from header through end-of-string."""
        payload = '### Only Section\nsome content here\nmore lines'
        result = _extract_section(payload, '### Only Section')
        assert result == '### Only Section\nsome content here\nmore lines'

    def test_returns_empty_string_when_header_absent(self):
        """When the header does not appear in payload, helper returns ''."""
        payload = '### Other Section\nsome content'
        result = _extract_section(payload, '### Missing Header')
        assert result == ''

    def test_extracts_section_when_header_at_byte_zero(self):
        """Header at byte 0 is found correctly; body ends at the next '\\n#' boundary."""
        payload = '### Start\nbody line\n### Next\nother'
        result = _extract_section(payload, '### Start')
        assert result == '### Start\nbody line'

    def test_extracts_empty_section_for_adjacent_headers(self):
        """Adjacent headers with no body between them yield the header text only."""
        payload = '### Empty\n### Next\nbody'
        result = _extract_section(payload, '### Empty')
        assert result == '### Empty'

    def test_extracts_first_occurrence_when_header_repeats(self):
        """First-occurrence semantics: slice ends at the second '\\n#' boundary, not EOF."""
        payload = '### Dup\nfirst body\n### Dup\nsecond body'
        result = _extract_section(payload, '### Dup')
        assert result == '### Dup\nfirst body'


class TestInvariantAfterTask643:
    """Regression guard for the FilteredTaskTree done_count/done_tasks invariant.

    Task 643 removed the dead ``elif filtered.done_count > 0`` branch from
    ``TaskKnowledgeSync.assemble_payload`` on the grounds that
    ``filter_task_tree()`` guarantees ``done_count > 0 → len(done_tasks) > 0``
    (they are always appended together, capped at ``MAX_DONE_TASKS_RETAINED=30``).
    Task 782 hardens this invariant with a defensive callsite warning and places
    regression guards here at the stage/callsite layer.
    """

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

    @pytest.mark.asyncio
    async def test_warns_when_filtered_task_tree_violates_cancelled_invariant(
        self, mock_deps, watermark, caplog
    ):
        """Integration guard: a FilteredTaskTree with cancelled_count>0 but empty cancelled_tasks triggers a WARNING.

        This test exercises the full ``assemble_payload`` method intentionally — it
        verifies that ``_check_filtered_tree_invariant`` is correctly wired into the
        ``assemble_payload`` call chain for the cancelled pair, not just that the helper
        itself works.  For isolated testing of the helper, see
        ``test_check_filtered_tree_invariant_warns_on_cancelled_violation``.

        The invariant-violating state can only be reached by external callers that
        construct a ``FilteredTaskTree`` directly (bypassing ``filter_task_tree``).
        """
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'test_project'
        stage.project_root = '/tmp/test_project'
        # Construct invariant-violating tree: cancelled_count > 0 but cancelled_tasks is empty.
        # This state is impossible via filter_task_tree() but can arise from external
        # construction — exactly the case the task-828 defensive check guards against.
        # total_count = 1 active + 0 done + 4 cancelled + 0 other = 5
        stage.filtered_task_tree = FilteredTaskTree(
            active_tasks=[self._make_task(1, 'in-progress')],
            done_tasks=[],
            done_count=0,
            cancelled_tasks=[],
            cancelled_count=4,
            other_count=0,
            total_count=5,
        )

        with caplog.at_level(
            logging.WARNING,
            logger='fused_memory.reconciliation.stages.task_knowledge_sync',
        ):
            payload = await stage.assemble_payload([], watermark, [])

        # The warning must be emitted…
        assert any(
            rec.levelno == logging.WARNING
            and rec.name == 'fused_memory.reconciliation.stages.task_knowledge_sync'
            and 'cancelled_count' in rec.message
            and 'cancelled_tasks' in rec.message
            for rec in caplog.records
        ), (
            'Expected a WARNING about cancelled_count/cancelled_tasks invariant from '
            'fused_memory.reconciliation.stages.task_knowledge_sync, '
            f'got records: {[(r.name, r.levelno, r.message) for r in caplog.records]}'
        )
        # …but the warning must be non-fatal: assemble_payload still returns a valid payload.
        assert payload and 'Stage 2' in payload, (
            f'assemble_payload should complete and return a Stage 2 payload even when '
            f'the cancelled invariant is violated; got: {payload!r}'
        )

    @pytest.mark.asyncio
    async def test_warns_when_filtered_task_tree_violates_invariant(
        self, mock_deps, watermark, caplog
    ):
        """Integration guard: a FilteredTaskTree with done_count>0 but empty done_tasks triggers a WARNING.

        This test exercises the full ``assemble_payload`` method intentionally — it
        verifies that ``_check_filtered_tree_invariant`` is correctly wired into the
        ``assemble_payload`` call chain, not just that the helper itself works.  For
        isolated testing of the helper, see
        ``test_check_filtered_tree_invariant_warns_on_violation``.

        The invariant-violating state can only be reached by external callers that
        construct a ``FilteredTaskTree`` directly (bypassing ``filter_task_tree``).
        """
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'test_project'
        stage.project_root = '/tmp/test_project'
        # Construct invariant-violating tree: done_count > 0 but done_tasks is empty.
        # This state is impossible via filter_task_tree() but can arise from external
        # construction — exactly the case the task-782 defensive check guards against.
        stage.filtered_task_tree = FilteredTaskTree(
            active_tasks=[self._make_task(1, 'in-progress')],
            done_tasks=[],
            done_count=5,
            cancelled_tasks=[],
            cancelled_count=0,
            other_count=0,
            total_count=6,
        )

        with caplog.at_level(
            logging.WARNING,
            logger='fused_memory.reconciliation.stages.task_knowledge_sync',
        ):
            await stage.assemble_payload([], watermark, [])

        assert any(
            rec.levelno == logging.WARNING
            and rec.name == 'fused_memory.reconciliation.stages.task_knowledge_sync'
            and 'done_count' in rec.message
            and 'done_tasks' in rec.message
            for rec in caplog.records
        ), (
            'Expected a WARNING about done_count/done_tasks invariant from '
            'fused_memory.reconciliation.stages.task_knowledge_sync, '
            f'got records: {[(r.name, r.levelno, r.message) for r in caplog.records]}'
        )

    def test_check_filtered_tree_invariant_warns_on_cancelled_violation(self, mock_deps, caplog):
        """Unit test for _check_filtered_tree_invariant: warns when cancelled invariant is violated.

        Calls the private helper directly with a FilteredTaskTree that has
        cancelled_count=3 but cancelled_tasks=[] — an impossible state from
        filter_task_tree() but reachable via external construction.  Asserts
        that a WARNING containing 'cancelled_count' and 'cancelled_tasks' is
        emitted.  Mirrors test_check_filtered_tree_invariant_warns_on_violation
        for the done pair.
        """
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        violating_tree = FilteredTaskTree(
            active_tasks=[],
            done_tasks=[],
            done_count=0,
            cancelled_tasks=[],
            cancelled_count=3,
            other_count=0,
            total_count=3,
        )
        with caplog.at_level(
            logging.WARNING,
            logger='fused_memory.reconciliation.stages.task_knowledge_sync',
        ):
            stage._check_filtered_tree_invariant(violating_tree)

        assert any(
            rec.levelno == logging.WARNING
            and 'cancelled_count' in rec.message
            and 'cancelled_tasks' in rec.message
            for rec in caplog.records
        )

    def test_check_filtered_tree_invariant_warns_on_violation(self, mock_deps, caplog):
        """Unit test for _check_filtered_tree_invariant: warns when invariant is violated.

        Calls the private helper directly — no ``assemble_payload`` involved — so
        changes to the rest of ``assemble_payload``'s rendering logic cannot break
        this test.
        """
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        violating_tree = FilteredTaskTree(
            active_tasks=[],
            done_tasks=[],
            done_count=3,
            cancelled_tasks=[],
            cancelled_count=0,
            other_count=0,
            total_count=3,
        )
        with caplog.at_level(
            logging.WARNING,
            logger='fused_memory.reconciliation.stages.task_knowledge_sync',
        ):
            stage._check_filtered_tree_invariant(violating_tree)

        assert any(
            rec.levelno == logging.WARNING
            and 'done_count' in rec.message
            and 'done_tasks' in rec.message
            for rec in caplog.records
        )

    def test_check_filtered_tree_invariant_no_warning_when_cancelled_ok(self, mock_deps, caplog):
        """Unit test for _check_filtered_tree_invariant: no warning when cancelled invariant holds.

        Constructs a FilteredTaskTree with cancelled_count=2 and cancelled_tasks populated
        with 2 tasks — the invariant holds.  Asserts no WARNING records are emitted.
        Verifies the new check does not false-positive on valid trees.  Mirrors
        test_check_filtered_tree_invariant_no_warning_when_ok for the done pair.
        """
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        ok_tree = FilteredTaskTree(
            active_tasks=[],
            done_tasks=[],
            done_count=0,
            cancelled_tasks=[self._make_task(1, 'cancelled'), self._make_task(2, 'cancelled')],
            cancelled_count=2,
            other_count=0,
            total_count=2,
        )
        with caplog.at_level(
            logging.WARNING,
            logger='fused_memory.reconciliation.stages.task_knowledge_sync',
        ):
            stage._check_filtered_tree_invariant(ok_tree)

        assert not any(rec.levelno == logging.WARNING for rec in caplog.records)

    def test_check_filtered_tree_invariant_no_warning_when_ok(self, mock_deps, caplog):
        """Unit test for _check_filtered_tree_invariant: no warning when invariant holds."""
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        ok_tree = FilteredTaskTree(
            active_tasks=[],
            done_tasks=[self._make_task(1, 'done')],
            done_count=1,
            cancelled_tasks=[],
            cancelled_count=0,
            other_count=0,
            total_count=1,
        )
        with caplog.at_level(
            logging.WARNING,
            logger='fused_memory.reconciliation.stages.task_knowledge_sync',
        ):
            stage._check_filtered_tree_invariant(ok_tree)

        assert not any(rec.levelno == logging.WARNING for rec in caplog.records)

    def test_filter_task_tree_invariant_cancelled_count_and_cancelled_tasks_populated_together(self):
        """Regression guard: filter_task_tree() sets cancelled_count>0 ↔ cancelled_tasks non-empty.

        Mirrors the done-pair regression guard.  Verifies that any future refactor of
        filter_task_tree that breaks the cancelled_count↔cancelled_tasks invariant trips
        this test in addition to guards in test_task_filter.py.
        """
        tasks_data = {
            'tasks': [
                self._make_task(1, 'cancelled'),
                self._make_task(2, 'cancelled'),
                self._make_task(3, 'cancelled'),
            ]
        }
        result = filter_task_tree(tasks_data)

        assert result.cancelled_count > 0, (
            f'Expected cancelled_count > 0 for 3 cancelled tasks, got {result.cancelled_count}'
        )
        assert len(result.cancelled_tasks) > 0, (
            f'Expected non-empty cancelled_tasks for cancelled_count={result.cancelled_count}, '
            f'got cancelled_tasks={result.cancelled_tasks!r}'
        )

    def test_filter_task_tree_invariant_done_count_and_done_tasks_populated_together(self):
        """Regression guard: filter_task_tree() sets done_count>0 ↔ done_tasks non-empty.

        Task 643 removed a dead ``elif filtered.done_count > 0`` branch from
        ``TaskKnowledgeSync.assemble_payload`` on the basis of this invariant.
        Task 782 places this regression guard at the stage/callsite layer so that
        any future refactor of ``filter_task_tree`` that breaks the invariant trips
        this test in addition to the guards in test_task_filter.py.
        """
        tasks_data = {
            'tasks': [
                self._make_task(1, 'done'),
                self._make_task(2, 'done'),
                self._make_task(3, 'done'),
            ]
        }
        result = filter_task_tree(tasks_data)

        assert result.done_count > 0, (
            f'Expected done_count > 0 for 3 done tasks, got {result.done_count}'
        )
        assert len(result.done_tasks) > 0, (
            f'Expected non-empty done_tasks for done_count={result.done_count}, '
            f'got done_tasks={result.done_tasks!r}'
        )

    def test_filter_task_tree_invariant_holds_with_over_cap_cancelled_tasks(self):
        """Regression guard: at the >MAX_CANCELLED_TASKS_RETAINED boundary the invariant still holds.

        Even when cancelled_count exceeds MAX_CANCELLED_TASKS_RETAINED=15 (tasks are capped
        in cancelled_tasks), cancelled_tasks must remain non-empty.  Mirrors the done-pair
        over-cap test placed by task-782 and guards against future refactors of
        filter_task_tree that might inadvertently empty the cancelled list under the cap.
        """
        n_tasks = MAX_CANCELLED_TASKS_RETAINED + 5
        tasks_data = {
            'tasks': [self._make_task(i, 'cancelled') for i in range(1, n_tasks + 1)]
        }
        result = filter_task_tree(tasks_data)

        assert result.cancelled_count > MAX_CANCELLED_TASKS_RETAINED, (
            f'Expected cancelled_count > {MAX_CANCELLED_TASKS_RETAINED}, got {result.cancelled_count}'
        )
        assert len(result.cancelled_tasks) == MAX_CANCELLED_TASKS_RETAINED, (
            f'Expected cancelled_tasks capped at {MAX_CANCELLED_TASKS_RETAINED}, '
            f'got {len(result.cancelled_tasks)}'
        )
        # Invariant holds implicitly: the assertion above already proves cancelled_tasks
        # is non-empty (MAX_CANCELLED_TASKS_RETAINED == 15).

    def test_filter_task_tree_invariant_holds_with_over_cap_done_tasks(self):
        """Regression guard: at the >MAX_DONE_TASKS_RETAINED boundary the invariant still holds.

        Even when done_count exceeds MAX_DONE_TASKS_RETAINED=30 (tasks are capped in
        done_tasks), done_tasks must remain non-empty.  This is the cap-boundary case of
        the invariant that task-643 relied on.  Task 782 places this guard at the
        stage/callsite layer to complement test_task_filter.py's existing cap tests.
        """
        n_tasks = MAX_DONE_TASKS_RETAINED + 5
        tasks_data = {
            'tasks': [self._make_task(i, 'done') for i in range(1, n_tasks + 1)]
        }
        result = filter_task_tree(tasks_data)

        assert result.done_count > MAX_DONE_TASKS_RETAINED, (
            f'Expected done_count > {MAX_DONE_TASKS_RETAINED}, got {result.done_count}'
        )
        assert len(result.done_tasks) == MAX_DONE_TASKS_RETAINED, (
            f'Expected done_tasks capped at {MAX_DONE_TASKS_RETAINED}, '
            f'got {len(result.done_tasks)}'
        )
        # Invariant holds implicitly: the assertion above already proves done_tasks
        # is non-empty (MAX_DONE_TASKS_RETAINED == 30).


# ---------------------------------------------------------------------------
# Cap-exception propagation from run_stage_via_cli
# ---------------------------------------------------------------------------


class TestRunStageCapHandling:
    """Verify run_stage_via_cli re-raises AllAccountsCappedException."""

    @pytest.mark.asyncio
    async def test_run_stage_via_cli_reraises_all_accounts_capped(self, tmp_path):
        """AllAccountsCappedException must propagate out of run_stage_via_cli.

        Before step-12 impl: the current broad `except Exception` swallows the
        exception into a StageResult(error=str(e)) — no re-raise.
        After step-12 impl: the exception propagates, allowing the harness to
        handle deferral gracefully.
        """
        config = ReconciliationConfig(
            enabled=True,
            explore_codebase_root=str(tmp_path),
            agent_llm_model='sonnet',
            agent_max_steps=5,
            stage_timeout_seconds=600,
        )

        cap_exc = AllAccountsCappedException(
            retries=5, elapsed_secs=180.0, label='Reconciliation stage (sonnet)'
        )

        with patch(
            'fused_memory.reconciliation.cli_stage_runner.invoke_with_cap_retry',
            new=AsyncMock(side_effect=cap_exc),
        ), pytest.raises(AllAccountsCappedException):
            await run_stage_via_cli(
                system_prompt='x',
                payload='y',
                disallowed_tools=[],
                config=config,
                mcp_config={'mcpServers': {}},
            )


# ---------------------------------------------------------------------------
# _format_flagged: no-silent-truncation tests (step-1)
# ---------------------------------------------------------------------------


class TestFormatFlaggedNoSilentTruncation:
    """_format_flagged renders all items — no silent [:50] truncation."""

    def test_all_100_items_present_in_text(self):
        """100 items must all appear in the rendered text — no [:50] cap."""
        items = [{'description': f'item-{i}', 'severity': 'minor'} for i in range(100)]
        text, _count = _format_flagged(items)
        for i in range(100):
            assert f'item-{i}' in text, (
                f'Expected item-{i} description in rendered text; got:\n{text[:500]}...'
            )

    def test_no_truncation_footer_when_all_items_rendered(self):
        """When all 100 items render, there must be no '... and N more' footer line."""
        items = [{'description': f'item-{i}', 'severity': 'minor'} for i in range(100)]
        text, _count = _format_flagged(items)
        assert '... and ' not in text, (
            f'Unexpected truncation footer found in rendered text; got:\n{text[:500]}...'
        )

    def test_empty_list_returns_no_flagged_items(self):
        """Empty list must return the sentinel string."""
        text, count = _format_flagged([])
        assert text == 'No flagged items.'
        assert count == 0


# ---------------------------------------------------------------------------
# _format_flagged: char budget tests (step-3)
# ---------------------------------------------------------------------------


class TestFormatFlaggedCharBudget:
    """_format_flagged applies a char budget and emits a warning when truncating."""

    def test_under_budget_no_warning(self, caplog):
        """10 small items stay well under the 40000-char budget — no warning."""
        items = [{'description': f'small-{i}', 'severity': 'minor'} for i in range(10)]
        with caplog.at_level(
            logging.WARNING,
            logger='fused_memory.reconciliation.stages.task_knowledge_sync',
        ):
            _format_flagged(items)

        assert not any(
            rec.levelno == logging.WARNING
            and rec.name == 'fused_memory.reconciliation.stages.task_knowledge_sync'
            for rec in caplog.records
        ), (
            'Expected no WARNING for 10 small items; '
            f'got: {[(r.name, r.levelno, r.message) for r in caplog.records]}'
        )

    def test_over_budget_text_capped(self):
        """200 items with ~300-byte descriptions exceed 40000 chars; text is capped."""
        # Each item produces ~320+ chars when JSON-serialised + '- ' prefix
        items = [{'description': 'x' * 300, 'index': i} for i in range(200)]
        text, _count = _format_flagged(items)
        # Tight upper bound: running_chars ≤ budget + \n separator + footer line.
        # running_chars can be at most budget_chars when truncation fires.
        max_dropped = len(items)  # worst-case: only first item fully rendered
        max_footer = len(f'... and {max_dropped} more (truncated: char budget)')
        tight_bound = _FLAGGED_ITEMS_CHAR_BUDGET + 1 + max_footer
        assert len(text) <= tight_bound, (
            f'Expected text ≤ {tight_bound} chars '
            f'(budget={_FLAGGED_ITEMS_CHAR_BUDGET} + 1 newline + footer={max_footer}) '
            f'but got {len(text)}'
        )

    def test_over_budget_has_footer(self):
        """Over-budget render must end with a truncation footer line."""
        items = [{'description': 'x' * 300, 'index': i} for i in range(200)]
        text, _count = _format_flagged(items)
        assert '... and ' in text, (
            f'Expected truncation footer in text; last 200 chars: {text[-200:]!r}'
        )

    def test_over_budget_emits_warning_with_structured_extras(self, caplog):
        """Over-budget render must emit exactly one WARNING with correct extra keys."""
        items = [{'description': 'x' * 300, 'index': i} for i in range(200)]
        with caplog.at_level(
            logging.WARNING,
            logger='fused_memory.reconciliation.stages.task_knowledge_sync',
        ):
            _format_flagged(items)  # result not needed here; warning is what we check

        warning_records = [
            rec for rec in caplog.records
            if rec.levelno == logging.WARNING
            and rec.name == 'fused_memory.reconciliation.stages.task_knowledge_sync'
        ]
        assert len(warning_records) == 1, (
            f'Expected exactly 1 WARNING; got {len(warning_records)}: '
            f'{[(r.message, getattr(r, "__dict__", {})) for r in warning_records]}'
        )
        rec = warning_records[0]
        # All four structured-extra keys must be present
        for key in ('total', 'rendered', 'dropped', 'budget_chars'):
            assert hasattr(rec, key), (
                f'Expected extra key {key!r} on WARNING record; '
                f'record __dict__: {rec.__dict__}'
            )
        total = rec.total
        rendered = rec.rendered
        dropped = rec.dropped
        budget_chars = rec.budget_chars
        assert total == rendered + dropped, (
            f'total={total} must equal rendered={rendered} + dropped={dropped}'
        )
        assert rendered > 0, f'rendered must be > 0, got {rendered}'
        assert dropped > 0, f'dropped must be > 0, got {dropped}'
        assert budget_chars == 40000, f'budget_chars must be 40000, got {budget_chars}'


# ---------------------------------------------------------------------------
# _format_flagged: first-item-exceeds-budget edge case (amendment: suggestion 3)
# ---------------------------------------------------------------------------


class TestFormatFlaggedFirstItemEdgeCase:
    """_format_flagged always renders at least a truncated fragment of the first item."""

    def test_single_oversized_item_renders_truncated_fragment(self, caplog):
        """A single item whose JSON exceeds the budget produces a truncated line, not a footer-only body."""
        # One item whose JSON far exceeds the 40000-char budget
        items = [{'description': 'y' * 50_000, 'severity': 'critical'}]
        with caplog.at_level(
            logging.WARNING,
            logger='fused_memory.reconciliation.stages.task_knowledge_sync',
        ):
            text, rendered_count = _format_flagged(items)

        # The LLM must see SOMETHING about the item — not just a footer
        assert '… [item truncated]' in text, (
            f'Expected truncation marker in text; got first 200 chars: {text[:200]!r}'
        )
        # No "... and N more" footer when there are no additional items to report
        assert '... and ' not in text, (
            f'Unexpected "... and N more" footer for single-item list; text: {text[:200]!r}'
        )
        # Text must not exceed the budget by more than the marker length
        assert len(text) <= _FLAGGED_ITEMS_CHAR_BUDGET + len('… [item truncated]') + 10, (
            f'Text too long: {len(text)} chars; expected ≤ {_FLAGGED_ITEMS_CHAR_BUDGET}'
        )
        # A truncation warning must still be emitted
        warning_records = [
            rec for rec in caplog.records
            if rec.levelno == logging.WARNING
            and rec.name == 'fused_memory.reconciliation.stages.task_knowledge_sync'
        ]
        assert len(warning_records) == 1, (
            f'Expected exactly 1 WARNING for oversized single item; got {len(warning_records)}'
        )

    def test_first_of_many_oversized_renders_fragment_plus_footer(self, caplog):
        """When first item of many exceeds budget, fragment + N-more footer is shown."""
        # First item is huge; second item is small
        items = [
            {'description': 'z' * 50_000, 'severity': 'critical'},
            {'description': 'small', 'severity': 'minor'},
        ]
        with caplog.at_level(
            logging.WARNING,
            logger='fused_memory.reconciliation.stages.task_knowledge_sync',
        ):
            text, rendered_count = _format_flagged(items)

        assert '… [item truncated]' in text, (
            'Expected truncation marker for oversized first item'
        )
        # The second item was not rendered; the footer should note it
        assert '... and 1 more (truncated: char budget)' in text, (
            f'Expected "... and 1 more" footer; got last 200 chars: {text[-200:]!r}'
        )


# ---------------------------------------------------------------------------
# _format_flagged: returns rendered_count as second element (step-5)
# ---------------------------------------------------------------------------


class TestFormatFlaggedReturnsRenderedCount:
    """_format_flagged returns a (str, int) tuple where int is rendered_count."""

    def test_empty_list_returns_tuple_with_zero_count(self):
        """Empty list → ('No flagged items.', 0)."""
        result = _format_flagged([])
        assert isinstance(result, tuple), f'Expected tuple, got {type(result)}'
        text, count = result
        assert text == 'No flagged items.'
        assert count == 0

    def test_two_items_returns_rendered_count_two(self):
        """2 items under budget → (str, 2)."""
        items = [{'description': 'a'}, {'description': 'b'}]
        result = _format_flagged(items)
        assert isinstance(result, tuple), f'Expected tuple, got {type(result)}'
        text, count = result
        assert isinstance(text, str)
        assert count == 2

    def test_over_budget_rendered_count_less_than_total(self, caplog):
        """Over-budget render → rendered_count < total and matches warning extra."""
        items = [{'description': 'x' * 300, 'index': i} for i in range(200)]
        with caplog.at_level(
            logging.WARNING,
            logger='fused_memory.reconciliation.stages.task_knowledge_sync',
        ):
            result = _format_flagged(items)

        assert isinstance(result, tuple), f'Expected tuple, got {type(result)}'
        text, rendered_count = result
        assert rendered_count < 200, (
            f'Expected rendered_count < 200 for over-budget input; got {rendered_count}'
        )
        # The rendered_count must match the warning's extra 'rendered' value
        warning_records = [
            rec for rec in caplog.records
            if rec.levelno == logging.WARNING
            and rec.name == 'fused_memory.reconciliation.stages.task_knowledge_sync'
        ]
        assert len(warning_records) == 1
        assert warning_records[0].rendered == rendered_count, (
            f"Warning extra 'rendered'={warning_records[0].rendered} must match "
            f'returned rendered_count={rendered_count}'
        )


# ---------------------------------------------------------------------------
# Stage 2 handoff shortfall warning (step-7)
# ---------------------------------------------------------------------------


def _make_stage1_report_with_n_large_items(n: int) -> StageReport:
    """Build a Stage 1 StageReport whose items_flagged list has *n* large dicts.

    Each item's 'description' is ~300 bytes so that 200 items exceed the
    40000-char budget when rendered by _format_flagged.
    """
    now = datetime.now(tz=UTC)
    return StageReport(
        stage=StageId.memory_consolidator,
        started_at=now,
        completed_at=now,
        items_flagged=[
            {'description': 'x' * 300, 'index': i, 'severity': 'critical'}
            for i in range(n)
        ],
    )


class TestStage2HandoffShortfallWarning:
    """TaskKnowledgeSync.assemble_payload warns via _format_flagged when items are truncated.

    After collapsing the two-warning design (suggestions 1+2), the single
    ``reconciliation.flagged_items_truncated`` warning emitted by ``_format_flagged``
    carries ``run_stage='stage2'`` so ops can correlate the drop to Stage 2 without
    a separate stage-specific shortfall warning.
    """

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
    async def test_shortfall_warning_emitted_when_budget_truncates(
        self, mock_deps, watermark, caplog
    ):
        """When stage1 items exceed the char budget, a truncation warning with run_stage='stage2' fires."""
        # 200 items × ~330 chars each = ~66000 chars — well over the 40000-char budget
        n_items = 200
        stage1_report = _make_stage1_report_with_n_large_items(n_items)

        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'reify'
        stage.project_root = '/home/leo/src/reify'
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}

        with caplog.at_level(
            logging.WARNING,
            logger='fused_memory.reconciliation.stages.task_knowledge_sync',
        ):
            await stage.assemble_payload([], watermark, [stage1_report])

        # The single collapsed warning comes from _format_flagged with run_stage='stage2'
        truncation_records = [
            rec for rec in caplog.records
            if rec.levelno == logging.WARNING
            and rec.name == 'fused_memory.reconciliation.stages.task_knowledge_sync'
            and 'reconciliation.flagged_items_truncated' in rec.getMessage()
            and getattr(rec, 'run_stage', None) == 'stage2'
        ]
        assert len(truncation_records) == 1, (
            f'Expected exactly 1 flagged_items_truncated WARNING with run_stage=stage2; '
            f'got {len(truncation_records)}: '
            f'{[(r.getMessage(), r.__dict__) for r in truncation_records]}'
        )
        rec = truncation_records[0]
        # All structured-extra keys must be present
        for key in ('total', 'rendered', 'dropped', 'budget_chars', 'run_stage'):
            assert hasattr(rec, key), (
                f'Expected extra key {key!r} on truncation WARNING; '
                f'record __dict__: {rec.__dict__}'
            )
        assert rec.total == n_items
        assert rec.rendered < n_items
        assert rec.dropped == n_items - rec.rendered
        assert rec.run_stage == 'stage2'

    @pytest.mark.asyncio
    async def test_no_shortfall_warning_when_all_items_rendered(
        self, mock_deps, watermark, caplog
    ):
        """When all stage1 items fit in the budget, no truncation warning is emitted."""
        # 5 small items — far under the 40000-char budget
        now = datetime.now(tz=UTC)
        stage1_report = StageReport(
            stage=StageId.memory_consolidator,
            started_at=now,
            completed_at=now,
            items_flagged=[
                {'description': f'flag-{i}', 'severity': 'minor'} for i in range(5)
            ],
        )

        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'reify'
        stage.project_root = '/home/leo/src/reify'
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}

        with caplog.at_level(
            logging.WARNING,
            logger='fused_memory.reconciliation.stages.task_knowledge_sync',
        ):
            await stage.assemble_payload([], watermark, [stage1_report])

        truncation_records = [
            rec for rec in caplog.records
            if rec.levelno == logging.WARNING
            and rec.name == 'fused_memory.reconciliation.stages.task_knowledge_sync'
            and 'flagged_items_truncated' in rec.getMessage()
            and getattr(rec, 'run_stage', None) == 'stage2'
        ]
        assert len(truncation_records) == 0, (
            f'Expected no flagged_items_truncated WARNING (run_stage=stage2) for 5 small items; '
            f'got: {[(r.getMessage(), r.__dict__) for r in truncation_records]}'
        )
