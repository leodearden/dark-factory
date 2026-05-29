"""Tests for reconciliation stage configuration (CLI-native MCP execution)."""

import json
import logging
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _fm_helpers import assert_id_title_pairing, make_8df8_scenario
from shared.cli_invoke import AgentResult, AllAccountsCappedException

import fused_memory.reconciliation.stages.base as base_module
from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import StageId, StageReport, Watermark
from fused_memory.reconciliation.cli_stage_runner import (
    DISALLOW_BUILTIN,
    DISALLOW_MEMORY_WRITES,
    DISALLOW_SUBTASK_CREATE,
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
from fused_memory.reconciliation.prompts.stage3 import STAGE3_SYSTEM_PROMPT
from fused_memory.reconciliation.stages.base import BaseStage
from fused_memory.reconciliation.stages.memory_consolidator import MemoryConsolidator
from fused_memory.reconciliation.stages.task_knowledge_sync import (
    _FLAGGED_ITEMS_CHAR_BUDGET,
    IntegrityCheck,
    TaskKnowledgeSync,
    _check_flag_counter_completeness,
    _check_stall_guard_freshness,
    _classify_terminal_state_violations,
    _format_flagged,
    _needs_hint_conversion,
    _queue_briefing_refresh_tasks,
    _render_done_provenance_section,
    _resolve_live_status,
    _run_briefing_known_gaps_script,
    _select_proactive_sample,
    _suppress_same_run_human_operator_dups,
    _verify_set_task_status_post_action,
)
from fused_memory.reconciliation.task_filter import (
    MAX_CANCELLED_TASKS_RETAINED,
    MAX_DONE_TASKS_RETAINED,
    FilteredTaskTree,
    filter_task_tree,
    format_task_list,
    id_key,
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


def make_configured_task_knowledge_sync_stage(
    deps: dict, *, project_id: str, project_root: str, run_id: str = 'test-run'
) -> "TaskKnowledgeSync":
    """Create a TaskKnowledgeSync stage with _current_run_id pre-populated.

    Using this helper instead of inline setup ensures _current_run_id is always
    set, preventing the RuntimeError guard from firing unexpectedly.

    The default run_id='test-run' is intentional: it matches the run_id written
    into test flag objects so that flags are not excluded by the run-partition
    filter (only the relevant scope/task filters are exercised).  Pass a
    different run_id to tests that specifically need to control partition
    boundaries or that construct their own flags with a custom run_id.

    Args:
        deps: Keyword dependencies dict (memory_service, taskmaster, journal, config).
        project_id: Project identifier string (e.g. 'reify', 'dark_factory').
        project_root: Absolute path to the project root (e.g. '/home/leo/src/reify').
        run_id: Value for _current_run_id; defaults to 'test-run'.

    Returns:
        A TaskKnowledgeSync instance ready for use in assemble_payload() tests.
    """
    stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **deps)
    stage.project_id = project_id
    stage.project_root = project_root
    stage._current_run_id = run_id
    return stage


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

    def test_stage2_disallows_builtins_plus_subtask_create(self):
        """STAGE2_DISALLOWED must equal DISALLOW_BUILTIN + DISALLOW_SUBTASK_CREATE.

        Stage 2 has full memory + task access except for add_subtask, which is
        blocked because the orchestrator scheduler is top-level-only (see
        DISALLOW_SUBTASK_CREATE comment in cli_stage_runner.py).
        """
        assert STAGE2_DISALLOWED == DISALLOW_BUILTIN + DISALLOW_SUBTASK_CREATE

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

    def test_add_task_not_in_disallow_task_writes(self):
        """add_task facade has been removed — the disallow list must no longer reference a non-existent tool."""
        assert 'mcp__fused-memory__add_task' not in DISALLOW_TASK_WRITES

    def test_stage2_blocks_add_subtask(self):
        """add_subtask must be blocked in Stage 2.

        The orchestrator scheduler is top-level-only (iterates ``tasks`` without
        descending into ``t['subtasks']``).  Any subtask created during Stage 2
        reconciliation is permanently invisible to the dispatcher and will never
        be executed — a silent orphan.  Closing the CREATION path in Stage 2
        prevents a planning-budget-overflow escalation from re-introducing the
        trap (see procedural memory fca61c20).
        """
        assert 'mcp__fused-memory__add_subtask' in STAGE2_DISALLOWED


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
        for field in ('description', 'severity', 'actionable', 'category', 'suggested_action'):
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
        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='reify', project_root='/home/leo/src/reify')
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}

        await stage.assemble_payload([], watermark, [])

        mock_deps['taskmaster'].get_tasks.assert_called_once_with(
            project_root='/home/leo/src/reify'
        )

    @pytest.mark.asyncio
    async def test_payload_uses_dynamic_project_root_in_instructions(self, mock_deps, watermark):
        """assemble_payload() instruction text must use self.project_root, not hardcoded path."""
        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='reify', project_root='/home/leo/src/reify')
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}

        payload = await stage.assemble_payload([], watermark, [])

        assert 'project_root="/home/leo/src/reify"' in payload
        assert 'project_root="/home/leo/src/dark-factory"' not in payload

    @pytest.mark.asyncio
    async def test_payload_dark_factory_project_still_works(self, mock_deps, watermark):
        """When project_root IS dark-factory, payload still contains the correct path."""
        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='dark_factory', project_root='/home/leo/src/dark-factory')
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}
        wm = Watermark(project_id='dark_factory')

        payload = await stage.assemble_payload([], wm, [])

        assert 'project_root="/home/leo/src/dark-factory"' in payload

    @pytest.mark.asyncio
    async def test_payload_contains_project_id_for_memory_tools(self, mock_deps, watermark):
        """assemble_payload() instruction text still uses self.project_id for fused-memory calls."""
        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='reify', project_root='/home/leo/src/reify')
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}

        payload = await stage.assemble_payload([], watermark, [])

        # The project_id should appear in the memory tools instruction (line 98)
        assert 'project_id="reify"' in payload


class TestTaskKnowledgeSyncKnownProjectsSection:
    """Stage 2 surfaces a "Known Projects" section so the LLM can re-route findings."""

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
    async def test_section_omitted_when_only_one_project_known(self, mock_deps, watermark):
        """A single-project deployment doesn't have a cross-project dimension —
        the section would only add noise."""
        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='reify', project_root='/home/leo/src/reify')
        stage.known_projects = {'reify': '/home/leo/src/reify'}
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}

        payload = await stage.assemble_payload([], watermark, [])
        assert '### Known Projects' not in payload

    @pytest.mark.asyncio
    async def test_section_omitted_when_no_known_projects(self, mock_deps, watermark):
        """Default empty known_projects (harness hasn't set it) → no section."""
        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='reify', project_root='/home/leo/src/reify')
        # known_projects unset → empty dict from BaseStage default
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}

        payload = await stage.assemble_payload([], watermark, [])
        assert '### Known Projects' not in payload

    @pytest.mark.asyncio
    async def test_section_rendered_when_multiple_projects_known(
        self, mock_deps, watermark,
    ):
        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='reify', project_root='/home/leo/src/reify')
        stage.known_projects = {
            'reify': '/home/leo/src/reify',
            'dark_factory': '/home/leo/src/dark-factory',
            'autopilot_video': '/home/leo/src/autopilot-video',
        }
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}

        payload = await stage.assemble_payload([], watermark, [])
        assert '### Known Projects (for cross-project routing)' in payload
        assert 'reify' in payload
        assert '/home/leo/src/reify' in payload
        assert 'dark_factory' in payload
        assert '/home/leo/src/dark-factory' in payload
        assert 'autopilot_video' in payload

    @pytest.mark.asyncio
    async def test_section_marks_current_project(self, mock_deps, watermark):
        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='reify', project_root='/home/leo/src/reify')
        stage.known_projects = {
            'reify': '/home/leo/src/reify',
            'dark_factory': '/home/leo/src/dark-factory',
        }
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}

        payload = await stage.assemble_payload([], watermark, [])
        # The current project's row carries the "(current)" marker; others don't.
        lines = [
            line for line in payload.splitlines()
            if line.strip().startswith('-') and (
                '/home/leo/src/reify' in line
                or '/home/leo/src/dark-factory' in line
            )
        ]
        # First listed should be the current project with its marker.
        assert any('reify' in line and '(current)' in line for line in lines)
        assert all(
            '(current)' not in line for line in lines if 'dark_factory' in line
        )

    @pytest.mark.asyncio
    async def test_payload_instructs_cross_project_routing(self, mock_deps, watermark):
        """The trailing project_root line points the LLM at "Known Projects"
        for cross-project routing."""
        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='reify', project_root='/home/leo/src/reify')
        stage.known_projects = {
            'reify': '/home/leo/src/reify',
            'dark_factory': '/home/leo/src/dark-factory',
        }
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}

        payload = await stage.assemble_payload([], watermark, [])
        assert 'tasks scoped to this project' in payload
        assert 'For cross-project routing see "Known Projects"' in payload


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
        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='p', project_root=str(tmp_path))
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
        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='p', project_root=str(tmp_path))
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
        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='p', project_root=str(tmp_path))
        mock_deps['taskmaster'].get_tasks.return_value = {
            'tasks': [{'id': 11, 'status': 'done', 'title': 'Legacy'}],
        }

        payload = await stage.assemble_payload([], Watermark(project_id='p'), [])

        section = _extract_section(payload, '### Done-task Provenance')
        assert 'provenance: unknown (legacy)' in section

    @pytest.mark.asyncio
    async def test_no_done_tasks_omits_section(self, mock_deps, tmp_path):
        """Empty done_tasks → provenance section is not injected."""
        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='p', project_root=str(tmp_path))
        mock_deps['taskmaster'].get_tasks.return_value = {
            'tasks': [{'id': 1, 'status': 'pending', 'title': 'WIP'}],
        }

        payload = await stage.assemble_payload([], Watermark(project_id='p'), [])

        assert '### Done-task Provenance' not in payload

    @pytest.mark.asyncio
    async def test_invalid_commit_gracefully_omits_file_list(self, mock_deps, tmp_path):
        """Unresolvable commit → section header emitted, no file list, no exception."""
        self._init_repo(tmp_path)
        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='p', project_root=str(tmp_path))
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
        assert result.stats == {'entity_summary_snapshot_lines_stripped': 0}
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
        assert result.stats == {'entity_summary_snapshot_lines_stripped': 0}
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
        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='test_project', project_root='/tmp/test_project')
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
        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='test_project', project_root='/tmp/test_project')
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

    def test_stage2_directs_cancellation_via_set_task_status(self):
        """Stage 2 prompt must direct agents to ``set_task_status('cancelled')``.

        Forensics 2026-05-08: server now rejects ``update_task(status=…)``,
        so any historical "delete via update_task(status='cancelled', …)"
        guidance is broken. Redirect agents to set_task_status — the only
        sanctioned writer for terminal status. The ``cancellation_reason``
        metadata field has zero downstream consumers, so explicit
        cancellation is enough; rationale belongs in
        ``add_memory(category='observations_and_summaries')``.
        """
        from fused_memory.reconciliation.prompts.stage2 import STAGE2_SYSTEM_PROMPT

        assert "set_task_status('cancelled')" in STAGE2_SYSTEM_PROMPT, (
            "Stage 2 prompt must explicitly direct agents to "
            "set_task_status('cancelled') for cancellation."
        )
        # Server rejects update_task(status=…) — calling out the path here
        # prevents agents from rediscovering the broken bypass route.
        assert 'update_task(status=' not in STAGE2_SYSTEM_PROMPT, (
            'Stage 2 prompt must not reference update_task(status=…); the '
            'server now rejects that call shape.'
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
        """_select_proactive_sample sorts non-parseable string ids identically to id_key fallback=0.

        Non-int ids map to 0 via id_key, so they sort last (after all positive-int ids)
        within the same status bucket. This documents the expected behaviour and acts as
        a regression guard before the inline sort_key is replaced with id_key in step-4.
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
        # because id_key('abc') == id_key('xyz') == 0 < 2 < 5, sorted descending
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
        # Verify id_key agrees: all non-int ids yield 0
        for t in tasks:
            if isinstance(t['id'], str):
                assert id_key(t) == 0, f'id_key should return 0 for non-int id {t["id"]!r}'

    # --- Step 13: empty task tree handled gracefully ---

    @pytest.mark.asyncio
    async def test_proactive_sample_empty_task_tree(self, mock_deps, watermark):
        """When taskmaster returns 0 tasks, proactive sample section shows 'No tasks.'."""
        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='test_project', project_root='/tmp/test_project')
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
        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='test_project', project_root='/tmp/test_project')
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


class TestNeedsHintConversion:
    """Tests for _needs_hint_conversion helper.

    Verifies the three-branch classification from the task 1275 pseudo-code:
      1. list memory_hints  -> True  (legacy list-of-dict format, conversion target)
      2. falsy memory_hints -> True  (missing key or empty dict, existing falsy path)
      3. truthy non-list   -> False (assumed already-structured dict, skip)
    """

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _make_task(memory_hints) -> dict:
        return {'id': 1, 'title': 'T', 'status': 'pending', 'metadata': {'memory_hints': memory_hints}}

    @staticmethod
    def _make_task_no_hints() -> dict:
        return {'id': 1, 'title': 'T', 'status': 'pending', 'metadata': {}}

    # ------------------------------------------------------------------ branch 1: list

    def test_list_memory_hints_classified_as_conversion_target(self):
        """Non-empty list-of-dict memory_hints returns True (NEW branch — legacy format)."""
        task = self._make_task([{'entity': 'Foo', 'query': 'what is Foo'}])
        assert _needs_hint_conversion(task) is True

    def test_empty_list_memory_hints_classified_as_conversion_target(self):
        """Empty list memory_hints returns True via the list branch (not the falsy branch)."""
        task = self._make_task([])
        assert _needs_hint_conversion(task) is True

    # ------------------------------------------------------------------ branch 2: falsy

    def test_missing_memory_hints_classified_as_conversion_target(self):
        """Task with metadata dict that has no memory_hints key returns True."""
        task = self._make_task_no_hints()
        assert _needs_hint_conversion(task) is True

    def test_empty_dict_memory_hints_classified_as_conversion_target(self):
        """Empty dict memory_hints returns True (existing falsy path)."""
        task = self._make_task({})
        assert _needs_hint_conversion(task) is True

    # ------------------------------------------------------------------ branch 3: already-valid dict

    def test_structured_dict_memory_hints_not_flagged(self):
        """Task with {entities: [...], queries: [...]} dict returns False (already valid)."""
        task = self._make_task({'entities': ['Foo'], 'queries': ['what is Foo']})
        assert _needs_hint_conversion(task) is False

    def test_truthy_non_dict_non_list_memory_hints_not_flagged(self):
        """Truthy non-list, non-dict values (string, int) return False — pins branch-3 contract.

        Per design decision 4 in plan 1275: the three-branch pseudo-code's 'else' clause is
        unconditional for any truthy non-list value; adding an isinstance(task_hints, dict) guard
        to flag malformed scalars is a separable robustness concern deferred to a follow-up task.
        This test documents the current contract so future narrowing is a deliberate, visible change.
        """
        assert _needs_hint_conversion(self._make_task('oops')) is False
        assert _needs_hint_conversion(self._make_task(42)) is False

    # ------------------------------------------------------------------ defensive edge cases

    def test_task_without_metadata_key_classified_as_conversion_target(self):
        """Task dict with no 'metadata' key at all returns True (treated as no hints attached)."""
        task = {'id': 1, 'title': 'T', 'status': 'pending'}
        assert _needs_hint_conversion(task) is True

    def test_task_with_none_metadata_classified_as_conversion_target(self):
        """Task with metadata=None returns True (malformed metadata can't carry valid hints)."""
        task = {'id': 1, 'title': 'T', 'status': 'pending', 'metadata': None}
        assert _needs_hint_conversion(task) is True

    def test_task_with_non_dict_metadata_string_classified_as_conversion_target(self):
        """Task with metadata as a string returns True (defensive: non-dict metadata treated as no hints)."""
        task = {'id': 1, 'title': 'T', 'status': 'pending', 'metadata': 'not-a-dict'}
        assert _needs_hint_conversion(task) is True


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


class TestStage2GuardrailProjectIdGating:
    """Tests that the contamination guardrail is injected only for the
    autopilot_video project, not for other projects like dark_factory.

    The load-bearing behavioural contract:
    - build_stage2_system_prompt('autopilot_video') includes the guardrail section.
    - build_stage2_system_prompt('dark_factory') does NOT include the guardrail section.

    This prevents the dark_factory misfire: when any non-autopilot project runs Stage 2,
    the shared prompt must not contain the autopilot-specific halt instruction.
    """

    def test_guardrail_renders_for_autopilot_video_project(self):
        """build_stage2_system_prompt('autopilot_video') must include a
        content-based Contamination Guardrail section with no numeric task-ID
        ceiling, positioned before ## Available Tools."""
        import re

        from fused_memory.reconciliation.prompts.stage2 import build_stage2_system_prompt
        prompt = build_stage2_system_prompt(project_id='autopilot_video')

        # (a) guardrail heading is present and precedes ## Available Tools
        assert 'Contamination Guardrail' in prompt, (
            "Expected 'Contamination Guardrail' in the autopilot_video prompt — "
            "the guardrail must be injected when project_id == 'autopilot_video'."
        )
        assert prompt.index('Contamination Guardrail') < prompt.index('## Available Tools'), (
            "Contamination Guardrail section must appear BEFORE ## Available Tools "
            "in the autopilot_video prompt so the LLM reads the gate before any "
            "tool-use guidance."
        )

        # (b) no numeric task-ID ceiling in the guardrail block
        guardrail_block = _extract_section(prompt, 'Contamination Guardrail')
        assert '606' not in guardrail_block, (
            "The guardrail block must not contain the legacy numeric ceiling '606' — "
            "high task IDs are normal project growth, not contamination."
        )
        assert not re.search(r'exceeds\s+\d+', guardrail_block), (
            "The guardrail block must not use 'exceeds <number>' phrasing — "
            "cross-project contamination is content-based, not ID-magnitude-based."
        )
        assert 'task ceiling' not in guardrail_block.lower(), (
            "The guardrail block must not reference a 'task ceiling' — "
            "the ceiling concept has been removed in favour of content-based detection."
        )

        # (c) content-based phrasing is present — require specific multi-word phrases
        # to avoid false-positive matches on incidental single words like 'path'.
        guardrail_lower = guardrail_block.lower()
        assert any(phrase in guardrail_lower for phrase in ('file path', 'cross-project routing')), (
            "Expected specific content-based phrasing ('file path' or 'Cross-Project Routing') "
            "in the guardrail block — contamination must be judged by cited file paths/modules, "
            "not task-ID magnitude."
        )

    def test_guardrail_omitted_for_other_projects(self):
        """build_stage2_system_prompt('dark_factory') must NOT include the
        Contamination Guardrail section — the guardrail is autopilot_video-specific."""
        from fused_memory.reconciliation.prompts.stage2 import build_stage2_system_prompt
        prompt = build_stage2_system_prompt(project_id='dark_factory')
        assert 'Contamination Guardrail' not in prompt, (
            "Expected 'Contamination Guardrail' to be absent from the dark_factory "
            "prompt — the guardrail must NOT fire for non-autopilot projects."
        )


class TestStage2NoTaskIdCeiling:
    """Task IDs above the old 606 ceiling must NOT block task writes.

    Regression guard for the bug described in task-1517: legitimate
    autopilot_video tasks 607-610 (created 2026-05-26) caused EVERY Stage 2
    cycle to abort because the contamination code gate fired on task IDs
    exceeding the hardcoded AUTOPILOT_VIDEO_TASK_CEILING=606.

    After the fix (step-4), get_disallowed_tools() must always return
    STAGE2_DISALLOWED regardless of task ID magnitude.
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

    @pytest.mark.asyncio
    async def test_high_task_ids_do_not_block_task_writes(self, mock_deps):
        """Task IDs 607-610 (above the old 606 ceiling) must not cause
        get_disallowed_tools() to append DISALLOW_TASK_WRITES.

        Failure mode (pre-fix): _contamination_detected=True was set during
        assemble_payload() when excessive_autopilot_video_ids() found IDs > 606,
        causing get_disallowed_tools() to return STAGE2_DISALLOWED + DISALLOW_TASK_WRITES
        and aborting every Stage 2 cycle silently.
        """
        watermark = Watermark(project_id='autopilot_video')
        stage = make_configured_task_knowledge_sync_stage(
            mock_deps,
            project_id='autopilot_video',
            project_root='/home/leo/src/autopilot-video',
        )
        # Synthetic tree with task IDs that exceed the old 606 ceiling
        mock_deps['taskmaster'].get_tasks.return_value = {
            'tasks': [
                {'id': 607, 'title': 'Task 607', 'status': 'pending', 'dependencies': []},
                {'id': 608, 'title': 'Task 608', 'status': 'in-progress', 'dependencies': []},
                {'id': 609, 'title': 'Task 609', 'status': 'done', 'dependencies': []},
                {'id': 610, 'title': 'Task 610', 'status': 'cancelled', 'dependencies': []},
            ]
        }

        payload = await stage.assemble_payload([], watermark, [])

        # assemble_payload must complete without aborting and include the
        # high-ID tasks in the rendered output.  Tasks 607 and 608 are
        # active (pending / in-progress) so they appear in the active task
        # tree section with the format '- [607] (pending) Task 607 deps=[]'.
        # If a numeric-ID gate were re-introduced and fired early, these
        # tasks would be absent from the payload.
        assert '[607]' in payload, (
            "Task 607 must appear in the rendered payload — "
            "assemble_payload must not abort on task IDs above the old 606 ceiling."
        )
        assert '[608]' in payload, (
            "Task 608 must appear in the rendered payload — "
            "assemble_payload must not short-circuit on high task IDs."
        )

        assert stage.get_disallowed_tools() == STAGE2_DISALLOWED, (
            f'Expected get_disallowed_tools() == STAGE2_DISALLOWED but got '
            f'{stage.get_disallowed_tools()!r} — high task IDs must not append '
            'DISALLOW_TASK_WRITES; the numeric-ID contamination gate has been removed.'
        )


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
        """Stage 1 guideline does not include task write tools (Stage 1 is memory-only)."""
        from fused_memory.reconciliation.prompts import _STAGE1_PROJECT_ID_GUIDELINE
        assert 'get_tasks' not in _STAGE1_PROJECT_ID_GUIDELINE
        assert 'set_task_status' not in _STAGE1_PROJECT_ID_GUIDELINE
        assert 'add_task' not in _STAGE1_PROJECT_ID_GUIDELINE
        assert 'submit_task' not in _STAGE1_PROJECT_ID_GUIDELINE
        assert 'resolve_ticket' not in _STAGE1_PROJECT_ID_GUIDELINE

    def test_stage3_does_not_include_write_tools(self):
        """Stage 3 guideline does not include write tools (Stage 3 is read-only)."""
        from fused_memory.reconciliation.prompts import _STAGE3_PROJECT_ID_GUIDELINE
        assert 'add_memory' not in _STAGE3_PROJECT_ID_GUIDELINE
        assert 'delete_memory' not in _STAGE3_PROJECT_ID_GUIDELINE
        assert 'set_task_status' not in _STAGE3_PROJECT_ID_GUIDELINE
        assert 'add_task' not in _STAGE3_PROJECT_ID_GUIDELINE
        assert 'submit_task' not in _STAGE3_PROJECT_ID_GUIDELINE
        assert 'resolve_ticket' not in _STAGE3_PROJECT_ID_GUIDELINE



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
                ['submit_task', 'resolve_ticket', 'set_task_status'],  # Stage 2 has full MCP access
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
            stage._current_run_id = 'test-run'
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
        stage = make_configured_task_knowledge_sync_stage(mock_deps_for_stage, project_id='test_proj', project_root='/home/leo/src/test_proj')
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
        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='test_project', project_root='/tmp/test_project')
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
        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='test_project', project_root='/tmp/test_project')
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

        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='test_project', project_root='/tmp/test_project')
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
        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='test_project', project_root='/tmp/test_project')
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
        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='test_project', project_root='/tmp/test_project')
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
        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='test_project', project_root='/tmp/test_project')
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
        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='test_project', project_root='/tmp/test_project')
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
        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='test_project', project_root='/tmp/test_project')
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
        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='test_project', project_root='/tmp/test_project')
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
        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='test_project', project_root='/tmp/test_project')
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
        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='test_project', project_root='/tmp/test_project')
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
        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='test_project', project_root='/tmp/test_project')
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
        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='test_project', project_root='/tmp/test_project')
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
        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='test_project', project_root='/tmp/test_project')
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
        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='test_project', project_root='/tmp/test_project')
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
# _RECONCILIATION_STAGE_CAP_WAIT_SANITY_SECS: canonical minutes-scale override
# forwarded to invoke_with_cap_retry (task 1401); single value pin lives here.
# ---------------------------------------------------------------------------


class TestCliStageRunnerCapWaitSanityBound:
    """_RECONCILIATION_STAGE_CAP_WAIT_SANITY_SECS is the single canonical constant
    imported by all three stage runners and forwarded to invoke_with_cap_retry;
    prevents stalling the reconciliation queue under cap."""

    def test_constant_importable_and_minutes_scale(self):
        """(a) _RECONCILIATION_STAGE_CAP_WAIT_SANITY_SECS is importable from
        fused_memory.reconciliation and pinned to the documented 30-min policy."""
        from fused_memory.reconciliation import (
            _RECONCILIATION_STAGE_CAP_WAIT_SANITY_SECS,
        )

        assert _RECONCILIATION_STAGE_CAP_WAIT_SANITY_SECS == 1800.0

    @pytest.mark.asyncio
    async def test_run_stage_via_cli_forwards_cap_wait_sanity_secs(self, tmp_path):
        """(b) run_stage_via_cli forwards cap_wait_sanity_secs=_RECONCILIATION_STAGE_CAP_WAIT_SANITY_SECS."""
        from fused_memory.reconciliation import (
            _RECONCILIATION_STAGE_CAP_WAIT_SANITY_SECS,
        )

        config = ReconciliationConfig(
            enabled=True,
            explore_codebase_root=str(tmp_path),
            agent_llm_model='sonnet',
            agent_max_steps=5,
            stage_timeout_seconds=600,
        )
        empty_result = AgentResult(success=True, output='')
        mock = AsyncMock(return_value=empty_result)
        with patch(
            'fused_memory.reconciliation.cli_stage_runner.invoke_with_cap_retry',
            new=mock,
        ):
            await run_stage_via_cli(
                system_prompt='x',
                payload='y',
                disallowed_tools=[],
                config=config,
                mcp_config={'mcpServers': {}},
            )
        assert mock.call_args.kwargs['cap_wait_sanity_secs'] == _RECONCILIATION_STAGE_CAP_WAIT_SANITY_SECS


# ---------------------------------------------------------------------------
# _format_flagged: no-silent-truncation tests (step-1)
# ---------------------------------------------------------------------------


class TestFormatFlaggedNoSilentTruncation:
    """_format_flagged renders all items — no silent [:50] truncation."""

    def test_all_100_items_present_in_text(self):
        """100 items must all appear in the rendered text — no [:50] cap."""
        items = [{'description': f'item-{i}', 'severity': 'minor'} for i in range(100)]
        text = _format_flagged(items)
        for i in range(100):
            assert f'item-{i}' in text, (
                f'Expected item-{i} description in rendered text; got:\n{text[:500]}...'
            )

    def test_no_truncation_footer_when_all_items_rendered(self):
        """When all 100 items render, there must be no '... and N more' footer line."""
        items = [{'description': f'item-{i}', 'severity': 'minor'} for i in range(100)]
        text = _format_flagged(items)
        assert '... and ' not in text, (
            f'Unexpected truncation footer found in rendered text; got:\n{text[:500]}...'
        )

    def test_empty_list_returns_no_flagged_items(self):
        """Empty list must return the sentinel string."""
        text = _format_flagged([])
        assert text == 'No flagged items.'


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
        text = _format_flagged(items)
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
        text = _format_flagged(items)
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
        # All five structured-extra keys must be present
        for key in ('total', 'rendered', 'dropped', 'budget_chars', 'first_item_fragmented'):
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
        assert budget_chars == _FLAGGED_ITEMS_CHAR_BUDGET, (
            f'budget_chars must be {_FLAGGED_ITEMS_CHAR_BUDGET}, got {budget_chars}'
        )
        # 200×~330-char items — none of them is the oversized-first-item case
        assert rec.first_item_fragmented is False, (
            f'first_item_fragmented must be False for multi-item over-budget; '
            f'got {rec.first_item_fragmented!r}'
        )


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
            text = _format_flagged(items)

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
            text = _format_flagged(items)

        assert '… [item truncated]' in text, (
            'Expected truncation marker for oversized first item'
        )
        # The second item was not rendered; the footer should note it
        assert '... and 1 more (truncated: char budget)' in text, (
            f'Expected "... and 1 more" footer; got last 200 chars: {text[-200:]!r}'
        )

    def test_first_item_exceeds_budget_warning_has_rendered_zero_and_fragmented_true(
        self, caplog
    ):
        """When the first item alone exceeds the budget, rendered==0 and first_item_fragmented==True."""
        items = [{'description': 'y' * 50_000, 'severity': 'critical'}]
        with caplog.at_level(
            logging.WARNING,
            logger='fused_memory.reconciliation.stages.task_knowledge_sync',
        ):
            _format_flagged(items)

        warning_records = [
            rec for rec in caplog.records
            if rec.levelno == logging.WARNING
            and rec.name == 'fused_memory.reconciliation.stages.task_knowledge_sync'
            and rec.message == 'reconciliation.flagged_items_truncated'
        ]
        assert len(warning_records) == 1, (
            f'Expected exactly 1 truncation WARNING; got {len(warning_records)}'
        )
        rec = warning_records[0]
        assert rec.rendered == 0, (
            f'rendered must be 0 when first item exceeds budget (fragment != full); '
            f'got {rec.rendered}'
        )
        assert rec.first_item_fragmented is True, (
            f'first_item_fragmented must be True; got {rec.first_item_fragmented!r}'
        )
        assert rec.total == 1, f'total must be 1, got {rec.total}'
        assert rec.dropped == 1, f'dropped must be 1, got {rec.dropped}'
        assert rec.total == rec.rendered + rec.dropped, (
            f'total={rec.total} must equal rendered={rec.rendered} + dropped={rec.dropped}'
        )

    def test_two_item_first_exceeds_budget_warning_and_footer(self, caplog):
        """Two items, first oversized: rendered==0, first_item_fragmented==True, dropped==2, footer shows 1 more."""
        items = [
            {'description': 'z' * 50_000, 'severity': 'critical'},
            {'description': 'small', 'severity': 'minor'},
        ]
        with caplog.at_level(
            logging.WARNING,
            logger='fused_memory.reconciliation.stages.task_knowledge_sync',
        ):
            text = _format_flagged(items)

        # Footer shows only completely_missing items (not the fragmented first item)
        assert '... and 1 more (truncated: char budget)' in text, (
            f'Expected "... and 1 more" footer; last 200 chars: {text[-200:]!r}'
        )

        warning_records = [
            rec for rec in caplog.records
            if rec.levelno == logging.WARNING
            and rec.name == 'fused_memory.reconciliation.stages.task_knowledge_sync'
            and rec.message == 'reconciliation.flagged_items_truncated'
        ]
        assert len(warning_records) == 1
        rec = warning_records[0]
        assert rec.rendered == 0, f'rendered must be 0; got {rec.rendered}'
        assert rec.first_item_fragmented is True, (
            f'first_item_fragmented must be True; got {rec.first_item_fragmented!r}'
        )
        assert rec.dropped == 2, f'dropped must be 2 (fragmented+missing); got {rec.dropped}'
        assert rec.total == rec.rendered + rec.dropped, (
            f'total={rec.total} must equal rendered={rec.rendered} + dropped={rec.dropped}'
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

        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='reify', project_root='/home/leo/src/reify')
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

        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='reify', project_root='/home/leo/src/reify')
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


class TestBriefingKnownGapsRefresh:
    """Tests for _run_briefing_known_gaps_script and _queue_briefing_refresh_tasks helpers."""

    @pytest.fixture
    def mock_deps(self, tmp_path):
        config = ReconciliationConfig(enabled=True, explore_codebase_root=str(tmp_path))
        return {
            'memory_service': AsyncMock(),
            'taskmaster': AsyncMock(),
            'journal': AsyncMock(),
            'config': config,
        }

    @pytest.fixture
    def watermark(self):
        return Watermark(project_id='reify')

    # ------------------------------------------------------------------ #
    # _run_briefing_known_gaps_script                                       #
    # ------------------------------------------------------------------ #

    @pytest.mark.asyncio
    async def test_run_script_returns_none_when_script_missing(self, tmp_path):
        """No scripts/refresh_briefing_known_gaps.py → returns None without subprocess."""
        # tmp_path has neither the script nor the briefing file
        with patch('asyncio.create_subprocess_exec') as mock_subproc:
            result = await _run_briefing_known_gaps_script(str(tmp_path))

        assert result is None
        mock_subproc.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_script_returns_none_when_briefing_missing(self, tmp_path):
        """Script present but review/briefing.yaml absent → returns None without subprocess."""
        # Create the script file but NOT the briefing
        script_dir = tmp_path / 'scripts'
        script_dir.mkdir()
        (script_dir / 'refresh_briefing_known_gaps.py').touch()
        # review/briefing.yaml intentionally absent

        with patch('asyncio.create_subprocess_exec') as mock_subproc:
            result = await _run_briefing_known_gaps_script(str(tmp_path))

        assert result is None
        mock_subproc.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_script_parses_json_mismatches_on_success(self, tmp_path):
        """Both artifacts present; subprocess returns exit 1 with JSON mismatches."""
        # Create both required artifacts
        script_dir = tmp_path / 'scripts'
        script_dir.mkdir()
        (script_dir / 'refresh_briefing_known_gaps.py').touch()
        review_dir = tmp_path / 'review'
        review_dir.mkdir()
        (review_dir / 'briefing.yaml').touch()

        mismatch_data = [
            {'task_id': '1751', 'title': 'Foo', 'subproject': 'bar', 'what': 'gap'}
        ]
        stdout_bytes = json.dumps(mismatch_data).encode()

        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(stdout_bytes, b''))

        with patch('asyncio.create_subprocess_exec', return_value=mock_proc) as mock_exec:
            result = await _run_briefing_known_gaps_script(str(tmp_path))

        assert result == mismatch_data

        # Verify the subprocess was called with the expected flags
        call_args = mock_exec.call_args
        pos_args = call_args[0]
        assert '--briefing' in pos_args
        assert '--tasks' in pos_args
        assert '--json' in pos_args
        # The briefing path must point to <project_root>/review/briefing.yaml
        briefing_idx = pos_args.index('--briefing')
        assert pos_args[briefing_idx + 1] == str(tmp_path / 'review' / 'briefing.yaml')
        # The tasks path must point to <project_root>/.taskmaster/tasks/tasks.json
        tasks_idx = pos_args.index('--tasks')
        assert pos_args[tasks_idx + 1] == str(tmp_path / '.taskmaster' / 'tasks' / 'tasks.json')

    @pytest.mark.asyncio
    async def test_run_script_returns_none_on_subprocess_error(self, tmp_path, caplog):
        """Exit code 2 → returns None and emits a WARNING naming the exit code."""
        script_dir = tmp_path / 'scripts'
        script_dir.mkdir()
        (script_dir / 'refresh_briefing_known_gaps.py').touch()
        review_dir = tmp_path / 'review'
        review_dir.mkdir()
        (review_dir / 'briefing.yaml').touch()

        mock_proc = MagicMock()
        mock_proc.returncode = 2
        mock_proc.communicate = AsyncMock(
            return_value=(b'', b'ERROR: cannot parse briefing.yaml: bad syntax')
        )

        with patch('asyncio.create_subprocess_exec', return_value=mock_proc), caplog.at_level(
            logging.WARNING,
            logger='fused_memory.reconciliation.stages.task_knowledge_sync',
        ):
            result = await _run_briefing_known_gaps_script(str(tmp_path))

        assert result is None
        warning_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and r.name == 'fused_memory.reconciliation.stages.task_knowledge_sync'
        ]
        assert len(warning_records) == 1
        # The warning must use the canonical message key and carry the exit code in extra.
        assert 'briefing_known_gaps_script_failed' in warning_records[0].getMessage()
        assert getattr(warning_records[0], 'returncode', None) == 2

    # ------------------------------------------------------------------ #
    # _queue_briefing_refresh_tasks                                         #
    # ------------------------------------------------------------------ #

    @pytest.mark.asyncio
    async def test_queue_refresh_tasks_calls_add_task_for_each_mismatch(self):
        """No existing tasks → add_task called once per mismatch with canonical title."""
        taskmaster = AsyncMock()
        taskmaster.get_tasks.return_value = {'tasks': []}
        taskmaster.add_task.return_value = {'id': '9001', 'message': 'created'}

        mismatches = [
            {'task_id': '1751', 'title': 'Old gap title', 'subproject': 'fused-memory', 'what': 'Description of the gap'},
            {'task_id': '1820', 'title': 'Other', 'subproject': 'orchestrator', 'what': 'Other gap'},
        ]

        await _queue_briefing_refresh_tasks(taskmaster, '/tmp/p', mismatches)

        assert taskmaster.add_task.call_count == 2

        first_call_kwargs = taskmaster.add_task.call_args_list[0][1]
        assert first_call_kwargs['title'] == 'Refresh briefing: remove task 1751 from known_gaps'
        assert 'fused-memory' in first_call_kwargs['description']
        assert 'Old gap title' in first_call_kwargs['description']

        second_call_kwargs = taskmaster.add_task.call_args_list[1][1]
        assert second_call_kwargs['title'] == 'Refresh briefing: remove task 1820 from known_gaps'

    @pytest.mark.asyncio
    async def test_queue_refresh_tasks_skips_existing_pending_with_same_title(self):
        """Existing pending task with canonical title → add_task not called; skipped list populated."""
        taskmaster = AsyncMock()
        taskmaster.get_tasks.return_value = {
            'tasks': [{
                'id': 999,
                'status': 'pending',
                'title': 'Refresh briefing: remove task 1751 from known_gaps',
            }]
        }

        mismatches = [{'task_id': '1751', 'title': 'Foo', 'subproject': 'bar', 'what': 'gap'}]

        result = await _queue_briefing_refresh_tasks(taskmaster, '/tmp/p', mismatches)

        taskmaster.add_task.assert_not_called()
        assert '1751' in result['skipped']

    @pytest.mark.asyncio
    async def test_queue_refresh_tasks_creates_when_existing_with_same_title_is_done(self):
        """Done task with same canonical title → add_task IS called (regression can re-file)."""
        taskmaster = AsyncMock()
        taskmaster.get_tasks.return_value = {
            'tasks': [{
                'id': 999,
                'status': 'done',
                'title': 'Refresh briefing: remove task 1751 from known_gaps',
            }]
        }
        taskmaster.add_task.return_value = {'id': '1000', 'message': 'created'}

        mismatches = [{'task_id': '1751', 'title': 'Foo', 'subproject': 'bar', 'what': 'gap'}]

        result = await _queue_briefing_refresh_tasks(taskmaster, '/tmp/p', mismatches)

        taskmaster.add_task.assert_called_once()
        assert '1751' not in result['skipped']

    @pytest.mark.asyncio
    async def test_queue_refresh_tasks_dedup_is_exact_title_match(self):
        """Dedup uses case-sensitive exact string equality — near-misses do NOT dedup.

        A pending task whose title differs from the canonical title by trailing
        whitespace or casing is NOT treated as a duplicate.  This documents the
        intended behavior: if we ever want tolerance, we should normalise before
        comparing and add a test for that normalisation.
        """
        taskmaster = AsyncMock()
        canonical = 'Refresh briefing: remove task 1751 from known_gaps'
        taskmaster.get_tasks.return_value = {
            'tasks': [
                # trailing space — should NOT dedup
                {'id': 900, 'status': 'pending', 'title': canonical + ' '},
                # uppercase — should NOT dedup
                {'id': 901, 'status': 'pending', 'title': canonical.upper()},
            ]
        }
        taskmaster.add_task.return_value = {'id': '9999', 'message': 'created'}

        mismatches = [{'task_id': '1751', 'title': 'Foo', 'subproject': 'bar', 'what': 'gap'}]

        result = await _queue_briefing_refresh_tasks(taskmaster, '/tmp/p', mismatches)

        # Neither near-miss deduped → add_task must be called exactly once
        taskmaster.add_task.assert_called_once()
        assert '1751' not in result['skipped']

    # ------------------------------------------------------------------ #
    # TaskKnowledgeSync.run() integration                                   #
    # ------------------------------------------------------------------ #

    @pytest.mark.asyncio
    async def test_run_invokes_briefing_refresh_hook_before_super_run(self, mock_deps, tmp_path):
        """run() calls _maybe_queue_briefing_refresh_tasks then super().run()."""
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'reify'
        stage.project_root = str(tmp_path)

        mismatch = {'task_id': '1751', 'title': 'Foo', 'subproject': 'bar', 'what': 'gap'}
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}
        mock_deps['taskmaster'].add_task.return_value = {'id': '9001', 'message': 'created'}

        with (
            patch(
                'fused_memory.reconciliation.stages.task_knowledge_sync._run_briefing_known_gaps_script',
                new=AsyncMock(return_value=[mismatch]),
            ),
            patch.object(stage, 'assemble_payload', new=AsyncMock(return_value='payload')),
            patch(
                'fused_memory.reconciliation.stages.base.run_stage_via_cli',
                new=AsyncMock(return_value=MagicMock(
                    success=True, report={'summary': 'ok'},
                )),
            ),
        ):
            report = await stage.run(
                events=[],
                watermark=Watermark(project_id='reify'),
                prior_reports=[],
                run_id='test-run-1086',
            )

        # The hook should have called add_task once
        mock_deps['taskmaster'].add_task.assert_called_once()
        call_kwargs = mock_deps['taskmaster'].add_task.call_args[1]
        assert call_kwargs['title'] == 'Refresh briefing: remove task 1751 from known_gaps'

        # And the run should have completed with a StageReport
        assert report is not None

    @pytest.mark.asyncio
    async def test_run_success_log_uses_non_reserved_logrecord_keys(
        self, mock_deps, tmp_path, caplog,
    ):
        """Success-path INFO log must not collide with reserved LogRecord attrs.

        Regression: 'created' is a reserved LogRecord attribute (timestamp);
        passing it via extra= raises KeyError inside logging.makeRecord. Prior
        to the fix, the success path was wrapped in a broad try/except that
        masked the KeyError and emitted a misleading 'briefing_refresh_hook_failed'
        WARNING even when tasks had been queued successfully. We assert here
        that the success path emits the INFO record cleanly and that no
        'briefing_refresh_hook_failed' WARNING is produced.
        """
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'reify'
        stage.project_root = str(tmp_path)

        mismatch = {'task_id': '1751', 'title': 'Foo', 'subproject': 'bar', 'what': 'gap'}
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}
        mock_deps['taskmaster'].add_task.return_value = {'id': '9001', 'message': 'created'}

        with (
            patch(
                'fused_memory.reconciliation.stages.task_knowledge_sync._run_briefing_known_gaps_script',
                new=AsyncMock(return_value=[mismatch]),
            ),
            patch.object(stage, 'assemble_payload', new=AsyncMock(return_value='payload')),
            patch(
                'fused_memory.reconciliation.stages.base.run_stage_via_cli',
                new=AsyncMock(return_value=MagicMock(
                    success=True, report={'summary': 'ok'},
                )),
            ),
            caplog.at_level(
                logging.INFO,
                logger='fused_memory.reconciliation.stages.task_knowledge_sync',
            ),
        ):
            await stage.run(
                events=[],
                watermark=Watermark(project_id='reify'),
                prior_reports=[],
                run_id='test-run-1086-log',
            )

        target_logger = 'fused_memory.reconciliation.stages.task_knowledge_sync'
        info_records = [
            r for r in caplog.records
            if r.name == target_logger
            and r.levelno == logging.INFO
            and 'briefing_refresh_tasks_queued' in r.getMessage()
        ]
        assert len(info_records) == 1, (
            'expected exactly one briefing_refresh_tasks_queued INFO record'
        )
        rec = info_records[0]
        # The renamed extras must be present on the LogRecord.
        assert getattr(rec, 'created_ids', None) == ['9001']
        assert getattr(rec, 'skipped_ids', None) == []

        # And no false-positive failure WARNING should have fired.
        failure_warnings = [
            r for r in caplog.records
            if r.name == target_logger
            and r.levelno == logging.WARNING
            and 'briefing_refresh_hook_failed' in r.getMessage()
        ]
        assert failure_warnings == [], (
            'success path must not emit briefing_refresh_hook_failed'
        )

    @pytest.mark.asyncio
    async def test_run_dedupes_when_invoked_twice_with_same_mismatch(self, mock_deps, tmp_path):
        """Calling run() twice: second call skips add_task because first created pending task."""
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'reify'
        stage.project_root = str(tmp_path)

        mismatch = {'task_id': '1751', 'title': 'Foo', 'subproject': 'bar', 'what': 'gap'}
        canonical_title = 'Refresh briefing: remove task 1751 from known_gaps'

        # First call: no existing tasks
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}
        mock_deps['taskmaster'].add_task.return_value = {'id': '9001', 'message': 'created'}

        fake_cli_result = MagicMock(success=True, report={'summary': 'ok'})

        with (
            patch(
                'fused_memory.reconciliation.stages.task_knowledge_sync._run_briefing_known_gaps_script',
                new=AsyncMock(return_value=[mismatch]),
            ),
            patch.object(stage, 'assemble_payload', new=AsyncMock(return_value='payload')),
            patch(
                'fused_memory.reconciliation.stages.base.run_stage_via_cli',
                new=AsyncMock(return_value=fake_cli_result),
            ),
        ):
            await stage.run(
                events=[],
                watermark=Watermark(project_id='reify'),
                prior_reports=[],
                run_id='test-run-1',
            )

            # Now update get_tasks to include the just-created task as pending
            mock_deps['taskmaster'].get_tasks.return_value = {
                'tasks': [{'id': '9001', 'status': 'pending', 'title': canonical_title}]
            }
            mock_deps['taskmaster'].add_task.reset_mock()

            await stage.run(
                events=[],
                watermark=Watermark(project_id='reify'),
                prior_reports=[],
                run_id='test-run-2',
            )

        # Second invocation must not call add_task again
        mock_deps['taskmaster'].add_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_swallows_helper_failure(self, mock_deps, tmp_path, caplog):
        """If _run_briefing_known_gaps_script raises, run() still completes and logs WARNING."""
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'reify'
        stage.project_root = str(tmp_path)

        fake_cli_result = MagicMock(success=True, report={'summary': 'ok'})

        with (
            patch(
                'fused_memory.reconciliation.stages.task_knowledge_sync._run_briefing_known_gaps_script',
                new=AsyncMock(side_effect=RuntimeError('boom')),
            ),
            patch.object(stage, 'assemble_payload', new=AsyncMock(return_value='payload')),
            patch(
                'fused_memory.reconciliation.stages.base.run_stage_via_cli',
                new=AsyncMock(return_value=fake_cli_result),
            ),
            caplog.at_level(
                logging.WARNING,
                logger='fused_memory.reconciliation.stages.task_knowledge_sync',
            ),
        ):
            report = await stage.run(
                events=[],
                watermark=Watermark(project_id='reify'),
                prior_reports=[],
                run_id='test-run-swallow',
            )

        assert report is not None

        warning_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and r.name == 'fused_memory.reconciliation.stages.task_knowledge_sync'
            and 'briefing_refresh_hook_failed' in r.getMessage()
        ]
        assert len(warning_records) == 1

    # ------------------------------------------------------------------ #
    # _run_briefing_known_gaps_script — timeout path                        #
    # ------------------------------------------------------------------ #

    @pytest.mark.asyncio
    async def test_run_script_returns_none_on_timeout(self, tmp_path, caplog):
        """asyncio.wait_for raises TimeoutError → proc.kill() called, returns None, WARN emitted
        (timeout flows through the real wait_for call site, not from communicate())."""
        script_dir = tmp_path / 'scripts'
        script_dir.mkdir()
        (script_dir / 'refresh_briefing_known_gaps.py').touch()
        review_dir = tmp_path / 'review'
        review_dir.mkdir()
        (review_dir / 'briefing.yaml').touch()

        mock_proc = MagicMock()
        mock_proc.communicate = AsyncMock(return_value=(b'', b''))

        with patch(
            'fused_memory.reconciliation.stages.task_knowledge_sync.asyncio.create_subprocess_exec',
            return_value=mock_proc,
        ), patch(
            'fused_memory.reconciliation.stages.task_knowledge_sync.asyncio.wait_for',
            side_effect=TimeoutError,
        ), caplog.at_level(
            logging.WARNING,
            logger='fused_memory.reconciliation.stages.task_knowledge_sync',
        ):
            result = await _run_briefing_known_gaps_script(str(tmp_path))

        assert result is None
        mock_proc.kill.assert_called_once()
        warning_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and r.name == 'fused_memory.reconciliation.stages.task_knowledge_sync'
        ]
        assert len(warning_records) == 1
        assert 'briefing_known_gaps_script_timeout' in warning_records[0].getMessage()

    # ------------------------------------------------------------------ #
    # _run_briefing_known_gaps_script — bad JSON path                       #
    # ------------------------------------------------------------------ #

    @pytest.mark.asyncio
    async def test_run_script_returns_none_on_bad_json(self, tmp_path, caplog):
        """Subprocess returns non-JSON stdout → returns None, WARN carries error key."""
        script_dir = tmp_path / 'scripts'
        script_dir.mkdir()
        (script_dir / 'refresh_briefing_known_gaps.py').touch()
        review_dir = tmp_path / 'review'
        review_dir.mkdir()
        (review_dir / 'briefing.yaml').touch()

        mock_proc = MagicMock()
        mock_proc.returncode = 1  # "mismatches found" exit code
        mock_proc.communicate = AsyncMock(return_value=(b'not valid json {', b''))

        with patch(
            'fused_memory.reconciliation.stages.task_knowledge_sync.asyncio.create_subprocess_exec',
            return_value=mock_proc,
        ), caplog.at_level(
            logging.WARNING,
            logger='fused_memory.reconciliation.stages.task_knowledge_sync',
        ):
            result = await _run_briefing_known_gaps_script(str(tmp_path))

        assert result is None
        warning_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and r.name == 'fused_memory.reconciliation.stages.task_knowledge_sync'
        ]
        assert len(warning_records) == 1
        assert 'briefing_known_gaps_script_bad_json' in warning_records[0].getMessage()
        # The WARN's extra must carry an 'error' key with a string from JSONDecodeError
        assert isinstance(getattr(warning_records[0], 'error', None), str)

    # ------------------------------------------------------------------ #
    # _queue_briefing_refresh_tasks — exception path                        #
    # ------------------------------------------------------------------ #

    @pytest.mark.asyncio
    async def test_queue_refresh_tasks_appends_failed_when_add_task_raises(self, caplog):
        """add_task raises → task_id in failed (not created), WARN emitted."""
        taskmaster = AsyncMock()
        taskmaster.get_tasks.return_value = {'tasks': []}
        taskmaster.add_task.side_effect = RuntimeError('mcp transport dead')

        mismatches = [{'task_id': '1751', 'title': 'Foo', 'subproject': 'bar', 'what': 'gap'}]

        with caplog.at_level(
            logging.WARNING,
            logger='fused_memory.reconciliation.stages.task_knowledge_sync',
        ):
            result = await _queue_briefing_refresh_tasks(taskmaster, '/tmp/p', mismatches)

        assert result['failed'] == ['1751']
        assert result['created'] == []
        assert result['skipped'] == []
        warning_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and r.name == 'fused_memory.reconciliation.stages.task_knowledge_sync'
        ]
        assert len(warning_records) == 1
        assert 'briefing_refresh_add_task_failed' in warning_records[0].getMessage()
        assert getattr(warning_records[0], 'task_id', None) == '1751'

    # ------------------------------------------------------------------ #
    # _queue_briefing_refresh_tasks — unexpected-shape (contract-drift)     #
    # ------------------------------------------------------------------ #

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'bogus_result',
        [
            None,
            {'message': 'no id field'},
            'string-not-dict',
            {'id': None},
            {'id': ''},
            {'id': 123},
            {'id': True},
            {'id': '   '},
            {'id': '  100  '},
        ],
        ids=[
            'none',
            'dict_without_id',
            'non_dict',
            'dict_id_none',
            'dict_id_empty',
            'int_not_str',
            'id_bool',
            'id_whitespace',
            'id_mixed_whitespace',
        ],
    )
    async def test_queue_refresh_tasks_treats_unexpected_shape_as_failure(
        self, bogus_result, caplog,
    ):
        """add_task returns wrong shape → task_id in failed (NOT created), WARN with new key."""
        taskmaster = AsyncMock()
        taskmaster.get_tasks.return_value = {'tasks': []}
        taskmaster.add_task.return_value = bogus_result

        mismatches = [{'task_id': '1751', 'title': 'Foo', 'subproject': 'bar', 'what': 'gap'}]

        with caplog.at_level(
            logging.WARNING,
            logger='fused_memory.reconciliation.stages.task_knowledge_sync',
        ):
            result = await _queue_briefing_refresh_tasks(taskmaster, '/tmp/p', mismatches)

        assert result['created'] == [], (
            f'created should be [] when add_task returns {bogus_result!r}, got {result["created"]!r}'
        )
        assert result['failed'] == ['1751'], (
            f'failed should be [\"1751\"] when add_task returns {bogus_result!r}'
        )
        assert result['skipped'] == []
        warning_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and r.name == 'fused_memory.reconciliation.stages.task_knowledge_sync'
        ]
        assert len(warning_records) == 1
        assert 'briefing_refresh_add_task_unexpected_shape' in warning_records[0].getMessage()
        assert getattr(warning_records[0], 'task_id', None) == '1751'

    # ------------------------------------------------------------------ #
    # _queue_briefing_refresh_tasks — mixed success/failure in one call    #
    # ------------------------------------------------------------------ #

    @pytest.mark.asyncio
    async def test_queue_refresh_tasks_partitions_created_and_failed_correctly_in_mixed_run(
        self, caplog,
    ):
        """Loop continues past a failed creation; created/failed are correctly partitioned.

        Pins post-confirmation partitioning for the multi-mismatch path:
        a failed add_task must NOT abort the loop or inflate the created list.
        """
        taskmaster = AsyncMock()
        taskmaster.get_tasks.return_value = {'tasks': []}
        taskmaster.add_task.side_effect = [
            {'id': '100', 'message': 'created'},   # success
            {'id': ''},                              # DTO-violating empty id → failed
            {'id': '300', 'message': 'created'},   # success
        ]

        mismatches = [
            {'task_id': '1751', 'title': 'Foo', 'subproject': 'bar', 'what': 'gap1'},
            {'task_id': '1820', 'title': 'Baz', 'subproject': 'bar', 'what': 'gap2'},
            {'task_id': '1900', 'title': 'Qux', 'subproject': 'bar', 'what': 'gap3'},
        ]

        with caplog.at_level(
            logging.WARNING,
            logger='fused_memory.reconciliation.stages.task_knowledge_sync',
        ):
            result = await _queue_briefing_refresh_tasks(taskmaster, '/tmp/p', mismatches)

        assert result['created'] == ['100', '300'], (
            f'expected created=[100, 300], got {result["created"]!r}'
        )
        assert result['failed'] == ['1820'], (
            f'expected failed=[1820], got {result["failed"]!r}'
        )
        assert result['skipped'] == []
        # All three mismatches must be attempted; a failure must not break the loop.
        assert taskmaster.add_task.call_count == 3
        # Exactly one warning must be emitted for the failed creation, pinning observability.
        warning_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and 'briefing_refresh_add_task_unexpected_shape' in r.getMessage()
        ]
        assert len(warning_records) == 1
        assert getattr(warning_records[0], 'task_id', None) == '1820'


# ---------------------------------------------------------------------------
# MemoryConsolidator.run() dedup hook tests (step-11)
# ---------------------------------------------------------------------------


class TestMemoryConsolidatorFlagDedup:
    """MemoryConsolidator.run() calls dedup_flags after super().run() for normal cycles."""

    @pytest.fixture
    def mock_deps(self):
        config = ReconciliationConfig(enabled=True, explore_codebase_root='/tmp/test')
        return {
            'memory_service': AsyncMock(),
            'taskmaster': AsyncMock(),
            'journal': AsyncMock(),
            'config': config,
        }

    @pytest.mark.asyncio
    async def test_normal_cycle_invokes_dedup_flags(self, mock_deps):
        """Normal (non-remediation) cycle calls dedup_flags with the report's flagged items."""
        stage = MemoryConsolidator(StageId.memory_consolidator, **mock_deps)
        stage.project_id = 'p'
        stage.episode_limit = 10
        stage.memory_limit = 10

        flagged = [
            {'task_id': '1', 'flag_type': 'missing_deliverable'},
            {'task_id': '2', 'flag_type': 'stale_metadata'},
        ]
        base_report = StageReport(
            stage=StageId.memory_consolidator,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items_flagged=list(flagged),
            stats={},
            llm_calls=1,
            tokens_used=100,
        )

        deduplicated = [dict(f) for f in flagged]
        dedup_mock = AsyncMock(return_value=deduplicated)

        with (
            patch.object(BaseStage, 'run', new=AsyncMock(return_value=base_report)),
            patch(
                'fused_memory.reconciliation.stages.memory_consolidator.dedup_flags',
                new=dedup_mock,
            ),
        ):
            report = await stage.run(
                events=[],
                watermark=Watermark(project_id='p'),
                prior_reports=[],
                run_id='r-uuid',
            )

        # dedup_flags must be awaited exactly once with the correct arguments
        dedup_mock.assert_awaited_once()
        call_args = dedup_mock.await_args
        assert call_args is not None
        assert call_args.kwargs.get('project_id') == 'p' or call_args.args[1] == 'p'
        assert call_args.kwargs.get('run_id') == 'r-uuid' or call_args.args[2] == 'r-uuid'
        flags_arg = call_args.kwargs.get('flags') or call_args.args[3]
        assert len(flags_arg) == 2

        assert report is not None

    @pytest.mark.asyncio
    async def test_remediation_run_skips_dedup_flags(self, mock_deps):
        """Remediation runs (remediation_findings set) must NOT call dedup_flags."""
        stage = MemoryConsolidator(StageId.memory_consolidator, **mock_deps)
        stage.project_id = 'p'
        stage.episode_limit = 10
        stage.memory_limit = 10
        stage.remediation_findings = [{'description': 'x'}]  # remediation mode

        base_report = StageReport(
            stage=StageId.memory_consolidator,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items_flagged=[{'task_id': '1', 'flag_type': 'foo'}],
            stats={},
            llm_calls=1,
            tokens_used=100,
        )

        dedup_mock = AsyncMock(return_value=[])

        with (
            patch.object(BaseStage, 'run', new=AsyncMock(return_value=base_report)),
            patch(
                'fused_memory.reconciliation.stages.memory_consolidator.dedup_flags',
                new=dedup_mock,
            ),
        ):
            report = await stage.run(
                events=[],
                watermark=Watermark(project_id='p'),
                prior_reports=[],
                run_id='r-uuid',
            )

        # dedup_flags must NOT have been called for remediation runs
        dedup_mock.assert_not_awaited()
        assert report is not None


# ---------------------------------------------------------------------------
# Task 1201 — MemoryConsolidator stale human-operator detector integration
# ---------------------------------------------------------------------------


class TestMemoryConsolidatorStaleOperatorDetector:
    """MemoryConsolidator.run() invokes stall-detector helpers after dedup_flags."""

    @pytest.fixture
    def mock_deps(self):
        config = ReconciliationConfig(enabled=True, explore_codebase_root='/tmp/test')
        return {
            'memory_service': AsyncMock(),
            'taskmaster': AsyncMock(),
            'journal': AsyncMock(),
            'config': config,
        }

    def _make_base_report(self, items_flagged=None):
        return StageReport(
            stage=StageId.memory_consolidator,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items_flagged=items_flagged or [],
            stats={},
            llm_calls=1,
            tokens_used=100,
        )

    @pytest.mark.asyncio
    async def test_hor_flag_at_threshold_escalated(self, mock_deps):
        """(a) HOR flag at threshold → track called, compute returns stalled, escalate called; stats recorded."""
        stage = MemoryConsolidator(StageId.memory_consolidator, **mock_deps)
        stage.project_id = 'p'
        stage.episode_limit = 10
        stage.memory_limit = 10
        fake_queue = MagicMock()
        stage._escalation_queue = fake_queue

        hor_flag = {'task_id': '1155', 'flag_type': 'assumption_invalid', 'resolution_status': 'human_operator_required'}
        base_report = self._make_base_report(items_flagged=[hor_flag])

        track_mock = AsyncMock(return_value={'1155': 7})
        compute_mock = MagicMock(return_value=['1155'])
        escalate_mock = AsyncMock(return_value=['1155'])
        dedup_mock = AsyncMock(return_value=[hor_flag])

        with (
            patch.object(BaseStage, 'run', new=AsyncMock(return_value=base_report)),
            patch('fused_memory.reconciliation.stages.memory_consolidator.dedup_flags', new=dedup_mock),
            patch('fused_memory.reconciliation.stages.memory_consolidator.track_human_operator_stalls', new=track_mock),
            patch('fused_memory.reconciliation.stages.memory_consolidator.compute_stalled_task_ids', new=compute_mock),
            patch('fused_memory.reconciliation.stages.memory_consolidator.maybe_escalate_stalled_tasks', new=escalate_mock),
        ):
            report = await stage.run(
                events=[],
                watermark=Watermark(project_id='p'),
                prior_reports=[],
                run_id='r-1201',
            )

        # track and compute and escalate all called
        track_mock.assert_awaited_once()
        compute_mock.assert_called_once_with({'1155': 7})
        escalate_mock.assert_awaited_once()
        # escalate called with the right queue
        escalate_call = escalate_mock.await_args
        assert escalate_call is not None
        assert escalate_call.kwargs.get('escalation_queue') is fake_queue or escalate_call.args[0] is fake_queue

        assert report.stats['stage1_human_operator_stalled'] == 1
        assert report.stats['stage1_human_operator_escalated'] == 1

    @pytest.mark.asyncio
    async def test_hor_flag_below_threshold_no_escalation(self, mock_deps):
        """(b) stall count below threshold → maybe_escalate NOT called; stalled=0, escalated=0."""
        stage = MemoryConsolidator(StageId.memory_consolidator, **mock_deps)
        stage.project_id = 'p'
        stage.episode_limit = 10
        stage.memory_limit = 10
        fake_queue = MagicMock()
        stage._escalation_queue = fake_queue

        hor_flag = {'task_id': '1155', 'resolution_status': 'human_operator_required'}
        base_report = self._make_base_report(items_flagged=[hor_flag])

        track_mock = AsyncMock(return_value={'1155': 3})
        compute_mock = MagicMock(return_value=[])   # empty → below threshold
        escalate_mock = AsyncMock(return_value=[])
        dedup_mock = AsyncMock(return_value=[hor_flag])

        with (
            patch.object(BaseStage, 'run', new=AsyncMock(return_value=base_report)),
            patch('fused_memory.reconciliation.stages.memory_consolidator.dedup_flags', new=dedup_mock),
            patch('fused_memory.reconciliation.stages.memory_consolidator.track_human_operator_stalls', new=track_mock),
            patch('fused_memory.reconciliation.stages.memory_consolidator.compute_stalled_task_ids', new=compute_mock),
            patch('fused_memory.reconciliation.stages.memory_consolidator.maybe_escalate_stalled_tasks', new=escalate_mock),
        ):
            report = await stage.run(
                events=[],
                watermark=Watermark(project_id='p'),
                prior_reports=[],
                run_id='r-1201-b',
            )

        track_mock.assert_awaited_once()
        escalate_mock.assert_not_awaited()
        assert report.stats['stage1_human_operator_stalled'] == 0
        assert report.stats['stage1_human_operator_escalated'] == 0

    @pytest.mark.asyncio
    async def test_no_hor_flags_zero_cost_path(self, mock_deps):
        """(c) no HOR flags → none of the new helpers are awaited."""
        stage = MemoryConsolidator(StageId.memory_consolidator, **mock_deps)
        stage.project_id = 'p'
        stage.episode_limit = 10
        stage.memory_limit = 10
        stage._escalation_queue = MagicMock()

        non_hor_flag = {'task_id': '99', 'flag_type': 'missing_deliverable', 'resolution_status': 'automated'}
        base_report = self._make_base_report(items_flagged=[non_hor_flag])

        track_mock = AsyncMock(return_value={})
        escalate_mock = AsyncMock(return_value=[])
        dedup_mock = AsyncMock(return_value=[non_hor_flag])

        with (
            patch.object(BaseStage, 'run', new=AsyncMock(return_value=base_report)),
            patch('fused_memory.reconciliation.stages.memory_consolidator.dedup_flags', new=dedup_mock),
            patch('fused_memory.reconciliation.stages.memory_consolidator.track_human_operator_stalls', new=track_mock),
            patch('fused_memory.reconciliation.stages.memory_consolidator.maybe_escalate_stalled_tasks', new=escalate_mock),
        ):
            report = await stage.run(
                events=[],
                watermark=Watermark(project_id='p'),
                prior_reports=[],
                run_id='r-1201-c',
            )

        track_mock.assert_not_awaited()
        escalate_mock.assert_not_awaited()
        assert report.stats.get('stage1_human_operator_stalled', 0) == 0
        assert report.stats.get('stage1_human_operator_escalated', 0) == 0

    @pytest.mark.asyncio
    async def test_no_escalation_queue_skips_all_stall_logic(self, mock_deps):
        """(d) _escalation_queue is None → entire stall-detector block skipped; no crash.

        Tracking is suppressed (not just escalation) to avoid accumulating Mem0
        markers that nothing will ever consume when escalation is unavailable.
        """
        stage = MemoryConsolidator(StageId.memory_consolidator, **mock_deps)
        stage.project_id = 'p'
        stage.episode_limit = 10
        stage.memory_limit = 10
        stage._escalation_queue = None  # explicit None

        hor_flag = {'task_id': '1155', 'resolution_status': 'human_operator_required'}
        base_report = self._make_base_report(items_flagged=[hor_flag])

        track_mock = AsyncMock(return_value={'1155': 6})
        escalate_mock = AsyncMock(return_value=[])
        dedup_mock = AsyncMock(return_value=[hor_flag])

        with (
            patch.object(BaseStage, 'run', new=AsyncMock(return_value=base_report)),
            patch('fused_memory.reconciliation.stages.memory_consolidator.dedup_flags', new=dedup_mock),
            patch('fused_memory.reconciliation.stages.memory_consolidator.track_human_operator_stalls', new=track_mock),
            patch('fused_memory.reconciliation.stages.memory_consolidator.maybe_escalate_stalled_tasks', new=escalate_mock),
        ):
            report = await stage.run(
                events=[],
                watermark=Watermark(project_id='p'),
                prior_reports=[],
                run_id='r-1201-d',
            )

        # when queue is None, track is NOT called (no Mem0 markers written)
        track_mock.assert_not_awaited()
        # escalate also NOT called
        escalate_mock.assert_not_awaited()
        # stats not set (no stall logic ran)
        assert report.stats.get('stage1_human_operator_stalled', 0) == 0
        assert report.stats.get('stage1_human_operator_escalated', 0) == 0

    @pytest.mark.asyncio
    async def test_remediation_mode_skips_all_stall_logic(self, mock_deps):
        """(e) remediation_findings set → ALL stale-operator logic skipped; no crash."""
        stage = MemoryConsolidator(StageId.memory_consolidator, **mock_deps)
        stage.project_id = 'p'
        stage.episode_limit = 10
        stage.memory_limit = 10
        stage.remediation_findings = [{'description': 'fix me'}]  # remediation mode

        hor_flag = {'task_id': '1155', 'resolution_status': 'human_operator_required'}
        base_report = self._make_base_report(items_flagged=[hor_flag])

        dedup_mock = AsyncMock(return_value=[hor_flag])
        track_mock = AsyncMock(return_value={})
        escalate_mock = AsyncMock(return_value=[])

        with (
            patch.object(BaseStage, 'run', new=AsyncMock(return_value=base_report)),
            patch('fused_memory.reconciliation.stages.memory_consolidator.dedup_flags', new=dedup_mock),
            patch('fused_memory.reconciliation.stages.memory_consolidator.track_human_operator_stalls', new=track_mock),
            patch('fused_memory.reconciliation.stages.memory_consolidator.maybe_escalate_stalled_tasks', new=escalate_mock),
        ):
            report = await stage.run(
                events=[],
                watermark=Watermark(project_id='p'),
                prior_reports=[],
                run_id='r-1201-e',
            )

        dedup_mock.assert_not_awaited()
        track_mock.assert_not_awaited()
        escalate_mock.assert_not_awaited()
        assert report is not None


# ---------------------------------------------------------------------------
# Task 1139 — FIX A / FIX D helpers
# ---------------------------------------------------------------------------

class TestShouldSkipKnownBug1139Flag:
    """_should_skip_known_bug_1139_flag returns True only for task-1139/bug-mechanics flags."""

    def _import(self):
        from fused_memory.reconciliation.stages.task_knowledge_sync import (
            _should_skip_known_bug_1139_flag,
        )
        return _should_skip_known_bug_1139_flag

    def test_skip_task_id_string_1139(self):
        fn = self._import()
        assert fn({'task_id': '1139'}) is True

    def test_skip_task_id_int_1139(self):
        """Integer task_id is coerced to string for comparison."""
        fn = self._import()
        assert fn({'task_id': 1139}) is True

    def test_skip_bug_mechanics_content_no_task_id(self):
        """Flag with bug-mechanics content is skipped even without task_id."""
        fn = self._import()
        flag = {
            'content': (
                'Stage 1 LLM writes flags to Mem0 with metadata.flag_for_stage2 '
                'but does NOT include them in flagged_items'
            ),
        }
        assert fn(flag) is True

    def test_skip_content_marker_flag_for_stage2_not_include(self):
        """The second content marker substring also triggers a skip."""
        fn = self._import()
        flag = {'content': 'flag_for_stage2=true but does NOT include them in flagged_items'}
        assert fn(flag) is True

    def test_pass_different_task_id_742(self):
        fn = self._import()
        assert fn({'task_id': '742'}) is False

    def test_pass_different_task_id_with_unrelated_content(self):
        fn = self._import()
        assert fn({'task_id': '742', 'content': 'unrelated finding'}) is False

    def test_pass_empty_dict(self):
        fn = self._import()
        assert fn({}) is False

    def test_pass_content_mentions_stage1_but_not_bug(self):
        fn = self._import()
        assert fn({'content': 'mentions Stage 1 but not the bug'}) is False


class TestAssumeUtc:
    """_assume_utc: naive datetimes get UTC attached; aware datetimes pass through unchanged."""

    def _import(self):
        from fused_memory.reconciliation.stages.task_knowledge_sync import _assume_utc
        return _assume_utc

    def test_naive_datetime_gets_utc(self):
        """A naive datetime has UTC tzinfo attached."""
        from datetime import UTC, datetime
        fn = self._import()
        naive = datetime(2026, 5, 15, 10, 0, 0)
        result = fn(naive)
        assert result.tzinfo is UTC
        assert result == datetime(2026, 5, 15, 10, 0, 0, tzinfo=UTC)

    def test_aware_utc_datetime_unchanged(self):
        """An already-UTC-aware datetime is returned unchanged."""
        from datetime import UTC, datetime
        fn = self._import()
        aware = datetime(2026, 5, 15, 10, 0, 0, tzinfo=UTC)
        result = fn(aware)
        assert result is aware

    def test_aware_non_utc_offset_unchanged(self):
        """An aware datetime with a non-UTC offset is returned unchanged (not coerced)."""
        from datetime import datetime, timedelta, timezone
        fn = self._import()
        plus5 = timezone(timedelta(hours=5))
        aware = datetime(2026, 5, 15, 15, 0, 0, tzinfo=plus5)
        result = fn(aware)
        assert result is aware
        assert result.tzinfo is plus5

    def test_naive_wall_clock_digits_preserved(self):
        """replace(tzinfo=UTC) is used — wall-clock digits are unchanged, not converted."""
        from datetime import UTC, datetime
        fn = self._import()
        naive = datetime(2026, 5, 15, 14, 30, 0)
        result = fn(naive)
        assert result.hour == 14
        assert result.minute == 30
        assert result.tzinfo is UTC

    def test_idempotent_on_naive(self):
        """Calling _assume_utc twice on a naive datetime yields the same result."""
        from datetime import UTC, datetime
        fn = self._import()
        naive = datetime(2026, 5, 15, 10, 0, 0)
        once = fn(naive)
        twice = fn(once)
        assert once == twice
        assert twice.tzinfo is UTC


class TestQueryStage2Flags:
    """_query_stage2_flags retrieves and filters Mem0 active-query flags."""

    def _make_result(self, id, content, metadata, created_at=None):
        from types import SimpleNamespace
        return SimpleNamespace(id=id, content=content, metadata=metadata, created_at=created_at)

    @pytest.mark.asyncio
    async def test_returns_flags_with_flag_for_stage2(self):
        from fused_memory.reconciliation.stages.task_knowledge_sync import _query_stage2_flags
        memory_service = AsyncMock()
        memory_service.search.return_value = [
            self._make_result('id-1', 'flag content', {'flag_for_stage2': True, 'task_id': '742', 'run_id': 'r-current'}),
            self._make_result('id-2', 'no flag', {}),
        ]
        current_flags, stale_missing_ids, stale_mismatched_ids, _rescued = await _query_stage2_flags(memory_service, 'reify', 'r-current')
        assert len(current_flags) == 1
        assert current_flags[0]['id'] == 'id-1'
        assert stale_missing_ids == []
        assert stale_mismatched_ids == []

    @pytest.mark.asyncio
    async def test_excludes_memories_without_either_marker(self):
        from fused_memory.reconciliation.stages.task_knowledge_sync import _query_stage2_flags
        memory_service = AsyncMock()
        memory_service.search.return_value = [
            self._make_result('id-4', 'irrelevant', {'some_other_key': True}),
            self._make_result('id-5', 'also irrelevant', {}),
        ]
        current_flags, stale_missing_ids, stale_mismatched_ids, _rescued = await _query_stage2_flags(memory_service, 'reify', 'r-current')
        assert current_flags == []
        assert stale_missing_ids == []
        assert stale_mismatched_ids == []

    @pytest.mark.asyncio
    async def test_preserves_fields_and_extracts_task_id(self):
        from fused_memory.reconciliation.stages.task_knowledge_sync import _query_stage2_flags
        memory_service = AsyncMock()
        meta = {'flag_for_stage2': True, 'task_id': '742', 'extra': 'x', 'run_id': 'r-current'}
        memory_service.search.return_value = [
            self._make_result('id-6', 'content here', meta),
        ]
        current_flags, stale_missing_ids, stale_mismatched_ids, _rescued = await _query_stage2_flags(memory_service, 'reify', 'r-current')
        assert len(current_flags) == 1
        flag = current_flags[0]
        assert flag['id'] == 'id-6'
        assert flag['content'] == 'content here'
        assert flag['metadata'] == meta
        assert flag['task_id'] == '742'

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_search_exception(self, caplog):
        import logging

        from fused_memory.reconciliation.stages.task_knowledge_sync import _query_stage2_flags
        memory_service = AsyncMock()
        memory_service.search.side_effect = RuntimeError('Mem0 unavailable')
        with caplog.at_level(logging.WARNING):
            current_flags, stale_missing_ids, stale_mismatched_ids, _rescued = await _query_stage2_flags(memory_service, 'reify', 'r-current')
        assert current_flags == []
        assert stale_missing_ids == []
        assert stale_mismatched_ids == []
        assert any(r.levelno >= logging.WARNING for r in caplog.records)

    @pytest.mark.asyncio
    async def test_calls_search_with_project_id(self):
        from fused_memory.reconciliation.stages.task_knowledge_sync import _query_stage2_flags
        memory_service = AsyncMock()
        memory_service.search.return_value = []
        await _query_stage2_flags(memory_service, 'my_project', 'r-current')
        call_kwargs = memory_service.search.call_args
        assert call_kwargs is not None
        # project_id must be passed as kwarg or positional
        kwargs = call_kwargs.kwargs
        args = call_kwargs.args
        all_args = list(args) + list(kwargs.values())
        assert 'my_project' in all_args or kwargs.get('project_id') == 'my_project'

    @pytest.mark.asyncio
    async def test_partitions_by_run_id(self):
        """Markers with matching run_id go to current; mismatched or missing go to stale IDs."""
        from fused_memory.reconciliation.stages.task_knowledge_sync import _query_stage2_flags
        memory_service = AsyncMock()
        memory_service.search.return_value = [
            self._make_result(
                'mem0-current',
                'content for task 742',
                {'flag_for_stage2': True, 'task_id': '742', 'run_id': 'r-current'},
            ),
            self._make_result(
                'mem0-prior',
                'STALE content from prior run',
                {'flag_for_stage2': True, 'task_id': '888', 'run_id': 'r-prior'},
            ),
            self._make_result(
                'mem0-no-run-id',
                'NO_RUN_ID legacy content',
                {'flag_for_stage2': True, 'task_id': '999'},
            ),
        ]
        current_flags, stale_missing_ids, stale_mismatched_ids, _rescued = await _query_stage2_flags(memory_service, 'reify', 'r-current')
        stale_marker_ids = stale_missing_ids + stale_mismatched_ids

        # Only the current-cycle marker should be in current_flags
        assert len(current_flags) == 1
        assert current_flags[0]['id'] == 'mem0-current'

        # Combined stale partition contains only IDs (not dicts)
        assert set(stale_marker_ids) == {'mem0-prior', 'mem0-no-run-id'}
        assert len(stale_marker_ids) == 2
        # Confirm stale_marker_ids contains strings, not dicts
        assert all(isinstance(sid, str) for sid in stale_marker_ids)

    @pytest.mark.asyncio
    @pytest.mark.parametrize('metadata_run_id,filter_run_id,expect_current', [
        ('42', '42', True),   # string match → current
        ('43', '42', False),  # string mismatch → stale
        (None, '42', False),  # absent key → treated as missing → stale
        ('', '42', False),    # empty marker run_id → falsy guard → stale
        ('', '', False),      # both empty → still stale (truthy guard prevents matching)
        # Int variants omitted: production always writes string run_ids via the
        # LLM/prompt template; testing int coercion pins behavior with no
        # production caller.  str() coercion is still applied for robustness but
        # is not a contract we need to maintain for non-string producers.
    ])
    async def test_run_id_string_match_and_missing_cases(self, metadata_run_id, filter_run_id, expect_current):
        """Partition behaviour for str match, mismatch, and absent run_id."""
        from fused_memory.reconciliation.stages.task_knowledge_sync import _query_stage2_flags
        memory_service = AsyncMock()
        meta = {'flag_for_stage2': True, 'task_id': '1'}
        if metadata_run_id is not None:
            meta['run_id'] = metadata_run_id
        memory_service.search.return_value = [
            self._make_result('test-id', 'content', meta),
        ]
        current_flags, stale_missing_ids, stale_mismatched_ids, _rescued = await _query_stage2_flags(
            memory_service, 'reify', filter_run_id
        )
        stale_marker_ids = stale_missing_ids + stale_mismatched_ids
        if expect_current:
            assert len(current_flags) == 1
            assert current_flags[0]['id'] == 'test-id'
            assert stale_marker_ids == []
        else:
            assert current_flags == []
            assert stale_marker_ids == ['test-id']

    @pytest.mark.asyncio
    async def test_partition_separates_missing_from_mismatched_stale(self):
        """Missing-run_id and mismatched-run_id markers land in distinct partition fields."""
        from fused_memory.reconciliation.stages.task_knowledge_sync import (
            Stage2FlagPartition,
            _query_stage2_flags,
        )
        memory_service = AsyncMock()
        memory_service.search.return_value = [
            # (a) matching run_id — should be in current
            self._make_result('current', 'content a', {'flag_for_stage2': True, 'task_id': '1', 'run_id': 'r-current'}),
            # (b) mismatched run_id — should be in stale_mismatched_run_id_ids
            self._make_result('prior', 'content b', {'flag_for_stage2': True, 'task_id': '2', 'run_id': 'r-prior'}),
            # (c) run_id key absent — should be in stale_missing_run_id_ids
            self._make_result('no-run-id', 'content c', {'flag_for_stage2': True, 'task_id': '3'}),
            # (d) run_id empty string — should be in stale_missing_run_id_ids
            self._make_result('empty-run-id', 'content d', {'flag_for_stage2': True, 'task_id': '4', 'run_id': ''}),
        ]
        partition = await _query_stage2_flags(memory_service, 'reify', 'r-current')

        # Partition must be the right type with 4 fields
        assert isinstance(partition, Stage2FlagPartition)
        assert len(partition) == 4

        # (a) only matching marker in current
        assert len(partition.current) == 1
        assert partition.current[0]['id'] == 'current'

        # (c) and (d) — missing run_id (absent or empty) in stale_missing_run_id_ids
        # Use set equality: order is incidental (follows search result iteration order)
        assert set(partition.stale_missing_run_id_ids) == {'no-run-id', 'empty-run-id'}

        # (b) — present but mismatched run_id
        assert partition.stale_mismatched_run_id_ids == ['prior']

    @pytest.mark.asyncio
    async def test_warning_logged_when_missing_run_id_count_nonzero(self, caplog):
        """A WARNING is emitted when any markers have absent/empty run_id."""
        import logging

        from fused_memory.reconciliation.stages.task_knowledge_sync import _query_stage2_flags
        memory_service = AsyncMock()
        memory_service.search.return_value = [
            # matching marker — current
            self._make_result('current', 'c', {'flag_for_stage2': True, 'task_id': '1', 'run_id': 'r-now'}),
            # absent run_id — missing
            self._make_result('no-id-1', 'a', {'flag_for_stage2': True, 'task_id': '2'}),
            # empty run_id — missing
            self._make_result('no-id-2', 'b', {'flag_for_stage2': True, 'task_id': '3', 'run_id': ''}),
        ]
        with caplog.at_level(
            logging.WARNING,
            logger='fused_memory.reconciliation.stages.task_knowledge_sync',
        ):
            await _query_stage2_flags(memory_service, 'reify', 'r-now')

        warning_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and 'missing' in r.getMessage().lower()
            and 'run_id' in r.getMessage().lower()
        ]
        assert len(warning_records) == 1, (
            f'Expected exactly 1 missing-run_id WARNING, got {len(warning_records)}: '
            f'{[r.getMessage() for r in warning_records]}'
        )
        # The structured extra dict must carry the exact count — avoids fragile substring match
        assert warning_records[0].missing_run_id_count == 2, (
            f'Expected missing_run_id_count=2 in log extra, '
            f'got: {getattr(warning_records[0], "missing_run_id_count", "<absent>")}'
        )

    @pytest.mark.asyncio
    async def test_no_missing_run_id_warning_when_count_is_zero(self, caplog):
        """No WARNING about missing run_id when all stale markers have a truthy run_id."""
        import logging

        from fused_memory.reconciliation.stages.task_knowledge_sync import _query_stage2_flags
        memory_service = AsyncMock()
        memory_service.search.return_value = [
            # matching marker
            self._make_result('current', 'c', {'flag_for_stage2': True, 'task_id': '1', 'run_id': 'r-now'}),
            # mismatched but truthy run_id — stale_mismatched, not missing
            self._make_result('prior', 'p', {'flag_for_stage2': True, 'task_id': '2', 'run_id': 'r-old'}),
        ]
        with caplog.at_level(
            logging.WARNING,
            logger='fused_memory.reconciliation.stages.task_knowledge_sync',
        ):
            await _query_stage2_flags(memory_service, 'reify', 'r-now')

        missing_warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and 'missing' in r.getMessage().lower()
        ]
        assert missing_warnings == [], (
            f'Expected no missing-run_id WARNING, but got: {[r.getMessage() for r in missing_warnings]}'
        )

    # ------------------------------------------------------------------ #
    # step-3 (task-1369): window-aware partition tests                     #
    # These FAIL until step-4 adds run_window_start param + window logic.  #
    # ------------------------------------------------------------------ #

    @pytest.mark.asyncio
    async def test_missing_run_id_in_window_routes_to_current(self):
        """(a) Marker with MISSING run_id but created_at >= run_window_start goes to current."""
        from fused_memory.reconciliation.stages.task_knowledge_sync import _query_stage2_flags

        run_window_start = datetime(2026, 5, 15, 10, 0, 0, tzinfo=UTC)
        memory_service = AsyncMock()
        memory_service.search.return_value = [
            self._make_result(
                'in-window-no-rid',
                'same-cycle flag',
                {'flag_for_stage2': True, 'task_id': '77'},
                created_at='2026-05-15T10:00:01+00:00',  # 1s after window start
            ),
        ]
        partition = await _query_stage2_flags(
            memory_service, 'reify', 'r-current', run_window_start=run_window_start
        )
        # In-window marker must go to current, NOT to stale buckets
        assert any(f['id'] == 'in-window-no-rid' for f in partition.current), (
            'Same-cycle marker (missing run_id but in window) must appear in current'
        )
        assert 'in-window-no-rid' not in partition.stale_missing_run_id_ids
        assert 'in-window-no-rid' not in partition.stale_mismatched_run_id_ids

    @pytest.mark.asyncio
    async def test_mismatched_run_id_in_window_routes_to_current(self):
        """(b) Marker with MISMATCHED run_id but created_at >= run_window_start goes to current."""
        from fused_memory.reconciliation.stages.task_knowledge_sync import _query_stage2_flags

        run_window_start = datetime(2026, 5, 15, 10, 0, 0, tzinfo=UTC)
        memory_service = AsyncMock()
        memory_service.search.return_value = [
            self._make_result(
                'in-window-mismatch',
                'mis-stamped run_id flag',
                {'flag_for_stage2': True, 'task_id': '88', 'run_id': 'wrong-run-id'},
                created_at='2026-05-15T10:00:02+00:00',
            ),
        ]
        partition = await _query_stage2_flags(
            memory_service, 'reify', 'r-current', run_window_start=run_window_start
        )
        assert any(f['id'] == 'in-window-mismatch' for f in partition.current), (
            'Same-cycle marker (mismatched run_id but in window) must appear in current'
        )
        assert 'in-window-mismatch' not in partition.stale_mismatched_run_id_ids
        assert 'in-window-mismatch' not in partition.stale_missing_run_id_ids

    @pytest.mark.asyncio
    async def test_out_of_window_stale_marker_stays_stale(self):
        """(c) Marker with stale run_id and created_at BEFORE run_window_start stays in stale bucket."""
        from fused_memory.reconciliation.stages.task_knowledge_sync import _query_stage2_flags

        run_window_start = datetime(2026, 5, 15, 10, 0, 0, tzinfo=UTC)
        memory_service = AsyncMock()
        memory_service.search.return_value = [
            self._make_result(
                'out-of-window',
                'prior cycle flag',
                {'flag_for_stage2': True, 'task_id': '99', 'run_id': 'old-run'},
                created_at='2026-05-15T09:00:00+00:00',  # 1h BEFORE window start
            ),
        ]
        partition = await _query_stage2_flags(
            memory_service, 'reify', 'r-current', run_window_start=run_window_start
        )
        assert 'out-of-window' in partition.stale_mismatched_run_id_ids, (
            'Prior-cycle marker (out of window) must remain in stale bucket'
        )
        assert not any(f['id'] == 'out-of-window' for f in partition.current)

    @pytest.mark.asyncio
    async def test_run_window_start_none_preserves_original_behavior(self):
        """(d) run_window_start=None -> same partition as today (stale stays stale)."""
        from fused_memory.reconciliation.stages.task_knowledge_sync import _query_stage2_flags

        memory_service = AsyncMock()
        memory_service.search.return_value = [
            self._make_result(
                'no-run-id',
                'legacy flag',
                {'flag_for_stage2': True, 'task_id': '5'},
                created_at='2026-05-15T10:00:00+00:00',
            ),
            self._make_result(
                'mismatch',
                'old run flag',
                {'flag_for_stage2': True, 'task_id': '6', 'run_id': 'r-old'},
                created_at='2026-05-15T10:00:00+00:00',
            ),
        ]
        # No run_window_start — original behavior: stale stays stale
        partition = await _query_stage2_flags(
            memory_service, 'reify', 'r-current', run_window_start=None
        )
        assert 'no-run-id' in partition.stale_missing_run_id_ids
        assert 'mismatch' in partition.stale_mismatched_run_id_ids
        assert partition.current == []

    @pytest.mark.asyncio
    async def test_none_created_at_with_valid_window_stays_stale(self):
        """(e) created_at=None with stale run_id and a valid window -> stale, no exception."""
        from fused_memory.reconciliation.stages.task_knowledge_sync import _query_stage2_flags

        run_window_start = datetime(2026, 5, 15, 10, 0, 0, tzinfo=UTC)
        memory_service = AsyncMock()
        memory_service.search.return_value = [
            self._make_result(
                'no-ts',
                'no timestamp',
                {'flag_for_stage2': True, 'task_id': '7'},  # missing run_id
                created_at=None,
            ),
        ]
        # Must not raise; created_at=None -> window guard dormant -> stale bucket
        partition = await _query_stage2_flags(
            memory_service, 'reify', 'r-current', run_window_start=run_window_start
        )
        assert 'no-ts' in partition.stale_missing_run_id_ids, (
            'Marker with created_at=None must fall through to stale bucket (window guard dormant)'
        )
        assert not any(f['id'] == 'no-ts' for f in partition.current)

    @pytest.mark.asyncio
    async def test_unparseable_created_at_with_valid_window_stays_stale(self):
        """(e-b) Unparseable created_at with stale run_id and a valid window -> stale, no exception."""
        from fused_memory.reconciliation.stages.task_knowledge_sync import _query_stage2_flags

        run_window_start = datetime(2026, 5, 15, 10, 0, 0, tzinfo=UTC)
        memory_service = AsyncMock()
        memory_service.search.return_value = [
            self._make_result(
                'bad-ts',
                'bad timestamp',
                {'flag_for_stage2': True, 'task_id': '8', 'run_id': 'old-run'},
                created_at='not-a-date',
            ),
        ]
        partition = await _query_stage2_flags(
            memory_service, 'reify', 'r-current', run_window_start=run_window_start
        )
        assert 'bad-ts' in partition.stale_mismatched_run_id_ids, (
            'Marker with unparseable created_at must fall through to stale bucket'
        )
        assert not any(f['id'] == 'bad-ts' for f in partition.current)

    @pytest.mark.asyncio
    async def test_matching_run_id_always_current_regardless_of_window(self):
        """(f) Matching run_id always goes to current even when created_at precedes window."""
        from fused_memory.reconciliation.stages.task_knowledge_sync import _query_stage2_flags

        run_window_start = datetime(2026, 5, 15, 10, 0, 0, tzinfo=UTC)
        memory_service = AsyncMock()
        memory_service.search.return_value = [
            self._make_result(
                'match-old-ts',
                'correct run_id, old timestamp',
                {'flag_for_stage2': True, 'task_id': '9', 'run_id': 'r-current'},
                created_at='2026-05-14T10:00:00+00:00',  # 1 day before window
            ),
        ]
        partition = await _query_stage2_flags(
            memory_service, 'reify', 'r-current', run_window_start=run_window_start
        )
        assert len(partition.current) == 1
        assert partition.current[0]['id'] == 'match-old-ts', (
            'Matching run_id must always go to current regardless of created_at/window'
        )
        assert partition.stale_missing_run_id_ids == []
        assert partition.stale_mismatched_run_id_ids == []

    @pytest.mark.asyncio
    async def test_clock_skew_grace_rescues_marker_just_before_window_start(self):
        """Clock-skew grace: marker timestamped a few seconds before run_window_start is rescued.

        The run_window_start is sourced from the orchestrator clock while created_at is
        stamped by the Mem0 server.  A small negative inter-host skew can legitimately
        produce created_at < run_window_start even for a same-cycle write.
        _CLOCK_SKEW_GRACE (30 s) absorbs this: any marker within 30 s before the window
        start is treated as in-window and routed to current rather than swept.
        """
        from fused_memory.reconciliation.stages.task_knowledge_sync import (
            _CLOCK_SKEW_GRACE,
            _query_stage2_flags,
        )

        run_window_start = datetime(2026, 5, 15, 10, 0, 0, tzinfo=UTC)
        # Marker written 10 s before run_window_start — within the 30 s grace window
        skewed_created_at = (run_window_start - timedelta(seconds=10)).isoformat()
        memory_service = AsyncMock()
        memory_service.search.return_value = [
            self._make_result(
                'skewed-ts',
                'same-cycle flag with clock skew',
                {'flag_for_stage2': True, 'task_id': '5'},  # missing run_id
                created_at=skewed_created_at,
            ),
        ]
        partition = await _query_stage2_flags(
            memory_service, 'reify', 'r-current', run_window_start=run_window_start
        )
        assert any(f['id'] == 'skewed-ts' for f in partition.current), (
            f'Marker written {int(_CLOCK_SKEW_GRACE.total_seconds())}s-grace before '
            f'run_window_start must be rescued to current, but got: '
            f'current={[f["id"] for f in partition.current]}, '
            f'stale_missing={partition.stale_missing_run_id_ids}'
        )
        assert 'skewed-ts' not in partition.stale_missing_run_id_ids
        assert 'skewed-ts' not in partition.stale_mismatched_run_id_ids

    @pytest.mark.asyncio
    async def test_marker_outside_grace_window_stays_stale(self):
        """Marker written more than _CLOCK_SKEW_GRACE before run_window_start stays stale."""
        from fused_memory.reconciliation.stages.task_knowledge_sync import (
            _CLOCK_SKEW_GRACE,
            _query_stage2_flags,
        )

        run_window_start = datetime(2026, 5, 15, 10, 0, 0, tzinfo=UTC)
        # Marker written 60 s before run_window_start — outside the 30 s grace window
        outside_created_at = (run_window_start - timedelta(seconds=60)).isoformat()
        memory_service = AsyncMock()
        memory_service.search.return_value = [
            self._make_result(
                'outside-grace',
                'pre-window flag outside grace',
                {'flag_for_stage2': True, 'task_id': '6'},  # missing run_id
                created_at=outside_created_at,
            ),
        ]
        partition = await _query_stage2_flags(
            memory_service, 'reify', 'r-current', run_window_start=run_window_start
        )
        assert 'outside-grace' in partition.stale_missing_run_id_ids, (
            f'Marker {int(_CLOCK_SKEW_GRACE.total_seconds())+30}s before run_window_start '
            f'must NOT be rescued (outside {int(_CLOCK_SKEW_GRACE.total_seconds())}s grace)'
        )
        assert not any(f['id'] == 'outside-grace' for f in partition.current)

    @pytest.mark.asyncio
    @pytest.mark.parametrize('case_id, created_at, metadata, in_window', [
        pytest.param(
            'naive-no-offset',
            '2026-05-15T10:00:05',  # no offset, no Z — triggers replace(tzinfo=UTC) branch
            {'flag_for_stage2': True, 'task_id': '101'},  # no run_id → stale-missing absent guard
            True,   # 10:00:05 UTC > 09:59:30 threshold → in-window
            id='naive_no_offset',
        ),
        pytest.param(
            'non-utc-offset',
            '2026-05-15T10:00:10+05:00',  # = 05:00:10 UTC — BEFORE the 09:59:30 threshold
            {'flag_for_stage2': True, 'task_id': '102', 'run_id': 'wrong-run'},  # mismatched → stale absent guard
            False,  # 05:00:10 UTC < 09:59:30 threshold → out-of-window; stays stale
            id='non_utc_offset_out_of_window',
        ),
        pytest.param(
            'z-suffixed',
            '2026-05-15T10:00:05Z',  # Z suffix — fromisoformat returns tz-aware UTC directly
            {'flag_for_stage2': True, 'task_id': '103'},  # no run_id → stale-missing absent guard
            True,   # 10:00:05 UTC > 09:59:30 threshold → in-window
            id='z_suffix',
        ),
    ])
    async def test_created_at_timestamp_format_normalisation(
        self, case_id, created_at, metadata, in_window
    ):
        """_marker_is_within_run_window normalises created_at regardless of ISO 8601 format.

        Three timestamp formats correspond to distinct parse paths, each tested via a
        parametrised case using run_window_start=2026-05-15T10:00:00+00:00
        (threshold = window_start - _CLOCK_SKEW_GRACE = 09:59:30 UTC):

          naive_no_offset — '2026-05-15T10:00:05' (no offset, no Z suffix):
            fromisoformat returns a naive datetime; the ``if parsed.tzinfo is None``
            branch inside _marker_is_within_run_window normalises it to 10:00:05+00:00.
            The marker (no run_id) is trivially after window start → in-window →
            rescued to partition.current.

          non_utc_offset_out_of_window — '2026-05-15T10:00:10+05:00' = 05:00:10 UTC:
            fromisoformat returns an offset-aware datetime. 05:00:10 UTC is *before*
            the 09:59:30 threshold, so the marker (mismatched run_id) stays in
            stale_mismatched_run_id_ids. A naive-stripping regression would read
            the wall-clock digits as 10:00:10 UTC (> threshold) and wrongly rescue
            it — making this the decisive differentiator between correct offset-aware
            comparison and a buggy naive-stripping implementation. Mirrors the
            MemoryResult.created_at contract: "UTC offset is not guaranteed — Mem0
            may stamp in any offset".

          z_suffix — '2026-05-15T10:00:05Z':
            fromisoformat on Python >= 3.11 returns tz-aware UTC directly (project
            requires-python >=3.11,<4), so the ``tzinfo is None`` branch is NOT
            entered. 10:00:05+00:00 is after window start → in-window → rescued to
            partition.current.

        Regression guard: locks in already-correct behaviour against future refactors
        of _marker_is_within_run_window.
        """
        from fused_memory.reconciliation.stages.task_knowledge_sync import _query_stage2_flags

        run_window_start = datetime(2026, 5, 15, 10, 0, 0, tzinfo=UTC)
        memory_service = AsyncMock()
        memory_service.search.return_value = [
            self._make_result(case_id, 'ts-format normalisation test', metadata, created_at=created_at),
        ]
        partition = await _query_stage2_flags(
            memory_service, 'reify', 'r-current', run_window_start=run_window_start
        )
        if in_window:
            assert any(f['id'] == case_id for f in partition.current), (
                f'{case_id!r}: in-window created_at must rescue marker to partition.current'
            )
            assert case_id not in partition.stale_missing_run_id_ids
            assert case_id not in partition.stale_mismatched_run_id_ids
        else:
            assert not any(f['id'] == case_id for f in partition.current), (
                f'{case_id!r}: out-of-window created_at must NOT rescue marker to current'
            )
            # run_id='wrong-run' routes to stale_mismatched when not rescued
            assert case_id in partition.stale_mismatched_run_id_ids

    @pytest.mark.asyncio
    async def test_info_log_emitted_when_missing_run_id_marker_rescued(self, caplog):
        """An INFO log is emitted when a missing-run_id marker is rescued by the run-window guard."""
        import logging

        from fused_memory.reconciliation.stages.task_knowledge_sync import _query_stage2_flags

        run_window_start = datetime(2026, 5, 15, 10, 0, 0, tzinfo=UTC)
        memory_service = AsyncMock()
        memory_service.search.return_value = [
            self._make_result(
                'rescued-missing',
                'rescued same-cycle flag',
                {'flag_for_stage2': True, 'task_id': '7'},  # no run_id
                created_at='2026-05-15T10:00:05+00:00',  # in-window
            ),
        ]
        with caplog.at_level(logging.INFO,
                             logger='fused_memory.reconciliation.stages.task_knowledge_sync'):
            await _query_stage2_flags(
                memory_service, 'reify', 'r-current', run_window_start=run_window_start
            )

        info_records = [
            r for r in caplog.records
            if r.levelno == logging.INFO and 'rescued' in r.getMessage().lower()
        ]
        assert info_records, (
            'Expected an INFO log mentioning "rescued" when run-window guard rescues '
            'a same-cycle missing-run_id marker'
        )

    @pytest.mark.asyncio
    async def test_partition_return_is_4_field_stage2flagpartition(self):
        """Return is a 4-field Stage2FlagPartition (current, stale_missing_run_id_ids,
        stale_mismatched_run_id_ids, rescued_ids) with run_window_start active."""
        from fused_memory.reconciliation.stages.task_knowledge_sync import (
            Stage2FlagPartition,
            _query_stage2_flags,
        )

        run_window_start = datetime(2026, 5, 15, 10, 0, 0, tzinfo=UTC)
        memory_service = AsyncMock()
        memory_service.search.return_value = []
        partition = await _query_stage2_flags(
            memory_service, 'reify', 'r-current', run_window_start=run_window_start
        )
        assert isinstance(partition, Stage2FlagPartition)
        assert len(partition) == 4
        assert partition.rescued_ids == []

    @pytest.mark.asyncio
    async def test_rescued_ids_field_collects_window_guard_rescued_marker_ids(self):
        """Stage2FlagPartition exposes a rescued_ids field tracking markers rescued by the
        run-window guard in BOTH branches (missing run_id + mismatched run_id), but NOT
        clean markers or genuinely-stale out-of-window markers.
        """
        from fused_memory.reconciliation.stages.task_knowledge_sync import (
            _query_stage2_flags,
        )

        run_window_start = datetime(2026, 5, 15, 10, 0, 0, tzinfo=UTC)
        memory_service = AsyncMock()
        memory_service.search.return_value = [
            # (a) clean matching run_id — in partition.current, NOT rescued
            self._make_result(
                'clean-current',
                'clean current flag',
                {'flag_for_stage2': True, 'task_id': '1', 'run_id': 'r-current'},
                created_at=None,
            ),
            # (b) missing run_id, created_at in-window — rescued by window guard
            self._make_result(
                'rescued-missing',
                'rescued missing run_id flag',
                {'flag_for_stage2': True, 'task_id': '2'},
                created_at='2026-05-15T10:00:05+00:00',
            ),
            # (c) mismatched run_id, created_at in-window — rescued by window guard
            self._make_result(
                'rescued-mismatch',
                'rescued mismatched run_id flag',
                {'flag_for_stage2': True, 'task_id': '3', 'run_id': 'r-other'},
                created_at='2026-05-15T10:00:10+00:00',
            ),
            # (d) missing run_id, created_at out-of-window — genuinely stale, NOT rescued
            self._make_result(
                'stale-out-of-window',
                'genuinely stale out-of-window flag',
                {'flag_for_stage2': True, 'task_id': '4'},
                created_at='2026-05-15T08:00:00+00:00',
            ),
        ]

        partition = await _query_stage2_flags(
            memory_service, 'reify', 'r-current', run_window_start=run_window_start
        )

        # The partition must expose the rescued_ids field (4th field of Stage2FlagPartition)
        assert hasattr(partition, 'rescued_ids'), (
            "Stage2FlagPartition must expose a 'rescued_ids' field populated by "
            "the run-window guard branches"
        )

        # (b) and (c) must appear in rescued_ids (both rescue branches)
        assert set(partition.rescued_ids) == {'rescued-missing', 'rescued-mismatch'}, (
            f"rescued_ids should contain exactly the two window-guard-rescued markers; "
            f"got: {partition.rescued_ids}"
        )

        # (a) clean marker is NOT in rescued_ids (it matched run_id cleanly)
        assert 'clean-current' not in partition.rescued_ids, (
            "Clean matching-run_id marker must NOT be in rescued_ids"
        )

        # (d) genuinely stale marker is NOT in rescued_ids
        assert 'stale-out-of-window' not in partition.rescued_ids, (
            "Out-of-window genuinely stale marker must NOT be in rescued_ids"
        )

        # (b) and (c) must still be in partition.current (rescue routes them to current)
        current_ids = {f['id'] for f in partition.current}
        assert 'rescued-missing' in current_ids, (
            "Rescued missing-run_id marker must still appear in partition.current"
        )
        assert 'rescued-mismatch' in current_ids, (
            "Rescued mismatched-run_id marker must still appear in partition.current"
        )

        # (d) stale out-of-window marker goes to stale_missing_run_id_ids
        assert 'stale-out-of-window' in partition.stale_missing_run_id_ids, (
            "Out-of-window missing-run_id marker must be in stale_missing_run_id_ids"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize('case_id, created_at, metadata, naive_window_start, expect_rescued', [
        pytest.param(
            'naive-window-in-window',
            '2026-05-15T10:00:05+00:00',   # tz-aware, 5 s after window start
            {'flag_for_stage2': True, 'task_id': '201'},  # no run_id → triggers missing-run_id branch
            datetime(2026, 5, 15, 10, 0, 0),              # NAIVE: no tzinfo
            True,   # must be rescued (guard must stay active after normalization)
            id='in_window_with_naive_run_window_start',
        ),
        pytest.param(
            'naive-window-out-of-window',
            '2026-05-15T10:00:05+00:00',   # tz-aware, 10:00:05 UTC — ~1 h before threshold
            {'flag_for_stage2': True, 'task_id': '202', 'run_id': 'wrong-run'},  # mismatched
            datetime(2026, 5, 15, 11, 0, 0),              # NAIVE: 1 h later → threshold = 10:59:30 UTC
            False,  # 10:00:05 UTC < 10:59:30 threshold → out-of-window; must NOT be rescued
            id='out_of_window_with_naive_run_window_start',
        ),
    ])
    async def test_naive_run_window_start_does_not_disable_window_guard(
        self, case_id, created_at, metadata, naive_window_start, expect_rescued
    ):
        """Naive run_window_start must NOT silently disable the run-window sweep guard.

        Root cause (pre-fix): ``_marker_is_within_run_window`` normalizes the
        parsed *created_at* to UTC when naive but does NOT normalize
        *run_window_start*.  When *run_window_start* is naive the comparison
        ``parsed(tz-aware) >= run_window_start(naive) - _CLOCK_SKEW_GRACE`` raises
        ``TypeError: can't compare offset-naive and offset-aware datetimes``, which
        the ``except (ValueError, TypeError)`` clause swallows by returning ``False``
        for *every* marker.  The run-window guard is thereby silently disabled for
        the entire cycle — same-cycle Stage-1 markers whose run_id was
        omitted/mis-stamped are swept instead of rescued (the task-1369 regression).

        Fix (step-2): normalize a naive ``run_window_start`` to UTC immediately
        after the ``isinstance(run_window_start, datetime)`` guard (mirroring the
        existing ``parsed`` normalization convention documented in the docstring).

        Two parametrised cases both use a NAIVE ``run_window_start`` (no tzinfo):

          in_window_with_naive_run_window_start:
            run_window_start = datetime(2026-05-15 10:00:00) [naive, assumed UTC after fix]
            created_at = '2026-05-15T10:00:05+00:00' (5 s after window start, in-window)
            metadata has no run_id → triggers the missing-run_id guard branch.
            Before fix: guard disabled (TypeError swallowed) → marker swept, NOT rescued.
            After fix:  guard active → marker rescued to partition.current.

          out_of_window_with_naive_run_window_start:
            run_window_start = datetime(2026-05-15 11:00:00) [naive, assumed UTC after fix]
            threshold = 11:00:00 - 30 s = 10:59:30 UTC
            created_at = '2026-05-15T10:00:05+00:00' = 10:00:05 UTC (< threshold, out-of-window)
            metadata has mismatched run_id → stale_mismatched_run_id_ids when not rescued.
            Before fix: guard disabled → marker incorrectly swept (stale_mismatched).
            After fix:  guard active AND ordering preserved → marker correctly stays stale.
            Proves normalization does not over-rescue (window ordering is maintained).
        """
        from fused_memory.reconciliation.stages.task_knowledge_sync import _query_stage2_flags

        memory_service = AsyncMock()
        memory_service.search.return_value = [
            self._make_result(case_id, 'naive-window-guard test', metadata, created_at=created_at),
        ]
        partition = await _query_stage2_flags(
            memory_service, 'reify', 'r-current', run_window_start=naive_window_start
        )
        if expect_rescued:
            assert any(f['id'] == case_id for f in partition.current), (
                f'{case_id!r}: in-window marker must be rescued to partition.current '
                f'even when run_window_start is naive (was guard silently disabled?)'
            )
            assert case_id not in partition.stale_missing_run_id_ids
            assert case_id not in partition.stale_mismatched_run_id_ids
        else:
            assert not any(f['id'] == case_id for f in partition.current), (
                f'{case_id!r}: out-of-window marker must NOT be rescued to partition.current'
            )
            # mismatched run_id routes to stale_mismatched when not rescued
            assert case_id in partition.stale_mismatched_run_id_ids, (
                f'{case_id!r}: out-of-window mismatched-run_id marker must stay in '
                f'stale_mismatched_run_id_ids (normalization must preserve ordering)'
            )


class TestSweepStaleFixcMarkers:
    """_sweep_stale_fixc_markers deletes stale fixc markers in parallel and returns count."""

    @pytest.mark.asyncio
    async def test_deletes_each_id_in_parallel_and_returns_count(self):
        from fused_memory.reconciliation.stages.task_knowledge_sync import _sweep_stale_fixc_markers
        memory_service = AsyncMock()
        memory_service.delete_memory = AsyncMock(return_value=None)
        stale_ids = ['m1', 'm2', 'm3']

        result = await _sweep_stale_fixc_markers(
            memory_service, project_id='reify', stale_ids=stale_ids, run_id='r-current'
        )

        assert result == 3
        assert memory_service.delete_memory.await_count == 3

        # Verify each call carries the required kwargs
        called_memory_ids = {
            call.kwargs.get('memory_id') or call.args[0]
            for call in memory_service.delete_memory.call_args_list
        }
        assert called_memory_ids == {'m1', 'm2', 'm3'}

        for call in memory_service.delete_memory.call_args_list:
            kwargs = call.kwargs
            assert kwargs.get('store') == 'mem0'
            assert kwargs.get('project_id') == 'reify'
            assert kwargs.get('causation_id') == 'r-current'
            assert kwargs.get('_source') == 'stage2_stale_fixc_sweep'

    @pytest.mark.asyncio
    async def test_empty_input_returns_zero_and_no_calls(self):
        from fused_memory.reconciliation.stages.task_knowledge_sync import _sweep_stale_fixc_markers
        memory_service = AsyncMock()
        memory_service.delete_memory = AsyncMock(return_value=None)

        result = await _sweep_stale_fixc_markers(
            memory_service, project_id='reify', stale_ids=[], run_id='r-current'
        )

        assert result == 0
        memory_service.delete_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_individual_failure_logs_warning_and_skipped_from_count(self, caplog):
        import logging

        from fused_memory.reconciliation.stages.task_knowledge_sync import _sweep_stale_fixc_markers
        memory_service = AsyncMock()
        # Middle delete raises; first and last succeed
        memory_service.delete_memory = AsyncMock(
            side_effect=[None, RuntimeError('boom'), None]
        )

        with caplog.at_level(
            logging.WARNING,
            logger='fused_memory.reconciliation.stages.task_knowledge_sync',
        ):
            result = await _sweep_stale_fixc_markers(
                memory_service,
                project_id='reify',
                stale_ids=['ok-1', 'bad', 'ok-2'],
                run_id='r-current',
            )

        # Must not raise; successful deletes counted; failure excluded
        assert result == 2
        # Exactly one WARNING for the failing memory_id
        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_records) == 1
        assert 'bad' in warning_records[0].getMessage()

    @pytest.mark.asyncio
    async def test_all_failures_returns_zero_and_does_not_raise(self, caplog):
        import logging

        from fused_memory.reconciliation.stages.task_knowledge_sync import _sweep_stale_fixc_markers
        memory_service = AsyncMock()
        memory_service.delete_memory = AsyncMock(
            side_effect=[RuntimeError('boom1'), RuntimeError('boom2'), RuntimeError('boom3')]
        )

        with caplog.at_level(
            logging.WARNING,
            logger='fused_memory.reconciliation.stages.task_knowledge_sync',
        ):
            result = await _sweep_stale_fixc_markers(
                memory_service,
                project_id='reify',
                stale_ids=['a', 'b', 'c'],
                run_id='r-current',
            )

        assert result == 0
        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_records) == 3


class TestComputeStaleFlags:
    """_compute_stale_flags returns flag_ids whose persistence count >= threshold."""

    def _fn(self):
        from fused_memory.reconciliation.stages.task_knowledge_sync import _compute_stale_flags
        return _compute_stale_flags

    def test_empty_dict_returns_empty(self):
        assert self._fn()({}) == []

    def test_all_below_threshold_returns_empty(self):
        counts = {'flag-A': 1, 'flag-B': 2}
        assert self._fn()(counts, threshold=3) == []

    def test_counts_at_threshold_are_returned(self):
        counts = {'flag-A': 3, 'flag-B': 2}
        result = self._fn()(counts, threshold=3)
        assert 'flag-A' in result
        assert 'flag-B' not in result

    def test_counts_above_threshold_are_returned(self):
        counts = {'flag-A': 5, 'flag-B': 1}
        result = self._fn()(counts, threshold=3)
        assert 'flag-A' in result
        assert 'flag-B' not in result

    def test_result_is_sorted(self):
        counts = {'flag-C': 4, 'flag-A': 5, 'flag-B': 3}
        result = self._fn()(counts, threshold=3)
        assert result == sorted(result)

    def test_custom_threshold_value(self):
        counts = {'flag-X': 7, 'flag-Y': 3}
        result = self._fn()(counts, threshold=5)
        assert result == ['flag-X']

    def test_threshold_constant_equals_3(self):
        from fused_memory.reconciliation.stages.task_knowledge_sync import (
            STAGE2_FLAG_PERSISTENCE_THRESHOLD,
        )
        assert STAGE2_FLAG_PERSISTENCE_THRESHOLD == 3


class TestTrackFlagPersistence:
    """_track_flag_persistence writes markers and returns cycle counts.

    Counts prior markers via ``count_memories_by_metadata`` (deterministic
    Qdrant payload-filtered count) — NOT semantic search — so the count is
    reliable under realistic Mem0 load.
    """

    @pytest.mark.asyncio
    async def test_writes_marker_and_returns_prior_plus_one(self):
        from fused_memory.reconciliation.stages.task_knowledge_sync import _track_flag_persistence
        memory_service = AsyncMock()
        # 2 prior markers for flag-A — returned directly by the metadata-filtered count
        memory_service.count_memories_by_metadata.return_value = 2
        memory_service.add_memory.return_value = {'memory_ids': ['new-marker-1']}

        result = await _track_flag_persistence(memory_service, 'reify', 'run-1', ['flag-A'])

        assert result == {'flag-A': 3}  # 2 prior + 1 (this cycle)
        memory_service.add_memory.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_count_called_with_metadata_filter_for_flag_id(self):
        from fused_memory.reconciliation.stages.task_knowledge_sync import _track_flag_persistence
        memory_service = AsyncMock()
        memory_service.count_memories_by_metadata.return_value = 0
        memory_service.add_memory.return_value = {'memory_ids': []}

        await _track_flag_persistence(memory_service, 'proj', 'run-2', ['flag-B'])

        call = memory_service.count_memories_by_metadata.call_args
        assert call is not None
        kwargs = call.kwargs
        assert kwargs.get('project_id') == 'proj'
        filters = kwargs.get('filters') or {}
        # Filter must pin BOTH the marker source AND the flag id so the count
        # is exact, not a similarity ranking that can drop matches.
        assert filters.get('source') == 'stage2_persistence_marker'
        assert filters.get('flag_id') == 'flag-B'
        # Critically: must NOT use semantic search anymore.
        memory_service.search.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_add_memory_called_with_persistence_marker_metadata(self):
        from fused_memory.reconciliation.stages.task_knowledge_sync import _track_flag_persistence
        memory_service = AsyncMock()
        memory_service.count_memories_by_metadata.return_value = 0
        memory_service.add_memory.return_value = {'memory_ids': ['m1']}

        await _track_flag_persistence(memory_service, 'proj', 'run-3', ['flag-C'])

        call = memory_service.add_memory.call_args
        assert call is not None
        kwargs = call.kwargs
        meta = kwargs.get('metadata', {})
        assert meta.get('source') == 'stage2_persistence_marker'
        assert meta.get('flag_id') == 'flag-C'
        assert meta.get('run_id') == 'run-3'

    @pytest.mark.asyncio
    async def test_count_failure_degrades_to_count_1(self, caplog):
        """On count failure, prior count is 0, so this cycle count = 1."""
        import logging

        from fused_memory.reconciliation.stages.task_knowledge_sync import _track_flag_persistence
        memory_service = AsyncMock()
        memory_service.count_memories_by_metadata.side_effect = RuntimeError('Mem0 down')
        memory_service.add_memory.return_value = {'memory_ids': []}

        with caplog.at_level(logging.WARNING):
            result = await _track_flag_persistence(memory_service, 'proj', 'run-4', ['flag-D'])

        assert result == {'flag-D': 1}
        assert any(r.levelno >= logging.WARNING for r in caplog.records)

    @pytest.mark.asyncio
    async def test_add_memory_failure_still_returns_count(self, caplog):
        import logging

        from fused_memory.reconciliation.stages.task_knowledge_sync import _track_flag_persistence
        memory_service = AsyncMock()
        memory_service.count_memories_by_metadata.return_value = 1
        memory_service.add_memory.side_effect = RuntimeError('write failed')

        with caplog.at_level(logging.WARNING):
            result = await _track_flag_persistence(memory_service, 'proj', 'run-5', ['flag-E'])

        assert result == {'flag-E': 2}  # 1 prior + 1
        assert any(r.levelno >= logging.WARNING for r in caplog.records)

    @pytest.mark.asyncio
    async def test_empty_flag_ids_returns_empty_no_calls(self):
        from fused_memory.reconciliation.stages.task_knowledge_sync import _track_flag_persistence
        memory_service = AsyncMock()

        result = await _track_flag_persistence(memory_service, 'proj', 'run-6', [])

        assert result == {}
        memory_service.count_memories_by_metadata.assert_not_awaited()
        memory_service.add_memory.assert_not_awaited()


class TestFilterAlreadyEscalatedFlags:
    """_filter_already_escalated_flags suppresses flags that already carry an escalation marker."""

    @pytest.mark.asyncio
    async def test_partitions_by_marker_presence(self):
        from fused_memory.reconciliation.stages.task_knowledge_sync import (
            _filter_already_escalated_flags,
        )
        memory_service = AsyncMock()

        async def count_side_effect(*, project_id, filters):
            assert filters['source'] == 'stage2_escalation_marker'
            return 1 if filters['flag_id'] == 'flag-old' else 0

        memory_service.count_memories_by_metadata.side_effect = count_side_effect

        newly, already = await _filter_already_escalated_flags(
            memory_service, 'proj', ['flag-old', 'flag-new'],
        )
        assert newly == ['flag-new']
        assert already == ['flag-old']

    @pytest.mark.asyncio
    async def test_count_failure_treats_flag_as_newly_escalating(self, caplog):
        """A transient Qdrant glitch must not silently suppress a real escalation."""
        import logging

        from fused_memory.reconciliation.stages.task_knowledge_sync import (
            _filter_already_escalated_flags,
        )
        memory_service = AsyncMock()
        memory_service.count_memories_by_metadata.side_effect = RuntimeError('Mem0 down')

        with caplog.at_level(logging.WARNING):
            newly, already = await _filter_already_escalated_flags(
                memory_service, 'proj', ['flag-X'],
            )
        assert newly == ['flag-X']
        assert already == []
        assert any(r.levelno >= logging.WARNING for r in caplog.records)


class TestWriteEscalationMarkers:
    """_write_escalation_markers persists per-flag escalation markers."""

    @pytest.mark.asyncio
    async def test_marker_written_with_correct_metadata(self):
        from fused_memory.reconciliation.stages.task_knowledge_sync import (
            _write_escalation_markers,
        )
        memory_service = AsyncMock()
        memory_service.add_memory.return_value = {'memory_ids': ['m1']}

        await _write_escalation_markers(memory_service, 'proj', 'run-7', ['flag-Y'])

        call = memory_service.add_memory.call_args
        assert call is not None
        meta = call.kwargs.get('metadata', {})
        assert meta.get('source') == 'stage2_escalation_marker'
        assert meta.get('flag_id') == 'flag-Y'
        assert meta.get('run_id') == 'run-7'

    @pytest.mark.asyncio
    async def test_write_failure_logs_warning_does_not_raise(self, caplog):
        import logging

        from fused_memory.reconciliation.stages.task_knowledge_sync import (
            _write_escalation_markers,
        )
        memory_service = AsyncMock()
        memory_service.add_memory.side_effect = RuntimeError('write down')

        with caplog.at_level(logging.WARNING):
            await _write_escalation_markers(memory_service, 'proj', 'run-8', ['flag-Z'])

        assert any(r.levelno >= logging.WARNING for r in caplog.records)


class TestTaskKnowledgeSyncActiveQueryFlags:
    """assemble_payload merges Mem0 active-query flags into the flagged section."""

    @pytest.fixture
    def mock_deps(self):
        from fused_memory.config.schema import ReconciliationConfig
        config = ReconciliationConfig(enabled=True, explore_codebase_root='/tmp/test')
        memory_service = AsyncMock()
        # Persistence/escalation counters default to 0 so _track_flag_persistence
        # and _filter_already_escalated_flags produce arithmetic-safe ints when
        # this fixture's tests don't care about stale-flag rendering.
        memory_service.count_memories_by_metadata.return_value = 0
        return {
            'memory_service': memory_service,
            'taskmaster': AsyncMock(),
            'journal': AsyncMock(),
            'config': config,
        }

    @pytest.fixture
    def watermark(self):
        return Watermark(project_id='reify')

    def _make_mem0_flag(self, flag_id, content, task_id, run_id=None):
        from types import SimpleNamespace
        meta: dict = {'flag_for_stage2': True, 'task_id': task_id}
        if run_id is not None:
            meta['run_id'] = run_id
        return SimpleNamespace(id=flag_id, content=content, metadata=meta)

    @pytest.mark.asyncio
    async def test_payload_contains_both_stage1_and_mem0_flags(self, mock_deps, watermark):
        """Merged flagged section must contain Stage 1 items_flagged AND Mem0 active-query flags."""
        from datetime import UTC, datetime

        from fused_memory.models.reconciliation import StageId, StageReport
        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='reify', project_root='/home/leo/src/reify')

        # Mem0 active-query flags (both carry matching run_id so they are current)
        mock_deps['memory_service'].search.return_value = [
            self._make_mem0_flag('mem0-flag-1', 'mem0 flag content for task 742', '742', run_id='test-run'),
            self._make_mem0_flag('mem0-flag-2', 'mem0 flag content for task 888', '888', run_id='test-run'),
        ]
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}

        # Stage 1 items_flagged
        stage1_report = StageReport(
            stage=StageId.memory_consolidator,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items_flagged=[{'task_id': '99', 'flag_type': 'assumption_invalid',
                            'description': 'stage1 flagged content'}],
            stats={},
            llm_calls=1,
            tokens_used=100,
        )
        prior_reports = [stage1_report]

        payload = await stage.assemble_payload([], watermark, prior_reports)
        section = _extract_section(payload, '### Stage 1 Flagged Items')

        assert 'stage1 flagged content' in section, \
            'Stage 1 items_flagged must appear in the flagged section'
        assert 'mem0 flag content for task 742' in section, \
            'Mem0 active-query flag for task 742 must appear in the flagged section'
        assert 'mem0 flag content for task 888' in section, \
            'Mem0 active-query flag for task 888 must appear in the flagged section'

    @pytest.mark.asyncio
    async def test_search_exception_still_renders_stage1_flags(self, mock_deps, watermark):
        """When memory_service.search raises, payload still contains Stage 1 flags."""
        from datetime import UTC, datetime

        from fused_memory.models.reconciliation import StageId, StageReport
        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='reify', project_root='/home/leo/src/reify')

        mock_deps['memory_service'].search.side_effect = RuntimeError('Mem0 unavailable')
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}

        stage1_report = StageReport(
            stage=StageId.memory_consolidator,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items_flagged=[{'task_id': '77', 'description': 'stage1 only flag'}],
            stats={},
            llm_calls=1,
            tokens_used=100,
        )

        # Should not raise
        payload = await stage.assemble_payload([], watermark, [stage1_report])
        section = _extract_section(payload, '### Stage 1 Flagged Items')
        assert 'stage1 only flag' in section

    @pytest.mark.asyncio
    async def test_search_called_with_project_id(self, mock_deps, watermark):
        """memory_service.search must be called with project_id='reify'."""
        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='reify', project_root='/home/leo/src/reify')
        mock_deps['memory_service'].search.return_value = []
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}

        await stage.assemble_payload([], watermark, [])

        # At least one call to search should have used project_id='reify'
        calls = mock_deps['memory_service'].search.call_args_list
        assert len(calls) >= 1
        first_call = calls[0]
        project_id_used = (
            first_call.kwargs.get('project_id') or
            (first_call.args[1] if len(first_call.args) > 1 else None)
        )
        assert project_id_used == 'reify', \
            f'search must be called with project_id="reify", got: {project_id_used}'

    @pytest.mark.asyncio
    async def test_payload_excludes_stale_run_id_flags_and_sweeps_them(self, mock_deps, watermark):
        """Stale markers (wrong/absent run_id) must be excluded from payload and swept via delete_memory."""
        from types import SimpleNamespace
        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='reify', project_root='/home/leo/src/reify')

        mock_deps['memory_service'].delete_memory = AsyncMock(return_value=None)
        mock_deps['memory_service'].search.return_value = [
            SimpleNamespace(
                id='mem0-current',
                content='content for task 742',
                metadata={'flag_for_stage2': True, 'task_id': '742', 'run_id': 'test-run'},
            ),
            SimpleNamespace(
                id='mem0-prior',
                content='STALE content from prior run',
                metadata={'flag_for_stage2': True, 'task_id': '888', 'run_id': 'r-old'},
            ),
            SimpleNamespace(
                id='mem0-no-run-id',
                content='NO_RUN_ID legacy content',
                metadata={'flag_for_stage2': True, 'task_id': '999'},
            ),
        ]
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}

        payload = await stage.assemble_payload([], watermark, [])
        section = _extract_section(payload, '### Stage 1 Flagged Items')

        # Only current-cycle marker rendered to LLM
        assert 'content for task 742' in section
        assert 'STALE content from prior run' not in section
        assert 'NO_RUN_ID legacy content' not in section

        # Two stale markers swept via delete_memory
        assert mock_deps['memory_service'].delete_memory.await_count == 2
        swept_ids = {
            call.kwargs.get('memory_id')
            for call in mock_deps['memory_service'].delete_memory.call_args_list
        }
        assert swept_ids == {'mem0-prior', 'mem0-no-run-id'}
        for call in mock_deps['memory_service'].delete_memory.call_args_list:
            assert call.kwargs.get('store') == 'mem0'
            assert call.kwargs.get('_source') == 'stage2_stale_fixc_sweep'

    @pytest.mark.asyncio
    async def test_search_failure_yields_zero_swept_and_no_stale_partition(
        self, mock_deps, watermark,
    ):
        """On Mem0 search failure, assemble_payload must not raise, not sweep, and
        stage.run() must record stale_fixc_markers_swept=0 (not absent)."""
        from datetime import UTC, datetime

        from fused_memory.models.reconciliation import StageId, StageReport

        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='reify', project_root='/home/leo/src/reify')
        mock_deps['memory_service'].delete_memory = AsyncMock(return_value=None)
        mock_deps['memory_service'].search.side_effect = RuntimeError('Mem0 unavailable')
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}

        stage1_report = StageReport(
            stage=StageId.memory_consolidator,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items_flagged=[{'task_id': '77', 'description': 'stage1 only flag'}],
            stats={},
            llm_calls=1,
            tokens_used=100,
        )

        # (a) assemble_payload must not raise
        payload = await stage.assemble_payload([], watermark, [stage1_report])

        # (b) payload renders normally — Stage 1 section still present
        assert '### Stage 1 Flagged Items' in payload

        # (c) no sweep on search failure
        mock_deps['memory_service'].delete_memory.assert_not_awaited()

        # (d) stage.run() records stale_fixc_markers_swept=0 explicitly (not absent)
        mock_deps['memory_service'].add_memory.return_value = {'memory_ids': []}
        fake_cli_result = MagicMock(
            success=True,
            report={'flagged_items': [], 'summary': 'ok', 'stats': {}},
            llm_calls=1,
            tokens_used=0,
            cost_usd=0.0,
            model='test-model',
            error=None,
        )
        with patch('fused_memory.reconciliation.stages.base.run_stage_via_cli',
                   new=AsyncMock(return_value=fake_cli_result)):
            report = await stage.run(
                events=[], watermark=watermark, prior_reports=[], run_id='test-run',
            )

        assert 'stale_fixc_markers_swept' in report.stats
        assert report.stats['stale_fixc_markers_swept'] == 0

    @pytest.mark.asyncio
    async def test_assemble_payload_raises_before_search_when_current_run_id_is_none(
        self, mock_deps, watermark,
    ):
        """Guard must raise RuntimeError BEFORE any filter_task_tree / Mem0 / Taskmaster I/O
        when _current_run_id is not set, so callers get an early, attributable failure.

        After the guard was hoisted to the top of assemble_payload() (task 1273), the raise
        happens before the filter_task_tree branch, so neither taskmaster.get_tasks nor
        memory_service.search should be awaited.
        """
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'reify'
        stage.project_root = '/home/leo/src/reify'
        # Intentionally NOT setting stage._current_run_id — that is the SUT condition.
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}

        with pytest.raises(RuntimeError) as excinfo:
            await stage.assemble_payload([], watermark, [])

        # Guard must fire BEFORE any Taskmaster I/O (filter_task_tree fetch).
        mock_deps['taskmaster'].get_tasks.assert_not_awaited()
        # Guard must fire BEFORE the Mem0 search — no search round-trip on bad setup.
        mock_deps['memory_service'].search.assert_not_awaited()

        # Error message must name the attribute.
        msg = str(excinfo.value)
        assert '_current_run_id' in msg


class TestTaskKnowledgeSyncKnownBug1139ScopeFilter:
    """Scope filter suppresses task-1139/bug-mechanics flags from Mem0 active-query path."""

    @pytest.fixture
    def mock_deps(self):
        from fused_memory.config.schema import ReconciliationConfig
        config = ReconciliationConfig(enabled=True, explore_codebase_root='/tmp/test')
        memory_service = AsyncMock()
        memory_service.count_memories_by_metadata.return_value = 0
        return {
            'memory_service': memory_service,
            'taskmaster': AsyncMock(),
            'journal': AsyncMock(),
            'config': config,
        }

    @pytest.fixture
    def watermark(self):
        return Watermark(project_id='reify')

    def _make_flag(self, flag_id, content, task_id, run_id=None):
        from types import SimpleNamespace
        meta: dict = {'flag_for_stage2': True, 'task_id': task_id}
        if run_id is not None:
            meta['run_id'] = run_id
        return SimpleNamespace(
            id=flag_id,
            content=content,
            metadata=meta,
        )

    @pytest.mark.asyncio
    async def test_task_742_flag_passes_through(self, mock_deps, watermark):
        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='reify', project_root='/home/leo/src/reify')
        mock_deps['memory_service'].search.return_value = [
            self._make_flag('mem-742', 'legitimate finding for task 742', '742', run_id='test-run'),
            self._make_flag('mem-1139', 'some flag for task 1139', '1139', run_id='test-run'),
        ]
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}

        payload = await stage.assemble_payload([], watermark, [])
        section = _extract_section(payload, '### Stage 1 Flagged Items')

        assert 'legitimate finding for task 742' in section, \
            'task 742 flag must appear in the payload'

    @pytest.mark.asyncio
    async def test_task_1139_flag_suppressed(self, mock_deps, watermark):
        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='reify', project_root='/home/leo/src/reify')
        mock_deps['memory_service'].search.return_value = [
            # run_id='test-run' matches stage._current_run_id so this flag is NOT
            # excluded by the run-partition filter.  The task_id=1139 scope filter
            # is what we're testing here — it must suppress the flag regardless.
            self._make_flag('mem-1139', 'some flag for task 1139', '1139', run_id='test-run'),
        ]
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}

        payload = await stage.assemble_payload([], watermark, [])
        section = _extract_section(payload, '### Stage 1 Flagged Items')

        assert 'some flag for task 1139' not in section, \
            'task_id=1139 flags must be suppressed by the scope filter'

    @pytest.mark.asyncio
    async def test_bug_mechanics_content_suppressed(self, mock_deps, watermark):
        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='reify', project_root='/home/leo/src/reify')
        bug_content = (
            'Stage 1 LLM writes flags to Mem0 with metadata.flag_for_stage2 '
            'but does NOT include them in flagged_items'
        )
        mock_deps['memory_service'].search.return_value = [
            # run_id='test-run' matches stage._current_run_id so the flag passes the
            # run-partition filter; the bug-mechanics content-based scope filter is
            # what we're testing here — it must suppress the flag.
            self._make_flag('mem-bug', bug_content, '', run_id='test-run'),
        ]
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}

        payload = await stage.assemble_payload([], watermark, [])
        section = _extract_section(payload, '### Stage 1 Flagged Items')

        assert bug_content not in section, \
            'Bug-mechanics content must be suppressed by scope filter'

    @pytest.mark.asyncio
    async def test_stage1_items_flagged_for_task_1139_not_suppressed(self, mock_deps, watermark):
        """Stage 1 structured-output flags for task 1139 are NOT filtered.
        The scope filter only applies to the Mem0 active-query path."""
        from datetime import UTC, datetime

        from fused_memory.models.reconciliation import StageReport
        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='reify', project_root='/home/leo/src/reify')
        mock_deps['memory_service'].search.return_value = []
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}

        stage1_report = StageReport(
            stage=StageId.memory_consolidator,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items_flagged=[
                {'task_id': '1139', 'flag_type': 'assumption_invalid',
                 'description': 'stage1 emitted this intentionally for 1139'},
            ],
            stats={},
            llm_calls=1,
            tokens_used=100,
        )

        payload = await stage.assemble_payload([], watermark, [stage1_report])
        section = _extract_section(payload, '### Stage 1 Flagged Items')

        assert 'stage1 emitted this intentionally for 1139' in section, \
            'Stage 1 items_flagged are never scope-filtered; only Mem0 active-query flags are'


class TestTaskKnowledgeSyncStaleFlagEscalation:
    """assemble_payload renders a stale-flag section and logs a warning when count >= threshold."""

    @pytest.fixture
    def mock_deps(self):
        from fused_memory.config.schema import ReconciliationConfig
        config = ReconciliationConfig(enabled=True, explore_codebase_root='/tmp/test')
        memory_service = AsyncMock()
        # Per-test setup overrides this via _setup_persistence_count (which
        # installs a side_effect routing on the filter['source'] key).
        memory_service.count_memories_by_metadata.return_value = 0
        return {
            'memory_service': memory_service,
            'taskmaster': AsyncMock(),
            'journal': AsyncMock(),
            'config': config,
        }

    @pytest.fixture
    def watermark(self):
        return Watermark(project_id='reify')

    def _make_active_flag(self, flag_id, task_id='742', run_id=None):
        from types import SimpleNamespace
        meta: dict = {'flag_for_stage2': True, 'task_id': task_id}
        if run_id is not None:
            meta['run_id'] = run_id
        return SimpleNamespace(
            id=flag_id,
            content=f'active flag content for {flag_id}',
            metadata=meta,
        )

    def _setup_persistence_count(self, mock_deps, *, prior_count: int, escalated_count: int = 0):
        """Configure ``count_memories_by_metadata`` to return *prior_count* for
        persistence-marker queries and *escalated_count* for escalation-marker
        queries.  Returns the side-effect function (not strictly needed by
        callers — wired to the mock for you)."""
        async def count_side_effect(*, project_id, filters):
            source = filters.get('source')
            if source == 'stage2_persistence_marker':
                return prior_count
            if source == 'stage2_escalation_marker':
                return escalated_count
            return 0
        mock_deps['memory_service'].count_memories_by_metadata.side_effect = count_side_effect
        return count_side_effect

    @pytest.mark.asyncio
    async def test_stale_flag_section_rendered_when_count_at_threshold(
        self, mock_deps, watermark, caplog,
    ):
        """When a flag has 2 prior markers (cycle=3 >= threshold 3), payload shows stale section."""
        import logging
        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='reify', project_root='/home/leo/src/reify')
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}
        mock_deps['memory_service'].add_memory.return_value = {'memory_ids': []}

        # search returns active flags; count returns 2 prior persistence markers.
        mock_deps['memory_service'].search.return_value = [self._make_active_flag('flag-A', run_id='test-run')]
        self._setup_persistence_count(mock_deps, prior_count=2, escalated_count=0)

        with caplog.at_level(logging.WARNING):
            payload = await stage.assemble_payload([], watermark, [])

        assert '### Stale Flags Requiring Escalation' in payload, \
            'Payload must contain stale-flag section when cycle count >= threshold'
        assert 'flag-A' in payload

    @pytest.mark.asyncio
    async def test_stale_flag_warning_logged(self, mock_deps, watermark, caplog):
        """A WARNING is logged with reconciliation.stale_flag_escalated message."""
        import logging
        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='reify', project_root='/home/leo/src/reify')
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}
        mock_deps['memory_service'].add_memory.return_value = {'memory_ids': []}

        mock_deps['memory_service'].search.return_value = [self._make_active_flag('flag-A', run_id='test-run')]
        self._setup_persistence_count(mock_deps, prior_count=2, escalated_count=0)

        with caplog.at_level(logging.WARNING):
            await stage.assemble_payload([], watermark, [])

        stale_records = [
            r for r in caplog.records
            if r.getMessage().startswith('reconciliation.stale_flag_escalated')
        ]
        assert len(stale_records) >= 1, 'Must log reconciliation.stale_flag_escalated warning'
        assert getattr(stale_records[0], 'flag_id', None) == 'flag-A' or \
            'flag-A' in str(stale_records[0].__dict__)

    @pytest.mark.asyncio
    async def test_no_stale_section_when_count_below_threshold(self, mock_deps, watermark, caplog):
        """When persistence count is 1 (no prior markers), no stale-flag section, no warning."""
        import logging
        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='reify', project_root='/home/leo/src/reify')
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}
        mock_deps['memory_service'].add_memory.return_value = {'memory_ids': []}

        mock_deps['memory_service'].search.return_value = [self._make_active_flag('flag-B', run_id='test-run')]
        self._setup_persistence_count(mock_deps, prior_count=0, escalated_count=0)

        with caplog.at_level(logging.WARNING):
            payload = await stage.assemble_payload([], watermark, [])

        assert '### Stale Flags Requiring Escalation' not in payload
        stale_records = [
            r for r in caplog.records
            if r.getMessage().startswith('reconciliation.stale_flag_escalated')
        ]
        assert len(stale_records) == 0

    @pytest.mark.asyncio
    async def test_stale_flag_detection_runs_when_no_prior_reports(self, mock_deps, watermark):
        """Stale flag detection runs even with no prior_reports (empty Stage 1)."""
        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='reify', project_root='/home/leo/src/reify')
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}
        mock_deps['memory_service'].add_memory.return_value = {'memory_ids': []}

        mock_deps['memory_service'].search.return_value = [self._make_active_flag('flag-C', run_id='test-run')]
        self._setup_persistence_count(mock_deps, prior_count=2, escalated_count=0)

        # No prior_reports → prior_reports=[] (no Stage 1 report)
        payload = await stage.assemble_payload([], watermark, [])
        assert '### Stale Flags Requiring Escalation' in payload

    @pytest.mark.asyncio
    async def test_already_escalated_flag_suppressed_from_section(
        self, mock_deps, watermark, caplog,
    ):
        """When an escalation marker exists for the flag, do NOT re-render it.

        This guards against the cycle-spam failure mode where FIX C deletion
        fails: without dedup, every subsequent cycle re-emits the same flag in
        the stale section and the LLM re-escalates it indefinitely.
        """
        import logging
        stage = make_configured_task_knowledge_sync_stage(mock_deps, project_id='reify', project_root='/home/leo/src/reify')
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}
        mock_deps['memory_service'].add_memory.return_value = {'memory_ids': []}

        mock_deps['memory_service'].search.return_value = [self._make_active_flag('flag-A', run_id='test-run')]
        # Persistence count >= threshold AND an escalation marker already exists.
        self._setup_persistence_count(mock_deps, prior_count=5, escalated_count=1)

        with caplog.at_level(logging.INFO):
            payload = await stage.assemble_payload([], watermark, [])

        assert '### Stale Flags Requiring Escalation' not in payload, \
            'Already-escalated flag must NOT re-appear in the stale section'
        suppressed = [
            r for r in caplog.records
            if r.getMessage().startswith('reconciliation.stale_flag_escalation_suppressed')
        ]
        assert len(suppressed) >= 1, 'Suppression must be logged for operator visibility'

    @pytest.mark.asyncio
    async def test_escalation_marker_written_when_section_renders(
        self, mock_deps, watermark,
    ):
        """A newly-rendered stale section must persist an escalation marker."""
        from unittest.mock import call as mock_call
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'reify'
        stage.project_root = '/home/leo/src/reify'
        stage._current_run_id = 'run-marker-test'
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}
        mock_deps['memory_service'].add_memory.return_value = {'memory_ids': ['m1']}

        mock_deps['memory_service'].search.return_value = [self._make_active_flag('flag-A', run_id='run-marker-test')]
        self._setup_persistence_count(mock_deps, prior_count=2, escalated_count=0)

        await stage.assemble_payload([], watermark, [])

        # Persistence marker (from _track_flag_persistence) AND escalation marker
        # (from _write_escalation_markers) must both be written.
        sources_written = [
            (c.kwargs.get('metadata') or {}).get('source')
            for c in mock_deps['memory_service'].add_memory.call_args_list
        ]
        assert 'stage2_persistence_marker' in sources_written
        assert 'stage2_escalation_marker' in sources_written
        # Sanity: avoid lint warning on unused import
        assert mock_call


class TestTaskKnowledgeSyncStaleFixcSweptStat:
    """TaskKnowledgeSync.run() sets report.stats['stale_fixc_markers_swept'] after super().run()."""

    @pytest.fixture
    def mock_deps(self):
        from fused_memory.config.schema import ReconciliationConfig
        config = ReconciliationConfig(enabled=True, explore_codebase_root='/tmp/test')
        memory_service = AsyncMock()
        memory_service.count_memories_by_metadata.return_value = 0
        memory_service.delete_memory = AsyncMock(return_value=None)
        return {
            'memory_service': memory_service,
            'taskmaster': AsyncMock(),
            'journal': AsyncMock(),
            'config': config,
        }

    @pytest.mark.asyncio
    async def test_stale_fixc_markers_swept_stat_set_after_run(self, mock_deps):
        """run() injects stale_fixc_markers_swept into report.stats after super().run().

        Uses the run_stage_via_cli mock pattern (not BaseStage.run mock) so that
        assemble_payload executes for real and _stale_fixc_markers_swept is populated
        by the sweep before run() injects the count into report.stats.
        """
        from types import SimpleNamespace

        from fused_memory.models.reconciliation import StageId

        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'reify'
        stage.project_root = '/home/leo/src/reify'

        # One current-cycle marker and two stale markers
        mock_deps['memory_service'].search.return_value = [
            SimpleNamespace(
                id='current', content='current content',
                metadata={'flag_for_stage2': True, 'task_id': '1', 'run_id': 'test-run'},
            ),
            SimpleNamespace(
                id='stale-1', content='stale 1',
                metadata={'flag_for_stage2': True, 'task_id': '2', 'run_id': 'old-run'},
            ),
            SimpleNamespace(
                id='stale-2', content='stale 2',
                metadata={'flag_for_stage2': True, 'task_id': '3'},
            ),
        ]
        mock_deps['memory_service'].add_memory.return_value = {'memory_ids': []}
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}

        # Fake LLM result — assemble_payload runs for real; only the CLI call is mocked.
        fake_cli_result = MagicMock(
            success=True,
            report={'flagged_items': [], 'summary': 'ok', 'stats': {}},
            llm_calls=1,
            tokens_used=0,
            cost_usd=0.0,
            model='test-model',
            error=None,
        )
        watermark = Watermark(project_id='reify')

        with patch('fused_memory.reconciliation.stages.base.run_stage_via_cli',
                   new=AsyncMock(return_value=fake_cli_result)):
            report = await stage.run(
                events=[], watermark=watermark, prior_reports=[], run_id='test-run'
            )

        assert report.stats.get('stale_fixc_markers_swept') == 2

    @pytest.mark.asyncio
    async def test_zero_stale_markers_stat_is_explicitly_set(self, mock_deps):
        """When no stale markers exist, stat is 0 (explicitly set, not absent)."""
        from datetime import UTC, datetime
        from types import SimpleNamespace
        from unittest.mock import AsyncMock as AM
        from unittest.mock import patch

        from fused_memory.models.reconciliation import StageId, StageReport
        from fused_memory.reconciliation.stages.base import BaseStage

        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'reify'
        stage.project_root = '/home/leo/src/reify'

        # All markers have matching run_id → zero stale
        mock_deps['memory_service'].search.return_value = [
            SimpleNamespace(
                id='current', content='current content',
                metadata={'flag_for_stage2': True, 'task_id': '1', 'run_id': 'test-run'},
            ),
        ]
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}

        base_report = StageReport(
            stage=StageId.task_knowledge_sync,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items_flagged=[],
            stats={},
            llm_calls=0,
            tokens_used=0,
        )
        watermark = Watermark(project_id='reify')

        with patch.object(BaseStage, 'run', new=AM(return_value=base_report)):
            report = await stage.run(
                events=[], watermark=watermark, prior_reports=[], run_id='test-run'
            )

        # Stat must be present and == 0, not absent
        assert 'stale_fixc_markers_swept' in report.stats
        assert report.stats['stale_fixc_markers_swept'] == 0


class TestTaskKnowledgeSyncMissingRunIdMarkersStat:
    """TaskKnowledgeSync.run() sets report.stats['stale_missing_run_id_markers'] after super().run()."""

    @pytest.fixture
    def mock_deps(self):
        from fused_memory.config.schema import ReconciliationConfig
        config = ReconciliationConfig(enabled=True, explore_codebase_root='/tmp/test')
        memory_service = AsyncMock()
        memory_service.count_memories_by_metadata.return_value = 0
        memory_service.delete_memory = AsyncMock(return_value=None)
        return {
            'memory_service': memory_service,
            'taskmaster': AsyncMock(),
            'journal': AsyncMock(),
            'config': config,
        }

    @pytest.mark.asyncio
    async def test_missing_run_id_markers_stat_set_after_run(self, mock_deps):
        """run() injects stale_missing_run_id_markers into report.stats after super().run().

        Verifies that markers with absent run_id are counted in the new stat and
        that the combined stale sweep count (missing + mismatched) is still correct.
        """
        from types import SimpleNamespace

        from fused_memory.reconciliation.stages.task_knowledge_sync import TaskKnowledgeSync

        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'reify'
        stage.project_root = '/home/leo/src/reify'

        # One current-cycle marker, one mismatched, and two with absent run_id
        mock_deps['memory_service'].search.return_value = [
            SimpleNamespace(
                id='current', content='current content',
                metadata={'flag_for_stage2': True, 'task_id': '1', 'run_id': 'test-run'},
            ),
            SimpleNamespace(
                id='mismatched', content='mismatched content',
                metadata={'flag_for_stage2': True, 'task_id': '2', 'run_id': 'old-run'},
            ),
            SimpleNamespace(
                id='missing-1', content='no run_id',
                metadata={'flag_for_stage2': True, 'task_id': '3'},
            ),
            SimpleNamespace(
                id='missing-2', content='empty run_id',
                metadata={'flag_for_stage2': True, 'task_id': '4', 'run_id': ''},
            ),
        ]
        mock_deps['memory_service'].add_memory.return_value = {'memory_ids': []}
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}

        fake_cli_result = MagicMock(
            success=True,
            report={'flagged_items': [], 'summary': 'ok', 'stats': {}},
            llm_calls=1,
            tokens_used=0,
            cost_usd=0.0,
            model='test-model',
            error=None,
        )
        watermark = Watermark(project_id='reify')

        with patch('fused_memory.reconciliation.stages.base.run_stage_via_cli',
                   new=AsyncMock(return_value=fake_cli_result)):
            report = await stage.run(
                events=[], watermark=watermark, prior_reports=[], run_id='test-run'
            )

        # 2 markers had absent/empty run_id
        assert report.stats.get('stale_missing_run_id_markers') == 2
        # Combined sweep: 1 mismatched + 2 missing = 3
        assert report.stats.get('stale_fixc_markers_swept') == 3
        # Verify delete_memory was called for each stale ID with the exact kwargs
        # that _sweep_stale_fixc_markers emits — pins the partition→sweep contract.
        # Three layered checks: `call_count == 3` rejects duplicate or extra calls;
        # the set check rejects missing/wrong IDs; the per-ID kwargs check pins
        # the full invocation contract.
        delete_memory = mock_deps['memory_service'].delete_memory
        assert delete_memory.call_count == 3
        calls_by_id = {c.kwargs['memory_id']: c.kwargs for c in delete_memory.call_args_list}
        assert set(calls_by_id) == {'missing-1', 'missing-2', 'mismatched'}
        shared_kwargs = {
            'store': 'mem0', 'project_id': 'reify',
            'causation_id': 'test-run', '_source': 'stage2_stale_fixc_sweep',
        }
        for mid, actual_kwargs in calls_by_id.items():
            assert actual_kwargs == {'memory_id': mid, **shared_kwargs}

    @pytest.mark.asyncio
    async def test_zero_missing_run_id_markers_stat_explicitly_set(self, mock_deps):
        """When all markers have matching run_id, stale_missing_run_id_markers is 0 (explicitly set)."""
        from types import SimpleNamespace

        from fused_memory.reconciliation.stages.task_knowledge_sync import TaskKnowledgeSync

        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'reify'
        stage.project_root = '/home/leo/src/reify'

        # All markers have matching run_id → zero missing
        mock_deps['memory_service'].search.return_value = [
            SimpleNamespace(
                id='current', content='current content',
                metadata={'flag_for_stage2': True, 'task_id': '1', 'run_id': 'test-run'},
            ),
        ]
        mock_deps['memory_service'].add_memory.return_value = {'memory_ids': []}
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}

        fake_cli_result = MagicMock(
            success=True,
            report={'flagged_items': [], 'summary': 'ok', 'stats': {}},
            llm_calls=1,
            tokens_used=0,
            cost_usd=0.0,
            model='test-model',
            error=None,
        )
        watermark = Watermark(project_id='reify')

        with patch('fused_memory.reconciliation.stages.base.run_stage_via_cli',
                   new=AsyncMock(return_value=fake_cli_result)):
            report = await stage.run(
                events=[], watermark=watermark, prior_reports=[], run_id='test-run'
            )

        # Stat must be present and explicitly 0, not absent
        assert 'stale_missing_run_id_markers' in report.stats
        assert report.stats['stale_missing_run_id_markers'] == 0


class TestAssemblePayloadRunWindowStart:
    """step-5 (task-1369): assemble_payload threads journal.get_run().started_at into _query_stage2_flags.

    All tests FAIL until step-6: assemble_payload currently does not call
    self.journal.get_run(), so journal.get_run.assert_awaited_* always fails.
    """

    @pytest.fixture
    def mock_deps(self):
        from fused_memory.config.schema import ReconciliationConfig
        config = ReconciliationConfig(enabled=True, explore_codebase_root='/tmp/test')
        memory_service = AsyncMock()
        memory_service.count_memories_by_metadata.return_value = 0
        memory_service.delete_memory = AsyncMock(return_value=None)
        memory_service.search.return_value = []
        memory_service.add_memory.return_value = {'memory_ids': []}
        return {
            'memory_service': memory_service,
            'taskmaster': AsyncMock(),
            'journal': AsyncMock(),
            'config': config,
        }

    def _fake_cli_result(self):
        return MagicMock(
            success=True,
            report={'flagged_items': [], 'summary': 'ok', 'stats': {}},
            llm_calls=1, tokens_used=0, cost_usd=0.0,
            model='test-model', error=None,
        )

    @pytest.mark.asyncio
    async def test_run_window_start_sourced_from_journal_started_at(self, mock_deps):
        """(a) journal.get_run(run_id).started_at is passed as run_window_start to _query_stage2_flags."""
        from fused_memory.reconciliation.stages.task_knowledge_sync import (
            Stage2FlagPartition,
            TaskKnowledgeSync,
        )

        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'reify'
        stage.project_root = '/home/leo/src/reify'
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}

        run_window_start = datetime(2026, 5, 15, 10, 0, 0, tzinfo=UTC)
        mock_run = MagicMock()
        mock_run.started_at = run_window_start
        mock_deps['journal'].get_run = AsyncMock(return_value=mock_run)

        # Capture run_window_start kwarg passed to _query_stage2_flags
        captured_kwargs: dict = {}

        async def capture_query(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return Stage2FlagPartition([], [], [], [])

        with (
            patch(
                'fused_memory.reconciliation.stages.task_knowledge_sync._query_stage2_flags',
                side_effect=capture_query,
            ),
            patch(
                'fused_memory.reconciliation.stages.base.run_stage_via_cli',
                new=AsyncMock(return_value=self._fake_cli_result()),
            ),
        ):
            await stage.run(events=[], watermark=Watermark(project_id='reify'),
                            prior_reports=[], run_id='test-run')

        # journal.get_run must be called with the current run_id
        mock_deps['journal'].get_run.assert_awaited_once_with('test-run')
        # run_window_start must equal the stub's started_at
        assert captured_kwargs.get('run_window_start') == run_window_start, (
            f"Expected run_window_start={run_window_start!r}, "
            f"got {captured_kwargs.get('run_window_start')!r}"
        )

    @pytest.mark.asyncio
    async def test_run_window_start_none_when_journal_raises(self, mock_deps):
        """(b) When journal.get_run raises, assemble_payload completes with run_window_start=None."""
        from fused_memory.reconciliation.stages.task_knowledge_sync import (
            Stage2FlagPartition,
            TaskKnowledgeSync,
        )

        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'reify'
        stage.project_root = '/home/leo/src/reify'
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}

        mock_deps['journal'].get_run = AsyncMock(side_effect=RuntimeError('journal unavailable'))

        captured_kwargs: dict = {}

        async def capture_query(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return Stage2FlagPartition([], [], [], [])

        with (
            patch(
                'fused_memory.reconciliation.stages.task_knowledge_sync._query_stage2_flags',
                side_effect=capture_query,
            ),
            patch(
                'fused_memory.reconciliation.stages.base.run_stage_via_cli',
                new=AsyncMock(return_value=self._fake_cli_result()),
            ),
        ):
            # Must not raise even though journal.get_run raises
            await stage.run(events=[], watermark=Watermark(project_id='reify'),
                            prior_reports=[], run_id='test-run')

        # journal.get_run was still attempted
        mock_deps['journal'].get_run.assert_awaited_once_with('test-run')
        # Graceful degradation: run_window_start=None
        assert captured_kwargs.get('run_window_start') is None

    @pytest.mark.asyncio
    async def test_run_window_start_none_when_started_at_not_datetime(self, mock_deps):
        """(c) When get_run() returns an object whose started_at is not a datetime, run_window_start=None."""
        from fused_memory.reconciliation.stages.task_knowledge_sync import (
            Stage2FlagPartition,
            TaskKnowledgeSync,
        )

        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'reify'
        stage.project_root = '/home/leo/src/reify'
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}

        mock_run = MagicMock()
        mock_run.started_at = 'not-a-datetime'  # not a datetime instance
        mock_deps['journal'].get_run = AsyncMock(return_value=mock_run)

        captured_kwargs: dict = {}

        async def capture_query(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return Stage2FlagPartition([], [], [], [])

        with (
            patch(
                'fused_memory.reconciliation.stages.task_knowledge_sync._query_stage2_flags',
                side_effect=capture_query,
            ),
            patch(
                'fused_memory.reconciliation.stages.base.run_stage_via_cli',
                new=AsyncMock(return_value=self._fake_cli_result()),
            ),
        ):
            await stage.run(events=[], watermark=Watermark(project_id='reify'),
                            prior_reports=[], run_id='test-run')

        mock_deps['journal'].get_run.assert_awaited_once_with('test-run')
        assert captured_kwargs.get('run_window_start') is None

    @pytest.mark.asyncio
    async def test_warns_when_journal_started_at_is_naive(self, mock_deps, caplog):
        """assemble_payload emits a WARNING when journal.get_run().started_at is a naive datetime.

        Root cause being detected: the orchestrator journal is expected to persist
        ``started_at`` via ``datetime.now(UTC)`` (always tz-aware).  A naive
        ``started_at`` indicates a journal contract violation.  Without a WARNING
        the condition masquerades as a clean cycle — same-cycle-sweep regressions
        (task-1369) that are caused by a naive ``started_at`` silently disabling
        the run-window guard become invisible in logs.

        Observability contract (this test):
          1. A WARNING-level log record is emitted whose message mentions "naive"
             and "started_at" (stable substrings of the message chosen in step-4).
          2. The guard is NOT silently disabled — ``run_window_start`` is a
             tz-aware UTC datetime (normalized at the call site per task-1383
             Amendment 2), NOT forced to None, so the guard remains active.
             ``_marker_is_within_run_window`` also normalizes as defence-in-depth
             for any direct callers that bypass assemble_payload.

        This test FAILS before step-4: assemble_payload emits no WARNING today for
        a naive ``started_at``.
        """
        import logging

        from fused_memory.reconciliation.stages.task_knowledge_sync import (
            Stage2FlagPartition,
            TaskKnowledgeSync,
        )

        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'reify'
        stage.project_root = '/home/leo/src/reify'
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}

        naive_started_at = datetime(2026, 5, 15, 10, 0, 0)  # NAIVE — no tzinfo
        mock_run = MagicMock()
        mock_run.started_at = naive_started_at
        mock_deps['journal'].get_run = AsyncMock(return_value=mock_run)

        captured_kwargs: dict = {}

        async def capture_query(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return Stage2FlagPartition([], [], [], [])

        with (
            patch(
                'fused_memory.reconciliation.stages.task_knowledge_sync._query_stage2_flags',
                side_effect=capture_query,
            ),
            patch(
                'fused_memory.reconciliation.stages.base.run_stage_via_cli',
                new=AsyncMock(return_value=self._fake_cli_result()),
            ),
            caplog.at_level(logging.WARNING),
        ):
            await stage.run(events=[], watermark=Watermark(project_id='reify'),
                            prior_reports=[], run_id='test-run')

        # (1) A WARNING must be emitted mentioning naive and started_at
        warning_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and 'naive' in r.getMessage()
            and 'started_at' in r.getMessage()
        ]
        assert warning_records, (
            'Expected a WARNING log mentioning "naive" and "started_at" when '
            'journal.get_run().started_at is a naive datetime; got records: '
            + str([r.getMessage() for r in caplog.records if r.levelno == logging.WARNING])
        )

        # (2) run_window_start must NOT be None (guard NOT disabled), and must be
        # tz-aware UTC (normalized at the call site per Amendment 2 / task-1383).
        expected_aware = naive_started_at.replace(tzinfo=UTC)
        result_rws = captured_kwargs.get('run_window_start')
        assert result_rws is not None, (
            'assemble_payload must NOT drop run_window_start to None for a naive started_at'
        )
        assert result_rws.tzinfo is not None, (
            f'assemble_payload must normalize naive started_at to UTC; got tzinfo=None: {result_rws!r}'
        )
        assert result_rws == expected_aware, (
            'assemble_payload must normalize naive started_at to tz-aware UTC; '
            f'expected {expected_aware!r}, got {result_rws!r}'
        )


class TestSameCycleSweepFix:
    """step-7 (task-1369): end-to-end reproduction of cycle-00d1e252 defect.

    Proves that the run-window guard (steps 4+6) prevents same-cycle Stage-1
    markers (missing run_id) from being swept before Stage 2 can process them,
    while genuine prior-cycle residue is still swept correctly.
    """

    @pytest.fixture
    def mock_deps(self):
        from fused_memory.config.schema import ReconciliationConfig
        config = ReconciliationConfig(enabled=True, explore_codebase_root='/tmp/test')
        memory_service = AsyncMock()
        memory_service.count_memories_by_metadata.return_value = 0
        memory_service.delete_memory = AsyncMock(return_value=None)
        memory_service.add_memory.return_value = {'memory_ids': []}
        return {
            'memory_service': memory_service,
            'taskmaster': AsyncMock(),
            'journal': AsyncMock(),
            'config': config,
        }

    def _fake_cli_result(self):
        return MagicMock(
            success=True,
            report={'flagged_items': [], 'summary': 'ok', 'stats': {}},
            llm_calls=1, tokens_used=0, cost_usd=0.0,
            model='test-model', error=None,
        )

    @pytest.mark.asyncio
    async def test_in_window_missing_run_id_not_swept_prior_cycle_is_swept(self, mock_deps):
        """Same-cycle marker (missing run_id, created_at in window) is NOT swept;
        prior-cycle residue (mismatched run_id, created_at out of window) IS swept."""
        from types import SimpleNamespace

        from fused_memory.reconciliation.stages.task_knowledge_sync import TaskKnowledgeSync

        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'reify'
        stage.project_root = '/home/leo/src/reify'
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}

        T0 = datetime(2026, 5, 15, 10, 0, 0, tzinfo=UTC)
        mock_run = MagicMock()
        mock_run.started_at = T0
        mock_deps['journal'].get_run = AsyncMock(return_value=mock_run)

        mock_deps['memory_service'].search.return_value = [
            # (iii) clean current marker — matching run_id
            SimpleNamespace(
                id='current', content='current flag',
                metadata={'flag_for_stage2': True, 'task_id': '1', 'run_id': 'test-run'},
                created_at=None,
            ),
            # (i) same-cycle marker — MISSING run_id, created_at within window
            SimpleNamespace(
                id='in-window-missing', content='same-cycle flag',
                metadata={'flag_for_stage2': True, 'task_id': '2'},
                created_at='2026-05-15T10:00:01+00:00',  # T0 + 1s
            ),
            # (ii) genuine prior-cycle marker — mismatched run_id, created_at before window
            SimpleNamespace(
                id='out-of-window-stale', content='prior-cycle flag',
                metadata={'flag_for_stage2': True, 'task_id': '3', 'run_id': 'old-run'},
                created_at='2026-05-15T09:00:00+00:00',  # T0 - 1h
            ),
        ]

        watermark = Watermark(project_id='reify')
        with patch('fused_memory.reconciliation.stages.base.run_stage_via_cli',
                   new=AsyncMock(return_value=self._fake_cli_result())):
            report = await stage.run(events=[], watermark=watermark, prior_reports=[],
                                     run_id='test-run')

        deleted_ids = {
            c.kwargs['memory_id']
            for c in mock_deps['memory_service'].delete_memory.call_args_list
        }
        # (i) same-cycle marker must NOT be swept
        assert 'in-window-missing' not in deleted_ids, (
            'Same-cycle in-window marker must NOT be swept (run-window guard fix)'
        )
        # (ii) prior-cycle residue MUST be swept
        assert 'out-of-window-stale' in deleted_ids, (
            'Prior-cycle out-of-window marker must still be swept'
        )
        # Verify the expected _source kwarg on the stale-sweep delete call
        stale_call_kwargs = {
            c.kwargs['memory_id']: c.kwargs
            for c in mock_deps['memory_service'].delete_memory.call_args_list
            if c.kwargs.get('_source') == 'stage2_stale_fixc_sweep'
        }
        assert 'out-of-window-stale' in stale_call_kwargs
        # Only 1 marker swept
        assert report.stats.get('stale_fixc_markers_swept') == 1

    @pytest.mark.asyncio
    async def test_control_journal_raises_reverts_to_sweeping_in_window_marker(self, mock_deps):
        """Control: when journal.get_run raises, window guard is dormant and the
        same-cycle marker (missing run_id) IS swept — proves guard is doing the rescue."""
        from types import SimpleNamespace

        from fused_memory.reconciliation.stages.task_knowledge_sync import TaskKnowledgeSync

        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'reify'
        stage.project_root = '/home/leo/src/reify'
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}

        # Journal raises → run_window_start=None → window guard dormant
        mock_deps['journal'].get_run = AsyncMock(side_effect=RuntimeError('journal unavailable'))

        mock_deps['memory_service'].search.return_value = [
            # Matching run_id marker — always current
            SimpleNamespace(
                id='current', content='current flag',
                metadata={'flag_for_stage2': True, 'task_id': '1', 'run_id': 'test-run'},
                created_at=None,
            ),
            # Same-cycle marker with missing run_id — WITHOUT guard, this gets swept
            SimpleNamespace(
                id='in-window-missing', content='same-cycle flag',
                metadata={'flag_for_stage2': True, 'task_id': '2'},
                created_at='2026-05-15T10:00:01+00:00',
            ),
            # Prior-cycle residue — always swept
            SimpleNamespace(
                id='out-of-window-stale', content='prior-cycle flag',
                metadata={'flag_for_stage2': True, 'task_id': '3', 'run_id': 'old-run'},
                created_at='2026-05-15T09:00:00+00:00',
            ),
        ]

        watermark = Watermark(project_id='reify')
        with patch('fused_memory.reconciliation.stages.base.run_stage_via_cli',
                   new=AsyncMock(return_value=self._fake_cli_result())):
            report = await stage.run(events=[], watermark=watermark, prior_reports=[],
                                     run_id='test-run')

        deleted_ids = {
            c.kwargs['memory_id']
            for c in mock_deps['memory_service'].delete_memory.call_args_list
        }
        # Without the guard (journal failed), the same-cycle marker reverts to being swept
        assert 'in-window-missing' in deleted_ids, (
            'Control: without run-window guard, same-cycle missing-run_id marker IS swept'
        )
        assert report.stats.get('stale_fixc_markers_swept') == 2


class TestRescuedInWindowMarkersStat:
    """TaskKnowledgeSync.run() sets report.stats['rescued_in_window_markers'] (task-1369 amendment).

    Non-zero when the run-window guard rescues same-cycle Stage-1 markers whose
    run_id was omitted or mis-stamped.  Surfaced alongside stale_missing_run_id_markers
    so operators can distinguish rescued (benign, processed) from genuinely stale
    (swept, never reached Stage 2).
    """

    @pytest.fixture
    def mock_deps(self):
        from fused_memory.config.schema import ReconciliationConfig
        config = ReconciliationConfig(enabled=True, explore_codebase_root='/tmp/test')
        memory_service = AsyncMock()
        memory_service.count_memories_by_metadata.return_value = 0
        memory_service.delete_memory = AsyncMock(return_value=None)
        memory_service.add_memory.return_value = {'memory_ids': []}
        return {
            'memory_service': memory_service,
            'taskmaster': AsyncMock(),
            'journal': AsyncMock(),
            'config': config,
        }

    def _fake_cli_result(self):
        return MagicMock(
            success=True,
            report={'flagged_items': [], 'summary': 'ok', 'stats': {}},
            llm_calls=1, tokens_used=0, cost_usd=0.0,
            model='test-model', error=None,
        )

    @pytest.mark.asyncio
    async def test_rescued_in_window_markers_count_nonzero_when_guard_fires(self, mock_deps):
        """rescued_in_window_markers equals the number of same-cycle markers the guard rescued."""
        from types import SimpleNamespace

        from fused_memory.reconciliation.stages.task_knowledge_sync import TaskKnowledgeSync

        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'reify'
        stage.project_root = '/home/leo/src/reify'
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}

        T0 = datetime(2026, 5, 15, 10, 0, 0, tzinfo=UTC)
        mock_run = MagicMock()
        mock_run.started_at = T0
        mock_deps['journal'].get_run = AsyncMock(return_value=mock_run)

        mock_deps['memory_service'].search.return_value = [
            # Clean current marker (matching run_id) — should NOT be counted as rescued
            SimpleNamespace(
                id='current', content='current flag',
                metadata={'flag_for_stage2': True, 'task_id': '1', 'run_id': 'test-run'},
                created_at=None,
            ),
            # Same-cycle marker — MISSING run_id, rescued by window guard
            SimpleNamespace(
                id='rescued-missing', content='rescued same-cycle flag missing run_id',
                metadata={'flag_for_stage2': True, 'task_id': '2'},
                created_at='2026-05-15T10:00:01+00:00',  # T0 + 1s, in-window
            ),
            # Prior-cycle residue — swept, NOT rescued
            SimpleNamespace(
                id='stale', content='prior-cycle flag',
                metadata={'flag_for_stage2': True, 'task_id': '3', 'run_id': 'old-run'},
                created_at='2026-05-15T09:00:00+00:00',  # T0 - 1h, out-of-window
            ),
        ]

        watermark = Watermark(project_id='reify')
        with patch('fused_memory.reconciliation.stages.base.run_stage_via_cli',
                   new=AsyncMock(return_value=self._fake_cli_result())):
            report = await stage.run(events=[], watermark=watermark, prior_reports=[],
                                     run_id='test-run')

        assert 'rescued_in_window_markers' in report.stats, (
            "report.stats must contain 'rescued_in_window_markers' key (explicit zero required)"
        )
        assert report.stats['rescued_in_window_markers'] == 1, (
            f"Expected 1 rescued marker, got {report.stats.get('rescued_in_window_markers')}"
        )

    @pytest.mark.asyncio
    async def test_rescued_in_window_markers_zero_when_no_rescue_fires(self, mock_deps):
        """rescued_in_window_markers is 0 (explicit) when all active markers have matching run_id."""
        from types import SimpleNamespace

        from fused_memory.reconciliation.stages.task_knowledge_sync import TaskKnowledgeSync

        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'reify'
        stage.project_root = '/home/leo/src/reify'
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}
        mock_deps['journal'].get_run = AsyncMock(side_effect=RuntimeError('journal down'))

        mock_deps['memory_service'].search.return_value = [
            SimpleNamespace(
                id='clean', content='clean current',
                metadata={'flag_for_stage2': True, 'task_id': '1', 'run_id': 'test-run'},
                created_at=None,
            ),
        ]

        watermark = Watermark(project_id='reify')
        with patch('fused_memory.reconciliation.stages.base.run_stage_via_cli',
                   new=AsyncMock(return_value=self._fake_cli_result())):
            report = await stage.run(events=[], watermark=watermark, prior_reports=[],
                                     run_id='test-run')

        assert 'rescued_in_window_markers' in report.stats
        assert report.stats['rescued_in_window_markers'] == 0, (
            'rescued_in_window_markers must be explicitly 0, not absent, when no rescue fires'
        )

    @pytest.mark.asyncio
    async def test_rescued_in_window_count_reads_partition_rescued_ids_not_rederived(self, mock_deps):
        """rescued_in_window_markers must equal len(partition.rescued_ids), not a re-derived
        predicate over active_flags metadata.

        Injects a hand-built partition where `current` contains TWO flags whose
        metadata.run_id does NOT match the run_id ('mismatch' != 'test-run'), but
        rescued_ids contains only ONE id ('rescued-1').  The old re-derivation
        (sum over active_flags whose run_id != current) would yield 2; the single-source
        contract (len(partition.rescued_ids)) must yield 1.

        This test fails against the re-derivation at assemble_payload lines 1656-1660 and
        passes only once the consumer reads partition.rescued_ids directly.
        """
        from unittest.mock import AsyncMock, patch

        from fused_memory.reconciliation.stages.task_knowledge_sync import (
            Stage2FlagPartition,
            TaskKnowledgeSync,
        )

        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'reify'
        stage.project_root = '/home/leo/src/reify'
        mock_deps['taskmaster'].get_tasks.return_value = {'tasks': []}
        mock_deps['journal'].get_run = AsyncMock(side_effect=RuntimeError('journal down'))

        # Inject a partition where current has 2 non-matching-run_id flags but rescued_ids
        # has only 1 entry — old re-derivation yields 2, single-source yields 1.
        injected_partition = Stage2FlagPartition(
            current=[
                {'id': 'rescued-1', 'content': 'flag 1', 'metadata': {'run_id': 'mismatch'}, 'task_id': '1'},
                {'id': 'not-rescued', 'content': 'flag 2', 'metadata': {'run_id': 'mismatch'}, 'task_id': '2'},
            ],
            stale_missing_run_id_ids=[],
            stale_mismatched_run_id_ids=[],
            rescued_ids=['rescued-1'],  # only 1, even though both flags have non-matching run_id
        )

        with (
            patch(
                'fused_memory.reconciliation.stages.task_knowledge_sync._query_stage2_flags',
                new=AsyncMock(return_value=injected_partition),
            ),
            patch(
                'fused_memory.reconciliation.stages.base.run_stage_via_cli',
                new=AsyncMock(return_value=self._fake_cli_result()),
            ),
        ):
            report = await stage.run(
                events=[], watermark=Watermark(project_id='reify'),
                prior_reports=[], run_id='test-run',
            )

        assert report.stats['rescued_in_window_markers'] == 1, (
            f"rescued_in_window_markers must equal len(partition.rescued_ids)==1, "
            f"not a re-derived predicate over active_flags (which would yield 2); "
            f"got: {report.stats.get('rescued_in_window_markers')}"
        )


class TestStage3PayloadIncludesProjectRoot:
    """IntegrityCheck.assemble_payload() must emit a Use project_root="..." directive.

    Mirrors Stage 2's pattern (task_knowledge_sync.py:662) so the Stage 3 CLI
    agent receives an explicit project_root binding rather than guessing.

    These tests FAIL before step-12 because Stage 3's assemble_payload currently
    never emits the directive (task 1143 step-11).
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

    @pytest.mark.asyncio
    async def test_integrity_check_payload_emits_use_project_root_directive(self, mock_deps):
        """assemble_payload() for reify must contain Use project_root="/home/leo/src/reify"."""
        stage = IntegrityCheck(StageId.integrity_check, **mock_deps)
        stage.project_id = 'reify'
        stage.project_root = '/home/leo/src/reify'

        watermark = Watermark(project_id='reify')
        payload = await stage.assemble_payload([], watermark, [])

        assert 'Use project_root="/home/leo/src/reify"' in payload, (
            f'Stage 3 payload must contain Use project_root="..." directive '
            f'mirroring Stage 2 (task 1143 step-12). Got payload:\n{payload[:500]}'
        )
        # Must not silently bleed dark-factory's path into another project's payload
        assert '/home/leo/src/dark-factory' not in payload, (
            'Stage 3 payload for reify must not contain dark-factory path'
        )

    @pytest.mark.asyncio
    async def test_integrity_check_payload_for_dark_factory_uses_dark_factory_root(
        self, mock_deps
    ):
        """assemble_payload() for dark_factory must use dark-factory root in directive."""
        stage = IntegrityCheck(StageId.integrity_check, **mock_deps)
        stage.project_id = 'dark_factory'
        stage.project_root = '/home/leo/src/dark-factory'

        watermark = Watermark(project_id='dark_factory')
        payload = await stage.assemble_payload([], watermark, [])

        assert 'Use project_root="/home/leo/src/dark-factory"' in payload, (
            f'Stage 3 payload for dark_factory must contain Use project_root="..." directive. '
            f'Got payload:\n{payload[:500]}'
        )


# ---------------------------------------------------------------------------
# Task 1154 — Stage 2 same-run Stage 1 human_operator_required suppression
# ---------------------------------------------------------------------------

class TestSuppressSameRunHumanOperatorDups:
    """Unit tests for _suppress_same_run_human_operator_dups(stage2_flagged, stage1_flagged)."""

    def test_both_empty_returns_empty_tuples(self):
        """Both empty inputs → ([], [])."""
        kept, suppressed = _suppress_same_run_human_operator_dups([], [])
        assert kept == []
        assert suppressed == []

    def test_exact_match_suppressed(self):
        """Stage 1 human_operator_required + Stage 2 same (task_id, flag_type, resolution_status) → suppressed."""
        stage1 = [
            {'task_id': '42', 'flag_type': 'assumption_invalid', 'resolution_status': 'human_operator_required', 'description': 's1 finding'},
        ]
        stage2 = [
            {'task_id': '42', 'flag_type': 'assumption_invalid', 'resolution_status': 'human_operator_required', 'description': 's2 dup'},
        ]
        kept, suppressed = _suppress_same_run_human_operator_dups(stage2, stage1)
        assert kept == []
        assert len(suppressed) == 1
        assert suppressed[0]['description'] == 's2 dup'

    def test_stage1_not_human_operator_required_keeps_stage2(self):
        """Stage 1 has same (task_id, flag_type) but resolution_status='resolved' → Stage 2 item kept."""
        stage1 = [
            {'task_id': '42', 'flag_type': 'assumption_invalid', 'resolution_status': 'resolved'},
        ]
        stage2 = [
            {'task_id': '42', 'flag_type': 'assumption_invalid', 'resolution_status': 'human_operator_required', 'description': 'stage2 unique'},
        ]
        kept, suppressed = _suppress_same_run_human_operator_dups(stage2, stage1)
        assert len(kept) == 1
        assert suppressed == []

    def test_stage2_different_resolution_status_kept(self):
        """Stage 1 is human_operator_required but Stage 2 item has different resolution_status → Stage 2 kept."""
        stage1 = [
            {'task_id': '42', 'flag_type': 'assumption_invalid', 'resolution_status': 'human_operator_required'},
        ]
        stage2 = [
            {'task_id': '42', 'flag_type': 'assumption_invalid', 'resolution_status': 'in_progress', 'description': 'different status'},
        ]
        kept, suppressed = _suppress_same_run_human_operator_dups(stage2, stage1)
        assert len(kept) == 1
        assert kept[0]['resolution_status'] == 'in_progress'
        assert suppressed == []

    def test_different_task_id_kept(self):
        """Stage 1 human_operator_required for task 42; Stage 2 item has different task_id → kept."""
        stage1 = [
            {'task_id': '42', 'flag_type': 'assumption_invalid', 'resolution_status': 'human_operator_required'},
        ]
        stage2 = [
            {'task_id': '99', 'flag_type': 'assumption_invalid', 'resolution_status': 'human_operator_required'},
        ]
        kept, suppressed = _suppress_same_run_human_operator_dups(stage2, stage1)
        assert len(kept) == 1
        assert suppressed == []

    def test_different_flag_type_kept(self):
        """Stage 1 human_operator_required for flag_type X; Stage 2 item has different flag_type → kept."""
        stage1 = [
            {'task_id': '42', 'flag_type': 'assumption_invalid', 'resolution_status': 'human_operator_required'},
        ]
        stage2 = [
            {'task_id': '42', 'flag_type': 'stale_dependency', 'resolution_status': 'human_operator_required'},
        ]
        kept, suppressed = _suppress_same_run_human_operator_dups(stage2, stage1)
        assert len(kept) == 1
        assert suppressed == []

    def test_stage1_missing_task_id_forms_no_key_all_stage2_kept(self):
        """Stage 1 entry missing task_id → no key formed → all Stage 2 items kept."""
        stage1 = [
            {'flag_type': 'assumption_invalid', 'resolution_status': 'human_operator_required'},
        ]
        stage2 = [
            {'task_id': '42', 'flag_type': 'assumption_invalid', 'resolution_status': 'human_operator_required'},
        ]
        kept, suppressed = _suppress_same_run_human_operator_dups(stage2, stage1)
        assert len(kept) == 1
        assert suppressed == []

    def test_stage1_missing_flag_type_forms_no_key_all_stage2_kept(self):
        """Stage 1 entry missing flag_type → no key formed → all Stage 2 items kept."""
        stage1 = [
            {'task_id': '42', 'resolution_status': 'human_operator_required'},
        ]
        stage2 = [
            {'task_id': '42', 'flag_type': 'assumption_invalid', 'resolution_status': 'human_operator_required'},
        ]
        kept, suppressed = _suppress_same_run_human_operator_dups(stage2, stage1)
        assert len(kept) == 1
        assert suppressed == []

    def test_integer_vs_string_task_id_coercion_suppresses(self):
        """Stage 1 emits int task_id=42; Stage 2 emits string '42' → suppressed via str() coercion."""
        stage1 = [
            {'task_id': 42, 'flag_type': 'assumption_invalid', 'resolution_status': 'human_operator_required'},
        ]
        stage2 = [
            {'task_id': '42', 'flag_type': 'assumption_invalid', 'resolution_status': 'human_operator_required'},
        ]
        kept, suppressed = _suppress_same_run_human_operator_dups(stage2, stage1)
        assert kept == []
        assert len(suppressed) == 1

    def test_mixed_stage2_items_exact_split(self):
        """Multiple Stage 2 items: duplicate + non-duplicate → exact split in (kept, suppressed)."""
        stage1 = [
            {'task_id': '42', 'flag_type': 'assumption_invalid', 'resolution_status': 'human_operator_required'},
            {'task_id': '77', 'flag_type': 'stale_dependency', 'resolution_status': 'human_operator_required'},
        ]
        stage2 = [
            # duplicate — should be suppressed
            {'task_id': '42', 'flag_type': 'assumption_invalid', 'resolution_status': 'human_operator_required', 'description': 'dup'},
            # different task_id — should be kept
            {'task_id': '55', 'flag_type': 'assumption_invalid', 'resolution_status': 'human_operator_required', 'description': 'unique task'},
            # different flag_type for stage1 task 77 — should be kept
            {'task_id': '77', 'flag_type': 'other_flag', 'resolution_status': 'human_operator_required', 'description': 'unique flag_type'},
            # exact dup for task 77 — should be suppressed
            {'task_id': '77', 'flag_type': 'stale_dependency', 'resolution_status': 'human_operator_required', 'description': 'dup2'},
        ]
        kept, suppressed = _suppress_same_run_human_operator_dups(stage2, stage1)
        assert len(kept) == 2
        assert len(suppressed) == 2
        kept_descs = {item['description'] for item in kept}
        suppressed_descs = {item['description'] for item in suppressed}
        assert kept_descs == {'unique task', 'unique flag_type'}
        assert suppressed_descs == {'dup', 'dup2'}


class TestTaskKnowledgeSyncSuppressesStage1HumanOperatorDups:
    """Integration tests: TaskKnowledgeSync.run() applies the Stage 1 dup suppression post-processor."""

    @pytest.fixture
    def mock_deps(self):
        config = ReconciliationConfig(enabled=True, explore_codebase_root='/tmp/test')
        return {
            'memory_service': AsyncMock(),
            'taskmaster': AsyncMock(),
            'journal': AsyncMock(),
            'config': config,
        }

    def _make_cli_result(self, flagged_items: list[dict]) -> MagicMock:
        """Return a fake StageResult-like object for patching run_stage_via_cli."""
        return MagicMock(
            success=True,
            report={'flagged_items': flagged_items, 'summary': 'ok'},
            llm_calls=1,
            tokens_used=0,
            cost_usd=0.0,
            model='m',
        )

    @pytest.mark.asyncio
    async def test_run_suppresses_stage1_dup_and_keeps_unique(self, mock_deps, caplog):
        """run() drops Stage 2 items that duplicate Stage 1 human_operator_required flags."""
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'dark_factory'

        _now = datetime.now(tz=UTC)
        stage1_report = StageReport(
            stage=StageId.memory_consolidator,
            started_at=_now,
            completed_at=_now,
            items_flagged=[
                {
                    'task_id': '99',
                    'flag_type': 'assumption_invalid',
                    'resolution_status': 'human_operator_required',
                    'description': 's1 finding',
                }
            ],
        )
        stage2_flagged = [
            # duplicate — same (task_id, flag_type, resolution_status)
            {
                'task_id': '99',
                'flag_type': 'assumption_invalid',
                'resolution_status': 'human_operator_required',
                'description': 's2 dup',
            },
            # unique — different task_id
            {
                'task_id': '888',
                'flag_type': 'assumption_invalid',
                'resolution_status': 'human_operator_required',
                'description': 's2 unique',
            },
        ]

        with (
            patch.object(stage, 'assemble_payload', new=AsyncMock(return_value='payload')),
            patch(
                'fused_memory.reconciliation.stages.base.run_stage_via_cli',
                new=AsyncMock(return_value=self._make_cli_result(stage2_flagged)),
            ),
            caplog.at_level(
                logging.INFO,
                logger='fused_memory.reconciliation.stages.task_knowledge_sync',
            ),
        ):
            report = await stage.run(
                events=[],
                watermark=Watermark(project_id='dark_factory'),
                prior_reports=[stage1_report],
                run_id='test-run-1154-a',
            )

        # Only the unique item should survive
        assert len(report.items_flagged) == 1
        assert report.items_flagged[0]['description'] == 's2 unique'

        # Key-based assertion: the suppressed (task_id, flag_type) must be absent
        # (suggestion 4 — description-only assertion would miss a wrong-item drop)
        assert not any(
            (it['task_id'], it['flag_type']) == ('99', 'assumption_invalid')
            and it.get('resolution_status') == 'human_operator_required'
            for it in report.items_flagged
        )

        # suppressed_count recorded in stats (suggestion 2)
        assert report.stats.get('stage2_stage1_dups_suppressed') == 1

        # INFO log with suppressed_count should have fired
        target_logger = 'fused_memory.reconciliation.stages.task_knowledge_sync'
        suppression_logs = [
            r for r in caplog.records
            if r.name == target_logger
            and r.levelno == logging.INFO
            and 'stage2_suppressed_stage1_dup_flags' in r.getMessage()
        ]
        assert len(suppression_logs) == 1, (
            f'expected one stage2_suppressed_stage1_dup_flags INFO log, got {len(suppression_logs)}'
        )
        rec = suppression_logs[0]
        assert getattr(rec, 'suppressed_count', None) == 1
        assert getattr(rec, 'run_id', None) == 'test-run-1154-a'

    @pytest.mark.asyncio
    async def test_run_empty_prior_reports_no_op(self, mock_deps, caplog):
        """Empty prior_reports → no suppression, no log, items_flagged unchanged."""
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'dark_factory'

        stage2_flagged = [
            {'task_id': '99', 'flag_type': 'assumption_invalid', 'resolution_status': 'human_operator_required', 'description': 'item'},
        ]

        with (
            patch.object(stage, 'assemble_payload', new=AsyncMock(return_value='payload')),
            patch(
                'fused_memory.reconciliation.stages.base.run_stage_via_cli',
                new=AsyncMock(return_value=self._make_cli_result(stage2_flagged)),
            ),
            caplog.at_level(
                logging.INFO,
                logger='fused_memory.reconciliation.stages.task_knowledge_sync',
            ),
        ):
            report = await stage.run(
                events=[],
                watermark=Watermark(project_id='dark_factory'),
                prior_reports=[],
                run_id='test-run-1154-b',
            )

        # All items kept unchanged
        assert len(report.items_flagged) == 1
        assert 'stage2_stage1_dups_suppressed' not in report.stats

        # No suppression log
        target_logger = 'fused_memory.reconciliation.stages.task_knowledge_sync'
        suppression_logs = [
            r for r in caplog.records
            if r.name == target_logger and 'stage2_suppressed_stage1_dup_flags' in r.getMessage()
        ]
        assert suppression_logs == []

    @pytest.mark.asyncio
    async def test_run_empty_stage1_flags_no_op(self, mock_deps, caplog):
        """prior_reports[0].items_flagged empty → no suppression, no log."""
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'dark_factory'

        _now = datetime.now(tz=UTC)
        stage1_report = StageReport(
            stage=StageId.memory_consolidator,
            started_at=_now,
            completed_at=_now,
            items_flagged=[],
        )
        stage2_flagged = [
            {'task_id': '99', 'flag_type': 'assumption_invalid', 'resolution_status': 'human_operator_required', 'description': 'item'},
        ]

        with (
            patch.object(stage, 'assemble_payload', new=AsyncMock(return_value='payload')),
            patch(
                'fused_memory.reconciliation.stages.base.run_stage_via_cli',
                new=AsyncMock(return_value=self._make_cli_result(stage2_flagged)),
            ),
            caplog.at_level(
                logging.INFO,
                logger='fused_memory.reconciliation.stages.task_knowledge_sync',
            ),
        ):
            report = await stage.run(
                events=[],
                watermark=Watermark(project_id='dark_factory'),
                prior_reports=[stage1_report],
                run_id='test-run-1154-c',
            )

        # All items kept unchanged
        assert len(report.items_flagged) == 1
        assert 'stage2_stage1_dups_suppressed' not in report.stats

        # No suppression log
        target_logger = 'fused_memory.reconciliation.stages.task_knowledge_sync'
        suppression_logs = [
            r for r in caplog.records
            if r.name == target_logger and 'stage2_suppressed_stage1_dup_flags' in r.getMessage()
        ]
        assert suppression_logs == []

    @pytest.mark.asyncio
    async def test_run_no_duplicates_no_suppression_log(self, mock_deps, caplog):
        """Stage 2 emits non-duplicate items only → all kept, no suppression log fired."""
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'dark_factory'

        _now = datetime.now(tz=UTC)
        stage1_report = StageReport(
            stage=StageId.memory_consolidator,
            started_at=_now,
            completed_at=_now,
            items_flagged=[
                {'task_id': '42', 'flag_type': 'assumption_invalid', 'resolution_status': 'human_operator_required'},
            ],
        )
        stage2_flagged = [
            # different task_id — not a dup
            {'task_id': '99', 'flag_type': 'assumption_invalid', 'resolution_status': 'human_operator_required', 'description': 'unique'},
        ]

        with (
            patch.object(stage, 'assemble_payload', new=AsyncMock(return_value='payload')),
            patch(
                'fused_memory.reconciliation.stages.base.run_stage_via_cli',
                new=AsyncMock(return_value=self._make_cli_result(stage2_flagged)),
            ),
            caplog.at_level(
                logging.INFO,
                logger='fused_memory.reconciliation.stages.task_knowledge_sync',
            ),
        ):
            report = await stage.run(
                events=[],
                watermark=Watermark(project_id='dark_factory'),
                prior_reports=[stage1_report],
                run_id='test-run-1154-d',
            )

        # All items kept
        assert len(report.items_flagged) == 1
        assert 'stage2_stage1_dups_suppressed' not in report.stats

        # No suppression log because nothing was suppressed
        target_logger = 'fused_memory.reconciliation.stages.task_knowledge_sync'
        suppression_logs = [
            r for r in caplog.records
            if r.name == target_logger and 'stage2_suppressed_stage1_dup_flags' in r.getMessage()
        ]
        assert suppression_logs == []

    @pytest.mark.asyncio
    async def test_run_prior_reports_first_stage_not_memory_consolidator_no_op(self, mock_deps, caplog):
        """prior_reports[0].stage != memory_consolidator → guard fires, no suppression even if items would match."""
        stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps)
        stage.project_id = 'dark_factory'

        _now = datetime.now(tz=UTC)
        # Use StageId.task_knowledge_sync (a real-but-wrong stage) so the
        # prior_reports[0].stage guard fires before any dedup logic runs.
        wrong_stage_report = StageReport(
            stage=StageId.task_knowledge_sync,
            started_at=_now,
            completed_at=_now,
            items_flagged=[
                {
                    'task_id': '99',
                    'flag_type': 'assumption_invalid',
                    'resolution_status': 'human_operator_required',
                }
            ],
        )
        # Shape the Stage 2 item to exactly match the prior-report entry so that
        # suppression *would* fire if the guard were removed — this makes the test
        # non-trivial: only the guard prevents suppression here.
        stage2_flagged = [
            {
                'task_id': '99',
                'flag_type': 'assumption_invalid',
                'resolution_status': 'human_operator_required',
                'description': 'should-not-suppress',
            }
        ]

        with (
            patch.object(stage, 'assemble_payload', new=AsyncMock(return_value='payload')),
            patch(
                'fused_memory.reconciliation.stages.base.run_stage_via_cli',
                new=AsyncMock(return_value=self._make_cli_result(stage2_flagged)),
            ),
            caplog.at_level(
                logging.INFO,
                logger='fused_memory.reconciliation.stages.task_knowledge_sync',
            ),
        ):
            report = await stage.run(
                events=[],
                watermark=Watermark(project_id='dark_factory'),
                prior_reports=[wrong_stage_report],
                run_id='test-run-1168-e',
            )

        # Item must be kept — the guard skips suppression for the wrong stage
        assert len(report.items_flagged) == 1
        assert 'stage2_stage1_dups_suppressed' not in report.stats

        # No suppression log
        target_logger = 'fused_memory.reconciliation.stages.task_knowledge_sync'
        suppression_logs = [
            r for r in caplog.records
            if r.name == target_logger and 'stage2_suppressed_stage1_dup_flags' in r.getMessage()
        ]
        assert suppression_logs == []


# ── Task-1137: Stage 2 post-flight guards ────────────────────────────────────


def _make_stage2_guard_cli_result(flagged_items: list[dict], stats: dict | None = None) -> MagicMock:
    """Return a fake StageResult-like object for patching run_stage_via_cli."""
    report = {'flagged_items': flagged_items, 'summary': 'ok'}
    if stats:
        report['stats'] = stats
    return MagicMock(
        success=True,
        report=report,
        llm_calls=1,
        tokens_used=0,
        cost_usd=0.0,
        model='m',
    )


@pytest.fixture
def stage2_guard_mock_deps():
    config = ReconciliationConfig(enabled=True, explore_codebase_root='/tmp/test')
    write_journal_mock = MagicMock()
    write_journal_mock.get_ops_by_causation = AsyncMock(return_value=[])
    journal_mock = MagicMock()
    journal_mock.write_journal = write_journal_mock
    return {
        'memory_service': AsyncMock(),
        'taskmaster': AsyncMock(),
        'journal': journal_mock,
        'config': config,
    }


class TestTaskKnowledgeSyncStage2Guards:
    """Parent namespace for the four Stage 2 post-flight guard tests."""

    class TestResolveLiveStatus:
        """Unit tests for the _resolve_live_status shared helper."""

        def _make_op(
            self,
            *,
            op_id: str = 'op-1',
            agent_id: str = 'recon-stage-task_knowledge_sync',
            operation: str = 'update_task',
            params: dict | str | None = None,
        ) -> dict:
            """Return a minimal write_journal op dict for _resolve_live_status testing."""
            if params is None:
                params = {'task_id': '42'}
            params_str = json.dumps(params) if isinstance(params, dict) else params
            return {
                'id': op_id,
                'agent_id': agent_id,
                'operation': operation,
                'params': params_str,
                'layer': 'write_op',
                'causation_id': 'run-test',
                'created_at': '2026-01-01T00:00:00',
            }

        @pytest.mark.asyncio
        async def test_returns_tuple_for_update_task_with_cache_hit(self):
            """update_task op with cache hit -> ('42', 'done') tuple, get_task not called."""
            taskmaster = AsyncMock()
            op = self._make_op(operation='update_task', params={'task_id': '42'})
            result = await _resolve_live_status(
                op, taskmaster, '/project', {'42': 'done'},
                '_classify_terminal_state_violations',
            )
            assert result == ('42', 'done')
            taskmaster.get_task.assert_not_called()

        @pytest.mark.asyncio
        async def test_returns_tuple_for_set_task_status_with_cache_hit(self):
            """set_task_status op reads task_id from params (not metadata)."""
            taskmaster = AsyncMock()
            op = self._make_op(
                operation='set_task_status',
                params={'task_id': '7', 'status': 'done'},
            )
            result = await _resolve_live_status(
                op, taskmaster, '/project', {'7': 'pending'},
                '_verify_set_task_status_post_action',
            )
            assert result == ('7', 'pending')
            taskmaster.get_task.assert_not_called()

        @pytest.mark.asyncio
        async def test_returns_tuple_for_add_memory_reads_metadata_task_id(self):
            """add_memory op reads task_id from params['metadata']['task_id']."""
            taskmaster = AsyncMock()
            op = self._make_op(
                operation='add_memory',
                params={'metadata': {'task_id': '11', 'snapshot_status': 'in-progress'}},
            )
            result = await _resolve_live_status(
                op, taskmaster, '/project', {'11': 'done'},
                '_check_stall_guard_freshness',
            )
            assert result == ('11', 'done')
            taskmaster.get_task.assert_not_called()

        @pytest.mark.asyncio
        async def test_returns_tuple_with_fallback_get_task(self):
            """status_cache=None -> fallback to taskmaster.get_task."""
            taskmaster = AsyncMock()
            taskmaster.get_task.return_value = {'status': 'in-progress'}
            op = self._make_op(operation='update_task', params={'task_id': '42'})
            result = await _resolve_live_status(
                op, taskmaster, '/project', None,
                '_classify_terminal_state_violations',
            )
            assert result == ('42', 'in-progress')
            taskmaster.get_task.assert_called_once_with('42', '/project')

        @pytest.mark.asyncio
        async def test_returns_unknown_when_fallback_returns_non_dict(self):
            """Non-dict get_task result in fallback mode -> live_status='unknown'."""
            taskmaster = AsyncMock()
            taskmaster.get_task.return_value = 'oops'
            op = self._make_op(operation='update_task', params={'task_id': '42'})
            result = await _resolve_live_status(
                op, taskmaster, '/project', None,
                '_classify_terminal_state_violations',
            )
            assert result == ('42', 'unknown')

        @pytest.mark.asyncio
        async def test_returns_unknown_when_fallback_returns_dict_without_status(self):
            """Dict get_task result with no 'status' key -> _extract_status fallback -> 'unknown'."""
            taskmaster = AsyncMock()
            taskmaster.get_task.return_value = {'id': '42'}  # dict, but no 'status' key
            op = self._make_op(operation='update_task', params={'task_id': '42'})
            result = await _resolve_live_status(
                op, taskmaster, '/project', None,
                '_classify_terminal_state_violations',
            )
            assert result == ('42', 'unknown')
            taskmaster.get_task.assert_called_once_with('42', '/project')

        @pytest.mark.asyncio
        async def test_returns_none_on_malformed_params_json(self, caplog):
            """Malformed params JSON -> returns None and emits WARNING containing op_name."""
            taskmaster = AsyncMock()
            op = self._make_op(operation='update_task')
            op['params'] = 'not json'
            with caplog.at_level(
                logging.WARNING,
                logger='fused_memory.reconciliation.stages.task_knowledge_sync',
            ):
                result = await _resolve_live_status(
                    op, taskmaster, '/project', None,
                    '_classify_terminal_state_violations',
                )
            assert result is None
            assert '_classify_terminal_state_violations' in caplog.text

        @pytest.mark.asyncio
        async def test_returns_none_on_missing_task_id_for_update_task(self):
            """update_task op with params={} (no task_id) -> returns None."""
            taskmaster = AsyncMock()
            op = self._make_op(operation='update_task', params={})
            result = await _resolve_live_status(
                op, taskmaster, '/project', None,
                '_classify_terminal_state_violations',
            )
            assert result is None
            taskmaster.get_task.assert_not_called()

        @pytest.mark.asyncio
        async def test_returns_none_on_missing_task_id_for_add_memory(self):
            """add_memory op with metadata={} (no task_id) -> returns None."""
            taskmaster = AsyncMock()
            op = self._make_op(operation='add_memory', params={'metadata': {}})
            result = await _resolve_live_status(
                op, taskmaster, '/project', None,
                '_check_stall_guard_freshness',
            )
            assert result is None
            taskmaster.get_task.assert_not_called()

        @pytest.mark.asyncio
        async def test_returns_none_on_add_memory_with_non_dict_metadata(self):
            """add_memory op with metadata='oops' (non-dict) -> returns None."""
            taskmaster = AsyncMock()
            op = self._make_op(operation='add_memory', params={'metadata': 'oops'})
            result = await _resolve_live_status(
                op, taskmaster, '/project', None,
                '_check_stall_guard_freshness',
            )
            assert result is None
            taskmaster.get_task.assert_not_called()

        @pytest.mark.asyncio
        async def test_returns_none_on_cache_miss(self):
            """Cache provided but task_id not in cache -> returns None, get_task NOT called."""
            taskmaster = AsyncMock()
            op = self._make_op(operation='update_task', params={'task_id': '42'})
            result = await _resolve_live_status(
                op, taskmaster, '/project', {},
                '_classify_terminal_state_violations',
            )
            assert result is None
            taskmaster.get_task.assert_not_called()

        @pytest.mark.asyncio
        async def test_returns_none_on_fallback_get_task_exception(self, caplog):
            """Fallback get_task raises -> returns None and emits WARNING."""
            taskmaster = AsyncMock()
            taskmaster.get_task.side_effect = RuntimeError('boom')
            op = self._make_op(operation='update_task', params={'task_id': '42'})
            with caplog.at_level(
                logging.WARNING,
                logger='fused_memory.reconciliation.stages.task_knowledge_sync',
            ):
                result = await _resolve_live_status(
                    op, taskmaster, '/project', None,
                    '_classify_terminal_state_violations',
                )
            assert result is None
            assert '_classify_terminal_state_violations' in caplog.text

        @pytest.mark.asyncio
        async def test_parsed_params_bypasses_json_parse(self, caplog):
            """_parsed_params={'task_id': '42'} skips json.loads(op['params']) entirely — bypass contract from Stage 2 amend (task 1177)."""
            taskmaster = AsyncMock()
            op = self._make_op(operation='update_task')
            op['params'] = 'not valid json'
            with caplog.at_level(
                logging.WARNING,
                logger='fused_memory.reconciliation.stages.task_knowledge_sync',
            ):
                result = await _resolve_live_status(
                    op, taskmaster, '/project', {'42': 'done'},
                    '_test_parsed_params_bypass',
                    _parsed_params={'task_id': '42'},
                )
            assert result == ('42', 'done')
            taskmaster.get_task.assert_not_called()
            assert 'failed to parse params JSON' not in caplog.text

    class TestTerminalStatePreCheck:
        """Unit tests for _classify_terminal_state_violations helper."""

        def _make_op(
            self,
            *,
            op_id: str = 'op-1',
            agent_id: str = 'recon-stage-task_knowledge_sync',
            operation: str = 'update_task',
            params: dict | None = None,
        ) -> dict:
            """Return a minimal write_journal op dict for testing."""
            return {
                'id': op_id,
                'agent_id': agent_id,
                'operation': operation,
                'params': json.dumps(params or {'task_id': '42'}),
                'layer': 'write_op',
                'causation_id': 'run-test',
                'created_at': '2026-01-01T00:00:00',
            }

        @pytest.mark.asyncio
        async def test_unit_terminal_done_status_detected(self):
            """update_task op for stage-2 agent on a done task -> one violation."""
            taskmaster = AsyncMock()
            taskmaster.get_task.return_value = {'status': 'done'}

            ops = [self._make_op(op_id='op-42', params={'task_id': '42'})]
            violations = await _classify_terminal_state_violations(
                ops, taskmaster, '/project', 'recon-stage-task_knowledge_sync'
            )

            assert len(violations) == 1
            v = violations[0]
            assert v['op_id'] == 'op-42'
            assert v['task_id'] == '42'
            assert v['live_status'] == 'done'
            assert v['reason'] == 'not_applicable'

        @pytest.mark.asyncio
        async def test_unit_task_interceptor_op_excluded(self):
            """update_task op from task-interceptor agent is NOT classified as a violation."""
            taskmaster = AsyncMock()
            taskmaster.get_task.return_value = {'status': 'done'}

            ops = [
                self._make_op(
                    op_id='op-interceptor',
                    agent_id='task-interceptor',
                    params={'task_id': '42'},
                )
            ]
            violations = await _classify_terminal_state_violations(
                ops, taskmaster, '/project', 'recon-stage-task_knowledge_sync'
            )

            assert violations == []
            taskmaster.get_task.assert_not_called()

        @pytest.mark.asyncio
        async def test_unit_non_terminal_status_no_violation(self):
            """update_task op for stage-2 agent on an in-progress task -> no violation."""
            taskmaster = AsyncMock()
            taskmaster.get_task.return_value = {'status': 'in-progress'}

            ops = [self._make_op(params={'task_id': '7'})]
            violations = await _classify_terminal_state_violations(
                ops, taskmaster, '/project', 'recon-stage-task_knowledge_sync'
            )

            assert violations == []

    class TestTerminalStatePreCheckIntegration:
        """Integration tests: TaskKnowledgeSync.run() applies terminal-state pre-check guard."""

        @pytest.mark.asyncio
        async def test_run_applies_terminal_state_guard(self, stage2_guard_mock_deps, caplog):
            """run() decrements tasks_modified and adds not_applicable_count for terminal violations."""
            stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **stage2_guard_mock_deps)
            stage.project_id = 'dark_factory'
            stage.project_root = '/project'

            # Synthetic op: stage-2 agent updated task 42 which is now done
            terminal_op = {
                'id': 'op-term-1',
                'agent_id': 'recon-stage-task_knowledge_sync',
                'operation': 'update_task',
                'params': json.dumps({'task_id': '42'}),
                'layer': 'write_op',
                'causation_id': 'test-run-1137-a',
                'created_at': '2026-01-01T00:00:00',
            }
            stage2_guard_mock_deps['journal'].write_journal.get_ops_by_causation.return_value = [terminal_op]

            # taskmaster.get_task returns done status for task 42
            stage2_guard_mock_deps['taskmaster'].get_task.return_value = {'status': 'done'}

            with (
                patch.object(stage, 'assemble_payload', new=AsyncMock(return_value='payload')),
                patch(
                    'fused_memory.reconciliation.stages.base.run_stage_via_cli',
                    new=AsyncMock(return_value=_make_stage2_guard_cli_result(
                        [], stats={'tasks_modified': 3}
                    )),
                ),
                caplog.at_level(
                    logging.INFO,
                    logger='fused_memory.reconciliation.stages.task_knowledge_sync',
                ),
            ):
                report = await stage.run(
                    events=[],
                    watermark=Watermark(project_id='dark_factory'),
                    prior_reports=[],
                    run_id='test-run-1137-a',
                )

            # Guard 1: not_applicable_count incremented
            assert report.stats.get('not_applicable_count') == 1

            # Guard 1: tasks_modified decremented from 3 to 2
            assert report.stats.get('tasks_modified') == 2

            # Guard 1: one INFO log reconciliation.skipped_done_task
            target_logger = 'fused_memory.reconciliation.stages.task_knowledge_sync'
            guard_logs = [
                r for r in caplog.records
                if r.name == target_logger
                and r.levelno == logging.INFO
                and 'skipped_done_task' in r.getMessage()
            ]
            assert len(guard_logs) == 1, (
                f'expected one skipped_done_task INFO log, got {len(guard_logs)}'
            )
            rec = guard_logs[0]
            assert getattr(rec, 'task_id', None) == '42'
            assert getattr(rec, 'reason', None) == 'not_applicable'

    class TestSetTaskStatusPostActionVerification:
        """Unit + integration tests for _verify_set_task_status_post_action helper."""

        def _make_op(
            self,
            *,
            op_id: str = 'op-sts-1',
            agent_id: str = 'recon-stage-task_knowledge_sync',
            operation: str = 'set_task_status',
            params: dict | None = None,
        ) -> dict:
            """Return a minimal write_journal op dict for set_task_status testing."""
            return {
                'id': op_id,
                'agent_id': agent_id,
                'operation': operation,
                'params': json.dumps(params or {'task_id': '7', 'status': 'done'}),
                'layer': 'write_op',
                'causation_id': 'run-test',
                'created_at': '2026-01-01T00:00:00',
            }

        @pytest.mark.asyncio
        async def test_unit_live_mismatch_returns_violation(self):
            """set_task_status op targeting done but live is pending -> one mismatch."""
            taskmaster = AsyncMock()
            taskmaster.get_task.return_value = {'status': 'pending'}

            ops = [self._make_op(op_id='op-sts-mismatch', params={'task_id': '7', 'status': 'done'})]
            mismatches = await _verify_set_task_status_post_action(
                ops, taskmaster, '/project', 'recon-stage-task_knowledge_sync'
            )

            assert len(mismatches) == 1
            m = mismatches[0]
            assert m['op_id'] == 'op-sts-mismatch'
            assert m['task_id'] == '7'
            assert m['target_status'] == 'done'
            assert m['live_status'] == 'pending'

        @pytest.mark.asyncio
        async def test_unit_live_matches_target_no_mismatch(self):
            """set_task_status op where live status equals target -> no mismatch."""
            taskmaster = AsyncMock()
            taskmaster.get_task.return_value = {'status': 'done'}

            ops = [self._make_op(params={'task_id': '7', 'status': 'done'})]
            mismatches = await _verify_set_task_status_post_action(
                ops, taskmaster, '/project', 'recon-stage-task_knowledge_sync'
            )

            assert mismatches == []

        @pytest.mark.asyncio
        async def test_run_records_set_task_status_mismatch(self, stage2_guard_mock_deps, caplog):
            """run() decrements tasks_modified and adds set_task_status_post_action_mismatches."""
            stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **stage2_guard_mock_deps)
            stage.project_id = 'dark_factory'
            stage.project_root = '/project'

            # Synthetic op: stage-2 agent called set_task_status(done) for task 7
            # but live status is pending (the interceptor rejected it silently)
            sts_op = {
                'id': 'op-sts-integration-1',
                'agent_id': 'recon-stage-task_knowledge_sync',
                'operation': 'set_task_status',
                'params': json.dumps({'task_id': '7', 'status': 'done'}),
                'layer': 'write_op',
                'causation_id': 'test-run-1137-b',
                'created_at': '2026-01-01T00:00:00',
            }
            stage2_guard_mock_deps['journal'].write_journal.get_ops_by_causation.return_value = [sts_op]

            # live status is pending (transition did not stick)
            stage2_guard_mock_deps['taskmaster'].get_task.return_value = {'status': 'pending'}

            with (
                patch.object(stage, 'assemble_payload', new=AsyncMock(return_value='payload')),
                patch(
                    'fused_memory.reconciliation.stages.base.run_stage_via_cli',
                    new=AsyncMock(return_value=_make_stage2_guard_cli_result(
                        [], stats={'tasks_modified': 5}
                    )),
                ),
                caplog.at_level(
                    logging.WARNING,
                    logger='fused_memory.reconciliation.stages.task_knowledge_sync',
                ),
            ):
                report = await stage.run(
                    events=[],
                    watermark=Watermark(project_id='dark_factory'),
                    prior_reports=[],
                    run_id='test-run-1137-b',
                )

            # Guard 3: mismatch counter incremented
            assert report.stats.get('set_task_status_post_action_mismatches') == 1

            # Guard 3: tasks_modified decremented from 5 to 4
            assert report.stats.get('tasks_modified') == 4

            # Guard 3: WARNING log
            target_logger = 'fused_memory.reconciliation.stages.task_knowledge_sync'
            guard_logs = [
                r for r in caplog.records
                if r.name == target_logger
                and r.levelno == logging.WARNING
                and 'set_task_status_post_action_mismatch' in r.getMessage()
            ]
            assert len(guard_logs) == 1, (
                f'expected one set_task_status_post_action_mismatch WARNING log, got {len(guard_logs)}'
            )
            rec = guard_logs[0]
            assert getattr(rec, 'task_id', None) == '7'
            assert getattr(rec, 'target_status', None) == 'done'
            assert getattr(rec, 'live_status', None) == 'pending'

    class TestStallGuardFreshnessGate:
        """Unit + integration tests for _check_stall_guard_freshness helper."""

        def _make_add_memory_op(
            self,
            *,
            op_id: str = 'op-mem-1',
            agent_id: str = 'recon-stage-task_knowledge_sync',
            metadata: dict | None = None,
        ) -> dict:
            """Return a minimal write_journal add_memory op dict."""
            if metadata is None:
                metadata = {'task_id': '11', 'snapshot_status': 'in-progress'}
            return {
                'id': op_id,
                'agent_id': agent_id,
                'operation': 'add_memory',
                'params': json.dumps({'metadata': metadata}),
                'layer': 'write_op',
                'causation_id': 'run-test',
                'created_at': '2026-01-01T00:00:00',
            }

        @pytest.mark.asyncio
        async def test_unit_freshness_violation_snapshot_status(self):
            """add_memory op with snapshot_status='in-progress', live='done' -> one violation."""
            taskmaster = AsyncMock()
            taskmaster.get_task.return_value = {'status': 'done'}

            ops = [self._make_add_memory_op(
                op_id='op-stall-1',
                metadata={'task_id': '11', 'snapshot_status': 'in-progress'},
            )]
            violations = await _check_stall_guard_freshness(ops, taskmaster, '/project', 'recon-stage-task_knowledge_sync')

            assert len(violations) == 1
            v = violations[0]
            assert v['op_id'] == 'op-stall-1'
            assert v['task_id'] == '11'
            assert v['snapshot_status'] == 'in-progress'
            assert v['live_status'] == 'done'

        @pytest.mark.asyncio
        async def test_unit_freshness_violation_observed_status_alias(self):
            """observed_status alias is accepted in addition to snapshot_status."""
            taskmaster = AsyncMock()
            taskmaster.get_task.return_value = {'status': 'done'}

            ops = [self._make_add_memory_op(
                op_id='op-stall-alias',
                metadata={'task_id': '11', 'observed_status': 'in-progress'},
            )]
            violations = await _check_stall_guard_freshness(ops, taskmaster, '/project', 'recon-stage-task_knowledge_sync')

            assert len(violations) == 1
            assert violations[0]['snapshot_status'] == 'in-progress'

        @pytest.mark.asyncio
        async def test_unit_no_snapshot_status_key_skipped(self):
            """add_memory op without snapshot_status/observed_status -> no violation."""
            taskmaster = AsyncMock()
            taskmaster.get_task.return_value = {'status': 'done'}

            ops = [self._make_add_memory_op(
                metadata={'task_id': '11'},  # no snapshot_status key
            )]
            violations = await _check_stall_guard_freshness(ops, taskmaster, '/project', 'recon-stage-task_knowledge_sync')

            assert violations == []
            taskmaster.get_task.assert_not_called()

        @pytest.mark.asyncio
        async def test_unit_snapshot_matches_live_no_violation(self):
            """add_memory op where snapshot_status matches live status -> no violation."""
            taskmaster = AsyncMock()
            taskmaster.get_task.return_value = {'status': 'in-progress'}

            ops = [self._make_add_memory_op(
                metadata={'task_id': '11', 'snapshot_status': 'in-progress'},
            )]
            violations = await _check_stall_guard_freshness(ops, taskmaster, '/project', 'recon-stage-task_knowledge_sync')

            assert violations == []

        @pytest.mark.asyncio
        async def test_run_records_stall_guard_freshness_violation(self, stage2_guard_mock_deps, caplog):
            """run() adds stall_guard_freshness_violations when snapshot_status mismatches live."""
            stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **stage2_guard_mock_deps)
            stage.project_id = 'dark_factory'
            stage.project_root = '/project'

            stall_op = {
                'id': 'op-stall-integration-1',
                'agent_id': 'recon-stage-task_knowledge_sync',
                'operation': 'add_memory',
                'params': json.dumps({
                    'metadata': {'task_id': '11', 'snapshot_status': 'in-progress'},
                }),
                'layer': 'write_op',
                'causation_id': 'test-run-1137-c',
                'created_at': '2026-01-01T00:00:00',
            }
            stage2_guard_mock_deps['journal'].write_journal.get_ops_by_causation.return_value = [stall_op]
            stage2_guard_mock_deps['taskmaster'].get_task.return_value = {'status': 'done'}

            with (
                patch.object(stage, 'assemble_payload', new=AsyncMock(return_value='payload')),
                patch(
                    'fused_memory.reconciliation.stages.base.run_stage_via_cli',
                    new=AsyncMock(return_value=_make_stage2_guard_cli_result([], stats={})),
                ),
                caplog.at_level(
                    logging.WARNING,
                    logger='fused_memory.reconciliation.stages.task_knowledge_sync',
                ),
            ):
                report = await stage.run(
                    events=[],
                    watermark=Watermark(project_id='dark_factory'),
                    prior_reports=[],
                    run_id='test-run-1137-c',
                )

            # Guard 2: freshness violations counter incremented
            assert report.stats.get('stall_guard_freshness_violations') == 1

            # Guard 2: WARNING log
            target_logger = 'fused_memory.reconciliation.stages.task_knowledge_sync'
            guard_logs = [
                r for r in caplog.records
                if r.name == target_logger
                and r.levelno == logging.WARNING
                and 'stall_guard_freshness_violation' in r.getMessage()
            ]
            assert len(guard_logs) == 1, (
                f'expected one stall_guard_freshness_violation WARNING, got {len(guard_logs)}'
            )

    class TestFlagCounterCompleteness:
        """Unit + integration tests for _check_flag_counter_completeness helper."""

        def test_unit_mismatch_low_reported(self):
            """prior_reports[0] has 5 items_flagged, stats reports 3 -> mismatch True."""
            prior_report = StageReport(
                stage=StageId.memory_consolidator,
                started_at=datetime.now(tz=UTC),
                completed_at=datetime.now(tz=UTC),
                items_flagged=[{'id': str(i)} for i in range(5)],
            )
            result = _check_flag_counter_completeness({'stage1_flags_processed': 3}, [prior_report])
            assert result['expected'] == 5
            assert result['reported'] == 3
            assert result['mismatch'] is True

        def test_unit_counts_match_no_mismatch(self):
            """prior_reports[0] has 5 items_flagged, stats reports 5 -> mismatch False."""
            prior_report = StageReport(
                stage=StageId.memory_consolidator,
                started_at=datetime.now(tz=UTC),
                completed_at=datetime.now(tz=UTC),
                items_flagged=[{'id': str(i)} for i in range(5)],
            )
            result = _check_flag_counter_completeness({'stage1_flags_processed': 5}, [prior_report])
            assert result['expected'] == 5
            assert result['reported'] == 5
            assert result['mismatch'] is False

        def test_unit_empty_prior_reports_no_mismatch(self):
            """No prior reports -> expected=0, mismatch=False."""
            result = _check_flag_counter_completeness({'stage1_flags_processed': 3}, [])
            assert result['expected'] == 0
            assert result['reported'] == 3
            assert result['mismatch'] is False

        def test_unit_zero_stats_value_matches_empty_flagged(self):
            """prior_reports[0].items_flagged=[], stats reports 0 -> no mismatch."""
            prior_report = StageReport(
                stage=StageId.memory_consolidator,
                started_at=datetime.now(tz=UTC),
                completed_at=datetime.now(tz=UTC),
                items_flagged=[],
            )
            result = _check_flag_counter_completeness({}, [prior_report])
            assert result['expected'] == 0
            assert result['reported'] == 0
            assert result['mismatch'] is False

        def test_unit_wrong_stage_skips_baseline(self):
            """prior_reports[0] is not memory_consolidator -> mismatch=False, no clamp."""
            # Simulate a pipeline reorder: prior_reports[0] is Stage 3
            prior_report = StageReport(
                stage=StageId.task_knowledge_sync,
                started_at=datetime.now(tz=UTC),
                completed_at=datetime.now(tz=UTC),
                items_flagged=[{'id': str(i)} for i in range(5)],
            )
            result = _check_flag_counter_completeness({'stage1_flags_processed': 3}, [prior_report])
            assert result['mismatch'] is False
            assert result['expected'] == 0  # no baseline used
            assert result['reported'] == 3

        @pytest.mark.asyncio
        async def test_run_clamps_stage1_flags_processed_on_mismatch(self, stage2_guard_mock_deps, caplog):
            """run() clamps stage1_flags_processed to prior_reports[0] truth and warns."""
            stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **stage2_guard_mock_deps)
            stage.project_id = 'dark_factory'
            stage.project_root = '/project'

            _now = datetime.now(tz=UTC)
            stage1_prior = StageReport(
                stage=StageId.memory_consolidator,
                started_at=_now,
                completed_at=_now,
                items_flagged=[{'id': str(i)} for i in range(5)],
            )

            with (
                patch.object(stage, 'assemble_payload', new=AsyncMock(return_value='payload')),
                patch(
                    'fused_memory.reconciliation.stages.base.run_stage_via_cli',
                    new=AsyncMock(return_value=_make_stage2_guard_cli_result(
                        [], stats={'stage1_flags_processed': 3}
                    )),
                ),
                caplog.at_level(
                    logging.WARNING,
                    logger='fused_memory.reconciliation.stages.task_knowledge_sync',
                ),
            ):
                report = await stage.run(
                    events=[],
                    watermark=Watermark(project_id='dark_factory'),
                    prior_reports=[stage1_prior],
                    run_id='test-run-1137-d',
                )

            # Guard 4: stage1_flags_processed clamped to 5 (truth from prior_reports)
            assert report.stats.get('stage1_flags_processed') == 5

            # Guard 4: WARNING log
            target_logger = 'fused_memory.reconciliation.stages.task_knowledge_sync'
            guard_logs = [
                r for r in caplog.records
                if r.name == target_logger
                and r.levelno == logging.WARNING
                and 'stage1_flags_processed_mismatch' in r.getMessage()
            ]
            assert len(guard_logs) == 1, (
                f'expected one stage1_flags_processed_mismatch WARNING, got {len(guard_logs)}'
            )
            rec = guard_logs[0]
            assert getattr(rec, 'expected', None) == 5
            assert getattr(rec, 'reported', None) == 3

    class TestComposition:
        """All four guards fire in a single stage.run() invocation."""

        @pytest.fixture
        def mock_deps_composition(self):
            config = ReconciliationConfig(enabled=True, explore_codebase_root='/tmp/test')
            write_journal_mock = MagicMock()
            write_journal_mock.get_ops_by_causation = AsyncMock(return_value=[])
            journal_mock = MagicMock()
            journal_mock.write_journal = write_journal_mock
            return {
                'memory_service': AsyncMock(),
                'taskmaster': AsyncMock(),
                'journal': journal_mock,
                'config': config,
            }

        def _make_cli_result(self, flagged_items: list[dict], stats: dict | None = None) -> MagicMock:
            report = {'flagged_items': flagged_items, 'summary': 'ok'}
            if stats:
                report['stats'] = stats
            return MagicMock(
                success=True,
                report=report,
                llm_calls=1,
                tokens_used=0,
                cost_usd=0.0,
                model='m',
            )

        @pytest.mark.asyncio
        async def test_all_four_guards_fire_together(self, mock_deps_composition, caplog):
            """All four guards fire simultaneously in a single stage.run() call.

            Op stream contains:
              (a) terminal-state update_task violation (task 42 -> done)
              (b) set_task_status post-action mismatch (task 7: target=done, live=pending)
              (c) stall-guard freshness violation (task 11: snapshot=in-progress, live=done)
            Stats seed: tasks_modified=5, memories_written=3, stage1_flags_processed=1
            Stage 1 prior report: 4 items_flagged -> Guard 4 expects 4, reported 1 -> mismatch.
            """
            stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps_composition)
            stage.project_id = 'dark_factory'
            stage.project_root = '/project'

            # Three ops that each trip a different guard
            ops = [
                {  # Guard 1: terminal-state update_task for task 42
                    'id': 'op-term-42',
                    'agent_id': 'recon-stage-task_knowledge_sync',
                    'operation': 'update_task',
                    'params': json.dumps({'task_id': '42'}),
                    'layer': 'write_op',
                    'causation_id': 'test-run-compose-1',
                    'created_at': '2026-01-01T00:00:00',
                },
                {  # Guard 3: set_task_status target=done but live=pending for task 7
                    'id': 'op-sts-7',
                    'agent_id': 'recon-stage-task_knowledge_sync',
                    'operation': 'set_task_status',
                    'params': json.dumps({'task_id': '7', 'status': 'done'}),
                    'layer': 'write_op',
                    'causation_id': 'test-run-compose-1',
                    'created_at': '2026-01-01T00:00:01',
                },
                {  # Guard 2: stall-guard freshness: snapshot=in-progress but live=done for task 11
                    'id': 'op-mem-11',
                    'agent_id': 'recon-stage-task_knowledge_sync',
                    'operation': 'add_memory',
                    'params': json.dumps({
                        'content': 'task 11 is stalled',
                        'metadata': {'task_id': '11', 'snapshot_status': 'in-progress'},
                    }),
                    'layer': 'write_op',
                    'causation_id': 'test-run-compose-1',
                    'created_at': '2026-01-01T00:00:02',
                },
            ]
            mock_deps_composition['journal'].write_journal.get_ops_by_causation.return_value = ops

            # taskmaster.get_task returns different live statuses per task_id
            async def get_task_side_effect(task_id, project_root):
                return {
                    '42': {'status': 'done'},    # triggers Guard 1 (terminal)
                    '7': {'status': 'pending'},  # triggers Guard 3 (sts mismatch)
                    '11': {'status': 'done'},    # triggers Guard 2 (freshness)
                }.get(task_id, {'status': 'pending'})

            mock_deps_composition['taskmaster'].get_task.side_effect = get_task_side_effect

            # Stage 1 prior report with 4 items_flagged (Guard 4 expects 4, reports 1)
            _now = datetime.now(tz=UTC)
            stage1_prior = StageReport(
                stage=StageId.memory_consolidator,
                started_at=_now,
                completed_at=_now,
                items_flagged=[{'id': str(i)} for i in range(4)],
            )

            with (
                patch.object(stage, 'assemble_payload', new=AsyncMock(return_value='payload')),
                patch(
                    'fused_memory.reconciliation.stages.base.run_stage_via_cli',
                    new=AsyncMock(return_value=self._make_cli_result(
                        [],
                        stats={
                            'tasks_modified': 5,
                            'memories_written': 3,
                            'stage1_flags_processed': 1,
                        },
                    )),
                ),
                caplog.at_level(
                    logging.INFO,
                    logger='fused_memory.reconciliation.stages.task_knowledge_sync',
                ),
            ):
                report = await stage.run(
                    events=[],
                    watermark=Watermark(project_id='dark_factory'),
                    prior_reports=[stage1_prior],
                    run_id='test-run-compose-1',
                )

            target_logger = 'fused_memory.reconciliation.stages.task_knowledge_sync'

            # Guard 1: not_applicable_count=1, tasks_modified decremented from 5 to 4
            assert report.stats.get('not_applicable_count') == 1

            # Guard 3: set_task_status_post_action_mismatches=1, tasks_modified further decremented to 3
            assert report.stats.get('set_task_status_post_action_mismatches') == 1

            # Guard 1 + Guard 3 combined: tasks_modified = 5 - 1 - 1 = 3
            assert report.stats.get('tasks_modified') == 3

            # Guard 2: stall_guard_freshness_violations=1
            assert report.stats.get('stall_guard_freshness_violations') == 1

            # Guard 4: stage1_flags_processed clamped to 4 (truth from prior_reports)
            assert report.stats.get('stage1_flags_processed') == 4

            # Exactly four structured log records (one per guard):
            # 1 INFO skipped_done_task
            info_logs = [
                r for r in caplog.records
                if r.name == target_logger
                and r.levelno == logging.INFO
                and 'skipped_done_task' in r.getMessage()
            ]
            assert len(info_logs) == 1, f'expected 1 skipped_done_task INFO, got {len(info_logs)}'

            # 1 WARNING set_task_status_post_action_mismatch
            sts_logs = [
                r for r in caplog.records
                if r.name == target_logger
                and r.levelno == logging.WARNING
                and 'set_task_status_post_action_mismatch' in r.getMessage()
            ]
            assert len(sts_logs) == 1, (
                f'expected 1 set_task_status_post_action_mismatch WARNING, got {len(sts_logs)}'
            )

            # 1 WARNING stall_guard_freshness_violation
            freshness_logs = [
                r for r in caplog.records
                if r.name == target_logger
                and r.levelno == logging.WARNING
                and 'stall_guard_freshness_violation' in r.getMessage()
            ]
            assert len(freshness_logs) == 1, (
                f'expected 1 stall_guard_freshness_violation WARNING, got {len(freshness_logs)}'
            )

            # 1 WARNING stage1_flags_processed_mismatch
            flag_logs = [
                r for r in caplog.records
                if r.name == target_logger
                and r.levelno == logging.WARNING
                and 'stage1_flags_processed_mismatch' in r.getMessage()
            ]
            assert len(flag_logs) == 1, (
                f'expected 1 stage1_flags_processed_mismatch WARNING, got {len(flag_logs)}'
            )

        @pytest.mark.asyncio
        async def test_null_write_journal_degrades_gracefully(self, mock_deps_composition, caplog):
            """When write_journal is None, Guards 1-3 skip gracefully; Guard 4 still fires.

            No exception should be raised. Stats keys for Guards 1-3 should be absent.
            Guard 4 still emits its warning because it is pure stats arithmetic.
            """
            # Override write_journal to None to simulate a stand without a journal
            mock_deps_composition['journal'].write_journal = None

            stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps_composition)
            stage.project_id = 'dark_factory'
            stage.project_root = '/project'

            _now = datetime.now(tz=UTC)
            stage1_prior = StageReport(
                stage=StageId.memory_consolidator,
                started_at=_now,
                completed_at=_now,
                items_flagged=[{'id': str(i)} for i in range(4)],
            )

            with (
                patch.object(stage, 'assemble_payload', new=AsyncMock(return_value='payload')),
                patch(
                    'fused_memory.reconciliation.stages.base.run_stage_via_cli',
                    new=AsyncMock(return_value=self._make_cli_result(
                        [],
                        stats={
                            'tasks_modified': 5,
                            'stage1_flags_processed': 1,
                        },
                    )),
                ),
                caplog.at_level(
                    logging.WARNING,
                    logger='fused_memory.reconciliation.stages.task_knowledge_sync',
                ),
            ):
                report = await stage.run(
                    events=[],
                    watermark=Watermark(project_id='dark_factory'),
                    prior_reports=[stage1_prior],
                    run_id='test-run-compose-null-journal',
                )

            target_logger = 'fused_memory.reconciliation.stages.task_knowledge_sync'

            # Guards 1-3 should NOT have fired (no write_journal ops available)
            assert 'not_applicable_count' not in report.stats
            assert 'set_task_status_post_action_mismatches' not in report.stats
            assert 'stall_guard_freshness_violations' not in report.stats
            # tasks_modified unchanged (no guard decrements applied)
            assert report.stats.get('tasks_modified') == 5

            # Guard 4 must still fire (pure stats arithmetic, no write_journal dependency)
            assert report.stats.get('stage1_flags_processed') == 4  # clamped to truth

            flag_logs = [
                r for r in caplog.records
                if r.name == target_logger
                and r.levelno == logging.WARNING
                and 'stage1_flags_processed_mismatch' in r.getMessage()
            ]
            assert len(flag_logs) == 1, (
                f'expected 1 stage1_flags_processed_mismatch WARNING even with null journal, '
                f'got {len(flag_logs)}'
            )

        @pytest.mark.asyncio
        async def test_null_journal_object_degrades_gracefully(self, mock_deps_composition, caplog):
            """When self.journal is None entirely, Guards 1-3 skip; Guard 4 still fires."""
            deps = {**mock_deps_composition, 'journal': None}
            stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **deps)
            stage.project_id = 'dark_factory'
            stage.project_root = '/project'

            _now = datetime.now(tz=UTC)
            stage1_prior = StageReport(
                stage=StageId.memory_consolidator,
                started_at=_now,
                completed_at=_now,
                items_flagged=[{'id': str(i)} for i in range(3)],
            )

            with (
                patch.object(stage, 'assemble_payload', new=AsyncMock(return_value='payload')),
                patch(
                    'fused_memory.reconciliation.stages.base.run_stage_via_cli',
                    new=AsyncMock(return_value=self._make_cli_result(
                        [],
                        stats={'tasks_modified': 2, 'stage1_flags_processed': 0},
                    )),
                ),
                caplog.at_level(
                    logging.WARNING,
                    logger='fused_memory.reconciliation.stages.task_knowledge_sync',
                ),
            ):
                report = await stage.run(
                    events=[],
                    watermark=Watermark(project_id='dark_factory'),
                    prior_reports=[stage1_prior],
                    run_id='test-run-null-journal-obj',
                )

            target_logger = 'fused_memory.reconciliation.stages.task_knowledge_sync'
            assert 'not_applicable_count' not in report.stats
            assert 'set_task_status_post_action_mismatches' not in report.stats
            assert 'stall_guard_freshness_violations' not in report.stats
            # Guard 4 fires: stage1_flags_processed clamped from 0 to 3
            assert report.stats.get('stage1_flags_processed') == 3
            flag_logs = [
                r for r in caplog.records
                if r.name == target_logger
                and 'stage1_flags_processed_mismatch' in r.getMessage()
            ]
            assert len(flag_logs) == 1

        @pytest.mark.asyncio
        async def test_get_ops_by_causation_raises_degrades_gracefully(
            self, mock_deps_composition, caplog
        ):
            """When get_ops_by_causation raises, a WARNING is emitted, ops default to [],
            Guards 1-3 skip, Guard 4 still fires, and run completes without exception."""
            mock_deps_composition['journal'].write_journal.get_ops_by_causation = AsyncMock(
                side_effect=RuntimeError('db connection lost')
            )
            stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps_composition)
            stage.project_id = 'dark_factory'
            stage.project_root = '/project'

            _now = datetime.now(tz=UTC)
            stage1_prior = StageReport(
                stage=StageId.memory_consolidator,
                started_at=_now,
                completed_at=_now,
                items_flagged=[{'id': str(i)} for i in range(2)],
            )

            with (
                patch.object(stage, 'assemble_payload', new=AsyncMock(return_value='payload')),
                patch(
                    'fused_memory.reconciliation.stages.base.run_stage_via_cli',
                    new=AsyncMock(return_value=self._make_cli_result(
                        [],
                        stats={'tasks_modified': 1, 'stage1_flags_processed': 0},
                    )),
                ),
                caplog.at_level(
                    logging.WARNING,
                    logger='fused_memory.reconciliation.stages.task_knowledge_sync',
                ),
            ):
                report = await stage.run(
                    events=[],
                    watermark=Watermark(project_id='dark_factory'),
                    prior_reports=[stage1_prior],
                    run_id='test-run-ops-raise',
                )

            target_logger = 'fused_memory.reconciliation.stages.task_knowledge_sync'
            # WARNING about get_ops_by_causation failure
            ops_fail_logs = [
                r for r in caplog.records
                if r.name == target_logger
                and 'get_ops_by_causation failed' in r.getMessage()
            ]
            assert len(ops_fail_logs) == 1, (
                f'expected 1 get_ops_by_causation failure WARNING, got {len(ops_fail_logs)}'
            )
            # Guards 1-3 skipped (no ops)
            assert 'not_applicable_count' not in report.stats
            assert 'set_task_status_post_action_mismatches' not in report.stats
            assert 'stall_guard_freshness_violations' not in report.stats
            # Guard 4 still fires
            assert report.stats.get('stage1_flags_processed') == 2

        @pytest.mark.asyncio
        async def test_cache_build_failure_no_false_positives(
            self, mock_deps_composition, caplog
        ):
            """When get_task raises during cache build, Guards 2 and 3 must NOT fire.

            Scenario: three Stage-2 ops on task_id='99' (one per guard: update_task,
            set_task_status target=done, add_memory snapshot_status=in-progress).
            taskmaster.get_task raises unconditionally for '99'.

            Before the fix, the 'unknown' sentinel triggers 'unknown' != 'done' and
            'unknown' != 'in-progress', so Guards 2 (freshness) and 3 (sts post-action)
            fire falsely; Guard 3 also decrements tasks_modified.  After the fix, the
            cache omits '99' and all helpers skip.

            Assertions (b) and (c) are the principal regressions; (d) catches the
            false tasks_modified decrement from Guard 3; (e) verifies the failure was
            still logged.
            """
            stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps_composition)
            stage.project_id = 'dark_factory'
            stage.project_root = '/project'

            # Three ops on the same task_id='99' — one per guard
            ops = [
                {  # Guard 1: terminal-state update_task
                    'id': 'op-term-99',
                    'agent_id': 'recon-stage-task_knowledge_sync',
                    'operation': 'update_task',
                    'params': json.dumps({'task_id': '99'}),
                    'layer': 'write_op',
                    'causation_id': 'test-run-1176-cache-fail',
                    'created_at': '2026-01-01T00:00:00',
                },
                {  # Guard 3 (sts): set_task_status post-action target=done
                    'id': 'op-sts-99',
                    'agent_id': 'recon-stage-task_knowledge_sync',
                    'operation': 'set_task_status',
                    'params': json.dumps({'task_id': '99', 'status': 'done'}),
                    'layer': 'write_op',
                    'causation_id': 'test-run-1176-cache-fail',
                    'created_at': '2026-01-01T00:00:01',
                },
                {  # Guard 2 (freshness): add_memory snapshot_status=in-progress
                    'id': 'op-mem-99',
                    'agent_id': 'recon-stage-task_knowledge_sync',
                    'operation': 'add_memory',
                    'params': json.dumps({
                        'content': 'task 99 is stalled',
                        'metadata': {'task_id': '99', 'snapshot_status': 'in-progress'},
                    }),
                    'layer': 'write_op',
                    'causation_id': 'test-run-1176-cache-fail',
                    'created_at': '2026-01-01T00:00:02',
                },
            ]
            mock_deps_composition['journal'].write_journal.get_ops_by_causation.return_value = ops

            # get_task always raises — '99' cannot be fetched from Taskmaster
            async def _get_task_raises(task_id, project_root):
                raise RuntimeError('boom')

            mock_deps_composition['taskmaster'].get_task.side_effect = _get_task_raises

            with (
                patch.object(stage, 'assemble_payload', new=AsyncMock(return_value='payload')),
                patch(
                    'fused_memory.reconciliation.stages.base.run_stage_via_cli',
                    new=AsyncMock(return_value=self._make_cli_result(
                        [],
                        stats={'tasks_modified': 5, 'memories_written': 1},
                    )),
                ),
                caplog.at_level(
                    logging.WARNING,
                    logger='fused_memory.reconciliation.stages.task_knowledge_sync',
                ),
            ):
                report = await stage.run(
                    events=[],
                    watermark=Watermark(project_id='dark_factory'),
                    prior_reports=[],
                    run_id='test-run-1176-cache-fail',
                )

            target_logger = 'fused_memory.reconciliation.stages.task_knowledge_sync'

            # (a) Guard 1 did not fire — 'unknown' is not in TERMINAL_STATUSES
            assert 'not_applicable_count' not in report.stats

            # (b) Guard 3 (sts post-action) did NOT falsely fire — principal regression
            assert 'set_task_status_post_action_mismatches' not in report.stats

            # (c) Guard 2 (freshness) did NOT falsely fire — principal regression
            assert 'stall_guard_freshness_violations' not in report.stats

            # (d) tasks_modified is unchanged — no false decrements
            assert report.stats.get('tasks_modified') == 5

            # (e) Exactly one cache-build WARNING was emitted naming task_id=99
            cache_fail_logs = [
                r for r in caplog.records
                if r.name == target_logger
                and r.levelno == logging.WARNING
                and 'get_task failed for task_id=99 during cache build' in r.getMessage()
            ]
            assert len(cache_fail_logs) == 1, (
                f'expected 1 cache-build WARNING for task_id=99, got {len(cache_fail_logs)}'
            )

        @pytest.mark.asyncio
        async def test_cache_build_nondict_result_no_false_positives(
            self, mock_deps_composition, caplog
        ):
            """When get_task returns a non-dict, Guards 2 and 3 must NOT fire.

            Scenario: same three Stage-2 ops as test_cache_build_failure_no_false_positives
            (one per guard, task_id='99'), but get_task returns None (non-dict result)
            instead of raising.

            Before the fix, the non-dict result was stored as status_cache['99'] = 'unknown'
            (via ``_extract_status(result) if isinstance(result, dict) else 'unknown'``),
            so Guards 2 (freshness) and 3 (sts post-action) would fire falsely; Guard 3
            also decrements tasks_modified.  After the fix, the cache omits '99' (non-dict
            → continue) and all helpers skip uniformly.
            """
            stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **mock_deps_composition)
            stage.project_id = 'dark_factory'
            stage.project_root = '/project'

            ops = [
                {  # Guard 1: terminal-state update_task
                    'id': 'op-term-99c',
                    'agent_id': 'recon-stage-task_knowledge_sync',
                    'operation': 'update_task',
                    'params': json.dumps({'task_id': '99'}),
                    'layer': 'write_op',
                    'causation_id': 'test-run-1176-nondict',
                    'created_at': '2026-01-01T00:00:00',
                },
                {  # Guard 3 (sts): set_task_status post-action target=done
                    'id': 'op-sts-99c',
                    'agent_id': 'recon-stage-task_knowledge_sync',
                    'operation': 'set_task_status',
                    'params': json.dumps({'task_id': '99', 'status': 'done'}),
                    'layer': 'write_op',
                    'causation_id': 'test-run-1176-nondict',
                    'created_at': '2026-01-01T00:00:01',
                },
                {  # Guard 2 (freshness): add_memory snapshot_status=in-progress
                    'id': 'op-mem-99c',
                    'agent_id': 'recon-stage-task_knowledge_sync',
                    'operation': 'add_memory',
                    'params': json.dumps({
                        'content': 'task 99 status unknown',
                        'metadata': {'task_id': '99', 'snapshot_status': 'in-progress'},
                    }),
                    'layer': 'write_op',
                    'causation_id': 'test-run-1176-nondict',
                    'created_at': '2026-01-01T00:00:02',
                },
            ]
            mock_deps_composition['journal'].write_journal.get_ops_by_causation.return_value = ops

            # get_task returns None — simulates an unexpected non-dict API response
            async def _get_task_returns_none(task_id, project_root):
                return None

            mock_deps_composition['taskmaster'].get_task.side_effect = _get_task_returns_none

            with (
                patch.object(stage, 'assemble_payload', new=AsyncMock(return_value='payload')),
                patch(
                    'fused_memory.reconciliation.stages.base.run_stage_via_cli',
                    new=AsyncMock(return_value=self._make_cli_result(
                        [],
                        stats={'tasks_modified': 5, 'memories_written': 1},
                    )),
                ),
                caplog.at_level(
                    logging.WARNING,
                    logger='fused_memory.reconciliation.stages.task_knowledge_sync',
                ),
            ):
                report = await stage.run(
                    events=[],
                    watermark=Watermark(project_id='dark_factory'),
                    prior_reports=[],
                    run_id='test-run-1176-nondict',
                )

            # (a) Guard 1 did not fire — 'unknown' is not in TERMINAL_STATUSES
            assert 'not_applicable_count' not in report.stats

            # (b) Guard 3 (sts post-action) did NOT falsely fire
            assert 'set_task_status_post_action_mismatches' not in report.stats

            # (c) Guard 2 (freshness) did NOT falsely fire
            assert 'stall_guard_freshness_violations' not in report.stats

            # (d) tasks_modified is unchanged — no false decrements from Guard 3
            assert report.stats.get('tasks_modified') == 5

            target_logger = 'fused_memory.reconciliation.stages.task_knowledge_sync'

            # (e) Silent-skip contract: non-dict result must NOT emit the cache-build
            # warning that the raised-exception branch emits (see sibling test).
            assert not any(
                r.name == target_logger
                and r.levelno == logging.WARNING
                and 'during cache build' in r.getMessage()
                and 'task_id=99' in r.getMessage()
                for r in caplog.records
            )


# ---------------------------------------------------------------------------
# Task 1201 — BaseStage._escalation_queue attribute + harness wiring
# ---------------------------------------------------------------------------


class TestBaseStageEscalationQueueAttribute:
    """BaseStage.__init__ initialises _escalation_queue to None."""

    @pytest.fixture
    def mock_deps(self):
        config = ReconciliationConfig(enabled=True, explore_codebase_root='/tmp/test')
        return {
            'memory_service': AsyncMock(),
            'taskmaster': AsyncMock(),
            'journal': AsyncMock(),
            'config': config,
        }

    def test_escalation_queue_initialised_to_none(self, mock_deps):
        """(a) Fresh MemoryConsolidator has _escalation_queue == None."""
        stage = MemoryConsolidator(StageId.memory_consolidator, **mock_deps)
        assert stage._escalation_queue is None

    def test_escalation_queue_is_settable(self, mock_deps):
        """(b) _escalation_queue is settable and round-trips."""
        stage = MemoryConsolidator(StageId.memory_consolidator, **mock_deps)
        fake_queue = MagicMock()
        stage._escalation_queue = fake_queue
        assert stage._escalation_queue is fake_queue


@pytest.fixture
def minimal_harness():
    """Construct a minimal ReconciliationHarness with all deps mocked."""
    from fused_memory.config.schema import FusedMemoryConfig
    from fused_memory.reconciliation.harness import ReconciliationHarness

    config = FusedMemoryConfig()
    memory_service = AsyncMock()
    taskmaster = AsyncMock()
    journal = AsyncMock()
    event_buffer = AsyncMock()
    harness = ReconciliationHarness(
        memory_service=memory_service,
        taskmaster=taskmaster,
        journal=journal,
        event_buffer=event_buffer,
        config=config,
    )
    return harness


class TestHarnessWiresEscalationQueueOntoStages:
    """ReconciliationHarness._make_stages wires _escalation_queue onto each stage."""

    def test_make_stages_wires_escalation_queue(self, minimal_harness):
        """_make_stages() propagates harness._escalation_queue to each returned stage."""
        fake_queue = MagicMock()
        minimal_harness._escalation_queue = fake_queue

        stages = minimal_harness._make_stages()

        for stage in stages:
            assert stage._escalation_queue is fake_queue, (
                f'Stage {stage.stage_id} did not receive _escalation_queue from harness'
            )

    def test_make_stages_no_queue_leaves_stages_at_none(self, minimal_harness):
        """When _escalation_queue is None, stages are left with _escalation_queue=None."""
        # Ensure harness has no queue (default)
        minimal_harness._escalation_queue = None

        stages = minimal_harness._make_stages()

        for stage in stages:
            assert stage._escalation_queue is None


class TestPropagateEscalationQueueHelper:
    """ReconciliationHarness._propagate_escalation_queue wires URL and queue onto an arbitrary stage list."""

    def test_propagates_url_and_queue_onto_pre_existing_stages(self, minimal_harness):
        """Helper propagates harness URL and queue to every stage in the supplied list."""
        fake_queue = MagicMock()
        minimal_harness._escalation_url = 'http://test.local:9999/mcp'
        minimal_harness._escalation_queue = fake_queue

        stages = [MagicMock(_escalation_url=None, _escalation_queue=None) for _ in range(3)]
        minimal_harness._propagate_escalation_queue(stages)

        for stage in stages:
            assert stage._escalation_url == 'http://test.local:9999/mcp', (
                'Stage did not receive _escalation_url from harness'
            )
            assert stage._escalation_queue is fake_queue, (
                'Stage did not receive _escalation_queue from harness'
            )

    def test_no_op_when_harness_has_no_url_or_queue(self, minimal_harness):
        """When harness has no URL or queue, pre-existing stage values are not overwritten."""
        sentinel_queue = MagicMock()
        minimal_harness._escalation_url = None
        minimal_harness._escalation_queue = None

        stages = [
            MagicMock(_escalation_url='preexisting', _escalation_queue=sentinel_queue)
            for _ in range(3)
        ]
        minimal_harness._propagate_escalation_queue(stages)

        for stage in stages:
            assert stage._escalation_url == 'preexisting', (
                'Helper must not overwrite _escalation_url when harness has no URL'
            )
            assert stage._escalation_queue is sentinel_queue, (
                'Helper must not overwrite _escalation_queue when harness has no queue'
            )

    def test_propagates_url_only_when_queue_is_none(self, minimal_harness):
        """When only URL is set on harness, URL propagates but queue is left at pre-existing value."""
        sentinel_queue = MagicMock()
        minimal_harness._escalation_url = 'http://test.local:9999/mcp'
        minimal_harness._escalation_queue = None

        stages = [
            MagicMock(_escalation_url=None, _escalation_queue=sentinel_queue)
            for _ in range(3)
        ]
        minimal_harness._propagate_escalation_queue(stages)

        for stage in stages:
            assert stage._escalation_url == 'http://test.local:9999/mcp', (
                'Helper must propagate _escalation_url when harness has a URL'
            )
            assert stage._escalation_queue is sentinel_queue, (
                'Helper must not overwrite _escalation_queue when harness queue is None'
            )

    def test_propagates_to_single_pass_iterable(self, minimal_harness):
        """Helper must propagate URL and queue even when stages is a single-pass iterable (e.g., generator/iter())."""
        fake_queue = MagicMock()
        minimal_harness._escalation_url = 'http://test.local:9999/mcp'
        minimal_harness._escalation_queue = fake_queue

        stages = [MagicMock(_escalation_url=None, _escalation_queue=None) for _ in range(3)]
        minimal_harness._propagate_escalation_queue(iter(stages))

        for stage in stages:
            assert stage._escalation_url == 'http://test.local:9999/mcp', (
                'Stage did not receive _escalation_url — single-pass iterator was exhausted by URL pass'
            )
            assert stage._escalation_queue is fake_queue, (
                'Stage did not receive _escalation_queue — single-pass iterator was exhausted by URL pass'
            )

    def test_propagates_queue_only_when_url_is_none(self, minimal_harness):
        """When only queue is set on harness, queue propagates but URL is left at pre-existing value.

        Symmetric counterpart of test_propagates_url_only_when_queue_is_none; completes
        the asymmetric-guard matrix: (url+queue, neither, url-only, queue-only).
        """
        fake_queue = MagicMock()
        minimal_harness._escalation_url = None
        minimal_harness._escalation_queue = fake_queue

        stages = [
            MagicMock(_escalation_url='preexisting', _escalation_queue=None)
            for _ in range(3)
        ]
        minimal_harness._propagate_escalation_queue(stages)

        for stage in stages:
            assert stage._escalation_queue is fake_queue, (
                'Helper must propagate _escalation_queue when harness has a queue'
            )
            assert stage._escalation_url == 'preexisting', (
                'Helper must not overwrite _escalation_url when harness has no URL'
            )


class TestStage2HintConversionDetection:
    """assemble_payload() surfaces Tasks Needing Memory Hint Attention section.

    Tests that the new conditional ``### Tasks Needing Memory Hint Attention``
    section is produced by ``assemble_payload()`` exactly when active tasks fail
    ``_needs_hint_conversion()``.  Uses the harness-injection pattern
    (``stage.filtered_task_tree``) from ``TestTaskKnowledgeSyncFilteredTaskTree``
    to control the task pool without calling taskmaster.
    """

    _SECTION_HEADER = '### Tasks Needing Memory Hint Attention'

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

    def _make_task_with_hints(self, tid: int, status: str, hints) -> dict:
        """Build a minimal task dict with an explicit memory_hints value.

        Always pass a real hints value (list or dict).  For tasks without
        any hints use ``_make_task_no_hints`` (empty ``metadata`` dict) or
        ``_make_task_no_metadata`` (missing ``metadata`` key entirely).
        """
        return {
            'id': tid,
            'title': f'Task {tid}',
            'status': status,
            'dependencies': [],
            'metadata': {'memory_hints': hints},
        }

    def _make_task_no_hints(self, tid: int, status: str) -> dict:
        """Build a task dict whose ``metadata`` dict has no ``memory_hints`` key.

        Covers the ``not task_hints`` (falsy) branch of ``_needs_hint_conversion``.
        For the complementary case where ``metadata`` is absent entirely, use
        ``_make_task_no_metadata``.
        """
        return {'id': tid, 'title': f'Task {tid}', 'status': status, 'dependencies': [], 'metadata': {}}

    def _make_task_no_metadata(self, tid: int, status: str) -> dict:
        """Build a task dict with no ``metadata`` key at all.

        Exercises the ``isinstance(metadata, dict) else None`` guard in
        ``_needs_hint_conversion``: ``task.get('metadata')`` returns ``None``,
        which is not a dict, so ``task_hints`` is forced to ``None`` (falsy →
        ``_needs_hint_conversion`` returns ``True``).
        """
        return {'id': tid, 'title': f'Task {tid}', 'status': status, 'dependencies': []}

    def _make_tree(self, active_tasks: list[dict]):
        """Wrap *active_tasks* in a FilteredTaskTree with empty done/cancelled lists."""
        return FilteredTaskTree(
            active_tasks=active_tasks,
            done_tasks=[],
            cancelled_tasks=[],
            done_count=0,
            cancelled_count=0,
            other_count=0,
            total_count=len(active_tasks),
        )

    @pytest.mark.asyncio
    async def test_list_format_task_appears_in_hint_conversion_section(
        self, mock_deps, watermark
    ):
        """Active task with legacy list-format hints must appear in the new section."""
        task = self._make_task_with_hints(
            10, 'in-progress', [{'entity': 'Foo', 'query': 'q'}]
        )
        stage = make_configured_task_knowledge_sync_stage(
            mock_deps, project_id='test_project', project_root='/tmp/test_project'
        )
        stage.filtered_task_tree = self._make_tree([task])

        payload = await stage.assemble_payload([], watermark, [])

        assert self._SECTION_HEADER in payload, (
            f'Expected "{self._SECTION_HEADER}" in payload, but it was absent.\n'
            f'Payload snippet: {payload[:3000]!r}'
        )
        section = _extract_section(payload, self._SECTION_HEADER)
        assert '[10]' in section, (
            f'Task 10 (list-format hints) must appear in the section body.\n'
            f'Section: {section!r}'
        )

    @pytest.mark.asyncio
    async def test_valid_dict_hints_task_excluded_from_section(
        self, mock_deps, watermark
    ):
        """Active task with valid dict hints must NOT produce the section (empty list → omit)."""
        task = self._make_task_with_hints(
            20, 'pending', {'entities': ['Bar'], 'queries': ['q']}
        )
        stage = make_configured_task_knowledge_sync_stage(
            mock_deps, project_id='test_project', project_root='/tmp/test_project'
        )
        stage.filtered_task_tree = self._make_tree([task])

        payload = await stage.assemble_payload([], watermark, [])

        assert self._SECTION_HEADER not in payload, (
            f'Section header must be absent when all active tasks have valid dict hints.\n'
            f'Payload snippet: {payload[:3000]!r}'
        )

    @pytest.mark.asyncio
    async def test_no_hints_task_appears_in_section(self, mock_deps, watermark):
        """Active task with no memory_hints key must appear in the section (falsy branch)."""
        task = self._make_task_no_hints(30, 'pending')
        stage = make_configured_task_knowledge_sync_stage(
            mock_deps, project_id='test_project', project_root='/tmp/test_project'
        )
        stage.filtered_task_tree = self._make_tree([task])

        payload = await stage.assemble_payload([], watermark, [])

        assert self._SECTION_HEADER in payload, (
            f'Expected "{self._SECTION_HEADER}" when task has no hints.\n'
            f'Payload snippet: {payload[:3000]!r}'
        )
        section = _extract_section(payload, self._SECTION_HEADER)
        assert '[30]' in section, (
            f'Task 30 (no hints) must appear in the section body.\n'
            f'Section: {section!r}'
        )

    @pytest.mark.asyncio
    async def test_mixed_tree_filters_correctly(self, mock_deps, watermark):
        """Mixed tree: only the list-format task appears in the new section; both in Active Task Tree."""
        list_task = self._make_task_with_hints(
            40, 'in-progress', [{'entity': 'X', 'query': 'q'}]
        )
        dict_task = self._make_task_with_hints(
            50, 'pending', {'entities': ['Y'], 'queries': ['q2']}
        )
        stage = make_configured_task_knowledge_sync_stage(
            mock_deps, project_id='test_project', project_root='/tmp/test_project'
        )
        stage.filtered_task_tree = self._make_tree([list_task, dict_task])

        payload = await stage.assemble_payload([], watermark, [])

        # The section must be present (at least one qualifying task)
        assert self._SECTION_HEADER in payload, (
            f'Expected "{self._SECTION_HEADER}" with one list-format task in tree.'
        )
        section = _extract_section(payload, self._SECTION_HEADER)

        # List-format task (id=40) must be in the section
        assert '[40]' in section, (
            f'Task 40 (list-format hints) must appear in hint section.\nSection: {section!r}'
        )
        # Valid-dict task (id=50) must NOT be in the section
        assert '[50]' not in section, (
            f'Task 50 (valid dict hints) must NOT appear in hint section.\nSection: {section!r}'
        )

        # Both tasks must appear in the Active Task Tree (section sanity check)
        active_section = _extract_section(payload, '### Active Task Tree')
        assert '[40]' in active_section, (
            'Task 40 must appear in Active Task Tree section'
        )
        assert '[50]' in active_section, (
            'Task 50 must appear in Active Task Tree section'
        )

    @pytest.mark.asyncio
    async def test_no_metadata_key_task_appears_in_section(self, mock_deps, watermark):
        """Active task with no 'metadata' key at all must appear in the section.

        Exercises the ``isinstance(metadata, dict) else None`` guard in
        ``_needs_hint_conversion``: ``task.get('metadata')`` returns ``None``
        (not a dict), so ``task_hints`` is forced to ``None`` → falsy →
        ``_needs_hint_conversion`` returns ``True``.
        """
        task = self._make_task_no_metadata(60, 'pending')
        stage = make_configured_task_knowledge_sync_stage(
            mock_deps, project_id='test_project', project_root='/tmp/test_project'
        )
        stage.filtered_task_tree = self._make_tree([task])

        payload = await stage.assemble_payload([], watermark, [])

        assert self._SECTION_HEADER in payload, (
            f'Expected "{self._SECTION_HEADER}" when task has no metadata key.\n'
            f'Payload snippet: {payload[:3000]!r}'
        )
        section = _extract_section(payload, self._SECTION_HEADER)
        assert '[60]' in section, (
            f'Task 60 (no metadata key) must appear in the section body.\n'
            f'Section: {section!r}'
        )

    @pytest.mark.asyncio
    async def test_empty_active_tasks_omits_section(self, mock_deps, watermark):
        """Empty active_tasks list must produce no hint-attention section header.

        This is the most common production state (no active tasks).  The section
        must be unconditionally absent because there are no candidates to flag.
        """
        stage = make_configured_task_knowledge_sync_stage(
            mock_deps, project_id='test_project', project_root='/tmp/test_project'
        )
        stage.filtered_task_tree = self._make_tree([])

        payload = await stage.assemble_payload([], watermark, [])

        assert self._SECTION_HEADER not in payload, (
            f'Section header must be absent when active_tasks is empty.\n'
            f'Payload snippet: {payload[:3000]!r}'
        )

    @pytest.mark.asyncio
    async def test_hint_section_suppressed_in_remediation_mode(
        self, mock_deps, watermark
    ):
        """hint_conversion_section must be absent when remediation_mode=True.

        Mirrors test_proactive_sample_skipped_in_remediation_mode: the
        hint-attention section is a 'general sync' activity and must not
        surface during focused remediation runs.
        """
        task = self._make_task_with_hints(
            11, 'in-progress', [{'entity': 'X', 'query': 'q'}]
        )
        stage = make_configured_task_knowledge_sync_stage(
            mock_deps, project_id='test_project', project_root='/tmp/test_project'
        )
        stage.remediation_mode = True
        stage.filtered_task_tree = self._make_tree([task])

        payload = await stage.assemble_payload([], watermark, [])

        assert self._SECTION_HEADER not in payload, (
            f'Hint conversion section must be absent in remediation_mode=True.\n'
            f'Payload snippet: {payload[:3000]!r}'
        )

    @pytest.mark.asyncio
    async def test_hint_section_slice_then_filter_excludes_qualifying_tasks_past_cap(
        self, mock_deps, watermark
    ):
        """Discriminator test: slice-then-filter must produce exactly 10 hint tasks,
        not 20 as filter-then-slice would.

        Input: 60 tasks (IDs 1..60, status='pending').
          - Positions 1..40: dict-format hints  → NON-qualifying (already converted)
          - Positions 41..60: list-format hints → qualifying (need conversion)

        slice-then-filter (correct):
          slice to positions 1..50, then filter → positions 41..50 qualify → 10 tasks.

        filter-then-slice (wrong ordering):
          filter first → positions 41..60 qualify → 20 tasks, then slice → all 20.

        Assertions pin the slice-then-filter outcome unambiguously:
          (a) IDs 41..50 are present   — the qualifying tasks inside the cap
          (b) IDs 51..60 are absent    — qualifying tasks OUTSIDE the cap
          (c) IDs 1..40 are absent     — non-qualifying tasks
          (d) exact count == 10        — distinguishes from filter-then-slice (count=20)
        """
        import re as _re

        # Positions 1..40: dict-format hints (non-qualifying via _needs_hint_conversion)
        tasks = [
            self._make_task_with_hints(
                i, 'pending', {'entities': [f'E{i}'], 'queries': ['q']}
            )
            for i in range(1, 41)
        ]
        # Positions 41..60: list-format hints (qualifying)
        tasks += [
            self._make_task_with_hints(
                i, 'pending', [{'entity': f'E{i}', 'query': 'q'}]
            )
            for i in range(41, 61)
        ]

        stage = make_configured_task_knowledge_sync_stage(
            mock_deps, project_id='test_project', project_root='/tmp/test_project'
        )
        stage.filtered_task_tree = self._make_tree(tasks)

        payload = await stage.assemble_payload([], watermark, [])

        hint_section = _extract_section(payload, self._SECTION_HEADER)

        # (a) IDs 41..50 must be present (qualifying, inside the 50-task cap).
        for tid in range(41, 51):
            assert f'[{tid}]' in hint_section, (
                f'Task {tid} (qualifying, position <= 50) must appear in hint section.\n'
                f'Hint section: {hint_section!r}'
            )

        # (b) IDs 51..60 must be absent (qualifying, but outside the 50-task cap).
        for tid in range(51, 61):
            assert f'[{tid}]' not in hint_section, (
                f'Task {tid} (qualifying, position > 50) must NOT appear in hint section '
                f'(slice-then-filter: cap applied before filter).\n'
                f'Hint section: {hint_section!r}'
            )

        # (c) IDs 1..40 must be absent (non-qualifying dict-format hints).
        for tid in range(1, 41):
            assert f'[{tid}]' not in hint_section, (
                f'Task {tid} (non-qualifying dict hints) must NOT appear in hint section.\n'
                f'Hint section: {hint_section!r}'
            )

        # (d) Exact count == 10 distinguishes slice-then-filter (10) from
        #     filter-then-slice (20).
        count = len(_re.findall(r'^- \[\d+\] ', hint_section, _re.MULTILINE))
        assert count == 10, (
            f'Expected exactly 10 hint tasks (positions 41-50), got {count}.\n'
            f'filter-then-slice would produce 20; slice-then-filter produces 10.\n'
            f'Hint section: {hint_section!r}'
        )

    @pytest.mark.asyncio
    async def test_hint_section_subset_of_active_tree_under_max_chars_clamp(
        self, mock_deps, watermark
    ):
        """Every task ID in the hint section must also appear in the rendered Active Task Tree.

        When 50 verbose active tasks push the rendered tree past the 50_000-char
        max_chars clamp, format_filtered_task_tree drops tail tasks from the body.
        The hint section must mirror this: it must NOT reference task IDs that were
        dropped from the tree.

        With the current raw-slice code (filtered.active_tasks[:MAX_ACTIVE_TASKS_RENDERED])
        in assemble_payload, hint_ids = {1..50} but active_tree_ids = {1..N} for some
        N < 50 — so this test FAILS until step-4 replaces the slice with
        select_visible_active.
        """
        # 50 tasks, each with a ~1100-char title so that 50 * 1100 = 55_000 > 50_000,
        # forcing format_filtered_task_tree to drop tail tasks.  All tasks have
        # list-format hints so every task qualifies for the hint section.
        pad = 'x' * 1100
        tasks = [
            self._make_task_with_hints(
                i, 'pending', [{'entity': f'E{i}', 'query': 'q'}]
            )
            for i in range(1, 51)
        ]
        # Overwrite titles with padded versions (make_task_with_hints uses f'Task {tid}').
        for t in tasks:
            t['title'] = pad

        stage = make_configured_task_knowledge_sync_stage(
            mock_deps, project_id='test_project', project_root='/tmp/test_project'
        )
        stage.filtered_task_tree = self._make_tree(tasks)

        payload = await stage.assemble_payload([], watermark, [])

        active_section = _extract_section(payload, '### Active Task Tree')
        hint_section = _extract_section(payload, self._SECTION_HEADER)

        # Precondition: the max_chars clamp must have actually fired; otherwise the
        # test is vacuous.  The truncation notice appears when lines were dropped.
        assert 'truncated for budget' in active_section, (
            'Precondition failed: max_chars clamp did not fire — increase title length '
            'or task count so the rendered tree exceeds 50_000 chars.\n'
            f'Active section length: {len(active_section)}'
        )

        import re

        active_tree_ids = set(re.findall(r'\[(\d+)\]', active_section))
        hint_ids = set(re.findall(r'\[(\d+)\]', hint_section))

        # Main assertion: hint section must only reference tasks visible in the tree.
        orphaned = hint_ids - active_tree_ids
        assert not orphaned, (
            f'Hint section references task IDs absent from the Active Task Tree: {sorted(orphaned)}\n'
            f'hint_ids={sorted(hint_ids)}, active_tree_ids={sorted(active_tree_ids)}'
        )


# ── Regression: cycle 8df8bdcd — Stage 2 Recently-Completed/Done-Provenance
#    title↔task_id contract (task 1379) ─────────────────────────────────────
#
# Scenario data centralised in _fm_helpers.make_8df8_scenario(with_provenance=True).
# See tests/_fm_helpers.py for the canonical fixture definition.


class TestStage2RecentlyCompletedAndProvenancePreserveIdTitlePairing:
    """Stage 2 formatters: id↔title pairing locked across all provenance branches.

    Rendering notes for _render_done_provenance_section:
    - Legacy branch (no metadata.done_provenance): always renders '- [id] title — ...'
    - Commit branch (project_root set):            renders '- [id] title\n  commit: sha'
    - Commit branch (project_root=None):           renders nothing (commit and project_root check fails)
    - Note-only branch:                            renders '  note: text' with NO [id] header
    Tests assert id↔title for every branch that DOES render a parseable id-line,
    and verify no neighbor-title bleed in each branch's individual output.
    """

    # 8df8bdcd scenario with provenance metadata:
    #   1369 — commit branch, 1355 — note-only branch, 1361 — legacy/none branch
    _TASKS, _TITLE_BY_ID = make_8df8_scenario(id_type=int, status='done', with_provenance=True)

    @pytest.mark.asyncio
    async def test_render_done_provenance_section_commit_and_legacy_preserve_id_title(self):
        """_render_done_provenance_section: rendered [id] headers carry each task's OWN title.

        Covers the commit branch (project_root set → fires [id] header) and the legacy
        branch.  The note-only branch (id=1355) does NOT render a [id] header by design.
        Uses parse_rendered_id_title_pairs(kind='provenance') via assert_id_title_pairing.
        _git_show_name_only is patched to '' to avoid spawning a real git subprocess.
        """
        # Patch _git_show_name_only so commit branch fires deterministically without a
        # real subprocess (git show result is irrelevant to the id↔title assertion).
        with patch(
            'fused_memory.reconciliation.stages.task_knowledge_sync._git_show_name_only',
            new=AsyncMock(return_value=''),
        ):
            rendered = await _render_done_provenance_section(list(self._TASKS), project_root='/tmp')

        assert '### Done-task Provenance' in rendered, (
            f'Provenance section header missing:\n{rendered!r}'
        )

        # Commit (1369) and legacy (1361) render [id] lines; note-only (1355) has no [id] line
        # assert_id_title_pairing checks own-title + expected_ids + anti-vacuity
        assert_id_title_pairing(
            rendered, self._TITLE_BY_ID, kind='provenance',
            expected_ids={1369, 1361},
        )

    def test_format_task_list_recently_completed_preserves_id_title_pairing(self):
        """format_task_list for Recently Completed section: id↔title locked.

        The '### Recently Completed Tasks' section is rendered via format_task_list
        on done_tasks.  Verifies the 8df8bdcd scenario across completion-order
        ≠ id-sort-order, covering all provenance branch types.
        """
        rendered = format_task_list(list(self._TASKS))
        assert rendered != 'No tasks.'
        assert_id_title_pairing(
            rendered, self._TITLE_BY_ID, kind='active',
            expected_ids={1369, 1355, 1361},
        )

    @pytest.mark.asyncio
    async def test_provenance_branch_commit_renders_own_id_and_title(self):
        """commit branch (project_root set): header carries this task's OWN id and title.

        _git_show_name_only is patched to '' to avoid spawning a real git subprocess
        (env-dependent if project_root is inside a git worktree).
        """
        task = self._TASKS[0]  # id=1369, commit branch
        # Patch _git_show_name_only so the commit branch fires without a real subprocess.
        with patch(
            'fused_memory.reconciliation.stages.task_knowledge_sync._git_show_name_only',
            new=AsyncMock(return_value=''),
        ):
            rendered = await _render_done_provenance_section([task], project_root='/tmp')

        assert f'[{task["id"]}]' in rendered, (
            f'Commit branch: id {task["id"]} missing:\n{rendered!r}'
        )
        assert task['title'] in rendered, (
            f'Commit branch: title {task["title"]!r} missing:\n{rendered!r}'
        )
        # Neighbor titles must NOT appear
        for other in self._TASKS[1:]:
            assert other['title'] not in rendered, (
                f'Commit branch: neighbor title {other["title"]!r} leaked into:\n{rendered!r}'
            )

    @pytest.mark.asyncio
    async def test_provenance_branch_note_content_does_not_bleed_neighbor_titles(self):
        """note-only branch: rendered note text contains no neighbor task titles.

        The note branch renders '  note: <text>' WITHOUT a [id] header (by design).
        The contract asserted here: no other task's title bleeds into the note line.
        """
        task = self._TASKS[1]  # id=1355, note branch
        rendered = await _render_done_provenance_section([task], project_root=None)

        # Note line is rendered
        assert 'note: Covered by sibling task 1354' in rendered, (
            f'Note branch: note text missing:\n{rendered!r}'
        )
        # No neighbor title should appear anywhere
        for other in [self._TASKS[0], self._TASKS[2]]:
            assert other['title'] not in rendered, (
                f'Note branch: neighbor title {other["title"]!r} leaked:\n{rendered!r}'
            )

    @pytest.mark.asyncio
    async def test_provenance_branch_legacy_renders_own_id_and_title(self):
        """legacy branch: header line carries this task's OWN id and title."""
        task = self._TASKS[2]  # id=1361, legacy branch
        rendered = await _render_done_provenance_section([task], project_root=None)

        assert f'[{task["id"]}]' in rendered, (
            f'Legacy branch: id {task["id"]} missing:\n{rendered!r}'
        )
        assert task['title'] in rendered, (
            f'Legacy branch: title {task["title"]!r} missing:\n{rendered!r}'
        )
        for other in self._TASKS[:2]:
            assert other['title'] not in rendered, (
                f'Legacy branch: neighbor title {other["title"]!r} leaked:\n{rendered!r}'
            )
