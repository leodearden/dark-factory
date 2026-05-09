"""Tests for Stage 1 Memory Consolidator prompt and payload — task 1174.

Covers four concerns:
- FQN tool names in the stage1 system prompt (TestStage1PromptFqnToolNames)
- project_root threading through assemble_payload / _format_assembled_payload
  (TestStage1PayloadThreadsProjectRootLegacy, TestStage1PayloadThreadsProjectRootAssembled)
- project_root omitted when empty (TestStage1PayloadOmitsProjectRootWhenUnset)
- Error-envelope handling guidance in the prompt
  (TestStage1PromptHandlesGetTaskErrorEnvelope)
- Pinned flag_type constant (TestTerminalStatePreCheckFlagTypeConstant,
  TestStage1PromptReferencesPinnedFlagType)
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from fused_memory.models.reconciliation import (
    AssembledPayload,
    StageId,
    Watermark,
)
from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT
from fused_memory.reconciliation.stages.memory_consolidator import MemoryConsolidator
from fused_memory.config.schema import ReconciliationConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_section(payload: str, header: str) -> str:
    """Return the body of *header* up to the next '\\n#' boundary, or '' if absent.

    Mirrors the helper at tests/test_stages.py:53-65, with one addition: we
    search for '\\n' + header so we match only genuine section-level markdown
    headers that start at a line boundary, not in-line cross-references (e.g.
    'see ## Terminal-State Pre-Check Discipline below' in the helper note).
    """
    needle = '\n' + header
    start = payload.find(needle)
    if start == -1:
        return ''
    # Skip the leading '\n' so the returned slice starts at the header itself.
    start = start + 1
    end = payload.find('\n#', start + 1)
    if end == -1:
        end = len(payload)
    return payload[start:end]


def _make_consolidator(project_root: str = '') -> MemoryConsolidator:
    """Build a MemoryConsolidator with mocked deps — mirrors test_stages.py ~L1418."""
    config = ReconciliationConfig()
    memory_mock = AsyncMock()
    memory_mock.get_episodes = AsyncMock(return_value=[])
    memory_mock.mem0 = AsyncMock()
    memory_mock.mem0.get_all = AsyncMock(return_value={'results': []})
    memory_mock.get_status = AsyncMock(return_value={})

    stage = MemoryConsolidator(
        StageId.memory_consolidator,
        memory_mock,
        AsyncMock(),  # taskmaster
        AsyncMock(),  # journal
        config,
    )
    stage.project_id = 'test_project'
    stage.project_root = project_root
    stage.episode_limit = 5
    stage.memory_limit = 10
    return stage


# ---------------------------------------------------------------------------
# step-1: FQN tool names
# ---------------------------------------------------------------------------


class TestStage1PromptFqnToolNames:
    """STAGE1_SYSTEM_PROMPT uses mcp__fused-memory__* FQN names where required."""

    def test_helper_note_uses_fqn_get_task(self):
        """The helper-note paragraph (lines 32-34 region) uses the FQN form of get_task."""
        # The helper note section is the prose before "## Your Consolidation Tasks"
        preamble = _extract_section(STAGE1_SYSTEM_PROMPT, 'You do not have access to task')
        assert 'mcp__fused-memory__get_task' in preamble, (
            'Helper note should reference mcp__fused-memory__get_task'
        )

    def test_terminal_state_section_uses_fqn_get_task(self):
        """Terminal-State Pre-Check Discipline uses mcp__fused-memory__get_task."""
        section = _extract_section(STAGE1_SYSTEM_PROMPT, '## Terminal-State Pre-Check Discipline')
        assert section, 'Terminal-State Pre-Check Discipline section not found'
        assert 'mcp__fused-memory__get_task' in section, (
            'Terminal-State Pre-Check Discipline should use mcp__fused-memory__get_task'
        )

    def test_uuid_resolution_section_uses_fqn_search(self):
        """UUID Resolution Discipline uses mcp__fused-memory__search."""
        section = _extract_section(STAGE1_SYSTEM_PROMPT, '## UUID Resolution Discipline')
        assert section, 'UUID Resolution Discipline section not found'
        assert 'mcp__fused-memory__search' in section, (
            'UUID Resolution Discipline should use mcp__fused-memory__search'
        )

    def test_uuid_resolution_section_uses_fqn_delete_memory(self):
        """UUID Resolution Discipline uses mcp__fused-memory__delete_memory."""
        section = _extract_section(STAGE1_SYSTEM_PROMPT, '## UUID Resolution Discipline')
        assert section, 'UUID Resolution Discipline section not found'
        assert 'mcp__fused-memory__delete_memory' in section, (
            'UUID Resolution Discipline should use mcp__fused-memory__delete_memory'
        )

    def test_uuid_resolution_section_no_bare_search_call(self):
        """UUID Resolution Discipline does NOT contain bare 'search()' (step-text form)."""
        section = _extract_section(STAGE1_SYSTEM_PROMPT, '## UUID Resolution Discipline')
        # The bare `search()` pattern in the 3-step instructions should be replaced
        assert 'search()' not in section, (
            'UUID Resolution Discipline should not contain bare search() call — '
            'use mcp__fused-memory__search instead'
        )

    def test_uuid_resolution_section_no_bare_delete_memory(self):
        """UUID Resolution Discipline does NOT contain bare 'delete_memory(' in step text."""
        section = _extract_section(STAGE1_SYSTEM_PROMPT, '## UUID Resolution Discipline')
        # The bare `delete_memory(id=...)` pattern in step 3 should be replaced
        # We check specifically for the step-instruction form "delete_memory(id="
        assert 'delete_memory(id=' not in section, (
            'UUID Resolution Discipline should not contain bare delete_memory(id=...) — '
            'use mcp__fused-memory__delete_memory instead'
        )

    def test_terminal_state_section_no_bare_get_task_call(self):
        """Terminal-State Pre-Check Discipline does NOT contain bare 'get_task(' step text."""
        section = _extract_section(STAGE1_SYSTEM_PROMPT, '## Terminal-State Pre-Check Discipline')
        # The bare `get_task(id=<task_id>, ...)` in step 1 should be replaced
        assert 'get_task(id=' not in section, (
            'Terminal-State Pre-Check Discipline should not contain bare get_task(id=...) — '
            'use mcp__fused-memory__get_task instead'
        )


# ---------------------------------------------------------------------------
# step-3: project_root in legacy (time-windowed) assemble_payload
# ---------------------------------------------------------------------------


class TestStage1PayloadThreadsProjectRootLegacy:
    """assemble_payload includes project_root directive when project_root is set."""

    @pytest.mark.asyncio
    async def test_assemble_payload_includes_project_root(self):
        """Legacy assemble_payload emits 'Use project_root=...' when project_root is set."""
        stage = _make_consolidator(project_root='/home/leo/src/test_proj')
        watermark = Watermark(project_id='test_project')

        result = await stage.assemble_payload(
            events=[], watermark=watermark, prior_reports=[]
        )

        assert 'Use project_root="/home/leo/src/test_proj"' in result, (
            'assemble_payload should emit project_root directive when project_root is set'
        )


# ---------------------------------------------------------------------------
# step-5: project_root in assembled-payload branch (_format_assembled_payload)
# ---------------------------------------------------------------------------


class TestStage1PayloadThreadsProjectRootAssembled:
    """_format_assembled_payload includes project_root directive when project_root is set."""

    @pytest.mark.asyncio
    async def test_format_assembled_payload_includes_project_root(self):
        """Assembled-payload branch emits 'Use project_root=...' when project_root is set."""
        stage = _make_consolidator(project_root='/home/leo/src/test_proj')

        # Set assembled_payload to trigger the ContextAssembler branch
        stage.assembled_payload = AssembledPayload(
            events=[],
            context_items={},
        )

        watermark = Watermark(project_id='test_project')
        result = await stage.assemble_payload(
            events=[], watermark=watermark, prior_reports=[]
        )

        assert 'Use project_root="/home/leo/src/test_proj"' in result, (
            '_format_assembled_payload should emit project_root directive when project_root is set'
        )


# ---------------------------------------------------------------------------
# step-7: project_root omitted when empty (BaseStage default '')
# ---------------------------------------------------------------------------


class TestStage1PayloadOmitsProjectRootWhenUnset:
    """When project_root is '' (BaseStage default), no project_root line is emitted."""

    @pytest.mark.asyncio
    async def test_assemble_payload_omits_project_root_when_empty(self):
        """Legacy assemble_payload does NOT emit project_root line when project_root=''."""
        stage = _make_consolidator(project_root='')
        watermark = Watermark(project_id='test_project')

        result = await stage.assemble_payload(
            events=[], watermark=watermark, prior_reports=[]
        )

        assert 'Use project_root=' not in result, (
            'assemble_payload should omit project_root directive when project_root is empty'
        )
        assert 'Use project_root=""' not in result

    @pytest.mark.asyncio
    async def test_format_assembled_payload_omits_project_root_when_empty(self):
        """Assembled-payload branch does NOT emit project_root line when project_root=''."""
        stage = _make_consolidator(project_root='')
        stage.assembled_payload = AssembledPayload(
            events=[],
            context_items={},
        )

        watermark = Watermark(project_id='test_project')
        result = await stage.assemble_payload(
            events=[], watermark=watermark, prior_reports=[]
        )

        assert 'Use project_root=' not in result, (
            '_format_assembled_payload should omit project_root directive when project_root is empty'
        )
        assert 'Use project_root=""' not in result


# ---------------------------------------------------------------------------
# step-9: error-envelope handling guidance in the prompt
# ---------------------------------------------------------------------------


class TestStage1PromptHandlesGetTaskErrorEnvelope:
    """Terminal-State Pre-Check Discipline instructs agents on get_task error envelopes."""

    def test_prompt_mentions_error_field(self):
        """Terminal-State section mentions an 'error' field returned by get_task."""
        section = _extract_section(STAGE1_SYSTEM_PROMPT, '## Terminal-State Pre-Check Discipline')
        assert section, 'Terminal-State Pre-Check Discipline section not found'
        assert 'error' in section, (
            'Terminal-State section should mention the error field from get_task responses'
        )

    def test_prompt_instructs_inconclusive_skip_on_error(self):
        """Terminal-State section tells agent to skip/treat-as-inconclusive on error envelope."""
        section = _extract_section(STAGE1_SYSTEM_PROMPT, '## Terminal-State Pre-Check Discipline')
        assert section, 'Terminal-State Pre-Check Discipline section not found'
        # Must mention either 'inconclusive' or 'skip' (or both) in the error context
        has_inconclusive = 'inconclusive' in section
        has_skip = 'skip' in section.lower()
        assert has_inconclusive or has_skip, (
            'Terminal-State section should instruct the agent to treat verification as '
            'inconclusive or skip the write when get_task returns an error envelope'
        )


# ---------------------------------------------------------------------------
# step-11: TERMINAL_STATE_PRE_CHECK_FLAG_TYPE constant in flag_dedup
# ---------------------------------------------------------------------------


class TestTerminalStatePreCheckFlagTypeConstant:
    """TERMINAL_STATE_PRE_CHECK_FLAG_TYPE constant exists in flag_dedup and has correct value."""

    def test_constant_importable(self):
        """TERMINAL_STATE_PRE_CHECK_FLAG_TYPE can be imported from flag_dedup."""
        from fused_memory.reconciliation.flag_dedup import TERMINAL_STATE_PRE_CHECK_FLAG_TYPE  # noqa: F401

    def test_constant_value(self):
        """TERMINAL_STATE_PRE_CHECK_FLAG_TYPE equals 'terminal_state_pre_check'."""
        from fused_memory.reconciliation.flag_dedup import TERMINAL_STATE_PRE_CHECK_FLAG_TYPE
        assert TERMINAL_STATE_PRE_CHECK_FLAG_TYPE == 'terminal_state_pre_check'

    def test_constant_is_str(self):
        """TERMINAL_STATE_PRE_CHECK_FLAG_TYPE is a str instance."""
        from fused_memory.reconciliation.flag_dedup import TERMINAL_STATE_PRE_CHECK_FLAG_TYPE
        assert isinstance(TERMINAL_STATE_PRE_CHECK_FLAG_TYPE, str)


# ---------------------------------------------------------------------------
# step-13: prompt references the pinned flag_type constant value
# ---------------------------------------------------------------------------


class TestStage1PromptReferencesPinnedFlagType:
    """STAGE1_SYSTEM_PROMPT contains the literal value of TERMINAL_STATE_PRE_CHECK_FLAG_TYPE."""

    def test_terminal_state_section_contains_flag_type_value(self):
        """Terminal-State Pre-Check Discipline contains the literal flag_type string."""
        from fused_memory.reconciliation.flag_dedup import TERMINAL_STATE_PRE_CHECK_FLAG_TYPE

        section = _extract_section(STAGE1_SYSTEM_PROMPT, '## Terminal-State Pre-Check Discipline')
        assert section, 'Terminal-State Pre-Check Discipline section not found'
        assert TERMINAL_STATE_PRE_CHECK_FLAG_TYPE in section, (
            f'Terminal-State Pre-Check Discipline should contain the literal flag_type value '
            f'"{TERMINAL_STATE_PRE_CHECK_FLAG_TYPE}" so it stays coupled to the dedup module'
        )
