"""The ``### Known Projects`` payload section, and Stage 3 rendering it (task 6109).

Stage 3's prompt tells the model a foreign citation resolves only when the
foreign project is listed in "Known Projects"; these tests pin that the Stage-3
payload renders that listing from the same registry ``cite_task`` validates
against. Helpers are copied rather than imported across test modules (house norm).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import StageId, Watermark
from fused_memory.models.scope import ProjectId, ProjectRoot, ProjectScope
from fused_memory.reconciliation.known_projects_section import (
    KNOWN_PROJECTS_HEADER,
    render_known_projects_section,
)
from fused_memory.reconciliation.stages.task_knowledge_sync import IntegrityCheck

THREE_PROJECTS = {
    'reify': '/home/leo/src/reify',
    'dark_factory': '/home/leo/src/dark-factory',
    'autopilot_video': '/home/leo/src/autopilot-video',
}


def _scope(project_id: str, project_root: str) -> ProjectScope:
    return ProjectScope(ProjectId(project_id), ProjectRoot(project_root))


def _deps() -> dict:
    return {
        'memory_service': AsyncMock(),
        'taskmaster': AsyncMock(),
        'journal': AsyncMock(),
        'config': ReconciliationConfig(enabled=True, explore_codebase_root='/tmp/test'),
    }


def _rows(section: str) -> list[str]:
    return [line for line in section.splitlines() if line.startswith('- ')]


class TestRenderKnownProjectsSection:
    def test_empty_registry_renders_nothing(self):
        assert render_known_projects_section({}, current_project_id='reify') == ''

    def test_single_project_renders_nothing(self):
        assert render_known_projects_section(
            {'reify': '/home/leo/src/reify'}, current_project_id='reify'
        ) == ''

    def test_multi_project_section_lists_every_project_current_first(self):
        section = render_known_projects_section(THREE_PROJECTS, current_project_id='reify')

        assert KNOWN_PROJECTS_HEADER == '### Known Projects (for cross-project routing)'
        assert KNOWN_PROJECTS_HEADER in section.splitlines()
        rows = _rows(section)
        assert len(rows) == 3
        expected_order = ['reify', 'autopilot_video', 'dark_factory']
        for row, project_id in zip(rows, expected_order, strict=True):
            assert row.split()[1] == project_id
            assert f'→ {THREE_PROJECTS[project_id]}' in row
        assert [row for row in rows if '(current)' in row] == [rows[0]]

    def test_current_project_absent_from_registry_gets_no_current_row(self):
        section = render_known_projects_section(
            {'b': '/b', 'a': '/a'}, current_project_id='z'
        )

        rows = _rows(section)
        assert [row.split()[1] for row in rows] == ['a', 'b']
        assert not any('(current)' in row for row in rows)


class TestStage3PayloadRendersKnownProjects:
    async def _payload(self, known_projects: dict[str, str]) -> str:
        stage = IntegrityCheck(
            StageId.integrity_check,
            **_deps(),
            scope=_scope('reify', '/home/leo/src/reify'),
            known_projects=known_projects,
        )
        return await stage.assemble_payload([], Watermark(project_id='reify'), [])

    @pytest.mark.asyncio
    async def test_payload_embeds_the_known_projects_section_when_multiple_projects_registered(
        self,
    ):
        payload = await self._payload(THREE_PROJECTS)

        section = render_known_projects_section(THREE_PROJECTS, current_project_id='reify')
        assert section in payload
        assert payload.index(KNOWN_PROJECTS_HEADER) < payload.index('## Your Task')

    @pytest.mark.asyncio
    async def test_payload_omits_section_for_single_project_registry(self):
        payload = await self._payload({'reify': '/home/leo/src/reify'})

        assert KNOWN_PROJECTS_HEADER not in payload
