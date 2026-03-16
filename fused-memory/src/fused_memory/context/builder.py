"""Context builder — produces structured text briefings from memory + task state."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fused_memory.backends.taskmaster_client import TaskmasterBackend
    from fused_memory.services.memory_service import MemoryService

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Builds structured context briefings from memory and task state.

    Produces markdown documents that agents can use as working context
    at session start or before complex tasks.
    """

    def __init__(
        self,
        memory_service: MemoryService,
        taskmaster: TaskmasterBackend | None = None,
    ):
        self.memory = memory_service
        self.taskmaster = taskmaster

    async def build_briefing(
        self,
        project_id: str,
        task_id: str | None = None,
        memory_hints: dict[str, Any] | None = None,
        include_task_tree: bool = True,
        project_root: str | None = None,
    ) -> str:
        """Build a structured context briefing.

        Args:
            project_id: Project scope for memory searches.
            task_id: Specific task to focus on (fetches details + hints).
            memory_hints: Explicit hints to execute (queries + entity names).
                Format: {"queries": [...], "entities": [...]}
            include_task_tree: Whether to include the full task tree.
            project_root: Absolute path for task operations.

        Returns:
            Formatted markdown briefing document.
        """
        sections: list[str] = []

        # 1. Project context
        overview = await self._search_section(
            project_id,
            'project overview architecture goals',
            'Project Overview',
        )
        if overview:
            sections.append(overview)

        # 2. Recent decisions
        decisions = await self._search_section(
            project_id,
            'recent decisions and rationale',
            'Recent Decisions',
            categories=['decisions_and_rationale'],
        )
        if decisions:
            sections.append(decisions)

        # 3. Active conventions
        conventions = await self._search_section(
            project_id,
            'coding conventions and project norms',
            'Active Conventions',
            categories=['preferences_and_norms'],
        )
        if conventions:
            sections.append(conventions)

        # 4. Task-specific context
        if task_id and self.taskmaster and project_root:
            task_section = await self._build_task_section(
                task_id, project_id, project_root, memory_hints
            )
            if task_section:
                sections.append(task_section)
        elif memory_hints:
            hints_section = await self._execute_hints(project_id, memory_hints)
            if hints_section:
                sections.append(hints_section)

        # 5. Task tree
        if include_task_tree and self.taskmaster and project_root:
            tree_section = await self._build_task_tree_section(project_root)
            if tree_section:
                sections.append(tree_section)

        if not sections:
            return '# Context Briefing\n\nNo context available yet.'

        return '# Context Briefing\n\n' + '\n\n---\n\n'.join(sections)

    async def _search_section(
        self,
        project_id: str,
        query: str,
        title: str,
        categories: list[str] | None = None,
        limit: int = 5,
    ) -> str | None:
        """Run a search and format results as a markdown section."""
        try:
            results = await self.memory.search(
                query=query,
                project_id=project_id,
                categories=categories,
                limit=limit,
            )
        except Exception as e:
            logger.error(f'Search failed for "{query}": {e}')
            return None

        if not results:
            return None

        lines = [f'## {title}\n']
        for r in results:
            source = r.source_store.value if r.source_store else 'unknown'
            category = r.category.value if r.category else 'uncategorized'
            lines.append(f'- [{source}/{category}] {r.content}')

        return '\n'.join(lines)

    async def _build_task_section(
        self,
        task_id: str,
        project_id: str,
        project_root: str,
        extra_hints: dict[str, Any] | None = None,
    ) -> str | None:
        """Fetch task details and execute memory hints."""
        if not self.taskmaster:
            return None

        try:
            task = await self.taskmaster.get_task(
                task_id=task_id, project_root=project_root
            )
        except Exception as e:
            logger.error(f'Failed to fetch task {task_id}: {e}')
            return None

        lines = [f'## Current Task: {task_id}\n']

        # Extract task fields
        task_data = task.get('task', task) if isinstance(task, dict) else task
        if isinstance(task_data, dict):
            if task_data.get('title'):
                lines.append(f'**Title:** {task_data["title"]}')
            if task_data.get('description'):
                lines.append(f'**Description:** {task_data["description"]}')
            if task_data.get('status'):
                lines.append(f'**Status:** {task_data["status"]}')

        # Collect hints from task metadata + explicit hints
        hints = self._extract_hints(task_data, extra_hints)
        if hints:
            hint_results = await self._execute_hints(project_id, hints)
            if hint_results:
                lines.append('')
                lines.append(hint_results)

        return '\n'.join(lines)

    def _extract_hints(
        self,
        task_data: Any,
        extra_hints: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Extract memory_hints from task metadata, merged with any explicit hints."""
        hints: dict[str, list] = {'queries': [], 'entities': []}

        # From task metadata
        if isinstance(task_data, dict):
            metadata = task_data.get('metadata', {})
            if isinstance(metadata, dict):
                task_hints = metadata.get('memory_hints', {})
                if isinstance(task_hints, dict):
                    hints['queries'].extend(task_hints.get('queries', []))
                    hints['entities'].extend(task_hints.get('entities', []))

        # From explicit hints
        if extra_hints:
            hints['queries'].extend(extra_hints.get('queries', []))
            hints['entities'].extend(extra_hints.get('entities', []))

        if not hints['queries'] and not hints['entities']:
            return None
        return hints

    async def _execute_hints(
        self,
        project_id: str,
        hints: dict[str, Any],
    ) -> str | None:
        """Execute memory hints (queries + entity lookups) and format results."""
        lines = ['### Memory Hints\n']
        has_content = False

        # Execute search queries
        for query in hints.get('queries', []):
            try:
                results = await self.memory.search(
                    query=query, project_id=project_id, limit=3
                )
                if results:
                    has_content = True
                    lines.append(f'**Query: "{query}"**')
                    for r in results:
                        lines.append(f'  - {r.content}')
            except Exception as e:
                logger.error(f'Hint search failed for "{query}": {e}')

        # Execute entity lookups
        for entity_name in hints.get('entities', []):
            try:
                entity = await self.memory.get_entity(
                    name=entity_name, project_id=project_id
                )
                if entity and (entity.get('nodes') or entity.get('edges')):
                    has_content = True
                    lines.append(f'**Entity: "{entity_name}"**')
                    for node in entity.get('nodes', []):
                        if node.get('summary'):
                            lines.append(f'  - {node["name"]}: {node["summary"]}')
                        elif node.get('name'):
                            lines.append(f'  - {node["name"]}')
                    for edge in entity.get('edges', []):
                        if edge.get('fact'):
                            lines.append(f'  - {edge["fact"]}')
            except Exception as e:
                logger.error(f'Hint entity lookup failed for "{entity_name}": {e}')

        if not has_content:
            return None
        return '\n'.join(lines)

    async def _build_task_tree_section(self, project_root: str) -> str | None:
        """Fetch and format the task tree."""
        if not self.taskmaster:
            return None

        try:
            result = await self.taskmaster.get_tasks(project_root=project_root)
        except Exception as e:
            logger.error(f'Failed to fetch task tree: {e}')
            return None

        tasks = result.get('tasks', []) if isinstance(result, dict) else []
        if not tasks:
            return None

        lines = ['## Task Tree\n']
        for task in tasks:
            if isinstance(task, dict):
                tid = task.get('id', '?')
                title = task.get('title', 'Untitled')
                status = task.get('status', 'unknown')
                lines.append(f'- [{status}] {tid}: {title}')

                # Subtasks
                for sub in task.get('subtasks', []):
                    if isinstance(sub, dict):
                        sid = sub.get('id', '?')
                        stitle = sub.get('title', 'Untitled')
                        sstatus = sub.get('status', 'unknown')
                        lines.append(f'  - [{sstatus}] {sid}: {stitle}')

        return '\n'.join(lines)
