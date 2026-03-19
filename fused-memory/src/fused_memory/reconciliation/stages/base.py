"""Base class for reconciliation pipeline stages."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import (
    ReconciliationEvent,
    StageId,
    StageReport,
    Watermark,
)
from fused_memory.reconciliation.agent_loop import AgentLoop, ToolDefinition

if TYPE_CHECKING:
    from fused_memory.backends.taskmaster_client import TaskmasterBackend
    from fused_memory.reconciliation.journal import ReconciliationJournal
    from fused_memory.services.memory_service import MemoryService

logger = logging.getLogger(__name__)


class BaseStage:
    """Base class for reconciliation pipeline stages."""

    def __init__(
        self,
        stage_id: StageId,
        memory_service: MemoryService,
        taskmaster: TaskmasterBackend | None,
        journal: ReconciliationJournal,
        config: ReconciliationConfig,
    ):
        self.stage_id = stage_id
        self.memory = memory_service
        self.taskmaster = taskmaster
        self.journal = journal
        self.config = config
        self.project_id: str = ''
        self.project_root: str = ''

    def get_tools(self) -> dict[str, ToolDefinition]:
        """Override in subclass — return tools available to this stage."""
        raise NotImplementedError

    def get_system_prompt(self) -> str:
        """Override in subclass."""
        raise NotImplementedError

    async def assemble_payload(
        self,
        events: list[ReconciliationEvent],
        watermark: Watermark,
        prior_reports: list[StageReport],
    ) -> str:
        """Override in subclass — build structured initial context."""
        raise NotImplementedError

    async def run(
        self,
        events: list[ReconciliationEvent],
        watermark: Watermark,
        prior_reports: list[StageReport],
        run_id: str,
    ) -> StageReport:
        """Execute this stage via agent loop."""
        payload = await self.assemble_payload(events, watermark, prior_reports)
        tools = self.get_tools()

        # Add terminal tool
        tools['stage_complete'] = ToolDefinition(
            name='stage_complete',
            description=(
                'Signal that this stage is complete. Provide a structured report with keys: '
                '"flagged_items" (list of items for next stage), "stats" (dict of counts), '
                '"summary" (string).'
            ),
            parameters={
                'type': 'object',
                'properties': {
                    'report': {
                        'type': 'object',
                        'properties': {
                            'flagged_items': {'type': 'array', 'items': {'type': 'object'}},
                            'stats': {'type': 'object'},
                            'summary': {'type': 'string'},
                        },
                    }
                },
                'required': ['report'],
            },
            function=lambda **kw: kw,
            is_mutation=False,
            target_system='reconciliation',
        )

        agent = AgentLoop(
            config=self.config,
            system_prompt=self.get_system_prompt(),
            tools=tools,
            terminal_tool='stage_complete',
        )

        started = datetime.now(UTC)
        try:
            result, journal_entries = await asyncio.wait_for(
                agent.run(payload),
                timeout=self.config.stage_timeout_seconds,
            )
        except TimeoutError:
            logger.error(
                f'Stage {self.stage_id.value} timed out after {self.config.stage_timeout_seconds}s'
            )
            result = {'warning': 'stage_timeout'}
            journal_entries = agent._journal_entries
        completed = datetime.now(UTC)

        # Persist journal entries
        for entry in journal_entries:
            entry.run_id = run_id
            entry.stage = self.stage_id
            await self.journal.add_entry(entry)

        report_data = result.get('report', {})

        stage_report = StageReport(
            stage=self.stage_id,
            started_at=started,
            completed_at=completed,
            actions_taken=[e.model_dump(mode='json') for e in journal_entries],
            items_flagged=report_data.get('flagged_items', []),
            stats=report_data.get('stats', {}),
            llm_calls=agent.llm_call_count,
            tokens_used=agent.token_count,
        )

        duration = (completed - started).total_seconds()
        logger.info(
            'reconciliation.stage_completed',
            extra={
                'run_id': run_id,
                'stage': self.stage_id.value,
                'duration_seconds': round(duration, 1),
                'actions_taken': len(journal_entries),
                'llm_calls': agent.llm_call_count,
                'tokens_used': agent.token_count,
            },
        )

        return stage_report

    # ── Shared tool builders ───────────────────────────────────────────

    def _memory_read_tools(self) -> dict[str, ToolDefinition]:
        """Memory read tools shared across stages."""
        tools: dict[str, ToolDefinition] = {}
        project_id = self.project_id

        async def search(query: str, limit: int = 10, categories: list[str] | None = None):
            return [
                r.model_dump()
                for r in await self.memory.search(
                    query=query, project_id=project_id, categories=categories, limit=limit
                )
            ]

        tools['search'] = ToolDefinition(
            name='search',
            description='Search across both memory stores (Graphiti + Mem0).',
            parameters={
                'type': 'object',
                'properties': {
                    'query': {'type': 'string', 'description': 'Natural language query'},
                    'limit': {'type': 'integer', 'default': 10},
                    'categories': {
                        'type': 'array',
                        'items': {'type': 'string'},
                        'description': 'Filter to specific taxonomy categories',
                    },
                },
                'required': ['query'],
            },
            function=search,
        )

        async def get_entity(name: str):
            return await self.memory.get_entity(name=name, project_id=project_id)

        tools['get_entity'] = ToolDefinition(
            name='get_entity',
            description='Look up an entity in the knowledge graph. Returns nodes + edges.',
            parameters={
                'type': 'object',
                'properties': {
                    'name': {'type': 'string', 'description': 'Entity name'},
                },
                'required': ['name'],
            },
            function=get_entity,
        )

        async def get_episodes(last_n: int = 20):
            return await self.memory.get_episodes(project_id=project_id, last_n=last_n)

        tools['get_episodes'] = ToolDefinition(
            name='get_episodes',
            description='Retrieve recent episodes from the knowledge graph.',
            parameters={
                'type': 'object',
                'properties': {
                    'last_n': {'type': 'integer', 'default': 20},
                },
            },
            function=get_episodes,
        )

        async def get_status():
            return await self.memory.get_status(project_id=project_id)

        tools['get_status'] = ToolDefinition(
            name='get_status',
            description='Health check and statistics for both memory backends.',
            parameters={'type': 'object', 'properties': {}},
            function=get_status,
        )

        return tools

    def _memory_write_tools(self) -> dict[str, ToolDefinition]:
        """Memory write (mutation) tools."""
        tools: dict[str, ToolDefinition] = {}
        project_id = self.project_id

        async def add_memory(
            content: str,
            category: str | None = None,
            metadata: dict | None = None,
        ):
            result = await self.memory.add_memory(
                content=content, category=category, project_id=project_id, metadata=metadata
            )
            return result.model_dump()

        tools['add_memory'] = ToolDefinition(
            name='add_memory',
            description='Write a classified memory to the appropriate store.',
            parameters={
                'type': 'object',
                'properties': {
                    'content': {'type': 'string', 'description': 'The memory content'},
                    'category': {
                        'type': 'string',
                        'description': 'Category (auto-classified if omitted)',
                    },
                    'metadata': {'type': 'object', 'description': 'Arbitrary metadata'},
                },
                'required': ['content'],
            },
            function=add_memory,
            is_mutation=True,
            target_system='memory',
        )

        async def delete_memory(memory_id: str, store: str):
            return await self.memory.delete_memory(
                memory_id=memory_id, store=store, project_id=project_id
            )

        tools['delete_memory'] = ToolDefinition(
            name='delete_memory',
            description='Delete a specific memory from a store.',
            parameters={
                'type': 'object',
                'properties': {
                    'memory_id': {'type': 'string'},
                    'store': {'type': 'string', 'enum': ['graphiti', 'mem0']},
                },
                'required': ['memory_id', 'store'],
            },
            function=delete_memory,
            is_mutation=True,
            target_system='memory',
        )

        return tools

    def _task_read_tools(self) -> dict[str, ToolDefinition]:
        """Task read tools (requires taskmaster)."""
        if not self.taskmaster:
            return {}
        tools: dict[str, ToolDefinition] = {}
        taskmaster = self.taskmaster
        project_root = self.project_root or self.project_id

        async def get_tasks():
            return await taskmaster.get_tasks(project_root=project_root)

        tools['get_tasks'] = ToolDefinition(
            name='get_tasks',
            description='List all tasks in the project.',
            parameters={'type': 'object', 'properties': {}},
            function=get_tasks,
        )

        async def get_task(id: str):
            return await taskmaster.get_task(task_id=id, project_root=project_root)

        tools['get_task'] = ToolDefinition(
            name='get_task',
            description='Get a single task by ID.',
            parameters={
                'type': 'object',
                'properties': {
                    'id': {'type': 'string', 'description': 'Task ID'},
                },
                'required': ['id'],
            },
            function=get_task,
        )

        return tools

    def _task_write_tools(self) -> dict[str, ToolDefinition]:
        """Task write (mutation) tools (requires taskmaster)."""
        if not self.taskmaster:
            return {}
        tools: dict[str, ToolDefinition] = {}
        taskmaster = self.taskmaster
        project_root = self.project_root or self.project_id

        async def set_task_status(id: str, status: str):
            return await taskmaster.set_task_status(
                task_id=id, status=status, project_root=project_root
            )

        tools['set_task_status'] = ToolDefinition(
            name='set_task_status',
            description='Update task status.',
            parameters={
                'type': 'object',
                'properties': {
                    'id': {'type': 'string'},
                    'status': {
                        'type': 'string',
                        'enum': ['pending', 'done', 'in-progress', 'review', 'deferred', 'cancelled'],
                    },
                },
                'required': ['id', 'status'],
            },
            function=set_task_status,
            is_mutation=True,
            target_system='taskmaster',
        )

        async def task_add_task(prompt: str | None = None, title: str | None = None):
            return await taskmaster.add_task(
                prompt=prompt, title=title, project_root=project_root
            )

        tools['add_task'] = ToolDefinition(
            name='add_task',
            description='Create a new task.',
            parameters={
                'type': 'object',
                'properties': {
                    'prompt': {'type': 'string', 'description': 'Task description for AI generation'},
                    'title': {'type': 'string', 'description': 'Manual task title'},
                },
            },
            function=task_add_task,
            is_mutation=True,
            target_system='taskmaster',
        )

        async def task_update_task(id: str, prompt: str | None = None, metadata: str | None = None):
            return await taskmaster.update_task(
                task_id=id, prompt=prompt, metadata=metadata, project_root=project_root
            )

        tools['update_task'] = ToolDefinition(
            name='update_task',
            description='Update an existing task.',
            parameters={
                'type': 'object',
                'properties': {
                    'id': {'type': 'string'},
                    'prompt': {'type': 'string'},
                    'metadata': {'type': 'string', 'description': 'JSON metadata to merge'},
                },
                'required': ['id'],
            },
            function=task_update_task,
            is_mutation=True,
            target_system='taskmaster',
        )

        async def task_add_subtask(
            parent_id: str, title: str | None = None, description: str | None = None
        ):
            return await taskmaster.add_subtask(
                parent_id=parent_id, title=title, description=description, project_root=project_root
            )

        tools['add_subtask'] = ToolDefinition(
            name='add_subtask',
            description='Add a subtask to an existing task.',
            parameters={
                'type': 'object',
                'properties': {
                    'parent_id': {'type': 'string'},
                    'title': {'type': 'string'},
                    'description': {'type': 'string'},
                },
                'required': ['parent_id'],
            },
            function=task_add_subtask,
            is_mutation=True,
            target_system='taskmaster',
        )

        async def task_remove_task(id: str):
            return await taskmaster.remove_task(task_id=id, project_root=project_root)

        tools['remove_task'] = ToolDefinition(
            name='remove_task',
            description='Remove a task.',
            parameters={
                'type': 'object',
                'properties': {
                    'id': {'type': 'string'},
                },
                'required': ['id'],
            },
            function=task_remove_task,
            is_mutation=True,
            target_system='taskmaster',
        )

        async def add_dependency(id: str, depends_on: str):
            return await taskmaster.add_dependency(
                task_id=id, depends_on=depends_on, project_root=project_root
            )

        tools['add_dependency'] = ToolDefinition(
            name='add_dependency',
            description='Add a dependency between tasks.',
            parameters={
                'type': 'object',
                'properties': {
                    'id': {'type': 'string'},
                    'depends_on': {'type': 'string'},
                },
                'required': ['id', 'depends_on'],
            },
            function=add_dependency,
            is_mutation=True,
            target_system='taskmaster',
        )

        async def remove_dependency(id: str, depends_on: str):
            return await taskmaster.remove_dependency(
                task_id=id, depends_on=depends_on, project_root=project_root
            )

        tools['remove_dependency'] = ToolDefinition(
            name='remove_dependency',
            description='Remove a dependency between tasks.',
            parameters={
                'type': 'object',
                'properties': {
                    'id': {'type': 'string'},
                    'depends_on': {'type': 'string'},
                },
                'required': ['id', 'depends_on'],
            },
            function=remove_dependency,
            is_mutation=True,
            target_system='taskmaster',
        )

        async def attach_memory_hints(task_id: str, entities: list[str] | None = None, queries: list[str] | None = None):
            import json as json_mod
            hints = {'entities': entities or [], 'queries': queries or []}
            return await taskmaster.update_task(
                task_id=task_id,
                metadata=json_mod.dumps({'memory_hints': hints}),
                project_root=project_root,
            )

        tools['attach_memory_hints'] = ToolDefinition(
            name='attach_memory_hints',
            description='Attach memory retrieval hints to a task. Use entity references and semantic queries, NOT inline content.',
            parameters={
                'type': 'object',
                'properties': {
                    'task_id': {'type': 'string'},
                    'entities': {'type': 'array', 'items': {'type': 'string'}},
                    'queries': {'type': 'array', 'items': {'type': 'string'}},
                },
                'required': ['task_id'],
            },
            function=attach_memory_hints,
            is_mutation=True,
            target_system='taskmaster',
        )

        return tools
