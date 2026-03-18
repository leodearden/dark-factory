"""MCP client proxy to Taskmaster AI server."""

import json
import logging
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from fused_memory.config.schema import TaskmasterConfig

logger = logging.getLogger(__name__)


class TaskmasterBackend:
    """Connects to Taskmaster's MCP server and proxies tool calls."""

    def __init__(self, config: TaskmasterConfig):
        self.config = config
        self._session: ClientSession | None = None
        self._stdio_ctx = None
        self._session_ctx = None

    async def initialize(self) -> None:
        """Start Taskmaster MCP server process and establish client session."""
        if self.config.transport == 'stdio':
            env = None
            if self.config.tool_mode:
                import os

                env = {**os.environ, 'TASK_MASTER_TOOLS': self.config.tool_mode}

            server_params = StdioServerParameters(
                command=self.config.command,
                args=self.config.args,
                cwd=self.config.cwd or None,
                env=env,
            )
            self._stdio_ctx = stdio_client(server_params)
            try:
                read_stream, write_stream = await self._stdio_ctx.__aenter__()
                self._session_ctx = ClientSession(read_stream, write_stream)
                self._session = await self._session_ctx.__aenter__()
                await self._session.initialize()
            except Exception:
                # Clean up partially opened contexts to prevent dangling cancel scopes
                await self._cleanup_contexts()
                raise
            logger.info('Taskmaster MCP client connected via stdio')
        else:
            raise ValueError(f'Unsupported Taskmaster transport: {self.config.transport}')

    async def _cleanup_contexts(self) -> None:
        """Best-effort cleanup of partially opened async contexts."""
        if self._session_ctx:
            try:
                await self._session_ctx.__aexit__(None, None, None)
            except Exception:
                pass
            self._session_ctx = None
        if self._stdio_ctx:
            try:
                await self._stdio_ctx.__aexit__(None, None, None)
            except Exception:
                pass
            self._stdio_ctx = None
        self._session = None

    async def close(self) -> None:
        """Shut down client session and server process."""
        if self._session_ctx:
            await self._session_ctx.__aexit__(None, None, None)
        if self._stdio_ctx:
            await self._stdio_ctx.__aexit__(None, None, None)
        self._session = None
        logger.info('Taskmaster MCP client disconnected')

    def _require_session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError('Taskmaster not connected — call initialize() first')
        return self._session

    async def call_tool(self, name: str, arguments: dict) -> Any:
        """Call a Taskmaster MCP tool and return the parsed result."""
        session = self._require_session()
        result = await session.call_tool(name, arguments)
        # MCP tool results come as content blocks
        if result.content:
            text_parts = [
                block.text for block in result.content if hasattr(block, 'text')
            ]
            combined = '\n'.join(text_parts)
            try:
                return json.loads(combined)
            except (json.JSONDecodeError, ValueError):
                return {'text': combined}
        return {}

    # ── Convenience methods ────────────────────────────────────────────

    def _base_args(self, project_root: str | None = None, tag: str | None = None) -> dict:
        args: dict[str, Any] = {}
        args['projectRoot'] = project_root or self.config.project_root
        if tag:
            args['tag'] = tag
        return args

    async def get_tasks(
        self, project_root: str | None = None, tag: str | None = None
    ) -> dict:
        args = self._base_args(project_root, tag)
        return await self.call_tool('get_tasks', args)

    async def get_task(
        self, task_id: str, project_root: str | None = None, tag: str | None = None
    ) -> dict:
        args = self._base_args(project_root, tag)
        args['id'] = task_id
        return await self.call_tool('get_task', args)

    async def set_task_status(
        self,
        task_id: str,
        status: str,
        project_root: str | None = None,
        tag: str | None = None,
    ) -> dict:
        args = self._base_args(project_root, tag)
        args['id'] = task_id
        args['status'] = status
        return await self.call_tool('set_task_status', args)

    async def add_task(
        self,
        prompt: str | None = None,
        title: str | None = None,
        description: str | None = None,
        details: str | None = None,
        dependencies: str | None = None,
        priority: str | None = None,
        project_root: str | None = None,
        tag: str | None = None,
    ) -> dict:
        args = self._base_args(project_root, tag)
        if prompt:
            args['prompt'] = prompt
        if title:
            args['title'] = title
        if description:
            args['description'] = description
        if details:
            args['details'] = details
        if dependencies:
            args['dependencies'] = dependencies
        if priority:
            args['priority'] = priority
        return await self.call_tool('add_task', args)

    async def update_task(
        self,
        task_id: str,
        prompt: str | None = None,
        metadata: str | None = None,
        append: bool = False,
        project_root: str | None = None,
        tag: str | None = None,
    ) -> dict:
        args = self._base_args(project_root, tag)
        args['id'] = task_id
        if prompt:
            args['prompt'] = prompt
        if metadata:
            args['metadata'] = metadata
        if append:
            args['append'] = True
        return await self.call_tool('update_task', args)

    async def add_subtask(
        self,
        parent_id: str,
        title: str | None = None,
        description: str | None = None,
        details: str | None = None,
        project_root: str | None = None,
        tag: str | None = None,
    ) -> dict:
        args = self._base_args(project_root, tag)
        args['id'] = parent_id
        if title:
            args['title'] = title
        if description:
            args['description'] = description
        if details:
            args['details'] = details
        return await self.call_tool('add_subtask', args)

    async def remove_task(
        self,
        task_id: str,
        project_root: str | None = None,
        tag: str | None = None,
    ) -> dict:
        args = self._base_args(project_root, tag)
        args['id'] = task_id
        args['confirm'] = True
        return await self.call_tool('remove_task', args)

    async def add_dependency(
        self,
        task_id: str,
        depends_on: str,
        project_root: str | None = None,
        tag: str | None = None,
    ) -> dict:
        args = self._base_args(project_root, tag)
        args['id'] = task_id
        args['dependsOn'] = depends_on
        return await self.call_tool('add_dependency', args)

    async def remove_dependency(
        self,
        task_id: str,
        depends_on: str,
        project_root: str | None = None,
        tag: str | None = None,
    ) -> dict:
        args = self._base_args(project_root, tag)
        args['id'] = task_id
        args['dependsOn'] = depends_on
        return await self.call_tool('remove_dependency', args)

    async def validate_dependencies(
        self, project_root: str | None = None, tag: str | None = None
    ) -> dict:
        args = self._base_args(project_root, tag)
        return await self.call_tool('validate_dependencies', args)

    async def expand_task(
        self,
        task_id: str,
        project_root: str | None = None,
        num: str | None = None,
        prompt: str | None = None,
        force: bool = False,
        tag: str | None = None,
    ) -> dict:
        args = self._base_args(project_root, tag)
        args['id'] = task_id
        if num:
            args['num'] = num
        if prompt:
            args['prompt'] = prompt
        if force:
            args['force'] = True
        return await self.call_tool('expand_task', args)

    async def parse_prd(
        self,
        input_path: str,
        project_root: str | None = None,
        num_tasks: str | None = None,
        tag: str | None = None,
    ) -> dict:
        args = self._base_args(project_root, tag)
        args['input'] = input_path
        if num_tasks:
            args['numTasks'] = num_tasks
        return await self.call_tool('parse_prd', args)
