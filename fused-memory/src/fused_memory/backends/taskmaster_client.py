"""MCP client proxy to Taskmaster AI server."""

import asyncio
import contextlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import TextContent

from fused_memory.config.schema import TaskmasterConfig

logger = logging.getLogger(__name__)


class TaskmasterBackend:
    """Connects to Taskmaster's MCP server and proxies tool calls."""

    def __init__(self, config: TaskmasterConfig, reconnect_cooldown_seconds: float = 30.0):
        self.config = config
        self.reconnect_cooldown_seconds = reconnect_cooldown_seconds
        self._session: ClientSession | None = None
        self._stdio_ctx = None
        self._session_ctx = None
        self._last_reconnect_attempt: float = 0.0
        self._reconnect_lock = asyncio.Lock()

    @property
    def connected(self) -> bool:
        """Whether the client has an active session."""
        return self._session is not None

    async def ensure_connected(self) -> None:
        """Reconnect to Taskmaster if not connected, respecting cooldown."""
        if self._session is not None:
            return

        async with self._reconnect_lock:
            # Re-check inside lock (another coroutine may have reconnected)
            if self._session is not None:
                return

            now = time.monotonic()
            elapsed = now - self._last_reconnect_attempt
            if self._last_reconnect_attempt > 0 and elapsed < self.reconnect_cooldown_seconds:
                remaining = self.reconnect_cooldown_seconds - elapsed
                raise RuntimeError(
                    f'Taskmaster reconnection on cooldown ({remaining:.0f}s remaining)'
                )

            self._last_reconnect_attempt = now
            try:
                await self.initialize()
                logger.info('Taskmaster reconnected successfully')
            except Exception as exc:
                raise RuntimeError(f'Taskmaster reconnection failed: {exc}') from exc

    async def initialize(self) -> None:
        """Start Taskmaster MCP server process and establish client session."""
        if self.config.transport == 'stdio':
            env = None
            if self.config.tool_mode:
                import os

                env = {
                    **os.environ,
                    'TASK_MASTER_TOOLS': self.config.tool_mode,
                    'TASK_MASTER_ALLOW_METADATA_UPDATES': 'true',
                }

            # Use project_root as subprocess CWD so relative paths in args
            # (e.g. ./taskmaster-ai/dist/mcp-server.js) resolve correctly
            # regardless of where the fused-memory server process was started.
            cwd = self.config.cwd or self.config.project_root or None
            if cwd:
                cwd = str(Path(cwd).resolve())
            server_params = StdioServerParameters(
                command=self.config.command,
                args=self.config.args,
                cwd=cwd,
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
            with contextlib.suppress(Exception):
                await self._session_ctx.__aexit__(None, None, None)
            self._session_ctx = None
        if self._stdio_ctx:
            with contextlib.suppress(Exception):
                await self._stdio_ctx.__aexit__(None, None, None)
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
        try:
            result = await session.call_tool(name, arguments)
        except (BrokenPipeError, ConnectionError, EOFError, OSError):
            await self._cleanup_contexts()
            raise
        # MCP tool results come as content blocks
        if result.content:
            text_parts = [
                block.text for block in result.content if isinstance(block, TextContent)
            ]
            combined = '\n'.join(text_parts)
            try:
                return json.loads(combined)
            except (json.JSONDecodeError, ValueError):
                return {'text': combined}
        return {}

    # ── Convenience methods ────────────────────────────────────────────

    def _base_args(self, project_root: str, tag: str | None = None) -> dict:
        if not project_root or not os.path.isabs(project_root):
            raise ValueError(
                'project_root is required and must be an absolute path; '
                f'got {project_root!r}'
            )
        logger.debug('taskmaster projectRoot=%s', project_root)
        args: dict[str, Any] = {'projectRoot': project_root}
        if tag:
            args['tag'] = tag
        return args

    async def get_tasks(
        self,
        project_root: str,
        tag: str | None = None,
    ) -> dict:
        """Get all tasks from Taskmaster.

        Returns the full raw response from the upstream Taskmaster MCP tool.
        Status filtering is the responsibility of the TaskInterceptor layer
        (middleware), not the backend. This keeps the backend as a faithful
        proxy of the upstream response.
        """
        args = self._base_args(project_root, tag)
        return await self.call_tool('get_tasks', args)

    async def get_task(
        self, task_id: str, project_root: str, tag: str | None = None
    ) -> dict:
        args = self._base_args(project_root, tag)
        args['id'] = task_id
        return await self.call_tool('get_task', args)

    async def set_task_status(
        self,
        task_id: str,
        status: str,
        project_root: str,
        tag: str | None = None,
    ) -> dict:
        args = self._base_args(project_root, tag)
        args['id'] = task_id
        args['status'] = status
        return await self.call_tool('set_task_status', args)

    async def add_task(
        self,
        project_root: str,
        prompt: str | None = None,
        title: str | None = None,
        description: str | None = None,
        details: str | None = None,
        dependencies: str | None = None,
        priority: str | None = None,
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
        project_root: str,
        prompt: str | None = None,
        metadata: str | None = None,
        append: bool = False,
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
        project_root: str,
        title: str | None = None,
        description: str | None = None,
        details: str | None = None,
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
        project_root: str,
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
        project_root: str,
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
        project_root: str,
        tag: str | None = None,
    ) -> dict:
        args = self._base_args(project_root, tag)
        args['id'] = task_id
        args['dependsOn'] = depends_on
        return await self.call_tool('remove_dependency', args)

    async def validate_dependencies(
        self, project_root: str, tag: str | None = None
    ) -> dict:
        args = self._base_args(project_root, tag)
        return await self.call_tool('validate_dependencies', args)

    async def expand_task(
        self,
        task_id: str,
        project_root: str,
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
        project_root: str,
        num_tasks: str | None = None,
        tag: str | None = None,
    ) -> dict:
        args = self._base_args(project_root, tag)
        args['input'] = input_path
        if num_tasks:
            args['numTasks'] = num_tasks
        return await self.call_tool('parse_prd', args)
