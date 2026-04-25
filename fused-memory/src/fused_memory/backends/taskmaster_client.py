"""MCP client proxy to Taskmaster AI server."""

import asyncio
import contextlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import anyio
import httpx
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.shared.exceptions import McpError
from mcp.types import CONNECTION_CLOSED, TextContent

from fused_memory.backends.taskmaster_types import (
    AddSubtaskResult,
    AddTaskResult,
    DependencyResult,
    ExpandTaskResult,
    GetTasksResult,
    ParsePrdResult,
    RemoveTaskResult,
    SetTaskStatusResult,
    TaskmasterError,
    UpdateTaskResult,
    ValidateDependenciesResult,
)
from fused_memory.config.schema import TaskmasterConfig

logger = logging.getLogger(__name__)


def _unwrap(method_name: str, raw: Any) -> dict:
    """Validate a Taskmaster wire envelope and return its ``data`` field.

    Raises :class:`TaskmasterError` when the response is a tool-level
    failure (``createErrorResponse`` emits plain text starting with
    ``Error:``) or when the envelope is missing the expected ``data``
    key. Tool-level errors surface with ``code='TASKMASTER_TOOL_ERROR'``;
    shape problems with ``code='UNEXPECTED_RESPONSE_SHAPE'``.
    """
    if not isinstance(raw, dict):
        raise TaskmasterError(
            'UNEXPECTED_RESPONSE_SHAPE',
            f'{method_name}: non-dict response: {raw!r}',
            raw=raw,
        )
    # createErrorResponse path: text-only content, no JSON.
    text = raw.get('text')
    if isinstance(text, str) and text.startswith('Error:'):
        first_line = text.split('\n', 1)[0]
        message = first_line[len('Error:'):].strip() or 'taskmaster tool error'
        raise TaskmasterError('TASKMASTER_TOOL_ERROR', message, raw=raw)
    data = raw.get('data')
    if not isinstance(data, dict):
        raise TaskmasterError(
            'UNEXPECTED_RESPONSE_SHAPE',
            f"{method_name}: missing 'data' in envelope",
            raw=raw,
        )
    return data


def _require_field(method_name: str, data: dict, field: str, raw: Any) -> Any:
    """Return ``data[field]`` or raise UNEXPECTED_RESPONSE_SHAPE."""
    if field not in data:
        raise TaskmasterError(
            'UNEXPECTED_RESPONSE_SHAPE',
            f"{method_name}: 'data.{field}' missing",
            raw=raw,
        )
    return data[field]


# MCP session maps TimeoutError to an McpError with this HTTP-style code
# (see mcp/shared/session.py:296).
_REQUEST_TIMEOUT_CODE = int(httpx.codes.REQUEST_TIMEOUT)

# Transport-level exception classes that mean the stdio session is dead
# and should be torn down so ensure_connected() can reconnect.
_TRANSPORT_DEAD_EXCEPTIONS: tuple[type[BaseException], ...] = (
    BrokenPipeError,
    ConnectionError,
    EOFError,
    OSError,
    anyio.ClosedResourceError,
    anyio.BrokenResourceError,
)


class TaskmasterBackend:
    """Connects to Taskmaster's MCP server and proxies tool calls."""

    def __init__(
        self,
        config: TaskmasterConfig,
        reconnect_cooldown_seconds: float = 30.0,
        alive_cache_ttl_seconds: float = 2.0,
        probe_timeout_seconds: float = 2.0,
    ):
        self.config = config
        self.reconnect_cooldown_seconds = reconnect_cooldown_seconds
        self._session: ClientSession | None = None
        self._stdio_ctx = None
        self._session_ctx = None
        self._last_reconnect_attempt: float = 0.0
        self._reconnect_lock = asyncio.Lock()
        self._alive_cache_ttl_seconds = alive_cache_ttl_seconds
        self._probe_timeout_seconds = probe_timeout_seconds
        # (alive, error_msg, monotonic_timestamp)
        self._alive_cache: tuple[bool, str | None, float] | None = None

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
        await self._cleanup_contexts()
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
        except _TRANSPORT_DEAD_EXCEPTIONS:
            await self._cleanup_contexts()
            raise
        except McpError as exc:
            # Connection-closed and request-timeout live at the transport
            # layer even though they arrive wrapped in McpError — tear down
            # so the next mutating call can reconnect. Tool-level errors
            # (INTERNAL_ERROR, INVALID_PARAMS, …) mean the proxy is alive
            # and must pass through unchanged.
            code = getattr(getattr(exc, 'error', None), 'code', None)
            if code in (CONNECTION_CLOSED, _REQUEST_TIMEOUT_CODE):
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

    async def is_alive(self) -> tuple[bool, str | None]:
        """Report whether the Taskmaster proxy is live right now.

        Result is cached briefly (``_alive_cache_ttl_seconds``) so recon hot
        paths don't stdio-round-trip on every get_status call.

        On failure, ``call_tool``'s except branch tears down the session so
        the next mutating call can reconnect via ``ensure_connected``.
        ``is_alive`` itself is read-only and does not attempt reconnect.
        """
        now = time.monotonic()
        if self._alive_cache is not None:
            cached_alive, cached_err, cached_at = self._alive_cache
            if now - cached_at < self._alive_cache_ttl_seconds:
                return cached_alive, cached_err

        result: tuple[bool, str | None]
        if self._session is None:
            result = (False, 'not connected')
        else:
            try:
                with anyio.fail_after(self._probe_timeout_seconds):
                    await self.call_tool(
                        'get_tasks', {'projectRoot': self.config.project_root},
                    )
                result = (True, None)
            except McpError as exc:
                # Transport-level McpError (CONNECTION_CLOSED / REQUEST_TIMEOUT)
                # means call_tool already tore down the session — report
                # dead. Any other McpError is a tool-level response (e.g.
                # project not found, invalid params) which proves the proxy
                # is alive.
                code = getattr(getattr(exc, 'error', None), 'code', None)
                if code in (CONNECTION_CLOSED, _REQUEST_TIMEOUT_CODE):
                    result = (False, f'McpError: {exc}')
                else:
                    result = (True, None)
            except Exception as exc:
                result = (False, f'{type(exc).__name__}: {exc}')

        self._alive_cache = (result[0], result[1], now)
        return result

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
        self, project_root: str, tag: str | None = None
    ) -> GetTasksResult:
        args = self._base_args(project_root, tag)
        raw = await self.call_tool('get_tasks', args)
        data = _unwrap('get_tasks', raw)
        tasks = data.get('tasks', [])
        if not isinstance(tasks, list):
            raise TaskmasterError(
                'UNEXPECTED_RESPONSE_SHAPE',
                "get_tasks: 'data.tasks' is not a list",
                raw=raw,
            )
        return {'tasks': tasks}

    async def get_task(
        self, task_id: str, project_root: str, tag: str | None = None
    ) -> dict:
        """Return the task dict directly (Taskmaster puts the task at
        ``data`` for a single id, not ``data.task``).
        """
        args = self._base_args(project_root, tag)
        args['id'] = task_id
        raw = await self.call_tool('get_task', args)
        data = _unwrap('get_task', raw)
        if not isinstance(data, dict):
            raise TaskmasterError(
                'UNEXPECTED_RESPONSE_SHAPE',
                f'get_task: data is not a dict: {type(data).__name__}',
                raw=raw,
            )
        return data

    async def set_task_status(
        self,
        task_id: str,
        status: str,
        project_root: str,
        tag: str | None = None,
    ) -> SetTaskStatusResult:
        args = self._base_args(project_root, tag)
        args['id'] = task_id
        args['status'] = status
        raw = await self.call_tool('set_task_status', args)
        data = _unwrap('set_task_status', raw)
        return {
            'message': str(data.get('message', '')),
            'tasks': data.get('tasks') or [],
        }

    async def add_task(
        self,
        project_root: str,
        prompt: str | None = None,
        title: str | None = None,
        description: str | None = None,
        details: str | None = None,
        dependencies: str | None = None,
        priority: str | None = None,
        metadata: str | None = None,
        tag: str | None = None,
    ) -> AddTaskResult:
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
        if metadata:
            args['metadata'] = metadata
        raw = await self.call_tool('add_task', args)
        data = _unwrap('add_task', raw)
        task_id = _require_field('add_task', data, 'taskId', raw)
        return {
            'id': str(task_id),
            'message': str(data.get('message', '')),
        }

    async def update_task(
        self,
        task_id: str,
        project_root: str,
        prompt: str | None = None,
        metadata: str | None = None,
        append: bool = False,
        tag: str | None = None,
    ) -> UpdateTaskResult:
        args = self._base_args(project_root, tag)
        args['id'] = task_id
        if prompt:
            args['prompt'] = prompt
        if metadata:
            args['metadata'] = metadata
        if append:
            args['append'] = True
        raw = await self.call_tool('update_task', args)
        data = _unwrap('update_task', raw)
        wire_id = data.get('taskId', task_id)
        updated_task = data.get('updatedTask')
        if updated_task is not None and not isinstance(updated_task, dict):
            updated_task = None
        return {
            'id': str(wire_id),
            'message': str(data.get('message', '')),
            'updated': bool(data.get('updated', False)),
            'updated_task': updated_task,
        }

    async def add_subtask(
        self,
        parent_id: str,
        project_root: str,
        title: str | None = None,
        description: str | None = None,
        details: str | None = None,
        tag: str | None = None,
    ) -> AddSubtaskResult:
        args = self._base_args(project_root, tag)
        args['id'] = parent_id
        if title:
            args['title'] = title
        if description:
            args['description'] = description
        if details:
            args['details'] = details
        raw = await self.call_tool('add_subtask', args)
        data = _unwrap('add_subtask', raw)
        subtask = _require_field('add_subtask', data, 'subtask', raw)
        if not isinstance(subtask, dict):
            raise TaskmasterError(
                'UNEXPECTED_RESPONSE_SHAPE',
                'add_subtask: data.subtask is not a dict',
                raw=raw,
            )
        subtask_id = _require_field('add_subtask', subtask, 'id', raw)
        return {
            'id': str(subtask_id),
            'parent_id': str(parent_id),
            'message': str(data.get('message', '')),
            'subtask': subtask,
        }

    async def remove_task(
        self,
        task_id: str,
        project_root: str,
        tag: str | None = None,
    ) -> RemoveTaskResult:
        args = self._base_args(project_root, tag)
        args['id'] = task_id
        args['confirm'] = True
        raw = await self.call_tool('remove_task', args)
        data = _unwrap('remove_task', raw)
        removed = data.get('removedTasks', [])
        removed_ids: list[str] = []
        if isinstance(removed, list):
            for entry in removed:
                if isinstance(entry, dict) and 'id' in entry:
                    removed_ids.append(str(entry['id']))
                else:
                    removed_ids.append(str(entry))
        return {
            'successful': int(data.get('successful', 0) or 0),
            'failed': int(data.get('failed', 0) or 0),
            'removed_ids': removed_ids,
            'message': str(data.get('message', '')),
        }

    async def add_dependency(
        self,
        task_id: str,
        depends_on: str,
        project_root: str,
        tag: str | None = None,
    ) -> DependencyResult:
        args = self._base_args(project_root, tag)
        args['id'] = task_id
        args['dependsOn'] = depends_on
        raw = await self.call_tool('add_dependency', args)
        data = _unwrap('add_dependency', raw)
        return {
            'id': str(data.get('taskId', task_id)),
            'dependency_id': str(data.get('dependencyId', depends_on)),
            'message': str(data.get('message', '')),
        }

    async def remove_dependency(
        self,
        task_id: str,
        depends_on: str,
        project_root: str,
        tag: str | None = None,
    ) -> DependencyResult:
        args = self._base_args(project_root, tag)
        args['id'] = task_id
        args['dependsOn'] = depends_on
        raw = await self.call_tool('remove_dependency', args)
        data = _unwrap('remove_dependency', raw)
        return {
            'id': str(data.get('taskId', task_id)),
            'dependency_id': str(data.get('dependencyId', depends_on)),
            'message': str(data.get('message', '')),
        }

    async def validate_dependencies(
        self, project_root: str, tag: str | None = None
    ) -> ValidateDependenciesResult:
        args = self._base_args(project_root, tag)
        raw = await self.call_tool('validate_dependencies', args)
        data = _unwrap('validate_dependencies', raw)
        return {'message': str(data.get('message', ''))}

    async def expand_task(
        self,
        task_id: str,
        project_root: str,
        num: str | None = None,
        prompt: str | None = None,
        force: bool = False,
        tag: str | None = None,
    ) -> ExpandTaskResult:
        args = self._base_args(project_root, tag)
        args['id'] = task_id
        if num:
            args['num'] = num
        if prompt:
            args['prompt'] = prompt
        if force:
            args['force'] = True
        raw = await self.call_tool('expand_task', args)
        data = _unwrap('expand_task', raw)
        task = data.get('task')
        if not isinstance(task, dict):
            task = {}
        return {
            'task': task,
            'subtasks_added': int(data.get('subtasksAdded', 0) or 0),
            'has_existing_subtasks': bool(data.get('hasExistingSubtasks', False)),
        }

    async def parse_prd(
        self,
        input_path: str,
        project_root: str,
        num_tasks: str | None = None,
        tag: str | None = None,
    ) -> ParsePrdResult:
        args = self._base_args(project_root, tag)
        args['input'] = input_path
        if num_tasks:
            args['numTasks'] = num_tasks
        raw = await self.call_tool('parse_prd', args)
        data = _unwrap('parse_prd', raw)
        return {
            'output_path': str(data.get('outputPath', '')),
            'message': str(data.get('message', '')),
        }
