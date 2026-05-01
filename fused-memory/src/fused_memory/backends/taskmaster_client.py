"""MCP client proxy to Taskmaster AI server.

The Taskmaster session lives inside a long-running *supervisor task* that owns
the ``stdio_client`` and ``ClientSession`` async-context managers. Earlier
versions opened those contexts via raw ``__aenter__`` calls inside the
``run_server`` task — anyio binds the inner task groups' cancel scopes to the
opening task, so any internal failure (a dead read loop, a broken pipe write)
cancelled ``run_server`` and cascaded the entire process to ``exit 1``.

Moving the ``async with`` blocks lexically inside a dedicated task isolates
the cancel scope: an inner failure cancels only the supervisor, which logs
the exception, sleeps the reconnect cooldown, and reopens the session. The
HTTP listener and every other in-process subsystem keep running.
"""

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

# Transport-level exception classes that mean the stdio session is dead.
# When any of these fire inside ``call_tool``, the caller gets the original
# exception and ``_session_ready`` is cleared so subsequent calls block
# until the supervisor respawns the session.
_TRANSPORT_DEAD_EXCEPTIONS: tuple[type[BaseException], ...] = (
    BrokenPipeError,
    ConnectionError,
    EOFError,
    OSError,
    anyio.ClosedResourceError,
    anyio.BrokenResourceError,
)

# After this long without a healthy session, emit a single structured ERROR
# log so dashboard / escalation watcher can pick it up.
_ESCALATE_THRESHOLD_SECONDS = 180.0

# Default for how long ``call_tool`` waits for the session to come up before
# raising. Short by design: callers expect a fast failure when Taskmaster is
# down so reconciliation/retry layers can handle it.
_DEFAULT_SESSION_READY_TIMEOUT = 5.0

# Default for how long ``start()`` blocks waiting on the very first session.
# If the first connect fails inside this window, ``start()`` logs a warning
# and returns — the supervisor stays running and keeps retrying.
_DEFAULT_STARTUP_TIMEOUT = 30.0

# How long ``close()`` waits for the supervisor to exit cleanly before
# cancelling the task.
_SHUTDOWN_TIMEOUT_SECONDS = 5.0


class TaskmasterBackend:
    """Connects to Taskmaster's MCP server and proxies tool calls.

    The session is owned by a long-running supervisor task created in
    :meth:`start`. Public methods (``call_tool``, ``is_alive``, the
    ``connected`` property, the convenience wrappers) are unchanged from
    the prior context-manager-based implementation; only the internal
    machinery is different.
    """

    def __init__(
        self,
        config: TaskmasterConfig,
        reconnect_cooldown_seconds: float = 30.0,
        alive_cache_ttl_seconds: float = 2.0,
        probe_timeout_seconds: float = 2.0,
        session_ready_timeout_seconds: float = _DEFAULT_SESSION_READY_TIMEOUT,
        startup_timeout_seconds: float = _DEFAULT_STARTUP_TIMEOUT,
    ):
        self.config = config
        self.reconnect_cooldown_seconds = reconnect_cooldown_seconds
        self._alive_cache_ttl_seconds = alive_cache_ttl_seconds
        self._probe_timeout_seconds = probe_timeout_seconds
        self._session_ready_timeout = session_ready_timeout_seconds
        self._startup_timeout = startup_timeout_seconds

        # Built once in start(); cached to avoid repeating env construction
        # on every reconnect.
        self._server_params: StdioServerParameters | None = None

        # Supervisor lifecycle state.
        self._supervisor_task: asyncio.Task[None] | None = None
        self._session: ClientSession | None = None
        self._session_ready = asyncio.Event()
        self._shutdown_event = asyncio.Event()
        # Set by call_tool when it sees a transport-dead exception. The
        # supervisor parks on either ``_shutdown_event`` or ``_respawn_event``
        # — without this signal a clean-EOF child death (which does NOT fire
        # the inner task group's cancel scope) would leave the supervisor
        # parked forever.
        self._respawn_event = asyncio.Event()
        self._call_lock = asyncio.Lock()

        # Observability + 3-min escalation.
        self._down_since: float | None = None
        self._escalated: bool = False
        self._restart_count: int = 0
        self._last_error_summary: str | None = None

        # (alive, error_msg, monotonic_timestamp)
        self._alive_cache: tuple[bool, str | None, float] | None = None

    # ── Public lifecycle ───────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        """Whether a Taskmaster session is currently up.

        Reads through ``_session_ready`` rather than checking ``_session``
        directly because ``call_tool`` may have cleared the event after
        observing a transport-dead exception even if the supervisor
        hasn't yet noticed.
        """
        return self._session_ready.is_set()

    @property
    def restart_count(self) -> int:
        """Number of times the supervisor has successfully opened a session."""
        return self._restart_count

    async def start(self) -> None:
        """Launch the supervisor task and wait briefly for the first session.

        If the first session does not come up within ``startup_timeout``,
        returns without raising. The supervisor keeps running and will
        retry. Callers that hit ``call_tool`` before the first connect see
        :class:`asyncio.TimeoutError`, consistent with the prior
        ``RuntimeError('Taskmaster not connected')`` surface.
        """
        if self._supervisor_task is not None and not self._supervisor_task.done():
            return
        if self.config.transport != 'stdio':
            raise ValueError(f'Unsupported Taskmaster transport: {self.config.transport}')

        self._server_params = self._build_server_params()
        self._shutdown_event.clear()
        self._session_ready.clear()
        self._respawn_event.clear()
        self._supervisor_task = asyncio.create_task(
            self._supervisor_loop(),
            name='taskmaster-supervisor',
        )
        try:
            await asyncio.wait_for(
                self._session_ready.wait(),
                timeout=self._startup_timeout,
            )
            logger.info('Taskmaster MCP client connected via stdio')
        except TimeoutError:
            logger.warning(
                'Taskmaster start: first session did not come up within %.1fs; '
                'supervisor will keep retrying',
                self._startup_timeout,
            )

    # Back-compat alias — prior callers (and tests) referenced ``initialize()``.
    async def initialize(self) -> None:
        """Alias for :meth:`start` (preserved for back-compat callers)."""
        await self.start()

    async def ensure_connected(self) -> None:
        """Wait for the supervisor to bring up a session.

        The previous implementation managed reconnect cooldowns inline; the
        cooldown now lives inside the supervisor loop, so this is a simple
        bounded wait. Raises :class:`asyncio.TimeoutError` if the session
        is not up within ``session_ready_timeout``.
        """
        if self._supervisor_task is None:
            raise RuntimeError('Taskmaster not started — call start() first')
        if self._session_ready.is_set():
            return
        await asyncio.wait_for(
            self._session_ready.wait(),
            timeout=self._session_ready_timeout,
        )

    async def close(self) -> None:
        """Shut down the supervisor and Taskmaster process.

        Sets ``_shutdown_event`` so the supervisor's parking ``await``
        returns and the loop exits cleanly. If the supervisor doesn't
        exit within ``_SHUTDOWN_TIMEOUT_SECONDS`` (e.g. a stuck cleanup),
        the task is cancelled.
        """
        if self._supervisor_task is None:
            return
        self._shutdown_event.set()
        self._session_ready.clear()
        try:
            await asyncio.wait_for(
                asyncio.shield(self._supervisor_task),
                timeout=_SHUTDOWN_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            self._supervisor_task.cancel()
            with contextlib.suppress(BaseException):
                await asyncio.wait_for(self._supervisor_task, timeout=2.0)
        except BaseException:
            # Supervisor finished with its own exception; nothing else to do.
            pass
        finally:
            self._supervisor_task = None
            self._session = None
            self._session_ready.clear()
        logger.info('Taskmaster MCP client disconnected')

    # ── Supervisor internals ───────────────────────────────────────────

    def _build_server_params(self) -> StdioServerParameters:
        env = None
        if self.config.tool_mode:
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
        return StdioServerParameters(
            command=self.config.command,
            args=self.config.args,
            cwd=cwd,
            env=env,
        )

    async def _supervisor_loop(self) -> None:
        """Long-lived task that owns the stdio + ClientSession contexts.

        The cancel scopes of ``stdio_client`` and ``ClientSession`` bind
        to *this* task. When an internal subtask in either group fails,
        only this task is cancelled — ``run_server`` is no longer in the
        chain. The loop catches the failure, logs, sleeps the cooldown,
        and reopens.
        """
        assert self._server_params is not None  # set by start()
        loop = asyncio.get_running_loop()

        while not self._shutdown_event.is_set():
            try:
                async with (
                    stdio_client(self._server_params) as (read_stream, write_stream),
                    ClientSession(read_stream, write_stream) as session,
                ):
                    await session.initialize()
                    self._session = session
                    self._down_since = None
                    self._escalated = False
                    self._respawn_event.clear()
                    self._restart_count += 1
                    self._session_ready.set()
                    logger.info(
                        'Taskmaster session up (restart #%d)',
                        self._restart_count,
                    )
                    # Park here. Two paths can wake us:
                    #   (a) An inner subtask raises and anyio fires the
                    #       cancel scope on this task — surfaces as
                    #       CancelledError or BaseExceptionGroup.
                    #   (b) call_tool detects a transport-dead exception
                    #       (or McpError REQUEST_TIMEOUT) and sets
                    #       ``_respawn_event`` — clean child-death
                    #       (process killed while idle) does NOT fire
                    #       the inner cancel scope, so we need this
                    #       explicit signal to know we should respawn.
                    await self._wait_for_wakeup()
            except asyncio.CancelledError:
                # close() called supervisor_task.cancel(); honour it.
                self._session = None
                self._session_ready.clear()
                raise
            except BaseException as exc:
                self._last_error_summary = f'{type(exc).__name__}: {exc}'
                logger.exception(
                    'Taskmaster supervisor: session died, will respawn',
                )
            finally:
                self._session = None
                self._session_ready.clear()

            if self._shutdown_event.is_set():
                break
            if self._down_since is None:
                self._down_since = loop.time()
            self._maybe_escalate(now=loop.time())
            # Sleep up to the cooldown, but break out early if shutdown
            # arrives mid-sleep — without this, ``close()`` would block
            # for the full cooldown.
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.reconnect_cooldown_seconds,
                )
        logger.info('Taskmaster supervisor exiting')

    async def _wait_for_wakeup(self) -> None:
        """Park until either ``_shutdown_event`` or ``_respawn_event`` fires.

        Implements the two-event wait without leaking detached tasks: each
        helper future is created, awaited via :func:`asyncio.wait`, and any
        survivor is cancelled and reaped before returning.
        """
        shutdown_t = asyncio.create_task(self._shutdown_event.wait())
        respawn_t = asyncio.create_task(self._respawn_event.wait())
        try:
            await asyncio.wait(
                [shutdown_t, respawn_t],
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            for t in (shutdown_t, respawn_t):
                if not t.done():
                    t.cancel()
                    with contextlib.suppress(BaseException):
                        await t

    def _maybe_escalate(self, *, now: float) -> None:
        """Emit a one-shot ERROR log if the session has been down ≥ threshold.

        The line is rate-limited via ``_escalated``; it re-arms when the
        supervisor next opens a session successfully (see
        :meth:`_supervisor_loop`). ``now`` is passed in (rather than read
        from the loop here) so unit tests can control the threshold without
        mocking the loop clock.
        """
        if self._escalated or self._down_since is None:
            return
        if now - self._down_since < _ESCALATE_THRESHOLD_SECONDS:
            return
        logger.error(
            'TASKMASTER_UNAVAILABLE_3MIN restart_attempts=%d last_error=%s',
            self._restart_count,
            self._last_error_summary or 'unknown',
        )
        self._escalated = True

    def _require_session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError('Taskmaster not connected — call start() first')
        return self._session

    # ── Tool dispatch ─────────────────────────────────────────────────

    async def call_tool(self, name: str, arguments: dict) -> Any:
        """Call a Taskmaster MCP tool and return the parsed result.

        Waits up to ``session_ready_timeout`` for the supervisor to bring
        up a session, then dispatches under ``self._call_lock``. The
        underlying MCP request-ID + stream-registration block is atomic
        at the asyncio level (see ``mcp/shared/session.py:240-313``); the
        lock is cheap insurance against subtle reordering after a
        cascade-driven outage.

        Includes a defensive retry across the supervisor-respawn boundary.
        Taskmaster has been observed to surface spurious tool-level
        failures (e.g. "No tasks found") for calls that land during the
        window where the stdio session is being torn down and respawned.
        If a call sees a transport-dead exception, an :class:`McpError`
        with ``REQUEST_TIMEOUT`` / ``CONNECTION_CLOSED``, or a tool-level
        ``Error:`` envelope **and** either ``restart_count`` advanced
        during the call or ``_session_ready`` was cleared during the
        call, the call is replayed up to two more times after waiting on
        :meth:`ensure_connected`. Backoffs: ``0.1s``, ``0.5s``.

        Malformed envelopes (``UNEXPECTED_RESPONSE_SHAPE`` — neither a
        ``data`` field nor an ``Error:`` text) are never retried; they
        signal a contract bug, not a transient transport condition. The
        envelope is returned as-is and the caller's ``_unwrap`` raises
        downstream.
        """
        await asyncio.wait_for(
            self._session_ready.wait(),
            timeout=self._session_ready_timeout,
        )

        last_envelope: Any = None
        last_exc: BaseException | None = None
        for attempt in range(3):
            snap_restart_count = self._restart_count
            snap_session_ready_was_set = self._session_ready.is_set()

            retry_signal = False
            last_envelope = None
            last_exc = None
            try:
                envelope = await self._dispatch(name, arguments)
            except _TRANSPORT_DEAD_EXCEPTIONS as exc:
                # Tear-down notice for the next caller AND the supervisor.
                # A clean-EOF child death does NOT fire the inner task
                # group's cancel scope, so the supervisor needs an
                # explicit nudge — without it the session would stay
                # stuck "ready" until something else cancels it.
                self._session_ready.clear()
                self._respawn_event.set()
                retry_signal = True
                last_exc = exc
            except McpError as exc:
                code = getattr(getattr(exc, 'error', None), 'code', None)
                if code in (CONNECTION_CLOSED, _REQUEST_TIMEOUT_CODE):
                    self._session_ready.clear()
                    self._respawn_event.set()
                    retry_signal = True
                    last_exc = exc
                else:
                    # Tool-level McpError from the proxy — not a transport
                    # failure. Re-raise as before.
                    raise
            else:
                last_envelope = envelope
                if (
                    isinstance(envelope, dict)
                    and isinstance(envelope.get('text'), str)
                    and envelope['text'].startswith('Error:')
                ):
                    # Tool-level Error envelope — eligible for retry only
                    # if the supervisor respawned mid-call.
                    retry_signal = True
                else:
                    # Either a clean success envelope or a malformed one.
                    # Both surface to the caller as-is — UNEXPECTED_
                    # RESPONSE_SHAPE is detected downstream in ``_unwrap``
                    # and is never retried.
                    if attempt > 0:
                        logger.info(
                            'taskmaster.respawn_recovered tool=%s '
                            'project_root=%s attempt=%d',
                            name,
                            self.config.project_root,
                            attempt,
                        )
                    return envelope

            if attempt == 2:
                break
            progressed = self._restart_count > snap_restart_count
            cleared = (
                snap_session_ready_was_set and not self._session_ready.is_set()
            )
            if not (retry_signal and (progressed or cleared)):
                break

            backoff = 0.1 if attempt == 0 else 0.5
            await asyncio.sleep(backoff)
            try:
                await self.ensure_connected()
            except (TimeoutError, asyncio.TimeoutError, RuntimeError):
                # Supervisor not up (RuntimeError) or didn't bring up a
                # session in time (TimeoutError). Give up retrying and
                # surface the original failure.
                break

        # Budget exhausted (or no respawn signal). Surface the last failure
        # exactly as it would have surfaced without the retry: re-raise
        # transport / McpError, return tool-error envelope so downstream
        # ``_unwrap`` raises ``TASKMASTER_TOOL_ERROR``.
        if last_exc is not None:
            raise last_exc
        return last_envelope

    async def _dispatch(self, name: str, arguments: dict) -> Any:
        """Single dispatch attempt — the IO half of :meth:`call_tool`.

        Acquires :attr:`_call_lock`, invokes the underlying
        ``session.call_tool``, and returns the parsed envelope (a JSON
        dict on success, ``{'text': ...}`` for non-JSON content, ``{}``
        for an empty content block). Exceptions propagate unchanged so
        the retry layer in :meth:`call_tool` can classify them.
        """
        session = self._session
        if session is None:
            # Window between ready being set and a concurrent close
            # clearing the session. Surface the same RuntimeError as
            # before so callers see a consistent contract.
            raise RuntimeError('Taskmaster not connected')
        async with self._call_lock:
            result = await session.call_tool(name, arguments)
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

        Result is cached briefly (``_alive_cache_ttl_seconds``) so recon
        hot paths don't stdio-round-trip on every ``get_status`` call.

        On a transport-level failure, ``call_tool`` has already cleared
        ``_session_ready`` — the next mutating call will block on the
        supervisor reopening. ``is_alive`` itself is read-only and does
        not attempt reconnect.
        """
        now = time.monotonic()
        if self._alive_cache is not None:
            cached_alive, cached_err, cached_at = self._alive_cache
            if now - cached_at < self._alive_cache_ttl_seconds:
                return cached_alive, cached_err

        result: tuple[bool, str | None]
        if not self._session_ready.is_set():
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
                # means call_tool already cleared ``_session_ready`` — report
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
