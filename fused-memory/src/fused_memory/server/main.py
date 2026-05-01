"""Entry point for the Fused Memory MCP server."""

import argparse
import asyncio
import contextlib
import logging
import os
import signal
import socket
import sys
import threading
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv

load_dotenv()

from fused_memory.config.schema import FusedMemoryConfig  # noqa: E402
from fused_memory.server.tools import create_mcp_server  # noqa: E402
from fused_memory.services.memory_service import MemoryService  # noqa: E402

if TYPE_CHECKING:
    from fused_memory.middleware.task_interceptor import TaskInterceptor
    from fused_memory.middleware.ticket_store import TicketStore
    from fused_memory.reconciliation.event_queue import EventQueue
    from fused_memory.reconciliation.harness import ReconciliationHarness
    from fused_memory.reconciliation.journal import ReconciliationJournal
    from fused_memory.reconciliation.sqlite_watchdog import SqliteWatchdog

# Logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    stream=sys.stderr,
)
logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
logging.getLogger('mcp.server.streamable_http_manager').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Maximum seconds to wait for harness_loop_task to honour cancellation before
# giving up and continuing with the rest of the shutdown sequence.
_HARNESS_CANCEL_TIMEOUT = 25.0

# Per-step cleanup budget inside _graceful_shutdown. Harness step uses
# _HARNESS_CANCEL_TIMEOUT instead; all other steps are capped by this.
_CLEANUP_STEP_TIMEOUT = 5.0

# Total wall-clock budget between _graceful_shutdown starting and a hard
# os._exit(1) firing from a background thread. Must exceed the sum of per-step
# timeouts plus headroom so clean shutdowns never trip it.
# Worst case (Fix A): drain(5)+close(5)+sqlite_watchdog(5)+event_queue(5)
# +harness_cancel(25)+memory_close(5)+journal_close(5) = 55s; +20s headroom = 75.
_FORCE_EXIT_BUDGET = 75.0

# systemd watchdog heartbeat interval. Must be comfortably less than
# WatchdogSec in the unit file (we use 30s there) so a single missed tick
# doesn't trigger a restart.
_WATCHDOG_INTERVAL = 10.0


def _sd_notify(state: str) -> None:
    """Send a single message to systemd's notify socket (no-op if unset).

    Implements the sd_notify protocol directly so we don't need a
    python-systemd / cysystemd dependency. Silent on failure: systemd
    integration is advisory and must never break local-dev runs.
    """
    addr = os.environ.get('NOTIFY_SOCKET')
    if not addr:
        return
    # Abstract-namespace sockets are advertised with a leading "@".
    if addr.startswith('@'):
        addr = '\0' + addr[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            sock.sendto(state.encode('utf-8'), addr)
    except OSError:
        logger.exception('sd_notify failed (state=%s)', state)


_shutdown_watchdog: threading.Timer | None = None

# Set by the operator-stop signal handler (Fix B). main() reads this in its
# finally clause to choose between exit 0 (operator wanted stop — let systemd
# leave the service stopped) and exit 1 (cascade — let Restart=on-failure fire).
_operator_stop_received: bool = False


def _arm_force_exit(budget_secs: float = _FORCE_EXIT_BUDGET) -> None:
    """Start a daemon timer that hard-exits the process after `budget_secs`.

    Idempotent. Runs in a separate OS thread so it fires even if the asyncio
    loop is wedged or deadlocked. Covers the case where _graceful_shutdown
    hangs or non-daemon third-party threads (e.g. mem0 PostHog consumers)
    keep the interpreter alive after asyncio.run() returns.
    """
    global _shutdown_watchdog
    if _shutdown_watchdog is not None:
        return

    def _force_exit() -> None:
        # Write directly to stderr — logger may already be torn down.
        sys.stderr.write(
            f'[fused-memory] shutdown watchdog fired after {budget_secs}s — os._exit(1)\n'
        )
        sys.stderr.flush()
        os._exit(1)

    t = threading.Timer(budget_secs, _force_exit)
    t.daemon = True
    t.start()
    _shutdown_watchdog = t


def _cancel_force_exit() -> None:
    """Cancel the force-exit watchdog (call after clean asyncio.run exit)."""
    global _shutdown_watchdog
    if _shutdown_watchdog is not None:
        _shutdown_watchdog.cancel()
        _shutdown_watchdog = None


async def _run_shielded(
    name: str,
    coro_factory: Callable[[], Awaitable[Any]],
    timeout: float = _CLEANUP_STEP_TIMEOUT,
) -> None:
    """Run a single cleanup step with asyncio.shield + bounded timeout.

    `coro_factory` is a zero-arg callable that returns the coroutine to run.
    Shielding decouples the cleanup task from the caller's cancellation, so
    cleanup actually makes progress even when _graceful_shutdown itself is
    running inside a cancelled task.

    On caller-cancel: we keep awaiting the shielded inner task (bounded by
    the step deadline) so cleanup completes instead of being abandoned as
    an orphan that hangs ``asyncio.run()``. If the deadline expires while
    the inner is still running, we explicitly cancel it.

    The force-exit watchdog is the ultimate backstop.
    """
    loop = asyncio.get_running_loop()
    inner_task = asyncio.ensure_future(coro_factory())
    deadline = loop.time() + timeout
    try:
        await asyncio.wait_for(asyncio.shield(inner_task), timeout=timeout)
    except asyncio.CancelledError:
        # Caller was cancelled. The shielded inner task continues running.
        # Keep waiting (bounded by remaining budget) using asyncio.wait —
        # which returns instead of raising on timeout, sidestepping the
        # re-cancel-on-every-await trap.
        while not inner_task.done():
            remaining = max(0.0, deadline - loop.time())
            if remaining <= 0:
                break
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait({inner_task}, timeout=remaining)
        if not inner_task.done():
            inner_task.cancel()
            with contextlib.suppress(BaseException):
                await asyncio.wait({inner_task}, timeout=1.0)
        logger.warning(
            '_graceful_shutdown: %s caller cancelled, finished waiting (done=%s)',
            name,
            inner_task.done(),
        )
        # Surface a non-cancelled failure of the inner task for observability.
        if inner_task.done() and not inner_task.cancelled():
            exc = inner_task.exception()
            if exc is not None and not isinstance(exc, asyncio.CancelledError):
                logger.exception(
                    '_graceful_shutdown: %s raised (after caller-cancel)',
                    name,
                    exc_info=exc,
                )
    except TimeoutError:
        # Step exceeded its budget — cancel the inner so it can't linger
        # as a detached orphan that wedges asyncio.run().
        inner_task.cancel()
        with contextlib.suppress(BaseException):
            await asyncio.wait({inner_task}, timeout=1.0)
        logger.warning('_graceful_shutdown: %s timed out after %.1fs', name, timeout)
    except Exception:
        logger.exception('_graceful_shutdown: %s raised', name)


class _ASGIExceptionShield:
    """ASGI middleware that contains BaseException escapes from the MCP app.

    Wraps mcp.streamable_http_app() so unhandled exceptions inside a request
    (including CancelledError from client disconnect, BaseExceptionGroup from
    anyio, etc.) cannot reach the StreamableHTTPSessionManager's shared task
    group. This is the direct defence against the 2026-04-23 wedge, where a
    client-disconnect CancelledError poisoned the shared task group and
    cascaded into uvicorn's main_loop.

    Trade-off: we swallow CancelledError at this boundary to break the
    cascade. That deviates from textbook asyncio semantics but is the only
    way to keep the server up when the MCP SDK's own except-Exception guard
    lets a BaseException through. KeyboardInterrupt/SystemExit are always
    re-raised so legitimate interpreter shutdown still works.
    """

    _RESPONSE_START = {
        'type': 'http.response.start',
        'status': 500,
        'headers': [(b'content-type', b'application/json')],
    }
    _RESPONSE_BODY = {
        'type': 'http.response.body',
        'body': b'{"error":"Internal Server Error"}',
    }

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope.get('type') != 'http':
            # lifespan / websocket — pass through untouched.
            await self.app(scope, receive, send)
            return
        response_started = False

        async def _tracking_send(message: dict) -> None:
            nonlocal response_started
            if message.get('type') == 'http.response.start':
                response_started = True
            await send(message)

        try:
            await self.app(scope, receive, _tracking_send)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            logger.exception(
                'MCP request failed (%s); emitting 500 and suppressing cascade',
                type(exc).__name__,
            )
            if not response_started:
                with contextlib.suppress(Exception):
                    await send(self._RESPONSE_START)
                    await send(self._RESPONSE_BODY)
            # Deliberate: do NOT re-raise. Letting CancelledError or any
            # other BaseException escape here is exactly what wedged the
            # server overnight.
            return


def configure_uvicorn_logging():
    for logger_name in ['uvicorn', 'uvicorn.error', 'uvicorn.access']:
        uv_logger = logging.getLogger(logger_name)
        uv_logger.handlers.clear()
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        uv_logger.addHandler(handler)
        uv_logger.propagate = False


async def run_server():
    """Parse args, load config, init service, run MCP transport."""
    parser = argparse.ArgumentParser(description='Fused Memory MCP Server')
    default_config = Path(__file__).resolve().parents[3] / 'config' / 'config.yaml'
    parser.add_argument(
        '--config', type=Path, default=None,
        help='Path to YAML configuration file',
    )
    parser.add_argument(
        '--transport', choices=['stdio', 'sse', 'http'],
        help='Transport override',
    )
    parser.add_argument(
        '--stateless', action='store_true', default=None,
        help='Enable stateless HTTP mode (no session tracking)',
    )
    parser.add_argument(
        '--json-response', action='store_true', default=None,
        help='Return JSON instead of SSE for HTTP transport',
    )
    args = parser.parse_args()

    # Set CONFIG_PATH for the settings source — prefer explicit --config,
    # then existing CONFIG_PATH env var, then the built-in default.
    if args.config:
        os.environ['CONFIG_PATH'] = str(args.config)
    elif 'CONFIG_PATH' not in os.environ:
        os.environ['CONFIG_PATH'] = str(default_config)

    config = FusedMemoryConfig()
    if args.transport:
        config.server.transport = args.transport
    if args.stateless is True:
        config.server.stateless_http = True
    if args.json_response is True:
        config.server.json_response = True

    logger.info('Fused Memory MCP Server starting')
    logger.info(f'  LLM: {config.llm.provider}/{config.llm.model}')
    logger.info(f'  Embedder: {config.embedder.provider}/{config.embedder.model}')
    logger.info(f'  Graphiti: {config.graphiti.provider} ({config.graphiti.falkordb.uri})')
    logger.info(f'  Mem0/Qdrant: {config.mem0.qdrant_url}')
    logger.info(f'  Transport: {config.server.transport}')

    # Initialize memory service
    memory_service = MemoryService(config)
    await memory_service.initialize()

    # Initialize write journal
    from fused_memory.services.write_journal import WriteJournal

    wj_data_dir = Path(config.reconciliation.data_dir) if config.reconciliation else Path('./data')
    write_journal = WriteJournal(wj_data_dir)
    await write_journal.initialize()
    memory_service.set_write_journal(write_journal)

    # Initialize task backend (Taskmaster MCP / SqliteTaskBackend / DualCompare).
    taskmaster = None
    task_interceptor = None
    if config.taskmaster:
        taskmaster = await _build_task_backend(config.taskmaster)
        # Wire the backend into memory_service before initialize() so
        # get_status reports an accurate connection state even if the
        # first initialize fails. is_alive() will probe the live session.
        memory_service.taskmaster = taskmaster
        try:
            # start() launches the supervisor task (Taskmaster) or opens the
            # SQLite connection lazily (SqliteTaskBackend); both return once
            # the backend is usable (or after the appropriate startup timeout
            # if not). For Taskmaster the supervisor owns the stdio +
            # ClientSession context managers — internal failures cancel only
            # the supervisor, not run_server.
            await taskmaster.start()
            logger.info(
                '  Task backend: %s (mode=%s)',
                type(taskmaster).__name__,
                config.taskmaster.backend_mode,
            )
        except Exception as e:
            logger.warning(
                '  Task backend: start failed (%s), will retry on next tool call', e,
            )

    # Initialize reconciliation system
    harness_loop_task = None
    recon_journal = None
    reconciliation_harness = None
    # Single curator escalator shared by both TaskInterceptor construction
    # paths — lazy per-project queue handles live inside it.
    from fused_memory.middleware.curator_escalator import CuratorEscalator
    from fused_memory.middleware.project_prefix_registry import ProjectPrefixRegistry
    from fused_memory.middleware.scope_violation_escalator import (
        ScopeViolationEscalator,
    )
    from fused_memory.models.scope import (
        build_known_projects_map,
        known_project_roots_from_env,
    )

    curator_escalator = CuratorEscalator()

    # Multi-project path-scope guard: built from the configured project +
    # DASHBOARD_KNOWN_PROJECT_ROOTS env var so the same registry covers
    # every project the dashboard already knows about.  Empty registry
    # (no extra roots) leaves the TaskInterceptor in back-compat mode
    # where the dark-factory-only guard runs.
    _primary_root = (
        config.taskmaster.project_root if config.taskmaster else ''
    ) or ''
    if _primary_root:
        _primary_root = str(Path(_primary_root).expanduser().resolve())
    _extra_roots = known_project_roots_from_env()
    _known_projects_map = build_known_projects_map(_primary_root, _extra_roots)
    if len(_known_projects_map) > 1:
        prefix_registry: ProjectPrefixRegistry | None = (
            ProjectPrefixRegistry.from_roots(list(_known_projects_map.values()))
        )
        scope_violation_escalator: ScopeViolationEscalator | None = (
            ScopeViolationEscalator()
        )
        logger.info(
            '  Path-scope guard: multi-project mode (%d projects)',
            len(_known_projects_map),
        )
    else:
        prefix_registry = None
        scope_violation_escalator = None
        logger.info(
            '  Path-scope guard: dark-factory-only back-compat mode '
            '(set DASHBOARD_KNOWN_PROJECT_ROOTS to enable multi-project)',
        )

    # Curator UsageGate — independent instance reading the same shared
    # accounts file the orchestrator / eval runner use. Protects the curator
    # from silent outages when its default account caps.
    curator_usage_gate = None
    if config.usage_cap is not None and config.usage_cap.enabled:
        from shared.usage_gate import UsageGate

        curator_usage_gate = UsageGate(config.usage_cap)
        logger.info(
            f'  Curator usage gate: {curator_usage_gate.account_count} account(s) '
            f'from {config.usage_cap.accounts_file or "inline"}',
        )

    event_queue = None
    sqlite_watchdog = None
    backlog_policy = None
    if config.reconciliation and config.reconciliation.enabled:
        from fused_memory.middleware.task_interceptor import TaskInterceptor
        from fused_memory.reconciliation.backlog_policy import BacklogPolicy
        from fused_memory.reconciliation.bulk_reset_guard import BulkResetGuard
        from fused_memory.reconciliation.event_buffer import EventBuffer
        from fused_memory.reconciliation.event_queue import EventQueue
        from fused_memory.reconciliation.harness import ReconciliationHarness
        from fused_memory.reconciliation.journal import ReconciliationJournal
        from fused_memory.reconciliation.sqlite_watchdog import SqliteWatchdog
        from fused_memory.reconciliation.targeted import TargetedReconciler
        from fused_memory.services.orchestrator_detector import (
            is_orchestrator_live_for,
        )

        recon_journal = ReconciliationJournal(Path(config.reconciliation.data_dir))
        await recon_journal.initialize()
        recon_journal.set_write_journal(write_journal)

        db_path = Path(config.reconciliation.data_dir) / 'reconciliation.db'
        assert memory_service.durable_queue is not None  # set by initialize()
        event_buffer = EventBuffer(
            db_path=db_path,
            buffer_size_threshold=config.reconciliation.buffer_size_threshold,
            max_staleness_seconds=config.reconciliation.max_staleness_seconds,
            conditional_trigger_ratio=config.reconciliation.conditional_trigger_ratio,
            burst_window_seconds=config.reconciliation.burst_window_seconds,
            burst_cooldown_seconds=config.reconciliation.burst_cooldown_seconds,
            stale_lock_seconds=config.reconciliation.stale_lock_seconds,
            queue_stats_fn=memory_service.durable_queue.get_stats if memory_service.durable_queue else None,
        )
        await event_buffer.initialize()

        # WP-B: in-memory fire-and-forget queue fronts the SQLite buffer so
        # the MCP hot path never awaits aiosqlite. Drainer persists events in
        # the background with retry on OperationalError.
        event_queue = EventQueue(
            event_buffer,
            dead_letter_path=(
                Path(config.reconciliation.data_dir) / 'event_dead_letter.jsonl'
            ),
            maxsize=config.reconciliation.event_queue_capacity,
            retry_initial_seconds=config.reconciliation.event_queue_retry_initial_seconds,
            retry_max_seconds=config.reconciliation.event_queue_retry_max_seconds,
            shutdown_flush_seconds=config.reconciliation.event_queue_shutdown_flush_seconds,
            max_bytes=config.reconciliation.event_dead_letter_max_bytes,
            keep_rotations=config.reconciliation.event_dead_letter_keep_rotations,
        )
        await event_queue.start()

        # WP-D: bounded-backlog escalation/rejection policy. Constructed
        # before the watchdog so the watchdog can call into it on wedge.
        backlog_policy = BacklogPolicy(
            event_buffer,
            event_queue,
            is_orchestrator_live_for,
            hard_limit=config.reconciliation.backlog_hard_limit,
            rate_limit_seconds=(
                config.reconciliation.backlog_escalation_rate_limit_seconds
            ),
        )

        # WP-C: watchdog over the drainer — ERROR-logs structured diagnostics
        # if the drainer stalls (no commit in N seconds with non-empty queue).
        # Surfaces the SQLite-lock condition that previously rotted silently.
        if config.reconciliation.event_queue_watchdog_enabled:
            # ``on_watchdog_wedge`` returns ``list[BacklogVerdict]`` for
            # tests/introspection, but ``WedgeCallback`` is typed as returning
            # ``Awaitable[None]`` — the watchdog discards the return value.
            # Wrap to satisfy the narrower callback type without changing
            # runtime behavior.
            async def _on_wedge(payload: dict) -> None:
                await backlog_policy.on_watchdog_wedge(payload)

            sqlite_watchdog = SqliteWatchdog(
                event_queue,
                check_interval_seconds=(
                    config.reconciliation.event_queue_watchdog_check_interval_seconds
                ),
                stall_threshold_seconds=(
                    config.reconciliation.event_queue_watchdog_stall_threshold_seconds
                ),
                rearm_after_seconds=(
                    config.reconciliation.event_queue_watchdog_rearm_after_seconds
                ),
                wedge_callback=_on_wedge,
            )
            await sqlite_watchdog.start()

        # Wire event emission into memory_service
        memory_service.set_event_buffer(event_buffer)

        # Targeted reconciler (needs memory_service + taskmaster + journal)
        targeted = None
        if taskmaster and taskmaster.connected:
            targeted = TargetedReconciler(
                memory_service, taskmaster, recon_journal, config, event_buffer,
            )
            # Wire the planned episode registry so _on_task_done can promote planned
            # episodes when tasks complete (otherwise the promotion code is dead code).
            targeted.planned_episode_registry = memory_service.planned_episode_registry

        from fused_memory.middleware.task_file_committer import TaskFileCommitter

        # Task 918: defence-in-depth bulk-reset circuit-breaker.  Constructed
        # here (inside the reconciliation branch) so it shares the same lifecycle
        # as the other reconciliation-layer guards.  When reconciliation is
        # disabled the guard is left as None in the TaskInterceptor (no guard).
        bulk_reset_guard = BulkResetGuard(
            enabled=config.reconciliation.bulk_reset_guard_enabled,
            done_threshold=config.reconciliation.bulk_reset_guard_done_to_pending_threshold,
            in_progress_threshold=config.reconciliation.bulk_reset_guard_in_progress_to_pending_threshold,
            window_seconds=config.reconciliation.bulk_reset_guard_window_seconds,
            escalation_rate_limit_seconds=(
                config.reconciliation.bulk_reset_guard_escalation_rate_limit_seconds
            ),
            write_failure_backoff_seconds=(
                config.reconciliation.bulk_reset_guard_write_failure_backoff_seconds
            ),
        )

        task_committer = TaskFileCommitter()
        ticket_store = await _build_ticket_store(Path(config.reconciliation.data_dir))
        task_interceptor = TaskInterceptor(
            taskmaster, targeted, event_buffer, task_committer,
            config=config, escalator=curator_escalator,
            event_queue=event_queue,
            backlog_policy=backlog_policy,
            usage_gate=curator_usage_gate,
            ticket_store=ticket_store,
            bulk_reset_guard=bulk_reset_guard,
            prefix_registry=prefix_registry,
            scope_violation_escalator=scope_violation_escalator,
        )
        await task_interceptor.start()

        # Full reconciliation harness (background loop)
        reconciliation_harness = ReconciliationHarness(
            memory_service, taskmaster, recon_journal, event_buffer, config,
            backlog_policy=backlog_policy,
        )
        harness_loop_task = asyncio.create_task(reconciliation_harness.run_loop())
        logger.info('  Reconciliation: enabled (background loop started)')

        # SIGUSR1 triggers harness drain (stop new cycles, let current ones finish)
        _register_drain_signal_handler(reconciliation_harness)
    else:
        # Always create task_interceptor for tool registration
        from fused_memory.middleware.task_file_committer import TaskFileCommitter
        from fused_memory.middleware.task_interceptor import TaskInterceptor
        from fused_memory.reconciliation.event_buffer import EventBuffer

        event_buffer = EventBuffer(db_path=None)
        await event_buffer.initialize()
        task_committer = TaskFileCommitter()
        _disabled_data_dir = Path(config.reconciliation.data_dir) if config.reconciliation else Path('./data')
        ticket_store = await _build_ticket_store(_disabled_data_dir)
        task_interceptor = TaskInterceptor(
            taskmaster, None, event_buffer, task_committer,
            config=config, escalator=curator_escalator,
            usage_gate=curator_usage_gate,
            ticket_store=ticket_store,
            prefix_registry=prefix_registry,
            scope_violation_escalator=scope_violation_escalator,
        )
        await task_interceptor.start()

    # Create MCP server with both memory and task tools
    mcp = create_mcp_server(
        memory_service, task_interceptor, write_journal,
        reconciliation_harness=reconciliation_harness,
        backlog_policy=backlog_policy,
        event_queue=event_queue,
    )

    # Defence-in-depth wrapper at FastMCP's central tool-dispatch chokepoint.
    # Catches BaseException escapes (SystemExit, BaseExceptionGroup, etc.) that
    # would otherwise poison StreamableHTTPSessionManager's shared task group
    # and cascade into uvicorn's main loop. Re-raises CancelledError because
    # it is required for asyncio cancellation semantics.
    _install_safe_tool_wrapper(mcp)

    mcp.settings.host = config.server.host
    mcp.settings.port = config.server.port
    mcp.settings.stateless_http = config.server.stateless_http
    mcp.settings.json_response = config.server.json_response

    # Thread monitor — log count every 60s, warn if growing unexpectedly
    async def _thread_monitor():
        prev = 0
        while True:
            await asyncio.sleep(60)
            count = threading.active_count()
            delta = count - prev
            if delta != 0 or count > 30:
                level = logging.WARNING if count > 30 else logging.INFO
                logger.log(level, f'thread_monitor: threads={count} delta={delta:+d}')
            prev = count

    asyncio.create_task(_thread_monitor())

    # Run transport
    transport = config.server.transport
    logger.info(f'Starting MCP server with transport: {transport}')

    watchdog_task: asyncio.Task[None] | None = None
    try:
        if transport == 'stdio':
            await mcp.run_stdio_async()
        elif transport == 'sse':
            await mcp.run_sse_async()
        elif transport == 'http':
            import uvicorn

            display_host = 'localhost' if config.server.host == '0.0.0.0' else config.server.host
            logger.info(f'  MCP Endpoint: http://{display_host}:{config.server.port}/mcp/')
            configure_uvicorn_logging()
            starlette_app = mcp.streamable_http_app()

            # Return JSON (not plain-text) for 404s so that MCP SDK
            # clients attempting OAuth discovery against well-known
            # endpoints don't crash on JSON.parse("Not Found").
            from starlette.exceptions import HTTPException
            from starlette.requests import Request as StarletteRequest
            from starlette.responses import JSONResponse

            async def _json_http_error(request: StarletteRequest, exc: HTTPException) -> JSONResponse:
                return JSONResponse({'error': exc.detail}, status_code=exc.status_code)

            starlette_app.add_exception_handler(HTTPException, _json_http_error)  # type: ignore[arg-type]

            # Outermost ASGI layer: catches any BaseException escaping the
            # MCP app before it can poison the SDK's shared task group.
            shielded_app = _ASGIExceptionShield(starlette_app)

            uv_config = uvicorn.Config(
                shielded_app,
                host=config.server.host,
                port=config.server.port,
                log_level='info',
                timeout_keep_alive=config.server.keepalive_timeout,
            )
            server = uvicorn.Server(uv_config)

            # Take ownership of SIGTERM/SIGINT before uvicorn's serve() can
            # install its own handlers. We need to differentiate operator-stop
            # (clean exit 0, do not restart) from cascade-shutdown (exit 1, do
            # restart) — uvicorn's default handlers don't expose that distinction.
            server.install_signal_handlers = lambda: None
            _install_operator_stop_handler(
                lambda: setattr(server, 'should_exit', True),
            )

            # Systemd watchdog heartbeat: ping every _WATCHDOG_INTERVAL so a
            # wedged asyncio loop (no ticks) triggers a restart via
            # WatchdogSec in the unit file. No-op when NOTIFY_SOCKET unset.
            async def _watchdog_heartbeat() -> None:
                while True:
                    _sd_notify('WATCHDOG=1')
                    await asyncio.sleep(_WATCHDOG_INTERVAL)

            watchdog_task = asyncio.create_task(_watchdog_heartbeat())
            _sd_notify('READY=1')
            await server.serve()
        else:
            raise ValueError(f'Unsupported transport: {transport}')
    finally:
        if watchdog_task is not None:
            watchdog_task.cancel()
            with contextlib.suppress(BaseException):
                await watchdog_task
        await _shutdown_with_watchdog(
            memory_service=memory_service,
            task_interceptor=task_interceptor,
            harness_loop_task=harness_loop_task,
            recon_journal=recon_journal,
            event_queue=event_queue,
            sqlite_watchdog=sqlite_watchdog,
            taskmaster=taskmaster,
        )


async def _build_task_backend(taskmaster_config):
    """Construct the configured task backend.

    Selection comes from ``taskmaster_config.backend_mode``:

    - ``taskmaster`` (default) — legacy MCP proxy (:class:`TaskmasterBackend`)
    - ``sqlite`` — in-process SQLite (:class:`SqliteTaskBackend`)
    - ``dual_compare`` — both, wrapped in :class:`DualCompareBackend` so
      one is the served primary and the other is mirrored + compared.
      ``dual_compare_primary`` picks which side serves callers.

    See ``plans/do-1-on-a-happy-pony.md`` §Cycle 2 for the cutover dance.
    """
    from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
    from fused_memory.backends.taskmaster_client import TaskmasterBackend

    mode = taskmaster_config.backend_mode
    if mode == 'taskmaster':
        return TaskmasterBackend(taskmaster_config)
    if mode == 'sqlite':
        return SqliteTaskBackend(taskmaster_config)
    if mode == 'dual_compare':
        from fused_memory.backends.dual_compare_backend import DualCompareBackend

        tm = TaskmasterBackend(taskmaster_config)
        sql = SqliteTaskBackend(taskmaster_config)
        if taskmaster_config.dual_compare_primary == 'sqlite':
            return DualCompareBackend(
                primary=sql, secondary=tm,
                primary_label='sqlite', secondary_label='taskmaster',
            )
        return DualCompareBackend(
            primary=tm, secondary=sql,
            primary_label='taskmaster', secondary_label='sqlite',
        )
    raise ValueError(f'Unknown taskmaster.backend_mode: {mode!r}')


async def _build_ticket_store(data_dir: Path) -> 'TicketStore':
    """Construct and initialise a :class:`TicketStore` for the given data directory.

    The store is backed by ``data_dir/tickets.db`` (sibling to
    ``reconciliation.db``).  :meth:`~TicketStore.initialize` is called before
    returning so the SQLite schema and WAL pragmas are applied.

    Used by both the *reconciliation-enabled* and *disabled* startup branches in
    :func:`run_server` so the same setup is applied consistently.
    """
    from fused_memory.middleware.ticket_store import TicketStore

    store = TicketStore(data_dir / 'tickets.db')
    await store.initialize()
    return store


def _install_safe_tool_wrapper(mcp: Any) -> None:
    """Wrap FastMCP's ToolManager.call_tool to contain BaseException escapes.

    FastMCP's :meth:`ToolManager.call_tool` is the central dispatch chokepoint —
    every ``@mcp.tool()`` call goes through it. Wrapping here covers all
    currently-registered tools and any added later, in one place. Wrapping
    ``add_tool`` would be too late: ``@mcp.tool()`` decorators register at
    import time, before ``create_mcp_server`` returns.

    Contract:

    - :class:`asyncio.CancelledError` is **re-raised** — required for asyncio
      cancellation semantics; swallowing it would leak tasks and break
      structured concurrency.
    - Every other :class:`BaseException` (including :class:`SystemExit`,
      :class:`KeyboardInterrupt`, :class:`BaseExceptionGroup`) is logged at
      ERROR with ``tool_name`` + traceback and a structured error dict is
      returned to the caller, matching the existing tool error-return shape
      ({'error': str, 'error_type': str}).

    Idempotent: if already wrapped (re-entry under tests), the existing
    wrapping is left in place.
    """
    tool_manager = mcp._tool_manager
    if getattr(tool_manager, '_fused_memory_safe_wrapped', False):
        return

    original_call_tool = tool_manager.call_tool

    async def _safe_call_tool(name: str, arguments: dict, *args: Any, **kwargs: Any):
        try:
            return await original_call_tool(name, arguments, *args, **kwargs)
        except asyncio.CancelledError:
            # Required: cancellation must propagate. Never swallow.
            raise
        except BaseException as exc:
            logger.exception(
                'Tool handler escaped exception (defence-in-depth wrapper caught it)',
                extra={'tool_name': name, 'exc_class': type(exc).__name__},
            )
            return {'error': str(exc), 'error_type': type(exc).__name__}

    tool_manager.call_tool = _safe_call_tool
    tool_manager._fused_memory_safe_wrapped = True


def _install_operator_stop_handler(on_operator_stop: Callable[[], None]) -> None:
    """Install SIGTERM/SIGINT handlers that record operator-initiated shutdown.

    Sets the module-level :data:`_operator_stop_received` flag and invokes the
    supplied callback (typically ``lambda: setattr(server, 'should_exit', True)``).
    Mirrors :func:`_register_drain_signal_handler`'s loop / signal.signal
    fallback pattern.

    Must be called BEFORE ``uvicorn.Server.serve()`` and uvicorn's
    ``install_signal_handlers`` must be neutered, otherwise uvicorn replaces
    our handlers from inside ``serve()`` and we never observe SIGTERM.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.warning(
            '_install_operator_stop_handler: no running event loop; '
            'SIGTERM/SIGINT handlers not installed',
        )
        return

    def _operator_stop(signame: str) -> None:
        global _operator_stop_received
        _operator_stop_received = True
        logger.info('Received %s — initiating operator shutdown', signame)
        try:
            on_operator_stop()
        except Exception:
            logger.exception('on_operator_stop callback raised')

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _operator_stop, sig.name)
        except NotImplementedError:
            # Windows: no add_signal_handler support. signal.signal hands us
            # (signum, frame); adapt to our (signame,) callback.
            signal.signal(
                sig,
                lambda signum, frame: _operator_stop(signal.Signals(signum).name),
            )


def _register_drain_signal_handler(reconciliation_harness: 'ReconciliationHarness') -> None:
    """Register a SIGUSR1 handler that triggers reconciliation_harness.drain().

    Uses loop.add_signal_handler (asyncio-safe) when a running event loop is
    available.  Falls back to signal.signal when add_signal_handler raises
    NotImplementedError (Windows only — no add_signal_handler support there).
    If no running event loop is found, logs a warning and installs no handler.

    RuntimeError from add_signal_handler is NOT caught: that error means we are
    not on the main OS thread, and signal.signal would itself raise ValueError in
    that scenario — attempting the fallback would trade one exception for another.
    Since this helper is only invoked from run_server() on the main asyncio thread,
    the RuntimeError path should be unreachable in production; if it is hit, loud
    propagation is preferable to a misleading crash from the fallback.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.warning('_register_drain_signal_handler: no running event loop; SIGUSR1 handler not installed')
        return

    def _handle_drain_signal() -> None:
        logger.info('SIGUSR1 received — triggering harness drain')
        reconciliation_harness.drain()

    try:
        loop.add_signal_handler(signal.SIGUSR1, _handle_drain_signal)
    except NotImplementedError:
        # NotImplementedError: Windows (no add_signal_handler support).
        # NOTE: RuntimeError is intentionally NOT caught here — it means we are not on
        # the main OS thread, and signal.signal would itself raise ValueError in that case.
        # Let the RuntimeError propagate for an honest failure rather than an unrecoverable fallback.
        logger.debug('loop.add_signal_handler unavailable; falling back to signal.signal for SIGUSR1')
        signal.signal(signal.SIGUSR1, lambda signum, frame: _handle_drain_signal())


async def _graceful_shutdown(
    memory_service: MemoryService,
    task_interceptor: 'TaskInterceptor | None',
    harness_loop_task: 'asyncio.Task[None] | None',
    recon_journal: 'ReconciliationJournal | None',
    event_queue: 'EventQueue | None' = None,
    sqlite_watchdog: 'SqliteWatchdog | None' = None,
    taskmaster: Any = None,
) -> None:
    """Perform an ordered, exception-resilient server shutdown.

    Shutdown order:
    1. Drain task_interceptor (flush pending commits / targeted reconciliation).
    2. Close task_interceptor (release curator's Qdrant client).
    3. Cancel SQLite watchdog (purely observational; close before drainer so
       the wedge-detection logic doesn't trip during a legitimate shutdown drain).
    4. Close event_queue (WP-B: bounded flush to SQLite; residue → dead-letter).
       Must happen BEFORE memory_service.close(), because the drainer writes
       into the SQLite event buffer that memory_service owns.
    5. Cancel harness loop task (stops background reconciliation + escalation server).
    6. Close Taskmaster supervisor (cleanly tears down the Node child after
       harness/interceptor stop generating new requests). Done before
       memory_service.close() because MemoryService still holds a reference.
    7. Close memory_service (backends, durable queue, write journal, event buffer).
    8. Close reconciliation journal (separate SQLite connection).

    Each step runs under asyncio.shield with a bounded timeout so cleanup
    makes progress even when this coroutine itself is being cancelled, and
    one stuck step can't starve the rest.

    Note: the force-exit watchdog is NOT armed here.  It is armed by
    _shutdown_with_watchdog (the lifespan-only entry point) so that unit
    tests can call _graceful_shutdown directly without leaking a 45s
    os._exit(1) daemon timer.
    """
    _sd_notify('STOPPING=1')

    if task_interceptor is not None:
        await _run_shielded('task_interceptor.drain', task_interceptor.drain)
        await _run_shielded('task_interceptor.close', task_interceptor.close)

    if sqlite_watchdog is not None:
        await _run_shielded('sqlite_watchdog.close', sqlite_watchdog.close)

    if event_queue is not None:
        await _run_shielded('event_queue.close', event_queue.close)

    if harness_loop_task is not None:
        harness_loop_task.cancel()

        async def _await_harness() -> None:
            with contextlib.suppress(asyncio.CancelledError):
                await harness_loop_task

        await _run_shielded(
            'harness_loop_task',
            _await_harness,
            timeout=_HARNESS_CANCEL_TIMEOUT,
        )

    if taskmaster is not None:
        await _run_shielded('taskmaster.close', taskmaster.close)

    await _run_shielded('memory_service.close', memory_service.close)

    if recon_journal is not None:
        await _run_shielded('recon_journal.close', recon_journal.close)


async def _shutdown_with_watchdog(**kwargs: Any) -> None:
    """Lifespan-only entry point: arm the force-exit watchdog, then run graceful shutdown.

    The watchdog is interpreter-shutdown safety — it guarantees the process dies
    even if cleanup hangs or non-daemon third-party threads keep the interpreter
    alive after asyncio.run() returns.  It is armed here (not inside
    _graceful_shutdown) so unit tests of pure cleanup orchestration don't leak a
    45s os._exit(1) timer.  The watchdog is cancelled in main() after
    asyncio.run() returns cleanly.

    All kwargs are forwarded to _graceful_shutdown unchanged.  Using **kwargs
    rather than repeating the explicit signature avoids the two functions
    diverging silently if _graceful_shutdown grows a new parameter.
    """
    _arm_force_exit()
    await _graceful_shutdown(**kwargs)


_singleton_socket = None  # Module-level ref to prevent GC


def _acquire_singleton_lock() -> None:
    """Acquire a system-wide singleton lock via a bound socket.

    Raises SystemExit if another instance is already running.
    """
    global _singleton_socket
    import socket

    _singleton_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        # Abstract socket namespace — no filesystem cleanup needed
        _singleton_socket.bind('\0fused-memory-singleton')
    except OSError:
        logger.error(
            'Another fused-memory instance is already running. '
            'Kill it first or use systemctl --user restart fused-memory'
        )
        raise SystemExit(1) from None


def main():
    global _operator_stop_received
    _acquire_singleton_lock()
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        # Pre-handler Ctrl-C only (rare race before the asyncio signal handler
        # is installed). Treat as operator-initiated stop.
        _operator_stop_received = True
        logger.info('KeyboardInterrupt before signal handler installed')
    except asyncio.CancelledError:
        # Top-level CancelledError indicates a cascade — the shutdown was not
        # operator-initiated. Let _operator_stop_received stay False so we
        # exit 1 and systemd's Restart=on-failure brings us back up.
        logger.warning('Top-level CancelledError — likely cascade shutdown')
    except Exception:
        logger.exception('Server error')
    finally:
        # asyncio.run() has returned; cleanup either succeeded or the
        # force-exit watchdog will fire. Hard-exit so non-daemon third-party
        # threads (mem0 PostHog consumer, etc.) can't keep the interpreter
        # alive with a dead event loop and a still-bound listen socket.
        _cancel_force_exit()
        sys.stderr.flush()
        # Operator stop → exit 0 (systemd does not restart).
        # Anything else (cascade, exception) → exit 1 (Restart=on-failure fires).
        os._exit(0 if _operator_stop_received else 1)


if __name__ == '__main__':
    main()
