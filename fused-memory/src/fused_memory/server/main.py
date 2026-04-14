"""Entry point for the Fused Memory MCP server."""

import argparse
import asyncio
import logging
import os
import signal
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

load_dotenv()

from fused_memory.config.schema import FusedMemoryConfig  # noqa: E402
from fused_memory.server.tools import create_mcp_server  # noqa: E402
from fused_memory.services.memory_service import MemoryService  # noqa: E402

if TYPE_CHECKING:
    from fused_memory.middleware.task_interceptor import TaskInterceptor
    from fused_memory.reconciliation.journal import ReconciliationJournal

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

    # Initialize Taskmaster backend
    taskmaster = None
    task_interceptor = None
    if config.taskmaster:
        from fused_memory.backends.taskmaster_client import TaskmasterBackend

        taskmaster = TaskmasterBackend(config.taskmaster)
        try:
            await taskmaster.initialize()
            logger.info(f'  Taskmaster: connected via {config.taskmaster.transport}')
            memory_service.taskmaster_connected = True
        except Exception as e:
            logger.warning(f'  Taskmaster: failed to connect ({e}), will retry on next tool call')

    # Initialize reconciliation system
    harness_loop_task = None
    recon_journal = None
    reconciliation_harness = None
    if config.reconciliation and config.reconciliation.enabled:
        from fused_memory.middleware.task_interceptor import TaskInterceptor
        from fused_memory.reconciliation.event_buffer import EventBuffer
        from fused_memory.reconciliation.harness import ReconciliationHarness
        from fused_memory.reconciliation.journal import ReconciliationJournal
        from fused_memory.reconciliation.targeted import TargetedReconciler

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

        task_committer = TaskFileCommitter()
        task_interceptor = TaskInterceptor(
            taskmaster, targeted, event_buffer, task_committer, config=config,
        )

        # Full reconciliation harness (background loop)
        reconciliation_harness = ReconciliationHarness(
            memory_service, taskmaster, recon_journal, event_buffer, config
        )
        harness_loop_task = asyncio.create_task(reconciliation_harness.run_loop())
        logger.info('  Reconciliation: enabled (background loop started)')

        # SIGUSR1 triggers harness drain (stop new cycles, let current ones finish)
        def _handle_drain_signal(signum: int, frame: object) -> None:
            logger.info('SIGUSR1 received — triggering harness drain')
            reconciliation_harness.drain()

        signal.signal(signal.SIGUSR1, _handle_drain_signal)
    else:
        # Always create task_interceptor for tool registration
        from fused_memory.middleware.task_file_committer import TaskFileCommitter
        from fused_memory.middleware.task_interceptor import TaskInterceptor
        from fused_memory.reconciliation.event_buffer import EventBuffer

        event_buffer = EventBuffer(db_path=None)
        await event_buffer.initialize()
        task_committer = TaskFileCommitter()
        task_interceptor = TaskInterceptor(
            taskmaster, None, event_buffer, task_committer, config=config,
        )

    # Create MCP server with both memory and task tools
    mcp = create_mcp_server(
        memory_service, task_interceptor, write_journal,
        reconciliation_harness=reconciliation_harness,
    )
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
            uv_config = uvicorn.Config(
                starlette_app,
                host=config.server.host,
                port=config.server.port,
                log_level='info',
                timeout_keep_alive=config.server.keepalive_timeout,
            )
            server = uvicorn.Server(uv_config)
            await server.serve()
        else:
            raise ValueError(f'Unsupported transport: {transport}')
    finally:
        await _graceful_shutdown(
            memory_service=memory_service,
            task_interceptor=task_interceptor,
            harness_loop_task=harness_loop_task,
            recon_journal=recon_journal,
        )


async def _graceful_shutdown(
    memory_service: MemoryService,
    task_interceptor: 'TaskInterceptor | None',
    harness_loop_task: 'asyncio.Task[None] | None',
    recon_journal: 'ReconciliationJournal | None',
) -> None:
    """Perform an ordered, exception-resilient server shutdown.

    Shutdown order:
    1. Drain task_interceptor (flush pending commits / targeted reconciliation).
    2. Close task_interceptor (release curator's Qdrant client).
    3. Cancel harness loop task (stops background reconciliation + escalation server).
    4. Close memory_service (backends, durable queue, write journal, event buffer).
    5. Close reconciliation journal (separate SQLite connection).

    Each step is independently guarded so a failure in one step does not
    prevent subsequent steps from running.
    """
    if task_interceptor is not None:
        try:
            await task_interceptor.drain()
        except Exception:
            logger.exception('_graceful_shutdown: error draining task_interceptor')
        try:
            await task_interceptor.close()
        except Exception:
            logger.exception('_graceful_shutdown: error closing task_interceptor')

    if harness_loop_task is not None:
        try:
            harness_loop_task.cancel()
            await asyncio.wait_for(harness_loop_task, timeout=_HARNESS_CANCEL_TIMEOUT)
        except asyncio.CancelledError:
            pass
        except TimeoutError:
            logger.warning('_graceful_shutdown: harness_loop_task timed out during cancellation')
        except Exception:
            logger.exception('_graceful_shutdown: unexpected error from harness_loop_task')

    try:
        await memory_service.close()
    except Exception:
        logger.exception('_graceful_shutdown: error closing memory_service')

    if recon_journal is not None:
        try:
            await recon_journal.close()
        except Exception:
            logger.exception('_graceful_shutdown: error closing recon_journal')



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
    _acquire_singleton_lock()
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        logger.info('Server shutting down...')
    except Exception as e:
        logger.error(f'Server error: {e}')
        raise


if __name__ == '__main__':
    main()
