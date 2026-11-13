"""Real escalation servers over fixture project roots, and a recording fused-memory fake.

For the sitting-tooling suites that apply rulings across more than one queue.
A non-test sibling module, importable by bare name because conftest.py appends
scripts/tests/ to sys.path (see cli_subprocess_timeout for the convention).

``serve_project_queue`` writes the project's ``dark-factory-orchestrator.yaml``
naming the port it serves on, so the dashboard's discovery finds the server the
same way production does. The serve/teardown lifecycle follows
escalation/tests/conftest.py::_serve_escalation_mcp_impl, which this suite
cannot import (conftest names collide under --import-mode=importlib): the
listener is bound before the port is published and held until teardown,
readiness is a real MCP handshake rather than a TCP connect, and teardown joins
the serving thread with a bound and asserts it exited.
"""
from __future__ import annotations

import asyncio
import contextlib
import socket
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import uvicorn
import yaml
from escalation.dedupe import DedupeConfig
from escalation.queue import EscalationQueue
from escalation.server import create_server
from escalation.store_identity import StoreIdentity
from shared.mcp_session import McpCallError, McpFailureKind, SyncMcpSession

QUEUE_SUBDIR = Path('data') / 'escalations'
READY_TIMEOUT_S = 60.0
TEARDOWN_JOIN_S = 5.0


@dataclass(frozen=True)
class ServedQueue:
    port: int
    base_url: str
    queue_dir: Path
    queue: EscalationQueue


def _drain_pending_tasks(loop: asyncio.AbstractEventLoop) -> None:
    pending = asyncio.all_tasks(loop)
    if not pending:
        return
    for task in pending:
        task.cancel()
    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))


def _write_orchestrator_config(project_root: Path, port: int) -> None:
    config = {'escalation': {'queue_dir': str(QUEUE_SUBDIR), 'port': port}}
    (project_root / 'dark-factory-orchestrator.yaml').write_text(yaml.safe_dump(config))


def _handshake_ready(base_url: str, errors: list[BaseException]) -> bool:
    try:
        with SyncMcpSession(base_url, client_name='sitting-test-ready', timeout=5.0):
            return True
    except McpCallError as exc:
        errors[:] = [exc]
        return False


@contextlib.contextmanager
def serve_project_queue(project_root: Path, project_id: str) -> Iterator[ServedQueue]:
    """Serve ``<project_root>/data/escalations`` on an ephemeral localhost port."""
    queue_dir = project_root / QUEUE_SUBDIR
    queue_dir.mkdir(parents=True, exist_ok=True)
    queue = EscalationQueue(queue_dir)
    mcp = create_server(
        queue,
        startup_sweep=False,
        dedupe_config=DedupeConfig(infra_dedupe_enabled=False),
        store_identity=StoreIdentity('project', queue_dir, project_id, project_root),
    )
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(('127.0.0.1', 0))
    port = listener.getsockname()[1]
    _write_orchestrator_config(project_root, port)
    loop = asyncio.new_event_loop()
    state: dict[str, Any] = {'stopping': False, 'serve_error': None}

    def _serve_forever() -> None:
        asyncio.set_event_loop(loop)
        try:
            server = uvicorn.Server(uvicorn.Config(mcp.http_app(), log_level='error', ws='none'))
            loop.run_until_complete(server.serve(sockets=[listener]))
        except BaseException as exc:
            if not state['stopping']:
                state['serve_error'] = exc
        finally:
            _drain_pending_tasks(loop)
            loop.close()

    thread = threading.Thread(target=_serve_forever, name=f'sitting-escalation-{port}', daemon=True)
    thread.start()
    base_url = f'http://127.0.0.1:{port}'
    try:
        _await_ready(base_url, thread, state)
        yield ServedQueue(port=port, base_url=base_url, queue_dir=queue_dir, queue=queue)
    finally:
        state['stopping'] = True
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=TEARDOWN_JOIN_S)
        listener.close()
        assert not thread.is_alive(), f'escalation server thread on port {port} did not stop'


def _await_ready(base_url: str, thread: threading.Thread, state: Mapping[str, Any]) -> None:
    deadline = time.monotonic() + READY_TIMEOUT_S
    errors: list[BaseException] = []
    while thread.is_alive() and time.monotonic() < deadline:
        if _handshake_ready(base_url, errors):
            return
        time.sleep(0.05)
    raise RuntimeError(
        f'escalation server at {base_url} never completed an MCP handshake '
        f'(thread alive: {thread.is_alive()}, serve error: {state["serve_error"]!r}, '
        f'last handshake error: {errors[-1] if errors else None!r})'
    )


class RecordingToolCaller:
    """A fused-memory stand-in: records every call, answers from a per-tool reply table."""

    def __init__(
        self,
        replies: Mapping[str, dict] | None = None,
        *,
        fail: frozenset[str] = frozenset(),
    ) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._replies = dict(replies or {})
        self._fail = fail

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> dict:
        self.calls.append((name, dict(arguments)))
        if name in self._fail:
            raise McpCallError(McpFailureKind.TOOL_ERROR, f'scripted failure of {name}')
        return dict(self._replies.get(name, {}))

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]
