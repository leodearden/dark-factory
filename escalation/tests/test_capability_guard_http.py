"""HTTP integration tests for the resolve_issue capability guard (PRD alpha).

These tests drive a RUNNING FastMCP escalation server over real HTTP rather
than calling ``tool.fn(...)`` in-process, because
``fastmcp.server.dependencies.get_http_headers()`` only resolves real request
headers under an ASGI request context — an in-process ``tool.fn(...)`` call
always sees ``{}``, which would make the X-Escalation-Levels /
X-Escalation-Identity capability gate untestable.

See plans/escalation-connection-capability-guard-prd.md (task alpha).
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

from escalation.models import Escalation
from escalation.queue import EscalationQueue
from escalation.server import create_server

# ---------------------------------------------------------------------------
# Harness: a real escalation MCP server served over HTTP in a daemon thread.
# ---------------------------------------------------------------------------


def _free_port() -> int:
    """Return an ephemeral TCP port free on 127.0.0.1 at the time of the call.

    Binds to port 0 (OS-assigned free port) and immediately closes the
    socket. There is an inherent (small) TOCTOU window before the real
    server binds the same port; acceptable for a single-threaded test run.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


@pytest.fixture(scope='module')
def http_server(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[str, EscalationQueue]]:
    """Serve a real escalation MCP server over HTTP for this module's tests.

    Builds an ``EscalationQueue`` + ``create_server(queue, startup_sweep=False)``
    (``startup_sweep=False`` so pre-seeded queue files are not relocated by the
    startup sweep — mirrors the existing test convention), then serves it via
    ``FastMCP.run_http_async`` on a free localhost port inside a daemon thread
    running its own event loop. Readiness is polled by attempting a raw TCP
    connect to the port (bounded to ~10s total) — no fixed sleep.

    Yields ``(base_url, queue)``: tests seed pending escalations directly via
    ``queue.submit()`` (file-backed, per-id locked — safe across the test
    thread and the server's serving thread) and then connect a
    ``fastmcp.Client(StreamableHttpTransport(f'{base_url}/mcp/', headers=...))``
    per scenario to send per-connection capability headers over real HTTP.

    The serving thread is a daemon thread with no explicit shutdown: it is
    killed automatically when the test process exits.
    """
    queue_dir = tmp_path_factory.mktemp('esc_capability_guard')
    queue = EscalationQueue(queue_dir)
    mcp = create_server(queue, startup_sweep=False)
    port = _free_port()

    def _serve_forever() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(
            mcp.run_http_async(
                host='127.0.0.1', port=port, show_banner=False, log_level='error',
            )
        )

    thread = threading.Thread(
        target=_serve_forever, name='escalation-capability-guard-http', daemon=True,
    )
    thread.start()

    # Poll for readiness (bounded ~10s) instead of a fixed sleep.
    deadline = time.monotonic() + 10.0
    ready = False
    while time.monotonic() < deadline:
        with contextlib.suppress(OSError), socket.create_connection(('127.0.0.1', port), timeout=0.2):
            ready = True
        if ready:
            break
        time.sleep(0.05)
    if not ready:
        raise RuntimeError(
            f'escalation HTTP test server did not become ready on '
            f'127.0.0.1:{port} within 10s'
        )

    yield f'http://127.0.0.1:{port}', queue


# ---------------------------------------------------------------------------
# Seeding + call helpers
# ---------------------------------------------------------------------------


def _seed(
    queue: EscalationQueue,
    *,
    level: int,
    task_id: str,
    agent_role: str = 'implementer',
    **kw: Any,
) -> Escalation:
    """Seed a pending escalation at *level* directly via ``queue.submit()``.

    Bypasses the MCP tools entirely (mirrors the ``_seed_esc`` helper in
    test_server.py). ``severity``/``category``/``summary`` default to
    innocuous values but can be overridden via **kw.
    """
    kw.setdefault('severity', 'blocking')
    kw.setdefault('category', 'scope_violation')
    kw.setdefault('summary', f'capability-guard test escalation (level={level})')
    esc = Escalation(
        id=queue.make_id(task_id),
        task_id=task_id,
        agent_role=agent_role,
        level=level,
        **kw,
    )
    queue.submit(esc)
    return esc


async def _resolve_over_http(
    base_url: str,
    *,
    levels: str | None = None,
    identity: str | None = None,
    **resolve_kwargs: Any,
) -> dict[str, Any]:
    """Call ``resolve_issue`` over real HTTP, optionally with capability headers.

    *levels* / *identity*, when not None, are sent as the literal
    ``X-Escalation-Levels`` / ``X-Escalation-Identity`` request headers; when
    None the header is omitted entirely (never sent as an empty string), so a
    header-less call exercises the exact same default-open path a real
    header-less client would hit.
    """
    headers: dict[str, str] = {}
    if levels is not None:
        headers['X-Escalation-Levels'] = levels
    if identity is not None:
        headers['X-Escalation-Identity'] = identity
    transport = StreamableHttpTransport(f'{base_url}/mcp/', headers=headers)
    async with Client(transport) as client:
        result = await client.call_tool('resolve_issue', resolve_kwargs)
        return result.data


# ---------------------------------------------------------------------------
# TestHarnessSanity: fixture plumbing only — no capability-guard behaviour yet.
# ---------------------------------------------------------------------------


class TestHarnessSanity:
    """Confirms the HTTP harness itself starts/stops cleanly and round-trips.

    Not a capability-guard behavioural test (those land in the TDD steps that
    follow this prerequisite) — this only proves the fixture, seeding helper,
    and HTTP client plumbing work end-to-end before any gate assertions are
    added.
    """

    @pytest.mark.asyncio
    async def test_seeded_escalation_readable_over_http(
        self, http_server: tuple[str, EscalationQueue],
    ) -> None:
        base_url, queue = http_server
        esc = _seed(queue, level=2, task_id='task-sanity')

        async with Client(StreamableHttpTransport(f'{base_url}/mcp/')) as client:
            result = await client.call_tool('get_escalation', {'escalation_id': esc.id})

        assert result.data['id'] == esc.id, f'Expected id {esc.id!r}, got: {result.data}'
        assert result.data['level'] == 2
        assert result.data['status'] == 'pending'


# ---------------------------------------------------------------------------
# TestLevelForbidden: X-Escalation-Levels denies out-of-set resolve/park;
# a header-less connection stays default-open.
# ---------------------------------------------------------------------------


class TestLevelForbidden:
    """X-Escalation-Levels='0,1' forbids resolve/park on an L2 record; a
    header-less connection is unaffected (default-open)."""

    @pytest.mark.asyncio
    async def test_denied_resolve_and_park_vs_default_open(
        self, http_server: tuple[str, EscalationQueue],
    ) -> None:
        base_url, queue = http_server
        esc_deny_resolve = _seed(queue, level=2, task_id='task-deny-resolve')
        esc_deny_park = _seed(queue, level=2, task_id='task-deny-park')
        esc_open = _seed(queue, level=2, task_id='task-open')

        # (a) 0,1 client denied a close_only resolve on an L2 record — no mutation.
        result_a = await _resolve_over_http(
            base_url, levels='0,1',
            escalation_id=esc_deny_resolve.id, resolution='x', action='close_only',
        )
        assert result_a.get('code') == 'level_forbidden', (
            f"Expected code='level_forbidden', got: {result_a}"
        )
        reread_a = queue.get(esc_deny_resolve.id)
        assert reread_a is not None
        assert reread_a.status == 'pending', f'Expected pending, got: {reread_a.status}'
        assert (queue.queue_dir / f'{esc_deny_resolve.id}.json').exists(), (
            'Denied record must remain in the queue root (not archived)'
        )
        assert reread_a.resolution_action is None, (
            f'Expected no resolution_action stamp, got: {reread_a.resolution_action}'
        )

        # (b) 0,1 client denied park on an L2 record — no park stamp either.
        result_b = await _resolve_over_http(
            base_url, levels='0,1',
            escalation_id=esc_deny_park.id, action='park', resolution='x',
        )
        assert result_b.get('code') == 'level_forbidden', (
            f"Expected code='level_forbidden', got: {result_b}"
        )
        reread_b = queue.get(esc_deny_park.id)
        assert reread_b is not None
        assert reread_b.status == 'pending', f'Expected pending, got: {reread_b.status}'
        assert reread_b.resolution_action is None, (
            f'Expected no resolution_action stamp, got: {reread_b.resolution_action}'
        )
        assert reread_b.resolution is None, (
            f'Expected no resolution text (no park stamp), got: {reread_b.resolution!r}'
        )

        # (c) Contrast: a header-less client stays default-open on the same level.
        result_c = await _resolve_over_http(
            base_url, escalation_id=esc_open.id, resolution='ok', action='close_only',
        )
        assert 'code' not in result_c and 'error' not in result_c, (
            f'Expected a clean success (no code/error), got: {result_c}'
        )
        reread_c = queue.get(esc_open.id)
        assert reread_c is not None
        assert reread_c.status in {'resolved', 'dismissed'}, (
            f'Expected archived status, got: {reread_c.status}'
        )
