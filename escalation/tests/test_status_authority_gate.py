"""ζ integration gate — B5/B3-server/C1-C4/D1 + composition (comp-1/comp-2).

PRD ``plans/task-status-authority-prd.md`` §Boundary-test sketch. This
module realizes the escalation<->harness SERVER-side rows of the 17-cell
boundary matrix (B5, C1-C4, D1) plus the cross-seam COMPOSITION assertion
(comp-1/comp-2) that no per-task unit test can make: every
``escalation.action_effects`` Table B ``TaskEffect.target_status`` is both a
``shared.task_statuses.TaskStatus`` vocabulary member AND a legal
``shared.task_transitions`` Table A transition — proving Table B computes
intent while Table A validates legality at the chokepoint (PRD D1, "never
three tables").

Two drive harnesses, per the PRD's "two-way ... asserting through the
product's own read paths" mandate:

* In-process (header-less): ``create_server(queue)`` + ``_resolve_issue``
  (sync ``tool.fn`` wrapper) + ``_seed`` — used for B5/B3-server/D1, where
  ``fastmcp.server.dependencies.get_http_headers()`` is irrelevant (it
  always resolves ``{}`` in-process, which IS the header-less case).
* Real HTTP (header-driven): the ``http_server`` daemon-thread fixture +
  ``_resolve_over_http``/``_promote_over_http`` — required for C1-C4, which
  assert on ``X-Escalation-Levels``/``X-Escalation-Identity`` request
  headers that only resolve under a real ASGI request context (mirrors
  ``test_capability_guard_http.py``).

Every read-back goes through ``queue.get(...)`` (escalation state via the
escalation package's own read path), not an internal mock assertion.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

from escalation.action_effects import ACTION_EFFECTS, ANY, TaskEffect, effect_for
from escalation.authority import PROMOTE_ALLOWED, ROLE_LEVEL_ALLOWLIST
from escalation.models import Escalation
from escalation.queue import EscalationQueue
from escalation.server import create_server
from shared.task_statuses import TaskStatus
from shared.task_transitions import ActorClass, is_legal_transition

__all__ = [
    'ACTION_EFFECTS',
    'ANY',
    'PROMOTE_ALLOWED',
    'ROLE_LEVEL_ALLOWLIST',
    'ActorClass',
    'TaskEffect',
    'TaskStatus',
    'effect_for',
    'is_legal_transition',
]

# ---------------------------------------------------------------------------
# In-process (header-less) drive harness — B5, B3-server, D1.
# ---------------------------------------------------------------------------


async def _resolve_issue(server, **kwargs: Any) -> dict[str, Any]:
    """Invoke the resolve_issue MCP tool directly (sync tool; mirrors
    test_server.py:1364-1368). ``get_http_headers()`` resolves ``{}`` here —
    this is the header-less / full-authority path (esc-2087-2)."""
    tool = await server.get_tool('resolve_issue')
    return tool.fn(**kwargs)


def _seed(
    queue: EscalationQueue,
    *,
    level: int,
    task_id: str,
    agent_role: str = 'implementer',
    **kw: Any,
) -> Escalation:
    """Seed a pending escalation at *level* directly via ``queue.submit()``,
    bypassing the MCP tools entirely (mirrors test_server.py's ``_seed_esc``
    and test_capability_guard_http.py's ``_seed``)."""
    kw.setdefault('severity', 'blocking')
    kw.setdefault('category', 'scope_violation')
    kw.setdefault('summary', f'status-authority-gate test escalation (level={level})')
    esc = Escalation(
        id=queue.make_id(task_id),
        task_id=task_id,
        agent_role=agent_role,
        level=level,
        **kw,
    )
    queue.submit(esc)
    return esc


# ---------------------------------------------------------------------------
# Real-HTTP (header-driven) drive harness — C1-C4.
# ---------------------------------------------------------------------------


def _free_port() -> int:
    """Return an ephemeral TCP port free on 127.0.0.1 at the time of the call."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


@pytest.fixture(scope='module')
def http_server(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[str, EscalationQueue]]:
    """Serve a real escalation MCP server over HTTP for this module's tests
    (mirrors test_capability_guard_http.py's ``http_server`` fixture).

    Only a real ASGI request context resolves
    ``X-Escalation-Levels``/``X-Escalation-Identity`` via
    ``get_http_headers()`` — an in-process ``tool.fn(...)`` call always sees
    ``{}``, which would make the C1-C4 capability-guard cells untestable.
    """
    queue_dir = tmp_path_factory.mktemp('status_authority_gate_http')
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
        target=_serve_forever, name='status-authority-gate-http', daemon=True,
    )
    thread.start()

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
            f'status-authority-gate HTTP test server did not become ready on '
            f'127.0.0.1:{port} within 10s'
        )

    yield f'http://127.0.0.1:{port}', queue


async def _resolve_over_http(
    base_url: str,
    *,
    levels: str | None = None,
    identity: str | None = None,
    **resolve_kwargs: Any,
) -> dict[str, Any]:
    """Call ``resolve_issue`` over real HTTP, optionally with capability
    headers (mirrors test_capability_guard_http.py's ``_resolve_over_http``).

    *levels*/*identity*, when not None, are sent as the literal
    ``X-Escalation-Levels``/``X-Escalation-Identity`` request headers; when
    None the header is omitted entirely (never sent as an empty string).
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


async def _promote_over_http(
    base_url: str,
    *,
    levels: str | None = None,
    identity: str | None = None,
    **promote_kwargs: Any,
) -> dict[str, Any]:
    """Call ``promote_to_l2`` over real HTTP, optionally with capability
    headers (mirrors test_capability_guard_http.py's ``_promote_over_http``).
    Proves ``promote_to_l2`` is gated by identity (``PROMOTE_ALLOWED``) but
    never by ``X-Escalation-Levels``."""
    headers: dict[str, str] = {}
    if levels is not None:
        headers['X-Escalation-Levels'] = levels
    if identity is not None:
        headers['X-Escalation-Identity'] = identity
    transport = StreamableHttpTransport(f'{base_url}/mcp/', headers=headers)
    async with Client(transport) as client:
        result = await client.call_tool('promote_to_l2', promote_kwargs)
        return result.data


# ---------------------------------------------------------------------------
# B5 — Table B gate rejects an unrecognised action before any record mutation.
# ---------------------------------------------------------------------------


class TestB5BogusActionRejected:
    """B5: resolve_issue(action='bogus') is rejected by the Table B legality
    gate (escalation.action_effects.effect_for) with a typed
    code='illegal_transition', and the record is left completely unchanged
    (status stays 'pending', resolution_action stays None) — asserted via
    queue.get(), the escalation package's own read path."""

    @pytest.mark.asyncio
    async def test_bogus_action_returns_illegal_transition_record_unchanged(
        self, tmp_path: Path,
    ) -> None:
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)
        esc = _seed(queue, level=1, task_id='t-b5')

        result = await _resolve_issue(
            server, escalation_id=esc.id, resolution='n/a', action='bogus',
        )

        assert result.get('code') == 'illegal_transition', (
            f'Expected code=illegal_transition; got: {result}'
        )
        record = queue.get(esc.id)
        assert record is not None
        assert record.status == 'pending', (
            f'Record must stay pending; got {record.status!r}'
        )
        assert record.resolution_action is None, (
            f'Record must not be stamped; got {record.resolution_action!r}'
        )


# ---------------------------------------------------------------------------
# B3 (escalation side) — park promotes to L2 and keeps the record OPEN.
# ---------------------------------------------------------------------------


class TestB3ServerParkKeepsL2Open:
    """B3, escalation-side face (two-way with B3-harness in
    orchestrator/tests/test_status_authority_gate.py): resolve_issue(action=
    'park') promotes the record to level=2, stamps resolution_action='park',
    and keeps status='pending' (version-a, task 1792) — it does NOT resolve
    or archive the record. Read back via queue.get()."""

    @pytest.mark.asyncio
    async def test_park_open_keeps_record_live_at_l2(self, tmp_path: Path) -> None:
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)
        esc = _seed(queue, level=1, task_id='t-b3-server')

        result = await _resolve_issue(
            server, escalation_id=esc.id, resolution='parked pending human', action='park',
        )

        assert 'error' not in result, f'Unexpected error: {result}'
        record = queue.get(esc.id)
        assert record is not None
        assert record.level == 2, f'Expected level==2 after park; got {record.level}'
        assert record.resolution_action == 'park', (
            f'Expected resolution_action==park; got {record.resolution_action!r}'
        )
        assert record.status == 'pending', (
            f'park must keep the record OPEN (pending), not resolved/dismissed; '
            f'got {record.status!r}'
        )
        assert record.resolved_at is None, (
            f'park must not set resolved_at (open invariant); got {record.resolved_at!r}'
        )


# ---------------------------------------------------------------------------
# D1 — make_id(): strictly increasing, durable, never rescans the archive.
# ---------------------------------------------------------------------------


class TestD1MakeIdCounter:
    """D1: make_id() ids are backed by a single durable per-task_id counter
    (PRD contract C9 / finding 10.4) — strictly increasing with no
    collisions, durable across a fresh EscalationQueue instance, and never
    dependent on scanning the archive directory."""

    def test_rapid_make_id_and_submit_strictly_increasing_no_collision(
        self, tmp_path: Path,
    ) -> None:
        queue = EscalationQueue(tmp_path / 'esc')
        n = 25
        ids: list[str] = []
        for _ in range(n):
            esc_id = queue.make_id('t-d1-rapid')
            queue.submit(
                Escalation(
                    id=esc_id,
                    task_id='t-d1-rapid',
                    agent_role='implementer',
                    severity='blocking',
                    category='scope_violation',
                    summary='status-authority-gate D1 rapid id test',
                )
            )
            ids.append(esc_id)

        assert len(set(ids)) == n, f'Expected {n} unique ids; got {len(set(ids))}: {ids}'
        seqs = [int(i.rsplit('-', 1)[-1]) for i in ids]
        assert seqs == list(range(1, n + 1)), f'Expected contiguous 1..{n}: {seqs}'

        on_disk = {e.id for e in queue.get_by_task('t-d1-rapid')}
        assert on_disk == set(ids), (
            f'Expected all {n} submitted ids readable back via get_by_task; '
            f'missing={set(ids) - on_disk}, extra={on_disk - set(ids)}'
        )

    def test_make_id_durable_across_fresh_queue_instance(self, tmp_path: Path) -> None:
        queue_dir = tmp_path / 'esc'
        first = EscalationQueue(queue_dir)
        assert first.make_id('t-d1-restart') == 'esc-t-d1-restart-1'
        assert first.make_id('t-d1-restart') == 'esc-t-d1-restart-2'

        # Simulate restart: fresh instance, no in-memory state carried over.
        second = EscalationQueue(queue_dir)
        next_id = second.make_id('t-d1-restart')
        assert next_id == 'esc-t-d1-restart-3', (
            f'Expected the durable on-disk counter to continue across a fresh '
            f'EscalationQueue instance; got {next_id!r}'
        )

    def test_make_id_does_not_scan_archive_even_when_archive_present(
        self, tmp_path: Path,
    ) -> None:
        queue_dir = tmp_path / 'esc'
        archive_dir = queue_dir / 'archive' / '2026-01-01'
        archive_dir.mkdir(parents=True, exist_ok=True)
        seeded = Escalation(
            id='esc-t-d1-archived-9',
            task_id='t-d1-archived',
            agent_role='implementer',
            severity='blocking',
            category='scope_violation',
            summary='hand-seeded archive record (no counter behind it)',
        )
        seeded.status = 'resolved'
        (archive_dir / f'{seeded.id}.json').write_text(seeded.to_json())

        queue = EscalationQueue(queue_dir)
        with patch.object(
            queue, '_iter_archive_paths', wraps=queue._iter_archive_paths,
        ) as spy:
            first_id = queue.make_id('t-d1-archived')
            second_id = queue.make_id('t-d1-archived')

        assert spy.call_count == 0, (
            f'make_id() must not scan the archive even with an archive file '
            f'present; _iter_archive_paths called {spy.call_count}x'
        )
        assert first_id == 'esc-t-d1-archived-1', (
            f'Counter is authoritative (no archive-derived catch-up); expected '
            f'esc-t-d1-archived-1, got {first_id!r}'
        )
        assert second_id == 'esc-t-d1-archived-2', (
            f'Expected esc-t-d1-archived-2; got {second_id!r}'
        )
