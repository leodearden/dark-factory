"""HTTP integration tests for the resolve_issue capability guard.

The complete seven-scenario two-way boundary gate for the escalation
level+identity capability seam (PRD tasks alpha/beta/gamma).

These tests drive a RUNNING FastMCP escalation server over real HTTP rather
than calling ``tool.fn(...)`` in-process, because
``fastmcp.server.dependencies.get_http_headers()`` only resolves real request
headers under an ASGI request context — an in-process ``tool.fn(...)`` call
always sees ``{}``, which would make the X-Escalation-Levels /
X-Escalation-Identity capability gate untestable.

Seven boundary scenarios, each mapped to its asserting test class:

  1. Deny an out-of-set resolve on an L2 record            -> TestLevelForbidden
  2. Deny an out-of-set park on an L2 record                -> TestLevelForbidden
  3. Allowed L1 resolve + X-Escalation-Identity override    -> TestIdentityOverride
  4. promote_to_l2 is never gated by X-Escalation-Levels    -> TestIdentityOverride
  5. Header-less (human) connection keeps FULL L2 authority
     (resume / park / close_only all succeed; resolved_by
     comes from the tool arg, not a server override)        -> TestHumanUnrestrictedFullAuthority
  6. l2-cascade member resolution is unaffected by the
     capability guard for a permitted header-less resolve   -> TestCascadeIntactHeaderless
  7. A malformed X-Escalation-Levels header fails closed
     (bad_capability_header, no mutation)                   -> TestMalformedHeaderFailsClosed

Two-way lockstep (beyond the seven scenarios above): the REAL orchestrator
constant ``orchestrator.harness._WATCHER_ESCALATION_HEADERS`` is imported and
driven against this running server, proving the client-side header contract
and the server-side header parser stay in lockstep -> TestWatcherConstantLockstep.

See plans/escalation-connection-capability-guard-prd.md (tasks alpha/beta/gamma).
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


async def _promote_over_http(
    base_url: str,
    *,
    levels: str | None = None,
    identity: str | None = None,
    **promote_kwargs: Any,
) -> dict[str, Any]:
    """Call ``promote_to_l2`` over real HTTP, optionally with capability headers.

    Mirrors ``_resolve_over_http`` — used to prove ``promote_to_l2`` is never
    gated by X-Escalation-Levels (it is intentionally left ungated).
    """
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


# ---------------------------------------------------------------------------
# TestIdentityOverride: X-Escalation-Identity overrides resolved_by on an
# allowed resolve; promote_to_l2 stays ungated by X-Escalation-Levels.
# ---------------------------------------------------------------------------


class TestIdentityOverride:
    """X-Escalation-Identity wins over the resolved_by tool arg; promote_to_l2
    is never gated by X-Escalation-Levels (contrast)."""

    @pytest.mark.asyncio
    async def test_identity_override_and_promote_ungated_contrast(
        self, http_server: tuple[str, EscalationQueue],
    ) -> None:
        base_url, queue = http_server

        # (a) Identity header wins over the tool-arg resolved_by on an allowed resolve.
        esc_l1 = _seed(queue, level=1, task_id='task-l1-identity')
        result_a = await _resolve_over_http(
            base_url, levels='0,1', identity='orchestrator-escalation-watcher-auto',
            escalation_id=esc_l1.id, resolution='fixed', action='resume',
            resolved_by='spoofed-by-agent',
        )
        assert 'code' not in result_a and 'error' not in result_a, (
            f'Expected a clean success (level 1 is in {{0,1}}), got: {result_a}'
        )
        reread = queue.get(esc_l1.id)
        assert reread is not None
        assert reread.status == 'resolved', f'Expected resolved, got: {reread.status}'
        assert reread.resolved_by == 'orchestrator-escalation-watcher-auto', (
            f'Expected the header identity to win over the tool arg, got: {reread.resolved_by!r}'
        )

        # (b) Contrast: promote_to_l2 is never gated by X-Escalation-Levels.
        m1 = _seed(queue, level=1, task_id='task-promote-m1')
        m2 = _seed(queue, level=1, task_id='task-promote-m2')
        result_b = await _promote_over_http(
            base_url, levels='0,1',
            task_id='task-promote-cluster', agent_role='escalation-watcher-auto',
            member_ids=[m1.id, m2.id], root_cause='rc-α', evidence='e',
            options=['A', 'B'], summary='cluster',
        )
        assert result_b.get('status') in {'created', 'updated'}, (
            f"Expected status in {{'created','updated'}}, got: {result_b}"
        )


# ---------------------------------------------------------------------------
# TestMalformedHeaderFailsClosed: an unparseable X-Escalation-Levels rejects
# the call — with NO mutation — for both resolve and park.
# ---------------------------------------------------------------------------


class TestMalformedHeaderFailsClosed:
    """A malformed X-Escalation-Levels header fails closed with no mutation,
    for both a plain resolve and a park action."""

    @pytest.mark.asyncio
    async def test_malformed_levels_header_rejected_before_any_mutation(
        self, http_server: tuple[str, EscalationQueue],
    ) -> None:
        base_url, queue = http_server
        esc_bad = _seed(queue, level=2, task_id='task-bad-resolve')
        esc_bad_park = _seed(queue, level=2, task_id='task-bad-park')

        # Malformed header rejected before a close_only resolve — no mutation.
        result = await _resolve_over_http(
            base_url, levels='garbage',
            escalation_id=esc_bad.id, resolution='x', action='close_only',
        )
        assert result.get('code') == 'bad_capability_header', (
            f"Expected code='bad_capability_header', got: {result}"
        )
        reread = queue.get(esc_bad.id)
        assert reread is not None
        assert reread.status == 'pending', f'Expected pending, got: {reread.status}'
        assert (queue.queue_dir / f'{esc_bad.id}.json').exists(), (
            'Rejected record must remain in the queue root (not archived)'
        )
        assert reread.resolution_action is None, (
            f'Expected no resolution_action stamp, got: {reread.resolution_action}'
        )

        # Malformed header rejected before park too.
        result_park = await _resolve_over_http(
            base_url, levels='garbage',
            escalation_id=esc_bad_park.id, action='park', resolution='x',
        )
        assert result_park.get('code') == 'bad_capability_header', (
            f"Expected code='bad_capability_header', got: {result_park}"
        )
        reread_park = queue.get(esc_bad_park.id)
        assert reread_park is not None
        assert reread_park.status == 'pending', f'Expected pending, got: {reread_park.status}'
        assert reread_park.resolution_action is None, (
            f'Expected no resolution_action stamp, got: {reread_park.resolution_action}'
        )


# ---------------------------------------------------------------------------
# TestHumanUnrestrictedFullAuthority: a header-less (human) connection keeps
# FULL L2 authority — resume, park, and close_only all succeed, and
# resolved_by comes from the tool arg (no server-side override).
# ---------------------------------------------------------------------------


class TestHumanUnrestrictedFullAuthority:
    """A header-less connection retains full L2 authority across all three
    resolve_issue actions that touch an L2 record; resolved_by always comes
    from the tool arg since no X-Escalation-Identity header is present to
    override it.

    Strengthens alpha's partial scenario 5, which only proved header-less
    close_only and never asserted resolved_by provenance.
    """

    @pytest.mark.asyncio
    async def test_resume_park_close_only_all_succeed_header_less(
        self, http_server: tuple[str, EscalationQueue],
    ) -> None:
        base_url, queue = http_server
        esc_resume = _seed(queue, level=2, task_id='task-human-resume')
        esc_park = _seed(queue, level=2, task_id='task-human-park')
        esc_close = _seed(queue, level=2, task_id='task-human-close')

        # (a) resume — header-less connection resolves an L2; resolved_by is
        # the tool arg verbatim (no Identity header to override it).
        result_a = await _resolve_over_http(
            base_url,
            escalation_id=esc_resume.id, resolution='fixed', action='resume',
            resolved_by='escalation-watcher',
        )
        assert 'code' not in result_a and 'error' not in result_a, (
            f'Expected a clean success, got: {result_a}'
        )
        reread_a = queue.get(esc_resume.id)
        assert reread_a is not None
        assert reread_a.status == 'resolved', f'Expected resolved, got: {reread_a.status}'
        assert reread_a.resolved_by == 'escalation-watcher', (
            f'Expected tool-arg resolved_by (no Identity header), got: {reread_a.resolved_by!r}'
        )

        # (b) park — keeps the L2 OPEN (status stays pending); park stamps
        # resolution_action + resolution text + resolved_by, all from the
        # tool call since there is no Identity header.
        result_b = await _resolve_over_http(
            base_url,
            escalation_id=esc_park.id, resolution='parked pending human',
            action='park', resolved_by='escalation-watcher',
        )
        assert 'code' not in result_b and 'error' not in result_b, (
            f'Expected a clean success, got: {result_b}'
        )
        reread_b = queue.get(esc_park.id)
        assert reread_b is not None
        assert reread_b.status == 'pending', (
            f'Expected pending (park keeps the L2 open), got: {reread_b.status}'
        )
        assert reread_b.resolution_action == 'park', (
            f"Expected resolution_action='park', got: {reread_b.resolution_action!r}"
        )
        assert reread_b.resolution == 'parked pending human', (
            f'Expected resolution text stamped, got: {reread_b.resolution!r}'
        )
        assert reread_b.resolved_by == 'escalation-watcher', (
            f'Expected tool-arg resolved_by (no Identity header), got: {reread_b.resolved_by!r}'
        )

        # (c) close_only — dismisses with no workflow effect.
        result_c = await _resolve_over_http(
            base_url,
            escalation_id=esc_close.id, resolution='no longer relevant',
            action='close_only', resolved_by='escalation-watcher',
        )
        assert 'code' not in result_c and 'error' not in result_c, (
            f'Expected a clean success, got: {result_c}'
        )
        reread_c = queue.get(esc_close.id)
        assert reread_c is not None
        assert reread_c.status == 'dismissed', f'Expected dismissed, got: {reread_c.status}'
        assert reread_c.resolved_by == 'escalation-watcher', (
            f'Expected tool-arg resolved_by (no Identity header), got: {reread_c.resolved_by!r}'
        )


# ---------------------------------------------------------------------------
# TestCascadeIntactHeaderless: l2-cascade member resolution is unaffected by
# the capability guard for a permitted header-less parent-L2 resolve.
# ---------------------------------------------------------------------------


class TestCascadeIntactHeaderless:
    """A header-less caller resolving an L2 that carries member L1s still
    triggers the l2-cascade member resolution (queue.py:483-499): each
    member is archived with resolved_by=f'l2-cascade:{l2_id}', proving the
    pre-existing cascade is unaffected by the capability guard on a
    permitted (header-less) parent-L2 resolve.
    """

    @pytest.mark.asyncio
    async def test_cascade_resolves_members_header_less(
        self, http_server: tuple[str, EscalationQueue],
    ) -> None:
        base_url, queue = http_server
        m1 = _seed(queue, level=1, task_id='task-cascade-m1')
        m2 = _seed(queue, level=1, task_id='task-cascade-m2')
        l2 = _seed(
            queue, level=2, task_id='task-cascade-l2', members=[m1.id, m2.id],
        )

        result = await _resolve_over_http(
            base_url,
            escalation_id=l2.id, resolution='cluster resolved',
            action='close_only', resolved_by='escalation-watcher',
        )
        assert 'code' not in result and 'error' not in result, (
            f'Expected a clean success, got: {result}'
        )

        reread_l2 = queue.get(l2.id)
        assert reread_l2 is not None
        assert reread_l2.status in {'resolved', 'dismissed'}, (
            f'Expected the L2 archived, got: {reread_l2.status}'
        )

        for member in (m1, m2):
            reread_member = queue.get(member.id)
            assert reread_member is not None, (
                f'Expected member {member.id} still readable (archived), got None'
            )
            assert reread_member.status in {'resolved', 'dismissed'}, (
                f'Expected member {member.id} archived, got: {reread_member.status}'
            )
            assert reread_member.resolved_by == f'l2-cascade:{l2.id}', (
                f'Expected the cascade stamp to override the tool arg for member '
                f'{member.id}, got: {reread_member.resolved_by!r}'
            )


# ---------------------------------------------------------------------------
# TestWatcherConstantLockstep: the REAL orchestrator constant
# (_WATCHER_ESCALATION_HEADERS) is driven against this running server,
# proving the client-side header contract and the server-side header parser
# stay in lockstep.
# ---------------------------------------------------------------------------


class TestWatcherConstantLockstep:
    """Drives the actual ``orchestrator.harness._WATCHER_ESCALATION_HEADERS``
    constant against the running escalation server: its Levels='0,1' denies
    an L2 resolve and its Identity stamps resolved_by on an allowed L1
    resolve, overriding a spoofed tool arg.

    Closes the gap flagged at orchestrator/src/orchestrator/harness.py:206-211
    ('No test in this repo exercises this constant against the live server
    parser') — this is the genuine two-way (client-constant <-> server-
    enforcement) half of the boundary gate, beyond the seven scenarios above.
    """

    @pytest.mark.asyncio
    async def test_real_watcher_headers_deny_l2_and_stamp_l1_identity(
        self, http_server: tuple[str, EscalationQueue],
    ) -> None:
        from orchestrator.harness import _WATCHER_ESCALATION_HEADERS

        base_url, queue = http_server
        levels = _WATCHER_ESCALATION_HEADERS['X-Escalation-Levels']
        identity = _WATCHER_ESCALATION_HEADERS['X-Escalation-Identity']

        # (a) Real watcher headers deny a close_only resolve on an L2 — no mutation.
        esc_l2 = _seed(queue, level=2, task_id='task-lockstep-l2')
        result_a = await _resolve_over_http(
            base_url, levels=levels, identity=identity,
            escalation_id=esc_l2.id, resolution='x', action='close_only',
        )
        assert result_a.get('code') == 'level_forbidden', (
            f"Expected code='level_forbidden', got: {result_a}"
        )
        reread_a = queue.get(esc_l2.id)
        assert reread_a is not None
        assert reread_a.status == 'pending', f'Expected pending, got: {reread_a.status}'
        assert (queue.queue_dir / f'{esc_l2.id}.json').exists(), (
            'Denied record must remain in the queue root (not archived)'
        )
        assert reread_a.resolution_action is None, (
            f'Expected no resolution_action stamp, got: {reread_a.resolution_action}'
        )

        # (b) Real watcher headers allow an L1 resolve; the Identity header
        # stamps resolved_by, overriding a spoofed tool arg.
        esc_l1 = _seed(queue, level=1, task_id='task-lockstep-l1')
        result_b = await _resolve_over_http(
            base_url, levels=levels, identity=identity,
            escalation_id=esc_l1.id, resolution='fixed', action='resume',
            resolved_by='spoofed-by-agent',
        )
        assert 'code' not in result_b and 'error' not in result_b, (
            f'Expected a clean success (level 1 is in the watcher set), got: {result_b}'
        )
        reread_b = queue.get(esc_l1.id)
        assert reread_b is not None
        assert reread_b.status == 'resolved', f'Expected resolved, got: {reread_b.status}'
        assert reread_b.resolved_by == identity, (
            f'Expected the header identity to win over the spoofed tool arg, got: '
            f'{reread_b.resolved_by!r}'
        )
