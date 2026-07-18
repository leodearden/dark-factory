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


def _drain_pending_tasks(loop: asyncio.AbstractEventLoop) -> None:
    """Cancel and await every task still pending on *loop* before it closes.

    Mirrors ``asyncio.run()``'s own internal shutdown sequence
    (``_cancel_all_tasks``). Without this, a task still suspended
    mid-``await`` when the loop is stopped -- the server's own root task
    (``run_http_async``), or an internal one it spawns (e.g.
    sse_starlette's ``_shutdown_watcher``) -- is only unwound later at
    uncontrolled garbage-collection time, outside of any running task
    context, which crashes anyio's shielded lifespan cleanup with an
    unraisable ``TypeError``/``NoEventLoopError`` instead of shutting down
    cleanly (task 2741).
    """
    pending = asyncio.all_tasks(loop)
    if not pending:
        return
    for task in pending:
        task.cancel()
    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))


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

    Teardown is explicit: the event loop is created here in the fixture
    body (not inside the thread target), so it can be stopped thread-safely
    at teardown — ``loop.call_soon_threadsafe(loop.stop)`` unblocks
    ``run_until_complete`` in the serving thread, which is then joined
    (bounded to 5s) before this fixture finishes tearing down. This
    prevents the daemon thread's event loop from outliving the test and
    acting as a background ``time.monotonic()`` caller for the rest of the
    process (task 2741). Any task still pending at that point is cancelled
    and drained (``_drain_pending_tasks``) inside the serving thread before
    the loop closes, so cleanup runs inside a live task context instead of
    at uncontrolled GC time. A ``stopping`` flag distinguishes that
    deliberate, stop-induced ``RuntimeError`` from a genuine startup
    failure, so ``serve_error`` below still reflects only real errors
    (mirrors ``test_status_authority_gate.py``'s ``http_server`` fixture).
    A teardown that fails to stop the thread within the 5s join bound is
    asserted rather than swallowed, so a genuine hang surfaces loudly
    instead of silently leaking a daemon thread past this fixture.
    """
    queue_dir = tmp_path_factory.mktemp('esc_capability_guard')
    queue = EscalationQueue(queue_dir)
    mcp = create_server(queue, startup_sweep=False)
    port = _free_port()
    loop = asyncio.new_event_loop()
    serve_error: BaseException | None = None
    stopping = False

    def _serve_forever() -> None:
        nonlocal serve_error
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                mcp.run_http_async(
                    host='127.0.0.1', port=port, show_banner=False, log_level='error',
                )
            )
        except RuntimeError as exc:
            # Expected when teardown stops the loop mid-serve ("Event loop
            # stopped before Future completed") -- but only when *we*
            # induced it; a RuntimeError raised before teardown began is a
            # genuine startup failure and must be surfaced below.
            if not stopping:
                serve_error = exc
        finally:
            _drain_pending_tasks(loop)
            loop.close()

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
        detail = f' (server thread raised: {serve_error!r})' if serve_error else ''
        raise RuntimeError(
            f'escalation HTTP test server did not become ready on '
            f'127.0.0.1:{port} within 10s{detail}'
        )

    try:
        yield f'http://127.0.0.1:{port}', queue
    finally:
        stopping = True
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5.0)
        assert not thread.is_alive(), (
            'escalation-capability-guard-http serving thread did not stop '
            'within the 5s teardown bound -- event loop stop is hung or the '
            'server task is stuck in a non-cancellable await (task 2741)'
        )


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


async def _stamp_triage_over_http(
    base_url: str,
    *,
    levels: str | None = None,
    identity: str | None = None,
    **stamp_kwargs: Any,
) -> dict[str, Any]:
    """Call ``stamp_triage`` over real HTTP, optionally with capability headers.

    Mirrors ``_resolve_over_http`` / ``_promote_over_http`` — used to prove
    ``stamp_triage`` is never gated by X-Escalation-Levels (a triage-ack
    annotation, not a state transition) while ``triaged_by`` is still
    server-attributed from X-Escalation-Identity when present.
    """
    headers: dict[str, str] = {}
    if levels is not None:
        headers['X-Escalation-Levels'] = levels
    if identity is not None:
        headers['X-Escalation-Identity'] = identity
    transport = StreamableHttpTransport(f'{base_url}/mcp/', headers=headers)
    async with Client(transport) as client:
        result = await client.call_tool('stamp_triage', stamp_kwargs)
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

    def test_http_server_fixture_stops_serving_thread_on_teardown(
        self, tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """Regression test (task 2741): the module-scoped ``http_server``
        fixture must explicitly stop its daemon serving thread + event loop
        at teardown instead of relying on process exit to kill it.

        Drives the fixture's own generator directly via ``__wrapped__``
        (bypassing pytest's fixture caching) so this test's own serving
        thread can be identified precisely — via a ``threading.enumerate()``
        before/after diff — independently of the module-scoped fixture
        instance already serving this class's other tests under the same
        thread name.
        """
        before = set(threading.enumerate())
        gen = http_server.__wrapped__(tmp_path_factory)  # pyright: ignore[reportAttributeAccessIssue]
        try:
            base_url, queue = next(gen)
            new = [
                t for t in set(threading.enumerate()) - before
                if t.name == 'escalation-capability-guard-http'
            ]
            assert len(new) == 1, (
                f'Expected exactly one new escalation-capability-guard-http '
                f'serving thread; found {new}'
            )
            serving = new[0]
            assert serving.is_alive(), 'Serving thread must be alive during yield'

            with pytest.raises(StopIteration):
                next(gen)

            assert not serving.is_alive(), (
                'http_server fixture leaked its daemon serving thread past '
                'teardown (generator finalized via StopIteration) -- the '
                'fixture must explicitly stop its event loop and join the '
                'thread instead of relying on process exit to kill it.'
            )
        finally:
            # Best-effort: ensure finalization ran even if an assertion
            # above failed, so a RED failure does not leave extra servers
            # running for the rest of the suite. No-op if already exhausted.
            gen.close()


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


# ---------------------------------------------------------------------------
# TestIdentityDerivedCeiling: X-Escalation-Identity derives a role-based level
# ceiling server-side (PRD task-status-authority C8/D7) — closes the
# general-case default-open hole left by 2041's header-opt-in guard. A
# present X-Escalation-Levels header may only NARROW within the role
# ceiling, never widen it; an identity absent from
# escalation.authority.ROLE_LEVEL_ALLOWLIST keeps the 2041 header-opt-in
# fallback.
# ---------------------------------------------------------------------------


class TestIdentityDerivedCeiling:
    """A mapped identity is capped at its role ceiling even with no levels
    header (PRD row C2); a header cannot widen past that ceiling (PRD row
    C3) but can narrow it further. An unmapped identity is unaffected —
    it keeps the pre-existing 2041 header-opt-in behaviour.
    """

    @pytest.mark.asyncio
    async def test_mapped_identity_denied_l2_close_only_no_levels_header(
        self, http_server: tuple[str, EscalationQueue],
    ) -> None:
        """PRD C2 — identity alone caps at {0,1}; no levels header sent."""
        base_url, queue = http_server
        esc = _seed(queue, level=2, task_id='task-ceiling-c2')

        result = await _resolve_over_http(
            base_url, identity='orchestrator-escalation-watcher-auto',
            escalation_id=esc.id, resolution='x', action='close_only',
        )
        assert result.get('code') == 'level_forbidden', (
            f"Expected code='level_forbidden', got: {result}"
        )
        reread = queue.get(esc.id)
        assert reread is not None
        assert reread.status == 'pending', f'Expected pending, got: {reread.status}'
        assert (queue.queue_dir / f'{esc.id}.json').exists(), (
            'Denied record must remain in the queue root (not archived)'
        )
        assert reread.resolution_action is None, (
            f'Expected no resolution_action stamp, got: {reread.resolution_action}'
        )

    @pytest.mark.asyncio
    async def test_mapped_identity_levels_header_cannot_widen_past_ceiling(
        self, http_server: tuple[str, EscalationQueue],
    ) -> None:
        """PRD C3 — levels='0,1,2' cannot widen the {0,1} role ceiling."""
        base_url, queue = http_server
        esc = _seed(queue, level=2, task_id='task-ceiling-c3-widen')

        result = await _resolve_over_http(
            base_url, levels='0,1,2', identity='orchestrator-escalation-watcher-auto',
            escalation_id=esc.id, resolution='x', action='close_only',
        )
        assert result.get('code') == 'level_forbidden', (
            f"Expected code='level_forbidden', got: {result}"
        )
        reread = queue.get(esc.id)
        assert reread is not None
        assert reread.status == 'pending', f'Expected pending, got: {reread.status}'
        assert reread.resolution_action is None, (
            f'Expected no resolution_action stamp, got: {reread.resolution_action}'
        )

    @pytest.mark.asyncio
    async def test_mapped_identity_levels_header_narrows_ceiling(
        self, http_server: tuple[str, EscalationQueue],
    ) -> None:
        """levels='0' narrows the {0,1} ceiling to {0}, excluding an L1 record."""
        base_url, queue = http_server
        esc = _seed(queue, level=1, task_id='task-ceiling-narrow')

        result = await _resolve_over_http(
            base_url, levels='0', identity='orchestrator-escalation-watcher-auto',
            escalation_id=esc.id, resolution='x', action='close_only',
        )
        assert result.get('code') == 'level_forbidden', (
            f"Expected code='level_forbidden', got: {result}"
        )
        reread = queue.get(esc.id)
        assert reread is not None
        assert reread.status == 'pending', f'Expected pending, got: {reread.status}'
        assert reread.resolution_action is None, (
            f'Expected no resolution_action stamp, got: {reread.resolution_action}'
        )

    @pytest.mark.asyncio
    async def test_mapped_identity_within_ceiling_succeeds_and_stamps(
        self, http_server: tuple[str, EscalationQueue],
    ) -> None:
        """Identity alone (no levels header) allows an L1 resolve; stamps resolved_by."""
        base_url, queue = http_server
        esc = _seed(queue, level=1, task_id='task-ceiling-positive')

        result = await _resolve_over_http(
            base_url, identity='orchestrator-escalation-watcher-auto',
            escalation_id=esc.id, resolution='fixed', action='resume',
            resolved_by='spoofed',
        )
        assert 'code' not in result and 'error' not in result, (
            f'Expected a clean success (level 1 is within the {{0,1}} ceiling), got: {result}'
        )
        reread = queue.get(esc.id)
        assert reread is not None
        assert reread.status == 'resolved', f'Expected resolved, got: {reread.status}'
        assert reread.resolved_by == 'orchestrator-escalation-watcher-auto', (
            f'Expected the identity to stamp resolved_by, got: {reread.resolved_by!r}'
        )

    @pytest.mark.asyncio
    async def test_unmapped_identity_header_less_stays_open(
        self, http_server: tuple[str, EscalationQueue],
    ) -> None:
        """An identity absent from ROLE_LEVEL_ALLOWLIST + no levels header
        preserves the 2041 fallback: unmapped+header-less stays open."""
        base_url, queue = http_server
        esc = _seed(queue, level=2, task_id='task-ceiling-unmapped-open')

        result = await _resolve_over_http(
            base_url, identity='some-other-agent',
            escalation_id=esc.id, resolution='ok', action='close_only',
        )
        assert 'code' not in result and 'error' not in result, (
            f'Expected a clean success (unmapped identity, no levels header), got: {result}'
        )
        reread = queue.get(esc.id)
        assert reread is not None
        assert reread.status in {'resolved', 'dismissed'}, (
            f'Expected archived status, got: {reread.status}'
        )

    @pytest.mark.asyncio
    async def test_unmapped_identity_with_levels_header_still_gated(
        self, http_server: tuple[str, EscalationQueue],
    ) -> None:
        """An unmapped identity + a levels header still applies the 2041 gate."""
        base_url, queue = http_server
        esc = _seed(queue, level=2, task_id='task-ceiling-unmapped-gated')

        result = await _resolve_over_http(
            base_url, levels='0,1', identity='some-other-agent',
            escalation_id=esc.id, resolution='x', action='close_only',
        )
        assert result.get('code') == 'level_forbidden', (
            f"Expected code='level_forbidden', got: {result}"
        )
        reread = queue.get(esc.id)
        assert reread is not None
        assert reread.status == 'pending', f'Expected pending, got: {reread.status}'

    @pytest.mark.asyncio
    async def test_real_watcher_identity_still_capped_without_levels_header(
        self, http_server: tuple[str, EscalationQueue],
    ) -> None:
        """Proves the deployed watcher is still capped if it drops its levels header."""
        from orchestrator.harness import _WATCHER_ESCALATION_HEADERS

        base_url, queue = http_server
        identity = _WATCHER_ESCALATION_HEADERS['X-Escalation-Identity']
        esc = _seed(queue, level=2, task_id='task-ceiling-real-watcher-no-levels')

        result = await _resolve_over_http(
            base_url, identity=identity,
            escalation_id=esc.id, resolution='x', action='close_only',
        )
        assert result.get('code') == 'level_forbidden', (
            f"Expected code='level_forbidden', got: {result}"
        )
        reread = queue.get(esc.id)
        assert reread is not None
        assert reread.status == 'pending', f'Expected pending, got: {reread.status}'


# ---------------------------------------------------------------------------
# TestPromoteL2Gate: PROMOTE_ALLOWED gates the create-side of promote_to_l2
# on identity (PRD task-status-authority C8/D7 row C4) — an identity not in
# PROMOTE_ALLOWED may not mint a new L2 cluster.
# ---------------------------------------------------------------------------


class TestPromoteL2Gate:
    """An identity absent from PROMOTE_ALLOWED is denied promote_to_l2 (no L2
    minted); header-less callers and the deployed auto-watcher are unaffected.
    """

    @pytest.mark.asyncio
    async def test_disallowed_identity_denied_no_l2_minted(
        self, http_server: tuple[str, EscalationQueue],
    ) -> None:
        """PRD C4 — identity not in PROMOTE_ALLOWED => level_forbidden, no L2 minted."""
        base_url, queue = http_server
        m1 = _seed(queue, level=1, task_id='task-promote-gate-denied-m1')
        root_cause = 'rc-promote-gate-denied'

        result = await _promote_over_http(
            base_url, identity='some-other-agent',
            task_id='task-promote-gate-denied', agent_role='some-other-agent',
            member_ids=[m1.id], root_cause=root_cause, evidence='e',
            options=['A', 'B'], summary='cluster',
        )
        assert result.get('code') == 'level_forbidden', (
            f"Expected code='level_forbidden', got: {result}"
        )
        assert queue.find_pending_l2_by_root_cause(root_cause) is None, (
            'Expected no L2 minted for a disallowed identity'
        )

    @pytest.mark.asyncio
    async def test_header_less_allowed(
        self, http_server: tuple[str, EscalationQueue],
    ) -> None:
        """Header-less (no identity) callers remain allowed — unchanged."""
        base_url, queue = http_server
        m1 = _seed(queue, level=1, task_id='task-promote-gate-headerless-m1')
        root_cause = 'rc-promote-gate-headerless'

        result = await _promote_over_http(
            base_url,
            task_id='task-promote-gate-headerless', agent_role='escalation-watcher-auto',
            member_ids=[m1.id], root_cause=root_cause, evidence='e',
            options=['A', 'B'], summary='cluster',
        )
        assert result.get('status') in {'created', 'updated'}, (
            f"Expected status in {{'created','updated'}}, got: {result}"
        )

    @pytest.mark.asyncio
    async def test_real_watcher_identity_still_allowed(
        self, http_server: tuple[str, EscalationQueue],
    ) -> None:
        """No-op regression guard: the deployed watcher can still mint L2."""
        from orchestrator.harness import _WATCHER_ESCALATION_HEADERS

        base_url, queue = http_server
        identity = _WATCHER_ESCALATION_HEADERS['X-Escalation-Identity']
        m1 = _seed(queue, level=1, task_id='task-promote-gate-realwatcher-m1')
        root_cause = 'rc-promote-gate-realwatcher'

        result = await _promote_over_http(
            base_url, identity=identity,
            task_id='task-promote-gate-realwatcher', agent_role='escalation-watcher-auto',
            member_ids=[m1.id], root_cause=root_cause, evidence='e',
            options=['A', 'B'], summary='cluster',
        )
        assert result.get('status') == 'created', (
            f"Expected status='created', got: {result}"
        )


# ---------------------------------------------------------------------------
# TestL2AutoCloseCarveout (task 2630): a narrow above-ceiling close_only
# carve-out for the auto-watcher identity at L2 — an allowlisted class match
# plus required structural evidence in the resolution text lets
# resolve_issue fall through the {0,1} ceiling gate instead of returning
# level_forbidden. Everything else at L2 (other actions, non-allowlisted
# categories, missing evidence, the design_concern/milestone_gate/
# orchestrator-deterministic denylist) stays forbidden; a header-less human
# connection retains full L2 authority untouched (esc-2087-2).
# ---------------------------------------------------------------------------


class TestL2AutoCloseCarveout:
    """Task 2630's narrow ``close_only``-only carve-out above the auto-watcher's
    ``{0,1}`` level ceiling (``escalation.authority.l2_auto_close_class``).

    Unless noted, these drive the REAL
    ``orchestrator.harness._WATCHER_ESCALATION_HEADERS`` constant
    (levels='0,1', identity='orchestrator-escalation-watcher-auto') against
    the running server — the same header pair the deployed watcher sends —
    for two-way lockstep fidelity (mirrors TestWatcherConstantLockstep).

    RED at this step: ``resolve_issue`` does not yet consult
    ``l2_auto_close_class``, so both positive scenarios below (which must
    succeed once step-4 wires the carve-out in) currently return
    ``level_forbidden`` like everything else at L2.
    """

    @pytest.mark.asyncio
    async def test_allowlisted_classes_close_and_stamp_identity(
        self, http_server: tuple[str, EscalationQueue],
    ) -> None:
        """(1) Class (a) superseded_main_sweep and class (b) self_cleared_infra
        both close (dismissed, resolution_action='close_only') with
        resolved_by stamped from the identity header — NOT the spoofed tool
        arg — and the record archived out of the queue root."""
        from orchestrator.harness import _WATCHER_ESCALATION_HEADERS

        base_url, queue = http_server
        levels = _WATCHER_ESCALATION_HEADERS['X-Escalation-Levels']
        identity = _WATCHER_ESCALATION_HEADERS['X-Escalation-Identity']

        # (a) superseded_main_sweep — newer sweep esc id + ancestor/green proof,
        # mirrors harness._close_superseded_main_sweep_escalations's own filing
        # shape (category='infra_issue', task_id='main-sweep-<sha12>').
        esc_a = _seed(
            queue, level=2, task_id='main-sweep-abc123def456',
            agent_role='orchestrator-main-sweep', category='infra_issue',
        )
        result_a = await _resolve_over_http(
            base_url, levels=levels, identity=identity,
            escalation_id=esc_a.id, action='close_only',
            resolution=(
                'Superseded by newer sweep escalation esc-main-sweep-9f8e7d6c5b4a; '
                'main advanced and verifies clean; swept SHA abc123def456 '
                'is-ancestor of the current clean tip.'
            ),
            resolved_by='spoofed-by-agent',
        )
        assert 'code' not in result_a and 'error' not in result_a, (
            f'Expected a clean success (allowlisted class + evidence), got: {result_a}'
        )
        reread_a = queue.get(esc_a.id)
        assert reread_a is not None
        assert reread_a.status == 'dismissed', f'Expected dismissed, got: {reread_a.status}'
        assert reread_a.resolution_action == 'close_only', (
            f"Expected resolution_action='close_only', got: {reread_a.resolution_action!r}"
        )
        assert reread_a.resolved_by == 'orchestrator-escalation-watcher-auto', (
            f'Expected the identity header to stamp resolved_by (not the spoofed '
            f'tool arg), got: {reread_a.resolved_by!r}'
        )
        assert not (queue.queue_dir / f'{esc_a.id}.json').exists(), (
            'Expected the auto-closed record archived out of the queue root'
        )

        # (b) self_cleared_infra — a quoted live-probe key=value liveness token,
        # mirrors harness._revalidate_open_deterministic_escalation.
        esc_b = _seed(
            queue, level=2, task_id='task-self-cleared-infra',
            agent_role='escalation-watcher-auto', category='infra_issue',
        )
        result_b = await _resolve_over_http(
            base_url, levels=levels, identity=identity,
            escalation_id=esc_b.id, action='close_only',
            resolution='live probe: curator paused=false — transient infra self-cleared',
            resolved_by='spoofed-by-agent',
        )
        assert 'code' not in result_b and 'error' not in result_b, (
            f'Expected a clean success (allowlisted class + evidence), got: {result_b}'
        )
        reread_b = queue.get(esc_b.id)
        assert reread_b is not None
        assert reread_b.status == 'dismissed', f'Expected dismissed, got: {reread_b.status}'
        assert reread_b.resolution_action == 'close_only', (
            f"Expected resolution_action='close_only', got: {reread_b.resolution_action!r}"
        )
        assert reread_b.resolved_by == 'orchestrator-escalation-watcher-auto', (
            f'Expected the identity header to stamp resolved_by, got: {reread_b.resolved_by!r}'
        )
        assert not (queue.queue_dir / f'{esc_b.id}.json').exists(), (
            'Expected the auto-closed record archived out of the queue root'
        )

    @pytest.mark.asyncio
    async def test_stale_task_scoped_non_infra_category_closes_over_http(
        self, http_server: tuple[str, EscalationQueue],
    ) -> None:
        """(1b) Class (c) stale_task_scoped is the broadest, category-agnostic
        class — unlike (a)/(b) it can auto-close non-infra categories too
        (e.g. task_failure). Verify the full end-to-end mutation (dismissed,
        resolution_action stamped, resolved_by from the identity header,
        archived out of the queue root) over HTTP, not just the pure
        ``l2_auto_close_class`` unit return value."""
        from orchestrator.harness import _WATCHER_ESCALATION_HEADERS

        base_url, queue = http_server
        levels = _WATCHER_ESCALATION_HEADERS['X-Escalation-Levels']
        identity = _WATCHER_ESCALATION_HEADERS['X-Escalation-Identity']

        esc = _seed(
            queue, level=2, task_id='task-stale-task-scoped',
            agent_role='implementer', category='task_failure',
        )
        result = await _resolve_over_http(
            base_url, levels=levels, identity=identity,
            escalation_id=esc.id, action='close_only',
            resolution='Subject task status=done per get_task; escalation moot.',
            resolved_by='spoofed-by-agent',
        )
        assert 'code' not in result and 'error' not in result, (
            f'Expected a clean success (stale_task_scoped class + evidence), got: {result}'
        )
        reread = queue.get(esc.id)
        assert reread is not None
        assert reread.status == 'dismissed', f'Expected dismissed, got: {reread.status}'
        assert reread.resolution_action == 'close_only', (
            f"Expected resolution_action='close_only', got: {reread.resolution_action!r}"
        )
        assert reread.resolved_by == 'orchestrator-escalation-watcher-auto', (
            f'Expected the identity header to stamp resolved_by (not the spoofed '
            f'tool arg), got: {reread.resolved_by!r}'
        )
        assert not (queue.queue_dir / f'{esc.id}.json').exists(), (
            'Expected the auto-closed record archived out of the queue root'
        )

    @pytest.mark.asyncio
    async def test_non_allowlisted_category_still_forbidden(
        self, http_server: tuple[str, EscalationQueue],
    ) -> None:
        """(2) A category outside the allowlist (scope_violation, no class-c
        status citation either) stays level_forbidden — no mutation."""
        from orchestrator.harness import _WATCHER_ESCALATION_HEADERS

        base_url, queue = http_server
        levels = _WATCHER_ESCALATION_HEADERS['X-Escalation-Levels']
        identity = _WATCHER_ESCALATION_HEADERS['X-Escalation-Identity']
        esc = _seed(
            queue, level=2, task_id='task-non-allowlisted-category',
            agent_role='implementer', category='scope_violation',
        )

        result = await _resolve_over_http(
            base_url, levels=levels, identity=identity,
            escalation_id=esc.id, action='close_only', resolution='x',
        )
        assert result.get('code') == 'level_forbidden', (
            f"Expected code='level_forbidden', got: {result}"
        )
        reread = queue.get(esc.id)
        assert reread is not None
        assert reread.status == 'pending', f'Expected pending, got: {reread.status}'
        assert (queue.queue_dir / f'{esc.id}.json').exists(), (
            'Denied record must remain in the queue root (not archived)'
        )
        assert reread.resolution_action is None, (
            f'Expected no resolution_action stamp, got: {reread.resolution_action}'
        )

    @pytest.mark.asyncio
    async def test_action_other_than_close_only_still_forbidden(
        self, http_server: tuple[str, EscalationQueue],
    ) -> None:
        """(3) An otherwise-matching record (infra_issue + valid probe evidence)
        stays level_forbidden when action is 'resume' or 'park' — only
        close_only is carved out."""
        from orchestrator.harness import _WATCHER_ESCALATION_HEADERS

        base_url, queue = http_server
        levels = _WATCHER_ESCALATION_HEADERS['X-Escalation-Levels']
        identity = _WATCHER_ESCALATION_HEADERS['X-Escalation-Identity']
        probe_resolution = 'live probe: curator paused=false — transient infra self-cleared'

        esc_resume = _seed(
            queue, level=2, task_id='task-action-resume-denied',
            agent_role='implementer', category='infra_issue',
        )
        result_resume = await _resolve_over_http(
            base_url, levels=levels, identity=identity,
            escalation_id=esc_resume.id, action='resume', resolution=probe_resolution,
        )
        assert result_resume.get('code') == 'level_forbidden', (
            f"Expected code='level_forbidden', got: {result_resume}"
        )
        reread_resume = queue.get(esc_resume.id)
        assert reread_resume is not None
        assert reread_resume.status == 'pending', (
            f'Expected pending, got: {reread_resume.status}'
        )
        assert reread_resume.resolution_action is None, (
            f'Expected no resolution_action stamp, got: {reread_resume.resolution_action}'
        )

        esc_park = _seed(
            queue, level=2, task_id='task-action-park-denied',
            agent_role='implementer', category='infra_issue',
        )
        result_park = await _resolve_over_http(
            base_url, levels=levels, identity=identity,
            escalation_id=esc_park.id, action='park', resolution=probe_resolution,
        )
        assert result_park.get('code') == 'level_forbidden', (
            f"Expected code='level_forbidden', got: {result_park}"
        )
        reread_park = queue.get(esc_park.id)
        assert reread_park is not None
        assert reread_park.status == 'pending', f'Expected pending, got: {reread_park.status}'
        assert reread_park.resolution_action is None, (
            f'Expected no resolution_action stamp, got: {reread_park.resolution_action}'
        )
        assert reread_park.resolution is None, (
            f'Expected no park stamp at all, got: {reread_park.resolution!r}'
        )

    @pytest.mark.asyncio
    async def test_missing_evidence_still_forbidden(
        self, http_server: tuple[str, EscalationQueue],
    ) -> None:
        """(4) An allowlisted category (infra_issue) with close_only but a
        resolution carrying no structural evidence token stays level_forbidden."""
        from orchestrator.harness import _WATCHER_ESCALATION_HEADERS

        base_url, queue = http_server
        levels = _WATCHER_ESCALATION_HEADERS['X-Escalation-Levels']
        identity = _WATCHER_ESCALATION_HEADERS['X-Escalation-Identity']
        esc = _seed(
            queue, level=2, task_id='task-missing-evidence',
            agent_role='implementer', category='infra_issue',
        )

        result = await _resolve_over_http(
            base_url, levels=levels, identity=identity,
            escalation_id=esc.id, action='close_only', resolution='looks fine now',
        )
        assert result.get('code') == 'level_forbidden', (
            f"Expected code='level_forbidden', got: {result}"
        )
        reread = queue.get(esc.id)
        assert reread is not None
        assert reread.status == 'pending', f'Expected pending, got: {reread.status}'
        assert (queue.queue_dir / f'{esc.id}.json').exists(), (
            'Denied record must remain in the queue root (not archived)'
        )
        assert reread.resolution_action is None, (
            f'Expected no resolution_action stamp, got: {reread.resolution_action}'
        )

    @pytest.mark.asyncio
    async def test_denylist_categories_and_roles_still_forbidden(
        self, http_server: tuple[str, EscalationQueue],
    ) -> None:
        """(5) The denylist wins even when a class predicate would otherwise
        match: a design_concern record carrying valid class-c evidence, and an
        orchestrator-deterministic-filed infra_issue carrying valid class-b
        evidence, both stay level_forbidden."""
        from orchestrator.harness import _WATCHER_ESCALATION_HEADERS

        base_url, queue = http_server
        levels = _WATCHER_ESCALATION_HEADERS['X-Escalation-Levels']
        identity = _WATCHER_ESCALATION_HEADERS['X-Escalation-Identity']

        # (a) design_concern — a born-at-L2 human-judgment category — is never
        # auto-closable even though the resolution cites a terminal task status
        # (class (c)'s evidence).
        esc_design = _seed(
            queue, level=2, task_id='task-denylist-design-concern',
            agent_role='implementer', category='design_concern',
        )
        result_design = await _resolve_over_http(
            base_url, levels=levels, identity=identity,
            escalation_id=esc_design.id, action='close_only',
            resolution='Subject task main-sweep-abc status=done (terminal); safe to close.',
        )
        assert result_design.get('code') == 'level_forbidden', (
            f"Expected code='level_forbidden', got: {result_design}"
        )
        reread_design = queue.get(esc_design.id)
        assert reread_design is not None
        assert reread_design.status == 'pending', (
            f'Expected pending, got: {reread_design.status}'
        )
        assert reread_design.resolution_action is None, (
            f'Expected no resolution_action stamp, got: {reread_design.resolution_action}'
        )

        # (b) orchestrator-deterministic — the born-at-L2 human-gate sentinel
        # role — is never auto-closable even for an infra_issue carrying valid
        # class (b) evidence.
        esc_deterministic = _seed(
            queue, level=2, task_id='task-denylist-deterministic-role',
            agent_role='orchestrator-deterministic', category='infra_issue',
        )
        result_deterministic = await _resolve_over_http(
            base_url, levels=levels, identity=identity,
            escalation_id=esc_deterministic.id, action='close_only',
            resolution='live probe: ActiveState=active',
        )
        assert result_deterministic.get('code') == 'level_forbidden', (
            f"Expected code='level_forbidden', got: {result_deterministic}"
        )
        reread_deterministic = queue.get(esc_deterministic.id)
        assert reread_deterministic is not None
        assert reread_deterministic.status == 'pending', (
            f'Expected pending, got: {reread_deterministic.status}'
        )
        assert reread_deterministic.resolution_action is None, (
            f'Expected no resolution_action stamp, got: {reread_deterministic.resolution_action}'
        )

    @pytest.mark.asyncio
    async def test_header_less_human_connection_unaffected(
        self, http_server: tuple[str, EscalationQueue],
    ) -> None:
        """(6) A header-less (human) connection closing a class-matching
        infra_issue L2 succeeds via its PRE-EXISTING full L2 authority
        (esc-2087-2) — resolved_by comes from the tool arg, never the carve-out
        (which never engages: identity is None)."""
        base_url, queue = http_server
        esc = _seed(
            queue, level=2, task_id='task-header-less-infra',
            agent_role='escalation-watcher-auto', category='infra_issue',
        )

        result = await _resolve_over_http(
            base_url,
            escalation_id=esc.id, action='close_only',
            resolution='live probe: curator paused=false — transient infra self-cleared',
            resolved_by='escalation-watcher',
        )
        assert 'code' not in result and 'error' not in result, (
            f'Expected a clean success (header-less keeps full L2 authority), got: {result}'
        )
        reread = queue.get(esc.id)
        assert reread is not None
        assert reread.status == 'dismissed', f'Expected dismissed, got: {reread.status}'
        assert reread.resolved_by == 'escalation-watcher', (
            f'Expected the tool-arg resolved_by (no Identity header to override it), '
            f'got: {reread.resolved_by!r}'
        )


# ---------------------------------------------------------------------------
# TestTriageAnnotationUngated (task 2555): stamp_triage is NOT gated by the
# connection-capability level check — a {0,1}-level-capped connection (the
# deployed auto-watcher) can annotate a pending L2 it is still forbidden to
# resolve. triaged_by is server-attributed from X-Escalation-Identity,
# non-spoofable, mirroring resolve_issue's resolved_by override.
# ---------------------------------------------------------------------------


class TestTriageAnnotationUngated:
    """A {0,1}-capped connection can stamp triage on a pending L2 (ungated,
    server-attributed identity) but is still denied resolve_issue(action=
    'resume') on that same record — proving triage is an annotation, not a
    resolution.

    Uses action='resume' (not close_only) for the deny assertion so it stays
    robust against task 2630's narrow l2_auto_close_class close_only
    carve-out — resume/restart/park/abandon are unconditionally
    level_forbidden at L2 for the {0,1} cap.
    """

    @pytest.mark.asyncio
    async def test_capped_connection_stamps_but_cannot_resolve(
        self, http_server: tuple[str, EscalationQueue],
    ) -> None:
        base_url, queue = http_server
        esc = _seed(queue, level=2, task_id='task-triage-ungated')

        # (a) A {0,1}-capped connection successfully stamps triage on the
        # pending L2 — the result is the record dict, not level_forbidden.
        # triaged_by is server-attributed from the identity header, winning
        # over a spoofed tool arg.
        result_a = await _stamp_triage_over_http(
            base_url, levels='0,1', identity='orchestrator-escalation-watcher-auto',
            escalation_id=esc.id,
            triaged_by='spoofed-by-agent',
            triage_note='task-604 status==done | probe: get_task 604 -> status=done',
        )
        assert result_a.get('code') != 'level_forbidden', (
            f'Expected the annotation to succeed (ungated), got: {result_a}'
        )
        assert 'error' not in result_a, f'Unexpected error: {result_a}'
        assert result_a['id'] == esc.id
        assert result_a['triaged_at'] is not None
        assert result_a['triaged_by'] == 'orchestrator-escalation-watcher-auto', (
            f'Expected the identity header to win over the spoofed tool arg, got: '
            f"{result_a['triaged_by']!r}"
        )
        assert result_a['triage_note'] == (
            'task-604 status==done | probe: get_task 604 -> status=done'
        )
        assert result_a['status'] == 'pending', f"Expected pending, got: {result_a['status']}"

        reread_a = queue.get(esc.id)
        assert reread_a is not None
        assert reread_a.status == 'pending'
        assert reread_a.triaged_at is not None
        assert reread_a.triaged_by == 'orchestrator-escalation-watcher-auto', (
            f'Expected the on-disk record to carry the header-attributed identity, '
            f'got: {reread_a.triaged_by!r}'
        )
        assert reread_a.triage_note == (
            'task-604 status==done | probe: get_task 604 -> status=done'
        )
        stamped_triaged_at = reread_a.triaged_at

        # (b) The SAME capped connection is still denied a resolve (action=
        # 'resume') on the same record — no mutation, triaged_at unchanged.
        result_b = await _resolve_over_http(
            base_url, levels='0,1', identity='orchestrator-escalation-watcher-auto',
            escalation_id=esc.id, resolution='x', action='resume',
        )
        assert result_b.get('code') == 'level_forbidden', (
            f"Expected code='level_forbidden', got: {result_b}"
        )
        reread_b = queue.get(esc.id)
        assert reread_b is not None
        assert reread_b.status == 'pending', f'Expected pending, got: {reread_b.status}'
        assert (queue.queue_dir / f'{esc.id}.json').exists(), (
            'Denied record must remain in the queue root (not archived)'
        )
        assert reread_b.resolution_action is None, (
            f'Expected no resolution_action stamp, got: {reread_b.resolution_action}'
        )
        assert reread_b.triaged_at == stamped_triaged_at, (
            'Expected the triage stamp from (a) unchanged by the denied resolve attempt'
        )
