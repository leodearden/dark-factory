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
from shared.task_statuses import TaskStatus
from shared.task_transitions import ActorClass, is_legal_transition

from escalation.action_effects import ACTION_EFFECTS, ANY, TaskEffect, effect_for
from escalation.authority import PROMOTE_ALLOWED, ROLE_LEVEL_ALLOWLIST
from escalation.models import Escalation
from escalation.queue import EscalationQueue
from escalation.server import create_server

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


# ---------------------------------------------------------------------------
# C1 — header-less connection retains full L2 authority (esc-2087-2).
# ---------------------------------------------------------------------------


class TestC1HeaderlessFullL2Authority:
    """C1: a header-less HTTP connection (no X-Escalation-Levels /
    X-Escalation-Identity) is NEVER narrowed by the capability guard — it
    keeps full authority to resolve an L2 record via resume, park, AND
    close_only (the esc-2087-2 human-channel guarantee). Only the real HTTP
    harness can exercise this: get_http_headers() always resolves {} for an
    in-process tool.fn() call, which is indistinguishable from "header-less
    over HTTP" for this gate — proved directly here instead."""

    @pytest.mark.asyncio
    async def test_resume_park_close_only_all_succeed_header_less(
        self, http_server: tuple[str, EscalationQueue],
    ) -> None:
        base_url, queue = http_server
        esc_resume = _seed(queue, level=2, task_id='zeta-c1-resume')
        esc_park = _seed(queue, level=2, task_id='zeta-c1-park')
        esc_close = _seed(queue, level=2, task_id='zeta-c1-close')

        result_resume = await _resolve_over_http(
            base_url, escalation_id=esc_resume.id, resolution='fixed', action='resume',
        )
        assert 'error' not in result_resume, f'Unexpected error: {result_resume}'
        reread_resume = queue.get(esc_resume.id)
        assert reread_resume is not None
        assert reread_resume.status == 'resolved', (
            f'Expected resolved, got: {reread_resume.status}'
        )

        result_park = await _resolve_over_http(
            base_url, escalation_id=esc_park.id, resolution='parked', action='park',
        )
        assert 'error' not in result_park, f'Unexpected error: {result_park}'
        reread_park = queue.get(esc_park.id)
        assert reread_park is not None
        assert reread_park.status == 'pending', (
            f'park must keep the record OPEN; got {reread_park.status}'
        )
        assert reread_park.level == 2
        assert reread_park.resolution_action == 'park'

        result_close = await _resolve_over_http(
            base_url, escalation_id=esc_close.id, resolution='n/a', action='close_only',
        )
        assert 'error' not in result_close, f'Unexpected error: {result_close}'
        reread_close = queue.get(esc_close.id)
        assert reread_close is not None
        assert reread_close.status == 'dismissed', (
            f'Expected dismissed, got: {reread_close.status}'
        )


# ---------------------------------------------------------------------------
# C2 — a mapped identity is capped at its role ceiling with no levels header.
# ---------------------------------------------------------------------------


class TestC2MappedIdentityCeiling:
    """C2: an identity mapped in ROLE_LEVEL_ALLOWLIST (the auto-watcher, {0,1})
    is capped at that ceiling even with no X-Escalation-Levels header — an L2
    record is denied and left completely unchanged (still live, unarchived)."""

    @pytest.mark.asyncio
    async def test_watcher_identity_denied_l2_no_levels_header(
        self, http_server: tuple[str, EscalationQueue],
    ) -> None:
        base_url, queue = http_server
        esc = _seed(queue, level=2, task_id='zeta-c2-denied')

        result = await _resolve_over_http(
            base_url, identity='orchestrator-escalation-watcher-auto',
            escalation_id=esc.id, resolution='x', action='close_only',
        )

        assert result.get('code') == 'level_forbidden', (
            f"Expected code='level_forbidden'; got: {result}"
        )
        reread = queue.get(esc.id)
        assert reread is not None
        assert reread.status == 'pending', f'Expected pending; got {reread.status}'
        assert reread.resolution_action is None, (
            f'Expected no resolution_action stamp; got {reread.resolution_action!r}'
        )
        assert (queue.queue_dir / f'{esc.id}.json').exists(), (
            'Denied record must remain live in the queue root (not archived)'
        )


# ---------------------------------------------------------------------------
# C3 — a levels header can only NARROW a mapped identity's ceiling, never widen.
# ---------------------------------------------------------------------------


class TestC3LevelsHeaderNarrowsNotWidens:
    """C3: X-Escalation-Levels intersects with (never extends past) a mapped
    identity's role ceiling — widening past {0,1} still fails, while a
    genuine narrow ({0,1} & {0} = {0}) further restricts what the same
    identity may touch."""

    @pytest.mark.asyncio
    async def test_widen_past_ceiling_still_denied(
        self, http_server: tuple[str, EscalationQueue],
    ) -> None:
        base_url, queue = http_server
        esc = _seed(queue, level=2, task_id='zeta-c3-widen')

        result = await _resolve_over_http(
            base_url, levels='0,1,2', identity='orchestrator-escalation-watcher-auto',
            escalation_id=esc.id, resolution='x', action='close_only',
        )

        assert result.get('code') == 'level_forbidden', (
            f"levels='0,1,2' must not widen the {{0,1}} ceiling; got: {result}"
        )
        reread = queue.get(esc.id)
        assert reread is not None
        assert reread.status == 'pending', f'Expected pending; got {reread.status}'
        assert reread.resolution_action is None

    @pytest.mark.asyncio
    async def test_narrow_below_ceiling_allows_l0_denies_l1(
        self, http_server: tuple[str, EscalationQueue],
    ) -> None:
        base_url, queue = http_server
        esc_l0 = _seed(queue, level=0, task_id='zeta-c3-narrow-l0')
        esc_l1 = _seed(queue, level=1, task_id='zeta-c3-narrow-l1')

        result_l0 = await _resolve_over_http(
            base_url, levels='0', identity='orchestrator-escalation-watcher-auto',
            escalation_id=esc_l0.id, resolution='fixed', action='resume',
        )
        assert 'error' not in result_l0, (
            f'levels=0 narrows {{0,1}} to {{0}}; an L0 record must still '
            f'resolve: {result_l0}'
        )
        reread_l0 = queue.get(esc_l0.id)
        assert reread_l0 is not None
        assert reread_l0.status == 'resolved', f'Expected resolved; got {reread_l0.status}'

        result_l1 = await _resolve_over_http(
            base_url, levels='0', identity='orchestrator-escalation-watcher-auto',
            escalation_id=esc_l1.id, resolution='x', action='close_only',
        )
        assert result_l1.get('code') == 'level_forbidden', (
            f'levels=0 narrows {{0,1}} to {{0}}; an L1 record must now be '
            f'denied: {result_l1}'
        )
        reread_l1 = queue.get(esc_l1.id)
        assert reread_l1 is not None
        assert reread_l1.status == 'pending', f'Expected pending; got {reread_l1.status}'


# ---------------------------------------------------------------------------
# C4 — promote_to_l2 is gated by identity (PROMOTE_ALLOWED), never by levels.
# ---------------------------------------------------------------------------


class TestC4PromoteToL2IdentityGate:
    """C4: promote_to_l2 is gated solely by PROMOTE_ALLOWED identity
    membership: a disallowed identity is denied with no L2 minted at all
    (find_pending_l2_by_root_cause stays None); header-less callers remain
    allowed (the esc-2087-2 guarantee applies here too)."""

    @pytest.mark.asyncio
    async def test_disallowed_identity_denied_no_l2_minted(
        self, http_server: tuple[str, EscalationQueue],
    ) -> None:
        base_url, queue = http_server
        m1 = _seed(queue, level=1, task_id='zeta-c4-denied-m1')
        root_cause = 'zeta-rc-c4-denied'

        result = await _promote_over_http(
            base_url, identity='some-other-agent',
            task_id='zeta-c4-denied', agent_role='some-other-agent',
            member_ids=[m1.id], root_cause=root_cause, evidence='e',
            options=['A', 'B'], summary='cluster',
        )

        assert result.get('code') == 'level_forbidden', (
            f"Expected code='level_forbidden'; got: {result}"
        )
        assert queue.find_pending_l2_by_root_cause(root_cause) is None, (
            'Expected no L2 minted for a disallowed identity'
        )

    @pytest.mark.asyncio
    async def test_header_less_promote_allowed(
        self, http_server: tuple[str, EscalationQueue],
    ) -> None:
        base_url, queue = http_server
        m1 = _seed(queue, level=1, task_id='zeta-c4-headerless-m1')
        root_cause = 'zeta-rc-c4-headerless'

        result = await _promote_over_http(
            base_url,
            task_id='zeta-c4-headerless', agent_role='escalation-watcher-auto',
            member_ids=[m1.id], root_cause=root_cause, evidence='e',
            options=['A', 'B'], summary='cluster',
        )

        assert result.get('status') in {'created', 'updated'}, (
            f"Expected status in {{'created','updated'}}; got: {result}"
        )


# ---------------------------------------------------------------------------
# Composition comp-1 — Table B (escalation.action_effects) targets are Table A
# (shared.task_transitions/task_statuses) vocabulary members AND legal
# transitions. PRD D1: "Table B computes intent; Table A validates legality
# at the chokepoint; never three tables." This is the load-bearing GATE
# assertion no per-cell unit test can make — it crosses both tables.
# ---------------------------------------------------------------------------

_TASK_STATUS_VALUES = frozenset(s.value for s in TaskStatus)

# Representative escalation-side SOURCE statuses per action, chosen to prove
# both a blocked-task resolution (the common case, B1) and an in-progress-task
# resolution (orphan/cascade re-pend and stranded-revert, B1/B2) reach a
# LEGAL Table A transition for the same Table B target. close_only's
# TaskEffect has target_status=None (no task-status effect), so it has no
# representative sources and is excluded from the transition check (still
# covered separately below as a recognised action).
_REPRESENTATIVE_SOURCES: dict[str, tuple[str, ...]] = {
    'resume': ('blocked',),
    'restart': ('blocked',),
    'park': ('blocked', 'in-progress'),
    'abandon': ('blocked', 'in-progress'),
}


class TestCompositionTableBSubsetOfTableA:
    """comp-1: every ACTION_EFFECTS entry with a non-None target_status names
    a real TaskStatus vocabulary member AND is reachable via a legal Table A
    ESCALATION-actor transition from its representative source(s)."""

    def test_every_target_status_is_a_taskstatus_member(self) -> None:
        for (action, _level, _category), effect in ACTION_EFFECTS.items():
            if effect.target_status is None:
                continue
            assert effect.target_status in _TASK_STATUS_VALUES, (
                f'action={action!r} targets {effect.target_status!r}, which is '
                f'not a shared.task_statuses.TaskStatus member: {sorted(_TASK_STATUS_VALUES)}'
            )

    def test_every_target_status_is_a_legal_escalation_transition(self) -> None:
        checked_actions: set[str] = set()
        for (action, _level, _category), effect in ACTION_EFFECTS.items():
            if effect.target_status is None:
                continue
            sources = _REPRESENTATIVE_SOURCES.get(action)
            assert sources is not None, (
                f'No representative source(s) registered for action {action!r}; '
                'update _REPRESENTATIVE_SOURCES to cover every ACTION_EFFECTS '
                'action with a non-None target_status.'
            )
            for src in sources:
                assert is_legal_transition(
                    src, effect.target_status, ActorClass.ESCALATION,
                ), (
                    f'Table B action={action!r} targets {effect.target_status!r}, '
                    f'but Table A rejects {src}->{effect.target_status} for '
                    f'ActorClass.ESCALATION — Table B and Table A have drifted '
                    f'apart (PRD D1 violation).'
                )
            checked_actions.add(action)

        assert checked_actions == set(_REPRESENTATIVE_SOURCES), (
            f'Expected to check exactly {set(_REPRESENTATIVE_SOURCES)}; '
            f'checked {checked_actions} (has ACTION_EFFECTS action set drifted?)'
        )

    def test_close_only_has_no_target_status_but_is_still_a_recognised_action(
        self,
    ) -> None:
        effect = ACTION_EFFECTS[('close_only', ANY, ANY)]
        assert effect.target_status is None, (
            f'close_only must have no task-status effect; got {effect.target_status!r}'
        )
        # Still resolves through effect_for — it is a recognised action, just
        # one with WORKFLOW_NONE disposition and no Table A check to make.
        assert effect_for('close_only', 0, 'scope_violation') == effect


# ---------------------------------------------------------------------------
# Composition comp-2 — ONE shared effect_for table serves both the escalation
# server (B5, resolve_issue) and the orchestrator harness (comp-2 harness
# face, _on_escalation_resolved, step-11) — level/category never narrow.
# ---------------------------------------------------------------------------


class TestCompositionSharedEffectForTable:
    """comp-2: effect_for(action, level, category) resolves a TaskEffect for
    each of the 5 RESOLVE_ACTIONS and None for an unrecognised action,
    regardless of level or category — the SAME table consumed by both the
    escalation server's resolve_issue (B5) and the orchestrator harness's
    _on_escalation_resolved (comp-2 harness face, step-11)."""

    _RESOLVE_ACTIONS = ('resume', 'restart', 'park', 'abandon', 'close_only')
    _LEVELS = (0, 1, 2)
    # Includes 'milestone_gate', a category absent from server.CATEGORIES
    # (G6 archive verification) — proves category never narrows legality.
    _CATEGORIES = ('scope_violation', 'milestone_gate')

    def test_all_five_actions_resolve_across_level_and_category(self) -> None:
        for action in self._RESOLVE_ACTIONS:
            for level in self._LEVELS:
                for category in self._CATEGORIES:
                    effect = effect_for(action, level, category)
                    assert isinstance(effect, TaskEffect), (
                        f'effect_for({action!r}, {level}, {category!r}) must '
                        f'resolve a TaskEffect (level/category never narrow); '
                        f'got {effect!r}'
                    )

    def test_unknown_action_is_none_across_level_and_category(self) -> None:
        for level in self._LEVELS:
            for category in self._CATEGORIES:
                assert effect_for('bogus', level, category) is None, (
                    f"effect_for('bogus', {level}, {category!r}) must be None "
                    f'(unrecognised action) — the same table both the server '
                    f'(B5) and the harness (comp-2 harness face) reject on.'
                )
