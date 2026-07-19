"""ζ boundary-integration-gate matrix — operational-routing two-way boundary
(PRD ``plans/operational-ask-routing-prd.md``, task ζ/2806).

Unlike most of ``fused-memory/tests``, this suite is deliberately an
INTEGRATION gate, not a synthetic-input unit test: every scenario drives the
REAL read paths — the real ``submit_task``/``resolve_ticket``/``get_task``
MCP tools backed by a real :class:`SqliteTaskBackend` + real
``TaskInterceptor`` (task state), and a real ``EscalationQueue`` (escalation
state) — rather than mocking the interceptor and inspecting the metadata
FORWARDED to it (the pattern the α–ε leaf tests already use). All production
substrate this matrix asserts on landed via four already-merged dependency
tasks:

- α/2801: ``TaskMetadata.operational_mode`` Literal['gate','llm']='gate';
  enforce-mode ``parse_metadata(direction='write', enforce=True)`` raises on
  an invalid value.
- β/2802: ``inject_operational_routing`` (operational_routing_guard.py),
  wired into both ``tools.py:submit_task`` and
  ``TaskInterceptor.submit_task`` — coerces a declared
  ``execution_class ∈ {operational, decision}`` submission to a
  deterministic pure gate; ``operational`` + ``operational_mode='llm'``
  additionally stamps the ``x_operational_llm_gate`` marker.
- γ/2803: ``DeterministicRunner`` emits the ``operational_llm_needs_lane``
  token in the born-at-L2 escalation summary/detail when that marker is
  present (orchestrator-side; see ``orchestrator/tests/test_operational_routing_boundary_matrix.py``
  for the γ/scheduler-side realization, matrix row 4b).
- δ/2804: ``operational_ask_registry.match_candidate`` returns ``None``
  early for a TAGGED operational/decision candidate (the β boundary owns
  it); the untagged-legacy substring fallback is retained.
- ε/2805: ``render_source_completion_section`` is wired into both the
  Stage 1 and Stage 2 recon system prompts.

A RED result anywhere in this module is therefore a genuine integration
gap in already-landed, already-merged code — not a synthetic-input failure.
"""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from fused_memory.server.tools import create_mcp_server


@pytest.fixture(autouse=True)
def passthrough_main_checkout(monkeypatch):
    """Stub resolve_main_checkout to pass its argument through unchanged.

    Mirrors test_task_tools.py's fixture of the same name: this suite's
    project roots are synthetic ``tmp_path``-based paths, not real git
    worktrees, so the real resolver would reject them.
    """
    monkeypatch.setattr(
        'fused_memory.server.tools.resolve_main_checkout', lambda p: str(p),
    )


async def _build_stack(tmp_path, *, enforce: bool):
    """Construct a live SqliteTaskBackend + EventBuffer + TicketStore +
    real TaskInterceptor + create_mcp_server stack.

    Mirrors ``real_task_stack`` (test_task_tools.py:51-93), with one
    addition this matrix's normal-path scenarios need: a real, non-None
    ``config`` (curator enabled — the default) so
    ``TaskInterceptor._get_curator()`` actually constructs a
    :class:`TaskCurator` instead of short-circuiting to ``None``. Scenario 1
    (step-5) neutralizes ``TaskCurator.curate`` at the class level and needs
    a REAL curator instance for that patch to matter (proving the boundary,
    not a merely-absent curator, coerced); scenario 5 (step-7) needs the
    curator's untagged-legacy substring fallback to genuinely run.
    ``config.taskmaster`` is explicitly cleared so ``_get_curator()``'s
    one-shot corpus-backfill background task never fires (it requires a
    truthy ``config.taskmaster.project_root``) — keeps the stack hermetic
    regardless of what the ambient fused-memory/config/config.yaml contains.
    ``config.curator.operational_ask_registry_path`` is set to a fixed
    placeholder so scenario 5's registry-injection helper (step-6) can
    short-circuit ``_maybe_route_deterministic``'s ``cfg_path is None``
    early-out; the placeholder is never actually read from disk because
    scenario 5 monkeypatches the loader function itself.
    """
    from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
    from fused_memory.config.schema import CuratorConfig, FusedMemoryConfig, TaskmasterConfig
    from fused_memory.middleware.task_interceptor import TaskInterceptor
    from fused_memory.middleware.ticket_store import TicketStore
    from fused_memory.reconciliation.event_buffer import EventBuffer

    backend = SqliteTaskBackend(
        TaskmasterConfig(project_root=str(tmp_path)), task_metadata_enforce=enforce,
    )
    await backend.start()
    event_buffer = EventBuffer(db_path=tmp_path / 'zeta_eb.db', buffer_size_threshold=100)
    await event_buffer.initialize()
    ticket_store = TicketStore(tmp_path / 'zeta_tickets.db')
    await ticket_store.initialize()

    config = FusedMemoryConfig()
    config.taskmaster = None
    config.curator = CuratorConfig(
        operational_ask_registry_path='zeta_unused_operational_ask_registry.yaml',
    )

    interceptor = TaskInterceptor(backend, None, event_buffer, config, ticket_store=ticket_store)
    server = create_mcp_server(AsyncMock(), task_interceptor=interceptor)
    return server, interceptor, backend, event_buffer, ticket_store


async def _teardown_stack(interceptor, ticket_store, event_buffer, backend):
    """Mirrors real_task_stack's teardown: close ticket_store, cancel any
    live curator-worker tasks, then close event_buffer and backend."""
    await ticket_store.close()
    for _wt in list(interceptor._worker_tasks.values()):
        if not _wt.done():
            _wt.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await _wt
    await event_buffer.close()
    await backend.close()


@pytest_asyncio.fixture
async def _real_stack(tmp_path):
    """Warn-mode real stack: yields ``(server, interceptor)``."""
    server, interceptor, backend, event_buffer, ticket_store = await _build_stack(
        tmp_path, enforce=False,
    )
    try:
        yield server, interceptor
    finally:
        await _teardown_stack(interceptor, ticket_store, event_buffer, backend)


@pytest_asyncio.fixture
async def _enforce_stack(tmp_path):
    """Enforce-mode real stack (``task_metadata_enforce=True``): yields
    ``(server, interceptor)``.

    Mirrors test_task_metadata_boundary.py's ``make_backend(enforce=True)``
    — scenario 6 (step-8) needs this to exercise the backend write-boundary
    rejection (the default backend runs warn-only and would not reject).
    """
    server, interceptor, backend, event_buffer, ticket_store = await _build_stack(
        tmp_path, enforce=True,
    )
    try:
        yield server, interceptor
    finally:
        await _teardown_stack(interceptor, ticket_store, event_buffer, backend)


async def _get_task_meta(server, root: str, task_id: str) -> dict:
    """Read back a task's metadata dict via the real get_task MCP tool."""
    result = await server._tool_manager.call_tool(
        'get_task', {'id': task_id, 'project_root': root},
    )
    assert 'error' not in result, f'get_task failed unexpectedly: {result!r}'
    return result['metadata']


# ---------------------------------------------------------------------------
# Scenario 2 — gate via planning_mode (matrix row 1/2: operational+gate)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planning_mode_operational_gate_coerces_to_pure_gate(_real_stack, tmp_path):
    """A planning_mode submit declaring execution_class='operational' +
    operational_mode='gate' is coerced to a deterministic pure gate.

    planning_mode bypasses the curator entirely (_submit_task_planning_mode
    is a synchronous, curator-free path), so this independently proves the
    submit-boundary coercion (inject_operational_routing, task β) fires on
    its own — the curator is never even consulted on this path.
    """
    server, _interceptor = _real_stack
    root = str(tmp_path / 'proj')

    result = await server._tool_manager.call_tool(
        'submit_task',
        {
            'project_root': root,
            'title': 'Zeta boundary probe: gate via planning_mode',
            'description': 'Operational human-gated ask, submitted via planning_mode.',
            'planning_mode': True,
            'metadata': {
                'execution_class': 'operational',
                'operational_mode': 'gate',
            },
        },
    )
    assert 'error' not in result, f'submit_task failed unexpectedly: {result!r}'
    task_id = result['task_id']

    meta = await _get_task_meta(server, root, task_id)

    assert meta['task_kind'] == 'deterministic', f'got {meta!r}'
    assert meta['always_escalates'] is True, f'got {meta!r}'
    assert 'before_done' not in meta, f'got {meta!r}'


# ---------------------------------------------------------------------------
# Scenario 3 — decision -> pure gate (matrix row 1/3: decision, any mode)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planning_mode_decision_coerces_to_pure_gate(_real_stack, tmp_path):
    """A planning_mode submit declaring execution_class='decision' (no
    operational_mode) is also coerced to a deterministic pure gate.

    Confirms operational_routing_guard.inject_operational_routing maps
    execution_class='decision' (any/absent operational_mode) to the same
    pure-gate shape as the operational+gate case (scenario 2).
    """
    server, _interceptor = _real_stack
    root = str(tmp_path / 'proj')

    result = await server._tool_manager.call_tool(
        'submit_task',
        {
            'project_root': root,
            'title': 'Zeta boundary probe: decision ask',
            'description': 'A judgment-call decision ask, submitted via planning_mode.',
            'planning_mode': True,
            'metadata': {'execution_class': 'decision'},
        },
    )
    assert 'error' not in result, f'submit_task failed unexpectedly: {result!r}'
    task_id = result['task_id']

    meta = await _get_task_meta(server, root, task_id)

    assert meta['task_kind'] == 'deterministic', f'got {meta!r}'
    assert meta['always_escalates'] is True, f'got {meta!r}'
    assert 'before_done' not in meta, f'got {meta!r}'
