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

from fused_memory.middleware.operational_ask_registry import OperationalAskEntry
from fused_memory.middleware.operational_routing_guard import OPERATIONAL_LLM_GATE_MARKER_KEY
from fused_memory.middleware.task_curator import CuratorDecision, TaskCurator
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


async def _submit_normal_and_get(
    stack: tuple, root: str, metadata: dict, monkeypatch: pytest.MonkeyPatch,
) -> dict:
    """Drive a NORMAL (non-planning_mode) submit_task -> resolve_ticket ->
    get_task round-trip, with the curator neutralized to a no-op 'create'.

    Monkeypatches ``TaskCurator.curate`` at the CLASS level to an async stub
    that always returns a plain ``action='create'`` decision carrying no
    corpus/LLM-derived routing opinion of its own. Patching the class
    (rather than a specific instance) matters because the interceptor's
    curator is lazily constructed inside the ticket worker
    (``TaskInterceptor._get_curator``) AFTER this helper runs — patching the
    class before the submit call means whichever instance ``_get_curator``
    goes on to build still resolves ``.curate`` through the patched class
    attribute, so a REAL ``TaskCurator`` is still consulted (proving the
    boundary — not a merely-absent curator — is what coerces the result).
    This isolates the submit-boundary coercion (``inject_operational_routing``,
    task β) as the SOLE possible source of a deterministic pure-gate
    ``task_kind`` on this path: a no-op 'create' decision carries no routing
    opinion of its own.
    """
    async def _neutral_curate(self, candidate, project_id, project_root):
        return CuratorDecision(action='create', justification='ζ-test-neutralized')

    monkeypatch.setattr(TaskCurator, 'curate', _neutral_curate)

    server, _interceptor = stack
    submit_result = await server._tool_manager.call_tool(
        'submit_task',
        {
            'project_root': root,
            'title': 'Zeta boundary probe: normal submit',
            'description': 'Normal (non-planning_mode) submission for the ζ boundary matrix.',
            'metadata': metadata,
        },
    )
    assert 'error' not in submit_result, f'submit_task failed unexpectedly: {submit_result!r}'
    ticket = submit_result['ticket']

    resolved = await server._tool_manager.call_tool(
        'resolve_ticket',
        {'ticket': ticket, 'project_root': root, 'timeout_seconds': 30},
    )
    assert resolved.get('status') == 'created', f'resolve_ticket did not create: {resolved!r}'
    task_id = resolved['task_id']

    return await _get_task_meta(server, root, task_id)


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


# ---------------------------------------------------------------------------
# Scenario 4a — llm task state (matrix row 4, producer half: the llm-gate
# marker + preserved operational_mode that task γ's DeterministicRunner
# (orchestrator/tests/test_operational_routing_boundary_matrix.py, scenario
# 4b) consumes).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planning_mode_operational_llm_stamps_gate_marker(_real_stack, tmp_path):
    """A planning_mode submit declaring execution_class='operational' +
    operational_mode='llm' is coerced to a deterministic pure gate AND
    stamped with the machine-checkable llm-gate marker.

    Confirms inject_operational_routing's llm sub-branch: the pure-gate
    shape applies (task_kind/always_escalates/before_done, as in scenario
    2) PLUS OPERATIONAL_LLM_GATE_MARKER_KEY is stamped True, PLUS the
    declared operational_mode='llm' survives the pure-gate stamp verbatim
    (both are readable by consumers — task γ's DeterministicRunner reads
    either).
    """
    server, _interceptor = _real_stack
    root = str(tmp_path / 'proj')

    result = await server._tool_manager.call_tool(
        'submit_task',
        {
            'project_root': root,
            'title': 'Zeta boundary probe: llm gate needs a lane',
            'description': 'Operational ask that needs an LLM lane, submitted via planning_mode.',
            'planning_mode': True,
            'metadata': {
                'execution_class': 'operational',
                'operational_mode': 'llm',
            },
        },
    )
    assert 'error' not in result, f'submit_task failed unexpectedly: {result!r}'
    task_id = result['task_id']

    meta = await _get_task_meta(server, root, task_id)

    assert meta['task_kind'] == 'deterministic', f'got {meta!r}'
    assert meta['always_escalates'] is True, f'got {meta!r}'
    assert meta[OPERATIONAL_LLM_GATE_MARKER_KEY] is True, f'got {meta!r}'
    assert meta['operational_mode'] == 'llm', f'got {meta!r}'


# ---------------------------------------------------------------------------
# Scenario 1 — gate via NORMAL submit (matrix row 1/2: operational+gate,
# the two-way twin of scenario 2's planning_mode realization).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normal_submit_operational_gate_coerces_to_pure_gate(
    _real_stack, tmp_path, monkeypatch,
):
    """A NORMAL (non-planning_mode) submit declaring execution_class='operational'
    + operational_mode='gate' is coerced to a deterministic pure gate, even
    with a REAL (curator-neutralized) curator in the loop.

    Uses _submit_normal_and_get (step-4), which monkeypatches TaskCurator.curate
    to a no-op 'create' decision carrying no routing opinion of its own — so a
    deterministic task_kind surviving persist+read-back on THIS path can only
    have come from the submit-boundary coercion (inject_operational_routing,
    task β), never from the curator. Two-way with scenario 2: that scenario
    proves the coercion on the curator-BYPASSING planning_mode path; this one
    proves it on the curator-CONSULTED normal path.
    """
    root = str(tmp_path / 'proj')

    meta = await _submit_normal_and_get(
        _real_stack,
        root,
        {'execution_class': 'operational', 'operational_mode': 'gate'},
        monkeypatch,
    )

    assert meta['task_kind'] == 'deterministic', f'got {meta!r}'
    assert meta['always_escalates'] is True, f'got {meta!r}'
    assert 'before_done' not in meta, f'got {meta!r}'


def _inject_operational_registry(
    monkeypatch: pytest.MonkeyPatch, entry: OperationalAskEntry,
) -> None:
    """Monkeypatch the operational-ask registry loader to return a single
    injected entry, for scenario 5's untagged-legacy substring-fallback test.

    Patches ``fused_memory.middleware.operational_ask_registry.load_operational_registry``
    at the MODULE level (never the shipped config/operational_ask_registry.yaml —
    a production-config edit is out of scope for this test-only task).
    ``TaskCurator._maybe_route_deterministic`` resolves this name via a local
    ``from ... import load_operational_registry`` INSIDE its function body on
    its first call (lazy-load, then cached on ``self._operational_registry``
    for the lifetime of the TaskCurator instance) — so patching the module
    attribute before that first call is picked up regardless of whatever
    (unread — scenario 5 never touches disk) path
    ``config.curator.operational_ask_registry_path`` names. The curator is
    deliberately left ACTIVE here (unlike ``_submit_normal_and_get``'s
    ``TaskCurator.curate`` neutralization) — scenario 5 must reach
    ``_maybe_route_deterministic`` for real.
    """
    def _fake_load(path):  # noqa: ARG001 — path unread; see docstring
        return [entry]

    monkeypatch.setattr(
        'fused_memory.middleware.operational_ask_registry.load_operational_registry',
        _fake_load,
    )
