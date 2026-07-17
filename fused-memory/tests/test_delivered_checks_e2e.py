"""End-to-end integration gate for the capability-delivered-checks PRD
(``plans/capability-delivered-checks-prd.md``, task zeta).

Proves the whole feature works through the product's own surfaces — no
product code changes here, this module is purely test authoring covering
the PRD's full 10-row Boundary-test sketch table (rows 1, 2, 3, 4, 5, 6, 8,
9 are owned by this file; rows 3/4/7/10 are re-exercised through the real
git runner in ``orchestrator/tests/test_delivered_check_gate_e2e.py``).

This file drives the CROSS-CUTTING headline: ``commit_planning`` (fused
gamma, task 2578) stamping a producer's ``metadata.delivered_checks`` from
a capability-manifest sidecar, THEN a real orchestrator Scheduler tick
(delta/epsilon, tasks 2580/2583) gating dispatch of a dependent task on
those checks passing against a real git ``main`` branch.

Rig shape (see plan.json design_decisions for the full rationale):
- ONE real temp git repo used as ``project_root`` by the fused-memory
  backend, ``commit_planning``, and the orchestrator scheduler's delivered-
  check runner (``git -C project_root grep|rev-parse ... main``).
- A real ``SqliteTaskBackend`` + ``TaskInterceptor`` + ``TicketStore`` +
  ``EventBuffer`` + ``create_mcp_server`` stack (the ``real_task_stack``
  pattern from ``test_task_tools.py``), driven via
  ``server._tool_manager.call_tool(...)`` for submit_task/add_dependency/
  commit_planning/get_task — the product's own MCP surface.
- A real orchestrator ``Harness``/``Scheduler`` with a small delegating
  ``_BackendMcpSession`` injected at ``scheduler._mcp_session`` so
  ``acquire_next()`` reads/writes the SAME live backend state the
  ``commit_planning`` call stamped, and a real ``EscalationQueue`` so the
  born-at-L2 escalation is genuinely filed and read back.

Assertions go exclusively through product read/observe paths: ``get_task``
(metadata + status), ``EscalationQueue.get_by_task(..., level=2,
agent_role='orchestrator-scheduler')`` (the born-at-L2), and
``acquire_next()``'s returned ``TaskAssignment`` / ``None`` (dispatch vs
withhold) — never by peeking at scheduler internals or event-store state
for the primary assertions.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
import yaml

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import TaskmasterConfig
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.middleware.ticket_store import TicketStore
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.server.tools import create_mcp_server

from escalation.queue import EscalationQueue

from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventType
from orchestrator.harness import Harness


# ─────────────────────────────────────────────────────────────────────────────
# Import-resolution smoke test (prerequisite pre-1)
# ─────────────────────────────────────────────────────────────────────────────


def test_cross_package_imports_resolve():
    """Prerequisite pre-1: fused_memory + orchestrator + escalation import
    together inside fused-memory's pytest env (pythonpath includes
    ../orchestrator/src per fused-memory/pyproject.toml)."""
    assert SqliteTaskBackend is not None
    assert TaskmasterConfig is not None
    assert TaskInterceptor is not None
    assert TicketStore is not None
    assert EventBuffer is not None
    assert create_mcp_server is not None
    assert EscalationQueue is not None
    assert OrchestratorConfig is not None
    assert EventType is not None
    assert Harness is not None


# ─────────────────────────────────────────────────────────────────────────────
# Headline fixture constants
# ─────────────────────────────────────────────────────────────────────────────

_PRD_PATH = 'plans/e2e-fixture-prd.md'
_PRODUCER_LABEL = 'producer'
_CAPABILITY_NAME = 'zeta_cap'
_CAPABILITY_TOKEN = 'ZETA_CAPABILITY_TOKEN_V1'
_MARKER_REL_PATH = 'src/marker.py'
_DEPENDENT_REL_PATH = 'src/dependent_target.py'

# Headline part B (step-3): the grace_cycles used by the withhold->escalate
# arc, set explicitly for tick-count determinism (not relying on whatever
# DeliveredChecksConfig's own default happens to be).
_GRACE_CYCLES = 3


# ─────────────────────────────────────────────────────────────────────────────
# Rig helpers
# ─────────────────────────────────────────────────────────────────────────────


def _run_git(project_root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ['git', '-C', str(project_root), *args],
        check=True, capture_output=True, text=True,
    )


def _init_git_repo(root: Path) -> Path:
    """Initialize a real git repo at *root* on branch main with an initial
    commit that already tracks the marker file the headline's grep check
    targets.

    The marker file is committed with PLACEHOLDER content (no capability
    token yet) so the pathspec exists on ``main`` from the start — this
    keeps the grep check's initial "capability absent" state a clean
    no-match (``git grep`` rc=1 -> FAILED) rather than a pathspec-not-found
    error (rc>=2 -> ERRORED), matching PRD row 4 (withhold), not row 7
    (runner error).
    """
    subprocess.run(
        ['git', 'init', '-b', 'main', str(root)],
        check=True, capture_output=True, text=True,
    )
    _run_git(root, 'config', 'user.email', 'e2e-test@example.com')
    _run_git(root, 'config', 'user.name', 'E2E Test')
    marker = root / _MARKER_REL_PATH
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text('# marker file -- capability token lands here later\n', encoding='utf-8')
    _run_git(root, 'add', _MARKER_REL_PATH)
    _run_git(root, 'commit', '-m', 'initial commit')
    return root


@pytest_asyncio.fixture
async def backend_stack(tmp_path):
    """Real fused-memory backend stack (SqliteTaskBackend + TaskInterceptor +
    TicketStore + EventBuffer + create_mcp_server — the real_task_stack
    pattern from test_task_tools.py) rooted at a real temp git repo, so
    submit_task/commit_planning/get_task and the orchestrator scheduler's
    git-backed delivered-check runner all agree on one project_root.
    """
    project_root = _init_git_repo(tmp_path)
    backend = SqliteTaskBackend(TaskmasterConfig(project_root=str(project_root)))
    await backend.start()
    event_buffer = EventBuffer(
        db_path=project_root / 'real_stack_eb.db', buffer_size_threshold=100,
    )
    await event_buffer.initialize()
    ticket_store = TicketStore(project_root / 'real_stack_tickets.db')
    await ticket_store.initialize()
    interceptor = TaskInterceptor(backend, None, event_buffer, ticket_store=ticket_store)
    server = create_mcp_server(AsyncMock(), task_interceptor=interceptor)
    try:
        yield server, interceptor, project_root
    finally:
        await ticket_store.close()
        for _wt in list(interceptor._worker_tasks.values()):
            if not _wt.done():
                _wt.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await _wt
        await event_buffer.close()
        await backend.close()


def _write_sidecar(
    project_root: Path,
    *,
    prd_path: str,
    label: str,
    capability_name: str,
    pattern: str,
    paths: list[str],
) -> Path:
    """Write a capability-manifest sidecar (α schema) with ONE grep-kind
    capability for *label*, derived strictly the same way
    ``stamp_capability_manifests`` derives it: ``re.sub(r'\\.md$', '', prd_path)
    + '.capability-manifest.yaml'``."""
    sidecar_rel = re.sub(r'\.md$', '', prd_path) + '.capability-manifest.yaml'
    sidecar_path = project_root / sidecar_rel
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        'prd': prd_path,
        'schema_version': 1,
        'tasks': [
            {
                'label': label,
                'task_id': None,
                'title': f'Producer {label}',
                'capabilities': [
                    {
                        'name': capability_name,
                        'binding': 'grep for the capability token',
                        'verdict': 'PASS',
                        'delivered_check': {
                            'kind': 'grep',
                            'pattern': pattern,
                            'expect': 'present',
                            'paths': paths,
                        },
                    },
                ],
            },
        ],
    }
    sidecar_path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding='utf-8')
    return sidecar_path


async def _call(server, name: str, **arguments) -> dict:
    """Call an MCP tool via the product's own ToolManager surface (mirrors
    test_task_tools.py's ``server._tool_manager.call_tool(...)`` pattern)."""
    return await server._tool_manager.call_tool(name, arguments)


async def _file_planning_batch(
    server, project_root: Path, *, prd_path: str | None,
) -> tuple[str, str]:
    """File a producer+dependent planning batch (both ``planning_mode=True``).

    The producer carries ``metadata.prd_path``/``metadata.prd_task_label``
    (matching the sidecar written by ``_write_sidecar``) when *prd_path* is
    given; passing ``None`` files a legacy batch with no PRD metadata at all
    (PRD row 2). The dependent depends on the producer via submit_task's
    ``dependencies`` kwarg. Returns ``(producer_id, dependent_id)``.
    """
    producer_metadata: dict = {'files': [_MARKER_REL_PATH]}
    if prd_path is not None:
        producer_metadata['prd_path'] = prd_path
        producer_metadata['prd_task_label'] = _PRODUCER_LABEL

    submit_producer = await _call(
        server, 'submit_task',
        project_root=str(project_root),
        title='Producer task',
        planning_mode=True,
        metadata=producer_metadata,
    )
    assert submit_producer['status'] == 'deferred', f'got {submit_producer!r}'
    producer_id = submit_producer['task_id']

    submit_dependent = await _call(
        server, 'submit_task',
        project_root=str(project_root),
        title='Dependent task',
        planning_mode=True,
        dependencies=producer_id,
        metadata={'files': [_DEPENDENT_REL_PATH]},
    )
    assert submit_dependent['status'] == 'deferred', f'got {submit_dependent!r}'
    dependent_id = submit_dependent['task_id']

    return producer_id, dependent_id


async def _commit_planning(server, project_root: Path, ids: list[str]) -> dict:
    return await _call(
        server, 'commit_planning',
        project_root=str(project_root),
        task_ids=','.join(ids),
    )


async def _get_task(server, project_root: Path, task_id: str) -> dict:
    return await _call(server, 'get_task', id=task_id, project_root=str(project_root))


def _commit_capability(project_root: Path, rel_path: str, token: str) -> str:
    """Land *token* into *rel_path* and commit it to branch main, advancing
    the SHA so the delivered-check gate's stale-cache prune self-heals the
    withheld dependent on the very next tick (PRD row 6: zero further
    operator action beyond a manual re-pend). Returns the new main SHA.
    """
    target = project_root / rel_path
    target.write_text(
        target.read_text(encoding='utf-8') + f'{token}\n', encoding='utf-8',
    )
    _run_git(project_root, 'add', rel_path)
    _run_git(project_root, 'commit', '-m', f'land capability token {token}')
    return _run_git(project_root, 'rev-parse', 'main').stdout.strip()


class _BackendMcpSession:
    """Delegating MCP session double: forwards every ``call_tool`` verbatim to
    the LIVE fused-memory ToolManager (the same ``server._tool_manager
    .call_tool`` surface ``_call`` above uses), wrapping the raw dict result
    in the JSON-RPC text-content envelope the scheduler's
    ``dispatch_tool``/``parse_tool_result`` expects (mirrors
    ``TwoProjectMcpSession._envelope`` in
    ``orchestrator/tests/test_cross_project_dispatch_integration.py``).

    Unlike that reference double's in-memory task list, this forwards to the
    SAME live backend ``commit_planning`` just stamped -- a true closed loop:
    the scheduler's ``get_tasks``/``get_task``/``get_statuses``/
    ``set_task_status`` calls all read/write the real SqliteTaskBackend via
    the real TaskInterceptor, so flip-done and manual re-pend go through the
    product's own write path exactly like a human operator would.
    """

    def __init__(self, server) -> None:
        self._server = server
        self._request_id = 0

    async def call_tool(self, name: str, arguments: dict, timeout: float = 30) -> dict:
        raw = await self._server._tool_manager.call_tool(name, arguments)
        self._request_id += 1
        return {
            'jsonrpc': '2.0',
            'id': self._request_id,
            'result': {'content': [{'type': 'text', 'text': json.dumps(raw)}]},
        }


def _build_harness(
    project_root: Path, server, escalation_dir: Path, *, grace_cycles: int,
) -> Harness:
    """Build a real orchestrator Harness/Scheduler delegating onto the SAME
    live backend the fused-memory server call site drives, plus a real
    EscalationQueue -- so the born-at-L2 grace-streak escalation
    (``Harness._block_and_escalate_delivered_check``) is genuinely filed and
    read back. Mirrors ``build_harness()`` in
    test_cross_project_dispatch_integration.py.
    """
    config = OrchestratorConfig(project_root=project_root)
    config.delivered_checks.grace_cycles = grace_cycles
    harness = Harness(config)
    # Inject the delegating session into the scheduler dispatch seam.
    harness.scheduler._mcp_session = _BackendMcpSession(server)
    # Wire a real EscalationQueue so escalations can be filed and read back.
    harness._escalation_queue = EscalationQueue(escalation_dir)
    # run_tick() drives harness.scheduler.acquire_next() directly, bypassing
    # harness.run()'s startup sweeps -- satisfy the startup gate (task 2235)
    # the same way those sweeps would.
    harness.scheduler.finish_startup()
    return harness


async def _run_tick(harness: Harness) -> str | None:
    """Run one acquire_next() tick; return the dispatched task_id or None.

    Dispatch is OBSERVED through the returned TaskAssignment -- never by
    reading task storage. This matches the task's "dispatch path" requirement.
    """
    assignment = await harness.scheduler.acquire_next()
    return assignment.task_id if assignment is not None else None


async def _flip_status(server, project_root: Path, task_id: str, status: str) -> dict:
    """Flip a task's status via the product's own set_task_status MCP tool
    (the same write path a human operator's manual re-pend would use)."""
    return await _call(
        server, 'set_task_status',
        id=task_id, status=status, project_root=str(project_root),
    )


def _l2_for(harness: Harness, task_id: str) -> list:
    """Return pending born-at-L2 delivered-check escalations for *task_id*.

    Scoped exactly like ``Harness._block_and_escalate_delivered_check``'s own
    dedupe read (``level=2``, ``agent_role='orchestrator-scheduler'``) so an
    unrelated open L2 for the same task never masks (or is masked by) this
    check.
    """
    queue = harness._escalation_queue
    if queue is None:
        return []
    return queue.get_by_task(
        task_id, status='pending', level=2, agent_role='orchestrator-scheduler',
    )


# ─────────────────────────────────────────────────────────────────────────────
# TestHeadline — rows 1, 4, 5, 6, 3: stamp -> withhold -> escalate -> heal -> dispatch
# ─────────────────────────────────────────────────────────────────────────────


class TestHeadline:
    """The closed-loop headline scenario, built up incrementally across
    steps 1/3/5 (each step appends more of the scenario to this same test
    method — see plan.json step descriptions)."""

    @pytest.mark.asyncio
    async def test_headline_stamp_withhold_escalate_heal_dispatch(self, backend_stack):
        """Headline part A (step-1): row 1 — commit_planning stamps the
        producer's real task_id into the sidecar and copies the sidecar's
        mechanical delivered_check into the producer's
        metadata.delivered_checks, visible via get_task."""
        server, _interceptor, project_root = backend_stack

        _write_sidecar(
            project_root,
            prd_path=_PRD_PATH,
            label=_PRODUCER_LABEL,
            capability_name=_CAPABILITY_NAME,
            pattern=_CAPABILITY_TOKEN,
            paths=[_MARKER_REL_PATH],
        )

        producer_id, dependent_id = await _file_planning_batch(
            server, project_root, prd_path=_PRD_PATH,
        )

        # --- row 1: commit_planning stamps the sidecar + copies delivered_checks ---
        result = await _commit_planning(server, project_root, [producer_id, dependent_id])

        expected_sidecar_rel = re.sub(r'\.md$', '', _PRD_PATH) + '.capability-manifest.yaml'
        assert result['manifest_stamping'] == {
            'path': expected_sidecar_rel,
            'stamped': [_PRODUCER_LABEL],
            'missing_labels': [],
            'errors': [],
        }

        sidecar_path = project_root / expected_sidecar_rel
        reloaded = yaml.safe_load(sidecar_path.read_text(encoding='utf-8'))
        assert reloaded['tasks'][0]['task_id'] == int(producer_id)

        producer_task = await _get_task(server, project_root, producer_id)
        checks = producer_task['metadata']['delivered_checks']
        assert len(checks) == 1
        assert checks[0]['name'] == _CAPABILITY_NAME
        assert checks[0]['kind'] == 'grep'
        assert checks[0]['pattern'] == _CAPABILITY_TOKEN
        assert checks[0]['expect'] == 'present'
        assert checks[0]['paths'] == [_MARKER_REL_PATH]

        # Status flip landed too (target_status defaults to 'pending').
        assert producer_task['status'] == 'pending'
        dependent_task = await _get_task(server, project_root, dependent_id)
        assert dependent_task['status'] == 'pending'

        # --- Headline part B (step-3): rows 4+5 -----------------------------
        # Scope-cut: flip the producer straight to 'done' via the product
        # write path while the capability token is still absent from main.
        # A real orchestrator Harness/Scheduler (delegating _mcp_session onto
        # this SAME live backend + a real EscalationQueue) then ticks:
        # ticks 1..G-1 must withhold dispatch with no L2 filed yet; tick G
        # must file exactly one born-at-L2 escalation naming the failed
        # check, the producer's dep id, and the main SHA, and the dependent
        # must be 'blocked'.
        harness = _build_harness(
            project_root, server, project_root / 'escalations',
            grace_cycles=_GRACE_CYCLES,
        )

        await _flip_status(server, project_root, producer_id, 'done')

        for tick in range(1, _GRACE_CYCLES):
            result = await _run_tick(harness)
            assert result is None, (
                f'tick {tick}: dep-done-with-failing-check must withhold '
                f'dispatch of the dependent; got {result!r}'
            )
            assert _l2_for(harness, dependent_id) == [], (
                f'tick {tick}: no L2 escalation must exist before grace_cycles '
                f'({_GRACE_CYCLES}) is reached'
            )

        # Tick G: born-at-L2 escalation fires naming the exact failed check;
        # the dependent is not dispatched and is now blocked.
        result = await _run_tick(harness)
        assert result is None, (
            f'tick {_GRACE_CYCLES}: dependent must still not be dispatched; '
            f'got {result!r}'
        )

        escs = _l2_for(harness, dependent_id)
        assert len(escs) == 1, (
            f'expected exactly one pending born-at-L2 escalation for the '
            f'dependent on tick {_GRACE_CYCLES}; got {escs!r}'
        )
        esc = escs[0]
        assert 'DEP_CAPABILITY_NOT_DELIVERED' in esc.summary, esc.summary
        assert _CAPABILITY_NAME in esc.summary, esc.summary
        assert producer_id in esc.summary, esc.summary
        main_sha = _run_git(project_root, 'rev-parse', 'main').stdout.strip()
        assert main_sha[:12] in esc.summary, (
            f'expected main sha prefix {main_sha[:12]!r} in summary; '
            f'got {esc.summary!r}'
        )

        dependent_after_block = await _get_task(server, project_root, dependent_id)
        assert dependent_after_block['status'] == 'blocked'

        # --- Headline part C (step-5): rows 6+3 -----------------------------
        # Self-heal: commit a file containing the capability token to main
        # (advancing the SHA -- the stale-cache prune picks up the fix), then
        # manually re-pend the dependent via the product write path. The very
        # next tick must dispatch the dependent with ZERO further operator
        # action beyond the re-pend, and no NEW L2 escalation may be filed
        # (the one from part B stays as the sole pending record).
        _commit_capability(project_root, _MARKER_REL_PATH, _CAPABILITY_TOKEN)

        await _flip_status(server, project_root, dependent_id, 'pending')

        result = await _run_tick(harness)
        assert result == dependent_id, (
            f'expected the dependent ({dependent_id!r}) to dispatch once the '
            f'capability landed on main and it was re-pended; got {result!r}'
        )

        post_heal_escs = _l2_for(harness, dependent_id)
        assert len(post_heal_escs) == 1 and post_heal_escs[0].id == esc.id, (
            f'no NEW L2 escalation may be filed on the self-heal/dispatch tick; '
            f'expected only the part-B escalation {esc.id!r} to remain, got '
            f'{post_heal_escs!r}'
        )


# ─────────────────────────────────────────────────────────────────────────────
# TestLegacyNoSidecar — row 2: no PRD/sidecar involvement -> byte-identical
# ─────────────────────────────────────────────────────────────────────────────


def _assert_row2_legacy_response(result: dict, producer_task: dict) -> None:
    """Row-2 assertion helper: a legacy batch (no PRD/sidecar involvement)
    must produce a byte-identical ``commit_planning`` response -- no
    ``manifest_stamping`` key -- and the producer must not have gained
    ``metadata.delivered_checks``."""
    assert 'manifest_stamping' not in result, (
        f"a batch with no PRD metadata must produce a byte-identical "
        f"legacy response (no 'manifest_stamping' key); got {result!r}"
    )
    assert 'delivered_checks' not in producer_task['metadata'], (
        f"a legacy batch must not stamp metadata.delivered_checks; got "
        f"{producer_task['metadata']!r}"
    )


class TestLegacyNoSidecar:
    """Row 2: a legacy planning batch with no PRD metadata (and therefore no
    sidecar on disk at all) must produce a byte-identical legacy
    ``commit_planning`` response — no ``manifest_stamping`` key — and must
    not stamp ``metadata.delivered_checks`` onto the producer."""

    @pytest.mark.asyncio
    async def test_legacy_batch_no_sidecar_byte_identical(self, backend_stack):
        server, _interceptor, project_root = backend_stack

        # No _write_sidecar call at all, and prd_path=None files a batch with
        # no metadata.prd_path/prd_task_label on either task (PRD row 2).
        producer_id, dependent_id = await _file_planning_batch(
            server, project_root, prd_path=None,
        )

        result = await _commit_planning(server, project_root, [producer_id, dependent_id])
        producer_task = await _get_task(server, project_root, producer_id)

        _assert_row2_legacy_response(result, producer_task)


# ─────────────────────────────────────────────────────────────────────────────
# TestManualOnlySidecar — row 8: manual-only capability -> stamped id, no gate
# ─────────────────────────────────────────────────────────────────────────────

_MANUAL_PRD_PATH = 'plans/e2e-manual-fixture-prd.md'


def _write_manual_sidecar(project_root: Path, *, prd_path: str, label: str) -> Path:
    """Write a capability-manifest sidecar (α schema) whose sole capability
    for *label* is a manual-kind check (reusing the ``_MANUAL`` shape from
    ``test_manifest_stamping.py``): ``commit_planning`` stamps the label's
    real task_id into the sidecar like any other label match, but a
    manual-kind check is excluded from the mechanical
    metadata.delivered_checks copy — so the scheduler's delivered-check gate
    never engages for this producer at all (PRD row 8)."""
    sidecar_rel = re.sub(r'\.md$', '', prd_path) + '.capability-manifest.yaml'
    sidecar_path = project_root / sidecar_rel
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        'prd': prd_path,
        'schema_version': 1,
        'tasks': [
            {
                'label': label,
                'task_id': None,
                'title': f'Producer {label}',
                'capabilities': [
                    {
                        'name': 'manual_only_cap',
                        'binding': 'eyeball the UI',
                        'verdict': 'PASS',
                        'delivered_check': {
                            'kind': 'manual',
                            'reason': 'no automated check available',
                        },
                    },
                ],
            },
        ],
    }
    sidecar_path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding='utf-8')
    return sidecar_path


class TestManualOnlySidecar:
    """Row 8: a sidecar label whose ONLY capability is manual-kind still gets
    its task_id stamped (it's a normal label match), but is excluded from the
    mechanical metadata.delivered_checks copy — so once the producer is
    done, the dependent must dispatch immediately on a pure status-only
    gate, byte-identical to pre-manifest-stamping dispatch behaviour."""

    @pytest.mark.asyncio
    async def test_manual_only_stamps_id_with_no_gate_effect(self, backend_stack):
        server, _interceptor, project_root = backend_stack

        _write_manual_sidecar(
            project_root, prd_path=_MANUAL_PRD_PATH, label=_PRODUCER_LABEL,
        )

        producer_id, dependent_id = await _file_planning_batch(
            server, project_root, prd_path=_MANUAL_PRD_PATH,
        )

        result = await _commit_planning(server, project_root, [producer_id, dependent_id])
        expected_sidecar_rel = re.sub(r'\.md$', '', _MANUAL_PRD_PATH) + '.capability-manifest.yaml'
        assert result['manifest_stamping'] == {
            'path': expected_sidecar_rel,
            'stamped': [_PRODUCER_LABEL],
            'missing_labels': [],
            'errors': [],
        }

        producer_task = await _get_task(server, project_root, producer_id)
        assert 'delivered_checks' not in producer_task['metadata'], (
            f'a manual-only capability must not stamp metadata.delivered_checks; '
            f'got {producer_task["metadata"]!r}'
        )

        harness = _build_harness(
            project_root, server, project_root / 'escalations',
            grace_cycles=_GRACE_CYCLES,
        )
        await _flip_status(server, project_root, producer_id, 'done')

        result = await _run_tick(harness)
        assert result == dependent_id, (
            f'a manual-only producer capability must not gate dispatch — '
            f'expected the dependent ({dependent_id!r}) to dispatch on the '
            f'very first tick; got {result!r}'
        )


# ─────────────────────────────────────────────────────────────────────────────
# TestMalformedSidecar — row 9: alpha-invalid sidecar -> fail-soft loud, flip proceeds
# ─────────────────────────────────────────────────────────────────────────────

_MALFORMED_PRD_PATH = 'plans/e2e-malformed-fixture-prd.md'


def _write_malformed_sidecar(project_root: Path, *, prd_path: str, label: str) -> Path:
    """Write a capability-manifest sidecar (α schema) whose sole capability
    for *label* FAILS α validation — a grep-kind check missing the required
    ``expect`` field (reusing the ``_MALFORMED_SIDECAR_YAML`` shape from
    ``test_manifest_stamping.py``): ``commit_planning`` must fail-soft on
    this (report an error, stamp nothing, copy no metadata) but never block
    the batch's own status flip (PRD row 9)."""
    sidecar_rel = re.sub(r'\.md$', '', prd_path) + '.capability-manifest.yaml'
    sidecar_path = project_root / sidecar_rel
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        'prd': prd_path,
        'schema_version': 1,
        'tasks': [
            {
                'label': label,
                'task_id': None,
                'title': f'Producer {label}',
                'capabilities': [
                    {
                        'name': 'broken_check',
                        'binding': 'grep for something',
                        'verdict': 'PASS',
                        'delivered_check': {
                            'kind': 'grep',
                            'pattern': 'something',
                            # intentionally missing 'expect' -> alpha validation failure
                        },
                    },
                ],
            },
        ],
    }
    sidecar_path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding='utf-8')
    return sidecar_path


class TestMalformedSidecar:
    """Row 9: a sidecar present on disk but failing α validation (a grep
    capability missing the required ``expect`` field) must fail loud-but-soft
    — commit_planning still flips the batch to pending (never stranded on a
    docs artifact) and reports the validation error naming the offending
    field, but stamps nothing to disk and copies no metadata."""

    @pytest.mark.asyncio
    async def test_malformed_sidecar_fails_soft_and_flip_proceeds(self, backend_stack):
        server, _interceptor, project_root = backend_stack

        sidecar_path = _write_malformed_sidecar(
            project_root, prd_path=_MALFORMED_PRD_PATH, label=_PRODUCER_LABEL,
        )
        before = sidecar_path.read_text(encoding='utf-8')

        producer_id, dependent_id = await _file_planning_batch(
            server, project_root, prd_path=_MALFORMED_PRD_PATH,
        )

        result = await _commit_planning(server, project_root, [producer_id, dependent_id])

        manifest_stamping = result['manifest_stamping']
        assert manifest_stamping['stamped'] == [], manifest_stamping
        errors = manifest_stamping['errors']
        assert len(errors) == 1, errors
        assert 'expect' in errors[0], errors

        after = sidecar_path.read_text(encoding='utf-8')
        assert after == before, (
            'a malformed sidecar must be byte-identical on disk — no partial '
            'task_id stamp'
        )

        producer_task = await _get_task(server, project_root, producer_id)
        assert producer_task['status'] == 'pending', (
            f"the batch flip must proceed even though the sidecar is "
            f"malformed; got status {producer_task['status']!r}"
        )
        assert 'delivered_checks' not in producer_task['metadata'], (
            f"a malformed capability must not stamp metadata.delivered_checks; "
            f"got {producer_task['metadata']!r}"
        )
