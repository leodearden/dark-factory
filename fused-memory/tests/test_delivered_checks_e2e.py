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
