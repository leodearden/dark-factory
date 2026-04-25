"""Integration tests for TaskmasterBackend wire→DTO contract against a live subprocess.

These tests drive a real Taskmaster MCP subprocess (via
``TaskmasterBackend.initialize()``) through two paths:

1. Happy-path round-trip: ``add_task → get_task → set_task_status → remove_task``
2. Tool-level error path: ``get_task('99999')`` → ``TaskmasterError(code='TASKMASTER_TOOL_ERROR')``

The companion canned-payload suite lives in
``tests/test_taskmaster_client_contract.py`` and mocks ``session.call_tool`` with
hand-crafted wire envelopes to cover all 12 wrapper methods.  These live tests
exist exclusively to catch wire-shape drift between Taskmaster's JS tools and the
canned mocks — drift that static mocks cannot detect.

**Intentional scope**: only the basic CRUD path (``add_task``, ``get_task``,
``set_task_status``, ``remove_task``) and one error path are tested here.
Methods like ``update_task``, ``add_subtask``, ``expand_task``,
``add_dependency``, and ``parse_prd`` are fully covered by the canned suite;
adding live calls for each would not meaningfully raise drift-detection probability
(wire shapes are symmetric — if ``add_task`` round-trips correctly the others
overwhelmingly will too).  Expand this suite only when a specific wrapper's wire
shape has been observed to drift in the field.

**Skip semantics**: this suite is skipped automatically unless
``taskmaster-ai/dist/mcp-server.js`` is present.  To enable it::

    git submodule update --init taskmaster-ai
    cd taskmaster-ai && npm install && npm run build
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import pytest_asyncio

from fused_memory.backends.taskmaster_client import TaskmasterBackend
from fused_memory.backends.taskmaster_types import TaskmasterError
from fused_memory.config.schema import TaskmasterConfig

# ── Dist path resolution ─────────────────────────────────────────────
# parents[3]: tests/integration/<file>.py → tests/integration → tests → fused-memory → repo-root
_REPO_ROOT = Path(__file__).resolve().parents[3]
_TASKMASTER_DIST = _REPO_ROOT / 'taskmaster-ai' / 'dist' / 'mcp-server.js'

pytestmark = pytest.mark.skipif(
    not _TASKMASTER_DIST.exists(),
    reason=(
        f'Taskmaster MCP dist not built ({_TASKMASTER_DIST}). '
        'To enable: git submodule update --init taskmaster-ai '
        '&& cd taskmaster-ai && npm install && npm run build'
    ),
)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def taskmaster_backend(tmp_path):
    """Function-scoped fixture: start a live Taskmaster MCP subprocess.

    Creates a temporary project_root pre-seeded with an empty
    ``.taskmaster/tasks/tasks.json`` (``{"master": {"tasks": []}}``),
    initializes the backend, yields ``(backend, project_root_str)``, then
    tears down via ``close()`` in a finally block (close() suppresses
    teardown exceptions so test failures surface their real cause).
    """
    # Pre-seed the tasks file so the first Taskmaster call doesn't hit
    # non-deterministic first-write behaviour.
    tasks_dir = tmp_path / '.taskmaster' / 'tasks'
    tasks_dir.mkdir(parents=True)
    (tasks_dir / 'tasks.json').write_text(json.dumps({'master': {'tasks': []}}))

    config = TaskmasterConfig(
        transport='stdio',
        command='node',
        args=[str(_TASKMASTER_DIST)],
        project_root=str(tmp_path),
        tool_mode='all',
    )
    backend = TaskmasterBackend(config)
    try:
        await backend.initialize()
        yield backend, str(tmp_path)
    finally:
        await backend.close()


# ── Tests ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_add_task_get_task_set_status_remove_task_round_trip(taskmaster_backend):
    """Happy-path: full CRUD round-trip against the live Taskmaster subprocess."""
    backend, project_root = taskmaster_backend

    # (a) add_task returns a non-empty id.
    # Use prompt= (not title=): Taskmaster's JS add_task tool requires either
    # `prompt` alone or both `title` and `description` together.  `prompt=` is
    # the AI-generated happy path and mirrors the canned suite convention at
    # tests/test_taskmaster_client_contract.py:84.  Passing title-only compiles
    # but is rejected server-side ("Either the prompt parameter or both title
    # and description are required").
    add_result = await backend.add_task(
        project_root=project_root,
        prompt='Integration test task',
    )
    task_id = add_result['id']
    assert task_id, f'add_task returned empty id: {add_result!r}'

    # (b) get_task returns the task with matching id/title and a known start-state
    task = await backend.get_task(task_id, project_root=project_root)
    # `add_task` (the wrapper) forces str(taskId) at taskmaster_client.py:380 so
    # task_id is always a str.  `get_task` is a raw passthrough — the JS payload
    # may return the id as int.  Cast LHS so the round-trip assertion doesn't
    # spuriously fail on a type mismatch when the wire path is healthy.
    assert str(task['id']) == task_id
    # Pin the observed wire type: str or int are both valid (JS may return either).
    # A flip in observed type is an intentional contract change — it must also
    # update the canned mocks in tests/test_taskmaster_client_contract.py so that
    # static and live suites stay in sync.
    assert isinstance(task['id'], (str, int)), (
        f"task['id'] has unexpected type {type(task['id']).__name__!r}; "
        'expected str or int — if the wire type changed update the canned mocks '
        'in tests/test_taskmaster_client_contract.py'
    )
    assert task.get('title') == 'Integration test task'
    assert task.get('status') == 'pending', (
        f"Unexpected initial status {task.get('status')!r}; "
        f"expected 'pending' — update this assertion if Taskmaster changes its default"
    )

    # (c) set_task_status returns a message containing 'done'
    status_result = await backend.set_task_status(
        task_id, 'done', project_root=project_root
    )
    assert 'done' in status_result['message'].lower(), (
        f'Expected "done" in set_task_status message, got: {status_result["message"]!r}'
    )

    # (d) remove_task returns successful==1 and the id in removed_ids
    remove_result = await backend.remove_task(task_id, project_root=project_root)
    assert remove_result['successful'] == 1, (
        f'Expected successful=1, got: {remove_result!r}'
    )
    assert task_id in remove_result['removed_ids'], (
        f'Expected {task_id!r} in removed_ids: {remove_result["removed_ids"]!r}'
    )


@pytest.mark.asyncio
async def test_get_task_unknown_id_raises_taskmaster_error(taskmaster_backend):
    """Error path: get_task with an unknown id raises TaskmasterError(TASKMASTER_TOOL_ERROR).

    Taskmaster emits ``createErrorResponse`` for unknown task ids, which the
    ``_unwrap`` adapter surfaces as ``TaskmasterError(code='TASKMASTER_TOOL_ERROR')``.
    """
    backend, project_root = taskmaster_backend

    with pytest.raises(TaskmasterError) as exc_info:
        await backend.get_task('99999', project_root=project_root)

    assert exc_info.value.code == 'TASKMASTER_TOOL_ERROR', (
        f'Expected code=TASKMASTER_TOOL_ERROR, got: {exc_info.value.code!r}'
    )
