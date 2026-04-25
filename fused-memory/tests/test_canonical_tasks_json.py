"""End-to-end tests for WP-G canonical tasks.json redirection.

The MCP server layer normalizes every ``project_root`` it receives to the
main git checkout before handing off to the TaskInterceptor or the
TaskFileCommitter. Worktrees never own their own tasks.json copy.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from fused_memory.middleware.task_file_committer import (
    TASKS_REL_PATH,
    TaskFileCommitter,
)
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.middleware.ticket_store import TicketStore
from fused_memory.models import scope
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.server.tools import create_mcp_server


@pytest.fixture(autouse=True)
def clear_resolver_cache():
    scope._MAIN_CHECKOUT_CACHE.clear()
    yield
    scope._MAIN_CHECKOUT_CACHE.clear()


def _init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'init', '-q'], cwd=path, check=True)
    subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=path, check=True)
    subprocess.run(['git', 'config', 'user.name', 'Test'], cwd=path, check=True)
    subprocess.run(['git', 'config', 'commit.gpgsign', 'false'], cwd=path, check=True)
    (path / 'README.md').write_text('seed\n')
    subprocess.run(['git', 'add', 'README.md'], cwd=path, check=True)
    subprocess.run(
        ['git', 'commit', '-q', '--no-verify', '-m', 'init'], cwd=path, check=True,
    )
    return path.resolve()


def _add_worktree(main: Path, wt_dir: Path, branch: str) -> Path:
    wt_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ['git', 'worktree', 'add', '-q', '-b', branch, str(wt_dir)],
        cwd=main, check=True,
    )
    return wt_dir.resolve()


@pytest.mark.asyncio
async def test_submit_task_normalises_worktree_path_to_main(tmp_path):
    """MCP submit_task called with a worktree path must normalise to the main
    checkout path before forwarding to TaskInterceptor.

    Note: this test verifies path normalisation only (project_root_seen == main).
    End-to-end file-creation coverage (tasks.json appearing in main, not worktree,
    after a full submit_task → curator worker → tm.add_task round-trip) is handled
    by test_submit_task_creates_tasks_json_in_main_not_worktree below.
    ``test_committer_commits_to_main_from_worktree_path`` is NOT that test — it
    seeds tasks.json on disk first and exercises only ``TaskFileCommitter.commit()``.
    """
    main = _init_repo(tmp_path / 'repo')
    wt = _add_worktree(main, tmp_path / 'wt', 'feature')

    async def fake_submit_task(*, project_root: str, **kwargs):
        return {'ticket': 'tkt_1', 'project_root_seen': project_root}

    task_interceptor = AsyncMock()
    task_interceptor.submit_task = AsyncMock(side_effect=fake_submit_task)

    server = create_mcp_server(AsyncMock(), task_interceptor=task_interceptor)
    result = await server._tool_manager.call_tool(
        'submit_task',
        {'project_root': str(wt), 'title': 'from worktree'},
    )

    assert result.get('ticket') == 'tkt_1'
    assert result['project_root_seen'] == str(main)


@pytest.mark.asyncio
async def test_committer_commits_to_main_from_worktree_path(tmp_path):
    """TaskFileCommitter.commit() with a worktree path auto-commits against
    the main checkout — worktree branch stays untouched."""
    main = _init_repo(tmp_path / 'repo')
    wt = _add_worktree(main, tmp_path / 'wt', 'feature')

    tasks_file = Path(main) / TASKS_REL_PATH
    tasks_file.parent.mkdir(parents=True, exist_ok=True)
    tasks_file.write_text(json.dumps({'master': {'tasks': [{'id': 1, 'title': 't1'}]}}))

    main_head_before = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=main, capture_output=True, text=True, check=True,
    ).stdout.strip()
    wt_head_before = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=wt, capture_output=True, text=True, check=True,
    ).stdout.strip()

    committer = TaskFileCommitter()
    await committer.commit(str(wt), 'add_task')

    main_head_after = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=main, capture_output=True, text=True, check=True,
    ).stdout.strip()
    wt_head_after = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=wt, capture_output=True, text=True, check=True,
    ).stdout.strip()

    assert main_head_after != main_head_before, 'main should have the new commit'
    assert wt_head_after == wt_head_before, 'worktree branch must be untouched'

    log = subprocess.run(
        ['git', 'log', '-1', '--pretty=%s'], cwd=main, capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert 'auto-commit after add_task' in log


@pytest.mark.asyncio
async def test_submit_task_creates_tasks_json_in_main_not_worktree(tmp_path):
    """End-to-end: MCP submit_task from a worktree path writes tasks.json to main only.

    Drives the full submit_task → curator worker → tm.add_task chain through the
    real MCP ``_tool_manager.call_tool`` surface. The mocked ``tm.add_task`` writes
    ``.taskmaster/tasks/tasks.json`` under whatever ``project_root`` it receives, so
    file existence under main vs worktree is the assertion lever for whether path
    normalisation survives all the way through the async worker.

    Complements ``test_submit_task_normalises_worktree_path_to_main`` (which only
    checks path normalisation at the MCP wrapper) and supersedes the misleading
    reference to ``test_committer_commits_to_main_from_worktree_path`` (which seeds
    tasks.json on disk first and exercises only ``TaskFileCommitter.commit()``).
    """
    main = _init_repo(tmp_path / 'repo')
    wt = _add_worktree(main, tmp_path / 'wt', 'feature')

    # Mock taskmaster: add_task writes tasks.json under whatever project_root it receives.
    # If normalisation works the file lands in main; if it breaks it lands in wt.
    taskmaster = AsyncMock()
    taskmaster.get_tasks = AsyncMock(return_value={'tasks': []})

    add_task_calls: list[dict] = []

    async def fake_add_task(**kwargs):
        add_task_calls.append(dict(kwargs))
        project_root = kwargs.get('project_root', '')
        tasks_file = Path(project_root) / TASKS_REL_PATH
        tasks_file.parent.mkdir(parents=True, exist_ok=True)
        task = {'id': '1', 'title': kwargs.get('title', 'new')}
        tasks_file.write_text(json.dumps({'master': {'tasks': [task]}}))
        return task

    taskmaster.add_task = fake_add_task

    # Build real TicketStore + EventBuffer + TaskInterceptor (no curator, no committer).
    # Without a curator, _dispatch_ticket_decision falls straight through to the
    # create branch.
    # Note: interceptor.close() owns the ticket_store and closes it (see close() docstring
    # point 6); no separate store.close() call is needed in the finally block.
    store = TicketStore(tmp_path / 'e2e_tickets.db')
    await store.initialize()
    event_buffer = EventBuffer(db_path=tmp_path / 'e2e_eb.db', buffer_size_threshold=100)
    await event_buffer.initialize()
    interceptor = TaskInterceptor(taskmaster, None, event_buffer, ticket_store=store)
    await interceptor.start()  # mirrors production lifecycle: start() before submit_task

    server = create_mcp_server(AsyncMock(), task_interceptor=interceptor)

    try:
        # Phase 1: submit_task with worktree path — MCP layer normalises to main
        submit_result = await server._tool_manager.call_tool(
            'submit_task',
            {'project_root': str(wt), 'title': 'from worktree e2e'},
        )
        assert 'ticket' in submit_result, f'submit_task returned error: {submit_result}'
        ticket = submit_result['ticket']

        # Phase 2: resolve_ticket — waits for the worker to call tm.add_task
        resolve_result = await server._tool_manager.call_tool(
            'resolve_ticket',
            {'ticket': ticket, 'project_root': str(wt), 'timeout_seconds': 10.0},
        )

        assert resolve_result.get('status') == 'created', (
            f'resolve_ticket did not return created: {resolve_result}'
        )
        assert len(add_task_calls) == 1, (
            f'tm.add_task expected exactly 1 call, got {len(add_task_calls)}: {add_task_calls}'
        )
        assert add_task_calls[0].get('project_root') == str(main), (
            f'tm.add_task received project_root={add_task_calls[0].get("project_root")!r}, '
            f'expected normalised main path {str(main)!r}'
        )
        assert (Path(main) / TASKS_REL_PATH).exists(), (
            'tasks.json must exist in main checkout after submit_task from worktree'
        )
        assert not (Path(wt) / TASKS_REL_PATH).exists(), (
            'tasks.json must NOT exist in worktree — path normalisation must redirect to main'
        )
        tasks_data = json.loads((Path(main) / TASKS_REL_PATH).read_text())
        tasks = tasks_data['master']['tasks']
        assert any(t.get('title') == 'from worktree e2e' for t in tasks), (
            'tasks.json in main must contain the submitted task'
        )
    finally:
        await interceptor.close()
        await event_buffer.close()
