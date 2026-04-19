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
from fused_memory.models import scope
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
async def test_add_task_from_worktree_path_writes_to_main(tmp_path):
    """MCP add_task called with a worktree path must forward the main path
    to the TaskInterceptor, so the canonical tasks.json lives only in main."""
    main = _init_repo(tmp_path / 'repo')
    wt = _add_worktree(main, tmp_path / 'wt', 'feature')

    async def fake_add_task(*, project_root: str, **kwargs):
        tasks_file = Path(project_root) / TASKS_REL_PATH
        tasks_file.parent.mkdir(parents=True, exist_ok=True)
        existing = (
            json.loads(tasks_file.read_text()) if tasks_file.exists() else {'master': {'tasks': []}}
        )
        existing['master']['tasks'].append({'id': 1, 'title': kwargs.get('title', 'new')})
        tasks_file.write_text(json.dumps(existing))
        return {'success': True, 'project_root_seen': project_root}

    task_interceptor = AsyncMock()
    task_interceptor.add_task = AsyncMock(side_effect=fake_add_task)

    server = create_mcp_server(AsyncMock(), task_interceptor=task_interceptor)
    result = await server._tool_manager.call_tool(
        'add_task',
        {'project_root': str(wt), 'title': 'from worktree'},
    )

    assert result.get('success') is True
    assert result['project_root_seen'] == str(main)

    main_tasks = Path(main) / TASKS_REL_PATH
    wt_tasks = Path(wt) / TASKS_REL_PATH
    assert main_tasks.exists(), 'main checkout must own tasks.json'
    assert not wt_tasks.exists(), 'worktree must not have its own tasks.json'

    data = json.loads(main_tasks.read_text())
    titles = [t['title'] for t in data['master']['tasks']]
    assert 'from worktree' in titles


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
