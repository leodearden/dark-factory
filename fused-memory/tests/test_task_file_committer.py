"""Tests for TaskFileCommitter — auto-commits tasks.json after task mutations."""

import asyncio
import subprocess

import pytest

from fused_memory.middleware.task_file_committer import TASKS_REL_PATH, TaskFileCommitter


@pytest.fixture
def git_repo(tmp_path):
    """Create a minimal git repo with an initial commit."""
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@test.com"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Test"],
        check=True,
        capture_output=True,
    )
    (tmp_path / ".gitkeep").touch()
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-m", "init"],
        check=True,
        capture_output=True,
    )
    return tmp_path


def _git_log(repo_path, max_count=10):
    """Return list of commit messages."""
    result = subprocess.run(
        ["git", "-C", str(repo_path), "log", f"--max-count={max_count}", "--format=%s"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.strip().splitlines() if line]


def _write_tasks_json(repo_path, content="{}"):
    """Create .taskmaster/tasks/tasks.json with given content."""
    tasks_file = repo_path / TASKS_REL_PATH
    tasks_file.parent.mkdir(parents=True, exist_ok=True)
    tasks_file.write_text(content)


@pytest.mark.asyncio
async def test_commit_adds_and_commits(git_repo):
    """Write tasks.json, commit, verify git log shows the auto-commit."""
    _write_tasks_json(git_repo, '{"tasks": []}')
    committer = TaskFileCommitter()

    await committer.commit(str(git_repo), "add_task")

    messages = _git_log(git_repo)
    assert messages[0] == "chore(tasks): auto-commit after add_task"


@pytest.mark.asyncio
async def test_commit_idempotent_no_changes(git_repo):
    """Commit twice on same content — only 1 auto-commit created."""
    _write_tasks_json(git_repo, '{"tasks": []}')
    committer = TaskFileCommitter()

    await committer.commit(str(git_repo), "add_task")
    await committer.commit(str(git_repo), "update_task(1)")

    messages = _git_log(git_repo)
    # Only the first auto-commit + the initial commit
    assert len(messages) == 2
    assert messages[0] == "chore(tasks): auto-commit after add_task"
    assert messages[1] == "init"


@pytest.mark.asyncio
async def test_commit_skips_when_file_missing(tmp_path):
    """No tasks.json, no error — just a debug log."""
    # tmp_path is not even a git repo, but the file check comes first
    committer = TaskFileCommitter()
    # Should not raise
    await committer.commit(str(tmp_path), "add_task")


@pytest.mark.asyncio
async def test_commit_skips_when_not_git_repo(tmp_path):
    """Non-git directory with tasks.json — logs warning, doesn't raise."""
    _write_tasks_json(tmp_path, '{"tasks": []}')
    committer = TaskFileCommitter()
    # Should not raise even though git add will fail
    await committer.commit(str(tmp_path), "add_task")


@pytest.mark.asyncio
async def test_commits_serialized_same_project(git_repo):
    """Concurrent commits for the same project don't produce git errors."""
    committer = TaskFileCommitter()

    async def write_and_commit(content, op):
        _write_tasks_json(git_repo, content)
        await committer.commit(str(git_repo), op)

    # Launch 5 concurrent commits with different content
    await asyncio.gather(
        write_and_commit('{"v": 1}', "op1"),
        write_and_commit('{"v": 2}', "op2"),
        write_and_commit('{"v": 3}', "op3"),
        write_and_commit('{"v": 4}', "op4"),
        write_and_commit('{"v": 5}', "op5"),
    )

    # Should have at least 1 auto-commit (serialization means some may be no-ops)
    messages = _git_log(git_repo)
    auto_commits = [m for m in messages if m.startswith("chore(tasks):")]
    assert len(auto_commits) >= 1


@pytest.mark.asyncio
async def test_commit_message_includes_operation(git_repo):
    """Verify commit message contains the operation name."""
    _write_tasks_json(git_repo, '{"tasks": []}')
    committer = TaskFileCommitter()

    await committer.commit(str(git_repo), "set_task_status(42=done)")

    messages = _git_log(git_repo)
    assert "set_task_status(42=done)" in messages[0]


@pytest.mark.asyncio
async def test_commit_after_content_change(git_repo):
    """Two commits with different content — both should be recorded."""
    committer = TaskFileCommitter()

    _write_tasks_json(git_repo, '{"v": 1}')
    await committer.commit(str(git_repo), "op1")

    _write_tasks_json(git_repo, '{"v": 2}')
    await committer.commit(str(git_repo), "op2")

    messages = _git_log(git_repo)
    assert messages[0] == "chore(tasks): auto-commit after op2"
    assert messages[1] == "chore(tasks): auto-commit after op1"
