"""Teardown-backstop coverage: GitOps.cleanup_worktree archives any un-archived
agent transcript BEFORE removing the worktree (task 2786,
plans/agent-transcript-archival-prd.md β).

The producer hook (α/2742, test_transcript_archive_producer_hook.py) covers the
normal per-invocation case; this backstop closes the abandoned-in-flight tail —
a role in-flight when the orchestrator died, whose task is reaped without a
completed resume. It is idempotent with the producer (same archive_root +
task_id → the helper's size/mtime skip fires) and best-effort (a broken
archiver can never block ``git worktree remove``).

Fixtures are kept module-local (no conftest.py), mirroring
test_transcript_archive_producer_hook.py's established convention.
"""

from __future__ import annotations

import asyncio
import gzip
from pathlib import Path

import pytest

from orchestrator.config import GitConfig, TranscriptArchiveConfig
from orchestrator.git_ops import GitOps, _run

# The encoded-project dir the fake transcript is laid down under.
ENC = '-home-leo-projX'


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_init_repo(repo))
    return repo


async def _init_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'lib.py').write_text('def greet(name): return name\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


def _make_git_ops(git_repo: Path, **kwargs) -> GitOps:
    """Build a GitOps rooted at *git_repo*; **kwargs pass through to __init__
    (notably ``transcript_archive=...``)."""
    return GitOps(
        GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
        git_repo,
        **kwargs,
    )


def _config_dir(worktree: Path, task_id: str) -> Path:
    """The on-disk per-task Claude config dir the backstop reconstructs."""
    return worktree / '.task' / f'claude-config-{task_id}'


def _write_transcript(worktree: Path, task_id: str, sid: str, data: bytes) -> Path:
    """Lay down an un-archived transcript at
    ``<config_dir>/projects/<ENC>/<sid>.jsonl`` and return its path."""
    p = _config_dir(worktree, task_id) / 'projects' / ENC / f'{sid}.jsonl'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def _archived_gz(git_repo: Path, task_id: str, sid: str) -> Path:
    """The durable gz mirror the backstop should produce for *sid*."""
    return (
        git_repo / 'data' / 'orchestrator' / 'agent-transcripts'
        / task_id / ENC / f'{sid}.jsonl.gz'
    )


@pytest.mark.asyncio
class TestBackstop:

    async def test_end_to_end_transcript_gz_appears(self, git_repo):
        """A worktree with an un-archived transcript is archived, then removed."""
        tid = '2786'
        sid = 'sess-A'
        git_ops = _make_git_ops(git_repo, transcript_archive=TranscriptArchiveConfig())
        wt = await git_ops.create_worktree(tid)
        fake_bytes = b'{"transcript":"hello"}\n'
        _write_transcript(wt.path, tid, sid, fake_bytes)

        await git_ops.cleanup_worktree(wt.path, tid)

        gz = _archived_gz(git_repo, tid, sid)
        assert gz.exists()
        with gzip.open(gz, 'rb') as fh:
            assert fh.read() == fake_bytes
        # Teardown still happened — the backstop runs before removal, not instead.
        assert not wt.path.exists()
