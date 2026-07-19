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
import os
from pathlib import Path
from unittest.mock import patch

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

    async def test_idempotent_skip_leaves_prior_archive_untouched(self, git_repo):
        """A session already archived (matching mtime) is skipped; a fresh one is
        archived. Proves producer→backstop idempotency (E3) rides the shared
        helper's int-truncated size/mtime skip — the backstop is a no-op for
        transcripts the producer already captured."""
        tid = '2786'
        git_ops = _make_git_ops(git_repo, transcript_archive=TranscriptArchiveConfig())
        wt = await git_ops.create_worktree(tid)

        # Session A: already archived. Lay down the source, then pre-create its
        # gz mirror with a DISTINCTIVE marker payload and a mirrored mtime so the
        # helper's ``int(dest.mtime) == int(src.mtime)`` skip fires — re-archiving
        # would overwrite the marker with a real gzip of the source.
        src_a = _write_transcript(wt.path, tid, 'A', b'{"a":1}\n')
        gz_a = _archived_gz(git_repo, tid, 'A')
        gz_a.parent.mkdir(parents=True, exist_ok=True)
        gz_a.write_bytes(b'STALE-MARKER-NOT-A-REAL-GZIP')
        src_mtime = src_a.stat().st_mtime
        os.utime(gz_a, (src_mtime, src_mtime))

        # Session B: un-archived.
        _write_transcript(wt.path, tid, 'B', b'{"b":2}\n')

        await git_ops.cleanup_worktree(wt.path, tid)

        # A skipped: marker payload AND mtime both unchanged.
        assert gz_a.read_bytes() == b'STALE-MARKER-NOT-A-REAL-GZIP'
        assert int(gz_a.stat().st_mtime) == int(src_mtime)
        # B newly archived to a real gzip of its source.
        gz_b = _archived_gz(git_repo, tid, 'B')
        assert gz_b.exists()
        with gzip.open(gz_b, 'rb') as fh:
            assert fh.read() == b'{"b":2}\n'
        assert not wt.path.exists()

    async def test_arg_wiring_positional_session_none_and_archive_root(self, git_repo):
        """The backstop calls the helper with ``(config_dir, task_id, None)``
        positionally and ``archive_root=project_root/root`` — the exact contract
        the producer uses, so the size/mtime skip fires across producer→backstop.
        session_id=None is what makes it archive ANY un-archived transcript (the
        abandoned-in-flight tail), not just one session's."""
        tid = '2786'
        git_ops = _make_git_ops(git_repo, transcript_archive=TranscriptArchiveConfig())
        wt = await git_ops.create_worktree(tid)
        # Lay down a transcript so the config dir exists (survives step-4's
        # config_dir.exists() fast-skip guard).
        _write_transcript(wt.path, tid, 'A', b'{"a":1}\n')

        with patch('orchestrator.git_ops.archive_task_transcripts') as mock_helper:
            await git_ops.cleanup_worktree(wt.path, tid)

        mock_helper.assert_called_once_with(
            wt.path / '.task' / f'claude-config-{tid}',
            tid,
            None,
            archive_root=git_repo / 'data' / 'orchestrator' / 'agent-transcripts',
        )
        assert not wt.path.exists()

    async def test_missing_config_dir_is_a_thread_free_no_op(self, git_repo):
        """When the per-task config dir is already gone (external worktrees,
        already-cleaned dirs), the backstop is a cheap no-op — it never spins up a
        worker thread to archive nothing. Drives step-4's ``config_dir.exists()``
        fast-skip: without that guard the helper is still invoked on an empty
        glob."""
        tid = '2786'
        git_ops = _make_git_ops(git_repo, transcript_archive=TranscriptArchiveConfig())
        wt = await git_ops.create_worktree(tid)
        # Deliberately lay down NO transcript, so the config dir never exists.
        assert not (wt.path / '.task' / f'claude-config-{tid}').exists()

        with patch('orchestrator.git_ops.archive_task_transcripts') as mock_helper:
            await git_ops.cleanup_worktree(wt.path, tid)

        mock_helper.assert_not_called()
        assert not wt.path.exists()

    async def test_kill_switch_disabled_skips_archival(self, git_repo):
        """``transcript_archive.enabled=False`` short-circuits the backstop
        entirely — no helper call, no archive produced — while the worktree is
        still removed. Mirrors the producer's ``if ta.enabled`` guard."""
        tid = '2786'
        git_ops = _make_git_ops(
            git_repo, transcript_archive=TranscriptArchiveConfig(enabled=False)
        )
        wt = await git_ops.create_worktree(tid)
        # An un-archived transcript exists — but a disabled config must ignore it.
        _write_transcript(wt.path, tid, 'A', b'{"a":1}\n')

        with patch('orchestrator.git_ops.archive_task_transcripts') as mock_helper:
            await git_ops.cleanup_worktree(wt.path, tid)

        mock_helper.assert_not_called()
        # Nothing was written under the archive root at all.
        assert not (git_repo / 'data' / 'orchestrator' / 'agent-transcripts').exists()
        # Teardown still happened — the kill switch gates archival, not removal.
        assert not wt.path.exists()
