"""Tests for orchestrator.rebase_drops: commits a rebase silently drops (task 4669).

``git rebase`` drops a branch commit whose change the target already carries,
and exits 0.  rebase_drops records each such commit under the branch it was
dropped from (RebaseDropLedger), so a later reader can tell "this branch never
wrote the change" from "this branch wrote it and a rebase dropped it".

Every test runs against a REAL temp git repository; the git runner is
orchestrator.git_ops._run, injected exactly as production injects it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from orchestrator.git_ops import _run
from orchestrator.rebase_drops import (
    REBASE_DROP_REF_NAMESPACE,
    DroppedCommit,
    RebaseDropLedger,
)

# ---------------------------------------------------------------------------
# Helpers (copied locally from test_branch_stack.py, per-file convention)
# ---------------------------------------------------------------------------


async def _init_repo(path: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=path)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=path)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=path)


async def _git(cwd: Path, *args: str) -> str:
    rc, out, err = await _run(['git', *args], cwd=cwd)
    assert rc == 0, f'git {args} failed: {err}'
    return out


async def _commit(cwd: Path, subject: str, files: dict[str, str]) -> str:
    """Write and commit *files* only, with *subject*; return the new sha."""
    for name, content in files.items():
        path = cwd / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        await _git(cwd, 'add', '--', name)
    await _git(cwd, 'commit', '-m', subject)
    return (await _git(cwd, 'rev-parse', 'HEAD')).strip()


async def _dropped_refs(repo: Path) -> list[str]:
    out = await _git(
        repo, 'for-each-ref', '--format=%(refname)', REBASE_DROP_REF_NAMESPACE,
    )
    return out.split()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    path = tmp_path / 'repo'
    path.mkdir()
    return path


# ---------------------------------------------------------------------------
# RebaseDropLedger
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRebaseDropLedger:
    async def test_record_then_dropped_of_round_trips_in_sha_order(
        self, repo: Path,
    ) -> None:
        await _init_repo(repo)
        c1 = await _commit(repo, 'feat: write x', {'x.txt': 'x\n'})
        c2 = await _commit(
            repo, 'feat: write web', {'web/b.js': 'b\n', 'web/a.js': 'a\n'},
        )
        ledger = RebaseDropLedger(repo, _run)

        await ledger.record('task/1', [c1, c2])

        expected = {
            c1: DroppedCommit(sha=c1, subject='feat: write x', paths=('x.txt',)),
            c2: DroppedCommit(
                sha=c2, subject='feat: write web', paths=('web/a.js', 'web/b.js'),
            ),
        }
        assert await ledger.dropped_of('task/1') == tuple(
            expected[sha] for sha in sorted(expected)
        )
        assert sorted(await _dropped_refs(repo)) == sorted(
            ledger.ref_for('task/1', sha) for sha in (c1, c2)
        )
        assert ledger.ref_for('task/1', c1) == (
            f'refs/dark-factory/rebase-dropped/task/1/{c1}'
        )

    async def test_recording_the_same_sha_twice_yields_one_entry(
        self, repo: Path,
    ) -> None:
        await _init_repo(repo)
        c1 = await _commit(repo, 'feat: write x', {'x.txt': 'x\n'})
        ledger = RebaseDropLedger(repo, _run)

        await ledger.record('task/1', [c1])
        await ledger.record('task/1', [c1, c1])

        assert await ledger.dropped_of('task/1') == (
            DroppedCommit(sha=c1, subject='feat: write x', paths=('x.txt',)),
        )

    async def test_dropped_of_an_unrecorded_branch_is_empty(self, repo: Path) -> None:
        await _init_repo(repo)
        await _commit(repo, 'feat: write x', {'x.txt': 'x\n'})

        assert await RebaseDropLedger(repo, _run).dropped_of('task/1') == ()

    async def test_branch_names_sharing_a_prefix_do_not_collide(
        self, repo: Path,
    ) -> None:
        await _init_repo(repo)
        c1 = await _commit(repo, 'feat: one', {'one.txt': '1\n'})
        c10 = await _commit(repo, 'feat: ten', {'ten.txt': '10\n'})
        ledger = RebaseDropLedger(repo, _run)

        await ledger.record('task/1', [c1])
        await ledger.record('task/10', [c10])

        assert [c.sha for c in await ledger.dropped_of('task/1')] == [c1]
        assert [c.sha for c in await ledger.dropped_of('task/10')] == [c10]

    async def test_a_record_keeps_its_commit_alive_through_gc(
        self, repo: Path,
    ) -> None:
        await _init_repo(repo)
        await _commit(repo, 'base', {'base.txt': 'base\n'})
        dropped = await _commit(repo, 'chore: bump', {'index.html': 'v=44\n'})
        ledger = RebaseDropLedger(repo, _run)
        await ledger.record('task/1', [dropped])

        await _git(repo, 'reset', '--hard', 'HEAD~1')
        await _git(repo, 'reflog', 'expire', '--expire=now', '--all')
        await _git(repo, 'gc', '--prune=now', '--quiet')

        await _git(repo, 'cat-file', '-e', dropped)
        assert await ledger.dropped_of('task/1') == (
            DroppedCommit(sha=dropped, subject='chore: bump', paths=('index.html',)),
        )

    async def test_prune_orphans_drops_only_records_of_deleted_branches(
        self, repo: Path,
    ) -> None:
        await _init_repo(repo)
        await _commit(repo, 'base', {'base.txt': 'base\n'})
        await _git(repo, 'branch', 'task/1')
        await _git(repo, 'branch', 'task/2')
        c1 = await _commit(repo, 'feat: one', {'one.txt': '1\n'})
        c2a = await _commit(repo, 'feat: two a', {'two-a.txt': '2a\n'})
        c2b = await _commit(repo, 'feat: two b', {'two-b.txt': '2b\n'})
        ledger = RebaseDropLedger(repo, _run)
        await ledger.record('task/1', [c1])
        await ledger.record('task/2', [c2a, c2b])
        await _git(repo, 'branch', '-D', 'task/2')

        assert await ledger.prune_orphans() == ('task/2',)

        assert await ledger.dropped_of('task/2') == ()
        assert [c.sha for c in await ledger.dropped_of('task/1')] == [c1]
        assert await ledger.prune_orphans() == ()

    async def test_record_never_raises_outside_a_repository(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        not_a_repo = tmp_path / 'not-a-repo'
        not_a_repo.mkdir()
        ledger = RebaseDropLedger(not_a_repo, _run)

        with caplog.at_level(logging.WARNING, logger='orchestrator.rebase_drops'):
            assert await ledger.record('task/1', ['0' * 40]) is None

        warnings = [
            r for r in caplog.records
            if r.name == 'orchestrator.rebase_drops' and r.levelno == logging.WARNING
        ]
        assert warnings, caplog.text
        assert 'task/1' in warnings[0].getMessage()
