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
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import pytest
import pytest_asyncio

from orchestrator.git_ops import _run
from orchestrator.rebase_drops import (
    REBASE_DROP_REF_NAMESPACE,
    DroppedCommit,
    RebaseDropLedger,
    run_rebase_reporting_drops,
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


# ---------------------------------------------------------------------------
# run_rebase_reporting_drops
# ---------------------------------------------------------------------------

_OWN_A = 'feat(1): own a (task 1)'
_BUMP = 'chore(1): bump 43 -> 44 (task 1)'
_OWN_B = 'test(1): own b (task 1)'


async def _add_worktree(repo: Path, name: str, start_ref: str) -> Path:
    """Create a linked worktree on new branch task/<name> at *start_ref*."""
    wt = repo.parent / f'wt-{name}'
    await _git(repo, 'worktree', 'add', '-b', f'task/{name}', str(wt), start_ref)
    return wt


def _module_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage() for r in caplog.records
        if r.name == 'orchestrator.rebase_drops' and r.levelno == logging.WARNING
    ]


async def _rebase_in_progress(wt: Path) -> bool:
    rebase_dir = (await _git(wt, 'rev-parse', '--git-path', 'rebase-merge')).strip()
    return (wt / rebase_dir).exists()


@dataclass(frozen=True)
class Shape3490:
    """task/1 carries an own commit, a version bump, and another own commit."""

    repo: Path
    wt: Path
    own_a: str
    bump: str
    own_b: str

    @property
    def bump_dropped(self) -> DroppedCommit:
        return DroppedCommit(sha=self.bump, subject=_BUMP, paths=('index.html',))


@pytest_asyncio.fixture
async def shape(repo: Path) -> AsyncIterator[Shape3490]:
    await _init_repo(repo)
    await _commit(repo, 'base', {'index.html': 'v=43\n', 'a.txt': 'a\n'})
    wt = await _add_worktree(repo, '1', 'main')
    own_a = await _commit(wt, _OWN_A, {'a.txt': 'a\nown\n'})
    bump = await _commit(wt, _BUMP, {'index.html': 'v=44\n'})
    own_b = await _commit(wt, _OWN_B, {'b.txt': 'b\n'})
    yield Shape3490(repo=repo, wt=wt, own_a=own_a, bump=bump, own_b=own_b)
    if await _rebase_in_progress(wt):
        await _git(wt, 'rebase', '--abort')


async def _branch_subjects(s: Shape3490) -> list[str]:
    out = await _git(s.repo, 'log', '--reverse', '--format=%s', 'main..task/1')
    return out.splitlines()


@pytest.mark.asyncio
class TestRunRebaseReportingDrops:
    async def test_a_commit_that_became_empty_is_reported_and_recorded(
        self, shape: Shape3490, caplog: pytest.LogCaptureFixture,
    ) -> None:
        await _commit(shape.repo, 'main: bump and more', {
            'index.html': 'v=44\n', 'm.txt': 'm\n',
        })

        with caplog.at_level(logging.WARNING, logger='orchestrator.rebase_drops'):
            result = await run_rebase_reporting_drops(_run, shape.wt, onto='main')

        assert result.ok
        assert result.dropped == (shape.bump_dropped,)
        assert await RebaseDropLedger(shape.repo, _run).dropped_of('task/1') == (
            shape.bump_dropped,
        )
        warnings = _module_warnings(caplog)
        assert len(warnings) == 1, warnings
        assert 'task/1' in warnings[0]
        assert shape.bump_dropped.label in warnings[0]
        assert await _branch_subjects(shape) == [_OWN_A, _OWN_B]

    async def test_a_patch_identical_commit_is_reported_and_recorded(
        self, shape: Shape3490,
    ) -> None:
        await _commit(shape.repo, 'main: the same bump', {'index.html': 'v=44\n'})

        result = await run_rebase_reporting_drops(_run, shape.wt, onto='main')

        assert result.ok
        assert result.dropped == (shape.bump_dropped,)
        assert await RebaseDropLedger(shape.repo, _run).dropped_of('task/1') == (
            shape.bump_dropped,
        )
        assert await _branch_subjects(shape) == [_OWN_A, _OWN_B]

    async def test_a_clean_rebase_reports_and_records_nothing(
        self, shape: Shape3490, caplog: pytest.LogCaptureFixture,
    ) -> None:
        await _commit(shape.repo, 'main: unrelated', {'m.txt': 'm\n'})

        with caplog.at_level(logging.WARNING, logger='orchestrator.rebase_drops'):
            result = await run_rebase_reporting_drops(_run, shape.wt, onto='main')

        assert result.ok
        assert result.dropped == ()
        assert await _dropped_refs(shape.repo) == []
        assert _module_warnings(caplog) == []
        assert await _branch_subjects(shape) == [_OWN_A, _BUMP, _OWN_B]

    async def test_a_linearized_merge_commit_is_never_reported(
        self, repo: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        await _init_repo(repo)
        await _commit(repo, 'base', {'base.txt': 'base\n'})
        wt = await _add_worktree(repo, '1', 'main')
        await _git(wt, 'switch', '-c', 'side')
        await _commit(wt, 'side: own s', {'s.txt': 's\n'})
        await _git(wt, 'switch', 'task/1')
        await _commit(wt, _OWN_A, {'a.txt': 'a\n'})
        await _git(wt, 'merge', '--no-ff', 'side', '-m', 'merge side')
        await _commit(repo, 'main: unrelated', {'m.txt': 'm\n'})

        with caplog.at_level(logging.WARNING, logger='orchestrator.rebase_drops'):
            result = await run_rebase_reporting_drops(_run, wt, onto='main')

        assert result.ok
        assert result.dropped == ()
        assert await _dropped_refs(repo) == []
        assert _module_warnings(caplog) == []

    async def test_upstream_form_reports_only_the_replayed_range(
        self, repo: Path,
    ) -> None:
        await _init_repo(repo)
        await _commit(repo, 'base', {'index.html': 'v=43\n', 'a.txt': 'a\n'})
        p_wt = await _add_worktree(repo, 'P', 'main')
        p1 = await _commit(p_wt, 'feat(P): p1', {'p.txt': 'p\n'})
        wt = await _add_worktree(repo, '1', p1)
        await _commit(wt, _OWN_A, {'a.txt': 'a\nown\n'})
        bump = await _commit(wt, _BUMP, {'index.html': 'v=44\n'})
        main_sha = await _commit(repo, 'main: bump and more', {
            'index.html': 'v=44\n', 'm.txt': 'm\n',
        })

        result = await run_rebase_reporting_drops(
            _run, wt, onto=main_sha, upstream=p1,
        )

        assert result.ok
        assert result.dropped == (
            DroppedCommit(sha=bump, subject=_BUMP, paths=('index.html',)),
        )
        assert [c.sha for c in await RebaseDropLedger(repo, _run).dropped_of('task/1')] == [
            bump,
        ]
        subjects = (await _git(repo, 'log', '--reverse', '--format=%s', 'main..task/1'))
        assert subjects.splitlines() == [_OWN_A]

    async def test_a_conflict_fails_without_aborting_and_records_nothing(
        self, shape: Shape3490,
    ) -> None:
        await _commit(shape.repo, 'main: a differently', {'a.txt': 'a\nmain\n'})

        result = await run_rebase_reporting_drops(_run, shape.wt, onto='main')

        assert not result.ok
        assert result.stderr.strip()
        assert result.dropped == ()
        assert await _rebase_in_progress(shape.wt)
        assert await _dropped_refs(shape.repo) == []

    async def test_a_detached_head_is_reported_but_not_recorded(
        self, shape: Shape3490, caplog: pytest.LogCaptureFixture,
    ) -> None:
        await _commit(shape.repo, 'main: bump and more', {
            'index.html': 'v=44\n', 'm.txt': 'm\n',
        })
        await _git(shape.wt, 'checkout', '--detach')

        with caplog.at_level(logging.WARNING, logger='orchestrator.rebase_drops'):
            result = await run_rebase_reporting_drops(_run, shape.wt, onto='main')

        assert result.ok
        assert result.dropped == (shape.bump_dropped,)
        warnings = _module_warnings(caplog)
        assert len(warnings) == 1, warnings
        assert 'detached HEAD' in warnings[0]
        assert shape.bump[:12] in warnings[0]
        assert _BUMP in warnings[0]
        assert await _dropped_refs(shape.repo) == []

    async def test_a_failed_measurement_never_changes_the_rebase_verdict(
        self, shape: Shape3490, caplog: pytest.LogCaptureFixture,
    ) -> None:
        await _commit(shape.repo, 'main: bump and more', {
            'index.html': 'v=44\n', 'm.txt': 'm\n',
        })

        async def log_fails(cmd: list[str], cwd: Path | None = None, **kwargs: object):
            if cmd[:2] == ['git', 'log']:
                return 128, '', 'fatal: simulated log failure'
            return await _run(cmd, cwd=cwd)

        with caplog.at_level(logging.WARNING, logger='orchestrator.rebase_drops'):
            result = await run_rebase_reporting_drops(log_fails, shape.wt, onto='main')

        assert result.ok
        assert result.dropped == ()
        assert any('could not measure' in w.lower() for w in _module_warnings(caplog))
        await _git(shape.repo, 'merge-base', '--is-ancestor', 'main', 'task/1')
        assert await _dropped_refs(shape.repo) == []

    async def test_a_detected_drop_with_unreadable_subjects_still_names_and_records_its_sha(
        self, shape: Shape3490, caplog: pytest.LogCaptureFixture,
    ) -> None:
        await _commit(shape.repo, 'main: bump and more', {
            'index.html': 'v=44\n', 'm.txt': 'm\n',
        })

        async def show_fails(cmd: list[str], cwd: Path | None = None, **kwargs: object):
            if cmd[:2] == ['git', 'show']:
                return 128, '', 'fatal: simulated show failure'
            return await _run(cmd, cwd=cwd)

        with caplog.at_level(logging.WARNING, logger='orchestrator.rebase_drops'):
            result = await run_rebase_reporting_drops(show_fails, shape.wt, onto='main')

        assert result.ok
        assert result.dropped == ()
        warnings = _module_warnings(caplog)
        assert any(
            'task/1' in w and shape.bump in w and 'dropped 1 commit' in w
            for w in warnings
        ), warnings
        assert await _dropped_refs(shape.repo) == [
            RebaseDropLedger(shape.repo, _run).ref_for('task/1', shape.bump),
        ]
        assert await _branch_subjects(shape) == [_OWN_A, _OWN_B]

    async def test_a_detected_drop_with_an_unreadable_branch_is_named_but_not_recorded(
        self, shape: Shape3490, caplog: pytest.LogCaptureFixture,
    ) -> None:
        await _commit(shape.repo, 'main: bump and more', {
            'index.html': 'v=44\n', 'm.txt': 'm\n',
        })

        async def symbolic_ref_fails(
            cmd: list[str], cwd: Path | None = None, **kwargs: object,
        ):
            if cmd[:2] == ['git', 'symbolic-ref']:
                return 128, '', 'fatal: simulated symbolic-ref failure'
            return await _run(cmd, cwd=cwd)

        with caplog.at_level(logging.WARNING, logger='orchestrator.rebase_drops'):
            result = await run_rebase_reporting_drops(
                symbolic_ref_fails, shape.wt, onto='main',
            )

        assert result.ok
        assert result.dropped == (shape.bump_dropped,)
        warnings = _module_warnings(caplog)
        assert any(
            shape.bump_dropped.label in w and 'not recorded' in w for w in warnings
        ), warnings
        assert await _dropped_refs(shape.repo) == []
