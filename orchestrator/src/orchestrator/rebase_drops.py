"""Commits a rebase silently dropped, recorded under the branch they left (task 4669).

``git rebase`` drops a branch commit whose change the rebase target already
carries, and exits 0.  It says so only on stderr, in free text: "patch
contents already upstream" when replaying the commit leaves it empty, and
"skipped previously applied commit" when it is patch-identical to a target
commit.  Afterwards the branch simply no longer carries the commit, so a
later reader asking "did this branch change file X?" concludes it never did.

This module keeps the fact that it did.  :class:`RebaseDropLedger` holds one
git ref per dropped commit, ``<namespace><full_branch>/<sha>``, pointing at
the dropped commit itself.  A ref is atomic, is shared by every worktree,
survives restarts, and keeps the dropped commit reachable after the reflog
that once held it expires.  It is the same convention as
``orchestrator/src/orchestrator/branch_stack.py::StackBaseLedger``, and the
two share their record-lifetime mechanics in
``orchestrator/src/orchestrator/ref_ledger.py``.

:func:`run_rebase_reporting_drops` runs a task-branch rebase and detects its
drops without reading git's stderr, whose wording varies across versions.  It
lists the replayed range before the rebase and ``<onto>..HEAD`` after it, and
matches the two as a multiset on the identity a rebase preserves: author name,
author email, raw author date and subject.  Merge commits are left out, since
a rebase linearizes them by design, and post-rebase commits with no
pre-rebase match are ignored, so a target that moved mid-rebase cannot
fabricate a drop.

Accepted residuals:

1. A drop on a DETACHED HEAD is logged but not recorded: there is no branch
   to key the record on.
2. Records move with a branch ``GitOps`` renames (``rename_worktree``,
   ``quarantine_worktree``, and the recut ruling's
   ``park_prior_incarnation``), so a branch cut afresh under the old name
   starts with none.  A DELETED branch's records stay until the next record
   anywhere prunes them, so a branch re-created under that name before then
   inherits them.
3. Records of a rebase the caller later reverts (``BranchResetError`` in
   ``GitOps.rebase_preserving_task_commits``) remain.
4. The replayed range is every commit on the branch past the target, so a
   branch stacked on predecessors (a coalesce train tip, a suffix frozen-tip
   rebase) records a dropped PREDECESSOR commit under its own name.  The
   subject a message cites still names the task that wrote it.

Records are only consulted for a plan file the branch did NOT touch, so none
of these can make a passing gate fail, and every message that cites a record
names its sha and subject for a reader to judge.

Layering: this module sits BELOW ``orchestrator/src/orchestrator/branch_stack.py``
and ``orchestrator/src/orchestrator/git_ops.py``, which both import it.  The
git runner is injected, exactly as
``orchestrator/src/orchestrator/rebase_recovery.py::guarded_abort`` receives
its runner, and nothing here imports git_ops or merge_lane.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Awaitable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, TypeVar

from orchestrator.ref_ledger import (
    BRANCH_REF_NAMESPACE,
    GitReadError,
    GitRunner,
    git_lines,
    git_out,
    move_branch_records,
    prune_orphaned_records,
)

logger = logging.getLogger(__name__)

_T = TypeVar('_T')

REBASE_DROP_REF_NAMESPACE = 'refs/dark-factory/rebase-dropped/'


@dataclass(frozen=True)
class DroppedCommit:
    """A commit a rebase dropped: its pre-rebase sha, its subject, and the
    paths it changed, sorted as git lists them."""

    sha: str
    subject: str
    paths: tuple[str, ...]

    @property
    def label(self) -> str:
        """How a log line or a block reason names this commit."""
        return f'{self.sha[:12]} "{self.subject}"'


@dataclass(frozen=True)
class RebaseDropLedger:
    """One ref per dropped commit: ``<namespace><full_branch>/<sha>`` -> sha.

    *repo_root* may be any checkout of the repository: refs outside HEAD and
    the per-worktree namespaces are shared by every worktree.
    """

    repo_root: Path
    run: GitRunner

    def ref_for(self, full_branch: str, sha: str) -> str:
        return f'{_records_of(full_branch)}{sha}'

    async def record(self, full_branch: str, shas: Iterable[str]) -> None:
        """Record each of *shas* as dropped from *full_branch*.

        A failed git command is logged, not raised.
        """
        for sha in dict.fromkeys(shas):
            rc, _, err = await self.run(
                ['git', 'update-ref', self.ref_for(full_branch, sha), sha],
                cwd=self.repo_root,
            )
            if rc != 0:
                logger.warning(
                    'Could not record rebase-dropped commit %s of %s: %s',
                    sha, full_branch, err.strip(),
                )

    async def dropped_of(self, full_branch: str) -> tuple[DroppedCommit, ...]:
        """Every commit recorded as dropped from *full_branch*, ordered by sha.

        A failed git read is logged and yields ().
        """
        try:
            shas = await git_lines(
                self.run, self.repo_root,
                'for-each-ref', '--format=%(objectname)', _records_of(full_branch),
            )
            return await _read_dropped_commits(self.run, self.repo_root, sorted(shas))
        except GitReadError as exc:
            logger.warning(
                'Could not read rebase-dropped commits of %s: %s', full_branch, exc,
            )
            return ()

    async def prune_orphans(self) -> tuple[str, ...]:
        """Drop every record whose branch no longer exists; return those branches.

        A failed git command is logged, not raised.
        """
        return await prune_orphaned_records(
            self.run, self.repo_root, REBASE_DROP_REF_NAMESPACE,
            branch_of=lambda key: key.rpartition('/')[0],
        )

    async def move_branch(self, old_branch: str, new_branch: str) -> None:
        """Re-key *old_branch*'s records to *new_branch*, which it was renamed to.

        A failed git command is logged, not raised.
        """
        await move_branch_records(
            self.run, self.repo_root, _records_of(old_branch), _records_of(new_branch),
        )


def _records_of(full_branch: str) -> str:
    """The ref prefix every record of *full_branch* sits beneath."""
    return f'{REBASE_DROP_REF_NAMESPACE}{full_branch}/'


class RebaseRun(NamedTuple):
    """A rebase's verdict, its stderr, and the commits it dropped."""

    ok: bool
    stderr: str
    dropped: tuple[DroppedCommit, ...]


class _ReplayedCommit(NamedTuple):
    sha: str
    identity: tuple[str, ...]


class _Baseline(NamedTuple):
    onto_sha: str
    replayed: tuple[_ReplayedCommit, ...]


_REPLAYED_FORMAT = '%H%x1f%an%x1f%ae%x1f%ad%x1f%s'


async def run_rebase_reporting_drops(
    run: GitRunner, worktree: Path, *, onto: str, upstream: str | None = None,
) -> RebaseRun:
    """Run ``git rebase [--onto <onto> <upstream> | <onto>]`` in *worktree*,
    and report every commit it dropped because *onto* already carried it.

    Each drop is logged as a WARNING and, when a branch is checked out,
    recorded in :class:`RebaseDropLedger`.  A failed rebase is returned as
    is, NOT aborted: the caller owns the abort.  Measuring the drops never
    raises and never changes ``ok``; a measurement that fails before any drop
    is detected is logged and reports no drops.
    """
    baseline = await _measured(
        _read_baseline(run, worktree, onto, upstream), None, 'rebase drops', worktree,
    )
    argv = ['git', 'rebase', onto] if upstream is None else [
        'git', 'rebase', '--onto', onto, upstream,
    ]
    rc, _, err = await run(argv, cwd=worktree)
    if rc != 0:
        return RebaseRun(ok=False, stderr=err, dropped=())
    dropped: tuple[DroppedCommit, ...] = ()
    if baseline is not None:
        dropped = await _measured(
            _report_drops(run, worktree, baseline), (), 'rebase drops', worktree,
        )
    return RebaseRun(ok=True, stderr=err, dropped=dropped)


async def _measured(
    measurement: Awaitable[_T], fallback: _T, what: str, worktree: Path,
) -> _T:
    """*measurement*'s result, or *fallback* once its failure is logged."""
    try:
        return await measurement
    except Exception as exc:
        logger.warning('Could not measure %s in %s: %s', what, worktree, exc)
        return fallback


async def _read_baseline(
    run: GitRunner, worktree: Path, onto: str, upstream: str | None,
) -> _Baseline:
    onto_sha = (await git_out(
        run, worktree, 'rev-parse', '--verify', f'{onto}^{{commit}}',
    )).strip()
    start = onto_sha if upstream is None else upstream
    return _Baseline(onto_sha, await _replayed(run, worktree, f'{start}..HEAD'))


async def _replayed(
    run: GitRunner, worktree: Path, revision_range: str,
) -> tuple[_ReplayedCommit, ...]:
    out = await git_out(
        run, worktree, 'log', '--no-merges', '--reverse', '-z', '--date=raw',
        f'--format={_REPLAYED_FORMAT}', revision_range,
    )
    fields = (record.split('\x1f', 4) for record in out.split('\x00') if record.strip())
    return tuple(_ReplayedCommit(f[0].strip(), tuple(f[1:])) for f in fields)


def _lost(
    before: Sequence[_ReplayedCommit], after: Sequence[_ReplayedCommit],
) -> list[str]:
    """Shas of *before* with no identity match left in *after*, in order."""
    surviving = Counter(commit.identity for commit in after)
    lost: list[str] = []
    for commit in before:
        if surviving[commit.identity] > 0:
            surviving[commit.identity] -= 1
        else:
            lost.append(commit.sha)
    return lost


async def _report_drops(
    run: GitRunner, worktree: Path, baseline: _Baseline,
) -> tuple[DroppedCommit, ...]:
    """Warn about, then record, each commit the rebase dropped; return them.

    Once a drop is detected it is always warned about.  A later read that
    fails degrades the report instead of discarding it: an unreadable
    checkout leaves the drop unrecorded, and unreadable subjects leave the
    warning naming bare shas and the returned tuple empty.
    """
    after = await _replayed(run, worktree, f'{baseline.onto_sha}..HEAD')
    lost = _lost(baseline.replayed, after)
    if not lost:
        return ()
    checkout = await _measured(
        _read_checkout(run, worktree), _UNREADABLE_CHECKOUT,
        'the branch a rebase dropped commits from', worktree,
    )
    dropped = await _measured(
        _read_dropped_commits(run, worktree, lost), None,
        'the subjects of commits a rebase dropped', worktree,
    )
    labels = lost if dropped is None else [commit.label for commit in dropped]
    _warn_dropped(worktree, checkout.named, baseline.onto_sha, labels)
    if checkout.branch is not None:
        ledger = RebaseDropLedger(worktree, run)
        await ledger.prune_orphans()
        await ledger.record(checkout.branch, lost)
    return dropped or ()


class _Checkout(NamedTuple):
    """The branch a drop is recorded under (None: not recorded), and how a
    warning names the checkout."""

    branch: str | None
    named: str


_UNREADABLE_CHECKOUT = _Checkout(None, 'an unreadable checkout (not recorded)')


async def _read_checkout(run: GitRunner, worktree: Path) -> _Checkout:
    """What is checked out in *worktree*: a branch, or a detached HEAD."""
    command = ('git', 'symbolic-ref', '--quiet', 'HEAD')
    rc, out, err = await run(list(command), cwd=worktree)
    if rc == 1:
        return _Checkout(None, 'detached HEAD (not recorded)')
    if rc != 0:
        raise GitReadError(command, err)
    branch = out.strip().removeprefix(BRANCH_REF_NAMESPACE)
    return _Checkout(branch, branch)


def _warn_dropped(
    worktree: Path, checkout: str, onto_sha: str, labels: Sequence[str],
) -> None:
    logger.warning(
        'Rebase of %s in %s onto %s dropped %d commit(s) because the target '
        'already carried each change: %s',
        checkout, worktree, onto_sha[:12], len(labels), '; '.join(labels),
    )


#: Opens each commit's header in :func:`_read_dropped_commits`'s log, so a
#: header is told apart from the NUL-terminated paths listed after it.  It is
#: not whitespace to ``str.strip``, unlike ``\x1c``-``\x1f``, so a runner that
#: strips stdout cannot eat the first header's mark.
_HEADER_MARK = '\x01'


async def _read_dropped_commits(
    run: GitRunner, cwd: Path, shas: Sequence[str],
) -> tuple[DroppedCommit, ...]:
    """Subject and changed paths of each of *shas*, in the order given, in one read.

    ``--no-renames --root`` lists both sides of a rename and every path of a
    root commit, whatever the repository's diff configuration says.  Raises
    :class:`orchestrator.ref_ledger.GitReadError` when the read fails.
    """
    if not shas:
        return ()
    out = await git_out(
        run, cwd, 'log', '--no-walk=unsorted', '--no-renames', '--root', '-z',
        '--name-only', '--format=%x01%H%x1f%s', *shas,
    )
    return _parse_name_only_log(out)


def _parse_name_only_log(out: str) -> tuple[DroppedCommit, ...]:
    """The commits in ``git log -z --name-only`` output, in order.

    Each commit is a ``<mark><sha>\\x1f<subject>`` header followed by its
    NUL-terminated paths, the first of which git prefixes with a newline.
    """
    commits: list[tuple[str, str, list[str]]] = []
    for token in out.split('\x00'):
        if token.startswith(_HEADER_MARK):
            sha, _, subject = token.removeprefix(_HEADER_MARK).partition('\x1f')
            commits.append((sha, subject.strip(), []))
        elif token and commits:
            paths = commits[-1][2]
            paths.append(token if paths else token.removeprefix('\n'))
    return tuple(
        DroppedCommit(sha=sha, subject=subject, paths=tuple(paths))
        for sha, subject, paths in commits
    )
