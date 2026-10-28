#!/usr/bin/env python3
"""Census of task branches stacked on an unlanded sibling before task 5618's
stack-base record existed (task 6125).

Task 5618 records the base a branch was stacked onto at the moment of stacking
(orchestrator/src/orchestrator/branch_stack.py::StackBaseLedger), and merge
admission strips that base's commits again
(orchestrator/src/orchestrator/git_ops.py::GitOps.unstack_from_unlanded_base).
A branch stacked before that record existed carries the same foreign prefix
with nothing to say so, and would merge its sibling's commits as its own.

This census finds those branches.  A branch's OWN reflog says which way a
stack points: ``rebase (finish): <ref> onto <X>`` or ``branch: Created from
...`` at an X that is off main is the base the stacker moved it onto, the same
SHA 5618 records.  Containment of a sibling's tip is direction-blind, so it
only surfaces what the reflog cannot explain.  Each unlanded task branch gets
at most one :class:`BranchFinding`, carrying a :class:`Verdict`.

READ-ONLY by default.  ``--adopt-confirmed`` is the sole write: it records a
5618 stack-base ref for each CONFIRMED finding, which hands the rewrite to
5618's admission un-stack.  The census never rewrites a branch.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import _checkout as _checkout
from orchestrator.branch_stack import base_owners, foreign_commit_cut
from shared.git_async import run_git

TASK_BRANCH_PREFIX = 'task/'
_HEADS = 'refs/heads/'
_REBASE_FINISH = re.compile(r'^rebase \(finish\): \S+ onto (?P<base>[0-9a-f]{40})$')
_CREATED_FROM = 'branch: Created from '


class Verdict(StrEnum):
    CONFIRMED = 'confirmed'


class CensusGitError(RuntimeError):
    """A git read the whole census depends on failed."""

    def __init__(self, command: Sequence[str], stderr: str) -> None:
        super().__init__(f'{" ".join(command)} failed: {stderr.strip()}')
        self.command = tuple(command)
        self.stderr = stderr


@dataclass(frozen=True)
class ReflogEntry:
    sha: str
    subject: str


@dataclass(frozen=True)
class StackingEvent:
    """The branch was moved onto *base*, an off-main commit.

    ``prior_tip`` is the branch's tip just before a rebase; a branch created
    on *base* has none.
    """

    base: str
    prior_tip: str | None


@dataclass(frozen=True)
class BranchFinding:
    branch: str
    task_status: str | None
    verdict: Verdict
    base: str | None
    cut: str | None
    contained_siblings: tuple[str, ...]
    base_owners: tuple[str, ...]
    events: tuple[StackingEvent, ...]
    detail: str


@dataclass(frozen=True)
class CensusReport:
    main_branch: str
    main_sha: str
    unlanded_branches: int
    findings: tuple[BranchFinding, ...]
    elapsed_secs: float


async def _git(
    cmd: Sequence[str], cwd: Path | str | None = None, *, input_text: str | None = None,
) -> tuple[int, str, str]:
    """``orchestrator.branch_stack.GitRunner`` over ``shared.git_async.run_git``."""
    result = await run_git(cmd, cwd, input_text=input_text)
    return result.returncode, result.stdout, result.stderr


# ---------------------------------------------------------------------------
# Stage 1: cheap reads over every unlanded task branch
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Snapshot:
    main_sha: str
    main_commits: frozenset[str]
    tips: Mapping[str, str]
    dag: Mapping[str, tuple[str, ...]]
    reflogs: Mapping[str, tuple[ReflogEntry, ...]]


async def _read(repo_root: Path, *args: str, input_text: str | None = None) -> str:
    command = ('git', *args)
    rc, out, err = await _git(command, repo_root, input_text=input_text)
    if rc != 0:
        raise CensusGitError(command, err)
    return out


async def _main_sha(repo_root: Path, main_branch: str) -> str:
    return (await _read(repo_root, 'rev-parse', '--verify', f'{main_branch}^{{commit}}')).strip()


async def _main_commits(repo_root: Path, main_sha: str) -> frozenset[str]:
    return frozenset((await _read(repo_root, 'rev-list', main_sha)).split())


async def _unlanded_tips(repo_root: Path, main_sha: str) -> dict[str, str]:
    out = await _read(
        repo_root, 'for-each-ref', f'--no-merged={main_sha}',
        '--format=%(refname) %(objectname)', f'{_HEADS}{TASK_BRANCH_PREFIX}',
    )
    tips: dict[str, str] = {}
    for line in out.splitlines():
        refname, sha = line.split()
        tips[refname.removeprefix(_HEADS)] = sha
    return tips


async def _unlanded_dag(
    repo_root: Path, main_sha: str, tips: Mapping[str, str],
) -> dict[str, tuple[str, ...]]:
    """``{commit: parents}`` for every commit of an unlanded tip that main lacks."""
    if not tips:
        return {}
    revisions = [*sorted(set(tips.values())), f'^{main_sha}']
    out = await _read(
        repo_root, 'rev-list', '--parents', '--stdin', input_text='\n'.join(revisions) + '\n',
    )
    dag: dict[str, tuple[str, ...]] = {}
    for line in out.splitlines():
        sha, *parents = line.split()
        dag[sha] = tuple(parents)
    return dag


async def _reflog(repo_root: Path, branch: str) -> tuple[ReflogEntry, ...]:
    """*branch*'s reflog, newest first."""
    out = await _read(repo_root, 'reflog', 'show', '--format=%H%x09%gs', f'{_HEADS}{branch}')
    entries = (line.partition('\t') for line in out.splitlines() if line.strip())
    return tuple(ReflogEntry(sha=sha.strip(), subject=subject) for sha, _, subject in entries)


async def _snapshot(repo_root: Path, main_branch: str) -> _Snapshot:
    main_sha = await _main_sha(repo_root, main_branch)
    tips = await _unlanded_tips(repo_root, main_sha)
    names = sorted(tips)
    main_commits, dag, reflogs = await asyncio.gather(
        _main_commits(repo_root, main_sha),
        _unlanded_dag(repo_root, main_sha, tips),
        asyncio.gather(*(_reflog(repo_root, name) for name in names)),
    )
    return _Snapshot(
        main_sha=main_sha, main_commits=main_commits, tips=tips, dag=dag,
        reflogs=dict(zip(names, reflogs, strict=True)),
    )


# ---------------------------------------------------------------------------
# Pure derivations from the snapshot
# ---------------------------------------------------------------------------


def _ancestry(dag: Mapping[str, tuple[str, ...]], start: str) -> frozenset[str]:
    """*start* and its ancestors, as far as the unlanded DAG reaches."""
    seen: set[str] = set()
    pending = [start] if start in dag else []
    while pending:
        sha = pending.pop()
        if sha not in seen:
            seen.add(sha)
            pending.extend(parent for parent in dag[sha] if parent in dag)
    return frozenset(seen)


def contained_siblings(snapshot: _Snapshot, branch: str) -> tuple[str, ...]:
    """Other unlanded task branches whose tip *branch* contains, sorted."""
    reachable = _ancestry(snapshot.dag, snapshot.tips[branch])
    return tuple(sorted(
        name for name, tip in snapshot.tips.items() if name != branch and tip in reachable
    ))


def stacking_events(
    reflog: Sequence[ReflogEntry], main_commits: frozenset[str],
) -> tuple[StackingEvent, ...]:
    """Every move of the branch onto an off-main commit, newest first."""
    found = (_event_at(reflog, index) for index in range(len(reflog)))
    return tuple(
        event for event in found if event is not None and event.base not in main_commits
    )


def _event_at(reflog: Sequence[ReflogEntry], index: int) -> StackingEvent | None:
    entry = reflog[index]
    rebased = _REBASE_FINISH.match(entry.subject)
    if rebased is not None:
        older = reflog[index + 1].sha if index + 1 < len(reflog) else None
        return StackingEvent(base=rebased['base'], prior_tip=older)
    if entry.subject.startswith(_CREATED_FROM):
        return StackingEvent(base=entry.sha, prior_tip=None)
    return None


def _reflog_holders(snapshot: _Snapshot, sha: str) -> frozenset[str]:
    return frozenset(
        name for name, reflog in snapshot.reflogs.items()
        if any(entry.sha == sha for entry in reflog)
    )


# ---------------------------------------------------------------------------
# Stage 2: adjudication of one branch
# ---------------------------------------------------------------------------


async def _live_owners(
    repo_root: Path, snapshot: _Snapshot, branch: str, base: str,
) -> tuple[str, ...]:
    """Branches that still carry *base*: at their tip now, or once in their reflog."""
    at_tip = await base_owners(_git, repo_root, base)
    return tuple(sorted((set(at_tip) | _reflog_holders(snapshot, base)) - {branch}))


async def _finding(repo_root: Path, snapshot: _Snapshot, branch: str) -> BranchFinding | None:
    events = stacking_events(snapshot.reflogs[branch], snapshot.main_commits)
    if not events:
        return None
    base = events[0].base
    cut = await foreign_commit_cut(
        _git, repo_root, tip=snapshot.tips[branch], main_ref=snapshot.main_sha, base=base,
    )
    return BranchFinding(
        branch=branch,
        task_status=None,
        verdict=Verdict.CONFIRMED,
        base=base,
        cut=cut,
        contained_siblings=contained_siblings(snapshot, branch),
        base_owners=await _live_owners(repo_root, snapshot, branch, base),
        events=events,
        detail='',
    )


async def run_census(
    repo_root: Path, *, main_branch: str = 'main', statuses: Mapping[int, str],
) -> CensusReport:
    """Adjudicate every unlanded task branch of *repo_root* against *main_branch*.

    Raises :class:`CensusGitError` when a stage-1 read fails.
    """
    started = time.monotonic()
    snapshot = await _snapshot(repo_root, main_branch)
    found = await asyncio.gather(
        *(_finding(repo_root, snapshot, branch) for branch in sorted(snapshot.tips)),
    )
    return CensusReport(
        main_branch=main_branch,
        main_sha=snapshot.main_sha,
        unlanded_branches=len(snapshot.tips),
        findings=tuple(finding for finding in found if finding is not None),
        elapsed_secs=time.monotonic() - started,
    )
