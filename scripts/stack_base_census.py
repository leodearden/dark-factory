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
from _task_db_scan import connect_ro
from orchestrator.branch_stack import (
    STACK_BASE_REF_NAMESPACE,
    StackBaseLedger,
    StackInspectionError,
    base_owners,
    foreign_commit_cut,
)
from shared.git_async import run_git
from shared.task_statuses import TERMINAL

TASK_BRANCH_PREFIX = 'task/'
_HEADS = 'refs/heads/'
_REBASE_FINISH = re.compile(r'^rebase \(finish\): \S+ onto (?P<base>[0-9a-f]{40})$')
_CREATED_FROM = 'branch: Created from '


class Verdict(StrEnum):
    """What the census concluded about one unlanded task branch."""

    RECORDED = 'recorded'
    """A 5618 stack-base record exists, so merge admission un-stacks it."""
    INERT = 'inert'
    """Looks stacked, but its task is done or cancelled, so it never merges again."""
    UNADJUDICATED = 'unadjudicated'
    """The evidence is ambiguous; a human must read it."""
    ALREADY_CLEAN = 'already_clean'
    """Stacked once, but carries none of the base's commits any more."""
    STRANDED_BASE = 'stranded_base'
    """Carries an unlanded base that no live branch holds any more."""
    CONFIRMED = 'confirmed'
    """Carries an unlanded base a live branch still holds: adopt the record."""


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
class TaskBranch:
    """An unlanded ``task/<id>`` branch; ``task_id`` is None unless ``<id>`` is a number."""

    name: str
    task_id: int | None

    @classmethod
    def parse(cls, name: str) -> TaskBranch:
        suffix = name.removeprefix(TASK_BRANCH_PREFIX)
        numeric = suffix.isascii() and suffix.isdigit()
        return cls(name=name, task_id=int(suffix) if numeric else None)


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
    recorded: frozenset[str]


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


async def _recorded_branches(repo_root: Path) -> frozenset[str]:
    """Branches holding a 5618 stack-base record."""
    out = await _read(repo_root, 'for-each-ref', '--format=%(refname)', STACK_BASE_REF_NAMESPACE)
    return frozenset(line.removeprefix(STACK_BASE_REF_NAMESPACE) for line in out.split())


async def _snapshot(repo_root: Path, main_branch: str) -> _Snapshot:
    main_sha = await _main_sha(repo_root, main_branch)
    tips = await _unlanded_tips(repo_root, main_sha)
    names = sorted(tips)
    main_commits, dag, reflogs, recorded = await asyncio.gather(
        _main_commits(repo_root, main_sha),
        _unlanded_dag(repo_root, main_sha, tips),
        asyncio.gather(*(_reflog(repo_root, name) for name in names)),
        _recorded_branches(repo_root),
    )
    return _Snapshot(
        main_sha=main_sha, main_commits=main_commits, tips=tips, dag=dag,
        reflogs=dict(zip(names, reflogs, strict=True)), recorded=recorded,
    )


# ---------------------------------------------------------------------------
# Pure derivations from the snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Evidence:
    """What stage 1 alone says about one branch."""

    branch: TaskBranch
    task_status: str | None
    tip: str
    events: tuple[StackingEvent, ...]
    contained: tuple[str, ...]
    unexplained: tuple[str, ...]

    @property
    def is_empty(self) -> bool:
        return not (self.events or self.contained)

    @property
    def can_still_merge(self) -> bool:
        """An unknown status counts as live: the conservative reading."""
        return self.task_status not in TERMINAL


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


def _evidence(snapshot: _Snapshot, statuses: Mapping[int, str], name: str) -> _Evidence:
    """A contained sibling is explained when any stacking event's base reaches it."""
    branch = TaskBranch.parse(name)
    events = stacking_events(snapshot.reflogs[name], snapshot.main_commits)
    contained = contained_siblings(snapshot, name)
    explained: set[str] = set()
    for event in events:
        explained |= _ancestry(snapshot.dag, event.base)
    return _Evidence(
        branch=branch,
        task_status=None if branch.task_id is None else statuses.get(branch.task_id),
        tip=snapshot.tips[name],
        events=events,
        contained=contained,
        unexplained=tuple(name for name in contained if snapshot.tips[name] not in explained),
    )


def _reflog_holders(snapshot: _Snapshot, sha: str) -> frozenset[str]:
    return frozenset(
        name for name, reflog in snapshot.reflogs.items()
        if any(entry.sha == sha for entry in reflog)
    )


# ---------------------------------------------------------------------------
# Stage 2: per-branch git checks, for branches whose task can still merge
# ---------------------------------------------------------------------------


async def _checked(repo_root: Path, *args: str, answers: tuple[int, ...] = (0,)) -> int:
    """Run git; any exit code outside *answers* is a failed inspection."""
    command = ('git', *args)
    rc, _, err = await _git(command, repo_root)
    if rc not in answers:
        raise StackInspectionError(command, err)
    return rc


async def _commit_exists(repo_root: Path, sha: str) -> bool:
    rc, _, _ = await _git(['git', 'cat-file', '-e', f'{sha}^{{commit}}'], repo_root)
    return rc == 0


async def _is_ancestor(repo_root: Path, ancestor: str, descendant: str) -> bool:
    rc = await _checked(
        repo_root, 'merge-base', '--is-ancestor', ancestor, descendant, answers=(0, 1),
    )
    return rc == 0


async def _patch_landed(repo_root: Path, main_sha: str, base: str) -> bool:
    """Every commit of *base* that main lacks has a patch-equivalent on main."""
    command = ('git', 'cherry', main_sha, base)
    rc, out, err = await _git(command, repo_root)
    if rc != 0:
        raise StackInspectionError(command, err)
    return not any(line.startswith('+') for line in out.splitlines())


class _LandedChecks:
    """One ``git cherry`` per distinct base per census, however many branches share it."""

    def __init__(self, repo_root: Path, main_sha: str) -> None:
        self._repo_root = repo_root
        self._main_sha = main_sha
        self._checks: dict[str, asyncio.Future[bool]] = {}

    async def landed(self, base: str) -> bool:
        check = self._checks.get(base)
        if check is None:
            check = asyncio.ensure_future(_patch_landed(self._repo_root, self._main_sha, base))
            self._checks[base] = check
        return await check


@dataclass(frozen=True)
class _CensusRun:
    repo_root: Path
    snapshot: _Snapshot
    statuses: Mapping[int, str]
    landed: _LandedChecks


@dataclass(frozen=True)
class _Adjudication:
    """Stage 2's facts.  ``base`` is the newest surviving event's base."""

    surviving: tuple[StackingEvent, ...] = ()
    base: str | None = None
    cut: str | None = None
    residual: tuple[str, ...] = ()
    owners: tuple[str, ...] = ()
    error: str | None = None


async def _survives(census_run: _CensusRun, event: StackingEvent) -> bool:
    """A base that is gone, the branch's own history, or patch-landed stacks nothing."""
    if not await _commit_exists(census_run.repo_root, event.base):
        return False
    if event.prior_tip is not None and await _is_ancestor(
        census_run.repo_root, event.base, event.prior_tip,
    ):
        return False
    return not await census_run.landed.landed(event.base)


async def _residual_bases(
    census_run: _CensusRun, tip: str, cut: str | None, newest: str, older: Sequence[StackingEvent],
) -> tuple[str, ...]:
    """Older bases whose commits still sit below the newest base's cut."""
    main_ref = cut or census_run.snapshot.main_sha
    candidates = dict.fromkeys(event.base for event in older if event.base != newest)
    return tuple([
        base for base in candidates
        if await foreign_commit_cut(
            _git, census_run.repo_root, tip=tip, main_ref=main_ref, base=base,
        ) is not None
    ])


async def _live_owners(census_run: _CensusRun, branch: str, base: str) -> tuple[str, ...]:
    """Branches that still carry *base*: at their tip now, or once in their reflog."""
    at_tip = await base_owners(_git, census_run.repo_root, base)
    held = _reflog_holders(census_run.snapshot, base)
    return tuple(sorted((set(at_tip) | held) - {branch}))


async def _adjudicate(census_run: _CensusRun, evidence: _Evidence) -> _Adjudication:
    try:
        surviving = tuple([event for event in evidence.events if await _survives(census_run, event)])
        if not surviving:
            return _Adjudication()
        base = surviving[0].base
        cut = await foreign_commit_cut(
            _git, census_run.repo_root, tip=evidence.tip, main_ref=census_run.snapshot.main_sha, base=base,
        )
        return _Adjudication(
            surviving=surviving,
            base=base,
            cut=cut,
            residual=await _residual_bases(census_run, evidence.tip, cut, base, surviving[1:]),
            owners=await _live_owners(census_run, evidence.branch.name, base),
        )
    except StackInspectionError as exc:
        newest = evidence.events[0].base if evidence.events else None
        return _Adjudication(base=newest, error=str(exc))


def _verdict(evidence: _Evidence, adjudication: _Adjudication) -> Verdict | None:
    """Precedence after RECORDED and INERT, which need no stage-2 fact (see _finding)."""
    if adjudication.error is None and not adjudication.surviving and not evidence.unexplained:
        return None
    if evidence.unexplained or adjudication.residual or adjudication.error is not None:
        return Verdict.UNADJUDICATED
    if adjudication.cut is None:
        return Verdict.ALREADY_CLEAN
    if not adjudication.owners:
        return Verdict.STRANDED_BASE
    return Verdict.CONFIRMED


def _unadjudicated_detail(evidence: _Evidence, adjudication: _Adjudication) -> str:
    clauses = [
        f'contains {name} with no stacking event that explains it'
        for name in evidence.unexplained
    ]
    clauses += [
        f'older base {base} still sits below the cut' for base in adjudication.residual
    ]
    if adjudication.error is not None:
        clauses.append(f'git inspection failed: {adjudication.error}')
    return '; '.join(clauses)


# ---------------------------------------------------------------------------
# One finding per branch
# ---------------------------------------------------------------------------


async def _recorded_finding(census_run: _CensusRun, evidence: _Evidence) -> BranchFinding:
    base = await StackBaseLedger(census_run.repo_root, _git).base_of(evidence.branch.name)
    return _unchecked_finding(evidence, Verdict.RECORDED, base)


def _inert_finding(evidence: _Evidence) -> BranchFinding:
    newest = evidence.events[0].base if evidence.events else None
    return _unchecked_finding(evidence, Verdict.INERT, newest)


def _unchecked_finding(evidence: _Evidence, verdict: Verdict, base: str | None) -> BranchFinding:
    """A finding decided without stage 2: it has no cut and no owners."""
    return BranchFinding(
        branch=evidence.branch.name, task_status=evidence.task_status, verdict=verdict,
        base=base, cut=None, contained_siblings=evidence.contained, base_owners=(),
        events=evidence.events, detail='',
    )


async def _adjudicated_finding(census_run: _CensusRun, evidence: _Evidence) -> BranchFinding | None:
    adjudication = await _adjudicate(census_run, evidence)
    verdict = _verdict(evidence, adjudication)
    if verdict is None:
        return None
    detail = (
        _unadjudicated_detail(evidence, adjudication)
        if verdict is Verdict.UNADJUDICATED else ''
    )
    return BranchFinding(
        branch=evidence.branch.name, task_status=evidence.task_status, verdict=verdict,
        base=adjudication.base,
        cut=adjudication.cut, contained_siblings=evidence.contained,
        base_owners=adjudication.owners, events=evidence.events, detail=detail,
    )


async def _finding(census_run: _CensusRun, name: str) -> BranchFinding | None:
    """RECORDED and INERT need no stage-2 fact, so they are decided before it runs."""
    evidence = _evidence(census_run.snapshot, census_run.statuses, name)
    if name in census_run.snapshot.recorded:
        return await _recorded_finding(census_run, evidence)
    if evidence.is_empty:
        return None
    if not evidence.can_still_merge:
        return _inert_finding(evidence)
    return await _adjudicated_finding(census_run, evidence)


async def run_census(
    repo_root: Path, *, main_branch: str = 'main', statuses: Mapping[int, str],
) -> CensusReport:
    """Adjudicate every unlanded task branch of *repo_root* against *main_branch*.

    Raises :class:`CensusGitError` when a stage-1 read fails.  A stage-2
    failure makes only that branch UNADJUDICATED.
    """
    started = time.monotonic()
    snapshot = await _snapshot(repo_root, main_branch)
    census_run = _CensusRun(
        repo_root, snapshot, statuses, _LandedChecks(repo_root, snapshot.main_sha),
    )
    found = await asyncio.gather(*(_finding(census_run, branch) for branch in sorted(snapshot.tips)))
    return CensusReport(
        main_branch=main_branch,
        main_sha=snapshot.main_sha,
        unlanded_branches=len(snapshot.tips),
        findings=tuple(finding for finding in found if finding is not None),
        elapsed_secs=time.monotonic() - started,
    )


def load_task_statuses(db_path: Path) -> dict[int, str]:
    """``{task id: status}`` for tag 'master', read through a read-only connection."""
    conn = connect_ro(db_path)
    try:
        rows = conn.execute("SELECT id, status FROM tasks WHERE tag = 'master'").fetchall()
    finally:
        conn.close()
    return {int(task_id): str(status) for task_id, status in rows}
