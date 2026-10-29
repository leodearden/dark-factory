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

import argparse
import asyncio
import dataclasses
import json
import re
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import _checkout as _checkout
from _task_db_scan import UNREADABLE_STORE_ERRORS, connect_ro, tasks_db_path
from orchestrator.branch_stack import (
    STACK_BASE_REF_NAMESPACE,
    StackBaseLedger,
    StackInspectionError,
    base_owners,
    foreign_commit_cut,
)
from shared.git_async import run_git
from shared.project_layout import default_project_root
from shared.task_statuses import TERMINAL

TASK_BRANCH_PREFIX = 'task/'
_HEADS = 'refs/heads/'
_REBASE_FINISH = re.compile(r'^rebase \(finish\): \S+ onto (?P<base>[0-9a-f]{40})$')
_CREATED_FROM = 'branch: Created from '


class Verdict(StrEnum):
    """What the census concluded about one unlanded task branch."""

    RECORDED = 'recorded'
    """A 5618 stack-base record names a commit, so merge admission un-stacks it."""
    INERT = 'inert'
    """Looks stacked, but its task is done or cancelled, so it never merges again."""
    UNADJUDICATED = 'unadjudicated'
    """The evidence is ambiguous; a human must read it."""
    ALREADY_CLEAN = 'already_clean'
    """Stacked once, but carries none of the base's commits any more."""
    STRANDED_BASE = 'stranded_base'
    """Carries an unlanded base that no other branch holds as its own work any more."""
    CONFIRMED = 'confirmed'
    """Carries an unlanded base another branch still holds as its own: adopt the record."""


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

    def __post_init__(self) -> None:
        if self.verdict in _VERDICTS_WITH_A_BASE and self.base is None:
            raise ValueError(f'{self.verdict} finding for {self.branch} needs a base')
        if self.verdict in _VERDICTS_WITH_A_CUT and self.cut is None:
            raise ValueError(f'{self.verdict} finding for {self.branch} needs a cut')


_VERDICTS_WITH_A_CUT = frozenset({Verdict.STRANDED_BASE, Verdict.CONFIRMED})
_VERDICTS_WITH_A_BASE = _VERDICTS_WITH_A_CUT | {Verdict.ALREADY_CLEAN}


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
    """Other unlanded task branches whose tip is a strict ancestor of *branch*'s, sorted.

    A sibling at the very same commit is no evidence either way: neither
    branch has a commit the other lacks.
    """
    own_tip = snapshot.tips[branch]
    reachable = _ancestry(snapshot.dag, own_tip) - {own_tip}
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
    """False only when git answers that *sha* names no commit; an unreadable object raises."""
    rc = await _checked(
        repo_root, 'rev-parse', '--verify', '--quiet', f'{sha}^{{commit}}', answers=(0, 1),
    )
    return rc == 0


async def _is_ancestor(repo_root: Path, ancestor: str, descendant: str) -> bool:
    rc = await _checked(
        repo_root, 'merge-base', '--is-ancestor', ancestor, descendant, answers=(0, 1),
    )
    return rc == 0


async def _all_patches_upstream(repo_root: Path, *cherry_args: str) -> bool:
    """``git cherry <cherry_args>`` lists no commit that lacks an upstream patch-equivalent."""
    command = ('git', 'cherry', *cherry_args)
    rc, out, err = await _git(command, repo_root)
    if rc != 0:
        raise StackInspectionError(command, err)
    return not any(line.startswith('+') for line in out.splitlines())


async def _patch_landed(repo_root: Path, main_sha: str, base: str) -> bool:
    """Every commit of *base* that main lacks has a patch-equivalent on main."""
    return await _all_patches_upstream(repo_root, main_sha, base)


async def _tip_carries(repo_root: Path, main_sha: str, tip: str, base: str) -> bool:
    """*tip* holds every commit of *base* that main lacks, as itself or a patch-equivalent."""
    if await _is_ancestor(repo_root, base, tip):
        return True
    return await _all_patches_upstream(repo_root, tip, base, main_sha)


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
    """Branches other than *branch* that hold *base* as their own work, sorted.

    A branch at *base* holds it.  A branch whose reflog once held *base*
    holds it only while its tip still carries it.  A branch stacked onto
    *base* carries it as a foreign prefix, so it is no owner.
    """
    snapshot = census_run.snapshot
    at_tip = set(await base_owners(_git, census_run.repo_root, base))
    once_held = sorted(_reflog_holders(snapshot, base) - at_tip - {branch})
    still_held = {
        name for name in once_held
        if await _tip_carries(census_run.repo_root, snapshot.main_sha, snapshot.tips[name], base)
    }
    return tuple(sorted(
        name for name in (at_tip | still_held) - {branch}
        if not _stacked_onto(snapshot, name, base)
    ))


def _stacked_onto(snapshot: _Snapshot, name: str, base: str) -> bool:
    events = stacking_events(snapshot.reflogs.get(name, ()), snapshot.main_commits)
    return any(event.base == base for event in events)


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
    """Merge admission reads the record as no record unless it names a commit."""
    ledger = StackBaseLedger(census_run.repo_root, _git)
    base = await ledger.base_of(evidence.branch.name)
    if base is not None:
        return _unchecked_finding(evidence, Verdict.RECORDED, base)
    return _unchecked_finding(
        evidence, Verdict.UNADJUDICATED, None,
        detail=(
            f'{ledger.ref_for(evidence.branch.name)} names no commit, '
            'so merge admission would not un-stack the branch'
        ),
    )


def _inert_finding(evidence: _Evidence) -> BranchFinding:
    newest = evidence.events[0].base if evidence.events else None
    return _unchecked_finding(evidence, Verdict.INERT, newest)


def _unchecked_finding(
    evidence: _Evidence, verdict: Verdict, base: str | None, *, detail: str = '',
) -> BranchFinding:
    """A finding decided without stage 2: it has no cut and no owners."""
    return BranchFinding(
        branch=evidence.branch.name, task_status=evidence.task_status, verdict=verdict,
        base=base, cut=None, contained_siblings=evidence.contained, base_owners=(),
        events=evidence.events, detail=detail,
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


# ---------------------------------------------------------------------------
# Adoption: the census's only write
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdoptOutcome:
    branch: str
    base: str
    written: bool
    detail: str


async def adopt_confirmed(repo_root: Path, report: CensusReport) -> tuple[AdoptOutcome, ...]:
    """Record a 5618 stack-base ref for every CONFIRMED finding, in branch order.

    Every other verdict is skipped.  ``StackBaseLedger.record`` logs and
    swallows its own failure, so each record is read back, and ``written``
    reports only what the ref says afterwards.
    """
    ledger = StackBaseLedger(repo_root, _git)
    confirmed = sorted(
        (finding.branch, finding.base) for finding in report.findings
        if finding.verdict is Verdict.CONFIRMED and finding.base is not None
    )
    return tuple([await _adopt(ledger, branch, base) for branch, base in confirmed])


async def _adopt(ledger: StackBaseLedger, branch: str, base: str) -> AdoptOutcome:
    await ledger.record(branch, base)
    found = await ledger.base_of(branch)
    if found == base:
        return AdoptOutcome(
            branch=branch, base=base, written=True,
            detail=f'recorded {ledger.ref_for(branch)} -> {base}',
        )
    return AdoptOutcome(
        branch=branch, base=base, written=False,
        detail=(
            f'the record of {branch} at {base} did not read back: '
            f'{ledger.ref_for(branch)} reads {found or "nothing"}'
        ),
    )


def load_task_statuses(db_path: Path) -> dict[int, str]:
    """``{task id: status}`` for tag 'master', read through a read-only connection."""
    conn = connect_ro(db_path)
    try:
        rows = conn.execute("SELECT id, status FROM tasks WHERE tag = 'master'").fetchall()
    finally:
        conn.close()
    return {int(task_id): str(status) for task_id, status in rows}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

ACTIONABLE = (Verdict.CONFIRMED, Verdict.STRANDED_BASE, Verdict.UNADJUDICATED)
_NOT_ACTIONABLE = (Verdict.ALREADY_CLEAN, Verdict.RECORDED, Verdict.INERT)


def render_markdown(
    report: CensusReport, outcomes: Sequence[AdoptOutcome] | None = None,
) -> str:
    """The census as a markdown report; *outcomes* is None unless adopting."""
    lines = [
        '# Legacy stack-base census',
        '',
        f'- main: `{report.main_branch}` at `{report.main_sha}`',
        f'- unlanded task branches: {report.unlanded_branches}',
        f'- findings: {len(report.findings)}',
        f'- elapsed: {report.elapsed_secs:.1f}s',
        '',
        '| verdict | branches |',
        '|---|---|',
        *(f'| {verdict} | {len(_with(report, verdict))} |' for verdict in (
            *ACTIONABLE, *_NOT_ACTIONABLE,
        )),
        '',
        '## Actionable findings',
        '',
        *_actionable_sections(report),
        '## Not actionable',
        '',
        *(_branch_list(verdict, _with(report, verdict)) for verdict in _NOT_ACTIONABLE),
        *_adoption_section(outcomes),
    ]
    return '\n'.join(lines).rstrip('\n') + '\n'


def _with(report: CensusReport, verdict: Verdict) -> tuple[BranchFinding, ...]:
    return tuple(finding for finding in report.findings if finding.verdict is verdict)


def _actionable_sections(report: CensusReport) -> list[str]:
    actionable = [finding for finding in report.findings if finding.verdict in ACTIONABLE]
    if not actionable:
        return ['None.', '']
    return [line for finding in actionable for line in _finding_section(report, finding)]


def _finding_section(report: CensusReport, finding: BranchFinding) -> list[str]:
    evidence = [f'- evidence: {finding.detail}'] if finding.detail else []
    return [
        f'### {finding.branch}: {finding.verdict}',
        '',
        f'- task status: {finding.task_status or "unknown, adjudicated as live"}',
        f'- base: {_sha(finding.base)}',
        f'- live owners of the base: {", ".join(finding.base_owners) or "none"}',
        f'- cut: {_sha(finding.cut)}',
        f'- contained siblings: {", ".join(finding.contained_siblings) or "none"}',
        f'- stacking events, newest first: {_events(finding.events)}',
        *evidence,
        f'- remedy: {_remedy(report, finding)}',
        '',
    ]


def _sha(sha: str | None) -> str:
    return f'`{sha}`' if sha is not None else 'none'


def _events(events: Sequence[StackingEvent]) -> str:
    described = (
        f'created on `{event.base}`' if event.prior_tip is None
        else f'rebased from `{event.prior_tip}` onto `{event.base}`'
        for event in events
    )
    return '; '.join(described) or 'none'


def _remedy(report: CensusReport, finding: BranchFinding) -> str:
    if finding.verdict is Verdict.CONFIRMED:
        return (
            'run `scripts/stack_base_census.py --adopt-confirmed` to record the base, so '
            "task 5618's merge admission un-stacks the branch; by hand: "
            f'`git rebase --onto {report.main_branch} {finding.cut} {finding.branch}`'
        )
    if finding.verdict is Verdict.STRANDED_BASE:
        return (
            'no other branch holds the base as its own work any more, so un-stacking '
            "could drop the last copy of the base's commits. Decide first whether they "
            'must land; once adjudicated, record the base with '
            f'`git update-ref {STACK_BASE_REF_NAMESPACE}{finding.branch} {finding.base}`'
        )
    return (
        'adjudicate by hand from the contained siblings, stacking events and evidence '
        'above; record a base only once the direction of the stack is settled'
    )


def _branch_list(verdict: Verdict, findings: Sequence[BranchFinding]) -> str:
    names = ', '.join(finding.branch for finding in findings) or 'none'
    return f'- {verdict} ({len(findings)}): {names}'


def _adoption_section(outcomes: Sequence[AdoptOutcome] | None) -> list[str]:
    if outcomes is None:
        return []
    lines = [_adoption_line(outcome) for outcome in outcomes]
    return ['', '## Adoption', '', *(lines or ['No CONFIRMED finding to adopt.'])]


def _adoption_line(outcome: AdoptOutcome) -> str:
    state = 'written' if outcome.written else 'NOT WRITTEN'
    return f'adopt {outcome.branch} at {outcome.base}: {state} ({outcome.detail})'


def render_json(report: CensusReport, outcomes: Sequence[AdoptOutcome] | None = None) -> str:
    """The census as JSON; verdicts are their string values."""
    payload = dataclasses.asdict(report)
    if outcomes is not None:
        payload['adoptions'] = [dataclasses.asdict(outcome) for outcome in outcomes]
    return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Census of task branches stacked on an unlanded sibling before task '
        '5618 recorded stack bases. Read-only unless --adopt-confirmed.',
    )
    parser.add_argument(
        '--project-root', type=Path, default=None,
        help='repository to census (default: the main checkout)',
    )
    parser.add_argument('--main-branch', default='main')
    parser.add_argument(
        '--tasks-db', type=Path, default=None,
        help='task store to read statuses from (default: <project-root>/.taskmaster/tasks/tasks.db)',
    )
    parser.add_argument('--json', action='store_true', help='print JSON instead of markdown')
    parser.add_argument(
        '--adopt-confirmed', action='store_true',
        help='record a 5618 stack-base ref for every CONFIRMED finding (the only write)',
    )
    return parser


def _statuses_or_warn(db_path: Path) -> dict[int, str]:
    try:
        return load_task_statuses(db_path)
    except UNREADABLE_STORE_ERRORS as exc:
        print(
            f'warning: task store {db_path} is unreadable ({exc}); '
            'every branch is adjudicated as live',
            file=sys.stderr,
        )
        return {}


async def _census_and_adopt(
    project_root: Path, main_branch: str, statuses: Mapping[int, str], adopt: bool,
) -> tuple[CensusReport, tuple[AdoptOutcome, ...] | None]:
    report = await run_census(project_root, main_branch=main_branch, statuses=statuses)
    outcomes = await adopt_confirmed(project_root, report) if adopt else None
    return report, outcomes


def main(argv: Sequence[str]) -> int:
    """0 on success; 1 when an adopted record did not read back; 2 when git failed."""
    args = _parser().parse_args(argv)
    project_root: Path = args.project_root or default_project_root()
    if not project_root.is_dir():
        print(f'stack_base_census: {project_root} is not a directory', file=sys.stderr)
        return 2
    statuses = _statuses_or_warn(args.tasks_db or tasks_db_path(project_root))
    try:
        report, outcomes = asyncio.run(_census_and_adopt(
            project_root, args.main_branch, statuses, args.adopt_confirmed,
        ))
    except CensusGitError as exc:
        print(f'stack_base_census: {exc}', file=sys.stderr)
        return 2
    render = render_json if args.json else render_markdown
    print(render(report, outcomes))
    return 0 if all(outcome.written for outcome in outcomes or ()) else 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
