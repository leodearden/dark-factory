"""Is a task's work already on main? Answered rebase-aware, from git alone.

Only POSITIVE evidence yields ``landed=True``: a fresh ``Merge task/<id> into
main`` merge marker when the branch carries no unmerged commits of its own, or
a twin on main (same author date, author email and subject) for EVERY one of
the branch's own commits, which is what a rebase or cherry-pick leaves behind.
A failed probe yields *unknown* (``landed=None``), and unknown is never
``False``.

The decision table, and why patch-id comparison (``git cherry``) was rejected,
are recorded in task 4874's plan design decisions.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import NamedTuple

from shared.git_async import run_git

from fused_memory.services.live_workflow_detector import (
    DEFAULT_BASE_BRANCH,
    DEFAULT_BRANCH_PREFIX,
)

logger = logging.getLogger(__name__)

#: Per-probe timeout. The merge-marker scan walks all of main's history
#: (measured 2.3 s wall over 70k commits on an idle host), so it gets head-room
#: for a loaded one.
_GIT_TIMEOUT_SECONDS = 30.0

_FIELD_SEPARATOR = '\x1f'


class LandingEvidence(StrEnum):
    MERGE_MARKER = 'merge-marker'
    REBASED_TWINS = 'rebased-twins'


@dataclass(frozen=True)
class LandingQuery:
    """One task to probe; *reopened_at* (timezone-aware) discards older evidence."""

    task_id: str
    reopened_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.reopened_at is not None and self.reopened_at.tzinfo is None:
            raise ValueError(f'reopened_at must be timezone-aware, got {self.reopened_at!r}')


@dataclass(frozen=True)
class LandingVerdict:
    """``landed`` is True (with evidence), False, or None for unknown."""

    landed: bool | None
    evidence: LandingEvidence | None
    commit: str | None

    @classmethod
    def unknown(cls) -> LandingVerdict:
        return cls(landed=None, evidence=None, commit=None)

    @classmethod
    def not_landed(cls) -> LandingVerdict:
        return cls(landed=False, evidence=None, commit=None)

    @classmethod
    def landed_by(cls, evidence: LandingEvidence, commit: str | None = None) -> LandingVerdict:
        return cls(landed=True, evidence=evidence, commit=commit)


class CommitFingerprint(NamedTuple):
    """What a rebase or cherry-pick preserves, so a twin on main can be matched."""

    author_epoch: int
    author_email: str
    subject: str


class MarkerHit(NamedTuple):
    sha: str
    committed_at: datetime


def _merge_subject(branch: str, main: str) -> str:
    """Mirror of ``orchestrator/src/orchestrator/git_ops.py::_merge_subject``.

    fused-memory cannot import orchestrator, so the format is mirrored here and
    every marker spelling below is derived from this one function.
    """
    return f'Merge {branch} into {main}'


_BRANCH_SENTINEL = '\x00BRANCH\x00'
_SUBJECT_PREFIX, _, _SUBJECT_SUFFIX = _merge_subject(_BRANCH_SENTINEL, DEFAULT_BASE_BRANCH).partition(
    _BRANCH_SENTINEL
)
_MARKER_GREP = _SUBJECT_PREFIX + DEFAULT_BRANCH_PREFIX
_MARKER_SUBJECT = re.compile(
    r'\A' + re.escape(_SUBJECT_PREFIX) + r'(\S+)' + re.escape(_SUBJECT_SUFFIX) + r'\Z'
)


def _branch_for(task_id: str) -> str:
    return f'{DEFAULT_BRANCH_PREFIX}{task_id}'


async def _git_stdout(project_root: str, probe: str, *args: str) -> str | None:
    """Run one probe; on spawn error, non-zero rc or timeout log once at WARNING and return None."""
    try:
        result = await run_git(['git', '-C', project_root, *args], timeout=_GIT_TIMEOUT_SECONDS)
    except OSError as exc:
        logger.warning(
            'landed_on_main.probe_unavailable: %s could not spawn for %s (%s) — landing is unknown',
            probe, project_root, exc,
        )
        return None
    if not result.ok:
        logger.warning(
            'landed_on_main.probe_unavailable: %s %s for %s (rc=%d): %s — landing is unknown',
            probe, 'TIMED OUT' if result.timed_out else 'failed',
            project_root, result.returncode, result.stderr,
        )
        return None
    return result.stdout


def _records(stdout: str) -> Iterable[list[str]]:
    return (line.split(_FIELD_SEPARATOR) for line in stdout.splitlines() if line)


async def _task_branches(project_root: str) -> frozenset[str] | None:
    stdout = await _git_stdout(
        project_root, 'branch listing',
        'for-each-ref', '--format=%(refname:lstrip=2)', f'refs/heads/{DEFAULT_BRANCH_PREFIX}',
    )
    if stdout is None:
        return None
    return frozenset(line for line in stdout.splitlines() if line)


async def _merge_marker_index(project_root: str) -> dict[str, MarkerHit] | None:
    """Map each merged branch to its NEWEST marker; only an exact SUBJECT counts."""
    stdout = await _git_stdout(
        project_root, 'merge-marker scan',
        'log', DEFAULT_BASE_BRANCH, '--fixed-strings', f'--grep={_MARKER_GREP}',
        '--format=%H%x1f%ct%x1f%s', '--',
    )
    if stdout is None:
        return None
    index: dict[str, MarkerHit] = {}
    for sha, committed_epoch, subject in _records(stdout):
        match = _MARKER_SUBJECT.match(subject)
        if match:
            committed_at = datetime.fromtimestamp(int(committed_epoch), tz=UTC)
            index.setdefault(match.group(1), MarkerHit(sha, committed_at))
    return index


async def _own_commits(project_root: str, branch: str) -> tuple[CommitFingerprint, ...] | None:
    stdout = await _git_stdout(
        project_root, f'own-commit scan of {branch}',
        'log', f'{DEFAULT_BASE_BRANCH}..{branch}', '--format=%at%x1f%ae%x1f%s', '--',
    )
    if stdout is None:
        return None
    return tuple(
        CommitFingerprint(int(author_epoch), email, subject)
        for author_epoch, email, subject in _records(stdout)
    )


async def _main_twins(
    project_root: str, own_commits: Iterable[tuple[CommitFingerprint, ...] | None],
) -> dict[CommitFingerprint, int] | None:
    """Map each main commit's fingerprint to its newest committer epoch.

    One scan covers every probed branch: ``--since`` filters by committer date,
    which for a rebased or cherry-picked twin is never before its author date.
    """
    author_epochs = [fp.author_epoch for commits in own_commits if commits for fp in commits]
    if not author_epochs:
        return {}
    stdout = await _git_stdout(
        project_root, 'main twin scan',
        'log', DEFAULT_BASE_BRANCH, f'--since=@{min(author_epochs)}',
        '--format=%at%x1f%ae%x1f%s%x1f%ct', '--',
    )
    if stdout is None:
        return None
    twins: dict[CommitFingerprint, int] = {}
    for author_epoch, email, subject, committed_epoch in _records(stdout):
        fingerprint = CommitFingerprint(int(author_epoch), email, subject)
        twins[fingerprint] = max(int(committed_epoch), twins.get(fingerprint, 0))
    return twins


def _is_fresh(committed_at: datetime, reopened_at: datetime | None) -> bool:
    return reopened_at is None or committed_at >= reopened_at


def _classify_by_marker(
    branch: str, markers: Mapping[str, MarkerHit] | None, reopened_at: datetime | None,
) -> LandingVerdict:
    if markers is None:
        return LandingVerdict.unknown()
    hit = markers.get(branch)
    if hit is None or not _is_fresh(hit.committed_at, reopened_at):
        return LandingVerdict.not_landed()
    return LandingVerdict.landed_by(LandingEvidence.MERGE_MARKER, hit.sha)


def _classify_by_twins(
    own: tuple[CommitFingerprint, ...],
    twins: Mapping[CommitFingerprint, int] | None,
    reopened_at: datetime | None,
) -> LandingVerdict:
    if twins is None:
        return LandingVerdict.unknown()
    for fingerprint in own:
        committed_epoch = twins.get(fingerprint)
        if committed_epoch is None:
            return LandingVerdict.not_landed()
        if not _is_fresh(datetime.fromtimestamp(committed_epoch, tz=UTC), reopened_at):
            return LandingVerdict.not_landed()
    return LandingVerdict.landed_by(LandingEvidence.REBASED_TWINS)


def _classify(
    query: LandingQuery,
    branches: frozenset[str] | None,
    markers: Mapping[str, MarkerHit] | None,
    own_commits: Mapping[str, tuple[CommitFingerprint, ...] | None],
    twins: Mapping[CommitFingerprint, int] | None,
) -> LandingVerdict:
    """Apply the decision table: own commits need twins, otherwise a fresh marker."""
    if branches is None:
        return LandingVerdict.unknown()
    branch = _branch_for(query.task_id)
    if branch in branches:
        own = own_commits.get(branch)
        if own is None:
            return LandingVerdict.unknown()
        if own:
            return _classify_by_twins(own, twins, query.reopened_at)
    return _classify_by_marker(branch, markers, query.reopened_at)


async def probe_landing(
    project_root: str, queries: Sequence[LandingQuery],
) -> Mapping[str, LandingVerdict]:
    """Return a verdict per queried task id. Git failures yield unknown and never raise."""
    if not queries:
        return {}
    branches, markers = await asyncio.gather(
        _task_branches(project_root), _merge_marker_index(project_root),
    )
    existing = sorted({_branch_for(q.task_id) for q in queries} & (branches or frozenset()))
    own_results = await asyncio.gather(*(_own_commits(project_root, b) for b in existing))
    own_commits = dict(zip(existing, own_results, strict=True))
    twins = await _main_twins(project_root, own_commits.values())
    return {
        query.task_id: _classify(query, branches, markers, own_commits, twins)
        for query in queries
    }
