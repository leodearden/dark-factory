"""Which paths a merged branch created and deleted before it merged."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from shared.git_async import run_git

logger = logging.getLogger(__name__)

_GIT_TIMEOUT_SECONDS = 30.0

_RECORD_SEPARATOR = '\x1e'
_COMMIT_RECORD_FORMAT = '--format=%x1e%H'
_NUL = '\x00'


class ProbeOutcome(StrEnum):
    ANALYZED = 'analyzed'
    NOT_A_MERGE = 'not_a_merge'
    UNKNOWN = 'unknown'


@dataclass(frozen=True)
class UnlandedPath:
    """A path the branch's own commits touched (full shas, oldest first) that did not land."""

    path: str
    branch_commits: tuple[str, ...]


@dataclass(frozen=True)
class UnlandedPathsProbe:
    outcome: ProbeOutcome
    merge_commit: str
    unlanded: tuple[UnlandedPath, ...] = ()

    @classmethod
    def unknown(cls, merge_commit: str) -> UnlandedPathsProbe:
        return cls(outcome=ProbeOutcome.UNKNOWN, merge_commit=merge_commit)

    @classmethod
    def not_a_merge(cls, merge_commit: str) -> UnlandedPathsProbe:
        return cls(outcome=ProbeOutcome.NOT_A_MERGE, merge_commit=merge_commit)


async def _git_stdout(project_root: str, probe: str, *args: str) -> str | None:
    """Run one probe; on spawn error, non-zero rc or timeout log once at WARNING and return None."""
    try:
        result = await run_git(['git', '-C', project_root, *args], timeout=_GIT_TIMEOUT_SECONDS)
    except OSError as exc:
        logger.warning(
            'unlanded_branch_paths.probe_unavailable: %s could not spawn for %s (%s)',
            probe, project_root, exc,
        )
        return None
    if not result.ok:
        logger.warning(
            'unlanded_branch_paths.probe_unavailable: %s %s for %s (rc=%d): %s',
            probe, 'TIMED OUT' if result.timed_out else 'failed',
            project_root, result.returncode, result.stderr,
        )
        return None
    return result.stdout


def _nul_separated(stdout: str) -> frozenset[str]:
    return frozenset(path for path in stdout.split(_NUL) if path)


def _touches_by_path(log_stdout: str) -> Mapping[str, tuple[str, ...]]:
    """Map each path to the commits that touched it, oldest first (git log lists newest first)."""
    newest_first: dict[str, list[str]] = {}
    for record in log_stdout.split(_RECORD_SEPARATOR):
        sha, _, names = record.partition(_NUL)
        for path in names.removeprefix('\n').split(_NUL):
            if path:
                newest_first.setdefault(path, []).append(sha)
    return {path: tuple(reversed(shas)) for path, shas in newest_first.items()}


async def probe_unlanded_paths(project_root: str, merge_commit: str) -> UnlandedPathsProbe:
    """Paths the merge's second-parent branch touched in its own commits that are absent at the merge.

    A path is unlanded iff a branch-own commit touched it, it is absent from
    the net diff first-parent..merge, and it is absent from the merge's tree.
    Git failure yields an UNKNOWN probe; this never raises for it.
    """
    parents = await _git_stdout(
        project_root, 'parent lookup', 'rev-list', '--parents', '-n', '1', merge_commit, '--',
    )
    if parents is None:
        return UnlandedPathsProbe.unknown(merge_commit)
    tokens = parents.split()
    if len(tokens) < 3:
        return UnlandedPathsProbe.not_a_merge(merge_commit)
    merge, first_parent, second_parent = tokens[0], tokens[1], tokens[2]

    branch_log = await _git_stdout(
        project_root, 'branch log',
        'log', '-z', '--no-merges', '--no-renames', '--name-only', _COMMIT_RECORD_FORMAT,
        f'{first_parent}..{second_parent}', '--',
    )
    if branch_log is None:
        return UnlandedPathsProbe.unknown(merge_commit)
    touched = _touches_by_path(branch_log)
    if not touched:
        return UnlandedPathsProbe(outcome=ProbeOutcome.ANALYZED, merge_commit=merge_commit)

    net_diff = await _git_stdout(
        project_root, 'net diff', 'diff', '-z', '--name-only', '--no-renames', first_parent, merge, '--',
    )
    if net_diff is None:
        return UnlandedPathsProbe.unknown(merge_commit)
    candidates = touched.keys() - _nul_separated(net_diff)
    if not candidates:
        return UnlandedPathsProbe(outcome=ProbeOutcome.ANALYZED, merge_commit=merge_commit)

    tree = await _git_stdout(project_root, 'merge tree', 'ls-tree', '-r', '-z', '--name-only', merge)
    if tree is None:
        return UnlandedPathsProbe.unknown(merge_commit)
    unlanded = sorted(candidates - _nul_separated(tree))
    return UnlandedPathsProbe(
        outcome=ProbeOutcome.ANALYZED,
        merge_commit=merge_commit,
        unlanded=tuple(UnlandedPath(path=path, branch_commits=touched[path]) for path in unlanded),
    )
