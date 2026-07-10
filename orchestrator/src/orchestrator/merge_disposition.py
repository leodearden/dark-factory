"""Merge-failure disposition classifier — mechanism M1 of the merge-skew
attribution PRD (plans/merge-skew-attribution-prd.md, task 2381 α).

``classify_merge_failure_disposition`` refines the caller-computed
``preexisting`` bucket (see ``verify_failure_is_preexisting_on_main`` in
:mod:`orchestrator.verify`) into a :class:`MergeFailureDisposition`:

  MAIN_RED         — the failure is already on main (preexisting=True).
  INTEGRATION_SKEW — the branch verified green pre-merge, but a landing on
                      main between the branch's merge-base and main's tip
                      touched a file implicated by the branch's failing
                      tests — i.e. the branch was correct against a main
                      that has since moved.
  BRANCH_BUG       — no landing on main is implicated; the failure is the
                      branch's own.
  INDETERMINATE    — the honest fallback: evidence is inconclusive, or an
                      internal error occurred.

Invariants (see the task-2381 plan / merge-skew-attribution-prd.md):
  I1 — never re-probes ``verify_failure_is_preexisting_on_main``; takes
       ``preexisting`` as an input and never calls the probe itself.
  I2 — read-only: only issues ``git log`` (no worktree add/mutation, no
       verify run). The repo HEAD/working-tree is never touched.
  I3 — fail-open: any internal error degrades to (INDETERMINATE, None),
       logged at WARNING, never propagated.
  I5 — INTEGRATION_SKEW requires a *positively confirmed* pre-merge green
       verify for the branch, sourced from EventType.workflow_verify event-
       store history keyed by task_id (the branch's OWN pre-merge verdict,
       branch-vs-its-merge-base — NOT EventType.merge_verify, which is the
       merge worker's POST-rebase verdict: a passing merge_verify means the
       merge SUCCEEDED, and a first-attempt skew has no prior passing
       merge_verify row at all). Implicated landings without a confirmed
       workflow_verify green degrade to INDETERMINATE, not INTEGRATION_SKEW.
       An ABSENT workflow_verify (no rows) is honest, not a no-op: the
       orchestrator workflow is the most common but not the only path to the
       merge queue (/merge-queue, /unblock, /do submit branches and emit no
       workflow_verify), so absent-green -> INDETERMINATE is byte-identical
       to today's behaviour. Task 2381 α owns this member + reader; task
       2383 β emits the event at the workflow VERIFY-pass site.

This module is intentionally self-contained (task 2381 α scope): it defines
the classifier, its data types, and private helpers only. Wiring into the
merge gate/surfaces (β) and runs.db event emission (γ) are separate tasks
that depend on this module's stable contract.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from orchestrator.event_store import EventType
from orchestrator.git_ops import _run

if TYPE_CHECKING:
    from collections.abc import Iterable

    from orchestrator.event_store import EventStore
    from orchestrator.verify import VerifyResult

logger = logging.getLogger(__name__)


class MergeFailureDisposition(StrEnum):
    """Refined attribution of a post-merge verify failure."""

    MAIN_RED = 'main_red'
    INTEGRATION_SKEW = 'integration_skew'
    BRANCH_BUG = 'branch_bug'
    INDETERMINATE = 'indeterminate'


@dataclass(frozen=True)
class SkewEvidence:
    """Evidence bundle attached to an INTEGRATION_SKEW disposition.

    ``implicated_commits`` — SHAs of commits landed on main (within
    ``merge_base_sha..main_sha``) that touched a candidate file.
    ``failing_tests`` — failing test identifiers parsed from the branch's
    VerifyResult.
    ``overlap_files`` — candidate files that are both implicated by a
    failing test AND touched by an implicated landing.
    """

    implicated_commits: tuple[str, ...]
    failing_tests: tuple[str, ...]
    overlap_files: tuple[str, ...]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

# pytest-style: ``FAILED tests/test_foo.py::test_bar - AssertionError`` — the
# failing-test id is the first whitespace-delimited token after "FAILED ".
_PYTEST_FAILED_ID_RE = re.compile(r'^FAILED\s+(\S+)', re.MULTILINE)

# Rust test-runner style: ``test crate::mod::test_x ... FAILED`` — captures
# the ``crate::mod::test_x`` identifier (a ``::``-separated path).
_RUST_FAILED_ID_RE = re.compile(
    r'^(?:test\s+)?([A-Za-z_]\w*(?:::[A-Za-z_]\w*)+)(?:\s+\.\.\.)?\s+FAILED\s*$',
    re.MULTILINE,
)

# File-shaped token: a path (word chars / slashes / hyphens) ending in a
# recognised source or test-file extension. Deliberately conservative — the
# extension list is fixed and the character class excludes ``.``/``:`` so a
# match never swallows a pytest ``::``-separated test id past the file path.
_FILE_TOKEN_RE = re.compile(
    r'[\w][\w/-]*\.(?:py|rs|ts|tsx|js|jsx|go|rb|sh|java|kt|cpp|cc|c|h|hpp)\b',
)


def _extract_failing_tests_and_candidate_files(
    verify_result: VerifyResult,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Best-effort extraction of failing-test ids and candidate files from a
    VerifyResult's free-text ``test_output``/``cause_hint`` (Open Q1:
    VerifyResult carries no structured per-test list).

    Biased toward an empty candidate set on ambiguous/uninformative input —
    an uncertain mapping must degrade to no-implicated-landings
    (BRANCH_BUG/INDETERMINATE), never manufacture a false INTEGRATION_SKEW.
    """
    combined = '\n'.join(
        part for part in (verify_result.cause_hint, verify_result.test_output) if part
    )
    if not combined or not combined.strip():
        return (), ()

    failing_tests: list[str] = []

    def _add_test(test_id: str) -> None:
        if test_id and test_id not in failing_tests:
            failing_tests.append(test_id)

    for m in _PYTEST_FAILED_ID_RE.finditer(combined):
        _add_test(m.group(1))
    for m in _RUST_FAILED_ID_RE.finditer(combined):
        _add_test(m.group(1))

    candidate_files: set[str] = set(_FILE_TOKEN_RE.findall(combined))

    # test-id -> path heuristic: a pytest-style id's path segment (before
    # "::") is itself a candidate file, even when not independently caught
    # by the file-token scan above.
    for test_id in failing_tests:
        path_part = test_id.split('::', 1)[0]
        if _FILE_TOKEN_RE.fullmatch(path_part):
            candidate_files.add(path_part)

    return tuple(failing_tests), tuple(sorted(candidate_files))


# Full 40-hex-char commit SHA, as emitted by ``git log --format=%H``.
_FULL_SHA_RE = re.compile(r'^[0-9a-f]{40}$')


async def _implicated_landings(
    repo_root: Path,
    merge_base_sha: str,
    main_sha: str,
    candidate_files: Iterable[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Read-only ``git log --name-only`` over ``merge_base_sha..main_sha``,
    restricted to *candidate_files*, to find landings on main that touched a
    file implicated by the branch's failing tests.

    I2 (read-only): issues only ``git log`` — no worktree add/mutation, no
    verify run. 2357: ``merge_base_sha``/``main_sha`` are consumed exactly as
    given (authoritative dispatch-time inputs) and never re-derived here.

    Returns ``(implicated_commits, overlap_files)`` — empty tuples when
    *candidate_files* is empty, on any git error (non-zero exit, missing/
    non-git *repo_root*), or on any exception (fail-safe: never raises).
    """
    candidate_files = tuple(candidate_files)
    if not candidate_files:
        return (), ()

    try:
        rc, stdout, _stderr = await _run(
            [
                'git', 'log', '--name-only', '--format=%H',
                f'{merge_base_sha}..{main_sha}', '--', *candidate_files,
            ],
            cwd=repo_root,
        )
        if rc != 0:
            return (), ()

        candidate_set = set(candidate_files)
        implicated_commits: list[str] = []
        overlap_files: set[str] = set()
        for raw_line in stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if _FULL_SHA_RE.fullmatch(line):
                if line not in implicated_commits:
                    implicated_commits.append(line)
                continue
            if line in candidate_set:
                overlap_files.add(line)

        return tuple(implicated_commits), tuple(sorted(overlap_files))
    except Exception:
        logger.warning(
            '_implicated_landings: git log failed for repo_root=%s '
            '(merge_base_sha=%s, main_sha=%s); degrading to no implicated '
            'landings (fail-safe)',
            repo_root, merge_base_sha, main_sha,
            exc_info=True,
        )
        return (), ()


def _branch_pre_merge_verify_green(
    event_store: EventStore | None,
    task_id: str | None,
) -> bool | None:
    """Read the branch's OWN pre-merge verify verdict (the I5 green fact) from
    ``EventType.workflow_verify`` event-store history, keyed by ``task_id``.

    Returns:
        ``True``  — at least one workflow_verify row for this task_id has a
                    truthy ``data.passed`` (any-prior-green, NOT
                    most-recent-wins: a later failed re-verify does not erase
                    an earlier confirmed green).
        ``False`` — workflow_verify rows exist for this task_id but none passed.
        ``None``  — no workflow_verify rows for this task_id (evidence ABSENT
                    -> the caller degrades to INDETERMINATE), or ``event_store``
                    / ``task_id`` is None, or any read error (fail-safe: never
                    raises).

    Source note (I5, task-2381 design amendment): the green fact is the
    workflow VERIFY phase's branch-vs-merge-base verdict, NOT the merge
    worker's post-rebase ``EventType.merge_verify`` (a *passing* merge_verify
    means the merge already succeeded; a first-attempt skew has no prior
    passing merge_verify row at all, so sourcing I5 from it would make
    INTEGRATION_SKEW never fire on the exact case it targets).
    """
    if event_store is None or task_id is None:
        return None
    try:
        rows = event_store.fetch_events_by_type(EventType.workflow_verify)
        matched = [row for row in rows if row.get('task_id') == task_id]
        if not matched:
            return None
        return any(bool((row.get('data') or {}).get('passed')) for row in matched)
    except Exception:
        logger.warning(
            '_branch_pre_merge_verify_green: event-store read failed for '
            'task_id=%s; degrading to None (evidence absent, fail-safe)',
            task_id,
            exc_info=True,
        )
        return None


async def classify_merge_failure_disposition(
    *,
    verify_result: VerifyResult,
    branch: str,
    merge_base_sha: str,
    main_sha: str,
    preexisting: bool,
    task_id: str | None = None,
    repo_root: Path | None = None,
    event_store: EventStore | None = None,
) -> tuple[MergeFailureDisposition, SkewEvidence | None]:
    """Classify a merge-verify failure's disposition (git-only, read-only, fail-open).

    Args:
        verify_result: the failing VerifyResult from the branch's merge-time verify.
        branch: the task branch name (informational; not used for git plumbing).
        merge_base_sha: merge-base(branch, main) at dispatch time (2357 constraint:
            authoritative caller-supplied input, never re-derived here).
        main_sha: main's tip SHA at dispatch time (same constraint as above).
        preexisting: caller-computed result of
            ``verify_failure_is_preexisting_on_main`` (I1: never re-probed here).
        task_id: scheduler task id, used to key the I5 event-store green lookup.
            None degrades I5 to indeterminate (fail-open).
        repo_root: git repository root for read-only ``git log`` plumbing. None
            degrades the git-dependent path to indeterminate (fail-open).
        event_store: EventStore to read prior EventType.workflow_verify rows
            from for the I5 branch-green fact (the branch's own pre-merge
            verdict). None degrades I5 to indeterminate (fail-open).

    Returns:
        ``(disposition, evidence)`` — ``evidence`` is a :class:`SkewEvidence`
        iff ``disposition is MergeFailureDisposition.INTEGRATION_SKEW``, else
        ``None``.
    """
    try:
        if preexisting:
            # [boundary row 1] I1: refine the caller-computed bucket only —
            # never re-probe verify_failure_is_preexisting_on_main.
            return (MergeFailureDisposition.MAIN_RED, None)

        # Extract the branch's failing-test ids and the source/test files they
        # implicate (Open Q1: VerifyResult carries no structured per-test list,
        # so this is a best-effort parse biased toward an empty candidate set).
        failing_tests, candidate_files = _extract_failing_tests_and_candidate_files(
            verify_result,
        )

        # Ambiguity clause: no candidate file could be identified from the
        # failure, or there is no repo to search -> evidence is unavailable ->
        # INDETERMINATE. Never guess BRANCH_BUG from an empty extraction.
        if not candidate_files or repo_root is None:
            return (MergeFailureDisposition.INDETERMINATE, None)

        # Map candidate files to landings on main between the branch's merge-base
        # and main's tip (I2 read-only git log; 2357: both SHAs are the
        # authoritative dispatch-time inputs, consumed as-given, never
        # re-derived here).
        implicated_commits, overlap_files = await _implicated_landings(
            repo_root, merge_base_sha, main_sha, candidate_files,
        )

        # Searched, and nothing on main touching a failing file is implicated
        # -> the failure is the branch's own. (workflow_verify presence is
        # irrelevant here: with nothing on main to blame, a recorded green
        # cannot turn a branch bug into a skew.)
        if not implicated_commits:
            return (MergeFailureDisposition.BRANCH_BUG, None)

        # A landing IS implicated. I5: call it INTEGRATION_SKEW only when the
        # branch is *positively confirmed* green pre-merge (workflow_verify).
        # Absent green (None, e.g. a first-attempt skew or a non-orchestrator
        # submit path) or a failed green (False) degrades to the honest
        # INDETERMINATE — never a fabricated skew.
        if _branch_pre_merge_verify_green(event_store, task_id) is True:
            return (
                MergeFailureDisposition.INTEGRATION_SKEW,
                SkewEvidence(
                    implicated_commits=implicated_commits,
                    failing_tests=failing_tests,
                    overlap_files=overlap_files,
                ),
            )
        return (MergeFailureDisposition.INDETERMINATE, None)
    except Exception:
        logger.warning(
            'classify_merge_failure_disposition: internal error; degrading to '
            'INDETERMINATE (fail-open, I3)',
            exc_info=True,
        )
        return (MergeFailureDisposition.INDETERMINATE, None)
