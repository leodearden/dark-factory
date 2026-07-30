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
       The workflow_verify read is run-AGNOSTIC (durable across restarts,
       via EventStore.fetch_events_by_type_all_runs): task 2752's verify
       checkpoint SUPPRESSES the current-run re-emit at an unchanged branch
       tip, so the branch's only green may live under a prior run_id — a
       run-scoped read would then miss it and lose the INTEGRATION_SKEW
       classification after a fleet redeploy. Reading across runs keeps the
       durable prior-run green visible (any-prior-green semantics unchanged;
       result set bounded by the task_id filter).
  I7 — INTEGRATION_SKEW additionally requires at least one failing-test id
       carrying a real test-node SHAPE — ``path::test_name`` for pytest,
       ``crate::mod::test`` for Rust (task 2871 introduced the non-empty
       requirement; task 3178 added the shape floor). A guard/ratchet failure
       whose only ``FAILED`` token is a bare filename parses zero node-shaped
       ids and degrades to INDETERMINATE. See the I7 incident account below.

THE I7 INCIDENT (task 3178) — this is the ONE authoritative account; every
other site in the codebase that touches this behaviour points here rather than
restating it, because six copies of an incident narrative is how the false
premise below survived two task cycles in the first place.

  What went wrong. ``_PYTEST_FAILED_ID_RE`` was an unconstrained
  ``^FAILED\\s+(\\S+)``, and reify renders a shell-guard trip as ``FAILED
  <guard>.sh``. The guard's own FILENAME was therefore returned as a
  "failing-test id" and satisfied task 2871's non-empty I7 requirement
  VACUOUSLY. The same hole parsed the English word "to" out of "FAILED to
  release semaphore slot".

  Blast radius. All 8 INTEGRATION_SKEW dispositions between 2026-07-24 and
  2026-07-28 were shell/infra guards, not skews: reify 5316/5373/5302
  (test_harness_kloc_cap.sh), 5300 x3 (test_verify_scope.sh), 5566 x2
  (test_reify_audit_ptodo.sh), 5321
  (test_deterministic_gate_closure_staleness_sweep.sh). Each told a debugger
  to "port the landed commit — do not hunt your own diff" about a failure
  that was not a skew. reify 5187 (a Rust-shaped id) was the ONE genuine skew
  in that population and is provably unaffected: ``_RUST_FAILED_ID_RE`` has
  always required ``::`` by construction, so the shape floor only mirrors an
  invariant the sibling regex already had.

  Why it survived. Task 2871's comment asserted guard output "matches neither
  ``_PYTEST_FAILED_ID_RE`` nor ``_RUST_FAILED_ID_RE``" — FALSE in production.
  Its test passed only because the fixture was invented (``HARNESS_KLOC_CAP
  FAIL …``, which has no ``^FAILED <token>`` and so genuinely parses zero
  ids), and ``merge_attempt`` rows persisted only ``{disposition, outcome}``,
  so nobody could read the evidence back out of runs.db to check. Task 2918
  then built on the same premise. The two fixes are therefore paired: the
  shape floor in ``_extract_failing_tests_and_candidate_files``, and the
  evidence now persisted on BOTH the promoted and the degraded path (the
  ``observed_evidence`` slot of :class:`ClassificationResult`) so the next
  false premise is checkable. Every regression fixture is now a VERBATIM
  captured production string — see ``TestShellGuardFilenameIsNotATestId``.

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
from typing import TYPE_CHECKING, NamedTuple

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


class ClassificationResult(NamedTuple):
    """What ``classify_merge_failure_disposition`` returns.

    A NamedTuple, so every existing positional unpack keeps working while call
    sites can read the two evidence slots BY NAME. That matters here because
    the slots differ only in their gate, not their type, and picking the wrong
    one silently re-creates the bug this module was fixed for.

    ``evidence`` — the ADJUDICATED attribution: non-``None`` **iff**
    ``disposition is INTEGRATION_SKEW``. This is what
    ``_render_skew_surfaces`` turns into the "port the landed commit … do not
    hunt your own diff" directive, so its non-``None``-ness must keep meaning
    "this IS a skew".

    ``observed_evidence`` — the bundle GATHERED (task 3178): non-``None``
    whenever implicated landings were found, REGARDLESS of verdict. On the
    promoted path it is the same object as ``evidence``; on an I7/I5 DEGRADE it
    is the evidence the gate refused to promote — which is what makes the
    degrade measurable. ``None`` when no landings were implicated, when
    classification could not proceed (no candidate files / no repo), and on the
    fail-open path.

    Explicitly NOT a skew verdict: read ``disposition`` for that. The
    adjudicated slot is deliberately kept narrow — "a non-empty evidence field
    means this is a skew" is the inference that let the false I7 premise
    survive two task cycles (see the module docstring's I7 account).
    """

    disposition: MergeFailureDisposition
    evidence: SkewEvidence | None
    observed_evidence: SkewEvidence | None


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
        tid = m.group(1)
        if '::' not in tid:
            # Node-id SHAPE floor — see the module docstring's I7 incident
            # account (task 3178). ACCEPTED edge case: a pytest COLLECTION-level
            # ``FAILED tests/test_foo.py`` (no ``::``) now parses zero ids and
            # degrades to INDETERMINATE rather than skew. pytest emits ``ERROR
            # <path>`` for collection errors, so this shape is rare, and the
            # degrade errs toward the honest fallback, not a fabricated skew.
            continue
        _add_test(tid)
    for m in _RUST_FAILED_ID_RE.finditer(combined):
        # No shape floor needed: _RUST_FAILED_ID_RE requires a ``::``-separated
        # path by construction (see its definition above).
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


# Bound on how many SHAs / paths the degrade WARNING spells out (task 3178).
# reify 5566 attempt-2 cited 22 SHAs touching 7 files; an unbounded list would
# make the log line unreadable. The true count is logged alongside EVERY slice,
# so the truncation is never silent.
#
# Deliberately SMALLER than merge_queue._MAX_EVENT_EVIDENCE_ITEMS (10), which
# bounds the same bundle on the runs.db merge_attempt row: this cap is tuned for
# one-line log readability (a human greps it), that one for row size (a census
# queries it, and wants more of the citation). A reader comparing a log line
# against its row will therefore see two different truncation points — that is
# intended, not drift.
_MAX_LOGGED_EVIDENCE_ITEMS = 5

# Full 40-hex-char commit SHA, as emitted by ``git log --format=%H``.
_FULL_SHA_RE = re.compile(r'^[0-9a-f]{40}$')


async def _is_commit_ancestor_of(
    repo_root: Path,
    commit: str,
    ancestor_ref: str,
) -> bool | None:
    """Read-only, fail-safe ``git merge-base --is-ancestor <commit> <ancestor_ref>``.

    Returns:
        ``True``  — rc 0: *commit* is an ancestor of *ancestor_ref* (both
                    objects resolved).
        ``False`` — rc 1 ONLY: *commit* is DEFINITIVELY not an ancestor of
                    *ancestor_ref* (both objects resolved, no ancestry).
        ``None``  — any other rc (e.g. 128 = an object could not be resolved,
                    such as an unresolvable/sentinel *ancestor_ref*), or any
                    exception. Fail-open: an uncertain result is never read as
                    a definitive negative, so it can never cause false pruning;
                    never raises.

    I2 (read-only): issues only ``git merge-base --is-ancestor`` — no worktree
    add/mutation. I3 (fail-open): degrades to None on any uncertainty.
    """
    try:
        rc, _stdout, _stderr = await _run(
            ['git', 'merge-base', '--is-ancestor', commit, ancestor_ref],
            cwd=repo_root,
        )
    except Exception:
        logger.warning(
            '_is_commit_ancestor_of: git merge-base --is-ancestor raised for '
            'repo_root=%s (commit=%s, ancestor_ref=%s); degrading to None '
            '(fail-safe)',
            repo_root, commit, ancestor_ref,
            exc_info=True,
        )
        return None
    if rc == 0:
        return True
    if rc == 1:
        return False
    return None


async def _implicated_landings(
    repo_root: Path,
    merge_base_sha: str,
    main_sha: str,
    candidate_files: Iterable[str],
    real_main_head_sha: str | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Read-only ``git log --name-only`` over ``merge_base_sha..main_sha``,
    restricted to *candidate_files*, to find landings on main that touched a
    file implicated by the branch's failing tests.

    I2 (read-only): issues only ``git log`` / ``git merge-base --is-ancestor``
    — no worktree add/mutation, no verify run. 2357: ``merge_base_sha``/
    ``main_sha`` are consumed exactly as given (authoritative dispatch-time
    inputs) and never re-derived here.

    Orphan/ancestor discriminator (task 2869, reify esc-5260-8): *main_sha* is
    the frozen dispatch-time base (``item.base_sha``), which may be a
    SPECULATIVE/coalesced merge-queue train tip that never fast-forwarded onto
    real main. Walking ``merge_base_sha..main_sha`` then cites the ORPHANED
    speculative-train commits — none of which are ancestors of the real
    published main HEAD. When *real_main_head_sha* is supplied AND differs from
    *main_sha*, each cited commit is filtered to ancestors of real main HEAD:
    an orphaned speculative commit is DEFINITIVELY not an ancestor and is
    pruned, while a genuine landing (still reachable from real main even after
    main advances) survives. Pruning happens ONLY on a definitive
    not-an-ancestor (``_is_commit_ancestor_of`` returns ``False``, i.e.
    ``git merge-base --is-ancestor`` rc 1); a ``True`` or ``None`` (an
    unresolvable/sentinel ref or a transient git error) KEEPS the commit
    (fail-open: never falsely prune, never suppress a genuine INTEGRATION_SKEW).
    When *real_main_head_sha* is ``None`` or ``== main_sha``, every cited
    commit is kept — byte-identical to the pre-2869 reference frame.

    This discriminates by commit SHA and so assumes a genuine landing keeps its
    SHA on real main (the lane CAS-advances main to the exact rebased-onto-main
    train tip via ``git_ops.advance_main``); a landing that re-lands under a NEW
    SHA (a CAS-retry re-rebase), or a real-main read lagging a just-landed
    commit, is conservatively pruned to BRANCH_BUG and never re-cited — the
    opposite failure mode from the false-INTEGRATION_SKEW this filter targets.
    See the SHA-identity note at the filter site below for the full rationale.

    Returns ``(implicated_commits, overlap_files)`` — empty tuples when
    *candidate_files* is empty, on any git error (non-zero exit, missing/
    non-git *repo_root*), or on any exception (fail-safe: never raises).
    ``overlap_files`` is rebuilt from the SURVIVING commits only, so the
    evidence stays honest after any pruning.
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
        # Parse the git-log output into ORDERED per-commit groups: each %H SHA
        # line opens a new group; subsequent candidate-file lines accumulate
        # into that group's touched-files set.
        groups: list[tuple[str, set[str]]] = []
        current: tuple[str, set[str]] | None = None
        for raw_line in stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if _FULL_SHA_RE.fullmatch(line):
                current = (line, set())
                groups.append(current)
                continue
            if line in candidate_set and current is not None:
                current[1].add(line)

        # Ancestor filter (task 2869): prune orphaned speculative-train commits
        # that are DEFINITIVELY not ancestors of the real main HEAD. Applied
        # only when a distinct real main HEAD is supplied; a None/definitively-
        # negative gate never falsely prunes (fail-open, see docstring).
        #
        # SHA-identity assumption (reviewer_comprehensive robustness note, task
        # 2869): this discriminates by COMMIT SHA, so it relies on a genuinely-
        # landed change keeping the SAME SHA on real main that it carried on the
        # speculative base walked above. The merge lane upholds this on the
        # common path — a speculative train is built by rebasing its members
        # onto CURRENT real main, and ``git_ops.advance_main`` CAS-advances
        # refs/heads/main to that exact pre-computed tip SHA, so member SHAs are
        # preserved verbatim. The lane's OWN landed-detection uses the identical
        # test (``reconcile_landed_row`` -> ``git_ops.is_ancestor(
        # row.advanced_sha, main_sha)``), so this filter matches the lane's
        # definition of "landed" rather than inventing a new one.
        #
        # When the assumption does NOT hold, the fallback is conservative, not
        # the bug this fix targets. If a CAS retry re-rebases a landing to a NEW
        # advanced_sha (real main moved under the train), or ``get_main_sha()``
        # resolves a real-main HEAD lagging a just-landed commit not yet
        # reachable, that genuine landing's speculative SHA is correctly not an
        # ancestor of the resolved real main and is pruned — so classify() falls
        # through to BRANCH_BUG. That is the OPPOSITE failure mode from the
        # original bug: it never fabricates a phantom "port landed commit X"
        # citation for a commit that is not on real main; at worst it under-cites
        # a genuine skew and degrades to the same "nothing implicated -> hunt
        # your own diff" the module already emits, self-healing on the next
        # classification once the advance/read catches up. A rebased-under-a-new-
        # SHA landing is therefore INTENTIONALLY treated as BRANCH_BUG rather
        # than risk re-citing a dangling SHA.
        if real_main_head_sha and real_main_head_sha != main_sha:
            surviving: list[tuple[str, set[str]]] = []
            for sha, files in groups:
                is_ancestor = await _is_commit_ancestor_of(
                    repo_root, sha, real_main_head_sha,
                )
                if is_ancestor is False:
                    continue  # definitively not on real main -> orphan, prune
                surviving.append((sha, files))
            groups = surviving

        # Rebuild implicated_commits (ordered, deduped) and overlap_files
        # (sorted union) from the surviving groups only.
        implicated_commits: list[str] = []
        overlap_files: set[str] = set()
        for sha, files in groups:
            if sha not in implicated_commits:
                implicated_commits.append(sha)
            overlap_files.update(files)

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
    ``EventType.workflow_verify`` event-store history, keyed by ``task_id``,
    reading run-agnostically across ALL runs (durable across restarts).

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

    Cross-run durability (task 2752): the read is via
    ``EventStore.fetch_events_by_type_all_runs(..., task_id=task_id)`` — the
    run-agnostic reader — NOT the run-scoped ``fetch_events_by_type``. This is
    load-bearing: task 2752's verify checkpoint SUPPRESSES the current-run
    workflow_verify re-emit at an unchanged branch tip, so on the cross-restart
    fast-path the branch's only ``workflow_verify(passed=True)`` green lives
    under a PRIOR run_id. A run-scoped read would find no row -> return None ->
    degrade a genuine INTEGRATION_SKEW to INDETERMINATE after a fleet redeploy.
    The any-prior-green, tip-agnostic semantics are unchanged (a single-run
    store is still fully visible to the all-runs reader); visibility is merely
    extended across runs, and the result set stays bounded by the task_id SQL
    filter.

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
        rows = event_store.fetch_events_by_type_all_runs(
            EventType.workflow_verify, task_id=task_id,
        )
        if not rows:
            return None
        return any(bool((row.get('data') or {}).get('passed')) for row in rows)
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
    real_main_head_sha: str | None = None,
    task_id: str | None = None,
    repo_root: Path | None = None,
    event_store: EventStore | None = None,
) -> ClassificationResult:
    """Classify a merge-verify failure's disposition (git-only, read-only, fail-open).

    Invariants (continued from the module docstring):
      I6 — orphan/ancestor discriminator (task 2869, reify esc-5260-8):
           implicated landings are filtered to ancestors of the CURRENT real
           main HEAD (*real_main_head_sha*), so orphaned speculative-train
           commits — the frozen dispatch-time *main_sha* may be a coalesced
           merge-queue TRAIN TIP that never fast-forwarded onto real main, and
           walking ``merge_base_sha..main_sha`` then cites its dangling
           commits — are never cited. Filtering is purely subtractive and
           fail-open: an emptied implicated set falls through the existing
           ``if not implicated_commits: return BRANCH_BUG`` branch (no new
           disposition), while ``real_main_head_sha=None`` (unresolved real
           main) or an unresolvable/sentinel ref keeps today's pre-2869
           reference frame (see ``_implicated_landings``). *main_sha* stays the
           frozen dispatch-time input (2357 untouched); *real_main_head_sha* is
           a distinct, additional input used only for the ancestor filter.
      I7 — INTEGRATION_SKEW additionally requires at least one failing-test id
           carrying a real test-node SHAPE (``path::test_name`` for pytest,
           ``crate::mod::test`` for Rust). A guard/ratchet failure whose only
           ``FAILED`` token is a bare filename parses zero node-shaped ids and
           degrades to INDETERMINATE. See the module docstring's I7 incident
           account for why (task 2871 non-empty floor, task 3178 shape floor).

    Args:
        verify_result: the failing VerifyResult from the branch's merge-time verify.
        branch: the task branch name (informational; not used for git plumbing).
        merge_base_sha: merge-base(branch, main) at dispatch time (2357 constraint:
            authoritative caller-supplied input, never re-derived here).
        main_sha: main's tip SHA at dispatch time (same constraint as above) —
            may be a SPECULATIVE/coalesced merge-queue train tip (item.base_sha).
        preexisting: caller-computed result of
            ``verify_failure_is_preexisting_on_main`` (I1: never re-probed here).
        real_main_head_sha: the CURRENT real published main HEAD (I6). When
            supplied AND != main_sha, implicated landings are filtered to
            ancestors of it (orphaned speculative commits pruned). None (the
            caller could not resolve real main) skips the filter -> today's
            pre-2869 reference frame (fail-open, additive default).
        task_id: scheduler task id, used to key the I5 event-store green lookup.
            None degrades I5 to indeterminate (fail-open).
        repo_root: git repository root for read-only ``git log`` plumbing. None
            degrades the git-dependent path to indeterminate (fail-open).
        event_store: EventStore to read prior EventType.workflow_verify rows
            from for the I5 branch-green fact (the branch's own pre-merge
            verdict). None degrades I5 to indeterminate (fail-open).

    Returns:
        A :class:`ClassificationResult` — see its docstring for the contract
        distinguishing the ADJUDICATED ``evidence`` slot from the GATHERED
        ``observed_evidence`` slot. Unpacks positionally as
        ``(disposition, evidence, observed_evidence)``.
    """
    try:
        if preexisting:
            # [boundary row 1] I1: refine the caller-computed bucket only —
            # never re-probe verify_failure_is_preexisting_on_main.
            return ClassificationResult(MergeFailureDisposition.MAIN_RED, None, None)

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
            return ClassificationResult(MergeFailureDisposition.INDETERMINATE, None, None)

        # Map candidate files to landings on main between the branch's merge-base
        # and main's tip (I2 read-only git log; 2357: both SHAs are the
        # authoritative dispatch-time inputs, consumed as-given, never
        # re-derived here). I6 (task 2869): implicated landings are filtered to
        # ancestors of the CURRENT real main HEAD, so an orphaned speculative
        # main_sha train tip never cites its dangling commits.
        implicated_commits, overlap_files = await _implicated_landings(
            repo_root, merge_base_sha, main_sha, candidate_files,
            real_main_head_sha=real_main_head_sha,
        )

        # Searched, and nothing on main touching a failing file is implicated
        # -> the failure is the branch's own. (workflow_verify presence is
        # irrelevant here: with nothing on main to blame, a recorded green
        # cannot turn a branch bug into a skew.)
        #
        # Review note (task 2871): I7's failing_tests requirement below (see
        # the implicated-commits branch) does NOT gate this return — a
        # harness-ratchet/guard failure with zero parsed failing-test ids
        # whose candidate files happen to overlap no main landing at all
        # still resolves to BRANCH_BUG here, unconditionally on failing_tests.
        # This is deliberate, not an oversight: I7 exists to hedge the
        # heterogeneous true-cause ambiguity (own-diff vs main-red) that only
        # arises once a main landing IS implicated (2871's census: 5053/5056
        # vs 5288/5266). With NO landing implicated at all, there is nothing
        # on main left to spuriously blame, so the branch's own diff is the
        # only remaining candidate cause and BRANCH_BUG stays the correct
        # fail-open default regardless of failing_tests. Precisely
        # discriminating a genuine own-diff guard trip from a mis-extracted
        # candidate-file false negative here would need guard-text detection
        # (e.g. recognizing kLOC-cap/baseline-manifest guard output
        # specifically) — a separate, not-yet-scoped follow-up, not a gap in
        # this task's I7 fix.
        if not implicated_commits:
            # No landings -> nothing gathered, so observed_evidence is None too
            # (task 3178). That is what keeps the BRANCH_BUG merge_attempt row's
            # payload byte-identical under the widened emit guard.
            return ClassificationResult(MergeFailureDisposition.BRANCH_BUG, None, None)

        # A landing IS implicated. I5: call it INTEGRATION_SKEW only when the
        # branch is *positively confirmed* green pre-merge (workflow_verify).
        # Absent green (None, e.g. a first-attempt skew or a non-orchestrator
        # submit path) or a failed green (False) degrades to the honest
        # INDETERMINATE — never a fabricated skew.
        #
        # I7 (task 2871, reify esc-5053-13 & esc-5056-11; SHAPE FLOOR added by
        # task 3178 — see the module docstring's I7 incident account): require
        # at least one NODE-SHAPED failing-test id. The floor itself lives in
        # _extract_failing_tests_and_candidate_files, so by the time control
        # reaches here `failing_tests` holds only node-shaped ids.
        #
        # Load-bearing local choice: an empty failing_tests degrades to
        # INDETERMINATE, NOT BRANCH_BUG. A harness-ratchet / shell-guard failure
        # (kLOC cap, baseline-manifest grandfathering, other tests/infra/*.sh
        # guards) has as its only "evidence" a spurious file-token overlap on
        # the runner (run_all.sh / verify.sh) or the guard's own artifact —
        # causally irrelevant even when the implicated landing is a genuine
        # real-main ancestor (2869's I6 filter therefore cannot prune it). The
        # class's true cause is heterogeneous across the census (own-diff for
        # 5053/5056 vs main-red for 5288/5266), so BRANCH_BUG would both
        # mislabel the main-red instances and emit a misleading merge_attempt
        # disposition row. INDETERMINATE is the honest fallback.
        #
        # The bundle GATHERED — built exactly once, returned as
        # observed_evidence on BOTH the promoted and the degraded path (see
        # ClassificationResult).
        observed = SkewEvidence(
            implicated_commits=implicated_commits,
            failing_tests=failing_tests,
            overlap_files=overlap_files,
        )
        green = _branch_pre_merge_verify_green(event_store, task_id)
        if failing_tests and green is True:
            return ClassificationResult(
                MergeFailureDisposition.INTEGRATION_SKEW, observed, observed,
            )

        # Loud degrade (task 3178). This complements — does not replace — the
        # merge_attempt row the caller now emits: the row is the machine-readable
        # census surface, this is the greppable operator surface. Both honour the
        # repo's loud-over-silent / structured-facts-at-failure invariant, and
        # the silent return this replaced is half of why the false I7 premise
        # went unchecked (module docstring, THE I7 INCIDENT).
        reasons: list[str] = []
        if not failing_tests:
            reasons.append('no node-shaped failing-test id')
        if green is not True:
            reasons.append('branch pre-merge green not confirmed')
        # Every list logs its TRUE length beside the slice (see
        # _MAX_LOGGED_EVIDENCE_ITEMS): a truncation a reader cannot detect is
        # the same silent-degrade failure mode this warning exists to end.
        # failing_tests is logged too, so an I5-ONLY degrade (node-shaped ids
        # present, green unconfirmed) names the ids that were on the table.
        logger.warning(
            'classify_merge_failure_disposition: task=%s degrading implicated '
            'landings to INDETERMINATE (%s); failing_tests=%d %s '
            'implicated_commits=%d %s overlap_files=%d %s',
            task_id,
            '; '.join(reasons),
            len(failing_tests),
            failing_tests[:_MAX_LOGGED_EVIDENCE_ITEMS],
            len(implicated_commits),
            implicated_commits[:_MAX_LOGGED_EVIDENCE_ITEMS],
            len(overlap_files),
            overlap_files[:_MAX_LOGGED_EVIDENCE_ITEMS],
        )
        return ClassificationResult(
            MergeFailureDisposition.INDETERMINATE, None, observed,
        )
    except Exception:
        logger.warning(
            'classify_merge_failure_disposition: internal error; degrading to '
            'INDETERMINATE (fail-open, I3)',
            exc_info=True,
        )
        # A classifier fault gathered nothing and MUST NOT fabricate a bundle:
        # observed_evidence stays None, which is what keeps I3's fail-open path
        # byte-identical downstream (no merge_attempt row is emitted for it).
        return ClassificationResult(MergeFailureDisposition.INDETERMINATE, None, None)
