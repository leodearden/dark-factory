"""Shared landing-evidence helper for already-landed re-derivation sites (task 2678).

Five always-on sites re-derive "has this task's work already landed on
main?" from live git state before stamping a task ``done``:

  1. ``Harness._already_landed_dispatch_gate``'s ancestry path
  2. ``Harness._already_landed_dispatch_gate``'s branch-deleted merge-marker path
  3. ``Harness._already_landed_dispatch_gate``'s content-equivalence fallback
  4. the stranded-in-progress sweep (``Harness._reconcile_one_stranded``)
  5. ``SpeculativeMergeWorker._redrive_coalesce_members`` (coalesce re-drive)

Prior to task 2678 each site inlined its own subset of two primitives landed
by task 2675 (dep δ): ``git_ops.find_task_citation_commit`` (FIX 2,
subject-anchored citation discovery) and ``git_ops.commit_effect_present_in_main``
(FIX 1', merge second-parent/octopus-aware effect-still-present check) — an
inline-per-site shape that let two of the five sites (the merge-marker path
and the coalesce re-drive) ship WITHOUT the effect-present guard at all (the
task-1175 clobber: a reverted merge still read as a genuine landing) and let
two more lean on a silent ``x or <fallback-sha>`` expression that fabricated
provenance when discovery came up empty.

This module is the single, INV-5 extraction point: ONE async function,
:func:`validate_landing_evidence`, that both ``harness.py`` (×4 call sites)
and ``merge_queue.py`` (×1 call site) delegate to.

**Module-level, not a method** — a standalone function taking ``git_ops`` as
its first parameter, deliberately NOT a ``GitOps`` method and NOT a
``Harness`` method:

- ``harness.py`` already imports from ``merge_queue.py``; a helper living in
  either module risks an import cycle from the other. A standalone module
  both simply import (duck-typing ``git_ops``) has none.
- Existing gate-wiring tests construct ``h.git_ops = MagicMock()`` and stub
  its sub-methods (``find_task_citation_commit`` / ``is_ancestor`` /
  ``commit_effect_present_in_main``). A ``GitOps`` *method* named
  ``validate_landing_evidence`` would auto-mock under that MagicMock and
  silently bypass the real logic under test; a module-level function that
  merely CALLS those same (already-stubbed) sub-methods keeps exercising the
  real, shared decision logic.

**Pure / read-only** — this function never marks a task done, never
escalates, and never mutates git or task state. It returns a frozen
:class:`LandingEvidenceVerdict` describing whether the evidence is
attributable and effect-present; each call site owns its own stamp-vs-
escalate-vs-revert action, which differs per site (the dispatch gate returns
a bool, the sweep reverts to pending, the coalesce re-drive calls
``redrive_member``).

**Two modes**, selected by whether ``candidate_sha`` is given:

- **DISCOVERY** (``candidate_sha=None``) — the branch ref is live: discover
  a citation via ``find_task_citation_commit``, classify the effect anchor
  with ``is_ancestor(citation, branch)`` (True → an in-branch work commit,
  anchor on ``branch_tip_sha``; False → this branch's own no-ff merge commit
  OR a divergent/realigned branch ref, anchor on the citation itself), then
  apply the FIX 1' effect-present guard to that anchor. **Task 2870 removed
  the former FIX 2 bidirectional lineage HARD-REJECT** (a citation an
  ancestor of the branch in NEITHER direction used to reject as
  ``lineage_mismatch``): every DISCOVERY caller already establishes the
  branch's content is on main before delegating here, and
  ``find_task_citation_commit`` greps ``git log main`` so its citation is by
  construction reachable from main — branch-ref ancestry is NOT the landing
  authority, so a stale/divergent/realigned branch tip that fails both
  directions is still a genuine landing (reify esc-5252-9); hard-rejecting
  it caused a close↔refile ping-pong that never self-resolved. The FIX 1'
  effect-present guard stays the real gate. The ``branch_tip_sha`` anchor
  for an in-branch work-commit citation handles a stale intermediate commit
  (the branch's actual final state is its tip); the citation anchor for a
  no-ff merge commit is NOT a no-op — task 2675 made
  ``commit_effect_present_in_main`` merge-aware, so it diffs each of the
  merge commit's non-first parents' content against current main HEAD,
  correctly catching a reverted no-ff merge (the task-1175 shape). Used by
  the ancestry path, the content-equivalence fallback, and the coalesce
  re-drive.
- **CANDIDATE** (``candidate_sha`` given) — attribution was already
  established by the caller (a merge-marker subject match, or a stranded-
  sweep ground-truth report): skip citation discovery and the lineage guard
  entirely, and apply ONLY the FIX 1' effect-present guard to
  ``candidate_sha``. Used by the merge-marker path and the stranded-
  in-progress sweep.

A second shared predicate lives here for the same INV-5 reason (task 3103):
:func:`branch_is_degenerate` (with its :func:`is_valid_sha_40` helper) answers
"did this branch ever advance past its recorded creation point?" — the #1226
degeneracy signal that must run BEFORE :func:`validate_landing_evidence` on
every already-landed re-derivation. Both were previously private to
``harness.py``, which left the escalation server's ``merge_status`` Tier-3.5
and ``merge_request`` fast-path unguarded against a branch parked at an OLD
main commit: such a branch IS an ancestor of main and is NOT at main's tip,
so an ancestry-only check answers a confident ``done`` against a commit
containing none of the task's work. Consumers: ``harness.py`` (×3 sites, via
the ``Harness._branch_is_degenerate`` delegation), ``escalation/server.py``
``merge_status`` (both git-authority arms) and ``merge_request``'s
``already_merged`` fast-path.

Also shared here (INV-5): :func:`format_unattributed_landing_detail` renders
a rejected verdict into a human-facing ``(summary, detail)`` pair, and
:func:`file_unattributed_landing_escalation` is the dedup-guarded L1 filing
boilerplate (queue-None guard, ``has_open_l1`` dedup, ``Escalation``
construction) that both ``Harness._file_unattributed_landing_escalation``
and ``SpeculativeMergeWorker._file_unattributed_landing_escalation``
delegate to — the two differ only in the ``agent_role`` they pass.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeGuard

if TYPE_CHECKING:
    from escalation.queue import EscalationQueue

    from orchestrator.git_ops import GitOps

logger = logging.getLogger(__name__)


def is_valid_sha_40(s: object) -> TypeGuard[str]:
    """Return True iff *s* is a well-formed 40-char lowercase hex SHA.

    Used to validate ``branch_base_sha`` values read from task metadata
    before comparing them against live git output.  Any non-conforming
    value is treated as missing so the reconciler falls through to the
    existing citation-grep guard rather than making a bogus comparison.
    """
    return (
        isinstance(s, str)
        and len(s) == 40
        and all(c in '0123456789abcdef' for c in s)
    )


async def branch_is_degenerate(
    git_ops: GitOps, branch: str, metadata: dict[str, Any],
) -> bool:
    """Return True iff the branch is a provisioning-only degenerate branch.

    A branch is degenerate when its live tip SHA equals the recorded
    branch_base_sha (#1226), meaning zero commits were ever pushed beyond
    the creation point.  Such a branch is still an ancestor of main and is
    still distinct from main's tip, so every ancestry-based already-landed
    check needs this predicate to avoid attributing a foreign commit's
    content to the task.

    Returns False when:
    - branch_base_sha is absent or not a valid 40-hex SHA (backward compat
      for pre-#1226 tasks or tasks whose metadata write failed transiently);
    - resolve_branch_sha returns None (branch ref vanished mid-sweep —
      treat as non-degenerate so the caller falls through to escalate); or
    - the live tip has advanced past the recorded base SHA.

    The metadata check deliberately precedes the git call: an absent or
    malformed base must cost nothing and must never reach a comparison
    against live git output.

    Args:
        git_ops: A ``GitOps`` instance (or a duck-typed stand-in exposing
            ``resolve_branch_sha``).
        branch: The task's branch name (e.g. ``f'task/{task_id}'``).
        metadata: The task's metadata dict, read for ``branch_base_sha``.
    """
    branch_base_sha = metadata.get('branch_base_sha')
    if not is_valid_sha_40(branch_base_sha):
        return False
    branch_tip_sha = await git_ops.resolve_branch_sha(branch)
    return branch_tip_sha is not None and branch_tip_sha == branch_base_sha


@dataclass(frozen=True)
class LandingEvidenceVerdict:
    """The verdict of :func:`validate_landing_evidence`.

    Attributes:
        accepted: Whether the evidence is attributable AND effect-present.
        evidence_sha: The commit sha to anchor provenance on when
            ``accepted`` is True (the discovered citation, or
            ``candidate_sha`` in CANDIDATE mode); ``None`` when rejected.
        reason: Machine-readable code — ``'ok'`` when accepted, else one of
            ``'no_citation'`` (DISCOVERY only — no commit on main cites the
            task) or ``'effect_absent'`` (FIX 1': the evidence sha's effect
            is not present at current main HEAD). (Task 2870 removed the
            former ``'lineage_mismatch'`` DISCOVERY reject — a citation
            unreachable from the branch in both directions is now accepted;
            see :func:`validate_landing_evidence`.)
        probe: Structured facts about the check — ``task_id``, ``branch``,
            ``branch_tip_sha``, ``citation`` (the discovered citation or the
            candidate), ``effect_check_sha`` (the sha the effect-present
            guard actually ran against), and ``reason`` — so a caller can
            build a structured-facts escalation without prose-parsing.
    """

    accepted: bool
    evidence_sha: str | None
    reason: str
    probe: dict[str, Any]


async def validate_landing_evidence(
    git_ops: GitOps,
    task_id: str,
    branch: str,
    *,
    branch_tip_sha: str | None,
    candidate_sha: str | None = None,
    pattern_template: str | None = None,
) -> LandingEvidenceVerdict:
    """Validate already-landed evidence for *task_id* on *branch*.

    See the module docstring for the DISCOVERY (``candidate_sha=None``) vs
    CANDIDATE (``candidate_sha`` given) mode split.

    Args:
        git_ops: A ``GitOps`` instance (or a duck-typed stand-in exposing
            ``find_task_citation_commit`` / ``is_ancestor`` /
            ``commit_effect_present_in_main``).
        task_id: Bare task id (no ``task/`` prefix).
        branch: The task's branch name (e.g. ``f'task/{task_id}'``), used
            for the FIX 2 lineage guard in DISCOVERY mode. Not consulted in
            CANDIDATE mode (attribution is already established).
        branch_tip_sha: The branch's current tip sha, used as the
            effect-present anchor in DISCOVERY mode when the citation is an
            in-branch work commit. May be ``None`` in CANDIDATE mode (the
            branch may no longer exist).
        candidate_sha: When given, switches to CANDIDATE mode: skip
            discovery/lineage and apply the effect-present guard to this
            sha only.
        pattern_template: Optional override forwarded to
            ``find_task_citation_commit`` (DISCOVERY mode only).

    Returns:
        A :class:`LandingEvidenceVerdict`.
    """
    probe: dict[str, Any] = {
        'task_id': task_id,
        'branch': branch,
        'branch_tip_sha': branch_tip_sha,
        'citation': None,
        'effect_check_sha': None,
    }

    def _reject(reason: str) -> LandingEvidenceVerdict:
        probe['reason'] = reason
        return LandingEvidenceVerdict(
            accepted=False, evidence_sha=None, reason=reason, probe=dict(probe),
        )

    def _accept(evidence_sha: str) -> LandingEvidenceVerdict:
        probe['reason'] = 'ok'
        return LandingEvidenceVerdict(
            accepted=True, evidence_sha=evidence_sha, reason='ok', probe=dict(probe),
        )

    if candidate_sha is not None:
        # CANDIDATE mode: attribution is already established by the caller
        # (a merge-marker subject match, or a stranded-sweep ground-truth
        # report) — skip discovery and the FIX 2 lineage guard entirely and
        # apply ONLY the FIX 1' effect-present guard to candidate_sha.
        probe['citation'] = candidate_sha
        probe['effect_check_sha'] = candidate_sha
        if not await git_ops.commit_effect_present_in_main(candidate_sha):
            return _reject('effect_absent')
        return _accept(candidate_sha)

    citation = await git_ops.find_task_citation_commit(
        task_id, pattern_template=pattern_template,
    )
    if citation is None:
        return _reject('no_citation')
    probe['citation'] = citation

    # Citation-anchor selection (task 2870, esc-5252-9; formerly the FIX 2
    # bidirectional lineage HARD-REJECT). is_ancestor(citation, branch) still
    # classifies the anchor for the FIX 1' effect-present guard below — True
    # for a WORK commit ON the branch (anchor on the branch tip, which may
    # have advanced past this intermediate commit); False for this branch's
    # OWN no-ff merge commit OR a divergent/realigned branch ref (anchor on
    # the citation itself). A citation unreachable from the branch in BOTH
    # directions is NO LONGER rejected as 'lineage_mismatch': every DISCOVERY
    # caller has already established the branch's content is on main before
    # delegating here (the dispatch-gate ancestry path, the
    # content-equivalence fallback, and the coalesce re-drive), and
    # find_task_citation_commit greps `git log main` so its citation is by
    # construction reachable from main — branch-ref ancestry is NOT the
    # landing authority. A stale/divergent/realigned branch tip that fails
    # both is_ancestor directions is still a genuine landing; hard-rejecting
    # it caused a close<->refile ping-pong (the auto-watcher closes the
    # benign L1, the gate re-opens it) that never self-resolved. The FIX 1'
    # effect-present guard below stays the real gate against a reverted
    # (task-1175) landing.
    citation_on_branch = await git_ops.is_ancestor(citation, branch)

    # FIX 1' effect-present guard (task 2500/2675): ancestry alone doesn't
    # mean the effect survives at HEAD — a later commit on main may have
    # reverted exactly the paths the citation touched. Anchor on the branch
    # TIP for an in-branch work commit (it may be a stale intermediate
    # commit); anchor on the citation itself for a no-ff merge commit (task
    # 2675 made this a REAL check — it diffs each non-first parent's content
    # against current main HEAD, not a no-op; see
    # commit_effect_present_in_main), for a divergent/realigned branch ref
    # (task 2870 — citation_on_branch False, the branch tip is not
    # authoritative), or in the defensive case a DISCOVERY caller omitted
    # branch_tip_sha despite citation_on_branch.
    effect_check_sha = citation
    if citation_on_branch and branch_tip_sha is not None:
        effect_check_sha = branch_tip_sha
    probe['effect_check_sha'] = effect_check_sha
    if not await git_ops.commit_effect_present_in_main(effect_check_sha):
        return _reject('effect_absent')

    return _accept(citation)


_REASON_EXPLANATIONS: dict[str, str] = {
    'no_citation': (
        'No commit on main cites this task (find_task_citation_commit found '
        'nothing) — there is no positive evidence to attribute a landing to.'
    ),
    'effect_absent': (
        "The evidence commit's own effect is not present at current main "
        'HEAD (FIX 1\', task 2500/2675) — a later commit on main reverted '
        'exactly the paths it touched, so the ancestry is real but stale.'
    ),
}


def format_unattributed_landing_detail(
    task_id: str, branch: str, verdict: LandingEvidenceVerdict,
) -> tuple[str, str]:
    """Render a rejected :class:`LandingEvidenceVerdict` as (summary, detail).

    Shared (INV-5) across every escalating call site — the harness
    ``_file_unattributed_landing_escalation`` helper and the merge_queue
    coalesce re-drive escalation — so the human-facing message is
    single-sourced rather than five independently-drifting inline f-strings.

    Args:
        task_id: The task id the escalation is filed for.
        branch: The branch the evidence check ran against.
        verdict: A rejected verdict (``accepted`` False); the reason and
            probe are rendered into the detail text regardless of value,
            but this is intended to be called only on rejection.

    Returns:
        A ``(summary, detail)`` tuple — ``summary`` is a one-line, ``[:200]``-
        safe string suitable for ``Escalation.summary``; ``detail`` is a
        multi-line block for ``Escalation.detail``.
    """
    explanation = _REASON_EXPLANATIONS.get(
        verdict.reason, f'Unrecognized reason code: {verdict.reason}',
    )
    summary = (
        f'Task {task_id}: landing evidence on branch {branch!r} could not '
        f'be attributed ({verdict.reason})'
    )[:200]
    detail = (
        f'validate_landing_evidence rejected the landing evidence for task '
        f'{task_id} on branch {branch!r}.\n\n'
        f'reason: {verdict.reason}\n'
        f'{explanation}\n\n'
        f'probe: {verdict.probe}\n\n'
        'The task was NOT marked done and remains pending — it will be '
        're-evaluated on the next dispatch tick. If this landing is '
        'genuine, investigate why attribution/effect-present failed (e.g. '
        'a reverted merge, an unattributed commit, or a missing '
        'task-citing commit); resolve this escalation once confirmed.'
    )
    return summary, detail


def file_unattributed_landing_escalation(
    escalation_queue: EscalationQueue | None,
    task_id: str,
    branch: str,
    verdict: LandingEvidenceVerdict,
    *,
    agent_role: str,
) -> None:
    """Best-effort, dedup-guarded L1 escalation for unattributable landing
    evidence (task 2678, INV-5; extracted in the amendment pass — review
    finding: ``Harness._file_unattributed_landing_escalation`` and
    ``SpeculativeMergeWorker._file_unattributed_landing_escalation`` were a
    near-verbatim copy of this filing boilerplate, differing only in
    ``agent_role``).

    Called by every site that found a positive landing signal (a merge
    marker, branch content equivalent to main, or an on-main coalesce
    member) but whose :func:`validate_landing_evidence` verdict came back
    rejected — an unattributed or effect-absent landing that must not be
    silently stamped done (the task-1175 clobber this task closes).
    Escalate-instead-of-stamp is deliberately non-status-blocking: the
    caller leaves the task/member row pending (or flips it to pending) and
    it is re-evaluated next tick; the caller's own open-L1 veto naturally
    suppresses reprocessing while this L1 stays open — no separate status
    transition happens here.

    Best-effort (a no-op when *escalation_queue* is None, e.g. bare-harness
    or bare-worker unit tests) and deduped via ``has_open_l1`` so repeated
    ticks re-observing the same unattributable evidence don't stack
    duplicate L1s — one open escalation per task at a time.

    Args:
        escalation_queue: The caller's ``EscalationQueue``, or ``None``.
        task_id: The task (or coalesce member) id the escalation is filed for.
        branch: The branch the evidence check ran against.
        verdict: A rejected :class:`LandingEvidenceVerdict`.
        agent_role: The filing caller's role, e.g. ``'harness-reconcile'``
            or ``'orchestrator-merge-worker'`` — the only thing that
            distinguishes the two call sites.
    """
    if not escalation_queue:
        return
    try:
        if escalation_queue.has_open_l1(task_id):
            return
        from escalation.models import Escalation  # noqa: PLC0415

        summary, detail = format_unattributed_landing_detail(task_id, branch, verdict)
        esc = Escalation(
            id=escalation_queue.make_id(task_id),
            task_id=task_id,
            agent_role=agent_role,
            severity='blocking',
            category='provenance_unattributed',
            summary=summary,
            detail=detail,
            suggested_action='investigate_unattributed_landing_evidence',
            level=1,
        )
        escalation_queue.submit(esc)
        logger.warning(
            'Filed provenance_unattributed escalation %s for task %s '
            '(branch %s, reason %s, agent_role %s)',
            esc.id, task_id, branch, verdict.reason, agent_role,
        )
    except Exception:
        logger.warning(
            'Failed to file provenance_unattributed escalation for task %s '
            '(branch %s) — continuing without it',
            task_id, branch, exc_info=True,
        )
