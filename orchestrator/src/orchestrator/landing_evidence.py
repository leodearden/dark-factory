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
  a citation via ``find_task_citation_commit``, apply the FIX 2
  citation-lineage guard (``is_ancestor`` checked in BOTH directions against
  the branch — a genuine citation is either an in-branch work commit or this
  branch's own no-ff merge commit; a citation that is an ancestor in
  NEITHER direction is an unrelated task's commit that merely matched the
  grep pattern), then the FIX 1' effect-present guard anchored on
  ``branch_tip_sha`` for an in-branch work-commit citation (it may be a
  stale intermediate commit — the branch's actual final state is its tip)
  or on the citation itself for a no-ff merge commit (its diff-tree is
  empty, so checking it is an intentional no-op). Used by the ancestry
  path, the content-equivalence fallback, and the coalesce re-drive.
- **CANDIDATE** (``candidate_sha`` given) — attribution was already
  established by the caller (a merge-marker subject match, or a stranded-
  sweep ground-truth report): skip citation discovery and the lineage guard
  entirely, and apply ONLY the FIX 1' effect-present guard to
  ``candidate_sha``. Used by the merge-marker path and the stranded-
  in-progress sweep.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from orchestrator.git_ops import GitOps


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
            task), ``'lineage_mismatch'`` (DISCOVERY only — FIX 2: the
            citation is not reachable from the branch in either direction),
            or ``'effect_absent'`` (FIX 1': the evidence sha's effect is not
            present at current main HEAD).
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

    citation = await git_ops.find_task_citation_commit(
        task_id, pattern_template=pattern_template,
    )
    if citation is None:
        return _reject('no_citation')
    probe['citation'] = citation

    # FIX 2 citation-lineage guard (task 2500/2675): the grep-found citation
    # must be tied to THIS task's own branch, not merely match the citation
    # grep pattern. Two shapes legitimately qualify: (a) a WORK commit ON
    # the branch (is_ancestor(citation, branch) True), or (b) this branch's
    # OWN no-ff merge commit (is_ancestor(branch, citation) True — the
    # branch tip is one of its parents). Neither direction holding means an
    # unrelated task's commit merely matched the grep pattern.
    citation_on_branch = await git_ops.is_ancestor(citation, branch)
    if not citation_on_branch and not await git_ops.is_ancestor(branch, citation):
        return _reject('lineage_mismatch')

    # FIX 1' effect-present guard (task 2500/2675): ancestry alone doesn't
    # mean the effect survives at HEAD — a later commit on main may have
    # reverted exactly the paths the citation touched. Anchor on the branch
    # TIP for an in-branch work commit (it may be a stale intermediate
    # commit); anchor on the citation itself for a no-ff merge commit (its
    # diff-tree is empty — an intentional no-op check).
    effect_check_sha = branch_tip_sha if citation_on_branch else citation
    probe['effect_check_sha'] = effect_check_sha
    if not await git_ops.commit_effect_present_in_main(effect_check_sha):
        return _reject('effect_absent')

    return _accept(citation)
