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
(FIX 1', merge second-parent/octopus-aware effect-still-present check —
since task 3116 a threshold LINE-SURVIVAL test, not byte-identity) — an
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
  ``commit_effect_present_in_main``, and optionally
  ``describe_commit_effect_in_main`` for the task-3116 divergence
  diagnostics). A ``GitOps`` *method* named
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
    *, branch_tip_sha: str | None = None,
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
            ``resolve_branch_sha``).  Not consulted when *branch_tip_sha*
            is supplied.
        branch: The task's branch name (e.g. ``f'task/{task_id}'``).
        metadata: The task's metadata dict, read for ``branch_base_sha``.
        branch_tip_sha: An already-resolved tip for *branch*.  When given,
            it is used verbatim and ``resolve_branch_sha`` is NOT called.
            This is not merely a saved subprocess: a caller that already
            ran an ancestry (or patch-id) check against a tip it observed
            must judge degeneracy against that SAME tip.  Re-reading the
            ref here would let a concurrent warm-lane reseed land between
            the two reads, so a branch whose ancestry was checked at SHA A
            could be judged degenerate/non-degenerate at SHA B and the
            resulting verdict would be anchored on evidence validated for
            a tip that is no longer current (task 3103 review).  Pass
            ``None`` (the default) only when you have not resolved the ref
            yourself — the harness call sites do exactly that.
    """
    branch_base_sha = metadata.get('branch_base_sha')
    if not is_valid_sha_40(branch_base_sha):
        return False
    if branch_tip_sha is None:
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

            On an ``'effect_absent'`` REJECT ONLY, four further keys carry
            the divergence diagnostics (task 3116; absent on every accept and
            on a ``'no_citation'`` reject):

            - ``diverged_paths`` — the touched paths that no longer match
              main HEAD, as a list.  ``None`` means "could not be
              determined" (the probe itself failed); ``[]`` means
              "determined, and empty".  The two are deliberately DISTINCT —
              collapsing them would render an unprobeable ``git_ops``
              stand-in as a clean no-divergence result.
            - ``effect_failure`` — the probe's structured failure code (see
              :class:`~orchestrator.git_ops.CommitEffectProbe`), or ``None``
              when the re-probe found the effect PRESENT, which means main
              HEAD advanced between the decision and the probe.
            - ``effect_anchor_sha`` — the commit the divergence comparison
              actually ran against (a merge citation is judged against a
              parent, not against ``effect_check_sha`` itself).
            - ``effect_probe_error`` — present only when the probe raised;
              the ``repr`` of the exception.  Recorded rather than swallowed
              so the escalation states plainly that the paths are unknown.
    """

    accepted: bool
    evidence_sha: str | None
    reason: str
    probe: dict[str, Any]


async def _record_effect_divergence(
    git_ops: GitOps, effect_check_sha: str, probe: dict[str, Any],
) -> None:
    """Enrich *probe* with WHICH paths diverged, on the effect_absent reject
    path only (task 3116).

    Awaits ``git_ops.describe_commit_effect_in_main(effect_check_sha)`` and
    writes ``diverged_paths`` / ``effect_failure`` / ``effect_anchor_sha``.
    Called ONLY after the boolean ``commit_effect_present_in_main`` has
    already rejected, and never on the accept path — the decision is the
    bool's, this is only its explanation.  The extra git work is free in
    practice because both calls hit the same ``(commit_sha, main_sha)`` memo.

    Diagnostic-only, so it must NEVER break a gate decision: any exception is
    contained.  But it is NOT swallowed — the failure is recorded into
    ``probe``, which is rendered verbatim into the escalation a human reads,
    and logged at WARNING with a traceback.  That is the repo's
    structured-facts-at-failure / no-silent-fail-soft invariant applied to
    this try/except.  The realistic trigger is a duck-typed ``git_ops``
    stand-in predating this method (the shape seven gate-wiring test files
    construct), which raises AttributeError here.

    On failure ``diverged_paths`` is set to None — "could not be determined",
    deliberately distinct from ``[]`` — so a formatter can never render an
    unprobeable stand-in as a clean no-divergence result.
    """
    try:
        result = await git_ops.describe_commit_effect_in_main(effect_check_sha)
        probe['diverged_paths'] = list(result.diverged_paths)
        probe['effect_failure'] = result.failure
        probe['effect_anchor_sha'] = result.anchor_sha
    except Exception as exc:
        probe['diverged_paths'] = None
        probe['effect_probe_error'] = repr(exc)
        logger.warning(
            'describe_commit_effect_in_main failed for %s — the effect_absent '
            'escalation will report that diverged paths could not be determined',
            effect_check_sha, exc_info=True,
        )


#: Per-check differential verdicts recorded in
#: ``probe['delivered_checks_legs']`` (task 3116).  ``'made_true_by_this_commit'``
#: is the only one that confirms; every other value is explicitly NO SIGNAL,
#: never evidence against a landing.
_DIFFERENTIAL_CONFIRMED = 'made_true_by_this_commit'


async def _delivered_checks_differential(
    git_ops: GitOps,
    effect_check_sha: str,
    delivered_checks: list[dict[str, Any]],
    probe: dict[str, Any],
) -> bool:
    """The SECOND accept path: did *effect_check_sha* MAKE a declared
    capability true? (task 3116 part b.)

    Threshold line survival (``commit_effect_present_in_main``) recovers most
    of the false-positive class, but it remains a heuristic over line sets.
    This is orthogonal POSITIVE evidence, and it is the task's OWN declared
    ground truth: run each ``metadata.delivered_checks`` entry at three refs —

        ``<sha>^1``   the pre-merge parent (first parent of a merge; the
                      commit's own parent for a non-merge — ``^1`` is both)
        ``<sha>``     the citation itself
        ``main``      current HEAD

    — and confirm when a check was FALSE at the parent, TRUE at the citation
    and TRUE now.  That sequence is the whole signal: a check that merely
    passes at main proves nothing, because the capability may have arrived by
    any other route or have been true all along.  Only the three legs together
    say THIS commit made it true, which no amount of line-set decay can argue
    with.

    **UPGRADE-ONLY.**  This is called from inside the ``effect_absent`` reject
    branch and nowhere else, so it is structurally incapable of rejecting a
    landing that survival accepted — the rule holds by construction rather
    than by care.  That matters because the stored checks are NOT trustworthy
    input: they rot when a path is renamed (erroring forever), they are
    sometimes written broad enough to have been true before the merge, and 2
    of the 458 live entries are ``script`` kind, which takes no ref at all.
    Every one of those degrades to NO SIGNAL.

    ``expect`` is honoured ONE LAYER DOWN and must not be re-applied here.
    :func:`~orchestrator.delivered_checks.run_delivered_check` already
    resolves it (for ``expect='absent'`` it is NO-match that means DELIVERED),
    so the leg pattern reads uniformly for both expects.  Inverting again for
    the 43 stored ``expect='absent'`` checks would read a capability a LATER
    commit removed as proof that this one delivered it — the exact
    double-inversion this docstring exists to prevent.

    The ``main`` leg deliberately names the branch rather than a sha resolved
    here: it asks whether the capability holds at HEAD *now*, which is the
    question, and it matches ``run_delivered_check``'s own contract (grep is
    evaluated against the committed tree at *ref*).  The first two legs are
    immutable history, so a HEAD advance mid-differential can only change the
    freshness of the third, not the "this commit made it true" core.

    LAZY import (``# noqa: PLC0415``), mirroring
    :func:`file_unattributed_landing_escalation`'s ``escalation.models``
    import: this module deliberately imports nothing from ``orchestrator`` at
    module level (see the module docstring) because
    ``orchestrator.delivered_checks`` imports ``orchestrator.git_ops``, and a
    module-level edge here would re-create the harness<->merge_queue cycle
    this module was extracted to avoid.

    Writes ``delivered_checks_legs`` (every check's three-leg outcome) and
    ``delivered_checks_outcome`` (``'confirmed'`` / ``'no_signal'``) into
    *probe*, so the escalation can say "capability X was made true by this
    merge" or state plainly that there was no signal.  Returns True only on a
    confirmation; any exception is contained, recorded in
    ``delivered_checks_error`` and logged, and declines the upgrade — the
    fail-safe direction for an accept path is to not accept.
    """
    from orchestrator.delivered_checks import (  # noqa: PLC0415
        DeliveredCheckResult,
        run_delivered_check,
    )

    legs: list[dict[str, Any]] = []
    confirmed = False
    try:
        parent_ref = f'{effect_check_sha}^1'
        for check in delivered_checks:
            kind = check.get('kind')
            if kind != 'grep':
                # A script check execs against the LIVE CHECKOUT and takes no
                # ref, so it cannot express a differential at all: running it
                # three times would answer identically each time.  Recorded as
                # an explicit carve-out rather than silently dropped.
                legs.append({
                    'name': check.get('name'),
                    'kind': kind,
                    'verdict': (
                        'script_kind_no_signal' if kind == 'script'
                        else 'unsupported_kind_no_signal'
                    ),
                })
                continue
            results = {}
            for label, ref in (
                ('parent', parent_ref),
                ('citation', effect_check_sha),
                ('main', 'main'),
            ):
                results[label] = await run_delivered_check(
                    check, project_root=git_ops.project_root, ref=ref,
                )
            made_true = (
                results['parent'] is DeliveredCheckResult.FAILED
                and results['citation'] is DeliveredCheckResult.DELIVERED
                and results['main'] is DeliveredCheckResult.DELIVERED
            )
            confirmed = confirmed or made_true
            legs.append({
                'name': check.get('name'),
                'kind': kind,
                'expect': check.get('expect'),
                'parent': results['parent'].value,
                'citation': results['citation'].value,
                'main': results['main'].value,
                'verdict': _DIFFERENTIAL_CONFIRMED if made_true else 'no_signal',
            })
    except Exception as exc:
        # Contained but NOT swallowed (structured-facts-at-failure): the whole
        # differential declines rather than upgrading on partial evidence.
        confirmed = False
        probe['delivered_checks_error'] = repr(exc)
        logger.warning(
            'delivered-checks differential failed for %s — the landing will '
            'be reported as unattributed with no differential signal',
            effect_check_sha, exc_info=True,
        )
    probe['delivered_checks_legs'] = legs
    probe['delivered_checks_outcome'] = 'confirmed' if confirmed else 'no_signal'
    return confirmed


async def validate_landing_evidence(
    git_ops: GitOps,
    task_id: str,
    branch: str,
    *,
    branch_tip_sha: str | None,
    candidate_sha: str | None = None,
    pattern_template: str | None = None,
    delivered_checks: list[dict[str, Any]] | None = None,
) -> LandingEvidenceVerdict:
    """Validate already-landed evidence for *task_id* on *branch*.

    See the module docstring for the DISCOVERY (``candidate_sha=None``) vs
    CANDIDATE (``candidate_sha`` given) mode split.

    Args:
        git_ops: A ``GitOps`` instance (or a duck-typed stand-in exposing
            ``find_task_citation_commit`` / ``is_ancestor`` /
            ``commit_effect_present_in_main``, plus
            ``describe_commit_effect_in_main`` for the effect_absent
            divergence diagnostics — a stand-in lacking that last method
            still decides identically; the failure is recorded in
            ``probe['effect_probe_error']``).
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
        delivered_checks: The task's ``metadata.delivered_checks`` list,
            enabling the SECOND accept path (task 3116; see
            :func:`_delivered_checks_differential`). THREE-STATE, and the
            states are not interchangeable: ``None`` means NOT SUPPLIED — a
            call site that has not been wired yet — while ``[]`` means
            supplied by a task that declares no checks. Both are recorded in
            ``probe['delivered_checks_state']`` so a permanently-unwired site
            is distinguishable from a task with nothing to check; collapsing
            them would leave the capstone task that makes this parameter
            required with no signal to act on. Consulted ONLY on the
            ``effect_absent`` reject path, so it can upgrade a rejection to an
            acceptance and can NEVER do the reverse (b6: the interface stays
            binary — omitting this parameter reproduces the pre-task behaviour
            exactly).

    Returns:
        A :class:`LandingEvidenceVerdict`.
    """
    probe: dict[str, Any] = {
        'task_id': task_id,
        'branch': branch,
        'branch_tip_sha': branch_tip_sha,
        'citation': None,
        'effect_check_sha': None,
        # Supply state, recorded unconditionally — wiring is a property of the
        # CALL SITE, not of the outcome, so an accepted verdict must show it
        # too.  Otherwise the only way to learn a site is unwired is to wait
        # for it to reject.
        'delivered_checks_state': (
            'unwired' if delivered_checks is None
            else 'evaluated' if delivered_checks
            else 'none_declared'
        ),
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
            await _record_effect_divergence(git_ops, candidate_sha, probe)
            # SECOND ACCEPT PATH, and it lives INSIDE the reject branch on
            # purpose: a differential can only ever rescue a rejection, never
            # produce one (task 3116).
            if delivered_checks and await _delivered_checks_differential(
                git_ops, candidate_sha, delivered_checks, probe,
            ):
                return _accept(candidate_sha)
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
    # CHANGED exactly the paths the citation touched (a revert, or ordinary
    # later evolution — this check cannot distinguish them; task 3116). Anchor on the branch
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
        await _record_effect_divergence(git_ops, effect_check_sha, probe)
        # SECOND ACCEPT PATH — anchored on the sha the survival check actually
        # ran against, so both modes ask the same question.  Inside the reject
        # branch, so it is structurally upgrade-only (task 3116).
        if delivered_checks and await _delivered_checks_differential(
            git_ops, effect_check_sha, delivered_checks, probe,
        ):
            return _accept(citation)
        return _reject('effect_absent')

    return _accept(citation)


_REASON_EXPLANATIONS: dict[str, str] = {
    'no_citation': (
        'No commit on main cites this task (find_task_citation_commit found '
        'nothing) — there is no positive evidence to attribute a landing to.'
    ),
    'effect_absent': (
        "The evidence commit's own effect is not present at current main "
        "HEAD (FIX 1', task 2500/2675) — the paths it touched no longer "
        'match main HEAD. This means EITHER a later commit reverted them OR '
        'they simply evolved further (another already-landed task\'s '
        'follow-up edit, or a merge whose branch predates an unrelated '
        'change to a co-touched file); this check CANNOT distinguish the '
        'two. Judge the diverged paths below: if none of them is this '
        "task's deliverable, this is skew, not a revert."
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
    divergence_block, summary_fragment = _render_effect_divergence(verdict)
    differential_block = _render_delivered_checks_differential(verdict)
    summary = (
        f'Task {task_id}: landing evidence on branch {branch!r} could not '
        f'be attributed ({verdict.reason}){summary_fragment}'
    )[:200]
    detail = (
        f'validate_landing_evidence rejected the landing evidence for task '
        f'{task_id} on branch {branch!r}.\n\n'
        f'reason: {verdict.reason}\n'
        f'{explanation}\n\n'
        f'{divergence_block}'
        f'{differential_block}'
        f'probe: {verdict.probe}\n\n'
        'The task was NOT marked done. It is left pending (or flipped to '
        'pending by the coalesce re-drive, from merge-deferred), which means '
        'it will be DISPATCHED TO AN AGENT on the next dispatch tick — a '
        'full plan/verify/review cycle, not a cheap idempotent re-check. If '
        'this landing is genuine, that dispatch is pure waste and will '
        'REPEAT every tick, because this condition does not heal on its own. '
        'Investigate why attribution/effect-present failed (e.g. ordinary '
        'later evolution of the touched paths, a branch-alias landing, a '
        'genuine reverted merge, an unattributed commit, or a missing '
        'task-citing commit); resolve this escalation once confirmed.'
    )
    return summary, detail


def _render_effect_divergence(
    verdict: LandingEvidenceVerdict,
) -> tuple[str, str]:
    """Render the effect_absent divergence diagnostics (task 3116).

    Returns a ``(detail_block, summary_fragment)`` pair; both are empty
    strings for any reason other than ``'effect_absent'`` and for a legacy
    ``probe={}`` that predates the diagnostics, so a caller constructing
    either still renders cleanly.

    The labelled block is the point: the raw ``probe: {...}`` dict repr
    already contained the paths, but nothing pointed a reader at them.
    Naming them under a header is the one line that resolves the
    "is this a revert or just skew?" question the reason prose now poses.
    """
    if verdict.reason != 'effect_absent':
        return '', ''
    probe = verdict.probe
    if 'diverged_paths' not in probe:
        return '', ''

    paths = probe['diverged_paths']
    if paths is None:
        error = probe.get('effect_probe_error', '<no detail recorded>')
        return (
            f'diverged paths could not be determined: {error}\n\n'
        ), ''

    if not paths:
        failure = probe.get('effect_failure')
        if failure:
            return (
                'no path divergence recorded — the effect check failed '
                f'structurally: {failure}\n\n'
            ), ''
        # The decision said absent but the re-probe says present: main HEAD
        # advanced between the two calls.  Render the race rather than let
        # the escalation silently contradict itself.
        return (
            'no path divergence recorded — the re-probe found the effect '
            'present; main HEAD may have advanced between the decision and '
            'this probe.\n\n'
        ), ''

    lines = [
        'diverged paths (touched by the evidence commit, no longer '
        'matching main HEAD):',
    ]
    lines.extend(f'  - {path}' for path in paths)
    anchor = probe.get('effect_anchor_sha')
    if anchor:
        lines.append(f'effect anchor: {anchor}')
    failure = probe.get('effect_failure')
    if failure:
        lines.append(f'effect failure: {failure}')
    block = '\n'.join(lines) + '\n\n'

    fragment = f'; diverged: {paths[0]}'
    if len(paths) > 1:
        fragment += f' +{len(paths) - 1} more'
    return block, fragment


def _render_delivered_checks_differential(
    verdict: LandingEvidenceVerdict,
) -> str:
    """Render the delivered-checks differential outcome (task 3116 part b).

    Returns a detail block, or ``''`` for a legacy ``probe`` predating the
    key so a caller constructing one still renders cleanly.

    The UNWIRED state is named EXPLICITLY, and separately from
    ``none_declared``.  Omitting it would hide precisely the degradation this
    second accept path introduces: a call site nobody wired reads, in an
    escalation, exactly like a task that declares nothing to check — and the
    two are fixed by different people.  An unwired site is an orchestrator
    gap; a task with no checks is a task-authoring gap.
    """
    state = verdict.probe.get('delivered_checks_state')
    if state is None:
        return ''
    if state == 'unwired':
        return (
            'delivered-checks differential: NOT RUN — this call site is '
            'unwired (it passes no delivered_checks), so the second accept '
            'path could not be attempted. If this landing is genuine, wiring '
            "the site is what would let the task's own declared capability "
            'checks confirm it.\n\n'
        )
    if state == 'none_declared':
        return (
            'delivered-checks differential: NOT RUN — this task declares '
            'no delivered_checks, so there is no capability to confirm it '
            'delivered. Declaring one would give this landing a second, '
            'independent way to be accepted.\n\n'
        )

    outcome = verdict.probe.get('delivered_checks_outcome')
    if outcome is None:
        # Checks were supplied but the differential never ran — the survival
        # check accepted, so there was nothing to rescue.
        return (
            'delivered-checks differential: not consulted (the effect check '
            'did not reject).\n\n'
        )
    lines = [f'delivered-checks differential: {outcome}']
    for leg in verdict.probe.get('delivered_checks_legs') or []:
        name = leg.get('name')
        if 'parent' not in leg:
            lines.append(f'  - {name}: {leg.get("verdict")}')
            continue
        lines.append(
            f'  - {name}: parent={leg.get("parent")} '
            f'citation={leg.get("citation")} main={leg.get("main")} '
            f'-> {leg.get("verdict")}'
        )
    error = verdict.probe.get('delivered_checks_error')
    if error:
        lines.append(f'  differential error: {error}')
    lines.append(
        '  (a check must FAIL at the parent and PASS at both the citation '
        'and main to confirm — a static pass at main proves nothing)'
    )
    return '\n'.join(lines) + '\n\n'


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
    duplicate L1s — one open escalation per task PER CATEGORY at a time.

    **The category scoping is load-bearing** (task 3116). This call used to
    pass a bare ``task_id``, and ``category=None`` matches ANY open L1 on the
    task — a two-way blindfold in which an unrelated pending escalation (a
    ``task_failure``, say) silently suppressed a ``provenance_unattributed``
    filing, so a provenance defect hid behind an escalation that had nothing
    to do with it. Observed live on task 4105. The narrower dedup
    deliberately accepts slightly higher L1 volume — a task can now hold one
    open L1 per category rather than one overall — in exchange for not hiding
    provenance defects behind unrelated escalations. Do not widen it back.

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
        if escalation_queue.has_open_l1(
            task_id, category='provenance_unattributed',
        ):
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
