"""Classify a submitted branch tip against the tip already in flight for it.

:class:`TipRelation` and the decision tables built on it: the attach action
(:func:`resolve_attach_action`), the C3 submit-gate action
(:func:`decide_c3_submit_action`) and the recovery winner
(:func:`select_recovery_winner`).
"""
from __future__ import annotations

from enum import Enum

from orchestrator.git_ops import GitOps
from orchestrator.merge_lane.landing_evidence import patch_content_contained


class TipRelation(Enum):
    """Topological relationship between two branch-tip SHAs.

    Used by the γ1 multi-waiter coalescing substrate to decide how to handle
    a new submission when a merge is already in-flight for the same branch.

    Values:
    - SAME: both tips are the same SHA.
    - SUPERSET: *new_tip* is a strict ancestor-descendant of *old_tip*
      (new has all of old's commits plus more).
    - SUBSET: *new_tip* is an ancestor of *old_tip* (new is strictly behind).
    - DIVERGENT: neither tip is an ancestor of the other; must be resolved via
      :func:`resolve_divergent` before passing to :func:`decide_attach_action`.

    Consumer wiring (pending as of γ1):
    - γ2 (task 1640): wires :func:`classify_tip_relation` /
      :func:`decide_attach_action` into the worker's post-merge-equivalence
      path to trigger generation auto-chaining; flips ``entry.verifying`` via
      :meth:`InFlightMergeRegistry.set_verifying`.
    - γ3 (task 1641): wires :func:`classify_tip_relation` /
      :func:`decide_attach_action` at the workflow merge-phase submission
      site to attach as a peer waiter.
    """

    SAME = 'same'
    SUPERSET = 'superset'
    SUBSET = 'subset'
    DIVERGENT = 'divergent'


async def classify_tip_relation(
    new_tip: str,
    old_tip: str,
    git_ops: GitOps,
) -> TipRelation:
    """Return the topological relationship between *new_tip* and *old_tip*.

    Checks performed in order (cheap first):

    1. SHA equality → SAME.
    2. ``is_ancestor(old_tip, new_tip)`` → SUPERSET (old is strictly behind new).
    3. ``is_ancestor(new_tip, old_tip)`` → SUBSET (new is strictly behind old).
    4. Neither → DIVERGENT (no ancestor relation; caller must use
       :func:`resolve_divergent` to determine the patch-id relationship).

    Consumed by γ2 (worker wiring) and γ3 (workflow caller) via the pure
    :func:`decide_attach_action` mapping.
    """
    if new_tip == old_tip:
        return TipRelation.SAME
    if await git_ops.is_ancestor(old_tip, new_tip):
        return TipRelation.SUPERSET
    if await git_ops.is_ancestor(new_tip, old_tip):
        return TipRelation.SUBSET
    return TipRelation.DIVERGENT


async def resolve_divergent(
    new_tip: str,
    old_tip: str,
    git_ops: GitOps,
) -> TipRelation:
    """Resolve a DIVERGENT tip-relation using patch-id content comparison.

    Called when :func:`classify_tip_relation` returns DIVERGENT (neither tip
    is a topological ancestor of the other).

    If :func:`patch_content_contained` determines that *new_tip*'s commits are
    all already present in *old_tip* (by patch-id — covers rebase/cherry-pick
    rewrites), returns SUBSET (the new submission is content-equivalent to
    what's already in-flight, just rebased).

    Otherwise returns SUPERSET (the new submission has genuinely new content).
    """
    if await patch_content_contained(new_tip, old_tip, git_ops):
        return TipRelation.SUBSET
    return TipRelation.SUPERSET


class AttachAction(Enum):
    """Action to take when a new submission arrives for an already in-flight branch.

    Derived from the PRD §7.2 dispatch table.  Computed by the pure
    :func:`decide_attach_action` function given a resolved :class:`TipRelation`
    and the current ``verifying`` flag.

    Values:
    - COALESCE: tip identical — attach as peer, resolve with same outcome.
    - RESNAPSHOT: new tip is a superset and merge hasn't started verifying —
      update snapshot_tip and re-try the merge with the new tip.
    - ATTACH_AND_CHAIN: new tip is a superset but verify is already running —
      attach as peer and set up gen-2 chaining (γ2 worker wiring).
    - ATTACH_CONTAINMENT: new tip is a subset — attach as peer; at finalize
      time the worker resolves via containment logic (boundary test 13).

    Consumer wiring (pending as of γ1): γ2 (task 1640) and γ3 (task 1641).
    See :class:`TipRelation` for per-phase details.
    """

    COALESCE = 'coalesce'
    RESNAPSHOT = 'resnapshot'
    ATTACH_AND_CHAIN = 'attach_and_chain'
    ATTACH_CONTAINMENT = 'attach_containment'


def decide_attach_action(
    relation: TipRelation,
    *,
    verifying: bool,
) -> AttachAction:
    """Return the coalescing action for *relation* given the *verifying* flag.

    Pure function (no git I/O) — maps the PRD §7.2 table exactly:

    +------------------+----------------+---------------------+
    | relation         | verifying=False | verifying=True      |
    +==================+================+=====================+
    | SAME             | COALESCE        | COALESCE            |
    +------------------+----------------+---------------------+
    | SUPERSET         | RESNAPSHOT      | ATTACH_AND_CHAIN    |
    +------------------+----------------+---------------------+
    | SUBSET           | ATTACH_CONTAINMENT | ATTACH_CONTAINMENT |
    +------------------+----------------+---------------------+
    | DIVERGENT        | ValueError      | ValueError          |
    +------------------+----------------+---------------------+

    DIVERGENT must be resolved via :func:`resolve_divergent` before calling
    this function; it raises :class:`ValueError` otherwise to enforce the
    "patch-id compare first" contract.
    """
    if relation is TipRelation.SAME:
        return AttachAction.COALESCE
    if relation is TipRelation.SUBSET:
        return AttachAction.ATTACH_CONTAINMENT
    if relation is TipRelation.SUPERSET:
        return AttachAction.ATTACH_AND_CHAIN if verifying else AttachAction.RESNAPSHOT
    # TipRelation.DIVERGENT
    raise ValueError(
        'DIVERGENT must be resolved via resolve_divergent() first before '
        'decide_attach_action() can map it to an AttachAction.'
    )


async def resolve_attach_action(
    new_tip: str,
    old_tip: str,
    *,
    verifying: bool,
    git_ops: GitOps,
) -> AttachAction:
    """Classify and decide the attach action for a new tip vs an in-flight tip.

    Shared helper used by BOTH the workflow path (workflow.py) and the MCP
    coalesce path (coalesce_or_enqueue_merge_request) so the two paths share
    the same tip-recency *classification decision* and cannot diverge on which
    :class:`AttachAction` class to return.

    Note: the two consumers handle the resulting action differently.  The
    workflow path on RESNAPSHOT re-snapshots and then coalesces the new request
    as a peer waiter onto the in-flight entry (the peer future resolves with the
    in-flight outcome).  The MCP path on RESNAPSHOT re-snapshots and then
    independent-enqueues via :func:`register_and_enqueue_merge_request` — the
    new submission gets its own merge+verify pass; the old in-flight is not
    cancelled.  Both paths on ATTACH_AND_CHAIN independent-enqueue (identical
    behaviour).

    Composes :func:`classify_tip_relation` → :func:`resolve_divergent` →
    :func:`decide_attach_action` in sequence:

    1. Classify the topological relationship between *new_tip* and *old_tip*.
    2. If DIVERGENT, resolve via patch-id content comparison
       (:func:`resolve_divergent`) to either SUBSET or SUPERSET.
    3. Map the resolved relation + *verifying* to an :class:`AttachAction`.

    Returns one of:
    - :attr:`AttachAction.COALESCE` — tips are identical (SAME).
    - :attr:`AttachAction.RESNAPSHOT` — new tip is a SUPERSET and not verifying.
    - :attr:`AttachAction.ATTACH_AND_CHAIN` — new tip is a SUPERSET and verifying.
    - :attr:`AttachAction.ATTACH_CONTAINMENT` — new tip is a SUBSET.
    """
    relation = await classify_tip_relation(new_tip, old_tip, git_ops)
    if relation is TipRelation.DIVERGENT:
        relation = await resolve_divergent(new_tip, old_tip, git_ops)
    return decide_attach_action(relation, verifying=verifying)


class C3SubmitAction(Enum):
    """Action for the SHA-sensitive merge_request submit gate (PRD §4 C3 / §5 D1-D3).

    Computed by the pure :func:`decide_c3_submit_action` from a *resolved*
    :class:`TipRelation` (DIVERGENT already folded to SUBSET/SUPERSET via
    :func:`resolve_divergent`) and the current ``verifying`` signal.

    Values:
    - COALESCE: same SHA (SAME) or the new tip is contained in the in-flight
      snapshot (SUBSET — stale retry or patch-id-equivalent rebased twin) —
      attach to the existing work item; never a second dispatch (D1).
    - REPLACE: new tip is a strict superset of a QUEUED (verify-not-started)
      in-flight snapshot — drop the old entry, clean its scratch, dispatch
      fresh (D2).
    - REJECT: new tip is a strict superset of an IN-VERIFY in-flight snapshot —
      structured ``duplicate_in_verify`` reject; the caller must merge_cancel
      then resubmit (D3).
    """

    COALESCE = 'coalesce'
    REPLACE = 'replace'
    REJECT = 'reject'


def decide_c3_submit_action(
    relation: TipRelation,
    *,
    verifying: bool,
) -> C3SubmitAction:
    """Return the C3 submit-gate action for *relation* given the *verifying* flag.

    Pure function (no git I/O) — maps the PRD §4 C3 / §5 D1-D3 table exactly:

    +------------------+------------------+------------------+
    | relation         | verifying=False  | verifying=True   |
    +==================+==================+==================+
    | SAME             | COALESCE         | COALESCE         |
    +------------------+------------------+------------------+
    | SUBSET           | COALESCE         | COALESCE         |
    +------------------+------------------+------------------+
    | SUPERSET         | REPLACE          | REJECT           |
    +------------------+------------------+------------------+
    | DIVERGENT        | ValueError       | ValueError       |
    +------------------+------------------+------------------+

    SAME/SUBSET always COALESCE (D1: same SHA never rejects/replaces even mid-
    verify; SUBSET covers stale retries and patch-id-equivalent rebased twins).
    SUPERSET REPLACEs when the in-flight merge is still QUEUED and REJECTs when
    it is already IN VERIFY (mid-verify teardown is the exact failure C3 fixes).

    DIVERGENT must be resolved via :func:`resolve_divergent` before calling this
    function; it raises :class:`ValueError` otherwise to enforce the "patch-id
    compare first" contract (mirrors :func:`decide_attach_action`).
    """
    if relation is TipRelation.SAME:
        return C3SubmitAction.COALESCE
    if relation is TipRelation.SUBSET:
        return C3SubmitAction.COALESCE
    if relation is TipRelation.SUPERSET:
        return C3SubmitAction.REJECT if verifying else C3SubmitAction.REPLACE
    # TipRelation.DIVERGENT
    raise ValueError(
        'DIVERGENT must be resolved via resolve_divergent() first before '
        'decide_c3_submit_action() can map it to a C3SubmitAction.'
    )


async def select_recovery_winner(
    tips: list[str | None],
    git_ops: GitOps,
) -> tuple[int, bool]:
    """Pick the descendant-most snapshot tip among *tips* for recovery dedup.

    Recovery-path (task 2926, C3 γ) counterpart of the submit-path C3 gate:
    given the persisted ``snapshot_tip`` of every surviving journal record for
    ONE branch (in journal order), return ``(winner_idx, divergence_seen)``
    where ``winner_idx`` indexes the record whose tip should become the single
    enqueued merge (the descendant-most), and ``divergence_seen`` is True when
    any pairwise comparison had to be folded through :func:`resolve_divergent`
    (a force-push, D2 — the recovery caller logs a WARNING naming the branch).

    Pure on *tips* (snapshot-tip SHA strings or None) + *git_ops* — it takes
    SHA strings, NOT ``PersistedMergeRequest`` / ``MergeRequest`` objects, so it
    carries no ``merge_queue`` → ``merge_queue_store`` dependency and can be
    lazy-imported by the recovery entry point without a circular import.

    Reduces *tips* pairwise, folding each candidate against the running winner
    through the SAME shared disposition table the submit path uses
    (:func:`classify_tip_relation` → on DIVERGENT :func:`resolve_divergent` →
    :func:`decide_c3_submit_action` with ``verifying=False``):

    * REPLACE  → the candidate is a strict descendant; it becomes the winner.
    * COALESCE → SAME / SUBSET; keep the current winner (earliest-seen wins ties).
    * REJECT   → unreachable at recovery (nothing is verifying, so ``verifying``
      is always False and SUPERSET always maps to REPLACE, never REJECT).

    Routing recovery through the exact function the submit path uses makes the
    γ/δ invariant "the two paths cannot diverge on which action class" hold by
    construction rather than by two hand-rolled ancestry checks.

    None-tip handling (never crashes, never silently degrades to journal order):

    * A None *candidate* has no resolvable SHA and can never be a strict
      descendant, so it is skipped and the current winner kept.
    * A None *current winner* is shadowed by the first later candidate that has
      a real tip: that candidate is PROMOTED to winner.  Without this a
      tip-less earliest record (e.g. a legacy journal entry persisted before
      snapshot_tip existed) would win the whole reduce — every comparison
      against a None current is skipped — so "descendant-most wins" would
      silently collapse to first-seen order for the whole group.
    """
    winner_idx = 0
    divergence_seen = False
    for i in range(1, len(tips)):
        candidate = tips[i]
        current = tips[winner_idx]
        if candidate is None:
            # A tip-less candidate can never be a strict descendant — keep the
            # current winner.
            continue
        if current is None:
            # The current winner has no resolvable tip but this candidate does.
            # Prefer the record with a real tip so a mixed None/real-tip group
            # does not silently fall back to journal order (the None winner would
            # otherwise survive the whole reduce by skipping every comparison).
            winner_idx = i
            continue
        relation = await classify_tip_relation(candidate, current, git_ops)
        if relation is TipRelation.DIVERGENT:
            divergence_seen = True
            relation = await resolve_divergent(candidate, current, git_ops)
        if decide_c3_submit_action(relation, verifying=False) is C3SubmitAction.REPLACE:
            winner_idx = i
    return (winner_idx, divergence_seen)
