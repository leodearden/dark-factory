"""THE orchestrator-side recovery/redispatch pin predicate (task 3541).

PRD ``plans/task-escalation-state-graph-prd.md`` task eta (D3); spec
``docs/task-escalation-state-spec.md`` S6/E7; INV-5.

This module is the orchestrator half of the single shared veto predicate.  The
split is deliberate and is the whole reason this module exists:

* ``escalation.pins`` owns the pin CLASS — is a given open record a live
  handoff, a dead-filer L0, or a non-pinning annotation?  It is PURE,
  config-free and CATEGORY-free, and is shared with the escalation server, so
  orchestrator policy must not leak into it.
* THIS module owns the orchestrator's CATEGORY policy — which escalation
  classes a merge can itself remediate — and composes the two into the one
  predicate every recovery/redispatch veto site consumes.

It lives in its own module rather than in ``harness.py`` so
``scheduler.py`` can import it too: before task eta the scheduler's
stranded-blocked redispatch sweep re-derived a bare ``bool(rows)`` because the
relaxation was a private ``Harness`` staticmethod it could not reach, and the
two mechanisms drifted (E7's catalogued five copies).  Keep this module
IMPORT-PURE — ``escalation.pins`` plus stdlib, and nothing from
``orchestrator.harness`` / ``orchestrator.scheduler`` /
``orchestrator.task_ground_truth`` — or that drift reopens as an import cycle.
``orchestrator/tests/test_recovery_pins.py::TestNoOrchestratorCycle`` enforces
this by parsing the source.

The precedence chain that decides a record's pin class is documented once, in
``escalation/src/escalation/pins.py``; this module deliberately does not
restate it.  Neither does it restate the store-correctness contract, but it
does INHERIT one obligation from it: a caller whose escalation read FAILED
passes ``records=None``, never ``[]`` — a false "no records" routes a
genuinely-pinned strand into the plain revert branch (esc-3163).  Both
predicates below fail safe to pinning on ``None``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from escalation.pins import PinRecord, classify_pins

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    'MERGE_REMEDIABLE_ESC_CATEGORIES',
    'only_merge_remediable',
    'records_pin_blocked_recovery',
    'records_pin_recovery',
]


# Escalation categories a MERGE can itself remediate (PRD leaf δ §2.2).
#
# The stranded-blocked reaper's own `stranded_blocked` L1 is filed to
# REQUEST exactly the remediation the verified-green auto-merge performs —
# so letting it veto that merge is an anti-synergy: the escalation asking
# for the merge blocks the merge.  Membership here means "an open
# escalation of this class does NOT veto the sweep-side self-heal", and
# nothing more: the merge is still gated by detect_verified_green's 3-part
# shape check and the merge queue's own re-verify (§2.2 "never bypasses").
#
# Deliberately MINIMAL — widen only with evidence:
#   * `stranded_merge_failed` is EXCLUDED on purpose.  It is the DURABLE
#     merge/verify-failure born-at-L2 (see Harness._file_stranded_merge_failed):
#     a re-merge cannot remediate a branch that already failed the queue's
#     verify, so a task carrying only that escalation must keep vetoing or
#     the reaper would re-submit into the same failure.
#   * every human-concern class (design_concern / task_failure /
#     review_issues / operator-action / infra_issue / ...) is excluded by
#     omission — it names a problem a merge does not fix, and must keep
#     holding the task for its handler.
MERGE_REMEDIABLE_ESC_CATEGORIES: frozenset[str] = frozenset({
    'stranded_blocked',
})


def only_merge_remediable(open_escalations: Sequence) -> bool:
    """Are *open_escalations* ALL of a merge-remediable class?

    The single category authority for the relaxed blocked-arm veto (INV-5),
    consumed through :func:`records_pin_blocked_recovery` by BOTH
    ``Harness._reconcile_one_stranded``'s blocked clauses and
    ``Scheduler._phase_redispatch_stranded_blocked``.

    Vacuously ``True`` for an empty list — so a task with no open escalation
    classifies exactly as it does today.  ``False`` as soon as ONE escalation
    falls outside :data:`MERGE_REMEDIABLE_ESC_CATEGORIES`, preserving the
    safety invariant that a human-concern escalation still vetoes the
    self-heal.

    Takes any record carrying a ``category`` — both
    ``orchestrator.task_ground_truth.EscalationRef`` (the resolver-side ref)
    and ``escalation.models.Escalation`` (the store-side row) qualify.
    ``category`` is deliberately NOT part of ``escalation.pins.PinRecord``:
    that protocol is the pin-CLASS surface, and category is orchestrator
    policy.
    """
    return all(
        ref.category in MERGE_REMEDIABLE_ESC_CATEGORIES
        for ref in open_escalations
    )


def records_pin_recovery(
    task_id: str,
    records: Sequence[PinRecord] | None,
    *,
    live_claimant: bool,
    live_claimant_id: str | None = None,
) -> bool:
    """Do *records* pin this task against RECOVERY / REDISPATCH?

    Exactly ``escalation.pins.classify_pins(...).pins`` — this wrapper exists
    so every orchestrator veto site names one function instead of re-deriving
    ``bool(open_escalations)``, and so the blocked-arm twin below can compose
    the same answer with the category relaxation.

    A ``dead_l0`` deliberately does NOT pin here (its handoff has no consumer
    left, so conversion proceeds per spec S4), while an ``info`` record never
    pins at any level.  The MARK_DONE veto is deliberately MORE conservative
    and is a different attribute of the same report —
    ``classify_pins(...).vetoes_done_flip``, consumed by
    ``task_ground_truth._shape`` and the already-landed dispatch gate.  Both
    answers come from ONE classification, which is what makes "one predicate at
    all sites" true even though the sites read two attributes.

    See :func:`escalation.pins.classify_pins` for the argument contract; pass
    ``records=None`` (never ``[]``) when the escalation read failed.
    """
    return classify_pins(
        task_id,
        records,
        live_claimant=live_claimant,
        live_claimant_id=live_claimant_id,
    ).pins


def records_pin_blocked_recovery(
    task_id: str,
    records: Sequence[PinRecord] | None,
    *,
    live_claimant: bool,
    live_claimant_id: str | None = None,
) -> bool:
    """Do *records* pin a BLOCKED task against its sweep-side self-heal?

    :func:`records_pin_recovery` narrowed by the merge-remediable relaxation:
    a record that pins by CLASS still does not veto the blocked arm when every
    open record is of a class the remediation itself resolves.

    ``records is None`` (an unreadable store) pins WITHOUT consulting the
    relaxation, and that ordering is load-bearing: the relaxation is a
    judgement about record categories, and a failed read produced no categories
    to judge — ``only_merge_remediable(())`` would be vacuously True and
    silently relax a strand nobody can see.
    """
    if records is None:
        return True
    return records_pin_recovery(
        task_id,
        records,
        live_claimant=live_claimant,
        live_claimant_id=live_claimant_id,
    ) and not only_merge_remediable(records)
