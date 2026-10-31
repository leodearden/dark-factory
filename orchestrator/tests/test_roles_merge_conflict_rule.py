"""Anchor/contract test for the steward's merge-conflict rule (task 7726).

Sibling of ``test_roles_merge_halt_safety.py`` and the rest of the
``test_roles_*.py`` family.

THE INCIDENT. Handed a rebase bounce on tasks 3493 and 3580 (and on reify
8363), the steward rewrote the queued branch instead of merging main on top of
it: a reset to main, a replacement squash branch at main, HEAD checked out on
main. The workflow's 900s steward wait then auto-dismissed the escalation and
retried the merge from a HEAD holding nothing to land.

Every assertion below is a containment or identity check against a named
constant, never a string literal pinned against the rule's prose; the reason
is recorded once, in ``test_roles_wait_pattern.py``'s module docstring.
"""

from __future__ import annotations

from escalation.categories import EscalationCategory

from orchestrator.agents.merge_conflict_guidance import STEWARD_MERGE_CONFLICT_RULE


def test_merge_conflict_rule_is_nonempty() -> None:
    assert STEWARD_MERGE_CONFLICT_RULE.strip(), (
        'STEWARD_MERGE_CONFLICT_RULE is empty. The empty string is a substring '
        'of every string, so every containment check in this file would pass '
        'vacuously while the steward reads no merge-conflict rule at all.'
    )


def test_merge_conflict_rule_names_the_rebase_bounce_category() -> None:
    """The steward must recognise the category a rebase bounce is filed under.

    ``orchestrator/src/orchestrator/suffix_graph.py`` files it as
    ``MergeBlockCategory.MERGE_NEEDS_REBASE``, whose value comes from this
    registry member. The registry is imported rather than
    ``orchestrator.merge_lane``, so the merge-lane ratchet's
    lane-importing-test count does not move.
    """
    category = EscalationCategory.MERGE_NEEDS_REBASE.value
    assert category in STEWARD_MERGE_CONFLICT_RULE, (
        f'STEWARD_MERGE_CONFLICT_RULE does not name `{category}`, the category '
        'the merge queue files a rebase bounce under, so a steward handed one '
        'cannot tell the rule applies to it.'
    )
