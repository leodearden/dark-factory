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
from orchestrator.agents.roles import MERGE_HALT_SAFETY_RULE, ROLES, STEWARD


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


def test_merge_conflict_rule_follows_rule_6_in_steward_prompt() -> None:
    """THE SPLICE: the rule reaches the steward, directly after Rule 6.

    One containment check of the two named constants together proves both
    that the role handling merge bounces carries the rule and that the Rules
    list numbering stays contiguous.
    """
    assert MERGE_HALT_SAFETY_RULE + STEWARD_MERGE_CONFLICT_RULE in STEWARD.system_prompt, (
        "STEWARD's system_prompt does not carry STEWARD_MERGE_CONFLICT_RULE "
        'directly after MERGE_HALT_SAFETY_RULE. The steward handles merge '
        'bounces; without this rule it has no prohibition against rewriting '
        'the queued branch, and its Rules list stops being numbered 6 then 7.'
    )


def test_only_the_steward_carries_the_merge_conflict_rule() -> None:
    carriers = [
        name for name, role in ROLES.items()
        if STEWARD_MERGE_CONFLICT_RULE in role.system_prompt
    ]
    assert carriers == ['steward'], (
        f'STEWARD_MERGE_CONFLICT_RULE is carried by {carriers}, expected only '
        "['steward']. The rule's `escalate_blocker(..., level=1)` recourse is "
        "the steward's route; the escalation-ladder tests forbid level=1 to "
        'every other role (see '
        'orchestrator/tests/test_roles_escalation_ladder.py::_L1_FILER_ROLES).'
    )
