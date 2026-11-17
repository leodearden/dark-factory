"""Anchor/contract test for the steward's guard-intent rule (task 10464).

Sibling of ``test_roles_merge_conflict_rule.py`` and the rest of the
``test_roles_*.py`` family.

THE STUDY. ``plans/unreviewed-merge-changes-study-2026-11-16/REPORT.md`` found
four steward merge-phase changes that got past a failing repo guard at a cost
to the design: an allowlist row, an exemption for the task's own writer, a
staged writer split into separate writes, and a git call hidden from the
history-leg check. Steward merge-phase changes land without review, so nothing
downstream catches such a change.

Every assertion below is a containment or identity check against a named
constant, never a string literal pinned against the rule's prose; the reason
is recorded once, in ``test_roles_wait_pattern.py``'s module docstring.
"""

from __future__ import annotations

from orchestrator.agents.detector_guidance import (
    STEWARD_GUARD_INTENT_RULE,
    STEWARD_SUPPRESSION_RULES,
)
from orchestrator.agents.merge_conflict_guidance import STEWARD_MERGE_CONFLICT_RULE
from orchestrator.agents.roles import ROLES, STEWARD


def test_guard_intent_rule_is_nonempty() -> None:
    assert STEWARD_GUARD_INTENT_RULE.strip(), (
        'STEWARD_GUARD_INTENT_RULE is empty. The empty string is a substring '
        'of every string, so every containment check in this file would pass '
        'vacuously while the steward reads no guard-intent rule at all.'
    )


def test_guard_intent_rule_follows_rule_7_in_steward_prompt() -> None:
    """THE SPLICE: the rule reaches the steward, directly after Rule 7.

    The steward handles merge-phase escalations, and the Rules list stays
    numbered 7 then 8.
    """
    assert STEWARD_MERGE_CONFLICT_RULE + STEWARD_GUARD_INTENT_RULE in STEWARD.system_prompt, (
        "STEWARD's system_prompt does not carry STEWARD_GUARD_INTENT_RULE "
        'directly after STEWARD_MERGE_CONFLICT_RULE. Without it the steward '
        'may satisfy a failing repo guard by weakening what the guard '
        'protects, and its Rules list stops being numbered 7 then 8.'
    )


def test_guard_intent_rule_sits_directly_above_the_suppression_rules() -> None:
    assert STEWARD_GUARD_INTENT_RULE + STEWARD_SUPPRESSION_RULES in STEWARD.system_prompt, (
        "STEWARD's system_prompt does not carry STEWARD_SUPPRESSION_RULES "
        'directly after STEWARD_GUARD_INTENT_RULE, so the rule\'s pointer to '
        'the suppression section below it no longer resolves.'
    )


def test_only_the_steward_carries_the_guard_intent_rule() -> None:
    carriers = [
        name for name, role in ROLES.items()
        if STEWARD_GUARD_INTENT_RULE in role.system_prompt
    ]
    assert carriers == ['steward'], (
        f'STEWARD_GUARD_INTENT_RULE is carried by {carriers}, expected only '
        "['steward']. The rule's `escalate_blocker(..., level=1)` recourse is "
        "the steward's route; the escalation-ladder tests forbid level=1 to "
        'every other role (see '
        'orchestrator/tests/test_roles_escalation_ladder.py::_L1_FILER_ROLES).'
    )
