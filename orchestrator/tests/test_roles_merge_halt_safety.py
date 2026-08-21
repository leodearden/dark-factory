"""Anchor/contract test for the steward's merge-halt safety rule (task 4130).

Sibling of ``test_roles_wait_pattern.py``, ``test_roles_escalation_ladder.py``,
``test_roles_background_warning.py`` and the rest of the ``test_roles_*.py``
family.

THE INCIDENT. The merge queue halts on THREE escalation categories, not two:
``Harness._rehydrate_merge_halt`` gates on ``{'wip_conflict',
'unmerged_state', 'stash_failed'}``.  ``stash_failed`` is the
main-checkout-hygiene halt from task 2758 — ``advance_main`` could not park
``project_root``'s dirty tracked WIP.  But the steward's Rule 6, the one
prompt site that tells the escalation handler NEVER to auto-resolve a
working-tree halt, named only the first two.  Task 3870 corrected this same
two-of-three framing twice and it came back a third time, because nothing tied
any prose site to the harness gate: a category added there propagated nowhere.

THE SAFETY CONSEQUENCE, which is why this is a contract test and not a docs
nit.  A steward handed a ``stash_failed`` L1 read a rule that did not cover its
escalation, so it received no prohibition against the single most tempting
"recovery" for a failed stash — running ``git stash`` / ``git stash pop``
against the main checkout.  CLAUDE.md forbids exactly that in any dark-factory
checkout: ``refs/stash`` is ONE ref shared by every worktree in the checkout
(it is not per-worktree), so a pop can apply an unrelated task's WIP into the
tree, and the merge worker's advance path consumes the same stack.  That is
incident 13674d3c68.  The hands-off ``level=1`` re-escalate path was likewise
unstated for the category.

THE FIX'S SHAPE, and why it is a pair of named constants rather than an edit
to the prose in place.  ``test_roles_wait_pattern.py``'s module docstring
states this family's assertion rule: the sanctioned shape is ``SOME_CONSTANT
in ROLES[name].system_prompt``, "never a string literal asserted against a
constant's prose.  The latter has no correctness content in either direction —
it passes on prose reworded to say the opposite and fails on a legitimate
tightening."  Two such literal pins were tried and removed in the task 3607
review; do not reintroduce them and do not "strengthen" one into a regex.  So
``assert 'stash_failed' in STEWARD.system_prompt`` is NOT available, and the
rule had to be lifted into ``MERGE_HALT_SAFETY_RULE`` / ``MERGE_HALT_
ESCALATION_CATEGORIES`` to give this file something named to assert against.

Every assertion below is accordingly a containment / identity check against a
named constant.  The two token tuples defined below are the deliberate,
narrow exception and are NOT prose pins: they hold executable git vocabulary
(the literal commands the rule forbids) and an API call shape
(``escalate_blocker`` / ``level=1``), which are the same class of machine
vocabulary as the category names themselves — reword the rule's sentences
however you like and they keep holding; delete the forbidden commands or the
mandated call and they fail, which is the point.

Sites are cited BY NAME, never by line number.  Every line number in this
task's own brief had already gone stale before the work started, which is the
same drift class the file exists to end.
"""

from __future__ import annotations

import inspect
import re

import pytest

from orchestrator.agents.roles import (
    MERGE_HALT_ESCALATION_CATEGORIES,
    MERGE_HALT_SAFETY_RULE,
    STEWARD,
)
from orchestrator.harness import Harness

# The destructive git commands Rule 6 must keep forbidding against the main
# project root.  ``git stash`` leads the list because the ``stash_failed``
# category IS a stash-shaped temptation: the queue failed to park WIP, and the
# obvious-looking manual fix is the one CLAUDE.md forbids outright (shared
# ``refs/stash``, incident 13674d3c68).
_DESTRUCTIVE_GIT_TOKENS = (
    'git stash',
    'git reset',
    'git checkout -- .',
    'git restore',
    'git clean',
)

# The hands-off recourse the rule mandates INSTEAD of any of the above.  The
# steward is the one role whose own prompt mandates level=1 (see
# test_roles_escalation_ladder.py); a rule that named the categories but
# dropped the recourse would leave the handler with a prohibition and no exit.
_RE_ESCALATE_MANDATE_TOKENS = (
    'escalate_blocker',
    'level=1',
    'manual_intervention',
)

# The authoritative gate.  ``Harness._rehydrate_merge_halt`` is cited by name;
# the set literal is read out of its source rather than restated here, because
# a hand-written expected set in this file would drift exactly the way the
# prose drifted three times.
_GATE_PATTERN = r'esc\.category in \{([^}]*)\}'


def test_merge_halt_constants_are_nonempty() -> None:
    """Both mandated constants are non-empty.

    Mirrors ``test_roles_wait_pattern.py::test_wait_pattern_guidance_is_nonempty``.
    NOT redundant with the containment tests below, though it reads that way:
    the empty string is a substring of every string, so
    ``test_safety_rule_is_spliced_into_steward_prompt`` holds VACUOUSLY if the
    rule is ever emptied, and every per-category and per-token containment
    check degenerates the same way against an empty tuple or an empty rule.

    Emptying either constant would therefore pass every other test in this
    file — which is precisely the silent-removal-during-a-prompt-refactor
    regression this family exists to catch.
    """
    assert MERGE_HALT_SAFETY_RULE.strip(), (
        'MERGE_HALT_SAFETY_RULE is empty.  Every containment test in this file '
        'still passes when it is — the empty string is a substring of anything — '
        "so this assertion is the sole guard against the steward's "
        'never-auto-resolve rule being silently dropped in a prompt refactor.'
    )
    assert MERGE_HALT_ESCALATION_CATEGORIES, (
        'MERGE_HALT_ESCALATION_CATEGORIES is empty.  The per-category '
        'containment test iterates it, so an empty tuple makes that test pass '
        'vacuously while the steward reads a rule naming no categories at all.'
    )


def test_safety_rule_is_spliced_into_steward_prompt() -> None:
    """THE SPLICE: the rule reaches the role that must obey it.

    The sanctioned shape for this family — a named constant asserted to be
    contained in a composed ``system_prompt``.  A refactor that rewrites Rule 6
    inline again, dropping the constant, fails here rather than silently
    restoring the unanchored state this task exists to end.
    """
    assert MERGE_HALT_SAFETY_RULE in STEWARD.system_prompt, (
        "MERGE_HALT_SAFETY_RULE is not spliced into STEWARD's system_prompt. "
        'The steward is the role that handles merge-halt escalations; without '
        'this block it has no prohibition against auto-resolving one with a '
        'destructive git command against the shared main checkout.'
    )


def test_every_halt_category_is_named_in_the_safety_rule() -> None:
    """THE FIX: all three halt categories are named, not two of three.

    This is the assertion that fails on today's prose — ``stash_failed`` is
    absent from ``STEWARD.system_prompt`` entirely — and the one that would
    have caught task 3870's second recurrence.  It iterates the named tuple
    rather than hard-coding three strings, so adding a fourth halt category to
    the constant automatically extends the requirement to the rule text.
    """
    missing = sorted(
        category for category in MERGE_HALT_ESCALATION_CATEGORIES
        if category not in MERGE_HALT_SAFETY_RULE
    )
    assert missing == [], (
        f'Halt categories missing from MERGE_HALT_SAFETY_RULE: {missing}.  A '
        'steward handed an escalation in an unnamed category reads a rule that '
        'does not cover it, and so receives no prohibition against '
        '"recovering" the main checkout with a destructive git command.  This '
        'is the third recurrence of exactly that gap (task 3870 fixed the '
        'prose twice); name the category in the rule rather than narrowing '
        'MERGE_HALT_ESCALATION_CATEGORIES to match the prose.'
    )


def test_destructive_git_prohibition_travels_with_the_categories() -> None:
    """CO-TRAVEL: the prohibition lives in the SAME constant as the categories.

    Pinned STRUCTURALLY rather than editorially, mirroring
    ``test_roles_wait_pattern.py``'s "the two rules are mandated to travel
    TOGETHER".  The category list is only load-bearing because of the
    prohibition attached to it: a future prompt refactor that split the two
    into separate splice units could drop the prohibition while every
    category-naming test above still passed.  Asserting both against the one
    constant makes that split impossible.
    """
    missing = [
        token for token in _DESTRUCTIVE_GIT_TOKENS
        if token not in MERGE_HALT_SAFETY_RULE
    ]
    assert missing == [], (
        f'Forbidden git commands missing from MERGE_HALT_SAFETY_RULE: {missing}. '
        'These are the destructive recoveries a steward must never run against '
        'the shared main checkout.  `git stash` in particular is non-negotiable: '
        '`refs/stash` is a single ref shared by every worktree in the checkout, '
        'so a pop can apply an unrelated task\'s WIP into the tree '
        '(incident 13674d3c68), and it is the recovery a `stash_failed` halt '
        'most invites.'
    )


def test_re_escalate_mandate_travels_with_the_categories() -> None:
    """CO-TRAVEL: the hands-off recourse lives in the same constant too.

    A prohibition with no stated alternative is an invitation to improvise.
    The steward is the one role whose own body mandates
    ``escalate_blocker(..., level=1)`` (see ``test_roles_escalation_ladder.py``,
    which splits the ladder so the steward stops reading the non-steward "level=1
    is not yours" gate); this pins that Rule 6 keeps carrying that recourse
    alongside the categories it applies to.
    """
    missing = [
        token for token in _RE_ESCALATE_MANDATE_TOKENS
        if token not in MERGE_HALT_SAFETY_RULE
    ]
    assert missing == [], (
        f'Re-escalate mandate tokens missing from MERGE_HALT_SAFETY_RULE: '
        f'{missing}.  The rule forbids auto-resolution, so it must state the '
        'sanctioned recourse — an `escalate_blocker(..., level=1)` re-escalation '
        "preserving the original category with suggested_action="
        "'manual_intervention'.  Forbidding every action without naming one is "
        'how a steward ends up improvising a destructive fix.'
    )


def test_safety_rule_categories_match_the_harness_gate() -> None:
    """THE COHERENCE ANCHOR: roles.py's tuple equals the authoritative gate.

    This is the only assertion in the file that actually prevents a FOURTH
    recurrence.  Task 3870 corrected this prose twice and it drifted again
    because no site was mechanically tied to ``Harness._rehydrate_merge_halt``
    — a category added to that gate propagated to nothing.  A hand-written
    expected set here would drift identically, so the set literal is read out
    of the gate's own source.  ``inspect.getsource`` introspection is already
    an established technique in this suite (test_merge_queue_frozen_prefix.py,
    test_merge_queue_lifecycle_registry.py, test_offline_lane_integration.py,
    and others).

    ``harness.py`` is READ here, never modified.
    """
    src = inspect.getsource(Harness._rehydrate_merge_halt)
    match = re.search(_GATE_PATTERN, src)

    if match is None:
        pytest.fail(
            'Could not locate the `esc.category in {...}` set literal in '
            '`Harness._rehydrate_merge_halt`.  This anchor is DELIBERATELY '
            'loud rather than skipped: a silent skip would quietly restore the '
            'unanchored state that let this drift recur three times.\n\n'
            'The foreseeable benign trigger is task 3859 replacing the literal '
            'with a shared `MERGE_HALT_CATEGORIES` constant.  If that has '
            'landed, REWIRE this test and '
            '`orchestrator.agents.roles.MERGE_HALT_ESCALATION_CATEGORIES` to '
            'import that constant instead of re-spelling the membership a '
            'fourth time — do not delete this anchor, and do not soften it '
            'into a skip.'
        )

    gate_categories = set(re.findall(r"'([^']+)'", match.group(1)))
    assert gate_categories == set(MERGE_HALT_ESCALATION_CATEGORIES), (
        f'MERGE_HALT_ESCALATION_CATEGORIES '
        f'{sorted(MERGE_HALT_ESCALATION_CATEGORIES)} has diverged from the '
        f'authoritative merge-halt gate in `Harness._rehydrate_merge_halt` '
        f'{sorted(gate_categories)}.\n\n'
        'The gate is the authority: it decides which escalation categories own '
        "a merge-queue halt.  If you added a category there, name it in the "
        "steward's MERGE_HALT_SAFETY_RULE and in OPERATIONS.md's merge-halt "
        'section too — the steward must know never to auto-resolve it.  If you '
        'removed one, drop it from the tuple and the rule.  Do NOT silence this '
        'by editing the tuple alone; the whole point is that the two cannot '
        'drift apart unnoticed.'
    )
