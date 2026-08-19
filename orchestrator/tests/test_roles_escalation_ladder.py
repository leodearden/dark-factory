"""Anchor/contract test for the split of `_ESCALATION_INSTRUCTIONS` (task 4169).

Sibling of ``test_roles_staging_command.py``, ``test_roles_background_warning.py``,
and ``test_roles_wait_pattern.py``. Before this task, STEWARD's composed
``system_prompt`` ended with ``_ESCALATION_INSTRUCTIONS`` in full — and that
block's own final paragraph is the non-steward level gate: "``level=1`` is the
STEWARD's route, not yours ... Filing at level 1 to jump the queue buys no
faster resolution, only an audit trail showing you bypassed your handler." So
the LAST thing the steward read about escalation levels reframed the very
recourse its own body mandates four times — in STEWARD's own prompt body, at
Rule 2 (be conservative), Rule 6 (wip_conflict/unmerged_state, the
never-auto-resolve class), and the MAX_TURNS and CLI_INPUT_REJECTED
classification entries — as a queue-jumping bypass rather than the mandated
recourse it actually is. (Cite these sites by name, not line number: roles.py
churns and a hand-copied offset goes stale silently -- see
test_roles_wait_pattern.py's module docstring for the prior incident.)

The fix splits the literal at its existing blank line into
``ESCALATION_LADDER_CORE`` (the mechanics: what ``escalate_info`` /
``escalate_blocker`` do, severity policy, and the level=0-vs-1 ladder) and
``NON_STEWARD_LEVEL_GATE`` (the "not yours" framing — correct for every OTHER
escalating role, but wrong for the steward, which is the one role that must
use level=1). ``_ESCALATION_INSTRUCTIONS`` survives as their concatenation,
unchanged in value, so the six other escalating roles that still splice it in
as a single unit are provably unaffected; only STEWARD is re-spliced onto the
core alone.

Per ``test_roles_wait_pattern.py``'s rule for this exact class of test (stated
in its module docstring), every assertion here is a containment /
non-containment / identity check against a NAMED constant — never a string
literal pinned to a constant's prose. A prose pin "passes on prose reworded to
say the opposite and fails on a legitimate tightening" and only taxes future
prompt edits; it has no correctness content in either direction.
"""

from __future__ import annotations

from orchestrator.agents.roles import (
    _ESCALATION_INSTRUCTIONS,
    _ESCALATION_TOOLS,
    ESCALATION_LADDER_CORE,
    NON_STEWARD_LEVEL_GATE,
    ROLES,
)

# The one role whose own prompt body mandates `escalate_blocker(..., level=1)`
# re-escalation -- at Rule 2 (be conservative), Rule 6
# (wip_conflict/unmerged_state), and the MAX_TURNS and CLI_INPUT_REJECTED
# classification entries. This is the role the fix is FOR: it must keep
# reading the ladder mechanics (ESCALATION_LADDER_CORE) but must stop reading
# the non-steward gate telling it that its own mandated recourse is a
# queue-jumping bypass.
_L1_FILER_ROLES = ('steward',)

# The six roles that carry the full pre-split block today (grep-verified against
# `+ _ESCALATION_INSTRUCTIONS` splice sites; see task 4169 plan analysis point
# 6) -- judge and reviewer_comprehensive carry neither half. The gate is
# CORRECT for all six: none of them may ever file at level=1, so they must
# keep both halves, unchanged.
_NON_STEWARD_ESCALATING_ROLES = (
    'architect',
    'implementer',
    'debugger',
    'merger',
    'deep_reviewer',
    'simple_task',
)


def test_both_halves_are_nonempty() -> None:
    """Sanity: neither new constant is empty.

    An empty constant would make every containment assertion below pass
    vacuously -- the empty string is a substring of everything.
    """
    assert ESCALATION_LADDER_CORE.strip(), 'ESCALATION_LADDER_CORE is empty'
    assert NON_STEWARD_LEVEL_GATE.strip(), 'NON_STEWARD_LEVEL_GATE is empty'


def test_composite_is_core_plus_gate() -> None:
    """`_ESCALATION_INSTRUCTIONS` is exactly the two halves concatenated, core first.

    Pins the split as byte-exact: the six non-steward role prompts splice
    `_ESCALATION_INSTRUCTIONS` unchanged, so this equality is what guarantees
    those six prompts stay provably identical to before the split. It also
    blocks a future edit that grows one half only, or swaps their order.
    """
    assert _ESCALATION_INSTRUCTIONS == ESCALATION_LADDER_CORE + NON_STEWARD_LEVEL_GATE, (
        '_ESCALATION_INSTRUCTIONS is no longer exactly '
        'ESCALATION_LADDER_CORE + NON_STEWARD_LEVEL_GATE. The six non-steward '
        'escalating roles splice `_ESCALATION_INSTRUCTIONS` as a single unit and '
        'rely on this identity to stay byte-for-byte unchanged by the split.'
    )


def test_l1_filer_roles_carry_the_ladder_core() -> None:
    """The steward still reads the level=0-vs-1 mechanics its own mandates depend on."""
    offenders = sorted(
        name for name in _L1_FILER_ROLES
        if ESCALATION_LADDER_CORE not in ROLES[name].system_prompt
    )
    assert offenders == [], (
        f'Role(s) missing ESCALATION_LADDER_CORE from system_prompt: {offenders}. '
        'These roles mandate `escalate_blocker(..., level=1)` re-escalation in '
        'their own body (Rule 2 be-conservative, Rule 6 wip_conflict/unmerged_state, '
        'MAX_TURNS, CLI_INPUT_REJECTED) and need the ladder mechanics to know what '
        'level=1 does and that only 0 and 1 are accepted.'
    )


def test_l1_filer_roles_omit_the_non_steward_gate() -> None:
    """THE FIX: the steward's composed prompt no longer contains the "not yours" gate.

    Before this task, STEWARD's system_prompt ended with the full
    `_ESCALATION_INSTRUCTIONS` block, so the LAST thing it read about escalation
    levels was `NON_STEWARD_LEVEL_GATE` telling it that `level=1` -- the exact
    call its own body mandates at Rule 2 (be conservative), Rule 6
    (wip_conflict/unmerged_state), and the MAX_TURNS and CLI_INPUT_REJECTED
    classification entries -- is "not yours" and "buys no faster resolution,
    only an audit trail showing you bypassed your handler". This is both the
    fix and the regression guard: a future refactor that re-splices the full
    `_ESCALATION_INSTRUCTIONS` block onto the steward fails here again.
    """
    offenders = sorted(
        name for name in _L1_FILER_ROLES
        if NON_STEWARD_LEVEL_GATE in ROLES[name].system_prompt
    )
    assert offenders == [], (
        f'Role(s) whose system_prompt still contains NON_STEWARD_LEVEL_GATE: '
        f'{offenders}. These roles mandate `escalate_blocker(..., level=1)` '
        're-escalation in their own body, so ending their prompt with the '
        '"level=1 is not yours" gate contradicts their own mandated recourse.'
    )


def test_non_steward_escalating_roles_carry_both_halves() -> None:
    """The six non-steward escalating roles are unaffected by the split.

    Catches the inverse regression: a worker role accidentally wired to
    `ESCALATION_LADDER_CORE` alone, silently losing the queue-jump guidance
    that tells it `level=1` is not its own recourse.
    """
    offenders = sorted(
        name for name in _NON_STEWARD_ESCALATING_ROLES
        if ESCALATION_LADDER_CORE not in ROLES[name].system_prompt
        or NON_STEWARD_LEVEL_GATE not in ROLES[name].system_prompt
    )
    assert offenders == [], (
        f'Role(s) missing one or both halves of the escalation instructions: '
        f'{offenders}. These roles must never file at level=1, so they need '
        'both ESCALATION_LADDER_CORE (the mechanics) and NON_STEWARD_LEVEL_GATE '
        '(the "level=1 is not yours" framing).'
    )


def test_escalation_tool_grant_matches_hand_classification() -> None:
    """Derive "which roles can escalate" from `ROLES`' actual tool grants and
    require it to match the hand-maintained tuples above.

    `_L1_FILER_ROLES` / `_NON_STEWARD_ESCALATING_ROLES` are hand-maintained,
    so a future role wired with an escalation tool (anything in
    `_ESCALATION_TOOLS`, i.e. `escalate_info` and/or `escalate_blocker`) but
    never added to either tuple would otherwise be invisible: every other
    assertion in this module iterates the tuples, not `ROLES`, so it would
    silently pass regardless of what that role's system_prompt says. This
    computes the "escalating" set independently, from `allowed_tools`, and
    requires it to equal the tuples' union -- so adding a new escalating role
    forces a deliberate classification here.

    Deliberately checks membership in `_ESCALATION_TOOLS` as a whole, not
    just `escalate_blocker`: DEEP_REVIEWER is granted `escalate_info` only --
    no `escalate_blocker` string appears anywhere in its `allowed_tools` --
    yet it still carries the full escalation-ladder text and belongs in
    `_NON_STEWARD_ESCALATING_ROLES`. That tool/prompt asymmetry predates and
    is unrelated to task 4169; checking only `escalate_blocker` would fail
    this assertion on that pre-existing asymmetry instead of on an actual
    classification gap, which is not what this test is for.
    """
    classified = set(_L1_FILER_ROLES) | set(_NON_STEWARD_ESCALATING_ROLES)
    escalating_by_tools = {
        name for name, role in ROLES.items()
        if any(tool in _ESCALATION_TOOLS for tool in role.allowed_tools)
    }
    assert escalating_by_tools == classified, (
        f'Roles granted an escalation tool {sorted(escalating_by_tools)} do not '
        f'match the hand-classified tuples {sorted(classified)}. A role was '
        'added to ROLES (or had its tool grants changed) without updating '
        '_L1_FILER_ROLES / _NON_STEWARD_ESCALATING_ROLES above.'
    )


def test_unclassified_roles_carry_neither_escalation_half() -> None:
    """Every role NOT in either tuple above carries neither escalation half.

    Makes the comment claim above `_NON_STEWARD_ESCALATING_ROLES` --
    "judge and reviewer_comprehensive carry neither half" -- an actual
    assertion instead of an unverified comment. Also independently catches
    a role whose system_prompt gets `ESCALATION_LADDER_CORE` or
    `NON_STEWARD_LEVEL_GATE` spliced in (e.g. by copy-pasting another role's
    prompt tail) without a matching tool grant or tuple entry -- the
    tool-grant check above would not see this, since it never inspects
    system_prompt content.
    """
    classified = set(_L1_FILER_ROLES) | set(_NON_STEWARD_ESCALATING_ROLES)
    offenders = sorted(
        name for name in ROLES
        if name not in classified
        and (
            ESCALATION_LADDER_CORE in ROLES[name].system_prompt
            or NON_STEWARD_LEVEL_GATE in ROLES[name].system_prompt
        )
    )
    assert offenders == [], (
        f'Role(s) not listed in _L1_FILER_ROLES or _NON_STEWARD_ESCALATING_ROLES '
        f'but whose system_prompt carries escalation-ladder text: {offenders}. '
        'Classify them into the appropriate tuple above.'
    )
