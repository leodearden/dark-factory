"""Anchor/contract test for the split of `_ESCALATION_INSTRUCTIONS` (task 4169).

Sibling of ``test_roles_staging_command.py``, ``test_roles_background_warning.py``,
and ``test_roles_wait_pattern.py``. Before this task, STEWARD's composed
``system_prompt`` ended with ``_ESCALATION_INSTRUCTIONS`` in full — and that
block's own final paragraph is the non-steward level gate: "``level=1`` is the
STEWARD's route, not yours ... Filing at level 1 to jump the queue buys no
faster resolution, only an audit trail showing you bypassed your handler." So
the LAST thing the steward read about escalation levels reframed the very
recourse its own body mandates four times — roles.py :1456-1461 (Rule 2,
be-conservative), :1472-1475 (Rule 6, wip_conflict/unmerged_state, the
never-auto-resolve class), :1505 (MAX_TURNS), :1522-1524 (CLI_INPUT_REJECTED)
— as a queue-jumping bypass rather than the mandated recourse it actually is.

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
    ESCALATION_LADDER_CORE,
    NON_STEWARD_LEVEL_GATE,
    ROLES,
    _ESCALATION_INSTRUCTIONS,
)

# The one role whose own prompt body mandates `escalate_blocker(..., level=1)`
# re-escalation -- roles.py :1456-1461, :1472-1475, :1505, :1522-1524. This is
# the role the fix is FOR: it must keep reading the ladder mechanics
# (ESCALATION_LADDER_CORE) but must stop reading the non-steward gate telling
# it that its own mandated recourse is a queue-jumping bypass.
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
        'their own body (roles.py :1456-1461, :1472-1475, :1505, :1522-1524) and '
        'need the ladder mechanics to know what level=1 does and that only 0 and '
        '1 are accepted.'
    )


def test_l1_filer_roles_omit_the_non_steward_gate() -> None:
    """THE FIX: the steward's composed prompt no longer contains the "not yours" gate.

    Before this task, STEWARD's system_prompt ended with the full
    `_ESCALATION_INSTRUCTIONS` block, so the LAST thing it read about escalation
    levels was `NON_STEWARD_LEVEL_GATE` telling it that `level=1` -- the exact
    call its own body mandates at roles.py :1456-1461 (Rule 2), :1472-1475
    (Rule 6, wip_conflict/unmerged_state), :1505 (MAX_TURNS), and :1522-1524
    (CLI_INPUT_REJECTED) -- is "not yours" and "buys no faster resolution, only
    an audit trail showing you bypassed your handler". This is both the fix and
    the regression guard: a future refactor that re-splices the full
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
