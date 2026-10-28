"""Anchor/contract test for the split of `_ESCALATION_INSTRUCTIONS` (task 4169).

Sibling of ``test_roles_staging_command.py``, ``test_roles_background_warning.py``,
and ``test_roles_wait_pattern.py``. Before this task, STEWARD's composed
``system_prompt`` ended with ``_ESCALATION_INSTRUCTIONS`` in full — and that
block's own final paragraph is the non-steward level gate: "``level=1`` is the
STEWARD's route, not yours ... Filing at level 1 to jump the queue buys no
faster resolution, only an audit trail showing you bypassed your handler." So
the LAST thing the steward read about escalation levels reframed the very
recourse its own body mandates four times — in STEWARD's own prompt body, at
Rule 2 (be conservative), Rule 6 (the merge-halt categories, the
never-auto-resolve class), and the MAX_TURNS and CLI_INPUT_REJECTED
classification entries — as a queue-jumping bypass rather than the mandated
recourse it actually is. (Cite these sites by name, not line number: roles.py
churns and a hand-copied offset goes stale silently -- see
test_roles_wait_pattern.py's module docstring for the prior incident.)

Rule 6 is enumerated in `roles.MERGE_HALT_ESCALATION_CATEGORIES` and carried
by `roles.MERGE_HALT_SAFETY_RULE`; `test_roles_merge_halt_safety.py` (task
4130) guards it.  Earlier revisions of this file spelled that rule
"wip_conflict/unmerged_state", naming two of its three categories -- the same
drift task 4130 fixed at the source.  Say "the merge-halt categories" rather
than re-spelling the membership here.

The fix splits the literal at its existing blank line into
``ESCALATION_LADDER_CORE`` (the mechanics: what ``escalate_info`` /
``escalate_blocker`` do, severity policy, and the level=0-vs-1 ladder) and
``NON_STEWARD_LEVEL_GATE`` (the "not yours" framing — correct for every OTHER
escalating role, but wrong for the steward, which is the one role that must
use level=1). ``_ESCALATION_INSTRUCTIONS`` survives as their concatenation,
unchanged in value, so the other blocker-holding roles that still splice it in
as a single unit are provably unaffected; only STEWARD is re-spliced onto the
core alone. Task 6010 classifies a third shape: a role that holds
``escalate_info`` only reads ``INFO_ONLY_ESCALATION_INSTRUCTIONS``, which names
no tool it lacks.

Per ``test_roles_wait_pattern.py``'s rule for this exact class of test (stated
in its module docstring), every assertion here is a containment /
non-containment / identity check against a NAMED constant — never a string
literal pinned to a constant's prose. A prose pin "passes on prose reworded to
say the opposite and fails on a legitimate tightening" and only taxes future
prompt edits; it has no correctness content in either direction. The
``RESERVED_BOUND_ROUTING_RULE`` checks (task 5721) are this ordinary shape.

Task 5368 adds containment checks for `escalation.models`' filer-facing action
tokens, and they are NOT that banned shape even though what they search is
prose. The tokens are a cross-package WIRE VALUE: `escalate_blocker` emits one
in its response and no code reads it, so the prompt's verbatim copy is the
entire mechanism by which an agent learns what the value means. Pinning them
here therefore survives any rewording of the surrounding sentence and fails
only when the prompt and the emitted value genuinely disagree -- which is a
real defect, not a prompt-edit tax. That is also why they are imported rather
than spelled: a literal would carry no coupling to the emission site at all.
"""

from __future__ import annotations

import pytest
from escalation.models import ACTION_KEEP_DRIVING, ACTION_TERMINATE_CLEANLY

from orchestrator.agents.harness_guidance import EXTERNAL_KILL_GUIDANCE
from orchestrator.agents.roles import (
    _ESCALATION_INSTRUCTIONS,
    _ESCALATION_TOOLS,
    ESCALATE_BLOCKER_DECISION_ROUTE,
    ESCALATE_BLOCKER_TOOL,
    ESCALATE_INFO_CONTRACT,
    ESCALATION_LADDER_CORE,
    INFO_ONLY_ESCALATION_INSTRUCTIONS,
    NON_STEWARD_LEVEL_GATE,
    RESERVED_BOUND_ROUTING_RULE,
    ROLES,
)

# The one role whose own prompt body mandates `escalate_blocker(..., level=1)`
# re-escalation -- at Rule 2 (be conservative), Rule 6
# (the merge-halt categories), and the MAX_TURNS and CLI_INPUT_REJECTED
# classification entries. This is the role the fix is FOR: it must keep
# reading the ladder mechanics (ESCALATION_LADDER_CORE) but must stop reading
# the non-steward gate telling it that its own mandated recourse is a
# queue-jumping bypass.
_L1_FILER_ROLES = ('steward',)

# The five roles that carry the full pre-split block (the `+
# _ESCALATION_INSTRUCTIONS` splice sites) -- judge and reviewer_comprehensive
# carry neither half. The gate is CORRECT for all five: none of them may ever
# file at level=1, so they must keep both halves, unchanged.
_NON_STEWARD_ESCALATING_ROLES = (
    'architect',
    'implementer',
    'debugger',
    'merger',
    'simple_task',
)

# The roles granted `escalate_info` but not `escalate_blocker` (task 6010).
# Every ladder half prescribes `escalate_blocker`, so they read
# INFO_ONLY_ESCALATION_INSTRUCTIONS instead. Membership is pinned to grants by
# test_escalation_tool_grant_matches_hand_classification.
_INFO_ONLY_ESCALATING_ROLES = ('deep_reviewer',)

# Constants that mention an escalation tool only conditionally ("if you hold
# it, otherwise ..."), so they are correct in a role that lacks that tool.
# test_role_prompt_names_only_held_escalation_tools strips them before
# scanning; test_conditional_mention_exemptions_still_name_an_escalation_tool
# keeps each entry load-bearing.
_CONDITIONAL_ESCALATION_MENTIONS = {'EXTERNAL_KILL_GUIDANCE': EXTERNAL_KILL_GUIDANCE}


def _bare_tool_name(tool: str) -> str:
    return tool.rpartition('__')[2]


def test_escalation_constants_are_nonempty() -> None:
    """Sanity: no escalation constant is empty.

    An empty constant would make every containment assertion below pass
    vacuously -- the empty string is a substring of everything.
    """
    assert ESCALATION_LADDER_CORE.strip(), 'ESCALATION_LADDER_CORE is empty'
    assert NON_STEWARD_LEVEL_GATE.strip(), 'NON_STEWARD_LEVEL_GATE is empty'
    assert ESCALATE_INFO_CONTRACT.strip(), 'ESCALATE_INFO_CONTRACT is empty'
    assert INFO_ONLY_ESCALATION_INSTRUCTIONS.strip(), (
        'INFO_ONLY_ESCALATION_INSTRUCTIONS is empty'
    )
    assert ESCALATE_BLOCKER_DECISION_ROUTE.strip(), (
        'ESCALATE_BLOCKER_DECISION_ROUTE is empty'
    )


def test_ladder_core_carries_the_escalate_info_contract() -> None:
    """The one escalate_info contract is rendered into the mechanics half (PRD D3)."""
    assert ESCALATE_INFO_CONTRACT in ESCALATION_LADDER_CORE, (
        'ESCALATION_LADDER_CORE no longer contains ESCALATE_INFO_CONTRACT. Every '
        'blocker-holding role, steward included, learns the escalate_info '
        'contract from the core; render it there exactly once.'
    )


def test_ladder_core_carries_the_blocker_decision_route() -> None:
    """The decision route names `escalate_blocker`, so it lives in the blocker ladder."""
    assert ESCALATE_BLOCKER_DECISION_ROUTE in ESCALATION_LADDER_CORE, (
        'ESCALATION_LADDER_CORE no longer contains ESCALATE_BLOCKER_DECISION_ROUTE. '
        'Splice it into the core, right after ESCALATE_INFO_CONTRACT; '
        'INFO_ONLY_ESCALATION_INSTRUCTIONS must not carry it.'
    )


def test_info_only_section_carries_the_escalate_info_contract() -> None:
    """The info-only section renders the same one contract, not a restatement (PRD D3)."""
    assert ESCALATE_INFO_CONTRACT in INFO_ONLY_ESCALATION_INSTRUCTIONS, (
        'INFO_ONLY_ESCALATION_INSTRUCTIONS no longer contains ESCALATE_INFO_CONTRACT. '
        'An escalate_info-only role learns the contract from this section; splice '
        'the constant in rather than restating it.'
    )


def test_info_only_roles_carry_the_info_only_section() -> None:
    """Each escalate_info-only role reads the section that matches its grants."""
    offenders = sorted(
        name for name in _INFO_ONLY_ESCALATING_ROLES
        if INFO_ONLY_ESCALATION_INSTRUCTIONS not in ROLES[name].system_prompt
    )
    assert offenders == [], (
        f'escalate_info-only role(s) missing INFO_ONLY_ESCALATION_INSTRUCTIONS from '
        f'system_prompt: {offenders}. Splice it where the role would otherwise '
        'splice _ESCALATION_INSTRUCTIONS.'
    )


def test_info_only_roles_carry_neither_ladder_half() -> None:
    """Both ladder halves prescribe `escalate_blocker`, which these roles lack."""
    offenders = sorted(
        name for name in _INFO_ONLY_ESCALATING_ROLES
        if ESCALATION_LADDER_CORE in ROLES[name].system_prompt
        or NON_STEWARD_LEVEL_GATE in ROLES[name].system_prompt
    )
    assert offenders == [], (
        f'escalate_info-only role(s) whose system_prompt carries a ladder half: '
        f'{offenders}. ESCALATION_LADDER_CORE and NON_STEWARD_LEVEL_GATE tell the '
        'agent to call escalate_blocker, which it is not granted; splice '
        'INFO_ONLY_ESCALATION_INSTRUCTIONS instead.'
    )


@pytest.mark.parametrize('role_key', sorted(ROLES))
def test_role_prompt_names_only_held_escalation_tools(role_key: str) -> None:
    """A role's prompt names an escalation tool only if the role holds it.

    Scans BARE names because the fully-qualified sibling,
    test_roles_ancestry_check.py::test_role_holds_every_mcp_tool_its_prompt_names,
    cannot see a backticked `escalate_blocker`: that is how nine mentions of
    it reached deep_reviewer, which cannot call it (task 6010). A tool name is
    a capability identifier, like the task-5368 tokens in the module docstring.
    """
    role = ROLES[role_key]
    text = role.system_prompt
    for mention in _CONDITIONAL_ESCALATION_MENTIONS.values():
        text = text.replace(mention, '')
    ungranted = sorted(
        tool for tool in _ESCALATION_TOOLS
        if _bare_tool_name(tool) in text and tool not in role.allowed_tools
    )
    assert ungranted == [], (
        f"{role_key}'s system_prompt names escalation tool(s) absent from its "
        f'allowed_tools: {ungranted}. Either grant the tool, or splice the '
        "escalation section that matches the role's grants "
        '(INFO_ONLY_ESCALATION_INSTRUCTIONS for an escalate_info-only role).'
    )


def test_conditional_mention_exemptions_still_name_an_escalation_tool() -> None:
    """Each exemption must still be needed, or it hides nothing and should go."""
    bare_names = [_bare_tool_name(tool) for tool in _ESCALATION_TOOLS]
    dead = sorted(
        name for name, mention in _CONDITIONAL_ESCALATION_MENTIONS.items()
        if not any(bare in mention for bare in bare_names)
    )
    unspliced = sorted(
        name for name, mention in _CONDITIONAL_ESCALATION_MENTIONS.items()
        if not any(mention in role.system_prompt for role in ROLES.values())
    )
    assert dead == [], (
        f'_CONDITIONAL_ESCALATION_MENTIONS entries that name no escalation tool: '
        f'{dead}. Delete the exemption.'
    )
    assert unspliced == [], (
        f'_CONDITIONAL_ESCALATION_MENTIONS entries spliced into no role: '
        f'{unspliced}. Delete the exemption.'
    )


def test_non_steward_gate_omits_the_escalate_info_contract() -> None:
    """The contract belongs to the core, which the steward reads.

    The same placement argument as
    `test_non_steward_gate_names_neither_filer_action_token`: the steward is
    composed from ESCALATION_LADDER_CORE alone, so a contract placed in the gate
    half would be invisible to it.
    """
    assert ESCALATE_INFO_CONTRACT not in NON_STEWARD_LEVEL_GATE, (
        'NON_STEWARD_LEVEL_GATE contains ESCALATE_INFO_CONTRACT. The steward is '
        'composed from ESCALATION_LADDER_CORE alone; move the contract into the core.'
    )


def test_composite_is_core_plus_gate() -> None:
    """`_ESCALATION_INSTRUCTIONS` is exactly the two halves concatenated, core first.

    Pins the split as byte-exact: the non-steward blocker-holding role prompts
    splice `_ESCALATION_INSTRUCTIONS` unchanged, so this equality is what
    guarantees those prompts stay provably identical to before the split. It
    also blocks a future edit that grows one half only, or swaps their order.
    """
    assert _ESCALATION_INSTRUCTIONS == ESCALATION_LADDER_CORE + NON_STEWARD_LEVEL_GATE, (
        '_ESCALATION_INSTRUCTIONS is no longer exactly '
        'ESCALATION_LADDER_CORE + NON_STEWARD_LEVEL_GATE. The non-steward '
        'blocker-holding roles splice `_ESCALATION_INSTRUCTIONS` as a single unit '
        'and rely on this identity to stay byte-for-byte unchanged by the split.'
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
        'their own body (Rule 2 be-conservative, Rule 6 merge-halt categories, '
        'MAX_TURNS, CLI_INPUT_REJECTED) and need the ladder mechanics to know what '
        'level=1 does and that only 0 and 1 are accepted.'
    )


def test_l1_filer_roles_omit_the_non_steward_gate() -> None:
    """THE FIX: the steward's composed prompt no longer contains the "not yours" gate.

    Before this task, STEWARD's system_prompt ended with the full
    `_ESCALATION_INSTRUCTIONS` block, so the LAST thing it read about escalation
    levels was `NON_STEWARD_LEVEL_GATE` telling it that `level=1` -- the exact
    call its own body mandates at Rule 2 (be conservative), Rule 6
    (the merge-halt categories), and the MAX_TURNS and CLI_INPUT_REJECTED
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
    """The non-steward blocker-holding roles are unaffected by the split.

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

    `_L1_FILER_ROLES` / `_NON_STEWARD_ESCALATING_ROLES` /
    `_INFO_ONLY_ESCALATING_ROLES` are hand-maintained, so a future role wired
    with an escalation tool (anything in `_ESCALATION_TOOLS`, i.e.
    `escalate_info` and/or `escalate_blocker`) but never added to a tuple
    would otherwise be invisible: every other assertion in this module
    iterates the tuples, not `ROLES`, so it would silently pass regardless of
    what that role's system_prompt says. This computes the "escalating" set
    independently, from `allowed_tools`, and requires it to equal the tuples'
    union -- so adding a new escalating role forces a deliberate
    classification here.

    The DEEP_REVIEWER tool/prompt asymmetry this docstring once excused is now
    classified: holding `escalate_info` without `escalate_blocker` is exactly
    `_INFO_ONLY_ESCALATING_ROLES`, pinned by the second assertion.
    """
    classified = (
        set(_L1_FILER_ROLES)
        | set(_NON_STEWARD_ESCALATING_ROLES)
        | set(_INFO_ONLY_ESCALATING_ROLES)
    )
    escalating_by_tools = {
        name for name, role in ROLES.items()
        if any(tool in _ESCALATION_TOOLS for tool in role.allowed_tools)
    }
    assert escalating_by_tools == classified, (
        f'Roles granted an escalation tool {sorted(escalating_by_tools)} do not '
        f'match the hand-classified tuples {sorted(classified)}. A role was '
        'added to ROLES (or had its tool grants changed) without updating '
        '_L1_FILER_ROLES / _NON_STEWARD_ESCALATING_ROLES / '
        '_INFO_ONLY_ESCALATING_ROLES above.'
    )
    info_only_by_tools = {
        name for name in escalating_by_tools
        if ESCALATE_BLOCKER_TOOL not in ROLES[name].allowed_tools
    }
    assert info_only_by_tools == set(_INFO_ONLY_ESCALATING_ROLES), (
        f'Roles granted an escalation tool but not escalate_blocker '
        f'{sorted(info_only_by_tools)} do not match _INFO_ONLY_ESCALATING_ROLES '
        f'{sorted(_INFO_ONLY_ESCALATING_ROLES)}. Reclassify the role whose grant '
        'changed: a role without escalate_blocker must read '
        'INFO_ONLY_ESCALATION_INSTRUCTIONS, and a role with it the ladder.'
    )


def test_unclassified_roles_carry_neither_escalation_half() -> None:
    """Every role NOT in any tuple above carries no escalation section.

    Makes the comment claim above `_NON_STEWARD_ESCALATING_ROLES` --
    "judge and reviewer_comprehensive carry neither half" -- an actual
    assertion instead of an unverified comment. Also independently catches
    a role whose system_prompt gets `ESCALATION_LADDER_CORE`,
    `NON_STEWARD_LEVEL_GATE` or `INFO_ONLY_ESCALATION_INSTRUCTIONS` spliced in
    (e.g. by copy-pasting another role's prompt tail) without a matching tool
    grant or tuple entry -- the tool-grant check above would not see this,
    since it never inspects system_prompt content.
    """
    classified = (
        set(_L1_FILER_ROLES)
        | set(_NON_STEWARD_ESCALATING_ROLES)
        | set(_INFO_ONLY_ESCALATING_ROLES)
    )
    offenders = sorted(
        name for name in ROLES
        if name not in classified
        and (
            ESCALATION_LADDER_CORE in ROLES[name].system_prompt
            or NON_STEWARD_LEVEL_GATE in ROLES[name].system_prompt
            or INFO_ONLY_ESCALATION_INSTRUCTIONS in ROLES[name].system_prompt
        )
    )
    assert offenders == [], (
        f'Role(s) not listed in _L1_FILER_ROLES, _NON_STEWARD_ESCALATING_ROLES or '
        f'_INFO_ONLY_ESCALATING_ROLES but whose system_prompt carries an '
        f'escalation section: {offenders}. Classify them into the appropriate '
        'tuple above.'
    )


def test_ladder_core_names_both_filer_action_tokens() -> None:
    """(a)+(b) The prompt tells the agent both values of the `action` key.

    `escalate_blocker` appends `action` to its response, and NO CODE READS IT
    -- the only consumer is the agent. Since task 5368 that key is
    two-valued: `terminate_cleanly` when persistence was observed,
    `keep_driving` when a post-write re-read could not confirm the record, in
    which case no handler will ever see the filing and stopping would strand
    the task. An agent that has never been told the second token exists reads
    the unconditional "STOP" instruction and does exactly the wrong thing.

    Both tokens are required, not just the new one: naming only the exception
    would leave the default unstated, so the agent would have to infer what an
    unnamed `action` value means instead of discriminating on a value it can
    read straight off the response.
    """
    missing = [
        token for token in (ACTION_TERMINATE_CLEANLY, ACTION_KEEP_DRIVING)
        if token not in ESCALATION_LADDER_CORE
    ]
    assert missing == [], (
        f'ESCALATION_LADDER_CORE does not name filer action token(s): {missing}. '
        'These are wire values emitted by `escalate_blocker` and defined in '
        '`escalation.models.FILER_ACTIONS`; the prompt quotes them verbatim '
        '(role prompts are plain literals, not f-strings), so the prompt and '
        'the emission site must move together.'
    )


def test_non_steward_gate_names_neither_filer_action_token() -> None:
    """(c) The amendment belongs to the mechanics half, which the steward reads.

    `NON_STEWARD_LEVEL_GATE` is the one half the steward is deliberately NOT
    composed with (`test_l1_filer_roles_omit_the_non_steward_gate`). Putting
    the `keep_driving` discrimination there would silently exempt the role
    that files the most blockers -- the same split-audience bug task 4169
    created this file to prevent. This assertion makes the placement
    structural rather than a convention someone has to remember.
    """
    misplaced = [
        token for token in (ACTION_TERMINATE_CLEANLY, ACTION_KEEP_DRIVING)
        if token in NON_STEWARD_LEVEL_GATE
    ]
    assert misplaced == [], (
        f'NON_STEWARD_LEVEL_GATE names filer action token(s): {misplaced}. '
        'The steward is composed from ESCALATION_LADDER_CORE alone, so anything '
        'in the gate half is invisible to the heaviest filer of blockers. Move '
        'it into the core.'
    )


def test_every_blocker_holding_role_reads_both_filer_action_tokens() -> None:
    """(d) Derived coverage: the composed prompts, not just the constant.

    Iterates the hand-classified tuples rather than re-listing role names, and
    `test_escalation_tool_grant_matches_hand_classification` independently
    pins those tuples to `ROLES`' actual tool grants -- so a newly-wired
    blocker-holding role is covered here without editing this test. Checks the
    COMPOSED `system_prompt`, which is what an agent actually receives: a role
    spliced with some other escalation text, or with none, fails here even
    while the constant itself stays correct.
    """
    offenders = sorted(
        name for name in _L1_FILER_ROLES + _NON_STEWARD_ESCALATING_ROLES
        if ACTION_TERMINATE_CLEANLY not in ROLES[name].system_prompt
        or ACTION_KEEP_DRIVING not in ROLES[name].system_prompt
    )
    assert offenders == [], (
        f'Blocker-holding role(s) whose composed system_prompt is missing one or both '
        f'filer action tokens: {offenders}. Every role that can file a blocker '
        'receives the `action` key in its response and must be told how to read '
        'both of its values.'
    )


def test_reserved_bound_rule_lives_in_the_ladder_core() -> None:
    """The rule has one home: the mechanics half, which the steward reads too."""
    assert RESERVED_BOUND_ROUTING_RULE.strip(), 'RESERVED_BOUND_ROUTING_RULE is empty'
    assert RESERVED_BOUND_ROUTING_RULE in ESCALATION_LADDER_CORE, (
        'RESERVED_BOUND_ROUTING_RULE is not spliced into ESCALATION_LADDER_CORE. '
        'Splice it into the core -- not NON_STEWARD_LEVEL_GATE, which the steward '
        'never reads, and not per role.'
    )


def test_every_blocker_holding_role_reads_the_reserved_bound_rule() -> None:
    """Derived coverage over the composed prompts each blocker-holding role receives."""
    offenders = sorted(
        name for name in _L1_FILER_ROLES + _NON_STEWARD_ESCALATING_ROLES
        if RESERVED_BOUND_ROUTING_RULE not in ROLES[name].system_prompt
    )
    assert offenders == [], (
        f'Blocker-holding role(s) whose composed system_prompt is missing '
        f'RESERVED_BOUND_ROUTING_RULE: {offenders}.'
    )
