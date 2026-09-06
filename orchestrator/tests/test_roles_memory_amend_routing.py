"""Structural guard for `MEMORY_AMEND_ROUTING_RULE` (task 4548).

Amending an EXISTING memory in place -- `update_memory`'s `content_amend`
arm -- is gated on an `agent_id`-prefix allowlist that ships as
`['recon-stage-', 'curator-']`
(`fused_memory/config/schema.py::Mem0UpdateConfig`, enforced by
`fused_memory/server/mem0_update_authz.py::resolve_mem0_update_authorization`).
Every orchestrator-dispatched `agent_id` is `claude-task-<id>-<role>`
(`orchestrator/src/orchestrator/agents/briefing.py::BriefingAssembler._agent_identity`),
so no dispatched role can ever clear that bar -- and no role is granted the
tool in the first place (see `test_no_role_is_granted_the_in_place_update_tool`
below). A plan step instructing an agent to make that call is therefore
unexecutable twice over.

That is not hypothetical. Task 3891's plan step-6 instructed an implementer
to call the arm, pre-specifying a fallback only for a metadata rejection; it
produced five escalations (esc-3891-4 through -8) and left residual task
4478 -- a curator sitting -- outstanding, for a single memory edit. The rule
that such work must be routed to a curator sitting or a reconciliation stage
existed only in curator- and operator-facing files
(`skills/curate-fused-memories/SKILL.md`,
`plans/mem0-in-place-update-decision.md`, `OPERATIONS.md`), none of which a
dispatched plan AUTHOR reads. Task 4548 put it where an architect will see
it: `docs/task-authoring.md` section 10 for the durable, greppable statement,
and `MEMORY_AMEND_ROUTING_RULE` spliced into
`orchestrator/src/orchestrator/agents/roles.py::_MEMORY_INSTRUCTIONS` -- the
`## Memory` block already carried by the two plan AUTHORS (architect,
simple_task) and the two plan EXECUTORS who would receive such a step
(implementer, debugger), plus deep_reviewer. This module guards that splice.

ASSERTION SHAPE -- every assertion below is a containment / count /
membership check against the IMPORTED NAMED CONSTANT, never a string literal
or regex pinned to its prose. `test_roles_scope_boundary.py`'s docstring
records that two such prose pins were tried and REMOVED under task 3607
review with an explicit "do not reintroduce them, and do not 'strengthen'
one into a regex": a prose pin passes on prose reworded to say the opposite
and fails on a legitimate tightening, so it has no correctness content in
either direction. It also records the complementary failure -- a check that
stayed green through the exact regression it advertised as catching -- which
is why `test_plan_authoring_roles_carry_the_rule` asserts its derived role
set is non-empty before looping.

The `docs/task-authoring.md` half of task 4548 is deliberately NOT tested
here. Documentation prose is not runtime behaviour, and a substring pin on
doc wording is the same meta-test class. A composed role `system_prompt` is
different in kind -- it is a runtime artifact this repo already
contract-tests -- so only that half is test-driven.
"""

from __future__ import annotations

from orchestrator.agents.roles import (
    _MEMORY_INSTRUCTIONS,
    ARCHITECT,
    DEBUGGER,
    DEEP_REVIEWER,
    IMPLEMENTER,
    MEMORY_AMEND_ROUTING_RULE,
    ROLES,
    SIMPLE_TASK,
)

_PLAN_STEP_AUTHORING_TOOL = 'mcp__plan-tools__add_plan_step'
_IN_PLACE_UPDATE_TOOL = 'mcp__fused-memory__update_memory'

# Deliberately hand-listed rather than derived: these are the five roles that
# splice `_MEMORY_INSTRUCTIONS` today. A sixth role added later should require
# a deliberate decision to extend this tuple, not silently inherit coverage.
# `test_plan_authoring_roles_carry_the_rule` below is the derived counterpart
# that a future PLAN-AUTHORING role cannot escape.
_MEMORY_INSTRUCTION_ROLES = (ARCHITECT, IMPLEMENTER, DEBUGGER, DEEP_REVIEWER, SIMPLE_TASK)


def _tool_is_granted(tool: str, granted: set[str]) -> bool:
    """True if `tool` is granted directly, or covered by a `mcp__<family>__*` wildcard.

    Mirrors `test_roles_ancestry_check.py::_tool_is_granted` rather than
    importing it, so neither module's private helper becomes a cross-suite
    dependency. Wildcard awareness is load-bearing in both directions here:
    it keeps the plan-authoring selection correct if a role ever switches to
    a family wildcard, and it keeps the "nobody holds update_memory" guard
    honest if someone grants `mcp__fused-memory__*`.
    """
    if tool in granted:
        return True
    family_prefix, sep, _name = tool.rpartition('__')
    return sep == '__' and f'{family_prefix}__*' in granted


def test_memory_instructions_splice_the_amend_routing_rule() -> None:
    """`_MEMORY_INSTRUCTIONS` carries the rule exactly once."""
    assert MEMORY_AMEND_ROUTING_RULE, (
        'MEMORY_AMEND_ROUTING_RULE is empty. An empty constant makes every '
        'containment check in this module vacuously true.'
    )
    occurrences = _MEMORY_INSTRUCTIONS.count(MEMORY_AMEND_ROUTING_RULE)
    assert occurrences == 1, (
        f'_MEMORY_INSTRUCTIONS contains MEMORY_AMEND_ROUTING_RULE '
        f'{occurrences} times, expected exactly 1. Zero means the splice was '
        'dropped and the five roles below lose the rule; more than one means '
        'a double splice, which silently doubles the per-invocation token '
        'cost in all five prompts.'
    )


def test_memory_instruction_roles_carry_the_rule() -> None:
    """The five roles that splice `_MEMORY_INSTRUCTIONS` each carry the rule."""
    offenders = [
        role.name for role in _MEMORY_INSTRUCTION_ROLES
        if MEMORY_AMEND_ROUTING_RULE not in role.system_prompt
    ]
    assert offenders == [], (
        f'Role(s) missing MEMORY_AMEND_ROUTING_RULE from system_prompt: '
        f'{offenders}. These roles reach the rule through the shared '
        '_MEMORY_INSTRUCTIONS block -- restore that splice rather than '
        'pasting the rule into an individual role prompt.'
    )


def test_plan_authoring_roles_carry_the_rule() -> None:
    """Every role that can author a plan step carries the rule.

    The invariant with real correctness content: the measured failure was a
    plan AUTHOR writing the forbidden call into a step for someone else, so
    the capability that matters is plan-step authoring, not membership in the
    hand-listed tuple above. Derived from `ROLES` so a future plan-authoring
    role that ships without the rule fails here.
    """
    authors = [
        role for role in ROLES.values()
        if _tool_is_granted(_PLAN_STEP_AUTHORING_TOOL, set(role.allowed_tools))
    ]
    # Anti-vacuity: if `add_plan_step` is ever renamed, the selection above
    # goes empty and the offender check below passes while guarding nothing.
    assert authors, (
        f'No role in ROLES grants {_PLAN_STEP_AUTHORING_TOOL} (directly or '
        'via an `mcp__plan-tools__*` wildcard). Either the tool was renamed '
        '-- update _PLAN_STEP_AUTHORING_TOOL here -- or plan authoring moved, '
        'in which case re-derive this test against whatever capability now '
        'identifies a plan author. Do not delete this assertion: without it '
        'the check below passes vacuously.'
    )
    offenders = [
        role.name for role in authors
        if MEMORY_AMEND_ROUTING_RULE not in role.system_prompt
    ]
    assert offenders == [], (
        f'Plan-authoring role(s) missing MEMORY_AMEND_ROUTING_RULE from '
        f'system_prompt: {offenders}. A role that can write a plan step must '
        'carry the rule saying which steps it may not write -- splice '
        '_MEMORY_INSTRUCTIONS (or MEMORY_AMEND_ROUTING_RULE directly) into it.'
    )


def test_no_role_is_granted_the_in_place_update_tool() -> None:
    """No role holds `update_memory`, directly or via a family wildcard.

    The premise guard for the rule the three tests above carry: both the
    prompt constant and `docs/task-authoring.md` section 10 assert that a
    dispatched agent cannot amend a memory in place. Granting the tool to a
    role would make that claim false while every containment check above
    stayed green.

    This assertion is GREEN before the implementation too -- it is a guard,
    not this module's RED signal (that is the collection-time ImportError on
    MEMORY_AMEND_ROUTING_RULE). Do not "fix" it by deleting it. If a role is
    ever deliberately granted the tool, the fix is to revisit the rule's
    wording in both homes, then update this test -- not to drop it.
    """
    offenders = [
        role.name for role in ROLES.values()
        if _tool_is_granted(_IN_PLACE_UPDATE_TOOL, set(role.allowed_tools))
    ]
    assert offenders == [], (
        f'Role(s) granted {_IN_PLACE_UPDATE_TOOL}: {offenders}. No dispatched '
        'role may amend a memory in place -- the arm is gated on an '
        "`agent_id`-prefix allowlist no `claude-task-<id>-<role>` identity "
        'matches, so the grant buys a deterministic Mem0UpdateNotAuthorized '
        'refusal, and it contradicts the rule stated in '
        'MEMORY_AMEND_ROUTING_RULE and docs/task-authoring.md section 10.'
    )
