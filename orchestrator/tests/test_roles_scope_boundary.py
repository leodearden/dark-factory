"""Structural regression guard for the Scope Boundary sections (task 4370).

Per the ratified whole-worktree sandbox granularity (PRD D1,
``plans/os-sandbox-worktree-containment-prd.md`` decision D1), the sandbox
only blocks writes OUTSIDE the task worktree (the main checkout, sibling
worktrees, other tasks' ``.task-meta/``, ``refs/heads/main``, ``~/.claude``).
Every file INSIDE the worktree is writable -- including files never listed
in the plan -- and nothing detects or blocks an in-worktree write outside
the plan's declared file set. IMPLEMENTER, DEBUGGER, and SIMPLE_TASK's
``## Scope Boundary`` sections used to tell the agent the opposite: that an
out-of-scope write gets an OS "permission error". Task 4370 rewrote the
prose to state the true boundary -- worktree containment, not per-file plan
scope -- and this file guards that the rewritten sections stay spliced in.

An earlier revision of THIS FILE pinned the fix with a literal string
comparison (``'permission error' not in role.system_prompt``) plus a
``'scope_violation' in role.system_prompt`` check, and was deleted (commit
d794419730) for two reasons that generalize beyond this file:

1. Prose-pinning has no correctness content in either direction -- it
   passes on prose reworded to say the same false thing ("the sandbox will
   block you") and fails on a legitimate tightening that happens to use the
   flagged words. ``test_roles_wait_pattern.py``'s module docstring records
   that two such literal pins were tried and REMOVED under task 3607
   review, with an explicit "do not reintroduce them, and do not
   'strengthen' one into a regex" -- the sanctioned shape is
   ``SOME_CONSTANT in ROLES[name].system_prompt``, never a string literal
   asserted against a constant's prose.
2. The ``'scope_violation'`` check was vacuous: IMPLEMENTER, DEBUGGER, and
   SIMPLE_TASK each contain that string TWICE -- once from the Scope
   Boundary section and once from the shared ``_ESCALATION_INSTRUCTIONS``
   block spliced onto all three, which ``test_roles_escalation_ladder.py``
   already guards structurally. Deleting the entire Scope Boundary section
   left the count at 1, so the check stayed green through the exact
   regression it advertised as catching. Do NOT reintroduce a
   ``'scope_violation'`` assertion in this file -- that coverage belongs to
   the escalation-ladder suite, not this one.

This revision follows the sanctioned shape instead. Each Scope Boundary
section is hoisted into a named module-level constant --
``SCOPE_BOUNDARY_GUIDANCE`` for IMPLEMENTER/DEBUGGER's byte-identical block,
``SCOPE_BOUNDARY_GUIDANCE_SIMPLE`` for SIMPLE_TASK's deliberately different,
shorter variant (it routes back to the architect instead of just naming the
escalation) -- and every assertion here is a containment check against one
of those constants, never a string literal or regex against prompt prose.
Roles are referenced BY IDENTITY (imported objects), not via a scan of
``ROLES`` or of the module source, so a fourth sandboxed role added later
requires a deliberate decision to extend this file.
"""

from __future__ import annotations

from orchestrator.agents.roles import (
    DEBUGGER,
    IMPLEMENTER,
    SCOPE_BOUNDARY_GUIDANCE,
    SCOPE_BOUNDARY_GUIDANCE_SIMPLE,
    SIMPLE_TASK,
)

# IMPLEMENTER and DEBUGGER carry the byte-identical block. Referenced by
# identity (the imported objects), not a scan of ROLES or of the module
# source, so a fourth sandboxed role added later requires a deliberate
# decision to extend this tuple.
_SHARED_GUIDANCE_ROLES = (IMPLEMENTER, DEBUGGER)


def test_shared_scope_boundary_roles_carry_the_guidance() -> None:
    """IMPLEMENTER and DEBUGGER each splice in SCOPE_BOUNDARY_GUIDANCE."""
    offenders = [
        role.name for role in _SHARED_GUIDANCE_ROLES
        if SCOPE_BOUNDARY_GUIDANCE not in role.system_prompt
    ]
    assert offenders == [], (
        f'Role(s) missing SCOPE_BOUNDARY_GUIDANCE from system_prompt: '
        f'{offenders}. This is the worktree-containment rewrite from task '
        '4370 -- restore the splice rather than leaving the role with no '
        'Scope Boundary section at all.'
    )


def test_simple_task_carries_the_simple_variant() -> None:
    """SIMPLE_TASK splices in its deliberately different, shorter variant."""
    assert SCOPE_BOUNDARY_GUIDANCE_SIMPLE in SIMPLE_TASK.system_prompt, (
        'SIMPLE_TASK is missing SCOPE_BOUNDARY_GUIDANCE_SIMPLE from its '
        'system_prompt. Unlike IMPLEMENTER/DEBUGGER, SIMPLE_TASK routes an '
        'out-of-scope write back to the architect -- restore the splice '
        'rather than substituting the shared SCOPE_BOUNDARY_GUIDANCE.'
    )
