"""Regression guard for task 4370: the sandboxed roles' `## Scope Boundary`
sections must not promise an OS "permission error" for an out-of-scope write.

Under the ratified whole-worktree sandbox granularity (PRD D1,
plans/os-sandbox-worktree-containment-prd.md decision D1), the sandbox only
blocks writes OUTSIDE the task worktree (the main checkout, sibling
worktrees, other tasks' `.task-meta/`, `refs/heads/main`, `~/.claude`).
Every file INSIDE the worktree is writable -- including files never listed
in the plan -- and nothing detects or blocks an in-worktree write outside
the plan's declared file set. Before this task, IMPLEMENTER, DEBUGGER, and
SIMPLE_TASK's `## Scope Boundary` prose told the agent the opposite: that an
out-of-scope write "will get a permission error". An agent that believes
that has no reason to self-report via `escalate_blocker`, so an ungranted
scope widening silently lands (measured instance: task 3143, commit
3244953ff7, see task 4370's description).

Per `test_roles_escalation_ladder.py`'s rule for this exact class of test,
assertions here are containment checks against the three role objects
imported BY IDENTITY (not a scan of `ROLES` or of the module source), so a
fourth sandboxed role added later requires a deliberate decision to extend
this test rather than being silently covered or silently skipped.
"""

from __future__ import annotations

from orchestrator.agents.roles import DEBUGGER, IMPLEMENTER, SIMPLE_TASK

# The three roles named in task 4370 as carrying the false claim. Referenced
# by identity (the imported role objects), not by name lookup into ROLES.
_SANDBOXED_SCOPE_ROLES = (IMPLEMENTER, DEBUGGER, SIMPLE_TASK)

_FALSE_CLAIM_SUBSTRING = 'permission error'


def test_scope_boundary_no_longer_promises_a_permission_error() -> None:
    """THE FIX: none of the three prompts claim an out-of-scope write errors.

    This is the discriminating assertion: it must fail against the
    pre-fix IMPLEMENTER/DEBUGGER text ("you will get a permission error...")
    and the check is written to catch it via a real string comparison, not a
    vacuous `'x' in prompt` tautology -- `_FALSE_CLAIM_SUBSTRING` is exactly
    the phrase the original prose used.
    """
    offenders = [
        role.name for role in _SANDBOXED_SCOPE_ROLES
        if _FALSE_CLAIM_SUBSTRING in role.system_prompt
    ]
    assert offenders == [], (
        f'Role(s) still promise an OS permission error for an out-of-scope '
        f'write: {offenders}. Under the whole-worktree sandbox (PRD D1), '
        'every file inside the worktree is writable and no per-file plan '
        'scope is enforced -- rewrite the "## Scope Boundary" section to '
        'describe worktree containment instead.'
    )


def test_scope_boundary_still_names_scope_violation() -> None:
    """The self-report path (`escalate_blocker(category='scope_violation')`)
    must remain the documented recourse -- it is now the ONLY mechanism that
    can catch an out-of-scope write, so removing the pointer to it would
    leave agents with no instructed path to request scope expansion.
    """
    offenders = [
        role.name for role in _SANDBOXED_SCOPE_ROLES
        if 'scope_violation' not in role.system_prompt
    ]
    assert offenders == [], (
        f"Role(s) no longer name 'scope_violation' anywhere in their "
        f'system_prompt: {offenders}. This is the only self-report channel '
        'for an out-of-scope write under the whole-worktree sandbox; keep '
        "directing agents to `escalate_blocker(category='scope_violation')`."
    )
