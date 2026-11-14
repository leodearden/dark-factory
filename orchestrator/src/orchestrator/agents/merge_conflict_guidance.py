"""Steward rule: resolve a merge conflict ON the queued task branch.

`orchestrator/src/orchestrator/agents/roles.py::STEWARD` splices
`STEWARD_MERGE_CONFLICT_RULE` into its Rules list as Rule 7, directly after
`orchestrator/src/orchestrator/agents/roles.py::MERGE_HALT_SAFETY_RULE`. It is
a leaf module, imported downward by roles.py.

Why the worktree HEAD matters: the merge queue lands what the queued branch's
worktree HEAD holds (`orchestrator/src/orchestrator/branch_mergeability.py`),
and `orchestrator/src/orchestrator/workflow.py::TaskWorkflow._settle_unmergeable_branch`
refuses a HEAD with nothing beyond main.

`orchestrator/tests/test_roles_merge_conflict_rule.py` checks the rule's
splice, position and category, never its wording.
"""

from escalation.categories import EscalationCategory

STEWARD_MERGE_CONFLICT_RULE: str = f"""\
7. **Merge conflicts on the task branch (`{EscalationCategory.MERGE_NEEDS_REBASE.value}` category, or any
   escalation that asks you to integrate main into the task branch).**
   The merge queue lands whatever the queued task branch's worktree HEAD holds, so the
   resolution must be a commit ON that branch. Before changing anything, record
   `git symbolic-ref --short HEAD` (the queued branch) and `git rev-parse HEAD` (the
   queued tip). If HEAD is detached, or is not the escalated task's `task/<task id>`
   branch, change nothing and do not call `resolve_issue`: re-escalate via
   `escalate_blocker(..., level=1)`, reporting the HEAD you observed. Otherwise, in the
   task worktree, run `git merge --no-commit --no-ff main`, which always stops before
   committing, even when main merges cleanly. Resolve the conflicted paths, run the
   tests, stage per the Git Staging Rules below, and `git commit`. That adds one merge
   commit above the queued tip and moves nothing else.
   - NEVER reset, rebase, amend, squash or recreate the queued branch. NEVER create a
     replacement branch. NEVER check out main or any other branch or commit in the task
     worktree. The workflow can stop waiting for you at any moment and retry the merge
     from whatever HEAD then holds, so until your merge commit exists the branch must
     still point at the queued tip.
   - NEVER end with staged-but-uncommitted work, and never bypass a hook with
     `--no-verify`. Commit or delete every untracked file you created (a ratchet-raise
     file, a scratch log). Before `resolve_issue`, confirm all of these: HEAD is still
     the queued branch; `git rev-parse -q --verify MERGE_HEAD` prints nothing;
     `git merge-base --is-ancestor <queued tip> HEAD` exits 0; and
     `git status --porcelain` prints nothing.
   - If you cannot commit (a pre-commit hook or a ratchet gate refuses the commit, the
     tests fail, the resolution needs judgement, or a check above fails), do NOT call
     `resolve_issue`, because resolving re-queues the merge onto a branch with nothing
     new to land. If `MERGE_HEAD` exists, run `git merge --abort`, which puts HEAD back
     on the queued tip. Then re-escalate via `escalate_blocker(..., level=1)`, quoting
     the refusal output verbatim and naming the conflicted paths.
"""
