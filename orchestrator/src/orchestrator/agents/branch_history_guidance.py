"""Role-prompt sections that keep a task branch's history append-only.

`BRANCH_HISTORY_GUIDANCE` is the writer section, for the roles whose commits the
harness attributes to plan steps. `REVIEWER_BRANCH_HISTORY_RULE` is the reviewer
section. `orchestrator/src/orchestrator/agents/roles.py` splices both;
`orchestrator/src/orchestrator/agents/briefing.py`'s amender prompt points at the
writer section by `BRANCH_HISTORY_HEADING`.

`orchestrator/tests/test_roles_branch_history_guidance.py` checks the splices,
never the wording.
"""

from orchestrator.git_ops import WIP_SAFETY_COMMIT_PREFIXES

BRANCH_HISTORY_HEADING = "This branch's history is append-only"

BRANCH_HISTORY_GUIDANCE = (
    """
## """
    + BRANCH_HISTORY_HEADING
    + """

Only ever add commits to this branch. Never amend, squash, reword, split or
rebase a commit already on it, including the harness's own WIP safety commits,
and even when a review suggestion asks you to. The harness records each plan
step's commit sha, and scopes the re-review after an amendment pass to the tree
diff since that pass began. It rebases this branch itself, and keeps those
records in step only for the rewrites it makes.

When a commit needs rationale its message lacks, write the rationale in the
message of a NEW commit that names the commit it explains by its subject or
plan step id. When you are committing a change anyway, put the rationale in
that commit's message. Only if nothing else needs committing, make the commit
with `git commit --allow-empty`, and never pass its sha to `mark_step_done`: it
carries none of any step's change.

Never cite one of this branch's commits by sha in a commit message. The harness
rebases this branch between iterations, before verify and on requeue, which
gives every commit above the base a new sha, so after the merge the cited sha
names a commit main does not contain. Cite the plan step id or the commit's
subject instead.
"""
)


# Sorted, so the prompt bytes do not depend on the tuple's declaration order.
_WIP_SUBJECTS = ' or '.join(
    '`' + prefix.rstrip() + '`…' for prefix in sorted(WIP_SAFETY_COMMIT_PREFIXES)
)

REVIEWER_BRANCH_HISTORY_RULE = (
    """
## """
    + BRANCH_HISTORY_HEADING
    + """

Agents may only add commits to this branch. The harness records each plan
step's commit sha, scopes a re-review to the tree diff since an amendment pass
began, and rebases the branch itself, so a commit an agent rewrites leaves those
records pointing at nothing. Never suggest amending, squashing, rewording,
splitting or rebasing a commit: a `suggested_fix` asks for a new commit instead.

A commit whose subject starts with """
    + _WIP_SUBJECTS
    + """ is the harness's own safety
commit of uncommitted work. Its message carries no rationale by construction,
so do not report that.

When a change needs rationale that no commit message and no tracked file gives,
ask for it in the message of a new commit that names the commit it explains by
its subject or plan step id.
"""
)
