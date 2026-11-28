"""Prompt guidance on `.task/`: which task state its links expose, and which they do not.

`orchestrator/src/orchestrator/agents/roles.py` splices `LANE_TASK_LINKS_SECTION`
once into every role whose prompt names one of those links.
`UNLINKED_TASK_STATE_GUIDANCE` is also spliced on its own into the unblock-auto
investigation prompt by
`orchestrator/src/orchestrator/dry_run_unblock.py::run_dry_run_unblock`. It
says only what is absent: each consumer follows it with where its own reader
finds that state.
"""

UNLINKED_TASK_STATE_GUIDANCE = """\
Of this task's state, the worktree's `.task/` exposes only the `plan.json` and
`iterations.jsonl` links. `metadata.json` and the `reviews/` and `verdicts/`
directories stay in orchestrator storage outside the worktree, so a read of any
of them under `.task/` fails by design and does not mean the worktree is
broken."""

LANE_TASK_LINKS_GUIDANCE = """\
The worktree's `.task/plan.json` and `.task/iterations.jsonl` are symlinks into
this task's durable storage outside the worktree. Read them, and append to
them, at those `.task/` paths: every tool follows the link, so there is nothing
to resolve, and chasing the target with `ls -la` or `readlink` gains nothing.
A link is listed even before its target exists: `.task/plan.json` until a plan
is created, and `.task/iterations.jsonl` until the first iteration is logged.
A not-found read of either therefore means that file has not been written yet,
not that the worktree is broken. Before the orchestrator hands this worktree to
the implementer, it checks that `.task/plan.json` reads this task's plan. So
once a plan exists, a missing or empty `.task/plan.json` is an infra fault:
escalate it with `category='infra_issue'` rather than searching for the file.

""" + UNLINKED_TASK_STATE_GUIDANCE + """ What you need from them is already in your
prompt.
"""

LANE_TASK_LINKS_SECTION = '\n## The .task/ links\n\n' + LANE_TASK_LINKS_GUIDANCE
