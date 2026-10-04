# Eval replay frame (task 4844)

This is the normative home of the decision to brief an eval architect with an
**honest replay frame**. The code that applies the frame points here:
`orchestrator/src/orchestrator/evals/replay_frame.py`,
`orchestrator/src/orchestrator/evals/metrics.py::EvalMetrics.replay_frame`,
`orchestrator/src/orchestrator/evals/runner.py` and
`orchestrator/src/orchestrator/cli.py::_run_plan_only`.

## 1. Problem

An architect eval replays a historical task. It checks out the fixture's
`pre_task_commit` and asks the architect to plan it. The replay leaked the
future in three ways:

- `orchestrator/src/orchestrator/evals/snapshots.py::create_eval_worktree` runs
  `git worktree add --detach <path> <pre_task_commit>` from the live checkout.
  The eval worktree shares the live object DB and refs, so `main`, the reflog
  and every later commit are reachable from the historical HEAD.
- The production ARCHITECT system prompt
  (`orchestrator/src/orchestrator/agents/roles.py::ARCHITECT`) tells the agent to
  verify premises before planning. It teaches `git log --all` to find landed
  work, and `report_task_already_done`, `report_blocking_dependency` and
  `report_false_premise` as exits.
- The eval briefing was byte-identical to production. Nothing told the agent it
  was in a replay.

Tranche 1 (2026-08-26): 47 of 53 architect cells produced no plan. Forensics
over the first 51 cells found that every no-plan cell (44) ended in an explicit,
server-accepted decline, and every decline checked was adversarially verified
TRUE in the present. The worst case is `df_task_2260`, whose base is the first
parent of its own landing merge `97059b1dd8`. The architect that investigated
honestly was required by its role contract to report the task already done. The
headline planRate was measuring willingness to plan a moot task, not planning.
The forensic record lives only in the main checkout, untracked:
`data/eval-campaign/tranche1/investigation/FINDINGS.md`.

## 2. Decision: option B, HONEST FRAME (`honest-frame-v1`)

The eval briefing gains an appended section, built by
`replay_frame.py::build_replay_frame_block(base_commit)`. It tells the
architect:

- this is a replay, and the base commit stands in for `main` at dispatch time;
- refs, objects and live MCP state not reachable from HEAD are post-base
  information, not evidence about the task's premises;
- how to stay in frame: `git log HEAD -- <path>` instead of `git log --all`,
  `git merge-base --is-ancestor <commit> HEAD` before citing a commit, and the
  worktree files themselves;
- the decline exits remain correct when the IN-FRAME evidence supports them;
- otherwise, plan as if `main` were at the base.

Why B:

- It is cheap: one appended string per cell.
- Option A, TRUE REPLAY (a truncated clone, or hiding refs per cell), costs disk
  and time on every cell. It is also incomplete on its own: live MCP (task store,
  memory, escalations) exposes the same future, and cutting that is task 4757,
  still pending.

What B costs, stated honestly:

- It asks the model to ignore evidence it can see. A model that does not follow
  the frame still leaks.
- The eval briefing now diverges from production. The divergence is limited to
  an APPENDED section, so the production prefix is byte-identical, and the cell
  is stamped so a readout can tell framed cells apart.

Rejected: suppressing the decline protocols ("never decline in an eval"). Some
tranche-1 declines were correct in frame too, for example `reify_task_3883`,
whose premise was false at its base. Forbidding declines would score a correct
refusal as a failure and teach the opposite of the role contract.

## 3. Enforcement

- `runner.py::run_architect_eval` appends `build_replay_frame_block(pre_task_commit)`
  to the production assembler's prompt.
- `orchestrator eval --plan-only` (`cli.py::_run_plan_only`) appends it the same
  way, per fixture, because one assembler is shared across fixtures with
  different bases.
- `runner.py::run_end_to_end` (the both-live matrix/confirm path) passes
  `replay_frame.py::ReplayFramedBriefingAssembler` through `build_workflow`'s
  `briefing` parameter. TaskWorkflow's PLAN phase calls its
  `build_architect_prompt`, which returns the production prompt plus the block.
  It overrides nothing else.

Load-bearing constraint: `orchestrator/src/orchestrator/agents/briefing.py` and
`orchestrator/src/orchestrator/workflow.py` stay byte-identical to production.
Production dispatch never constructs the subclass, so it can never be framed.

## 4. Reading `metrics.replay_frame`

| value | meaning |
|---|---|
| `'honest-frame-v1'` | the cell's live architect was briefed in this frame |
| `None`, or key missing | unframed: a cell from before task 4844, or `run_eval`'s frozen-plan implementer path, which has no live architect |

The stamp names the REGIME the harness ran the cell under. It is written on
every `run_architect_eval` and `run_end_to_end` cell, including harness-error,
cap-tainted and collect-metrics-failed cells; whether the prompt was delivered
is what `cap_tainted` and `invocation_error` say.

A readout that mixes framed and unframed cells must split on this field. Bump
`REPLAY_FRAME_ID` whenever the block text changes materially, so one id always
means one briefing.

## 5. Deliberately still unfixed

- Live MCP visibility and writes from eval agents: task 4757.
- TaskWorkflow's follow-up architect prompts are not framed: plan completion,
  revalidation, repair, replan and plan tightening. They operate on a plan that
  already exists, and the tranche-1 declines were reached from the PLAN-phase
  prompt.
- The simple-task route, and the ARCHITECT system prompt itself (production
  text, still teaching `git log --all`). The frame overrides it in the user
  prompt for the replay only.
- `--plan-only` writes plans into fixture JSON without a frame stamp.

Moving toward option A must be recorded here, with the measurement that
justified it.

## 6. Demonstration

**The live cell was not run to completion.** On 2026-10-04 the task 4844
implementer tried ONE cell from its sandboxed task worktree, with the code at
commit `990e6a3a64`: `run_architect_eval` on
`orchestrator/src/orchestrator/evals/tasks_hard_v2/df_task_2260.json`, config
`EvalConfig('architect-sonnet-replay-demo', 'claude', 'sonnet', 'high',
role='architect', max_budget_usd=3.0)`, `timeout_override=30`. It failed
before the architect was invoked, while creating the eval worktree. No run id
was persisted, no metrics were produced and no money was spent. The raw error:

```
RuntimeError: Command git worktree add --detach /home/leo/src/dark-factory-eval-worktrees/df_task_2260/run-19aacddd 20c934ca597c7c9e3e2eb79f572970a8099f9243 failed (rc=128): Preparing worktree (detached HEAD 20c934ca59)
fatal: could not create directory of '.git/worktrees/run-19aacddd': Permission denied
```

The task-worktree sandbox forbids writes to the shared `.git` directory, and
`create_eval_worktree` must register a new worktree there. A live cell has to
be run from an unsandboxed session.

Until then, the hermetic tests are the deterministic demonstration that the
architect's briefing carries the frame and that every cell is stamped
`honest-frame-v1`:

- `orchestrator/tests/test_eval_architect.py::TestArchitectCellCarriesReplayFrame`
- `orchestrator/tests/test_eval_driver.py::TestEndToEndCarriesReplayFrame`
- `orchestrator/tests/test_eval_replay_frame.py::TestPlanOnlyCliAppendsReplayFrame`

When a live cell does run, record here its date, commit, run id, and the
`replay_frame`, `terminal_kind`, `plan_steps`, `plan_quality`, `cost_usd` and
`invocation_error` values, verbatim. A decline is an acceptable result if it is
grounded in in-frame evidence.
