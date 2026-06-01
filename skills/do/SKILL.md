---
description: "Hand the direction just agreed in this conversation off to a fresh context for autonomous execution. ONLY runs when the user explicitly types /do — never auto-invoke it. Enters plan mode, distills the whole preceding discussion into a self-contained plan, and appends the user's fixed execution protocol (implement in a worktree → /merge-queue → /reflect) so the plan runs end-to-end after 'Clear Context and Follow Plan'."
argument-hint: "[what to do — omit to use the direction just agreed in this conversation]"
model: opus
---

# /do — package this decision, then execute it in a fresh context

You've just finished a freeform discussion: investigating, weighing alternatives, picking a direction, resolving open questions. The user now wants that direction **carried out autonomously in a clean context**, following their standard recipe: build it in a worktree, merge it, reflect.

**What to do:** $ARGUMENTS

> If the line above is empty, the task *is* whatever direction was just agreed upon in this conversation. Infer it from the discussion — don't ask the user to restate it.

## Why this command exists

The user is about to choose **"Clear Context and Follow Plan."** When they do, **everything in this conversation disappears except the plan file.** The alternatives you weighed, the decision you reached, the constraints you uncovered, the reason you rejected the other approach — the fresh executor will see *none* of it unless you write it down.

So your real job here is not to type boilerplate. It is to **compress this entire session's hard-won context into a plan that a competent agent with zero memory of this conversation can execute correctly without relitigating anything.** A plan that just says "do X" throws away the most valuable thing the discussion produced: *why* X, and *why not* the alternatives.

## Steps

### 1. Enter plan mode
Call `EnterPlanMode` now. You're switching to read-only: you'll research and write the plan, not change anything yet. (This replaces the manual Shift+Tab switch.)

### 2. Resolve anything still genuinely open — now, not later
If a material decision is still unsettled — something that would actually change the approach — settle it with the user via `AskUserQuestion` **before** you write the plan. Once context clears, the executor can't ask you, so an unresolved question silently becomes a guess. Don't manufacture questions to fill this step; only surface real forks the discussion left open.

### 3. Do any remaining research
Lean on what this conversation already established. Read only what you still need to make the plan concrete and correct — exact file paths, function names, current behavior, test commands.

### 4. Write the plan to the plan file
Write it to the plan file named in the plan-mode system message. Write for a reader who saw **none** of this conversation. Use this structure:

- **Objective** — what we're changing, fixing, or building, in concrete terms.
- **Decisions & rationale** — the direction chosen and *why*; the alternatives considered and why they were rejected (briefly, so the executor doesn't reopen them); the constraints and assumptions this discussion resolved. **This is the part that is otherwise lost forever — spend the most care here.**
- **Implementation** — ordered, concrete steps. Name files and functions wherever you already know them.
- **Verification** — how to confirm it actually works: which tests / lint / type-check to run, or what behavior to observe.
- **Execution protocol** — include this, adapted to the task. It is the user's fixed recipe and must live *inside* the plan, because the prompt that asked for it won't survive the context clear:
  1. Do all the work in a **git worktree** — call `EnterWorktree`. Name it so the resulting branch follows the `task/<short-slug>` convention `/merge-queue` expects; rename the branch if EnterWorktree's default doesn't match.
  2. Implement the plan in the worktree and run the verification above until it passes.
  3. Run **`/merge-queue`** to land the branch on main. It auto-detects whether the orchestrator is running and routes the merge safely either way.
  4. Run **`/reflect`** to capture the decisions and discoveries from the implementation to memory.
  5. Work **autonomously** — this plan is the contract. Don't pause for confirmation unless you hit something genuinely blocking that the plan doesn't cover.

### 5. Exit plan mode for approval
Call `ExitPlanMode`. Populate its `allowedPrompts` with the Bash categories the executor will need so it isn't permission-blocked after the handoff — typically: run tests, run lint / type-check, git operations, create a worktree, install dependencies.

The user then chooses "Clear Context and Follow Plan," and the fresh context runs the plan from top to bottom.
