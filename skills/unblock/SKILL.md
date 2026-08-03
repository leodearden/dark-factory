---
name: unblock
description: "Unblock a stuck orchestrator task — whether blocked on review issues, merge failures, or agent escalations. Use this skill when the user mentions a blocked or stuck task, wants to resolve review issues in a worktree, handle an escalation for a specific task, or says things like 'unblock task 64', 'task 107 is stuck', 'fix the review issues', or '/unblock <number>'. Also trigger when the user references a specific task number alongside words like 'blocked', 'stuck', 'failed', 'escalated', or 'review issues' — even if they don't say 'unblock' explicitly."
---

# Unblock Task

You're unblocking an orchestrator task that's stuck. The task is either **blocked** (failed reviews, merge conflicts, or verification exhaustion — task status is `blocked`, worktree preserved) or **escalated** (agent found an issue outside its scope and paused via the escalation MCP — task status may still be `in-progress`, but the agent is waiting for a resolution).

Your job: triage the issues, discuss them with the user, fix what needs fixing now, defer what can wait, and get the task either cleanly merged or its escalation resolved.

## Two competing goals

**A: Get unblocked** without ugly hacks or undue scope creep.
**B: Keep quality high** — the codebase has stringent standards.

The critical decisions are *how much to do* and *what to do now (this session) vs later (via new orchestrator tasks)*. Blockers get fixed this session. Everything else gets queued.

---

## Step 0: Locate the task

Extract the task number from the user's message. Set these for the rest of the workflow:
- `TASK_ID` — the number
- `PROJECT_ROOT` — the current working directory (where `.taskmaster/` and `.worktrees/` live)
- `WORKTREE` — `<PROJECT_ROOT>/.worktrees/<TASK_ID>/`

**Check the worktree exists.** If `.worktrees/<TASK_ID>/` is not found, tell the user:
> I can't find a worktree for task `<TASK_ID>` at `<WORKTREE>/`

and stop. There's nothing to unblock without a worktree.

**Claim the unblock lease.** Before gathering context (Step 1), claim the
`unblock-<project>#<TASK_ID>` lease (Attention Rail T7,
`orchestrator/src/orchestrator/session_registry.py`) with `warn-and-proceed` policy — unlike the
escalation-watcher's stand-down lease, a second `/unblock` on the same task is allowed to proceed,
just made visible:

```bash
python3 $DARK_FACTORY_ROOT/orchestrator/src/orchestrator/session_registry.py lease-claim \
  --name "unblock-<project>#<TASK_ID>" --slug "unblock-<project>-<TASK_ID>-$$" --pid $$ \
  --policy warn-and-proceed
```

(`<project>` is the same short project token used elsewhere for this task, e.g. the basename of
`PROJECT_ROOT`.) Parse the two printed lines (`decision=<acquired|proceed>` + message):

- **`decision=proceed` with a holder reported in the message**: surface that line verbatim to the
  user (`lease held by <session> (alive|dead, heartbeat Ns ago) — proceeding anyway`) — this is
  exactly the near-duplicate second-`/unblock`-on-the-same-task case (reify 06-28) — then continue
  normally into Step 1. Never stand down or exit; `warn-and-proceed` never blocks this session.
- **`decision=acquired`**: no prior holder; continue normally.

**Fail-soft.** A lease-substrate fault also reports `decision=proceed` (fail-open), just with no
holder to report — note it in passing and continue; a lease fault must never block an `/unblock`
session.

**Release on exit.** When this `/unblock` session ends (Step 4.5 reflect, or an early stop), release
the lease so it doesn't linger and falsely report a holder to the next `/unblock` on this task:

```bash
python3 $DARK_FACTORY_ROOT/orchestrator/src/orchestrator/session_registry.py lease-release --name "unblock-<project>#<TASK_ID>"
```

---

## Step 1: Gather context

Collect all available information in parallel:

### 1a. Worktree artifacts
Read from `<WORKTREE>/.task/`:
- `metadata.json` — task identity, base commit
- `plan.json` — TDD plan: which steps completed, which are pending, design decisions
- `iterations.jsonl` — execution log: what each agent iteration attempted and accomplished
- `reviews/*.json` — all reviewer verdicts and issues (there may be up to 5: architect, test analyst, robustness, performance, reuse auditor)

### 1b. Escalations
Query the escalation MCP:
```
get_pending_escalations(task_id="<TASK_ID>")
```

### 1c. Task status
```
get_task(id="<TASK_ID>", project_root="<PROJECT_ROOT>")
```

### 1d. Git state
In the worktree:
- `git log --oneline -10` — recent commits on the task branch
- `git diff $(git merge-base main HEAD)..HEAD --stat` (equivalently `git diff main...HEAD --stat`) — scope of changes; use the merge-base/three-dot form, not two-dot `main..HEAD`, which charges everything that landed on main since the branch base to the task branch
- Whether the branch can cleanly rebase on current main

---

## Step 2: Deep analysis

This is where you determine the nature and severity of each issue. The goal is to understand each issue well enough to classify it and recommend a course of action.

**Use an agent team.** Spawn parallel Explore agents — one per issue (or per cluster of related issues). Each agent should:

1. **Read the specific code** at the locations referenced in the issue
2. **Understand the surrounding architecture** — what module is this in, what are its responsibilities, what depends on it, what patterns does the codebase use in similar situations
3. **Consult fused-memory** for prior decisions about this area:
   ```
   search(query="<relevant architectural topic>", project_id="<project_id>")
   ```
4. **Check for related issues** in other reviewers' feedback or other tasks

### For each issue, determine:

**Architecture, code quality, or both?**
- Architecture: structural problems, wrong abstractions, missing layers, coupling
- Code quality: naming, error handling, edge cases, test coverage, style
- Both: a surface-level code quality issue that reveals a deeper architectural tension

**If it looks like pure code quality, probe deeper.** A missing error handler might mean the error path isn't designed. A naming issue might reveal confused responsibilities. A test gap might mean the interface is untestable by design. Don't take code quality issues at face value — ask whether they're symptoms of something structural.

**Clear-cut or complex?**
- Clear-cut: one right fix, no real alternatives
- Complex: genuine uncertainty about what's best — either *now*, or *in general*, or *in the long run*

**Blocker or nice-for-later?**
- Blocker: can't safely merge or continue without resolving this
- Nice-for-later: real issue, but safe to defer to the backlog

---

## Step 3: Present findings

Present the user with a structured summary. This is a decision point — they need enough context to make good calls quickly.

### Blockers (must fix before merge)

For each blocker, labeled B1, B2, ...:

> **B1: [Short title]** *[architecture | code quality | both]*
> [If clear-cut]: One or two sentences describing the fix.
> [If complex]:
> - Option A: [description] — [trade-off]
> - Option B: [description] — [trade-off]
> - Recommended: [A or B], because [reason]

### Non-blockers (recommend deferring)

For each non-blocker, labeled S1, S2, ...:

> **S1: [Short title]** — [one-sentence description]. Modules: [which modules it touches].

Recommend queuing all non-blockers. Group them by module overlap so the user can see which ones would naturally become tasks together.

### If there are pending escalations

Present each escalation with its severity, category, summary, and the agent's suggested action. Recommend whether to resolve-and-resume, terminate-and-reschedule, or fix-manually-and-merge.

### Open questions

Finish with a numbered list of every decision the user needs to make:

1. **[Question]** — Recommended: [answer]. Reason: [why].
2. **[Question]** — Recommended: [answer]. Reason: [why].

**Wait for the user.** They may approve your recommendations, override some, ask for more detail, or reclassify issues. Iterate until they say "do it" or similar.

---

## Step 3.5: Hard-abort on an explicit stop instruction

Before executing anything in Step 4, check every source gathered so far — the task
description (1a/1c), the escalation text (1b), prior turns in this session, the runbook
itself, or any instruction the user has given — for an explicit stop instruction: "do not
apply", "do not merge", "do not proceed", "do not run", "do not execute", "do not
self-authorize", "human rehearsal", "hold for human", "awaiting human", or any phrase in
the shared `orchestrator.stop_instruction.STOP_INSTRUCTION_PHRASES` family
(`orchestrator/src/orchestrator/stop_instruction.py` — the single source of truth for this
phrase set, also used by `b3_gate` and `DeterministicRunner`).

**If a stop instruction is present anywhere: HARD ABORT.** Leave the task and any pending
escalation exactly as they are (still `blocked` / still pending) for a human to act on. Do
NOT fix it yourself, do NOT resolve the escalation, do NOT merge, and do NOT narrate the
result as human-authorized — a stop instruction overrides an otherwise-safe, obviously-
correct fix. Tell the user plainly that a stop instruction was found and that you stopped
because of it.

This precondition closes reconciliation finding 0aac21b4 (task 2509): an autonomous
`/unblock` session on task 2407 self-authorized an irreversible Mem0 mutation despite an
explicit "do not apply" instruction, then narrated the result as human-authorized. Its
sibling session on task 2273, audited in the same episode, is what right looks like — it
honored its runbook's human-rehearsal mandate and was SIGTERM-killed before it could
auto-apply. This step turns that external kill into a self-halt: don't wait to be killed —
stop yourself.

A mechanical backstop exists on the autonomous path (`unblock-low-risk`'s `b3_gate check`
now hard-aborts on the same phrase family read from the task description and proposal
text — see `skills/unblock-low-risk/SKILL.md`), but the mechanical gate can only see text
already persisted to the task. This prose check is the only guard for a stop instruction
that lives solely in this session's context (a prior turn, a runbook mandate, something
the user just said) — never rely on the mechanical gate alone to catch it.

---

## Step 4: Execute

When the user approves, proceed in this order:

### 4.1: Triage non-blockers

Before queuing, check each non-blocker against this filter: **Is it in code you're already modifying for a blocker fix, AND is it trivial to resolve reliably?** Both conditions must hold. A rename in a function you're rewriting qualifies. A "missing test coverage" issue in an adjacent module does not — even if it's trivial, you're not already in that code. And a design concern in code you're touching doesn't qualify either — it's not trivial.

Non-blockers that pass this filter: fold them into the blocker fix. Note in the plan what you're picking up and why (so the user can see you're not scope-creeping).

Everything else: group into logically coherent tasks and queue them.
- Things that hit the same modules go together
- Not too large (one agent should complete it in a TDD cycle)
- Not too small (don't create a task for a single rename)
- Include specific file locations and what the reviewers flagged

```
# Phase 1: submit — returns immediately with a ticket id
submit_result = submit_task(
    title="<descriptive title>",
    description="<specific issues, file locations, reviewer references>",
    project_root="<PROJECT_ROOT>",
    priority="<medium|high>",
    metadata={
        "source": "unblock-triage",
        "spawn_context": "unblock",
        "modules": ["<path/to/affected/module>"],
    },
)
ticket = submit_result["ticket"]

# Phase 2: block until the curator decides
resolve = resolve_ticket(ticket=ticket, project_root="<PROJECT_ROOT>", timeout_seconds=<see _shared/ticket-failure-handling.md>)

if resolve["status"] == "created":
    task_id = resolve["task_id"]           # new task queued successfully
elif resolve["status"] == "combined":
    task_id = resolve["task_id"]           # folded into existing task — still counts as queued
elif resolve["status"] == "failed":
    # On `failed`: escalate the reason to the user and skip queuing this non-blocker.
    # See skills/_shared/ticket-failure-handling.md for the retryable/terminal reason
    # matrix and R4 idempotency guidance. Unblock-triage tasks don't natively set
    # escalation_id/suggestion_hash; to opt into R4 de-duplication on retry, synthesize
    # a stable pair per that doc's guidance.
    handle_failure(resolve["reason"])
```

### 4.2: Reflect on analysis

Use the reflect skill (`/reflect`) to capture:
- What issues were found and how they were classified
- Any architectural insights discovered during analysis
- The blocker/non-blocker split and rationale

### 4.3: Plan the fix

Enter plan mode. The plan covers two parts:

**Part 1: Fix all blocking issues** in the worktree, on the task branch.

**Part 2: Merge or resolve.** The procedure depends on the entry path:

*If this is a blocked task (task status is `blocked`, fixing in worktree):*

The merge procedure is iterative — don't assume one pass will be enough:

1. **Release the orchestrator's grip** on the task before merging — call `release_workflow(task_id="<TASK_ID>", timeout_secs=30)` on the escalation MCP. This soft-cancels any active workflow so the orchestrator stops processing the task while you finish it manually. Once the slot clears, if the task is still `in-progress` the tool parks it as `blocked` (returned as `parked: "blocked"`) — this both stops the orchestrator from re-dispatching it AND protects the worktree from the stranded-in-progress reconciliation sweep (the reaper) while you work. If `was_active` is False the orchestrator wasn't running it; you can skip this step in that case.
2. Rebase on main. Resolve any conflicts.
3. Run the project's full verification suite (tests, lint, type-check).
4. Fix any failures.
5. On green: rebase on main again — other tasks may have merged while you were fixing.
6. **Decide whether you must loop back to step 3, or may proceed to step 7.** "Repeat steps 3-5 until rebase is clean AND verify passes" has no termination condition and does not converge on a busy fleet — main can land a commit every few minutes while a full branch verify takes much longer, so "rebase is clean" is rarely durably true. Steps 3-6 actually fuse **two separate decisions** with two separate tests; keep them separate, because step 7's `verified_green` sits downstream of only one of them:

   **D1 — must I rebase and re-verify again?** A file-set intersection is a legitimate terminator:
   ```
   git diff <old-main-tip>..<new-main-tip> --name-only   # what landed on main since your last verify
   git diff main...HEAD --name-only                       # what your branch touches
   ```
   **Empty intersection** → you may proceed to step 7 without another loop. **Non-empty** → rebase on the new tip and loop back to step 3.

   Treating an empty intersection as a terminator requires ALL three preconditions to hold:
   1. You are **re-landing** a branch that already carries a complete, end-to-end-verified implementation. Mid-fix (still iterating on step 4's failures), re-verification is the whole point — this precondition does not hold and D1 is not in play yet.
   2. The verify you're relying on ran against **the exact tip you're about to submit**, unchanged since.
   3. The intersection is taken over the **effective gate surface**, not literal path names — a name-only diff is blind to cross-file gate coupling (shared lockfiles, test-selection config, ratcheting baseline manifests). If this project ships a "does this changed-file set require the full gate" oracle, consult it instead of raw name matching (e.g. reify's `scripts/verify-pipeline-guard.sh requires-full-gate`).

   **D2 — may I pass `verified_green=True` in step 7?** A completely different question, and **the D1 intersection is irrelevant to it** — never cite an empty intersection to justify `verified_green=True`. The governing rule is unchanged from below: `True` only if the full-scope verify ran against the exact commit being submitted, *this* iteration. **Satisfying D1 does not satisfy D2.**

   *Why it's safe to shortcut D1 at all:* for merges submitted through the merge queue (step 7), the merge worker always re-runs its own authoritative full gate against the merged result, regardless of what `verified_green` says. Skipping the local re-verify loop therefore cannot land a red `main` on that path — the cost of being wrong is a wasted queue cycle, not a broken `main`. `verified_green` feeds failure **attribution** (`INTEGRATION_SKEW` vs `BRANCH_BUG`) downstream, never merge admission.

   **That safety net has exactly one hole:** the orchestrator-down direct-merge fallback (*Immediate-response failures* below — `{"error": "Merge queue not available — orchestrator not running"}` → `git merge --no-ff` + `git push origin main`) runs no worker gate at all; nothing downstream re-checks it. If you took D1's empty-intersection shortcut and then land on that fallback, you MUST rebase onto the current main tip and re-run the full verify before `git merge --no-ff` — on that path, skipping both the loop and a fresh check is how a red `main` actually happens.

   Do not adopt this D1 shortcut as a blanket replacement for looping 3-5: it applies only when D1's three preconditions hold, and D1 holding never implies D2 holds.
7. **Invariant:** *Every `merge_request` call passes an explicit bounded `wait_secs`; completion is awaited only via `merge_status` polling.* `queued` or `attached` responses are successful submissions (durable intent), never failures.

   Submit to the merge queue with an explicit bounded wait:
   ```
   result = mcp__escalation__merge_request(
       task_id="<TASK_ID>",
       branch="task/<TASK_ID>",
       worktree="<WORKTREE>",
       description="<brief description of what landed>",
       wait_secs=100,
       verified_green=<decide per step 6's D2 — see the note below; do not default this to True>,
   )
   ```
   `wait_secs=100` equals the server's `_MAX_WAIT_SECS` clamp ceiling. A fast merge can resolve terminally inside this single bounded call; a backlogged queue returns `queued` or `attached` within ≤100 s.

   **`verified_green` is the D2 vouch from step 6 — decide it, don't assume it.** Pass `True` only if the full-scope verification suite ran and passed against the exact commit you are submitting, *this* iteration; otherwise pass `False` (or omit it). In particular: **if you reached step 7 via step 6's D1 empty-intersection terminator, pass `verified_green=False`** — D1 lets you skip the re-verify loop, which is precisely the case where no fresh verify backs this tip. Same for a resubmission after a conflict fix-up you didn't re-verify. When `True` is warranted it emits a `workflow_verify` event so a later merge failure caused by an unrelated main landing can be attributed as `INTEGRATION_SKEW` instead of degrading to `INDETERMINATE`; passing it unearned buys nothing and corrupts that attribution.

   **Caution — not retractable:** the classifier's green fact is *any-prior-green, keyed by task ID*, not scoped to the specific commit that was verified. Once `verified_green=True` has been emitted once for this task ID, a **later** resubmission for the same task (e.g. another `/unblock` pass after a conflict fix-up you re-verified via step 6's D1 path instead of a fresh full verify) can still inherit that earlier green even though this round wasn't re-verified — a genuine `BRANCH_BUG` could then be misattributed to `INTEGRATION_SKEW`. Only pass `True` when the full verify just passed, in this iteration, on the branch you're submitting now — **D1's empty-intersection terminator never satisfies D2**; they are checked independently every time you reach this step.

   **Classify the immediate response** (`merge_request` discriminates on `status`):

   - `status: "done"` or `status: "already_merged"` → **terminal success.** Thread the merge commit SHA:
     - Normal `done`: SHA is in `result["commit"]`.
     - `already_merged`: SHA is in `result["commit"]` for the fast-path case. The worker-path `already_merged` may carry `commit=None`; when `result["commit"]` is falsy, re-derive with the same exact-subject search the canonical check uses — `git log main --fixed-strings --grep="Merge task/<TASK_ID> into main" --max-count=1 --format=%H` — or, if that comes back empty, fall back to `done_provenance={"note": "merge already present on main"}`. **Do not eyeball `git log main --oneline | head -5` and pick a SHA**: it is not scoped to this task and you would record an unrelated task's merge as this one's provenance.

     Go directly to step 8.

   - `status: "superseded"` → **this submission was absorbed into a coalesced train, or replaced
     by a generation-advance resubmission, before the bounded wait returned.** Absorption
     resolves the waiting future directly (`MergeOutcome('superseded', superseded_by=train_id)`,
     `orchestrator/src/orchestrator/merge_queue.py:12703`) and `merge_request` returns that status
     verbatim (`escalation/server.py:1733`), so with `wait_secs=100` this is the *ordinary*
     absorption outcome, not an exotic one. It is always submission-scoped here — it is your own
     call's response, so none of the unscoped-handle staleness guard applies — so go straight to
     the *Polled terminal failures* `superseded` bullet below and follow `result["superseded_by"]`
     exactly as the `request_id` arm does. **Never resubmit and never direct-merge**: the
     successor is already in flight and either would race it. In particular do **not** reach for
     the orchestrator-down direct-merge rule below — the orchestrator is up, it just coalesced you.

   - `status: "queued"` or `status: "attached"` → **durable intent confirmed** — the request is enqueued; proceed to poll:
     ```
     if result["status"] == "queued":
         poll_by = "request_id"
         poll_kwargs = {"request_id": result["request_id"]}
     else:  # "attached" — pick the handle the response discloses (task 3148); a
            # missing poll_by (pre-3148 server) degrades to request_id, today's behaviour
         poll_by = result.get("poll_by", "request_id")
         if poll_by == "request_id":
             poll_kwargs = {"request_id": result["request_id"]}
         elif poll_by == "task_id":
             poll_kwargs = {"task_id": result["inflight_task_id"]}
         else:  # "branch" (pollable == False): neither handle known, another merger
                # owns the worktree; the returned request_id was never enqueued, so a
                # first-tick unknown here is NOT the "server lost its record" case below —
                # keep polling by branch and confirm via git merge-base --is-ancestor
                # (see *Polled terminal failures*) before concluding anything
             poll_kwargs = {"branch": "task/<TASK_ID>"}

     # unknown is terminal only when the polled id was actually enqueued — on the branch arm
     # nothing was, so unknown is that arm's live state until the git-authority tier resolves it.
     # superseded means this request was superseded by another one — either absorbed into a
     # coalesced train, or replaced by a generation-advance resubmission (see *Polled terminal
     # failures* below; the two need different remediation). It stays terminal on every arm
     # (dropping it here would resurrect the spin-forever bug this tuple exists to fix), but it
     # is submission-scoped ONLY on the request_id arm — on branch/task_id it is subject to the
     # same UNSCOPED-HANDLE STALENESS GUARD as every other non-done terminal below.
     terminal = ("done", "conflict", "blocked", "abandoned", "superseded") if poll_by == "branch" \
         else ("done", "conflict", "blocked", "abandoned", "unknown", "superseded")
     # 20-min hard ceiling on BOTH unscoped arms (branch and task_id): each can reject a
     # terminal `done` (see accept_terminal), and a durable tier re-serves the same stale
     # record every tick, so without a floor the loop would spin forever. request_id is
     # submission-scoped, never rejects, and keeps its unbounded wait.
     deadline = None if poll_by == "request_id" else now() + 1200
     timed_out = False
     poll_interval = 15  # seconds; ramp up to 60 s

     # UNSCOPED-HANDLE STALENESS GUARD (applies to the `branch` AND `task_id` arms).
     # Neither key is tied to *this* submission once the live-snapshot tier stops serving
     # it (e.g. a mid-flight orchestrator restart): the durable tiers resolve both to the
     # most-recent *finalized* record for that key — retention ring `get_by_branch` /
     # `get_by_task` → event store `latest_merge_finalized(branch=...)` /
     # `latest_merge_finalized(task_id=...)` (escalation/server.py:2244-2287) — and the
     # event store survives restarts. Branches AND task_ids are both reused verbatim
     # across resubmissions (this skill's own retry loop is "fix in worktree, resubmit"
     # on the same `task/<TASK_ID>`), so a previous round's done/conflict/blocked/
     # abandoned record can satisfy the terminal test on the very FIRST tick, before this
     # attempt has done anything. Only git ancestry proves that *this* attempt landed.
     def accept_terminal(poll):
         if poll_by == "request_id":
             return True          # request_id names exactly this in-flight entry — always submission-scoped
         # branch and task_id arms: both unscoped, both gated identically.
         if poll["state"] == "done":
             if poll.get("kind") == "found_on_main":
                 # Tier-3.5 git-authority response — a LIVE probe of main, reached only
                 # because the durable tiers MISSED. Structurally cannot be a stale
                 # prior-round record, so it is not what this guard defends against.
                 # Accept it directly: re-gating it on ancestry is what deadlocks a
                 # merged-and-cleaned-up branch (see the rc=128 case below).
                 return True
             # Durable-tier `done` (retention ring / event store: no `kind`, no
             # `merge_sha`) — this is the stale-record case. Confirm with git.
             return branch_on_main()   # canonical check below; not-landed → keep polling
         return True              # remaining non-done terminals (conflict/blocked/abandoned/
                                   # unknown/superseded): exit the loop, but treat as UNCONFIRMED.
                                   # For superseded specifically, do not spin here re-polling this
                                   # same key hoping for done — for a coalesce-absorbed member it
                                   # structurally never arrives (see *Polled terminal failures*
                                   # below for the real resolution: ancestry + landing signals,
                                   # not more polling).

     loop:
         sleep(poll_interval)
         poll = mcp__escalation__merge_status(**poll_kwargs)
         if poll["state"] in terminal and accept_terminal(poll):
             break
         if deadline is not None and now() >= deadline:
             timed_out = True
             break
         eta = poll.get("eta_seconds") or poll_interval * 2  # eta_seconds may be None
         poll_interval = min(max(eta, 15), 60)
     ```

     <a id="branch-on-main"></a>**The canonical ancestry check (`branch_on_main`) — three outcomes, not two.** Every "is it on main?" confirmation in this skill means *this* check. **Never use the two-way idiom `git merge-base --is-ancestor ... && echo "on main" || echo "not on main"`**: a deleted branch ref exits **128**, which that idiom silently reports as "not on main" — inverting the truth for the single most common post-merge state, since the merge lane deletes task branches on cleanup (`_delete_branch_if_on_main`, `orchestrator/src/orchestrator/git_ops.py:7538-7574`), and on the `branch` arm a *foreign* merger's cleanup deletes it out from under you.
     ```bash
     git merge-base --is-ancestor task/<TASK_ID> main; rc=$?; echo "ancestry rc=$rc"
     # The trailing `echo` is REQUIRED, not decoration. `--is-ancestor` prints
     # nothing on rc=0 OR rc=1, and the `rc=$?` assignment itself exits 0, so
     # without it the tool reports exit 0 and identical empty output for "on
     # main" and "NOT on main" -- silence you would have to guess at. Echoing
     # the numeric rc is NOT the two-outcome `&& echo` idiom banned above: it
     # prints on every path and keeps all three outcomes distinguishable. Do
     # not "tidy" it away.
     # rc=0   → landed. Accept as done/found_on_main.
     # rc=1   → genuinely not on main. Keep polling / resubmit, per the arm.
     # rc=128 → branch ref is GONE ("fatal: Not a valid object name"). This is the
     #          normal state AFTER a successful merge + cleanup — it is NOT "not on
     #          main". Search main for THIS branch's merge commit (see below):
     git log main --fixed-strings --grep="Merge task/<TASK_ID> into main" \
         --max-count=1 --format=%H
     #          Non-empty output → that SHA IS the true merge commit; landed.
     #          Empty output    → not landed (branch never existed, or never merged).
     ```
     **The rc=128 search must be the exact-subject one above — never an unfiltered `git log main --merges | head -5`.** An unfiltered listing takes no task argument, so on any repo with merge history it always prints something; "a hit" would be unconditionally true, "no hit" unreachable, and every rc=128 — *including a typo'd branch name, the wrong worktree, or a branch that was never pushed*, which all exit 128 too — would be recorded as landed with some unrelated task's merge SHA. The server's `done_provenance` backstop is only `git merge-base --is-ancestor <sha> main`, which any recent merge on main passes, so nothing downstream would catch it.

     This is the shape the in-repo authority uses — `GitOps.find_merge_marker` (`orchestrator/src/orchestrator/git_ops.py:7862-7905`), the same function `merge_status`'s git-authority tier calls on the deleted-branch path. `--fixed-strings` against the exact subject from `_merge_subject(branch, main_branch)` (`git_ops.py:1874`, canonical form `Merge <full-branch> into <main-branch>`) is what makes it substring-safe: `Merge task/1 into main` cannot match inside `Merge task/10 into main`, because the `0` falls where the pattern has a space. Do **not** substitute a bare `--grep="task/<TASK_ID>"` — that is BRE, unrestricted to merge commits, matches any commit merely *mentioning* the task, and re-opens the `task/1`/`task/10` collision. If a project overrides `git.branch_prefix` (default `task/`) or `git.main_branch`, build the subject from `_merge_subject` rather than hardcoding.

     `branch_on_main()` above returns True for rc=0 and for a *non-empty* rc=128 marker search, False for rc=1 or an empty rc=128 search.

     After the loop exits:
     - `timed_out` (either unscoped arm's 20-minute deadline reached without an accepted terminal state — i.e. the only `done` on offer never became an ancestor of main) → do NOT resubmit and do NOT direct-merge; run the [canonical ancestry check](#branch-on-main) one final time — **including its rc=128 merge-marker search**, since a branch deleted by a successful merge is the likeliest reason you got here — and stop-and-report to the human only if that too comes back not-landed, per *Polled terminal failures*'s `unknown` bullet below.
     - `poll["state"] == "done"` → **if the response carries `merge_sha`** (the git-authority tier's `kind: "found_on_main"` shape), thread it as `done_provenance={"kind": "found_on_main", "commit": "<merge_sha>", "note": "<explanation>"}` — **not** a bare `commit`. `merge_sha` is not always the merge commit: on the live-branch resolution path it's the *branch tip* SHA, a distinct commit from the actual merge commit for a `--no-ff` merge; only the deleted-branch path's `merge_sha` is the true merge-commit SHA (`_found_on_main_response`'s docstring, `escalation/server.py:2290-2305`). **Otherwise** — including on either unscoped arm (`poll_by` `"branch"` or `"task_id"`), where a durable retention-ring/event-store record resolves `done` with only `state`/`request_id`/`generation`/`outcome`/`finished_at` and *no* `merge_sha` (`escalation/server.py:2404-2420`) — `merge_status` gives you no commit hash (`poll["outcome"]` is the raw state string `"done"`), so re-derive the true merge commit from git:
       ```bash
       git log main --fixed-strings --grep="Merge task/<TASK_ID> into main" \
           --max-count=1 --format=%H
       ```
       Thread that SHA into `done_provenance={"commit": "<sha>"}`. **Do not fall back to eyeballing `git log main --oneline | head -5`** — it is not scoped to this task, so any SHA you pick from it is likely an unrelated task's merge, and the server's only provenance backstop (`git merge-base --is-ancestor <sha> main`) passes for every recent commit on main and would not catch it. If the search comes back empty, fall back to `{"note": "<explanation>"}`. Then proceed to step 8.
     - `poll["state"] in ("conflict", "blocked", "abandoned", "unknown")` → see *Polled terminal failures* below. **On the unscoped arms (`poll_by` `"branch"` or `"task_id"`) these are UNCONFIRMED** — per the staleness guard above they may be a prior round's record for this same reused branch/task_id rather than this submission's outcome. Before acting on one, re-check `mcp__escalation__get_merge_queue()` and who owns the worktree; if this branch is still in flight, keep polling to the 20-minute ceiling instead of resubmitting on a stale failure.
     - `poll["state"] == "superseded"` → **on the `request_id` arm** (always submission-scoped) follow the train/successor directly. **On the unscoped arms (`poll_by` `"branch"` or `"task_id"`) this is UNCONFIRMED** per the staleness guard above — with a further wrinkle for a coalesce-absorbed member, where that arm's `superseded` can be permanent rather than merely stale. See *Polled terminal failures* below for the full follow-the-train procedure and why ancestry plus the two landing signals there, not re-polling this same handle, is the real resolution.

   *(Immediate-response failure edges — `conflict`, `blocked`, `unknown_branch`, `failed`, orchestrator-down — plus the `superseded` absorption edge above, and cancellation, are covered below.)*

8. `set_task_status(id="<TASK_ID>", status="done", project_root="<PROJECT_ROOT>", done_provenance={"commit": "<sha>"})`
   - Pass `{"commit": "<sha>"}` when the merge landed a single commit on main — thread the SHA from `result["commit"]` for an immediate terminal response, or re-derive from `git log main` for a polled terminal response (see polled-done note above). Fall back to `{"note": "<one-sentence explanation>"}` for fast-forward or covered-by-sibling cases where no single commit applies.
9. Clean up: `git worktree remove .worktrees/<TASK_ID>` and `git branch -d task/<TASK_ID>`

**Merge-step failure and abandonment edges:**

*Immediate-response failures (from `merge_request`):*

- `status: "conflict"` or `status: "blocked"` → read `result["reason"]`, fix the conflict in the worktree, rebase on main, then **loop back to step 7** (resubmit).
- `status: "unknown_branch"` → the branch was not found by the merge queue. Verify the branch exists in this repo (`git branch`) and you are targeting the correct escalation MCP endpoint. Push the branch if needed, then loop back to step 7.
- `status: "failed"` → read `result["reason"]` and address accordingly, then loop back to step 7.
- `{"error": "Merge queue not available — orchestrator not running"}` → orchestrator is down; fall back to a direct merge (**this is the ONLY situation where a direct merge is appropriate — NEVER use it in response to `state: "unknown"`**):
  ```
  git merge --no-ff task/<TASK_ID>   # run from the main branch checkout
  git push origin main               # advance the remote ref so downstream dispatch sees it
  ```
  **No downstream gate checks this path** — unlike the merge-queue path in step 7, nothing re-verifies after `git merge --no-ff` lands. Before running it, confirm you are on the current main tip and that a full verify passed against exactly that rebased tip, this iteration. If your last verify predates any main landing since — including because step 6's D1 empty-intersection terminator let you skip a re-verify loop — rebase onto the current tip and re-run the full suite first (see step 6's D1 carve-out above). Then proceed to step 8 with the resulting commit SHA.

*Polled terminal failures (from `merge_status`):*

- `poll["state"] == "conflict"`, `poll["state"] == "blocked"`, or `poll["state"] == "abandoned"` → same fix-and-resubmit loop: fix in worktree, rebase on main, loop back to step 7. (For `abandoned`, also verify the cancellation was not intentional before resubmitting.)
- `poll["state"] == "unknown"` (orchestrator restarted or retention ring expired) → `merge_status` now self-resolves a landed merge via its git-authority tier and returns `state: "done"` with `kind: "found_on_main"` and `merge_sha` when the branch is provably on main. If `merge_status` still returns `unknown`, confirm deterministically:
  ```bash
  git merge-base --is-ancestor task/<TASK_ID> main; rc=$?; echo "ancestry rc=$rc"
  # The trailing `echo` is REQUIRED -- see the [canonical ancestry
  # check](#branch-on-main) above for why: without it, "on main" and "NOT on
  # main" print identical empty output and exit 0, indistinguishable.
  # rc=0 (on main): proceed to step 8 with done_provenance kind='found_on_main',
  #   commit=<landing sha: git log --format=%H -1 main>
  #   (git log gives the merge commit; git merge-base gives the common ancestor, NOT the merge commit)
  # rc=128 (branch ref gone — already cleaned up after a successful merge): do NOT read
  #   this as "not on main". Run the exact-subject merge-marker search from the canonical
  #   check above:
  #     git log main --fixed-strings --grep="Merge task/<TASK_ID> into main" \
  #         --max-count=1 --format=%H
  #   Non-empty → proceed to step 8 with that SHA as kind='found_on_main'.
  #   Empty     → not landed. (Do NOT substitute an unfiltered `git log main --merges`:
  #                it takes no task argument and would report "landed" for every rc=128.)
  # rc=1 (genuinely not on main) AND queue healthy: loop back to step 7 (resubmit).
  ```
  **Never fall back to direct merge in response to `unknown`** — `unknown` means the server lost its record, not that the merge failed. **This block's `resubmit` line does not apply to the `poll_by == "branch"` arm** — there nothing was ever enqueued, so `unknown` is that arm's expected live state, not a lost record; that arm never reaches this bullet as a terminal state (it's excluded from step 7's terminal set) — it arrives here only via the branch arm's 20-minute deadline, and the action there is to run the same [canonical ancestry check](#branch-on-main) once more (rc=128 marker search included) and STOP and report to the human only if it still comes back not-landed, rather than resubmitting.

- `poll["state"] == "superseded"` → this request was superseded by another one. Two distinct
  mechanisms produce that state, and they need different remediation — check `superseded_by`'s
  shape below to tell them apart. **On the `request_id` arm** (always submission-scoped) follow
  the successor directly. **On the unscoped arms (`poll_by` `"branch"` or `"task_id"`)** this is
  UNCONFIRMED per the staleness guard above — but for a **coalesce-train absorption, that arm's
  `superseded` is permanent by construction, not merely stale: it will never itself turn
  `done`.** Nothing overwrites it — the absorbed member's own `merge_finalized` record is
  written under its own branch/task keys at absorption time
  (`orchestrator/src/orchestrator/merge_queue.py:4353-4354, 4373-4377`), the train instead lands
  under a brand-new `GroupMergeRequest` that bypasses `enqueue_merge_request` via direct queue
  surgery (`orchestrator/src/orchestrator/merge_queue.py:12685-12696`), and `mark_member_done`
  (`orchestrator/src/orchestrator/harness.py:1011`) flips scheduler status without writing a
  merge record. Because the durable tiers keep serving that stale hit, Tier 3.5's git-authority
  probe — gated behind a durable-tier *miss* (`escalation/server.py:2407-2420`) — never runs to
  correct it. So for a coalesce absorption, treat the canonical ancestry check plus the two
  landing signals below as the **primary** confirmation, not a post-timeout fallback; reserve
  resuming branch-handle polling for the derail/re-drive case below, where the orchestrator
  itself re-lands or re-dispatches the member. (A generation-advance `mr-*` successor is
  simpler: it is enqueued the normal way,
  `orchestrator/src/orchestrator/merge_queue.py:4289`, so branch/task_id polling does eventually
  reflect its outcome there — see its dispatch below.) Once you are following a successor,
  **never resubmit and never direct-merge, on any arm**, while it is still unresolved — it may
  already be in flight and either would race it. `superseded_by` names one of two shapes:

  - **`mr-*` id** (generation-advance path — a plain resubmission of *this same task* at a newer
    generation; not a train, nothing absorbed, nothing to re-drive). A real request id. Poll it:
    ```
    mcp__escalation__merge_status(request_id="<superseded_by value>")
    ```
    with the same 15 s→60 s backoff. This successor is not your own submission, so bound the
    poll with its own 20-minute wall-clock ceiling rather than waiting unbounded. While it is
    still unresolved the never-resubmit rule above holds — do not act on a non-terminal poll.
    Once it reaches a terminal state, dispatch on that outcome:
    - `done` → landed. This successor merges the same single branch (no tip/train distinction),
      so the standard polled-done procedure applies directly: thread `merge_sha` if present,
      else re-derive via the exact-subject marker search for `task/<TASK_ID>` above.
    - `conflict` or `blocked` → the successor has now failed on its own terms, and nothing
      auto-retries it — `_redrive_coalesce_members` is gated on
      `isinstance(req, GroupMergeRequest)` and the train id starting with `coalesce-`
      (`orchestrator/src/orchestrator/merge_queue.py:12914, 12928`), neither of which holds for
      a generation-advance successor. Fix in the worktree, rebase on main, and resubmit — the
      standard *Polled terminal failures* remediation, loop back to step 7. (Both states are
      reachable here: this successor is an ordinary solo merge through `classify_and_merge`,
      which returns `conflict` (`merge_queue.py:5746`) and which `_map_terminal_state` passes
      through unchanged (`escalation/server.py:2194-2195`). The conflict→`blocked` collapse
      (`merge_queue.py:6339, 6357`) is inside `_do_train_merge` — train path only.)
    - `abandoned` → stop and report to the human. Do not resubmit; the resubmission may have
      been cancelled deliberately.
    - `superseded` → the successor was itself superseded (a further generation advance, or
      absorption into a train). Re-read its `superseded_by` and re-enter this bullet from the
      top against that value; the 20-minute ceiling is shared across the whole chain.
    - `unknown` → the successor's record is gone (orchestrator restarted, or the retention ring
      expired). This is terminal on the `request_id` arm from the **very first tick** — it does
      not wait for the 20-minute ceiling. Do **not** fall through to *Polled terminal failures*'
      plain `unknown` rule: that ends in "loop back to step 7 (resubmit)", and the successor may
      still be in flight, which is the double-merge race this bullet exists to forbid. Go
      straight to the branch-handle fallback below.
  - **`coalesce-*` id** (coalesce-train path) — this names the *train*, not a request. It
    resolves through none of `merge_status`'s tiers (no retention-ring alias is ever recorded
    for a train id, no event-store finalized row is keyed on one, and Tier 3.5's git-authority
    probe is skipped when only `request_id` is passed) — polling it by `request_id` returns an
    honest `state: "unknown"` that will never resolve to anything else. Do not poll it by
    `request_id`.

  For the `coalesce-*` case, or an `mr-*` poll that returns `unknown` on **any** tick (including
  the first) or is still unresolved at its 20-minute ceiling: **do not fall through to step 7's
  plain `unknown` rule** — that rule resubmits, which is exactly the race this bullet exists to
  forbid. Instead stop polling by `request_id` and fall back to the `branch` handle plus the
  [canonical ancestry check](#branch-on-main) — rc=128 exact-subject merge-marker search
  included:
  ```
  mcp__escalation__merge_status(branch="task/<TASK_ID>")
  ```
  If still in flight, keep polling the branch handle with the same backoff, **under the
  [resumed-poll terminal set](#resumed-poll) below**. Never resubmit and never direct-merge.

  <a id="resumed-poll"></a>**Resumed-poll terminal set.** Wherever this section sends you to
  branch-handle polling with a `superseded` hit already in hand — after an rc=1 ancestry read
  (the case below), or because its `superseded_by` was unpollable or never resolved — drop
  `superseded` from the terminal set for that resumed loop. The branch handle will otherwise
  re-serve the identical record on tick 1, `accept_terminal` accepts it unconditionally, and you
  bounce straight back into the bullet you came from — a ping-pong that burns the entire
  20-minute budget without ever observing a `merge_status` state change. Use:
  ```
  terminal_resumed = ("done", "conflict", "blocked", "abandoned")
  # plus: a `superseded` whose `superseded_by` DIFFERS from the one you just disregarded —
  #       that is a genuinely new absorption; re-enter the superseded bullet against it.
  # An identical `superseded`/`superseded_by` pair is the stale record: keep polling.
  ```
  **`merge_status` will never itself change for a coalesce-absorbed member** — nothing overwrites
  its `superseded` record (see above) — so `terminal_resumed` alone can starve forever even after
  the real merge lands. On every tick, alongside the `merge_status` check, also re-run the
  [canonical ancestry check](#branch-on-main): break the instant it — or, once it reaches
  rc=128-with-empty-marker, either landing signal below — reports landed. Only stop-and-report
  once ancestry (and, where reached, both signals) is still not-landed when `terminal_resumed`'s
  20-minute ceiling arrives; that final check is what "if it never lands" means below. This does
  **not** contradict `accept_terminal`'s "do not spin here re-polling this same key": that rule
  governs the *first* loop, where exiting on `superseded` is exactly what gets you to the
  ancestry check. This governs the *resumed* loop, which is driven by that check, not by
  `merge_status`'s frozen state.

  **Here an empty rc=128 marker search does NOT mean "not landed."** A coalesce train stacks its
  members linearly and merges only the **tip** branch into main (`tip_branch=tip_req.branch`,
  `orchestrator/src/orchestrator/merge_queue.py:12673`), so a non-tip absorbed member gets its
  commits onto main with **no `Merge task/<TASK_ID> into main` marker of its own** — and its branch
  is still deleted by cleanup, because it genuinely *is* an ancestor of main. rc=128-with-empty-marker
  is thus the *expected* reading for a non-tip member, which is precisely the caller this bullet
  serves; taking it as "not landed" would report a successful merge to the human as a failure and
  leave the task un-flipped. So:
  - Ancestry `rc=0` is authoritative — landed — while the ref still exists.
  - Ancestry `rc=1` (the branch ref **exists** and its commits are genuinely not on main) means
    only "not landed **yet**" — right after absorption the train (or successor) is typically
    still in flight, so this round's commits have legitimately not reached main. It is **not**
    evidence that the `superseded` hit is a stale prior-round record, and it is not a reason to
    give up: disregard the raw `superseded`/`superseded_by` value as an action signal (do not
    try to poll or follow it) and resume branch-handle polling **under the
    [resumed-poll terminal set](#resumed-poll)**, which re-derives the real answer from ancestry
    itself on every tick rather than from this frozen record. Stop-and-report only if rc=1 still
    holds at that loop's 20-minute ceiling. Never resubmit here.
  - **Only under rc=128-with-empty-marker**, do not conclude anything yet — and only here are
    signals (a) and (b) consultable at all. There are exactly **two** affirmative landing
    signals, and only these two:
    **(a)** the **tip's** merge marker on main — `git log main --fixed-strings
    --grep="Merge task/<TIP_ID> into main" --max-count=1 --format=%H` (with a
    `coalesce-<TIP_ID>-<hex>` id the tip id is readable straight off it); and
    **(b)** **this task's own scheduler status** having been flipped to `done`, which the
    orchestrator does for every absorbed member once the train lands (`mark_member_done`,
    `orchestrator/src/orchestrator/harness.py:1011`).
  - Under rc=128-with-empty-marker only, either one saying landed → the merge succeeded; proceed
    to step 8 with the train's advanced SHA as `done_provenance={"kind": "found_on_main",
    "commit": "<sha>", "note": "absorbed into train <train_id>"}`. If the task is already `done`,
    the flip happened for you — no write needed.
  - **`get_merge_queue()` no longer showing the train is NOT a landing signal.** It means only
    "stop waiting on the train," and is equally consistent with a **derail**: on any non-`done`
    train outcome the orchestrator re-pends the still-unlanded members for solo re-merge
    (`_redrive_coalesce_members`, `orchestrator/src/orchestrator/merge_queue.py:12264`), which
    also removes the train from the queue with nothing of yours on main. On queue-absence with
    neither (a) nor (b), the correct action is to **resume polling the `branch` handle** to the
    20-minute ceiling **under the [resumed-poll terminal set](#resumed-poll)** — the
    orchestrator's re-drive lands it — never to flip the task.

  Stop-and-report to the human in exactly two cases: under rc=128-with-empty-marker once both
  signal (a) and signal (b) come back not-landed, or under rc=1 once the branch-handle polling
  above has reached its ceiling. An rc=1 ancestry result is never overridden by signal (a),
  signal (b), or queue-absence — signals (a)/(b) are consultable **only** under
  rc=128-with-empty-marker.

*Abandonment (`merge_cancel`):*

To abandon a submitted merge (e.g. the task needs redesign after submission):
```
mcp__escalation__merge_cancel(request_id=result["request_id"])
```
Whether the `request_id` you received cancels the in-flight entry depends on how you got it: for `queued`, and for `attached` with `poll_by == "request_id"`, it already is the in-flight entry's id (`attached` responses with a known `inflight_request_id` set `req_id_override=dispatch.inflight_request_id`) and cancel works as described above. For `attached` with `poll_by` `"task_id"` or `"branch"`, the id you received names your own coalesced submission, not the in-flight entry — treat the cancel as best-effort (`cancelled: false` / `state: "unknown"` is the expected outcome, not evidence of a lost record). Re-check the real state by the handle `poll_by` names (`merge_status(task_id=...)` or `merge_status(branch="task/<TASK_ID>")`) before deciding whether the merge still needs abandoning.

If `merge_cancel` returns `{state: "unknown"}` on the `request_id` handle (or after the re-check above still leaves it unresolved), the entry has no live waiter in this server instance (restarted or finalized) — poll `mcp__escalation__merge_status(request_id)` first (it now self-resolves via the git-authority tier and returns `state: "done"` / `kind: "found_on_main"` / `merge_sha` when the branch is provably on main). If `merge_status` still returns `unknown`, confirm deterministically:
```bash
git merge-base --is-ancestor task/<TASK_ID> main; rc=$?; echo "ancestry rc=$rc"
# The trailing `echo` is REQUIRED -- see the [canonical ancestry
# check](#branch-on-main) above for why: without it, "on main" and "NOT on
# main" print identical empty output and exit 0, indistinguishable.
# rc=0 (on main): treat as done; proceed to step 8 with done_provenance kind='found_on_main'
# rc=128 (branch ref gone after a successful merge + cleanup): NOT the same as rc=1 —
#   run the merge-marker search from the canonical check above before concluding anything
# rc=1 (not on main): the merge did not land; decide whether to resubmit or discard
```
Never fall back to direct merge in response to `unknown` — `unknown` means the server lost its record, not that the merge failed.

*If this is an escalated task (pending escalation, agent is paused):*

Choose one of these based on the analysis:

- **Resolve and resume** — if your blocker fixes address the escalation concern, resolve with actionable instructions for the resumed agent:
  ```
  resolve_issue(escalation_id="<id>", resolution="<specific instructions>", action='resume', resolved_by="interactive", resolution_turns=<N>)
  ```
  The agent resumes with your resolution injected into its briefing. Task stays `in-progress`.

- **Restart and reschedule** — if the task needs fundamental redesign:
  ```
  resolve_issue(escalation_id="<id>", resolution="<reason for restart>", action='restart', resolved_by="interactive", resolution_turns=<N>)
  ```
  Then create or update tasks as needed. Task goes to `pending`.

**Turn counting:** `<N>` is the number of user messages since this skill was invoked (count each time the user sent a message, starting from the `/unblock` invocation). This tracks how much human attention the resolution required. If you lose count, estimate conservatively.

- **Fix manually and merge** — if you fix the issue yourself in the worktree, follow the blocked-task merge procedure above.

### 4.4: Execute the plan

Exit plan mode and execute. **Keep the task in its current status during the work** — don't manually change it until you've successfully merged or resolved. This prevents the orchestrator from trying to start new agents on it. One deliberate exception: in the blocked-task merge procedure, `release_workflow` intentionally moves an escalated/in-progress task to `blocked` (the reaper-immune holding state) when it releases the slot — that's the safe status to work from, and the final `set_task_status(done)` after merge is then the normal blocked→done transition.

### 4.5: Final reflect

**The task is not done until this step completes.** After the merge succeeds (or escalation is resolved) and the task status is updated, invoke `/reflect` to capture:
- What was fixed and how
- Any decisions made during the fix (e.g., chose approach A over B because...)
- Architectural insights that surfaced during the work
- Brief summary: what was accomplished, what was deferred (with task numbers)

This is the last step. Do not consider the unblock workflow complete until reflect has run.

---

## End states

After this skill completes, the task should be in one of these states:

| Starting state | Outcome | Final state |
|---------------|---------|-------------|
| Blocked | Successfully merged to main, verification green | Done |
| Blocked | Needs redesign, can't fix this session | Pending (update task description first) |
| In Progress (escalated) | Escalation resolved, agent resumes | In Progress |
| In Progress (escalated) | Escalation terminated, work rescheduled | Pending |
| In Progress (escalated) | Fixed manually, merged to main | Done |

---

## Project-specific verification

The verification commands depend on the project. Check `orchestrator.yaml` in the project root for `test_command`, `lint_command`, and `type_check_command` overrides. If there's no override, use whatever the project's standard tooling is (check for Cargo.toml, package.json, pyproject.toml, etc.).
