---
name: escalation-watcher
description: "Watch for and handle level-2 escalations from the dark-factory orchestrator in a long-running loop. Under the 3-tier escalation ladder (L0→per-task steward, L1→escalation-watcher-auto, L2→human), this skill is the L2 consumer. Use this skill when the user wants to monitor escalations, says 'watch escalations', 'handle escalations', 'babysit the orchestrator', or wants a long-running session to catch and triage issues that the auto-watcher couldn't resolve. Also trigger when the user starts an orchestrator run and asks you to keep an eye on it, mentions escalations piling up, or wants automated escalation handling. This is a continuous loop skill that runs until stopped."
---

# Escalation Watcher

You are running a long-running escalation watch loop. Your job is to monitor for **level-2 escalations** from the dark-factory orchestrator, handle them appropriately, and keep the development pipeline moving.

The 3-tier escalation ladder determines which agent handles each level:
- **L0** → per-task steward (handles routine agent problems automatically)
- **L1** → escalation-watcher-auto (handles steward-escalated issues; performs root-cause clustering, triage, and automated resolution where possible)
- **L2** → this skill / human (handles issues the auto-watcher judged as needing human judgement)

L2 items reach this queue via two paths: (a) **born-at-L2** — severity `critical` or `urgent` at the escalation creation chokepoint, bypassing L0/L1 entirely; (b) **promoted from L1** — the auto-watcher attempted resolution and determined human input is required, typically packaging the escalation as a causal cluster with hypothesis, evidence, and proposed options pre-formed. Default to caution over speed.

## Prerequisites

Before starting, verify these are in place. If anything is missing, ask the user — don't guess.

1. **`DARK_FACTORY_ROOT`** env var — path to the dark-factory repository (contains the `escalation` package used by the watcher)
2. **Running orchestrator** with escalation MCP accessible (port `8102` for dark-factory — set in `orchestrator/config.yaml` and matching `.mcp.json`; the code default is `8100`, which other projects may use)
3. **Escalation queue directory** at `<project_root>/data/escalations/`

Terminal discovery for spawned `/unblock` sessions is handled lazily by the `/spawn` skill — no setup is required here.

## The Main Loop

```
1. Start the watcher (background task, filtered to L2); confirm its process is alive
2. Drain pending L2 escalations — only NOW, with the watcher confirmed up (drain-after-up)
3. Handle each drained escalation
4. Wait for a wake signal: the watcher firing (it exits on the first new L2 escalation), or — if
   an auto-unblock sub-agent (B3) is in flight — that sub-agent completing. Handle whichever arrives.
5. Read the escalation from the watcher output — this is the wake signal; the drain in
   step 2 of the next pass is the authoritative source of what to handle
6. Go to 1 (restart watcher → confirm up → drain → handle)
```

The fired escalation (step 5) is just the wake; you do not handle it inline. Looping back
re-arms the watcher first, then the drain re-finds it (still pending) plus anything new — so
handling always happens with a live watcher in place and nothing slips through the gap.

### Draining pending escalations

Check for all pending L2 escalations — **compact** to keep context small:

```
mcp__escalation__get_pending_escalations(level=2, compact=True)
```

`compact=True` returns only the triage fields (`id`, `task_id`, `category`, `severity`, `level`,
`status`, `summary`, `suggested_action`, `timestamp`) and drops the heavy free-text/cluster fields
(`detail`, `members`, `options`, `root_cause`, `train_state`, …). Triage from that; fetch the full
record with `get_escalation(id)` **only** for the one item you're about to act on — and prefer doing
that full read inside the handling sub-agent (see Context Conservation). During an AFK window the
pending pile grows, and a full-dict drain every cycle is the dominant context sink — `compact=True`
is what keeps a long-running session alive.

**Drain-after-up — ordering matters.** Always (re)start the watcher and confirm its process is
alive *before* you drain, never the other way round. A pre-start drain races inotify
registration: an L2 file created in the gap between your drain and the watcher's `add_watch` is
seen by neither, and sits unhandled until some *unrelated* later escalation happens to fire the
watcher and trigger the next drain (real incident: esc-1573-8 sat 21h). Starting the watcher
first closes the gap — anything born during startup is caught by the drain that immediately
follows. This drain catches any L2 escalations that accumulated while no watcher was active.

### L2-only contract

This skill drains and waits only on **level-2 escalations**. Both the watcher subprocess and the `get_pending_escalations` draining call are filtered to `level == 2` (see details in the relevant sections below).

- **L0** is owned by per-task stewards — do not drain or handle L0 escalations here.
- **L1** is owned by escalation-watcher-auto — do not drain or handle L1 escalations here.

Never process L0 or L1 from this skill, even if explicitly asked — doing so would race with the per-task steward and escalation-watcher-auto, which own those queues and rely on their own resolution callbacks. If the user wants to handle lower-level escalations, they should invoke the appropriate skill for that level.

### Starting the watcher

```bash
cd $DARK_FACTORY_ROOT && uv run --project escalation python -m escalation.watcher \
  --queue-dir <project_root>/data/escalations --level 2 \
  [--exclude-id <esc-id>] [...] 2>&1
```

Run as a **background task** (Bash with `run_in_background`). The `--level 2` flag restricts the inotify watcher to L2 escalation files only. The watcher uses inotify and exits after the first matching L2 escalation, printing its JSON to stdout. If a matching L2 escalation is already pending when the watcher starts, it may fire immediately at launch — this is expected, not an error, and is consistent with drain-after-up ordering (the subsequent drain re-finds it).

**Re-arming over deliberately-pending items:** any L2 item you deliberately left pending (Priority 3b, `design_concern`, `risk_identified`, `infra_issue`, AFK leave-pending paths) sits in the queue and causes every subsequent watcher start to instantly re-fire on it — degenerating into a busy-loop. Pass `--exclude-id <esc-id>` (repeatable) for each such item so the initial scan and event loop skip it. `--exclude-id` also suppresses event-loop wakes from dedupe rewrites of those files (MOVED_TO events on the excluded file are silently ignored). Both the bare id form (`esc-42-1`) and the `.json`-suffixed form are accepted.

**Process safety**: only stop watcher processes you started via background task controls. Never `pkill` by pattern — other orchestrators, the user, or other sessions may have their own watchers.

### When the watcher fires

The watcher's printed JSON is just your **wake signal** — note the `id`, but you don't need to keep
the whole blob in context. Loop back, re-arm the watcher, and let the next compact drain be the
authoritative list of what to handle. Fetch the full record via
`mcp__escalation__get_escalation(escalation_id="esc-XX-N")` only for the specific item you're about
to act on — ideally inside the handling sub-agent rather than at top level.

## Priority Hierarchy

Every decision must respect this order:

### 1. System & infrastructure stability

**Hard constraints — violating these is never acceptable:**
- Never delete tasks, databases, or anything outside the project directory
- Never kill processes belonging to other orchestrators, the user, or the system
- Never directly modify `.taskmaster/tasks/tasks.json` — all task mutations go through fused-memory MCP
- If the MCP is down, ask the human for help. MCP task mutations trigger reconciliation that maintains memory quality; bypassing it silently degrades the system.

**tasks.json corruption detection:**
If tasks.json has shrunk, task IDs are mismatched/duplicated, or tasks have disappeared — this is a **critical infrastructure error**:
1. Find the orchestrator process **for this project only** — verify its command-line args reference this project's root before doing anything
2. Send SIGTERM (not SIGKILL) and let it finish gracefully
3. Tell the human immediately with full details
4. **Do NOT clean up any state** — preserve everything for post-mortem debugging
5. Wait for instructions

### 2. Software quality

Quality is king. In the long term, high quality is fast and cheap, but bugs and compounding technical debt are ruinously expensive.
- Prefer fixes that address root causes over workarounds
- Don't skip actionable suggestions just to move faster
- When in doubt about whether a suggestion has merit, err toward accepting it

### 3. Task progress

**3a — Clear-cut decisions: act decisively.** When there's one obviously correct resolution, or when multiple solutions are equally good and the choice genuinely doesn't matter for quality or velocity, resolve it and move on.

**3b — Unclear decisions that matter: ask the human.** When the best action is ambiguous AND the choice has real consequences:
- Leave the escalation pending on the queue
- Tell the human about it with full context (they may be away for hours — that's OK)
- Create a local task/todo to track the need for resolution
- Continue handling other escalations while you wait
- Periodically remind (every ~3-5 escalation cycles, not more)
- **Pass `--exclude-id <esc-id>` on every subsequent watcher (re)start** while the item is deliberately pending, so the initial scan does not instantly re-fire on it and busy-loop. The flag also suppresses event-loop wakes from dedupe rewrites of that file.

It is better to stall development than to bake in a significant bad decision.

## Merge Submissions — Bounded Submit, Then Poll

`mcp__escalation__merge_request` with `wait_secs=None` (the legacy default) blocks until the merge
worker finishes rebasing, running the full verify suite, and CAS-advancing main. On a large/slow
repo (e.g. reify) a single legacy call could take **30+ minutes** — made in the foreground it would
freeze the entire watch loop for that long: no draining, no watcher re-arm, a born-at-L2 `critical`
sits unseen (real incident: esc-2831-78 wedged a reify watcher >30 min on a direct foreground
retry-land). The watch loop's latency budget must stay bounded.

**Protocol invariant:** every `merge_request` call passes an explicit bounded `wait_secs`;
completion is awaited only via `merge_status` polling (15 s → 60 s backoff using `eta_seconds`).
Because no call can block >100 s, top-level submission is safe BY PROTOCOL.

**§7.3 Submit → poll mechanics:**

1. **Submit** with an explicit bounded `wait_secs` (use `100`):
   ```
   mcp__escalation__merge_request(
     task_id=..., branch=..., worktree=..., description=..., wait_secs=100
   )
   ```
   A return within the window yields a terminal outcome shape (`status` ∈
   `done | conflict | blocked | already_merged | unknown_branch | failed`).
   A timeout yields a non-terminal queued shape: `{status: 'queued'|'attached', request_id,
   snapshot_tip, generation, position, queue_depth, eta_seconds}`.
   Both are a **successful, durable submission** — the entry survives disconnect (PRD D2);
   intent persists even if the MCP session drops mid-bounded-wait.
   - `status='attached'` on a coalesced submission means the merge is already queued under the
     existing entry's `request_id` — already covered; do **not** re-submit.

2. **If non-terminal**, poll until resolution:
   ```
   mcp__escalation__merge_status(request_id=...)
   ```
   Back off 15 s → 60 s, using `eta_seconds` as the hint when present. Terminal states:
   `done | conflict | blocked | already_merged`. After an orchestrator restart,
   `{state: 'unknown', hint: 'check git log main'}` → fall back to `git log main` (PRD I3).

3. **To abandon** a queued entry before it is picked up:
   ```
   mcp__escalation__merge_cancel(request_id=...)
   ```
   Returns `{cancelled: bool, state, reason?}`. On success (`cancelled: true`) the entry is
   dropped without halting the queue; `merge_status` subsequently returns `state: 'abandoned'`.

**Tracking in-flight merges:** record `{task_id, escalation_id (if any), request_id}` and
never submit a second merge for a `task_id` that already has one in flight. The coalesced
`status='attached'` response is the backstop — if you see it, the merge is already covered.

## AFK Mode (extended unattended operation)

When the human will be away for an extended period (hours to days) and cannot adjudicate 3b
decisions, switch posture from "stall and ask" to "keep the pipeline moving, defer the judgement,
and leave a clean trail." Confirm AFK mode with the human if you can; otherwise infer it from an
explicit "I'll be away" or a long silence after one. Three behavioural shifts:

1. **Defer, don't wedge.** For a 3b item (ambiguous AND consequential), stalling the whole queue for
   days helps no one. Where the decision can be safely *postponed* without baking anything in:
   - Queue a follow-up task capturing the decision to be made (two-phase `submit_task` →
     `resolve_ticket`), and
   - `resolve_issue(..., action='park')` so the blocking task lands `deferred` (D2: invisible to the
     scheduler and to the stranded-blocked sweep; never circularly re-asked).
   This is parking a decision for later human review — NOT making it. Only park when the task has no
   half-merged or destructive state. The Priority Hierarchy bar still holds: better to defer than to
   bake in a bad decision — when in real doubt, fall back to "leave pending + digest."

2. **Don't spawn unattended terminals.** The interactive `/spawn` → `/unblock` path needs a human at
   a terminal; while AFK those sit idle and the task stays blocked anyway. So in AFK mode:
   - **`task_failure` / `review_issues`:** run the **low-risk auto-unblock gate** first (see the
     [Low-risk auto-unblock gate (B3)](#low-risk-auto-unblock-gate-b3) subsection for the full
     gate procedure and applicability rule). If the gate does not launch (abort / over-cap /
     already-attempted) OR the launched sub-agent aborts, leave the escalation pending and add
     it to the digest — do NOT spawn an interactive `/unblock`.
   - **`wip_conflict` / `unmerged_state` / `dependency_discovered`-with-no-task / `design_concern` /
     `risk_identified` / `infra_issue` / `recon_*`:** leave pending + digest. These need a human;
     a terminal nobody attends just clutters. Pass `--exclude-id <esc-id>` on the next watcher
     (re)start for each item left pending so the initial scan does not busy-loop on it.

3. **Batch into a digest, don't ping per-item.** Reminding "every 3-5 cycles" is noise when nobody is
   reading. Maintain a single rolling manifest at `<project_root>/data/escalations/afk-digest.md`
   (overwrite each cycle) listing every pending item: id, task_id, category, severity, age, and a
   one-line "why it's waiting / what decision is needed." On return the human reads one file. If
   phone push is configured (`--ntfy-url` on the watcher command), a born-at-L2 `critical`/`urgent`
   still pushes immediately — those are the only items worth interrupting an AFK human for.

### Low-risk auto-unblock gate (B3)

When this gate applies (see Applicability below), before leaving the item for the human, run the
mechanical gate to check whether the at-block-time dry-run investigation found a **low-risk** fix:

**Gate check** — run from `$DARK_FACTORY_ROOT`:

```bash
.venv/bin/python -m orchestrator.b3_gate check \
  --task-id <task_id> \
  --worktree <worktree> \
  --project-root <project_root> \
  --category <task_failure|review_issues> \
  --config <watched-project orchestrator config, e.g. orchestrator/config.yaml>
```

> **`--tag` note:** Both `check` and `record-launch` default `--tag` to `master` — the
> taskmaster tag under which the watched project stores its tasks. If the watched project
> uses a non-`master` tag, supply `--tag <tag>` to both verbs. Without it, `check` will
> silently find no proposal row and return `drift` or `abort` on every call (the behavior is
> fail-safe — it never launches — but the watcher will appear stuck with no signal that the
> tag was wrong).

Parse the JSON output: `verdict` (`fresh`|`drift`|`abort`), `reason`, `cap_remaining`,
`already_attempted`, `head_sha`, `main_sha`, `age_seconds`.

**Decision table:**

| Condition | Action |
|---|---|
| `already_attempted == true` OR `cap_remaining == 0` | Leave escalation pending + digest line; do NOT launch |
| `verdict == "abort"` | Leave escalation pending + digest line carrying the gate's `reason` |
| `verdict == "drift"` | Drift path (see Drift path section below) |
| `verdict == "fresh"` | Record-launch + launch (see below) |

**On `fresh` — record-launch then launch:**

Before launching, run from `$DARK_FACTORY_ROOT`:

```bash
.venv/bin/python -m orchestrator.b3_gate record-launch \
  --task-id <task_id> \
  --worktree <worktree> \
  --project-root <project_root> \
  --config <watched-project orchestrator config>
```

If `record-launch` returns `already_attempted: true` (concurrent or restart race), skip the
launch: leave pending + digest line. Otherwise it durably records the launch — this is the
don't-double-launch marker.

Then launch the **`unblock-low-risk`** skill as a NON-INTERACTIVE **background** sub-agent (the
`Agent` tool, general-purpose, **`run_in_background: true`** — NOT `/spawn`), passing `task_id`,
`escalation_id`, `project_root`, the `worktree` path, and the latest proposal, and instructing it
to read and follow `skills/unblock-low-risk/SKILL.md`. It applies the fix scoped to
`files_referenced`, runs the verify suite, and merges via the queue — or aborts cleanly.

**Background, not foreground — why.** The unblock-low-risk sub-agent runs a full
apply → verify → submit → poll cycle in its own context — verify alone can take several minutes on
a large repo. Run in the *foreground* (`Agent` without `run_in_background`), that entire cycle
occupies the watch loop's context, making it unresponsive to incoming L2 escalations throughout.
Backgrounding keeps the foreground lean and responsive: record the launch (above), then immediately
loop back to re-arm the watcher and drain. The harness re-invokes you with the sub-agent's result
when it completes — that completion is itself a wake signal (Main Loop step 4), handled below.

**Record the launch; don't double-launch.** The durable `b3_gate record-launch` call above
serializes concurrent and restart races. Stash `{task_id, escalation_id, background-task-id}` in
your context. The escalation stays `pending` until the background sub-agent resolves it, so the
*next* drain WILL re-find it — before the gate check for any task, check your context: if that
`task_id` already has an in-flight or this-cycle-completed unblock-low-risk sub-agent, skip it.
The durable rolling-24h merge cap is enforced by `b3_gate charge` inside the unblock-low-risk
sub-agent immediately before its merge-submit — a charge refusal causes the sub-agent to ABORT.
The watcher consults only `check`'s `cap_remaining` to skip launches that charge would refuse.

**Drift path — one-shot re-investigation:**

On `verdict == "drift"`, spawn ONE read-only background sub-agent running
`skills/unblock-auto/SKILL.md` in the worktree (Agent tool, general-purpose,
`run_in_background: true`), passing `task_id`, `worktree`, and the block reason. Capture the git
anchor at re-investigation start:

```bash
head_sha=$(git -C <worktree> rev-parse HEAD)
main_sha=$(git -C <worktree> rev-parse main)
```

When the sub-agent returns `{proposal_text, files_referenced, risk_label}`, build a proposal
entry mirroring `_build_entry` success-path keys and append it via
`mcp__fused-memory__update_task(id=<task_id>, project_root=<project_root>, metadata={"dry_run_proposals": [entry]}, append=true)`:

```json
{
  "proposal_text":    "<from sub-agent>",
  "risk_label":       "<from sub-agent>",
  "files_referenced": ["<from sub-agent>"],
  "block_reason":     "<original block reason>",
  "investigated_at":  "<ISO now at re-investigation start>",
  "timestamp":        "<ISO now>",
  "head_sha":         "<captured above>",
  "main_sha":         "<captured above>"
}
```

A malformed entry is fail-safe by construction — it simply fails the next `check` (`b3_gate check`
is the single shape validator). Then **re-gate once**: re-run `b3_gate check`. If `fresh` →
record-launch + launch; if `drift` again (a second drift in the same handling cycle) → leave
pending + digest (drift-reinvestigated outcome — main is moving inside the task's footprint; a
human should look). **At most one re-investigation per handling cycle.**

**Completion handling:**

On the sub-agent's **completion** (you're notified asynchronously — match the result to a recorded
launch by `task_id` / background-task-id):
- `outcome == "merged"`: it has already set the task done and resolved the escalation. Add a
  digest entry. In attended mode, also emit an immediate in-session report: one-line summary +
  merge sha + diff pointer.
- `outcome == "aborted"`: it changed nothing terminal and left the escalation pending. Keep the
  `task_id` in your context as completed this cycle (do NOT re-launch it), record the abort reason
  in the digest, and move on — do NOT retry, and do NOT spawn an interactive `/unblock` in AFK
  mode; it waits for the human. If the abort reason indicates drift/staleness and the one-shot has
  not been used this cycle, route through the drift path once.

The sub-agent re-checks the gate defensively and refuses anything not unambiguously low-risk; treat
its abort as authoritative.

**Applicability:**

B3 applies in AFK mode always. In attended mode it applies when the watched project's orchestrator
config `UnblockAutoConfig.attended_b3_enabled` (e.g. `orchestrator/config.yaml` →
`unblock_auto.attended_b3_enabled`, default `false`) is `true` OR the human enabled it for this
session via a session override. A session override wins in either direction — a human may turn it on
even if config is false, or off even if config is true.

**Digest line format** (written into `<project_root>/data/escalations/afk-digest.md` — the single
shared B3 outcome ledger for both AFK and attended modes; the "afk" prefix reflects the file's
original AFK-only scope, but it is now the unified record for all B3 outcomes regardless of session
mode; AFK shift 3 manages it):
- **Merged**: `B3 <task_id> — merged: <one-line summary> (sha: <merge_sha>)`
- **Aborted**: `B3 <task_id> — aborted: <reason>`
- **Drift-reinvestigated, second drift**: `B3 <task_id> — drift re-investigated; re-gate: drift-again → pending`
- **Drift-reinvestigated, relaunched**: `B3 <task_id> — drift re-investigated; re-gate: fresh → launched`

## Handling Escalations by Category

For every escalation, read the `suggested_action` field. It's a free-text hint — sometimes a conventional verb, sometimes natural language. First determine the escalation's **L2 origin**, then interpret the hint accordingly:

**Born-at-L2** (severity `critical` or `urgent` at creation — bypassed L0 and L1 entirely):
Neither the per-task steward nor the auto-watcher has seen this record. Read `suggested_action` as the originating agent's own annotation — a starting point, not evidence of prior triage. `investigate_and_retry` here means what it says: a retry may well succeed since no automated attempt has been made yet.

**Promoted-from-L1** (the auto-watcher attempted resolution and escalated to human):
- **`manual_intervention`** — The auto-watcher explicitly gave up. This is authoritative: the issue genuinely needs human judgment. Always respect it.
- **`investigate_and_retry`** — Misleading for promoted items. The item has already passed through *both* the per-task steward (L0) *and* the auto-watcher (L1) and persisted through their combined triage and retry budgets. Treat as a deeply persistent problem, not transient. Don't just retry.
- **`triage_suggestions` / `fix_review_issues`** — Routing hints confirming what the category tells you. No new information.
- **Free-form text** (e.g., "Restore Value::Frame from previous commits") — Valuable diagnostic context about what the escalating agent *thought* would help. Read it as a starting point for investigation, not as instructions — the agent was stuck, so its diagnosis may be incomplete.

### `review_suggestions` (info)

> **This handler is unreachable at L2.** Review suggestions reach live workflows as curator tickets
> via `_route_review_suggestions_to_curator` in workflow.py (call site ~line 3064), with no
> escalation file written; they fall back to level-0 steward escalations filed around
> workflow.py:6272 and consumed by `_next_escalation` in steward.py. They do not reach this
> queue. This stub is kept only to document why `review_suggestions` must not be re-added here.

### `review_issues` (blocking)

Blocking issues found during code review — the review cycle exhausted without the agent fixing them. The task agent is stopped.

This is distinct from `review_suggestions` (info-level, non-blocking). Review issues are real problems that prevented the task from merging.

**Spawn an interactive `/unblock` session** via the `/spawn` skill: invoke `/spawn` with `prompt="/unblock <task_id>"`, `cwd=<project_root>`, `skip_permissions=true`. Leave the escalation pending — `/unblock` resolves it when the human finishes. The human needs to see the specific blocking issues and decide how to fix them.

If the low-risk auto-unblock gate applies — see [Low-risk auto-unblock gate (B3)](#low-risk-auto-unblock-gate-b3) — try it first.

### `task_failure` (blocking)

Merge conflicts, verification failures, build breaks. The task agent is stopped and waiting.

**Spawn an interactive `/unblock` session** so the human can investigate and resolve it: invoke `/spawn` with `prompt="/unblock <task_id>"`, `cwd=<project_root>`, `skip_permissions=true`. Leave the escalation pending — the `/unblock` skill resolves it when the human finishes. Track the spawned session so you can report its status if asked.

If the low-risk auto-unblock gate applies — see [Low-risk auto-unblock gate (B3)](#low-risk-auto-unblock-gate-b3) — try it first.

### `wip_conflict` / `unmerged_state` (blocking, halt-owner)

These escalations mean the **merge queue is globally halted** — no other task can merge until exactly one of them (the "halt owner") is resolved. The orchestrator records which escalation owns the halt on the merge worker (`_halt_owner_esc_id`); resolving that specific escalation via MCP un-halts the queue. Resolving any other escalation — even another `wip_conflict` — will NOT release the halt (fixed 2026-04-19; prior code relied on a category heuristic that caused phantom-L1 bugs like esc-1888-57).

Two flavours:
- **`wip_conflict`** — the merge queue tripped on uncommitted work in `project_root`. Three sub-variants distinguishable from the `detail`:
  - WIP overlaps the merge diff (merge did not land; workflow will retry after resolution).
  - Stash pop conflicted after the merge landed (merge IS on main; WIP preserved on `wip/recovery-<task>-<ts>`).
  - Stash pop conflicted on CAS-failure path (merge did NOT land; WIP on recovery branch; task blocks).
- **`unmerged_state`** — `project_root` already had UU/AA/DD markers before the merge attempted to advance (pre-existing corruption, not caused by this merge).

**Never auto-resolve** — `manual_intervention` is authoritative. The human has to inspect `project_root`:
- For `wip_conflict`: recovery branch named in the detail preserves the user's WIP; they may need to cherry-pick or reapply before resolving.
- For `unmerged_state`: run `git status` in `project_root`; UU/AA/DD files need `git mergetool`, manual edit, or `git reset` depending on intent.

**Spawn an interactive `/unblock` session** via `/spawn` (`prompt="/unblock <task_id>"`, `cwd=<project_root>`, `skip_permissions=true`) so the human can see the recovery branch, inspect `project_root`, and resolve the escalation when finished.

**Phantom-halt check:** if the orchestrator log shows "Merge queue un-halted: halt owner &lt;esc.id&gt; resolved" but the escalation file still has `status: pending`, that is a bug — report to the human; do **not** silently dismiss. (Historical context: pre-fix, this was a common symptom of the category-match un-halt bug.)

### `scope_violation` (info or blocking)

Agent discovered it needs modules beyond its assigned scope.

1. Extend the required modules in task metadata via `mcp__fused-memory__update_task`
2. Re-pend the task — it will be dispatched with the expanded module lock set:
   ```
   mcp__escalation__resolve_issue(
     escalation_id="...",
     resolution="Scope expanded to include [modules]. Task re-pends with updated module locks.",
     action='resume',   # flips blocked→pending; task redispatches with expanded scope
     resolved_by="escalation-watcher"
   )
   ```

### `dependency_discovered` (info or blocking)

Agent found it depends on work that isn't done yet.

1. Check if the prerequisite is an **existing task** that isn't Done yet.
2. **If yes**: add the dependency via `mcp__fused-memory__add_dependency`, then re-pend — the
   dependency gate will hold the task until the prerequisite completes:
   ```
   mcp__escalation__resolve_issue(
     escalation_id="...",
     resolution="Added dependency on task <dep_id>. Task re-pends; held by dependency gate until dep completes.",
     action='resume',   # flips blocked→pending; dependency gate holds dispatch until dep is done
     resolved_by="escalation-watcher"
   )
   ```
3. **If no matching task exists**: spawn an interactive `/unblock` session via `/spawn` (`prompt="/unblock <task_id>"`, `cwd=<project_root>`, `skip_permissions=true`).

### `design_concern` (info or blocking)

Architectural or design questions. These already failed steward auto-resolution — they're genuinely ambiguous.

**Always escalate to the human:**
1. Present the concern with full context
2. Leave the escalation pending
3. Create a local task/todo to track it
4. Continue handling other escalations while waiting
5. Pass `--exclude-id <esc-id>` on every subsequent watcher (re)start while this item is pending

### `risk_identified` (info)

An agent flagged a risk during development. Risk assessment requires human judgment.

**Escalate to the human.** Tell them, track as todo, continue with other work. Pass
`--exclude-id <esc-id>` on every subsequent watcher (re)start while this item is pending.

### `cleanup_needed` (info, rarely blocking)

Technical debt or cleanup discovered during development.

- **Info**: queue as a follow-up task using the two-phase pattern:

  ```python
  suggestion_hash = hashlib.sha256(
      (escalation['detail'] or escalation['summary'] or escalation['id']).encode()
  ).hexdigest()[:16]   # Case A — escalation id already in scope; see _shared/ticket-failure-handling.md

  # Phase 1: submit — returns immediately with a ticket id
  submit_result = mcp__fused-memory__submit_task(
      project_root="<project_root>",
      title="<cleanup description>",
      description="<what needs cleaning up, with file paths and specifics>",
      priority="medium",
      metadata={
          "source": "escalation-info",
          "escalation_id": escalation_id,
          "suggestion_hash": suggestion_hash,   # (escalation_id, suggestion_hash) is the idempotency key
          "spawn_context": "steward-triage",
          "modules": ["<path/to/module>"],
      },
  )
  ticket = submit_result["ticket"]

  # Phase 2: block until the curator decides
  resolve = mcp__fused-memory__resolve_ticket(
      ticket=ticket, project_root="<project_root>",
      timeout_seconds=<see skills/_shared/ticket-failure-handling.md>
  )

  if resolve["status"] in ("created", "combined"):
      task_id = resolve["task_id"]
  elif resolve["status"] == "failed":
      # Record reason in escalation resolution note; skip this item.
      # See skills/_shared/ticket-failure-handling.md for the retryable/terminal reason matrix.
      handle_failure(resolve["reason"])
  ```

  Resolve via `mcp__escalation__resolve_issue` once the ticket resolves.
- **Blocking** (rare): spawn an interactive `/unblock` session via `/spawn` (`prompt="/unblock <task_id>"`, `cwd=<project_root>`, `skip_permissions=true`).

### `infra_issue` (blocking)

Infrastructure problems — database connectivity, MCP failures, service outages.

**Priority 1 — system stability:**
1. Tell the human immediately with full details
2. Leave the escalation pending
3. Do NOT attempt automated infrastructure fixes
4. Wait for human instructions
5. Pass `--exclude-id <esc-id>` on every subsequent watcher (re)start while this item is pending

### `recon_*` categories

`recon_failure`, `recon_backlog_overflow`, `recon_stale_run`, `recon_integrity_issue` — these are all fused-memory reconciliation problems.

Reconciliation is infrastructure that affects memory quality across the entire system. **Tell the human** with full details. Track as a todo. These may indicate systematic issues that need root-cause investigation rather than point fixes.

## Context Conservation

You're in a long-running session — conserve your context window aggressively. Over a multi-day AFK
window this is the difference between one durable session and repeated restarts.

**Read compact, expand lazily:**
- Drain with `get_pending_escalations(level=2, compact=True)` — never pull full dicts just to triage.
- Don't keep the watcher's wake-signal JSON in context; triage from the compact drain.
- Pull the full record (`get_escalation(id)`) for only the one item you're about to act on, and
  prefer doing that read inside the handling sub-agent so the heavy `detail`/`evidence` never lands
  at top level.

**Delegate to sub-agents:**
- Researching escalation context for ANY category that needs code reading (e.g. `task_failure`,
  `design_concern`): have the sub-agent fetch the full escalation, read the code/reviews, and return
  only a compact verdict + recommended action — not the raw material
- The low-risk auto-unblock sub-agent (`unblock-low-risk`) — run it in the **background**
  (`run_in_background: true`) so its full apply→verify→submit→poll cycle stays in its own context,
  keeping the watch loop lean and responsive; it returns only a small JSON result when it completes
- ANY other merge submission (e.g. retrying the land of a done-but-unmerged task) — submit
  top-level using the bounded submit→poll protocol; see "Merge Submissions — Bounded Submit, Then Poll"
- Creating follow-up tasks (once you've decided what to create, have a sub-agent do the MCP calls)

**Keep in top-level context:**
- The watch loop itself (your core job)
- Decision-making about how to handle each escalation
- Communication with the human
- Tracking pending human decisions and spawned `/unblock` sessions

When delegating, give the sub-agent complete context — paste the escalation JSON and explicit instructions. The sub-agent cannot see your conversation history or MCP state.

## Tracking Pending Human Decisions

Maintain awareness of escalations waiting for human input. When the human returns or asks for status:

1. List all pending items with brief context
2. Note how long each has been waiting
3. Prioritize: infra issues first, then blocking issues, then info-level items

Remind about unresolved items roughly every 3-5 escalation handling cycles — enough to keep them visible without being noisy.

## Resolving Escalations

**Via MCP (always prefer this):**
```
mcp__escalation__resolve_issue(
  escalation_id="esc-XX-N",
  resolution="<text injected into the agent's briefing when it resumes>",
  action='resume',   # default least-destructive intent; see C1 table below
  resolved_by="escalation-watcher"
)
```

### C1 — `action` semantics (single source of truth)

| `action` | Record disposition | Live workflow | Task status effect | Intent |
|---|---|---|---|---|
| `resume` (default) | `resolved` | resumes; resolution text injected (L0 live path) | `blocked` → `pending` (any task-attached level ≥ 1, incl. memberless born-at-L2) | "Here's the answer — continue." |
| `restart` | `resolved` | killed (soft-cancel → grace → hard) | → `pending` (from `in-progress` or `blocked`) | "This run is off-course — re-run fresh." |
| `park` | `dismissed` | killed | → `deferred` (from any non-terminal status) | "Stop; human decides later; machine must not touch." |
| `abandon` | `dismissed` | killed | → `cancelled` | "Never run again." |
| `close_only` | `dismissed` | untouched | none | "Record is noise/duplicate — change nothing." |

**C1 notes:**
- Terminal task statuses (`done`, `cancelled`) are never overwritten by any action.
- The removed `terminate` parameter now raises a hard error naming the five actions above.
- **L2 cluster cascade**: the action applies uniformly to the L2 and every member task. `queue.resolve()` cascades members via `resolved_by='l2-cascade:<L2-id>'`; the harness member callback reads the parent action from the queue read API.
- Legacy in-process callers with `resolution_action=None`: `dismiss=True` maps to `close_only`; `dismiss=False` maps to `resume`.

**Where the `resolution` text actually goes.** It reaches the working agent **only** in the L0
steward-resolved path, where a workflow is still live and waiting (`_wait_for_resolution` →
`build_resume_prompt`). That is *not* the usual L2 case. For the escalations this skill resolves:

- **L2 cluster (has member L1s), `action='resume'`:** the resolution cascades to each member L1,
  flipping the member task `blocked→pending`. It re-dispatches into a **fresh** workflow that does
  **not** read your resolution text — the harness propagates status only. Don't rely on the string
  reaching the agent. If the agent needs specific guidance, either spawn an interactive `/unblock`
  (drive the worktree directly) or write durable guidance into fused-memory / task metadata, which
  the fresh workflow's briefing memory-search may surface.
- **Memberless born-at-L2 (a direct `critical`/`urgent` blocker with no L1 members):** under D7
  (task β), `action='resume'` on a memberless born-at-L2 now flips `blocked→pending` — the orphan
  flip accepts any task-attached `level >= 1`. The resolution text is recorded for audit only and
  does not reach the agent (no live workflow); write durable guidance into fused-memory / task
  metadata instead. To re-run fresh use `action='restart'` (→ `pending` from scratch); to park for
  later use `action='park'` (→ `deferred`); to abandon permanently use `action='abandon'`
  (→ `cancelled`); to close the record without touching the task use `action='close_only'`.

Either way, still write a clear, specific `resolution` (file paths, function names, the decision and
why): it is the audit record and the human-readable trail even when no agent re-reads it.

**L2 cluster cascade (live).** When a resolved L2 represents a causal cluster (member L1
escalations packaged by the auto-watcher), resolving the L2 here cascades to close its L1 members
via the escalation server — this skill resolves only the L2 itself, never each member directly. The
action applies uniformly across the cluster. The cascade is implemented in `queue.resolve()`: it
recurses over `esc.members`, resolving each with `resolved_by='l2-cascade:<L2-id>'`, and the
auto-watcher files clusters via `promote_to_l2`. For design details, see
`plans/escalation-l2-tiering.md`.

You may still occasionally see multiple *unclustered* L2s that share a root cause — the auto-watcher
deduplicates by exact root-cause string, so near-miss hypotheses file separately. When you do, scan
them for shared files, summaries, or task IDs and handle related ones together, noting the
relationship in your resolution text.

**If MCP is unreachable:** ask the human for help. Don't try to resolve escalations by writing directly to the queue files — this bypasses callbacks and can leave the orchestrator in an inconsistent state.

## Failure Modes

**"Too many open files" (historical — no longer expected)**: Early sessions could exhaust the background-task fd pool after ~35 watcher restart cycles. This is no longer observed in practice — 100+ cycle sessions are routine. The watcher exits promptly via `sys.exit(0)`, so its inotify fd is reclaimed by the kernel and the background task is reaped shortly after. If you ever do hit it, start a fresh Claude Code session.

**Orchestrator not running**: If no new escalations arrive for an extended period, the orchestrator may have crashed or finished. Check with the human.

**Stale escalations**: On orchestrator startup, `dismiss_all_pending()` auto-dismisses **L0** escalations from prior runs (filter: `level == 0`) — **L1 and L2 escalations are preserved across restarts**. So an L2 with a timestamp from a previous session that is still `status: pending` is legitimate carry-over, not stale; handle it normally. If an escalation genuinely looks wrong (e.g. references a task that is already Done), tell the human rather than dismissing it yourself — it may contain useful diagnostic information.
