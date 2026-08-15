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

## Launching this watcher (default lane: tmux)

This is a long-running loop session (PRD `plans/fleet-cockpit-prd.md` §3 fork 1) — launch it in
the crash-survivable, reattachable **tmux lane** by default: spawn with `CLAUDE_SPAWN_BACKEND=tmux`
(see `skills/spawn/spawn-claude.sh`'s header) so the session gets a `display.kind=tmux`
session-registry record and a `tmux attach`-reattachable window whose record persists across a
crash. Interactive one-off skills this watcher spawns (e.g. `/unblock` sessions) stay as ordinary
WM terminal windows, unchanged.

## Claiming the Watcher Lease (single-owner-per-role)

**Before entering the Main Loop for the first time**, claim the `watcher-<project>` lease (Attention
Rail T7, `orchestrator/src/orchestrator/session_registry.py`). This is a deterministic,
single-owner-per-role replacement for any pgrep/ps-tree archaeology to detect a duplicate watcher —
run once, at session startup, not on every loop iteration:

```bash
# Reap anything stale first so a genuinely-dead prior holder never blocks this claim.
python3 $DARK_FACTORY_ROOT/orchestrator/src/orchestrator/session_registry.py lease-reap

# Claim watcher-<project> (e.g. watcher-df) — STAND_DOWN policy: a live duplicate wins the lease
# and this session must exit rather than run a second watch loop against the same project.
python3 $DARK_FACTORY_ROOT/orchestrator/src/orchestrator/session_registry.py lease-claim \
  --name watcher-<project> --slug watcher-<project>-$$ --pid $$ --policy stand-down
```

Parse the two printed lines: `decision=<acquired|stand-down|proceed>` followed by a human-readable
message.
- **`decision=stand-down`**: print the message verbatim (`lease held by <session> (alive, heartbeat
  Ns ago) — standing down`) and **exit immediately** — do not start the watcher, do not drain.
- **`decision=acquired` or `decision=proceed`**: continue into the Main Loop below. `proceed` is the
  fail-open outcome (see below) and is handled identically to `acquired`.

**INTERACTIVE-ONLY.** This lease claim belongs to the interactive L2 watcher (this skill) only. The
headless `escalation-watcher-auto` rotation (L1) never claims or contends this lease — it has no
`lease-claim` call site at all, by design (it is a supervised, always-on rotation, not a
single-owner-per-session actor). If you are running as `escalation-watcher-auto`, skip this section
entirely.

**Fail-soft (fail-open).** A lease-substrate fault (disk error, unwritable `~/.claude/fleet/`, …) is
logged loudly by `session_registry` and reported back as `decision=proceed` — never a false
`stand-down`. A lease fault must never block a watch session from starting.

**Heartbeat + release.** Once claimed, touch the lease every Main Loop cycle (see "Starting the
watcher" below) so it never appears stale to another session's claim attempt, and release it when
the watch session ends (clean exit, or the human stops it):

```bash
python3 $DARK_FACTORY_ROOT/orchestrator/src/orchestrator/session_registry.py lease-release --name watcher-<project>
```

## The Main Loop

```
1. Start the watcher (background task, filtered to L2); confirm its process is alive
2. Drain pending L2 escalations — only NOW, with the watcher confirmed up (drain-after-up)
3. Handle each drained escalation
4. Wait for a wake signal: the watcher firing (it exits on the first new L2 escalation), or — if
   an auto-unblock sub-agent (B3) is in flight — that sub-agent completing. Handle whichever arrives.
5. Read the escalation from the watcher output — this is the wake signal; the drain in
   step 2 of the next pass is the authoritative source of what to handle
6. Run `reap-decisions` to close any parked DecisionRecord whose escalation has since resolved
   (see "Filing Parked Decisions to the Cockpit Registry" below) — once per cycle
7. Go to 1 (restart watcher → confirm up → drain → handle)
```

The fired escalation (step 5) is just the wake; you do not handle it inline. Looping back
re-arms the watcher first, then the drain re-finds it (still pending) plus anything new — so
handling always happens with a live watcher in place and nothing slips through the gap.

### Draining pending escalations

Check for all pending L2 escalations — **compact** to keep context small:

```
mcp__escalation__get_pending_escalations(level=2, compact=True)
```

`compact=True` returns the triage fields (`id`, `task_id`, `category`, `severity`, `level`,
`status`, `summary`, `suggested_action`, `timestamp`) plus the triage-ack annotation fields
(`triaged_at`, `triaged_by`, `triage_note`, `updated_at` — see "Reading a triage-ack annotation"
below), and drops the heavy free-text/cluster fields (`detail`, `members`, `options`, `root_cause`,
`train_state`, …). Triage from that; fetch the full record with `get_escalation(id)` **only** for
the one item you're about to act on — and prefer doing that full read inside the handling sub-agent
(see Context Conservation). During an AFK window the pending pile grows, and a full-dict drain every
cycle is the dominant context sink — `compact=True` is what keeps a long-running session alive.

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
cd $DARK_FACTORY_ROOT && scripts/watcher-rearm.sh \
  --queue-dir <project_root>/data/escalations --level 2 --timeout 3600 [--baseline] 2>&1
```

Run as a **background task** (Bash with `run_in_background`). `scripts/watcher-rearm.sh` is the
canonical bounded-wait + re-arm wrapper around `escalation.watcher` shared by this skill and
escalation-watcher-auto. `--level 2` restricts the inotify watcher to L2 escalation files only; the
watcher exits after the first matching L2 escalation, printing its JSON to stdout. If a matching L2
escalation is already pending when the watcher starts, it may fire immediately at launch — this is
expected, not an error, and is consistent with drain-after-up ordering (the subsequent drain
re-finds it). The wrapper preserves the underlying watcher's exit code (`0`=fired, `124`=timeout)
and emits a `WATCHER_REARM_OUTCOME: <FIRED|CEILING|KILLED|ERROR> exit=<rc>` line to **stderr** on
every run — do NOT pipe `2>&1` when you parse stdout as the escalation JSON, or you'll corrupt the
parse.

**Bash-tool timeout contract:** the wrapper blocks for up to `--timeout` seconds per slice before
returning, and **every** call — background *and* foreground — must carry an explicit Bash-tool
`timeout` parameter sized to at least `(--timeout + 60s) × 1000` ms — e.g. `timeout: 3660000` for
`--timeout 3600`. `run_in_background: true` does **not** exempt a call from the harness timeout:
measured 2026-08-10, background arms launched with no `timeout` parameter were killed after ~116s
(≈ the 120000ms Bash default) against a configured `--timeout 540` slice, so the slice length was
never the constraint; re-arming the identical command with `timeout: 3660000` survived 6m32s and
exited cleanly with `WATCHER_REARM_OUTCOME: FIRED exit=0`. With that parameter passed, use the long
canonical slice: `--timeout 3600` yields at most one heartbeat wake per hour (`CEILING`) while a
real L2 escalation still fires instantly via inotify. A short slice buys no protection — the old
`--timeout 540` merely forced a wake-notify-rearm turn every 9 minutes. A slice that long requires
`BASH_MAX_TIMEOUT_MS` ≥ that value in the settings env (dark-factory onboarding provisions it —
see `skills/factory-init`). Only on a machine WITHOUT that setting does the harness's 600000ms
(10 min) cap apply: there, cap the slice at `--timeout 540` and set the Bash tool's `timeout`
**≥ 600000ms** so the slice can return before the harness kills it.

**Diagnostic — watcher dies at ~2 minutes with no outcome line:** if an arm disappears after
roughly 120s and stderr carries **no** `WATCHER_REARM_OUTCOME` line, the Bash `timeout` parameter
was omitted. The wrapper emits that line on every exit path including its own timeout, so its
absence means the exit handler never ran — an external SIGKILL to the process group, not a wrapper
or watcher fault (the 07-09 exit-143 failure mode the wrapper exists to bound; it cannot bound a
kill of the whole group). This hides during a backlog drain, where every arm FIRES within seconds
and finishes well inside the 120s window; the kills only start once the queue is fully triaged and
the watcher genuinely waits, leaving a loop that looks armed but reaps itself every ~2 minutes.

**Re-arming over deliberately-pending items:** any L2 item you deliberately left pending (Priority
3b, `design_concern`, `risk_identified`, `infra_issue`, AFK leave-pending paths) sits in the queue
and would cause every subsequent watcher start to instantly re-fire on it — degenerating into a
busy-loop. Append `<esc-id>` (one per line; both the bare `esc-42-1` and `.json`-suffixed forms are
accepted) to the wrapper-owned exclude-file instead of hand-maintaining a growing `--exclude-id`
list — the wrapper always wires `--exclude-file` into the watcher invocation and re-reads it every
poll, so an append takes effect without restarting the watcher, and it also suppresses event-loop
wakes from dedupe rewrites of that file. The resolved path (default
`<queue-dir>/.watcher-rearm-exclude-l2`, or your `--exclude-file` override) is printed to stderr on
every wrapper run; pass `--check` for a dry-run print of the resolved path without starting the
watcher.

**Process safety**: only stop watcher processes you started via background task controls. Never `pkill` by pattern — other orchestrators, the user, or other sessions may have their own watchers.

**Lease heartbeat (each cycle):** each time you (re)start this watcher subprocess (Main Loop steps 1
and 7), also touch the `watcher-<project>` lease claimed at session startup (see "Claiming the
Watcher Lease" above):

```bash
python3 $DARK_FACTORY_ROOT/orchestrator/src/orchestrator/session_registry.py lease-heartbeat --name watcher-<project>
```

This is what makes a second session's `lease-claim` observe this session as "alive" and stand down —
there is no need to separately pgrep/ps-tree for other watcher processes.

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
4. File a DecisionRecord via `write-decision` (see "Filing Parked Decisions to the Cockpit Registry" below) — IN ADDITION to telling the human directly
5. **Do NOT clean up any state** — preserve everything for post-mortem debugging
6. Wait for instructions

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
- **Append `<esc-id>` to the wrapper-owned exclude-file** (see "Starting the watcher" above; path printed to stderr by `scripts/watcher-rearm.sh`) while the item is deliberately pending, so the initial scan does not instantly re-fire on it and busy-loop. Because the wrapper re-reads the exclude-file every poll, this also suppresses event-loop wakes from dedupe rewrites of that file without needing a watcher restart.
- **File a DecisionRecord via `write-decision`** (see "Filing Parked Decisions to the Cockpit Registry" below) — IN ADDITION to the reminder above, so this item surfaces in the cockpit decision queue.

It is better to stall development than to bake in a significant bad decision.

## Filing Parked Decisions to the Cockpit Registry (C8)

Fleet Cockpit C8 (`plans/fleet-cockpit-prd.md`): every time this skill parks a decision for the
human — Priority 3b, an AFK-mode deferral, a B3 gate abort/drift-pending outcome, or an
`infra_issue`/`recon_*` "tell the human" — also file it to the cockpit decision registry, **IN
ADDITION to** (not instead of) the in-session note and the `afk-digest.md` line. The registry is
what makes the cockpit decision queue (C5b) the primary return-triage surface; `afk-digest.md` is
**retained** (demoted to a generated history view, not removed in this batch), so nothing that
already reads it breaks.

```bash
python3 $DARK_FACTORY_ROOT/orchestrator/src/orchestrator/session_registry.py write-decision \
  --id <stable-id> --project <project> --text "<one-line question>" \
  [--task-id <task_id>] [--escalation-id <escalation_id>] [--session-id watcher-<project>-$$] \
  [--severity <esc.severity>] [--escalations-dir <project_root>/data/escalations]
```

- **`--id`**: a stable id you can recompute idempotently for the same pending item — the
  escalation id (`esc-42-1`) is usually the natural choice. Re-filing the same id overwrites the
  prior record rather than duplicating it (`write-decision` always writes the whole file).
  **INTERIM RULE — check before you overwrite.** Decision ids are fleet-global, so *another*
  watcher (notably the recon watcher, which runs its own queue) may already have filed a decision
  for the same underlying human gate under this id. Before filing, check whether a decision for
  that id already exists and is still `open`; if it is, do **not** overwrite it — a second watcher
  observing the same gate must enrich or no-op, never clobber richer context or downgrade an
  existing record's severity. Park your own record and add the id to your handled set instead.
  (Observed with `esc-5914-1`, where both queues surfaced the same reify gate; that duplicate
  landing on one id is the *correct* outcome — one question, one cockpit row — but only if the
  second filer doesn't degrade the first one's record.)
- **`--text`**: the one-line question a human needs to answer — the same summary you'd otherwise
  only give in-session or in the digest.
- **`--task-id` / `--escalation-id` / `--session-id`**: thread through whatever you have — the
  blocked task, the escalation this resolves, and this watcher's own session slug (see "Claiming
  the Watcher Lease" above) — so the cockpit can cross-link the decision to its source.
- **`--severity`**: pass through the parked escalation's own severity (`esc.severity` —
  `info`/`blocking`/`critical`/`urgent`). This now weights the cockpit decision-queue rank, so a
  freshly-filed `critical`/`urgent` park surfaces at the top of the queue instead of being buried
  under stale awaiting-input sessions.
- **`--escalations-dir`**: the escalation **queue** your `--escalation-id` belongs to — for this
  watcher, `<project_root>/data/escalations`. It must name the SAME queue you later pass to
  `reap-decisions` (below). Decision records are fleet-global while an escalation id
  (`esc-<taskid>-<n>`) is unique only *within* one queue, and a project can run several
  (dark_factory also runs `data/reconciliation/escalations` over the same id namespace), so this is
  what lets the reaper join a decision back to the right per-queue id namespace instead of matching
  an unrelated same-named escalation. Stored normalized, so any spelling of the same directory
  works. Omitting it files a queue-less record — see the reaper caveat below.
  There is a third value the field can hold: `<unknown>` (`session_registry.UNKNOWN_QUEUE`) —
  "this record's owning queue was investigated and could not be determined". You never write it;
  task 3640's back-fill did, for legacy records whose escalation id resolved in several queues at
  once. It is **not** a respelling of the queue-less `''` state: `''` means *nobody told us* and
  falls back to project-only scoping, while `<unknown>` means *we looked and could not tell* and
  the reaper refuses to close it at all.
- The verb always files `state=open` and prints the filed id on success for your own cross-link
  (e.g. into the digest line). It is fail-soft — a registry fault is logged and swallowed, never
  raised, so filing a decision can never crash the watch loop or block the park itself.

This is additive at every "leave pending" / "tell the human" / "park" moment below — do the
existing action exactly as documented, and also run `write-decision` once per parked item.

### Closing parked decisions on resolve

A filed DecisionRecord stays `state=open` — and therefore visible in the cockpit decision queue —
until its escalation reaches a terminal status. The watcher that files a decision is almost never
the one that resolves it: resolution typically happens later, via a spawned `/unblock` session, an
L2 cascade, or the human acting directly on the dashboard. So closing a parked decision is a
separate, recurring step, not something the `write-decision` call itself can do.

Once per Main Loop cycle (see step 6 above), run:

```bash
python3 $DARK_FACTORY_ROOT/orchestrator/src/orchestrator/session_registry.py reap-decisions \
  --project <project> --escalations-dir <project_root>/data/escalations
```

This closes (`answered`/`dropped`) any `state=open`, `escalation_id`-bearing decision whose
escalation has since resolved (`resolved` → `answered`) or been dismissed (`dismissed` →
`dropped`) — regardless of who resolved it: this session, `/unblock`, an L2 cascade, or the human.
The join is scoped on **two** axes, project *and* queue: a decision stamped (via
`write-decision --escalations-dir`) with a queue **other** than the `--escalations-dir` you pass
here is skipped outright, so your reaper can never close the recon watcher's decisions against
your own same-named escalations. A decision filed **without** `--escalations-dir` — every record
predating that flag — falls back to project-only scoping and therefore has **no** such protection:
it can still be closed by whichever queue's reaper reaches it first. That is the reason to always
pass the flag when filing. Task 3640 then **back-filled** the pre-existing open population, so
that unprotected set is now drained rather than merely shrinking as new records are filed — but
only for records that existed at back-fill time. A decision you file today without the flag lands
straight back in it.

A decision stamped `<unknown>` is **refused**, not closed: its owning queue was investigated and
could not be determined, so *no* reaper may close it and it stays a visible cockpit row until a
human closes it. The reaper never defaults to closing on doubt — that direction is deliberate,
since an over-held decision is a triageable row while a falsely-closed one is invisible.
If unstamped open records ever reappear, the re-runnable remedy is
`scripts/backfill_decision_queue_stamp.py` (dry-run by default; `--verify` exits non-zero while
any open record still lacks a stamp).
It is read-only with respect to escalations (it only ever writes the decision's own state field)
and fail-soft, exactly like `write-decision` — a registry fault is logged and swallowed, never
raised, so it can never crash the watch loop. A decision filed with **no** `escalation_id` (e.g.
the tasks.json-corruption park) is never auto-closed this way and needs explicit human closure.
Likewise, a decision whose `escalation_id` never resolves to a status — the escalation was purged
by archive retention pruning, or never existed — also stays `open` forever and needs the same
explicit human closure; until then, every cycle repeats a full scan of the escalations archive
looking for it.

## Merge Submissions — Bounded Submit, Then Poll

An unbounded foreground `mcp__escalation__merge_request` blocks until the merge worker finishes
rebasing, running the full verify suite, and CAS-advancing main. On a large/slow repo (e.g. reify)
such a call could take **30+ minutes** — made in the foreground it would freeze the entire watch
loop for that long: no draining, no watcher re-arm, a born-at-L2 `critical` sits unseen (real
incident: esc-2831-78 wedged a reify watcher >30 min on a direct foreground retry-land). The watch
loop's latency budget must stay bounded.

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
   - `resolve_issue(..., action='park')` so the blocking task lands `blocked`, held under an open L2
     (no re-dispatch while the escalation is open; the stranded-blocked sweep skips a blocked task
     that has an open escalation), and
   - File a DecisionRecord via `write-decision` (see "Filing Parked Decisions to the Cockpit
     Registry" below) — IN ADDITION to the follow-up task, so the parked decision surfaces in the
     cockpit decision queue.
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
     a terminal nobody attends just clutters. Append `<esc-id>` to the wrapper-owned exclude-file
     (see "Starting the watcher" above) for each item left pending so the initial scan does not
     busy-loop on it.
   - Either way, also **file a DecisionRecord via `write-decision`** (see "Filing Parked Decisions
     to the Cockpit Registry" below) for each item left pending — IN ADDITION to the digest entry.

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

Every "leave escalation pending + digest line" outcome above also files a DecisionRecord via
`write-decision` (see "Filing Parked Decisions to the Cockpit Registry" below) — IN ADDITION to
the digest line, not instead of it.

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
human should look) and file a DecisionRecord via `write-decision`. **At most one re-investigation
per handling cycle.**

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
  not been used this cycle, route through the drift path once. Also file a DecisionRecord via
  `write-decision` carrying the abort reason as `--text` (see "Filing Parked Decisions to the
  Cockpit Registry" below).

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

`afk-digest.md` is **retained** (Fleet Cockpit C8) — it is not removed in this batch, but it is
demoted to a generated history view: the cockpit decision queue (C5b), fed by `write-decision`
(see "Filing Parked Decisions to the Cockpit Registry" above), is now the primary return-triage
surface for any of the above that left a decision open (i.e. every outcome except "Merged").

**When a line is derived from a triage ack.** If a digest line above, or a `write-decision --text`
registry entry (see "Filing Parked Decisions to the Cockpit Registry" above), draws on an existing
`triage_note` rather than fresh investigation, carry forward the predicate and probe it names — not
just its conclusion (see "Reading a triage-ack annotation" above). A conclusion-only line forces a
returning human, or the next rotation, to re-derive the disposition from scratch instead of
re-checking a named, machine-checkable predicate.

## Merge-failure disposition vocabulary (skew ⇒ port, don't debug)

Merge-gate verify failures now carry a **disposition** — a classification of whose
fault the failure is, orthogonal to the failure's `category` (`plans/merge-skew-attribution-prd.md`,
task β, `orchestrator/src/orchestrator/merge_disposition.py`). It is a **closed enum**:

| Disposition | Meaning |
|---|---|
| `main_red` | The failure already reproduces on main tip (preexisting-main-break probe fired) — reported through the existing "fix main:" dedup path, not new. |
| `integration_skew` | The branch's own pre-merge verify was green, but a commit landed on main since the branch's merge-base that touches the files the failing test(s) depend on. The branch is *semantically stale* against that landing — nothing is wrong with the branch's own diff. |
| `branch_bug` | No landed commit is implicated — the failure is the branch's own bug. |
| `indeterminate` | The classifier couldn't reach a verdict (evidence inconclusive, or an internal error — fail-open). Treat exactly like today's undifferentiated failure. |

**Where it surfaces:** `integration_skew` **does** get its own escalation
category: the workflow layer files the block with `category='integration_skew'`
and `suggested_action='port_landed_change'` (workflow.py, INTEGRATION_SKEW
disposition branch). The task's block reason also carries an appended suffix of
the form `integration_skew: port landed commit(s) <sha[, sha...]>
touching <files> — do not hunt your own diff`, and the same disposition +
implicated commits + overlap files are available verbatim in `merge_status`'s
`failure_diagnostic` field. Look for that category/suffix/field before you (or a
spawned `/unblock` session) start reading the branch's own diff for a bug that
isn't there.

**Triage rule — the load-bearing part:**

- **`integration_skew` ⇒ PORT the named landed commit(s) into the branch.** The fix
  is an agent edit that brings the branch's content in line with what already
  landed on main — cherry-pick, reapply, or hand-port the relevant hunks of the
  named sha(s) into the files the diagnostic lists. Do **not** debug the branch's
  own diff; the branch's logic was fine when it was cut, and re-running verify or
  hunting for a regression in the branch's own change is wasted effort.
- **Skew is NOT a flake, and must not be auto-filed into flake stats.** It reads
  superficially like a transient failure (retry-after-rebase would make it pass),
  but it has a deterministic, name-able cause. Any flake-tracking/auto-file path
  (e.g. reify 5142 / DF 2358's flaky ledger) must filter on `disposition ==
  'integration_skew'` and exclude it, and a human doing manual triage should
  likewise never wave it off as "just flaky."
- `main_red` and `branch_bug` need no special handling beyond what their existing
  category sections already say; `indeterminate` degrades to today's behavior —
  treat it as an undifferentiated failure.

This vocabulary matters most for `task_failure` and `wip_conflict` (see those
sections below) — check the block reason / `failure_diagnostic` for a disposition
before spawning `/unblock` or trying the low-risk auto-unblock gate.

## Handling Escalations by Category

For every escalation, read the `suggested_action` field. It's a free-text hint — sometimes a conventional verb, sometimes natural language. First determine the escalation's **L2 origin**, then interpret the hint accordingly:

**Born-at-L2** (severity `critical` or `urgent` at creation — bypassed L0 and L1 entirely):
Neither the per-task steward nor the auto-watcher has seen this record. Read `suggested_action` as the originating agent's own annotation — a starting point, not evidence of prior triage. `investigate_and_retry` here means what it says: a retry may well succeed since no automated attempt has been made yet.

**Promoted-from-L1** (the auto-watcher attempted resolution and escalated to human):
- **`manual_intervention`** — The auto-watcher explicitly gave up. This is authoritative: the issue genuinely needs human judgment. Always respect it.
- **`investigate_and_retry`** — Misleading for promoted items. The item has already passed through *both* the per-task steward (L0) *and* the auto-watcher (L1) and persisted through their combined triage and retry budgets. Treat as a deeply persistent problem, not transient. Don't just retry.
- **`triage_suggestions` / `fix_review_issues`** — Routing hints confirming what the category tells you. No new information.
- **Free-form text** (e.g., "Restore Value::Frame from previous commits") — Valuable diagnostic context about what the escalating agent *thought* would help. Read it as a starting point for investigation, not as instructions — the agent was stuck, so its diagnosis may be incomplete.

**Additive-context convention for spawned `/unblock` prompts.** Several categories below spawn an
interactive `/unblock` session with a prompt of the form `/unblock <task_id> (esc <escalation_id>,
<category>, <severity>: <summary>)`. Only the leading `/unblock <task_id>` token is load-bearing:
`/unblock`'s own Step 0 (Locate the task) extracts `<task_id>` from it, and Step 1 (Gather context)
re-derives all context — escalation, task status, git state — fresh from `TASK_ID`
(`skills/unblock/SKILL.md`). The trailing `(esc ...)` context is purely additive, there only to
orient the human reading the terminal before `/unblock` runs, so it stays non-load-bearing even if
a section's summary below drifts out of date.

Two mechanical caveats when building this prompt string:
- **Which number is the task id.** The parenthetical introduces a second number (`<escalation_id>`);
  Step 0 must take only the number immediately following `/unblock` as `<task_id>` — `<escalation_id>`
  always sits behind the `esc ` token inside the parenthetical, never in the leading position, so the
  leading number is unambiguous.
- **Escape or truncate `<summary>` before interpolating.** Escalation summaries are free text and may
  contain single quotes, parens, or newlines. Collapse `<summary>` to a single line, and either
  shell-escape inner single quotes as `'\''` per `/spawn`'s Arguments section (`skills/spawn/SKILL.md`)
  or truncate it to a short slug — otherwise an unescaped quote in the summary breaks the
  single-quoted `<prompt>` argument passed to `spawn-claude.sh`, and the spawn fails or truncates.

Each call site below notes this briefly rather than repeating the full explanation.

### Reading a triage-ack annotation

A pending record — most often an L2 promoted from L1, but any pending item in principle — may
carry a triage-ack annotation: `triaged_at`, `triaged_by`, `triage_note` (plus `updated_at`, the
record's own last-substantive-change marker). This means an earlier rotation — almost always
`escalation-watcher-auto`, since stamping is how that skill records "I assessed this without
resolving or promoting it" — already looked at the item and left a handoff note so the next reader
doesn't have to re-derive its disposition from scratch. Read it exactly like `suggested_action`
above — **a starting point, not a verdict**.

**Re-verify (re-run the probe yourself) instead of trusting the note** when either:
- `triaged_at` is older than roughly 6 hours, or
- the record changed since triage — `updated_at` is **not** `None` **and** is newer than
  `triaged_at` (e.g. the L2 cluster gained a new member via `promote_to_l2` after the stamp was
  written). `updated_at` defaults to `None` (never bumped) until the record's first real content
  change, so a triaged record with no changes since still reads `updated_at = None` — treat that as
  "not newer than `triaged_at`", never as an ordering comparison between `None` and a timestamp
  string.

A well-formed `triage_note` names a machine-checkable **predicate** (e.g. `` `task-604
status==done` ``) and the **probe** used to check it (e.g. `` `probe: get_task 604 ->
status=done` ``) — re-run that same probe before trusting the predicate still holds. A
`triage_note` that states only a **conclusion**, with no predicate or probe behind it, is untrusted
prose, not a substitute for investigation. This is exactly the esc-2584 failure mode: a
conclusion-only recommendation ("resume will close it") was taken at face value, got refuted, and
cost two churn cycles and five separate `resolve_issue` calls before the item was actually closed.

`triaged_by` is server-attributed from the stamping connection's `X-Escalation-Identity` header and
cannot be spoofed by the caller — the identical non-spoofable attribution contract this skill
already documents for `resolved_by` (see "Recognizing the supervised auto-watcher's resolutions"
below).

### `review_suggestions` (info)

> **This handler is unreachable at L2.** Review suggestions reach live workflows as curator tickets
> via `_route_review_suggestions_to_curator` in workflow.py (call site ~line 3064), with no
> escalation file written; they fall back to level-0 steward escalations filed around
> workflow.py:6272 and consumed by `_next_escalation` in steward.py. They do not reach this
> queue. This stub is kept only to document why `review_suggestions` must not be re-added here.

### `review_issues` (blocking)

Blocking issues found during code review — the review cycle exhausted without the agent fixing them. The task agent is stopped.

This is distinct from `review_suggestions` (info-level, non-blocking). Review issues are real problems that prevented the task from merging.

**Spawn an interactive `/unblock` session** via the `/spawn` skill: invoke `/spawn` with `prompt="/unblock <task_id> (esc <escalation_id>, review_issues, <severity>: <summary>)"`, `terminal_title="unblock:<project>#<task_id> <short-slug>"` (e.g. `unblock:df#2085 routing-mechanism`; abbreviate the project token per the emergent convention), `cwd=<project_root>`, `skip_permissions=true`. Leave the escalation pending — `/unblock` resolves it when the human finishes. The human needs to see the specific blocking issues and decide how to fix them. The trailing `(esc ...)` context is additive only (see the additive-context convention note above).

If the low-risk auto-unblock gate applies — see [Low-risk auto-unblock gate (B3)](#low-risk-auto-unblock-gate-b3) — try it first.

### `task_failure` (blocking)

Merge conflicts, verification failures, build breaks. The task agent is stopped and waiting.

Before investigating, check the block reason / `merge_status.failure_diagnostic` for a
`disposition` — see [Merge-failure disposition vocabulary](#merge-failure-disposition-vocabulary-skew--port-dont-debug)
above. If it reads `integration_skew`, the fix is porting the named landed commit(s), not
debugging the branch's own diff, and it must not be counted as a flake.

**Spawn an interactive `/unblock` session** so the human can investigate and resolve it: invoke `/spawn` with `prompt="/unblock <task_id> (esc <escalation_id>, task_failure, <severity>: <summary>)"`, `terminal_title="unblock:<project>#<task_id> <short-slug>"` (e.g. `unblock:df#2085 routing-mechanism`; abbreviate the project token per the emergent convention), `cwd=<project_root>`, `skip_permissions=true`. Leave the escalation pending — the `/unblock` skill resolves it when the human finishes. Track the spawned session so you can report its status if asked. The trailing `(esc ...)` context is additive only (see the additive-context convention note above).

If the low-risk auto-unblock gate applies — see [Low-risk auto-unblock gate (B3)](#low-risk-auto-unblock-gate-b3) — try it first.

### `wip_conflict` / `unmerged_state` (blocking, halt-owner)

These escalations mean the **merge queue is globally halted** — no other task can merge until exactly one of them (the "halt owner") is resolved. The orchestrator records which escalation owns the halt on the merge worker (`_halt_owner_esc_id`); resolving that specific escalation via MCP un-halts the queue. Resolving any other escalation — even another `wip_conflict` — will NOT release the halt (fixed 2026-04-19; prior code relied on a category heuristic that caused phantom-L1 bugs like esc-1888-57).

Two flavours:
- **`wip_conflict`** — the merge queue tripped on uncommitted work in `project_root`. Three sub-variants distinguishable from the `detail`:
  - WIP overlaps the merge diff (merge did not land; workflow will retry after resolution).
  - Stash pop conflicted after the merge landed (merge IS on main; WIP preserved on `wip/recovery-<task>-<ts>`).
  - Stash pop conflicted on CAS-failure path (merge did NOT land; WIP on recovery branch; task blocks).
- **`unmerged_state`** — `project_root` already had UU/AA/DD markers before the merge attempted to advance (pre-existing corruption, not caused by this merge).

As with `task_failure`, check for a `disposition` in the block reason / `failure_diagnostic`
before assuming this is a raw conflict to resolve mechanically — see
[Merge-failure disposition vocabulary](#merge-failure-disposition-vocabulary-skew--port-dont-debug)
above. `integration_skew` still needs a port of the named landed commit(s), not just conflict
resolution, and is never a flake.

**Never auto-resolve** — `manual_intervention` is authoritative. The human has to inspect `project_root`:
- For `wip_conflict`: recovery branch named in the detail preserves the user's WIP; they may need to cherry-pick or reapply before resolving.
- For `unmerged_state`: run `git status` in `project_root`; UU/AA/DD files need `git mergetool`, manual edit, or `git reset` depending on intent.

**Spawn an interactive `/unblock` session** via `/spawn` (`prompt="/unblock <task_id> (esc <escalation_id>, <wip_conflict|unmerged_state>, <severity>: <summary>)"`, `terminal_title="unblock:<project>#<task_id> <short-slug>"` — e.g. `unblock:df#2085 routing-mechanism`; abbreviate the project token per the emergent convention — `cwd=<project_root>`, `skip_permissions=true`) so the human can see the recovery branch, inspect `project_root`, and resolve the escalation when finished. The trailing `(esc ...)` context is additive only (see the additive-context convention note above).

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
3. **If no matching task exists**: spawn an interactive `/unblock` session via `/spawn` (`prompt="/unblock <task_id> (esc <escalation_id>, dependency_discovered, <severity>: <summary>)"`, `terminal_title="unblock:<project>#<task_id> <short-slug>"` — e.g. `unblock:df#2085 routing-mechanism`; abbreviate the project token per the emergent convention — `cwd=<project_root>`, `skip_permissions=true`). The trailing `(esc ...)` context is additive only (see the additive-context convention note above).

### `design_concern` (info or blocking)

Architectural or design questions. These already failed steward auto-resolution — they're genuinely ambiguous.

**Always escalate to the human:**
1. Present the concern with full context
2. Leave the escalation pending — the open escalation record IS the durable record that something
   needs doing
3. Create a local todo **for this session only** — it does not survive session end and is not the
   record
3a. File (or confirm one already exists for this `esc-id`) a cockpit DecisionRecord via
   `write-decision` — **this, not the todo, is what makes the item recoverable across sessions and
   after this session ends** (same registry as the Priority-3b instructions above). Skipping this
   step is how esc-3223-4/-5 kept task 3223 blocked for 11 days: the question never reached the
   cockpit queue the human actually reads.
4. Continue handling other escalations while waiting
5. Append `<esc-id>` to the wrapper-owned exclude-file (see "Starting the watcher" above) while this item is pending

### `risk_identified` (info)

An agent flagged a risk during development. Risk assessment requires human judgment.

**Escalate to the human.** Tell them; create a session-only todo (an attention aid — the pending
escalation, not the todo, is the durable record); file (or confirm) a cockpit DecisionRecord via
`write-decision` for this `esc-id`, exactly as in `design_concern` step 3a; continue with other
work. Append `<esc-id>` to the wrapper-owned exclude-file (see "Starting the watcher" above) while
this item is pending.

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
  elif resolve["status"] == "refused":
      # Deliberately NOT in the tuple above: a refusal has no task_id.
      # A deterministic guard rejected the candidate; no task was created.
      # Record resolve["reason"] in the escalation resolution note.
      # Do NOT retry and do NOT record a task id.
      note_refused(resolve["reason"])
  elif resolve["status"] == "failed":
      # Record reason in escalation resolution note; skip this item.
      # See skills/_shared/ticket-failure-handling.md for the retryable/terminal reason matrix.
      handle_failure(resolve["reason"])
  ```

  Resolve via `mcp__escalation__resolve_issue` once the ticket resolves.
- **Blocking** (rare): spawn an interactive `/unblock` session via `/spawn` (`prompt="/unblock <task_id> (esc <escalation_id>, cleanup_needed, <severity>: <summary>)"`, `terminal_title="unblock:<project>#<task_id> <short-slug>"` — e.g. `unblock:df#2085 routing-mechanism`; abbreviate the project token per the emergent convention — `cwd=<project_root>`, `skip_permissions=true`). The trailing `(esc ...)` context is additive only (see the additive-context convention note above).

### `infra_issue` (blocking)

Infrastructure problems — database connectivity, MCP failures, service outages.

**Priority 1 — system stability:**
1. Tell the human immediately with full details
2. Leave the escalation pending
3. Do NOT attempt automated infrastructure fixes
4. File a DecisionRecord via `write-decision` (see "Filing Parked Decisions to the Cockpit
   Registry" above) — IN ADDITION to telling the human directly
5. Wait for human instructions
6. Append `<esc-id>` to the wrapper-owned exclude-file (see "Starting the watcher" above) while this item is pending

### `recon_*` categories

`recon_failure`, `recon_backlog_overflow`, `recon_stale_run`, `recon_integrity_issue` — these are all fused-memory reconciliation problems.

Reconciliation is infrastructure that affects memory quality across the entire system. **Tell the human** with full details. Track as a todo. These may indicate systematic issues that need root-cause investigation rather than point fixes. Also file a DecisionRecord via `write-decision` (see "Filing Parked Decisions to the Cockpit Registry" above).

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

### Joining a completed spawn: read `result.md`, don't explore

When a `/spawn`-launched `/unblock` background task completes, that's the **join** step of the
fan-out/join pattern (fan-out was the spawn; the background task's completion is your join signal).
Exit codes are **liveness-only** — present-vs-died-silent, not a semantic outcome signal, including
the observed 129-on-clean-exit race (see `skills/spawn/SKILL.md`'s Verification section) — so never
infer success, failure, or completeness from the background task's exit code.

Instead, **read the session's result file** for the structured outcome:
1. Locate the session-registry record for the spawn (the slug you launched it under, or the
   matching record under `~/.claude/fleet/sessions/` for the task/title you spawned).
2. Read `record.result_file` — equivalently `~/.claude/fleet/sessions/<slug>/result.md` — for the
   `outcome` (`done|blocked|abandoned|handed-off`), `changed` (commits/branches/task ids touched),
   and `action_needed` fields the spawned session wrote before ending.
3. Use that structured outcome to decide your next step (resolve the escalation, follow up, escalate
   further) — `result.md` is the authoritative outcome source, so this replaces exploring the
   task/worktree yourself to reconstruct what happened.

**Fail-soft:** the write is best-effort — if `result.md` is absent, empty, or unparsable (the
spawned session ended without writing it, or the registry write itself faulted so the record never
got a `result_file`), fall back to the existing exploration (task/escalation state via MCP, the
worktree itself) rather than blocking on the file.

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
| `park` | kept open at L2 | killed | → `blocked` (from any non-terminal status) | "Stop; human decides later; held blocked under an open L2." |
| `abandon` | `dismissed` | killed | → `cancelled` | "Never run again." |
| `close_only` | `dismissed` | untouched | none | "Record is noise/duplicate — change nothing." |

**C1 notes:**
- Terminal task statuses (`done`, `cancelled`) are never overwritten by any action.
- The removed `terminate` parameter now raises a hard error naming the five actions above.
- **L2 cluster cascade**: the action applies uniformly to the L2 and every member task. `queue.resolve()` cascades members via `resolved_by='l2-cascade:<L2-id>'`; the harness member callback reads the parent action from the queue read API. For `action='park'`: `queue.park()` keeps the L2 and all member L1s open (status=`pending`); each member task ends `blocked`, covered by its still-open member L1 escalation — the stranded-blocked sweep skips each because Fix #1b finds the open L1.
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
  later use `action='park'` (→ `blocked`, held under an open L2); to abandon permanently use `action='abandon'`
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

### Recognizing the supervised auto-watcher's resolutions (not a rogue actor)

You may see `resolved_by="orchestrator-escalation-watcher-auto"` stamped on archived L0/L1
records, or `agent_role="escalation-watcher-auto"` on a `promote_to_l2` call that filed an L2.
Both are the **trusted, supervised** identity of the dark-factory orchestrator's own autonomous
auto-watcher (spawned per rotation by the watcher-supervisor, task 1326; runs the
`escalation-watcher-auto` skill). Per the connection-capability guard
(`plans/escalation-connection-capability-guard-prd.md`), the orchestrator wires that watcher's MCP
connection with `X-Escalation-Identity: orchestrator-escalation-watcher-auto`, and the escalation
server stamps `resolved_by` from that header on `resolve_issue`'s resolve/park path —
server-attributed, not something the watcher agent can spoof or drift. `promote_to_l2` has no such
override: its `agent_role` is a plain tool argument, and the auto-watcher skill always passes
`agent_role="escalation-watcher-auto"` (the literal tool-arg value) on every promote call, so a
promoted L2 carries that value, never the `orchestrator-`-prefixed form. Seeing
`resolved_by="orchestrator-escalation-watcher-auto"` on L0/L1 resolutions, or
`agent_role="escalation-watcher-auto"` on `promote_to_l2` calls, is expected, routine behavior.
**Do not stand down, halt the watch loop, or treat it as an anomaly** — it is the same auto-watcher
this skill hands L1 items off to, working as designed.

Distinguish it from a **genuinely unknown resolver**: the same connection is capped at
`X-Escalation-Levels: 0,1`, so the server rejects (`level_forbidden`, no state change) any attempt
by that identity to `resolve_issue`/`park` a level-2 escalation. If you ever see
`resolved_by="orchestrator-escalation-watcher-auto"` (or `agent_role="escalation-watcher-auto"`) on
an L2 record's *own* resolution/park — as opposed to an L1 member cascade-resolved via
`resolved_by='l2-cascade:<id>'`, or an L1/L0 admin item, or a `promote_to_l2` call — that should not
be possible under the enforced guard, and is the actual anomaly worth reporting to the human
(possible guard regression, bypassed connection headers, or a stale pre-guard orchestrator/server
pair that hasn't been restarted onto the fix yet).

**Capped identity leaking into an interactive session (task 2796):** the supervised auto rotation
now attaches its capped `escalation` connection with `--strict-mcp-config`, scoping the rotation to
only its own MCP config and no longer merging the ambient project `.mcp.json`. Because the capped
block shares the identical server name (`escalation`) and URL (port `8102`) as an interactive
session's header-less block, that non-strict ambient merge was the path by which a capped
`X-Escalation-Identity: orchestrator-escalation-watcher-auto` / `X-Escalation-Levels: 0,1` identity
could bleed into a concurrent header-less **interactive** watcher session — making that session's own
L2 `resolve_issue`/`park` fail `level_forbidden` as if it were the capped auto-watcher. With the
strict isolation now in place, a capped identity should no longer leak into an interactive session.
**If it recurs anyway**, the supervised rotation is no longer the likely culprit — suspect a
hand-injected `--mcp-config` (e.g. a `CLAUDE_SPAWN_CLAUDE_ARGS` passthrough) or a config generator
writing capped headers directly into `.mcp.json`.

**If MCP is unreachable:** ask the human for help. Don't try to resolve escalations by writing directly to the queue files — this bypasses callbacks and can leave the orchestrator in an inconsistent state.

## Red-on-main recovery (enforce-safe, break-glass)

When a bad merge has turned `main` RED and the ref must be moved to a known-good SHA, use the `recover_main` CLI for a **single atomic move**. This avoids the rewind-then-readvance back-and-forth that produces a net no-op (the jun9 fumble: f4101683→4001d48d→f4101683).

**This move is BACKWARD (non-fast-forward):** `<current-main>` is not an ancestor of `<good-sha>`, so it rewrites history. A project whose main-gate hook has an **always-on non-fast-forward guard** (reify) REJECTS a backward ref move *unconditionally* — before the sanction/sentinel check is even reached — so `git.main_gate_mark_command` (the forward-move sanction that `advance_main` uses) is **not** sufficient on its own for recovery. For those projects the CLI instead engages a **durable break-glass bypass** of the non-ff guard for exactly the CAS window (see the prerequisite check). Projects with only a sanction gate (no non-ff guard) still get the mark and land the move as `SANCTIONED`.

### Prerequisite check

- The watched project's `project_root` working tree must be **clean** (no uncommitted WIP). The CLI does not stash/pop — that is out of scope for a break-glass operation.
- **If the watched project has an always-on non-ff main-gate guard (reify):** confirm its `orchestrator.yaml` `git:` block sets **BOTH** `git.main_gate_bypass_command` (engages the durable bypass) **AND** `git.main_gate_bypass_clear_command` (clears it). When both are set, `recover_main` **AUTO-engages** the durable bypass immediately before its CAS `update-ref` and **AUTO-clears** it immediately after on every path (success, CAS failure, exception), so the recovery move is allowed through without leaking the bypass into later ref moves. These are generic per-project shell commands (run via `sh -c` in `project_root`); they should drive the same gate-side knob the project's hook reads — see the fallback subsection below for the knob forms. When the bypass command is set it **supersedes** `git.main_gate_mark_command`: the CLI skips the mark entirely (running both would leave the one-shot sanction sentinel unconsumed and falsely sanction the next ref move).
- **If the watched project has only a sanction gate (no non-ff guard):** confirm `git.main_gate_mark_command` is set. Leave the bypass commands unset. If the mark is also unset, the move is still atomic but will not be sanctioned by the reference-transaction hook (use the manual fallback below).

### Step 1 — Identify the bad merge and good target

```bash
# In the WATCHED project's repo:
git log --oneline -10 main
# Find the last known-good SHA (pre-bad-merge) and the current bad SHA.
```

- `<good-sha>` = the commit you want to restore `main` to (pre-bad-merge)
- `<current-main>` = current value of `refs/heads/main` (the bad merge; CAS old-value)

### Step 2 — Perform the single atomic recovery move

Run from `$DARK_FACTORY_ROOT`:

```bash
.venv/bin/python -m orchestrator.recover_main \
  --project-root <watched-project-root> \
  --config <watched-orchestrator.yaml, e.g. orchestrator/config.yaml> \
  --target-sha <good-sha> \
  --expected-main <current-main>
```

Parse JSON output: `{"result": "rewound"|"cas_failed"|"error", "target_sha": "..."}` (on an unexpected exception the object also carries a `"detail"` key).

- `rewound` (exit 0): `main` is now at `<good-sha>`; the durable bypass (if configured) has already been cleared. Proceed to step 3.
- `cas_failed` (exit 1): another writer moved the ref first. Re-read current `main` and retry — or escalate to the human if the situation is unclear.
- `error` (exit 1): a SHA failed pre-validation (typo, or not a commit in the watched repo). Fix the SHA and retry — do **not** treat this as a CAS race.

### Step 3 — Re-advance the fix through the normal merge queue

Once `main` is at the good SHA, the fix commit must land through the normal merge queue (already sanctioned by `advance_main`). Do **not** repeat a raw `git update-ref` — use the queue.

### Break-glass bypass fallback (bypass commands NOT configured)

Prefer the automatic path above: set `git.main_gate_bypass_command` / `git.main_gate_bypass_clear_command` in the watched project's `orchestrator.yaml` so `recover_main` engages and clears the bypass for exactly the CAS window. Use this **manual** fallback only when a non-ff-guarded project has **not** yet configured those two commands and you cannot add them right now.

The gate itself honors a project-specific break-glass bypass; the two config commands above are just generic `sh -c` wrappers that toggle one of these gate-side knobs. Check the watched project's config for the knob it reads:

- **Environment variable**: `<PROJ>_MAIN_GATE_BYPASS=1` (e.g. `REIFY_MAIN_GATE_BYPASS=1`)
- **Git config key**: `git config <proj>.mainGate.bypass true` in the watched repo
- **Flag file**: a sentinel file under the repo's git common-dir (path from the watched project's gate config)

These are gate-side controls read by the watched project's reference-transaction hook — they are not dark-factory settings. Consult the watched project's gate documentation for the exact knob. To recover manually: **(1)** engage the bypass (set the knob), **(2)** run a raw CAS `git update-ref refs/heads/main <good-sha> <current-main>` in the watched repo (which the guard now allows through), **(3)** **clear the bypass immediately after** — on both success and failure, since the bypass is durable and would otherwise leak into every later ref move. Once the bypass commands are configured, `recover_main` does steps 1–3 for you; wire them up so the next recovery is one command.

## Failure Modes

**"Too many open files" (historical — no longer expected)**: Early sessions could exhaust the background-task fd pool after ~35 watcher restart cycles. This is no longer observed in practice — 100+ cycle sessions are routine. The watcher exits promptly via `sys.exit(0)`, so its inotify fd is reclaimed by the kernel and the background task is reaped shortly after. If you ever do hit it, start a fresh Claude Code session.

**Orchestrator not running**: If no new escalations arrive for an extended period, the orchestrator may have crashed or finished. Check with the human.

**Stale escalations**: On orchestrator startup, `dismiss_all_pending()` auto-dismisses **L0** escalations from prior runs (filter: `level == 0`) — **L1 and L2 escalations are preserved across restarts**. So an L2 with a timestamp from a previous session that is still `status: pending` is legitimate carry-over, not stale; handle it normally. If an escalation genuinely looks wrong (e.g. references a task that is already Done), tell the human rather than dismissing it yourself — it may contain useful diagnostic information.
