---
name: recon-escalation-watcher
description: "Watch and close fused-memory's RECONCILIATION escalation queue (port 8103) in a long-running loop. This is the consumer for the recon queue — the integrity/operational findings the reconciliation harness files, NOT the orchestrator's task-pipeline escalations (those are escalation-watcher's job, ports 8100/8102). Use when the user says 'watch recon escalations', 'monitor the fused-memory escalation queue', 'babysit reconciliation', mentions the 8103 queue, recon_integrity_issue / recon_failure / recon_stale_run findings piling up, or wants the reconciliation queue triaged and closed. This is a continuous loop skill; the watcher is the SOLE closer of the recon queue (recon never resolves its own findings). This is NOT for orchestrator task escalations, blocked tasks, merge failures, or worktrees."
---

# Recon Escalation Watcher

You are running a long-running watch loop over **fused-memory's reconciliation
escalation queue** (port **8103**, dir `<project_root>/data/reconciliation/escalations/`).
Your job is to triage and **close** the integrity and operational findings the
reconciliation harness files, keeping the queue a small, meaningful signal.

This is a **sibling** of `escalation-watcher`, not the same skill. It shares the
MCP poll-loop scaffolding but the subject and semantics are different:

| | escalation-watcher | recon-escalation-watcher (this) |
|---|---|---|
| Queue | orchestrator (8100/8102) | reconciliation (**8103**) |
| Subject | task-pipeline blockers | memory/task integrity & recon ops |
| Tiering | L0→L1→L2 ladder | **none** — recon files flat, no levels |
| On resolve | resumes/abandons an agent in a worktree | just **marks the finding handled** — no agent, no worktree, no resume |
| Closer | steward / auto-watcher / human | **this watcher is the SOLE closer** (recon never resolves its own) |

There is no merge queue, no worktree, no steward, no L0/L1/L2 here. You never
spawn `/unblock`. Most findings carry a synthetic `recon-<runid>` task_id that
identifies the reconciliation run, not a real Taskmaster task.

## Prerequisites

Verify these before starting. If anything is missing, ask the user — don't guess.

1. **`DARK_FACTORY_ROOT`** — path to the dark-factory repo (default
   `/home/leo/src/dark-factory`). The `escalation` package (watcher) lives here.
2. **This session's `escalation` MCP must point at 8103**, and `fused-memory` at
   8002. The stock repo `.mcp.json` points `escalation` at 8102 — wrong queue.
   Launch via the recon-watch config (see "Launching" below); confirm with
   `mcp__escalation__get_pending_escalations()` returning recon findings
   (ids like `esc-recon-<hex>-N`, categories `recon_*`).
3. **The 8103 server is up** — it runs inside `fused-memory.service`. If
   `get_pending_escalations` errors, the service is down → tell the user
   (priority 1, system stability); do not try to start it yourself.

## Launching (run a SEPARATE Claude session pointed at 8103)

MCP connections are per-process, so a dedicated session leaves any 8102
escalation-watcher session untouched. Use the launcher config that names BOTH
servers — the watcher needs `fused-memory` (8002) for its fix/file actions and
`escalation` (8103) for the queue:

```bash
claude --strict-mcp-config --mcp-config "$DARK_FACTORY_ROOT/recon-watch/mcp.json" \
  "/recon-escalation-watcher"
```

**Default lane: tmux.** This is a long-running loop session (PRD
`plans/fleet-cockpit-prd.md` §3 fork 1) — run it in the crash-survivable, reattachable tmux lane
by default: wrap the invocation above via `skills/spawn/spawn-claude.sh` with
`CLAUDE_SPAWN_BACKEND=tmux` (see that script's header) so the session gets a `display.kind=tmux`
session-registry record and a `tmux attach`-reattachable window. A killed watcher is reattachable
with `tmux attach`, and its record persists across the crash. Interactive one-offs are unaffected
— this watcher never spawns interactive sessions of its own (see the table above: no `/unblock`,
no worktree).

`recon-watch/mcp.json` (created by the setup; both servers required):

```json
{
  "mcpServers": {
    "escalation":   { "type": "http", "url": "http://127.0.0.1:8103/mcp" },
    "fused-memory": { "type": "http", "url": "http://127.0.0.1:8002/mcp" }
  }
}
```

## The Main Loop

```
1. Drain all pending recon escalations
2. Start the watcher (background task, recon queue dir, NO --level)
3. Wait for the watcher to fire (it exits on the first new escalation)
4. Read the escalation from watcher output; fetch full detail via MCP
5. Drain any other pending escalations
6. Handle each
7. Run `reap-decisions` to close any parked DecisionRecord whose escalation has since resolved
   (see "Filing Parked Decisions to the Cockpit Registry" below) — once per cycle
8. Go to 2
```

### Draining

```
mcp__escalation__get_pending_escalations()      # NO level arg — recon is flat
```

**Priority order** when several are pending: blocking severity first; then within
a severity, highest **`dedupe_count`** first (recurrence = persistence = signal).

### Starting the watcher

```bash
cd $DARK_FACTORY_ROOT && uv run --project escalation python -m escalation.watcher \
  --queue-dir $DARK_FACTORY_ROOT/data/reconciliation/escalations \
  [--exclude-id <esc-id>] [--exclude-file <path>] [...] 2>&1
```

Run as a **background task** (`run_in_background`). **No `--level`** — recon
escalations have no level field worth filtering. The watcher uses inotify and
exits after the first new escalation file, printing its JSON to stdout.

**Re-arming over deliberately-pending items:** the watcher's initial scan emits
the oldest matching pending escalation and exits immediately. Any item you
deliberately left pending (Priority 3 "leave pending, tell the human") sits in
the queue and causes every subsequent watcher start to instantly re-fire on it,
degenerating into a busy-loop. Pass `--exclude-id <esc-id>` (repeatable) for
each item deliberately left pending so the initial scan and event loop skip it.
`--exclude-id` also suppresses event-loop wakes from dedupe rewrites of those
files (MOVED_TO events on the excluded file are silently ignored). Pass both the
bare id (`esc-recon-abc-1`) and `.json`-suffixed forms — both are accepted.
For a parked set this large, prefer the bulk form of the same mechanism:
`--exclude-file <path>` takes a newline-delimited esc-id list, and
`current_excludes()` (`escalation/src/escalation/watcher.py:224-225`) calls
`_read_exclude_file` on every invocation — once for the initial scan
(`watcher.py:246`) and again on every poll of the event loop
(`watcher.py:269`) — so it excludes only the ids you actually listed, with
no masking window, and the parked set can grow mid-run just by appending a
line, no watcher restart needed.

Since this skill invokes `python -m escalation.watcher` directly — no
wrapper script owns an exclude-file for you the way the sibling skill's
does (see below) — name a conventional path yourself:
`<queue-dir>/.watcher-exclude-recon`, a dotfile that the watcher's own
`esc-*.json` glob safely ignores (`_initial_scan`, `watcher.py:85`; the
same pattern also gates `EscalationQueue`'s own reads,
`escalation/src/escalation/queue.py:431` and `:509`). The name must also
not end in `.json`: the dashboard's read-only recon subsection globs the
broader `*.json` (not `esc-*.json`) over this same directory
(`dashboard/src/dashboard/data/escalations.py:89`), so a `.json`-suffixed
name would surface there as a phantom escalation record. The chosen name
mirrors the sibling wrapper's own in-queue-dir dotfile convention
(`.watcher-rearm-exclude-l2`). Append with `echo <esc-id> >>
<path>`: `_read_exclude_file`'s docstring (`watcher.py:128-132`) blesses a
single short-line append as atomic on POSIX, and a torn multi-write append
is self-healing on the next poll. The reader is fail-open — a missing or
unreadable file yields an empty set, retried next poll
(`watcher.py:134-140`) — so a lost or not-yet-created exclude file just
degrades to re-firing on parked items: noisy, never silent. Blank lines and
`#` comments are skipped (`watcher.py:143-146`), so the file can be
annotated.

**`--baseline` is NOT safe for this loop — do not reach for it here**,
even though `escalation.watcher` accepts it as a flag. `_snapshot_pending_ids()`
(`watcher.py:151-158`) freezes the pending-id set exactly once, at launch,
strictly before `add_watch` arms the inotify watch (`watcher.py:238` runs
before `watcher.py:243`) — and unlike `--exclude-file`, that snapshot is
never re-read afterward (`current_excludes()`, `watcher.py:225`; the code's own comment at
`watcher.py:227-237` reasons only about the narrower snapshot→`add_watch`
race, not this one). This loop's handling phase is long — draining,
handling dozens of parked records, and filing a cockpit DecisionRecord for
each takes real wall-clock time between one watcher restart and the next — so
any escalation recon files *during that window* is already on disk by the
time the following restart takes its `--baseline` snapshot. That new
escalation gets folded straight into the snapshot and is excluded from both
the initial scan (`watcher.py:246`) and the event loop (`watcher.py:269-286`)
for that run's entire lifetime, despite never having been drained or
triaged by anyone. It is worse than a one-cycle miss: the snapshot is
retaken at *every* restart while the item is still pending, so it stays
masked cycle after cycle until some unrelated, newer escalation happens to
fire the watcher and the next drain re-finds it. And the wait here is
**unbounded** — this skill's watcher invocation carries no `--timeout`, so
`deadline` stays `None` and the event loop blocks indefinitely
(`watcher.py:252-264`), with no expiry to force a restart-and-redrain. Net
effect: with `--exclude-id` / `--exclude-file`, an escalation filed *during
the handling window* is not on the exclude list, so it fires at the next
watcher start like any other pending item — normal, expected, and loud.
With `--baseline` that same new escalation is instead swallowed by the
launch snapshot into a silent, unbounded delay in this queue's **sole
closer**.

With PARK now the default disposition for `reconciliation_stale_gate_backlog` /
`reconciliation_stale_human_operator`, the parked set is structurally large
— these two categories make up the *entire* pending queue today (dozens of
records; see the dated census in the playbook row below for the current
count) — so hand-listing one `--exclude-id` per record does not scale; use
`--exclude-file` for it, not `--baseline`. The sibling
`skills/escalation-watcher/SKILL.md` documents `--baseline` as an available
flag too, but that is not a counterexample: its Main Loop restarts the
watcher *first* and drains only after confirming it's up, and its step 7 is
`Go to 1`, so **every** restart there is immediately followed by a fresh,
authoritative drain — that skill says as much at its lines 93-94, that the
fired escalation "is just the wake ... the drain re-finds it (still
pending) plus anything new." Its documented invocation also passes
`--timeout 3600` (`skills/escalation-watcher/SKILL.md:133-136`), so even a
completely quiet queue force-restarts and re-drains at least once an hour.
Tellingly, that skill's own re-arming guidance for its parked set does not
reach for `--baseline` either — it routes through a wrapper-owned
`--exclude-file` instead (`skills/escalation-watcher/SKILL.md:165-175`,
`scripts/watcher-rearm.sh`, default path
`<queue-dir>/.watcher-rearm-exclude-l2`). `--exclude-file` — not
`--baseline` — is what is actually consistent across both watcher skills.

**Process safety:** only stop watcher processes you started via background task
controls. Never `pkill` by pattern.

## The Action Set

For each finding, decide among four closures. Read `summary`, `detail` (usually
JSON with `description`/`affected_ids`/`actionable`), and `dedupe_count`.

1. **verify-fixed** — Check current state via `mcp__fused-memory__search`,
   `get_entity`, `get_task`. If the finding is already true-resolved (the edge
   exists, the task is in the expected state, the contamination is gone), close
   it: `resolve_issue(..., action='resume', resolution="Verified fixed: <what you checked>")`.

2. **accept-as-known** — The finding is non-actionable or an accepted state (e.g.
   a deliberately-deferred task, a known intractable item, an auto-recovered
   stale run). Dismiss it: `resolve_issue(..., action='close_only', resolution="Accepted as known: <why>")`.

3. **file-a-real-task** — The finding is genuinely actionable dev work. File it,
   then resolve the escalation. Two-phase pattern:
   ```
   sub = mcp__fused-memory__submit_task(
       project_root="<project_root>", title="<title>", description="<what + specifics>",
       priority="medium",
       metadata={"source": "recon-watcher", "escalation_id": escalation_id,
                 "spawn_context": "steward-triage"},
   )
   res = mcp__fused-memory__resolve_ticket(ticket=sub["ticket"], project_root="<project_root>",
                                           timeout_seconds=<see _shared/ticket-failure-handling.md>)
   # status created|combined -> task_id ; failed -> record reason, leave escalation pending
   ```
   Then `resolve_issue(..., action='resume', resolution="Filed task <id>: <title>")`.

4. **fix-directly via fused-memory** — For memory-integrity findings you can
   safely repair yourself, use the fused-memory write tools, then resolve:
   - `mcp__fused-memory__update_edge` — correct a stale/wrong Graphiti edge fact
   - `mcp__fused-memory__delete_memory` — remove a duplicate/incorrect memory
   - `mcp__fused-memory__merge_entities` — consolidate duplicate entity nodes
   - `mcp__fused-memory__refresh_entity_summary` — rebuild a stale summary
   Then `resolve_issue(..., action='resume', resolution="Fixed directly: <tool + what changed>")`.

   **Caution:** fixing directly mutates the knowledge graph. When the right
   repair is ambiguous or wide-reaching, prefer file-a-real-task or ask the
   human — quality over speed.

**Resolution-text convention:** `action='resume'` → status `resolved` (you took
action). `action='close_only'` → status `dismissed` (accepted-as-known, no action).
Both archive the record. Be specific in the note — it is the only audit trail.

## Per-Category Playbook

- **`recon_integrity_issue`** (info) — memory/task consistency findings. Run the
  action set: verify-fixed → fix-directly → file-a-real-task → accept-as-known.
  High `dedupe_count` = a persistent intractable item; if you've accepted it as
  known before, accept-as-known again briefly. See the caveat below — these
  re-fire every cycle until the recon-side gating fix lands.
- **`recon_stale_run`** (info) — "Run stale, recovered". The harness already
  self-recovered. **accept-as-known** (dismiss) unless several cluster, which
  signals a stuck recon loop → tell the human (and file a DecisionRecord, see
  "Filing Parked Decisions to the Cockpit Registry" below).
- **`recon_failure` / `recon_backlog_overflow`** (blocking) — a reconciliation
  run failed or the queue is overflowing. This is **infrastructure**: tell the
  human with full detail, leave pending, do NOT attempt automated fixes. Also
  file a DecisionRecord via `write-decision`.
- **`infra_issue`** (blocking) — DB/MCP/service problems. **Priority 1 — system
  stability:** tell the human immediately, leave pending, do not auto-fix. Also
  file a DecisionRecord via `write-decision`.
- **`risk_identified`** (info) — needs human judgment. Tell the human, track as
  todo, continue. Also file a DecisionRecord via `write-decision`.
- **`dependency_discovered`** — if a real prerequisite task exists, note it; else
  file-a-real-task. Then resolve.
- **`cleanup_needed`** — file-a-real-task (two-phase), then resolve.
- **`reconciliation_stale_gate_backlog`** (blocking) — a task has sat
  `blocked` on a human-gated milestone gate for 48h+ (Stage 1's aging
  detector, `fused-memory/src/fused_memory/reconciliation/stage1_stall_detector.py`).
  **Default: PARK.** Leave it pending, file a cockpit DecisionRecord via
  `write-decision` (see "Filing Parked Decisions to the Cockpit Registry
  (C8)" below), and append its esc-id to the exclude file
  (`<queue-dir>/.watcher-exclude-recon`) so the watcher's next (re)start
  skips it — not `--exclude-id`, which does not scale at this queue's size
  (see "Re-arming over deliberately-pending items" above).
  **Why park, not resolve:** recon files a fresh gate-backlog escalation for
  a task only when `has_open_l1(task_id,
  category='reconciliation_stale_gate_backlog')` is false
  (`maybe_escalate_stalled_gate_backlog`, same module, lines 483-492) — the
  open pending record itself IS the dedup. Resolving it re-arms the filing
  rule, and if the task is still `blocked` with a `metadata.gate_escalated_at`
  stamp older than 48h, recon re-files `esc-<task>-N+1` on the very next
  Stage 1 cycle (`extract_stalled_gate_backlog_task_ids`, same module, lines
  198-237). This is not theoretical: resolving-to-tidy causes measured
  re-file churn, so do **not** resolve this category just to shrink the
  queue. Observed in `data/reconciliation/escalations/archive/` (as of
  2026-08-02): `esc-650-1` resolved 2026-07-25T12:58Z → `esc-650-2` filed
  and resolved by 2026-07-25T17:03Z the same day (~4h round trip);
  `esc-646-1` dismissed 2026-08-01T14:10Z → `esc-646-2` filed and still
  pending now; `esc-3361-1` resolved 2026-08-02T10:40Z → `esc-3361-2` filed
  and still pending now. As of 2026-08-02 the pending queue holds 32
  records, 100% this category, with only one (`esc-648-1`) ever
  triage-stamped — treat the exact count as a snapshot, not a fixture: it
  moves with every filing/resolve cycle, so re-census
  `data/reconciliation/escalations/` yourself if the number matters to your
  decision. **Resolve only** when the underlying task will genuinely
  stop qualifying for re-selection — you completed the gate and can close
  the task, or recon will reconcile the task to a terminal status next
  cycle regardless. **You cannot drive the gate itself terminal from
  here:** the gate is a born-at-L2 `milestone_gate` escalation filed on the
  *target project's own* orchestrator queue, not this one — resolving the
  recon surfacing changes nothing about the gate or the task. If you find a
  satisfied-but-unclosable gate (the work is done but you have no path to
  close the task), say so explicitly in the cockpit decision text as a
  NO-OP marker, so the next human triaging the queue doesn't have to
  re-derive that conclusion.
- **`reconciliation_stale_human_operator`** (blocking) — a task has stayed
  flagged `human_operator_required` for `STAGE1_HUMAN_OPERATOR_STALL_THRESHOLD`
  = 5 Stage 1 cycles that survived dedup, filed once per task while it has
  no open level-1 escalation (`maybe_escalate_stalled_tasks`,
  `fused-memory/src/fused_memory/reconciliation/stage1_stall_detector.py`).
  Same aging/park shape as `reconciliation_stale_gate_backlog` above —
  **default: PARK**, file a cockpit DecisionRecord, append its esc-id to
  the exclude file on the watcher's next start (not `--exclude-id` — see
  that row above for why); see that row above too for the mechanism and
  churn evidence, not repeated here. **Asymmetry to know:**
  this path dedups on an *un-categorized* `has_open_l1(task_id)` inside
  `maybe_escalate_stalled_tasks` (same module, line 385), so an open
  `reconciliation_stale_gate_backlog` L1 on
  the **same task** CAN suppress a would-be HOR escalation, while the
  reverse can't happen (the gate-backlog lookup is category-scoped).
  Practically: a parked gate-backlog record may be masking a HOR condition
  on that same task — don't read "no pending HOR record for this task" as
  "no HOR condition for this task". **Same resolve-to-tidy trap as above,
  for a different reason:** the stall marker that counts cycles never
  resets (same module docstring, lines 48-59) — resolving a HOR escalation
  while the task is still flagged `human_operator_required` can cause an
  immediate re-file on the very next cycle, since the accumulated cycle
  count is already at or past the threshold.

**Drain-side shortcut — the exclude file only quiets the watcher, not the
drain:** Main Loop step 5's `get_pending_escalations()` has no exclusion
mechanism of its own and returns every pending record each cycle, parked
or not. For an esc-id you've already parked in a prior cycle, don't
re-read and re-reason about it from scratch each time: the cockpit
`--id` is idempotent on the esc-id ("Filing Parked Decisions to the
Cockpit Registry (C8)" below — re-filing the same id overwrites rather
than duplicates), so if you recognize you've already filed a decision for
it, just re-affirm PARK and move on instead of re-deriving the decision
text.

**A note on tiering for these two categories:** the comparison table above
already says "Tiering | **none** — recon files flat, no levels" — that
stays architecturally true, but it is tempting to reach for
`mcp__escalation__promote_to_l2` anyway when a finding feels L2-worthy.
Doing so from a recon-watch session actually **succeeds**: the recon
harness builds its 8103 server with the same `create_escalation_server(...)`
the orchestrator uses
(`fused-memory/src/fused_memory/reconciliation/harness.py:2016`), so
`promote_to_l2` is registered there too, and the identity gate
(`escalation/src/escalation/authority.py:65`) only denies *identified*
callers outside `PROMOTE_ALLOWED` — a header-less `recon-watch/mcp.json`
session is never identified, so the call goes through and mints a real L2
file. **It works, and that is exactly the problem:** nothing automated ever
consumes it. The orchestrator harness supervises `escalation-watcher-auto`
as its own subprocess (`_start_watcher_supervisor`,
`orchestrator/src/orchestrator/harness.py:10572`), and the rotation prompt
it builds binds the watched dir to `self.config.escalation.queue_dir`
(`_run_watcher_rotation`, `harness.py:10618`, queue-dir line at `:10645`
— `data/escalations` behind 8100/8102); it never opens
`data/reconciliation/escalations`. That
does **not** mean the recon queue is invisible to humans — the dashboard's
Escalations tab does render it, as a read-only `reconciliation` subsection
built from this same directory (`build_escalation_queues`,
`dashboard/src/dashboard/data/escalations.py`) — the real gap is that a
promoted record has no automated L2 triage consumer and no L2 cascade
actor, so it just sits there as extra queue noise nobody is watching for.
**This watcher is the actual human-facing consumer of recon findings:** the
sanctioned way to raise one to a human is the cockpit DecisionRecord
(§"Filing Parked Decisions to the Cockpit Registry (C8)" below), whose C5b
decision queue is a surface a human actually works — not `promote_to_l2`.

## Priority Hierarchy

1. **System & infrastructure stability** — never touch anything outside the
   project dir; never kill other processes; if the 8103 server or fused-memory is
   down, ask the human. Never edit queue JSON files by hand — resolve only via
   `mcp__escalation__resolve_issue`.
2. **Software / memory quality** — prefer root-cause repair over papering over.
   When a direct fix is ambiguous or risky, file a task or ask, rather than
   guessing at a graph mutation.
3. **Throughput** — clear-cut closures: act decisively. Ambiguous-and-consequential:
   leave pending, tell the human, track it, move on. For each item deliberately
   left pending, pass `--exclude-id <esc-id>` on the next watcher (re)start so
   the initial scan does not instantly re-fire on it and busy-loop. Also file a
   DecisionRecord via `write-decision` (see "Filing Parked Decisions to the
   Cockpit Registry" below) — IN ADDITION to telling the human.

## Filing Parked Decisions to the Cockpit Registry (C8)

Fleet Cockpit C8 (`plans/fleet-cockpit-prd.md`): every time this skill leaves a finding pending
for the human — `recon_failure`/`recon_backlog_overflow`, `infra_issue`, `risk_identified`, a
clustered `recon_stale_run`, or the general Priority 3 "leave pending, tell the human" case —
also file it to the cockpit decision registry, **IN ADDITION to** telling the human in-session.
This is the same registry `escalation-watcher` files to (`skills/escalation-watcher/SKILL.md`,
"Filing Parked Decisions to the Cockpit Registry"), so the cockpit decision queue (C5b) becomes
the primary return-triage surface across both watchers:

```bash
python3 $DARK_FACTORY_ROOT/orchestrator/src/orchestrator/session_registry.py write-decision \
  --id <stable-id> --project <project> --text "<one-line question>" \
  [--task-id <task_id>] [--escalation-id <escalation_id>] [--severity <esc.severity>] \
  [--escalations-dir $DARK_FACTORY_ROOT/data/reconciliation/escalations]
```

- **`--id`**: a stable id you can recompute idempotently for the same pending item — the
  escalation id (e.g. `esc-recon-abc-1`) is the natural choice. Re-filing the same id overwrites
  the prior record rather than duplicating it.
  **INTERIM RULE — check before you overwrite.** Decision ids are fleet-global, so the
  orchestrator's `escalation-watcher` may already have filed a decision under this id for the same
  underlying human gate. Before filing, check whether a decision for that id already exists and is
  still `open`; if it is, do **not** overwrite it — a second watcher observing the same gate must
  enrich or no-op, never clobber richer context or downgrade an existing record's severity. Park
  your recon record and append the id to your handled set instead.
- **`--text`**: the one-line question a human needs to answer.
- **`--task-id` / `--escalation-id`**: thread through the synthetic `recon-<runid>` task id (if
  any) and the escalation id, so the cockpit can cross-link the decision to its source.
- **`--severity`**: pass through the parked escalation's own severity (`esc.severity` —
  `info`/`blocking`/`critical`/`urgent`). This now weights the cockpit decision-queue rank, so a
  freshly-filed `critical`/`urgent` park surfaces at the top of the queue instead of being buried
  under stale awaiting-input sessions.
- **`--escalations-dir`**: the escalation **queue** your `--escalation-id` belongs to — for this
  watcher, `$DARK_FACTORY_ROOT/data/reconciliation/escalations`. It must name the SAME queue you
  later pass to `reap-decisions` (below). Decision records are fleet-global while an escalation id
  is unique only *within* one queue, and this project runs two of them over the same
  `esc-<taskid>-<n>` namespace, so this is what lets the reaper join a decision back to the right
  per-queue id namespace instead of matching an unrelated same-named orchestrator escalation.
  Stored normalized, so any spelling of the same directory works. Omitting it files a queue-less
  record — see the hazard note below.
- The verb always files `state=open` and is fail-soft (a registry fault is logged and swallowed,
  never raised) — filing a decision can never crash the watch loop or block the "leave pending"
  action itself.

### Closing parked decisions on resolve

A filed DecisionRecord stays `state=open` — and therefore visible in the cockpit decision queue —
until its escalation reaches a terminal status. The watcher that files a decision is almost never
the one that resolves it: resolution typically happens later, via the human acting on the
dashboard or an L2 cascade. So closing a parked decision is a separate, recurring step, not
something the `write-decision` call itself can do.

Once per Main Loop cycle (see step 7 above), run:

```bash
python3 $DARK_FACTORY_ROOT/orchestrator/src/orchestrator/session_registry.py reap-decisions \
  --project <project> --escalations-dir $DARK_FACTORY_ROOT/data/reconciliation/escalations
```

**Note the queue dir**: recon escalations live under `data/reconciliation/escalations/` (the
flat recon queue on port 8103), **not** `data/escalations/` (the levelled orchestrator queue that
`escalation-watcher` reaps). A mis-pointed reaper does **not** merely no-op. Escalation ids
(`esc-<taskid>-<n>`) are unique only *within* a queue and both queues share that namespace, so
pointing `--escalations-dir` at `data/escalations` here makes the reaper **falsely close** this
watcher's decisions against unrelated orchestrator escalations that happen to share an id — the
decision vanishes from the cockpit queue while its own escalation is still pending. Observed with
`esc-3036-1` (task 3528): a blocking recon gate was silently closed and sat invisible for ~7 days;
15 ids were resolved in both queues at time of measurement.

Passing `--escalations-dir` on `write-decision` is what makes the reaper skip cross-queue records:
it stamps the owning queue on the decision, and a reaper scanning a *different* queue leaves that
decision alone. Keep the two dirs straight regardless — a queue-less legacy record (filed before
that flag existed) falls back to project-only scoping and has no such protection.

Two collision modes exist and must not be conflated:

- **MODE 1 — unrelated same-id escalations** (`esc-3036-1`): two different escalations that merely
  collide in the shared id namespace. Closing across queues here is a straight bug, and the queue
  stamp is what prevents it.
- **MODE 2 — same-subject duplicates** (`esc-5914-1`): both queues surfacing the *same* underlying
  human gate. These must collapse to **ONE** cockpit decision — a human asked the same question
  twice is its own regression. That is why the queue is recorded as a *field* on the record rather
  than namespaced into the decision id (`recon:esc-…` / `orch:esc-…` would double-file the same
  question), and why a second watcher must never downgrade an existing record's severity (see the
  interim rule under `--id` above).

This closes (`answered`/`dropped`) any `state=open`, `escalation_id`-bearing decision whose
escalation has since resolved (`resolved` → `answered`) or been dismissed (`dismissed` →
`dropped`) — regardless of who resolved it. It is read-only with respect to escalations (it only
ever writes the decision's own state field) and fail-soft, exactly like `write-decision` — a
registry fault is logged and swallowed, never raised, so it can never crash the watch loop. A
decision filed with **no** `escalation_id` is never auto-closed this way and needs explicit human
closure. Likewise, a decision whose `escalation_id` never resolves to a status — the escalation
was purged by archive retention pruning, or never existed — also stays `open` forever and needs
the same explicit human closure; until then, every cycle repeats a full scan of the escalations
archive looking for it.

## Caveat: recon re-files until the go-forward fix lands

As of 2026-05-27 the recon harness still escalates non-actionable info findings
into 8103 every cycle (the content-fingerprint dedup, A7a/A7b, is ineffective —
the records have no stable identity). The 5,958-item historical pile was bulk-
dismissed (Direction 3); the queue starts at ~90. Until the recon-side fix lands
(stop escalating non-actionable info findings — tracked as a separate task),
expect a steady trickle of fresh `recon_integrity_issue` items. Handle them
efficiently — most are accept-as-known. If the trickle is heavy, remind the
human that the upstream fix is the real lever; you are holding the line, not
solving the source.

## Context Conservation

Long-running session — conserve context. **Delegate to sub-agents**: researching
a finding's current state (search/get_entity/get_task reads), and executing
file-a-real-task MCP calls once you've decided. Keep in top-level context: the
loop, closure decisions, human communication, and which findings are accepted-as-
known so you don't re-investigate them each cycle.

## Failure Modes

- **"Too many open files"** after many watcher restart cycles → fd pool
  exhaustion from accumulated background tasks; tell the user to start a fresh
  session.
- **`get_pending_escalations` errors / empty when you expect items** → confirm
  this session's `escalation` server points at **8103**, not 8102. The stock
  repo `.mcp.json` is 8102.
- **8103 server unreachable** → fused-memory.service is down. Priority 1: tell
  the human; do not start services yourself.
