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
7. Go to 2
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
  --queue-dir $DARK_FACTORY_ROOT/data/reconciliation/escalations 2>&1
```

Run as a **background task** (`run_in_background`). **No `--level`** — recon
escalations have no level field worth filtering. The watcher uses inotify and
exits after the first new escalation file, printing its JSON to stdout.

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
  signals a stuck recon loop → tell the human.
- **`recon_failure` / `recon_backlog_overflow`** (blocking) — a reconciliation
  run failed or the queue is overflowing. This is **infrastructure**: tell the
  human with full detail, leave pending, do NOT attempt automated fixes.
- **`infra_issue`** (blocking) — DB/MCP/service problems. **Priority 1 — system
  stability:** tell the human immediately, leave pending, do not auto-fix.
- **`risk_identified`** (info) — needs human judgment. Tell the human, track as
  todo, continue.
- **`dependency_discovered`** — if a real prerequisite task exists, note it; else
  file-a-real-task. Then resolve.
- **`cleanup_needed`** — file-a-real-task (two-phase), then resolve.

## Priority Hierarchy

1. **System & infrastructure stability** — never touch anything outside the
   project dir; never kill other processes; if the 8103 server or fused-memory is
   down, ask the human. Never edit queue JSON files by hand — resolve only via
   `mcp__escalation__resolve_issue`.
2. **Software / memory quality** — prefer root-cause repair over papering over.
   When a direct fix is ambiguous or risky, file a task or ask, rather than
   guessing at a graph mutation.
3. **Throughput** — clear-cut closures: act decisively. Ambiguous-and-consequential:
   leave pending, tell the human, track it, move on.

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
