# Operations

The operator's runbook for running Dark Factory day-to-day: starting and
stopping a project, watching a run, unblocking stuck work, merging, config
reload, model routing, fleet redeploy, nightly maintenance timers, and
troubleshooting.

This document assumes the factory is already installed and at least one
project has been onboarded. For first-time setup see [README.md](README.md)
and [SETUP.md](SETUP.md). For architecture background see
[ARCHITECTURE.md](ARCHITECTURE.md). For the task-metadata authoring
reference (deterministic task kinds, milestones, delivered-checks,
model pins, metadata vocabulary) see
[docs/task-authoring.md](docs/task-authoring.md) — this document only
covers the *operator* surface of those features (config knobs, dispatch
policy tables), not how to author them into a task.

---

## 1. The operating model

Dark Factory runs as a small set of long-lived processes plus a set of
Claude Code skills that a human drives interactively:

- **One orchestrator process per project**, normally a per-project systemd
  user unit (`orchestrator-<name>.service`). Each orchestrator owns its
  own scheduler, merge queue, and escalation server, and drives that one
  project's task tree end to end (plan → implement → verify → review →
  merge).
- **One shared fused-memory process** for the whole fleet — the unified
  memory + task-management backend (Graphiti temporal graph, Mem0 vector
  store, and the task database), consumed by every orchestrator, the
  dashboard, and every interactive Claude session.
- **One shared dashboard process** — a read-only web UI that polls every
  registered project's state (tasks, escalations, merge queue, costs,
  scheduler) and fused-memory's own state (memory, reconciliation).
- **Humans drive the loop via Claude Code skills**, not by calling MCP
  tools directly in the common case: `/prd` to author and queue work,
  `/orchestrate` to run and check status, `/escalation-watcher` to babysit
  a run, `/unblock` to fix a stuck task, `/review` to audit landed code.
  Skills are listed in full in [§2](#2-which-skill-when).

Work escalates up a three-level ladder, and *you* (or your standing
`/escalation-watcher` session) are the top of it:

```
L0  agent's own steward     → tries to resolve the issue itself, budget-capped
L1  escalation-watcher-auto → automated triage rotation (a Claude CLI subprocess
                              the orchestrator's watcher supervisor spawns on demand)
L2  human                   → you, or your long-running /escalation-watcher session
```

An issue a task's own steward can't resolve climbs to L1 (an automatic
triage rotation the orchestrator spawns for itself); if that can't resolve
it either — or if it's a category that's a human call by policy (design
concerns, risk judgments) — it climbs to L2. Some categories are **born at
L2** and skip the climb entirely — filed directly at level 2 because they
represent an infrastructure-level condition (e.g. a scheduler-level
capability check failing repeatedly) that should never wait behind
automated triage. See [§4](#4-watching-a-run) and [§5](#5-unblocking-work).

**Posture:** production operation is the long-running per-project systemd
unit (§3). The `/orchestrate` skill's direct `orchestrator run --config`
invocation is the *ad-hoc* path — first runs, debugging, one-off sessions —
not how a project should run once it's stable and unattended.

---

## 2. Which skill when

All operator-facing workflows are Claude Code skills, invoked as
`/<skill-name>` inside a Claude Code session. This is the core map: what
each does, and when to reach for it.

| Skill | Category | Invoke when | What it does |
|---|---|---|---|
| `/factory-init` | Setup | Onboarding a new or existing repo as an orchestrator target | Picks a project_id + free escalation port, writes the per-project config, registers the project with fused-memory, then routes to review or PRD authoring |
| `/review-briefing` | Setup | Creating or updating a project's review context | Captures durable truths a review can't infer from code alone — purpose, key scenarios, decisions, conventions, known gaps — that seed `/review` |
| `/prd` | Authoring | Writing a new PRD, or decomposing/queuing one into tasks | Author mode (conversational → committed PRD) or decompose mode (gates for a named consumer, an observable leaf signal, a verified substrate, cross-PRD seam ownership, design-first for high-stakes work, and premise validity) before tasks are queued |
| `/orchestrate` | Running | Running the orchestrator, checking status, resuming after a stop | Drives the concurrent plan→implement→verify→review→merge pipeline against a target project's config; also the entry point for checking status and stopping gracefully |
| `/merge-queue` | Running | Merging a task branch to main while the orchestrator might be running | submit→poll protocol against the merge queue MCP tools; falls back to a direct merge only if the escalation MCP is unreachable |
| `/escalation-watcher` | Monitoring / unblocking | Babysitting a running orchestrator | Long-running loop draining and triaging L2 escalations; spawns `/unblock` sessions or resolves categories directly per policy |
| `/recon-escalation-watcher` | Monitoring / unblocking | Babysitting fused-memory's reconciliation queue (port 8103) | Sole closer of `recon_integrity_issue` / `recon_failure` / `recon_stale_run` findings — a separate queue from the orchestrator's task-pipeline escalations |
| `/unblock` | Monitoring / unblocking | A specific task is blocked, stuck, or escalated | Gathers context, investigates in parallel, presents findings to you for a decision, then fixes/merges or resolves the escalation |
| `/spawn` | Monitoring / unblocking | Another skill needs a fresh interactive terminal | Opens a new terminal running the Claude CLI with a given prompt (e.g. escalation-watcher spawning `/unblock <task>` for a human) |
| `/review` | Quality | Post-orchestrator "does it actually work" check | Three-phase deep review: integration verification, architectural coherence, then triage findings into new tasks |
| `/reflect` | Quality | End of a session | Writes decisions, discoveries, and a session summary to fused-memory |
| `/study` | Quality | Before a hard discussion or design decision | Loads a deep, discussion-ready understanding of a specific piece of code |
| `/hotspot-survey` | Quality | Deciding what to refactor based on bug history | Multi-agent survey (~25-30 agents, 60-90 min) mining git/task/postmortem history for bug-cluster root causes, feeding `/prd` |
| `/census` | Quality | Sweeping for confusion sightings against the legibility codebook | Saturation-mines confusion sightings, updates the codebook, files remediation |
| `/do` | Other | You've just agreed on a direction and want it executed autonomously | Distills the conversation into a self-contained plan plus the fixed worktree → `/merge-queue` → `/reflect` execution recipe, run in a fresh context |
| `/warm` | Other | Ad-hoc interactive work that wants a pre-seeded worktree | Claims a copy-on-write warm worktree; falls back to a cold worktree if none is available |

**Internal-only — never type these yourself:** `escalation-watcher-auto`
(the L1 automated triage rotation the orchestrator spawns for itself),
`unblock-auto` (a read-only investigation sub-agent that produces a
risk-labelled fix proposal but mutates nothing), and `unblock-low-risk`
(the unattended counterpart to `/unblock`, launched only by the
escalation-watcher's AFK gate when a proposal is labelled `low` risk — see
[§4](#4-watching-a-run)). These exist as sub-agents other skills invoke,
not as things an operator runs directly.

**Wiring:** `scripts/setup-host.sh` installs the skills via two mechanisms
— flat `~/.claude/commands/*.md` symlinks for most, and whole-directory
`~/.claude/skills/<name>` symlinks for the three that carry their own
`references/`/`scripts/` (`factory-init`, `prd`, `hotspot-survey`). If a
slash command turns out to be missing (e.g. a skill added after your last
bootstrap), re-run the installer or symlink it by hand — see
[SETUP.md](SETUP.md) §"Skill wiring".

---

## 3. Running the factory

### Starting a project (production posture: systemd)

Each project's orchestrator runs as its own `orchestrator-<name>.service`
user unit:

```bash
systemctl --user status orchestrator-<name>.service
systemctl --user start  orchestrator-<name>.service
systemctl --user stop   orchestrator-<name>.service
journalctl --user -u orchestrator-<name>.service --since '1 min ago'
```

The unit's `ExecStart` is always an explicit
`orchestrator run --config <path-to-dark-factory-orchestrator.yaml>` —
**never a bare `orchestrator run`**. This is deliberate: `orchestrator run`
has no cwd auto-discovery and no default config. Running it without
`--config` or `ORCH_CONFIG_PATH` set is a hard, educational error, not a
convenience default — a past incident had an orchestrator run 12 hours
against the wrong project and lose work, hence the guard.

**Cold-start hazard:** an orchestrator started against a task tree with
*zero* `pending` tasks logs an idle banner and sits in "run-until-stopped"
mode waiting for work — it does not crash. But if the unit is freshly
created and something else (a missing dependency, a config error) makes it
actually exit, and the unit is `enabled` with the watchdog probing it, you
get a 60-second crash-loop. **Create + enable + start the unit only after
the project already has ≥1 `pending` task** (a `/prd` batch landed, or
`/review` filed work) — confirm with `orchestrator status --config <path>`
first. If you must install the unit earlier, leave it `disabled` until
tasks exist; the watchdog skips disabled units.

### Ad-hoc / one-off runs

For a first run, a debugging session, or work you want to watch directly
in a foreground terminal, use the `/orchestrate` skill or invoke the CLI
directly:

```bash
cd <dark-factory-repo>
uv run --project orchestrator orchestrator run --config /path/to/target/dark-factory-orchestrator.yaml [--prd <prd.md>]
```

This is the ad-hoc path, not the recommended steady-state posture — once a
project is stable and you want it running unattended, move it to the
systemd unit above.

### Checking status

```bash
cd <dark-factory-repo>
uv run --project orchestrator orchestrator status --config /path/to/target/dark-factory-orchestrator.yaml
```

Prints each task's status and title (`No tasks found.` if the tree is
empty). This does not require a running orchestrator process — it reads
the task tree directly. For status of a *running* orchestrator's live
state, prefer the fused-memory/escalation MCP reads in [§4](#4-watching-a-run)
or the dashboard.

### Stopping gracefully

Send `SIGTERM` to the **innermost `orchestrator` process**, not the shell
wrapper or the `uv` process:

```bash
pgrep -af 'orchestrator run --config'   # find the PID; the chain is bash → uv → orchestrator
kill <orchestrator_pid>
```

On `SIGTERM` the orchestrator: interrupts its main loop, cancels in-flight
agent tasks, then runs its shutdown `finally` block (finalizes metrics,
stops the in-process MCP server, shuts down the merge worker and
escalation server) before exiting. Task results are persisted
**incrementally** as each task completes — not batched at the end — so
completed work survives even an ungraceful termination. If child agent
subprocesses are left orphaned, kill them by specific PID; never use a
broad `pkill` pattern, which can hit other projects' orchestrators running
on the same host.

Stopping via `systemctl --user stop orchestrator-<name>.service` does the
same thing under the hood (systemd sends `SIGTERM`, waits up to
`TimeoutStopSec` before `SIGKILL`).

### The no-auto-discovery rule

There is no default project and no cwd-based discovery, anywhere in the
CLI. Every invocation needs an explicit `--config <path>` or an
`ORCH_CONFIG_PATH` environment variable pointing at that specific project's
`dark-factory-orchestrator.yaml`. Do not set `ORCH_CONFIG_PATH` globally in
your shell profile — it will silently redirect an unrelated invocation at
the wrong project. Prefer a per-project `.envrc` (direnv) instead, and note
that direnv isn't installed by default on a fresh host.

---

## 4. Watching a run

### The standing `/escalation-watcher` session

For any project running unattended, keep one long-running
`/escalation-watcher` Claude Code session open per project. Its loop:

1. Claims a single-owner lease for that project (so two watcher sessions
   never fight over the same queue).
2. Re-arms a background watcher process, then drains
   `get_pending_escalations(level=2)` — only L2 items; L0/L1 are handled
   automatically inside the orchestrator itself.
3. Handles each pending escalation, dispatching by category:
   - `review_issues` / `task_failure` / `wip_conflict` / `unmerged_state` /
     unmatched `dependency_discovered` / blocking `cleanup_needed` → spawn
     an interactive `/unblock` session via `/spawn`.
   - `scope_violation` / matched `dependency_discovered` → resolve
     directly.
   - `design_concern` / `risk_identified` → always a human judgment call —
     never auto-resolved.
4. Waits for the next escalation to fire (or for a spawned sub-agent to
   finish), heartbeats its lease, and repeats.

Its priority order when triaging is: infra stability first, software
quality second, task progress third. Clear-cut issues get acted on
immediately; ambiguous-but-consequential ones are left pending with a
recorded decision rationale and a digest entry rather than guessed at.

**Merges never block the watcher.** Merge requests it issues use the same
bounded `wait_secs=100` submit → poll protocol as everything else (§5) —
the watcher polls with backoff rather than sitting on an unbounded wait.

**AFK mode.** When you've told the watcher (or it's inferred) that no
human is at a terminal, `task_failure` / `review_issues` escalations route
through the **B3 low-risk gate** instead of spawning an interactive
`/unblock`:
- The gate first runs a read-only `unblock-auto` investigation, which
  returns a risk-labelled fix proposal (`risk_label`, `files_referenced`,
  proposal text).
- If the proposal is **fresh** and labelled **`low`** risk, the watcher
  launches `/unblock-low-risk` as a background sub-agent: it re-derives
  the fix, applies it scoped to the referenced files, runs the project's
  verify suite, and merges through the queue — aborting cleanly (leaving
  the escalation pending) on any doubt: non-low risk, a stale proposal,
  scope creep, a rebase conflict, a verify failure, or any merge-queue
  outcome other than `done`.
- Anything else (drift, an aborted low-risk attempt, a capped budget) is
  left pending, written to the project's `afk-digest.md`, and recorded as
  a decision for the next human return-triage pass.
- `unblock-low-risk` never touches main directly, never skips verification,
  and never retries — one clean attempt or a pending escalation.

### `/recon-escalation-watcher` for the reconciliation queue

fused-memory's own reconciliation harness runs a **separate** escalation
queue on port **8103** — integrity findings about the memory/task
reconciliation process itself (`recon_integrity_issue`, `recon_failure`,
`recon_stale_run`), not orchestrator task-pipeline escalations. Keep a
`/recon-escalation-watcher` session running per fleet (it watches the one
shared fused-memory instance, not per-project) — it is the **sole**
closer of this queue; the reconciliation harness never resolves its own
findings.

### Dashboard

The dashboard (typically `http://127.0.0.1:8080`, a shared process for the
whole fleet) polls every registered project every few seconds and presents
these tabs: **Overview**, **Orchestrators**, **Tasks**, **Scheduler**,
**Curator**, **Performance**, **Memory**, **Reconciliation**, **Merge
Queue**, **Costs**, **Burndown**, **Escalations**, and **Analytics**
(escalation analytics). It's read-only situational awareness across the
whole fleet at a glance — reach for direct MCP reads (below) when you need
a specific answer right now.

### Useful MCP reads

These are read-only and safe to call at any time from an interactive
session:

| Tool | Use for |
|---|---|
| `mcp__escalation__get_pending_escalations` | List open escalations (filter by `level`) |
| `mcp__escalation__get_merge_queue` | Live merge-queue snapshot: in-flight/queued items, conflict graph, frozen verify prefix |
| `mcp__escalation__get_merge_halt_status` | Whether the merge queue is currently halted, and by which escalation |
| `mcp__escalation__get_task_runtime_state` | Live per-task phase/loop/attempt projection (what a task is doing *right now*) |
| `mcp__fused-memory__get_tasks` / `get_task` | Full task tree, or one task's full record |
| `mcp__fused-memory__get_statuses` | Compact `{id: status}` map — cheap when you only need status, not full records |
| `mcp__fused-memory__get_status` | Backend health, plus `reconciliation_halt` — whether reconciliation is halted, why, since when, and whether the cooldown has expired. Pass no `project_id` for the fleet-wide `halted_projects` list |
| `mcp__fused-memory__get_queue_stats` | Durable-write-queue counts, plus (per-project) `reconciliation_backlog` **and** `reconciliation_halt` — read both together, see below |

### Is reconciliation halted, or just behind?

A large `reconciliation_backlog` has two causes with **opposite remedies**,
and they look identical from the number alone — that ambiguity cost two days
of mis-triage on 2026-07-20.

- Read `reconciliation_halt.halted` from the **same** `get_queue_stats` /
  `get_status` probe; `halt_reason` and `halted_at` say why and since when.
- **Halted** → `mcp__fused-memory__unhalt_reconciliation(project_id=...)`.
- **Not halted** → it's capacity; the backlog is draining too slowly (task 3049).

`trigger_reconciliation` on a halted project now answers `status='halted'`
(with the reason and remedy) instead of `'requested'` — it used to report
success while the harness skipped every cycle.

This deployment runs `reconciliation.auto_unhalt_after_cooldown: true`
(`fused-memory/config/config.yaml`, where the rationale sits next to the
knob): a halt auto-resumes once its cooldown expires, and the judge re-halts
if the pipeline is still sick. So a halt you find with an already-expired
cooldown is about to clear itself on the next ~5s tick — in that one window a
manual trigger IS consumed, and the tool says so.

---

## 5. Unblocking work

### `/unblock` flow

`/unblock <task-id>` is the standard interactive path for a stuck task
(blocked on review issues, a merge failure, or an open escalation):

1. Gathers context in parallel — the task's worktree artifacts
   (`.task/plan.json`, `iterations.jsonl`, review verdicts), any open
   escalation record, the task record itself, and the worktree's git state
   (diff, rebase state).
2. Runs parallel investigation sub-agents per distinct issue found,
   classifying each as a real blocker or safely deferrable, consulting
   memory for prior similar issues.
3. **Presents findings to you and waits** — this is the human decision
   point; a stop instruction from you at this point hard-aborts the flow.
4. Once you've approved a direction: for a blocked task, it releases the
   workflow, rebases, runs the full verify loop, submits through the merge
   queue with `verified_green=True`, polls to a terminal outcome, then
   marks the task `done` with provenance and cleans up. For an escalated
   task, it resolves the escalation with `action='resume'` (the fix
   addresses the concern) or `action='restart'` (a redesign is needed), or
   does a manual fix-and-merge.
5. Ends with a mandatory `/reflect` — decisions and discoveries from the
   unblock session get written to memory before the session closes.

### Merging: the submit → poll protocol

**Never** `git merge --no-ff` directly into a project's main while its
orchestrator might be running — you'll race the orchestrator's own merge
worker for the same ref. Always go through `/merge-queue`:

1. Rebase your branch on main and run verification (test/lint/typecheck)
   first — this reduces conflict odds and lets you honestly pass
   `verified_green=True`.
2. Confirm the escalation MCP is reachable with any lightweight read (e.g.
   `get_pending_escalations`). If it errors or times out, the orchestrator
   isn't running — fall back to a direct merge (step 4 below) as the
   **only** sanctioned exception to "always use the queue."
3. Submit:
   ```
   mcp__escalation__merge_request(
     task_id="<id>", branch="<id>", worktree="<worktree path>",
     wait_secs=100, verified_green=True
   )
   ```
   `wait_secs=100` is always passed explicitly — it's the server's maximum
   bounded wait, so a fast or idle-queue merge often resolves terminally in
   this same call. A **terminal** result (`done`, `already_merged`,
   `conflict`, `blocked`, `unknown_branch`, `failed`, `superseded`) needs no
   further polling (except `superseded` — see below). A **non-terminal**
   result (`queued`, `attached`) means the request was accepted as durable
   intent; poll for it.
4. Poll `merge_status(request_id)` on a 15s→60s backoff (or the server's
   `eta_seconds` if given) until terminal. `state: "superseded"` means your
   request was absorbed into an in-flight coalesced train — poll
   `superseded_by` instead of falling back to a direct merge, which would
   race the train.
5. **Direct-merge fallback** is only for the escalation MCP being
   genuinely unreachable (orchestrator not running) — never as a shortcut
   when the queue is merely slow.

### `resolve_issue` actions

Resolving an L2 escalation takes an `action`, which maps to a specific
task-status effect:

| Action | Task effect | Use when |
|---|---|---|
| `resume` | → `pending` (resumes in place) | Your fix addresses the concern; let the task continue from here |
| `restart` | → `pending` (restart from scratch) | The approach needs a redesign, not a patch |
| `park` | → `blocked`; the L2 stays **open** | You need to leave it blocked but haven't resolved the underlying issue yet |
| `abandon` | → `cancelled` | The task should not be pursued |
| `close_only` | no task effect | The escalation itself is stale/moot but the task's own status is already correct — closes the escalation record without touching the task |

Always read the returned `resolution_action` back to confirm it matches
what you intended — the parameter ordering matters (`action` before a long
free-text `resolution`) and a mis-ordered call can silently record the
wrong action.

### Merge-halt semantics (`wip_conflict` / `unmerged_state`)

These two escalation categories mean the **entire merge queue is halted**
— no other task can merge until the one escalation that owns the halt is
resolved. The orchestrator tracks exactly one "halt owner" escalation
internally; resolving any *other* escalation, even another `wip_conflict`,
does **not** release the halt.

- `wip_conflict` — the merge queue tripped over uncommitted work sitting in
  the project's main checkout (`project_root`). A recovery branch
  preserving that WIP is named in the escalation detail.
- `unmerged_state` — `project_root` already had unresolved merge markers
  (`UU`/`AA`/`DD`) *before* the merge queue tried to advance — pre-existing
  corruption, not caused by the attempted merge. Needs `git mergetool`,
  manual resolution, or `git reset`, depending on intent.

Resolve the halt-owner escalation specifically (check `get_merge_halt_status`
if unsure which one owns it) — `resolve_issue` on it un-halts the whole
queue. If the log shows the halt cleared but the escalation record still
shows `pending`, that's a genuine bug, not something to dismiss.

---

## 6. Config reload vs restart

`mcp__escalation__reload_config` hot-applies an edited
`dark-factory-orchestrator.yaml` to an **already-running** orchestrator
process — no restart, no dropped in-flight agents or verify suites. It
takes no arguments: it always re-reads that process's own
`ORCH_CONFIG_PATH`, never another project's.

**Green tier (hot-reloadable):**

- Per-role `models` / `budgets` / `max_turns` / `effort` / `timeouts` /
  `backends`
- `routing.*` (`allowed_models`, `ladder`, `per_model_daily_ceiling_usd`,
  `rules` — see [§7](#7-model-routing))
- Steward grace (`steward_completion_timeout`, `steward_lifetime_budget`)
- Scheduler and watcher tuning knobs
- `review.*` checkpoint knobs
- `unblock_auto.*`
- `verify_env`
- The `git.offline_lane_*` leaf tunables
- `config_key_census.*` (the unknown-key census escape hatch — see
  [§6a](#6a-unknown-config-key-census); green-tier on purpose, so a
  false-positive L2 can be cleared on a live unit)

**Red tier (restart-only — the edit is accepted into the file but has no
effect until a full restart):**

- `max_concurrent_tasks`
- Pool sizes / `verify_runners`
- `escalation` bind host/port
- `sandbox.backend`
- `project_root`
- The merge-lane `git.*` structural fields
- `usage_cap.*` (the whole multi-account failover config, including
  `scoped_cap_models` — see [§7](#7-model-routing))

**Always read the returned `applied` / `restart_required` dispositions**
from the reload call — not just its top-level `reloaded` flag. A reload
can report success while individual fields you cared about landed in the
restart-only bucket. See `plans/config-hot-reload-prd.md` for the
authoritative allowlist.

**Fused-memory has its own, separate green tier.** Everything above is the
*orchestrator*'s (`dark-factory-orchestrator.yaml`, applied by
`mcp__escalation__reload_config`). The fused-memory server keeps a second
allowlist over `fused-memory/config/config.yaml`, applied by
`mcp__fused-memory__reload_config` — same tier discipline, different
process, different file. Do not look for one config's leaves in the
other's list. Its green tier currently covers several `reconciliation.*`
leaves (the near-dup/topic-guard knobs and `stale_run_recovery_seconds`),
`write_triage.*`, and the five in-place-update leaves below — read
`RELOADABLE_FIELDS` in `fused-memory/src/fused_memory/config/reload.py` as
the authoritative list rather than treating this enumeration as exhaustive:

- `mem0_update.enabled` — **the kill switch for `update_memory`**, the
  in-place Mem0 amend tool. Green-tier on purpose: this is what you flip
  to stop an in-flight silent-rewrite incident, and a restart-only kill
  switch is no kill switch. Flipping it to `false` denies every caller on
  the next call, allowlisted or not.
- `mem0_update.content_amend_allowed_agent_prefixes` and
  `mem0_update.metadata_patch_allowed_agent_prefixes` — two independently
  configurable `agent_id`-prefix allowlists, both shipping as
  `['recon-stage-', 'curator-']`: `recon-stage-` admits every
  reconciliation stage agent, and `curator-` admits the interactive
  memory-consolidation sitting defined by `skills/curate-fused-memories`
  (esc-3524-1 ruling (b), promoted to an all-deployments schema default on
  2026-08-12; it deliberately holds both arms because retain-and-tag
  stamps retained peers via metadata-only patches). The lists are separate
  because the arms carry different risk: widening
  **`metadata_patch_allowed_agent_prefixes` alone** remains the supported
  way to admit a new interactive tagging flow without granting anyone
  content-amend authority. A mistagged record is cheap to notice and cheap
  to correct; a silent content rewrite is neither. (`agent_id` is
  self-reported, so this is a misuse deterrent for cooperating callers,
  not a security boundary.)
- `mem0_update.storm_threshold` and `mem0_update.storm_window_seconds` —
  the content-amend burst alarm (escalates, never blocks; metadata-only
  calls do not count toward it). Genuinely reload-safe because the shared
  `StormCounter` takes both values per `record()` call rather than
  capturing them at construction.

A full restart is expensive by comparison: a graceful stop (SIGTERM, up to
90s before SIGKILL) cancels in-flight agent tasks and verify suites, then
the cold start pays a warm-lane reseed, a module-tagger pass, and up to a
few minutes waiting on fused-memory to answer. For a pure tuning-knob
change, reload buys you all of that for free — reserve a restart for
red-tier changes.

---

## 6a. Unknown-config-key census

`OrchestratorConfig` uses pydantic `extra='ignore'`, which **discards
unknown keys before validation** — on the top-level model *and* on every
nested one. A misplaced key is therefore accepted silently, with no error
and no log line. On 2026-07-22 a top-level `spare_warm_lanes: 8` in a
project YAML did nothing for three weeks for exactly this reason: the
field lives on `git.`, so the top-level spelling was dropped.

Pydantic can never detect this itself (it never sees the dropped keys), so
a **separate raw-YAML-vs-model pass** walks the project config and reports
keys with no matching model field. It runs in three places, all off the
same single walk:

- **at startup** — a clean census resolves any prior escalation; a dirty
  one files a born-at-L2 (`critical`, `agent_role`
  `orchestrator-config-key-guard`), deduped on the unknown-key-set
  signature so an unchanged key-set files exactly once;
- **on `reload_config`** — the report carries `unknown_config_keys` and
  `ignored_config_keys`, and files/self-heals the L2 symmetrically with
  startup;
- **offline**, via the gate below.

### The offline gate

```bash
uv run --project orchestrator orchestrator check-config \
    --config /path/to/dark-factory-orchestrator.yaml
```

Exit **1** iff at least one genuinely-unknown key is found; exit **0**
otherwise — *including* when excused keys were listed. It calls the census
directly rather than building a validated config, so it still reports
phantom keys when the config has an unrelated value-level validation error
that a full load would raise on first.

Each unknown key may carry a placement hint (`→ did you mean
git.spare_warm_lanes?`). **Hints are advisory**: a hint is a *name* match
against the model tree and can be a coincidental collision. Confirm the
key is really misplaced before moving it — see the worked example below
for a case where following the hint would take the unit down.

### Excusing an intentional key

A project YAML may legitimately carry keys for **non-orchestrator**
consumers — knobs the project's own scripts read. Those must not be
deleted, and must not file a permanent L2. Two opt-outs, both classified
at the same point in the walk:

| Opt-out | Scope | Use for |
|---|---|---|
| Reserved `x_` / `x-` name prefix | any depth, case-insensitive, no config ceremony | **new** non-orchestrator knobs — mirrors the task-metadata Tier-C `x_` namespace in `docs/task-authoring.md` |
| `config_key_census.ignore` | dotted paths in the same YAML, fnmatch globs | **existing** key names other tooling already greps for, where renaming would be a breaking change |

```yaml
config_key_census:
  ignore:
    - 'cpu_governance.*'      # `*` spans dots → whole namespace
    - 'fairness.scheduler_v2' # exact path
    - 'warm_lane_pool'        # top-level dict key — MUST be exact
```

> **fnmatch trap:** `<name>.*` does **not** match the bare parent key
> `<name>`. Opting out a top-level dict key requires listing it exactly.
> Getting this wrong leaves the L2 firing.

Excused keys are still **listed by `check-config`** with their reason (at
exit 0) and reported as `ignored_config_keys` by `reload_config`, so an
over-broad glob stays auditable instead of becoming an invisible blind
spot.

### Worked example: a mixed-consumer namespace

`reify`'s config puts reify-owned knobs in the *same YAML blocks* as real
model fields — `cpu_governance.enabled` is an `OrchestratorConfig` field,
while its siblings `cpu_governance.weights` / `agent_admit` /
`DF_AGENT_CPU_GOVERN` / `fleet_load_detector` are read verbatim by
`scripts/cpu-governed-exec.sh` and friends. Same for
`fairness.skip_threshold` (model) vs `fairness.scheduler_v2` (reify).

Its top-level `warm_lane_pool:` is the cautionary case: the census hints at
`git.warm_lane_pool`, but that field is a **`bool`** and reify already sets
it correctly — the top-level key is an unrelated reify-owned *dict*.
Following the hint would feed a dict to a bool field and hard-fail config
validation, taking the unit down. It belongs in the allowlist (listed
exactly), not moved.

### Clearing the escalation

Fix the config — move/remove a genuinely misplaced key, or excuse an
intentional one — then **restart, or hot-reload** (`config_key_census.*`
is green-tier, see [§6](#6-config-reload-vs-restart)). Either path re-runs
the census, and an empty census **auto-resolves** the pending L2; no manual
resolve is needed. Verify with `check-config` first.

**Scope:** the census walks the **top-level project config only**.
`defaults.yaml` is version-controlled and trusted, and the per-package
`orchestrator.yaml` files found by module discovery are not censused —
a typo in one of those is still silently dropped. Extending the walk to
module configs is a known follow-up, not an oversight.

---

## 7. Model routing

The orchestrator resolves `(model, effort, budget_usd, max_turns)` for
every LLM invocation through one layered resolver
(`orchestrator.routing.resolve_route`). This section is the operator-facing
`routing.*` config surface; the task-author knob
(`metadata.model_overrides`, set per task) is documented in
[docs/task-authoring.md](docs/task-authoring.md).

### Config block (`routing.*`)

```yaml
routing:
  allowed_models: [haiku, sonnet, opus]   # fail-fast admission list
  ladder: [haiku, sonnet, opus]           # weakest -> strongest, for "+N" bumps
  per_model_daily_ceiling_usd: {}         # optional per-model trailing-24h USD ceiling
  rules: []                               # ordered policy table, first match wins
```

- **`allowed_models`** — the fail-fast admission list every claude-backend
  role's configured model, and `unblock_auto.model`, is validated against.
  A model string outside this list raises a validation error at config
  load or reload. Defaults to `haiku`, `sonnet`, `opus`.
- **`ladder`** — models ordered weakest → strongest; consulted **only**
  for a policy rule's ladder-relative `"+N"` bump. A bump clamps at the top
  of the ladder; a model absent from `ladder` can't be bumped from.
- **`per_model_daily_ceiling_usd`** — optional per-model trailing-24h USD
  spend ceiling. Empty by default (no ceiling check, no extra cost-store
  read).
- **`rules`** — the ordered policy-rule table, first match wins. Empty in
  the base schema; any shipped default rule lives in the orchestrator's
  own defaults file so it can be retuned without a code change.

### Closed condition/override vocabulary

Each rule is `{id, match, set}`. Both `match` and `set` reject unrecognized
keys — a typo in a rule raises a structured error at load time or reload
time, rather than silently matching nothing (or everything).

`match` (all optional; a rule matches iff every condition it sets holds):

| Condition | Matches when |
|---|---|
| `role` | The invoking role is a member of this list |
| `task_complexity` | `task_metadata['complexity']` equals this value |
| `task_priority` | `task_metadata['priority']` equals this value — **caveat:** this reads `metadata['priority']`, not the task's top-level `priority` field; you must populate `metadata['priority']` explicitly for this condition to ever fire |
| `plan_min_steps` | The task's plan has at least this many steps (a task with no plan yet fails this and the other plan conditions) |
| `plan_min_modules` | The plan touches at least this many modules |
| `module_prefix` | At least one plan module path starts with this string |
| `min_routing_tier` | The task's persisted routing-tier counter is at least this value |
| `min_dispatch_count` | The task's dispatch count is at least this value |
| `simple_saturated` | The task's routing metadata's `simple_saturated` flag equals this bool |

`set` (all optional — a rule may set any subset; unset fields fall through
to the next-lower layer):

| Field | Notes |
|---|---|
| `model` | Absolute model string, or ladder-relative `"+N"` |
| `effort` | Reasoning effort |
| `budget_usd` | Per-invocation USD budget |
| `max_turns` | Per-invocation turn cap |

### Layered precedence

Resolved highest-precedence first; `effort`/`budget_usd`/`max_turns` are
each resolved independently from the highest layer that specifies them:

1. **Per-task model pin** — a task's `metadata.model_overrides[role]`, if
   set (see [docs/task-authoring.md](docs/task-authoring.md)). Sets
   `model` only.
2. **Policy rule** — the first matching rule in `routing.rules`.
3. **Config** — the role's configured `models` / `budgets` / `max_turns` /
   `effort`.
4. **Role default** — the role's own built-in defaults. Always available
   and unconditional — the resolver never raises, because this layer is
   never subject to the fail-safe validation below.

### Fail-safe validation

Whenever a higher layer would set `model`, the candidate is checked
against `routing.allowed_models` and `per_model_daily_ceiling_usd`. On
failure, that layer's model assignment is skipped — the next-lower layer's
already-validated model is kept — and a reason string
(`model-not-in-allowlist`, `model-ceiling-exhausted`, or
`model-capacity-exhausted` — see below) is recorded on the routing
decision. **A dispatch is never blocked by a routing misconfiguration.**
This validation applies only to claude-backend roles.

### Model-scoped account caps (`usage_cap.scoped_cap_models`)

`usage_cap.scoped_cap_models` (default `['claude-fable-5']`) gives the
multi-account usage gate a per-(account, model) cap dimension layered on
top of its existing per-account failover. An invocation's *scope* is its
model string if that string is in `scoped_cap_models`, else "general".
Setting `scoped_cap_models: []` is the kill switch — no scope state is
ever allocated and the gate behaves exactly like the pre-scope gate.

- A cap hit on a scoped invocation marks only that account's scope — the
  account-level state is untouched, so it keeps serving general-scope
  invocations normally.
- **Scoped failover:** an invocation in scope `m` skips any account whose
  scope `m` is capped (and not yet past its `resets_at`), landing on an
  account with headroom in that scope; the resulting `failover` cost event
  carries `scope: m` in its details.
- An account-level cap (or auth failure) excludes that account for
  **every** scope; a scope cap excludes only that one scope.
- If every account is capped for a given scope, a caller in that scope
  waits on its own wake mechanism (or the routing resolver degrades per
  the fail-safe rule above) — it never freezes the fleet's general
  dispatch.

`usage_cap.*` (including this field) is **restart-tier**, not
hot-reloadable — see [§6](#6-config-reload-vs-restart).

### Observability

Every resolved invocation emits a `routing_decision` event (model, effort,
budget, turns, which layer set the model, which rule matched, any
rejections) and mirrors the latest decision onto the task's own
`metadata.routing` — a bounded history (last 5), a routing-tier counter,
and a `simple_saturated` flag. Today this is what you read directly
(`routing_decision` events, `metadata.routing` per task) — there is not
yet an auto-escalating retry ladder or a rendered rollup panel; see below.

### Adaptive-routing substrate (not yet active)

The resolver already supports the `min_routing_tier` and
`simple_saturated` match conditions and ladder-relative bumps, and the
`metadata.routing` mirror already stores the counters — this is
substrate for a later automated fleet rule (retry-tier escalation,
saturation → full-path). **Not yet live on main:** automatic retry-tier
increment, an automatic "simple-task saturated" stamp, admission of any
model beyond the stock `haiku`/`sonnet`/`opus` set, and a per-(model×role)
rollup panel (done/blocked/cap-hit rates, cost-per-done) in the digest or
dashboard.

---

## 8. Fleet redeploy & watchdog

Three restart mechanisms act on the orchestrator fleet. They're
deliberately kept orthogonal — don't conflate them when debugging a
restart:

- **Liveness = brokenness.** The watchdog's port-probe pass revives a
  wedged or port-down unit immediately: per-unit, uncapped, not gated by
  any fleet-wide clock, and it never stamps that clock. A single
  wedged-unit revive is not a fleet deploy.
- **Staleness = a scheduled fleet deploy.** The watchdog's staleness pass
  is the backstop: capped to at most one fleet-wide redeploy per 8 hours
  (`orchestrator_restart_min_interval_secs`, default 28800) via a shared
  clock file (`data/orchestrator/last_redeploy_orchestrator.json`),
  delegating the actual restart to
  `scripts/restart-all-orchestrators.sh --drain` — which is drain-aware
  (defers a unit that's mid-merge, then force-restarts it after roughly 75
  minutes of continuous busy) and stamps the clock only once the restart
  is fully verified (the script exits 0).
- **Coordinator = the polite, event-driven trigger for that same deploy.**
  It fires on a clean idle window, or force-fires after
  `orchestrator_restart_force_fire_after_secs` (default 4500s / 75 min) of
  eligibility — bypassing the idle/debounce gates — but honors the same
  shared 8-hour clock, so the coordinator and the watchdog backstop never
  both redeploy inside the same window.

Both staleness and the coordinator funnel through the single
`restart-all-orchestrators.sh --drain` chokepoint, so drain behavior and
clock-stamping are defined once and can't drift between the two triggers.

### Reading `--report`

```bash
scripts/orchestrator-watchdog.py --report
```

Strictly read-only — zero mutating `systemctl` calls, no clock write. In
addition to the unit / start-time / newest-watched-commit / verdict
columns, it prints:

- **DEPLOY-AGE** — time since the last *verified* fleet deploy (the shared
  clock), fleet-wide, in hours; `unknown` if the clock has never been
  stamped.
- **MERGE-IDLE** — the unit's current heartbeat classification
  (`idle` / `busy` / `stale` / `absent`) — the same classification the
  drain gate itself uses.
- **WOULD-DEFER** — `yes` iff MERGE-IDLE is `busy`: the one case the next
  drain-aware deploy would actually hold back. `idle` proceeds
  immediately; `stale` / `absent` proceed after a short unknown-grace.

Run `--report` before manually restarting a unit, or to check whether an
upcoming fleet deploy is likely to be held up by an in-flight merge.

### Known gap: the watched list is hardcoded

The watchdog's `WATCHED` list — which (port, systemd-unit-name) pairs it
probes — is a **hardcoded Python list** in
`scripts/orchestrator-watchdog.py`, one entry per project. Onboarding a new
project's supervised unit onto watchdog coverage means editing that list
by hand (and updating the drift test that cross-checks it against each
project's configured escalation port) — it is not auto-discovered from
registered projects. This is fine and common to skip entirely (a unit can
rely on `Restart=on-failure` alone without watchdog coverage) — see
[SETUP.md](SETUP.md) for adding a project to the watched list.

### Soak signal to monitor

The whole scheme assumes the 8-hour window comfortably exceeds the longest
single merge-verify in the fleet (at the time of writing, the fleet's
longest verifies run over 30 minutes but well under 8 hours). If any
project's verify time ever approaches 8 hours, the guarantee that a merge
started right after a deploy survives to the next boundary weakens. This
isn't enforced in code — it's an operational soak signal worth watching as
the fleet grows; see
`plans/orchestrator-fleet-redeploy-throughput-prd.md` for the full
rationale.

---

## 9. Working in the main checkout

The dark-factory repo's own `project_root` checkout is **machine-operated**
— the merge worker, the startup reconciler, and git hooks all act on it
directly, not just interactive sessions. Treat it accordingly:

- For a direct-to-main commit under contention, use
  `git commit --only <path>` (not a bare `git commit`) so you don't sweep
  up unrelated staged or dirty state from a concurrent process.
- `pre-commit` runs pyright three times — pass a generous timeout (five
  minutes or more) to whatever you use to run commit commands, or run it
  detached and poll, rather than letting a default timeout kill it mid-hook.
- **Never run `git stash`** in the main checkout: the stash stack is
  consumed by the merge worker's own advance path, so a stash you push can
  be popped out from under you by an unrelated process. Park work-in-progress
  as commits on a branch instead.
- A pure docs-only commit landing under contention (index lock held by a
  concurrent process) may use `--no-verify` — docs changes don't need the
  code-quality hooks, and retrying past a lock contest is safe for a
  no-code change. Reach for `--only` first regardless.

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Orchestrator won't start; dirty-tree escalation filed | The `project_root` checkout has uncommitted tracked changes at startup (the dirty-tree guard) | If a live human/agent session owns those changes, wait for it — never commit or stash over live work-in-progress. If it's genuinely stray state, resolve manually then restart the unit |
| Orchestrator won't start; systemd shows the unit failed/inactive repeatedly | A stale `orchestrator.lock` file is held by a dead (zombie) PID from a previous crash | `systemctl --user stop orchestrator-<name>.service` then `start` — the stop path clears the lock; starting fresh re-acquires it |
| Newly-created unit crash-loops every ~60s with "No pending tasks" | The unit was enabled/started before any task existed in `pending` (see [§3](#3-running-the-factory)) | Confirm `orchestrator status --config <path>` shows ≥1 pending task before enabling the unit; disable it in the meantime |
| Escalation MCP calls time out or connect-refuse from a Claude session | The target project's `.mcp.json` escalation port doesn't match its `dark-factory-orchestrator.yaml`'s `escalation.port` | Sync the two files — the port is per-project and must match exactly; see [SETUP.md](SETUP.md) for the onboarding step that sets both |
| Reconciliation escalation queue looks permanently poisoned for a project | The project's `project_id` was never registered on fused-memory's `DASHBOARD_KNOWN_PROJECT_ROOTS` before its first task was queued | This must be prevented, not fixed after the fact — always complete project registration (factory-init Stage 6) before queuing any task. If it's already happened, this needs an operator-level fused-memory intervention, not a task-level fix |
| A task blocks at VERIFY with failures unrelated to its own changes | Verification currently runs test/lint/typecheck **repo-wide** by default, not scoped to the task's own modules | Check whether the failures are pre-existing on main outside the task's scope; if so, fix main's cleanliness first (module-scoped verification is planned but not yet implemented) |
| A task blocks at VERIFY with `AttributeError` / assertion mismatches for code it just wrote | Worktrees share the main checkout's `.venv` — Python imports resolve the *installed* (main) package, not the worktree's modified source | Manually verify the worktree's code is actually correct, merge to main, and confirm tests pass post-merge — don't trust the worktree-local verify result for import-level correctness |
| All work across the fleet stalls; every account shows capped | The multi-account usage gate is **all-or-nothing per scope**: capacity in a scope only frees up once at least one account is uncapped for it | Check account-level cap/reset times; this self-heals once any single account resets — it is not a per-request failure to retry individually |
| A burst of zero-output invocations across many tasks at once | Usually a transient upstream 529 (overloaded), not a real failure | Check host health via PSI (`full` pressure), not load average; these typically clear on their own — avoid treating them as a code regression |
| MCP tools stop responding mid-session for every open Claude session | fused-memory was restarted while sessions were live — it's a single shared process, so every session's MCP connection is severed at once | Never restart fused-memory without explicit operator sign-off (it affects the whole fleet, not one project); expect to reconnect/restart affected sessions after a planned restart |

---

## 11. Cost & capacity

**Daily cost ceilings** — trailing-24h USD trip wires that pause the
scheduler (`cost_ceiling_watcher_exceeded` / `cost_ceiling_orch_exceeded`)
when crossed. Both are **restart-tier** (not in the hot-reload allowlist —
unlike the per-role `budgets.*` knobs, which are green-tier; when in doubt,
check `applied`/`restart_required` on your reload call, per
[§6](#6-config-reload-vs-restart)):

| Knob | Default | Scope |
|---|---|---|
| `watcher_daily_cost_ceiling_usd` | `50.0` | All `escalation-watcher` (L1 auto-triage) invocations for the day |
| `orch_daily_cost_ceiling_usd` | `200.0` | All orchestrator invocations (every role) for the day |

**Multi-account failover (`usage_cap.*`)** is entirely optional — a
single-user setup works fine on plain CLI OAuth login with no accounts
file configured. If you do configure a pool
(`usage_cap.accounts_file` → an external YAML of `oauth_token_env` entries
resolved from `.env`), it gives the orchestrator per-account failover when
one account gets capped, plus the per-(account, model) scoped-cap
dimension described in [§7](#7-model-routing). This whole block is
restart-tier, not hot-reloadable.

**Park-and-stop breaker.** If ≥15 distinct tasks transition to `blocked`
within a rolling 1-hour window, the scheduler pauses itself entirely — no
new dispatch, and the merge queue drains what's in flight but accepts
nothing new. This is a circuit breaker against a systemic problem causing
a blocking cascade, not a per-task limit. Recovery is **not** automatic:
an operator must call `mcp__escalation__resume_scheduler` once the
underlying cause is understood and addressed. Check `get_pending_escalations`
first — a park-and-stop trip is usually accompanied by a cluster of related
escalations worth triaging before you resume, not just resuming blind.

---

## 12. Nightly maintenance timers

Recurring maintenance runs as systemd **user** timers, not as orchestrator
work. Each job is the same four files — a committed wrapper `.sh` (all the
logic, so it is testable and runnable by hand), a thin `Type=oneshot`
`.service` whose `ExecStart` is that wrapper, a `.timer` carrying the
cadence, and an idempotent self-verifying `install-*-timer.sh`.

The cadence lives in the unit's `OnCalendar` — **not** in
`dark-factory-orchestrator.yaml`. The orchestrator process is not the thing
being scheduled, and that file loads once at startup with no hot-reload, so
a re-cadence there would cost a cross-repo commit plus a fleet redeploy
instead of a one-line unit edit and a re-run of the installer.

### The nightly ladder

Slots are staggered deliberately: these jobs share one machine and, in two
cases, the same backing stores. **Check this table before adding a job** —
04:00 is already double-booked.

| Slot | Job | Units |
|---|---|---|
| 03:00 | Legibility trickle coder | `legibility-trickle@.timer` |
| 03:30 | fused-memory flag-marker drain | `fused-memory-flag-marker-sweep.timer` |
| 04:00 | Orphaned-worktree reclaim | `reclaim-orphaned-worktrees.timer` |
| 04:00 | Legibility transcript check | `legibility-transcript-check@.timer` |
| 04:30 | reify closure-staleness sweep + drain | `reify-closure-staleness-sweep.timer` |

All timers carry `Persistent=true` (a night missed to a sleeping laptop is
caught up on next boot/login rather than silently skipped) and
`RandomizedDelaySec=300`.

Per-job docs: [docs/flag-marker-sweep-recurring.md](docs/flag-marker-sweep-recurring.md)
for the 03:30 job; the section below for the 04:30 one.

### Nightly reify closure-staleness sweep (04:30)

**What it does.** Runs reify's deterministic-gate closure-staleness sweep,
then drains the re-dispatch requests that sweep emitted — one job, in
sequence, so the consumer always reads a directory the sweep has just
refreshed.

This is a **cross-repo seam**: reify ships the primitive, dark-factory wires
the invocation. reify's
`scripts/deterministic-gate-closure-staleness-sweep.sh` is read-only on all
task state by design (its invariant L6); it adjudicates stranded rows and
writes one request file per confirmed hit into
`/home/leo/src/reify/data/redispatch-requests/`. dark-factory's
`scripts/consume_redispatch_requests.py` performs the actual writes through
the fused-memory MCP server, so every transition goes through the
reconciliation-triggering path.

**The normative contract is the reify script itself** — its
`--emit-requests consumer contract` header block and `_write_request`. Not
this section, and not reify's
`docs/notes/deterministic-gate-closure-staleness-sweep.md` digest. Read the
script before changing either half.

| File | Role |
|---|---|
| `scripts/reify-closure-staleness-sweep.sh` | Wrapper: sweep, then consume |
| `scripts/reify-closure-staleness-sweep.service` | `Type=oneshot` around the wrapper |
| `scripts/reify-closure-staleness-sweep.timer` | `OnCalendar=*-*-* 04:30:00` |
| `scripts/install-reify-closure-staleness-sweep-timer.sh` | Installer |
| `scripts/consume_redispatch_requests.py` | The consumer |

**The three actions**, fixed by the sweep's class of finding:

| Class | Action | Write | Legal row status |
|---|---|---|---|
| `gate_closure` | `close` | `set_task_status('cancelled')` | `blocked` only |
| `merge_verify_red` | `reverify` | clear claimant, then `set_task_status('pending')` | `blocked`, `in-progress` |
| `unmet_dependency` | `redispatch` | clear claimant, then `set_task_status('pending')` | `blocked` only |

The claimant clear goes **first**: once the row reads `pending` a competing
dispatcher may stamp a fresh claimant that a late-landing clear would
clobber (same ordering, and same reason, as the orchestrator's own
stranded-blocked re-dispatch path in `scheduler.py`).

**The guards.** Before each write the consumer re-reads the row and skips
when it is already at the target status (an already-applied request is a
no-op, not a second transition), when its status is outside the class's
legal scope above, or when its `updatedAt`/`heartbeat_at` post-dates the
request file's mtime — the row moved after the sweep observed it, so the
next sweep re-adjudicates. The request body deliberately carries no
wall-clock field (re-emission is byte-idempotent), which is why mtime is the
recency signal. Every uncertainty skips: the fail-safe direction is always
do-nothing.

`--max-writes` (default 5) caps the blast radius. It counts write-bearing
**attempts** — applied *plus* failed — not successes: the re-dispatch path
clears the claimant before it flips the status, so a run whose flips are all
being rejected still mutates every row it touches, and a cap keyed on
successes alone would never engage on exactly that run. The remainder is
reported as deferred and picked up next run.

Applied requests are archived into a `consumed/` subdirectory —
retraction-safe, since the sweep's retraction globs `redispatch-*.json` at
the top level only. A failed apply leaves its file in place so the retry is
immediate rather than waiting on re-emission. If the archive move itself
fails the write still counts as applied and says so loudly; the next run's
guard then skips the file as already-applied rather than re-transitioning
the row.

**Two things the consumer deliberately will not do.**

1. **Re-derive the sweep's predicates.** None of the L1 heartbeat/claimant
   liveness guard, the escalation terminal-allowlist oracle, merge-verify
   ancestry, the dependency roll-up, or the corruption signatures is
   reproduced on this side. There is one implementation of the
   adjudication, in reify, where the evidence lives.
2. **Act on a `CORRUPT-HOLD` row.** The sweep emits only on
   `verdict=STALE` (its invariant L5), so a corrupt-hold row produces no
   file at all — declining to read the sweep's stdout report is sufficient
   to honour it. Those rows need the human git-history adjudication in
   reify's `docs/notes/offline-lane-red-corruption-remediation.md` §4.

**First run: dry-run before arming.** The installer deliberately does *not*
kick an immediate run — unlike its two siblings, this job mutates the live
reify task store. Read the planned writes first:

```bash
python3 scripts/consume_redispatch_requests.py --dry-run \
    --requests-dir /home/leo/src/reify/data/redispatch-requests
```

Then arm the timer:

```bash
scripts/install-reify-closure-staleness-sweep-timer.sh
```

No `dark-factory-orchestrator.yaml` change and no orchestrator redeploy is
involved in either step.

**Reading the output.** Every line is prefixed
`consume_redispatch_requests:` and the run ends with exactly one summary
line — on **every** exit path, including a night that could not reach the
server at all:

```
consume_redispatch_requests: task 5321 (gate_closure): close applied -> cancelled [escalation esc-5321-1 resolved 2026-07-29T04:11Z]
consume_redispatch_requests: SUMMARY applied=2 skipped=7 failed=0 deferred=0 planned=0
```

The bracketed tail on an applied (or `--dry-run` `WOULD`) line is the
sweep's own `evidence` string — the only statement of *why* that travels
with a request, and worth reading before undoing anything, since the file
itself is archived out of the way the moment the write lands.

- `applied` — writes that landed (checked against the tool response, not
  assumed from the absence of an exception: a JSON-RPC `error` envelope, a
  FastMCP `isError` result, and an embedded `success: False` all count as
  failures)
- `skipped` — a guard declined, or the file failed validation; the reason is
  on its own line above
- `failed` — a write was attempted and did not land; the file stays put
- `deferred` — `--max-writes` was reached
- `planned` — `--dry-run` only: writes that *would* have been made

```bash
journalctl --user -u reify-closure-staleness-sweep.service -n 100
systemctl --user list-timers reify-closure-staleness-sweep.timer
```

The service always exits 0 on a valid invocation: a recurring `oneshot` that
can fail enters systemd `failed` state and stays there, silently stopping
the whole nightly job. Per-request failures are reported and counted
instead, so a red run is found by reading the summary line, not the unit
state.

A night that could not run at all logs a `RUN FAILED` line ahead of its
(zeroed) summary, and distinguishes the two cases it could be — `could not
reach the MCP server` (a transport problem: check the fused-memory unit) vs
`aborted on an unexpected error` (a bug in the consumer: read the exception
type on that line). Requests are left in place either way.

---

## 13. One-off: transcript archive gunzip migration

`scripts/migrate_transcript_archive_gunzip.py` converts the agent-transcript
archive (`data/orchestrator/agent-transcripts/`) from `.jsonl.gz` to one
plain, greppable `.jsonl` corpus. Task 3618, leaf α of
`plans/transcript-preservation-seam-prd.md`.

**This is a HUMAN-OPERATED step.** The task's implementer deliberately did not
run `--apply`: an autonomous agent must not delete thousands of live files
(the norm from task 1500, with precedents 1939/1945). The repo change and the
live migration are separate acts, and the second one is yours.

### The contract

| | |
|---|---|
| Default | **Dry-run.** A bare invocation changes nothing. |
| `--apply` | Opts into mutation. Without it, nothing is written or deleted. |
| `--root` | Archive root (default `data/orchestrator/agent-transcripts`). |
| Exit 0 | Every file migrated (or was already migrated). |
| Exit 1 | **At least one file failed** — act on it. Unlike the `gc_agent_transcripts` sweep, which always exits 0 so a watchdog does not alarm, this is a one-off whose failures need a human. |
| Exit 2 | **At least one conflict needs adjudication** (and nothing failed) — see below. Nothing is damaged; a `.gz` was retained next to a longer plain twin. Exit 1 wins when both are present. |

stdout is a single JSON summary (`scanned` / `migrated` / `skipped` /
`conflicts` / `conflict_paths` / `failed` / `failed_paths`); the LOUD lines go
to stderr, so `... --apply | jq` works while failures stay visible.

Per file it decompresses to a staging sibling (`<name>.jsonl.migrate-tmp`),
**reads the result back**, mirrors the `.gz` mtime, atomically renames it over
`<name>.jsonl`, and only then unlinks the source. A file it cannot corroborate
is never destroyed — it is retained, counted and named. The mtime mirror
matters: `gc_agent_transcripts` derives each task dir's retention age from its
newest descendant mtime, so stamping `now` would silently reset the whole
90-day retention window.

Nothing is ever written over a plain `.jsonl` in place, which is what makes the
re-run in step 5 safe when both forms coexist: if the `.gz` turns out to be
damaged, the existing plain twin is left whole rather than being truncated by
the attempt. A hard kill (SIGKILL, power loss) can strand a `.migrate-tmp`
file; it is inert — no reader's `*.jsonl` glob matches it — and the next
`--apply` rewrites it, so `find data/orchestrator/agent-transcripts -name
'*.migrate-tmp' -delete` is cleanup, never recovery.

It is **idempotent and re-runnable**. A `.gz` whose `.jsonl` twin already
reads back cleanly is skipped and its source dropped, so a run you kill
half-way resumes correctly — and so you can re-sweep later residue (see
below).

### The operator sequence

```bash
cd /home/leo/src/dark-factory

# 1. Baseline.
find data/orchestrator/agent-transcripts -name '*.gz' | wc -l

# 2. Dry-run first — READ-ONLY, and real validation: it decompresses and
#    UTF-8-decodes every source without writing anything.
python3 scripts/migrate_transcript_archive_gunzip.py | jq

#    Gate: `scanned` should equal the baseline count and `failed` MUST be 0.
#    A non-zero `failed` names the offending files in `failed_paths` —
#    investigate those before going further.
#    `conflicts` need NOT be 0 to proceed — the sweep leaves those files
#    untouched — but read `conflict_paths` first and resolve them per below.

# 3. Migrate.
python3 scripts/migrate_transcript_archive_gunzip.py --apply | jq

# 4. Confirm.
find data/orchestrator/agent-transcripts -name '*.gz' | wc -l
#    expect 0 — or exactly `conflicts + failed` from step 3. BOTH counters
#    leave their `.gz` on disk: conflicts are the files deliberately retained
#    for you to resolve, and a failed file retains its source precisely
#    because it could not be corroborated. Account for the residue against
#    that sum, not against `conflicts` alone, or a run that develops new
#    failures during `--apply` reads as an unexplained leftover. Resolve
#    both, then re-run 3 and 4.
```

**Run this only AFTER the fleet has redeployed** past the task-3618 merge
(§8). Until every orchestrator restarts, live units keep writing `.gz`, so a
migration run before then leaves transition-window residue. Re-run step 3
after the redeploy to sweep it — that is exactly what idempotency buys.

### Resolving a conflict

That re-run is also what produces conflicts, so expect them on the second
pass. A session in flight ACROSS the redeploy has BOTH forms on disk: the
pre-redeploy process archived `<sid>.jsonl.gz`, then the post-redeploy
producer hook wrote a plain `<sid>.jsonl` for the same session — resumed, and
therefore **longer**.

A residual `.gz` with a longer plain twin beside it is exactly that shape.
**The plain file is the authoritative copy.** The sweep refuses to gunzip over
it (that would overwrite a live transcript with stale content and roll its
mtime backwards, aging the task dir out early), and refuses to delete the
`.gz` on its own judgement — being longer proves the twin is not a truncated
stub, but never proves it *contains* the archived bytes. That last step is
yours:

```bash
# For each path in conflict_paths — confirm the plain twin really is a superset
# before deleting anything.
(
  set -o pipefail
  gzip -t <sid>.jsonl.gz \
    && n=$(zcat <sid>.jsonl.gz | wc -c) \
    && zcat <sid>.jsonl.gz | diff - <(head -c "$n" <sid>.jsonl) \
    && rm <sid>.jsonl.gz
)
```

**`gzip -t` and `pipefail` are load-bearing, not decoration.** A bare
`zcat ... | diff ... && rm` takes its exit status from `diff`, never from
`zcat`. A *damaged* `.gz` still emits a short prefix before it dies, so `diff`
would compare that prefix against the same-length head of the twin, succeed,
and the `&& rm` would delete the only copy of the unrecoverable tail — exactly
the unverified destruction the sweep refuses to do on its own judgement.
`gzip -t` rejects such a file up front; `pipefail` makes the comparison itself
fail loudly if `zcat` dies mid-stream anyway. A `.gz` that fails `gzip -t` is
not a conflict to resolve — it is a damaged archive copy, and the plain twin
beside it is all you have.

If the two disagree, do NOT delete: that is a genuinely divergent pair, and
the archive copy is the only record of the pre-redeploy content.

Expect roughly a 4x expansion (≈485 MB → ≈1.9 GB at the sampled 3.93x ratio),
a one-off cost against ~2.1 TB free.

### The accepted gap between merge and your run

The reader-side changes in task 3618 land at merge, but this migration runs
later. In that window the legibility toolchain **under-reports** the archive:
`_iter_archive_transcripts` now walks `rglob('*.jsonl')`, so a residual `.gz`
is not enumerated at all — skipped rather than loudly failed.

The gap announces itself rather than relying on you to remember it:

- `inventory._iter_archive_transcripts` counts the residue on every walk and
  emits one greppable WARNING — `rg 'residual \*.jsonl.gz'` over the nightly
  logs — naming the count and this runbook.
- `memory_eval_transcript_corpus`'s coverage JSON carries a `residual_gz`
  field, and its report says in words how many transcripts the run could not
  see. That is separate from `transcripts_found` and from `parse_failures`:
  the residue was never found, and never failed to parse.

Both are counts, not reads — no reader reopens a gzip stream — and both go to
zero on their own once you have run the sweep.

This is a known, accepted cost of not letting an agent delete live data, not
a defect. It closes when you complete the sequence above. Validation for the
migration itself was done at full scale: a dry run over all 4,574 live files
decompressed and decoded every one of them with zero failures.

---

## See also

- [README.md](README.md) — what Dark Factory is, quick start
- [SETUP.md](SETUP.md) — installing the factory and onboarding a project
- [ARCHITECTURE.md](ARCHITECTURE.md) — system design and component boundaries
- [docs/task-authoring.md](docs/task-authoring.md) — task-metadata authoring
  reference (deterministic task kinds, milestones, delivered-checks,
  model pins, metadata vocabulary)
- `plans/config-hot-reload-prd.md` — full config hot-reload allowlist
- `plans/orchestrator-fleet-redeploy-throughput-prd.md` — fleet redeploy
  design rationale
- `docs/legibility/design-invariants.md` — the invariants gating PRD
  decomposition and review
