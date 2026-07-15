# PRD: dashboard-task-runtime-endpoint — orchestrator-served per-task runtime state

**Status:** active — 2026-07-15. **Approach B+H** (cross-process contract
between the orchestrator/escalation server and the dashboard; 3-package blast
radius). Contract-resolving PRD: it resolves the long-deferred "single owner"
seam for the dashboard's hand-rolled worktree/artifact reader.

## Goal

The dashboard **Orchestrators** tab (and the Tasks tab detail) shows **real**
Loops / Tries / Age for every project's active tasks — including projects on
the **warm-lane pool** (reify, autopilot-video, …), which today all read
`0 / 0 / age 0`. Additionally it surfaces warm-lane state the dashboard is
currently blind to: which **lane** a task holds, its **phase**, and its **lane
state** (assigned / quarantined / released). When a project's orchestrator is
down, those cells read a legible `—` (offline), never a fabricated `0`.

User-observable surface: `GET /api/v2/dashboard/tasks` returns non-sentinel
`loops`/`attempts`/`started`/`lane`/`phase`/`lane_state` for a warm-lane task
that has actually iterated; the OrchTab table renders them (plus new Lane /
Phase / State columns) instead of zeros.

## Background

The dashboard derives Loops/Tries/Age by scanning `<project_root>/.worktrees/`
and parsing per-task artifacts **by hand** — `dashboard/src/dashboard/data/
orchestrator.py` (`_scan_worktrees`, `_extract_task_id`, `read_task_artifacts`)
feeding `active_tasks.py:_shape_one_project`. `_extract_task_id` only
understands worktree directories named `task-<id>` or `<id>`.

**reify (and any warm-lane-pool project) breaks this join.** Its `.worktrees`
is a symlink to a warm-lane pool whose live worktrees are named `_lane-N`
(`WarmLanePool`, PRD **W11** `worktree-lane-lifecycle-prd.md`).
`_extract_task_id('_lane-7')` → `None`, so every lane worktree is skipped; the
per-task worktree join returns empty; `loops`/`attempts`/`started` all fall
back to `0`. The artifacts exist and are current — they're keyed by **lane**
(`<pool>/.task-meta/_lane-N/…`, and `<pool>/.lane-state/_lane-N.json` carries
the `task_id`) — the dashboard simply can't map task → lane. Root-cause
analysis: `plans/` sibling investigation, 2026-07-15.

This is a **FORMAT COUPLING** failure: `orchestrator.py`'s module docstring
already flags itself as a hand-maintained twin of orchestrator-owned formats
that "must be updated by hand," and names a "FUTURE SINGLE OWNER: stream W11's
TaskArtifacts … this doc block is the marker W11 greps for to find and migrate
this dashboard reader." W11 **declined** that migration
(`worktree-lane-lifecycle-prd.md:267` — "moves no derivation logic"; task ε2
only taught the reader the `.task-meta` path) because dashboard-alignment
decision *no-import-unification* forbids the dashboard from importing the
orchestrator package (`dashboard-alignment-prd.md:36`). So the seam has been
**documented-but-unowned** since W11.

## Sketch of approach

Make the **orchestrator the sole producer** of per-task runtime state and have
the dashboard **consume it over MCP** — never importing orchestrator internals,
so the no-import-unification rule holds while the single-owner aspiration is
finally met.

- **Producer (orchestrator/escalation).** The per-project escalation MCP server
  runs **in-process with the harness** (`harness.py:7105` boots it with
  `harness=self`), so a new tool has a live harness reference. Add a
  `Harness.task_runtime_snapshot()` accessor that, for each active task,
  gathers `{loops, attempts, started, lane, phase, lane_state}` using the
  orchestrator's **own** format owners — `LaneLifecycle` (task_id-keyed
  `.lane-state/<lane>.json` records) for task→lane + lane_state, and
  `TaskArtifacts` (`read_iteration_log`, `read_reviews`, `read_plan`) for
  loops/attempts/phase — reading that host's local disk in-process. The
  accessor is **layout-aware via the harness's own config**: pooled projects
  map through `.lane-state`; non-pooled projects (dark_factory, per-task
  `.worktrees/<id>`) resolve task→worktree directly and report `lane=null`.
  A thin `@mcp.tool() get_task_runtime_state()` in `escalation/server.py`
  delegates to it and projects the declared return schema.

- **Consumer (dashboard).** A per-project MCP fan-out helper
  (`fetch_task_runtime`) clones the existing `merge_halt.py` pattern
  (concurrent, per-call timeout, per-project offline marker) over
  `config.escalation_urls`. `active_tasks._shape_one_project` sources
  `loops`/`attempts`/`started`/`agent`/`lane`/`phase`/`lane_state` from that
  map instead of `_scan_worktrees`. The hand reader
  (`_scan_worktrees`, `_extract_task_id`, `read_task_artifacts`, and
  `discover_orchestrators`' now-unused `worktrees` dict) is **deleted**.
  OrchTab gains Lane / Phase / State columns and renders `—` for a project
  whose runtime call came back offline.

Because the accessor reads local disk **through the orchestrator's own owners**,
even the disk-reading path lives with the format owner — the dashboard twin is
retired. (Serving iteration/phase from a live in-memory `TaskWorkflow` registry
was considered and **rejected**: it touches the hot dispatch lifecycle for no
real gain, since `.task-meta` is the source of truth the workflow already
writes to on the same host.)

## Resolved design decisions

1. **MCP endpoint, not a shared import.** The dashboard consumes runtime state
   via an escalation-server MCP tool, preserving no-import-unification while
   making the orchestrator the single owner. Resolves the W11 "future single
   owner" seam (see Cross-PRD relationship).
2. **Server reads local disk via its own owners (pragmatic), not a live
   workflow registry.** Small, low-risk, no hot-path change. Rejected: a
   task-keyed live `TaskWorkflow` registry on the harness.
3. **Offline is legible, not zero.** An unreachable orchestrator → per-project
   `offline` marker → `—` cells. No silent fabrication of `0`. Honors design
   invariants `structured-facts-at-failure` and the loud-over-silent-degradation
   norm. Consequently the hand reader is **deleted outright** (no disk-read
   fallback path — that would keep the format-coupling bug class alive).
4. **Warm-lane visibility is in-scope.** The endpoint returns `lane` / `phase`
   / `lane_state`; OrchTab surfaces them. This is why the endpoint exists at
   all (the pool is where the dashboard is blind), and each new field has an
   in-PRD UI consumer (G1).
5. **task→lane via `.lane-state` (task_id-keyed), not the branch-keyed
   in-memory `_assignments`.** `WarmLanePool._assignments` is keyed by branch
   (`task/<id>`); the `LaneLifecycle` durable record is the authoritative
   task_id-keyed source and also carries `lane_state`.

## Pre-conditions for activating

**None external.** All substrate exists on main (G3 verified 2026-07-15):
- escalation server in-process with harness, `create_server(harness=…)` —
  `harness.py:7105`, `escalation/server.py:167`.
- `WarmLanePool.assignments_snapshot()` / `LaneLifecycle.read()` +
  task_id-keyed `.lane-state/<lane>.json` — `warm_lane_pool.py:238`,
  `lane_lifecycle.py:182,253`.
- `TaskArtifacts.{read_iteration_log,read_reviews,read_plan,meta_root_for}` —
  `artifacts.py`.
- dashboard per-project MCP fan-out (`escalation_urls`, `mcp_tool_call`,
  session cache, offline pattern) — `config.py:23`, `data/memory.py`,
  `data/merge_halt.py`.

No novel substrate is introduced — the tool is pure infrastructure wiring of
existing owners. **G3: N/A** beyond the verification above.

## Cross-PRD relationship

| Other PRD | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `plans/worktree-lane-lifecycle-prd.md` (W11) | consumes ← | `TaskArtifacts` / `LaneLifecycle` as single format owner; the dashboard-reader migration W11 documented-but-deferred | **this PRD** | resolved here (W11 explicitly "moves no derivation logic"; this PRD performs the migration via MCP) |
| `plans/dashboard-alignment-prd.md` (task ζ) | consumes ← | FORMAT-COUPLING doc block in `dashboard/data/orchestrator.py`; no-import-unification rule | **this PRD** | ζ's doc marker is retired here; the MCP approach honors no-import-unification |

**G4 resolution:** this PRD **owns** the seam. Both prior PRDs left it
deferred; neither is a contested live owner. Companion correction task ζ
(below) updates the `orchestrator.py` FORMAT-COUPLING marker and adds a
retirement note to the two PRDs above, so no stale "W11 will migrate this
reader" pointer survives.

## Contract section (B+H)

**MCP tool** (escalation server, per-project — one call returns that project's
whole active-task runtime map):

```
get_task_runtime_state() -> {
  "offline": false,                 # true only on the dashboard side's marker; server returns tasks
  "tasks": [
    {
      "task_id":      int,          # top-level task id
      "has_worktree": bool,         # a lane/worktree is currently assigned
      "loops":        int,          # >= 0; iterations.jsonl line count (real, non-sentinel)
      "attempts":     int,          # >= 0; count of review verdicts recorded
      "started":      str | null,   # ISO-8601 UTC; task_started / lane metadata created_at; null if unknown
      "lane":         str | null,   # lane name e.g. "_lane-7"; null for non-pooled or unassigned
      "phase":        "PLAN" | "EXECUTE" | "DONE",
      "lane_state":   "assigned" | "quarantined" | "released" | null   # null for non-pooled
    }, ...
  ]
}
```

**Return-shape invariants (machine-checked via a declared schema — a
TypedDict/pydantic model, per invariant `contracts-machine-checked`; not
prose):**
- `loops`, `attempts` are non-negative ints sourced from `TaskArtifacts` reads
  (a real count, never a placeholder). A task that genuinely has not iterated
  reports `0` honestly.
- `started` is a valid ISO-8601 UTC string or `null`; the dashboard computes
  age from it (existing `_minutes_since`).
- `lane`/`lane_state` are `null` **iff** the project is non-pooled or the task
  holds no lane; they are never `null` for an assigned pooled task.
- `phase` is one of the three enum values (coarse, from `plan.json` steps —
  same derivation the dashboard uses today).
- The accessor abstracts pooled vs non-pooled entirely; the **dashboard never
  branches on layout**.

**Offline semantics (load-bearing):** the dashboard fan-out treats a server
that is unreachable / errors / times out as `{offline: true}` **for that
project only** (exact `merge_halt.py` degradation shape). Offline rows render
`—`; the dashboard never substitutes `0`/`unknown` for a real value.

## Boundary-test sketch (B+H) — the integration-gate task's signal

Two-way suite facing **producer** (accessor/tool) and **consumer** (dashboard
join + render):

| # | Scenario | Preconditions | Postconditions asserted |
|---|---|---|---|
| B1 | Pooled task round-trip | fixture harness; `.lane-state/_lane-3.json` → task 42; `.task-meta/_lane-3/iterations.jsonl` has N=3 lines, 1 review PASS | accessor/tool returns task 42 with `loops=3, attempts=1, lane="_lane-3", lane_state="assigned"` |
| B2 | Non-pooled task | fixture harness, per-task `.worktrees/42`, 2 iterations | returns task 42 `loops=2, lane=null, lane_state=null`, phase from plan.json |
| B3 | Quarantined lane surfaced | `.lane-state/_lane-5.json` state=`quarantined` | task's `lane_state="quarantined"` (today invisible in dashboard) |
| B4 | Honest zero | assigned lane, empty iterations.jsonl | `loops=0` (not treated as missing) |
| B5 | Consumer join | dashboard `_shape_one_project` given a mocked runtime map | row `loops/attempts/started/lane/phase/lane_state` populated from the map; no `_scan_worktrees` call |
| B6 | Consumer offline degradation | runtime fan-out returns `{offline:true}` for project P | P's rows carry the offline marker; OrchTab renders `—`, **not** `0` |
| B7 | Hand reader gone | grep the dashboard tree | `_scan_worktrees` / `_extract_task_id` / `read_task_artifacts` are absent (retired) |

Signal for the integration-gate task (ε): `pytest` of B1–B7 green.

## Decomposition plan

Greek labels; task IDs assigned at decompose. Approach B+H → Phase 1 foundation,
Phase 2 vertical slice, Phase 3 integration gate, Phase 4 companion corrections.

- **α — `Harness.task_runtime_snapshot()` accessor.**
  Modules: `orchestrator`. Enumerates active tasks; maps task→lane via
  `LaneLifecycle` (task_id-keyed `.lane-state`) with non-pooled direct-worktree
  fallback; reads loops/attempts/phase via `TaskArtifacts`; started via lane
  metadata / event_store. *Intermediate* — unlocks β. Observable prereq
  unlocked: the MCP tool (β).
- **β — `get_task_runtime_state` MCP tool + declared return schema.**
  Modules: `escalation` (+ shared schema model). Thin wrapper over α; guards
  `harness is None`; projects the contract schema. Deps: α. *Intermediate* —
  unlocks γ, ε. Observable prereq unlocked: dashboard consumption (γ).
- **γ — Dashboard consumes MCP runtime; delete the hand reader.**
  Modules: `dashboard` (`active_tasks.py`, new `fetch_task_runtime` helper,
  `orchestrator.py` deletions). Sources loops/attempts/started/agent + new
  lane/phase/lane_state from the fan-out map; per-project offline marker;
  removes `_scan_worktrees`/`_extract_task_id`/`read_task_artifacts` and
  `discover_orchestrators`' `worktrees` dict. Deps: β. *Intermediate* —
  unlocks δ, ζ. Signal: `GET /api/v2/dashboard/tasks` returns non-zero
  `loops/attempts/started` and populated `lane/phase/lane_state` for a
  warm-lane task with ≥1 iteration; an offline project's rows carry the
  offline marker (not `0`).
- **δ — OrchTab Lane / Phase / State columns + offline `—` rendering.**
  Modules: `dashboard` frontend (`tabs.jsx`; TaskDetail in `tab_tasks.jsx`
  optional). Deps: γ. *Leaf* (user-observable — this is the reported-bug fix).
  Signal: the Orchestrators tab shows real Loops/Tries/Age for reify tasks and
  new Lane/Phase/State columns; when the project's orchestrator is down the row
  shows `—` (offline), not zeros.
- **ε — B+H integration gate: two-way boundary suite B1–B7.**
  Modules: `orchestrator`/`escalation`/`dashboard` tests
  (`test_task_runtime_snapshot`, dashboard join/offline tests). Deps: α, β, γ,
  δ. *Leaf* (C-as-integration-gate for G2 — ropes α/β foundation). Signal:
  `pytest` of B1–B7 green.
- **ζ — Retire the FORMAT-COUPLING reader marker (companion correction).**
  Modules: `dashboard/data/orchestrator.py` docstring + retirement notes in
  `worktree-lane-lifecycle-prd.md` and `dashboard-alignment-prd.md`.
  `metadata.complexity="simple"`. Deps: γ (deletion has landed). *Leaf*.
  Signal: `grep` shows the "FUTURE SINGLE OWNER / W11 greps for this marker"
  block removed/redirected; both PRDs note the reader is retired in favor of
  this endpoint.

DAG: α → β → γ → {δ, ζ}; ε depends on {α, β, γ, δ}.

## Out of scope for this PRD

- Fine-grained phase from `event_store` `phase_enter`/`phase_exit` (coarse
  `plan.json`-derived PLAN/EXECUTE/DONE is retained; see Open questions).
- A live in-memory `TaskWorkflow` registry / zero-disk server path (rejected,
  decision 2).
- Surfacing merge-lane / `_merge-*` / `_mainprobe-*` pool worktrees as tasks
  (they are not task-bearing lanes).
- Any change to the `user_observable_signal` / `consumer_ref` metadata the
  orchestrator does not yet read.

## Open questions (tactical — decide at implementation)

1. **Phase granularity.** Coarse `plan.json`-derived phase vs finer
   `event_store` phase. **Suggested:** ship coarse (parity with today); refine
   later only if a UI consumer wants it. Decide during α.
2. **Schema home.** Declare the return model as a `shared/` TypedDict/pydantic
   twin vs an escalation-local model the dashboard mirrors structurally.
   **Suggested:** a `shared/` model both sides import (dashboard already depends
   on `dark-factory-shared`), so the contract is machine-checked on both ends.
   Decide during β.
3. **TaskDetail (Tasks tab) enrichment.** Whether the Tasks-tab detail pane also
   shows lane/phase, or only OrchTab. **Suggested:** OrchTab is the committed
   consumer; add TaskDetail only if trivial. Decide during δ.
