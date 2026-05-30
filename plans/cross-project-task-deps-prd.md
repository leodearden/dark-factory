# Cross-Project Task Dependencies — PRD

**Status:** active · **Approach:** B+H · **Date:** 2026-05-30 · **Packages:** fused-memory, orchestrator, dashboard

## Goal

Let a task in one project declare a dependency on a task in **another** project, and have the scheduler refuse to dispatch it until that foreign task is `done`. Concretely: a `reify` task can wait on a `dark_factory` task, and reify's orchestrator will not pick it up until the dark_factory task lands. Today this is impossible — `add_dependency` rejects any endpoint outside the same `project_root`, and the scheduler's dependency gate only ever sees one project's task tree — so cross-project sequencing is done by hand, which has repeatedly bitten us (reify features that needed a fused-memory/platform change in dark_factory to land first).

## Background

Each project is isolated across three boundaries, and a cross-project dep has to cross all three:

- **Storage** — per-project SQLite DB at `<project_root>/.taskmaster/tasks/tasks.db`; the `dependencies` table is `(tag, task_id, parent_id, depends_on)` with an **integer** `depends_on` and no project column (`fused-memory/src/fused_memory/backends/sqlite_task_backend.py:83`). Task IDs are per-project autoincrement and collide across projects.
- **Write API** — `add_dependency(id, depends_on, project_root)` validates that **both** endpoints exist in the **same** `project_root` and rejects anything else with `No tasks found for ID(s)` (`sqlite_task_backend.py:1042`).
- **Gate** — the scheduler's `_deps_satisfied()` builds `status_map` from a single `get_tasks(project_root=self._project_root)` and does `status_map.get(dep_id, 'unknown')` (`orchestrator/src/orchestrator/scheduler.py:1444`, `2089`). A foreign dep ID falls to `'unknown'`, which isn't terminal → the task blocks forever, silently.

The asset that makes this **cheap**: fused-memory is a **single shared server** (port 8002) that already serves every registered project's task DB, `get_statuses(project_root=…)` already reads any registered project, and the `project_id → project_root` registry already exists (`fused-memory/src/fused_memory/models/scope.py:154`, `resolve_project_id`/`build_known_projects_map`). The read is already cross-project-capable; nothing consumes it for gating yet.

### Why this approach (Option 2 of 5 considered)

| Option | Why not chosen |
|---|---|
| Manual serialization (status quo) | The recurring pain we're removing; needs a human in the loop every time. |
| Mirror/proxy task + sync bridge | Zero scheduler change, but a permanently-`blocked` mirror generates recon/escalation churn and reads as a phantom-blocked task; needs a separate daemon. |
| **Metadata external-deps + scheduler cross-project read (this PRD)** | Reuses the shared-server read + registry; no schema migration; no new daemon (the existing 15s poll re-evaluates); deps are first-class and durable. |
| Reactive event-bridge resolver | Lower latency but builds genuinely new cross-project event plumbing (durable cursor, replay, missed-event handling) the recurring need doesn't justify yet. |
| Unified global task store | Dynamites the per-project isolation that reconciliation, the dashboard registry, and one-orchestrator-per-project binding rely on. |

## Activation status

**Active** — no blocking prerequisites. Every substrate capability already exists (shared fused-memory server, per-project `get_statuses`, the project registry, task `metadata`, the `update_task(append=true)` write path, the `_mark_blocked(escalate_to_human=True)` escalation pathway). This is pure wiring of existing capabilities plus one new read tool. G3 is therefore near-N/A; the one new surface is the `get_external_statuses` tool (Task α) and its consumption in the scheduler (Task γ).

## Sketch of approach

1. **Declare** — a task carries `metadata.external_deps`, a list of canonical `"project_id:task_id"` strings (e.g. `"dark_factory:13"`). `add_dependency` learns to accept a qualified `depends_on`; a qualified id is routed to `metadata.external_deps` (append-safe), a bare integer stays in the integer `dependencies` table as today. The integer table and its schema are untouched — **no migration**.
2. **Resolve** — a new read-only fused-memory MCP tool `get_external_statuses(deps)` takes a list of `"project_id:task_id"` strings, resolves each `project_id` via the registry fused-memory already owns, reads the foreign project's task status, and returns `{dep: status}` with explicit failure sentinels (`unknown_project`, `unknown_task`, `malformed`).
3. **Gate** — the scheduler's `_deps_satisfied()` reads `metadata.external_deps`, looks each up in a **per-tick batched** external-status cache (one `get_external_statuses` call per `acquire_next` tick covering every pending task's external deps), and folds the result into the satisfied/blocked decision with the policy in the Contract section. A task is dispatchable only if **all** local deps **and** all external deps are satisfied.
4. **Surface** — the dashboard renders a waiting task's `external_deps` and each upstream's resolved status, so a cross-project-blocked task has a legible cause instead of reading as a phantom-blocked task.

The gate lives in the **dependent's** scheduler: when reify's orchestrator runs, it checks dark_factory status; when dark_factory's orchestrator runs it ignores reify tasks (different project). Consistent and one-directional.

## Resolved design decisions

1. **Cancelled-upstream is strict.** A foreign dep resolving to `cancelled` does **not** satisfy the dependent (unlike intra-project, where `cancelled ∈ terminal`). It parks the dependent and raises a human escalation — a cancelled upstream usually means the awaited capability isn't coming, and silently green-lighting is the dangerous outcome.
2. **Status read goes through a new fused-memory MCP tool**, `get_external_statuses`, not scheduler-side resolution. The `project_id→project_root` registry stays single-sourced in fused-memory; the scheduler stays thin; unknown-target handling lives in one place.
3. **B+H.** Blast radius is 3 packages and the change touches the load-bearing scheduler dispatch gate; the contract + boundary-test sketch below pin the seam so the integration task lands first-class rather than starving under the narrow-file-lock orchestrator.
4. **Dashboard visibility ships in this PRD** as its own (pre-split) package task — directly attacking the recurring "phantom blocked" confusion.
5. **One dependency-declaration API.** `add_dependency` is extended rather than adding a parallel API; a qualified `depends_on` (contains `:`) → `metadata.external_deps`, a bare integer → the existing integer table.
6. **Lenient write, gate-time resolve.** `add_dependency` does **not** verify the foreign target exists at write time (the target may be filed later, or in another decompose batch — verifying at write time would reintroduce the write/read asymmetry that has caused recon storms). Existence is resolved at gate time.
7. **Unknown/unresolved never blocks silently forever.** An `unknown_project`/`unknown_task`/`malformed` resolution keeps the task waiting and increments a per-`(task, dep)` unresolved-cycle counter; at a threshold it escalates to a human (reusing the `_check_*_thrash` counter+signature+threshold shape). A transient tool error (timeout, server hiccup) is **not** counted — it's a fail-safe "not satisfied this tick", retried next tick.
8. **Dispatch-time gate only.** External deps gate the *transition into dispatch*; they are not a continuous invariant. Re-opening a foreign upstream (`done → pending`) after the dependent already ran is out of scope (you can't un-run a task).
9. **project_id is canonical underscore** in the dep string (`dark_factory`, not `dark-factory`); the tool normalizes `-`→`_` defensively given the known hyphen/underscore basename asymmetry.

## Pre-conditions for activating

None blocking. Substrate relied upon, all present today:

| Capability | Evidence it exists |
|---|---|
| Shared fused-memory server serving all registered projects | single server on port 8002; `DASHBOARD_KNOWN_PROJECT_ROOTS` registry |
| `project_id → project_root` resolution | `fused_memory/models/scope.py:117` (`resolve_project_id`), `:154` (`build_known_projects_map`) |
| Per-project status read | `get_statuses` MCP tool; `sqlite_task_backend.py` per-project read path |
| Task `metadata` field + append-safe write | task wire shape carries `metadata`; `update_task(append=true)` |
| Escalate-to-human pathway | `_mark_blocked(escalate_to_human=True)` / `_check_*_thrash` pattern |
| Per-tick task-tree re-evaluation | `scheduler.acquire_next()` reads the full tree each tick (`scheduler.py:2071`) |

## Cross-PRD relationship

No cross-PRD seams — this is foundational infrastructure with no contested ownership against another PRD. It **is** cross-**package** (fused-memory + orchestrator + dashboard), which is why the decomposition is pre-split by package per the split-multi-package-before-architect norm.

---

## Contract section (B+H)

### New tool: `get_external_statuses`

```
get_external_statuses(deps: list[str]) -> dict[str, str]
```

- **Input** — `deps`: list of `"<project_id>:<task_id>"` strings. `project_id` is normalized `-`→`_` before registry lookup. `task_id` is a top-level integer id (subtask cross-project deps are out of scope, mirroring the existing `add_dependency` subtask rejection).
- **Output** — a dict keyed by the **input string verbatim** (so the caller can correlate), value is one of:
  - any real task status of the foreign task (`done`, `pending`, `in-progress`, `blocked`, `deferred`, `cancelled`, `merge-deferred`, …)
  - `"unknown_project"` — `project_id` not in the registry
  - `"unknown_task"` — project known, no top-level task with that id
  - `"malformed"` — not parseable as `project_id:task_id`
- **Semantics** — read-only; **no reconciliation side effects**, no event emission (it is a status read, not a transition). Resolves each project via the existing registry; reads status via the same per-project backend `get_statuses` path. Idempotent. Compact like `get_statuses` (status-only, no task bodies).
- **Errors** — registry/DB unavailability raises (transient), it does **not** map to a sentinel — the sentinels are *semantic* "the dep can't be resolved", distinct from *transient* "the server couldn't answer right now". The scheduler distinguishes these (sentinel → grace-then-escalate; raise → fail-safe wait).

### Extension: `add_dependency` qualified-id routing

`add_dependency(id, depends_on, project_root, tag=None)`:
- `depends_on` **contains `:`** → treat as qualified cross-project dep. Validate shape (`project_id:int`), reject self/malformed, then append the canonical-normalized string to `metadata.external_deps` of `id` via the append-safe metadata write. **Do not** verify the foreign target exists.
- `depends_on` is a **bare integer** → unchanged existing behavior (integer `dependencies` table, both-exist-in-project_root validation).
- `remove_dependency` symmetrically removes a qualified id from `metadata.external_deps`.

### Extension: scheduler `_deps_satisfied()` + per-tick batch

In `acquire_next()`, alongside building `status_map` from the local tree, collect the union of `metadata.external_deps` across all pending tasks and issue **one** `get_external_statuses` call; cache the result for the tick. `_deps_satisfied(task, status_map, external_status_cache, …)` then, for each entry in `task.metadata.external_deps`:

| Resolved status | Decision |
|---|---|
| `done` | satisfied |
| `cancelled` | **not** satisfied → `_mark_blocked(escalate_to_human=True)` (strict policy) |
| `unknown_project` / `unknown_task` / `malformed` | not satisfied; increment per-`(task, dep)` unresolved counter; at threshold → escalate; else keep waiting silently |
| any other live status | not satisfied; keep waiting |
| (batch call raised) | not satisfied this tick (fail-safe); no counter increment; retry next tick |

### Invariants

1. A task is dispatched only when every local dep **and** every external dep is satisfied.
2. A `cancelled` foreign upstream or a persistently-unresolvable dep **always** surfaces as a human escalation — never a silent forever-block.
3. `get_external_statuses` never mutates state (no transitions, no reconciliation events).
4. External-dep checks gate dispatch only; they are not re-evaluated after a task has been dispatched.
5. Exactly one `get_external_statuses` call per scheduler tick regardless of how many tasks/deps are pending (no per-task fan-out).
6. A transient resolver error degrades to "wait and retry", never to a false-satisfied or a spurious escalation.

## Boundary-test sketch (B+H)

Each row faces **both** the producer side (`get_external_statuses` / `add_dependency` in fused-memory) and the consumer side (scheduler gate). These are the integration-gate task's (δ) observable signal.

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| 1 | Foreign dep done → dispatch | reify task T has `external_deps=["dark_factory:N"]`; DF#N `done` | `get_external_statuses` returns `done`; scheduler dispatches T |
| 2 | Foreign dep pending → no dispatch | DF#N `pending` | returns `pending`; T not dispatched; no escalation |
| 3 | Foreign dep cancelled → escalate | DF#N `cancelled` | returns `cancelled`; T not dispatched; human escalation raised |
| 4 | Unknown project → grace then escalate | `external_deps=["nope:1"]` | returns `unknown_project`; T waits; escalation after threshold cycles, not before |
| 5 | Unknown task in known project | `external_deps=["dark_factory:999999"]` | returns `unknown_task`; same grace-then-escalate |
| 6 | Malformed dep | `external_deps=["garbage"]` | returns `malformed`; grace-then-escalate (and `add_dependency` should have rejected it at write time) |
| 7 | Mixed local + external, all satisfied | T has a local dep (done) and `dark_factory:N` (done) | dispatched |
| 8 | Mixed, external unsatisfied | local dep done, `dark_factory:N` pending | not dispatched |
| 9 | Batching | 3 pending tasks reference 5 distinct foreign deps | exactly one `get_external_statuses` call that tick |
| 10 | Hyphen-form normalization | `external_deps=["dark-factory:N"]` | resolves identically to `dark_factory:N` |
| 11 | Transient resolver error | `get_external_statuses` raises (e.g. timeout) | T not dispatched this tick; no counter increment; next tick retries |
| 12 | Write routing | `add_dependency(reify T, "dark_factory:N")` then `get_task(T)` | `metadata.external_deps` contains `"dark_factory:N"`; integer `dependencies` table unchanged |

---

## Decomposition plan

Pre-split by package. Greek labels; actual task ids assigned at decompose. Phase 1 = fused-memory foundation; Phase 2 = orchestrator gate + the end-to-end integration gate (the leaf, C-as-integration-gate); Phase 3 = dashboard; companion correction = docs.

### Phase 1 — foundation (fused-memory)

**α — Add `get_external_statuses` MCP tool**
- *Modules:* fused-memory (`server/tools.py`, backend read path, registry)
- *Observable signal:* calling the tool with `["dark_factory:<real>", "nope:1", "dark_factory:999999", "garbage", "dark-factory:<real>"]` returns, respectively, the real status, `unknown_project`, `unknown_task`, `malformed`, and the same real status as the underscore form. (API/service response difference.)
- *Type:* intermediate → unlocks γ. *Prereqs:* none.

**β — Route qualified `depends_on` to `metadata.external_deps` in `add_dependency`/`remove_dependency`**
- *Modules:* fused-memory (`server/tools.py`, `backends/sqlite_task_backend.py`, `middleware/task_interceptor.py`)
- *Observable signal:* `add_dependency(id=T, depends_on="dark_factory:N", project_root=<reify>)` then `get_task(T)` shows `metadata.external_deps` contains `"dark_factory:N"` and the integer `dependencies` table is unchanged; a self/malformed qualified id is rejected with a specific error; `remove_dependency` removes it. (Persisted-state change via the read path.)
- *Type:* intermediate → unlocks δ, ε. *Prereqs:* none.

### Phase 2 — gate + integration (orchestrator)

**γ — Extend scheduler `_deps_satisfied()` + per-tick batch resolution + escalation policy**
- *Modules:* orchestrator (`scheduler.py`)
- *Observable signal:* none in isolation (foundation logic) — roped into δ.
- *Type:* intermediate → unlocks δ. *Prereqs:* α.

**δ — Integration gate: cross-project dispatch behavior end-to-end** *(the leaf)*
- *Modules:* orchestrator (integration test harness spanning two project task DBs)
- *Observable signal:* the boundary-test sketch scenarios pass against a two-project setup — specifically rows 1 (dispatch on upstream done), 2 (no dispatch while pending), 3 (cancelled → escalation), 4/5 (unknown → grace-then-escalate), 9 (single batched call), 11 (transient error → fail-safe wait). The user-observable surface: a dependent task that does **not** appear as dispatched while its upstream is unfinished, then **does** on the tick after the upstream goes `done`, observed through the scheduler/orchestrator's own dispatch path (not by peeking at storage).
- *Type:* **leaf** (integration gate). *Prereqs:* β, γ.

### Phase 3 — visibility (dashboard)

**ε — Render external deps + upstream status on a waiting task**
- *Modules:* dashboard (API + JSX)
- *Observable signal:* for a task with `external_deps`, the dashboard shows each dep id and its resolved upstream status, observed through the dashboard's own read path (Playwright/API), with no synthetic data when a dep can't be resolved (show the sentinel, e.g. `unknown`, not a fabricated status).
- *Type:* leaf. *Prereqs:* α, β.

### Companion correction — docs

**ζ — Document cross-project deps**
- *Modules:* docs (`CLAUDE.md` Task Routing section; `skills/prd/references/decompose-mode.md` cross-PRD wiring note; mention the qualified `add_dependency` form)
- *Observable signal:* `CLAUDE.md` and the decompose-mode reference name `get_external_statuses`, the `metadata.external_deps` field, and the qualified `add_dependency` form; a grep for `external_deps` in docs returns the new prose.
- *Type:* leaf (companion correction). *Prereqs:* δ.

### DAG

```
α ──► γ ──► δ ──► ζ
β ──────────►┘
α,β ─────────► ε
```

## Capability manifest (draft — committed beside the PRD at decompose)

Per-leaf capability→evidence bindings (G3+G6 mechanization). Re-checked at decompose; FAIL blocks the batch.

- **α (leaf for its own signal):**
  - registry resolution `project_id→root` → `grep:fused_memory/models/scope.py:117,154` (wired, on main) — PASS
  - per-project status read → existing `get_statuses` backend path — PASS
  - `unknown_task` distinguishable (project known, task absent) → per-project DB query returns empty for absent id — PASS
- **β:**
  - `metadata.external_deps` population → producer (`add_dependency`) writes a **non-sentinel** list value via append-safe metadata write; verify via `get_task` read path (field-population sub-check) — PASS
  - integer-table path unchanged → existing `add_dependency` validation untouched for bare ints — PASS
- **δ (integration-gate leaf):**
  - `get_external_statuses` (capability) → `producer:α`, upstream in dep closure — PASS (DAG-direction: δ depends on α via γ)
  - `metadata.external_deps` write/read → `producer:β`, upstream — PASS
  - scheduler gate logic → `producer:γ`, upstream — PASS
  - escalation pathway → `grep` `_mark_blocked(escalate_to_human=True)` on main — PASS
- **ε:**
  - reads `external_deps` field → `producer:β`, upstream — PASS
  - reads upstream status → `producer:α`, upstream — PASS
- **ζ:** docs only — N/A.

No leaf asserts a numeric bound or closed-form exactness (tooling domain) — G6 branches 1/2 N/A; branch 3 (dependency-direction) clean across all leaves.

## Out of scope

- **Reopen-after-run** — a foreign upstream going `done → pending` after the dependent already dispatched. The dependent isn't re-gated.
- **Cross-project cycle detection** — no global cycle detector; usage is expected to be one-directional (consumer → platform).
- **Subtask cross-project deps** — mirrors the existing `add_dependency` subtask rejection; only top-level foreign tasks.
- **Schema migration / qualified ids in the integer `dependencies` table** — deliberately avoided; foreign deps live in metadata.
- **Reactive/push resolution** (event bridge) — this PRD is pull-based via the existing poll; a push upgrade is a separate future PRD if latency ever matters.
- **Cross-project *priority*/scheduling fairness** — only the dependency gate, not how two projects' orchestrators share capacity.

## Open questions (tactical — decide at impl time)

1. **Unresolved-cycle escalation threshold.** Default to the same value the existing `_check_*_thrash` guards use (reset-to-1, threshold → escalate). Confirm the exact N when implementing γ. *Suggested:* match the nearest existing thrash guard.
2. **Escalation payload shape for cancelled/unresolved external deps.** Reuse the standard blocker escalation with a reason prefix (e.g. `EXTERNAL_DEP_CANCELLED` / `EXTERNAL_DEP_UNRESOLVED`) so the watcher can route it. *Suggested:* add the prefixes; decide exact strings in γ.
3. **`metadata.external_deps` write must survive a status cycle.** Per the known `set_task_status` replaces-whole-metadata-blob hazard, ensure β's write and any later status transitions use `append=true` / read-modify-write so external_deps isn't nuked. *Decide in β.*
4. **Dashboard refresh of upstream status.** Whether ε resolves upstream status live per render or piggybacks on an existing dashboard poll. *Suggested:* piggyback on the existing poll; tactical.
