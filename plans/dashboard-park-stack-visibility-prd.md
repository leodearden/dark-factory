# PRD — Dashboard module-park-stack visibility + stranded-park eviction

**Status:** authored 2026-06-23 · **Project:** dark_factory · **Approach:** B + H (contracts + two-way boundary tests)

## 1. Consumer + user-observable surface (G1)

**Read path.** The dashboard **Scheduler tab** (`dashboard/src/dashboard/static/redux/tab_scheduler.jsx`) and its
`/api/v2/dashboard/scheduler` endpoint. Today it joins the scheduler snapshot's `parks` dict against active
tasks and renders a per-task `park_state` (heatmap "parked-by-me/-other" + a Modules-view badge + park age).

**Write path.** The live `Scheduler` / `ModuleLockTable` in the long-lived orchestrator process, reached **only**
via a DB the scheduler already polls each tick — there is no live RPC from the dashboard into the scheduler.

**User-observable surface.** An operator looking at the Scheduler tab sees, per contended module, the *full*
LIFO park stack (active top **and** the shadowed owners beneath it), with each owner's rank/tier, age, and
live-or-dead status; a **stranded-parks alert** when a module's top owner is not a live task; and a guarded
**"evict"** control on each stranded row that clears the ghost park.

**Shared lock-chip surface (added per operator request).** The *general* module-lock display chip — the unified
entity (task 1508) rendered in the **Tasks tab's Task-detail card** (`tab_tasks.jsx:334-342`, inline) and the
**Orchestrators view** (`tabs.jsx:117 LockChip`), both fed per-module by `buildSchedLockInfo`
(`scheduler_utils.jsx`) — today colors a module only `lock-free` (grey, "available") or `lock-taken` (red,
"held by T-N"). It gains a third state: **orange `lock-parked`, naming the park-owning task**, shown when a
module is parked but **not** held — i.e. orange replaces grey for parked-but-unheld modules, mirroring the
existing red claimed mode. Precedence: held (red) > parked (orange) > available (grey), so orange only ever
displaces the grey/available case the operator called out.

## 2. Motivation / premise (G6)

The dashboard only *partially* shows scheduler module-lock "parks". Two **structural** blind spots — the exact
pair that made the 2026-06-22 strand incident invisible ("deps satisfied, no live holder, so dashboard shows
nothing", memory `project_park_stack_fix_2026_06_22`):

1. **Buried (shadowed) park owners are hidden by design.** `ModuleLockTable.snapshot_parks()`
   (`orchestrator/.../scheduler.py:748-772`) reports **only the active top** of each per-module LIFO stack
   (INV-7). Every shadowed owner beneath the top — the whole point of the per-module park stack added by
   df 1865 — is invisible to the snapshot, hence to the dashboard.
2. **Orphaned/stranded parks render nothing at all.** `dashboard/src/dashboard/data/scheduler.py:255` joins
   parks over `active_tasks` **only**. A park whose owner is not a live task (the "immortal partial park")
   produces **no row** — the loop never visits it. That is the literal "dashboard shows nothing".

df 1865 fixed the *cause* of that specific strand (destructive eviction → per-module LIFO stack). This PRD
closes the **monitoring** gap that let it go undiagnosed, as a standing guardrail against any future strand
(other causes, regressions), and adds an operator remediation lever.

**Premise validity.** Both blind spots are verified facts of the current code (read this session), not
conjecture. No leaf signal asserts a numeric bound, exactness, or accuracy claim — branches G6-1/G6-2 are N/A.
The one negative/guard assertion (evict must **refuse** a live owner) is bound to an explicit rejection
mechanism in the contract below and tested in task δ.

## 3. Verified substrate (G3)

All confirmed by reading code this session — no novel **assumed** substrate; every *new* capability is filed as
a task in §7:

| Assumed capability | Evidence it exists today |
|---|---|
| Snapshot is a pass-through producer→disk→MCP→dashboard | `Scheduler.get_state_snapshot()` (`scheduler.py:3419`) builds the dict; orchestrator writes `data/orchestrator/scheduler_state.json`; `read_scheduler_state()` (`fused-memory/.../mcp_tools/scheduler_state.py:51`) returns it verbatim via `json.loads`; `get_scheduler_state` MCP tool serves it. Adding a new top-level key flows end-to-end with only the `_empty_skeleton()` default to add. |
| LIFO park stack with per-owner install timestamps | `ModuleLockTable._parked: dict[str, list[tuple[owner, rank]]]` + `_park_install_at` (`scheduler.py:450,452`). |
| Per-owner eviction with LIFO restoration | `ModuleLockTable.prune_owners(predicate) -> (evicted, restored)` already used by the park-GC sweep (`scheduler.py:2959-2969`); `reservation_evicted` EventType retained (`scheduler.py:2465`). `force_clear(owner)` is a thin wrapper. |
| A DB the scheduler polls every tick (the only viable control channel) | The override path: scheduler reads `self._override_store.get_overrides(...)` each tick (`scheduler.py:3110`) and consumes-and-clears one-shot flags exactly like `reserve_now` (`scheduler.py:3143-3173`). The new request table mirrors this **lifecycle** (drain → act → delete) without touching the override schema. |
| Liveness oracle for stranded detection | fused-memory `get_statuses` / `get_tasks` already feed the dashboard's task data; the dashboard already knows which task_ids are active. |

**Hazard noted (drove the §6 decision).** The `overrides` table schema is **hand-mirrored across three
packages with no shared import**: `orchestrator/src/orchestrator/overrides.py` (`class OverrideStore`),
`fused-memory/.../server/tools.py:71` (a re-declared `CREATE TABLE overrides` DDL + raw-SQL writes — explicitly
"OverrideStore (no import) — the same schema"), and a duplicated `_VALID_CLEAR_FIELDS` in both `tools.py` and
dashboard `app.py`. Overloading it is high-blast-radius on a render-equality-tested path — see §6.

**The `parks` snapshot key is left untouched** (backward compat — dashboard contention counts and
`get_scheduler_state` consumers depend on its INV-7 active-top-only shape). The new state is an **additive**
`park_stacks` key.

## 4. Resolved design decisions

- **D1 — Placement: extend the existing Scheduler tab**, not a new top-level tab. Parks are scheduler state; the
  Scheduler tab already owns module contention and per-task park badges. Add a "Module Park Stacks" section + a
  stranded-parks alert; the alert may also bubble to Overview (out of scope for v1, noted in §8).
- **D2 — Scope: read display + stranded detection + a guarded evict action** (operator remediation included).
- **D3 — Evict control channel: a dedicated single-purpose `park_eviction_requests` table**, drained and
  deleted by the scheduler each tick. Mirrors `reserve_now`'s consume-and-clear **lifecycle** without overloading
  its **table**. Narrow 3-column contract; isolates blast radius from the shared override schema. (Alternative
  considered: add a `force_evict_parks` column to `overrides` — rejected for the §3 hazard: it spreads across the
  hand-mirrored DDL in two packages + dual validation sets + the dashboard, risking regressions in the live
  pin/boost/reserve/ttl path for a rarely-used guardrail.)
- **D4 — Authoritative safety guard lives in the scheduler, not the UI.** The evict consumer **must refuse** to
  clear parks for an owner that is currently a **live, dispatchable** task, emitting a distinct
  `reservation_force_evict_refused` audit event. This is load-bearing: an unguarded force-clear of a *live*
  owner's park would re-introduce exactly the destructive-eviction starvation that df 1865 fixed. The UI guard
  (button enabled only for verified-stranded rows) is defense-in-depth, not the authority.
- **D5 — Eviction granularity is per-owner.** `force_clear(owner)` evicts *all* of that owner's parks across all
  modules, with LIFO restoration of any live shadowed owner beneath (handled by `prune_owners`). A stranded
  *module* whose dead top shadows a live owner is healed by restoring that owner — the desired behavior, already
  implemented in `prune_owners`.
- **D6 — Shared lock-chip "parked" state is data-driven from a new per-module `parked_by` field, not a new
  client call.** The data layer (`_module_contention_counts`) attaches the active park owner per module (resolved
  by the same `modules_conflict` prefix rule it already uses for `holder`), so the chips read `m.parked_by`
  exactly as they read `m.holder` — no change to `buildSchedLockInfo` or a new endpoint. Producing this field is
  folded into γ (it owns all `data/scheduler.py` park-shaping, avoiding a second task editing that file); the chip
  rendering is task η. The orange chip needs only the *active-top* park owner (already in the existing `parks`
  key) plus liveness; it does not strictly require `park_stacks`, but derives from γ's already-computed park view
  for consistency.

## 5. Contract (H) — seam signatures + invariants

**Read seam — `park_stacks` snapshot key (producer = task α, consumers = β, γ):**
```
park_stacks: dict[module: str, list[entry]]   # full LIFO, bottom→top; last element = active top
entry = {
  "owner":        str,    # owner task_id
  "rank":         int,    # priority rank (lower = higher priority); strictly decreasing top-ward (INV-3, depth ≤ 5)
  "shadowed":     bool,   # True for every entry except the active top
  "installed_at": str,    # ISO8601 from _park_install_at[owner], or "" if unknown
}
```
- INV: `parks` (existing, INV-7 top-only) is unchanged and remains the source for contention counts.
- INV: `park_stacks` is a fresh, non-aliasing copy (callers cannot mutate `_parked`).
- INV: a module absent from `park_stacks` has an empty stack.

**Read seam — per-module `parked_by` on the SCHEDULER `modules` array (producer = task γ, consumer = task η):**
```
module_entry += {
  "parked_by":         str | None,   # active park owner task_id for this module (None if unparked)
  "parked_by_project": str | None,   # project scoping the park owner (ids are project-scoped)
  "parked_owner_live": bool,         # False ⇒ stranded (owner not a live task)
}
```
- INV: `holder` is unchanged; `parked_by` is additive. A module may have both a `holder` (live claim) and a
  `parked_by` (reservation behind it); the chip renders `holder` (red) with precedence over `parked_by` (orange).
- INV: `parked_by` is resolved by the same `modules_conflict` hierarchical-prefix rule as `holder`.

**Write seam — `park_eviction_requests` table (producer = task ε via MCP, consumer = task δ scheduler tick):**
```
park_eviction_requests(task_id TEXT, project_root TEXT, requested_at TEXT)   # one row per request
request_park_eviction(task_id, project_root) -> {"requested": true, "task_id": ...}   # MCP tool, inserts a row
```
- INV (drain): each scheduler tick reads all rows, and for each **deletes the row** (one-shot; never re-processed).
- INV (guard, D4): the tick calls `force_clear(task_id)` **iff** `task_id` is not a live dispatchable owner;
  otherwise it emits `reservation_force_evict_refused` and clears nothing.
- INV (restoration): `force_clear` removes the owner from every stack and restores newly-exposed live tops via
  the existing `prune_owners`/`_remove_owner` LIFO machinery; emits `reservation_force_evicted{owner, modules}`.
- INV (idempotence): a request for an owner with no parks is a no-op (still deletes the row; no event or a
  benign no-op event).

## 5a. Boundary-test sketch (H) — scenarios facing both sides

| # | Precondition | Action | Postcondition (producer side) | Postcondition (consumer side) |
|---|---|---|---|---|
| B1 | Module M has stack `[low(shadowed), high(top)]` | snapshot | `park_stacks[M]` lists **both** owners, `shadowed` flags correct | Scheduler tab renders both; buried owner visible (α→β→γ) |
| B2 | Park owned by a **done/absent** task T on module M | snapshot + dashboard join | `park_stacks[M]` top = T | row appears and is **flagged stranded** (γ) — not silently dropped |
| B3 | Stranded park (owner T dead) | enqueue evict(T); tick | `force_clear(T)` fires, `reservation_force_evicted` emitted, M restored to any live shadowed owner | next snapshot: alert clears; M no longer stranded (δ + ε + ζ round-trip) |
| **B4** | **Park owned by a LIVE dispatchable task T** | enqueue evict(T); tick | **REFUSED**: no force_clear, `reservation_force_evict_refused` emitted, park intact | UI never offers the button (guard); if forced, no destructive effect (δ — the safety boundary) |
| B5 | evict(T) where T holds no parks | tick | row deleted, no-op | no error surfaced |
| B6 | Module M parked by T, **not held** | render Task-detail card / Orchestrators chip | `modules[M].parked_by == T` | chip is **orange** `lock-parked` naming T-T, **not grey** (η) |
| B7 | Module M **held** by H **and** parked by T | render chip | both fields set | chip renders **red** held-by-H (precedence), not orange (η) |

B4 is the load-bearing safety test — it proves the evict lever cannot re-create the df-1865 starvation.
B6/B7 prove the shared-chip orange state replaces only the grey/available case, never the red claimed case.

## 6. Out of scope (v1)

- Bubbling the stranded-parks alert onto the Overview tab (Scheduler-tab-only for v1).
- Auto-eviction / auto-healing of stranded parks (operator-initiated only; the *cause* is already fixed by 1865).
- Park history / duration histograms / time-series of stack depth.
- Any change to the `overrides` table or the pin/boost/reserve/ttl path.
- Exposing `park_stacks` through `get_scheduler_state`'s *typed* consumers beyond the dashboard.

## 7. Decomposition plan

Pre-split by package (the "split multi-package tasks before the architect pass" lesson — orchestrator +
fused-memory + dashboard each exceed the architect budget combined). The DAG also serializes the two same-file
edit collisions: `scheduler.py` (α→δ) and `tab_scheduler.jsx` (γ→ζ).

- **α (orchestrator, read producer)** — `ModuleLockTable.snapshot_park_stacks()` returning the §5 full-LIFO
  shape; add `park_stacks` key to `get_state_snapshot()`. Unit tests assert buried owners appear with correct
  `shadowed`/`rank`. *Intermediate; consumers β, δ.* **Deps:** none.
- **β (fused-memory, read seam)** — add `park_stacks: {}` to `_empty_skeleton()` and confirm `get_scheduler_state`
  passes the key through; test pass-through + the file-absent skeleton. *Intermediate; consumer γ.* **Deps:** α.
- **γ (dashboard, read + UI) — LEAF** — in `data/scheduler.py`: (a) add an **orphan/stranded pass** —
  cross-reference every `park_stacks` owner against the live-task set (owner not live ⇒ stranded), so stranded
  owners get a row instead of being dropped at the `active_tasks` join; (b) extend `_module_contention_counts` to
  attach the §5 per-module **`parked_by` / `parked_by_project` / `parked_owner_live`** fields (resolved by the
  same `modules_conflict` rule as `holder`) — the data η consumes. Render a "Module Park Stacks" section (full
  top→bottom stack per contended module: owner, rank/tier, age from `installed_at`, live/dead) + a stranded-parks
  alert in `tab_scheduler.jsx`. Dashboard tests (test_scheduler_page.py fixture pattern) cover B1, B2 and assert
  `modules[].parked_by` is populated.
  **Signal:** with a snapshot containing a shadowed park and an orphaned park, the Scheduler tab visibly shows
  the buried owner in the module stack and flags the stranded module. **Deps:** β.
- **η (dashboard, shared lock-chip parked state) — LEAF** — extend the *general* module-lock display used in the
  Tasks Task-detail card (`tab_tasks.jsx:334-342`) and the Orchestrators view (`tabs.jsx` `LockChip`) to render
  a third state **orange `lock-parked`** naming the park owner (`m.parked_by`) when a module is parked but not
  held, with precedence held(red) > parked(orange) > available(grey) (D6 / §5 read seam). Add the `lock-parked`
  CSS class (orange). Optional (tactical): a small stranded marker when `parked_owner_live` is false. Component
  tests assert B6 (orange-not-grey for parked-unheld) and B7 (red precedence when also held).
  **Signal:** in the Tasks Task-detail card and the Orchestrators view, a parked-but-unheld module renders orange
  with the park-owner task id instead of grey. **Deps:** γ (consumes `modules[].parked_by`; no shared file —
  γ owns `data/scheduler.py`+`tab_scheduler.jsx`, η owns `tabs.jsx`+`tab_tasks.jsx`).
- **δ (orchestrator, evict primitive + drain)** — `ModuleLockTable.force_clear(owner)` on `prune_owners`, emitting
  `reservation_force_evicted`; new `park_eviction_requests` store/table (orchestrator owns the schema); scheduler
  tick drains+deletes each request and applies the **D4 live-owner guard** (refuse + `reservation_force_evict_refused`).
  Tests cover B3 (evict a dead owner restores a live shadow) and **B4 (refuse a live owner — safety)**.
  *Intermediate; consumers ε, ζ.* **Deps:** α (shares `scheduler.py`/`ModuleLockTable`; serialized to avoid
  same-file collision).
- **ε (fused-memory, evict tool)** — MCP tool `request_park_eviction(task_id, project_root)` inserting a row into
  `park_eviction_requests` (raw-SQL, matching δ's schema contract; the table is not shared via import).
  Test: tool inserts a well-formed row. *Intermediate; consumer ζ.* **Deps:** δ.
- **ζ (dashboard, evict action) — LEAF** — POST `/api/v2/dashboard/scheduler/evict-park` proxying to
  `request_park_eviction`; a guarded "evict" button on each stranded-park row in `tab_scheduler.jsx`, enabled only
  when the owner is verified non-live. Tests cover the endpoint + the button-enablement guard, and B3's
  round-trip at the dashboard boundary.
  **Signal:** clicking "evict" on a stranded row enqueues the request; the ghost park clears and the
  stranded alert disappears on the next snapshot. **Deps:** γ (shares `tab_scheduler.jsx`; serialized), ε.

**Dependency edges:** β→α · γ→β · η→γ · δ→α · ε→δ · ζ→γ · ζ→ε.

## 8. Cross-PRD relationship

No cross-PRD seams — this is a self-contained dark_factory observability+remediation feature building only on
landed substrate (1230 get_scheduler_state, 1231 Scheduler page, 1865 park stack). All seams are intra-system and
owned within this batch. G4 N/A.

## 9. Open questions (tactical)

- Exact React layout of the Module Park Stacks section (table vs. nested chips) — architect's call; γ.
- Whether the stranded alert shows a per-module count or a per-owner count — cosmetic; γ.
- Audit-event payload shape for `reservation_force_evicted` / `_refused` (reuse the existing reservation-event
  envelope) — δ.
