# PRD — fm-task-dedup (durable task uniqueness + single write-authority seam)

**Stream:** W8 (Wave 1) of the Bug-Hotspot Remediation Program 2026-07-06
(`plans/bug-hotspot-remediation-program-2026-07-06.md`).
**Status:** deferred — decompose-and-queue in the same session.
**Approach:** B + H (contract + two-way boundary tests). High-stakes: the task
tracker is the core durable store the whole factory dispatches from.
**Date:** 2026-07-06.
**Findings addressed:** fm-task-layer cluster (findings 1, 3, 5) in
`plans/bug-hotspot-survey-2026-07-06-full-findings.json`.

---

## 1. Goal (user-observable behaviour)

Three durable-integrity outcomes an operator can observe:

1. **Duplicate tasks become impossible at the store, not merely improbable in
   memory.** After deploy, submitting two tasks with the same normalized
   title + files — through *any* path (curator batch, `planning_mode`
   decomposition, recon Stage-2 submit) — creates exactly **one** row; the
   second resolves `combined` pointing at the first's id. The six stacked
   in-memory dedup layers (each keyed differently, all lost on restart) become
   optimizations, not the last line of defence. The property survives a
   fused-memory restart — the in-memory caches do not.

2. **`done_provenance` / status write-authority is enforced at one privileged
   seam.** `update_task(metadata.done_provenance=…)` and `update_task(status=…)`
   are rejected by a single backend floor emitting one canonical error shape —
   not by three drifting guard copies that already disagree on error shape. The
   sanctioned writer (`set_task_status(done, …)`) still stamps provenance,
   through a privileged internal API the public surface cannot reach.

3. **Cross-project mis-filing is rejected on structured evidence, not prose
   regex.** A task whose `metadata.files` point at another project's tree is
   rejected; a task that merely *mentions* another project's directory name in
   prose is created (with a `possible_scope_mismatch` advisory + escalation),
   not falsely rejected — ending the false-positive tuning treadmill that cost
   a wasted L2 round-trip per over-fire.

## 2. Background

The fm-task-layer is a four-deck stack (`server/tools.py` → `task_interceptor.py`
→ `task_curator.py` → `sqlite_task_backend.py`). Three architectural seams have
each accreted a compensation stack instead of a root fix:

- **Curator dedup fails open to CREATE on every failure branch**
  (`_process_add_tickets_batch_prepared`: idempotency-check failure → create,
  curator exception → create, batch-target out of range → create, sibling
  failed → create, dependency cycle → create). Because CREATE is the failure
  default, every new failure mode manifests as *duplicate tasks* — hence six
  in-memory dedup layers with three different keys, none durable. The tracker
  itself holds duplicate-ID pairs (999/1000, 1001/1002, 1026/1028) and task
  1042 exists to scan for residuals. (Finding 1, confirmed.)
- **`update_task` write-authority is re-implemented at every deck**: `tools.py`
  `_reject_status_in_update_task` / `_reject_done_provenance_in_metadata`,
  `task_interceptor.py` module-level copies "defence-in-depth", and a backend
  status floor. The copies already **disagree on error shape**
  (`{'error':…,'error_type':'ValidationError'}` vs
  `{'success':False,'error':'done_provenance_via_update_task'}`), forcing
  `interceptor_write_succeeded` to enumerate every known rejection dict. Root
  cause: the sanctioned writer persists provenance through the *same public
  `tm.update_task`* it must guard, so "who may write" can only be enforced by
  re-checking at every entry point. (Finding 3, confirmed. History:
  8daed1734a "9 unauthorized done writes in 36h".)
- **Path routing scans free prose for path prefixes** (`path_scope_guard.py`
  regex, hand-tuned boundary class) where *mention ≠ ownership* — precision/
  recall is structurally unwinnable at the regex layer, compensated by a
  four-stage patch stack (lookbehind tuning 1095/1096/1100/1111/1118, a 454-line
  LLM adjudicator 1822, a `routing_override_reason` escape hatch 1845) plus a
  `dark_factory_path_guard` shim declared "for one merge cycle" yet still the
  import in `task_interceptor.py`. (Finding 5, confirmed.)

Prior work superseded (all `done`/`cancelled`) — these are the *layering* fixes
this PRD replaces with roots, not open work: 833, 981, 1004, 1042, 1140, 1664,
1088, 1095, 1100, 1494, 1822, 1845.

## 3. Sketch of approach

Four workstreams; the durable `candidate_key` is the correctness root, the rest
are single-seam consolidations.

### A. Durable `candidate_key` (root fix — finding 1 part 1)

- One pure normalization leaf `compute_candidate_key(title, files) -> str`
  co-located in a **low-dependency module** both `sqlite_task_backend` and
  `task_curator` import (no import cycle; see contract §C-A). The key definition
  is exactly the existing `TaskCurator._normalize_key`:
  `sha256_16(normalized_title | '\n'.join(sorted(files)))`.
- Computed **at the single store-level INSERT chokepoint**
  (`SqliteTaskBackend.add_task`) so **every** insert path populates it — curator
  batch, `planning_mode` (which bypasses the curator entirely), recon direct,
  and the `targeted.py` direct-backend fallback. This is the load-bearing
  reason the computation lives in the backend, not the curator: a key computed
  only in the curator would leave `planning_mode` batches free to reintroduce
  duplicates (the brief's explicit verify item — confirmed:
  `_submit_task_planning_mode` calls `tm.add_task` directly).
- A **partial UNIQUE index** `(tag, candidate_key)` over
  `WHERE candidate_key IS NOT NULL AND status != 'cancelled'` makes duplicate
  creation impossible at the DB. On the `IntegrityError` the backend raises a
  typed `DuplicateCandidateKeyError(existing_id)`; the interceptor create-
  dispatch and the `planning_mode` path both resolve the ticket **`combined`**
  with `existing_id` — no orphan row.
- **Two-stage migration** (the migration caution): stage-1 adds the nullable
  column + backfills existing non-cancelled rows + emits a *report-only*
  violation-groups audit (no auto-delete); stage-2 creates the UNIQUE index and
  is **self-gating** — it queries for residual non-cancelled duplicate groups
  and **refuses loudly** (deploy aborts, residuals named) if any remain, else
  builds the index. NULL keys and cancelled rows are excluded by construction.

### B. Per-ticket lifecycle struct (clarity — finding 1 part 2)

Replace the correlated parallel arrays (`decisions[i]`,
`curator_degrade_reasons[i]`, `resolved_task_ids[i]`, `non_none_to_ticket_data`)
and the two index-spaces ("non_none-space" vs "ticket_data-space" remapping) in
`_process_add_tickets_batch_prepared` with **one dataclass per ticket** carrying
its candidate, prepared bundle, decision, degrade reason, and resolved terminal
(status + task_id). The index-space remapping — the structure that produced the
pass-1/pass-2 blank-title mismatch — dies.

### C. Single `update_task` write-authority seam (finding 3)

- `SqliteTaskBackend.update_task` **unconditionally rejects
  `metadata.done_provenance`** (extending the existing status floor), raising a
  typed error whose canonical rejection dict is defined **once** in
  `task_backend_errors.py`.
- A **privileged non-protocol** method `stamp_audit_metadata(task_id,
  project_root, fields, tag)` on the backend (NOT in `TaskBackendProtocol`) that
  only `TaskInterceptor` holds a reference to, performing the read-modify-write
  of audit fields (`done_provenance`, `reopen_*`).
- `_apply_status_transition` persists reopen/provenance via
  `stamp_audit_metadata`, not public `tm.update_task`. **Delete** the `tools.py`
  and interceptor `_reject_*` copies (tools keeps only the ticket-id shape
  check); `interceptor_write_succeeded` recognizes the **one** canonical shape.

### D. Structured path routing (finding 5)

- `ProjectPrefixRegistry.project_for_path(file_path)` — exact leading-path-
  component match of a concrete file path against registered prefixes (no regex,
  no prose).
- Split the decision by signal quality: **REJECT only** when `metadata.files`
  contains a path owned by another project (structured, certain). **Demote prose
  scanning to advisory**: on a prose-only hit, create the task, attach
  `metadata.possible_scope_mismatch`, and fire the scope-violation escalator
  (loud, non-blocking) — the existing LLM adjudicator/curator triage the
  ambiguous middle asynchronously. No hard reject on prose.
- Delete the `dark_factory_path_guard` shim: switch interceptor imports to
  `path_scope_guard`, fold the hard-coded `DARK_FACTORY_PATH_PREFIXES` into the
  registry default.

### Deploy

Deterministic capstone restarts fused-memory (out-of-cgroup
`scripts/restart-fused-memory.sh`, **no `--drain`** — resolved decision #6) so
the new schema migration (runs at connection-open on `user_version`) + the
guards go live. fused-memory is the only process that needs restart — the
orchestrator consumes none of these mechanisms cross-process.

## 4. Resolved design decisions (do not relitigate)

1. **`candidate_key` computed in the backend, not the curator.** Durability
   requires every INSERT path to populate it; `planning_mode` and recon-direct
   bypass the curator, so a curator-only key reopens the duplicate window. The
   normalization *definition* is owned by `task_curator` (it is the existing
   `_normalize_key`) but physically lives in a low-dep leaf module the backend
   imports without a cycle. The curator imports the same function so its
   in-memory layers agree on the one key.
2. **The UNIQUE index is partial over `candidate_key IS NOT NULL AND status !=
   'cancelled'`.** Cancelled rows never collide (a task can be cancelled and its
   work re-filed); NULL keys (legacy rows that could not compute one during
   backfill) never collide (SQLite treats NULLs as distinct). Since the backend
   computes a key on every new insert, NULL only ever applies to un-backfillable
   legacy rows.
3. **Constraint violation resolves `combined`, never `failed`.** A duplicate is
   a *successful dedup*, not an error. The backend raises
   `DuplicateCandidateKeyError(existing_id)`; both the interceptor create-branch
   and `planning_mode` return the existing id with a `combined` disposition.
4. **Migration is two-stage, report-then-constrain, and the index build is
   fail-safe — never service-fatal.** Stage-1 (column + backfill + report) is
   idempotent and non-destructive — it **never auto-deletes** duplicates, only
   reports groups. Stage-2 (index) is **self-gating and skip-and-escalate**: at
   connection-open it queries for residual non-cancelled duplicate groups; if any
   remain it **logs a loud ERROR, fires a scope-level escalation naming the
   residual groups, and SKIPS the index build** — it does **not** raise, because
   a migration that raises at connection-open would crash-loop fused-memory (a
   hard outage). The service stays up; the six in-memory dedup layers (which this
   PRD demotes to optimizations but does **not** delete) keep catching duplicates
   best-effort until an operator cleans the residuals (1042-style) and the next
   deploy lands the index. When the audit is clean (the expected case — 1042 has
   already run), the index builds and enforcement is live. This is faithful to
   the migration caution ("the UNIQUE index lands only after the backfill audit
   is clean") while staying autonomous and fail-safe.
5. **Write-authority rejection preserves the canonical dict shape while
   collapsing to one definition.** The backend raises a typed
   `TaskmasterError` subclass whose `.to_error_dict()` returns the existing
   canonical `{'success': False, 'error': 'status_via_update_task' /
   'done_provenance_via_update_task', …}` shape, defined **once** in
   `task_backend_errors.py`. tools.py's `update_task` handler and
   `interceptor_write_succeeded` recognize that one shape. This keeps the client
   contract (callers branching on `error == 'done_provenance_via_update_task'`
   keep working) while deleting the drifting copies — a bare
   `{'error': str(e)}` would silently break those callers.
6. **Prose demotion trades a false-reject for a loud advisory, never silent
   degradation.** A task that genuinely belongs elsewhere but carries *no*
   `metadata.files` and only prose evidence is now created-in-place **with a
   fired scope-violation escalation** — the escalation is preserved (loud), so
   the adjudicator/human still catches the genuine misroute. We accept this over
   the current hard-reject, which costs a wasted L2 round-trip on every
   incidental-mention false positive (esc-task-path-guard-10 class). Consistent
   with the "loud escalation over silent degradation" directive.
7. **Fused-memory restart via `scripts/restart-fused-memory.sh` (no args).**
   Resolved decision #6 of the program doc: out-of-cgroup `systemctl --user
   restart`; the `--drain` path (hung, task 2090) is opt-in and not used. The
   capstone is cross-unit (orchestrator → fused-memory), so the blocking
   deterministic path with fresh-`MainPID` verify applies.

## 5. Pre-conditions for activating

- No upstream stream deps (W8 is Wave 1, upstream deps `—`). W5 coordination is
  read-only at authoring time (see Cross-PRD §7).
- SQLite ≥ 3.8.0 for partial indexes — verified 3.45.1.
- `scripts/restart-fused-memory.sh` exists + executable — verified (deploy
  capstone `before_done` validation).

## 6. Contract section (H)

### C-A — `candidate_key` seam

```
# new leaf module: fused_memory/middleware/candidate_key.py
#   (low-dep: stdlib hashlib only; NO import of task_curator/backend/qdrant)
def compute_candidate_key(title: str | None, files: Iterable[str] | None) -> str | None:
    """sha256_16( normalize_title(title) | '\n'.join(sorted(files or [])) ).
    Returns None iff title is empty/None (uncomputable → row excluded from the
    partial index). Identical algorithm to TaskCurator._normalize_key; that
    method is refactored to delegate here so there is ONE definition."""
```
Invariants:
- Deterministic + order-insensitive on files; case/whitespace-insensitive on
  title (delegates `normalize_title`).
- `SqliteTaskBackend.add_task` computes it from `title` + `metadata.files`
  (Lock-charter Contract-1: file-level) and stores it in the new column on
  **every** insert, inside the existing write-lock/txn.
- Partial UNIQUE index: `CREATE UNIQUE INDEX ux_tasks_candidate_key ON
  tasks(tag, candidate_key) WHERE candidate_key IS NOT NULL AND status !=
  'cancelled'`.
- On `IntegrityError` against that index, `add_task` SELECTs the surviving
  non-cancelled row with that `(tag, candidate_key)` and raises
  `DuplicateCandidateKeyError(existing_id=<id>)`. Caller resolves `combined`.

Error semantics: `DuplicateCandidateKeyError` is a typed `TaskmasterError`
subclass; interceptor create-dispatch and `planning_mode` catch it explicitly
(never the generic `except Exception` → failed branch).

### C-C — write-authority seam

```
# fused_memory/backends/task_backend_errors.py  (single definition)
def status_via_update_task_error(task_id, status) -> dict: ...          # canonical shape
def done_provenance_via_update_task_error(task_id) -> dict: ...         # canonical shape
class StatusWriteAuthorityError(TaskmasterError): ...  # .to_error_dict() -> canonical
class DoneProvenanceWriteAuthorityError(TaskmasterError): ...

# SqliteTaskBackend
async def update_task(...): # raises the two above for status / metadata.done_provenance
async def stamp_audit_metadata(self, task_id, project_root, fields: dict, tag=None):
    """Privileged read-modify-write of audit fields (done_provenance, reopen_*).
    NOT declared in TaskBackendProtocol; only TaskInterceptor holds a ref.
    Merges `fields` into the row's metadata under the write-lock, preserving
    memory_hints/files/external_deps."""
```
Invariants:
- `update_task` is the ONLY public metadata writer and rejects both status and
  `metadata.done_provenance` unconditionally, before the row SELECT.
- `stamp_audit_metadata` is the ONLY writer of `done_provenance` / `reopen_*`
  and is reachable only from `_apply_status_transition`.
- tools.py surfaces the raised typed errors as their canonical `.to_error_dict()`
  (its `update_task` handler already wraps backend exceptions).
- `interceptor_write_succeeded` recognizes exactly the one canonical shape.

### C-D — routing seam

```
# ProjectPrefixRegistry
def project_for_path(self, file_path: str) -> str | None:
    """Owner of the longest registered prefix that is a leading path component
    of file_path (exact, '/'-boundary; NOT regex-over-prose). None if unowned."""
```
Invariants:
- Files decision: any `metadata.files` entry whose `project_for_path(...)` is a
  known project ≠ submitting project → **reject** (structured).
- Prose decision: a `find_paths` hit in title/description/details with no
  files-level mismatch → **advisory**: create + `metadata.possible_scope_mismatch`
  + fire escalator. Never reject.
- `routing_override_reason` (task 1845) still short-circuits both.

## 6b. Boundary-test sketch (H) — faces both sides of each seam

| # | Scenario | Preconditions | Postconditions (both sides) |
|---|---|---|---|
| BT-A1 | Two tasks, identical title+files, via **curator** submit→resolve | index live | producer(backend): 2nd INSERT hits index → `DuplicateCandidateKeyError`; consumer(interceptor): 2nd ticket resolves `combined`→id of 1st; `get_tasks` shows exactly one non-cancelled row |
| BT-A2 | Two `planning_mode` adds, identical key | index live | producer(backend): raises on 2nd; consumer(planning_mode): returns 1st id, `combined`; no orphan row — **the brief's planning-mode reintroduction guard** |
| BT-A3 | Crash injected between INSERT and COMMIT in `add_task` | fault hook | producer: txn rolls back, zero orphan rows (`get_tasks` count unchanged); index invariant intact on reconnect |
| BT-A4 | Duplicate submitted, process restarted, duplicate re-submitted | index live | property holds across restart (in-memory caches are cold) — combine still fires |
| BT-A5 | Cancel task X, re-file same title+files | X cancelled | new row created (partial index excludes cancelled) — no false combine |
| BT-C1 | `update_task(metadata.done_provenance=…)` via tools.py **and** via interceptor | — | both surfaces return the **byte-identical** canonical rejection dict |
| BT-C2 | `set_task_status('done', done_provenance=…)` | valid provenance | `stamp_audit_metadata` persists it; `get_task` shows `done_provenance` — sanctioned writer still works end-to-end |
| BT-C3 | `update_task(status='done')` | — | rejected by backend floor via canonical `status_via_update_task` shape |
| BT-D1 | Task: prose mentions `orchestrator/…`, `metadata.files` all in submitting project | multi-project registry | **created** with `possible_scope_mismatch` + escalation fired — NOT rejected |
| BT-D2 | Task: `metadata.files` = `orchestrator/foo.py`, filed under reify | registry | **rejected** with structured error naming the owning project |
| BT-D3 | `grep dark_factory_path_guard fused-memory/src` after D2 | — | empty (shim deleted); dark-factory mis-file still rejected via registry default |

## 7. Cross-PRD relationship (G4)

| Other stream | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| W5 recon-reliability | none | `execution_class` is recon **execution** routing (deterministic/normal); W8's path-guard is **project-ownership** routing (`metadata.files` → `ProjectPrefixRegistry`). Different axis. | — (no seam) | n/a — W5 PRD not committed; no field to consume |
| W3 task-metadata-schema | independent | W8's `candidate_key` is a **column**, not metadata (brief out-of-scope). W8's write-authority seam enforces *who writes* `done_provenance`; W3 types *what shape* it is. Compose later; no dep either way. | W8 owns write-authority; W3 owns schema | independent |
| W2 task-status-authority | orthogonal | W8's backend floor rejects `status` in `update_task` (a *write-authority* floor); W2's transition table governs *which* transitions are legal (a *legality* gate). Distinct mechanisms, no overlap. | W2 owns transition table | orthogonal |

Per the program seam map, **W8 owns `candidate_key` uniqueness + the update_task
write-authority seam**; consumers `task_curator`/`interceptor` do not redefine
them.

## 8. Decomposition plan

Labels are PRD-local; task IDs assigned at decompose. `[modules]` are file-level.

**A — durable candidate_key (correctness root)**

- **A1** — `candidate_key` column + compute-on-every-insert + backfill + report.
  New leaf `compute_candidate_key`; `add_task` computes+stores; schema-v2
  migration adds nullable column, backfills non-cancelled rows, emits a
  report-only violation-groups audit; `_row_to_task` exposes `candidate_key`;
  `TaskCurator._normalize_key` delegates to the leaf.
  *Signal (leaf/observable):* after deploy, `get_task` returns `candidate_key`
  for a row; the migration emits an audit log line naming the count of duplicate
  candidate_key groups found (**count may be 0** — no numeric premise asserted).
  *Consumer:* A2. *[candidate_key.py, sqlite_task_backend.py, task_curator.py]*
- **A2** — self-gating partial UNIQUE index + collision→`combined` wiring.
  Schema-v3 migration: detect residual non-cancelled duplicate groups → **log
  ERROR + fire escalation + SKIP the index** (fail-safe, never raise at
  connection-open, decision #4); else build the partial UNIQUE index. `add_task`
  maps `IntegrityError` → `DuplicateCandidateKeyError(existing_id)`; interceptor
  create-dispatch and `planning_mode` catch it → resolve `combined`.
  *Signal:* submitting a second task with an identical normalized title+files
  (via both curator submit→resolve and `planning_mode`) returns a `combined`
  disposition pointing at the first id; `get_tasks` shows exactly one
  non-cancelled row. *Prereq:* A1. *[sqlite_task_backend.py, task_interceptor.py]*
- **A3** — B+H boundary-test integration gate for the candidate_key seam.
  Implements BT-A1…BT-A5 driving the **real** submit→resolve / planning_mode /
  crash-injection / restart paths (not isolated synthetic-input units).
  *Signal:* the candidate_key boundary suite passes end-to-end, exercising the
  backend-constraint side and the interceptor/planning_mode combine side,
  including the crash-injection no-orphan-row assertion and the cross-restart
  durability assertion. *Prereq:* A2.
  *[fused-memory/tests/test_candidate_key_boundary.py (+ harness helpers)]*

**B — per-ticket lifecycle struct (clarity)**

- **B1** — replace the correlated parallel arrays + two index-spaces in
  `_process_add_tickets_batch_prepared` with one per-ticket dataclass.
  *Signal:* submit a mixed two-phase batch (some tickets combine, some create,
  one degrades) via the real submit→resolve path; each ticket's `resolve_ticket`
  returns its correct terminal (`created`/`combined`/`failed`) with the correct
  `task_id` — no cross-ticket id substitution (the pass-1/pass-2 blank-title
  mismatch class can no longer occur). *Prereq:* A2 (serializes the shared
  method edits; builds atop the collision-aware dispatch). *[task_interceptor.py]*

**C — single update_task write-authority seam**

- **C1** — backend privileged seam. `update_task` floor rejects
  `metadata.done_provenance` (+ existing status); `stamp_audit_metadata`
  non-protocol method; canonical rejection shapes in `task_backend_errors.py`.
  *Signal:* `update_task(metadata={'done_provenance':…})` through the MCP
  surface returns the single canonical `done_provenance_via_update_task`
  rejection (sourced from `task_backend_errors.py`); a direct
  `stamp_audit_metadata` call persists provenance visible via `get_task`.
  *Consumer:* C2. *[sqlite_task_backend.py, task_backend_errors.py, task_backend_protocol.py]*
- **C2** — rewire sanctioned writer + delete guard copies (C integration-gate).
  `_apply_status_transition` uses `stamp_audit_metadata`; delete tools.py +
  interceptor `_reject_status`/`_reject_done_provenance` copies; collapse
  `interceptor_write_succeeded` to the one canonical shape (BT-C1…BT-C3).
  *Signal:* the done_provenance rejection dict is **byte-identical** whether the
  call hits tools.py or the interceptor path; `set_task_status('done',
  done_provenance=…)` still stamps provenance end-to-end (`get_task` shows it);
  `grep '_reject_done_provenance' fused-memory/src` shows one definition.
  *Prereq:* C1. *[server/tools.py, task_interceptor.py]*

**D — structured path routing**

- **D1** — `ProjectPrefixRegistry.project_for_path` + reject-on-files /
  advise-on-prose in the interceptor path-guard decision.
  *Signal:* a task whose prose mentions another project's dir but whose
  `metadata.files` are all in-project is **created** with
  `metadata.possible_scope_mismatch` + a fired escalation; a task whose
  `metadata.files` point at another project's tree is **rejected** with the
  structured error naming the owner. *Consumer:* D2 + submit_task user surface.
  *[project_prefix_registry.py, path_scope_guard.py, task_interceptor.py]*
- **D2** — delete the `dark_factory_path_guard` shim. Switch interceptor imports
  to `path_scope_guard`; fold `DARK_FACTORY_PATH_PREFIXES` into the registry
  default; remove the shim module.
  *Signal:* `grep dark_factory_path_guard fused-memory/src` returns nothing; a
  dark-factory-files task mis-filed into another project is still rejected via
  the registry default. *Prereq:* D1.
  *[task_interceptor.py, path_scope_guard.py, project_prefix_registry.py, dark_factory_path_guard.py (deleted)]*

**Z — deploy capstone (deterministic)**

- **Z** — restart fused-memory so the schema migrations + guards go live.
  `task_kind='deterministic'`, `before_done.script='scripts/restart-fused-memory.sh'`
  (no `--drain`), `always_escalates=false` (auto-deploy preset: escalate only on
  failure), cross-unit (`target_unit=None`) → blocking + fresh-`MainPID` verify.
  *Signal:* after the capstone, `get_status` shows a fresh `uptime_seconds`
  (restart occurred). Then **one of two honest outcomes**: (a) clean audit — a
  duplicate title+files create on the **live** service resolves `combined`
  (candidate_key enforcement is live); or (b) residual dups — the connection-open
  migration logged the residual groups and fired the escalation, index skipped,
  service healthy (per decision #4). Either outcome is a successful, fail-safe
  deploy. *Prereqs:* A3, B1, C2, D2.

**Dependency DAG:** A1→A2→A3; A2→B1; C1→C2; D1→D2; {A3,B1,C2,D2}→Z.
Clusters A, B, C, D run in parallel up to the module lock; only Z barriers them.

## 9. Out of scope

- **Finding 2** (scheduler_overrides.db / park-eviction dual-implementation
  across two processes). The brief marks it optional/weakened; extracting a
  shared store module is an M-stream-shaped change orthogonal to task-dedup and
  would widen W8's blast radius. **Deferred** — file separately if the mirror
  comments rot into a live drift.
- **Finding 0 / W3** (versioned `TaskMetadata` schema). `candidate_key` is a
  **column**, deliberately not metadata.
- **Finding 4 / W2** (status vocabulary StrEnum + store-level CHECK).
- **Finding 6** (deterministic-task invariants — `task_kind`/`before_done`/
  `always_escalates` — enforced only at `submit_task`, mutable via `update_task`).
  W8's `stamp_audit_metadata` + floor covers `done_provenance`/status/reopen, not
  the deterministic-kind fields; the finding itself notes these "collapse into
  the shared TaskMetadata schema (W3) if that lands first". Deferred to W3.
- Recon write policy (W5).

## 10. Open questions (tactical — not design-blocking)

1. **Column vs metadata for `possible_scope_mismatch`.** D1 attaches the advisory
   to `metadata.possible_scope_mismatch`. If a future consumer needs to query it
   at the store, revisit as a column. *Suggested:* metadata for now (advisory,
   not dispatch-gating). Decide during D1.
2. **tools.py canonical-shape passthrough.** Verify at C1/C2 impl that tools.py's
   `update_task` handler returns the typed error's `.to_error_dict()` verbatim
   (not `{'error': str(e)}`) so the canonical shape reaches the client
   unchanged. *Suggested:* have the handler special-case the two typed errors, or
   make the generic `except` prefer `err.to_error_dict()` when present. Decide
   during C2.
3. **Backfill batching for ~2100 rows.** A1's backfill computes a key per non-
   cancelled row in one migration txn. *Suggested:* single txn is fine at 2100
   rows; chunk only if the migration exceeds the connection-open budget. Decide
   during A1.
4. **`AFK/autonomy default recorded:** Finding 2 scoping and the W5 no-seam
   determination were taken as safe defaults per the program doc without operator
   confirmation (operator AFK). Revisit if W5's committed PRD claims project-
   routing territory.
5. **`metadata.files` extraction in `add_task`.** A1 reads files from the
   metadata JSON already passed to `add_task`; confirm the field name matches the
   Lock-charter file-level `files` key (not `files_to_modify`) at the backend
   boundary. *Suggested:* accept both keys defensively, normalize to the file
   list. Decide during A1.

---

_Authored 2026-07-06 by claude-prd-fm-task-dedup. Capability manifest:
`plans/fm-task-dedup-prd.capability-manifest.md`._
