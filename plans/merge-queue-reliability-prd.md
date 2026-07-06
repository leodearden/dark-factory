# Merge-Queue Reliability — PRD

**Stream:** W1 (bug-hotspot remediation program 2026-07-06) · **Wave:** 1 · **Upstream deps:** none
**Status:** active · authored 2026-07-06 · **Approach:** B + H (high-stakes; §Contract + §Boundary-test sketch below)
**Program doc (authoritative G4 seam map):** `plans/bug-hotspot-remediation-program-2026-07-06.md`
**Findings:** `plans/bug-hotspot-survey-2026-07-06-full-findings.json` (cluster 0, merge-queue, all 6 findings)
**Brief:** `/home/leo/.claude/spawn-briefs/df-hotspot-2026-07-06/W1-merge-queue-reliability.md`

---

## 1. Goal

Make the merge-queue subsystem's core invariants **structural rather than audited-after-the-fact**, and close the crash window between a merge landing on `main` and the task being marked `done`. Concretely, after this PRD lands:

1. A crash between "merge advances `main`" and "task marked `done`" **self-heals on the next orchestrator start** — a durable landed-outbox row drives the task to `done` with `merged` provenance, instead of the task re-dispatching against an already-merged branch (the ghost-loop). A public `MergeProvenance.lookup(task_id)` answers "did branch X land as SHA Y for task Z?" — the substrate W9 (workflow guard collapse) and W10 (harness sweep retirement) consume.
2. The speculation semaphore is **owned by permit tokens**: conservation is `slot_available + len(ledger.live) == depth` by construction, so a new `await` point in the pipeline can no longer create an uncounted window and a false-positive leak alarm (the I4 `merge_resource_leak` false-positive class — tasks 2063/2068/2096).
3. Item lifecycle is a **single registry with an explicit state enum + legal-transition table**; `snapshot()`, the permit audit, and the liveness ledger become single registry reads that cannot disagree.
4. Branch identity is a **parsed-once value type** (`QueuedBranch`), so mixed-shape (`"4778"` vs `"task/4778"`) is unrepresentable past the boundary and pyright enforces it at every current drift site.
5. The retired serial worker and the `_verify_and_advance` compat shim stop **freezing production structure** — tests drive the public surface; production docstrings stop citing the retired worker as a byte-identical spec.

**Observability at the user surface:** the operator sees the I4 `merge_resource_leak` false-positive escalations stop (permit conservation is structural); ghost-loop / phantom-done incidents that today require manual `/unblock` (operator memory: *"reify merge queue from a DF session"* — manual merge lands but task stays pending → re-dispatch hazard) self-heal at startup; `merge_queue.py` stops being the #1 churn file because satellite changes no longer route back through it.

---

## 2. Background

`merge_queue.py` is a 9,407-line god-module built around one class, `SpeculativeMergeWorker` (a merger coroutine that creates speculative merge commits, a verifier coroutine that dispatch-fills verify hosts and CAS-advances `main` in submission order). The survey (cluster 0) found six confirmed structural hazards; the brief scopes program priorities **1** (landed-outbox journal) and **7** (structural refactor) as one **linear PRD chain**, precedent: the 17-task merge-queue refactor batch **df 1985-2002** (a strictly linear chain to avoid rebase churn on the #1-churn shared file).

**Prior work this builds on (all `done`):** 1628/1639 (MergeRequest identity, multi-waiter entries), 1772 (`MergeQueueStore` durable accept-side journal), 1993 (`SpeculationController` — merger-side permit lifecycle only), 1991 (`_resolve_and_release` chokepoint), 1992 (request-liveness ledger), 1994 (`speculation_accounting_violations` / `worktree_ledger_violations` audits), 2063/2068/2096 (I4 census patches: `_redispatch`, snapshot parity, `_finalizing_head` term), 1895 (two-layer merge-queue B+H integration gate). This PRD **completes** the seams those tasks half-extracted — it does not redo them.

**Why the survey calls the invariants "audited-after-the-fact":** the recent refactor (df 1985-2002) *extracted text, not state or dependency edges*. Satellite modules (`merge_gates`, `merge_drift`, `merge_shadow`, `merge_liveness`) are function-bags that take the live worker and mutate its privates, and reach BACK through `orchestrator.merge_queue` at function level solely so the test suite's string-path monkeypatches keep resolving — a deliberate circular dependency (verified: `merge_gates.py:361/439/1353`, `merge_drift.py:254`, `merge_shadow.py:1012`, `merge_liveness.py:181/280/702`). That freeze is why the extraction stopped at re-export shims; unfreezing it (scope 2) is the prerequisite for scopes 3 and 4.

---

## 3. Substrate reality check (G3) — all verified 2026-07-06 against HEAD `365e63b9`

| Assumed capability | Status | Evidence (current main) |
|---|---|---|
| `MergeQueueStore` exists, durable | ✅ exists | `merge_queue_store.py` — atomic JSON (`tmp + os.replace`, `_save_raw` :182), keyed by `request_id`, `remove()` on terminal (:131). **NOT SQLite; no fsync.** → landed-outbox needs its **own** append-durable, **fsync'd** structure (see §Contract; a landed row must not live in the accept-side dict that is `remove()`d on terminal). |
| CAS `main` advance site(s) | ✅ 2 sites | `_finalize_inflight` `merge_queue.py:8783-8794` (`advance_main`, single-branch); train path `merge_queue.py:2961-2967`. At the advance, available: `req.task_id`, `item.merged_branch_tip` (branch tip SHA), `item.base_sha` (expected prior main), `current_sha`/`merge_commit` (SHA about to become main), post-advance `adv_outcome.advanced_sha`. |
| A best-effort *post*-advance landed hook already exists | ✅ exists (loses on crash) | `_on_merge_landed(req.task_id, item.base_sha, outcome.merge_sha)` `merge_queue.py:8831-8833` → `harness._note_merge_all`. Fires **after** the advance, fail-open — the exact lost-callback the WAL supersedes. |
| `done_provenance` merged-write path | ✅ exists | `Scheduler.mark_done(kind='merged', sha=...)` `scheduler.py:1704-1729` → `set_task_status('done', done_provenance={'kind':'merged','commit':sha})`. Server validator `_validate_done_provenance` `task_interceptor.py:3556` accepts `merged` with `commit`, backstopped by `git merge-base --is-ancestor <sha> main` (:3582). Reconciler's done-write is **producible and will pass the gate** (advanced_sha IS on main). |
| Scheduler dispatch decision point | ✅ exists | `Scheduler.acquire_next` `scheduler.py:3305`; eligibility predicate `_eligible_for_dispatch` :3890; commit points `try_acquire` :3899/:3969. Rejection family `SetTaskStatusRejected`/`TerminalExitRejection`/`DoneGateRejection`/`ProvenanceValidationRejection` :93/:112/:138/:160. |
| `find_merge_marker` archaeology (W10 retires) | ✅ exists | `git_ops.py:4004`; harness reconcile fast-path `harness.py:3229`. **W1 does not touch it** — W10 owns its retirement. |
| Speculation permit substrate | ✅ exists | `_speculation_slot` `merge_queue.py:3942`; census `_inflight_speculative_count` :5006-5084 (5 locations); `SpeculationController` (merger-side only, docstring :51-55); transient fields :3985/:3996/:4003/:4015; `SuffixConflictTracker` lambda-injection precedent :4116-4121. |
| `merge_types.py` home for typed values | ✅ exists | `MergeRequest` :609 (`branch: str` :614 — bare), `InflightEntry` :909 (`phase: str` :954), `InflightStatus` :890. `QueuedBranch` will be added here. |
| `OutcomeKind` merge-attempt enum | ⚠️ **does NOT exist** | Repo-wide grep: zero code hits; M3 **introduces** it in `merge_types.py`. **W1 must not create a competing outcome enum** (§Out of scope). |

**No novel substrate is assumed that does not exist** (the one gap — `OutcomeKind` — is out of scope, owned by M3). G3 passes.

---

## 4. Sketch of approach

Six scopes, one **strictly linear** dependency spine (`α → β → … → ο`). Every scope touches `merge_queue.py` (the #1 churn file), so a linear chain — matching the df 1985-2002 precedent — is the safe default: it prevents constant rebase conflict on the shared god-file. Scope order matches the brief: **1 → 2 → 3 → 4 → (5, 6)**, with (5,6) linearized (see §Open questions Q1).

- **Scope 1 — Landed-outbox write-ahead journal (α–δ).** A new durable, fsync'd `LandedOutbox` (separate from the accept-side journal) written **write-ahead** — the `(task_id, branch_tip_sha, advanced_sha, wall_time)` row is fsynced **before** the CAS advance at both advance sites. A public `MergeProvenance.lookup(task_id)` reads it. A startup reconciler drives `status=done` with `merged` provenance from unconsumed rows; the scheduler consults it before dispatch. This is the substrate W9/W10 consume — **α is the early, clearly-titled leaf they target.**
- **Scope 2 — Monkeypatch-path migration (ε).** One-time: move the test suite off `orchestrator.merge_queue.<private>` reach-back string paths onto the defining/owning module, and install a grep-guard ratchet forbidding NEW reach-back patches. Unfreezes the satellite seams — prerequisite for 3 and 4.
- **Scope 3 — SpecPermit + PermitLedger (ζ–θ).** A `SpecPermit` token stored ON the item, owned by a `PermitLedger` that is the single owner of `_speculation_slot`; `release(token)` idempotent. Conservation structural; the 5-location census, `_dispatching_item`/`_finalizing_head` census terms, the `release_resources` caller flags, and ~10 raw `.release()` sites die. Same for `_merge_ahead_cap` (`CapPermit`). Raw semaphore access banned via grep-guard.
- **Scope 4 — ItemLifecycle registry (ι–λ).** A registry keyed by `request_id` with an explicit state enum + legal-transition table; `transition()` at every put/pop. The four transient side-fields become states; `snapshot()`/permit-audit/liveness become single registry reads. Delete `_verify_phase`/`_verify_item` dual-writes and the free-form `phase:str`.
- **Scope 5 — QueuedBranch (μ–ν).** Frozen `QueuedBranch` value type parsed once at the boundary; `MergeRequest.branch` becomes `QueuedBranch`; normalizers/strip-readd/try-both logic die.
- **Scope 6 — Retire serial worker + compat shims (ξ–ο).** Migrate `_verify_and_advance` direct-call tests to the public surface and delete the shim; ratchet-freeze `_serial_merge_worker.py`; strip the normative "mirrors the test-local reference" docstrings.

---

## 5. Resolved design decisions (do not relitigate)

1. **Write-ahead ordering** (program decision #2): the landed row is fsynced **before** the CAS `main` advance. Write-after re-opens the exact crash window this whole chain exists to close. Both advance sites (`_finalize_inflight` and the train path) route through **one** helper so the ordering is single-sourced.
2. **Substrate = a new `LandedOutbox`, NOT the accept-side `MergeQueueStore` dict, NOT `event_store`, NOT git notes.** The accept-side journal is `remove()`d on terminal (so a landed row can't live there), and `event_store`'s `merge_finalized` ring is *run-scoped and written post-advance* (observability, not a done-reconciler — task 1750 confirms it's run-scoped by design). The landed-outbox is a distinct, cross-run, write-ahead, done-reconciling structure. It MAY be a second file managed by the `MergeQueueStore` module (same atomic-write discipline **plus fsync**) — the "extend MergeQueueStore" framing in finding 2 means "same module/owner," not "same dict."
3. **Reconciler cites `advanced_sha` as the `merged` commit.** `advanced_sha` is on `main` by construction, so the server-side `is-ancestor` backstop passes. The reconciler reuses `Scheduler.mark_done(kind='merged', sha=advanced_sha)` — it does NOT invent a new done-write path.
4. **Fused-memory server-side gates STAY** (program decision) — defense in depth at the durable write boundary. W1 adds an orchestrator-side source of truth; it does not remove the server gates.
5. **`PermitLedger` becomes the single owner of `_speculation_slot`**; `SpeculationController` (1993, merger-side) is refactored to acquire/transfer/release **through** the ledger rather than holding a raw semaphore reference. One owner, not two.
6. **Conservation is structural, not asserted.** `speculation_accounting_violations` becomes `slot_available + len(ledger.live) == depth` — a construction identity, so the audit can never false-fire on a new await window. The I4 audit is retained but reads the ledger.
7. **`QueuedBranch` is parse-don't-validate.** A single `parse(raw, branch_prefix)` classmethod is the ONLY place prefix logic lives; git refs are always built from `.full_name`. Mixed shape is unrepresentable past the two entry boundaries.
8. **The speculative worker is the sole behavioral spec.** The retired serial `MergeWorker` and `_verify_and_advance` shim are anchors to pay down, not references to preserve; production docstrings stop citing the retired worker as normative.

---

## 6. Pre-conditions for activating

- **None upstream.** W1 is wave 1 with no upstream deps.
- **Runtime activation is out of W1's scope.** These are library/refactor changes to the orchestrator; they take effect on the next orchestrator restart (dormant on `main` until then, harmless). W1 files **no deploy capstone** — fleet activation is a program-level concern (one restart after several streams land), to avoid thrashing the fleet once per stream. See §Open questions Q2.

---

## 7. Cross-PRD relationship (G4 — per program seam map)

| Other stream/PRD | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| W9 (workflow-state-machine) | **W9 consumes W1** | `MergeProvenance.lookup(task_id)` + landed-outbox → collapse workflow's three already-merged guards | **W1 owns the journal + API; W9 owns the collapse** | queued (W1 α provides; W9 wave-2 wires dep on α) |
| W10 (harness-supervision) | **W10 consumes W1** | landed-outbox → retire `find_merge_marker` archaeology in harness sweeps | **W1 owns the journal; W10 owns the sweep retirement** | queued (W1 α/γ provide; W10 wave-2 wires dep) |
| M3 (dashboard-alignment) | W1 must **not** collide | `OutcomeKind` merge-attempt enum in `merge_types.py` | **M3 owns `OutcomeKind`** | W1 introduces **no** competing outcome enum |
| W7 (verify-plan) | W1 must **not** collide | dry-run proposal spawning on the merge-verify block path; `BlockRecord` | **W7 owns** | W1 does not touch the merge-verify block-path wiring |
| scheduler dispatch gate | **W1 owns (in-batch)** | `_eligible_for_dispatch` consults `MergeProvenance.lookup` before dispatch | **W1 (δ)** | in-batch |
| I4 leak detector / `merge_liveness` / `snapshot()` | **W1 owns (in-batch)** | permit audit + liveness read the `PermitLedger` / `ItemLifecycle` registry | **W1 (ζ–λ)** | in-batch |

**No reciprocal-ownership ambiguity.** Every W9/W10 consumption edge is a clean "W1 produces, they consume" — the program doc's seam map is authoritative and W1 is a pure producer to wave 2.

---

## 8. Contract section (B + H) — the write-ahead landed-outbox seam

The high-stakes seam is the **write-ahead ordering of the landed-outbox row vs the CAS `main` advance**, and the reconciler that consumes it. An architect implementing the producer side must honor these signatures and invariants.

### 8.1 Value types (in `merge_queue_store.py` / a new `landed_outbox.py` under the same owner)

```
@dataclass(frozen=True)
class LandedRow:
    task_id: str            # canonical bare id (QueuedBranch.bare_id once ν lands)
    branch_tip_sha: str     # item.merged_branch_tip — the branch HEAD that was merged
    advanced_sha: str       # the SHA main was advanced TO (== the merge commit)
    landed_at: float        # wall time at write

class LandedOutbox:
    def record(self, row: LandedRow) -> None: ...   # append/update keyed by task_id; fsync BEFORE returning
    def lookup(self, task_id: str) -> LandedRow | None: ...   # public read; None if no row
    def all(self) -> list[LandedRow]: ...            # for the startup reconciler
    def consume(self, task_id: str) -> None: ...     # prune a row once its task is confirmed done (idempotent)

# public façade the survey named:
class MergeProvenance:
    @staticmethod
    def lookup(task_id: str) -> LandedRow | None: ...   # thin read over the worker's LandedOutbox
```

### 8.2 Invariants (prose-enforced contract → boundary tests below make them executable)

- **WA-1 (write-ahead durability).** `record(row)` MUST fsync the row to disk **before it returns**, and the call site MUST `await` it to completion **before** invoking `git_ops.advance_main(...)`. Ordering is single-sourced: both advance sites (`_finalize_inflight` :8783, train :2961) call one helper `_journal_landed_then_advance(...)` (or record-then-advance inline through a shared function). fsync covers the file **and its parent directory** (a rename without a dir-fsync can be lost on crash).
- **WA-2 (keyed by task_id, last-write-wins).** A task lands at most once to `done`; re-recording the same `task_id` overwrites. `lookup` returns the current row or `None`.
- **WA-3 (consumed = task confirmed done).** A row is "unconsumed" iff it exists AND its task is not yet `done`. `consume(task_id)` prunes it (idempotent). The row is NOT auto-removed on merge-terminal (unlike the accept-side journal) — it survives until the reconciler or the normal done-path confirms `done` and prunes it.
- **RC-1 (crash between fsync and advance → no phantom done).** If the process dies after `record` but before `advance_main` commits, `branch_tip_sha`/`advanced_sha` is NOT an ancestor of `main` at next start. The reconciler MUST detect this (`is_ancestor(advanced_sha, main) == False`), MUST NOT mark the task done, and MUST prune the row (the task re-dispatches through normal channels).
- **RC-2 (crash between advance and done-write → row drives done).** If the process dies after `advance_main` commits but before `mark_done`, `advanced_sha` IS an ancestor of `main`. The reconciler MUST `mark_done(task_id, kind='merged', sha=advanced_sha)` and then `consume(task_id)`.
- **RC-3 (already-done → prune only).** If the task is already `done` at reconcile time, `consume(task_id)` prunes the row; no done-write.
- **SD-1 (scheduler consult).** Before dispatching task Z, the scheduler consults `MergeProvenance.lookup(Z)`; if a row exists whose `advanced_sha` is an ancestor of `main`, the scheduler MUST NOT dispatch Z and MUST drive it through the same reconcile-to-done routine (shared with the startup reconciler). This is additive to (does not remove) the workflow guards W9 owns.
- **DEF-1 (defense in depth preserved).** The fused-memory `_validate_done_provenance` gate is unchanged; every reconciler done-write passes it because `advanced_sha` is on `main`.

### 8.3 Permit / lifecycle contract (scopes 3–4)

- **P-1 (permit conservation, structural).** `PermitLedger` owns `_speculation_slot`; at all times `slot_available + len(ledger.live) == depth`. `release(token)` is idempotent (`token.released` flag) and asserts the token is live; double-release is a no-op. The token lives ON the item and travels through every queue/park/dispatch/finalize.
- **P-2 (no raw semaphore access).** After θ, no code outside `PermitLedger` calls `_speculation_slot.acquire()/.release()` or `_merge_ahead_cap.acquire()/.release()` — enforced by a grep-guard test.
- **L-1 (single lifecycle source).** `ItemLifecycle` registry keyed by `request_id`; every put/pop calls `transition(rid, from, to)`; illegal transitions raise/escalate. `snapshot()`, `speculation_accounting_violations`, and the liveness ledger derive **only** from the registry — a test asserts they agree across a full pipeline run.

---

## 9. Boundary-test sketch (B + H) — faces both producer and consumer sides

The crash-window contract test (RC-1 / RC-2) is the **headline G5 two-way boundary artifact** and is γ's observable signal. Crashes are simulated by an injected fault point (a hook that stops the pipeline after `record` / after `advance_main`), not a real process kill.

| # | Scenario | Preconditions | Postconditions (asserted) | Faces |
|---|---|---|---|---|
| B1 | Normal land | merge succeeds end-to-end | row recorded write-ahead; task `done`; row consumed (`lookup==None` after done) | producer + consumer |
| B2 | Crash between fsync and advance (RC-1) | fault after `record`, before `advance_main` | at restart: `advanced_sha` NOT on main → reconciler prunes row, task stays re-dispatchable, **no phantom done** | producer + reconciler |
| B3 | Crash between advance and done-write (RC-2) | fault after `advance_main` commits, before `mark_done` | at restart: `advanced_sha` on main, task not done → reconciler `mark_done(merged, advanced_sha)`, row consumed | producer + reconciler |
| B4 | Already-done at reconcile (RC-3) | row present, task already `done` | reconciler prunes row, no second done-write | reconciler + fused-memory gate |
| B5 | Scheduler consult (SD-1) | landed row for Z on main, Z still pending | `acquire_next` does NOT dispatch Z; Z driven done via shared routine | scheduler + reconciler |
| B6 | Both advance sites journal | single-branch AND train finalize | both `_finalize_inflight` and the train path write a row before advancing (fake `git_ops` asserts row present when `advance_main` invoked) | producer (both sites) |
| B7 | Permit conservation under new await (P-1) | acquire N permits, park/dispatch/finalize across queues | `slot_available + len(ledger.live) == depth` holds at every step; double-release is a no-op | permit ledger |
| B8 | Registry single-source (L-1) | run items through the full lifecycle | `snapshot()`, permit audit, liveness all derive identical state from the registry (cannot disagree) | lifecycle registry |
| B9 | QueuedBranch round-trip (scope 5) | `parse('4778', 'task/')` and `parse('task/4778', 'task/')` | both yield `bare_id='4778'`, `full_name='task/4778'`; git refs built from `.full_name`; mixed shape unrepresentable | branch identity boundary |

---

## 10. Decomposition plan (the linear spine; Greek labels → task IDs at decompose)

Every leaf carries `force_full_path=true` (each touches the shared god-file with cross-cutting design implications; none is a safe fast-path candidate). `metadata.files` is file-level throughout. Priorities: journal chain (α–δ) and permit chain (ζ–θ) **high** (crash-window correctness + the live I4 false-positive class, and α unblocks wave 2); the rest **medium**.

### Scope 1 — Landed-outbox write-ahead journal

- **α — `LandedOutbox` durable store + `MergeProvenance.lookup(task_id)` public API.** *(the early, clearly-titled leaf W9/W10 target)*
  Modules: `orchestrator/src/orchestrator/merge_queue_store.py` (or new `orchestrator/src/orchestrator/landed_outbox.py`), `orchestrator/src/orchestrator/merge_queue.py` (worker holds the `LandedOutbox` instance).
  Signal: a unit test constructs the store, `record(row)`, re-opens it from disk (simulated restart), `lookup(task_id)` returns the row; `lookup(unknown)` returns `None`; `consume` prunes idempotently; the write is fsync-durable (survives simulated crash). Prereqs: —
- **β — Write-ahead wiring at both CAS advance sites (fsync-before-advance).**
  Modules: `merge_queue.py`.
  Signal: at both `_finalize_inflight` (:8783) and the train finalize path (:2961), a `LandedRow` is fsynced BEFORE `advance_main` — a fake `git_ops` asserts `lookup(task_id)` is non-None at the moment `advance_main` is invoked (boundary test B6). Prereqs: α.
- **γ — Startup reconciler + crash-window contract test (G5 two-way boundary).**
  Modules: `merge_queue.py` (or the recovery entry-point alongside `recover_pending_merges`), `orchestrator/tests/` (contract test).
  Signal: the crash-window contract test (B2 + B3 + B4) is green — kill-after-fsync converges to *re-dispatchable, no phantom done*; kill-after-advance converges to *task done with `{kind:merged, commit:advanced_sha}`*, row consumed; already-done prunes only. Prereqs: β.
- **δ — Scheduler consult-before-dispatch gate.**
  Modules: `orchestrator/src/orchestrator/scheduler.py`, `merge_queue.py`.
  Signal: a scheduler test with a landed row for Z present (advanced_sha on main) → `acquire_next` does not dispatch Z and Z is driven `done` via the shared reconcile routine (boundary test B5). Prereqs: γ.

### Scope 2 — Monkeypatch-path migration

- **ε — One-time test monkeypatch-path migration + grep-guard ratchet.**
  Modules: `orchestrator/tests/` (the reach-back-patching test files), a new grep-guard test.
  Signal: the full suite is green after repointing the `orchestrator.merge_queue.<private>` string-path monkeypatches to the defining/owning module; a grep-guard ratchet test is present, passes on the current tree (allowlist of any residual), and **fails when a new `orchestrator.merge_queue`-private string-path patch is added** (demonstrated by a fixture). Prereqs: δ.

### Scope 3 — SpecPermit + PermitLedger

- **ζ — `SpecPermit` token + `PermitLedger` (single owner of `_speculation_slot`).**
  Modules: `merge_queue.py`, `orchestrator/src/orchestrator/merge_speculation_controller.py`, `merge_types.py` (token type).
  Signal: unit test — acquire N permits → `len(ledger.live)==N` and `slot_available==depth-N`; `release(token)` idempotent (double-release is a no-op, no over-release); conservation identity `slot_available + len(ledger.live) == depth` holds (boundary test B7). Prereqs: ε.
- **η — Thread the token through the pipeline; delete the 5-location census, census side-fields, `release_resources` flags, raw release sites.**
  Modules: `merge_queue.py`, `merge_speculation_controller.py`.
  Signal: `_inflight_speculative_count` is deleted/reduced to `len(ledger.live)`; `speculation_accounting_violations` reads the ledger; the ~10 raw `_speculation_slot.release()` sites and the `release_resources=`/`_entry_released` caller flags are gone; existing merge-queue tests green. Prereqs: ζ.
- **θ — `CapPermit` for `_merge_ahead_cap` + grep-guard ban on raw semaphore access.**
  Modules: `merge_queue.py`, `orchestrator/tests/` (grep-guard).
  Signal: `_merge_ahead_cap` is owned by the ledger (`CapPermit`); a grep-guard test asserts no raw `_speculation_slot`/`_merge_ahead_cap` `.acquire()`/`.release()` outside `PermitLedger` (boundary contract P-2); tests green. Prereqs: η.

### Scope 4 — ItemLifecycle registry

- **ι — `ItemLifecycle` registry + state enum + legal-transition table + `transition()`.**
  Modules: `merge_queue.py`, `merge_types.py` (state enum).
  Signal: unit test — the legal transition sequence (QUEUED→…→TERMINAL) succeeds; an illegal transition raises/escalates; the registry is the single source of an item's state. Prereqs: θ.
- **κ — Call `transition()` at every put/pop; convert the 4 transient side-fields to states; make `snapshot()`/permit-audit/liveness single registry reads.**
  Modules: `merge_queue.py`, `merge_liveness.py`.
  Signal: `snapshot()`, `speculation_accounting_violations`, and the liveness ledger derive from the registry and provably agree across a pipeline run (boundary test B8); the four transient fields (`_inflight_req`/`_remerging_item`/`_finalizing_head`/`_dispatching_item`) are removed. Prereqs: ι.
- **λ — Delete `_verify_phase`/`_verify_item` dual-writes and the free-form `phase:str` (phase = registry state).**
  Modules: `merge_queue.py`, `merge_types.py`.
  Signal: the vestigial `_verify_item`/`_verify_phase`/`_verify_started_at` fields and their 4 dual-write sites are deleted; `InflightEntry.phase` derives from the registry; the stale-phantom-snapshot class (comment :5552) is structurally impossible; tests green. Prereqs: κ.

### Scope 5 — QueuedBranch

- **μ — `QueuedBranch` frozen value type + `parse(raw, branch_prefix)` classmethod in `merge_types.py`.**
  Modules: `merge_types.py`.
  Signal: unit test — `parse('4778', prefix)` and `parse('task/4778', prefix)` both yield `bare_id='4778'`, `full_name='task/4778'`; round-trips; mixed shape unrepresentable (boundary test B9, producer half). Prereqs: λ.
- **ν — `MergeRequest.branch` → `QueuedBranch` at the two boundaries; build refs from `.full_name`; delete `canonical_queued_branch_name`, try-both `resolve_queued_branch_ref`, journal strip/re-add.**
  Modules: `merge_types.py`, `orchestrator/src/orchestrator/git_ops.py`, `merge_queue_store.py`, `merge_queue.py`.
  Signal: `MergeRequest.branch` is a `QueuedBranch`; the ~8 inline `f'{branch_prefix}{...}'` constructions and the normalizers are deleted; pyright is clean (it enforces the invariant at every prior drift site); tests green. Prereqs: μ.

### Scope 6 — Retire serial worker + compat shims

- **ξ — Migrate the `_verify_and_advance` direct-call tests to the public surface + delete the shim.**
  Modules: `merge_queue.py`, `orchestrator/tests/test_merge_queue.py`, `orchestrator/tests/test_merge_queue_invariant_integration_gate.py`, `orchestrator/tests/test_merge_item_union.py`.
  Signal: `_verify_and_advance` is deleted from `merge_queue.py`; the ~30 direct-call sites (3 test files) drive `run()`/`_dispatch_item`+`_finalize_inflight`; suite green. Prereqs: ν.
- **ο — Freeze `_serial_merge_worker.py` (no-NEW-imports ratchet) + strip the normative "mirrors the test-local reference" production docstrings.**
  Modules: `orchestrator/tests/` (ratchet test), `merge_queue.py` (docstrings).
  Signal: a ratchet test forbids NEW imports of `_serial_merge_worker`; the ~10 production docstrings citing the retired worker as a byte-identical spec are de-normativized; a grep confirms no production docstring cites the retired worker as normative. Prereqs: ξ.

---

## 11. Out of scope for this PRD

- **`OutcomeKind` merge-attempt enum** — owned by M3 (`merge_types.py`). W1 consumes it if/when it lands; W1 introduces **no** competing outcome enum.
- **Workflow already-merged guard collapse** — W9 consumes W1's `MergeProvenance.lookup`; the collapse of `workflow.py:1642/1757` guards is W9's.
- **Harness `find_merge_marker` sweep retirement** — W10 consumes the landed-outbox; the sweep changes are W10's.
- **Dry-run proposal spawning on the merge-verify block path / `BlockRecord`** — W7 owns that wiring.
- **The satellite state-line extraction along `DriftDetectorSchedule`/`ShadowCompareScheduler` owning-objects** (finding 3's deeper proposal) — this PRD does the **monkeypatch-path migration (ε)** that *unfreezes* the seam, and the permit/lifecycle owning-objects (scopes 3–4). The full per-satellite owning-object extraction is deliberately **deferred** to a follow-up (the linear chain here already caps at 15 leaves matching the df 1985-2002 precedent; see §Open questions Q3).
- **Fleet deploy / orchestrator restart to activate** — program-level (§6, Q2).

---

## 12. Open questions (surfaced but not decided — tactical; AFK-safe defaults recorded)

1. **Linearization of scopes 5 and 6.** The brief writes the chain as `1 → 2 → 3 → 4 → (5, 6)` — 5 and 6 as parallel branches off 4. **Default taken: fully linearize `λ → μ → ν → ξ → ο`** (single spine). Rationale: both scopes touch `merge_queue.py`/its tests heavily; parallel dispatch would guarantee rebase conflict on the #1-churn file — the exact hazard the linear df 1985-2002 precedent avoids. Revisit only if throughput (not correctness) becomes the constraint. Decide-at: dispatch of μ.
2. **Deploy/activation capstone.** **Default taken: W1 files NO deploy capstone.** Changes are dormant-on-main until an orchestrator restart; a per-stream restart would thrash the 6-unit fleet once per stream. Activation is a program-level restart after several streams land (resolved decision #6 in the program doc governs the mechanics: out-of-cgroup `systemctl --user restart` for fused-memory; deterministic task-kind self-restart for orchestrators, 2064/2105). Decide-at: program-level deploy planning.
3. **Depth of scope 2 vs the full satellite extraction.** ε unfreezes the reach-back convention; it does NOT itself move satellite state into owning objects (finding 3's `DriftDetectorSchedule`/`ShadowCompareScheduler`). **Default taken: ε = migration + ratchet only; owning-object extraction deferred** to keep the chain at the 15-leaf precedent size and because scopes 3–4 already prove the owning-object pattern on the higher-stakes permit/lifecycle state. A follow-up PRD can extract the drift/shadow satellites once ε has removed the freeze. Decide-at: post-W1 review.
4. **`LandedOutbox` file layout.** Second file under the `MergeQueueStore` owner vs a subkey namespace in the existing journal. **Default: a separate file with the same atomic-write discipline + added fsync** (keeps the accept-side `remove()`-on-terminal semantics cleanly separate from the survive-until-consumed landed rows). Local/recoverable; an architect may pick either. Decide-at: task α impl.
5. **Scheduler consult on a live (non-startup) row.** SD-1 says the scheduler drives done via the shared routine on a consult-hit. Whether δ invokes the reconciler inline or defers to the startup reconciler for the rare live-desync case is tactical. **Default: consult-hit drives done inline via the shared routine** (converges immediately, no re-dispatch). Decide-at: task δ impl.

---

## 13. Note on tracking metadata

Per the prd skill: the orchestrator does **not** currently read the `user_observable_signal` / `consumer_ref` / substrate-confirmed metadata fields these tasks carry — they are substrate for a future tracking-infra session. The capability manifest beside this PRD (`plans/merge-queue-reliability-prd.capability-manifest.md`) is the artifact a dispatch-time architect or downstream verifier diffs against substrate.
