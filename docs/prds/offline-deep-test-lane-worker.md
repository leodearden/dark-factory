# PRD — Offline deep-test lane, Part B (dark-factory async lane worker)

**Status:** author-complete, gates passed (2026-07-01). Decompose-ready.
**Slug:** `offline-deep-test-lane-worker` · **Milestone:** orchestrator verify/merge infra (`docs/prds/`).
**Authoritative design:** `/home/leo/src/reify/docs/design/offline-deep-test-lane.md` (ratified D1–D5,
2026-06-09; §5 trigger/single-flight, §7 failure handling, §8 warm worktree, §10 ownership, §11 invariants).
**Part A (the reify-local primitives this PRD consumes):** `/home/leo/src/reify/docs/prds/offline-deep-test-lane.md`
+ `.capability-manifest.md` (author-complete, decompose-ready).
**Companion baseline:** reify `docs/notes/warmer-builds-phase0-baseline.md`; `docs/design/warmer-builds-merge-verify.md`.

This is **Part B** of the two-PRD decomposition the design §12 names: the **dark-factory async lane
worker** — the `on_post_merge` trigger, the singleton single-flight/coalescing/always-from-head worker,
the dedup'd fix-task spawn + staged escalation, the second persistent warm worktree, and the gate flip.
It **consumes** Part A's reify-local slice (the `heavy` filter, the `DF_VERIFY_ROLE=offline` role, and
`scripts/run-offline-deep.sh`) and **pulls** Part A's off-by-default flip seam. This PRD does **not
re-design** — it converts the ratified design into an implementation chain.

**Approach: B + H** (contract + boundary-test sketch). Blast radius ≥ 6 orchestrator modules, ~8
mechanisms, touches the load-bearing merge-landing seam (`on_merge_landed`), carries the hard
"**never a gate**" invariant, and is cross-repo. The design already supplies the contract-level
detail (§5/§7/§8); §Contract and §Boundary-test sketch below transcribe and pin it so the
integration step lands as a first-class task rather than starving under the narrow-lock orchestrator.

---

## 0. Goal & user-observable surface (G1)

**Goal.** Run reify's heavy numeric suite **post-merge, asynchronously, off the verify hot path**, on a
single-flight cadence keyed to `main` advancing, with autonomous failure handling — then flip the gate
to stop paying for the heavy set synchronously. The merge gate gets **faster** *and* coverage
**increases** (the lane runs the full matrix + the currently-`#[ignore]`'d convergence studies). The
lane is a safety net; it **never blocks a merge** (D1/§11).

**What an operator observes when this lands:**
- On every `main` advance the orchestrator lands, an offline lane run kicks off (a log line at the
  advance moment; a run record with the snapshot head SHA, pass/fail, duration).
- The lane runs **from head** — a run started while another is in flight coalesces and re-runs at the
  newest head, never a stale SHA (observable: the run log's head SHA equals current `main`).
- A **confirmed** red heavy test produces a **normal `pending` fix task** in the task tree
  (`get_tasks`, with failing-test IDs + suspect commit range in `metadata`) **and** an `escalate_info`
  (`get_pending_escalations`) — not a red-main autofix, not a merge block.
- The same failing-test set on a later advance **updates** the existing fix task (appends the suspect
  range) and spawns **no duplicate**; a fail-then-pass logs a low-severity "intermittent
  nondeterminism" and spawns **no** task.
- Merges keep flowing the whole time — a normal advance is never blocked, halted, or gated by an
  in-flight lane run or a red lane result.
- Once the lane is live, the flip lands and reify's `DF_VERIFY_ROLE=merge ./scripts/verify.sh
  --print-plan` emits `not (heavy)` — the heavy set is off the gate, merges are faster, and the change
  is one reversible config line.

**Mechanisms introduced, each with a named consumer (no orphan producers):**

| Mechanism | Consumer |
|---|---|
| `on_post_merge` trigger (fan-out on `on_merge_landed`, `harness.py`/`merge_queue.py`) | the singleton lane worker (sets its `dirty` flag) |
| Singleton lane worker (single-flight / coalesce / always-from-head; lockfile) | consumes the trigger + the warm worktree; invokes reify `scripts/run-offline-deep.sh`; its runs feed the failure handler + operator logs |
| Dedup'd fix-task spawn (failing-test-set fingerprint) + `escalate_info`/`escalate_blocker` staging | the orchestrator's standard TDD→PR→merge-gate loop (the fix task) + the escalation queue / L2 watcher (the escalations) |
| Second persistent warm worktree `_offline-deep` (Phase-1 machinery, 2nd instantiation) | the lane worker (runs the suite in it) |
| `flip-gate-exclude-heavy` (deterministic config deploy) | the reify merge/task gate — pulls Part A's `REIFY_GATE_EXCLUDE_HEAVY` knob to `1`; user surface = a faster gate running `not (heavy)` |

## 1. Background & premise (G6 record)

Phase 0 (reify `docs/notes/warmer-builds-phase0-baseline.md`, measured 2026-06-09 on an idle box) found
a *warm* merge-gate verify ≈ 11 min, of which **~643 s is test-exec warmth cannot touch** (compile
collapses to ~9 s under the warm worktree; the exec floor does not move). The floor is
**tail-latency-bound** — ~11k fast tests clear in 30–60 s, then a handful of long numeric tests
(`reify-solver-elastic` `determinism::*` thread sweeps; `analytical_validation` P2 + the `#[ignore]`'d
convergence studies; `modal_benchmarks`; `buckling_smoke`; two heavy `reify-eval` OCCT FEA binaries)
run 60–120 s with most cores idle, and the heavy set that runs on the gate runs in **both** the debug
and release passes (paid ~twice). They add little marginal *delta* coverage.

**G6 premise — design-first, not re-opened; sanity-checked vs HEAD.** The design's premise is ratified
(D1–D5) and was re-validated empirically against reify `main @ 0113758b11` (2026-07-01) in **Part A §1**
after **LPT nextest scheduling landed** (task #4627): the ~643 s warm exec floor was *measured* not
asserted; LPT cannot shrink a single test that spans a whole pass
(`default_parallel_tolerance_equivalent_across_thread_counts` is `SLOW [>120 s]`); and LPT's overrides
target the *same* heavy binaries — living evidence they are still the long poles. LPT and the offline
lane are **complementary**. Part B adds **no new numeric premise** — its leaf signals are structural
(log lines, task-tree/escalation appearance, `--print-plan` diffs), so the G6 numeric/exactness
branches do not fire here (the one numeric floor in this effort is Part A's A3 gate smoke, already
pinned to an already-green bound). See §Capability manifest for the per-leaf G3/G6 bindings.

## 2. Ratified decisions (from the design; Part-B-realized)

| # | Decision | Part-B realization |
|---|---|---|
| **D1** | Tier, don't remove — never a gate. | The lane is async and non-blocking; the flip only removes the heavy set from the gate *after* the lane is live to catch it (§Contract). |
| **D2** | Trigger on main-advance; single-flight; **always-from-head**. | `on_post_merge` fan-out → `dirty` flag → snapshot-head-at-run-start worker loop; correctness lives in the snapshot, not the trigger (β1/β2). |
| **D3** | Footprint = idle class + `--test-threads=N` cap, off the merge jobserver. | Inherited from Part A's `offline` role (`nice -n 19 ionice -c3`, `CARGO_MAKEFLAGS` unset). Part B does not re-implement footprint; it invokes the role. |
| **D4** | Failure = confirmation re-run → dedup → normal `pending` fix task + `escalate_info`; `escalate_blocker` only on stall. | β3. Deliberately **more** autonomous than red-main: the fix goes *through the gate* (a normal queued task), **not** the B3 red-main fix-forward path (`b3_gate.py:295` hard-aborts that class to a human). |
| **D5** | Warm build = dedicated self-warming worktree reusing Phase-1 machinery — never shared/overlaid. | δ instantiates the Phase-1 machinery (`git_ops.py`/`warm_lane_pool.py`) a **second** time as `_offline-deep`. |

**Part-B-specific decisions (resolved this session — conversions of the design, not new design):**

- **DB1 — The trigger is the in-process `on_merge_landed` fan-out; the worker is orchestrator-managed
  (in-process singleton, lockfile-guarded).** The design (§5) offered "systemd `--user` unit **or**
  orchestrator-managed singleton." Chosen: **orchestrator-managed**, because the *primary* trigger seam
  is an in-process callback (`merge_queue.py:10575 → harness.py:_note_merge_all`, fail-open) — an
  in-process worker receives it directly, mirroring the existing `note_merge` coordinator fan-out. The
  reify `hooks/reference-transaction` main-move log stays as the **fallback** trigger for `scripts/land.sh`
  landings when the orchestrator is down (§Contract). A lockfile still enforces single-instance
  (systemd-unit form remains a drop-in alternative that pairs with the log-fallback trigger — Open Q).

- **DB2 — The flip is a dark-factory `task_kind='deterministic'` config deploy, so the `→A4` edge is a
  genuine cross-project dependency.** `REIFY_GATE_EXCLUDE_HEAVY=1` is set in the **reify-repo-tracked**
  `/home/leo/src/reify/orchestrator.yaml` `verify_env:` block, which the DF orchestrator deep-merges per
  project (`config.py:122-125`). Because that file lives in reify's tree, a *normal* dark-factory task
  cannot edit it (normal tasks land on their own repo's `main`). A **deterministic config-deploy** task
  (CLAUDE.md "auto-deploy" preset: `before_done` present, `always_escalates=false`) runs a committed
  script that flips the knob and reloads — and its dep on Part A's `A4` is a real cross-project
  `add_dependency` edge (`reify:<A4-id>`), exactly as the design/Part A §6 require. See §Contract.

- **DB3 — Dedup keys on the failing-test-set signature, NOT `main_sha`.** Model on
  `workflow.py:402 compute_preexisting_main_break_fingerprint`, but fingerprint the **set of failing
  test IDs** so one open fix task absorbs the same red **across advances** (append the new suspect
  range); a *different* failing set spawns its own task. Keying on `main_sha` (as the model does) would
  spawn a fresh task+escalation on every advance while red — the exact flood §7/§11 forbids.

## 3. Pre-conditions for activating

- **warmer-builds Phase 1 (warm-lane CoW pool) — LIVE.** reify task ε #4663; dark-factory
  `git_ops.py`/`warm_lane_pool.py`/`config.py` warm-lane knobs present. The worktree-machinery dep is
  **satisfied**; δ instantiates it a second time.
- **Part A reify-local primitives landed on reify `main`.** The lane worker's runtime entry
  (`scripts/run-offline-deep.sh`, the `DF_VERIFY_ROLE=offline` role, the `heavy` filter, the
  `REIFY_GATE_EXCLUDE_HEAVY` knob) is delivered by Part A (A1/A2/A4/A5). Part A is author-complete /
  decompose-ready but its **tasks are not yet landed**. Part B code can be *built* in parallel; the lane
  can only *run* (ζ) once Part A's runner + role are on reify `main`, and the flip (ε) can only fire once
  Part A's knob (A4) is on reify `main`. Wired via cross-project deps at decompose (§6).

## 4. Sketch of approach

- **Trigger (β1).** Add an `on_post_merge` notifiee to the existing post-advance fan-out. The merge
  worker already invokes `self._on_merge_landed(task_id, base_sha, head_sha)` fail-open at the advance
  moment (`merge_queue.py:10569-10580`); `harness.py:_note_merge_all` (4973) fans that to `note_merge`
  coordinators. Register the offline lane worker as an additional notifiee — full async context, exact
  SHAs, fires at the precise advance. Fallback: reify `hooks/reference-transaction` main-move log.
- **Worker (β2).** A singleton loop (new `offline_lane.py` module — design pointer `workflow.py`; final
  module placement tactical, reuses `workflow.compute_preexisting_main_break_fingerprint`), launched and
  owned by `harness.py`. `on_post_merge` sets `dirty`. Loop: when idle and dirty → **snapshot current
  head** (`git rev-parse main`), clear dirty, invoke `run-offline-deep.sh` in the `_offline-deep`
  worktree at that head; if dirty was re-set during the run, immediately re-run at the *new* head; else
  wait. Lockfile enforces one instance. A cheap `git rev-parse main` poll backstops a missed trigger
  (correctness is in the snapshot, not the trigger).
- **Failure handling (β3).** On a red run: **confirmation re-run** of only the failing tests,
  isolated/serial, once (filters infra flake + marginal nondeterminism — fail-then-pass ⇒ low-severity
  "intermittent nondeterminism" log, **no** task). On confirmed red: **fingerprint the failing-test
  set** (DB3); if an open fix task exists for that signature, **update** it (append suspect range); else
  **file a normal `pending` fix task** (failing test IDs + suspect commit range in `metadata`, via
  `_post_submit_tasks → submit_task`) that the orchestrator drives through the standard TDD→PR→**merge-gate**
  loop, and raise **`escalate_info`**. **Promote to `escalate_blocker`** (L2) only if the fix task
  can't land or the suite stays red past *N* advances.
- **Warm worktree (δ).** Instantiate the Phase-1 machinery a second time as a dedicated `_offline-deep`
  persistent worktree (model on `git_ops.py:117 PERSISTENT_MERGE_WORKTREE_NAME='_merge-verify'`),
  self-warming at head, single-consumer of its **own** `target/`, narrow-cone build scope
  (`reify-solver-elastic` + `reify-eval` dependency cone, not the full workspace), honoring warm-lane
  §11 invariants (never shared/overlaid; exempt from prune).
- **Flip (ε).** A deterministic config-deploy (DB2) sets `REIFY_GATE_EXCLUDE_HEAVY: "1"` in reify's
  `orchestrator.yaml verify_env`, deps on Part A `A4` (cross-project) **and** ζ (lane-live), and reloads
  the orchestrator so the gate immediately runs `not (heavy)`. Reversible (delete the line / set `0`).

## 5. Contract (B+H — seam signatures, invariants, ordering)

**C1 — Trigger callback.** The lane worker exposes an async notifiee with the existing landing
signature `on_post_merge(task_id: str, base_sha: str, head_sha: str) -> Awaitable[None]`, registered
into the `on_merge_landed` fan-out (`harness.py:_note_merge_all`). **Invariants:** fail-open (a raising
notifiee never blocks or fails a merge — mirrors `merge_queue.py:10575-10581`); the callback does
**only** `dirty := True` + a wakeup (no work on the merge worker's thread); exact SHAs are passed
through but are **advisory** — the worker re-snapshots head at run-start (C2). Fallback source: the
reify `hooks/reference-transaction` main-move log yields the same `(base, head)` for orchestrator-down
landings.

**C2 — Single-flight / from-head ordering.** At most one run in flight (lockfile). Each run samples
`head = git rev-parse main` at **run-start**, not trigger-time. Coalescing rule: `dirty` set during a
run ⇒ exactly one re-run after it, at the *then-current* head; multiple advances during a run collapse
to one re-run. **Invariant:** a run's reported head SHA is always a real ancestor-or-equal of current
`main` at run-start — never a stale trigger SHA. A missed trigger costs *granularity*, never
*correctness* (the poll backstop catches up).

**C3 — Failure-signature dedup.** `fingerprint = stable_hash(sorted(failing_test_ids))` (DB3; **not**
`main_sha`). State: `open_fix_tasks: {fingerprint → task_id}`. On confirmed red for `S`: if
`S ∈ open_fix_tasks` ⇒ update that task's `metadata.suspect_ranges` (append), no new task, no new
escalation; else ⇒ `submit_task(status='pending', metadata={failing_tests, suspect_range})` + record +
`escalate_info`. **Invariant:** while a fix for `S` is in flight, no advance spawns a second task/escalation
for `S`. Confirmation re-run precedes all of this: a test that fails-then-passes is **never** fingerprinted.

**C4 — Escalation staging.** `escalate_info` (visibility, no page) on first confirmed red for `S`.
`escalate_blocker` (L2) **only** when the fix task for `S` reaches a terminal non-`done` state (can't
land) **or** `S` stays red past `N` advances (`N` = tunable, Open Q). Built via
`escalation.models.Escalation` enqueue (the path `harness.py`/`workflow.py` already use).

**C5 — Warm-worktree invariants (warm-lane §11, verbatim).** `_offline-deep` is a dedicated persistent
worktree at a fixed path, **single-consumer of its own `target/`**, **never** sharing or overlaying the
merge lane's `target/` (design §9 rejects live-shared/reflink/overlayfs reuse for this lane), exempt
from prune, self-warming at head (run 2 at the same head is a near-pure fingerprint pass, compile ≈ s).
Build scope is the `reify-solver-elastic` + `reify-eval` dependency cone only.

**C6 — Flip seam (the one cross-repo interface, owned by Part A).** reify `scripts/verify.sh`, on role
`task`/`merge`, applies `not (heavy)` **iff** `REIFY_GATE_EXCLUDE_HEAVY == "1"` (any other value ⇒ full
set, unchanged). Part A owns the seam + the default (`0`); Part B's ε **pulls** it to `1` in reify's
`orchestrator.yaml verify_env` via a deterministic config deploy. **Ordering invariant:** ε must not
fire until **both** A4 (the knob exists on reify `main`) **and** ζ (the lane is live and catching
offline runs) are satisfied — enforced by the two dependency edges, so the flip is immediate + atomic
the instant the lane can catch what the gate stops running (no double-pay window, no coverage gap).

**C7 — Never a gate (hard invariant, D1/§11).** No Part-B mechanism may block, halt, gate, or delay the
merge queue: the trigger is fail-open (C1), the worker runs out-of-band at idle class (D3), a red
result files a *normal queued* task (not a merge block, not the B3 red-main path). Enforced executably
by ζ's boundary test.

## 6. Cross-PRD / cross-repo relationship + seam ownership (G4 — load-bearing)

Same cross-repo seam class as cpu-governance (α/β/γ ↔ ζ) and warm-lane D8: **reify ships the primitives,
dark-factory wires the consumer.** Ownership is unambiguous — no reciprocal "the other owns it."

| Seam / deliverable | Direction | Mechanism | Owner | Status |
|---|---|---|---|---|
| reify Part A runner + role | Part B **consumes** | `scripts/run-offline-deep.sh`, `DF_VERIFY_ROLE=offline` (the worker's subprocess entry) | reify (Part A A5/A2) | precondition: landed on reify `main`; ζ deps cross-project on A5 at decompose |
| **flip seam** | Part B **pulls** what Part A **produces** | `REIFY_GATE_EXCLUDE_HEAVY` knob in reify `verify.sh` (A4); read from `orchestrator.yaml verify_env` | reify owns seam + default `0`; **dark-factory (ε) pulls to `1`** | wired via cross-project edge **ε → `reify:A4`** (§Sequencing) |
| fallback trigger | Part B **consumes** | reify `hooks/reference-transaction` main-move log (orchestrator-down landings) | reify (existing) | fallback only; primary is `on_post_merge` |
| `on_post_merge` trigger | internal (DF) | `on_merge_landed` fan-out (`harness.py`/`merge_queue.py`) | **dark-factory (β1)** | new |
| singleton lane worker | internal (DF) | single-flight / coalesce / from-head (`offline_lane.py`/`harness.py`) | **dark-factory (β2)** | new |
| dedup'd fix-task spawn + escalation staging | internal (DF) | failing-test-set fingerprint (`workflow.py` model) + `escalate_info`/`escalate_blocker` | **dark-factory (β3)** | new |
| 2nd persistent warm worktree | internal (DF); machinery LIVE | `_offline-deep` (Phase-1 machinery, `git_ops.py`/`warm_lane_pool.py`) | **dark-factory (δ)** | machinery live (task ε #4663); 2nd instantiation new |

**Reciprocity resolved.** Part A owns the seam + default (`0`); Part B owns the pull (`1`) **and** the
async lane that makes the pull safe. The flip is the reciprocal seam — resolved by the **cross-project
dependency edge** (ε → `reify:A4`) plus ε → ζ, **not prose** (per the brief's directive).

**Sequencing (the coordinated edge).** Part A's decompose runs in parallel. The cross-project flip edge
(ε → `reify:A4`) is wired once **both** batches exist. At Part B decompose: if Part A's tasks are
already filed, wire the edge from the Part B side via `add_dependency(id=<ε>, depends_on="reify:<A4-id>")`;
if not, leave a **documented follow-up** to wire it when Part A files, and record the pending edge in the
capability manifest. The same applies to ζ → `reify:<A5-id>` (the runner-landed precondition).

## 7. Decomposition plan (leaf tasks — each names a user-observable signal, G2)

> Greek labels; task IDs assigned at decompose. `metadata.files` follows the tight-or-empty rule.
> Per-leaf G3/G6 bindings are in the committed capability manifest beside this PRD.

- **β1 — `on_post_merge` trigger fan-out.** Add the offline-lane notifiee to `on_merge_landed`
  (`harness.py:_note_merge_all`), fail-open, passing `(task_id, base_sha, head_sha)`; document the reify
  `hooks/reference-transaction` log fallback. *Intermediate* — unlocks β2. *Signal:* on each landed
  advance an operator sees a log line `offline-lane: on_post_merge <base>..<head>` and the worker's
  `dirty` flag flips (observable in worker state/log); a notifiee exception is swallowed (a merge still
  lands). *Modules:* `harness.py`, `merge_queue.py`.
- **β2 — Singleton lane worker (single-flight / coalesce / always-from-head).** New `offline_lane.py`,
  launched by `harness.py`; lockfile singleton; dirty-flag loop; snapshot head at run-start; invoke
  `run-offline-deep.sh` in `_offline-deep`; coalesce; poll backstop. *Signal:* start the worker and
  advance `main` twice quickly → **exactly one** run executes and its run log's head SHA equals current
  `main`; a second advance during a run yields **exactly one** coalesced re-run at the newer head; a
  second worker instance refuses to start (lockfile). *Modules:* `offline_lane.py`, `harness.py`.
  *Deps:* β1, δ; cross-project precondition `reify:A5` (runner) at ζ.
- **β3 — Dedup'd fix-task spawn + staged escalation.** Confirmation re-run (isolate/serial, once) →
  failing-test-set fingerprint (DB3, model `workflow.py:402`) → update-or-file a normal `pending` fix
  task (`_post_submit_tasks → submit_task`) → `escalate_info`; promote to `escalate_blocker` on
  stall/past-N-advances. *Signal:* inject a reproducible red into the heavy set → a `pending` fix task
  appears (`get_tasks`) with failing-test IDs + suspect range in `metadata` **and** an `escalate_info`
  appears (`get_pending_escalations`); a later advance with the **same** failing set **updates** that
  task and spawns **no** duplicate task/escalation; a fail-then-pass logs "intermittent nondeterminism"
  and spawns **nothing**; the merge queue is untouched throughout. *Modules:* `offline_lane.py`,
  `workflow.py`. *Deps:* β2.
- **δ — Second persistent warm worktree `_offline-deep`.** Instantiate Phase-1 machinery a second time
  (`git_ops.py`/`warm_lane_pool.py`, model `PERSISTENT_MERGE_WORKTREE_NAME`); dedicated, self-warming at
  head, narrow-cone scope, honoring warm-lane §11 invariants (C5). *Signal:* the worktree exists at its
  fixed path and is prune-exempt; run 1 cold-bootstraps and run 2 at the same head compiles in ≈ seconds
  (near-pure fingerprint pass — visible in the run log's compile timing); the lane never touches the
  merge lane's `target/`. *Modules:* `git_ops.py`, `warm_lane_pool.py`, `config.py` (knob). *Deps:*
  warm-lane pool (LIVE). Feeds β2.
- **ζ — Lane-live integration gate (the B+H integration-gate leaf).** Stand up trigger + worker + warm
  worktree + failure handling **end-to-end** against a live reify checkout with Part A landed; run the
  §Boundary-test sketch scenarios. *Signal:* the boundary-test scenarios pass — a real `main` advance
  triggers a from-head offline run in `_offline-deep`; a normal advance flows through the merge queue
  **unblocked** during an in-flight lane run; an injected red spawns a **deduped** fix task +
  `escalate_info` **without touching the merge queue** (C7). This is the leaf the flip (ε) depends on.
  *Modules:* integration (no new production module). *Deps:* β1, β2, β3, δ; cross-project precondition
  `reify:A5`/`reify:A2` (runner + role on reify `main`).
- **ε1 — Commit the flip deploy script.** New `scripts/deploy/flip-reify-gate-exclude-heavy.sh` (idempotent:
  set `REIFY_GATE_EXCLUDE_HEAVY: "1"` in reify `orchestrator.yaml verify_env`, commit in the reify repo,
  signal orchestrator config reload; a `--check`/dry-run mode reports the intended diff). *Signal:* the
  script is committed + executable; `--check` prints the one-line diff it would apply and exits 0 on an
  unflipped config. *Modules:* `scripts/deploy/`. *Intermediate* — unlocks ε2 (the deterministic task's
  `before_done` must reference a committed, executable script).
- **ε2 — `flip-gate-exclude-heavy` (deterministic config deploy).** dark-factory
  `submit_task(task_kind='deterministic', metadata.before_done={script: ε1, timeout_secs, target_unit:
  <orchestrator unit>}, always_escalates=false)` — the "auto-deploy" preset (run action; escalate only
  on failure; else `done`). *Signal:* on dispatch the knob flips and the orchestrator reloads → reify
  `DF_VERIFY_ROLE=merge ./scripts/verify.sh --print-plan` now emits `not (heavy)` (the heavy set is off
  the gate); `done_provenance` stamped `kind='deterministic-deploy…'`; reverting the line restores the
  full gate. *Modules:* reify `orchestrator.yaml` (via ε1's script). *Deps:* **ζ** (local) + **`reify:A4`**
  (cross-project — the named flip edge). Reversible; no reify *code* change.

**Suggested edges:** β2 → {β1, δ}; β3 → β2; ζ → {β1, β2, β3, δ}; ε1 → ζ; ε2 → {ε1, ζ, `reify:A4`};
δ → (warm-lane pool, LIVE). Cross-project preconditions ζ → `reify:A5`/`reify:A2` wired at decompose if
Part A ids are available, else documented follow-ups.

## 8. Boundary-test sketch (B+H — cross-module, both sides of the seam)

The ζ integration-gate task's observable signal (closes G2). Each row faces both the **producer** (merge
worker / trigger / warm machinery) and the **consumer** (lane worker / gate).

| # | Scenario | Preconditions | Postconditions asserted |
|---|---|---|---|
| B1 | Advance triggers a from-head run | worker up; `_offline-deep` warm; Part A on reify `main` | one lane run starts; its snapshot head SHA == current `main`; `on_post_merge` log line present |
| B2 | Coalescing under a burst | a run in flight; 3 advances land during it | exactly **one** re-run afterward, at the newest head; no queue of stale runs |
| B3 | **Never a gate** (the load-bearing invariant) | a lane run in flight; a normal task ready to merge | the merge lands **unblocked**; merge-queue latency unaffected; no halt/gate event |
| B4 | Confirmed red → normal fix task + info | a heavy test deterministically red at head | a `pending` fix task appears with failing-test IDs + suspect range; an `escalate_info` appears; **no** merge block; **not** the B3 red-main path |
| B5 | Dedup across advances | scenario B4's fix task open; another advance, same failing set | the existing task is **updated** (suspect range appended); **no** duplicate task/escalation |
| B6 | Flake filtered | a heavy test fails-then-passes on the confirmation re-run | a low-severity "intermittent nondeterminism" log; **no** fix task, **no** escalation |
| B7 | Stall → blocker | a fix task for `S` can't land, or `S` red past `N` advances | `escalate_info` **promotes** to `escalate_blocker` (L2); still no merge block |
| B8 | Self-warming | run 2 at the same head | compile ≈ seconds (fingerprint pass); the lane's `target/` is never the merge lane's |
| B9 | Flip is live + atomic + gated | ε2 dispatched after ζ green + A4 on reify `main` | reify gate `--print-plan` flips to `not (heavy)`; the flip did **not** fire before both edges satisfied; reverting restores the full gate |

## 9. Out of scope (Part B)

- The reify-local partition, `offline` role, gate smoke, `run-offline-deep.sh`, and the
  `REIFY_GATE_EXCLUDE_HEAVY` **seam** itself — **Part A** (this PRD only *pulls* the seam).
- A dedicated lean build profile for the lane (design §8/§10 "optional") — deferred follow-up; δ uses
  narrow-cone scoping, which is sufficient to bound disk.
- Re-tuning reify's `--test-threads=N` / footprint beyond invoking Part A's `offline` role — footprint
  is D3/Part A's; Part B picks a starting `N` (Open Q) and invokes.
- Fixing any test that turns RED when a currently-`#[ignore]`'d convergence study first runs first-class
  offline — that is exactly what β3 files as a **normal fix task** (non-blocking, D1); it is a *product*
  of this lane, not a Part-B blocker.
- The CoW/reflink/overlayfs artifact-reuse alternatives — **rejected** for this single-flight lane
  (design §9); δ uses the dedicated self-warming worktree.

## 10. Invariants / do-nots

- **Never a gate.** No Part-B mechanism blocks, halts, gates, or delays the merge queue (C7/B3).
- **Fail-open trigger.** A raising `on_post_merge` notifiee must never fail or delay a merge (C1).
- **Dedup is mandatory.** Key on the failing-test-set signature, not `main_sha` (DB3/C3) — no naive
  per-advance fix-spawn.
- **Confirmation re-run before escalating.** Always filter flake/contention first (C3/B6).
- **Dedicated `target/` only.** `_offline-deep` never shares or overlays the merge lane's `target/`
  (C5; design §9; warm-lane §11).
- **Off the merge jobserver.** Inherited from Part A's `offline` role — never draw from
  `/tmp/reify-jobserver-*` (priority-blind admission).
- **The fix goes through the gate.** File a normal queued fix task; never the B3 red-main fix-forward
  path (`b3_gate.py:295`) and never an unattended `main` edit.
- **Flip only when both edges are satisfied.** ε2 fires iff A4 (knob on reify `main`) **and** ζ
  (lane live) — enforced by dependency edges, not timing (C6/B9).

## 11. Open questions (surfaced but not decided — tactical)

1. **Worker host: orchestrator-managed in-process singleton (DB1 default) vs systemd `--user` unit.**
   In-process pairs with the primary in-process trigger; the systemd form pairs with the log-fallback
   trigger and survives an orchestrator restart independently. **Suggested:** in-process (DB1). Decide at β2.
2. **Starting `--test-threads=N` for the `offline` role invocation.** Design §6: start modest (not 1),
   measure, tune. **Suggested:** a small fixed N (e.g. 4–8); it is a knob, not frozen. Decide at β2/ζ.
3. **`N` = advances-red-before-`escalate_blocker` promotion (C4).** **Suggested:** a small integer
   (e.g. 3) or a wall-clock equivalent. Decide at β3.
4. **`offline_lane.py` module placement.** Design pointer is `workflow.py`; a dedicated module reusing
   `workflow.compute_preexisting_main_break_fingerprint` is likely cleaner. Decide at β2.
5. **`_offline-deep` narrow-cone seed mechanism.** Whether reify's existing `seed-warm-lane.sh` supports
   a scoped (solver+eval-cone) seed or needs a scope arg. **Suggested:** reuse as-is; add a scope arg
   only if the full-workspace seed is too costly. Decide at δ.
6. **Config-reload mechanics for the flip.** Whether ε2's `before_done` reloads config in-place or
   restarts the orchestrator unit (deterministic-runner `target_unit` self-kill vs cross-unit). **Suggested:**
   the deterministic runner's standard restart path. Decide at ε2.
7. **Should reify's local `hooks/pre-merge-commit` (land.sh) path also honor the flip?** Mirrors Part A
   Open Q. **Suggested:** defer; the orchestrator path is primary. Revisit post-ζ.
