# Harness supervision — PRD

**Stream:** W10 (harness-supervision) of the bug-hotspot remediation program 2026-07-06 · **Wave:** 2
**Status:** active — authored 2026-07-06.
**Approach:** **B + H** (contract + two-way boundary tests). HIGH-STAKES: this refactors the
process-supervision, restart, deploy-lifecycle, and crash-recovery machinery of a **running**
factory — the exact layer whose diffuse re-hardening produced the 2064/2105/1900/2059/2066/2091
incident chain.
**Program doc (authoritative G4 seam map + resolved decisions):**
`plans/bug-hotspot-remediation-program-2026-07-06.md`.
**Brief:** `/home/leo/.claude/spawn-briefs/df-hotspot-2026-07-06/W10-harness-supervision.md`.
**Findings:** `plans/bug-hotspot-survey-2026-07-06-full-findings.json` — cluster 2 (harness, all
6) + cluster 4 (scheduler) findings 4.0(state-machine half), 4.1(TaskGroundTruth half), 4.5,
4.6. All `verdict: confirmed`; every anchor re-verified against current main on 2026-07-06 (line
numbers in the capability manifest beside this PRD).

---

## 1. Goal — what an operator observes if this lands

The four-way-duplicated restart machinery, the stamp-archaeology deploy lifecycle, the eleven
hand-rolled background loops, the seven independent reconciliation sweeps, and the private
Harness↔Scheduler monkey-patch seam each collapse to **one owned contract**. Concretely:

1. **A restart can never self-kill on unset `ORCH_UNIT`.** Every restart path — deterministic
   self-restart, cross-unit deploy, stale-service recovery, fleet restart-all — routes through a
   single `RestartPlan.execute()` that FAILS CLOSED (refuses + escalates) when it cannot prove the
   target is a different unit, always passes `--working-directory`, always absolutizes the script,
   always wraps detached payloads with the on-failure escalation shell, and always fresh-PID
   verifies blocking restarts. The 2064 fleet self-kill and the 2105 exit-127 cwd bug become
   **structurally impossible**, and the `schedule_detached_systemd_restart` "accepted gap"
   (its own docstring) dies.
2. **A deterministic deploy always lands in a NAMED phase.** Deploy progress is one
   `metadata.deploy_state` phase enum (`scheduled → ran → verified|failed|escalated → done`) with a
   persisted verify baseline and a transition table enforced at every write (illegal → loud
   escalation, never silent). Every crash window between two stamps now recovers by a defined
   action instead of inventing a new undefined state (the 1900 phantom-done / 2059 silent-strand /
   2066 lost-writeback class). Post-hoc freshness verification becomes possible (persisted baseline
   vs live `ActiveEnterTimestampMonotonic`), and an unrelated escalation can no longer alias as
   "human resolved the deploy."
3. **Shutdown stops hanging.** The eleven `_start/_stop/_loop` triplets become registrations in a
   `LifecycleRegistry`; `start_all()` runs in declared order, `stop_all()` cancels in reverse with
   a per-service timeout and one uniform `CancelledError` contract. `run()`'s comment-ordered
   startup ritual and the finally-ladder collapse. The recurring SIGTERM-hang class (108, 161/162/
   169, 875, 1080) becomes structurally impossible; `cli.py`'s `os._exit(137)` watchdog stays only
   as a last resort.
4. **Task ground truth has one owner.** A `TaskGroundTruth.derive_truth(tid)` resolver produces a
   typed `TruthReport`, and ONE classification table maps it to a recovery action. The seven sweeps
   become thin schedulers over it; `find_merge_marker` archaeology is consulted only as a fallback
   where W1's landed-outbox journal (`MergeProvenance.lookup`) does not answer. A new crash shape
   costs a table row, not a new 400-line sweep with fresh race guards.
5. **The Harness↔Scheduler seam is an explicit contract.** `SchedulerCallbacks` (a frozen
   dataclass) is passed to `Scheduler.__init__`; public accessors (`is_dispatched`,
   `is_actively_held`, `requeue_history`) replace the `_dispatched`/`lock_table._held`/
   `_requeue_history` reach-ins; a `finish_startup()` flag turns the comment-only "sweeps run before
   the first `acquire_next`" ordering invariant into a runtime assertion. Tests fake one protocol,
   not nine private attributes.
6. **`acquire_next` sequencing is data, not prose.** The 720-line tick becomes a `TickContext`
   built once + an ordered list of phase callables whose order a unit test asserts. Phase-level
   tests stop needing a full-tick harness; a reordering regression is caught by the list literal.

**User-observable surface:** every row of the §7 boundary-test sketch is asserted through a
product read path (task status via fused-memory, escalation state via the escalation APIs, restart
behaviour via injected fake `systemd-run`/inspector, shutdown behaviour via an injected wedging
service, dispatch behaviour via the real scheduler tick). None is a synthetic-input unit test.

---

## 2. Background — why the invariants are diffuse

Harness (`harness.py`, ~9,028 lines) is the god-module: `run()` is a ~650-line hand-numbered
startup ritual (step labels 0…1c1c1) followed by a finally-ladder shutdown; it hosts eleven
background-loop triplets and reaches into Scheduler privates. The survey confirmed six structural
hazards in this layer plus three in the scheduler that share its seams. The common cause is the
same one the whole program targets: **mechanism was built repeatedly without a single owner**, so
each re-hardening (cwd, script-absolutize, own-unit detection, escalate-on-fire-failure,
verify-by-fresh-PID, timeout-group-kill; reset-on-recovery, leak-on-terminal; is_ancestor →
rev-parse → find_merge_marker) was rediscovered in a different copy and the copies drifted.

The prior work this builds on is **all landed**: task 2091 (runner inspector timeout), 2064
(ORCH_UNIT self-kill fix), 2105 (detached-cwd exit-127 fix), 1900/2059/2066 (deploy crash-window
guards), 1807/1855/1880 (scheduler streak semantics). W10 does not redo those point-fixes — it
installs the **single contract** they each half-extracted, so the next crash shape does not spawn
the next point-fix.

Mechanism↔finding map (survey clusters 2 & 4):

| W10 mechanism | Survey finding(s) |
|---|---|
| M1 `proc_supervision.RestartPlan` | 2.0 (four restart mechanisms) |
| M2 `DeployState` typed schema | 2.1 (four scattered stamps) + 4.0 (state-machine half — MERGED: the scheduler reviewer's `metadata.deterministic_state` proposal is this) |
| M3 `BackgroundService`/`LifecycleRegistry` | 2.3 (eleven triplets + shutdown-hang class) |
| M4 `TaskGroundTruth` resolver | 2.4 (seven sweeps) + 4.1 (ground-truth half — the transition-table half is W2's) |
| M5 `SchedulerCallbacks` seam | 2.5 + 4.5 (subsumes the workflow reviewer's `_SchedulerLike` protocol at `workflow.py:212`) |
| M6 `acquire_next` phase decomposition | 4.6 |

---

## 3. Substrate reality check (G3) — all anchors re-verified against current main 2026-07-06

W10 introduces **no novel substrate that does not exist**: five of the six mechanisms are
pure-infrastructure re-wiring of capabilities already on main (the four restart mechanisms, the
four deploy stamps, the eleven triplets, the seven sweeps, the nine `_on_*` installs, the 720-line
tick — all confirmed present, current line numbers in the manifest). The only assumed capabilities
W10 does **not** own are **other streams' deliverables**, every one of which is a filed task wired
as a hard prerequisite:

| Assumed capability | Status | Owner / evidence | W10 resolution |
|---|---|---|---|
| `inspect_systemd_unit(unit, *, timeout_secs, reap_grace_secs)` module-level helper | queued (in-progress) | M2 α = **task 2119** → `orchestrator/systemd_inspect.py` (not on main yet) | γ **imports** it for the fresh-PID verify; γ `depends_on 2119` (never a second copy — program seam table) |
| `register_metadata_submodel(key, model)` extension point | queued (in-progress) | W3 α = **task 2158** → `shared/task_metadata.py` | ε registers `DeployState`; ε `depends_on 2158` |
| `claimant_run_id`/`heartbeat_at` columns + `is_stranded()` predicate | queued (pending) | W2 ρ2 = **task 2182** | θ1's `TruthReport.live_claimant` reads them (plan.lock fallback); θ1 `depends_on 2182` |
| `MergeProvenance.lookup(task_id)` landed-outbox journal | queued (pending) | W1 α = **task 2153** | θ1's `branch_state` resolves journal-first; θ1 `depends_on 2153` |
| collapsed single backfill helper + `StreakRegistry.gc(stale_ids)` | queued (in-progress) | M2 ε = **task 2124** | β's Backfill + stale-sweep phases **wrap** them (β does not re-collapse); β `depends_on 2124` |
| `EscalationQueue.get_by_task(..., agent_role=…)` scoped query | queued (pending) | M2 β = **task 2120** | ζ's "ever_escalated inference dies" needs scoped queries; ζ `depends_on 2120` |

`TerminalReport` / `WorkflowStateMachine` (W9) are **NOT on main** (grep empty) and W9's batch is
**not filed**. Per the brief, W10 therefore leaves the harness `_last_block_*` attr-read side
channel **untouched** — none of W10's six mechanisms reads it — and W9 owns replacing it later
(§8 Out of scope, §6 G4).

`Scheduler.__init__` (scheduler.py:960) takes **no** callbacks today (all `_on_*` are set
post-construction as attributes) — so M5's constructor-injection is additive; `TickContext` does
not exist (M6 introduces it); none of the six new target modules exists on main. G3 passes.

---

## 4. Sketch of approach

Six mechanisms → ten tasks (Greek labels α…ι). Because five of six touch the two god-files
(`harness.py`, `scheduler.py`) and the two runner/service files, the batch is a **mostly-linear
spine** — the W1 precedent (`df 1985-2002`, `df 2153-2183`): parallel dispatch on a #1-churn file
guarantees rebase conflict, which the linear chain avoids. New-file foundation tasks (ε
`deploy_state.py`, θ1 `task_ground_truth.py`) branch where their cross-stream deps allow but still
land before their god-file consumers.

- **M5 — `SchedulerCallbacks` seam (α).** A frozen `SchedulerCallbacks` dataclass injected at
  `Scheduler.__init__`; public `is_dispatched`/`is_actively_held`/`requeue_history` accessors
  (the cancel-grace liveness judgment moves in beside `_dispatched`/`lock_table._held`, its single
  writer); a `finish_startup()`/`started` flag asserting `acquire_next` is not called before the
  orphan reaper + stranded sweep have run. Harness stops assigning nine `scheduler._on_*` and stops
  reading `_dispatched`/`_held`/`_requeue_history`. Subsumes `workflow.py:212`'s `_SchedulerLike`
  protocol (one owned seam, not a parallel protocol split). **Excludes** `_module_cache` — that
  seam is M2 δ's `Scheduler.seed_modules` (task 2122); α's grep-guard scopes to the callback +
  liveness reads only.
- **M6 — `acquire_next` phase decomposition (β).** A `TickContext` dataclass built once + an
  ordered list of phase callables (`Hygiene`: park-eviction-drain → park-GC → stale-sweep →
  cooldown-GC; `Policy`: external-dep → starvation; `Selection`: pins → scored). Ordering becomes a
  list literal a unit test asserts. The three duplicated backfill loops and the five-dict stale-id
  sweep are **not** re-collapsed here — they consume M2 ε's single backfill helper and
  `StreakRegistry.gc` (task 2124) inside the `Backfill`/`Hygiene` phases. Split along real seams
  only; do not split the file for line count.
- **M1 — `proc_supervision.RestartPlan` + `execute()` (γ, δ).** The `RestartPlan` dataclass
  `{script: absolute Path, args, cwd: REQUIRED, target_unit, own_unit, on_failure_escalation:
  EscalationSpec|None, verify: FreshPidVerify|None}` and one `execute()` honoring the five
  invariants (§5 contract). γ builds it + the full contract-test matrix and converts
  `service_restart`'s two mechanisms to thin delegates (the "accepted gap" dies); δ converts
  DeterministicRunner's detached + blocking-verify paths to delegate. `execute()` imports M2's
  `inspect_systemd_unit` for the fresh-PID verify.
- **M2 — `DeployState` typed schema (ε, ζ).** ε: `deploy_state.py` — a pydantic sub-model
  (`from_metadata`/`to_metadata`, phase enum, persisted verify baseline, transition table) +
  `register_metadata_submodel('deploy_state', DeployState)` **from a `shared`-visible location so
  both processes populate W3's registry identically** (§5.2 seam note — prevents W3's warn-census
  from firing on every deploy write). ζ: DeterministicRunner writes phase+stamp atomically in one
  `update_task`; the harness deterministic sweep and resume path branch on the explicit phase; the
  `_is_stranded_deterministic_shape` stamp-archaeology (harness.py:396) dies; the `ever_escalated =
  bool(get_by_task(task_id))` inference (deterministic_runner.py, scoped by M2 β) is replaced by
  `state == failed_escalated` + that escalation's resolved record.
- **M3 — `BackgroundService`/`LifecycleRegistry` (η).** A `BackgroundService` (name, interval/
  backoff, one async `pass_fn`, bounded failure-log) + a `LifecycleRegistry` (declared-order
  `start_all()`; reverse `stop_all()` with per-service timeout + one `CancelledError` contract).
  The eleven triplets collapse to registrations; `run()`'s finally-ladder collapses to one
  `stop_all()`. `cli.py`'s `os._exit(137)` watchdog stays as last resort.
- **M4 — `TaskGroundTruth` resolver (θ1, θ2).** θ1: `task_ground_truth.py` —
  `derive_truth(tid) → TruthReport{db_status, live_claimant (W2 field, plan.lock fallback),
  branch_state ∈ {on_main(sha), exists_off_main, gone_with_merge_marker(sha), gone_no_marker},
  worktree_present, open_escalations}` + ONE `TruthReport → recovery-action` classification table;
  `branch_state` resolves `MergeProvenance.lookup` **first**, git `find_merge_marker` only as
  fallback; deterministic tasks' truth reads `DeployState.phase`. θ2: the seven sweeps become thin
  schedulers over `derive_truth`, registered via the LifecycleRegistry; per-sweep dedup counters/
  race guards subsumed by the resolver + M5's `is_actively_held` accessor.
- **Integration gate ι.** The B+H composition gate: a deterministic self-restart deploy driven
  through `RestartPlan.execute()` (own-unit self-restart cell + cross-unit blocking-verify cell),
  crashed mid-phase, recovered by `TaskGroundTruth` + `DeployState` to the correct named phase +
  recovery action; plus the RestartPlan `(own-unit × target_unit × script-shape)` matrix and the
  registry shutdown-order-with-wedging-service test green.

---

## 5. Contract (B + H §1)

### 5.1 `orchestrator/proc_supervision.py` — the restart seam (M1)

```
@dataclass(frozen=True)
class EscalationSpec:  ...            # what to file if a detached restart fails at fire time
@dataclass(frozen=True)
class FreshPidVerify:                 # blocking-restart verification
    baseline_active_enter_monotonic: int   # persisted BEFORE the restart
    baseline_main_pid: int
    inspect_timeout_secs: float

@dataclass(frozen=True)
class RestartPlan:
    script: Path                     # absolutized at construction; must exist
    args: list[str]
    cwd: Path                        # REQUIRED (no implicit cwd — the 2105 fix, structural)
    target_unit: str
    own_unit: str | None             # None/'' → unknown (ORCH_UNIT unset)
    on_failure_escalation: EscalationSpec | None
    verify: FreshPidVerify | None
    async def execute(self, *, runner=..., inspector=...) -> RestartOutcome: ...
```

**Invariants (→ boundary tests make them executable):**
- **RP-1 (fail-closed on unknown own-unit).** If `own_unit` is falsy AND `target_unit` could be
  self, `execute()` MUST refuse and file `on_failure_escalation` — it MUST NOT take the blocking
  self-restart path. (The 2064 cell: `ORCH_UNIT` unset must never self-kill.)
- **RP-2 (detached-vs-blocking is derived, not a knob).** `target_unit == own_unit` (both truthy) →
  detached `systemd-run --user` self-restart (done = scheduled). `target_unit != own_unit` →
  blocking subprocess + fresh-PID verify (done = deployed-and-verified). Matches CLAUDE.md.
- **RP-3 (always `--working-directory` + absolute script).** Every detached payload passes
  `--working-directory=<cwd>` and an absolute script path. (The 2105 cell.)
- **RP-4 (always the on-failure escalation wrapper).** Every detached payload is wrapped with the
  `/bin/sh -c … || <escalation-submit>` shell — the `schedule_detached_systemd_restart` "accepted
  gap" is closed for all callers.
- **RP-5 (fresh-PID verify against a persisted baseline).** A blocking restart verifies a
  strictly-later `ActiveEnterTimestampMonotonic` AND a fresh non-sentinel `MainPID > 0` via
  `inspect_systemd_unit`; failure escalates loudly. The baseline is a **field**, not a local (the
  2.1 CAVEAT that made retroactive verification impossible).

`service_restart.schedule_detached_systemd_restart` / `_default_restart_executor` and
DeterministicRunner's `_default_schedule_detached_restart` / blocking cross-unit run become thin
callers that build a `RestartPlan` and `await execute()`.

### 5.2 `orchestrator/deploy_state.py` — the deploy lifecycle (M2)

```
class DeployPhase(StrEnum):  scheduled | ran | verified | failed | escalated | done
class DeployState(BaseModel):                 # metadata.deploy_state — a W3 sub-model
    phase: DeployPhase
    verify_baseline: VerifyBaseline | None    # persisted ActiveEnterTimestampMonotonic + MainPID
    ran_at / verified_at / escalated_at: str | None
    def from_metadata(md) -> DeployState; def to_metadata(self) -> dict
_LEGAL: dict[(DeployPhase, DeployPhase), bool]  # transition table; illegal → loud escalation
register_metadata_submodel('deploy_state', DeployState)   # §5.2 seam note below
```

- **DS-1 (single write).** DeterministicRunner writes `{phase, stamp, baseline}` atomically in one
  `update_task(metadata_mode='merge')`; no independent stamp keys drive state.
- **DS-2 (enforced transitions).** Every write asserts `_LEGAL[(old, new)]`; an illegal transition
  files a loud escalation, never silently proceeds.
- **DS-3 (persisted baseline).** `verify_baseline` is written at the `scheduled→ran` edge so
  freshness verification (baseline vs live monotonic) is possible retroactively.
- **DS-4 (recovery reads phase, not stamps).** The harness sweep and resume path read
  `DeployState.phase`; `_is_stranded_deterministic_shape` stamp-combination archaeology is deleted.
- **§5.2 seam note (registry visibility).** W3's `_SUBMODEL_REGISTRY` is a per-process module
  global. If `deploy_state` is registered only in the orchestrator process, fused-memory's
  `parse_metadata` treats it as an unregistered non-`x_` key and emits a
  `task_metadata.schema_warning` on **every** deploy write — which would prevent W3's enforce-flip
  gate (θ2 = task 2184; requires zero warnings over a recon cycle) from ever flipping. Therefore
  the `register_metadata_submodel('deploy_state', DeployState)` call is placed **`shared`-visible**
  so both processes populate the registry identically (the single sanctioned `shared/` touch — one
  registration, not a discriminated-union arm; W3 decision #4's intent is preserved). The
  orchestrator-domain state-machine (transition table + runner wiring) stays in `deploy_state.py`.

### 5.3 `orchestrator/background_service.py` — the lifecycle seam (M3)

```
@dataclass
class BackgroundService:
    name: str; pass_fn: Callable[[], Awaitable]; interval_secs: float
    backoff: BackoffPolicy; stop_timeout_secs: float; max_failure_logs: int
class LifecycleRegistry:
    def register(self, svc: BackgroundService) -> None            # declared order
    async def start_all(self) -> None                             # in order
    async def stop_all(self) -> None                              # reverse, per-service timeout
```

- **LR-1 (uniform CancelledError contract).** A `pass_fn` that raises `CancelledError` propagates
  (never swallowed); a `pass_fn` raising anything else is logged (bounded) and the loop backs off —
  one contract, not eleven per-copy variants.
- **LR-2 (bounded reverse stop).** `stop_all()` cancels in reverse registration order; a service
  that will not stop within `stop_timeout_secs` is abandoned with a WARNING and stop proceeds — one
  wedged service can never hang the ladder (the shutdown-hang class).
- **LR-3 (order is data).** Start/stop order is the registration list; a test asserts it.

### 5.4 `orchestrator/task_ground_truth.py` — the ground-truth seam (M4)

```
@dataclass(frozen=True)
class TruthReport:
    db_status: str
    live_claimant: Claimant | None      # W2 claimant_run_id/heartbeat_at; plan.lock fallback
    branch_state: BranchState           # on_main(sha) | exists_off_main | gone_with_merge_marker(sha) | gone_no_marker
    worktree_present: bool
    open_escalations: list[EscalationRef]
    deploy_phase: DeployPhase | None    # for deterministic tasks (from DeployState)
class TaskGroundTruth:
    async def derive_truth(self, tid: str) -> TruthReport: ...
_RECOVERY: table[TruthReport-shape → RecoveryAction]   # mark_done_with_provenance | revert_to_pending | re_file_escalation | leave
```

- **TG-1 (journal-first branch state).** `branch_state`'s merged-sha comes from
  `MergeProvenance.lookup(tid)` first; `git_ops.find_merge_marker` is consulted **only** when the
  journal has no row (fallback retained until the journal is proven populated fleet-wide — §9 Q3).
- **TG-2 (one classification table).** `_RECOVERY` is the single `TruthReport → action` map; the
  seven sweeps read it, none re-derives recovery policy. The table is a **recovery-action** table,
  distinct from W2's `(from,to,actor)` status-**legality** table — recovery writes go through the
  normal fused-memory chokepoint where W2's table validates them (§6 G4).
- **TG-3 (liveness via the accessor).** `live_claimant` and "actively held" use M5's public
  `is_actively_held`/`is_dispatched`, not scheduler privates.

### 5.5 `scheduler.py` — the callback + tick seams (M5, M6)

```
@dataclass(frozen=True)
class SchedulerCallbacks:               # all nine _on_* hooks, injected at __init__
    on_park_stop_trip; on_external_dep_block; on_starvation_warn; on_starvation_resolve
    warm_base_health_probe; on_warm_base_warn; on_warm_base_promote_l2; on_warm_base_resolve
    suppress_blocked_write
class Scheduler:
    def __init__(self, config, *, callbacks: SchedulerCallbacks | None = None, ...): ...
    def is_dispatched(self, tid) -> bool
    def is_actively_held(self, tid) -> bool      # includes cancel-grace, computed where state lives
    def requeue_history(self, tid) -> tuple
    def finish_startup(self) -> None             # sets started; acquire_next asserts started
```

- **SC-1 (no post-hoc mutation).** Callbacks are constructor state; there is no half-wired window.
  The `_SchedulerLike` protocol (workflow.py:212) is retired in favour of this one seam.
- **SC-2 (single-writer liveness).** `_dispatched`/`lock_table._held`/`_requeue_history` are read
  ONLY through the public accessors; grep-guard forbids new reach-ins from harness. (Excludes
  `_module_cache` → M2 δ / task 2122.)
- **SC-3 (startup ordering is a runtime check).** `acquire_next` asserts `started` — the
  "sweeps run before the first tick" invariant is enforced, not commented.
- **TK-1 (tick is an ordered phase list).** `acquire_next` builds one `TickContext` and runs an
  ordered list of phase callables; a unit test asserts the order literal (park-eviction-drain
  **before** park-GC; cooldown-GC **before** both candidate loops; external-dep policy exactly-once
  per tick). Behaviour parity: the existing tick test suite passes unchanged.

---

## 6. Cross-PRD relationship (G4) — per the authoritative program seam map

| Seam / artifact | Direction | Owner | W10 status |
|---|---|---|---|
| `inspect_systemd_unit(unit,*,timeout_secs)` | W10 **consumes** | **M2** (task 2119) | γ imports/relocates — never a second copy |
| `DeployState` deterministic-deploy phase enum + persisted baseline | **W10 owns** | **W10** (ε/ζ) | M2 must NOT introduce a deploy-state enum (β = query-scoping only); the scheduler-reviewer `metadata.deterministic_state` proposal is MERGED into this — one mechanism |
| `register_metadata_submodel` extension point | W10 **registers into** | **W3** (task 2158) | ε registers `DeployState`; registration is `shared`-visible so both processes' registry agree (§5.2) — coordinated with W3 so the θ2 enforce-gate (task 2184) is not tripped by deploy-write warnings |
| `EscalationQueue.get_by_task(agent_role=…)` scoped query | W10 **consumes** | **M2** (task 2120) | ζ builds the explicit state on top of the scoped query; `ever_escalated` inference dies |
| `Scheduler.seed_modules` / `_module_cache` single-writer | boundary (do NOT cross) | **M2 δ** (task 2122) | α EXCLUDES the module-cache seam; grep-guard scoped to callbacks + liveness reads |
| collapsed backfill helper + `StreakRegistry.gc` | W10 **consumes** | **M2 ε** (task 2124) | β's Backfill/stale-sweep phases wrap them — β does not re-collapse |
| `claimant_run_id`/`heartbeat_at` + `is_stranded()` | W10 **consumes** | **W2** (task 2182) | θ1's `TruthReport.live_claimant` reads them (plan.lock fallback) |
| Task-status `(from,to,actor)` legality table | adjacent (no seam) | **W2** | θ2's `_RECOVERY` is a recovery-action table, NOT a legality table; recovery writes go through the chokepoint W2 validates |
| `MergeProvenance.lookup` landed-outbox journal | W10 **consumes** | **W1** (task 2153) | θ1 resolves journal-first; `find_merge_marker` demoted to fallback |
| `TerminalReport` / `WorkflowStateMachine` / harness `_last_block_*` side channel | W10 **defers** | **W9** (NOT filed) | W10 leaves the attr-read sites untouched; W9 owns replacing them when its batch is filed |
| `_SchedulerLike` protocol (workflow.py:212) | W10 **subsumes** | **W10** (α) | one owned `SchedulerCallbacks` seam replaces the parallel protocol-split idea |

**File-adjacency (no hard dep; the per-project module lock serializes, second-to-land rebases):**
`scheduler.py` is also edited by W2 ω2 (2191), W3 δ (2167), and M2 γ/δ/ε (2121/2122/2124);
`deterministic_runner.py` by W3 δ (2167) and M2 β (2120); `harness.py` by M2 (2119/2120/2124).
Only the genuine semantic deps above are wired; the rest are left to the module lock.

No reciprocal-ownership ambiguity: every W10 consumption edge is a clean "the other stream
produces, W10 consumes"; every W10-owned seam (DeployState, BackgroundService, TaskGroundTruth,
SchedulerCallbacks) is a pure producer with no filed downstream consumer in this program.

---

## 7. Boundary-test sketch (B + H §2) — the ι integration-gate signal

Each row faces both sides of a seam; postconditions assert through product read paths. Restarts/
crashes are simulated by injected fakes (fake `systemd-run` runner, fake `inspect_systemd_unit`, an
injected wedging `BackgroundService`, a fault point that stops a deploy after a phase write), not
real process kills.

| # | Scenario | Pre | Post |
|---|---|---|---|
| R1 | Own-unit self-restart cell | `own_unit==target_unit` (both set) | detached `systemd-run` with `--working-directory` + absolute script + `/bin/sh` on-failure wrapper; outcome `scheduled` (RP-2/3/4) |
| R2 | **2105 cwd cell** | `before_done.cwd` set, relative script | payload passes `--working-directory=<cwd>` and an absolute script; no exit-127 (RP-3) |
| R3 | **2064 self-kill cell** | `ORCH_UNIT` unset, `target_unit` could be self | `execute()` REFUSES + escalates; NO blocking self-restart taken (RP-1) |
| R4 | Cross-unit blocking-verify cell | `own_unit != target_unit`, restart succeeds | fresh-PID verify passes against the persisted baseline; outcome `deployed-and-verified` (RP-5) |
| R5 | Cross-unit verify FAILS | restart leaves stale MainPID / older monotonic | loud escalation; outcome not `done` (RP-5) |
| D1 | Deploy crash between `ran` and `verified` | fault after the `ran` phase write | at recovery: `DeployState.phase == ran` (a NAMED state) → `_RECOVERY` yields the defined action, not phantom-done (DS-1/DS-4) |
| D2 | Illegal deploy transition rejected | write `done` directly from `scheduled` | transition table escalates loudly; no silent write (DS-2) |
| D3 | Unrelated escalation no longer aliases | starvation-watchdog escalation present, no runner gate | task is NOT driven to `done` "resumed after human resolution"; resolution proof requires `phase==failed_escalated` + the runner's own resolved record (ζ + task 2120 scoped query) |
| S1 | Registry shutdown-order with a wedging service | one registered `BackgroundService` hangs in its `stop` | `stop_all()` abandons it after `stop_timeout_secs` with a WARNING and completes the reverse-order stop; no hang (LR-2) |
| S2 | CancelledError contract | a `pass_fn` raises `CancelledError` vs a plain exception | `CancelledError` propagates; the plain exception is bounded-logged + backs off (LR-1) |
| G1 | Sweep drives classified recovery | in-progress task, `heartbeat_at` older than ttl, no live claimant, branch gone with journal row on main | `derive_truth` → `on_main(sha)`+no-claimant → `_RECOVERY: mark_done_with_provenance(merged, sha)`; task `done`; journal-first (find_merge_marker NOT consulted) (TG-1/TG-2) |
| G2 | Journal miss → git fallback | same, but no journal row | `find_merge_marker` fallback resolves the merged sha; recovery identical (TG-1) |
| C1 | Callbacks are constructor state | `Scheduler(callbacks=…)` | the nine hooks fire; grep shows zero `scheduler._on_*`/`_dispatched`/`lock_table._held` reach-ins in harness.py (SC-1/SC-2) |
| C2 | Startup ordering enforced | call `acquire_next` before `finish_startup()` | raises (SC-3) |
| T1 | Tick phase order asserted | build the `TickContext` | the phase-order list literal matches the asserted order (park-eviction before park-GC; cooldown-GC before both loops; external-dep once); existing tick tests green (TK-1) |

The ι leaf's observable signal = **the D1+R3+R4+S1+G1 composition green in CI** (a deterministic
self-restart deploy through `RestartPlan.execute()`, crashed mid-phase, recovered by
`TaskGroundTruth`+`DeployState` to the correct named phase, with the restart matrix + shutdown-order
green). R1–R2 are γ's signal; T1 is β's; C1–C2 are α's; S1–S2 are η's; D1–D3 are ζ's; G1–G2 are
θ2's — each mechanism carries its own boundary artifact, ι ropes the high-stakes composition.

---

## 8. Out of scope

- **The `inspect_systemd_unit` helper itself** — M2 (task 2119); W10 imports it.
- **The task-status `(from,to,actor)` transition/legality table** — W2; W10's classification table
  is recovery-action only and writes through the W2-validated chokepoint.
- **The landed-outbox journal / `MergeProvenance` write-ahead wiring** — W1; W10 consumes the read
  API and demotes `find_merge_marker` to a fallback (does not delete it — §9 Q3).
- **`TerminalReport` / `WorkflowStateMachine` / the harness `_last_block_*` side channel** — W9
  (not filed); W10 leaves the attr-read sites untouched.
- **`Scheduler.seed_modules` / `_module_cache` single-writer + `module_charter.py`** — M2 δ (2122).
- **The collapsed backfill helper + `StreakRegistry` / `streaks.py`** — M2 ε (2124); β wraps them.
- **`EscalationQueue.get_by_task` `agent_role` filter** — M2 β (2120); ζ consumes it.
- **A per-stream deploy/activation capstone** — program-level (§9 Q1). W10's changes are dormant on
  main until an orchestrator restart; W10 files no deploy task.
- **Retuning any restart timeout, backoff interval, streak threshold, or sweep cadence** — all
  behaviour-preserving; a needed semantics change escalates rather than shipping silently.

---

## 9. Open questions (tactical — safe defaults taken; operator AFK)

1. **Deploy/activation capstone (decided: NONE in W10).** W10 refactors the restart machinery
   itself; deploying it via that same machinery mid-refactor is circular-risky, and per-stream
   fleet restarts thrash the six-unit fleet (W1 §6 precedent). **Default:** file no capstone;
   activation folds into the program-level fleet restart after wave-2 lands. The activating restart
   uses the current known-good path (CLAUDE.md deterministic conventions, 2064/2105 fixed) and then
   **verifies `RestartPlan` is the live path post-restart**. Revisit at program deploy planning.
2. **`DeployState` registration home (decided: `shared`-visible).** Registering `deploy_state`
   only orchestrator-side would make fused-memory's `parse_metadata` warn on every deploy write and
   block W3's θ2 enforce-flip (§5.2). **Default:** the single `register_metadata_submodel` call is
   placed where both processes import it (one sanctioned `shared/` touch), preserving W3 decision
   #4's intent (no discriminated-union edits). This is the one place W10 deviates from "register
   from W10's own package with zero `shared/` edits" — flagged for operator ratification; the
   default is the correctness-preserving choice. Decide-at: ε impl (coordinate with W3 β/2162).
3. **`find_merge_marker` retirement depth (decided: demote to fallback, don't delete).** Deleting
   the git archaeology before W1's journal is proven populated fleet-wide would strand any task
   whose merge predates the journal. **Default:** θ1 resolves journal-first, git-fallback; a
   follow-up removes the fallback once the journal is confirmed populated. Decide-at: post-W1
   soak.
4. **`TaskGroundTruth` sweep-migration granularity (decided: one θ2 task).** Migrating all seven
   sweeps is high-effort but mechanical once the resolver exists (each becomes a thin scheduler).
   **Default:** one `force_full_path` task; the implementer escalates to split if the harness diff
   proves too large. Decide-at: θ2 dispatch.
5. **`BackgroundService` interval/backoff parity (tactical).** The eleven loops have per-copy
   intervals/backoffs today. **Default:** each registration carries its current values verbatim
   (parity); no retuning. Decide-at: η impl.
6. **`is_actively_held` cancel-grace home (decided: in Scheduler).** The cancel-grace window
   currently lives in harness's `_workflow_cancel_recent`; **default:** move it beside
   `_dispatched`/`_held` so the holding judgment is computed where the state lives (SC-2). Decide-at:
   α impl.

---

## 10. META check

> If I decompose and queue this PRD without further oversight, will the architecture of what gets
> implemented be complete, coherent, cohesive, and good?

**Yes.** Each of the six diffuse mechanisms gets exactly one owner (`RestartPlan`, `DeployState`,
`LifecycleRegistry`, `TaskGroundTruth`, `SchedulerCallbacks`, the tick phase-list); every consumed
capability is a filed upstream task wired as a hard dep (2119/2158/2182/2153/2120/2124); every
W10-owned seam is a pure producer with no reciprocal-ownership ambiguity; the one genuine
cross-stream hazard the gates surfaced (DeployState registry visibility vs W3's enforce-gate) is
resolved with a named coordination; the high-stakes restart/deploy/recovery composition lands as a
first-class integration gate (ι) rather than starving at medium priority; and the two behaviour-
preserving refactors (β tick, θ2 sweeps) assert parity against the existing suites. No open
**design** question remains — the six open items are tactical.

## 11. Note on tracking metadata

Per the prd skill: the orchestrator does **not** currently read the `user_observable_signal` /
`consumer_ref` / `substrate_confirmed` metadata fields these tasks carry — they are substrate for a
future tracking-infra session. The capability manifest beside this PRD
(`plans/harness-supervision-prd.capability-manifest.md`) is the artifact a dispatch-time architect
or downstream verifier diffs against substrate.
