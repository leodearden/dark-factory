# Capability manifest — harness-supervision (W10)

Mechanizes G3 + G6 for `plans/harness-supervision-prd.md`. One block per task; each asserted
capability → evidence. Line numbers are **current main, re-verified 2026-07-06** (HEAD
`82094587fe`). Check legend: `grep:<file>:<line>` = wired on main; `producer:task-N` = delivered by
an upstream task in this leaf's transitive dependency closure; `rejection-check:<X>` = the boundary
test authors X and observes the refusal/diagnostic fire. Any FAIL blocks the batch.

Leaves (no in-batch task depends on them): **β**, **ι**. All others are intermediates naming a
downstream consumer. Evidence is provided for every task (batch-wide G3/G6).

---

## α — SchedulerCallbacks seam (M5) — intermediate (β, γ, θ1 depend)

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| nine `scheduler._on_*` installs exist to replace | capability→site | `grep:harness.py:790` (park_stop_trip) … `:795,:802,:803,:808,:809,:810,:811`, `:826` (suppress_blocked_write) | PASS |
| `_dispatched`/`_held`/`_requeue_history` reach-ins exist to replace | capability→site | `grep:harness.py:2560,:2717,:2770,:2882` (`_dispatched`), `:2883` (`lock_table._held`), `:5836` (`_requeue_history`) | PASS |
| `Scheduler.__init__` is additive-injectable (takes no callbacks today) | substrate | `grep:scheduler.py:960` — signature has no `_on_*` params; all set post-construction | PASS |
| `_SchedulerLike` protocol to subsume | capability→site | `grep:workflow.py:212` `class _SchedulerLike(Protocol)` (consumed `workflow.py:669`) | PASS |
| `acquire_next`-before-`finish_startup()` raises (SC-3, negative) | rejection-check:`acquire_next()` pre-startup | test calls `acquire_next` before `finish_startup()`, observes the raised assertion (C2) | PASS |
| `_module_cache` NOT touched (boundary, M2 δ owns) | anti-collision | α grep-guard scoped to `_on_*`/`_dispatched`/`_held`/`_requeue_history`; `_module_cache` (`grep:harness.py:1939`) is `producer:task-2122` territory | PASS |

## β — acquire_next → TickContext + ordered phase list (M6) — **LEAF**

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| `acquire_next` 720-line tick to decompose | capability→site | `grep:scheduler.py:3305` … `:4027` (single method) | PASS |
| `TickContext` does not already exist | anti-dup | grep empty repo-wide → β introduces it | PASS |
| single collapsed backfill helper to wrap | capability→producer | `producer:task-2124` (M2 ε — collapses the three `acquire_next` backfill loops into one helper) — **upstream** (β `depends_on 2124`) | PASS |
| `StreakRegistry.gc(stale_ids)` to wrap in the stale-sweep phase | capability→producer | `producer:task-2124` (M2 ε — registry `gc` replaces the five-dict manual sweep) — **upstream** | PASS |
| DAG-direction | anti-inversion | 2124 is upstream of β (dep wired); NOT downstream | PASS |
| `_eligible_for_dispatch` selection predicate stays intact | substrate | `grep:scheduler.py:2764` (call sites `:3837`, `:3890`) | PASS |
| phase-order parity (behaviour-preserving) | example-in-CI | new test asserts the phase-order list literal on the **real** `acquire_next`; existing tick suite green | PASS |

## γ — proc_supervision.RestartPlan + execute() + service_restart delegate (M1) — intermediate (δ depends)

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| four restart mechanisms exist to unify | capability→site | `grep:service_restart.py:481` (`_default_restart_executor`), `:78` (`schedule_detached_systemd_restart`, gap docstring `:88-140`, no `--working-directory` argv `:141-149`); `grep:deterministic_runner.py:249` (`_default_schedule_detached_restart`), `:1267` (blocking run), `:1310-1347` (fresh-PID verify) | PASS |
| `inspect_systemd_unit(unit,*,timeout_secs,reap_grace_secs)` for the fresh-PID verify | capability→producer | `producer:task-2119` (M2 α — `orchestrator/systemd_inspect.py`, **not on main yet**, in-progress) — γ `depends_on 2119`; **upstream** | PASS |
| own-unit derivation exists (`ORCH_UNIT`) | substrate | `grep:deterministic_runner.py:240` (`_default_resolve_own_unit`), `:247` `os.environ.get('ORCH_UNIT','')`; fail-open at `:1073-1074` | PASS |
| **fail-closed on unset ORCH_UNIT (RP-1, negative — the 2064 cell)** | rejection-check:`execute()` w/ `own_unit=''` & self-target | contract test authors unset-ORCH_UNIT + self-target, observes `execute()` REFUSE + escalate (NO blocking self-restart) — R3 | PASS |
| `--working-directory` + absolute script always (RP-3 — 2105 cell) | example-in-CI | matrix cell R2 asserts the payload carries `--working-directory=<cwd>` + absolute script | PASS |
| service_restart consumer wired (anti-orphan) | capability→consumer | γ converts `service_restart.py:78`/`:481` to thin `RestartPlan` callers in the same task — named consumer on main | PASS |
| `proc_supervision.py` does not already exist | anti-dup | grep empty → γ introduces it | PASS |

## δ — DeterministicRunner delegates to RestartPlan (M1) — intermediate (ζ, ι depend)

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| `RestartPlan.execute()` to delegate to | capability→producer | `producer:task-γ` (in-batch, upstream; δ `depends_on γ`) | PASS |
| detached + blocking runner paths to convert | capability→site | `grep:deterministic_runner.py:249` (detached), `:1267`+`:1310-1347` (blocking+verify) | PASS |
| fresh-PID verify baseline becomes a persisted field (RP-5) | capability→site | current CAVEAT `grep:harness.py:337-345` (baseline unpersisted) — δ/ε make it a `RestartPlan.FreshPidVerify` field | PASS |

## ε — deploy_state.py DeployState + W3 registration (M2) — intermediate (ζ, θ1 depend)

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| `register_metadata_submodel(key, model)` extension point | capability→producer | `producer:task-2158` (W3 α — `shared/task_metadata.py`, in-progress) — ε `depends_on 2158`; **upstream** | PASS |
| four write-once stamps to replace with a phase enum | capability→site | `grep:deterministic_runner.py:658` (verified_at), `:820` (gate_escalated_at), `:1060` (ran_at), `:1145` (scheduled_at); `done_provenance` writes `:699,:902,:964,:993,:1023,:1164` | PASS |
| **illegal transition rejected loudly (DS-2, negative)** | rejection-check:`_LEGAL[(scheduled,done)]` | test writes an illegal phase edge, observes the loud escalation (never silent) — D2 | PASS |
| registration `shared`-visible so both processes' registry agree | seam-coordination | §5.2 — registration placed shared-visible; coordinated w/ W3 θ2 (task 2184) so deploy writes don't warn-census | PASS |
| `deploy_state.py` does not already exist; M2 introduces NO deploy-state enum | anti-dup | grep empty; program seam table: M2 β = query-scoping only | PASS |

## ζ — runner + harness write DeployState phase (M2) — intermediate (η via chain, ι depends)

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| `DeployState` schema + transition table | capability→producer | `producer:task-ε` (in-batch upstream) | PASS |
| agent_role-scoped `get_by_task` (for ever_escalated death) | capability→producer | `producer:task-2120` (M2 β — `EscalationQueue.get_by_task(agent_role=…)`, pending) — ζ `depends_on 2120`; **upstream** | PASS |
| `_is_stranded_deterministic_shape` stamp-archaeology to delete | capability→site | `grep:harness.py:396` (def), called `:7827,:7985` | PASS |
| **unrelated escalation no longer aliases as resolution (D3, negative)** | rejection-check:starvation-esc present, no runner gate | test presents a starvation-watchdog esc (`agent_role='orchestrator-starvation-watchdog'`), observes task NOT driven to done; proof requires `phase==failed_escalated` + runner's own resolved record | PASS |
| deterministic_runner.py serialize | DAG-direction | ζ `depends_on δ` (both edit deterministic_runner.py) — δ upstream | PASS |

## η — BackgroundService / LifecycleRegistry (M3) — intermediate (θ2, ι depend)

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| eleven `_start/_stop/_loop` triplets to collapse | capability→site | `grep:harness.py:4455/4477/4489, 4551/4576/4588, 6085/6182, 6278/6322, 6504/6693, 6702/6728/6737, 6902/6926/6935, 6951/6989/7065, 7386/7411/7420, 7434/7458/7467, 8051/8076/8085` | PASS |
| `run()` finally-ladder to collapse to one stop_all() | capability→site | `grep:harness.py:1125` (`run`), `:1634` (finally), stop calls `:1689-1729` | PASS |
| `cli.py` os._exit watchdog stays as last resort | substrate | `grep:cli.py:51` (arm), `:83` (`os._exit`), `:119` (daemon thread), `:230-234` (armed after `asyncio.run`) | PASS |
| **wedging service cannot hang stop_all() (LR-2, negative)** | rejection-check:injected wedging `BackgroundService` | test registers a service whose `stop` hangs, observes `stop_all()` abandon it after `stop_timeout_secs` + WARNING and complete — S1 | PASS |
| CancelledError contract (LR-1) | example-in-CI | test asserts CancelledError propagates; plain exception bounded-logged + backoff — S2 | PASS |
| `background_service.py` does not already exist | anti-dup | grep empty → η introduces it | PASS |
| harness.py serialize | DAG-direction | η `depends_on ζ` (harness.py chain α→γ→ζ→η) — ζ upstream | PASS |

## θ1 — TaskGroundTruth resolver + classification table (M4) — intermediate (θ2 depends)

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| `MergeProvenance.lookup(task_id)` journal for branch_state | capability→producer | `producer:task-2153` (W1 α — `MergeProvenance.lookup`, pending) — θ1 `depends_on 2153`; **upstream** | PASS |
| `claimant_run_id`/`heartbeat_at` + `is_stranded()` for live_claimant | capability→producer | `producer:task-2182` (W2 ρ2 — claimant cols + `is_stranded`, pending) — θ1 `depends_on 2182`; **upstream** | PASS |
| `DeployState.phase` for the deterministic branch | capability→producer | `producer:task-ε` (in-batch upstream; θ1 `depends_on ε`) | PASS |
| `is_actively_held`/`is_dispatched` public accessor for liveness | capability→producer | `producer:task-α` (in-batch upstream; θ1 `depends_on α`) | PASS |
| `find_merge_marker` git fallback exists | substrate | `grep:git_ops.py:4004` (`find_merge_marker`), harness fast-path `grep:harness.py:3229` | PASS |
| DAG-direction (all four producers upstream) | anti-inversion | 2153, 2182, ε, α all upstream of θ1 (deps wired); NONE depends on θ1 | PASS |
| `task_ground_truth.py` does not already exist | anti-dup | grep empty → θ1 introduces it | PASS |

## θ2 — migrate seven sweeps to thin schedulers over derive_truth (M4) — intermediate (ι depends)

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| seven sweeps to make thin | capability→site | `grep:harness.py:1948` (`_recover_crashed_tasks`), `:2795` (`_reconcile_stranded_in_progress`), `:2980` (`_reconcile_one_stranded`; is_ancestor `:3025`, find_merge_marker `:3229`, degenerate `:3071/:3180`), `:3415` (`_revert_in_progress_if_no_live_claimant`), `:7304` (`_scan_for_terminal_active_tasks`), `:7499` (`_run_main_tip_sweep`), `:7922` (`_run_deterministic_recon_sweep`) | PASS |
| `TaskGroundTruth.derive_truth` + `_RECOVERY` table | capability→producer | `producer:task-θ1` (in-batch upstream) | PASS |
| `LifecycleRegistry` to register the sweeps | capability→producer | `producer:task-η` (in-batch upstream; θ2 `depends_on η`) | PASS |
| `_RECONCILE_SWEEP_STATUSES` sweep set | substrate | `grep:harness.py:97` | PASS |
| classification table is recovery-action, not status-legality (no W2 collision) | seam-boundary | recovery writes go through the fused-memory chokepoint W2's `(from,to,actor)` table validates (§6 G4) | PASS |
| find_merge_marker demoted (not deleted) — journal-first, git-fallback | anti-false-premise | TG-1: retiring archaeology before the journal is populated fleet-wide would strand pre-journal merges → fallback retained (§9 Q3) | PASS |

## ι — B+H supervision composition integration gate — **LEAF**

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| `RestartPlan.execute()` (own-unit + cross-unit cells) | capability→producer | `producer:task-γ,δ` (upstream) | PASS |
| `DeployState` phase + runner writes | capability→producer | `producer:task-ε,ζ` (upstream) | PASS |
| `LifecycleRegistry` shutdown-order | capability→producer | `producer:task-η` (upstream) | PASS |
| `TaskGroundTruth` recovery of a crashed deploy | capability→producer | `producer:task-θ2` (upstream) | PASS |
| DAG-direction (all producers upstream of the gate) | anti-inversion | ι `depends_on β,δ,ζ,η,θ2` — every asserted capability is delivered by a prerequisite, none by a task depending on ι | PASS |
| composition is a real end-to-end (not synthetic) | example-in-CI | drives a deterministic self-restart deploy through `execute()`, crashes it mid-phase, asserts named-phase recovery — the 2064/2105/1900/2059/2066 incident chain as one green test | PASS |

---

### Batch verdict

All bindings **PASS**. Every novel-substrate assumption resolves to an upstream producer task in
the wired dependency closure (2119, 2158, 2182, 2153, 2120, 2124) or to a grep-confirmed site on
current main; every negative/rejection signal (RP-1 fail-closed, DS-2 illegal-transition, D3
alias-suppression, LR-2 wedging-service, SC-3 startup-assert) is bound to a `rejection-check` whose
owning task authors the trigger and observes the refusal fire; no numeric/floor/grammar/
field-population premises are asserted (infra domain — G6 branches 1/2/field-population N/A). No
FAIL binding → batch is queueable.
