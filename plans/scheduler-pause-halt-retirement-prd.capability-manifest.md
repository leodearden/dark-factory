# Capability manifest — `plans/scheduler-pause-halt-retirement-prd.md`

Built at decompose, 2026-08-24, against main `4919b42b65` (the PRD amendment commit).
Mechanizes G3 (assumed substrate) + G6 (premise validity) per task: every capability each
task's user-observable signal asserts, bound to evidence. Any `FAIL` binding blocks the
batch. Machine-readable twin: `scheduler-pause-halt-retirement-prd.capability-manifest.yaml`.

**Evidence re-verified at decompose, not inherited from the PRD.** Three PRD §3 claims were
corrected here:

1. `CostStore.cost_totals_in_window` is defined in `shared/src/shared/cost_store.py`, not
   `harness.py` — harness holds the *call*, `harness.py::_enforce_cost_ceilings`. Capability
   confirmed; home corrected.
2. The `PauseClass` enum as authored covered 5 of the **7** live pause reasons.
   `watcher_misconfigured` and `watcher_crashloop` were unenumerated; ratified 2026-08-24 as
   a sixth class `watcher_guard` (PRD decision 9).
3. §6 invariant 5's structured-fields channel **did not exist**. `Escalation.evidence` is
   `list[EvidenceEntry]` = `{observation, measured_at, ref}`, all prose strings stored
   verbatim without shape validation. Resolved by assigning a typed sub-record to α
   (`TrainState` / `IndexHealthState` precedent), upstream of both consumers.

Task ι was **re-scoped** at decompose: its units-error half already landed via task 4559.

---

## α — pause-class taxonomy + policy table + `scheduler_state` migration

| Capability | Binding | Verdict |
|---|---|---|
| additive `scheduler_state` migration pattern | capability→producer (wired) — `RunStore._migrate_scheduler_state_ewa_value` defined `run_store.py:182`, invoked from `_ensure_schema` `run_store.py:121` | **PASS** |
| `scheduler_state` columns `pause_class`/`hold_owner`/`expires_at` | novel substrate, produced **by α itself**; grep confirms all three absent on main | **PASS** (producer = this task) |
| `PauseClass` enum + single policy table | novel substrate, produced by α | **PASS** (producer = this task) |
| typed structured-facts sub-record on `Escalation` | novel substrate, produced by α; house pattern `TrainState` / `IndexHealthState` in `escalation/src/escalation/models.py:280,289` | **PASS** (producer = this task) |
| 7 live pause reasons all classifiable | census at decompose: `ewa_trip_*` `harness.py:15455`; `park-stop: …` `scheduler.py:2383`; `cost_ceiling_watcher_exceeded`/`_orch_exceeded` `harness.py:9464,9479`; `watcher_misconfigured`/`watcher_crashloop` `harness.py:12620`; no-landings + operator via `force_halt_scheduler` `harness.py:7471,7355` | **PASS** |
| `get_scheduler_state` read surface (the signal) | capability→producer (wired) — `fused-memory/src/fused_memory/server/tools.py:9097`; harness state dict `harness.py:7375` | **PASS** |

## β — EWA trip escalates, never halts

| Capability | Binding | Verdict |
|---|---|---|
| the EWA halt actuator exists to delete | grep — `await self.pause_scheduler(f'ewa_trip_{new_ewa:.4f}')` live at `harness.py:15455` | **PASS** |
| 4559's restart re-check exists to delete as dead code | grep — `_load_persisted_scheduler_pause` `harness.py:15036`, `ewa_trip_` re-check `harness.py:15018,15105-15115` | **PASS** |
| `has_open_l1` dedup, category-scoped (INV-4 escape) | capability→producer (wired) — `escalation/src/escalation/queue.py:683`, `category` kwarg is a per-signature filter (task 2757) | **PASS** |
| `StormCounter` for the streak cap (PRD open question 1) | capability→producer (wired) — `shared/src/shared/storm_counter.py:44`, **already imported by harness** `harness.py:29` and in live use `harness.py:1649,6474`. **Open question 1 resolves to "reuse"**; the extraction has landed, so no local counter and no follow-up | **PASS** |
| structured trip facts on the escalation | producer:α upstream (typed sub-record) — see α | **PASS** |
| persisted `ewa_value` + digest split retained as forensics | grep — `ewa_value` column `run_store.py:70`, `RunStore.refresh_scheduler_pause_ewa` `run_store.py:461` | **PASS** |

## γ — park-stop trip escalates, never halts + cross-PRD correction pass

| Capability | Binding | Verdict |
|---|---|---|
| the park-stop halt actuator exists to delete | grep — `on_park_stop_trip=self.pause_scheduler` `harness.py:1556`; trip fires `scheduler.py:2383` (`_maybe_fire_park_stop_trip`) | **PASS** |
| api-error PRD decision 11 + boundary row 12 exist to retire | grep — `plans/server-side-api-error-handling-prd.md:112` (decision 11), `:194` (`park_stop_auto_resume` config), `:219` (boundary row 12) | **PASS** |
| task 3328 item 2 exists to drop | live read at decompose — 3328 `pending`, never dispatched; item 2 is verbatim `park_stop_auto_resume` per decision 11 | **PASS** |
| stranding PRD §4 out-of-scope line exists to amend | grep — `plans/stranding-remediation-scheduler-ergonomics-prd.md:88` | **PASS** |
| tasks 2890 / 2892 exist to amend | live read — 2890 `pending`/medium ("Resolution stays human-reserved — no auto-resume"); 2892 `pending`/low (annotation on the *pause reason string*) | **PASS** |
| task 4642 exists to cancel with a pointer | live read — 4642 `pending`/low; **both** its acceptable outcomes are about an `ewa_trip_` halt β deletes; no dependents | **PASS** |
| reify operators reachable as a named consumer | reify is a registered fleet project; park-stop fired there 10 of 12 times ever | **PASS** |

## δ — cost-ceiling self-clear

| Capability | Binding | Verdict |
|---|---|---|
| the `is_paused` early return exists to remove | grep — `_enforce_cost_ceilings` `harness.py:9417`; opens `if self.scheduler.is_paused: return` | **PASS** |
| trailing-24h spend query | capability→producer (wired) — `CostStore.cost_totals_in_window` **`shared/src/shared/cost_store.py:217`** (PRD said `harness.py`; that is the call site, `harness.py:9417` body) | **PASS** (home corrected) |
| `scheduler_resumed` event | capability→producer (wired) — `EventType.scheduler_resumed` `event_store.py:408` | **PASS** |
| INV-8 rate limit to the existing cost-check cadence | the cost check already runs per dispatch tick at `harness.py:2599`; δ reuses that cadence and adds no new scan | **PASS** |

## ε — operator holds carry owner + expiry

| Capability | Binding | Verdict |
|---|---|---|
| `force_halt_scheduler` exists to extend | grep — `harness.py:14790`, signature is `(self, reason: str)` — no owner, no expiry | **PASS** |
| `hold_owner` / `expires_at` columns | producer:α upstream | **PASS** |
| the requirement must NOT gate `no_landings` | **anti-inversion check** — the no-landings breaker calls the *same* `force_halt_scheduler` at `harness.py:7471`. Ratified: key on `pause_class` via the policy table, `pause_class` param defaults to `OPERATOR`. Boundary row 12 tests it | **PASS** (resolved at decompose) |
| expiry escalates rather than resumes | capability→producer (wired) — `_file_scheduler_pause_escalation` `harness.py:6779` | **PASS** |

## ζ — class-scoped resume + active-hold set

| Capability | Binding | Verdict |
|---|---|---|
| the reason-blind resume exists to fix | grep — `force_resume_scheduler` `harness.py:14758` calls `resume_scheduler()` unconditionally, never consulting `prior_reason` | **PASS** |
| the memory/disk divergence exists to fix | grep — `Scheduler.pause` is **first-wins in memory** (`scheduler.py:2281-2296`, "if already paused … keeping original") while `RunStore.save_scheduler_pause` (`run_store.py:429`) is last-write-wins on a `project_id` PRIMARY KEY row (`run_store.py:70`) | **PASS** |
| hold-set columns | producer:α upstream | **PASS** |

## η — 3876: merge-halt rehydration re-tests its predicate

| Capability | Binding | Verdict |
|---|---|---|
| the blind rehydration exists to fix | grep — `_rehydrate_merge_halt` `harness.py:11122-11150`: selects pending L≥1 `wip_conflict`/`unmerged_state`/`stash_failed`, then `halt_for_wip(reason)` + `set_halt_owner(esc.id)` with **no predicate re-test** | **PASS** |
| a clean/dirty `project_root` is testable | `git status --porcelain` / `--diff-filter=U`; task 3876 records both measured live on 2026-08-08 | **PASS** |
| independence from α | different subsystem (merge queue, not `scheduler_state`); needs no `pause_class` column | **PASS** (no DAG inversion) |

## θ — INV-7 backstop for the classes that keep a halt

| Capability | Binding | Verdict |
|---|---|---|
| hold age is computable | grep — `scheduler_state.pause_at` `run_store.py:70` (pre-existing, NOT NULL) | **PASS** |
| the three halting classes are identifiable | producer:α upstream — `cost_ceiling`, `no_landings`, `watcher_guard` | **PASS** |
| "exactly one" escalation per aged hold (INV-4) | θ inherits β/γ's escape: `has_open_l1` dedup `queue.py:683`. Called out explicitly because θ is itself a repeating detector | **PASS** |
| green-tier knob for the bound (open question 2) | capability→producer (wired) — `RELOADABLE_FIELDS` `config.py:5291+`; 4559 green-tiered the five `digest_*` fields, same mechanism | **PASS** |

## ι — config correction (docs) — RE-SCOPED

| Capability | Binding | Verdict |
|---|---|---|
| the false cold-start claim exists to correct | grep — `config.py:4185`: "reaching 24.6 from a cold start requires sustained high ratios across multiple digest steps" | **PASS** |
| the arithmetic refuting it | **numeric premise, re-derived at decompose**: `EWA(t+1) = α·ratio + (1−α)·EWA(t)`; cold start `EWA(t)=0`, `digest_ewa_alpha=0.3` ⇒ a single step trips at `ratio ≥ 24.6/0.3 = 82`. One step, not "multiple" | **PASS** |
| the units-error half | **ALREADY LANDED — do not re-file.** Task 4559 rewrote the provenance comment; `config.py:4159-4168` now states the recipe "describes a DAILY EWA over submissions only, while the runtime computes a per-N-lifecycle-event EWA … about 2.06x apart on DF data" | **N/A — dropped from scope** |
| "no longer gates a halt" | **producer-downstream if left on ι** — only true once β lands. **Re-homed to β**, which is already editing that comment block. This is the G6 branch-3 DAG-direction fix | **PASS** (re-homed) |
| no overlap with task 4632 | live read — 4632 documents the `digest_*` **reload tier** in OPERATIONS.md; ι corrects the **threshold's derivation** in `config.py`. Disjoint | **PASS** |

## κ — integration gate: the §7 boundary suite

| Capability | Binding | Verdict |
|---|---|---|
| all 13 boundary rows' mechanisms | producer:β,γ,δ,ε,ζ,η,θ — **all upstream** (κ depends on every one) | **PASS** (no DAG inversion) |
| genuinely concurrent holds of different classes | producer:ζ upstream (the hold set is what makes rows 8–9 expressible) | **PASS** |
| a harness able to run all rows in one suite | capability→producer (wired) — existing `orchestrator/tests/test_harness_park_stop.py`, `test_harness_watcher_supervisor.py`, `test_halt_owner.py` | **PASS** |

---

**No FAIL bindings. Batch is clear to queue.** Three resolutions were applied at decompose
to reach that state — the `watcher_guard` enumeration, the ε keying, and the α-owned
structured-facts sub-record — plus one re-scope (ι) and one re-homing (ι → β).
