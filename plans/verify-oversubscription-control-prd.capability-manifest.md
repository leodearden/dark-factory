# Capability manifest — verify-oversubscription-control PRD

Mechanizes G3 (assumed-substrate verified) + G6 (premise validity) per leaf. One
block per task binds each capability its user-observable signal asserts to
evidence. **Any FAIL binding blocks the batch.** Substrate anchors verified on
`main` (symbols cited; line numbers drift per the PRD's own caveat).

PRD: `plans/verify-oversubscription-control-prd.md` · batch: T1–T6 · decompose date: 2026-07-09.

Evidence vocabulary: `grep:<file>:<line> wired` (present on the production path on
main) · `producer:T-N upstream` (delivered by a task in the transitive dep
closure that is upstream of this leaf) · `producer:T-N self` (this task builds it)
· `substrate:stdlib|kernel` (language/OS primitive) · `identity:<clause>`
(exactness earned by construction) · `structural:<clause>` (bound enforced by a
mechanism, not a measured threshold) · `floor:<bound>` (numeric floor stated).

---

## T1 — `shared.verify_admission` module  *(intermediate; consumer T2; real-process signal roped into T4)*

Signal: N=1 serializes two acquirers; a SIGKILLed holder frees the slot; missing
dir → fail-open no-block; `merge` role never blocks; slot FD is non-inheritable.

| Capability asserted | Evidence | Verdict |
|---|---|---|
| `fcntl.flock` N-slot acquire | `substrate:stdlib fcntl.flock` | PASS |
| `os.set_inheritable(fd, False)` non-inheritable FD | `substrate:stdlib os.set_inheritable` | PASS |
| SIGKILLed holder frees slot (self-heal, no canary) | `substrate:kernel flock-released-on-process-exit` | PASS |
| `merge` role no-op pass-through (`held=False`) | `producer:T1 self` | PASS |
| fail-open no-op CM on OSError / missing dir | `producer:T1 self` | PASS |
| `nice_prefix(role)` tier table (merge5 / task15 / background19 / offline[]) | `producer:T1 self` | PASS |

DAG: root task, no upstream required. No numeric/exactness/rejection premise (N=1 is config; "serializes two acquirers" is a real-process behavior). **No FAIL.**

## T2 — Wire per-pytest acquire + nice into `verify.py`  *(intermediate; consumers T3/T4/T5)*

Signal: with admission on, a 2nd concurrent task-role verify's pytest waits until
the 1st releases; a merge verify starts immediately; with
`verify_admission_enabled=false`, the spawned command is byte-identical to today.

| Capability asserted | Evidence | Verdict |
|---|---|---|
| `acquire_task_slot` / `nice_prefix` | `producer:T1 upstream` | PASS |
| pytest spawn site to wrap = `_run_cmd` | `grep:orchestrator/src/orchestrator/verify.py:2346 wired` | PASS |
| role key `DF_VERIFY_ROLE` stamped by `_resolve_verify_env` | `grep:orchestrator/src/orchestrator/verify.py:2755 wired` | PASS |
| merge-govern sibling context `_maybe_govern_merge_cmd` | `grep:orchestrator/src/orchestrator/verify.py:2759 wired` | PASS |
| untimed acquire (before the pytest timeout clock) | `producer:T2 self (C-untimed-acquire)` | PASS |
| config knobs + `reload_config` green-tier | `grep:orchestrator/src/orchestrator/config.py:1772 Field-pattern wired` | PASS |
| **"byte-identical when disabled"** (exactness) | `identity:disabled-branch prepends nice_prefix=[] and skips acquire ⇒ argv unchanged` | PASS |

DAG: depends on T1 (upstream). **No FAIL.**

## T3 — `background` role for the sweep + fan-out coverage  *(intermediate; consumer T4)*

Signal: during a sweep, its subproject pytests run at nice 19 and a concurrent
task-lane verify acquires the slot between sweep subprojects (observable
interleave); total task+background live pytests ≤ N.

| Capability asserted | Evidence | Verdict |
|---|---|---|
| `background` nice tier (nice19 / ionice idle) | `producer:T1 upstream (nice_prefix)` | PASS |
| per-process acquire wrap (task/background) | `producer:T2 upstream` | PASS |
| sweep spawn site `_main_tip_sweep_loop` → `run_full_verification` | `grep:orchestrator/src/orchestrator/harness.py:7984 wired`, `grep:orchestrator/src/orchestrator/verify.py:3372 wired` | PASS |
| review-checkpoint pytest acquires as task-role | `grep:orchestrator/src/orchestrator/review_checkpoint.py:158 wired` | PASS |
| env-recovery retry (`_force_serial_pytest`) acquires as task-role | `grep:orchestrator/src/orchestrator/verify.py:773 wired` | PASS |
| **"≤ N live task+background pytests"** (bound) | `structural:bound = semaphore slot-count = config verify_admission_task_slots (T1 semaphore enforces)` | PASS |

DAG: depends on T2 → T1 (upstream). **No FAIL.**

## T4 — INTEGRATION GATE (the six §Boundary-test scenarios)  *(leaf — the G2 integration gate; consumer T6 + operator)*

Signal: all six boundary scenarios pass (global cap; merge-never-blocks; sweep
yields+interleaves; untimed-wait/no-requeue; self-heal; fail-open).

| Capability asserted (per scenario) | Evidence | Verdict |
|---|---|---|
| global cap | `producer:T1,T2 upstream` | PASS |
| merge-never-blocks | `producer:T1 (merge no-op),T2 (bypass) upstream` | PASS |
| sweep yields+interleaves | `producer:T3 upstream` | PASS |
| untimed-wait / no-requeue | `producer:T2 (C-untimed-acquire) upstream` | PASS |
| self-heal on holder death | `producer:T1 (kernel flock release) upstream` | PASS |
| fail-open | `producer:T1 (fail-open CM) upstream` | PASS |

DAG: depends on T2,T3; transitive closure T3→T2→T1 all upstream ⇒ every asserted
capability is producible from T4's own dependency set (G6 branch-3 clean). **No FAIL.**

## T5 — Cap merge internal fan-out  *(leaf; consumer = fleet-merge verify path / operator)*

Signal: a simulated root-`conftest` merge verify spawns ≤ cap concurrent pytests;
a scoped merge is unaffected.

| Capability asserted | Evidence | Verdict |
|---|---|---|
| internal-fanout bound substrate `_fanout_sem` ← `max_concurrent_module_verifies` | `grep:orchestrator/src/orchestrator/verify.py:3481 wired`, `grep:orchestrator/src/orchestrator/config.py:1772 wired` | PASS |
| **"≤ cap concurrent pytests"** (bound) | `floor:cap ≥ 1 (config, ge=1); bound = config cap, achievable by construction` | PASS |
| merge-role integration point | `producer:T2 upstream` | PASS |

DAG: depends on T2 (upstream). **No FAIL.**

## T6 — Benchmark-tuned N and default `-n`  *(leaf; follow-up; consumer = config defaults)*

Signal: a committed benchmark report with recommended
`verify_admission_task_slots` / default `-n` defaults.

| Capability asserted | Evidence | Verdict |
|---|---|---|
| committed benchmark report artifact | `producer:T6 self` | PASS |
| config knobs the report recommends values for | `producer:T2 upstream (verify_admission_* knobs)` | PASS |
| numeric premise at file-time | `n/a — the -n speedup curve + recommended defaults are the benchmark's OUTPUT, not an asserted floor` | PASS |

DAG: depends on T4 (integration gate green before tuning). **No FAIL.**

---

**Verdict: all bindings PASS — no FAIL, batch cleared to queue.** No binding
required rescoping/rehoming/relaxing. Every net-new capability is `producer:T-N
self`; every consumed capability resolves to an upstream producer or an on-main
substrate anchor.
