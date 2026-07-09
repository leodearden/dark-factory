# PRD: Dark-Factory verify oversubscription control — task-verify admission semaphore + role nice tiers

**Date:** 2026-07-09 (design resolved in session; supersedes the 2026-07-09 design-first draft — mechanism now decided, T0 design-lead cancelled as task 2379) · **Status:** approved for decomposition · **Research substrate:** `plans/oversubscription-reify-to-df-research-2026-07-09.md`. **Reify source of truth:** `/home/leo/src/reify/scripts/{lib_test_semaphore.sh,lib_slot_acquire.sh,cpu-admit.sh,verify.sh}`.

Cite by symbol; line refs are as-of `main` d895106bf7 and drift.

## 1. Consumer + user-observable surface (G1)

- **Consumer (named, wired here):** every `pytest -n auto` spawn on the orchestrator's verify paths — task-lane verify, merge verify, main-tip sweep, review-checkpoint, and the env-recovery retry — all routed through `orchestrator/src/orchestrator/verify.py`. Today each spawns 32 xdist workers ungated (`orchestrator/pyproject.toml:100`); with `max_concurrent_tasks:24` plus a ~190-worker main-tip sweep, the box hits load 96–193 on 32 cores.
- **User-observable surface:**
  1. Under high self-concurrency, **total concurrent task-role `pytest -n auto` processes ≤ N (default 1)** — so live xdist workers track ~nproc (+ at most one merge verify) instead of 6× oversubscribing; host loadavg falls toward ~nproc.
  2. **Merge verify never waits on admission** and runs at the highest verify CPU priority; a merge and a task verify overlapping share the box by nice weight, not by starving each other (the anti-livelock guarantee).
  3. The **main-tip sweep** (heaviest fan-out, ~190 workers, zero waiters) runs at the **lowest** priority tier and its per-subproject pytests **interleave** with real lane verifies instead of monopolizing them.
  4. **Fail-open:** with the admission substrate absent/unavailable, verify runs exactly as today (ungated `-n auto`) — no stall, no requeue.

Signals are *direction + recorded delta* (bounded worker count; merge-never-blocks; sweep-yields), not frozen thresholds (G6).

## 2. Premise validation (G6) + substrate (G3)

Grounded in this session's measurements + reify's working system:

1. **`-n 32` is NOT ~2× `-n 16`, so serializing task verifies costs almost no throughput.** *Basis (measured this session):* the orchestrator suite is **7,774 tests, 20.7 s serial collection+import** — a fixed per-xdist-worker tax that does not shrink with more workers; with `--dist loadgroup` serial islands and I/O-bound tests, `-n 32 ≈ 1.3–1.6× -n 16`. Concurrency also *multiplies* the collection tax. ⇒ N=1 task-serialization is near-free and eliminates redundant-collection waste.
2. **Local merge concurrency = 1**, so worst case is 1 merge + N task = (N+1)×32 workers; N=1 ⇒ **2× max**. *Basis:* the merge host allocator is "one slot per host, prefer-local" (`merge_queue.py:4526`); overlapping merge verifies offload to other hosts.
3. **The main-tip sweep is fire-and-forget with zero synchronous waiters**, so throttling/interleaving it is free. *Basis:* `_main_tip_sweep_loop` is a background `asyncio.Task` (`harness.py:7966`); periodic `main_tip_sweep_interval_secs` default 1800 s, SHA-deduped; runs `run_full_verification` = all subprojects in **parallel** (`verify.py:3423`); its only effect is an L1 `infra_issue` escalation on drift. No dispatch/merge/deploy path `await`s it. (Same for `review_checkpoint` `run_full_verification` at `review_checkpoint.py:158` and the env-recovery retry at `verify.py:3957/3991`.)
4. **flock self-heals on holder death** ⇒ no daemon, no canary, no leak-recovery. *Basis:* an flock FD released by the kernel on process exit; reify's `lib_slot_acquire.sh` N-slot model.
5. **nice gives merge CPU priority without a hard cap; both fully utilize the box when alone.** *Basis:* reify `verify.sh` role nice (merge `nice 5` / task `nice 15`), work-conserving under contention.

**Substrate (G3) — verified on `main` today:**
- `DF_VERIFY_ROLE ∈ {task, merge, offline}` stamped by `_resolve_verify_env` (`verify.py:~2755`) — the key the tier + bypass logic reads. **Verified.** (This PRD adds a `background` role for the sweep.)
- The pytest command runner / `_run_cmd` in `verify.py` — the single spawn site to wrap. **Verified.**
- `_maybe_govern_merge_cmd` (`verify.py:2765`) already wraps merge-role verify in reify's `cpu-governed-exec.sh` cgroup weighting (default-off) — the priority layer is half-wired; nice is its simpler sibling. **Verified.**
- `reload_config` green-tier hot-tunability (per-role knobs) — new admission knobs join it. **Verified.**
- `fcntl.flock` (Python stdlib) + `os.set_inheritable` (for the `9<&-` non-inheritance invariant). **Verified.**
- **NET-NEW (owned here, not assumed):** a small `shared` admission module (Python flock N-slot semaphore + role nice/ionice prefix), the acquire-wrapper in `verify.py`, the `background` role. No daemon, no systemd unit. G3 verdict: no unmet substrate assumption.

## 3. Approach (design resolved) + §Contract

**Mechanism:** a **per-`pytest`-process** N-slot flock semaphore. Each `pytest -n auto` spawn, keyed by `DF_VERIFY_ROLE`, either **acquires a task-slot** (roles `task` and `background`) or **bypasses the count** (role `merge`). All spawns get a role **nice/ionice** prefix. Per-*process* (not per-verify) granularity is load-bearing: a single verify invocation (sweep / review-checkpoint / fleet-fallback) fans out to ~6 concurrent pytests internally, so only per-process acquisition bounds it — and acquire/release per pytest lets multi-subproject verifies **interleave** instead of monopolizing the slot.

**Hard constraints (locked this session):**
- **C-merge-priority / never-block:** merge-role pytests never acquire the counting semaphore and get the highest verify nice; task can never starve merge (anti-livelock).
- **C-untimed-acquire:** the slot wait happens **before** the pytest command's timeout clock starts (outside the timed region); it never counts toward `verify_command_timeout_secs` and never triggers a requeue (reify clock-stop equivalent).
- **C-fail-open:** missing lock dir / acquire error / disabled → run ungated `-n auto` (today's behavior), never block.
- **C-no-FD-inheritance:** slot FDs are `os.set_inheritable(fd, False)` so no pytest worker / sccache-class descendant pins a slot after exit (reify `9<&-`).
- **C-no-load-derived-count:** N and `-n` are fixed config, never derived from instantaneous loadavg (reify positive-feedback-collapse lesson).

### §Contract (seam signatures)
- `shared.verify_admission`:
  - `acquire_task_slot(role: str, *, slots_dir: Path, n: int, wait: bool = True) -> ContextManager[bool]` — flock one of `n` slot-files; `role == 'merge'` → no-op pass-through (returns immediately, `held=False`); non-inheritable FD; fail-open (returns a no-op CM that yields `False`) on any OSError / missing dir. Untimed by contract (caller acquires before starting the pytest timeout).
  - `nice_prefix(role: str) -> list[str]` — argv prefix: `merge → ['nice','-n','5']`; `task → ['nice','-n','15','ionice','-c2','-n7']`; `background → ['nice','-n','19','ionice','-c3']`; `offline`/unknown → `[]`.
- **Acquire seam in `verify.py`:** the pytest-command runner acquires `acquire_task_slot(role, …)` around the pytest subprocess, and prepends `nice_prefix(role)` to the command argv. Only pytest spawns are wrapped (lint/type ride the same held slot as they already run concurrently within one verify).
- **Config (green-tier, `reload_config`):** `verify_admission_enabled: bool = True`, `verify_admission_task_slots: int = 1`, `verify_admission_slots_dir: str` (default `/tmp/df-verify-slots-$(id -u)`), and the three nice/ionice tiers as tunables.

### §Boundary-test sketch (T-GATE signal — G2/G5)
| Scenario | Precondition | Postcondition |
|---|---|---|
| Global cap | M task verifies dispatched, N=1 | ≤1 task-role pytest runs at a time; the rest block on the slot |
| Merge never blocks | 1 merge + saturated task slot | merge pytest starts immediately (no acquire), runs at nice 5 |
| Sweep yields + interleaves | sweep (background) + a task verify contend | sweep pytests run at nice 19 and a task verify acquires between sweep subprojects (no full-duration monopoly) |
| Untimed wait | task verify waits > 0 on the slot | wait excluded from `verify_command_timeout_secs`; no requeue |
| Self-heal | slot holder SIGKILLed | slot frees immediately (no canary) |
| Fail-open | slots_dir absent / disabled | ungated `-n auto`, verify passes, no block |

## 4. Pre-conditions for activating

None blocking. Structural verify fixes already on main (2361/2365/2368). Per-test load-hardening filed separately. No dep on the L3b batch (2326-2329) — complementary.

## 5. Resolved design decisions (this session)

- **Semaphore-on-tasks + merge-bypass + nice**, NOT a jobserver token pool — reify's *test* mechanism (the jobserver gates cargo compile-jobs, which are jobserver clients; xdist workers are not). Rejected a custom `pytest_xdist_make_scheduler` (task 1907 proved it deadlocks).
- **Per-pytest-process** acquisition (not per-verify-invocation) — bounds internal fan-out (sweep/review-checkpoint/fleet-fallback) and lets multi-subproject verifies interleave.
- **Three nice tiers:** merge (5) > task (15) > background/sweep (19, ionice idle). Sweep = `background` role (lowest), because it is fire-and-forget with zero waiters and must never delay real lane verifies.
- **N=1 default** (config knob) — sublinear `-n` scaling makes serialization near-free; 2× worst-case bound.
- **`-n auto` unchanged in v1**; `-n`/N tuning deferred to a benchmarked follow-up (T6).
- **Daemonless flock** (self-healing) — no custodian, no canary.
- **PSI reactivity deferred to phase 2** (the static N + the live L3b dispatch gate already bound load).

## 6. Out of scope

- 2361/2365/2368 (done) and the filed per-test hardening tickets.
- PSI-reactive task-pool shrink (phase 2).
- cgroup weighting (the existing `_maybe_govern_merge_cmd` path stays as-is; nice is the v1 priority layer).
- Migrating reify onto the extracted `shared.verify_admission` (follow-up).
- Bounding **agent inner-loop** test runs (only verify paths are gated in v1).

## 7. Cross-PRD relationship (G4)

| Other work | Direction | Seam | Owner | Status |
|---|---|---|---|---|
| L3b host-PSI dispatch-admission (df 2326-2329) | complements | dispatch gate bounds new **lanes**; this bounds **verify workers** + priority | this PRD owns verify admission | compose — no contested ownership |
| Reify test-semaphore (`lib_test_semaphore.sh`) | pattern-source | flock N-slot + role nice | this PRD owns the DF port (reimplemented in `shared`) | reference |

No reciprocal ambiguity. G4 satisfied.

## 8. Decomposition plan (each leaf names its observable signal)

- **T1 — `shared.verify_admission` module (foundation, intermediate).** Python flock N-slot semaphore (`acquire_task_slot`, non-inheritable FD, fail-open, merge no-op) + `nice_prefix` role tiers, per §Contract. Signal (roped into T-GATE): real-process unit tests — N=1 serializes two acquirers; a SIGKILLed holder frees the slot; missing dir → fail-open no-block; `merge` role never blocks; FD is non-inheritable. Consumer: T2.
- **T2 — Wire per-pytest acquire + nice into `verify.py` (leaf).** Wrap every pytest spawn: task/background acquire (untimed, before the pytest timeout; fail-open), merge bypass; prepend `nice_prefix(role)`. Add the config knobs + `reload_config` green-tier entries. Signal: with admission on, a 2nd concurrent task-role verify's pytest waits until the 1st releases; a merge verify starts immediately; with `verify_admission_enabled=false`, the spawned command is byte-identical to today. Depends: T1.
- **T3 — `background` role for the sweep + fan-out coverage (leaf).** Add `DF_VERIFY_ROLE=background` (lowest nice tier); stamp it on main-tip-sweep pytests; confirm review-checkpoint + env-recovery-retry pytests acquire as task-role (inherited from T2's per-process wrap). Signal: during a sweep, its subproject pytests run at nice 19 and a concurrent task-lane verify acquires the slot between sweep subprojects (observable interleave); total task+background live pytests ≤ N. Depends: T2.
- **T4 — INTEGRATION GATE (leaf).** The §Boundary-test sketch as an executable scenario (global cap; merge-never-blocks; sweep yields+interleaves; untimed-wait/no-requeue; self-heal; fail-open). Signal: all six boundary scenarios pass. This is the G2 integration gate; T1–T3 are its intermediates. Depends: T2, T3.
- **T5 — Cap merge internal fan-out (leaf; later in DAG).** Bound the concurrent subproject pytests a single merge/fleet verify spawns (e.g. a small merge-role internal fanout cap reusing `max_concurrent_module_verifies`, or a merge slot pool) so a root-`conftest` fleet-merge can't spawn ~190 workers. Signal: a simulated root-conftest merge verify spawns ≤ cap concurrent pytests; a scoped merge is unaffected. Depends: T2.
- **T6 — Benchmark-tuned N and default `-n` (leaf; follow-up).** Clean idle-window benchmark of the `-n` speedup curve (isolating the 20.7 s collection tax) and single-verify core utilization; commit results + recommended `verify_admission_task_slots` / default `-n` (or a config change). Signal: a committed benchmark report with recommended defaults. Depends: T4.

## 9. Open questions (tactical)

- Exact `slots_dir` lifecycle (tmpfiles vs first-use mkdir) and whether to pre-seed N slot-files or create lazily. (T1 picks; fail-open covers absence.)
- Whether `background` should also be excluded from `-n auto` capping later (T6 territory).
- Interaction with `_force_serial_pytest` env-recovery retry (already `-p no:xdist`, single worker) — it should still acquire a task/background slot but its 1-worker load is negligible; confirm it doesn't double-hold. (Tactical.)

## META check

Complete/coherent/good under decompose: the mechanism is fully resolved (semaphore-on-tasks + three nice tiers, per-pytest granularity), every hard constraint is a named contract clause, all substrate is verified, the integration gate (T4) closes the G2 loop, and the two follow-ups the user asked for (merge fan-out cap = T5; `-n`/N tuning = T6) are explicit tasks later in the DAG. No open design questions remain.
