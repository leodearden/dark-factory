# PRD: Dark-Factory verify oversubscription control — merge-prioritized, PSI-reactive pytest worker admission

**Date:** 2026-07-09 · **Status:** approved for **design-first** decomposition (B+H — queue the design lead task only; re-decompose implementation after it lands) · **Research substrate:** `plans/oversubscription-reify-to-df-research-2026-07-09.md` (authoritative — DF gap analysis + Reify design digest + anti-patterns; read it first). **Reify source of truth:** `/home/leo/src/reify/scripts/{jobserver-balancer.py,cpu-admit.sh,fleet-load-detector.sh,lib_test_semaphore.sh,lib_slot_acquire.sh,lib_cgroup.sh}`.

Cite by symbol; line refs are as-of `main` 20c934ca59 and drift.

## 1. Consumer + user-observable surface (G1)

- **Consumer (named, wired here):** the orchestrator's own **pytest verify subprocess** — both roles. Every task-lane and merge-lane verify runs pytest through `run_verification` / `run_scoped_verification` (`orchestrator/src/orchestrator/verify.py`); today each spawns `-n auto` = 32 xdist workers ungated (`orchestrator/pyproject.toml:100`). This PRD makes the orchestrator verify path acquire a **bounded, role-prioritized worker budget** before spawning xdist workers. The consumer is not a future PRD — it is the live verify path this PRD edits.
- **Secondary consumer (design option, not a blocker):** other DF-orchestrated projects (reify already has its own cargo jobserver; solar-challenge / autopilot-video verify). "Make generally available" (the extraction ask) means the primitives land in a **shared** location DF verify imports; cross-project adoption is a *follow-up*, not gated here.
- **User-observable surface:**
  1. Under high self-concurrency (many lanes), **total live pytest xdist workers across all lanes stays ≤ a host budget (≈nproc)** — observable as host load avg tracking ~nproc instead of 96–193, and as an emitted metric/log line reporting the live-worker count and per-role token split.
  2. **Merge-role verify is never starved by task-role verify:** under contention a merge verify acquires its worker budget ahead of task verifies (observable in the event log: a merge verify's workers admit while contending task verifies wait). This is the anti-livelock guarantee.
  3. **Fail-open:** with the jobserver/PSI substrate absent or unreadable, verify runs exactly as today (ungated `-n auto`) — no stall, no deadlock.

No frozen numeric threshold is asserted as the signal (G6): the signals are *direction + a recorded delta* (load bounded to ~nproc; worker count ≤ budget; merge-before-task ordering), matching the `concurrent-merge-verify-prd` convention.

## 2. Premise validation (G6) + substrate (G3)

Every premise is grounded in Reify's **already-working** production system (so these are ports, not guesses):

1. **A token-pool cap on the gated quantity bounds host load.** Reify caps compile *jobs* at `nproc` via a FIFO token pool and holds load stable; DF must cap the *xdist-worker* count analogously. *Achievability basis:* reify's `jobserver-balancer.py` in production. **Translation hazard (core design problem, routed to the design task):** reify's client is `rustc` (each build job is a jobserver client); DF's xdist workers are **not** jobserver clients, so the gated quantity must become the **worker count** — either derive `-n k` per lane from a token grant, or gate the controller to a role-scoped budget. (Research doc §"Port-critical translation notes".)
2. **Dual-pool merge>task priority prevents the task→merge livelock.** Reify's `decide()` ratchet (merge-demanded → move all task spare to merge; task give-back only when task fully starved) drifts monotonically to merge under contention. *Basis:* `jobserver-balancer.py:445-538` + `verify.sh:648-658`; asserted by `jobserver-acceptance.py::merge_ratchet_observed`. This is the mechanism satisfying the hard constraint.
3. **PSI hold-back throttles the task pool without touching merge.** Reify holds `min(free_task, headroom)` task tokens when `/proc/pressure/cpu` some-avg10 ≥ 50, releases < 40 (hysteresis), reservoir bounded by `nproc//4`, `suppress_giveback` protects merge. *Basis:* `jobserver-balancer.py:610-654`.
4. **Fail-open is a solved pattern here.** *Basis:* `shared/src/shared/pytest_jobserver.py:81-108` (plugin no-ops on missing FIFO/timeout), `shared/src/shared/psi.py:133-169` (`read_psi_sample` fail-open sentinel), the scheduler dispatch gate (`scheduler.py:4054-4064`).

**Substrate (G3) — verified to exist on `main` today:**
- `shared.psi`: `parse_pressure_file`, `read_pressure`, `PsiSample.saturated(cfg)`, `read_psi_sample` — `shared/src/shared/psi.py`; already consumed by the sampler and the scheduler gate. **REUSE (constraint 4).**
- `DF_VERIFY_ROLE ∈ {task, merge, offline}` stamped into the verify env by `_resolve_verify_env` (`verify.py:~2738`) — the role signal the priority scheme keys on. **Verified.**
- `pytest-jobserver.service` (running) + `/tmp/pytest-jobserver` FIFO (32 tokens) + client plugin `shared/src/shared/pytest_jobserver.py` — **exist but UNWIRED** (dropped from all conftests in `866db56dcf`) **and wrong granularity** (per-process, no `PYTEST_XDIST_WORKER` guard → 33 acquisitions/session deadlock trap). This PRD **replaces** the single-pool seeder and **reworks** the plugin; do not assume the current form is usable as-is.
- `cpu_governance` / `cpu-governed-exec.sh` cgroup weighting (merge 300 / task 100) — exists (`config.py:717`), reify-only default-off; **available** if the design adopts cgroup weight as a reinforcing priority layer.
- `reload_config` green-tier hot-tunability already covers `psi_admission` — the new config keys should join it.
- **NET-NEW (owned by this PRD, not assumed):** a DF pytest **dual-pool worker-admission balancer** (the reify `jobserver-balancer.py` analogue, retargeted to workers) + its systemd unit + canary. This is the core work, correctly queued — no substrate fiction.

G3 verdict: no unmet substrate assumption. The one novel component (the worker balancer) is explicit net-new work, not an assumed capability.

## 3. Approach — B + H (design-first; §Contract is the design task's deliverable)

This is a **load-bearing concurrency subsystem with a documented livelock hazard**, so it is authored **B+H**: a design-first lead task produces the contract + boundary-test sketch **before** any implementation task is decomposed. The core mechanism choice (per-lane static `-n k` grant vs. a live worker balancer) is genuinely open and is the design task's central decision — routing it to a first-class design task (rather than guessing here) is the correct handling for a B+H PRD.

**Hard constraints the design MUST satisfy (non-negotiable):**
- **C-merge-priority:** merge-role verify is never starved by task-role verify (implement reify's dual-pool + monotone ratchet, or an equivalent with a proven no-starve invariant). This is the anti-livelock requirement.
- **C-fail-open:** every gate no-ops to today's behaviour on missing/unreadable substrate — never deadlocks. Reuse `shared.psi` fail-open + the plugin's existing fail-open.
- **C-anti-patterns (all three, from the research doc):** (a) **no load-average-derived worker counts** (positive-feedback collapse to N=1 — `cargo-test-occt-gated.sh:114-121`); (b) **no FD inheritance** by descendant daemons pinning tokens/locks (the `9<&-` invariant — sccache-class wedge); (c) **no admit-on-timeout requeue** (resubmission storms — reify moved compile admission to a continuous never-requeue hold with clock-stop markers).
- **C-worker-granularity:** the gated quantity is the **xdist worker count**, not the pytest-process count (the current plugin's defect). Resolve per-worker-vs-per-process token semantics + a `PYTEST_XDIST_WORKER` guard so the 32 workers of one session do not each re-acquire.

### §Contract (skeleton — the design task fills every field)
- **Token/FIFO layout:** FIFO path(s), token = one xdist worker, total = f(nproc), dual pool `merge`/`task`, baseline partition (reify default: `task = max(1, nproc//4)`, `merge = nproc − task`).
- **Acquisition seam:** where in `verify.py` the verify acquires its role-scoped worker grant and how `-n <k>` is derived from it (replacing `-n auto`); how `shared/escalation/dashboard` segments (which declare no xdist) are handled (they must not receive `-n` — the same reason a uniform cap was rejected in `config.yaml`).
- **Priority invariant:** the exact `decide()`-equivalent rule + a stated no-starve proof obligation for merge.
- **PSI reactivity:** thresholds (default hold 50 / release 40 cpu some-avg10, mem full-avg10 10 — reify parity), what shrinks (task budget only), reservoir bound, all hot-tunable via `reload_config`.
- **Resilience:** canary sum-invariant + leak-recovery (tokens lost on orchestrator-restart SIGKILL), custodian FD-hold contract.
- **Packaging:** shell-in-place vs Python daemon vs **extract-to-shared-package** (the "generally available" ask) — decide and justify.

### §Boundary-test sketch (the integration-gate signal — G2/G5)
| Scenario | Precondition | Postcondition (both sides) |
|---|---|---|
| Global cap holds | N≫1 concurrent verifies dispatched | total live xdist workers across lanes ≤ budget (≈nproc); host load tracks ~nproc |
| Merge beats task | merge-role + task-role verify contend for the last tokens | merge acquires its budget first; task waits — **no task-starves-merge** |
| PSI shrink | `/proc/pressure/cpu` some-avg10 ≥ hold threshold | task budget shrinks; **merge budget unchanged** |
| Fail-open | FIFO/PSI absent | verify runs ungated `-n auto`, passes, no stall |
| Leak recovery | a verify's holder SIGKILLed mid-run | canary restores the sum invariant within its cycle |

## 4. Pre-conditions for activating

- None blocking. The structural verify fixes this defends against are already on main (tasks **2361/2365/2368** — do NOT re-file). The per-test load-hardening is filed separately (tickets `tkt_0RR2FNF…`, `tkt_0RR2FP9…`, `tkt_0RR2FPK…`) and is independent.
- Implementation tasks (T1+) are **blocked on the design lead task (T0)** landing.

## 5. Resolved design decisions

- **Reuse `shared.psi`, not a new parser.** (constraint 4; single source of truth.)
- **Key priority off `DF_VERIFY_ROLE`,** the existing merge/task/offline role env — do not invent a new role signal.
- **Replace, don't patch, the single-pool `pytest-jobserver.service`;** its per-process granularity is the wrong axis.
- **B+H / design-first:** the core mechanism decision is routed to T0 with a mandatory contract deliverable (above), not guessed in this PRD.
- **Complementary to L3b dispatch-admission (2326-2329), not a replacement:** dispatch-admission bounds *how many lanes start*; this bounds *workers per running verify* + enforces merge priority. Both compose (see §7).

## 6. Out of scope

- Re-doing tasks 2361/2365/2368 (done) or the filed per-test hardening tickets.
- Migrating reify itself onto the extracted shared primitive (follow-up if extraction is chosen).
- The scheduler dispatch-admission gate (owned by the L3b batch) — this PRD does not change it.
- cgroup weighting is *optional reinforcement*; the hard cap must come from the token/worker budget, not cgroup (cgroup is work-conserving, not a cap).

## 7. Cross-PRD relationship (G4)

| Other work | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| L3b host-PSI dispatch-admission (df 2326-2329) | complements | both read `/proc/pressure` via `shared.psi`; dispatch gate holds new lane dispatch, this gate bounds running verify workers + merge priority | **this PRD** owns verify-worker admission; L3b owns dispatch admission | compose — no contested ownership |
| Reify cargo jobserver (`reify-jobserver.service`) | pattern-source / potential shared consumer | `jobserver-balancer.py` design + `fleet-load-detector.sh` (built as the reify→DF seam) | **this PRD** owns the DF/pytest port; reify migration deferred | reference / optional-extract |

No reciprocal-ambiguity seam. G4 satisfied.

## 8. Decomposition plan (design-first; queue **T0 only** now)

**T0 — DESIGN LEAD (queue now).** *Design: DF pytest verify oversubscription control (dual-pool, merge-prioritized, PSI-reactive worker admission).* Produce a committed design doc `plans/verify-oversubscription-control-design.md` filling every §Contract field and the §Boundary-test sketch, resolving the core mechanism choice (per-lane `-n k` grant vs live balancer) and the worker-granularity/plugin semantics, honouring C-merge-priority / C-fail-open / C-anti-patterns / C-worker-granularity. **Signal (observable):** committed design doc whose §Contract and §Boundary-test tables are complete and whose merge no-starve invariant is stated with its proof obligation. (This is the H artifact and the gate for T1+.)

*Implementation tasks below are SPECIFIED BY T0 — re-run `/prd decompose` after T0 lands to file them with T0-informed detail. Listed here for shape/coverage only; each `depends_on: T0`.*

- **T1 — Worker balancer/custodian daemon** (dual FIFO, FIONREAD non-destructive sensing, `decide()` ratchet, PSI hold-back, `held_back` publication). Signal: daemon seeds N tokens split merge/task and the split moves per the ratchet under a synthetic demand harness (reify's `jobserver-acceptance.py` analogue).
- **T2 — systemd unit** replacing the single-pool `pytest-jobserver.service` (custodian FD-hold contract; `PartOf` the orchestrator unit for re-seed on restart). Signal: `systemctl --user` shows the dual-pool custodian; both FIFOs present with the baseline split.
- **T3 — Verify-side worker grant + `-n` derivation, role-keyed** wired into `verify.py` (acquire budget → derive `-n k`; merge role gets priority pool; no `-n` for xdist-less segments; fail-open). Signal: event log shows a task verify running with a bounded `-n k` drawn from the task pool; merge verify from the merge pool.
- **T4 — pytest plugin rework + conftest wiring:** `PYTEST_XDIST_WORKER` guard + token=worker semantics; restore registration in the subproject conftests (the `866db56dcf` regression, done correctly). Signal: under `-n k`, exactly the controller (not each of k workers) participates in admission; a missing FIFO runs unthrottled (fail-open test).
- **T5 — Canary + leak-recovery** (sum-invariant timer; restore tokens lost to SIGKILL). Signal: after a killed holder, the canary restores `sum == budget` within one cycle.
- **T6 — Config keys + `reload_config` green-tier wiring** (thresholds, pool sizes, enable flag; hot-tunable). Signal: a `reload_config` edit to the hold threshold takes effect on the running custodian without restart.
- **T7 — `fleet-load-detector` wiring (the DF seam) + escalation** on sustained host oversubscription. Signal: sustained load>threshold emits the oversubscription signal / files an escalation.
- **T8 — INTEGRATION GATE (leaf):** the §Boundary-test sketch as an executable scenario — N concurrent verifies ⇒ total live workers ≤ budget AND merge-role admits ahead of contending task-role AND fail-open path verified. Signal: the boundary-test scenarios pass (this is the G2 leaf; T1–T7 are its intermediates).
- **T9 (optional) — Extract primitives to a shared package** for cross-project reuse ("generally available"). Signal: DF verify imports the shared module; a second DF project can opt in via config.

## 9. Open questions (tactical — do not block save)

- Token total: exactly `nproc`, or `nproc − headroom` for the orchestrator/agent processes themselves? (T0 picks; reify uses `nproc`.)
- Whether merge priority needs the cgroup-weight reinforcement layer (reify 300/100) or the token partition alone suffices for pytest (no compile-job long-pole). (T0 decides; default: token partition first, cgroup optional.)
- Interaction with `_force_serial_pytest` env-recovery retry (`verify.py:3087`) — the serial retry should bypass the jobserver (it already sets `-p no:xdist`). (Tactical wiring detail.)

## META check

If decomposed **design-first** (T0 now; T1+ after T0 lands), the architecture is complete, coherent, cohesive, and good: the single open design question (core gating mechanism) is correctly routed to a first-class design task with a mandatory contract, the anti-livelock and anti-pattern constraints are stated as hard requirements, all substrate is verified, and the integration gate (T8) closes the G2 loop. **Do not** auto-queue T1–T9 before T0 lands — that is the design-first discipline this PRD is built on.
