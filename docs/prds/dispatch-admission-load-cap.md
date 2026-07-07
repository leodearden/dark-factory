# PRD — Dispatch-admission load cap (L3b, dark-factory)

**Status:** draft (2026-07-07) — for `/prd` check + decompose.
**Slug:** `dispatch-admission-load-cap`.
**Item:** L3b of reify PRD `docs/prds/run-all-pool-contention-tiering-fix.md` (§9), flagged there
as **dark-factory-owned, "not built here."**
**Depends (reify, LANDED):** `run-all-pool-contention-tiering-fix.md` **L0** — merge
`d2651f0d486a37a742a0949a39e2dda5d4d2d7ee` (2026-07-07). L0 moved the 103-test `run_all.sh`
infra pool to the single-flight **merge** tier and put per-task lanes on a cheap **selective**
subset — which **raises** the survivable concurrent-lane count M but does **not cap** it.
**Related (reify, deferred, different axis):** `cpu-load-admission-control.md` (cgroup `cpu.weight`
load governance of already-running work — see §4).

The dark-factory scheduler must **cap the number of concurrently-dispatched heavy
(Rust-compiling / verify-heavy) tasks by actual host load (PSI)** so a burst of ready tasks
cannot oversubscribe the build host — while staying **work-conserving** (never withhold dispatch
when the host is idle) and **deadlock-free** (never permanently starve dispatch). This is
"**govern dispatch by load**": a load-adaptive dispatch-admission gate, distinct from the
cgroup load-governance reify already owns.

---

## 0. Goal & user-observable surface (G1 — consumer named)

**Goal.** At the scheduler's dispatch decision, **defer starting a new heavy task while host load
(PSI) exceeds a threshold, resuming automatically when it drops** — bounding the concurrently-
dispatched heavy verifies to what the *host* can actually carry rather than to a fixed integer.

**The sole consumer is the DF scheduler's own dispatch decision** — the scored-candidate loop in
`orchestrator/src/orchestrator/scheduler.py::acquire_next` (`scheduler.py:4059`, the
`for … in scored:` loop that tries `lock_table.try_acquire` and, on success,
`self._dispatched.add(task_id)` + returns a `TaskAssignment`). There is **no orphan producer**: the
PSI reader and config knob exist only to feed this one gate; nothing else reads them.

**Mechanisms introduced, each with a named consumer:**

| Mechanism | Consumer |
|---|---|
| PSI reader (`/proc/pressure/{cpu,memory,io}` → parsed `some`/`full` avg10) | the dispatch gate (DA3) reads it each `acquire_next` tick |
| `PsiAdmissionConfig` submodel (enabled, per-metric thresholds, in-flight floor) + `RELOADABLE_FIELDS` registration | the dispatch gate reads thresholds live; the operator hot-tunes them via `reload_config` |
| Dispatch-admission gate in `acquire_next`'s scored loop + `EventType.dispatch_deferred` | the scheduler defers heavy dispatch; telemetry / operators observe the deferral + gating PSI value |

**User-observable surface when this lands:**
- Under a **burst of ready heavy tasks on a saturated host** (CPU-PSI `some avg10` high, and/or
  memory-PSI high, and/or io-PSI high), the count of concurrently-dispatched heavy verifies stays
  **bounded** — `acquire_next` **defers** new heavy dispatch (skips heavy candidates, emits a
  `dispatch_deferred` event carrying the gating metric + value + current in-flight count) instead of
  piling on. Visible in the event store / a rate-limited scheduler log line.
- **Work-conserving invariant (load-bearing):** when the host is **idle** (all gated PSI metrics
  below threshold), dispatch is **not** throttled — every eligible heavy task dispatches up to
  `max_concurrent_tasks`; no cores sit idle while ready tasks wait.
- **Progress/deadlock-freedom invariant:** even under *permanent* saturation (e.g. from sibling
  orchestrators or desktop apps), the scheduler **always keeps at least `min_inflight_floor` (≥1)
  heavy task in flight** — the cap defers *additional* dispatch, it never wedges the queue.
- **Deterministic (gate/deploy) tasks are exempt** — they hold no worktree/agent/build and add ~zero
  compile load, so they dispatch regardless of the PSI gate (a deterministic deploy still fires while
  heavy dispatch is deferred).

## 1. Premise (G6 — why the cap is needed, with cited evidence)

L0 (landed `d2651f0d`) **raises but does not cap the survivable M** (run-all PRD's own framing).
Under enough burst the shared 32-core host still oversubscribes:

- **run-all PRD §1 (esc-5029-42):** observed **M 20–88, load 168–618 on a 32-core host**; ~104
  subshells parked in `slot_acquire` rather than executing.
- **2026-07-07 /unblock session:** the host repeatedly hit **load 100–191 on 32 cores** with **three
  orchestrators** sharing it (reify + know-live + my-solar-challenge) plus desktop apps; even
  single-flight merge verifies were slow (~57 min on the slow "laptop" host, ~30 min on "local").
- **Host PSI at authoring time (2026-07-07, this 32-core host under light load):**
  `cpu some avg10=66.8`; `memory full avg10=0.95 / avg60=2.59` (i.e. **all-tasks memory stalls are
  already nonzero**); `io some avg10=6.4`. PSI is live and expressive on this host (Linux 6.14).

The premise is that a **fixed integer** cap (`max_concurrent_tasks`, enforced as
`asyncio.Semaphore(max_concurrent_tasks)` at `harness.py:1427`) **is a LANE cap, not a LOAD cap** —
it cannot adapt to per-task weight or to sibling/desktop load on a shared host, so it either
under-utilizes an idle host or oversubscribes a busy one. L3b gates on **actual** load. No leaf
asserts an impossible number: every leaf signal is **structural** (an event emitted / a dispatch
count bounded / a both-directions test passing / a reload disposition). The only numbers are the
PSI thresholds, which are **operator-tunable config**, not premises.

## 2. Ratified decisions

- **DA-D1 — Metric: PSI, ranked CPU → Memory → IO; OR-of-per-metric-thresholds.** Gate on
  `/proc/pressure` PSI, **not** process count and **not** loadavg alone (loadavg lags and counts
  uninterruptible sleep). Primary signal: **CPU `some avg10`**. **Memory PSI is gated and ranks
  ABOVE io PSI in practice** (operator directive, 2026-07-07): a Rust link/compile burst spikes RSS
  and can push the host into swap-thrash, which is catastrophic and escalates far faster than an
  io stall — so memory gets a **tighter threshold** (trips earlier) than io, and memory additionally
  supports a `full avg10` **hard-trip** (any sustained all-tasks memory stall defers). io `some avg10`
  is gated too, at a **looser** threshold (an io-bound host was over-committed by a full-fleet resume
  that loadavg-based gating missed — the io-overcommit playbook — so io must be *weighed*, just below
  memory). **Trip semantics: OR** — the gate holds if **any** enabled metric is at/over its threshold.
  Window: **avg10** (fast-reacting; mirrors reify `cpu-admit.sh`). Proposed tunable defaults (all
  green-tier, adjust post-observation): `cpu_some_avg10 ≥ 85`, `mem_some_avg10 ≥ 15`,
  `mem_full_avg10 ≥ 3`, `io_some_avg10 ≥ 40`. The **relative ordering** `mem_thr < io_thr` (memory
  trips earlier than io) is the load-bearing decision; the absolute numbers are tunable.
- **DA-D2 — Algorithm: hold-until-PSI-drops on the existing tick cadence, no timed admit.** The gate
  re-evaluates PSI **every `acquire_next` tick** and defers heavy dispatch while saturated; it
  **never admits on a timer** (mirrors reify `cpu-admit.sh` post-task-4920). No new background loop:
  when `acquire_next` returns `None` (or dispatches only deterministic/light work) with active tasks
  in flight, the harness main loop already re-polls within **≤15 s** (`asyncio.wait(active,
  timeout=15)` at `harness.py:1550`) — that retry **is** the heartbeat. Deferral is a plain skip in
  the scored loop; there is no queue mutation, no lock held, no state to unwind.
- **DA-D3 — Work-conserving + deadlock-free via a per-orchestrator in-flight floor.** `psi_hold` is
  true **only** when `enabled AND psi_saturated AND len(self._dispatched) >= min_inflight_floor`
  (`_dispatched` is the scheduler's own in-flight set, `scheduler.py:1029`). Consequences, all by
  construction: (a) **idle host ⇒ `psi_saturated` false ⇒ never hold** (work-conserving); (b) **0
  heavy in flight ⇒ `len < floor` ⇒ never hold** ⇒ the "hold with nothing running" state is
  **unreachable**, so the gate can never wedge the queue (deadlock-free); (c) the floor is measured
  against **this orchestrator's own** in-flight count (never a host-global count it cannot
  authoritatively read), so with N sibling orchestrators the worst case under permanent saturation is
  **N × floor** heavy tasks — bounded, survivable, and self-healing as siblings drain.
- **DA-D4 — Scope: gate heavy (normal-kind) dispatch only; exempt deterministic.** "Heavy" :=
  **normal-kind** task (runs the architect/implementer/verify pipeline that compiles). "Light" :=
  **deterministic-kind** (`Scheduler.is_deterministic(task)`, `scheduler.py:1495` — no worktree, no
  agent, no build; holds no module lock). The gate is applied **per candidate** inside the scored
  loop: `if psi_hold and not is_deterministic(task): continue` (skip → try next). Deterministic
  candidates are never skipped, so deploy/gate tasks keep flowing under load. This reuses the existing
  `is_deterministic` predicate — **no new task-heaviness classification** is invented (DF has no
  per-task weight metadata; in a Rust project every normal task compiles at verify, so normal-kind is
  the correct operational proxy for "heavy").
- **DA-D5 — Insertion is a per-candidate skip, not a top-level `return None`.** A top-level short-
  circuit would also stall exempt deterministic tasks and any light work behind a heavy head. The gate
  therefore lives **inside** the scored-candidate loop (`scheduler.py:4059`), skipping only heavy
  candidates; if the tick has an eligible deterministic candidate it still dispatches. The pinned-
  candidate loop (`scheduler.py:~3985`) applies the **same** rule (a pin does not make the host less
  saturated) — the floor guarantees a pinned heavy task dispatches as soon as in-flight drains below
  the floor, preserving priority **ordering** without granting oversubscription license.
- **DA-D6 — Fail-open on unreadable PSI (loud).** If `/proc/pressure/*` is absent or unparseable
  (non-Linux, cgroup-v1 host, read error), the reader returns a **not-saturated** sentinel and the
  gate degrades to today's behavior (dispatch normally) with a **loud, rate-limited warning**. A
  metrics-read failure must **never** wedge dispatch — fail-closed here would violate DA-D3's
  deadlock-freedom by starving the queue forever on a misconfigured host. (Consistent with the
  standing "loud escalation over silent degradation" norm: log loudly, but do not stall.)
- **DA-D7 — Ship enabled-by-default with conservative thresholds.** The host is *actively*
  oversubscribing (§1), and the gate is fail-open + work-conserving + floor-guaranteed (it provably
  cannot wedge), so the default is **`enabled: true`** with the DA-D1 conservative thresholds. **No
  separate flip/deploy task is needed** — unlike the offline-lane ε flips, there is no coverage-loss
  risk to stage behind a deterministic deploy. Thresholds are green-tier hot-tunable from tick one.
- **DA-D8 — Do not touch the merge queue's governor.** The merge queue runs its own single-flight
  merge tier + up-to-2-host verify concurrency with its **own** admission. The dispatch gate governs
  **task-lane dispatch only**; it never gates, halts, or delays the merge queue and never blocks
  in-flight work. Because PSI is host-global, merge-verify load is **already reflected** in the PSI
  the gate reads — so sibling merge verifies naturally back off *new task dispatch* **without**
  double-governing the merge queue and **without** a deadlock between the two (the gate defers only
  *future* dispatch; it can never block a merge from completing and freeing load).

## 3. Pre-conditions / substrate (G3 — all verified in-repo / on-host)

- **Insertion point exists.** `Scheduler.acquire_next` (`scheduler.py:3442`); the scored-candidate
  dispatch loop at `scheduler.py:4059`; the pinned loop at `scheduler.py:~3985`. Verified.
- **In-flight set exists.** `self._dispatched: set[str]` (`scheduler.py:1029`), `.add()` on dispatch
  (`scheduler.py:4062`), `.discard()` on release (`scheduler.py:4576`). This is the floor's counter.
- **Deterministic predicate exists.** `Scheduler.is_deterministic(task)` (`scheduler.py:1495`);
  deterministic tasks already acquire **no** module lock (`scheduler.py:4805`), so they dispatch
  freely today — the gate only needs to *not add* a hold for them.
- **Lane cap is a semaphore, not a load cap.** `asyncio.Semaphore(self.config.max_concurrent_tasks)`
  (`harness.py:1427`); `max_concurrent_tasks` default 24 (`defaults.yaml:5`, `config.yaml:4`). It is
  **restart-only (red tier)** — the L3b gate is **additive** to it and never changes its semantics.
- **Heartbeat cadence exists.** Harness main loop re-polls in-flight tasks within ≤15 s
  (`harness.py:1550`); `acquire_next` returning `None`/deferring rides this — **no new loop**.
- **PSI is live on the host.** `/proc/pressure/{cpu,memory,io}` all present and populated (Linux
  6.14, 32 cores). **DF has no PSI reader today** — grep of `orchestrator/ shared/ scripts/` finds
  none; the "PSI shim" in `workflow.py:7029` is reify's PATH-prepended `scripts/agent-bin` intercept
  (agent-side cargo), **not** a DF scheduler reader. So **the PSI reader is new code L3b builds**
  (DA1); no external prerequisite.
- **Config submodel + reload plumbing exists.** Submodels attach to `OrchestratorConfig` as
  `name: NameConfig = Field(default_factory=NameConfig)` (e.g. `fairness`, `starvation_watchdog`,
  `warm_base_hard_down` at `config.py:2202–2210`). `RELOADABLE_FIELDS` (`config.py:2621`) whitelists
  green-tier hot-reload leaves, whole submodels via `_submodel_leaf_paths('name', NameConfig)`
  (mirrors the `fairness.*` / `starvation_watchdog.*` entries). Adding
  `_submodel_leaf_paths('psi_admission', PsiAdmissionConfig)` makes every threshold green-tier.
- **Event type is extensible.** `EventType(StrEnum)` (`event_store.py:44`) — `dispatch_deferred` is a
  one-member addition alongside `lock_acquired` / `reservation_expired`.
- **fused-memory tagging (for decompose):** `project_id: "dark_factory"`,
  `project_root: "/home/leo/src/dark-factory"` — **confirm the exact `project_id` via `get_tasks` /
  an existing DF PRD's filed tasks at decompose time** before filing.

## 4. Cross-PRD / cross-repo seam (G4 — ownership)

| Other PRD / repo | Direction | Relationship |
|---|---|---|
| reify `run-all-pool-contention-tiering-fix.md` **L0** (LANDED `d2651f0d`) | L3b is its flagged companion | L0 raises M (per-task lanes cheap, full pool on merge tier) but **does not cap** it; L3b caps the burst M **at dispatch**. This is the "last mile" of L0. |
| reify `cpu-load-admission-control.md` (deferred) | **complementary, orthogonal axis** | Governs **LOAD via cgroup `cpu.weight`** on *already-running* work ("govern load not lanes"); it **explicitly excludes** the lane/dispatch cap. L3b is the **dispatch-admission** axis it leaves open. **Do not duplicate its cgroup mechanism.** Both are needed: cgroup governance slows running work; dispatch-admission prevents over-dispatch in the first place. |

**Ownership is unambiguous: L3b has no reify-side primitive** — it is a pure DF scheduler change, so
it is correctly a **standalone DF PRD** (not a reify spec). Note the *existing* DF↔reify cpu-governance
seam is already wired and is a **different** mechanism: `CpuGovernConfig` / `CpuPriorityConfig` /
`JobserverConfig` prepend `cpu-governed-exec.sh` (cgroup placement, DF-1), `nice` (DF-2), and jobserver
env onto **agent** invocations (`workflow.py:~7029`, `shared/cli_invoke.py:1125`). That governs the
CPU *weight* of running agents; L3b governs whether a **new** heavy task is *admitted* at all. No
overlap, no double-governance (DA-D8).

## 5. Decomposition plan (leaf tasks — each names a user-observable signal, G2)

- **DA1 — PSI reader primitive.** New `orchestrator/src/orchestrator/psi.py`: read
  `/proc/pressure/{cpu,memory,io}`, parse the `some` and `full` lines' `avg10` (and expose `avg60`
  for logging), return a small frozen dataclass (e.g. `PsiSample(cpu_some10, mem_some10, mem_full10,
  io_some10, …)`). Absent/unparseable file / read error ⇒ **fail-open sentinel** (`saturated()`
  returns False) + a caller-visible `read_ok=False` flag (DA-D6). *Signal:* a unit test parses a
  fixture `/proc/pressure/*` and returns the correct per-metric avg10; a malformed/absent-file
  fixture yields the fail-open sentinel with `read_ok=False`. *Modules:* `psi.py` (new). *Deps:* none.
- **DA2 — `PsiAdmissionConfig` submodel + green-tier reload registration.** New pydantic submodel
  (`enabled: bool = True`; `cpu_some_avg10`, `mem_some_avg10`, `mem_full_avg10`, `io_some_avg10`
  thresholds with the DA-D1 defaults; `min_inflight_floor: int = 1`, validated `>= 1`), attached as
  `psi_admission: PsiAdmissionConfig = Field(default_factory=PsiAdmissionConfig)` on
  `OrchestratorConfig`, and registered in `RELOADABLE_FIELDS` via
  `_submodel_leaf_paths('psi_admission', PsiAdmissionConfig)`. *Signal:* config loads with defaults;
  a test mirroring `test_config_reload_integration_gate.py` edits a threshold and asserts the reload
  disposition classifies it **`applied`** (green tier), **not** `restart_required`; a `floor < 1`
  config is rejected at load. *Modules:* `config.py`. *Deps:* none.
- **DA3 — dispatch-admission gate in `acquire_next` (load-bearing seam).** Wire DA1 + DA2 into the
  scored-candidate loop (and pinned loop) at `scheduler.py:4059`: once per tick compute
  `psi_hold = cfg.enabled AND psi.saturated(cfg-thresholds) AND len(self._dispatched) >=
  cfg.min_inflight_floor`; inside the loop `if psi_hold and not self.is_deterministic(task): continue`.
  On the first heavy skip of a tick, emit `EventType.dispatch_deferred` with `{metric, value,
  in_flight, floor}` and a **rate-limited** warning log (mirror `_last_paused_idle_log`). Add the
  `dispatch_deferred` `EventType` member. *Signal (this is the two-way G5 boundary — must prove BOTH
  directions):* (1) **throttle** — with injected saturated PSI and in-flight ≥ floor, a heavy
  candidate is deferred (event emitted, not dispatched) while a **deterministic** candidate in the
  same tick still dispatches; (2) **work-conserving** — with injected idle PSI, **all** heavy
  candidates dispatch up to `max_concurrent_tasks`, no deferral; (3) **floor/deadlock-freedom** —
  with injected saturated PSI and in-flight = 0, one heavy task still dispatches. *Modules:*
  `scheduler.py`, `event_store.py`. *Deps:* **DA1**, **DA2**.
- **DA4 — end-to-end scheduler boundary test (G5 "H" — two-way seam test).** A scheduler-level test
  driving `acquire_next` across a **saturation transition** (idle → saturated → idle) with a burst of
  ready heavy tasks + at least one deterministic task, against the real scored loop + lock table +
  event store (PSI reader injected via a seam). Asserts the observable **event stream** and **dispatch
  counts** match the invariants end-to-end: bounded heavy concurrency + `dispatch_deferred` events
  while saturated; deterministic dispatch never deferred; full dispatch restored (no residual hold)
  once PSI drops; ≥ `min_inflight_floor` heavy always in flight throughout. *Signal:* the transition
  scenarios pass against the live scheduler. *Modules:* integration test (no new production module).
  *Deps:* **DA3**.

**Suggested edges:** DA3 → {DA1, DA2}; DA4 → DA3. **No cross-project deps** (reify L0 is already
landed; L3b consumes no reify primitive). **No deploy/flip leaf** (DA-D7: enabled by default).

## 6. Invariants / do-nots

- **Work-conserving:** an idle host is **never** throttled — the gate holds only when a PSI metric is
  at/over threshold.
- **Deadlock-free:** the gate **never** holds with fewer than `min_inflight_floor` (≥1) heavy tasks in
  flight; the "hold with nothing running" state is unreachable by construction (DA-D3).
- **Gate LOAD, not lanes:** never re-key the cap off a fixed integer; PSI is the signal. Do not change
  `max_concurrent_tasks` semantics — L3b is additive to the existing semaphore.
- **PSI, not loadavg / process count:** loadavg lags and counts uninterruptible sleep; the reader uses
  `/proc/pressure`.
- **Memory ranks above io** (DA-D1): memory threshold is tighter than io, plus a memory `full`
  hard-trip; never let an io-only gate mask a memory-pressure host.
- **Fail-open on unreadable PSI** (DA-D6): degrade to today's dispatch behavior with a loud log; never
  wedge the queue on a metrics-read failure.
- **Exempt deterministic tasks** (DA-D4): deploy/gate tasks dispatch under load.
- **Never touch the merge queue** (DA-D8): the gate governs task-lane dispatch only; no halt, no
  delay, no double-governance of merge verifies.
- **Per-orchestrator floor** (DA-D3): the floor is measured against this orchestrator's own in-flight
  count, never a host-global count — this is what makes simultaneous multi-tenant stall impossible.

## 7. Out of scope

- cgroup `cpu.weight` load governance of running work (reify `cpu-load-admission-control.md`, deferred).
- The reify-side run_all tiering (L0 — landed) and its Phase-1 worker pool (L1/L2).
- Changing `max_concurrent_tasks` semantics beyond adding the additive PSI-adaptive gate.
- Cross-host / distributed load coordination (PSI is read per-host; each orchestrator gates on its own
  host's PSI).
- Per-task weight metadata / a richer "heaviness" classifier (normal-vs-deterministic is the proxy).

## 8. Open (tactical) questions

- **Threshold defaults (DA-D1):** the proposed `cpu 85 / mem_some 15 / mem_full 3 / io 40` are
  starting points — tune after observing PSI vs. actual oversubscription on the live 32-core host
  (all green-tier, so tunable without restart). The **ordering** `mem_thr < io_thr` is fixed by
  directive; only the numbers are open.
- **`min_inflight_floor` default:** 1 (strict progress guarantee, most conservative on the host) vs a
  small N>1 to keep this orchestrator productive under sustained sibling load. Ship 1; raise per
  operator observation.
- **avg10 vs a blend:** avg10 alone (fast, may be twitchy at the threshold edge) vs requiring
  avg10-over-threshold for K consecutive ticks (hysteresis to damp flapping). Ship avg10-only;
  add hysteresis only if edge-flapping is observed. (Note: the ≤15 s tick already coarsens reaction.)
- **`dispatch_deferred` event verbosity:** per-tick first-skip only (proposed, rate-limited) vs
  per-skipped-candidate. Ship per-tick-first-skip.
- **Recovery jitter:** whether to add a small per-orchestrator jitter to the re-evaluation so N
  siblings do not release in lockstep on PSI recovery. The per-orchestrator floor already bounds the
  post-recovery surge to N×(newly-admitted); add jitter only if a measured thundering-herd appears.
