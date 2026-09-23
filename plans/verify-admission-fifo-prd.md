# Task-verify admission: arrival order, role classes, and a recorded wait

Status: active — authored 2026-09-18 (interactive `/prd`). Supersedes task **5413**
(Part A answered, Part B declined by Leo 2026-09-18). Evidence base:
`plans/verify-merge-queue-policy-study-2026-09-18/SYNTHESIS.md` (v2, post-critic) — untracked
in the main checkout; every number this PRD relies on is restated here.

## Goal

The single task-verify admission slot serves its waiters in **arrival order, task verifies
ahead of background sweeps**, instead of by a 0.1 s poll race; and every wait at that slot is
**recorded** — per plan when it completes, and as a live gauge while it is still waiting.

What an operator observes when this lands:

- For every task verify, one `verify_admission_plan` row in `runs.db` giving, per leg, when it
  arrived, when it was granted the slot, how long it held it, how deep the queue was, and
  which ordering policy granted it.
- While anything is waiting, a periodic `verify_admission_heartbeat` row (depth, oldest
  waiter, who holds the slot), and a single structured alarm when the oldest waiter's age
  crosses a threshold. A task that enters `phase=verify` and then shows "nothing for 3 hours"
  (incident reify 7414, 2026-09-11) is now legible as *waiting at admission, position k*.
- After the policy flip, no waiter is overtaken by a later arrival of the same role class.

## Background

`shared/src/shared/verify_admission.py::acquire_task_slot` is a flock N-slot semaphore whose
waiters each poll every 0.1 s; whoever polls first after a release wins. Measured
2026-09-13 → 09-18 on DF: among 2,801 contended pairs a later arrival was served before an
earlier waiter in 1,410 = **50.3 %** — uniform random. `verify_admission_task_slots` is 1 in
both projects, the slot is ~99 % utilised, in-flight tasks are pinned at
`max_concurrent_tasks`, and the median verify phase is 5.3 h, 97 % of it neither agent time
nor test service.

What ordering can and cannot buy (closed-loop simulation validated against measured
throughput, 75.1 vs 75.7 plans/day): with WIP pinned and the slot saturated, **mean** plan
completion is invariant under every ordering (Little's law). Arrival order is a **tail**
fix: predicted mean ±0, median ~+35 %, p90 −30 %, max 34 h → 12 h. It is worth doing for
the tail, for determinism, and because it removes incident-7414-class starvation. It is not
a throughput or mean-duration lever and this PRD does not claim one. Size-aware ordering
(short legs first) was evaluated and **declined**: it moves wait from plans under 300 s
(4.4 h → 0.5 h) onto the orchestrator full-suite leg (5.1 h → 9.0 h), which is 86 % of slot
work.

No admission wait is recorded anywhere today: `verify.py::run_verification`'s nested
`_run_or_skip_timed` stamps `started_at` inside `_admission_slot`, so every wait figure in
every study is a proxy (lint-start to test-start), and reify's is unobservable. That is also
an INV-7 `holds-owned-and-bounded` gap: an unbounded wait with no age surfacing.

## Sketch of approach

An in-process, asyncio-native **ordered gate in front of the unmodified flock**.

- All gated-role acquirers are coroutines of the one orchestrator process per project
  (verified: `acquire_task_slot` is imported by `orchestrator/verify.py` alone; `offline_lane.py`
  imports only `nice_prefix`). So ordering needs no cross-process queue. The flock stays as
  the cross-process safety net and keeps its four contract clauses (C-merge-priority,
  C-untimed-acquire, C-fail-open, C-no-FD-inheritance) and its tests untouched.
- Waiters queue inside the gate as awaitables. Only the waiter(s) the gate has admitted go
  on to poll the flock, so the 64-thread `_admission_executor` holds at most
  `verify_admission_task_slots` pollers instead of one per waiter (INV-8
  `loop-thread-occupancy-bounded`; removes the saturation class behind task 5424 for good).
- Facts about a completed wait travel **by return value**: the gate yields an immutable grant,
  `run_verification` attaches it to the test `CheckRun`, and
  `verify.py::run_scoped_verification` — which already takes an `event_store` and sees every
  leg's result — emits one `verify_admission_plan` event. The gate does no I/O.
- The live view is a pure `snapshot()`; a small harness-owned watch reads it on a timer and
  emits the heartbeat and the aged-waiter alarm.

## Contract

New module `orchestrator/src/orchestrator/verify_admission_gate.py` (verify.py is 9.6k lines —
heuristic 14, no file too large; the gate is a deep module with a narrow interface,
heuristic 9, and must read on its own, heuristic 13).

```
gate_for(slots_dir: Path) -> AdmissionGate          # process-wide registry keyed by the
                                                    # flock's own identity; a test gets a
                                                    # fresh gate from a tmp slots_dir

@dataclass(frozen=True) AdmissionRequest:
    role: str                  # 'task' | 'background' (gated roles only reach the gate)
    plan_id: str               # one per run_scoped_verification / run_full_verification call
    plan_arrival: float        # monotonic, stamped ONCE per plan, shared by all its legs
    task_id: str | None
    module: str | None
    scope_kind: str | None     # from verify_plan.PlannedRun — a structured field, never
                               # parsed out of the command string (heuristic 12)

@dataclass(frozen=True) AdmissionGrant:
    request, order, arrived_at, granted_at, waited_secs,
    depth_at_arrival, slot_held: bool     # False = flock failed open (INV-11)

AdmissionGate.admit(request, *, slots: int, order: Order, promote_after_secs: float)
    -> async context manager yielding AdmissionGrant
AdmissionGate.snapshot() -> AdmissionSnapshot       # frozen: depth, waiters (request, age),
                                                    # holders, oldest_wait_secs
```

Ordering key under `order='arrival'`: `(role_class, plan_arrival, leg_seq)`.

- `role_class`: task = 0, background = 1. The class table lives in the gate module and its
  key set is asserted equal to `shared.verify_admission`'s gated roles (heuristic 11 SPOT;
  INV-5) — `is_gated_role` stays the single authority for *which* roles are gated.
- `plan_arrival` is per **plan**, not per leg. `run_scoped_verification` caps a plan's
  concurrent modules with `_fanout_sem` (`max_concurrent_module_verifies`, 4), so a plan's
  5th module reaches the gate only after an earlier one finishes; a per-leg stamp would send
  it to the back. A retry inside one `run_verification` call reuses the plan's stamp. A new
  verify call after a debugger fix is a new plan with a new stamp — no task-level aging (a
  repeat failer must not own the head of the queue).
- **Background promotion**: a background waiter older than `promote_after_secs` is ranked as
  class 0 **with its original `plan_arrival`**. It can never overtake a task that arrived
  before it; it stops being overtaken by tasks that arrived after it. So background is
  never worse off than plain FIFO and the main-tip sweep cannot starve at ρ≈1.
- Under `order='unordered'` the gate picks uniformly at random among all waiters — today's
  behaviour, with the same instrumentation.
- Arrival stamps are process-local. A restart cancels every in-flight task, so there are no
  waiters to carry across it; nothing is persisted (INV-7: the hold's record expires
  coherently with its waiters).

Invariants the implementation must hold, each checked at more than one point (heuristic 10):

1. A waiter cancelled while queued leaves the queue in the context manager's `finally`; a
   waiter cancelled after admission hands its turn on in the same `finally` (INV-6
   `status-matches-liveness`). The existing shielded-enter / done-callback flock release in
   `verify.py::_admission_slot` is preserved — the gate wraps it, it does not replace it.
2. Ungated roles never touch the gate or the executor (task 5424's fix stands).
3. `slots` is read at each grant, so `verify_admission_task_slots` stays green-tier.
4. A flock fail-open yields `slot_held=False` on the grant and therefore on the event — a
   caller can tell an ungated run from a gated one (INV-11 `no-silent-fail-soft`).

Config (all green-tier, added to `RELOADABLE_FIELDS`):

| key | default | meaning |
|---|---|---|
| `verify_admission_order` | `'unordered'` | `'unordered'` \| `'arrival'`. Code default preserves behaviour; projects flip in yaml |
| `verify_admission_background_promote_secs` | `7200` | background waiter ranks as task class after this age |
| `verify_admission_wait_alarm_secs` | `43200` | oldest-waiter age that raises the alarm (above today's p90 ≈ 11 h so it is not a standing alarm) |

Events (`event_store.py::EventType`):

- `verify_admission_plan` — one per gated plan, emitted by `run_scoped_verification` /
  `run_full_verification` from the legs' `CheckRun`s: `{plan_id, task_id, role, order,
  plan_arrival_at, completed_at, legs: [{module, scope_kind, arrived_at, granted_at,
  waited_secs, held_secs, depth_at_arrival, slot_held, rc, timed_out}]}`. Wall-clock ISO
  times in the payload; monotonic only inside the gate.
- `verify_admission_heartbeat` — from the harness watch, only while depth > 0, at the
  merge-heartbeat cadence: `{depth, oldest_wait_secs, oldest: {task_id, module, role},
  holders: [...], order}`.
- `verify_admission_wait_aged` — from the watch, when the **oldest** waiter first crosses
  the alarm age, and again only when the identity of the oldest aged waiter changes; payload
  names that waiter and the count of aged waiters. One fact per change, not one per waiter
  per tick (INV-4 `storm-escape-required`).

### Boundary scenarios (the code leaf's acceptance)

Driven through `gate_for(tmp_path)` and the real flock — no patching of private names
(docs/code-quality.md, Tests stance; INV-10 `guards-exercise-behaviour`).

| # | scenario | asserts |
|---|---|---|
| 1 | three task plans arrive A, B, C while the slot is held; `order='arrival'` | grants A, B, C |
| 2 | same, `order='unordered'`, many trials | every permutation occurs |
| 3 | background plan arrives before task plan T; slot held | T granted first |
| 4 | background waiter older than `promote_after_secs`; tasks T1 (older), T2 (younger) waiting | T1, background, T2 |
| 5 | plan with 6 modules under `_fanout_sem` 4; plan Q arrives after it | the plan's 5th and 6th legs are granted before Q |
| 6 | a queued waiter is cancelled | it is absent from `snapshot()`; the next waiter is granted; no slot leaks (probe with `wait=False`) |
| 7 | an admitted waiter is cancelled mid flock poll | turn passes on; the late-acquired flock is released |
| 8 | `role='merge'` | returns immediately; gate snapshot and executor untouched |
| 9 | slots dir unusable (flock fails open) | run proceeds, grant `slot_held=False`, plan event carries it |
| 10 | a sweep (background, 8 modules) holds the slot with module 1; a task arrives | task is granted before the sweep's module 2 — `test_verify_admission_integration_gate.py::TestSweepYieldsAndInterleaves` can drop its `_DeterministicAcquire` stand-in and run against the real primitive |
| 11 | workflow task verify end to end with a real event store | one `verify_admission_plan` row whose legs' `waited_secs` match the grants |
| 12 | depth > 0 across a watch tick; oldest waiter beyond the alarm age | one heartbeat row; exactly one `verify_admission_wait_aged` row across repeated ticks |

## Resolved design decisions

1. **In-process gate, flock unmodified.** A cross-process ordered semaphore (ticket files,
   arbiter) is not needed for either project and would retire contract clauses that have
   tests and incident history behind them.
2. **Plan-level arrival stamp**, no task-level aging (above).
3. **Two role classes with promotion**, not pure FIFO: pure FIFO would let an 8-module
   background sweep (~1 h of slot) sit contiguously ahead of every later task leg and would
   break the pinned sweep-yields property; strict classes without promotion would starve the
   sweep at ρ≈1. Promotion keeps the original stamp, so the bound cannot silently turn the
   policy into something else.
4. **`review_checkpoint.py` verifies at `role='background'`.** It is a periodic project-wide
   review on the main checkout, not a task's verify; it rides the task class today only
   because `run_full_verification`'s `role` defaults to `'task'`. `run_full_verification`'s
   `Literal` gains `'background'` so the two `# type: ignore[arg-type]` at the sweep call
   sites go away.
5. **Facts by return value; gate does no I/O.** No observer registration, no drain queue
   (heuristic 7, stateless interactions). The two `workflow.py` call sites pass their
   `event_store`.
6. **Ship behaviour-preserving, flip by config.** The code default is `'unordered'`; each
   project flips `verify_admission_order: arrival` in its yaml (green-tier, reversible by
   `reload_config`). Every event carries `order`, so before/after is self-describing
   whatever the deploy timing.
7. **No size classes, no priority key.** Declined / refuted (Background). The request carries
   `scope_kind` and `module` for measurement only.
8. **Reify needs no reify-script change.** The DF-side gate is where reify's verifies queue
   (its bash semaphore recorded 58 waits in a month, p50 3 s). Reify has one service class,
   so arrival order is the whole policy there. `REIFY_SLOT_EVENT_LOG` stays off — the new
   events measure the queue that matters.

## Pre-conditions for activating

- None in code. Substrate verified 2026-09-18: `run_scoped_verification(event_store=…)` exists
  (workflow does not pass it yet — α wires it); `CheckRun` is the per-leg record;
  `RELOADABLE_FIELDS` lists the existing `verify_admission_*` keys; `EventType` is a
  `StrEnum` in `event_store.py`.
- α takes effect at the next orchestrator restart (code); β and γ at the next
  `reload_config` or restart after α is live. A config flip against a process that predates
  α is inert — which is why measurement keys on the event's `order` field, never on dates.

## Cross-PRD relationship

| Other | Direction | Seam | Owner | Status |
|---|---|---|---|---|
| task 5413 (R8: slot ordering + two-class trial) | superseded | whole task | this PRD | cancel at decompose with pointer |
| task 5139 (how merge/mainprobe verifies are admitted or capped) | adjacent | `verify.py::_admission_slot` role branch | 5139 owns the *decision*; this PRD owns the gate and its role-class table, which is where a future mainprobe class would be added | α lands first or second — same function; serialise by dependency at decompose |
| `plans/verify-oversubscription-control-prd.md` | extends | C-clauses, sweep-yields boundary scenario | that PRD's contract is kept verbatim | wired |
| `plans/merge-lane-quality-prd.md` | none | — | — | no file in its set is touched |
| `plans/dashboard-one-datum-one-path-prd.md` | none now | a dashboard panel for the heartbeat would be a consumer there | that PRD, if wanted | out of scope here |
| task 5624 (env-recovery retry carries stale lint/type runs) | adjacent | `CheckRun` | 5624 | independent; both add fields to `CheckRun` — serialise |

Known residual the gate cannot order: `orchestrator/evals/metrics.py` calls
`run_verification` at the default `role='task'` from a separate process; pointed at the DF
root it contends on the flock directly. The flock still excludes it; it can overtake. Rare,
recorded, not fixed here.

## Decomposition plan

- **α — Ordered admission gate + recorded wait (code).** New
  `verify_admission_gate.py`; `verify.py` (`_admission_slot` wraps the gate; `AdmissionRequest`
  built once per plan in `run_scoped_verification` / `run_full_verification` and threaded to
  `run_verification`; grant attached to the test `CheckRun`; plan event emission);
  `config.py` (three knobs + reload registry + validation); `event_store.py` (three event
  types); `harness.py` (the watch, started and stopped with the harness);
  `review_checkpoint.py` (decision 4); `workflow.py` (pass `event_store` at the two
  task-verify call sites); tests for boundary scenarios 1–12; `OPERATIONS.md` (knobs, events,
  how to read a wait). Behaviour-preserving by default.
  *Signal:* boundary scenarios 1–12 green against the real flock and a real event store; on a
  running orchestrator every task verify produces a `verify_admission_plan` row and a waiting
  queue produces `verify_admission_heartbeat` rows.
  *Prereqs:* none in-batch; serialise against 5139 and 5624 (same files).
- **β — Flip Dark Factory to arrival order (config).** `verify_admission_order: arrival` in
  `dark-factory-orchestrator.yaml`, with a dated comment citing this PRD and the prediction
  to check (mean ±0, p90 −30 %, max −66 %). Milestone: delayed 72 h after α, so a baseline
  of `order='unordered'` events exists when a restart allows. `complexity: simple`.
  *Signal:* after the next reload/restart, DF `verify_admission_plan` rows carry
  `order='arrival'` and no same-class overtake appears in them.
  *Prereqs:* α.
- **γ — Flip reify to arrival order (config, reify repo).** Same edit in
  `/home/leo/src/reify/dark-factory-orchestrator.yaml`, filed in reify's task store with an
  external dependency on α. *Signal:* reify rows carry `order='arrival'`.
- **δ — Decide whether to cancel a plan's still-waiting legs when an earlier leg goes red.**
  Milestone: delayed 14 days after β. A normal agent task that measures, from
  `verify_admission_plan` rows with `order='arrival'` (≥ 14 days of them, else re-defer by
  escalating info rather than deciding on thin data): slot-seconds consumed, and wait
  imposed on others, by legs that were granted **after** a sibling leg of the same plan had
  already ended red (or after the plan's lint/type had failed). Decision rule: if that
  avoidable slot time is ≥ 3 % of slot-busy time (≈ 0.7 h/day on DF) file the cancel-on-red
  task with the measured numbers; otherwise record "not worth a task" with the numbers. Either
  way write the decision to fused-memory. The consumer that makes α's per-leg `rc`,
  `granted_at` and `plan_id` load-bearing (G1).
  *Signal:* a filed task id or a recorded declination, each carrying the measured share.

## Out of scope

Size- or priority-aware ordering (declined); a second slot (ruled 2026-09-08); WIP changes
(`max_concurrent_tasks` is Leo's ruling); capping or classing merge / mainprobe verifies
(5139); a dashboard panel; replacing the autouse `_neutralize_verify_admission` test seam
(worth doing — it patches a private name — but it touches every verify test's setup and
the merge-lane ratchet baseline; file separately if wanted); an MCP read tool for the
snapshot; fixing `evals/metrics.py`'s default role.

## Open questions (tactical)

1. Which harness loop hosts the watch, and its tick. **Suggested:** its own small task on the
   merge-heartbeat cadence. Decide in α.
2. Whether `run_full_verification` should take an `event_store` so background plans emit
   `verify_admission_plan` too. **Suggested:** yes — slot-occupancy accounting wants the
   sweep's share (measured ~5 %). Decide in α.
3. Within one plan, legs are granted in module order. If δ files cancel-on-red, that task
   should consider granting a plan's cheapest leg first — an ordering *inside* a plan moves
   no wait between plans. Decide in δ's follow-up, not here.
4. Retire `'unordered'` once arrival order has run for a quarter without a rollback.
