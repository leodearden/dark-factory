# Priority at the later gates: one comparator for verify admission and the merge pool, and a pin as a tier

**Status:** active. Authored 2026-11-13; rulings by Leo on 2026-11-12 and 2026-11-13 (section 3). Part 1 of 2:
part 2 (batches chosen from evidence, suspects and conviction) is deferred until task 8528 has landed and task
9908's recorded rows exist; its design notes are `plans/merge-pool-batches-design-notes.md`.
**Code anchors** verified against main `a8bd0b5eb1` (2026-11-13). Main moves fast: cite by symbol; re-locate at
implementation time. Branch facts for task 9908 are from `task/9908` at `ae8ef8298b` (not landed).
**Evidence:** `plans/priority-at-later-gates-prd.evidence/`: the study report (`REPORT.md`), the scout reports
(`scheduler.md`, `lane.md`), the measurements (`measure.md`) and the critic passes on this document and its
predecessor.

## 1. Goal

Three gates decide which task goes next: the scheduler, the verify-admission gate and the merge queue. Today only the
scheduler reads priority, and a pin acts only at the scheduler and only on the pinned task itself. This PRD makes
the two later gates order their waiters by the scheduler's priority, `(tier, age)`, strict across tiers, with one
shared comparator, and makes a pin a tier that reaches the pinned task's prerequisites and both later gates.

It does not change the scheduler's own order within a tier (D3), and the merge lane's "high" lane stays for main
repair (D10).

What an operator observes when it has landed:

1. With `merge_pool.order: priority`, a pinned or high-tier task that reaches the merge lane after lower-tier
   requests is picked before them, unless they are already merged or being verified. `merge_dequeued` rows show the
   picked request first among the recorded waiting set, by `rank` and `age_secs`.
2. With `verify_admission_order: priority`, a high-tier task's verify leg is granted before a lower-tier leg that
   arrived earlier.
3. At both gates a task's age counts from when it could first have run, has no cap, and survives dispatch, re-pend
   and restart.
4. A low-tier waiter that has waited longer than the gate's promotion period is treated as one tier higher.
5. `set_task_priority_override(pinned=True)` orders the pinned task first at both later gates at once, and, when
   the scheduler half has landed, lifts its unfinished prerequisites at all three gates.
   `get_scheduler_state().effective_priorities` shows them at tier `pinned`.

## 2. Background and premise validity (G6)

From the study (`evidence/REPORT.md`) and the measurements (`evidence/measure.md`; dark_factory, "recent" =
2026-10-30 to 2026-11-13, "paused" = 2026-11-09 to 2026-11-13):

- **Priority is read at one gate.** 39% of merge picks in the paused window took a request while a higher-tier or
  pinned one waited. `merge_lane/types.py::MergeRequest` has no tier; the pick is lane, then first-submission age
  among requests that touch the same files (`merge_lane/intake_buffers.py::MergeIntake.pop_next_pickable`). The
  verify-admission key is `(role class, arrival, seq)` (`verify_admission_gate.py::AdmissionGate._pick`).
- **The size of the prize is modest and is stated as such.** Simulated on the recorded queue, a priority pick is
  worth under an hour to an urgent request at today's arrival rate, more when the queue is long. The case for this
  PRD is that hand-set lanes and pins that stop at dispatch are standing in for an order the later gates do not
  have. No latency target is asserted.
- **The same-files rule is, in practice, oldest first.** Its test compares current main with each branch head, so it
  includes everything main changed since the branch was cut: at one snapshot it linked all 105 pairs of 15 queued
  requests, against 17 by what the branches themselves changed.
- **The rebase cap never binds.** `MERGE_BOUNCE_CAP` = 3 was exceeded 0 times in 19 days; real conflicts sent a
  request back 53 times, at the first occurrence.
- **No age survives past dispatch.** `metadata.pending_since` is deleted by every `metadata_mode='replace'` write,
  by design (task 3816's design decision 8): 26 of 27 done tasks with id 9000 or above have none. The watchdog's
  `eligible_since` (`eligibility_clock.py::EligibilityClock`) is forgotten at dispatch and at every re-pend.
- **A pin stops at the pinned task.** `Scheduler._compute_effective_priorities` takes the override's `boost_tier`
  and never `pinned`; a pinned task with unmet dependencies is not in the pin queue (`Scheduler._pin_queue`).
- **Measured waits** (the backstop defaults rest on them): verify-admission wait per task-role leg, 95th percentile
  13,135 s (recent, n = 1,817); merge-queue wait to first pick, 95th percentile 32,018 s (recent, n = 550). The
  recent window includes days when grouped merges distorted the merge-wait figures; the paused window gives
  25,305 s (n = 181).

**Premises checked.**
- *The priority pick does not break the pending bisect.* Task 8528 requeues a failed batch's members "at the front"
  and verifies the first half next. Under a computed order there is no front, so in priority order 6.4 keeps that task's suspect list
  first, and the priority pick sits behind a switch whose default, `arrival`, is today's code path untouched.
- *Dropping the same-files rule costs no extra rework in total.* When two waiting requests truly conflict, one must
  be reworked whichever lands first; order decides which. Under priority order the lower-priority one pays. The
  readout (ι1) reports, before the switch is flipped, how many overtakes would cross a recorded confirmed conflict.
- *A request with no age is not sent to the back.* A request whose task the scheduler never ranked (a hand-carried
  task is held `deferred`; an interactive merge) is aged from its own first enqueue (6.2).

## 3. Resolved design decisions

**Ruled by Leo (2026-11-12, 2026-11-13).**

- **R1.** Nothing already merged or being verified is displaced for a higher-priority arrival.
- **R2.** The merge queue is taken in priority order. Priority there is the scheduler's.
- **R3.** Priority is `(tier, age)`, strict across tiers, at both later gates, with a promotion after a long wait as
  the only starvation backstop. The backstop's default comes from real recorded waits.
- **R4.** Age keeps accruing after a task is scheduled, through to done, with no cap.
- **R5.** A pin is a tier above critical in one uniform tiering.
- **R8.** The pick change lands after task 8528. The recording task 9908 is hand-carried ahead of everything.
- **R9.** The same-files rule is dropped from the pick; the test is narrowed and kept as the conflict-probe filter.
  The rebase cap is relaxed substantially.

(R6 and R7, on batches and suspects, belong to part 2.)

**Ruled by Leo, 2026-11-13** (each was put to him with its alternative; he took the first option shown).

- **D2. Where age starts.**
  - *(ruled)* **First eligible, kept by the scheduler.** The scheduler's eligibility store
    (`eligibility_clock.py::EligibilityClockStore`, already written each tick for every candidate) gains a second
    table, `first_eligible(project_root, task_id, at)`. Time parked behind a dependency does not count. No task
    metadata key and no fused-memory change.
  - **First pending.** Make `pending_since` durable by reversing task 3816's design decision 8. Time parked behind a
    dependency then counts as age: with no cap, a task parked for two months outranks, within its tier at the later
    gates, one that has been ready for one. About 177 open tasks are parked behind task 8528 alone.
- **D3. The scheduler's own order within a tier.**
  - *(ruled)* **Unchanged.** The scheduler is already strict across tiers and its age term grows for as long
    as a task waits (it saturates; it does not stop). Only the later gates take raw uncapped age.
  - **Uncapped age at dispatch too**, with the dependents and work-product terms as fixed hours or as a multiplier.
    Fixed hours make dispatch first-come within a tier for tasks more than about two days apart and reverse the
    earlier scoring plan's founding case; a multiplier needs fitting. Either is its own PRD.
- **D20. Two PRDs.** Priority at the later gates and the pin tier are this PRD; batches chosen from evidence are
  part 2.
- **D21. α2 waits for task 9636**, which edits the same watchdog code and is already ruled.
- **D18. Task 8261 becomes this PRD's β.** It implements the shared comparator at verify admission directly,
  keeps its id, pin and dependents, and waits for α1. The alternative is to let it land its own blended key first
  and replace it later.

**Decided here.**

- **D1. One value, one comparator.** `orchestrator/task_priority.py` (created by task 9908 as `TaskPriority(tier,
  pinned)`) grows `base_tier`, `pin_order` and `age_anchor`, defines `PINNED = 'pinned'` and the gate rank order
  (`PINNED`, then `config.PRIORITY_TIERS` in order), and gains the one ordering function `order_key` (6.1). Verify
  admission and the merge pool sort by `order_key` and build no key of their own. Once task 6718 has created
  `shared/task_priority.py`, θ2 moves the name and rank order there.
- **D5. `pinned` is an effective tier, never a stored one.** From α1 the lookup returns tier `pinned` and the row's
  `pin_order` for a task whose own override row is pinned, so both later gates honour pins at once. In practice that
  is an operator pin: the watchdog's automatic pins are released at dispatch and never reach the later gates. θ1 adds inheritance to prerequisites. θ2 adds the refusals that make
  "never stored" true (the task store does not validate `priority` today).
- **D6. Pin order is the order inside the `pinned` tier**, and `reorder_pin_queue` keeps working. Inside the
  scheduler the walk returns, for each task, its base tier (as today, pins ignored) and a `pin_order` when it or an
  unfinished dependent is pinned (the smallest among them); its tier is `pinned` exactly when `pin_order` is set.
  The scheduler sorts by `(0 if pin_order is set else 1, pin_order, -score, id)`, the score computed from the base
  tier. `TIER_BASE`, the tier-width validator and the skip thresholds gain no entry. The candidate that accrues
  skips and earns tier parks is, as today, the best by score alone, so a waiting pin does not stop anyone else
  earning one. Score, skip threshold, park rank, `reserve_now` and `lock_acquired.priority` read the base tier;
  `effective_priorities` and the lookup read the tier. The pin pass's other duties run from the one loop for
  `pinned`-tier candidates: the reservation head (D7), `pin_blocked`, skipping a load-held heavy pin, exempting
  deterministic pins. Inside the scheduler the tier map stays the base tier and a second map holds pin
  orders; only the state snapshot and the lookup combine them, so no internal reader changes. Inheritance is
  through every unfinished dependent, direct or not, and a pinned row with no stored order counts as 0, as today.
  The watchdog neither counts lock refusals for, nor pins, a candidate whose tier is already `pinned`. A
  pinned-tier candidate gets the park-borrowing retry in pin order. The best-scored candidate accrues a skip on a
  tick where a pinned-tier candidate dispatches.
- **D7. The pin reservation stays bounded.** A park installed for a `pinned`-tier owner still ranks above every
  other tier and is ordered by `pin_order`. At most `pin_reservation_max_active` owners hold one, chosen as today,
  and `Scheduler._bound_pin_reservations` keeps enforcing it. Inherited prerequisites compete for that reservation
  by `pin_order`; they do not each get one. `pin_blocked`, `pin_reservations` and `pin_queue` keep their meaning:
  `pin_queue` lists stored pins only.
- **D8. No promotion backstop at the scheduler.** The starvation watchdog remains that gate's backstop.
- **D9. The promotion backstop is a step.** A waiter that has waited at a gate longer than that gate's promotion
  period is treated as one tier higher for each such period, stopping at `high`: it never ties with real `critical`
  or `pinned` work. Waiting is counted from the request's own enqueue (merge pool) or the plan's arrival (verify
  admission). Defaults are the 95th percentile of the gate's own recorded wait, rounded up to the hour, per
  project, printed by a committed script and committed in the project's config by the task that ships the knob.
  Derived once, not live: a live percentile would lengthen as the low tier waits longer.
- **D10. The "high" lane remains**, strictly ahead of everything, ordered inside by `order_key` when the order is `priority`. It stays reserved
  for main repair; moving ordinary work into it by hand should stop being needed once pins reach the lane.
- **D14. Each gate's new order sits behind its own switch, defaulting to today's.** `verify_admission_order` gains
  the value `priority` (per project, as today). `merge_pool.order` is `arrival` or `priority`. The merge pool's
  switch is flipped by a human gate that presents a readout of recorded picks (ι1, ι2), after task 8532's seven-day
  readout of the lane has been taken.
- **D17. New lane code goes in new modules** within the merge-lane ratchet's ceilings; `merge_pool` config is its
  own sub-model in its own file, mounted once.

**Considered and rejected.**
- *One PRD for priority and evidence-driven batches.* Two critic passes found 31 defects, nearly all where the
  batch design met lane code that is planned and not built (task 8528, task 5041). Part 2 is designed against the
  real code and the recorded rows once they exist.
- *A separate urgency value for the merge lane* (what a request blocks upstream). Left to the lock hand-off work.
- *Deriving the "high" lane from the tier; choosing the verify host by priority.* Simulated as harmful or worth
  minutes.
- *Making the pin reservation the ordinary top-candidate park.* It loses the cap of one and reserves for every
  pinned prerequisite, which the earlier scoring plan rejected.
- *Keeping the same-files rule and narrowing its test.* Possible; it would let urgency choose only among requests
  with no older truly-overlapping request. Leo ruled it dropped from the pick (R9).

## 4. Pre-conditions (G3)

Verified on main, by symbol:

| Assumed | Evidence |
|---|---|
| One pick site in the lane | `merge_lane/intake_buffers.py::MergeIntake.pop_next_pickable`, called from `worker.py::SpeculativeMergeWorker._acquire_next_request` and `::_look_ahead` |
| The pick affects only waiting requests | the same; nothing reaches `_verifier_queue`, `_inflight` or the redispatch park (`MergeIntake.relane_waiting` refuses with `already_picked`) |
| The same-files rule's site | `MergeIntake._clique_minimal_visible` |
| A fork-point changed-files query | `git_ops.py::GitOps.get_branch_changed_files` (`main...ref`), returning `(files, error)` |
| The admission key's site | `verify_admission_gate.py::AdmissionGate._pick`; policy built by `verify.py::admission_policy` |
| The candidate stamp, written each tick | `eligibility_clock.py::EligibilityClock.observe`, from `scheduler.py::Scheduler._apply_starvation_watchdog` |
| The effective-tier walk with inheritance | `scheduler.py::Scheduler._compute_effective_priorities` |
| The single consumer of the score is one sort | `scheduler.py::Scheduler._score_candidates` |
| Only the top-scored candidate earns skips and tier parks | `scheduler.py::Scheduler._phase_select_scored`, `::_bump_skip_and_maybe_park` |
| The bounded pin reservation | `scheduler.py::Scheduler._bound_pin_reservations`, `pin_reservation.py::PinOrder` |
| The override store with `pinned`, `pin_order`, `source` | `orchestrator/overrides.py::OverrideStore` |
| Hot-reloadable config sub-models | `config.py::MergeDeepConfig`, `RELOADABLE_FIELDS` via `_submodel_leaf_paths` |

Not on main:

| Needed | Delivered by | Status 2026-11-13 |
|---|---|---|
| `TaskPriority`, `read_task_priority`, the injected lookup (`Scheduler.task_priority`), `merge_pool_facts.py`, the recorded pick rows | task 9908 | implemented and reviewed on its branch; not merged |
| The suspect list task 8528 requeues | task 8528, behind 8527 and 8526 | pending |
| `shared/task_priority.py` and the override store's shared home | task 6718 | pending |
| The last `scheduler.py` leaf of the restructuring chain | task 9683 (confirm at decompose) | pending, 542 open tasks upstream |
| The durable first-eligible table | built here (α2) | — |

## 5. Cross-PRD relationships and seam owners (G4)

| Other work | Direction | Seam | Owner | Action |
|---|---|---|---|---|
| Task 9908 (recording) | consumes, extends | `TaskPriority`, `read_task_priority` (it rebuilds the value field by field), the lookup, pick records | 9908 builds; α1 extends | α1 edits the guard so new fields survive and gives every new field a `None` default |
| Task 8261 (verify admission by tier) | becomes β (D18) | the admission key | this PRD | **replace** its description and details at decompose with β's text (value `priority`, `order_key`, the promotion period); keep its id, pin and dependents; depends on α1 |
| `verify-admission-fifo-prd.md` decisions 2, 3, 7 | amends | no task-level aging; background promotion; no priority key | β | dated delta note |
| Tasks 5139, 9142 | same files | `verify.py`, `config.py`, `verify_admission_gate.py` | each its own | serialised by the file lock; dated note on 9142 that the gate takes a priority lookup |
| Tasks 5621, 6124, 6944 | same code | the footprint and bounce code in `suffix_graph.py` | each its own | dated note on each: under `order=priority` the footprint is the branch's own changes |
| Task 9705 (admission and lane knobs move) | same knobs | where admission config lives | 9705 | it already depends on 8261; unchanged |
| Task 8528 (bisect and depth table) | consumes | its suspect list stays first in the pool order | 8528 owns its code; γ reads the list | γ depends on 8528 |
| Task 8532 (seven-day readout) | none | reads the lane with `order=arrival` | 8532 | ι2 does not fire until 8532 has read |
| `merge-lane-quality-prd.md` frozen line | amends | "main advances strictly in submission order" becomes "in the order verifies were dispatched" | γ | dated delta note |
| `two-layer-merge-queue-prd.md`, task 1891 | supersedes, when `order=priority` | the same-files pick rule | γ | dated delta note |
| `deep-merge-ahead-prd.md` decision 1 | amends | "in submission order" becomes "in pool order" | γ | dated delta note |
| Tasks 9182, 9152, 9723 | same file | `merge_lane/intake_buffers.py` | each its own | serialised by the file lock |
| Task 9188 (merge tools move) | consumes | queue position in `get_merge_queue` and `merge_status` reads the pool order | 9188 moves; γ changes what they read | dated note on 9188 |
| Task 7921 (main-red hold) | same function | held requests are skipped by the pick | 7921 | dated note on 7921: its skip goes in the pick's filter, after `pool_order()`; γ leaves room for it |
| Task 5623 (journal loses the first-enqueue stamp) | consumes | `first_enqueued_at` is the fallback age (6.2) | 5623 | stays open; until it lands a restart resets that fallback |
| Task 9636 (watchdog cap, restart-safe inputs) | same file and method | `eligibility_clock.py`, `Scheduler._apply_starvation_watchdog` | each its own | a real edge: α2 waits for 9636 (D21) |
| Task 9679 (scheduler ledger, one record per task) | none | `first_eligible` lives in the eligibility store, not in the ledger | 9679 | dated note |
| `scheduler-dispatch-scoring-and-lock-layer-prd.md` | amends | the pin pass as a dispatch order; the pin reservation exception is kept (D7) | θ1 | dated delta note |
| Tasks 6328, 3824 | same rule | the park order and who earns tier parks | each its own | unchanged by D6; dated note on both |
| Tasks 6718, 6642, 6722, 7009, 4709 | consumes | tier vocabulary, validators, park payload, override events | each its own | θ2 after 6718; `pinned` is declared once |
| Part 2 (batches from evidence) | produces for | `pool_order()`, `order_key`, the `merge_pool` config file | this PRD | part 2 is authored after 8528 lands |
| Lock hand-off study (unruled) | none | reservations beyond the bounded one | that study | not designed here |

## 6. Contract (B + H)

### 6.1 Priority and its order

```
PINNED = 'pinned'
GATE_RANK: tuple[str, ...]              # PINNED, then config.PRIORITY_TIERS in order

@dataclass(frozen=True)
class TaskPriority:
    tier: str | None = None             # effective tier; PINNED when pin_order is set
    base_tier: str | None = None        # the tier with pins ignored
    pinned: bool | None = None          # task 9908's field, kept: the task's own row is pinned
    pin_order: int | None = None
    age_anchor: datetime | None = None  # first_eligible.at (D2)

def order_key(p: TaskPriority, *, now: datetime, waited_secs: float,
              promote_after_secs: float | None,
              fallback_anchor: datetime | None) -> tuple[int, int, float]
```

- Smaller sorts first: `(rank, pin_order or 0, -age)`. `rank` is the tier's index in `GATE_RANK`. `age` is `now`
  minus `age_anchor`, else minus `fallback_anchor`, else 0.
- Promotion: `rank` is reduced by `floor(waited_secs / promote_after_secs)`, never past the rank of `high`.
- An unknown tier or a failed lookup gives the rank of `medium`; the caller counts it on its event. A waiter with no
  task (a background leg at verify admission) is not a failed lookup. The lookup never raises. Where a lookup is wired, ten
  consecutive picks at one gate with an unknown tier file one L1 (deduplicated while it is open): the lookup is
  broken. α1 builds the streak counter; β and γ call it.
- Defaults (dark_factory): verify admission 4 h; merge pool 9 h (8 h on the paused window alone). β and γ re-run the
  script on the 14 days before they are implemented and commit what it prints. Other projects: an operator reload.

Invariants:
- **P1.** With its switch at `priority`, a gate orders waiters by `order_key` only: after role class at admission;
  after the high lane and task 8528's suspects at the pool.
- **P2.** Ties break on the caller's existing sequence (`seq` at admission, request id in the pool).
- **P3.** Among waiters not promoted, a higher tier always precedes a lower.

### 6.2 The age anchor

- `EligibilityClock.observe` inserts into `first_eligible` as well as the watchdog's table, INSERT OR IGNORE.
  `forget` (dispatch, re-pend) clears only the watchdog's table. Each tick, for rows whose task is absent from the
  active fetch, the scheduler makes one `Scheduler.get_statuses` call; a row is deleted when that read answers done
  or cancelled, and kept on an error or no answer. A task cancelled and re-pended between two such reads keeps its
  old age; that is accepted.
- At deploy the table is seeded once, before the first tick: for tasks that are candidates now, from their
  `eligibility_anchors` row; for tasks already dispatched, from their first `lock_acquired` or `task_started` event,
  else `updated_at`. This under-ages, except for a task cancelled and re-pended before the seed. Seeded rows are
  marked.
- `Scheduler.task_priority` returns the anchor from memory. At construction the scheduler fills its tier map from
  the last `scheduler_state.json` it wrote and its pin rows from the override store, so the lookup answers before
  the first tick and while the scheduler is paused. While paused it re-reads the override store at most once a
  minute, from the paused return of `Scheduler.acquire_next`, off the event-loop thread, into a pin map of its own
  (not the tick's override snapshot, whose diff emits the pin events on resume). A pin set during a pause is then
  honoured at the later gates. A tier from the file, and an inherited pin, are as old as the last tick before the
  stop; that is accepted.
- `fallback_anchor` is the request's `first_enqueued_at` at the pool and the plan's arrival at admission. A request
  aged from the fallback is counted on its event.
- **A1.** A task's `first_eligible.at` is unchanged by any dispatch, re-pend, metadata write or restart.
- **A2.** The stamping adds no task-store write to a tick, and at most one status read.

### 6.3 Verify admission

- `AdmissionOrder` gains `'priority'`. Under it `AdmissionGate._pick` takes the minimum of `(class,
  order_key(...), seq)` over the waiters. A task leg's priority is read through a lookup carried on its
  `AdmissionPlan` (set where the plan is built, in `task_verify_dispatch.py` and `verify_admission_record.py`); both
  gate sites use it (`gate_for(slots_dir)` and the per-host gates in `verify_hosts.py::HostPool`). The gate is
  in-process, so the order holds within one orchestrator; between processes the slot's poll race decides, as today.
- A background leg has class 1 and is ordered by plan arrival. Once promoted it has class 0 and is ordered as the
  oldest `high` waiter: it waits only for pinned and critical legs. The fallback age and the promotion clock are
  the plan's arrival, not the leg's.
- A task that fails its verify and returns keeps its age, so within its tier it returns towards the head. This
  reverses decision 2 of `verify-admission-fifo-prd.md` ("a repeat failer must not own the head"), as R4 requires.
- `verify_admission_priority_age_secs` is the promotion period (hot-reloadable). The tier, `base_tier` and age are
  read through the lookup at each pick and recorded on `verify_admission_plan` at arrival and at grant.
- **V1.** With the value `arrival` or `unordered` the gate behaves exactly as today.

### 6.4 The pool order

- With `merge_pool.order: arrival` the pick is today's code path, untouched: the lane buffers as they stand (task
  8528's requeue included), the same-files rule, and the footprint computed as today. `pool_order()` returns buffer order; the pick applies
  the same-files rule over it and each reader keeps the filter it has today.
- With `priority`, `pool_order()` is: the high lane, by `order_key`; then the requests of task 8528's suspect list
  in its order; then the rest by `order_key`. The same-files rule is not applied, and the conflict graph's footprint
  is each branch's own changes (`GitOps.get_branch_changed_files`), where a git error is a probe error, not an
  empty footprint. `two_layer_invariants` checks the graph only and is unaffected.
- `pool_order()` returns every buffered request; the pick filters out reserved requests and halted lanes, as today.
- Every reader of "the order requests will be taken in" reads `pool_order()`: the pick, `chain_snapshot`,
  `unfrozen_suffix`, `snapshot()`, the conflict graph's node order, `take_buffered`, `get_merge_queue` and
  `merge_status` positions, `merge_heartbeat.head_of_line` and task 9908's pick records. The shutdown drain is
  exempt; `_rebucket_relaned_requests` is a writer and is left as it is; `merge_queued.position` stays the arrival
  position. `pop_next_pickable` keeps its return type. The conflict graph's debounce signature is keyed on the set
  of waiting ids and on `merge_pool.order`, so a promotion does not force a recompute and a flip of the switch
  does.
- `merge_pool.bounce_cap` replaces `MERGE_BOUNCE_CAP`; default 20, a loop-breaker.
- Config `merge_pool` (own file, all leaves hot): `order`, `promote_after_secs`, `bounce_cap`.
- Recorded: `age_secs`, `base_tier` and `age_from` (`anchor` or `enqueue`) on `merge_queued` (α1); `rank` and `age_secs` on the picked and each waiting entry of `merge_dequeued` (γ). No existing key changes
  meaning.
- **O1.** Main advances in the order verifies were dispatched. Arrival order is not preserved.
- **O2.** A request being merged or verified is never reordered, re-based or returned to the pool by a later
  arrival (R1).
- **O3.** With `order=arrival` the sequence of picks is the same as before this PRD for the same arrivals.

## 7. Boundary-test sketch (B + H)

Driven through the lane's public seam (requests in, events out), the admission gate's entry and the scheduler's
dispatch entry.

| # | Scenario | Preconditions | Asserts |
|---|---|---|---|
| 1 | Later high-tier request | `order=priority`; three low-tier requests wait, none in flight; a high-tier request arrives | it is picked next; `merge_dequeued` lists the three with `rank` and `age_secs` |
| 2 | Nothing in flight is displaced | `order=priority`; a low-tier request is being verified; a request whose task is pinned arrives | the verify is not cancelled; the pinned request is the next pick |
| 3 | Promotion backstop | `order=priority`; a low-tier request has waited past the promotion period; fresh medium-tier requests keep arriving | it is picked ahead of them; a fresh high-tier request still precedes it after one promotion; after three periods it still does not precede a critical request |
| 4 | Main-repair lane | `order=priority`; a high-lane low-tier request and a pinned normal-lane request wait | the high-lane request is picked first |
| 5 | The pending bisect keeps its place | `order=priority`; task 8528's suspect list holds four requests; a high-tier request arrives | the next pick is the first suspect, not the arrival |
| 6 | Arrival order unchanged | `order=arrival`; a recorded sequence of arrivals, a re-submitted request among them | the picks are those recorded; the re-submitted request keeps its seniority |
| 7 | Age survives | a task is dispatched, re-pended, and the orchestrator restarts | its `first_eligible.at` is unchanged; `merge_queued.age_secs` is from it |
| 8 | Parked time does not count | two same-tier tasks: one filed 30 days ago behind a dependency that landed today, one a candidate for 5 days | at the pool the 5-day task's request precedes |
| 9 | No anchor | a request for a task the scheduler never ranked waits with two same-tier requests, one enqueued before it and one after | it is ordered by its first enqueue, between them by age; `merge_queued.age_from == 'enqueue'` |
| 10 | Cancelled and re-pended | a task is cancelled, a tick runs while it is cancelled, and it is returned to pending | its age starts again from its next candidacy |
| 11 | Operator pin on a running task | a low-tier task in progress is pinned | its next verify leg and its merge request are ordered at `pinned`; the recorded `base_tier` is `low` |
| 12 | Verify admission | `verify_admission_order: priority`; a high-tier task leg arrives after a low-tier one | the high-tier leg is granted first; a background leg's class rule is unchanged and it is not counted as a failed lookup |
| 13 | Admission unchanged | `verify_admission_order: arrival` | grants are in arrival order |
| 14 | Narrowed footprint | `order=priority`; two waiting branches forked before a landing that touched file F; neither changed F | they have no same-files link; a git error on one is a probe error |
| 15 | Pin reaches a prerequisite | task B pinned, depends on unfinished A (low tier); critical work is ready | A dispatches before the critical work; `effective_priorities[A] == 'pinned'`; `pin_queue` lists B only |
| 16 | Pin order inside the tier | B (order 2) and C (order 1) pinned, both with ready prerequisites | C's prerequisite precedes B's at the scheduler and at both later gates |
| 17 | Reservation stays bounded | B pinned with three unfinished prerequisites, all lock-blocked | exactly one holds a pin reservation |
| 18 | Pins do not stop fairness | a pinned task is lock-blocked; a critical task is passed over twice | the critical task holds a tier park |
| 19 | Recorded pin queue | a recorded pin queue with no prerequisites and no park borrow involved | the pinned tasks are tried in the same order; where two park-blocked pins could borrow, the lower `pin_order` now goes first |
| 20 | `pinned` is never stored | `set_task_priority_override(boost_tier='pinned')`; `submit_task` or `update_task` with `priority='pinned'` | each is refused with a message naming `pinned=True` |
| 21 | Restart | a pinned request and three low-tier requests wait; the orchestrator restarts | the first pick after start, before any tick, is the pinned request |
| 22 | Paused scheduler | the scheduler is paused; a waiting request's task is pinned | within a minute it is ordered at `pinned` |
| 23 | Promoted background leg | `verify_admission_order: priority`; a background leg is promoted while a critical, a high and a medium task leg wait | it is granted after the critical leg and before the high and medium legs |
| 24 | Inherited prerequisite and the watchdog | an inherited `pinned` prerequisite is lock-blocked for 200 dispatching scans | it gets no automatic pin and its refusals are not counted |
| 25 | Broken lookup | the lookup returns no tier for ten consecutive picks | one L1 is filed |

## 8. Decomposition plan

Greek labels are this plan's; ids are assigned at decompose. "Signal" is what an operator observes.

| Label | Task | Touches | Signal or what it unlocks | Needs |
|---|---|---|---|---|
| **α1** | `TaskPriority` with `base_tier`, `pin_order`, `age_anchor`; `PINNED`, `GATE_RANK` and `order_key`; the lookup returns `pinned` for an own pinned row and answers before the first tick and while paused; the unknown-tier streak; the `merge_pool` config file; the new `merge_queued` keys | `orchestrator/task_priority.py`, `scheduler.py` (the lookup, its start-up fill, the paused re-read), `orchestrator/merge_pool_facts.py`, new `orchestrator/merge_pool_config.py` mounted in `config.py`, tests | `merge_queued` rows carry `base_tier`, `age_secs` and `age_from` (`enqueue` until α2); a request queued after a restart and before the first tick records tier `pinned` for a pinned task (row 11's recorded fact). Unlocks β, γ, θ2 | 9908 |
| **α2** | The first-eligible table, its status-read cleanup and its seed; the lookup returns `age_anchor` | `eligibility_clock.py`, `scheduler.py` (the cleanup read, the anchor in the lookup), a seed script, tests | Rows 7, 8 and 10 as recorded facts: `merge_queued.age_from` is `anchor` for a task the scheduler has ranked; the seed reports how many rows it wrote from each source | α1, 9636 |
| **β** | Verify admission orders by `order_key` behind `verify_admission_order: priority`, with the promotion period and the promoted background leg's place (task 8261, its text replaced) | `verify_admission_gate.py`, `verify_admission_record.py`, `verify_admission_watch.py`, `task_verify_dispatch.py`, `verify_hosts.py`, `verify.py`, `config.py`, a defaults script, the verify-admission PRD's delta note | Rows 12, 13 and 23; `verify_admission_plan` rows carry tier, `base_tier` and age; the script prints the project's 95th percentile | α1 |
| **γ** | The pool order behind `merge_pool.order`: one order function and its readers, the narrowed footprint under `priority`, `bounce_cap`, the three delta notes | `merge_lane/intake_buffers.py`, new `merge_lane/pool_order.py`, `suffix_graph.py`, the order's readers in `merge_lane/worker.py`, `orchestrator/merge_pool_facts.py`, the three earlier PRDs | Rows 1 to 6, 9, 14, 21 and 22; with `order=priority`, `merge_dequeued` shows the pick first among `waiting` by `rank` and `age_secs`. `test_merge_queue.py::TestAgingPickOrder` stays valid for `arrival` | α1, 8528 |
| **ι1** | The order readout | a script under `scripts/` and its tests | The script reads task 9908's recorded picks and prints: picks that would differ under `order_key`, how often a higher-ranked request waited, for each recorded pick how far down the priority order the picked request was, and how many overtakes cross a recorded confirmed conflict (a pair with either request not `in_graph` counts as unknown). Rows from before α1 carry no `age_secs` and are counted and skipped; `no pick records yet` when there are none. Unlocks ι2 | γ, α2 |
| **ι2** | The human gate that flips `merge_pool.order` | a deterministic gate task; no code | The gate escalates to Leo with ι1's readout attached, run on the live event store, and the instruction to set `merge_pool.order: priority` by config reload | ι1; γ observed running (the readout prints picks recorded after γ); 8532 has read |
| **θ2** | `pinned` in the shared vocabulary, with its refusals | `shared/task_priority.py`, `orchestrator/task_priority.py`, fused-memory `tools.py` and task backend, dashboard validator | Row 20 | α1, 6718 |
| **θ1** | `pinned` inside the scheduler: the walk returns base tier and pin order, the sort puts the tier first, fairness unchanged, the pin pass's duties from one loop, the bounded reservation | `scheduler.py`, `pin_reservation.py`, `shared/scheduler_state.py`, the dashboard's scheduler view, scheduler tests, the scoring PRD's delta note | Rows 15 to 19 and 24; the dashboard shows inherited `pinned`; the state snapshot carries pin orders and base tiers, so the start-up fill restores an inherited pin with its order | θ2, 9683 |
| **κ1** | Integration gate for the two later gates | a boundary-test module driving admission and the lane together | Rows 1 to 14 and 21 to 23, 25 against the real modules | β, γ, α2 |
| **κ2** | Integration gate for the pin tier | the same module, extended | Rows 15 to 20 and 24, with a pinned task's prerequisite ordered first at all three gates | κ1, θ1 |

Notes for decompose:
- Every leaf that touches a function pinned in `orchestrator/tests/merge_lane_ratchet_baseline.json` names the
  authorised-raise path in its text.
- γ and task 5040 both touch `merge_lane/worker.py`; γ lists both the current file and the file 5040 moves its
  readers to, if any, in its declared files.
- "Observed running" and "has read" are delivered checks or deterministic gate tasks, not dependency edges.
- At decompose (2026-11-13): replace 8261's text with β's; add the dated notes to 9188, 7921, 9679, 6328, 3824,
  9636, 9142, 5621, 6124 and 6944.
- κ1 and κ2 are the signal leaves.
- Critical path to the priority pick: 9908 → α1 → γ (after 8528) → ι1 → ι2 (after α2 and after 8532 has read); β and α2 in
  parallel. Until α2 lands, age at the later gates is counted from the request's first enqueue or the plan's arrival. The pin tier at the
  later gates arrives with α1; inside the scheduler it waits for the restructuring chain.

## 9. Out of scope

- Batches chosen from evidence, suspects, conviction (part 2).
- The scheduler's order within a tier (D3); a promotion backstop at the scheduler; per-tier slot caps.
- Lock reservations beyond the bounded pin reservation; releasing locks before landing (ruled out by Leo).
- Deriving the high lane from the tier; choosing a verify host by priority.
- Priorities per PRD.
- The other projects' adoption beyond the defaults the script prints.

## 10. Open questions (tactical)

1. **Whether the seed prefers `lock_acquired` or `task_started`.** Suggested: the earlier of the two. Decide in α2.
2. **How `pool_order()` learns task 8528's suspect list.** Its planned form is an ordered list of task ids; γ maps
   each to its first buffered request, as `take_buffered` does. If 8528 lands a set, γ orders its members by buffer
   position, which is where 8528 put them. Decide in γ.
3. **Whether a promoted waiter's `rank` is recorded raw or promoted.** Suggested: both, as `rank` and
   `base_rank`. Decide in γ.
4. **Whether γ splits.** The footprint and the bounce cap separate cleanly from the pool order if γ outgrows the
   sizing band. Decide at decompose.
