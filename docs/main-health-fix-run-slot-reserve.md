# Main-health fix run-slot reserve (task 8480)

A main-health fix task can always get a run slot, even when every
`max_concurrent_tasks` slot is held. The dispatch loop also takes a slot before
it picks a task, so no picked task holds module locks while it waits for one.
Code: `orchestrator/src/orchestrator/run_slots.py`, wired in
`orchestrator/src/orchestrator/harness.py::Harness.run` and
`orchestrator/src/orchestrator/scheduler.py::Scheduler.acquire_next`.

## 1. The cycle

A workflow keeps its run-slot permit until `_run_slot` ends, including while it
waits in MERGE on the merge queue. When main is red:

1. Merge-waiters hold every `max_concurrent_tasks` permit.
2. The loop picks a task with `acquire_next`, then parks in `sem.acquire()`. The
   picked task's module locks stay held and its status stays pending.
3. No slot frees until a merge lands. No merge lands until main is healed. The
   heal needs a slot, and the parked loop never calls `acquire_next` again, so
   it never even sees a heal filed after it parked.

Two specimens:

- **2026-11-04**: 18 slots. Heals 8468, 8474 and 8483; esc-6484-3.
- **2026-11-15**: 36 slots. Heal 10239 was pinned at pin_order 1 and took its
  lock at once, then sat parked behind `sem.acquire` for 2.6 h. Signature
  owner: 10312.

## 2. Options weighed

- **(a) A reserved run slot that only a main-health fix may take: TAKEN**,
  together with slot-before-pick.
- **(b) A separate ceiling for merge-waiters: REJECTED.**
  - It changes what `max_concurrent_tasks` means for every task, and for every
    resource sized from it.
  - It re-creates the same hold one level down. A waiter whose merge fails
    re-enters execute and must re-take an execute slot from inside the
    workflow, while a red main fills the merge-waiter ceiling.
  - It couples workflow phase transitions to harness permits (heuristic 7).
- **(c) The high merge lane: ALREADY IN PLACE** for the fix owner.
  `main_health_fix_owner.py::build_main_fix_task_arguments` stamps
  `merge_lane: high`. It acts after dispatch, so on its own it cannot get a fix
  dispatched.
- **Sibling task 10399: precedence on filing, LANDED** (§7). The filer pins
  the fix at the head of the queue with `reserve_now`. That orders dispatch and
  lock acquisition, but it still needs a free permit, so it could not break the
  cycle alone. The reserve supplies the permit; the pin supplies precedence.

## 3. Admission

A task may take the reserve when it carries a non-blank
`metadata.main_health_signature`, read through
`orchestrator/src/orchestrator/main_health_fix_owner.py::is_main_health_fix`.
That key is structured, and it is written by the one filer that guarantees an
owner for each confirmed break (heuristics 11 and 12).

Steward-triage and info-l0-router heals (8468, 10239) carry no structured
marker, so they are not admitted. Routing on their prose would be a
meaningful-string parser. Task 8795 owns giving them a marker.

The consequence, plainly: on 2026-11-15 the admitted owner 10312 would have
dispatched on the reserve, unless a dependency on an unadmitted heal held it
back. A human did make 10312 depend on 10239. From then on 10312 could not
dispatch before 10239 was done, and 10239 still needed a general slot.

## 4. Holds (INV-7 `holds-owned-and-bounded`)

| Hold | Owner | Bound |
|---|---|---|
| The loop's wait while every slot is held | any live slot ending | `idle_poll_secs`; it holds no task and no lock |
| The reserve held by a fix | that fix's workflow | its existing limits and the `SlotLease` force-release; a restart rebuilds `RunSlots` from config |

The reserve's size, `main_health_fix_reserved_slots` (default 1), bounds heals
to that many slots beyond general capacity, however many signatures are open. A
second fix waits for the first to free the reserve, or competes for a freed
general slot like any other task.

The hold this task removed: a picked task's module locks, held while it waited
for a permit.

Operator visibility:

- `scheduler_tick.data.scope` is `main_health_fix` on a tick run against the
  reserve, and `any` otherwise.
- Each dispatch on the reserve logs an INFO line naming the task.
- The dashboard's E12 parity alarm judges live in-progress against
  `max_concurrent_tasks + main_health_fix_reserved_slots`, read by
  `dashboard/src/dashboard/data/orchestrator.py::read_run_slot_capacity`, so
  a fix running on the reserve is not recorded as a breach.

`main_health_fix_reserved_slots` is restart-only, like `max_concurrent_tasks`.
The warm-lane pool is sized from every run slot:
`max_concurrent_tasks + main_health_fix_reserved_slots + git.spare_warm_lanes`.

## 5. Heuristics applied

From `docs/code-quality.md`:

- **3**: pool capacity lives in `RunSlots`, admission in `DispatchScope`, and
  ranking is untouched in `Scheduler`.
- **9**: `RunSlots` hides two semaphores behind `free_scope` and `claim`.
- **10**: admission is checked twice through one predicate: by the scheduler's
  candidate filter, and again at `RunSlots.claim`, which raises
  `RunSlotInvariantError` on a breach.
- **11** and **12**: one structured marker, read in one place.

## 6. Residuals

- A fix no longer waits behind another pin head's reservation, a PSI-deferred
  one included: the filed fix is the pin head (§7). What remains is a fix whose
  planned files overlap modules a merge-waiter holds while the normal merge lane
  is halted for the heal. That holder cannot finish before the heal lands, and
  nothing may preempt it. Today only the starvation watchdog's `park_blocked`
  L1 names it for a human, after `park_blocked_age_secs` (default 4 h). Raising
  that case at once for a main-health fix is filed as a follow-up (ticket
  `tkt_0RXGKM3ZHJM1P4T50NG5AWKG1J`).
- Resolved by task 10600: an admitted fix is exempt from the PSI admission
  gate's hold on every tick, in both dispatch loops. The rationale and the load
  bound are DA-D10 of `docs/prds/dispatch-admission-load-cap.md`.
- While the general pool is full and the reserve is free, a tick runs every
  `idle_poll_secs`, at the same cadence and cost as an idle tick
  (INV-8 `loop-thread-occupancy-bounded`). A general slot that frees during
  such a restricted tick is refilled by an unrestricted tick at once, not
  after the next poll.

## 7. Precedence on filing (task 10399)

What is written, and by whom:

- `orchestrator/src/orchestrator/main_health_fix_owner.py::MainFixTaskFiler`
  grants precedence to an owner the scheduler can dispatch next. That is one
  it has just filed, a half-filed one that reads back pending once its commit
  is re-run, or an adopted owner that is pending.
- It never grants an in-progress, blocked, merge-deferred or hand-carried
  owner. On those, `reserve_now` would be refused and consumed at once.
- The grant goes through
  `orchestrator/src/orchestrator/scheduler.py::Scheduler.grant_main_health_fix_precedence`
  to `orchestrator/src/orchestrator/overrides.py::OverrideStore.pin_main_health_fix`.
- The row is owned by `OverrideSource.MAIN_HEALTH_FIX` (`main_health_fix`). It is
  pinned at the head, with `reserve_now` armed and no TTL.
- A failed grant is logged. It never costs the break its fix owner.

Head insert: the fix takes pin_order `min(1, lowest other order)`, and every
other pin of the project moves back one place, keeping its relative order.
Among several fixes the newest is first. A fix already pinned keeps its place,
and a re-grant only re-arms `reserve_now`, so the many adoptions a red main
produces never churn the queue.

Ownership rules:

- An operator's row for the task stands. It is never written.
- An operator's unpin of the fix stands. The owner's unpinned row is left as it
  is.
- A starvation-watchdog row is taken over and its TTL dropped, so
  `clear_expired` cannot later remove the fix's pin.

Why the in-process store, not the `set_task_priority_override` MCP tool: both
write `scheduler_overrides.db`, but the tool stamps `source='operator'`, so a
machine pin would pass for an operator's. It also refuses a `pin_order`
collision, the normal case once an operator has pinned anything.

Lock precedence: the fix parks behind a held lock and never preempts it.
"Preemption / force-release of a HELD lock" is explicitly rejected in
`plans/scheduler-dispatch-scoring-and-lock-layer-prd.md` §7.

- As pin head, a lock-blocked fix reserves its modules at pin rank, so it takes
  a released module ahead of every operator pin, tier park and critical
  candidate.
- `Scheduler._bound_pin_reservations` keeps the first
  `pin_reservation_max_active` reservation owners in queue order. The fix is
  first, so it is kept ahead of a PSI-deferred operator head, which removes the
  wait that the DA-D10 residual of `docs/prds/dispatch-admission-load-cap.md`
  describes.
- A fix filed with no files locks only its fallback key `task-<id>`. It meets
  contention only after plan refinement persists real files and requeues it.

The clear: the tick's override GC (`OverrideStore.clear_terminal`) deletes the
row when the fix is done or cancelled, as it does for every owner. The pin is
not released on dispatch, so a fix whose workflow requeues it to pending still
dispatches first.
