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
- **Sibling task 10399's interim fix** (pin_order 1 plus `reserve_now` on
  filing) orders lock acquisition but still needs a free permit, so it cannot
  break the cycle alone. With this task in place it stays useful for lock
  precedence.

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
- Known false alarm, until follow-up ticket `tkt_0RX39GHVZQB8HE6J4X8944DD3W`
  lands: the dashboard's E12 parity alarm
  (`dashboard/src/dashboard/data/burndown.py::compute_parity_alarm`) still
  takes `max_concurrent_tasks` alone as the cap. A fix running on the reserve
  therefore logs "Concurrency cap breached" and records a breach row, for
  example 37 live against 36. A breach of exactly
  `main_health_fix_reserved_slots` while `scope` is `main_health_fix` is the
  reserve working as designed, not an over-dispatch.

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

- A fix whose planned files overlap a merge-waiter's held modules, or a pin
  head's reservation, still waits on module locks. That lock precedence is owned
  by 10399 item (3) and by 8795. When the pin head is one the PSI hold defers,
  that wait has no bound; DA-D10 of `docs/prds/dispatch-admission-load-cap.md`
  records it.
- Resolved by task 10600: an admitted fix is exempt from the PSI admission
  gate's hold on every tick, in both dispatch loops. The rationale and the load
  bound are DA-D10 of `docs/prds/dispatch-admission-load-cap.md`.
- While the general pool is full and the reserve is free, a tick runs every
  `idle_poll_secs`, at the same cadence and cost as an idle tick
  (INV-8 `loop-thread-occupancy-bounded`). A general slot that frees during
  such a restricted tick is refilled by an unrestricted tick at once, not
  after the next poll.
