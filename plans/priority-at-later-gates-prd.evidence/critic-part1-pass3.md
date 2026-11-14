# Critic, third pass: `/tmp/prd-merge-pool/prd1-priority.md`

Read-only, 2026-11-13. Main is now `a8bd0b5eb1` (three commits past the draft's `0bb6dd2b96`; none touches a cited
file). Task 9908 has **not** merged: status `deferred`, branch `task/9908` now at `ae8ef8298b` (rebased; one
docstring change since `8c6d25dad6`). Task store opened `?mode=ro`. Nothing written but this file.

Counts: **7 answer-changing (A)**, **16 minor (B)**. Verdict: **not clean**.

## Second-pass items as folded

| Item | In this draft | Verdict |
|---|---|---|
| A1 first-eligible table | D2, 6.2, α | Folded as I wrote it. Three parts of my text do not hold against the code: **A7**. One gap I missed: **A1** |
| A2 rank of `pinned` in α, θ2 before θ1 | D1, section 8 | Correct |
| A3 pin tier on an unchanged sort | D6, rows 15 to 19 | Correct in structure. Two behaviours still change that the draft calls unchanged: **A5**. Reader list: B2 |
| A12 order readout and gate | ι | Correct. One measure in my text cannot be produced: B6 |
| A13 κ split, 5042 edge | κ1, κ2 | Correct. κ1 needs nothing from ι: its rows set `order=priority` themselves |

---

## A. Answer-changing

### A1. After every restart, and for as long as the scheduler is paused, the lookup knows no tier and no pin

**Draft.** 6.1: "An unknown tier or a failed lookup gives the rank of `medium`; the caller counts it on its event."
6.2: "`Scheduler.task_priority` returns the anchor from memory."

**Wrong.** On the 9908 branch, `Scheduler.task_priority` reads `_last_effective_priorities` and
`_prev_overrides_snapshot`. Both are empty until the first tick's `_phase_compute_priorities`; its docstring says
"Both are unknown (None) before the first tick". The harness launches the merge worker's loop
(`_launch_merge_worker_loop`) before its first call to `acquire_next`, and the worker picks from the recovered
journal as soon as a host is free; nothing makes it wait for a tick. So the first picks after every restart rank
every waiter `medium`: a low-tier request can be taken ahead of a pinned one, for a verify of 20 to 60 minutes.
Restarts are routine (the fleet redeploy clock is 8 h). A paused scheduler is worse: `Scheduler.acquire_next` returns
before any phase runs, so after a restart in the paused state the tiers are never filled. Other projects'
schedulers have been halted for days while their lanes still take hand-carried merges.

**Add to 6.2:**
> At construction the scheduler fills its tier map from the last `scheduler_state.json` it wrote and its pin rows
> from the override store, so the lookup answers before the first tick and while the scheduler is paused. While
> paused it re-reads the override store at most once a minute, so a pin set during a pause is honoured at the later
> gates. A tier that comes from the file is as old as the last tick before the stop; that is accepted.

Add a row: "Restart: a pinned request and three low-tier requests wait; the orchestrator restarts; the first pick
after start, before any tick, is the pinned request." Add both reads to α's work.

### A2. The watchdog's pins never reach the later gates; row 11 cannot pass

**Draft.** D5: "operator pins and the watchdog's automatic pins alike, so both later gates honour pins at once." Row
11: "the watchdog pins a starving low-tier task: its request and its verify legs are ordered at tier `pinned`". Open
question 4.

**Wrong.** A watchdog pin is deleted the moment its task dispatches: `Scheduler._settle_starvation_on_dispatch`
calls `_release_watchdog_pin` at both dispatch sites, and `_release_unpending_watchdog_pins` deletes it whenever the
task is not pending. By the time the task has a verify leg or a merge request, the row is gone. Operator pins stay
until cleared.

**Replace.** D5: "... for a task whose own override row is pinned. In practice that is an operator pin: the
watchdog's automatic pins are released at dispatch and never reach the later gates." Row 11: "Operator pin on a
running task: a low-tier task in progress is pinned; its next verify leg and its merge request are ordered at
`pinned`; the recorded `base_tier` is `low`." Delete open question 4. If Leo wants a task the watchdog rescued to
keep its place through verify and merge, that is a new rule (keep the pin until done) and belongs in the hand-back.

### A3. With `order=arrival` the pick still changes, inside task 8532's reading window

**Draft.** 6.4: "In both orders the conflict graph's footprint is each branch's own changes". O3: "... the same as
before this PRD ... except where the narrowed footprint removes a same-files link that was not a real overlap."
6.4 also puts task 8528's suspects first in both orders.

**Wrong.** The exception is the rule. Today the footprint links every pair (105 of 105 at the study's snapshot), so
`MergeIntake._clique_minimal_visible` returns the request with the oldest first submission: a request that was
bounced and re-submitted keeps its seniority although it re-enters at the tail of the buffer. With the narrowed
footprint (17 of 105) it returns the head of the buffer unless a truly overlapping older request sits behind it. A
re-submitted request then waits behind everything that arrived while it was away. That is a different arrival-mode
pick, live from γ's deploy, while task 8532 is reading the lane "with `order=arrival`" for its seven-day verdict on
task 8528. Row 6 is written to avoid the case ("with no false same-files links"). Putting 8528's suspects first in
arrival mode is a second change: 8528 already places them physically, and today's rule runs over that.

**Replace** the first and fourth bullets of 6.4 and O3 with:
> - With `merge_pool.order: arrival` the pick is today's code path, untouched: the lane buffers as they stand
>   (task 8528's requeue included), the same-files rule, and the footprint computed as today. `pool_order()` returns
>   that order.
> - With `priority`, `pool_order()` is: the high lane, by `order_key`; then the requests of task 8528's suspect
>   list in its order; then the rest by `order_key`. The same-files rule is not applied, and the conflict graph's
>   footprint is each branch's own changes (`GitOps.get_branch_changed_files`), where a git error is a probe error.
> - **O3.** With `order=arrival` the sequence of picks is the same as before this PRD for the same arrivals.

Row 6 drops its qualifier. Row 14 gains `order=priority`. `bounce_cap` may stay in both orders: the cap of 3 has
never bound.

### A4. Verify admission: a promoted background leg has no place in the new order, a recorded decision is reversed unnamed, and task 8261's body says something else

**Draft.** 6.3: "`_pick` takes the minimum of `(0 if promoted else role_class, order_key(...), seq)`. Background legs
keep their class rule and their existing promotion; they are ordered among themselves by arrival." D9: "Waiting is
counted from ... the leg's arrival". Section 5: "amend at decompose".

**Wrong.**
1. A background leg promoted after `verify_admission_background_promote_secs` joins class 0 and is compared with
   task legs by `order_key`. It has no task, so its rank is `medium` and its age is hours since arrival, against
   task legs aged in days since first eligible. It sorts behind every pinned, critical and high leg and behind
   nearly every medium one. The admission plan's decision 3 exists to prevent exactly this ("strict classes without
   promotion would starve the sweep"); today a promoted leg keeps its arrival stamp and "can never" be overtaken by
   a later task. Background legs are 1,480 of 3,297 legs in the recent window.
2. That plan's decision 2 is "no task-level aging (a repeat failer must not own the head of the queue)". Under
   `order_key` a task that fails its verify and comes back keeps its first-eligible age, so within its tier it does
   return to the head. R4 requires that, but the draft amends only decision 7, and task 8261's text says decision 2
   "stays untouched".
3. The same plan fixes arrival per plan, not per leg, because a plan's fifth module reaches the gate late. "The
   leg's arrival" as the fallback age and the promotion clock sends that leg to the back.
4. The gate is in-process (its module docstring: "In-process ordered admission gate in front of the task-verify
   flock slot"), so a lookup can be called. But `_pick` receives only an `AdmissionPolicy` built from config by
   `verify.py::admission_policy`, and there are two gate sites: `gate_for(slots_dir)` and one gate per remote host
   in `verify_hosts.py::VerifyHosts`. The draft does not say how the lookup arrives. Ordering holds within one
   orchestrator process; between processes the flock's poll race still decides, as today.
5. Task 8261 now has a body (key `(class, arrival + tier_rank x K, seq)`, value `priority_aged`, K as a blend), an
   amendment (K as backstop, 12 h, "pins ... not in scope") and a correction (4 h, measured). A fourth appended
   note leaves an implementing agent four layers to reconcile.

**Replace** 6.3's first bullet with:
> `AdmissionOrder` gains `'priority'`. Under it `AdmissionGate._pick` takes the minimum of `(class,
> order_key(...), seq)` over the waiters. A task leg's priority is read through a lookup carried on its
> `AdmissionPlan` (`AdmissionPlan.begin(..., priority=...)`, set where the plan is built in
> `task_verify_dispatch.py` and `verify_admission_record.py`); both gate sites use it. A background leg has class 1
> and is ordered by plan arrival. Once promoted it has class 0 and is ordered as the oldest `high` waiter: it waits
> only for pinned and critical legs. The fallback age and the promotion clock are the plan's arrival.

Add to D9: "... or the plan's arrival (verify admission)". Section 5, task 8261: "**replace** its description and
details at decompose with β's text (value `priority`, `order_key`, the promotion period); keep its id, pin and
dependents". β's delta note amends decisions 2, 3 and 7 of `verify-admission-fifo-prd.md`, and the hand-back to Leo
says in one line that a repeat failer keeps its age. β's files gain `task_verify_dispatch.py` and `verify_hosts.py`.

### A5. Two things D6 calls unchanged do change: the watchdog's count, and the order row 19 asserts

**Draft.** D6: "The pin pass's other duties run from the one loop". Row 19: "a recorded pin queue with no
prerequisites involved: dispatch order of the pinned tasks is unchanged".

**Wrong.**
1. The watchdog. `LockRefusals` counts "scans that refused the task its locks while a lower-ranked task dispatched"
   (`_lock_refusals.note_scan(refused)` in `Scheduler._phase_select_scored`). The pin pass does not count. Once
   pinned-tier candidates head the one loop, every lock-blocked one is above every winner and is counted on every
   scan that dispatches. An inherited prerequisite has no override row of its own, so at the threshold
   (`starvation_watchdog.classify`, gate `lock_refused`) the watchdog gives it an automatic pin. Those take the
   three slots from tasks that are starving with no pin at all, and feed the cap alarm task 9636 is quietening.
2. Row 19. Today a pin that fails its plain acquire in the pin pass gets a second try in the scored loop, at its
   score position, and that try includes the park-borrowing retry (`_backfill_admission`), which the pin pass has
   not. So today two park-blocked pins can dispatch in score order, not pin order. In one loop the first pin gets
   the borrowing retry first. The new behaviour is the better one; the row as written is not true.
3. A smaller one: today a tick in which a pin dispatches ends before the scored loop, so the best-scored candidate
   gets no skip. In one loop it is passed over and gets one.

The readers hold up otherwise. The park-borrowing retry, the load hold and `reserve_now` read no tier that changes
(B2), and "best by score alone" is right when that candidate is also pinned-tier: it earns tier parks at its base
tier exactly as a pinned task that tops the score does today.

**Add to D6:** "The watchdog neither counts lock refusals for, nor pins, a candidate whose tier is already `pinned`.
A pinned-tier candidate gets the park-borrowing retry in pin order. The best-scored candidate accrues a skip on a
tick where a pinned-tier candidate dispatches." Row 19: "a recorded pin queue with no prerequisites and no park
borrow involved: the pinned tasks are tried in the same order; where two park-blocked pins could borrow, the lower
`pin_order` now goes first." Add a row: "an inherited prerequisite is lock-blocked for 200 dispatching scans: it
gets no automatic pin."

### A6. γ waits for β only to mount a config file, and β is the most lock-starved task in the store

**Draft.** γ needs "α, β (same `config.py`), 8528". Critical path: "9908 → α → β → γ".

**Wrong.** β is task 8261. It has been refused its files 52 times in 9.9 days (`config.py`, `verify.py`), is pinned
and in the high lane, and is still pending. γ's only business in `config.py` is mounting the `merge_pool`
sub-model. That edge puts the priority pick behind 8261 for two lines.

**Replace.** α creates `orchestrator/merge_pool_config.py` (`order`, `promote_after_secs`, `bounce_cap`) and mounts
it in `config.py`; γ reads it and does not touch `config.py`. γ needs α and 8528. β needs α. Critical path:
9908 → α → γ (after 8528) → ι (after 8532 has read); β in parallel. `config.py` joins α's files, so β's edge to α
already serialises the two.

### A7. 6.2: the delete rule needs a read the tick does not have, "cancelled then pending" follows only if it is seen, and the seed cites a record that does not exist (mine)

**Draft.** 6.2: "A `first_eligible` row is deleted only when a status read shows the task done or cancelled ... so
`cancelled → pending` starts a new age." Seed: "from their first `lock_acquired` or `task_started` event after their
last `cancelled → pending`".

**Wrong.**
1. A tick's `status_map` is built from the active fetch (`get_tasks(statuses=ACTIVE_TASK_STATUSES)`). A done or
   cancelled task is simply absent from it. The override GC treats absent as terminal
   (`_phase_override_snapshot_gc`), which is the shortcut 6.2 forbids. The rule needs its own read, and the draft
   does not name one. `Scheduler.get_statuses(ids)` exists and returns `(statuses, error)`.
2. The new age follows only when some tick sees the task while it is cancelled. A cancel and re-pend inside one
   tick interval, or during a pause or an outage, is never seen, and the old age continues. Row 10 fails there.
3. No event records a status change: the event store has `lock_acquired` and `task_started` and nothing for cancel
   or re-pend. "After their last `cancelled → pending`" cannot be computed.

**Replace** the first two bullets of 6.2 with:
> - `EligibilityClock.observe` inserts into `first_eligible` as well as the watchdog's table, INSERT OR IGNORE.
>   `forget` clears only the watchdog's table. Each tick, for rows whose task is absent from the active fetch, the
>   scheduler makes one `get_statuses` call; a row is deleted when that read answers done or cancelled, and kept on
>   an error or no answer. A task cancelled and re-pended between two such reads keeps its old age; that is
>   accepted.
> - At deploy the table is seeded once, before the first tick: for tasks that are candidates now, from their
>   `eligibility_anchors` row; for tasks already dispatched, from their first `lock_acquired` or `task_started`
>   event, else `updated_at`. This under-ages, except for a task cancelled and re-pended before the seed. Seeded
>   rows are marked.

A2 becomes "The stamping adds no task-store write to a tick, and at most one status read." Row 10 gains "and a tick
runs while it is cancelled". Open question 1 drops "after the last `cancelled → pending`".

---

## B. Minor

1. **Task 9908's facts.** The file is `orchestrator/src/orchestrator/merge_pool_facts.py`, not under `merge_lane/`
   (α and γ "Touches", section 4). The header's branch sha is now `ae8ef8298b`. Its last commit redefines `in_graph`:
   the request "was in the suffix at the graph's last recompute, though its pairs may still be unassessed". ι must
   count a pair with either request not `in_graph` as unknown, not as no conflict.
2. **D6, readers of the tier map.** On main: `_phase_reserve_now`, `_complete_earned_fairness_parks`, `_pin_queue`
   (then `_dispatch_pin`), `_score_candidates` (then `top_pri`, `_dispatched_priority`, the `reservation_used`,
   `task_skipped` and `lock_acquired` payloads) all need the base tier; `get_state_snapshot` and 9908's
   `Scheduler.task_priority` need the published tier; the dashboard reads the snapshot. D6 names most. Say it as a
   rule: the internal map stays the base tier, a second map holds pin orders, and only the snapshot and the lookup
   combine them. Then no internal reader changes.
3. **D6, "exactly when `pin_order` is set".** A pinned row with no stored order counts as 0 today (`_pin_queue`).
   Keep that. Say inheritance is through every unfinished dependent, direct or not.
4. **6.4, readers.** All listed readers exist. Three clarifications: (a) `_rebucket_relaned_requests` is a writer
   that the pick calls first; leave it as it is, and drop it from the reader list; (b) `unfrozen_suffix`,
   `snapshot()`, the graph and the pick records list reserved requests and halted lanes today, and `snapshot()`
   also lists arrivals not yet drained: say `pool_order()` returns all buffered requests and the pick filters;
   (c) the conflict graph's debounce signature is the ordered ids, so a promotion would force a recompute: key it
   on the set. Not order readers, no change needed: `_decided_predispatch_ids`, `relane_waiting`,
   `_dissolve_coalesce_trains_if_halted`, `merge_queued.position` (arrival position; say it stays that).
5. **6.4, 8528's suspect list.** Its planned form is an ordered list of task ids
   (`ChainPolicyState.suspects`, plan D6: "ordered task ids of the links of the last failed tip"). γ maps each to
   its first buffered request, as `take_buffered` does. If 8528 lands a set, γ orders its members by buffer
   position, which is where 8528 put them. Put both sentences in open question 2.
6. **ι, "waits by tier before and after".** "After" needs a simulation of the lane, which nothing committed
   provides. Replace with "for each recorded pick, how far down the priority order the picked request was". Add:
   rows from before α carry no `age_secs` and are counted and skipped. (Mine.)
7. **Dropping the same-files rule.** `two_layer_invariants` checks the graph only (textual pairs inside footprint
   pairs; conflicts with main inside the nodes) and is unaffected; textual pairs are probed only among footprint
   pairs, so the first holds by construction under the narrowed footprint. The tests in
   `test_merge_queue.py::TestAgingPickOrder` stay valid for `arrival`. Say so in γ.
8. **Section 5, missing rows.** Task 5139 (`_admission_slot` role gate; `verify.py`, `config.py`, high) and task
   9142 (moves the admission gates; `verify_admission_gate.py`) share β's files. Tasks 5621, 6124 and 6944 share
   γ's footprint and bounce code in `suffix_graph.py`.
9. **Section 5, task 9636.** "Default α first" adds a dependency behind a hand-carry to a task Leo ruled on and
   that reverts the interim skip threshold. Put the direction to him, or default to 9636 first.
10. **Section 5, task 7921.** "Whichever runs second keeps the other's skip" has no owner if 7921 runs second: its
    text predates `pool_order()`. The dated note must say where its skip goes (in the pick's filter).
11. **D10, "for main repair only".** Task 8261 itself carries `merge_lane: high`. Hand relaning of ordinary work is
    in use. Say whether it stays once pins reach the lane, or leave "only" out.
12. **Row 20.** `submit_task` takes `priority` too. Add it.
13. **Row 7.** "A `replace`-mode metadata write occurs" no longer bears on anything; remove it.
14. **γ's size.** Pool order, footprint, bounce cap, record keys and three notes over about 13 files is at the top
    of the band. Acceptable; if it grows, the footprint and the cap split off cleanly.
15. **Unknown-tier count.** It is recorded and nobody is told. With A1 in place a streak of unknown tiers means the
    lookup is broken: ten consecutive picks with an unknown tier file one L1.
16. **Same-file edges.** α and θ1 on `scheduler.py`: covered through θ2. α and γ on `merge_pool_facts.py`: covered.
    α and 9636 on `eligibility_clock.py`: B9. β and γ on `config.py`: removed by A6.

---

## Invariants: hits that remain

| Invariant | Mechanism | Item |
|---|---|---|
| `corroborate-before-acting` | the pick acts on a tier map that is empty after a restart; the delete rule has no read | A1, A7 |
| `no-silent-fail-soft` | every waiter ranked `medium` after a restart looks like an ordinary pick | A1 |
| `holds-owned-and-bounded` | a promoted background leg loses its bound | A4 |
| `one-fact-one-home` | task 8261's body against its amendments; the admission plan's decision 2 | A4 |
| `guards-exercise-behaviour` | row 11 cannot pass; row 19 asserts what is not so; row 6 avoids the changed case | A2, A5, A3 |
| `storm-escape-required` | the unknown-tier count | B15 |

## META

Not yet, with the three [Leo] decisions confirmed as recommended. All three stand. What matters most: the lookup
must answer after a restart and during a pause (A1); `order=arrival` must really be today's lane while 8532 reads
it (A3); and verify admission needs its three open points settled (A4). A5 and A7 are corrections to text I wrote;
A2 and A6 are one-line fixes with large effect.

Not clean.
