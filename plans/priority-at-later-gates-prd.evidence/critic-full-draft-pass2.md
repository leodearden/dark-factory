# Critic, second pass: `/tmp/prd-merge-pool/draft.md` v2

Read-only, 2026-11-13. Code on main `0bb6dd2b96`; `task/9908` at `8c6d25dad6`; task store and
`data/orchestrator/scheduler_eligibility.db` opened `?mode=ro`. Nothing written this pass except this file
(`git patch-id` was checked on hand-made diffs fed on stdin).

Counts: **14 answer-changing (A)**, **22 minor (B)**. Verdict: **not clean**.

Four of the A items (A4, A5, A10 and part of A13) are defects in replacement text I supplied in the first pass, which
v2 copied faithfully. They are marked "mine".

## 1. First-pass findings: how each was folded

| v1 item | In v2 | Verdict |
|---|---|---|
| A1 head and members | 6.4 | Correct. Residue: B1, B19 |
| A2 observation row | 6.3 | Folded as I wrote it, and my text has no case for a failed batch: **A4** |
| A3 8528's bisect, `order` switch | 6.7, D14, row 5 | Correct. Consistent with 8528's text (D6: "requeue the k links at the FRONT in aging-key order") |
| A4 own run, only waiting requests | 6.5, rows 11, 13 | Correct |
| A5 content id | 6.3, 6.6 | Folded; the id as specified is not rebase-stable: **A8**. Row 15 asserts a state my text made unreachable: B2 |
| A6 suspect rules | 6.5 | Folded as I wrote it; the no-mixing rule idles hosts and contradicts the deadline: **A5**. The reading of R7 changed without a flag: **A14** |
| A7 batch rule | 6.4, B2 | Folded; the result is that the risk rule can never bind and row 8 is impossible: **A7**. The table's depth is not defined with `mode=on`: **A6** |
| A8 shadow calibration | 6.7, ι | Correct. Residue: B12 |
| A9 pin reservation bounded | D7 | Correct as far as it goes; the rest of the pin tier cannot be built on an unchanged scheduler: **A3** |
| A10 credits | D3 [Leo] | Correct at the recommended value. No leftover of δ, the legacy score or credits |
| A11 where age starts | D2 [Leo] | Correctly raised; the recommended option cannot be built as written: **A1** |
| A12 design decision 8 | section 2, D2 alternative | Correct (moot at the recommended value) |
| A13 `pinned` from α | D5, α | Folded; α has no source for the rank of `pinned`: **A2** |
| A14 6821, 8261, 5623 | D18, section 5 | Correct |
| A15 9144 and 7921 seams, no prefix | section 5, 6.6 | Correct |
| A16 pair rule, row 11, I3 | removed; row 14; I2 | Correct |
| A17 main red, slow prior, storm | D15, 6.3, D16 | Slow prior correct. The main-red rule as I worded it cannot include the triggering failure: **A10**. The storm limit as worded stops ordinary failures: **A9** |

Dependency check asked for in 2(g): **no cycle.** None of the open tasks upstream of 5041, 5042, 8933, 6718, 9683 or
8528 is 8261, 9705 or 9908, and every other leaf is new.

---

## A. Answer-changing

### A1. The eligibility stamp cannot be written the way D2 and 6.2 say, and stamping "the first tick" throws away the ages that exist

**Draft.** D2: "A durable `metadata.eligible_since`, written once by the scheduler the first tick the task is a
dispatch candidate ... Cost: one write site in `scheduler.py`, and the metadata key." 6.2: "It is blessed and
machine-authored in `shared/task_metadata.py`. ... the backend re-attaches it after
`strip_machine_authored_metadata`."

**Wrong.**
1. The scheduler's only task write is `Scheduler.update_task`, which goes to fused-memory's `update_task`. That path
   strips every machine-authored key from the caller's payload (`strip_machine_authored_metadata`, applied in
   `SqliteTaskBackend.update_task`). A key that is machine-authored is one the scheduler cannot write. A new
   fused-memory tool, backend method and interceptor path would be needed; the draft names none.
2. Cost. The natural site is `Scheduler._apply_starvation_watchdog`, beside `self._eligibility.observe(candidates,
   now)`. The live store holds **2,516** candidate anchors for dark_factory. The first tick after deploy would make
   2,516 metadata writes, one MCP round trip each, inside the tick (a tick took about 55 s when last measured,
   11-06).
3. "First tick" discards what is known. Those 2,516 anchors have a median age of 295 h (maximum 297 h). Stamping
   `now` resets all of them to zero.
4. "Written once" is not guaranteed by the clock. `EligibilityClock.forget` runs at dispatch and whenever a task is
   not pending, so `observe` sees the task as new again after a re-pend. An agent keying the write on the clock's
   "new" set writes twice. Nothing in 6.2 says what enforces once.
5. The repair does under-age as claimed (a first `lock_acquired` or `task_started` is at or after first candidacy,
   and so is `updated_at`), with one exception: an event from before a `cancelled → pending` over-ages. But the
   repair must run before the first stamping tick, or the stamp gets there first.
6. A request whose task the scheduler never ranked (a hand-carried task is `deferred`; an interactive merge) has no
   anchor, so age 0: with `order=priority` it is last in its tier. Today it is taken in arrival order. Task 9908
   itself would land this way.

**Replace D2's recommended option with:**
> *(recommended)* **First eligible, kept by the scheduler.** The scheduler's eligibility store
> (`eligibility_clock.py::EligibilityClockStore`, already written every tick for every candidate) gains a second
> table, `first_eligible(project_root, task_id, at)`. `EligibilityClock.observe` inserts into both, INSERT OR IGNORE.
> `forget` (dispatch, re-pend) clears only the watchdog's table. A `first_eligible` row is deleted only when a status
> read shows the task done or cancelled, never because it is absent from a fetch; so `cancelled → pending` starts a
> new age. No task-metadata key and no fused-memory change. Time parked behind a dependency does not count. If Leo
> wants the stamp on the task record instead, that needs a dedicated fused-memory write tool (write-if-absent, one
> batched call a tick); say so and add the leaf.

**Replace 6.2 with:**
> - The anchor is `first_eligible.at`. `Scheduler.task_priority` returns it as `age_anchor` from memory.
> - At deploy the table is seeded once, before the first tick: for tasks that are candidates now, from their
>   `eligibility_anchors` row; for tasks already dispatched, from their first `lock_acquired` or `task_started` event
>   after their last `cancelled → pending`, else `updated_at`. Both can only under-age. Seeded rows are marked.
> - A request whose task has no anchor is aged from its first enqueue (`MergeRequest.first_enqueued_at`) at the pool
>   and from the leg's arrival at admission, and is counted on the event.
> - **A1.** A task's `first_eligible.at` is unchanged by any dispatch, re-pend, metadata write or restart.

Consequences: row 6 drops "a `replace`-mode metadata write occurs"; 6.1's comment becomes `# first_eligible.at (D2)`;
α's "Touches" becomes `eligibility_clock.py`, `scheduler.py` (the lookup and the terminal drop), `task_priority.py`,
`merge_pool_facts.py`, a seed script, and loses fused-memory, `shared/task_metadata.py` and `docs/task-authoring.md`;
α's signal loses "`get_task` shows the original `eligible_since`". α does edit `scheduler.py`, as 9908 did; it runs
ahead of the restructuring chain (9665 has 306 open tasks upstream), so the sentence "nothing of this PRD waits
behind the chain except the pin tier" stays true. Add to section 5: task 9636 edits `eligibility_clock.py` and the
same watchdog method and has no open prerequisite, so α and 9636 need a real edge; task 9679's ledger note.

### A2. α has no source for the rank of `pinned`, and θ2 is ordered after the task that needs it

**Draft.** D1: "`shared/task_priority.py` (task 6718) owns the tier names and ranks, `pinned` included ...
`orchestrator/task_priority.py` ... imports them". α needs only 9908. θ2 ("`pinned` in the vocabulary",
`shared/task_priority.py`) needs θ1.

**Wrong.** `order_key` in α must rank `pinned`. The module D1 imports it from does not exist until 6718, and
`pinned` enters it in θ2, which is downstream of α through θ1 (G6: the producer is downstream). θ1 also makes the
scheduler publish the tier before θ2 has taught the readers the word; `config.coerce_tier('pinned')` turns an unknown
tier into `medium` with a warning.

**Replace** D1's first two sentences with:
> `orchestrator/task_priority.py` defines `PINNED = 'pinned'` and the gate rank order (`PINNED`, then
> `config.PRIORITY_TIERS` in order) in α. Once task 6718 has landed, θ2 moves both into `shared/task_priority.py`
> and adds the validators. θ2 comes before θ1.

Section 8: θ2 needs α and 6718; θ1 needs θ2 and 9683. θ2's signal keeps the refusal of `boost_tier='pinned'` and
adds the refusal of a stored `priority='pinned'` (B14); "the dashboard shows inherited `pinned`" moves to θ1.

### A3. With dispatch order unchanged, the pin tier inside the scheduler is not specified, and the obvious build stops fairness parks for everyone else

**Draft.** D6: "The separate pin pass as a dispatch order goes; pinned-tier candidates are simply first in the
scheduler's order." θ1 touches `scheduler.py`, `pin_reservation.py`.

**Wrong.** The scheduler still sorts by `(-score, id)`, and the score is `TIER_BASE[coerce_tier(tier)] + ...`
(`Scheduler._compute_score`).
1. `TIER_BASE` has no `pinned`. Passed as a tier it coerces to `medium`: a pinned task would score as medium.
   `FairnessConfig.skip_threshold_for('pinned')` coerces the same way.
2. `pin_order` is not in the score or the sort, so "pin order is the order inside the tier" has no mechanism.
3. Only the first candidate of the sorted list earns skips and tier parks
   (`Scheduler._phase_select_scored`, `top_id`). If pinned candidates are simply first, the top is a pinned task
   whenever one is a candidate. Pins are usually lock-blocked for days (8261: 52 refusals in 9.9 days) and the
   watchdog adds up to three more. So no critical or high task would earn a fairness park while any pin waits. This
   is the fault of pending task 6328, made general. θ1's signal ("dispatch order is unchanged") cannot see it.
4. The walk's output (`effective_priorities`) is read inside the scheduler for the score, the skip threshold, the
   park rank, `reserve_now` and `lock_acquired.priority`. Each would receive `pinned`.
5. What the pin pass does besides ordering (choosing the reservation head, `pin_blocked`, skipping a load-held heavy
   pin, exempting deterministic pins) has no stated home.

**Replace D6 with:**
> **D6. Pin order is the order inside the `pinned` tier**, and `reorder_pin_queue` keeps working. The walk returns,
> for each task, its base tier (as today, pins ignored) and a `pin_order` when it or an unfinished dependent is
> pinned; its tier is `pinned` exactly when `pin_order` is set. The scheduler sorts by
> `(0 if pin_order is set else 1, pin_order, -score, id)`, the score computed from the base tier. `TIER_BASE`, the
> tier-width validator and the skip thresholds gain no entry. The candidate that accrues skips and earns tier parks
> is, as today, the best by score alone; score, skip threshold, park rank, `reserve_now` and
> `lock_acquired.priority` read the base tier. The published `effective_priorities` and the lookup read the tier.
> The pin pass's other duties run from the one loop, for candidates of the `pinned` tier: reservation head (D7),
> `pin_blocked`, the load-hold skip, the deterministic exemption.

Add a row: "Pins do not stop fairness: a pinned task is lock-blocked; a critical task is passed over twice: the
critical task holds a tier park." Add `config.py` to θ1 only if the base-tier route is not taken.

### A4. A failed batch never produces an observation as 6.3 defines one (mine)

**Draft.** 6.3: "An observation exists when an entry is finalized as head of the frontier with a test verdict."

**Wrong.** That is true of passes and of a single request's failure. A failed batch is not finalized: the lane
requeues its requests at the verdict (`_run_inflight_verify`'s red-tip arm → `_requeue_from_verify`) and leaves a
`REQUEUED` entry in the frontier. Its dispatching item was stacked on the entry ahead (`_deep_chain_placement`
chains only a speculative item, until 8527), so at the verdict nobody knows whether the tree tested was main plus
its members. Failed batches are the observations the whole design runs on, and ε's signal ("every finalized verdict
writes a row") would produce none for them.

**Replace** the first paragraph of 6.3 with:
> **A passed observation** exists when an entry is finalized as head of the frontier and lands. **A failed
> observation** exists when a failed verdict's base is known to be on main. A single request's failed verify waits
> in the frontier and is finalized as head, so it is one then. A failed batch is requeued at its verdict and leaves
> a `REQUEUED` entry; its verdict is **provisional** until that entry reaches the head. If every entry ahead of it
> landed, it becomes a failed observation; if one ahead failed, or the cascade tore it down, it is dropped. With
> `mode=on`, the members of a provisional failed batch are suspects for picking and are not convicted until it
> resolves. A verify that was cancelled, voided, requeued without a verdict, or re-run as a gate re-verification is
> never an observation. Members dropped when `build_chain` truncated are not in it.

ε's signal: "every landed batch, every finalized single failure and every resolved failed batch writes a row". ε's
sites: `_finalize_inflight` (the pass arm and the `REQUEUED` sentinel arm) and the head-failed cascade. Add a row:
"a batch stacked on a head fails; the head then fails: no observation, no suspects".

### A5. "No mixing in the stack" idles a host and contradicts the deadline (mine)

**Draft.** 6.5: "The pool does not pick a head of the other kind while one is in flight ahead of it, unless the
frontier is empty." Against: "A suspect past `suspect_deadline_secs` is verified alone, oldest first, ahead of all
other picks", and S1: "... divided by the number of hosts."

**Wrong.** It cannot deadlock, because every verify ends. But:
- a clean batch is in flight and only suspects wait, some past the deadline: no pick is allowed until the clean
  verify ends (up to about an hour on the slow host), so the deadline rule is false and a host is idle;
- a suspect batch is in flight and only clean requests wait: the second host is idle for the whole narrowing. By
  my first-pass replay a failed batch of six needs about 2.8 further verifies to resolve, so this is a common
  state. Task 8528's text forbids nothing of the kind: it stacks the next dispatch on the suspect half;
- S1 divides by hosts that the rule does not let the suspects use.

The cost the rule avoids is small when the alternative is an idle host: a verify stacked on a batch that fails is
cancelled and re-dispatched, which loses nothing but the CPU.

**Replace** the "No mixing in the stack" bullet and S1 with:
> - **Kinds in the stack.** A free host never idles while a request can be picked. At a pick the pool prefers a
>   head of the same kind as the last entry picked or in flight. It takes the other kind when none of that kind can
>   be picked, or when a suspect is past the deadline. The decision row records `stacked_on_kind`.
> - **S1.** A suspect past the deadline is the next pick, whatever is in flight. With h hosts and m suspects past
>   the deadline, the last starts within the time of ceil(m / h) verifies, plus one re-dispatch for each base that
>   fails under it.
> - **S3.** The lane is work-conserving: with a free host and a pickable request, a pick happens.

D12 becomes "never mixed in one batch; mixed in one stack only to avoid an idle host or to meet the deadline". Row
10's last clause becomes "the next pick is the suspect batch; a clean head is picked only when the second host would
otherwise idle". Section 3's rejected "basing a batch off the frontier" drops "6.5's no-mixing rule is used instead".

### A6. With `mode=on`, task 8528's depth and state are undefined, and the hand-over between the two mechanisms is missing

**Draft.** 6.4: "`max_batch` at run time is the smaller of the config value (8) and the depth task 8528's table gives
for the current estimate." B2. 6.3: "`prior_batch` ... one minus task 8528's rolling estimate". I1: "the dispatch
proceeds by task 8528's rule". Row 21.

**Wrong.** 8528's planned surface is `choose_depth(state, queue_len, depth_table) -> (k, reason)` and
`on_tip_result(state, links, passed)` over `ChainPolicyState(suspects, recent)`. `choose_depth` returns the bisect
depth, ceil(suspects / 2), while its suspect list is not empty. The draft does not say whether the lane still calls
`on_tip_result` with `mode=on`.
- If it does, a failed pool batch fills 8528's suspect list. 8528's list empties only through its own bisect, which
  is not running. A conviction never reaches it. `choose_depth` then answers 1 or 2 indefinitely and the pool stops
  batching.
- If it does not, `recent` goes stale and both `prior_batch` and the table's row freeze.
- On `on → off` (row 21: "with the suspects first") 8528's list is empty, so there are no suspects to put first.
- I1's fallback "by 8528's rule" with `mode=on` has the same hole.

**Add to 6.4:**
> With `mode=on` the lane feeds task 8528's state the verdicts of clean batches only (the pool's suspect batches are
> its "bisect rounds", which D7 of that plan excludes), and never its suspect list. `table_depth` is the depth-table
> row for the estimate alone; ζ adds it as a pure function beside `choose_depth`. η1 owns both hand-overs:
> `off → on` clears 8528's suspect list (the pool already holds those failures as observations); `on → off` writes
> the pool's current suspects to it in `pool_order()`. A decision that cannot be computed with `mode=on` dispatches
> the head alone.

Change I1's last clause to "with `mode=on` the head is dispatched alone; otherwise 8528's rule applies". Add a row:
"`off → on` while 8528 is mid-bisect: the next pick follows 6.5 and `table_depth` is the table's row". The names are
planned, not landed; ζ re-locates them.

### A7. The risk rule can never bind, row 8 cannot occur, and clean batch size is the fallback's

**Draft.** 6.4: "Each candidate c is added when `(n + 1) * p_pass(S + c) > n * p_pass(S)`". Row 8: "eight equal-tier
waiting requests at prior 0.16; the table's depth is 8: the batch has 6 members". R6: "Batch size comes from
per-request failure probabilities".

**Wrong.** Every clean candidate has the same probability, `prior_batch` = 1 - p̂, and the table reads the same p̂.
The rule allows 14, 6, 4, 3, 3 and 2 members at p̂ = 0.93, 0.85, 0.80, 0.75, 0.70 and 0.55; the table gives 8, 4,
2, 2, 1 and 1. The table is the smaller at every value, so the rule skips nobody. Row 8 asks for a depth of 8
(p̂ at least 0.93) and a prior of 0.16 (p̂ = 0.84) at once. What v2 actually builds with `mode=on` is: clean batch
size from 8528's table; membership from priority order minus suspects and confirmed conflicts; evidence used for
suspect batches and conviction. That is defensible (it is what my first-pass A7 asked for), but it is not what R6
says, and Leo should be told in those words.

**Replace.** In 6.4 after the rule: "While B2 holds, every clean candidate shares one prior and this rule skips
nobody: the table is always the smaller. The rule is kept for the day ι's replay lifts B2, and is tested at the
function." Row 8: "`select_clean_batch` called with `max_batch` 8 and seven candidates at `p_bad` 0.16 returns 6
members and two `risk` skips." Row 9: "`mode=on`, p̂ 0.86, eight waiting: the batch has 4 members." Add to the
hand-back for Leo: "R6 is met for suspect batches and for who is excluded. The size of a clean batch stays task
8528's until a replay with stacking counted supports more."

### A8. The content id is not stable across a rebase as specified

**Draft.** 6.3: "`git patch-id --stable` of its fork-point diff ... It survives a rebase." Row 23.

**Wrong.**
1. `git patch-id` ignores line numbers but hashes context lines. A rebase that applies cleanly onto a main where a
   line within three lines of a hunk changed gives a different id. Checked on stdin: same context, moved 30 lines,
   same id; one context line changed, different id; with zero context (`-U0`), same id.
2. The conflict bounce rebases a waiting request onto the frozen tip, which is not on main. Recomputed against main
   there, the fork-point diff includes other tasks' unlanded changes. Row 23 holds only if the id is not recomputed,
   which the draft does not say.
3. An empty diff gives an empty string, so every such request would be "unchanged" from every other.
4. `GitOps.get_branch_changed_files` excludes `.task/`; the id must too, or plan files change it.

Cost is small: one `git diff` and one `git patch-id` per enqueue, tens of milliseconds, through the async runner.

**Replace** the content-id paragraph with:
> A request's **content id** is `git patch-id --stable` of `git diff -U0 main...ref -- :!.task/`, computed once in
> `merge_lane/intake.py::enqueue_merge_request` and carried on `MergeRequest`, the journal and `merge_queued`. It is
> not recomputed when the lane rebases a waiting request. An empty diff, a git error or a grouped request gives
> `None`; `None` equals nothing, and a request with no content id is never convicted by elimination.

Row 23: "a branch is rebased onto a main that changed a line next to one of its hunks, and re-submitted: same
content id. A waiting request rebased by the conflict bounce keeps its id and its observations."

### A9. The storm limit as worded stops ordinary failures

**Draft.** D13: "a failed own verify convicts at once." D16 and C2: "More than `convict_storm_max` (3) convictions in
an hour stops convictions". Row 18: "four convictions in one hour: the fourth does not happen".

**Wrong.** An own-run conviction is the lane's ordinary outcome for a single request that fails: about eight a day
(117 in the 14-day window), so four in an hour will happen. Read literally, the fourth failed solo verify of the hour
"does not happen": the lane would have a failed verify and no outcome to give it. (The ambiguity came from my
wording; D13 then made own runs convictions.)

**Replace** D16 and C2 with: "More than `convict_storm_max` (3) **eliminations** in the trailing hour stops
eliminations and files one L1, keyed so it is filed once per stop; a suspect that would have been eliminated is
verified alone. Eliminations resume when the trailing-hour count is back at or under the limit. Own-run outcomes are
never counted or stopped." Row 18: "four requests reach the threshold by elimination within an hour: the fourth is
verified alone."

### A10. The main-red rule cannot include the failure that revealed the break (mine)

**Draft.** D15: "every failed observation whose tested tree contained M is marked `main_red` retroactively and
excluded, including the one that triggered the probe."

**Wrong.** M is main when the break is confirmed. The break entered at some earlier commit. A batch that failed
before main reached M was built on an older main and does not contain M, so the rule misses exactly the failures it
is for, the triggering one included. The observation row also has nothing to test the rule with.

**Replace D15's second sentence with:** "When main is confirmed red, every failed observation made since the last
passed observation landed, all of whose red module legs are also red on bare main in the confirming probe, is marked
`main_red` and excluded." Add `red_units` (the `ran_red` legs from `merge_verify.units`) to the observation row.

### A11. A conviction has no failing evidence to carry

**Draft.** Goal 5: "goes back to its workflow with the failing evidence". 6.6: "`reason` is built by the existing
'Post-merge verification failed' builder from the failed batch's report."

**Wrong.** That builder works from the live `VerifyResult` (`failure_report()`, the new failing ids). A failed batch
writes none of it anywhere: no `merge_attempt`, no outcome, and `merge_verify` carries module legs only. A conviction
happens later, possibly after a restart. η2 would have nothing to build from, and the steward would receive
"convicted by elimination" with no failing test. The capability has to come from ε, upstream (G6).

**Add to 6.3:** "Each failed observation keeps its verify's failure report and new failing ids, capped at 20 kB, on
the `merge_pool_observation` row; the conviction's `reason` is built from the stored report of the latest failed
batch the request was in." Add the same to ε's work and to `ConvictionEvidence` (`report_event_id`).

### A12. The gate that flips `order` has no readout and waits for the batch work

**Draft.** D14: "Each is flipped by a human gate that presents a readout." ι: "Gate 1 flips `order`; gate 2 flips
`mode`"; needs "ζ and γ observed running; 8532 has read". The readout described is all about batches.

**Wrong.** Nothing in ι's readout bears on the pick order, so gate 1 presents nothing for its decision. And as one
leaf, gate 1 waits for ζ, which waits for ε, 5041 and 200 shadow decisions (about 15 days at 13 batch verifies a
day). The priority pick, which is goal 1 and needs only γ, is held for the batch calibration.

**Replace ι with two leaves.**
> **ι1. The order readout and its gate.** A script replays task 9908's recorded picks under `order_key`: picks that
> would differ, how often a higher-ranked request waited, waits by tier before and after, and how many overtakes
> cross a recorded confirmed conflict. Prints `no pick records yet` when there are none. The gate presents it and
> flips `order`. Needs γ observed running and 8532 has read.
> **ι2. The batch readout, the replay and its gate** (the rest of today's ι). Needs ζ observed running and ι1's gate.

### A13. η1 waits behind 143 open tasks it does not need, and κ behind 542

**Draft.** η1 needs "ζ, ι's second gate, 5041, 5042, 8933". κ needs "β, γ, η2, θ2". Critical path: "η1 (after 5042
and 8933)".

**Wrong.** Open tasks upstream today: 5041 has 4; 5042 has 143 (through 9265); 9683 has 542. η1's wiring is at the
pick, the placement and the failed-batch arm. None of that is landing code, which is what 5042 moves; ε touches
`_finalize_inflight` and rightly has no such edge. (The edge came from my first-pass B15.) κ is the PRD's signal
leaf and waits for θ2, so for the 542.

**Replace.** η1 needs ζ, ι2's gate, 5041 and 8933. Split κ: "κ1, the gates: rows 1 to 25, needs β, γ, η2" and "κ2,
the pin tier: rows 26 to 28, needs κ1 and θ1". The critical path line drops 5042.

### A14. R7 is now read as "bounded by the deadline", and that is not put to Leo

**Draft.** R7: "A suspect must not sit in the pool." 6.5: "Suspect work is ordered with clean work, not given a slot."

**Wrong.** v1 gave suspects a slot of their own. v2, on my advice, orders them by priority, so a low-tier suspect
waits behind higher-tier clean work for up to the deadline, 2 h by default. That may be what Leo meant; it is a
reading of his ruling, made in a "decided here" item.

**Replace** D12 with a fourth [Leo] item: "**D12 [Leo]. How R7 is read.** *(recommended)* Suspects are ordered by
priority with clean work and bounded by `suspect_deadline_secs` (2 h): a low-tier suspect can wait that long behind
higher-tier clean work. *Alternative:* a suspect batch is always the next pick; a pinned clean arrival then waits
behind the narrowing of lower-tier suspects whenever one host is available."

---

## B. Minor

1. **6.4, "With `mode` not `on`, the head is the first of `pool_order()`".** False with `order=arrival`, where the
   same-files rule still applies at the pick (6.7, and γ's "the recorded pick order is unchanged"). Add "after the
   same-files rule when `order=arrival`".
2. **6.6 and row 15, "a second elimination of the same content id".** Unreachable: the returning request is verified
   alone, so its failure is an own run, which already counts. Row 15: "... if that verify fails, the thrash counter
   advances." Delete the "second elimination" clause from 6.6 and D13. (Mine.)
3. **Section 3, "unequal weights are a config option".** No such key exists in 6.7 and the rule is written with
   weights 1. Delete the clause.
4. **α's signal, "`merge_queued` rows carry `age_secs`".** 6.7 lists only `content_id` as new on `merge_queued`. Add
   `age_secs` there, name γ as owner of the `merge_dequeued` keys, and add `merge_pool_facts.py` to both.
5. **Section 8, files and edges.** γ's readers include `chain_snapshot`, `unfrozen_suffix`, `snapshot()` and the
   heartbeat (`worker.py`) and the queue-position tools; none is in γ's "Touches". ε edits `worker.py` too, with no
   edge to γ although the critical path orders them: add γ to ε's needs. β and γ both edit `config.py` with no edge.
   ε and γ race task 5040, which moves `_run_inflight_verify`: list both homes in their files.
6. **Two priors, one `Posterior`.** Say which is used where: `p_pass` (batch choice, `p_pass_dispatched`) from
   `prior_batch`; `p_bad` for conviction from `prior_conviction`. That is two calls to `infer`.
7. **B2.** Scope it to clean batches. A suspect prefix can be three where the table gives two.
8. **O3 and ζ's signal.** "Every batch dispatch writes a row" leaves a head dispatched alone with no row, which
   reads as `mode=off`. Write "every dispatch".
9. **"Ahead of all other picks" (6.5) and "before anything else" (6.6) against D10.** State the order: high lane,
   then suspects past the deadline, then a returning convicted request, then the rest.
10. **When inference runs.** The pick is synchronous. Say: the posterior is recomputed when the cache changes,
    awaited in `_acquire_next_request` and `_look_ahead` beside the conflict-graph recompute; the pick reads the
    cached value. Bound the rebuild's event read (48 h).
11. **Readout.** Bin `p_pass_dispatched` against `merge_pool_observation` verdicts, not raw `merge_verify` rows,
    which include voided verdicts.
12. **Watchdog pins.** From α an automatic pin (up to three, 24 h) is `pinned` at both later gates. Say so. The
    recorded `tier` then hides the task's own tier: add `base_tier` to the recorded rows.
13. **D5, "a task's own `priority` cannot be `pinned`".** The task store does not validate `priority` at all. θ2
    adds the refusal at `submit_task` and `update_task`.
14. **The key name.** If a stamp keeps the name `eligible_since` it collides with the watchdog's column of the same
    name and different reset rule. Use `first_eligible`.
15. **Fail-soft counters with no threshold.** `decision_error`, `TooManySuspects` and the missing-anchor count are
    recorded but nobody is told at the hundredth. Ten consecutive `decision_error` rows with `mode=on` file one L1.
16. **Title and section 2.** "One priority order at every gate" and "one mechanism replacing three" overstate v2:
    the scheduler's order within a tier differs by design (D3) and hand-set lanes remain (D10).
17. **ι's replay.** Name its base (the study's simulator is under `/tmp` and uncommitted; copy it into the evidence
    directory) and its ground truth: bad means a failed own run with that content id, good means landed with it,
    the rest excluded.
18. **`Head` as the return of `pop_next_pickable`.** It has two callers that expect a `MergeRequest`. Keep the
    return type; carry `kind` on the picked item to the placement.
19. **β's switch.** D14 says "two switches". Verify admission is a third: 8261's `verify_admission_order` value, per
    project. Say β ships behind it.
20. **Row 2 and row 4.** Add `order=priority` to the preconditions, as rows 1 and 5 have.
21. **Sizes.** With A1's replacement α fits the band. η1 still carries ten rows; acceptable if the restart rebuild
    moves to ε, where the cache lives.
22. **6.6, "A wrong conviction costs ..."** The list is right only once A11 is in: without a stored report the
    steward pass has nothing to act on, for right and wrong convictions alike.

---

## Invariants: hits that remain in v2

| Invariant | Mechanism | Item |
|---|---|---|
| `contracts-machine-checked` | "own `priority` cannot be `pinned`" has no validator | B13 |
| `structured-facts-at-failure` | conviction evidence not captured at the failed verify | A11 |
| `corroborate-before-acting` | a failed batch's verdict used before its base is known; first-eligible row deleted on absence from a fetch | A4, A1 |
| `storm-escape-required` | counters without a threshold | B15 |
| `no-lockstep-duplication` | 8528's suspect list and the pool's suspects with no hand-over | A6 |
| `holds-owned-and-bounded` | S1 unmeetable under the no-mixing rule; the stopped-elimination state has no exit | A5, A9 |
| `loop-thread-occupancy-bounded` | 2,516 metadata writes on one tick | A1 |
| `guards-exercise-behaviour` | row 8 impossible, row 15 unreachable, row 23 true by omission | A7, B2, A8 |
| `no-silent-fail-soft` | a solo dispatch writes no decision row; a `None` content id | B8, A8 |
| `readers-prove-their-producer` | gate 1 has no readout over 9908's rows | A12 |

## META

Not yet, even with the three [Leo] decisions confirmed as recommended. D3 and D18 are sound at the recommended
value. D2's recommendation is right in substance and wrong in mechanism. The three changes that matter most:
**(1)** keep the first-eligible stamp in the scheduler's own store, seeded from the anchors that exist, and give α
the rank of `pinned` (A1, A2); **(2)** specify the pin tier against a scheduler that still sorts by score, with the
fairness top unchanged (A3); **(3)** close the lane contracts that the first rewrite left open or that my own text
got wrong: the failed-batch observation, the stack rule, 8528's state with `mode=on`, the content id, the storm
scope and the evidence a conviction carries (A4 to A11). A12 to A14 are sequencing and one more question for Leo.

Not clean.
