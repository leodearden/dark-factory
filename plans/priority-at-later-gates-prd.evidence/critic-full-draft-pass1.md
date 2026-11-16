# Critic of `/tmp/prd-merge-pool/draft.md` (merge pool PRD)

Read-only pass, 2026-11-13. Code read on main `0bb6dd2b96` (ten commits after the draft's `ec46ac6670`; none touches a
file the draft cites). Branch `task/9908` read at `8c6d25dad6`. Task store opened `?mode=ro`. Scratch arithmetic is in
`/tmp/prd-merge-pool/critic-scratch/` (`sim.py`, `t.py`); both run in under a minute.

Counts: **17 answer-changing (A)**, **21 minor (B)**, invariant walk (C) with 10 slugs hit and 1 advisory.

What checked out and needs no change: every symbol in the section 4 "verified on main" table exists with the stated
role; `ChainReservations` and `MergeIntake.take_buffered` work on any set of requests, so a batch that skips
candidates can be reserved and landed; `build_chain` takes any ordered sequence; `MergeOutcome` is a plain
`@dataclass`, so an optional `conviction` field breaks neither the journal (`PersistedMergeRequest`) nor the landed
outbox; the thrash counter can be skipped from a field on the outcome
(`workflow.py::TaskWorkflow._check_merge_outcome_thrash` runs only when `_last_merge_block_reason` is set); the
q < 1/n rule and 14 / 6 / 3, the 0.73 to 0.95, 0.86 and 0.99 posteriors, B2, rows 8 and 10, 6.94 points per hour,
17 h and 43 h, and every backstop default all recompute as stated.

---

## A. Answer-changing defects

### A1. The batch's first member is picked long before the batch is chosen, by a different function

**Draft.** 6.4: "`candidates` are the waiting, unreserved, unheld, non-suspect requests in `order_key` order. The first
candidate is always a member." 6.5: "the next free verify slot takes a suspect batch." 6.7: "The pick takes the first
request of `pool_order()`." 8 η: "wiring in the dispatch fill and the red-batch arm".

**Wrong.** The lane has two steps, and the draft's rules sit on neither.
1. The merger picks one request (`MergeIntake.pop_next_pickable`, called from
   `SpeculativeMergeWorker._acquire_next_request` and `::_look_ahead`), merges it onto the speculative base, and puts
   it on `_verifier_queue`. It picks ahead of free hosts, up to the speculation permits.
2. Only when a host is free does `::_dispatch_item` call `::_deep_chain_placement`, which adds members from
   `chain_snapshot()` on top of that item's own merge commit.

So the batch's first member is not a waiting request: it left the buffers at step 1. A suspect batch cannot be
"taken by a slot": its head has to be chosen at the pick. And the pick as written (first of `pool_order()`, with no
notion of suspect) will make a suspect the head of a clean batch, or a clean request the head of a suspect batch.
S2 cannot hold. An implementing agent has to invent where each rule goes.

**Evidence.** `_deep_chain_placement`: `candidates = snapshot[:d - 1]`, base `item.merge_result.merge_commit`;
`_acquire_next_request` and `_look_ahead` are the only callers of `pop_next_pickable`.

**Replace 6.4's first two bullets and 6.5's second bullet with:**
> A batch has a **head** and **members**. The head is the request the merger picked; the members are added when the
> head is dispatched. Both decisions are the pool's.
> - `next_head(pool, posterior, in_flight) -> (request, kind)` is what `MergeIntake.pop_next_pickable` returns. `kind`
>   is `suspect` when a suspect set exists and no suspect head is picked or in flight, else `clean`. A clean head is
>   the first non-suspect of `pool_order()`; a suspect head is the first request of the batch 6.5 selects.
> - `select_clean_batch(head, candidates, ...)` and `select_suspect_batch(head, suspect_set, ...)` run in
>   `_deep_chain_placement`, at dispatch. `candidates` exclude the head. The head is member 1 and lands first; the
>   other members follow in `order_key` order (this replaces B1's "members are in `order_key` order").
> - A head picked as `clean` that has become a suspect by the time it is dispatched is dispatched alone.
>
> Wiring sites for ζ and η: the pick (`pop_next_pickable`), the placement (`_deep_chain_placement`), the red-tip arm
> of `_run_inflight_verify`, and resolving a still-buffered request with an outcome (the existing precedent is
> `suffix_graph.py::SuffixConflictTracker._escalate`).

### A2. "The observations are the `merge_verify` rows" is false, and the log has four more inputs

**Draft.** D11: "the observations are the `merge_verify` rows, and the in-memory list is a cache rebuilt from the event
store at start." 8 ε: "`pool_observations.py` (reads `merge_verify` rows)". Row 17.

**Wrong, in two ways.**
1. A `merge_verify` row is written when a verify finishes, not when its verdict counts. A verify stacked on an
   in-flight predecessor (`base_relation` = `stacked`) tested a tree that contains the predecessor's requests. If the
   predecessor then fails, the lane throws the verdict away and re-merges (`_verifier_loop`'s head-failed cascade,
   `_void_and_remerge`). Read as an observation, that row blames or clears the wrong requests. The same holds for
   gate re-verifications (229 of 840 rows in the recent window, measure.md item 3), for verifies that raised or timed
   out, and for rows with no member list (trains, solo attribution; 9908 leaves `chain_request_ids` absent there).
   The draft never says what the likelihood of any of these is.
2. 6.3's own rules need events that are not `merge_verify`: landings (a landed member leaves earlier failed
   observations), convictions (a convicted member takes its failed observations with it), re-submissions with a new
   head (the observation is dropped), and the main-red mark. Rebuilt from `merge_verify` rows alone after a restart,
   the observations a conviction removed come back, and the co-members that were cleared are suspects again. Row 17
   fails.

**Replace D11's second sentence with:**
> An observation exists when an entry is finalized as head of the frontier with a test verdict: its base is then on
> main, so the tree tested was main plus exactly its members. A verify that was cancelled, voided by a failed base,
> requeued without a verdict (lease contention, runner unavailable, a raise on the chain arm), or re-run as a gate
> re-verification is not an observation. The pool emits one `merge_pool_observation` row per observation (members by
> request id and content id, passed, own_run, main_red, the `merge_verify` event it came from). The cache is rebuilt
> from four row kinds only: `merge_pool_observation`, `merge_finalized` (landings), `merge_convicted`, and
> `merge_queued` (content id of a re-submission). ε's reader names all four.

Add `merge_pool_observation` to 6.7's new events and to ζ's `event_store.py` work.

### A3. The pool order breaks task 8528's bisect, which is the fallback, and has no off switch

**Draft.** D14: "`off` restores task 8528's table and bisect at any time." 6.7: `chain_snapshot` and the pick read
`pool_order()`. Row 16. Section 5: 8532 "reads the lane with the pool `off` or `shadow`".

**Wrong.** 8528's bisect is positional. Its text: "requeue the k links at the FRONT in aging-key order, suspects :=
those links ... the next dispatch verifies the first ceil(|suspects|/2) as a chain" (policy PRD D6). After γ there is
no front: order is computed from `(tier, age)`. The next pick is the highest-priority waiting request, which may be a
fresh arrival, and the "suspect half" is then a batch of non-suspects led by it. γ lands before ζ and is not under
`merge_pool.mode`, so this breaks the bisect in `off` and `shadow`, the two modes the draft says keep 8528's
behaviour. γ needs only α and 8528, so it also lands inside 8532's seven-day canary window and changes the pick that
8532 is reading.

**Replace** the first bullet of 6.7 with:
> `MergeIntake.pool_order()` returns the waiting requests in this order: high lane first (D10); then, while
> `merge_pool.mode` is not `on`, the requests in task 8528's suspect list, in the order 8528 requeued them; then the
> rest by P1. `merge_pool.order` is `arrival` or `priority`, hot-reloadable, default `arrival` until task 8532 has
> read; with `arrival` the rest are ordered as today. γ ships the `priority` order behind that switch.

Add a boundary row: "8528 bisect under the pool order: a chain of 4 fails with `mode=off`, `order=priority`, and a
high-tier request arrives; the next dispatch verifies two of the four suspects, not the arrival." Add γ → after 8532
for the flip of `order`, as a line in ι's gate.

### A4. A request that fails its own verify is verified a second time; a request in flight can be convicted

**Draft.** 6.5: "When `p_bad(r) >= convict_threshold` (default 0.9), r's request is resolved back to its workflow ...
No verify is spent on it." "A set of one is verified alone unless it is already convicted." 6.6 names
`by: 'own_run'` but no rule produces it.

**Wrong.** At the seeded defaults a request verified alone that fails has p_bad = 0.16 / (0.16 + 0.84 x 0.03) = 0.864,
below 0.9. By the text it is a suspect set of one and is verified alone again; only the second failure (0.995)
convicts. Every real single failure then costs two verifies. Singles are 513 of 610 non-gate verifies in the recent
window. Second, nothing limits conviction to waiting requests: a suspect whose solo deadline verify is in flight can
cross the threshold when its co-members pass, and be resolved `blocked` while its verify goes on to pass and land.

**Add to 6.5:**
> **Own run.** A failed observation with `own_run` true resolves that request at once through the lane's existing
> own-verify arm (`_run_post_merge_verify`'s blocked outcome), with `conviction.by = 'own_run'`, whatever its `p_bad`.
> The threshold applies only to requests with no failed own run.
> **Only waiting requests are convicted.** A request that is merging, reserved or being verified is never convicted;
> its verdict decides (O2).

### A5. "Re-submitted with the same head" cannot be recognised, and 6.6 contradicts 6.3

**Draft.** 6.3: "`prior_unchanged_resubmission` = 0.73 for a request re-submitted with the same head after a failure of
its own (8 of 11)"; "An observation is dropped when a member is re-submitted with a new head"; "A member that is
convicted takes with it every failed observation it was in." 6.6: "A request re-submitted unchanged after an
elimination conviction enters at `prior_unchanged_resubmission`, so it is a suspect set of one and is verified
alone." Section 2: "8 of 11 over seven days (`merge-lane-policy-prd.md` D14 and its source seat)".

**Wrong.**
1. Every merge attempt rebases first (`workflow.py::TaskWorkflow._rebase_before_enqueue`, in the retry loop of the
   merge phase), and the suffix bounce rebases waiting requests in place. With about 50 landings a day the commit SHA
   changes on almost every re-submission, changed or not. Keyed on the SHA (9908 records `head_sha` =
   `MergeRequest.snapshot_tip`), "same head" almost never matches, and every rebased suspect has its observations
   dropped. The readout's "later verdict on the same head" (C1) and the refit of `false_failure` (ι) fail the same
   way. measure.md item 5 could not use a SHA either: the branch head was on no event, and the merge commit matched
   in 2 of 65 cases.
2. 6.6 does not follow from 6.3. An elimination conviction is not "a failure of its own", so the 0.73 prior does not
   apply. The conviction removed the request's failed observations, so it is not in `Posterior.suspects`. By 6.4 it
   is a clean candidate and can lead a batch of six. A truly bad request then loops: clean batch fails, narrowing,
   conviction, steward, re-submit. Nothing counts it, because D13 exempts it from the thrash counter; only
   `max_merge_retries` (3) ends it, after it has spoilt up to three clean batches.
3. "8 of 11" is not in D14 (which says "the unchanged-tip hold is worth 0.08 verifies/landing") nor in any file the
   draft lists as evidence. The only "8 of 11" in `plans/` is train derails with an own-defect member, a different
   fact. With n = 11 the interval is about 0.39 to 0.94.

**Replace** with:
> A request's **content id** is `git patch-id --stable` of its fork-point diff (`main...ref`), computed by the
> workflow at submission and carried on `MergeRequest` and `merge_queued`. It survives a rebase. "Unchanged" means
> equal content id. An observation is dropped only when a member's content id changes.
> A request whose task was convicted and that comes back with the same content id is **verified alone before
> anything else**, whatever the prior says; a request convicted twice by elimination with the same content id counts
> on the thrash counter. The lane learns this from `merge_convicted` rows keyed by task id and content id.
> `prior_unchanged_resubmission` is provisional at 0.73 (source to be located or re-measured in ι from content ids);
> ι refits it with `false_failure`.

Add `git patch-id` to the G3 table (it is a git builtin; no wrapper exists in `git_ops.py`, so α or ζ builds one).

### A6. The suspect rules cannot be met: the deadline bound, one host, and one batch at a time

**Draft.** 6.5: "While any suspect set exists and no suspect batch is being verified, the next free verify slot takes a
suspect batch." "A request that has been a suspect for longer than `suspect_deadline_secs` is verified alone next."
"**S1.** No request remains a suspect for longer than `suspect_deadline_secs` plus one verify." Goal 6.

**Wrong.**
1. S1 is false by construction. All members of a failed batch became suspects at the same instant, so they pass the
   deadline together. Five of them cannot each be "next". With one suspect batch at a time the last waits the
   deadline plus five verifies.
2. Load. At the defaults a clean batch of six passes with 0.84^6 = 0.35. My replay of the draft's own rules
   (`critic-scratch/sim.py`) needs 2.8 verifies per batch of six in total, so each failed batch needs about 2.8
   suspect verifies, and the suspect slot is asked for 0.65 x 2.8 = 1.8 verifies per clean verify. With two hosts and
   one suspect batch at a time that is more than it has. The draft states no premise for this.
3. With one host (the other quarantined or away) the only slot serves suspects whenever any exist, so a pinned clean
   arrival waits behind the narrowing of low-tier suspects. That inverts R2 and R3. With no clean candidates the
   second host idles, because only one suspect batch may be in flight.
4. Stacking. "A clean batch is dispatched first" puts the suspect batch on top of a batch that fails 65% of the time,
   and the next clean batch on top of a suspect batch chosen to fail half the time. Each failure cancels what is
   stacked on it. The draft rejects basing a batch off the frontier and does not count this cost.
5. A halted normal lane (main red) cannot verify suspects at all, so the deadline passes with nothing possible.

**Replace** the slot bullet, the deadline bullet and S1 with:
> - Suspect work is ordered with clean work, not given a slot: a suspect batch is taken when its first request
>   precedes the first clean candidate in `order_key` order, or when a suspect has passed the deadline. Any number of
>   suspect batches may be in flight, one per suspect set.
> - A suspect batch is never stacked on an unverified clean batch nor a clean batch on a suspect batch: the pool does
>   not pick a head of the other kind while one is in flight ahead of it, unless the frontier is empty. (If Leo
>   prefers, bring "base off the frontier" into scope instead; one of the two is needed.)
> - **Deadline.** Suspects past the deadline are verified alone, oldest first, ahead of all other picks.
> - **S1.** A suspect waits at most `suspect_deadline_secs` plus one verify for each suspect ahead of it past the
>   deadline, divided by the number of hosts. The clock stops while its lane is halted.
> - ι's readout reports, from a replay of recorded batches under these rules, verifies per landing and the suspect
>   wait at p50 and p95 beside the 8528 figures. The gate does not flip `on` without them.

### A7. The batch rule is not "the table's generalisation": it ignores what is stacked behind, and the first member

**Draft.** Section 2: "The rule is the table's generalisation, and is more aggressive, so `max_batch` caps it." 6.4:
"The first candidate is always a member. Each further candidate c is added when
`(W + w_c) * p_pass(S + c) > W * p_pass(S)`." 6.1: "`TIER_WEIGHT`: pinned 16, critical 8, high 4, medium 2, low 1".

**Wrong.**
1. The rule counts only this batch's landings. Every member also multiplies the survival of every verify stacked on
   this one by (1 - q). With one stacked successor of equal weight the same marginal test gives q < 1/(2n): 3 at
   0.16, which is where the table sits (4 at 0.85, 2 at 0.75). At the seeded prior (0.16) the pool builds 6 where
   the table gives 2 (0.84 is under 0.85). At the median of 8528's own rolling estimate (0.865, measure.md item 4)
   it builds 7 where the table gives 4. `max_batch` = 8 caps neither. Shadow mode checks calibration of `p_pass`,
   not throughput, so the gate would flip `on` to batches about twice to three times the fallback's with no
   evidence. (With nothing stacked and one slot, bigger batches do land more per verify under the draft's
   narrowing: `critic-scratch/sim.py` gives 1.78 at six against 1.15 at two. The missing term is the stacking.)
2. The test reduces to q_c < w_c / (W + w_c). It does not look at the members already chosen. A first candidate with
   p_bad 0.73 still collects five clean members, each of which then lands with probability 0.27 x 0.84^5 = 0.11
   instead of 0.84^5 = 0.42 without it. The batch is not worse than verifying the first alone (0.68 against 0.27
   expected landings), but verifying the first alone and batching the other five lands 0.27 + 5 x 0.42 = 2.4 in two
   verifies, 1.2 per verify against 0.68.
3. The weights have no stated basis, and their effect is not disclosed. After a pinned head (16) a low-tier request
   is admitted only if q < 1/17, so never at 0.16; a pinned head rides with at most a few criticals. After two highs
   and two mediums (W = 12) no low is admitted (1/13 = 0.077). Low-tier requests then land only when nothing higher
   waits or after promotion at 9 h, although an extra member costs no verify time (measure.md item 3).

**Replace.**
> - `max_batch` at run time is the smaller of the config value and the depth task 8528's table gives for the current
>   estimate. The pool chooses who is in the batch, never more of them than the fallback would take, until ι's replay
>   (A6) shows a larger batch lands more per verify with stacking counted.
> - A first candidate with `p_bad` above 0.5 is verified alone.
> - `TIER_WEIGHT` defaults to 1 for every tier (the rule then is q < 1/n). Unequal weights are a config option, with
>   the consequence above written beside it.

Section 2's premise becomes: "With equal weights and nothing stacked behind, the rule admits the n-th member while
q < 1/n: 14, 6 and 3 against the table's 8, 4 and 2. The table is smaller because a failed batch also cancels what is
stacked on it. The pool therefore never exceeds the table (6.4)."

### A8. Shadow calibration compares a prediction for one batch with the verdict of another

**Draft.** D14: "In `shadow` the decision is computed and recorded at every dispatch and never acted on." 6.7 event:
"members chosen, members actually dispatched, `p_pass`". Row 15: "records the would-be members and `p_pass`; the
dispatched members are 8528's". ι: "predicted against actual pass rate by bin".

**Wrong.** In shadow the verdict belongs to 8528's members; the recorded `p_pass` belongs to the would-be members.
Binning one against the other measures nothing. In `on` the same happens whenever `build_chain` truncates. The gate
that decides `on` rests on this readout (G6: the signal cannot be produced from ζ's rows). No minimum sample is
stated either; batch verifies run at about 13 a day (51 in the 3.9-day paused window), and in shadow no suspect ever
exists under the pool's rules, so suspect behaviour is not observed at all.

**Replace** the event line with: "`p_pass_chosen`, and `p_pass_dispatched` = the same posterior evaluated on the
members the verify actually covered; the readout bins `p_pass_dispatched`. A decision that could not be computed
writes a row with `decision_error`." Add to ι: "The gate needs at least 200 decisions with a verdict; fewer prints
`insufficient rows (n of 200)`. Suspect narrowing is evaluated by the replay in A6, since shadow produces none."

### A9. D7 removes the bound on pin reservations and reserves for every pinned prerequisite

**Draft.** D7: "that is the fairness park the top-ranked blocked candidate already earns, with a skip threshold of 0
for the `pinned` tier. Exactly one candidate holds such parks at a time, as today (`pin_reservation_max_active` = 1).
This stays inside the earlier PRD's rejection of reserving for every starving task." D6: "the negative park-rank band
(`pin_reservation.py::PinOrder.rank`) go."

**Wrong.**
1. Fairness parks are not one at a time. Only the tick's top candidate earns one
   (`Scheduler._bump_skip_and_maybe_park`), but it keeps it after it stops being top, until it dispatches or
   `Scheduler._park_hold_refusal` rejects it. The cap of one is enforced by `Scheduler._bound_pin_reservations`, which
   finds pin reservations by the rank band (`ModuleLockTable.snapshot_pin_reservations`). D6 deletes the band, so
   nothing enforces the cap.
2. D5 makes every unfinished prerequisite of a pinned task `pinned`. With threshold 0 each of them parks its files
   above critical the first time it is top and lock-blocked, and keeps the park. That is the "Below-rank-1 park
   INSTALLATION" row the scoring PRD rejects (minus 4% to 11% throughput, gridlock), whose task-6040 annotation
   allows only "the head eligible lock-blocked pin(s) ... capped by `pin_reservation_max_active`".
3. The pin pass does things the scored loop does not, and the draft is silent on each: a PSI-held heavy pin is
   skipped before it reserves (in the scored loop a PSI-deferred top is bumped and parks); deterministic pins never
   reserve; `pin_blocked` events; among pins the lower `pin_order` outranks on a shared key, where the park key is
   `(rank, id)` and would fall back to task id; `SchedulerStateSnapshot.pin_reservations` and `pin_queue`, read by
   the dashboard (`dashboard/data/scheduler.py`, `tab_scheduler.jsx`); `reservation_*` events' `source: pin`, derived
   from the band.
4. Tasks 6328 (a critical below a blocked critical never earns a reservation) and 3824 (durable parks) are pending
   on the same rule and are not in section 5.

**Replace D7 with:**
> **D7. The pin reservation stays bounded.** The reservation band in the park order stays: a park installed for a
> `pinned`-tier owner ranks above every other tier and is ordered by `pin_order`. At most
> `pin_reservation_max_active` owners hold one, chosen as today (the first lock-blocked, non-deterministic,
> not-PSI-held candidates of the `pinned` tier in `order_key` order), and `_bound_pin_reservations` keeps enforcing
> it. Inherited `pinned` prerequisites compete for that one reservation by `pin_order`; they do not each get one.
> What goes is the separate pin pass as a dispatch order, not the bounded reservation. `pin_blocked`,
> `pin_reservations` and `pin_queue` keep their meaning; `pin_queue` lists stored pins only, and
> `effective_priorities` shows the inherited tier.

Add rows for 6328 and 3824 to section 5, and to θ's files: `shared/scheduler_state.py`, `dashboard/data/scheduler.py`.

### A10. D3's credits reverse the scoring PRD's own worked case, and row 7 does not test the change

**Draft.** D3: "The credits are sized to today's exchange rate for a freshly pending task ... 17 h ... 43 h." Row 7:
"two same-tier tasks aged 30 and 60 days, no credits: the older dispatches first (today both are near the cap and
tie-break by id)".

**Wrong.**
1. 6.94 points an hour is the slope at age 0, the steepest point. The scoring PRD states the real crossover: "a `D=0`
   task overtakes a brand-new `D=32` task after 108h". Under D3 it is 43 h. Its founding case flips: task 3534
   (93.1 h, D = 9) scored 479.5 against 374.2 for the oldest D = 0 task (214.2 h); under D3 it is 93.1 + 28.3 =
   121.4 h against 214.2 h, and the chain head loses.
2. The live pending pool has a median age of 282 h, 90th percentile 1,329 h (3,176 anchored pending rows, read
   today). At 282 h today's age term is 398 of 500, so a work-product credit of 120 can never be overtaken by age.
   Under D3 it is worth 17 h. Within a tier dispatch becomes first-pending order; dependents and work product stop
   mattering for almost every real pair.
3. Row 7 passes today. The scores are floats: 500 x 720 / 792 = 454.5 and 500 x 1440 / 1512 = 476.2. The older one
   already goes first; nothing ties.
4. Pending task 3826 (the scoring PRD's integration gate) prints "the rank of the highest-D `high`-tier chain head"
   from the float score. It is not in section 5.

This is partly forced by R4: with uncapped age any fixed credit fades. It still has to be put to Leo as the change
it is, not as "today's exchange rate".

**Replace D3's sizing sentences with:**
> The credits cannot keep today's weight once age is uncapped. Two options for Leo:
> (a) fixed hours, 17 and 43: within a tier, order is by first-pending time for any two tasks more than about two
> days apart; the scoring PRD's worked case (3534) reverses; (b) the fixed hours of (a) plus a proportional term:
> age for dispatch is `(age + hours) x (1 + c_work + c_dep x f(D))`, so the credit grows with the wait as today's
> does. The coefficients are not chosen here. They are set in δ from a replay of recorded candidate sets, with the
> scoring PRD's worked case as the check (it holds from `c_dep` of about 2: 93.1 h x (1 + 2 x 0.66) = 216 h against
> 214.2 h). Either way δ reports, before and after, the rank of the highest-D chain head (task 3826's script).

Replace row 7 with: "a 30-day task with a work product and a 60-day task without, same tier and dependents: today
the 30-day task goes first (574.5 against 476.2); assert the order the chosen option gives." Add 3826 to section 5:
"amend at decompose: its script reports rank under `order_key`."

### A11. D2 and R4 say "first pending"; the ruling as recorded on task 8261 says "first eligible"

**Draft.** R4: "Age accrues without a cap from first pending to done." D2 (decided here): "It includes time spent
waiting for dependencies."

**Wrong or unproven.** Task 8261's amendment of 2026-11-13, the only record of this ruling in the store, reads: "The
age term is to become the scheduler's eligibility age, which Leo wants to keep accruing from first-eligible through
to done ... That needs a first-eligible stamp that survives dispatch, re-pend and restart, which does not exist
today." scheduler.md U2 leaves the same question open. D2 settles it as first pending and R4 restates the ruling in
D2's words. The two differ by the whole dependency wait, and Leo's standing practice is to park by adding
dependencies and re-pending: about 177 open tasks sit behind 8528 alone. With age uncapped, a task parked for two
months outranks, in its tier and at all three gates, a task that has been ready and starving for one.

**Replace.** Move the start of age out of "Ruled" unless Leo confirms the wording. In D2 add: "Task 8261's amendment
records the ruling as first-eligible. This PRD takes first pending because no first-eligible stamp exists
(`eligibility_clock.py::EligibilityClock` forgets at dispatch). Consequence: time parked behind a dependency counts
as age. The alternative is a durable `eligible_since` written once by the scheduler into task metadata the first
tick a task is a dispatch candidate, about one extra leaf. Listed first in the hand-back." Update 8261's delta note
to match whichever is chosen.

### A12. 6.2 reverses a recorded decision that a test guards, and the repair can only over-age

**Draft.** Section 2: "The earlier scoring PRD ruled that this must not happen ... the code does not deliver it."
6.2: "A `replace`-mode write that omits it keeps the stored value." "A one-off repair stamps rows that have lost it
with the task's earliest recorded time ... and marks them `pending_since_repaired`."

**Wrong.**
1. The drop is deliberate and recorded: task 3816's "design decision 8", stated in
   `sqlite_task_backend.py::SqliteTaskBackend.update_task`, documented twice in `docs/task-authoring.md` (§8 and
   "Mechanics": "It also drops the stored `pending_since` anchor, even one you copy into the blob"), and guarded by
   `fused-memory/tests/test_pending_since_anchor.py::test_replace_mode_still_drops_the_anchor`, whose docstring says
   preserving the anchor through replace "would be an unreviewed scope expansion". α's agent will meet a test named
   for a design decision and stall or escalate. The draft cites none of it.
2. The scoring PRD's D4 allows a back-fill only because it "can under-age a task but never over-age one, so it
   cannot manufacture a queue jump". The earliest recorded time is at or before first pending, so the repair
   over-ages, and uncapped age makes that a permanent jump. 17 of 18 in-progress rows have no anchor today, and 2 of
   those 17 have a `created_at` (the column exists; 383 rows are stamped, all recent). The draft does not say
   whether the repair reads that column.
3. `pending_since_repaired` is a new metadata key. It warns on every write until it is blessed in
   `shared/src/shared/task_metadata.py` and added to the machine-authored keys `strip_machine_authored_metadata`
   removes.
4. α's "the orchestrator's `replace`-mode writers" have nothing to change if the backend keeps the anchor.

**Replace 6.2's first two bullets with:**
> - This reverses task 3816's design decision 8. `update_task(metadata_mode='replace')` re-attaches the stored
>   `pending_since` (and `pending_since_backfilled`, `pending_since_repaired`) after
>   `strip_machine_authored_metadata`, in the backend only. α rewrites
>   `test_replace_mode_still_drops_the_anchor` to assert the opposite and corrects both passages of
>   `docs/task-authoring.md`. Reason: a fail-safe that reads as age 0 was sound when age was one term in a capped
>   score; under `(tier, age)` it sends the task to the back of its tier.
> - The repair stamps a row that has lost its anchor with the **latest** time it can be shown to have been pending
>   by: its first `task_started` or `lock_acquired` event, else `updated_at`. It under-ages, as D4 requires. It marks
>   the row `pending_since_repaired` (blessed and machine-authored in `shared/task_metadata.py`).

### A13. γ's rows 2 and 4 need the `pinned` tier, which only θ produces, months later

**Draft.** 8 γ: "Rows 1 to 4", needs "α, 8528". Row 2: "a pinned request arrives". Row 4: "a pinned normal-lane
request". 6.1: "`pinned: bool | None  # kept for task 9908's recorded rows`"; `rank` comes from `tier`. D5:
inheritance "is the existing walk"; θ is "the walk takes pins".

**Wrong.** Until θ lands nothing sets `tier` to `pinned`, and `order_key` does not read the `pinned` flag. θ waits
for δ, which waits for the S17 chain: twelve serial `scheduler.py` leaves (9665 to 9683), each with outside
prerequisites. So for months a pinned task has no rank at the merge pool or verify admission; goal 1 ("a pinned ...
task") is not delivered by γ; rows 2 and 4 cannot pass on γ's prerequisites (G6 branch 3). κ, the signal leaf, waits
for θ as well.

**Replace.** In α: "`Scheduler.task_priority` returns `tier='pinned'` and the row's `pin_order` for a task whose own
override row is pinned. θ later adds inheritance to prerequisites and the scheduler's own use of the tier." In
section 8 name the edge: δ depends on task 9683 (S17 ω, the last leaf that edits `scheduler.py`; confirm at
decompose). Move the pool half of row 5 (a prerequisite precedes a critical request) to κ.

### A14. Three supersessions and one dependency strand other tasks

**Draft.** Section 5 and the decompose notes: "cancel 8932 and 6821, close 5623 as superseded"; β needs "α, 8261";
"8261 lands its key; this PRD (β) replaces it".

**Wrong.**
1. Task 9679 (S17 η) depends on 6821 and its text says "η leaves `_pending_anchor`/`_anchor_pending_since` where 6821
   put them". 9679 is on the `scheduler.py` chain δ and θ wait for. Cancelling 6821 leaves 9679 with a cancelled
   prerequisite and a false premise, which blocks the chain this PRD depends on.
2. 8261 has been refused its files 52 times in 9.9 days (study report section 7), is pinned, and is still pending.
   β and κ inherit that wait. Its key is a blend (`arrival + tier_rank x K`), which β then deletes; 9705 (S16,
   47 files) depends on 8261 and moves the same knobs.
3. Closing 5623 leaves 9908's recorded `first_enqueued_at` wrong after a restart (9908's tip commit made
   `MergeRequest.first_enqueued_at` the one age the lane records).

**Replace.**
> - 6821 is not cancelled. δ deletes `age_anchors.py` after the chain; or, with Leo's confirmation, 6821 is cancelled
>   **and** 9679's dependency and its "where 6821 put them" sentence are removed in the same decompose step.
> - 8261 is amended to be β: it implements `order_key` at admission directly, depends on α, keeps its id, pin and
>   dependents (9705). No blend key is built. Its knob `verify_admission_priority_age_secs` becomes the promotion
>   period.
> - 5623 stays open, re-scoped to the recorded field.

### A15. Two seam rows give work to tasks whose text does not contain it

**Draft.** Section 5: "Task 9144 ... this PRD (η) adds the field; whichever lands second maps it to a kind." "Task 7921
... the verdict book is the 'main is red' read | Owner 7921 | ε reads the lane halt until it lands". 6.6: "`reason`
starts with a new prefix constant".

**Wrong.** "Whichever lands second" is the reciprocal form G4 exists to catch. 9144's text does not mention
`conviction`; if it lands second its agent will not map it, and 9144's follow-up makes `kind` mandatory for blocked
outcomes. 7921 was filed on 10-31 and knows nothing of the pool, so nobody switches the pool's main-red read to the
verdict book. 9144's title says "reason strings frozen", and the quality PRD freezes the 17 reason prefixes; a new
prefix beside a structured `conviction` field is a meaningful string where the structure already says it
(code-quality heuristic 12).

**Replace.** "η owns both orders: if `MergeOutcome.kind` exists when η runs, η adds the member; if not, η files the
follow-up as a dependent of 9144. At decompose a dated note is added to 9144 and to 7921 naming the pool's field and
read. The conviction outcome uses the existing 'Post-merge verification failed' reason builder; `conviction` is the
only discriminator. No new prefix."

### A16. The pair rule, row 11 and I3 describe states that cannot occur

**Draft.** 6.5: "If two requests have failed together twice and each has since passed without the other, the
lower-priority one is convicted." Row 11: "both halves then pass: all four return to their priors and are eligible
for clean batches." I3: "If every part of a failed set later passes, its members' `p_bad` returns to within 1e-9 of
their priors."

**Wrong.** A batch that passes lands. A request that "has since passed without the other" is on main, not in the
pool; the second request is then verified on top of it and fails as its own run (the lane already labels that
`integration_skew`). The pair rule can never fire, has no boundary row, and contradicts the model, under which a
request in a passing observation has p_bad exactly 0. Row 11's four requests have landed; none is "eligible". I3 is
true only because landed members are deleted from the observations; by the model their p_bad is 0, not the prior.

**Replace.** Delete "Pairs", `'pair'` and `counterpart_task_id`. Row 11: "a batch of 4 fails; both halves then pass:
all four land, no suspect remains, and the failed observation is counted as a false failure in the readout." I3: "A
failed observation all of whose members have landed contributes nothing to any posterior. A request that passed in a
verify whose landing was aborted (stale CAS in `_land_chain_prefix`) has `p_bad` 0 until its content id changes."

### A17. Failures caused by a red main become suspects, and convictions rise when the lane is failing

**Draft.** D15: "Beliefs are not updated from verdicts given while main is known red. ... 'known red' is the halted
normal lane." 6.3: "`prior_default` (one minus the rolling per-request pass estimate task 8528 maintains ...)".

**Wrong.**
1. The lane is halted only after a failed verify has been probed and main confirmed broken
   (`_auto_heal_main_health_deferred`). The failure that revealed the break, and every batch that failed between the
   break and the halt, were recorded while main was not yet "known red". Their members are suspects. After the heal,
   if two of three pass, the third is convicted by elimination for a fault that was main's.
2. 8528's estimate is over the last 30 verdicts and swings: 10th percentile 0.52, 90th 0.97 in the recent window
   (measure.md item 4). At q = 0.03 elimination gives 0.51 and never convicts; at q = 0.48 it gives 0.97 and convicts
   freely. The prior is highest during a run of failures, which is when main, a host or a flaky test is the likelier
   cause.

**Replace.** D15: "When main is confirmed red at commit M, every failed observation whose tested tree contained M is
marked `main_red` retroactively and excluded, including the one that triggered the probe." 6.3: "The rolling
estimate sizes batches only. Conviction uses `prior_conviction`, a slow value (the 14-day single-request failure
share, refitted by ι), and convictions stop, with one L1, while more than `convict_storm_max` (default 3) have fired
in an hour."

---

## B. Minor defects

1. **Section 2, "0.879 and 0.894".** measure.md item 4 gives 0.874 (recent) and 0.894 (paused). Write 0.874.
2. **Header, "`task/9908` at `e18d22c403`".** The branch is at `8c6d25dad6`, which adds
   `MergeRequest.first_enqueued_at` and `in_graph` on the pick record. Also: `read_task_priority` rebuilds
   `TaskPriority(tier=..., pinned=...)`, so new fields are dropped by the guard unless α edits it; and
   `UNKNOWN_PRIORITY = TaskPriority()` needs every new field to default to None. Say both in α.
3. **6.3, "0.016 to 0.07 per verify".** That is the joint rate. The model's `false_failure` is conditional on no bad
   member: divide by (1 - q), giving 0.02 to 0.09. 0.03 is still inside. Add validators: 0 < `false_failure` < 0.5,
   0.5 < `convict_threshold` <= 1. I2 needs `false_failure` = 0, where a pass after a fail of the same content makes
   the normaliser zero; say `infer` raises `ContradictoryObservations`.
4. **6.1, "`waited_secs`".** Undefined at the merge pool. Use the request's own `enqueued_at` (journaled, matches the
   episode rule the 9 h default was measured with), not the lineage's first submission.
5. **6.1, failed lookup "is counted".** Background legs at verify admission have no task, so every one counts as a
   failed lookup. Exempt them.
6. **6.1, promotion floor.** A low-tier waiter promoted three times ties with real criticals and beats the younger
   ones on age. State it or stop one tier short.
7. **6.7, "`merge_dequeued` shows the pick at the head of `waiting` by `(tier, age)`".** The pick record carries no
   age and no promoted rank. Add `age_secs` and `rank` to the picked entry and each waiting entry, or a promoted
   low-tier pick reads as a violation.
8. **6.7, "No other code enumerates the lane buffers in pick order".** Also: `get_merge_queue` and `merge_status`
   positions, `merge_heartbeat.head_of_line`, 9908's `PickRecords._snapshot`, `drain_lanes_for_shutdown`,
   `_rebucket_relaned_requests`. List them for γ.
9. **6.7, "members chosen, members actually dispatched".** Say that truncation by `build_chain` drops every later
   member, and that dropped members are not in the observation.
10. **6.5, "connected groups", "nearest 0.5".** Define: two suspects are connected when one remaining failed
    observation contains both. On a tie take the shorter prefix. In shadow, groups can exceed 16 because 8528 mixes
    suspects with clean requests; `TooManySuspects` must then be written on the decision row, since shadow cannot
    send anyone to a solo verify.
11. **6.6, "a wrong conviction costs one steward pass and one verify".** Also one of three `max_merge_retries` and a
    rebase. A wrong conviction is counted only if the steward leaves the content unchanged; say the count is a lower
    bound.
12. **6.3, "`Observation.own_run  # exactly one member`".** After landed members are removed a batch observation can
    have one member. Comment: "verified alone".
13. **Section 8, sizes.** η (suspects, conviction, deadline, restart, pick and placement wiring, `types.py`,
    `workflow.py`, twelve rows) is beyond one leaf; split into suspect batches and conviction. θ spans four packages
    and well over twelve files once the snapshot, dashboard and seven tier enumerations are counted.
14. **Section 8, same-file edges.** β, γ, δ and ζ all edit `config.py` with edges only γ → ζ → η and α → δ → θ. α
    edits `scheduler.py` with no S17 edge. Tasks 9705 and 9709 move the same config knobs. Add edges or put
    `merge_pool` in its own file.
15. **Section 8 note, "no dependency on the extraction tasks; the file lock serialises them".** Not safe as written.
    A leaf filed against `worker.py` holds no lock on `speculation.py` or `verify_dispatch.py` if 5040 or 5041 lands
    first; and each touched function pinned in `merge_lane_ratchet_baseline.json` needs an authorised raise. Either
    make ζ depend on 5041 and η on 5042, or list both homes in each leaf's files and name the raise path.
16. **Section 8, "9908 landed and deployed", "ζ deployed".** Not dependency edges. Use a `delivered_checks` gate or a
    deterministic gate task. ε's property test "against the real event store" cannot run in the merge gate; commit a
    fixture exported from real rows and name its date.
17. **Section 5, missing rows.** 6328, 3824, 3825, 3826 (A9, A10); 9636 (watchdog pins become the `pinned` tier);
    6642 (a new tier is its lockstep); 6658 (premise changes with a durable anchor); 7009 and 4709 (pin source and
    override events); 6722 and 9705 (validators and knobs); 8933 ("must land before or with": the red-tip requeue);
    9137 (host selection); 9182, 9152, 9723 (`intake_buffers.py`); 9188 (queue position).
18. **6.1, "written into the project's config with its date".** A script must not edit a machine-operated main
    checkout. It prints; β commits dark_factory's value on its branch; other projects are an operator reload. D9
    says 14 days, but the other projects' figures are 28-day windows, and the dark_factory window includes the train
    era measure.md calls least reliable.
19. **D1, two `task_priority.py`.** After 6718 there is `shared/task_priority.py` (names, ranks) and
    `orchestrator/task_priority.py` (the value, `order_key`, weights). Say the shared module owns names and ranks,
    `pinned` included as an effective-only tier, and the orchestrator module imports them.
20. **D4, "counted on the tick event".** This removes the scoring PRD's arming gate (D12), which exists because a
    bulk loss of anchors reorders the fleet silently. Keep a threshold: escalate when unanchored candidates exceed a
    share for a streak of ticks.
21. **C1, the readout joins `merge_convicted`.** ι lands before η. The join belongs to η, or ι prints "no conviction
    rows: pool not on" as its own state.

---

## C. Invariant walk

| Invariant | Mechanism as written | Hit |
|---|---|---|
| `contracts-machine-checked` | Routing a conviction by a new reason prefix (6.6); no validators on `false_failure`, `convict_threshold` | Yes (A15, B3) |
| `structured-facts-at-failure` | "Same head" derived from a SHA that a rebase changes; `p_pass` recorded for a batch other than the one verified | Yes (A5, A8) |
| `corroborate-before-acting` | Conviction and batch choice act on a posterior computed off the loop from a cache; no re-check that the request still waits with the same content, and in-flight requests can be convicted | Yes (A4, A2) |
| `storm-escape-required` | Conviction, `TooManySuspects` fallback, deadline solos and failed lookups have counters but no rate threshold; D4 removes the arming gate | Yes (A17, B20) |
| `no-lockstep-duplication` | `TIER_WEIGHT` and a rank table with `pinned` beside `config.PRIORITY_RANK` and five other enumerations; 8528's positional suspects and the pool's computed order must agree | Yes (A3, B19) |
| `status-matches-liveness` | A conviction resolves through the request's future and `_mark_blocked` | No, once A4's "only waiting requests" is added |
| `holds-owned-and-bounded` | Suspect hold bounded by S1, which cannot be met; no rule under a halted lane | Yes (A6) |
| `loop-thread-occupancy-bounded` | `infer` off the loop (I1): one call at 16 suspects takes 0.39 s in plain Python (`critic-scratch/t.py`), and a prefix search makes up to 16; the rebuild at start reads the event store | Advisory: state a budget, factor by connected group, bound the rebuild read |
| `one-fact-one-home` | The age ruling is recorded on task 8261 as first-eligible and in the PRD as first pending; main-red has two reads with no owner for the switch | Yes (A11, A15) |
| `guards-exercise-behaviour` | Row 7 passes before the change; row 11 asserts an impossible state; ε's property test targets a store CI lacks | Yes (A10, A16, B16) |
| `no-silent-fail-soft` | A shadow decision that raises writes no row, the same as `mode=off` | Yes (A8) |
| `exceptions-owned-or-ratified` | No allow-list added; ratchet raises go through the existing authorised path | No (B15 names the path) |
| `readers-prove-their-producer` | ι's readout handles "no rows yet"; ε's reader must treat rows without members as unknown, not single; the conviction join lands before its producer | Yes (B21) |
| `sampler-vocabulary-one-table` | Not touched | No |

---

## Would it pass META as it stands?

No. Decomposed without oversight, the agents would build a batch chooser that has no place in the lane's
pick-then-dispatch structure, an observation log that reads voided and re-verified rows as evidence and forgets
convictions on restart, and a pool order that silently breaks the 8528 bisect it names as its fallback; the human
gate would then be shown a calibration readout that compares predictions for one batch with the verdicts of another.
The three changes that matter most: **(1)** restate sections 6.3 to 6.7 against the lane as it is: head at the pick
and members at dispatch, an observation defined at finalize with its own row, the own-run rule, a content id for
"unchanged", and suspect rules whose bounds can be met (A1, A2, A4, A5, A6, A16); **(2)** make the fallback and the
graph hold: 8528's suspects first and an `order` switch, `p_pass_dispatched`, the `pinned` tier in α, and the 6821,
8261, 9144 and 7921 seams (A3, A8, A13, A14, A15); **(3)** take the scheduler half back to Leo with the numbers
rather than as settled: first pending against the recorded "first eligible", credits that reverse the scoring PRD's
worked case, the reversal of design decision 8, and a pin reservation that stays bounded (A9, A10, A11, A12).
