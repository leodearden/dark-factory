# MCP merge requests in coalesce trains: investigation memo

**Date:** 2026-10-26, amended 2026-10-29 · **Status:** RULED on 2026-10-29 (§0). Sections 1–7 are the 2026-10-26 discussion draft, kept as the record. Where they conflict with §0, §0 wins.
**Code pinned at:** main `70063c3638` for the draft. The §0 and §4.4 additions were read on main `6d2f46a04b`.
**Evidence window:** `data/orchestrator/runs.db` to 2026-10-26T16:28Z, plus orchestrator journald from 2026-10-07. §0 and §4.4 add chain events to 2026-10-29T14:41Z. Every figure's counting rule is in Appendix A, and the seats that produced the figures are in Appendix B.

## 0. Ruling and amendment (2026-10-29)

**What the draft missed.** Deep merge-ahead chains are a second way to land several queued singles with one verify, and **they do not care who submitted a request**. They come from `plans/deep-merge-ahead-prd.md` and are switched on by `merge_deep.chain_cap`. So MCP singles are already batched today, through chains.

- **When they went live here.** This repo's yaml flipped `chain_cap` to 6 on 2026-10-27 at 21:42Z (carrier 7619, task 5057). The first chain verify ran at 2026-10-29T01:17Z.
- **Why the draft missed them.** Chains were off throughout the draft's evidence window, where every `merge_verify` row had `chain_items` = 1. No seat looked at them.
- **How they compare.** §4.4 sets chains against coalesce trains.

**Leo's rulings, 2026-10-29.**

1. **A1 is withdrawn.** Admitting MCP singles to coalesce trains is superseded, because chains already batch them without any of the membership machinery A1 would need. §3 and §4.1 stay as the record of why coalesce trains can't simply admit MCP singles.
2. **`merge_train_coalesce_enabled: false`** in `dark-factory-orchestrator.yaml`.
   - The knob is not in `RELOADABLE_FIELDS`, so it takes effect at the next fleet restart.
   - Automatic coalescing of independent singles is treated as obsolete unless the measurements below say otherwise.
   - **Declared atomic trains are kept.** They are the only way to land interdependent stacked work as one merge (§4.4).
3. **Stream 1 goes to `/prd`**, worked by a sibling `/team` session, re-scoped for a lane with coalescing off:
   - **Kept, now applied to every verify:**
     - a seconds-cheap whole-tree screen (`inline_suppressions.py --check` and the other cheap `scripts/` checks) run before a merge item's verify. It matters most per link before a chain tip is built, because a link with its own defect fails the tip and halves the chain depth.
     - never start a verify (chain tip, speculative, or a declared train) on a base the lane has just judged main-red.
   - **Kept while the declared-train former is enabled** (`merge_train_former_enabled: true`): the bare-id `get_changed_line_ranges` fix, §6.1.
   - **Moot while coalescing is off:** unwinding on exceptions and `stop()` (it lives in coalesce-only code; task 6450 keeps it as-is), and the `has_workflow_waiter` read (the coalescer is its only consumer).
   - **Not low-risk; needs a code read and a measurement first:** reordering the verify legs to fail fast.
4. **Stream 2 is moot while coalescing is off.** It served members of coalesce trains that didn't land.

**What to measure in the chain canary week.**

| Measure | What it tells us |
|---|---|
| How often chains fire: the share of speculative dispatches that build a chain with `chain_items` ≥ 2 | Whether chains actually do the batching |
| Items landed per chain verify | How much each verify buys |
| MCP singles landed via a chain (`landed_via_chain`) | Whether MCP singles get batched in practice |
| Chain-attributed reds | The canary's existing stop rule |

**Baseline for the canary.** Since 4953, 135 of 435 single verifies were speculative, and only speculative dispatches can chain. Since the flip there have been 2 chain verifies: 3 items on `local` (47.4 min, passed) and 2 items on `laptop` (16.9 min, passed). 2 links landed via a chain.

**Still open for Leo.**

- **The declared-train former.** The workflow-side train former (`merge_train_former_enabled: true`) is the same kind of opportunistic batching as coalescing. It formed no trains in the evidence window. Turn it off too?
- **A measurement gap.** How many of the 56 post-4953 coalesce trains formed while no chain could have fired was not measured. That is the only coverage trains had that chains lack.

## 1. The question, and the short answer

> The short answer below is the 2026-10-26 draft. It is superseded by §0: chains already batch MCP singles, A1 is withdrawn, and coalescing is switched off.

**Question (Leo, 2026-10-26).** How do we make automatic coalesce trains work reliably for merge requests submitted over MCP (interactive `/unblock`, `/unblock-low-risk`, `/merge-queue`, hand carries), instead of excluding them?

**Short answer.**

1. **The exclusion has a real cost, but only a modest one.** Since task 4953 (2026-10-13), at most 19 of 82 MCP singles had an eligible train partner waiting at the same time with a clean own-branch history.
   - Admitting them would have bought **at most about 2.5–8.5 verify host-hours per 12 days (1.1–3.9% of verify compute)**. That is about what all trains net today: +8.2 to +9.6 h.
   - It is an upper bound because it ignores stackability and partner choice, and because a new 2-member train saves less per member than the average.
   - Most MCP singles never wait beside a partner: 48 of 82 waited under a second.
   - The eligible population is bursty and mostly hand carries. 12 of the 19 were hand-carry carriers or hand-carried fixes, and at least 4 of those came from a single `/team` session on 10-19/20.
2. **If we admit them, one design shape works: the held-member contract (A1, §4.1).**
   - The lane owns train membership, and joining a train writes no task status.
   - The absorbed request stays live: its journal row, registry slot and the submitter's handle all survive.
   - At the slot, each member is checked by its own readiness rule.
   - On landing, the lane credits `done` from the hold the submitter placed. On any exit that doesn't land, the lane re-queues the member solo under its own request id.
   - It needs no Table A edge.
   - The alternatives fall short. Parking MCP tasks at `merge-deferred` is blocked by Table A, the deferral write authority, the lack of a status compare-and-set, and recovery writers that all write `pending`. Handing the merge back to a `TaskWorkflow` founders on a false premise: a re-dispatched worktree is rebased, so the fast path rarely fires (§4.3).
3. **Today's trains have problems of their own that have nothing to do with MCP.** Trains net +8–10 h per 12 days, after 10.8 h of derail waste. That waste breaks down as:
   - 5.9 h from members with their own defects that whole-tree gates catch late;
   - 2.6 h from a red main;
   - 1.8 h from flakes;
   - 0.6 h from a train started on a base the lane had just judged red.

   Separately, fleet-drain dissolutions waste no verify time but cost each member a median 11.6 h. Largely, though confounded by restarts and a scheduler pause, that is because re-pending sends members back into a dispatch queue 1,636 deep. None of this involves MCP, and fixing it costs about as much as A1. The one piece they share is the routing of every non-landing exit through a single funnel.

**Recommendation (§5).** Two streams that pay off on today's traffic, plus a separate decision:

- **Stream 1: cheap gate fixes.** About 4–5 low-risk tasks; saves verify compute.
- **Stream 2: a lane-owned solo re-queue for members of trains that don't land.** About 3–5 medium-risk tasks; saves member latency and agent pipeline runs, not verify compute.
- **The decision: A1 admission.** About 5–6 more tasks; medium-high risk. Take it after measuring A1's real reach over a longer window, and ideally as an opt-in for hand carries. A cheaper, unevaluated alternative aimed at the actual population is a submitter-declared train for a hand-carry session's own carriers (§4, B5).

## 2. The measured premise

### 2.1 MCP submissions are about a sixth of the lane's work

- **Share of landings.** Since 2026-08-01, 295 of 1,914 landings (15.4%) were MCP submissions. Outside the 08-27→08-31 incident spike, the weekly share is typically 9–20%.
- **Share of verify compute.** MCP-origin requests used 217.8 of 1,283.4 verify host-hours (17%).
- **Since 4953.** 57 MCP solo landings took 25.7 verify host-hours, of which 22.3 h passed. All MCP-origin verify rows, landed or not, came to 33.2 of 219.5 h (15%). MCP queue-to-landing time had a median of 42.5 min and a p90 of 194 min.

The event store does not record the submitter. "MCP" here means *no workflow merge phase was open when the request was queued*. Journal-recovered requests are traced back to their first submission. The rule labels all 21 known MCP landings correctly. Its precision cannot be measured.

**The data does not suggest that MCP members would poison trains more than workflow members.** Since 4953, an MCP single passed its first merge verify 81% of the time (55 of 68) and a workflow single 75% of the time (211 of 283). Since 08-01 the rates are 80% and 82%. Both gaps are within sampling noise.

### 2.2 How many MCP singles could have joined a train

These are nested filters over the 82 MCP singles since 4953, using the lane's own **run-scoped** risk predicate. `worker.py::SpeculativeMergeWorker._default_coalesce_exclusion_reason` calls `EventStore.latest_merge_finalized`, whose `cross_run` default is `False`.

| Filter | Count | Notes |
|---|---|---|
| MCP singles since 4953 | 82 | |
| waited ≥ 1 s in the lane buffer | 34 | 48 waited under 1 s. Pass timing isn't observed, so "no pass saw them" is inferred. |
| had any other single waiting at the same time | 29 | |
| had an eligible workflow partner and a clean own-branch history in the same run | **19** | 6 appear as `no_awaiting_workflow` exclusions in trains that formed without them; 13 left no event because the pass ended with fewer than 2 candidates |

Separately, 16 of the 82 had a last outcome of `blocked` or `error` in the same run, which the predicate would exclude anyway. Counting across runs would give 27, which is the first draft's figure and wrong.

**Who the 19 are, by task title and description.**

- **11 hand-carry carriers** ("Carrier: land …" or "Hand-carry carrier"): 7033, 7035, 7149, 7197, 7204, 7231, 7252, 7287, 7293, 7328 and 7354. At least 4 of them (7197, 7204, 7231 and 7252) came from one `/team` throughput session (`team-dark_factory-440934`, 10-19/20).
- **1 hand-carried fix**: 7032.
- **6 `/unblock`-type or interactive tasks**: 4503, 5405, 5701, 6213, 6669 and 7174. Five of them have plan/execute history.
- **1 `/warm` slug branch**: `iact-stream-touch-units`.

The population is bursty, so 12 days is a short window to size it from. **Their status, deferral kind and worktree path at submit time were not measured.** Those three facts decide A1's reach (§4.1, §5).

### 2.3 What a train saves, and what derails cost

| Window | Merged trains | Train verify | Saved vs solo | Derailed trains (verify ran) | Wasted | **Net** |
|---|---|---|---|---|---|---|
| 2026-08-01 → 4953 | 57 | 48.5 h | 33.8–37.6 h | 28 | 25.6 h | **+8.2 to +12.0 h** |
| 4953 → 2026-10-26 | 31 | 19.7 h | 19.0–20.5 h | 12 | 10.8 h | **+8.2 to +9.6 h** |

**How savings are counted.** Each train verify is compared with the solo baseline on the **same runner**. `laptop` verifies run about 3× faster than `local`, and trains mostly run on `local`. That is partly structural: `worker.py::_TRAIN_REMOTE_VERIFY_MAX_MEMBERS = 2` pins 3-member trains to `local` until task 6106 lands.

The basis matters:

- **Same-runner basis.** A merged train's verify costs a median 1.13× one solo verify after 4953 (1.45× before). Trains save about 9% of verify host-hours gross and about 4% net.
- **Solo-verify equivalents.** Counting minutes divided by that runner's solo median, the net after 4953 is +16.9 of 536.8 equivalents (3.1%). The conclusion is the same.
- **Superseded figure.** A first-pass figure of "net −2.7 h" compared trains against a laptop-heavy solo median. It was a runner-mix artifact.

### 2.4 What admitting MCP singles would buy, as trains perform today

- **Per member.** Each member landed in a merged train since 4953 saved about 15.9 verify-minutes (19.0 h over 72 members). Net of derail waste across all 104 members of started trains, each saved about 4.7 min.
- **Member-equivalents.** The 6 MCP singles that had an exclusion event would have joined a train that formed anyway, so each counts once. For the 13 with no event, admission would have *created* a 2-member train, saving the partner's share too, so each counts twice. That gives 6 + 13 × 2 = 32 member-equivalents.
- **Value.** 32 × 4.7–15.9 min ≈ **2.5–8.5 h per 12 days (1.1–3.9% of 219.5 h)**. This is an upper bound: new 2-member trains save less per member than the average, and stackability and reach are ignored. Most of it comes from hand-carry campaigns, so it varies with how often Leo runs them.
- **Latency for interactive submitters.** A train's formation-to-start median is 178 min after 4953 (a selection-biased figure), against a 42.5 min solo median for MCP queue-to-landing. Absorbed `/unblock-low-risk` runs would hit their polling deadline (`handed_off`) more often.

### 2.5 Why trains still fail, whoever submits

After 4953, 56 trains formed. 31 merged, 15 derailed, and 10 were dissolved or lost before their verify started. **None involved an MCP member.** A root-cause pass over every failure found the following.

- **Union-verify derails (11).** None shows a genuine cross-member interaction, and none is a ratchet-ledger conflict between members.
  - **8 were members with their own defect.** The member fails the cited gate alone, or failed the same test solo. Typical checks: the `inline_suppressions.py` ratchet, drift guards, ratchet baselines. The package-scoped workflow verify never runs the `scripts/` leg, so the ratchet first trips at the whole-tree merge gate. It did so for 34 distinct tasks in 12 days. A 3-second check fails 6–7 min before the end of a 35–49 min train verify, and the legs after it still run. One bad member sinks the train: in 7071, 6856 and 5737, two clean members were lost with each defective one.
  - **3 were flake or infra:** 6588, 6201, and 6413, which is only leaning that way.
  - **The "Train union verify failed" tag over-reports interaction.**
- **A red base (1).** The lane judged train 6057's base red ("reproduces on bare main") at 08:57:23Z, then started train 5445 on the same base 64 s later. 5445 burned 38 min.
- **Rebase conflicts (3).** In each, a member conflicted with *main*, not with another member. A working pair screen would have prevented none of them.
- **Dissolutions and losses (10).** Every post-4953 train that a fleet drain caught in the buffer was dissolved; 0 survived. A train waits a median of about 3 h from formation to start (2.3 h for those dissolved), and a drain arrives about every 10 h.
  - Dissolution wastes 0 verify hours.
  - Members then take a median 11.6 h to land. About 8.5 h of that falls before their next `merge_queued`.
  - Dissolution flips members to `pending`, so each one rejoins a dispatch queue that holds 1,636 pending tasks against 24 in progress. The figure is confounded by restarts and a 10-26 scheduler pause.
  - E3 estimates the saving forfeited to dissolution at about 4.6 h.
- **Innocent members get hit too.**
  - 7 of the 15 post-4953 one-strike exclusion rows are innocent members (tasks 7071, 6703, 7394 and 5726). Three of those come from union-verify derails, where there is no attribution today.
  - 17 of the 29 members of union-derailed trains were still `pending` 0.7–3.3 days after the derail. That is the same deep-queue mechanism.

## 3. Why the exclusion exists: what `merge-deferred` does for a train member

Before 4953, an MCP single absorbed into a train failed the train's slot pre-check. That check requires every member to read exactly `merge-deferred` (`orchestrator/src/orchestrator/merge_lane/worker.py::_do_train_merge`, step (a)). Of the 164 trains formed before 4953, 52 (31.7%) were `train_incomplete`, and 42 of those held a member no workflow had parked. After 4953 there were none.

The pre-check is only the visible part. The `merge-deferred` status, together with the workflow that wrote it, gives a train six things. An MCP single has none of them today.

| What a train needs from a member | Workflow member today | MCP single today |
|---|---|---|
| **It is not dispatchable while the lane rewrites its branch** | `merge-deferred` is not dispatchable, and the stranded sweep and warm-lane reclaim leave it alone (`harness.py::_RECONCILE_SWEEP_STATUSES`, `_WARM_LANE_RECLAIM_PROTECTED_STATUSES`). | Depends on the hold. `deferred` with `metadata.deferral` holds. `blocked` holds only while a pinning escalation or a live claimant exists; otherwise `scheduler.py::Scheduler._phase_redispatch_stranded_blocked` re-pends it within minutes. `pending` is dispatchable. An orphan `in-progress` can be re-pended. |
| **Someone credits `done` on landing** | `mark_member_done` writes `merge-deferred → done`, kind `merged` (`harness.py::build_train_callback_factory`). | The submitter stamps it by hand. On the coalesce arm, the skills' landed-but-not-credited veto (`skills/merge-queue/SKILL.md` rule 2v) forbids that stamp unless the scheduler already reads `done`. |
| **A non-landing exit re-drives it** | `merge_lane/member_redrive.py::MemberRedriver.redrive` acts on `merge-deferred` only: `found_on_main → done`, otherwise `pending`. | `NOT_PARKED`. Nothing is written and nothing is resubmitted. The submitter has already been told `superseded`. |
| **A restart recovers it** | Trains are never journaled, and absorbed singles leave the journal on absorption. `merge_deferred_recovery.py::AbandonedMemberSweep` re-drives orphaned `merge-deferred` members after two 900 s passes. | Every durable trace is gone at absorption: the journal row, the registry slot and the lifecycle entry. The sweep reads only `merge-deferred` rows. |
| **The submitter can observe the outcome** | The workflow parked in `_handle_superseded` and exited. The `done` flip is the signal. | The request stays `superseded`, and the train id polls as `stale_record`/`coalesce_train_id`. The skills fall back to the tip marker plus task status, and the veto then stops them. |
| **The lane's addressing matches the request** | Branch `task/<id>`, worktree `.worktrees/<id>`. | `merge_request` takes `task_id`, `branch` and `worktree` independently. Stacking, unwind and re-drive address `branch_prefix+task_id` and `worktree_base/task_id` (`git_ops.py::GitOps.stack_train_branches`). |

**Why the lane can't just "park and restore" a held MCP task.**

- **Table A has one way in.** Table A (`shared/src/shared/task_transitions.py`, enforced) has exactly one edge into `merge-deferred`, and it comes from `in-progress`.
- **The deferral record can't be re-made.** Leaving `deferred` atomically deletes `metadata.deferral`. Only `set_task_status(deferred, deferral=…)` can re-create it, and a `held_by_session` hold must name a live `claude` pid.
- **No compare-and-set.** `Scheduler.set_task_status` has no expected-previous-status guard.
- **Recovery writes `pending`.** Every existing recovery writer writes `pending`.

### 3.1 Hazards any admission design must close

1. **Absorption leaves no durable trace.** A coalesce train's outcome has no consumer, and an absorbed MCP single has no durable trace. Restart, `stop()`, an exception escaping `_do_train_merge`, or a derail all drop it silently.
2. **One-strike marking hits every member** on any `blocked` outcome (`_mark_coalesce_derailed`, 7-day TTL).
3. **Re-drive and unwind skip non-parked members.** The re-drive filters on `merge-deferred`. The unwind's restorable statuses exclude `deferred` (`merge_lane/coalesce_unwind.py::_RESTORABLE_STATUSES`).
4. **The delivered-checks withhold reverts the member to `pending`.** For a task that was held, this releases the human's hold.
5. **Status changes mid-flight.**
   - The stranded-blocked sweep re-pends an unpinned `blocked` task.
   - An L2 `resume` re-pends a `blocked` task. One that arrives while the task is `merge-deferred` is skipped at DEBUG level (`harness.py::Harness._cascade_unblock_member`).
   - `restart`, `park` and `abandon` overwrite any non-terminal status.
6. **Cancellation.**
   - An absorbed request cannot be cancelled: the train is refused with `train_cancel_unsupported` until task 7309 lands.
   - The server's waiter path in `escalation/server.py::merge_cancel` bypasses `_coalescing_ids`. On a live absorbed future, it would reap a worktree the train is using.
7. **Non-tip members are not ancestors of main** after the tip's rebase, so credit must come from the landing itself, not from ancestry.
8. **Stacking rebases a live interactive worktree in place.**

## 4. Candidate designs

Two opus seats evaluated the brief's four directions and proposed others: D-A on the held-member family, and D-B on delegated owners, cheap derails and do-less options. The full records are in Appendix B. A critic seat then re-checked the verdicts.

| Option | One line | Table A | Cost | Verdict |
|---|---|---|---|---|
| **A1 Held member, lane credits** | The lane owns membership; joining writes no status; the absorbed request stays live; the lane credits `done` from the hold; the member is re-queued solo on any non-landing exit | none | ~6 tasks, plus the owner signal; ~5–6 after Streams 1–2, which supply only the exit routing | **the shape to build if general admission is wanted** |
| B5 Submitter-declared train *(surfaced during integration; no seat evaluated it)* | One hand-carry session submits its own N carriers as one `GroupMergeRequest`. The submitter holds every member and credits them all; if the train derails, the session resubmits solo | none | likely 2–3 tasks, unverified | **evaluate**: it fits the measured population, but captures no MCP+workflow pairs |
| A3 Held member, submitter credits | As A1, but the submitter stamps `done` as it does for a solo landing | none | ~5 tasks | viable; a dead submitter leaves a non-tip member uncredited |
| A2 / B1a Lane- or adopter-owned park | Park absorbed MCP tasks at `merge-deferred`; restore the hold on a non-landing exit | +3–5 edges | 6–11 tasks over 4 packages | reject |
| B1b Adopter walks legal edges | `blocked/deferred → pending → in-progress → merge-deferred`, held by a pinning L1 | none | 5–7 tasks | poor: a second ownership model enforced at four writers |
| B2 Hand back to a `TaskWorkflow` at the merge phase | Stamp `merge_retry_pending` and re-pend | none | 3 tasks | **reject as specified** (§4.3) |
| B3 Cheaper, safer derails | Unwind on exceptions and `stop()` (Stream 1); culprit-attributed strike (Stream 2). *The B3f re-drive stamp is dropped (§4.3)* | none | 1–2 tasks | **in Streams 1–2** |
| B4(i) Read the live waiter fact | The coalescer admits a request a live workflow actually awaits (`has_workflow_waiter`) | none | 1 task | in Stream 1; value unmeasured |
| A4 / B4(ii) Admit only taskless members and carriers already at `merge-deferred` | No slot change | none | 1–4 tasks | reject: the population is about zero, and absorbed requests are lost |

### 4.1 A1, the held-member contract

**Admission.** The coalescing pass sorts each candidate into one of three **member kinds**:

- **`WORKFLOW`**: the flag is set, or a live workflow waiter is attached.
- **`HELD`**: the task is `blocked`, `deferred` with any `metadata.deferral` (the `planning` kind included, because the standard hand-carry carrier is filed with `planning_mode=True`), or `merge-deferred`.
- **`TASKLESS`**: no task row exists.

A `HELD` or `TASKLESS` request is admitted only if:

- `branch == branch_prefix+task_id`;
- `worktree == worktree_base/task_id`;
- its lane is `normal`;
- the risk predicate passes;
- a knob, default off, is on.

`pending` and `in-progress` tasks stay excluded.

**Absorption.** A held request is **not** resolved `superseded`. It moves to a new lifecycle state, `ABSORBED`. Its future, registry slot, server waiter and journal row (annotated `absorbed_by_train`) all stay live. The train records each held member's readiness token: its status, plus `deferral.stamped_at`.

**A live merge request is an owner.** A1 keeps the absorbed request's registry slot. The owner signal is therefore "a live merge request" (an `InFlightMergeRegistry` slot), the same fact task 7022 gave the sweep. It is not the lane's `ABSORBED` state, which the scheduler must not read.

- **How it is consumed.** Inject it through the existing one-method `LiveMergeRequests` protocol (`merge_deferred_recovery.py`), so the coupling stays narrow (heuristic 9). `Scheduler._phase_redispatch_stranded_blocked` then must not re-pend the member while it waits in the buffer or is being verified. Without that, a dispatch rebases a worktree that stacking has already rebased onto a co-member (§3.1, hazard 5).
- **The L2 resume path.** `Harness._cascade_unblock_member` handles a human's L2 resume. It must defer the re-pend *visibly* until the request finalizes, not skip it silently.
- **Solo merges too.** The same signal closes the exposure that solo MCP merges carry today.
- **Fallback.** Admit a `blocked` task only while a pin holds it (`recovery_pins.records_pin_blocked_recovery`). An `/unblock` escalation pins; `stranded_blocked` alone does not.

**Planning-kind deferrals.** No hazard was found in admitting them. `commit_planning` takes explicit ids, and both of its possible effects are caught:

- **A commit to `pending`** either breaks the token at the slot, or lands in A1's credit-from-`pending` arm.
- **A re-stamp of `deferred`** breaks the compare-and-set. Nothing is written and an L1 is filed.

So `commit_planning` is one of the writers that break the readiness token.

**Slot.** Step (a) becomes a readiness rule per kind, in one new module, `merge_lane/held_members.py`:

| Kind | Must hold at the slot |
|---|---|
| `WORKFLOW` | reads `merge-deferred` |
| `HELD` | still equals its token |
| `TASKLESS` | still absent |

A failure is `train_incomplete`, which strikes nobody.

**Landing.** A `HELD` member is credited `done` (kind `merged`, with a train note), but only from its captured status or from `pending`.

- **Deferred members** use the existing fused-memory `expected_stamped_at` compare-and-set, which `Scheduler` must start forwarding.
- **Blocked members** get a read-then-write, with a small window and no CAS.
- **If the delivered-checks guard withholds**, the status is left as it is and a blocking L1 is filed. The member is never reverted to `pending`.
- **The held future then resolves `done`**, carrying `train_id` and the commit. The skills route that outcome to 2v, which means no self-stamp.

**Non-landing exits.** Derail, incomplete, dissolve, exception and shutdown all go through **one exit funnel**. It restores the pre-stack tips; held members are restorable at their captured status, deferred included. It then re-buffers each held request solo under its **own request id**. The routing (one funnel for every exit) is shared with Streams 1–2. The re-buffer differs from Stream 2: a held member re-buffers its original live request, whose solo landing the submitter credits as today.

**Restart.**

- `stop()` and a merger cancel resolve held futures `blocked+SHUTDOWN` and keep their journal rows, so recovery re-queues them solo.
- A per-member write-ahead `LandedRow` carrying `held_from` covers a crash between the CAS and the credit.

**Cancel.** Absorbed request ids are refused. The server's waiter path consults the lane before cancelling, until task 7309 lands.

**Residual risks to accept.**

- **Interactive worktree.** Stacking rebases the interactive worktree in place (hazard 8). The skills must not touch `.worktrees/<id>` until the held future resolves. The alternative, stacking in lane-owned copies, costs disk and setup time.
- **Credit read-then-write.** Credit from `blocked` is a read-then-write.
- **Latency.** Interactive submitters pay train latency. An opt-in knob would address this (§7, question 3).

**Code quality.** A1 has three gains:

- It makes the implicit "every member was parked by a workflow" an explicit set of member kinds, each with stated readiness (heuristic 10, clear invariants informatively, redundantly, uniformly enforced).
- Readiness gets one home instead of the literal `merge-deferred` spread across step (a), `MemberRedriver`, `_RESTORABLE_STATUSES` and the sweep (heuristic 11, SPOT).
- Skills get a structured `train_id` instead of parsing `coalesce-<tip>-<hex>` (heuristic 12).

The costs:

- Member kind crosses six exit kinds. It must be one per-kind policy owned by `held_members.py`, not if-chains in the 22.9k-line `worker.py` (heuristic 3, carefully factored orthogonal dimensions of variability).
- Satellite helpers of `worker.py` would be cheating (heuristic 14).
- The lane now holds live requests for a train's whole lifetime (heuristics 5 and 7).

**Why A1 rather than A3 (submitter credits).** With submitter credit, a non-tip held member goes uncredited if its submitter dies. The stranded-reconcile sweep credits only a branch that is on main, and a rebased non-tip ref isn't. A3 also moves a crediting rule into skill prose that nothing enforces (heuristic 11). A1 reuses the `mark_done` shape and the existing 2v veto.

### 4.2 Why the parking designs are rejected (A2, B1a)

All four walls in §3 were re-verified in code. Parking a held task needs at least three new Table A edges, and it destroys `metadata.deferral`, which the orchestrator cannot re-mint for a `held_by_session` hold. A restore from `merge-deferred` is an unconditional overwrite. Five recovery writers would each have to learn an origin-restore rule, which fails heuristics 10 and 11. `merge-deferred` would also stop meaning one thing (heuristic 6). Task 4953 rejected this for the same reasons, and A1 gets the same trains with no status write at join.

### 4.3 Why handback (B2) and the re-drive stamp (B3f) don't work as specified

Both rely on `workflow.py::TaskWorkflow._resume_merge_retry_if_pending`. When `metadata.merge_retry_pending.branch_head` equals the worktree HEAD, the workflow jumps straight to `_merge_and_finalise`. Two facts break that for re-dispatched work:

- **Dispatch rebases the worktree first.** On a reused `.worktrees/<id>`, `git_ops.py::GitOps.create_worktree` saves WIP and **rebases onto main** (`rebase_preserving_task_commits`). Dispatch runs `_setup_worktree_and_artifacts` before the fast-path check.
- **Main moves while the member waits.** Main moves at least 1.1 times per hour; that is the landing rate alone, before direct commits. A re-pended member waits hours for dispatch. The HEAD essentially never matches. On a mismatch the stamp is cleared and the full pipeline runs, which matches the method's own docstring.

The consequences:

- **B3f** would almost never fire. It would also keep the re-pend into the deep dispatch queue, and that queue is where most of the 11.6 h goes.
- **B2's** rule of "block, never fall through" would block most handbacks. It would also reach at most 5 of the 19 eligible singles, the ones with plan history, because carriers have no plan.

Making either work needs a patch-equivalence rule that the existing docstring argues against. **The lever that does work is not to re-pend at all** (Stream 2).

### 4.4 Deep merge-ahead chains: the mechanism the draft missed (added 2026-10-29)

**How a chain works.** When the speculative slot dispatches an item, `merge_lane/worker.py::SpeculativeMergeWorker._deep_chain_placement` may build a chain. It takes the item plus a contiguous FIFO run of the queued singles behind it, merges them `--no-ff` in a scratch lane, and verifies the tip. A passing tip lands the run link by link (`_land_chain_prefix`).

Each link goes through the same terminal path as a solo request: `_journal_landed_then_advance`, then `_finalize_advanced_merge`, then `_resolve_or_drop_abandoned`. So each link gets its own merge commit and its own `done`. The only items `_link_refusal` rejects are trains and branches stacked on an unlanded base. It never checks `awaited_by_workflow`.

| | Coalesce train | Deep chain |
|---|---|---|
| **Who can join** | Only singles a workflow awaits (the 4953 rule) | Any single |
| **The member's request** | Absorbed: future resolved `superseded`, request retired, journal row removed | Stays buffered and live until it lands |
| **The member's branch** | Rebased in place by stacking | Untouched; the chain is built in a scratch lane |
| **Landing and credit** | Lands the tip only. Members are credited by callbacks that require `merge-deferred` | Every link lands normally, with its own `done` |
| **Failure** | Derail, then unwind, re-pend and one-strike marks | Silent fall-back for the links, and the chain depth halves |
| **Restart or drain** | Not journaled; a drain dissolves it | Nothing to lose |
| **Latency** | Waits at the buffer tail | Built from the head of the queue when the verify is dispatched |
| **Effect on the merger** | Runs inline, so no other merge can start until it finishes | No effect; uses the speculative slot |
| **Depth** | `merge_train_max_members` = 3 | `chain_cap` = 6 |

**Interference when both are on.**

1. They compete for the same buffered singles. The coalesce test fixture pins `chain_cap=0` for exactly this reason (`orchestrator/tests/test_merge_queue_coalesce.py::coalesce_config`). `plans/deep-merge-ahead-prd.md` never mentions coalescing or trains.
2. A coalesce train sitting in the buffer stops any chain at its position (`_link_refusal` returns `train_request`).
3. Nothing stops the coalescer absorbing a chain link while the chain's tip is verifying. The landing walk then stops at that link ("no longer buffered — stopping"), wasting the coverage of everything behind it. Whether the coalescing pass actually runs while both verify slots are busy was not confirmed.

**What trains still offer that chains don't.**

1. **Atomic landing of dependent stacked work. This is declared trains only.** `plans/orchestrator-atomic-train-merge-prd.md` was written for reify's 3997 case, where α alone reds main and only α+β+γ together is green. Chains refuse stacked branches and land link by link, so they can't carry this.
2. **Batching when the speculative slot isn't in use.** Chains only fire from slot 2; slot 1 stays the trust anchor (deep-merge PRD decision 2). The value of this coverage is unmeasured (§0).
3. **Choosing members with a risk screen.** A train picks members out of FIFO order and excludes one-strike and recently-blocked branches. A chain takes the contiguous FIFO run and has no risk screen. When a bad link fails the tip, the cost is bounded: the links fall back silently and the depth halves.
4. **One merge commit for the batch.** A chain advances main once per link, and only the cumulative tip tree was verified. For independent items this is harmless: the replay study passed 57 of 59 chains, and both failures were due to a single bad item.

**Verify cost per landed item is roughly a wash.** A merged 2–3 member train costs about 1.13× one solo verify. A chain costs the head verify, which runs anyway, plus the tip at about +8% wall time per extra item. These two figures come from different sources: the lead's runner-matched measurement for trains, and the replay study for chains.

## 5. Recommendation

> Superseded by §0 on 2026-10-29. This was the draft's recommendation, kept for the record. Stream 1 is re-scoped in §0, Stream 2 is moot while coalescing is off, and the A1 decision is withdrawn.

**Stream 1: cheap gate and hygiene fixes.** About 4–5 low-risk tasks, none touching Table A.

1. **Run the seconds-cheap whole-tree checks at coalesce admission.** This is E3's L1-lite, `inline_suppressions.py --check` per candidate. Own-defect members caused 4 of the derailed trains here (4.5 of the 10.8 h of waste). Reordering the merge verify to run the `scripts/` legs first with fail-fast (L2) could save more, including on failing *solo* verifies (25% of solo merge verifies fail). That change is **not** low-risk: it interacts with the "new failing tests not present on main" comparison, which needs a complete failing set. It needs a code read and a measurement before anyone files it.
2. **Don't start a train on a base the lane just judged red** (E3 L3). This is worth 0.6 h for certain, and up to 3.2 h with 6057's re-verifies.
3. **Make every train exit unwind** (B3a, b). That means exceptions escaping `_do_train_merge`, and `stop()`. This makes the docstring of `_unwind_coalesce_train` true (heuristic 10). **Fold this into pending task 6450**, which also covers the status-read-failure strand on the same paths.
4. **Fix the line-range stackability screen** (§6.1). Pass the full branch ref and fail loudly on rc≠0. This is correctness hygiene. It would have prevented none of the 3 observed rebase derails. A live sha-prefix collision exists (`5563` resolves to commit `5563a70ca5`). Measure merged-train counts before and after, because a working screen can lower how many trains form.
5. **The coalescer reads `InFlightMergeRegistry.has_workflow_waiter`** as well as the per-request flag (B4i). Do not journal the flag. Its value is unmeasured and probably small.

**Stream 2: a lane-owned solo re-queue for members of trains that don't land.** This is E3's L5: about 3–5 tasks, medium risk. The hubs are `worker.py`, `harness.py`, `queue_store.py` and `landed_outbox.py`.

**What it does.** On a derail or dissolution, the lane restores each workflow member's tip. It then **re-queues the member as a solo request while the task stays `merge-deferred`**, instead of re-pending it into the dispatch queue. The design needs four parts, none of which exist today:

1. **Go through the existing door.** Enqueue via `coalesce_or_enqueue_merge_request`, not an in-lane re-buffer, so the request takes a registry slot. Without the slot, `AbandonedMemberSweep` would see no `SINGLE_REQUEST` owner and re-pend the member after two 900 s passes while its solo is still queued.
2. **Add a lane-owned credit hook.** Today the single-request path writes no task status; the workflow or the MCP submitter credits. The hook can reuse the shape of `mark_member_done`: delivered-checks guard, `MergeProvenance.consume`, lane release.
3. **Make the hook durable across restarts.** Dissolutions are caused by fleet drains, so restart is the main path, not an edge case:
   - Put a lane-owned marker on `PersistedMergeRequest` and re-attach the hook during recovery.
   - Add a reconcile rule for lane-owned `merge-deferred` rows. Today `reconcile_landed_row` RC-3 prunes them without writing `done`.

   Without this, credit falls to the sweep's ancestry test. A worker-rebased solo landing leaves `task/<id>` permanently off main's ancestry, so the sweep writes `pending`, which is the very re-pend Stream 2 exists to avoid.
4. **Hand a solo failure to the next owner.** A member that fails its solo re-verify is re-pended. That re-pend must carry the failure text, through the merge-failure L0 the workflow would file or through failure metadata. Otherwise a culprit pays one extra failing verify with nothing learned: about 10 culprits per 12 days, roughly 4 h. Alternatively, re-pend a culprit directly when the Stream 1 screen has already flagged it.

**Payoff, in member latency and saved agent pipeline runs.** This stream saves time, not verify compute:

- It removes most of the 11.6 h median dissolution cost, about 180–220 member-hours across the 26 dissolved members in 12 days (partly confounded by restarts and the 10-26 scheduler pause).
- Derailed members skip the deep dispatch queue.

**Pair it with culprit-attributed one-strike** (E3 L6): mark only the members that fail their own post-derail solo. B3d alone, which skips the mark for main-red and for conflicts with main, would spare only 1 of the 4 innocent tasks.

**Decision: A1 admission.** About 5–6 tasks, medium-high risk. Streams 1 and 2 supply only the exit routing. Everything else is A1-only:

- admission and the readiness snapshot;
- the `ABSORBED` state and the cancel guard;
- held-member credit with the CAS forwarding;
- the RC-3 change for held rows (a `LandedRow` carrying `held_from` does nothing while RC-3 prunes blocked and deferred rows);
- the live-merge-request owner signal;
- skills and docs.

How to decide:

- **Measure reach first, over more than 12 days.** For each eligible single, record its status, deferral kind and worktree path at submit. The population is bursty and mostly hand carries.
- **Breakeven.** Expected value = eligible member-equivalents × net saving per member, set against about 5–6 tasks. Today's upper bound is 32 × 4.7–15.9 min ≈ 2.5–8.5 h per 12 days. Stream 1 raises the per-member figure.
- **Make the measurement repeatable.** Commit the reconstruction under `scripts/` and name a dated measurement task. Per standing practice, a hold needs both a condition and a trigger.
- **Consider the cheaper paths.** If hand carries dominate the population, the cheaper ways in are **A1 as opt-in** (`merge_request(allow_train=True)`) or **B5**, a submitter-declared train of one session's own carriers.

**Why this order.** Stream 1 pays for itself in verify compute and also builds the single exit funnel (item 3), which Stream 2 and A1 both route through. Stream 2 pays in latency, provided its failure arm hands failures on. A1 is a separate bet on a bursty, mostly hand-carry population. The 4953 invariant ("only a workflow parks a task") holds throughout, because neither Stream 2 nor A1 parks anything.

**Existing tasks to map onto, rather than duplicate:**

| Task | Priority | Scope | Relation |
|---|---|---|---|
| 6450 | high | Re-drive strand on a status-read failure | Stream 1, item 3 |
| 6566 | low | An already-merged tip credits no member | Stream 2 crediting |
| 6755 | low | Train-verify parity | — |
| 6106 | medium | Laptop memory limit behind the 3-member local pin | — |
| 7309 | medium | Train cancel | A1's cancel guard |

## 6. Adjacent findings (independent of the MCP question)

1. **The line-range stackability screen is broken.**
   - **The bug.** `worker.py::SpeculativeMergeWorker._stack_coalesce_train` and the declared-train former in `workflow.py` call `git_ops.py::GitOps.get_changed_line_ranges` with a bare task id as the git ref. The former is enabled in this repo's yaml but formed no trains in the window.
   - **Ambiguous ids.** `git diff main...5925` fails (`short object ID 5925 is ambiguous`). The error is ignored, so the ranges come back `{}` and every pair counts as "stackable".
   - **Colliding ids.** `5563` resolves to the unrelated commit `5563a70ca5`. Task 5563 was a member of train 7071.
2. **Journal recovery drops `awaited_by_workflow`** (`merge_lane/queue_store.py::reconstruct_merge_request`). It was behind 5 of the 13 post-4953 `no_awaiting_workflow` exclusions: restart artifacts finalized "Merge worker shutting down".
   - The exclusion was **correct when it fired**, because a recovered request has no live workflow.
   - B4(i) re-admits such a request only once a re-dispatched workflow re-attaches.
3. **`stop()`, a mid-verify cancel, and exceptions escaping `_do_train_merge` skip `_unwind_coalesce_train`.**
4. **WIP halts raised by a coalesce train have no owner.** Only a declared-train tip workflow files the halt-owning L1 (`workflow.py::TaskWorkflow._escalate_train_halt`).
5. **Train landings skip `_note_merge_landing` and `_on_merge_landed`.**
6. **The derail tag "Train union verify failed (re-verifying members as singles)" over-reports interaction.** 0 of the 11 derails it labelled after 4953 were interactions.
7. **Queue surgery in `_stack_coalesce_train` drops buffer items by `task_id`, not `request_id`.**

## 7. Questions for Leo

> Answered on 2026-10-29 (§0). A1 is withdrawn, so questions 2–6 lapse, and question 1 is settled by Leo's ruling. Question 7 lapses with Stream 2. The questions still open are listed in §0.

1. **Sequencing.** Do Streams 1 and 2 now, and take A1 as a separate, measured decision? Or build A1 now as well?
2. **Which way in for MCP singles?** The measured population is mostly hand-carry campaigns. The options:
   - **A1 general.** Every held member is admitted.
   - **A1 opt-in.** Admission requires `allow_train=True`.
   - **B5.** A session declares a train of its own carriers. No seat has evaluated this; it would need a design pass.
3. **If A1, crediting.** Should the lane credit `done` on a held member's train landing (recommended), or the submitter (A3)?
4. **If A1, withheld credit.** When delivered-checks withholds credit for a held member, is "leave the hold in place, file a blocking L1" acceptable instead of the workflow member's revert to `pending`?
5. **Live merge request as an owner.** Should a live merge request (a registry slot) count as an owner for the stranded-blocked sweep, with the L2 `resume` deferred *visibly* until the request finalizes? This would cover solo MCP merges today, and A1 later. Or should A1 admit `blocked` tasks only while a pin holds them?
6. **If A1, cancellation.** Refuse cancel of absorbed requests until task 7309 lands, or make 7309 a dependency?
7. **Stream 2's failure arm.** Should a member that fails its lane-owned solo be handed on with a merge-failure L0 (the workflow's own shape)? Or should a culprit already flagged by the Stream 1 screen be re-pended directly?

## Appendix A — figure rules

- **MCP vs workflow** (E1).
  - **Rule.** A `merge_queued` event with no open `phase_enter(merge)` for the same (run, task) counts as MCP. A journal-recovered request takes the class of its origin event.
  - **Recovered.** A request is recovered if the same `request_id` was re-queued after a shutdown or abandoned finalize, or if a crash heuristic matches.
  - **Accuracy.** Recall is 21/21 on the known MCP set. Precision is unmeasured. The label also covers other non-workflow submitters.
- **Landings.** `merge_finalized` rows with state `done`, plus every member of a `train_merged`. Windowed by landing timestamp (UTC).
- **Verify host-hours.** The sum of `merge_verify.data.duration_ms`, which is runner dispatch wall time; parallel hosts add up.
  - A row belongs to a request when it has the same task and run and `queued.id < row.id <= finalized.id`.
  - Train rows belong to the tip task, between `train_started` and the train's terminal event.
- **Eligible MCP singles** (critic's run-scoped re-run of E1 Q3b, `lead/critic_e1_runscoped.py`).
  - **Waiting interval.** From `merge_queued` to the earlier of the next `merge_dequeued` or the `merge_finalized`.
  - **Own-branch history.** "Clean" means the latest `merge_finalized` for the branch **in the same run** before queueing was not `blocked` or `error`, matching `latest_merge_finalized(cross_run=False)`.
  - **Partner.** A non-recovered workflow request with the same clean run-scoped history, and not under a 7-day one-strike.
  - **Window.** Up to 2026-10-26T16:28:30Z.
  - **Bound.** This is an upper bound: stackability, the 3-member cap, debounce, lane and pass timing are all ignored.
- **Train economics** (lead, `lead/econ.py`).
  - **Saved.** Summed over merged trains: member count × the runner-matched solo passing-verify median (or mean) in the same before/after window, minus the train's verify minutes. A train's runner counts as `local` if any of its verify rows ran there.
  - **Waste.** Verify minutes of derailed trains. Members' re-verifies after a derail are excluded, since they were owed anyway.
  - **Split.** At 2026-10-13T21:48Z, the 4953 merge commit. The deploy was 10-14T03:30Z, and no train formed in between.
  - **End.** The data ends at 2026-10-26T16:28:30Z (`END` in the script). A re-run reproduces §2.3 exactly.
  - **Equivalents.** Train minutes divided by that runner's post-4953 solo median (local 40.2 min, laptop 14.2 min).
- **Admission value.**
  - **Per member.** 15.9 min = 19.0 h ÷ 72 merged-train members. 4.7 min = 8.2 h net ÷ the 104 members of all started trains.
  - **Member-equivalents.** 6 × 1 for the singles with an exclusion event, plus 13 × 2 for those without, where the partner also went solo.
- **First-verify pass rate** (lead, `lead/defect_rate.py`). Over non-recovered requests with at least one attributed verify, the share whose first attributed `merge_verify` passed.
- **Union-derail classes** (E3).
  - **Own defect.** The cited check fails on base + that member alone, reproduced by merging into a scratch `git clone --shared`. Alternatively, the member's own solo verify failed the same test.
  - **Flake.** The same test failed at the gates of many unrelated tasks, or it passed on rerun and in isolation.
- **Dissolution.** Detected from the journald line `admission halted before its verify started; dissolving it`. Its cost to members is the time to their first later `done` landing.

## Appendix B — how this memo was produced

A `/team` run on 2026-10-26/27, led by an opus session. Seat outputs (`w1_*.md`, `w2_*.md`) and the lead's notes are committed beside this memo, in `plans/mcp-coalesce-trains-2026-10-26/`. The `w1_E2.md` net-economics figure and the `w2_DB.md` B2/B3f premise are superseded, as recorded in the table below. The analysis scripts (`lead/*.py`) are not committed; they are in `~/.claude/fleet/sessions/team-df-319052/scratch/`.

| Seat | Model / effort | Brief | Outcome |
|---|---|---|---|
| R1 lane lifecycle | sonnet / high | Map coalesce-train lifecycle, exits, restart and custody | Full map; found the bare-id line-range defect |
| R2 status side | sonnet / high | Map Table A, sweeps, holds, deferral and callbacks | Full map; established the four walls against parking |
| R3 submitter side | sonnet / medium | Inventory MCP submitters and the server contract | Full map; the escalation server runs in-process |
| E1 premise | sonnet / high | MCP volume, counterfactual trains, verify cost | Figures; its own-branch test was cross-run, corrected by the critic |
| E2 derail taxonomy | sonnet / high | Every train's outcome and cause | Taxonomy; its net economics figure was corrected by the lead (runner mix) |
| D-A held member | opus / high | Directions 1 and 3 | A1 viable, A2 rejected |
| D-B delegated owner | opus / high | Directions 2 and 4, plus do-less | B3 and B4 recommended; its B2/B3f premise ("no rebase on reuse") was refuted by the critic |
| E3 derail root cause | sonnet / high | Post-4953 union-verify, rebase and dissolution causes | 8 of 11 own-defect; levers L1–L6 (first attempt died; re-run) |
| Critic | opus / xhigh | Adversarial review of the integrated draft | 16 findings: 1 blocking (B3f), 2 answer-changing (run-scoped eligibility and partner doubling; A1 reach), all folded in |
| Critic re-check | opus / xhigh (same seat, resumed) | Fold-in fidelity, plus the lead's new Stream 2 and A1-sequencing text | 7 findings: 2 answer-changing (Stream 2 had no credit path and no registry slot; the claimed overlap with A1 was overstated), all folded in |

**Cost.**
- Wave 1 (R1–R3, E1, E2): 1.35M tokens, 73 min.
- Wave 2 (D-A, D-B, E3): 0.90M tokens, 4.3 h wall clock, including E3's re-run.
- Critic, both passes: 0.38M tokens cumulative; 59 min, then 18 min.

**The lead's own checks.**
- Re-derived the train economics on a runner-matched basis.
- Measured MCP vs workflow first-verify pass rates.
- Confirmed the bare-id `git diff` failure and the live `5563` collision.
- Re-read `_resume_merge_retry_if_pending`, `InFlightMergeRegistry.attach`'s mirroring of `superseded`, the `expected_stamped_at` CAS, `create_worktree`'s rebase on reuse, and `latest_merge_finalized`'s run-scoped default.
- Re-ran the critic's run-scoped eligibility reconstruction: 19 eligible singles, 6 with an exclusion event and 13 without.

**Amendment, 2026-10-29.** §0 and §4.4 were written by the lead alone, with no seats, after Leo asked how this feature relates to deep speculative verifies. The lead read four things on main `6d2f46a04b`:
- the chain code in `worker.py`: `_deep_chain_placement`, `_link_refusal`, `chain_snapshot` and `_land_chain_prefix`;
- `SpeculationController.can_coalesce`;
- the deep-merge and atomic-train PRDs;
- the coalesce test fixture.

The lead also queried `runs.db` for chain verifies, chain landings and speculative verify counts. These additions were not reviewed by a critic.

**Model routing.** Fable was judged appropriate for the design and critic seats. They ran on opus because the brief did not authorise fable.
