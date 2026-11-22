# Seam scout for gate 10069 (merge-pool batches PRD)

All sources were read on 2026-11-19, read-only. "Mark" means the un-chained mark. The repo root is /home/leo/src/dark-factory.

## 1. The un-chained mark (10617, 10723, 8528 as amended, policy PRD D6/D7)

**(i) Contract**
- After a red chain tip, the owner and every link are marked "verify alone next". The tip-fail exit of `orchestrator/src/orchestrator/merge_lane/inflight_verify.py` (since the 5040 split) calls `ChainPlanning.mark_next_dispatch_unchained` (`merge_lane/dispatch_ports.py`). That writes `worker.py::SpeculativeMergeWorker._next_dispatch_unchained`.
- A marked task is never a chain owner (guard 2b of `worker.py::SpeculativeMergeWorker._deep_chain_placement`) and never a link (`::chain_snapshot`).
- The mark is removed only in `worker.py::SpeculativeMergeWorker._retire_item`, when the request is landed, blocked or dropped. A voided result or a cancelled verify leaves it in place (task 10723).
- `plans/merge-lane-policy-prd.md` D6 (amended 2026-11-18): the mark "owns red-chain isolation; the suspect bisect is retired". "Requests that are not marked keep chaining at the depth D7 gives." A passer lands and "the red one is blocked with its findings" after its own solo verify.
- 8528's AMENDED 2026-11-18 paragraph: "No suspect list, no requeue at the front, no queue fronting, no half-chain bisect rounds". The search was removed cleanly, not switched off. Its Quality line: "heuristic 11 SPOT (one mechanism isolates a red chain's members: the mark)".
- D7: `merge_lane/chain_policy.py::ChainPolicyState / choose_depth / on_tip_result`. p̂ is one lane-wide estimate from the last 30 tip verdicts of rounds the policy sized. Mark-forced solo verifies are not counted.
- D7 was amended 2026-11-19 by task 10920 (commit `a9e4125f99`). The table has one row per depth from 12 down to 3, derived from the cost 1/k + (1 − p^k), else depth 1. The live p̂ was 0.89–0.90, which gives depth 4.

Why the bisect was dropped (`plans/8528-unchained-mark-study-2026-11-18/REPORT.md`):
1. A reproduced defect: the branch released suspects when a verify returned, not when its verdict was applied.
2. In about 150k simulated runs the search never beat the mark, except one corner of about 4% (red share 0.02, 8 arrivals/h, all effects on) after repair and loosening (Q2).
3. Its "nobody chains during a search" rule was the main cost. Good requests waited 7.2–9.2 h against 5.6–6.8 h.
4. Measured red share: 0.24 before the mark, 0.08 with it (n=12, range 0.05–0.13).
5. The study named "one held set owned by the policy module… released only on an applied verdict" as the right shape if a search ever returns. Restarts forget marks: a culprit was re-chained 6–12 times per 72 h with all effects on.

**(ii) Status**
- 10617, 10723, 8528 and 10920 are done.
- 10624 (marks links before the reservation ends; coalescing honours the mark) is pending.
- 10759, 10761 and 10762 (chain-round and isolation records) are pending, re-scoped to "isolations". 10760 is cancelled.

**(iii) Question for the batch design, by use case**
- **(a) Keep a risky request out of chains before any red chain.** This is not isolation, so it does not touch the mark's ownership. It does contradict D6's "Requests that are not marked keep chaining at the depth D7 gives" and D7's single lane-wide p̂. Allowed only as an explicit amendment of D6/D7. 10069 item 1 already implies that amendment.
- **(b) Narrow a failed batch with sub-batches.** This contradicts the 2026-11-18 ruling as written: "no half-chain bisect rounds", "neither a chain owner nor a link" until retired. It would also bring back the second isolation mechanism that the SPOT argument removed. The study's measured evidence is against it.
- **(c) Convict by elimination without a solo verify.** This contradicts D6's wording: each marked member "verifies alone until its request is retired", and the red one is blocked on its own verdict. Formally, "blocked" is a retirement path, so the mark would release. But under the mark every member verifies alone in parallel, so elimination saves at most the culprit's own concurrent solo verify. Conviction without a solo verify needs D6 reopened.
- 10069's item 3 ("further batches clear it") and item 4 (Leo's 2026-11-13 conviction ruling) predate the 2026-11-18 ruling and were not amended.

**(iv) Ownership:** the mark (D6) stays the only owner of what happens after a red batch. The batch PRD owns which requests go into a batch and how deep it is, which replaces D7's lane-wide p̂. Anything that touches (b) or (c) must be raised with Leo as a D6 amendment.

## 2. Task 10721: a passed head that must re-verify is re-verified alone

**(i) Contract**
- When a head passed but main moved, the gate re-verify (`worker.py::SpeculativeMergeWorker._dispatch_rebased_head` → `merge_lane/rebased_head.py::rebased_head_item`) "verifies that head's own rebased tree, un-chained".
- Measured: 4 of 5 heads re-verified inside a 6-chain lost their landing.
- Left open by 10721: whether waiting requests may stack speculatively behind the re-verifying head.
- Related but distinct: 8329 (pending), a chain built on a head that the gate then rebases.

**(ii) Status:** pending, high, first in the pin order (10572 brief). Its only dependency, 10723, is done.

**(iii) Question:** a gate re-verify is a forced singleton. Is it a batch-chooser candidate at all, and does its verdict count as evidence about the request? D7 already excludes "a dispatch that may not chain" from p̂. Does the batch PRD treat a rebased head as a single request with no new evidence?

**(iv) Ownership:** 10721 owns "a gate re-verify is solo". The batch PRD excludes rebased heads from membership and states how their verdicts feed the probabilities. 8329 keeps its own case.

## 3. Merge pre-verify screen (plans/merge-pre-verify-screen-prd.md; 7919, 7922, 7921)

**(i) Contract**
- The "fast seven" checks live in `scripts/merge_screen.yaml` and run at `worker.py::classify_and_merge` (merger loop), `_do_train_merge`, and the chain tip in `_deep_chain_placement`. Re-merges (`_remerge`) are not re-screened (decision 1).
- Attribution is base-first: red on the item only means a bounce; red on both means proceed (decision 6, C2). Errors fail open (C3).
- A bounce is recorded as `MergeOutcome('blocked', failure_category='merge_screen', failure_cause_hint=<check ids>)` with `OutcomeKind.screen_failed`, and a `merge_attempt` row with outcome `screen_failed`. It has **no `merge_verify` row** (C4).
- The bounce "never reaches `_run_post_merge_verify`": no probe, no flake ledger, no halving/pass estimate. It does feed the workflow thrash guard (decision 7).
- Per-check data is `merge_lane/screen.py::ScreenCheck{id,status,findings,duration_s}` inside `ScreenVerdict`. A `merge_screen` event is emitted, with status `error` on failure.
- Chain tip (7922 = δ, C7): screen the tip and, on a finding, run a bounded prefix search (at most ⌈log₂ links⌉+1 static screens). Truncate before the first failing prefix with `ChainTruncationReason.screen_failed`, which is benign and not in `CHAIN_FAULT_REASONS`. The truncated link gets no outcome. While a main-red verdict is armed, no chain is built.
- Measured value: the screen catches 58/147 (39.5%) of merge-verify failures (§2.1).

**(ii) Status**
- 7918 is done. 7919 is pending (fifth in the pin order).
- 7921 is pending and parked behind gate 10577, which fires 7 days after 8532 (about 2026-12-03).
- 7922 is pending, depending on 7921, 7720 (done) and 7259 (pending).
- 7920 and 7924 are pending. The PRD header still says "draft… Not decomposed", which is stale since its tasks exist.

**(iii) Questions**
- A failure-probability reader that reads `merge_verify.chain_request_ids` (task 9908) will never see screen bounces, because no `merge_verify` row exists for them. To use them it must join `merge_attempt outcome=screen_failed` and `merge_screen` events per request. A screen pass removes roughly 40% of the failure classes, which matters for the prior.
- Does a passing `merge_screen` event carry per-request, per-check status? The PRD only guarantees the `error` status, plus the F-pass line in γ1's live confirmation. See "Could not determine".
- 7922 overlaps batch membership: it is a post-hoc truncation of the chosen batch. It must run after the chooser and must not be undone by it.
- 7922's text still says "never halves the chain depth" and "halving state unchanged". Halving is retired by D6/D7, so this should read "the pass estimate sees only tip verdicts".

**(iv) Ownership:** the screen PRD owns screen bounces and the static truncation of a built batch. The batch PRD chooses the batch before the screen runs and reads screen outcomes as per-request evidence. It never treats a screen truncation as a batch failure.

## 4. Task 5411: docs-only / comments-only narrow gate

**(i) Contract**
- `docs_gate` (a positive path allowlist) and `comments_gate` (AST unchanged after stripping comments) are new resolutions beside `full_gate`, set at merge role in `verify.py`. Any classifier miss gives `full_gate`.
- The resolution and the command set are emitted on the event.
- "Docs/comments items never join a train." Modules touched: "merge_queue.py (train eligibility exclusion)".
- Amended 2026-11-19: no shadow; on in DF when it lands. Reify turns it on after 7423.

**(ii) Status:** pending, depending on 5296 (pending). Dependents: 6661, 8258, 9131. The 10572 draft (Q6) proposes moving it up the pin order.

**(iii) Questions**
- **Today:** docs-only requests take the full gate and are ordinary chain candidates. `worker.py::_link_refusal` refuses only `GroupMergeRequest` and stacked-on-unlanded branches.
- **After 5411:** excluded from trains. Chains are not mentioned anywhere in 5411.
- Inference, not verified: the classifier judges the verify's whole diff, so a chain mixing docs and code misses the classifier and pays `full_gate`. A docs request batched with code then loses its cheap gate.
- The batch PRD must decide whether docs/comments-class requests are batch candidates (singleton only, or batched only with each other).
- 5411's "merge_queue.py" target does not exist under `orchestrator/` (only `dashboard/.../merge_queue.py`), so the train-exclusion site is not pinned.

**(iv) Ownership:** 5411 owns classification and the narrow gate. The batch PRD owns whether a docs/comments request may be a batch member, keyed on 5411's resolution, never re-classifying.

## 5. Trains vs chains

**(i) Contract**
- **Train:** `merge_lane/types.py::GroupMergeRequest`. An atomic, linear-stacked merge of `member_task_ids`, listed root to tip. It is landed by `_do_train_merge` with member callbacks (`status_check`, `mark_member_done`).
- Membership comes from task metadata `train` (declared), or from the line-range former or coalescer in `merge_lane/train_selection.py::select_train_members` (greedy, mutually line-stackable). Both of those are off in DF: `merge_train_former_enabled: false` and `merge_train_coalesce_enabled: false` in `dark-factory-orchestrator.yaml`.
- **Chain:** a speculative verify of base plus k independent queued requests in pool order (`build_chain` over `chain_snapshot`). One green tip lands the in-order prefix; a red tip marks every member (D6).
- **Train members never sit inside a chain:**
  - a train dispatches with `chain_ctx=None` ("trains never auto-chain", PRD D9);
  - a train in the snapshot is refused as a link (`_link_refusal` → `'train_request'`, a benign truncation that caps the chain before it).
- 9144 adds `TrainKind` (declared | coalesced).

**(ii) Status:** declared trains are live; former and coalesce are off (7739; policy PRD D14). 7860 (the human gate on train formation) is pending.

**(iii) Question:** treat a declared train as one indivisible unit? Since it can never be a batch member, it is a forced singleton and a batch-boundary cap.

**(iv) Ownership:** the atomic-train PRD and 6755 own trains. The batch PRD treats a `GroupMergeRequest` as non-batchable and states it.

## 6. Gate 10572: verify capacity, and submissions vs landings

**(i) Contract (task text):** decide how verify slots are shared between task verifies and merge verifies (cap, share or priority); what 5139 and 5203 must change; which plans drop the revoked rule "merge verify is never throttled"; and how the submission rate is balanced against the landing rate. The brief adds two items: routing docs commits off direct-to-main, and `max_concurrent_tasks` (36).

**(ii) Status:** pending, depending on 8532. A sitting is drafted but untracked and unruled: `/home/leo/src/dark-factory/plans/gate-10572-verify-balance-2026-11-19/SITTING.md` and `EVIDENCE.md`. Nothing in `git log -- plans/` or `git ls-files plans` matches it. Its draft recommendations, not yet ruled by Leo:
- Q1: no cap or share on merge verifies, and **hand "hold a merge verify until the pool is fuller" to 10069 as an input**.
- Q4: no submission gate.
- Q5: docs through the high lane once 5411 and 6661 are live.

**(iii) The batch PRD must NOT decide:**
- the host split or merge-verify cap/share/priority against task verifies;
- `max_concurrent_tasks`;
- any gate on submissions or task starts by queue depth;
- docs-commit routing and its lane;
- the 5139/5203 amendments.

Possibly in scope, only if Leo accepts Q1-C: waiting to start a single-request verify so that a batch can form.

**(iv) Ownership:** 10572 owns capacity and the inflow rate. The batch PRD owns composition given a dispatch slot, plus "wait for a fuller pool" only if 10572 rules it over.

## 7. Part 1 (plans/priority-at-later-gates-prd.md §6.4; 9933, 9935, 9936, 9937)

**(i) Contract**
- `order_key(p, *, now, waited_secs, promote_after_secs, fallback_anchor) -> (rank, pin_order or 0, -age)`, pure and never raising (§6.1). The merge pool and admission "build no key of their own" (§3, D-α1 text).
- `MergeIntake.pool_order()` in a new `merge_lane/pool_order.py`:
  - under `arrival`: buffer order, today's path untouched (O3);
  - under `priority`: high lane by `order_key`, then "task 8528's suspect list in its order", then the rest by `order_key`; the same-files rule is dropped.
- `pool_order()` returns every buffered request. The pick filters reserved and halted requests; the 7921 held skip goes in that filter.
- Every order reader uses `pool_order()`, **including `chain_snapshot`** (the batch candidate order).
- Config: `merge_pool` in its own file `orchestrator/merge_pool_config.py`, with `order`, `promote_after_secs` and `bounce_cap`, all hot-reloadable.
- O1: main advances in dispatch order. O2: nothing merged or verified is reordered, re-based or returned by a later arrival (Leo's R1).
- §5 row: "Part 2 | produces for | `pool_order()`, `order_key`, the `merge_pool` config file | this PRD".

**(ii) Status**
- 9933 (α1) is pending and pinned.
- 9935 (γ) is pending, depending on 9933 and 8528.
- 9936 (ι1) and 9937 (ι2, the flip gate, which also waits on 8532) are pending.
- 9908 (recording, incl. `merge_verify.chain_request_ids`/`chain_task_ids` and `merge_dequeued.waiting`) is done.
- 10069 depends on 9935, 8532 and 8528.

**(iii) Questions**
- Part 2 may read `pool_order()` (as the walk order for membership) and `order_key`, and may add keys to the `merge_pool` config file.
- It must not:
  - add a second ordering key;
  - change the `arrival` path;
  - displace or re-base anything in flight (O2);
  - bypass `pool_order()` for any order reader.
- Open seam: §6.4, boundary row 5, open question 2 and 9935's text all slot "8528's suspect list" second. That list does not exist after the 2026-11-18 ruling. Someone must say whether marked requests are fronted in `pool_order()`.

**(iv) Ownership:** part 1 owns order and the config file. Part 2 owns membership along that order. Fronting of marked or implicated requests needs one named owner: either a 9935 amendment or the batch PRD, not both.

## 8. Task 8532 (7-day readout) and task 8932 (design milestone)

**(i) Contract**
- **8532:** a delayed milestone whose dependencies were satisfied at 2026-11-19T04:16Z, so it fires about 2026-11-26T04:16Z. It commits `plans/merge-lane-policy-prd.measurements/<date>-canary.md` containing:
  - the D12 predicates (gates/landing, voids/landing, P(gate|gate), laptop share, violations, chain-attributed red main, median queue-to-land < 1 h, local merge-verify h/week < 60);
  - the `chain_items` distribution and `chain_depth_reason` rows;
  - the 3190 figures: deep-chain verify count, failure rate, items landed per verify, p90 deep-chain duration ≤ 10800 s;
  - verifies started on a red main;
  - per-item chain wall at depth ≥ 8 if such rows exist.
  
  If any predicate is false it escalates at L2 recommending `chain_cap: 0`.
- **8932:** a design milestone after 8532.
  - Q1: an idle-host prefix hedge, amended 2026-11-18 to be judged against the mark's parallel solo verifies.
  - Q2: a dense p̂→depth table, swept in 0.01 steps over every integer k.
  - Deliverable: `plans/merge-lane-policy-prd/idle-hedge-and-depth-table-<date>.md` plus a leaf if positive.

**(ii) Status:** both pending. 8932 was last updated 2026-11-18 21:51Z, before 10920 landed.

**(iii) Questions**
- 8532 is the batch PRD's calibration input; 10069 waits on it.
- 8932 Q2 is largely superseded by 10920 (a formula-derived row per depth 3–12, with a test deriving it). It would be fully superseded if per-request probabilities replace the lane-wide p̂.
- 8932 Q1 is a batch-composition decision ("verify this prefix on an idle host") and overlaps the batch chooser directly.

**(iv) Ownership:** 8532 owns the measurements. The batch PRD should absorb 8932, or explicitly re-scope it (Q2 closed by 10920, Q1 folded into the chooser) and record that on 8932.

## 9. Tasks 8933, 9144, 7921

- **8933** (in progress): after a red deep tip, the requeue (now `inflight_verify.py` → `custody.requeue_from_verify`) does not add the requeued items' speculative merge commits to `_dead_base_commits`. A successor can then verify on an unlanded base. Seam: any batch-failure handling must go through the same requeue path and dead-base recording; 8933 owns that.
- **9144** (pending): `MergeOutcome.kind`, `GroupMergeRequest.train_kind`, consumers routing on kind, and frozen reason strings. Any new outcome (for example a conviction or a suspect deadline) must be an `OutcomeKind` member: in `shared/src/shared/merge_outcomes.py` once 9766 lands, otherwise in `merge_lane/types.py`, plus a dashboard colour family. 9144 and 9766 own the vocabulary.
- **7921** (pending, parked behind 10577): main-red verdict and F-hold. Held requests are filtered after `pool_order()`, are never chain links, and no chain is built while a verdict is armed (7922). A held request is neither a batch candidate nor a suspect. 7921 owns "held"; the batch PRD must exclude held requests and not count holds as evidence.

## Conflicts between sources

1. **10069 vs the 2026-11-18 ruling.** 10069's title ("now that the bisect task… landed") and text (8528 = "suspect-list bisect"; suspects cleared by "further batches"; conviction by elimination) were not amended after 8528 landed without the bisect. D6 as amended forbids sub-batching marked members and implies each member's own solo verdict.
2. **Priority PRD §6.4, boundary row 5, OQ2 and task 9935** still put "task 8528's suspect list" second in `pool_order()`. No such list exists. The policy PRD §5 row 7 / D6 and 8528's amendment were updated; the priority PRD and 9935 were not.
3. **7922** (and screen PRD decision 7, C7) still say "halving state unchanged" / "never halves the chain depth". D6/D7 retired halving in favour of the pass-estimate window.
4. **8932** still describes D7's table as `[[0.93,8],[0.85,4],[0.75,2]]` and asks for a dense table. 10920 (done, 2026-11-19) already replaced it with a formula-derived table for depths 3–12.
5. **The 8528 study report vs the PRD:** the report recommended a top row of 6, Leo ruled 8, and 10920 then moved it to 12.
6. **Screen PRD header** says "draft… Not decomposed", but tasks 7918–7924 exist and two are done.
7. **5411** names `merge_queue.py` as the train-exclusion site; no such file exists under `orchestrator/`.

## Could not determine

- Whether a **passing** screen emits a per-request `merge_screen` event with per-check statuses. The PRD guarantees only `status error`, plus "F passing" in γ1's live check. 7919's event payload is not specified field by field in the text I read.
- Whether 5411's narrow gate applies to a chain whose members are all docs-class, or how a mixed chain resolves. 5411 is silent on chains; the "mixed chain → full_gate" point is my inference from "any classifier miss → full_gate".
- The 10572 rulings: the sitting is an uncommitted draft with an empty answer sheet. Whether "hold a verify for a fuller pool" moves to 10069 is unknown.
- Whether `merge_verify_cancelled` carries `chain_request_ids`. 9908 adds them there only "if that emit site already has the batch in hand". I did not check the emit site.
- I did not read the study's volatile evidence under `/tmp/8528-study/`, or the critic files of `plans/priority-at-later-gates-prd.evidence/`. `plans/merge-pool-batches-design-notes.md` (the 2026-11-13 part-2 draft, which says it is not decomposable) was read only to its first 30 lines.
