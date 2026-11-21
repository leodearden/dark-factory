# Critic pass 3: DRAFT 3 of `/tmp/gate-10069/draft/merge-pool-batches-prd.md`

Not a clean pass: 3 A-defects, all narrow. Read-only against main `eafdb1ec73`. Scratch is `/tmp/gate-10069/critic3/switch.py`. Code paths are relative to `/home/leo/src/dark-factory/orchestrator/src/orchestrator/`.

## A. Answer-changing

**A1. §6.1 "`bounce_conflicting_suffix_items` records the frozen tip as the bounced branch's stack base" undoes the bounce.**
- Recording is feasible: `StackBaseLedger.record` takes a sha.
- The identity cut is then right. `foreign_commit_cut` ends on the frozen tip for a tip made of lane merges, before and after it lands.
- But the same ledger is read by `worker.py::classify_and_merge` step 1b (`_unstack_before_merge` → `GitOps.unstack_from_unlanded_base`). While the recorded base is not on main, it replays the branch's own delta back onto main.
- The frozen tip is normally still unlanded at the next pick. The bounced request is therefore un-bounced before its merge, and then conflicts with the speculative base or is blocked by the unstack.
- `worker.py::_link_refusal` reads the ledger too and would refuse the request as a batch member.
- *Requirement:* a record that identity reads and the two unstack readers do not act on while the base is a live frontier commit; or state that a bounce loses identity. Add a row: bounced request, frozen tip unlanded, picked — it merges on the frozen tip as today.

**A2. §6.6 Rows, the truth definition.** "An observation… on a root proved before it, in which the key is the only one not cleared: `RED` is `defective`, `PASS` is `sound`."
- A pass of any tree holding the key shows it sound. The sentence also demands sole-uncleared and a proved root for a pass.
- A shadow or returning key is verified alone but stacked. When its tree passes before the entry ahead has returned, it yields no truth. A pass on an unproved root yields none either.
- A red that becomes sole later still counts, if evaluated after the fact (the text does not say when).
- Sound truths are therefore undercounted against defective ones. Both the on-criterion and the 2-of-20 trigger read a precision that is too high.
- *Requirement:* any pass recorded after the conviction row is `sound`; a red is `defective` once the key is its only uncleared one on a root proved before that red; say when that is evaluated.

**A3. L16 is not implemented for a hand-submitted request that crosses a restart.**
- §6.6 Who (d) reads "awaited by a workflow or owned by the recovered-merge owner".
- `RecoveredMergeOwner.adopt` takes every journal-recovered request.
- `merge_lane/queue_store.py::PersistedMergeRequest` records no submitter. `awaited_by_workflow` is not journaled and is also false for MCP submissions.
- After a restart a hand-submitted request is therefore indistinguishable from a workflow's and is convictable.
- §5 says "this PRD adds no journal field"; δ does not list `queue_store.py`.
- *Requirement:* carry the submitter kind on the request and through the journal, or rule the recovered case explicitly. Row 22 needs the combined case.

## B. Minor

1. **Rebuild and the live set.**
   - `rebuild(live: Mapping[str, ContentKey])` runs before the first pick, but §6.2 computes recovered ids in the background. `live` can only be request ids. Say which key a recovered request has until then, and whether it can be picked or deferred.
   - The newest-checkpoint lookup has no raising read: `fetch_events_after` cannot find it and `fetch_recent_events_by_type` returns `[]` on error. A tail that fills `limit` should fail the rebuild.
   - The checkpoint's root facts omit "a red was recorded on this unproved root", which rule (b) needs after a restart.
2. **Hold limit input.** It is defined on "a passed verify's duration in the ledger's rows"; `Observation` has no duration or dispatch time.
3. **Merger wake-ups.** `SpeculativeMergeWorker._acquire_next_request` wakes only on an arrival or `_resume_signal`. Name the three wake-ups: a return, a host freeing, limit expiry. No row covers the limit, so L14's bound can be inert. The precedent is `ChainReservations(on_release=self._resume_signal.set)`.
4. **Conviction and in-flight.** "Who" excludes reserved, picked and parked, but not a key in an unreturned tree. A link whose reservation expired qualifies (pass-1 B8). Use §6.5's in-flight definition.
5. **Row 14 is in β2a.** Sub-batching red-set members needs the mark gone, which is β2b.
6. **File lists.**
   - β2a is about 24 files. It needs `inflight_verify.py` and `dispatch_ports.py` (`note_tip_result`), `test_merge_queue_deep_reservation.py` (its own `DepthRow` pin) and about ten other test files.
   - β1b needs `types.py` to carry `predicted_pass` and `plan` from dispatch to verdict.
   - β1a is at or over the band.
7. **B3 pins size only until the first red.**
   - `defect_rate` is smoothed by one prior tree: 0 becomes 0.109 after one red of 6, then 0.167.
   - At rate 0 a red implicates nobody.
   - Fresh single-round scenes work, membership included: all candidates join in pool order, as today.
   - Scenes asserting sizes across reds need a prior-weight leaf or re-derivation. Examples: `[6, None, 6, 6]` and `[6, 3, 1]` in `test_merge_queue_deep_dispatch.py`, and `[6, None×6, 2]` in the integration gate.
8. **"Returns the facts beside its outcome".** Name an additive carrier, as `on_result` and `shadow_baseline_sink` are. A changed return type touches 410 references in 38 test files.
9. **Timed-out mapping.** `verdict_facts` must map the unscoped type-check timeout category to `TIMED_OUT`. `red_verify_outcome` handles it ahead of the ENOSPC test.
10. **`OFF → SHADOW` needs an observed config edge.** A yaml edit while the orchestrator is down never fires it. `off` then `auto` from `ON` is unspecified.
11. **§6.9 wording.** It says β1a declares all event types, then tags `merge_observation (α2)`.
12. **Silent write failure.** `EventStore.emit` never raises, so a failed observation or checkpoint write is silent (INV-11).
13. **D11 has no bound** on batches re-forming on a declared-red main (pass-1 B12). Cite gate 10577 as the owner.

## Verification of the pass-2 changes

| Pass-2 item | Result |
|---|---|
| A1 retention | Answered. Group retention; 0.15 is 0.146; row 10 holds. |
| A2 main unknown | Answered. See the detail below; B1 residual. |
| A3 plumbing | Answered. See the detail below; B8, B9. |
| A4 restart | Answered: one no-await step orders checkpoint and tail; live set; empty start with an exit; row 27 satisfiable. B1 residuals. |
| A5 identity | Partly: K1 restated truthfully, no inferred cut, `classify_content` untouched. New A1. |
| A6 deferral | Answered. See the detail below; B2, B3. |
| A7 switch | Partly: cumulative window, state rows and OFF are in. New A2 and A3; the L15 numbers go to Leo. |
| A8 decomposition | Answered: one chain, θ, η's edges. B5–B7. |
| A9 `predicted_pass` | Answered. |

Pass-2 B1–B15 and B17 are answered. B16 is open, as stated.

**A2 detail.**
- "Proved from birth" matches the landing code: an ordinary landing, a link-by-link batch landing, and a re-verified rebased head (`rebased_head.py::rebased_head_item`).
- The draft correctly leaves these unproved: `same_tree`, `disjoint` and `history_leg` clearances, trains and docs.
- Conviction no longer depends on `main_red`.
- A declared-red root wins over a later pass, because later reds on it are void as recorded.

**A3 detail.**
- Final facts exist in one place for cold, warm and remote paths: `post_merge_verify.py::_PostMergeVerify.run`, after the cross-check.
- The lane can build every observation in `InflightVerifyRun._take_verdict`. It is synchronous and runs before `_release_chain_resources`, so V2 holds.
- A red single is rightly recorded at verify return, not at finalize. The tree fact stands if the cascade later discards the verdict; the base's own red then explains it; batch-mates are released sooner. Between return and finalize it is neither waiting nor in flight, so it cannot be picked or convicted.
- A skip reaches the lane as `verify_skipped`.

**A6 detail.**
- Hook sites exist:
  - the pick: `MergeIntake.pop_next_pickable`;
  - reserved links: `_reserve_and_build_chain`;
  - the built tree: `VerifyDispatcher._start_verify`;
  - the verdict: `_take_verdict`;
  - every other exit: the AST-pinned `_requeue_request` and `_retire_item`. A restart clears in-memory state.
- "The pick returns nothing" is safe:
  - `_acquire_next_request` already blocks on an empty pick without spinning;
  - `_look_ahead` returns `PENDING` and retains its permit, exactly as when every request is reserved;
  - `LookaheadHold` is unaffected.
- With batching on, the merger holds one built item until it has a host (`_holds_lookahead`, `_dispatch_deciding`). "A host is free and nothing built waits" is readable from the allocator and the verifier queue, and needs a port.

## Gates

- **G1:** clean.
- **G2/G6:** the L15 operating point (below); the hold-limit input (B2).
- **G3:** A1 and A3. Other anchors hold.
- **G4:** the journal row (A3).
- **G5:** B1, B3.
- **G7:**
  - INV-11: B12 and the checkpoint lookup.
  - INV-7: the limit wake-up.
  - Others clean.
- **META, still open:** how a lane-initiated rebase keeps identity without tripping the unstack; what counts as a truth.

## Section 2 and the stated numbers

- **Layer table** reproduces from the result files:
  - always stack: +10 to +18, +8 to +15, +8 to +13, +6 to +11;
  - then +10.4, +12.3, about +12, +7;
  - then +16.7, +15.4, +13 to +14, +12 to +13;
  - then +8.2, +4.8, −0.3, +1.2.
- **Worked values** hold, including 0.15.
- **L15 "82%" and "4%"** are arithmetically right (0.824, 0.036) but omit the operating point. The simulation's own wrong share is 5.9% at a defect rate of 0.14 (0.28 wrong to 4.5 right per 100 landings), and 9.4% by construction at 0.08. Output of `switch.py`:

| Wrong share | Unchanged-return share | Trip per window of 20 | Median convictions to first suspension | Median to OFF |
|---|---|---|---|---|
| 5.9% | 1.0 | 0.33 | 30 | 72 |
| 5.9% | 0.5 | 0.12 | 80 | 193 |
| 9.4% | 1.0 | 0.57 | 18 | 41 |
| 9.4% | 0.5 | 0.24 | 42 | 98 |

- **The on-criterion** behaves as ruled. On within 100 truths: 0.93 at precision 0.906, 0.43 at 0.8, 0.16 at 0.7.

## Leo's call

1. **L15's trigger at the lane's own error rate.** Threshold 0.9 admits up to a tenth wrong. Two of 20 is a tenth observed. Unless the true false-failure rate is well under 1%, conviction is expected to turn itself OFF within about 40 to 200 convictions. Options: keep it and accept that; 3 or 4 of 20; or tie the trigger to the threshold.
2. **A3.** Journal the submitter kind, or accept that a recovered hand-submitted request is convictable.
3. **A1.** A separate bounce-base record, or accept that a bounce resets identity (the request becomes a new key at the prior).
4. **B16** (draft 1's "lane state survives restart") is still open, as the coordinator notes.

## Checked and sound

- §6.4 retention and worked values.
- Root rules against `_advance_candidate`, `_land_advanced_candidate`, `_land_rebased_or_park_gate` and `rebased_head_item`.
- L13 through Who (a) and C3; row 24.
- The one-producer path and V2.
- `fetch_events_after` plus the same-step checkpoint ordering.
- `foreign_commit_cut` on a frozen tip of merge commits.
- In-flight hook sites; merger and look-ahead behaviour on an empty pick.
- B3's algebra (all candidates join at rate 0 while false-failure is under 0.5).
- L15 arithmetic.
- §2's layer table.
- Anchors in §4.

## Not checked

- `evidence/REPORT.md` (still absent).
- `git_ops.advance_main` internals.
- How often the bounce fires on the live lane.
- The share of solo verifies whose pass arrives before the entry ahead returns (A2's size).
- Task texts beyond those read in passes 1 and 2.
- No repository tests were run.
