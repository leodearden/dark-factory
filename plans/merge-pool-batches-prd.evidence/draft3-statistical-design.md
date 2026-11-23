# Merge pool, part 2: one batch rule over recorded evidence

**Status:** DRAFT 3 (2026-11-20), not committed. Part 2 of 2; part 1 is `plans/priority-at-later-gates-prd.md`.
Authored for gate task 10069 in an attended session with Leo; his rulings are in section 3.
**Code anchors** verified against main `25354e0e09` (2026-11-20). Main moves fast: cite by symbol; re-locate at
implementation time.
**Evidence:** `plans/merge-pool-batches-prd.evidence/REPORT.md` (measurements, the forensic pass, the lane
simulation and the critic passes, each figure with its counting rule), with the scripts and the simulator beside
it.

## 1. Goal

The merge lane decides who is batched together, who is verified alone, who waits and who is sent back to its
workflow by one rule, computed from one record of what every verify actually tested.

Today those decisions come from two separate mechanisms. A depth table sizes a batch from one lane-wide pass
estimate (task 8528). An in-memory mark makes every member of a red batch verify alone (tasks 10617, 10723); it is
set the moment the verify returns, cannot be revised, and is lost on restart.

A "batch" is what the code calls a deep chain: one dispatched request plus further waiting requests verified as one
tree. A "tree" is everything a verify tested on top of main: the members of the unlanded entries it was built on,
plus its own members. The "root" of a tree is the main commit under all of it.

What an operator observes when it has landed:

1. Every verdict of a dispatched merge verify writes a `merge_observation` row naming every unlanded request in the
   tree (request, task, merge commit, branch tip, content id), the root, whether the verdict is a pass, a red or
   no evidence, the failing tests, and the pass probability the lane predicted for that tree.
2. Batches are sized and filled by the rule. `chain_depth_reason.basis` is `evidence_rule`; the depth table and the
   un-chained mark no longer exist.
3. A batch that goes red while built on unlanded work that then fails leaves its members free to be batched again.
   A batch that goes red on main, or on work that then lands, makes each member expensive to batch: the rule
   verifies it alone or with members of its own set, and passes the others over while that verify runs.
4. When a red batch's culprit is blocked, the batch's other members are batched again at once.
5. The sole request left in a red batch, once its probability of being defective reaches the conviction threshold,
   goes back to its workflow with the batches, failing tests and logs that convict it, and no verify of its own.
6. After an orchestrator restart the same waiting requests have the same probabilities.
7. `get_merge_queue` shows, for each waiting request, its probability of being defective, the rows that implicate
   it and what it is deferred for, and the three fitted rates. A committed readout checks predictions against
   verdicts.

## 2. Background and premise validity (G6)

All figures: dark_factory `runs.db`. The counting rule for each is in `evidence/REPORT.md`.

**How the lane treats a batch today.** `merge_lane/chain_policy.py::choose_depth` picks a depth from
`MergeDeepConfig.depth_table` by a rolling per-item pass estimate; the members are the next waiting requests
(`worker.py::SpeculativeMergeWorker.chain_snapshot`). When the tip verify returns red,
`merge_lane/inflight_verify.py::InflightVerifyRun._settle_chain` marks the owner and every link through
`merge_lane/dispatch_ports.py::ChainPlanning.mark_next_dispatch_unchained`. The mark is a set of task ids in
`SpeculativeMergeWorker._next_dispatch_unchained`, cleared only in `_retire_item`.

**Measured, 2026-11-15T11:19Z to 2026-11-20T07:10Z (214 verdicts, each with its tested tree rebuilt from git).**

- *Failures are deterministic.* No green tree contained a request proved defective with that content. No red tree
  on a sound main had every request in it proved sound (4 reds undetermined). Eight test files explain all 58 red
  batches. The estimator "a red whose members were all later cleared with the same content" found 0 beside 114
  passes (at most 2.6%); a red whose members changed content before a verdict is outside it, so this bounds the
  estimator, not the true rate.
- *A red on a sound base implicates the batch.* 45 reds on main or on a base that had already passed, main not
  red. A defective own member is proved in 21 of 21 batches where every label is proved, and in 43 of 43 when two
  culprits labelled by inference are counted. The reds come from 7 to 9 distinct defective contents.
- *A red on unlanded work says little until that work has a verdict.* 25 reds were on a base with no verdict yet.
  18 of those bases then went red; the batch's own members were then defective at 0.07 to 0.09 each, against a
  prior of 0.09 to 0.12. The base's verdict followed after 12.5 minutes at the median, 43 at most. Today all those
  members are marked anyway: 13 of 58 red batches were red only because of unlanded work ahead.
- *A red main is a third of the reds.* 30 of 62 reds on main fell in three episodes. Each began with a commit that
  did not come through the lane; one has no `main_health_red` row; the longest was healed by a task that landed
  through the lane (10239). Seven requests were blocked as branch bugs for a test that was failing on main.
- *The mark does not survive a restart.* After task 10617 deployed, all 31 red-batch members that were batched
  again before their solo verify had crossed an orchestrator restart.
- *A request id is not an identity.* The lane re-reads the branch at every merge attempt
  (`worker.py::classify_and_merge`) and rebases waiting branches itself
  (`suffix_graph.py::SuffixConflictTracker.bounce_conflicting_suffix_items`; `GitOps.unstack_from_unlanded_base`).
  Request `mr-8ddfc972` carried two tips a day apart.
- *Per-request defect rate.* 0.14 over 1,237 requests (0.07 to 0.24 by week).
- *A batch verifies in the time of one request* (ratio 0.98 against same-day, same-host singles, 81 batches).

**Simulated** (the 8528 study's lane simulator, extended and re-validated; 2 hosts, 3 entries in the frontier, 75
waiting; landings per verify-hour; main-red episodes and restarts modelled). Gains are against today's lane with
its accidental main-based verifies, which task 11002 removes. That first step has two calibrations that disagree,
so it is a range:

| Layer | Gain at defect rates 0.05 / 0.08 / 0.14 / 0.20 |
|---|---|
| Always stack (task 11002) | +10 to +18 / +8 to +15 / +8 to +13 / +6 to +11% |
| Then: reds resolved by the tree they tested, a red main discounted, state that survives restart | a further +10 / +12 / +11 / +6% |
| Then: release when the culprit is blocked, one test of a set at a time, conviction | a further +17 / +15 / +14 / +13% |
| Then: the rule of 6.5 in place of table depth | a further +8 / +5 / 0 / +1% |

Of the second row, the unlanded-work part alone is +1.3% at a defect rate of 0.14; most of it is surviving
restarts and discounting a red main. The simulation fixed the two rates the rule reads. The contract fits them
from live rows, which was not simulated as such; the one adaptive variant moved the rule's result between −3% and
+10%. β1b's shadow rows and ε exist to check this on the lane before β2a switches over.

**Premises checked.**

- *The rule is the table when it knows no more than the table.* With one failure chance for every request and
  "every member of a red batch alone" as the follow-up cost, the rule picks the table's depth at 18 of 20 pass
  estimates from 0.75 to 0.995 and one less at two exact row boundaries.
- *Conviction needs both restrictions of 6.6.* The rule mixes red-batch members into batches with fresh requests
  when that is cheaper; a second red can then push a sound request past the threshold (7 wrong in 485 with no
  false failure in the model). Restricted to the sole request left in a red observation, that error is gone (0 in
  2,556); what remains is false failures, 0.28 wrong per 100 landings at a defect rate of 0.14 and a false-failure
  rate of 1%. The second restriction, "the root was proved sound before the red", is not simulated: it answers the
  measured case of a red main healed through the lane.
- *Exact inference is cheap enough.* Requests that appear in exactly the same observations are interchangeable, so
  the states to enumerate are counts per such class. A batch of 6 built on two batches of 6 is 343 states.
- *Almost every number here comes from rows before task 8528 deployed.* After it there are 13 batches and 4 single
  verifies. No constant is taken from those. The lane fits its rates from its own rows; ε reports them.

## 3. Resolved design decisions

**Carried from part 1 and gate 10069 (not reopened).**

- **R1.** Nothing merged or being verified is displaced by a later arrival.
- **R7.** A request implicated by a failed batch does not sit unresolved: it is cleared, sent back, or verified.
- **R10.** A request that fails goes back to its workflow; every default is set from measured rows.

**Ruled by Leo, 2026-11-20, in this session.**

- **L1. Scope.** Evidence that is recorded, revised and restart-safe; release and conviction.
- **L2. Mechanism.** Exact joint inference over every recorded tree.
- **L3. One rule.** Batch membership is one rule computed from all available information. It supersedes the depth
  table and any special case for red-batch members; the follow-up cost of a red batch is a term of the rule.
- **L4. What a red verdict means.** On work with no verdict yet it is weak evidence against the batch's own
  members; on work that passed, strong; on work that went red, about none. One model yields all three.
- **L5. Main is one more unknown.** The main-health probe keeps the authority to declare main red.
- **L6. The false-failure rate** is `(unexplained reds + 1) / (sound-tree verdicts + 38)`: it starts at the
  measured bound of 0.026.
- **L7. Conviction** threshold 0.9 (ruled 2026-11-13). It runs in shadow and switches itself on from recorded
  rows. There is no control sample.
- **L8. Sequencing.** These tasks land ahead of task 5041's extraction, which waits for them.
- **L9. The 2026-11-18 ruling is amended.** The evidence ledger replaces the un-chained mark as the lane's single
  owner of red-batch isolation (decision D6 of `plans/merge-lane-policy-prd.md`), and the rule replaces its depth
  table (D7).
- **L10. Measure.** Fewest verifies per landing.
- **L11. Phase 2 is a design gate behind this PRD** (section 9).
- **L12. "Always stack"** is task 11002, filed by the merge-slot session; not part of this PRD.
- **L13. Only a red on a main already proved sound can convict.** A red recorded on an unproved main still shifts
  probabilities; a pass that arrives after reds on the same main does not prove it.
- **L14. Deferral is bounded per test in flight and is not a queue jump.** This amends gate 10069's "verified
  alone ahead of every other pick". Order stays part 1's, which promotes a request after a long wait.
- **L15. The switch.** On when at least 10 shadow truths have accumulated since the last state change and at most
  a tenth are wrong. Suspended when 2 of the last 20 convictions came back unchanged and passed. A suspension
  after one re-qualification turns conviction off until it is re-enabled by config.
- **L16. Reach.** A conviction is routed as a red verify of that request would be: to its workflow when one is
  waiting, as an L1 when the request was recovered after a restart. A request submitted by hand is never
  convicted.

**Decided here.**

- **D1. An observation is a tested tree with the verdict that counted**, recorded by the lane when the verdict is
  final. Nothing is "provisional": the probabilities are recomputed from all observations whenever one arrives.
- **D2. Identity is the branch's own changes**, by the definition task 10215 put on main (6.1). A member is keyed
  by `(task_id, content_id)`. That definition's ids include context lines, so a rebase that changes a line beside
  a hunk changes the id; the request is then treated as new. This loses evidence and never convicts. ε counts it.
- **D3. `merge_verify` stays the record of an attempt; `merge_observation` is the record of a verdict.** A verify
  can have several attempts, a flake re-run and a cross-check before its verdict counts.
- **D4. A pass clears; a block is not a rule.** A request blocked on its own red verify is explained by that red
  observation, which is why its batch-mates fall back to the prior. Evidence is kept while any request connected
  to it is still in the lane.
- **D5. A request that leaves without a verdict stays in its observations** as a possible culprit that can no
  longer be verified.
- **D6. Deferral.** A waiting request whose probability a verify in flight can still change is passed over while
  anything else can be dispatched. Information is free while waiting is free.
- **D7. A high-lane request is never deferred.** That lane is for main repair.
- **D8. Conviction is strict and at most once per content** (6.6). A request with no content id is never convicted.
- **D9. The work discarded behind a red batch is not a term of membership.** With the term the simulated lane lost
  14% at a defect rate of 0.20. That waste is decided by what the next entry is built on, which is phase 2.
- **D10. The rule plans as if the root is sound.** If main is red every tree fails whatever it holds. The root's
  probability enters the inference and the recorded prediction, not the batch size.
- **D11. A declared-red main voids every red on it**, whatever test failed. Nothing can land while main is red;
  a culprit hidden by the void is found again by the first batch after the heal.
- **D12. The rates have one home**, the ledger, as pure functions of its rows; their starting values are config.
- **D13. No table mode.** `merge_deep.chain_cap: 0` stays the switch that turns batching off. Tests pin a batch
  size by a starting defect rate of 0 and `max_fresh`.
- **D14. A failed rebuild starts from an empty record, says so, and escalates.** That is today's behaviour after
  every restart.
- **D15. Failing tests, the failure signature and the log location are stored with every observation**, as data.
- **D16. Docs-lane requests (task 10996), trains and a request still stacked on an unlanded base are never batch
  members.** Verifies that are not a full gate make no observation.
- **D17. New lane code goes in new modules** inside the merge-lane ratchet's ceilings; each leaf records its raise.

**Considered and rejected** (measured in `evidence/REPORT.md`).

- *"Add a member while this verify's expected landings rise"* (gate 10069's wording). No cost term: 7, 12 and 19
  members at defect rates 0.14, 0.08, 0.05, and −5 to −9%.
- *A rule that minimises the batch's average cost.* It adds safe requests to a risky request's verify because they
  lower the average.
- *Table depth with a "suspect" flag above 1.2 times the prior* (draft 1). A batch of 12 makes no suspect, and
  the identical batch can re-form.
- *Three bookkeeping rules instead of inference.* Same throughput at a defect rate of 0.14; more wrong
  convictions (0.30 against 0.21 per 100 landings), no prediction, no way to use better per-request information.
- *Extending the `merge_verify` row* (draft 2). It is written per attempt, before the cross-check.
- *"A pass proves the main under it"* (draft 2). False when the pass is the fix (L13).
- *Voiding only reds whose failure signature matches the probe's* (draft 2). D11 is simpler and loses one verify.
- *Batching off when the record cannot be rebuilt* (draft 2). It had no exit.
- *Crediting elimination inside the going cost.* Unstable in the simulation.
- *A control sample of convictions.* Leo ruled it out.
- *Holding a host idle behind a risky entry.* Gate 10572 ruled that spare capacity goes to merge verifies.
- *Context-free patch ids.* More stable under rebase, but a second definition of "same content" beside task
  10215's. Revisit in phase 2 if ε shows the loss matters.
- *A per-request prior from task features.* +1 to +2% with the best signal found.

## 4. Pre-conditions (G3)

Verified on main, by symbol:

| Assumed | Evidence |
|---|---|
| Singles, batches and rebased-head re-verifies are dispatched through one path | `merge_lane/verify_dispatch.py::VerifyDispatcher.dispatch` → `_place` → `_start_verify` |
| The entry a verify is built on can be found by base link | `worker.py::SpeculativeMergeWorker._frozen_base_chain`; `merge_lane/remerge_base.py::entry_successor_base`, `entry_bases` |
| A fact can be published once onto an in-flight entry after dispatch | `merge_lane/types.py::InflightEntrySlot.publish_verify_base` |
| The dispatching request's merged tip; a batch's links | `types.py::RealMergeItem.merged_branch_tip`; `types.py::ChainResult.links`, `link_request_ids` |
| A branch's patch ids and "same content" | `merge_lane/landing_evidence.py::range_patch_ids`; `merge_lane/content_relation.py::classify_content` |
| A stacked branch's own-delta base, from a recorded stack base | `git_ops.py::GitOps.own_delta_upstream`; `branch_stack.py::StackBaseLedger`, `foreign_commit_cut` |
| The final verify result has failing test ids, category, timeout, trivial and flake facts | `verify.py::VerifyResult` (`failing_test_ids` only for a full-breadth merge verify with parsed junit) |
| The lane learns a red's verdict synchronously, before the batch's links are released | `inflight_verify.py::InflightVerifyRun._take_verdict`, then `run`'s `finally` → `_release_chain_resources` |
| Where a red batch and a single are settled; where a raising batch is guarded | `InflightVerifyRun._settle_chain`, `_settle_single`, `_on_verify_error` |
| Where a batch excludes a request and where its size is chosen | `SpeculativeMergeWorker.chain_snapshot`; `_deep_chain_placement` (guard 2b; `choose_depth`) |
| A filter at the pick | `merge_lane/intake_buffers.py::MergeIntake._clique_minimal_visible` (its `visible` list, beside `is_reserved`) |
| Resolving a request that still waits, with its removal from the buffer | `suffix_graph.py::SuffixConflictTracker._escalate`; `MergeIntake.remove_buffered` (merger-owned) |
| Whether a workflow waits on a request | `merge_lane/types.py::MergeRequest.awaited_by_workflow` |
| The consumers of a blocked outcome | `workflow.py::TaskWorkflow._submit_to_merge_queue`; `recovered_merge_owner.py::RecoveredMergeOwner._file_l1`; `workflow.py::TaskWorkflow._check_merge_outcome_thrash` (skipped when `_last_merge_block_reason` is cleared) |
| A block category must be an escalation category | `merge_lane/outcome_facts.py::MergeBlockCategory`; `escalation/src/escalation/categories.py::EscalationCategory` |
| The site that confirms a red main after re-checking that main has not moved | `merge_lane/main_health_probe.py::_run_deferred_main_health_probe` |
| What main's tip is after a landing | a member's merge commit; after a batch, its chain tip, reached link by link; after a re-verified rebased head, the rebased commit (`merge_lane/rebased_head.py::rebased_head_item`) |
| A new event is one enum member; rows are read back oldest first across runs | `event_store.py::EventStore.emit`, `fetch_recent_events_by_type` |
| Recovered requests exist before the merge worker runs | `harness.py::Harness._recover_pending_merges` |
| Lane scenes with tree-keyed verdicts; deep scenes pin depth through the table | `orchestrator/tests/_merge_lane_fakes.py::make_lane`; `test_merge_queue_deep_unchained_mark.py`; `orchestrator/tests/_merge_deep_scene.py` |

Not on main; built here unless a producer is named:

| Needed | Delivered by |
|---|---|
| A stack-base record for a branch the lane bounced onto the frozen tip (the bounce writes none today, so `own_delta_upstream` answers main) | α1 |
| The tested tree on an in-flight entry; a link's branch tip (today only its merge commit is kept) | α1 |
| Verdict facts for a pass as well as a red, from the final result (today a pass returns `None`: `post_merge_verify.py::_PostMergeVerify._pass_outcome`) | α2 |
| A read of the event store that raises on failure and returns rows after a given row (today `fetch_recent_events_by_type` returns `[]` on error) | β1a |
| The main commit on the main-red row | γ |
| A block category and escalation category for a conviction | δ |
| A restart scene for lane tests (precedents: `test_merge_queue_restart_hook.py`, `TestChainPolicySeeding`) | β1a |

Used if present, not required: task 9144 (`MergeOutcome.kind`), task 7921 (main-red verdict book), task 9933
(`merge_pool_config.py`), task 9182 (one terminal hook), gate 10217's leaves (`MergeContent`, `ChainLink`).

## 5. Cross-PRD relationships and seam owners (G4)

| Other work | Direction | Seam | Owner | Action |
|---|---|---|---|---|
| Tasks 10617, 10723, 8528, 10920; `merge-lane-policy-prd.md` D6, D7 | supersedes | red-batch isolation; batch depth | this PRD (L9) | β2a removes `chain_policy.py` and `depth_table` with a dated delta note on D7; β2b removes the mark with one on D6 |
| Task 10624 (mark order, coalescing honours the mark) | absorbs | the same sites | this PRD | item 1 is α2's V2; item 3 is β2b's coalescing rule; item 2 is moot. Cancel at decompose; its dependent 10759 is re-pointed in the same step |
| `plans/one-merge-request-per-task-prd.md` (task 10215 done; 10216; gate 10217) | consumes | "a branch's own content"; the single writer of a waiting request's content; the link record | that PRD | α1 reuses `range_patch_ids` unchanged and adds the bounce's stack-base record. Dated note on gate 10217: its `MergeIntake.update_waiting` calls `EvidenceLedger.note_identity`; its `ChainLink` is the base of `TreeMember`; `CONVICTED` joins its `REJOIN_ON` set |
| Task 9908 (`merge_verify.chain_request_ids`) | none | what an attempt dispatched | 9908 | unchanged (D3) |
| Task 7824 (build the batch inside the verify task) | none | when a batch's members are known | 7824 | α1 publishes the tree when the build returns, wherever the build runs. Dated note on 7824 |
| Task 10995 (a green tree lands what it contains) | produces for | which requests a tree contains | α1 builds `InflightEntry.tree`; 10995 reads it | 10995 depends on α1. Its item (f) is this model's "green on a red entry": the pass clears every member |
| Task 11002 (always stack) | none | what an entry is built on | 11002 | its policy function is where phase 2's base decision plugs in |
| Task 10721 (a passed head re-verified alone) | consumes | a re-verify of a rebased head takes no members | 10721 | the rule adds nothing to such a dispatch; its verdict is one more observation on the new root |
| Tasks 8933, 8329, 10136, 9133 | same files | `worker.py`, `verify_dispatch.py` | each its own | serialised by the file lock; 8329's cancelled verifies write no observation |
| Task 5041 (extract `speculation.py`) | must follow | the placement code | 5041 moves; this PRD edits first (L8) | 5041 depends on β2b; gate 10573 re-points names |
| Part 1, task 9935 (pool order) | consumes; amends | the pick filter; "task 8528's suspect list" | 9935 owns `pool_order()`; β2b owns the deferral predicate | 9935's text: no fronted list exists; deferred requests are ordered like any request; the filter keeps β2b's predicate. 9935 depends on β2b |
| Tasks 7921, 9171, 9156; gate 10577 | consumes | "main was red at commit M" | the main-health probe | γ adds `main_sha` at the deferred emit site and owns the void. Dated note on 7921: its verdict book reads the same field and calls the same void; its "held" is a different thing from this PRD's "deferred". 9171 and 9156 move code around that site: file lock only |
| Task 9182 (one terminal hook) | consumes | a request leaving the lane | 9182 | β1a calls `note_gone` in `_retire_item`; 9182 carries it onto its hook. Dated note |
| Tasks 9144, 9207, 9852 | produces | the conviction outcome | δ adds the category, the field and one workflow arm | dated notes: 9144 adds the `convicted` kind, 9207 its table row, 9852 its colour |
| Task 9766 (moves `EventType`) | none | where event names live | 9766 | new event types are declared in `EventType` wherever it lives at dispatch |
| Tasks 5411, 10996 (docs gate, docs lane) | consumes | a docs-lane verify is not a full gate | 10996 | D16; dated note on 10996 |
| Tasks 7919, 7922 (pre-verify screen) | none | a screen bounce retires the request before any verify; a truncated batch is recorded as built | the screen PRD | — |
| `plans/merge-chain-visibility-prd.md`; tasks 10759, 10761, 10762 | amends | "isolation" and `unchained_mark` are defined by the mark | this PRD defines who is implicated (`implicated_by` in the snapshot); that PRD records and shows it | θ: a design gate that revises that PRD. 10759, 10761 and 10762 depend on θ |
| Task 8532 (7-day readout) | none | reads the lane as deployed | 8532 | it fires before β2a can land; no edge |
| Task 8932 (idle-host hedge, dense table) | superseded | the depth table; what a spare host verifies | this PRD; phase 2 | cancel at decompose |
| Gate 10572 rulings | consumes | merge verifies take spare capacity | 10572 | the lane stays work-conserving (6.5) |
| Tasks 5623, gate 10217 leaf C | none | the journal | each its own | this PRD adds no journal field |

## 6. Contract (B + H)

### 6.1 Identity

In `merge_lane/landing_evidence.py`, beside `range_patch_ids`:

```
async def own_delta_patch_ids(git_ops: GitOps, *, branch: QueuedBranch, tip: str, target: str) -> RangePatchIds
def content_id(ids: RangePatchIds) -> str | None
```

- `own_delta_patch_ids` is `range_patch_ids` from the branch's own-delta base to `tip`. The base is `target`, or
  the foreign cut when `StackBaseLedger` holds an unlanded base for the branch. `GitOps.own_delta_upstream` gains a
  `tip` argument so it can answer for a tip that is not the live branch.
- `SuffixConflictTracker.bounce_conflicting_suffix_items` records the frozen tip as the bounced branch's stack
  base. No other cut is inferred from the commit graph.
- `content_id` is a hash of the sorted patch ids. It is `None` when the range is empty or holds a merge commit
  (a branch that merged main into itself has no id).
- It is memoised by `(tip, base)`, never by tip alone.
- **K1.** The lane's unstack and its bounce onto the frozen tip leave `content_id` unchanged when the commits they
  move the branch across do not touch its hunks' context lines; so does any such rebase.
- **K2.** A request whose id is `None` is keyed by its tip and is never convicted.

### 6.2 The tested tree

```
@dataclass(frozen=True)
class TreeMember:
    request_id: str
    task_id: str
    merge_commit: str        # this member's layer in the tree
    tip: str                 # branch tip merged in that layer
    content_id: str | None
    own: bool                # True: a member of the verifying entry; False: of an entry it was built on

def tested_tree(own: Sequence[TreeMember], built_on: Sequence[TreeMember]) -> tuple[TreeMember, ...]
```

- `InflightEntry.tree` and `InflightEntry.tree_root` are published once, through the entry slot, when the entry's
  build returns and before its verify starts. They are never mutated.
- The base is resolved by link, never by position in the frontier:

  | The item's base is | `built_on` | `tree_root` |
  |---|---|---|
  | main's tip, or a commit main already contains | empty | that commit |
  | the successor base of a live in-flight entry | that entry's tree | that entry's `tree_root` |
  | anything else (a commit no live entry owns; task 8933's class) | unresolved | unresolved |

- A link's tip is its merge commit's second parent. A member's `content_id` is computed from its tip in that tree.
- A request recovered from the journal gets its content id in the background, at most four at a time. Until it is
  known the request is not a candidate member.
- Verifies with no in-flight entry are outside this contract: a train's inline verify and its gate re-verify
  (`worker.py::_do_train_merge`), `worker.py::reverify_member_solo`, `VerifyDispatcher.gate_history_leg`, and
  shadow verifies (`merge_lane/shadow.py`).
- **T1.** `tree` lists each request at most once; own members are last, in land order.
- **T2.** Two entries contain the same build of a request exactly when their trees hold the same `merge_commit`.

### 6.3 Observations

```
class Evidence(StrEnum): PASS, RED, NONE
class NoEvidence(StrEnum): TIMED_OUT, INFRA, TRIVIAL, SKIPPED, NOT_FULL_GATE, BASE_UNRESOLVED

@dataclass(frozen=True)
class VerdictFacts:
    evidence: Evidence
    no_evidence: NoEvidence | None            # set exactly when evidence is NONE
    failing_test_ids: tuple[str, ...]         # sorted junit node ids, first 50; empty when the runner gives none
    failure_signature: tuple[str, str] | None # (category, normalised cause hint)
    flake_suppressed: bool
    log_dir: str | None

def verdict_facts(result: VerifyResult) -> VerdictFacts

@dataclass(frozen=True)
class Observation:
    obs_id: str                               # merge sha of the verified tree
    root: str
    members: tuple[TreeMember, ...]
    facts: VerdictFacts
    root_proved: bool                         # was the root proved sound when this was recorded (6.4)
    predicted_pass: float | None              # for this tree, from the posterior at dispatch (β1b)
    plan: PlanFacts | None                    # 6.5 (β1b)
    at: datetime
```

- `verdict_facts` is total and pure, over the final result of a verify (after its attempts, its flake re-run and
  its cross-check):

  | Final verify result | Evidence |
  |---|---|
  | passed, suite ran | `PASS` (`flake_suppressed` when the re-run turned a red into this pass) |
  | passed, `trivial` (no suite ran) | `NONE`, `TRIVIAL` |
  | timed out | `NONE`, `TIMED_OUT` |
  | red in an infra class: ENOSPC, transient infra, lock contention | `NONE`, `INFRA` |
  | red, anything else | `RED` |

  The infra test is one predicate shared with `verify_failure_outcome.py::red_verify_outcome`.
- `post_merge_verify.py::run_post_merge_verify` returns the facts beside its outcome, for a pass as for a red.
- The lane is the only producer of an observation. `InflightVerifyRun` builds it from the entry's tree and the
  facts as soon as the verdict is known. It sets the three lane-level reasons itself: `SKIPPED` (the low-disk skip,
  which never dispatches), `BASE_UNRESOLVED` (6.2), `NOT_FULL_GATE` (D16).
- `EvidenceLedger.record` writes the `merge_observation` row and applies the observation in one step, with no
  await between. Wire types and decode live in `shared/src/shared/merge_evidence.py`; an unknown enum value raises
  with field and value, and a reader counts and skips such a row.
- A verify that raises or is cancelled makes no observation.
- 20 consecutive observations with `BASE_UNRESOLVED`, or with a member whose content id is `None`, file one L1,
  deduplicated while open.
- **V1.** A `NONE` verdict clears nothing, implicates nothing and proves no root.
- **V2.** A red's observation is recorded before the batch's links are released from their reservation.

### 6.4 The ledger and inference

```
ContentKey = tuple[str, str]                  # (task_id, content_id), or (task_id, 'tip:' + tip) with no id

class EvidenceLedger:
    def record(self, obs: Observation) -> None
    def declare_main_red(self, main_sha: str) -> None                          # 6.7
    def note_identity(self, request_id: str, key: ContentKey) -> None
    def note_in_flight(self, request_id: str, keys: Collection[ContentKey]) -> None   # picked; reserved; built
    def note_returned(self, request_id: str) -> None
    def note_gone(self, request_id: str) -> None
    def posterior(self) -> Posterior
    @classmethod
    def rebuild(cls, checkpoint: Checkpoint | None, tail: Sequence[EventRow],
                live: Mapping[str, ContentKey]) -> EvidenceLedger

@dataclass(frozen=True)
class Posterior:
    version: int
    def p_defective(self, key: ContentKey) -> float
    def p_all_sound(self, keys: Collection[ContentKey]) -> float     # given every root sound
    def p_root_sound(self, root: str) -> float
    def implicated_by(self, key: ContentKey) -> tuple[str, ...]      # non-void red observations holding it
    def sole_remaining(self, key: ContentKey) -> tuple[str, ...]     # of those, where it alone is left
    def group_of(self, key: ContentKey) -> frozenset[ContentKey]     # keys joined to it by shared observations
    def is_pending(self, key: ContentKey) -> bool
    rates: Rates

def infer(observations, cleared, roots: RootFacts, rates: Rates, *, max_states: int) -> Posterior
```

- **Roots.** A root is *proved sound* (a) from birth, when it is a member's merge commit in a recorded `PASS`
  tree, which is what main's tip is after the lane lands a verified tree; or (b) by a `PASS` recorded on it when
  no `RED` was recorded on it before. Any other root is *unproved*: an outside commit, a landing cleared by a gate
  without a verify of the landed tree, a train or docs landing. A root can also be *declared red* (6.7).
- **Model.** The unknowns are: each content key is defective (prior `defect_rate`); each unproved root is red
  (prior `main_red`). They are independent. A tree is red when its root is red or it holds a defective key;
  otherwise it is red with probability `false_failure`. The posterior is exact, over the non-void `RED`
  observations.
- **A pass** clears every key in its tree. A cleared key has probability 0 and is removed from every red
  observation. A red observation whose keys are all cleared, on a root proved before it was recorded, is an
  *unexplained red*.
- **Groups and retention (D4).** Observations that share a key are one group. A group leaves the ledger when none
  of its keys belongs to a live request. A shared unproved root couples the inference but does not make a group.
- **A key whose request has gone** stays in its observations (D5). **A changed content** is a new key at the
  prior; the old key stays as in D5.
- **Enumeration.** Keys in exactly the same observations are interchangeable; a group's states are the counts of
  defective keys per class. The states of an unproved root are summed over, not multiplied in. A group with more
  than `max_states` states is not enumerated: each of its keys takes the largest probability any single one of
  its observations gives it, none is convicted, and the dispatch's `plan` says so. Ten such dispatches in a row
  file one L1.
- **Recompute.** A group of at most `sync_states` states is recomputed inside the call that changed it. A larger
  one is recomputed off the event-loop thread; until it finishes its keys are `is_pending`, a result for an older
  version is discarded, and 6.5 and 6.6 treat a pending key as deferred.
- **Rates** (D12), one function over the ledger's rows; each starting value is a config leaf:

  | Rate | Definition | Starts at |
  |---|---|---|
  | `defect_rate` | from the last 300 *fresh* trees on a proved-sound root (no key seen in an earlier observation): `1 − pass_share ** (1 / mean size)`, smoothed by one prior tree at the starting value | 0.14, measured |
  | `false_failure` | `(unexplained reds + 1) / (passes + unexplained reds + 38)` over the last 500 such verdicts | 0.026 (L6) |
  | `main_red` | `(declared-red roots + 1) / (roots not proved from birth + 17)` over the last 200 such roots | 0.06: 3 episodes in about 50 such roots; β1a recounts over the 14 days before it is implemented |

  300 trees give a 95% interval of about ±0.02 at a defect rate of 0.14. 500 verdicts let an undetected red-main
  episode age out in about ten days. After conviction is on, `false_failure` is biased low: a convicted request
  never gets the pass that would have made its red unexplained. ε reports the count of such convictions.
- **I1.** `infer` is pure and does not depend on the order of its inputs.
- **I2.** With `false_failure` 0 and every root proved, `p_defective` is 1 only for a key that is defective in
  every hypothesis consistent with the observations.
- **I3.** `rebuild` from the rows a ledger wrote gives the same `Posterior` as that ledger held, for the requests
  in `live` (6.8).
- **I4.** After a red observation of a set S of fresh keys on a proved root, with nothing else recorded,
  `p_all_sound(S)` is `s·f / (1 − s·(1 − f))`, where `s = (1 − defect_rate)^|S|` and `f` is `false_failure`.

Worked values at a defect rate of 0.14, a false-failure rate of 0.01 and a proved root, for a batch of 3: red on
main, each member 0.38; red while built on an unverified batch of 3, each 0.23; that batch then passes, 0.38; it
goes red instead, 0.14; two of the three then pass alone, the third 0.94. A member blocked on its own red leaves
its two batch-mates at 0.15.

### 6.5 The rule

A pure module, `merge_lane/batch_rule.py`:

```
def going_cost(p: float, rates: Rates, *, max_fresh: int) -> float
def batch_cost(members: Sequence[ContentKey], built_on: Collection[ContentKey], post: Posterior,
               *, max_fresh: int) -> float

@dataclass(frozen=True)
class PlanFacts:
    members: tuple[str, ...]                          # request ids, the dispatching request first
    cost: float
    predicted_pass: float                             # for the planned tree
    deferred: tuple[tuple[str, str, datetime], ...]   # (request id, the request it waits for, since)
    capped: Literal['fresh', 'total'] | None
    not_enumerated: bool

def deferred_for(key: ContentKey, post: Posterior, in_flight: Collection[ContentKey]) -> ContentKey | None
def plan_batch(front: ContentKey, candidates: Sequence[ContentKey], built_on: Collection[ContentKey],
               post: Posterior, *, max_fresh: int, max_total: int) -> tuple[ContentKey, ...]
```

- **Probabilities.** For a set X built on A, with every root taken as sound (D10):
  `g(X) = p_all_sound(A ∪ X) / p_all_sound(A)`; a member's `p = 1 − g({member})`.
- **Going cost** `a(p)`: the expected verifies to resolve one request whose probability is `p`.
  `a(0) = 1 / (max_fresh · (1 − false_failure))`. For `p > 0`,
  `a(p) = min over k = 1..max_fresh of [1 + R_k · k · a(p / R_k)] / k`, with
  `R_k = 1 − (1 − p)^k (1 − false_failure)` and the `k = 1` term equal to 1. It is tabulated on a grid of 4,000
  steps from `p = 1` downward and interpolated.
- **Batch cost.** `P_red = 1 − (1 − false_failure) · g(S)`. `C(S) = 1` for one member; otherwise
  `C(S) = 1 + P_red · Σ a(p_i / P_red)`.
- **Members.** The dispatching request is the first member. The candidates are walked in pool order; `c` joins
  when `C(S ∪ {c}) − C(S) < a(p_c)`. Once `max_fresh` members that are not cleared have joined, the walk goes on
  for cleared requests only, up to `max_total` members.
- **Deferral (D6, L14).** `deferred_for` answers the in-flight key that a waiting key shares a group with, if any.
  *In flight* means, from `note_in_flight` until `note_returned`: a request that has been picked and has no
  verdict, in any state between; the members of its tree once built; and the links a batch being built has
  reserved. A pending key is deferred for itself. A deferred request is neither picked nor a candidate.
  - The clock is per test: it starts when the request is first passed over for a given in-flight request, and is
    dropped when that request returns. Past `hold_limit_factor` times the 95th percentile of a passed verify's
    duration in the ledger's rows, the request is no longer deferred for that test. No limit applies before the
    ledger holds 50 such durations.
  - A high-lane request is never deferred (D7).
  - The pick returns no request rather than a deferred one, unless a verify host is free and no built item is
    waiting for it; then it returns the first deferred request, which is dispatched alone.
- **A batch that returned no evidence, or whose verify raised.** Its owner's next dispatch takes no members. After
  a restart this is read back from `merge_observation` rows; a raise leaves no row and is forgotten.
- **Never members:** D16's classes, as `worker.py::_link_refusal` refuses them today.
- **Already built.** A request parked for redispatch is dispatched as it was built (R1). A re-verify of a rebased
  head takes no members (task 10721).
- **Where it runs.** The pick filter beside `is_reserved` in `MergeIntake._clique_minimal_visible` applies
  deferral. The placement calls `plan_batch` over `chain_snapshot`'s candidates in place of `choose_depth`.
  Coalescing skips a deferred request and one whose `p` is above the prior
  (`worker.py::_default_coalesce_exclusion_reason`).
- **Recorded.** `Observation.plan` and `Observation.predicted_pass`, where
  `predicted_pass = (1 − false_failure) · p_root_sound(root) · P(every key of the dispatched tree sound)`.
  `merge_verify.chain_depth_reason` keeps its keys: `basis` is `evidence_rule`, `pass_estimate` is
  `1 − defect_rate`, `row` is null.
- **Limits.** `max_fresh` 12, the deepest row the lane's table has ever allowed; `max_total` 24, which bound
  under 0.2% of simulated batches.
- **B1.** With one probability for every candidate and `a(p / R_k)` forced to 1, `plan_batch` returns the depth of
  the table task 10920 derived, at every pass estimate at least one grid step inside a table row.
- **B2.** After a red observation of a batch S on a proved root, with nothing else changed, `plan_batch` does not
  return S.
- **B3.** With a defect rate of 0 and no observations, `plan_batch` returns `max_fresh` members when that many
  wait. This is how a test pins a batch size (D13).
- **B4.** A free host and a request that can be dispatched always produce a dispatch.
- **B5.** `plan_batch`, `going_cost`, `batch_cost` and `deferred_for` are pure.

### 6.6 Conviction

```
@dataclass(frozen=True)
class ConvictionEvidence:
    conviction_id: str
    p_defective: float
    content_id: str
    observations: tuple[str, ...]                 # obs ids it alone is left in
    batches: tuple[tuple[str, ...], ...]          # task ids of each of those trees
    cleared: tuple[str, ...]                      # task ids cleared out of them
    failing_test_ids: tuple[str, ...]
    log_dirs: tuple[str, ...]

class ConvictionState(StrEnum): SHADOW, ON, SUSPENDED, OFF
```

- **Who.** A waiting request, not reserved, not picked and not parked, that (a) is the sole key left in at least
  one non-void red observation recorded with `root_proved` true (L13), (b) has `p_defective` at least
  `convict_threshold`, (c) has a content id, (d) is awaited by a workflow or is owned by the recovered-merge owner
  (L16), (e) has a key that was never convicted, and (f) is not pending, in a posterior whose version is the
  ledger's.
- **Where.** In the merger loop before a pick, which owns the buffers. The branch tip is read again; if it is not
  the tip the key was computed from, the identity is refreshed and nothing is convicted in that pass.
- **What.** The request is removed from its buffer and resolved
  `MergeOutcome('blocked', reason, block_category=MergeBlockCategory.CONVICTED, conviction=...)`. The reason
  carries the `conviction_id`, so it never repeats. `CONVICTED` is added to `EscalationCategory`.
  - `TaskWorkflow._submit_to_merge_queue` gets one arm on `result.conviction`, ahead of the generic arm: it
    writes the evidence into the merge-failure review, emits `merge_blocked`, clears the last-merge-failure
    fields, and calls `_mark_blocked` as the generic arm does for a post-merge verify failure. The thrash ladder
    therefore never counts a conviction.
  - `RecoveredMergeOwner._file_l1` files a recovered request's conviction under the new category, unchanged.
- **A convicted content that returns unchanged** is verified alone and is never convicted again (D8).
- **Rows.** `merge_conviction` (id, key, `mode: shadow | on`, the evidence), at most one per key and mode;
  `merge_conviction_truth` (conviction id, `defective | sound`, the obs id that showed it);
  `merge_conviction_state` (from, to, reason, counts). A truth is an observation recorded after the conviction
  row, on a root proved before it, in which the key is the only one not cleared: `RED` is `defective`, `PASS` is
  `sound`. The state is the newest `merge_conviction_state` row; with none, `SHADOW`.
- **State (L15).**
  - `SHADOW`: a would-be conviction writes its row and the request is treated as any other.
  - `SHADOW → ON` when the shadow truths since the last state row number at least 10 and at most a tenth are
    `sound`.
  - `ON → SUSPENDED` when 2 of the 20 newest `on` convictions have a `sound` truth. One L1. If half the wrongly
    convicted return unchanged, this catches a 30% wrong rate 82% of the time and trips at a 3% wrong rate 4% of
    the time.
  - `SUSPENDED → ON` when the shadow truths since the suspension number at least 20 and at most a tenth are
    `sound`.
  - A suspension after that is `OFF`, with one L1. `OFF → SHADOW` when `merge_evidence.conviction` changes from
    `off` to `auto` at a config reload; the lane writes that state row.
- With `false_failure` at its starting value nothing reaches 0.9 until the defect rate is above 0.19; at 0.14 the
  estimate must fall to 0.018, at 0.08 to 0.0097. The snapshot shows the highest probability among waiting
  requests beside the threshold, so an idle conviction is visible as such.
- `merge_evidence.conviction: off` stops convictions and shadow rows.
- **C1.** No request is convicted while it is merging, reserved, parked or being verified.
- **C2.** No content key has two `merge_conviction` rows with `mode: on`.
- **C3.** Every `merge_conviction` row names red observations that exist, are not void, were recorded with
  `root_proved`, and in which the key is the only one left.

### 6.7 A red main

- The row `_run_deferred_main_health_probe` writes (`merge_attempt`, `outcome: main_health_red`) gains
  `main_sha`. At that site the lane calls `EvidenceLedger.declare_main_red(main_sha)`.
- The ledger voids every red observation whose root is `main_sha` or whose tree holds `main_sha` as a member's
  merge commit (D11). A red recorded later on that root is voided as it is recorded. The declaration is a
  `merge_evidence_void` row naming the root and the observations.
- A conviction whose observation is voided afterwards is reported by ε. It is not withdrawn.
- **M1.** A void observation contributes to no probability, to no rate and to no truth.

### 6.8 Restart

- `EventStore.fetch_events_after(event_types, after_id, limit)` returns the rows after a row id, oldest first, and
  raises on failure.
- Every `checkpoint_every` recorded observations, and at every conviction state change, the ledger writes a
  `merge_evidence_checkpoint` row in the same no-await step as `record`: its groups, cleared keys, root facts
  (proved, declared red), rate tallies, convicted keys and owners that must dispatch alone. Every observation row
  older than a checkpoint row is in it; every newer one is in the tail.
- `SpeculativeMergeWorker.run` rebuilds before the first pick: the newest checkpoint, the ledger's rows after it,
  and the recovered requests as `live`. Keys of requests that are no longer live are treated as gone.
- Every start writes a `merge_evidence_rebuild` row: `ok`, checkpoint age, rows applied, rows skipped.
- If the read raises, or the checkpoint cannot be decoded, or more than 5 tail rows are skipped: the ledger starts
  empty at the starting rates, the row says `ok: false` with the reason, one L1 is filed, and a fresh checkpoint
  is written at once (D14). Conviction state is still read from its own rows; if that read fails it is `SHADOW`.
- Deferrals are recomputed from what is in flight after the restart; their clocks are not kept.
- **S1.** The same checkpoint, tail and live set give the same posterior.

### 6.9 Config, events, snapshot

- `merge_evidence` (own file `orchestrator/merge_evidence_config.py`, mounted once, all leaves hot, each with a
  range validator): `conviction` (`auto` | `off`), `convict_threshold` (0.9), `max_fresh` (12), `max_total` (24),
  `max_states` (65,536), `sync_states` (4,096), `checkpoint_every` (50), `hold_limit_factor` (2.0),
  `start_defect_rate` (0.14), `start_false_failure` (0.026), `start_main_red` (0.06).
- New event types, all declared by β1a in `EventType` wherever it lives at dispatch: `merge_observation` (α2),
  `merge_evidence_checkpoint`, `merge_evidence_rebuild`, `merge_evidence_void`, `merge_conviction`,
  `merge_conviction_truth`, `merge_conviction_state`.
- `SpeculativeMergeWorker.snapshot()`: each in-flight entry gains `tree` and `tree_root` (α1); each waiting entry
  gains `p_defective`, `implicated_by`, `deferred_for` and `deferred_since`; the top level gains
  `evidence: {defect_rate, false_failure, main_red, n, conviction_state, highest_p, rebuilt_ok}`.

## 7. Boundary-test sketch (B + H)

Driven through the lane's public seam with tree-keyed verdicts (requests in, events and outcomes out).

| # | Scenario | Preconditions | Asserts |
|---|---|---|---|
| 1 | The tree is recorded | a batch of 3 is built on an in-flight entry of 2; both return | the batch's `merge_observation` row lists 5 members, 3 of them `own`, each with merge commit, tip and content id, and the entry's root |
| 2 | Base cases | an item built on a landed commit that is not main's tip; an item built on a commit no live entry owns | the first row has no built-on members and that commit as root; the second has `evidence: none`, `base_unresolved` |
| 3 | Identity across the lane's rebases | a waiting request is unstacked and another is bounced onto the frozen tip, neither across a change to its hunks' context | each keeps its content id; a branch that gains a commit gets a new one; a branch that merged main into itself has none |
| 4 | No evidence | a verify times out; one is skipped for low disk; one passes trivially; one fails for lock contention | each row has `evidence: none` with its reason; no probability changes; nothing is cleared; no root is proved |
| 5 | The verdict that counts | a verify's remote attempt passes and its cross-check fails | one observation, `RED` |
| 6 | A raising batch does not re-form | a batch's verify raises | the owner's next dispatch has no members |
| 7 | Red on main | a batch of 3 on a proved root returns red | each member's `p_defective` is the model's value; the same batch is not built again; one of the three is dispatched and the other two are deferred for it |
| 8 | Red on work that then goes red | a batch is built on an unverified entry; the batch returns red, then the entry | after the second verdict the batch's members are at the prior and are batched |
| 9 | Red on work that then passes | as 8, the entry passes | the batch's members have the probabilities of row 7 |
| 10 | Culprit blocked | a batch of 3 is red; one member's own verify is red at the head | it is blocked; with it gone from the lane, the other two are within 0.02 of the prior and join the next batch |
| 11 | The table is a special case | one probability for every request, follow-up cost forced to "each member alone" | the batch has the depth task 10920's table gives, at estimates inside each row |
| 12 | Pinning a size | `start_defect_rate: 0`, `max_fresh: 4`, six requests waiting, no observations | the batch has 4 members |
| 13 | Cleared requests ride | two requests passed in a tree that was torn down before landing; 12 fresh requests wait | the next batch holds 12 fresh members and both cleared ones |
| 14 | Sub-batch when cheaper | a batch of 8 is red on a proved root at a defect rate of 0.05 | the next dispatch of its members holds more than one of them |
| 15 | Deferral is per test | a member of a red set is being verified; another reaches the front | the second is passed over; it is picked when the first returns; a third member is then deferred for the second with a new clock |
| 16 | Picking ahead | both hosts are busy; the only waiting requests are deferred | the pick returns nothing; when a host frees with nothing built for it, the first deferred request is dispatched alone |
| 17 | High lane | a high-lane request shares a group with a verify in flight | it is picked |
| 18 | Conviction | state `ON`; a batch of 3 is red on a proved root; two members pass alone and land | the third, waiting, is resolved `blocked` with `ConvictionEvidence` naming the batch, the two cleared tasks, the failing tests and the log directory; its task is blocked under the conviction category; the thrash counter does not advance |
| 19 | Shadow | as 18 in `SHADOW` | a `merge_conviction` row with `mode: shadow`; the third is verified; a `merge_conviction_truth` row follows its verdict |
| 20 | Not the sole one left | a red-set member is batched with fresh requests and that batch is red too | nobody is convicted, whatever the probabilities |
| 21 | Never in flight | as 18, the third has been picked when the second lands | it is not convicted |
| 22 | Reach | as 18, the third was submitted by hand; or has no content id; or was recovered after a restart | by hand or with no id: not convicted, verified alone; recovered: convicted, and an L1 is filed under the conviction category |
| 23 | Convicted content returns | a convicted request is re-submitted unchanged and passes alone | no second conviction; a truth row `sound` |
| 24 | A red main healed through the lane | an outside commit moves main to M; a batch of 3 is red on M; a fix passes on M and lands; two of the three then pass | the third is not convicted and is verified alone |
| 25 | Red main, declared | two unrelated singles are red on root M; the deferred probe confirms M red | both observations are void, the first included; neither request's probability is raised by them; a later red on M is void as recorded |
| 26 | A request leaves without a verdict | a batch of 3 is red; one member is cancelled; a second passes | the third is not convicted |
| 27 | Restart | observations, a cleared member, a declared-red root and a shadow conviction are recorded; the orchestrator restarts with the same requests waiting | the snapshot shows the same probabilities and conviction state; a `merge_evidence_rebuild` row with `ok: true` |
| 28 | Rebuild fails | the checkpoint row cannot be decoded; the read raises | in each case `ok: false`, one L1, an empty record, batching continues, and the next start rebuilds from the fresh checkpoint |
| 29 | Switching | 10 shadow truths with 1 `sound`; then 2 of 20 `on` convictions return unchanged and pass; then 20 clean shadow truths; then 2 of 20 again | `SHADOW → ON → SUSPENDED → ON → OFF`, a state row and at most one L1 for each; `off` then `auto` by reload gives `SHADOW` |
| 30 | Too many states | overlapping red trees exceed `max_states` | none of their requests is convicted; `plan.not_enumerated` is set |
| 31 | Coalescing | a deferred request is eligible for coalescing | it is not coalesced |
| 32 | Kill switch | `merge_deep.chain_cap: 0` | every request is dispatched alone; observations are still recorded |

## 8. Decomposition plan

Greek labels are this plan's; ids are assigned at decompose. "Signal" is what an operator observes.

| Label | Task | Touches | Signal or what it unlocks | Needs |
|---|---|---|---|---|
| **α1** | Identity and the tested tree | `merge_lane/landing_evidence.py`, `git_ops.py`, `suffix_graph.py`, new `merge_lane/tested_tree.py`, `types.py`, `verify_dispatch.py`, `dispatch_ports.py`, `worker.py`, `orchestrator/tests/_merge_lane_fakes.py` (the tree-keyed remote moves here) | Rows 2 (first half), 3. `get_merge_queue` shows each in-flight entry's `tree` and `tree_root`. Unlocks α2, task 10995 | — |
| **α2** | The lane records an observation for every verdict | new `shared/src/shared/merge_evidence.py`, new `merge_lane/observation.py`; `post_merge_verify.py`, `verify_failure_outcome.py`, `inflight_verify.py`, `dispatch_ports.py`, `worker.py`, `event_store.py` | On the live lane every verdict of a dispatched entry writes a `merge_observation` row with its tree, root and evidence class. Rows 1, 2, 4, 5. Unlocks β1a, ε | α1 |
| **β1a** | The ledger and inference, deciding nothing: roots, rates, checkpoint and rebuild, probabilities in the snapshot | new `merge_lane/evidence_ledger.py`, `merge_lane/evidence_inference.py`, `orchestrator/merge_evidence_config.py`; `config.py`, `event_store.py`, `worker.py`, `inflight_verify.py`; a restart scene helper in `_merge_lane_fakes.py` | `get_merge_queue` shows `p_defective` and `implicated_by` for each waiting request and the three rates with their n; `merge_evidence_checkpoint` and `merge_evidence_rebuild` rows. Rows 27, 28, 30 (probabilities). Unlocks β1b, γ, ε | α2 |
| **β1b** | The rule as pure functions, in shadow | new `merge_lane/batch_rule.py`; `worker.py`, `verify_dispatch.py`, `merge_lane/observation.py` | Every `merge_observation` row carries `predicted_pass` for the tree that was dispatched and the `plan` the rule would have made. Rows 11, 12 (as function tests). Unlocks β2a, η | β1a |
| **β2a** | The rule sizes and fills batches; `chain_policy.py` and `depth_table` are removed | `worker.py`, `verify_dispatch.py`, `merge_verify_facts.py`, `chain_policy.py` (deleted), `config.py`, `defaults.yaml`, `orchestrator/tests/_merge_deep_scene.py` and the deep tests' size pin, the table's tests, `OPERATIONS.md`, `skills/orchestrate/SKILL.md`, `plans/merge-lane-policy-prd.md` (D7) | Rows 7 (not rebuilt), 12 to 14, 32. On the live lane `chain_depth_reason.basis` is `evidence_rule` | β1b |
| **β2b** | Deferral at the pick, the no-evidence guard, coalescing; the mark is removed | `intake_buffers.py`, `worker.py`, `inflight_verify.py`, `dispatch_ports.py`, `merge_lane/batch_rule.py`, the mark's tests in the deep test files, `plans/merge-lane-policy-prd.md` (D6) | Rows 6 to 10, 15 to 17, 31. On the live lane the snapshot shows `deferred_for`, and no dispatched batch equals the red batch before it | β2a |
| **γ** | A declared-red main voids observations | `merge_lane/main_health_probe.py`, `evidence_ledger.py`, `worker.py` | The main-red row carries `main_sha`; a confirmed red main writes a `merge_evidence_void` row. Rows 24 (probabilities), 25 | β2b |
| **δ** | Conviction | new `merge_lane/conviction.py`; `merge_lane/outcome_facts.py`, `escalation/src/escalation/categories.py`, `types.py`, `worker.py`, `intake_buffers.py`, `workflow.py`, `evidence_ledger.py` | Rows 18 to 24, 26, 29. On the live lane: `merge_conviction` rows with `mode: shadow`, their truth rows, and `evidence.conviction_state` and `highest_p` in the snapshot, whether or not any request reaches the threshold | γ |
| **ε** | The readout | new `scripts/merge_evidence_readout.py` and its tests | Run on the live store it prints: verifies per landing by day; predicted pass against verdict in ten bins, entries on main and stacked entries apart; the three rates with n; the rule's shadow plan against what was dispatched; deferral times from `plan.deferred`; shadow convictions against their truths and the share of `on` convictions that returned unchanged; unexplained reds; voided reds and blocks delivered before the void; how often a waiting request's content id changed with no new submission. With no rows of a kind it prints `no <kind> rows yet`. It also prints the figures section 9 names | β1a |
| **ζ** | Integration gate | a boundary-test module over the real lane modules | Section 7 rows 1 to 32 against the real modules, and ε's output from a lane that has run β2b and δ | δ, ε |
| **η** | Design gate for phase 2 (a decision task; no code) | — | It escalates to Leo with ε's phase-2 figures and section 9 attached | β1b, ε; tasks 10721, 10995, 11002; a 7-day delay after they are met |
| **θ** | Design gate: revise `plans/merge-chain-visibility-prd.md` for the evidence ledger (a decision task; no code) | — | It escalates with this PRD's 6.4 and the list of that PRD's decisions, contract lines, boundary rows and delivered checks that name the mark (`evidence/REPORT.md` section 6). Tasks 10759, 10761 and 10762 wait for it | β2b |

Companion corrections, made by the decomposing session with Leo's word, not by a task:

- Dependencies: 5041 and 9935 on β2b; 10995 on α1; 10759, 10761 and 10762 on θ in place of 10624.
- Cancelled: 10624, 8932.
- Dated notes: 7824, 7921, 9144, 9182, 9207, 9852, 10217, 10721, 10996.
- A dated delta line in `plans/priority-at-later-gates-prd.md` 6.4: no fronted suspect list exists.

Notes for decompose:

- α1, α2, β1a, β1b, β2a, β2b, γ, δ are one chain: each edits `worker.py`. ε is parallel after β1a.
- Every leaf that touches the merge-lane cluster records its raise with
  `scripts/merge_lane_metrics.py --write-baseline ... --authorize-raise <task id>`.
- Nothing here waits for tasks 9935 or 8532. Gate 10069's own edges to them are removed when it closes.
- The PRD's commit carries `evidence/REPORT.md`, the measurement and forensic scripts, and the simulator.
- Signal leaves: ζ, η, θ.

## 9. Phase 2: a design gate (L11)

Not designed here. η fires a week after phase 1's shadow rows exist and tasks 10721, 10995 and 11002 have landed.
Its subject, with what the simulation measured over this PRD's design:

| Change | Measured | Open |
|---|---|---|
| A failed entry tears down only the entries built on it | needed for the rows below to reach their full value | the cascade's code; task 7358 |
| A passed entry that cannot land is dissolved; its members ride the next batch as cleared | with the next row, +8 to +17% under today's cascade | it replaces task 10721's re-verify; a pass is trusted across a moved main (0 of 169 re-verifies of a passed request failed) |
| The rule also chooses what an entry is built on | hand-tuned: +30 to +36% with the two rows above; the rule's own formula: +18 to +29%, crossing over near 0.7 without tuning | finish the formula; it earns nothing under today's cascade because it prices a torn-down entry as a total loss |
| The failing test as evidence about which member is the culprit | as a probability shift alone: +14% at 80% accuracy and a defect rate of 0.14, +40% at 0.05 | how a failing test names a request |
| A going cost that credits release when the culprit is blocked | the rule trails the hand-built variant by 4.6% at a defect rate of 0.03 | two refinements tried were unstable |
| Context-free content ids | — | whether ε shows identity lost to context-line changes often enough to matter |

ε prints what the gate needs from real rows: how often a request that passed fails when re-verified on a moved
main; predicted against actual "everything ahead is sound"; the share of reds on unlanded work; how often the
failing tests of a red batch are a subset of one member's later solo failure.

## 10. Out of scope

- The order requests are taken in (part 1). The scheduler.
- What an entry is built on (task 11002; phase 2). The head-failure cascade.
- How verify capacity is shared with task verifies (gate 10572).
- Detecting a red main (the probe; task 7921). Holding requests while main is red (gate 10577). A request blocked
  on its own red verify while main is red stays that probe's to prevent.
- How a passed batch lands; the post-landing type check (task 10998).
- Per-request priors from task features. The other projects' adoption beyond the shared code.
- Restoring lane state other than the ledger's across a restart (`evidence/REPORT.md` section 5 lists each item
  with the task that owns it).

## 11. Open questions (tactical)

1. **Where the content id is memoised.** Suggested: on `GitOps`, keyed by `(tip, base)`. Decide in α1.
2. **Whether the checkpoint is one row or one row per group.** Suggested: one row. Decide in β1a.
3. **When the going-cost table is rebuilt as `false_failure` moves.** Suggested: lazily, when it has moved by more
   than a tenth of its value since the table was built. Decide in β1b.
4. **Whether ε's bins are equal-width or equal-count.** Suggested: equal-width, n printed per bin. Decide in ε.
