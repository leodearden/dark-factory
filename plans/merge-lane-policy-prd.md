# Merge-lane dispatch and speculation policy

**Status:** authored 2026-11-05 under human gate 5098; decompose pending Leo's veto window on decision D11.
**Slug:** `merge-lane-policy` · **Approach:** B + H (the merge lane is a load-bearing seam; blast radius
`merge_lane/worker.py`, `merge_lane/speculation_controller.py`, `merge_lane/gates.py`, `harness.py`,
`hooks/project-checks`, `verify.py`).
**Origin:** `plans/portfolio-review-2026-11-01/followups/merge-throughput.md` (six evidence seats, a fable
critic, two delta passes; every figure there carries its counting rule) and Leo's rulings of 2026-11-04/05
recorded in §3. This is the "speculation/dispatch-policy follow-up PRD" that
`plans/merge-lane-throughput-prd.md` (§Companion PRD) and `plans/merge-lane-quality-prd.md` (§Cross-PRD,
row "speculation/dispatch-policy follow-up") defer to; it discharges gate 5098 and closes 5056 (its
measurement is the report).
**Code anchors** verified against main `72239dfd5d` (2026-11-05). Cited by symbol; re-locate at
implementation time.

> **Amended 2026-11-10** (Leo): ε no longer waits for tasks 7721 and 7824; both now land after it. See §8,
> the dated note under ε.

## 1. Goal

Stop the merge lane from spending two verifies on every landing. Today (post-coalesce window
11-02T00Z..11-04T14Z, 63 landings) the lane runs 2.59 merge verifies per landing: 1.00 that lands, 0.84 that
PASSED and were refused by the reverify gate, 0.65 that failed (35 of 41 inside one red-main episode), 0.09
voided or other. The 0.84 is one loop: a stacked item whose base died is re-merged on main non-speculatively
by an adjudicated carve-out, the merger is never told, two non-speculative verifies race on the same main,
the loser is refused under `merge_verify_breadth: full`, the refusal re-verifies the head and kills its
successors' base again. A non-speculative first dispatch made beside an in-flight verify never landed directly
(0 of 45). Meanwhile the laptop, the fast host, holds a passed verdict for a third of heartbeat hours waiting
in order behind the local head.

What an operator sees when this PRD lands:

1. `merge_reverify_gate` rows per `merge_finalized done` fall from 0.84 to below 0.3 over a 7-day window,
   and P(gate at landing i+1 | gate at landing i) falls from 0.79 to below 0.3.
2. `verdict_voided` rows per landing fall from 0.78 (82/109 dequeues) to below 0.1; a `verdict_voided` row
   names the item's own commit and the killer's commit.
3. Every `merge_verify` row carries `runner_reason` (why that host) and `base_relation`
   (`main | stacked | rebased_head`); a cancelled verify leaves a `merge_verify_cancelled` row.
4. `merge_heartbeat.occupancy.leases[].activity == 'finished'` on the laptop covers under 10% of heartbeat
   hours (from 33%).
5. Median last-`merge_queued`-to-`merge_finalized done` falls from 5.4 h to under 1 h; local merge-verify
   minutes per week fall from about 100 h to under 60 h.
6. `merge_verify.chain_items > 1` appears on ordinary dispatches, including the head, with
   `chain_depth_reason` naming the pass-rate estimate that chose the depth; a tip failure is followed by a
   half-chain verify, not a full-depth retry.
7. A direct commit to main that would turn a whole-tree guard or a module's type check red is refused by
   `hooks/project-checks` before it lands; a main-tip sweep red of the `type_error` class arms the 7921 hold.

## 2. Background and premise validity (G6)

Figures from the report (rules there; windows: 7d = 10-28T14Z..11-04T14Z, post-coalesce as above):

- **The merger already stacks.** `speculative_merge` on 100 of 109 `merge_dequeued` post-coalesce. The
  dispatch-time INV-3 re-check (`worker.py::_dispatch_item` → `_chain_dead_link`) voided 82 and
  `worker.py::_remerge_inner` re-merged them with `base_sha=actual_main, speculative=False`. 73 of 82 voids
  inherit a dead link that is itself a voided stacked build; voids run in 8 runs of length 1–22 and end only
  at a conflict (4) or a near-drain (1). [seats A, B]
- **The gate loop.** 53 refusals: 42 rebased onto the queue's own landing from the other slot, 11 onto a
  direct commit. P(gate|gate) 0.79 vs 0.47. 70% of landings came through a gate re-verify. [B]
- **Failure classes.** Outside the two direct-commit episodes the lane passed 122/128 (95%); per-item reds
  4–6 of 163; transient 0. The laptop's "66% vs local 90%" is where episode reds land (96% vs 95% ex-episode).
  [B §3]
- **Lease hold.** Laptop held a passed verdict 18.1 of 54.9 heartbeat hours; K = 1 + runners = 2;
  `worker.py::_gate_reverify_on_head_host` keeps the lease; `harness.py::Harness._speculation_k`. No local
  verify started while the laptop was free (0 of 58). The laptop per-root flock is inert for df
  (`git.persistent_merge_worktree: false` on the laptop yaml). [C]
- **Model.** Calibrated simulator (`merge-throughput-seats/scratch/F-sim.py`): always-stack with successors
  re-anchored on a gate-rebased head is ×1.7 over today's lane ceiling (sim 331 vs 193; on measured ex-red
  ~370–430/week); K = hosts + 1 with lease release reaches the two-host host-saturation bound (~526/week
  ex-episode); head chaining plus bisect retry at the pass-rate-chosen depth is a ~3× reserve and cuts local
  merge CPU from 0.84 to 0.14 h/landing; best first depth is 8 at per-item defect q 0.05 and 4 at q 0.14;
  bisect-and-keep-the-passing-prefix beats the coded halving by 14–30%; break-even for main dispatch is
  P(head fails) ≈ 0.7 (full gate), unreachable at df's 0.05. [F]
- **Realised yield is bounded by arrivals** (~196 first-queued tasks/week): the lane fix buys latency,
  headroom against episodes, ~45–100 local CPU hours/week and the task-concurrency permit held across the
  merge wait (344 of 1,488 slot-hours, 23%); landings rise only as the task side converts those. The
  acceptance thresholds in §1 are therefore lane-internal ratios and latencies, not landings. [critic]
- **Episodes are direct commits.** E5 (manifest), 11-02 pyright (`83046d186c`, a `scripts/*.py` commit:
  `hooks/project-checks::PYRIGHT_PACKAGES` does not include `scripts`, the merge gate's `module:scripts:type`
  does), M2 (`eb84ac9f41`, docs data files read by `tests/scripts/test_reify_closure_staleness_sweep_retired`).
  One episode cost 35 verifies and 10.2 lane-hours. [B, D, E, critic]

Numeric premises in §1 and their basis: the 0.3 gate threshold sits above the model's post-fix 0.1–0.15 and
far below the measured 0.84; the 0.1 void threshold above the model's 0.10–0.12 (P7+b) and below the measured
0.78; the 10% laptop-finished share is one third of the measured 33% and above the model's implied residual;
the 1 h latency is twice the model's 23–32 min at today's demand; 60 local h/week is between the model's 45
and the measured ~100. Each is provisional and re-set by leaf ι's readout, never tightened past the model.

## 3. Resolved design decisions

Leo's rulings (2026-11-04/05): reverse the §5.3 carve-out (D1); straight to K = hosts + 1 in code, no
serial-laptop interim (D4); the rolling pass-rate rule sets depth with no fixed cap, `chain_cap` stays as the
kill switch only, 3191 retired (D7); the pre-commit hook change goes (D8); close 5056 with the report (D13);
drop the family-green disjoint skip (D14); widen 8411 to type-error verdicts (D9).

**D1. Reverse the two-layer PRD §5.3 re-merge carve-out (task 3206).** Recovery re-merges no longer target
real main. Its three recorded grounds and how this PRD answers them:
1. *Circularity* (the frozen tip can itself be the dead or phantom commit): a re-anchor base must be a
   **live tip**: `worker.py::frozen_prefix_tip(main)` ≠ main, not in `_dead_base_commits`, and owned by an
   `InflightEntry` in the verify frontier that is not `vacated` (`merge_lane/types.py::InflightEntry.vacated`)
   and whose verify task is alive. When no live tip exists the re-merge targets main, as today. The 3082
   phantom-head class fails the "verify task is alive" test and falls to main; the §5.3 advisory guard stays
   advisory (its own decision is untouched).
2. *Permit accounting* (`speculative=True` ⇔ exactly one SpecPermit held and released once): a re-anchored
   replacement is `speculative=True` and must hold a SpecPermit. The dead-base and Mechanism-2 sites already
   carry the original item's permit (`item_permit`) and hand it to the replacement; the cascade,
   adoption-void and RUNNER_UNAVAILABLE sites release the permit before `_remerge` today and will instead
   transfer it to the replacement; when no permit is held (a non-speculative original), the replacement takes
   a non-blocking slot (`PermitLedger.slot_available() > 0`) or, if none, re-merges on main non-speculatively.
   The verifier coroutine never blocks on `acquire`. `speculation_accounting_violations` keeps its audit.
3. *Recovery semantics + INV-3 backstop*: recovery means "a base that can still land", not "main". INV-3's
   adoption-time void stays as the backstop for a base that dies after the re-anchor. The carve-out bullet in
   `plans/two-layer-merge-queue-prd.md` §5.3 is amended by this PRD's companion correction (leaf β) to point
   here; the five in-code cross-references follow.

**D2. The merger is re-anchored too.** `SpeculationController.on_base_invalidated(dead_sha, live_tip)` is
called from every site that records a base dead (`_record_dead_base`, `_record_item_commit_dead`) so
`spec_base`/`pending_spec_base` are re-pointed at the live tip (or cleared). This covers the ≤ 2 items the
merger has merged but the verifier has not yet seen; the dispatch-side re-anchor (D1) carries the rest.

**D3. A gate re-verify is a dispatch, and successors re-anchor on the rebased head.** Today
`worker.py::_land_rebased_or_park_gate` records the head's commit dead and runs
`gates.py::_reverify_rebased_tree` on the head's lease with no item, so successors are voided to main. Now
the rebased candidate becomes a `RealMergeItem` (`base_sha = rebased_onto`, merge commit = `rebased_sha`,
`speculative=False`, `remerge_recovery` cleared) dispatched through `_dispatch_item` onto the fastest free
host; its successors are re-anchored on `rebased_sha` by D1/D2 instead of voided. The gate decision row
(`reverify_gate_decision.py`) is still written before the verify. Because the rebased head is itself an
ordinary dispatch it may chain (D5). The 8329 class (void at adoption when the head was gate-rebased) is
subsumed: the successor is re-anchored, not voided.

**D4. K = hosts + 1 and the host lease is released at verdict.** `harness.py::Harness._speculation_k`
returns `1 + enabled runners + 1`. `_finalize_inflight` → `_resolve_and_release` releases the host lease as
soon as the verdict is in, while the `InflightEntry` stays in the frontier awaiting in-order landing. A gate
re-verify (D3) or a landing probe re-acquires a host through `HostAllocator.acquire` with the configured
policy rather than holding one. Landing itself (`finalizing`, ~3 min) does not need a verify host. The
serial-laptop policy value is NOT added (Leo: straight to K = hosts + 1).

**D5. The head chains when nothing else is in the frontier.** `_deep_chain_placement` guard 1 becomes
"not speculative AND another entry is in the verify frontier → None". `_adopt_head_on_tip_authority` tolerates
the chained item being the head (it currently excludes the tip's own entry and cancels `_inflight[0]`).

**D6. Retry by bisect with a suspect list.** `_chain_halving_state: int | None` becomes a suspect list
(ordered task ids of the links of the last failed tip). On a tip fail at built depth k: requeue the k links at
the FRONT in aging-key order, suspects := those links, cancel stacked successors and requeue their items
behind the links. The next dispatch, while suspects exist, verifies the first ⌈|suspects|/2⌉ suspects as a
chain; a passing half lands and is removed; a failing half becomes the new suspect set; a lone failing suspect
is blocked with its findings. Full depth resumes only when no suspects remain. `next_halving_state` is retired.

**D7. Depth from a rolling pass-rate estimate, no fixed cap.** p̂ = per-item pass estimate from the last 30
FRESH tip verdicts (bisect rounds excluded), p̂ = pass_rate^(1/mean k). Depth table: 8 if p̂ ≥ 0.93, 4 if
≥ 0.85, 2 if ≥ 0.75, else 1; capped only by queue depth. The table is a config list
(`merge_deep.depth_table: [[0.93, 8], [0.85, 4], [0.75, 2]]`, green-tier) so the readout (leaf ι) can extend
it past 8 once the per-item chain wall is measured at depth ≥ 8 (today measured only at k ≤ 2). The first
row's depth may exceed `chain_cap`: `chain_cap` becomes the kill switch only (0 = off; any positive value =
on), documented as such; 3191 ("promote chain_cap 6 → 32") is cancelled by decompose, 3190's canary
predicate stays (it reads gates per landing and chain-attributed reds, both still meaningful).
`chain_depth_reason` on `merge_verify` records p̂, the row chosen, the queue depth and the suspect state.

**D8. Direct commits meet the merge gate's checks before they land.** `hooks/project-checks` (main-only
tier) gains two legs: (a) for every staged file, the per-module `lint_command` and `type_check` of the module
the file maps to, taken from the same registry the merge gate reads
(`verify_plan.py::effective_merge_module_configs`), so a staged `scripts/*.py` runs `module:scripts:type`
exactly as the gate would (this closes the 11-02 class: `PYRIGHT_PACKAGES` is retired in favour of the
registry, and `tests/scripts/test_hook_pyright_packages_drift.py` pins the new derivation); (b) for a staged
file under `plans/`, `docs/`, `review/`, or any `*.yaml`/`*.json`/`*.md` outside a package, the whole-tree
screen list from `scripts/merge_screen.yaml` (the pre-verify screen PRD's fast seven, Leo 2026-10-29). If
that manifest does not yet exist when leaf η lands, η creates it in the shape `merge-pre-verify-screen-prd.md`
§3 decision 3 specifies and 7919 consumes it (seam owner: this PRD creates, 7919 consumes; recorded in §5).
Cost bound: ≤ 45 s per direct commit (the fast seven measure 9.8 s warm / 44.7 s cold; a module pyright is
seconds). `--no-verify` remains possible; `hooks/reference-transaction` is unchanged.

**D9. The main-red hold arms on type-error verdicts.** 8411 (pending high, depends on 7921) is amended at
decompose: its failing-id set F is `junit failing test ids ∪ {module:<m>:type}` for a `type_error` category
verdict, and the F-screen for a type unit is that module's `type_check` command. 7921's own design is
unchanged; 8411 carries the widening.

**D10. Instrumentation first.** Before any policy leaf: `merge_verify.runner_reason` (`prefer_remote_free |
remote_busy | remote_quarantined | local_only_rule | gate_on_head_host | retry_held_host`),
`merge_verify.base_relation`, `merge_verify_cancelled` (emitted by `_teardown_verify_task` with the elapsed
fraction and the cause), `verdict_voided.item_commit` and `.killer` (the entry or event that recorded the dead
link), and `merge_finalized.queue_to_land_secs` (from the last `merge_queued`). The dashboard's merge view and
`scripts/merge_lane_throughput.py` (task 5050) read them. These are the signals §1 and leaf ι measure.

**D11. Land in `merge_lane/worker.py` now; the κ/λ extraction moves the as-built code.** The quality PRD
(`merge-lane-quality-prd.md` §Cross-PRD) planned this PRD's feature tasks to depend on κ 5040
(`verify_dispatch.py`) and λ 5041 (`speculation.py`), both pending high behind a refactor chain with no
events. The lane is the binding constraint today, so this PRD's leaves land in `worker.py` serialized by the
same-file rule, and κ/λ move them unchanged, the same treatment the quality PRD already gives the deep-merge
code. Leaves β–ζ add `add_dependency` edges to each other only; κ/λ gain nothing new to wait for. **Leo may
veto this before decompose** (the alternative is β depends on 5041).

**D12. Canary predicates per leaf** (leaf ι, deterministic, delayed 7 d after the deploy that carries the
last of β–ζ): gates per landing < 0.3; voids per landing < 0.1; P(gate|gate) < 0.3; laptop `finished` share
< 10%; `verify_base_frozen_tip_violations` of the new class (re-anchor onto a vacated or dead tip) = 0;
`speculation_accounting_violations` = 0; no chain-attributed red main; median queue-to-land < 1 h. Any
predicate false → `chain_cap: 0` (hot) and an L2 escalation naming the predicate; the policy leaves are not
reverted by the predicate (reverting is the human's call).

**D13. 5056 closes with the report as its measurement; 5098 closes when this PRD commits.** 5059/7860 (the
chain assessment and the trains human gate) stay; 7860 reads leaf ι's readout.

**D14. Dropped:** the family-green disjoint skip (6943 stays as filed; it adds nothing once stacking lands,
and direct-commit gates stay full by P2); trains and coalescing (stay off per 7739); a second df laptop
slot; queue re-ordering (the unchanged-tip hold is worth 0.08 verifies/landing and is 7921's class).

## 4. Pre-conditions (G3)

Every symbol this PRD names exists on main `72239dfd5d` and is wired into the production path:
`worker.py::_dispatch_item`, `_chain_dead_link`, `_remerge`, `_remerge_inner`, `_record_dead_base`,
`_record_item_commit_dead`, `frozen_prefix_tip`, `_dead_base_commits`, `_deep_chain_placement`,
`select_chain_depth`, `next_halving_state`, `_note_chain_outcome`, `_chain_halving_state`,
`_adopt_head_on_tip_authority`, `_land_rebased_or_park_gate`, `_gate_reverify_on_head_host`,
`_finalize_inflight`, `_resolve_and_release`, `_teardown_verify_task`; `speculation_controller.py::
SpeculationController` (`spec_base`, `pending_spec_base`, `on_lookahead_found`, `on_dequeue`), `PermitLedger.
slot_available`; `types.py::InflightEntry.vacated`; `gates.py::_reverify_rebased_tree`,
`_disjoint_skip_blockers`; `reverify_gate_decision.py::emit_reverify_gate_decision`;
`harness.py::Harness._speculation_k`; `verify_runner.py::HostAllocator.acquire`;
`verify_plan.py::effective_merge_module_configs`; `verify.py::run_main_tip_sweep`; `config.py::MergeDeepConfig`
(`chain_cap`, `chain_build_timeout_secs`); `hooks/project-checks` (`PYRIGHT_PACKAGES`, the staged-file
tiers). Novel substrate introduced here, each owned by a leaf: `on_base_invalidated` (β),
`merge_deep.depth_table` (ζ), the suspect-list state (ζ), the gate-rebased `RealMergeItem` path (γ), the new
event fields (α), `scripts/merge_screen.yaml` if absent (η). Nothing assumes κ/λ (D11).

Prerequisite tasks: 7720 (chain links are popped by the look-ahead during a tip verify) before ε/ζ. Tasks 7721
(chain-build lane leak) and 7824 (build the chain off the verifier loop) were prerequisites of ε until
2026-11-10 and now land after it (§8, note under ε). 7918 → 7919 → 7921 before
8411's widening (θ) is testable. 7829 (deploy once the chain fixes land) is superseded by ι's deploy.

## 5. Cross-PRD relationships and seam owners (G4)

| Other PRD | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `plans/two-layer-merge-queue-prd.md` §5.3 | amends | the re-merge carve-out bullet (task 3206) and its five in-code cross-references | this PRD (leaf β carries the prose amendment in the same change) | queued |
| `plans/deep-merge-ahead-prd.md` (3183–3192, 7720–7829) | consumes / supersedes | `select_chain_depth`, `next_halving_state`, `_deep_chain_placement` guard 1, `chain_cap` semantics, η2 3191 | this PRD owns the policy; deep-merge PRD owns link reservation (7720), which ε/ζ depend on, and the lane-leak and off-loop-build fixes (7721, 7824), which follow ε since 2026-11-10; 3191 cancelled; 3190 unchanged | queued |
| `plans/merge-lane-quality-prd.md` κ 5040 / λ 5041 | will refactor | `verify_dispatch.py`, `speculation.py` extraction | quality PRD owns the extraction and moves this PRD's as-built code unchanged (D11) | wired (no new edges) |
| `plans/merge-pre-verify-screen-prd.md` 7918–7924 | consumes / produces | `scripts/merge_screen.yaml` (η creates if absent, 7919 consumes); 7921 hold + 8411 widening (θ amends 8411) | screen PRD owns `run_node_ids`, `screen.py`, 7921; this PRD owns the hook leg and the type-error widening | queued |
| `plans/merge-lane-throughput-prd.md` | discharges | gate 5098, measurement 5056, `scripts/merge_lane_throughput.py` reads α's fields | this PRD | queued |
| `plans/remote-task-leg-verify-prd.md` (7977 → 6670 → 6672 → 6673) | shares the host | laptop CPU; `HostAllocator` | that PRD; no arbitration added here (out of scope) | — |
| `plans/concurrent-merge-verify-prd.md`, `plans/merge-throughput-multihost-verify-prd.md` | consumes | `HostAllocator`, `_speculation_k`, host lease lifecycle | this PRD changes K and lease release (δ); the pool is unchanged | queued |
| docs lane (5411 / 8258 / 8260) | parallel | direct-commit rate | that work; D8 guards the commits that remain | — |

## 6. Contract (B + H)

**Re-anchor decision** (`worker.py`, pure function, unit-testable):
```
def choose_remerge_base(main_sha, frozen_tip, dead_bases, frontier) -> RemergeBase:
    # RemergeBase(base_sha, speculative: bool, reason: 'live_tip' | 'no_tip' | 'tip_dead' | 'tip_vacated' | 'tip_verify_dead' | 'no_permit')
```
Invariants: `speculative is True` ⇔ `base_sha != main_sha` ⇔ a SpecPermit is attached to the replacement
item before it is enqueued; `base_sha ∉ dead_bases`; `reason` is recorded on the replacement item and on the
`verdict_voided`/`speculative_discard` row that triggered it. Ordering: the live-tip test reads
`_frozen_inflight_entries` and `_dead_base_commits` under the single-writer verifier loop, never across an
await.

**Merger re-anchor:** `SpeculationController.on_base_invalidated(dead_sha, live_tip: str | None)`: if
`spec_base == dead_sha` or `pending_spec_base == dead_sha`, set it to `live_tip` (or None, releasing a retained
permit on FALLBACK semantics). Idempotent. Never acquires.

**Gate re-verify as dispatch:** `_land_rebased_or_park_gate` produces `RealMergeItem(base_sha=rebased_onto,
merge_commit=rebased_sha, speculative=False, permit=None, cap_permit=<taken non-blocking or None>)` and calls
`request_redispatch` at the FRONT; the gate row is emitted first. The head's old lease is released (D4).
Successors already in `_inflight` on the dead commit are re-anchored through `choose_remerge_base` with
`frozen_tip = rebased_sha`.

**Lease lifecycle:** `InflightEntry.lease` becomes None at verdict (`_resolve_and_release`); `status` stays
`verified_pending_landing`; landing needs no lease; a gate re-verify or probe calls `HostAllocator.acquire`
anew. K = `1 + runners + 1`. Invariant: `len(frontier entries with lease) ≤ hosts`, `len(frontier) ≤ K`.

**Depth and suspects:** `ChainPolicyState(suspects: list[task_id], recent: deque[(k, passed)])`;
`choose_depth(state, queue_len, depth_table) -> (k, reason)`; `on_tip_result(state, links, passed)`.
Pure; `_chain_halving_state` and `next_halving_state` deleted.

**Events (D10):** field names as listed; `merge_verify_cancelled{task_id, elapsed_ms, fraction, cause}`;
`verdict_voided{..., item_commit, killer}`; `merge_finalized{..., queue_to_land_secs}`. Consumers:
`scripts/merge_lane_throughput.py`, the dashboard merge view, leaf ι.

**Hook contract (D8):** `hooks/project-checks` derives `(module → lint, type)` from the yaml through a
small Python helper `scripts/hook_module_checks.py` that imports `verify_plan.effective_merge_module_configs`
and prints the commands for the staged paths; the hook runs them; exit non-zero blocks the commit with the
module and command named. The screen leg runs `scripts/merge_screen.yaml`'s node ids with plain `pytest`
(no `run_node_ids` dependency).

## 7. Boundary-test sketch (B + H)

| # | scenario | preconditions | postconditions |
|---|---|---|---|
| 1 | head A fails while B (stacked on A) and C (stacked on B) are in flight / queued | K = 3, chain_cap > 0 | B cancelled with a `merge_verify_cancelled` row; B re-merged on main (no live tip) `speculative=False`; C re-anchored on B′'s commit `speculative=True` with a permit; no `verdict_voided` for C |
| 2 | direct commit lands under head A; B stacked on A | A passed | gate row written; A″ dispatched as an item on the fastest free host; B re-anchored on A″ (`base_relation=rebased_head`), not voided; when A″ lands, B lands with no gate row |
| 3 | the frozen tip is a vacated entry (3082 shape) | tip entry `vacated=True` | `choose_remerge_base` returns main with reason `tip_vacated`; `verify_base_frozen_tip_violations` of the new class = 0 |
| 4 | permit conservation across a cascade re-anchor | ledger depth K | `speculation_accounting_violations` empty after 100 randomized cascades (property test) |
| 5 | laptop finishes B while local head A still verifies | K = 3 | laptop lease released at B's verdict; C dispatched to the laptop stacked on B; B lands after A with no gate |
| 6 | gate re-verify needs a host while both are busy | | A″ waits for a host (no deadlock: it is at the front of redispatch) and takes the first free one |
| 7 | chain of 4 fails at the tip | suspects empty before | links requeued at the front in aging order; suspects = 4; next dispatch verifies 2; on pass they land and suspects = 2; on the next fail of 1, the lone suspect is blocked with findings |
| 8 | p̂ crosses a depth-table boundary | 30 fresh verdicts | the next dispatch's `chain_depth_reason` names the new row; `chain_cap: 0` disables chaining regardless of the table |
| 9 | head alone in the frontier, queue ≥ 2 | D5 | the head itself chains (`chain_items > 1` on a non-speculative dispatch) |
| 10 | staged `scripts/foo.py` with a type error, commit on main | hook installed | commit refused naming `module:scripts:type`; `--no-verify` still lands it |
| 11 | staged `plans/x.capability-manifest.yaml` naming a script that does not exist | | commit refused by the screen leg naming the manifest test |
| 12 | main-tip sweep reports `type_error` on `module:scripts:type` | 8411 landed | hold armed with F containing `module:scripts:type`; an item whose tree passes the module's type check verifies at once |
| 13 | canary readout | 7 d after deploy | every §1 ratio computed from the new fields with the report's rules; a false predicate sets `chain_cap: 0` and files one L2 |

## 8. Decomposition plan

Priorities high for α–ζ, ι; medium for η, θ; serialized `worker.py` leaves in α → β → γ → δ → ε → ζ order by
real `add_dependency` edges. Sizing per the overlay bands (300–1,500 LOC; ≤ 12 files). `metadata.x_finding_key`
is not used (this is not a quality-instrument remediation).

- **α — Instrument the lane: runner_reason, base_relation, cancelled verifies, void provenance, queue-to-land**
  (intermediate; `worker.py`, `verify_runner.py`, `merge_lane/types.py`, dashboard merge view,
  `scripts/merge_lane_throughput.py`; ~400 LOC). Unlocks β–ζ's measurability and ι. Signal: the five fields
  appear on live rows written by the real producers within one day of deploy; the throughput script prints
  them. Prereqs: none.
- **β — Re-anchor recovery re-merges on the live tip; re-anchor the merger; amend §5.3** (intermediate;
  `worker.py`, `speculation_controller.py`, `plans/two-layer-merge-queue-prd.md` §5.3 prose, tests
  `test_merge_queue_verify_base_invariant.py`, `test_merge_queue_chain_intact.py`,
  `test_merge_queue_concurrent_verify.py`, `test_merge_queue_permit_conservation.py`; ~900 LOC). Unlocks γ.
  Signal: boundary rows 1, 3, 4; on df, `verdict_voided` per landing < 0.3 within 3 days (from 0.78) with
  `item_commit` named. Prereqs: α.
- **γ — Gate re-verify as a dispatch; successors re-anchor on the rebased head** (intermediate; `worker.py`,
  `gates.py`, `reverify_gate_decision.py`; ~800 LOC). Unlocks δ. Signal: boundary row 2; on df, P(gate|gate)
  < 0.5 within 3 days (from 0.79) and `base_relation=rebased_head` rows exist. Prereqs: β.
- **δ — K = hosts + 1; release the host lease at verdict** (intermediate; `harness.py`, `worker.py`,
  `verify_runner.py`; ~400 LOC). Unlocks ε. Signal: boundary rows 5, 6; on df, laptop `finished` share < 15%
  within 3 days (from 33%) and `inflight_total` reaches 3 in heartbeats. Prereqs: γ.
- **ε — Chain the head when alone** (intermediate; `worker.py`; ~400 LOC). Unlocks ζ. Signal: boundary row 9;
  on df, `chain_items > 1` on a `speculative=False` row. Prereqs: δ, 7720.

  *Amended 2026-11-10 (Leo).* ε's prerequisites were δ, 7720, 7721 and 7824. Tasks 7721 and 7824 now land
  after ε: 7721 depends on ε by a real edge, 7824 depends on 7721, and reify's resume capstone 8791 depends on
  both. Grounds, checked against main `29e22e3406`:
  - dark-factory runs no spec warm-lane pool (`git.merge_spec_warm_lane_pool` is false by default and unset in
    its yaml), so `git_ops.py::GitOps.acquire_spec_lane` always takes the cold path there. The leaked lane and
    the wasted warm seed that 7721 fixes cannot occur on dark-factory.
  - dark-factory's chain build took 2.3 to 7.7 s on the five chain verifies of 2026-11-09..10, against a 300 s
    `merge_deep.chain_build_timeout_secs`. ε chains the head only when the frontier holds nothing else, so the
    verifier-loop stall that 7824 removes holds up no other entry.
  - no code at ε's sites (`_deep_chain_placement` guard 1, the reset arm in `_run_inflight_verify`,
    `_adopt_head_on_tip_authority`) reads what 7721 or 7824 introduces.
  - of the nine orchestrator units on the host, only dark-factory and reify set `merge_deep.chain_cap` above 0,
    and only reify runs the warm pool. reify's scheduler is paused until 8791.

  Residual on dark-factory until 7721 lands: a chain build cancelled or timed out during cold acquisition can
  leave a partly created `_merge-*` worktree. Residual on reify: a merge submitted by hand while its scheduler
  is paused can chain a head with the inline build and the warm pool; set reify's `chain_cap: 0` if that
  matters before 7721 and 7824 land.

- **ζ — Suspect-list bisect, depth table, chain_cap as kill switch; cancel 3191** (intermediate;
  `worker.py`, `config.py`, yaml comment, docs; ~700 LOC). Unlocks ι. Signal: boundary rows 7, 8;
  `chain_depth_reason` on live rows. Prereqs: ε.
- **η — Hook parity with the merge gate's module checks, plus the screen leg for tree-read files** (leaf;
  `hooks/project-checks`, `scripts/hook_module_checks.py`, `scripts/merge_screen.yaml` if absent,
  `tests/scripts/test_hook_pyright_packages_drift.py` rewritten; ~400 LOC). Signal: boundary rows 10, 11
  against a throwaway repo in CI, and the first refused direct commit on main (hook output in the committer's
  terminal). Prereqs: none. G7: `exceptions-owned-or-ratified` — the hook refuses, it never allow-lists.
- **θ — Amend 8411: F includes type-error units** (decompose action, not a new leaf: `update_task` on 8411
  with D9's text; 8411 keeps its dependency on 7921). Signal: boundary row 12 belongs to 8411.
- **ι — Deploy and 7-day canary readout** (leaf, `task_kind='deterministic'`, delayed milestone 7 d after
  the fleet redeploy that carries ζ; before_done: the §1/D12 predicates computed by
  `scripts/merge_lane_throughput.py` from α's fields over the 7-day window; always escalates with the table).
  Signal: the readout file `plans/merge-lane-policy-prd.measurements/<date>-canary.md` committed with every
  predicate's value and rule, and `chain_cap` still > 0. Prereqs: ζ, η. G7: `readers-prove-their-producer` —
  the readout refuses to run when α's fields are absent from the window.
- **Closing actions at decompose:** 5056 → done (measurement = the report), 5098 → done (this PRD), 3191 →
  cancelled (D7), 7829 → cancelled (superseded by ι's deploy; 3190 re-pointed to depend on ι instead).

G7 walk (advisory, author mode): `holds-owned-and-bounded` — the suspect list is bounded by the failed tip's
depth and cleared on resolution; a re-anchored item that never lands is bounded by INV-3 and the existing
no-progress abort. `status-matches-liveness` — releasing the lease at verdict leaves `InflightEntry.status`
naming "verified, awaiting landing", written by the same choke point. `no-silent-fail-soft` — a re-anchor that
falls to main records its reason on the item and the event. `contracts-machine-checked` — the depth table is a
config schema with validation, not prose. `one-fact-one-home` — the carve-out's reversal lives in §5.3 (amended
by β) with this PRD as its pointer, not restated in code comments beyond a cross-reference.

## 9. Out of scope

Serial-laptop policy value; second df laptop slot; family-green disjoint skip (6943 unchanged); trains and
coalescing (7739, 7860); queue re-ordering; arbitration between merge legs and remote task legs on the laptop
(remote-task-leg PRD); the docs lane (5411/8258/8260); the κ/λ extraction itself; tightening the frozen-prefix
definition for phantom heads (the §5.3 "separate task"); the pre-verify screen's own leaves (7918–7924).

## 10. Open questions (tactical)

1. **Where p̂'s 30-verdict window persists across restarts.** Suggested: in-memory, seeded from the last 30
   `merge_verify` rows at start (the probe's `_recent_verify_outcomes` shape). Decide in ζ.
2. **Whether the gate-rebased item (γ) prefers the head's previous host.** DECIDED in γ (task 8525): no
   head-host preference. The rebased head is dispatched like any item, through `HostAllocator.acquire`
   with the configured `verify_host_policy`. See
   `orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker._dispatch_rebased_head`.
3. **The hook's screen-leg file set.** DECIDED in η (task 8531): exactly the D8 globs, not narrowed. A
   "package" is a uv workspace member with a `src` dir, so `scripts/` is not one. See
   `scripts/hook_module_checks.py::screen_triggered`.
4. **Whether `merge_verify_cancelled` should carry partial unit outcomes.** Suggested: only `elapsed_ms` and
   `fraction`. Decide in α.
