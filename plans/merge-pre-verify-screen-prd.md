# Merge pre-verify screen and main-red hold ("Stream 1")

**Status:** draft for Leo's approval — 2026-10-30. Not decomposed.
**Slug:** `merge-pre-verify-screen` · **Approach:** B + H (contract and two-way boundary tests; the merge lane is a load-bearing seam).
**Origin:** `plans/mcp-coalesce-trains-2026-10-26.md` §0 (Leo's ruling of 2026-10-29: Stream 1 goes to `/prd`), §2.5, §4.4, §6.1, and its E3 seat output (levers L1–L3).
**Evidence:** `plans/merge-pre-verify-screen-prd/`:
- five seat outputs: M1 screen premise, M2 red-base premise, C1 verify-path map, C2 red verdict and S1.3 substrate, D owners;
- `L-lead-measurements.md`, for the figures no seat produced.

Every figure below carries its counting rule there.
**Code anchors** verified against main `07bd68fd99` (2026-10-29). Cited by symbol; re-locate at implementation time.

## 1. Goal

Stop spending merge verifies whose outcome a seconds-long check already knows.

1. **S1.1, the screen.** Before a merge item's verify starts, run a short whole-tree screen on the exact tree the verify would test.
   - An item that fails it on its own defect bounces in seconds, carrying the check's findings, and no verify host is leased for it.
   - The same screen runs in every task-role verify, so most defects never reach the lane.
2. **S1.2, the main-red hold.** While the lane holds a confirmed main-red verdict, the verdict's failing tests join the screen.
   - An item whose tree still fails them, on a main that fails them too, waits in its lane: no verify, no bounce, no re-pend.
   - An item whose tree passes them is a fix, or the red was a flake, and it verifies at once.
   - The hold is owned by the red's existing L1. It clears when that L1 resolves or when bare main passes the tests, and it is bounded by a maximum age.
3. **S1.3, hygiene.** `get_changed_line_ranges` gets a canonical branch ref and fails loudly. The workflow-side train former is switched off.

**What an operator sees:**
- `runs.db` has `merge_attempt` rows with outcome `screen_failed`, and no `merge_verify` row for the bounced request.
- A blocked task's L0 detail carries the screen's `file:line` findings under review category `merge_screen`.
- During a red main, `get_merge_halt_status` shows:
  - the active verdict (main sha, failing tests, owning escalation id);
  - the held requests and the age of the oldest.
- After the verdict arms, the only merge verifies that start through the merger on that sha are those whose tree passed the failing tests. Items already dispatched or merged ahead are not covered (§9).

## 2. Background and premise validity (G6)

### 2.1 S1.1: a cheap check already knows over a third of merge-verify failures

Measured over 2026-10-13T21:48Z → 2026-10-29T17:40Z (15.8 days, 630 merge verifies, 147 failed). Source: M1 §2–§5, plus L4 for the non-train column.

| Screen set | Checks | Cost | Caught, all kinds | Caught, non-train (what recurs with coalescing off) |
|---|---|---|---|---|
| Ratchet only | `scripts/inline_suppressions.py --check` | 8 s CPU | 50 / 147 (34.0%), 24.7 h | 43 / 129 (33.3%), 17.6 h |
| **Fast seven (this PRD)** | ratchet + six pytest selections, each < 10 s CPU | ~25 s CPU; **9.8 s wall in parallel** (L3) | **58 / 147 (39.5%), 28.9 h** | **48 / 129 (37.2%), 20.0 h** |
| Next tier (not now) | + seven 14–31 s checks | ~160 s CPU | 71 / 147 (48.3%), 34.3 h | 58 / 129 (45.0%) |

"Caught" means failed on the item's own defect: the check is red on the verify's merge tree and green on its first parent. All 83 catches across the tiers reproduce that way. The hours are the verify time those failures spent, an upper bound on what the screen saves.

- **False-alarm evidence covers three checks only.**
  - The ratchet (C1) was red on 0 of 60 sampled passing verifies and on 0 of 95 sampled lane-landed main tips. C3 was red on 0 of 20 and C2 on 0 of 8.
  - The six pytest checks in the fast seven (C8, C11, C14, C15, C18, C20) have **no** false-alarm sample. Their green rests on three runs on one sha (M1 §1.3).
  - Each false bounce costs an L0 and a steward cycle, so ζ's readout counts false bounces per check.
- **The defect is on the branch tip.** In 49 of the 50 ratchet catches the branch tip alone, unmerged, already fails. That is why the task-role screen (decision 2) pays: the defect is visible before the merge request exists.
- **Nothing runs the check earlier today.**
  - The task-role verify is diff-scoped: an `orchestrator/`-only diff never runs the `scripts` leg (`verify_plan.py::_task_role_production_pytest_run`).
  - No hook runs any of the seven checks.
  - In the merge verify the `scripts` leg ends about 7.5 min before the verify does, and the merge verify has no fail-fast across modules (M1 §1.1).
- **The E3 figures still hold.**
  - The ratchet failed the merge gate for 44 distinct tasks in the window (E3: 34 in 12 days).
  - The ratchet's failure rate is unchanged since chains went on: 3.4 against 3.3 failing verifies a day.
  - The solo failure rate is 24.6% before chains, against 14.7% on n = 34 after.
- **A live specimen.** esc-3932-9 (L2): task 3932 thrashed twice at the merge gate on the ratchet, each time after a full suite of 12,136 passing tests.

What the lane screen does not save: a bounce still costs the workflow an L0 and a steward cycle. Only verify host time is saved. The task-role screen is what removes the cycle for workflow items, which are about 85% of landings (memo §2.1).

### 2.2 The per-link motive has no live instance yet

- There have been 2 chain-tip verifies since `chain_cap: 6` went live, and both passed (M1 §2.1).
- Until one fails, the per-link case rests on a prior, not a measurement. 9.5% of verified tasks carried a ratchet defect, so a 3-link chain carried at least one about 25% of the time (M1 §5). The task-role screen will lower that rate.
- The chain screen (δ) is kept because Leo's §0 ruling names the per-link case as the one that matters most. It is sequenced after the chain-walk fix, 7720.

### 2.3 S1.2: doomed verifies on a judged-red base

Measured over 2026-08-01 → 2026-10-29. Source: M2 §2–§5.

**Doomed starts.** 102 verifies started on a main sha after a `main_health_red` verdict on that same sha, and before main advanced. They used 3,620 host-min; 91 of them failed (3,247 min).

| Window | Starts | Failed (min) | Note |
|---|---|---|---|
| before task 4953 | 85 | 77 (3,030) | 4.7% of that window's verify host time |
| 4953 → chains on | 17 | 14 (217) | over 13 days |
| chains on | — | — | no verdict yet |

Cancelled verifies leave no row, so these counts are a lower bound.

- **Why verifies keep starting.** No stored red verdict exists, and no verify-start path reads one (C2 A1–A2). The only queue effect of a red is the `compile_error` halt of lane `normal`. Every `test_failure` red, which is every verdict in the window, halts nothing.
- **The verify cannot pass by subtraction.** The "N new failing test(s) not present on main" comparison only labels a verify that has already failed; it never turns a failure into a pass (C2 A4, M2 §5a). Under `merge_verify_breadth: "full"`, every verify on a red sha reproduces the red unless its own tree fixes it.

**The 11 verifies that passed on a judged-red base are the design constraint:**

| Why it passed | n | Would high-lane let it through? |
|---|---|---|
| Declared fix-main task (`merge_lane: high`) | 4 | yes |
| Stacked on a fix's merge commit | 1 | no |
| Diff touched the failing test's file (an ad-hoc fix) | 2 | no |
| Unrelated item; 3 of the 4 look like flake or host-quirk reds | 4 | no |

No lane, metadata or diff-reachability proxy separates fixing verifies from doomed ones. Diff reachability fails outright on data-file fixes: the 10-24 codebook red was fixed by restoring a docs entry. Running the failing tests themselves on the item's tree does separate them. All 11 passed their full verify, so they pass the failing tests, which are a subset of it. That is decision 9.

- **Lifetimes.** Red shas last a median 107 min, from first verdict to the next main advance. 25 of 36 were ended by a harmless commit while the break persisted (M2 §4). So a hold keyed on the sha alone would re-learn the red at every docs commit. The F-screen hold is keyed on the failing tests instead, and re-probes bare main at each advance.
- **The E3 case reproduces.** Train 5445 started 64 s after 6057's verdict on base `0f443e6c0e` and burned 38.2 min. 28 of those minutes ran after main had already moved to the fix (M2 §3, C2 A3).

### 2.4 S1.3 is dead-path hygiene

- **The bug reproduces** (C2 B8):
  - `git diff main...5925` fails with rc 128 (ambiguous short id), and `get_changed_line_ranges` returns `{}`.
  - `git diff main...5563` silently diffs the unrelated commit `5563a70ca5`.
- **Both callers sit behind flags that are being turned off.**
  - `workflow.py::TaskWorkflow._maybe_form_train` is behind `merge_train_former_enabled`, which this PRD turns off.
  - `worker.py::SpeculativeMergeWorker._stack_coalesce_train` is behind `merge_train_coalesce_enabled`, which carrier 7739 turns off.
- **Declared atomic trains don't read the former flag.** Their membership comes from task-authored `metadata.train`; `_maybe_form_train` is reached only when that is absent (C2 B7).

## 3. Resolved design decisions

1. **The lane screen runs per merge item, at three dispatch-preparation sites:**

   | Site | When it runs | Covers |
   |---|---|---|
   | `merge_lane/worker.py::classify_and_merge`, called from the merger loop | after the plan-targets drop guard (`gates.py::_check_plan_targets_in_tree`, the template) | slot 1 and slot 2 |
   | `merge_lane/worker.py::_do_train_merge` | after `merge_to_main`, before `_leased_train_verify` | declared trains |
   | `SpeculativeMergeWorker._deep_chain_placement` | after `build_chain`, seven checks only | chain tips (decision 7) |

   - **Re-merges on the verifier loop are not re-screened.** `SpeculativeMergeWorker._remerge` re-enters `classify_and_merge` after a head-failure cascade, a stale-main discard or a dead base. The item was screened at its first merge, and a re-screen there would occupy the loop that finalizes landings (INV-8).
   - **Admission screening was rejected because of the tree it sees.** At admission only the branch-level tree exists, and it goes stale while the item waits. The merge-time screen sees the exact tree the verify will run, and the task-role screen gives the branch-level verdict earlier (C1 §7, rows A and B).
2. **The task-role verify runs the same screen** (Leo, 2026-10-29).
   - `verify_plan.py::derive_verify_plan` adds the screen run to every task-role plan, whatever the diff scope.
   - A finding fails the task verify, and the implementer sees it in its own verify–review loop.
   - It applies the same base-first rule as the lane (decision 6). A finding that is also red on the task branch's merge-base with main is reported as a non-failing note, so a red main never fails every task fleet-wide.
3. **What the screen runs: the fast seven** (Leo, 2026-10-29). M1 ids:

   | Id | Check |
   |---|---|
   | C1 | inline-suppression ratchet |
   | C8 | remote-gate wire fingerprint |
   | C11 | watchdog fake-systemctl drift |
   | C14 | subproject test collection |
   | C15 | uv-run venv isolation |
   | C18 | recovery-veto predicate collapse |
   | C20 | disjoint-skip / breadth yaml guards |

   - The list lives once, in a project wrapper, `scripts/merge_screen.py` (heuristic 11, SPOT; INV-5).
   - The orchestrator holds only the command that runs the wrapper, so its code stays project-agnostic (heuristic 9, deep module, narrow interface). Reify sets no command, so nothing changes there.
   - The next tier (14–31 s checks) waits for the readout (ζ).
4. **The screen provisions its environment exactly as the verify does: `uv run` in the screened tree.**
   - Nothing is borrowed from the dispatcher (no interpreter, no `PYTHONPATH`), so there is no skew between dispatcher and tree.
   - Measured (L2, L3): a cold `uv sync --all-packages --frozen --offline` takes 1.6 s, hardlinked from the uv cache. All seven checks pass in 9.8 s wall run in parallel, at load average 120.
5. **The screen's verdict is structured data, never an exit code alone** (heuristic 12; INV-1; INV-2; INV-11).
   - The wrapper's `--json` mode prints a `merge-screen/1` report and exits 0 whenever it ran. The report has one entry per check: status `pass`, `finding` or `error`, the findings, and the duration.
   - The orchestrator validates the shape on parse. An unknown schema or a malformed entry is an `error`.
   - Exit codes alone can't be trusted. Under `python -I` an import failure exits 1, which is the scanner's finding code (L1). The wrapper reads the scanner's own `--json` report instead (`schema_version` 2, `status`, `violations`).
   - For a pytest check, exit 1 is a finding. Any other non-zero exit is an `error`: a collection error, an interruption, or nothing collected.
6. **Attribution is base-first.** A check that is red on the merged tree is re-run on the base, the merge commit's first parent. Base verdicts are cached per sha, so a base checkout happens only when an item is red.
   - **Red on the item only:** the item bounces.
   - **Red on both:** it is not the item's defect, and the item proceeds to its verify as today.
     - If that base is a red main, the verify fails. The main-health probe then confirms the red and files its owned L1 and fix-main task, and γ arms a verdict from that (decision 9).
     - A static check never arms a verdict by itself, because a screen-sourced verdict would have no owner.
   - Why base-first matters (M1 §2.5): on 10-19 a red direct-to-main commit sat under six unrelated items, and an item-only screen would have blamed all six.
7. **A screen bounce is an ordinary per-item outcome.**
   - **Shape:** `MergeOutcome('blocked', reason=<findings>, failure_category='merge_screen', failure_cause_hint=<failing check ids>)`, with a new `OutcomeKind.screen_failed`.
   - **Delivery:** like the drop guard's `Decided` outcome, through the existing generic blocked path. A workflow request gets an L0 with the findings as detail; an MCP caller gets the reason.
   - **Review category.** Today `workflow.py::TaskWorkflow._submit_to_merge_queue` picks the review category from the substring "verification failed" in the reason. α changes that site to map `failure_category` to the category, so the steward reads `merge_screen` (heuristic 12).
   - **What it doesn't feed.** The bounce never reaches `_run_post_merge_verify`, so it feeds none of the main-health probe, the dry-run, the flake ledger, skew classification or chain halving (C1 §5).
   - **The thrash guard.** It does feed the workflow's merge-thrash guard: it is a real defect of that item, so two identical bounces escalate to a human, as two identical verify failures do today.
   - **Chain links.** A chain link is never bounced at the chain site. δ truncates the chain before the first link whose cumulative prefix fails, with a new `ChainTruncationReason` member, `screen_failed`. It is a benign truncation, so it is **not** added to `CHAIN_FAULT_REASONS`, which is the lane-fault subset.
     - The truncated link gets no outcome (deep-merge decision 4).
     - Halving still sees only tip-verify results (decision 5).
     - The culprit is bounced later, at its own per-item screen, when it becomes the dispatching item.
8. **An instrument failure fails open, loudly** (INV-4; INV-11).
   - A wrapper crash, a sub-check `error` or a timeout never bounces an item and never fails a task verify. The verify proceeds as today.
   - The lane emits a `merge_screen` event with status `error`.
   - A streak of `merge_screen.error_streak_alarm` consecutive errors files one info L1, `merge_screen_unavailable`. The next clean screen closes it.
   - Verdict re-probe errors (decision 9) feed the same streak.
9. **S1.2 is an F-screen hold** (Leo, 2026-10-30).
   - **What a verdict is.** `MainRedVerdict` records:
     - the main sha;
     - the failing test ids F, being the triggering verify's `failing_test_ids` after the probe confirmed them wholly pre-existing;
     - the break signature;
     - the **owning escalation id**.
   - **Where it arms.** Only on the deferred probe path, inside `SpeculativeMergeWorker._auto_heal_main_health_deferred`, after the probe sha has been re-checked against current main (INV-3), and after the existing owner path has run:
     - `_escalate_owned_main_break`, which ensures a fix-main task and files the L1;
     - or, for `compile_error`, the halt branch, which binds `set_lane_halt_owner`.

     `_escalate_owned_main_break` returns `None` today. γ makes it return the escalation id that `_file_main_health_escalation` returns. A `None` id arms nothing, so the worker never holds a request for a red that has no L1 and no fixer (INV-7).
   - **The facts come from the shared builder.** `worker.py::_build_main_health_outcome` gains structured `probe_sha` and F fields, so the worker never parses the reason text.
   - **One verdict at a time** (INV-9).
     - Another confirmation on the same sha adds its ids to F.
     - A confirmation on a newer sha replaces the verdict.
     - `verify.py::known_failing_ids_on_main` stays the probe's own cache of raw baseline facts. The verdict records only what the probe confirmed, and holds point at the verdict.
   - **Screening.** While a verdict is armed, the screen at `classify_and_merge` (merger loop) and at `_do_train_merge` also runs F on the screened tree:

     | F on the item's tree | F on bare main | Outcome |
     |---|---|---|
     | passes | — | Proceed: the item is a fix, or the red was a flake |
     | fails | fails | **Hold**: no verify, no outcome, no status write, no re-pend |
     | fails | passes | The verdict clears and the item proceeds; its full verify judges it, as today |

     F never bounces anything; only the seven static checks do. A flaky F must never put an innocent item on the thrash ladder.
   - **How a hold is carried.** It takes the shape the operator-halt branch already uses:
     - `classify_and_merge` returns a held passthrough;
     - the merger calls `SpeculationController.on_abort()`, so no look-ahead stacks on a held merge commit;
     - the verifier applies the hold at the dispatch pre-check exactly as the halt branch does: cleanup, `_requeue_request` (the single requeue chokepoint, which also ends the request's stuck-ledger episode), and `REQUEUED_PREDISPATCH`.
   - **Held requests stay out of rotation.** `_pop_next_pickable` skips them, and they never become chain links (`chain_snapshot` excludes them).
     - Cancel, relane and a needs_rebase bounce act on a held request as on any buffered one, and drop its held mark.
     - For a declared train the hold uses `_stand_down_train`.
   - **Clearing.** Clearing releases every held request and calls `SpeculativeMergeWorker._signal_resume()`, so the waiting merger wakes. A verdict clears on any of:
     - F passes on bare main, re-probed at each main advance the worker observes and at most every `merge_main_red_hold.reprobe_interval_secs` otherwise, so a flaky verdict clears itself;
     - F fails on an item but passes on bare main (the screening table above);
     - its owning L1 resolves, through the halt-owner plumbing (`Harness._on_escalation_resolved`). For an operator this is `resolve_issue(owner_esc_id)`, the same path an owned halt uses; `force_unhalt_merge_queue` refuses owned halts by design;
     - an operator hot-reloads `merge_main_red_hold.enabled: false`, which disarms the verdict and releases every held request while the L1 stays open.
   - **Degrading.** A verdict older than `merge_main_red_hold.max_hold_secs`, or one whose re-probe errors for a streak, degrades to advisory: items proceed as today, and an event records it. The owning L1 stays open as the operator's surface.
   - **Scope.** The hold applies to every verify start the merger screens. High lane keeps only its existing meaning: fixers are popped first.
   - **Limits.**
     - Verdicts from the sync probe path arm nothing. Those are declared trains and solo re-verifies, 12 of 109 verdicts in the window, and for them the L1 is filed by the waiting workflow, not the worker.
     - A verdict with no test ids, or more than `max_failing_ids`, arms nothing.
     - The `compile_error` halt is unchanged.
     - A restart drops the verdict, and one to a few verifies fail before the probe re-arms it (INV-7: the hold expires coherently).
   - **The hold depends on the screen.** Config validation rejects `merge_main_red_hold.enabled: true` when `merge_screen.command` is unset (INV-1).
10. **S1.3: one canonical ref, loud failure.**
    - `GitOps.get_changed_line_ranges` builds its ref with `merge_lane/types.py::QueuedBranch.parse(ref, branch_prefix).full_name`, the existing canonical parser (heuristic 11).
    - A non-zero rc raises a typed error naming the ref, the rc and stderr (INV-11). The rc check goes in a new helper, which keeps the method's ratchet measure at 0 (C2 B8 gotcha).
    - Both callers are dormant once the flags are off. "Fail loudly" is therefore a direction chosen without a live reader, and wiring the former back up would re-judge it.
11. **The former goes off on this PRD's carrier** (Leo, 2026-10-29), following 7739's precedent.
    - The change is one line in `dark-factory-orchestrator.yaml`, plus its ruling comment. It is added after 7739 lands, because both carriers edit adjacent lines.
    - The knob is restart-only. A yaml-only merge does not trip the restart coordinator, whose watch list covers `src/` only. The flip goes live at the first of: the staleness backstop, a manual sweep, or the first `orchestrator/src/` merge of this PRD's leaves.
    - Deleting the former's code stays with task 5375, whose "wire or retire" question this answers for now.
12. **The chain canary measures stay with the deep-merge PRD** (Leo, 2026-10-29): η1 3190, θ 3192, and the 14-day readout 5059. D §3 lists the coverage gaps for measures 1, 3 and 4 for that program's owner; this PRD files nothing for them.

## 4. Pre-conditions (G3)

Each assumed capability is either verified by reading main `07bd68fd99` or built by this PRD:

| Capability | Status | Where |
|---|---|---|
| A merged tree checked out before dispatch, for every item kind | exists | `classify_and_merge` → `MergedOk.merge_wt`; `_do_train_merge`'s `merge_wt`; `ChainResult.lane`/`tip`/`links` (C1 §6; critic-verified) |
| Pre-verify bounce template (`Decided` blocked outcome + worktree cleanup) | exists | `classify_and_merge` with `_check_plan_targets_in_tree`; `OutcomeKind.dropped_plan_targets` |
| Requeue with no outcome, ending the stuck-ledger episode | exists | `_requeue_request`, the enforced single chokepoint; `request_ledger.py::RequestLedger.on_requeued`; the operator-halt branch's `REQUEUED_PREDISPATCH` |
| Waking the waiting merger | exists | `SpeculativeMergeWorker._signal_resume()`; the wait is in `_acquire_next_request` |
| A per-request "held" predicate in `_pop_next_pickable` / `chain_snapshot` | **built here (γ)** | both skip only halted lanes today. `test_merge_queue_build_chain.py::test_is_the_merge_request_typed_twin_of_unfrozen_suffix` pins `chain_snapshot() == unfrozen_suffix()`; γ states the new relation and updates the test |
| The red's owner (fix-main task + L1) and owner-resolution clearing | exists; one return value **built here (γ)** | `_file_main_health_escalation` returns the id (`Optional[str]`); `_escalate_owned_main_break` returns `None` today, and γ threads the id out; `set_lane_halt_owner`; `Harness._on_escalation_resolved`. `force_unhalt_merge_queue` refuses owned halts, so the operator exit is `resolve_issue` |
| The verdict's probe sha and failing ids at confirmation | exists inside the probe; **exported here (γ)** | `_build_main_health_outcome` is the one builder on both probe modes; `VerifyResult.failing_test_ids`; γ adds structured fields on the MAIN_RED outcome |
| Run named pytest node ids on a given tree and report each one's result | **built here (γ)** | `verify.py::confirm_isolated_rerun_verdict` takes a `VerifyResult` and returns a flake verdict per call-site policy. γ extracts its group → command → `run_verification` body into a public `verify.py::run_node_ids`, which both the discriminator and the screen call (INV-5) |
| `OutcomeKind` and `ChainTruncationReason` are extendable | exists | `merge_lane/types.py` |
| A throwaway worktree at any sha, for base screens and bare-main re-probes | exists | `GitOps.ephemeral_worktree(kind, sha)`. The new `WorktreeKind.MERGE_SCREEN` also needs its `git_ops.py::PROTECTED_PREFIXES` entry and its bash mirror in `orchestrator/scripts/warm-lane/lib_lane_state.sh::lane_protect_glob` |
| Provisioning a tree with `uv run` alone, without a cold-preprovision `npm ci` | exists, measured | `verify.py::run_verification` runs `_preprovision_shared_venv` only when `_is_verify_cold`, which is False without `.task/` (critic-verified); cost in L2 |
| The scanner's typed report | exists, measured | `scripts/inline_suppressions.py --json` (`REPORT_SCHEMA_VERSION` 2), L1 |
| The operator halt surface | exists, extended here | `harness.py::Harness.get_merge_halt_status` → `{wired, halted, owner_esc_id}`; γ adds the verdict, the held requests and the oldest age |
| The task-role plan seam | exists | `verify_plan.py::derive_verify_plan`, `PlannedRun` |
| The canonical branch-ref parser | exists | `QueuedBranch.parse(...).full_name` |
| Hot-reload plumbing for the new knobs | exists | `config.py::RELOADABLE_FIELDS`, composed from `_submodel_leaf_paths` |

## 5. Cross-PRD relationships and seam owners (G4)

| Seam | Direction | Mechanism | Owner |
|---|---|---|---|
| `plans/deep-merge-ahead-prd.md` | extends | δ adds a benign `screen_failed` truncation. A truncated link gets no outcome (decision 4); only a tip verify halves (decision 5); deep failures never feed the thrash guard | **this PRD** owns δ; deep-merge owns chain build, landing and the canary |
| task 7720 (chain walk lands 0 links) | δ depends on it | δ is built against a working walk | 7720 |
| task 7259 (`chain_truncated_reason` on the chain-arm event) | δ depends on it | δ's truncation becomes observable through 7259's field | 7259 |
| task 7721 (chain build latency) | coordinates | δ's tip screen runs after `build_chain`, under its own bound, and adds to verifier-loop occupancy (§8 G7) | 7721 owns the build budget |
| Main-health classification (1690, 3597, 5627), fix ownership (5471), auto-heal | consumes | γ arms only after the existing owner path has run. It does not change `verify_failure_is_preexisting_on_main`, `is_auto_heal_eligible` or `MainFixTaskFiler` | main-health; **this PRD** owns the verdict record and the hold |
| Verify-base guard ruling (3206, 3981) | distinct | γ judges base *health* (failing tests), not base *identity* (frozen prefix). The advisory-by-decision ruling is untouched | — |
| `plans/orchestrator-atomic-train-merge-prd.md`; task 6755 (train verify parity) | consumes, coordinates | α screens at `_do_train_merge`: a finding is a blocked outcome *without* `TRAIN_VERIFY_FAILED_REASON_PREFIX`, so the tip workflow escalates to a human. A γ hold uses `_stand_down_train`. α adds a call; 6755 rebinds the verify. There is no edge between them: whichever lands second rebases | **this PRD** owns the screen call; 6755 owns the verify binding |
| `verify.py` (12k lines) | γ edits it | `run_node_ids` is extracted from the isolated-rerun discriminator, which keeps working through it | γ |
| INV-12 inline-suppression gate (5611) | consumes | the wrapper reads the scanner's `--json` report and does not change its contract | INV-12 PRD |
| task 5375 (former deployed with 0 events) | answers | the flip rides this PRD's carrier; code deletion stays with 5375 | 5375 |
| task 7739 (coalesce off) | sequences | the former flip is added after 7739 lands (adjacent yaml lines) | 7739 |

## 6. Contract (B + H)

```
# scripts/merge_screen.py (project-owned; the only list of screen checks)
merge_screen.py --root <tree> --json            -> stdout: ScreenReport JSON, exit 0 if it ran
merge_screen.py --root <tree> [--only <id>...]  -> human report; exit 0 clean, 1 finding, 2 instrument error
ScreenReport = {schema: "merge-screen/1",
                checks: [{id, status: pass|finding|error, findings: [str], duration_s}]}

# orchestrator/src/orchestrator/verify.py (extracted, shared with the isolated-rerun discriminator)
async run_node_ids(tree: Path, config, module_configs, ids: Collection[str])
    -> Mapping[str, Literal['pass', 'fail', 'error']]

# orchestrator/src/orchestrator/merge_lane/screen.py (new; imports nothing from worker.py)
@dataclass(frozen=True) ScreenCheck   {id, status, findings: tuple[str, ...], duration_s}
@dataclass(frozen=True) ScreenVerdict {tree_sha, checks: tuple[ScreenCheck, ...]}
async run_screen(tree: Path, config, *, only: frozenset[str] | None = None) -> ScreenVerdict
async screen_item(merge_tree: Path, base_sha: str, main_sha: str, config,
                  verdict: MainRedVerdict | None) -> ScreenDecision
ScreenDecision = Proceed | Bounce(outcome) | Hold(generation) | ProceedAndClear(generation)

# orchestrator/src/orchestrator/merge_lane/main_red_verdict.py (new; one owner: the worker)
@dataclass(frozen=True) MainRedVerdict {main_sha, failing_ids, signature, owner_esc_id,
                                        confirmed_at, generation}
class VerdictBook:
    arm(v)              # same sha: union F; newer sha: replace
    clear(reason)       # reason: main_green | item_proved_green | owner_resolved | disabled | expired | reprobe_errors
    current() / note_main_advance(sha) / held() -> tuple[HeldRequest, ...]
```

Invariants:

- **C1, kill switch.** `merge_screen.command` empty ⇒ no screen code path executes at any site, and behaviour is byte-identical to today (asserted by test). `merge_main_red_hold.enabled: false` ⇒ no verdict arms. The hold enabled without a screen is a config-load error.
- **C2, attribution.** `Bounce` only when a static check is `finding` on the merged tree and `pass` on the base. `finding` on both is never a bounce. F never produces `Bounce`.
- **C3, fail open.** Any `error` or timeout ⇒ `Proceed`, plus a `merge_screen` event with status `error`, plus a step on the error streak. Never `Bounce`, never `Hold`.
- **C4, no verify on a bounce.** A `Bounce` request has no `merge_verify` row and no verify-host lease.
- **C5, hold rules.**
  - `Hold` only while a verdict is armed, F fails on the item, and F fails on bare `main_sha`; it never uses the item's `base_sha`, which for slot 2 is an unlanded merge commit.
  - A held request keeps its future pending and its task status untouched, sits in its lane buffer, is skipped by `_pop_next_pickable`, and is never a chain link.
  - It leaves the stuck ledger through `_requeue_request`.
  - Every `clear` releases every held request and signals resume.
- **C6, bounded hold.**
  - Every armed verdict carries an `owner_esc_id`.
  - It clears on owner resolution (the operator's `resolve_issue`), on a hot-reload of `enabled: false`, or when F passes on bare main.
  - It degrades after `max_hold_secs` or after an error streak on its re-probe.
  - `get_merge_halt_status` shows the oldest held age.
- **C7, chain truncation.** At least one screen runs on the chain tip, seven checks only. On a finding the chain is truncated before the first link whose cumulative prefix fails, using a bounded search over the link commits. That link gets no outcome. `_note_chain_outcome` sees only tip-verify results. While a verdict is armed, no chain is built and I0 verifies adjacent.
- **C8, one home for the verdict.** Only `VerdictBook` holds it (INV-9). Its events (`main_red_verdict` armed/cleared/degraded, `merge_held`) carry the sha, F and the owner id, unlike today's sha-less `main_health_red` row.

## 7. Boundary-test sketch (B + H)

| Scenario | Preconditions | Postconditions |
|---|---|---|
| Own-defect bounce | branch adds an unbaselined `# type: ignore`; base clean | outcome `screen_failed`; reason names `file:line` and the check id; no `merge_verify` row; L0 detail carries the finding (workflow); MCP reason carries it |
| Review category | the same bounce, workflow-submitted | `.task/reviews/merge.json` and the L0 carry category `merge_screen`, not `merge_ff_failed` |
| Red base is not blamed | the same finding on base and item | no bounce; the item verifies; no verdict armed from the screen |
| Instrument error fails open | wrapper crashes, times out, or prints an unknown schema | item verifies; `merge_screen` event with status `error`; K consecutive errors ⇒ one `merge_screen_unavailable` L1, closed by the next clean screen |
| Kill switch | `merge_screen.command` unset | byte-identical dispatch (golden transcript); the hold enabled without a screen fails config load |
| Task-role catch | orchestrator-only diff with an unowned suppression | task verify fails with the scanner finding |
| Task-role red main | screen check red on the task branch's merge-base | non-failing note; task verify not failed |
| Fixer verifies at once | verdict armed (F = {t}, owner L1 open); item's tree passes t | verify starts; lands; re-probe at the advance passes t; verdict cleared; held items released; merger woken |
| Doomed item is held | verdict armed; item fails t; bare main fails t | no verify, no outcome, status unchanged; `merge_held` event; listed with its age in `get_merge_halt_status`; not a chain link; no look-ahead stacked on it |
| Item proves main green | verdict armed; item fails t; bare main passes t | verdict cleared (`item_proved_green`); item verifies; no bounce |
| Flaky verdict clears | verdict armed; re-probe passes t with no advance | cleared at the re-probe interval; held items released |
| Owner resolved | verdict armed; its L1 is resolved | verdict cleared (`owner_resolved`); held items released |
| Operator clears | `resolve_issue` on the owning L1, or a hot reload of `merge_main_red_hold.enabled: false` | verdict cleared (`owner_resolved` / `disabled`); held items released |
| Hold expires | verdict older than `max_hold_secs` | degraded; items proceed as today; event recorded |
| Re-probe keeps erroring | K consecutive re-probe errors | degraded; `merge_screen_unavailable` L1 |
| Stacked on a fix | slot-2 item whose base is a high-lane fix's merge commit | its tree passes t ⇒ proceeds |
| Held item cancelled or relaned | `merge_cancel` / `merge_relane` on a held request | acts as on any buffered request; held mark dropped |
| Restart | verdict armed, items held | after restart no verdict; held requests re-merge normally; the next confirmed probe re-arms |
| No owner, no hold | a sync-path MAIN_RED (declared train) | no verdict armed |
| Chain link defect | 5-link chain, link 3 carries a finding | chain truncated to links 1–2 before the tip verify; `chain_truncated_reason = screen_failed`, not a fault reason; link 3 gets no outcome; halving state unchanged; link 3 bounced later at its own per-item screen |
| No chain on a red main | verdict armed; slot-2 dispatch with queue ≥ 2 | no chain built; I0 verifies adjacent |
| Declared train finding | train tip tree carries a finding | untagged blocked outcome ⇒ tip workflow human L1; no union verify |
| S1.3 ref | branch `task/5563` exists, and an unrelated commit abbreviates to `5563` | `get_changed_line_ranges('5563')` diffs `task/5563`; a missing ref raises naming ref, rc and stderr |

## 8. Decomposition plan

Rules for every leaf:

- **Sizing.** Leaves follow the overlay sizing bands.
- **Same-file edges.** Every leaf touching `merge_lane/worker.py` is serialized by a dependency edge (same-file rule).
- **New code goes in new modules.** `worker.py` is 23k lines, far past heuristic 14's alarm, so new mechanisms go in `screen.py` and `main_red_verdict.py`, and `worker.py` gains call sites only. Each new file must make sense in isolation and import nothing from `worker.py` (heuristic 13: layering is a downward import, never a satellite over the parent's private state).
- **The complexity ratchet.** `orchestrator/tests/merge_lane_ratchet_baseline.json` pins the cognitive complexity of every `worker.py` function these leaves add a branch to:

  | Function | Pinned |
  |---|---|
  | `classify_and_merge` | 31 |
  | `_do_train_merge` | 37 |
  | `_dispatch_item` | 43 |
  | `_pop_next_pickable` | 13 |
  | `chain_snapshot` | 3 |
  | `_deep_chain_placement` | 8 |

  Each leaf either extracts the new branch into a new function, or authorizes the raise with `scripts/merge_lane_metrics.py --authorize-raise`.

The leaves:

- **α, the screen and the per-item lane screen** (high).
  - Builds `scripts/merge_screen.py` (the seven checks, `merge-screen/1`), `merge_lane/screen.py`, and the `classify_and_merge` (merger path) and `_do_train_merge` call sites.
  - Adds:
    - `OutcomeKind.screen_failed` and its `dashboard/src/dashboard/data/outcome_colors.py::classify_outcome` family;
    - the `merge_screen` event and the error-streak L1;
    - `WorktreeKind.MERGE_SCREEN`, with its `PROTECTED_PREFIXES` entry and bash mirror;
    - the `failure_category` → review-category mapping in `workflow.py::_submit_to_merge_queue`;
    - the config block `merge_screen.{command, timeout_secs, error_streak_alarm}` (reloadable, default off), enabled in this repo's yaml.
  - Adds a guard test that *executes* each wrapper check's selection against the tree and fails if any selects nothing (INV-10): a renamed test must not silently empty the screen.
  - Modules: orchestrator, scripts, dashboard. ~1,100 LOC.
  - **Signal:** the boundary rows "own-defect bounce", "review category", "red base is not blamed", "instrument error fails open", "kill switch" and "declared train finding".
  - Live confirmation: a `merge_attempt outcome=screen_failed` row in runs.db with no `merge_verify` row for that request.
- **β, the task-role screen** (high). Depends on α.
  - `verify_plan.py::derive_verify_plan` adds the screen run to every task-role plan. It applies base-first against the branch's merge-base with main, and an `error` or a base-red finding is a non-failing note (decision 2, decision 8).
  - Modules: orchestrator. ~400 LOC.
  - **Signal:** the boundary rows "task-role catch" and "task-role red main".
  - Live confirmation: a task-role verify of an `orchestrator/`-only diff shows the screen run in its plan file (`data/verify-logs/<task>/attempt-*.plan-*.json`).
- **γ, the main-red verdict and F-screen hold** (medium). Depends on α (same file).
  - Extracts `verify.py::run_node_ids` from the isolated-rerun discriminator, which then calls it.
  - Builds `merge_lane/main_red_verdict.py`. The verdict is armed in `_auto_heal_main_health_deferred` after the owner path, using structured `probe_sha` and F from `_build_main_health_outcome`.
  - Wires the F run into the screen, and the hold into the held passthrough, `on_abort()`, the dispatch pre-check, `_pop_next_pickable` and `chain_snapshot` (with the twin test updated) and `_stand_down_train`.
  - Wires clearing through re-probe, owner resolution, the disable reload and expiry, plus `_signal_resume()`, and extends `get_merge_halt_status`. `_escalate_owned_main_break` returns the owner id.
  - Adds the config block `merge_main_red_hold.{enabled, reprobe_interval_secs, f_screen_timeout_secs, max_failing_ids, max_hold_secs}` (reloadable), validated against `merge_screen`.
  - Modules: orchestrator (`worker.py`, `verify.py`, `harness.py`, `config.py`). ~1,400 LOC, near the top of the band. If the architect's plan saturates, the cut is the `run_node_ids` extraction as its own predecessor.
  - **Signal:** the boundary rows from "fixer verifies at once" through "no owner, no hold".
  - Live confirmation: the next red episode shows a `main_red_verdict armed` event with sha, F and `owner_esc_id`. Every merge verify that starts through the merger on that sha after it belongs to an item whose `merge_screen` event shows F passing.
- **δ, the chain-tip screen** (medium). Depends on γ (same file), 7720 and 7259.
  - Screens the tip with the seven checks in `_deep_chain_placement` after `build_chain`, searches for the culprit prefix, and truncates with the benign `screen_failed`.
  - Builds no chain while a verdict is armed.
  - Its total added verifier-loop occupancy is bounded by `merge_screen.timeout_secs` per screen, with at most ⌈log₂ links⌉ + 1 screens.
  - Modules: orchestrator. ~450 LOC.
  - **Signal:** the boundary rows "chain link defect" and "no chain on a red main".
- **ε, canonical ref and loud failure in `get_changed_line_ranges`** (low; `complexity='simple'`). No dependencies.
  - Modules: orchestrator (`git_ops.py` and its tests). ~80 LOC.
  - **Signal:** the boundary row "S1.3 ref", run in a real-git fixture.
- **ζ, the readout, 14 days after β and γ land** (medium). Depends on β and γ.
  - `metadata.milestone` delayed 14 days, anchored when its last dependency lands; it has a condition and a trigger, per standing practice.
  - Commits a repeatable measurement script under `scripts/` and a dated readout under `plans/`. The readout reports:
    - screen bounces, and the verify-minutes they replaced;
    - false bounces **per check**: a bounced item whose next unchanged verify passed;
    - task-role catches and task-role base-red notes;
    - `error` streaks;
    - verdicts, holds, held-minutes, clears by reason, and fixers passed;
    - chain truncations, if δ has landed.
  - It decides, or files a declined record for:
    - the 14–31 s check tier;
    - whether E3's L2 (fail-fast leg reordering) is still worth a measurement task;
    - verdict persistence across restart;
    - arming on the sync probe path.
  - **Signal:** the readout file committed on main, with each of the four decisions filed or declined.

The former-off flip is not a leaf. It rides this PRD's carrier (decision 11).

**G7 (advisory walk, author mode).** Walked against `docs/legibility/design-invariants.md`. The critic's independent walk found INV-4, INV-7 and INV-8 hit in the first draft; the fixes are folded in above.

| Invariant | How the design meets it |
|---|---|
| INV-1 | the `merge-screen/1` shape is validated on parse; the hold-without-screen config is rejected at load |
| INV-2 | bounces carry structured findings and check ids; verdict events carry sha, F and owner |
| INV-3 | arming re-checks the probe sha against main; clearing is a fresh probe; holds key on real main |
| INV-4 | both the screen and the verdict re-probe feed one error streak, which files an L1; expiry and degradation are events |
| INV-5 / INV-9 | one check list (the wrapper); one node-id runner (`run_node_ids`); one verdict home (`VerdictBook`) |
| INV-6 | holds write no status |
| INV-7 | each hold is owned (`owner_esc_id`), bounded (`max_hold_secs`, degrade), operator-exitable (`resolve_issue` on the owner, or the disable reload) and visible with its age; a restart drops it coherently |
| INV-8 | every screen is an awaited subprocess with a timeout, so nothing blocks the loop thread |
| INV-10 | the selection guard executes each check's selection |
| INV-11 | `error` is distinct from `pass` and audible through the streak; S1.3 raises |
| INV-12 | the design adds no suppressor |
| INV-13 | the verdict's producer is the existing deferred probe, which fired 97 times in the window; `get_merge_halt_status` reports "no verdict" explicitly |
| INV-14 | n/a |

INV-8 occupancy figures:

| Loop | Added time |
|---|---|
| Serial merger | ~10 s per item; during a red episode, up to `f_screen_timeout_secs` per item, when only fixers can land and high-lane fixers are popped first |
| Verifier loop | δ only: ~10 s per chain dispatch on the pass path, and ⌈log₂ links⌉ + 1 bounded screens on a finding; no F run ever happens there |

No waiver needed.

## 9. Out of scope

| Item | Why |
|---|---|
| Reordering verify legs to fail fast (E3 L2) | It interacts with the "not present on main" comparison, which needs a complete failing set; ζ decides whether to measure it |
| The 14–31 s and 37–52 s check tiers | ζ decides from live data |
| Re-screening on `_remerge` (verifier loop); gate re-verify after a rebase (`gates.py::_reverify_rebased_tree`); declared-train member solo re-verifies | The first keeps the finalize loop free. The others follow a verify that already passed. Their volume is unmeasured: M2 could not separate gate re-verifies from other `depth=null` rows |
| Arming from the sync probe path (declared trains, solo re-verifies; 12 of 109 verdicts) | The L1 there is filed by the workflow, so the worker has no owner id; ζ decides |
| Aborting in-flight verifies when a verdict arms; protecting the items already merged ahead; the pre-verdict window (M2 §2d) | The first wasted verify of an episode is what discovers it |
| Persisting the verdict across restart | Costs one to a few verifies per restart per episode; ζ decides |
| Coalesce-only work: unwind on exceptions and `stop()` (6450), `has_workflow_waiter`, Stream 2 | Moot while coalescing is off |
| The chain canary measures and their gaps | Deep-merge PRD (decision 12) |
| The rc-discard pattern in the other 10 `GitOps` methods (review-all 2026-10-08) | ε fixes the one with a live wrong answer; this is not a sweep |

## 10. Open questions (tactical)

| # | Question | Suggested | Decided in |
|---|---|---|---|
| 1 | `merge_main_red_hold` defaults | `reprobe_interval_secs` 900, `f_screen_timeout_secs` 300, `max_failing_ids` 25 (disk-pressure mass failures carry 45–159 ids), `max_hold_secs` 21600 (6 h; M2's longest episode ran 1,360 min to its last advance). Use the probe's own timings | γ |
| 2 | `merge_screen` defaults | `timeout_secs` 180, `error_streak_alarm` 5 | α |
| 3 | Culprit search in δ: bisect or a linear prefix walk? | The bound is the contract (C7); pick by chain-lane checkout cost | δ |
| 4 | β's non-failing note | Needs a run shape the task-verify runner reports without failing; use `PlannedRun`'s existing fields | β |
| 5 | Do held requests stay in `unfrozen_suffix` for the needs_rebase conflict graph? | They are already out of `chain_snapshot`. Either way, the twin test states the chosen relation | γ |
