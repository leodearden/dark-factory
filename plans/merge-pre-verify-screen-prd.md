# Merge pre-verify screen and main-red hold ("Stream 1")

**Status:** draft for Leo's approval — 2026-10-30. Not decomposed.
**Slug:** `merge-pre-verify-screen` · **Approach:** B + H (contract and two-way boundary tests; the merge lane is a load-bearing seam).
**Origin:** `plans/mcp-coalesce-trains-2026-10-26.md` §0 (Leo's ruling of 2026-10-29: Stream 1 goes to `/prd`), §2.5, §4.4, §6.1, and its E3 seat output (levers L1–L3).
**Evidence:** five seat outputs in `plans/merge-pre-verify-screen-prd/` (M1 screen premise, M2 red-base premise, C1 verify-path map, C2 red verdict and S1.3 substrate, D owners). Every figure below carries its counting rule there.
**Code anchors** verified against main `07bd68fd99` (2026-10-29). Cited by symbol; re-locate at implementation time.

## 1. Goal

Stop spending merge verifies whose outcome a seconds-long check already knows:

1. **S1.1, the screen.** Before a merge item's verify starts, run a short whole-tree screen on the exact tree the verify would test. An item that fails it on its own defect bounces in seconds, carrying the check's findings, and no verify host is leased for it. The same screen runs in every task-role verify, so most defects never reach the lane.
2. **S1.2, the main-red hold.** While the lane holds a confirmed main-red verdict, the verdict's failing tests join the screen. An item whose tree still fails them on a base that fails them too waits in its lane, with no verify, no bounce and no re-pend. An item whose tree passes them is a fix, or the red was a flake, and verifies at once.
3. **S1.3, hygiene.** `get_changed_line_ranges` gets a canonical branch ref and fails loudly. The workflow-side train former is switched off.

What an operator sees:

- `runs.db` has `merge_attempt` rows with outcome `screen_failed`, and no `merge_verify` row for the bounced request.
- A blocked task's L0 detail carries the screen's `file:line` findings.
- During a red main, `get_merge_halt_status` shows the active verdict (main sha, failing tests) and the count of held requests. After the verdict arms, the only verifies that start on that sha are those whose tree passed the failing tests. Items already dispatched or merged ahead are not covered (§9).

## 2. Background and premise validity (G6)

### 2.1 S1.1: a cheap check already knows a third of all merge-verify failures

Measured over 2026-10-13T21:48Z → 2026-10-29T17:40Z (15.8 days, 630 merge verifies, 147 failed). Source: M1 §2–§4.

| Screen set | Checks | Serial CPU | Failed verifies caught on the item's own defect | Verify time spent on them |
|---|---|---|---|---|
| Ratchet only | `scripts/inline_suppressions.py --check` | 8 s | 50 / 147 (34.0%), 42 tasks | 24.7 h of 63.7 h failed-verify time |
| **Fast seven (this PRD)** | the ratchet + six pytest selections, each < 10 s CPU | ~25 s; **9.8 s wall in parallel** | **58 / 147 (39.5%), 49 tasks** | **28.9 h** |
| Next tier (not now) | + seven 14–31 s checks | ~160 s | 71 / 147 (48.3%) | 34.3 h |

- **Every catch reproduces.** For each of the 83 catches across all tiers, the check is red on the verify's merge tree and green on its first parent.
- **No false alarms in the samples.** The ratchet was red on 0 of 60 sampled passing verifies, and on 0 of 95 sampled lane-landed main tips.
- **The defect is on the branch tip.** In 49 of the 50 ratchet catches the branch tip alone, unmerged, already fails. That is why the task-role screen (decision 2) pays: the defect is visible before the merge request exists.
- **Nothing runs the check earlier today.**
  - The task-role verify is diff-scoped: an `orchestrator/`-only diff never runs the `scripts` leg (`verify_plan.py::_task_role_production_pytest_run`).
  - No hook runs the check.
  - In the merge verify the `scripts` leg runs late, ending about 7.5 min before the verify ends, and the merge verify has no fail-fast, so the other modules keep running after it fails (M1 §1.1).
- **The E3 figures still hold.** The ratchet failed the merge gate for 44 distinct tasks in the window, against E3's 34 in 12 days. The ratchet's failure rate is unchanged since chains went on (3.4 against 3.3 failing verifies a day). The solo failure rate is 24.6% before chains, against 14.7% on n = 34 after.
- **Coalescing going off removes some of these.** 7 of the 50 ratchet catches were coalesce-train unions. Restricted to the 129 non-train failures, the ratchet alone catches 43 (33.3%).
- **A live specimen.** esc-3932-9 (L2): task 3932 thrashed twice at the merge gate on the ratchet, each time after a full suite of 12,136 passing tests.

What the screen does not save: a bounce still costs the workflow an L0 and a steward cycle. Only the verify host time is saved. The task-role screen is what removes the cycle for workflow items, which are about 85% of lane traffic (memo §2.1).

### 2.2 The per-link motive has no live instance yet

There have been 2 chain-tip verifies since `chain_cap: 6` went live; both passed (M1 §2.1). Until one fails, this rests on a prior, not a measurement: 9.5% of verified tasks carried a ratchet defect, so a 3-link chain carried at least one about 25% of the time (M1 §5). The task-role screen will lower that rate. The chain screen (δ) is kept because Leo's §0 ruling names the per-link case as the one that matters most. It is sequenced after the chain-walk fix, 7720.

### 2.3 S1.2: doomed verifies on a judged-red base

Measured over 2026-08-01 → 2026-10-29. Source: M2 §2–§5.

**Doomed starts.** 102 verifies started on a main sha after a `main_health_red` verdict on that same sha, and before main advanced. They used 3,620 host-min; 91 failed (3,247 min).

| Window | Starts | Failed (min) | Rate |
|---|---|---|---|
| before task 4953 | 85 | 77 (3,030) | 4.7% of that window's verify host time |
| 4953 → chains on | 17 | 14 (217) | over 13 days |
| chains on | — | — | no verdict yet |

**Why verifies keep starting.** No stored red verdict exists, and no verify-start path reads one (C2 A1–A2). The only queue effect of a red is the `compile_error` halt of lane `normal`. Every `test_failure` red, which is all of the window's verdicts, halts nothing.

**The verify cannot pass by subtraction.** The "N new failing test(s) not present on main" comparison only labels a verify that has already failed; it never turns a failure into a pass (C2 A4, M2 §5a). Under `merge_verify_breadth: "full"`, every verify on a red sha reproduces the red unless its own tree fixes it.

**The 11 that passed on a judged-red base are the design constraint:**

| Why it passed | n | Would high-lane let it through? |
|---|---|---|
| Declared fix-main task (`merge_lane: high`) | 4 | yes |
| Stacked on a fix's merge commit | 1 | no |
| Diff touched the failing test's file (ad-hoc fix) | 2 | no |
| The red was a flake or host quirk | 4 | no |

So no lane, metadata or diff-reachability proxy separates fixing verifies from doomed ones. Diff reachability fails outright on data-file fixes: the 10-24 codebook red was fixed by restoring a docs entry. Running the failing tests themselves on the item's tree does separate them. All 11 passed their full verify, so they pass the failing tests, which are a subset of it. This is decision 9.

**Lifetimes.** Red shas last a median 107 min, from first verdict to the next main advance. 25 of 36 were ended by a harmless commit while the break persisted. So a hold keyed only on the sha re-learns the red at every docs commit (M2 §4). The F-screen hold is keyed on the failing tests and re-probes bare main at each advance.

**The E3 case reproduces.** Train 5445 started 64 s after 6057's verdict on base `0f443e6c0e` and burned 38.2 min. 28 of those minutes ran after main had already moved to the fix (M2 §3, C2 A3).

### 2.4 S1.3 is dead-path hygiene

- **The bug reproduces** (C2 B8):
  - `git diff main...5925` fails with rc 128 (ambiguous short id), and `get_changed_line_ranges` returns `{}`.
  - `git diff main...5563` silently diffs the unrelated commit `5563a70ca5`.
- **Both callers are behind flags turned off:**
  - `workflow.py::TaskWorkflow._maybe_form_train` sits behind `merge_train_former_enabled`, which this PRD turns off.
  - `worker.py::SpeculativeMergeWorker._stack_coalesce_train` sits behind `merge_train_coalesce_enabled`, which carrier 7739 turns off.
- **Declared atomic trains don't read the former flag.** Their membership comes from task-authored `metadata.train`, and `_maybe_form_train` is reached only when that is absent (C2 B7).

## 3. Resolved design decisions

1. **Where the screen runs: per merge item, not at lane admission.** The screen runs at three dispatch-preparation sites:
   - `merge_lane/worker.py::classify_and_merge`, after the plan-targets drop guard (`gates.py::_check_plan_targets_in_tree`, the template). This covers slot 1, slot 2 and re-merges.
   - `merge_lane/worker.py::_do_train_merge`, after `merge_to_main` and before `_leased_train_verify`. This covers declared trains.
   - `SpeculativeMergeWorker._deep_chain_placement`, after `build_chain`. This covers chain tips (decision 7).

   Admission screening was rejected:
   - It sees a stale branch-level tree.
   - It would stall the serial merger on `_acquire_next_request`.
   - The task-role screen gives the branch-level verdict earlier and better (C1 §7, rows A and B).

2. **The task-role verify runs the same screen** (Leo, 2026-10-29). `verify_plan.py::derive_verify_plan` adds the screen run to every task-role plan, whatever the diff scope. A finding fails the task verify, and the implementer sees it in its own verify–review loop.
3. **What the screen runs: the fast seven** (Leo, 2026-10-29), M1 ids:

   | M1 id | Check |
   |---|---|
   | C1 | the inline-suppression ratchet |
   | C8 | the remote-gate wire fingerprint |
   | C11 | watchdog fake-systemctl drift |
   | C14 | subproject test collection |
   | C15 | uv-run venv isolation |
   | C18 | recovery-veto predicate collapse |
   | C20 | disjoint-skip / breadth yaml guards |

   - The list lives once, in a project wrapper, `scripts/merge_screen.py` (heuristic 11, SPOT; INV-5).
   - The orchestrator holds only the command that runs the wrapper, so orchestrator code stays project-agnostic (heuristic 9, deep module, narrow interface). For reify the command is unset and nothing changes.
   - The next tier (14–31 s checks) waits for the readout (ζ).
4. **The screen provisions its environment exactly as the verify does: `uv run` in the screened tree.**
   - There is no interpreter or `PYTHONPATH` borrowing from the dispatcher, so there is no skew between the dispatcher's packages and the tree's.
   - Measured in a fresh worktree: a cold `uv sync --all-packages --frozen --offline` takes 1.6 s, hardlinked from the uv cache, and all seven checks pass in 9.8 s wall run in parallel, at load average 120.
5. **The screen's verdict is structured data, never an exit code alone** (heuristic 12; INV-2; INV-11).
   - The wrapper's `--json` mode prints a `merge-screen/1` report: one entry per check, with status `pass`, `finding` or `error`, the findings, and the duration. It exits 0 whenever it ran.
   - Why not trust exit codes: under `python -I` an import failure exits 1, which is the scanner's finding code (measured). The scanner's own `--json` report (`schema_version` 2, `status`, `violations`) is how the wrapper reads it.
   - For a pytest check, exit 1 is a finding. Any other non-zero exit (collection error, interruption, nothing collected) is an `error`.
6. **Attribution is base-first.**
   - A check red on the merged tree is re-run on the base (the merge commit's first parent). Each base verdict is cached per sha, and a base needs its own checkout only when an item is red.
   - Red on the item only: the item bounces.
   - Red on both: it is not the item's defect, so the item is not bounced. Under α it proceeds to its verify as today. Under γ, when the base is the real main sha, it arms a main-red verdict (decision 9).
   - The hazard is measured (M1 §2.5). On 10-19 a red direct-to-main commit sat under six unrelated items, and an item-only screen would have blamed all six.
7. **A screen bounce is an ordinary per-item outcome.**
   - Its shape: `MergeOutcome('blocked', reason=<findings>, failure_category='merge_screen', failure_cause_hint=<failing check ids>)`, emitted with a new `OutcomeKind.screen_failed`.
   - It is delivered as the drop guard's `Decided` outcome is, through the existing generic blocked path: an L0 for a workflow request with the findings as detail, and the reason for an MCP caller.
   - It never reaches `_run_post_merge_verify`, so it does not feed the main-health probe, the dry-run, the flake ledger, skew classification or chain halving (C1 §5).
   - It does feed the workflow's merge-thrash guard. It is a real defect of that item, so two identical bounces escalate to a human, as two identical verify failures do today.
   - The workflow picks the review category from `failure_category`, not from the reason text.
8. **An instrument failure fails open, loudly** (INV-4, INV-11).
   - A wrapper crash, a sub-check `error` or a timeout never bounces an item and never fails a task verify: the verify proceeds as today.
   - The lane emits a structured `merge_screen` event with status `error`.
   - A streak of `merge_screen.error_streak_alarm` consecutive errors files one info L1, `merge_screen_unavailable`. The next clean screen closes it.
9. **S1.2 is an F-screen hold** (Leo, 2026-10-30).
   - **Arming.** A confirmed main-red verdict arms a `MainRedVerdict`, holding the main sha, the failing test ids F and the break signature.
     - Its facts come from the one builder both probe modes share, `worker.py::_build_main_health_outcome`, which gains structured probe-sha and F fields.
     - The worker arms it only after re-checking the probe sha against current main (INV-3).
   - **Screening.** While it is armed, every screen (decision 1) also runs F on the screened tree:
     - F passes: the item verifies now. It is a fix, or the red was a flake.
     - F fails on the item and on bare main: the item is **held**. It goes back to its lane buffer unpickable, with no verify, no outcome, no status write and no re-pend.
     - F fails on the item only: it bounces under decision 7, because the item broke F.
   - **Clearing.** The verdict clears when F passes on bare main. Bare main is re-probed at each main advance the worker observes, and at most every `merge_main_red_hold.reprobe_interval_secs` otherwise, so a flaky verdict clears itself. Clearing releases every held request.
   - **Static checks too.** A static screen check that is red on the real main sha arms a verdict as well, with source `screen` and F being that check.
   - **Scope.** It applies to every verify start that decision 1 screens. High lane keeps only its existing meaning: fixers are popped first.
   - **Limits.**
     - A verdict with no test ids, or more than `max_failing_ids` of them, arms nothing, and behaviour is as today. That was 1 of 109 verdicts in the window.
     - The existing `compile_error` halt is unchanged.
     - A restart drops the verdict, and the first failing verify re-arms it (INV-7: the hold expires coherently).
10. **S1.3: one canonical ref, loud failure.**
    - `GitOps.get_changed_line_ranges` builds its ref with `merge_lane/types.py::QueuedBranch.parse(ref, branch_prefix).full_name`, the existing canonical parser (heuristic 11).
    - A non-zero rc raises a typed error naming the ref, the rc and stderr (INV-11).
    - The rc check lives in a new helper, which keeps the method's ratchet measure at 0 (C2 B8 gotcha).
11. **The former goes off on this PRD's carrier** (Leo, 2026-10-29), following 7739's precedent.
    - It is a one-line `dark-factory-orchestrator.yaml` edit with its ruling comment. It is restart-only, and it goes live at the next fleet restart.
    - Deleting the former's code stays with task 5375, whose "wire or retire" question this answers for now.
12. **The chain canary measures stay with the deep-merge PRD** (Leo, 2026-10-29): η1 3190, θ 3192, and the 14-day readout 5059. D §3 lists the coverage gaps for measures 1, 3 and 4 for that program's owner; this PRD files nothing for them.

## 4. Pre-conditions (G3)

Each assumed capability, verified by reading on main `07bd68fd99`, or built here:

| Capability | Status | Where |
|---|---|---|
| A merged tree checked out before any verify dispatch, for every item kind | exists | `RealMergeItem.merge_wt` from `GitOps.merge_to_main`; `ChainResult.lane` at `chain.tip`; the train's `merge_wt` (C1 §6) |
| Pre-verify bounce template: a `Decided` blocked outcome plus worktree cleanup, not delivered early for speculative items | exists | `classify_and_merge` with `_check_plan_targets_in_tree`; `OutcomeKind.dropped_plan_targets` |
| `OutcomeKind` and `ChainTruncationReason` are extendable enums/literals | exists | `merge_lane/types.py::OutcomeKind`, `::ChainTruncationReason`, `::CHAIN_FAULT_REASONS` |
| The scanner runs in a venv-less tree with plain `python3` and emits a typed report | exists, measured | `scripts/inline_suppressions.py` binds its own checkout via `scripts/_checkout.py`; `--json` gives `schema_version` 2 |
| A throwaway worktree at any sha, for base screens and bare-main probes | exists | `GitOps.ephemeral_worktree(kind, sha)`. A new `WorktreeKind.MERGE_SCREEN` member is added; the enum is the single source for naming and reaper protection |
| Run named pytest node ids on a given worktree and classify each | exists, wired in production | `verify.py::confirm_isolated_rerun_verdict` (main-probe and merge-gate call sites), `_merge_gate_isolated_rerun`; bare-main worktree via `verify.py::_on_main_probe_worktree` |
| The verdict's failing ids and probe sha at confirmation | exists inside the probe, not exported | `verify_failure_is_preexisting_on_main` returns `(True, probe_sha)` from the failing verify's ids. γ adds them as structured fields on the MAIN_RED outcome built by `_build_main_health_outcome` |
| Lane-level "not pickable" with an event-gated wait | exists | `_pop_next_pickable`, `chain_snapshot`, `_wait_until_any_lane_runnable` |
| The request's lane (`normal` / `high`) | exists | `MergeRequest.lane` |
| Operator halt surface | exists, extended here | `harness.py::Harness.get_merge_halt_status` returns `{wired, halted, owner_esc_id}`; γ adds the verdict and the held count |
| Task-role verify plan seam | exists | `verify_plan.py::derive_verify_plan` |
| Canonical branch-ref parser | exists | `QueuedBranch.parse(...).full_name` |
| Hot-reload plumbing for the new knobs | exists | `config.py::RELOADABLE_FIELDS` |

## 5. Cross-PRD relationships and seam owners (G4)

| Seam | Direction | Mechanism | Owner |
|---|---|---|---|
| `plans/deep-merge-ahead-prd.md` | extends | δ adds a `screen_failed` chain truncation. Decision 4 is kept: a truncated link gets no outcome. Decision 5 is kept: only a tip verify halves depth. Deep failures still never feed the thrash guard | **this PRD** owns δ; deep-merge owns chain build, landing and the canary |
| task 7720 (chain walk lands 0 links) | δ depends on it | δ is built against a working walk | 7720 |
| task 7259 (`chain_truncated_reason` on the chain-arm event) | δ depends on it | δ's truncation becomes observable through 7259's field | 7259 |
| task 7721 (chain build latency) | coordinates | δ's tip screen runs outside `CHAIN_BUILD_TIMEOUT_SECS`, with its own bound | 7721 owns the build budget |
| Main-health classification (1690, 3597, 5627) and fix ownership (5471) | consumes | γ arms from the shared outcome builder. It does not change `verify_failure_is_preexisting_on_main`, `is_auto_heal_eligible` or `MainFixTaskFiler`. Ownership of a red (fix-main task, L1) stays where it is | main-health; **this PRD** owns the verdict record and the hold |
| Verify-base guard ruling (3206, 3981) | distinct | γ judges base *health* (failing tests), not base *identity* (frozen prefix). The advisory-by-decision ruling is untouched | — |
| `plans/orchestrator-atomic-train-merge-prd.md`; task 6755 (train verify parity) | consumes, coordinates | α screens at `_do_train_merge`. A finding is a blocked outcome *without* `TRAIN_VERIFY_FAILED_REASON_PREFIX`, so the tip workflow escalates to a human. A γ hold uses `_stand_down_train`. α and 6755 both edit `_do_train_merge` | **this PRD** owns the screen call; 6755 owns the verify binding |
| INV-12 inline-suppression gate (5611) | consumes | The wrapper reads the scanner's `--json` report and does not change its contract | INV-12 PRD |
| task 5375 (former deployed with 0 events) | answers | The flip rides this PRD's carrier. Code deletion stays with 5375 | 5375 |
| task 7739 (coalesce off) | independent | Both flips are restart-only and go live at the same next fleet restart | 7739 |

## 6. Contract (B + H)

```
# scripts/merge_screen.py (project-owned; the only list of screen checks)
merge_screen.py --root <tree> --json           -> stdout: ScreenReport JSON, exit 0 if it ran
merge_screen.py --root <tree> [--only <id>...]  -> human report; exit 0 clean, 1 finding, 2 instrument error
ScreenReport = {schema: "merge-screen/1",
                checks: [{id, status: pass|finding|error, findings: [str], duration_s}]}

# orchestrator/src/orchestrator/merge_lane/screen.py (new; imports nothing from worker.py)
@dataclass(frozen=True) ScreenCheck   {id, status, findings: tuple[str, ...], duration_s}
@dataclass(frozen=True) ScreenVerdict {tree_sha, checks: tuple[ScreenCheck, ...]}
    .findings_on_item / .errors / .passed
async run_screen(tree: Path, config, *, only: frozenset[str] | None = None,
                 node_ids: MainRedFailingIds | None = None) -> ScreenVerdict
async screen_item(merge_tree: Path, base_sha: str, config, verdict: MainRedVerdict | None)
    -> ScreenDecision = Proceed | Bounce(outcome) | Hold(verdict_generation)

# orchestrator/src/orchestrator/merge_lane/main_red_verdict.py (new; one owner: the worker)
@dataclass(frozen=True) MainRedVerdict {main_sha, failing_ids, signature, source: probe|screen,
                                        confirmed_at, generation}
class VerdictBook: arm(v) / clear(reason) / current() / note_main_advance(sha) / held_request_ids()
```

Invariants:

- **C1, kill switch.** `merge_screen.command` empty ⇒ no screen code path executes on any site, and behaviour is byte-identical to today (asserted by test). `merge_main_red_hold.enabled: false` ⇒ no verdict is armed.
- **C2, attribution.** `Bounce` only when some check is `finding` on the merged tree and `pass` on the base. `finding` on both is never a bounce.
- **C3, fail open.** Any `error` or timeout ⇒ `Proceed` plus a `merge_screen` event with status `error`. Never `Bounce`, never `Hold`.
- **C4, no verify on a bounce.** A `Bounce` request has no `merge_verify` row and no verify-host lease.
- **C5, hold rules.**
  - `Hold` only while a verdict is armed, F fails on the item, and F fails on the base, which is the real main sha.
  - A held request keeps its future pending and its task status untouched, and sits in its lane buffer.
  - The stuck-request ledger (`merge_lane/request_ledger.py`) must not age it as stuck. γ ends its dequeue episode when it returns to the buffer; this is unverified today and has a boundary test.
  - Every held request is released on `clear`.
- **C6, chain truncation.** At least one screen runs on the chain tip. On a finding, the chain is truncated before the first link whose cumulative prefix fails, using a bounded search over the link commits. That link gets no outcome. `_note_chain_outcome` sees only tip-verify results.
- **C7, one home for the verdict.** Only `VerdictBook` holds it (INV-9). Its events (`main_red_verdict` armed/cleared, `merge_held`) carry the sha and F, unlike today's sha-less `main_health_red` row. `get_merge_halt_status` reads it from the book.

## 7. Boundary-test sketch (B + H)

| Scenario | Preconditions | Postconditions |
|---|---|---|
| Own-defect bounce | branch adds an unbaselined `# type: ignore`; base clean | outcome `screen_failed`, reason names `file:line` and check id; no `merge_verify` row; L0 detail carries the finding (workflow); MCP reason carries it |
| Red base is not blamed | the same finding on base and item | no bounce; with γ off the item verifies; with γ on and base == main, a verdict is armed (source `screen`) and the item is held |
| Instrument error fails open | wrapper crashes / times out | item verifies; `merge_screen` event status `error`; K consecutive errors ⇒ one `merge_screen_unavailable` L1, closed by the next clean screen |
| Kill switch | `merge_screen.command` unset | byte-identical dispatch (golden transcript) |
| Task-role catch | orchestrator-only diff with an unowned suppression | task verify fails with the scanner finding; a `scripts/`-free diff still runs the screen |
| Fixer verifies at once | verdict armed (F = {t}); item's tree passes t | verify starts; lands; verdict re-probed on bare main at the advance; cleared; held items released |
| Doomed item is held | verdict armed; item fails t; bare main fails t | no verify, no outcome, status unchanged; `merge_held` event; visible in `get_merge_halt_status` |
| Item broke F itself | verdict armed; item fails t; bare main passes t | bounce `screen_failed` naming t |
| Flaky verdict clears | verdict armed; re-probe passes t with no advance | cleared at the re-probe interval; held items released |
| Stacked on a fix | slot-2 item whose base is a high-lane fix's merge commit | its tree passes t ⇒ proceeds (no hold) |
| Restart | verdict armed, items held | after restart: no verdict, held requests re-merge normally; first failing verify re-arms |
| Held request is not stuck | verdict armed; a request held longer than the stuck threshold | no merge-request-stuck alarm for it; `get_merge_halt_status` lists it as held |
| Chain link defect | 5-link chain, link 3 carries a finding | chain truncated to links 1–2 before the tip verify; `chain_truncated_reason = screen_failed`; link 3 no outcome; halving state unchanged; link 3 bounced later at its own per-item screen |
| Declared train finding | train tip tree carries a finding | untagged blocked outcome ⇒ tip workflow human L1; no union verify |
| S1.3 ref | branch `task/5563` exists, and an unrelated commit abbreviates to `5563` | `get_changed_line_ranges('5563')` diffs `task/5563`; a missing ref raises naming ref, rc and stderr |

## 8. Decomposition plan

Leaves follow the overlay sizing bands. Every leaf touching `merge_lane/worker.py` is serialized by a dependency edge (same-file rule). `worker.py` is 23k lines, far past heuristic 14's alarm, so new mechanisms go in new modules (`screen.py`, `main_red_verdict.py`). Each of those must make sense in isolation and import nothing from `worker.py` (heuristic 13: layering is a downward import, never a satellite over the parent's private state). `worker.py` gains call sites only.

- **α: the screen and the per-item lane screen** (high).
  - Builds `scripts/merge_screen.py` (the seven checks, `merge-screen/1`), `merge_lane/screen.py`, and the `classify_and_merge` and `_do_train_merge` call sites.
  - Adds `OutcomeKind.screen_failed`, the `merge_screen` event and `WorktreeKind.MERGE_SCREEN`, plus the config block `merge_screen.{command, timeout_secs, error_streak_alarm}` (reloadable, default off). Enables it in this repo's yaml.
  - Adds a guard test that *executes* each wrapper check's selection against the tree and fails if any selects nothing (INV-10; a renamed test must not silently empty the screen).
  - Modules: orchestrator, scripts. ~1,000 LOC.
  - **Signal:** the boundary rows "own-defect bounce", "red base is not blamed" (γ-off arm), "instrument error", "kill switch" and "declared train finding".
  - Live confirmation: a `merge_attempt outcome=screen_failed` row in runs.db with no `merge_verify` row for that request.
- **β: the task-role screen** (high). Depends on α.
  - `verify_plan.py::derive_verify_plan` adds the screen run to every task-role plan. An `error` is a non-failing note, never a task-verify failure (decision 8).
  - Modules: orchestrator. ~300 LOC.
  - **Signal:** the boundary row "task-role catch". Live: a task-role verify of an `orchestrator/`-only diff shows the screen run in its plan file (`data/verify-logs/<task>/attempt-*.plan-*.json`).
- **γ: the main-red verdict and F-screen hold** (medium). Depends on α (same file).
  - Builds `merge_lane/main_red_verdict.py`. The verdict is armed from `_build_main_health_outcome` (structured probe sha and F), and from a static screen check red on real main.
  - The F run reuses the isolated-rerun node-id runner. Hold, release and re-probe are wired into `_pop_next_pickable` / `chain_snapshot`, the dispatch pre-check, and the train branch (`_stand_down_train`).
  - Extends `get_merge_halt_status`. Adds the config block `merge_main_red_hold.{enabled, reprobe_interval_secs, f_screen_timeout_secs, max_failing_ids}` (reloadable).
  - Modules: orchestrator. ~1,000 LOC.
  - **Signal:** the boundary rows "fixer verifies at once", "doomed item is held", "item broke F itself", "flaky verdict clears", "stacked on a fix", "restart", "held request is not stuck" and the γ-on arm of "red base is not blamed".
  - Live confirmation: the next red episode shows a `main_red_verdict armed` event with sha and F. Every merge verify that starts on that sha after it belongs to an item whose `merge_screen` event shows F passing.
- **δ: the chain-tip screen** (medium). Depends on γ (same file), 7720 and 7259.
  - Screens the tip in `_deep_chain_placement` after `build_chain` (with γ's F when a verdict is armed), searches for the culprit prefix, and truncates with `screen_failed` (added to `ChainTruncationReason` and `CHAIN_FAULT_REASONS`). Bounded separately from the build timeout.
  - Modules: orchestrator. ~400 LOC.
  - **Signal:** the boundary row "chain link defect".
- **ε: canonical ref and loud failure in `get_changed_line_ranges`** (low; `complexity='simple'`). No dependencies.
  - Modules: orchestrator (`git_ops.py` and its tests). ~80 LOC.
  - **Signal:** the boundary row "S1.3 ref", run in a real-git fixture.
- **ζ: readout, 14 days after β and γ land** (medium). Depends on β and γ. A `metadata.milestone` delayed by 14 days, anchored when its last dependency lands; it has a condition and a trigger, per standing practice.
  - Commits a repeatable measurement script under `scripts/` and a dated readout under `plans/`. The readout reports:
    - screen bounces, and the verify-minutes they replaced;
    - false bounces: a bounced item whose next unchanged verify passed;
    - task-role catches;
    - `error` streaks;
    - verdicts, holds and held-minutes, and fixers passed;
    - chain truncations, if δ has landed.
  - It decides, or files a declined record for: the 14–31 s check tier; whether E3's L2 (fail-fast leg reordering) is still worth a measurement task; and verdict persistence across restart.
  - **Signal:** the readout file committed on main, with each of the three decisions filed or declined.

The former-off flip is not a leaf. It rides this PRD's carrier (decision 11).

**G7 (advisory walk, author mode)**, against `docs/legibility/design-invariants.md`:

| Invariant | How this design meets it |
|---|---|
| INV-2 | the bounce carries structured findings and check ids; the verdict events carry sha and F |
| INV-3 | arming re-checks the probe sha against main; clearing is a fresh probe |
| INV-4 | the error-streak alarm |
| INV-5 / INV-9 | one check list (the wrapper) and one verdict home (`VerdictBook`) |
| INV-6 | holds write no status |
| INV-7 | the hold's owner is the worker; its exits are clearing and restart; it is visible in `get_merge_halt_status` |
| INV-8 | the screen is an awaited subprocess with a timeout, so it never blocks the loop thread. The serial merger task waits about 10 s per item. During a red episode the F run can hold it for up to `f_screen_timeout_secs` per item, when only fixers can land, and high-lane fixers are popped first |
| INV-10 | the selection guard executes each check's selection |
| INV-11 | an `error` is distinct from `pass`, and S1.3 raises |
| INV-12 | the design adds no suppressor |
| INV-13 | the verdict's producer is the existing probe, which fired 109 times in the window; `get_merge_halt_status` reports "no verdict" explicitly |
| INV-1 and INV-14 | no hits |

No waiver needed.

## 9. Out of scope

| Item | Why |
|---|---|
| Reordering verify legs to fail fast (E3 L2) | Interacts with the "not present on main" comparison, which needs a complete failing set; ζ decides whether to measure it |
| The 14–31 s and 37–52 s check tiers | ζ decides from live data |
| Gate re-verify after a rebase (`gates.py::_reverify_rebased_tree`) and declared-train member solo re-verifies | Both follow a verify that already passed; a small, separate population |
| Aborting in-flight verifies when a verdict arms; protecting the ≤ 2 items already merged ahead; the pre-verdict window (M2 §2d) | The first wasted verify of an episode is what discovers it |
| Persisting the verdict across restart | Costs one verify per restart per episode; ζ decides |
| Coalesce-only work: unwind on exceptions and `stop()` (6450), `has_workflow_waiter`, Stream 2 | Moot while coalescing is off |
| The chain canary measures and their gaps | Deep-merge PRD (decision 12) |
| The rc-discard pattern in the other 10 `GitOps` methods (review-all 2026-10-08) | ε fixes the one with a live wrong answer; not a sweep |

## 10. Open questions (tactical)

1. **`merge_main_red_hold` defaults.** Suggested: `reprobe_interval_secs` 900, `f_screen_timeout_secs` 300, `max_failing_ids` 25 (disk-pressure mass failures carry 45–159 ids). Decide in γ from the probe's own timings.
2. **`merge_screen` defaults.** Suggested: `timeout_secs` 180, `error_streak_alarm` 5. Decide in α.
3. **Culprit search in δ.** Bisect, or a linear prefix walk? The bound is the contract (C6); pick by chain-lane checkout cost. Decide in δ.
4. **β's non-failing `error` note.** It needs a run shape the task-verify runner can report without failing. Decide in β from `PlannedRun`'s existing fields.
