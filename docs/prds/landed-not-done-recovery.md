# PRD — landed-but-not-done recovery: non-decaying landing evidence + reachable recovery

**Status:** active · authored 2026-08-23 · dark_factory · approach **B + H**
**Verified against:** main @ `e0c859f566` (investigation) / `2dc87b3fb8` (revert measurement).
**Corrected at decompose 2026-08-23 against main @ `d8f165756b`** — see §Post-authoring corrections.
Every file:line anchor below was re-read on main during authoring; the `git_ops.py` block has since
drifted ~+1,000 lines and is corrected inline. Re-verify before editing (anchors drift).
**Source:** investigation `~/.claude/spawn-briefs/landed-not-done-recovery-gap-2026-08-22.md`, worked by
a 6-agent team (session deb-df-2278054) plus a 2-repo historical revert measurement. Leo-ratified
2026-08-22 (option set 7/5/4/1/6; option-1 shape pivoted on the measurement).

## Goal

A task whose work is provably on `main` reaches `done` **without a human**, and stays reachable no
matter how long it has been stranded or which non-terminal status it is parked in.

What an operator observes after this PRD:

- A landed task sitting in `pending`, `blocked`, or `merge-deferred` self-heals to `done` with
  attributed provenance on the next reconciler pass — including tasks stranded for weeks.
- A landing that is a **no-op** (merge marker present, empty net branch diff) is **never** stamped
  done — it re-dispatches instead.
- When a recovery is declined, the reason is visible in the recovery-disposition record instead of
  the gate returning silently.

## Post-authoring corrections (decompose walk, 2026-08-23, main `d8f165756b`)

Recorded rather than silently patched, because two of them change measured magnitudes this PRD
argues from. **No resolved design decision (D1–D8) changes direction; three change their stated
basis.**

1. **Task 3116 landed 12h40m after this PRD's anchor sha and rewrote the predicate §Background
   argues against.** `40c39cd8ee` (2026-08-22T20:41) is *not* an ancestor of `e0c859f566`
   (08-22T08:01). `commit_effect_present_in_main` is **no longer byte-identity**: since 3116 it is a
   threshold added-line **survival** test (`_EFFECT_SURVIVAL_AGGREGATE_THRESHOLD = 0.98`,
   `_EFFECT_SURVIVAL_PER_FILE_THRESHOLD = 0.90`, floor 25 added lines —
   `git_ops.py:1113/1121/1128`). 3116's own full-corpus measurement (2,827 merges, 2,822 usable):
   of 2,680 previously-rejected merges it now **accepts 1,050 (39.2%)**, leaving a **60.8%
   residual**. The `"the cost of a false negative here is a re-check"` docstring §"The accepted-risk
   premise that failed" refutes has already been deleted — 3116 replaced it with this PRD's own
   conclusion. **D1's direction survives on the residual** (≈1,630 still-absent verdicts against
   2 genuine reverts in 5.4 months), but the 95.5% / 0.04% arithmetic is pre-3116 and must be
   re-derived by leaf δ before ε removes anything.
2. **`branch_content_in_main` is unchanged and still byte-identity** (`git_ops.py:9155`, ends in
   `git diff --quiet`). The §Background decay bullet is therefore half-right: it holds for
   `branch_content_in_main`, not for `commit_effect_present_in_main`.
3. **The sync-merge misread is untouched by 3116** — the `merge-base(first_parent, other_parent)`
   anchor rule is preserved verbatim (`git_ops.py:9406-9435`), so B2 is genuinely new work. Its
   postcondition is the PRD's stated one — *the ~300-file main-history diff is never computed* —
   not "accepted flips to True"; under survival semantics that set now often passes **accidentally**,
   attesting that main's history survives in main rather than that the task's work landed.
4. **`orchestrator/src/orchestrator/landing_evidence.py` already exists** (1,096 lines, task 2678)
   and already exports `LandingEvidenceVerdict` (`:189`), `validate_landing_evidence` (`:527`),
   **`branch_is_degenerate` (`:137`)** — which is boundary row B6's predicate, already wired at five
   production sites — and `is_valid_sha_40` (`:122`). §Contract below reads as a new-file spec; it is
   not. δ **extends** that module and `branch_work_landed` must be its **single** producer, with
   `validate_landing_evidence` re-expressed as a thin mode over it. Two verdict types answering one
   question in one module is the "two disagreeing authorities" shape §Sketch rejects for `_RECOVERY`.
5. **There are four producers and seven consumers, not "three call sites".** Beyond
   `validate_landing_evidence`'s seven consumers (`harness.py:5702/:11671/:11713/:11741`,
   `merge_queue.py:14241`, `escalation/server.py:3506/:3552`), `task_ground_truth.py:455-480` holds a
   hand-mirrored fourth producer inside `derive_truth`. `merge_queue.py:14241`
   (`_redrive_coalesce_members`) is an **eighth landing-detection site** D1's "dispatch gate,
   Tier-3.5" scope limit does not name — δ must rule it explicitly in or out.
6bis. **The Contract's `LandingVerdict` is under-specified: it has no `probe`.** Two of the three
   dispatch-gate arms (`harness.py:11719`, `:11747`) pass their verdict to
   `_file_unattributed_landing_escalation` → `landing_evidence.file_unattributed_landing_escalation`
   (`:1014`) and `format_unattributed_landing_detail` (`:740`), whose bodies render `verdict.probe`,
   `_render_effect_divergence(verdict)` and `_render_delivered_checks_differential(verdict)`.
   Repointing those arms at a four-field `LandingVerdict` would **silently empty the L1 escalation
   body** — an `structured-facts-at-failure` regression. δ must carry a `probe` mapping forward (or
   add a `LandingVerdict` arm to both renderers); ε must not repoint an escalating arm until it does.

6. **D6's ratified half is right; its stated escape hatch does not exist.**
   `already_landed_gate` is correctly absent from `STREAK_CHARGING_SITES`
   (`recovery_emission.py:163-167`) — keep it that way. But `veto_streak_min_span_secs` is an inner
   condition of `emit_recovery_veto_streak_escalation`, which is only ever called behind
   `if site in STREAK_CHARGING_SITES` (`harness.py:9755`), so it **never runs for a non-charging
   site**. A tick-rate site therefore has no bound today; the owning leaf must supply one.
7. **D7's fold is in-progress-gated and would defeat B11.** `_resolve_live_claimant`
   (`task_ground_truth.py:520`) resolves its DB-claimant leg through `is_stranded`, which returns
   `False` unconditionally when status != `in-progress` (`task_claimant.py:129-131`). Its own
   docstring (`:547-560`) states the consequence: a non-in-progress task with a crash-left
   `claimant_run_id` *"is treated as LIVE here regardless of heartbeat age"* — pinned by
   `test_stale_db_claimant_on_blocked_task_is_treated_as_live_by_design`. η's entire population is
   non-in-progress, so D7 as literally specified skips every landed parked task forever: **B12 greens
   while B11 reds.** A partial substrate exists and is wired —
   `shared.task_claimant.has_live_claimant` (`task_claimant.py:187`, explicitly status-agnostic, used
   by `_eligible_for_dispatch` at `scheduler.py:4934`) — and it returns the *correct* answer for both
   B11 and B12, but it folds the **heartbeat leg only**, not `scheduler.is_actively_held` or the
   `plan.lock` leg, so composing the other two inside η would be a fourth copy of the fold (INV-5).
   The full status-agnostic predicate `is_stranded_any_status` has **0 hits in any `.py`** (all 14
   repo-wide hits are PRD prose) — it is task **4618**, and the `_resolve_live_claimant` repoint is
   task **4623**, both pending, both leaves of `claimant-invariant-enforcement`. **η therefore takes
   an out-of-batch dependency on 4623**, and §Pre-conditions' "None blocking" is corrected below.
   *If an operator would rather ship η early, the reversible alternative is to drop that edge and have
   η call `has_live_claimant` directly — weaker, but correct in direction for both B11 and B12.*
8. **D8's conclusion holds; its stated mechanism is wrong.** `_found_on_main_response` (`:3195`),
   `_git_authority_task_metadata` (`:3240`) and `merge_status` (`:3287`) are all at 4-space indent —
   **siblings inside `create_server` (`:596`)**, not closures inside `merge_status`. They still close
   over `create_server`'s locals, so they remain un-importable and the extraction is still required.
   But `_git_authority_task_metadata` **already has a caller outside `merge_status`** —
   `merge_request` at `:2293` — and `escalation/tests/test_merge_status_git_authority.py` contains
   **zero** references to `merge_request`. ζ's behaviour-preservation surface is two tools, and the
   second is unpinned.

9. **The η↔3539 "conflict" was a misreading, and is retracted (ruled by Leo 2026-08-24, gate task
   4673 / `esc-4673-1`).** The decompose walk recorded leaf η and task 3539 as targeting the same
   population with incompatible remedies. They do not. `_RECOVERY` is read only by
   `task_ground_truth.py::classify_recovery`, reached only through
   `task_ground_truth.py::TaskGroundTruth.recovery_for`, whose sole production caller is
   `harness.py::Harness._reconcile_one_stranded` — entered only for statuses in
   `harness.py::_RECONCILE_SWEEP_STATUSES` (`frozenset({'in-progress', 'blocked'})`), filtered
   upstream by the status loop in `harness.py::Harness._reconcile_stranded_in_progress`.
   (Line anchors for all of these are measured and SHA-stamped in correction 11.) 3539's rows are **all keyed `IN_PROGRESS`**; η owns `pending` and
   `merge-deferred`, which never reach that table. §Sketch's objection to "adding rows to
   `_RECOVERY`" is a claim about rows keyed on *parked* statuses and therefore never reached 3539.
   3539's own 2026-08-22 amendment says the same thing unprompted: *"the recovery table is a second,
   independent hole."* **The ratified partition:**

   | Owner | Population | Action |
   |---|---|---|
   | **η** (task 4651) | landed `pending` · `merge-deferred` | mark done with attributed provenance |
   | **β** (task 4645) | landed `blocked` | the existing sweep-side upgrade, veto narrowed |
   | **3539** | `in-progress` + escalation-pinned | `CONVERT_TO_BLOCKED` rows; plus the `plan_files_not_touched` already-landed carve-out in `merge_gates.py` |

   No two-authorities problem: η's only write is `done`, a **terminal absorbing state**, so it cannot
   oscillate against 3539's anti-churn work. The one adjacency is a *handoff* — a task 3539 converts
   `in-progress`→`blocked` lands in the sweep's blocked arm, which is β's territory.

10. **3539's unexplained re-pender is identified.** 3539 records that after a pin lifted on task 3717,
   *"within ~15 minutes the row was status=PENDING … Something re-pended it"*, and names an
   unconfirmed reconcile-sweep `REVERT_TO_PENDING` as the candidate. It is the mark-done applier's
   **own reject arm**: `harness.py::Harness._reconcile_one_stranded` calls
   `validate_landing_evidence(…, candidate_sha=…)` and, on `not verdict.accepted` with
   `status == 'in-progress'`, calls `harness.py::Harness._revert_in_progress_if_no_live_claimant`,
   which flips the task to `pending`. 3717's branch had been re-seeded from main while
   its work landed 2026-08-08, so the then-byte-identity effect check failed. There is no second path.
   The loop's **entry** is decayed landing evidence — leaf δ's subject — and the reject's silence at
   that site is task 4496's.

11. **Recheck 2026-08-24 (five proposed amendments, main `ea876cb624`).**

   > **Code anchors** in this correction verified against main `ea876cb624` (2026-08-24). Main moves
   > fast — cite-by-symbol elsewhere; re-locate these lines at implementation time. This correction is
   > deliberately line-anchored because re-measuring the ruling's anchors *is* its subject
   > (`CONTRIBUTING.md` §"Cite code by symbol" escape hatch). Do not re-anchor it later: freeze it and
   > move any still-live claim out.

   An adversarial review of the 4673 ruling proposed five amendments; three were applied, two
   rejected on gates. **The ruling's own
   line anchors were re-measured and three were stale:** `_reconcile_one_stranded` is defined at
   `harness.py:5479` and its `recovery_for` call — the only production one — is `:5553` (cited as
   `:5540`); the status filter is `:4959-4960` (cited as `:4947`); the in-progress revert arm is
   `:5728-5731` (cited as `:5721-5723`, which is the `logger.warning`). `harness.py:238` is correct as
   cited. Corrected above. **The structural claim survives in its exact form:** `_RECOVERY` is read
   only by `classify_recovery` (`task_ground_truth.py:916`), whose only production reach is inside
   `_reconcile_one_stranded`. Task 4673 is `done`; its ruling text is left as filed (audit record).

   **(b) η is not the sole `done`-writer over its own population — and the discriminator is not
   `metadata.train`.** Three pre-existing writers already stamp `done` on a task in η's statuses:
   - `merge_queue.reconcile_landed_row` **RC-2** (`merge_queue.py:6685`, `kind='merged'`). RC-3 at
     `:6616` short-circuits every `WORKFLOW_PRESERVE_STATUSES` member, so RC-2's reachable population
     is exactly `pending` (and `in-progress`). It fires only for a task still holding a
     **LandedOutbox row** — the merge crash window, and `:6757` already calls RC-3 *"a PURE
     crash-recovery backstop"*.
   - `build_train_callback_factory`'s `redrive_member` `found_on_main` arm (`harness.py:1330`,
     `kind='found_on_main'` — η's own provenance kind) and `mark_member_done` (`:1270`), driven from
     `SpeculativeMergeWorker._redrive_coalesce_members` (`merge_queue.py:14710`).

   **`metadata.train` does not separate those from η.** It is written at exactly one site,
   `TaskWorkflow._maybe_form_train` (`workflow.py:1954-1956`) — the *atomic-train* former, which is
   `merge_train_former_enabled`-gated and **off by default** (`:1871-1872`). A **coalesce** train
   (`merge_queue.py:15104-15123`) builds its `GroupMergeRequest` from the SAME callback factory over
   ordinary solo tasks that carry **no** `metadata.train`, and `_redrive_coalesce_members` selects its
   members by `statuses.get(mid) == 'merge-deferred'` (`:14774-14778`), never by train metadata. D5's
   `metadata.train is None ⇒ recoverable` is right for the **stranded residue** — 2543/2724/2923 are
   exactly that class — but it does not by itself exclude a *live* coalesce member.

   What separates them is **liveness of the `GroupMergeRequest`**: all three fire only while a merge
   worker holds that object in memory, and η's population is precisely the residue after it is
   destroyed. `harness.py:1261-1262` states the same gap from the other side — revert-to-pending is a
   withheld member's *"ONLY recovery edge (the stranded sweep provably cannot reach a merge-deferred
   task)"*. Consequences for η: reuse the attribution these sites already perform rather than
   re-deriving it (`no-lockstep-duplication`); and treat the window between a train advancing main and
   its callbacks completing as a race the `corroborate-before-acting` re-read must lose gracefully —
   `mark_member_done` also performs `MergeProvenance.consume` and `release_lane_for_terminal_task`,
   which a bare `mark_done` skips.

   **Task 4497 is NOT added to the partition and NOT made a prerequisite of η.** It owns one call
   site — `_redrive_coalesce_members`'s `delivered_checks` wiring and its escalation prose — not a
   recovery authority for a status, so putting it in the partition table would misstate ownership
   (G4). It delivers no capability η's signal asserts, and its declared files (`merge_queue.py`,
   `orchestrator/tests/test_merge_queue_coalesce.py`) do not intersect η's, so a dependency edge would
   serialise for nothing (G3). Note for whoever works 4497: its *"measured 95.4% `effect_absent`"*
   figure is pre-3116 and is superseded by correction 1's 60.8% residual, and its anchors
   (`merge_queue.py:14085/:14091/:14097-14105`) have drifted to `:14805/:14811/:14818-14825`.

   **(c) η's writer must carry two guards its spec did not name.**
   - `Harness._mark_in_progress_done` (`harness.py:6239`) is a **thin wrapper over
     `scheduler.mark_done`** and carries no capability guard of its own. Each of its four production
     callers applies `Harness._delivered_checks_block` (`:6193` →
     `delivered_checks.gate_mark_done_on_delivered_checks`) immediately first: `:5763`→`:5799`,
     `:11727`→`:11731`, `:11760`→`:11764`, `:11788`→`:11792`. `delivered_checks.py:249-275` is the
     enumeration of every attribution-shaped `mark_done` seam routed through that one decision, and
     `harness.py:6215` states the invariant ("one hot reload disarms all eleven"). η reusing the
     writer without the gate would be the first unguarded attribution seam in the set.
   - `_already_landed_dispatch_gate` refuses its own auto-done when
     `classify_pins(...).vetoes_done_flip` is True (`harness.py:11578`, `:11648`) — an open record at
     **any** level (`:11536-11543`), with `info` severity the only carve-out
     (`escalation/src/escalation/pins.py:144-147`, `:35`, `:237-238`). η's spec carried no such veto
     and no boundary row for the shape. Measured on `ea876cb624`: **5 distinct `pending` tasks hold a
     pending escalation record (7 records); 3 of the 5 would be vetoed** (4194, 4580, 4590 —
     `blocking`) and 2 would not (4218, 4521 — `info`-only). No `merge-deferred` task holds one. The
     shape therefore has both a live population and a live negative control. Reuse
     `recovery_emission.pin_buckets` (`:375-416`) rather than a fifth hand-rolled `classify_pins`
     call, and keep the veto predicate explicit at the site.

   Both are constraints on η's own writer, so they are recorded on task 4651 rather than filed as a
   separate task: such a task would own no mechanism of its own and would contend on `harness.py`.

   **(d) η's status filter is now mechanically pinned.** The partition binds 3539 *structurally*
   (`_RECONCILE_SWEEP_STATUSES`) but bound η by prose only, because η's actor is a new
   `BackgroundService` outside `_RECOVERY`. Two `delivered_check`s now express it; they gate both the
   attribution seam and θ's dependency on η (`docs/task-authoring.md` §3.3):
   - `_LANDED_RECONCILE_STATUSES[^=]*=[[:space:]]*frozenset\(\{'(pending|merge-deferred)', *'(pending|merge-deferred)'\}\)`
     — `expect: present` in `harness.py`. **Exercised against fixtures before filing:** it matches the
     two-member set in either order, with or without the `: frozenset[str]` annotation, and **fails**
     `{'pending','blocked'}`, `{'in-progress','merge-deferred'}` and any three-member set. It is a
     membership assertion, not a name grep. Declare the constant on one line — `harness.py:238` is
     the house model.
   - `in _LANDED_RECONCILE_STATUSES` — `expect: present` in `harness.py`. Non-vacuous against the
     first (the `= frozenset({...})` declaration line cannot satisfy it), so the set must be
     *consulted*, not merely declared.

   **(e) `deferred` gets an explicit non-owner with a measured basis** — see §Out of scope. No
   follow-up task filed: G1 names no consumer for a mechanism there, the population is empty by
   construction, and γ already owns and bounds the one artifact that accumulates in that corner.

## Background

### The measured defect

A full sweep of 930 non-terminal tasks found **7 tasks landed on main but never marked done**
(2543, 2724, 2923, 2949, 3103, 3610, plus 3902/3746/3604 already hand-fixed; 3916 correctly stays
pending on new scope). All seven were corrected by hand on 2026-08-22. The class is small but
**self-regenerating and silent**, and its downstream cost is real: task **2545** was dispatch-starved
for a month behind 2543, and 3610 had a live agent **redoing already-landed work** when it was closed.

The framing that opened the investigation — "`_RECOVERY` in `task_ground_truth.py` is the only
recovery path and its `MARK_DONE_WITH_PROVENANCE` rows are all keyed `IN_PROGRESS`" — was **refuted**.
Two corrections are load-bearing for this PRD:

1. **Row (f) is inert.** `(IN_PROGRESS, False, ON_MAIN, True, None) -> LEAVE`
   (`task_ground_truth.py:822`) maps to `LEAVE`, which is already the `.get` default
   (`classify_recovery`, `:909-916`). Deleting it changes nothing. The escalation veto is implemented
   by `has_open_escalation` being an **element of the lookup key** (`_shape`, `:899`). *Any change
   phrased as "narrow row (f)" is a no-op.*
2. **There are eight automatic landed-task detectors, not one.** `_RECOVERY` is consulted at exactly
   one production call site (the `recovery_for` call inside
   `harness.py::Harness._reconcile_one_stranded`), and even there two **sweep-side upgrades** later in
   that same method — the on-main `MARK_DONE_WITH_PROVENANCE` clause and the off-main
   `EXISTS_OFF_MAIN → RE_FILE_ESCALATION` clause, both gated on
   `Harness._only_merge_remediable` — override its verdict for `blocked` tasks, deliberately
   outside the table ("rather than a change to θ1's reviewed table — design decision, task 2243;
   esc-2243-4/5"). The project has **twice** chosen sweep-side upgrade over table edit.

### Why the eight detectors all missed

| Population | Detectors reaching it | Why it failed |
|---|---|---|
| landed `pending` | **one** — `_already_landed_dispatch_gate` | its evidence **decays** (below) |
| landed `blocked` | the sweep-side upgrade | vetoed by a **one-entry** category allowlist |
| landed `merge-deferred` | **none** | RC-3 *deletes* the landing evidence |

- **Decay.** `branch_content_in_main` (`git_ops.py:9155`) requires main HEAD to be
  **byte-identical** to the branch content across the touched path set; any *unrelated* later commit
  touching those paths flips it False. `commit_effect_present_in_main` (`git_ops.py:10005`) required
  the same **until task 3116 landed on 2026-08-22 evening** — it is now a threshold added-line
  survival test that tolerates ordinary additive evolution (correction 1). Decay therefore persists
  in full for `branch_content_in_main` and on 3116's measured **60.8% residual** for the survival
  predicate, and the condition stays **absorbing** either way — survival can only fall as main
  evolves. **The longer a task is stranded, the less recoverable it becomes.** Demonstrated: task
  3916 went detected-and-escalated (2026-08-19) → undetected (2026-08-22), same landing, evidence
  rotted beneath it.
- **Sync-merge misread.** For a branch tip that is a conflict-resolution merge (main merged *into* the
  branch), `parents[1]` is **main**, so `touched = diff(merge-base(p1,p2), p2)` is main's own history
  since the fork (~300 files for task 3103) and the check demands current main be byte-identical to an
  old main tip — **false by construction** one commit later. Any task whose branch tip is a sync-merge
  is permanently unrecoverable by this path.
- **Veto.** `_only_merge_remediable` (`harness.py:5286`) screens against
  `MERGE_REMEDIABLE_ESC_CATEGORIES = frozenset({'stranded_blocked'})` (`:5281`) — one of ~30 real
  categories. Task 3902's `preexisting_main_break` escalation therefore vetoed the recovery it had
  itself requested, for 39h42m. The governing precedent is already in the file (`:5262-5270`): *"the
  escalation asking for the merge blocks the merge."* PRD leaf δ fixed that shape for exactly one category.
- **Evidence destruction.** RC-3 (`merge_queue.py:6544-6546`) returns `already_done_pruned` and calls
  `outbox.consume(row.task_id)` whenever status is in `WORKFLOW_PRESERVE_STATUSES` — which includes
  `blocked`, `deferred`, `merge-deferred`. `consume` **deletes** the row and durably flushes
  (`landed_outbox.py:115-125`: *"a consumed row does not resurrect after a restart"*). So for exactly
  the parked statuses in this defect class, the system destroys the durable proof of landing, leaves
  the task not-done, and labels the outcome "already done".

### The accepted-risk premise that failed

`commit_effect_present_in_main` documents both its failure modes as deliberate accepted risks
(`git_ops.py:9052-9077`), justified by one premise: *"the caller's own recovery path on False is
idempotent … so the cost of a false negative here is a re-check."*

That premise is false on three measured counts: for a landed `pending` task there **is** no other
recovery path; the false negative is **monotonic** (each later attempt is strictly less likely to
pass); and the cost compounds into month-long dependency starvation. The fail-safe *direction* remains
right — never cement a false completion. What was wrong is the belief that the failure is cheap.

### The revert measurement (why effect-present goes, and what replaces it)

Both repos, full history 2026-03-13 → 2026-08-22, 5,727 task landings (DF 2,962 / reify 2,765):

| Class | DF | reify |
|---|---|---|
| Merge-lane auto-revert of a landing (**extinct**) | 7 events / 6 tasks | 7 / 7 |
| **Genuine post-hoc revert of landed work** | **1** (task 879) | **1** (tasks 5169/5178) |
| **No-op landing** (marker present, empty net branch diff) | **4** | **15** |
| `effect_absent` verdicts at HEAD | 2,829 (95.5%) | 2,640 (95.5%) |

- **2 genuine reverts in 5.4 months** against ~5,469 `effect_absent` verdicts ⇒ **≈0.04% precision**.
  Of the 7 firings actually recorded in the escalation/emission corpus, **7 were false positives, 0 genuine**.
  Both real reverts were found and repaired by humans within 31h and 18h *without* the guard.
- **14 of 16 lifetime true positives came from a code path deleted on day 9** (`af1e7de63a`,
  2026-03-22, "replace merge revert with reset and add merge lock"). Its replacement leaves no merge
  marker, so neither ancestry nor patch-id can false-positive on it. That class is extinct.
- **The guard's own motivating case is misdiagnosed.** `git_ops.py:8981` and `harness.py:5695` describe
  task 1175 as a merge whose deliverable "a later commit on main removed". Verified false:
  `614137480e` has `merge-base(p1,p2) == p1` and `git diff --name-only base p2` == **0 files** — an
  empty net branch diff. Nothing was ever on main to revert; the guard catches it via its
  empty-touched-set fail-safe, not via revert detection. `plans/found-on-main-provenance-integrity-prd.md`
  characterises the same event correctly as a **re-derivation clobber**; it is the two `git_ops`/`harness`
  docstrings that are wrong, and this PRD corrects them.
- **`esc-5181-2` is not a revert either** — `data/reconciliation/tickets.db` rowid 4030 records it
  verbatim as a *"scope re-route"* (`reroute_from: reify-task-5181`), a cross-repo mis-filing. The
  docstring's two-case evidence base is really one case, overstated 2×.

**The measurement surfaced a real target the guard was never aimed at.** `git cherry` patch-id is
revert-blind *and* blind to **no-op landings** — 19 across both repos, still live (most recent reify
task 5702, 2026-08-12, ~0.33% of task merges). A merge whose `merge-base(p1,p2)..p2` diff is empty has
all its commits in main, so patch-id reports "landed" while nothing landed. Detecting it is one
`diff --name-only`, and it carries none of the 95.5% false-positive load.

## Sketch of approach

Five changes, ordered by the option numbers Leo ratified.

- **[7] Speak on reject.** `_already_landed_dispatch_gate`'s ancestry arm returns `False` silently
  (`harness.py:11681-11693`) — which is why task 3103 has **zero escalations, ever**, despite being
  the only detector that could have recovered it. Emit a recovery disposition instead. *No escalation*
  — the existing code's objection to escalating here (per-tick noise) is correct and preserved.
- **[5] Stop destroying evidence.** Split RC-3's single branch: consume the LandedOutbox row only for
  genuinely **terminal** status (`done`/`cancelled`); for parked statuses decline to write but **keep
  the row**, so a later reconciler still has durable proof.
- **[4] Narrow the veto, on-main only.** A `preexisting_main_break` escalation must not veto a
  mark-done backed by on-main landing evidence — its ask (`await_preexisting_main_hotfix`) is
  satisfied by construction once the branch is on main. The `EXISTS_OFF_MAIN` clause is left alone,
  where the ask is genuinely unmet.
- **[1] Non-decaying landing evidence.** `git cherry` patch-id equivalence establishes attribution;
  a new empty-net-diff check rejects no-op landings; effect-present is retired **at the
  landing-detection sites only**.
- **[6] A reachable actor.** The escalation server already computes a fully-guarded `found_on_main`
  verdict for any task id in any status — and throws it away. Extract it and give it a periodic
  writer, so recovery no longer depends on the task being in a status the stranded sweep enumerates.

**Explicitly not done** (each considered and rejected on evidence):

- *Widening `_RECONCILE_SWEEP_STATUSES`* — already considered and **rejected** in
  `docs/prds/claimant-invariant-enforcement.md` (*"the set is load-bearing … pinned by
  `test_reconcile_stranded.py:4196` and `test_repend_state_machine.py:681`"*). Option 6 delivers the
  same reach without touching it.
- *Adding rows to `_RECOVERY`* — inert for `pending`/`merge-deferred` (filtered upstream by
  `harness.py::_RECONCILE_SWEEP_STATUSES`), redundant for `blocked` (the caller already upgrades), and would create two
  disagreeing authorities for the same shape.
- *Editing row (f)* — a no-op (see Background).

## Resolved design decisions

**D1 — Patch-id for attribution, empty-net-diff for no-op, effect-present retired at these sites.**
Ratified on the measurement above. Effect-present's 0.04% precision does not justify a guard whose
false-negative mode is the defect this PRD exists to fix. The empty-net-diff check preserves the only
failure mode the guard demonstrably caught. **Scope limit:** effect-present is retired at the
*landing-detection* sites only (dispatch gate, Tier-3.5); its other callers are out of scope and must
be enumerated by leaf δ before removal.

**D2 — Substrate already exists; no new git primitives.** `git cherry` is already used for exactly
this question at `git_ops.py:8784` (inside `rebase_preserving_task_commits`, def `:8661`) and
`git patch-id --stable` at `:10308`/`:10318` (inside `find_equivalent_commit`, def `:10249`). G3
therefore resolves against existing, production-proven code, not novel substrate — **re-verified at
decompose against `d8f165756b`; the anchors this decision originally cited (`:8544-8555`,
`:9229-9245`) drifted with the 3116 landing, the substance holds.** There is no named
`get_commit_parents` helper; δ reuses the inline `rev-list --parents` idiom at `git_ops.py:9390`.

**D3 — Option 4 needs a second category set, not a wider one.** `_only_merge_remediable` is called at
*both* upgrade clauses (`harness.py:5566` on-main, `:5596` off-main). Widening the shared
`MERGE_REMEDIABLE_ESC_CATEGORIES` would also widen the off-main clause, where a
`preexisting_main_break` ask is genuinely unmet. Introduce a distinct on-main set
(`MERGE_REMEDIABLE_ESC_CATEGORIES | {'preexisting_main_break'}`) consumed only by the on-main clause.

**D4 — RC-3 declines but preserves; it does not become a second writer.** Splitting the branch keeps
RC-3's write authority unchanged (parked statuses are still not written by it) while preserving the
evidence. The actor for a landed parked task is leaf η's reconciler — **one authority, not two**.

**D5 — `metadata.train` is the merge-deferred discriminator.** `merge-deferred` is a legitimate parked
state *only* for a real atomic-train member. None of 2543/2724/2923 carried `metadata.train`: they
reached the status via `_handle_superseded` (`workflow.py:1526-1558`), which hands ownership of a
**durable** status to a **volatile in-memory** `GroupMergeRequest` — destroyed for 2543/2923 by a
fleet restart 14 min into the train, and for 2724 by a stale `get_statuses` read 240 ms after
coalescing. `metadata.train is not None` ⇒ correctly parked, hands off. `is None` ⇒ recoverable.
Leaf η owns this discriminator; the explicit early-return at `harness.py:5488` stays.
**Scope corrected at the 2026-08-24 recheck (correction 11b):** `metadata.train` is written at exactly
one site — the *atomic-train* former `workflow.py::TaskWorkflow._maybe_form_train`, which is
`merge_train_former_enabled`-gated and off by default. A
**coalesce** member carries none, so `metadata.train is None` is sound for the stranded residue but is
NOT what separates η from the live coalesce writers. That separation is the liveness of the
`GroupMergeRequest`; see correction 11b for the three writers and the race η must lose gracefully.

**D6 — Option 7 emits, never escalates, and never charges the streak alarm.**
`RecoverySite.already_landed_gate` is **deliberately excluded** from `STREAK_CHARGING_SITES`
(`recovery_emission.py:152-168`) because it runs per dispatch *tick* — charging it would file a
blocking L1 within seconds. The existing `veto_streak_min_span_secs` backstop is the sanctioned path
for a tick-rate site. Leaf α must not add the site to `STREAK_CHARGING_SITES`.

**D7 — Leaf η must check live-claimancy before writing.** Task 3610 was `in-progress` with a heartbeat
54 s old while its work sat on main; a reconciler that wrote `done` under a live claimant would race a
running workflow. η resolves live-claimancy first and skips (emitting a disposition) rather than
racing — the same discipline `_RECOVERY` encodes by folding `live_claimant` into its key.
**Substrate (decompose correction 7):** call `shared.task_claimant.has_live_claimant`
(`task_claimant.py:187`, status-agnostic) **directly**. Do **not** route through
`_resolve_live_claimant` — its in-progress-only gate reads a stale claimant on a parked task as live
forever, which would defeat B11. No dependency on `claimant-invariant-enforcement` is required.

**D8 — Extraction before writer.** `_found_on_main_response` (`escalation/server.py:3195`) and
`_git_authority_task_metadata` (`:3240`) are **not callable from outside**, so Option 6 is two
leaves: extract to a shared module, then write. **Mechanism corrected at decompose (correction 8):**
they are *not* closures inside `merge_status` — all three are siblings at 4-space indent inside
`create_server` (`:596`). They still close over `create_server`'s locals, so the conclusion stands.
The extraction must be behaviour-preserving for `merge_status` (pinned by
`escalation/tests/test_merge_status_git_authority.py`, 28 tests) **and for `merge_request`, which
calls `_git_authority_task_metadata` at `:2293` and is pinned by nothing** — ζ must add that pin.

## Pre-conditions for activating

**Corrected at decompose — "none blocking" was wrong on two counts.** All five changes still act on
code live on main today, but:

- **η is blocked on task 4623** (`_resolve_live_claimant` repointed at the status-agnostic
  predicate), itself blocked on **4618** (`is_stranded_any_status`) — see correction 7. Without it
  D7's check skips every landed parked task and B11 can never fire.
- **ε and ζ are sequenced behind tasks 4496 and 4498**, which already own the exact call sites they
  touch and are unstarted. This is ordering, not a design blocker.

Sequencing internal to the decomposition (§Decomposition plan): δ before ε and η, ζ before η,
γ before η.

## Cross-PRD relationship

| Other PRD / task | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `plans/found-on-main-provenance-integrity-prd.md` | consumes | the six already-landed re-derivation sites; `done_provenance` attribution; the reopen-freshness gate (`_check_reopen_freshness`, `task_interceptor.py:5670`) | **that PRD** | active — this PRD changes the *evidence* those sites use, never the attribution/reopen rules |
| `docs/prds/claimant-invariant-enforcement.md` | consumes | `_resolve_live_claimant` / `is_stranded_any_status`; `_RECONCILE_SWEEP_STATUSES` | **that PRD** | active (Leo, 2026-08-21). This PRD does **not** touch either; D7's live-claimancy check must use whatever that PRD lands |
| `plans/orchestrator-atomic-train-merge-prd.md` | consumes | `merge-deferred` lifecycle, `metadata.train`, §9.8 guards | **that PRD** | leaf η adds a *recovery* edge for train-less members; the parked-member contract is unchanged |
| task **3541** (`classify_pins` veto collapse) | produces | the escalation-veto predicate at every recovery site | **3541** | pending, blocked on 3533/3540/3550/3563/3976. Leaf β is a deliberate one-category stopgap; 3541 supersedes it |
| task **4614** (merge-phase gate discards `already_done.json`) | sibling | the `already_done` escape hatch | **4614** | pending — a different landed-not-done channel; not folded in |
| task **3116** (effect_absent predicate + prose) | produces | `commit_effect_present_in_main`'s semantics; the optional `delivered_checks` parameter | **3116** | **done** (`40c39cd8ee`, 2026-08-22T20:41) — landed *after* this PRD's anchor sha; see §Post-authoring corrections 1 |
| task **4496** (harness landing-evidence call sites) | **supersedes leaf α** | `harness.py`'s ancestry arm + `_reconcile_one_stranded`: structured event on reject | **4496** | pending/high, unstarted. **Leaf α is a strict subset — α is NOT filed; θ depends on 4496** |
| task **4498** (`merge_status` git-authority arms) | produces | `escalation/server.py:3506`/`:3552` — the Tier-3.5 call sites ε re-points and ζ extracts around | **4498** | pending/med. ε depends on it; ζ must not run concurrently (identical file pair) |
| task **4497** (coalesce re-drive call site) | sibling | `merge_queue.py:14241` — the eighth landing-detection site | **4497** | pending/med. File-lock contention with γ (same file, ~7,700 lines apart) |
| task **4500** (capstone: `delivered_checks` required) | consumes | `validate_landing_evidence`'s signature and its seven wired call sites | **4500** | pending/med, deps `[3116, 4496, 4497, 4498]`. δ/ε **must not** change that signature or the call-site count |
| task **4606** (`_delivered_checks_differential` hardcodes `'main'`) | sibling | `landing_evidence.py` | **4606** | pending/med — same file as δ; sequence, don't co-run |
| task **3539** (`CONVERT_TO_BLOCKED` recovery backstop) | sibling | `in-progress` + escalation-pinned strands; the `plan_files_not_touched` already-landed carve-out | **3539** | pending/high. **Ruled 2026-08-24 (gate 4673): no conflict — disjoint on status** (correction 9). 3539 is `in-progress`-keyed; η owns `pending`/`merge-deferred`; β owns `blocked`. 3539's re-pender question is answered in correction 10 |
| task **4501** (14-day reject-event survey) | consumes | the reject events 4496 emits | **4501** | pending/med — depends on 4496 landing; another reason not to duplicate it as α |

Reciprocal-ownership: none. **But the G4 walk at decompose found this table originally omitted the
entire task-3116 sibling family (4496/4497/4498/4499/4500) — a family named in the code itself at
`landing_evidence.py:585-592`. Two of its members own leaves this PRD re-derives.** Rows added above.

## Contract (B + H)

The seam is **landing evidence**: what the system accepts as proof that task *N*'s work is on main.
Today **four producers serve seven consumers** inconsistently (correction 5). This PRD makes one
contract and points them all at it.

> **The module already exists** (correction 4). `landing_evidence.py` holds `LandingEvidenceVerdict`
> (`:189`), `validate_landing_evidence` (`:527`) and `branch_is_degenerate` (`:137`). The block below
> is the **target** shape, not a new file: `branch_work_landed` becomes the single producer and
> `validate_landing_evidence` is re-expressed as a thin mode over it, preserving its public signature
> (task 4500 flips its `delivered_checks` parameter to required and asserts a count of seven wired
> production call sites — do not invalidate that precondition). Do **not** add a second verdict type.

```python
# orchestrator/src/orchestrator/landing_evidence.py

@dataclass(frozen=True)
class LandingVerdict:
    accepted: bool
    evidence_sha: str | None      # commit to record as provenance; None iff not accepted
    reason: str                   # closed vocabulary, see below
    method: str                   # 'patch_id' | 'merge_marker' | 'citation'

async def branch_work_landed(
    git_ops, task_id: str, branch: str, *, branch_tip_sha: str | None,
) -> LandingVerdict: ...
```

**Invariants.**

1. **Non-decaying.** For a fixed `(task_id, branch, branch_tip_sha)` and a main that only ever gains
   commits, a verdict of `accepted=True` MUST NOT later become `False` unless the work is genuinely
   removed from main. Ordinary later evolution of the same paths MUST NOT change the verdict. *This is
   the invariant whose absence caused every stranding in this PRD, and it is the one boundary test that
   must not be waived.*
2. **No-op rejection.** If `merge-base(first_parent, branch_tip)..branch_tip` has an empty diff, the
   verdict is `accepted=False`, `reason='no_op_landing'` — regardless of what patch-id says. Ordering
   is normative: the no-op check runs **before** patch-id acceptance.
3. **Attribution.** `evidence_sha` MUST be a commit that carries this task's work — never a branch tip
   that is not on main, never `main`'s current tip as a fallback. When attribution cannot be
   established, the verdict is `accepted=False`, never accepted-with-a-guess.
4. **Fail-closed.** Any git failure (non-zero rc, unparseable output, unresolvable ref) yields
   `accepted=False` with a `reason` naming the failure. Never accept on doubt.
5. **Silent-free.** Every `accepted=False` yields a `reason` from the closed vocabulary, and every
   call site emits it. No call site may discard a verdict without recording the reason.

**Reason vocabulary (closed):** `landed` · `no_op_landing` · `not_landed` · `no_attribution` ·
`degenerate_branch` · `git_error`.

**Ordering rule.** `no_op_landing` outranks `landed`: a merge can simultaneously have all its commits
patch-id-present in main and an empty net diff, and the no-op verdict wins.

## Boundary-test sketch (B + H)

Faces both the producer (`landing_evidence`) and every consumer (dispatch gate, Tier-3.5, η's reconciler).

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| B1 | **Non-decay** — landed task, then N unrelated commits touch the same paths | branch landed; main advanced over those paths | verdict stays `accepted=True` at every step. *The regression pin for this whole PRD* |
| B2 | **Sync-merge tip** | branch tip is a conflict-resolution merge whose `parents[1]` is main | `accepted=True`; the ~300-file main-history diff is never computed |
| B3 | **Rebase landing** | commits landed with rewritten SHAs, no merge commit, no citation | `accepted=True`, `method='patch_id'` |
| B4 | **No-op landing** | merge marker on main, `merge-base..tip` diff empty | `accepted=False`, `reason='no_op_landing'`; task re-dispatches, is **not** stamped done |
| B5 | **Genuinely unlanded** | branch exists, commits absent from main | `accepted=False`, `reason='not_landed'` |
| B6 | **Degenerate branch** | tip == branch_base_sha, zero own commits | `accepted=False`, `reason='degenerate_branch'` |
| B7 | **Blocked + on-main + only a `preexisting_main_break` escalation open** | landed; that one escalation open | sweep marks done; the off-main clause is unaffected |
| B8 | **Blocked + on-main + a human-concern escalation open** | landed; e.g. `design_concern` open | sweep still LEAVEs — the narrowing did not become a blanket |
| B9 | **Parked landed task keeps its evidence** | landed; status `merge-deferred` | RC-3 declines to write **and** the LandedOutbox row survives; terminal status still consumes |
| B10 | **Train member untouched** | `merge-deferred` **with** `metadata.train` | η skips it; no write, disposition emitted |
| B11 | **Train-less coalesce orphan recovers** | `merge-deferred`, no `metadata.train`, landed | η marks done with attributed provenance |
| B12 | **Live claimant not raced** | landed; `in-progress` with a fresh heartbeat | η skips, emits a disposition, writes nothing (D7) |
| B13 | **Reject is audible** | dispatch gate rejects for any reason | a recovery disposition carrying the reason is recorded; **no** escalation; the site does not charge the streak alarm (D6) |

## Decomposition plan

Labels are intra-batch prereqs. All modules are under `orchestrator/` unless stated.

- **α — Emit a disposition when the dispatch gate declines a landing** *(option 7)*.
  **NOT FILED — superseded by existing task 4496** (pending/high, unstarted), which owns the same
  site (`harness.py:11687`, the ancestry arm's silent `return False`), reaches the same remedy
  (a structured record, explicitly **not** an escalation), and strictly supersets α by also covering
  `_reconcile_one_stranded` (`:5702`) and the `delivered_checks` wiring. Task **4501** already depends
  on 4496 landing. Filing α would have double-claimed a hot file for work already queued.
  θ depends on **4496** in α's place.
  Signal (retained here for the record, and satisfied by 4496): after a dispatch tick that declines
  task *N*, an operator reading the recovery-disposition record sees `site=already_landed_gate` with
  the decline reason; previously nothing was recorded (B13).
  Substrate note for whoever works 4496: `_emit_recovery_disposition` is **already called twice
  inside this same function** (`harness.py:11593`, `:11625`) — this is a third call to wired
  plumbing, not new machinery. The event-type discriminator needs no edit either: `harness.py:9791`
  routes anything that is not `escalation_pinned`/`provenance_arbitration` to `recovery_left` by
  fall-through. **One real gap:** the gate's only streak-release edge
  (`_clear_recovery_veto_streak`, `:11655`) sits *above* all git work on the escalation path, so a
  decline signature accumulated below it is never popped — on a per-tick site that is an unbounded
  tracker footprint. A decline emission needs its own release edge. The read path is `EventStore` → `runs.db` `events`
  (`recovery_vetoed`/`recovery_left`) → the dashboard scheduler page
  (`dashboard/data/scheduler.py:112-116`) and `get_scheduler_events`.
  G7 (`storm-escape-required`, unresolved by D6): keep the site out of `STREAK_CHARGING_SITES` as D6
  requires, but note correction 6 — `veto_streak_min_span_secs` never runs for a non-charging site,
  so the site has **no bound today**. Whoever lands 4496 must add one (a site-local span floor of
  ≥1h before a single sentinel-keyed L1, dedup'd via the existing `has_open_l1` path).

- **β — Stop a stale `preexisting_main_break` vetoing an on-main self-heal** *(option 4)*.
  Modules: `orchestrator` (`harness.py`). Prereqs: none.
  Signal: a `blocked` task whose branch is on main and whose only open escalation is
  `preexisting_main_break` reaches `done` on the next sweep (B7); one with a human-concern escalation
  still does not (B8).

- **γ — Preserve the LandedOutbox row for parked tasks** *(option 5)*.
  Modules: `orchestrator` (`merge_queue.py`, **`harness.py`**). Prereqs: none.
  `harness.py` added at decompose: the only operator-facing reader of RC-3's dispositions is the
  summary line at `harness.py:11233-11238`, so a new label added only in `merge_queue.py` would be
  read by nothing in γ's blast radius (G1 orphan).
  Signal: for a landed task in `merge-deferred`, the row is still present in
  `data/orchestrator/landed_outbox.json` after a reconcile pass, and the outcome label distinguishes
  "pruned" from "skipped-parked"; a `done` task's row is still consumed (B9).
  `WORKFLOW_PRESERVE_STATUSES` = `{done, cancelled, deferred, blocked, merge-deferred}`; the split is
  `{done, cancelled}` (consume) vs `{deferred, blocked, merge-deferred}` (retain).
  G7 `no-lockstep-duplication`: consume when `status in TERMINAL_STATUSES`, retain when
  `status in WORKFLOW_PRESERVE_STATUSES - TERMINAL_STATUSES`, both imported from
  `orchestrator.task_status` — never a hand-written `{'done','cancelled'}` literal.
  G7 `holds-owned-and-bounded`: "terminal status still consumes" bounds only the happy path. A
  retained row is a hold, so name η's reconciler as its exit owner in the retained row's disposition,
  surface each retained row's age in the reconcile summary line, and escalate past a configured age
  (default 24h) — η reaches neither `deferred`, real `metadata.train` members, nor deleted tasks, so
  those would otherwise retain a row forever with no owner and no visibility.

- **δ — `branch_work_landed`: patch-id attribution + no-op rejection** *(option 1, producer)*.
  Modules: `orchestrator` (`landing_evidence.py`, `git_ops.py`). Prereqs: none.
  Unlocks: ε, η. Intermediate — its consumers are ε and η.
  **Re-scoped at decompose: `landing_evidence.py` ALREADY EXISTS** (correction 4). Already present and
  NOT δ's work: `branch_is_degenerate` (`:137`, B6's predicate, wired at five production sites),
  `is_valid_sha_40` (`:122`), a frozen verdict with a `probe` (`LandingEvidenceVerdict`, `:189`), and
  the survival predicate. Genuinely new: `branch_work_landed`, patch-id as the *landing* predicate,
  the `no_op_landing` check and its ordering rule, the closed reason vocabulary (today's is
  `ok`/`no_citation`/`effect_absent`), and the non-decay invariant with its B1 pin.
  Signal (intermediate): B1–B6 green against real repo fixtures.
  Also enumerates every remaining `commit_effect_present_in_main` / `branch_content_in_main` caller
  and records which are in scope for ε (D1's scope limit) — and must rule explicitly on the eighth
  site, `merge_queue.py:14241` (`_redrive_coalesce_members`), which D1's "dispatch gate, Tier-3.5"
  wording does not name. Must also re-derive D1's precision arithmetic on 3116's 60.8% residual
  (correction 1), and fix the module docstring's stale "Five always-on sites" (there are seven).
  G7 `contracts-machine-checked`: declare the `reason` and `method` vocabularies as `enum.StrEnum`
  (house pattern — `RecoverySite`, `LeaveReason`, `PinClass`) and type the verdict fields as those
  enums, not `str`; today's vocabulary is prose-only and degrades to `'Unrecognized reason code: …'`
  (`landing_evidence.py:762`), i.e. discovered by failure. Pin that every member has an explanation.
  **Reusable attribution primitive found at decompose:** `merge_queue.patch_content_contained`
  (`merge_queue.py:3740-3767`) already runs `git cherry upstream head` and answers "every commit
  already present by patch-id" — exactly δ's attribution predicate. Import-cycle constraint:
  `merge_queue.py:46-49` imports *from* `landing_evidence`, so a top-level reverse import is a cycle;
  use the house function-scoped lazy import (precedent `merge_queue.py:816`).
  **The enumeration deliverable is smaller than D1 implies — 3 production sites, measured:**
  `commit_effect_present_in_main` has exactly two production callers, both inside
  `validate_landing_evidence` (`landing_evidence.py:637` CANDIDATE, `:697` DISCOVERY);
  `branch_content_in_main` has exactly one (`harness.py:11733`, the gate's content-equivalence
  fallback, where it is the **entry condition**, not an inner guard). Everything else is docstrings
  or test stubs.
  **Register every new reason in `_REASON_EXPLANATIONS`** (`:715`): `format_unattributed_landing_detail`
  renders the literal `'Unrecognized reason code: <x>'` into a human-facing escalation for anything
  missing (`:762`), so the `no_citation` → `no_attribution` rename is a *registration* change.
  G7 `no-lockstep-duplication`: do **not** add a second verdict type beside `LandingEvidenceVerdict`.
  Extend the incumbent and re-express `validate_landing_evidence` as a thin mode over
  `branch_work_landed`, preserving its public signature (task 4500's precondition).
  G7 `storm-escape-required`: `git_error` is a fail-soft degradation — never collapse it into
  `not_landed`, expose a per-reason tally, and escalate past a configured rate (default 10/h). It is
  the one reason whose repetition means the detector is broken, not the task unlanded.

- **ε — Point the dispatch gate and Tier-3.5 at the new contract** *(option 1, consumers)*.
  Modules: `orchestrator` (`harness.py`), `escalation` (`server.py`).
  Prereqs: **δ**, plus out-of-batch **4496** and **4498**.
  **Re-scoped at decompose:** 4496 owns `harness.py:11671/:11713/:11741/:5702` and 4498 owns
  `server.py:3506/:3552`; both are unstarted. ε is therefore **re-pointing only** — it does not
  re-do their event-emission work, and it depends on them so the three never contend for the two
  hottest files in this collision surface concurrently.
  Signal: a task stranded for weeks whose paths have since been edited by other work is recovered on
  the next dispatch tick — the exact 3916/3103 shape that is undetectable today (B1, B2).
  Also corrects the task-1175 docstrings — **three copies, not two**: `git_ops.py:10021-10022`, its
  verbatim twin at `git_ops.py:9249-9250` (inside `describe_commit_effect_in_main`), and
  `harness.py:5695`. The PRD originally cited `git_ops.py:8981`, a stale anchor.
  G7 `no-lockstep-duplication`: D1 retires effect-present at the dispatch gate and Tier-3.5 only,
  leaving the stranded sweep (`harness.py:5702`) and the coalesce re-drive (`merge_queue.py:14241`)
  on the old semantics — express that as one shared producer taking an explicit mode/policy argument,
  never two functions, and pin which call site passes which mode as a documented deliberate
  divergence keyed to δ's enumeration.

- **ζ — Extract the git-authority landing tier out of the escalation server** *(option 6, prerequisite)*.
  Modules: `escalation` (`server.py`) → shared module. Prereqs: out-of-batch **4498** (identical file
  pair — `server.py` + `test_merge_status_git_authority.py`; must not run concurrently).
  Unlocks: η. Intermediate.
  Signal (intermediate): `merge_status`'s existing suite
  (`escalation/tests/test_merge_status_git_authority.py`, 28 tests) stays green — behaviour-preserving
  by construction — and the tier is importable outside `create_server`'s scope.
  **Scope widened at decompose (correction 8):** the helpers are siblings inside `create_server`, not
  closures inside `merge_status`, and `_git_authority_task_metadata` has a **second caller**,
  `merge_request` at `:2293`, whose degenerate-branch guard (`:2287-2296`) the suite does **not**
  cover — zero references to `merge_request` in that file. ζ must preserve that surface **and add a
  pin for it**; "green by construction" is true only for `merge_status`.
  G7 `structured-facts-at-failure`: `_git_authority_task_metadata` returns `{}` on every failure mode
  (no harness, no `scheduler` attribute, `get_task` raising, genuinely metadata-less task) so a caller
  cannot recover what the emitter knew. The extracted API must return the fetch outcome alongside the
  metadata (a `metadata_unavailable` flag or a tri-state — the same discipline as
  `escalation_store_unavailable`). `merge_status` keeps failing open exactly as today so the
  extraction stays behaviour-preserving; the flag exists so η's writer can treat "could not read
  metadata" as "cannot verify the degeneracy guard" rather than "no degeneracy".

- **η — Periodic status-agnostic landed reconciler with a writer** *(option 6)*.
  Modules: `orchestrator` (new reconciler + harness wiring). Prereqs: **δ, ζ, γ**.
  Signal: a landed task in `pending` **or** `merge-deferred` (without `metadata.train`) reaches `done`
  with attributed provenance without any dispatch and without a human (B11); a real train member and a
  live-claimed task are both skipped (B10, B12).
  Hang it beside the stranded sweep as its own `BackgroundService`
  (`orchestrator/background_service.py:52`; registration precedent `harness.py:2088-2100`) with an
  independent `*_interval_secs` and `*_enabled` — zero new machinery (resolves Open question 1).
  **Boundary with task 3539 — ruled 2026-08-24 (gate 4673), no conflict.** η owns `pending` and
  `merge-deferred`; β owns landed `blocked`; 3539 owns `in-progress` + escalation-pinned. Disjoint by
  construction (correction 9). If η's scope drifts across a line, stop and re-raise rather than
  absorbing a neighbour's territory. 3539's *"something re-pended it"* is **identified** — the
  mark-done applier's own reject arm inside `Harness._reconcile_one_stranded` (correction 10) — so do not spend
  effort hunting a second re-pender; the same decayed-evidence reject is why δ is a hard prereq.
  G7 `status-matches-liveness` (D7): use `shared.task_claimant.has_live_claimant`
  (`task_claimant.py:187`) **directly**, never `_resolve_live_claimant` — see correction 7. Alarm
  rather than silently skipping when a claimant's heartbeat is older than the TTL.
  G7 `corroborate-before-acting`: re-read live status **and** claimant immediately before the `done`
  write, inside the same choke point that performs it, and abort if either changed since
  classification. D5 blames a 240 ms-stale `get_statuses` read for mis-parking 2724; a status-agnostic
  pass takes far longer than 240 ms. The git evidence is ground truth and needs no re-read; the
  task-store state does.
  G7 `storm-escape-required`: a disposition per skip is the structured-facts half but supplies no
  counter. Add a dedicated `RecoverySite` member for this reconciler and **include it in**
  `STREAK_CHARGING_SITES` — it runs at sweep cadence, not per dispatch tick, so it is exactly the site
  class that set exists for, and D6's exclusion names `already_landed_gate` only.
  **Added at the 2026-08-24 recheck (correction 11 b/c/d) — three constraints, all on η's writer:**
  (1) η is not the sole `done`-writer over `pending`/`merge-deferred`;
  `merge_queue.py::reconcile_landed_row`'s RC-2 arm and
  `harness.py::build_train_callback_factory`'s `redrive_member` / `mark_member_done` closures already
  are, gated by `GroupMergeRequest` liveness rather than by `metadata.train`. Reuse their attribution;
  do not re-derive it. (2) `harness.py::Harness._mark_in_progress_done` carries **no** capability
  guard — call `harness.py::Harness._delivered_checks_block` immediately before it, as all four
  existing callers do, or η becomes the first unguarded attribution seam in the set
  `delivered_checks.py::verify_delivered_checks_on_main`'s docstring enumerates. (3) Honour
  `escalation/src/escalation/pins.py::PinReport.vetoes_done_flip` (`info` carved out) exactly as
  `harness.py::Harness._already_landed_dispatch_gate` does, via
  `recovery_emission.py::pin_buckets`; add a boundary row for a landed `pending` task holding a
  `blocking` record (live specimens: 4194, 4580, 4590) and its `info`-only negative control (4218,
  4521). The status filter itself is now pinned by two `delivered_check`s — see correction 11d.
  G7 `loop-thread-occupancy-bounded`: both limbs apply. `GitOps` already uses the async runner, but
  `derive_truth`/`_resolve_live_claimant` do sync filesystem and escalation-store I/O per task on the
  loop thread — offload per-task resolution via `asyncio.to_thread`. And the candidate set is
  status-agnostic (930 non-terminal tasks in this PRD's own sweep), so cap per-pass items with an
  explicit configured bound that **logs what it dropped** and resumes from a rotating cursor — never a
  silent truncation. State the worst case as cap × per-item cost in the pass's docstring.

- **θ — Integration gate: the landed-recovery boundary suite** *(B+H integration task)*.
  Modules: `orchestrator`, `escalation`. Prereqs: **β, γ, ε, η** + out-of-batch **4496**
  (α's superseding owner — B13 is 4496's deliverable, not a filed leaf's).
  Signal (leaf): the full B1–B13 matrix green as one suite, exercising each of `pending`, `blocked`,
  and `merge-deferred` end-to-end against a real git fixture — a landed task in each status self-heals,
  a no-op landing does not, and a live-claimed task is never raced.

Dependency shape **as filed at decompose**: `δ, 4496, 4498 → ε`; `4498 → ζ`; `δ, ζ, γ → η`;
`4496, β, γ, ε, η → θ`. β, γ, δ are independently startable.

**The original claim that "the three modules each leaf touches are disjoint enough to avoid lock
contention except at θ" is falsified** — the collision walk found γ contending with 4497 on
`merge_queue.py`, δ with 4500 and 4606 on `landing_evidence.py`, ζ with 4498 on `server.py`, and β
with 3541/3539 in `harness.py`'s `:5281-5600` veto region. The out-of-batch dependency edges above
serialise the ones that would actually conflict; β's is left unserialised deliberately (it is a
declared stopgap 3541 supersedes).

## Out of scope

- **The `done_evidence_stale` trap for citation-corrected reopens.** A task reopened because its
  *citation* was fabricated (not because its work was absent) can never satisfy the reopen-freshness
  gate honestly: its evidence necessarily predates `reopen_at`, and no later landing exists or should.
  Task 3610 hit this on 2026-08-23 and needed the sanctioned `stale_evidence_override`. This is a real
  sixth defect in the same family, but it belongs to
  `plans/found-on-main-provenance-integrity-prd.md`, which owns the gate. **Not folded in** —
  recorded here so it is not lost.
- **Task 3541's `classify_pins` veto collapse.** β is a deliberate one-category stopgap. The general
  collapse stays with 3541 and its five dependencies.
- **Task 4614** (merge-phase gate discards `already_done.json`) — a different landed-not-done channel.
- **`deferred` status — explicit non-owner, decided at the 2026-08-24 recheck.** No automatic
  recovery path reaches it and none is added here. η *could* cover it and deliberately does not.
  Basis, measured on `ea876cb624`: (1) the orchestrator never **writes** `'deferred'` — every hit
  under `orchestrator/src/orchestrator/` is a `status == 'deferred'` **read**, in `workflow.py` and
  `chronic_flake.py`; the
  status is set by the planner (`submit_task(planning_mode=True)` / `commit_planning`) or an
  operator, so recovering it automatically would overwrite a human decision. (2) The population is
  empty by construction: of the 12 live `deferred` tasks (1147, 2035, 2217, 2290, 2324, 2346, 2349,
  2352, 2423, 3008, 3850, 4021) **none has a task branch at all**, against 771 task branches in the
  repo — there is no landed `deferred` task to recover. (3) The LandedOutbox row γ retains in this
  corner is not unowned: γ's `holds-owned-and-bounded` clause surfaces the row's age in the reconcile
  summary and escalates past a configured 24h, and the human resolving that escalation is the row's
  exit owner. Revisit if a landed `deferred` task is ever observed.
- **Retiring `commit_effect_present_in_main` entirely.** Only its landing-detection call sites are in
  scope (D1).
- **Re-litigating `_RECOVERY` or `_RECONCILE_SWEEP_STATUSES`** — see §Sketch.

## Open questions (tactical)

1. **Where does η's reconciler hang?** A `BackgroundService` pass beside the stranded sweep
   (`harness.py:2088-2096`) or its own cadence. **Suggested:** reuse `BackgroundService` with an
   independent interval — the stranded sweep's 900 s is tuned for a different population. Decide in η.
2. **Does the no-op check belong in `branch_work_landed` or as a standalone reusable primitive?**
   Both are defensible; a standalone primitive is more testable and may have callers beyond this PRD.
   **Suggested:** standalone in `git_ops`, called by `branch_work_landed`. Decide in δ.
3. **Disposition reason vocabulary reuse.** The decline reasons may or may not want to reuse
   `LeaveReason` (`recovery_emission.py:170`) versus the contract's own vocabulary. **Suggested:** keep
   the contract's `reason` distinct and map it at the emission boundary. Decide in **4496** (α's owner).
4. **Where the extracted git-authority tier lives** (new at decompose, ζ decides).
   `escalation/src/escalation/git_authority.py` is recommended: `orchestrator/pyproject.toml:20`
   already declares `escalation` as a workspace dependency, so η can import it **statically**.
   Putting it in `shared/` would instead place a runtime reverse-import of
   `orchestrator.landing_evidence` inside `shared/` — strictly worse layering than the status quo.
