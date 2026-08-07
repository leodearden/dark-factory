# PRD: Fable architect trial v2 — effort-matched two-stage screen over an exhaustion-hard pool

**Status:** active — authored 2026-08-04 (autonomous design session; design input
is the verified brief `~/.claude/spawn-briefs/fable-trial-v2-prd-brief.md` and
the resolution text on esc-2864-1, which carries Leo's ruled reopening
condition; both honored here, not re-litigated).
**Project:** dark_factory. **Approach:** bare **B** (G5 heuristic: instrument
edits are small additive diffs inside one package whose seams the
eval-framework-revival B+H contract already covers; the campaign itself is a
linear gate chain — no new cross-module seam is introduced).

## Goal

Re-open the fable admission question on evidence the completed campaign could
not produce: an **effort-matched** (`architect-fable-max` vs
`architect-opus-max`) two-candidate screen over fixtures **selected for
incumbent difficulty** by a stage-1 calibration pass — plus the instrument
fixes without which any further campaign spend is wasted, and a UsageGate on
`run_architect_eval` so cap exclusions can no longer silently bias a
comparison.

Terminal user-observable surface: a committed v2 decision record and a
resolved born-at-L2 ruling gate in which Leo ratifies admission, declines
again, or declares the historical seam exhausted and pivots the programme to
purpose-authored briefs.

## Premise (G6)

esc-2864-1 (fable admission ratification gate, task 2864) was **RULED DECLINE
by Leo on 2026-08-04**. claude-fable-5 was not admitted to
`routing.allowed_models` or the ladder; no per-model ceiling was set; all
three PRD reaches were declined; task 2544 (Routing ξ) was cancelled as
workless. The decline was on cost (precisely measured, adverse: fable $4.429
vs incumbent $3.693 per usable plan) and the absence of a significant quality
difference (verdict INDISTINGUISHABLE-FROM-INCUMBENT; exact permutation
p=0.1356). It was explicitly **not** a finding that fable is unreliable.

The resolution's reopening condition, **verbatim** (authoritative source:
esc-2864-1 resolution text, dark-factory escalation queue, port 8102):

> REOPENING CONDITION. This decline is on the evidence as it stands and is not a
> permanent judgement. Re-open on an effort-matched (fable-max vs opus-max)
> 2-candidate screen, selected by a stage-1 calibration pass: incumbent-only,
> 1 trial, at production ceilings (120 turns / $15) rather than the 50/60 fixture
> defaults, over the ~41 historical tasks whose architect hit
> max_turns/max_budget_usd at today-scale ceilings (events.data subtype =
> error_max_turns|error_max_budget_usd). Stratify by incumbent behaviour and
> DISCARD ONLY the ceiling band (plans 3/3 at high quality); RETAIN both the
> intermittent band and the band the incumbent cannot plan at all. The no-plan
> band is not noise — it is where the hypothesis under test lives ("fable can
> plan tasks the incumbent cannot"), and excluding it would repeat, at the floor,
> the same select-on-incumbent-success bias that limits the original corpus.
> Score the no-plan band on planRate (plan_steps > 0), which is judge-free and
> therefore valid without a reference diff; plan_quality applies only where a
> valid `reference` block exists. Report cap_excluded per candidate: architect
> cells run without a UsageGate, and the costlier candidate is more exposed to
> differential exclusion. NOTE: zero turn-exhaustions have occurred in 5,744
> architect invocations since 2026-06-01, so that pool may no longer be hard for
> the incumbent; if calibration finds no discriminating fixture, the historical
> seam is exhausted and a re-test requires purpose-authored briefs.

This PRD implements that condition. Two motivating problems:

**P1 — selection bias.** The v1 hard subset contained only tasks the incumbent
can complete: on the five fixtures where `architect-opus-max` planned 3/3,
fable's mean plan_quality lead is +0.0073; on the one fixture where the
incumbent fell short (`reify_task_12`, 1/3) it is +0.3233. The +0.0600
headline is one large effect diluted by five zeros, and difficulty moderation
/ missing ground truth / a coin flip are perfectly confounded at n=1.

**P2 — three instrument defects**, all verified in code, none flagged by
`plans/fable-architect-eval-decision-2026-07-30.md`: (1) effort mismatch —
`architect-fable-high` at effort `high` vs `architect-opus-max` at `max`
(`evals/configs.py` `ARCHITECT_EVAL_CONFIGS`), with the opus high→max step
worth +0.0317 inside the same screen, biased against fable; (2) empty
reference diff — `runner.py` reads `task.get('reference') or {}` and never
falls back to top-level `post_task_commit`, so `reify_task_12`,
`reify_task_27`, `df_task_18` were judged on plausibility, not fidelity;
(3) judge cost unrecorded — `judge_cost_usd` reads 0.00 across all 235
architect cells, so every v1 dollar figure understates absolute cost.

**Premise caveat carried into the design:** there have been ZERO
turn-exhaustions in 5,744 architect invocations since 2026-06-01 (last
anywhere 2026-05-27), confounded four ways (Opus 4.7→4.8→5, ceiling 75→120,
plan salvage `ca6fdba50e`, prompt evolution). **The live possibility is that
today's incumbent plans all 41 candidates.** Stage 1 exists to discover
exactly that for ~$150 instead of ~$400+; on that outcome the honest
conclusion is that the historical seam is exhausted, and the branch taken is
a pivot to **purpose-authored ambitious briefs** (a NEW PRD, authored on
Leo's instruction at the γ2 gate — not this one).

## Substrate facts (G3) — brief claims re-verified this session, plus four refinements

All brief facts confirmed against main on 2026-08-04. Session refinements:

- **Per-candidate `cap_excluded` already exists in the report layer** (tasks
  3118/3302/3379): `report.py`'s per-config accumulator carries
  `plan_quality_cap_excluded`, and the invariant prose is explicit that a
  cap-tainted cell is excluded from means and counted, never averaged as
  zero. Scope A's "mandatory reporting half" therefore reduces to *surfacing
  those counts in the v2 campaign driver's output*, not building them.
- **The UsageGate retrofit has a ready-made seam**:
  `shared.cli_invoke.invoke_with_cap_retry(usage_gate, label, **invoke_kwargs)`
  wraps `invoke_agent` with account failover, session-resume-on-cap, bounded
  patience (`cap_wait_sanity_secs` / `max_cap_retries` →
  `AllAccountsCappedException`), and has a standalone non-workflow caller
  precedent (`dry_run_unblock.py:340`). Feasibility for `run_architect_eval`
  is HIGH: construct the gate exactly as `run_eval` does (enabled-guarded,
  warn-and-degrade) and swap the bare `invoke_agent` call. The brief's
  fallback (quiet-window low `--max-parallel`) is contingency only.
- **`EvalConfig.max_budget_usd` defaults to $20** — v1 architect cells ran at
  50 fixture-default turns / $20 config-default budget, i.e. *neither* axis at
  the production 120/$15. The v2 recipes pin both explicitly.
- **Fixture-dir isolation is structural**: `cli._load_fixture_dir` globs
  `*.json` non-recursively, so a sibling fixtures directory can never leak
  into default eval runs.

Standing brief facts relied on (verified): the harness bypasses
`routing.allowed_models` (`run_architect_eval` calls `invoke_agent` directly
with `model=config.model`; the allowlist check lives only in
`routing_dispatch`) so **no admission is needed to gather this evidence**;
produced plans are NOT persisted per cell and eval worktrees are cleaned, so
**no cheap re-judge of v1 is possible**; `eval-sample` has no task-id
selector, so minting drives `capture_reference` / `build_fixture_record` /
`pin_eval_branch` from an explicit merge-SHA list; adversarial fixtures add
nothing to a plan-only screen; `eval-ofat` runs 8 candidates including ~10x
implementer/judge cells — the v1 campaign's 40-line custom driver
(`data/eval-campaign/fable_architect_only.py`) is the copy source;
`shared.usage_gate.scope_for` + `scoped_cap_models` (default
`['claude-fable-5']`) scope-aware failover is live (`cli_invoke.py` slot
loop); the load-bearing scoring behaviour that must NOT change:
`tainted = arch_unmeasurable and not is_scorable_plan(plan)` (`runner.py:731`)
— an architect that ran fine and produced nothing scores a genuine 0.0 and is
kept, a timeout is deliberately not tainted, only transport refusals with no
scorable artifact are excluded.

## Resolved design decisions

1. **Instrument single-ownership honored (eval-revival decision 11).** All
   instrument-code edits — `configs.py`, `runner.py`, `metrics.py`/`report.py`
   if touched — land as **eval-framework-revival lane tasks** (the ο/2825 and
   π/2861 paired-edit precedent): filed by this PRD's decompose batch with
   revival attribution and a paired amendment to
   `plans/eval-framework-revival-prd.md` + its manifest. This PRD's own tasks
   (fixtures, driver, gates, analysis) edit nothing in the instrument.
   `architect-fable-max` is precisely the tactical add π's text anticipated
   "if the consumer campaign's screen shows effort sensitivity" — now
   measured (+0.0317).
2. **UsageGate lands via `invoke_with_cap_retry`** with eval-appropriate
   bounded patience (order 30–60 min / a small `max_cap_retries`, NOT the
   14-day default — a fully-capped pool must fail loud into the existing
   `cap_tainted`/`cap_excluded` backstop rather than hang a campaign;
   INV-4 storm escape). The tainted predicate and timeout semantics stay
   byte-identical (pinned must-not-touch above).
3. **Campaign runs are operator out-of-band gates** (the 2848/τ1 pattern):
   `task_kind='deterministic'` pure gates filing born-at-L2 escalations that
   carry the operator recipe; eval campaigns run in an operator session
   against the live OAuth pool, never in task worktrees.
4. **Exactly two Leo sittings**, both genuinely owner-level and AFK-tolerant
   (gates sit pending until he returns): γ2 (banding ratification +
   comparison-regime ruling + stage-2 spend authorization + pivot authority)
   and η (admission re-ruling). Stage-1 execution (γ1) and stage-2 execution
   (δ) are operator/watcher-executable.
5. **Equal-cost vs equal-turns is NOT defaulted** (Leo raised it and did not
   rule). It is carried to γ2 as an explicit ruling with three options, each
   with a fully-specified recipe: **(a) equal-cost** — both candidates at
   `max_budget_usd=15`, the decision-relevant comparison for routing ($/usable
   plan is what gets paid), fable gets fewer turns per dollar; **(b)
   equal-turns** — both at `max_architect_turns=120` with fable's
   `max_budget_usd` lifted (e.g. $25) so turns, not price, bind — isolates
   capability; **(c) both arms** — runs (a) and (b), roughly +50–100% stage-2
   cost, separates the questions empirically. The PRD deliberately does not
   recommend; the γ2 escalation must present all three with the realized
   retained-set cost projection.
6. **Banding rule (Leo-corrected; load-bearing).** Stage 1 is 1 trial per
   fixture, so "plans 3/3" is operationalized: **discard ONLY the unambiguous
   ceiling band** — the single trial produced a plan AND a valid `reference`
   exists AND plan_quality ≥ Q_ceiling. **Retain everything else**: planned
   but below Q_ceiling, planned without valid reference, and the no-plan
   band. The no-plan band is where the hypothesis under test lives; it is
   scored on **planRate (`plan_steps > 0`), which is judge-free** and valid
   without a reference diff. plan_quality applies only where a valid
   `reference` block exists. Q_ceiling is **empirically anchored** (derived in
   γ1 from the v1 campaign's incumbent cells on validly-referenced fixtures),
   recorded provisional in the calibration report, and **ratified or adjusted
   by Leo at γ2** — never a bare guessed threshold (G6). Ambiguity → retain
   (misbanding-to-retain costs ~$20 of stage-2 spend; misbanding-to-discard
   loses signal permanently).
7. **Fixture pool.** The 41 distinct tasks whose architect exhausted
   max_turns/max_budget_usd at today-scale ceilings (turns-at-exhaustion 121;
   census 2026-08-04: reify 36, dark_factory 4, know_live 1 — attribution via
   `json_extract(events.data,'$.subtype')` on `invocation_end` rows, per-project
   `<project_root>/data/orchestrator/runs.db`), **plus** the three back-filled
   legacy fixtures (`reify_task_12`, `reify_task_27`, `df_task_18`) as
   continuity fixtures — an explicit extension beyond the resolution's pool,
   justified because `reify_task_12` carried the entire v1 result and
   re-banding it under a valid reference closes the n=1 confound. The reify
   skew is **accepted** (Leo: "Reify is the justifying use case for the
   creation of Dark Factory"); no project-balancing requirement. Curation:
   drop thin briefs (a 94-char brief that defeated the architect is a
   specification failure, not a difficulty signal) via a **committed
   include/exclude table** (task_id, brief length, decision, reason) rather
   than a guessed length threshold; exclude the two cancelled candidates
   (reify 3378, 3586) unless their abandonment reason is confirmed benign;
   skip adversarial fixtures. SPLIT-minority candidates without a clean
   single `Merge task/<id>` SHA are minted **planRate-only** (no `reference`
   block, loud-marked — see D9).
8. **v2 fixtures live in a committed sibling directory**
   (`orchestrator/src/orchestrator/evals/tasks_hard_v2/`), never merged into
   the standing corpus — the non-recursive fixture glob keeps every default
   eval run at today's cost. Fixtures pin production ceilings fixture-side:
   `max_architect_turns=120`, `timeout_minutes` sized from v1 wall-clock data
   so the timeout cannot bind before turns/budget (provisional floor 180;
   derivation recorded by β1) — the 50/60 defaults would manufacture
   artificial failures on a hard set, and a timeout scores a kept 0.0.
9. **Silent empty-reference degradation becomes loud** (INV-2
   structured-facts): alongside the 3-fixture back-fill, `run_architect_eval`
   records a structured `judged_without_reference` marker on any plan_quality
   cell scored with an empty `reference_diff`, aggregated per config in the
   report. ζ uses it mechanically to bound plan_quality validity; it can
   never again be discovered by archaeology.
10. **The v2 campaign driver is committed** (`scripts/` — unlike v1's
    gitignored `data/eval-campaign/fable_architect_only.py`, whose
    reproducibility depended on one file in one checkout). It copies the v1
    driver's shape and emits, per candidate: cell results, planRate,
    plan_quality (validity-bounded), `cap_excluded`, and
    `judged_without_reference` counts.
11. **Judge pinned to incumbent** (OFAT discipline — vary only the
    architect); judge cost recorded (ι3) so v2 dollar figures are honest.
    v1-vs-v2 absolute costs are not directly comparable (v1 excluded judge
    cost); ratios are. ζ must state this.
12. **No pre-filed flip task.** On ADMIT, η's resolution instructs filing a
    NEW routing task (successor to cancelled 2544's scope under
    `plans/adaptive-model-routing-prd.md`); on DECLINE or PIVOT nothing is
    filed. Pre-filing would repeat the 2544 file-then-cancel churn.

## Cross-PRD relationships (G4)

| Other PRD / artifact | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `plans/eval-framework-revival-prd.md` | this PRD consumes the instrument; instrument edits (ι1–ι4) land in ITS lane | `configs.py` / `runner.py` / `metrics.py` / `report.py` edits | **eval-revival** (decision 11; ο/π paired-edit precedent) — this batch files them with revival attribution + paired PRD/manifest amendment | queued by this decompose |
| `plans/usage-gate-model-scoped-caps-prd.md` (2855–2859, done) | consumes | `UsageGate`, `scope_for`, `scoped_cap_models` default `['claude-fable-5']`, `invoke_with_cap_retry` | that PRD (substrate, closed) — **no edits**; ι4 is a consumer call-site | wired |
| `plans/fable-architect-eval-admission-prd.md` (τ1–τ3 closed) | successor | the reopening condition on esc-2864-1 | this PRD | n/a |
| `plans/adaptive-model-routing-prd.md` (ξ=2544 cancelled) | downstream on ADMIT only | new flip task filed by η's resolution | that PRD's lane, filed later | deliberate no-pre-file (D12) |
| task 3099 (done) + ticket `tkt_0RRWM6P4MT95PWDG4N5E9F27KG` | adjacent instrument defects | `composite_score`/`tests_pass`; judge-vs-floor on stepless plans | already tracked — **not duplicated here** | n/a |

## Pre-conditions for activating

None external. All substrate verified live on main this session; the
usage-gate scope substrate is done; no dependency on task 2864's terminal
re-run (its escalation is resolved and the decision recorded).

## Decomposition plan

Instrument phase (eval-framework-revival lane, parallel; each self-declares
`task_kind='normal'`):

- **ι1 — `architect-fable-max` candidate config.** Add
  `EvalConfig('architect-fable-max', 'claude', 'claude-fable-5', 'max', role='architect')`
  to `ARCHITECT_EVAL_CONFIGS` + extend the candidate-set pin test; existing
  candidates byte-unchanged (parity discipline). Signal:
  `get_config_by_name('architect-fable-max')` resolves; pin test green.
  Unlocks: δ.
- **ι2 — reference back-fill + loud empty-reference marker.** Back-fill
  `reference` blocks (from each fixture's own `post_task_commit` SHA) on
  `reify_task_12`, `reify_task_27`, `df_task_18`; add the
  `judged_without_reference` structured marker (D9) to architect cell metrics
  + per-config report aggregation. Signal: a deterministic test materializes a
  non-empty reference diff for each back-filled fixture via
  `get_diff_between_commits` (no LLM spend); report schema carries the new
  count. Unlocks: γ1.
- **ι3 — record judge cost on architect cells.** Populate
  `metrics.judge_cost_usd` from the plan-judge invocation result in
  `run_architect_eval` (report layer already aggregates it —
  `report.py` per-config `judge_cost_usd` + surfacing). Signal: unit test
  wiring a judge result cost into the persisted cell; γ1's report shows
  nonzero judge cost per cell. Unlocks: γ1.
- **ι4 — UsageGate in `run_architect_eval`.** Construct the gate as
  `run_eval` does (enabled-guarded, warn-and-degrade) and swap the bare
  `invoke_agent` architect call for `invoke_with_cap_retry` with bounded
  patience (D2). The judge call may adopt the same seam if trivial; the
  tainted predicate, timeout semantics, and scoring behaviour stay
  byte-identical (pinned). Signal: unit test simulating a cap-hit shows
  failover invoked and the cell completing on the second account; the
  tainted-predicate tests unchanged. Unlocks: γ1. Feasibility fallback
  (only if implementation falsifies the assessment): run campaigns at low
  `--max-parallel` in a quiet window — the per-candidate `cap_excluded`
  visibility (already in the report layer + D10 driver) is mandatory
  regardless and does not depend on this task.

Campaign substrate (this PRD's lane):

- **β1 — mint the curated v2 hard pool.** ~20-line driver over
  `capture_reference` / `build_fixture_record` / `pin_eval_branch` from an
  explicit merge-SHA list; committed fixtures in `tasks_hard_v2/` (D8) with
  production ceilings pinned and derivation of `timeout_minutes` recorded;
  committed curation table over all 41 census candidates (+ the 2 cancelled,
  with their abandonment-reason check); SPLIT-minority minted planRate-only.
  Signal: committed fixture JSONs load via `_load_fixture_dir`; curation
  table accounts for every candidate; default-corpus runs untouched.
  Unlocks: γ1.
- **β2 — committed v2 campaign driver** (D10). Signal: dry-run over
  `tasks_hard_v2/` enumerates the expected cell matrix and the output schema
  carries planRate / plan_quality / cap_excluded / judged_without_reference
  per candidate. Unlocks: γ1, δ.

Campaign gates (linear chain; each `task_kind='deterministic'`, pure gate,
born-at-L2 escalation carrying its recipe — no worktree, no code):

- **γ1 — stage-1 calibration run gate** (operator-executable; depends ι2, ι3,
  ι4, β1, β2). Recipe: incumbent-only (`architect-opus-max`), 1 trial per
  fixture, plan-only cells, `max_budget_usd=15` + fixture 120-turn ceilings;
  re-run any cap-tainted cell so every fixture gets one admissible cell.
  Output: committed calibration report
  (`plans/fable-trial-v2-calibration-<date>.md`) partitioning the pool into
  ceiling / intermittent / no-plan bands with per-fixture planRate,
  plan_quality where valid, the Q_ceiling derivation (D6), cap_excluded and
  judge-cost actuals, and the realized stage-2 cost projection per regime
  option. **This report is the artifact that decides whether stage 2 runs at
  all** — the PRD's G2 top signal. ~34±4 cells ≈ $150–300 including
  now-recorded judge cost.
- **γ2 — banding ratification + regime ruling + stage-2 authorization gate**
  (LEO; depends γ1). The escalation presents the calibration report and asks
  Leo to: (1) ratify or adjust the band partition and Q_ceiling; (2) rule the
  comparison regime — equal-cost / equal-turns / both arms (D5, all three
  presented with costs; deliberately un-defaulted); (3) authorize stage-2
  spend — OR, if calibration found no discriminating fixture, declare the
  historical seam exhausted: resolution then instructs cancelling δ,
  re-scoping ζ to record the pivot, and authorizes authoring the
  purpose-authored-briefs PRD as the successor route. Signal: resolved
  escalation with the recorded ruling.
- **δ — stage-2 screen run gate** (operator-executable; depends γ2, ι1, β2).
  Recipe parameterized by γ2's ruling: `architect-fable-max` vs
  `architect-opus-max`, 3 trials, over the retained fixtures, judge pinned to
  incumbent. Output: results artifact under `data/eval-campaign/` (raw dumps
  stay gitignored, v1 precedent) + the per-candidate table including
  cap_excluded symmetry recorded in the gate resolution. ~48 cells
  (retained-set dependent) ≈ $250–400+ (both-arms roughly doubles the fable
  side). Signal: gate resolution carrying the per-candidate table with zero
  unexplained cap-excluded asymmetry between candidates.
- **ζ — v2 decision record** (normal docs task — the 2863 shape: author and
  commit the record through the standard pipeline; depends δ; re-scoped by
  γ2's resolution on the pivot branch).
  Committed `plans/fable-architect-trial-v2-decision-<date>.md`: planRate on
  the no-plan band (judge-free), plan_quality bounded by
  judged_without_reference validity, paired per-fixture stats, $/usable-plan
  with judge cost included (v1 comparability caveat, D11), cap-excluded
  symmetry, and a recommendation clearly separated from raw observation
  (INV-2 discipline, the 2863 shape). On the pivot branch: records the seam
  as exhausted with the calibration evidence and names purpose-authored
  briefs as the successor. Signal: the committed record.
- **η — admission re-ruling gate** (LEO; deterministic pure gate; depends ζ).
  Files the born-at-L2 escalation naming ζ's record and asking Leo to ratify
  or decline admission on the new evidence — or confirm the pivot. On ADMIT
  the resolution instructs filing the new flip task (D12). Signal: resolved
  escalation with Leo's recorded ruling — the PRD's terminal consumer (G1).

## Out of scope

- Any production routing change (`allowed_models`, ladder, ceilings) — η's
  resolution gates all of it, and execution belongs to a future
  adaptive-model-routing task.
- Authoring purpose-authored ambitious briefs — the pivot branch's successor
  PRD, authored only on Leo's γ2 instruction.
- Re-judging or re-scoring the v1 campaign (impossible: plans not persisted).
- An end-to-end confirm stage (implementer cells) — the reopening condition
  asks for the architect screen only; η's ruling may demand a confirm stage
  as its own condition, which would be a follow-up.
- Adversarial fixtures; `eval-ofat` changes; task 3099 / judge-vs-floor
  ticket scope (already tracked).

## Open questions (surfaced but not decided in this session)

1. **Final `timeout_minutes` per fixture.** Provisional floor 180; β1 derives
   from v1 wall-clock dumps so the timeout cannot bind before turns/budget.
   Decide during β1.
2. **Q_ceiling realized value.** Derivation method fixed (D6: v1 incumbent
   cells on validly-referenced fixtures); the number is computed in γ1 and
   ratified at γ2.
3. **Thin-brief exclusions.** The curation criterion is "brief fails to state
   an implementable goal", judged during β1 with every decision recorded in
   the committed table — auditable rather than threshold-pinned.
4. ~~**Driver filename/placement under `scripts/`.** Decide during β2.~~
   **RESOLVED in β2 (task 3632).** The driver landed at
   `scripts/run_fable_trial_v2_campaign.py`, with its tests at
   `scripts/tests/test_run_fable_trial_v2_campaign.py` (merged as
   `a98e91997a`, "Merge task/3632 into main"). All logic lives in that
   one script: no new module was added inside
   `orchestrator/src/orchestrator/evals/` — the β2 merge touched exactly
   two files, both under `scripts/`. The instrument (`run_ofat_stage`,
   `build_plan_quality_report`, `produced_a_plan`, `get_config_by_name`)
   is consumed unmodified.

   Why that constraint bound: decision **D1** above ("Instrument
   single-ownership honored") states this PRD's own tasks "edit nothing
   in the instrument", and task 3632 declared `Modules touched: scripts/`
   accordingly. A module under `orchestrator/evals/` would have crossed
   the eval-framework-revival lane's single-ownership boundary. The
   `run_<campaign>.py` filename spelling follows the committed
   eval-driver siblings `run_judge_ofat_pilot.py` and `run_vllm_eval.py`
   (cross-referencing **D10** by label, which already names the driver as
   committed under `scripts/`; this entry supplies the filename D10 left
   generic).
