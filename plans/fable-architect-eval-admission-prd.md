# PRD: Fable-architect eval on the hard-fixture subset + ratified admission

**Status:** active — authored 2026-07-20 (autonomous session; design input is the
verified brief `~/.claude/spawn-briefs/fable-architect-followup-prd-2026-07-20.md`;
Leo's constraints honored, not re-litigated). **Project:** dark_factory.
**Approach:** bare **B** — small task count, no new load-bearing seam of its
own; the high-stakes seam (UsageGate scoping) is the sibling PRD
`plans/usage-gate-model-scoped-caps-prd.md`, consumed here.

## Goal

Measure whether a **Fable-tier architect** (`claude-fable-5`) beats the
production architect (`architect-opus-max`) on **end-to-end quality/$ over the
HARD subset** of the eval suite — and, if Leo ratifies the verdict, admit fable
as an architect option (adaptive-routing task ξ = **2544**, amended) reachable
via overrides, the retry ladder's final rung, and a ratified
complexity/plan-shape architect rule.

**User-observable surface (G2 top signal):** a committed decision record under
`plans/` carrying the hard-subset OFAT composite table comparing
`architect-opus-max` / `architect-opus-high` / `architect-sonnet-high` /
`architect-fable` (per-architect plan quality, $/plan, and — for the screen
winner vs the incumbent — end-to-end composite and $/done with CIs), plus
either the ratified admission (fable in `allowed_models`, reload disposition)
or the recorded decision not to admit.

## Background

- The architect is the **#1 cost line** (42.7% of DF spend) and the highest
  quality-leverage role: a better plan means fewer implementer retries, debug
  cycles, and review kickbacks. Precedent: the reviewer-effort sweep (5×sonnet
  panel → 1×opus reviewer, 6.3× F1/$). Leo's fleet-economics heuristic —
  "Fable ONLY for synthesis" — points the same way: architect planning is the
  most synthesis-like step in the pipeline.
- **Easy fixtures don't discriminate architects** (opus-max ≈ opus-high ≈
  composite 1.0), so this eval runs on the **hard subset**: the adversarial
  variants (`df_task_2284_adv_regression`, `df_task_2339_adv_verify`,
  `df_task_2430_adv_plan`), the two high-complexity reify fixtures
  (`reify_task_12`, `reify_task_27`), and known-hard `df_task_18` — all
  verified present in `orchestrator/src/orchestrator/evals/tasks/`.
- **Key new fact (Leo, 2026-07-20):** all pool accounts have fable access at
  50% of token spend each — the eval can run against the pool now. Production
  admission additionally requires the sibling PRD's scope-aware failover.
- Task **2847** (eval-worktree bootstrap fix + `scripts/eval_bootstrap_smoke.sh`
  go/no-go gate) is **done on main** — the prerequisite that invalidated the
  killed 2026-07-20 campaign is cleared.
- Task **2539** (routing ι, architect effort decide-and-act) runs the
  opus-max-vs-high campaign on the full suite through the same infra
  (gated on pure-gate 2848); this PRD layers the fable candidate on top and
  reuses ι's refined methodology (eval-confirm only when the screen winner
  differs from the production setting).

## Resolved design decisions

1. **Instrument single-ownership respected (eval-revival decision 11; the ο
   precedent).** Adding the `architect-fable` candidate is an eval-instrument
   edit and therefore lands as eval-revival's **own task π** (paired edit to
   `plans/eval-framework-revival-prd.md` committed with this PRD), not as an
   edit from this PRD's adoption tasks. This PRD's tasks only *consume* π.
2. **Hard-subset via `--tasks-dir`, no candidate-selection flag.**
   `eval-ofat` screens all `ofat_candidates()` and has no per-candidate
   selection; restricting the fixture dir is the supported lever. The extra
   incumbent implementer/judge cells that ride along are accepted as baselines
   (bounded cost), not engineered away — adding a selection flag would be
   another instrument change with no consumer beyond this run.
3. **Methodology mirrors ι's refinement:** OFAT screen on the hard subset
   (architect cells are plan-only: 1 live architect call + 1 judge per cell,
   downstream frozen — cheap), then `eval-confirm` (both-live end-to-end,
   ≥3 trials) **only if** an architect other than the production setting tops
   the screen, comparing it against `architect-opus-max` for end-to-end
   composite and $/done.
4. **Admission is ALWAYS Leo-ratified — never auto-flipped.** ι's
   decide-and-act auto-applies a clear pass because it re-tunes already-admitted
   models; admitting a new top-tier model is **fleet-autonomy expansion**, which
   the owner ratifies (standing directive). Mechanism: a deterministic
   **pure-gate task** (born-at-L2, `always_escalates`, no `before_done`) naming
   the decision record; the flip itself stays in the amended 2544.
5. **The eval does NOT wait for the cap-scoping PRD.** It is a bounded spend
   against the 50%-cap pool accounts. Accepted residual risk under pre-scope
   semantics: a fable cap-hit mid-campaign CAPs that whole account until
   `resets_at` — the campaign escalation instructs the operator to run in a
   low-contention window. **Admission** (2544) does gate on the sibling PRD's
   integration gate.
6. **2544 amended in place, not refiled** (curator-vector precedent): its
   description drops the stale premises ("only the interactive account is
   proven"; "a fable cap-hit CAPs the whole account — accepted") in favor of
   the pool-wide 50% fact + scope-aware semantics, gains the ratification +
   scope-gate deps, and — on ratified admission — additionally installs the
   complexity/plan-shape architect rule the verdict recommends (superseding its
   original "never a fleet default rule" with "as ratified").
7. **Fable candidate effort = `high`** (one candidate). An `architect-fable-max`
   variant is a tactical add inside π if the screen looks effort-sensitive;
   starting with both would double fable cells for a hypothesis the opus
   max-vs-high axis (ι) already probes.

## Pre-conditions for activating

- **Task 2847 — done on main** (verified: `scripts/eval_bootstrap_smoke.sh`
  landed; `done_provenance` found_on_main 2026-07-20). No further external
  gate for the eval thread.
- G3 substrate verified this session: `ARCHITECT_EVAL_CONFIGS`
  (`evals/configs.py:435` — opus-max/opus-high/sonnet-high, no fable; adding
  one is a one-line `EvalConfig(..., role='architect')`); `eval-ofat` /
  `eval-confirm` CLI with `--tasks-dir`/`--trials` (cli.py:1297/1360); the six
  hard fixtures on disk; `probe-models` probing `FABLE_CANDIDATE_MODEL` per
  account (routing.py:59); eval memory null-sentinel + live OAuth pool sharing
  (brief, re-confirmed by the 2847 fix record).
- Fable pool access at 50%/account: Leo's stated fact; the `probe-models`
  artifact (`config/model-availability.yaml`) provides the per-account
  empirical confirmation the admission gate cites.

## Cross-PRD relationship

| Other PRD | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `plans/eval-framework-revival-prd.md` | consumes | task **π** (architect-fable OFAT candidate) — instrument edit in eval-revival's lane per its decision 11, added by paired edit (ο precedent); this PRD's campaign/decision tasks only consume it | **eval-revival** owns the instrument; **this-prd** owns the campaign + decision | π filed with this batch, prd_path → eval-revival |
| `plans/adaptive-model-routing-prd.md` | produces (the verdict) | ξ = task **2544** stays adaptive-routing's; amended in place (decision 6) and re-wired: deps += ratification gate τ3 + the sibling PRD's ε | **adaptive-routing** owns the flip; **this-prd** owns the evidence + ratification gate | amendment + dep wiring at decompose |
| `plans/usage-gate-model-scoped-caps-prd.md` (sibling) | consumes | scope-aware failover is the production-safety prerequisite: τ3 (ratification) and 2544 dep its integration gate ε; the eval campaign deliberately does not (decision 5) | **sibling PRD** | dep wired at decompose |
| task 2539 / 2848 (routing ι campaign) | shares infra, no dep | same eval instrument + out-of-band operator-run pattern; ι compares opus effort tiers on the full suite, this PRD compares fable on the hard subset. No data dependency either way — the hard-subset run produces its own opus baselines on the same fixtures | — | independent |

## Decomposition plan

- **π — `architect-fable` OFAT candidate (eval-revival's lane)**
  *(intermediate → unlocks τ1)*. Add
  `EvalConfig('architect-fable-high', 'claude', 'claude-fable-5', 'high',
  role='architect')` to `ARCHITECT_EVAL_CONFIGS` + a test pinning the candidate
  set (existing candidates byte-unchanged — the parity discipline). Filed with
  `prd_path` → eval-revival. Signal (intermediate): `ofat_candidates()` lists
  the fable candidate with role='architect'; existing config tests green.
  Modules: orchestrator/src/orchestrator/evals/configs.py, orchestrator/tests.
- **τ1 — Hard-subset campaign gate (deterministic pure gate:
  `always_escalates`, no `before_done`)** *(intermediate → unlocks τ2)*.
  Born-at-L2 escalation handing the operator the out-of-band run (the
  2848 pattern — campaigns run in an operator session against the live pool,
  not in a task worktree): run `scripts/eval_bootstrap_smoke.sh` (go/no-go),
  assemble the six hard fixtures into a subset dir, `eval-ofat --tasks-dir
  <subset> --trials 3`, then `eval-confirm --arch <winner> --impl <incumbent>`
  vs `architect-opus-max` iff the screen winner differs from production;
  capture stdout composite tables + result JSONs under `evals/results/`.
  Low-contention window; est. single-digit-$ screen (architect cells
  plan-only), tens-of-$ confirm. Signal: escalation resolved with the result
  artifacts present. Deps: π (2847 already done).
- **τ2 — Committed decision record** *(intermediate → unlocks τ3)*. Analyze
  the campaign artifacts into
  `plans/fable-architect-eval-decision-<date>.md`: per-architect hard-subset
  plan quality + $/plan; end-to-end composite + $/done + CI95 for the confirm
  pair; a recommendation (admit with which reach — overrides / retry rung /
  plan-shape rule — or don't) with the marginal-band reasoning explicit.
  Signal: the record committed, tables populated from the artifacts (not
  hand-waved), recommendation stated. Deps: τ1. Modules: plans/.
- **τ3 — Admission ratification gate (deterministic pure gate)** *(leaf)*.
  Born-at-L2 escalation naming τ2's record and asking Leo to ratify or decline
  fable admission + the recommended reach. Resolution text records the
  decision; the flip itself is 2544's. Signal: the escalation filed and
  resolved with Leo's recorded decision. Deps: τ2, **sibling-ε**
  (usage-gate-model-scoped-caps integration gate — ratification is only
  actionable once scope-safe failover exists).
- **Decompose-time amendment (not a task): task 2544** — description updated
  per decision 6; deps += τ3 + sibling-ε (both direct, so a hasty τ3
  resolution cannot bypass the scope gate). Its existing probe-artifact HARD
  GATE stands.

Leaf/intermediate: π, τ1, τ2 are intermediates (each names its consumer above);
τ3 is the leaf carrying the operator-observable end signal (ratified-or-declined
admission decision on record). The G2 escape hatch does not apply — every
intermediate's output is consumed in-batch.

## Out of scope

- The UsageGate scope mechanics — sibling PRD.
- The admission flip + routing-rule install — task 2544 (adaptive-routing),
  executed only after τ3 ratification.
- Any further eval-instrument work beyond π (fixture re-cuts, new scorers,
  candidate-selection flags) — eval-revival's lane.
- Fable for roles other than architect (judge/implementer/reviewer trials are
  κ's / eval-revival's axes; "Fable ONLY synthesis" says start and likely end
  at the architect).
- Auto-flip on a clear pass — deliberately excluded (decision 4).

## Open questions (tactical — surfaced, not blocking)

1. **Subset-dir mechanics in τ1** — symlinks vs copies vs a committed
   `evals/tasks-hard/` listing. Suggested: a throwaway dir of copies named in
   the escalation (no instrument/corpus change). Decide in τ1.
2. **`df_task_18` `max_review_cycles=2`** — confirm the fixture JSON carries it
   or pass the override at run time. Decide in τ1.
3. **`architect-fable-max` variant** — add inside π only if τ1's screen shows
   effort sensitivity worth a second fable column. Decide in π/τ1.
