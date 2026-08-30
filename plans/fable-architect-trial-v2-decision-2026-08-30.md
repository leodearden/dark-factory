# Fable-architect trial v2 decision record — tranche 1, the no-plan band (2026-08-30)

**Task:** 3636 (ζ) · **PRD:** `plans/fable-architect-trial-v2-prd.md` §ζ ·
**Ruling:** D9 (Leo, 2026-08-30, via gates esc-3635-1 / esc-4761-1) ·
**Upstream:** δ = task 3635 (tranche 1, stopped early) ·
**Consumer:** η (task 3637, admission re-ruling gate)

> Throughout this record, "ruling D9" means **Leo's 2026-08-30 ruling** on the tranche-1
> readout, whose three parts are cited as ruling A (the finding), ruling B (the stopping
> decision) and ruling C (forensic-only disposition). Where this record instead means the
> PRD's numbered *design decision* 9 — σ's `judged_without_reference` marker — it says
> **"PRD D9"** explicitly. The two are unrelated and are never abbreviated the same way.

**Run:** `architect-fable-max` (candidate) vs `architect-opus-max` (incumbent), 3 trials
per fixture, over the 11-fixture **no-plan band** selected by γ1, judge pinned to the
incumbent, production ceilings, cells plan-only (one live architect call + one plan-judge
call per cell, downstream frozen). The tranche was designed as 66 cells and **stopped
cleanly by Leo at 53 cells on 2026-08-26T08:51:18Z**, because the band had been found
untestable for the stated hypothesis. Total spend **$230.91**, of which judge **$5.55**.

**No config change applied by this record.** Admitting a top-tier architect model is
fleet-autonomy expansion, which the owner ratifies — it is never auto-applied on a
favourable readout, and this readout is in any case not favourable-or-unfavourable (§8).
`routing.allowed_models`, the routing ladder and the per-model ceilings are untouched by
this record. Admission remains **Leo-ratified downstream at η (task 3637)**, unaffected by
anything written here. This record produces evidence and a recommendation only.

> ⚠️ **Before acting on task 3635's stored description, read §9.** That description still
> names the **withdrawn** McNemar stopping rule and a contingent "fresh tranche-2 gate".
> Neither governs. §9 is self-contained for exactly this reason.

## How to read this record (INV-2)

Sections 2 through 7 are **RAW OBSERVATION** — recomputed independently from the 53
per-cell result JSONs (§2, §4, §5, §6), or transcribed with explicit attribution from the
forensic investigation and γ1's calibration report where this record cannot itself
re-derive them (§3, §7). Section 8 is the **sole INTERPRETATION** section — the finding and
the recommendation. Sections 9 and 10 are a stale-rule notice and a reproduction appendix.

This separation is deliberate and load-bearing: a downstream reader (η, task 3637, and Leo
ruling on admission) should be able to trust §2-§7 **without trusting this record's
judgment**, and should be able to find that judgment confined to one clearly marked place.

1. Provenance and standing (this section)
2. Cell inventory, the early stop, and the cap_excluded symmetry check — RAW OBSERVATION
3. Decline-consistency: what the 47 no-plan cells actually are — RAW OBSERVATION
4. Paired per-fixture statistics — RAW OBSERVATION
5. `plan_quality`, validity-bounded — RAW OBSERVATION
6. Cost and $/usable-plan — RAW OBSERVATION
7. γ1 errata carried forward (live vs moot) — RAW OBSERVATION
8. Finding and recommendation — **INTERPRETATION**
9. 🔴 Stale-stopping-rule notice for task 3635 (McNemar withdrawal)
10. Appendix: reproduction recipe

---

## 1. Provenance and standing

_RAW OBSERVATION._

**Standing: FORENSIC-ONLY (ruling C).** The 53 completed cells are retained as
**decline-consistency evidence** and are **NOT usable as capability data**. No planRate,
`plan_quality`, cost-per-plan or Δ figure anywhere in this record may be cited as a measure
of either arm's planning capability. §3 and §8 establish why; §4 and §5 restate the caveat
at the point of use.

**No further cells will be run under the 66-cell tranche-1 design.** That design is
**SUPERSEDED, not paused**, by the redesigned eval set Leo is drafting. The 13 unrun cells
are not pending; they will never be run. Any downstream artifact implying a resumable
tranche 1 or a contingent tranche 2 is stale — see §9.

**Primary data paths.** Every figure in this record was computed or verified against:

```
/home/leo/src/dark-factory/data/eval-campaign/tranche1/
  cells/*.json                      53 per-cell result JSONs (26 fable / 27 incumbent)
  FINAL-READOUT.txt                 the operator's per-candidate + per-fixture tables
  analyse_tranche1.py               the ruled readout (paired cluster bootstrap)
  investigation/FINDINGS.md         the forensic investigation (51-cell cut, see §3)
  investigation/claims.txt          the decline payloads
  investigation/summaries/          per-cell tool-level transcript extractions + _index.json
```

> 🔴 **This entire tree is GITIGNORED** (root `.gitignore` line 9, root-anchored `/data/`)
> and exists **only in the main checkout**. It is on no ref, in no worktree, and in no
> backup that this repository controls. Once the campaign is closed as forensic-only,
> nothing protects it from deletion.
>
> **That is precisely why every number below is quoted verbatim rather than referenced.**
> This committed record is the durable artifact. §10's reproduction recipe names the paths
> as a convenience for a reader who still has the tree — never as the carrier of any figure.

**What this record does not re-derive.** The transcript forensics in §3 and γ1's
calibration measurements in §5 and §7 are transcribed **with attribution**, not
recomputed: reconstructing them would require re-running seven verification agents against
`/home/leo/src/reify` and re-analysing 45 γ1 cells, neither of which this task did. Where
this record verified a transcribed claim itself, it says so at the point of use (§3e is the
one place it materially did).

---

## 2. Cell inventory, the early stop, and the cap_excluded symmetry check

_RAW OBSERVATION._ Every figure in this section was **recomputed directly from
`cells/*.json`**, not transcribed from `FINAL-READOUT.txt`. Where the two agree, that is
independent corroboration rather than a transcription (v1's device); where they disagree,
this record says so (§3e is the one such place, and it is a definitional disagreement, not
an arithmetic one).

### 2a. Per-fixture × per-arm cell counts

| fixture | fable cells | incumbent cells |
|---|---|---|
| `df_task_2260` | 3 | 3 |
| `reify_task_12` | 3 | 3 |
| `reify_task_2324` | 3 | 3 |
| `reify_task_2531` | 3 | 3 |
| `reify_task_2573` | 3 | 3 |
| `reify_task_2699` | 3 | 3 |
| `reify_task_2778` | 3 | 3 |
| `reify_task_3883` | 3 | 3 |
| **`reify_task_4026`** | **2** ⚠️ | **3** |
| **TOTAL** | **26** | **27** |

⚠️ **`reify_task_4026` is the sole unbalanced fixture: 2 fable cells against 3 incumbent.**
Every other scored fixture is 3+3. This asymmetry is not cosmetic — it is the entire reason
the cell-level and fixture-level planRates diverge in the fable arm (§4b). 26 + 27 = **53**.

### 2b. The two unrun fixtures

The band was 11 fixtures; 9 were scored. `reify_task_4370` and `reify_task_4832` were never
run. Per `FINAL-READOUT.txt`, **both were protocol declines in γ1** (`reify_task_4370`:
already-done; `reify_task_4832`: false-premise), so the unrun remainder is drawn from the
same decline population as the scored fixtures and would not have changed the finding.

### 2c. The stop

**53 of 66 cells**, stopped cleanly by Leo at **2026-08-26T08:51:18Z**. Not a crash, not a
cap kill, not a budget exhaustion — an operator decision taken once the band was found
untestable for the stated hypothesis. Total spend **$230.91**, of which judge **$5.55**.

Recomputed from the cells and reconciled against `FINAL-READOUT.txt`'s
`TOTAL SPEND: $230.91 (judge $5.55)`:

| arm | cells | Σ `cost_usd` | Σ `judge_cost_usd` |
|---|---|---|---|
| `architect-fable-max` | 26 | $123.6355 | $2.0911 |
| `architect-opus-max` | 27 | $107.2742 | $3.4560 |
| **TOTAL** | **53** | **$230.9097** → **$230.91** | **$5.5471** → **$5.55** |

Both totals tie out to the cent against the operator's published figures.

### 2d. The `cap_excluded` symmetry check — CONFIRMED, not assumed

The δ gate's acceptance criterion is *zero unexplained cap-excluded asymmetry between
candidates*. It exists to catch **differential exclusion bias**: if one arm's cells were
dropped for cap starvation at a higher rate than the other's, the surviving cells would no
longer be a like-for-like comparison, and any Δ computed over them would be confounded.

Computed directly as `sum(metrics.cap_tainted)` per arm over all 53 cells:

| arm | cells | `cap_tainted` |
|---|---|---|
| `architect-fable-max` | 26 | **0** |
| `architect-opus-max` | 27 | **0** |

**Result: 0 vs 0 → SYMMETRIC. The acceptance criterion is satisfied.**

This is the **trivial case**, and it is worth naming as such rather than claiming a strong
result: the ι4 `UsageGate` retrofit on `run_architect_eval` held for the entire campaign, so
**zero cells were cap-excluded in either arm**. Differential-exclusion bias cannot be present
in a population from which nothing was excluded. The criterion is met because there was
nothing to explain — not because a real asymmetry was measured and found small. The figures
are on the page above so a downstream reader can confirm that themselves rather than take
"trivially satisfied" on trust.

### 2e. `outcome` is `'done'` on all 53 cells — including every decline

Recomputed: `Counter(outcome)` over the 53 cells is `{'done': 53}`. There are no `'failed'`,
`'error'` or `'declined'` cells, **because no such outcome value can be produced on this
path**.

Per `investigation/FINDINGS.md` §"Harness mechanics" item 4, verified in code: each
`report_*` decline tool writes a distinct artifact
(`orchestrator/src/orchestrator/agents/plan_tools.py` → `artifacts.write_*`), but
`run_architect_eval` reads only `artifacts.read_plan()`, and `cleanup_eval_worktree` then
`shutil.rmtree`s the whole meta root. **`EvalMetrics` has no field for the terminal
`report_*` kind, so a deliberate protocol decline is byte-identical in the persisted JSON to
a content failure.**

This is not a footnote — it is the fact that **forces §3 to rest on transcript forensics**.
The result JSONs alone cannot distinguish "correctly declined via the role's own mandated
protocol" from "could not plan," and every one of the 47 no-plan cells is one of those two
things.

---

## 3. Decline-consistency: what the 47 no-plan cells actually are

_RAW OBSERVATION — transcript forensics, transcribed with attribution._ The terminal-cause
classification below cannot be derived from the result JSONs (§2e). It comes from
`investigation/FINDINGS.md` and its artifacts, and is attributed rather than claimed as this
record's own recomputation — **except §3e, which this record verified first-hand** because
the forensic investigation was cut before the cells in question existed.

**A note on the two cuts.** `FINDINGS.md` covers the **51 cells** persisted by ~07:30 BST on
2026-08-26 (fable 24 / incumbent 27). `FINAL-READOUT.txt` covers the final **53** (fable 26 /
incumbent 27). The two fable cells added after the cut are both on `reify_task_4026`. Where
this record quotes a 51-cell figure it says so.

### 3a. Terminal-cause distribution (53 cells, `FINAL-READOUT.txt`)

| terminal cause | fable | incumbent |
|---|---|---|
| planned | 2 | 4 |
| `report_task_already_done` | 15 | 7 |
| `report_blocking_dependency` | 4 | 10 |
| `report_false_premise` | 5 | 6 |
| **TOTAL DECLINES** | **24** | **23** |

**47 of 53 cells — 88.7% — ended in an explicit, server-accepted protocol decline.**

Per `FINDINGS.md` Answer 1, every one of those declines was invoked *after substantive
investigation* (median 41–98 assistant turns) and returned `status: ok` from the plan-tools
server. **Zero cells gave up silently. Zero hit turn or budget caps.**

### 3b. Per-fixture terminal causes

`a` = `report_task_already_done`, `b` = `report_blocking_dependency`,
`f` = `report_false_premise`, `P` = planned.

| fixture | fable | incumbent |
|---|---|---|
| `df_task_2260` | `aaf` | `Paf` |
| `reify_task_12` | `PPb` | `Pbb` |
| `reify_task_2324` | `aaa` | `aaa` |
| `reify_task_2531` | `abb` | `Pbb` |
| `reify_task_2573` | `aab` | `bbb` |
| `reify_task_2699` | `aaa` | `Paa` |
| `reify_task_2778` | `aaa` | `bbb` |
| `reify_task_3883` | `fff` | `fff` |
| `reify_task_4026` | `af` | `aff` |

Two fixtures are unanimous across all six cells of both arms: `reify_task_2324` (6/6
already-done) and `reify_task_3883` (6/6 false-premise).

### 3c. Adversarial verification of the decline grounds

Every decline claim was checked against the actual repositories — `/home/leo/src/reify` and
this repo — by git plumbing, by five adversarial verification agents. Per `FINDINGS.md`
("The claims are overwhelmingly TRUE"), the grounds were found **essentially all TRUE**:

| fixture | verification result |
|---|---|
| `reify_task_2324` | **TRUE** — 16-commit self-tagged "(task 2324, step-N)" series on main, zero of it at base |
| `reify_task_2699` | **TRUE**, precisely scoped — all 11 selector names on main incl. the required PRD amendment; base has 3/14 |
| `reify_task_2573` | **BOTH TRUE** — 2590's substrate absent at base; the task's own merge `69b4969f23` on main; live `done_provenance` matches fable's citation exactly |
| `reify_task_2778` | **BOTH TRUE** — `auto_type_substitution` absent at base, landed next day (`f90423d4fe`). **One minor fable imprecision** (below) |
| `reify_task_2531` | **TRUE**, with a gradient — 2530's FFI genuinely absent at base, but the needed OCCT primitives existed generically |
| `reify_task_12` | **TRUE but incomplete** — task 11's merge genuinely reverted in base ancestry, but the base is on a line **not ancestor of main**, so planning-around was the better answer |
| `df_task_2260` | **ALL TRUE** — task landed via `97059b1dd8`, whose first parent *is* the fixture base (pinned 13 min before landing); both false-premise mechanisms verified at exact `file:line` and **still live on main** |
| `reify_task_3883` | **TRUE, zero refutations** — `Frame3` resolves nowhere at base; today's main still says `"Frame3" is intentionally absent` |
| `reify_task_4026` | false-premise **TRUE** (grammar / `NAMED_DIMENSIONS` claims exact; the real implementation hit the same wall and escalated, esc-4026-121) |

**The one recorded imprecision, named rather than smoothed over.** On `reify_task_2778`, a
fable cell cited a **task/2779 merge-resolution commit** as adjacent supporting evidence.
The decline's substantive ground was verified true regardless; the citation was simply
imprecise. It is recorded here because a record claiming "essentially all TRUE" owes its
reader the exception.

### 3d. Method and provenance

Deterministic `tool_use`-level extraction over all 51 campaign transcripts
(`investigation/extract_transcripts.py` → `summaries/`), decline payloads captured in
`claims.txt`, then **7 verification agents**: 5 adversarially checking the architects' claims
against the live repos by git plumbing only, 1 on harness mechanics, 1 on the γ1 selection
trial. **No eval cells were re-run; nothing was written outside `investigation/`.**

### 3e. The plan-then-decline outcome class — verified first-hand

**The persisted metric and the forensic terminal cause disagree on exactly two cells.**

| source | fable "planned" | incumbent "planned" |
|---|---|---|
| persisted `plan_steps > 0` (53 cells, recomputed) | **3** | **5** |
| forensic terminal cause (`FINAL-READOUT.txt`) | **2** | **4** |

Both gaps are on **`reify_task_4026`**, and both are the same shape: a cell that built and
confirmed a complete plan and *then* declined.

- **Incumbent, `reify_task_4026` trial 3 (`e522b1b0`)** — documented at `FINDINGS.md` L31-34.
  Confirmed a 6-step plan and then called `report_task_already_done` at turn 176.
- **Fable, `reify_task_4026` trial 2 (`f2f205af`)** — **not covered by the forensic
  investigation**, whose 51-cell cut predates this cell. This record therefore verified it
  directly, by extracting the plan-tool call sequence from the cell's own transcript
  (`~/.claude/projects/-home-leo-src-reify-eval-worktrees-reify-task-4026-run-f2f205af/`).
  The sequence is:

  ```
  create_plan → add_plan_step ×6 → add_design_decision ×5 → add_reuse_item ×6
    → confirm_plan            (assistant event 143)
    → report_task_already_done (assistant event 166)
  ```

  **The shapes match exactly.** Both arms produced a plan-then-decline cell, on the same
  fixture, with the same 6 steps, both terminating in `report_task_already_done`.

The fable cell is if anything the sharper illustration, because it **explicitly repudiates
its own plan** in the decline payload:

> "The plan confirmed earlier this session predates this discovery and should be
> disregarded in favor of this provenance report."

Its grounds were substantive and self-verified: task 4026's work is on live main
(`ef2d452971` adding `SPEED_OF_LIGHT`, `9280967055` adding `BOLTZMANN_CONSTANT`, both
confirmed ancestors of main), the worktree base `ab0b4c66db` is ~3 months stale, and the
escalation server independently reported task 4026 `status=done`.

**The consequence, stated plainly: on this band `plan_steps > 0` and "produced a usable
plan" are different predicates.** A cell can satisfy the metric and still have concluded,
in its own final act, that the task must not be planned. **Wherever the two disagree, this
record uses the forensic terminal cause** — so §3's "planned" counts are 2 fable / 4
incumbent, while §4's planRate (a metric readout, reported in the metric's own terms)
uses `plan_steps > 0` and is 3 / 5. §6 reports $/usable-plan under **both** denominators
for this reason.

### 3f. The observation

Both arms recognise moot, blocked and ill-posed tasks at **high and roughly equal rates** —
24 declines of 26 fable cells, 23 of 27 incumbent — and exit through the **mandated
plan-tools protocol** rather than failing silently. Their declared grounds were adversarially
verified true in every checked case, with one minor citation imprecision (§3c). Zero cells
gave up silently; zero hit turn or budget caps.

_That is the observation. What it means for the admission question is §8's to say, not this
section's._
