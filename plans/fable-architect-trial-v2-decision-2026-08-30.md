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

---

## 4. Paired per-fixture statistics

_RAW OBSERVATION._ **Forensic / decline-consistency evidence — NOT a capability
comparison.** Read this section only alongside §3, which establishes what a "no-plan" cell
on this band actually is. Per ruling C these figures have no capability standing.

### 4a. Per-fixture planned/valid

Recomputed from `cells/*.json`; "valid" = non-`cap_tainted` cells, which here is all of them
(§2d).

| fixture | fable planned/valid | incumbent planned/valid |
|---|---|---|
| `df_task_2260` | 0/3 | 1/3 |
| `reify_task_12` | 2/3 | 1/3 |
| `reify_task_2324` | 0/3 | 0/3 |
| `reify_task_2531` | 0/3 | 1/3 |
| `reify_task_2573` | 0/3 | 0/3 |
| `reify_task_2699` | 0/3 | 1/3 |
| `reify_task_2778` | 0/3 | 0/3 |
| `reify_task_3883` | 0/3 | 0/3 |
| `reify_task_4026` | 1/**2** | 1/3 |
| `reify_task_4370` | *unrun* | *unrun* |
| `reify_task_4832` | *unrun* | *unrun* |

### 4b. planRate is published at TWO units, and they disagree

⚠️ **This is a live confusion between two committed artifacts, and this record resolves it
by labelling rather than by choosing.** Both upstream artifacts publish a number called
`planRate`, and they are not the same number:

| unit | fable | incumbent | who publishes it |
|---|---|---|---|
| **CELL-level** — planned cells ÷ all cells | 3/26 = **0.1154** | 5/27 = **0.1852** | `FINAL-READOUT.txt` |
| **FIXTURE-level** — mean of per-fixture rates over the 9 scored fixtures | **0.1296** | **0.1852** | `analyse_tranche1.py` (its Δ and CI) |

Both recomputed here, not copied. Fixture-level fable = (0/3 + 2/3 + 0/3 + 0/3 + 0/3 + 0/3 +
0/3 + 0/3 + 1/2) ÷ 9 = (2/3 + 1/2) ÷ 9 = **0.1296**. Fixture-level incumbent = (5 × 1/3) ÷ 9
= **0.1852**.

**Why they diverge in the fable arm and not the incumbent's:** `reify_task_4026` ran **2**
fable trials, not 3 (§2a). At the cell level that cell contributes 1 of 26; at the fixture
level it contributes a rate of 1/2 = 0.5, weighted equally with every other fixture. The
incumbent arm is balanced 3-per-fixture throughout, so its two units coincide at 0.1852.

`analyse_tranche1.py` uses the fixture-level unit **deliberately**: three trials on one
fixture are correlated, and treating 53 cells as independent overstates confidence by
roughly √3. A downstream reader comparing 0.1154 against 0.1296 and concluding one artifact
is wrong would be mistaken — **they are different statistics, both correct in their own
unit**.

### 4c. The paired diff and its bootstrap CI

Computed by re-running the ruled readout rather than reimplementing it:

```
python3 /home/leo/src/dark-factory/data/eval-campaign/tranche1/analyse_tranche1.py \
        --results /home/leo/src/dark-factory/data/eval-campaign/tranche1/cells \
        --marker <any file older than every cell>
```

Method of record: `analyse_tranche1.py::boot` — a **paired cluster bootstrap over
FIXTURES**, both arms resampled together on the same fixture draw, **10,000 draws, seed
20260825**.

| quantity | value | 95% CI |
|---|---|---|
| planRate fable (fixture-level) | 0.1296 | — |
| planRate incumbent (fixture-level) | 0.1852 | — |
| **Δ = fable − incumbent** | **−0.0556** | **[−0.2037, +0.0926]** |
| UNLOCKED fixtures | 0 of 9 | [0, 0] |

The CI lower bound does **not** exclude zero.

### 4d. The caveats the instrument itself emits

These are carried, not re-derived; each is `analyse_tranche1.py`'s own inline warning.

1. **The unit of replication is the FIXTURE, not the cell.** Three correlated trials per
   fixture overstate confidence by ~√3 if treated as independent. This is why §4b's two
   units exist at all, and why the CI above is computed over 9 fixtures rather than 53 cells.
2. **Degenerate-data / zero-width-CI guard.** The bootstrap returns `None` rather than a
   spurious interval when no fixture has both arms scored. It fires visibly in §4e, where
   the well-posed subset yields a genuinely zero-width `[+0.0000, +0.0000]` — that is a
   degenerate interval, not a precise one.
3. **The WELL-POSED-SUBSET diagnostic is the load-bearing one.** The band was selected on a
   **single γ1 trial** in which the incumbent planned 0 of 11. On re-run at 3 trials the
   incumbent planned on **5 of 9** scored fixtures. So the band's defining premise — *"the
   incumbent cannot plan this"* — **fails on 5 of the 9 fixtures**:
   `df_task_2260`, `reify_task_12`, `reify_task_2531`, `reify_task_2699`, `reify_task_4026`
   (each incumbent 0.33). Only **4 fixtures** still satisfy it.

   The instrument states the consequence directly: the ruled ±0.17/±0.22 resolution floor
   assumed **11** well-posed fixtures. Only **4** qualify. **The effective resolution is
   therefore materially WORSE than the floor stated before the run.**

### 4e. The well-posed subset

| quantity | value |
|---|---|
| premise holds on | 4 of 9 fixtures |
| fable planRate on the well-posed subset | 0.0000 |
| incumbent planRate there | 0.0000 (by construction) |
| **Δ restricted to well-posed fixtures** | **+0.0000**, 95% CI [+0.0000, +0.0000], n=4 |
| unlocked fixtures | **0 of 4** |

Every "unlocked" fixture — a fixture fable plans and the incumbent cannot — must by
definition come from this subset. **There are none.** The zero-width interval is the
degenerate-data guard firing, not a precise measurement.

### 4f. Reading the sign, and the standing of this whole section

**Δ is NEGATIVE: −0.0556. The incumbent planned MORE on this band than the candidate, not
less — the opposite of the trial's hypothesis in direction.**

**This is NOT evidence against fable's planning capability.** On a band where §3 shows 47 of
53 cells are verified-correct declines, **planning is frequently the defect and declining is
the correct behaviour**. A higher planRate on a verified-moot fixture is a worse result, not
a better one: `FINDINGS.md` Answer 3.1 records that the incumbent's extra plans on 2699,
2531 and 4026 are duplicate or moot work — one 17-step incumbent "success" (`d1a73352`)
re-implements already-landed registration and absorbs work the real task explicitly deferred.

The arms agree at the *should-this-be-planned* level on essentially every fixture, and two
fixtures are unanimous 6/6 across both arms (§3b). **The fixtures drove the outcome, not the
models.** Δ here is measuring the band, not the candidates.

> **NO DECISION HANGS ON THIS Δ OR ITS CI.** The CI/bootstrap readout rule is **inert** —
> see §9. No further cells are being scored, so there is no stopping criterion left for
> this interval to feed. These statistics are published because ζ's contract mandates
> paired per-fixture statistics as forensic evidence, and for no other reason. A reader who
> arrives at this table without having read §9 must not mistake it for a live decision rule.

---

## 5. `plan_quality`, validity-bounded

_RAW OBSERVATION._ **Forensic evidence, not a capability readout.** This section reports
`plan_quality` **only where a valid reference block exists**. Plausibility-judged cells are
enumerated individually and are **never averaged in silently**.

### 5a. Every cell with `plan_steps > 0`, enumerated

All 8 such cells across 53, recomputed from `cells/*.json`. The `reference valid?` column is
driven by σ's structured marker `metrics.judged_without_reference` (PRD D9): a cell with
`judged_without_reference == True` was scored against **no reference diff** — the judge
rated the plan's *plausibility*, not its *fidelity* to a known-good answer.

| fixture | arm | trial | `plan_steps` | `plan_quality` | reference valid? |
|---|---|---|---|---|---|
| `reify_task_12` | fable | 1 | 22 | 0.85 | ✅ yes (fidelity) |
| `reify_task_12` | fable | 2 | 24 | 0.94 | ✅ yes (fidelity) |
| `reify_task_4026` | fable | 2 | 6 | 0.93 | ❌ **no — plausibility** |
| `df_task_2260` | incumbent | 1 | 14 | 0.95 | ✅ yes (fidelity) |
| `reify_task_12` | incumbent | 3 | 26 | 0.90 | ✅ yes (fidelity) |
| `reify_task_2531` | incumbent | 1 | 10 | 0.70 | ❌ **no — plausibility** |
| `reify_task_2699` | incumbent | 2 | 17 | 0.95 | ❌ **no — plausibility** |
| `reify_task_4026` | incumbent | 3 | 6 | 0.91 | ❌ **no — plausibility** |

Note that **two of these eight cells are the plan-then-decline cells of §3e** (`reify_task_4026`,
both arms). They carry a `plan_quality` score for a plan their own author repudiated.

### 5b. The validity-bounded means

Bounding strictly to `judged_without_reference == False`:

| arm | fidelity-scored cells | `plan_quality` values | **bounded mean** | n |
|---|---|---|---|---|
| `architect-fable-max` | `reify_task_12` t1, t2 | 0.85, 0.94 | **0.895** | **2** |
| `architect-opus-max` | `df_task_2260` t1, `reify_task_12` t3 | 0.95, 0.90 | **0.925** | **2** |

Plausibility-scored and therefore excluded from both means: fable 0.93; incumbent 0.95,
0.91, 0.70.

### 5c. `FINAL-READOUT`'s `mean_pq` is SUPERSEDED by §5b

`FINAL-READOUT.txt` and `analyse_tranche1.py` both publish a per-candidate `mean_pq`:

| arm | published `mean_pq` | over | this record's bounded figure |
|---|---|---|---|
| `architect-fable-max` | **0.907** | 3 planned cells | **0.895** (n=2) |
| `architect-opus-max` | **0.882** | 5 planned cells | **0.925** (n=2) |

Those published means average **all** planned cells — including 1 of 3 fable and 3 of 5
incumbent that were plausibility-judged. That is precisely the operation ζ's contract
forbids, so **the published `mean_pq` is superseded by §5b for all purposes.**

Both figures are quoted here deliberately. Two committed artifacts now carry different
quality numbers for the same campaign; simply omitting the older one would leave a reader
to guess which is authoritative. It is **this record's §5b**.

> ⚠️ A further note for anyone reading `analyse_tranche1.py`'s output directly: its
> per-candidate table prints a `NOTE` claiming "plan_quality here is validity-bounded."
> **It is not.** The `mean_pq` that table prints is 0.907 / 0.882 — the all-planned-cells
> mean, arithmetically verifiable as including the plausibility-scored cells
> (fable: mean(0.85, 0.94, 0.93) = 0.9067). The note is best read as a *warning that
> plausibility-scored cells are present*, not as a description of the computation. The
> genuinely bounded figures are §5b's and appear nowhere in the instrument's output.

### 5d. Corroboration: γ1 measured the inflation

γ1's calibration report (`plans/fable-trial-v2-calibration-2026-08-24.md`, §"plan_quality
validity — the binding constraint") measured plausibility scoring across 15 cells and found
it **inflates `plan_quality` by +0.0810 on average** and **compresses the distribution into
[0.92, 0.95]**, while fidelity-scored cells spread across [0.70, 0.93].

The four plausibility-scored cells here are consistent with that pattern: three sit in or at
the edge of the compressed band (0.93, 0.95, 0.91). The fourth, the incumbent's 0.70 on
`reify_task_2531`, sits well below it — so the pattern is a tendency, not a rule, and this
record does not claim otherwise. **The operative point is that a plausibility score and a
fidelity score are not the same measurement and must not be averaged together.**

### 5e. The binding limitation

> **n = 2 valid-reference planned cells per arm, out of 26 and 27 cells respectively.**
>
> **`reify_task_12` is the ONLY fixture carrying a valid-reference planned cell in BOTH
> arms. There is therefore no paired quality comparison available at all** — the v1
> paired-per-fixture discipline cannot be applied to `plan_quality` on this tranche, because
> the pairing does not exist.
>
> **The bounded means in §5b carry NO inferential weight.** They are reported because ζ's
> contract requires the validity-bounded figure to be on the record, and for no other
> reason. A 2-cell mean supports no conclusion about either arm, and the 0.895-vs-0.925
> gap between them is not a finding.

Publishing a two-cell mean without this caveat would be worse than publishing nothing, which
is why the caveat travels attached to the figure rather than in a footnote.
