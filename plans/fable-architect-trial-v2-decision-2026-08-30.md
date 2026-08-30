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
