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
