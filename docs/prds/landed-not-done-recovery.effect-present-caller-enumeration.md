# Effect-present caller enumeration — the record leaf ε must cite

**Owner:** task 4647 (leaf δ) · **Consumer:** leaf ε
**Measured:** 2026-08-25, against this branch's tree.
**Parent PRD:** `docs/prds/landed-not-done-recovery.md`

D1's scope limit reads: *"effect-present is retired at the landing-detection
sites only (dispatch gate, Tier-3.5); its other callers are out of scope and
must be enumerated by leaf δ before removal."* This file is that enumeration.

> **Cite these anchors, not the PRD's.** The parent PRD's own enumeration
> (`landed-not-done-recovery.md:632-637`) states the right COUNTS — two + one
> = three production sites — but every line number in it is stale, and so is
> the one it gives for the eighth site. It says `harness.py:11733`,
> `git_ops.py:9155` and `git_ops.py:10005`; the real anchors are
> `harness.py:11772`, `git_ops.py:9207` and `git_ops.py:10201`. Its
> `merge_queue.py:14241` is really `merge_queue.py:14805`. The counts were
> re-measured independently here and confirmed; only the anchors moved.

---

## 1. `commit_effect_present_in_main` — exactly TWO production callers

Definition: `git_ops.py:10201`. Since task 3116 (`40c39cd8ee`) it is a
**one-line wrapper** — `git_ops.py:10366` returns
`(await self.describe_commit_effect_in_main(commit_sha)).present`, over
`describe_commit_effect_in_main` (`:9422`) → `_probe_commit_effect` (`:9570`).
ε must retire the *call sites*, not the primitive (see §4).

| # | Call site | Enclosing symbol | Role | ε scope |
|---|---|---|---|---|
| 1 | `landing_evidence.py:1635` | `validate_landing_evidence` — CANDIDATE arm | The FIX 1' guard applied to a caller-supplied `candidate_sha` | **IN** |
| 2 | `landing_evidence.py:1695` | `validate_landing_evidence` — DISCOVERY arm | The FIX 1' guard applied to the branch-tip-or-citation anchor | **IN** |

Both are inside the one function, which is why ε's edit is local. Every other
occurrence in the tree is prose or a test stub:

- `landing_evidence.py:14, :36, :113, :624, :748, :1261, :1538, :1685` — module
  and function docstrings, and one inline comment.
- `git_ops.py:1216, :1233, :9230, :9238, :9428, :9735` — docstrings.
- `harness.py:11527` — a comment explaining why the dispatch gate
  short-circuits *before* the git work.
- `docs/prds/*.md` — prose.

Note `landing_evidence.py:1261`: `branch_work_landed`'s docstring names this
predicate as one of the two things it may never await, pinned at a zero call
count by `test_branch_work_landed.py::TestB2SyncMergeTip`. That pin is a
*constraint on the new producer*, not a call site.

## 2. `branch_content_in_main` — exactly ONE production caller

Definition: `git_ops.py:9207`. Unchanged by 3116 and still byte-identity —
it ends in `git diff --quiet`.

| # | Call site | Enclosing symbol | Role | ε scope |
|---|---|---|---|---|
| 3 | `harness.py:11772` | `Harness._already_landed_dispatch_gate` (def `:11337`), third arm | **The arm's ENTRY CONDITION**, not an inner guard | **IN** |

The distinction matters to ε and is easy to misread from the PRD's one-line
summary. This is not a check applied *after* the arm decides to look; it is
`if await self.git_ops.branch_content_in_main(branch):`, the predicate that
decides whether the arm runs at all. The DISCOVERY-mode
`validate_landing_evidence` call it gates sits at `harness.py:11780`. So a
False here does not weaken the verdict — it means **no verdict is produced**,
and the gate falls through as if no landing evidence existed. Replacing this
predicate changes *which tasks are examined*, not merely how they are judged.

Everything else is prose: `landing_evidence.py:1260` (the same zero-call-count
constraint), `scheduler.py:5708` and `workflow.py:8650` (docstrings naming the
method as an example), `git_ops.py:9311, :9316, :9499, :10258`,
`harness.py:11351, :11435, :11448`.

## 3. The hardening asymmetry — and it points the wrong way

Found at revalidation, and it strengthens the decay argument exactly where it
matters:

| | `-z` / `core.quotePath=false` | Anchor |
|---|---|---|
| `commit_effect_present_in_main` | **hardened** (task 2500 amendment) | `git_ops.py:9463-9496` documents it; applied at `:9620-9621`, `:9659-9660`, `:9826-9827` |
| `branch_content_in_main` | **NOT hardened** | `git_ops.py:9230-9241` documents the hole explicitly |

`branch_content_in_main`'s own docstring spells out the consequence: a changed
path with non-ASCII bytes comes back quoted from `--name-only`, then fails to
match itself as a `--` pathspec on the follow-up `diff --quiet`, and an
empty/mismatched pathspec makes that call report **rc == 0**. That is a FALSE
POSITIVE for "content already landed" — the one direction this primitive is
otherwise fail-safe against.

The asymmetry is the wrong way round for the system as built: the UNHARDENED
predicate is the one at site 3, gating the dispatch gate's third arm, while
the hardened one only ever narrows an already-selected candidate. ε should
treat site 3 as the higher-value replacement of the three, not the lesser one
because it is only a single call.

## 4. Explicitly OUT of scope for ε

| Item | Ruling | Why |
|---|---|---|
| The primitives themselves (`git_ops.py:9207`, `:10201`, `:9422`, `:9570`) | **OUT** | D1 retires *landing-detection call sites*, not the primitives. `docs/prds/landed-not-done-recovery.md:775-776` says so in as many words. Both remain live for diagnostics: `_record_effect_divergence` (`landing_evidence.py:624`) calls `describe_commit_effect_in_main` to explain a rejection, which survives the retirement of the decision it explains. |
| **Site 8 — `merge_queue.py:14805`**, inside `SpeculativeMergeWorker._redrive_coalesce_members` (def `:14710`) | **OUT** | Ruled explicitly; see below. |
| `harness.py:5715`, in `Harness._reconcile_one_stranded` (def `:5479`) | **OUT** | The stranded-in-progress sweep, not a dispatch gate or a Tier-3.5 arm. It reaches effect-present only transitively, through `validate_landing_evidence`. |
| `escalation/server.py:3870`, `:3916`, in `merge_status` (def `:3651`) | **OUT** | A read-only status *query*. It reports what the evidence says; it never dispatches, stamps or reverts, so a decaying answer costs a wrong display, not a stranded task. |

### The ruling on site 8, stated rather than omitted

The parent PRD requires δ to rule on the eighth landing-detection site it never
names. **`merge_queue.py:14805` is OUT of ε's scope**, for three independent
reasons, any one of which suffices:

1. **It is task 4497's declared site.** Two leaves editing the same call in the
   same window is the lock contention this decomposition exists to avoid.
2. **It is not the shape D1 describes.** D1's scope limit enumerates "dispatch
   gate, Tier-3.5". A coalesce re-drive is neither: it re-drives *members of a
   coalesce train* whose landing was already established by the merge lane. Its
   recovery action is `redrive_member`, not a dispatch.
3. **D1's wording does not reach it**, and widening a scope limit by
   interpretation is how a bounded change becomes an unbounded one.

Recorded as a decision, not an omission: a reader who finds an eighth
`validate_landing_evidence` call site and no ruling on it cannot tell whether
it was considered and excluded or simply missed.

## 5. D1's precision arithmetic, RE-DERIVED

**ε must cite these figures, not the PRD's.** The parent's 95.5% / 0.04%
arithmetic (`landed-not-done-recovery.md:314-330`) predates `40c39cd8ee` and
is measured against a predicate that no longer exists.

**What changed.** Task 3116 replaced byte-identity with threshold added-line
survival — `_EFFECT_SURVIVAL_AGGREGATE_THRESHOLD = 0.98` (`git_ops.py:1165`),
`_EFFECT_SURVIVAL_PER_FILE_THRESHOLD = 0.90` (`:1173`),
`_EFFECT_SURVIVAL_PER_FILE_MIN_ADDED_LINES = 25` (`:1180`). (These are the real
symbol names; the task text abbreviates the latter two.) On its own full-corpus
measurement of 2,827 merges (2,822 usable), it accepts **1,050 of 2,680**
previously-rejected merges — **39.2%** — leaving a **60.8% residual**.

**The original.** 5,469 `effect_absent` verdicts across both repos
(DF 2,829 + reify 2,640), against **2 genuine post-hoc reverts in 5.4 months**
⇒ 2 / 5,469 = **0.037%**, which the PRD rounds to 0.04%.

**Re-derived, two ways.** The two published denominators are not the same
corpus, so both are given rather than picking the flattering one:

| Basis | Residual denominator | Precision |
|---|---|---|
| 3116's own corpus (2,680 previously-rejected merges × 60.8%) | ≈ **1,630** | 2 / 1,630 = **0.12%** |
| Both repos, like-for-like (5,469 × 60.8%) | ≈ **3,325** | 2 / 3,325 = **0.060%** |

The like-for-like row is the honest comparison: it holds the numerator's
population fixed, and shows 3116 improved precision by exactly the reciprocal
of the residual, **1.64×** — from 0.037% to 0.060%. The 0.12% row pairs a
single-corpus denominator with a both-repos numerator and is therefore an
UPPER bound, not a better estimate. The PRD's other headline moves the same
way: the 95.5% share of landings yielding `effect_absent` at HEAD becomes
95.5% × 0.608 ≈ **58.1%**.

**The conclusion, stated plainly.** *D1's DIRECTION survives on the residual;
its original arithmetic does not.* Even taking the most favourable reading,
effect-present's precision is **≈0.06–0.12%** — still two to three orders of
magnitude short of justifying a guard whose false-negative mode is the very
defect this PRD exists to fix. Two further facts are untouched by 3116 and
should be cited alongside:

- Of the **7 firings actually recorded** in the escalation/emission corpus,
  **7 were false positives and 0 genuine**. A threshold change cannot improve
  0 true positives.
- **14 of 16 lifetime true positives came from a code path deleted on day 9**
  (`af1e7de63a`). Its replacement leaves no merge marker, so neither ancestry
  nor patch-id can false-positive on it. That class is extinct.

What 3116 *did* change is the honesty of the framing: the residual is smaller
and the predicate is no longer the byte-identity strawman §Background argues
against. ε should say so, cite 0.06–0.12%, and note that the guard's
**monotonic** false-negative mode — not its precision — is the load-bearing
argument. A merely-imprecise guard is an annoyance; a monotonically-decaying
one converts a transient miss into a permanent loss, which is what stranded
tasks 3103 and 3916.

## 6. What δ shipped beside these sites (and what it did NOT)

`branch_work_landed` (`landing_evidence.py`) is the non-decaying producer, and
δ **added it beside** the sites above rather than removing anything:

- No effect-present call site is removed by 4647. Sites 1–3 are untouched.
- `validate_landing_evidence` is re-expressed as a MODE over the shared
  producer family — one verdict type, one reason vocabulary, one exit — with
  its public surface and every emitted reason code unchanged.
- `LandingMethod` is the discriminator ε needs: `patch_id` marks the
  non-decaying contract, `merge_marker` and `citation` mark the legacy policy
  ε is retiring. A consumer can read which policy decided a verdict without
  knowing which function produced it.
- `no_attribution` is **registered** in `LandingReason` but not yet emitted.
  Registration is the load-bearing half: it guarantees that when ε flips the
  emitted spelling, no escalation renders `'Unrecognized reason code'` into an
  L1 body. Flipping the emitted value is ε's edit, made together with
  repointing the consumers.
