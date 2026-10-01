# Effect-present caller enumeration — the record leaf ε must cite

**Owner:** task 4647 (leaf δ) · **Consumer:** leaf ε
**Measured:** 2026-08-25; re-measured 2026-08-29 against this branch's tree
(rebased onto main after task 3539 landed — see the callout below).
**Parent PRD:** `docs/prds/landed-not-done-recovery.md`

D1's scope limit reads: *"effect-present is retired at the landing-detection
sites only (dispatch gate, Tier-3.5); its other callers are out of scope and
must be enumerated by leaf δ before removal."* This file is that enumeration.

> **Cite these anchors, not the PRD's — and cite SYMBOLS, not line numbers.**
> The parent PRD's own enumeration (`docs/prds/landed-not-done-recovery.md`
> §"Decomposition plan", the leaf-δ bullet "The enumeration deliverable is
> smaller than D1 implies") is wrong in both halves.
>
> Its COUNT is wrong. It says `commit_effect_present_in_main` "has exactly two
> production callers, both inside `validate_landing_evidence`". That was true
> when this file was first measured (2026-08-25) and stopped being true the
> next day: task 3539 landed on main on 2026-08-26 (`9b18d7efa3` /
> `12aca23a58` / `7a6245b493`) and added a third — signal 4 of
> `orchestrator/src/orchestrator/merge_gates.py::_resolve_already_landed_branch`.
> Re-measured on this branch's own tree (which is based on main *after* 3539),
> there are **three**, so the PRD's "two + one = three production sites" is
> really four. §1 and §4 below carry the corrected table and the ruling.
>
> Its ANCHORS are wrong. It names `harness.py:11733`, `git_ops.py:9155`,
> `git_ops.py:10005` and `merge_queue.py:14241`; not one of them resolves to
> the code it describes. That is the ordinary fate of a line pin, which is why
> every anchor in this file is now `path/to/module.py::symbol` in the house
> form (`CLAUDE.md`, `CONTRIBUTING.md` §2). ε should cite them the same way:
> a symbol anchor survives the edits above it that rot a line number within
> days.

---

## 1. `commit_effect_present_in_main` — exactly THREE production callers

Definition:
`orchestrator/src/orchestrator/git_ops.py::GitOps.commit_effect_present_in_main`.
Since task 3116 (`40c39cd8ee`) its whole body is
`return (await self.describe_commit_effect_in_main(commit_sha)).present` — a
**one-line wrapper** over
`orchestrator/src/orchestrator/git_ops.py::GitOps.describe_commit_effect_in_main`
→ `orchestrator/src/orchestrator/git_ops.py::GitOps._probe_commit_effect`.
ε must retire the *call sites*, not the primitive (see §4).

| # | Anchor (`path::symbol`) | Call expression | Role | ε scope |
|---|---|---|---|---|
| 1 | `orchestrator/src/orchestrator/landing_evidence.py::validate_landing_evidence` | CANDIDATE arm (`if not await git_ops.commit_effect_present_in_main(candidate_sha)`) | The FIX 1' guard applied to a caller-supplied `candidate_sha` | **IN** |
| 2 | `orchestrator/src/orchestrator/landing_evidence.py::validate_landing_evidence` | DISCOVERY arm (`… (effect_check_sha)`) | The FIX 1' guard applied to the branch-tip-or-citation anchor | **IN** |
| 3 | `orchestrator/src/orchestrator/merge_gates.py::_resolve_already_landed_branch` | **signal 4 (SURVIVAL)** — `if not await git_ops.commit_effect_present_in_main(landed_sha)` | The merge-lane already-landed recognizer's only non-historical signal | **OUT — see §4** |

Sites 1 and 2 are inside the one function, which is why ε's edit there is
local. Site 3 is a different subsystem entirely and is ruled OUT in §4; it is
listed here because §1's job is the CENSUS, and a census that silently omits a
caller cannot be told from one that missed it. Every other occurrence in the
tree is prose or a test stub:

- `orchestrator/src/orchestrator/landing_evidence.py` — the module docstring
  (×3), and the docstrings of `::_record_effect_divergence`,
  `::_delivered_checks_differential`, `::branch_work_landed` and
  `::validate_landing_evidence`, plus one inline comment inside
  `::validate_landing_evidence` (8 mentions).
- `orchestrator/src/orchestrator/git_ops.py` — docstrings on
  `::CommitEffectProbe` (×2), `::GitOps.branch_content_in_main` (×2),
  `::GitOps.describe_commit_effect_in_main`, and one comment in
  `::GitOps._anchor_diff_lines` (6 mentions).
- `orchestrator/src/orchestrator/harness.py::Harness._already_landed_dispatch_gate`
  — a comment explaining why the gate short-circuits *before* the git work.
- `orchestrator/src/orchestrator/merge_gates.py::_resolve_already_landed_branch`
  — its own docstring, describing site 3 above.
- `docs/prds/*.md` — prose.

Note the mention inside
`orchestrator/src/orchestrator/landing_evidence.py::branch_work_landed`'s
docstring: it names this predicate as one of the two things the new producer
may never await, pinned at a zero call count by
`orchestrator/tests/test_branch_work_landed.py::TestB2SyncMergeTip`. That pin
is a *constraint on the new producer*, not a call site.

## 2. `branch_content_in_main` — exactly ONE production caller

Definition:
`orchestrator/src/orchestrator/git_ops.py::GitOps.branch_content_in_main`.
Unchanged by 3116 and still byte-identity — it ends in `git diff --quiet`.

| # | Anchor (`path::symbol`) | Call expression | Role | ε scope |
|---|---|---|---|---|
| 4 | `orchestrator/src/orchestrator/harness.py::Harness._already_landed_dispatch_gate`, third arm | `if await self.git_ops.branch_content_in_main(branch):` | **The arm's ENTRY CONDITION**, not an inner guard | **IN** |

The distinction matters to ε and is easy to misread from the PRD's one-line
summary. This is not a check applied *after* the arm decides to look; it is
`if await self.git_ops.branch_content_in_main(branch):`, the predicate that
decides whether the arm runs at all. The DISCOVERY-mode
`validate_landing_evidence` call it gates is the third and last of that
function's three `validate_landing_evidence` calls, in the body of that same
`if`. So a False here does not weaken the verdict — it means **no verdict is produced**,
and the gate falls through as if no landing evidence existed. Replacing this
predicate changes *which tasks are examined*, not merely how they are judged.

Everything else is prose:
`orchestrator/src/orchestrator/landing_evidence.py::branch_work_landed`'s
docstring (the same zero-call-count constraint),
`orchestrator/src/orchestrator/scheduler.py::Scheduler._consult_already_landed`
and
`orchestrator/src/orchestrator/workflow.py::TaskWorkflow._reconcile_done_step_commits`
(docstrings naming the method as an example),
`orchestrator/src/orchestrator/git_ops.py::GitOps.net_diff_is_empty` (×2),
`::GitOps.describe_commit_effect_in_main`,
`::GitOps.commit_effect_present_in_main`, and three mentions in
`orchestrator/src/orchestrator/harness.py::Harness._already_landed_dispatch_gate`'s
own docstring.

## 3. The hardening asymmetry — and it points the wrong way

Found at revalidation, and it strengthens the decay argument exactly where it
matters:

| | `-z` / `core.quotePath=false` | Anchor |
|---|---|---|
| `commit_effect_present_in_main` | **hardened** (task 2500 amendment) | Documented in `git_ops.py::GitOps.commit_effect_present_in_main`'s and `::GitOps.describe_commit_effect_in_main`'s docstrings; applied in `::GitOps._probe_commit_effect` (×2), `::GitOps._anchor_diff_lines`, `::GitOps._batch_added_line_counts` and `::GitOps._compare_touched_paths_to_main` |
| `branch_content_in_main` | **NOT hardened** | `git_ops.py::GitOps.branch_content_in_main`'s **Path-quoting caveat** paragraph documents the hole explicitly |

`branch_content_in_main`'s own docstring spells out the consequence: a changed
path with non-ASCII bytes comes back quoted from `--name-only`, then fails to
match itself as a `--` pathspec on the follow-up `diff --quiet`, and an
empty/mismatched pathspec makes that call report **rc == 0**. That is a FALSE
POSITIVE for "content already landed" — the one direction this primitive is
otherwise fail-safe against.

The asymmetry is the wrong way round for the system as built: the UNHARDENED
predicate is the one at site 4, gating the dispatch gate's third arm, while
the hardened one only ever narrows an already-selected candidate. ε should
treat site 4 as the higher-value replacement of the three in scope, not the
lesser one because it is only a single call.

## 4. Explicitly OUT of scope for ε

| Item | Ruling | Why |
|---|---|---|
| The primitives themselves (`git_ops.py::GitOps.branch_content_in_main`, `::GitOps.commit_effect_present_in_main`, `::GitOps.describe_commit_effect_in_main`, `::GitOps._probe_commit_effect`) | **OUT** | D1 retires *landing-detection call sites*, not the primitives. `docs/prds/landed-not-done-recovery.md` §"Out of scope" ("Retiring `commit_effect_present_in_main` entirely") says so in as many words. Both remain live for diagnostics: `orchestrator/src/orchestrator/landing_evidence.py::_record_effect_divergence` calls `describe_commit_effect_in_main` to explain a rejection, which survives the retirement of the decision it explains. |
| **Site 3 — `orchestrator/src/orchestrator/merge_gates.py::_resolve_already_landed_branch`**, signal 4 | **OUT, owned by task 3539** | Ruled explicitly; see below. |
| **Site 8 — `orchestrator/src/orchestrator/merge_queue.py::SpeculativeMergeWorker._redrive_coalesce_members`** | **OUT** | Ruled explicitly; see below. |
| `orchestrator/src/orchestrator/harness.py::Harness._reconcile_one_stranded` | **OUT** | The stranded-in-progress sweep, not a dispatch gate or a Tier-3.5 arm. It reaches effect-present only transitively, through `validate_landing_evidence`. |
| `escalation/src/escalation/server.py::merge_status` (both git-authority arms) | **OUT** | A read-only status *query*. It reports what the evidence says; it never dispatches, stamps or reverts, so a decaying answer costs a wrong display, not a stranded task. |

### The ruling on site 3, stated rather than omitted

**`orchestrator/src/orchestrator/merge_gates.py::_resolve_already_landed_branch`
signal 4 is OUT of ε's scope**, for three independent reasons, any one of
which suffices — the same three-reason shape as the site-8 ruling below:

1. **It is task 3539's declared site.** 3539 landed it on 2026-08-26
   (`7a6245b493`, the amendment pass that added signal 4) as one of four
   signals in a single fail-closed ladder, and owns it. Two leaves editing the
   same call in the same window is the lock contention this decomposition
   exists to avoid.
2. **It is not the shape D1 describes.** D1's scope limit enumerates "dispatch
   gate, Tier-3.5". This is neither: it is the MERGE LANE's already-landed
   recognizer, reached from
   `orchestrator/src/orchestrator/workflow.py::TaskWorkflow._submit_to_merge_queue`
   through the never-raises guard
   `merge_gates.py::resolve_already_landed_branch`. Its outcome is
   `OutcomeKind.plan_files_already_landed` — an advisory carve-out from the
   `plan_files_not_touched` failure — not a dispatch, a stamp or a revert.
3. **The decay argument does not transfer unaltered.** Signal 4 exists
   precisely BECAUSE signals 1–3 are claims about immutable history and cannot
   see a post-hoc revert; effect-present is deliberately the one non-historical
   leg of that ladder, and it fails CLOSED (`None` = "carve nothing out" =
   today's behaviour). Replacing it there is a different design question from
   retiring a *landing-detection* guard whose failure strands a task, and
   widening D1 by interpretation is how a bounded change becomes an unbounded
   one.

Recorded for the same reason as site 8: §4 says in its own words that a reader
who finds a caller with no ruling on it cannot tell whether it was considered
and excluded or simply missed. This one was considered.

### The ruling on site 8, stated rather than omitted

The parent PRD requires δ to rule on the eighth landing-detection site it never
names.
**`orchestrator/src/orchestrator/merge_queue.py::SpeculativeMergeWorker._redrive_coalesce_members`
is OUT of ε's scope**, for three independent reasons, any one of which
suffices:

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
arithmetic (`docs/prds/landed-not-done-recovery.md` §"The revert measurement
(why effect-present goes, and what replaces it)") predates `40c39cd8ee` and is
measured against a predicate that no longer exists.

**What changed.** Task 3116 replaced byte-identity with threshold added-line
survival —
`orchestrator/src/orchestrator/git_ops.py::_EFFECT_SURVIVAL_AGGREGATE_THRESHOLD`
`= 0.98`, `::_EFFECT_SURVIVAL_PER_FILE_THRESHOLD` `= 0.90`,
`::_EFFECT_SURVIVAL_PER_FILE_MIN_ADDED_LINES` `= 25` (all module-level
constants). (These are the real symbol names; the task text abbreviates the
latter two.) On its own full-corpus
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
