# PRD — landed-but-not-done recovery: non-decaying landing evidence + reachable recovery

**Status:** active · authored 2026-08-23 · dark_factory · approach **B + H**
**Verified against:** main @ `e0c859f566` (investigation) / `2dc87b3fb8` (revert measurement). Every
file:line anchor below was re-read on main during authoring; re-verify before editing (anchors drift).
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
   one production call site (`harness.py:5540`), and even there two **sweep-side upgrades**
   (`harness.py:5563-5571`, `:5591-5599`) override its verdict for `blocked` tasks, deliberately
   outside the table ("rather than a change to θ1's reviewed table — design decision, task 2243;
   esc-2243-4/5"). The project has **twice** chosen sweep-side upgrade over table edit.

### Why the eight detectors all missed

| Population | Detectors reaching it | Why it failed |
|---|---|---|
| landed `pending` | **one** — `_already_landed_dispatch_gate` | its evidence **decays** (below) |
| landed `blocked` | the sweep-side upgrade | vetoed by a **one-entry** category allowlist |
| landed `merge-deferred` | **none** | RC-3 *deletes* the landing evidence |

- **Decay.** `branch_content_in_main` (`git_ops.py:8894-8913`) and `commit_effect_present_in_main`
  (`git_ops.py:8965`) both require main HEAD to be **byte-identical** to the branch/commit content
  across the touched path set. Any *unrelated* later commit touching those paths flips them False.
  **The longer a task is stranded, the less recoverable it becomes.** Demonstrated: task 3916 went
  detected-and-escalated (2026-08-19) → undetected (2026-08-22), same landing, evidence rotted beneath it.
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
- *Adding rows to `_RECOVERY`* — inert for `pending`/`merge-deferred` (filtered upstream at
  `harness.py:4947`), redundant for `blocked` (the caller already upgrades), and would create two
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
this question at `git_ops.py:8544-8555` (rebase already-landed detection) and `git patch-id --stable`
at `:9229-9245`. G3 therefore resolves against existing, production-proven code, not novel substrate.

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

**D6 — Option 7 emits, never escalates, and never charges the streak alarm.**
`RecoverySite.already_landed_gate` is **deliberately excluded** from `STREAK_CHARGING_SITES`
(`recovery_emission.py:152-168`) because it runs per dispatch *tick* — charging it would file a
blocking L1 within seconds. The existing `veto_streak_min_span_secs` backstop is the sanctioned path
for a tick-rate site. Leaf α must not add the site to `STREAK_CHARGING_SITES`.

**D7 — Leaf η must check live-claimancy before writing.** Task 3610 was `in-progress` with a heartbeat
54 s old while its work sat on main; a reconciler that wrote `done` under a live claimant would race a
running workflow. η resolves live-claimancy first and skips (emitting a disposition) rather than
racing — the same discipline `_RECOVERY` encodes by folding `live_claimant` into its key.

**D8 — Extraction before writer.** `_found_on_main_response` (`escalation/server.py:3195`) and
`_git_authority_task_metadata` (`:3240`) are **closures inside the `merge_status` tool** (`:3287`),
not callable from outside. Option 6 is therefore two leaves: extract to a shared module, then write.
The extraction must be behaviour-preserving for `merge_status`, pinned by its existing suite
(`escalation/tests/test_merge_status_git_authority.py`).

## Pre-conditions for activating

None blocking. All five changes act on code live on main today. Sequencing constraints are internal to
the decomposition (§Decomposition plan) — δ before ε and η, ζ before η, γ before η.

## Cross-PRD relationship

| Other PRD / task | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `plans/found-on-main-provenance-integrity-prd.md` | consumes | the six already-landed re-derivation sites; `done_provenance` attribution; the reopen-freshness gate (`_check_reopen_freshness`, `task_interceptor.py:5670`) | **that PRD** | active — this PRD changes the *evidence* those sites use, never the attribution/reopen rules |
| `docs/prds/claimant-invariant-enforcement.md` | consumes | `_resolve_live_claimant` / `is_stranded_any_status`; `_RECONCILE_SWEEP_STATUSES` | **that PRD** | active (Leo, 2026-08-21). This PRD does **not** touch either; D7's live-claimancy check must use whatever that PRD lands |
| `plans/orchestrator-atomic-train-merge-prd.md` | consumes | `merge-deferred` lifecycle, `metadata.train`, §9.8 guards | **that PRD** | leaf η adds a *recovery* edge for train-less members; the parked-member contract is unchanged |
| task **3541** (`classify_pins` veto collapse) | produces | the escalation-veto predicate at every recovery site | **3541** | pending, blocked on 3533/3540/3550/3563/3976. Leaf β is a deliberate one-category stopgap; 3541 supersedes it |
| task **4614** (merge-phase gate discards `already_done.json`) | sibling | the `already_done` escape hatch | **4614** | pending — a different landed-not-done channel; not folded in |

No reciprocal-ownership ambiguity: every seam above is owned by the *other* PRD, and this one only
changes evidence quality behind them.

## Contract (B + H)

The seam is **landing evidence**: what the system accepts as proof that task *N*'s work is on main.
Today three call sites re-derive it inconsistently. This PRD makes one contract and points them all at it.

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
  Modules: `orchestrator` (`harness.py`). Prereqs: none.
  Signal: after a dispatch tick that declines task *N*, an operator reading the recovery-disposition
  record sees `site=already_landed_gate` with the decline reason; previously nothing was recorded (B13).
  G7: fixes an `structured-facts-at-failure` instance; `storm-escape-required` addressed by D6.

- **β — Stop a stale `preexisting_main_break` vetoing an on-main self-heal** *(option 4)*.
  Modules: `orchestrator` (`harness.py`). Prereqs: none.
  Signal: a `blocked` task whose branch is on main and whose only open escalation is
  `preexisting_main_break` reaches `done` on the next sweep (B7); one with a human-concern escalation
  still does not (B8).

- **γ — Preserve the LandedOutbox row for parked tasks** *(option 5)*.
  Modules: `orchestrator` (`merge_queue.py`). Prereqs: none.
  Signal: for a landed task in `merge-deferred`, the row is still present in
  `data/orchestrator/landed_outbox.json` after a reconcile pass, and the outcome label distinguishes
  "pruned" from "skipped-parked"; a `done` task's row is still consumed (B9).
  G7: `holds-owned-and-bounded` — growth stays bounded because terminal status still consumes.

- **δ — `branch_work_landed`: patch-id attribution + no-op rejection** *(option 1, producer)*.
  Modules: `orchestrator` (`landing_evidence.py`, `git_ops.py`). Prereqs: none.
  Unlocks: ε, η. Intermediate — its consumer is ε.
  Signal (intermediate): B1–B6 green against real repo fixtures.
  Also enumerates every remaining `commit_effect_present_in_main` / `branch_content_in_main` caller
  and records which are in scope for ε (D1's scope limit).

- **ε — Point the dispatch gate and Tier-3.5 at the new contract** *(option 1, consumers)*.
  Modules: `orchestrator` (`harness.py`), `escalation` (`server.py`). Prereqs: **δ**.
  Signal: a task stranded for weeks whose paths have since been edited by other work is recovered on
  the next dispatch tick — the exact 3916/3103 shape that is undetectable today (B1, B2).
  Also corrects the task-1175 docstrings at `git_ops.py:8981` and `harness.py:5695`.

- **ζ — Extract the git-authority landing tier out of the escalation server** *(option 6, prerequisite)*.
  Modules: `escalation` (`server.py`) → shared module. Prereqs: none. Unlocks: η. Intermediate.
  Signal (intermediate): `merge_status`'s existing suite
  (`escalation/tests/test_merge_status_git_authority.py`) stays green — behaviour-preserving by
  construction — and the tier is importable outside the tool closure.

- **η — Periodic status-agnostic landed reconciler with a writer** *(option 6)*.
  Modules: `orchestrator` (new reconciler + harness wiring). Prereqs: **δ, ζ, γ**.
  Signal: a landed task in `pending` **or** `merge-deferred` (without `metadata.train`) reaches `done`
  with attributed provenance without any dispatch and without a human (B11); a real train member and a
  live-claimed task are both skipped (B10, B12).
  G7: `status-matches-liveness` + D7 live-claimancy check; `storm-escape-required` — the pass must
  bound its per-tick work and emit a disposition rather than re-deriving silently;
  `loop-thread-occupancy-bounded` — git work must not block the event loop.

- **θ — Integration gate: the landed-recovery boundary suite** *(B+H integration task)*.
  Modules: `orchestrator`, `escalation`. Prereqs: **α, β, γ, ε, η**.
  Signal (leaf): the full B1–B13 matrix green as one suite, exercising each of `pending`, `blocked`,
  and `merge-deferred` end-to-end against a real git fixture — a landed task in each status self-heals,
  a no-op landing does not, and a live-claimed task is never raced.

Dependency shape: `δ → ε`; `δ, ζ, γ → η`; `α, β, γ, ε, η → θ`. α, β, γ, δ, ζ are all independently
startable, so the batch parallelises well under the narrow-file-lock model — and the three modules
each leaf touches are disjoint enough to avoid lock contention except at θ.

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
- **`deferred` status.** No automatic recovery path reaches it and none is added here; η could cover it
  but the status is human-set and hands-off by convention.
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
3. **Disposition reason vocabulary reuse.** α's decline reasons may or may not want to reuse
   `LeaveReason` (`recovery_emission.py`) versus the contract's own vocabulary. **Suggested:** keep the
   contract's `reason` distinct and map it at the emission boundary. Decide in α.
