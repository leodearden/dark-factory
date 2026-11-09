# Children of a superseded parent: what the reconciler should do

2026-11-01. Asked by Leo (L2 item 1) after ruling B on esc-5869-8 drew 31 false
"orphaned: parent 5869 cancelled" L1s (cluster esc-6905-2).

## 1. The mechanism today

`fused-memory/src/fused_memory/reconciliation/targeted.py::_sweep_cancelled_descendants`
runs on every `cancelled` transition (added 2026-05-11, `64563a3bf3`). For every
non-terminal task whose `metadata.spawned_from == parent` or whose dependencies include
the parent, it chooses one of three routes:

| route | condition | action |
|---|---|---|
| cancel | spawned_from + `escalation_id` set + none of the declared `metadata.files` exist | auto-cancel, `reopen_reason=parent_cancelled:<A>` |
| L1 | otherwise, orchestrator live | blocking `scope_violation` L1 "Task N orphaned: parent A cancelled (review-followup / dependent)", `expand_scope\|abort_task` |
| block | otherwise, orchestrator dead | `blocked` + `metadata.parent_cancelled`, `needs_recheck_against_main` |

What it ignores: why the parent was cancelled (the `x_superseded_by` / `x_subsumed_by` /
`x_coalesced_into` / `x_replacement` pointers), whether the child is itself named as a
superseder, and whether the canceller has already re-pointed or cancelled the child. It runs
once, within seconds of the cancel, and its only race guard is a single immediate live
re-read.

The L1 does not hold dispatch. I found no scheduler gate on an open escalation, and three
children were already running when Leo ruled on them: 771, 582 and 733. In the live-orchestrator
case the L1 is therefore advisory only, and does not stop a dependent that the cancel released
from dispatching. A cancelled dependency counts as satisfied
(`orchestrator/dependency_verdict.py::deps_verdict`).

## 2. Population measured

Every `descendant_escalated` row in `data/reconciliation/reconciliation.db` (journalled since
2026-10-03, `3f7be3548c`; the first row is 2026-10-07): **126 records**, joined to each
project's escalation archive and its resolution, and to the live task stores.
dark_factory 44, reify 20, solar_challenge + solar_challenge_platform 62. Relation:
71 review-followup, 55 dependent. No auto-cancel or auto-block row appears in the window,
so the L1 route covers the whole window. Raw data: `/tmp/sp/escalated.json`; the per-case
classification is in `/tmp/sp/tally.py`.

Each case was classified by the handling that turned out to be correct (from the
resolution text and the child's live state), and by **redundant**: the act that superseded
the parent (a ruling, a /prd or review-all decompose, a split session, a lead ruling) had
already dispositioned the child before anyone read the flag.

### Correct handling, by category

| code | handling | all 126 | excl. 5869 (95) | examples |
|---|---|---|---|---|
| F | fold into the superseder's scope, then cancel the child | 49 (39%) | 49 (52%) | 937/960 → S26 leaves; 1010/1011/930/964 (docket 143); 8199/8200 → 8812; 5160 → 5261 |
| S | the child **is** the superseder: leave untouched | 29 (23%) | 2 (2%) | 8032–8058 (5869 split); 928/930 (superseders of 582) |
| P | re-point the dependency or sequencing onto the superseder; keep the child | 23 (18%) | 21 (22%) | 5770 → 6580; 952 → 1112; 7738 → 8891; 4512/5119 → 9004/9007; 6905/7996 → 8032 |
| U | valid as is, independent of the parent's artifacts: leave untouched | 14 (11%) | 12 (13%) | 6906, 7007, 5532, 8207, 941, 733, 852; 7882 and 601 (a cancelled dep counts as satisfied) |
| M | cancel as moot: the premise existed only on the parent's branch or plan, and nothing replaces it | 6 (5%) | 6 (6%) | 6367, 771, 152, 504, 8586, 4721 |
| R | re-scope against main, trim, or defer | 4 (3%) | 4 (4%) | 5531 (deferred), 792, 261, 230 |
| D | already delivered: done `found_on_main` | 1 | 1 | 3378 |

By relation: **dependents** were F 53%, P 24%, U 11%, M 7%. **Review-followups excluding
5869** were F 45%, P 23%, U 18%, R 7%, M 5%.

### Redundancy

- **90 of 126 (71%) were redundant**; 61 of 95 (64%) excluding 5869. The flag fires in the
  seconds between the parent's cancel and the canceller's next writes. A decompose or ruling
  that cancels a parent then works through its children: it co-cancels them (37 F + 2 M were
  "already cancelled" when read; 8200 was cancelled seconds after its flag), re-points them
  (15 P), or names them as superseders (29 S).
- **36 records (29%) were the first look at the child.** Of these, 28 needed a real action:
  fold 12, re-point 8, moot-cancel 4, re-scope 3, done 1. The other 8 were correctly left
  untouched. Two of these cases show the flag catching real gaps: esc-937-1 found that the
  S26 PRD's seam table listed 793 but not its follow-ups 937/960, and esc-5770-3 found that a
  gate would otherwise have released early on a cancelled dep.

## 3. Answers to Leo's questions

**Q1. How often is "leave untouched" correct?** It is the correct final handling for
43 of 126 (34%) as measured, but only 14 of 95 (15%) once the 5869 batch is excluded, and 29
of the 43 are the child-is-the-superseder case, which is detectable mechanically. The more
useful number is different: once the superseding act has *finished*, **no further action
was needed in 71%** of cases (90 of 126), because the canceller had already handled the
child. In the remaining 36, leaving the child untouched was right 8 times (22%).

**Q2. Is there one correct handling?** No. No category reaches a majority (F is 39% overall,
52% excluding 5869), and the dominant category needs a judgment the reconciler cannot make:
whether the superseder's scope covers the child. The escalation-watcher skill's supersession
carve-out ("repoint every dependent onto the superseding task",
`skills/escalation-watcher/SKILL.md` §self-execute) is a single rule too, and it is right for
only 24% of dependents. For 53% the correct answer was to fold the dependent into the
superseder and cancel it, not re-point it.

**Q3. How do the correct handlings break down?** Two facts decide the category:

1. **Why the parent was superseded, and whether the child is named in that act.**
   - Split or replacement, and the child is a replacement → **S**, leave it.
   - Absorbed into a PRD or program leaf whose scope covers the child → **F**.
   - Ruled moot with no successor → **M** if the child's premise was the parent's
     change; **U** if the child stands on main.
2. **What the child stands on.**
   - Files and behaviour already on main; `spawned_from` is only provenance → **U**
     (6906, 7007, 8207).
   - The parent's *delivery*, e.g. "after task N lands" or a gate clock → **P** onto the
     superseder (952, 5770, 7738). This is the one class where silence does harm, because the
     cancel satisfies the dependency and the child is released early.
   - Artifacts only on the parent's unmerged branch → **M**, or **F** if the superseder
     re-creates them (6367, 321, 7997/7998).
   - Partly on main → **R**, trim against main (261, 792).
   - Already on main → **D**.

The **S** class and the **already-dispositioned** cases (child terminal, edge already moved,
child already depending on the superseder) can be decided mechanically from the store. F
versus P versus M needs the superseder's scope, which only the canceller or a human knows.

## 4. Options

| | option | ramification |
|---|---|---|
| **A** | **Supersession-aware, deferred sweep.** (1) Skip any child named in the parent's supersession pointer. (2) Evaluate after the superseding act has had time to finish (a grace window, or the next targeted-recon pass), not at the cancel instant. File only if the child is still live, still linked to the parent, does not already depend on a superseder, and carries no disposition stamp. (3) Emit **one** record per parent listing its residual children, with the classification above as the suggested-action menu, instead of one L1 per child. | Would have suppressed all 29 S records and most of the 61 other redundant ones. The ~36 needing attention remain, with roughly 1 record per parent instead of 126. Needs a deferred re-evaluation hook in recon, plus reading the pointer zoo (x_superseded_by, superseded_by, x_subsumed_by, subsumed_by, x_coalesced_into, x_replacement); a canonical key (heuristic 11/12) should be named. The window delays, but does not worsen, the early-release exposure, since today's L1 holds nothing either. |
| **B** | **Make it the canceller's duty.** A cancel of a parent with live children must carry a per-child disposition (`keep` / `repoint:<id>` / `fold:<id>` / `moot`), or the interceptor warns, refuses, or files the residue at once. Update the watcher carve-out and the /prd decompose to emit it. | Puts the decision where the knowledge is, at write time, with a durable stamp. Changes the cancel contract for every canceller (L2 watcher, /prd, review-all leads, recon itself) and is the largest change. Bare cancels without a supersession still need A's residue path. |
| **C** | **Hold dependents, ignore provenance.** Drop the `spawned_from` arm except for M-detection (declared files absent). For dependents whose parent has a superseder, add a hold so the cancel does not release them until a disposition lands. | Targets the one class where silence does harm (early release) and removes most of the noise; all 31 of the 5869 flags came from the `spawned_from` arm. Loses the first look at the 12 review-followups that needed action (fold 4, re-point 4, re-scope 2, moot 1, done 1; e.g. 6882, 7738, 653, 321), except the moot ones that M-detection keeps. The hold is new scheduler state. |
| **D** | Status quo plus only the S skip. | Cheapest; fixes exactly the 5869 shape (29 of 126). Leaves 61 redundant flags per ~4 weeks across the fleet, about 2 per day, closed as stale by L1 or L2. |

**Recommendation: A**, with B's per-child disposition stamp as the "no residue" signal A
reads, but optional rather than enforced. A removes almost all of the noise using facts the
store already has, keeps the first look that caught 937/960 and 5770, and needs no contract
change for cancellers. Separately, fix the watcher carve-out's blanket "repoint every
dependent" rule, which the data says is right for only a quarter of dependents.
