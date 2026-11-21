# Task 8528 and the un-chained mark: what is wrong, and the ways out

Study for escalation esc-8528-11, 2026-11-18. The study itself was read-only.
A critic seat attacked the first draft; this version carries its corrections (listed at the end).

**Ruled by Leo, 2026-11-18:** the recommendation below, with the depth table's top row at 8 (not
the 6 this report suggests), a clean removal of the suspect search, and tasks 10721 and 10624 left
as they are. The ruling was hand-carried on task 8528's branch, which is the change this file
landed with. The follow-ons listed under "What you need to decide" item 6 were applied the same
day (task 10760 cancelled; tasks 10759, 10761 and 10762 and the merge-chain visibility PRD
re-scoped from "searches" to "isolations").

## The short version

Task 8528 cannot land because one test that task 10723 added to main fails on its branch. The
escalation reads that as "the test's script is out of date and needs rewriting". That is half
right, and the wrong half matters.

1. **The test is right to fail.** On main, every member of a red chain is marked "verify alone
   next", and since 10723 that mark stays until the member lands or is blocked. The 8528 branch
   removes that marking. A red chain's members become "suspects" and are searched by halving. But
   the branch updates the suspect list when a verify *returns*, not when its verdict is *applied*.
   That is the defect 10723 fixed for the mark after task 5715 collected four red verifies without
   ever being blocked. We reproduced it on the branch: a suspect whose solo red was thrown away
   was chained again with two new arrivals.
2. **So rewriting the test alone (option A) or retiring it (option B) lands that defect again**,
   before 10723's fix has run in production for a day.
3. **The larger question is whether 8528's halving search should replace the mark at all.** That
   was decided on 11-05 (decision D6 of the merge-lane policy PRD), twelve days before you carried
   task 10617, which added the mark. The planner of 8528 noticed that the task would undo 10617's
   rule and filed a note about it this morning. The note went to a reviewer, was closed as
   "matches the contract", and never reached you. Our model does not show the search beating the
   mark, in mean landings or in waiting time, outside one unusual corner.
4. **Recommendation.** Keep the mark as the one mechanism that isolates a red chain's members.
   Land 8528's other half, the depth table, and take the suspect search out of the branch. A
   trial on a scratch copy showed the escalated test file passes with a four-line change in that
   direction. If the search is still wanted, bring it back later as its own task, repaired and
   measured on live data.

What only you can decide: D6 is your ruling, and the recommendation amends it, along with parts of
two PRDs and several filed tasks (listed under "What you need to decide").

## Words used here

- **Chain**: the merge lane checks several queued merge requests in one combined tree. The first
  is the owner, the rest are links. One green verify (about 43 minutes at present) lands them all.
  A red one says "at least one of these is bad" and nothing more.
- **The mark** (tasks 10617 and 10723, on main): after a red chain each member is marked, and its
  next verify runs alone, so the verdict is about that member only. The mark is removed when the
  member's request leaves the lane: landed, blocked or dropped.
- **Suspects** (task 8528, on its branch only): after a red chain its members go to the front of
  the queue and are searched by halving. The first half is verified as a chain. If it passes it
  lands, and the culprit is in the other half. If it fails, the culprit is in it. A suspect that
  fails alone is blocked.
- **Thrown-away result**: the lane verifies requests ahead of time, each built on top of the one
  ahead. If the one ahead turns out red, every result behind it is discarded and redone. Such a
  verify returned, but its verdict was never applied.
- **Depth table** (8528): the lane picks the chain depth (8, 4, 2 or 1) from the share of recent
  verifies that passed. Main uses a fixed cap of 6, halved after a red chain.
- **High lane**: a priority queue whose requests are picked before ordinary ones.

## Is the escalation's premise right?

Each claim was tested by running the tests on a scratch copy of the branch (commit be6a4d3fb8)
and by reading the code on main and on the branch.

| Claim in the escalation | Verdict | Evidence |
|---|---|---|
| The test fails on the branch | True | 1 failed, 2 passed. The file is byte-identical to main's. |
| On the branch a red chain marks no one | True | The branch sets the mark in one place (a verify that crashes). Main sets it in two (that, and a red chain). |
| The test's script cannot reach its state | True | Traced: after the red chain of four, the lane verifies the pair (S, F1), then S alone at the front. That red is applied and S is blocked. Nothing is thrown away, so the case the test is about never arises. |
| The steward's replacement trigger hung | True, for a different reason than assumed | The trigger made the local verifier crash, but the test scene sends the chain to the remote host, so the crash never fired. |
| "The property 10723 pins still holds under 8528" | **False for suspects.** True for the crash mark. | Reproduced; see below. |
| Stale wording in the test file | True, and worse | The second test in the file ("a dropped request takes its mark with it") still passes on the branch, but the request in it is never marked there, so it no longer tests what its name says. (Established by tracing the test; no separate mutation run was kept.) |

**The reproduced defect.** This is a test scene: the real lane code with stand-in verifiers. S
owns a red chain over L, so S and L are suspects. A high-lane request X is verified ahead of S
and kept waiting. S verifies alone, built on top of X, and returns red. The branch empties the
suspect list at that moment. X then turns out red and is blocked, so S's result is thrown away
and no verdict is applied to S. S's next verify is a fresh chain of three with two newly arrived
requests. S's verifies covered 2, 1, 3, 2, 1 requests where 2, 1, 1 is correct: two extra
verifies, two innocent requests pulled into a second search, and L released without a verdict.
This was one run. Timing varies between runs, and the defect needs S's solo verify to be built on
top of another request.

Why it happens: the branch updates the suspect list and the pass-rate history in one function,
called at the two places where a verify returns. The code that throws results away runs later and
never touches either. Nothing takes an update back. Three seats read this independently. Two
related effects of the same cause were also reproduced:

- A chain that is red only because the request ahead of it is red still turns its own innocent
  members into suspects, and counts as a failure in the pass-rate history.
- A new red chain replaces the suspect list, so a group already under search is released.

How often: in the model, the loop itself occurs 0.2 to 0.4 times per 72 hours, roughly once every
8 to 18 days. It is a correctness risk more than a throughput cost. 5715 showed what one
occurrence looks like.

The policy PRD is silent on thrown-away results. It says the suspect list is "cleared on
resolution"; the code clears it on return. 8528's own reviewer recorded the pass-rate history
side of this as a known leftover.

## Are the mark and the suspects two tools for one job?

Yes. Both hold the members of a red chain apart until someone is found guilty.

| | The mark (main) | Suspects (8528 branch) |
|---|---|---|
| How members are retried | each alone, at once, on whichever host is free | halving search, one round at a time |
| When a member is released | when its request leaves the lane | when a verify returns (the defect) |
| What other requests do meanwhile | keep chaining among themselves | **nobody chains**: every other request verifies alone while any suspect exists |
| Lost on a restart | yes | yes |
| Where it lives | one set in the lane worker | a small policy module plus a dozen support pieces in the lane worker |

On the branch they do not collide, because the mark is now set only by a crashed verify. They
duplicate one job with two different release rules. Setting both on a red chain does not work
either: a marked request cannot be a chain member, so no search round of two could ever run, and
the "nobody chains" rule would stay on until every member had left.

## The model, and how far to trust it

A seat built a simulator of the lane from the dispatch rules read out of the code: 2 verify hosts,
3 requests in flight, landing strictly in order, results thrown away behind a red, requeues. It
ran about 150,000 simulated 72-hour runs. The critic seat ran its own variants on the same engine.
Inputs measured from the run database:

| Measured on dark-factory | 11-10 → 11-17 20:49Z (before the mark) | 11-17 20:49Z → 11-18 18:34Z (with the mark) |
|---|---|---|
| Share of requests that are red (best estimate) | 0.24 (n=206) | 0.08 (n=12; plausible range 0.05–0.13) |
| Green verify, median | 25 min | 43 min |
| New requests per hour | 2.2 | 2.0 |
| Landings per hour | 1.79 | 2.23 |
| Queue depth, median | 15 | 58 |
| Thrown-away results per completed verify | 0.50 | 0.23 |

Rule for row 1: verifies of one request alone, built on main, failed ÷ all, with re-verifies at
the gate excluded. It includes reds caused by a red main or by infrastructure; the database
cannot separate them. The right-hand window is 22 hours and 47 verifies, so that column is soft.

**Checks.** With chains off the simulator matches the exact formula within 1%. With no red
requests a chaining policy lands the expected multiple.

**Limits that could change a ranking.**

- *Arrival rate.* At the measured 2 new requests per hour, every chaining policy lands the same
  number, because the lane lands what arrives. Differences then show as waiting time. The tables
  below use 4 per hour and a backlog of 60, which is the state the lane will be in when 8528 lands
  and releases the tasks parked behind it. Both views are given.
- *Pick order.* The order in which the lane picks the next request is modelled two ways, "arrival
  order" and "oldest first"; the real lane is in between. Headline figures differ by at most 0.08
  landings per hour between them. But the simulator reproduces the 11-17 collapse only under
  "oldest first", and the tables use "arrival order".
- *The claim behind D6.* 8528's text says the halving search beats the old halving-of-depth rule
  by 14 to 30%. In the simulator that holds under "oldest first" (+12 to +52%) and reverses under
  "arrival order" (−1 to −22%). Either way it was a comparison with the old rule, never with the
  mark, which did not exist yet.
- *Main moving.* "Main moved, so a passed request re-verifies" is modelled coarsely and costs
  every policy 20 to 35%.
- One simplification (where a discarded request rejoins the queue) may slightly handicap the
  mark-based policies under "arrival order". Not quantified.
- No host speed difference, no flaky reds, no merge conflicts.
- The simulator is not fitted to the live event stream. Treat differences under about 5% as ties.

## The options

Twelve options were collected: the three on the record, seven from an option-space seat and two
from the critic. Five are set aside with the reason; seven are scored.

**Set aside**

- **A: rewrite the test inside 8528, no code change.** A rewrite built on the crash mark would
  pass honestly, but it would leave the suspect search with no test of the property and the
  defect in place.
- **B: retire the test, keep a small property test.** The small test cannot be written where
  proposed. The pure policy functions have no notion of "returned" against "applied"; the defect
  is in when the lane worker calls them.
- **C: a separate re-authoring task that 8528 depends on.** The test needs 8528's code, so the new
  task could only run on 8528's own branch. It adds waiting and no information.
- **H: land 8528 with the test switched off and fix later.** Ships the reproduced defect.
- **J: revert 10723 on main.** Removes a fix that is correct and is not the problem.

**Scored**

- **M. Leave main as it is; 8528 stays stuck.** The mark, the old halving of depth, cap of 6.
- **K. Cancel 8528.** Same lane behaviour as M, with the parked tasks released. Gives up the depth
  table and the new "why this depth" field on verify records.
- **P. The branch as it is, forced through.** For comparison only.
- **Q1. Keep the search, repair its release rule inside 8528.** Members stay held until their
  request leaves the lane. A new red chain adds to the held set. The history counts applied
  verdicts only. The "nobody chains during a search" rule stays.
- **Q2. Q1, and requests that are not held keep chaining.**
- **R. Keep both: a red chain also sets the mark.** A few lines.
- **I. Depth table yes, suspect search no.** A red chain marks its members exactly as on main.
  The search, its queue-fronting and its tests come off the branch. **I-6** is the same with the
  table's top row at depth 6 in place of 8. A stopgap form, **I-switch**, makes the four-line
  change and leaves the search code in place but switched off, to be removed or repaired later.

### What we measured about I on the branch

On a scratch copy of the branch, two edits were made (four lines): a red chain marks its owner
and links again, and a red chain no longer creates suspects. Result:

- The escalated test file: **3 passed**.
- The branch's other chain tests (six files, 288 tests): 259 passed, **29 failed**. All 29 are
  tests of the search itself, or assert that a red chain is followed by a search.

So I's floor is four production lines plus 29 tests to rewrite or remove. Removing the search
cleanly is more than that: its support code is spread over about 14 source and test files (the
critic counted a dozen pieces in the lane worker, two values in the recorded "why this depth"
field, and five recorded size-limit raises). Q1 and Q2 were not sized by experiment. They need a
new release rule designed, built and tested, and they keep all of that support code.

### Landings per hour

Mean, with the 5th percentile across 200 runs in brackets. Settings: 4 new requests per hour,
starting backlog 60, current verify times, arrival-order picking. "All effects on" adds: main
moving every 2 hours, a restart every 12 hours, 3% of chain verifies crashing, and 30% of
arrivals in the high lane.

| Red share | M / K | P | Q1 | Q2 | R | I | I-6 | chains off |
|---|---|---|---|---|---|---|---|---|
| **Plain** 0.05 | 4.42 [3.85] | 4.32 [3.58] | 4.21 [3.43] | 4.34 [3.62] | 4.20 [3.37] | 4.37 [3.78] | 4.43 [3.94] | 2.49 |
| 0.08 (now) | 3.75 [2.86] | 3.50 [2.68] | 3.42 [2.69] | 3.58 [2.77] | 3.44 [2.75] | 3.74 [3.10] | 3.82 [3.04] | 2.39 |
| 0.14 | 2.65 [2.11] | 2.45 [2.01] | 2.48 [2.04] | 2.52 [2.06] | 2.52 [2.08] | 2.73 [2.21] | 2.72 [2.21] | 2.19 |
| 0.24 (before the mark) | 1.78 [1.50] | 1.82 [1.60] | 1.82 [1.58] | 1.81 [1.60] | 1.82 [1.61] | 1.89 [1.58] | 1.88 [1.60] | 1.85 |
| **All effects on** 0.05 | 3.37 [2.47] | 2.91 [2.05] | 2.69 [2.00] | 3.27 [2.33] | 2.64 | 3.30 [2.46] | 3.28 [2.40] | 1.70 |
| 0.08 | 2.56 [1.86] | 2.29 [1.71] | 2.17 [1.67] | 2.51 [1.89] | 2.19 | 2.56 [1.87] | 2.57 [1.90] | 1.64 |
| 0.14 | 1.87 [1.39] | 1.73 [1.39] | 1.70 [1.36] | 1.80 [1.39] | 1.72 | 1.85 [1.42] | 1.83 [1.42] | 1.53 |
| 0.24 | 1.29 [1.01] | 1.32 [1.10] | 1.31 [1.10] | 1.32 [1.10] | 1.32 | 1.34 [1.10] | 1.33 [1.08] | 1.34 |

Reading it:

- The branch as written (P) lands 2 to 8% less than main at red shares 0.05 to 0.14, and 7 to 14%
  less with all effects on. At 0.24 everything ties, and chains barely pay at all.
- Q1 is no better than P. Q2 recovers most of the gap and is level with I once all effects are on.
- I, I-6 and M are level within the model's noise.
- With task 10721 landed as well, figures move by at most 0.10. I is then 0.05 to 0.10 below M in
  three of the four cells, and Q2 loses 0.08 to 0.10.
- **At the measured 2 arrivals per hour** all chaining policies land the same in the plain cells
  (2.62–2.63 at 0.05; 2.53–2.55 at 0.08). With all effects on, P is 2 to 5% below M.

### Waiting time and behaviour under stress

| | M / K | P | Q1 | Q2 | R | I | I-6 |
|---|---|---|---|---|---|---|---|
| Median wait of a good request, hours (plain, red share 0.05, 4 arrivals/h) | **5.6** | 7.2 | 9.2 | 7.3 | 9.6 | 6.8 | 5.8 |
| Most red verifies one culprit collected before its block, plain | 5 | 8 | 5 | 6 | 4 | 4 | 4 |
| The same, all effects on | 10–11 | 10–11 | 10–11 | 10 | not extracted | 9–11 | 9–11 |
| Culprit put in a fresh chain again before any verdict, per run, plain | 0 | 1.3–2.7 | 0 | 0 | 0 | 0 | 0 |
| The same, all effects on | 6–12 | 3–5 | 1–3 | 4–6 | not extracted | 5–8 | 5–7 |
| 11-17-style start, bad requests at the front: landings in 6.5 h, empty history | 26–29 | 7–10 | 10 | 10 | 14 | 17 | 18–19 |
| The same, history pre-filled as the real code does (critic's runs, n=100, unreviewed) | 26–29 | **2–20** | not run | 14–19 | not run | 21–25 | 23–28 |

Rules and caveats for this table:

- Row 4 for P includes the search's by-design release of the "other half" after a red round, not
  only the defect.
- With all effects on, re-chaining is dominated by restarts forgetting the marks or suspects.
  There main is the worst and the search variants are better, because main's old halving rule
  lets the depth bounce back up.
- The incident rows start from 34 queued requests of which 3 are red, with 2 arrivals per hour.
  The range covers both pick orders. P's 2 is under "oldest first", where culprits also collect up
  to 14 red verifies. By 30 hours every policy has landed about the same number (roughly 85).

Across the full grid (108 combinations of red share 0.02 to 0.35, arrivals of 2, 4 or 8 per hour,
backlog 0 or 60, and plain or all effects on; arrival-order picking; 40 runs each for most):

| | M | P | Q1 | R | I |
|---|---|---|---|---|---|
| Share of combinations more than 10% below the best policy | 16% | 26% | 31% | 29% | 2% |
| Worst combination, as a share of the best policy there | 0.83 | 0.64 | 0.61 | 0.59 | 0.89 |

Two warnings on this table. All of M's weak combinations are at a red share of 0.35, far above
anything measured, where switching chains off is best; so I's edge over M here says little about
real conditions. And Q2 was never run on the grid. The search variants' worst combination is very
few reds (0.02) with heavy arrivals (8 per hour) and all effects on: P lands 3.65 there, against
M 5.02, I 5.41 and Q2 5.65. That is the one place a repaired search comes out ahead, by about 4%
over I, within one to two standard errors.

Why the search trails, from the follow-up runs:

- The "nobody chains while a search is on" rule is the main cost. Lifting it (Q2) closes half to
  nine-tenths of the gap at red shares 0.05 to 0.08.
- The remainder is the search itself: its rounds run one after another, while the mark sends
  every member to verify alone at once.
- Lifting that rule on the branch without repairing the release rule makes things worse: culprits
  are re-chained 3 to 16 times per run, and one collected 21 red verifies.

### Architecture

Judged against `docs/code-quality.md`, by heuristic name.

| | P (branch as is) | Q1 / Q2 | R | I | I-switch |
|---|---|---|---|---|---|
| **11 SPOT** | One fact ("not attributed yet, do not chain freely") kept in two places with two release rules | Two places, rules now agree | Two places written together | **One place, one rule** | One live, one dormant |
| **3 orthogonal dimensions** | Cause (red, crash), release rule and retry style fused into two bundles | Release unified; cause and style still fused | Fused and self-defeating | Only the cause varies; depth choice is separate | As I while off |
| **10 invariants** | "No release without an applied verdict" holds for the mark only | Holds for both, by two mechanisms | Holds via the mark; suspects are noise | Holds at one point that main already tests | As I; the dormant path keeps the defect |
| **9 deep modules** | Policy module is deep, but the lane worker drives fronting, pruning and a parallel mark | Same, with more calls | Same, with dead search code | Policy module shrinks to depth choice | Dead code kept |

A further end state, "one held set owned by the policy module, fed by a red chain and by a crash,
released only on an applied verdict", is the cleanest on paper on every row. It is unbuilt. In
the critic's runs it ties or slightly beats I at a red share of 0.05, trails at 0.14, and does
nothing for an incident start. It is the right shape if the search ever comes back, not something
to design inside a stuck task.

The merge-lane quality PRD's module map predates the policy module. It names no owner for the
mark, the suspect list, or the point where a verdict updates the policy. Task 5040 moves the
function that contains all of those calls out of the lane worker, and 5041 moves chain building.
Under I, 5040 carries one call (set the mark) and 5041 receives depth choice as a ready pure
module. Under P or Q, 5040 carries three calls into state that 5041 will own, plus the dozen
support pieces. These counts are from reading; 5040's plan predates both 10617 and 8528 and needs
redoing either way.

### Fit with the work around 8528

| Task | Written for | Under I | Under Q |
|---|---|---|---|
| 10721 re-verify a passed request alone (high; **does not wait on 8528**) | the mark | fits as written | fits; wording goes stale |
| 10624 mark links earlier; coalescing honours the mark (high; **does not wait on 8528**) | the mark | fits as written | two of its three items need re-scoping for suspects |
| 6267 journal keeps cancelled requests | either | unaffected | unaffected |
| 5621 conflict probe on the look-ahead pick | the mark | fits | the search's queue-fronting interacts with the pick it changes; unexamined |
| 5040, 5041 code extraction | the old halving rule | smaller carry | larger carry |
| 10747 gate for the 5040 carry; 10758 "Verifying now" panel | either | unaffected | unaffected |
| 10759 record chain rounds and culprit searches | **suspects** | **re-scope**: a "search" becomes "the members marked by one red chain" | fits; some rows move if the release rule moves |
| 10760 count bisect rounds apart | **suspects** | **cancel**: there are no bisect rounds; 10761 depends on it | fits |
| 10761, 10762 snapshot and panel | both | simpler | fits |
| 8532 canary readout; 7860 human gate (8528's named consumers) | 8528's stated signal | signal changes (below) | unaffected |

10721 and 10624 depend only on 10723, which is done, so they can be dispatched now and will edit
the same lines the 8528 branch edits. Each landing is another collision of the kind that produced
this escalation, whichever option is chosen.

## Recommendation

**Choose I, with the table's top row at 6 (I-6): keep the mark as the single owner of isolation,
land the depth table, take the suspect search out of the branch.**

The evidence chain:

1. The defect is real, by a run on the branch. No option that leaves the branch's code alone is
   acceptable (A, B, C, H, P).
2. The search is not shown to pay. At the measured arrival rate it lands the same as the mark. In
   a backlog it lands the same or less, and good requests wait longer (7.2 to 9.2 hours against
   5.6 to 6.8). Its one win is about 4% in a corner with almost no reds and heavy arrivals, and
   only after being repaired and loosened (Q2).
3. The mark's release rule is already correct, on main, and tested. Q rebuilds that rule for a
   second mechanism, and is the only route with new design work in it.
4. I is the measured small step: four lines turn the escalated file green, and the 29 tests that
   then fail are the search's own.
5. It leaves one mechanism where the other options leave two, and it is the world that 10721,
   10624 and 5621 were written for.
6. Depth 6 at the top of the table costs nothing where reds are 0.08 or fewer, matches main's
   waiting time (5.8 against 6.8 hours), and gives the best incident start of the table options.
   Depth 8 pays only in the low-red, heavy-arrival corner (5.41 against 5.04).

What the recommendation does not claim:

- That I beats main. I, I-6 and M are level in the model. The case for landing 8528 at all, over
  cancelling it (K), is the depth table at high red shares (1.89 against 1.78 at 0.24), the
  recorded "why this depth" on every verify, and not discarding the branch's other work. If those
  are not worth a rework, K is a legitimate choice and the cheapest.
- That a repaired search is worse than I on throughput. Q2 is level with I when all effects are
  on. The case against Q2 is points 3 to 5 and the weaker incident start, not landings.
- That I is small. Its floor is measured; a clean removal across about 14 files is not sized.
  If speed matters most, I-switch lands the four lines and the 29 test changes now and removes
  the search afterwards, at the price of dormant code that still carries the defect.

Not verified: the wider test suite on the branch; the model's absolute numbers; the critic's
pre-filled-history runs; 5715's own event sequence against the explanation given here.

## What you need to decide

1. **Does the suspect search replace the mark (D6 as ruled), or does the mark stay?**
   - Mark stays (I, recommended): amend D6.
   - Search stays (Q): choose Q1 or Q2. Q2 also changes D6's sentence that "any other dispatch
     verifies alone".
   - Neither is worth it: cancel 8528 (K).
2. **If I: top row of the depth table at 6 or 8?** Recommended 6.
3. **If I: clean removal now, or I-switch (search switched off, removed later)?**
4. **Who does the work on 8528?** The current problem has been filed four times up the ladder
   (esc-8528-8 to -11) without progress. A hand carry on the existing branch in the high lane is
   the suggested route; redispatching the implementer with a precise note is the alternative.
   196 open tasks depend directly on 8528 and 644 through other tasks (the critic's count from
   the task store today; the "139" in the brief dates from 11-10).
5. **Should 10721 and 10624 wait until 8528 has landed?** Both are free to dispatch now and edit
   the same lines.
6. **If I, these follow:**
   - Merge-lane policy PRD: D6, D7's wording, the contract section on depth and suspects, boundary
     row 7 (the bisect row), the task's stated signal, and the capability manifest.
   - Task 8528's own text and its stated signal ("a tip failure followed by a half-chain verify").
     A reviewer would reject I against the unamended text.
   - Merge-chain visibility PRD (ruled today): the decisions and rows about culprit searches;
     task 10759 re-scoped, 10760 cancelled, 10761's dependency on it removed.
   - The canary readout (8532) and human gate 7860, which name 8528's signal.

Two things noticed on the way that are not part of this decision:

- With 10721 not landed, the model has 10723 costing main 8 to 10% of landings. Good members whose
  solo verify was cancelled by a "main moved" re-verify stay marked and verify alone again.
  Landing 10721 removes the cost, which raises its urgency once 10723 is live.
- A restart forgets the marks. With all effects on, that lets a culprit be chained again 6 to 12
  times per 72-hour run on main. Task 6267 touches the same restart path.

## How this was produced

| Seat | Model | Effort | Tokens | Minutes | Outcome |
|---|---|---|---|---|---|
| Option space | opus | medium | 95k | 1 | 10 options, 8 discriminating questions |
| Premise | opus | high | 207k | 27 | Ran the tests; reproduced the defect |
| Mechanism | fable | high | 335k | 17 | State tables for both mechanisms; dispatch rules for the simulator |
| Upcoming work | sonnet | medium | 169k | 4 | 13 tasks and 3 PRDs read |
| Data | sonnet | medium | 153k | 5 | Rates from the run database, both windows |
| Model | opus | high | 357k | 42 | Simulator, about 150k runs, follow-up variants |
| Critic | fable | high | 379k | 13 | 14 must-fix findings on the draft, own model variants |
| Lead | fable | | | | Scouting, spot checks, the four-line trial, this report |

What the critic changed: the incident comparison (the first draft blamed depth 8; the real cause
was an empty pass-rate history the live code never has); the robustness claim (it excluded Q2 and
leaned on a red share of 0.35); the "level with main" claim (it hid a waiting-time cost at depth
8); the size comparison between I and Q (asserted, not measured, which prompted the four-line
trial); the arrival-rate caveat; the count of parked tasks; the list of what I amends; options K
and I-switch; and the note that never reached you.

Evidence files, all volatile under /tmp/8528-study/: `premise/` (probe tests and logs), `data/`,
`model/` (engine, results, FINDINGS.md), `crux/RULES.md`, `critic/`.

Dropped or degraded:

- Sub-seats could not save their own findings files; the mechanism, upcoming-work and critic
  findings exist only in the session transcript and in this report.
- The defect scene was run once.
- Q1 and Q2 were not sized on the branch; R and Q1 were not run in the pre-filled incident start.
- Whether a repo guard rejects a switched-off test (option H only) was not checked.
